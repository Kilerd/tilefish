"""A copy step in a launch plan: ``tilefish.copy(x)`` and ``param[...] = x`` into an ``InOut`` parameter.

A copy step is ``{"kind": "copy", "args": [{"name": "src", ...}, {"name": "dst", ...}]}``: no entry, no workspace, the dense bytes of ``src`` (a
whole buffer or a dense view: one run of bytes) into ``dst``, of one dtype and one number of elements, not overlapping. Both readers hold it to that
(``validate_doc`` here, ``launch_plan.rs`` in Rust, the same documents through ``python_plans.json``); a host launches it as
``cuMemcpyDtoDAsync`` on the plan's stream, which a CUDA graph captures as a memcpy node. Nothing here needs a GPU (``test_gpu_plan_copy.py``).
"""

import json

import plan_fakes
import pytest
from fake_torch import FakeTensor
from test_planrun import ARENA, FakeBackend, FakeProvider

import tilefish
from tilefish import Dim, InOut, Tensor, TilefishLoweringError, TilefishManifestError, TilefishPlanError, _plan, f32
from tilefish._planrun import PlanRunner

Rb = Dim("Rb", buckets=[8, 24])


@tilefish.kernel
def twice(x: Tensor[Rb, 64, f32]) -> Tensor[Rb, 64, f32]:
    return x * 2.0


@tilefish.kernel
def bump(state: InOut[Tensor[Rb, 64, f32]], y: Tensor[Rb, 64, f32]):
    state[...] = state + y


@tilefish.graph
def copied(x: Tensor[Rb, 64, f32], state: InOut[Tensor[Rb, 64, f32]]):
    r = tilefish.copy(x)  # a copy step: the caller's x into an intermediate, which an in-place call may then update
    bump(r, x)
    r = r.live  # the updated copy
    state[...] = r  # a store into an InOut parameter: a copy step into the caller's tensor
    return twice(r)


@tilefish.graph
def copied_view(x: Tensor[2, Rb, 64, f32]):
    first = tilefish.copy(x[1])  # a dense view (offset Rb * 64): one run of bytes
    return twice(first)


def steps(p, rb=8):
    return p.steps(Rb=rb)


def test_a_copy_and_a_store_are_copy_steps():
    p = tilefish.plan(copied)
    ss = steps(p)
    assert [s["kind"] for s in ss] == ["copy", "kernel", "copy", "kernel"]
    first, store = ss[0], ss[2]
    assert set(first) == {"kind", "args"} and [a["name"] for a in first["args"]] == ["src", "dst"]
    assert first["args"][0] == {"name": "src", "binding": "x"} and first["args"][1]["binding"] == "t0"
    assert ss[1]["args"][0]["binding"] == "t0"  # bump updates the copy in place
    assert store["args"] == [{"name": "src", "binding": "t0"}, {"name": "dst", "binding": "state"}]
    roles = {b["name"]: b["role"] for b in p.buffers}
    assert roles["t0"] == "intermediate" and roles["state"] == "inout"
    assert "copy" in p.describe()


def test_a_copy_of_a_dense_view_has_its_offset_and_the_store_of_a_value_onto_itself_is_no_step():
    p = tilefish.plan(copied_view)
    src = steps(p, 24)[0]["args"][0]
    assert src == {"name": "src", "binding": "x", "view": {"shape": [24, 64], "strides": [64, 1], "offset": 24 * 64}}

    @tilefish.graph
    def onto_itself(x: Tensor[Rb, 64, f32], state: InOut[Tensor[Rb, 64, f32]]):
        state[...] = state
        return twice(x)

    assert [s["kind"] for s in steps(tilefish.plan(onto_itself))] == ["kernel"]


def test_a_copy_of_a_view_that_is_not_dense_is_refused():
    @tilefish.graph
    def columns(x: Tensor[Rb, 128, f32]):
        a, b = x.split(64, axis=1)  # a column block: rows of 64 with a stride of 128
        return twice(tilefish.copy(a))

    with pytest.raises(TilefishPlanError, match="which is not dense: a copy step copies one run of bytes"):
        tilefish.plan(columns)


def test_a_copy_in_a_kernel_body_is_not_lowered():
    @tilefish.kernel
    def k(x: Tensor[Rb, 64, f32]) -> Tensor[Rb, 64, f32]:
        return tilefish.copy(x) * 2.0

    with pytest.raises(TilefishLoweringError, match="a copy step of a graph's plan"):
        tilefish.lower(k)


def test_the_plan_exports_and_loads_and_the_runner_copies_with_the_addresses_of_the_buffers(tmp_path):
    p = tilefish.plan(copied)
    exported = plan_fakes.export(p, tmp_path / "copied")
    assert exported.digest == p.with_manifest("sm_103", exported.manifest_digest).digest
    be = FakeBackend()
    runner = PlanRunner(p, FakeProvider(), be)
    x, state = FakeTensor((8, 64), "f32", 1 << 36), FakeTensor((8, 64), "f32", 2 << 36)
    out = runner.run((x, state), {})
    off = {b["name"]: b["offset"] for b in p.buffers}
    log = be.launches()
    assert [e[0] for e in log] == ["copy", "launch", "copy", "launch"]
    assert log[0][1:] == (ARENA + off["t0"], x._ptr, 8 * 64 * 4, 7)  # dst, src, bytes, stream
    assert log[2][1:] == (state._ptr, ARENA + off["t0"], 8 * 64 * 4, 7)
    assert log[3][3][-2] == ("pointer", out._ptr) or ("pointer", out._ptr) in log[3][3]


def test_the_runner_copies_a_dense_view_from_its_offset():
    p = tilefish.plan(copied_view)
    be = FakeBackend()
    x = FakeTensor((2, 24, 64), "f32", 1 << 36)
    PlanRunner(p, FakeProvider(), be).run((x,), {})
    copy = be.launches()[0]
    assert copy[0] == "copy" and copy[2] == x._ptr + 24 * 64 * 4 and copy[3] == 24 * 64 * 4


# -- the rules of a copy step in a document (validate_doc; the same documents go to the Rust reader through python_plans.json) ---------------


def doc_of(p):
    return json.loads(json.dumps(p._doc))


def refused(doc, message):
    with pytest.raises(TilefishManifestError, match=message):
        _plan.validate_doc(doc, made=True)


def every_bucket(doc, step, change):
    for b in doc["buckets"]:
        change(b["steps"][step])


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda st: st.__setitem__("entry", "copy"), "must have exactly the fields"),
        (lambda st: st.__setitem__("workspace", "workspace"), "must have exactly the fields"),
        (lambda st: st["args"].reverse(), r"has the arguments \['src', 'dst'\]"),
        (lambda st: st["args"].pop(), r"has the arguments \['src', 'dst'\]"),
        (lambda st: st["args"][0].__setitem__("view", {"shape": [4, 64], "strides": [64, 1], "offset": 0}), "the number of elements must be the same"),
        (lambda st: st["args"][0].__setitem__("view", {"shape": [4, 128], "strides": [1, 4], "offset": 0}), "which is not dense"),
    ],
)
def test_a_malformed_copy_step_is_refused(change, message):
    doc = doc_of(tilefish.plan(copied))
    every_bucket(doc, 0, change)
    refused(doc, message)


def test_a_copy_within_one_buffer_must_not_overlap_and_dtypes_must_agree():
    doc = doc_of(tilefish.plan(copied_view))
    for b in doc["buckets"]:
        rows = b["values"]["Rb"]
        b["steps"][0]["args"][1] = {"name": "dst", "binding": "x", "view": {"shape": [rows, 64], "strides": [64, 1], "offset": 32}}
    refused(doc, "whose role is input|overlap")
    doc = doc_of(tilefish.plan(copied))
    for b in doc["buckets"]:
        b["steps"][2]["args"][1]["binding"] = "out0"  # an f32 buffer of another shape: the counts agree, the role is writable
    _plan.validate_doc(doc, made=True)


def test_a_copy_writes_only_a_buffer_a_step_may_write():
    doc = doc_of(tilefish.plan(copied))
    every_bucket(doc, 2, lambda st: st["args"][1].__setitem__("binding", "x"))  # the store's destination made the caller's read-only x
    refused(doc, "a copy step writes 'x', whose role is input")


def test_a_copy_reads_only_what_was_written_before(tmp_path):
    p = tilefish.plan(copied)
    exported = plan_fakes.export(p, tmp_path / "copied")
    doc = exported.doc
    for b in doc["buckets"]:
        b["steps"][0]["args"][0]["binding"] = "out0"  # the copy now reads the result before the last step writes it
    with pytest.raises(TilefishManifestError, match="argument 'src' reads 'out0' before any step has written it"):
        _plan.check_against_manifest(doc, exported.manifest)
