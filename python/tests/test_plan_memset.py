"""A memset step in a launch plan: ``tilefish.zeros_like(x)`` in a graph body.

A memset step is ``{"kind": "memset", "args": [{"name": "dst", ...}]}``: no entry, no workspace, the dense bytes of ``dst`` (a whole buffer or a
dense view) set to zero. Both readers hold it to that (``validate_doc`` here, ``launch_plan.rs`` in Rust, the same documents through
``python_plans.json``); a host launches it as ``cuMemsetD8Async`` on the plan's stream, which a CUDA graph captures as a memset node. A plan without a
memset step is what it was (the step is additive: no schema change). Nothing here needs a GPU (``test_gpu_plan_memset.py``).
"""

import json

import gpu_kernels_plans as k
import plan_fakes
import pytest
from fake_torch import FakeTensor
from test_planrun import ARENA, FakeBackend, FakeProvider

import tilefish
from tilefish import Dim, InOut, Tensor, TilefishManifestError, TilefishTypeError, _plan, bf16, f32, i32
from tilefish._planrun import PlanRunner

Rb = Dim("Rb", buckets=[8, 24])


class Backend(FakeBackend):
    def memset(self, dst, nbytes, stream):
        self.log.append(("memset", dst, nbytes, stream))

    def current_device(self):
        return 0

    def launches(self):
        return [e for e in self.log if e[0] in ("launch", "gemm", "copy", "memset")]


@tilefish.kernel
def scale(x: Tensor[Rb, 64, f32]) -> Tensor[Rb, 64, f32]:
    return x * 2.0


@tilefish.kernel
def add_to(acc: InOut[Tensor[Rb, 64, f32]], x: Tensor[Rb, 64, f32]):
    acc[...] = acc + x


@tilefish.graph
def zeroed_scratch(x: Tensor[Rb, 64, f32]):
    acc = tilefish.zeros_like(x)  # an intermediate of the arena
    add_to(acc, x)
    return scale(acc.live)


def test_zeros_like_is_a_memset_step_into_a_buffer_of_its_own():
    p = tilefish.plan(zeroed_scratch)
    ss = p.steps(Rb=8)
    assert [s["kind"] for s in ss] == ["memset", "kernel", "kernel"]
    assert ss[0] == {"kind": "memset", "args": [{"name": "dst", "binding": "t0"}]}
    assert {b["name"]: b["role"] for b in p.buffers}["t0"] == "intermediate"
    assert "#0 memset (memset): t0" in p.describe()


def test_the_memsets_of_zero_and_add_write_the_callers_output_and_inout_tensor_directly():
    p = tilefish.plan(k.zero_and_add)
    ss = p.steps(Rs=24)
    assert [(s["kind"], s["args"][0]["binding"]) for s in ss] == [
        ("memset", "acc"),  # the returned accumulator: the caller's output buffer itself
        ("kernel", "acc"),
        ("kernel", "acc"),
        ("memset", "state"),  # state[...] = zeros_like(state): one memset of the caller's tensor, no intermediate and no copy
        ("kernel", "state"),
        ("memset", "zeros"),  # an i32 tensor of zeros returned
    ]
    assert not [b for b in p.buffers if b["role"] == "intermediate"] and p.arena_bytes == 0
    assert {b["name"]: b["dtype"] for b in p.buffers}["zeros"] == "i32"


def test_a_zeros_like_stored_into_a_parameter_and_read_again_stays_an_intermediate_and_a_copy():
    @tilefish.graph
    def reread(x: Tensor[Rb, 64, f32], state: InOut[Tensor[Rb, 64, f32]]):
        z = tilefish.zeros_like(x)
        state[...] = z
        return scale(z)  # z is read after the store: the memset cannot write the caller's tensor instead

    ss = tilefish.plan(reread).steps(Rb=8)
    assert [s["kind"] for s in ss] == ["memset", "copy", "kernel"]
    assert ss[0]["args"][0]["binding"] == "t0" and ss[1]["args"] == [{"name": "src", "binding": "t0"}, {"name": "dst", "binding": "state"}]


def test_zeros_like_is_an_op_of_a_graph_body_and_takes_a_dtype_with_a_storage_type():
    with pytest.raises(TilefishTypeError, match="zeros_like is an op of a graph body"):

        @tilefish.kernel
        def k_(x: Tensor[Rb, 64, f32]) -> Tensor[Rb, 64, f32]:
            return tilefish.zeros_like(x)

    @tilefish.graph
    def typed(x: Tensor[Rb, 64, f32]):
        return tilefish.zeros_like(x, dtype=bf16), tilefish.zeros_like(x, dtype=i32)

    assert [b["dtype"] for b in tilefish.plan(typed).buffers if b["role"] == "output"] == ["bf16", "i32"]
    with pytest.raises(TilefishTypeError, match="zeros_like: the dtype"):

        @tilefish.graph
        def untyped(x: Tensor[Rb, 64, f32]):
            return tilefish.zeros_like(x, dtype="f32")

    with pytest.raises(TilefishTypeError, match="zeros_like: the dtype"):

        @tilefish.graph
        def mask(x: Tensor[Rb, 64, f32]):
            return tilefish.zeros_like(x, dtype=tilefish.bool_)


def test_the_reference_mode_gives_zeros():
    torch = pytest.importorskip("torch")
    x = torch.randn(8, 64)
    state = torch.full((8, 64), 7.0)
    acc, zeros = tilefish.run(k.zero_and_add, x, state, mode="reference")
    assert torch.equal(acc, x + x) and torch.equal(state, x) and torch.equal(zeros, torch.zeros(8, 64, dtype=torch.int32))


def test_the_runner_launches_a_memset_with_the_address_and_the_bytes_of_its_buffer(tmp_path):
    p = tilefish.plan(zeroed_scratch)
    exported = plan_fakes.export(p, tmp_path / "zeroed")
    assert exported.digest == p.with_manifest("sm_103", exported.manifest_digest).digest
    be = Backend()
    x = FakeTensor((24, 64), "f32", 1 << 36)
    PlanRunner(p, FakeProvider(), be).run((x,), {})
    log = be.launches()
    assert [e[0] for e in log] == ["memset", "launch", "launch"]
    off = {b["name"]: b["offset"] for b in p.buffers}
    assert log[0][1:] == (ARENA + off["t0"], 24 * 64 * 4, 7)  # dst, bytes, stream


def test_the_runner_zeroes_the_callers_tensors():
    p = tilefish.plan(k.zero_and_add)
    be = Backend()
    x, state = FakeTensor((8, 64), "f32", 1 << 36), FakeTensor((8, 64), "f32", 2 << 36)
    acc, zeros = PlanRunner(p, FakeProvider(), be).run((x, state), {})
    memsets = [e for e in be.launches() if e[0] == "memset"]
    assert [m[1:3] for m in memsets] == [(acc._ptr, 8 * 64 * 4), (state._ptr, 8 * 64 * 4), (zeros._ptr, 8 * 64 * 4)]


def test_a_tap_after_a_memset_and_a_step_alone_of_one():
    p = tilefish.plan(zeroed_scratch)
    be = Backend()
    x = FakeTensor((8, 64), "f32", 1 << 36)
    tap = FakeTensor((8, 64), "f32", 3 << 36)
    PlanRunner(p, FakeProvider(), be).run((x,), {}, taps={("t0", 0): tap})
    log = be.launches()
    assert log[0][0] == "memset" and log[1][0] == "copy" and log[1][1] == tap._ptr  # the tap copies the zeroed buffer right after the memset
    be2 = Backend()
    t0 = FakeTensor((8, 64), "f32", 4 << 36)
    PlanRunner(p, FakeProvider(), be2).run_step(0, {"t0": t0})
    assert [e for e in be2.launches()] == [("memset", t0._ptr, 8 * 64 * 4, 7)]


# -- the rules of a memset step in a document (validate_doc; the same documents go to the Rust reader through python_plans.json) ------------


def doc_of(p):
    return json.loads(json.dumps(p._doc))


def every_bucket(doc, step, change):
    for b in doc["buckets"]:
        change(b["steps"][step])


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda st: st.__setitem__("entry", "memset"), "must have exactly the fields"),
        (lambda st: st.__setitem__("workspace", "workspace"), "must have exactly the fields"),
        (lambda st: st["args"][0].__setitem__("name", "src"), r"has the one argument \['dst'\]"),
        (lambda st: st["args"].append({"name": "src", "binding": "x"}), r"has the one argument \['dst'\]"),
        (lambda st: st["args"][0].__setitem__("view", {"shape": [4, 128], "strides": [1, 4], "offset": 0}), "which is not dense"),
        (lambda st: st["args"][0].__setitem__("binding", "x"), "a memset step writes 'x', whose role is input"),
    ],
)
def test_a_malformed_memset_step_is_refused(change, message):
    doc = doc_of(tilefish.plan(zeroed_scratch))
    every_bucket(doc, 0, change)
    with pytest.raises(TilefishManifestError, match=message):
        _plan.validate_doc(doc, made=True)


def test_a_memset_of_a_dense_view_is_accepted_and_makes_its_buffer_written(tmp_path):
    p = tilefish.plan(zeroed_scratch)
    doc = doc_of(p)
    for b in doc["buckets"]:
        rows = b["values"]["Rb"]
        b["steps"][0]["args"][0]["view"] = {"shape": [rows // 2, 64], "strides": [64, 1], "offset": 64}
    _plan.validate_doc(doc, made=True)
    exported = plan_fakes.export(p, tmp_path / "zeroed")
    doc = exported.doc
    for b in doc["buckets"]:
        b["steps"][0]["args"][0]["binding"] = "out0"  # the memset now writes the result first: t0 is read before it is written
    with pytest.raises(TilefishManifestError, match="reads 't0' before any step has written it"):
        _plan.check_against_manifest(doc, exported.manifest)


def test_a_memset_in_one_bucket_only_is_refused():
    doc = doc_of(tilefish.plan(zeroed_scratch))
    doc["buckets"][1]["steps"].insert(1, {"kind": "memset", "args": [{"name": "dst", "binding": "t0"}]})
    with pytest.raises(TilefishManifestError, match="every bucket must have the same number of steps"):
        _plan.validate_doc(doc, made=True)
