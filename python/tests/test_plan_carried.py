"""Carried buffers (``@tilefish.graph(carried=("h",))``, ``docs/en/plan-schema-1.md`` section 15.2), with no GPU: plans and runners on a fake backend.

A carried parameter is a buffer of role ``carried`` (offset ``null``: the host places it): the arena holds it in a region of its own after the plans'
regions, the same for every plan of the arena that names it, so one plan hands an intermediate to the next without a tensor of the caller's. A run takes
the other parameters; one that reads a carried buffer before writing it is refused unless the executions issued before it on the arena left it in the
shape it reads, and its dimensions are bound from that shape. The real launches are ``test_gpu_plan_carried.py``.
"""

import json

import gpu_kernels_plans as k
import plan_fakes
import pytest
from fake_torch import FakeTensor
from test_planrun import ARENA, FakeBackend, FakeProvider

import tilefish
from tilefish import (
    DevScalar,
    Dim,
    InOut,
    Tensor,
    TilefishDeclarationError,
    TilefishManifestError,
    TilefishPreconditionError,
    TilefishRuntimeError,
    _plan,
    bf16,
    f32,
    i32,
)
from tilefish._planrun import PlanRunner, SharedArena, carried_uses

ROW = 64 * 4  # the bytes of one row of the stages' [Rs, 64] f32 tensors


class Backend(FakeBackend):
    def memset(self, dst, nbytes, stream):
        self.log.append(("memset", dst, nbytes, stream))

    def current_device(self):
        return 0

    def launches(self):
        return [e for e in self.log if e[0] in ("launch", "gemm", "copy", "memset")]


def pointers(entry):
    return [v for kind, v in entry[3] if kind == "pointer"]


def shared(graphs=k.STAGES, name=None, backend=None):
    be = backend or Backend()
    return SharedArena([PlanRunner(tilefish.plan(g), FakeProvider(), be) for g in graphs], name=name), be


def x_of(rows, at=1 << 36):
    return FakeTensor((rows, 64), "f32", at)


# -- the declaration and the document ------------------------------------------------------------------------------------------------------------


def test_a_carried_parameter_is_a_buffer_of_role_carried_with_no_offset():
    p = tilefish.plan(k.stage_a)
    buffers = {b["name"]: b for b in p.buffers}
    assert buffers["h"] == {"name": "h", "role": "carried", "dtype": "f32", "shape": ["Rs", 64], "offset": None, "bytes": 24 * ROW}
    assert p.carried == ["h"] and p.outputs == ["y"] and p.arena_bytes == 0
    assert "carried (the arena's carried region)" in p.describe()
    # the store into h is written by the launch that makes the value: no intermediate, no copy step
    assert [(s["entry"], [a["binding"] for a in s["args"]]) for s in p.steps(Rs=8)] == [("tripled", ["x", "h"]), ("plus_one", ["x", "y"])]
    assert carried_uses(p) == ((), ("h",))
    assert carried_uses(tilefish.plan(k.stage_b)) == (("h",), ())
    assert carried_uses(tilefish.plan(k.stage_update)) == (("h",), ("h",))  # updated in place: read first
    assert carried_uses(tilefish.plan(k.stage_zero)) == ((), ("h",))  # zeroed by a memset: written first
    assert tilefish.plan(k.stage_zero).steps(Rs=8)[0] == {"kind": "memset", "args": [{"name": "dst", "binding": "h"}]}


def test_a_store_into_a_carried_parameter_is_a_copy_when_the_value_is_read_again():
    @tilefish.graph(carried=("h",))
    def twice_read(x: Tensor[k.Rs, 64, f32], h: InOut[Tensor[k.Rs, 64, f32]]):
        t = k.tripled(x)
        h[...] = t
        return k.plus_one(t)  # t is read after the store

    ss = tilefish.plan(twice_read).steps(Rs=8)
    assert [s.get("entry", s["kind"]) for s in ss] == ["tripled", "copy", "plus_one"]
    assert ss[1]["args"] == [{"name": "src", "binding": "t0"}, {"name": "dst", "binding": "h"}]


def test_the_declaration_names_tensor_parameters_in_or_inout():
    Rb = Dim("Rb", buckets=[8])
    with pytest.raises(TilefishDeclarationError, match="carried= names 'zz', which is not a parameter"):

        @tilefish.graph(carried=("zz",))
        def g1(x: Tensor[k.Rs, 64, f32]):
            return k.plus_one(x)

    with pytest.raises(TilefishDeclarationError, match="carried= names 'n', which is a DevScalar"):

        @tilefish.graph(carried=("n",))
        def g2(x: Tensor[k.Rs, 64, f32], n: DevScalar[i32]):
            return k.plus_one(x)

    with pytest.raises(TilefishDeclarationError, match="a name is given twice in carried="):

        @tilefish.graph(carried=("x", "x"))
        def g3(x: Tensor[k.Rs, 64, f32]):
            return k.plus_one(x)

    with pytest.raises(TilefishDeclarationError, match="inputs= names 'h', which is carried"):

        @tilefish.graph(carried=("h",), inputs=("h",))
        def g4(h: Tensor[k.Rs, 64, f32], x: Tensor[k.Rs, 64, f32]):
            return k.plus_one(h)

    @tilefish.graph(carried=("h",), inputs=("x",))
    def g5(h: Tensor[k.Rs, 64, f32], x: Tensor[k.Rs, 64, f32], w: Tensor[Rb, 64, bf16]):
        return k.plus_one(h)

    roles = {b["name"]: b["role"] for b in tilefish.plan(g5).buffers}
    assert roles == {"h": "carried", "x": "input", "w": "weight", "out0": "output"} and g5.param_roles == {"x": "input", "w": "weight"}


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d["buffers"][1].__setitem__("offset", 0), r"is carried \(the host places it in the arena's carried region\): its offset must be null"),
        (lambda d: d["buffers"][1].__setitem__("role", "carry"), "role must be one of"),
    ],
)
def test_a_carried_buffer_with_an_offset_is_refused(change, message):
    doc = json.loads(json.dumps(tilefish.plan(k.stage_a)._doc))
    change(doc)
    with pytest.raises(TilefishManifestError, match=message):
        _plan.validate_doc(doc, made=True)


def test_a_carried_buffer_is_read_before_any_step_writes_it_and_the_plan_exports_and_loads(tmp_path):
    planset = plan_fakes.export_set([tilefish.plan(g) for g in k.STAGES], tmp_path / "stages")
    for g in k.STAGES:  # check_against_manifest: stage_b and stage_update read h first, which an earlier execution wrote
        back = _plan.load_plan(tmp_path / "stages", g, pin=planset.digest)
        assert back.carried == ["h"] and back.digest == planset[g.name].digest


# -- the layout of the arena -------------------------------------------------------------------------------------------------------------------


def test_the_carried_region_follows_the_largest_plan_and_is_laid_out_by_name():
    @tilefish.graph(carried=("b", "a"))
    def two(x: Tensor[k.Rs, 64, f32], b: InOut[Tensor[k.Rs, 64, f32]], a: InOut[Tensor[k.Rs, 3, f32]]):
        b[...] = k.tripled(x)
        a[...] = tilefish.zeros_like(a)
        return k.plus_one(k.tripled(x))

    p = tilefish.plan(two, reuse=False)
    layout = _plan.carried_layout([p._doc])
    a_bytes = 24 * 3 * 4
    assert layout.base == p.arena_bytes and p.arena_bytes > 0  # the intermediate of the second tripled
    assert layout.regions == {"a": (p.arena_bytes, a_bytes, "f32"), "b": (p.arena_bytes + 512, 24 * ROW, "f32")}  # a first (code point order), 288 -> 512
    assert layout.total == p.arena_bytes + 512 + 24 * ROW
    s, _ = shared((k.stage_a, k.stage_b, two))
    assert s.carried == {"a": (layout.base, a_bytes, "f32"), "b": (layout.base + 512, 24 * ROW, "f32"), "h": (layout.base + 512 + 24 * ROW, 24 * ROW, "f32")}
    assert s.arena_bytes == layout.base + 512 + 2 * 24 * ROW and s.sizes == {"stage_a": 0, "stage_b": 0, "two": p.arena_bytes}


def test_two_plans_that_give_one_carried_name_two_dtypes_are_refused():
    @tilefish.graph(carried=("h",))
    def other(h: Tensor[k.Rs, 64, bf16]):
        return h.reshape(k.Rs * 64)

    with pytest.raises(TilefishRuntimeError, match="the carried buffer 'h' is f32 in the plan of graph 'stage_a' but bf16 in the plan of graph 'other'"):
        shared((k.stage_a, other))


# -- runs ------------------------------------------------------------------------------------------------------------------------------------------


def test_a_stage_hands_its_carried_buffer_to_the_next_on_a_shared_arena():
    s, be = shared()
    base = s.carried["h"][0]
    x = x_of(8)
    y = s.run(k.stage_a, x)  # writes h: x is the only argument
    assert y.shape == (8, 64)
    z = s.run(k.stage_b)  # reads h: no argument, Rs = 8 from the shape h was written with
    assert z.shape == (8, 64)
    launches = be.launches()
    assert pointers(launches[0]) == [x._ptr, ARENA + base]  # tripled writes the arena's carried buffer
    assert pointers(launches[2]) == [ARENA + base, z._ptr]  # plus_one reads it
    assert be.log.count(("arena", s.arena_bytes)) == 1 and s.arena_bytes == base + 24 * ROW
    w = s.run(k.stage_update, x_of(8, 2 << 36))
    assert pointers(be.launches()[3])[0] == ARENA + base and w.shape == (8, 64)
    s.run(k.stage_zero, x_of(24))  # h now holds the 24 rows a memset of its bytes left
    assert be.launches()[5] == ("memset", ARENA + base, 24 * ROW, 7)
    assert s.run(k.stage_b).shape == (24, 64)


def test_a_carried_buffer_read_before_any_execution_wrote_it_is_refused_and_nothing_is_enqueued():
    s, be = shared()
    with pytest.raises(TilefishPreconditionError, match="the dimension Rs of graph 'stage_b' is not determined"):
        s.run(k.stage_b)  # nothing binds Rs: h was never written
    with pytest.raises(TilefishPreconditionError, match="reads the carried buffer 'h' with the shape \\[8, 64\\], but no execution on this arena has written it"):
        s.run(k.stage_update, x_of(8))
    assert be.launches() == [] and not [e for e in be.log if e[0] == "wait"]


Nf = Dim("Nf", buckets=[512, 1536])


@tilefish.kernel
def flat_add(a: Tensor[Nf, f32], b: Tensor[Nf, f32]) -> Tensor[Nf, f32]:
    return a + b


@tilefish.graph(carried=("h",))
def flat_reader(h: Tensor[Nf, f32], z: Tensor[Nf, f32]):
    """Reads ``h`` as one flat axis: the shape the stages write (``[Rs, 64]``) is another rank, so it binds nothing and is refused under the lock."""
    return flat_add(h, z)


def test_a_carried_buffer_read_in_another_shape_than_it_was_written_is_refused_and_nothing_is_enqueued():
    s, be = shared()
    s.run(k.stage_a, x_of(8))
    enqueued = len(be.launches()), len([e for e in be.log if e[0] == "wait"])
    with pytest.raises(TilefishPreconditionError, match="(the carried buffer)"):
        s.run(k.stage_update, x_of(24))  # y says Rs = 24, h holds 8 rows: the dimensions conflict
    assert (len(be.launches()), len([e for e in be.log if e[0] == "wait"])) == enqueued
    s.run(k.stage_a, x_of(24))
    s.run(k.stage_update, x_of(24))
    # a plan that reads h in a shape of another rank binds nothing from it and is refused by the check under the arena's lock, before its wait
    s2, be2 = shared((k.stage_a, flat_reader))
    s2.run(k.stage_a, x_of(8))
    enqueued = len(be2.launches()), len([e for e in be2.log if e[0] == "wait"])
    with pytest.raises(TilefishPreconditionError, match=r"reads the carried buffer 'h' with the shape \[512\], but the execution on this arena that wrote it last left the shape \[8, 64\]"):
        s2.run(flat_reader, FakeTensor((512,), "f32", 9 << 36))
    assert (len(be2.launches()), len([e for e in be2.log if e[0] == "wait"])) == enqueued


def test_a_failed_execution_leaves_its_carried_buffers_unknown():
    s, be = shared()
    s.run(k.stage_a, x_of(8))
    original = be.launch

    def failing(loaded, grid, args, stream):
        if loaded.stem == "tripled":
            raise RuntimeError("launch failed")
        original(loaded, grid, args, stream)

    be.launch = failing
    with pytest.raises(RuntimeError, match="launch failed"):
        s.run(k.stage_a, x_of(8))
    be.launch = original
    with pytest.raises(TilefishPreconditionError, match="is not determined|no execution on this arena has written it"):
        s.run(k.stage_b)


def test_a_plan_alone_has_its_carried_buffer_in_an_arena_of_its_own():
    be = Backend()
    a = PlanRunner(tilefish.plan(k.stage_a), FakeProvider(), be)
    a.run((x_of(8),), {})
    assert ("arena", 24 * ROW) in be.log  # the plan's region (no bytes) and its carried buffer
    assert a.arena_of(0).shapes == {"h": (8, 64)}  # what this arena's h holds now
    b = PlanRunner(tilefish.plan(k.stage_b), FakeProvider(), be)
    with pytest.raises(TilefishPreconditionError, match="is not determined"):
        b.run((), {})  # another plan, another arena: it holds no h
    u = PlanRunner(tilefish.plan(k.stage_update), FakeProvider(), be)
    with pytest.raises(TilefishPreconditionError, match="no execution on this arena has written it"):
        u.run((x_of(8),), {})  # a plan that reads h first never runs alone: nothing but its own runs could have written its arena's h


def test_a_call_does_not_give_a_carried_parameter():
    s, _ = shared()
    with pytest.raises(TilefishPreconditionError, match="the carried parameters h are held by the plan's arena"):
        s.run(k.stage_a, x_of(8), x_of(8, 3 << 36))
    with pytest.raises(TilefishPreconditionError, match="the carried parameters h are held by the plan's arena"):
        s.run(k.stage_b, h=x_of(8))


def test_a_captured_plan_checks_and_writes_the_carried_buffer_at_every_replay():
    s, be = shared()
    s.run(k.stage_a, x_of(8))
    cap_b = s.capture(k.stage_b)
    cap_zero = s.capture(k.stage_zero, x_of(24))
    cap_b.replay()  # h holds 8 rows: fine
    cap_zero.replay()  # h now holds 24 rows
    with pytest.raises(TilefishPreconditionError, match="reads the carried buffer 'h' with the shape \\[8, 64\\], but the execution on this arena that wrote it last left the shape \\[24, 64\\]"):
        cap_b.replay()
    s.run(k.stage_a, x_of(8))
    cap_b.replay()


def test_a_step_alone_takes_the_callers_tensor_for_a_carried_buffer():
    s, be = shared()
    h, z = x_of(8, 5 << 36), x_of(8, 6 << 36)
    s.run_step(k.stage_b, 0, {"h": h, "z": z})
    assert pointers(be.launches()[0]) == [h._ptr, z._ptr]


def test_a_tap_of_a_carried_buffer_copies_it_from_the_arena():
    s, be = shared()
    tap = x_of(8, 7 << 36)
    s.runner(k.stage_a).run((x_of(8),), {}, taps={"h": tap})
    copy = next(e for e in be.launches() if e[0] == "copy")
    assert copy[1:4] == (tap._ptr, ARENA + s.carried["h"][0], 8 * ROW)
