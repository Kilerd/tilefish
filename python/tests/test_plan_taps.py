"""Debug taps: the intermediates of the very plan a product runs, copied out after a step, with the plan and its digest unchanged.

``tilefish.run(graph, ..., taps={"t3": tensor, ("t5", 4): tensor})`` (and ``tilefish.capture(..., taps=...)``) copies a buffer of the plan (all of it, at
the bucket) into the caller's tensor right after a step: the step given, or by default the last step of the bucket that writes the buffer. The copy is
enqueued on the plan's stream right after the step, before the next one may reuse the buffer's bytes (``reuse=True``), so it sees what that step
left. A tap is checked before anything is enqueued (the name, the step, the tensor's dtype, shape, contiguity, device, and that it overlaps nothing of
the call or the arena). The Rust host has ``PlanRun::launch_tapped``. Nothing here needs a GPU (``test_gpu_plan_taps.py``).
"""

import plan_model as pm
import pytest
from fake_torch import FakeTensor
from test_planrun import ARENA, FakeBackend, FakeProvider, Weights

import tilefish
from tilefish import TilefishPlanError, TilefishPreconditionError, TilefishRuntimeError
from tilefish._planrun import PlanRunner


def runner(reuse=False):
    p = tilefish.plan(pm.block, reuse=reuse)
    be = FakeBackend()
    return PlanRunner(p, FakeProvider(), be), be, p


def tap_tensor(p, name, m, ptr):
    dtype = next(b["dtype"] for b in p.buffers if b["name"] == name)
    return FakeTensor(p.shape_of(name, M=m), dtype, ptr)


def test_a_tap_copies_the_buffer_after_the_last_step_that_writes_it_and_the_plan_is_the_same():
    for reuse in (False, True):
        r, be, p = runner(reuse)
        digest = p.digest
        w = Weights(64)
        offsets = {b["name"]: b["offset"] for b in p.buffers}
        t1 = tap_tensor(p, "t1", 64, 9 << 40)
        t4 = tap_tensor(p, "t4", 64, 10 << 40)
        out = r.run(w.args(), {}, taps={"t1": t1, ("t4", 4): t4, "out0": tap_tensor(p, "out0", 64, 11 << 40)})
        log = [e for e in be.log if e[0] in ("launch", "gemm", "copy")]
        kinds = [e[0] if e[0] != "copy" else ("copy", e[1]) for e in log]
        # t1 (the fused qkv) is written by step 1 and copied right after it, before step 2; t4 after step 4; out0 after the last step
        assert kinds[2] == ("copy", t1._ptr) and kinds[6] == ("copy", t4._ptr) and kinds[-1] == ("copy", 11 << 40)
        assert log[2][2:4] == (ARENA + offsets["t1"], 64 * 768 * 2)
        assert log[-1][2] == out._ptr
        assert p.digest == digest  # the plan is what it was


def test_a_tap_is_checked_before_anything_runs():
    r, be, p = runner()
    w = Weights(32)
    good = tap_tensor(p, "t1", 32, 9 << 40)
    for taps, error, message in (
        ({"nope": good}, TilefishRuntimeError, "a tap is a buffer of the plan"),
        ({"workspace": good}, TilefishRuntimeError, "a tap is a buffer of the plan"),
        ({"x": good}, TilefishRuntimeError, "which no step writes"),
        ({("t1", 9): good}, TilefishRuntimeError, "the step is an index of the 9 steps"),
        ({("t1", True): good}, TilefishRuntimeError, "the step is an index"),
        ({"t1": FakeTensor((32, 700), "bf16", 9 << 40)}, TilefishPreconditionError, "must be a contiguous bf16 tensor of shape"),
        ({"t1": FakeTensor((32, 768), "f32", 9 << 40)}, TilefishPreconditionError, "must be a contiguous bf16 tensor"),
        ({"t1": FakeTensor((32, 768), "bf16", w.t["x"]._ptr)}, TilefishPreconditionError, "overlaps a tensor of the call"),
        ({"t1": good, "t2": FakeTensor(p.shape_of("t2", M=32), "bf16", (9 << 40) + 64)}, TilefishPreconditionError, "overlaps"),
        (["t1"], TilefishRuntimeError, "taps= maps a buffer name"),
    ):
        with pytest.raises(error, match=message):
            r.run(w.args(), {}, taps=taps)
    assert not [e for e in be.log if e[0] in ("launch", "gemm", "copy", "devscalar")]


def test_a_device_scalar_and_a_parameter_are_tapped_after_the_step_given_and_shape_of_names_the_bucket():
    r, be, p = runner()
    w = Weights(32)
    n = FakeTensor((), "i32", 9 << 40)
    r.run(w.args(), {}, taps={("n", 0): n})
    copy = [e for e in be.log if e[0] == "copy"]
    assert copy == [("copy", n._ptr, ARENA + 0, 4, 7)]
    assert p.shape_of("t1", M=128) == (128, 768) and p.shape_of("out0", M=32) == (32, 256)
    with pytest.raises(TilefishPlanError, match="has no bucket"):
        p.shape_of("t1", M=33)
    with pytest.raises(TilefishPlanError, match="no buffer 'nope'"):
        p.shape_of("nope", M=32)


def test_a_capture_holds_the_taps():
    r, be, p = runner()
    w = Weights(64)
    t1 = tap_tensor(p, "t1", 64, 9 << 40)
    cap = r.capture((w.t["x"], *[w.t[k] for k in pm.WEIGHT_NAMES]), {"n": 5}, None, None, taps={"t1": t1})
    inside = be.log[be.log.index(("begin", 99)) : be.log.index(("end", 99))]
    assert [e[1] for e in inside if e[0] == "copy"] == [t1._ptr]
    assert t1 in cap._externals
