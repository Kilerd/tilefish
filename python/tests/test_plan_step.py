"""One step of a plan, launched on the caller's tensors: ``tilefish.runtime.run_step(graph, step, buffers, mode="replay", plan=..., pin=...)``.

The binary is the plan's (in ``replay``, from the manifest the pinned plan pins) and so is the binding of its arguments: each argument is the buffer and
the view the step names, made of the caller's tensor for that buffer. Every check of the step is made with the caller's addresses before the launch.
What it is for: one op of an exported plan replayed on the inputs of a reference, under the pin the product uses. The Rust host has
``PlanRun::launch_step``. Nothing here needs a GPU (``test_gpu_plan_step.py``).
"""

import plan_model as pm
import pytest
from fake_torch import FakeTensor
from test_planrun import ARENA, FakeBackend, FakeProvider
from test_plan_names import named

import tilefish
from tilefish import TilefishPreconditionError, TilefishRuntimeError
from tilefish._planrun import PlanRunner


def runner(graph=None):
    p = tilefish.plan(graph or pm.block)
    be = FakeBackend()
    return PlanRunner(p, FakeProvider(), be), be, p


def tensor(p, name, m, ptr, **values):
    dtype = next(b["dtype"] for b in p.buffers if b["name"] == name)
    return FakeTensor(p.shape_of(name, M=m, **values), dtype, ptr)


def test_a_gemm_step_runs_on_the_callers_tensors_with_the_plans_workspace():
    r, be, p = runner()
    a, w, o = tensor(p, "t2", 64, 1 << 40), FakeTensor((256, 256), "bf16", 2 << 40), tensor(p, "t3", 64, 3 << 40)
    r.run_step(3, {"t2": a, "w_o": w, "t3": o})
    offsets = {b["name"]: b["offset"] for b in p.buffers}
    (gemm,) = be.launches()
    assert gemm[1:] == ("gemm_o.M64", a._ptr, w._ptr, None, o._ptr, ARENA + offsets["workspace"], 7)
    assert [e[0] for e in be.log].count("wait") == 1  # ordered with the runs of the plan, as a run is


def test_a_kernel_step_reads_the_views_the_plan_names():
    r, be, p = runner()
    qkv, out, n = tensor(p, "t1", 32, 1 << 40), tensor(p, "t2", 32, 2 << 40), FakeTensor((1,), "i32", 3 << 40)
    r.run_step(2, {"t1": qkv, "t2": out, "n": n})
    (launch,) = be.launches()
    pointers = [v for k, v in launch[3] if k == "pointer"]
    assert launch[1] == "attention" and pointers == [qkv._ptr, qkv._ptr + 256 * 2, qkv._ptr + 512 * 2, out._ptr, n._ptr]


def test_a_returned_value_is_given_by_its_name_or_by_out_i():
    r, be, p = runner(named)
    x, res, hidden = FakeTensor((8, 64), "f32", 1 << 40), FakeTensor((8, 64), "f32", 2 << 40), FakeTensor((8, 64), "f32", 3 << 40)
    for given in ({"residual": res, "x": x, "hidden": hidden}, {"out1": res, "x": x, "out0": hidden}):
        r.run_step(1, given)
    first, second = be.launches()
    assert first[3] == second[3]


def test_what_a_step_is_given_is_checked_before_it_runs():
    r, be, p = runner()
    a, w, o = tensor(p, "t2", 64, 1 << 40), FakeTensor((256, 256), "bf16", 2 << 40), tensor(p, "t3", 64, 3 << 40)
    for step, buffers, bucket, error, message in (
        (9, {"t2": a, "w_o": w, "t3": o}, None, TilefishRuntimeError, "the step is an index of the 9 steps"),
        (3, {"t2": a, "w_o": w}, None, TilefishRuntimeError, "t3 missing"),
        (3, {"t2": a, "w_o": w, "t3": o, "x": FakeTensor((64, 256), "bf16", 4 << 40)}, None, TilefishRuntimeError, "x not bound by it"),
        (3, {"t2": a, "w_o": w, "t3": o, "workspace": o}, None, TilefishRuntimeError, "is not a buffer of the plan"),
        (3, {"t2": a, "w_o": w, "t3": FakeTensor((32, 256), "bf16", 3 << 40)}, None, TilefishPreconditionError, "fit 0 buckets"),
        (3, {"t2": a, "w_o": w, "t3": o}, {"M": 32}, TilefishPreconditionError, "must be a contiguous bf16 tensor of shape"),
        (3, {"t2": a, "w_o": w, "t3": o}, {"M": 33}, TilefishPreconditionError, "is not a bucket"),
        (3, {"t2": a, "w_o": w, "t3": FakeTensor((64, 256), "bf16", (1 << 40) + 64)}, None, TilefishPreconditionError, "overlap"),
        (3, [a, w, o], None, TilefishRuntimeError, "buffers= maps"),
    ):
        with pytest.raises(error, match=message):
            r.run_step(step, buffers, bucket)
    assert not be.launches()
