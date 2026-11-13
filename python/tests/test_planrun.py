"""Running a launch plan, with no GPU: a runner on a fake backend that records what it is asked to do.

The runner is the part of ``tilefish.run(graph, ...)`` that matches the arguments, picks the bucket, checks every precondition of every step *before the first launch*,
writes the device scalars and launches the steps in order with the pointers ``base + offset``. A fake backend (no torch, no driver) records each launch with its grid and
arguments, so these tests see the order, the addresses and the scalars, and that a refusal launches nothing. The real launches are the gate on a GPU
(``test_gpu_plan.py``).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import contextlib
import math
from types import SimpleNamespace

import plan_model as pm
import pytest
from fake_torch import FakeDevice, FakeDtype, FakeTensor

import tilefish
from tilefish import TilefishPreconditionError, TilefishRuntimeError, _plan
from tilefish._planrun import Backend, JitProvider, PlanRunner

ARENA = 1 << 46  # the fake arena's address (256-byte aligned, like torch's)


class HostDevice:
    type = "cpu"

    def __str__(self):
        return "cpu"


class HostTensor(FakeTensor):
    """A tensor of one element on the host: what an ``int`` passed for a ``DevScalar`` becomes."""

    def __init__(self, value, dtype):
        super().__init__((), dtype, 0)
        self.device = HostDevice()
        self.value = value

    def reshape(self, *shape):
        return self


class Device(FakeTensor):
    pass


class FakeBackend(Backend):
    """Records every launch; allocates addresses by counting."""

    def __init__(self):
        self.log = []
        self.capture = False
        self.next = 1 << 40
        self.arenas = 0

    def device_index(self, device):
        return 0

    def capability(self, index):
        return 103

    def capturing(self, index):
        return self.capture

    def scope(self, index):
        return contextlib.nullcontext()

    def stream(self, index):
        return 7

    def alloc_arena(self, nbytes, index):
        self.arenas += 1
        self.log.append(("arena", nbytes))
        return object(), ARENA

    def empty(self, shape, dtype, index):
        self.next += 1 << 30
        self.log.append(("empty", tuple(shape), dtype))
        return FakeTensor(shape, dtype, self.next)

    def devscalar_tensor(self, value, dtype):
        return HostTensor(value, dtype)

    def devscalar_source(self, value, dtype, index):
        return value, value.device.type == "cuda"

    def write_devscalar(self, arena, offset, dtype, source, on_device):
        self.log.append(("devscalar", offset, getattr(source, "value", "tensor"), on_device))

    def load(self, built, index):
        self.log.append(("load", built.spec.stem))
        return SimpleNamespace(stem=built.spec.stem)

    def launch(self, loaded, grid, args, stream):
        self.log.append(("launch", loaded.stem, tuple(grid), tuple(args), stream))

    def gemm(self, spec, algo, a, b, bias, out, workspace, stream, index):
        self.log.append(("gemm", spec.stem, a, b, bias, out, workspace, stream))

    def warm(self, index, library):
        self.log.append(("warm", library))

    def launches(self):
        return [e for e in self.log if e[0] in ("launch", "gemm")]


class FakeProvider:
    def kernel(self, callee, spec, index):
        return SimpleNamespace(spec=spec, num_warps=spec.num_warps)

    def pin(self, spec, index):
        return SimpleNamespace(algo="ab" * 64)


class Weights:
    """Every weight and activation of the block as a fake tensor at its own address."""

    def __init__(self, m, base=1 << 36):
        self.next = base
        shapes = {
            "x": (m, 256), "w_ln1": (256,), "w_qkv": (768, 256), "b_qkv": (768,), "w_o": (256, 256), "w_ln2": (256,), "w_up": (1024, 256), "w_down": (256, 1024),
        }
        self.t = {k: self.make(s) for k, s in shapes.items()}
        self.n = FakeTensor((1,), "i32", self.take())

    def take(self):
        ptr = self.next
        self.next += 1 << 24
        return ptr

    def make(self, shape, dtype="bf16"):
        return FakeTensor(shape, dtype, self.take())

    def args(self):
        t = self.t
        return (t["x"], self.n, *[t[k] for k in pm.WEIGHT_NAMES])


def runner(plan=None, backend=None):
    plan = plan or tilefish.plan(pm.block)
    backend = backend or FakeBackend()
    return PlanRunner(plan, FakeProvider(), backend), backend


def offsets_of(plan):
    return {b["name"]: b["offset"] for b in plan.buffers}


def test_a_run_launches_the_steps_in_order_with_the_addresses_of_the_arena_and_the_externals():
    r, be = runner()
    w = Weights(64)
    out = r.run(w.args(), {})
    off = offsets_of(r.plan)
    assert out.shape == (64, 256) and out.dtype.short == "bf16"
    assert [e[1] for e in be.launches()] == [
        "rmsnorm", "gemm_qkv.M64", "attention", "gemm_o.M64", "add_rmsnorm", "gemm_up.M64", "gelu", "gemm_down.M64", "add",
    ]
    launches = be.launches()
    t = w.t
    # a kernel: pointers in ABI order, then the runtime scalars; the grid from the declared rule
    assert launches[0] == ("launch", "rmsnorm", (64,), (("pointer", t["x"]._ptr), ("pointer", t["w_ln1"]._ptr), ("pointer", ARENA + off["t0"])), 7)
    # an elementwise kernel gets the element count of its tensor as its runtime scalar, and a grid over it
    assert launches[6][2:4] == ((64,), (("pointer", ARENA + off["t6"]), ("pointer", ARENA + off["t7"]), ("scalar", 64 * 1024)))
    assert launches[8][2] == (16,) and launches[8][3][-1] == ("scalar", 64 * 256)  # cdiv(16384, 1024) blocks
    # the attention step reads q, k and v as views of the fused output: the pointer of the view, and the strides as scalars
    att = launches[2]
    fused = ARENA + off["t1"]
    assert att[2] == (64, 4)
    pointers = [v for k, v in att[3] if k == "pointer"]
    assert pointers == [fused, fused + 256 * 2, fused + 512 * 2, ARENA + off["t2"], ARENA + off["n"]]
    scalars = [v for k, v in att[3] if k == "scalar"]
    assert scalars == [768, 64, 768, 64, 768, 64, 64]  # q_s0, q_s1, k_s0, k_s1, v_s0, v_s1, rows
    # a GEMM: a, b, bias, out and the workspace of the arena, the same one for every GEMM
    gemm = launches[1]
    assert gemm[1:] == ("gemm_qkv.M64", ARENA + off["t0"], t["w_qkv"]._ptr, t["b_qkv"]._ptr, ARENA + off["t1"], ARENA + off["workspace"], 7)
    assert {g[6] for g in launches if g[0] == "gemm"} == {ARENA + off["workspace"]}
    # the last step writes the output the run allocated and returned
    assert launches[-1][3][2] == ("pointer", out._ptr)


def test_the_device_scalar_is_written_before_the_first_launch_from_what_the_caller_passed():
    r, be = runner()
    w = Weights(32)
    r.run((w.t["x"], 5, *[w.t[k] for k in pm.WEIGHT_NAMES]), {})  # an int
    kinds = [e[0] for e in be.log]
    assert kinds.index("devscalar") < kinds.index("launch")
    assert be.log[kinds.index("devscalar")] == ("devscalar", offsets_of(r.plan)["n"], 5, False)
    r.run(w.args(), {})  # a tensor on the device
    assert [e for e in be.log if e[0] == "devscalar"][-1][3] is True


def test_a_device_scalar_that_is_not_an_integer_of_its_dtype_is_refused_before_anything_runs():
    r, be = runner()
    w = Weights(32)
    for bad in (5.5, 2**31, -(2**31) - 1):
        with pytest.raises(TilefishPreconditionError, match="not an integer that fits i32"):
            r.run((w.t["x"], bad, *[w.t[k] for k in pm.WEIGHT_NAMES]), {})
    assert not be.launches() and not any(e[0] == "devscalar" for e in be.log)
    r.run((w.t["x"], 5.0, *[w.t[k] for k in pm.WEIGHT_NAMES]), {})  # a float that is a whole number is the integer
    assert [e for e in be.log if e[0] == "devscalar"][-1][2] == 5


def test_the_arena_is_allocated_once_and_the_binaries_are_loaded_once_per_bucket():
    r, be = runner()
    w = Weights(64)
    r.run(w.args(), {})
    r.run(w.args(), {})
    assert be.arenas == 1 and [e for e in be.log if e[0] == "arena"] == [("arena", r.plan.arena_bytes)]
    assert sorted(e[1] for e in be.log if e[0] == "load") == ["add", "add_rmsnorm", "attention", "gelu", "rmsnorm"]  # five kernels, once
    assert len(be.launches()) == 18
    w32 = Weights(32)
    r.run(w32.args(), {})
    assert be.arenas == 1 and len([e for e in be.log if e[0] == "load"]) == 10  # another bucket: its own binaries, the same arena


def test_each_bucket_selects_its_own_binaries_and_grids():
    r, be = runner()
    for m in pm.BUCKETS:
        be.log.clear()
        r.run(Weights(m).args(), {})
        launches = be.launches()
        assert [e[1] for e in launches if e[0] == "gemm"] == [f"gemm_qkv.M{m}", f"gemm_o.M{m}", f"gemm_up.M{m}", f"gemm_down.M{m}"]
        assert launches[0][2] == (m,) and launches[2][2] == (m, 4)


def test_prepare_does_everything_a_capture_cannot_and_a_captured_run_then_launches():
    r, be = runner()
    r.prepare(0)
    assert be.arenas == 1 and len([e for e in be.log if e[0] == "load"]) == 15 and ("warm", True) in be.log
    assert not be.launches() and not any(e[0] == "empty" for e in be.log)
    be.capture = True
    w = Weights(64)
    n = w.n
    out = FakeTensor((64, 256), "bf16", 1 << 50)
    r.run(w.args(), {}, outputs=out)  # captured: only launches and a device-to-device write
    assert len(be.launches()) == 9 and not any(e[0] in ("arena", "empty") for e in be.log[-12:])
    with pytest.raises(TilefishRuntimeError, match="host value"):
        r.run((w.t["x"], 5, *[w.t[k] for k in pm.WEIGHT_NAMES]), {}, outputs=out)
    assert n is w.n


def test_a_capture_that_needs_what_a_capture_cannot_make_says_so():
    r, be = runner()
    w = Weights(64)
    be.capture = True
    with pytest.raises(TilefishRuntimeError, match="has not been prepared for {'M': 64}"):
        r.run(w.args(), {})
    be.capture = False
    r._bucket({"M": 64}, 0)  # the binaries are loaded, the arena is not
    be.capture = True
    with pytest.raises(TilefishRuntimeError, match="arena of the plan of graph 'block' has not been allocated"):
        r.run(w.args(), {})
    be.capture = False
    r.prepare(0, [{"M": 64}])
    be.capture = True
    with pytest.raises(TilefishRuntimeError, match="has not been prepared for {'M': 32}"):
        r.run(Weights(32).args(), {})
    with pytest.raises(TilefishRuntimeError, match="cannot be prepared while a CUDA graph is being captured"):
        r.prepare(0)


def test_a_precondition_that_fails_launches_nothing_not_even_the_device_scalar():
    def nothing_ran(be):
        assert not [e for e in be.log if e[0] in ("launch", "gemm", "devscalar")], be.log

    # a weight that is not 16-byte aligned
    r, be = runner()
    w = Weights(64)
    w.t["w_up"]._ptr += 2
    with pytest.raises(TilefishPreconditionError, match="not aligned to 16 bytes") as e:
        r.run(w.args(), {})
    assert e.value.kind == "pointer_alignment" and "gemm_up.M64" in str(e.value)
    nothing_ran(be)
    # a bucket that is not one
    r, be = runner()
    with pytest.raises(TilefishPreconditionError, match=r"dimension M is 48, which is not one of the buckets \[32, 64, 128\]") as e:
        r.run(Weights(48).args(), {})
    assert e.value.kind == "bucket"
    nothing_ran(be)
    # a dtype
    r, be = runner()
    w = Weights(64)
    w.t["w_qkv"] = w.make((768, 256), "f32")
    with pytest.raises(TilefishPreconditionError, match="dtype") as e:
        r.run(w.args(), {})
    assert e.value.kind == "dtype"
    nothing_ran(be)
    # a shape
    r, be = runner()
    w = Weights(64)
    w.t["w_o"] = w.make((256, 128))
    with pytest.raises(TilefishPreconditionError):
        r.run(w.args(), {})
    nothing_ran(be)
    # a device scalar of two elements
    r, be = runner()
    w = Weights(64)
    w.n = FakeTensor((2,), "i32", w.take())
    with pytest.raises(TilefishPreconditionError, match="one element in device memory"):
        r.run(w.args(), {})
    nothing_ran(be)
    # an output that is also an input
    r, be = runner()
    w = Weights(64)
    with pytest.raises(TilefishPreconditionError, match="overlaps 'x'") as e:
        r.run(w.args(), {}, outputs=w.t["x"])
    assert e.value.kind == "alias"
    nothing_ran(be)
    # an output of the wrong shape
    r, be = runner()
    w = Weights(64)
    with pytest.raises(TilefishPreconditionError, match="must be a contiguous bf16 tensor of shape"):
        r.run(w.args(), {}, outputs=FakeTensor((64, 255), "bf16", 1 << 50))
    nothing_ran(be)
    with pytest.raises(TilefishPreconditionError, match="outputs= must give the 1 returned value"):
        r.run(w.args(), {}, outputs=(FakeTensor((64, 256), "bf16", 1 << 50), FakeTensor((64, 256), "bf16", 1 << 51)))
    nothing_ran(be)


def test_an_unaligned_output_given_by_the_caller_is_refused_by_the_step_that_writes_it():
    r, be = runner()
    w = Weights(64)
    with pytest.raises(TilefishPreconditionError, match="not aligned to 16 bytes"):
        r.run(w.args(), {}, outputs=FakeTensor((64, 256), "bf16", (1 << 50) + 8))
    assert not be.launches()


def test_the_outputs_the_caller_gives_are_the_ones_written_and_returned():
    r, be = runner()
    w = Weights(128)
    out = FakeTensor((128, 256), "bf16", 1 << 50)
    assert r.run(w.args(), {}, outputs=[out]) is out
    assert not any(e[0] == "empty" for e in be.log) and be.launches()[-1][3][2] == ("pointer", 1 << 50)


def test_a_run_by_keyword_is_the_same_as_by_position():
    r, be = runner()
    w = Weights(64)
    r.run((), dict(zip(["x", "n", *pm.WEIGHT_NAMES], w.args())))
    first = [e[1:] for e in be.launches()]
    r2, be2 = runner()
    r2.run(w.args(), {})
    assert [e[1:] for e in be2.launches()] == first


def test_jit_and_replay_run_one_code_path_so_the_same_plan_launches_the_same_things(tmp_path):
    import plan_fakes

    made = tilefish.plan(pm.block)
    exported = plan_fakes.export(made, tmp_path)
    a, b = FakeBackend(), FakeBackend()
    jit = PlanRunner(made, FakeProvider(), a)
    replay = PlanRunner(exported, FakeProvider(), b)
    for m in pm.BUCKETS:
        w = Weights(m)
        jit.run(w.args(), {}, outputs=FakeTensor((m, 256), "bf16", 1 << 50))
        replay.run(w.args(), {}, outputs=FakeTensor((m, 256), "bf16", 1 << 50))
    assert a.launches() == b.launches() and len(a.launches()) == 27


def test_an_architecture_the_plan_was_not_built_for_is_refused(tmp_path):
    import plan_fakes

    exported = plan_fakes.export(tilefish.plan(pm.block), tmp_path)
    be = FakeBackend()
    be.capability = lambda index: 90
    r = PlanRunner(exported, FakeProvider(), be)
    with pytest.raises(TilefishRuntimeError, match="was built for sm_103, but device 0 is sm_90"):
        r.run(Weights(64).args(), {})
    assert not be.launches()


def test_the_runner_is_made_and_a_plan_loaded_without_torch(tmp_path):
    import plan_fakes
    from tilefish import _planrun

    exported = plan_fakes.export(tilefish.plan(pm.block), tmp_path)
    runner_ = _planrun._replay_runner(pm.block, tmp_path, exported.digest, False)
    assert runner_.plan.digest == exported.digest
    _planrun.clear_caches()


def test_a_graph_with_an_in_place_tensor_writes_the_callers_tensor_and_allocates_one_buffer():
    p = tilefish.plan(pm.accumulate)
    r, be = runner(p)
    a = FakeTensor((16, 128), "f32", 1 << 36)
    state = FakeTensor((16, 128), "f32", 1 << 37)
    out = r.run((a, state), {})
    launches = be.launches()
    assert [e[1] for e in launches] == ["bump", "bump", "twice", "bump", "plus"]
    assert launches[0][3][0] == ("pointer", state._ptr) and launches[1][3][0] == ("pointer", state._ptr)  # the same tensor, in place, twice
    scratch = ARENA + offsets_of(p)["t0"]
    assert launches[2][3][1] == ("pointer", scratch) and launches[3][3][0] == ("pointer", scratch)
    assert launches[4][3][2] == ("pointer", out._ptr)
    with pytest.raises(TilefishPreconditionError, match="overlaps 'a'"):
        r.run((a, a), {})  # the in-place tensor is also an input


def test_a_graph_can_be_run_with_the_jit_provider_made_lazily():
    # the provider of jit asks runtime for the binary only when a step is bound: constructing the runner touches neither torch nor Triton
    r = PlanRunner(tilefish.plan(pm.block), JitProvider(), FakeBackend())
    assert r.plan.name == "block" and math.prod([1]) == 1 and _plan.PLAN_NAME == "plan.json"
