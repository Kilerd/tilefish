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
from fake_torch import FakeTensor

import tilefish
from tilefish import Tensor, TilefishPreconditionError, TilefishRuntimeError, _plan, f32
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
        self.capture_ident = 1  # the id of the capture the current stream takes part in, while `capture` is set
        self.current = 7  # the current stream
        self.next = 1 << 40
        self.arenas = 0
        self.graphs = 0

    def device_index(self, device):
        return 0

    def capability(self, index):
        return 103

    def capturing(self, index):
        return self.capture

    def scope(self, index):
        return contextlib.nullcontext()

    def stream(self, index):
        return self.current

    def synchronize_stream(self, stream):
        self.log.append(("drain", stream))

    def capture_id(self, stream):
        return self.capture_ident if self.capture else None

    @contextlib.contextmanager
    def use_stream(self, stream, index):
        saved = self.current
        self.current = saved if stream is None else stream
        try:
            yield
        finally:
            self.current = saved

    def capture_graph(self, index, stream, enqueue):
        self.graphs += 1
        on = 99 if stream is None else stream
        self.log.append(("begin", on))
        enqueue(on)
        self.log.append(("end", on))
        return SimpleNamespace(name=f"graph{self.graphs}")

    def replay_graph(self, graph, stream):
        self.log.append(("graph", graph.name, stream))

    def alloc_arena(self, nbytes, index):
        self.arenas += 1
        self.log.append(("arena", nbytes))
        return object(), ARENA

    def new_event(self, index):
        self.log.append(("event", index))
        return SimpleNamespace(name=f"event{self.arenas}")

    def wait_event(self, event, stream, capturing):
        self.log.append(("wait", event.name, stream, capturing))

    def record_event(self, event, stream, capturing):
        self.log.append(("record", event.name, stream, capturing))

    def synchronize_event(self, event, index):
        self.log.append(("synchronize", event.name, index))

    def destroy_event(self, event, index):
        self.log.append(("destroy", event.name, index))

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
    for bad in (5.5, 2**31, -(2**31) - 1, float("nan"), float("inf")):
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
        assert not [e for e in be.log if e[0] in ("launch", "gemm", "devscalar", "wait", "record")], be.log

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


def test_the_arena_belongs_to_the_plan_layout_not_to_a_runner_so_replacing_a_runner_never_frees_what_a_cuda_graph_holds(tmp_path):
    """A captured graph holds the address of the arena. Runners come and go (a directory is re-exported, jit and replay are both used); the arena stays."""
    import plan_fakes

    made = tilefish.plan(pm.block)
    exported = plan_fakes.export(made, tmp_path)
    be = FakeBackend()
    first = PlanRunner(made, FakeProvider(), be)
    first.run(Weights(64).args(), {})
    arena = first.arena_of(0)
    assert be.arenas == 1 and arena is not None
    second = PlanRunner(exported, FakeProvider(), be)  # a replay runner of the same layout: the same arena, though a different plan object and manifest
    second.run(Weights(64).args(), {})
    assert be.arenas == 1 and second.arena_of(0) is arena and made.layout_key == exported.layout_key
    del first
    third = PlanRunner(made, FakeProvider(), be)
    third.run(Weights(32).args(), {})
    assert be.arenas == 1 and third.arena_of(0) is arena  # the arena outlived the runner that made it
    other = PlanRunner(tilefish.plan(pm.block, reuse=True), FakeProvider(), be)  # another layout is another arena
    other.run(Weights(64).args(), {})
    assert be.arenas == 2 and other.arena_of(0) is not arena


def test_reference_mode_refuses_a_device_scalar_that_is_not_an_integer_of_its_dtype(monkeypatch):
    """``3.7`` for a ``DevScalar[i32]`` was truncated to 3 by reference mode (``torch.tensor(3.7, dtype=int32)``) where jit and replay refuse it: refused the same way."""
    from tilefish import runtime

    def reached(*args, **kwargs):
        pytest.fail("a tensor was made of a value that is refused")

    monkeypatch.setattr(runtime, "_torch", lambda: SimpleNamespace(int32="torch.int32", bfloat16="torch.bfloat16", tensor=reached))
    w = Weights(32)
    for bad in (3.7, -0.5, 2**31, float("nan"), float("inf")):
        with pytest.raises(TilefishPreconditionError, match="not an integer that fits i32") as e:
            tilefish.run(pm.block, w.t["x"], bad, *[w.t[k] for k in pm.WEIGHT_NAMES], mode="reference")
        assert e.value.arg == "n"
    from tilefish._planrun import devscalar_number

    n = next(p for p in pm.block.signature.params if p.name == "n")
    assert devscalar_number(n, 7.0) == 7 and type(devscalar_number(n, 7.0)) is int and devscalar_number(n, -(2**31)) == -(2**31)


def _run_ops(be):
    """The log of the runs, without what prepares them: the waits, the writes, the launches and the records."""
    return [e for e in be.log if e[0] in ("wait", "record", "devscalar", "launch", "gemm")]


def test_runs_on_two_streams_are_ordered_on_the_device_by_the_order_event_of_the_arena():
    """Every run of a layout on a device shares the arena, and the arena's lock orders only the enqueueing: each run makes its stream wait for the most recent
    record of the arena's event before its first write, and records the event on its stream after its last launch, so two runs on two streams never overlap
    on the device (the reviewer's race, 5 of 5 corrupted before: ``test_gpu_plan.py`` holds it at 0 of 20)."""
    be = FakeBackend()
    streams = iter([11, 22, 11, 33])
    be.stream = lambda index: next(streams)
    r, _ = runner(backend=be)
    r2 = PlanRunner(r.plan, FakeProvider(), be)  # another runner of the same layout: the same arena, and the same event
    for runner_, m in ((r, 64), (r2, 32), (r, 64), (r2, 128)):
        runner_.run(Weights(m).args(), {})
    assert be.log.count(("event", 0)) == 1 and be.arenas == 1  # one event, made with the arena
    ops = _run_ops(be)
    assert len(ops) == 4 * 12
    for i, stream in enumerate((11, 22, 11, 33)):
        run = ops[12 * i : 12 * (i + 1)]
        assert run[0] == ("wait", "event1", stream, False), run[0]  # before the device scalar is written: it is in the arena
        assert run[1][0] == "devscalar"
        assert [e[0] for e in run[2:-1]] == ["launch", "gemm", "launch", "gemm", "launch", "gemm", "launch", "gemm", "launch"]
        assert {e[-1] for e in run[2:-1]} == {stream}  # every launch on the run's stream
        assert run[-1] == ("record", "event1", stream, False)


def test_a_captured_run_waits_for_and_records_the_order_event_as_nodes_of_the_graph():
    """Inside a capture the wait and the record are external event nodes, so every replay of the graph is ordered with the runs (and other replays) issued
    before and after it: a captured graph's arena is the plan's arena."""
    r, be = runner()
    r.prepare(0)
    be.capture = True
    w = Weights(64)
    r.run(w.args(), {}, outputs=FakeTensor((64, 256), "bf16", 1 << 50))
    ops = _run_ops(be)
    assert ops[0] == ("wait", "event1", 7, True) and ops[-1] == ("record", "event1", 7, True) and len(ops) == 12


def test_a_run_that_fails_while_it_launches_still_records_the_order_event():
    """What a run enqueued before a launch failed still uses the arena: the next run must come after it."""

    class Failing(FakeBackend):
        def launch(self, loaded, grid, args, stream):
            if loaded.stem == "attention":
                raise TilefishRuntimeError("the launch failed")
            super().launch(loaded, grid, args, stream)

    be = Failing()
    r, _ = runner(backend=be)
    with pytest.raises(TilefishRuntimeError, match="the launch failed"):
        r.run(Weights(64).args(), {})
    ops = _run_ops(be)
    assert ops[0][0] == "wait" and [e[1] for e in ops[2:-1]] == ["rmsnorm", "gemm_qkv.M64"] and ops[-1] == ("record", "event1", 7, False)


def test_a_plan_with_no_arena_has_nothing_to_order():
    @tilefish.graph
    def single(a: Tensor[pm.Q, pm.W, f32]) -> Tensor[pm.Q, pm.W, f32]:
        return pm.twice(a)

    p = tilefish.plan(single)
    assert p.arena_bytes == 0
    r, be = runner(p)
    r.run((FakeTensor((16, 128), "f32", 1 << 36),), {})
    assert not any(e[0] in ("event", "wait", "record") for e in be.log) and len(be.launches()) == 1


def test_clear_caches_waits_for_the_last_run_on_every_arena_before_it_frees_it(monkeypatch):
    from tilefish import _planrun

    be = FakeBackend()
    monkeypatch.setattr(_planrun, "_torch_backend", [be])
    r, _ = runner(backend=be)
    r.run(Weights(64).args(), {})
    arena = r.arena_of(0)
    assert arena is not None
    _planrun.clear_caches()
    tail = be.log[-2:]
    assert tail == [("synchronize", "event1", 0), ("destroy", "event1", 0)] and r.arena_of(0) is None

    class Stuck(FakeBackend):
        def synchronize_event(self, event, index):
            raise TilefishRuntimeError("cannot wait while a stream is being captured")

    be = Stuck()
    monkeypatch.setattr(_planrun, "_torch_backend", [be])
    r, _ = runner(backend=be)
    r.run(Weights(64).args(), {})
    with pytest.raises(TilefishRuntimeError, match="cannot wait"):
        _planrun.clear_caches()
    assert r.arena_of(0) is not None  # nothing that a kernel may still use was freed


def test_clear_caches_waits_for_a_run_in_progress_on_another_thread_and_a_run_after_it_rebuilds(monkeypatch):
    """A run on another thread that is between its first launch and its record holds the gate: clear_caches waits for it before it waits for the order event
    and frees the arena (before, it freed the arena and destroyed the event under the run). A run that starts after clear_caches makes a new arena, a new
    event and loads its binaries again."""
    import threading

    from tilefish import _planrun

    inside, release = threading.Event(), threading.Event()

    class Slow(FakeBackend):
        def launch(self, loaded, grid, args, stream):
            if loaded.stem == "attention" and not release.is_set():
                inside.set()
                assert release.wait(10)
            super().launch(loaded, grid, args, stream)

    be = Slow()
    monkeypatch.setattr(_planrun, "_torch_backend", [be])
    r, _ = runner(backend=be)
    errors = []

    def run():
        try:
            r.run(Weights(64).args(), {})
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    t = threading.Thread(target=run)
    t.start()
    assert inside.wait(10)
    cleared = threading.Thread(target=_planrun.clear_caches)
    cleared.start()
    cleared.join(0.3)
    assert cleared.is_alive() and not any(e[0] in ("synchronize", "destroy") for e in be.log)  # it waits for the run
    release.set()
    t.join(10)
    cleared.join(10)
    assert not cleared.is_alive() and errors == []
    kinds = [e[0] for e in be.log]
    assert kinds.index("synchronize") > max(i for i, k in enumerate(kinds) if k == "record")  # after the run's record
    assert kinds[-2:] == ["synchronize", "destroy"] and r.arena_of(0) is None
    loads = kinds.count("load")
    r.run(Weights(64).args(), {})  # rebuilt cleanly: a new arena, a new event, the binaries loaded again
    assert be.arenas == 2 and kinds.count("event") + 1 == [e[0] for e in be.log].count("event") and [e[0] for e in be.log].count("load") > loads
    assert [e for e in _run_ops(be)][-1] == ("record", "event2", 7, False)


def test_clear_caches_inside_a_run_is_refused_instead_of_waiting_for_itself(monkeypatch):
    from tilefish import _planrun

    class Clearing(FakeBackend):
        def launch(self, loaded, grid, args, stream):
            _planrun.clear_caches()

    be = Clearing()
    monkeypatch.setattr(_planrun, "_torch_backend", [be])
    r, _ = runner(backend=be)
    with pytest.raises(TilefishRuntimeError, match="clear_caches was called while this thread runs a plan"):
        r.run(Weights(64).args(), {})
    assert _run_ops(be)[-1][0] == "record"  # the run that failed still recorded the order event
    _planrun.clear_caches()  # and the gate was released


def test_the_record_is_made_even_when_it_fails_once_and_the_stream_is_drained_when_it_fails_twice():
    """Once anything is enqueued after the wait for the order event, the record follows: a record interrupted once (a KeyboardInterrupt landing in it) is made
    again and the interrupt raised after; a record that fails twice drains the stream on the host before the error is raised, so the work cannot overlap the next
    run. A wait that failed enqueued nothing and is followed by no record."""

    class Flaky(FakeBackend):
        def __init__(self, failures, error):
            super().__init__()
            self.failures, self.error = failures, error

        def record_event(self, event, stream, capturing):
            if self.failures:
                self.failures -= 1
                raise self.error
            super().record_event(event, stream, capturing)

    be = Flaky(1, KeyboardInterrupt())
    r, _ = runner(backend=be)
    with pytest.raises(KeyboardInterrupt):
        r.run(Weights(64).args(), {})
    ops = _run_ops(be)
    assert ops[0][0] == "wait" and ops[-1] == ("record", "event1", 7, False) and len(be.launches()) == 9

    be = Flaky(2, TilefishRuntimeError("cuEventRecord failed"))
    r, _ = runner(backend=be)
    with pytest.raises(TilefishRuntimeError, match="cuEventRecord failed"):
        r.run(Weights(64).args(), {})
    assert be.log[-1] == ("drain", 7) and not any(e[0] == "record" for e in be.log)

    class NoWait(FakeBackend):
        def wait_event(self, event, stream, capturing):
            raise TilefishRuntimeError("cuStreamWaitEvent failed")

    be = NoWait()
    r, _ = runner(backend=be)
    with pytest.raises(TilefishRuntimeError, match="cuStreamWaitEvent failed"):
        r.run(Weights(64).args(), {})
    assert not any(e[0] in ("record", "launch", "gemm", "devscalar", "drain") for e in be.log)


def test_a_second_run_of_a_plan_in_one_capture_on_another_stream_is_refused():
    """Two runs of one plan on two forked streams of one torch capture: each has an external wait node and an external record node, and the second's wait does
    not depend on the first's record inside the graph, so a replay would run both at once on the arena. Refused before anything of the second is enqueued; on
    the same stream (ordered by the graph), or in another capture, it is allowed."""
    r, be = runner()
    r.prepare(0)
    be.capture, be.capture_ident = True, 41
    w = Weights(64)
    out = FakeTensor((64, 256), "bf16", 1 << 50)
    be.current = 11
    r.run(w.args(), {}, outputs=out)
    r.run(w.args(), {}, outputs=out)  # the same stream of the capture: ordered by the graph
    before = list(be.log)
    be.current = 22
    with pytest.raises(TilefishRuntimeError, match="already run in this CUDA graph capture on another stream"):
        r.run(w.args(), {}, outputs=out)
    assert be.log == before  # nothing enqueued
    be.capture_ident = 42  # another capture
    r.run(w.args(), {}, outputs=out)
    assert _run_ops(be)[-1] == ("record", "event1", 22, True)


def _captured(r, m=64, n=5, outputs=None, stream=None):
    w = Weights(m)
    return r.capture((w.t["x"], *[w.t[k] for k in pm.WEIGHT_NAMES]), {"n": n}, outputs, stream), w


def test_tilefish_capture_captures_only_the_launches_and_a_replay_orders_them_with_the_runs():
    """``tilefish.capture``: the graph holds the launches of the steps and nothing else (no event node, no write of a device scalar); ``replay`` takes the arena's
    lock, waits for the order event, writes the device scalars, launches the graph and records the event, as a run does."""
    r, be = runner()
    r.run(Weights(64).args(), {})  # prepares the bucket and the arena
    start = len(be.log)
    cap, _ = _captured(r)
    captured = be.log[start:]
    kinds = [e[0] for e in captured]
    assert kinds[0] == "empty" and kinds[1] == "begin" and kinds[-1] == "end"  # the output is allocated before the capture
    assert set(kinds[2:-1]) == {"launch", "gemm"} and len(kinds[2:-1]) == 9
    assert {e[-1] for e in captured[2:-1]} == {99}  # on the stream of the capture
    assert cap.bucket == {"M": 64} and cap.outputs.shape == (64, 256)
    start = len(be.log)
    cap.replay()
    assert be.log[start:] == [
        ("wait", "event1", 7, False),
        ("devscalar", offsets_of(r.plan)["n"], 5, False),
        ("graph", "graph1", 7),
        ("record", "event1", 7, False),
    ]
    start = len(be.log)
    cap.replay(stream=33)
    assert [e[-1] for e in be.log[start:]] == [False, False, 33, False] and be.log[start][2] == 33 and be.log[-1] == ("record", "event1", 33, False)


def test_set_devscalar_changes_the_value_the_next_replay_writes_and_enqueues_nothing():
    r, be = runner()
    r.prepare(0)
    cap, w = _captured(r)
    offset = offsets_of(r.plan)["n"]
    start = len(be.log)
    cap.set_devscalar("n", 9)
    assert be.log[start:] == []  # nothing on the device until the next replay
    cap.replay()
    assert ("devscalar", offset, 9, False) in be.log[start:]
    with pytest.raises(TilefishPreconditionError, match="not an integer that fits i32"):
        cap.set_devscalar("n", 3.7)
    with pytest.raises(TilefishPreconditionError, match="has dtype torch.int64"):
        cap.set_devscalar("n", HostTensor(4, "i64"))
    with pytest.raises(TilefishRuntimeError, match="'m' is not a device scalar of graph 'block'; its device scalars are \\['n'\\]"):
        cap.set_devscalar("m", 1)
    start = len(be.log)
    cap.replay()
    assert ("devscalar", offset, 9, False) in be.log[start:]  # a refused value changed nothing
    cap.set_devscalar("n", HostTensor(11, "i32"))
    cap.set_devscalar("n", w.n)  # a tensor on the device: read by a stream-ordered copy at every replay
    start = len(be.log)
    cap.replay()
    assert ("devscalar", offset, "tensor", True) in be.log[start:]


def test_a_captured_plan_keeps_its_externals_and_its_arena_alive_and_is_not_replayed_after_clear_caches(monkeypatch):
    import gc
    import weakref

    from tilefish import _planrun

    be = FakeBackend()
    monkeypatch.setattr(_planrun, "_torch_backend", [be])
    r, _ = runner(backend=be)
    r.prepare(0)
    cap, w = _captured(r)
    x, out, arena = weakref.ref(w.t["x"]), weakref.ref(cap.outputs), weakref.ref(r.arena_of(0))
    del w
    gc.collect()
    assert x() is not None and out() is not None  # the graph holds their addresses
    _planrun.clear_caches()
    gc.collect()
    assert r.arena_of(0) is None and arena() is not None  # out of the registry, but not freed: the captured plan holds it
    with pytest.raises(TilefishRuntimeError, match="clear_caches\\(\\) unloaded the binaries its graph launches"):
        cap.replay()
    del cap
    gc.collect()
    assert x() is None and out() is None and arena() is None


def test_tilefish_capture_says_what_it_takes():
    r, be = runner()
    r.prepare(0)
    w = Weights(64)
    weights = [w.t[k] for k in pm.WEIGHT_NAMES]
    with pytest.raises(TilefishPreconditionError, match="devscalars= gives the value of every device scalar of graph 'block' \\(n\\); n is missing"):
        r.capture((w.t["x"], *weights), {}, None, None)
    with pytest.raises(TilefishPreconditionError, match="m is not one"):
        r.capture((w.t["x"], *weights), {"n": 1, "m": 2}, None, None)
    with pytest.raises(TilefishPreconditionError, match="takes at most 8 externals"):
        r.capture((w.t["x"], w.n, *weights), {"n": 1}, None, None)
    be.capture = True
    with pytest.raises(TilefishRuntimeError, match="tilefish.capture makes a CUDA graph of its own"):
        r.capture((w.t["x"], *weights), {"n": 1}, None, None)
    be.capture = False
    cap = r.capture((w.t["x"], *weights), {"n": 1}, None, None)
    be.capture = True
    with pytest.raises(TilefishRuntimeError, match="not replayed while the stream captures"):
        cap.replay()
    with pytest.raises(TilefishRuntimeError, match="takes a @tilefish.graph"):
        tilefish.capture(pm.twice, w.t["x"])


def test_what_depends_on_the_bucket_only_is_checked_when_the_bucket_is_prepared_not_in_the_middle_of_a_run():
    """The device's grid limits and the validity of a pinned GEMM algorithm do not depend on an address: a refusal must come before the first launch, and never inside a capture."""

    class Refusing(FakeBackend):
        def __init__(self, what):
            super().__init__()
            self.what = what

        def check_grid(self, loaded, grid):
            if self.what == "grid" and loaded.stem == "attention":
                raise TilefishRuntimeError("the grid has too many blocks")

        def check_gemm(self, spec, algo, index):
            if self.what == "gemm" and spec.stem == "gemm_o.M64":
                raise TilefishRuntimeError("the pinned algorithm is not valid here")

    for what in ("grid", "gemm"):
        be = Refusing(what)
        r, _ = runner(backend=be)
        with pytest.raises(TilefishRuntimeError, match="too many blocks|not valid here"):
            r.run(Weights(64).args(), {})
        assert not be.launches() and not any(e[0] == "devscalar" for e in be.log)  # not one step ran before the refusal
    be = FakeBackend()
    r, _ = runner(backend=be)
    r.prepare(0)  # the checks ran at prepare: a capture that follows has nothing left to refuse


def test_a_tensor_of_the_caller_inside_the_arena_is_refused():
    r, be = runner()
    w = Weights(64)
    w.t["w_o"] = FakeTensor((256, 256), "bf16", ARENA + 4096)  # a weight that lives in the plan's own memory
    with pytest.raises(TilefishPreconditionError, match="'w_o' overlaps the arena of the plan") as e:
        r.run(w.args(), {})
    assert e.value.kind == "alias"
    assert not be.launches()
    w.t["w_o"] = FakeTensor((256, 256), "bf16", ARENA + r.plan.arena_bytes)  # just past it: fine
    r.run(w.args(), {})
    assert len(be.launches()) == 9


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
