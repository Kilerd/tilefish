"""Running a launch plan from Python: ``jit``, ``replay`` and ``reference`` of a ``@tilefish.graph`` (what :func:`tilefish.run` does with a graph), and
exporting one (:func:`tilefish.runtime.export`).

A :class:`PlanRunner` executes a plan document (:mod:`tilefish._plan`) on one stream, in order:

1. the arguments are matched to the graph's signature like a call is (:func:`~tilefish._launch.bind_call`); the dimensions they bind select the
   bucket, which must be one of the plan's exactly (there is no padding); the outputs are allocated (or the caller's ``outputs=``);
2. **every step's preconditions are checked before the first launch** (alignment of every pointer of every view, dtypes, static sizes, divisibility,
   aliasing, ``max_extent``, the exact bucket): a failure raises and launches nothing, not even the write of a device scalar;
3. the device scalars (``DevScalar`` inputs, slots of the arena) are written from the values the caller passed, and the steps are launched, each with
   the pointers ``base + offset``, the scalars and the grid that :func:`tilefish._plan.resolve_step` found (the one place that decides them, which is
   also what built the plan), through ``cuLaunchKernel`` or ``cublasLtMatmul`` on the current torch stream.

``jit`` and ``replay`` differ in one thing: where a step's binary comes from (built in memory through its compile spec, or read from a verified manifest).
Everything else is one code path, so they are identical by construction.

**Allocation and synchronization.** A run allocates nothing on the device but what the caller asks for (the outputs, unless ``outputs=`` gives them): the
arena is allocated once per plan layout and device and kept, the cuBLASLt workspace is a buffer of the arena, and a device scalar is written by a
stream-ordered copy (from a CUDA tensor) or an asynchronous copy from pinned memory (from an ``int`` or a CPU tensor). Nothing waits on the host for the device.

**Runs on several streams.** Every run of a plan layout on a device uses the same arena (its intermediates, device scalars and workspace), so two runs must
not overlap on the device. The arena has an *order event* (:class:`_Arena`): a run makes its stream (torch's current stream) wait for the event's most recent
record, enqueues its device-scalar writes and its launches, and records the event on its stream, all under the arena's lock (:func:`_ordered`; the record is
made whenever anything was enqueued after the wait, also when a launch fails or an interrupt lands). So **runs of one plan on different streams, from one
thread or many, are correct, and execute on the device one after the other**, in the order their host calls took the lock (the lock alone would order the
enqueueing, not the device); runs on one stream are ordered by the stream as before. Nothing waits on the host. (Runs of one plan do not overlap on the device,
so a run on another stream is not concurrent with one in flight; two plans with different layouts have different arenas and do run concurrently.) The externals
are the caller's: an input written on another stream, or an output read on one, is ordered by the caller, as for any torch op. :func:`clear_caches` frees
nothing under a run: it waits for the runs in progress on every thread (:class:`_Gate`) and for the last record of every arena's event.

**CUDA graphs: tilefish.capture.** :func:`capture` (``tilefish.capture(graph, *externals, devscalars=..., outputs=...)``) captures the plan in a CUDA graph
of its own and returns a :class:`CapturedPlan`. The graph holds the launches of the steps only; :meth:`CapturedPlan.replay` takes the arena's lock, makes its
stream wait for the order event, writes the device scalars (from pinned host memory: :meth:`CapturedPlan.set_devscalar` changes them, with no synchronization),
launches the graph and records the event, exactly as a run does. So **its replays are ordered with every run and every other replay of the plan, from any
thread, on any stream**. The ``CapturedPlan`` keeps the externals and the arena alive. This is the way to capture a plan that another thread may run.

**CUDA graphs: torch.cuda.graph.** A run under ``torch.cuda.graph(...)`` captures too. The things that would make a plan fail to capture are, each refused with a
message that says so: (1) the arena, the binary of a step (``cuModuleLoadData``, ``cuFuncLoad``) and the cuBLASLt handle must exist, so the plan has to have
run once (or :func:`prepare` called) outside the capture; (2) outputs allocated inside the capture belong to the graph (torch's private pool): that works, but
pass ``outputs=`` to have tensors of your own (and no allocation at all); (3) a device scalar passed as an ``int`` or a CPU tensor would be written by a host
copy, which cannot be captured: pass a CUDA tensor, which the captured copy reads at every replay (write it between replays to change the value); (4) the
externals must be the same tensors at every replay (a graph holds addresses); (5) the stream must be the one the capture uses (torch's current stream; the
capture is of that stream's work only), and every run of one plan in one capture must be on one stream (a second run on another, forked, stream of the same
capture is refused: its wait node would not depend on the first run's record node inside the graph, so a replay could run the two at once). Everything else a
run does is enqueued launches, which capture. **A captured graph's arena is the plan's arena** (the same memory, at the address the graph holds), and the wait
and the record of the order event are captured as *external* event nodes (``CU_EVENT_WAIT_EXTERNAL``, ``CU_EVENT_RECORD_EXTERNAL``): every replay of the graph
waits for the most recent record of the event made before it is launched, and records the event when its work is done.

**What that orders, and what it does not.** ``CUDAGraph.replay()`` is torch's: it takes no lock of tilefish. So a raw graph's replay is ordered only against
the runs whose record it sees: the runs and replays whose record was made before the replay is launched come before it, and the ones that wait after it is
launched come after it. A run on another thread that is between its wait and its record when the replay is launched has waited for the record before it and
has not recorded yet: **the replay and that run overlap on the arena** (the same intermediates, device scalar slots and workspace) and give wrong bytes with
no error (3 of 4000 runs of a plan on one thread, against a loop of replays on another, on a B300). A caller who replays a raw graph while another thread
may use the same plan (``tilefish.run``, the replay of a :class:`CapturedPlan`, whose wait and record are two calls too, or the replay of another raw graph)
must order them itself (one thread, or a lock of its own around all of them), or use :func:`capture`, whose replay takes the arena's lock.
"""

from __future__ import annotations

import contextlib
import math
import os
import pathlib
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import _plan, runtime
from ._launch import (
    TORCH_DTYPE_NAMES,
    bind_call,
    check_dim_constraints,
    check_tensor,
    dims_by_name,
    eval_dim,
    tensor_info,
)
from .decorators import ExternOp, GraphOp, KernelOp, LibraryOp
from .errors import TilefishManifestError, TilefishPreconditionError, TilefishRuntimeError
from ._plan import PLAN_NAME, Bound, Plan, plan as make_plan
from .trace import Value

_lock = threading.Lock()
_arena_lock = threading.Lock()
_loaded: "dict[tuple, tuple[tuple, Plan, PlanRunner]]" = {}  # (directory, graph) -> (stamp of its files, loaded plan, its runner), until a file in it changes
_runners: "weakref.WeakSet[PlanRunner]" = weakref.WeakSet()  # every runner alive, so that clear_caches can reset them (a runner is kept by its plan or by _loaded, never by a global)
_torch_backend: "list[TorchBackend]" = []
_epoch = [0]  # how many times clear_caches has run: a CapturedPlan is replayed only in the epoch it was captured in (its binaries are unloaded after)


class _Gate:
    """What keeps :func:`clear_caches` from freeing what a run is using: every run, capture, replay and ``prepare`` of a plan holds the gate *shared*
    while it uses an arena, a loaded binary or an order event (from the moment it looks them up to the record of the order event after its last launch);
    ``clear_caches`` holds it *exclusive* while it frees them and unloads the modules. Shared holders do not wait for each other; ``clear_caches`` waits for
    every run in progress to finish enqueueing, and a run that starts meanwhile waits for it (then finds no arena and no binary, and makes them again).

    Both sides are reentrant on one thread (``runtime.clear_caches`` holds the gate across ``_planrun.clear_caches`` and the unloading of the modules); a
    thread that holds it shared cannot take it exclusive (that would wait for itself), and is refused."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._shared = 0  # threads that hold it shared
        self._waiting = 0  # clear_caches calls waiting: a new run waits behind them, so that a stream of runs cannot starve a clear
        self._owner: "int | None" = None  # the thread that holds it exclusive
        self._local = threading.local()

    @contextlib.contextmanager
    def shared(self) -> Any:
        me = threading.get_ident()
        depth = getattr(self._local, "depth", 0)
        if depth or self._owner == me:  # this thread holds it already (shared, or exclusive)
            self._local.depth = depth + 1
            try:
                yield
            finally:
                self._local.depth = depth
            return
        with self._cond:
            while self._owner is not None or self._waiting:
                self._cond.wait()
            self._shared += 1
        self._local.depth = 1
        try:
            yield
        finally:
            self._local.depth = 0
            with self._cond:
                self._shared -= 1
                if not self._shared:
                    self._cond.notify_all()

    @contextlib.contextmanager
    def exclusive(self) -> Any:
        me = threading.get_ident()
        if self._owner == me:
            yield
            return
        if getattr(self._local, "depth", 0):
            raise TilefishRuntimeError("clear_caches was called while this thread runs a plan (it would wait for that run to end, which waits for it)")
        with self._cond:
            self._waiting += 1
            try:
                while self._owner is not None or self._shared:
                    self._cond.wait()
            finally:
                self._waiting -= 1
                self._cond.notify_all()  # the runs that waited behind this call, if it was interrupted
            self._owner = me
        try:
            yield
        finally:
            with self._cond:
                self._owner = None
                self._cond.notify_all()


_gate = _Gate()


def exclusive() -> Any:
    """The gate held exclusive (see :class:`_Gate`): no run, capture, replay or ``prepare`` of a plan is in progress until the block ends."""
    return _gate.exclusive()


def torch_backend() -> "TorchBackend":
    """The one :class:`TorchBackend` of the process: the arenas live in it, so every runner of one plan layout on one device shares one."""
    with _lock:
        if not _torch_backend:
            _torch_backend.append(TorchBackend())
        return _torch_backend[0]


def clear_caches() -> None:
    """Forget every loaded plan, reset every runner (its loaded binaries) and free every arena (for tests, and to give the memory back).

    **Nothing is freed under a run.** First the gate (:class:`_Gate`) is taken exclusive: it waits for every run, capture, replay and ``prepare`` that is in
    progress on any thread to have enqueued its last launch and recorded the order event, and the ones that start meanwhile wait until this returns (they then
    find no arena and no binary, and make them again, cleanly). Then, under each arena's lock, the arena is freed only once the work of every run on it is done:
    every run waits for the one before it (the arena's order event), so waiting on the host for the last record of that event (``cuEventSynchronize``) waits for
    all of them, whatever streams they ran on, and for the replays of a captured graph issued before this call. ``tilefish.runtime.clear_caches`` holds the gate
    across this and the unloading of the modules (after ``cuCtxSynchronize`` of every context that has one).

    A :class:`CapturedPlan` keeps its arena alive (the memory is not given back while it lives), but it cannot be replayed after this (its binaries are
    unloaded): its ``replay`` refuses, and it has to be captured again. **A CUDA graph captured with ``torch.cuda.graph`` from a plan holds the address of its
    arena** and nothing keeps that arena alive but this registry: the arenas are freed here and nowhere else (not when a runner is replaced, a plan directory
    re-exported or a plan garbage collected), so call it only when no such graph will be replayed again.
    """
    with _gate.exclusive():
        with _lock:
            _loaded.clear()
            runners = list(_runners)
            backends = list(_torch_backend)
            _epoch[0] += 1
        for r in runners:
            r._reset()
        for b in backends:
            registry = b.arena_registry()
            arenas = list(registry.items())
            for (_, index), arena in arenas:
                with arena.lock:
                    if arena.event is not None:
                        b.synchronize_event(arena.event, index)  # raises (and frees nothing) if it cannot wait: memory a kernel may still use is not given back
            for key, arena in arenas:
                with arena.lock:
                    if registry.get(key) is arena:
                        del registry[key]
                    if arena.event is not None:
                        b.destroy_event(arena.event, key[1])
                        arena.event = None


# -- what a runner asks of torch and the driver ---------------------------------------------------------------------------------------------


class Backend:
    """What a runner needs from torch and the CUDA driver. :class:`TorchBackend` is the real one; a test passes another that records what it is
    asked to do, which is how the order, the pointers and "nothing is launched when a precondition fails" are tested without a GPU."""

    def arena_registry(self) -> "dict[tuple[str, int], Any]":
        """The arenas this backend has made, by (layout of the plan, device): one per plan layout and device, kept until :func:`clear_caches`, whichever runners come and go."""
        return self.__dict__.setdefault("_arena_registry", {})

    def device_index(self, device: Any) -> int:
        raise NotImplementedError

    def capability(self, index: int) -> int:
        raise NotImplementedError

    def capturing(self, index: int) -> bool:
        raise NotImplementedError

    def scope(self, index: int) -> Any:
        raise NotImplementedError

    def stream(self, index: int) -> int:
        raise NotImplementedError

    def alloc_arena(self, nbytes: int, index: int) -> "tuple[Any, int]":
        """``(handle, address)`` of ``nbytes`` bytes of device memory, 256-byte aligned."""
        raise NotImplementedError

    def new_event(self, index: int) -> Any:
        """The order event of a new arena on device ``index`` (made with the arena, outside any capture)."""
        raise NotImplementedError

    def wait_event(self, event: Any, stream: int, capturing: bool) -> None:
        """Make what is enqueued on ``stream`` next wait for the most recent record of ``event``; ``capturing``: as an event wait node of the graph."""
        raise NotImplementedError

    def record_event(self, event: Any, stream: int, capturing: bool) -> None:
        """Record ``event`` on ``stream`` after what was enqueued; ``capturing``: as an event record node of the graph."""
        raise NotImplementedError

    def synchronize_event(self, event: Any, index: int) -> None:
        """Wait on the host for the most recent record of ``event``."""
        raise NotImplementedError

    def destroy_event(self, event: Any, index: int) -> None:
        raise NotImplementedError

    def synchronize_stream(self, stream: int) -> None:
        """Wait on the host for everything enqueued on ``stream`` (what a run does when it cannot record the order event after its launches)."""
        raise NotImplementedError

    def capture_id(self, stream: int) -> "int | None":
        """The id of the capture ``stream`` takes part in (``cuStreamGetCaptureInfo``), or ``None`` when it is not capturing."""
        raise NotImplementedError

    def use_stream(self, stream: Any, index: int) -> Any:
        """A context in which ``stream`` (a ``torch.cuda.Stream``, or ``None`` for the current one) is the current stream of device ``index``."""
        raise NotImplementedError

    def capture_graph(self, index: int, stream: Any, enqueue: Any) -> Any:
        """Capture what ``enqueue(stream_handle)`` enqueues in a new CUDA graph on ``stream`` (a ``torch.cuda.Stream``, or ``None`` for a stream of
        its own), in the thread-local capture mode, and return the graph (what :meth:`replay_graph` replays). The capture is ended, and the graph
        dropped, when ``enqueue`` raises."""
        raise NotImplementedError

    def replay_graph(self, graph: Any, stream: int) -> None:
        """Launch the graph on ``stream``, the current stream (``cuGraphLaunch``)."""
        raise NotImplementedError

    def empty(self, shape: "Sequence[int]", dtype: str, index: int) -> Any:
        raise NotImplementedError

    def devscalar_tensor(self, value: "int | float", dtype: str) -> Any:
        """A tensor of one element on the host holding ``value``: what an ``int`` passed for a ``DevScalar`` becomes."""
        raise NotImplementedError

    def devscalar_source(self, value: Any, dtype: str, index: int) -> "tuple[Any, bool]":
        """``(source, on_device)``: what to write into a slot, from the tensor the caller passed (a CPU tensor or a CUDA tensor)."""
        raise NotImplementedError

    def write_devscalar(self, arena: Any, offset: int, dtype: str, source: Any, on_device: bool) -> None:
        raise NotImplementedError

    def load(self, built: Any, index: int) -> Any:
        raise NotImplementedError

    def launch(self, loaded: Any, grid: "Sequence[int]", args: "Sequence[tuple[str, int]]", stream: int) -> None:
        raise NotImplementedError

    def gemm(self, spec: Any, algo: str, a: int, b: int, bias: "int | None", out: int, workspace: int, stream: int, index: int) -> None:
        raise NotImplementedError

    def warm(self, index: int, library: bool) -> None:
        raise NotImplementedError

    def check_grid(self, loaded: Any, grid: "Sequence[int]") -> None:
        """Refuse a grid the device cannot launch (it depends on the bucket only, so a runner checks it when it prepares the bucket, not at a launch)."""

    def check_gemm(self, spec: Any, algo: str, index: int) -> None:
        """Refuse a pinned algorithm that is not valid for the problem on this device, or needs more workspace than recorded (the same: once per bucket)."""


class TorchBackend(Backend):
    def __init__(self) -> None:
        self.torch = runtime._torch()

    def device_index(self, device: Any) -> int:
        if isinstance(device, int) and not isinstance(device, bool):
            return device
        return runtime._device_index(self.torch, str(device))

    def capability(self, index: int) -> int:
        return runtime._capability(self.torch, index)

    def capturing(self, index: int) -> bool:
        with self.torch.cuda.device(index):  # the current stream of that device
            return bool(self.torch.cuda.is_current_stream_capturing())

    def scope(self, index: int) -> Any:
        return self.torch.cuda.device(index)

    def stream(self, index: int) -> int:
        return int(self.torch.cuda.current_stream(index).cuda_stream)

    def alloc_arena(self, nbytes: int, index: int) -> "tuple[Any, int]":
        t = self.torch.empty(nbytes, dtype=self.torch.uint8, device=f"cuda:{index}")
        return t, int(t.data_ptr())

    def new_event(self, index: int) -> Any:
        from . import _cuda

        with self.torch.cuda.device(index):
            return _cuda.new_event()

    def wait_event(self, event: Any, stream: int, capturing: bool) -> None:
        from . import _cuda

        _cuda.wait_event(stream, event, external=capturing)

    def record_event(self, event: Any, stream: int, capturing: bool) -> None:
        from . import _cuda

        _cuda.record_event(stream, event, external=capturing)

    def synchronize_event(self, event: Any, index: int) -> None:
        from . import _cuda

        with self.torch.cuda.device(index):
            _cuda.synchronize_event(event)

    def destroy_event(self, event: Any, index: int) -> None:
        from . import _cuda

        with self.torch.cuda.device(index):
            _cuda.destroy_event(event)

    def synchronize_stream(self, stream: int) -> None:
        from . import _cuda

        _cuda.synchronize_stream(stream)

    def capture_id(self, stream: int) -> "int | None":
        from . import _cuda

        return _cuda.capture_id(stream)

    def use_stream(self, stream: Any, index: int) -> Any:
        if stream is None:
            return contextlib.nullcontext()
        if int(getattr(stream, "device_index", index)) != index:
            raise TilefishRuntimeError(f"stream= is a stream of device {stream.device_index}, but the plan runs on device {index}")
        return self.torch.cuda.stream(stream)

    def capture_graph(self, index: int, stream: Any, enqueue: Any) -> Any:
        torch = self.torch
        graph = torch.cuda.CUDAGraph()
        on = torch.cuda.Stream(device=index) if stream is None else stream  # a capture is not made on the default stream
        with torch.cuda.device(index), torch.cuda.stream(on):
            if torch.cuda.is_current_stream_capturing():
                raise TilefishRuntimeError("the stream given to tilefish.capture is capturing a CUDA graph already: a plan is captured on a stream that is not")
            # thread-local: another thread's calls are not refused while this one captures (torch.cuda.graph's default, "global", refuses them)
            graph.capture_begin(capture_error_mode="thread_local")
            try:
                enqueue(int(on.cuda_stream))
            except BaseException:
                with contextlib.suppress(Exception):
                    graph.capture_end()  # the stream is usable again; the graph is dropped
                raise
            graph.capture_end()
        return graph

    def replay_graph(self, graph: Any, stream: int) -> None:
        graph.replay()  # on torch's current stream, which is `stream` (the caller made it current)

    def empty(self, shape: "Sequence[int]", dtype: str, index: int) -> Any:
        return self.torch.empty(tuple(shape), dtype=getattr(self.torch, TORCH_DTYPE_NAMES[dtype].split(".")[1]), device=f"cuda:{index}")

    def _dtype(self, dtype: str) -> Any:
        return getattr(self.torch, TORCH_DTYPE_NAMES[dtype].split(".")[1])

    def devscalar_tensor(self, value: "int | float", dtype: str) -> Any:
        return self.torch.tensor(value, dtype=self._dtype(dtype))

    def devscalar_source(self, value: Any, dtype: str, index: int) -> "tuple[Any, bool]":
        return value.reshape(()), bool(value.is_cuda)

    def write_devscalar(self, arena: Any, offset: int, dtype: str, source: Any, on_device: bool) -> None:
        torch = self.torch
        want = self._dtype(dtype)
        slot = arena[offset : offset + _plan.ITEMSIZE[dtype]].view(want).reshape(())
        if on_device:
            slot.copy_(source)  # stream-ordered, on the current stream: it can be captured, and it re-reads the tensor at every replay
        else:
            pinned = torch.empty((), dtype=want, pin_memory=True)  # the pinned allocator keeps the block until the copy is done
            pinned.copy_(source)
            slot.copy_(pinned, non_blocking=True)

    def load(self, built: Any, index: int) -> Any:
        from . import _cuda

        with self.torch.cuda.device(index):
            loaded = _cuda.load(built, index)
            load = getattr(_cuda._driver(), "cuFuncLoad", None)  # CUDA 12.4+: load the function now, not at its first launch (lazy loading in a capture)
            if load is not None:
                with contextlib.suppress(Exception):
                    load(loaded.function)
        return loaded

    def launch(self, loaded: Any, grid: "Sequence[int]", args: "Sequence[tuple[str, int]]", stream: int) -> None:
        loaded.launch(grid, args, stream)

    def gemm(self, spec: Any, algo: str, a: int, b: int, bias: "int | None", out: int, workspace: int, stream: int, index: int) -> None:
        from . import _library

        _library.run_gemm_ptrs(spec, algo, a, b, bias, out, stream, index, workspace_ptr=workspace)

    def warm(self, index: int, library: bool) -> None:
        if library:
            from . import _cublaslt

            _cublaslt.warm(index)

    def check_grid(self, loaded: Any, grid: "Sequence[int]") -> None:
        dims = tuple(grid)
        padded = dims + (1,) * (3 - len(dims))
        for axis, (g, limit) in enumerate(zip(padded, loaded.grid_limits)):
            if not 1 <= g <= limit:
                raise TilefishRuntimeError(
                    f"'{loaded.built.stem}': the grid {dims} has {g} blocks along {'XYZ'[axis]}, but this device takes 1 to {limit} there"
                )

    def check_gemm(self, spec: Any, algo: str, index: int) -> None:
        from . import _cublaslt

        _cublaslt.check_algo(spec.m, spec.n, spec.k, spec.bias, algo, workspace_bytes=spec.workspace_bytes, device_index=index)


# -- where a step's binary comes from ----------------------------------------------------------------------------------------------------


class JitProvider:
    """The binaries ``jit`` builds: the current source through its compile spec, in memory; a GEMM pinned in memory the way ``export`` pins it."""

    def kernel(self, callee: _plan.Callee, spec: Any, index: int) -> Any:
        return runtime.jit_built(callee.target, spec, index)

    def pin(self, spec: Any, index: int) -> Any:
        return runtime.jit_pin(spec, index)


class ReplayProvider:
    """The binaries of a verified manifest, after every check a replay makes; ``pinned`` says the caller vouches for the manifest (it pinned the plan)."""

    def __init__(self, manifest: Any, pinned: bool) -> None:
        self.manifest = manifest
        self.pinned = pinned

    def kernel(self, callee: _plan.Callee, spec: Any, index: int) -> Any:
        return runtime.replay_built(callee.target, spec, self.manifest, self.pinned, index)

    def pin(self, spec: Any, index: int) -> Any:
        return runtime.replay_pin(spec, self.manifest, index)


# -- the runner --------------------------------------------------------------------------------------------------------------------------


@dataclass
class _Arena:
    """The memory of one plan layout on one device, and what orders the runs on it.

    ``event`` is the arena's *order event*: every run, under ``lock``, makes its stream wait for the event's most recent record, enqueues its writes and launches,
    and records the event on its stream (:func:`_ordered`). So the runs on one arena execute one after the other on the device, in the order their host calls took
    the lock, whatever streams and threads they come from (``lock`` alone orders the enqueueing, not the device); :meth:`CapturedPlan.replay` does the same around
    the launch of its graph. Under a ``torch.cuda.graph`` capture the wait and the record are external event nodes, so a replay of that graph takes its place in
    that order when it is launched, but takes no lock: see the module docstring for what that orders and what it does not.

    ``captured`` is ``(capture id, stream)`` of the last run captured in a ``torch.cuda.graph`` on this arena: a second run of the arena in the same capture on
    another stream is refused (its external wait node would not depend on the first run's record node, so one replay would race with itself).
    """

    handle: Any
    ptr: int
    nbytes: int
    lock: threading.Lock
    event: Any = None
    captured: "tuple[int, int] | None" = None


@dataclass
class _StepRT:
    bound: Bound
    loaded: Any  # the loaded kernel of a kernel step
    order: "tuple[tuple[str, str], ...]"  # (kind, name) of the arguments of a launch, in the ABI's order
    algo: "str | None"  # the pinned algorithm of a library step
    workspace: "str | None"


@dataclass
class _BucketRT:
    values: "dict[str, int]"
    shapes: "dict[str, tuple[str, tuple[int, ...]]]"
    steps: "list[_StepRT]"


@dataclass
class _BoundCall:
    """The arguments of a call matched to the graph: the binding, the device scalars' tensors by name, the bucket, the device."""

    call: Any
    sources: "dict[str, Any]"
    values: "dict[str, int]"
    index: int


@dataclass
class _Call:
    """A call ready to be enqueued: every precondition checked."""

    rt: _BucketRT
    arena: _Arena
    checked: "list[Any]"  # per step, the checked tensors by name (their addresses)
    bases: "dict[str, int]"
    slots: "list[tuple[dict, Any, bool]]"  # (buffer, source, on_device) of every device scalar
    results: "list[Any]"  # the outputs, in the order the graph returns them


def _ordered(be: Backend, arena: _Arena, stream: int, capturing: bool, work: Any) -> None:
    """``work()`` (what enqueues the writes and launches of a run, or a replay) between the wait for the arena's order event and its record, on ``stream``,
    under the arena's lock (held by the caller).

    **Once anything was enqueued after the wait, the record is made**, whatever happens: a failure of a launch, a ``KeyboardInterrupt`` anywhere after the
    wait has returned, or a failure of the record itself. A record that fails is tried once more (an interrupt that landed inside the first is not there
    twice), and if it fails again the host waits for the stream (``cuStreamSynchronize``, outside a capture) before the error is raised and the lock
    released: then the work is done before the next run on the arena is enqueued, so it cannot overlap it (a capture whose record failed is broken, and its
    graph cannot be made). A wait that failed has enqueued nothing, and is not followed by a record: a record then would stand for this stream's work
    alone, and the next run would no longer wait for the one before."""
    event = arena.event
    if event is None:  # a plan with no arena shares no memory between its runs: nothing to order
        work()
        return
    waited = failed = False
    try:
        be.wait_event(event, stream, capturing)
        waited = True  # from here on, the record must follow (an interrupt before this line comes before anything but the wait is enqueued)
        failed = True
        work()
        failed = False
    finally:
        if waited:
            if failed:  # the error of the work is the one raised; the record is made all the same
                with contextlib.suppress(BaseException):
                    _record(be, event, stream, capturing)
            else:
                _record(be, event, stream, capturing)


def _record(be: Backend, event: Any, stream: int, capturing: bool) -> None:
    """The record of :func:`_ordered`: tried twice, then the stream is drained on the host (see there). A failure of the first try is raised once the
    record is made (or the stream drained): a ``KeyboardInterrupt`` is not lost."""
    try:
        be.record_event(event, stream, capturing)
        return
    except BaseException as first:  # noqa: BLE001 - a KeyboardInterrupt too: the record is what must not be lost
        error = first
    try:
        be.record_event(event, stream, capturing)
    except BaseException:  # noqa: BLE001
        if not capturing:
            with contextlib.suppress(Exception):
                be.synchronize_stream(stream)
    raise error


class PlanRunner:
    """Runs one plan (:class:`~tilefish._plan.Plan`) with binaries from a provider, on a backend. See the module docstring."""

    def __init__(self, plan: Plan, provider: Any, backend: "Backend | None" = None) -> None:
        self.plan = plan
        self.provider = provider
        self._backend = backend
        self._doc = plan._doc
        self._buffers = {b["name"]: b for b in self._doc["buffers"]}
        self._rts: "dict[tuple, _BucketRT]" = {}
        self._lock = threading.RLock()
        self.has_library = any(s["kind"] == "library" for s in self._doc["buckets"][0]["steps"])
        _runners.add(self)

    @property
    def backend(self) -> Backend:
        """The backend (torch and the driver, made when first needed: a runner can be made, and a plan loaded, on a machine without torch)."""
        if self._backend is None:
            self._backend = torch_backend()
        return self._backend

    # -- preparing: everything that may allocate or wait ------------------------------------------------------------------------

    def prepare(self, index: int, buckets: "Sequence[Mapping[str, int]] | None" = None) -> None:
        """Allocate the arena of device ``index``, build or load the binary of every step of the given bucket combinations (all of them by default),
        load the functions into the device and create the cuBLASLt handle. Everything a run does that cannot be captured in a CUDA graph is done here
        (and a run does it for itself the first time, outside a capture)."""
        be = self.backend
        if be.capturing(index):
            raise TilefishRuntimeError("a plan cannot be prepared while a CUDA graph is being captured: its arena, binaries and handles are made outside the capture")
        with _gate.shared(), self._lock:
            self._arena(index)
            for values in buckets if buckets is not None else self.plan.buckets():
                self._bucket(dict(values), index)
            be.warm(index, self.has_library)

    def _reset(self) -> None:
        """Forget the binaries this runner loaded (``clear_caches`` unloads every module)."""
        with self._lock:
            self._rts.clear()

    def arena_of(self, index: int) -> "_Arena | None":
        """The arena of this plan's layout on device ``index``, if it exists (it is shared by every runner of that layout, and kept until ``clear_caches``)."""
        return self.backend.arena_registry().get((self.plan.layout_key, index))

    def _arena(self, index: int) -> _Arena:
        arena = self.arena_of(index)
        if arena is not None:
            return arena
        be = self.backend
        if be.capturing(index):
            raise TilefishRuntimeError(
                f"the arena of the plan of graph '{self.plan.name}' has not been allocated on device {index} yet, and memory is not allocated while a CUDA "
                "graph is being captured (the graph would own it): run the plan once outside the capture, or call tilefish.runtime.prepare"
            )
        with _arena_lock:
            arena = self.arena_of(index)
            if arena is None:
                nbytes = self._doc["arena_bytes"]
                handle, ptr = be.alloc_arena(nbytes, index) if nbytes else (None, 0)
                if nbytes and ptr % _plan.ARENA_ALIGNMENT:
                    raise TilefishRuntimeError(f"the arena was allocated at {ptr:#x}, which is not aligned to {_plan.ARENA_ALIGNMENT} bytes")
                event = be.new_event(index) if nbytes else None  # a plan with no arena shares no memory between its runs: nothing to order
                arena = be.arena_registry()[(self.plan.layout_key, index)] = _Arena(handle, ptr, nbytes, threading.Lock(), event)
        return arena

    def _bucket(self, values: "Mapping[str, int]", index: int) -> _BucketRT:
        key = (tuple(sorted(values.items())), index)
        rt = self._rts.get(key)
        if rt is not None:
            return rt
        be = self.backend
        if be.capturing(index):
            raise TilefishRuntimeError(
                f"the plan of graph '{self.plan.name}' has not been prepared for {dict(values)} on device {index}, and its binaries are not built or loaded "
                "while a CUDA graph is being captured: run it once outside the capture (or call tilefish.runtime.prepare)"
            )
        with self._lock:
            rt = self._rts.get(key)
            if rt is None:
                rt = self._rts[key] = self._build(dict(values), index)
        return rt

    def _build(self, values: "dict[str, int]", index: int) -> _BucketRT:
        be = self.backend
        doc = self._doc
        steps_doc = next((b["steps"] for b in doc["buckets"] if b["values"] == values), None)
        if steps_doc is None:  # pragma: no cover - the caller checked the bucket
            raise TilefishPreconditionError(f"the plan has no bucket {values}", kind="bucket")
        shapes = _plan.buffer_shapes(doc, values)
        callees = self.plan._callees
        steps: "list[_StepRT]" = []
        for i, st in enumerate(steps_doc):
            callee = callees.get(st["entry"])
            where = f"graph '{self.plan.name}' at {values or 'its only size'}, step #{i} ({st['entry']})"
            if callee is None:
                raise TilefishManifestError(f"{where}: the graph calls no operator that has the binary '{st['entry']}'")
            bound = _plan.resolve_step(callee, st["args"], shapes, where)
            if bound.stem != st["entry"]:
                raise TilefishManifestError(f"{where}: the arguments of the step select the binary '{bound.stem}', not '{st['entry']}'")
            if callee.kind == "library":
                ws = st["workspace"]
                if shapes[ws][1][0] < bound.spec.workspace_bytes:
                    raise TilefishManifestError(f"{where}: the workspace '{ws}' has {shapes[ws][1][0]} bytes, but the entry needs {bound.spec.workspace_bytes}")
                built = self.provider.pin(bound.spec, index)
                be.check_gemm(bound.spec, built.algo, index)  # once, here: a launch repeats it, but a refusal must not come after another step has run (or inside a capture)
                steps.append(_StepRT(bound, None, (), built.algo, ws))
            else:
                built = self.provider.kernel(callee, bound.spec, index)
                if built.spec.block != bound.spec.block or built.num_warps != bound.spec.num_warps:
                    raise TilefishRuntimeError(f"the binary of '{bound.spec.stem}' disagrees with its spec about num_warps")
                order = tuple((a.kind, a.name) for a in bound.spec.args if a.kind in ("pointer", "scalar"))
                loaded = be.load(built, index)
                be.check_grid(loaded, bound.grid)
                steps.append(_StepRT(bound, loaded, order, None, None))
        be.warm(index, self.has_library)
        return _BucketRT(dict(values), shapes, steps)

    # -- running ----------------------------------------------------------------------------------------------------------------

    def run(self, args: "tuple[Any, ...]", kwargs: "dict[str, Any]", outputs: Any = None) -> Any:
        be = self.backend
        bound = self._bind(args, kwargs)
        index = bound.index
        with _gate.shared():
            capturing = be.capturing(index)
            call = self._call(bound, outputs, slots_in_graph=capturing)
            arena = call.arena
            with arena.lock, be.scope(index):
                stream = be.stream(index)
                if capturing:
                    self._check_one_stream_per_capture(arena, stream)
                _ordered(be, arena, stream, capturing, lambda: self._enqueue(call, stream, index, slots=True))
        return runtime._result(call.results)

    def _bind(self, args: "tuple[Any, ...]", kwargs: "dict[str, Any]") -> "_BoundCall":
        """Match the arguments to the graph's signature (a ``DevScalar`` passed as a number becomes a tensor of one element on the host), bind the
        dimensions and select the bucket, find the device. Touches no arena and no binary."""
        be = self.backend
        graph = self.plan.graph
        sig = graph.signature
        doc = self._doc
        try:
            bound = sig.call_signature.bind(*args, **kwargs)
        except TypeError as e:
            raise TilefishPreconditionError(f"{graph.name}: {e}; signature {sig}", kind="shape") from None
        bound.apply_defaults()
        scalar_params = [p for p in sig.params if p.kind == "tensor" and p.is_devscalar]
        sources: "dict[str, Any]" = {}
        for p in scalar_params:
            v = bound.arguments[p.name]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                bound.arguments[p.name] = v = be.devscalar_tensor(devscalar_number(p, v), p.type.dtype.name)  # type: ignore[union-attr]
            sources[p.name] = v
        call = bind_call(graph.name, sig, bound.args, bound.kwargs)
        check_dim_constraints(call.dims, buckets=False)
        by_name = dims_by_name(call.dims)
        values: "dict[str, int]" = {}
        for n, allowed in doc["dims"].items():
            if n not in by_name:
                raise TilefishPreconditionError(f"the dimension {n} of graph '{graph.name}' is not determined by the arguments", kind="bucket", arg=n)
            if by_name[n] not in allowed:
                raise TilefishPreconditionError(
                    f"dimension {n} is {by_name[n]}, which is not one of the buckets {allowed} of the plan of graph '{graph.name}'; a bucketed dimension "
                    "must equal a bucket exactly (there is no padding: pad or slice in the host)",
                    kind="bucket",
                    arg=n,
                )
            values[n] = by_name[n]
        tensors = {n: t for n, t in call.tensors.items() if n not in sources}
        infos = {n: i for n, i in call.infos.items() if n not in sources}
        device = runtime._device_of(infos, tensors)
        if getattr(device, "type", None) != "cuda":
            raise TilefishRuntimeError(f"jit and replay run on a CUDA device; the tensors are on {device}")
        index = be.device_index(device)
        arch = self.plan.arch
        if arch is not None and arch != f"sm_{be.capability(index)}":
            raise TilefishRuntimeError(f"the plan of graph '{graph.name}' was built for {arch}, but device {index} is sm_{be.capability(index)}")
        for name in sources:  # a device scalar may be on the host, or on this device: a copy from another device is not what a plan orders
            where = call.infos[name].device
            if where != "cpu" and be.device_index(where) != index:
                raise TilefishPreconditionError(f"the device scalar '{name}' is on {where}, but the other tensors of the call are on device {index}", kind="device", arg=name)
        return _BoundCall(call, sources, values, index)

    def _call(self, bound: "_BoundCall", outputs: Any, slots_in_graph: bool) -> "_Call":
        """The bucket's binaries and the arena (made the first time, outside a capture), the outputs (the caller's, or allocated), the address of every
        buffer, and **every precondition of every step**, all before anything is enqueued. ``slots_in_graph``: the writes of the device scalars are captured
        in a ``torch.cuda.graph``, so a host value is refused (a captured copy of it would freeze it). Called with the gate held shared."""
        be = self.backend
        doc = self._doc
        call, index = bound.call, bound.index
        rt = self._bucket(bound.values, index)
        arena = self._arena(index)
        # the outputs and the address of every buffer
        out_names = doc["outputs"]
        made: "list[Any]" = []
        given = self._given_outputs(outputs, out_names)
        bases: "dict[str, int]" = {}
        written: "list[str]" = []
        for b in doc["buffers"]:
            name, role = b["name"], b["role"]
            if role in ("input", "weight", "inout"):
                t = call.infos[name]
                if t.shape != rt.shapes[name][1]:  # pragma: no cover - the dimensions are bound from these very shapes
                    raise TilefishPreconditionError(f"argument '{name}' has shape {list(t.shape)}, but the plan has {list(rt.shapes[name][1])}", kind="shape", arg=name)
                bases[name] = t.ptr
                if role == "inout":
                    written.append(name)
            elif role == "output":
                shape = rt.shapes[name][1]
                position = out_names.index(name)
                if given is not None:
                    t_out = given[position]
                    self._check_output(name, t_out, b["dtype"], shape, index)
                else:
                    t_out = be.empty(shape, b["dtype"], index)
                made.append((position, t_out))
                bases[name] = int(t_out.data_ptr())
                written.append(name)
            else:  # in the arena
                bases[name] = arena.ptr + b["offset"]
        self._check_externals(bases, written, rt, arena)
        # every precondition of every step, before anything is launched
        checked = [st.bound.check(bases) for st in rt.steps]
        slots = []
        for b in doc["buffers"]:
            if b["role"] == "devscalar":
                source, on_device = be.devscalar_source(bound.sources[b["name"]], b["dtype"], index)
                if slots_in_graph and not on_device:
                    raise TilefishRuntimeError(
                        f"the device scalar '{b['name']}' was passed as a host value (an int, a number or a CPU tensor), which is written by a host copy that "
                        "cannot be captured in a CUDA graph (and a captured copy of it would freeze the value): pass a CUDA tensor of one element, which the "
                        "captured copy reads at every replay, and write it between replays (or capture with tilefish.capture, whose set_devscalar changes it)"
                    )
                slots.append((b, source, on_device))
        results = [t for _, t in sorted(made, key=lambda p: p[0])]
        return _Call(rt, arena, checked, bases, slots, results)

    def _enqueue(self, call: "_Call", stream: int, index: int, slots: bool) -> None:
        """Enqueue the device-scalar writes (``slots``) and the launches of the steps on ``stream``, the current stream (the caller orders them: :func:`_ordered`)."""
        be = self.backend
        if slots:
            for b, source, on_device in call.slots:
                be.write_devscalar(call.arena.handle, b["offset"], b["dtype"], source, on_device)
        for st, real in zip(call.rt.steps, call.checked):
            if st.algo is not None:
                names = [a for a in st.bound.names]
                a_ptr, b_ptr = real[names[0]].ptr, real[names[1]].ptr
                bias = real[names[2]].ptr if len(names) == 4 else None
                be.gemm(st.bound.spec, st.algo, a_ptr, b_ptr, bias, real[names[-1]].ptr, call.bases[st.workspace], stream, index)  # type: ignore[index]
            else:
                launch_args = [("pointer", real[n].ptr) if kind == "pointer" else ("scalar", st.bound.scalars[n]) for kind, n in st.order]
                be.launch(st.loaded, st.bound.grid, launch_args, stream)

    def _check_one_stream_per_capture(self, arena: _Arena, stream: int) -> None:
        """Refuse a second run of this arena in one ``torch.cuda.graph`` capture on another stream than the first (under the arena's lock, before
        anything is enqueued). Each run captures an external wait node on the order event and an external record node; the second run's wait node would
        not depend on the first run's record node inside the graph (a forked stream joins the capture without it), so every replay would run the two at
        once on the same intermediates. On one stream the graph orders them."""
        if arena.event is None:
            return
        capture = self.backend.capture_id(stream)
        if capture is None:  # pragma: no cover - the caller saw the stream capturing
            return
        if arena.captured is not None and arena.captured[0] == capture and arena.captured[1] != stream:
            raise TilefishRuntimeError(
                f"the plan of graph '{self.plan.name}' was already run in this CUDA graph capture on another stream: two runs of one plan in one capture must be on "
                "one stream (each run waits for the run before it through the arena's order event, and inside one graph a run on a forked stream would not wait "
                "for the other, so a replay would run both at once on the same arena). Run them on the same stream, or capture them in two graphs, or use "
                "tilefish.capture"
            )
        arena.captured = (capture, stream)

    def capture(self, externals: "tuple[Any, ...]", devscalars: "Mapping[str, Any]", outputs: Any, stream: Any) -> "CapturedPlan":
        """:func:`capture` of this runner's plan (see there)."""
        be = self.backend
        sig = self.plan.graph.signature
        scalar_names = [p.name for p in sig.params if p.kind == "tensor" and p.is_devscalar]
        others = [n for n in sig.call_signature.parameters if n not in scalar_names]
        if len(externals) > len(others):
            raise TilefishPreconditionError(
                f"tilefish.capture of graph '{self.plan.name}' takes at most {len(others)} externals ({', '.join(others)}): the device scalars "
                f"({', '.join(scalar_names) or 'none'}) are given by devscalars=, got {len(externals)}",
                kind="shape",
            )
        if not isinstance(devscalars, Mapping):
            raise TilefishRuntimeError(f"devscalars= is a mapping of the name of every device scalar of the graph to its value, got {devscalars!r}")
        unknown = sorted(set(devscalars) - set(scalar_names))
        missing = [n for n in scalar_names if n not in devscalars]
        if unknown or missing:
            raise TilefishPreconditionError(
                f"devscalars= gives the value of every device scalar of graph '{self.plan.name}' ({', '.join(scalar_names) or 'it has none'})"
                + (f"; {', '.join(unknown)} is not one" if unknown else "")
                + (f"; {', '.join(missing)} is missing" if missing else ""),
                kind="shape",
            )
        bound = self._bind((), {**dict(zip(others, externals)), **devscalars})
        index = bound.index
        with _gate.shared():
            if be.capturing(index):
                raise TilefishRuntimeError(
                    "tilefish.capture makes a CUDA graph of its own and is called outside a capture (the current stream is capturing one): to capture a plan in a "
                    "graph of yours, call tilefish.run inside torch.cuda.graph (see the docs for how its replays are ordered)"
                )
            call = self._call(bound, outputs, slots_in_graph=False)
            arena = call.arena
            epoch = _epoch[0]
            with arena.lock, be.scope(index):  # no other run of the plan is enqueued while the graph is captured
                graph = be.capture_graph(index, stream, lambda s: self._enqueue(call, s, index, slots=False))
        held = tuple(bound.call.tensors.values()) + tuple(call.results)
        return CapturedPlan(self, graph, call, index, epoch, held)

    def _given_outputs(self, outputs: Any, names: "Sequence[str]") -> "list[Any] | None":
        if outputs is None:
            return None
        items = [outputs] if not isinstance(outputs, (tuple, list)) else list(outputs)
        if len(items) != len(names):
            raise TilefishPreconditionError(
                f"outputs= must give the {len(names)} returned value(s) of graph '{self.plan.name}' ({', '.join(names)}), got {len(items)}", kind="shape"
            )
        return items

    def _check_output(self, name: str, t: Any, dtype: str, shape: "tuple[int, ...]", index: int) -> None:
        info = tensor_info(f"outputs[{name}]", t)
        if info.dtype != TORCH_DTYPE_NAMES[dtype] or info.shape != tuple(shape) or not info.contiguous:
            raise TilefishPreconditionError(
                f"the output for '{name}' must be a contiguous {dtype} tensor of shape {list(shape)}, got {info.dtype} {list(info.shape)}"
                f"{'' if info.contiguous else ' (not contiguous)'}",
                kind="shape",
                arg=name,
            )
        if self.backend.device_index(info.device) != index:
            raise TilefishPreconditionError(f"the output for '{name}' is on {info.device}, not on device {index}", kind="device", arg=name)

    def _check_externals(self, bases: "Mapping[str, int]", written: "Sequence[str]", rt: _BucketRT, arena: _Arena) -> None:
        """A tensor the plan writes (an output or an in-place tensor) must not overlap any other external: the steps are ordered, but the plan is not
        written for two names for one memory. No external may overlap the arena: it is the plan's own."""
        ranges = {}
        for b in self._doc["buffers"]:
            if b["role"] in ("input", "weight", "inout", "output"):
                dtype, shape = rt.shapes[b["name"]]
                ranges[b["name"]] = (bases[b["name"]], bases[b["name"]] + math.prod(shape) * _plan.ITEMSIZE[dtype])
                if arena.nbytes and ranges[b["name"]][0] < arena.ptr + arena.nbytes and arena.ptr < ranges[b["name"]][1]:
                    raise TilefishPreconditionError(
                        f"'{b['name']}' overlaps the arena of the plan (the memory the plan keeps its intermediates, device scalars and workspace in): a tensor of the "
                        "caller must not be in it",
                        kind="alias",
                        arg=b["name"],
                    )
        for w in written:
            lo, hi = ranges[w]
            for other, (olo, ohi) in ranges.items():
                if other != w and lo < ohi and olo < hi:
                    raise TilefishPreconditionError(
                        f"'{w}' is written by the plan but overlaps '{other}': the tensors of a plan must not alias when one of them is written",
                        kind="alias",
                        arg=w,
                    )


class CapturedPlan:
    """A plan captured in a CUDA graph by :func:`tilefish.capture`, for one bucket and one set of externals: replay it with :meth:`replay`.

    **Its replays are ordered with every other run of the plan, from any thread and on any stream**: :meth:`replay` takes the arena's lock, makes the stream
    wait for the arena's order event, writes the device scalars, launches the graph and records the event, as a run does (so the graph itself has no event
    node: the ordering is the replay's, which is why a ``CapturedPlan`` is replayed only through :meth:`replay`). It keeps alive what the graph holds the
    address of: the externals (the tensors given, and the outputs) and the arena. Write new contents into the externals between replays (in place), and set
    a device scalar with :meth:`set_devscalar`. An external written or read on another stream than the replay's is ordered by the caller, as for any torch op.

    After ``tilefish.runtime.clear_caches()`` (which unloads the binaries the graph launches) it cannot be replayed: capture again.
    """

    def __init__(self, runner: PlanRunner, graph: Any, call: _Call, index: int, epoch: int, externals: "tuple[Any, ...]") -> None:
        self._runner = runner
        self._graph = graph
        self._call = call
        self._arena = call.arena  # the memory the graph holds the address of: kept alive with the graph
        self._index = index
        self._epoch = epoch
        self._externals = externals  # the tensors the graph holds the addresses of: kept alive with the graph
        self._slots = [(b, _host_copy(s, on_device), on_device) for b, s, on_device in call.slots]  # (buffer, source, on_device) of every device scalar: written before every replay
        self.outputs = runtime._result(list(call.results))  # what the graph writes its returned values into (the caller's outputs=, or allocated by capture)
        self.bucket = dict(call.rt.values)

    def __repr__(self) -> str:
        return f"CapturedPlan(graph={self._runner.plan.name!r}, bucket={self.bucket}, devscalars={[b['name'] for b, _, _ in self._slots]})"

    @property
    def plan(self) -> Plan:
        """The plan the graph was captured from."""
        return self._runner.plan

    def replay(self, stream: Any = None) -> None:
        """Replay the graph on ``stream`` (a ``torch.cuda.Stream`` of the plan's device; default: the current stream).

        Under the arena's lock: the stream waits for the arena's order event (the run or replay of the plan issued before this one, whatever its stream and
        thread), the device scalars are written into their slots (from the values :meth:`set_devscalar` set, through pinned host memory, or from the CUDA
        tensor given), the graph is launched, and the event is recorded (also when a step of this fails after something was enqueued). Nothing waits on the
        host. Refused inside a capture, and after ``clear_caches``."""
        runner = self._runner
        be = runner.backend
        index = self._index
        with _gate.shared():
            if self._epoch != _epoch[0]:
                raise TilefishRuntimeError(
                    f"the captured plan of graph '{runner.plan.name}' cannot be replayed: tilefish.runtime.clear_caches() unloaded the binaries its graph launches "
                    "since it was captured; capture it again"
                )
            arena = self._arena
            with be.scope(index), be.use_stream(stream, index):
                if be.capturing(index):
                    raise TilefishRuntimeError(
                        "a CapturedPlan is not replayed while the stream captures a CUDA graph: its replay waits for and records the arena's order event, which "
                        "are not what a capture records (capture tilefish.run in torch.cuda.graph instead)"
                    )
                with arena.lock:
                    s = be.stream(index)
                    _ordered(be, arena, s, False, lambda: self._enqueue(s))

    def _enqueue(self, stream: int) -> None:
        be = self._runner.backend
        for b, source, on_device in self._slots:
            be.write_devscalar(self._arena.handle, b["offset"], b["dtype"], source, on_device)
        be.replay_graph(self._graph, stream)

    def set_devscalar(self, name: str, value: Any) -> None:
        """Set the value the device scalar ``name`` has in the next replays: a number (an integer that fits an integer dtype, as at a call), or a tensor of one
        element on the host or on the plan's device. Nothing is enqueued here: the next :meth:`replay` writes the slot, on its stream, after it has waited
        for the run before it (so a run or another graph of the plan in between does not change what this graph reads). A number or a CPU tensor is
        copied now, and written from pinned host memory (no synchronization of the host); a CUDA tensor is read by a stream-ordered copy at every replay."""
        runner = self._runner
        be = runner.backend
        position = next((i for i, (b, _, _) in enumerate(self._slots) if b["name"] == name), None)
        if position is None:
            names = [b["name"] for b, _, _ in self._slots]
            raise TilefishRuntimeError(f"'{name}' is not a device scalar of graph '{runner.plan.name}'; its device scalars are {names}")
        buffer = self._slots[position][0]
        param = next(p for p in runner.plan.graph.signature.params if p.name == name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            source = be.devscalar_tensor(devscalar_number(param, value), buffer["dtype"])
        else:
            source = value
            info = tensor_info(name, source)
            check_tensor(param, info)
            if info.device != "cpu" and be.device_index(info.device) != self._index:
                raise TilefishPreconditionError(f"the device scalar '{name}' is on {info.device}, but the plan runs on device {self._index}", kind="device", arg=name)
        source, on_device = be.devscalar_source(source, buffer["dtype"], self._index)
        with self._arena.lock:  # not while a replay of this plan enqueues its writes
            self._slots[position] = (buffer, _host_copy(source, on_device), on_device)


def _host_copy(source: Any, on_device: bool) -> Any:
    """A device scalar's value on the host, copied (a later write to the caller's CPU tensor does not change what a :class:`CapturedPlan` writes); a CUDA
    tensor is kept as it is (the replay's stream-ordered copy reads it, as a captured copy does)."""
    if on_device or not hasattr(source, "clone"):
        return source
    return source.clone()


def devscalar_number(p: Any, v: "int | float") -> "int | float":
    """The value a number passed for the ``DevScalar`` parameter ``p`` is written as: for an integer dtype it must be an integer that fits the dtype
    (``3.7``, ``nan`` or ``2**31`` for an ``i32`` is refused, not truncated or wrapped), in ``jit``, ``replay`` and ``reference`` alike."""
    dtype = p.type.dtype
    if not dtype.is_int:
        return v
    lo, hi = dtype.int_range()
    try:
        fits = v == int(v) and lo <= v <= hi
    except (ValueError, OverflowError):  # nan, inf
        fits = False
    if not fits:
        raise TilefishPreconditionError(f"the device scalar '{p.name}' is {v!r}, which is not an integer that fits {dtype} ({lo} to {hi})", kind="shape", arg=p.name)
    return int(v)


# -- tilefish.run of a graph ---------------------------------------------------------------------------------------------------------------


def _default_plan(graph: GraphOp) -> Plan:
    """``tilefish.plan(graph)``, made once and kept on the graph (an attribute, so that no global table holds the graph or the plan alive)."""
    p = graph.__dict__.get("_tf_default_plan")
    if p is None:
        p = make_plan(graph)
        graph.__dict__.setdefault("_tf_default_plan", p)
        p = graph.__dict__["_tf_default_plan"]
    return p


def _jit_runner(p: Plan) -> PlanRunner:
    """The runner of a plan that ``jit`` runs, kept on the plan (a plan and its runner are freed together; the arena is in the backend's registry)."""
    with _lock:
        r = p._jit_runner
        if r is None:
            r = p._jit_runner = PlanRunner(p, JitProvider())
    return r


def _directory_stamp(path: pathlib.Path) -> "tuple[str, tuple]":
    directory = path if path.is_dir() else path.parent
    try:
        stamp = tuple(sorted((e.name, e.stat().st_mtime_ns, e.stat().st_size) for e in os.scandir(directory) if e.is_file()))
    except OSError as e:
        raise TilefishManifestError(f"cannot read the plan directory {str(directory)!r}: {e}") from e
    return str(path.resolve()), stamp


def _replay_runner(graph: GraphOp, path: Any, pin: "str | None", trust_directory: bool) -> PlanRunner:
    _plan.check_anchor(pin, trust_directory)
    try:
        p = pathlib.Path(path)
    except TypeError:
        raise TilefishRuntimeError(f"plan= takes a directory (or its plan.json) for replay, got {path!r}") from None
    where, stamp = _directory_stamp(p)
    key = (where, id(graph))  # one runner per directory and graph: a directory that changed replaces it (its arena stays: a CUDA graph may hold it)
    with _lock:
        found = _loaded.get(key)
    if found is not None and found[0] == stamp and found[1].graph is graph:
        _, loaded, runner = found
        if pin is not None and loaded.digest != pin:
            raise TilefishManifestError(
                f"the plan in {str(loaded.directory)!r} has digest {loaded.digest}, but the pin is {pin}: it is not the plan that was pinned "
                "(it was changed, or it is another export)"
            )
        if (pin is not None) == runner.provider.pinned:  # a load with the other trust anchor is made again: the manifest's `pinned` differs
            return runner
    loaded = _plan.load_plan(p, graph, pin=pin, trust_directory=trust_directory)
    runner = PlanRunner(loaded, ReplayProvider(loaded.manifest, pinned=pin is not None))
    with _lock:
        _loaded[key] = (stamp, loaded, runner)
    return runner


def run_graph(
    graph: GraphOp,
    args: "tuple[Any, ...]",
    kwargs: "dict[str, Any]",
    *,
    mode: str,
    plan: Any = None,
    pin: "str | None" = None,
    trust_directory: bool = False,
    outputs: Any = None,
) -> Any:
    """``tilefish.run`` of a graph (see :func:`tilefish.run`)."""
    if mode == "reference":
        if plan is not None or outputs is not None:
            raise TilefishRuntimeError("plan= and outputs= are for jit and replay of a graph; reference mode evaluates the ops in torch and returns fresh tensors")
        return reference_run(graph, args, kwargs)
    if mode == "jit":
        if isinstance(plan, Plan):
            p = plan
            if p.graph is not graph:
                raise TilefishRuntimeError(f"plan= is the plan of graph '{p.name}', not of '{graph.name}'")
        elif plan is None:
            p = _default_plan(graph)
        else:
            raise TilefishRuntimeError("jit takes plan=<a Plan made by tilefish.plan(graph, ...)> or none; a directory is for mode='replay'")
        return _jit_runner(p).run(args, kwargs, outputs)
    return _replay_runner(graph, _plan_directory(plan), pin, trust_directory).run(args, kwargs, outputs)


def capture(
    graph: GraphOp,
    externals: "tuple[Any, ...]",
    *,
    mode: str = "jit",
    plan: Any = None,
    pin: "str | None" = None,
    trust_directory: bool = False,
    devscalars: "Mapping[str, Any] | None" = None,
    outputs: Any = None,
    stream: Any = None,
) -> CapturedPlan:
    """:func:`tilefish.capture`: the plan of ``graph`` (``jit`` or ``replay``, chosen as by :func:`tilefish.run`) captured in a CUDA graph of its own for the
    externals given, as a :class:`CapturedPlan` whose :meth:`~CapturedPlan.replay` is ordered with every other run of the plan.

    ``externals`` are the graph's arguments in the order of its signature, **without its device scalars**, which ``devscalars={name: value}`` gives (the values
    the first replay writes; :meth:`~CapturedPlan.set_devscalar` changes them). ``outputs=`` gives the tensors the graph writes its returned values into (by
    default they are allocated here, before the capture, and are ``CapturedPlan.outputs``). ``stream=`` is the stream the graph is captured on (default: a
    new one; the default stream cannot capture). The plan is prepared (arena, binaries, cuBLASLt handle) first, outside the capture; nothing is allocated
    inside it and nothing runs: the graph holds the launches of the steps only (no write of a device scalar and no event node: :meth:`~CapturedPlan.replay`
    does those)."""
    if mode not in ("jit", "replay"):
        raise TilefishRuntimeError(f"tilefish.capture: mode is 'jit' or 'replay', got {mode!r}")
    if mode == "jit":
        if pin is not None or trust_directory:
            raise TilefishRuntimeError("pin= and trust_directory= are for mode='replay' (they vouch for a plan directory); mode 'jit' runs no plan directory")
        if isinstance(plan, Plan):
            if plan.graph is not graph:
                raise TilefishRuntimeError(f"plan= is the plan of graph '{plan.name}', not of '{graph.name}'")
            runner = _jit_runner(plan)
        elif plan is None:
            runner = _jit_runner(_default_plan(graph))
        else:
            raise TilefishRuntimeError("jit takes plan=<a Plan made by tilefish.plan(graph, ...)> or none; a directory is for mode='replay'")
    else:
        if trust_directory is not True and trust_directory is not False:
            raise TilefishRuntimeError(f"trust_directory= is True or False, got {trust_directory!r}")
        runner = _replay_runner(graph, _plan_directory(plan), pin, trust_directory)
    return runner.capture(tuple(externals), {} if devscalars is None else devscalars, outputs, stream)


def _plan_directory(plan: Any) -> Any:
    """``plan=`` of a replay: a directory (or its ``plan.json``), or the :class:`~tilefish._plan.Plan` that ``export`` returned (it knows its directory)."""
    if isinstance(plan, Plan):
        if plan.directory is None:
            raise TilefishRuntimeError(f"the plan of graph '{plan.name}' was only made, not exported: replay runs an exported directory (tilefish.runtime.export)")
        return plan.directory
    if plan is None:
        raise TilefishRuntimeError(
            "replay runs the plan of an exported directory: pass plan=<directory> (written by tilefish.runtime.export(graph, directory)) and pin=<the plan's digest> "
            "or trust_directory=True"
        )
    return plan


def prepare(
    graph: GraphOp,
    *,
    mode: str = "jit",
    plan: Any = None,
    pin: "str | None" = None,
    trust_directory: bool = False,
    device: Any = None,
    buckets: "Sequence[Mapping[str, int]] | None" = None,
) -> None:
    """Do ahead of time everything that a run of the plan does for itself the first time and that cannot happen while a CUDA graph is being captured:
    allocate the arena on ``device`` (default: torch's current device), build (``jit``) or load and verify (``replay``) the binary of every step of
    ``buckets`` (default: all), load them into the device and create the cuBLASLt handle. Call it before ``torch.cuda.graph`` instead of running the plan once."""
    if mode not in ("jit", "replay"):
        raise TilefishRuntimeError(f"prepare: mode is 'jit' or 'replay', got {mode!r}")
    if mode == "jit":
        runner = _jit_runner(plan if isinstance(plan, Plan) else _default_plan(graph))
    else:
        runner = _replay_runner(graph, _plan_directory(plan), pin, trust_directory)
    torch = runtime._torch()
    index = torch.cuda.current_device() if device is None else runner.backend.device_index(device)
    runner.prepare(index, buckets)


# -- export --------------------------------------------------------------------------------------------------------------------------------


def export_graph(
    graph: GraphOp,
    directory: "str | os.PathLike[str]",
    *,
    arch: "int | str | None",
    overwrite: bool,
    buckets: Any,
    reuse: bool,
) -> Plan:
    """:func:`tilefish.runtime.export` of a graph: build the binary of every step (and pin every GEMM), write the manifest (merged into one that is
    already there) and ``plan.json`` next to it, read both back, and return the plan loaded from the directory, which has the digest to pin."""
    from . import _library
    from ._build import build, parse_arch
    from ._binary import build_binary

    p = make_plan(graph, buckets, reuse=reuse)
    if arch is None:
        torch = runtime._torch()
        cap = runtime._capability(torch, torch.cuda.current_device())
    else:
        cap = parse_arch(arch)
    built: list = []
    pinned: list = []
    for stem, callee in sorted(p.callees.items(), key=lambda kv: kv[0]):
        spec = callee.by_stem[stem]
        if callee.kind == "library":
            pinned.append(_library.build_library(spec, arch=cap))
        elif callee.extern.binary is not None:  # type: ignore[union-attr]
            built.append(build_binary(spec, callee.extern.binary, arch=cap))  # type: ignore[union-attr]
        else:
            built.append(build(spec, arch=cap, fn=callee.target.jit_function(spec)))
    return write_export(p, built, pinned, directory, cap, overwrite)


def write_export(p: Plan, built: "Sequence[Any]", pinned: "Sequence[Any]", directory: "str | os.PathLike[str]", cap: int, overwrite: bool) -> Plan:
    """The half of :func:`export_graph` that writes: the builds (``BuiltKernel`` and ``BuiltLibrary``, one per entry of the plan) go to the manifest in
    ``directory`` (merged into one that is there), ``plan.json`` is written next to it against the digest of the manifest as read back, and the plan is
    loaded back from the directory (verified) and returned."""
    from ._build import write_file, write_manifest_doc
    from ._manifest import MANIFEST_NAME, Manifest, manifest_digest

    directory = pathlib.Path(directory)
    plan_path = directory / PLAN_NAME
    if plan_path.exists() and not overwrite:
        previous = _existing_graph(plan_path)
        if previous is not None and previous != p.name:
            raise TilefishManifestError(
                f"{str(plan_path)!r} is the plan of graph '{previous}'; exporting '{p.name}' here would replace it (a directory holds one plan: use another "
                "directory, or pass overwrite=True)"
            )
    path, written = write_manifest_doc(
        built, directory / MANIFEST_NAME, cubins=True, merge=True, overwrite=overwrite, library=pinned
    )
    loaded = Manifest.load(path)
    expected = manifest_digest(written)
    if loaded.digest != expected:
        raise TilefishManifestError(
            f"the manifest in {str(path.parent)!r} has digest {loaded.digest} when it is read back, but the manifest that was written has digest "
            f"{expected}: the directory changed while it was being exported"
        )
    final = p.with_manifest(f"sm_{cap}", loaded.digest)
    _plan.validate_doc(final._doc)
    _plan.check_against_manifest(final._doc, loaded)
    write_file(plan_path, final.to_json().encode("utf-8"))
    return _plan.load_plan(plan_path, p.graph, pin=final.digest)


def _existing_graph(path: pathlib.Path) -> "str | None":
    import json

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        name = doc.get("graph") if isinstance(doc, dict) else None
        return name if isinstance(name, str) else None
    except (OSError, ValueError):
        return None


# -- reference mode ---------------------------------------------------------------------------------------------------------------------


def reference_run(graph: GraphOp, args: "tuple[Any, ...]", kwargs: "dict[str, Any]") -> Any:
    """``mode="reference"`` of a graph: the eager composition of the reference of each op, in torch, node by node.

    A view operation is torch's (``reshape``, ``permute``, ``select``, ``split``); a call to a lowered ``@tilefish.kernel`` is its IR interpreted
    (:mod:`tilefish._reference`); a pinned cuBLASLt GEMM is an f32 matmul rounded once to bf16; a hand-written extern is its ``reference=`` (a torch
    function given to ``@tilefish.extern``), and an extern that has none is refused. An in-place call on a view updates a copy of it: the IR says
    a chunk that is updated does not change its parent. Needs no CUDA: it runs where the tensors are. (A torch view of a tensor that an op then updates in place
    sees the update here, where the IR says a view keeps the contents it had: a plan refuses such a graph, so the two never disagree on a graph that can be planned.)
    """
    torch = runtime._torch()
    sig = graph.signature
    g = graph.graph
    try:
        bound = sig.call_signature.bind(*args, **kwargs)
    except TypeError as e:
        raise TilefishPreconditionError(f"{graph.name}: {e}; signature {sig}", kind="shape") from None
    bound.apply_defaults()
    anchor = None
    for p in sig.params:
        if p.kind == "tensor" and not p.is_devscalar and p.access != "out":
            anchor = bound.arguments[p.name]
            break
    for p in sig.params:  # a DevScalar passed as an int is a tensor of one element, on the device of the others
        if p.kind == "tensor" and p.is_devscalar:
            v = bound.arguments[p.name]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                dtype = getattr(torch, TORCH_DTYPE_NAMES[p.type.dtype.name].split(".")[1])  # type: ignore[union-attr]
                bound.arguments[p.name] = torch.tensor(devscalar_number(p, v), dtype=dtype, device=getattr(anchor, "device", None))  # 3.7 for an i32 is refused, not truncated
    call = bind_call(graph.name, sig, bound.args, bound.kwargs)
    check_dim_constraints(call.dims, buckets=False)
    dims = call.dims
    env: "dict[Value, Any]" = {}
    views: "set[Value]" = set()
    for v in g.inputs:
        if v.access != "out":
            env[v] = call.tensors[v.name]

    def size(e: Any) -> int:
        out = eval_dim(e, dims)
        if out is None:
            raise TilefishRuntimeError(f"graph '{graph.name}': a size {e} in the graph is not determined by the arguments")
        return out

    with torch.no_grad():
        for index, n in enumerate(g.nodes):
            kind = n.attrs.get("kind")
            if kind is None:
                x = env[n.inputs[0]]  # type: ignore[index]
                if n.op == "reshape":
                    results = [x.reshape([size(d) for d in n.attrs["shape"]])]
                elif n.op == "permute":
                    results = [x.permute(*n.attrs["axes"])]
                elif n.op == "select":
                    results = [x.select(n.attrs["axis"], n.attrs["index"])]
                elif n.op == "split":
                    results = list(torch.split(x, [size(s) for s in n.attrs["sizes"]], dim=n.attrs["axis"]))
                else:
                    raise TilefishRuntimeError(
                        f"reference mode of graph '{graph.name}' evaluates the calls to declared operators and the views, not the op '{n.op}' (node #{index}), "
                        "which a plan cannot launch either: put arithmetic in a @tilefish.kernel"
                    )
                views.update(n.outputs)
                for o, r in zip(n.outputs, results):
                    env[o] = r
                continue
            if n.callee is None or kind not in ("kernel", "extern", "library"):
                raise TilefishRuntimeError(f"reference mode of graph '{graph.name}' does not evaluate a {kind!r} call (node #{index}, '{n.op}')")
            op = n.callee
            op_sig = op.signature
            ins = {p.name: env[v] for p, v in zip(op_sig.tensor_inputs, n.inputs)}
            inout = [p for p in op_sig.tensor_inputs if p.access == "inout"]
            for p, v in zip(op_sig.tensor_inputs, n.inputs):
                if p.access == "inout" and v in views:
                    ins[p.name] = ins[p.name].clone()  # the IR: updating a chunk or a view does not change its parent
            if isinstance(op, LibraryOp):
                from . import _library

                if op.impl is None:
                    raise TilefishRuntimeError(f"reference mode needs the library op '{op.name}' to have an impl (a pinned GEMM): it has only a signature")
                device = next(iter(ins.values())).device
                produced = [_library.reference_gemm(ins).to(device)]
            elif isinstance(op, KernelOp):
                given = n.attrs.get("consts") or {}
                changed = {p.name: given[p.name] for p in op_sig.consts if p.name in given and given[p.name] is not p.default}  # refused when it differs from what is compiled in
                res = runtime._run_reference(runtime._target(op), (), {**ins, **changed})
                produced = [] if res is None else (list(res) if isinstance(res, tuple) else [res])
                produced += [ins[p.name] for p in inout]
            elif isinstance(op, ExternOp):
                if op.reference is None:
                    raise TilefishRuntimeError(
                        f"reference mode of graph '{graph.name}' needs a reference for the extern op '{op.name}': it wraps a hand-written kernel, which has no IR to "
                        "interpret. Give @tilefish.extern a reference=<a torch function of the op's tensor inputs> that returns its Out tensors"
                    )
                res = op.reference(**ins)
                produced = [] if res is None else (list(res) if isinstance(res, (tuple, list)) else [res])
                if len(produced) != len(op_sig.tensor_outputs):
                    raise TilefishRuntimeError(
                        f"the reference of the extern op '{op.name}' returned {len(produced)} tensor(s), but the op has {len(op_sig.tensor_outputs)} Out parameter(s)"
                    )
                produced += [ins[p.name] for p in inout]
            else:  # pragma: no cover - callee_of refuses the rest
                raise TilefishRuntimeError(f"cannot evaluate the op '{op.name}'")
            if len(produced) != len(n.outputs):
                raise TilefishRuntimeError(f"the reference of '{op.name}' made {len(produced)} tensor(s) for {len(n.outputs)} outputs")
            for o, r in zip(n.outputs, produced):
                env[o] = r
        results = []
        for v in g.outputs:
            r = env[v]
            results.append(r.clone() if (v.is_param or v in views) else r)
    return runtime._result(results)
