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
enqueueing, not the device); runs on one stream are ordered by the stream as before, so **a run on the stream of the event's most recent record skips the wait**
(one ``cuEventRecord`` per run while the runs stay on one stream, one ``cuStreamWaitEvent`` more for a run on another stream; never inside a capture, on the
per-thread default stream, or on an arena a ``torch.cuda.graph`` captured; the event is first recorded where the arena was allocated). Nothing waits on the host. (Runs of one plan do not overlap on the device,
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
must order them itself (one thread, or a lock of its own around all of them), replay through :func:`replay_graph` (which takes the arena's lock), or use
:func:`capture`, whose replay takes the arena's lock. The second thread that uses an arena such a graph holds gets a ``TilefishOrderWarning``
(:func:`_note_thread`).

**An arena shared by several plans.** :func:`share_arena` (``tilefish.runtime.share_arena([g1, g2, g3], ...)``) runs the plans of a set on one arena of
the largest plan's size, each plan's buffers at their own offsets in it (:class:`SharedArena`). Its runners hold one :class:`_Arena` between them, so
everything above holds for the executions of all of them as for the runs of one plan: one lock and one order event, so a run, a capture, the replay of a
:class:`CapturedPlan` and a ``run_step`` of plan A are ordered with those of plan B, whatever their threads and streams. Nothing in the plans' own regions
outlives a run; a **carried buffer** (``@tilefish.graph(carried=("h",))``) lives in a region of the arena after them, and the arena notes what each holds in
the order of the executions (:func:`_check_carried`). Executions that must run concurrently take an arena each (``share_arena(..., name=...)``).
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import json
import os
import pathlib
import threading
import time
import warnings
import weakref
import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import _plan, _probe, runtime
from ._build import check_launch_config
from ._launch import (
    TORCH_DTYPE_NAMES,
    bind_call,
    bind_infos,
    check_dim_constraints,
    check_tensor,
    dims_by_name,
    eval_dim,
    solve_dims,
    tensor_info,
)
from .decorators import ExternOp, GraphOp, KernelOp, LibraryOp
from .dims import DimExpr
from .signature import MISSING
from .errors import TilefishManifestError, TilefishOrderWarning, TilefishPreconditionError, TilefishRuntimeError
from ._plan import PLAN_NAME, Bound, Plan, plan as make_plan
from .trace import Value

_lock = threading.Lock()
_arena_lock = threading.Lock()
_loaded: "dict[tuple, tuple[tuple, Plan, PlanRunner]]" = {}  # (directory, graph) -> (stamp of its files, loaded plan, its runner), until a file in it changes
_runners: "weakref.WeakSet[PlanRunner]" = weakref.WeakSet()  # every runner alive, so that clear_caches can reset them (a runner is kept by its plan or by _loaded, never by a global)
_torch_backend: "list[TorchBackend]" = []
_epoch = [0]  # how many times clear_caches has run: a CapturedPlan is replayed only in the epoch it was captured in (its binaries are unloaded after)
_host_clock = time.perf_counter  # tilefish.profile_plan(method="events"): the host's time to enqueue the steps (a test gives its own clock)
_HOLD_TRIES = 12  # ... and how often a run is made again with a longer spin before it gives up
_HOLD_MAX_US = 2e6  # (a spin of more than 2 s: the host cannot keep up with the device, or the driver's queue is full)


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

    def current_device(self) -> int:
        """The index of torch's current CUDA device: where a call that gives no tensor runs (a plan whose parameters are all carried)."""
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

    def new_timing_event(self, index: int) -> Any:
        """An event that records a time (``tilefish.profile_plan``; recorded with :meth:`record_event`, read with :meth:`elapsed_ms`)."""
        raise NotImplementedError

    def elapsed_ms(self, start: Any, end: Any, index: int) -> float:
        """Milliseconds between the most recent records of two timing events (both done)."""
        raise NotImplementedError

    def hold(self, stream: int, microseconds: float, index: int) -> None:
        """Keep ``stream`` busy for about ``microseconds`` on the device (a spinning kernel), so that what the host enqueues after it is queued before
        the device reaches it (``tilefish.profile_plan`` without a graph: the steps then run back to back, not as fast as the host launches them)."""
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

    def launch(self, loaded: Any, grid: "Sequence[int]", args: "Sequence[tuple[str, Any]]", stream: int, scratch: "int | None" = None) -> None:
        raise NotImplementedError

    def encode_tensor_map(self, arguments: Any) -> bytes:
        """The 128 bytes of the ``CUtensorMap`` of a TMA descriptor argument (``cuTensorMapEncodeTiled`` of a
        :class:`~tilefish._tensormap.EncodeArguments`)."""
        raise NotImplementedError

    def copy(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        """Copy ``nbytes`` bytes of device memory from ``src`` to ``dst`` on ``stream`` (a copy step of the plan; captured as a memcpy node)."""
        raise NotImplementedError

    def memset(self, dst: int, nbytes: int, stream: int) -> None:
        """Set ``nbytes`` bytes of device memory at ``dst`` to zero on ``stream`` (a memset step of the plan; captured as a memset node)."""
        raise NotImplementedError

    def gemm(self, spec: Any, algo: str, a: int, b: int, bias: "int | None", out: int, workspace: int, stream: int, index: int) -> None:
        raise NotImplementedError

    def conv(self, built: Any, x: int, w: int, out: int, workspace: int, stream: int, index: int) -> None:
        """Run a pinned cuDNN convolution (a ``BuiltConv``) on device addresses, the workspace in the arena."""
        raise NotImplementedError

    def warm(self, index: int, library: bool) -> None:
        raise NotImplementedError

    def check_grid(self, loaded: Any, grid: "Sequence[int]") -> None:
        """Refuse a grid the device cannot launch (it depends on the bucket only, so a runner checks it when it prepares the bucket, not at a launch)."""

    def check_gemm(self, spec: Any, algo: str, index: int) -> None:
        """Refuse a pinned algorithm that is not valid for the problem on this device, or needs more workspace than recorded (the same: once per bucket)."""

    def check_conv(self, built: Any, index: int) -> None:
        """Refuse a cuDNN pin that is not valid for its problem on this device (once per bucket, outside any capture: it also makes the plan)."""


class TorchBackend(Backend):
    def __init__(self) -> None:
        self.torch = runtime._torch()

    def device_index(self, device: Any) -> int:
        if isinstance(device, int) and not isinstance(device, bool):
            return device
        return runtime._device_index(self.torch, str(device))

    def current_device(self) -> int:
        return int(self.torch.cuda.current_device())

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

    def new_timing_event(self, index: int) -> Any:
        from . import _cuda

        with self.torch.cuda.device(index):
            return _cuda.new_timing_event()

    def elapsed_ms(self, start: Any, end: Any, index: int) -> float:
        from . import _cuda

        with self.torch.cuda.device(index):
            return _cuda.elapsed_ms(start, end)

    def hold(self, stream: int, microseconds: float, index: int) -> None:
        from . import _cuda

        rates = self.__dict__.setdefault("_clock_khz", {})
        rate = rates.get(index) or rates.setdefault(index, _cuda.clock_khz(index))  # cycles per millisecond
        self.torch.cuda._sleep(max(1, int(microseconds * rate / 1000)))  # on torch's current stream, which is `stream` (the caller made it current)

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

    def launch(
        self, loaded: Any, grid: "Sequence[int]", args: "Sequence[tuple[str, Any]]", stream: int, scratch: "int | None" = None, probes: "int | None" = None
    ) -> None:
        if probes is None:
            loaded.launch(grid, args, stream, scratch=scratch)
        else:
            loaded.launch(grid, args, stream, scratch=scratch, probes=probes)

    def encode_tensor_map(self, arguments: Any) -> bytes:
        from . import _cuda

        return _cuda.encode_tensor_map(arguments)

    def copy(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        from . import _cuda

        _cuda.copy_async(dst, src, nbytes, stream)

    def memset(self, dst: int, nbytes: int, stream: int) -> None:
        from . import _cuda

        _cuda.memset_async(dst, nbytes, stream)

    def gemm(self, spec: Any, algo: str, a: int, b: int, bias: "int | None", out: int, workspace: int, stream: int, index: int) -> None:
        from . import _library

        _library.run_gemm_ptrs(spec, algo, a, b, bias, out, stream, index, workspace_ptr=workspace)

    def conv(self, built: Any, x: int, w: int, out: int, workspace: int, stream: int, index: int) -> None:
        from . import _conv

        _conv.run_conv_ptrs(built, x, w, out, stream, index, workspace_ptr=workspace)

    def warm(self, index: int, library: bool) -> None:
        if library:
            from . import _cublaslt

            _cublaslt.warm(index)

    def check_grid(self, loaded: Any, grid: "Sequence[int]") -> None:
        dims = tuple(grid)
        blocks = loaded.launch_grid(dims)  # (a persistent grid's programs on the SMs it was loaded for, times a cluster)
        for axis, (g, limit) in enumerate(zip(blocks, loaded.grid_limits)):
            if not 1 <= g <= limit:
                shown = dims if blocks == dims + (1,) * (3 - len(dims)) else f"{dims} ({blocks} blocks)"
                raise TilefishRuntimeError(
                    f"'{loaded.built.stem}': the grid {shown} has {g} blocks along {'XYZ'[axis]}, but this device takes 1 to {limit} there"
                )

    def check_gemm(self, spec: Any, algo: str, index: int) -> None:
        from . import _cublaslt

        _cublaslt.check_algo(
            spec.m, spec.n, spec.k, spec.bias, algo, workspace_bytes=spec.workspace_bytes, device_index=index, form=spec.form,
            sm_count_target=spec.sm_count_target, **spec.opt_ins,
        )  # fmt: skip

    def check_conv(self, built: Any, index: int) -> None:
        from . import _conv

        _conv.check_built(built, index)


# -- where a step's binary comes from ----------------------------------------------------------------------------------------------------


class JitProvider:
    """The binaries ``jit`` builds: the current source through its compile spec, in memory (the instrumented variant with ``instrument``); a GEMM
    pinned in memory the way ``export`` pins it."""

    def __init__(self, instrument: "Any" = None) -> None:
        self.instrument = instrument

    def kernel(self, callee: _plan.Callee, spec: Any, index: int) -> Any:
        return runtime.jit_built(callee.target, spec, index, self.instrument)

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

    ``last`` is the stream of the event's most recent record, when tilefish made it outside a capture (the record after the arena's allocation, on the stream
    it was allocated on, or the record of a run or a replay); ``None`` when that is not known. A run on that same stream skips the wait (:func:`_ordered`):
    stream order already puts it after the work the record stands for. Once ``caller_graphs`` is set (below), the replays of
    such a graph record the event on streams and at times tilefish does not see, so from then on every run waits.

    ``carried`` is where the carried buffers of the arena's plans are (section 15.2 of the plan schema: after the plans' own regions), and ``shapes`` what
    each holds, in the order the executions on the arena were issued (under ``lock``, which is the order they run in on the device): the shape at the bucket
    of the execution that wrote it last, or ``None`` when no execution wrote it (or one that may have failed). An execution that reads a carried buffer
    before it writes it is refused unless it holds the shape that execution reads.

    ``caller_graphs`` says a run on this arena was captured in a CUDA graph of the caller's (``torch.cuda.graph``), whose replays take no lock;
    ``threads`` are the threads that ran, captured, replayed or ``run_step``-ed on the arena; ``warned`` that the warning of :func:`_note_thread` was given;
    ``ordered`` that a caller's graph was replayed through :func:`replay_graph`, which takes the lock.
    """

    handle: Any
    ptr: int
    nbytes: int
    lock: threading.Lock
    event: Any = None
    captured: "tuple[int, int] | None" = None
    last: "int | None" = None
    carried: "_plan.CarriedLayout | None" = None
    shapes: "dict[str, tuple[int, ...] | None]" = dataclasses.field(default_factory=dict)
    caller_graphs: bool = False
    threads: "set[int]" = dataclasses.field(default_factory=set)
    warned: bool = False
    ordered: bool = False


@dataclass
class _CopyBound:
    """A copy step at one bucket: ``nbytes`` bytes from ``src`` (byte ``src_offset`` of that buffer) to ``dst`` (byte ``dst_offset``)."""

    src: str
    src_offset: int
    dst: str
    dst_offset: int
    nbytes: int

    @classmethod
    def of(cls, args: "Sequence[Mapping[str, Any]]", shapes: "Mapping[str, tuple[str, tuple[int, ...]]]") -> "_CopyBound":
        (src_shape, _, src_off), (dst_shape, _, dst_off) = (_plan.copy_view(a, shapes) for a in args)
        itemsize = _plan.ITEMSIZE[shapes[args[0]["binding"]][0]]
        return cls(args[0]["binding"], src_off * itemsize, args[1]["binding"], dst_off * itemsize, math.prod(src_shape) * itemsize)

    def check(self, bases: "Mapping[str, int]") -> "tuple[int, int, int]":
        """``(src, dst, nbytes)`` with the real addresses: the two runs of bytes must not overlap (externals that overlap are refused before this)."""
        src, dst, n = bases[self.src] + self.src_offset, bases[self.dst] + self.dst_offset, self.nbytes
        if src < dst + n and dst < src + n:
            raise TilefishPreconditionError(f"the copy from '{self.src}' to '{self.dst}' overlaps itself", kind="alias", arg=self.dst)
        return src, dst, n


@dataclass
class _MemsetBound:
    """A memset step at one bucket: ``nbytes`` bytes of ``dst`` (from byte ``dst_offset`` of that buffer) set to zero."""

    dst: str
    dst_offset: int
    nbytes: int

    @classmethod
    def of(cls, args: "Sequence[Mapping[str, Any]]", shapes: "Mapping[str, tuple[str, tuple[int, ...]]]") -> "_MemsetBound":
        shape, _, offset = _plan.copy_view(args[0], shapes)
        itemsize = _plan.ITEMSIZE[shapes[args[0]["binding"]][0]]
        return cls(args[0]["binding"], offset * itemsize, math.prod(shape) * itemsize)

    def check(self, bases: "Mapping[str, int]") -> "tuple[int, int]":
        """``(dst, nbytes)`` with the real address."""
        return bases[self.dst] + self.dst_offset, self.nbytes


@dataclass
class _Tap:
    """A debug tap of a run: the buffer ``buffer`` (``nbytes`` bytes, at the bucket) copied into the caller's tensor at ``dst`` after step ``after``."""

    buffer: str
    after: int
    tensor: Any
    dst: int
    nbytes: int


def _is_conv_spec(spec: Any) -> bool:
    from . import _conv

    return isinstance(spec, _conv.ConvSpec)


@dataclass(frozen=True)
class _ProbeStep:
    """Where an instrumented step's records go (``docs/instrument.md``): ``offset`` bytes into the bucket's probe region, ``nbytes`` of them, for the
    CUDA grid of its launch at the bucket; ``built`` is its binary (whose ``instrument`` says the capacity and the probes)."""

    offset: int
    nbytes: int
    cuda_grid: "tuple[int, int, int]"
    built: Any


@dataclass
class _StepRT:
    bound: "Bound | _CopyBound | _MemsetBound"
    loaded: Any  # the loaded kernel of a kernel step
    order: "tuple[tuple[str, str], ...]"  # (kind, name) of the arguments of a launch, in the ABI's order
    algo: Any  # the pinned algorithm of a GEMM step (its hex), the BuiltConv of a convolution step; None for the other steps
    workspace: "str | None"
    probe: "_ProbeStep | None" = None  # an instrumented binary of an instrumented runner: where its records go


@dataclass
class _BucketRT:
    values: "dict[str, int]"
    shapes: "dict[str, tuple[str, tuple[int, ...]]]"
    steps: "list[_StepRT]"
    # the carried buffers the steps read before any of them writes them (which an earlier execution on the arena must have written, in this shape),
    # and the ones they write (which hold this bucket's shape after the execution)
    carried_reads: "tuple[str, ...]" = ()
    carried_writes: "tuple[str, ...]" = ()
    # an instrumented runner: the bytes of the probe region of the bucket (every instrumented step's records, at 256-byte aligned offsets) and the
    # region itself, ``(handle, address)``, allocated with the bucket's binaries (outside any capture, like the arena); ``None`` when no step records
    probe_bytes: int = 0
    probe_region: "tuple[Any, int] | None" = None


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
    taps: "list[_Tap]" = dataclasses.field(default_factory=list)  # the copies of buffers the caller asked for, each after a step (taps=)
    # per step, its TMA descriptors by argument name (a TensorMapArg each), encoded from the checked tensors of this call: a run launches with them, and a
    # captured graph holds their bytes (a descriptor is a function of the addresses, shapes and strides the call binds, which a graph holds too)
    descs: "list[dict[str, Any]]" = dataclasses.field(default_factory=list)
    # probes= of the call: per step the address of its records (None: a step that records nothing), and the tensor they are in
    probes: "list[int | None]" = dataclasses.field(default_factory=list)
    probe_tensor: Any = None


def _ordered(be: Backend, arena: _Arena, stream: int, capturing: bool, work: Any) -> None:
    """``work()`` (what enqueues the writes and launches of a run, or a replay) between the wait for the arena's order event and its record, on ``stream``,
    under the arena's lock (held by the caller).

    **Once anything was enqueued after the wait, the record is made**, whatever happens: a failure of a launch, a ``KeyboardInterrupt`` anywhere after the
    wait has returned, or a failure of the record itself. A record that fails is tried once more (an interrupt that landed inside the first is not there
    twice), and if it fails again the host waits for the stream (``cuStreamSynchronize``, outside a capture) before the error is raised and the lock
    released: then the work is done before the next run on the arena is enqueued, so it cannot overlap it (a capture whose record failed is broken, and its
    graph cannot be made). A wait that failed has enqueued nothing, and is not followed by a record: a record then would stand for this stream's work
    alone, and the next run would no longer wait for the one before.

    **The wait is skipped when the event's most recent record was made on this same stream** by tilefish, outside a capture (``arena.last``): everything
    enqueued on a stream runs after what was enqueued on it before, so the run already comes after the work that record stands for, and the wait would
    add nothing but host time. The record is always made, so a later run on another stream waits for this one; and since every run records after it
    has waited (or after the record before it on its own stream), the most recent record still stands for every run before it. The wait is never skipped
    inside a capture (the graph must hold the wait node), for the per-thread default stream (one handle, another stream in each thread), or on an arena
    a ``torch.cuda.graph`` captured (``arena.caller_graphs``: the graph's replays record the event where tilefish does not see them). A stream is told by its
    handle: the driver releases a destroyed stream (whose handle a new stream could get) only once its work is done, so a new stream at the handle of
    the last record's stream comes after that work too."""
    event = arena.event
    if event is None:  # a plan with no arena shares no memory between its runs: nothing to order
        work()
        return
    if capturing:
        arena.caller_graphs = True
    skip = not capturing and not arena.caller_graphs and arena.last is not None and arena.last == stream and stream != _PER_THREAD_STREAM
    waited = failed = False
    try:
        if not skip:
            be.wait_event(event, stream, capturing)
        waited = True  # from here on, the record must follow (an interrupt before this line comes before anything but the wait is enqueued)
        failed = True
        work()
        failed = False
    finally:
        if waited:
            if failed:  # the error of the work is the one raised; the record is made all the same
                with contextlib.suppress(BaseException):
                    _record(be, arena, stream, capturing)
            else:
                _record(be, arena, stream, capturing)


# The handle of the per-thread default stream (CU_STREAM_PER_THREAD): one handle that names another stream in each thread.
_PER_THREAD_STREAM = 0x2


def _record(be: Backend, arena: _Arena, stream: int, capturing: bool) -> None:
    """The record of :func:`_ordered`: tried twice, then the stream is drained on the host (see there). A failure of the first try is raised once the
    record is made (or the stream drained): a ``KeyboardInterrupt`` is not lost. Once a record is made, ``arena.last`` is its stream (``None`` for a record
    node of a capture, or on the per-thread default stream); when neither try made one, ``last`` is forgotten, and the next run waits."""
    event = arena.event
    try:
        be.record_event(event, stream, capturing)
        arena.last = None if capturing or stream == _PER_THREAD_STREAM else stream
        return
    except BaseException as first:  # noqa: BLE001 - a KeyboardInterrupt too: the record is what must not be lost
        error = first
    try:
        be.record_event(event, stream, capturing)
        arena.last = None if capturing or stream == _PER_THREAD_STREAM else stream
    except BaseException:  # noqa: BLE001
        arena.last = None
        if not capturing:
            with contextlib.suppress(Exception):
                be.synchronize_stream(stream)
    raise error


def carried_uses(plan: Plan) -> "tuple[tuple[str, ...], tuple[str, ...]]":
    """``(reads, writes)`` of a plan's carried buffers (section 15.2): the ones whose first use in the steps is a read (a pointer of the binary that is
    ``in`` or ``inout``, an operand of a GEMM, the source of a copy), which an earlier execution on the arena must have written, and the ones a step
    writes. The same in every bucket (the bindings are, and so is the ABI of an operator); read from the document and the operators, so it needs no
    binary."""
    carried = set(plan.carried)
    if not carried:
        return (), ()
    reads: "list[str]" = []
    writes: "list[str]" = []
    seen: "set[str]" = set()
    for st in plan._doc["buckets"][0]["steps"]:
        names = [a["name"] for a in st["args"]]
        if st["kind"] == "copy":
            access = {"src": "in", "dst": "out"}
        elif st["kind"] == "memset":
            access = {"dst": "out"}
        elif st["kind"] == "library":
            access = {n: "out" if n == "out" else "in" for n in names}
        else:
            extern = plan._callees[st["entry"]].variants()[0].extern
            assert extern is not None
            access = {p.name: p.access for p in extern.signature.params if p.kind == "tensor"}
        for a in st["args"]:
            name, how = a["binding"], access.get(a["name"], "in")
            if name not in carried:
                continue
            if name not in seen and how in ("in", "inout"):
                reads.append(name)
            seen.add(name)
            if how in ("out", "inout") and name not in writes:
                writes.append(name)
    return tuple(reads), tuple(writes)


def _check_carried(arena: _Arena, rt: "_BucketRT", graph: str) -> None:
    """Refuse an execution (under the arena's lock, before anything is enqueued) that reads a carried buffer the executions issued before it on the
    arena have not left in the shape it reads (none wrote it, or one wrote another shape, or one that wrote it failed)."""
    for name in rt.carried_reads:
        want = rt.shapes[name][1]
        have = arena.shapes.get(name)
        if have != want:
            why = (
                "no execution on this arena has written it (run a plan that writes it first: one that stores into it, zeroes it or updates it in place)"
                if have is None
                else f"the execution on this arena that wrote it last left the shape {list(have)}"
            )
            raise TilefishPreconditionError(
                f"the plan of graph '{graph}' reads the carried buffer '{name}' with the shape {list(want)}, but {why}", kind="shape", arg=name
            )


def _carried_written(arena: _Arena, rt: "_BucketRT", done: bool) -> None:
    """After an execution was enqueued (``done``) or failed: what the carried buffers it writes hold now (its shapes; unknown after a failure)."""
    for name in rt.carried_writes:
        arena.shapes[name] = rt.shapes[name][1] if done else None


def _note_thread(arena: _Arena, graph: str, caller_graph: bool = False) -> None:
    """Note the thread that uses the arena (under its lock), and warn once when an arena that a CUDA graph of the caller's holds is used by a second
    thread: a replay of such a graph takes no lock of tilefish, so it is not ordered with an execution another thread has between its wait for the
    order event and its record (docs/plan-schema-1.md, section 10, item 7). ``caller_graph``: this use is the capture of such a graph."""
    arena.threads.add(threading.get_ident())
    arena.caller_graphs = arena.caller_graphs or caller_graph
    if arena.caller_graphs and len(arena.threads) > 1 and not arena.warned and not arena.ordered:
        # noted once the warning is given: under simplefilter("error", TilefishOrderWarning) it raises (before anything is enqueued) at every such use
        warnings.warn(
            f"the arena of the plan of graph '{graph}' is used by a CUDA graph you captured (tilefish.run inside torch.cuda.graph) and from more than "
            "one thread: the replay of such a graph takes no lock of tilefish, so it can overlap an execution of the arena on another thread (wrong "
            "bytes, no error). Capture with tilefish.capture (its replays are ordered), replay your graphs with tilefish.runtime.replay_graph(graph), "
            "or order them yourself (docs/plan-schema-1.md, section 10, item 7)",
            TilefishOrderWarning,
            stacklevel=2,
        )
        arena.warned = True


class PlanRunner:
    """Runs one plan (:class:`~tilefish._plan.Plan`) with binaries from a provider, on a backend. See the module docstring.

    ``arena`` is ``(key, layout)`` of the arena the runner uses when it is not the plan's own (one per plan layout and device): the arena of a
    :class:`SharedArena`, which several plans use one at a time, and where its carried buffers are (:class:`~tilefish._plan.CarriedLayout`)."""

    def __init__(
        self,
        plan: Plan,
        provider: Any,
        backend: "Backend | None" = None,
        *,
        arena: "tuple[str, _plan.CarriedLayout] | None" = None,
        instrumented: bool = False,
    ) -> None:
        self.plan = plan
        self.provider = provider
        self.instrumented = instrumented  # its binaries may record probes (docs/instrument.md): each bucket gets a probe region for them
        self._backend = backend
        self._doc = plan._doc
        self._buffers = {b["name"]: b for b in self._doc["buffers"]}
        self._rts: "dict[tuple, _BucketRT]" = {}
        self._lock = threading.RLock()
        self.has_library = any(s["kind"] == "library" for s in self._doc["buckets"][0]["steps"])
        self.shared = arena is not None  # the arena is a SharedArena's (other plans run on it too)
        # an arena of its own: the plan's region, then the regions of its carried buffers (section 15.2), which a plan alone keeps between its runs
        self._arena_key, self._carried = arena if arena is not None else (plan.layout_key, _plan.carried_layout([self._doc]))
        self._arena_bytes = self._carried.total
        missing = [b["name"] for b in self._doc["buffers"] if b["role"] == _plan.CARRIED and b["name"] not in self._carried.regions]
        if self._arena_bytes < self._doc["arena_bytes"] or missing:  # pragma: no cover - a SharedArena is laid out for every plan it has
            raise TilefishRuntimeError(f"the arena of {self._arena_bytes} bytes was not laid out for the plan of graph '{plan.name}'")
        sig = plan.graph.signature
        carried = set(plan.carried)
        # what a call takes: the parameters but the carried ones, which the arena holds
        self._call_signature = sig.call_signature.replace(parameters=[q for q in sig.call_signature.parameters.values() if q.name not in carried])
        self._carried_uses = carried_uses(plan)
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
        """The arena of this runner on device ``index``, if it exists: the one of this plan's layout (shared by every runner of that layout), or the
        one of the :class:`SharedArena` the runner belongs to (shared by the runners of its plans); kept until ``clear_caches``."""
        return self.backend.arena_registry().get((self._arena_key, index))

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
                nbytes = self._arena_bytes
                handle, ptr = be.alloc_arena(nbytes, index) if nbytes else (None, 0)
                if nbytes and ptr % _plan.ARENA_ALIGNMENT:
                    raise TilefishRuntimeError(f"the arena was allocated at {ptr:#x}, which is not aligned to {_plan.ARENA_ALIGNMENT} bytes")
                event = be.new_event(index) if nbytes else None  # a plan with no arena shares no memory between its runs: nothing to order
                arena = _Arena(handle, ptr, nbytes, threading.Lock(), event, carried=self._carried)
                if event is not None:
                    # the memory was allocated on the current stream, after whatever used it before there (torch's allocator reuses it in stream
                    # order): the event stands for that from the start, so a first run on another stream waits for it, and one on this stream need not
                    try:
                        with be.scope(index):
                            _record(be, arena, be.stream(index), False)
                    except BaseException:
                        with contextlib.suppress(Exception):
                            be.destroy_event(event, index)
                        raise
                be.arena_registry()[(self._arena_key, index)] = arena
        return arena

    def _bucket(self, values: "Mapping[str, int]", index: int) -> _BucketRT:
        from . import _cuda

        key = (tuple(sorted(values.items())), index, _cuda.current_partition())  # (a persistent kernel is loaded for the SMs of the partition)
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
        probe_bytes = 0
        for i, st in enumerate(steps_doc):
            if st["kind"] == "copy":  # a device-to-device copy: no binary to build or load
                steps.append(_StepRT(_CopyBound.of(st["args"], shapes), None, (), None, None))
                continue
            if st["kind"] == "memset":  # bytes set to zero: no binary either
                steps.append(_StepRT(_MemsetBound.of(st["args"], shapes), None, (), None, None))
                continue
            callee = callees.get(st["entry"])
            where = f"graph '{self.plan.name}' at {values or 'its only size'}, step #{i} ({st['entry']})"
            if callee is None:
                raise TilefishManifestError(f"{where}: the graph calls no operator that has the binary '{st['entry']}'")
            if callee.implementations is not None:  # an extern declared with by_arch=: the implementation of the device's architecture
                callee = callee.at(be.capability(index))
            bound = _plan.resolve_step(callee, st["args"], shapes, where, st.get("given"))
            if bound.stem != st["entry"]:
                raise TilefishManifestError(f"{where}: the arguments of the step select the binary '{bound.stem}', not '{st['entry']}'")
            if callee.kind == "library":
                ws = st["workspace"]
                if shapes[ws][1][0] < bound.spec.workspace_bytes:
                    raise TilefishManifestError(f"{where}: the workspace '{ws}' has {shapes[ws][1][0]} bytes, but the entry needs {bound.spec.workspace_bytes}")
                built = self.provider.pin(bound.spec, index)
                # once, here: a launch repeats it, but a refusal must not come after another step has run (or inside a capture)
                if _is_conv_spec(bound.spec):
                    be.check_conv(built, index)
                    steps.append(_StepRT(bound, None, (), built, ws))  # a convolution step runs its BuiltConv
                else:
                    be.check_gemm(bound.spec, built.algo, index)
                    steps.append(_StepRT(bound, None, (), built.algo, ws))
            else:
                built = self.provider.kernel(callee, bound.spec, index)
                check_launch_config(built, bound.spec)  # the declared warps (a warp-specialised kernel's default partition, and more)
                order = tuple((a.kind, a.name) for a in bound.spec.args if a.kind in ("pointer", "tensordesc", "scalar", "given"))
                loaded = be.load(built, index)
                be.check_grid(loaded, bound.grid)
                probe = None
                if self.instrumented and getattr(built, "instrument", None) is not None:  # its records: a region of the bucket's probe region
                    cuda_grid = tuple(loaded.launch_grid(bound.grid))
                    nbytes = _probe.buffer_bytes(int(built.instrument["capacity"]), -(-built.block[0] // 32), math.prod(cuda_grid))
                    probe = _ProbeStep(-(-probe_bytes // 256) * 256, nbytes, cuda_grid, built)  # type: ignore[arg-type]
                    probe_bytes = probe.offset + nbytes
                ws = st.get("workspace")  # a kernel with a global scratch: the workspace of the arena, every CTA of the launch its share
                if (ws is None) != (bound.spec.global_scratch == 0):
                    raise TilefishManifestError(
                        f"{where}: the entry has {bound.spec.global_scratch} bytes of global scratch per CTA, and the step "
                        + ("names no workspace" if ws is None else f"names the workspace '{ws}'")
                    )
                if ws is not None:
                    need = loaded.scratch_bytes(bound.grid)  # (of the programs a persistent grid runs on the SMs it was loaded for)
                    if shapes[ws][1][0] < need:
                        raise TilefishManifestError(f"{where}: the workspace '{ws}' has {shapes[ws][1][0]} bytes, but the global scratch of the launch is {need}")
                steps.append(_StepRT(bound, loaded, order, None, ws, probe))
        be.warm(index, self.has_library)
        rt = _BucketRT(dict(values), shapes, steps, *self._carried_uses)
        if probe_bytes:  # the probe region of the bucket: allocated here, as the arena is, outside any capture (a captured graph holds its address)
            rt.probe_bytes = probe_bytes
            rt.probe_region = be.alloc_arena(probe_bytes, index)
        return rt

    # -- running ----------------------------------------------------------------------------------------------------------------

    def run(self, args: "tuple[Any, ...]", kwargs: "dict[str, Any]", outputs: Any = None, taps: Any = None, probes: Any = None) -> Any:
        be = self.backend
        bound = self._bind(args, kwargs)
        index = bound.index
        with _gate.shared():
            capturing = be.capturing(index)
            if capturing and probes is not None:
                raise TilefishRuntimeError(
                    "probes= reads the records back after the run, which a capture does not run: use tilefish.capture(..., probes=...) and CapturedPlan.collect()"
                )
            call = self._call(bound, outputs, slots_in_graph=capturing, taps=taps, probes=probes)
            arena = call.arena
            with arena.lock, be.scope(index):
                stream = be.stream(index)
                _check_carried(arena, call.rt, self.plan.name)  # (before the capture is noted: a refused run is no run of the capture)
                if capturing:
                    self._check_one_stream_per_capture(arena, stream)
                _note_thread(arena, self.plan.name, caller_graph=capturing)
                done = False
                try:
                    _ordered(be, arena, stream, capturing, lambda: self._enqueue(call, stream, index, slots=True))
                    done = True
                finally:
                    _carried_written(arena, call.rt, done)
            if probes is not None:
                with be.scope(index):
                    self.collect(call, probes)
        return runtime._result(call.results)

    def collect(self, call: "_Call", probes: Any) -> None:
        """Read back the records of every instrumented step of an executed call into ``probes`` (a :class:`tilefish.Probes`), after the stream it ran on."""
        digest = getattr(getattr(self.provider, "manifest", None), "digest", None)
        for i, st in enumerate(call.rt.steps):
            address = call.probes[i] if call.probes else None
            if st.probe is None or address is None:
                continue
            built = st.probe.built
            probes.collect(
                call.probe_tensor,
                address,
                built=built,
                grid=st.probe.cuda_grid,
                block=built.block,
                stem=built.stem,
                arch=built.arch,
                op=st.bound.spec.op,  # type: ignore[union-attr]
                graph=self.plan.name,
                step=i,
                bucket=call.rt.values,
                manifest_digest=digest,
            )

    def profile(self, args: "tuple[Any, ...]", kwargs: "dict[str, Any]", *, reps: int, warmup: int, method: str) -> "dict[str, Any]":
        """The device time of every step of the bucket the arguments select (:func:`tilefish.profile_plan`): ``{"steps": [[µs per rep] per step],
        "plan": [µs per rep of the plan without the per-step events]}``. A runtime path of its own: the plan, its binaries and every byte it writes are
        those of a run (it is one, repeated); between the steps it records timing events, which no run has. What the runs change that outlives them (the
        carried buffers the plan writes, the caller's ``InOut`` tensors) is copied aside before the first and back after the last, and what the arena
        notes of the carried buffers is as before.

        ``method="graph"``: the steps are captured once in a CUDA graph with an event record node before the first and after each step, and the graph is
        replayed (what a host that captures the plan runs; the device scalars are written once, before). ``method="events"``: the steps are launched as a
        run launches them, with an event recorded on the stream before the first and after each; a spinning kernel first keeps the device busy while the
        host enqueues them, so that they run back to back. Every replay or run is ordered on the plan's arena as a run is."""
        if method not in ("graph", "events"):
            raise TilefishRuntimeError(f"profile: method is 'graph' or 'events', got {method!r}")
        be = self.backend
        bound = self._bind(args, kwargs)
        index = bound.index
        with _gate.shared():
            if be.capturing(index):
                raise TilefishRuntimeError("tilefish.profile_plan is called outside a CUDA graph capture (it captures and replays its own)")
            call = self._call(bound, None, slots_in_graph=False)
            arena = call.arena
            steps = list(zip(call.rt.steps, call.checked))
            n = len(steps)
            with arena.lock, be.scope(index):
                stream = be.stream(index)
                _check_carried(arena, call.rt, self.plan.name)
                _note_thread(arena, self.plan.name)
                # what the runs change that outlives them: the carried buffers the plan writes (and what the arena notes of them) and the caller's
                # InOut tensors. Copied aside before the first run and back after the last, so that profiling leaves the state a run would find
                state: "list[tuple[int, int]]" = []
                for name in call.rt.carried_writes:
                    state.append((call.bases[name], self._carried.regions[name][1]))
                for b in self._doc["buffers"]:
                    if b["role"] == "inout":
                        dtype, shape = call.rt.shapes[b["name"]]
                        state.append((call.bases[b["name"]], math.prod(shape) * _plan.ITEMSIZE[dtype]))
                noted = {name: arena.shapes.get(name) for name in call.rt.carried_writes}
                events: "list[Any]" = []
                ends: "list[Any]" = []
                graphs: "dict[bool, Any]" = {}
                saved: "list[tuple[Any, int, int]]" = []

                def slots() -> None:
                    for b, source, on_device in call.slots:
                        be.write_devscalar(arena.handle, b["offset"], b["dtype"], source, on_device)

                def enqueue(s: int, timed: bool, external: bool) -> None:
                    if timed:
                        be.record_event(events[0], s, external)
                    for i, (st, real) in enumerate(steps):
                        self._enqueue_step(st, real, call.bases, s, index, call.descs[i] if call.descs else {})
                        if timed:
                            be.record_event(events[i + 1], s, external)

                # the events method: how long the spinning kernel keeps the device busy while the host enqueues the steps. 20 us a step at first;
                # every run checks it (the host's time to enqueue the steps against the device's time in the spin, measured by an event before
                # it) and a run whose host was slower is made again with a longer one: a host may take 30 us a step, and the spin may run faster
                # than the clock rate it is sized by (an RTX 5090 boosts past it)
                hold = [20.0 * n + 50.0]

                def one(timed: bool) -> None:
                    host = [0.0]

                    def work() -> None:
                        if method == "graph":
                            if not timed:
                                be.record_event(ends[0], stream, False)
                            be.replay_graph(graphs[timed], stream)
                            if not timed:
                                be.record_event(ends[1], stream, False)
                        else:
                            slots()
                            be.record_event(ends[2], stream, False)
                            be.hold(stream, hold[0], index)
                            t0 = _host_clock()
                            if not timed:
                                be.record_event(ends[0], stream, False)
                            enqueue(stream, timed, False)
                            if not timed:
                                be.record_event(ends[1], stream, False)
                            host[0] = (_host_clock() - t0) * 1e6

                    for _ in range(_HOLD_TRIES):
                        _ordered(be, arena, stream, False, work)
                        if method == "graph":
                            return
                        be.synchronize_event(events[-1] if timed else ends[1], index)
                        held = be.elapsed_ms(ends[2], events[0] if timed else ends[0], index) * 1000.0
                        if host[0] < 0.9 * held:
                            return  # every step was enqueued before the device left the spin: they ran back to back
                        # the device waited for the host: again, with a spin long enough for what this run measured (of the host, and of the spin)
                        hold[0] = max(2.0 * hold[0], 1.5 * host[0] * hold[0] / max(held, 1.0))
                        if hold[0] > _HOLD_MAX_US:
                            break
                    raise TilefishRuntimeError(
                        f"tilefish.profile_plan(method='events'): the host took {host[0]:.0f} us to enqueue the {n} steps of '{self.plan.name}', longer "
                        f"than a spinning kernel of up to {_HOLD_MAX_US / 1e6:.0f} s kept the device busy: the steps would run as fast as the host launches "
                        "them. Use method='graph'")

                out: "dict[str, Any]" = {"steps": [[] for _ in range(n)], "plan": [], "bucket": dict(bound.values), "device": index}
                try:
                    saved = [(be.empty((nbytes,), "u8", index), addr, nbytes) for addr, nbytes in state]
                    _ordered(be, arena, stream, False, lambda: [be.copy(int(t.data_ptr()), addr, nbytes, stream) for t, addr, nbytes in saved])
                    events = [be.new_timing_event(index) for _ in range(n + 1)]
                    ends = [be.new_timing_event(index) for _ in range(3)]  # the plan's start and end without the per-step events; the spin's start
                    if method == "graph":
                        _ordered(be, arena, stream, False, slots)  # the device scalars, once: the graphs hold the launches only
                        graphs[True] = be.capture_graph(index, None, lambda s: enqueue(s, True, True))
                        graphs[False] = be.capture_graph(index, None, lambda s: enqueue(s, False, False))
                    for timed in (True, False):
                        for _ in range(warmup):
                            one(timed)
                    be.synchronize_stream(stream)
                    for _ in range(reps):
                        one(True)
                        be.synchronize_event(events[-1], index)
                        prev = 0.0
                        for i in range(n):
                            at = be.elapsed_ms(events[0], events[i + 1], index)
                            out["steps"][i].append((at - prev) * 1000.0)
                            prev = at
                        one(False)
                        be.synchronize_event(ends[1], index)
                        out["plan"].append(be.elapsed_ms(ends[0], ends[1], index) * 1000.0)
                finally:
                    restored = False
                    try:
                        if len(saved) == len(state):
                            _ordered(be, arena, stream, False, lambda: [be.copy(addr, int(t.data_ptr()), nbytes, stream) for t, addr, nbytes in saved])
                            restored = True
                    finally:
                        for name, shape in noted.items():  # what the arena notes of each: as before the runs, or unknown if the bytes were not put back
                            arena.shapes[name] = shape if restored else None
                        with contextlib.suppress(Exception):
                            be.synchronize_stream(stream)
                        graphs.clear()  # (they hold event record nodes of the events destroyed next)
                        for e in events + ends:
                            with contextlib.suppress(Exception):
                                be.destroy_event(e, index)
        return out

    def _bind(self, args: "tuple[Any, ...]", kwargs: "dict[str, Any]") -> "_BoundCall":
        """Match the arguments to the graph's signature (a ``DevScalar`` passed as a number becomes a tensor of one element on the host), bind the
        dimensions and select the bucket, find the device. Touches no arena and no binary."""
        be = self.backend
        graph = self.plan.graph
        sig = graph.signature
        doc = self._doc
        carried = self.plan.carried
        try:
            bound = self._call_signature.bind(*args, **kwargs)
        except TypeError as e:
            held = f" (the carried parameters {', '.join(carried)} are held by the plan's arena: a call does not give them)" if carried else ""
            raise TilefishPreconditionError(f"{graph.name}: {e}; signature {sig}{held}", kind="shape") from None
        bound.apply_defaults()
        scalar_params = [p for p in sig.params if p.kind == "tensor" and p.is_devscalar]
        sources: "dict[str, Any]" = {}
        for p in scalar_params:
            v = bound.arguments[p.name]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                bound.arguments[p.name] = v = be.devscalar_tensor(devscalar_number(p, v), p.type.dtype.name)  # type: ignore[union-attr]
            sources[p.name] = v
        if not carried:
            call = bind_call(graph.name, sig, bound.args, bound.kwargs)
        else:  # the tensors of the other parameters, then the dimensions from them and from the carried buffers the plan reads (their shapes)
            call, equations = self._bind_given(bound.arguments)
            index = self._device_index(call, sources)
            arena = self.arena_of(index)
            params = {p.name: p for p in sig.params}
            for name in self._carried_uses[0]:
                have = arena.shapes.get(name) if arena is not None else None
                p = params[name]
                if have is not None and len(have) == len(p.type.shape):  # (otherwise the read is refused under the arena's lock)
                    equations.extend((name, axis, pattern, actual, f"{p.type} (the carried buffer)", "dimension") for axis, (pattern, actual) in enumerate(zip(p.type.shape, have)))
            call.dims = solve_dims(graph.name, equations)
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
        if not carried:  # (with carried buffers the device was found first: their shapes are the arena's on that device)
            index = self._device_index(call, sources)
        arch = self.plan.arch
        if arch is not None and arch != f"sm_{be.capability(index)}":
            raise TilefishRuntimeError(f"the plan of graph '{graph.name}' was built for {arch}, but device {index} is sm_{be.capability(index)}")
        for name in sources:  # a device scalar may be on the host, or on this device: a copy from another device is not what a plan orders
            where = call.infos[name].device
            if where != "cpu" and be.device_index(where) != index:
                raise TilefishPreconditionError(f"the device scalar '{name}' is on {where}, but the other tensors of the call are on device {index}", kind="device", arg=name)
        return _BoundCall(call, sources, values, index)

    def _device_index(self, call: Any, sources: "Mapping[str, Any]") -> int:
        """The device of a call: its tensors' (not the device scalars', which may be on the host). A call of a plan whose parameters are all carried or
        device scalars has none: it runs on torch's current device."""
        be = self.backend
        tensors = {n: t for n, t in call.tensors.items() if n not in sources}
        infos = {n: i for n, i in call.infos.items() if n not in sources}
        if not tensors and self.plan.carried:
            return be.current_device()
        device = runtime._device_of(infos, tensors)
        if getattr(device, "type", None) != "cuda":
            raise TilefishRuntimeError(f"jit and replay run on a CUDA device; the tensors are on {device}")
        return be.device_index(device)

    def _bind_given(self, arguments: "Mapping[str, Any]") -> "tuple[Any, list[tuple[Any, ...]]]":
        """What :func:`~tilefish._launch.bind_call` does for a graph with carried parameters, up to the dimensions: the tensor of every other ``In`` and
        ``InOut`` parameter checked against its type (dtype, rank, density) and its shape as equations, which the caller solves with those of the carried
        buffers the plan reads."""
        from ._launch import Call

        sig = self.plan.graph.signature
        carried = set(self.plan.carried)
        tensors: "dict[str, Any]" = {}
        infos: "dict[str, Any]" = {}
        equations: "list[tuple[Any, ...]]" = []
        for p in sig.params:
            if p.kind != "tensor" or p.access == "out" or p.name in carried:
                continue
            t = arguments[p.name]
            info = tensor_info(p.name, t)
            check_tensor(p, info)
            tensors[p.name] = t
            infos[p.name] = info
            assert p.type is not None
            equations.extend((p.name, axis, pattern, actual, str(p.type), "dimension") for axis, (pattern, actual) in enumerate(zip(p.type.shape, info.shape)))
        return Call(tensors, infos, {}, {}), equations

    def _call(self, bound: "_BoundCall", outputs: Any, slots_in_graph: bool, taps: Any = None, probes: Any = None) -> "_Call":
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
            elif role == _plan.CARRIED:  # in the arena's carried region, after the plans' own regions
                bases[name] = arena.ptr + self._carried.regions[name][0]
            else:  # in the arena
                bases[name] = arena.ptr + b["offset"]
        self._check_externals(bases, written, rt, arena)
        # every precondition of every step, before anything is launched
        checked = [st.bound.check(bases) for st in rt.steps]
        with be.scope(index):  # the TMA descriptors of every step, encoded from these addresses, also before anything is launched (an encode may fail)
            descs = [self._descriptors(st, real) for st, real in zip(rt.steps, checked)]
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
        call = _Call(rt, arena, checked, bases, slots, results, self._taps(taps, rt, bases, arena, index), descs)
        if probes is not None:  # the records of each instrumented step: in the bucket's probe region, or in the caller's buffer
            if not self.instrumented:  # pragma: no cover - run_graph gives probes= to an instrumented runner only
                raise TilefishRuntimeError(f"the plan of graph '{self.plan.name}' was not loaded with probes: run it with probes= from the start")
            if probes.buffer is not None:
                tensor, base = probes.region(rt.probe_bytes, index)
            elif rt.probe_region is not None:
                tensor, base = rt.probe_region
            else:
                tensor, base = None, 0
            call.probes = [base + st.probe.offset if st.probe is not None and tensor is not None else None for st in rt.steps]
            call.probe_tensor = tensor
        return call

    def _descriptors(self, st: "_StepRT", real: Any) -> "dict[str, Any]":
        """The TMA descriptors of a checked kernel step (``real``: its tensors with the addresses of the call), by argument name: each encoded from
        its tensor's address, shape and strides (:func:`tilefish.runtime.tensor_map_arg`). Encoded at every execution from what it binds, so a
        descriptor is never one of another binding (another external, another arena)."""
        if isinstance(st.bound, (_CopyBound, _MemsetBound)) or st.algo is not None:
            return {}
        spec = st.bound.spec
        return {a.name: runtime.tensor_map_arg(spec, a.name, real[a.name], self.backend.encode_tensor_map) for a in spec.args if a.kind == "tensordesc"}

    def _enqueue(self, call: "_Call", stream: int, index: int, slots: bool) -> None:
        """Enqueue the device-scalar writes (``slots``) and the launches of the steps on ``stream``, the current stream (the caller orders them: :func:`_ordered`)."""
        be = self.backend
        if slots:
            for b, source, on_device in call.slots:
                be.write_devscalar(call.arena.handle, b["offset"], b["dtype"], source, on_device)
        taps = iter(call.taps)  # sorted by the step they follow
        tap = next(taps, None)
        for i, (st, real) in enumerate(zip(call.rt.steps, call.checked)):
            self._enqueue_step(st, real, call.bases, stream, index, call.descs[i] if call.descs else {}, call.probes[i] if call.probes else None)
            while tap is not None and tap.after == i:  # a debug tap: the buffer as this step left it, copied out before the next step may reuse its bytes
                be.copy(tap.dst, call.bases[tap.buffer], tap.nbytes, stream)
                tap = next(taps, None)

    def _enqueue_step(
        self,
        st: "_StepRT",
        real: Any,
        bases: "Mapping[str, int]",
        stream: int,
        index: int,
        descs: "Mapping[str, Any] | None" = None,
        probes: "int | None" = None,
    ) -> None:
        """Enqueue one checked step (``real``: what its ``check`` returned with the real addresses; ``descs``: its encoded TMA descriptors) on ``stream``."""
        be = self.backend
        if isinstance(st.bound, _CopyBound):
            src, dst, nbytes = real
            be.copy(dst, src, nbytes, stream)
        elif isinstance(st.bound, _MemsetBound):
            dst, nbytes = real
            be.memset(dst, nbytes, stream)
        elif st.algo is not None and _is_conv_spec(st.bound.spec):
            names = list(st.bound.names)
            be.conv(st.algo, real[names[0]].ptr, real[names[1]].ptr, real[names[-1]].ptr, bases[st.workspace], stream, index)  # type: ignore[index]
        elif st.algo is not None:
            names = [a for a in st.bound.names]
            a_ptr, b_ptr = real[names[0]].ptr, real[names[1]].ptr
            bias = real[names[2]].ptr if len(names) == 4 else None
            be.gemm(st.bound.spec, st.algo, a_ptr, b_ptr, bias, real[names[-1]].ptr, bases[st.workspace], stream, index)  # type: ignore[index]
        else:
            launch_args: "list[tuple[str, Any]]" = []
            for kind, n in st.order:
                if kind == "pointer":
                    launch_args.append(("pointer", real[n].ptr))
                elif kind == "tensordesc":
                    launch_args.append(("tensordesc", (descs or {})[n]))
                else:
                    launch_args.append((kind, st.bound.scalars[n]))
            extra = {"probes": probes} if probes is not None else {}  # (an instrumented step: its probe buffer in the profile scratch slot)
            if st.workspace is None:
                be.launch(st.loaded, st.bound.grid, launch_args, stream, **extra)
            else:  # the global scratch of the kernel: the arena's workspace
                be.launch(st.loaded, st.bound.grid, launch_args, stream, scratch=bases[st.workspace], **extra)

    def _written(self, st: "_StepRT") -> "set[str]":
        """The buffers a step writes: a copy's or a memset's destination, a GEMM's ``out``, the ``out`` and ``inout`` pointers of a kernel."""
        b = st.bound
        if isinstance(b, (_CopyBound, _MemsetBound)):
            return {b.dst}
        written = {"out"} if st.algo is not None else b.written
        return {binding for name, binding in zip(b.names, b.bindings) if name in written}

    def _taps(self, taps: Any, rt: "_BucketRT", bases: "Mapping[str, int]", arena: _Arena, index: int) -> "list[_Tap]":
        """``taps=`` of a run or a capture: ``{buffer: tensor}`` or ``{(buffer, step): tensor}``, a copy of the buffer (all of it, at the bucket) into the
        tensor right after the step (by default after the last step of the bucket that writes the buffer). Outside the plan and its digest: a debug
        view of the intermediates of the very plan a product runs. Checked before anything is enqueued."""
        if taps is None:
            return []
        if not isinstance(taps, Mapping):
            raise TilefishRuntimeError(f"taps= maps a buffer name (or a (name, step) pair) to the tensor it is copied into, got {taps!r}")
        out: "list[_Tap]" = []
        n_steps = len(rt.steps)
        ranges = [(arena.ptr, arena.ptr + arena.nbytes)] if arena.nbytes else []
        for b in self._doc["buffers"]:
            if b["role"] in ("input", "weight", "inout", "output"):
                dtype, shape = rt.shapes[b["name"]]
                ranges.append((bases[b["name"]], bases[b["name"]] + math.prod(shape) * _plan.ITEMSIZE[dtype]))
        for spec, tensor in taps.items():
            if isinstance(spec, str):
                name, after = spec, None
            elif isinstance(spec, tuple) and len(spec) == 2:
                name, after = spec
            else:
                name, after = None, None
            buffer = name if isinstance(name, str) and name in self._buffers else _plan.output_buffer(self._doc, name)
            if buffer is None or self._buffers[buffer]["role"] == "workspace":
                raise TilefishRuntimeError(
                    f"taps= names {spec!r}: a tap is a buffer of the plan of graph '{self.plan.name}' (not the workspace), or (buffer, step); its buffers are "
                    f"{', '.join(b['name'] for b in self._doc['buffers'] if b['role'] != 'workspace')}"
                )
            if after is None:
                writers = [i for i, st in enumerate(rt.steps) if buffer in self._written(st)]
                if not writers:
                    raise TilefishRuntimeError(f"taps= names '{buffer}', which no step writes: give the step to copy it after, ({buffer!r}, step)")
                after = writers[-1]
            if isinstance(after, bool) or not isinstance(after, int) or not 0 <= after < n_steps:
                raise TilefishRuntimeError(f"taps= {spec!r}: the step is an index of the {n_steps} steps of the bucket, got {after!r}")
            dtype, shape = rt.shapes[buffer]
            self._check_output(f"taps[{spec!r}]", tensor, dtype, shape, index)
            nbytes = math.prod(shape) * _plan.ITEMSIZE[dtype]
            dst = int(tensor.data_ptr())
            for lo, hi in ranges:
                if dst < hi and lo < dst + nbytes:
                    raise TilefishPreconditionError(f"the tensor of taps[{spec!r}] overlaps a tensor of the call or the plan's arena", kind="alias", arg=buffer)
            ranges.append((dst, dst + nbytes))
            out.append(_Tap(buffer, after, tensor, dst, nbytes))
        return sorted(out, key=lambda t: t.after)

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
            who = f"a plan of the shared arena of graph '{self.plan.name}'" if self.shared else f"the plan of graph '{self.plan.name}'"
            raise TilefishRuntimeError(
                f"{who} was already run in this CUDA graph capture on another stream: two runs on one arena in one capture must be on one stream (each run "
                "waits for the run before it through the arena's order event, and inside one graph a run on a forked stream would not wait for the other, so a "
                "replay would run both at once on the same arena). Run them on the same stream, or capture them in two graphs, or use tilefish.capture"
            )
        arena.captured = (capture, stream)

    def capture(
        self, externals: "tuple[Any, ...]", devscalars: "Mapping[str, Any]", outputs: Any, stream: Any, taps: Any = None, probes: Any = None
    ) -> "CapturedPlan":
        """:func:`capture` of this runner's plan (see there)."""
        be = self.backend
        sig = self.plan.graph.signature
        scalar_names = [p.name for p in sig.params if p.kind == "tensor" and p.is_devscalar]
        others = [n for n in self._call_signature.parameters if n not in scalar_names]  # (not the carried parameters: the arena holds them)
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
            call = self._call(bound, outputs, slots_in_graph=False, taps=taps, probes=probes)
            arena = call.arena
            epoch = _epoch[0]
            with arena.lock, be.scope(index):  # no other run of the plan is enqueued while the graph is captured
                graph = be.capture_graph(index, stream, lambda s: self._enqueue(call, s, index, slots=False))
        held = tuple(bound.call.tensors.values()) + tuple(call.results) + tuple(t.tensor for t in call.taps)
        captured = CapturedPlan(self, graph, call, index, epoch, held)
        captured.probes = probes
        return captured

    def _given_outputs(self, outputs: Any, names: "Sequence[str]") -> "list[Any] | None":
        if outputs is None:
            return None
        if isinstance(outputs, Mapping):  # by name: the graph's names of its returned values, or out0, out1, ... (_plan.output_buffer)
            by_buffer: "dict[str, Any]" = {}
            for key, t in outputs.items():
                buffer = _plan.output_buffer(self._doc, key)
                if buffer is None:
                    raise TilefishPreconditionError(
                        f"outputs= names {key!r}, which is not a returned value of graph '{self.plan.name}' (they are {', '.join(names)}, or out0, out1, ... "
                        "in the order they are returned)",
                        kind="shape",
                    )
                if buffer in by_buffer:
                    raise TilefishPreconditionError(f"outputs= gives the returned value '{buffer}' twice", kind="shape", arg=buffer)
                by_buffer[buffer] = t
            missing = [n for n in names if n not in by_buffer]
            if missing:
                raise TilefishPreconditionError(f"outputs= must give every returned value of graph '{self.plan.name}'; {', '.join(missing)} is missing", kind="shape")
            return [by_buffer[n] for n in names]
        items = [outputs] if not isinstance(outputs, (tuple, list)) else list(outputs)
        if len(items) != len(names):
            raise TilefishPreconditionError(
                f"outputs= must give the {len(names)} returned value(s) of graph '{self.plan.name}' ({', '.join(names)}), got {len(items)}", kind="shape"
            )
        return items

    def run_step(self, step: Any, buffers: Any, bucket: "Mapping[str, int] | None" = None) -> None:
        """:func:`run_step` of this runner's plan (see there)."""
        be = self.backend
        doc = self._doc
        if not isinstance(buffers, Mapping):
            raise TilefishRuntimeError(f"run_step: buffers= maps the name of every buffer the step binds to its tensor, got {buffers!r}")
        n_steps = len(doc["buckets"][0]["steps"])
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < n_steps:
            raise TilefishRuntimeError(f"run_step: the step is an index of the {n_steps} steps of the plan of graph '{self.plan.name}', got {step!r}")
        given: "dict[str, Any]" = {}
        for key, t in buffers.items():
            name = key if isinstance(key, str) and key in self._buffers else _plan.output_buffer(doc, key)
            if name is None or self._buffers[name]["role"] == "workspace":
                raise TilefishRuntimeError(f"run_step: {key!r} is not a buffer of the plan of graph '{self.plan.name}' (the workspace is the plan's own)")
            if name in given:
                raise TilefishRuntimeError(f"run_step: the buffer '{name}' is given twice")
            given[name] = t
        infos = {n: tensor_info(n, t) for n, t in given.items()}
        if bucket is None:  # the one bucket whose shapes are the tensors' (a devscalar of one element fits any)
            fits = []
            for b in doc["buckets"]:
                shapes = _plan.buffer_shapes(doc, b["values"])
                if all(shapes[n][1] == infos[n].shape or (not shapes[n][1] and math.prod(infos[n].shape) == 1) for n in given):
                    fits.append(b["values"])
            if len(fits) != 1:
                raise TilefishPreconditionError(
                    f"run_step: the shapes of the tensors given fit {len(fits)} buckets of the plan of graph '{self.plan.name}' "
                    f"({', '.join(map(str, fits[:8])) or 'none'}): give bucket=",
                    kind="bucket",
                )
            values = dict(fits[0])
        else:
            values = dict(bucket)
            if not any(b["values"] == values for b in doc["buckets"]):
                raise TilefishPreconditionError(f"run_step: {values} is not a bucket of the plan of graph '{self.plan.name}'", kind="bucket")
        st_doc = next(b["steps"] for b in doc["buckets"] if b["values"] == values)[step]
        needed = [a["binding"] for a in st_doc["args"]]
        missing, extra = [n for n in needed if n not in given], sorted(set(given) - set(needed))
        if missing or extra:
            raise TilefishRuntimeError(
                f"run_step: step #{step} of graph '{self.plan.name}' binds the buffers {sorted(set(needed))}"
                + (f"; {', '.join(dict.fromkeys(missing))} missing" if missing else "")
                + (f"; {', '.join(extra)} not bound by it" if extra else "")
            )
        device = runtime._device_of(infos, given)
        if getattr(device, "type", None) != "cuda":
            raise TilefishRuntimeError(f"run_step runs on a CUDA device; the tensors are on {device}")
        index = be.device_index(device)
        arch = self.plan.arch
        if arch is not None and arch != f"sm_{be.capability(index)}":
            raise TilefishRuntimeError(f"the plan of graph '{self.plan.name}' was built for {arch}, but device {index} is sm_{be.capability(index)}")
        with _gate.shared():
            if be.capturing(index):
                raise TilefishRuntimeError("run_step is not captured: run the step outside a CUDA graph capture")
            rt = self._bucket(values, index)
            arena = self._arena(index)
            st = rt.steps[step]
            bases: "dict[str, int]" = {}
            ranges = []
            for name, t in given.items():
                dtype, shape = rt.shapes[name]
                info = infos[name]
                if self._buffers[name]["role"] == "devscalar" and math.prod(info.shape) == 1 and info.dtype == TORCH_DTYPE_NAMES[dtype]:
                    shape = tuple(info.shape)  # a device scalar is a tensor of one element
                self._check_output(f"buffers[{name!r}]", t, dtype, shape, index)
                bases[name] = int(info.ptr)
                ranges.append((name, bases[name], bases[name] + math.prod(shape) * _plan.ITEMSIZE[dtype]))
            if st.workspace is not None:
                bases[st.workspace] = arena.ptr + self._buffers[st.workspace]["offset"]
            written = self._written(st)
            for name, lo, hi in ranges:
                if arena.nbytes and lo < arena.ptr + arena.nbytes and arena.ptr < hi:
                    raise TilefishPreconditionError(f"run_step: the tensor of '{name}' overlaps the arena of the plan", kind="alias", arg=name)
                for other, olo, ohi in ranges:
                    if other != name and (name in written or other in written) and lo < ohi and olo < hi:
                        raise TilefishPreconditionError(f"run_step: the tensors of '{name}' and '{other}' overlap, and the step writes one", kind="alias", arg=name)
            real = st.bound.check(bases)  # every precondition of the step with the caller's addresses
            with be.scope(index):
                descs = self._descriptors(st, real)  # of the caller's tensors
            with arena.lock, be.scope(index):
                stream = be.stream(index)
                _note_thread(arena, self.plan.name)
                _ordered(be, arena, stream, False, lambda: self._enqueue_step(st, real, bases, stream, index, descs))

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
                        "caller must not be in it"
                        + (
                            " (the arena is shared with other plans, which write all of its bytes: a value that outlives a run is a returned value, in a "
                            "tensor of the caller, or a carried buffer of the arena)"
                            if self.shared
                            else ""
                        ),
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
        self.probes: Any = None  # tilefish.capture(..., probes=): the recorder collect() reads the records into

    def collect(self, probes: Any = None) -> None:
        """Read the records the last replay of an instrumented plan made (``tilefish.capture(..., probes=rec)``, ``docs/instrument.md``) into
        ``probes`` (default: the recorder given to the capture). Waits for the device to finish what was enqueued."""
        rec = probes if probes is not None else self.probes
        if rec is None or not self._call.probes:
            raise TilefishRuntimeError("CapturedPlan.collect: the plan was captured without probes= (tilefish.capture(..., probes=tilefish.Probes()))")
        be = self._runner.backend
        with be.scope(self._index):
            be.synchronize_stream(be.stream(self._index))
            runtime._torch().cuda.synchronize(self._index)
            self._runner.collect(self._call, rec)

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
                    rt = self._call.rt
                    _check_carried(arena, rt, runner.plan.name)
                    _note_thread(arena, runner.plan.name)
                    done = False
                    try:
                        _ordered(be, arena, s, False, lambda: self._enqueue(s))
                        done = True
                    finally:
                        _carried_written(arena, rt, done)

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


def _jit_runner(p: Plan, instrument: Any = None) -> PlanRunner:
    """The runner of a plan that ``jit`` runs, kept on the plan (a plan and its runner are freed together; the arena is in the backend's registry). With
    ``instrument`` (a run with ``probes=``): the runner of the instrumented variant, one per setting, kept on the plan too."""
    with _lock:
        if instrument is not None:
            runners = p.__dict__.setdefault("_tf_probe_runners", {})
            r = runners.get(instrument)
            if r is None:
                r = runners[instrument] = PlanRunner(p, JitProvider(instrument), instrumented=True)
            return r
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


def directory_has_set(path: pathlib.Path) -> bool:
    """Whether the plan directory of ``path`` (a directory or a plan file) has the index of a plan set, whose digest a pin may be."""
    directory = path if path.is_dir() else path.parent
    return (directory / _plan.PLAN_SET_NAME).is_file()


def _replay_runner(graph: GraphOp, path: Any, pin: "str | None", trust_directory: bool, instrumented: bool = False) -> PlanRunner:
    _plan.check_anchor(pin, trust_directory)
    try:
        p = pathlib.Path(path)
    except TypeError:
        raise TilefishRuntimeError(f"plan= takes a directory (or its plan.json) for replay, got {path!r}") from None
    where, stamp = _directory_stamp(p)
    key = (where, id(graph), instrumented)  # one runner per directory and graph: a directory that changed replaces it (its arena stays: a CUDA graph may hold it)
    with _lock:
        found = _loaded.get(key)
    if found is not None and found[0] == stamp and found[1].graph is graph:
        _, loaded, runner = found
        if pin is not None and pin not in (loaded.digest, loaded.set_digest):
            if not directory_has_set(p):  # (in a plan set's directory the pin may be the set's digest: loaded again below, which checks it)
                raise TilefishManifestError(
                    f"the plan in {str(loaded.directory)!r} has digest {loaded.digest}, but the pin is {pin}: it is not the plan that was pinned "
                    "(it was changed, or it is another export)"
                )
        elif (pin is not None) == runner.provider.pinned:  # a load with the other trust anchor is made again: the manifest's `pinned` differs
            return runner
    loaded = _plan.load_plan(p, graph, pin=pin, trust_directory=trust_directory, instrumented=instrumented)
    if instrumented and not loaded.manifest.instrumented:
        raise TilefishManifestError(
            f"probes= replays the plan of an instrumented export (tilefish.runtime.export(graph, directory, instrument=True), docs/instrument.md), and "
            f"the plan in {str(loaded.directory)!r} is of a production one, whose binaries record nothing"
        )
    runner = PlanRunner(loaded, ReplayProvider(loaded.manifest, pinned=pin is not None), instrumented=instrumented)
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
    taps: Any = None,
    probes: Any = None,
) -> Any:
    """``tilefish.run`` of a graph (see :func:`tilefish.run`)."""
    if mode == "reference":
        if plan is not None or outputs is not None or taps is not None or probes is not None:
            raise TilefishRuntimeError("plan=, outputs=, taps= and probes= are for jit and replay of a graph; reference mode evaluates the ops in torch and returns fresh tensors")
        return reference_run(graph, args, kwargs)
    instrument = probes.instrument if probes is not None else None
    if mode == "jit":
        if isinstance(plan, Plan):
            p = plan
            if p.graph is not graph:
                raise TilefishRuntimeError(f"plan= is the plan of graph '{p.name}', not of '{graph.name}'")
        elif plan is None:
            p = _default_plan(graph)
        else:
            raise TilefishRuntimeError("jit takes plan=<a Plan made by tilefish.plan(graph, ...)> or none; a directory is for mode='replay'")
        return _jit_runner(p, instrument).run(args, kwargs, outputs, taps, probes)
    return _replay_runner(graph, _plan_directory(plan), pin, trust_directory, probes is not None).run(args, kwargs, outputs, taps, probes)


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
    taps: Any = None,
    probes: Any = None,
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
        instrument = probes.instrument if probes is not None else None
        if isinstance(plan, Plan):
            if plan.graph is not graph:
                raise TilefishRuntimeError(f"plan= is the plan of graph '{plan.name}', not of '{graph.name}'")
            runner = _jit_runner(plan, instrument)
        elif plan is None:
            runner = _jit_runner(_default_plan(graph), instrument)
        else:
            raise TilefishRuntimeError("jit takes plan=<a Plan made by tilefish.plan(graph, ...)> or none; a directory is for mode='replay'")
    else:
        if trust_directory is not True and trust_directory is not False:
            raise TilefishRuntimeError(f"trust_directory= is True or False, got {trust_directory!r}")
        runner = _replay_runner(graph, _plan_directory(plan), pin, trust_directory, probes is not None)
    return runner.capture(tuple(externals), {} if devscalars is None else devscalars, outputs, stream, taps, probes)


def _runner_for(graph: GraphOp, what: str, mode: str, plan: Any, pin: "str | None", trust_directory: bool) -> PlanRunner:
    """The runner ``mode`` (``jit`` or ``replay``) chooses for ``graph``, as :func:`capture` chooses it."""
    if mode not in ("jit", "replay"):
        raise TilefishRuntimeError(f"{what}: mode is 'jit' or 'replay', got {mode!r}")
    if mode == "jit":
        if pin is not None or trust_directory:
            raise TilefishRuntimeError("pin= and trust_directory= are for mode='replay' (they vouch for a plan directory); mode 'jit' runs no plan directory")
        if isinstance(plan, Plan):
            if plan.graph is not graph:
                raise TilefishRuntimeError(f"plan= is the plan of graph '{plan.name}', not of '{graph.name}'")
            return _jit_runner(plan)
        if plan is None:
            return _jit_runner(_default_plan(graph))
        raise TilefishRuntimeError("jit takes plan=<a Plan made by tilefish.plan(graph, ...)> or none; a directory is for mode='replay'")
    if trust_directory is not True and trust_directory is not False:
        raise TilefishRuntimeError(f"trust_directory= is True or False, got {trust_directory!r}")
    return _replay_runner(graph, _plan_directory(plan), pin, trust_directory)


class WeightedRun:
    """The plan of a graph with its **weights bound once**: :func:`bind_weights` checked them against the plan, and every :meth:`run` and
    :meth:`capture` takes the other arguments only (the inputs, the ``InOut`` tensors and the device scalars, in the order of the signature, or by
    name). The roles are the graph's declaration (``@tilefish.graph(inputs=..., weights=...)``), which the plan records.

    It holds the weight tensors (a run reads them where they are: write new values into them in place, as into any input). Runs and captures go
    through the plan's runner, so they are ordered with every other run of the plan as :func:`tilefish.run` is."""

    def __init__(self, runner: PlanRunner, weights: "dict[str, Any]") -> None:
        self._runner = runner
        self._weights = weights
        sig = runner.plan.graph.signature
        call = runner._call_signature  # the parameters a call gives (not the carried ones, which the arena holds)
        self._names = [n for n in call.parameters if n not in weights]  # what a run takes, in the order of the signature
        scalars = {p.name for p in sig.params if p.kind == "tensor" and p.is_devscalar}
        self._others = [n for n in call.parameters if n not in scalars]  # the externals of a capture, weights included

    def __repr__(self) -> str:
        return f"WeightedRun(graph={self.plan.name!r}, weights={list(self._weights)}, takes={self._names})"

    @property
    def plan(self) -> Plan:
        return self._runner.plan

    @property
    def weights(self) -> "dict[str, Any]":
        """The weights, by name (the tensors bound: write into them in place to change them)."""
        return dict(self._weights)

    def _arguments(self, args: "tuple[Any, ...]", kwargs: "Mapping[str, Any]", names: "Sequence[str]", what: str) -> "dict[str, Any]":
        given = [n for n in kwargs if n in self._weights]
        if given:
            raise TilefishPreconditionError(
                f"{what} of graph '{self.plan.name}': {', '.join(given)} is a weight, bound once by bind_weights (write new values into the tensor bound)",
                kind="shape",
                arg=given[0],
            )
        if len(args) > len(names):
            raise TilefishPreconditionError(
                f"{what} of graph '{self.plan.name}' takes at most {len(names)} argument(s) besides its weights ({', '.join(names)}), got {len(args)}",
                kind="shape",
            )
        return {**self._weights, **dict(zip(names, args)), **kwargs}

    def run(self, *args: Any, outputs: Any = None, taps: Any = None, **kwargs: Any) -> Any:
        """``tilefish.run`` of the graph with the weights bound: ``args`` are the parameters that are not weights, in the order of the signature (or by
        name); ``outputs=`` and ``taps=`` as there."""
        return self._runner.run((), self._arguments(args, kwargs, self._names, "a run"), outputs, taps)

    def capture(self, *externals: Any, devscalars: "Mapping[str, Any] | None" = None, outputs: Any = None, stream: Any = None, taps: Any = None) -> CapturedPlan:
        """``tilefish.capture`` of the graph with the weights bound: ``externals`` are the parameters that are neither weights nor device scalars, in
        the order of the signature; ``devscalars=``, ``outputs=``, ``stream=`` and ``taps=`` as there."""
        names = [n for n in self._others if n not in self._weights]
        merged = self._arguments(externals, {}, names, "a capture")
        missing = [n for n in self._others if n not in merged]
        if missing:
            raise TilefishPreconditionError(f"a capture of graph '{self.plan.name}' needs {', '.join(missing)}", kind="shape", arg=missing[0])
        return self._runner.capture(tuple(merged[n] for n in self._others), {} if devscalars is None else devscalars, outputs, stream, taps)

    def prepare(self, device: Any = None, buckets: "Sequence[Mapping[str, int]] | None" = None) -> None:
        """:func:`prepare` of the plan (its arena, binaries and handles), on ``device`` (default: the weights')."""
        if device is None and self._weights:
            device = next(iter(self._weights.values())).device
        index = runtime._torch().cuda.current_device() if device is None else self._runner.backend.device_index(device)
        self._runner.prepare(index, buckets)


def bind_weights(
    graph: GraphOp,
    weights: "Mapping[str, Any]",
    *,
    mode: str = "jit",
    plan: Any = None,
    pin: "str | None" = None,
    trust_directory: bool = False,
) -> WeightedRun:
    """Bind the weights of ``graph`` once: a :class:`WeightedRun` whose runs and captures take only the other arguments.

    The plan must declare its roles (``@tilefish.graph(inputs=..., weights=...)``: ``"roles": "declared"`` in the document; a plan whose roles were
    chosen from the shapes is refused, since a fully static graph labels its inputs weights), and ``weights`` must give a tensor for every parameter of
    role ``weight`` and for nothing else, each on a CUDA device, of its dtype and rank, contiguous (and of its static sizes); the rest is checked at
    every run, as for any argument. ``mode``, ``plan``, ``pin`` and ``trust_directory`` choose the plan as for :func:`tilefish.run`."""
    if not isinstance(graph, GraphOp) or isinstance(graph, KernelOp):
        raise TilefishRuntimeError(f"bind_weights takes a @tilefish.graph, got {graph!r}")
    return weighted(_runner_for(graph, "bind_weights", mode, plan, pin, trust_directory), weights)


def weighted(runner: PlanRunner, weights: "Mapping[str, Any]") -> WeightedRun:
    """:func:`bind_weights` on a runner: the checks of the weights against its plan, then the :class:`WeightedRun`."""
    graph = runner.plan.graph
    p = runner.plan
    if not p.roles_declared:
        raise TilefishRuntimeError(
            f"the plan of graph '{p.name}' does not declare which parameters are weights (its roles were chosen from the shapes, and a fully static graph "
            "labels its inputs weights): declare them with @tilefish.graph(inputs=..., weights=...) and plan or export it again"
        )
    if not isinstance(weights, Mapping):
        raise TilefishRuntimeError(f"bind_weights: weights= maps the name of every weight of graph '{p.name}' to its tensor, got {weights!r}")
    want = p.weights
    unknown = [n for n in weights if n not in want]
    missing = [n for n in want if n not in weights]
    if unknown or missing:
        raise TilefishPreconditionError(
            f"bind_weights: graph '{p.name}' has the weights {want}"
            + (f"; {', '.join(map(str, unknown))} is not one" if unknown else "")
            + (f"; {', '.join(missing)} is missing" if missing else ""),
            kind="shape",
            arg=(unknown or missing)[0],
        )
    params = {q.name: q for q in graph.signature.params}
    device = None
    shapes = {b["name"]: b["shape"] for b in p.doc["buffers"]}
    for name, t in weights.items():
        info = tensor_info(name, t)
        check_tensor(params[name], info)
        if not str(info.device).startswith("cuda"):
            raise TilefishPreconditionError(f"bind_weights: the weight '{name}' is on {info.device}; a plan runs on a CUDA device", kind="device", arg=name)
        if device is None:
            device = info.device
        elif runner.backend.device_index(info.device) != runner.backend.device_index(device):
            raise TilefishPreconditionError(f"bind_weights: the weight '{name}' is on {info.device}, the others on {device}", kind="device", arg=name)
        for axis, (want_size, size) in enumerate(zip(shapes[name], info.shape)):
            if isinstance(want_size, int) and want_size != size:
                raise TilefishPreconditionError(
                    f"bind_weights: the weight '{name}' has size {size} on axis {axis}, but the plan has {want_size} there", kind="static_size", arg=name
                )
    return WeightedRun(runner, {n: weights[n] for n in want})


# -- an arena shared by several plans ------------------------------------------------------------------------------------------------------

SHARED_ARENA_TAG = b"tilefish.shared-arena.v1\0"


class SharedArena:
    """Several plans that a host runs **one at a time** on **one arena** (``docs/plan-schema-1.md``, section 15.1): the plans of a set (stage A, then
    B, then C of a request; a variable number of tile plans), which would otherwise each have an arena of their own.

    **The layout.** The arena has the bytes of the largest arena of the plans (``arena_bytes``: the largest ``arena_bytes`` of their documents), and
    each plan keeps its own layout in it, from the same first byte: every buffer at the ``offset`` its plan says, so a CUDA graph captured from any of
    them holds fixed addresses. Nothing of a plan's document changes (its digest neither, nor the set's): the layout is the plans' own, which their
    pin covers. The plans' intermediates, device-scalar slots and cuBLASLt workspaces overlap; that is sound because no two executions on the arena
    overlap on the device (next item) and an execution writes every arena buffer it reads before it reads it (a device scalar's slot before the first
    launch).

    **The order.** The arena has one lock and one order event (:class:`_Arena`): every run, capture, replay of a :class:`CapturedPlan` and
    :meth:`run_step` of **any** of the plans takes the lock, makes its stream wait for the event and records it after its last launch, exactly as the
    runs of one plan do. So two executions on the arena, of one plan or of two, from any threads and on any streams, run one after the other on the
    device, in the order they were issued: they cannot run concurrently. A ``torch.cuda.graph`` capture of runs on the arena on two streams of one
    capture is refused, as for one plan.

    **What outlives a run.** In the plans' own regions, nothing: the next execution, of any plan of the arena, writes over it. A value one plan
    makes and a later plan reads is a **returned value** of the first (a tensor of the caller's: ``outputs=``, or allocated by the run) passed as an
    argument of the second, or a **carried buffer** (section 15.2): a parameter the plans declare ``carried=`` (``@tilefish.graph(carried=("h",))``),
    which the arena holds in a region of its own after the plans' regions (:attr:`carried`), which no other buffer of any plan shares, and which a
    plan that writes it leaves there for the next. An execution that reads a carried buffer is refused unless the executions issued before it on the
    arena left it in the shape it reads. A tensor of the caller that overlaps the arena, anywhere in it (not only in the part of the plan that
    runs), is refused, as is a tap or a tensor of :meth:`run_step` that does: what would be overwritten is refused, never corrupted.

    Arenas are kept per layout as for one plan: every ``SharedArena`` of the same plans (the same layouts, in ``jit`` or ``replay``) and the same
    ``name`` on one device uses one arena, kept until ``tilefish.runtime.clear_caches()``; another ``name`` is another arena, with its own lock, order
    event and carried buffers, so executions on two of them run concurrently (one arena per stream, say). A plan run alone (``tilefish.run``) has its
    own arena, not this one."""

    def __init__(self, runners: "Sequence[PlanRunner]", name: "str | None" = None) -> None:
        runners = list(runners)
        if not runners:
            raise TilefishRuntimeError("a shared arena is the arena of one plan or more: give the plans that share it")
        if name is not None and (not isinstance(name, str) or not name):
            raise TilefishRuntimeError(f"name= of a shared arena is a non-empty string (one arena per name: one per stream, say), got {name!r}")
        names = [r.plan.name for r in runners]
        twice = sorted({n for n in names if names.count(n) > 1})
        if twice:
            raise TilefishRuntimeError(f"a shared arena has each plan once: {', '.join(twice)} is given twice")
        # a runner that has not run yet has no backend: it gets the process's TorchBackend when it first needs one (as a runner that ran has it)
        default = _torch_backend[0] if _torch_backend else None
        backends = {id(r._backend if r._backend is not None else default) for r in runners}
        if len(backends) != 1:
            raise TilefishRuntimeError("the plans of a shared arena run on one backend")
        layouts = sorted(r.plan.layout_key for r in runners)
        self.name = name
        self.key = "shared:" + hashlib.sha256(SHARED_ARENA_TAG + "\0".join(layouts).encode("ascii")).hexdigest()
        if name is not None:  # another arena of the same plans: its own memory, lock, order event and carried buffers
            self.key += ":" + hashlib.sha256(name.encode("utf-8")).hexdigest()
        self.layout = _plan.carried_layout([r.plan._doc for r in runners])
        self.arena_bytes = self.layout.total
        self._runners = {r.plan.name: PlanRunner(r.plan, r.provider, r._backend, arena=(self.key, self.layout)) for r in runners}

    def __repr__(self) -> str:
        named = "" if self.name is None else f", name={self.name!r}"
        return f"SharedArena(plans={list(self._runners)}, arena_bytes={self.arena_bytes}{named})"

    @property
    def carried(self) -> "dict[str, tuple[int, int, str]]":
        """The carried buffers of the plans (section 15.2), by name: ``(offset from the arena's first byte, bytes, dtype)``, after the plans' own
        regions."""
        return dict(self.layout.regions)

    @property
    def names(self) -> "list[str]":
        """The graphs of the plans, in the order they were given."""
        return list(self._runners)

    @property
    def plans(self) -> "dict[str, Plan]":
        """The plans, by graph name."""
        return {n: r.plan for n, r in self._runners.items()}

    @property
    def sizes(self) -> "dict[str, int]":
        """The ``arena_bytes`` of each plan (its own part of the arena, from the first byte), by graph name."""
        return {n: r.plan.arena_bytes for n, r in self._runners.items()}

    def runner(self, graph: Any) -> PlanRunner:
        """The runner of ``graph`` (the graph, or its name) on the shared arena; a graph whose plan is not one of the arena's is refused."""
        name = graph if isinstance(graph, str) else getattr(graph, "name", None)
        r = self._runners.get(name) if isinstance(name, str) else None
        if r is None or (not isinstance(graph, str) and r.plan.graph is not graph):
            raise TilefishRuntimeError(f"{graph!r} is not a graph of this shared arena: its plans are {list(self._runners)}")
        return r

    def run(self, graph: Any, *args: Any, outputs: Any = None, taps: Any = None, **kwargs: Any) -> Any:
        """``tilefish.run(graph, *args, outputs=, taps=)`` of one plan of the arena, on the arena (ordered with every other execution on it)."""
        return self.runner(graph).run(args, kwargs, outputs, taps)

    def capture(
        self, graph: Any, *externals: Any, devscalars: "Mapping[str, Any] | None" = None, outputs: Any = None, stream: Any = None, taps: Any = None
    ) -> CapturedPlan:
        """``tilefish.capture`` of one plan of the arena, on the arena: the :class:`CapturedPlan`'s replays are ordered with every run and replay of
        every plan of the arena."""
        return self.runner(graph).capture(tuple(externals), {} if devscalars is None else devscalars, outputs, stream, taps)

    def bind_weights(self, graph: Any, weights: "Mapping[str, Any]") -> WeightedRun:
        """``bind_weights`` of one plan of the arena: its runs and captures are on the arena."""
        return weighted(self.runner(graph), weights)

    def run_step(self, graph: Any, step: int, buffers: "Mapping[str, Any]", bucket: "Mapping[str, int] | None" = None) -> None:
        """``run_step`` of one plan of the arena (its GEMM workspace is in the arena: ordered with the other executions on it)."""
        self.runner(graph).run_step(step, buffers, bucket)

    def prepare(self, device: Any = None, buckets: "Mapping[str, Sequence[Mapping[str, int]]] | None" = None) -> None:
        """``prepare`` of every plan of the arena (the arena once, the binaries of every bucket, or of ``buckets[graph name]``), on ``device``
        (default: torch's current device)."""
        first = next(iter(self._runners.values()))
        index = runtime._torch().cuda.current_device() if device is None else first.backend.device_index(device)
        if buckets is not None:
            unknown = sorted(set(buckets) - set(self._runners))
            if unknown:
                raise TilefishRuntimeError(f"prepare: buckets= names {unknown}, which are not graphs of this shared arena ({list(self._runners)})")
        for name, r in self._runners.items():
            r.prepare(index, None if buckets is None or name not in buckets else buckets[name])

    def arena_of(self, index: int) -> "_Arena | None":
        """The arena on device ``index``, if it was allocated."""
        return next(iter(self._runners.values())).arena_of(index)


def share_arena(
    graphs: "Sequence[Any]",
    *,
    mode: str = "jit",
    plan: Any = None,
    pin: "str | None" = None,
    trust_directory: bool = False,
    name: "str | None" = None,
) -> SharedArena:
    """The plans of ``graphs`` on **one arena** they share, run one at a time (:class:`SharedArena`; ``docs/plan-schema-1.md``, section 15.1).

    ``replay``: the plans of the graphs in the plan set ``plan=<directory>``, verified with ``pin=`` (the set's digest) or ``trust_directory=True``, as
    :func:`tilefish.run` loads them. ``jit``: an item is a graph (its default plan) or a :class:`~tilefish._plan.Plan` that ``tilefish.plan`` made.

    ``name=`` names the arena: every ``share_arena`` of the same plans with the same name (or without one) on a device is one arena, and another name
    is another one. The executions on one arena run one after the other on the device, whatever their streams; executions that must run concurrently
    (on two streams) need an arena each: ``share_arena([g], name="lane0")``, ``share_arena([g], name="lane1")`` for one plan on two streams."""
    if isinstance(graphs, (str, GraphOp, Plan)) or not isinstance(graphs, (list, tuple)):
        raise TilefishRuntimeError(f"share_arena takes a list of graphs (or of plans, in jit), got {graphs!r}")
    if mode not in ("jit", "replay"):
        raise TilefishRuntimeError(f"share_arena: mode is 'jit' or 'replay', got {mode!r}")
    if mode == "jit" and plan is not None:
        raise TilefishRuntimeError("share_arena in jit takes the plans in the list (tilefish.plan(graph, ...)); plan= is the directory of a replay")
    runners = []
    for item in graphs:
        if mode == "jit" and isinstance(item, Plan):
            runners.append(_runner_for(item.graph, "share_arena", mode, item, pin, trust_directory))
        elif isinstance(item, GraphOp) and not isinstance(item, KernelOp):
            runners.append(_runner_for(item, "share_arena", mode, plan, pin, trust_directory))
        else:
            raise TilefishRuntimeError(f"share_arena takes @tilefish.graph objects{' (or plans)' if mode == 'jit' else ''}, got {item!r}")
    return SharedArena(runners, name=name)


def replay_graph(graph: Any, stream: Any = None) -> None:
    """Replay a CUDA graph of the caller's (a ``torch.cuda.CUDAGraph`` in which ``tilefish.run`` of plans was captured) **ordered with every
    execution of those plans on any thread**: ``graph.replay()`` on ``stream`` (default: the current stream) under the lock of every arena such a
    graph was captured on. Its event nodes order it on the device (``docs/plan-schema-1.md``, section 10, item 7), and the lock makes sure no
    execution of another thread is between its wait for the order event and its record while the graph is launched. A replay that bypasses this
    (``graph.replay()`` itself) is ordered only with what the thread that issues it does; ``tilefish.capture`` makes a graph whose own replay is
    ordered."""
    be = torch_backend()
    with _gate.shared():
        arenas = sorted((a for a in list(be.arena_registry().values()) if a.caller_graphs), key=id)  # one order of the locks: no two replays deadlock
        with contextlib.ExitStack() as held:
            for a in arenas:
                held.enter_context(a.lock)
            for a in arenas:
                a.ordered = True  # the caller orders the replays of its graphs: the warning of _note_thread is not for it
                a.threads.add(threading.get_ident())
            with be.torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                graph.replay()


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


def run_step(
    graph: GraphOp,
    step: int,
    buffers: "Mapping[str, Any]",
    *,
    mode: str = "jit",
    plan: Any = None,
    pin: "str | None" = None,
    trust_directory: bool = False,
    bucket: "Mapping[str, int] | None" = None,
) -> None:
    """Launch one step of the plan of ``graph`` on the caller's tensors: the binary the plan's manifest has for the step's entry (``replay``: the plan
    verified against ``pin``, which pins that manifest, or ``trust_directory=True``; ``jit``: built in memory) and the binding the plan says (each
    argument the buffer and the view the step names), with every check of the step made with the caller's addresses before the launch.

    ``buffers`` maps the name of every buffer the step binds (a returned value also as ``out<i>``; not the workspace: a GEMM's is the plan's arena) to a
    tensor of the buffer's dtype and its shape at the bucket, contiguous (a device scalar: a tensor of one element). ``bucket`` is the bucket
    combination, found from the shapes when exactly one fits them. The launch is on torch's current stream, ordered with the runs of the plan
    (the arena's order event). What it is for: one op of an exported plan replayed on the inputs of a reference, under the pin the product uses."""
    if mode not in ("jit", "replay"):
        raise TilefishRuntimeError(f"run_step: mode is 'jit' or 'replay', got {mode!r}")
    if mode == "jit":
        if pin is not None or trust_directory:
            raise TilefishRuntimeError("pin= and trust_directory= are for mode='replay' (they vouch for a plan directory); mode 'jit' runs no plan directory")
        runner = _jit_runner(plan if isinstance(plan, Plan) else _default_plan(graph))
    else:
        runner = _replay_runner(graph, _plan_directory(plan), pin, trust_directory)
    runner.run_step(step, buffers, bucket)


# -- export --------------------------------------------------------------------------------------------------------------------------------


def export_graph(
    graph: GraphOp,
    directory: "str | os.PathLike[str]",
    *,
    arch: "int | str | None",
    overwrite: bool,
    buckets: Any,
    reuse: bool,
    verify: "bool | None" = None,
    trace: bool = False,
    instrument: Any = None,
) -> Plan:
    """:func:`tilefish.runtime.export` of a graph: build the binary of every step (and pin every GEMM), write the manifest (merged into one that is
    already there) and ``plan.json`` next to it, read both back, and return the plan loaded from the directory, which has the digest to pin.

    A graph whose body has arithmetic first has its fused regions verified against their unfused composition (``verify``: ``None`` takes the
    graph's ``tilefish.fusion(verify=...)``, which is ``True`` by default) and gets its fusion report, ``<graph>.fusion.json``."""
    from ._build import parse_arch

    p = make_plan(graph, buckets, reuse=reuse)
    if arch is None:
        torch = runtime._torch()
        cap = runtime._capability(torch, torch.cuda.current_device())
    else:
        cap = parse_arch(arch)
    checked = _verify_fusion(graph, p, cap, verify)
    built, pinned, traces = _build_callees(sorted(p.callees.items(), key=lambda kv: kv[0]), cap, trace, at=True, instrument=instrument)
    written = write_export(p, built, pinned, directory, cap, overwrite)
    _write_fusion_report(graph, directory, checked)
    if trace:
        from . import _trace
        from ._manifest import Manifest

        _trace.write(directory, traces, Manifest.load(directory, instrumented=True), [(graph, written, PLAN_NAME)])
    return written


def _build_callees(
    callees: "Sequence[tuple[str, Any]]", cap: int, trace: bool, at: bool, instrument: Any = None
) -> "tuple[list, list, list]":
    """Build (or pin) the binary of every ``(stem, callee)`` for ``sm_<cap>`` (``at``: take the callee's implementation of the architecture first);
    with ``trace``, keep every level of each compile and the trace items (:mod:`tilefish._trace`); with ``instrument``, the instrumented variant of
    every Triton binary of an operator it covers (``docs/instrument.md``)."""
    from ._build import build
    from ._binary import build_binary

    built: list = []
    pinned: list = []
    traces: list = []
    for stem, callee in callees:
        if at:
            callee = callee.at(cap)  # an extern declared with by_arch=: the implementation of the target architecture
        spec = callee.by_stem[stem]
        jit = None
        if callee.kind == "library":
            pinned.append(_pin_library(spec, cap))
            continue
        elif callee.extern.binary is not None:  # type: ignore[union-attr]
            built.append(build_binary(spec, callee.extern.binary, arch=cap))  # type: ignore[union-attr]
        else:
            jit = callee.target.jit_function(spec)
            cfg = instrument if instrument is not None and instrument.covers(spec.op) else None
            built.append(build(spec, arch=cap, fn=jit, stages=trace, instrument=cfg))
        if trace:
            from . import _trace

            traces.append(_trace.item_of(built[-1], callee.target, jit))
    return built, pinned, traces


def _verify_fusion(graph: GraphOp, p: Plan, cap: int, verify: "bool | None") -> "dict[int, Any] | None":
    """The export-time check of a graph's fused regions against their unfused composition, at every bucket of its plan (:mod:`tilefish._fusion`)."""
    from . import _fusion

    if verify is not None and not isinstance(verify, bool):
        raise TilefishRuntimeError(f"verify= is True, False or None (the graph's tilefish.fusion(verify=...)), got {verify!r}")
    return _fusion.verify_for_export(graph, _plan.bucket_combinations(p.dims), cap, verify)


def _write_fusion_report(graph: GraphOp, directory: Any, checked: "dict[int, Any] | None") -> None:
    """``<graph>.fusion.json`` next to the plan, for a graph whose body has arithmetic: its fused regions, what each took and keeps (informative: no
    host reads it, and Python makes it again from the graph)."""
    from . import _fusion
    from ._build import write_file

    if not _fusion.has_arithmetic(graph):
        return
    doc = _fusion.report(graph, checked)
    write_file(pathlib.Path(directory) / f"{graph.graph.name}.fusion.json", (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("utf-8"))


def _pin_library(spec: Any, cap: int) -> Any:
    """Pin a library step's spec for ``cap``: a cuBLASLt GEMM's algorithm or a cuDNN convolution's pin."""
    from . import _conv, _library

    return _conv.build_conv(spec, arch=cap, canary=True) if isinstance(spec, _conv.ConvSpec) else _library.build_library(spec, arch=cap)


def write_export(p: Plan, built: "Sequence[Any]", pinned: "Sequence[Any]", directory: "str | os.PathLike[str]", cap: int, overwrite: bool) -> Plan:
    """The half of :func:`export_graph` that writes: the builds (``BuiltKernel`` and ``BuiltLibrary``, one per entry of the plan) go to the manifest in
    ``directory`` (merged into one that is there), ``plan.json`` is written next to it against the digest of the manifest as read back, and the plan is
    loaded back from the directory (verified) and returned."""
    from ._build import write_file, write_manifest_doc
    from ._manifest import MANIFEST_NAME, Manifest, manifest_digest

    directory = pathlib.Path(directory)
    plan_path = directory / PLAN_NAME
    if (directory / _plan.PLAN_SET_NAME).exists() and not overwrite:
        raise TilefishManifestError(
            f"{str(directory)!r} holds a plan set (plans.json): exporting one graph there would change the manifest its plans were built against; use "
            "another directory, export the set again with the graph in it, or pass overwrite=True"
        )
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
    loaded = Manifest.load(path, instrumented=True)  # (what this export wrote: either variant, never the two merged)
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
    return _plan.load_plan(plan_path, p.graph, pin=final.digest, instrumented=True)


def _graph_and_buckets(item: Any) -> "tuple[GraphOp, Any]":
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], GraphOp):
        return item[0], item[1]
    return item, None


def export_graphs(
    graphs: "Sequence[Any]",
    directory: "str | os.PathLike[str]",
    *,
    arch: "int | str | None",
    overwrite: bool,
    buckets: Any,
    reuse: bool,
    verify: "bool | None" = None,
    trace: bool = False,
    instrument: Any = None,
) -> "_plan.PlanSet":
    """:func:`tilefish.runtime.export` of several graphs: a **plan set**, one manifest for all of them (each binary built once, whichever plans
    launch it), ``<graph>.plan.json`` per graph and the index ``plans.json`` (``docs/plan-schema-1.md``, section 15). An item is a graph, or
    ``(graph, buckets)``; ``buckets=`` may map a graph's name to its buckets instead. Returns the :class:`~tilefish._plan.PlanSet` loaded back."""
    from ._build import parse_arch

    plans: "list[Plan]" = []
    for item in graphs:
        g, b = _graph_and_buckets(item)
        if not isinstance(g, GraphOp) or isinstance(g, KernelOp):
            raise TilefishRuntimeError(f"export of a plan set takes graphs (or (graph, buckets) pairs), got {item!r}")
        if b is None and isinstance(buckets, Mapping):
            b = buckets.get(g.name)
        plans.append(make_plan(g, b, reuse=reuse))
    names = [p.name for p in plans]
    if len(set(names)) != len(names):
        raise TilefishRuntimeError(f"the graphs of a plan set have distinct names (a plan's file is <graph>.plan.json), got {names}")
    if arch is None:
        torch = runtime._torch()
        cap = runtime._capability(torch, torch.cuda.current_device())
    else:
        cap = parse_arch(arch)
    checked = {p.name: _verify_fusion(p.graph, p, cap, verify) for p in plans}
    callees: "dict[str, Any]" = {}
    for p in plans:
        for stem, callee in p.callees.items():
            callee = callee.at(cap)  # an extern declared with by_arch=: the implementation of the target architecture
            other = callees.setdefault(stem, callee)
            if other is not callee and callee.by_stem[stem] != other.by_stem[stem]:
                raise TilefishRuntimeError(f"two plans of the set launch different binaries under one name '{stem}': rename one of the operators")
    built, pinned, traces = _build_callees(sorted(callees.items(), key=lambda kv: kv[0]), cap, trace, at=False, instrument=instrument)
    written = write_plan_set(plans, built, pinned, directory, cap, overwrite)
    for p in plans:
        _write_fusion_report(p.graph, directory, checked[p.name])
    if trace:
        from . import _trace
        from ._manifest import Manifest

        graphs = [(p.graph, written.plans[p.name] if hasattr(written, "plans") else p, _plan.plan_file_name(p.name)) for p in plans]
        _trace.write(directory, traces, Manifest.load(directory, instrumented=True), graphs)
    return written


def write_plan_set(
    plans: "Sequence[Plan]", built: "Sequence[Any]", pinned: "Sequence[Any]", directory: "str | os.PathLike[str]", cap: int, overwrite: bool
) -> "_plan.PlanSet":
    """The half of :func:`export_graphs` that writes: one manifest of every build (merged into one that is there), each plan against its digest as
    ``<graph>.plan.json``, the index ``plans.json``; then the set is read back (verified, every plan through the set's pin) and returned."""
    from . import __version__
    from ._build import write_file, write_manifest_doc
    from ._manifest import MANIFEST_NAME, Manifest, manifest_digest

    directory = pathlib.Path(directory)
    if (directory / PLAN_NAME).exists() and not overwrite:
        raise TilefishManifestError(f"{str(directory)!r} holds the plan of one graph (plan.json): a plan set goes to another directory (or pass overwrite=True)")
    path, written = write_manifest_doc(built, directory / MANIFEST_NAME, cubins=True, merge=True, overwrite=overwrite, library=pinned)
    loaded = Manifest.load(path, instrumented=True)
    if loaded.digest != manifest_digest(written):
        raise TilefishManifestError(f"the manifest in {str(path.parent)!r} changed while the plan set was being exported")
    digests: "dict[str, str]" = {}
    for p in plans:
        final = p.with_manifest(f"sm_{cap}", loaded.digest)
        _plan.validate_doc(final._doc)
        _plan.check_against_manifest(final._doc, loaded)
        write_file(directory / _plan.plan_file_name(p.name), final.to_json().encode("utf-8"))
        digests[p.name] = final.digest
    index: "dict[str, Any]" = {
        "schema": _plan.PLAN_SET_SCHEMA,
        "tilefish": __version__,
        "arch": f"sm_{cap}",
        "manifest_digest": loaded.digest,
        "plans": dict(sorted(digests.items())),
    }
    index["set_digest"] = _plan.set_digest(index)
    _plan.validate_set(index)
    write_file(directory / _plan.PLAN_SET_NAME, (json.dumps(index, indent=2) + "\n").encode("utf-8"))
    doc = _plan.read_plan_set(directory, pin=index["set_digest"])
    back = {p.name: _plan.load_plan(directory, p.graph, pin=index["set_digest"], instrumented=True) for p in plans}
    return _plan.PlanSet(doc, directory, back)


def _existing_graph(path: pathlib.Path) -> "str | None":
    import json

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        name = doc.get("graph") if isinstance(doc, dict) else None
        return name if isinstance(name, str) else None
    except (OSError, ValueError):
        return None


# -- reference mode ---------------------------------------------------------------------------------------------------------------------


def _reference_dims(op: ExternOp, ins: "dict[str, Any]", outs: "Sequence[Value]", size: Any) -> "dict[str, int]":
    """What the reference of an extern is given besides its tensor inputs: nothing, unless a dimension only its ``Out`` parameters mention (an extern
    that writes a ``[B]`` tensor and reads a device scalar), which no input tells the reference; then its runtime dimension parameters (and stride
    scalars), by name, with the values a launch passes: bound from the inputs, and from the shapes the graph gives the ``Out`` tensors."""
    sig = op.signature
    if not sig.output_dims:
        return {}
    dims = bind_infos(op.name, sig, {name: tensor_info(name, t) for name, t in ins.items()})
    equations = [
        (p.name, axis, pattern, size(actual), str(p.type), "dimension")
        for p, v in zip(sig.tensor_outputs, outs)
        for axis, (pattern, actual) in enumerate(zip(p.type.shape, v.type.shape))  # type: ignore[union-attr]
    ]
    dims = solve_dims(op.name, equations, dims)
    return {p.name: eval_dim(p.dim, dims) for p in sig.dim_params}  # type: ignore[arg-type,misc]


def reference_run(graph: GraphOp, args: "tuple[Any, ...]", kwargs: "dict[str, Any]") -> Any:
    """``mode="reference"`` of a graph: the eager composition of the reference of each op, in torch, node by node.

    A view operation is torch's (``reshape``, ``permute``, ``select``, ``split``); the arithmetic of the body is interpreted op by op (as the
    fused regions of the plan, :mod:`tilefish._fusion`, whose IR is the body's); a call to a lowered ``@tilefish.kernel`` is its IR interpreted
    (:mod:`tilefish._reference`); a pinned cuBLASLt GEMM is an f32 matmul rounded once to its result's type; a hand-written extern is its ``reference=`` (a torch
    function given to ``@tilefish.extern``), and an extern that has none is refused. An in-place call on a view updates a copy of it: the IR says
    a chunk that is updated does not change its parent. Needs no CUDA: it runs where the tensors are. (A torch view of a tensor that an op then updates in place
    sees the update here, where the IR says a view keeps the contents it had: a plan refuses such a graph, so the two never disagree on a graph that can be planned.)
    """
    from ._fusion import planned_graph

    torch = runtime._torch()
    sig = graph.signature
    g = planned_graph(graph)  # the arithmetic of the body as the fused regions a plan launches, each interpreted op by op
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
            if kind is None and n.op == "copy":  # a new tensor of its own
                env[n.outputs[0]] = env[n.inputs[0]].contiguous().clone()  # type: ignore[index]
                continue
            if kind is None and n.op == "zeros":  # tilefish.zeros_like: a new tensor of zero bytes
                t = n.outputs[0].type
                dtype = getattr(torch, TORCH_DTYPE_NAMES[t.dtype.name].split(".")[1])  # type: ignore[union-attr]
                env[n.outputs[0]] = torch.zeros([size(d) for d in t.shape], dtype=dtype, device=getattr(anchor, "device", None))
                continue
            if kind is None and n.op == "store":  # `param[...] = value` into an InOut parameter: the caller's tensor gets the value
                target = env[n.inputs[0]]  # type: ignore[index]
                if n.inputs[1] is not n.inputs[0]:
                    target.copy_(env[n.inputs[1]])  # type: ignore[index]
                for o in n.outputs:
                    env[o] = target
                continue
            if kind is None:
                x = env[n.inputs[0]] if n.inputs else None  # type: ignore[index]  (iota reads no input)
                if n.op == "reshape":
                    results = [x.reshape([size(d) for d in n.attrs["shape"]])]
                elif n.op == "permute":
                    results = [x.permute(*n.attrs["axes"])]
                elif n.op == "select":
                    results = [x.select(n.attrs["axis"], n.attrs["index"])]
                elif n.op == "split":
                    sizes = [size(s) for s in n.attrs["sizes"]]
                    if "start" in n.attrs:  # a chunk of a split whose count depends on a dimension (tilefish.ops.Chunks): from its start
                        x = x.narrow(n.attrs["axis"], size(n.attrs["start"]), sum(sizes))
                    results = list(torch.split(x, sizes, dim=n.attrs["axis"]))
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
                from . import _conv, _library

                if op.impl is None:
                    raise TilefishRuntimeError(f"reference mode needs the library op '{op.name}' to have an impl (a pinned GEMM): it has only a signature")
                device = next(iter(ins.values())).device
                produced = [(_conv.reference_conv(op, ins) if _conv.is_conv(op) else _library.reference_of(op, ins)).to(device)]
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
                consts = n.attrs.get("consts") or {}
                given = {  # a prebuilt binary's fixed arguments per call: the values this call gives (an expression of the graph's dimensions evaluated)
                    p.name: size(consts[p.name]) if isinstance(consts[p.name], DimExpr) else consts[p.name]
                    for p in op_sig.consts
                    if p.default is MISSING and p.name in consts
                }
                res = op.reference(**ins, **_reference_dims(op, ins, n.outputs[: len(op_sig.tensor_outputs)], size), **given)
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
