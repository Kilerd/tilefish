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
arena is allocated once per runner and device and kept, the cuBLASLt workspace is a buffer of the arena, and a device scalar is written by a
stream-ordered copy (from a CUDA tensor) or an asynchronous copy from pinned memory (from an ``int`` or a CPU tensor). Nothing waits for the device.

**CUDA graphs.** A run under ``torch.cuda.graph(...)`` captures. The things that would make a plan fail to capture are, each refused with a message that
says so: (1) the arena, the binary of a step (``cuModuleLoadData``, ``cuFuncLoad``) and the cuBLASLt handle must exist, so the plan has to have run once
(or :func:`prepare` called) outside the capture; (2) the outputs must not be allocated inside the capture unless the graph owns them (pass ``outputs=``);
(3) a device scalar passed as an ``int`` or a CPU tensor would be written by a host copy, which cannot be captured: pass a CUDA tensor, which the captured
copy reads at every replay (write it between replays to change the value); (4) the externals must be the same tensors at every replay (a graph holds
addresses); (5) the stream must be the one the capture uses (torch's current stream; the capture is of that stream's work only); (6) a plan is not
reentrant: two runs of one plan on two streams at once share the arena. Everything else a run does is enqueued launches, which capture.
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
    dims_by_name,
    eval_dim,
    tensor_info,
)
from .decorators import ExternOp, GraphOp, KernelOp, LibraryOp
from .errors import TilefishManifestError, TilefishPreconditionError, TilefishRuntimeError
from ._plan import PLAN_NAME, Bound, Plan, plan as make_plan
from .trace import Value

_lock = threading.Lock()
_default_plans: "weakref.WeakKeyDictionary[GraphOp, Plan]" = weakref.WeakKeyDictionary()
_loaded: "dict[tuple, tuple[Plan, PlanRunner]]" = {}  # a loaded plan directory, until a file in it changes
_jit_runners: "weakref.WeakKeyDictionary[Plan, PlanRunner]" = weakref.WeakKeyDictionary()


def clear_caches() -> None:
    """Forget every loaded plan and every runner, and free their arenas (for tests, and to give the memory back)."""
    with _lock:
        _loaded.clear()
        _jit_runners.clear()
        _default_plans.clear()


# -- what a runner asks of torch and the driver ---------------------------------------------------------------------------------------------


class Backend:
    """What a runner needs from torch and the CUDA driver. :class:`TorchBackend` is the real one; a test passes another that records what it is
    asked to do, which is how the order, the pointers and "nothing is launched when a precondition fails" are tested without a GPU."""

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


class TorchBackend(Backend):
    def __init__(self) -> None:
        self.torch = runtime._torch()

    def device_index(self, device: Any) -> int:
        return runtime._device_index(self.torch, str(device))

    def capability(self, index: int) -> int:
        return runtime._capability(self.torch, index)

    def capturing(self, index: int) -> bool:
        return bool(self.torch.cuda.is_current_stream_capturing())

    def scope(self, index: int) -> Any:
        return self.torch.cuda.device(index)

    def stream(self, index: int) -> int:
        return int(self.torch.cuda.current_stream(index).cuda_stream)

    def alloc_arena(self, nbytes: int, index: int) -> "tuple[Any, int]":
        t = self.torch.empty(nbytes, dtype=self.torch.uint8, device=f"cuda:{index}")
        return t, int(t.data_ptr())

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
    handle: Any
    ptr: int
    nbytes: int
    lock: threading.Lock


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


class PlanRunner:
    """Runs one plan (:class:`~tilefish._plan.Plan`) with binaries from a provider, on a backend. See the module docstring."""

    def __init__(self, plan: Plan, provider: Any, backend: "Backend | None" = None) -> None:
        self.plan = plan
        self.provider = provider
        self._backend = backend
        self._doc = plan._doc
        self._buffers = {b["name"]: b for b in self._doc["buffers"]}
        self._rts: "dict[tuple, _BucketRT]" = {}
        self._arenas: "dict[int, _Arena]" = {}
        self._lock = threading.RLock()
        self.has_library = any(s["kind"] == "library" for s in self._doc["buckets"][0]["steps"])

    @property
    def backend(self) -> Backend:
        """The backend (torch and the driver, made when first needed: a runner can be made, and a plan loaded, on a machine without torch)."""
        if self._backend is None:
            self._backend = TorchBackend()
        return self._backend

    # -- preparing: everything that may allocate or wait ------------------------------------------------------------------------

    def prepare(self, index: int, buckets: "Sequence[Mapping[str, int]] | None" = None) -> None:
        """Allocate the arena of device ``index``, build or load the binary of every step of the given bucket combinations (all of them by default),
        load the functions into the device and create the cuBLASLt handle. Everything a run does that cannot be captured in a CUDA graph is done here
        (and a run does it for itself the first time, outside a capture)."""
        be = self.backend
        if be.capturing(index):
            raise TilefishRuntimeError("a plan cannot be prepared while a CUDA graph is being captured: its arena, binaries and handles are made outside the capture")
        with self._lock:
            self._arena(index)
            for values in buckets if buckets is not None else self.plan.buckets():
                self._bucket(dict(values), index)
            be.warm(index, self.has_library)

    def _arena(self, index: int) -> _Arena:
        arena = self._arenas.get(index)
        if arena is not None:
            return arena
        be = self.backend
        if be.capturing(index):
            raise TilefishRuntimeError(
                f"the arena of the plan of graph '{self.plan.name}' has not been allocated on device {index} yet, and memory is not allocated while a CUDA "
                "graph is being captured (the graph would own it): run the plan once outside the capture, or call tilefish.runtime.prepare"
            )
        with self._lock:
            arena = self._arenas.get(index)
            if arena is None:
                nbytes = self._doc["arena_bytes"]
                handle, ptr = be.alloc_arena(nbytes, index) if nbytes else (None, 0)
                if nbytes and ptr % _plan.ARENA_ALIGNMENT:
                    raise TilefishRuntimeError(f"the arena was allocated at {ptr:#x}, which is not aligned to {_plan.ARENA_ALIGNMENT} bytes")
                arena = self._arenas[index] = _Arena(handle, ptr, nbytes, threading.Lock())
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
                steps.append(_StepRT(bound, None, (), built.algo, ws))
            else:
                built = self.provider.kernel(callee, bound.spec, index)
                if built.spec.block != bound.spec.block or built.num_warps != bound.spec.num_warps:
                    raise TilefishRuntimeError(f"the binary of '{bound.spec.stem}' disagrees with its spec about num_warps")
                order = tuple((a.kind, a.name) for a in bound.spec.args if a.kind in ("pointer", "scalar"))
                steps.append(_StepRT(bound, be.load(built, index), order, None, None))
        be.warm(index, self.has_library)
        return _BucketRT(dict(values), shapes, steps)

    # -- running ----------------------------------------------------------------------------------------------------------------

    def run(self, args: "tuple[Any, ...]", kwargs: "dict[str, Any]", outputs: Any = None) -> Any:
        be = self.backend
        graph = self.plan.graph
        sig = graph.signature
        doc = self._doc
        # the arguments: a DevScalar may be passed as an int (or a CPU tensor), which becomes a tensor of one element
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
                bound.arguments[p.name] = v = be.devscalar_tensor(v, p.type.dtype.name)  # type: ignore[union-attr]
            sources[p.name] = v
        call = bind_call(graph.name, sig, bound.args, bound.kwargs)
        check_dim_constraints(call.dims, buckets=False)
        by_name = dims_by_name(call.dims)
        values: "dict[str, int]" = {}
        for n, allowed in doc["dims"].items():
            if n not in by_name:
                raise TilefishPreconditionError(f"the dimension {n} of graph '{graph.name}' is not determined by the arguments", kind="shape", arg=n)
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
        capturing = be.capturing(index)
        rt = self._bucket(values, index)
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
        self._check_externals(bases, written, rt)
        # every precondition of every step, before anything is launched
        checked = [st.bound.check(bases) for st in rt.steps]
        slots = []
        for b in doc["buffers"]:
            if b["role"] == "devscalar":
                source, on_device = be.devscalar_source(sources[b["name"]], b["dtype"], index)
                if capturing and not on_device:
                    raise TilefishRuntimeError(
                        f"the device scalar '{b['name']}' was passed as a host value (an int, a number or a CPU tensor), which is written by a host copy that "
                        "cannot be captured in a CUDA graph (and a captured copy of it would freeze the value): pass a CUDA tensor of one element, which the "
                        "captured copy reads at every replay, and write it between replays"
                    )
                slots.append((b, source, on_device))
        with arena.lock, be.scope(index):
            stream = be.stream(index)
            for b, source, on_device in slots:
                be.write_devscalar(arena.handle, b["offset"], b["dtype"], source, on_device)
            for st, real in zip(rt.steps, checked):
                if st.algo is not None:
                    names = [a for a in st.bound.names]
                    a_ptr, b_ptr = real[names[0]].ptr, real[names[1]].ptr
                    bias = real[names[2]].ptr if len(names) == 4 else None
                    be.gemm(st.bound.spec, st.algo, a_ptr, b_ptr, bias, real[names[-1]].ptr, bases[st.workspace], stream, index)  # type: ignore[index]
                else:
                    launch_args = [("pointer", real[n].ptr) if kind == "pointer" else ("scalar", st.bound.scalars[n]) for kind, n in st.order]
                    be.launch(st.loaded, st.bound.grid, launch_args, stream)
        results = [t for _, t in sorted(made, key=lambda p: p[0])]
        return runtime._result(results)

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

    def _check_externals(self, bases: "Mapping[str, int]", written: "Sequence[str]", rt: _BucketRT) -> None:
        """A tensor the plan writes (an output or an in-place tensor) must not overlap any other external: the steps are ordered, but the plan is not
        written for two names for one memory."""
        ranges = {}
        for b in self._doc["buffers"]:
            if b["role"] in ("input", "weight", "inout", "output"):
                dtype, shape = rt.shapes[b["name"]]
                ranges[b["name"]] = (bases[b["name"]], bases[b["name"]] + math.prod(shape) * _plan.ITEMSIZE[dtype])
        for w in written:
            lo, hi = ranges[w]
            for other, (olo, ohi) in ranges.items():
                if other != w and lo < ohi and olo < hi:
                    raise TilefishPreconditionError(
                        f"'{w}' is written by the plan but overlaps '{other}': the tensors of a plan must not alias when one of them is written",
                        kind="alias",
                        arg=w,
                    )


# -- tilefish.run of a graph ---------------------------------------------------------------------------------------------------------------


def _default_plan(graph: GraphOp) -> Plan:
    with _lock:
        p = _default_plans.get(graph)
    if p is None:
        p = make_plan(graph)
        with _lock:
            _default_plans[graph] = p
    return p


def _jit_runner(p: Plan) -> PlanRunner:
    with _lock:
        r = _jit_runners.get(p)
        if r is None:
            r = _jit_runners[p] = PlanRunner(p, JitProvider())
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
    key = (*_directory_stamp(p), id(graph))
    with _lock:
        found = _loaded.get(key)
    if found is not None:
        loaded, runner = found
        if pin is not None and loaded.digest != pin:
            raise TilefishManifestError(
                f"the plan in {str(loaded.directory)!r} has digest {loaded.digest}, but the pin is {pin}: it is not the plan that was pinned "
                "(it was changed, or it is another export)"
            )
        if (pin is not None) != runner.provider.pinned:
            found = None  # a load with the other trust anchor: the manifest's `pinned` differs, so load again
        else:
            return runner
    loaded = _plan.load_plan(p, graph, pin=pin, trust_directory=trust_directory)
    runner = PlanRunner(loaded, ReplayProvider(loaded.manifest, pinned=pin is not None))
    with _lock:
        _loaded[key] = (loaded, runner)
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
    a chunk that is updated does not change its parent. Needs no CUDA: it runs where the tensors are.
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
                bound.arguments[p.name] = torch.tensor(v, dtype=dtype, device=getattr(anchor, "device", None))
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
            n_returned = len(op_sig.returns)
            if isinstance(op, LibraryOp):
                from . import _library

                if op.impl is None:
                    raise TilefishRuntimeError(f"reference mode needs the library op '{op.name}' to have an impl (a pinned GEMM): it has only a signature")
                device = next(iter(ins.values())).device
                produced = [_library.reference_gemm(ins).to(device)]
            elif isinstance(op, KernelOp):
                res = runtime._run_reference(runtime._target(op), (), dict(ins))
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
