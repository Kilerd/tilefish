"""Running kernels from Python in three modes: ``jit``, ``replay`` and ``reference``.

``tilefish.run(op, *args, mode="jit", **kwargs)`` (also ``op.run(...)``) runs a ``@tilefish.kernel`` (lowered
to Triton, see :mod:`tilefish.lower`) or an ``@tilefish.extern`` whose Triton function is importable. The
arguments are torch tensors and scalars matched to the signature the way a call is matched in a trace:
inputs and ``InOut`` tensors are passed (an ``InOut`` tensor is updated in place), ``Out`` tensors are
allocated, ``const`` parameters are compiled in (a call may restate their value, not change it), and the
result is what the call returns when traced: a kernel's returned values, then its ``Out`` tensors; an
extern's ``Out`` tensors; ``None``, one tensor or a tuple.

=============  ============================================================================================
mode           what runs
=============  ============================================================================================
``jit``        the current source, built through its compile spec (``spec.ast_source_inputs()``, the path
               of :func:`tilefish.aot.build`) in memory, cached by digest, and launched through the CUDA
               driver
``replay``     the prebuilt cubins of a manifest directory (``manifest=``, written by
               :func:`export` or ``write_manifest(..., cubins=True)``), loaded with ``cuModuleLoadData``
               and launched through the same driver path: exactly what a C++ or Rust host does
``reference``  the traced IR interpreted with torch (:mod:`tilefish._reference`); a kernel only, an
               extern has no IR to interpret
=============  ============================================================================================

``jit`` and ``replay`` launch the same way, so they run the same machine code whenever the cubin was built from
the same source with the same toolchain: a cubin that ``jit`` builds has the ``sha256_code`` of the one in the
manifest, and the outputs are byte-identical (both are checked on a GPU by the tests). Triton's own JIT launcher is
not a mode: it applies value specializations (an ``i32`` equal to 1 becomes a constant, divisibility by 16 is
assumed wherever it happens to hold) that make its machine code differ from the spec's binary. Neither mode needs a
C compiler (Triton's JIT launcher does, to build a stub); ``replay`` needs neither Triton nor ``ptxas``, only a
manifest directory and the CUDA driver.

``replay`` loads a manifest when it is first used, and refuses a cubin whose ``sha256`` or ``sha256_code`` is not
the manifest's, and a manifest entry that is not what the compile spec of the operator asks for (ABI, grid,
block, constants, preconditions, options, declaration digest): a binary never runs under assumptions it was not
compiled for. Before ``jit`` and ``replay`` launch, every precondition of the binary is checked (dtypes, static
sizes, ``divisible_by``, 16-byte aligned pointers, contiguity, aliasing, ``max_elements`` of a generated kernel);
a bucketed dimension must equal one bucket value exactly, there is no padding. ``reference`` checks the types, not
the alignment or the buckets. Everything here imports torch, Triton and cuda-python only when it runs.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import importlib.util
import math
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Iterator

from . import _cuda
from ._build import BuiltKernel, build, resolve_triton, write_manifest
from ._launch import (
    TORCH_DTYPE_NAMES,
    TensorInfo,
    bind_call,
    check_consts,
    check_dim_constraints,
    check_i32,
    check_no_alias,
    check_spec_preconditions,
    eval_dim,
    eval_grid,
    select_spec,
    tensor_info,
    tf_dtype_name,
)
from .aot import CompileSpec, compile_specs
from .decorators import ExternOp, KernelOp, Op
from .dims import Dim, DimExpr
from ._manifest import MANIFEST_NAME, Manifest
from .errors import (
    TilefishBuildError,
    TilefishError,
    TilefishManifestError,
    TilefishPreconditionError,
    TilefishRuntimeError,
    TilefishTypeError,
)
from .lower import LoweredKernel, lower
from .signature import MISSING, Param, Signature

MODES = ("jit", "replay", "reference")

_lock = threading.Lock()
_builds: "dict[tuple, BuiltKernel]" = {}
_lowered: "weakref.WeakKeyDictionary[KernelOp, LoweredKernel]" = weakref.WeakKeyDictionary()
_specs: "weakref.WeakKeyDictionary[ExternOp, list[CompileSpec]]" = weakref.WeakKeyDictionary()  # an op is immutable
_manifests: "dict[tuple, Manifest]" = {}  # a loaded manifest directory, until a file in it changes
_modules: "dict[str, Any]" = {}  # sha256 of a lowered source -> the imported JITFunction
_gen_root: "list[pathlib.Path]" = []


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:
        raise TilefishRuntimeError("running a kernel needs torch: pip install 'tilefish[gpu]'") from e
    return torch


def clear_caches() -> None:
    """Forget every built kernel and unload every loaded module (for tests and for rebuilding after a toolchain change)."""
    with _lock:
        _builds.clear()
        _manifests.clear()
    try:
        _cuda.unload_all()
    except TilefishError:
        pass


# -- what is being run ----------------------------------------------------------------------------------------------------


@dataclass
class _Target:
    name: str
    public: Signature  # the signature a caller sees
    kernel: "KernelOp | None"  # the traced kernel, for reference mode and for the order of the results
    extern: "ExternOp | None"  # what gets launched (the lowered stub of a kernel), resolved lazily
    lowered: "LoweredKernel | None"
    specs: "list[CompileSpec] | None" = None

    def launch_extern(self) -> ExternOp:
        if self.extern is None:
            assert self.kernel is not None
            self.lowered = _lower_cached(self.kernel)
            self.extern = self.lowered.extern
        return self.extern

    def compile_specs(self) -> "list[CompileSpec]":
        if self.specs is None:
            extern = self.launch_extern()
            with _lock:
                specs = _specs.get(extern)
            if specs is None:
                specs = compile_specs(extern)
                with _lock:
                    _specs[extern] = specs
            self.specs = specs
        return self.specs

    def jit_function(self, spec: CompileSpec) -> Any:
        self.launch_extern()
        if self.lowered is not None:
            return load_lowered(self.lowered)
        try:
            return resolve_triton(spec.triton)
        except TilefishBuildError as e:
            raise TilefishRuntimeError(str(e)) from e


def _lower_cached(kernel: KernelOp) -> LoweredKernel:
    with _lock:
        low = _lowered.get(kernel)
    if low is None:
        low = lower(kernel)
        with _lock:
            _lowered[kernel] = low
    return low


def _target(op: Any) -> _Target:
    if isinstance(op, LoweredKernel):
        if op.kernel is None:
            raise TilefishRuntimeError(f"the lowered kernel '{op.name}' does not know the kernel it came from")
        return _Target(op.name, op.kernel.signature, op.kernel, op.extern, op)
    if isinstance(op, KernelOp):
        return _Target(op.name, op.signature, op, None, None)
    if isinstance(op, ExternOp):
        return _Target(op.name, op.signature, None, op, None)
    if isinstance(op, Op):
        raise TilefishRuntimeError(
            f"cannot run the {op.kind} op '{op.name}': only @tilefish.kernel and @tilefish.extern run for now "
            "(library operators and graphs have no runtime yet)"
        )
    raise TilefishRuntimeError(f"tilefish.run takes a kernel or an extern op, got {op!r}")


def load_lowered(lowered: LoweredKernel) -> Any:
    """Write the generated source of a lowered kernel to a scratch directory, import it, and return its ``JITFunction``.

    The module is imported from a file because Triton reads the source of a JIT function from disk. The
    scratch directory lives until the process exits; the result is cached by the sha256 of the source.
    """
    key = hashlib.sha256(lowered.source.encode("utf-8")).hexdigest()
    with _lock:
        fn = _modules.get(key)
        if fn is not None:
            return fn
        if not _gen_root:
            root = pathlib.Path(tempfile.mkdtemp(prefix="tilefish-gen-"))
            atexit.register(shutil.rmtree, root, True)
            _gen_root.append(root)
        path = lowered.write(_gen_root[0] / key[:16])
        name = f"{lowered.module.replace('.', '_')}__{key[:12]}"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        fn = getattr(module, lowered.fn_name)
        _modules[key] = fn
        return fn


# -- one call, prepared ---------------------------------------------------------------------------------------------------


@dataclass
class _Prepared:
    target: _Target
    spec: CompileSpec
    device: str
    tensors: "dict[str, Any]"  # every pointer parameter of the launched kernel
    scalars: "dict[str, int]"
    grid: "tuple[int, ...]"
    results: "list[Any]"


def _alloc(torch: Any, p: Param, dims: Any, device: Any, where: str) -> Any:
    assert p.type is not None
    shape = []
    for d in p.type.shape:
        v = eval_dim(d, dims)
        if v is None:
            raise TilefishRuntimeError(f"cannot allocate {where}: dimension {d} is not determined by the arguments")
        shape.append(v)
    dtype = getattr(torch, TORCH_DTYPE_NAMES[tf_dtype_name(p.type.dtype, where)].split(".")[1])
    return torch.empty(shape, dtype=dtype, device=device)


def _device_of(infos: "dict[str, TensorInfo]", tensors: "dict[str, Any]") -> Any:
    devices = {i.device for i in infos.values()}
    if len(devices) > 1:
        raise TilefishPreconditionError(
            f"the tensors are on different devices ({', '.join(sorted(devices))}); every tensor of a call must be on one device",
            kind="device",
        )
    if not tensors:
        raise TilefishRuntimeError("a call needs at least one input tensor")
    return next(iter(tensors.values())).device


def _declared_consts(sig: Signature, explicit: "dict[str, Any]") -> None:
    for p in sig.consts:
        if p.name in explicit and p.default is not MISSING and not isinstance(p.default, DimExpr):
            v = explicit[p.name]
            if not (v == p.default and isinstance(v, bool) == isinstance(p.default, bool)):
                raise TilefishTypeError(
                    f"constexpr '{p.name}' is compiled in as {p.default!r}; it cannot be changed to {v!r} at a call "
                    "(change the default in the declaration)"
                )


def _extern_names(target: _Target, extern: ExternOp) -> "dict[str, str]":
    """Public parameter name -> its name in the launched kernel.

    A lowered kernel's stub lists the kernel's own parameters first, in order, but renames those that
    collide with names the generated module uses (``tl``, ``grid``, ...).
    """
    if target.kernel is None:
        return {p.name: p.name for p in extern.signature.params}
    return {p.name: q.name for p, q in zip(target.kernel.signature.params, extern.signature.params)}


def _prepare(target: _Target, args: "tuple[Any, ...]", kwargs: "dict[str, Any]") -> _Prepared:
    torch = _torch()
    extern = target.launch_extern()
    call = bind_call(target.name, target.public, args, kwargs)
    check_dim_constraints(call.dims)
    specs = target.compile_specs()
    spec = select_spec(target.name, specs, call.dims)
    names = _extern_names(target, extern)
    check_consts(spec, {names[k]: v for k, v in call.consts.items()})
    device = _device_of(call.infos, call.tensors)
    if getattr(device, "type", None) != "cuda":
        raise TilefishRuntimeError(f"jit and replay run on a CUDA device; the tensors are on {device}")

    # the tensors of the launched kernel, by its parameter names: the supplied ones, and every Out (allocated here)
    tensors = {names[k]: v for k, v in call.tensors.items()}
    out_params = [p for p in extern.signature.params if p.kind == "tensor" and p.access == "out"]
    allocated = {p.name: _alloc(torch, p, call.dims, device, f"Out parameter '{p.name}'") for p in out_params}
    tensors.update(allocated)
    if target.kernel is not None:
        # a kernel returns its returned values, then its Out tensors; the stub lists the kernel's own
        # parameters (the Out ones among them), then one output per returned value
        user = len(target.kernel.signature.params)
        returned = [p for p in extern.signature.params[user:] if p.kind == "tensor" and p.access == "out"]
        results = [allocated[p.name] for p in returned]
        results += [allocated[names[p.name]] for p in target.kernel.signature.tensor_outputs]
    else:
        results = [allocated[p.name] for p in out_params]

    infos = {name: tensor_info(name, t) for name, t in tensors.items()}
    by_name = {p.name: p for p in extern.signature.params}
    scalars: "dict[str, int]" = {}
    for a in spec.args:
        if a.kind != "scalar":
            continue
        p = by_name[a.name]
        assert p.dim is not None
        v = eval_dim(p.dim, call.dims)
        if a.count_of is not None:  # a generated kernel's n: the element count of that tensor argument
            count = math.prod(infos[a.count_of].shape)
            if v is not None and v != count:
                raise TilefishPreconditionError(
                    f"runtime argument '{a.name}' is {p.dim} = {v}, but the tensor '{a.count_of}' has {count} elements",
                    kind="shape",
                    arg=a.name,
                )
            if v is None and isinstance(p.dim, Dim):
                call.dims[p.dim] = count  # the grid may be written over it
            v = count
        elif v is None:
            raise TilefishRuntimeError(f"cannot determine the runtime argument '{a.name}' ({p.dim}) from the tensor shapes")
        scalars[a.name] = v
    check_dim_constraints(call.dims)
    check_i32(scalars)
    written = {p.name for p in extern.signature.params if p.kind == "tensor" and p.access != "in"}
    check_no_alias(infos, written)
    check_spec_preconditions(spec, infos, scalars, call.dims)
    assert extern.grid is not None
    grid = eval_grid(spec, extern.grid, call.dims)
    return _Prepared(target, spec, str(device), tensors, scalars, grid, results)


def _launch_values(prep: _Prepared) -> "list[tuple[str, str, Any]]":
    """The kernel's arguments in declaration order: ``(kind, name, value)`` with a tensor, an int or a constexpr value."""
    out = []
    for a in prep.spec.args:
        if a.kind == "pointer":
            out.append(("pointer", a.name, prep.tensors[a.name]))
        elif a.kind == "scalar":
            out.append(("scalar", a.name, prep.scalars[a.name]))
        else:
            out.append(("constexpr", a.name, a.value))
    return out


# -- the modes ------------------------------------------------------------------------------------------------------------


def _result(results: "list[Any]") -> Any:
    if not results:
        return None
    return results[0] if len(results) == 1 else tuple(results)


def _device_index(torch: Any, device: str) -> int:
    d = torch.device(device)
    return d.index if d.index is not None else torch.cuda.current_device()


_options = threading.local()


@contextlib.contextmanager
def build_options(**options: Any) -> "Iterator[None]":
    """Extra Triton compile options for the ``jit`` builds made inside the block (on this thread).

    An experiment, not a mode: ``with build_options(enable_fp_fusion=False): run(op, ..., mode="jit")`` runs a
    binary compiled without FMA contraction. (A lowered kernel asks for that itself with
    ``@tilefish.kernel(fp_fusion=False)``; this is for an ``extern``, whose spec says nothing about it.) The options
    are part of the cache key and are recorded in the :class:`~tilefish.aot.BuiltKernel`, and they make the build
    differ from the compile spec, so a manifest written from it is refused by ``replay``.
    """
    previous = getattr(_options, "value", {})
    _options.value = {**previous, **options}
    try:
        yield
    finally:
        _options.value = previous


def _capability(torch: Any, device_index: int) -> int:
    major, minor = torch.cuda.get_device_capability(device_index)
    return major * 10 + minor


def build_for_device(spec: CompileSpec, fn: Any, device_index: int) -> BuiltKernel:
    """The (cached, in memory) build of ``spec`` for the architecture of device ``device_index``."""
    torch = _torch()
    cap = _capability(torch, device_index)
    extra = dict(getattr(_options, "value", {}))
    key = (spec.stem, cap, spec.declaration_digest, spec._fingerprint(), getattr(fn, "cache_key", None), tuple(sorted(extra.items())))
    with _lock:
        built = _builds.get(key)
    if built is None:
        built = build(spec, arch=cap, fn=fn, options=extra)
        with _lock:
            _builds[key] = built
    return built


def _launch_built(prep: _Prepared, built: BuiltKernel, index: int) -> None:
    """Launch a binary through the CUDA driver on torch's current stream, with the ABI of the manifest."""
    torch = _torch()
    if built.spec.block != prep.spec.block or built.num_warps != prep.spec.num_warps:
        raise TilefishRuntimeError(f"the binary of '{prep.spec.stem}' disagrees with its spec about num_warps")
    args: "list[tuple[str, int]]" = []
    for kind, _, v in _launch_values(prep):
        if kind == "pointer":
            args.append(("pointer", int(v.data_ptr())))
        elif kind == "scalar":
            args.append(("scalar", int(v)))
    with torch.cuda.device(index):
        loaded = _cuda.load(built, index)
        stream = torch.cuda.current_stream(index).cuda_stream
        loaded.launch(prep.grid, args, stream)


def _run_jit(prep: _Prepared) -> None:
    """Build the current source through the compile spec, in memory, and launch that binary."""
    torch = _torch()
    index = _device_index(torch, prep.device)
    fn = prep.target.jit_function(prep.spec)
    _launch_built(prep, build_for_device(prep.spec, fn, index), index)


def _load_manifest(manifest: Any) -> Manifest:
    """A :class:`Manifest` from a directory, a manifest file or a loaded manifest (a directory is cached until a file in it changes)."""
    if isinstance(manifest, Manifest):
        return manifest
    try:
        path = pathlib.Path(manifest)
    except TypeError:
        raise TilefishRuntimeError(f"manifest= takes a directory or a Manifest, got {manifest!r}") from None
    directory = path if path.is_dir() else path.parent
    try:
        stamp = tuple(sorted((e.name, e.stat().st_mtime_ns, e.stat().st_size) for e in os.scandir(directory) if e.is_file()))
    except OSError as e:
        raise TilefishManifestError(f"cannot read the manifest directory {str(directory)!r}: {e}") from e
    key = (str(path.resolve()), stamp)
    with _lock:
        loaded = _manifests.get(key)
    if loaded is None:
        loaded = Manifest.load(path)
        with _lock:
            _manifests[key] = loaded
    return loaded


def _run_replay(prep: _Prepared, manifest: Manifest) -> None:
    """Launch the prebuilt cubin of the manifest for this spec and this device's architecture."""
    torch = _torch()
    index = _device_index(torch, prep.device)
    built = manifest.built_kernel(prep.spec, f"sm_{_capability(torch, index)}")
    _launch_built(prep, built, index)


def _run_reference(target: _Target, args: "tuple[Any, ...]", kwargs: "dict[str, Any]") -> Any:
    from ._reference import interpret

    if target.kernel is None:
        raise TilefishRuntimeError(
            f"reference mode interprets the IR of a @tilefish.kernel; the extern op '{target.name}' wraps a hand-written "
            "Triton kernel and has no IR (use mode='jit' or 'replay')"
        )
    torch = _torch()
    kernel = target.kernel
    call = bind_call(target.name, kernel.signature, args, kwargs)
    check_dim_constraints(call.dims, buckets=False)  # the IR has no binaries, so no bucket to be exact about
    _declared_consts(kernel.signature, call.consts)
    device = _device_of(call.infos, call.tensors)
    tensors = dict(call.tensors)
    outs = []
    for p in kernel.signature.tensor_outputs:
        t = _alloc(torch, p, call.dims, device, f"Out parameter '{p.name}'")
        tensors[p.name] = t
        outs.append(t)
    written = {p.name for p in kernel.signature.params if p.kind == "tensor" and p.access != "in"}
    check_no_alias({n: tensor_info(n, t) for n, t in tensors.items()}, written)
    returned = interpret(kernel, tensors)
    return _result(returned + outs)


def export(ops: Any, directory: "str | os.PathLike[str]", *, arch: "int | str | None" = None) -> pathlib.Path:
    """Build every binary of one or several kernels and externs and write them as a manifest directory.

    For each compile spec of each operator (one per bucket) the cubin is built for ``arch`` (default: the
    architecture of the current CUDA device) and ``<directory>/manifest.json`` is written together with the
    cubins, which is what ``tilefish.run(..., mode="replay", manifest=directory)`` loads, and what a C++ or Rust
    host reads. Returns the path of the manifest.
    """
    items = list(ops) if isinstance(ops, (list, tuple)) else [ops]
    if arch is None:
        torch = _torch()
        cap = _capability(torch, torch.cuda.current_device())
    else:
        from ._build import parse_arch

        cap = parse_arch(arch)
    built: "list[BuiltKernel]" = []
    for op in items:
        target = _target(op)
        for spec in target.compile_specs():
            built.append(build(spec, arch=cap, fn=target.jit_function(spec)))
    return write_manifest(built, pathlib.Path(directory) / MANIFEST_NAME, cubins=True)


def run(op: Any, /, *args: Any, mode: str = "jit", manifest: Any = None, **kwargs: Any) -> Any:
    """Run a kernel or an extern op on tensors. See the module docstring for the three modes.

    ``op`` is a ``@tilefish.kernel`` (or its :class:`~tilefish.lower.LoweredKernel`) or an ``@tilefish.extern`` whose
    Triton function can be imported. ``mode`` is ``"jit"``, ``"replay"`` or ``"reference"``. ``manifest`` is the
    directory (or the :class:`~tilefish.aot.Manifest`) that ``"replay"`` loads its cubins from, and is required
    there and nowhere else. ``mode`` and ``manifest`` are the only keywords that are not arguments of ``op`` (an
    operator with a parameter of one of those names takes it positionally). Raises
    :class:`~tilefish.errors.TilefishPreconditionError` when an argument breaks a precondition of the binary,
    :class:`~tilefish.errors.TilefishManifestError` when the manifest is not what was built or not what the
    operator asks for, and :class:`~tilefish.errors.TilefishRuntimeError` when the call cannot run.
    """
    if mode not in MODES:
        raise TilefishRuntimeError(f"unknown mode {mode!r}: use one of {', '.join(MODES)}")
    if mode == "replay" and manifest is None:
        raise TilefishRuntimeError(
            "replay runs prebuilt cubins from a manifest directory: pass manifest=<directory> (write one with "
            "tilefish.runtime.export(op, directory) or tilefish.aot.write_manifest(built, path, cubins=True))"
        )
    if mode != "replay" and manifest is not None:
        raise TilefishRuntimeError(f"manifest= is for mode='replay'; mode {mode!r} builds or interprets the kernel itself")
    target = _target(op)
    if mode == "reference":
        return _run_reference(target, args, kwargs)
    prep = _prepare(target, args, kwargs)
    if mode == "jit":
        _run_jit(prep)
    else:
        _run_replay(prep, _load_manifest(manifest))
    return _result(prep.results)
