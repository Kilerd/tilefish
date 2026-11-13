"""Running kernels from Python in three modes: ``jit``, ``replay`` and ``reference``.

``tilefish.run(op, *args, mode="jit", **kwargs)`` (also ``op.run(...)``) runs a ``@tilefish.kernel`` (lowered
to Triton, see :mod:`tilefish.lower`), an ``@tilefish.extern`` whose Triton function is importable or which declares a prebuilt PTX or
cubin (:mod:`tilefish._binary`), or a ``@tilefish.library`` op with an ``impl`` (a pinned cuBLASLt GEMM, :mod:`tilefish._library`: ``jit`` pins
the algorithm in memory, ``replay`` runs the one a manifest recorded, ``reference`` is torch). A tensor argument of a ``Strided[...]``
parameter may be a view (its strides and its pointer go to the kernel), and one of a ``DevScalar`` parameter is a tensor of one element. The
arguments are torch tensors and scalars matched to the signature the way a call is matched in a trace:
inputs and ``InOut`` tensors are passed (an ``InOut`` tensor is updated in place), ``Out`` tensors are
allocated, ``const`` parameters are compiled in (a call may restate their value, not change it), and the
result is what the call returns when traced: a kernel's returned values, then its ``Out`` tensors; an
extern's ``Out`` tensors; ``None``, one tensor or a tuple.

=============  ============================================================================================
mode           what runs
=============  ============================================================================================
``jit``        the current kernel, built through its compile spec (``spec.ast_source_inputs()``, the path
               of :func:`tilefish.aot.build`) in memory, cached by the digest of its Triton IR (in memory,
               and on disk by tilefish's own build cache), and launched through the CUDA driver
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
C compiler (Triton's JIT launcher does, to build a stub). ``replay`` of a generated kernel needs neither Triton nor
``ptxas``, only a manifest directory and the CUDA driver.

A ``@tilefish.graph`` runs through its launch plan (:mod:`tilefish._plan`, :mod:`tilefish._planrun`, ``docs/plan-schema-1.md``): ``jit`` builds the plan's binaries in memory, ``replay`` runs the
plan of a directory that :func:`export` wrote (``plan=``, with ``pin=`` the plan's digest, which pins the manifest and so every binary), ``reference`` composes the references of its
ops in torch. Every step's preconditions are checked before the first launch, a run allocates and waits for nothing, and a replay can be captured in a CUDA graph.

What ``replay`` checks. It loads the manifest when it is first used (and again when a file in the directory changes)
and refuses

* a cubin whose ``sha256`` or ``sha256_code`` is not the manifest's, a file name that leaves the directory, a kernel
  that uses shared memory with ``shared = 0``, and a launch that does not fit the device (shared memory, grid);
* a manifest entry that is not what the compile spec of the operator asks for (ABI, grid, block, constants,
  preconditions, options, declaration digest): a binary never runs under assumptions it was not compiled for;
* a stale binary. **The identity of a binary is the Triton IR its kernel compiles to** (:mod:`tilefish._ir`): Triton's
  frontend is run on the spec's source in this process (a few milliseconds, no compile) and the TTIR it builds, without locations
  or names, is hashed. Every manifest entry records that digest (``source_digest``, kind ``"tilefish.triton-ir.v1"``), and when Triton
  is importable ``replay`` recomputes it from the kernel *as this process holds it* and refuses a mismatch ("stale"), pinned or not.
  Whatever the frontend reads is in the IR, so no route a kernel reads a constant by escapes it: a module attribute, ``getattr``
  (with a literal or a computed name), a class attribute, a closure variable, a ``triton.constexpr_function`` that reads the
  environment, a ``tl.constexpr`` of an object with hidden state, a helper two modules away, a user module that is named like a
  standard library one. Comments, blank lines, the path of the file and local variable names are not in it. A generated kernel's
  spec also carries ``source_sha256``, the sha256 of the generated Triton source, so a kernel redeclared with another body but the
  same name and signature does not match its old manifest, and that needs no Triton. Where the IR of a hand-written ``extern`` cannot
  be recomputed (the module does not import without Triton, or the Triton is not the version that built the entry) ``replay``
  refuses, unless ``pin=`` is given. (Triton's ``cache_key`` is not used: it misses most of those routes. For the same reason a build
  never reads Triton's disk cache. It compiles from scratch unless tilefish's own cache has the binary, keyed by the IR digest, the
  architecture, the toolchain, the Triton knobs that change code generation and the options, and verified on every read:
  :mod:`tilefish._cache`.)

Trust. A replay needs a trust anchor and refuses to run without one: ``run(..., mode="replay", manifest=directory)`` raises
unless it is given ``pin=`` or ``trust_directory=True``. (This is a deliberate break from 0.0.1 and 0.0.2, where the
directory was trusted by default.) The digests in a manifest are written next to the files they describe, so whoever can
write the directory can rewrite both and make ``replay`` run any binary. ``pin=`` is the digest of the whole manifest
(``Manifest.digest``, returned by :func:`export`): hold it somewhere you trust (a build log, a lock file) and a changed
directory is refused. ``trust_directory=True`` is the explicit statement that whatever the directory holds is fine, for a
directory you wrote yourself a moment ago; it does not combine with ``pin=``. A pin vouches for the manifest, not for the
machine code being correct, and not for an IR digest that is known to differ.

What a process holds is what is compiled. ``export`` in a process whose module was edited and not reloaded compiles, and stamps,
what the process holds; a fresh process that has the edited file computes another IR digest and refuses that manifest, with no
check of files, imports or values. What is not in the IR is not part of a binary's identity: the options, the architecture and the
toolchain (they are in the manifest entry and in the cache key), and the knobs of Triton that change code generation (the cache key;
``TRITON_KERNEL_OVERRIDE`` is refused by a build). The checks of earlier versions that tried to *find* what Triton reads (a digest of
the files a kernel reaches, the import watch with ``TILEFISH_IMPORT_WATCH``, the snapshot of module attributes, the walk over
closures) are gone.

Before ``jit`` and ``replay`` launch, every precondition of the binary is checked (dtypes, static sizes,
``divisible_by``, 16-byte aligned pointers, contiguity, aliasing, ``max_elements`` of a generated kernel); a
bucketed dimension must equal one bucket value exactly, there is no padding. ``reference`` checks the types, not
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
from types import SimpleNamespace
from typing import Any, Iterator

from . import _cache, _cuda, _ir
from ._build import BuiltKernel, build, resolve_triton, write_manifest_doc
from ._launch import (
    TORCH_DTYPE_NAMES,
    TensorInfo,
    bind_call,
    check_consts,
    check_dim_constraints,
    check_no_alias,
    check_scalar_ranges,
    check_spec_preconditions,
    eval_dim,
    eval_grid,
    select_spec,
    tensor_info,
    tf_dtype_name,
)
from .aot import CompileSpec, compile_specs
from .decorators import ExternOp, GraphOp, KernelOp, LibraryOp, Op
from .dims import Dim, DimExpr
from ._manifest import MANIFEST_NAME, Manifest, manifest_digest
from .errors import (
    TilefishBuildError,
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
    """Forget every built kernel and unload every loaded module (for tests and for rebuilding after a toolchain change).

    Nothing is freed under work that may still run: no run, capture or replay of a launch plan is in progress on any thread while this frees and unloads (it
    waits for the ones in progress, and the ones that start meanwhile wait for it, then build and load again), the arenas of launch plans are freed after the
    last run on each is done (its order event), and the modules after every context that has one is synchronized (a launch of a kernel may still be running on
    any stream). Call it outside a CUDA graph capture, and only when no graph captured from a plan with ``torch.cuda.graph`` will be replayed again (it holds
    the address of the arena); a :class:`~tilefish._planrun.CapturedPlan` refuses to replay after it."""
    from . import _planrun

    with _planrun.exclusive():
        with _lock:
            _builds.clear()
            _manifests.clear()
            _pins.clear()
        _planrun.clear_caches()  # the runners of launch plans, and the arenas they hold
        _cuda.unload_all()  # nothing to do (and no driver imported) when no module was loaded


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
        extern = self.launch_extern()
        if extern.binary is not None:  # a prebuilt binary has no Python function: its bytes are in the declaration
            return None
        if self.lowered is not None:
            return load_lowered(self.lowered)
        try:
            return resolve_triton(spec.triton)
        except TilefishBuildError as e:
            raise TilefishRuntimeError(str(e)) from e


def _lower_cached(kernel: KernelOp) -> LoweredKernel:
    """The lowering of a kernel, made once and kept on the kernel (a table keyed weakly by the kernel would never let it go: the lowering points back at it)."""
    low = kernel.__dict__.get("_tf_lowered")
    if low is None:
        low = lower(kernel)
        with _lock:
            low = kernel.__dict__.setdefault("_tf_lowered", low)
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
            f"cannot run the {op.kind} op '{op.name}': @tilefish.kernel, @tilefish.extern, a @tilefish.library with an impl and a @tilefish.graph run "
            "(a library operator that is only a signature has no runtime)"
        )
    raise TilefishRuntimeError(f"tilefish.run takes a kernel or an extern op (or a library op with an impl, or a graph), got {op!r}")


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
    scalars, grid = _finish_launch(extern, spec, call.dims, infos)
    return _Prepared(target, spec, str(device), tensors, scalars, grid, results)


def _finish_launch(extern: ExternOp, spec: CompileSpec, dims: "dict[Dim, int]", infos: "dict[str, TensorInfo]") -> "tuple[dict[str, int], tuple[int, ...]]":
    """The runtime scalars and the grid of one launch of ``spec``, after every check a launch makes (``dims`` are the bound dimensions,
    ``infos`` the facts about every pointer argument of the launched kernel by its parameter name).

    This is the half of :func:`_prepare` that does not depend on where the facts come from: a call on torch tensors has them from the
    tensors, a plan from the views it launches with (:mod:`tilefish._plan`).
    """
    by_name = {p.name: p for p in extern.signature.params}
    scalars: "dict[str, int]" = {}
    for a in spec.args:
        if a.kind != "scalar":
            continue
        p = by_name[a.name]
        assert p.dim is not None
        v = eval_dim(p.dim, dims)
        if a.count_of is not None:  # a generated kernel's n: the element count of that tensor argument
            count = math.prod(infos[a.count_of].shape)
            if v is not None and v != count:
                raise TilefishPreconditionError(
                    f"runtime argument '{a.name}' is {p.dim} = {v}, but the tensor '{a.count_of}' has {count} elements",
                    kind="shape",
                    arg=a.name,
                )
            if v is None and isinstance(p.dim, Dim):
                dims[p.dim] = count  # the grid may be written over it
            v = count
        elif v is None:
            raise TilefishRuntimeError(f"cannot determine the runtime argument '{a.name}' ({p.dim}) from the tensor shapes")
        scalars[a.name] = v
    check_dim_constraints(dims)
    check_scalar_ranges(spec, scalars)
    written = {p.name for p in extern.signature.params if p.kind == "tensor" and p.access != "in"}
    check_no_alias(infos, written)
    check_spec_preconditions(spec, infos, scalars, dims)
    assert extern.grid is not None
    return scalars, eval_grid(spec, extern.grid, dims)


def _launch_values(prep: _Prepared) -> "list[tuple[str, str, Any]]":
    """The kernel's arguments in declaration order: ``(kind, name, value)`` with a tensor, an int or a constexpr value."""
    out = []
    for a in prep.spec.args:
        if a.kind == "pointer":
            out.append(("pointer", a.name, prep.tensors[a.name]))
        elif a.kind == "scalar":
            out.append(("scalar", a.name, prep.scalars[a.name]))
        elif a.kind == "fixed":
            continue  # a prebuilt binary's constant argument: its value is in the manifest, and the launch writes it
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
    differ from the compile spec, so a manifest written from it is refused by ``replay``. An option that names a file is
    keyed by the content of the file (``extern_libs``: a rewritten library is another binary), and ``ir_override``, a file
    that replaces the IR of a stage, is refused (a ``TilefishBuildError``).
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


def _ir_digest_of(spec: CompileSpec, fn: Any, cap: int, extra: "dict[str, Any]") -> str:
    """The IR digest of the kernel ``fn`` as this process holds it, for ``spec`` on architecture ``cap`` (see :mod:`tilefish._ir`)."""
    inputs = spec.ast_source_inputs()
    options = {**inputs.options, **extra}
    return _ir.ir_digest(_ir.ttir_text(fn, inputs, arch=cap, options=options, what=f"'{spec.stem}' ({spec.triton})"))


def build_for_device(spec: CompileSpec, fn: Any, device_index: int) -> BuiltKernel:
    """The (cached, in memory) build of ``spec`` for the architecture of device ``device_index``.

    The cache is keyed by the digest of the IR Triton's frontend builds for ``fn`` now (a few milliseconds per call), so a kernel
    whose process state changed is not served the binary of the old one, and the same Triton knobs that key the disk cache.
    """
    torch = _torch()
    cap = _capability(torch, device_index)
    extra = dict(getattr(_options, "value", {}))
    files = _cache.compile_files({**spec.ast_source_inputs().options, **extra})  # refuses ir_override; the content of extern_libs
    digest = _ir_digest_of(spec, fn, cap, extra)
    environment = repr(sorted(_cache.codegen_environment().items()))
    key = (spec.stem, cap, spec._fingerprint(), digest, repr(sorted(extra.items())), tuple(sorted(files.items())), environment)
    with _lock:
        built = _builds.get(key)
    if built is None:
        built = build(spec, arch=cap, fn=fn, options=extra, ir_digest=digest)
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


def build_binary_for_device(spec: CompileSpec, target: Any, device_index: int) -> BuiltKernel:
    """The (cached, in memory) build of a prebuilt binary for the architecture of device ``device_index``: PTX assembled, or the cubin declared for it."""
    from ._binary import build_binary

    torch = _torch()
    cap = _capability(torch, device_index)
    key = (spec.stem, cap, spec._fingerprint(), tuple(sorted(target.digests.items())), os.environ.get("TILEFISH_PTXAS_PATH", ""))
    with _lock:
        built = _builds.get(key)
    if built is None:
        built = build_binary(spec, target, arch=cap)
        with _lock:
            _builds[key] = built
    return built


def jit_built(target: _Target, spec: CompileSpec, index: int) -> BuiltKernel:
    """The binary of ``spec`` as ``jit`` builds it (in memory, for the architecture of device ``index``): the current source through the compile
    spec, or a prebuilt PTX assembled / a cubin taken for the device."""
    extern = target.launch_extern()
    if extern.binary is not None:
        return build_binary_for_device(spec, extern.binary, index)
    return build_for_device(spec, target.jit_function(spec), index)


def _run_jit(prep: _Prepared) -> None:
    """Build the current source through the compile spec, in memory, and launch that binary."""
    torch = _torch()
    index = _device_index(torch, prep.device)
    _launch_built(prep, jit_built(prep.target, prep.spec, index), index)


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


def _current_ir(prep: "_Prepared", cap: int, built_with: "str | None") -> "tuple[str | None, str | None]":
    """``(IR digest, Triton version)`` of the kernel this process holds for the spec, or no digest where it cannot be made or compared.

    ``built_with`` is the Triton version the manifest entry was built with. The versions are compared *before* the frontend runs: a
    Triton that is not that version prints the IR differently, so its digest could not be compared, and its frontend may not even
    have the entry points tilefish calls (a ``TilefishBuildError`` instead of the pin-only fallback). The result is then ``(None,
    version)`` and nothing of Triton's is run. ``(None, None)`` is when Triton cannot be imported (a module that imports it does
    not import either), or the module that defines a hand-written kernel cannot be imported. A frontend of the right version that
    cannot build the kernel is an error, not a missing digest.
    """
    if not _ir.triton_importable():
        return None, None
    version = _ir.triton_version()
    if version != built_with:
        return None, version
    try:
        fn = prep.target.jit_function(prep.spec)
    except TilefishRuntimeError:
        return None, None
    return _ir_digest_of(prep.spec, fn, cap, {}), version


def replay_built(target: _Target, spec: CompileSpec, manifest: Manifest, pinned: bool, index: int) -> BuiltKernel:
    """The binary of ``spec`` in a manifest for the architecture of device ``index``, after every check a replay makes: the entry is what the
    spec asks for, and the IR digest of the kernel this process holds is the entry's when it can be recomputed (``pinned`` says the caller
    vouches for the manifest)."""
    torch = _torch()
    cap = _capability(torch, index)
    arch = f"sm_{cap}"
    if spec.binary is not None:  # a prebuilt binary is identified by the sha256 of its PTX or cubin, compared without Triton
        return manifest.built_kernel(spec, arch, pinned=pinned)
    artifact = manifest.get(spec.stem, arch)
    digest = version = None
    if artifact is not None and artifact.entry.get("source_digest_kind") == _ir.IR_DIGEST_KIND:  # an entry with no IR digest has nothing to compare
        toolchain = artifact.entry.get("toolchain")
        digest, version = _current_ir(SimpleNamespace(target=target, spec=spec), cap, toolchain.get("triton") if isinstance(toolchain, dict) else None)  # type: ignore[arg-type]
    return manifest.built_kernel(spec, arch, ir_digest=digest, ir_triton=version, pinned=pinned)


def _run_replay(prep: _Prepared, manifest: Manifest, pinned: bool) -> None:
    """Launch the prebuilt cubin of the manifest for this spec and this device's architecture."""
    torch = _torch()
    index = _device_index(torch, prep.device)
    _launch_built(prep, replay_built(prep.target, prep.spec, manifest, pinned, index), index)


_library_specs: "weakref.WeakKeyDictionary[LibraryOp, list[Any]]" = weakref.WeakKeyDictionary()  # an op is immutable
_pins: "dict[tuple, Any]" = {}  # (stem, arch, device, its SM count, workspace, cuBLASLt version, declaration) -> the BuiltLibrary that jit pinned in memory


def _run_library(op: LibraryOp, args: "tuple[Any, ...]", kwargs: "dict[str, Any]", mode: str, manifest: "Manifest | None") -> Any:
    """Run a pinned library operation (a cuBLASLt GEMM). ``jit`` pins the algorithm in memory exactly as ``export`` does and runs it; ``replay``
    runs the algorithm a manifest recorded, after checking that its entry is what the declaration asks for and that cuBLASLt is the version that
    chose it; ``reference`` is torch: an f32 matmul of the operands with the bias added in f32, rounded once to bf16."""
    from . import _library

    torch = _torch()
    call = bind_call(op.name, op.signature, args, kwargs)
    check_dim_constraints(call.dims, buckets=mode != "reference")
    if mode == "reference":
        device = _device_of(call.infos, call.tensors)
        return _library.reference_gemm(call.tensors).to(device)
    with _lock:
        specs = _library_specs.get(op)
    if specs is None:
        specs = _library.library_specs(op)
        with _lock:
            _library_specs[op] = specs
    spec = select_spec(op.name, specs, call.dims)
    device = _device_of(call.infos, call.tensors)
    if getattr(device, "type", None) != "cuda":
        raise TilefishRuntimeError(f"jit and replay run on a CUDA device; the tensors are on {device}")
    shape = [eval_dim(d, call.dims) for d in op.signature.returns[0].shape]
    out = torch.empty(shape, dtype=torch.bfloat16, device=device)
    check_spec_preconditions(spec, {**call.infos, "out": tensor_info("out", out)}, {}, call.dims)
    index = _device_index(torch, str(device))
    if mode == "jit":
        built = jit_pin(spec, index)
    else:
        assert manifest is not None
        built = replay_pin(spec, manifest, index)
    with torch.cuda.device(index):
        stream = torch.cuda.current_stream(index).cuda_stream
        _library.run_gemm(spec, built.algo, call.tensors, out, stream, index)
    return out


def jit_pin(spec: Any, index: int) -> Any:
    """The algorithm ``jit`` pins in memory for a library spec on device ``index`` (what ``export`` writes), cached for the process."""
    from . import _cublaslt, _library

    cap = _capability(_torch(), index)
    # a pin is for one device: its index (two devices of one architecture may have other SM counts, a MIG slice fewer than the whole GPU) and its SM count
    key = (spec.stem, f"sm_{cap}", index, _library.sm_count(index), spec.workspace_bytes, _cublaslt.version(), spec.declaration_digest)
    with _lock:
        built = _pins.get(key)
    if built is None:
        built = _library.build_library(spec, arch=cap, device_index=index)
        with _lock:
            _pins[key] = built
    return built


def replay_pin(spec: Any, manifest: Manifest, index: int) -> Any:
    """The algorithm a manifest recorded for a library spec on device ``index``, after checking that it was pinned on a device of this one's SM count,
    that its entry is what the declaration asks for and that cuBLASLt is the version that chose it."""
    from . import _library

    return _library.built_from_manifest(manifest, spec, f"sm_{_capability(_torch(), index)}", _library.sm_count(index), index)


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


def export(
    ops: Any,
    directory: "str | os.PathLike[str]",
    *,
    arch: "int | str | None" = None,
    overwrite: bool = False,
    buckets: Any = None,
    reuse: bool = False,
) -> Any:
    """Build every binary of one or several kernels and externs and write them as a manifest directory.

    For each compile spec of each operator (one per bucket) the cubin is built for ``arch`` (default: the
    architecture of the current CUDA device) and written to ``directory`` together with ``manifest.json``, which is
    what ``tilefish.run(..., mode="replay", manifest=directory)`` loads, and what a C++ or Rust host reads.

    An existing manifest in ``directory`` is merged, not replaced: its entries for other architectures and other
    kernels stay, and a new binary replaces the entry of the same stem and architecture. An existing entry that is a
    different binary (other machine code, declaration, source or options) is replaced only with ``overwrite=True``.

    No build reads Triton's disk cache; a binary comes from tilefish's own build cache when it has one for exactly this
    source, architecture, toolchain and options (verified on every read), and is compiled otherwise. The files of a
    hand-written kernel must be the ones the process imported (see :func:`tilefish.aot.build`). Files are written
    through temporary files and renamed, and a symbolic link where a file goes is refused.

    A library op with an ``impl`` (a cuBLASLt GEMM) is pinned for the device (``arch`` must be its architecture) and gets an entry in the manifest's
    ``library`` section; a prebuilt PTX is assembled by ``ptxas`` and a cubin is taken as it is (:mod:`tilefish._binary`).

    Returns the loaded and verified :class:`~tilefish.aot.Manifest` (``.pinned``: it is what ``hosts.rust.emit`` takes): ``.directory``, ``.path`` and ``.digest``, the
    digest of the whole manifest, which is what to record (in a build log, a lock file) and pass back as
    ``run(..., mode="replay", manifest=directory, pin=digest)`` to refuse a directory that was changed after the export.
    The digest is the one of the manifest that was written from the builds in memory; the directory is read back and
    must have the same digest, or this raises :class:`~tilefish.errors.TilefishManifestError`.
    """
    items = list(ops) if isinstance(ops, (list, tuple)) else [ops]
    graphs = [o for o in items if isinstance(o, GraphOp) and not isinstance(o, KernelOp)]
    if graphs:
        if len(items) != 1:
            raise TilefishRuntimeError("export takes one graph by itself (its manifest holds the binaries of its plan); export the other operators separately")
        from ._planrun import export_graph

        return export_graph(graphs[0], directory, arch=arch, overwrite=overwrite, buckets=buckets, reuse=reuse)
    if buckets is not None or reuse:
        raise TilefishRuntimeError("buckets= and reuse= are for exporting a graph (they shape its launch plan); kernels and externs are built for the buckets of their own dimensions")
    if arch is None:
        torch = _torch()
        cap = _capability(torch, torch.cuda.current_device())
    else:
        from ._build import parse_arch

        cap = parse_arch(arch)
    built: "list[BuiltKernel]" = []
    pinned_library: "list[Any]" = []
    for op in items:
        if isinstance(op, LibraryOp) and op.impl is not None:  # a pinned library operation (a cuBLASLt GEMM): an entry, no file
            from ._library import build_library, library_specs

            for lspec in library_specs(op):
                pinned_library.append(build_library(lspec, arch=cap))
            continue
        target = _target(op)
        extern = target.launch_extern()
        for spec in target.compile_specs():
            if extern.binary is not None:
                from ._binary import build_binary

                built.append(build_binary(spec, extern.binary, arch=cap))
            else:
                built.append(build(spec, arch=cap, fn=target.jit_function(spec)))
    path, written = write_manifest_doc(
        built, pathlib.Path(directory) / MANIFEST_NAME, cubins=True, merge=True, overwrite=overwrite, library=pinned_library
    )
    loaded = Manifest.load(path)
    expected = manifest_digest(written)
    if loaded.digest != expected:
        raise TilefishManifestError(
            f"the manifest in {str(path.parent)!r} has digest {loaded.digest} when it is read back, but the manifest that was "
            f"written has digest {expected}: the directory changed while it was being exported"
        )
    loaded.pinned = True  # its digest is the one of the manifest this process wrote from its own builds
    return loaded


def _run_graph(op: GraphOp, args: "tuple[Any, ...]", kwargs: "dict[str, Any]", mode: str, manifest: Any, pin: "str | None", trust_directory: bool) -> Any:
    """``run`` of a ``@tilefish.graph``: its launch plan (:mod:`tilefish._plan`), run by :mod:`tilefish._planrun`. ``plan=`` and ``outputs=`` are the two
    keywords that are not arguments of the graph (a graph with a parameter of those names takes it positionally)."""
    plan = kwargs.pop("plan", None)
    outputs = kwargs.pop("outputs", None)
    if manifest is not None:
        raise TilefishRuntimeError("manifest= is for a kernel or an extern; a graph replays plan=<directory> (written by tilefish.runtime.export(graph, directory))")
    if mode != "replay" and (pin is not None or trust_directory):
        raise TilefishRuntimeError(f"pin= and trust_directory= are for mode='replay' (they vouch for a plan directory); mode {mode!r} runs no plan directory")
    if trust_directory is not True and trust_directory is not False:
        raise TilefishRuntimeError(f"trust_directory= is True or False, got {trust_directory!r}")
    if pin is not None and not (isinstance(pin, str) and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin)):
        raise TilefishRuntimeError(f"pin= is the 64 lowercase hex digits of a plan digest, got {pin!r}")
    from ._planrun import run_graph

    return run_graph(op, args, kwargs, mode=mode, plan=plan, pin=pin, trust_directory=trust_directory, outputs=outputs)


def capture(
    op: GraphOp,
    /,
    *externals: Any,
    mode: str = "jit",
    plan: Any = None,
    pin: "str | None" = None,
    trust_directory: bool = False,
    devscalars: Any = None,
    outputs: Any = None,
    stream: Any = None,
) -> Any:
    """Capture the launch plan of a ``@tilefish.graph`` in a CUDA graph of its own: a :class:`~tilefish._planrun.CapturedPlan`, whose ``replay(stream=None)``
    is ordered with every other run and replay of the plan, from any thread and on any stream (it takes the arena's lock and waits for and records the arena's
    order event around the launch of the graph), and whose ``set_devscalar(name, value)`` changes a device scalar for the next replays.

    ``externals`` are the graph's arguments in the order of its signature without its ``DevScalar`` parameters, whose values ``devscalars={name: value}``
    gives; ``outputs=`` the tensors the returned values are written into (allocated before the capture by default: ``CapturedPlan.outputs``). ``mode``,
    ``plan``, ``pin`` and ``trust_directory`` choose the plan as for :func:`run`. The ``CapturedPlan`` keeps the externals and the arena alive. See
    :func:`tilefish._planrun.capture` and ``docs/plan-schema-1.md``, section 10."""
    if not isinstance(op, GraphOp) or isinstance(op, KernelOp):
        raise TilefishRuntimeError(f"tilefish.capture takes a @tilefish.graph, got {op!r}")
    if pin is not None and not (isinstance(pin, str) and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin)):
        raise TilefishRuntimeError(f"pin= is the 64 lowercase hex digits of a plan digest, got {pin!r}")
    from ._planrun import capture as _capture

    return _capture(
        op, externals, mode=mode, plan=plan, pin=pin, trust_directory=trust_directory, devscalars=devscalars, outputs=outputs, stream=stream
    )


def prepare(op: GraphOp, **options: Any) -> None:
    """Do ahead of time what a run of a graph's plan does for itself the first time (allocate its arena, build or load every binary, create the cuBLASLt
    handle), none of which can happen while a CUDA graph is being captured. ``prepare(graph, mode="jit" | "replay", plan=..., pin=..., device=..., buckets=...)``;
    see :func:`tilefish._planrun.prepare`."""
    from ._planrun import prepare as _prepare_plan

    _prepare_plan(op, **options)


def run(
    op: Any,
    /,
    *args: Any,
    mode: str = "jit",
    manifest: Any = None,
    pin: "str | None" = None,
    trust_directory: bool = False,
    **kwargs: Any,
) -> Any:
    """Run a kernel or an extern op on tensors, or a graph through its launch plan. See the module docstring for the three modes.

    ``op`` can be a ``@tilefish.graph``: ``jit`` runs its plan (``plan=tilefish.plan(graph, ...)`` or, by default, ``tilefish.plan(graph)``) with binaries built in
    memory, ``replay`` runs the plan of an exported directory (``plan=<directory>``, with ``pin=<the plan's digest>`` or ``trust_directory=True``: the plan
    pins its manifest, so it pins every binary), ``reference`` composes the references of its ops in torch. ``outputs=`` gives the tensors a plan writes its
    returned values into instead of allocating them. See :mod:`tilefish._plan` and ``docs/plan-schema-1.md``.

    ``op`` is a ``@tilefish.kernel`` (or its :class:`~tilefish.lower.LoweredKernel`) or an ``@tilefish.extern`` whose
    Triton function can be imported. ``mode`` is ``"jit"``, ``"replay"`` or ``"reference"``. ``manifest`` is the
    directory (or the :class:`~tilefish.aot.Manifest`) that ``"replay"`` loads its cubins from, and is required
    there and nowhere else.

    A replay needs a trust anchor, one of ``pin`` and ``trust_directory``, and refuses to run without. ``pin`` is the
    digest of the manifest (``Manifest.digest``, what ``export`` returns, kept somewhere you trust): a manifest with any
    other digest is refused, so the directory is trusted only as far as the digest you hold. ``trust_directory=True`` says
    that you trust whatever the directory holds, which is what whoever can write it makes it (the digests sit next to the
    files they describe); it is for a directory you wrote yourself a moment ago, and is not combined with ``pin``. For a
    hand-written kernel whose IR digest cannot be recomputed, only ``pin`` is the statement that you vouch for the
    manifest. ``mode``, ``manifest``, ``pin`` and ``trust_directory`` are the only keywords that are not arguments of
    ``op`` (an operator with a parameter of one of those names takes it positionally). Raises
    :class:`~tilefish.errors.TilefishPreconditionError` when an argument breaks a precondition of the binary,
    :class:`~tilefish.errors.TilefishManifestError` when the manifest is not what was built, not what the operator asks
    for, or not the pinned one, and :class:`~tilefish.errors.TilefishRuntimeError` when the call cannot run (a replay
    without a trust anchor, for one).
    """
    if mode not in MODES:
        raise TilefishRuntimeError(f"unknown mode {mode!r}: use one of {', '.join(MODES)}")
    if isinstance(op, GraphOp) and not isinstance(op, KernelOp):
        return _run_graph(op, args, kwargs, mode, manifest, pin, trust_directory)
    if mode == "replay" and manifest is None:
        raise TilefishRuntimeError(
            "replay runs prebuilt cubins from a manifest directory: pass manifest=<directory> (write one with "
            "tilefish.runtime.export(op, directory) or tilefish.aot.write_manifest(built, path, cubins=True))"
        )
    if mode != "replay" and manifest is not None:
        raise TilefishRuntimeError(f"manifest= is for mode='replay'; mode {mode!r} builds or interprets the kernel itself")
    if mode != "replay" and pin is not None:
        raise TilefishRuntimeError(f"pin= is for mode='replay' (it pins a manifest); mode {mode!r} has none")
    if trust_directory is not True and trust_directory is not False:
        raise TilefishRuntimeError(f"trust_directory= is True or False, got {trust_directory!r}")
    if mode != "replay" and trust_directory:
        raise TilefishRuntimeError(f"trust_directory= is for mode='replay'; mode {mode!r} runs no manifest to trust")
    if pin is not None and not (isinstance(pin, str) and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin)):
        raise TilefishRuntimeError(f"pin= is the 64 lowercase hex digits of a manifest digest, got {pin!r}")
    if mode == "replay":
        if pin is not None and trust_directory:
            raise TilefishRuntimeError(
                "pass either pin= (a manifest with another digest is refused) or trust_directory=True (whatever the directory "
                "holds is accepted), not both: one of them says which manifest to trust and the other that any will do"
            )
        if pin is None and not trust_directory:
            raise TilefishRuntimeError(
                "replay needs a trust anchor: the digests in a manifest directory sit next to the files they describe, so whoever "
                "can write the directory can make a replay run any binary. Pass pin=<the manifest's digest> (Manifest.digest, which "
                "export returns; keep it somewhere you trust), or trust_directory=True to run whatever the directory holds"
            )
    is_library = isinstance(op, LibraryOp) and op.impl is not None
    target = None if is_library else _target(op)
    if mode == "reference" and not is_library:
        return _run_reference(target, args, kwargs)
    loaded = None
    if mode == "replay":
        loaded = _load_manifest(manifest)
        if pin is not None and loaded.digest != pin:
            raise TilefishManifestError(
                f"the manifest in {str(loaded.directory)!r} has digest {loaded.digest}, but pin= says {pin}: "
                "it is not the manifest that was pinned (it was changed, or it is another export)"
            )
    if is_library:
        return _run_library(op, args, kwargs, mode, loaded)
    assert target is not None
    prep = _prepare(target, args, kwargs)
    if mode == "jit":
        _run_jit(prep)
    else:
        assert loaded is not None
        _run_replay(prep, loaded, pinned=pin is not None)
    return _result(prep.results)
