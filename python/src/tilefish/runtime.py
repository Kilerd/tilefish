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
  or names, is hashed (Gluon's frontend and its TTGIR for a Gluon kernel). Every manifest entry records that digest (``source_digest``,
  kind ``"tilefish.triton-ir.v1"``, or ``"tilefish.gluon-ir.v1"`` for a Gluon kernel), and when Triton
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
machine code being correct, and not for an IR digest that is known to differ. The digest covers the files byte for byte, and an
export is reproducible: a cubin names its sources by module path (:func:`tilefish._ir.host_independent_paths`) and a generated
source has a content-addressed file (:func:`load_lowered`), so the same operators exported with the same toolchain for the same
device, from another directory, process or machine, have the same digest, and a pin recorded for one export checks another.

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
import importlib.machinery
import importlib.util
import linecache
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

from . import _cache, _cuda, _cudamath, _ir, _source
from ._build import BuiltKernel, build, check_launch_config, resolve_triton, write_file, write_manifest_doc
from ._cuda import sm_partition as sm_partition  # tilefish.runtime.sm_partition(sms): the partition (a green context) a persistent grid is resolved for
from ._launch import (
    TORCH_DTYPE_NAMES,
    TensorInfo,
    bind_call,
    bind_infos,
    bind_outputs,
    canonical_strides,
    check_tensor,
    check_consts,
    check_dim_constraints,
    check_no_alias,
    check_scalar_ranges,
    check_spec_preconditions,
    eval_dim,
    eval_grid,
    given_values,
    select_spec,
    solve_dims,
    tensor_info,
    tf_dtype_name,
)
from ._specvalid import ALIGNMENT
from .aot import CompileSpec, compile_specs
from .decorators import ExternOp, GraphOp, KernelOp, LibraryOp, Op
from .dtypes import BlockScaled
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
_gen_root: "list[pathlib.Path]" = []  # this process's private scratch directory for generated sources, once it was needed
_warned_gen_roots: "set[str]" = set()


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

    def at(self, cap: "int | str") -> "_Target":
        """This target on architecture ``cap``: for an extern declared with ``by_arch=``, the implementation it runs there (refused when it has none for
        ``cap``); itself for any other op."""
        extern = self.launch_extern()
        if extern.implementations is None:
            return self
        return _Target(self.name, self.public, self.kernel, extern.for_arch(cap), self.lowered)

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
    if isinstance(op, ExternOp):  # a generated op (tilefish.contiguous, im2col, im2col1d, col2im1d) carries the Triton source tilefish wrote for it
        return _Target(op.name, op.signature, None, op, op.__dict__.get("_tf_generated"))
    if isinstance(op, Op):
        raise TilefishRuntimeError(
            f"cannot run the {op.kind} op '{op.name}': @tilefish.kernel, @tilefish.extern, a @tilefish.library with an impl and a @tilefish.graph run "
            "(a library operator that is only a signature has no runtime)"
        )
    raise TilefishRuntimeError(f"tilefish.run takes a kernel or an extern op (or a library op with an impl, or a graph), got {op!r}")


def _private_gen_root() -> pathlib.Path:
    """This process's own scratch directory for generated sources (made once, private to the user, removed at exit): where a source goes when
    the cache directory cannot hold it (:func:`_generated_file`)."""
    if not _gen_root:
        root = pathlib.Path(tempfile.mkdtemp(prefix="tilefish-gen-"))
        atexit.register(shutil.rmtree, root, True)
        _gen_root.append(root)
    return _gen_root[0]


def _holds(path: pathlib.Path, data: bytes) -> bool:
    """Whether ``path`` is a regular file (not a symbolic link) whose bytes are exactly ``data``."""
    try:
        return not path.is_symlink() and path.is_file() and path.read_bytes() == data
    except OSError:
        return False


def _generated_file(lowered: LoweredKernel, key: str, data: bytes) -> pathlib.Path:
    """The file that holds the generated source ``data`` (sha256 ``key``) of ``lowered``: ``<root>/<key>/tilefish_generated/<name>.py``.

    The path is addressed by the content, so every process (and every user who shares the directory) that generates the same source uses the same
    file, and a file that is there is used only when it holds exactly ``data``. The root is ``generated/`` in tilefish's cache directory
    (``$TILEFISH_CACHE_DIR``, else ``~/.cache/tilefish``); a file is written through a temporary file and an atomic rename, so processes that write
    it at the same time do not tear it, and it is read back before it is used. When that directory is switched off (an empty
    ``$TILEFISH_CACHE_DIR``), cannot be written, or does not hold the source after the write, the source goes to this process's private scratch
    directory (one :class:`~tilefish.errors.TilefishCacheWarning` per directory says why). Where the file is does not reach a binary: its debug
    information names it ``tilefish_generated/<name>.py`` (:func:`tilefish._source.source_name`).
    """
    shared = _cache.cache_directory()
    problems = []
    for root in ([shared / "generated"] if shared is not None else []) + [None]:
        path = (root if root is not None else _private_gen_root()) / key / lowered.filename
        if _holds(path, data):
            return path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_file(path, data)
        except (OSError, TilefishBuildError) as e:  # a directory that cannot be written; a symbolic link where the file goes
            problems.append(f"cannot write {str(path)!r} ({type(e).__name__}: {e})")
            continue
        if _holds(path, data):
            return path
        problems.append(f"{str(path)!r} does not hold the generated source after it was written")
    raise TilefishRuntimeError(f"cannot write the generated source of '{lowered.name}' anywhere: {'; '.join(problems)}")


def _fell_back(lowered: LoweredKernel, path: pathlib.Path) -> None:
    """One warning per shared directory that could not take a generated source (the source went to the private scratch directory instead)."""
    shared = _cache.cache_directory()
    if shared is None or not _gen_root or not path.is_relative_to(_gen_root[0]):
        return
    where = str(shared / "generated")
    if where in _warned_gen_roots:
        return
    _warned_gen_roots.add(where)
    _cache._warn(
        f"cannot keep the generated source of '{lowered.name}' in {where!r}; generated sources go to {str(_gen_root[0])!r} for this process "
        f"instead (set {_cache.ENV_DIR} to a writable directory, or to an empty value to switch the cache off). Binaries do not depend on it"
    )


def load_lowered(lowered: LoweredKernel) -> Any:
    """Write the generated source of a lowered kernel (or of a generated op) to its content-addressed file, import it, and return its ``JITFunction``.

    The file is ``<tilefish cache directory>/generated/<sha256 of the source>/tilefish_generated/<name>.py`` (:func:`_generated_file`, with a
    fallback to a private scratch directory), so its path is the same in every process that generates the source, and nothing is left in a
    temporary directory. A file is there because Triton reads the source of a JIT function from a file, and because a build checks that the
    function's file is the source its spec claims. The module is executed from the source in memory (the bytes the file was checked to hold, never
    a byte code cache), and Python's line cache is given the same text, which is what Triton reads. The binaries do not depend on where the file is:
    the frontend of a build names it ``tilefish_generated/<name>.py`` (:func:`tilefish._ir.host_independent_paths`), as it names the module of a
    :meth:`~tilefish.lower.LoweredKernel.write` that was imported from a directory on ``sys.path``. The result is cached by the sha256 of the source;
    a cached function whose file has gone is given its file back.
    """
    data = lowered.source.encode("utf-8")
    key = hashlib.sha256(data).hexdigest()
    with _lock:
        fn = _modules.get(key)
        if fn is not None:
            path = pathlib.Path(fn.__globals__["__file__"])
            if not _holds(path, data):  # removed (a cache directory deleted under a running process): put it back where the function says it is
                with contextlib.suppress(OSError, TilefishBuildError):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    write_file(path, data)
            return fn
        path = _generated_file(lowered, key, data)
        _fell_back(lowered, path)
        _source.register_source_name(str(path), lowered.filename)
        name = f"{lowered.module.replace('.', '_')}__{key[:12]}"
        lines = lowered.source.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        linecache.cache[str(path)] = (len(lowered.source), None, lines, str(path))  # what Triton reads (inspect.getsource): the checked text
        spec = importlib.machinery.ModuleSpec(name, None, origin=str(path))
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(lowered.source, str(path), "exec"), module.__dict__)  # noqa: S102 - the source tilefish generated, as checked on disk
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
            raise TilefishRuntimeError(
                f"cannot allocate {where}: dimension {d} is not determined by the arguments (a dimension that only an Out parameter mentions "
                "is bound from the Out tensor: give it with outputs=)"
            )
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


def _result_names(target: _Target, extern: ExternOp) -> "list[str]":
    """The launched kernel's parameters that hold the results of a call, in the order ``run`` returns them: a kernel's returned values (the stub
    lists the kernel's own parameters, then one output per returned value), then its ``Out`` tensors; an extern's ``Out`` tensors."""
    out_params = [p.name for p in extern.signature.params if p.kind == "tensor" and p.access == "out"]
    if target.kernel is None:
        return out_params
    names = _extern_names(target, extern)
    user = len(target.kernel.signature.params)
    returned = [p.name for p in extern.signature.params[user:] if p.kind == "tensor" and p.access == "out"]
    return returned + [names[p.name] for p in target.kernel.signature.tensor_outputs]


def _given_outputs(target: _Target, outputs: Any, result_names: "list[str]") -> "dict[str, Any]":
    """``outputs=`` of a call: the tensors to write the results into, by the launched kernel's parameter name (none given: ``{}``)."""
    if outputs is None:
        return {}
    given = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    if len(given) != len(result_names):
        raise TilefishRuntimeError(
            f"outputs= gives {len(given)} tensor(s), but '{target.name}' has {len(result_names)} result(s) ({', '.join(result_names) or 'none'}): give "
            "one tensor per result, in the order the call returns them"
        )
    return dict(zip(result_names, given))


def _prepare(target: _Target, args: "tuple[Any, ...]", kwargs: "dict[str, Any]", outputs: Any = None, specialise: bool = True) -> _Prepared:
    """Check a call and decide its binary, scalars and grid. ``outputs`` are the tensors the results are written into (a tensor, or one per result in
    the order the call returns them) instead of allocated ones: their shapes bind the dimensions only the ``Out`` parameters mention, and each is held to
    the dtype, the rank, the contiguity and the shape its parameter has. A ``@tilefish.kernel`` given a read-only tensor that is not 16-byte aligned runs
    its specialisation for it (:func:`_specialised`), unless ``specialise=False``."""
    torch = _torch()
    call = bind_call(target.name, target.public, args, kwargs)
    if specialise:
        target = _specialised(target, call.infos)
    extern = target.launch_extern()
    names = _extern_names(target, extern)
    result_names = _result_names(target, extern)
    given = _given_outputs(target, outputs, result_names)
    given_infos = {name: tensor_info(name, t) for name, t in given.items()}
    public_of = {v: k for k, v in names.items()}
    # the dimensions the inputs leave unbound, from the Out tensors given (a kernel's returned values are not parameters of its signature)
    dims = bind_outputs(target.name, target.public, {public_of[n]: i for n, i in given_infos.items() if n in public_of}, call.dims)
    check_dim_constraints(dims)
    if extern.implementations is not None:  # an implementation per architecture: the one of the device's
        _declared_consts(target.public, call.consts)  # a const of the op's stub restated: its value there (and the implementation's, below, if it has one)
        device = _device_of({**call.infos, **given_infos}, {**call.tensors, **given})
        if getattr(device, "type", None) != "cuda":
            raise TilefishRuntimeError(f"jit and replay run on a CUDA device; the tensors are on {device}")
        target = target.at(_capability(torch, _device_index(torch, str(device))))
        extern = target.launch_extern()
        # its own ABI: the op's tensors (the same names, in the same order: the declaration checks it), and its own scalars, constexprs or fixed
        # arguments, and the dimensions its stub names (the stride dimensions of a view passed as a TMA descriptor are its own)
        names = _extern_names(target, extern)
        result_names = _result_names(target, extern)
        own = extern.signature
        dims = bind_infos(target.name, own, {p.name: call.infos[p.name] for p in own.params if p.kind == "tensor" and p.access != "out"})
        dims = bind_outputs(target.name, own, {n: i for n, i in given_infos.items() if n in names}, dims)
        check_dim_constraints(dims)
    specs = target.compile_specs()
    spec = select_spec(target.name, specs, dims)
    check_consts(spec, {names.get(k, k): v for k, v in call.consts.items()})  # (a const of the op's stub that the implementation does not have: not its)
    device = _device_of({**call.infos, **given_infos}, {**call.tensors, **given})
    if getattr(device, "type", None) != "cuda":
        raise TilefishRuntimeError(f"jit and replay run on a CUDA device; the tensors are on {device}")

    # the tensors of the launched kernel, by its parameter names: the supplied ones, and every Out (given, or allocated here)
    tensors = {names[k]: v for k, v in call.tensors.items()}
    for p in extern.signature.params:
        if p.kind != "tensor" or p.access != "out":
            continue
        if p.name in given:
            _check_given(target.name, p, given_infos[p.name], dims)
            tensors[p.name] = given[p.name]
        else:
            tensors[p.name] = _alloc(torch, p, dims, device, f"Out parameter '{p.name}'")
    results = [tensors[n] for n in result_names]

    infos = {name: tensor_info(name, t) for name, t in tensors.items()}
    scalars, grid = _finish_launch(extern, spec, dims, infos, {names.get(k, k): v for k, v in call.consts.items()})
    return _Prepared(target, spec, str(device), tensors, scalars, grid, results)


def _specialised(target: _Target, infos: "dict[str, TensorInfo]") -> _Target:
    """The target that runs a call of a ``@tilefish.kernel`` whose read-only tensors include a dense one at an address that is not a multiple of 16
    bytes (and is one of its element size): the kernel's specialisation for those parameters (:meth:`tilefish.decorators.KernelOp.unaligned`), whose
    pointers are compiled aligned to their element only, as Triton's own launcher would specialise on the address. Any other call, and a kernel lowered
    by the caller with options of its own (a :class:`~tilefish.lower.LoweredKernel`), runs as it is (an address that is not 16-byte aligned is then
    refused by the binary's precondition)."""
    kernel = target.kernel
    if kernel is None or target.lowered is not None or target.extern is not None:
        return target
    names = []
    for p in kernel.signature.tensor_inputs:
        info = infos.get(p.name)
        if p.access != "in" or p.is_devscalar or isinstance(p.type.dtype, BlockScaled) or info is None or not info.contiguous:  # type: ignore[union-attr]
            continue
        if info.ptr % ALIGNMENT and not info.ptr % max(1, info.itemsize):
            names.append(p.name)
    if not names:
        return target
    return _target(kernel.unaligned(*names))


def _grid_of(extern: ExternOp, spec: CompileSpec) -> "tuple[DimExpr, ...]":
    """The grid expressions of the binary ``spec`` of ``extern``: its declared grid, or the one ``by_bucket=`` gives that bucket."""
    grid = extern.grid_for(spec.buckets)
    assert grid is not None
    return grid


def _check_given(op_name: str, p: Param, info: TensorInfo, dims: "dict[Dim, int]") -> None:
    """An ``Out`` tensor a call is given: its dtype, rank and contiguity, and the shape its parameter has under the bound dimensions (what a launch
    would allocate)."""
    check_tensor(p, info)
    assert p.type is not None
    solve_dims(op_name, [(p.name, axis, pattern, actual, str(p.type), "dimension") for axis, (pattern, actual) in enumerate(zip(p.type.shape, info.shape))], dims)


def _finish_launch(
    extern: ExternOp, spec: CompileSpec, dims: "dict[Dim, int]", infos: "dict[str, TensorInfo]", given: "dict[str, Any] | None" = None
) -> "tuple[dict[str, int], tuple[int, ...]]":
    """The runtime scalars and the grid of one launch of ``spec``, after every check a launch makes (``dims`` are the bound dimensions,
    ``infos`` the facts about every pointer argument of the launched kernel by its parameter name). ``given`` are the values of the fixed
    arguments per call (a prebuilt binary's ``given`` arguments, by name), which join the scalars once checked.

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
    scalars.update(given_values(spec, given or {}))  # a fixed argument per call: an int of its type, any sign
    written = {p.name for p in extern.signature.params if p.kind == "tensor" and p.access != "in"}
    check_no_alias(infos, written, {p.name: p.alias for p in extern.signature.params if p.alias is not None})
    check_spec_preconditions(spec, infos, scalars, dims)
    check_tensor_maps(spec, infos)
    return scalars, eval_grid(spec, _grid_of(extern, spec), dims)


def check_tensor_maps(spec: CompileSpec, infos: "dict[str, TensorInfo]") -> None:
    """For each TMA descriptor of ``spec`` (a ``tensordesc`` argument), in the order of its arguments, what ``cuTensorMapEncodeTiled`` requires of
    its tensor besides the 16-byte address of its ``pointer_alignment`` precondition: strides that are multiples of 16 bytes (the innermost 1),
    dimensions that fit the ``i32`` the kernel gets them as (:func:`tilefish._tensormap.check_tensor_map`, a refusal of kind ``tensor_map``)."""
    from ._tensormap import check_tensor_map

    for a in spec.args:
        if a.kind == "tensordesc":
            info = infos[a.name]
            check_tensor_map(spec.stem, a.name, a.dtype or "", info.shape, canonical_strides(info.shape, info.strides))


def tensor_map_arg(spec: CompileSpec, name: str, info: TensorInfo, encode: Any) -> Any:
    """The launch argument of the descriptor ``name`` of ``spec`` for the tensor ``info`` (its real address, shape and strides): the
    ``CUtensorMap`` encoded by ``encode`` (a function of :class:`~tilefish._tensormap.EncodeArguments`: the driver's ``cuTensorMapEncodeTiled``),
    the shape and the canonical element strides (:class:`~tilefish._tensormap.TensorMapArg`). Encoded for every launch from what the launch has, so
    a descriptor is always of the tensor it is launched with (a CUDA graph holds the bytes of the launch it captured, of the addresses it holds)."""
    from . import _tensormap

    a = next(x for x in spec.args if x.name == name)
    assert a.kind == "tensordesc" and a.tensor_map is not None and a.dtype is not None
    strides = canonical_strides(info.shape, info.strides)
    arguments = _tensormap.encode_arguments(a.dtype, _tensormap.spec_of_json(a.tensor_map), info.ptr, info.shape, strides)
    return _tensormap.TensorMapArg(encode(arguments), tuple(info.shape), tuple(strides))


def _launch_values(prep: _Prepared) -> "list[tuple[str, str, Any]]":
    """The kernel's arguments in declaration order: ``(kind, name, value)`` with a tensor, an int or a constexpr value."""
    out = []
    for a in prep.spec.args:
        if a.kind in ("pointer", "tensordesc"):  # a tensor: its pointer, or the descriptor the launch encodes of it
            out.append((a.kind, a.name, prep.tensors[a.name]))
        elif a.kind in ("scalar", "given"):  # a runtime scalar, or a fixed argument per call (the value the call gave)
            out.append((a.kind, a.name, prep.scalars[a.name]))
        elif a.kind in ("fixed", "null"):
            continue  # a prebuilt binary's constant argument, or a null pointer: the value is the ABI's, and the launch writes it
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
    ``@tilefish.kernel(fp_fusion=False)``, and an ``extern`` with ``@tilefish.extern(enable_fp_fusion=False)``, which puts it in
    its specs, its manifest entries and its declaration digest; this is for an experiment on an ``extern`` whose spec says
    nothing about it.) The options
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
    return _ir.ir_digest(_ir.ttir_text(fn, inputs, arch=cap, options=options, what=f"'{spec.stem}' ({spec.triton})"), spec.language)


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
    _cudamath.check(f"'{spec.stem}' (sm_{cap})", built.cuda_math)  # torch's CUDA release against the one its math was measured on
    return built


def _launch_built(prep: _Prepared, built: BuiltKernel, index: int) -> None:
    """Launch a binary through the CUDA driver on torch's current stream, with the ABI of the manifest."""
    torch = _torch()
    check_launch_config(built, prep.spec)  # its warps are the declared ones (a warp-specialised kernel: those of its default partition, and more)
    with torch.cuda.device(index):
        args: "list[tuple[str, Any]]" = []
        for kind, name, v in _launch_values(prep):
            if kind == "pointer":
                args.append(("pointer", int(v.data_ptr())))
            elif kind == "tensordesc":  # encoded for this launch, from the tensor it is given
                args.append(("tensordesc", tensor_map_arg(prep.spec, name, tensor_info(name, v), _cuda.encode_tensor_map)))
            elif kind in ("scalar", "given"):
                args.append((kind, int(v)))
        loaded = _cuda.load(built, index)
        stream = torch.cuda.current_stream(index).cuda_stream
        if not built.launch.global_scratch_size:
            loaded.launch(prep.grid, args, stream)
            return
        # the global scratch of this launch: allocated on its stream (torch's blocks start on 512 bytes), freed after it (the caching allocator
        # reuses it on this stream only after the launch, which is ordered before any later work of the stream)
        scratch = torch.empty(max(loaded.scratch_bytes(prep.grid), 1), dtype=torch.uint8, device=f"cuda:{index}")
        loaded.launch(prep.grid, args, stream, scratch=int(scratch.data_ptr()))


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
    # an entry with no IR digest has nothing to compare, and one of the other frontend's IR is refused by the manifest without a frontend run
    if artifact is not None and artifact.entry.get("source_digest_kind") == _ir.digest_kind(spec.language):
        toolchain = artifact.entry.get("toolchain")
        digest, version = _current_ir(SimpleNamespace(target=target, spec=spec), cap, toolchain.get("triton") if isinstance(toolchain, dict) else None)  # type: ignore[arg-type]
    built = manifest.built_kernel(spec, arch, ir_digest=digest, ir_triton=version, pinned=pinned)
    if built.cuda_math:  # torch's CUDA release, and the libdevice a build here would link, against the entry's (tilefish._cudamath)
        _cudamath.check(f"'{spec.stem}' ({arch})", built.cuda_math, libdevice_here=_cudamath.libdevice_sha256(built.options))
    return built


def _run_replay(prep: _Prepared, manifest: Manifest, pinned: bool) -> None:
    """Launch the prebuilt cubin of the manifest for this spec and this device's architecture."""
    torch = _torch()
    index = _device_index(torch, prep.device)
    kernel = prep.target.kernel
    if kernel is not None and kernel.base is not None and manifest.get(prep.spec.stem, f"sm_{_capability(torch, index)}") is None:
        names = ", ".join(repr(n) for n in sorted(kernel.unaligned_params))
        raise TilefishManifestError(
            f"the call gives {names} of kernel '{kernel.base.name}' a tensor that is not 16-byte aligned, which runs the kernel's specialisation for "
            f"pointers aligned to their element only, '{kernel.name}' (its binaries do not assume 16 bytes), and the manifest in {str(manifest.directory)!r} "
            f"has no binary of it: export kernel.unaligned({names}) too (the export of a graph has the specialisations its calls use), or pass a "
            "16-byte aligned tensor"
        )
    _launch_built(prep, replay_built(prep.target, prep.spec, manifest, pinned, index), index)


_library_specs: "weakref.WeakKeyDictionary[LibraryOp, list[Any]]" = weakref.WeakKeyDictionary()  # an op is immutable
_pins: "dict[tuple, Any]" = {}  # (stem, arch, device, its SM count, workspace, cuBLASLt version, declaration) -> the BuiltLibrary that jit pinned in memory


def library_specs_of(op: LibraryOp) -> "list[Any]":
    """The specs of a library op with an implementation, one per bucket combination (cached): ``LibrarySpec`` of a cuBLASLt GEMM, ``ConvSpec`` of a
    cuDNN convolution."""
    from . import _conv, _library

    with _lock:
        specs = _library_specs.get(op)
    if specs is None:
        specs = _conv.conv_specs(op) if _conv.is_conv(op) else _library.library_specs(op)
        with _lock:
            _library_specs[op] = specs
    return specs


def _run_library(op: LibraryOp, args: "tuple[Any, ...]", kwargs: "dict[str, Any]", mode: str, manifest: "Manifest | None") -> Any:
    """Run a pinned library operation (a cuBLASLt GEMM, a cuDNN convolution). ``jit`` pins it in memory exactly as ``export`` does and runs it;
    ``replay`` runs the pin a manifest recorded, after checking that its entry is what the declaration asks for and that the library is the version
    that chose it; ``reference`` is torch: a GEMM is an f32 matmul of the operands with the bias added in f32, rounded once to the result's type, a
    convolution is computed in f64 and rounded once."""
    from . import _conv, _library
    from ._launch import TORCH_DTYPE_NAMES

    torch = _torch()
    call = bind_call(op.name, op.signature, args, kwargs)
    check_dim_constraints(call.dims, buckets=mode != "reference")
    conv = _conv.is_conv(op)
    if mode == "reference":
        device = _device_of(call.infos, call.tensors)
        return (_conv.reference_conv(op, call.tensors) if conv else _library.reference_of(op, call.tensors)).to(device)
    specs = library_specs_of(op)
    spec = select_spec(op.name, specs, call.dims)
    device = _device_of(call.infos, call.tensors)
    if getattr(device, "type", None) != "cuda":
        raise TilefishRuntimeError(f"jit and replay run on a CUDA device; the tensors are on {device}")
    shape = [eval_dim(d, call.dims) for d in op.signature.returns[0].shape]
    out_dtype = getattr(torch, TORCH_DTYPE_NAMES[op.signature.returns[0].dtype.name].removeprefix("torch."))
    out = torch.empty(shape, dtype=out_dtype, device=device)
    check_spec_preconditions(spec, {**call.infos, "out": tensor_info("out", out)}, {}, call.dims)
    index = _device_index(torch, str(device))
    if mode == "jit":
        built = jit_pin(spec, index)
    else:
        assert manifest is not None
        built = replay_pin(spec, manifest, index)
    with torch.cuda.device(index):
        stream = torch.cuda.current_stream(index).cuda_stream
        if conv:
            _conv.run_conv(built, call.tensors, out, stream, index)
        else:
            _library.run_gemm(spec, built.algo, call.tensors, out, stream, index)
    return out


def jit_pin(spec: Any, index: int) -> Any:
    """The pin ``jit`` makes in memory for a library spec on device ``index`` (what ``export`` writes), cached for the process."""
    from . import _conv, _library

    cap = _capability(_torch(), index)
    # a pin is for one device: its index (two devices of one architecture may have other SM counts, a MIG slice fewer than the whole GPU) and its SM count
    if isinstance(spec, _conv.ConvSpec):
        from . import _cudnn

        key: "tuple[Any, ...]" = (spec.stem, f"sm_{cap}", index, _library.sm_count(index), spec.workspace_bytes, "cudnn", _cudnn.version(), spec.declaration_digest)
    else:
        from . import _cublaslt

        key = (spec.stem, f"sm_{cap}", index, _library.sm_count(index), spec.workspace_bytes, _cublaslt.version(), spec.declaration_digest)
    with _lock:
        built = _pins.get(key)
    if built is None:
        built = _conv.build_conv(spec, arch=cap, device_index=index) if isinstance(spec, _conv.ConvSpec) else _library.build_library(spec, arch=cap, device_index=index)
        with _lock:
            _pins[key] = built
    return built


def replay_pin(spec: Any, manifest: Manifest, index: int) -> Any:
    """The pin a manifest recorded for a library spec on device ``index``, after checking that it was pinned on a device of this one's SM count,
    that its entry is what the declaration asks for and that the library is the version that chose it."""
    from . import _conv, _library

    arch, sms = f"sm_{_capability(_torch(), index)}", _library.sm_count(index)
    if isinstance(spec, _conv.ConvSpec):
        return _conv.built_from_manifest(manifest, spec, arch, sms, index)
    return _library.built_from_manifest(manifest, spec, arch, sms, index)


def _run_extern_reference(op: ExternOp, args: "tuple[Any, ...]", kwargs: "dict[str, Any]") -> Any:
    """``reference`` mode of an extern that has a ``reference=`` (a generated op has one): the torch function of its inputs, which returns
    its ``Out`` tensors. An ``InOut`` tensor is not supported here (the reference returns, it does not update)."""
    sig = op.signature
    if any(p.access == "inout" for p in sig.tensor_inputs):
        raise TilefishRuntimeError(f"reference mode of the extern op '{op.name}' does not run an op with InOut parameters on its own")
    call = bind_call(op.name, sig, args, kwargs)
    check_dim_constraints(call.dims, buckets=False)
    assert op.reference is not None
    given = {p.name: call.consts[p.name] for p in sig.consts if p.default is MISSING and p.name in call.consts}  # fixed arguments per call
    res = op.reference(**{p.name: call.tensors[p.name] for p in sig.tensor_inputs}, **given)
    produced = [] if res is None else (list(res) if isinstance(res, (tuple, list)) else [res])
    if len(produced) != len(sig.tensor_outputs):
        raise TilefishRuntimeError(
            f"the reference of the extern op '{op.name}' returned {len(produced)} tensor(s), but the op has {len(sig.tensor_outputs)} Out parameter(s)"
        )
    return _result(produced)


def _run_reference(target: _Target, args: "tuple[Any, ...]", kwargs: "dict[str, Any]") -> Any:
    from ._reference import interpret

    if target.kernel is None and target.extern is not None and target.extern.reference is not None:
        return _run_extern_reference(target.extern, args, kwargs)
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
    _cudamath.check_graph(f"kernel '{kernel.name}'", kernel.graph, getattr(kernel, "math", "fast"))  # torch's ops are the reference's
    returned = interpret(kernel, tensors, call.dims)
    return _result(returned + outs)


def export(
    ops: Any,
    directory: "str | os.PathLike[str]",
    *,
    arch: "int | str | None" = None,
    overwrite: bool = False,
    buckets: Any = None,
    reuse: bool = False,
    verify: "bool | None" = None,
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

    A graph exports its launch plan (``plan.json`` next to the manifest; returns the :class:`~tilefish._plan.Plan`); **several graphs** (or
    ``(graph, buckets)`` pairs) export a *plan set*: one manifest for all of them, each binary built once whichever plans launch it,
    ``<graph>.plan.json`` per graph and ``plans.json``, the index whose digest pins every plan (returns the :class:`~tilefish._plan.PlanSet`; a replay
    takes ``plan=directory`` and ``pin=`` the plan's digest or the set's). A graph whose body has arithmetic (fused regions, :mod:`tilefish._fusion`)
    has every region of more than one op run against its ops one by one on this device, at every bucket, before anything is written, and a
    difference refuses the export; ``verify=False`` skips that (``None``: the graph's ``tilefish.fusion(verify=...)``, ``True`` by default). Such a
    graph also gets ``<graph>.fusion.json``, its fusion report.

    A library op with an ``impl`` (a cuBLASLt GEMM) is pinned for the device (``arch`` must be its architecture) and gets an entry in the manifest's
    ``library`` section; a prebuilt PTX is assembled by ``ptxas`` and a cubin is taken as it is (:mod:`tilefish._binary`).

    Returns the loaded and verified :class:`~tilefish.aot.Manifest` (``.pinned``: it is what ``hosts.rust.emit`` takes): ``.directory``, ``.path`` and ``.digest``, the
    digest of the whole manifest, which is what to record (in a build log, a lock file) and pass back as
    ``run(..., mode="replay", manifest=directory, pin=digest)`` to refuse a directory that was changed after the export.
    The digest is the one of the manifest that was written from the builds in memory; the directory is read back and
    must have the same digest, or this raises :class:`~tilefish.errors.TilefishManifestError`.
    """
    items = list(ops) if isinstance(ops, (list, tuple)) else [ops]
    is_graph = [
        (isinstance(o, GraphOp) and not isinstance(o, KernelOp))
        or (isinstance(o, tuple) and len(o) == 2 and isinstance(o[0], GraphOp) and not isinstance(o[0], KernelOp))
        for o in items
    ]
    if any(is_graph):
        if not all(is_graph):
            raise TilefishRuntimeError("export takes graphs, or kernels and externs, not both (a manifest of a plan holds the binaries of its plans); export them separately")
        if len(items) > 1:  # a plan set: one manifest for every plan (docs/plan-schema-1.md, section 15)
            from ._planrun import export_graphs

            return export_graphs(items, directory, arch=arch, overwrite=overwrite, buckets=buckets, reuse=reuse, verify=verify)
        from ._planrun import export_graph

        graph, own = items[0] if isinstance(items[0], tuple) else (items[0], buckets)
        return export_graph(graph, directory, arch=arch, overwrite=overwrite, buckets=own, reuse=reuse, verify=verify)
    if verify is not None:
        raise TilefishRuntimeError("verify= is for exporting a graph (it checks the graph's fused regions); kernels and externs have none")
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
        if isinstance(op, LibraryOp) and op.impl is not None:  # a pinned library operation (a cuBLASLt GEMM, a cuDNN convolution): an entry, no file
            from ._conv import build_conv, is_conv
            from ._library import build_library

            for lspec in library_specs_of(op):
                pinned_library.append(build_conv(lspec, arch=cap, canary=True) if is_conv(op) else build_library(lspec, arch=cap))
            continue
        target = _target(op).at(cap)  # an extern declared with by_arch=: the implementation of the target architecture
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
    """``run`` of a ``@tilefish.graph``: its launch plan (:mod:`tilefish._plan`), run by :mod:`tilefish._planrun`. ``plan=``, ``outputs=`` and ``taps=`` are the
    three keywords that are not arguments of the graph (a graph with a parameter of those names takes it positionally)."""
    plan = kwargs.pop("plan", None)
    outputs = kwargs.pop("outputs", None)
    taps = kwargs.pop("taps", None)
    if manifest is not None:
        raise TilefishRuntimeError("manifest= is for a kernel or an extern; a graph replays plan=<directory> (written by tilefish.runtime.export(graph, directory))")
    if mode != "replay" and (pin is not None or trust_directory):
        raise TilefishRuntimeError(f"pin= and trust_directory= are for mode='replay' (they vouch for a plan directory); mode {mode!r} runs no plan directory")
    if trust_directory is not True and trust_directory is not False:
        raise TilefishRuntimeError(f"trust_directory= is True or False, got {trust_directory!r}")
    if pin is not None and not (isinstance(pin, str) and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin)):
        raise TilefishRuntimeError(f"pin= is the 64 lowercase hex digits of a plan digest, got {pin!r}")
    from ._planrun import run_graph

    return run_graph(op, args, kwargs, mode=mode, plan=plan, pin=pin, trust_directory=trust_directory, outputs=outputs, taps=taps)


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
    taps: Any = None,
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
        op, externals, mode=mode, plan=plan, pin=pin, trust_directory=trust_directory, devscalars=devscalars, outputs=outputs, stream=stream, taps=taps
    )


def bind_weights(op: GraphOp, weights: Any, **options: Any) -> Any:
    """Bind the weights of a graph whose plan declares its roles (``@tilefish.graph(inputs=..., weights=...)``) once:
    ``w = bind_weights(graph, {"w1": t1, ...}, mode="replay", plan="build/", pin=p.digest)``, then ``w.run(x, n)`` and ``w.capture(x, devscalars=...)``
    take only the other arguments. See :func:`tilefish._planrun.bind_weights`."""
    from ._planrun import bind_weights as _bind_weights

    return _bind_weights(op, weights, **options)


def prepare(op: GraphOp, **options: Any) -> None:
    """Do ahead of time what a run of a graph's plan does for itself the first time (allocate its arena, build or load every binary, create the cuBLASLt
    handle), none of which can happen while a CUDA graph is being captured. ``prepare(graph, mode="jit" | "replay", plan=..., pin=..., device=..., buckets=...)``;
    see :func:`tilefish._planrun.prepare`."""
    from ._planrun import prepare as _prepare_plan

    _prepare_plan(op, **options)


def warm_cudnn(device: Any = None) -> None:
    """Load cuDNN and create the cuDNN handle of ``device`` (default: the current device) now. ``cudnnCreate`` waits for **all** the work queued on the device,
    on every stream (measured with cuDNN 9.10.2 on a B300: one second of work queued on the NULL stream, on a blocking stream, on a non-blocking stream or on
    a torch stream each made it take one second), so the first pinned convolution prepared while other work runs stalls behind that work. Called at start-up,
    before any work is queued, it costs a few milliseconds; the handle is kept for the process, and every convolution (alone, in a plan, in any manifest)
    uses it, so none makes one again. A plan with convolutions creates it in :func:`prepare` too. (The first run of a cuDNN engine on the device, a replay's
    canary, may also wait for the device while CUDA loads its kernels lazily: prepare the convolutions at start-up too.)"""
    from . import _cudnn

    index = device if device is None or isinstance(device, int) else _device_index(_torch(), str(device))
    _cudnn.warm(index)


def run_step(op: GraphOp, step: int, buffers: Any, **options: Any) -> None:
    """Launch one step of a graph's plan, with the binary and the binding of its arguments the (verified) plan says, on the caller's tensors:
    ``run_step(graph, 3, {"t2": a, "w_o": w, "t3": o}, mode="replay", plan="build/", pin=p.digest, bucket=None)``. ``buffers`` gives a tensor for every
    buffer the step binds (by name; a returned value also as ``out<i>``; not the workspace, which is the plan's arena), each of the buffer's dtype and its
    shape at the bucket, contiguous; the step reads and writes them through the views the plan gives it. ``bucket`` is found from the shapes when one
    bucket fits them. A single-op replay under the plan's pin: see :func:`tilefish._planrun.run_step` and ``docs/plan-schema-1.md``, section 9."""
    from ._planrun import run_step as _run_step

    _run_step(op, step, buffers, **options)


def share_arena(graphs: Any, **options: Any) -> Any:
    """The plans of several graphs (the plans of a set) on **one arena** they share and run **one at a time**: a
    :class:`~tilefish._planrun.SharedArena`, ``s = share_arena([g1, g2, g3], mode="replay", plan="build/", pin=planset.digest)``, then ``s.run(g1, x)``,
    ``s.capture(g2, ...)``, ``s.bind_weights(g3, {...})``, ``s.run_step(...)``, ``s.prepare()``. The arena has the bytes of the largest arena of the plans,
    each plan's buffers at their own offsets in it, then the carried buffers the plans declare; every execution on it, of any of the plans, is ordered with
    every other (one lock, one order event). What a later plan reads is a returned value of an earlier one, or a carried buffer (``@tilefish.graph(carried=
    ("h",))``), which the arena holds between executions. ``name=`` makes another arena of the same plans (one per stream: executions on two arenas run
    concurrently). See :func:`tilefish._planrun.share_arena` and ``docs/plan-schema-1.md``, sections 15.1 and 15.2."""
    from ._planrun import share_arena as _share_arena

    return _share_arena(graphs, **options)


def replay_graph(graph: Any, stream: Any = None) -> None:
    """Replay a ``torch.cuda.CUDAGraph`` in which ``tilefish.run`` of plans was captured, ordered with every execution of those plans on any thread
    (under the lock of the arenas such graphs hold): ``replay_graph(g)`` instead of ``g.replay()`` where another thread may run the same plans. See
    :func:`tilefish._planrun.replay_graph` and ``docs/plan-schema-1.md``, section 10, item 7."""
    from ._planrun import replay_graph as _replay_graph

    _replay_graph(graph, stream)


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
    returned values into instead of allocating them (a list in order, or a mapping by name). ``taps={"t3": tensor, ("t5", 4): tensor}`` copies buffers of the
    plan into the caller's tensors right after a step (by default the last one that writes the buffer), to observe the intermediates of the very plan a
    product runs: the plan and its digest are unchanged (``Plan.shape_of`` gives a buffer's shape at a bucket). See :mod:`tilefish._plan` and
    ``docs/plan-schema-1.md``.

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
    manifest. ``outputs=`` gives the tensors the results are written into (one tensor, or one per result in the order the call
    returns them) instead of allocating them; each must have the dtype, the rank and the shape of its parameter and be contiguous,
    and it is what binds a dimension that only an ``Out`` parameter mentions (an extern that writes a ``[B]`` tensor and reads
    no tensor of ``B``), which a call cannot run without. ``mode``, ``manifest``, ``pin``, ``trust_directory`` and ``outputs`` are
    the only keywords that are not arguments of ``op`` (an operator with a parameter of one of those names takes it positionally). Raises
    :class:`~tilefish.errors.TilefishPreconditionError` when an argument breaks a precondition of the binary,
    :class:`~tilefish.errors.TilefishManifestError` when the manifest is not what was built, not what the operator asks
    for, or not the pinned one, and :class:`~tilefish.errors.TilefishRuntimeError` when the call cannot run (a replay
    without a trust anchor, for one).
    """
    if mode not in MODES:
        raise TilefishRuntimeError(f"unknown mode {mode!r}: use one of {', '.join(MODES)}")
    if isinstance(op, GraphOp) and not isinstance(op, KernelOp):
        return _run_graph(op, args, kwargs, mode, manifest, pin, trust_directory)
    outputs = kwargs.pop("outputs", None)
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
    if outputs is not None and (is_library or mode == "reference"):
        raise TilefishRuntimeError(
            "outputs= gives the tensors a kernel or an extern writes its results into, in jit and replay (and a graph's, in any mode); "
            + ("a library op allocates its result" if is_library else "reference mode allocates the results it computes")
        )
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
    prep = _prepare(target, args, kwargs, outputs)
    if mode == "jit":
        _run_jit(prep)
    else:
        assert loaded is not None
        _run_replay(prep, loaded, pinned=pin is not None)
    return _result(prep.results)
