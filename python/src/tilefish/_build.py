"""Building compile specs into cubins with Triton, and the manifest that describes the result.

``build(spec, arch=...)`` feeds a verified :class:`~tilefish.aot.CompileSpec` to ``triton.compile`` exactly
as ``spec.ast_source_inputs()`` says (this is the recipe in the docstring of :mod:`tilefish.aot`) and
returns a :class:`BuiltKernel`: the cubin and the PTX, the symbol, the shared memory size, both digests and
the toolchain versions. ``write_manifest`` writes what a host needs to load and launch the cubins without
Python: for each kernel the ABI, the grid rule, the block, the compiled-in constants and the digests.

Triton is imported inside the functions that need it, so ``import tilefish`` never imports it.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import json
import os
import pathlib
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping

from ._elf import parse_elf, code_digest
from . import _source
from ._source import SOURCE_DIGEST_KIND
from .errors import TilefishBuildError, TilefishManifestError

if TYPE_CHECKING:
    from .aot import CompileSpec

# 2: every pointer parameter of the ABI records its declared shape (``["Rows", 1024]``: a dimension expression, or the size of a static
# axis) and every entry the constraints of its named dimensions (``dims``), so that a host binds the dimensions from the shapes of
# its tensors and checks buckets, ``divisible_by``, ``max_elements`` and ``count_of`` from the manifest alone. Schema 1 had neither.
MANIFEST_SCHEMA = 2
MIN_TRITON = (3, 5)
_ARCH = re.compile(r"(?:sm_?)?([0-9]{2,3})a?\Z")
_ENTRY = re.compile(r"\.entry\s+(\w+)\s*\(")
_PTX_PARAM = re.compile(r"\.param\s+\.(\w+)")
_PTX_TARGET = re.compile(r"^\s*\.target\s+(\S+)", re.M)
_MISSING = object()
_compile_lock = threading.Lock()  # a Triton knob is process-wide state: one compile at a time while it is set

# the PTX type of each kind of launch parameter (all the AOT specs produce): a pointer is 64 bits, an i32 is 32
_PTX_TYPES = {"pointer": ("u64", 8), "scalar": ("u32", 4)}


def parse_arch(arch: "int | str") -> int:
    """``103``, ``"103"``, ``"sm_103"`` and ``"sm_90a"`` give the compute capability as an int (103, 90).

    Triton picks the ``a`` suffix of the PTX target itself (``sm_90a`` from 90 up); see ``BuiltKernel.ptx_target``.
    """
    if isinstance(arch, bool) or not isinstance(arch, (int, str)):
        raise TilefishBuildError(f"arch must be a compute capability such as 90 or 'sm_90', got {arch!r}")
    if isinstance(arch, int):
        cap = arch
    else:
        m = _ARCH.match(arch.strip())
        if m is None:
            raise TilefishBuildError(f"arch must look like 'sm_90' or 90, got {arch!r}")
        cap = int(m.group(1))
    if not 70 <= cap <= 999:
        raise TilefishBuildError(f"arch {arch!r} is not a supported compute capability (70 to 999, e.g. 80, 90, 100, 103)")
    return cap


def _import_triton() -> Any:
    try:
        import triton
    except ImportError as e:
        raise TilefishBuildError(
            "building a cubin needs Triton, which is not installed: pip install 'tilefish[gpu]' (triton>=3.5)"
        ) from e
    version = _version_tuple(getattr(triton, "__version__", "0"))
    if version < MIN_TRITON:
        raise TilefishBuildError(
            f"tilefish builds with Triton >= {'.'.join(map(str, MIN_TRITON))}, found {triton.__version__}"
        )
    return triton


def _version_tuple(text: str) -> "tuple[int, ...]":
    parts = []
    for piece in str(text).split("."):
        m = re.match(r"[0-9]+", piece)
        if m is None:
            break
        parts.append(int(m.group()))
    return tuple(parts)


@functools.lru_cache(maxsize=None)
def _ptxas_version(path: str) -> str:
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:
        raise TilefishBuildError(f"cannot read the version of ptxas at {path!r}: {e}") from e
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    keep = [ln for ln in lines if ln.startswith(("Cuda compilation tools", "Build "))]
    return "; ".join(keep) if keep else (lines[-1] if lines else "unknown")


def ptxas_version() -> str:
    """The version string of the ptxas Triton compiles with (``TRITON_PTXAS_PATH`` or its bundled one)."""
    triton = _import_triton()
    try:
        path = triton.knobs.nvidia.ptxas.path
    except Exception as e:  # noqa: BLE001 - whatever Triton raises when it cannot find ptxas
        raise TilefishBuildError(f"cannot locate the ptxas Triton compiles with: {e}") from e
    return _ptxas_version(str(path))


def resolve_triton(ref: str) -> Any:
    """Import ``module:function`` and return the object (a ``triton.JITFunction`` for a real kernel)."""
    module, _, name = ref.partition(":")
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise TilefishBuildError(
            f"cannot import the module '{module}' of the Triton kernel '{ref}': {e}. Put it on sys.path, or pass the "
            "JITFunction yourself (build(spec, fn=...)); for a lowered kernel, write the source with "
            "LoweredKernel.write(directory) and import it from there"
        ) from e
    try:
        return getattr(mod, name)
    except AttributeError:
        raise TilefishBuildError(f"the module '{module}' has no attribute '{name}' (Triton kernel '{ref}')") from None


def _check_jit_function(fn: Any, spec: "CompileSpec") -> None:
    from triton.runtime.jit import JITFunction

    if not isinstance(fn, JITFunction):
        what = "an autotuned or heuristics-wrapped kernel" if hasattr(fn, "fn") else type(fn).__name__
        raise TilefishBuildError(
            f"'{spec.triton}' is {what}, not a triton.jit function; a compile spec fixes every launch choice, so it "
            "needs the plain JITFunction (pass it as build(spec, fn=...))"
        )
    names = list(spec.signature)
    if list(fn.arg_names) != names:
        raise TilefishBuildError(
            f"the parameters of '{spec.triton}' {list(fn.arg_names)} are not the ones its compile spec "
            f"'{spec.stem}' was made for {names}"
        )


def refuse_symlink(path: pathlib.Path) -> None:
    """Raise :class:`~tilefish.errors.TilefishBuildError` when ``path`` is a symbolic link (dangling or not)."""
    if path.is_symlink():
        raise TilefishBuildError(f"{str(path)!r} is a symbolic link; refusing to write through it (remove it, or write elsewhere)")


def write_file(path: pathlib.Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file in the same directory and an atomic rename.

    A symbolic link at ``path`` is refused. The rename never follows a link, so even a link created between the
    check and the rename is replaced, not written through; the temporary file is removed when anything fails.
    """
    refuse_symlink(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


@dataclass(frozen=True)
class LaunchAttributes:
    """What the binary asks of the launch, besides grid, block and shared memory (all from Triton's metadata)."""

    global_scratch_size: int
    profile_scratch_size: int
    launch_pdl: bool
    cluster_dims: "tuple[int, int, int]"
    num_ctas: int
    cooperative: bool

    def replay_blockers(self) -> "list[str]":
        """The attributes a plain ``cuLaunchKernel`` replay does not set up, as ``name=value`` strings."""
        out = []
        if self.global_scratch_size:
            out.append(f"global_scratch_size={self.global_scratch_size}")
        if self.profile_scratch_size:
            out.append(f"profile_scratch_size={self.profile_scratch_size}")
        if self.launch_pdl:
            out.append("launch_pdl=True")
        if tuple(self.cluster_dims) != (1, 1, 1) or self.num_ctas != 1:
            out.append(f"cluster_dims={tuple(self.cluster_dims)}, num_ctas={self.num_ctas}")
        if self.cooperative:
            out.append("launch_cooperative_grid=True")
        return out

    @classmethod
    def from_dict(cls, d: "Mapping[str, Any]") -> "LaunchAttributes":
        """The inverse of :meth:`to_dict`."""
        return cls(
            global_scratch_size=int(d["global_scratch_size"]),
            profile_scratch_size=int(d["profile_scratch_size"]),
            launch_pdl=bool(d["launch_pdl"]),
            cluster_dims=tuple(int(c) for c in d["cluster_dims"]),  # type: ignore[arg-type]
            num_ctas=int(d["num_ctas"]),
            cooperative=bool(d["cooperative"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_scratch_size": self.global_scratch_size,
            "profile_scratch_size": self.profile_scratch_size,
            "launch_pdl": self.launch_pdl,
            "cluster_dims": list(self.cluster_dims),
            "num_ctas": self.num_ctas,
            "cooperative": self.cooperative,
        }


def abi_of(spec: "CompileSpec") -> dict[str, Any]:
    """The launch ABI a spec promises: ``params`` (the non-constexpr arguments, in declaration order, one
    ``kernelParams`` slot each) followed by the two ``scratch`` slots, which a launch passes as null 64-bit pointers."""
    params: list[dict[str, Any]] = []
    for a in spec.args:
        if a.kind == "constexpr":
            continue
        ptx_type, size = _PTX_TYPES[a.kind]
        entry: dict[str, Any] = {
            "slot": len(params),
            "index": a.index,
            "name": a.name,
            "kind": a.kind,
            "type": a.type,
            "ptx_type": ptx_type,
            "size": size,
        }
        if a.kind == "pointer":
            if a.shape_dims is None:
                raise TilefishBuildError(
                    f"the compile spec of '{spec.stem}' has no declared shape for '{a.name}' (it was written by a tilefish before "
                    "manifest schema 2): regenerate it with compile_specs(op)"
                )
            entry["dtype"] = a.dtype
            entry["access"] = a.access
            entry["shape"] = list(a.shape_dims)
        else:
            entry["dim"] = a.dim
            if a.count_of is not None:  # the value is the element count of that pointer argument
                entry["count_of"] = a.count_of
        params.append(entry)
    scratch = [
        {"slot": len(params) + k, "name": p.name, "type": p.type, "ptx_type": "u64", "size": 8, "value": p.value}
        for k, p in enumerate(spec.launch_params[len(params) :])
    ]
    return {"params": params, "scratch": scratch}


@dataclass(frozen=True, eq=False)
class BuiltKernel:
    """One compiled binary of a compile spec, with everything a host needs to load and launch it.

    ``cubin`` is the ELF file and ``ptx`` the PTX it was assembled from. ``sha256`` covers the whole file and
    ``sha256_code`` only the machine code (see :func:`tilefish._elf.code_digest`): two builds of one kernel from
    different directories have the same ``sha256_code`` and may differ in ``sha256``, because the debug
    information holds the build path. ``options`` are the options handed to ``triton.compile``.

    ``source_digest`` is the source digest of a hand-written kernel at build time (:func:`tilefish._source.source_digest`:
    a sha256 over the text of the kernel's module file, of every user module file reachable from its globals, and the
    Triton version), or ``None`` for a binary of a generated kernel (its spec carries ``source_sha256``), for one that
    was not built here, and for a kernel whose files cannot be read. It is what tells a manifest of a hand-written
    kernel that its source was edited, which no declaration can; any edit to a reachable file changes it.
    """

    spec: "CompileSpec"
    arch: str  # "sm_103"
    cubin: bytes
    ptx: str
    symbol: str
    shared: int
    num_warps: int
    sha256: str
    sha256_code: str
    triton_version: str
    ptxas_version: str
    ptx_target: str  # the ``.target`` of the PTX: "sm_103a"
    options: "dict[str, Any]"
    launch: LaunchAttributes
    source_digest: "str | None" = None

    @property
    def stem(self) -> str:
        return self.spec.stem

    @property
    def cubin_name(self) -> str:
        """``<stem>.<arch>.cubin``, the file name the manifest gives this binary."""
        return f"{self.spec.stem}.{self.arch}.cubin"

    def write_cubin(self, directory: "str | pathlib.Path") -> pathlib.Path:
        """Write the cubin to ``<directory>/<stem>.<arch>.cubin`` and return the path.

        The file name must be a plain name inside ``directory`` (a stem or an arch with a separator or ``..`` in it
        would write elsewhere) and must not be a symbolic link, or this raises :class:`~tilefish.errors.TilefishBuildError`.
        The file is written to a temporary file in ``directory`` and renamed into place, so a reader sees the old
        file or the new one, and nothing is written through a link.
        """
        directory = pathlib.Path(directory)
        path = self.cubin_path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_file(path, self.cubin)
        return path

    def cubin_path(self, directory: "str | pathlib.Path") -> pathlib.Path:
        """``<directory>/<cubin_name>``, after checking that the name is a plain file name and no symbolic link is in the way."""
        name = self.cubin_name
        if pathlib.PurePath(name).name != name or name.startswith(".") or "\0" in name or "\\" in name:
            raise TilefishBuildError(f"the cubin file name {name!r} is not a plain file name; refusing to write outside the directory")
        path = pathlib.Path(directory) / name
        refuse_symlink(path)
        return path

    def abi(self) -> dict[str, Any]:
        """The launch ABI of the spec; see :func:`abi_of`."""
        return abi_of(self.spec)

    def __repr__(self) -> str:
        return (
            f"<BuiltKernel {self.spec.stem} {self.arch} {self.symbol} {len(self.cubin)} bytes "
            f"code={self.sha256_code[:12]}>"
        )


def _launch_attributes(md: Any) -> LaunchAttributes:
    def need(name: str) -> Any:
        v = getattr(md, name, _MISSING)
        if v is _MISSING:
            raise TilefishBuildError(
                f"Triton's kernel metadata has no '{name}': this Triton is not supported (tilefish needs >= "
                f"{'.'.join(map(str, MIN_TRITON))})"
            )
        return v

    return LaunchAttributes(
        global_scratch_size=int(need("global_scratch_size")),
        profile_scratch_size=int(need("profile_scratch_size")),
        launch_pdl=bool(need("launch_pdl")),
        cluster_dims=tuple(int(c) for c in need("cluster_dims")),  # type: ignore[arg-type]
        num_ctas=int(need("num_ctas")),
        cooperative=bool(need("launch_cooperative_grid")),
    )


def _check_abi(spec: "CompileSpec", ptx: str, symbol: str) -> None:
    """The PTX entry point must take exactly the parameters the spec's launch list promises (types in order)."""
    m = next((e for e in _ENTRY.finditer(ptx) if e.group(1) == symbol), None)
    if m is None:
        raise TilefishBuildError(f"the PTX of '{spec.stem}' has no entry point '{symbol}'")
    close = ptx.find(")", m.end())
    if close < 0:
        raise TilefishBuildError(f"the PTX entry '{symbol}' of '{spec.stem}' has no parameter list end")
    found = _PTX_PARAM.findall(ptx[m.end() : close])
    expected = [_PTX_TYPES[a.kind][0] for a in spec.args if a.kind != "constexpr"] + ["u64", "u64"]
    if found != expected:
        raise TilefishBuildError(
            f"the binary of '{spec.stem}' does not have the ABI its spec promises: the PTX entry '{symbol}' takes "
            f"{found}, the spec's launch parameters {[p.name for p in spec.launch_params]} need {expected}"
        )


@contextlib.contextmanager
def _always_compile(triton: Any) -> "Iterator[None]":
    """Make ``triton.compile`` compile, never read its disk cache, for the duration of the block.

    Triton's disk cache is keyed by ``JITFunction.cache_key``, which does not see everything that changes a binary (a
    ``tl.constexpr`` read as a module attribute, a dtype alias, a constant in a helper module), so a changed kernel could
    be served its old cubin. A build that goes into a manifest, and a jit build, must be what the source says.

    The knob is process-wide: builds are serialized while it is set, and a Triton kernel that other code of the same
    process compiles in the meantime compiles from scratch too (slower, never wrong). It is put back as it was.
    """
    knobs = getattr(triton, "knobs", None)
    compilation = getattr(knobs, "compilation", None)
    if compilation is None or not hasattr(compilation, "always_compile"):
        raise TilefishBuildError(
            f"this Triton ({triton.__version__}) has no knobs.compilation.always_compile, which a build needs to be sure it "
            f"does not get a stale binary from Triton's cache (tilefish needs >= {'.'.join(map(str, MIN_TRITON))})"
        )
    with _compile_lock:
        previous = compilation.always_compile
        compilation.always_compile = True
        try:
            yield
        finally:
            compilation.always_compile = previous


def _check_source(jit: Any, spec: "CompileSpec", expected: "str | None") -> "tuple[str | None, _source.Sources | None]":
    """Check that the source behind ``jit`` is the one the spec is for and return its source digest (``None``: not recorded)
    and the files and functions it reaches.

    A generated kernel's spec carries ``source_sha256``: the file of the function's module must have exactly that
    sha256 (for a lowered kernel that file is the generated source), whatever got imported from ``sys.path`` or was
    passed as ``fn``. A hand-written kernel's source digest is computed (:func:`tilefish._source.source_digest`) and, when
    the caller knows what it must be (``expected``), compared. For both, what the process holds must be what is on disk:
    every module the kernel reaches must be the file the process imported, and no constant a kernel function reads may
    have been assigned since (:class:`~tilefish._source.StaleSourceError`, see :mod:`tilefish._source`), and the lines of
    the kernel's functions that the process imported must be the ones in their files now.
    """
    reached = _source.reachable(jit)
    if spec.source_sha256 is not None:
        path = _source.function_file(jit)
        if path is None:
            raise TilefishBuildError(
                f"cannot check the source of '{spec.triton}' against the compile spec '{spec.stem}': the function is not "
                "defined in a Python source file"
            )
        try:
            found = _source.file_sha256(path)
        except OSError as e:
            raise TilefishBuildError(f"cannot read {path!r}, the source of '{spec.triton}': {e}") from e
        if found != spec.source_sha256:
            raise TilefishBuildError(
                f"'{spec.triton}' comes from {path!r}, whose sha256 is {found}, but the compile spec '{spec.stem}' is for "
                f"Triton source with sha256 {spec.source_sha256}: the file is not the source this kernel generates now (a stale "
                "LoweredKernel.write(directory) output on sys.path, or another function passed as fn). Refusing to build "
                "it, as the manifest would say it is the new source"
            )
        _source.check_current(reached)
        if reached is not None:
            _source.check_imported_lines(reached.functions)
        return None, reached
    digest = _source.digest_of(reached)  # checks the files and the values against what the process imported, then digests those bytes
    if reached is not None:
        _source.check_imported_lines(reached.functions)
    if digest is not None and expected is not None and digest != expected:
        raise TilefishBuildError(
            f"the source of '{spec.triton}' has digest {digest}, but {expected} was expected: it is not the source the "
            "caller meant to build"
        )
    return digest, reached


def build(
    spec: "CompileSpec",
    *,
    arch: "int | str",
    fn: Any = None,
    options: "Mapping[str, Any] | None" = None,
    source_digest: "str | None" = None,
    cache: bool = True,
) -> BuiltKernel:
    """Compile one spec to a cubin for ``arch`` (``"sm_103"`` or ``103``) with Triton.

    ``fn`` is the ``triton.jit`` function; by default it is imported from ``spec.triton`` (``module:function``).
    For a lowered kernel pass the JITFunction imported from the written source. The spec must be verified
    (:attr:`~tilefish.aot.CompileSpec.verified`; ``compile_specs`` output is).

    Triton is called as ``ASTSource(fn, signature=..., constexprs=None, attrs=...)``, then ``src.constants = ...``,
    then ``triton.compile(src, target=GPUTarget("cuda", arch, 32), options=...)``, with the three values and the
    options exactly as ``spec.ast_source_inputs()`` gives them. ``options`` adds Triton compile options on top
    of the spec's (an experiment, such as ``{"enable_fp_fusion": False}``); they are recorded in the result, and
    each one is checked against the metadata Triton returns, because Triton silently ignores an option it does
    not know. The result is checked against the spec: the options in effect, the symbol, and the parameters of
    the PTX entry (the ABI). Raises :class:`~tilefish.errors.TilefishBuildError`.

    The source is checked before anything is compiled. A spec with a ``source_sha256`` (a generated kernel) is built only
    from a function whose module file has exactly that sha256, so a stale file on ``sys.path`` (or another ``fn``) cannot
    be stamped with the digest of source it is not. For a hand-written kernel the result records the source digest of its
    files (:attr:`BuiltKernel.source_digest`), which must equal ``source_digest`` when that is given. In both cases every
    module the kernel reaches must be the file the process imported (an edited module that was not reloaded is refused:
    "module X changed on disk since it was imported; reload it or restart"), no constant that a kernel function reads may
    have been assigned to its module since, and the lines of the kernel's functions that the process imported must be the
    ones in their files now.

    The compile itself never reads Triton's disk cache: its key does not cover everything that changes a binary, so a
    build would not be sure to be what the source says. tilefish has a cache of its own instead (:mod:`tilefish._cache`:
    ``$TILEFISH_CACHE_DIR`` or ``~/.cache/tilefish``), keyed by the spec's declaration digest and fingerprint, the source
    identity (``source_sha256`` or the source digest, so only after the checks above), the architecture, the Triton and
    ``ptxas`` versions and the compile options, and verified on every read (the ``sha256`` and ``sha256_code`` of the cubin
    are recomputed, among other things): an entry that is not as it was written is a miss with a
    :class:`~tilefish.errors.TilefishCacheWarning`, and the kernel is compiled. A hit has the cubin of the build that made
    it, so its ``sha256`` (whole file, with the debug path of that build) can differ from a fresh compile's while its
    ``sha256_code`` is the same. ``cache=False`` neither reads nor writes it.
    """
    cap = parse_arch(arch)
    inp = spec.ast_source_inputs()  # raises TilefishSpecError for a spec that is not verified
    triton = _import_triton()
    jit = fn if fn is not None else resolve_triton(spec.triton)
    _check_jit_function(jit, spec)
    recorded_digest, reached = _check_source(jit, spec, source_digest)
    opts: dict[str, Any] = {**inp.options, **dict(options or {})}

    store = None
    cache_id = ""
    identity = ("source_sha256", spec.source_sha256) if spec.source_sha256 is not None else ("source_digest", recorded_digest)
    if cache and identity[1] is not None:  # after the checks of the source above: the key is of a source this process holds
        from . import _cache

        store = _cache.default_cache()
    if store is not None:
        triton_version, ptxas = str(triton.__version__), ptxas_version()
        cache_id = _cache.cache_key(
            spec,
            arch=cap,
            source=(identity[0], str(identity[1])),
            triton_version=triton_version,
            ptxas_version=ptxas,
            options=opts,
            environment=_cache.triton_environment(),
            values=_source.values_read(reached),
        )
        hit = store.load(
            cache_id, spec, arch=f"sm_{cap}", options=opts, triton_version=triton_version, ptxas_version=ptxas, source_digest=recorded_digest
        )
        if hit is not None:
            return hit

    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    src = ASTSource(jit, signature=inp.signature, constexprs=None, attrs=inp.attrs)
    src.constants = inp.constants
    try:
        with _always_compile(triton):
            compiled = triton.compile(src, target=GPUTarget("cuda", cap, 32), options=opts)
    except Exception as e:  # noqa: BLE001 - a compile failure of any kind is a build error, with Triton's message
        raise TilefishBuildError(f"Triton could not compile '{spec.stem}' ({spec.triton}) for sm_{cap}: {type(e).__name__}: {e}") from e

    md = compiled.metadata
    for key, want in opts.items():
        got = getattr(md, key, _MISSING)
        if got is _MISSING:
            raise TilefishBuildError(f"'{spec.stem}': Triton does not know the compile option '{key}' (it ignored it)")
        if got != want:
            raise TilefishBuildError(f"'{spec.stem}': the compile option {key}={want!r} was not applied (Triton used {got!r})")
    if getattr(md.target, "arch", None) != cap:
        raise TilefishBuildError(f"'{spec.stem}': asked for sm_{cap}, Triton built for {md.target!r}")
    symbol = str(md.name)
    cubin = bytes(compiled.asm["cubin"])
    ptx = str(compiled.asm["ptx"])
    elf = parse_elf(cubin)
    if elf.section(f".text.{symbol}") is None:
        raise TilefishBuildError(f"the cubin of '{spec.stem}' has no code section '.text.{symbol}'")
    _check_abi(spec, ptx, symbol)
    from hashlib import sha256

    target = _PTX_TARGET.search(ptx)
    built = BuiltKernel(
        spec=spec,
        arch=f"sm_{cap}",
        cubin=cubin,
        ptx=ptx,
        symbol=symbol,
        shared=int(md.shared),
        num_warps=int(md.num_warps),
        sha256=sha256(cubin).hexdigest(),
        sha256_code=code_digest(cubin),
        triton_version=str(triton.__version__),
        ptxas_version=ptxas_version(),
        ptx_target=target.group(1) if target else "",
        options=opts,
        launch=_launch_attributes(md),
        source_digest=recorded_digest,
    )
    if store is not None:
        store.store(cache_id, built)
    return built


# -- the manifest ---------------------------------------------------------------------------------------------------------


def spec_entry(spec: "CompileSpec") -> dict[str, Any]:
    """The part of a manifest entry that the compile spec alone determines (a replay checks it against the operator's spec)."""
    from .aot import _encode

    constexprs = []
    for a in spec.args:
        if a.kind != "constexpr":
            continue
        c: dict[str, Any] = {"index": a.index, "name": a.name, "value": _encode(a.value)}
        if a.dtype is not None:
            c["dtype"] = a.dtype
        constexprs.append(c)
    entry: dict[str, Any] = {
        "stem": spec.stem,
        "op": spec.op,
        "abi": abi_of(spec),
        "grid": spec.grid,
        "block": list(spec.block),
        "num_warps": spec.num_warps,
        "num_stages": spec.num_stages,
        "constexprs": constexprs,
        "buckets": dict(spec.buckets),
        "dims": {name: json.loads(json.dumps(c)) for name, c in spec.dims.items()},
        "preconditions": [p.to_dict() for p in spec.preconditions],
        "options": dict(spec.ast_source_inputs().options),
        "declaration_digest": spec.declaration_digest,
    }
    if spec.source_sha256 is not None:  # a generated kernel: the sha256 of the Triton source it was built from
        entry["source_sha256"] = spec.source_sha256
    return entry


def _kernel_entry(b: BuiltKernel) -> dict[str, Any]:
    spec_part = spec_entry(b.spec)
    entry = {
        "stem": spec_part["stem"],
        "op": spec_part["op"],
        "arch": b.arch,
        "ptx_target": b.ptx_target,
        "symbol": b.symbol,
        "cubin": b.cubin_name,
        "abi": spec_part["abi"],
        "grid": spec_part["grid"],
        "block": spec_part["block"],
        "shared": b.shared,
        "num_warps": b.num_warps,
        "num_stages": spec_part["num_stages"],
        "launch": b.launch.to_dict(),
        "constexprs": spec_part["constexprs"],
        "buckets": spec_part["buckets"],
        "dims": spec_part["dims"],
        "preconditions": spec_part["preconditions"],
        "options": dict(b.options),
        "declaration_digest": spec_part["declaration_digest"],
        "sha256": b.sha256,
        "sha256_code": b.sha256_code,
        "toolchain": {"triton": b.triton_version, "ptxas": b.ptxas_version},
    }
    if "source_sha256" in spec_part:
        entry["source_sha256"] = spec_part["source_sha256"]
    if b.source_digest is not None:  # the digest of the files of a hand-written kernel: what its source is checked by
        entry["source_digest"] = b.source_digest
        entry["source_digest_kind"] = SOURCE_DIGEST_KIND
    return entry


def manifest_dict(built: "Iterable[BuiltKernel]") -> dict[str, Any]:
    """The manifest as a JSON-compatible dict: one entry per kernel, in the order given."""
    from . import __version__

    return {
        "schema": MANIFEST_SCHEMA,
        "tilefish": __version__,
        "kernels": [_kernel_entry(b) for b in built],
    }


def _conflicts(old: "dict[str, Any]", new: "dict[str, Any]") -> "str | None":
    """What makes an existing manifest entry a different binary from a new one for the same stem and arch (``None``: the same)."""
    for key in ("sha256_code", "declaration_digest", "source_sha256", "source_digest", "options", "shared"):
        if old.get(key) != new.get(key):
            return key
    return None


def write_manifest(
    built: "Iterable[BuiltKernel]",
    path: "str | pathlib.Path",
    *,
    cubins: bool = False,
    merge: bool = False,
    overwrite: bool = False,
) -> pathlib.Path:
    """Write the manifest of built kernels as strict JSON to ``path`` and return it.

    For each kernel: ``stem``, ``symbol``, ``abi`` (``params`` in launch order, then the two ``scratch`` slots; a pointer
    records its declared ``shape``, a list of dimension expressions and sizes such as ``["Rows", 1024]``), ``dims`` (for each
    named dimension its ``divisible_by`` and ``buckets`` and, for a bucketed one, the ``bucket`` this binary is built for: a
    host binds the dimensions from the shapes of its tensors and checks them with these alone), ``grid`` (an expression over the
    dimension names, with ``cdiv(a, b)``), ``block``, ``shared``, ``launch``
    (scratch sizes, PDL, cluster and cooperative flags: all zero or false for a binary a plain
    ``cuLaunchKernel`` can run), ``constexprs``, ``preconditions`` (what a host checks before it launches),
    ``declaration_digest`` and, for a generated kernel, ``source_sha256`` (the Triton source it was built from), for a
    hand-written one ``source_digest`` (the digest of its files, see :mod:`tilefish._source`), ``sha256`` (the file) and
    ``sha256_code`` (the machine code, independent of the build directory), and the ``toolchain`` (Triton and ptxas
    versions). With ``cubins=True`` the cubins are written next to the manifest under the file names the manifest gives.

    Every file is written to a temporary file in its directory and renamed into place, and a symbolic link where a file
    is to go (``manifest.json`` or a cubin, dangling or not) is refused before anything is written.

    The manifest as a whole has a digest (:attr:`tilefish.aot.Manifest.digest`) that a host can pin.

    With ``merge=True`` the entries of a manifest that already exists at ``path`` are kept (those for other
    architectures, other kernels), and the new ones replace the entry of the same stem and architecture. An existing
    entry that is a *different* binary (other machine code, declaration, source, options or shared memory) is not
    replaced unless ``overwrite=True``; a merge into a manifest that does not verify is an error.
    """
    return write_manifest_doc(built, path, cubins=cubins, merge=merge, overwrite=overwrite)[0]


def write_manifest_doc(
    built: "Iterable[BuiltKernel]",
    path: "str | pathlib.Path",
    *,
    cubins: bool = False,
    merge: bool = False,
    overwrite: bool = False,
) -> "tuple[pathlib.Path, dict[str, Any]]":
    """:func:`write_manifest`, returning the path and the manifest document that was written (what ``export`` checks the directory against)."""
    entries = list(built)
    doc = manifest_dict(entries)
    keys = [(e["stem"], e["arch"]) for e in doc["kernels"]]
    if len(set(keys)) != len(keys):
        raise TilefishBuildError("a manifest needs one build per (stem, arch); some appear twice")
    path = pathlib.Path(path)
    refuse_symlink(path)
    targets = [b.cubin_path(path.parent) for b in entries] if cubins else []  # every name is checked before a byte is written
    if merge and path.exists():
        from ._manifest import Manifest

        existing = Manifest.load(path)
        new_by_key = dict(zip(keys, doc["kernels"]))
        kept: list[dict[str, Any]] = []
        for old in existing.doc["kernels"]:
            key = (old["stem"], old["arch"])
            new = new_by_key.pop(key, None)
            if new is None:
                kept.append(old)
                continue
            why = _conflicts(old, new)
            if why is not None and not overwrite:
                raise TilefishManifestError(
                    f"the manifest {str(path)!r} already has a different binary for '{key[0]}' on {key[1]} ({why} differs); "
                    "pass overwrite=True to replace it"
                )
            kept.append(new)
        doc["kernels"] = kept + list(new_by_key.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    for b, target in zip(entries, targets):
        write_file(target, b.cubin)
    write_file(path, (json.dumps(doc, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return path, doc
