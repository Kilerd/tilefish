"""Building compile specs into cubins with Triton, and the manifest that describes the result.

``build(spec, arch=...)`` feeds a verified :class:`~tilefish.aot.CompileSpec` to ``triton.compile`` exactly
as ``spec.ast_source_inputs()`` says (this is the recipe in the docstring of :mod:`tilefish.aot`) and
returns a :class:`BuiltKernel`: the cubin and the PTX, the symbol, the shared memory size, both digests and
the toolchain versions. ``write_manifest`` writes what a host needs to load and launch the cubins without
Python: for each kernel the ABI, the grid rule, the block, the compiled-in constants and the digests.

Triton is imported inside the functions that need it, so ``import tilefish`` never imports it.
"""

from __future__ import annotations

import functools
import importlib
import json
import pathlib
import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from ._elf import parse_elf, code_digest
from .errors import TilefishBuildError, TilefishManifestError

if TYPE_CHECKING:
    from .aot import CompileSpec

MANIFEST_SCHEMA = 1
SOURCE_DIGEST_KIND = "triton.JITFunction.cache_key"
MIN_TRITON = (3, 5)
_ARCH = re.compile(r"(?:sm_?)?([0-9]{2,3})a?\Z")
_ENTRY = re.compile(r"\.entry\s+(\w+)\s*\(")
_PTX_PARAM = re.compile(r"\.param\s+\.(\w+)")
_PTX_TARGET = re.compile(r"^\s*\.target\s+(\S+)", re.M)
_MISSING = object()

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
            entry["dtype"] = a.dtype
            entry["access"] = a.access
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

    ``source_digest`` is Triton's ``JITFunction.cache_key`` of the kernel at build time, or ``None`` for a binary
    that was not built here (a loaded manifest entry without one). ``cache_key`` is a sha256 of the text of the
    function, of the ``cache_key`` of every ``@triton.jit`` function it calls (transitively), of the line number
    where the function starts, and of the values of the ``tl.constexpr`` globals it reads. It does not cover
    other Python globals (Triton only compares their values at launch), the Triton version, or the compile
    options; and it changes when the function merely moves in its file. It is what tells a manifest of a
    hand-written kernel that its source was edited, which no declaration can.
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
        would write elsewhere), or this raises :class:`~tilefish.errors.TilefishBuildError`.
        """
        name = self.cubin_name
        if pathlib.PurePath(name).name != name or name.startswith(".") or "\0" in name or "\\" in name:
            raise TilefishBuildError(f"the cubin file name {name!r} is not a plain file name; refusing to write outside the directory")
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        if path.resolve().parent != directory.resolve():
            raise TilefishBuildError(f"the cubin path {str(path)!r} is not inside {str(directory)!r}; refusing to write it")
        path.write_bytes(self.cubin)
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


def build(
    spec: "CompileSpec",
    *,
    arch: "int | str",
    fn: Any = None,
    options: "Mapping[str, Any] | None" = None,
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
    """
    cap = parse_arch(arch)
    inp = spec.ast_source_inputs()  # raises TilefishSpecError for a spec that is not verified
    triton = _import_triton()
    jit =fn if fn is not None else resolve_triton(spec.triton)
    _check_jit_function(jit, spec)
    opts: dict[str, Any] = {**inp.options, **dict(options or {})}

    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    src = ASTSource(jit, signature=inp.signature, constexprs=None, attrs=inp.attrs)
    src.constants = inp.constants
    try:
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
    return BuiltKernel(
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
        source_digest=getattr(jit, "cache_key", None),
    )


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
        "preconditions": spec_part["preconditions"],
        "options": dict(b.options),
        "declaration_digest": spec_part["declaration_digest"],
        "sha256": b.sha256,
        "sha256_code": b.sha256_code,
        "toolchain": {"triton": b.triton_version, "ptxas": b.ptxas_version},
    }
    if "source_sha256" in spec_part:
        entry["source_sha256"] = spec_part["source_sha256"]
    if b.source_digest is not None:  # Triton's cache_key of the kernel: what a hand-written kernel's source is checked by
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

    For each kernel: ``stem``, ``symbol``, ``abi`` (``params`` in launch order, then the two ``scratch`` slots),
    ``grid`` (an expression over the dimension names, with ``cdiv(a, b)``), ``block``, ``shared``, ``launch``
    (scratch sizes, PDL, cluster and cooperative flags: all zero or false for a binary a plain
    ``cuLaunchKernel`` can run), ``constexprs``, ``preconditions`` (what a host checks before it launches),
    ``declaration_digest`` and, for a generated kernel, ``source_sha256`` (the Triton source it was built from),
    ``source_digest`` (Triton's ``cache_key`` of the kernel), ``sha256`` (the file) and ``sha256_code`` (the machine
    code, independent of the build directory), and the ``toolchain`` (Triton and ptxas versions). With
    ``cubins=True`` the cubins are written next to the manifest under the file names the manifest gives.

    The manifest as a whole has a digest (:attr:`tilefish.aot.Manifest.digest`) that a host can pin.

    With ``merge=True`` the entries of a manifest that already exists at ``path`` are kept (those for other
    architectures, other kernels), and the new ones replace the entry of the same stem and architecture. An existing
    entry that is a *different* binary (other machine code, declaration, source, options or shared memory) is not
    replaced unless ``overwrite=True``; a merge into a manifest that does not verify is an error.
    """
    entries = list(built)
    doc = manifest_dict(entries)
    keys = [(e["stem"], e["arch"]) for e in doc["kernels"]]
    if len(set(keys)) != len(keys):
        raise TilefishBuildError("a manifest needs one build per (stem, arch); some appear twice")
    path = pathlib.Path(path)
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
    if cubins:
        for b in entries:
            b.write_cubin(path.parent)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
