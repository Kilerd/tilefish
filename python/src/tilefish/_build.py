"""Building compile specs into cubins with Triton, and the manifest that describes the result.

``build(spec, arch=...)`` feeds a verified :class:`~tilefish.aot.CompileSpec` to ``triton.compile`` exactly
as ``spec.ast_source_inputs()`` says (this is the recipe in the docstring of :mod:`tilefish.aot`) and
returns a :class:`BuiltKernel`: the cubin and the PTX, the symbol, the shared memory size, the digests (of the binary, and
the ``ir_digest`` of the Triton IR it was compiled from, :mod:`tilefish._ir`) and the toolchain versions. ``write_manifest``
writes what a host needs to load and launch the cubins without Python: for each kernel the ABI, the grid rule, the block, the
compiled-in constants and the digests.

Triton is imported inside the functions that need it, so ``import tilefish`` never imports it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib
import json
import math
import os
import pathlib
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping

from ._elf import parse_elf, code_digest
from . import _ir, _source
from ._specvalid import IMPLEMENTATION_FALLBACK, stride_of_json
from .errors import TilefishBuildError, TilefishManifestError

BINARY_DIGEST_KIND = "tilefish.binary.v1"  # the ``source_digest_kind`` of an entry whose ``source_digest`` is the sha256 of a prebuilt PTX or cubin

if TYPE_CHECKING:
    from .aot import CompileSpec

# 2: every pointer parameter of the ABI records its declared shape (``["Rows", 1024]``: a dimension expression, or the size of a static
# axis) and every entry the constraints of its named dimensions (``dims``), so that a host binds the dimensions from the shapes of
# its tensors and checks buckets, ``divisible_by``, ``max_elements`` and ``count_of`` from the manifest alone. Schema 1 had neither.
# 3: strided views (a pointer records ``strides`` and ``offset``, a stride scalar ``stride_of``, the ``max_extent`` precondition), device
# scalars (a pointer with the shape ``[]``), prebuilt binaries (an entry of ``kind`` ``"binary"``, ``fixed`` arguments, no scratch
# slots: the ABI records ``scratch_slots``), the ``library`` section (a pinned cuBLASLt GEMM) and the digest tag ``v3``.
# ``docs/manifest-schema-3.md`` specifies every field.
MANIFEST_SCHEMA = 3
LEGACY_MANIFEST_SCHEMA = 2  # what the Rust host reads; a manifest of this schema is refused here (export it again)
ENTRY_KIND_TRITON = "triton"
ENTRY_KIND_BINARY = "binary"
MIN_TRITON = (3, 5)
_ARCH = re.compile(r"(?:sm_?)?([0-9]{2,3})a?\Z")
_ENTRY = re.compile(r"\.entry\s+(\w+)\s*\(")
# one parameter of a PTX entry: ``.param .u64 .ptr .global .align 1 k_param_0`` (a scalar or a pointer: its type) or ``.param .align 64 .b8
# k_param_0[128]`` (a by-value array, a CUtensorMap: ``b8[128]``)
_PTX_PARAM = re.compile(r"\.param\s+(?:\.align\s+\d+\s+)?\.(\w+)[^,\[)]*?(?:\[(\d+)\])?\s*(?=,|\)|$)")
TENSORDESC_PTX = "b8[128]"  # the PTX type of a CUtensorMap parameter, as the manifest names it (``.param .align 64 .b8 name[128]``)
_PTX_TARGET = re.compile(r"^\s*\.target\s+(\S+)", re.M)
# the threads a PTX entry requires of a launch, among the directives between its parameter list and its body: ``.reqntid 256`` (or
# ``.reqntid 256, 1, 1``); Triton 3.5.1 writes it for every kernel (``32 * num_warps``, the total of a warp-specialised kernel's partitions)
_PTX_REQNTID = re.compile(r"\.reqntid\s+(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?")
# a warp-specialised region of a kernel in its TTGIR (Gluon's ``gl.warp_specialize``, or the partitions Triton's own warp specialisation makes):
# ``ttg.warp_specialize(%a, %b) attributes {...}``, or ``%0:2 = ttg.warp_specialize(...)`` when the default partition returns values
_WARP_SPECIALIZE = re.compile(r"^\s*(?:%[^=\n]*=\s*)?ttg\.warp_specialize\(", re.M)
MAX_WARPS = 32  # 1024 threads: the most a block has
_MISSING = object()
_compile_lock = _ir.LOCK  # a Triton knob is process-wide state, and Triton's frontend is not known to be thread-safe: one compile (or frontend) at a time

# the PTX type of each kind of launch parameter: a pointer is 64 bits, a scalar is as wide as its C type says
_PTX_SCALARS = {"i32": ("u32", 4), "i64": ("u64", 8), "f32": ("f32", 4)}


def ptx_type_of(kind: str, type_: str) -> "tuple[str, int]":
    """The PTX type name and the size in bytes of one launch parameter, as Triton declares it (a descriptor: its first slot, the CUtensorMap)."""
    if kind == "tensordesc":
        return (TENSORDESC_PTX, 128)
    return ("u64", 8) if kind == "pointer" else _PTX_SCALARS[type_]


def ptx_slots_of(kind: str, type_: str, rank: int = 0) -> "list[tuple[str, int]]":
    """The PTX type and size of every ``kernelParams`` slot one launch parameter takes: one for a pointer or a scalar, and for a TMA descriptor of a
    tensor of ``rank`` axes the CUtensorMap, then ``rank`` ``u32`` (its shape) and ``rank`` ``u64`` (its element strides), as Triton 3.5.1
    expands a host tensor descriptor."""
    if kind == "tensordesc":
        return [(TENSORDESC_PTX, 128)] + [("u32", 4)] * rank + [("u64", 8)] * rank
    return [ptx_type_of(kind, type_)]


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


def check_implementation_arch(spec: "CompileSpec", cap: int) -> None:
    """The spec of the implementation an op declared with ``by_arch=`` declares for one architecture is built for that architecture only (a
    fallback for any): a manifest entry's ``implementation`` is its ``arch`` or ``"fallback"``, which both readers hold it to."""
    if spec.implementation is not None and spec.implementation not in (f"sm_{cap}", IMPLEMENTATION_FALLBACK):
        raise TilefishBuildError(
            f"'{spec.stem}' is the implementation declared for {spec.implementation}, not for sm_{cap}: build the implementation of sm_{cap} "
            f"(compile_specs(op, arch={cap}))"
        )


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
            f"'{spec.triton}' is {what}, not a {'gluon.jit' if spec.language == 'gluon' else 'triton.jit'} function; a compile spec fixes "
            "every launch choice, so it needs the plain JITFunction (pass it as build(spec, fn=...))"
        )
    _check_language(fn, spec)
    names = list(spec.signature)
    if list(fn.arg_names) != names:
        raise TilefishBuildError(
            f"the parameters of '{spec.triton}' {list(fn.arg_names)} are not the ones its compile spec "
            f"'{spec.stem}' was made for {names}"
        )


def _check_language(jit: Any, spec: "CompileSpec") -> None:
    """The function must be a kernel of the frontend the spec says (``spec.language``): a Gluon kernel for ``gluon``, a ``triton.jit`` one for
    ``triton`` (:func:`tilefish._ir.check_language`), checked before anything is run."""
    _ir.check_language(jit, spec.language, f"'{spec.stem}' ({spec.triton})")


def _check_identity(jit: Any, spec: "CompileSpec") -> None:
    """``fn`` must be the function ``spec.triton`` names (``module:function``), not another one with the same parameters.

    Without this a build of one kernel's spec from another kernel's function is a binary of the wrong kernel stamped with the first
    one's declaration, and the build cache would then serve it to a later, correct build. A function is the one the spec names when
    it is the very object ``spec.triton`` imports to, or is defined under that module and name. A generated kernel's module is a
    scratch import of the written source (its name has the digest of the source in it), so it is told by the function's name here
    and by the sha256 of its file (:func:`_check_source`).
    """
    module, _, name = spec.triton.partition(":")
    defined = _source.defined_in(jit)
    if spec.source_sha256 is not None:
        if defined[1] == name:
            return
    else:
        try:
            resolved = resolve_triton(spec.triton)
        except TilefishBuildError:
            resolved = None  # not importable here: only where it is defined can say
        if resolved is not None and (resolved is jit or _source.jit_function(resolved) is jit):
            return
        if defined == (module, name):
            return
    raise TilefishBuildError(
        f"the function passed as fn is '{defined[0]}:{defined[1]}', but the compile spec '{spec.stem}' is for '{spec.triton}': "
        "a build compiles the function the spec names (the spec says nothing about the body of another one), so it is refused, "
        "and nothing is compiled or cached"
    )


def _applied(key: str, want: Any, got: Any) -> bool:
    """Whether the option ``key`` that was asked for as ``want`` is what the metadata of the compile says (``got``).

    ``extern_libs`` is asked for as a mapping of names to paths; Triton keeps it as pairs (a list of lists once the metadata went through
    its cache file) and adds ``libdevice`` when it is not among them, so it is applied when every library asked for is there with its path.
    """
    if key == "extern_libs":
        try:
            have = {str(k): v for k, v in dict(got).items()}
            return all(have.get(str(k)) == v for k, v in dict(want).items())
        except (TypeError, ValueError):
            return False
    return bool(got == want)


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
    ``kernelParams`` slot each), followed by the ``scratch`` slots, which a launch passes as null 64-bit pointers, and
    ``scratch_slots``, their number: 2 for a Triton kernel, 0 for a prebuilt binary (its C ABI has none).

    A ``pointer`` is ``base + offset * itemsize`` of a tensor; with ``strides`` and ``offset`` (a ``Strided[...]`` parameter) the
    tensor is a view, and the ``scalar`` that is a ``stride_of`` carries one of its strides. A ``fixed`` argument is a scalar whose
    value is in the manifest (a prebuilt binary's constant argument, such as a scratch slot written out as a parameter); a ``given`` one is
    a fixed argument per call, whose value each call gives (a plan records it in the step).

    A ``tensordesc`` (a TMA descriptor of a tensor, ``TensorDesc(...)``) has the fields of a pointer and its ``tensor_map``, and takes ``1 + 2 * rank``
    consecutive slots from its ``slot`` on: the 128-byte CUtensorMap, the shape (``u32`` each), the element strides (``u64`` each); its ``type`` is
    ``"tensordesc"`` and its ``ptx_type``/``size`` are those of the first. The ``slot`` of every parameter is where its slots begin.
    """
    params: list[dict[str, Any]] = []
    slot = 0
    for a in spec.args:
        if a.kind == "constexpr":
            continue
        ptx_type, size = ptx_type_of(a.kind, a.type)
        entry: dict[str, Any] = {
            "slot": slot,
            "index": a.index,
            "name": a.name,
            "kind": a.kind,
            "type": "tensordesc" if a.kind == "tensordesc" else a.type,
            "ptx_type": ptx_type,
            "size": size,
        }
        slot += len(ptx_slots_of(a.kind, a.type, len(a.shape or ())))
        if a.kind == "tensordesc":
            entry["dtype"] = a.dtype
            entry["access"] = a.access
            entry["shape"] = list(a.shape_dims or ())
            if a.strides is not None:  # a descriptor of a view: its own stride dimensions (passed in its stride slots), the last axis's 1
                entry["strides"] = list(a.strides)
                entry["offset"] = a.offset
            entry["tensor_map"] = json.loads(json.dumps(a.tensor_map))
        elif a.kind == "pointer":
            if a.shape_dims is None:
                raise TilefishBuildError(
                    f"the compile spec of '{spec.stem}' has no declared shape for '{a.name}' (it was written by a tilefish before "
                    "manifest schema 2): regenerate it with compile_specs(op)"
                )
            entry["dtype"] = a.dtype
            entry["access"] = a.access
            entry["shape"] = list(a.shape_dims)
            if a.strides is not None:  # a pointer that takes a strided view: base + offset * itemsize, and these strides
                entry["strides"] = list(a.strides)
                entry["offset"] = a.offset
            if a.alias is not None:  # an out pointer that may be written over that in pointer: a launch takes the two as exactly one tensor
                entry["alias"] = a.alias
        elif a.kind == "fixed":
            from .aot import _encode

            entry["value"] = _encode(a.value)
        elif a.kind == "given":  # a fixed argument per call: the call (a step of a plan) gives the value
            pass
        else:
            entry["dim"] = a.dim
            if a.count_of is not None:  # the value is the element count of that pointer argument
                entry["count_of"] = a.count_of
            if a.stride_of is not None:  # the value is the stride of that pointer argument along that axis (of each of them, for a shared one)
                entry["stride_of"] = stride_of_json(a.stride_of)
        params.append(entry)
    scratch = [
        {"slot": slot + k, "name": p.name, "type": p.type, "ptx_type": "u64", "size": 8, "value": p.value}
        for k, p in enumerate(spec.launch_params[len(params) :])
    ]
    return {"params": params, "scratch": scratch, "scratch_slots": len(scratch)}


def abi_slot_sizes(abi: "Mapping[str, Any]") -> "list[int]":
    """The size of every ``kernelParams`` slot of a manifest's ABI, in order: each parameter's (a descriptor's ``1 + 2 * rank`` slots), then the
    scratch slots. What the loaded function must report (``cuFuncGetParamInfo``)."""
    sizes: "list[int]" = []
    for p in abi["params"]:
        if p["kind"] == "tensordesc":
            rank = len(p["shape"])
            sizes += [p["size"]] + [4] * rank + [8] * rank
        else:
            sizes.append(p["size"])
    return sizes + [p["size"] for p in abi["scratch"]]


@dataclass(frozen=True, eq=False)
class BuiltKernel:
    """One compiled binary of a compile spec, with everything a host needs to load and launch it.

    ``cubin`` is the ELF file and ``ptx`` the PTX it was assembled from. ``sha256`` covers the whole file and
    ``sha256_code`` only the machine code (see :func:`tilefish._elf.code_digest`): two builds of one kernel from
    different directories have the same ``sha256_code`` and may differ in ``sha256``, because the debug
    information holds the build path. ``options`` are the options handed to ``triton.compile``.

    ``ir_digest`` is the digest of the Triton IR the binary was compiled from (:func:`tilefish._ir.ir_digest`: the TTIR that
    Triton's frontend builds for the kernel, without locations or names), or ``None`` for one that was not built here. It is
    what tells a manifest that the kernel it was built from is not the kernel a process holds now, for a hand-written ``extern``
    (whose declaration says nothing about its body) and a lowered kernel alike: whatever the process holds that the frontend reads
    (the body, a constant read by any route, a closure) is in the IR, so a change of any of them changes it.

    ``num_warps`` is the warps of a block of the compiled kernel, as Triton's metadata says: the declared ``num_warps`` (the compile option),
    except for a **warp-specialised** kernel (``warp_specialized``: its TTGIR has ``ttg.warp_specialize`` regions, from Gluon's
    ``gl.warp_specialize`` or from Triton's own warp specialisation), whose partitions add their warps to the declared ones, the warps of its
    default partition: there it is their total (Triton's ``ttg.total-num-warps``, 8 for a 4-warp default partition and two 1-warp workers). A
    launch takes ``block`` = ``32 * num_warps`` threads, and a prebuilt binary (``num_warps`` 0) its declared block.
    """

    spec: "CompileSpec"
    arch: str  # "sm_103"
    cubin: bytes
    ptx: str
    symbol: str
    shared: int
    num_warps: int  # the warps of a block of the compiled kernel (a warp-specialised kernel: every partition's), 0 for a prebuilt binary
    sha256: str
    sha256_code: str
    triton_version: str
    ptxas_version: str
    ptx_target: str  # the ``.target`` of the PTX: "sm_103a"
    options: "dict[str, Any]"
    launch: LaunchAttributes
    ir_digest: "str | None" = None
    # a prebuilt binary (``spec.binary``): the sha256 of the PTX or cubin file it is made of (the ``source_digest`` of its entry, kind
    # ``"tilefish.binary.v1"``), and how it was made: ``{"source": "ptx", "ptxas_flags": [...]}`` or ``{"source": "cubin"}``
    binary_digest: "str | None" = None
    binary_build: "dict[str, Any] | None" = None
    # a warp-specialised kernel: its partitions add warps to the declared num_warps (its default partition's), and ``num_warps`` is their total
    warp_specialized: bool = False
    # a generated kernel whose ops are the bits of CUDA math library functions, or that links libdevice: the ``cuda_math`` of its manifest entry
    # (:mod:`tilefish._cudamath`: the functions, the CUDA releases they were measured on, the sha256 of the libdevice linked); ``None`` otherwise
    cuda_math: "dict[str, Any] | None" = None

    @property
    def stem(self) -> str:
        return self.spec.stem

    @property
    def block(self) -> "tuple[int, int, int]":
        """The thread block a launch takes: ``32 * num_warps`` threads of the compiled kernel (every partition of a warp-specialised one), or the
        block a prebuilt binary declares."""
        if self.spec.binary is not None:
            return tuple(self.spec.block)  # type: ignore[return-value]
        return (32 * self.num_warps, 1, 1)

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


def ptx_params(text: str) -> "list[str]":
    """The type of every parameter of a PTX entry's parameter list (``text``, from after its ``(``): ``u64``, ``u32``, ... and ``b8[128]`` for a
    by-value array of 128 bytes."""
    return [f"{ty}[{n}]" if n else ty for ty, n in _PTX_PARAM.findall(text)]


def _check_abi(spec: "CompileSpec", ptx: str, symbol: str) -> None:
    """The PTX entry point must take exactly the parameters the spec's launch list promises (types in order)."""
    m = next((e for e in _ENTRY.finditer(ptx) if e.group(1) == symbol), None)
    if m is None:
        raise TilefishBuildError(f"the PTX of '{spec.stem}' has no entry point '{symbol}'")
    close = ptx.find(")", m.end())
    if close < 0:
        raise TilefishBuildError(f"the PTX entry '{symbol}' of '{spec.stem}' has no parameter list end")
    found = ptx_params(ptx[m.end() : close])
    expected = [t for a in spec.args if a.kind != "constexpr" for t, _ in ptx_slots_of(a.kind, a.type, len(a.shape or ()))] + ["u64", "u64"]
    if found != expected:
        raise TilefishBuildError(
            f"the binary of '{spec.stem}' does not have the ABI its spec promises: the PTX entry '{symbol}' takes "
            f"{found}, the spec's launch parameters {[p.name for p in spec.launch_params]} need {expected}"
        )


def warps_problem(declared: int, num_warps: Any, warp_specialized: bool) -> "str | None":
    """What is wrong with a binary of a kernel declared with ``declared`` warps that is launched with ``num_warps`` (``None``: nothing).

    A kernel that is not warp-specialised has exactly the warps it was declared with (a silent change of them is an error: the block, the
    layouts and the reduction trees are written for that number). A warp-specialised one has more, from ``declared + 1`` (its partitions each
    have a warp at least) to 32 (1024 threads). A prebuilt binary (``declared`` 0) has no warps to change and is never warp-specialised.
    """
    if not isinstance(num_warps, int) or isinstance(num_warps, bool):
        return f"its warp count is {num_warps!r}, not an int"
    if not warp_specialized:
        return None if num_warps == declared else f"it has {num_warps} warps, but the declaration gives num_warps={declared}"
    if declared == 0:
        return "a prebuilt binary is not warp-specialised"
    if not declared < num_warps <= MAX_WARPS:
        return (
            f"it is warp-specialised with {num_warps} warps in all, but its partitions add warps to the {declared} of its default partition (the "
            f"declared num_warps), and a block has at most {MAX_WARPS}"
        )
    return None


def check_launch_config(built: BuiltKernel, spec: "CompileSpec") -> None:
    """Raise :class:`~tilefish.errors.TilefishRuntimeError` unless ``built`` is a binary of ``spec``'s launch configuration: its block, and the
    warps its declaration gives (all of them, or those of the default partition of a warp-specialised kernel)."""
    from .errors import TilefishRuntimeError

    problem = warps_problem(spec.num_warps, built.num_warps, built.warp_specialized)
    if built.spec.block != spec.block or built.spec.num_warps != spec.num_warps or problem is not None:
        raise TilefishRuntimeError(f"the binary of '{spec.stem}' disagrees with its spec about num_warps" + (f": {problem}" if problem else ""))


def is_warp_specialized(ttgir: Any) -> bool:
    """Whether a kernel's TTGIR (``compiled.asm["ttgir"]``) has a warp-specialised region (``ttg.warp_specialize``): Gluon's
    ``gl.warp_specialize``, or the partitions of Triton's own warp specialisation (``tl.range(..., warp_specialize=True)`` on sm_100 and up)."""
    return isinstance(ttgir, str) and _WARP_SPECIALIZE.search(ttgir) is not None


def ptx_reqntid(ptx: str, symbol: str) -> "int | None":
    """The threads per block the PTX entry ``symbol`` requires (its ``.reqntid``, the product of its sizes), or ``None`` when it states none."""
    m = next((e for e in _ENTRY.finditer(ptx) if e.group(1) == symbol), None)
    if m is None:
        return None
    close = ptx.find(")", m.end())
    body = ptx.find("{", close)
    if close < 0 or body < 0:
        return None
    found = _PTX_REQNTID.search(ptx, close, body)
    if found is None:
        return None
    return math.prod(int(g) for g in found.groups() if g is not None)


def check_threads(spec: "CompileSpec", ptx: str, symbol: str, num_warps: int) -> None:
    """The threads the PTX entry requires (``.reqntid``, which Triton writes for every kernel) must be ``32 * num_warps``, the block a launch takes:
    the machine code and the metadata agree about the warps of a block (the driver refuses a launch of a ``.reqntid`` kernel with another block)."""
    threads = ptx_reqntid(ptx, symbol)
    if threads is not None and threads != 32 * num_warps:
        raise TilefishBuildError(
            f"'{spec.stem}': the PTX entry '{symbol}' requires {threads} threads per block (.reqntid), but Triton's metadata says {num_warps} warps "
            f"({32 * num_warps} threads)"
        )


def check_tensordesc_meta(spec: "CompileSpec", meta: Any) -> None:
    """The compiled kernel expects, for each of its descriptor parameters in order, the map the declaration says (Triton's
    ``metadata.tensordesc_meta``: the swizzle, the element size and type, the box, no fp4 padding). A Triton kernel's compiler chooses the swizzle
    of a descriptor, so a declaration that names another is refused here with the compiler's choice; a kernel the compiler did not lower to TMA
    (an architecture before sm_90) has no metadata, and another launch ABI, and is refused too."""
    from . import _tensormap

    descs = [a for a in spec.args if a.kind == "tensordesc"]
    if not descs:
        return
    if not meta or len(meta) != len(descs):
        raise TilefishBuildError(
            f"'{spec.stem}': Triton did not lower the descriptor parameters {[a.name for a in descs]} to TMA descriptors (its metadata has "
            f"{len(meta or [])} of them): a TMA descriptor needs sm_90 or newer, and the kernel would take another launch ABI"
        )
    for a, got in zip(descs, meta):
        assert a.tensor_map is not None and a.dtype is not None
        want = _tensormap.spec_of_json(a.tensor_map).triton_meta(a.dtype)
        got = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(got).items()}
        if got != want:
            swizzles = {v: k for k, v in _tensormap.SWIZZLE_CODES.items()}
            hint = ""
            spec_map = _tensormap.spec_of_json(a.tensor_map)
            other = swizzles.get(got.get("swizzle"))  # type: ignore[arg-type]
            if other is not None:
                box = list(_tensormap.box_of(spec_map.block, other, _tensormap.ITEMSIZE[a.dtype] * 8))
                if {**want, "swizzle": got["swizzle"], "block_size": box} == got:  # the compiled kernel expects another swizzle (and so its box)
                    hint = f": the compiled kernel expects a {other}-byte swizzle, declare swizzle={other}"
            raise TilefishBuildError(
                f"'{spec.stem}': the descriptor '{a.name}' is declared {want}, but the compiled kernel expects {got} (Triton's tensordesc_meta){hint}"
            )


TF32_FIXES = (
    "Fixes: give that dot a tile of fewer than 64 rows (M = 32 or 16), which Triton 3.5.1 keeps on FMAs; write the product in a SIMT form (a loop "
    'of tl.fma over k, no tl.dot); or, where TF32 is what the kernel may compute, say so on the dot with input_precision="tf32" (or "tf32x3"), '
    "which this check never refuses"
)


def _text_or_none(value: Any) -> "str | None":
    return value if isinstance(value, str) else None


def check_tf32(
    spec: "CompileSpec", cap: int, frontend_ir: str, ptx: str, triton_version: str, *, ttir: "str | None" = None, ttgir: "str | None" = None
) -> None:
    """Refuse a binary whose kernel asks for IEEE f32 dots but was compiled into TF32 tensor-core MMAs (:mod:`tilefish._tf32guard`).

    ``frontend_ir`` is the IR the frontend built (the text of the IR digest), ``ptx`` the binary's, and ``ttir`` and ``ttgir`` those of the compile
    (``compiled.asm``; a cache hit has none). Every kernel Triton builds is checked, a lowered ``@kernel`` (it has no dot today) as much as an
    ``extern``; a kernel whose dots allow TF32 is never refused. Raises :class:`~tilefish.errors.TilefishBuildError` with the fixes.
    """
    from . import _tf32guard

    found = _tf32guard.problem(frontend_ir, ptx, language=spec.language, ttir=ttir, ttgir=ttgir)
    if found is None:
        return
    raise TilefishBuildError(
        f"'{spec.stem}' ({spec.triton}) for sm_{cap} with Triton {triton_version}: {found}. A TF32 MMA keeps 10 of the 23 bits of each f32 "
        "mantissa, so the binary would not compute what the kernel asks for, and nothing at run time would say so: refusing to build it. (Triton "
        "3.5.1 sends an IEEE f32 dot to tcgen05 on sm_100 and sm_103 when an operand derives from a load narrower than 32 bits, an f16, bf16 or "
        f"i8 tensor converted to f32, or from tl.join, and its tile has M % 64 == 0, N % 16 == 0 and 4 or 8 warps.) {TF32_FIXES}"
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


def _check_generated(jit: Any, spec: "CompileSpec") -> None:
    """For a lowered kernel, check that the function behind ``jit`` is the generated source the spec is for.

    A generated kernel's spec carries ``source_sha256``: the file of the function's module must have exactly that sha256 (that
    file is the generated source), whatever got imported from ``sys.path`` or was passed as ``fn``, and the lines the process
    imported of the function must be the ones in the file now (:func:`tilefish._source.check_imported_lines`). The IR digest says
    what the process holds, not whether that is the source the spec claims, and a manifest entry records both. A hand-written
    kernel's spec has no source: its identity is its IR alone (:mod:`tilefish._ir`).
    """
    if spec.source_sha256 is None:
        return
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
    _source.check_imported_lines(_source.jit_function(jit))


def _generated_cuda_math(jit: Any, spec: "CompileSpec", options: "Mapping[str, Any]") -> "dict[str, Any] | None":
    """The ``cuda_math`` of a generated kernel's entry (:mod:`tilefish._cudamath`), from the header line of its source (which
    :func:`_check_generated` has checked is the spec's) and the libdevice Triton links for ``options``; ``None`` for a hand-written kernel."""
    if spec.source_sha256 is None:
        return None
    from . import _cudamath

    path = _source.function_file(jit)
    if path is None:  # pragma: no cover - _check_generated refuses it first
        return None
    try:
        with open(path, encoding="utf-8") as f:
            source = f.read()
    except OSError as e:
        raise TilefishBuildError(f"cannot read {path!r}, the source of '{spec.triton}': {e}") from e
    return _cudamath.entry_field(source, options)


def _refuse_unidentified(triton: Any, spec: "CompileSpec") -> None:
    """Refuse a process state in which Triton does not compile the IR that identifies the binary.

    ``TRITON_KERNEL_OVERRIDE`` makes Triton replace the IR of a stage with a file; so does the compile option ``ir_override``
    (:func:`tilefish._cache.compile_files` refuses that); and an active Proton instrumentation (``CUDABackend.instrumentation``)
    patches the passes that lower the IR. In each the binary would not be compiled from the IR whose digest identifies it.
    """
    compilation = getattr(getattr(triton, "knobs", None), "compilation", None)
    if getattr(compilation, "override", False):
        raise TilefishBuildError(
            f"'{spec.stem}': Triton's kernel override is on (TRITON_KERNEL_OVERRIDE), so it may replace the IR of a stage with the "
            "contents of a file, and the binary would not be compiled from the IR whose digest identifies it. Switch it off to build "
            "(a build, an export and a cache entry are of the IR the kernel compiles to)"
        )
    try:
        from triton.backends.nvidia.compiler import CUDABackend
    except ImportError:  # not the NVIDIA backend (or not Triton at all)
        return
    if getattr(CUDABackend, "instrumentation", None) is not None:
        raise TilefishBuildError(
            f"'{spec.stem}': an instrumentation plugin of Triton is active (CUDABackend.instrumentation, as Proton's instrumentation "
            "backend sets it), and it patches the passes that lower the IR, so the binary would not be compiled from the IR whose "
            "digest identifies it. Stop the profiler to build"
        )


def build(
    spec: "CompileSpec",
    *,
    arch: "int | str",
    fn: Any = None,
    options: "Mapping[str, Any] | None" = None,
    ir_digest: "str | None" = None,
    cache: bool = True,
) -> BuiltKernel:
    """Compile one spec to a cubin for ``arch`` (``"sm_103"`` or ``103``) with Triton.

    ``fn`` is the ``triton.jit`` function; by default it is imported from ``spec.triton`` (``module:function``).
    For a lowered kernel pass the JITFunction imported from the written source. A ``fn`` that is not the function ``spec.triton``
    names (the same object, or one defined under that module and name) is refused before anything is built or cached,
    even when its parameters are the same. The spec must be verified
    (:attr:`~tilefish.aot.CompileSpec.verified`; ``compile_specs`` output is).

    Triton is called as ``ASTSource(fn, signature=..., constexprs=None, attrs=...)`` (Gluon's ``GluonASTSource`` for the spec of a Gluon
    kernel, ``spec.language == "gluon"``; ``fn`` must then be a ``gluon.jit`` function, and for any other spec it must not be one), then
    ``src.constants = ...``, then ``triton.compile(src, target=GPUTarget("cuda", arch, 32), options=...)``, with the three values and the
    options exactly as ``spec.ast_source_inputs()`` gives them. ``options`` adds Triton compile options on top
    of the spec's (an experiment, such as ``{"enable_fp_fusion": False}``); they are recorded in the result, and
    each one is checked against the metadata Triton returns, because Triton silently ignores an option it does
    not know. The result is checked against the spec: the options in effect, the symbol, and the parameters of
    the PTX entry (the ABI). Raises :class:`~tilefish.errors.TilefishBuildError`.

    **Warp specialisation.** ``num_warps`` is the warps of the kernel's default partition, which is all of a kernel that is not
    warp-specialised: Triton must report exactly that number for it. A kernel whose TTGIR has ``ttg.warp_specialize`` regions (Gluon's
    ``gl.warp_specialize``, or Triton's own warp specialisation of a ``tl.range(..., warp_specialize=True)`` loop) has partitions that add their
    warps, and Triton reports the total (a 4-warp default partition and two 1-warp workers: 8): it must be more than ``num_warps`` and at most 32,
    and :attr:`BuiltKernel.num_warps` is that total, the block of a launch (``warp_specialized`` is set). Either way the threads the PTX entry
    requires (``.reqntid``) must be ``32 *`` the reported warps.

    **IEEE f32 dots.** A kernel whose IR asks for an f32 dot with IEEE input precision (``tl.dot(..., input_precision="ieee")``) is refused when
    its binary has TF32 tensor-core MMAs made of such a dot (:func:`check_tf32`, :mod:`tilefish._tf32guard`): Triton 3.5.1 lowers some of them to
    ``tcgen05.mma ... kind::tf32`` on sm_100 and sm_103. It is judged from the PTX, and from the TTGIR traced back to the TTIR by source location
    where the kernel also has dots that allow TF32; a cache hit is checked again from its PTX.

    **The identity of the binary is the IR that Triton compiles** (:mod:`tilefish._ir`). Before anything is looked up or compiled,
    Triton's frontend is run on the spec's source (the first step of ``triton.compile``) and the TTIR it builds (the TTGIR of a Gluon
    kernel), without locations or names, is hashed: that is :attr:`BuiltKernel.ir_digest`, which must equal ``ir_digest`` when that is
    given. Whatever the
    process holds that the frontend reads is in it, whichever route the kernel reads it by, so nothing is guessed from source
    files. The compile then uses a source that records the IR it was given, and a build whose IR has another digest than the one it
    was looked up under is refused, so a binary is only ever recorded under the digest of the IR it was compiled from. A spec with a
    ``source_sha256`` (a generated kernel) is also built only from a function whose module file has exactly that sha256, and whose
    imported lines are the file's.

    The compile itself never reads Triton's disk cache: its key does not cover everything that changes a binary, so a
    build would not be sure to be what the kernel says. tilefish has a cache of its own instead (:mod:`tilefish._cache`:
    ``$TILEFISH_CACHE_DIR`` or ``~/.cache/tilefish``), keyed by the cache recipe and the tilefish version, the IR digest, the
    architecture, the Triton and ``ptxas`` versions, the Triton knobs that change code generation and the compile options, and
    verified on every read (the ``sha256`` and ``sha256_code`` of the cubin
    are recomputed, among other things): an entry that is not as it was written is a miss with a
    :class:`~tilefish.errors.TilefishCacheWarning`, and the kernel is compiled. A hit has the cubin of the build that made
    it, so its ``sha256`` (whole file, with the debug path of that build) can differ from a fresh compile's while its
    ``sha256_code`` is the same. ``cache=False`` neither reads nor writes it. Whatever would make Triton compile something other than the IR is refused:
    its kernel override (``TRITON_KERNEL_OVERRIDE``) and the compile option ``ir_override`` replace the IR of a stage with a
    file, and an active instrumentation plugin patches the passes. A compile option that names a file Triton reads
    (``extern_libs``), ``LLVM_PASS_PLUGIN_PATH`` and the libdevice file are in the cache key by the sha256 of the content of
    the file, not by its path, and Triton reads them later, during the compile: they are hashed again afterwards, and a build
    in which one of them changed is refused and stores nothing (a file rewritten and put back within the compile is not seen).

    **Concurrency.** Triton's frontend and compile are not known to be thread-safe, and a compile runs with a knob set that is
    process-wide, so a build holds one lock (:data:`tilefish._ir.LOCK`) for the *whole* compile, not only for the frontend. That is
    a latency trade-off: while one thread compiles (seconds, for a large kernel), every other thread's build, ``jit`` call and
    replay that needs the frontend waits for it. A cache hit does not compile, but it still runs the frontend (milliseconds) and
    waits for a compile in progress.
    """
    cap = parse_arch(arch)
    inp = spec.ast_source_inputs()  # raises TilefishSpecError for a spec that is not verified
    check_implementation_arch(spec, cap)
    triton = _import_triton()
    jit = fn if fn is not None else resolve_triton(spec.triton)
    _check_jit_function(jit, spec)
    _check_identity(jit, spec)  # before anything is built, so that no digest, key or binary is of a function the spec is not for
    _check_generated(jit, spec)
    _refuse_unidentified(triton, spec)
    opts: dict[str, Any] = {**inp.options, **dict(options or {})}
    cuda_math = _generated_cuda_math(jit, spec, opts)
    from . import _cache

    files = _cache.compile_files(opts)  # refuses ir_override; the sha256 of the files the options name (extern_libs)
    what = f"'{spec.stem}' ({spec.triton})"
    frontend_ir = _ir.ttir_text(jit, inp, arch=cap, options=opts, what=what)
    digest = _ir.ir_digest(frontend_ir, spec.language)
    if ir_digest is not None and digest != ir_digest:
        raise TilefishBuildError(
            f"the IR of {what} has digest {digest}, but {ir_digest} was expected: it is not the kernel the caller meant to build"
        )

    store = None
    cache_id = ""
    triton_version, ptxas = str(triton.__version__), ptxas_version()
    environment = _cache.codegen_environment()
    keyed = _cache.keyed_contents(files, environment)  # the content of every file the binary is identified by: hashed again after the compile
    if cache:
        store = _cache.default_cache()
    if store is not None:
        cache_id = _cache.cache_key(
            arch=cap,
            ir_digest=digest,
            triton_version=triton_version,
            ptxas_version=ptxas,
            options=opts,
            environment=environment,
            files=files,
            language=spec.language,
        )
        hit = store.load(cache_id, spec, arch=f"sm_{cap}", options=opts, triton_version=triton_version, ptxas_version=ptxas, ir_digest=digest)
        if hit is not None:
            check_tf32(spec, cap, frontend_ir, hit.ptx, triton_version)  # an entry has its PTX only, which the build that stored it was judged by too
            return dataclasses.replace(hit, cuda_math=cuda_math)  # (read from the source and the libdevice here, not from the cache)

    from triton.backends.compiler import GPUTarget

    src = _ir.recording_source(_ir.source_class(spec.language))(jit, signature=inp.signature, constexprs=None, attrs=inp.attrs)
    src.constants = inp.constants
    try:
        with _always_compile(triton):
            compiled = triton.compile(src, target=GPUTarget("cuda", cap, 32), options=opts)
    except Exception as e:  # noqa: BLE001 - a compile failure of any kind is a build error, with Triton's message
        raise TilefishBuildError(f"Triton could not compile '{spec.stem}' ({spec.triton}) for sm_{cap}: {type(e).__name__}: {e}") from e
    changed = _cache.changed_contents(keyed, opts)
    if changed:
        raise TilefishBuildError(
            f"'{spec.stem}': {'; '.join(changed)} changed while Triton was compiling it, so the binary may be of the content the "
            "build was identified by or of the new one, and the cache key (which holds the content of these files) cannot say which: "
            "the binary was neither returned nor stored. Build again when nothing is rewriting the file"
        )
    if src.ir_text is None or _ir.ir_digest(src.ir_text, spec.language) != digest:
        raise TilefishBuildError(
            f"'{spec.stem}': the IR Triton compiled is not the IR whose digest {digest} the build was identified by (Triton's frontend "
            "gave another module the second time it was run on the same kernel); refusing to record or cache the binary"
        )

    md = compiled.metadata
    # a warp-specialised kernel: Triton's metadata num_warps is the total of its partitions, which add warps to the num_warps it was given (its
    # default partition's); any other kernel has exactly the num_warps it was given
    warp_specialized = is_warp_specialized(getattr(compiled, "asm", {}).get("ttgir"))
    for key, want in opts.items():
        got = getattr(md, key, _MISSING)
        if got is _MISSING:
            raise TilefishBuildError(f"'{spec.stem}': Triton does not know the compile option '{key}' (it ignored it)")
        if key == "num_warps" and warp_specialized:
            problem = warps_problem(want, got, True)
            if problem is not None:
                raise TilefishBuildError(f"'{spec.stem}': the compile option num_warps={want!r} was not applied as it should be: {problem}")
            continue
        if not _applied(key, want, got):
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
    check_threads(spec, ptx, symbol, int(md.num_warps))
    check_tensordesc_meta(spec, getattr(md, "tensordesc_meta", None))
    asm = getattr(compiled, "asm", {})
    check_tf32(spec, cap, frontend_ir, ptx, triton_version, ttir=_text_or_none(asm.get("ttir")), ttgir=_text_or_none(asm.get("ttgir")))
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
        triton_version=triton_version,
        ptxas_version=ptxas,
        ptx_target=target.group(1) if target else "",
        options=opts,
        launch=_launch_attributes(md),
        ir_digest=digest,
        warp_specialized=warp_specialized,
        cuda_math=cuda_math,
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
        "kind": ENTRY_KIND_TRITON if spec.binary is None else ENTRY_KIND_BINARY,
        "abi": abi_of(spec),
        "grid": spec.grid,
        "block": list(spec.block),
    }
    if spec.binary is None:
        entry["num_warps"] = spec.num_warps
        entry["num_stages"] = spec.num_stages
    entry["constexprs"] = constexprs
    entry["buckets"] = dict(spec.buckets)
    entry["dims"] = {name: json.loads(json.dumps(c)) for name, c in spec.dims.items()}
    entry["preconditions"] = [p.to_dict() for p in spec.preconditions]
    if spec.binary is None:
        entry["options"] = dict(spec.ast_source_inputs().options)
    else:  # a prebuilt binary declares its dynamic shared memory, and the symbol of its kernel
        entry["symbol"] = spec.binary["symbol"]
        entry["shared"] = spec.binary["shared"]
    entry["declaration_digest"] = spec.declaration_digest
    if spec.implementation is not None:  # which implementation of an op declared with by_arch= this is (its key, in the declaration digest too)
        entry["implementation"] = spec.implementation
    if spec.source_sha256 is not None:  # a generated kernel: the sha256 of the Triton source it was built from
        entry["source_sha256"] = spec.source_sha256
    return entry


def expected_entry(spec: "CompileSpec", entry: "Mapping[str, Any]") -> dict[str, Any]:
    """What a manifest ``entry`` of a binary of ``spec`` must have, as JSON reads it back (lists, strings, plain numbers): :func:`spec_entry`, with
    the block a warp-specialised kernel's build found (``32 * total_num_warps``: the warps of all its partitions; a reader checks the field
    itself, that it is more than the declared ``num_warps``) in place of the declared one."""
    expected = json.loads(json.dumps(spec_entry(spec)))
    total = entry.get("total_num_warps")
    if total is not None and spec.binary is None:
        expected["block"] = [32 * total, 1, 1]
    return expected


def _kernel_entry(b: BuiltKernel) -> dict[str, Any]:
    spec_part = spec_entry(b.spec)
    entry: dict[str, Any] = {
        "stem": spec_part["stem"],
        "op": spec_part["op"],
        "kind": spec_part["kind"],
        "arch": b.arch,
        "ptx_target": b.ptx_target,
        "symbol": b.symbol,
        "cubin": b.cubin_name,
        "abi": spec_part["abi"],
        "grid": spec_part["grid"],
        "block": list(b.block),  # the compiled kernel's: 32 * the warps of every partition of a warp-specialised kernel
        "shared": b.shared,
    }
    if b.spec.binary is None:
        problem = warps_problem(b.spec.num_warps, b.num_warps, b.warp_specialized)
        if problem is not None:
            raise TilefishBuildError(f"'{b.stem}' ({b.arch}): {problem}")
        entry["num_warps"] = spec_part["num_warps"]  # the compile option: the declared warps (a warp-specialised kernel's default partition)
        entry["num_stages"] = spec_part["num_stages"]
        if b.warp_specialized:  # its partitions' warps in all, which the block is made of
            entry["total_num_warps"] = b.num_warps
    entry["launch"] = b.launch.to_dict()
    entry["constexprs"] = spec_part["constexprs"]
    entry["buckets"] = spec_part["buckets"]
    entry["dims"] = spec_part["dims"]
    entry["preconditions"] = spec_part["preconditions"]
    if b.spec.binary is None:
        entry["options"] = dict(b.options)
    entry["declaration_digest"] = spec_part["declaration_digest"]
    if "implementation" in spec_part:
        entry["implementation"] = spec_part["implementation"]
    entry["sha256"] = b.sha256
    entry["sha256_code"] = b.sha256_code
    if b.spec.binary is None:
        entry["toolchain"] = {"triton": b.triton_version, "ptxas": b.ptxas_version}
    else:  # a prebuilt PTX is assembled by ptxas (recorded, with its flags); a cubin is used as it is
        entry["toolchain"] = {"ptxas": b.ptxas_version} if b.ptxas_version else {}
        assert b.binary_build is not None
        entry["binary"] = json.loads(json.dumps(b.binary_build))
    if "source_sha256" in spec_part:
        entry["source_sha256"] = spec_part["source_sha256"]
    if b.cuda_math is not None:  # the CUDA math functions whose bits the binary has, and the libdevice it links (section 4.3, tilefish._cudamath)
        entry["cuda_math"] = json.loads(json.dumps(b.cuda_math))
    if b.ir_digest is not None:  # the digest of the Triton IR the binary was compiled from (the field and its kind, so a reader of schema 2 reads it as before)
        entry["source_digest"] = b.ir_digest
        entry["source_digest_kind"] = _ir.digest_kind(b.spec.language)  # the TTIR of a triton.jit kernel, the TTGIR of a Gluon one
    elif b.binary_digest is not None:  # the digest of the PTX or cubin bytes a prebuilt binary is made of
        entry["source_digest"] = b.binary_digest
        entry["source_digest_kind"] = BINARY_DIGEST_KIND
    return entry


def manifest_dict(built: "Iterable[BuiltKernel]", library: "Iterable[Any]" = ()) -> dict[str, Any]:
    """The manifest as a JSON-compatible dict: one entry per kernel, in the order given, and the ``library`` entries (pinned library
    operations such as a cuBLASLt GEMM, each a :class:`~tilefish._library.BuiltLibrary`)."""
    from . import __version__

    return {
        "schema": MANIFEST_SCHEMA,
        "tilefish": __version__,
        "kernels": [_kernel_entry(b) for b in built],
        "library": [lib.entry() for lib in library],
    }


def _conflicts(old: "dict[str, Any]", new: "dict[str, Any]") -> "str | None":
    """What makes an existing manifest entry a different binary from a new one for the same stem and arch (``None``: the same)."""
    for key in (
        "sha256_code",
        "declaration_digest",
        "implementation",
        "source_sha256",
        "source_digest",
        "source_digest_kind",
        "options",
        "shared",
        "total_num_warps",
        "algo",
        "cublaslt_version",
        "sm_count",
        "workspace_bytes",
        "pin",
        "cudnn_version",
    ):
        if old.get(key) != new.get(key):
            return key
    # a cuDNN convolution's canary: two of them that differ are two builds of cuDNN (an entry without one, an older export, is replaced by one with it)
    if "canary" in old and "canary" in new and old["canary"] != new["canary"]:
        return "canary"
    return None


def write_manifest(
    built: "Iterable[BuiltKernel]",
    path: "str | pathlib.Path",
    *,
    cubins: bool = False,
    merge: bool = False,
    overwrite: bool = False,
    library: "Iterable[Any]" = (),
) -> pathlib.Path:
    """Write the manifest of built kernels as strict JSON to ``path`` and return it.

    ``library`` is the built library operations (a :class:`~tilefish._library.BuiltLibrary` each: a pinned cuBLASLt GEMM): their entries
    go in the manifest's ``library`` section, which has no files.

    For each kernel: ``stem``, ``symbol``, ``abi`` (``params`` in launch order, then the two ``scratch`` slots; a pointer
    records its declared ``shape``, a list of dimension expressions and sizes such as ``["Rows", 1024]``), ``dims`` (for each
    named dimension its ``divisible_by`` and ``buckets`` and, for a bucketed one, the ``bucket`` this binary is built for: a
    host binds the dimensions from the shapes of its tensors and checks them with these alone), ``grid`` (an expression over the
    dimension names, with ``cdiv(a, b)``), ``block`` (``32 *`` the warps of a block of the compiled kernel: the declared ``num_warps``, or for a
    warp-specialised kernel the total of its partitions, which ``total_num_warps`` records), ``shared``, ``launch``
    (scratch sizes, PDL, cluster and cooperative flags: all zero or false for a binary a plain
    ``cuLaunchKernel`` can run), ``constexprs``, ``preconditions`` (what a host checks before it launches),
    ``declaration_digest`` and, for a generated kernel, ``source_sha256`` (the Triton source it was built from), for every kernel
    ``source_digest`` with ``source_digest_kind`` ``"tilefish.triton-ir.v1"`` (the digest of the Triton IR it was compiled from, see
    :mod:`tilefish._ir`; the field has this name for schema 2 readers, and holds an IR digest), ``sha256`` (the file) and
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
    return write_manifest_doc(built, path, cubins=cubins, merge=merge, overwrite=overwrite, library=library)[0]


def write_manifest_doc(
    built: "Iterable[BuiltKernel]",
    path: "str | pathlib.Path",
    *,
    cubins: bool = False,
    merge: bool = False,
    overwrite: bool = False,
    library: "Iterable[Any]" = (),
) -> "tuple[pathlib.Path, dict[str, Any]]":
    """:func:`write_manifest`, returning the path and the manifest document that was written (what ``export`` checks the directory against)."""
    entries = list(built)
    doc = manifest_dict(entries, library)
    keys = [(e["stem"], e["arch"]) for e in doc["kernels"]]
    if len(set(keys)) != len(keys):
        raise TilefishBuildError("a manifest needs one build per (stem, arch); some appear twice")
    lib_keys = [(e["stem"], e["arch"]) for e in doc["library"]]
    if len(set(lib_keys)) != len(lib_keys) or set(lib_keys) & set(keys):
        raise TilefishBuildError("a manifest needs one build per (stem, arch), kernels and library entries together; some appear twice")
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
        new_lib = dict(zip(lib_keys, doc["library"]))
        kept_lib: list[dict[str, Any]] = []
        for old in existing.doc["library"]:
            key = (old["stem"], old["arch"])
            new = new_lib.pop(key, None)
            if new is None:
                kept_lib.append(old)
                continue
            why = _conflicts(old, new)
            if why is not None and not overwrite:
                raise TilefishManifestError(
                    f"the manifest {str(path)!r} already has a different library entry for '{key[0]}' on {key[1]} ({why} differs); "
                    "pass overwrite=True to replace it"
                )
            kept_lib.append(new)
        doc["library"] = kept_lib + list(new_lib.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    for b, target in zip(entries, targets):
        write_file(target, b.cubin)
    write_file(path, (json.dumps(doc, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return path, doc
