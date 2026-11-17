"""CuTe DSL kernels as externs: ``@tilefish.extern(cute=host, cute_args=..., cute_options=...)`` (``docs/en/cute.md``).

CUTLASS's CuTe DSL (the ``nvidia-cutlass-dsl`` package, ``import cutlass.cute as cute``) compiles a ``@cute.jit`` host function and the
``@cute.kernel`` it launches: Python to MLIR, MLIR to PTX with the NVVM it carries, PTX to a cubin with the ``ptxas`` it carries. The host function
is compiled too (to LLVM, JIT-executed): it computes the grid, the block, the cluster and the dynamic shared memory of the launch, and encodes the TMA
descriptors of the kernel's tensors itself, on the host. tilefish takes the result like any other extern:

* **The binary** is the DSL's own cubin (``cute.compile(..., options="--keep-cubin --keep-ptx --dump-dir=...")``), for the architecture of the device it is built
  on; its entry is a ``binary`` entry (manifest schema 3, section 4.9) of ``binary.source`` ``"cute"``, pinned like any other (``sha256``,
  ``sha256_code``, the manifest digest), its PTX the ``source_digest`` (kind :data:`DIGEST_KIND`).
* **The launch** is taken from the compiled host function, without running the kernel: tilefish calls it on a stream that is being captured into a
  CUDA graph (nothing executes), three times with tensors at other addresses, and reads the kernel node of each graph: its function, grid, block,
  dynamic shared memory, cluster, scheduling policy and the bytes of every kernel parameter. Every byte of a parameter the kernel reads (its PTX's
  ``ld.param`` loads, :func:`ptx_param_reads`; all of one whose address it takes) must be the same in the three captures, or be the address of one of
  the operator's tensors (:func:`classify_params`); a byte it does not read (padding), and the bytes 64 to 127 of a TMA descriptor, which the DSL's host
  code does not write, are passed as zeros. So every parameter is one of:

  - ``pointer``: the address of a tensor (8 bytes);
  - ``tensormap``: a TMA descriptor of a tensor as the DSL encodes it for a static shape (128 bytes, 64-byte aligned): a template with the
    tensor's address in its first 8 bytes;
  - ``bytes``: constant bytes (a static layout, an empty struct, a value compiled for the bucket).

  A host packs them from the manifest alone (Python :mod:`tilefish._cuda`, the Rust crate, the C ABI), in a single launch, a plan and a CUDA graph.

* **Shapes are static.** A CuTe DSL kernel is compiled for the layouts of its arguments; tilefish compiles one per bucket, so every dimension of a
  CuTe extern's tensors is a number or a bucketed dimension (``Dim("M", buckets=[...])``). The tensors are dense (row-major, at offset 0) and
  16-byte aligned (``assumed_align=16``); ``cute_args`` makes the arguments of the host function from them (views of them: a permutation, an added
  axis), and tilefish appends the stream.

Nothing here imports the DSL, torch or the driver until a build or a launch needs them: a declaration only reads the DSL's version
(``importlib.metadata``) and the file that defines the host function.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import re
import struct
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from .errors import TilefishBuildError, TilefishDeclarationError

if TYPE_CHECKING:
    from ._build import BuiltKernel
    from .aot import CompileSpec

DIGEST_KIND = "tilefish.cute.v1"  # the ``source_digest_kind`` of a CuTe DSL binary: its ``source_digest`` is the sha256 of the PTX the DSL generated
DISTRIBUTION = "nvidia-cutlass-dsl"  # the package whose version is in the declaration digest and every entry
DESCRIPTOR_BYTES = 128  # a CUtensorMap
DESCRIPTOR_WRITTEN = 64  # the bytes of it the DSL's host code writes (8 words); a host passes zeros after them
ADDRESS_LIMIT = 1 << 57  # the DSL encodes ``(address / 16) & (2**53 - 1)`` at bit 4 of a descriptor's first word: 57 bits of address
MAX_PARAM_BYTES = 32764  # the parameter space of a kernel (sm_70 and up)
CAPTURES = 3  # the launches the build captures: the parameters of each are compared byte by byte
_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}(?:[A-Za-z0-9.+_-]*)?\Z")
_OPTIONS = re.compile(r"[A-Za-z0-9 =,._:/+-]*\Z")
# the compile options tilefish passes itself, which a declaration may not (the artifacts and where they go, the architecture, line information)
_RESERVED_OPTIONS = ("--keep-cubin", "--keep-ptx", "--keep-sass", "--gpu-arch", "--dump-dir", "--enable-tvm-ffi", "--generate-line-info")


def dsl_version() -> "str | None":
    """The version of the CuTe DSL package installed in this environment (``None``: not installed). A test may replace it."""
    try:
        return importlib.metadata.version(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True, eq=False)
class CuteTarget:
    """A CuTe DSL kernel as a declaration holds it (``@tilefish.extern(cute=...)``, ``tilefish.impl(cute=...)``).

    ``host`` is the ``@cute.jit`` function (or an object whose ``__call__`` is one, such as an instance of a kernel class of the CUTLASS examples), and
    ``ref`` names it (``module:qualname``); ``source_sha256`` is the sha256 of the file that defines it, read once, when the operator is declared;
    ``dsl`` the version of the CuTe DSL package then (both are in the declaration digest, so another file or another DSL is another declaration).
    ``options`` are compile options of the DSL (``"--opt-level 3"``) besides the ones tilefish passes. ``args`` makes the positional arguments of
    ``host`` from the operator's tensors (torch tensors, in the order of the stub); ``None`` passes the tensors themselves. tilefish appends the stream.

    It stands where a prebuilt binary's :class:`~tilefish.decorators.BinaryTarget` does (``ExternOp.binary``): ``kind`` is ``"cute"``, and what a
    prebuilt binary declares (its entry point, block, shared memory, cluster) is the compiled kernel's, found by the build.
    """

    host: Any
    ref: str
    source_sha256: str
    dsl: str
    options: str = ""
    args: "Callable[..., Any] | None" = None
    kind: str = field(default="cute", init=False)
    # what a prebuilt binary declares, and a CuTe kernel does not (the build finds it): never a launch's
    symbol: str = field(default="", init=False)
    block: "tuple[int, int, int]" = field(default=(1, 1, 1), init=False)
    shared: int = field(default=0, init=False)
    cluster: "tuple[int, int, int]" = field(default=(1, 1, 1), init=False)

    @property
    def sources(self) -> "dict[str, bytes]":
        return {}

    @property
    def digests(self) -> "dict[str, str]":
        return {"source": self.source_sha256}

    def spec_section(self) -> "dict[str, Any]":
        """The ``binary`` section of a compile spec of this kernel (and of its canonical declaration): what it is made from."""
        return {"kind": "cute", "host": self.ref, "source_sha256": self.source_sha256, "cutlass_dsl": self.dsl, "options": self.options}


def _host_function(host: Any) -> Any:
    """The Python function behind ``host``: a function (unwrapped), or the ``__call__`` of an object."""
    fn = host
    if not inspect.isfunction(fn) and not inspect.ismethod(fn):
        call = getattr(type(host), "__call__", None)
        if call is not None and not isinstance(host, type):
            fn = call
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return getattr(fn, "__func__", fn)


def host_ref(host: Any) -> "tuple[str, str]":
    """``(ref, file)`` of a host function: ``module:qualname`` (an object: its class's qualname, ``()``) and the file of its module."""
    if host is None or isinstance(host, (str, bytes, type)) or not callable(host):
        raise TilefishDeclarationError(
            f"extern: cute= is a @cute.jit host function, or an object whose __call__ is one (an instance of a kernel class), got {host!r}"
        )
    is_object = not (inspect.isfunction(host) or inspect.ismethod(host) or hasattr(host, "__wrapped__") or hasattr(host, "__qualname__"))
    fn = _host_function(host)
    module = getattr(fn, "__module__", None) or getattr(type(host), "__module__", None)
    qualname = getattr(fn, "__qualname__", None) or type(host).__qualname__
    if is_object:
        qualname = type(host).__qualname__ + "()"
    mod = sys.modules.get(module or "")
    file = getattr(mod, "__file__", None)
    if not isinstance(file, str):
        try:
            file = inspect.getsourcefile(fn)
        except TypeError:
            file = None
    if not module or not isinstance(file, str):
        raise TilefishDeclarationError(
            f"extern: the CuTe DSL host function {host!r} has no module file (it is in the declaration digest: define it in a module)"
        )
    return f"{module}:{qualname}", file


def check_options(options: Any, where: str) -> str:
    """The compile options of a declaration: a string of the DSL's command-line options, none of the ones tilefish passes itself."""
    if options is None:
        return ""
    if not isinstance(options, str) or not _OPTIONS.fullmatch(options) or len(options) > 1024:
        raise TilefishDeclarationError(
            f"{where}: cute_options= is a string of CuTe DSL compile options (letters, digits, spaces and = , . _ : / + -), got {options!r}"
        )
    for word in options.split():
        if word.split("=", 1)[0] in _RESERVED_OPTIONS:
            raise TilefishDeclarationError(
                f"{where}: cute_options= may not give {word.split('=', 1)[0]}: tilefish passes the options that make the cubin and the PTX "
                "itself, for the architecture of the device, without line information (paths of the host in the binary)"
            )
    return " ".join(options.split())


def cute_target(host: Any, args: Any, options: Any, where: str) -> CuteTarget:
    """The :class:`CuteTarget` of a declaration, checked: the host function, its file (read now), the DSL installed, the options and ``args``."""
    ref, file = host_ref(host)
    if args is not None and (not callable(args) or isinstance(args, type)):
        raise TilefishDeclarationError(f"{where}: cute_args= makes the host function's arguments from the tensors (a callable), got {args!r}")
    version = dsl_version()
    if version is None:
        raise TilefishDeclarationError(
            f"{where}: a CuTe DSL kernel needs the CuTe DSL ({DISTRIBUTION}) installed: its version is in the declaration digest. pip install "
            f"{DISTRIBUTION} (Linux, a CUDA 12.9 or 13 driver)"
        )
    if not _VERSION.fullmatch(version):
        raise TilefishDeclarationError(f"{where}: the version of {DISTRIBUTION} is {version!r}, which is not a version string")
    try:
        with open(file, "rb") as f:
            data = f.read()
    except OSError as e:
        raise TilefishDeclarationError(f"{where}: cannot read the file of the CuTe DSL host function {ref} ({file!r}): {e}") from None
    return CuteTarget(host, ref, hashlib.sha256(data).hexdigest(), version, check_options(options, where), args)


# -- the parameters of a launch ----------------------------------------------------------------------------------------------------------------


def ptx_entry_params(ptx: str, symbol: str) -> "list[tuple[int, int]]":
    """``(size, alignment)`` of every parameter of the PTX entry ``symbol``, in order: ``.param .align 64 .b8 name[128]`` is ``(128, 64)``,
    ``.param .u64 name`` ``(8, 8)``."""
    m = re.search(r"\.entry\s+" + re.escape(symbol) + r"\s*\(", ptx)
    if m is None:
        raise TilefishBuildError(f"the PTX has no entry point '{symbol}'")
    close = ptx.find(")", m.end())
    if close < 0:
        raise TilefishBuildError(f"the PTX entry '{symbol}' has no parameter list end")
    out: "list[tuple[int, int]]" = []
    for decl in [d.strip() for d in ptx[m.end() : close].split(",") if d.strip()]:
        pm = re.match(r"\.param\s+(?:\.align\s+(\d+)\s+)?\.([a-z]+)(\d+)\s+\S+?(?:\[(\d+)\])?\Z", decl)
        if pm is None:
            raise TilefishBuildError(f"the PTX entry '{symbol}' has a parameter tilefish does not read: {decl!r}")
        align, bits, count = pm.group(1), int(pm.group(3)), pm.group(4)
        width = bits // 8
        size = width * int(count) if count is not None else width
        out.append((size, int(align) if align is not None else width))
    return out


def ptx_reqntid(ptx: str, symbol: str) -> "tuple[int, int, int] | None":
    """The block a PTX entry requires (``.reqntid x, y, z``), or ``None``."""
    m = re.search(r"\.entry\s+" + re.escape(symbol) + r"\s*\(", ptx)
    if m is None:
        return None
    body = ptx.find("{", m.end())
    head = ptx[m.end() : body if body >= 0 else len(ptx)]
    r = re.search(r"\.reqntid\s+(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?", head)
    if r is None:
        return None
    return (int(r.group(1)), int(r.group(2) or 1), int(r.group(3) or 1))


def ptx_cluster(ptx: str, symbol: str) -> "tuple[int, int, int] | None":
    """The cluster a PTX entry requires (``.reqnctapercluster x, y, z``), or ``None``."""
    m = re.search(r"\.entry\s+" + re.escape(symbol) + r"\s*\(", ptx)
    if m is None:
        return None
    body = ptx.find("{", m.end())
    head = ptx[m.end() : body if body >= 0 else len(ptx)]
    r = re.search(r"\.reqnctapercluster\s+(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?", head)
    if r is None:
        return None
    return (int(r.group(1)), int(r.group(2) or 1), int(r.group(3) or 1))


def ptx_tools(ptx: str) -> str:
    """The NVVM release the PTX header names (``Cuda compilation tools, release 13.4, V13.4.59``), or ``""``."""
    m = re.search(r"^//\s*(Cuda compilation tools, release [^\n]+?)\s*$", ptx, re.M)
    return m.group(1) if m else ""


def cubin_ptxas(cubin: bytes) -> str:
    """The ptxas release the cubin records in ``.note.nv.tkinfo`` (``Cuda compilation tools, release 12.9, V12.9.83``), or ``""``."""
    from ._elf import parse_elf

    elf = parse_elf(cubin)
    note = elf.section(".note.nv.tkinfo")
    if note is None:
        return ""
    m = re.search(rb"Cuda compilation tools, release [^\x00]+", elf.data(note))
    return m.group().decode("ascii", "replace") if m else ""


@dataclass(frozen=True)
class Capture:
    """One captured launch: the kernel node of a CUDA graph (``name`` of its function, ``grid`` of blocks, ``block``, dynamic ``shared`` memory,
    ``cluster``, the cluster scheduling ``policy``, and whether it asks for a programmatic or cooperative launch) and the bytes of every parameter."""

    name: str
    grid: "tuple[int, int, int]"
    block: "tuple[int, int, int]"
    shared: int
    cluster: "tuple[int, int, int]"
    policy: int
    params: "tuple[bytes, ...]"
    pdl: bool = False
    cooperative: bool = False


_LD_PARAM = re.compile(r"\bld\.param(?:\.v(2|4))?\.[a-z]+?(8|16|32|64)\b[^\[\n]*\[\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:\+\s*([0-9]+))?\s*\]")


def ptx_param_reads(ptx: str, symbol: str) -> "list[frozenset[int] | None]":
    """The bytes of each parameter of the PTX entry ``symbol`` its code reads: the union of its ``ld.param`` loads at constant offsets
    (``ld.param.v2.b8 {%rs2, %rs3}, [k_param_4+8]`` reads the bytes 8 and 9), or ``None`` for a parameter whose address the code takes in any other way
    (a TMA descriptor, which ``cp.async.bulk.tensor`` reads by its address): all of it then. A byte no load reads is one the kernel does not depend on,
    such as the padding of a structure, which the DSL's host code leaves as it was on its stack."""
    count = len(ptx_entry_params(ptx, symbol))
    names = [f"{symbol}_param_{j}" for j in range(count)]
    m = re.search(r"\.entry\s+" + re.escape(symbol) + r"\s*\(", ptx)
    assert m is not None  # (ptx_entry_params found it)
    body = ptx[ptx.find(")", m.end()) + 1 :]
    reads: "list[set[int] | None]" = [set() for _ in names]
    loads = {}
    for ld in _LD_PARAM.finditer(body):
        name = ld.group(3)
        if name not in names:
            continue
        j = names.index(name)
        width = int(ld.group(2)) // 8 * int(ld.group(1) or 1)
        offset = int(ld.group(4) or 0)
        loads[j] = loads.get(j, 0) + 1
        reads[j].update(range(offset, offset + width))  # type: ignore[union-attr]
    for j, name in enumerate(names):
        mentions = len(re.findall(r"(?<![A-Za-z0-9_$])" + re.escape(name) + r"(?![A-Za-z0-9_$])", body))
        if mentions != loads.get(j, 0):  # its address taken (a descriptor, a byte array the code indexes): every byte may be read
            reads[j] = None
    return [frozenset(r) if r is not None else None for r in reads]


def classify_params(
    stem: str,
    tensors: "Sequence[Any]",
    layout: "Sequence[tuple[int, int]]",
    captures: "Sequence[Sequence[bytes]]",
    addresses: "Sequence[Mapping[str, int]]",
    reads: "Sequence[frozenset[int] | None] | None" = None,
) -> "list[dict[str, Any]]":
    """The ABI of a CuTe DSL kernel from its captured launches.

    ``tensors`` are the operator's tensor arguments (each with ``index``, ``name``, ``dtype``, ``access`` and ``shape_dims``: the compile spec's
    pointer :class:`~tilefish.aot.ArgSpec`), ``layout`` the ``(size, alignment)`` of each kernel parameter (its PTX entry), ``captures[t][j]`` the bytes
    of parameter ``j`` in capture ``t`` and ``addresses[t][name]`` the address of each tensor in capture ``t`` (every tensor at another address in
    every capture). A parameter is

    * ``pointer`` of a tensor: 8 bytes that are its address in every capture;
    * ``tensormap`` of a tensor: 128 bytes, 64-byte aligned, whose first 8 are its address in every capture (a TMA descriptor as the DSL's host code
      encodes it for a static shape): the template is the bytes with the address and the bytes from 64 on (which that code does not write, and a host
      passes as zeros) cleared;
    * ``bytes``: the same bytes in every capture, and no address.

    ``reads[j]`` are the bytes of parameter ``j`` the kernel's code reads (:func:`ptx_param_reads`; ``None``: all of them, and ``reads`` ``None``: all of
    every parameter): a byte it does not read (the padding of a structure) is not compared, and is zero in what a host passes. Anything else is refused:
    a byte the kernel reads that differs between captures and is no tensor's address, an address inside a structure, a tensor in two parameters or in
    none. Raises :class:`~tilefish.errors.TilefishBuildError`.
    """
    if len(captures) < 2 or len(addresses) != len(captures):
        raise TilefishBuildError(f"'{stem}': the parameters of a CuTe DSL kernel are told from at least two captured launches")
    count = len(layout)
    for t, cap in enumerate(captures):
        if len(cap) != count or any(len(b) != layout[j][0] for j, b in enumerate(cap)):
            raise TilefishBuildError(
                f"'{stem}': capture {t} has parameters of {[len(b) for b in cap]} bytes, but the kernel's PTX entry declares {[s for s, _ in layout]}"
            )
    names = [a.name for a in tensors]
    for t, addr in enumerate(addresses):
        values = [addr[n] for n in names]
        if len(set(values)) != len(values):
            raise TilefishBuildError(f"'{stem}': two tensors share an address in capture {t}: their parameters cannot be told apart")
    for n in names:
        if len({addr[n] for addr in addresses}) != len(addresses):
            raise TilefishBuildError(f"'{stem}': the tensor '{n}' is at the same address in two captures: its parameters cannot be told apart")
    by_name = {a.name: a for a in tensors}
    params: "list[dict[str, Any]]" = []
    used: "dict[str, int]" = {}
    total = 0
    for j, (size, align) in enumerate(layout):
        total = -(-total // max(align, 1)) * max(align, 1) + size
        if size < 1 or size > MAX_PARAM_BYTES:
            raise TilefishBuildError(f"'{stem}': parameter {j} has {size} bytes")
        found: "list[tuple[int, str]]" = []
        for off in range(0, size - 7, 8):
            for n in names:
                if all(struct.unpack_from("<Q", cap[j], off)[0] == addr[n] for cap, addr in zip(captures, addresses)):
                    found.append((off, n))
        read = reads[j] if reads is not None else None
        if read is not None:  # an address is an 8-byte word the code reads whole
            found = [(o, n) for o, n in found if all(o + k in read for k in range(8))]
        descriptor = size == DESCRIPTOR_BYTES and align == 64 and [o for o, _ in found] == [0]
        ignored = set(range(DESCRIPTOR_WRITTEN, DESCRIPTOR_BYTES)) if descriptor else set()
        if read is not None:
            ignored |= set(range(size)) - set(read)
        covered = {o + k for o, _ in found for k in range(8)}
        unstable = [k for k in range(size) if k not in covered and k not in ignored and len({cap[j][k] for cap in captures}) > 1]
        if unstable:
            raise TilefishBuildError(
                f"'{stem}': byte(s) {unstable[:8]}{' ...' if len(unstable) > 8 else ''} of kernel parameter {j} ({size} bytes) differ between the captured "
                "launches and are no tensor's address: the kernel takes a value tilefish cannot reproduce at a launch (a pointer the host function "
                "allocated, a value of the stream or of the time). Only the operator's tensors, at offset 0, and constants may be its parameters"
            )
        if not found:
            value = bytes(0 if k in ignored else b for k, b in enumerate(captures[0][j]))  # (a byte the kernel does not read: zero)
            params.append({"slot": j, "index": j, "name": f"param_{j}", "kind": "bytes", "type": "bytes", "ptx_type": f"b8[{size}]", "size": size,
                           "value": value.hex()})  # fmt: skip
            continue
        if len(found) != 1 or found[0][0] != 0 or not (size == 8 or descriptor):
            raise TilefishBuildError(
                f"'{stem}': kernel parameter {j} ({size} bytes, aligned to {align}) holds the address of {[n for _, n in found]} at byte(s) "
                f"{[o for o, _ in found]}: tilefish takes a tensor as an 8-byte pointer or as a TMA descriptor (128 bytes, its address first), not inside a "
                "structure (declare the tensor's layout static: a CuTe tensor of a static layout is passed as its pointer)"
            )
        n = found[0][1]
        if n in used:
            raise TilefishBuildError(
                f"'{stem}': the tensor '{n}' is in kernel parameters {used[n]} and {j}: tilefish takes each tensor of a CuTe DSL kernel in one parameter"
            )
        used[n] = j
        a = by_name[n]
        entry: "dict[str, Any]" = {"slot": j, "index": a.index, "name": n}
        if descriptor:
            template = bytearray(captures[0][j])
            template[0:8] = bytes(8)
            template[DESCRIPTOR_WRITTEN:] = bytes(DESCRIPTOR_BYTES - DESCRIPTOR_WRITTEN)
            entry.update(kind="tensormap", type="tensormap", ptx_type="b8[128]", size=128)
        else:
            from .dtypes import ALL_DTYPES

            entry.update(kind="pointer", type=ALL_DTYPES[a.dtype].triton_ptr, ptx_type="u64", size=8)
        entry.update(dtype=a.dtype, access=a.access, shape=list(a.shape_dims or ()))
        if descriptor:
            entry["template"] = bytes(template).hex()
        params.append(entry)
    if total > MAX_PARAM_BYTES:
        raise TilefishBuildError(f"'{stem}': the kernel's parameters take {total} bytes, more than the {MAX_PARAM_BYTES} a kernel may have")
    missing = [n for n in names if n not in used]
    if missing:
        raise TilefishBuildError(
            f"'{stem}': no kernel parameter holds the address of {missing}: every tensor of a CuTe DSL extern is a parameter of its kernel, as a "
            "pointer or a TMA descriptor (one the host function reads only, or passes inside a structure, is not)"
        )
    order = [p["name"] for p in params if p["kind"] != "bytes"]
    if order != names:
        raise TilefishBuildError(
            f"'{stem}': the kernel takes the tensors in the order {order}, and the stub declares {names}: declare them in the kernel's order (a host, "
            "a plan and every implementation of an operator take its tensors in the order of its stub)"
        )
    return params


def descriptor(template_hex: str, address: int) -> bytes:
    """The 128 bytes a launch passes for a ``tensormap`` parameter: its template with ``address`` (16-byte aligned, below 2**57) in the first 8."""
    if isinstance(address, bool) or not isinstance(address, int) or address <= 0 or address % 16 or address >= ADDRESS_LIMIT:
        raise ValueError(f"a TMA descriptor of a CuTe DSL kernel holds a 16-byte aligned address below 2**57, got {address!r}")
    raw = bytearray.fromhex(template_hex)
    raw[0:8] = struct.pack("<Q", address)
    return bytes(raw)


def launch_of(captures: "Sequence[Capture]", symbol: str, stem: str) -> "tuple[tuple[int, int, int], tuple[int, int, int], int, tuple[int, int, int]]":
    """``(grid of programs, block, shared, cluster)`` of a CuTe DSL kernel from its captured launches, after checking they agree and are what a host
    launches: the kernel ``symbol``, a cluster of at most 8 CTAs (the spread scheduling policy every host asks for, or the default), no programmatic
    or cooperative launch, a grid of blocks that is a whole number of clusters. The grid of an entry counts clusters (manifest schema 3, section 4.5)."""
    from ._build import MAX_CLUSTER

    first = captures[0]
    for t, c in enumerate(captures[1:], 1):
        if (c.name, c.grid, c.block, c.shared, c.cluster, c.policy) != (first.name, first.grid, first.block, first.shared, first.cluster, first.policy):
            raise TilefishBuildError(
                f"'{stem}': the captured launches differ: {first} and {c} (capture {t}): the host function's launch depends on more than the shapes"
            )
    if first.name != symbol:
        raise TilefishBuildError(f"'{stem}': the captured launch is of the function '{first.name}', but the compiled kernel is '{symbol}'")
    if first.pdl or first.cooperative:
        raise TilefishBuildError(
            f"'{stem}': the host function launches with {'programmatic dependent launch' if first.pdl else 'a cooperative launch'}, which no tilefish host sets up"
        )
    cluster = first.cluster
    if min(cluster) < 1:
        cluster = (1, 1, 1)
    if cluster[0] * cluster[1] * cluster[2] > MAX_CLUSTER:
        raise TilefishBuildError(f"'{stem}': the host function launches clusters of {cluster}, more than the portable {MAX_CLUSTER} CTAs")
    if cluster != (1, 1, 1) and first.policy not in (0, 1):  # CU_CLUSTER_SCHEDULING_POLICY_DEFAULT / _SPREAD (what every host asks for)
        raise TilefishBuildError(f"'{stem}': the host function asks for the cluster scheduling policy {first.policy}, not the spread one every host asks for")
    if any(g % c for g, c in zip(first.grid, cluster)):
        raise TilefishBuildError(f"'{stem}': the captured grid of {first.grid} blocks is not a whole number of clusters of {cluster}")
    grid = tuple(g // c for g, c in zip(first.grid, cluster))
    if any(not 1 <= b <= 1024 for b in first.block) or first.block[0] * first.block[1] * first.block[2] > 1024:
        raise TilefishBuildError(f"'{stem}': the captured block {first.block} is not a block a launch takes")
    return grid, first.block, first.shared, cluster  # type: ignore[return-value]


def grid_text(grid: "Sequence[int]") -> str:
    """The grid of an entry of a CuTe DSL kernel: its integers, ``"(4, 8, 1)"`` (static shapes: no dimension name)."""
    return "(" + ", ".join(str(int(g)) for g in grid) + ")"


# -- the build (a GPU, the DSL, torch) ---------------------------------------------------------------------------------------------------------


def _import_dsl(target: CuteTarget, stem: str) -> Any:
    try:
        import cutlass
        import cutlass.cute as cute  # noqa: F401
    except ImportError as e:
        raise TilefishBuildError(f"'{stem}': building a CuTe DSL kernel needs the CuTe DSL ({DISTRIBUTION}): {e}") from e
    version = dsl_version()
    if version != target.dsl:
        raise TilefishBuildError(
            f"'{stem}': the operator was declared with {DISTRIBUTION} {target.dsl}, and this process has {version}: a binary of another DSL is another "
            "binary (declare the operator again under the DSL that builds it)"
        )
    return cutlass


def _host_args(target: CuteTarget, tensors: "Sequence[Any]", stem: str) -> "list[Any]":
    """The positional arguments of the host function for these torch tensors (``cute_args``), each torch tensor made a CuTe tensor of its static
    layout, 16-byte aligned (``from_dlpack(t, assumed_align=16)``)."""
    from cutlass.cute.runtime import from_dlpack
    import torch

    made = target.args(*tensors) if target.args is not None else tuple(tensors)
    if not isinstance(made, (list, tuple)):
        raise TilefishBuildError(f"'{stem}': cute_args= returns the positional arguments of the host function (a tuple), got {type(made).__name__}")

    def convert(a: Any) -> Any:  # a torch tensor anywhere in a list or a tuple of the arguments (elementwise_apply's inputs) too
        if isinstance(a, torch.Tensor):
            return from_dlpack(a, assumed_align=16)
        if isinstance(a, (list, tuple)):
            return type(a)(convert(x) for x in a)
        return a

    return [convert(a) for a in made]


def _runtime_args(compiled: Any, args: "list[Any]") -> "list[Any]":
    """The arguments a call of the compiled host function takes of ``args`` (the host function's): every one but its ``Constexpr`` parameters, which
    the compile baked in (the DSL's own filter, ``ExecutionArgs.get_rectified_args_from_original_args``)."""
    execution = getattr(compiled, "execution_args", None)
    original = getattr(execution, "original_signature", None)
    if original is None:
        return list(args)
    from cutlass.base_dsl.runtime.jit_arg_adapters import is_arg_annotation_constexpr

    params = list(original.parameters.items())
    if params and params[0][0] == "self":  # the __call__ of a kernel object: the object is bound, not an argument
        params = params[1:]
    out = []
    for i, (value, (name, param)) in enumerate(zip(args, params)):
        if not is_arg_annotation_constexpr(param.annotation, name, i, None):
            out.append(value)
    return out + list(args[len(params) :])


def _driver() -> Any:
    from ._cuda import _driver as driver

    return driver()


def _check(result: Any, what: str) -> Any:
    from ._cuda import _check as check

    return check(result, what)


def _node_attribute(drv: Any, node: Any, name: str) -> Any:
    attr = getattr(drv.CUlaunchAttributeID, name, None)
    if attr is None:
        return None
    res = drv.cuGraphKernelNodeGetAttribute(node, attr)
    return res[1] if res[0] == drv.CUresult.CUDA_SUCCESS else None


def capture_launch(run: Callable[[], Any], stream: int, stem: str) -> Capture:
    """Run ``run()`` (a call of the compiled host function on ``stream``) while the stream is captured into a CUDA graph, so that the kernel is not
    launched, and read the one kernel node of the graph."""
    import ctypes

    drv = _driver()
    s = drv.CUstream(stream)
    _check(drv.cuStreamBeginCapture(s, drv.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED), "cuStreamBeginCapture")
    try:
        run()
    finally:
        (graph,) = _check(drv.cuStreamEndCapture(s), "cuStreamEndCapture")
    try:
        _, count = _check(drv.cuGraphGetNodes(graph, 0), "cuGraphGetNodes")
        nodes, count = _check(drv.cuGraphGetNodes(graph, count), "cuGraphGetNodes")
        nodes = list(nodes)[:count]
        kernels = [n for n in nodes if _check(drv.cuGraphNodeGetType(n), "cuGraphNodeGetType")[0] == drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL]
        if len(nodes) != 1 or len(kernels) != 1:
            raise TilefishBuildError(
                f"'{stem}': the host function enqueues {len(nodes)} operations ({len(kernels)} kernel launches): tilefish takes a host function that "
                "launches one kernel and does nothing else on the stream"
            )
        node = kernels[0]
        (p,) = _check(drv.cuGraphKernelNodeGetParams(node), "cuGraphKernelNodeGetParams")
        func = p.func
        (name,) = _check(drv.cuFuncGetName(func), "cuFuncGetName")
        name = name.decode() if isinstance(name, bytes) else str(name)
        sizes: "list[int]" = []
        while True:
            res = drv.cuFuncGetParamInfo(func, len(sizes))
            if res[0] != drv.CUresult.CUDA_SUCCESS:
                break
            sizes.append(int(res[2]))
        slots = ctypes.cast(int(p.kernelParams), ctypes.POINTER(ctypes.c_void_p))
        params = tuple(ctypes.string_at(slots[j], sizes[j]) for j in range(len(sizes)))
        cluster = _node_attribute(drv, node, "CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION")
        policy = _node_attribute(drv, node, "CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE")
        pdl = _node_attribute(drv, node, "CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION")
        coop = _node_attribute(drv, node, "CU_LAUNCH_ATTRIBUTE_COOPERATIVE")
        return Capture(
            name=name,
            grid=(int(p.gridDimX), int(p.gridDimY), int(p.gridDimZ)),
            block=(int(p.blockDimX), int(p.blockDimY), int(p.blockDimZ)),
            shared=int(p.sharedMemBytes),
            cluster=(int(cluster.clusterDim.x), int(cluster.clusterDim.y), int(cluster.clusterDim.z)) if cluster is not None else (1, 1, 1),
            policy=int(policy.clusterSchedulingPolicyPreference) if policy is not None else 0,
            params=params,
            pdl=bool(int(pdl.programmaticStreamSerializationAllowed)) if pdl is not None else False,
            cooperative=bool(int(coop.cooperative)) if coop is not None else False,
        )
    finally:
        drv.cuGraphDestroy(graph)


def _tensors_of(spec: "CompileSpec", device: Any) -> "list[Any]":
    """Fresh torch tensors of the spec's tensor arguments (their shapes at its buckets), in the order of the stub, each an allocation of its own."""
    import torch

    from ._launch import TORCH_DTYPE_NAMES

    out = []
    for a in spec.args:
        shape = [int(d) for d in (a.shape or ())]
        dtype = getattr(torch, TORCH_DTYPE_NAMES[a.dtype].split(".")[1])
        out.append(torch.empty(shape, dtype=dtype, device=device))
    return out


def compile_kernel(spec: "CompileSpec", target: CuteTarget, device_index: int) -> "tuple[Any, list[Any], Any]":
    """Compile the host function for the spec's shapes on device ``device_index``: ``(compiled function, the tensors it was compiled with, the
    stream)``. ``cute.compile`` reads no cache of the DSL's (it compiles every time); the PTX and the cubin it keeps are written to a private
    temporary directory (``--dump-dir``: by default the DSL writes them to the working directory) and read back into the compiled function."""
    import tempfile

    import torch

    _import_dsl(target, spec.stem)
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda_driver

    device = torch.device("cuda", device_index)
    tensors = _tensors_of(spec, device)
    args = _host_args(target, tensors, spec.stem)
    stream = torch.cuda.Stream(device=device)
    with tempfile.TemporaryDirectory(prefix="tilefish-cute-") as dump:
        options = " ".join(["--keep-cubin", "--keep-ptx", f"--dump-dir={dump}", target.options]).strip()
        try:
            with torch.cuda.device(device_index):
                compiled = cute.compile(target.host, *args, cuda_driver.CUstream(stream.cuda_stream), options=options)
        except TilefishBuildError:
            raise
        except Exception as e:  # noqa: BLE001 - whatever the DSL raises (a DSL error, a compile error)
            raise TilefishBuildError(f"'{spec.stem}': the CuTe DSL could not compile {target.ref}: {type(e).__name__}: {e}") from e
    return compiled, tensors, stream


def ptx_digest(spec: "CompileSpec", target: CuteTarget, device_index: int) -> str:
    """The sha256 of the PTX the DSL generates for the spec now (what a replay compares with an entry's ``source_digest``)."""
    compiled, _, _ = compile_kernel(spec, target, device_index)
    ptx = getattr(compiled, "__ptx__", None)
    if not isinstance(ptx, str) or not ptx:
        raise TilefishBuildError(f"'{spec.stem}': the CuTe DSL kept no PTX of {target.ref}")
    return hashlib.sha256(ptx.encode("utf-8")).hexdigest()


def build_cute(spec: "CompileSpec", target: CuteTarget, *, arch: "int | str", device_index: "int | None" = None) -> "BuiltKernel":
    """Build the CuTe DSL kernel ``spec`` is for, on the device of architecture ``arch`` (the current device, or ``device_index``): compile it with the
    DSL, take its cubin and PTX, capture its launch :data:`CAPTURES` times and make its ABI (:func:`classify_params`), its grid, block, shared memory
    and cluster (:func:`launch_of`), and check them against the cubin and the PTX; then the SASS assertions of the declaration. Raises
    :class:`~tilefish.errors.TilefishBuildError`."""
    import torch

    from ._build import BuiltKernel, LaunchAttributes, check_implementation_arch, check_sass, parse_arch
    from ._elf import code_digest, parse_elf

    cap = parse_arch(arch)
    check_implementation_arch(spec, cap)
    if spec.binary is None or spec.binary.get("kind") != "cute" or spec.binary != target.spec_section():
        raise TilefishBuildError(f"the declaration passed to build is not the one the compile spec '{spec.stem}' was made from")
    index = torch.cuda.current_device() if device_index is None else device_index
    major, minor = torch.cuda.get_device_capability(index)
    if major * 10 + minor != cap:
        raise TilefishBuildError(
            f"'{spec.stem}': a CuTe DSL kernel is built on a device of its architecture (its launch is captured there): sm_{cap} was asked, and device "
            f"{index} is sm_{major * 10 + minor}"
        )
    compiled, tensors, stream = compile_kernel(spec, target, index)
    kernels = list(getattr(compiled, "kernel_info", {}) or {})
    if len(kernels) != 1:
        raise TilefishBuildError(f"'{spec.stem}': {target.ref} compiles to {len(kernels)} kernels ({kernels}): tilefish takes a host function of one kernel")
    symbol = kernels[0]
    attrs = dict(compiled.kernel_info[symbol] or {})
    cubin, ptx = compiled.__cubin__, compiled.__ptx__
    if not isinstance(cubin, (bytes, bytearray)) or not cubin or not isinstance(ptx, str) or not ptx:
        raise TilefishBuildError(f"'{spec.stem}': the CuTe DSL kept no cubin or no PTX of {target.ref} (--keep-cubin --keep-ptx)")
    cubin = bytes(cubin)
    elf = parse_elf(cubin)
    if elf.section(f".text.{symbol}") is None:
        raise TilefishBuildError(f"'{spec.stem}': the cubin has no code section '.text.{symbol}'")
    from ._binary import cubin_sm

    if cubin_sm(cubin) != cap:
        raise TilefishBuildError(f"'{spec.stem}': the CuTe DSL made a cubin for sm_{cubin_sm(cubin)}, not sm_{cap}")
    layout = ptx_entry_params(ptx, symbol)
    reads = ptx_param_reads(ptx, symbol)
    # the launches: CAPTURES calls of the compiled host function on a captured stream, each with tensors at addresses of their own
    sets = [tensors] + [_tensors_of(spec, tensors[0].device if tensors else torch.device("cuda", index)) for _ in range(CAPTURES - 1)]
    captures = []
    with torch.cuda.device(index):
        import cuda.bindings.driver as cuda_driver

        for ts in sets:
            args = _host_args(target, ts, spec.stem)

            def call(args: "list[Any]" = args) -> None:
                compiled(*_runtime_args(compiled, [*args, cuda_driver.CUstream(stream.cuda_stream)]))

            try:
                captures.append(capture_launch(call, stream.cuda_stream, spec.stem))
            except TilefishBuildError:
                raise
            except Exception as e:  # noqa: BLE001 - whatever the host function raises
                raise TilefishBuildError(f"'{spec.stem}': the compiled host function {target.ref} could not be called: {type(e).__name__}: {e}") from e
    addresses = [{a.name: int(t.data_ptr()) for a, t in zip(spec.args, ts)} for ts in sets]
    grid, block, shared, cluster = launch_of(captures, symbol, spec.stem)
    for t, c in enumerate(captures):
        if [len(b) for b in c.params] != [s for s, _ in layout]:
            raise TilefishBuildError(
                f"'{spec.stem}': the loaded function takes parameters of {[len(b) for b in c.params]} bytes, its PTX entry declares {[s for s, _ in layout]}"
            )
    params = classify_params(spec.stem, list(spec.args), layout, [c.params for c in captures], addresses, reads)
    required = ptx_reqntid(ptx, symbol)
    if required is not None and required != block:
        raise TilefishBuildError(f"'{spec.stem}': the kernel requires the block {required} (.reqntid), and the host function launches {block}")
    explicit = ptx_cluster(ptx, symbol)
    if explicit is not None and explicit != cluster:
        raise TilefishBuildError(f"'{spec.stem}': the kernel requires clusters of {explicit} (.reqnctapercluster), and the host function launches {cluster}")
    known = {"CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES", "CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED"}
    unknown = [str(k) for k in attrs if getattr(k, "name", str(k)).split(".")[-1] not in known]
    if unknown:
        raise TilefishBuildError(f"'{spec.stem}': the kernel asks for function attributes {unknown}, which no tilefish host sets")
    launch = LaunchAttributes(0, 0, False, cluster, cluster[0] * cluster[1] * cluster[2], False)
    abi = {"params": params, "scratch": [], "scratch_slots": 0}
    checked = check_sass(spec, cap, cubin, ptx, symbol, {}, prebuilt=True)
    ptx_target = re.search(r"^\s*\.target\s+(\S+)", ptx, re.M)
    return BuiltKernel(
        spec=spec,
        arch=f"sm_{cap}",
        cubin=cubin,
        ptx=ptx,
        symbol=symbol,
        shared=shared,
        num_warps=0,
        sha256=hashlib.sha256(cubin).hexdigest(),
        sha256_code=code_digest(cubin),
        triton_version="",
        ptxas_version=cubin_ptxas(cubin),
        ptx_target=ptx_target.group(1) if ptx_target else "",
        options={},
        launch=launch,
        binary_digest=hashlib.sha256(ptx.encode("utf-8")).hexdigest(),
        binary_build={"source": "cute", "cutlass_dsl": target.dsl, "options": target.options},
        sass_checked=checked,
        cute={"abi": abi, "grid": list(grid), "block": list(block), "nvvm": ptx_tools(ptx)},
    )

