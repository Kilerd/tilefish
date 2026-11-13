"""Building a prebuilt binary: PTX that ``ptxas`` assembles, or a cubin that is used as it is.

``@tilefish.extern(ptx="k.ptx", symbol=..., block=..., grid=...)`` and ``@tilefish.extern(cubin={"sm_103": "k.cubin"}, ...)`` declare a kernel
that Triton does not compile (:class:`~tilefish.decorators.BinaryTarget`). Its identity is the sha256 of the bytes of the PTX or cubin
(``source_digest``, kind ``"tilefish.binary.v1"``), which the declaration read once.

* **PTX** is assembled by the configured ``ptxas`` for the target architecture with flags that do not depend on the caller:
  ``-lineinfo -v --gpu-name=<target>``, where ``<target>`` is the ``.target`` of the PTX when it names the architecture exactly
  (``sm_103a`` for ``sm_103``: an architecture-specific target only runs there) and ``sm_<arch>`` when the PTX asks for an older
  generic target. They are the flags Triton 3.5 gives ptxas (in the same spelling: ``ptxas`` records its command line in the cubin's
  ``.note.nv.tkinfo`` section, so the machine code digest of a cubin depends on it), which is what makes the PTX of a Triton build,
  declared as a prebuilt extern, assemble to a cubin with Triton's own ``sha256_code``. The flags and the version of ``ptxas`` are
  recorded in the manifest entry (``binary.ptxas_flags``, ``toolchain.ptxas``). ``ptxas`` is
  ``$TILEFISH_PTXAS_PATH``, else the one Triton is configured with (``TRITON_PTXAS_PATH`` or its bundled copy) when Triton can be
  imported, else ``$TRITON_PTXAS_PATH`` or the first ``ptxas`` on the ``PATH``.
* A **cubin** is used as it is, for the architecture it is declared for: it is refused for another (the declaration has no cubin for it),
  and a cubin whose ELF header says another architecture than its key is refused too.

In both cases the entry point must exist in the code (``.text.<symbol>``) and its parameters must be the C ABI the stub declares (PTX:
checked from the ``.entry``; a cubin: by the loaded function's parameter sizes when the driver is asked, see :mod:`tilefish._cuda`).
The result is a :class:`~tilefish._build.BuiltKernel` with no Triton in it (``triton_version`` empty).
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING

from ._build import BINARY_DIGEST_KIND, BuiltKernel, LaunchAttributes, _ptxas_version, parse_arch, ptx_type_of
from ._elf import code_digest, parse_elf
from .errors import TilefishBuildError

if TYPE_CHECKING:
    from .aot import CompileSpec
    from .decorators import BinaryTarget

_PTX_TARGET = re.compile(r"^\s*\.target\s+(sm_[0-9]+[a-z]?)\b", re.M)
_ENTRY = re.compile(r"\.entry\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
_PARAM = re.compile(r"\.param\s+(?:\.align\s+[0-9]+\s+)?\.(\w+)")
PTX_FLAGS = ("-lineinfo", "-v")  # then ``--gpu-name=<target>``: what Triton 3.5 passes (no ``--fmad``: a prebuilt PTX has its fma instructions explicit)
# the PTX parameter types that carry each C type of a launch parameter (a prebuilt kernel is free to use the signed or the bit type)
_ACCEPTED = {
    "pointer": {"u64", "s64", "b64"},
    "i32": {"u32", "s32", "b32"},
    "i64": {"u64", "s64", "b64"},
    "f32": {"f32", "b32"},
}
_ZERO_LAUNCH = LaunchAttributes(0, 0, False, (1, 1, 1), 1, False)

__all__ = ["BINARY_DIGEST_KIND", "build_binary", "find_ptxas", "ptx_gpu_name", "check_ptx_abi", "cubin_sm"]


def find_ptxas() -> str:
    """The ``ptxas`` that assembles prebuilt PTX (see the module docstring for where it comes from)."""
    env = os.environ.get("TILEFISH_PTXAS_PATH")
    if env:
        return env
    try:
        import triton  # noqa: F401
    except ImportError:
        pass
    else:
        try:
            return str(triton.knobs.nvidia.ptxas.path)
        except Exception:  # noqa: BLE001 - a Triton without the knob: fall through to the environment
            pass
    return os.environ.get("TRITON_PTXAS_PATH") or shutil.which("ptxas") or _no_ptxas()


def _no_ptxas() -> str:
    raise TilefishBuildError(
        "assembling PTX needs ptxas: set TILEFISH_PTXAS_PATH (or TRITON_PTXAS_PATH), put ptxas on the PATH, or install Triton"
    )


def _text(label: str, data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise TilefishBuildError(f"{label} is not text: PTX is UTF-8 text") from None


def ptx_gpu_name(ptx: str, cap: int, label: str) -> str:
    """The ``--gpu-name`` for assembling ``ptx`` on architecture ``cap``: its own architecture-specific target, or ``sm_<cap>``.

    The ``.target`` of the PTX must not be newer than ``cap``; one with a suffix (``sm_103a``, ``sm_100f``) is code for that
    architecture family only, so it is built for exactly ``cap`` and refused for any other.
    """
    m = _PTX_TARGET.search(ptx)
    if m is None:
        raise TilefishBuildError(f"{label} has no .target directive: it is not PTX")
    target = m.group(1)
    digits = re.match(r"sm_([0-9]+)([a-z]?)\Z", target)
    assert digits is not None
    want, suffix = int(digits.group(1)), digits.group(2)
    if suffix:
        if want != cap:
            raise TilefishBuildError(
                f"{label} is PTX for the architecture-specific target {target}, which only runs on sm_{want}; it cannot be built for sm_{cap}"
            )
        return target
    if want > cap:
        raise TilefishBuildError(f"{label} is PTX for {target}, which is newer than the sm_{cap} it is to be built for")
    return f"sm_{cap}"


def check_ptx_abi(spec: "CompileSpec", ptx: str, symbol: str) -> None:
    """The ``.entry`` of ``symbol`` must take the C parameters the spec declares, in order (the PTX types are compared by width and kind)."""
    m = next((e for e in _ENTRY.finditer(ptx) if e.group(1) == symbol), None)
    if m is None:
        raise TilefishBuildError(f"the PTX of '{spec.stem}' has no entry point '{symbol}' (it has {[e.group(1) for e in _ENTRY.finditer(ptx)]})")
    close = ptx.find(")", m.end())
    if close < 0:
        raise TilefishBuildError(f"the PTX entry '{symbol}' of '{spec.stem}' has no parameter list end")
    found = _PARAM.findall(ptx[m.end() : close])
    want = [("pointer" if a.kind == "pointer" else a.type) for a in spec.args]
    ok = len(found) == len(want) and all(f in _ACCEPTED[w] for f, w in zip(found, want))
    if not ok:
        raise TilefishBuildError(
            f"the PTX entry '{symbol}' of '{spec.stem}' does not have the C ABI its stub declares: it takes {found}, the stub's parameters "
            f"{[(a.name, ptx_type_of(a.kind, a.type)[0]) for a in spec.args]} need {[sorted(_ACCEPTED[w]) for w in want]}"
        )


def cubin_sm(cubin: bytes) -> int:
    """The SM version a cubin is for, from its ELF header.

    Checked on cubins that ptxas 12.9 made for sm_80 to sm_120: a cubin whose ``EI_OSABI`` is 0x33 (up to sm_90) has the version in the low
    byte of ``e_flags`` (80, 86, 89, 90; bit 11 is the ``a`` suffix), one whose ``EI_OSABI`` is 0x41 (sm_100 and later, the ``a``/``f``
    suffix in the low byte) has it in the second byte (100, 103, 120). Any other OS ABI is refused: the version cannot be read.
    """
    elf = parse_elf(cubin)
    osabi = elf.ident[7]
    if osabi == 0x33:
        return elf.flags & 0xFF
    if osabi == 0x41:
        return (elf.flags >> 8) & 0xFF
    raise TilefishBuildError(f"the cubin's ELF header has the OS ABI {osabi:#x}, which is not one of a CUDA cubin (0x33 or 0x41): its architecture cannot be read")


def build_binary(spec: "CompileSpec", target: "BinaryTarget", *, arch: "int | str") -> BuiltKernel:
    """Build the prebuilt binary ``spec`` is for, for ``arch``: assemble the PTX, or take the cubin declared for that architecture.

    ``target`` is the declaration's :class:`~tilefish.decorators.BinaryTarget` (it holds the bytes). Raises
    :class:`~tilefish.errors.TilefishBuildError`.
    """
    if spec.binary is None:
        raise TilefishBuildError(f"'{spec.stem}' is not a prebuilt binary")
    cap = parse_arch(arch)
    arch_name = f"sm_{cap}"
    if target.digests != spec.binary["digests"] or target.symbol != spec.binary["symbol"] or target.kind != spec.binary["kind"]:
        raise TilefishBuildError(
            f"the declaration passed to build is not the one the compile spec '{spec.stem}' was made from: its {target.kind} has "
            f"digests {target.digests}, the spec's {spec.binary['kind']} has {spec.binary['digests']}"
        )
    symbol = target.symbol
    ptxas_version = ""
    if target.kind == "cubin":
        data = target.sources.get(arch_name)
        if data is None:
            raise TilefishBuildError(
                f"'{spec.stem}' has no cubin for {arch_name} (it has {sorted(target.sources)}): a cubin is used as it is, for its own "
                "architecture, and is not built for another"
            )
        if cubin_sm(data) != cap:
            raise TilefishBuildError(
                f"the cubin declared for {arch_name} of '{spec.stem}' is for sm_{cubin_sm(data)} (its ELF header says so): a cubin is used as it "
                "is, for the architecture it was built for"
            )
        cubin = data
        build = {"source": "cubin"}
        ptx_target = ""
    else:
        text = _text(f"the PTX of '{spec.stem}'", target.sources["ptx"])
        gpu = ptx_gpu_name(text, cap, f"the PTX of '{spec.stem}'")
        check_ptx_abi(spec, text, symbol)
        ptxas = find_ptxas()
        ptxas_version = _ptxas_version(ptxas)
        flags = [*PTX_FLAGS, f"--gpu-name={gpu}"]
        with tempfile.TemporaryDirectory(prefix="tilefish-ptx-") as tmp:
            src, out = pathlib.Path(tmp) / "kernel.ptx", pathlib.Path(tmp) / "kernel.cubin"
            src.write_bytes(target.sources["ptx"])
            done = subprocess.run([ptxas, *flags, str(src), "-o", str(out)], capture_output=True, text=True, timeout=600)
            if done.returncode != 0:
                raise TilefishBuildError(f"ptxas could not assemble the PTX of '{spec.stem}' for {gpu}: {done.stderr.strip()[:2000]}")
            cubin = out.read_bytes()
        build = {"source": "ptx", "ptxas_flags": flags}
        ptx_target = gpu
    elf = parse_elf(cubin)
    if elf.section(f".text.{symbol}") is None:
        raise TilefishBuildError(f"the cubin of '{spec.stem}' has no code section '.text.{symbol}' (the entry point '{symbol}' of the declaration)")
    return BuiltKernel(
        spec=spec,
        arch=arch_name,
        cubin=cubin,
        ptx="",
        symbol=symbol,
        shared=spec.binary["shared"],
        num_warps=0,
        sha256=hashlib.sha256(cubin).hexdigest(),
        sha256_code=code_digest(cubin),
        triton_version="",
        ptxas_version=ptxas_version,
        ptx_target=ptx_target,
        options={},
        launch=_ZERO_LAUNCH,
        ir_digest=None,
        binary_digest=target.digests["ptx" if target.kind == "ptx" else arch_name],
        binary_build=build,
    )
