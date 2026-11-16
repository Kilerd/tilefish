"""The contraction check: a kernel compiled with ``enable_fp_fusion=False`` has no multiply and add that ``ptxas`` fused into one instruction.

``enable_fp_fusion=False`` asks Triton not to contract: its LLVM emits ``mul.rn`` and ``add.rn`` (no ``fma``) and its ``ptxas`` runs with
``--fmad=false``. That is not enough. On sm_100 and later LLVM packs two lanes' f32 operations into ``mul.rn.f32x2`` / ``add.rn.f32x2``, and
ptxas 12.9 contracts such a pair all the same: into ``FFMA2`` on sm_100 and sm_103, and into scalar ``FFMA`` on sm_120 (which splits the packed
PTX op), whatever ``--fmad`` says. A scalar ``mul.rn.f32`` and ``add.rn.f32`` are never contracted. Which kernels it hits depends on what LLVM
packs (measured with Triton 3.5.1 on sm_103: ``x * w + b`` of a row of widened bf16 values is contracted, ``a * b + c`` of three f32 loads
is not), so this module reads what the compiler made of each kernel rather than predicting it.

**The check.** ``S`` is the SASS of the binary. ``S0`` is the SASS of the same PTX, assembled with the same flags, with every float
multiply replaced by a bitwise ``or`` of its operands (:func:`neutralised`): a kernel that has no multiply has nothing to contract, and every
fused instruction ``S0`` still has is an explicit ``fma`` of the PTX or part of what ptxas expands an instruction into (an IEEE division, a
square root, a reciprocal). The kernel is refused when ``S`` has more fused multiply-add lanes than ``S0`` (:func:`fused`: ``FFMA``,
``FFMA32I`` and ``DFMA`` one lane, ``FFMA2`` and ``HFMA2`` two; not the idioms that multiply or add nothing: a multiplicand ``RZ``, which is how
ptxas materialises a constant, ``HFMA2 R0, -RZ, RZ, 1.5, 0``, a multiplicand ``1``, which is an add, ``HFMA2 R7, R20, 1, 1, R16``, or an addend
``RZ``, a product rounded once, which is a multiply). The
values of ``S0`` are not computed; only its instructions are counted. (An ``or``, not an ``xor``: ``x * x`` as ``x ^ x`` would be the constant
0, and ptxas would fold away what reads it, the explicit ``fma`` of a polynomial in ``x * x`` included; ``x | x`` is ``x``.) The allowance is never
below the lanes of the PTX's explicit ``fma`` instructions, in case ``S0`` lost one to a simplification.

**The disassembler.** ``nvdisasm``: ``$TILEFISH_NVDISASM_PATH``, the one of an ``nvidia-cuda-nvdisasm`` wheel (``pip install 'tilefish[sass]'``),
Triton's (``$TRITON_NVDISASM_PATH`` or its bundled copy), the one next to the ``ptxas`` Triton uses, then the first on the ``PATH`` or in
``$CUDA_HOME/bin``, the first of them that decodes the binary's architecture. Triton 3.5.1 bundles CUDA 12.8's, which decodes sm_90, sm_100 and sm_120 but not sm_103: where
none decodes the architecture, ``S`` and ``S0`` are both the PTX assembled for the base architecture of its family (``sm_100a`` for
``sm_103a``, the same instruction set), and the message of a refusal says so. ``pip install nvidia-cuda-nvdisasm`` (CUDA 13) decodes sm_103.

The functions on text are pure; :func:`check` runs ``ptxas`` and ``nvdisasm`` (no GPU).
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Iterable

from .errors import TilefishBuildError

NVDISASM_ENV = "TILEFISH_NVDISASM_PATH"
_FAMILY_BASE = {"sm_103a": "sm_100a", "sm_103f": "sm_100f", "sm_121a": "sm_120a", "sm_121f": "sm_120f"}  # a target and the base of its family

# a float multiply of PTX (any rounding, flush and saturation modifier; scalar or packed), up to the end of its statement
_MUL = re.compile(r"(?<![\w.$%])mul((?:\.r[nzmp])?(?:\.ftz)?(?:\.sat)?)\.(f32x2|f32|f64|f16x2|f16|bf16x2|bf16)(\s+)([^;{}]*);")
_BITS = {"f32x2": 64, "f32": 32, "f64": 64, "f16x2": 32, "f16": 16, "bf16x2": 32, "bf16": 16}
_FLOAT_IMM = re.compile(r"\b0[fFdD]([0-9A-Fa-f]+)\b")
# the float arithmetic of PTX that ptxas could contract: a multiply, and an add or a subtract (scalar or packed)
_ADD = re.compile(r"(?<![\w.$%])(?:add|sub)(?:\.r[nzmp])?(?:\.ftz)?(?:\.sat)?\.(?:f32x2|f32|f64|f16x2|f16|bf16x2|bf16)\s")
_EXPLICIT_FMA = re.compile(r"(?<![\w.$%])(?:fma|mad)(?:\.r[nzmp])?(?:\.ftz)?(?:\.sat)?(?:\.relu)?\.(f32x2|f32|f64|f16x2|f16|bf16x2|bf16)\s")
_LANES_PTX = {"f32x2": 2, "f16x2": 2, "bf16x2": 2}
# one SASS instruction of nvdisasm's listing: ``/*0240*/  @!P0 FFMA2 R18, R18.F32x2.HI_LO, ... ;``
_SASS = re.compile(r"/\*([0-9a-fA-F]+)\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_]*)((?:\.[A-Z0-9_]+)*)\s*([^;]*);")
_LANES_SASS = {"FFMA": 1, "FFMA32I": 1, "DFMA": 1, "FFMA2": 2, "HFMA2": 2, "HFMA2_MMA": 2}
_RZ = re.compile(r"^[-|!]*\|?U?RZ\b")


def neutralised(ptx: str) -> "tuple[str, int]":
    """``ptx`` with every float multiply replaced by a bitwise ``or.b<bits>`` of the same operands (a float immediate written as the integer of
    its bits), and how many were replaced. The result assembles (PTX lets a ``.b<bits>`` instruction take registers of any type of its size) and
    multiplies nothing: what ptxas fuses in it is the PTX's explicit ``fma`` and its own expansions, never a contraction. It depends on both
    operands and is no constant (``x * x`` is ``x``), so nothing that reads it is folded away."""

    def one(m: "re.Match[str]") -> str:
        operands = _FLOAT_IMM.sub(lambda i: "0x" + i.group(1), m.group(4))
        return f"or.b{_BITS[m.group(2)]}{m.group(3)}{operands};"

    return _MUL.subn(one, ptx)


def contractible(ptx: str) -> bool:
    """Whether the PTX has a float multiply and a float add or subtract: without both there is nothing ptxas could fuse."""
    return _MUL.search(ptx) is not None and _ADD.search(ptx) is not None


def explicit_fma_lanes(ptx: str) -> int:
    """The lanes of the explicit ``fma`` (and float ``mad``) instructions of a PTX: one per scalar, two per packed instruction."""
    return sum(_LANES_PTX.get(m.group(1), 1) for m in _EXPLICIT_FMA.finditer(ptx))


@dataclass(frozen=True)
class Fused:
    """One fused multiply-add instruction of a SASS listing: its address, its text and its lanes."""

    address: int
    text: str
    lanes: int


def _operands(text: str) -> "list[str]":
    return [o.strip() for o in text.split(",") if o.strip()]


def _unit(operand: str) -> bool:
    """Whether an operand is the immediate 1 or -1 (a multiply by it is exact: the instruction is an add)."""
    try:
        return abs(float(operand.strip("-|"))) == 1.0
    except ValueError:
        return False


def fused(sass: str) -> "list[Fused]":
    """The fused multiply-add instructions of an ``nvdisasm`` listing that multiply and add: ``FFMA`` (``FFMA32I``, ``DFMA``) and the packed
    ``FFMA2`` and ``HFMA2`` (any modifier), but not one whose multiplicand is ``RZ`` (a constant materialised: ``HFMA2 R5, -RZ, RZ, 0, 0``) or
    the immediate 1 (an add: ``HFMA2 R7, R20, 1, 1, R16``), or whose addend is ``RZ`` (a product rounded once: a multiply)."""
    out = []
    for m in _SASS.finditer(sass):
        op, mods, rest = m.group(2), m.group(3), m.group(4)
        lanes = _LANES_SASS.get(op)
        if lanes is None:
            continue
        ops = _operands(rest)
        if len(ops) < 4:
            continue
        if _RZ.match(ops[1]) or _RZ.match(ops[2]) or _RZ.match(ops[-1]) or _unit(ops[1]) or _unit(ops[2]):
            continue
        out.append(Fused(int(m.group(1), 16), f"{op}{mods} {rest.strip()}".strip(), lanes))
    return out


def lanes(found: "Iterable[Fused]") -> int:
    return sum(f.lanes for f in found)


# -- tools ---------------------------------------------------------------------------------------------------------------------------------


def _triton_tool(name: str) -> "str | None":
    try:
        import triton

        return str(getattr(triton.knobs.nvidia, name).path)
    except Exception:  # noqa: BLE001 - no Triton, or one without the knob
        return None


_WHEELS = {"nvdisasm": "nvidia-cuda-nvdisasm"}  # the wheel that ships a tool (tilefish[sass])


def _wheel_tool(name: str) -> "list[str]":
    """``bin/<name>`` among the installed files of the wheel that ships it (``nvidia-cuda-nvdisasm``: ``nvidia/cu13/bin/nvdisasm``), wherever pip
    put it, then ``nvidia/*/bin/<name>`` of the ``nvidia`` package. The installed files come first: ``import nvidia`` finds one directory when
    another location on the path has a regular ``nvidia`` package (the ``nvidia/__init__.py`` of CUDA 12 wheels), so a wheel installed with
    ``--user``, ``--target`` or into a virtual environment over the system's packages is not under it."""
    found = []
    wheel = _WHEELS.get(name)
    if wheel is not None:
        try:
            from importlib import metadata

            dist = metadata.distribution(wheel)
            found += [str(dist.locate_file(f)) for f in dist.files or () if f.name == name and f.parent.name == "bin"]
        except Exception:  # noqa: BLE001 - not installed, or a record that cannot be read
            pass
    try:
        import nvidia  # type: ignore[import-not-found]
    except ImportError:
        return found
    for root in list(getattr(nvidia, "__path__", [])):
        for p in sorted(pathlib.Path(root).glob(f"*/bin/{name}")):
            found.append(str(p))
    return found


def ptxas_path() -> str:
    """The ``ptxas`` Triton compiles with (``TRITON_PTXAS_PATH`` or its bundled copy)."""
    path = _triton_tool("ptxas")
    if path is None:
        raise TilefishBuildError("the contraction check needs the ptxas Triton compiles with, and Triton could not say where it is")
    return path


def nvdisasm_candidates() -> "list[str]":
    """Every ``nvdisasm`` this process could use, in the order they are tried (see the module docstring), each once."""
    out: "list[str]" = []
    env = os.environ.get(NVDISASM_ENV)
    if env:
        out.append(env)
    out += _wheel_tool("nvdisasm")  # an nvidia-cuda-nvdisasm wheel (tilefish[sass]: CUDA 13, which reads every architecture tilefish builds for)
    triton_path = _triton_tool("nvdisasm")
    if triton_path:
        out.append(triton_path)
    ptxas = _triton_tool("ptxas")
    if ptxas:
        out.append(str(pathlib.Path(ptxas).with_name("nvdisasm")))
    which = shutil.which("nvdisasm")
    if which:
        out.append(which)
    for var in ("CUDA_HOME", "CUDA_PATH"):
        home = os.environ.get(var)
        if home:
            out.append(str(pathlib.Path(home) / "bin" / "nvdisasm"))
    seen, unique = set(), []
    for p in out:
        if p not in seen and os.path.isfile(p) and os.access(p, os.X_OK):
            seen.add(p)
            unique.append(p)
    return unique


_decodes: "dict[tuple[str, str], bool]" = {}


def _disassemble(nvdisasm: str, cubin: bytes, flags: "Iterable[str]" = ()) -> "str | None":
    """The listing of ``cubin`` by ``nvdisasm`` (with ``flags``: ``-c -g`` for the code with line information, :mod:`tilefish._sass`), or
    ``None`` when this ``nvdisasm`` cannot decode its architecture."""
    with tempfile.TemporaryDirectory(prefix="tilefish-sass-") as d:
        path = pathlib.Path(d) / "k.cubin"
        path.write_bytes(cubin)
        r = subprocess.run([nvdisasm, *flags, str(path)], capture_output=True, text=True, cwd=d)
    if r.returncode != 0 or "Cannot decode architecture" in r.stderr or "fatal" in r.stderr:
        return None
    return r.stdout


def disassembler_for(target: str, sample: bytes) -> "str | None":
    """The first ``nvdisasm`` that decodes ``sample`` (a cubin for ``target``), or ``None``."""
    for path in nvdisasm_candidates():
        key = (path, target)
        if key not in _decodes:
            _decodes[key] = _disassemble(path, sample) is not None
        if _decodes[key]:
            return path
    return None


def assemble(ptx: str, target: str, flags: "list[str]") -> bytes:
    """``ptx`` assembled by the ``ptxas`` Triton compiles with for ``target`` (``--gpu-name``) with ``flags``, in an empty directory."""
    ptxas = ptxas_path()
    with tempfile.TemporaryDirectory(prefix="tilefish-fpcheck-") as d:
        src, out = pathlib.Path(d) / "k.ptx", pathlib.Path(d) / "k.cubin"
        src.write_text(ptx)
        r = subprocess.run([ptxas, *flags, f"--gpu-name={target}", str(src), "-o", str(out)], capture_output=True, text=True, cwd=d)
        if r.returncode != 0:
            raise TilefishBuildError(f"ptxas could not assemble the PTX for {target}: {r.stderr.strip()[:2000]}")
        return out.read_bytes()


_TARGET = re.compile(r"^(\s*\.target\s+)(sm_[0-9]+[a-z]?)", re.M)


def ptxas_flags(options: "dict[str, Any]") -> "list[str]":
    """The flags of Triton 3.5.1's ``make_cubin`` that change code generation, for a kernel compiled with ``options`` (``enable_fp_fusion=False``)."""
    flags = ["--fmad=false"]
    if options.get("maxnreg"):
        flags.append(f"--maxrregcount={int(options['maxnreg'])}")
    extra = options.get("ptx_options")
    if extra:
        flags += str(extra).split()
    return flags


@dataclass(frozen=True)
class Verdict:
    """What :func:`check` found: the fused lanes of the binary and of its neutralised PTX, the target the SASS was read for (the binary's own,
    or the base of its family), the ``nvdisasm`` that read it, and the fused instructions of the binary."""

    lanes: int
    allowed: int
    target: str
    proxy: bool
    nvdisasm: str
    instructions: "tuple[Fused, ...]"

    @property
    def contracted(self) -> bool:
        return self.lanes > self.allowed


_verdicts: "dict[tuple[str, str], Verdict]" = {}


def check(ptx: str, cubin: bytes, options: "dict[str, Any]", key: str) -> "Verdict | None":
    """Read the SASS of a binary built with ``enable_fp_fusion=False`` and of its neutralised PTX (module docstring). ``None`` when the PTX has
    nothing to contract (no float multiply, or no float add); the verdict otherwise, kept for ``key`` (the binary's sha256) in this process.
    Raises :class:`~tilefish.errors.TilefishBuildError` when no ``nvdisasm`` decodes the architecture or its family's base."""
    if not contractible(ptx):
        return None
    m = _TARGET.search(ptx)
    if m is None:
        raise TilefishBuildError("the PTX of the binary has no .target directive")
    target = m.group(2)
    cached = _verdicts.get((key, target))
    if cached is not None:
        return cached
    flags = ptxas_flags(options)
    plain, _ = neutralised(ptx)
    nvdisasm = disassembler_for(target, cubin)
    proxy = False
    if nvdisasm is not None:
        sass = _disassemble(nvdisasm, cubin)
        sass0 = _disassemble(nvdisasm, assemble(plain, target, flags))
    else:
        base = _FAMILY_BASE.get(target)
        if base is None:
            raise TilefishBuildError(f"no nvdisasm here decodes {target} ({nvdisasm_candidates() or 'none found'}); {_HOW}")
        proxied = _TARGET.sub(lambda t: t.group(1) + base, ptx, count=1)
        binary = assemble(proxied, base, flags)
        nvdisasm = disassembler_for(base, binary)
        if nvdisasm is None:
            raise TilefishBuildError(f"no nvdisasm here decodes {target} or {base} ({nvdisasm_candidates() or 'none found'}); {_HOW}")
        sass = _disassemble(nvdisasm, binary)
        sass0 = _disassemble(nvdisasm, assemble(_TARGET.sub(lambda t: t.group(1) + base, plain, count=1), base, flags))
        target, proxy = base, True
    if sass is None or sass0 is None:
        raise TilefishBuildError(f"{nvdisasm} could not disassemble the binary for {target}; {_HOW}")
    found = fused(sass)
    verdict = Verdict(lanes(found), max(lanes(fused(sass0)), explicit_fma_lanes(ptx)), target, proxy, nvdisasm, tuple(found))
    _verdicts[(key, m.group(2))] = verdict
    return verdict


_HOW = f"set {NVDISASM_ENV} to an nvdisasm that does (CUDA 12.9 or later for sm_103: pip install nvidia-cuda-nvdisasm)"
