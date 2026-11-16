"""SASS: what ``ptxas`` made of a kernel, read from ``nvdisasm`` listings, and the SASS assertions a declaration makes about it.

Every kernel goes DSL / Triton / Gluon -> TTIR -> TTGIR -> LLVM IR -> PTX -> ``ptxas`` -> SASS. This module reads the last step:

* **Listings.** :func:`parse_listing` reads an ``nvdisasm -c -g`` listing (CUDA 12.8 to 13.x; sm_90, sm_100, sm_103, sm_120): each instruction's
  function, address, predicate, opcode, modifiers and operands, the source line its line information names (``//## File "...", line N``), and,
  merged from an ``nvdisasm -c -gp`` listing of the same binary, the line of the PTX it was assembled from. The scheduling annotations of CUDA 12.8
  (``&req={0}``, ``?trans3``) and the notes of 13.x (``(*"SpillRefill"*)``) are kept apart from the operands.
* **Classes.** :func:`class_of` puts every instruction in one class of :data:`CLASSES` (``mma``, ``global_load``, ``shared_store``, ``tma``,
  ``local``, ``barrier``, ...), by its opcode alone. :func:`summary` counts them, lists the MMAs (the SASS opcode and its shape; on sm_100 and
  later the kind and shape of a ``UTC*MMA``, which its opcode does not spell, from its instruction descriptor where the listing sets it as a
  constant) and the PTX MMA instructions (whose kind the PTX spells), and adds what the ELF file says (:func:`kernel_attributes`: registers,
  stack frame, static shared memory, named barriers) and an estimate of the occupancy. :func:`by_line` counts per source line.
* **Assertions.** :func:`sass` makes a :class:`SassChecks`, the declaration of ``@tilefish.kernel(sass=...)``, ``@tilefish.extern(sass=...)``
  and ``tilefish.fusion(sass=...)``: no TF32 MMA, MMA kinds that must be used, no local-memory traffic (spills), at most N registers, instruction
  classes or opcodes that must not appear, and per architecture (``by_arch=``). It is in the declaration digest (only when given: every other
  digest is as it was). :func:`evaluate` judges a binary: :func:`tilefish._build.check_sass` runs it at every build, a cache hit included,
  and refuses a binary that violates it, naming the instructions and their source lines.
* **The disassembler.** :func:`disassemble` takes the ``nvdisasm`` the contraction check takes (:mod:`tilefish._fpcontract`:
  ``$TILEFISH_NVDISASM_PATH``, Triton's, the one next to its ``ptxas``, an ``nvidia-cuda-nvdisasm`` wheel, the ``PATH``), the first that decodes the
  architecture; where none does (Triton 3.5.1 bundles CUDA 12.8's, which cannot decode sm_103), the PTX is assembled again for the base of its
  family (``sm_100a`` for ``sm_103a``, the same instruction set) and that is read, labelled as such (:attr:`Disassembly.label`).

The functions on text are pure; :func:`disassemble` runs ``nvdisasm`` (and ``ptxas`` for the family base), never a GPU.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import re
import struct
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .errors import TilefishBuildError, TilefishDeclarationError

# -- listings ---------------------------------------------------------------------------------------------------------------------------------

_TARGET = re.compile(r"^\s*\.target\s+(sm_\w+)", re.M)
_HEADER_SM = re.compile(r"\.headerflags\s+@\"[^\"]*?\bEF_CUDA_SM(\d+)\b")
_SECTION = re.compile(r"^\s*\.section\s+\.text\.([^,\s]+)\s*,")
_LINE_INFO = re.compile(r'^\s*//##\s*File\s+"([^"]*)",\s*line\s+(\d+)(.*)$')
_INLINED = re.compile(r'inlined at\s+"([^"]*)",\s*line\s+(\d+)')
_INSTR = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+([^;]*);(.*)$")
_BODY = re.compile(r"^(@!?U?P[A-Z0-9]+\s+)?([A-Z][A-Z0-9_]*)((?:\.[A-Za-z0-9_]+)*)\s*(.*)$")
# scheduling annotations nvdisasm 12.8 prints inside an instruction (sm_120): ``&req={0}``, ``&wr=0x2``, ``?trans3``, ``?WAIT4_END_GROUP``
_SCHED = re.compile(r"\s+(?:&\w+=\{?[^\s}]*\}?|\?[A-Za-z_][\w.]*)")
_NOTE = re.compile(r'\(\*"([^"]*)"\*\)')
_LABEL = re.compile(r"^\s*[.$\w]+:\s*$")  # ``.L_x_12:``, ``mm:``, ``.text.mm:``: the next instruction is a branch target (or a function's first)
_CALL_TARGET = re.compile(r"`\(([^)`]+)\)")
PTX_LINE_FILE = ".nv_debug_ptx_txt"  # the file ``nvdisasm -gp`` names: the PTX embedded in the cubin, whose lines are the PTX's


@dataclass(frozen=True)
class Instruction:
    """One SASS instruction of a listing: its function, address, predicate (``""``, ``"@!P0"``), opcode (``"HMMA"``), modifiers
    (``("16816", "F32", "BF16")``), operands, and where it came from: the source file and line of its line information (``None`` where the
    listing has none), the source lines it was inlined at (outermost last), the line of the PTX it was assembled from, and nvdisasm's notes
    (``("SpillRefill",)``)."""

    function: str
    address: int
    predicate: str
    opcode: str
    modifiers: "tuple[str, ...]"
    operands: str
    file: "str | None" = None
    line: "int | None" = None
    inlined: "tuple[tuple[str, int], ...]" = ()
    ptx_line: "int | None" = None
    notes: "tuple[str, ...]" = ()
    labelled: bool = False  # a label is before it: a branch may land here

    @property
    def mnemonic(self) -> str:
        """The opcode with its modifiers: ``HMMA.16816.F32.BF16``."""
        return ".".join((self.opcode, *self.modifiers))

    @property
    def text(self) -> str:
        """The instruction as nvdisasm prints it, without the annotations: ``@!P0 FFMA R1, R2, R3, R4``."""
        return " ".join(p for p in (self.predicate, self.mnemonic, self.operands) if p)

    @property
    def where(self) -> str:
        """``file:line`` of its line information, or ``""``."""
        return f"{self.file}:{self.line}" if self.file is not None and self.line is not None else ""

    def describe(self) -> str:
        """``/*0410*/ STL [R1+0xf0], R7 (kernels/k.py:24)``: for a message."""
        at = f" ({self.where})" if self.where else ""
        return f"/*{self.address:04x}*/ {self.text}{at}"


@dataclass(frozen=True)
class Listing:
    """A parsed listing: the target it was disassembled for (``sm_103a``) and the instructions of each function, in address order."""

    target: "str | None"
    functions: "dict[str, tuple[Instruction, ...]]"

    def instructions(self, function: "str | None" = None) -> "tuple[Instruction, ...]":
        """The instructions of ``function`` (every function's, in order, for ``None``)."""
        if function is not None:
            return self.functions.get(function, ())
        return tuple(i for f in self.functions.values() for i in f)


def _parse(text: str) -> "tuple[str | None, dict[str, list[Instruction]]]":
    m = _TARGET.search(text)
    target = m.group(1) if m else None
    if target is None:  # an sm_90 listing names its target in the header flags: ``.headerflags @"... EF_CUDA_ACCELERATORS EF_CUDA_SM90 ..."``
        h = _HEADER_SM.search(text)
        if h is not None:
            target = f"sm_{h.group(1)}" + ("a" if "EF_CUDA_ACCELERATORS" in text[: h.end() + 200] else "")
    functions: "dict[str, list[Instruction]]" = {}
    current: "str | None" = None
    file: "str | None" = None
    line: "int | None" = None
    inlined: "tuple[tuple[str, int], ...]" = ()
    labelled = False
    for raw in text.split("\n"):
        s = _SECTION.match(raw)
        if s is not None:
            current = s.group(1)
            functions.setdefault(current, [])
            file, line, inlined, labelled = None, None, (), True
            continue
        if _LABEL.match(raw):
            labelled = True
            continue
        li = _LINE_INFO.match(raw)
        if li is not None:
            file, line = li.group(1), int(li.group(2))
            inlined = tuple((f, int(n)) for f, n in _INLINED.findall(li.group(3)))
            continue
        ins = _INSTR.match(raw)
        if ins is None or current is None:
            continue
        notes = tuple(_NOTE.findall(ins.group(2))) + tuple(_NOTE.findall(ins.group(3)))  # (13.x prints a note before the ";")
        body = _SCHED.sub("", _NOTE.sub("", ins.group(2))).strip()
        b = _BODY.match(body)
        if b is None:
            continue
        mods = tuple(x for x in b.group(3).split(".") if x)
        functions[current].append(
            Instruction(
                function=current,
                address=int(ins.group(1), 16),
                predicate=(b.group(1) or "").strip(),
                opcode=b.group(2),
                modifiers=mods,
                operands=re.sub(r"\s+", " ", b.group(4)).strip(),
                file=file,
                line=line,
                inlined=inlined,
                notes=notes,
                labelled=labelled,
            )
        )
        labelled = False
    return target, functions


def reached(listing: Listing, symbol: str) -> "list[str]":
    """The functions of a listing the kernel ``symbol`` runs: itself, then every function it reaches through ``CALL`` (a ``noinline`` function, a
    subroutine of ptxas), in order; never another ``.entry`` of the binary (a kernel calls no kernel)."""
    seen = [symbol] if symbol in listing.functions else []
    todo = list(seen)
    while todo:
        for ins in listing.functions.get(todo.pop(0), ()):
            if ins.opcode == "CALL":
                for target in _CALL_TARGET.findall(ins.operands):
                    if target in listing.functions and target not in seen:
                        seen.append(target)
                        todo.append(target)
    return seen


def kernel_instructions(listing: Listing, symbol: str) -> "tuple[Instruction, ...]":
    """Every instruction the kernel ``symbol`` runs: its own function's, then those of the functions it calls (:func:`reached`)."""
    return tuple(i for name in reached(listing, symbol) for i in listing.functions[name])


def parse_listing(text: str, ptx_lines: "str | None" = None) -> Listing:
    """Parse an ``nvdisasm -c -g`` listing (``-c`` alone parses too: no line information), and, given the ``nvdisasm -c -gp`` listing of the same
    binary (``ptx_lines``), the PTX line of each instruction. Pure; lines that are not instructions, sections or line information are skipped."""
    target, functions = _parse(text)
    if ptx_lines is not None:
        _, gp = _parse(ptx_lines)
        lines = {(i.function, i.address): i.line for f in gp.values() for i in f if i.file == PTX_LINE_FILE}
        for name, ins in functions.items():
            functions[name] = [dataclasses.replace(i, ptx_line=lines.get((name, i.address))) for i in ins]
    return Listing(target, {k: tuple(v) for k, v in functions.items()})


# -- classes ----------------------------------------------------------------------------------------------------------------------------------

# Every instruction has one class, by its opcode: the first rule that matches. Extend the table to add a class (a class name is what
# ``tilefish.sass(forbid=...)`` names, so a new one is a new word a declaration can use; an opcode pattern needs no class).
_MMA = re.compile(r"^(?:[HIQOBD]MMA|[HIQOB]GMMA|UTC[A-Z]*MMA)$")
CLASS_RULES: "tuple[tuple[str, str, Callable[[str], bool]], ...]" = (
    ("mma", "tensor-core MMA: HMMA, IMMA, QMMA (mma.sync), HGMMA, IGMMA, QGMMA (wgmma), UTCHMMA, UTCQMMA, UTCIMMA, UTCOMMA (tcgen05.mma)", lambda o: _MMA.match(o) is not None),
    ("tma", "TMA and bulk copies: UTMALDG, UTMASTG, UTMAPF, UTMAREDG, UTMACCTL, UBLKCP, UBLKRED, UBLKPF", lambda o: o.startswith(("UTMA", "UBLK"))),
    ("tensor_memory", "tensor memory (tcgen05.ld/st/cp/alloc/shift): LDTM, STTM, UTCCP, UTCATOMSWS, UTCSHIFT", lambda o: o in ("LDTM", "STTM", "UTCCP", "UTCATOMSWS", "UTCSHIFT")),
    ("local", "local memory (spills and stack): LDL, STL", lambda o: o in ("LDL", "STL")),
    ("global_load", "global (and generic) loads, cp.async: LDG, LDGSTS, LD", lambda o: o in ("LDG", "LDGSTS", "LD")),
    ("global_store", "global (and generic) stores: STG, ST", lambda o: o in ("STG", "ST")),
    ("shared_load", "shared loads: LDS, LDSM", lambda o: o in ("LDS", "LDSM")),
    ("shared_store", "shared stores: STS, STSM", lambda o: o in ("STS", "STSM")),
    ("atomic", "global (and generic) atomics and reductions: ATOM, ATOMG, RED, REDG, REDAS", lambda o: o in ("ATOM", "ATOMG", "RED", "REDG", "REDAS")),
    ("shared_atomic", "shared-memory atomics: ATOMS (a tcgen05 kernel's tensor-memory allocation uses them)", lambda o: o == "ATOMS"),
    (
        "barrier",
        "barriers and waits: BAR, SYNCS (mbarrier), WARPGROUP, WARPSYNC, UTCBAR (tcgen05.commit), DEPBAR, LDGDEPBAR, ARRIVES, cluster barriers",
        lambda o: o in ("BAR", "BARRIER", "SYNCS", "WARPGROUP", "WARPSYNC", "UTCBAR", "DEPBAR", "LDGDEPBAR", "ARRIVES") or "CGABAR" in o,
    ),
    ("fence", "fences: MEMBAR, FENCE, ERRBAR, CCTL", lambda o: o in ("MEMBAR", "FENCE", "ERRBAR", "CCTL", "CCTLL")),
    ("constant", "constant-bank loads: LDC, LDCU, ULDC", lambda o: o in ("LDC", "LDCU", "ULDC")),
    ("fp64", "double precision: DADD, DMUL, DFMA, DSETP, DMNMX, DSET", lambda o: o in ("DADD", "DMUL", "DFMA", "DSETP", "DMNMX", "DSET")),
    ("mufu", "the special function unit: MUFU", lambda o: o == "MUFU"),
    ("convert", "conversions: F2F, F2I, I2F, F2FP, I2FP, FRND, F2IP", lambda o: o in ("F2F", "F2I", "I2F", "F2FP", "I2FP", "FRND", "F2IP", "I2I", "I2IP")),
    ("shuffle", "warp shuffles and votes: SHFL, VOTE, VOTEU, MATCH, REDUX, CREDUX", lambda o: o in ("SHFL", "VOTE", "VOTEU", "MATCH", "REDUX", "CREDUX")),
    (
        "control",
        "control flow: BRA, BRX, CALL, RET, EXIT, BSSY, BSYNC, BREAK, NOP, ELECT, NANOSLEEP, YIELD, KILL",
        lambda o: o in ("BRA", "BRX", "BRXU", "JMP", "JMX", "CALL", "RET", "EXIT", "BSSY", "BSYNC", "BREAK", "NOP", "ELECT", "NANOSLEEP", "YIELD", "KILL", "BPT", "ACQBULK", "ACQSHMINIT"),
    ),
    ("special", "special registers: S2R, S2UR, CS2R, R2UR, S2UR", lambda o: o in ("S2R", "S2UR", "CS2R", "R2UR", "R2P", "P2R")),
    ("float", "f32 and f16 arithmetic: FADD, FMUL, FFMA, FFMA2, FMNMX, FSETP, HADD2, HFMA2, ...", lambda o: o[:1] in ("F", "H") and not o.startswith("FENCE")),
    ("uniform", "uniform datapath: UMOV, UIADD3, ULOP3, ...", lambda o: o.startswith("U")),
    ("integer", "integer and logic: IMAD, IADD3, LOP3, SHF, LEA, ISETP, MOV, SEL, PRMT, ...", lambda o: True),
)
CLASSES: "dict[str, str]" = {name: what for name, what, _ in CLASS_RULES}  # class -> what it holds
# the columns of the per-line view: a few classes each, in order
LINE_COLUMNS: "tuple[tuple[str, tuple[str, ...]], ...]" = (
    ("mma", ("mma",)),
    ("gmem", ("global_load", "global_store")),
    ("smem", ("shared_load", "shared_store")),
    ("tma", ("tma",)),
    ("tmem", ("tensor_memory",)),
    ("local", ("local",)),
    ("atom", ("atomic", "shared_atomic")),
    ("sync", ("barrier", "fence")),
)


@functools.lru_cache(maxsize=4096)
def class_of_opcode(opcode: str) -> str:
    """The class of an opcode (:data:`CLASSES`)."""
    for name, _, rule in CLASS_RULES:
        if rule(opcode):
            return name
    return "integer"  # pragma: no cover - the last rule takes everything


def class_of(ins: Instruction) -> str:
    """The class of an instruction (:data:`CLASSES`), by its opcode."""
    return class_of_opcode(ins.opcode)


# -- MMAs -------------------------------------------------------------------------------------------------------------------------------------

_UMOV = re.compile(r"^(UR\d+),\s*(0x[0-9a-fA-F]+|\d+)$")
_DEST = re.compile(r"^U?R(\d+)\b")
_UDEST = re.compile(r"^UR(\d+)\b")
_IDESC = re.compile(r"idesc\[(UR\d+)\]")
# the type fields of a tcgen05 instruction descriptor (PTX ISA, "Instruction descriptor"), per SASS opcode: kind::f16 and kind::tf32 are UTCHMMA,
# kind::f8f6f4 is UTCQMMA, kind::i8 is UTCIMMA; the D type is bits 4-5, A bits 7-9, B bits 10-12, N >> 3 bits 17-22, M >> 4 bits 24-28
_ATYPES = {"UTCHMMA": {0: "f16", 1: "bf16", 2: "tf32"}, "UTCQMMA": {0: "e4m3", 1: "e5m2", 3: "e2m3", 4: "e3m2", 5: "e2m1"}, "UTCIMMA": {0: "u8", 1: "s8"}}
_KINDS = {"UTCHMMA": {"f16": "f16", "bf16": "f16", "tf32": "tf32"}, "UTCQMMA": "f8f6f4", "UTCIMMA": "i8"}
_DTYPES = {0: "f16", 1: "f32", 2: "s32"}
_PTX_MMA = re.compile(r"(?<![\w.])(tcgen05\.mma[\w.:]*|wgmma\.mma_async[\w.:]*|wmma\.mma[\w.:]*|mma(?:\.sp(?:::ordered_metadata)?)?\.sync[\w.:]*)")
_SASS_SHAPE = re.compile(r"^(?:\d+x\d+x\d+|\d{3,5})$")


def decode_idesc(opcode: str, value: int) -> "dict[str, Any] | None":
    """What a tcgen05 instruction descriptor says (``kind``, ``a``, ``b``, ``d`` types, ``m``, ``n``), for a ``UTCHMMA``, ``UTCQMMA`` or
    ``UTCIMMA``; ``None`` for another opcode. Best effort: the fields of the PTX ISA's table, for a dense, one-CTA MMA."""
    atypes = _ATYPES.get(opcode)
    if atypes is None:
        return None
    a, b = atypes.get((value >> 7) & 7, "?"), atypes.get((value >> 10) & 7, "?")
    kinds = _KINDS[opcode]
    kind = kinds.get(a, "?") if isinstance(kinds, dict) else kinds
    return {"kind": kind, "a": a, "b": b, "d": _DTYPES.get((value >> 4) & 3, "?"), "m": ((value >> 24) & 0x1F) << 4, "n": ((value >> 17) & 0x3F) << 3}


def mma_descriptors(instructions: "Sequence[Instruction]") -> "dict[int, dict[str, Any]]":
    """For each ``UTC*MMA`` of a function (by address) whose instruction descriptor the listing sets as a constant (``UMOV UR27, 0x8210490``
    before it, nothing writing the register in between), what the descriptor says (:func:`decode_idesc`). Best effort: a descriptor computed at
    run time, set on another path (the constants are forgotten at every label, where a branch may land) or written in part, is not found."""
    consts: "dict[str, int]" = {}
    out: "dict[int, dict[str, Any]]" = {}
    for ins in instructions:
        if ins.labelled:
            consts.clear()
        if ins.opcode.startswith("UTC") and ins.opcode.endswith("MMA"):
            m = _IDESC.search(ins.operands)
            if m is not None and m.group(1) in consts:
                found = decode_idesc(ins.opcode, consts[m.group(1)])
                if found is not None:
                    out[ins.address] = found
            continue
        if ins.opcode == "UMOV" and not ins.predicate and not ins.modifiers:
            u = _UMOV.match(ins.operands)
            if u is not None:
                consts[u.group(1)] = int(u.group(2), 0)
                continue
        d = _UDEST.match(ins.operands)
        if d is not None:  # a write of a uniform register: of two or four with .64 or .128 (LDCU.64 UR4: UR4 and UR5)
            width = 4 if "128" in ins.modifiers else 2 if "64" in ins.modifiers else 1
            for k in range(int(d.group(1)), int(d.group(1)) + width):
                consts.pop(f"UR{k}", None)
    return out


def is_tf32_mma(ins: Instruction, descriptor: "Mapping[str, Any] | None" = None) -> bool:
    """Whether an MMA multiplies TF32: a ``.TF32`` modifier (``HMMA.1688.F32.TF32``, ``HGMMA.64x64x8.F32.TF32``), or a ``UTCHMMA`` whose
    descriptor says ``kind::tf32``."""
    if class_of(ins) != "mma":
        return False
    if "TF32" in ins.modifiers:
        return True
    return bool(descriptor and descriptor.get("kind") == "tf32")


def is_tf32_ptx(opcode: str) -> bool:
    """Whether a PTX MMA opcode multiplies TF32: ``tcgen05.mma ... kind::tf32``, ``wgmma ... .tf32``, ``wmma.mma ... .tf32``, ``mma.sync ... .tf32``."""
    return "kind::tf32" in opcode if opcode.startswith("tcgen05") else ".tf32" in opcode


def ptx_instructions(ptx: str) -> "list[tuple[int, str]]":
    """The MMA instructions of a PTX text: ``(line, opcode)``, line from 1, comments skipped."""
    out = []
    for n, line in enumerate(ptx.split("\n"), 1):  # (not splitlines: a form feed in a comment must not move the line numbers)
        code = line.split("//", 1)[0]
        for m in _PTX_MMA.finditer(code):
            out.append((n, m.group(1)))
    return out


_PTX_LOC = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)")
_PTX_FILE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]*)"')


def ptx_locations(ptx: str) -> "dict[int, tuple[str, int]]":
    """The source line of each line of a PTX text (from 1): the file and line of the last ``.loc`` before it (line 0 is no line)."""
    files = {m.group(1): m.group(2) for m in (_PTX_FILE.match(x) for x in ptx.split("\n")) if m}
    out: "dict[int, tuple[str, int]]" = {}
    current: "tuple[str, int] | None" = None
    for n, line in enumerate(ptx.split("\n"), 1):
        m = _PTX_LOC.match(line)
        if m is not None:
            current = (files.get(m.group(1), f"file {m.group(1)}"), int(m.group(2))) if int(m.group(2)) > 0 else None
            continue
        if current is not None:
            out[n] = current
    return out


_PTX_FUNC = re.compile(r"^\s*(?:\.(?:visible|weak|extern)\s+)*\.(entry|func)\b(?:\s*\([^)]*\))?\s*([\w$]+)")
_PTX_CALL = re.compile(r"\bcall(?:\.uni)?\s+(?:\([^)]*\)\s*,\s*)?([\w$]+)")


def ptx_for_entry(ptx: str, symbol: str) -> str:
    """``ptx`` with the bodies of every function the entry ``symbol`` does not reach (another ``.entry``, a ``.func`` it never calls) blanked, the
    lines kept in place (so the line numbers, the ``.file`` and the ``.loc`` directives are the PTX's); ``ptx`` itself when it has no such entry."""
    lines = ptx.split("\n")
    bodies: "dict[str, tuple[int, int]]" = {}
    kinds: "dict[str, str]" = {}
    k = 0
    while k < len(lines):
        m = _PTX_FUNC.match(lines[k])
        if m is None:
            k += 1
            continue
        depth, opened, start, end = 0, False, k, None
        for j in range(k, len(lines)):
            code = lines[j].split("//", 1)[0]
            if not opened and "{" not in code and code.rstrip().endswith(";"):
                break  # a prototype: no body
            for ch in code:
                if ch == "{":
                    depth, opened = depth + 1, True
                elif ch == "}":
                    depth -= 1
            if opened and depth == 0:
                end = j
                break
        if end is None:
            k += 1
            continue
        bodies[m.group(2)] = (start, end)
        kinds[m.group(2)] = m.group(1)
        k = end + 1
    if kinds.get(symbol) != "entry":
        return ptx
    keep = [symbol]
    todo = [symbol]
    while todo:
        s, e = bodies[todo.pop()]
        for line in lines[s : e + 1]:
            for callee in _PTX_CALL.findall(line.split("//", 1)[0]):
                if callee in bodies and callee not in keep:
                    keep.append(callee)
                    todo.append(callee)
    for name, (s, e) in bodies.items():
        if name not in keep:
            for j in range(s, e + 1):
                lines[j] = ""
    return "\n".join(lines)


# -- what the ELF file says --------------------------------------------------------------------------------------------------------------------

_EIATTR = {0x2F: "registers", 0x11: "frame_size", 0x12: "min_stack_size", 0x23: "max_stack_size"}  # per function, in .nv.info
_EIATTR_OWN = {0x1B: "maxreg_count", 0x4C: "num_barriers", 0x10: "reqntid", 0x05: "max_threads"}  # in .nv.info.<function>
_SYM = struct.Struct("<IBBHQQ")


def _nv_info(data: bytes) -> "list[tuple[int, int, bytes]]":
    """The attributes of an ``.nv.info`` section: ``(format, attribute, value bytes)``; formats 1 (no value), 2 (a byte), 3 (16 bits), 4 (sized)."""
    out = []
    off = 0
    while off + 2 <= len(data):
        fmt, attr = data[off], data[off + 1]
        if fmt == 1:
            value, size = b"", 2
        elif fmt == 2:
            value, size = data[off + 2 : off + 3], 4
        elif fmt == 3:
            value, size = data[off + 2 : off + 4], 4
        elif fmt == 4:
            if off + 4 > len(data):
                break
            (n,) = struct.unpack_from("<H", data, off + 2)
            value, size = data[off + 4 : off + 4 + n], 4 + n
        else:
            break
        out.append((fmt, attr, value))
        off += (size + 3) & ~3
    return out


def _symbols(elf: Any) -> "dict[int, str]":
    sym = next((s for s in elf.sections if s.type == 2), None)  # SHT_SYMTAB
    if sym is None or sym.link >= len(elf.sections):
        return {}
    names = elf.data(elf.sections[sym.link])
    data = elf.data(sym)
    out = {}
    for i in range(len(data) // _SYM.size):
        name_off = _SYM.unpack_from(data, i * _SYM.size)[0]
        end = names.find(b"\0", name_off)
        out[i] = names[name_off:end].decode("utf-8", "replace") if end >= 0 else ""
    return out


def kernel_attributes(cubin: bytes, symbol: str) -> "dict[str, Any]":
    """What the cubin says about the function ``symbol``: ``registers`` (per thread), ``frame_size`` (local memory per thread: spills and
    stack), ``min_stack_size``/``max_stack_size``, ``static_shared`` (``.nv.shared.<symbol>``), ``num_barriers`` (named barriers),
    ``maxreg_count``, ``reqntid``/``max_threads`` (when set) and ``sm``. From the real binary, never from a re-assembly. Raises
    :class:`~tilefish.errors.TilefishBuildError` for a file that is not ELF."""
    from ._elf import parse_elf, shared_sizes, sm_of

    elf = parse_elf(cubin)
    out: "dict[str, Any]" = {"sm": sm_of(elf)}
    text = elf.section(f".text.{symbol}")
    if text is not None and text.info >> 24:  # (an older layout; EIATTR_REGCOUNT below says it where it is)
        out["registers"] = text.info >> 24
    symbols = _symbols(elf)
    info = elf.section(".nv.info")
    if info is not None:
        for fmt, attr, value in _nv_info(elf.data(info)):
            if fmt == 4 and attr in _EIATTR and len(value) >= 8:
                index, v = struct.unpack_from("<II", value)
                if symbols.get(index) == symbol:
                    out[_EIATTR[attr]] = v
    own = elf.section(f".nv.info.{symbol}")
    if own is not None:
        for fmt, attr, value in _nv_info(elf.data(own)):
            name = _EIATTR_OWN.get(attr)
            if name is None:
                continue
            if fmt == 2 and value:
                out[name] = value[0]
            elif fmt == 3 and len(value) == 2:
                out[name] = struct.unpack("<H", value)[0]
            elif fmt == 4 and len(value) >= 12:
                out[name] = list(struct.unpack_from("<III", value))
    declared = shared_sizes(elf, symbol)[0]
    sm = out["sm"] or 0
    # the .nv.shared.<symbol> section of sm_90 and later holds the 1024 bytes the system reserves per block too (a Triton kernel's is those 1024
    # bytes alone, a hand-written one with 4096 bytes of static shared memory has 5120: tilefish._elf.shared_sizes); the reserve is counted once,
    # by the occupancy estimate
    out["static_shared"] = max(0, declared - SHARED_RESERVE) if sm >= 90 else declared
    out["local"] = max(out.get(k, 0) or 0 for k in ("frame_size", "min_stack_size", "max_stack_size"))
    return out


SHARED_RESERVE = 1024  # the shared memory the system reserves per block on sm_90 and later


def code_shape(cubin: bytes) -> "dict[str, tuple[int, int, int]]":
    """For every function of a cubin (``.text.<name>``): its registers, stack frame and code size, what a re-assembly must have to stand for the
    binary (:func:`disassemble`)."""
    from ._elf import parse_elf

    elf = parse_elf(cubin)
    symbols = _symbols(elf)
    found: "dict[int, dict[str, int]]" = {0x2F: {}, 0x11: {}}  # EIATTR_REGCOUNT and EIATTR_FRAME_SIZE, per function
    info = elf.section(".nv.info")
    if info is not None:
        for fmt, attr, value in _nv_info(elf.data(info)):
            if fmt == 4 and attr in found and len(value) >= 8:
                index, v = struct.unpack_from("<II", value)
                found[attr][symbols.get(index, "")] = v
    out = {}
    for s in elf.sections:
        if s.name.startswith(".text."):
            name = s.name[len(".text.") :]
            out[name] = (found[0x2F].get(name, s.info >> 24), found[0x11].get(name, 0), s.size)
    return out


# -- occupancy (an estimate) ------------------------------------------------------------------------------------------------------------------

# per SM: registers, warps, resident blocks, shared memory (bytes), and the shared memory the system reserves per block
_LIMITS = {
    90: (65536, 64, 32, 233472, SHARED_RESERVE),
    100: (65536, 64, 32, 233472, SHARED_RESERVE),
    103: (65536, 64, 32, 233472, SHARED_RESERVE),
    120: (65536, 48, 32, 102400, SHARED_RESERVE),
}


def occupancy(arch: int, registers: int, threads: int, shared: int) -> "dict[str, Any] | None":
    """An estimate of the blocks and warps per SM a launch of ``threads`` threads with ``registers`` per thread and ``shared`` bytes of shared
    memory (static and dynamic, without the system's reserve, which is added once per block) gets on ``arch`` (registers allocated per warp in
    units of 256), and what limits it; ``None`` for an architecture this does not know.
    Not the driver's figure (``cudaOccupancyMaxActiveBlocksPerMultiprocessor``): what the attributes allow."""
    limits = _LIMITS.get(arch)
    if limits is None or threads <= 0:
        return None
    regs_sm, warps_sm, blocks_sm, smem_sm, reserved = limits
    warps = -(-threads // 32)
    per_warp = -(-max(registers, 1) * 32 // 256) * 256
    by = {
        "registers": regs_sm // (per_warp * warps) if per_warp * warps else blocks_sm,
        "warps": warps_sm // warps,
        "blocks": blocks_sm,
        "shared": smem_sm // (shared + reserved),
    }
    blocks = min(by.values())
    return {
        "blocks_per_sm": blocks,
        "warps_per_sm": blocks * warps,
        "occupancy": round(blocks * warps / warps_sm, 3),
        "limited_by": sorted(k for k, v in by.items() if v == blocks),
    }


# -- the summary and the per-line view --------------------------------------------------------------------------------------------------------


def _counts(items: "Iterable[str]") -> "dict[str, int]":
    out: "dict[str, int]" = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def mma_shape(ins: Instruction) -> str:
    """The shape a SASS MMA spells (``16816``, ``64x128x16``), or ``""``."""
    return next((m for m in ins.modifiers if _SASS_SHAPE.match(m)), "")


def summary(
    instructions: "Sequence[Instruction]",
    *,
    ptx: "str | None" = None,
    attributes: "Mapping[str, Any] | None" = None,
    threads: int = 0,
    shared: int = 0,
    arch: "int | None" = None,
) -> "dict[str, Any]":
    """The instruction mix of one function: ``instructions`` (total), ``classes`` (counts per class), ``mma`` (each SASS MMA mnemonic with its
    count and, for a ``UTC*MMA``, what its descriptor says), ``ptx_mma`` (the PTX MMA opcodes, which spell the kind), ``tf32_mma``, ``spills``
    (local loads and stores; ``SpillRefill`` notes), ``barriers`` and ``fences``; with ``attributes`` (:func:`kernel_attributes`) the
    registers, stack frame, static shared memory and named barriers, and, with ``threads`` (a block) and ``shared`` (the dynamic shared memory
    of a launch) on a known ``arch``, :func:`occupancy`."""
    descriptors = mma_descriptors(instructions)
    classes = _counts(class_of(i) for i in instructions)
    mmas: "dict[str, dict[str, Any]]" = {}
    for i in instructions:
        if class_of(i) != "mma":
            continue
        d = descriptors.get(i.address)
        label = i.mnemonic if d is None else f"{i.mnemonic} (kind::{d['kind']} {d['a']}->{d['d']} M{d['m']} N{d['n']})"
        entry = mmas.setdefault(label, {"sass": i.mnemonic, "count": 0, "shape": mma_shape(i)})
        entry["count"] += 1
        if d is not None:
            entry["descriptor"] = d
    local = [i for i in instructions if class_of(i) == "local"]
    out: "dict[str, Any]" = {
        "instructions": len(instructions),
        "classes": classes,
        "mma": sorted(mmas.values(), key=lambda e: (-e["count"], e["sass"])),
        "tf32_mma": sum(1 for i in instructions if is_tf32_mma(i, descriptors.get(i.address))),
        "spills": {
            "loads": sum(1 for i in local if i.opcode == "LDL"),
            "stores": sum(1 for i in local if i.opcode == "STL"),
            "noted": sum(1 for i in instructions if "SpillRefill" in i.notes),
        },
        "barriers": _counts(i.mnemonic for i in instructions if class_of(i) == "barrier"),
        "fences": _counts(i.mnemonic for i in instructions if class_of(i) == "fence"),
        "tma": _counts(i.mnemonic for i in instructions if class_of(i) == "tma"),
    }
    if ptx is not None:
        out["ptx_mma"] = [{"ptx": op, "count": n} for op, n in _counts(op for _, op in ptx_instructions(ptx)).items()]
    if attributes is not None:
        for key in ("registers", "frame_size", "max_stack_size", "static_shared", "num_barriers", "maxreg_count"):
            if key in attributes:
                out[key] = attributes[key]
        if arch is not None and "registers" in attributes and threads:
            occ = occupancy(arch, int(attributes["registers"]), threads, shared + int(attributes.get("static_shared", 0) or 0))
            if occ is not None:
                out["occupancy"] = occ
    return out


def column_of(cls: str) -> str:
    """The column of the per-line view a class counts in (``alu`` for the rest)."""
    return next((name for name, classes in LINE_COLUMNS if cls in classes), "alu")


def by_line(instructions: "Sequence[Instruction]") -> "list[dict[str, Any]]":
    """Instructions per source line: ``file``, ``line`` (``None``: no line information), ``count``, ``columns`` (counts per column of
    :data:`LINE_COLUMNS`, and ``alu``), ``first`` (the address of its first instruction); in order of first address."""
    rows: "dict[tuple[Any, Any], dict[str, Any]]" = {}
    for i in instructions:
        key = (i.file, i.line)
        row = rows.get(key)
        if row is None:
            row = rows[key] = {"file": i.file, "line": i.line, "count": 0, "columns": {}, "first": i.address}
        row["count"] += 1
        col = column_of(class_of(i))
        row["columns"][col] = row["columns"].get(col, 0) + 1
    return sorted(rows.values(), key=lambda r: r["first"])


def diff_summaries(a: "Mapping[str, Any]", b: "Mapping[str, Any]") -> "list[str]":
    """What differs between two summaries (:func:`summary`), one line each: ``classes.mma: 4 -> 128``."""
    out = []
    for key in ("instructions", "registers", "frame_size", "static_shared", "num_barriers", "tf32_mma"):
        if a.get(key) != b.get(key):
            out.append(f"{key}: {a.get(key)} -> {b.get(key)}")
    for group in ("classes", "spills", "barriers", "tma"):
        x, y = a.get(group, {}) or {}, b.get(group, {}) or {}
        for k in sorted(set(x) | set(y)):
            if x.get(k, 0) != y.get(k, 0):
                out.append(f"{group}.{k}: {x.get(k, 0)} -> {y.get(k, 0)}")
    ma = {e["sass"] + (f" {e['descriptor']['kind']}" if "descriptor" in e else ""): e["count"] for e in a.get("mma", [])}
    mb = {e["sass"] + (f" {e['descriptor']['kind']}" if "descriptor" in e else ""): e["count"] for e in b.get("mma", [])}
    for k in sorted(set(ma) | set(mb)):
        if ma.get(k, 0) != mb.get(k, 0):
            out.append(f"mma {k}: {ma.get(k, 0)} -> {mb.get(k, 0)}")
    pa = {e["ptx"]: e["count"] for e in a.get("ptx_mma", [])}
    pb = {e["ptx"]: e["count"] for e in b.get("ptx_mma", [])}
    for k in sorted(set(pa) | set(pb)):
        if pa.get(k, 0) != pb.get(k, 0):
            out.append(f"ptx {k}: {pa.get(k, 0)} -> {pb.get(k, 0)}")
    oa, ob = a.get("occupancy"), b.get("occupancy")
    if oa and ob and oa.get("occupancy") != ob.get("occupancy"):
        out.append(f"occupancy (estimate): {oa['occupancy']} -> {ob['occupancy']}")
    return out


# -- the assertions ---------------------------------------------------------------------------------------------------------------------------

_ARCH_KEY = re.compile(r"sm_[1-9][0-9]{1,2}\Z")
_SASS_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*(?:\.[A-Za-z0-9_]+)*\Z")
_PTX_PATTERN = re.compile(r"[a-z][a-z0-9_]*(?:[. ][a-z0-9_:]+)*\Z")
_PTX_MMA_ROOTS = ("tcgen05.mma", "wgmma", "mma")
MAX_REGISTERS = 255


def _tokens(pattern: str) -> "list[str]":
    return [t for t in re.split(r"[.\s]+", pattern.strip()) if t]


def _subsequence(want: "Sequence[str]", have: "Sequence[str]") -> bool:
    it = iter(have)
    return all(any(w == h for h in it) for w in want)


def sass_pattern_matches(pattern: str, ins: Instruction) -> bool:
    """Whether a SASS opcode pattern (``HMMA.16816.F32``, ``LDL``, ``UTCHMMA``) matches an instruction: the same opcode, and the pattern's
    modifiers among the instruction's, in order."""
    want = _tokens(pattern)
    return bool(want) and want[0] == ins.opcode and _subsequence(want[1:], ins.modifiers)


def ptx_pattern_matches(pattern: str, opcode: str) -> bool:
    """Whether a PTX pattern (``tcgen05.mma kind::f16``, ``mma.sync m16n8k16``) matches a PTX opcode (``tcgen05.mma.cta_group::1.kind::f16``):
    its tokens (split on dots and spaces) among the opcode's, in order, the first two the opcode's own."""
    want, have = _tokens(pattern), _tokens(opcode)
    return bool(want) and have[: min(2, len(want))] == want[: min(2, len(want))] and _subsequence(want, have)


def _check_pattern(where: str, pattern: Any, *, mma: bool) -> str:
    if not isinstance(pattern, str) or not pattern.strip():
        raise TilefishDeclarationError(f"{where}: a pattern is a non-empty string, got {pattern!r}")
    p = " ".join(pattern.split())
    if _SASS_PATTERN.fullmatch(p):
        if mma and class_of_opcode(_tokens(p)[0]) != "mma":
            raise TilefishDeclarationError(f"{where}: {p!r} is not an MMA opcode of SASS (HMMA, HGMMA, UTCHMMA, ...)")
        return p
    if mma and _PTX_PATTERN.fullmatch(p) and p.startswith(_PTX_MMA_ROOTS):
        return p
    if mma:
        raise TilefishDeclarationError(
            f"{where}: an MMA is a PTX pattern (lower case: 'tcgen05.mma kind::f16', 'wgmma.mma_async bf16', 'mma.sync m16n8k16') or a SASS "
            f"opcode pattern (upper case: 'HMMA.16816.F32.BF16', 'HGMMA.64x128x16.F32.BF16'), got {pattern!r}"
        )
    if p in CLASSES:
        return p
    raise TilefishDeclarationError(
        f"{where}: forbid= takes instruction classes ({', '.join(CLASSES)}) or SASS opcode patterns (upper case: 'LDL', 'ATOMG', "
        f"'HMMA.1688.F32.TF32'), got {pattern!r}"
    )


@dataclass(frozen=True)
class SassChecks:
    """The SASS assertions of a declaration (:func:`sass`). ``None`` is "not said" (a ``by_arch`` entry then keeps the op's)."""

    no_tf32: "bool | None" = None
    mma: "tuple[str, ...] | None" = None
    no_spills: "bool | None" = None
    max_registers: "int | None" = None
    forbid: "tuple[str, ...] | None" = None
    by_arch: "tuple[tuple[str, SassChecks], ...]" = ()

    def for_arch(self, arch: "int | str") -> "SassChecks":
        """The assertions on ``arch`` (``103`` or ``"sm_103"``): the op's, with every field its ``by_arch`` entry for the architecture says
        replaced."""
        name = arch if isinstance(arch, str) and arch.startswith("sm_") else f"sm_{int(arch)}"
        own = dict(self.by_arch).get(name)
        base = SassChecks(self.no_tf32, self.mma, self.no_spills, self.max_registers, self.forbid)
        if own is None:
            return base
        return SassChecks(
            own.no_tf32 if own.no_tf32 is not None else base.no_tf32,
            own.mma if own.mma is not None else base.mma,
            own.no_spills if own.no_spills is not None else base.no_spills,
            own.max_registers if own.max_registers is not None else base.max_registers,
            own.forbid if own.forbid is not None else base.forbid,
        )

    @property
    def empty(self) -> bool:
        return not self.canonical()

    def canonical(self) -> "dict[str, Any]":
        """The assertions as plain data (what the declaration digest covers): only the fields said."""
        out: "dict[str, Any]" = {}
        for key in ("no_tf32", "no_spills", "max_registers"):
            v = getattr(self, key)
            if v is not None:
                out[key] = v
        for key in ("mma", "forbid"):
            v = getattr(self, key)
            if v is not None:
                out[key] = list(v)
        if self.by_arch:
            out["by_arch"] = {k: v.canonical() for k, v in self.by_arch}
        return out

    def describe(self) -> str:
        return json.dumps(self.canonical(), sort_keys=True)

    def needs_sass(self, has_ptx: bool) -> bool:
        """Whether judging a binary needs its SASS (an ``nvdisasm``), not only its PTX and its ELF file (``no_spills`` and ``max_registers`` are
        the ELF file's)."""
        return bool(self.forbid or any(_SASS_PATTERN.fullmatch(p) for p in self.mma or ()) or (self.no_tf32 and not has_ptx))


def sass(
    *,
    no_tf32: "bool | None" = None,
    mma: "str | Sequence[str] | None" = None,
    no_spills: "bool | None" = None,
    max_registers: "int | None" = None,
    forbid: "str | Sequence[str] | None" = None,
    by_arch: "Mapping[str, SassChecks] | None" = None,
) -> SassChecks:
    """``tilefish.sass(...)``: assertions about the SASS of every binary of a declaration, checked at every build (a cache hit included) from
    its SASS, PTX and ELF file; a binary that violates one is refused with the instructions and source lines that do.

    * ``no_tf32=True``: no TF32 tensor-core MMA (a ``kind::tf32`` or ``.tf32`` MMA of the PTX; a ``.TF32`` SASS MMA; a ``UTCHMMA`` whose
      descriptor says ``kind::tf32``).
    * ``mma="tcgen05.mma kind::f16"`` (or several): each pattern is used by at least one instruction. Lower case is a PTX pattern, its tokens
      among the opcode's in order (``"tcgen05.mma kind::f16"``, ``"wgmma.mma_async bf16"``, ``"mma.sync m16n8k16 bf16"``); upper case a SASS
      opcode pattern (``"HMMA.16816.F32.BF16"``, ``"HGMMA.64x128x16.F32"``, ``"UTCHMMA"``). On sm_100 and later the SASS of a tcgen05 MMA does not
      spell its kind: say it in PTX.
    * ``no_spills=True``: no local memory: the ELF file of the binary records no stack frame (a spill, or a local array, whichever instruction
      reaches it), and no ``LDL``/``STL`` where the SASS is read.
    * ``max_registers=128``: the registers per thread the ELF file records are at most that.
    * ``forbid=("atomic", "fp64", "ATOMG")``: no instruction of these classes (:data:`CLASSES`) or opcode patterns.
    * ``by_arch={"sm_120": tilefish.sass(mma="HMMA.16816.F32.BF16")}``: per architecture, every field it says replaces the op's (``no_spills=False``
      there switches it off).

    The assertions are in the declaration digest (only when given), so a binary built under other assertions is not the binary of this
    declaration."""
    where = "tilefish.sass"
    for key, v in (("no_tf32", no_tf32), ("no_spills", no_spills)):
        if v is not None and not isinstance(v, bool):
            raise TilefishDeclarationError(f"{where}: {key} is True, False or None, got {v!r}")
    if max_registers is not None and (isinstance(max_registers, bool) or not isinstance(max_registers, int) or not 1 <= max_registers <= MAX_REGISTERS):
        raise TilefishDeclarationError(f"{where}: max_registers is an int from 1 to {MAX_REGISTERS} (registers per thread), got {max_registers!r}")

    def patterns(key: str, value: Any, is_mma: bool) -> "tuple[str, ...] | None":
        if value is None:
            return None
        items = [value] if isinstance(value, str) else value
        if not isinstance(items, (list, tuple)):
            raise TilefishDeclarationError(f"{where}: {key} is a string or a list of strings, got {value!r}")
        out = tuple(_check_pattern(f"{where}({key}=...)", p, mma=is_mma) for p in items)
        return tuple(dict.fromkeys(out))

    arch_entries: "list[tuple[str, SassChecks]]" = []
    if by_arch is not None:
        if not isinstance(by_arch, Mapping):
            raise TilefishDeclarationError(f"{where}: by_arch maps architectures ('sm_120') to tilefish.sass(...), got {by_arch!r}")
        for k, v in by_arch.items():
            if not isinstance(k, str) or not _ARCH_KEY.fullmatch(k):
                raise TilefishDeclarationError(f"{where}: by_arch keys are architectures such as 'sm_120', got {k!r}")
            if not isinstance(v, SassChecks):
                raise TilefishDeclarationError(f"{where}: by_arch[{k!r}] is a tilefish.sass(...), got {v!r}")
            if v.by_arch:
                raise TilefishDeclarationError(f"{where}: by_arch[{k!r}] has a by_arch of its own; an architecture's entry says its fields only")
            arch_entries.append((k, v))
    return SassChecks(
        no_tf32=no_tf32,
        mma=patterns("mma", mma, True),
        no_spills=no_spills,
        max_registers=max_registers,
        forbid=patterns("forbid", forbid, False),
        by_arch=tuple(sorted(arch_entries)),
    )


def checks_or_none(value: Any, what: str) -> "SassChecks | None":
    """A declaration's ``sass=``: a :class:`SassChecks` (``None`` when it says nothing), or a :class:`~tilefish.errors.TilefishDeclarationError`."""
    if value is None:
        return None
    if not isinstance(value, SassChecks):
        raise TilefishDeclarationError(f"{what}: sass= is a tilefish.sass(...) or None, got {value!r}")
    return None if value.empty else value


@dataclass(frozen=True)
class Violation:
    """One assertion a binary violates: its name (``no_spills``), what is wrong, and the instructions (or PTX lines) that show it."""

    check: str
    message: str
    where: "tuple[str, ...]" = ()

    def describe(self, limit: int = 6) -> str:
        shown = "; ".join(self.where[:limit]) + (f"; ... ({len(self.where) - limit} more)" if len(self.where) > limit else "")
        return f"{self.check}: {self.message}" + (f": {shown}" if shown else "")


def _ptx_where(ptx: str, lines: "Iterable[tuple[int, str]]") -> "tuple[str, ...]":
    locs = ptx_locations(ptx)
    out = []
    for n, op in lines:
        loc = locs.get(n)
        out.append(f"PTX line {n}: {op}" + (f" ({loc[0]}:{loc[1]})" if loc else ""))
    return tuple(out)


def evaluate(
    checks: SassChecks,
    *,
    ptx: "str | None",
    attributes: "Mapping[str, Any]",
    instructions: "Callable[[], Sequence[Instruction]] | None",
) -> "list[Violation]":
    """Judge one binary against the assertions of its architecture (``checks``, after :meth:`SassChecks.for_arch`): its PTX (``None`` for a
    prebuilt cubin), its ELF attributes (:func:`kernel_attributes`) and, when the assertions need them (:meth:`SassChecks.needs_sass`), the SASS
    instructions of its code (``instructions``, called once; a failure to read them is raised, except where it only names the instructions of a
    violation the ELF file decided). Pure; returns every violation."""
    out: "list[Violation]" = []
    sass_ins: "Sequence[Instruction] | None" = None

    def get() -> "Sequence[Instruction]":
        nonlocal sass_ins
        if sass_ins is None:
            if instructions is None:
                raise TilefishBuildError("the SASS assertions need the SASS of the binary, and no listing was given")
            sass_ins = list(instructions())
        return sass_ins

    if checks.max_registers is not None:
        regs = attributes.get("registers")
        if regs is None:
            out.append(Violation("max_registers", f"the ELF file does not record the registers of the function, so at most {checks.max_registers} cannot be checked"))
        elif regs > checks.max_registers:
            out.append(Violation("max_registers", f"the binary uses {regs} registers per thread, more than the {checks.max_registers} declared"))
    if checks.no_tf32:
        in_ptx = _ptx_where(ptx, [(n, op) for n, op in ptx_instructions(ptx) if is_tf32_ptx(op)]) if ptx is not None else ()
        in_sass: "tuple[str, ...]" = ()
        undecided: "tuple[str, ...]" = ()
        if ptx is None or checks.needs_sass(True):  # the PTX spells every kind; the SASS is read when it is read anyway, or is all there is
            ins = get()
            descriptors = mma_descriptors(ins)
            in_sass = tuple(i.describe() for i in ins if is_tf32_mma(i, descriptors.get(i.address)))
            if ptx is None:  # a UTCHMMA is kind::f16 or kind::tf32, and only its descriptor says which
                undecided = tuple(i.describe() for i in ins if i.opcode == "UTCHMMA" and i.address not in descriptors)
        if in_ptx or in_sass:
            counts = [f"{len(x)} in its {w}" for x, w in ((in_ptx, "PTX"), (in_sass, "SASS")) if x]
            out.append(Violation("no_tf32", f"TF32 tensor-core MMAs, which keep 10 of the 23 bits of each f32 mantissa ({', '.join(counts)})", in_ptx + in_sass))
        if undecided:
            out.append(
                Violation(
                    "no_tf32",
                    f"{len(undecided)} UTCHMMA (kind::f16 or kind::tf32) whose instruction descriptor the SASS does not set as a constant, in a binary "
                    "without PTX: whether it multiplies TF32 cannot be told, so it is not taken to be exact (declare the binary as PTX)",
                    undecided,
                )
            )
    if checks.mma:
        for p in checks.mma:
            if _SASS_PATTERN.fullmatch(p):
                ins = get()
                if not any(class_of(i) == "mma" and sass_pattern_matches(p, i) for i in ins):
                    have = sorted({i.mnemonic for i in ins if class_of(i) == "mma"})
                    out.append(Violation("mma", f"it declares the SASS MMA {p!r}, and the binary has none: its SASS MMAs are {have or 'none'}"))
            else:
                ops = [op for _, op in ptx_instructions(ptx)] if ptx is not None else []
                if ptx is None:
                    out.append(Violation("mma", f"it declares the PTX MMA {p!r}, and the binary has no PTX (a prebuilt cubin): say it as a SASS opcode"))
                elif not any(ptx_pattern_matches(p, op) for op in ops):
                    out.append(Violation("mma", f"it declares the PTX MMA {p!r}, and the PTX has none: its MMAs are {sorted(set(ops)) or 'none'}"))
    if checks.no_spills:
        local_bytes = int(attributes.get("local", 0) or 0)
        named: "tuple[str, ...]" = ()
        unread = ""
        if local_bytes or checks.needs_sass(ptx is not None):  # (the ELF file decides; the SASS names the instructions)
            try:
                named = tuple(i.describe() + (" [spill]" if "SpillRefill" in i.notes else "") for i in get() if class_of(i) == "local")
            except TilefishBuildError as e:
                if not local_bytes:
                    raise
                unread = f" (its SASS could not be read to name the instructions: {e})"
        if local_bytes or named:
            what = f"a stack frame of {local_bytes} bytes per thread in its ELF file" if local_bytes else "no stack frame in its ELF file"
            count = f", {len(named)} local-memory instruction{'s' if len(named) != 1 else ''}" if named else ""
            out.append(Violation("no_spills", f"local memory, spills or a local array: {what}{count}{unread}", named))
    for p in checks.forbid or ():
        ins = get()
        hit = [i for i in ins if (class_of(i) == p if p in CLASSES else sass_pattern_matches(p, i))]
        if hit:
            what = f"the class {p!r} ({CLASSES[p]})" if p in CLASSES else f"the opcode {p!r}"
            out.append(Violation("forbid", f"{len(hit)} instruction{'s' if len(hit) != 1 else ''} of {what}", tuple(i.describe() for i in hit)))
    return out


# -- the disassembler -------------------------------------------------------------------------------------------------------------------------

INSTALL = "install a CUDA 13 nvdisasm, which reads sm_103: pip install 'tilefish[sass]' (nvidia-cuda-nvdisasm), or set TILEFISH_NVDISASM_PATH"


@dataclass(frozen=True)
class Disassembly:
    """The SASS of a binary: the ``nvdisasm -c -g -gi`` listing (``text``), its ``-c -gp`` listing (``ptx_text``: the PTX line of each instruction;
    empty when that run failed), both parsed (``listing``), the target they were read for, whether that is a re-assembly for the base of the family
    (``proxy``) and, for one, whether its registers, stack frames and code sizes are the binary's (``matches``; ``mismatch`` says how not), the
    ``nvdisasm`` (its path and version), and the size of the code of each function of what was read (``code_sizes``)."""

    text: str
    ptx_text: str
    listing: Listing
    target: str
    proxy: bool
    nvdisasm: str
    version: str
    binary_target: str = ""
    matches: "bool | None" = None
    mismatch: str = ""
    code_sizes: "Mapping[str, int]" = field(default_factory=dict)

    @property
    def label(self) -> str:
        """How a message or a view says where the SASS comes from."""
        tool = f"nvdisasm {self.version}" if self.version else self.nvdisasm
        if self.proxy:
            reassembly = (
                f"the SASS of its PTX assembled again for {self.target}, the base of its family, with the ptxas and the flags of the binary, read by "
                f"{tool}, which cannot decode {self.binary_target or 'the binary'}"
            )
            if self.matches:
                return f"{reassembly}: judged on a re-assembly whose registers, stack frames and code sizes are the binary's"
            return f"{reassembly}, and which is not the binary ({self.mismatch})"
        return f"its SASS, read by {tool}"


LISTING_FLAGS = ("-c", "-g", "-gi")  # the code, with the source line of each instruction (and the lines it was inlined at, where ptxas has them)
PTX_LINE_FLAGS = ("-c", "-gp")  # the code, with the line of the PTX each instruction was assembled from


@functools.lru_cache(maxsize=None)
def nvdisasm_version(path: str) -> str:
    """``13.5.37`` from ``nvdisasm --version`` (``""`` when it does not say)."""
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    m = re.search(r"V(\d+\.\d+\.\d+)", out)
    return m.group(1) if m else ""


def triton_ptxas_flags(options: "Mapping[str, Any]") -> "list[str]":
    """The flags Triton 3.5.1's ``make_cubin`` gives ``ptxas`` for a kernel compiled with ``options`` (before ``--gpu-name``): the line information
    (``-lineinfo``; ``-lineinfo -suppress-debug-info`` with ``knobs.compilation.disable_line_info``; ``-g`` with ``knobs.nvidia.disable_ptxas_opt``),
    ``--fmad=false`` where the kernel does not contract (``enable_fp_fusion``, else ``knobs.language.default_fp_fusion``), ``-v``, ``--opt-level 0``
    with ``disable_ptxas_opt``, and ``ptx_options``. Without Triton the knobs are their defaults."""
    knobs: Any = None
    try:
        import triton

        knobs = triton.knobs
    except Exception:  # noqa: BLE001 - no Triton (or one without knobs): the defaults
        pass

    def knob(group: str, name: str, default: Any) -> Any:
        try:
            return getattr(getattr(knobs, group), name)
        except Exception:  # noqa: BLE001
            return default

    no_line_info = bool(knob("compilation", "disable_line_info", False))
    no_opt = bool(knob("nvidia", "disable_ptxas_opt", False))
    debug = ["-lineinfo", "-suppress-debug-info"] if no_line_info else ["-g"] if no_opt else ["-lineinfo"]
    fused = options.get("enable_fp_fusion")
    if fused is None:
        fused = bool(knob("language", "default_fp_fusion", True))
    flags = [*debug, *([] if fused else ["--fmad=false"]), "-v", *(["--opt-level", "0"] if no_opt else [])]
    extra = options.get("ptx_options")
    if extra:
        flags += str(extra).split()
    return flags


def proxy_flags(options: "Mapping[str, Any]") -> "list[str]":
    """The ``ptxas`` flags a Triton kernel compiled with ``options`` was assembled with (:func:`triton_ptxas_flags`)."""
    return triton_ptxas_flags(options)


def _assembler(prebuilt: bool, options: "Mapping[str, Any]") -> "tuple[str, list[str]]":
    """The ``ptxas`` and the flags the binary was assembled with: a prebuilt PTX's (:func:`tilefish._binary.find_ptxas`, ``PTX_FLAGS``) or Triton's."""
    if prebuilt:
        from ._binary import PTX_FLAGS, find_ptxas

        return find_ptxas(), list(PTX_FLAGS)
    from . import _fpcontract

    return _fpcontract.ptxas_path(), triton_ptxas_flags(options)


def _assemble(ptxas: str, ptx: str, target: str, flags: "Sequence[str]") -> bytes:
    """``ptx`` assembled by ``ptxas`` for ``target`` with ``flags``, in an empty directory (ptxas would record a source file it found there)."""
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory(prefix="tilefish-sass-") as d:
        src, out, cwd = pathlib.Path(d) / "k.ptx", pathlib.Path(d) / "k.cubin", pathlib.Path(d) / "cwd"
        cwd.mkdir()
        src.write_text(ptx)
        r = subprocess.run([ptxas, *flags, f"--gpu-name={target}", str(src), "-o", str(out)], capture_output=True, text=True, cwd=cwd)
        if r.returncode != 0:
            raise TilefishBuildError(f"ptxas could not assemble the PTX for {target}: {r.stderr.strip()[:2000]}")
        return out.read_bytes()


_cache_lock = threading.Lock()
_disassemblies: "dict[tuple[Any, ...], Disassembly]" = {}


def _target_of(cubin: bytes, ptx: "str | None") -> str:
    """The target the cubin is code for: the PTX's ``.target`` when it names the cubin's architecture (``sm_103a``), else the architecture of the
    cubin's ELF header (``sm_103`` for a prebuilt PTX of an older generic target assembled for it)."""
    from ._elf import parse_elf, sm_of

    sm = sm_of(parse_elf(cubin))
    m = _TARGET.search(ptx) if ptx else None
    if m is not None and (sm is None or re.fullmatch(rf"sm_{sm}[af]?", m.group(1))):
        return m.group(1)
    return f"sm_{sm}" if sm is not None else "sm_unknown"


# the base of a family, whose instruction set an architecture shares: the target a PTX is assembled for where no nvdisasm decodes its own
_PROXY_BASE = {"sm_103a": "sm_100a", "sm_103f": "sm_100f", "sm_103": "sm_100", "sm_121a": "sm_120a", "sm_121f": "sm_120f", "sm_121": "sm_120"}
_TARGET_LINE = re.compile(r"^(\s*\.target\s+)(sm_[0-9]+[a-z]?)", re.M)


def retarget(ptx: str, base: str) -> str:
    """``ptx`` to assemble for the family base ``base``: a ``.target`` of the family's member (``sm_103a``, ``sm_103f``, ``sm_103``) becomes the base's,
    with the same suffix; an older target (a prebuilt PTX of ``.target sm_80``) is left as it is, ptxas assembles it for any newer ``--gpu-name``.
    Nothing else changes (a member's target needs a PTX ISA that has the base's already)."""
    member = {v: k for k, v in _PROXY_BASE.items()}.get(base)

    def one(t: "re.Match[str]") -> str:
        return t.group(1) + base if t.group(2) == member else t.group(0)

    return _TARGET_LINE.sub(one, ptx, count=1)


def _shapes_differ(real: "Mapping[str, tuple[int, int, int]]", again: "Mapping[str, tuple[int, int, int]]") -> str:
    """How the code of a re-assembly is not the binary's (``""``: the same functions, each with the same registers, stack frame and code size)."""
    if set(real) != set(again):
        return f"its functions are {sorted(again)}, the binary's {sorted(real)}"
    out = []
    for name in sorted(real):
        if real[name] != again[name]:
            (r, f, n), (r2, f2, n2) = real[name], again[name]
            out.append(f"{name}: {r2} registers, a {f2}-byte stack frame and {n2} bytes of code where the binary has {r}, {f} and {n}")
    return "; ".join(out)


def disassemble(cubin: bytes, *, ptx: "str | None" = None, options: "Mapping[str, Any] | None" = None, prebuilt: bool = False) -> Disassembly:
    """The SASS of ``cubin`` with line information (``nvdisasm -c -g -gi`` and ``-c -gp``), by the first ``nvdisasm`` that decodes its architecture
    (:func:`tilefish._fpcontract.nvdisasm_candidates`: a ``tilefish[sass]`` wheel's CUDA 13 first). Where none does, ``ptx`` (the binary's PTX) is
    assembled for the base of the family (``sm_100a`` for ``sm_103a``) by the ``ptxas`` and with the flags the binary was assembled with (Triton's
    for ``options``, or a prebuilt PTX's), that is read (``proxy``), and its registers, stack frames and code sizes are compared with the binary's
    (``matches``). Kept per cubin, assembler and disassembler in this process. Raises :class:`~tilefish.errors.TilefishBuildError` when nothing
    can read it."""
    from . import _fpcontract

    candidates = tuple(_fpcontract.nvdisasm_candidates())
    key = (hashlib.sha256(cubin).hexdigest(), candidates, prebuilt, json.dumps(dict(options or {}), sort_keys=True, default=str), hashlib.sha256((ptx or "").encode()).hexdigest())
    with _cache_lock:
        found = _disassemblies.get(key)
    if found is not None:
        return found
    target = _target_of(cubin, ptx)
    nvdisasm = _fpcontract.disassembler_for(target, cubin)
    binary, read_for, proxy, matches, mismatch = cubin, target, False, None, ""
    if nvdisasm is None:
        base = _PROXY_BASE.get(target)
        if base is None:
            raise TilefishBuildError(f"no nvdisasm here decodes {target} ({list(candidates) or 'none found'}); {INSTALL}")
        if not ptx:
            raise TilefishBuildError(f"no nvdisasm here decodes {target}, and the binary has no PTX to assemble for {base}, the base of its family; {INSTALL}")
        ptxas, flags = _assembler(prebuilt, options or {})
        binary = _assemble(ptxas, retarget(ptx, base), base, flags)
        nvdisasm = _fpcontract.disassembler_for(base, binary)
        if nvdisasm is None:
            raise TilefishBuildError(f"no nvdisasm here decodes {target} or {base} ({list(candidates) or 'none found'}); {INSTALL}")
        read_for, proxy = base, True
        mismatch = _shapes_differ(code_shape(cubin), code_shape(binary))
        matches = not mismatch
    text = _fpcontract._disassemble(nvdisasm, binary, LISTING_FLAGS)
    if text is None:
        raise TilefishBuildError(f"{nvdisasm} could not disassemble the binary for {read_for}")
    ptx_text = _fpcontract._disassemble(nvdisasm, binary, PTX_LINE_FLAGS) or ""  # (the PTX lines are a help: without them the SASS is the same)
    sizes = {name: size for name, (_, _, size) in code_shape(binary).items()}
    out = Disassembly(
        text, ptx_text, parse_listing(text, ptx_text or None), read_for, proxy, nvdisasm, nvdisasm_version(nvdisasm), target, matches, mismatch, sizes
    )
    with _cache_lock:
        _disassemblies[key] = out
    return out


def check_complete(d: Disassembly, symbol: str) -> "list[str]":
    """The functions of ``symbol``'s code in a listing (:func:`reached`), each with exactly as many instructions as its code has 16-byte slots.
    Raises :class:`~tilefish.errors.TilefishBuildError` otherwise: the listing is not what this parser reads (another nvdisasm format, an opcode
    it does not know, a section that did not parse), and nothing is judged on it."""
    names = reached(d.listing, symbol)
    if not names:
        raise TilefishBuildError(
            f"the listing of the binary ({d.label}) has no function '{symbol}' (it has {sorted(d.listing.functions)}): nothing can be judged on it"
        )
    for name in names:
        count, size = len(d.listing.functions[name]), d.code_sizes.get(name)
        if size is None or count == 0 or count != size // 16:
            raise TilefishBuildError(
                f"the listing of '{name}' ({d.label}) has {count} instructions where its code ({size} bytes) has {None if size is None else size // 16}: "
                "the nvdisasm output is not what tilefish reads, so nothing is judged on it (please report the nvdisasm version)"
            )
    return names


@dataclass
class Judged:
    """What :func:`judge` found: the violations, and the label of what it read: where the SASS came from, or ``""`` when it needed none."""

    violations: "list[Violation]" = field(default_factory=list)
    label: str = ""


def judge(
    checks: SassChecks, *, cubin: bytes, ptx: "str | None", symbol: str, options: "Mapping[str, Any] | None" = None, prebuilt: bool = False
) -> Judged:
    """Judge a binary (its cubin, its PTX or ``None``, its entry ``symbol``) against ``checks`` (already for its architecture). The PTX checks read
    the entry and the functions it calls (:func:`ptx_for_entry`), the SASS checks the code of the entry and of the functions it calls, every
    instruction of it (:func:`check_complete`). Disassembles only when the assertions need the SASS (:meth:`SassChecks.needs_sass`), or to name the
    instructions of a violation; a re-assembly for the family base that is not the binary (:class:`Disassembly`) refuses the build."""
    attributes = kernel_attributes(cubin, symbol)
    own_ptx = ptx_for_entry(ptx, symbol) if ptx else None
    result = Judged()

    def instructions() -> "Sequence[Instruction]":
        d = disassemble(cubin, ptx=ptx or None, options=options, prebuilt=prebuilt)
        if d.proxy and not d.matches:
            raise TilefishBuildError(f"its SASS cannot be judged: {d.label}. {INSTALL[0].upper() + INSTALL[1:]}")
        check_complete(d, symbol)
        result.label = d.label
        return kernel_instructions(d.listing, symbol)

    result.violations = evaluate(checks, ptx=own_ptx, attributes=attributes, instructions=instructions)
    return result
