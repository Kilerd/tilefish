"""A strict ELF64 section walk, and the digest of the machine code in a cubin.

A cubin is an ELF file. Besides the machine code it carries debug sections (``.debug_line``,
``.nv_debug_ptx_txt``, ...) that hold the path of the Python source the kernel was compiled from, so two
builds of the same kernel from different directories differ byte for byte while running the same code.
:func:`code_digest` is a sha256 that ignores exactly that: it covers the identity fields of the ELF
header and every section that is not debug information (name, type, flags, sizes, link and contents),
and skips ``.debug_*`` and ``.nv_debug_*`` together with their wrapped variants (``.rela.debug_line``, the
mercury copies ``.nv.merc.debug_line`` and ``.nv.merc.rela.debug_line``, ...). In a cubin built by Triton
from two directories exactly the debug sections differ: ``.debug_line``, ``.nv_debug_ptx_txt``,
``.rela.debug_line`` and their ``.nv.merc.*`` copies, all of which hold the path of the source.

The walk is strict: anything that is not a well-formed little-endian ELF64 file with a section header
table whose sections lie inside the file is a :class:`~tilefish.errors.TilefishBuildError`. Nothing is
guessed, and file offsets (which move when a debug section changes size) never enter the digest.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

from .errors import TilefishBuildError

CODE_DIGEST_FORMAT = 1  # bump when the canonical byte stream below changes: every digest changes with it

SHT_NULL, SHT_SYMTAB, SHT_STRTAB, SHT_NOBITS, SHT_DYNSYM = 0, 2, 3, 8, 11
SHF_INFO_LINK = 0x40
SHN_XINDEX = 0xFFFF

_EHDR = struct.Struct("<16sHHIQQQIHHHHHH")  # ident, type, machine, version, entry, phoff, shoff, flags, ehsize, phentsize, phnum, shentsize, shnum, shstrndx
_SHDR = struct.Struct("<IIQQQQIIQQ")  # name, type, flags, addr, offset, size, link, info, addralign, entsize
EHDR_SIZE = _EHDR.size  # 64
SHDR_SIZE = _SHDR.size  # 64
assert EHDR_SIZE == 64 and SHDR_SIZE == 64

# prefixes that wrap a section name: relocations of it, and the mercury (sm_100 and later) copy of it
_WRAPPERS = (".nv.merc.", ".rela.", ".rel.")
_DEBUG_PREFIXES = (".debug_", ".zdebug_", ".nv_debug_", ".nv.debug_")


def is_debug_section(name: str) -> bool:
    """Whether a section holds debug information: ``.debug_*``, ``.nv_debug_*`` or a wrapped variant of one.

    ``.rela.debug_line``, ``.nv.merc.debug_line`` and ``.nv.merc.rela.nv_debug_line_sass`` are debug sections;
    ``.rela.text.k``, ``.nv.merc.nv.info`` and ``.text.debug_k`` are not.
    """
    base = name
    stripped = True
    while stripped:
        stripped = False
        for prefix in _WRAPPERS:
            if base.startswith(prefix):
                base = "." + base[len(prefix) :]
                stripped = True
    return base.startswith(_DEBUG_PREFIXES)


@dataclass(frozen=True)
class Section:
    index: int
    name: str
    type: int
    flags: int
    addr: int
    offset: int
    size: int
    link: int
    info: int
    addralign: int
    entsize: int


@dataclass(frozen=True)
class Elf:
    """The parts of an ELF64 file the digest needs: the identity fields, the sections and their contents."""

    ident: bytes  # the 16 bytes of e_ident: class, data, version, OS ABI, ABI version
    type: int
    machine: int
    version: int
    flags: int
    sections: "tuple[Section, ...]"
    blob: bytes
    shstrndx: int = 0  # the index of the section name table

    def data(self, section: Section) -> bytes:
        """The contents of a section (empty for a ``NOBITS`` section, which has none in the file)."""
        if section.type == SHT_NOBITS:
            return b""
        return self.blob[section.offset : section.offset + section.size]

    def section(self, name: str) -> "Section | None":
        return next((s for s in self.sections if s.name == name), None)


def _bad(message: str) -> Any:
    raise TilefishBuildError(f"malformed ELF file: {message}")


def parse_elf(blob: "bytes | bytearray | memoryview") -> Elf:
    """Walk the section header table of an ELF64 little-endian file, checking every bound on the way."""
    blob = bytes(blob)
    size = len(blob)
    if size < EHDR_SIZE:
        _bad(f"{size} bytes is shorter than the {EHDR_SIZE}-byte ELF header")
    ident, e_type, machine, version, _entry, _phoff, shoff, flags, ehsize, _phentsize, _phnum, shentsize, shnum, shstrndx = (
        _EHDR.unpack_from(blob, 0)
    )
    if ident[:4] != b"\x7fELF":
        _bad(f"bad magic {ident[:4]!r}")
    if ident[4] != 2:
        _bad(f"EI_CLASS is {ident[4]}, not 2 (ELFCLASS64)")
    if ident[5] != 1:
        _bad(f"EI_DATA is {ident[5]}, not 1 (little endian)")
    if ident[6] != 1:
        _bad(f"EI_VERSION is {ident[6]}, not 1")
    if ehsize != EHDR_SIZE:
        _bad(f"e_ehsize is {ehsize}, not {EHDR_SIZE}")
    if shentsize != SHDR_SIZE:
        _bad(f"e_shentsize is {shentsize}, not {SHDR_SIZE}")
    if shnum == 0:
        _bad("it has no section header table (e_shnum is 0)")
    if shstrndx == SHN_XINDEX:
        _bad("extended section numbering (e_shstrndx = SHN_XINDEX) is not supported")
    if not 0 <= shstrndx < shnum:
        _bad(f"e_shstrndx {shstrndx} is outside the {shnum} sections")
    if shoff < EHDR_SIZE or shoff + shnum * SHDR_SIZE > size:
        _bad(f"the section header table (offset {shoff}, {shnum} entries) does not fit in {size} bytes")

    raw = [_SHDR.unpack_from(blob, shoff + i * SHDR_SIZE) for i in range(shnum)]
    for i, (_n, typ, _f, _a, offset, sz, _l, _i, _al, _e) in enumerate(raw):
        if typ == SHT_NOBITS:
            continue
        if sz and offset < EHDR_SIZE:
            _bad(f"section {i} starts at offset {offset}, inside the ELF header")
        if offset + sz > size:
            _bad(f"section {i} (offset {offset}, size {sz}) extends past the end of the {size}-byte file")
    if raw[0][1] != SHT_NULL:
        _bad(f"section 0 has type {raw[0][1]}, not 0 (SHT_NULL)")
    strtab = raw[shstrndx]
    if strtab[1] != SHT_STRTAB:
        _bad(f"the section name table (section {shstrndx}) has type {strtab[1]}, not 3 (SHT_STRTAB)")
    names = blob[strtab[4] : strtab[4] + strtab[5]]

    sections: list[Section] = []
    for i, (name_off, typ, sh_flags, addr, offset, sz, link, info, addralign, entsize) in enumerate(raw):
        if name_off >= len(names):
            _bad(f"section {i}: the name offset {name_off} is outside the {len(names)}-byte name table")
        end = names.find(b"\0", name_off)
        if end < 0:
            _bad(f"section {i}: its name is not NUL terminated")
        try:
            name = names[name_off:end].decode("utf-8")
        except UnicodeDecodeError:
            _bad(f"section {i}: its name is not valid UTF-8")
        sections.append(Section(i, name, typ, sh_flags, addr, offset, sz, link, info, addralign, entsize))
    return Elf(bytes(ident), e_type, machine, version, flags, tuple(sections), blob, shstrndx)


def sm_of(elf: Elf) -> "int | None":
    """The SM version a cubin is for (``103`` for sm_103), from its ELF header, or ``None`` when the header does not say it.

    The Rust reader's ``Elf::sm`` (``crates/tilefish/src/elf.rs``) and :func:`tilefish._binary.cubin_sm`, which refuses where this
    gives ``None``: checked on cubins that ptxas 12.9 made for sm_80 to sm_120, a cubin whose ``EI_OSABI`` is 0x33 (up to sm_90) has the
    version in the low byte of ``e_flags`` (80, 86, 89, 90; bit 11 is the ``a`` suffix), one whose ``EI_OSABI`` is 0x41 (sm_100 and
    later, the ``a``/``f`` suffix in the low byte) has it in the second byte (100, 103, 120). Another OS ABI says no version.
    """
    osabi = elf.ident[7]
    if osabi == 0x33:
        return elf.flags & 0xFF
    if osabi == 0x41:
        return (elf.flags >> 8) & 0xFF
    return None


def uses_shared(elf: Elf, symbol: str) -> bool:
    """Whether the kernel ``symbol`` uses shared memory: it has a ``.nv.shared.<symbol>`` section.

    Triton gives a kernel that needs shared memory dynamic shared memory (an ``extern`` array), and ptxas then
    emits that section (a ``NOBITS`` section of 1024 bytes for sm_90 and later, of size 0 for sm_80: its size says
    nothing about how much the code addresses); a kernel that needs none has no such section. Checked on
    sm_80, sm_90, sm_100, sm_103 and sm_120 for a softmax (shared) and a gelu (not). The ELF does not say how much
    dynamic shared memory the code addresses: that is the ``shared`` the build recorded, which a launch has to pass.
    """
    return elf.section(f".nv.shared.{symbol}") is not None


def shared_sizes(elf: Elf, symbol: str) -> "tuple[int, int]":
    """``(declared, reserved)``: the bytes of static shared memory the kernel ``symbol`` declares, and the bytes the driver reserves.

    They are the sizes of the ``NOBITS`` sections ``.nv.shared.<symbol>`` and ``.nv.shared.reserved.0`` (0 when
    absent). A fact about the file, not what the driver enforces: a launch is checked against the driver's own figure
    (``CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES``, see :mod:`tilefish._cuda`), which is smaller. Measured on an sm_103 device: a
    Triton kernel with a 1024-byte ``.nv.shared.<symbol>`` section reports 0 static bytes (the section is the per-block system
    reserve the device's opt-in maximum already excludes), and a hand-written one with 4096 static bytes has a 5120-byte section
    and reports 4096.
    """
    declared = elf.section(f".nv.shared.{symbol}")
    reserved = elf.section(".nv.shared.reserved.0")
    return (declared.size if declared is not None else 0, reserved.size if reserved is not None else 0)


def _reference(elf: Elf, value: int, *, resolve: bool) -> bytes:
    """A link or info field as the section it names (so that moving debug sections does not change it), else as a number."""
    if resolve and 0 < value < len(elf.sections):
        return b"S:" + elf.sections[value].name.encode("utf-8")
    return b"N:" + str(value).encode("ascii")


def code_digest(cubin: "bytes | bytearray | memoryview") -> str:
    """The sha256 (lowercase hex) of the machine code of a cubin: everything but build-path debug information.

    The digest covers, as one canonical byte stream, the ELF identity (``e_ident``, type, machine, version
    and flags, which carry the target architecture) and, for every section after the null section that is
    not a debug section (see :func:`is_debug_section`), its name, type, flags, address, size, alignment,
    entry size, its ``sh_link`` and ``sh_info`` (as section names where they name a section) and its
    contents (nothing for ``NOBITS``). File offsets are not covered, and neither is the section name table
    itself (``.shstrtab``): it only spells the names, which are hashed section by section, and it also
    spells the names of the debug sections. It is the same for two builds of one
    kernel with the same options from different directories, and different when any instruction, constant,
    size or symbol changes.

    The section order and indices are part of the layout, so a build with different line-info settings
    (which adds or removes debug sections and shifts the indices in the symbol table) has another digest.
    So does a build with other ptxas options: ``.note.nv.tkinfo`` is not a debug section, and it records the
    ptxas version and command line (``-arch sm_103a -fmad false``), so two cubins with the same instructions
    but built with different flags are told apart.
    """
    elf = parse_elf(cubin)
    h = hashlib.sha256()

    def put(b: bytes) -> None:
        h.update(struct.pack("<Q", len(b)))
        h.update(b)

    h.update(f"tilefish.cubin-code.v{CODE_DIGEST_FORMAT}\0".encode("ascii"))
    put(elf.ident)
    h.update(struct.pack("<HHII", elf.type, elf.machine, elf.version, elf.flags))
    for s in elf.sections[1:]:
        if is_debug_section(s.name) or s.index == elf.shstrndx:
            continue
        put(s.name.encode("utf-8"))
        h.update(struct.pack("<IQQQQQ", s.type, s.flags, s.addr, s.size, s.addralign, s.entsize))
        put(_reference(elf, s.link, resolve=True))
        put(_reference(elf, s.info, resolve=bool(s.flags & SHF_INFO_LINK) and s.type not in (SHT_SYMTAB, SHT_DYNSYM)))
        put(elf.data(s))
    return h.hexdigest()
