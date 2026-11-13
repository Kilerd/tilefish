"""Builds small ELF64 files for the tests (not collected: the name has no ``test_`` prefix).

The layout is the one a cubin has: header, section contents, the section name table, then the section
header table. A section's ``link`` and ``info`` can name another section, so a test can reorder sections
and still have the references point where they did.
"""

import struct
from dataclasses import dataclass
from typing import Union

SHT_PROGBITS, SHT_SYMTAB, SHT_STRTAB, SHT_NOBITS = 1, 2, 3, 8
SHF_ALLOC, SHF_EXECINSTR, SHF_INFO_LINK = 0x2, 0x4, 0x40
EM_CUDA = 190


@dataclass
class S:
    name: str
    type: int = SHT_PROGBITS
    flags: int = 0
    data: bytes = b""
    link: Union[int, str] = 0  # a section index, or the name of the section it refers to
    info: Union[int, str] = 0
    align: int = 1
    entsize: int = 0
    size: "int | None" = None  # NOBITS sections have a size and no contents


def build_elf(
    sections,
    *,
    osabi=0x41,
    abiversion=8,
    machine=EM_CUDA,
    flags=0x0900670A,
    gap=0,
    shstrndx=None,
    shentsize=64,
    ehsize=64,
    ei_class=2,
    ei_data=1,
):
    """An ELF64 file with a null section, ``sections`` in order, and a ``.shstrtab`` at the end.

    ``gap`` inserts that many zero bytes before every section's contents, so file offsets move without
    any section changing.
    """
    sections = list(sections)
    index = {s.name: i + 1 for i, s in enumerate(sections)}
    shstr_index = len(sections) + 1
    index[".shstrtab"] = shstr_index
    names = b"\0"
    name_off = {}
    for s in [*sections, S(".shstrtab", SHT_STRTAB)]:
        name_off[s.name] = len(names)
        names += s.name.encode() + b"\0"

    body = b""
    offsets = {}
    for s in sections:
        body += b"\0" * gap
        offsets[s.name] = 64 + len(body)
        if s.type != SHT_NOBITS:
            body += s.data
    body += b"\0" * gap
    offsets[".shstrtab"] = 64 + len(body)
    body += names

    def ref(v):
        return index[v] if isinstance(v, str) else v

    headers = struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # the null section
    for s in sections:
        size = len(s.data) if s.size is None else s.size
        off = 0 if s.type == SHT_NOBITS else offsets[s.name]
        headers += struct.pack(
            "<IIQQQQIIQQ", name_off[s.name], s.type, s.flags, 0, off, size, ref(s.link), ref(s.info), s.align, s.entsize
        )
    headers += struct.pack(
        "<IIQQQQIIQQ", name_off[".shstrtab"], SHT_STRTAB, 0, 0, offsets[".shstrtab"], len(names), 0, 0, 1, 0
    )
    shoff = 64 + len(body)
    ident = bytes([0x7F, 0x45, 0x4C, 0x46, ei_class, ei_data, 1, osabi, abiversion]) + b"\0" * 7
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        ident,
        2,  # ET_EXEC
        machine,
        1,
        0,
        0,
        shoff,
        flags,
        ehsize,
        0,
        0,
        shentsize,
        len(sections) + 2,
        shstr_index if shstrndx is None else shstrndx,
    )
    return header + body + headers


def arch_header(arch):
    """The ELF header fields (``osabi``, ``abiversion``, ``flags``) of a cubin ptxas makes for ``arch`` (``"sm_90"``, ``"sm_103"`` or ``103``).

    A manifest reader checks the ``arch`` of an entry against them (``tilefish._elf.sm_of``, the Rust ``Elf::sm``): up to sm_90 the OS ABI is
    0x33 and the SM is the low byte of ``e_flags``, from sm_100 it is 0x41 and the SM is the second byte. ``build_elf`` without them makes an
    sm_103 cubin, so ``cubin_like(**arch_header("sm_103"))`` is ``cubin_like()``, byte for byte.
    """
    try:
        sm = arch if isinstance(arch, int) else int(arch.removeprefix("sm_"))
    except ValueError:
        return {}  # (a name that is no architecture, which a test hands a writer to see it refused: any header)
    if sm < 100:
        return dict(osabi=0x33, abiversion=7, flags=0x00500500 | sm)
    return dict(osabi=0x41, abiversion=8, flags=0x09000000 | (sm << 8) | 0x0A)


def cubin_like(*, debug="path/a.py", code=b"\x01\x02\x03\x04" * 16, gap=0, debug_first=False, shared=1024, symbol="tf_k", **kw):
    """A small file shaped like a Mercury cubin: code and constants, symbol tables, and debug sections that hold ``debug``.

    ``shared`` is the size of the shared memory section ``.nv.shared.tf_k`` of the kernel: 1024 as on sm_90 and later, ``0``
    for the empty section of sm_80, ``None`` for a kernel that uses no shared memory (no section)."""
    dbg = debug.encode()
    debug_sections = [
        S(".debug_line", data=b"line:" + dbg),
        S(".nv_debug_ptx_txt", data=b"ptx .file " + dbg),
        S(".rela.debug_line", type=4, data=dbg[:8].ljust(24, b"\0"), link=".symtab", info=".debug_line", flags=SHF_INFO_LINK),
        S(".nv.merc.debug_line", data=b"merc:" + dbg),
        S(".nv.merc.rela.debug_line", type=4, data=dbg[:4].ljust(24, b"\0"), link=".symtab", flags=SHF_INFO_LINK),
    ]
    sym = symbol
    code_sections = [
        S(".strtab", SHT_STRTAB, data=b"\0" + sym.encode() + b"\0"),
        S(".symtab", SHT_SYMTAB, data=b"\0" * 48, link=".strtab", info=2, align=8, entsize=24),
        S(f".nv.info.{sym}", 0x70000000, flags=SHF_INFO_LINK, data=b"\x04\x2f\x08\x00" * 4, link=".symtab", info=f".text.{sym}", align=4),
        S(f".text.{sym}", SHT_PROGBITS, flags=SHF_ALLOC | SHF_EXECINSTR, data=code, link=".symtab", info=1, align=128),
        *([S(f".nv.shared.{sym}", SHT_NOBITS, flags=SHF_ALLOC | 0x1, size=shared, align=16)] if shared is not None else []),
        S(f".nv.constant0.{sym}", SHT_PROGBITS, flags=SHF_ALLOC, data=b"\0" * 32, align=4),
    ]
    sections = [*debug_sections, *code_sections] if debug_first else [*code_sections, *debug_sections]
    return build_elf(sections, gap=gap, **kw)
