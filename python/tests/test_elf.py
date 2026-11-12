"""The strict ELF64 section walk and the machine-code digest of a cubin (``tilefish._elf``).

The files are built by ``elf_builder`` in the layout of a Mercury cubin. The real thing, built by Triton from
two different directories, is checked on a GPU machine in ``test_gpu_build.py``.
"""

import hashlib
import re
import struct

import pytest
from elf_builder import S, SHF_ALLOC, SHF_EXECINSTR, SHT_NOBITS, SHT_PROGBITS, build_elf, cubin_like

from tilefish._elf import CODE_DIGEST_FORMAT, code_digest, is_debug_section, parse_elf
from tilefish.errors import TilefishBuildError


@pytest.mark.parametrize(
    "name",
    [
        ".debug_line",
        ".debug_frame",
        ".debug_info",
        ".nv_debug_line_sass",
        ".nv_debug_ptx_txt",
        ".nv_debug_info_reg_sass",
        ".rela.debug_line",
        ".rel.debug_frame",
        ".nv.merc.debug_frame",
        ".nv.merc.nv_debug_ptx_txt",
        ".nv.merc.rela.debug_line",
        ".nv.merc.rela.nv_debug_line_sass",
        ".zdebug_info",
        ".nv.debug_x",
    ],
)
def test_debug_sections(name):
    assert is_debug_section(name)


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".text.tf_k",
        ".text.debug_k",  # a kernel called debug_k
        ".rela.text.tf_k",
        ".nv.merc.rela.text.tf_k",
        ".nv.info",
        ".nv.merc.nv.info",
        ".nv.constant0.tf_k",
        ".nv.shared.tf_k",
        ".nv.capmerc.text.tf_k",
        ".note.nv.tkinfo",
        ".symtab",
        ".nv.merc.symtab",
        ".shstrtab",
        ".strtab",
        "debug_line",
    ],
)
def test_other_sections_are_not_debug_sections(name):
    assert not is_debug_section(name)


# -- the digest ------------------------------------------------------------------------------------------------------------


def test_digest_is_hex_and_deterministic():
    d = code_digest(cubin_like())
    assert re.fullmatch(r"[0-9a-f]{64}", d)
    assert code_digest(cubin_like()) == d
    assert code_digest(bytearray(cubin_like())) == d
    assert code_digest(memoryview(cubin_like())) == d


def test_digest_format_is_pinned():
    # the canonical byte stream is part of the contract: a change must bump CODE_DIGEST_FORMAT
    assert CODE_DIGEST_FORMAT == 1
    assert code_digest(cubin_like()) == "5832df4d6a469acf13049acf1d8a3d2b7944953cbb63d6c47fce1174af91ae99"
    assert code_digest(cubin_like()) != hashlib.sha256(cubin_like()).hexdigest()


def test_digest_ignores_the_build_path_in_debug_sections():
    a = cubin_like(debug="/tmp/build_one/tilefish_generated/softmax.py")
    b = cubin_like(debug="/home/someone/else/entirely/longer/path/tilefish_generated/softmax.py")
    assert a != b and len(a) != len(b)
    assert code_digest(a) == code_digest(b)


def test_digest_ignores_where_sections_sit_in_the_file():
    assert code_digest(cubin_like(gap=0)) == code_digest(cubin_like(gap=37))


def test_digest_survives_debug_sections_moving_and_shifting_the_indices():
    # the references between sections are hashed as names, so moving the debug sections to the front changes
    # every index after them and nothing else
    assert code_digest(cubin_like()) == code_digest(cubin_like(debug_first=True))


def test_digest_covers_the_code():
    base = code_digest(cubin_like())
    flipped = bytearray(b"\x01\x02\x03\x04" * 16)
    flipped[17] ^= 0x01
    assert code_digest(cubin_like(code=bytes(flipped))) != base
    assert code_digest(cubin_like(code=b"\x01\x02\x03\x04" * 17)) != base


@pytest.mark.parametrize(
    "change",
    [
        {"osabi": 0x40},
        {"abiversion": 9},
        {"machine": 62},
        {"flags": 0x0900680A},  # another architecture
    ],
)
def test_digest_covers_the_elf_identity(change):
    assert code_digest(cubin_like(**change)) != code_digest(cubin_like())


def _sections(**overrides):
    base = [
        S(".symtab", 2, data=b"\0" * 24, link=".strtab", info=1, align=8, entsize=24),
        S(".strtab", 3, data=b"\0k\0"),
        S(".text.k", SHT_PROGBITS, flags=SHF_ALLOC | SHF_EXECINSTR, data=b"\xaa" * 16, link=".symtab", align=128),
        S(".nv.shared.k", SHT_NOBITS, flags=SHF_ALLOC, size=64, align=16),
        S(".debug_line", data=b"x"),
    ]
    for i, s in enumerate(base):
        if s.name in overrides:
            base[i] = overrides[s.name]
    return base


def test_digest_covers_every_field_of_a_section():
    base = code_digest(build_elf(_sections()))
    variants = {
        "name": _sections(**{".text.k": S(".text.j", SHT_PROGBITS, flags=SHF_ALLOC | SHF_EXECINSTR, data=b"\xaa" * 16, link=".symtab", align=128)}),
        "type": _sections(**{".text.k": S(".text.k", 7, flags=SHF_ALLOC | SHF_EXECINSTR, data=b"\xaa" * 16, link=".symtab", align=128)}),
        "flags": _sections(**{".text.k": S(".text.k", SHT_PROGBITS, flags=SHF_ALLOC, data=b"\xaa" * 16, link=".symtab", align=128)}),
        "align": _sections(**{".text.k": S(".text.k", SHT_PROGBITS, flags=SHF_ALLOC | SHF_EXECINSTR, data=b"\xaa" * 16, link=".symtab", align=64)}),
        "entsize": _sections(**{".text.k": S(".text.k", SHT_PROGBITS, flags=SHF_ALLOC | SHF_EXECINSTR, data=b"\xaa" * 16, link=".symtab", align=128, entsize=8)}),
        "link": _sections(**{".text.k": S(".text.k", SHT_PROGBITS, flags=SHF_ALLOC | SHF_EXECINSTR, data=b"\xaa" * 16, link=".strtab", align=128)}),
        "nobits size": _sections(**{".nv.shared.k": S(".nv.shared.k", SHT_NOBITS, flags=SHF_ALLOC, size=128, align=16)}),
        "data": _sections(**{".text.k": S(".text.k", SHT_PROGBITS, flags=SHF_ALLOC | SHF_EXECINSTR, data=b"\xab" * 16, link=".symtab", align=128)}),
    }
    digests = {name: code_digest(build_elf(s)) for name, s in variants.items()}
    assert all(d != base for d in digests.values()), {n for n, d in digests.items() if d == base}
    assert len(set(digests.values())) == len(digests)


def test_digest_covers_an_extra_non_debug_section_but_not_an_extra_debug_one():
    base = code_digest(build_elf(_sections()))
    assert code_digest(build_elf([*_sections(), S(".nv.constant0.k", data=b"\0" * 8)])) != base
    assert code_digest(build_elf([*_sections(), S(".debug_info", data=b"more debug")])) == base
    assert code_digest(build_elf([*_sections(), S(".nv.merc.rela.debug_frame", data=b"\1" * 24)])) == base


def test_a_section_hashes_by_its_contents_not_by_what_the_file_has_at_its_offset():
    # a NOBITS section has a size and no bytes; its (meaningless) offset is not looked at
    a = build_elf(_sections())
    b = build_elf(_sections(), gap=5)
    assert code_digest(a) == code_digest(b)


# -- the walk ----------------------------------------------------------------------------------------------------------------


def test_parse_elf_reads_sections():
    elf = parse_elf(cubin_like())
    names = [s.name for s in elf.sections]
    assert names[0] == "" and names[-1] == ".shstrtab"
    assert ".text.tf_k" in names and ".nv.merc.debug_line" in names
    text = elf.section(".text.tf_k")
    assert elf.data(text) == b"\x01\x02\x03\x04" * 16
    shared = elf.section(".nv.shared.tf_k")
    assert shared.size == 1024 and elf.data(shared) == b""
    assert elf.machine == 190 and elf.flags == 0x0900670A
    assert elf.section(".nope") is None


def _patch(blob, offset, fmt, *values):
    out = bytearray(blob)
    struct.pack_into(fmt, out, offset, *values)
    return bytes(out)


GOOD = cubin_like()
E_SHOFF, E_EHSIZE, E_SHENTSIZE, E_SHNUM, E_SHSTRNDX = 0x28, 0x34, 0x3A, 0x3C, 0x3E


@pytest.mark.parametrize(
    "blob, message",
    [
        (b"", "shorter than"),
        (GOOD[:63], "shorter than"),
        (b"\x7fELG" + GOOD[4:], "bad magic"),
        (build_elf([S(".text.k")], ei_class=1), "EI_CLASS"),
        (build_elf([S(".text.k")], ei_data=2), "EI_DATA"),
        (_patch(GOOD, 6, "<B", 2), "EI_VERSION"),
        (_patch(GOOD, E_EHSIZE, "<H", 52), "e_ehsize"),
        (_patch(GOOD, E_SHENTSIZE, "<H", 40), "e_shentsize"),
        (_patch(GOOD, E_SHNUM, "<H", 0), "no section header table"),
        (_patch(GOOD, E_SHSTRNDX, "<H", 0xFFFF), "extended section numbering"),
        (_patch(GOOD, E_SHSTRNDX, "<H", 400), "e_shstrndx"),
        (_patch(GOOD, E_SHOFF, "<Q", len(GOOD) - 10), "does not fit"),
        (_patch(GOOD, E_SHOFF, "<Q", 10), "does not fit"),
        (_patch(GOOD, E_SHNUM, "<H", 5000), "does not fit"),
        (GOOD[: len(GOOD) - 64], "does not fit"),
    ],
)
def test_malformed_headers_are_rejected(blob, message):
    with pytest.raises(TilefishBuildError, match=message):
        parse_elf(blob)
    with pytest.raises(TilefishBuildError, match=message):
        code_digest(blob)


def _header_at(blob, index):
    (shoff,) = struct.unpack_from("<Q", blob, E_SHOFF)
    return shoff + 64 * index


def test_a_section_that_leaves_the_file_is_rejected():
    elf = parse_elf(GOOD)
    i = elf.section(".text.tf_k").index
    huge = _patch(GOOD, _header_at(GOOD, i) + 32, "<Q", len(GOOD))  # sh_size
    with pytest.raises(TilefishBuildError, match=f"section {i} .* extends past the end"):
        parse_elf(huge)
    inside_header = _patch(GOOD, _header_at(GOOD, i) + 24, "<Q", 8)  # sh_offset
    with pytest.raises(TilefishBuildError, match="inside the ELF header"):
        parse_elf(inside_header)


def test_a_nobits_section_may_claim_any_size():
    elf = parse_elf(_patch(GOOD, _header_at(GOOD, parse_elf(GOOD).section(".nv.shared.tf_k").index) + 32, "<Q", 1 << 40))
    assert elf.section(".nv.shared.tf_k").size == 1 << 40


def test_section_zero_must_be_null():
    with pytest.raises(TilefishBuildError, match="SHT_NULL"):
        parse_elf(_patch(GOOD, _header_at(GOOD, 0) + 4, "<I", 1))


def test_the_name_table_must_be_a_string_table():
    elf = parse_elf(GOOD)
    last = len(elf.sections) - 1
    with pytest.raises(TilefishBuildError, match="SHT_STRTAB"):
        parse_elf(_patch(GOOD, _header_at(GOOD, last) + 4, "<I", 1))


def test_names_must_lie_in_the_name_table_and_end():
    elf = parse_elf(GOOD)
    i = elf.section(".text.tf_k").index
    with pytest.raises(TilefishBuildError, match="name offset"):
        parse_elf(_patch(GOOD, _header_at(GOOD, i), "<I", 100000))
    # the name table ends without a NUL: shorten it so that the last name is cut
    last = len(elf.sections) - 1
    table = elf.sections[last]
    cut = _patch(GOOD, _header_at(GOOD, last) + 32, "<Q", table.size - 1)
    with pytest.raises(TilefishBuildError, match="not NUL terminated"):
        parse_elf(cut)
    # not UTF-8
    bad = bytearray(GOOD)
    bad[table.offset + 1] = 0xFF  # the first byte of the first section's name
    with pytest.raises(TilefishBuildError, match="UTF-8"):
        parse_elf(bytes(bad))
