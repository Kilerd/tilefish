"""SASS listings, the instruction mix and the declared SASS assertions (:mod:`tilefish._sass`), on recorded listings, with no GPU and no CUDA tool.

``sass_fixtures/`` holds listings of the kernels of ``record_sass_fixtures.py`` (``tl.dot`` in bf16 and in TF32, a kernel that spills, one with
an atomic and f64 arithmetic) for sm_90, sm_100, sm_103 and sm_120, read by CUDA 13.x's nvdisasm and, for sm_100 (the base of sm_103's family)
and sm_120 (whose listings carry scheduling annotations), by 12.8's; trimmed to windows around the instructions the tests look at. They were
recorded with ``record_sass_fixtures.py``, whose kernels are on the lines the listings name, and ``expected.json`` is what the parser read in
them then. The same checks on fresh binaries: ``test_gpu_sass.py``.
"""

import json
import pathlib
import re
import struct

import pytest
from elf_builder import S, SHT_NOBITS, SHT_STRTAB, SHT_SYMTAB, build_elf

import tilefish
from tilefish import TilefishBuildError, TilefishDeclarationError, _build, _sass
from tilefish.aot import compile_specs, declaration_digest

HERE = pathlib.Path(__file__).parent
FIX = HERE / "sass_fixtures"
EXPECTED = json.loads((FIX / "expected.json").read_text())
RECORDER = (HERE / "record_sass_fixtures.py").read_text().splitlines()
DOT_LINE = 1 + next(i for i, line in enumerate(RECORDER) if "acc = tl.dot(a, b)" in line)
TF32_LINE = 1 + next(i for i, line in enumerate(RECORDER) if 'input_precision="tf32"' in line)
_INSTRUCTION = re.compile(r"^\s*/\*[0-9a-fA-F]+\*/\s")


def listing(name, ptx_lines=False):
    gp = FIX / f"{name}.gp.txt"
    return _sass.parse_listing((FIX / f"{name}.g.txt").read_text(), gp.read_text() if ptx_lines and gp.exists() else None)


def instructions(name):
    return listing(name).instructions(EXPECTED[name]["function"])


# -- listings ---------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_listing_parses_as_it_was_recorded(name):
    e = EXPECTED[name]
    text = (FIX / f"{name}.g.txt").read_text()
    found = listing(name)
    assert found.target == e["target"]
    assert list(found.functions) == [e["function"]]
    ins = found.instructions(e["function"])
    assert len(ins) == sum(1 for line in text.splitlines() if _INSTRUCTION.match(line))  # every instruction line is one instruction
    s = _sass.summary(ins)
    for key in ("instructions", "classes", "mma", "tf32_mma", "spills"):
        assert s[key] == e["trimmed"][key], key
    assert all(i.opcode.isupper() and ";" not in i.operands for i in ins)


@pytest.mark.parametrize("arch", ["sm_90", "sm_100", "sm_103", "sm_120"])
def test_the_mmas_of_a_dot_are_on_the_line_of_the_dot(arch):
    for kernel, line in (("dot_bf16", DOT_LINE), ("dot_tf32", TF32_LINE)):
        mmas = [i for i in instructions(f"{kernel}.{arch}.nvdisasm13") if _sass.class_of(i) == "mma"]
        assert mmas, (kernel, arch)
        if arch == "sm_90":  # ptxas may schedule the last asynchronous wgmma under the line of the store that waits for it (gsb0)
            assert {(i.file, i.line) for i in mmas} <= {("record_sass_fixtures.py", line), ("record_sass_fixtures.py", line + 1)}
            assert sum(i.line == line for i in mmas) > len(mmas) // 2
        else:
            assert {(i.file, i.line) for i in mmas} == {("record_sass_fixtures.py", line)}, (kernel, arch)


def test_the_ptx_line_of_each_instruction_is_merged_from_the_gp_listing():
    for name in ("dot_bf16.sm_103.nvdisasm13", "dot_bf16.sm_120.nvdisasm13", "dot_bf16.sm_90.nvdisasm13"):
        ins = listing(name, ptx_lines=True).instructions("dot_bf16")
        assert all(i.file != _sass.PTX_LINE_FILE for i in ins)  # the -gp file is not a source
        gp = _sass.parse_listing((FIX / f"{name}.gp.txt").read_text()).instructions("dot_bf16")
        lines = {i.address: i.line for i in gp if i.file == _sass.PTX_LINE_FILE}
        assert len(lines) > 20
        for i in ins:  # (the two listings are trimmed apart: where both have an instruction, it has the -gp listing's PTX line)
            assert i.ptx_line == lines.get(i.address)


def test_the_scheduling_annotations_of_cuda_12_8_are_not_operands():
    old = instructions("dot_bf16.sm_120.nvdisasm12")
    assert "&req" in (FIX / "dot_bf16.sm_120.nvdisasm12.g.txt").read_text()
    assert not any("&" in i.operands or "?" in i.operands for i in old)
    hmma = [i for i in old if i.opcode == "HMMA"]
    assert hmma and all(re.fullmatch(r"R\d+, R\d+, R\d+, (RZ|R\d+)", i.operands) for i in hmma)


def test_the_target_of_a_listing_and_of_the_family_base():
    assert listing("dot_bf16.sm_90.nvdisasm13").target == "sm_90a"  # from the header flags: an sm_90 listing has no .target
    assert listing("dot_bf16.sm_103.nvdisasm13").target == "sm_103a"
    assert listing("dot_bf16.sm_100.nvdisasm12").target == "sm_100a"  # what CUDA 12.8 reads of a sm_103 kernel assembled for its family base


def test_a_listing_line_with_inlined_at_and_notes():
    text = (
        "\t.target\tsm_120a\n"
        '\t.section\t.text.k,"ax",@progbits\n'
        '\t//## File "a/b.py", line 7 inlined at "a/c.py", line 3\n'
        '        /*0010*/                   STL [R1+0xf0], R7                              (*"SpillRefill"*);\n'
        '        /*0020*/              @!P0 LDL.64 R2, [R1+0x8] ;      /* 0x000fe20000100a00 */\n'
    )
    a, b = _sass.parse_listing(text).instructions("k")
    assert (a.file, a.line, a.inlined, a.opcode, a.operands) == ("a/b.py", 7, (("a/c.py", 3),), "STL", "[R1+0xf0], R7")
    assert a.notes == ("SpillRefill",)
    assert (b.predicate, b.mnemonic, b.operands, b.address) == ("@!P0", "LDL.64", "R2, [R1+0x8]", 0x20)
    assert b.describe() == "/*0020*/ @!P0 LDL.64 R2, [R1+0x8] (a/b.py:7)"
    assert _sass.summary([a, b])["spills"] == {"loads": 1, "stores": 1, "noted": 1}


# -- classes and MMAs -------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opcode,cls",
    [
        ("HMMA", "mma"), ("IMMA", "mma"), ("QMMA", "mma"), ("HGMMA", "mma"), ("QGMMA", "mma"), ("UTCHMMA", "mma"), ("UTCQMMA", "mma"),
        ("UTCIMMA", "mma"), ("UTMALDG", "tma"), ("UBLKCP", "tma"), ("LDTM", "tensor_memory"), ("UTCATOMSWS", "tensor_memory"), ("LDL", "local"),
        ("STL", "local"), ("LDG", "global_load"), ("LDGSTS", "global_load"), ("STG", "global_store"), ("LDS", "shared_load"),
        ("LDSM", "shared_load"), ("STS", "shared_store"), ("ATOMG", "atomic"), ("RED", "atomic"), ("ATOMS", "shared_atomic"), ("BAR", "barrier"), ("SYNCS", "barrier"),
        ("UTCBAR", "barrier"), ("WARPGROUP", "barrier"), ("UCGABAR_ARV", "barrier"), ("MEMBAR", "fence"), ("FENCE", "fence"), ("LDC", "constant"),
        ("DFMA", "fp64"), ("MUFU", "mufu"), ("F2FP", "convert"), ("SHFL", "shuffle"), ("BRA", "control"), ("S2R", "special"), ("FFMA2", "float"),
        ("HFMA2", "float"), ("UMOV", "uniform"), ("IMAD", "integer"), ("LOP3", "integer"),
    ],
)  # fmt: skip
def test_the_class_of_an_opcode(opcode, cls):
    assert _sass.class_of_opcode(opcode) == cls


def test_a_tcgen05_descriptor_says_the_kind_and_the_shape():
    # the descriptors ptxas set for the recorded kernels (sm_100 and sm_103): bf16 128 x 128, bf16 128 x 64, tf32 128 x 64
    assert _sass.decode_idesc("UTCHMMA", 0x8210490) == {"kind": "f16", "a": "bf16", "b": "bf16", "d": "f32", "m": 128, "n": 128}
    assert _sass.decode_idesc("UTCHMMA", 0x8100490) == {"kind": "f16", "a": "bf16", "b": "bf16", "d": "f32", "m": 128, "n": 64}
    assert _sass.decode_idesc("UTCHMMA", 0x8100910)["kind"] == "tf32"
    assert _sass.decode_idesc("UTCQMMA", 0x8100090)["kind"] == "f8f6f4"
    assert _sass.decode_idesc("UTCIMMA", 0x8100080)["kind"] == "i8"
    assert _sass.decode_idesc("HMMA", 0x8100490) is None


def test_a_descriptor_is_read_from_the_constant_that_sets_its_register_and_forgotten_when_it_is_written():
    text = (
        '\t.section\t.text.k,"ax",@progbits\n'
        "        /*0000*/                   UMOV UR27, 0x8100910 ;\n"
        "        /*0010*/                   UTCHMMA gdesc[UR6], gdesc[UR4], tmem[UR34], tmem[UR26], idesc[UR27], UPT ;\n"
        "        /*0020*/                   UIADD3 UR27, UPT, UPT, UR27, 0x1, URZ ;\n"
        "        /*0030*/                   UTCHMMA gdesc[UR6], gdesc[UR4], tmem[UR34], tmem[UR26], idesc[UR27], UPT ;\n"
    )
    ins = _sass.parse_listing(text).instructions("k")
    found = _sass.mma_descriptors(ins)
    assert set(found) == {0x10} and found[0x10]["kind"] == "tf32"
    assert _sass.summary(ins)["tf32_mma"] == 1


def test_the_lines_view_counts_per_source_line():
    rows = _sass.by_line(instructions("dot_bf16.sm_103.nvdisasm13"))
    dot = next(r for r in rows if r["line"] == DOT_LINE)
    assert dot["file"] == "record_sass_fixtures.py" and dot["columns"]["mma"] == 2
    assert sum(r["count"] for r in rows) == EXPECTED["dot_bf16.sm_103.nvdisasm13"]["trimmed"]["instructions"]
    assert [r["first"] for r in rows] == sorted(r["first"] for r in rows)


def test_the_summary_of_two_listings_differs_where_they_do():
    a = _sass.summary(instructions("dot_bf16.sm_103.nvdisasm13"))
    b = _sass.summary(instructions("dot_bf16.sm_120.nvdisasm13"))
    changes = _sass.diff_summaries(a, b)
    assert any(c.startswith("mma HMMA.16816.F32.BF16: 0 -> 18") for c in changes)
    assert any(c.startswith("mma UTCHMMA f16: 2 -> 0") for c in changes)
    assert _sass.diff_summaries(a, a) == []


# -- the ELF file and the occupancy -----------------------------------------------------------------------------------------------------------


def _attr(fmt, attr, value):
    if fmt == 4:
        return struct.pack("<BBH", fmt, attr, len(value)) + value
    if fmt == 3:
        return struct.pack("<BBH", fmt, attr, value)
    return struct.pack("<BBBB", fmt, attr, value, 0)


def test_the_attributes_of_a_function_from_its_elf_file():
    symtab = b"\0" * 24 + struct.pack("<IBBHQQ", 1, 0x12, 0x10, 4, 0, 0) + struct.pack("<IBBHQQ", 3, 0x12, 0x10, 5, 0, 0)
    info = _attr(4, 0x2F, struct.pack("<II", 2, 99)) + _attr(4, 0x2F, struct.pack("<II", 1, 40)) + _attr(4, 0x11, struct.pack("<II", 1, 16))
    own = _attr(3, 0x1B, 0xFF) + _attr(2, 0x4C, 1) + _attr(4, 0x10, struct.pack("<III", 128, 1, 1))
    cubin = build_elf(
        [
            S(".strtab", SHT_STRTAB, data=b"\0k\0j\0"),
            S(".symtab", SHT_SYMTAB, data=symtab, link=".strtab", entsize=24),
            S(".nv.info", 0x70000000, data=info),
            S(".text.k", flags=0x6, data=b"\0" * 32, info=(40 << 24) | 1),
            S(".text.j", flags=0x6, data=b"\0" * 32, info=(99 << 24) | 2),
            S(".nv.info.k", 0x70000000, data=own),
            S(".nv.shared.k", SHT_NOBITS, size=4096),
        ]
    )
    a = _sass.kernel_attributes(cubin, "k")
    assert a == {"sm": 103, "registers": 40, "frame_size": 16, "maxreg_count": 255, "num_barriers": 1, "reqntid": [128, 1, 1], "static_shared": 4096}
    assert _sass.kernel_attributes(cubin, "j")["registers"] == 99


def test_the_occupancy_estimate():
    assert _sass.occupancy(103, 32, 128, 0) == {"blocks_per_sm": 16, "warps_per_sm": 64, "occupancy": 1.0, "limited_by": ["registers", "warps"]}
    o = _sass.occupancy(103, 168, 256, 0)
    assert o["blocks_per_sm"] == 1 and o["limited_by"] == ["registers"]
    assert _sass.occupancy(120, 32, 128, 64 * 1024)["limited_by"] == ["shared"]
    assert _sass.occupancy(80, 32, 128, 0) is None


# -- the declaration --------------------------------------------------------------------------------------------------------------------------


def test_the_assertions_are_plain_data_and_per_architecture():
    c = tilefish.sass(no_tf32=True, mma="tcgen05.mma kind::f16", no_spills=True, max_registers=128, forbid=["atomic", "LDL"],
                      by_arch={"sm_120": tilefish.sass(mma="HMMA.16816.F32.BF16", no_spills=False)})  # fmt: skip
    assert c.canonical() == {
        "no_tf32": True, "no_spills": True, "max_registers": 128, "mma": ["tcgen05.mma kind::f16"], "forbid": ["atomic", "LDL"],
        "by_arch": {"sm_120": {"no_spills": False, "mma": ["HMMA.16816.F32.BF16"]}},
    }  # fmt: skip
    on120 = c.for_arch(120)
    assert (on120.mma, on120.no_spills, on120.max_registers, on120.by_arch) == (("HMMA.16816.F32.BF16",), False, 128, ())
    assert c.for_arch("sm_103").mma == ("tcgen05.mma kind::f16",) and c.for_arch(103).no_spills is True
    assert tilefish.sass().empty and _sass.checks_or_none(tilefish.sass(), "x") is None


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"mma": "FFMA"}, "is not an MMA opcode of SASS"),
        ({"mma": "fma.rn.f32"}, "an MMA is a PTX pattern"),
        ({"forbid": "spills"}, "forbid= takes instruction classes"),
        ({"forbid": [""]}, "a pattern is a non-empty string"),
        ({"max_registers": 0}, "max_registers is an int from 1 to 255"),
        ({"max_registers": True}, "max_registers is an int from 1 to 255"),
        ({"no_spills": 1}, "no_spills is True, False or None"),
        ({"by_arch": {"103": tilefish.sass()}}, "by_arch keys are architectures"),
        ({"by_arch": {"sm_103": {"no_spills": True}}}, "is a tilefish.sass"),
        ({"by_arch": {"sm_103": tilefish.sass(by_arch={"sm_120": tilefish.sass()})}}, "has a by_arch of its own"),
    ],
)
def test_a_wrong_assertion_is_a_declaration_error(kwargs, message):
    with pytest.raises(TilefishDeclarationError, match=re.escape(message)):
        tilefish.sass(**kwargs)


def test_patterns_match_tokens_in_order():
    hmma = instructions("dot_bf16.sm_120.nvdisasm13")[0].__class__(
        function="k", address=0, predicate="", opcode="HMMA", modifiers=("16816", "F32", "BF16"), operands="R0, R1, R2, RZ"
    )
    assert _sass.sass_pattern_matches("HMMA", hmma) and _sass.sass_pattern_matches("HMMA.16816.F32.BF16", hmma)
    assert _sass.sass_pattern_matches("HMMA.F32", hmma) and not _sass.sass_pattern_matches("HMMA.BF16.F32", hmma)
    assert not _sass.sass_pattern_matches("HGMMA", hmma)
    op = "tcgen05.mma.cta_group::1.kind::f16"
    assert _sass.ptx_pattern_matches("tcgen05.mma kind::f16", op) and _sass.ptx_pattern_matches("tcgen05.mma.kind::f16", op)
    assert not _sass.ptx_pattern_matches("tcgen05.mma kind::tf32", op) and not _sass.ptx_pattern_matches("mma.sync", op)
    assert _sass.ptx_pattern_matches("mma.sync m16n8k16 bf16", "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
    assert _sass.ptx_pattern_matches("wgmma.mma_async bf16", "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16")


N = tilefish.Dim("N", divisible_by=16)


def _extern(sass=None, name="sx_copy"):
    @tilefish.extern(triton="sx_mod:k", num_warps=4, grid=lambda N: (tilefish.cdiv(N, 256),), name=name, sass=sass)
    def op(x: tilefish.Tensor[N, tilefish.f32], y: tilefish.Out[tilefish.Tensor[N, tilefish.f32]], n: N, *, BLOCK: tilefish.const = 256): ...

    return op


def test_the_assertions_are_in_the_declaration_digest_only_when_given():
    plain = declaration_digest(_extern())
    assert declaration_digest(_extern(tilefish.sass())) == plain  # nothing said: the digest of before
    checked = declaration_digest(_extern(tilefish.sass(no_spills=True)))
    assert checked != plain
    assert checked != declaration_digest(_extern(tilefish.sass(no_spills=True, by_arch={"sm_120": tilefish.sass(no_spills=False)})))
    spec = compile_specs(_extern(tilefish.sass(max_registers=64)))[0]
    assert spec._sass.max_registers == 64 and "sass" not in spec.to_dict()  # in the digest, not in the spec's document
    again = type(spec).from_dict(spec.to_dict())
    assert again._sass is None
    again.verify_against(_extern(tilefish.sass(max_registers=64)))
    assert again._sass == spec._sass


def test_a_kernel_and_a_fused_region_carry_their_assertions_into_the_lowered_declaration():
    @tilefish.kernel(sass=tilefish.sass(max_registers=32))
    def sx_twice(x: tilefish.Tensor[N, tilefish.f32]) -> tilefish.Tensor[N, tilefish.f32]:
        return x * 2.0

    @tilefish.kernel
    def sx_twice_plain(x: tilefish.Tensor[N, tilefish.f32]) -> tilefish.Tensor[N, tilefish.f32]:
        return x * 2.0

    low = sx_twice.lower()
    assert low.extern.sass.max_registers == 32 and compile_specs(low.extern)[0]._sass.max_registers == 32
    assert sx_twice_plain.lower().extern.sass is None
    assert low.source.replace("sx_twice", "") == sx_twice_plain.lower().source.replace("sx_twice_plain", "")  # the source does not change

    from tilefish import _fusion

    def body(x: tilefish.Tensor[N, tilefish.bf16]) -> tilefish.Tensor[N, tilefish.bf16]:
        return (x.to(tilefish.f32) * 2.0 + 1.0).to(tilefish.bf16)

    plain = tilefish.graph(name="sx_g")(body)
    checked = tilefish.graph(name="sx_gc", fusion=tilefish.fusion(sass=tilefish.sass(no_spills=True)))(body)
    (a,), (b,) = _fusion.fused(plain).regions, _fusion.fused(checked).regions
    assert a.kernel.name != b.kernel.name  # a region's name is its content, its assertions included
    assert a.kernel.sass is None and b.kernel.sass.no_spills is True
    assert "sass" not in _fusion.settings_of(plain).canonical() and b.kernel.lower().extern.sass.no_spills


# -- judging a binary -------------------------------------------------------------------------------------------------------------------------


PTX = """
.version 8.8
.target sm_103a
.address_size 64
.file 1 "kernels/mm.py"
.visible .entry k(.param .u64 p){
 .loc 1 24 20
 @%p7 tcgen05.mma.cta_group::1.kind::tf32 [ %r398 + 0 ], %rd56, %rd57, %r161, %p6;
 .loc 1 25 4
 @%p7 tcgen05.mma.cta_group::1.kind::f16 [ %r398 + 0 ], %rd56, %rd57, %r161, %p6;
 ret;
}
"""


def judge(checks, name=None, *, ptx=None, attributes=None):
    calls = []

    def get():
        calls.append(1)
        return instructions(name)

    found = _sass.evaluate(checks, ptx=ptx, attributes=attributes or {"registers": 40}, instructions=get if name else None)
    return found, len(calls)


def test_no_spills_names_each_local_memory_instruction_and_its_source_line():
    found, _ = judge(tilefish.sass(no_spills=True), "dot_bf16.sm_120.nvdisasm13")
    (v,) = found
    assert v.check == "no_spills" and "4 local-memory instructions" in v.message
    # (CUDA 13.x's nvdisasm marks a spill: "SpillRefill")
    assert len(v.where) == 4 and all(re.search(r"/\*[0-9a-f]{4}\*/ (STL|LDL)\S* .* \(record_sass_fixtures\.py:\d+\) \[spill\]$", w) for w in v.where)
    assert judge(tilefish.sass(no_spills=True), "dot_bf16.sm_103.nvdisasm13")[0] == []
    spill = judge(tilefish.sass(no_spills=True), "spill.sm_120.nvdisasm13")[0][0]
    assert len(spill.where) == 26 and "..." in spill.describe()


def test_no_tf32_reads_the_ptx_and_where_it_has_none_the_sass():
    (v,), calls = judge(tilefish.sass(no_tf32=True), ptx=PTX)
    assert calls == 0 and v.where == ("PTX line 8: tcgen05.mma.cta_group::1.kind::tf32 (kernels/mm.py:24)",)
    for name in ("dot_tf32.sm_120.nvdisasm13", "dot_tf32.sm_90.nvdisasm13", "dot_tf32.sm_103.nvdisasm13", "dot_tf32.sm_100.nvdisasm12"):
        (v,), calls = judge(tilefish.sass(no_tf32=True), name)  # a prebuilt cubin: no PTX, its SASS (on sm_100+ the descriptor says the kind)
        assert calls == 1 and v.check == "no_tf32" and f":{TF32_LINE})" in v.where[0], name
    for name in ("dot_bf16.sm_120.nvdisasm13", "dot_bf16.sm_103.nvdisasm13"):
        assert judge(tilefish.sass(no_tf32=True), name)[0] == []


def test_an_mma_kind_is_found_in_the_ptx_or_in_the_sass():
    assert judge(tilefish.sass(mma="tcgen05.mma kind::f16"), ptx=PTX) == ([], 0)
    (v,), _ = judge(tilefish.sass(mma="tcgen05.mma kind::f8f6f4"), ptx=PTX)
    assert "the PTX has none: its MMAs are ['tcgen05.mma.cta_group::1.kind::f16', 'tcgen05.mma.cta_group::1.kind::tf32']" in v.message
    assert judge(tilefish.sass(mma="HMMA.16816.F32.BF16"), "dot_bf16.sm_120.nvdisasm13")[0] == []
    (v,), _ = judge(tilefish.sass(mma="HMMA.16816.F32.BF16"), "dot_tf32.sm_120.nvdisasm13")
    assert "its SASS MMAs are ['HMMA.1688.F32.TF32']" in v.message
    (v,), _ = judge(tilefish.sass(mma="tcgen05.mma kind::f16"), "dot_bf16.sm_103.nvdisasm13")  # a cubin without PTX
    assert "has no PTX" in v.message


def test_max_registers_reads_the_elf_file_and_forbid_reads_the_sass():
    assert judge(tilefish.sass(max_registers=40))[0] == []
    (v,), calls = judge(tilefish.sass(max_registers=32))
    assert calls == 0 and v.message == "the binary uses 40 registers per thread, more than the 32 declared"
    found, _ = judge(tilefish.sass(forbid=["atomic", "fp64", "HMMA"]), "atomics.sm_103.nvdisasm13")
    assert [v.check for v in found] == ["forbid", "forbid"]
    assert "the class 'atomic'" in found[0].message and "the class 'fp64'" in found[1].message
    assert all("(record_sass_fixtures.py:" in w for v in found for w in v.where)
    (v,), _ = judge(tilefish.sass(forbid="LDL.64"), "spill.sm_120.nvdisasm13")
    assert v.where and all(re.search(r" LDL(\.\w+)*\.64 ", w) for w in v.where)


def test_a_build_is_refused_with_the_assertions_the_instructions_and_where_the_sass_came_from(monkeypatch):
    spec = compile_specs(_extern(tilefish.sass(no_spills=True, max_registers=64, by_arch={"sm_103": tilefish.sass(no_spills=False)})))[0]
    text = (FIX / "dot_bf16.sm_120.nvdisasm13.g.txt").read_text()
    seen = []

    def disassemble(cubin, *, ptx=None, options=None):
        seen.append(ptx)
        return _sass.Disassembly(text, "", _sass.parse_listing(text), "sm_100a", True, "/x/nvdisasm", "12.8.55", "sm_103a")

    monkeypatch.setattr(_sass, "disassemble", disassemble)
    monkeypatch.setattr(_sass, "kernel_attributes", lambda cubin, symbol: {"registers": 80})
    spec_120 = compile_specs(_extern(tilefish.sass(no_spills=True, max_registers=64)))[0]
    with pytest.raises(TilefishBuildError) as e:
        _build.check_sass(spec_120, 120, b"", PTX, "dot_bf16", {})
    message = str(e.value)
    assert "'sx_copy' (sx_mod:k) for sm_120 violates the SASS assertions of its declaration" in message
    assert "max_registers: the binary uses 80 registers per thread, more than the 64 declared" in message
    assert "no_spills: 4 local-memory instructions" in message and "(record_sass_fixtures.py:" in message
    assert "assembled again for sm_100a" in message and "nvdisasm 12.8.55" in message
    with pytest.raises(TilefishBuildError, match="max_registers") as e:
        _build.check_sass(spec, 103, b"", PTX, "dot_bf16", {})  # sm_103 switches no_spills off: the SASS is not read
    assert "  - no_spills" not in str(e.value) and len(seen) == 1
    _build.check_sass(compile_specs(_extern())[0], 103, b"", PTX, "dot_bf16", {})  # no assertions: nothing looked at


def test_every_build_checks_the_assertions_a_cache_hit_too(tmp_path, monkeypatch):
    import sys

    import fake_compile

    from tilefish import runtime

    compiler = fake_compile.install(monkeypatch)
    monkeypatch.setattr(runtime, "_modules", {})
    monkeypatch.setattr(runtime, "_gen_root", [])
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    @tilefish.kernel(sass=tilefish.sass(max_registers=8))
    def sx_hit(x: tilefish.Tensor[N, tilefish.f32]) -> tilefish.Tensor[N, tilefish.f32]:
        return x * 2.0

    judged = []

    def judge(checks, *, cubin, ptx, symbol, options=None):
        judged.append((checks.max_registers, symbol))
        found = _sass.Judged()
        if len(judged) > 1:
            found.violations = [_sass.Violation("max_registers", "the binary uses 9 registers per thread, more than the 8 declared")]
        return found

    monkeypatch.setattr(_sass, "judge", judge)
    runtime.export(sx_hit, tmp_path / "a", arch=103)
    assert compiler.count == 1 and judged == [(8, "tf_k")]
    with pytest.raises(TilefishBuildError, match="more than the 8 declared"):
        runtime.export(sx_hit, tmp_path / "b", arch=103)  # served by the cache, and judged again
    assert compiler.count == 1 and len(judged) == 2


def test_every_function_of_the_binary_is_judged_and_the_kernel_s_comes_first():
    text = (
        '\t.section\t.text.helper,"ax",@progbits\n'
        "        /*0000*/                   STL [R1], R2 ;\n"
        "        /*0010*/                   RET.REL.NODEC R20 `(k) ;\n"
        '\t.section\t.text.k,"ax",@progbits\n'
        "        /*0000*/                   CALL.REL.NOINC `(helper) ;\n"
        "        /*0010*/                   EXIT ;\n"
    )
    listing = _sass.parse_listing(text)
    assert [(i.function, i.opcode) for i in _sass.kernel_instructions(listing, "k")] == [("k", "CALL"), ("k", "EXIT"), ("helper", "STL"), ("helper", "RET")]
    found = _sass.evaluate(tilefish.sass(no_spills=True), ptx=None, attributes={}, instructions=lambda: _sass.kernel_instructions(listing, "k"))
    assert [v.check for v in found] == ["no_spills"]


def test_the_target_a_cubin_is_read_for():
    from elf_builder import arch_header, cubin_like

    cubin = cubin_like(**arch_header("sm_103"))
    assert _sass._target_of(cubin, ".version 8.8\n.target sm_103a\n") == "sm_103a"
    assert _sass._target_of(cubin, ".version 7.0\n.target sm_80\n") == "sm_103"  # a prebuilt PTX of an older target, assembled for sm_103
    assert _sass._target_of(cubin, None) == "sm_103"
    assert _sass._target_of(cubin_like(**arch_header("sm_90")), ".target sm_90a\n") == "sm_90a"
    assert _sass._PROXY_BASE["sm_103"] == "sm_100" and _sass._PROXY_BASE["sm_103a"] == "sm_100a"


def test_the_disassembly_label_says_when_the_sass_is_a_reassembly():
    d = _sass.Disassembly("", "", _sass.Listing(None, {}), "sm_103a", False, "/x/nvdisasm", "13.5.37")
    assert d.label == "its SASS, read by nvdisasm 13.5.37"
    p = _sass.Disassembly("", "", _sass.Listing(None, {}), "sm_100a", True, "/x/nvdisasm", "12.8.55", "sm_103a")
    assert p.label.startswith("the SASS of its PTX assembled again for sm_100a, the base of its family, read by nvdisasm 12.8.55, which cannot decode sm_103a")
    assert _sass.proxy_flags({"enable_fp_fusion": False, "ptx_options": "-O3"}) == ["-lineinfo", "--fmad=false", "-O3"]
    assert _sass.proxy_flags({}) == ["-lineinfo"]
