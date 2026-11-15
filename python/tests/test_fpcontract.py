"""The contraction check (:mod:`tilefish._fpcontract`, :func:`tilefish._build.check_contraction`) on text, with no GPU and no CUDA tools.

The check on real binaries, on sm_90, sm_100, sm_103 and sm_120: ``test_gpu_fpcontract.py``.
"""

import os
import stat

import pytest

from tilefish import _build, _fpcontract
from tilefish.errors import TilefishBuildError

PTX = """
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 p){
 mul.rn.f32x2 %rd5, %rd3, %rd4;
 add.rn.f32x2 %rd6, %rd5, %rd3;
 mul.rn.f32 %f3, %f1, 0f3F000000;
 @%p1 mul.f32 %f4, %f1, %f2;
 mul.rn.ftz.sat.f32 %f5, %f1, %f2;
 mul.rn.bf16x2 %r5, %r3, %r4;
 mul.rn.f16 %rs3, %rs1, %rs2;
 mul.rn.f64 %fd3, %fd1, 0d3FE0000000000000;
 mul.lo.s32 %r9, %r7, %r8;
 mul.wide.u32 %rd9, %r7, 4;
 fma.rn.f32 %f6, %f1, %f2, %f3;
 fma.rn.f32x2 %rd7, %rd3, %rd4, %rd5;
 // begin inline asm
 { .reg .f32 t; mul.rn.f32 t, %f1, %f2; add.rn.f32 %f7, t, %f3; }
 // end inline asm
 ret;
}
"""


def test_neutralised_ptx_multiplies_nothing_and_keeps_everything_else():
    # an or, not an xor: x * x as x ^ x would be 0, and ptxas would fold the fma of a polynomial in it away
    out, n = _fpcontract.neutralised(PTX)
    assert n == 8
    assert "or.b64 %rd5, %rd3, %rd4;" in out and "add.rn.f32x2 %rd6, %rd5, %rd3;" in out
    assert "or.b32 %f3, %f1, 0x3F000000;" in out  # a float immediate as the integer of its bits
    assert "@%p1 or.b32 %f4, %f1, %f2;" in out and "or.b32 %f5, %f1, %f2;" in out
    assert "or.b32 %r5, %r3, %r4;" in out and "or.b16 %rs3, %rs1, %rs2;" in out and "or.b64 %fd3, %fd1, 0x3FE0000000000000;" in out
    assert "mul.lo.s32 %r9, %r7, %r8;" in out and "mul.wide.u32 %rd9, %r7, 4;" in out  # integer multiplies stay
    assert "fma.rn.f32 %f6, %f1, %f2, %f3;" in out and "fma.rn.f32x2 %rd7, %rd3, %rd4, %rd5;" in out  # explicit fma stays
    assert "{ .reg .f32 t; or.b32 t, %f1, %f2; add.rn.f32 %f7, t, %f3; }" in out  # one line of inline asm
    assert " mul." not in out.replace("mul.lo", "").replace("mul.wide", "")


def test_contractible_and_explicit_fma():
    assert _fpcontract.contractible(PTX)
    assert not _fpcontract.contractible("mul.rn.f32 %f1, %f2, %f3;\nfma.rn.f32 %f4, %f1, %f2, %f3;")  # no add
    assert not _fpcontract.contractible("add.rn.f32 %f1, %f2, %f3;\nmul.lo.s32 %r1, %r2, %r3;")  # no float multiply
    assert _fpcontract.explicit_fma_lanes(PTX) == 3


SASS = """
        /*0040*/                   HFMA2 R7, -RZ, RZ, 0, 0 ;
        /*0050*/                   FFMA2 R6, R4.F32x2.HI_LO, R6.F32x2.HI_LO, R4.F32x2.HI_LO ;
        /*0060*/              @!P0 FFMA R9, -R9, R9, R16 ;
        /*0070*/                   FFMA R10, R11.reuse, R12, -RZ ;
        /*0080*/                   HFMA2.MMA R5, -RZ, RZ, 1.875, 0 ;
        /*0090*/                   FFMA.FTZ R1, R2, R3, R4 ;
        /*00a0*/                   DFMA R2, R4, R6, R8 ;
        /*00b0*/                   HFMA2 R0, R2.H0_H0, R3.H0_H0, R4.H0_H0 ;
        /*00c0*/                   FMUL R9, R4, R6 ;
        /*00d0*/                   FADD R4, R4, R9 ;
        /*00e0*/                   FFMA32I R1, R2, 1.5, R1 ;
        /*00f0*/                   HFMA2 R7, R20, 1, 1, R16 ;
        /*0100*/                   FFMA R3, R4, -1, R5 ;
"""


def test_the_fused_instructions_of_a_listing():
    found = _fpcontract.fused(SASS)
    assert [f.address for f in found] == [0x50, 0x60, 0x90, 0xA0, 0xB0, 0xE0]  # not the RZ idioms, not an add by a multiply by 1, not FMUL / FADD
    assert [f.lanes for f in found] == [2, 1, 1, 1, 2, 1]
    assert _fpcontract.lanes(found) == 8
    assert found[0].text.startswith("FFMA2 R6, R4.F32x2.HI_LO")


def test_nothing_to_contract_is_not_disassembled(monkeypatch):
    monkeypatch.setattr(_fpcontract, "assemble", lambda *a, **k: pytest.fail("assembled"))
    assert _fpcontract.check("add.rn.f32 %f1, %f2, %f3;", b"", {"enable_fp_fusion": False}, key="x") is None


def test_the_flags_of_a_reassembly():
    assert _fpcontract.ptxas_flags({"enable_fp_fusion": False}) == ["--fmad=false"]
    assert _fpcontract.ptxas_flags({"maxnreg": 128, "ptx_options": "-O3 --foo"}) == ["--fmad=false", "--maxrregcount=128", "-O3", "--foo"]


def _tool(directory, name):
    p = directory / name
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


def test_the_nvdisasm_named_by_the_environment_is_tried_first(tmp_path, monkeypatch):
    mine = _tool(tmp_path, "nvdisasm")
    monkeypatch.setenv(_fpcontract.NVDISASM_ENV, mine)
    assert _fpcontract.nvdisasm_candidates()[0] == mine
    monkeypatch.setenv(_fpcontract.NVDISASM_ENV, str(tmp_path / "missing"))
    assert str(tmp_path / "missing") not in _fpcontract.nvdisasm_candidates()  # a path that is not an executable file is skipped


class _Spec:
    stem = "row_axpb"
    triton = "kernels:row_axpb"
    binary = None
    source_sha256 = None
    enable_fp_fusion = False


def _verdict(proxy=False):
    fused = tuple(_fpcontract.fused(SASS))
    return _fpcontract.Verdict(lanes=8, allowed=1, target="sm_100a" if proxy else "sm_103a", proxy=proxy, nvdisasm="/x/nvdisasm", instructions=fused)


def test_a_contracted_extern_is_refused_with_its_name_and_the_fixes(monkeypatch):
    monkeypatch.setattr(_fpcontract, "check", lambda *a, **k: _verdict())
    with pytest.raises(TilefishBuildError) as e:
        _build.check_contraction(_Spec(), 103, PTX, b"", {"enable_fp_fusion": False}, "s")
    msg = str(e.value)
    assert "'row_axpb' (kernels:row_axpb) for sm_103" in msg and "8 lanes of fused multiply-adds" in msg and "has 1" in msg
    assert "/*0050*/ FFMA2 R6" in msg and "mul.rn.f32 $0, $1, $2;" in msg and "enable_fp_fusion=True" in msg
    monkeypatch.setattr(_fpcontract, "check", lambda *a, **k: _verdict(proxy=True))
    with pytest.raises(TilefishBuildError, match="assembled for sm_100a, the base of its architecture's family"):
        _build.check_contraction(_Spec(), 103, PTX, b"", {"enable_fp_fusion": False}, "s")


def test_a_contracted_lowered_kernel_is_a_bug_of_the_lowering(monkeypatch):
    spec = _Spec()
    spec.source_sha256 = "0" * 64
    monkeypatch.setattr(_fpcontract, "check", lambda *a, **k: _verdict())
    with pytest.raises(TilefishBuildError, match="bug of tilefish's lowering"):
        _build.check_contraction(spec, 103, PTX, b"", {"enable_fp_fusion": False}, "s")


def test_a_lowered_kernel_built_with_the_option_over_the_default_code_is_told_to_lower_with_fp_fusion_false(monkeypatch):
    spec = _Spec()
    spec.source_sha256, spec.enable_fp_fusion = "0" * 64, True
    monkeypatch.setattr(_fpcontract, "check", lambda *a, **k: _verdict())
    with pytest.raises(TilefishBuildError, match="lowered with fp_fusion=True and built with enable_fp_fusion=False as a build option"):
        _build.check_contraction(spec, 120, PTX, b"", {"enable_fp_fusion": False}, "s")


def test_only_a_binary_compiled_without_fusion_is_checked(monkeypatch):
    monkeypatch.setattr(_fpcontract, "check", lambda *a, **k: pytest.fail("checked"))
    for options in ({}, {"enable_fp_fusion": True}):
        _build.check_contraction(_Spec(), 103, PTX, b"", options, "s")
    spec = _Spec()
    spec.binary = {"source": "ptx"}  # a prebuilt binary: its fma instructions are its own
    _build.check_contraction(spec, 103, PTX, b"", {"enable_fp_fusion": False}, "s")


def test_no_tool_is_run_where_the_process_has_no_cuda(monkeypatch):
    monkeypatch.delenv(_fpcontract.NVDISASM_ENV, raising=False)
    assert all(os.path.isfile(p) for p in _fpcontract.nvdisasm_candidates())
