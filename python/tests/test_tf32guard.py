"""The TF32 guard, with no Triton and no GPU: the detector on IR text, and the build that refuses (``tilefish._tf32guard``, ``_build.check_tf32``).

A kernel that asks for IEEE f32 dots (``tl.dot(..., input_precision="ieee")``, which TTIR prints with no ``inputPrecision``) must not be built into
TF32 tensor-core MMAs. Triton 3.5.1 does that on sm_100 and sm_103 when an operand of the dot was loaded narrower than 32 bits (an f16 tile
converted to f32) and the tile has 64 rows. The IR below is what Triton 3.5.1 printed for small kernels on sm_103, sm_90 and sm_120 (abridged to
the lines that matter: the frontend's dot, the TTIR's dots and the TTGIR's MMAs with their locations, the PTX's MMA instructions; the path of the
kernel file is shortened to ``kernels.py``). ``test_gpu_tf32guard.py`` compiles and runs the real kernels.
"""

import importlib
import linecache
import sys

import fake_compile
import pytest

import tilefish
from tilefish import Dim, Out, Tensor, TilefishBuildError, _cache, _tf32guard, const, f32
from tilefish._tf32guard import f32_dots, problem, ptx_tf32_mmas, tensor_core_ops
from tilefish.aot import build, compile_specs

# -- an f32 dot at M = 64 whose operand a was loaded as f16: tcgen05, TF32 (sm_103) ------------------------------------------------------------

FRONTEND_IEEE = "    %34 = tt.dot %18, %33, %cst_10 : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32>\n"
FRONTEND_IEEE_M32 = "    %34 = tt.dot %18, %33, %cst_10 : tensor<32x16xf32> * tensor<16x32xf32> -> tensor<32x32xf32>\n"
FRONTEND_TF32 = "    %34 = tt.dot %18, %33, %cst_10, inputPrecision = tf32 : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32>\n"
FRONTEND_TF32X3 = "    %34 = tt.dot %18, %33, %cst_10, inputPrecision = tf32x3 : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32>\n"

TTIR_IEEE = """    %c_19 = tt.dot %a_9, %b_18, %c : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32> loc(#loc26)
#loc2 = loc("kernels.py":22:18)
#loc26 = loc("c"(#loc2))
"""
TTGIR_IEEE = """    ttng.tc_gen5_mma %c_28, %b_26, %c_27, %true, %true, %c_29[%true] {is_async} : !ttg.memdesc<64x16xf32, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<16x32xf32, #shared, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc30)
#loc8 = loc("kernels.py":22:18)
#loc30 = loc("c"(#loc8))
"""
PTX_TCGEN05_TF32 = """	// begin inline asm
	@%p8 tcgen05.mma.cta_group::1.kind::tf32 [ %r157 + 0 ], [ %r90 + 0 ], %rd20, %r91, %p7;
	// end inline asm
	// begin inline asm
	@%p8 tcgen05.mma.cta_group::1.kind::tf32 [ %r157 + 0 ], [ %r90 + 8 ], %rd21, %r91, %p7;
	// end inline asm
"""
# the same kernel at M = 32: an FMA dot (sm_103)
TTGIR_FMA = """#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
    %c = tt.dot %a_28, %b_29, %cst : tensor<32x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf32, #blocked> loc(#loc37)
"""
PTX_FMA = "\tfma.rn.f32 \t%r133, %r35, %r101, 0f00000000;\n\tfma.rn.f32 \t%r134, %r36, %r103, %r133;\n"
# input_precision="tf32" at M = 64 on sm_90 (wgmma) and sm_120 (mma.sync)
TTGIR_SM90 = """#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 8]}>
    %c = ttng.warp_group_dot %a_17, %b_28, %cst_2 {inputPrecision = 0 : i32} : tensor<64x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * !ttg.memdesc<16x32xf32, #shared1, #smem> -> tensor<64x32xf32, #mma> loc(#loc37)
"""
PTX_SM90 = "\twgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32 {%r61,%r62,%r63,%r64}, {%r21,%r22,%r23,%r24}, %rd13, 0, 1, 1;\n"
TTGIR_SM120 = """#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
    %c = tt.dot %a_17, %b_29, %cst, inputPrecision = tf32 : tensor<64x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x32xf32, #mma> loc(#loc37)
"""
PTX_SM120 = "\tmma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 { %r71, %r72, %r73, %r74 }, { %r37, %r38, %r39, %r40 }, { %r55, %r56 }, { %r71, %r72, %r73, %r74 };\n"

# -- a kernel with an IEEE dot (line 36) and a TF32 dot in a K loop (line 41), sm_103 --------------------------------------------------------------
# the IEEE dot's a is loaded as f16 in MIXED_TRAPPED (tcgen05, TF32), as f32 in MIXED_CLEAN (FMA); the pipelined loop gives the TF32 dot two MMAs

FRONTEND_MIXED = """    %34 = tt.dot %18, %33, %cst_10 : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32>
      %115 = tt.dot %91, %114, %arg7, inputPrecision = tf32 : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32>
"""
TTIR_MIXED = """    %c = tt.dot %a_11, %b_19, %cst_0 : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32> loc(#loc62)
      %acc_34 = tt.dot %a2_28, %b2_33, %acc_20, inputPrecision = tf32 : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32> loc(#loc74)
#loc19 = loc("kernels.py":36:18)
#loc30 = loc("kernels.py":41:29)
#loc62 = loc("c"(#loc19))
#loc74 = loc("acc"(#loc30))
"""
_LOOP_MMAS = """    ttng.tc_gen5_mma %a2_41, %b2_46, %acc, %false, %acc_39, %acc_36[%true] {is_async} : !ttg.memdesc<64x16xf32, #shared2, #smem, mutable, 3x64x16>, !ttg.memdesc<16x32xf32, #shared, #smem, mutable, 3x16x32>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2x1> loc(#loc60)
      ttng.tc_gen5_mma %a2_107, %b2_108, %acc, %true, %true, %acc_109[%true] {is_async} : !ttg.memdesc<64x16xf32, #shared2, #smem, mutable, 3x64x16>, !ttg.memdesc<16x32xf32, #shared, #smem, mutable, 3x16x32>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2x1> loc(#loc60)
"""
_MIXED_LOCS = """#loc9 = loc("kernels.py":36:18)
#loc19 = loc("kernels.py":41:29)
#loc50 = loc("c"(#loc9))
#loc60 = loc("acc"(#loc19))
"""
TTGIR_MIXED_TRAPPED = (
    "    ttng.tc_gen5_mma %c_31, %b_29, %c_30, %true, %true, %c_32[%true] {is_async} : !ttg.memdesc<64x16xf32, #tmem1, #ttng.tensor_memory>, "
    "!ttg.memdesc<16x32xf32, #shared, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, "
    "mutable> loc(#loc50)\n" + _LOOP_MMAS + _MIXED_LOCS
)
TTGIR_MIXED_CLEAN = (
    "#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>\n"
    "    %c = tt.dot %a_28, %b_29, %cst : tensor<64x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x32xf32, #ttg.dot_op<{opIdx = 1, "
    "parent = #blocked}>> -> tensor<64x32xf32, #blocked> loc(#loc50)\n" + _LOOP_MMAS + _MIXED_LOCS
)
PTX_MIXED = PTX_TCGEN05_TF32 * 2


# -- the detector ---------------------------------------------------------------------------------------------------------------------------------


def test_an_ieee_f32_dot_is_one_that_says_no_precision_or_ieee():
    assert [(d.name, d.precision) for d in f32_dots(FRONTEND_IEEE)] == [("tt.dot", "ieee")]  # IEEE is the default, which MLIR does not print
    assert [d.precision for d in f32_dots(FRONTEND_TF32 + FRONTEND_TF32X3)] == ["tf32", "tf32x3"]
    assert [d.precision for d in f32_dots(FRONTEND_IEEE.replace("%cst_10 :", "%cst_10, inputPrecision = ieee :"))] == ["ieee"]
    # Hopper's warp_group_dot prints its precision as an attribute (the enum of Triton 3.5.1: 0 tf32, 1 tf32x3, 2 ieee), and IEEE not at all
    assert [d.precision for d in f32_dots(TTGIR_SM90)] == ["tf32"]
    assert [d.precision for d in f32_dots(TTGIR_SM90.replace(" {inputPrecision = 0 : i32}", ""))] == ["ieee"]
    assert [d.precision for d in f32_dots(TTGIR_SM90.replace("= 0 : i32", "= 2 : i32"))] == ["ieee"]
    # not an f32 dot: 16-bit operands, a scaled dot, the wait of a warp group
    assert f32_dots("    %5 = tt.dot %1, %2, %3 : tensor<64x16xf16> * tensor<16x32xf16> -> tensor<64x32xf32>\n") == []
    assert f32_dots("    %5 = tt.dot_scaled %1, %2, %3 lhs = e4m3 rhs = e4m3 : tensor<64x32xf32> * tensor<32x32xf32> -> tensor<64x32xf32>\n") == []
    assert f32_dots("    %6 = ttng.warp_group_dot_wait %5 {pendings = 0 : i32} : tensor<64x32xf32, #mma>\n") == []


def test_the_tensor_core_ops_of_a_ttgir():
    (op,) = tensor_core_ops(TTGIR_IEEE)
    assert (op.name, op.precision, op.where) == ("ttng.tc_gen5_mma", None, "kernels.py:22:18")  # an MMAv5 op keeps no precision
    assert tensor_core_ops(TTGIR_FMA) == []  # a tt.dot of a blocked layout is an FMA dot
    assert [(o.name, o.precision) for o in tensor_core_ops(TTGIR_SM90)] == [("ttng.warp_group_dot", "tf32")]
    assert [(o.name, o.precision) for o in tensor_core_ops(TTGIR_SM120)] == [("tt.dot", "tf32")]  # an nvidia_mma result layout: MMAv2
    assert tensor_core_ops(TTGIR_IEEE.replace("64x16xf32", "64x16xf16").replace("16x32xf32", "16x32xf16")) == []  # f16 operands: not TF32


def test_the_tf32_instructions_of_a_ptx():
    assert ptx_tf32_mmas(PTX_TCGEN05_TF32) == ["tcgen05.mma.cta_group::1.kind::tf32"] * 2
    assert ptx_tf32_mmas(PTX_SM90) == ["wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32"]
    assert ptx_tf32_mmas(PTX_SM120) == ["mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"]
    others = (
        "\ttcgen05.mma.cta_group::1.kind::f16 [ %r1 + 0 ], %rd2, %rd3, %r4, %p5;\n"
        "\twgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%r1}, %rd1, %rd2, 0, 1, 1, 1, 1;\n"
        "\tmma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r1 }, { %r2 }, { %r3 }, { %r4 };\n"
        "\t// tcgen05.mma.cta_group::1.kind::tf32 in a comment\n"
        "\ttcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r1];\n"
    )
    assert ptx_tf32_mmas(PTX_FMA + others) == []


def test_an_ieee_dot_compiled_into_tf32_mmas_is_refused():
    found = problem(FRONTEND_IEEE, PTX_TCGEN05_TF32, ttir=TTIR_IEEE, ttgir=TTGIR_IEEE)
    assert found == (
        "it asks for 1 f32 dot with IEEE input precision, and was compiled into TF32 tensor-core MMAs: 2 x tcgen05.mma.cta_group::1.kind::tf32 in "
        "its PTX; 1 x ttng.tc_gen5_mma of f32 operands in its TTGIR (the dot at kernels.py:22:18)"
    )
    # a build served by the cache has its PTX only, which is enough for a kernel whose f32 dots all ask for IEEE
    assert problem(FRONTEND_IEEE, PTX_TCGEN05_TF32) == (
        "it asks for 1 f32 dot with IEEE input precision, and was compiled into TF32 tensor-core MMAs: 2 x tcgen05.mma.cta_group::1.kind::tf32 in its PTX"
    )
    # either signal alone refuses: an MMA in the TTGIR, an instruction in the PTX
    assert problem(FRONTEND_IEEE, PTX_FMA, ttir=TTIR_IEEE, ttgir=TTGIR_IEEE) is not None
    assert problem(FRONTEND_IEEE, PTX_TCGEN05_TF32, ttir=TTIR_IEEE, ttgir=TTGIR_FMA) is not None
    # an IEEE warp_group_dot (a later Triton, or Gluon's warpgroup_mma) that became wgmma of TF32 inputs
    ieee_sm90 = TTGIR_SM90.replace(" {inputPrecision = 0 : i32}", "")
    assert "1 x wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32 in its PTX" in problem(ieee_sm90, PTX_SM90, ttgir=ieee_sm90)


def test_an_ieee_dot_on_fmas_and_a_dot_that_allows_tf32_pass():
    assert problem(FRONTEND_IEEE_M32, PTX_FMA, ttgir=TTGIR_FMA) is None  # M = 32: an FMA dot, exact
    assert problem(FRONTEND_IEEE, PTX_FMA, ttgir=TTGIR_FMA) is None  # sm_90 and sm_120 keep the IEEE dot on FMAs too
    for frontend in (FRONTEND_TF32, FRONTEND_TF32X3):  # input_precision="tf32" or "tf32x3" asks for TF32: never refused
        assert problem(frontend, PTX_TCGEN05_TF32, ttgir=TTGIR_IEEE) is None
        assert problem(frontend, PTX_TCGEN05_TF32) is None
    assert problem(FRONTEND_TF32, PTX_SM90, ttgir=TTGIR_SM90) is None
    assert problem(FRONTEND_TF32, PTX_SM120, ttgir=TTGIR_SM120) is None
    assert problem("module {\n}\n", PTX_TCGEN05_TF32, ttgir=TTGIR_IEEE) is None  # no f32 dot at all (a Gluon kernel's own tcgen05_mma)


def test_a_kernel_that_mixes_ieee_and_tf32_dots_is_judged_op_by_op():
    """The TF32 dot's MMAs (two: the pipelined loop) are traced to the dot at line 41, which allows TF32; the IEEE dot's, to line 36."""
    assert problem(FRONTEND_MIXED, PTX_MIXED, ttir=TTIR_MIXED, ttgir=TTGIR_MIXED_CLEAN) is None
    assert problem(FRONTEND_MIXED, PTX_MIXED, ttir=TTIR_MIXED, ttgir=TTGIR_MIXED_TRAPPED) == (
        "it asks for 1 f32 dot with IEEE input precision, and 1 TF32 tensor-core MMA of its TTGIR is made of it: 1 x ttng.tc_gen5_mma of f32 "
        "operands (the dot at kernels.py:36:18)"
    )
    # from the cache (PTX only) a mixed kernel is not judged again: the build that stored it was, with its TTGIR
    assert problem(FRONTEND_MIXED, PTX_MIXED) is None


def test_an_mma_of_a_mixed_kernel_that_cannot_be_traced_is_refused():
    elsewhere = TTGIR_MIXED_CLEAN.replace('#loc19 = loc("kernels.py":41:29)', '#loc19 = loc("kernels.py":99:1)')  # a location no dot of the TTIR has
    assert "2 TF32 tensor-core MMAs of its TTGIR cannot be traced, by source location, to a dot that allowed TF32" in problem(
        FRONTEND_MIXED, PTX_MIXED, ttir=TTIR_MIXED, ttgir=elsewhere
    )
    assert "cannot be traced" in problem(FRONTEND_MIXED, PTX_MIXED, ttir=None, ttgir=TTGIR_MIXED_CLEAN)  # no TTIR to trace them to
    unknown = TTGIR_MIXED_CLEAN.replace("ttng.tc_gen5_mma", "ttng.tcgen5_mma_v2")  # a TTGIR whose MMA this check does not know
    assert "its TTGIR has no tensor-core op this check knows" in problem(FRONTEND_MIXED, PTX_MIXED, ttir=TTIR_MIXED, ttgir=unknown)


def test_a_gluon_kernel_has_its_own_tensor_core_ops():
    """A Gluon kernel's tcgen05_mma is in its frontend TTGIR already: f32 operands on TF32 tensor cores by the kernel's choice, and its MMAs are
    never traced to dots (Gluon makes no MMA of a dot). An IEEE warp_group_dot next to it is judged by the precision it keeps."""
    own = "    %37 = ttng.tc_gen5_mma %33, %36, %result[], %false, %true : !ttg.memdesc<128x16xf32, #shared, #smem, mutable>, !ttg.memdesc<16x32xf32, #shared2, #smem, mutable>, !ttg.memdesc<128x32xf32, #tmem, #ttng.tensor_memory, mutable>\n"
    assert problem(own, PTX_TCGEN05_TF32, language="gluon", ttgir=own) is None
    ieee_wgmma = TTGIR_SM90.replace(" {inputPrecision = 0 : i32}", "")
    assert problem(own + ieee_wgmma, PTX_TCGEN05_TF32 + PTX_SM90, language="gluon", ttgir=own) is None
    assert "1 x ttng.warp_group_dot of f32 operands" in problem(own + ieee_wgmma, PTX_SM90, language="gluon", ttgir=own + ieee_wgmma)


def test_locations_resolve_through_call_sites_and_names():
    """A dot in a helper called twice: the TTIR and the TTGIR of one compile give each call site's dot and MMA the same location (Triton 3.5.1)."""
    ttir = """    %c = tt.dot %a_10, %b_19, %cst : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32> loc(#loc51)
    %d = tt.dot %a_10, %b_19, %cst, inputPrecision = tf32 : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32> loc(#loc52)
#loc18 = loc("kernels.py":13:21)
#loc19 = loc("kernels.py":23:22)
#loc20 = loc("kernels.py":24:22)
#loc49 = loc("c"(#loc19))
#loc50 = loc("d"(#loc20))
#loc51 = loc(callsite(#loc18 at #loc49))
#loc52 = loc(callsite(#loc18 at #loc50))
"""
    mma = "    ttng.tc_gen5_mma %c_28, %b_26, %{acc}, %true, %true, %c_29[%true] {{is_async}} : !ttg.memdesc<64x16xf32, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<16x32xf32, #shared, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc{n})\n"
    locs = """#loc8 = loc("kernels.py":13:21)
#loc9 = loc("kernels.py":23:22)
#loc17 = loc("kernels.py":24:22)
#loc36 = loc("c"(#loc9))
#loc44 = loc("d"(#loc17))
#loc45 = loc(callsite(#loc8 at #loc36))
#loc46 = loc(callsite(#loc8 at #loc44))
"""
    frontend = FRONTEND_IEEE + FRONTEND_TF32
    both = mma.format(acc="c_27", n=45) + mma.format(acc="d", n=46) + locs
    assert [o.loc for o in tensor_core_ops(both)] == [d.loc for d in f32_dots(ttir)]
    assert tensor_core_ops(both)[0].where == "kernels.py:13:21"
    assert "made of it" in problem(frontend, PTX_MIXED, ttir=ttir, ttgir=both)  # the IEEE call site's MMA
    assert problem(frontend, PTX_TCGEN05_TF32, ttir=ttir, ttgir=mma.format(acc="d", n=46) + locs) is None  # the TF32 call site's only


# -- the build ------------------------------------------------------------------------------------------------------------------------------------

R = Dim("R")
ARCH = 103

KERNELS = '''from fake_jit import jit


@jit
def dot_k(a, b, c, BM, PREC):
    precision = ["", ", inputPrecision = tf32", ", inputPrecision = tf32x3"][PREC]  # tl.dot's input_precision: "ieee", "tf32", "tf32x3"
    return f"%c = tt.dot %a, %b, %cst{precision} : tensor<{BM}x16xf32> * tensor<16x32xf32> -> tensor<{BM}x32xf32>"


@jit
def mixed_k(a, b, c, d):
    return (
        "%c = tt.dot %a, %b, %cst : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32>",
        "%d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32>",
    )
'''


@pytest.fixture
def world(tmp_path, monkeypatch):
    compiler = fake_compile.install(monkeypatch)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    directory = tmp_path / "project"
    directory.mkdir()
    (directory / "tf32_fake_kernels.py").write_text(KERNELS)
    monkeypatch.syspath_prepend(str(directory))
    sys.modules.pop("tf32_fake_kernels", None)
    importlib.invalidate_caches()
    linecache.clearcache()
    yield compiler
    sys.modules.pop("tf32_fake_kernels", None)


PRECISIONS = {"ieee": 0, "tf32": 1, "tf32x3": 2}  # a const is a number


def declare(bm=64, prec="ieee"):
    @tilefish.extern(triton="tf32_fake_kernels:dot_k", num_warps=4, grid=lambda R, BM: (tilefish.cdiv(R, BM),), name=f"dot_{prec}_{bm}")
    def op(a: Tensor[R, 16, f32], b: Tensor[16, 32, f32], c: Out[Tensor[R, 32, f32]], BM: const = bm, PREC: const = PRECISIONS[prec]): ...

    return op


def declare_mixed():
    @tilefish.extern(triton="tf32_fake_kernels:mixed_k", num_warps=4, grid=lambda R: (tilefish.cdiv(R, 64),), name="mixed")
    def op(a: Tensor[R, 16, f32], b: Tensor[16, 32, f32], c: Out[Tensor[R, 32, f32]], d: Out[Tensor[R, 32, f32]]): ...

    return op


def test_a_build_refuses_an_ieee_dot_compiled_into_tf32_mmas_and_stores_nothing(world):
    compiler = world
    compiler.ptx_body = PTX_TCGEN05_TF32
    (spec,) = compile_specs(declare())
    with pytest.raises(TilefishBuildError) as raised:
        build(spec, arch=ARCH)
    message = str(raised.value)
    assert "'dot_ieee_64' (tf32_fake_kernels:dot_k) for sm_103 with Triton 3.5.1: it asks for 1 f32 dot with IEEE input precision" in message
    assert "2 x tcgen05.mma.cta_group::1.kind::tf32 in its PTX" in message and "refusing to build it" in message
    assert "fewer than 64 rows" in message and "tl.fma" in message and 'input_precision="tf32"' in message  # the fixes
    with pytest.raises(TilefishBuildError, match="IEEE input precision"):
        build(spec, arch=ARCH)
    assert compiler.count == 2  # the refused binary was not stored: the second build compiled again


def test_a_build_takes_an_ieee_dot_on_fmas_and_a_tf32_dot_on_tensor_cores(world):
    compiler = world
    compiler.ptx_body = PTX_FMA
    (spec,) = compile_specs(declare(bm=32))
    assert build(spec, arch=ARCH).ptx.endswith(PTX_FMA)
    compiler.ptx_body = PTX_TCGEN05_TF32
    for prec in ("tf32", "tf32x3"):
        (spec,) = compile_specs(declare(prec=prec))
        assert "kind::tf32" in build(spec, arch=ARCH).ptx


def test_a_cache_entry_stored_without_the_check_is_refused_when_it_is_read(world, monkeypatch):
    """An entry of the cache has the PTX of its binary: a hit is checked again (the recipe is bumped too, so no older entry is read at all)."""
    compiler = world
    compiler.ptx_body = PTX_TCGEN05_TF32
    (spec,) = compile_specs(declare())
    with monkeypatch.context() as m:
        m.setattr(_tf32guard, "problem", lambda *a, **k: None)
        build(spec, arch=ARCH)  # stored
    assert compiler.count == 1
    with pytest.raises(TilefishBuildError, match=r"tcgen05\.mma\.cta_group::1\.kind::tf32 in its PTX"):
        build(spec, arch=ARCH)
    assert compiler.count == 1  # a hit, refused


def test_the_recipe_of_the_cache_is_the_one_with_the_check():
    assert _cache.CACHE_RECIPE == 4


def test_a_mixed_kernel_is_judged_on_the_ttir_and_ttgir_of_its_compile(world):
    compiler = world
    compiler.ptx_body = PTX_MIXED
    compiler.ttir = TTIR_MIXED
    compiler.ttgir_body = TTGIR_MIXED_TRAPPED
    (spec,) = compile_specs(declare_mixed())
    with pytest.raises(TilefishBuildError, match=r"1 TF32 tensor-core MMA of its TTGIR is made of it: .* \(the dot at kernels\.py:36:18\)"):
        build(spec, arch=ARCH)
    compiler.ttgir_body = TTGIR_MIXED_CLEAN
    built = build(spec, arch=ARCH)  # the TF32 dot's MMAs only
    assert compiler.count == 2
    assert build(spec, arch=ARCH).sha256_code == built.sha256_code and compiler.count == 2  # a hit: stored by the build that was judged
    compiler.ttir = None
    with pytest.raises(TilefishBuildError, match="cannot be traced"):  # a compile with no TTIR to trace its MMAs to
        build(spec, arch=ARCH, cache=False)
