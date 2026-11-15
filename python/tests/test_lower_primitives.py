"""``erf``, ``sin_approx``, ``rcp``, ``neg`` (``-x``), ``fma`` and ordered sums: traces, refusals and the generated source, with no GPU (notes 15 to
17 of ``tilefish.lower``). The bytes against CUDA C++: ``test_gpu_primitives.py`` and ``test_gpu_reduction_order.py``."""

import primitive_cases as pc
import pytest
from test_lower import check_golden

import tilefish
from tilefish import Dim, Tensor, TilefishLoweringError, TilefishTypeError, bf16, f16, f32, i32, i64, u8
from tilefish.aot import compile_specs

N = Dim("N")
R = Dim("R")
D = Dim("D", 96)


def test_the_ops_trace(tr):
    x, y = tr.val(Tensor[N, f32]), tr.val(Tensor[N, f32])
    for out in (tilefish.erf(x), tilefish.erf(x, math="ieee"), tilefish.sin_approx(x), tilefish.rcp(x), -x, tilefish.neg(x), tilefish.fma(x, y, 1.0)):
        assert out.type == Tensor[N, f32]
    assert [(n.op, n.attrs.get("math")) for n in tr.ctx.nodes] == [
        ("erf", None), ("erf", "ieee"), ("sin_approx", None), ("rcp", None), ("neg", None), ("neg", None), ("fma", None)
    ]  # fmt: skip
    assert -tr.val(Tensor[N, i32]) is not None  # an integer negates too
    with pytest.raises(TilefishTypeError, match="must be a floating-point"):
        tilefish.rcp(tr.val(Tensor[N, i32]))
    with pytest.raises(TilefishTypeError, match="must be a floating-point"):
        tilefish.fma(x, y, tr.val(Tensor[N, i32]))
    with pytest.raises(TilefishTypeError, match="mask"):
        -(x > 0)


def test_tree_orders_are_checked():
    t = tilefish.tree(128, offsets=(16, 8, 4, 2, 1, 64, 32), vector=8)
    assert (t.lanes, t.vector, t.offsets) == (128, 8, (16, 8, 4, 2, 1, 64, 32))
    assert str(t) == "tree(lanes=128, vector=8, offsets=(16, 8, 4, 2, 1, 64, 32))"
    assert tilefish.tree(256).offsets == (128, 64, 32, 16, 8, 4, 2, 1)  # the halving tree by default
    assert tilefish.tree(1, vector=4) == tilefish.tree(1) and str(tilefish.tree(1)) == "sequential"
    for bad in (dict(lanes=3), dict(lanes=2048), dict(lanes=0), dict(lanes=True), dict(lanes=4, vector=32), dict(lanes=4, vector=3)):
        with pytest.raises(TilefishTypeError, match="power of two"):
            tilefish.tree(**bad)
    for offsets in ((1, 2), (2, 1, 1), (4, 2, 1, 8), (1, 2, 3), "21", (1.0, 2)):
        with pytest.raises(TilefishTypeError, match="offsets"):
            tilefish.tree(8, offsets=offsets)


def test_an_ordered_sum_traces_its_order(tr):
    x, w = tr.val(Tensor[R, D, f32]), tr.val(Tensor[D, f32])
    a = tilefish.sum(x, -1, order="sequential")
    b = x.sum(-1, order=tilefish.tree(32))
    c = tilefish.sum(x * w, -1, order=tilefish.tree(32), fma=True)
    assert a.type == b.type == c.type == Tensor[R, f32]
    sums = [n for n in tr.ctx.nodes if n.op == "sum"]
    assert [str(n.attrs["order"]) for n in sums] == ["sequential", str(tilefish.tree(32)), str(tilefish.tree(32))]
    assert [n.attrs.get("fma", False) for n in sums] == [False, False, True]
    assert "order=sequential" in str(sums[0])
    with pytest.raises(TilefishTypeError, match="order must be"):
        tilefish.sum(x, -1, order="tree")
    with pytest.raises(TilefishTypeError, match="needs an explicit order"):
        tilefish.sum(x * w, -1, fma=True)
    with pytest.raises(TilefishTypeError, match="takes a product a \\* b"):
        tilefish.sum(x + w, -1, order="sequential", fma=True)
    with pytest.raises(TilefishTypeError, match="takes a product a \\* b"):
        tilefish.sum(x, -1, order="sequential", fma=True)
    with pytest.raises(TilefishTypeError, match="fma must be True or False"):
        tilefish.sum(x, -1, order="sequential", fma=1)


def test_golden_sources():
    check_golden("primitives_ieee", pc.erf_ieee.lower().source)
    check_golden("sum_orders", tilefish.lower(pc.sumsq_shfl128v8).source)
    check_golden("sum_sequential", tilefish.lower(pc.sum_sequential).source)


def test_the_generated_source_says_each_instruction():
    src = {k: getattr(pc, k).lower().source for k in ("erf_fast", "erf_ieee", "sin_approx", "rcp", "neg_f32", "neg_bf16", "neg_f16", "fma3")}
    assert "v1 = tl.math.erf(v0)" in src["erf_fast"] and "inline_asm" not in src["erf_fast"]
    assert '"setp.ltu.f32 p0, t0, 0f3F8060FE; "' in src["erf_ieee"] and '"copysign.f32 t7, $1, t7; "' in src["erf_ieee"]
    assert '"sin.approx.f32 $0, $1;"' in src["sin_approx"]
    assert '"rcp.rn.f32 $0, $1;"' in src["rcp"]
    assert '"neg.f32 $0, $1;"' in src["neg_f32"] and '"neg.bf16 $0, $1;"' in src["neg_bf16"] and '"neg.f16 $0, $1;"' in src["neg_f16"]
    assert "dtype=tl.bfloat16" in src["neg_bf16"] and '"=h,h"' in src["neg_bf16"]
    assert '"fma.rn.f32 $0, $1, $2, $3;"' in src["fma3"] and "v3 = _tf_fma(v0, v1, v2)" in src["fma3"]
    scalar = pc.fma_scalar.lower().source
    assert "_tf_fma(v0, 0.1, v1)" in scalar and "fp_fusion: False" in scalar  # one rounding whatever fp_fusion says


def test_integer_negation_is_zero_minus_x():
    @tilefish.kernel
    def negate(x: Tensor[N, i64]) -> Tensor[N, i64]:
        return -x

    src = negate.lower().source
    assert "v1 = -v0" in src and "inline_asm" not in src


@pytest.mark.parametrize(
    "dtype,body,rule",
    [
        (bf16, lambda x: tilefish.rcp(x), "math-dtype"),
        (f16, lambda x: tilefish.sin_approx(x), "math-dtype"),
        (bf16, lambda x: tilefish.erf(x, math="ieee"), "math-dtype"),
        (bf16, lambda x: tilefish.fma(x, x, x), "math-dtype"),
        (u8, lambda x: -x, "arithmetic-dtype"),
    ],
)
def test_the_dtypes_each_op_is_lowered_for(dtype, body, rule):
    @tilefish.kernel
    def k(x: Tensor[N, dtype]) -> Tensor[N, dtype]:
        return body(x)

    with pytest.raises(TilefishLoweringError) as e:
        k.lower()
    assert e.value.rule == rule


def test_an_ordered_sum_is_a_helper_of_its_order():
    src = tilefish.lower(pc.sumsq_shfl128v8).source
    assert "def _tf_sum_fma_l128_v8_o16_8_4_2_1_64_32(a, b, D: tl.constexpr, BLOCK: tl.constexpr):" in src
    assert "for s in range((BLOCK + 1023) // 1024 * 8):" in src and "col = (s // 8) * 1024 + lane * 8 + s % 8" in src
    assert "p = tl.where(col < D, _tf_fma(tl.gather(a, k, 0), tl.gather(b, k, 0), p), p)" in src
    assert src.count("tl.split(") == 7 and "tl.reshape(p, (4, 2, 16))" in src  # o = 16 first: lanes j and j + 16 of each 32
    assert "_tf_sum_fma_l128_v8_o16_8_4_2_1_64_32(v0, v0, D, BLOCK_D)" in src  # the product's operands
    seq = tilefish.lower(pc.sum_sequential).source
    assert "for s in range(BLOCK):" in seq and "tl.split" not in seq
    halving = tilefish.lower(pc.sum_halving256).source
    assert "tl.reshape(p, (1, 2, 128))" in halving and "tl.reshape(p, (1, 2, 1))" in halving


def test_offsets_are_compacted_in_the_order_they_are_taken():
    @tilefish.kernel
    def k(x: Tensor[R, D, f32]) -> Tensor[R, f32]:
        return tilefish.sum(x, -1, order=tilefish.tree(8, offsets=(2, 4, 1)))

    src = k.lower().source
    # o = 2 of 8 lanes; then o = 4 is 2 in the 4 lanes left (bit 2 taken out); then o = 1 stays 1
    assert "tl.reshape(p, (2, 2, 2))" in src and "tl.reshape(p, (1, 2, 2))" in src and "tl.reshape(p, (1, 2, 1))" in src


def test_an_ordered_sum_needs_the_row_template_and_f32():
    @tilefish.kernel
    def half(x: Tensor[R, D, bf16]) -> Tensor[R, bf16]:
        return tilefish.sum(x, -1, order="sequential")

    with pytest.raises(TilefishLoweringError) as e:
        half.lower()
    assert e.value.rule == "reduction-dtype"

    @tilefish.kernel
    def flat(x: Tensor[N, f32]) -> Tensor[f32]:
        return tilefish.sum(x, order="sequential")

    with pytest.raises(TilefishLoweringError):
        flat.lower()


def test_the_order_changes_the_binary():
    a = tilefish.lower(pc.sum_halving256).source
    b = tilefish.lower(pc.sum_sequential).source
    assert compile_specs(tilefish.lower(pc.sum_halving256).extern)[0].source_sha256 != compile_specs(tilefish.lower(pc.sum_sequential).extern)[0].source_sha256
    assert a != b


def test_a_mean_and_a_max_have_no_order(tr):
    x = tr.val(Tensor[R, D, f32])
    with pytest.raises(TypeError):
        tilefish.mean(x, -1, order="sequential")
    assert -tr.val(Tensor[R, D, i64]) is not None


def test_a_product_only_fma_sums_read_is_not_computed_and_one_read_elsewhere_is():
    assert " = v0 * v0" not in tilefish.lower(pc.sumsq_shfl128v8).source

    @tilefish.kernel
    def k(x: Tensor[R, D, f32], w: Tensor[D, f32]) -> Tensor[R, D, f32]:
        p = x * w
        return p + tilefish.sum(p, -1, keepdims=True, order="sequential", fma=True)

    src = k.lower().source
    assert "v2 = v0 * v1" in src and "_tf_sum_fma_seq(v0, v1, D, BLOCK_D)" in src
