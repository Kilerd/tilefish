"""Where the generated source writes a rounding out as inline PTX, with no GPU (notes 1, 2, 16 and 17 of ``tilefish.lower``).

* ``fp_fusion=False``: every float ``+ - *`` (f32, bf16, f16) is one scalar ``.rn`` instruction, and so are the multiplies of ``silu`` and
  ``gelu``; every narrowing float cast is ``cvt.rn``. ``fp_fusion=True`` leaves the arithmetic to the compiler.
* In every kernel: a narrowing cast of a widened value (directly, or through ``where``) is ``cvt.rn``, out of the reach of the folds that
  would keep an input NaN's payload; ``-x`` gives the canonical NaN explicitly; a product that an ordered sum without ``fma`` adds is
  ``mul.rn``.

The bytes on a GPU: ``test_gpu_roundings.py``.
"""

import pytest

import tilefish
from tilefish import Dim, Tensor, bf16, f16, f32, i32

N = Dim("N")
R = Dim("R")
D = Dim("D", 96)


def test_fp_fusion_false_writes_each_float_op_as_one_rn_instruction():
    @tilefish.kernel(fp_fusion=False)
    def mix(a: Tensor[N, f32], b: Tensor[N, bf16], c: Tensor[N, f16], i: Tensor[N, i32]):
        return a * a + a - 1.5, b * b + b, c * c - c, i * i + i, tilefish.silu(a), tilefish.gelu(a)

    src = mix.lower().source
    for op in ("mul", "add", "sub"):
        assert f'"{op}.rn.f32 $0, $1, $2;"' in src
    for op in ("mul", "add"):
        assert f'"{op}.rn.bf16 $0, $1, $2;"' in src
    for op in ("mul", "sub"):
        assert f'"{op}.rn.f16 $0, $1, $2;"' in src
    assert "v6 = _tf_sub_rn_f32(v5, 1.5)" in src
    assert "_tf_mul_rn_f32(v0, tl.sigmoid(v0))" in src  # silu's multiply
    assert "_tf_mul_rn_f32(_tf_mul_rn_f32(0.5, v0), _tf_add_rn_f32(1.0, tl.math.erf(_tf_mul_rn_f32(v0, 0.7071067811865476))))" in src  # gelu's
    assert "v3 * v3" in src and "+ v3" in src  # integers stay Triton's
    assert "inline_asm" not in mix.lower(fp_fusion=True).source  # the compiler's choice


def test_fp_fusion_true_leaves_the_arithmetic_to_the_compiler():
    @tilefish.kernel
    def axpb(a: Tensor[N, f32], b: Tensor[N, f32], c: Tensor[N, f32]) -> Tensor[N, f32]:
        return tilefish.silu(a * b + c)

    src = axpb.lower().source
    assert "v3 = v0 * v1" in src and "v4 = v3 + v2" in src and "v5 = v4 * tl.sigmoid(v4)" in src and "inline_asm" not in src


@pytest.mark.parametrize("narrow", [bf16, f16])
def test_a_narrowing_cast_of_a_widened_value_is_inline_ptx_in_every_kernel(narrow):
    @tilefish.kernel
    def roundtrip(x: Tensor[N, bf16], y: Tensor[N, f16], m: Tensor[N, f32]):
        wx, wy = x.to(f32), y.to(f32)
        return (
            wx.to(narrow),  # fptrunc(fpext(x)): folded to x
            tilefish.where(m > 0, wx, m).to(narrow),  # a select of a widened value: narrowed into a select of x
            tilefish.where(m > 0, tilefish.where(m > 1, wy, 0.0), wx).to(narrow),  # through two
            (wx * 2.0).to(narrow),  # a product: no fold reaches the cast, Triton's .to()
            m.to(narrow),  # not widened: Triton's .to()
        )

    src = roundtrip.lower().source
    name = narrow.name
    helper = f"_tf_cvt_{name}"
    assert f'"cvt.rn.{name}.f32 $0, $1;"' in src
    lines = [ln.strip() for ln in src.splitlines() if "  # rounding point" in ln]
    assert len(lines) == 5
    assert all(helper in ln for ln in lines[:3]), lines
    assert all(f".to(tl.{'bfloat16' if name == 'bf16' else 'float16'})" in ln for ln in lines[3:]), lines


def test_a_product_that_an_ordered_sum_adds_is_rounded_first_whatever_fp_fusion_says():
    @tilefish.kernel
    def dot_plain(x: Tensor[R, D, f32], w: Tensor[D, f32]) -> Tensor[R, f32]:
        return tilefish.sum(x * w, -1, order=tilefish.tree(32))

    @tilefish.kernel
    def dot_fma(x: Tensor[R, D, f32], w: Tensor[D, f32]) -> Tensor[R, f32]:
        return tilefish.sum(x * w, -1, order=tilefish.tree(32), fma=True)

    @tilefish.kernel
    def tree_sum(x: Tensor[R, D, f32], w: Tensor[D, f32]) -> Tensor[R, f32]:
        return tilefish.sum(x * w, -1)

    assert '"mul.rn.f32 $0, $1, $2;"' in dot_plain.lower().source  # acc + round(x * w), each step one rounding (note 17)
    assert "_tf_mul_rn" not in dot_fma.lower().source  # fma(x, w, acc): the product is not computed on its own
    assert "_tf_mul_rn" not in tree_sum.lower().source and "x * w" not in tree_sum.lower().source  # Triton's tree: the compiler's choice


def test_neg_gives_the_canonical_nan_explicitly_in_every_kernel():
    @tilefish.kernel
    def negsel(a: Tensor[N, f32], b: Tensor[N, bf16]):
        return tilefish.where(a > 0, -a, a), -b

    src = negsel.lower().source
    assert "_tf_negc_f32(v0)" in src and "_tf_negc_bf16(v1)" in src
    assert '"selp.f32 $0, 0f7FFFFFFF, t, p; "' in src and '"selp.b16 $0, 0x7FFF, t, p; "' in src
    assert '"neg.f32 $0, $1;"' not in src
