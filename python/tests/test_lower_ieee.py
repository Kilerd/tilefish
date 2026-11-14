"""IEEE math (``@tilefish.kernel(math="ieee")`` and ``math="ieee"`` at an op): declarations, traces and the generated source, with no GPU.

Note 13 of ``tilefish.lower``: in IEEE math ``/`` is ``div.rn.f32`` and ``sqrt`` ``sqrt.rn.f32`` (IEEE 754, no flush to zero), ``exp`` is CUDA's
``expf``, ``silu`` torch's ``x / (1 + expf(-x))``, ``rsqrt`` ``rsqrtf`` and ``sin`` keeps a subnormal, each written out as the inline PTX nvcc
emits. The bytes against torch: ``test_gpu_ieee.py``.
"""

import ieee_cases as ic
import pytest
from test_lower import check_golden

import tilefish
from tilefish import Dim, Tensor, TilefishDeclarationError, TilefishLoweringError, TilefishTypeError, bf16, f32
from tilefish.aot import compile_specs

N = Dim("N")


def test_the_golden_source():
    check_golden("math_ieee", ic.math_ieee.lower().source)


def test_a_kernel_declares_its_math():
    assert ic.math_ieee.math == "ieee" and ic.math_fast.math == "fast"
    with pytest.raises(TilefishDeclarationError, match="math must be one of"):

        @tilefish.kernel(math="precise")
        def bad(x: Tensor[N, f32]) -> Tensor[N, f32]:
            return x


def test_ieee_math_writes_out_the_ptx_nvcc_emits():
    src = ic.math_ieee.lower().source
    assert "# math: ieee (note 13:" in src
    for helper in ("_tf_exp_ieee", "_tf_sqrt_ieee", "_tf_rsqrt_ieee", "_tf_sin_ieee", "_tf_silu_ieee"):
        assert f"def {helper}(x):" in src
    assert '"sqrt.rn.f32 $0, $1;"' in src and '"rsqrt.approx.f32 $0, $1;"' in src
    assert '"fma.rm.f32 t2, t1, 0f437C0000, 0f4B400001; "' in src and '"mul.rn.f32 $0, t6, t5; "' in src  # expf
    assert '"fma.rn.f32 t7, t6, t5, 0f3F800000; "' in src and '"div.rn.f32 $0, $1, t7; "' in src  # silu: torch's contraction, IEEE division
    assert "tl.where(tl.abs(x) < 1.1754943508222875e-38, x, tl.sin(x))" in src
    ptx = [line for line in src.splitlines() if line.strip().startswith('"')]
    assert all("ftz" not in line or "ex2.approx.ftz.f32" in line for line in ptx)  # nothing flushes but the ex2 of r in [-1, 1], as in expf
    div = ic.div_ieee.lower().source
    assert '"div.rn.f32 $0, $1, $2;"' in div and "v2 = _tf_div_ieee(v0, v1)" in div
    assert "_tf_div_ieee(v0, 3.0)" in ic.div_by_three_ieee.lower().source  # a constant divisor is divided by


def test_the_default_math_is_unchanged():
    src = ic.math_fast.lower().source
    assert "tl.exp(v0)" in src and "tl.sqrt_rn(v0)" in src and "tl.rsqrt(v0)" in src and "tl.sin(v0)" in src and "v0 * tl.sigmoid(v0)" in src
    assert "inline_asm" not in src and "math: ieee" not in src


def test_an_op_says_its_own_math():
    src = ic.per_op.lower().source
    assert "v2 = _tf_div_ieee(v0, v1)  # math: ieee" in src and "v3 = _tf_exp_ieee(v0)  # math: ieee" in src
    assert "v4 = v0 / v1" in src and "v5 = tl.exp(v0)" in src
    assert "# math: ieee (note 13" not in src  # the kernel's math is the default

    @tilefish.kernel(math="ieee")
    def fast_in_ieee(x: Tensor[N, f32]) -> Tensor[N, f32]:
        return tilefish.exp(x, math="fast") + tilefish.exp(x)

    src = fast_in_ieee.lower().source
    assert "tl.exp(v0)" in src and "_tf_exp_ieee(v0)" in src


def test_the_math_of_an_op_is_in_its_node(tr):
    x = tr.val(Tensor[N, f32])
    tilefish.exp(x, math="ieee")
    tilefish.div(x, x, math="fast")
    _ = x / x
    tilefish.silu(x)
    assert [n.attrs.get("math") for n in tr.ctx.nodes] == ["ieee", "fast", None, None]
    with pytest.raises(TilefishTypeError, match="math must be one of"):
        tilefish.exp(x, math="IEEE")
    with pytest.raises(TilefishTypeError, match="floating-point"):
        tilefish.div(tr.val(Tensor[N, tilefish.i32]), 2, math="ieee")


def test_lower_can_override_the_math():
    assert "_tf_exp_ieee" in tilefish.lower(ic.math_fast, math="ieee").source
    assert "_tf_exp_ieee" not in tilefish.lower(ic.math_ieee, math="fast").source
    with pytest.raises(TilefishLoweringError) as e:
        tilefish.lower(ic.math_fast, math="exact")
    assert e.value.rule == "math"


def test_the_math_is_part_of_the_binarys_identity():
    fast, ieee = compile_specs(ic.math_fast.lower().extern), compile_specs(tilefish.lower(ic.math_fast, math="ieee").extern)
    assert fast[0].source_sha256 != ieee[0].source_sha256


def test_ieee_math_is_f32_only():
    @tilefish.kernel(math="ieee")
    def half(x: Tensor[N, bf16]) -> Tensor[N, bf16]:
        return tilefish.exp(x)

    with pytest.raises(TilefishLoweringError) as e:
        half.lower()
    assert e.value.rule == "math-dtype"


def test_the_row_template_takes_ieee_math():
    low = ic.softmax_ieee.lower()
    assert low.template == "row" and "_tf_exp_ieee(" in low.source and "_tf_div_ieee(" in low.source
    low = ic.periodic.lower()
    assert low.template == "row" and "_tf_div_ieee(1.0, " in low.source and "_tf_sin_ieee(" in low.source
    assert "fp_fusion: False" in low.source
