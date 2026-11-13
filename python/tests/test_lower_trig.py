"""``sin`` and ``cos``: traced like the other math functions, f32 only, lowered to ``tl.sin`` / ``tl.cos`` (bytes: ``test_gpu_trig.py``)."""

import pytest
import trig_cases as tc
from test_lower import check_golden

import tilefish
from tilefish import Dim, Tensor, TilefishLoweringError, bf16, f32

N = Dim("N")


def test_sin_and_cos_trace_like_the_other_math_functions(tr):
    x = tr.val(Tensor[N, f32])
    assert tilefish.sin(x).type == Tensor[N, f32] and tilefish.cos(x).type == Tensor[N, f32]
    assert [n.op for n in tr.ctx.nodes] == ["sin", "cos"]


def test_golden_source():
    check_golden("rope_tables", tc.rope_tables.lower().source)


def test_they_lower_to_tl_sin_and_tl_cos():
    src = tc.sin_cos.lower().source
    assert "v1 = tl.sin(v0)" in src and "v2 = tl.cos(v0)" in src
    rope = tc.rope_tables.lower()
    assert rope.template == "row" and "tl.cos(" in rope.source and "tl.sin(" in rope.source


def test_they_are_f32_only():
    @tilefish.kernel
    def half(x: Tensor[N, bf16]) -> Tensor[N, bf16]:
        return tilefish.cos(x)

    with pytest.raises(TilefishLoweringError) as e:
        half.lower()
    assert e.value.rule == "math-dtype"
