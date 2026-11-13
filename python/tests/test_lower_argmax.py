"""``argmax`` and ``argmin``: what they trace to, what they refuse, and the Triton they lower to (bytes: ``test_gpu_argmax.py``)."""

import argmax_cases as ac
import pytest
from test_lower import check_golden

import tilefish
from tilefish import Dim, Tensor, TilefishLoweringError, TilefishTypeError, bf16, f32, i32, i64, u8
from tilefish.aot import compile_specs

M = Dim("M")
D = Dim("D", 128)


def test_argmax_traces_to_one_node_with_its_index_type(tr):
    x = tr.val(Tensor[M, D, bf16])
    r = tilefish.argmax(x, axis=-1, dtype=i64)
    assert r.type == Tensor[M, i64]
    (node,) = tr.ctx.nodes
    assert node.op == "argmax" and node.attrs == {"axes": (1,), "keepdims": False, "dtype": i64}
    assert tilefish.argmin(x, 1, dtype=i32, keepdims=True).type == Tensor[M, 1, i32]
    assert x.argmax(0, dtype=i32).type == Tensor[D, i32]


def test_argmax_needs_its_index_type_and_one_axis(tr):
    x = tr.val(Tensor[M, D, f32])
    for bad, match in (
        (lambda: tilefish.argmax(x), "say the index type"),
        (lambda: x.argmin(), "say the index type"),
        (lambda: tilefish.argmax(x, dtype=u8), "dtype must be i32 or i64"),
        (lambda: tilefish.argmax(x, axis=(0, 1), dtype=i32), "axis must be one int"),
        (lambda: tilefish.argmax(x, axis=2, dtype=i32), "out of range"),
        (lambda: tilefish.argmax(x, dtype=i32, keepdims=1), "keepdims must be True or False"),
        (lambda: tilefish.argmax(x > 0.0, dtype=i32), "is the mask"),
    ):
        with pytest.raises(TilefishTypeError, match=match):
            bad()


def test_golden_source():
    check_golden("argmax_bf16", ac.argmax_bf16.lower().source)
    check_golden("arg_ints", ac.arg_ints.lower().source)


def test_argmax_of_a_half_type_widens_to_f32_and_tests_for_a_nan():
    src = ac.argmax_bf16.lower().source
    assert "v1_x = v0.to(tl.float32)  # exact" in src
    assert "v1_nan = (v1_x != v1_x) & mask" in src
    assert 'v1_best = tl.max(tl.where(mask, v1_x, float("-inf")), axis=0)' in src
    assert "v1 = tl.min(tl.where(v1_hit, cols, BLOCK_D), axis=0).to(tl.int64)" in src
    assert "tl.argmax" not in src and "tl.argmin" not in src  # Triton's own rule for a NaN is not torch's
    arg = ac.argmin_bf16.lower().source
    assert 'tl.min(tl.where(mask, v1_x, float("inf")), axis=0)' in arg and "tl.store(out + row, v1)" in arg


def test_argmax_of_integers_masks_with_the_bounds_of_their_type():
    src = ac.arg_ints.lower().source
    assert "tl.where(mask, v0, -2147483648)" in src and "tl.where(mask, v0, 2147483647)" in src
    assert "tl.cast(-9223372036854775808, tl.int64)" in src and "tl.cast(9223372036854775807, tl.int64)" in src
    assert "tl.cast(0, tl.uint8)" in src and "tl.cast(255, tl.uint8)" in src
    assert "_nan" not in src


def test_argmax_lowers_to_the_row_template_only():
    assert ac.arg_f16.lower().template == "row"
    with pytest.raises(TilefishLoweringError) as e:
        tilefish.lower(ac.argmax_bf16, template="elementwise")
    assert e.value.rule == "elementwise.reduction"

    @tilefish.kernel
    def first_axis(x: Tensor[M, D, f32]) -> Tensor[D, i32]:
        return tilefish.argmax(x, axis=0, dtype=i32)

    with pytest.raises(TilefishLoweringError) as e:
        first_axis.lower()
    assert any(c.rule == "row.last-axis-reduction" for c in e.value.causes)


def test_a_bucketed_row_length_gives_a_spec_per_bucket():
    specs = compile_specs(ac.arg_f16.lower().extern)
    assert [s.stem for s in specs] == ["arg_f16.W7", "arg_f16.W96", "arg_f16.W1000"]
    assert [s.num_warps for s in specs] == [1, 1, 4]


def test_argmax_of_a_computed_row():
    src = ac.greedy_token.lower().source
    assert "v5 = v4.to(tl.bfloat16)  # rounding point" in src and "v6_x = v5.to(tl.float32)" in src
