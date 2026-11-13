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


def test_a_long_row_is_walked_in_chunks_with_the_combine_of_the_order():
    low = ac.argmax_bf16.lower()
    src = low.source
    assert low.num_warps == 2 and low.bucket_num_warps == {}
    assert "def _tf_argmax_float(v1, i1, v2, i2):" in src
    assert "win = tl.where(n1 != n2, n1, gt | ((~gt) & (~lt) & (i1 < i2)))" in src
    assert "v0_bv = tl.full([BLOCK_D], float(\"-inf\"), tl.float32)" in src and "v0_bi = tl.full([BLOCK_D], 2147483647, tl.int32)" in src
    assert "for c0 in tl.static_range(0, D, BLOCK_D):" in src
    assert "v0_x = v1.to(tl.float32)  # exact" in src
    assert "v0_bv, v0_bi = _tf_argmax_float(v0_bv, v0_bi, tl.where(mask, v0_x, float(\"-inf\")), tl.where(mask, cols, 2147483647))" in src
    assert "v0_v, v0_i = tl.reduce((v0_bv, v0_bi), 0, _tf_argmax_float)" in src and "v0 = v0_i.to(tl.int64)" in src
    assert "tl.argmax" not in src and "tl.max(" not in src  # Triton's own argmax has another rule for a NaN
    (spec,) = low.specs()
    assert spec.constants[(3,)] == 1024 and spec.num_warps == 2  # BLOCK_D: one chunk


def test_everything_per_element_is_computed_chunk_by_chunk():
    src = ac.greedy_token.lower().source
    body = src.split("for c0 in tl.static_range(0, D, BLOCK_D):", 1)[1].split("v0_v, v0_i = tl.reduce", 1)[0]
    assert "tl.load(logits + offs" in body and "tl.load(bias + cols" in body and "v6 = v5.to(tl.bfloat16)  # rounding point" in body


def test_a_short_or_bucketed_row_is_one_block():
    src = ac.arg_f16.lower().source
    assert "static_range" not in src and "def _tf_argmin_float(v1, i1, v2, i2):" in src
    assert "v1_v, v1_i = tl.reduce((tl.where(mask, v1_x, float(\"-inf\")), tl.where(mask, cols, 2147483647)), 0, _tf_argmax_float)" in src
    assert 'tl.where(mask, v2_x, float("inf"))' in src and "_tf_argmin_float)" in src


def test_argmax_of_integers_masks_with_the_bounds_of_their_type():
    src = ac.arg_ints.lower().source
    assert "tl.where(mask, v0, -2147483648)" in src and "tl.where(mask, v0, 2147483647)" in src
    assert "tl.cast(-9223372036854775808, tl.int64)" in src and "tl.cast(9223372036854775807, tl.int64)" in src
    assert "tl.cast(0, tl.uint8)" in src and "tl.cast(255, tl.uint8)" in src
    assert "win = (v1 > v2) | ((v1 == v2) & (i1 < i2))" in src and "win = (v1 < v2) | ((v1 == v2) & (i1 < i2))" in src
    assert "n1 = v1 != v1" not in src


def test_a_kernel_with_a_sum_is_not_chunked():
    @tilefish.kernel
    def with_sum(x: Tensor[M, ac.C, f32]) -> tuple[Tensor[M, i32], Tensor[M, f32]]:
        return tilefish.argmax(x, dtype=i32), tilefish.sum(x, axis=-1)

    @tilefish.kernel
    def depends(x: Tensor[M, ac.C, f32]) -> Tensor[M, ac.C, f32]:
        best = tilefish.argmax(x, dtype=i32, keepdims=True)
        return tilefish.where(tilefish.iota(x, axis=-1) == best, x, 0.0)

    for k in (with_sum, depends):
        low = k.lower()
        assert "static_range" not in low.source and low.num_warps == 16, k.name


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
