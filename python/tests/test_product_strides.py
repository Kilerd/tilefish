"""Strides and offsets that are products of dimensions (``L*L`` for a dense ``[H, L, L]``), with no GPU.

The declared language stays affine: a shape, a grid or a ``//`` of two symbolic dimensions is refused. What tilefish derives from shapes, the strides
and the offsets of a tensor and of its views, may be a product of symbolic sizes (``tilefish.dims.mul``, a product atom), which a view, a
``Strided[...]`` argument and a plan carry symbolically and evaluate at a bucket. The gpu side: ``test_gpu_product_strides.py``.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import pickle

import broadcast_cases as bc
import gpu_kernels_p7 as p7
import pytest

import tilefish
from tilefish import Dim, Out, StrideOf, Strided, Tensor, TilefishTypeError, bf16, f32
from tilefish.dims import as_dim, dim_bounds, mul
from tilefish.types import row_major_strides

L = Dim("L")
M = Dim("M")
N = Dim("N")
L2 = Dim("L2", divisible_by=2)
L16 = Dim("L16", divisible_by=16)
Lb = Dim("Lb", buckets=[8, 24])


# -- the product atom -----------------------------------------------------------------------------------------------------------------


def test_a_product_of_the_same_factors_is_one_atom_whatever_their_order_and_their_integer_content():
    assert str(mul(L, L)) == "L*L"
    assert mul(L, M) == mul(M, L)
    assert mul(2 * L, L) == 2 * mul(L, L) == mul(L, 2 * L)
    assert mul(mul(L, M), N) == mul(L, mul(M, N)) and str(mul(mul(L, M), N)) == "L*M*N"
    assert str(mul(L + 6, M)) == "(L + 6)*M" and mul(2 * L + 12, M) == 2 * mul(L + 6, M)
    assert mul(4, L) == 4 * L and mul(L, 1) == L  # a static factor scales: affine stays affine
    assert mul(L, L) != mul(L, M) and hash(mul(L, M)) == hash(mul(M, L))


def test_the_declared_language_stays_affine():
    with pytest.raises(TilefishTypeError, match="affine"):
        L * L
    with pytest.raises(TilefishTypeError):
        Tensor[L * M, f32]


def test_a_product_has_a_value_at_every_binding():
    assert mul(L, L).subst({L: 8}) == 64
    assert mul(L, M).subst({L: 8}) == 8 * M
    assert (3 * mul(L, M) + L + 1).subst({L: 2, M: 5}) == 33
    assert mul(L + 6, L).subst({L: 2}) == 16
    assert mul(L, M).free_dims() == {L, M}
    assert dim_bounds(mul(Lb, Lb)) == (64, 576) and dim_bounds(mul(Lb, L))[1] is None


def test_a_product_is_divisible_by_what_its_factors_prove():
    assert mul(L16, L).is_divisible_by(16) and not mul(L16, L).is_divisible_by(32)
    assert mul(L16, L16).is_divisible_by(256)
    assert mul(L2, 3 * L2).is_divisible_by(12)
    assert not mul(L, M).is_divisible_by(2)


def test_a_product_pickles():
    for d in (mul(L, M), 2 * mul(L + 6, Lb) + 1):
        assert pickle.loads(pickle.dumps(d)) == d


# -- tensors and views ----------------------------------------------------------------------------------------------------------------


def test_a_dense_tensor_of_two_symbolic_sizes_has_its_strides(tr):
    assert [str(s) for s in row_major_strides(Tensor[4, L, L, f32].shape)] == ["L*L", "L", "1"]
    assert [str(s) for s in row_major_strides(Tensor[L, M, N, f32].shape)] == ["M*N", "N", "1"]
    x = tr.val(Tensor[4, L, L, f32])
    t = x.permute(0, 2, 1).type
    assert [str(s) for s in t.layout_strides()] == ["L*L", "1", "L"] and str(t.offset) == "0" and not t.contiguous
    h = x[2].type
    assert [str(s) for s in h.layout_strides()] == ["L", "1"] and str(h.offset) == "2*L*L"
    assert h.is_dense and not h.contiguous  # 2*L*L f32: not provably a multiple of 16 bytes


def test_the_offset_of_a_head_is_aligned_when_the_factors_prove_it(tr):
    assert tr.val(Tensor[4, L2, L2, f32])[1].type.contiguous  # L2*L2 is a multiple of 4: 16 bytes of f32
    assert not tr.val(Tensor[4, L2, L, f32])[1].type.contiguous  # L2*L: a multiple of 2 only
    assert tr.val(Tensor[3, L, L16, bf16])[1].type.contiguous  # L*L16 bf16: a multiple of 16 elements


def test_a_reshape_of_a_tensor_of_two_symbolic_sizes_is_counted_factor_by_factor(tr):
    x = tr.val(Tensor[4, L, L, f32])
    assert x.reshape(4 * L, L).type.shape == (4 * L, L)
    with pytest.raises(TilefishTypeError, match="as many elements"):
        x.reshape(4 * L, M)


Q = Dim("Q")


@tilefish.extern(triton="no_module:heads_last", grid=lambda Q: (Q,))
def heads_last(
    x: Strided[Tensor[Q, Q, 4, f32]], y: Out[Tensor[Q, Q, 4, f32]], s0: StrideOf("x", 0), s1: StrideOf("x", 1), s2: StrideOf("x", 2)
): ...


def test_a_strided_argument_binds_a_stride_that_is_a_product():
    @tilefish.graph
    def g(x: Tensor[4, Lb, Lb, f32]):
        return heads_last(x.permute(1, 2, 0))  # [Lb, Lb, 4] with the strides (Lb, 1, Lb*Lb)

    assert g.graph.nodes[-1].op == "heads_last"
    p = tilefish.plan(g)
    for lb in (8, 24):
        (step,) = p.steps(Lb=lb)
        (x,) = [a for a in step["args"] if a["name"] == "x"]
        assert x["view"] == {"shape": [lb, lb, 4], "strides": [lb, 1, lb * lb], "offset": 0}


# -- a plan --------------------------------------------------------------------------------------------------------------------------


def test_a_plan_has_buffers_whose_strides_are_products_and_evaluates_their_views_at_every_bucket():
    p = tilefish.plan(p7.attention_mask)
    shapes = {b["name"]: b["shape"] for b in p.buffers}
    assert shapes["t0"] == [4, "Lb", "Lb"] and shapes["head1"] == ["Lb", "Lb"] and shapes["row_max"] == ["4*Lb"]
    for lb in (8, 24):
        steps = p.steps(Lb=lb)
        assert [s.get("entry") for s in steps] == ["outer_heads", "key_mask_heads", "causal_heads", "contiguous_r3_f32_t32x64", f"row_max_lb.Lb{lb}", None]
        (x,) = [a for a in steps[3]["args"] if a["name"] == "x"]
        assert x["view"] == {"shape": [4, lb, lb], "strides": [lb * lb, 1, lb], "offset": 0}
        assert steps[5]["kind"] == "copy" and steps[5]["args"][0]["view"] == {"shape": [lb, lb], "strides": [lb, 1], "offset": lb * lb}
        assert p.shape_of("t0", Lb=lb) == (4, lb, lb)
    assert {b["name"]: b["bytes"] for b in p.buffers}["t0"] == 4 * 24 * 24 * 4


def test_the_plan_of_a_product_stride_is_a_function_of_the_graph():
    assert tilefish.plan(p7.attention_mask).digest == tilefish.plan(p7.attention_mask).digest


# -- the lowering: broadcasts along any axis (note 12) ------------------------------------------------------------------------------------


def test_an_operand_broadcast_along_a_middle_axis_is_read_at_the_offset_of_its_coordinates():
    low = bc.key_mask_heads.lower()
    assert low.template == "elementwise"
    assert "v1 = tl.load(keep + ((offs // (size_L * size_L)) * size_L + offs % size_L), mask=mask, other=0)  # keep" in low.source
    assert "'keep' [4, 1, L] broadcasts over the axes where it has size 1" in low.source
    assert "tl.load(r + (offs // size_D), mask=mask, other=0)" in bc.scale_each_row.lower().source
    assert "tl.load(z + ((offs // (size_T * size_D)) * size_D + offs % size_D), mask=mask, other=0)" in bc.middle_rows.lower().source
    assert "tl.load(c + ((offs // size_D) % size_T), mask=mask, other=0)" in bc.column_add.lower().source
    outer = bc.outer_heads.lower().source
    assert "tl.load(q + (offs // size_L)" in outer and "tl.load(k + ((offs // (size_L * size_L)) * size_L + offs % size_L)" in outer


def test_an_iota_along_any_axis_is_the_lanes_coordinate():
    src = bc.causal_heads.lower().source
    assert "v1 = offs % size_L" in src and "v2 = (offs // size_L) % size_L" in src


def test_the_sizes_a_coordinate_needs_are_runtime_scalars_of_the_dimensions():
    (spec,) = bc.middle_rows.lower().specs()
    scalars = {a.name: (a.dim, a.count_of) for a in spec.args if a.kind == "scalar"}
    assert scalars["size_T"] == ("T", None) and scalars["size_D"] == ("D", None) and scalars["n"][1] == "x"
    (spec,) = bc.key_mask_heads.lower().specs()
    scalars = {a.name: (a.dim, a.count_of) for a in spec.args if a.kind == "scalar"}
    assert scalars["size_L"] == ("L", None) and "keep_n" not in scalars  # a broadcast along the middle has no count


def test_a_static_size_is_a_number_and_needs_no_scalar():
    @tilefish.kernel
    def heads(s: Tensor[4, 16, 32, f32], keep: Tensor[4, 1, 32, f32]) -> Tensor[4, 16, 32, f32]:
        return s * keep

    src = heads.lower().source
    assert "tl.load(keep + ((offs // 512) * 32 + offs % 32), mask=mask, other=0)" in src and "size_" not in src


def test_the_golden_source_of_a_middle_broadcast():
    from test_lower import check_golden

    check_golden("key_mask_heads", bc.key_mask_heads.lower().source)


def test_as_dim_of_a_product_is_the_product():
    assert as_dim(mul(L, M)) == mul(L, M)
