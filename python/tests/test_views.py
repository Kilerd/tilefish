"""Strided views in the type system: ``TensorType`` strides and offset, and the zero-copy view operations ``reshape``, ``permute``, ``select`` (``x[i]``)
and ``split`` on traced values.

A view's type carries its element strides and element offset as dimension expressions. ``contiguous`` means dense **and** provably starting on a
16-byte boundary: such a tensor can be passed where a ``Tensor[...]`` is wanted. Any other view is a ``View[...]``.
"""

import copy
import pickle

import pytest

import tilefish
from tilefish import BlockScaled, Dim, Tensor, TensorType, TilefishTypeError, bf16, e4m3, f32, ue8m0
from tilefish.types import row_major_strides

M = Dim("M")
S = Dim("S", divisible_by=8)
A = Dim("A")
B = Dim("B")


def layout(v):
    t = v.type
    return [str(s) if s is not None else None for s in t.layout_strides()], None if t.offset is None else str(t.offset), t.contiguous


# -- the type -----------------------------------------------------------------------------------------------------------------------------


def test_a_fresh_tensor_is_dense_at_offset_zero_and_contiguous():
    t = Tensor[M, 8, 64, bf16]
    assert t.strides is None and str(t.offset) == "0" and t.contiguous and t.is_dense
    assert [str(s) for s in t.layout_strides()] == ["512", "64", "1"]
    assert t.layout_text() == "strides (512, 64, 1), offset 0"


def test_row_major_strides_of_symbolic_shapes_are_unknown_where_they_are_not_affine():
    assert [str(s) for s in row_major_strides(Tensor[M, 4, 8, f32].shape)] == ["32", "8", "1"]
    # A * B is not an affine expression: the stride of the first axis of [X, A, B] is unknown, the others are not
    strides = row_major_strides(Tensor[M, A, B, f32].shape)
    assert strides[0] is None and str(strides[1]) == "B" and str(strides[2]) == "1"


def test_explicit_strides_equal_to_the_row_major_ones_are_the_default():
    same = TensorType([M, 4, 8], f32, strides=[32, 8, 1])
    assert same.strides is None and same.contiguous
    view = TensorType([M, 4, 8], f32, strides=[64, 8, 1])
    assert view.strides is not None and not view.contiguous and repr(view) == "View[M, 4, 8, f32]"
    # the stride of a size-1 axis never matters
    assert TensorType([1, 8], f32, strides=[1000, 1]).contiguous


def test_a_dense_view_that_does_not_start_on_16_bytes_is_not_contiguous():
    assert TensorType([4, 8], bf16, offset=8).contiguous  # 16 bytes in
    assert not TensorType([4, 8], bf16, offset=4).contiguous  # 8 bytes in
    assert not TensorType([4, 8], bf16, offset=None).contiguous  # unknown
    assert TensorType([S, 8], bf16, offset=2 * S).contiguous  # 2 * 8k elements: a multiple of 16 bytes
    assert not TensorType([M, 8], bf16, offset=M).contiguous


def test_equality_is_by_shape_dtype_and_contiguity_not_layout():
    a = TensorType([4, 8], f32, strides=[16, 1])
    b = TensorType([4, 8], f32, strides=[32, 1], offset=3)
    assert a == b and hash(a) == hash(b) and not a.same_layout(b)
    assert a == Tensor[4, 8, f32].as_view() and a.same_layout(Tensor[4, 8, f32].as_view())  # an unknown layout agrees with anything
    assert a != Tensor[4, 8, f32]


def test_types_survive_copy_and_pickle_with_their_layout():
    for t in (
        Tensor[M, 8, bf16],
        TensorType([M, 8], bf16, strides=[2 * M + 8, 1], offset=M),
        TensorType([4, 8], f32, strides=[16, 1], offset=6),
        Tensor[4, 8, f32].as_view(),
        TensorType([1, 8], f32, strides=[1000, 1]),
    ):
        for other in (copy.copy(t), copy.deepcopy(t), pickle.loads(pickle.dumps(t))):
            assert other == t and other.contiguous == t.contiguous and other.strides == t.strides and other.offset == t.offset


def test_a_devscalar_is_a_rank_0_type_of_its_own():
    s = tilefish.DevScalar[tilefish.i32]
    assert s.devscalar and s.shape == () and s.contiguous and repr(s) == "DevScalar[i32]"
    assert s != Tensor[tilefish.i32] and Tensor[tilefish.i32].shape == ()
    assert pickle.loads(pickle.dumps(s)).devscalar
    with pytest.raises(tilefish.TilefishDeclarationError, match="scalar dtype"):
        tilefish.DevScalar[M]
    with pytest.raises(tilefish.TilefishDeclarationError, match="read-only"):
        tilefish.InOut[s]


# -- the view operations --------------------------------------------------------------------------------------------------------------


def test_reshape_of_a_dense_tensor_is_dense_and_keeps_the_offset(tr):
    x = tr.val(Tensor[M, 3 * 8 * 64, bf16])
    y = x.reshape(M, 3, 8, 64)
    assert y.type == Tensor[M, 3, 8, 64, bf16] and y.type.contiguous
    assert [n.op for n in tr.nodes] == ["reshape"] and tr.nodes[0].attrs["shape"] == (M, 3, 8, 64)
    assert y.reshape((M, 24, 64)).type == Tensor[M, 24, 64, bf16] and y.reshape(M * 1, 1536).type.shape[1] == 1536
    # a chunk at an unaligned offset stays a view: its offset is carried through the reshape
    c = tr.val(Tensor[4, 6, bf16]).split([1, 3], axis=0)[1]  # 6 elements = 12 bytes in
    assert not c.type.contiguous and not c.reshape(18).type.contiguous and str(c.reshape(18).type.offset) == "6"


def test_reshape_must_keep_the_number_of_elements_and_needs_a_dense_input(tr):
    x = tr.val(Tensor[M, 64, bf16])
    with pytest.raises(TilefishTypeError, match=r"cannot be shown to have as many elements as \['M', '8', '9'\]"):
        x.reshape(M, 8, 9)
    with pytest.raises(TilefishTypeError, match="needs the new shape"):
        x.reshape()
    with pytest.raises(TilefishTypeError, match="there is no -1 inference"):
        x.reshape(M, -1)
    with pytest.raises(TilefishTypeError, match="the shape must be ints or dimensions"):
        x.reshape(M, "64")
    view = tr.val(Tensor[M, 128, bf16]).split(64, axis=1)[0]
    with pytest.raises(TilefishTypeError, match=r"strides \(128, 1\), offset 0\); only a dense tensor can be reshaped without a copy"):
        view.reshape(M, 8, 8)
    assert (view + 0).reshape(M, 8, 8).type.contiguous


def test_reshape_compares_products_of_symbolic_dimensions_factor_by_factor(tr):
    x = tr.val(Tensor[A, B, 4, bf16])
    assert x.reshape(A, B * 2, 2).type.shape[1] == B * 2  # A * B * 4 = A * (2B) * 2
    assert x.reshape(A, 4 * B).type.shape == (A, 4 * B)
    with pytest.raises(TilefishTypeError, match="cannot be shown"):
        x.reshape(A, B, 5)


def test_permute_moves_the_strides_with_the_axes(tr):
    x = tr.val(Tensor[M, 8, 64, bf16])
    y = x.permute(1, 0, 2)
    assert y.shape == (8, M, 64) and layout(y) == (["64", "512", "1"], "0", False)
    assert repr(y.type) == "View[8, M, 64, bf16]"
    assert x.permute(0, 1, 2) .type is x.type  # the identity permutation is the tensor itself
    assert x.permute(-3, -1, -2).shape == (M, 64, 8)
    assert layout(y.permute(1, 0, 2)) == (["512", "64", "1"], "0", True)  # back to row-major: contiguous again
    with pytest.raises(TilefishTypeError, match="is not a permutation"):
        x.permute(0, 0, 1)
    with pytest.raises(TilefishTypeError, match="needs 3 integer axes"):
        x.permute(0, 1)
    with pytest.raises(TilefishTypeError, match="out of range"):
        x.permute(0, 1, 3)


def test_select_removes_an_axis_and_moves_the_offset(tr):
    x = tr.val(Tensor[M, 3, 8, 64, bf16])
    k = x[:, 1]
    assert k.shape == (M, 8, 64) and layout(k) == (["1536", "64", "1"], "512", False)  # 512 elements = 1024 bytes in, but the rows are not dense
    assert layout(x[0, 2]) == (["64", "1"], "1024", True)  # [8, 64] at 2048 bytes: dense, aligned
    assert x[..., 3].shape == (M, 3, 8) and x[0, ..., 5].shape == (3, 8)
    assert layout(x.select(1, -1)) == (["1536", "64", "1"], "1024", False)
    first = tr.val(Tensor[4, 8, bf16])[2]  # a row of 8 bf16 = 16 bytes
    assert first.type.contiguous and str(first.type.offset) == "16"
    odd = tr.val(Tensor[4, 6, bf16])[1]  # 6 elements = 12 bytes in
    assert not odd.type.contiguous and str(odd.type.offset) == "6"
    assert [n.op for n in tr.nodes][-1] == "select" and tr.nodes[-1].attrs == {"axis": 0, "index": 1}


def test_select_checks_the_index(tr):
    x = tr.val(Tensor[M, 4, bf16])
    assert x[0].shape == (4,)  # M is at least 1
    with pytest.raises(TilefishTypeError, match=r"index 1 is not provably in range for axis 0 of Tensor\[M, 4, bf16\]: its size M can be as small as 1"):
        x[1]
    with pytest.raises(TilefishTypeError, match="a negative index needs a static size"):
        x[-1]
    with pytest.raises(TilefishTypeError, match=r"index 4 is out of range for axis 1 .* \(size 4\)"):
        x.select(1, 4)
    assert x.select(1, -4).type.offset == 0
    bucketed = tr.val(Tensor[Dim("Bk", buckets=[8, 16]), 4, bf16])
    assert bucketed[7].shape == (4,)  # every bucket has at least 8 rows
    with pytest.raises(TilefishTypeError, match="not provably in range"):
        bucketed[8]
    with pytest.raises(TilefishTypeError, match="must be a python int"):
        x.select(0, 1.5)
    with pytest.raises(TilefishTypeError, match="an index must be a python int, ':' or '...'"):
        x["a"]
    with pytest.raises(TilefishTypeError, match="only integers, ':' and '...' are indices"):
        x[0:2]
    with pytest.raises(TilefishTypeError, match="3 indices for Tensor"):
        x[0, 0, 0]
    assert x[:].type == x.type and x[...] is not None


def test_indexing_a_value_with_no_axis_left_is_a_rank_0_view(tr):
    x = tr.val(Tensor[4, bf16])
    s = x[3]
    assert s.shape == () and str(s.type.offset) == "3"


def test_split_chunks_are_views_with_the_strides_of_the_parent_and_their_own_offset(tr):
    x = tr.val(Tensor[M, 3 * 512, bf16])
    q, k, v = x.split(512, axis=1)
    assert [layout(c) for c in (q, k, v)] == [(["1536", "1"], "0", False), (["1536", "1"], "512", False), (["1536", "1"], "1024", False)]
    assert q.type == Tensor[M, 512, bf16].as_view() and tr.nodes[-1].attrs["sizes"] == (512, 512, 512)
    # along the leading axis the chunks are dense slabs, contiguous where they start on 16 bytes
    top, bottom = tr.val(Tensor[8, 64, bf16]).split(4, axis=0)
    assert top.type.contiguous and bottom.type.contiguous and str(bottom.type.offset) == "256"
    # and a view of a view composes its offsets
    k_rows = k.split([256, 256], axis=1)[1]
    assert layout(k_rows) == (["1536", "1"], "768", False)


def test_a_view_of_a_view_composes_strides_and_offsets(tr):
    x = tr.val(Tensor[16, 3, 8, 64, bf16])
    v = x[:, 1].permute(1, 0, 2)  # [8, 16, 64]
    assert layout(v) == (["64", "1536", "1"], "512", False)
    w = v[2]  # [16, 64]: row r at 512 + 2 * 64 + r * 1536
    assert layout(w) == (["1536", "1"], "640", False)
    assert layout(w[3]) == (["1"], str(640 + 3 * 1536), True)  # a row: dense, and 10496 bytes in


def test_a_view_of_a_block_scaled_tensor_is_refused_except_split(tr):
    mx = BlockScaled(e4m3, ue8m0, 32)
    x = tr.val(Tensor[8, 64, mx])
    for use in (lambda: x.reshape(4, 128), lambda: x.permute(1, 0), lambda: x[0]):
        with pytest.raises(TilefishTypeError, match="block-scaled dtype"):
            use()
    assert x.split(4, axis=0)[0].type.contiguous  # split still takes both buffers along


def test_a_devscalar_is_read_with_load_and_nothing_else(tr):
    s = tr.val(tilefish.DevScalar[f32], name="s")
    v = tr.val(Tensor[M, 8, f32])
    with pytest.raises(TilefishTypeError, match=r"is the DevScalar %s \(DevScalar\[f32\]\).*read it first with tilefish.load\(s\)"):
        v * s
    with pytest.raises(TilefishTypeError, match="read it first with tilefish.load"):
        s.to(bf16)
    with pytest.raises(TilefishTypeError, match="read it first with tilefish.load"):
        tilefish.sum(s)
    with pytest.raises(TilefishTypeError, match="takes a DevScalar parameter"):
        tilefish.load(v)
    r = tilefish.load(s)
    assert r.type == Tensor[f32] and r.shape == () and r.type.contiguous and not r.type.devscalar
    assert (v * r).type == Tensor[M, 8, f32] and (r + 1.0).shape == ()
    assert [n.op for n in tr.nodes] == ["load", "mul", "add"]


def test_views_flow_into_elementwise_ops_and_a_copy_is_contiguous_again(tr):
    x = tr.val(Tensor[M, 3, 8, 64, bf16])
    k = x[:, 1]
    assert (k * 2).type.contiguous and (k + k).type == Tensor[M, 8, 64, bf16] and k.to(f32).type == Tensor[M, 8, 64, f32]
    assert tilefish.sum(k, axis=-1).type == Tensor[M, 8, bf16]
    assert (k + 0).reshape(M, 512).type.contiguous
