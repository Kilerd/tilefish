import pytest

import tilefish
from tilefish import (
    BlockScaled,
    Dim,
    Tensor,
    TilefishTypeError,
    bf16,
    e4m3,
    f16,
    f32,
    i32,
    ue8m0,
)
from tilefish.ir import Scalar
from tilefish.trace import Const

M = Dim("M")
N = Dim("N")
D = Dim("D", 2560)
K = Dim("K", divisible_by=32)


# -- elementwise arithmetic -----------------------------------------------------------------


@pytest.mark.parametrize("op,name", [("__add__", "add"), ("__sub__", "sub"), ("__mul__", "mul"), ("__truediv__", "div")])
def test_binary_ops_same_type(tr, op, name):
    a = tr.val(Tensor[M, D, f32])
    b = tr.val(Tensor[M, D, f32])
    r = getattr(a, op)(b)
    assert r.type == Tensor[M, D, f32]
    assert [n.op for n in tr.nodes] == [name]
    assert tr.nodes[0].inputs == (a, b)


def test_scalars_adopt_the_tensor_dtype(tr):
    x = tr.val(Tensor[M, D, bf16])
    r = 2 * x + 0.5
    assert r.type == Tensor[M, D, bf16]
    first, second = tr.nodes
    assert first.inputs == (Scalar(2.0, bf16), x)  # operand order is kept for reflected ops
    assert second.inputs[1] == Scalar(0.5, bf16)
    assert (1 - x).type == x.type


def test_int_scalar_with_int_tensor_and_float_scalar_rejected(tr):
    n = tr.val(Tensor[M, i32])
    assert (n + 3).type == Tensor[M, i32]
    with pytest.raises(TilefishTypeError, match=r"add \(\+\): right operand 0.5 is a float scalar.*integer dtype i32"):
        n + 0.5
    with pytest.raises(TilefishTypeError, match="does not fit"):
        tr.val(Tensor[M, tilefish.u8]) + 300


def test_untyped_consts_adopt_the_tensor_dtype(tr):
    x = tr.val(Tensor[M, bf16])
    scale = Const("scale", 0.5, None)
    r = x + scale
    assert r.type == Tensor[M, bf16]
    assert tr.nodes[0].inputs[1] == Scalar(0.5, bf16, "scale")
    assert (scale * x).type == x.type
    with pytest.raises(TilefishTypeError, match="no value at trace time"):
        x + Const("none", None, None)


def test_typed_consts_must_match_the_tensor_dtype(tr):
    eps = Const("eps", 1e-6, f32)
    assert (tr.val(Tensor[M, f32]) + eps).type == Tensor[M, f32]
    assert tr.nodes[0].inputs[1] == Scalar(1e-6, f32, "eps")
    x = tr.val(Tensor[M, bf16])
    with pytest.raises(
        TilefishTypeError,
        match=r"add \(\+\): right operand const 'eps' is const\[f32\] but the other operand has dtype bf16.*\.to\(f32\)",
    ):
        x + eps
    with pytest.raises(TilefishTypeError, match=r"mul \(\*\): left operand const 'eps' is const\[f32\]"):
        eps * x
    # the user writes the cast, like for tensors
    assert (x.to(f32) + eps).type == Tensor[M, f32]
    # untyped python literals still adopt the tensor dtype
    assert (x + 1e-6).type == x.type


def test_const_with_const_is_rejected():
    with pytest.raises(TilefishTypeError, match="only be combined with a tensor"):
        Const("a", 1.0, f32) + Const("b", 2.0, f32)
    with pytest.raises(TilefishTypeError, match="only be combined with a tensor"):
        Const("a", 1.0, f32) * 2


def test_no_implicit_promotion_between_tensor_dtypes(tr):
    a = tr.val(Tensor[M, D, bf16])
    b = tr.val(Tensor[M, D, f32])
    with pytest.raises(TilefishTypeError, match=r"add \(\+\): no implicit promotion.*left operand is Tensor\[M, D, bf16\].*right operand is Tensor\[M, D, f32\]"):
        a + b
    with pytest.raises(TilefishTypeError, match="no implicit promotion"):
        b * a
    with pytest.raises(TilefishTypeError, match="no implicit promotion"):
        tr.val(Tensor[M, f16]) - tr.val(Tensor[M, bf16])
    assert tr.nodes == []  # a rejected op leaves nothing behind


def test_explicit_cast_makes_it_legal(tr):
    a = tr.val(Tensor[M, D, bf16])
    b = tr.val(Tensor[M, D, f32])
    assert (a.to(f32) + b).type == Tensor[M, D, f32]


def test_broadcast_trailing_dims(tr):
    x = tr.val(Tensor[M, D, f32])
    w = tr.val(Tensor[D, f32])
    col = tr.val(Tensor[M, 1, f32])
    assert (x * w).type == Tensor[M, D, f32]
    assert (x * col).type == Tensor[M, D, f32]
    assert (col + tr.val(Tensor[1, D, f32])).type == Tensor[M, D, f32]
    assert (w + x).type == Tensor[M, D, f32]
    assert (tr.val(Tensor[f32]) + x).type == x.type


def test_broadcast_mismatch_names_the_op_and_the_dimensions(tr):
    x = tr.val(Tensor[M, D, f32])
    row = tr.val(Tensor[M, f32])
    with pytest.raises(TilefishTypeError, match=r"mul \(\*\): cannot broadcast Tensor\[M, D, f32\] with Tensor\[M, f32\].*D vs M at axis -1"):
        x * row
    with pytest.raises(TilefishTypeError, match="cannot broadcast"):
        tr.val(Tensor[M, f32]) + tr.val(Tensor[N, f32])
    with pytest.raises(TilefishTypeError, match="cannot broadcast"):
        tr.val(Tensor[D, f32]) + tr.val(Tensor[2 * D, f32])


def test_storage_and_blockscaled_dtypes_have_no_arithmetic(tr):
    q = tr.val(Tensor[M, D, e4m3])
    with pytest.raises(TilefishTypeError, match="storage dtype e4m3.*cast it first"):
        q * 2
    w = tr.val(Tensor[N, K, BlockScaled(e4m3, ue8m0, 32)])
    with pytest.raises(TilefishTypeError, match="block-scaled dtype"):
        w + w


def test_integer_division_needs_a_float(tr):
    with pytest.raises(TilefishTypeError, match=r"div \(/\): left operand must be a floating-point"):
        tr.val(Tensor[M, i32]) / 2


def test_binary_operand_checks(tr):
    x = tr.val(Tensor[M, f32])
    with pytest.raises(TilefishTypeError, match="python number or a const parameter"):
        x + "a"
    with pytest.raises(TilefishTypeError, match="python number or a const parameter"):
        x + True
    out = tr.val(Tensor[M, f32], access="out")
    with pytest.raises(TilefishTypeError, match="write-only"):
        x + out
    other = type(tr)()
    with pytest.raises(TilefishTypeError, match="different trace"):
        x + other.val(Tensor[M, f32])


# -- unary ----------------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [tilefish.silu, tilefish.gelu, tilefish.rsqrt, tilefish.exp, tilefish.sqrt])
def test_unary_ops(tr, fn):
    x = tr.val(Tensor[M, D, bf16])
    assert fn(x).type == x.type
    assert tr.nodes[-1].op == fn.__name__
    with pytest.raises(TilefishTypeError, match=fn.__name__ + r": argument 'x' must be a floating-point"):
        fn(tr.val(Tensor[M, i32]))
    with pytest.raises(TilefishTypeError, match="traced tensor value"):
        fn(1.0)


# -- casts and rounding points --------------------------------------------------------------


def test_widening_cast_is_not_a_rounding_point(tr):
    r = tr.val(Tensor[M, D, bf16]).to(f32)
    assert r.type == Tensor[M, D, f32]
    (node,) = tr.nodes
    assert node.op == "cast" and node.attrs == {"to": f32, "rounding": False}
    assert not node.is_rounding_point


def test_narrowing_cast_is_a_rounding_point(tr):
    r = tr.val(Tensor[M, D, f32]).to(bf16)
    assert r.type == Tensor[M, D, bf16]
    (node,) = tr.nodes
    assert node.attrs == {"to": bf16, "rounding": True}
    assert node.is_rounding_point
    assert "# rounding point" in str(node)


def test_cast_to_same_dtype_is_a_no_op(tr):
    x = tr.val(Tensor[M, f32])
    assert x.to(f32) is x
    assert tr.nodes == []


def test_cast_to_storage_format_rounds(tr):
    assert tr.val(Tensor[M, f32]).to(e4m3).type == Tensor[M, e4m3]
    assert tr.nodes[0].is_rounding_point


def test_cast_errors(tr):
    x = tr.val(Tensor[M, f32])
    with pytest.raises(TilefishTypeError, match="block-scaled"):
        x.to(BlockScaled(e4m3, ue8m0, 32))
    with pytest.raises(TilefishTypeError, match="expected a dtype"):
        x.to("bf16")


# -- reductions -----------------------------------------------------------------------------


def test_reductions_follow_numpy(tr):
    x = tr.val(Tensor[M, D, f32])
    assert tilefish.sum(x, axis=-1).type == Tensor[M, f32]
    assert tilefish.sum(x, axis=0).type == Tensor[D, f32]
    assert tilefish.mean(x, axis=-1, keepdims=True).type == Tensor[M, 1, f32]
    assert tilefish.max(x, axis=(0, 1)).type == Tensor[f32]
    assert tilefish.max(x, axis=(0, 1), keepdims=True).type == Tensor[1, 1, f32]
    assert tilefish.sum(x).type == Tensor[f32]
    assert x.mean(axis=1).type == Tensor[M, f32]
    assert [n.op for n in tr.nodes] == ["sum", "sum", "mean", "max", "max", "sum", "mean"]
    assert tr.nodes[2].attrs == {"axes": (1,), "keepdims": True}


def test_reduction_result_broadcasts_back_only_with_keepdims(tr):
    x = tr.val(Tensor[M, D, f32])
    assert (x * tilefish.mean(x, axis=-1, keepdims=True)).type == x.type
    with pytest.raises(TilefishTypeError, match="cannot broadcast"):
        x * tilefish.mean(x, axis=-1)


def test_reduction_errors(tr):
    x = tr.val(Tensor[M, D, f32])
    with pytest.raises(TilefishTypeError, match="axis 2 is out of range"):
        tilefish.sum(x, axis=2)
    with pytest.raises(TilefishTypeError, match="duplicate axes"):
        tilefish.sum(x, axis=(0, -2))
    with pytest.raises(TilefishTypeError, match="axis must be an int"):
        tilefish.sum(x, axis="a")
    with pytest.raises(TilefishTypeError, match="mean: argument 'x' must be a floating-point"):
        tilefish.mean(tr.val(Tensor[M, i32]), axis=0)
    assert tilefish.sum(tr.val(Tensor[M, i32]), axis=0).type == Tensor[i32]


# -- split ----------------------------------------------------------------------------------


def test_split_equal_chunks(tr):
    F = Dim("F", 3072)
    x = tr.val(Tensor[M, 2 * F, bf16])
    g, u = x.split(F, axis=-1)
    assert g.type == u.type == Tensor[M, F, bf16].as_view()  # a slab of the last axis is a view
    (node,) = tr.nodes
    assert node.op == "split" and node.attrs == {"axis": 1, "sizes": (F, F)}
    assert node.outputs == (g, u)
    assert len(x.split(1024, axis=1)) == 6


def test_split_symbolic_dimension(tr):
    n = Dim("n", divisible_by=8)  # 8 bf16 elements are 16 bytes: every chunk starts aligned
    a, b, c = tr.val(Tensor[3 * n, bf16]).split(n, axis=0)
    assert a.type == b.type == c.type == Tensor[n, bf16]


def test_split_explicit_sizes(tr):
    x = tr.val(Tensor[M, 100, f32])
    q, k = x.split([60, 40], axis=1)
    assert q.type == Tensor[M, 60, f32].as_view() and k.type == Tensor[M, 40, f32].as_view()
    with pytest.raises(TilefishTypeError, match="add up to 90 but axis 1"):
        x.split([60, 30], axis=1)


def test_split_errors(tr):
    x = tr.val(Tensor[M, 100, f32])
    with pytest.raises(TilefishTypeError, match="not a whole number of chunks of size 30"):
        x.split(30, axis=1)
    with pytest.raises(TilefishTypeError, match="not a whole number of chunks"):
        x.split(16, axis=0)  # M is not provably a multiple of 16
    with pytest.raises(TilefishTypeError, match="axis 2 is invalid"):
        x.split(10, axis=2)
    with pytest.raises(TilefishTypeError, match="not a whole number of chunks"):
        x.split(200, axis=1)


def test_split_keeps_block_scaled_validity(tr):
    w = tr.val(Tensor[N, 128, BlockScaled(e4m3, ue8m0, 32)])
    assert [p.shape[1] for p in w.split(64, axis=1)] == [64, 64]
    with pytest.raises(TilefishTypeError, match="chunk type is invalid.*not divisible by block=32"):
        w.split(16, axis=1)


# -- stores ---------------------------------------------------------------------------------


def test_store_into_inout(tr):
    res = tr.val(Tensor[M, D, f32], name="residual", access="inout")
    r = tr.val(Tensor[M, D, f32]) + 1
    res[...] = r
    node = tr.nodes[-1]
    assert node.op == "store" and node.inputs == (res, r) and node.attrs == {"target": "residual"}
    # a store copies: its output is a fresh value, the parameter's new version
    (version,) = node.outputs
    assert version is not r and version.type == res.type and res.live is version


def test_store_errors(tr):
    res = tr.val(Tensor[M, D, f32], name="residual", access="inout")
    r = tr.val(Tensor[M, D, bf16])
    with pytest.raises(TilefishTypeError, match=r"store into 'residual': expected Tensor\[M, D, f32\], got Tensor\[M, D, bf16\].*\.to\(dtype\)"):
        res[...] = r
    with pytest.raises(TilefishTypeError, match="expected Tensor"):
        res[...] = tr.val(Tensor[M, f32])
    with pytest.raises(TilefishTypeError, match="read-only"):
        tr.val(Tensor[M, D, f32], name="x")[...] = tr.val(Tensor[M, D, f32])
    with pytest.raises(TilefishTypeError, match="only whole-tensor assignment"):
        res[0] = tr.val(Tensor[M, D, f32])
    with pytest.raises(TilefishTypeError, match="traced tensor value"):
        res[...] = 0.0


def test_out_params_are_write_only(tr):
    y = tr.val(Tensor[M, f32], name="y", access="out")
    y[...] = tr.val(Tensor[M, f32])
    with pytest.raises(TilefishTypeError, match="Out parameter 'y', which is write-only"):
        tilefish.silu(y)


def test_value_is_not_indexable_iterable_or_truthy(tr):
    x = tr.val(Tensor[M, f32])
    with pytest.raises(TilefishTypeError, match="an index must be a python int"):
        x[1.5]  # integers, ':' and '...' index a tensor (it is a view); nothing else does
    with pytest.raises(TilefishTypeError, match="cannot be unpacked"):
        a, b = x
    with pytest.raises(TilefishTypeError, match="truth value"):
        if x:
            pass


# -- operators and attributes outside the M1 language ---------------------------------------------------------


@pytest.mark.parametrize(
    "use",
    [
        lambda x: -x,
        lambda x: +x,
        lambda x: abs(x),
        lambda x: x**2,
        lambda x: 2**x,
        lambda x: x @ x,
        lambda x: len(x),
        lambda x: int(x),
        lambda x: float(x),
        lambda x: range(x),
        lambda x: x << 1,
    ],
)
def test_unsupported_operators_raise_tilefish_errors(tr, use):
    x = tr.val(Tensor[M, f32])
    with pytest.raises(TilefishTypeError, match="not supported in M1"):
        use(x)


@pytest.mark.parametrize("use", [lambda x: x // 2, lambda x: x % 2])
def test_integer_division_of_floats_is_refused(tr, use):
    x = tr.val(Tensor[M, f32])
    with pytest.raises(TilefishTypeError, match="use / on floating-point tensors"):
        use(x)


@pytest.mark.parametrize("use", [lambda x: ~x, lambda x: x & x, lambda x: x | x, lambda x: x ^ x])
def test_bitwise_operators_on_numbers_are_refused(tr, use):
    x = tr.val(Tensor[M, f32])
    with pytest.raises(TilefishTypeError, match="not a mask"):
        use(x)


def test_unknown_attributes_are_tilefish_errors_and_still_attribute_errors(tr):
    x = tr.val(Tensor[M, f32])
    for name in ("T", "view", "transpose", "size", "item"):  # reshape, permute and select are views now (tests/test_views.py)
        with pytest.raises(tilefish.TilefishAttributeError, match=f"no attribute '{name}'"):
            getattr(x, name)
    with pytest.raises(tilefish.TilefishError):
        x.reshape(1)
    assert not hasattr(x, "anything")  # AttributeError subclass: hasattr/getattr defaults keep working
    assert getattr(x, "anything", 7) == 7


def test_method_arity_errors_are_tilefish_errors(tr):
    x = tr.val(Tensor[M, f32])
    with pytest.raises(TilefishTypeError, match=r"to: needs a dtype"):
        x.to()
    with pytest.raises(TilefishTypeError, match=r"split: needs a size and an axis"):
        x.split(2)
    with pytest.raises(TilefishTypeError, match=r"split: needs a size and an axis"):
        x.split()
    with pytest.raises(TilefishTypeError, match=r"split: sizes must be ints or dimensions"):
        x.split("a", axis=0)
    with pytest.raises(TilefishTypeError, match="split: axis 'a' is invalid"):
        x.split(2, axis="a")


def test_split_sizes_must_be_positive(tr):
    x = tr.val(Tensor[M, 10, f32])
    with pytest.raises(TilefishTypeError, match="split: chunk type is invalid"):
        x.split([11, -1], axis=1)
    with pytest.raises(TilefishTypeError, match="split: chunk type is invalid"):
        x.split([10, 0], axis=1)


def test_dimension_mismatch_in_broadcast_shows_constraints_when_names_collide(tr):
    plain, constrained = Dim("K"), Dim("K", divisible_by=32)
    x = tr.val(Tensor[M, plain, f32])
    y = tr.val(Tensor[M, constrained, f32])
    with pytest.raises(TilefishTypeError, match=r"dimension Dim\('K'\) vs Dim\('K', divisible_by=32\) at axis -1"):
        x + y


# -- versions: reading after a store ----------------------------------------------------------------------


def test_reading_an_inout_param_after_a_store_reads_the_stored_copy(tr):
    res = tr.val(Tensor[M, D, f32], name="res", access="inout")
    r = res + 1
    assert tr.nodes[-1].inputs[0] is res  # before the store it reads the parameter
    res[...] = r
    version1 = res.live
    assert version1 is not r and version1 is not res  # a store makes a fresh value: a copy
    z = res * 2
    assert tr.nodes[-1].inputs[0] is version1  # after the store it reads the new version
    res[...] = z
    version2 = res.live
    assert version2 is not z and version2 is not version1
    w = tilefish.silu(res)
    assert tr.nodes[-1].inputs == (version2,)
    assert w.type == res.type
    assert [n.op for n in tr.nodes] == ["add", "store", "mul", "store", "silu"]
    assert tr.nodes[1].outputs == (version1,) and tr.nodes[3].outputs == (version2,)


def test_storing_a_value_into_itself_makes_a_copy_and_does_not_loop(tr):
    res = tr.val(Tensor[M, f32], name="res", access="inout")
    res[...] = res
    assert res.live is not res
    assert tr.nodes[-1].inputs == (res, res)
    assert (res + 1).type == res.type


def test_storing_into_an_out_param_has_no_version(tr):
    y = tr.val(Tensor[M, f32], name="y", access="out")
    y[...] = tr.val(Tensor[M, f32])
    assert tr.nodes[-1].op == "store" and tr.nodes[-1].outputs == ()


def test_out_params_stay_write_only_before_and_after_a_store(tr):
    y = tr.val(Tensor[M, f32], name="y", access="out")
    with pytest.raises(TilefishTypeError, match="Out parameter 'y', which is write-only"):
        y + 1
    y[...] = tr.val(Tensor[M, f32])
    with pytest.raises(TilefishTypeError, match="Out parameter 'y', which is write-only"):
        tilefish.exp(y)
    with pytest.raises(TilefishTypeError, match="Out parameter 'y', which is write-only"):
        tilefish.sum(y)
    with pytest.raises(TilefishTypeError, match="Out parameter 'y', which is write-only"):
        y.to(bf16)
    with pytest.raises(TilefishTypeError, match="Out parameter 'y', which is write-only"):
        y.split(1, axis=0)
    out2 = tr.val(Tensor[M, f32], name="z", access="out")
    with pytest.raises(TilefishTypeError, match="write-only"):
        out2[...] = y


# -- strided views ------------------------------------------------------------------------------------------


def test_split_along_a_leading_axis_is_contiguous_and_along_another_axis_is_a_view(tr):
    x = tr.val(Tensor[4, 8, f32])
    a, b = x.split(2, axis=0)
    assert a.type.contiguous and a.type == Tensor[2, 8, f32]
    c, d = x.split(4, axis=1)
    assert not c.type.contiguous
    assert c.type == Tensor[4, 4, f32].as_view()
    assert c.type != Tensor[4, 4, f32]
    assert repr(c.type) == "View[4, 4, f32]" and repr(a.type) == "Tensor[2, 8, f32]"
    # size-1 leading dimensions keep a slab contiguous
    assert tr.val(Tensor[1, 8, f32]).split(4, axis=1)[0].type.contiguous
    assert not tr.val(Tensor[M, 8, f32]).split(4, axis=1)[0].type.contiguous
    # a slab of a view, and a slab of a contiguous chunk along another axis
    assert not c.split(2, axis=0)[0].type.contiguous
    assert not a.split(4, axis=1)[0].type.contiguous


def test_views_flow_into_elementwise_ops_reductions_and_casts(tr):
    x = tr.val(Tensor[4, 8, f32])
    c, _ = x.split(4, axis=1)
    contiguous = Tensor[4, 4, f32]
    assert (c * 2).type == contiguous
    assert (c + c).type == contiguous
    assert tilefish.silu(c).type == contiguous
    assert tilefish.sum(c, axis=0).type == Tensor[4, f32]
    assert c.to(bf16).type == Tensor[4, 4, bf16]
    assert c.to(f32) is c  # a no-op cast stays the view itself
    assert c.type.materialized() == contiguous and contiguous.materialized() is contiguous


def test_a_view_cannot_be_stored(tr):
    res = tr.val(Tensor[4, 4, f32], name="res", access="inout")
    c, _ = tr.val(Tensor[4, 8, f32]).split(4, axis=1)
    with pytest.raises(TilefishTypeError, match=r"expected Tensor\[4, 4, f32\], got View\[4, 4, f32\].*strided view cannot be stored"):
        res[...] = c
    res[...] = c * 1  # computing from it is fine


def test_split_needs_at_least_one_whole_chunk(tr):
    x = tr.val(Tensor[M, 10, f32])
    for bad in (-5, -10, 0):
        with pytest.raises(TilefishTypeError, match="split: axis 1 of .* is not a whole number of chunks"):
            x.split(bad, axis=1)
    n = Dim("n")
    y = tr.val(Tensor[3 * n, bf16])
    with pytest.raises(TilefishTypeError, match="not a whole number of chunks"):
        y.split(-n, axis=0)  # the ratio is -3, not a chunk count
    assert len(y.split(n, axis=0)) == 3
