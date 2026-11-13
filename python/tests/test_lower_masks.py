"""Comparisons, masks, ``where`` and ``iota``: what they trace to, what they refuse, and the Triton they lower to.

The bytes they compute are checked on a GPU by ``test_gpu_masks.py``. Regenerate the goldens with ``UPDATE_GOLDEN=1`` and review the diff:
a changed snapshot is a change of the numerics contract (``tilefish.lower``).
"""

import math
import struct

import mask_cases as mc
import pytest
from test_lower import check_golden

import tilefish
from tilefish import DevScalar, Dim, InOut, Out, Tensor, TilefishDeclarationError, TilefishLoweringError, TilefishTypeError
from tilefish import bf16, bool_, const, e4m3, f16, f32, i32, i64, u8
from tilefish.aot import compile_specs
from tilefish.lower import _round_float

M = Dim("M")
N = Dim("N")
D = Dim("D", 64)


# -- tracing ------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sym,op", [("==", "eq"), ("!=", "ne"), ("<", "lt"), ("<=", "le"), (">", "gt"), (">=", "ge")])
def test_a_comparison_is_a_mask_of_the_broadcast_shape(tr, sym, op):
    a = tr.val(Tensor[M, D, f32])
    b = tr.val(Tensor[D, f32])
    m = eval(f"a {sym} b")
    assert m.type == Tensor[M, D, bool_] and m.dtype is bool_
    (node,) = tr.ctx.nodes
    assert node.op == op and node.inputs == (a, b)


def test_a_scalar_takes_the_dtype_of_the_tensor_it_is_compared_with(tr):
    x = tr.val(Tensor[M, bf16])
    m = 0.5 < x  # reflected: x > 0.5
    (node,) = tr.ctx.nodes
    assert node.op == "gt" and node.inputs[0] is x
    assert node.inputs[1].dtype == bf16 and node.inputs[1].value == 0.5
    assert m.type == Tensor[M, bool_]


def test_comparisons_do_not_promote(tr):
    with pytest.raises(TilefishTypeError, match="no implicit promotion"):
        tr.val(Tensor[M, bf16]) < tr.val(Tensor[M, f32])
    with pytest.raises(TilefishTypeError, match="float scalar but the other operand has integer dtype"):
        tr.val(Tensor[M, i32]) < 0.5
    with pytest.raises(TilefishTypeError, match="does not fit in u8"):
        tr.val(Tensor[M, u8]) < 300


def test_a_mask_has_no_arithmetic_and_no_order(tr):
    m = tr.val(Tensor[M, f32]) > 0.0
    for use in (lambda: m + m, lambda: m * 2, lambda: m < m, lambda: tilefish.exp(m), lambda: tilefish.sum(m, axis=-1)):
        with pytest.raises(TilefishTypeError, match="is the mask"):
            use()


def test_masks_combine_with_logical_operators(tr):
    x = tr.val(Tensor[M, f32])
    p, q = x > 0.0, x < 1.0
    got = [p & q, p | q, p ^ q, ~p]
    assert all(g.type == Tensor[M, bool_] for g in got)
    assert [n.op for n in tr.ctx.nodes[-4:]] == ["and", "or", "xor", "not"]
    with pytest.raises(TilefishTypeError, match="not a mask"):
        p & x
    with pytest.raises(TilefishTypeError, match="must be a mask"):
        p & True


def test_a_mask_casts_to_a_number_and_nothing_casts_to_a_mask(tr):
    m = tr.val(Tensor[M, f32]) != 0.0
    for dt in (u8, i32, i64, f32, bf16, f16):
        assert m.to(dt).type == Tensor[M, dt]
        assert not tr.ctx.nodes[-1].is_rounding_point  # 0 and 1 are exact in every dtype
    with pytest.raises(TilefishTypeError, match="cannot be cast to a mask"):
        tr.val(Tensor[M, u8]).to(bool_)
    assert m.to(bool_) is m


def test_where_selects_between_tensors_and_scalars(tr):
    x = tr.val(Tensor[M, D, bf16])
    y = tr.val(Tensor[D, bf16])
    m = tr.val(Tensor[M, 1, f32]) > 0.0
    assert tilefish.where(m, x, y).type == Tensor[M, D, bf16]
    w = tilefish.where(m, x, 0.25)
    node = tr.ctx.nodes[-1]
    assert w.type == Tensor[M, D, bf16] and node.op == "where" and node.inputs[2].dtype == bf16
    assert tilefish.where(m, 1.0, 0.0, dtype=f32).type == Tensor[M, 1, f32]
    assert tilefish.where(m, m, ~m).type == Tensor[M, 1, bool_]


def test_where_refuses_what_it_cannot_type(tr):
    x = tr.val(Tensor[M, bf16])
    m = x > 0.0
    with pytest.raises(TilefishTypeError, match="no implicit promotion"):
        tilefish.where(m, x, tr.val(Tensor[M, f32]))
    with pytest.raises(TilefishTypeError, match="pass dtype="):
        tilefish.where(m, 1.0, 0.0)
    with pytest.raises(TilefishTypeError, match="dtype=f32 was asked for"):
        tilefish.where(m, x, 0.0, dtype=f32)
    with pytest.raises(TilefishTypeError, match="not a mask"):
        tilefish.where(x, x, x)
    with pytest.raises(TilefishTypeError, match="must be a mask"):
        tilefish.where(True, x, x)
    with pytest.raises(TilefishTypeError, match="masks combine with masks"):
        tilefish.where(m, m, 1)
    with pytest.raises(TilefishTypeError, match="cannot broadcast"):
        tilefish.where(tr.val(Tensor[N, f32]) > 0.0, x, x)


def test_iota_counts_along_one_axis_with_a_broadcastable_shape(tr):
    x = tr.val(Tensor[M, D, f32])
    assert tilefish.iota(x, axis=0).type == Tensor[M, 1, i32]
    assert tilefish.iota(x, axis=-1).type == Tensor[D, i32]
    assert tilefish.iota(x, 1, dtype=i64).type == Tensor[D, i64]
    node = tr.ctx.nodes[-1]
    assert node.op == "iota" and node.inputs == () and node.attrs["dim"] == D
    o = tr.val(Tensor[M, f32], access="out")
    assert tilefish.iota(o, 0).type == Tensor[M, i32]  # only the shape is used: an Out parameter is fine
    for bad, match in (
        (lambda: tilefish.iota(x, 2), "axis 2 is invalid"),
        (lambda: tilefish.iota(x, 0, dtype=u8), "dtype must be i32 or i64"),
        (lambda: tilefish.iota(3, 0), "takes a traced tensor"),
    ):
        with pytest.raises(TilefishTypeError, match=match):
            bad()


def test_a_mask_is_neither_a_parameter_nor_a_result():
    with pytest.raises(TilefishDeclarationError, match="a mask"):

        @tilefish.kernel
        def k1(m: Tensor[N, bool_]) -> Tensor[N, f32]: ...

    with pytest.raises(TilefishDeclarationError, match="a mask"):

        @tilefish.kernel
        def k2(x: Tensor[N, f32], m: Out[Tensor[N, bool_]]): ...

    with pytest.raises(TilefishDeclarationError, match="a mask"):

        @tilefish.extern(triton="m:k", grid=lambda N: (N,))
        def k3(m: DevScalar[bool_]): ...

    with pytest.raises(TilefishTypeError, match="is a mask, which has no storage type"):

        @tilefish.kernel
        def k4(x: Tensor[N, f32]):
            return x > 0.0


# -- lowering ------------------------------------------------------------------------------------------------------------------


GOLDEN = {"mask_rows": mc.mask_rows, "key_mask": mc.key_mask, "compare_scalar_bf16": mc.compare_scalar_bf16}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_source(name):
    check_golden(name, GOLDEN[name].lower().source)


def test_row_masks_use_the_row_and_the_columns():
    low = mc.positions.lower()
    assert low.template == "row"
    body = low.source.split("):\n", 1)[1]
    assert "v0 = row\n" in body and "v1 = cols\n" in body and "v4 = row.to(tl.int64)" in body
    assert "tl.store(out1 + row, v4)" in body  # [R, 1]: one value per row


def test_the_elementwise_template_counts_the_flat_index():
    low = mc.flat_positions.lower()
    assert low.template == "elementwise" and "v0 = offs.to(tl.int64)" in low.source


def test_a_broadcast_iota_goes_to_the_row_template():
    low = mc.mask_rows.lower()
    assert low.template == "row"
    with pytest.raises(TilefishLoweringError) as e:
        tilefish.lower(mc.mask_rows, template="elementwise")
    assert e.value.rule == "elementwise.same-shape"


def test_scalars_of_comparisons_and_selects_are_typed_constants_rounded_to_their_dtype():
    src = mc.compare_scalar_bf16.lower().source
    assert "v0 > tl.cast(0.10009765625, tl.bfloat16)" in src  # 0.1 rounded to bf16 in the source
    assert "v0 <= tl.cast(c, tl.bfloat16)" in src  # a const is rounded by Triton
    assert "tl.where(v7, v0, tl.cast(0.10009765625, tl.bfloat16))" in src
    f16src = mc.compare_scalar_f16.lower().source
    assert "tl.cast(0.0999755859375, tl.float16)" in f16src and "tl.cast(65504.0, tl.float16)" in f16src
    ints = mc.int_compare.lower().source
    assert "tl.cast(3000000000, tl.int64)" in ints and "tl.cast(200, tl.uint8)" in ints
    flt = mc.logic.lower().source
    assert "v2 = v0 > 0.0" in flt  # f32 and i32 are Triton's own literal types: written as they are


def test_both_arms_of_a_select_can_be_scalars():
    src = mc.key_mask.lower().source
    assert "tl.where(v2, tl.cast(0.0, tl.bfloat16), tl.cast(-3.3895313892515355e+38, tl.bfloat16))" in src


def test_specs_of_mask_kernels_are_ordinary():
    (spec,) = compile_specs(mc.zero_rows.lower().extern)
    assert spec.signature == {"x": "*bf16", "n": "*i32", "out": "*bf16", "D": "constexpr", "BLOCK_D": "constexpr"}
    (spec,) = compile_specs(mc.key_mask.lower().extern)
    assert spec.signature["n_1"] == "i32" and [a.count_of for a in spec.args if a.name == "n_1"] == ["keys"]


@pytest.mark.parametrize(
    "value,dtype,expected",
    [
        (0.1, "bf16", 0x3DCD),
        (0.1, "f16", 0x2E66),
        (65519.0, "f16", 0x7BFF),
        (65520.0, "f16", 0x7C00),
        (-3.3895313892515355e38, "bf16", 0xFF7F),
        (3.4e38, "bf16", 0x7F80),  # rounds to f32 3.4e38, then above the largest bf16: inf
        (1e-40, "bf16", 0x0001),  # a subnormal: 1e-40 in f32 is 0x000116C2, which rounds to the bf16 subnormal 0x0001
        (1e39, "f32", None),
    ],
)
def test_round_float_rounds_through_f32_to_nearest_even(value, dtype, expected):
    got = _round_float(value, dtype)
    if dtype == "f32":
        assert got == math.inf
        return
    if dtype == "bf16":
        (bits,) = struct.unpack("<I", struct.pack("<f", got))
        assert bits & 0xFFFF == 0 and bits >> 16 == expected
    else:
        (bits,) = struct.unpack("<H", struct.pack("<e", got))
        assert bits == expected


def test_round_float_keeps_nan_and_infinities():
    assert math.isnan(_round_float(math.nan, "bf16"))
    assert _round_float(-math.inf, "f16") == -math.inf


def test_masks_in_a_kernel_with_an_inout_parameter():
    @tilefish.kernel
    def clamp_negative(x: InOut[Tensor[M, D, f32]], *, floor: const[f32] = -1.0):
        x[...] = tilefish.where(x < floor, floor, x)

    low = clamp_negative.lower()
    assert "v1 = v0 < floor" in low.source and "tl.where(v1, floor, v0)" in low.source


def test_a_comparison_of_fp8_values_is_refused_at_the_trace():
    with pytest.raises(TilefishTypeError, match="storage dtype e4m3"):

        @tilefish.kernel
        def k(x: Tensor[N, e4m3]) -> Tensor[N, u8]:
            return (x > 0.0).to(u8)
