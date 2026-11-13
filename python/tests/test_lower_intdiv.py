"""``//`` and ``%`` of integers by a positive constant: Python's semantics on top of Triton's C ones (bytes: ``test_gpu_masks.py``)."""

import mask_cases as mc
import pytest

import tilefish
from tilefish import Dim, Tensor, TilefishTypeError, const, f32, i32, i64, u8

N = Dim("N")
S = Dim("S", 12)


def test_floor_division_and_modulo_trace_by_a_constant(tr):
    x = tr.val(Tensor[N, i32])
    assert (x // 3).type == Tensor[N, i32] and (x % S).type == Tensor[N, i32]  # a static dimension is a constant
    assert [n.op for n in tr.ctx.nodes] == ["floordiv", "mod"]
    assert tr.ctx.nodes[1].inputs[1].value == 12


def test_a_divisor_is_a_positive_constant_and_a_dividend_an_integer(tr):
    x = tr.val(Tensor[N, i32])
    for bad, match in (
        (lambda: x // 0, "must be positive"),
        (lambda: x % -3, "must be positive"),
        (lambda: x // x, "is a traced tensor"),
        (lambda: 7 // x, "is a traced tensor"),
        (lambda: 7 % x, "is a traced tensor"),
        (lambda: tr.val(Tensor[N, f32]) // 2, "use / on floating-point tensors"),
        (lambda: x // 2.5, "float scalar"),
        (lambda: tr.val(Tensor[N, u8]) % 256, "does not fit in u8"),
    ):
        with pytest.raises(TilefishTypeError, match=match):
            bad()


def test_signed_division_is_moved_to_the_floor_and_unsigned_is_not():
    src = mc.int_divmod.lower().source
    assert "v3_r = v0 % c" in src and "v3 = tl.where(v3_r < 0, v0 // c - 1, v0 // c)" in src
    assert "v4 = tl.where(v4_r < 0, v4_r + c, v4_r)" in src
    assert "v6_r = v1 % tl.cast(1000000007, tl.int64)" in src
    assert "v7 = v2 // tl.cast(5, tl.uint8)" in src and "v8 = v2 % tl.cast(5, tl.uint8)" in src


def test_index_arithmetic_in_the_row_template():
    low = mc.tile_coords.lower()
    assert low.template == "row" and "v1 = v0 * 96" in low.source and "v5 = tl.where(v5_r < 0, v5_r + 8, v5_r)" in low.source


def test_a_const_divisor_must_be_positive_at_the_declaration():
    with pytest.raises(TilefishTypeError, match="must be positive"):

        @tilefish.kernel
        def k(x: Tensor[N, i64], *, c: const = 0) -> Tensor[N, i64]:
            return x % c
