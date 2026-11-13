"""Kernels with comparisons, masks, ``where`` and ``iota``, shared by the lowering tests and the gpu tests (not collected).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import tilefish
from tilefish import DevScalar, Dim, Tensor, bf16, const, f16, f32, i32, i64, u8

R = Dim("R")
D = Dim("D", 96)
N = Dim("N")
B = Dim("B")
BF16_MIN = -3.3895313892515355e38  # the most negative finite bf16 (0xFF7F)


@tilefish.kernel
def mask_rows(x: Tensor[R, D, bf16], n: DevScalar[i32]) -> Tensor[R, D, bf16]:
    """Rows below ``n`` as they are; from ``n`` on, ``x * 0.0`` (a multiply, so -1.5 gives -0.0 and inf or NaN give NaN)."""
    keep = tilefish.iota(x, axis=0) < tilefish.load(n)
    return tilefish.where(keep, x, (x.to(f32) * 0.0).to(bf16))


@tilefish.kernel
def zero_rows(x: Tensor[R, D, bf16], n: DevScalar[i32]) -> Tensor[R, D, bf16]:
    """Rows from ``n`` on are +0.0."""
    return tilefish.where(tilefish.iota(x, axis=0) < tilefish.load(n), x, 0.0)


@tilefish.kernel
def key_mask(keys: Tensor[B, bf16], n: DevScalar[i32]) -> Tensor[B, bf16]:
    """An additive mask: 0 for the keys below ``n``, the most negative finite bf16 for the others (both arms are scalars)."""
    return tilefish.where(tilefish.iota(keys, axis=0) < tilefish.load(n), 0.0, BF16_MIN, dtype=bf16)


@tilefish.kernel
def compare_f32(
    a: Tensor[N, f32], b: Tensor[N, f32]
) -> tuple[Tensor[N, u8], Tensor[N, u8], Tensor[N, u8], Tensor[N, u8], Tensor[N, u8], Tensor[N, u8]]:
    return (
        (a == b).to(u8),
        (a != b).to(u8),
        (a < b).to(u8),
        (a <= b).to(u8),
        (a > b).to(u8),
        (a >= b).to(u8),
    )


@tilefish.kernel
def compare_scalar_bf16(x: Tensor[N, bf16], *, c: const = 0.1) -> tuple[Tensor[N, u8], Tensor[N, u8], Tensor[N, u8], Tensor[N, bf16]]:
    """A scalar is rounded to bf16 before it is compared: bf16(0.1) = 0.10009765625 is not > 0.1."""
    return (x > 0.1).to(u8), (x <= c).to(u8), (0.1 == x).to(u8), tilefish.where(x >= -2.5, x, 0.1)


@tilefish.kernel
def compare_scalar_f16(x: Tensor[N, f16]) -> tuple[Tensor[N, u8], Tensor[N, f16]]:
    return (x < 0.1).to(u8), tilefish.where(x != x, 65519.0, x)  # 65519 rounds to 65504 in f16


@tilefish.kernel
def select(m: Tensor[N, u8], a: Tensor[N, bf16], b: Tensor[N, bf16]) -> Tensor[N, bf16]:
    """A raw select: NaN payloads and signed zeros come through."""
    return tilefish.where(m != 0, a, b)


@tilefish.kernel
def logic(a: Tensor[N, f32], b: Tensor[N, f32]) -> tuple[Tensor[N, u8], Tensor[N, u8], Tensor[N, u8], Tensor[N, f32]]:
    p = a > 0.0
    q = b > 0.0
    return (p & q).to(u8), (p | q).to(u8), (p ^ q).to(u8), (~p).to(f32)


@tilefish.kernel
def int_compare(x: Tensor[N, i64], y: Tensor[N, u8]) -> tuple[Tensor[N, u8], Tensor[N, u8], Tensor[N, i64]]:
    return (x < 3000000000).to(u8), (y >= 200).to(u8), tilefish.where(x < 0, 7, x)


@tilefish.kernel
def positions(x: Tensor[R, D, f32]) -> tuple[Tensor[R, D, i32], Tensor[R, 1, i64]]:
    """The row and column of every element, from the row template's ``row`` and ``cols``."""
    rows = tilefish.iota(x, axis=0)
    cols = tilefish.iota(x, axis=-1)
    return rows * 1000 + cols, tilefish.iota(x, axis=0, dtype=i64)


@tilefish.kernel
def flat_positions(x: Tensor[N, f32]) -> Tensor[N, i64]:
    return tilefish.iota(x, axis=0, dtype=i64)


@tilefish.kernel
def int_divmod(
    x: Tensor[N, i32], y: Tensor[N, i64], z: Tensor[N, u8], *, c: const = 7
) -> tuple[Tensor[N, i32], Tensor[N, i32], Tensor[N, i64], Tensor[N, i64], Tensor[N, u8], Tensor[N, u8]]:
    """Python's // and % (floor; a remainder in [0, c)) of signed and unsigned integers by positive constants."""
    return x // c, x % c, y // 3, y % 1000000007, z // 5, z % 5


@tilefish.kernel
def tile_coords(x: Tensor[R, D, f32]) -> tuple[Tensor[R, D, i32], Tensor[R, D, i32]]:
    """The tile and the position in it of every element, from its flat index: index arithmetic in the row template."""
    flat = tilefish.iota(x, axis=0) * 96 + tilefish.iota(x, axis=-1)
    return flat // 8, flat % 8
