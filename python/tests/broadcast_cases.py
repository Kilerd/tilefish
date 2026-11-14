"""Elementwise kernels with an operand broadcast over the leading axes or along any axis, shared by the lowering tests and the gpu tests (not
collected).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import tilefish
from tilefish import Dim, InOut, Tensor, bf16, f32, u8

K = Dim("K")
T = Dim("T")
D = Dim("D")
R = Dim("R")


@tilefish.kernel
def pos_add(y: InOut[Tensor[K, T, D, bf16]], pe: Tensor[T, D, bf16]):
    """A position table added to every chunk of a batch, in place: one f32 add, one rounding."""
    y[...] = (y.to(f32) + pe.to(f32)).to(bf16)


@tilefish.kernel
def scale_cols(x: Tensor[R, D, f32], w: Tensor[D, f32], b: Tensor[D, f32]) -> Tensor[R, D, f32]:
    """A dynamic row length: the row template cannot take it, the elementwise template broadcasts w and b."""
    return x * w + b


# -- broadcasts along any axis (note 12 of tilefish.lower): an operand of size 1 on an axis that is not a leading one --------------------------

H = 4  # heads: static
L = Dim("L")  # a dynamic length: a [H, L, L] tensor has the stride L*L along its first axis, a product of two symbolic sizes


@tilefish.kernel
def key_mask_heads(s: Tensor[H, L, L, f32], keep: Tensor[H, 1, L, u8]) -> Tensor[H, L, L, f32]:
    """A mask of the keys per head, which every query row of the head reads: a key whose ``keep`` is 0 gets ``-inf`` (a select)."""
    return tilefish.where(keep != 0, s, float("-inf"))


@tilefish.kernel
def causal_heads(s: Tensor[H, L, L, f32]) -> Tensor[H, L, L, f32]:
    """A causal mask from the lane's coordinates: query ``i`` keeps the keys ``j <= i``, the others get ``-inf``."""
    return tilefish.where(tilefish.iota(s, axis=2) <= tilefish.iota(s, axis=1), s, float("-inf"))


@tilefish.kernel
def scale_each_row(x: Tensor[K, T, D, f32], r: Tensor[K, T, 1, f32]) -> Tensor[K, T, D, f32]:
    """One scale per row of every chunk: ``r`` repeats along the last axis."""
    return x * r


@tilefish.kernel
def middle_rows(x: Tensor[K, T, D, bf16], z: Tensor[K, 1, D, bf16]) -> Tensor[K, T, D, bf16]:
    """A row per chunk, repeated along the middle axis: one f32 multiply, one rounding."""
    return (x.to(f32) * z.to(f32)).to(bf16)


@tilefish.kernel
def column_add(x: Tensor[K, T, D, f32], c: Tensor[T, 1, f32]) -> Tensor[K, T, D, f32]:
    """One value per position ``t``, repeated along the chunks and along the last axis."""
    return x + c


@tilefish.kernel
def outer_heads(q: Tensor[H, L, 1, f32], k: Tensor[H, 1, L, f32]) -> Tensor[H, L, L, f32]:
    """The outer product of two vectors per head: both operands are broadcasts, and the result is ``[H, L, L]``."""
    return q * k
