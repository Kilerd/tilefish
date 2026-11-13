"""Kernels with ``argmax`` and ``argmin``, shared by the lowering tests and the gpu tests (not collected).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import tilefish
from tilefish import Dim, Tensor, bf16, f16, f32, i32, i64, u8

R = Dim("R")
C = Dim("C", 5000)  # not a power of two: masked lanes in every row
P = Dim("P", 256)  # a power of two: none
W = Dim("W", buckets=[7, 96, 1000])


@tilefish.kernel
def argmax_bf16(x: Tensor[R, C, bf16]) -> Tensor[R, i64]:
    return tilefish.argmax(x, axis=-1, dtype=i64)


@tilefish.kernel
def argmin_bf16(x: Tensor[R, C, bf16]) -> Tensor[R, 1, i32]:
    return tilefish.argmin(x, axis=-1, dtype=i32, keepdims=True)


@tilefish.kernel
def arg_f32(x: Tensor[R, P, f32]) -> tuple[Tensor[R, i32], Tensor[R, i32]]:
    return x.argmax(dtype=i32), x.argmin(dtype=i32)


@tilefish.kernel
def arg_f16(x: Tensor[R, W, f16]) -> tuple[Tensor[R, i64], Tensor[R, i64]]:
    return tilefish.argmax(x, dtype=i64), tilefish.argmin(x, dtype=i64)


@tilefish.kernel
def arg_ints(a: Tensor[R, W, i32], b: Tensor[R, W, i64], c: Tensor[R, W, u8]) -> tuple[Tensor[R, i32], Tensor[R, i32], Tensor[R, i32], Tensor[R, i32], Tensor[R, i32], Tensor[R, i32]]:
    return (
        tilefish.argmax(a, dtype=i32),
        tilefish.argmin(a, dtype=i32),
        tilefish.argmax(b, dtype=i32),
        tilefish.argmin(b, dtype=i32),
        tilefish.argmax(c, dtype=i32),
        tilefish.argmin(c, dtype=i32),
    )


@tilefish.kernel
def greedy_token(logits: Tensor[R, C, bf16], bias: Tensor[C, bf16]) -> Tensor[R, i64]:
    """An argmax of a computed row: the logits plus a per-column bias, rounded to bf16 first."""
    return tilefish.argmax((logits.to(f32) + bias.to(f32)).to(bf16), dtype=i64)


@tilefish.kernel
def argmax_and_sum(x: Tensor[R, C, bf16]) -> tuple[Tensor[R, i64], Tensor[R, f32]]:
    """A sum in the kernel keeps the row one block: its argmax must be the chunked one's."""
    return tilefish.argmax(x, dtype=i64), tilefish.sum(x.to(f32), axis=-1)
