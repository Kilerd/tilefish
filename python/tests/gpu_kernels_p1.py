"""Hand-written Triton kernels for the gpu tests of strided views and device scalars (not collected: the name has no ``test_`` prefix).

They are written from scratch for these tests and declared with ``tilefish.extern``. Importing this module imports triton, so only the gpu
tests do. No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import triton
import triton.language as tl

import tilefish
from tilefish import DevScalar, Dim, Out, StrideOf, Strided, Tensor, bf16, const, f32, i32

S = Dim("S")
R = Dim("R")

# -- a head split: y[s, h, :] = x[s, h, :] for a view x with any strides (the q, k or v of a fused qkv, a column block, a transpose) -------


@triton.jit
def head_copy_kernel(x, y, x_s0, x_s1, D: tl.constexpr):
    s = tl.program_id(0)
    h = tl.program_id(1)
    cols = tl.arange(0, D)
    v = tl.load(x + s * x_s0 + h * x_s1 + cols)
    tl.store(y + (s * tl.num_programs(1) + h) * D + cols, v)


@tilefish.extern(triton=head_copy_kernel, num_warps=1, grid=lambda S: (S, 8))
def head_copy(
    x: Strided[Tensor[S, 8, 64, bf16]],
    y: Out[Tensor[S, 8, 64, bf16]],
    x_s0: StrideOf("x", 0),
    x_s1: StrideOf("x", 1),
    D: const = 64,
): ...


# the same kernel compiled for a 16-byte aligned pointer (tt.divisibility = 16): a view that does not start on 16 bytes is refused
@tilefish.extern(triton=head_copy_kernel, num_warps=1, grid=lambda S: (S, 8), name="head_copy16")
def head_copy16(
    x: Strided[Tensor[S, 8, 64, bf16], 16],
    y: Out[Tensor[S, 8, 64, bf16]],
    x_s0: StrideOf("x", 0, divisible_by=8),
    x_s1: StrideOf("x", 1, divisible_by=8),
    D: const = 64,
): ...


# -- a row softmax over the last axis of a view whose first two axes are permuted ------------------------------------------------------


@triton.jit
def softmax3d_kernel(x, y, x_s0, x_s1, N: tl.constexpr):
    a = tl.program_id(0)
    b = tl.program_id(1)
    cols = tl.arange(0, N)
    xv = tl.load(x + a * x_s0 + b * x_s1 + cols).to(tl.float32)
    m = tl.max(xv, 0)
    e = tl.exp(xv - m)
    tl.store(y + (a * tl.num_programs(1) + b) * N + cols, e / tl.sum(e, 0))


A = Dim("A")
B = Dim("B")


@tilefish.extern(triton=softmax3d_kernel, num_warps=1, grid=lambda A, B: (A, B))
def softmax3d(
    x: Strided[Tensor[A, B, 128, f32]],
    y: Out[Tensor[A, B, 128, f32]],
    x_s0: StrideOf("x", 0),
    x_s1: StrideOf("x", 1),
    N: const = 128,
): ...


# the same with 64-bit offsets: its strides are i64
@triton.jit
def head_copy64_kernel(x, y, x_s0, x_s1, D: tl.constexpr):
    s = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, D)
    v = tl.load(x + s * x_s0 + h * x_s1 + cols)
    tl.store(y + (s * tl.num_programs(1) + h) * D + cols, v)


@tilefish.extern(triton=head_copy64_kernel, num_warps=1, grid=lambda S: (S, 8), index_bits=64)
def head_copy64(
    x: Strided[Tensor[S, 8, 64, bf16]],
    y: Out[Tensor[S, 8, 64, bf16]],
    x_s0: StrideOf("x", 0),
    x_s1: StrideOf("x", 1),
    D: const = 64,
): ...


# -- a device scalar: the rows from n_tok on are zero, whatever n_tok is; the binary does not know it ---------------------------------


@triton.jit
def masked_rows_kernel(x, y, n_tok, D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    n = tl.load(n_tok)
    xv = tl.load(x + row * D + cols)
    tl.store(y + row * D + cols, tl.where(row < n, xv * 2.0, 0.0))


@tilefish.extern(triton=masked_rows_kernel, num_warps=1, grid=lambda R: (R,))
def masked_rows(x: Tensor[R, 128, f32], y: Out[Tensor[R, 128, f32]], n_tok: DevScalar[i32], D: const = 128): ...


# the same read by a lowered kernel: tilefish.load(s) is one tl.load(ptr), broadcast
M = Dim("M")


@tilefish.kernel
def scale_by(x: Tensor[M, 256, f32], s: DevScalar[f32]) -> Tensor[M, 256, f32]:
    return x * tilefish.load(s)


T = Dim("T", divisible_by=16)


@tilefish.kernel
def scale_flat(x: Tensor[T, f32], s: DevScalar[f32], b: DevScalar[f32]) -> Tensor[T, f32]:
    return x * tilefish.load(s) + tilefish.load(b)
