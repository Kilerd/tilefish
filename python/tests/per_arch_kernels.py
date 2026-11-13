"""Hand-written Triton kernels for the gpu tests of an op with an implementation per architecture (not collected: the name has no ``test_`` prefix).

Written from scratch for these tests, and public. No ``from __future__ import annotations``: tilefish evaluates annotations.

``row_sum`` is ``y[r] = sum(x[r, :1024])`` in f32, declared once with ``@tilefish.extern(by_arch=...)`` and two tilings of the reduction:

* on ``sm_103``, ``row_sum_tiled``: ``ROWS = 2`` rows per program, each row summed ``BLOCK = 128`` columns at a time (eight partial sums, each a tree
  over 128 columns, added to the accumulator in order), four warps;
* on ``sm_120`` and as the fallback, ``row_sum_whole``: one row per program, summed as one tree over all 1024 columns, eight warps (four for the
  fallback).

The two are other machine code and, for most inputs, other bits (another association of the same 1024 additions). The grid follows ``ROWS``. The
graph ``scaled_row_sum`` doubles ``x`` in a lowered kernel and sums its rows with ``row_sum``, one step of each.
"""

import torch
import triton
import triton.language as tl

import tilefish
from tilefish import Dim, Out, Tensor, const, f32

R = Dim("R")
Rb = Dim("Rb", buckets=[8, 24])  # the rows of the plan of scaled_row_sum


@triton.jit
def row_sum_tiled(x, y, n_rows, ROWS: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    keep = rows < n_rows
    acc = tl.zeros((ROWS,), dtype=tl.float32)
    for k in range(0, C, BLOCK):
        cols = k + tl.arange(0, BLOCK)
        acc += tl.sum(tl.load(x + rows[:, None] * C + cols[None, :], mask=keep[:, None], other=0.0), axis=1)
    tl.store(y + rows, acc, mask=keep)


@triton.jit
def row_sum_whole(x, y, n_rows, ROWS: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    keep = rows < n_rows
    cols = tl.arange(0, BLOCK)
    total = tl.sum(tl.load(x + rows[:, None] * C + cols[None, :], mask=keep[:, None] & (cols[None, :] < C), other=0.0), axis=1)
    tl.store(y + rows, total, mask=keep)


def row_sum_reference(x):
    return x.double().sum(-1).float()


TILED = tilefish.impl(triton=row_sum_tiled, num_warps=4, consts={"ROWS": 2, "BLOCK": 128})
WHOLE = tilefish.impl(triton=row_sum_whole, num_warps=8)


@tilefish.extern(
    by_arch={"sm_103": TILED, "sm_120": WHOLE},
    fallback=tilefish.impl(triton=row_sum_whole, num_warps=4),
    grid=lambda R, ROWS: (tilefish.cdiv(R, ROWS),),
    reference=row_sum_reference,
)
def row_sum(x: Tensor[R, 1024, f32], y: Out[Tensor[R, f32]], n_rows: R, ROWS: const = 1, C: const = 1024, BLOCK: const = 1024): ...


# the two tilings as ops of one implementation each, on any architecture: what the other architecture's implementation would compute here
@tilefish.extern(triton=row_sum_tiled, num_warps=4, grid=lambda R: (tilefish.cdiv(R, 2),))
def tiled_only(x: Tensor[R, 1024, f32], y: Out[Tensor[R, f32]], n_rows: R, ROWS: const = 2, C: const = 1024, BLOCK: const = 128): ...


@tilefish.extern(triton=row_sum_whole, num_warps=8, grid=lambda R: (R,))
def whole_only(x: Tensor[R, 1024, f32], y: Out[Tensor[R, f32]], n_rows: R, ROWS: const = 1, C: const = 1024, BLOCK: const = 1024): ...


def implementation_here() -> str:
    """The key of the implementation ``row_sum`` runs on this device."""
    major, minor = torch.cuda.get_device_capability()
    return row_sum.for_arch(major * 10 + minor).implementation  # type: ignore[return-value]


@tilefish.kernel
def doubled(x: Tensor[Rb, 1024, f32]) -> Tensor[Rb, 1024, f32]:
    return x * 2.0


@tilefish.graph
def scaled_row_sum(x: Tensor[Rb, 1024, f32]) -> Tensor[Rb, f32]:
    """``sum(2x)`` per row: a lowered step that doubles ``x`` into the arena, and the step of ``row_sum`` this architecture runs."""
    return row_sum(doubled(x))
