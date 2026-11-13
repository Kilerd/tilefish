"""Hand-written Gluon kernels for the gpu tests of Gluon externs (not collected: the name has no ``test_`` prefix).

Written from scratch for these tests, and public; declared with ``tilefish.extern(gluon=...)``. Importing this module imports Triton's Gluon
(``triton.experimental.gluon``), so only the gpu tests and the gate do. No ``from __future__ import annotations``: tilefish evaluates annotations.

* ``row_dot_kernel``: ``y[r] = SCALE * sum(a[r, :C] * b[r, :C])`` in f32, over two views that the kernel reads with **one** row stride, an ``i32``
  (``StrideOf(("a", "b"), 0, dtype=i32)``), and offsets it computes in ``i64`` (``index_bits=64``). The layout is pinned: four warps, one row per
  warp, and each thread holds four contiguous columns of every 128 (``BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])``), so the sum is each
  thread's own elements in order and then the warp's butterfly, the order the layout fixes and no compiler choice changes. ``C`` is a ``const``
  from the bucketed ``Cb`` (128 or 256: one binary per bucket) and ``SCALE`` a typed ``const[f32]``. ``row_dot`` is declared with
  ``enable_fp_fusion=False`` (each product rounded, then added) and ``row_dot_fma`` with ``True`` (one FMA per element).
* ``halves_dot`` is a graph: a lowered kernel doubles ``x`` into the arena and ``row_dot`` reads its two halves, two views of one buffer, in one
  step of the plan.
* ``row_sum_kernel`` is the same reduction of one view with another layout (eight columns per thread: ``[1, 8]``); ``row_sum4`` and ``row_sum8``
  declare it for four and eight warps, ``row_sum_wrong_warps`` for a number of warps its layout is not written for (Gluon's frontend refuses it).
"""

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import tilefish
from tilefish import Dim, Out, StrideOf, Strided, Tensor, const, f32, i32

R = Dim("R")
Cb = Dim("Cb", buckets=[128, 256])
Rb = Dim("Rb", buckets=[8, 24])  # the rows of the plan of halves_dot


@gluon.jit
def row_dot_kernel(a, b, y, s_row, n_rows, C: gl.constexpr, SCALE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0])
    rows = gl.program_id(0) * 4 + gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, C, layout=gl.SliceLayout(0, layout))
    at = rows.to(gl.int64)[:, None] * s_row + cols[None, :]  # an i32 stride, 64-bit offsets
    keep = rows[:, None] < n_rows
    total = gl.sum(gl.load(a + at, mask=keep, other=0.0) * gl.load(b + at, mask=keep, other=0.0), axis=1)
    gl.store(y + rows, total * SCALE, mask=rows < n_rows)


def row_dot_reference(a, b):
    """``0.5 * sum(a * b)`` per row, in float64: what the kernel computes, up to the rounding of its order."""
    return ((a.double() * b.double()).sum(-1) * 0.5).float()


def _declare(name, fusion):
    @tilefish.extern(
        gluon=row_dot_kernel,
        num_warps=4,
        grid=lambda R: (tilefish.cdiv(R, 4),),
        index_bits=64,
        enable_fp_fusion=fusion,
        name=name,
        reference=row_dot_reference,
    )
    def row_dot(
        a: Strided[Tensor[R, Cb, f32]],
        b: Strided[Tensor[R, Cb, f32]],
        y: Out[Tensor[R, f32]],
        s_row: StrideOf(("a", "b"), 0, dtype=i32),
        n_rows: R,
        C: const = Cb,
        SCALE: const[f32] = 0.5,
    ): ...

    return row_dot


row_dot = _declare("row_dot", False)
row_dot_fma = _declare("row_dot_fma", True)


@tilefish.kernel
def doubled(x: Tensor[Rb, 256, f32]) -> Tensor[Rb, 256, f32]:
    return x * 2.0


@tilefish.graph
def halves_dot(x: Tensor[Rb, 256, f32]) -> Tensor[Rb, f32]:
    """``2x`` made in the arena, and its two halves (two ``[Rb, 128]`` views with the row stride 256) read by one Gluon step."""
    pair = doubled(x).reshape(Rb, 2, 128)
    return row_dot(pair[:, 0], pair[:, 1])


# -- a layout per number of warps --------------------------------------------------------------------------------------------------------------


@gluon.jit
def row_sum_kernel(x, y, s_row, n_rows, C: gl.constexpr, WARPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[1, 32], warps_per_cta=[WARPS, 1], order=[1, 0])
    rows = gl.program_id(0) * WARPS + gl.arange(0, WARPS, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, C, layout=gl.SliceLayout(0, layout))
    at = rows.to(gl.int64)[:, None] * s_row + cols[None, :]
    total = gl.sum(gl.load(x + at, mask=rows[:, None] < n_rows, other=0.0), axis=1)
    gl.store(y + rows, total, mask=rows < n_rows)


def _row_sum(name, warps, layout_warps):
    @tilefish.extern(gluon=row_sum_kernel, num_warps=warps, grid=lambda R: (tilefish.cdiv(R, layout_warps),), index_bits=64, name=name)
    def row_sum(
        x: Strided[Tensor[R, 256, f32]], y: Out[Tensor[R, f32]], s_row: StrideOf("x", 0, dtype=i32), n_rows: R, C: const = 256, WARPS: const = layout_warps
    ): ...

    return row_sum


row_sum4 = _row_sum("row_sum4", 4, 4)
row_sum8 = _row_sum("row_sum8", 8, 8)
row_sum_wrong_warps = _row_sum("row_sum_wrong_warps", 8, 4)  # a layout of 4 warps compiled for 8: refused by Gluon's frontend


# -- the same reduction as a triton.jit kernel: Triton picks its layout (and is declared with triton=, never gluon=) -------------------------


@triton.jit
def row_sum_triton(x, y, s_row, n_rows, C: tl.constexpr, WARPS: tl.constexpr):
    rows = tl.program_id(0) * WARPS + tl.arange(0, WARPS)
    cols = tl.arange(0, C)
    at = rows.to(tl.int64)[:, None] * s_row + cols[None, :]
    total = tl.sum(tl.load(x + at, mask=rows[:, None] < n_rows, other=0.0), axis=1)
    tl.store(y + rows, total, mask=rows < n_rows)
