"""Hand-written Triton kernels and graphs for the gpu tests of the plan features of M3.3 (not collected: the name has no ``test_`` prefix).

Written from scratch for these tests, and public; declared with ``tilefish.extern``. Importing this module imports triton, so only the gpu tests
do (``host_ops`` imports it with a stand-in for Triton where there is none). No ``from __future__ import annotations``: tilefish evaluates annotations.

* ``add_into``: ``r_out = r + x`` and ``y = 0.5 * r_out`` per row of 64 f32, where ``r_out`` may be written over ``r``
  (``Out[..., MayAlias("r")]``: every program loads its row of ``r`` before it stores the same row of ``r_out``). ``residual_chain`` is a graph
  that makes the residual in the arena and updates it with two calls, which a plan makes in place (one buffer for the three versions).
"""

import triton
import triton.language as tl

import tilefish
from tilefish import Dim, MayAlias, Out, Tensor, const, f32

Rb = Dim("Rb", buckets=[8, 24])  # the rows of the plans


# -- an Out that may be written over an input -----------------------------------------------------------------------------------------------


@triton.jit
def add_into_kernel(x, r, y, r_out, D: tl.constexpr):
    row = tl.program_id(0)
    cols = row * D + tl.arange(0, D)
    s = tl.load(r + cols) + tl.load(x + cols)  # the row of r is read before the same row of r_out is written: one tensor may be both
    tl.store(r_out + cols, s)
    tl.store(y + cols, s * 0.5)


def add_into_reference(x, r):
    s = r + x
    return s * 0.5, s


@tilefish.extern(triton=add_into_kernel, num_warps=1, grid=lambda Rb: (Rb,), reference=add_into_reference)
def add_into(
    x: Tensor[Rb, 64, f32], r: Tensor[Rb, 64, f32], y: Out[Tensor[Rb, 64, f32]], r_out: Out[Tensor[Rb, 64, f32], MayAlias("r")], D: const = 64
): ...


@tilefish.kernel
def scaled(x: Tensor[Rb, 64, f32]) -> Tensor[Rb, 64, f32]:
    return x * 2.0


@tilefish.graph
def residual_chain(x: Tensor[Rb, 64, f32]):
    """``r = 2x`` in the arena, then two in-place updates of it: a plan writes both ``r_out`` over the ``r`` before them, so the three versions of
    the residual are one buffer (the one the graph returns)."""
    r = scaled(x)
    y1, r = add_into(x, r)
    y2, r = add_into(y1, r)
    return y2, r
