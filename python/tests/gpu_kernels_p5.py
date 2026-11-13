"""Hand-written Triton kernels and graphs for the gpu tests of the plan features of M3.3 (not collected: the name has no ``test_`` prefix).

Written from scratch for these tests, and public; declared with ``tilefish.extern``. Importing this module imports triton, so only the gpu tests
do (``host_ops`` imports it with a stand-in for Triton where there is none). No ``from __future__ import annotations``: tilefish evaluates annotations.

* ``add_into``: ``r_out = r + x`` and ``y = 0.5 * r_out`` per row of 64 f32, where ``r_out`` may be written over ``r``
  (``Out[..., MayAlias("r")]``: every program loads its row of ``r`` before it stores the same row of ``r_out``). ``residual_chain`` is a graph
  that makes the residual in the arena and updates it with two calls, which a plan makes in place (one buffer for the three versions).
* ``copy_steps``: a graph with two copy steps, ``tilefish.copy`` of a dense view into the arena and a store into an ``InOut`` parameter.
* ``view_returns``: a graph that returns a reshape of a result (no copy), a dense view of a parameter and the result again (copy steps).
"""

import triton
import triton.language as tl

import tilefish
from tilefish import Dim, InOut, MayAlias, Out, Tensor, const, f32

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


@tilefish.graph(outputs=("y", "residual"))
def residual_chain(x: Tensor[Rb, 64, f32]):
    """``r = 2x`` in the arena, then two in-place updates of it: a plan writes both ``r_out`` over the ``r`` before them, so the three versions of
    the residual are one buffer (the one the graph returns)."""
    r = scaled(x)
    y1, r = add_into(x, r)
    y2, r = add_into(y1, r)
    return y2, r


# -- copy steps -------------------------------------------------------------------------------------------------------------------------------


@tilefish.graph
def copy_steps(x: Tensor[2, Rb, 64, f32], state: InOut[Tensor[Rb, 64, f32]]):
    """``x[1]`` (a dense view at an offset) copied into the arena with ``tilefish.copy``, updated in place there (``add_into`` writes over the copy, which
    nothing reads after it), and stored into the caller's ``state`` (a store into an ``InOut`` parameter): two copy steps around one launch."""
    r = tilefish.copy(x[1])
    y, r = add_into(x[0], r)
    state[...] = r
    return y


# -- returned views ---------------------------------------------------------------------------------------------------------------------------


@tilefish.kernel
def flat_scaled(x: Tensor[Rb * 64, f32]) -> Tensor[Rb * 64, f32]:
    return x * 3.0


@tilefish.graph
def view_returns(x: Tensor[2, Rb, 64, f32]):
    """A flat result returned as ``[Rb, 64]`` (all of its buffer in another shape: the caller's output, no copy), a dense view of a parameter (a copy
    step) and the result again (a copy step)."""
    y = flat_scaled(x[1].reshape(Rb * 64))
    return y.reshape(Rb, 64), x[0], y.reshape(Rb, 64)
