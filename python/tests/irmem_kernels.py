"""A Gluon extern with a configuration per bucket over many buckets, for the memory and the time of its export (not collected: the name has
no ``test_`` prefix). Written from scratch, and public. Importing this module imports Triton's Gluon (``triton.experimental.gluon``), so only
the gpu tests and :mod:`irmem_export` do. No ``from __future__ import annotations``: tilefish evaluates annotations.

* ``chain_kernel``: ``y[t, :] = f(x[t, :])``, an unrolled chain of ``STEPS`` multiply-adds of each element (``gl.static_range``), so the IR
  Gluon's frontend builds grows with ``STEPS``: a stand-in for a large kernel, whose frontend IR is large.
* ``chain_op(steps)`` declares it over ``[T, D]`` with ``T`` and ``D`` bucketed (12 x 11 = 132 binaries) and ``by_bucket=`` choosing one of three
  configurations (warps, ``BLOCK``) by ``D``: 132 manifest entries, 3 distinct frontend inputs. The bucket values reach the kernel only as runtime
  scalars, so every bucket of one configuration has the same frontend IR, the same IR digest and the same binary.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import tilefish
from tilefish import Dim, Out, Tensor, const, f32

T = Dim("T", buckets=[8 * k for k in range(1, 13)])  # 12 buckets
D = Dim("D", buckets=[128 * k for k in range(1, 12)])  # 11 buckets


@gluon.jit
def chain_kernel(x, y, n_t, n_d, BLOCK: gl.constexpr, STEPS: gl.constexpr, WARPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[BLOCK // (32 * WARPS)], threads_per_warp=[32], warps_per_cta=[WARPS], order=[0])
    row = gl.program_id(0)
    cols = gl.program_id(1) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    keep = (cols < n_d) & (row < n_t)
    at = row.to(gl.int64) * n_d + cols
    v = gl.load(x + at, mask=keep, other=0.0)
    acc = v
    for i in gl.static_range(STEPS):
        acc = acc * 0.5 + v * (i + 1.0)
    gl.store(y + at, acc, mask=keep)


def chain_reference(x, steps):
    """What ``chain_kernel`` computes, in float64 (up to the rounding of each step)."""
    v = x.double()
    acc = v
    for i in range(steps):
        acc = acc * 0.5 + v * (i + 1.0)
    return acc.float()


def _config(T, D):
    """Three configurations, by the width of the bucket (each binary is one bucket of ``T`` and one of ``D``)."""
    if D <= 384:
        return tilefish.config(num_warps=2, BLOCK=128, WARPS=2)
    if D <= 896:
        return tilefish.config(num_warps=4, BLOCK=256, WARPS=4)
    return tilefish.config(num_warps=8, BLOCK=512, WARPS=8)


def chain_op(steps=64, name=None):
    """The extern ``chain_<steps>`` (or ``name``): 132 bucket entries, 3 configurations, an unrolled chain of ``steps`` multiply-adds."""

    @tilefish.extern(
        gluon=chain_kernel,
        num_warps=4,
        grid=lambda T, D, BLOCK: (T, tilefish.cdiv(D, BLOCK)),
        by_bucket=_config,
        name=name or f"chain_{steps}",
        reference=lambda x: chain_reference(x, steps),
    )
    def chain(x: Tensor[T, D, f32], y: Out[Tensor[T, D, f32]], n_t: T, n_d: D, BLOCK: const = 256, STEPS: const = steps, WARPS: const = 4): ...

    return chain
