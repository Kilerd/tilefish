"""Hand-written Triton kernels for the gpu tests of M3.3 (not collected: the name has no ``test_`` prefix).

Written from scratch for these tests, and public; declared with ``tilefish.extern``. Importing this module imports triton, so only the gpu tests
do (``host_ops`` imports it with a stand-in for Triton where there is none). No ``from __future__ import annotations``: tilefish evaluates annotations.

* ``fma_plain`` and ``fma_fused``: ``y = a * b + c`` in f32, the one kernel declared with ``enable_fp_fusion=False`` (a multiply and an add, each
  rounded) and ``enable_fp_fusion=True`` (one FMA, one rounding).
"""

import triton
import triton.language as tl

import tilefish
from tilefish import Dim, Out, Tensor, const, f32

L = Dim("L")

# -- an extern's contraction: the same source, two declarations, two binaries ------------------------------------------------------------


@triton.jit
def fma_kernel(a, b, c, y, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    av = tl.load(a + offsets, mask=mask)
    bv = tl.load(b + offsets, mask=mask)
    cv = tl.load(c + offsets, mask=mask)
    tl.store(y + offsets, av * bv + cv, mask=mask)


@tilefish.extern(triton=fma_kernel, num_warps=4, grid=lambda L: (tilefish.cdiv(L, 1024),), enable_fp_fusion=False, name="fma_plain")
def fma_plain(a: Tensor[L, f32], b: Tensor[L, f32], c: Tensor[L, f32], y: Out[Tensor[L, f32]], n: L, BLOCK: const = 1024): ...


@tilefish.extern(triton=fma_kernel, num_warps=4, grid=lambda L: (tilefish.cdiv(L, 1024),), enable_fp_fusion=True, name="fma_fused")
def fma_fused(a: Tensor[L, f32], b: Tensor[L, f32], c: Tensor[L, f32], y: Out[Tensor[L, f32]], n: L, BLOCK: const = 1024): ...
