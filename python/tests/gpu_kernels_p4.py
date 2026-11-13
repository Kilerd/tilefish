"""Hand-written Triton kernels for the gpu tests of M3.3 (not collected: the name has no ``test_`` prefix).

Written from scratch for these tests, and public; declared with ``tilefish.extern``. Importing this module imports triton, so only the gpu tests
do (``host_ops`` imports it with a stand-in for Triton where there is none). No ``from __future__ import annotations``: tilefish evaluates annotations.

* ``fma_plain`` and ``fma_fused``: ``y = a * b + c`` in f32, the one kernel declared with ``enable_fp_fusion=False`` (a multiply and an add, each
  rounded) and ``enable_fp_fusion=True`` (one FMA, one rounding).
* ``qkv_sum`` and ``qkv_sum16``: ``o = q + k + v`` (f32, rounded once to bf16) over three views that the kernel reads with **one** pair of
  strides, ``StrideOf(("q", "k", "v"), 0)`` and ``StrideOf(("q", "k", "v"), 1)``: the q, k and v of a fused ``[R, 3 * 256]`` tensor (offsets 0, 256
  and 512 elements, strides ``(768, 64, 1)`` each); ``qkv_sum16`` is compiled for 16-byte aligned pointers. ``fused_qkv`` is a graph that makes the
  fused tensor with a lowered kernel and passes the three views of it, which a plan binds in one step.
* ``key_mask`` and ``key_mask_any``: a ``[B]`` bf16 mask, ``0`` for the keys ``j < n`` and ``-inf`` for the rest, made from a device scalar ``n``
  alone: no input has ``B``, which only the ``Out`` tensor determines (``key_mask`` has the bucketed ``Rb`` of the plan, ``key_mask_any`` any ``B``).
  ``mask_rows`` is a graph that makes the mask (its ``Rb`` is the graph's) and adds it to the rows of ``x`` with ``apply_mask``.
"""

import triton
import triton.language as tl

import tilefish
from tilefish import DevScalar, Dim, Out, StrideOf, Strided, Tensor, bf16, const, f32, i32

L = Dim("L")
Rq = Dim("Rq")  # the rows of a view
Rb = Dim("Rb", buckets=[8, 24])  # the rows of the plans of fused_qkv and mask_rows
Bm = Dim("Bm")  # the keys of key_mask_any: any number

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


# -- one stride pair for three views of one buffer: StrideOf(("q", "k", "v"), axis) ---------------------------------------------------------


@triton.jit
def qkv_sum_kernel(q, k, v, o, s_row, s_head, D: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, D)
    at = row * s_row + head * s_head + cols  # the one stride pair, for each of the three views
    total = tl.load(q + at).to(tl.float32) + tl.load(k + at).to(tl.float32) + tl.load(v + at).to(tl.float32)
    tl.store(o + (row * tl.num_programs(1) + head) * D + cols, total.to(tl.bfloat16))


def qkv_sum_reference(q, k, v):
    """``(q + k) + v`` in f32, rounded once to bf16: what the kernel computes."""
    import torch

    return ((q.float() + k.float()) + v.float()).to(torch.bfloat16)


@tilefish.extern(triton=qkv_sum_kernel, num_warps=1, grid=lambda Rq: (Rq, 4), reference=qkv_sum_reference)
def qkv_sum(
    q: Strided[Tensor[Rq, 4, 64, bf16]],
    k: Strided[Tensor[Rq, 4, 64, bf16]],
    v: Strided[Tensor[Rq, 4, 64, bf16]],
    o: Out[Tensor[Rq, 4, 64, bf16]],
    s_row: StrideOf(("q", "k", "v"), 0),
    s_head: StrideOf(("q", "k", "v"), 1),
    D: const = 64,
): ...


# the same kernel compiled for 16-byte aligned pointers and strides that are multiples of 8 elements (tt.divisibility = 16 on each)
@tilefish.extern(triton=qkv_sum_kernel, num_warps=1, grid=lambda Rq: (Rq, 4), reference=qkv_sum_reference, name="qkv_sum16")
def qkv_sum16(
    q: Strided[Tensor[Rq, 4, 64, bf16], 16],
    k: Strided[Tensor[Rq, 4, 64, bf16], 16],
    v: Strided[Tensor[Rq, 4, 64, bf16], 16],
    o: Out[Tensor[Rq, 4, 64, bf16]],
    s_row: StrideOf(("q", "k", "v"), 0, divisible_by=8),
    s_head: StrideOf(("q", "k", "v"), 1, divisible_by=8),
    D: const = 64,
): ...


@tilefish.kernel
def doubled(x: Tensor[Rb, 768, bf16]) -> Tensor[Rb, 768, bf16]:
    return (x.to(f32) * 2.0).to(bf16)


@tilefish.graph
def fused_qkv(x: Tensor[Rb, 768, bf16]) -> Tensor[Rb, 4, 64, bf16]:
    """A fused ``[Rb, 768]`` tensor made in the arena, and its q, k and v column blocks read through one stride pair (one step, three views)."""
    qkv = doubled(x).reshape(Rb, 3, 4, 64)
    return qkv_sum16(qkv[:, 0], qkv[:, 1], qkv[:, 2])


# -- a dimension that only an Out tensor determines: a mask of B keys made from a device scalar ---------------------------------------------


@triton.jit
def key_mask_kernel(mask, n_valid, b, BLOCK: tl.constexpr):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(n_valid)
    tl.store(mask + j, tl.where(j < n, 0.0, float("-inf")).to(tl.bfloat16), mask=j < b)


def key_mask_reference(n_valid, b):
    """``0`` for ``j < n``, ``-inf`` after: the mask of ``b`` keys (``b`` is the runtime dimension parameter: no input tells the reference what it is)."""
    import torch

    j = torch.arange(b, device=n_valid.device)
    return torch.where(j < n_valid.reshape(()), 0.0, float("-inf")).to(torch.bfloat16)


@tilefish.extern(triton=key_mask_kernel, num_warps=1, grid=lambda Rb: (tilefish.cdiv(Rb, 128),), reference=key_mask_reference)
def key_mask(mask: Out[Tensor[Rb, bf16]], n_valid: DevScalar[i32], b: Rb, BLOCK: const = 128): ...


@tilefish.extern(triton=key_mask_kernel, num_warps=1, grid=lambda Bm: (tilefish.cdiv(Bm, 128),), reference=key_mask_reference, name="key_mask_any")
def key_mask_any(mask: Out[Tensor[Bm, bf16]], n_valid: DevScalar[i32], b: Bm, BLOCK: const = 128): ...


@triton.jit
def apply_mask_kernel(x, m, y, D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    tl.store(y + row * D + cols, tl.load(x + row * D + cols) + tl.load(m + row).to(tl.float32))


def apply_mask_reference(x, m):
    return x + m.float()[:, None]


@tilefish.extern(triton=apply_mask_kernel, num_warps=1, grid=lambda Rb: (Rb,), reference=apply_mask_reference)
def apply_mask(x: Tensor[Rb, 64, f32], m: Tensor[Rb, bf16], y: Out[Tensor[Rb, 64, f32]], D: const = 64): ...


@tilefish.graph
def mask_rows(x: Tensor[Rb, 64, f32], n: DevScalar[i32]) -> Tensor[Rb, 64, f32]:
    """The mask of the rows of ``x`` from ``n`` alone (its ``Rb`` is the graph's: no argument of the call has it), added to ``x``."""
    m = key_mask(n)
    return apply_mask(x, m)
