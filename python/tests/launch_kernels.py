"""Hand-written Triton kernels and a prebuilt PTX of every launch shape (manifest schema 3, sections 4.3 to 4.6), for the gpu tests and the Rust gate
(not collected: the name has no ``test_`` prefix).

Written from scratch for these tests, and public. Importing this module imports Triton, so only the gpu tests and the gate do. No
``from __future__ import annotations``: tilefish evaluates annotations. Every kernel computes what torch computes in the same rounding, so its bytes
are torch's (an ``x * 2`` is exact, so ``fma(x, 2, c)`` rounds once as ``x * 2 + c`` does):

* ``product_scale``: ``y = x * 2 + 1`` over a ``[T, C]`` tensor, one program per 1024 elements: its grid is ``cdiv(T * C, 1024)``, a product of two
  dimensions (one binary for every ``T`` and ``C``);
* ``persistent_scale``: ``y = x * 2 + 0.25`` over ``[N]`` with a persistent grid (``tilefish.persistent()``): the grid is the tiles of 1024, the host
  launches at most one program per SM, and each program walks the tiles with a stride of the programs launched, so its bytes do not depend on the SM
  count; ``persistent_pinned`` is the same kernel declared for 30 SMs (``sm_count=30``: a kernel whose bytes would depend on it);
* ``rows_by_bucket``: ``y = x - max(x, row)`` over ``[TB, 128]`` with a configuration per bucket (``by_bucket=``): ``TB = 64`` is built with 2 warps
  and 16 rows per program, ``TB = 4096`` with 8 warps and 64 rows per program (the grid under each ``BLOCK_T``);
* ``cluster_scale``: ``product_scale``'s arithmetic over ``[N]`` with ``num_ctas=2``: a program is a cluster of two CTAs (each half of its tile), so a
  launch without the cluster gives wrong bytes; ``cluster_swap``: the prebuilt PTX ``ptx/cluster_swap_f32.ptx``, clusters of two blocks that read
  each other's shared memory (``cluster=(2,)``, ``.explicitcluster``);
* ``desc_scale``: ``y = x * 2`` over ``[M, 64]`` f32 through two **device-side** TMA descriptors (``tl.make_tensor_descriptor``), which Triton
  writes into the global scratch of each program: ``global_scratch=256`` (two 128-byte maps per CTA);
* graphs: ``grid_plan`` (``product_scale`` twice over ``[Rb, Cb]``), ``persistent_plan`` (``persistent_scale`` twice), ``bucket_plan``
  (``rows_by_bucket``), ``cluster_plan`` (``cluster_scale``, then ``cluster_swap``) and ``scratch_plan`` (``desc_scale`` twice: two steps with the
  plan's workspace as their scratch).
"""

import triton
import triton.language as tl

import tilefish
from tilefish import Dim, Out, Tensor, const, f32

T, C, N, M = Dim("T"), Dim("C"), Dim("N"), Dim("M")
TB = Dim("TB", buckets=[64, 4096])
Rb, Cb = Dim("Rb", buckets=[8, 33]), Dim("Cb", buckets=[32, 100])
Nb = Dim("Nb", buckets=[1000, 70000])
Mb = Dim("Mb", buckets=[64, 1000])


@triton.jit
def _product_scale(x, y, n_t, n_c, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_t * n_c
    tl.store(y + offs, tl.load(x + offs, mask=m) * 2.0 + 1.0, mask=m)


@tilefish.extern(triton=_product_scale, num_warps=4, grid=lambda T, C, BLOCK: (tilefish.cdiv(T * C, BLOCK),), reference=lambda x: x * 2 + 1)
def product_scale(x: Tensor[T, C, f32], y: Out[Tensor[T, C, f32]], n_t: T, n_c: C, BLOCK: const = 1024): ...


@triton.jit
def _persistent_scale(x, y, n, BLOCK: tl.constexpr):
    tiles = tl.cdiv(n, BLOCK)
    for t in range(tl.program_id(0), tiles, tl.num_programs(0)):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(y + offs, tl.load(x + offs, mask=m) * 2.0 + 0.25, mask=m)


@tilefish.extern(
    triton=_persistent_scale, num_warps=4, grid=lambda N, BLOCK: (tilefish.cdiv(N, BLOCK),), persistent=tilefish.persistent(),
    reference=lambda x: x * 2 + 0.25,
)  # fmt: skip
def persistent_scale(x: Tensor[N, f32], y: Out[Tensor[N, f32]], n: N, BLOCK: const = 1024): ...


@tilefish.extern(
    triton=_persistent_scale, num_warps=4, grid=lambda N, BLOCK: (tilefish.cdiv(N, BLOCK),), persistent=tilefish.persistent(sm_count=30),
    name="persistent_pinned",
)  # fmt: skip
def persistent_pinned(x: Tensor[N, f32], y: Out[Tensor[N, f32]], n: N, BLOCK: const = 1024): ...


@triton.jit
def _rows(x, y, n_rows, BLOCK_T: tl.constexpr):
    r = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    c = tl.arange(0, 128)[None, :]
    m = r < n_rows
    v = tl.load(x + r * 128 + c, mask=m, other=0.0)
    tl.store(y + r * 128 + c, v - tl.max(v, axis=1)[:, None], mask=m)


def _rows_config(TB):
    return tilefish.config(num_warps=2, BLOCK_T=16) if TB == 64 else tilefish.config(num_warps=8, BLOCK_T=64)


@tilefish.extern(
    triton=_rows, num_warps=4, grid=lambda TB, BLOCK_T: (tilefish.cdiv(TB, BLOCK_T),), by_bucket=_rows_config,
    reference=lambda x: x - x.max(dim=1, keepdim=True).values,
)  # fmt: skip
def rows_by_bucket(x: Tensor[TB, 128, f32], y: Out[Tensor[TB, 128, f32]], n_rows: TB, BLOCK_T: const = 32): ...


@triton.jit
def _cluster_scale(x, y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y + offs, tl.load(x + offs, mask=m) * 2.0 + 1.0, mask=m)


@tilefish.extern(triton=_cluster_scale, num_warps=4, num_ctas=2, grid=lambda N, BLOCK: (tilefish.cdiv(N, BLOCK),), reference=lambda x: x * 2 + 1)
def cluster_scale(x: Tensor[N, f32], y: Out[Tensor[N, f32]], n: N, BLOCK: const = 1024): ...


def swap_reference(x):
    """What ``cluster_swap`` writes: each block of 128 elements takes its partner's in the cluster of two, plus its rank (0 or 1)."""
    import torch

    n = x.numel()
    blocks = -(-n // 256) * 2
    padded = torch.zeros(blocks * 128, dtype=x.dtype, device=x.device)
    padded[:n] = x
    rows = padded.view(blocks, 128)
    partner = rows.view(-1, 2, 128).flip(1).reshape(blocks, 128)
    rank = torch.arange(blocks, device=x.device, dtype=x.dtype).remainder(2)[:, None]
    return (partner + rank).reshape(-1)[:n]


@tilefish.extern(
    ptx="ptx/cluster_swap_f32.ptx", symbol="cluster_swap_f32", block=(128,), cluster=(2,), grid=lambda N: (tilefish.cdiv(N, 256),),
    reference=swap_reference,
)  # fmt: skip
def cluster_swap(x: Tensor[N, f32], y: Out[Tensor[N, f32]], n: N): ...


@triton.jit
def _desc_scale(x, y, n_rows, BM: tl.constexpr):
    dx = tl.make_tensor_descriptor(x, shape=[n_rows, 64], strides=[64, 1], block_shape=[BM, 64])
    dy = tl.make_tensor_descriptor(y, shape=[n_rows, 64], strides=[64, 1], block_shape=[BM, 64])
    row = tl.program_id(0) * BM
    dy.store([row, 0], dx.load([row, 0]) * 2.0)


@tilefish.extern(
    triton=_desc_scale, num_warps=4, grid=lambda M, BM: (tilefish.cdiv(M, BM),), global_scratch=256, reference=lambda x: x * 2,
)  # fmt: skip
def desc_scale(x: Tensor[M, 64, f32], y: Out[Tensor[M, 64, f32]], n_rows: M, BM: const = 64): ...


@tilefish.graph
def grid_plan(x: Tensor[Rb, Cb, f32]):
    return product_scale(product_scale(x))


@tilefish.graph
def persistent_plan(x: Tensor[Nb, f32]):
    return persistent_scale(persistent_scale(x))


@tilefish.graph
def bucket_plan(x: Tensor[TB, 128, f32]):
    return rows_by_bucket(x)


@tilefish.graph
def cluster_plan(x: Tensor[Nb, f32]):
    return cluster_swap(cluster_scale(x))


@tilefish.graph
def scratch_plan(x: Tensor[Mb, 64, f32]):
    return desc_scale(desc_scale(x))


def ops():
    """The operators, by name (what the gpu tests export and the Rust gate launches)."""
    return {
        op.name: op
        for op in (product_scale, persistent_scale, persistent_pinned, rows_by_bucket, cluster_scale, cluster_swap, desc_scale)
    }


def graphs():
    """The graphs and their bucket sizes, by name."""
    return {g.name: g for g in (grid_plan, persistent_plan, bucket_plan, cluster_plan, scratch_plan)}
