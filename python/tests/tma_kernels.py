"""Hand-written Triton and Gluon kernels that read and write tiles through TMA descriptors, for the gpu tests of ``TensorDesc`` and the Rust gate (not
collected: the name has no ``test_`` prefix).

Written from scratch for these tests, and public. Each kernel takes host tensor descriptors (Triton's ``tl.tensor_descriptor`` parameters, Gluon's
``tma`` copies) and is declared with ``tilefish.TensorDesc``: a call passes tensors, and the host encodes a ``CUtensorMap`` of each at every launch.
Importing this module imports Triton and its Gluon, so only the gpu tests and the gate do. No ``from __future__ import annotations``: tilefish
evaluates annotations.

* ``tile_copy``: ``y = x`` tile by tile, a [64, 64] f16 block in and out (a 128-byte swizzle, the box the block); the edge tiles of a size that is no
  multiple of 64 read zeros past the tensor and drop the stores past it.
* ``shift_rows``: ``y[r] = x[r - 3]``, the first three rows zero: the load's row coordinate is negative for the first tile, which TMA fills with zeros
  (what a causal convolution's descriptor of its input does).
* ``wide_copy``: a [32, 256] bf16 block: its rows are 512 bytes, wider than the 128-byte swizzle span, so the map's box is [32, 64] and the kernel moves
  a block as four boxes.
* ``tma_matmul``: ``c = a @ b.T`` in f16 with an f32 accumulator, ``a``, ``b`` and ``c`` descriptors of [128, 64], [128, 64] and [128, 128] blocks:
  ``tl.dot`` of swizzled shared-memory tiles (tcgen05 on sm_100/103), ``K`` a runtime scalar.
* ``tile_copy_view``: ``tile_copy``'s kernel over a descriptor of a **view** (``TensorDesc(Strided[...])``): a column block of a wider tensor, whose
  row stride is the view's.
* ``gluon_tile_copy``: a Gluon kernel that copies a [64, 64] f16 block global -> shared -> global with ``tma.async_copy_global_to_shared`` and
  ``tma.async_copy_shared_to_global`` (an mbarrier for the load), its descriptors typed with their ``NVMMASharedLayout``.
* graphs: ``tma_chain`` (a lowered kernel doubles ``x`` into the arena, ``shift_rows`` reads that buffer and writes another, ``tile_copy`` writes the
  result: descriptors of arena buffers and of the caller's output, two buckets of rows) and ``tma_view`` (``tile_copy_view`` of the second half of the
  columns of a parameter: a descriptor of a view of the caller's tensor).
"""

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma

import tilefish
from tilefish import Dim, Out, Strided, Tensor, TensorDesc, bf16, const, f16

R, C = Dim("R"), Dim("C")
M, N, K = Dim("M"), Dim("N"), Dim("K")
Rb = Dim("Rb", buckets=[64, 192])  # the rows of the plans
SHIFT = 3


@triton.jit
def tile_copy_kernel(x, y, BM: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0) * BM
    n = tl.program_id(1) * BN
    y.store([m, n], x.load([m, n]))


@triton.jit
def shift_rows_kernel(x, y, SHIFT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0) * BM
    n = tl.program_id(1) * BN
    y.store([m, n], x.load([m - SHIFT, n]))  # rows before the first read as zero (TMA's out-of-bounds fill)


@triton.jit
def tma_matmul_kernel(a, b, c, k_len, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM
    n = tl.program_id(1) * BN
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, k_len, BK):
        acc = tl.dot(a.load([m, k]), b.load([n, k]).T, acc)
    c.store([m, n], acc.to(tl.float16))


@gluon.jit
def gluon_tile_copy_kernel(x, y, BM: gl.constexpr, BN: gl.constexpr):
    tile = gl.allocate_shared_memory(x.dtype, [BM, BN], x.layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    m = gl.program_id(0) * BM
    n = gl.program_id(1) * BN
    mbarrier.expect(bar, x.block_type.nbytes)
    tma.async_copy_global_to_shared(x, [m, n], bar, tile)
    mbarrier.wait(bar, 0)
    mbarrier.invalidate(bar)
    fence_async_shared()
    tma.async_copy_shared_to_global(y, [m, n], tile)
    tma.store_wait(0)


def _grid64(R, C):
    return (tilefish.cdiv(R, 64), tilefish.cdiv(C, 64))


@tilefish.extern(triton=tile_copy_kernel, num_warps=4, grid=_grid64, reference=lambda x: x.clone())
def tile_copy(
    x: TensorDesc(Tensor[R, C, f16], block=(64, 64)),
    y: Out[TensorDesc(Tensor[R, C, f16], block=(64, 64))],
    BM: const = 64,
    BN: const = 64,
): ...


def _shifted(x):
    import torch

    out = torch.zeros_like(x)
    out[SHIFT:] = x[:-SHIFT]
    return out


@tilefish.extern(triton=shift_rows_kernel, num_warps=4, grid=_grid64, reference=_shifted)
def shift_rows(
    x: TensorDesc(Tensor[R, C, f16], block=(64, 64)),
    y: Out[TensorDesc(Tensor[R, C, f16], block=(64, 64))],
    SHIFT: const = SHIFT,
    BM: const = 64,
    BN: const = 64,
): ...


@tilefish.extern(triton=tile_copy_kernel, num_warps=4, grid=lambda R, C: (tilefish.cdiv(R, 32), tilefish.cdiv(C, 256)), name="wide_copy")
def wide_copy(
    x: TensorDesc(Tensor[R, C, bf16], block=(32, 256)),
    y: Out[TensorDesc(Tensor[R, C, bf16], block=(32, 256))],
    BM: const = 32,
    BN: const = 256,
): ...


@tilefish.extern(
    triton=tma_matmul_kernel, num_warps=4, num_stages=3, grid=lambda M, N: (tilefish.cdiv(M, 128), tilefish.cdiv(N, 128))
)
def tma_matmul(
    a: TensorDesc(Tensor[M, K, f16], block=(128, 64)),
    b: TensorDesc(Tensor[N, K, f16], block=(128, 64)),
    c: Out[TensorDesc(Tensor[M, N, f16], block=(128, 128))],
    k_len: K,
    BM: const = 128,
    BN: const = 128,
    BK: const = 64,
): ...


@tilefish.extern(triton=tile_copy_kernel, num_warps=4, grid=_grid64, name="tile_copy_view", reference=lambda x: x.clone())
def tile_copy_view(
    x: TensorDesc(Strided[Tensor[R, C, f16]], block=(64, 64)),
    y: Out[TensorDesc(Tensor[R, C, f16], block=(64, 64))],
    BM: const = 64,
    BN: const = 64,
): ...


@tilefish.extern(gluon=gluon_tile_copy_kernel, num_warps=4, grid=_grid64, reference=lambda x: x.clone())
def gluon_tile_copy(
    x: TensorDesc(Tensor[R, C, f16], block=(64, 64)),
    y: Out[TensorDesc(Tensor[R, C, f16], block=(64, 64))],
    BM: const = 64,
    BN: const = 64,
): ...


@tilefish.kernel
def doubled(x: Tensor[Rb, 256, f16]) -> Tensor[Rb, 256, f16]:
    return (x.to(tilefish.f32) * 2.0).to(f16)


@tilefish.graph
def tma_chain(x: Tensor[Rb, 256, f16]) -> Tensor[Rb, 256, f16]:
    return tile_copy(shift_rows(doubled(x)))


@tilefish.graph
def tma_view(x: Tensor[Rb, 512, f16]) -> Tensor[Rb, 256, f16]:
    _, right = x.split(256, axis=-1)
    return tile_copy_view(right)


def ops():
    """``{name: extern}`` of the descriptor kernels (the Rust gate exports these)."""
    return {
        "tile_copy": tile_copy,
        "shift_rows": shift_rows,
        "wide_copy": wide_copy,
        "tma_matmul": tma_matmul,
        "tile_copy_view": tile_copy_view,
        "gluon_tile_copy": gluon_tile_copy,
    }
