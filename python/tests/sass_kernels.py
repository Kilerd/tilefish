"""Kernels and declarations of the SASS tests on a GPU (``test_gpu_sass.py``; not collected: the name has no ``test_`` prefix).

* ``mm_with(sass)``: the ``tl.dot`` matrix product of ``mm_kernels`` (f16 operands, f32 accumulator, 8 warps), declared with the SASS assertions
  ``sass``; ``MM_CHECKS`` are the ones it meets on each architecture (its MMA kind in PTX and SASS spelling, no TF32, no atomic).
* ``mm_tf32_with(sass)``: the same product of f32 operands with ``input_precision="tf32"``: TF32 tensor-core MMAs on every architecture.
* ``spill_with(sass)``: a row softmax of one warp over a ``[8, 2048]`` f32 tile, 512 values per thread, more than the 255 registers a thread has:
  it spills to local memory on every architecture.
* ``count_with(sass)``: a sum whose result is added to a counter with ``tl.atomic_add``.

Importing this module imports Triton. No ``from __future__ import annotations``: tilefish evaluates annotations.
"""

import triton
import triton.language as tl

import mm_kernels
import tilefish
from mm_kernels import MmK, MmM, MmN
from tilefish import Dim, Out, Tensor, const, f16, f32

R, C = Dim("R"), Dim("C")


def mm_with(sass, name="mm_checked"):
    @tilefish.extern(
        triton=mm_kernels.mm_kernel, num_warps=8, num_stages=4, grid=lambda MmM, MmN: (tilefish.cdiv(MmM, 128), tilefish.cdiv(MmN, 128)), name=name,
        sass=sass,
    )  # fmt: skip
    def mm(
        a_ptr: Tensor[MmM, MmK, f16], b_ptr: Tensor[MmK, MmN, f16], c_ptr: Out[Tensor[MmM, MmN, f32]], m: MmM, n: MmN, k: MmK, *,
        BM: const = 128, BN: const = 128, BK: const = 64,
    ): ...  # fmt: skip

    return mm


# what the f16 product is on each architecture: its MMA in PTX spelling and in SASS spelling (on sm_100 and later a tcgen05 MMA's SASS does not
# spell its kind)
MM_CHECKS = tilefish.sass(
    no_tf32=True,
    max_registers=255,
    forbid=["atomic", "fp64"],
    by_arch={
        "sm_90": tilefish.sass(mma=["wgmma.mma_async f16", "HGMMA.F32"]),
        "sm_100": tilefish.sass(mma=["tcgen05.mma kind::f16", "UTCHMMA"]),
        "sm_103": tilefish.sass(mma=["tcgen05.mma kind::f16", "UTCHMMA"]),
        "sm_120": tilefish.sass(mma=["mma.sync m16n8k16 f16", "HMMA.16816.F32"]),
    },
)


@triton.jit
def mm_tf32_kernel(a_ptr, b_ptr, c_ptr, m, n, k, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, k, BK):
        a = tl.load(a_ptr + offs_m[:, None] * k + (kk + offs_k)[None, :])
        b = tl.load(b_ptr + (kk + offs_k)[:, None] * n + offs_n[None, :])
        acc = tl.dot(a, b, acc, input_precision="tf32")
    tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], acc)


def mm_tf32_with(sass, name="mm_tf32_checked"):
    @tilefish.extern(triton=mm_tf32_kernel, num_warps=4, grid=lambda MmM, MmN: (tilefish.cdiv(MmM, 128), tilefish.cdiv(MmN, 64)), name=name, sass=sass)
    def mm_tf32(
        a_ptr: Tensor[MmM, MmK, f32], b_ptr: Tensor[MmK, MmN, f32], c_ptr: Out[Tensor[MmM, MmN, f32]], m: MmM, n: MmN, k: MmK, *,
        BM: const = 128, BN: const = 64, BK: const = 32,
    ): ...  # fmt: skip

    return mm_tf32


@triton.jit
def spill_kernel(x_ptr, y_ptr, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    x = tl.load(x_ptr + rows[:, None] * COLS + cols[None, :])
    e = tl.exp(x - tl.max(x, 1)[:, None])
    tl.store(y_ptr + rows[:, None] * COLS + cols[None, :], e / tl.sum(e, 1)[:, None])


def spill_with(sass, name="spill_checked"):
    @tilefish.extern(triton=spill_kernel, num_warps=1, grid=lambda R: (tilefish.cdiv(R, 8),), name=name, sass=sass)
    def spill(x_ptr: Tensor[R, 2048, f32], y_ptr: Out[Tensor[R, 2048, f32]], *, ROWS: const = 8, COLS: const = 2048): ...

    return spill


@triton.jit
def count_kernel(x_ptr, acc_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.atomic_add(acc_ptr, tl.sum(tl.load(x_ptr + offs, mask=offs < n, other=0.0), 0))


def count_with(sass, name="count_checked"):
    @tilefish.extern(triton=count_kernel, num_warps=4, grid=lambda C: (tilefish.cdiv(C, 1024),), name=name, sass=sass)
    def count(x_ptr: Tensor[C, f32], acc_ptr: Out[Tensor[1, f32]], n: C, *, BLOCK: const = 1024): ...

    return count
