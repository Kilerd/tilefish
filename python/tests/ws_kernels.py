"""Warp-specialised Gluon and Triton kernels over TMA descriptors, for the gpu tests of warp specialisation and the Rust gate (not collected: the
name has no ``test_`` prefix).

Written from scratch for these tests, and public. A kernel is warp-specialised when its partitions run on warps of their own: Gluon's
``gl.warp_specialize(default_args, default_partition, worker_args, worker_partitions, worker_num_warps, worker_num_regs)``, or Triton's own warp
specialisation of a ``tl.range(..., warp_specialize=True)`` loop. The declaration's ``num_warps`` is the warps of the default partition; the workers
add theirs, and Triton 3.5.1 reports the total as the kernel's ``num_warps``, which tilefish launches with (``total_num_warps`` in the manifest).
Importing this module imports Triton and its Gluon, so only the gpu tests and the gate do. No ``from __future__ import annotations``: tilefish
evaluates annotations.

* ``ws_scale``: ``y = 2 * x + 1`` (in f32, rounded to f16 once), a TMA producer partition (1 warp) feeding a compute partition (the default one, 4
  warps) through a ring of three shared-memory tiles with a ``ready`` and an ``empty`` mbarrier each; one program per 64 rows loops over the
  [64, 64] tiles of its rows, and the compute partition stores each result through a TMA descriptor of ``y``. TMA and mbarriers only: sm_90 and up.
  8 warps in all.
* ``ws_matmul``: ``c = a @ b.T``, f16 operands, an f32 accumulator in tensor memory, a 1-warp TMA producer, a 1-warp ``tcgen05_mma`` issuer and a
  4-warp epilogue (the default partition): the shape of a Blackwell GEMM, 4 + 1 + 1 warps, 8 in all. sm_100 and sm_103 only (tcgen05).
* ``ws_matmul_wide``: the same kernel with an 8-warp epilogue: 8 + 1 + 1, which Triton 3.5.1 pads to 12.
* ``tl_ws_matmul``: ``c = a @ b.T`` in Triton, the K loop a ``tl.range(..., warp_specialize=True)``: Triton partitions it itself on sm_100 and up
  (8 warps from the declared 4), and not before.
* graphs: ``ws_chain`` (``ws_scale`` of the input into the arena, then the plain ``tile_copy`` of ``tma_kernels``: steps of two blocks, over two
  buckets of rows) and, on sm_100 and sm_103, ``ws_mm_chain`` (``ws_matmul`` into the arena, then ``ws_scale`` of it).
"""

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_32x32b_reg_layout,
    tcgen05_commit,
    tcgen05_mma,
)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma

import tilefish
import tma_kernels as tk
from tilefish import Dim, Out, Tensor, TensorDesc, const, f16

R, C = Dim("R"), Dim("C")
M, N, K = Dim("M"), Dim("N"), Dim("K")
Rb = Dim("Rb", buckets=[64, 192])  # the rows of ws_chain
Mb = Dim("Mb", buckets=[128, 256])  # the rows of ws_mm_chain
WORKER_REGS = gl.constexpr(24)  # the registers a worker partition asks for (setmaxnreg): a producer or an issuer holds a few scalars


# -- ws_scale: a TMA producer feeding a compute partition ---------------------------------------------------------------------------------------


@gluon.jit
def _scale_load(x, bufs, ready, empty, n_cols, BM: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr):
    m = gl.program_id(0) * BM
    for i in range(gl.cdiv(n_cols, BN)):
        s = i % STAGES
        mbarrier.wait(empty.index(s), ((i // STAGES) & 1) ^ 1)  # the slot is free (a fresh barrier passes the wait for the phase before its first)
        mbarrier.expect(ready.index(s), x.block_type.nbytes)
        tma.async_copy_global_to_shared(x, [m, i * BN], ready.index(s), bufs.index(s))


@gluon.jit
def _scale_compute(y, bufs, out, ready, empty, n_cols, BM: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr, NW: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    m = gl.program_id(0) * BM
    for i in range(gl.cdiv(n_cols, BN)):
        s = i % STAGES
        mbarrier.wait(ready.index(s), (i // STAGES) & 1)
        tile = bufs.index(s).load(layout)
        mbarrier.arrive(empty.index(s), count=1)  # the producer may refill the slot
        result = (tile.to(gl.float32) * 2.0 + 1.0).to(gl.float16)
        tma.store_wait(0)  # the previous tile's store has read the staging tile
        out.store(result)
        fence_async_shared()
        tma.async_copy_shared_to_global(y, [m, i * BN], out)
    tma.store_wait(0)


@gluon.jit
def ws_scale_kernel(x, y, n_cols, BM: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr, NW: gl.constexpr):
    bufs = gl.allocate_shared_memory(x.dtype, [STAGES, BM, BN], x.layout)
    out = gl.allocate_shared_memory(y.dtype, [BM, BN], y.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    for s in gl.static_range(STAGES):
        mbarrier.init(ready.index(s), count=1)
        mbarrier.init(empty.index(s), count=1)
    gl.warp_specialize(
        (y, bufs, out, ready, empty, n_cols, BM, BN, STAGES, NW),
        _scale_compute,
        (x, bufs, ready, empty, n_cols, BM, BN, STAGES),
        [_scale_load],
        [1],
        [WORKER_REGS],
    )


def scaled(x):
    """The reference of ``ws_scale``."""
    import torch

    return (x.to(torch.float32) * 2.0 + 1.0).to(torch.float16)


@tilefish.extern(gluon=ws_scale_kernel, num_warps=4, grid=lambda R: (tilefish.cdiv(R, 64),), reference=lambda x: scaled(x))
def ws_scale(
    x: TensorDesc(Tensor[R, C, f16], block=(64, 64)),
    y: Out[TensorDesc(Tensor[R, C, f16], block=(64, 64))],
    n_cols: C,
    BM: const = 64,
    BN: const = 64,
    STAGES: const = 3,
    NW: const = 4,
): ...


# -- ws_matmul: a TMA producer, an MMA issuer and an epilogue (tcgen05) ------------------------------------------------------------------------


@gluon.jit
def _mm_load(a, b, a_bufs, b_bufs, acc, ready, empty, acc_bar, k_len, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, STAGES: gl.constexpr):
    m = gl.program_id(0) * BM
    n = gl.program_id(1) * BN
    for i in range(gl.cdiv(k_len, BK)):
        s = i % STAGES
        mbarrier.wait(empty.index(s), ((i // STAGES) & 1) ^ 1)
        mbarrier.expect(ready.index(s), a.block_type.nbytes + b.block_type.nbytes)
        tma.async_copy_global_to_shared(a, [m, i * BK], ready.index(s), a_bufs.index(s))
        tma.async_copy_global_to_shared(b, [n, i * BK], ready.index(s), b_bufs.index(s))


@gluon.jit
def _mm_issue(a, b, a_bufs, b_bufs, acc, ready, empty, acc_bar, k_len, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, STAGES: gl.constexpr):
    use_acc = False
    for i in range(gl.cdiv(k_len, BK)):
        s = i % STAGES
        mbarrier.wait(ready.index(s), (i // STAGES) & 1)
        tcgen05_mma(a_bufs.index(s), b_bufs.index(s).permute((1, 0)), acc, use_acc=use_acc, mbarriers=[empty.index(s)])  # frees the slot when done
        use_acc = True
    tcgen05_commit(acc_bar)  # the accumulator is complete once every MMA issued before is


@gluon.jit
def _mm_epilogue(c, acc, acc_bar, c_tile, BM: gl.constexpr, BN: gl.constexpr, NW: gl.constexpr):
    mbarrier.wait(acc_bar, 0)
    reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(BM, BN, [BM, BN], NW)
    values = acc.load(reg_layout)
    c_tile.store(values.to(gl.float16))
    fence_async_shared()
    tma.async_copy_shared_to_global(c, [gl.program_id(0) * BM, gl.program_id(1) * BN], c_tile)
    tma.store_wait(0)


@gluon.jit
def ws_matmul_kernel(a, b, c, k_len, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, STAGES: gl.constexpr, NW: gl.constexpr):
    a_bufs = gl.allocate_shared_memory(a.dtype, [STAGES, BM, BK], a.layout)
    b_bufs = gl.allocate_shared_memory(b.dtype, [STAGES, BN, BK], b.layout)
    c_tile = gl.allocate_shared_memory(c.dtype, [BM, BN], c.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    acc_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    for s in gl.static_range(STAGES):
        mbarrier.init(ready.index(s), count=1)
        mbarrier.init(empty.index(s), count=1)
    mbarrier.init(acc_bar, count=1)
    acc = allocate_tensor_memory(gl.float32, [BM, BN], TensorMemoryLayout(block=(BM, BN), unpacked=True))
    gl.warp_specialize(
        (c, acc, acc_bar, c_tile, BM, BN, NW),
        _mm_epilogue,
        (a, b, a_bufs, b_bufs, acc, ready, empty, acc_bar, k_len, BM, BN, BK, STAGES),
        [_mm_load, _mm_issue],
        [1, 1],
        [WORKER_REGS, WORKER_REGS],
    )


def _grid_mm(M, N):
    return (tilefish.cdiv(M, 128), tilefish.cdiv(N, 128))


def matmul_reference(a, b):
    import torch

    return (a.to(torch.float32) @ b.to(torch.float32).T).to(torch.float16)


@tilefish.extern(gluon=ws_matmul_kernel, num_warps=4, grid=_grid_mm, reference=lambda a, b: matmul_reference(a, b))
def ws_matmul(
    a: TensorDesc(Tensor[M, K, f16], block=(128, 64)),
    b: TensorDesc(Tensor[N, K, f16], block=(128, 64)),
    c: Out[TensorDesc(Tensor[M, N, f16], block=(128, 128))],
    k_len: K,
    BM: const = 128,
    BN: const = 128,
    BK: const = 64,
    STAGES: const = 3,
    NW: const = 4,
): ...


@tilefish.extern(gluon=ws_matmul_kernel, num_warps=8, grid=_grid_mm, name="ws_matmul_wide")
def ws_matmul_wide(
    a: TensorDesc(Tensor[M, K, f16], block=(128, 64)),
    b: TensorDesc(Tensor[N, K, f16], block=(128, 64)),
    c: Out[TensorDesc(Tensor[M, N, f16], block=(128, 128))],
    k_len: K,
    BM: const = 128,
    BN: const = 128,
    BK: const = 64,
    STAGES: const = 3,
    NW: const = 8,
): ...


# -- tl_ws_matmul: Triton's own warp specialisation --------------------------------------------------------------------------------------------


@triton.jit
def tl_ws_matmul_kernel(a, b, c, k_len, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM
    n = tl.program_id(1) * BN
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(0, k_len, BK, warp_specialize=True):
        acc = tl.dot(a.load([m, k]), b.load([n, k]).T, acc)
    c.store([m, n], acc.to(tl.float16))


@tilefish.extern(triton=tl_ws_matmul_kernel, num_warps=4, num_stages=3, grid=_grid_mm)
def tl_ws_matmul(
    a: TensorDesc(Tensor[M, K, f16], block=(128, 64)),
    b: TensorDesc(Tensor[N, K, f16], block=(128, 64)),
    c: Out[TensorDesc(Tensor[M, N, f16], block=(128, 128))],
    k_len: K,
    BM: const = 128,
    BN: const = 128,
    BK: const = 64,
): ...


# -- graphs -----------------------------------------------------------------------------------------------------------------------------------------


@tilefish.graph
def ws_chain(x: Tensor[Rb, 256, f16]) -> Tensor[Rb, 256, f16]:
    return tk.tile_copy(ws_scale(x))


@tilefish.graph
def ws_mm_chain(a: Tensor[Mb, 192, f16], b: Tensor[128, 192, f16]) -> Tensor[Mb, 128, f16]:
    return ws_scale(ws_matmul(a, b))


def tcgen05(capability):
    """Whether a device of this ``(major, minor)`` capability has ``tcgen05`` (sm_100 and sm_103): what ``ws_matmul`` needs."""
    return capability[0] == 10


def ops(capability):
    """``{name: extern}`` of the warp-specialised kernels a device of this capability runs (the Rust gate exports these)."""
    out = {"ws_scale": ws_scale, "tl_ws_matmul": tl_ws_matmul}
    if tcgen05(capability):
        out.update(ws_matmul=ws_matmul, ws_matmul_wide=ws_matmul_wide)
    return out
