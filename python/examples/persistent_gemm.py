"""A persistent, warp-specialised GEMM with epilogue hooks (``docs/rfcs/0003-epilogue-fusion.md``, experiment X8).

``c = a @ b`` with 16-bit operands (bf16 or f16) and an f32 accumulator, in the four layouts of ``cublaslt.gemm`` (``a`` is ``[M, K]`` or stored
transposed as ``[K, M]``, ``b`` is ``[N, K]`` or ``[K, N]``), whose epilogue a ``@tilefish.graph`` fuses into the kernel through the hooks of
``tilefish._epilogue.ABI`` (``ACC`` on the f32 accumulator, ``EPILOGUE`` on the tile rounded to the output's type), as the example GEMMs of
``epilogue_gemm.py`` take them, but fast enough that the fusion pays.

* :func:`gemm_sm100_kernel` (sm_100 and sm_103, Gluon): a persistent kernel (``tilefish.persistent()``: one program per SM walks the output tiles
  in a grouped order) whose partitions run on warps of their own: a one-warp TMA producer that fills a ring of ``STAGES`` shared-memory slots (an
  mbarrier ``ready`` and ``empty`` each), a one-warp ``tcgen05_mma`` issuer that accumulates each tile into one of two tensor-memory buffers, and
  the epilogue on ``PARTS`` partitions of ``NW`` warps (the default partition, and a worker with ``PARTS = 2``, each draining one buffer), which
  reads a finished accumulator in sub-tiles of ``EPI_N`` columns, calls the hooks on each and stores it through shared memory by TMA while the
  issuer accumulates the next tile into the other buffer. The sub-tile loop is not unrolled, so a heavy hook's code is in the binary once. The
  whole ``k`` of a tile is one program's, in order, ``BK`` columns a step: no split-K and no stream-K, so the bytes depend on no configuration,
  number of programs or SM count (measured equal to cuBLASLt's pin on sm_103 wherever that pin does not split ``k``: RFC 0003, X3 and X8). The
  phases carry ``tilefish.probe``s, which a production build does not contain (``docs/instrument.md``). Triton 3.5.1's Gluon has no two-CTA MMA
  and no TMA multicast, so a tile is one CTA's.
* :func:`gemm_sm120_kernel` (sm_120, Triton): TMA loads and ``tl.dot`` (``mma.sync`` ``m16n8k16``, ``HMMA.16816.F32`` in the SASS) in a
  software-pipelined persistent loop, with the epilogue on sub-tiles. On sm_100 and sm_103 the same kernel compiles to ``tcgen05``, which is how
  its logic is tested without an sm_120 device.
* :func:`gemm` declares either for an architecture, with one configuration per bucket of ``M`` (:func:`config_sm100`, :func:`config_sm120`:
  tile sizes, stages, warps; ``tilefish.config`` through ``by_bucket=``) for the ``N`` and ``K`` of a model's projection.

Importing this module imports Triton and its Gluon. No ``from __future__ import annotations``: tilefish evaluates annotations.
"""

import inspect

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
from triton.runtime.jit import constexpr_function

import tilefish
from tilefish import Dim, Out, Tensor, TensorDesc, bf16, const

WORKER_REGS = gl.constexpr(40)  # the registers the producer and the issuer ask for (setmaxnreg): a few scalars and barrier phases each
EPILOGUE_REGS = gl.constexpr(112)  # the registers of the second epilogue partition (PARTS = 2)
LAYOUTS = ("TN", "NN", "TT", "NT")  # cublaslt.gemm's names: the first letter says how b is stored, the second how a is ("N": as the product reads it)


# -- sm_100 / sm_103: Gluon, tcgen05 --------------------------------------------------------------------------------------------------------------


@constexpr_function
def _epilogue_layout(BM, EPI_N, NW):
    """The register layout an epilogue partition of ``NW`` warps (4 or 8: what Triton 3.5.1 lowers a tensor-memory load for) reads a ``BM x EPI_N``
    sub-tile of the accumulator in (``tcgen05.ld 32x32b``): warp ``w`` reads the rows of its quarter of tensor memory (``w % 4``) and the
    ``w // 4``-th of ``NW / 4`` slices of the columns (``get_tmem_32x32b_reg_layout``'s)."""
    if BM == 64:
        return get_tmem_32x32b_reg_layout(BM, EPI_N, [BM, EPI_N], NW)
    groups = NW // 4
    return gl.BlockedLayout([1, EPI_N // groups], [32, 1], [4, groups], [0, 1])


@gluon.jit
def _tile_of(t, num_m, num_n, GROUP_M: gl.constexpr):
    """The (row, column) tile of the ``t``-th tile in a grouped order: ``GROUP_M`` tiles of rows share each column of tiles (``b``'s tiles are read
    from L2 that many times in a row), the last group narrower."""
    width = GROUP_M * num_n
    first = (t // width) * GROUP_M
    size = gl.minimum(num_m - first, GROUP_M)
    return first + (t % width) % size, (t % width) // size


@gluon.jit
def _copy_in(desc, x0, x1, bar, buf, ROWS: gl.constexpr, COLS: gl.constexpr):
    """TMA copies of the ``[ROWS, COLS]`` tile of ``desc`` at ``(x0, x1)`` into ``buf``, as many boxes of the descriptor's block as tile it (one when
    the block is the tile: a configuration per bucket may take larger tiles than the block the declaration fixes)."""
    BR: gl.constexpr = desc.block_type.shape[0]
    BC: gl.constexpr = desc.block_type.shape[1]
    if ROWS == BR and COLS == BC:
        tma.async_copy_global_to_shared(desc, [x0, x1], bar, buf)
    else:
        for r in gl.static_range(ROWS // BR):
            for q in gl.static_range(COLS // BC):
                tma.async_copy_global_to_shared(desc, [x0 + r * BR, x1 + q * BC], bar, buf.slice(r * BR, BR, dim=0).slice(q * BC, BC, dim=1))


@gluon.jit
def _copy_out(desc, x0, x1, buf, ROWS: gl.constexpr, COLS: gl.constexpr):
    """TMA stores of ``buf`` (``[ROWS, COLS]``) into ``desc`` at ``(x0, x1)``, as boxes of the descriptor's block."""
    BR: gl.constexpr = desc.block_type.shape[0]
    BC: gl.constexpr = desc.block_type.shape[1]
    if ROWS == BR and COLS == BC:
        tma.async_copy_shared_to_global(desc, [x0, x1], buf)
    else:
        for r in gl.static_range(ROWS // BR):
            for q in gl.static_range(COLS // BC):
                tma.async_copy_shared_to_global(desc, [x0 + r * BR, x1 + q * BC], buf.slice(r * BR, BR, dim=0).slice(q * BC, BC, dim=1))


@gluon.jit
def _sm100_load(a, b, c, a_bufs, b_bufs, c_smem, acc_tm, ready, empty, acc_ready, acc_empty, m, n, k, BM: gl.constexpr, BN: gl.constexpr,
                BK: gl.constexpr, STAGES: gl.constexpr, EPI_N: gl.constexpr, SPLIT: gl.constexpr, GROUP_M: gl.constexpr, A_KMAJOR: gl.constexpr,
                B_KMAJOR: gl.constexpr, NW: gl.constexpr, PARTS: gl.constexpr, ACC: gl.constexpr, ACC_ARGS, EPILOGUE: gl.constexpr, EPILOGUE_ARGS):  # fmt: skip
    """The TMA producer (one warp): for each tile of this program, each step of ``k``: wait for a free slot, load the ``a`` tile and the ``b`` tile
    (``SPLIT`` of them: the rows ``h * n / SPLIT + off_n`` of ``b``, side by side) into it."""
    chunk = n // SPLIT
    num_m = gl.cdiv(m, BM)
    num_n = gl.cdiv(chunk, BN)
    steps = gl.cdiv(k, BK)
    nbytes: gl.constexpr = (BM + SPLIT * BN) * BK * a.dtype.primitive_bitwidth // 8
    i = 0
    for t in range(gl.program_id(0), num_m * num_n, gl.num_programs(0)):
        pid_m, pid_n = _tile_of(t, num_m, num_n, GROUP_M)
        off_m = pid_m * BM
        off_n = pid_n * BN
        for kk in range(steps):
            s = i % STAGES
            tilefish.probe("load.wait")
            mbarrier.wait(empty.index(s), ((i // STAGES) & 1) ^ 1)  # (a fresh barrier passes the wait for the phase before its first)
            tilefish.probe("load.issue")
            bar = ready.index(s)
            mbarrier.expect(bar, nbytes)
            if A_KMAJOR:
                _copy_in(a, off_m, kk * BK, bar, a_bufs.index(s), BM, BK)
            else:
                _copy_in(a, kk * BK, off_m, bar, a_bufs.index(s), BK, BM)
            for h in gl.static_range(SPLIT):
                if B_KMAJOR:
                    _copy_in(b, h * chunk + off_n, kk * BK, bar, b_bufs.index(s).slice(h * BN, BN, dim=0), BN, BK)
                else:
                    _copy_in(b, kk * BK, h * chunk + off_n, bar, b_bufs.index(s).slice(h * BN, BN, dim=1), BK, BN)
            i += 1


@gluon.jit
def _sm100_mma(a, b, c, a_bufs, b_bufs, c_smem, acc_tm, ready, empty, acc_ready, acc_empty, m, n, k, BM: gl.constexpr, BN: gl.constexpr,
               BK: gl.constexpr, STAGES: gl.constexpr, EPI_N: gl.constexpr, SPLIT: gl.constexpr, GROUP_M: gl.constexpr, A_KMAJOR: gl.constexpr,
               B_KMAJOR: gl.constexpr, NW: gl.constexpr, PARTS: gl.constexpr, ACC: gl.constexpr, ACC_ARGS, EPILOGUE: gl.constexpr, EPILOGUE_ARGS):  # fmt: skip
    """The MMA issuer (one warp): for each tile, wait until the epilogue has read the accumulator buffer it takes (two, alternating), then one
    ``tcgen05_mma`` per step of ``k``, in order, each freeing its slot when it completes, and a commit that tells the epilogue the tile is done."""
    chunk = n // SPLIT
    num_m = gl.cdiv(m, BM)
    num_n = gl.cdiv(chunk, BN)
    steps = gl.cdiv(k, BK)
    i = 0
    tc = 0
    for t in range(gl.program_id(0), num_m * num_n, gl.num_programs(0)):
        buf = tc % 2
        tilefish.probe("mma.acc_wait")
        mbarrier.wait(acc_empty.index(buf), ((tc // 2) & 1) ^ 1)
        acc = acc_tm.index(buf)
        use_acc = False
        for kk in range(steps):
            s = i % STAGES
            tilefish.probe("mma.wait")
            mbarrier.wait(ready.index(s), (i // STAGES) & 1)
            tilefish.probe("mma.issue")
            if A_KMAJOR:
                if B_KMAJOR:
                    tcgen05_mma(a_bufs.index(s), b_bufs.index(s).permute((1, 0)), acc, use_acc=use_acc, mbarriers=[empty.index(s)])
                else:
                    tcgen05_mma(a_bufs.index(s), b_bufs.index(s), acc, use_acc=use_acc, mbarriers=[empty.index(s)])
            else:
                if B_KMAJOR:
                    tcgen05_mma(a_bufs.index(s).permute((1, 0)), b_bufs.index(s).permute((1, 0)), acc, use_acc=use_acc, mbarriers=[empty.index(s)])
                else:
                    tcgen05_mma(a_bufs.index(s).permute((1, 0)), b_bufs.index(s), acc, use_acc=use_acc, mbarriers=[empty.index(s)])
            use_acc = True
            i += 1
        tcgen05_commit(acc_ready.index(buf))
        tilefish.probe("mma.done")
        tc += 1


@gluon.jit
def _sm100_epilogue(c, acc_tm, acc_ready, acc_empty, c_smem, m, n, BM: gl.constexpr, BN: gl.constexpr, EPI_N: gl.constexpr, SPLIT: gl.constexpr,
                    GROUP_M: gl.constexpr, NW: gl.constexpr, PART: gl.constexpr, PARTS: gl.constexpr, ACC: gl.constexpr, ACC_ARGS,
                    EPILOGUE: gl.constexpr, EPILOGUE_ARGS):  # fmt: skip
    """The epilogue (``NW`` warps; the default partition, and with ``PARTS = 2`` a worker partition as well, each draining one of the two
    accumulator buffers, so a heavy epilogue has twice the warps): for each of its tiles, wait for the accumulator, read it in sub-tiles of
    ``EPI_N`` columns (``tcgen05.ld 32x32b``; the buffer is handed back to the issuer once the last is read), and for each: the ``ACC`` hook, the
    rounding to ``c``'s type, the ``EPILOGUE`` hook, a store into one of its two shared-memory tiles and a TMA store of it (the previous store of
    that tile drained first)."""
    chunk = n // SPLIT
    num_m = gl.cdiv(m, BM)
    num_n = gl.cdiv(chunk, BN)
    reg: gl.constexpr = _epilogue_layout(BM, EPI_N, NW)
    SUB: gl.constexpr = BN // EPI_N
    tc = PART
    sc = 0
    for t in range(gl.program_id(0) + PART * gl.num_programs(0), num_m * num_n, PARTS * gl.num_programs(0)):
        pid_m, pid_n = _tile_of(t, num_m, num_n, GROUP_M)
        off_m = pid_m * BM
        off_n = pid_n * BN
        buf = tc % 2
        tilefish.probe("epi.wait")
        mbarrier.wait(acc_ready.index(buf), (tc // 2) & 1)
        tilefish.probe("epi.tile")
        # the accumulator as SPLIT * SUB sub-tiles of EPI_N columns (tensor-memory columns j * EPI_N on), read in a loop that is not unrolled:
        # the hooks' code once in the binary, however many sub-tiles (an unrolled loop repeats a heavy hook's code, and its instruction fetches)
        subs = acc_tm.index(buf)._reinterpret(gl.float32, [SPLIT * SUB, BM, EPI_N], TensorMemoryLayout(block=(BM, EPI_N), unpacked=True))
        rows = off_m + gl.arange(0, BM, layout=gl.SliceLayout(1, reg))
        row_mask = rows < m
        for j in range(SUB):
            cols = off_n + j * EPI_N + gl.arange(0, EPI_N, layout=gl.SliceLayout(0, reg))
            col_mask = cols < chunk
            if SPLIT == 1:
                x = subs.index(j).load(reg)
                if j == SUB - 1:
                    gl.thread_barrier()  # every warp has read its rows of the buffer
                    mbarrier.arrive(acc_empty.index(buf), count=1)
                tilefish.probe("epi.ld")
                tile = ACC(x, rows, cols, row_mask, col_mask, ACC_ARGS)
                tile = EPILOGUE(tile.to(c.dtype), rows, cols, row_mask, col_mask, EPILOGUE_ARGS)
                staged = c_smem.index(2 * PART + sc % 2)
                tma.store_wait(1)
                tilefish.probe("epi.hook")
                staged.store(tile.to(c.dtype))
                fence_async_shared()
                _copy_out(c, off_m, off_n + j * EPI_N, staged, BM, EPI_N)
                sc += 1
            else:
                g = subs.index(j).load(reg)
                u = subs.index(SUB + j).load(reg)
                if j == SUB - 1:
                    gl.thread_barrier()
                    mbarrier.arrive(acc_empty.index(buf), count=1)
                tiles = ACC((g, u), rows, cols, row_mask, col_mask, ACC_ARGS)
                if len(tiles) == 2:
                    tiles = EPILOGUE((tiles[0].to(c.dtype), tiles[1].to(c.dtype)), rows, cols, row_mask, col_mask, EPILOGUE_ARGS)
                for h in gl.static_range(len(tiles)):  # the hook returned one tile (a fused gated activation) or both: c's rows hold that many chunks
                    staged = c_smem.index(2 * PART + sc % 2)
                    tma.store_wait(1)
                    staged.store(tiles[h].to(c.dtype))
                    fence_async_shared()
                    _copy_out(c, off_m, h * chunk + off_n + j * EPI_N, staged, BM, EPI_N)
                    sc += 1
        tilefish.probe("epi.done")
        tc += PARTS
    tma.store_wait(0)


@gluon.jit
def _sm100_epilogue_worker(a, b, c, a_bufs, b_bufs, c_smem, acc_tm, ready, empty, acc_ready, acc_empty, m, n, k, BM: gl.constexpr, BN: gl.constexpr,
                           BK: gl.constexpr, STAGES: gl.constexpr, EPI_N: gl.constexpr, SPLIT: gl.constexpr, GROUP_M: gl.constexpr,
                           A_KMAJOR: gl.constexpr, B_KMAJOR: gl.constexpr, NW: gl.constexpr, PARTS: gl.constexpr, ACC: gl.constexpr, ACC_ARGS,
                           EPILOGUE: gl.constexpr, EPILOGUE_ARGS):  # fmt: skip
    """The second epilogue partition (``PARTS = 2``): the odd tiles of the program, from the second accumulator buffer."""
    _sm100_epilogue(c, acc_tm, acc_ready, acc_empty, c_smem, m, n, BM, BN, EPI_N, SPLIT, GROUP_M, NW, 1, PARTS, ACC, ACC_ARGS, EPILOGUE, EPILOGUE_ARGS)


@gluon.jit
def gemm_sm100_kernel(a, b, c, m, n, k, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, STAGES: gl.constexpr, EPI_N: gl.constexpr,
                      GROUP_M: gl.constexpr, SPLIT: gl.constexpr, A_KMAJOR: gl.constexpr, B_KMAJOR: gl.constexpr, NW: gl.constexpr,
                      PARTS: gl.constexpr, ACC: gl.constexpr, ACC_ARGS, EPILOGUE: gl.constexpr, EPILOGUE_ARGS):  # fmt: skip
    """``c = a @ b`` (``a``: ``[m, k]`` when ``A_KMAJOR``, else ``[k, m]``; ``b``: ``[n, k]`` when ``B_KMAJOR``, else ``[k, n]``; all TMA
    descriptors), one ``BM x BN`` tile of ``c`` (``SPLIT`` of them side by side: the dual accumulator of a gated activation) per step of the
    persistent loop, the epilogue on ``PARTS`` partitions of ``NW`` warps (1, or 2 for a heavy epilogue). Shared memory: ``STAGES`` slots of
    ``(BM + SPLIT * BN) * BK`` operands and two staging tiles of ``BM x EPI_N`` per epilogue partition; tensor memory: two accumulators of
    ``BM x SPLIT * BN`` f32 (``2 * SPLIT * BN <= 512`` columns)."""
    gl.static_assert(2 * SPLIT * BN <= 512, "two accumulators of SPLIT * BN f32 columns fill at most the 512 columns of tensor memory")
    gl.static_assert(BN % EPI_N == 0, "the epilogue's sub-tiles divide the tile")
    if A_KMAJOR:
        a_bufs = gl.allocate_shared_memory(a.dtype, [STAGES, BM, BK], a.layout)
    else:
        a_bufs = gl.allocate_shared_memory(a.dtype, [STAGES, BK, BM], a.layout)
    if B_KMAJOR:
        b_bufs = gl.allocate_shared_memory(b.dtype, [STAGES, SPLIT * BN, BK], b.layout)
    else:
        b_bufs = gl.allocate_shared_memory(b.dtype, [STAGES, BK, SPLIT * BN], b.layout)
    c_smem = gl.allocate_shared_memory(c.dtype, [2 * PARTS, BM, EPI_N], c.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    acc_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for s in gl.static_range(STAGES):
        mbarrier.init(ready.index(s), count=1)
        mbarrier.init(empty.index(s), count=1)
    for s in gl.static_range(2):
        mbarrier.init(acc_ready.index(s), count=1)
        mbarrier.init(acc_empty.index(s), count=1)
    acc_tm = allocate_tensor_memory(gl.float32, [2, BM, SPLIT * BN], TensorMemoryLayout(block=(BM, SPLIT * BN), unpacked=True))
    tilefish.probe("init")
    # (every worker partition takes the same arguments; the tuples are written out in each call: a tuple variable cannot hold a jit function)
    if PARTS == 1:
        gl.warp_specialize(
            (c, acc_tm, acc_ready, acc_empty, c_smem, m, n, BM, BN, EPI_N, SPLIT, GROUP_M, NW, 0, PARTS, ACC, ACC_ARGS, EPILOGUE, EPILOGUE_ARGS),
            _sm100_epilogue,
            (a, b, c, a_bufs, b_bufs, c_smem, acc_tm, ready, empty, acc_ready, acc_empty, m, n, k, BM, BN, BK, STAGES, EPI_N, SPLIT, GROUP_M,
             A_KMAJOR, B_KMAJOR, NW, PARTS, ACC, ACC_ARGS, EPILOGUE, EPILOGUE_ARGS),
            [_sm100_load, _sm100_mma],
            [1, 1],
            [WORKER_REGS, WORKER_REGS],
        )
    else:
        gl.static_assert(PARTS == 2, "the epilogue runs on one or two partitions")
        gl.warp_specialize(
            (c, acc_tm, acc_ready, acc_empty, c_smem, m, n, BM, BN, EPI_N, SPLIT, GROUP_M, NW, 0, PARTS, ACC, ACC_ARGS, EPILOGUE, EPILOGUE_ARGS),
            _sm100_epilogue,
            (a, b, c, a_bufs, b_bufs, c_smem, acc_tm, ready, empty, acc_ready, acc_empty, m, n, k, BM, BN, BK, STAGES, EPI_N, SPLIT, GROUP_M,
             A_KMAJOR, B_KMAJOR, NW, PARTS, ACC, ACC_ARGS, EPILOGUE, EPILOGUE_ARGS),
            [_sm100_epilogue_worker, _sm100_load, _sm100_mma],
            [NW, 1, 1],
            [EPILOGUE_REGS, WORKER_REGS, WORKER_REGS],
        )


# -- sm_120: Triton, mma.sync ---------------------------------------------------------------------------------------------------------------------


@triton.jit
def _tile_of_tl(t, num_m, num_n, GROUP_M: tl.constexpr):
    width = GROUP_M * num_n
    first = (t // width) * GROUP_M
    size = tl.minimum(num_m - first, GROUP_M)
    return first + (t % width) % size, (t % width) // size


@triton.jit
def _sm120_store(c, tile, rows, cols, row_mask, col_mask, x0, x1, ACC: tl.constexpr, ACC_ARGS, EPILOGUE: tl.constexpr, EPILOGUE_ARGS):
    """One sub-tile's epilogue: the ``ACC`` hook, the rounding, the ``EPILOGUE`` hook, a TMA store at ``(x0, x1)``."""
    tile = ACC(tile, rows, cols, row_mask, col_mask, ACC_ARGS)
    tile = EPILOGUE(tile.to(c.dtype), rows, cols, row_mask, col_mask, EPILOGUE_ARGS)
    c.store([x0, x1], tile.to(c.dtype))


@triton.jit
def gemm_sm120_kernel(a, b, c, m, n, k, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, STAGES: tl.constexpr, EPI_N: tl.constexpr,
                      GROUP_M: tl.constexpr, SPLIT: tl.constexpr, A_KMAJOR: tl.constexpr, B_KMAJOR: tl.constexpr, NW: tl.constexpr,
                      ACC: tl.constexpr, ACC_ARGS, EPILOGUE: tl.constexpr, EPILOGUE_ARGS):  # fmt: skip
    """``c = a @ b`` as :func:`gemm_sm100_kernel` (the same parameters; ``NW`` unused), in Triton: a persistent loop over the tiles in the grouped
    order, the ``k`` loop software-pipelined over ``STAGES`` TMA loads, ``tl.dot`` (``mma.sync`` on sm_120, ``tcgen05`` on sm_100), the epilogue on
    ``BN // EPI_N`` (1 or 2) sub-tiles, each stored by TMA. The descriptors' blocks are the tiles."""
    tl.static_assert(BN // EPI_N == 1 or BN // EPI_N == 2, "the epilogue runs on one or two sub-tiles")
    chunk = n // SPLIT
    num_m = tl.cdiv(m, BM)
    num_n = tl.cdiv(chunk, BN)
    for t in tl.range(tl.program_id(0), num_m * num_n, tl.num_programs(0)):
        pid_m, pid_n = _tile_of_tl(t, num_m, num_n, GROUP_M)
        off_m = pid_m * BM
        off_n = pid_n * BN
        acc0 = tl.zeros((BM, BN), dtype=tl.float32)
        acc1 = tl.zeros((BM, BN), dtype=tl.float32)
        for kk in tl.range(0, k, BK, num_stages=STAGES):
            if A_KMAJOR:
                x = a.load([off_m, kk])
            else:
                x = a.load([kk, off_m]).T
            if B_KMAJOR:
                acc0 = tl.dot(x, b.load([off_n, kk]).T, acc0)
                if SPLIT == 2:
                    acc1 = tl.dot(x, b.load([chunk + off_n, kk]).T, acc1)
            else:
                acc0 = tl.dot(x, b.load([kk, off_n]), acc0)
                if SPLIT == 2:
                    acc1 = tl.dot(x, b.load([kk, chunk + off_n]), acc1)
        rows = off_m + tl.arange(0, BM)
        row_mask = rows < m
        if SPLIT == 1:
            if BN == EPI_N:
                cols = off_n + tl.arange(0, BN)
                _sm120_store(c, acc0, rows, cols, row_mask, cols < chunk, off_m, off_n, ACC, ACC_ARGS, EPILOGUE, EPILOGUE_ARGS)
            else:
                h0, h1 = acc0.reshape(BM, 2, EPI_N).permute(0, 2, 1).split()
                cols = off_n + tl.arange(0, EPI_N)
                _sm120_store(c, h0, rows, cols, row_mask, cols < chunk, off_m, off_n, ACC, ACC_ARGS, EPILOGUE, EPILOGUE_ARGS)
                _sm120_store(c, h1, rows, cols + EPI_N, row_mask, cols + EPI_N < chunk, off_m, off_n + EPI_N, ACC, ACC_ARGS, EPILOGUE, EPILOGUE_ARGS)
        else:
            tl.static_assert(BN == EPI_N, "a dual GEMM's epilogue sees whole tiles")
            cols = off_n + tl.arange(0, BN)
            col_mask = cols < chunk
            tiles = ACC((acc0, acc1), rows, cols, row_mask, col_mask, ACC_ARGS)
            if len(tiles) == 2:
                tiles = EPILOGUE((tiles[0].to(c.dtype), tiles[1].to(c.dtype)), rows, cols, row_mask, col_mask, EPILOGUE_ARGS)
            for h in tl.static_range(len(tiles)):  # one tile (a fused gated activation) or both: c's rows hold that many chunks
                c.store([off_m, h * chunk + off_n], tiles[h].to(c.dtype))


# -- the declarations -----------------------------------------------------------------------------------------------------------------------------

SMEM_BYTES = {100: 232448, 103: 232448, 120: 101376}  # the dynamic shared memory a block may have (227 KB on sm_100/103, 99 KB on sm_120)
SMEM_RESERVE = 2048  # the barriers and what the compiler adds


def _stub(params: "list[tuple[str, object, object]]", name: str) -> object:
    """A stub function with these ``(name, annotation, default)`` parameters (what a declaration in a module writes by hand)."""
    sig = inspect.Signature([inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=a, default=d) for n, a, d in params])

    def stub(*args, **kwargs): ...

    stub.__signature__ = sig
    stub.__name__ = stub.__qualname__ = name
    stub.__module__ = __name__
    return stub


_E = inspect.Parameter.empty


def _stages(arch: int, BM: int, BN: int, BK: int, EPI_N: int, split: int, parts: int = 1, top: int = 8) -> int:
    """The most stages (at most ``top``) whose operand slots and two staging tiles per epilogue partition fit the architecture's shared memory."""
    per = (BM + split * BN) * BK * 2
    room = SMEM_BYTES[arch] - SMEM_RESERVE - 2 * parts * BM * EPI_N * 2
    return max(2, min(top, room // per))


# The cost model of config_sm100, fitted on a B300 slice of 30 SMs (RFC 0003, X8): a k step of a 128-row tile takes BN / 256 * 271 ns of tcgen05
# (a 64-row tile half that), or its operands' bytes at 220 GB/s into an SM, whichever is longer; a tile adds 0.3 us; the programs walk the tiles in
# ceil(tiles / SMs) rounds. Over the 64 shapes of the suite the tile it picks is 0.4% slower than the best of the sweep on average, 8% at most.
MMA_NS_256 = 271.0
SM_GBPS = 220.0
TILE_US = 0.3


def _model_us(m: int, chunk: int, k: int, BM: int, BN: int, split: int, sm_count: int) -> float:
    tiles = -(-m // BM) * -(-chunk // BN)
    rounds = -(-tiles // sm_count)
    mma = split * BN / 256 * MMA_NS_256 * (1.0 if BM == 128 else 0.5) / 1000.0
    load = (BM + split * BN) * 64 * 2 / (SM_GBPS * 1000.0)
    return rounds * (-(-k // 64) * max(mma, load) + TILE_US)


def config_sm100(m: int, n: int, k: int, *, split: int = 1, sm_count: int = 148, nw: "int | None" = None, parts: "int | None" = None) -> "dict[str, int]":
    """The configuration of :func:`gemm_sm100_kernel` for ``m`` rows (a bucket of ``M``), ``n`` columns (both halves of a dual GEMM) and ``k``
    on a device of ``sm_count`` SMs: the tile of the least modelled time (:func:`_model_us`: rows of 64 only for ``m <= 64``, 32 to 256 columns,
    ``2 * split * BN <= 512`` tensor-memory columns), as many stages as shared memory holds (at most 8), and the epilogue on sub-tiles of 32
    columns in ``parts`` partitions of ``nw`` warps: by default 2 of 8 (measured on a B300 slice: a heavy hook, erf GELU or a periodic
    activation, runs 1.1 to 2.2 times faster than on one partition of 4 warps, and the GEMM alone is at most 0.4 us slower), 1 of 4 for
    ``m <= 64`` (a tile of 64 rows has a light epilogue, and the launch is short). A speed choice only: no configuration changes the bytes (the
    whole ``k`` of a tile is one program's, in order)."""
    chunk = n // split
    widest = min(256 // split, max(32, 1 << (max(chunk, 1) - 1).bit_length()))
    tiles = [(BM, BN) for BM in ((64, 128) if m <= 64 else (128,)) for BN in (32, 64, 128, 256) if BN <= widest]
    BM, BN = min(tiles, key=lambda t: (_model_us(m, chunk, k, t[0], t[1], split, sm_count), -t[1]))
    nw = nw if nw is not None else 4 if m <= 64 else 8
    parts = parts if parts is not None else 1 if m <= 64 else 2
    EPI_N = min(BN, 32)
    return {"BM": BM, "BN": BN, "BK": 64, "STAGES": _stages(100, BM, BN, 64, EPI_N, split, parts), "EPI_N": EPI_N, "GROUP_M": 8, "NW": nw, "PARTS": parts}


def config_sm120(m: int, n: int, k: int, *, split: int = 1, sm_count: int = 170) -> "dict[str, int]":
    """The configuration of :func:`gemm_sm120_kernel`: tiles of 128 x 128 (64 columns a side for a dual GEMM: two register accumulators), ``BK``
    64, three stages (Triton's pipeliner keeps ``STAGES - 1`` slots of 32 KB in flight, and the store stages 16 KB: sm_120 has 99 KB a block), the epilogue on sub-tiles of 64 columns, 8 warps
    (4 for a few rows). Its descriptors' blocks are the tiles, so every bucket has the same tile and only the warps and the grouping change. Not
    measured on an sm_120 device yet (a later gate's)."""
    BN = 128 if split == 1 else 64
    return {"BM": 128, "BN": BN, "BK": 64, "STAGES": 3, "EPI_N": 64, "GROUP_M": 8, "NW": 4 if m <= 64 else 8}


def _layout_flags(layout: str) -> "tuple[bool, bool]":
    """Whether ``a`` is stored ``[M, K]`` and whether ``b`` is stored ``[N, K]`` (K-major, each), for one of :data:`LAYOUTS`."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout is one of {list(LAYOUTS)} (cublaslt.gemm's names), got {layout!r}")
    return layout[1] == "N", layout[0] == "T"


def _reference(layout: str, out_dtype: object, acc: bool = False):
    a_k, b_k = _layout_flags(layout)

    def ref(a, b):
        y = (a if a_k else a.T).float() @ (b.T if b_k else b).float()
        return y if acc else y.to(a.dtype if out_dtype is None else out_dtype)

    return ref


def _by_bucket(M, configs: "dict[int, dict[str, int]]", keys: "tuple[str, ...]"):
    """``by_bucket=`` over ``M``: each bucket's configuration (warps and the consts ``keys``)."""
    if not configs:
        return None

    def by_bucket(**values):
        (mv,) = values.values()
        c = configs[mv]
        return tilefish.config(num_warps=c["NW"], **{key: c[key] for key in keys})

    by_bucket.__signature__ = inspect.Signature([inspect.Parameter(M.name, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    return by_bucket


def _sm100(M, N, K, dtype, out_dtype, layout, split, configs, default, name, sass, persistent):
    a_k, b_k = _layout_flags(layout)
    every = [default, *configs.values()]
    bm = min(c["BM"] for c in every)  # the descriptors' blocks: the smallest tile of any bucket (a larger one is copied as several boxes)
    bn = min(c["BN"] for c in every)
    en = min(c["EPI_N"] for c in every)
    bk = default["BK"]
    if any(c["BK"] != bk for c in every):
        raise ValueError("every bucket's BK is the same (the descriptors' inner block)")
    odt = dtype if out_dtype is None else out_dtype
    a_ann = TensorDesc(Tensor[M, K, dtype], block=(bm, bk)) if a_k else TensorDesc(Tensor[K, M, dtype], block=(bk, bm))
    b_ann = TensorDesc(Tensor[N, K, dtype], block=(bn, bk)) if b_k else TensorDesc(Tensor[K, N, dtype], block=(bk, bn))
    stub = _stub(
        [
            ("a", a_ann, _E),
            ("b", b_ann, _E),
            ("c", Out[TensorDesc(Tensor[M, N, odt], block=(bm, en))], _E),
            ("m", M, _E),
            ("n", N, _E),
            ("k", K, _E),
            ("BM", const, default["BM"]),
            ("BN", const, default["BN"]),
            ("BK", const, bk),
            ("STAGES", const, default["STAGES"]),
            ("EPI_N", const, default["EPI_N"]),
            ("GROUP_M", const, default["GROUP_M"]),
            ("SPLIT", const, split),
            ("A_KMAJOR", const, int(a_k)),
            ("B_KMAJOR", const, int(b_k)),
            ("NW", const, default["NW"]),
            ("PARTS", const, default["PARTS"]),
        ],
        name,
    )
    hooks = (
        tilefish.hook("ACC", value="acc", out="c", split=split, block_n="EPI_N", reference=_reference(layout, None, acc=True)),
        tilefish.hook("EPILOGUE", value="out", out="c", split=split, block_n="EPI_N"),
    )
    return tilefish.extern(
        gluon=gemm_sm100_kernel,
        name=name,
        num_warps=default["NW"],
        grid=lambda m, n, BM, BN, SPLIT: (tilefish.cdiv(m, BM) * tilefish.cdiv(n, SPLIT * BN),),
        persistent=tilefish.persistent() if persistent else None,
        by_bucket=_by_bucket(M, configs, ("BM", "BN", "STAGES", "EPI_N", "GROUP_M", "NW", "PARTS")),
        reference=_reference(layout, out_dtype),
        epilogue=hooks,
        sass=sass,
    )(stub)


def _sm120(M, N, K, dtype, out_dtype, layout, split, configs, default, name, sass, persistent):
    a_k, b_k = _layout_flags(layout)
    tile = ("BM", "BN", "BK", "EPI_N")
    for c in configs.values():
        if any(c[key] != default[key] for key in tile):
            raise ValueError(f"gemm_sm120_kernel takes one tile ({', '.join(tile)}) for every bucket (its descriptors' blocks are the tiles)")
    BM, BN, BK, EN = (default[key] for key in tile)
    odt = dtype if out_dtype is None else out_dtype
    a_ann = TensorDesc(Tensor[M, K, dtype], block=(BM, BK)) if a_k else TensorDesc(Tensor[K, M, dtype], block=(BK, BM))
    b_ann = TensorDesc(Tensor[N, K, dtype], block=(BN, BK)) if b_k else TensorDesc(Tensor[K, N, dtype], block=(BK, BN))
    stub = _stub(
        [
            ("a", a_ann, _E),
            ("b", b_ann, _E),
            ("c", Out[TensorDesc(Tensor[M, N, odt], block=(BM, EN))], _E),
            ("m", M, _E),
            ("n", N, _E),
            ("k", K, _E),
            ("BM", const, BM),
            ("BN", const, BN),
            ("BK", const, BK),
            ("STAGES", const, default["STAGES"]),
            ("EPI_N", const, EN),
            ("GROUP_M", const, default["GROUP_M"]),
            ("SPLIT", const, split),
            ("A_KMAJOR", const, int(a_k)),
            ("B_KMAJOR", const, int(b_k)),
            ("NW", const, default["NW"]),
        ],
        name,
    )
    hooks = (
        tilefish.hook("ACC", value="acc", out="c", split=split, block_n="EPI_N", reference=_reference(layout, None, acc=True)),
        tilefish.hook("EPILOGUE", value="out", out="c", split=split, block_n="EPI_N"),
    )
    return tilefish.extern(
        triton=gemm_sm120_kernel,
        name=name,
        num_warps=default["NW"],
        num_stages=default["STAGES"],
        grid=lambda m, n, BM, BN, SPLIT: (tilefish.cdiv(m, BM) * tilefish.cdiv(n, SPLIT * BN),),
        persistent=tilefish.persistent() if persistent else None,
        by_bucket=_by_bucket(M, configs, ("STAGES", "GROUP_M", "NW")),
        reference=_reference(layout, out_dtype),
        epilogue=hooks,
        sass=sass,
    )(stub)


def gemm(
    arch: int,
    *,
    M,
    n: int,
    k: int,
    N=None,
    K=None,
    name: "str | None" = None,
    dtype=bf16,
    out_dtype=None,
    layout: str = "TN",
    split: int = 1,
    sm_count: int = 148,
    config=None,
    persistent: bool = True,
    impl: "str | None" = None,
    sass=None,
):
    """``c = a @ b`` for ``arch`` (100 or 103: :func:`gemm_sm100_kernel`; 120: :func:`gemm_sm120_kernel`), declared with the hooks ``ACC`` and
    ``EPILOGUE`` on ``c`` (``split`` tiles at once: ``split=2`` is the dual accumulator of a gated activation, ``b``'s first and second halves of
    rows side by side).

    * ``M``: the rows, a ``Dim`` (with ``buckets=[...]``: one configuration per bucket, ``by_bucket=``); ``n`` and ``k``: the columns of ``c`` (both
      halves for ``split=2``) and the inner size the configurations are chosen for; ``N`` and ``K``: their ``Dim``s in the stub (by default runtime
      dimensions, multiples of 16: the kernel serves any size, the configuration is chosen for ``n`` and ``k``);
    * ``layout``: ``cublaslt.gemm``'s (``"TN"``: ``a [M, K]``, ``b [N, K]``; ``"NN"``: ``b [K, N]``; ``"TT"``: ``a [K, M]``; ``"NT"``: both);
    * ``dtype``: ``bf16`` or ``f16`` operands; ``out_dtype``: ``c``'s (``dtype`` by default);
    * ``config``: a function ``(m, n, k, split=, sm_count=) -> dict`` (by default :func:`config_sm100` or :func:`config_sm120`) or a dict for every
      bucket; ``sm_count``: the SMs it chooses for (a speed choice: the bytes are the same on every device);
    * ``impl``: ``"gluon"`` (:func:`gemm_sm100_kernel`, the default on sm_100 and sm_103) or ``"triton"`` (:func:`gemm_sm120_kernel`, the only
      one on sm_120; on sm_100 and sm_103 its ``tl.dot`` is a ``tcgen05`` MMA, which is how its logic is tested without an sm_120 device);
    * ``sass``: by default the architecture's MMA (``tcgen05.mma kind::f16``; ``mma.sync m16n8k16``) and no TF32, asserted at every build of the
      GEMM and of every fused op made of it (no ``no_spills``: a fused hook's ``sin`` keeps libdevice's local array for huge arguments).
    """
    if arch not in (100, 103, 120):
        raise ValueError(f"persistent_gemm.gemm: arch is 100, 103 or 120, got {arch!r}")
    if split not in (1, 2):
        raise ValueError(f"split is 1 or 2, got {split!r}")
    if impl not in (None, "gluon", "triton") or (impl == "gluon" and arch == 120):
        raise ValueError(f"impl is 'gluon' (sm_100, sm_103) or 'triton', got {impl!r} for sm_{arch}")
    N = Dim("N", divisible_by=16) if N is None else N
    K = Dim("K", divisible_by=16) if K is None else K
    if callable(config):
        rule = config
    elif config is not None:
        rule = lambda m_, n_, k_, **kw: dict(config)  # noqa: E731
    else:
        rule = config_sm120 if arch == 120 or impl == "triton" else config_sm100
    buckets = list(getattr(M, "buckets", None) or [])
    configs = {b: rule(b, n, k, split=split, sm_count=sm_count) for b in buckets}
    default = rule(max(buckets) if buckets else 2048, n, k, split=split, sm_count=sm_count)
    name = name or f"pgemm_{layout.lower()}_{n}x{k}" + ("_dual" if split == 2 else "")
    if sass is None:  # (not no_spills: a hook's sin keeps libdevice's local array for huge arguments, and Triton 3.5.1's TMA loads keep one on sm_120)
        sass = tilefish.sass(mma="mma.sync m16n8k16" if arch == 120 else "tcgen05.mma kind::f16", no_tf32=True)
    if impl == "triton" or arch == 120:
        return _sm120(M, N, K, dtype, out_dtype, layout, split, configs, default, name, sass, persistent)
    return _sm100(M, N, K, dtype, out_dtype, layout, split, configs, default, name, sass, persistent)
