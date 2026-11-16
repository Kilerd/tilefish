"""Kernels with probes (``tilefish.probe``) for the gpu tests of the instrumented variant (``docs/instrument.md``), not collected: the name has no
``test_`` prefix. Written from scratch for these tests, and public. Importing this module imports Triton and its Gluon, so only the gpu tests do.
No ``from __future__ import annotations``: tilefish evaluates annotations.

* ``ws_scale_p``: ``ws_kernels.ws_scale`` (``y = 2 * x + 1`` in f16, a 1-warp TMA producer feeding a 4-warp compute partition through a ring of three
  tiles), its kernel ``ws_scale_kernel`` with probes around each wait: the producer's wait for an empty slot (``load.wait`` to ``load.issue``) and the
  consumer's wait for a full one (``compute.wait`` to ``compute.work``). Without the probes it is ``ws_kernels``'s kernel, line for line.
* ``ws_matmul_p``: ``ws_kernels.ws_matmul`` (``c = a @ b.T``, f16, tcgen05: a 1-warp TMA producer, a 1-warp MMA issuer, a 4-warp epilogue), with probes
  around the producer's wait for a free slot, the issuer's wait for a loaded one, its commit, and the epilogue's wait for the accumulator. sm_100 and
  sm_103 only.
* ``persistent_scale_p``: ``launch_kernels.persistent_scale`` (``y = x * 2 + 0.25`` with a persistent grid: one program per SM walks the tiles), with a
  probe at the start of each tile and after its store.
* ``scale_add_p``: a ``@tilefish.kernel`` (``x * 2 + y`` in f32, the elementwise template) with a probe after the multiply and after the add.
* ``probe_chain``: a graph of a ``@tilefish.kernel`` with a probe (``bump_p``) and ``ws_scale_p``: a plan of two instrumented steps.
* ``ws_scale_g``, ``persistent_g``, ``scale_g``, ``mm_g``: a graph of one step of each, captured with and without probes to time what they cost.
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
from tilefish import Dim, Out, Tensor, TensorDesc, const, f16, f32

R, C = Dim("R"), Dim("C")
M, N, K = Dim("M"), Dim("N"), Dim("K")
L = Dim("L", divisible_by=16)
Rb = Dim("Rb", buckets=[256])
WORKER_REGS = gl.constexpr(24)


# -- ws_scale_p --------------------------------------------------------------------------------------------------------------------------------


@gluon.jit
def _scale_load(x, bufs, ready, empty, n_cols, BM: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr):
    m = gl.program_id(0) * BM
    for i in range(gl.cdiv(n_cols, BN)):
        s = i % STAGES
        tilefish.probe("load.wait")
        mbarrier.wait(empty.index(s), ((i // STAGES) & 1) ^ 1)
        tilefish.probe("load.issue")
        mbarrier.expect(ready.index(s), x.block_type.nbytes)
        tma.async_copy_global_to_shared(x, [m, i * BN], ready.index(s), bufs.index(s))


@gluon.jit
def _scale_compute(y, bufs, out, ready, empty, n_cols, BM: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr, NW: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    m = gl.program_id(0) * BM
    for i in range(gl.cdiv(n_cols, BN)):
        s = i % STAGES
        tilefish.probe("compute.wait")
        mbarrier.wait(ready.index(s), (i // STAGES) & 1)
        tilefish.probe("compute.work")
        tile = bufs.index(s).load(layout)
        mbarrier.arrive(empty.index(s), count=1)
        result = (tile.to(gl.float32) * 2.0 + 1.0).to(gl.float16)
        tma.store_wait(0)
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


@tilefish.extern(gluon=ws_scale_kernel, num_warps=4, grid=lambda R: (tilefish.cdiv(R, 64),), name="ws_scale_p")
def ws_scale_p(
    x: TensorDesc(Tensor[R, C, f16], block=(64, 64)),
    y: Out[TensorDesc(Tensor[R, C, f16], block=(64, 64))],
    n_cols: C,
    BM: const = 64,
    BN: const = 64,
    STAGES: const = 3,
    NW: const = 4,
): ...


# -- ws_matmul_p (tcgen05) ---------------------------------------------------------------------------------------------------------------------


@gluon.jit
def _mm_load(a, b, a_bufs, b_bufs, acc, ready, empty, acc_bar, k_len, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, STAGES: gl.constexpr):
    m = gl.program_id(0) * BM
    n = gl.program_id(1) * BN
    for i in range(gl.cdiv(k_len, BK)):
        s = i % STAGES
        tilefish.probe("load.wait")
        mbarrier.wait(empty.index(s), ((i // STAGES) & 1) ^ 1)
        tilefish.probe("load.issue")
        mbarrier.expect(ready.index(s), a.block_type.nbytes + b.block_type.nbytes)
        tma.async_copy_global_to_shared(a, [m, i * BK], ready.index(s), a_bufs.index(s))
        tma.async_copy_global_to_shared(b, [n, i * BK], ready.index(s), b_bufs.index(s))


@gluon.jit
def _mm_issue(a, b, a_bufs, b_bufs, acc, ready, empty, acc_bar, k_len, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, STAGES: gl.constexpr):
    use_acc = False
    for i in range(gl.cdiv(k_len, BK)):
        s = i % STAGES
        tilefish.probe("mma.wait")
        mbarrier.wait(ready.index(s), (i // STAGES) & 1)
        tilefish.probe("mma.issue")
        tcgen05_mma(a_bufs.index(s), b_bufs.index(s).permute((1, 0)), acc, use_acc=use_acc, mbarriers=[empty.index(s)])
        use_acc = True
    tcgen05_commit(acc_bar)
    tilefish.probe("mma.committed")


@gluon.jit
def _mm_epilogue(c, acc, acc_bar, c_tile, BM: gl.constexpr, BN: gl.constexpr, NW: gl.constexpr):
    tilefish.probe("epi.wait")
    mbarrier.wait(acc_bar, 0)
    tilefish.probe("epi.store")
    reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(BM, BN, [BM, BN], NW)
    values = acc.load(reg_layout)
    c_tile.store(values.to(gl.float16))
    fence_async_shared()
    tma.async_copy_shared_to_global(c, [gl.program_id(0) * BM, gl.program_id(1) * BN], c_tile)
    tma.store_wait(0)
    tilefish.probe("epi.done")


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


@tilefish.extern(gluon=ws_matmul_kernel, num_warps=4, grid=lambda M, N: (tilefish.cdiv(M, 128), tilefish.cdiv(N, 128)), name="ws_matmul_p")
def ws_matmul_p(
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


# -- persistent_scale_p --------------------------------------------------------------------------------------------------------------------------


@triton.jit
def _persistent_scale(x, y, n, BLOCK: tl.constexpr):
    tiles = tl.cdiv(n, BLOCK)
    for t in range(tl.program_id(0), tiles, tl.num_programs(0)):
        tilefish.probe("tile")
        offs = t * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(y + offs, tl.load(x + offs, mask=m) * 2.0 + 0.25, mask=m)
        tilefish.probe("stored")


@tilefish.extern(
    triton=_persistent_scale, num_warps=4, grid=lambda N, BLOCK: (tilefish.cdiv(N, BLOCK),), persistent=tilefish.persistent(),
    name="persistent_scale_p",
)  # fmt: skip
def persistent_scale_p(x: Tensor[N, f32], y: Out[Tensor[N, f32]], n: N, BLOCK: const = 1024): ...


# -- scale_add_p and probe_chain -------------------------------------------------------------------------------------------------------------


@tilefish.kernel
def scale_add_p(x: Tensor[L, f32], y: Tensor[L, f32]) -> Tensor[L, f32]:
    a = x * 2.0
    tilefish.probe("scaled")
    b = a + y
    tilefish.probe("added")
    return b


@tilefish.kernel
def bump_p(x: Tensor[Rb, 1024, f16]) -> Tensor[Rb, 1024, f16]:
    tilefish.probe("start")
    return (x.to(f32) + 0.5).to(f16)


@tilefish.graph
def probe_chain(x: Tensor[Rb, 1024, f16]) -> Tensor[Rb, 1024, f16]:
    return ws_scale_p(bump_p(x))


# graphs of one step each, captured with and without probes to time what the probes cost (a CUDA graph replay has no host in it)
Wb = Dim("Wb", buckets=[1024])
Pb = Dim("Pb", buckets=[1 << 23])
Lb = Dim("Lb", buckets=[1 << 22])
Mb = Dim("Mb", buckets=[512])


@tilefish.graph
def ws_scale_g(x: Tensor[Wb, 4096, f16]) -> Tensor[Wb, 4096, f16]:
    return ws_scale_p(x)


@tilefish.graph
def persistent_g(x: Tensor[Pb, f32]) -> Tensor[Pb, f32]:
    return persistent_scale_p(x)


@tilefish.graph
def scale_g(x: Tensor[Lb, f32], y: Tensor[Lb, f32]) -> Tensor[Lb, f32]:
    return scale_add_p(x, y)


@tilefish.graph
def mm_g(a: Tensor[Mb, 8192, f16], b: Tensor[512, 8192, f16]) -> Tensor[Mb, 512, f16]:
    return ws_matmul_p(a, b)


def tcgen05(cap) -> bool:
    """Whether the device of capability ``cap`` (a ``(major, minor)`` pair) has tcgen05: sm_100 and sm_103."""
    return tuple(cap) in ((10, 0), (10, 3))


def ops(cap) -> "list":
    """The externs and kernels of the tests on the device of capability ``cap``."""
    return [ws_scale_p, persistent_scale_p, scale_add_p] + ([ws_matmul_p] if tcgen05(cap) else [])
