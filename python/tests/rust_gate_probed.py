"""Kernels with probes (``tilefish.probe``, ``docs/en/instrument.md``) for the instrumented cases of the Rust gate (``rust_gate.py``: an instrumented export the
Rust host launches with a probe buffer, and an instrumented plan it runs with probes). Not collected: the name has no ``test_`` prefix.

Written from scratch for these tests, and public. Importing this module imports Triton and its Gluon, so only the gate does.

* ``ws_scale_probed``: ``y = 2 * x + 1`` (f32, rounded to f16), a TMA producer partition (1 warp) feeding a compute partition (the default one, 4 warps)
  through a ring of three shared-memory tiles, with a probe before and after each wait of each partition: the shape of ``ws_kernels.ws_scale``. TMA and
  mbarriers: sm_90 and up.
* ``scale_add_probed``: ``2 * x + y`` in f32, a ``@tilefish.kernel`` with a probe between its two ops and one before its store.
* ``cast_up_probed`` and the graph ``probed_chain``: ``ws_scale_probed(cast_up_probed(x))``, a plan of two instrumented steps over one bucket of rows.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma

import tilefish
from tilefish import Dim, Out, Tensor, TensorDesc, const, f16, f32

N = Dim("N", divisible_by=16)
R, C = Dim("R"), Dim("C")
Rb = Dim("Rb", buckets=[192])


@gluon.jit
def _load(x, bufs, ready, empty, n_cols, BM: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr):
    m = gl.program_id(0) * BM
    for i in range(gl.cdiv(n_cols, BN)):
        s = i % STAGES
        tilefish.probe("load.wait")
        mbarrier.wait(empty.index(s), ((i // STAGES) & 1) ^ 1)
        tilefish.probe("load.issue")
        mbarrier.expect(ready.index(s), x.block_type.nbytes)
        tma.async_copy_global_to_shared(x, [m, i * BN], ready.index(s), bufs.index(s))


@gluon.jit
def _compute(y, bufs, out, ready, empty, n_cols, BM: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr, NW: gl.constexpr):
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
def ws_scale_probed_kernel(x, y, n_cols, BM: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr, NW: gl.constexpr):
    bufs = gl.allocate_shared_memory(x.dtype, [STAGES, BM, BN], x.layout)
    out = gl.allocate_shared_memory(y.dtype, [BM, BN], y.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    for s in gl.static_range(STAGES):
        mbarrier.init(ready.index(s), count=1)
        mbarrier.init(empty.index(s), count=1)
    gl.warp_specialize(
        (y, bufs, out, ready, empty, n_cols, BM, BN, STAGES, NW),
        _compute,
        (x, bufs, ready, empty, n_cols, BM, BN, STAGES),
        [_load],
        [1],
        [24],
    )


@tilefish.extern(gluon=ws_scale_probed_kernel, num_warps=4, grid=lambda R: (tilefish.cdiv(R, 64),))
def ws_scale_probed(
    x: TensorDesc(Tensor[R, C, f16], block=(64, 64)),
    y: Out[TensorDesc(Tensor[R, C, f16], block=(64, 64))],
    n_cols: C,
    BM: const = 64,
    BN: const = 64,
    STAGES: const = 3,
    NW: const = 4,
): ...


@tilefish.kernel
def scale_add_probed(x: Tensor[N, f32], y: Tensor[N, f32]) -> Tensor[N, f32]:
    a = x * 2.0
    tilefish.probe("scaled")
    b = a + y
    tilefish.probe("added")
    return b


@tilefish.kernel
def cast_up_probed(x: Tensor[Rb, 512, f16]) -> Tensor[Rb, 512, f16]:
    tilefish.probe("start")
    return (x.to(f32) + 0.5).to(f16)


@tilefish.graph
def probed_chain(x: Tensor[Rb, 512, f16]) -> Tensor[Rb, 512, f16]:
    return ws_scale_probed(cast_up_probed(x))


def ops():
    """``{name: op}``: the operators the instrumented export of the gate has."""
    return {"ws_scale_probed": ws_scale_probed, "scale_add_probed": scale_add_probed}
