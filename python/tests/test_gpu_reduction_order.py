"""Ordered sums (note 17 of the contract) on a real GPU, against CUDA C++ kernels that reduce a row the way the order says.

* ``order="sequential"`` is one CUDA thread's loop over the row from ``+0.0``;
* ``tree(256)`` is 256 threads that sum the columns ``t, t + 256, ...`` and then a shared-memory halving tree;
* ``tree(128, offsets=(16, 8, 4, 2, 1, 64, 32), vector=8)`` with ``fma=True`` is 128 threads that each take 8 consecutive columns at a time
  with ``acc += x * x`` (contracted by nvcc into an fma), a ``__shfl_down_sync`` tree in each warp, and ``(w0 + w2) + (w1 + w3)`` over the 4 warps;
* ``tree(64)`` with ``fma=True`` is a dot product with a weight row in ``fmaf`` chains, then the halving tree;

byte for byte over every bucket of a bucketed row length (from 1 to 4096 columns: rows shorter than the lanes, lengths that are not a multiple of
the lanes' width), on values of every magnitude (where the order shows), on rows of signed zeros, NaNs, infinities, subnormals and products that
round to ``-0.0``. An rmsnorm written as a 128-thread CUDA kernel is that kernel's bytes. An ordered sum gives the same bytes at every
``num_warps`` (Triton's own ``tl.sum`` does not), and ``jit``, ``replay`` and ``reference`` agree byte for byte.

Nothing here is specific to sm_103: the same tests run on an sm_120 GPU.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import cuda_ref  # noqa: E402
import primitive_cases as pc  # noqa: E402
from gpu_util import differing, randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import runtime  # noqa: E402

F32, I32, BF16 = torch.float32, torch.int32, torch.bfloat16
CUDA = "cuda"
ROWS = 1024
LENGTHS = pc.Db.buckets

SOURCE = r"""
extern "C" __global__ void k_seq(const float* x, float* out, int D) {
  const float* r = x + (long)blockIdx.x * D;
  float acc = 0.f;
  for (int i = 0; i < D; ++i) acc += r[i];
  out[blockIdx.x] = acc;
}
extern "C" __global__ void k_halving256(const float* x, float* out, int D) {
  __shared__ float s[256];
  const float* r = x + (long)blockIdx.x * D;
  int t = threadIdx.x;
  float acc = 0.f;
  for (int i = t; i < D; i += 256) acc += r[i];
  s[t] = acc;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) { if (t < o) s[t] += s[t + o]; __syncthreads(); }
  if (t == 0) out[blockIdx.x] = s[0];
}
extern "C" __global__ void k_sumsq_shfl128v8(const float* x, float* out, int D) {
  __shared__ float w[4];
  const float* r = x + (long)blockIdx.x * D;
  int t = threadIdx.x;
  float acc = 0.f;
  for (int c = 0; c * 1024 < D; ++c)
    for (int v = 0; v < 8; ++v) { int k = c * 1024 + t * 8 + v; if (k < D) acc += r[k] * r[k]; }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if ((t & 31) == 0) w[t >> 5] = acc;
  __syncthreads();
  if (t == 0) out[blockIdx.x] = (w[0] + w[2]) + (w[1] + w[3]);
}
extern "C" __global__ void k_dot_halving64(const float* x, const float* wt, float* out, int D) {
  __shared__ float s[64];
  const float* r = x + (long)blockIdx.x * D;
  int t = threadIdx.x;
  float acc = 0.f;
  for (int i = t; i < D; i += 64) acc = fmaf(r[i], wt[i], acc);
  s[t] = acc;
  __syncthreads();
  for (int o = 32; o > 0; o >>= 1) { if (t < o) s[t] += s[t + o]; __syncthreads(); }
  if (t == 0) out[blockIdx.x] = s[0];
}
"""

RMSNORM_SOURCE = r"""
#include <cuda_bf16.h>
extern "C" __global__ void k_rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* y) {
  __shared__ float ws[4];
  const __nv_bfloat16* r = x + (long)blockIdx.x * 1024;
  __nv_bfloat16* o = y + (long)blockIdx.x * 1024;
  int t = threadIdx.x;
  float v[8];
  float acc = 0.f;
  for (int j = 0; j < 8; ++j) { v[j] = __bfloat162float(r[t * 8 + j]); acc += v[j] * v[j]; }
  for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
  if ((t & 31) == 0) ws[t >> 5] = acc;
  __syncthreads();
  float ss = (ws[0] + ws[2]) + (ws[1] + ws[3]);
  float rr = rsqrtf(ss / 1024.0f + 1e-6f);
  for (int j = 0; j < 8; ++j) o[t * 8 + j] = __float2bfloat16_rn((v[j] * rr) * __bfloat162float(w[t * 8 + j]));
}
"""


@pytest.fixture(scope="module")
def cuda_c():
    try:
        return cuda_ref.cuda_module(SOURCE)
    except cuda_ref.ReferenceUnavailable as e:
        pytest.skip(str(e))


def rows(d, seed=0, rows=ROWS):
    """Values of every magnitude (2**-20 to 2**20 times a normal), so that another order of the adds rounds differently; the first rows are
    the special cases: all ``-0.0``, a NaN with a payload, both infinities, subnormals, ``+-1e30`` cancelling around small values, products of
    two tiny values that round to ``-0.0`` (an fma chain then holds ``-0.0``)."""
    gen = torch.Generator(device=CUDA)
    gen.manual_seed(seed * 7919 + d)
    x = torch.randn((rows, d), generator=gen, device=CUDA) * torch.pow(2.0, torch.randint(-20, 21, (rows, d), generator=gen, device=CUDA).float())
    x[0] = -0.0
    x[1, d // 2] = torch.tensor(0x7F800123, dtype=I32).view(F32).item()
    x[2, 0] = float("inf")
    x[2, -1] = float("-inf")
    x[3] = torch.randn(d, generator=gen, device=CUDA) * 2.0**-140
    x[4, ::2] = 1e30
    x[4, 1::2] = -1e30
    x[5] = -1e-30 * torch.rand(d, generator=gen, device=CUDA)
    return x


def launch(module, name, block, x, *extra):
    out = torch.empty(x.shape[0], device=CUDA, dtype=F32)
    module.launch(name, (x.shape[0],), (block,), x, *extra, out, x.shape[1])
    return out


def tiny_weights(d, seed=0):
    gen = torch.Generator(device=CUDA)
    gen.manual_seed(seed + d)
    return torch.randn(d, generator=gen, device=CUDA) * 1e-30  # with row 5, products of 1e-60: an fma of +0.0 rounds them to -0.0


@pytest.mark.parametrize("d", LENGTHS)
def test_each_order_is_the_cuda_kernel_of_that_order(cuda_c, d):
    x = rows(d)
    w = tiny_weights(d)
    cases = [
        ("sequential", pc.sum_sequential, (x,), launch(cuda_c, "k_seq", 1, x)),
        ("halving256", pc.sum_halving256, (x,), launch(cuda_c, "k_halving256", 256, x)),
        ("sumsq_shfl128v8", pc.sumsq_shfl128v8, (x,), launch(cuda_c, "k_sumsq_shfl128v8", 128, x)),
        ("dot_halving64", pc.dot_halving64, (x, w), launch(cuda_c, "k_dot_halving64", 64, x, w)),
    ]
    for name, op, args, want in cases:
        got = tilefish.run(op, *args)
        assert same_bytes(got, want), f"{name} D={d}: {differing(got, want)} of {x.shape[0]} rows differ"
    if d >= 96:  # the test can tell orders apart: Triton's own tree is not the sequential chain
        assert differing(tilefish.run(pc.sum_default, x), cases[0][3]) > ROWS // 4


def test_a_dot_chain_keeps_a_negative_zero_as_the_cuda_loop_does(cuda_c):
    d = 100  # not a multiple of the 64 lanes: the lanes past the row's end are skipped, not given +0.0 products
    x = -torch.full((8, d), 1e-30, device=CUDA)
    w = torch.full((d,), 1e-30, device=CUDA)
    got = tilefish.run(pc.dot_halving64, x, w)
    assert same_bytes(got, launch(cuda_c, "k_dot_halving64", 64, x, w))


def test_an_rmsnorm_of_128_threads_is_the_cuda_kernels_bytes():
    try:
        module = cuda_ref.cuda_module(RMSNORM_SOURCE)
    except cuda_ref.ReferenceUnavailable as e:
        pytest.skip(str(e))
    for seed, scale in enumerate((1.0, 30.0, 1e-3)):
        x = randn((777, 1024), BF16, seed=seed, scale=scale)
        w = randn((1024,), BF16, seed=seed + 100)
        want = torch.empty_like(x)
        module.launch("k_rmsnorm", (x.shape[0],), (128,), x, w, want)
        got = tilefish.run(pc.rmsnorm_ordered, x, w)
        assert same_bytes(got, want), f"scale {scale}: {differing(got, want)} elements differ"


@pytest.mark.parametrize("d", [96, 1000, 4096])
def test_an_ordered_sum_has_the_same_bytes_at_every_num_warps_and_tritons_does_not(d):
    x = rows(d, seed=3)
    defaults = {}
    for op in (pc.sum_halving256, pc.sumsq_shfl128v8, pc.sum_sequential, pc.sum_default):
        outs = {w: tilefish.lower(op, num_warps=w, module=f"tf_warps_{op.name}_{w}").run(x) for w in (1, 2, 4, 8, 16)}
        if op is pc.sum_default:
            defaults = outs
            continue
        for w, out in outs.items():
            assert same_bytes(out, outs[1]), f"{op.name} D={d}: num_warps={w} differs from 1 warp in {differing(out, outs[1])} rows"
    differ = {w: differing(out, defaults[1]) for w, out in defaults.items()}
    print(f"\n[num_warps] D={d}: Triton's tl.sum differs from its 1-warp bytes in {differ} of {ROWS} rows; every ordered sum in none")
    assert any(differ.values())


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("orders")
    ops = [pc.sum_sequential, pc.sum_halving256, pc.sumsq_shfl128v8, pc.dot_halving64, pc.rmsnorm_ordered]
    return directory, runtime.export(ops, directory)


@pytest.mark.parametrize("d", [1, 31, 257, 1024, 4096])
def test_jit_replay_and_reference_agree(exported, d):
    directory, m = exported
    x = rows(d, seed=5, rows=257)
    w = tiny_weights(d) * 1e30
    for op, args in ((pc.sum_sequential, (x,)), (pc.sum_halving256, (x,)), (pc.sumsq_shfl128v8, (x,)), (pc.dot_halving64, (x, w))):
        jit = tilefish.run(op, *args)
        assert same_bytes(jit, tilefish.run(op, *args, mode="replay", manifest=directory, pin=m.digest)), (op.name, d)
        assert same_bytes(jit, tilefish.run(op, *args, mode="reference")), (op.name, d)


def test_the_rmsnorm_in_replay_and_reference(exported):
    directory, m = exported
    x, w = randn((300, 1024), BF16, seed=9, scale=4.0), randn((1024,), BF16, seed=10)
    jit = tilefish.run(pc.rmsnorm_ordered, x, w)
    assert same_bytes(jit, tilefish.run(pc.rmsnorm_ordered, x, w, mode="replay", manifest=directory, pin=m.digest))
    assert same_bytes(jit, tilefish.run(pc.rmsnorm_ordered, x, w, mode="reference"))
