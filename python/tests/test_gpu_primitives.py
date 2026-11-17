"""The primitives of notes 15 and 16 of the contract on a real GPU, against CUDA C++ compiled by NVRTC (nvcc's defaults) and against PTX.

* ``tilefish.erf(x, math="ieee")`` is CUDA's ``erff`` on every f32: of the NVRTC torch ships (CUDA 12.8 on a cu128 build) and of CUDA 13.0, whose
  PTX is committed (``ptx/erff_cuda13.0.ptx``); CUDA 13.2 changed ``erff`` (``ptx/erff_cuda13.2.ptx``), which differs on 353 987 744 inputs by one ULP;
* ``tilefish.sin_approx`` is ``__sinf`` and ``sin.approx.f32``, ``tilefish.rcp`` is ``__frcp_rn``, ``1.0f / x`` and ``torch.reciprocal``, and ``-x`` is
  CUDA's negation (``neg.f32``: a NaN becomes ``0x7FFFFFFF``, unlike a flip of the sign bit), each on every f32; ``-x`` of a bf16 and of an f16 is
  CUDA's on every one of the 2**16 patterns;
* ``tilefish.fma`` is ``fmaf`` on a large grid of bit patterns, special values and cancellations, in a kernel without contraction too, where
  ``a * b + c`` rounds twice; the reference (an f64 computation rounded to odd) is the binary's bytes;
* ``jit``, ``replay`` and ``reference`` agree byte for byte where the reference is exact (``erf`` in IEEE math, ``rcp``, ``neg``, ``fma``).

The comparisons run on the GPU of the test; nothing in them is specific to sm_103 (an sm_120 GPU runs the same tests).
"""

import pathlib

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.cuda_math]  # the exhaustive comparisons with CUDA's math library: docs/en/numerics.md

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import cuda_ref  # noqa: E402
import primitive_cases as pc  # noqa: E402
from gpu_util import same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import runtime  # noqa: E402

F32, I32, I16 = torch.float32, torch.int32, torch.int16
CUDA = "cuda"
CHUNK = 1 << 26
PTX = pathlib.Path(__file__).resolve().parent / "ptx"
SUBNORMALS = 2 * ((1 << 23) - 1)
NANS = 2 * ((1 << 23) - 1)

SOURCE = r"""
extern "C" __global__ void k_sinf_approx(const float* x, float* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = __sinf(x[i]); }
extern "C" __global__ void k_frcp_rn(const float* x, float* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = __frcp_rn(x[i]); }
extern "C" __global__ void k_recip(const float* x, float* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = 1.0f / x[i]; }
extern "C" __global__ void k_erff(const float* x, float* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = erff(x[i]); }
extern "C" __global__ void k_neg(const float* x, float* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = -x[i]; }
extern "C" __global__ void k_fmaf(const float* a, const float* b, const float* c, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = fmaf(a[i], b[i], c[i]);
}
"""

HALF_SOURCE = r"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
extern "C" __global__ void k_neg_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = -x[i]; }
extern "C" __global__ void k_neg_f16(const __half* x, __half* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = -x[i]; }
"""

SIN_APPROX_PTX = """
.version 7.0
.target sm_80
.address_size 64
.visible .entry k_sin_approx(.param .u64 px, .param .u64 py, .param .u32 pn) {
  .reg .pred %p; .reg .b32 %r<6>; .reg .f32 %f<3>; .reg .b64 %rd<8>;
  ld.param.u64 %rd1, [px]; ld.param.u64 %rd2, [py]; ld.param.u32 %r1, [pn];
  mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x; mad.lo.s32 %r5, %r2, %r3, %r4;
  setp.ge.s32 %p, %r5, %r1; @%p bra DONE;
  cvta.to.global.u64 %rd3, %rd1; cvta.to.global.u64 %rd4, %rd2; mul.wide.s32 %rd5, %r5, 4; add.s64 %rd6, %rd3, %rd5; add.s64 %rd7, %rd4, %rd5;
  ld.global.f32 %f1, [%rd6];
  sin.approx.f32 %f2, %f1;
  st.global.f32 [%rd7], %f2;
DONE:
  ret;
}
"""


def reference(build):
    try:
        return build()
    except cuda_ref.ReferenceUnavailable as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def cuda_c():
    return reference(lambda: cuda_ref.cuda_module(SOURCE))


def chunks():
    for start in range(0, 1 << 32, CHUNK):
        yield torch.arange(start, start + CHUNK, dtype=torch.int64, device=CUDA).to(I32).view(F32)


def every_f32(op, *references):
    """Over every f32 bit pattern: the number of inputs on which the kernel's bits differ from each reference's (a function of the input)."""
    counts = [0] * len(references)
    for x in chunks():
        got = tilefish.run(op, x).view(I32)
        for i, ref in enumerate(references):
            counts[i] += int((got != ref(x).view(I32)).sum())
    return counts


def test_erf_in_ieee_math_is_cudas_erff_on_every_f32(cuda_c):
    committed = {v: reference(lambda v=v: cuda_ref.Module((PTX / f"erff_cuda{v}.ptx").read_text())) for v in ("13.0", "13.2")}
    nvrtc = cuda_ref.nvrtc()
    live, cuda130, cuda132, fast = every_f32(
        pc.erf_ieee,
        lambda x: cuda_ref.map_f32(cuda_c, "k_erff", x),
        lambda x: cuda_ref.map_f32(committed["13.0"], "k_erff", x),
        lambda x: cuda_ref.map_f32(committed["13.2"], "k_erff", x),
        lambda x: tilefish.run(pc.erf_fast, x),
    )
    print(f"\n[erf] IEEE math vs erff over 2**32 f32: NVRTC {nvrtc.version} {live}, CUDA 13.0 {cuda130}, CUDA 13.2 {cuda132}; "
          f"libdevice's (the default math) differs on {fast}")  # fmt: skip
    assert live == 0 and cuda130 == 0
    assert cuda132 == 353_987_744  # CUDA 13.2 changed erff: the cuda_math guard's table lists it for 12.8 to 13.1 only
    assert fast > 0  # libdevice's erff is another implementation: the default math is not CUDA's


def test_erf_in_ieee_math_is_torchs_where_torch_is_built_with_cuda_12_8_to_13_1():
    cuda = tuple(int(p) for p in (torch.version.cuda or "0.0").split(".")[:2])
    if not (12, 8) <= cuda <= (13, 1):
        pytest.skip(f"torch here is built with CUDA {torch.version.cuda}, whose erff is not the one this erf reproduces (the cuda_math guard flags it)")
    differs = sum(int((tilefish.run(pc.erf_ieee, x).view(I32) != torch.erf(x).view(I32)).sum()) for x in chunks())
    assert differs == 0


def test_sin_approx_is_sinf_approx_on_every_f32(cuda_c):
    ptx = reference(lambda: cuda_ref.Module(SIN_APPROX_PTX))
    intrinsic, instruction = every_f32(
        pc.sin_approx, lambda x: cuda_ref.map_f32(cuda_c, "k_sinf_approx", x), lambda x: cuda_ref.map_f32(ptx, "k_sin_approx", x)
    )
    assert intrinsic == 0 and instruction == 0


def test_rcp_is_the_correctly_rounded_reciprocal_on_every_f32(cuda_c):
    intrinsic, division, torchs = every_f32(
        pc.rcp,
        lambda x: cuda_ref.map_f32(cuda_c, "k_frcp_rn", x),
        lambda x: cuda_ref.map_f32(cuda_c, "k_recip", x),
        torch.reciprocal,
    )
    assert intrinsic == 0 and division == 0 and torchs == 0


def test_negation_is_cudas_on_every_f32_and_not_a_sign_flip_of_a_nan(cuda_c):
    flip = lambda x: (x.view(I32) ^ -(2**31)).view(F32)  # noqa: E731
    cudas, flipped = every_f32(pc.neg_f32, lambda x: cuda_ref.map_f32(cuda_c, "k_neg", x), flip)
    assert cudas == 0
    assert flipped == NANS - 1  # every NaN but 0xFFFFFFFF becomes 0x7FFFFFFF, which the flip of 0xFFFFFFFF is
    zero = torch.zeros(4, device=CUDA)
    assert same_bytes(tilefish.run(pc.neg_f32, zero), -zero) and int(tilefish.run(pc.neg_f32, zero).view(I32)[0]) == -(2**31)


@pytest.mark.parametrize("dtype,kernel,name", [(torch.bfloat16, "neg_bf16", "k_neg_bf16"), (torch.float16, "neg_f16", "k_neg_f16")])
def test_negation_of_a_half_type_is_cudas_on_every_pattern(dtype, kernel, name):
    module = reference(lambda: cuda_ref.cuda_module(HALF_SOURCE))
    x = torch.arange(-(1 << 15), 1 << 15, dtype=I32, device=CUDA).to(I16).view(dtype)
    got = tilefish.run(getattr(pc, kernel), x)
    assert same_bytes(got, cuda_ref.map_f32(module, name, x))
    assert same_bytes(got, tilefish.run(getattr(pc, kernel), x, mode="reference"))


def fma_operands(count=1 << 24, seed=0):
    """Random bit patterns (every class of value), cancellations ``c ~ -a * b``, products and sums at the edge of the subnormal range, and every
    combination of special values."""
    gen = torch.Generator(device=CUDA)
    gen.manual_seed(seed)

    def words(n):
        return torch.randint(-(2**31), 2**31, (n,), generator=gen, device=CUDA, dtype=torch.int64).to(I32).view(F32)

    a, b, c = words(count), words(count), words(count)
    n = count // 4
    near = torch.randn(n, generator=gen, device=CUDA)
    a[:n] = near
    b[:n] = torch.randn(n, generator=gen, device=CUDA)
    c[:n] = -(a[:n].double() * b[:n].double()).float() * (1 + torch.randn(n, generator=gen, device=CUDA) * 2.0**-20)  # cancellation
    tiny = torch.pow(2.0, torch.randint(-80, -60, (n,), generator=gen, device=CUDA).float())
    a[n : 2 * n] = torch.randn(n, generator=gen, device=CUDA) * tiny
    b[n : 2 * n] = torch.randn(n, generator=gen, device=CUDA) * tiny
    c[n : 2 * n] = torch.randn(n, generator=gen, device=CUDA) * 2.0**-140  # around the subnormals
    special = torch.tensor([0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan"), 2.0**-149, -(2.0**-149), 2.0**-126, 3.4028234663852886e38],
                           device=CUDA)  # fmt: skip
    k = special.numel()
    grid = torch.cartesian_prod(torch.arange(k, device=CUDA), torch.arange(k, device=CUDA), torch.arange(k, device=CUDA))
    a[-grid.shape[0] :], b[-grid.shape[0] :], c[-grid.shape[0] :] = special[grid[:, 0]], special[grid[:, 1]], special[grid[:, 2]]
    nan_payload = torch.tensor([0x7F800001, 0xFFC00123 - (1 << 32)], device=CUDA, dtype=torch.int64).to(I32).view(F32)
    a[2 * n : 2 * n + 2] = nan_payload
    return a, b, c


def test_fma_is_fmaf_on_a_large_grid_and_the_reference_is_its_bytes(cuda_c):
    for seed in range(3):
        a, b, c = fma_operands(seed=seed)
        got = tilefish.run(pc.fma3, a, b, c)
        assert same_bytes(got, cuda_ref.map_f32(cuda_c, "k_fmaf", a, b, c)), seed
        assert same_bytes(got, tilefish.run(pc.fma3, a, b, c, mode="reference")), seed
        twice = tilefish.run(pc.mul_add, a, b, c)
        assert int((twice.view(I32) != got.view(I32)).sum()) > a.numel() // 10  # a * b + c without contraction rounds twice: not an fma


def test_an_fma_with_a_scalar_rounds_once_in_a_kernel_without_contraction(cuda_c):
    a, _, c = fma_operands(1 << 20, seed=5)
    b = torch.full_like(a, 0.1)
    assert same_bytes(tilefish.run(pc.fma_scalar, a, c), cuda_ref.map_f32(cuda_c, "k_fmaf", a, b, c))


def f32_patterns(count=1 << 22, seed=0):
    gen = torch.Generator(device=CUDA)
    gen.manual_seed(seed)
    words = torch.randint(-(2**31), 2**31, (count,), generator=gen, device=CUDA, dtype=torch.int64).to(I32)
    words[:6] = torch.tensor([0, -(2**31), 1, -(2**31) + 1, 0x007FFFFF, 0x7FC00000], device=CUDA, dtype=torch.int64).to(I32)
    return words.view(F32)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("primitives")
    ops = [pc.erf_fast, pc.erf_ieee, pc.sin_approx, pc.rcp, pc.neg_f32, pc.neg_bf16, pc.fma3]
    return directory, runtime.export(ops, directory)


def test_jit_replay_and_reference_agree(exported):
    directory, m = exported
    x = f32_patterns()
    exact = {pc.erf_ieee, pc.rcp, pc.neg_f32}
    for op in (pc.erf_fast, pc.erf_ieee, pc.sin_approx, pc.rcp, pc.neg_f32):
        jit = tilefish.run(op, x)
        assert same_bytes(jit, tilefish.run(op, x, mode="replay", manifest=directory, pin=m.digest)), op.name
        if op in exact and (op is not pc.erf_ieee or (12, 8) <= tuple(int(p) for p in (torch.version.cuda or "0.0").split(".")[:2]) <= (13, 1)):
            assert same_bytes(jit, tilefish.run(op, x, mode="reference")), op.name
    a, b, c = fma_operands(1 << 20, seed=7)
    jit = tilefish.run(pc.fma3, a, b, c)
    assert same_bytes(jit, tilefish.run(pc.fma3, a, b, c, mode="replay", manifest=directory, pin=m.digest))


def test_the_approximate_ones_are_within_their_bounds_of_the_reference():
    """``sin_approx`` (the hardware's sine) and the default math's ``erf`` (libdevice's) against torch, which the reference uses: measured bounds."""
    x = torch.rand(1 << 22, device=CUDA) * 2 * torch.pi - torch.pi  # sin.approx's documented range, [-pi, pi]
    err = (tilefish.run(pc.sin_approx, x).double() - tilefish.run(pc.sin_approx, x, mode="reference").double()).abs().max().item()
    print(f"\n[sin_approx] max abs error vs torch.sin on [-pi, pi]: {err:.3g}")
    assert err < 2.0**-20.5
    y = torch.randn(1 << 22, device=CUDA) * 3
    err = (tilefish.run(pc.erf_fast, y).double() - tilefish.run(pc.erf_fast, y, mode="reference").double()).abs().max().item()
    print(f"[erf] default math (libdevice) max abs error vs torch.erf: {err:.3g}")
    assert err < 2.0**-22
