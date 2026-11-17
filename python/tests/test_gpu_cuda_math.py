"""The CUDA math guard on a real GPU (``tilefish._cudamath``, ``docs/en/numerics.md``): its table, measured, and what an export and a replay do.

* **The table.** For every CUDA release whose NVRTC is here (torch's, and each ``libnvrtc.so`` that ``$TILEFISH_NVRTC_PATHS`` names, separated by
  ``:``), every function of :data:`tilefish._cudamath.VERIFIED`, compiled by that NVRTC with nvcc's defaults, is compared with the ops built on it over
  every f32: ``exp`` and ``silu`` in IEEE math with ``expf`` (and ``x / (1 + expf(-x))``), ``rsqrt`` with ``rsqrtf``, ``erf`` with ``erff``, ``sin``,
  ``cos`` and ``tanh`` with ``sinf``, ``cosf`` and ``tanhf`` (in the default math, libdevice built with flush to zero, on every input but a
  subnormal argument of ``sin`` and ``tanh``, which is a zero of its sign). A release the table lists must give the op's bits; one it does not
  list must not (else the table is out of date: add the release). This is the procedure to run on a CUDA or torch change (``pytest -m cuda_math``).
* **An export** records ``cuda_math`` on the entries of the kernels that have these ops (the functions, the releases, the sha256 of the
  libdevice Triton linked) and on no other; **a replay** warns where torch is built with a release the functions were not measured on, or
  where a build here would link another libdevice.

Nothing here is specific to sm_103 (the comparisons run on the GPU of the test).
"""

import os
import warnings

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.cuda_math]

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import cuda_ref  # noqa: E402
import primitive_cases as pc  # noqa: E402
import tanh_cases as tc  # noqa: E402
from gpu_util import same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, TilefishCudaMathWarning, f32, runtime  # noqa: E402
from tilefish import _cudamath as cm  # noqa: E402

F32, I32 = torch.float32, torch.int32
CUDA = "cuda"
CHUNK = 1 << 26
N = Dim("N")

SOURCE = r"""
#define MAP(name, expr) extern "C" __global__ void name(const float* x, float* y, int n) { \
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float a = x[i]; y[i] = (expr); } }
MAP(k_expf, expf(a))
MAP(k_silu, a / (1.0f + expf(-a)))
MAP(k_rsqrtf, rsqrtf(a))
MAP(k_erff, erff(a))
MAP(k_sinf, sinf(a))
MAP(k_cosf, cosf(a))
MAP(k_tanhf, tanhf(a))
"""


@tilefish.kernel(math="ieee")
def exp_ieee(x: Tensor[N, f32]) -> Tensor[N, f32]:
    return tilefish.exp(x)


@tilefish.kernel(math="ieee")
def silu_ieee(x: Tensor[N, f32]) -> Tensor[N, f32]:
    return tilefish.silu(x)


@tilefish.kernel(math="ieee")
def rsqrt_ieee(x: Tensor[N, f32]) -> Tensor[N, f32]:
    return tilefish.rsqrt(x)


@tilefish.kernel(math="ieee")
def sin_ieee(x: Tensor[N, f32]) -> Tensor[N, f32]:
    return tilefish.sin(x)


@tilefish.kernel
def sin_fast(x: Tensor[N, f32]) -> Tensor[N, f32]:
    return tilefish.sin(x)


@tilefish.kernel
def cos_fast(x: Tensor[N, f32]) -> Tensor[N, f32]:
    return tilefish.cos(x)


@tilefish.kernel
def gelu_fast(x: Tensor[N, f32]) -> Tensor[N, f32]:
    return tilefish.gelu(x)


# function -> [(the op's kernel, the CUDA C kernel, whether a subnormal argument is flushed to a zero of its sign by the op)]
CASES = {
    "expf": [(exp_ieee, "k_expf", False), (silu_ieee, "k_silu", False)],
    "rsqrtf": [(rsqrt_ieee, "k_rsqrtf", False)],
    "erff": [(pc.erf_ieee, "k_erff", False)],
    "sinf": [(sin_ieee, "k_sinf", False), (sin_fast, "k_sinf", True)],
    "cosf": [(cos_fast, "k_cosf", False)],
    "tanhf": [(tc.tanh_ieee, "k_tanhf", False), (tc.tanh_fast, "k_tanhf", True)],
}


def compilers():
    paths = [None] + [p for p in os.environ.get("TILEFISH_NVRTC_PATHS", "").split(":") if p]
    out = []
    for p in paths:
        try:
            out.append(cuda_ref.nvrtc(p))
        except (cuda_ref.ReferenceUnavailable, OSError) as e:
            out.append(pytest.param(None, marks=pytest.mark.skip(reason=f"NVRTC {p}: {e}")))
    return out


def chunks():
    for start in range(0, 1 << 32, CHUNK):
        yield torch.arange(start, start + CHUNK, dtype=torch.int64, device=CUDA).to(I32).view(F32)


def test_every_function_of_the_table_is_measured():
    assert set(CASES) == set(cm.VERIFIED)


@pytest.mark.parametrize("compiler", compilers(), ids=lambda c: f"nvrtc-{c.version[0]}.{c.version[1]}" if c else "nvrtc")
def test_the_table_is_what_each_release_computes(compiler):
    release = f"{compiler.version[0]}.{compiler.version[1]}"
    try:
        module = cuda_ref.Module(cuda_ref.compile_cuda(SOURCE, compiler=compiler))
    except cuda_ref.ReferenceUnavailable as e:
        pytest.skip(f"CUDA {release}: {e}")
    stale = []
    for function, cases in CASES.items():
        differs = 0
        for kernel, name, flushes in cases:
            for x in chunks():
                got = tilefish.run(kernel, x).view(I32)
                want = cuda_ref.map_f32(module, name, x)
                if flushes:  # libdevice built with flush to zero: a subnormal argument is a zero of its sign
                    want = torch.where((x != 0) & (x.abs() < 2.0**-126), x * 0.0, want)
                differs += int((got != want.view(I32)).sum())
        listed = release in cm.VERIFIED[function]
        print(f"[cuda math] CUDA {release} {function}: {differs} inputs differ ({'listed' if listed else 'not listed'})")
        if listed and differs:
            stale.append(f"{function} is listed for CUDA {release} but differs on {differs} inputs: take {release} out of VERIFIED")
        if not listed and not differs:
            stale.append(f"{function} is the op's bits on CUDA {release} on every f32: add {release} to VERIFIED['{function}']")
    assert not stale, stale


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cuda-math")
    return directory, runtime.export([pc.erf_ieee, tc.tanh_fast, gelu_fast, pc.fma3, exp_ieee], directory)


def test_an_export_records_the_functions_the_releases_and_the_libdevice(exported):
    directory, m = exported
    entries = {e["op"]: e for e in m.doc["kernels"]}
    libdevice = cm.libdevice_sha256()
    assert len(libdevice) == 64
    assert entries["erf_ieee"]["cuda_math"] == {"functions": ["erff"], "cuda": list(cm.VERIFIED["erff"])}  # inline PTX: no libdevice
    assert entries["tanh_fast"]["cuda_math"] == {"functions": ["tanhf"], "cuda": list(cm.VERIFIED["tanhf"]), "libdevice": libdevice}
    assert entries["gelu_fast"]["cuda_math"] == {"functions": [], "cuda": [], "libdevice": libdevice}  # libdevice's erf claims no release
    assert entries["exp_ieee"]["cuda_math"] == {"functions": ["expf"], "cuda": list(cm.VERIFIED["expf"])}
    assert "cuda_math" not in entries["fma3"]


def test_a_replay_warns_where_torch_is_built_with_a_release_the_functions_were_not_measured_on(exported, monkeypatch):
    directory, m = exported
    x = torch.randn(4096, device=CUDA)
    jit = tilefish.run(pc.erf_ieee, x)
    monkeypatch.setattr(cm, "torch_release", lambda: "13.2")
    with pytest.warns(TilefishCudaMathWarning, match=r"'erf_ieee' \(sm_\d+\): its erff is the bits of CUDA's math library on CUDA 12.8, 12.9, 13.0, 13.1"):
        got = tilefish.run(pc.erf_ieee, x, mode="replay", manifest=directory, pin=m.digest)
    assert same_bytes(got, jit)  # the binary is what it was: a warning, not another result
    with warnings.catch_warnings():
        warnings.simplefilter("error", TilefishCudaMathWarning)
        tilefish.run(tc.tanh_fast, x, mode="replay", manifest=directory, pin=m.digest)  # tanhf was measured on 13.2
        with pytest.raises(TilefishCudaMathWarning):  # and an error where a program asks for one
            tilefish.run(pc.erf_ieee, x, mode="replay", manifest=directory, pin=m.digest)


def test_a_replay_warns_where_a_build_here_would_link_another_libdevice(exported, monkeypatch):
    directory, m = exported
    x = torch.randn(4096, device=CUDA)
    monkeypatch.setattr(cm, "libdevice_sha256", lambda options=None: "0" * 64)
    with pytest.warns(TilefishCudaMathWarning, match="a build here would link another"):
        tilefish.run(tc.tanh_fast, x, mode="replay", manifest=directory, pin=m.digest)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TilefishCudaMathWarning)
        tilefish.run(pc.erf_ieee, x, mode="replay", manifest=directory, pin=m.digest)  # no libdevice: nothing to compare
