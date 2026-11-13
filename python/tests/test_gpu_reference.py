"""``replay`` against ``reference``, on a real GPU: the prebuilt binaries of a manifest directory agree with the
torch interpretation of the IR within a tolerance that depends on the input distribution, per golden. (``jit`` runs
the same machine code, see ``test_gpu_byte_identity.py``, so the numbers hold for it too.)

The reference follows the numerics contract of ``tilefish.lower`` node by node (casts only where the IR has
them, math in the IR's dtypes, scalars rounded to the tensor's dtype, a constant divisor as a multiply by
the rounded reciprocal), but it is torch: the exact functions, one rounding per operation, torch's
reductions. What the binary does differently is therefore the whole difference, and it is not zero:

* ``exp`` is ``ex2.approx.f32`` of ``x * log2(e)``: its relative error grows with ``|x|`` (the product is
  rounded to f32 before the exponentiation), about 1 ULP per 1.5 of ``|x|`` at the far end of a softmax row;
* ``/`` between tensors is ``div.full.f32``, a reciprocal approximation times the numerator (up to about
  2 ULP), where torch divides correctly rounded;
* ``rsqrt`` is ``rsqrt.approx.ftz.f32`` and ``erf`` is libdevice's polynomial with flush-to-zero FMAs;
* reductions (``sum``, ``mean``, ``max``) add in a different order than torch (a tree over the lanes of the
  block and the warps), so a row sum differs in its last bits;
* FMA contraction: the binary fuses ``a * b + c`` into one rounding (``fma.rn.f32``), the reference rounds
  the product and then the sum.

Where the output is rounded to bf16 these f32 differences usually vanish: only an element within a few f32
ULP of a bf16 rounding boundary flips, by one bf16 ULP.

**What the numbers depend on.** A ULP count is only meaningful where the result is far from zero: an output
that is the difference of two large numbers (a layernorm of data with a large mean, a gelu of a large negative
number) is tiny, so an error that is small against the *scale of the data* is many ULP of the result. The
tolerances are therefore stated per input distribution, in the metric that means something there:

* bf16 outputs: the largest absolute error, as a fraction of the largest value of the reference
  (``max |a - b| <= 2**-7 * max |b|``, one half bf16 step at the top of the range), and the fraction of elements
  that differ;
* the f32 softmax (its outputs are at most 1): the largest absolute error, 5e-7, and its ULP where it is meaningful;
* the gelu: the largest absolute error, 3e-5.

Measured on an NVIDIA B300 (sm_103), Triton 3.5.1, ptxas 12.9.86, seeds 0 to 2 (the tests print the numbers they
measure; run with ``-s``). ``N(m, s)`` is a normal sample with mean ``m`` and standard deviation ``s``:

========================  ===============================  ============  =================  ===================  ====================
golden (dtype)            input distribution               max ULP       max abs error      max abs / max|ref|   elements that differ
========================  ===============================  ============  =================  ===================  ====================
softmax (257, 1000) f32   N(0, 1)                          8 to 10       3.7e-9 to 5.6e-9   6e-8 to 9e-8         70%
                          N(0, 3)                          17 to 28      1.2e-7 to 1.8e-7   1.2e-7 to 2.1e-7     87.5%
                          N(0, 10)                         58 to 61      3.0e-7             3.0e-7               96%
                          N(0, 30)                         63 to 64      1.8e-7 to 3.0e-7   1.8e-7 to 3.0e-7     40% to 42%
                          N(1000, 1)  (shift invariant)    8             3.7e-9 to 7.5e-9   6e-8 to 8e-8         70%
layernorm (257, 768) bf16 N(0, 1)                          1             3.9e-3 to 7.8e-3   3.7e-4 to 6.9e-4     4 to 7 of 197376
                          N(10, 1)                         4 to 26       3.9e-3 to 1.6e-2   3.5e-4 to 1.5e-3     45 to 46
                          N(100, 1)                        17 to 134     3.1e-2             2.4e-3 to 3.1e-3     235 to 283 (0.12%-0.14%)
                          N(1000, 1)                       117 to 28415  3.1e-2 to 6.3e-2   1.5e-3 to 3.4e-3     2596 to 2972 (1.3%-1.5%)
                          N(0, 0.01)                       1 to 12975    4.9e-4 to 1.6e-2   4.5e-5 to 1.6e-3     9 to 11
                          N(0, 30)                         1 to 2        9.8e-4 to 7.8e-3   9.4e-5 to 6.0e-4     4 to 6
residual_add_rmsnorm out  x N(0, 0.01 to 30), residual     1             2.4e-4 to 7.8e-3   1.8e-5 to 7.4e-4     1 to 3 of 657920
  bf16                    N(0, 1)
                          x N(0, 0.01 to 30), residual     0 to 1        2.0e-3 to 1.6e-2   4.9e-4 to 2.3e-3     0 to 5 of 657920
                          N(100, 1)
residual (f32, InOut)     the same                         0             0                  0                    0: one f32 add, exact
gelu_exact (2**20) bf16   N(0, 0.5)                        0             0                  0                    0
                          N(0, 3) and N(0, 8)              128           1.5e-5             3.7e-7 to 1.1e-6     14409 to 14781 (1.4%)
swiglu (257, 3072) bf16   N(0, 1)                          1             3.9e-3             3.4e-4 to 4.0e-4     4 to 10 of 789504
                          N(0, 6)                          1             1.6e-2 to 3.1e-2   4.3e-5 to 7.9e-5     2 to 5
                          N(0, 20)                         1             9.5e-7 to 3.1e-2   2.2e-10 to 7.8e-6    33 to 43 (0.005%)
========================  ===============================  ============  =================  ===================  ====================

The tolerances asserted below, per distribution: softmax at most 5e-7 absolute and 16 ULP for N(0, 1) and a shift,
64 for N(0, 3), 128 for N(0, 10) and N(0, 30); every bf16 kernel at most 2**-7 of the largest reference value (the
measured worst case is 3.4e-3, against 7.8e-3), and at most 0.02% (a mean of 0, or a scale of 0.01 or 30), 0.05% (a mean of 10), 0.3%
(a mean of 100) or 3% (a mean of 1000) of the elements different for the layernorm, 0.01% for the rmsnorm,
0.02% for the swiglu; the gelu at most 3e-5 absolute and 3% of the elements different.

The ULP columns of the layernorm with a large mean are not a tolerance: 117 to 28415 ULP are the *tiny* outputs of
``x - mean`` (the data is bf16, so ``x - mean`` of numbers near 1000 is quantized in steps of 4, and the near-zero
results are meaningless in ULP). What bounds the error there is that it stays within half a bf16 step of the largest
value. The 128 ULP of the gelu are the same effect (``1 + erf(x/sqrt 2)`` for large negative ``x``).
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from conftest import REPO_ROOT, make_my_kernels, readme_snippet  # noqa: E402
from gpu_util import differing, error, randn, same_bytes  # noqa: E402
from lower_cases import gelu_exact, layernorm, softmax, swiglu  # noqa: E402

import tilefish  # noqa: E402
from tilefish import runtime  # noqa: E402

BF16, F32 = torch.bfloat16, torch.float32
SEEDS = (0, 1, 2)
HALF_BF16_STEP = 2.0**-7  # of the largest value of the reference


@pytest.fixture(scope="module")
def rmsnorm():
    if not (REPO_ROOT / "README.md").exists():
        pytest.skip("README.md is not available (not running from a source checkout)")
    ns = {"my_kernels": make_my_kernels()}
    exec(compile(readme_snippet(), "README.md", "exec"), ns)
    return ns["residual_add_rmsnorm"]


@pytest.fixture(scope="module")
def manifest(tmp_path_factory, rmsnorm):
    """The five goldens built into a manifest directory, as a host would load them."""
    return runtime.export([softmax, layernorm, rmsnorm, gelu_exact, swiglu], tmp_path_factory.mktemp("goldens")).directory


def both(manifest, op, *args):
    """The replayed binary's result and the reference's, from the same inputs; in-place tensors are returned too."""
    out = {}
    for mode in ("replay", "reference"):
        clones = [a.clone() if torch.is_tensor(a) else a for a in args]
        out[mode] = (tilefish.run(op, *clones, mode=mode, **({"manifest": manifest, "trust_directory": True} if mode == "replay" else {})), clones)
    return out


def measure(name, binary, reference):
    """``(max ULP, max abs, max abs / max |reference|, fraction of elements that differ)``, printed."""
    ulp, absolute, count, total = error(binary, reference)
    largest = float(reference.double().abs().max())
    relative = absolute / largest if largest else 0.0
    print(f"{name}: max ULP {ulp}, max abs error {absolute:.3g}, max abs / max|ref| {relative:.3g}, {count} of {total} elements differ")
    return ulp, absolute, relative, count / total


def check_bf16(name, binary, reference, *, differ):
    ulp, absolute, relative, fraction = measure(name, binary, reference)
    assert relative <= HALF_BF16_STEP, name  # within half a bf16 step of the largest value
    assert fraction <= differ, name


# -- softmax (f32, outputs at most 1) ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scale, shift, max_ulp", [(1.0, 0.0, 16), (3.0, 0.0, 64), (10.0, 0.0, 128), (30.0, 0.0, 128), (1.0, 1000.0, 16)])
def test_softmax(manifest, scale, shift, max_ulp):
    for seed in SEEDS:
        r = both(manifest, softmax, randn((257, 1000), F32, seed, scale=scale, shift=shift))
        ulp, absolute, _, fraction = measure(f"softmax N({shift:g}, {scale:g}) seed {seed}", r["replay"][0], r["reference"][0])
        assert absolute <= 5e-7 and ulp <= max_ulp
        assert fraction > 0  # they do differ: ex2.approx and div.full against torch's exp and division
        assert torch.allclose(r["replay"][0].sum(-1), torch.ones(257, device="cuda"), atol=1e-5)


# -- layernorm (bf16): the mean of the data ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mean, scale, differ",
    [(0.0, 1.0, 0.0002), (10.0, 1.0, 0.0005), (100.0, 1.0, 0.003), (1000.0, 1.0, 0.03), (0.0, 0.01, 0.0002), (0.0, 30.0, 0.0002)],
)
def test_layernorm(manifest, mean, scale, differ):
    for seed in SEEDS:
        args = (randn((257, 768), BF16, seed, scale=scale, shift=mean), randn((768,), BF16, seed + 100), randn((768,), BF16, seed + 200))
        r = both(manifest, layernorm, *args)
        check_bf16(f"layernorm N({mean:g}, {scale:g}) seed {seed}", r["replay"][0], r["reference"][0], differ=differ)


def test_layernorm_in_ulp_only_where_the_output_is_not_tiny(manifest):
    """On data around 0 the layernorm is within 3 ULP; ULP says nothing about data around 1000 (see the module docstring)."""
    for seed in SEEDS:
        args = (randn((257, 768), BF16, seed), randn((768,), BF16, seed + 100), randn((768,), BF16, seed + 200))
        r = both(manifest, layernorm, *args)
        assert error(r["replay"][0], r["reference"][0])[0] <= 3
    shifted = (randn((257, 768), BF16, 0, shift=1000.0), randn((768,), BF16, 100), randn((768,), BF16, 200))
    r = both(manifest, layernorm, *shifted)
    assert error(r["replay"][0], r["reference"][0])[0] > 3  # it is far more in ULP, and still within half a bf16 step of the largest value


# -- residual add + rmsnorm --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("x_scale, residual_mean", [(0.01, 0.0), (1.0, 0.0), (30.0, 0.0), (0.01, 100.0), (1.0, 100.0), (30.0, 100.0)])
def test_residual_add_rmsnorm(rmsnorm, manifest, x_scale, residual_mean):
    for seed in SEEDS:
        args = (
            randn((257, 2560), BF16, seed, scale=x_scale),
            randn((257, 2560), F32, seed + 50, shift=residual_mean),
            randn((2560,), BF16, seed + 100),
        )
        r = both(manifest, rmsnorm, *args)
        check_bf16(f"rmsnorm x N(0, {x_scale:g}) residual N({residual_mean:g}, 1) seed {seed}", r["replay"][0], r["reference"][0], differ=0.0001)
        assert error(r["replay"][1][1], r["reference"][1][1]) == (0, 0.0, 0, 257 * 2560)  # a single f32 add: the same bytes
        assert same_bytes(r["replay"][1][1], r["reference"][1][1])


# -- gelu ----------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.5, 3.0, 8.0])
def test_gelu_exact(manifest, scale):
    for seed in SEEDS:
        r = both(manifest, gelu_exact, randn((1 << 20,), BF16, seed, scale=scale))
        ulp, absolute, relative, fraction = measure(f"gelu N(0, {scale:g}) seed {seed}", r["replay"][0], r["reference"][0])
        assert absolute <= 3e-5 and fraction <= 0.03  # an absolute bound: the ULP of a result near zero is not one


# -- swiglu --------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 6.0, 20.0])
def test_swiglu(manifest, scale):
    for seed in SEEDS:
        r = both(manifest, swiglu, randn((257, 3072), BF16, seed, scale=scale), randn((257, 3072), BF16, seed + 100, scale=scale))
        check_bf16(f"swiglu N(0, {scale:g}) seed {seed}", r["replay"][0], r["reference"][0], differ=0.0002)


def test_the_reference_is_deterministic_and_does_not_depend_on_the_binary(manifest):
    args = (randn((17, 1000), F32, 0, scale=3.0),)
    a = tilefish.run(softmax, *args, mode="reference")
    b = tilefish.run(softmax, *args, mode="reference")
    assert same_bytes(a, b)
    assert differing(a, tilefish.run(softmax, *args, mode="replay", manifest=manifest, trust_directory=True)) > 0
