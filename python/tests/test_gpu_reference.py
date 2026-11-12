"""``replay`` against ``reference``, on a real GPU: the prebuilt binaries of a manifest directory agree with the
torch interpretation of the IR within a documented tolerance, per golden. (``jit`` runs the same machine code, see
``test_gpu_byte_identity.py``, so the numbers hold for it too.)

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

Measured on an NVIDIA B300 (sm_103), Triton 3.5.1, ptxas 12.9.86, seeds 0 to 2 (the tests below print
the numbers they measure; run with ``-s``):

============================  ======  ===========  ================  ===========================================
golden (shape)                dtype   max ULP      max abs error     elements that differ
============================  ======  ===========  ================  ===========================================
softmax (257, 1000)           f32     17 to 28     1.8e-7            87% (a sum of 1000 terms, ``ex2.approx``)
layernorm (257, 768)          bf16    1            7.8e-3            4 to 7 of 197376 (one bf16 step, |y| < 2)
residual_add_rmsnorm out      bf16    1            2.4e-4 to 7.8e-3  1 to 3 of 657920
residual_add_rmsnorm InOut    f32     0            0                 0: one f32 add, exact
gelu_exact (2**20)            bf16    128          1.5e-5            1.4%: ``1 + erf(x/sqrt 2)`` cancels for x < 0
swiglu (257, 3072)            bf16    1            9.8e-4 to 3.9e-3  2 to 6 of 789504
============================  ======  ===========  ================  ===========================================

The 128 ULP of the gelu are values near zero (its output for large negative inputs is the difference of
two numbers near 1, so one f32 step of ``erf`` is a large relative error there); the absolute error is what
bounds it. The tolerances asserted below leave room over the measured values: softmax at most 64 ULP and
5e-7, the bf16 kernels at most 2 ULP, one bf16 step (2**-6) at the top of their range and 0.1% of the
elements, the gelu at most 3e-5 and 3% of the elements.
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
        out[mode] = (tilefish.run(op, *clones, mode=mode, **({"manifest": manifest} if mode == "replay" else {})), clones)
    return out


def report(name, e):
    ulp, absolute, count, total = e
    print(f"{name}: max ULP {ulp}, max abs error {absolute:.3g}, {count} of {total} elements differ")


def test_softmax(manifest):
    worst = [0, 0.0]
    for seed in SEEDS:
        r = both(manifest, softmax, randn((257, 1000), F32, seed, scale=3.0))
        e = error(r["replay"][0], r["reference"][0])
        report(f"softmax seed {seed}", e)
        worst = [max(worst[0], e[0]), max(worst[1], e[1])]
        assert e[2] > 0  # they do differ: ex2.approx and div.full against torch's exp and division
        assert torch.allclose(r["replay"][0].sum(-1), torch.ones(257, device="cuda"), atol=1e-5)
    assert worst[0] <= 64 and worst[1] <= 5e-7


def test_layernorm(manifest):
    for seed in SEEDS:
        args = (randn((257, 768), BF16, seed), randn((768,), BF16, seed + 100), randn((768,), BF16, seed + 200))
        r = both(manifest, layernorm, *args)
        e = error(r["replay"][0], r["reference"][0])
        report(f"layernorm seed {seed}", e)
        assert e[0] <= 2 and e[1] <= 2 ** -6 and e[2] <= 0.001 * e[3]


def test_residual_add_rmsnorm(rmsnorm, manifest):
    for seed in SEEDS:
        args = (randn((257, 2560), BF16, seed), randn((257, 2560), F32, seed + 50), randn((2560,), BF16, seed + 100))
        r = both(manifest, rmsnorm, *args)
        e_out = error(r["replay"][0], r["reference"][0])
        e_residual = error(r["replay"][1][1], r["reference"][1][1])
        report(f"rmsnorm out seed {seed}", e_out)
        report(f"rmsnorm InOut residual seed {seed}", e_residual)
        assert e_out[0] <= 2 and e_out[1] <= 2 ** -6 and e_out[2] <= 0.001 * e_out[3]
        assert e_residual == (0, 0.0, 0, 257 * 2560)  # a single f32 add: the same bytes
        assert same_bytes(r["replay"][1][1], r["reference"][1][1])


def test_gelu_exact(manifest):
    for seed in SEEDS:
        r = both(manifest, gelu_exact, randn((1 << 20,), BF16, seed, scale=3.0))
        e = error(r["replay"][0], r["reference"][0])
        report(f"gelu seed {seed}", e)
        assert e[1] <= 3e-5 and e[2] <= 0.03 * e[3]


def test_swiglu(manifest):
    for seed in SEEDS:
        r = both(manifest, swiglu, randn((257, 3072), BF16, seed, scale=2.0), randn((257, 3072), BF16, seed + 100))
        e = error(r["replay"][0], r["reference"][0])
        report(f"swiglu seed {seed}", e)
        assert e[0] <= 2 and e[1] <= 2 ** -6 and e[2] <= 0.001 * e[3]


def test_the_reference_is_deterministic_and_does_not_depend_on_the_binary(manifest):
    args = (randn((17, 1000), F32, 0, scale=3.0),)
    a = tilefish.run(softmax, *args, mode="reference")
    b = tilefish.run(softmax, *args, mode="reference")
    assert same_bytes(a, b)
    assert differing(a, tilefish.run(softmax, *args, mode="replay", manifest=manifest)) > 0
