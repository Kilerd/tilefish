"""The facts behind the numerics contract of ``tilefish.lower``, asserted on the PTX and on the outputs of real binaries.

Every claim the contract makes about what a construct lowers to is checked here, on a GPU, against the
PTX that Triton produces and the bytes that the binary computes. Where a claim was wrong the contract was
fixed (see its docstring); this file is what keeps it true. Measured with Triton 3.5.1, ptxas 12.9.86,
on an NVIDIA B300 (sm_103). The PTX facts hold for every architecture of the same Triton unless a test says
otherwise (they are checked on the device's own architecture).

Summary of what the tests below pin down:

* ``a / b`` is ``div.full.f32``: the numerator times an approximate reciprocal, up to 2 ULP (1.8e-7 relative
  measured), never ``div.rn.f32``. ``x / c`` with a constant ``c`` is still ``div.full.f32`` in the PTX, and
  bit-identical to ``x * f32(1 / c)`` in its results, which differ from correctly rounded division in 15% to
  58% of the inputs (none for a power of two).
* ``tl.sqrt_rn`` is ``sqrt.rn.ftz.f32``: round to nearest, but subnormal inputs and results flush to zero.
  ``tl.rsqrt`` is ``rsqrt.approx.ftz.f32``. ``tl.exp`` is ``ex2.approx.f32`` of ``x * log2(e)``.
* ``tl.math.erf`` (and ``tl.erf``) exists, links (libdevice is inlined, no calls remain) and is a polynomial
  of ``fma.rn.ftz.f32`` and ``ex2.approx.ftz.f32``.
* ``float("-inf")`` works inside a Triton function, in JIT and in the AOT build alike (``0fFF800000``).
* ``a * b + c`` is contracted into one FMA, with one rounding instead of two: ``fma.rn.f32`` for f32 and
  ``fma.rn.bf16x2`` for bf16 (a native packed bf16 FMA). Without contraction it is ``mul.rn`` and ``add.rn``.
  The two differ in 23% of random elements, in f32 and in bf16 alike.
* A python float or a constexpr float combined with a bf16 tensor stays bf16 (the scalar is rounded to
  bf16); it is not promoted to f32. Only ``/`` promotes a bf16 operand (Triton divides in f32).
* Float to integer and float to fp8 casts saturate or wrap as the tables of ``cast_cases`` say, which is
  not what torch does.
* ``num_warps`` changes the order of a reduction, so the same kernel on the same data is not byte-identical
  across warp counts, though always within a few ULP of the others and of the reference.
"""

import collections
import re

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import cast_cases  # noqa: E402
import gpu_kernels  # noqa: E402
import triton.language as tl  # noqa: E402
from conftest import REPO_ROOT, make_my_kernels, readme_snippet  # noqa: E402
from gpu_util import differing, error, randn, same_bytes  # noqa: E402
from lower_cases import T, gelu_exact, layernorm, softmax, swiglu  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, bf16, const, f32  # noqa: E402
from tilefish import runtime  # noqa: E402
from tilefish.aot import build  # noqa: E402

BF16, F32 = torch.bfloat16, torch.float32
CAP = torch.cuda.get_device_capability()
ARCH = f"sm_{CAP[0] * 10 + CAP[1]}"
N = 1 << 20


def built(kernel, **options):
    """The cubin of a kernel (one binary), built with Triton options on top of the spec's."""
    low = kernel if hasattr(kernel, "extern") else kernel.lower()
    (spec,) = low.specs()
    return build(spec, arch=ARCH, fn=runtime.load_lowered(low), options=options)


def instructions(ptx):
    """Instruction mnemonics of a PTX text and how often each occurs (predicates stripped)."""
    ops = collections.Counter()
    for line in ptx.splitlines():
        s = line.strip()
        if not s or s[0] in ".{}$()" or s.startswith("//"):
            continue
        m = re.match(r"(?:@!?%p\d+\s+)?([a-z][a-z0-9_.]*)", s)
        if m:
            ops[m.group(1)] += 1
    return ops


def replay(kernel, *args, **options):
    clones = [a.clone() if torch.is_tensor(a) else a for a in args]
    with runtime.build_options(**options):
        return tilefish.run(kernel, *clones, mode="replay")


def probe(values, dtype=F32):
    return torch.tensor(values + [0.0] * (-len(values) % 16), dtype=dtype, device="cuda")


@pytest.fixture(scope="module")
def rmsnorm():
    if not (REPO_ROOT / "README.md").exists():
        pytest.skip("README.md is not available (not running from a source checkout)")
    ns = {"my_kernels": make_my_kernels()}
    exec(compile(readme_snippet(), "README.md", "exec"), ns)
    return ns["residual_add_rmsnorm"]


# -- one kernel per construct ----------------------------------------------------------------------------------------------


@tilefish.kernel
def sqrt_of(x: Tensor[T, f32]) -> Tensor[T, f32]:
    return tilefish.sqrt(x)


@tilefish.kernel
def rsqrt_of(x: Tensor[T, f32]) -> Tensor[T, f32]:
    return tilefish.rsqrt(x)


@tilefish.kernel
def exp_of(x: Tensor[T, f32]) -> Tensor[T, f32]:
    return tilefish.exp(x)


@tilefish.kernel
def divide(a: Tensor[T, f32], b: Tensor[T, f32]) -> Tensor[T, f32]:
    return a / b


@tilefish.kernel
def fma_bf16(a: Tensor[T, bf16], b: Tensor[T, bf16], c: Tensor[T, bf16]) -> Tensor[T, bf16]:
    return a * b + c


@tilefish.kernel
def fma_f32(a: Tensor[T, f32], b: Tensor[T, f32], c: Tensor[T, f32]) -> Tensor[T, f32]:
    return a * b + c


@tilefish.kernel
def scalars_with_bf16(x: Tensor[T, bf16], *, eps: const = 1e-3) -> Tensor[T, bf16]:
    return x * 0.1 + eps


# -- the PTX of the contract ---------------------------------------------------------------------------------------------------------


def test_division_is_div_full_f32_and_never_div_rn(rmsnorm):
    for kernel in (softmax, layernorm, swiglu, rmsnorm, divide):
        ops = instructions(built(kernel).ptx)
        assert ops["div.full.f32"] > 0, kernel.name
        assert not [o for o in ops if o.startswith("div.") and o != "div.full.f32"], (kernel.name, dict(ops))


def test_sqrt_rn_is_the_flush_to_zero_form_of_ieee_square_root():
    ops = instructions(built(sqrt_of).ptx)
    assert ops["sqrt.rn.ftz.f32"] > 0
    assert ops["sqrt.rn.f32"] == 0 and ops["sqrt.approx.f32"] == 0 and ops["sqrt.approx.ftz.f32"] == 0
    # round to nearest: the same bytes as IEEE square root on normal numbers ...
    x = randn((N,), F32, 0, scale=10.0).abs() + 1e-3
    assert same_bytes(replay(sqrt_of, x), torch.sqrt(x))
    # ... and subnormal inputs flush to zero, which IEEE does not do
    tiny = probe([1e-40, 1e-44, 1e-38, 1.5e-38])  # the smallest normal f32 is 1.18e-38: the first three are subnormal
    out = replay(sqrt_of, tiny)
    assert out[:3].tolist() == [0.0, 0.0, 0.0] and out[3] > 0
    assert (torch.sqrt(tiny)[:3] > 0).all()  # IEEE gives 1e-20 for 1e-40


def test_rsqrt_is_the_approximate_flush_to_zero_instruction():
    ops = instructions(built(rsqrt_of).ptx)
    assert ops["rsqrt.approx.ftz.f32"] > 0 and ops["rsqrt.approx.f32"] == 0
    x = randn((N,), F32, 1, scale=10.0).abs() + 1e-3
    exact = (1.0 / x.double().sqrt()).float()
    ulp, _, count, total = error(replay(rsqrt_of, x), exact)
    print(f"rsqrt: max ULP {ulp}, {count} of {total} elements differ from the correctly rounded value")
    assert 1 <= ulp <= 4  # approximate, not exact
    assert replay(rsqrt_of, probe([1e-40]))[0] == float("inf")  # a subnormal input is zero


def test_exp_is_ex2_approx_of_x_times_log2_e():
    ops = instructions(built(exp_of).ptx)
    assert ops["ex2.approx.f32"] > 0 and ops["ex2.approx.ftz.f32"] == 0
    assert "0f3FB8AA3B" in built(exp_of).ptx  # log2(e) as an f32
    near = randn((N,), F32, 2, scale=0.5)
    far = randn((N,), F32, 3, scale=8.0)
    for name, x, bound in (("|x| < 2", near, 8), ("|x| up to 40", far, 64)):
        exact = x.double().exp().float()
        ulp, _, count, total = error(replay(exp_of, x), exact)
        print(f"exp {name}: max ULP {ulp}, {count} of {total} differ from the correctly rounded value")
        assert 1 <= ulp <= bound  # the argument is rounded to f32 after the multiply: the error grows with |x|


def test_erf_exists_links_and_is_inlined():
    assert hasattr(tl.math, "erf") and hasattr(tl, "erf")
    b = built(gelu_exact)  # compiling it needs tl.math.erf to resolve through libdevice
    ops = instructions(b.ptx)
    assert not [o for o in ops if o == "call" or o.startswith("call.")], "an unresolved libdevice call is left"
    assert "%__nv_erff.exit" in b.ptx  # libdevice's __nv_erff was inlined (the label of its end is left as a comment)
    assert ops["fma.rn.ftz.f32"] > 20 and ops["ex2.approx.ftz.f32"] > 0  # the polynomial and the exponential inside erf


def test_float_minus_inf_works_in_jit_and_in_the_binary():
    b = built(softmax)
    assert "0fFF800000" in b.ptx  # -inf, the value of the masked lanes of the max reduction
    # a row of 1000 values around -1000 padded to 1024 lanes: if the masked lanes were 0 instead of -inf the
    # max would be 0, exp(-1000) would underflow, and the row would be 0 / 0
    x = randn((9, 1000), F32, 0, scale=2.0, shift=-1000.0)
    jit = tilefish.run(softmax, x, mode="jit")
    assert torch.isfinite(jit).all() and torch.allclose(jit.sum(-1), torch.ones(9, device="cuda"), atol=1e-5)
    assert same_bytes(jit, tilefish.run(softmax, x, mode="replay"))
    # and a literal -inf in the input is a masked-out element
    y = randn((3, 1000), F32, 1)
    y[:, 5] = float("-inf")
    out = tilefish.run(softmax, y, mode="jit")
    assert (out[:, 5] == 0).all() and torch.isfinite(out).all()


# -- casts -----------------------------------------------------------------------------------------------------------------------------


def cast_runs(kernel, values):
    x = probe(values)
    outputs = {m: tilefish.run(kernel, x, mode=m) for m in ("jit", "replay", "reference")}
    n = len(values)
    return x, {m: o[:n] for m, o in outputs.items()}, outputs


@pytest.mark.parametrize(
    "kernel, instruction, values, expected",
    [
        (cast_cases.to_u8, "cvt.rzi.u16.f32", cast_cases.U8_INPUT, cast_cases.U8_OUTPUT),
        (cast_cases.to_i32, "cvt.rzi.s32.f32", cast_cases.I32_INPUT, cast_cases.I32_OUTPUT),
        (cast_cases.to_i64, "cvt.rzi.s64.f32", cast_cases.I64_INPUT, cast_cases.I64_OUTPUT),
    ],
)
def test_float_to_integer_casts(kernel, instruction, values, expected):
    assert instructions(built(kernel).ptx)[instruction] > 0
    x, out, _ = cast_runs(kernel, values)
    assert out["replay"].tolist() == expected
    assert same_bytes(out["jit"], out["replay"]) and same_bytes(out["reference"], out["replay"])  # the reference follows


def test_float_to_u8_is_neither_saturating_nor_the_cast_torch_does():
    # u8: the value is converted to a 16-bit unsigned integer (saturating there) and the low byte is kept
    x, out, _ = cast_runs(cast_cases.to_u8, cast_cases.U8_INPUT)
    by_value = dict(zip(cast_cases.U8_INPUT, out["replay"].tolist()))
    assert by_value[255.5] == 255 and by_value[256.0] == 0 and by_value[300.0] == 44  # wraps modulo 256 up to 65535 ...
    assert by_value[65536.0] == 255 and by_value[1e10] == 255 and by_value[float("inf")] == 255  # ... then 65535 & 255
    assert by_value[-1.0] == 0 and by_value[-1000.0] == 0 and by_value[float("-inf")] == 0  # negative: 0
    assert out["replay"][cast_cases.U8_INPUT.index(cast_cases.NAN)] == 0  # NaN: 0
    torch_cast = x.to(torch.uint8)[: len(cast_cases.U8_INPUT)]
    assert torch_cast[cast_cases.U8_INPUT.index(-1.5)] == 255 and torch_cast[cast_cases.U8_INPUT.index(65536.0)] == 0
    assert not torch.equal(torch_cast, out["replay"])


@pytest.mark.parametrize(
    "kernel, instruction, table, torch_dtype, torch_says",
    [
        (cast_cases.to_e4m3, "cvt.rn.satfinite.e4m3x2.f32", cast_cases.E4M3_TABLE, torch.float8_e4m3fn, {1000.0: 0x7F, float("inf"): 0x7F}),
        (cast_cases.to_e5m2, "cvt.rn.satfinite.e5m2x2.f32", cast_cases.E5M2_TABLE, torch.float8_e5m2, {1e5: 0x7C, float("inf"): 0x7C}),
    ],
)
def test_float_to_fp8_saturates_to_the_largest_finite_value(kernel, instruction, table, torch_dtype, torch_says):
    assert instructions(built(kernel).ptx)[instruction] > 0
    values = [v for v, _ in table]
    x, out, _ = cast_runs(kernel, values)
    bytes_out = out["replay"].view(torch.uint8).tolist()
    assert bytes_out == [b for _, b in table]
    assert out["jit"].view(torch.uint8).tolist() == bytes_out and out["reference"].view(torch.uint8).tolist() == bytes_out
    torch_bytes = dict(zip(values, x.to(torch_dtype).view(torch.uint8)[: len(values)].tolist()))
    for value, byte in torch_says.items():  # torch overflows to NaN (e4m3) or infinity (e5m2) instead
        assert torch_bytes[value] == byte
        assert bytes_out[values.index(value)] != byte


@pytest.mark.parametrize("kernel, dtype", [(cast_cases.to_f16, torch.float16), (cast_cases.to_bf16, torch.bfloat16)])
def test_float_to_half_types_round_to_nearest_even_and_overflow_to_infinity(kernel, dtype):
    instruction = "cvt.rn.f16x2.f32" if dtype == torch.float16 else "cvt.rn.bf16x2.f32"
    assert instructions(built(kernel).ptx)[instruction] > 0
    x, out, _ = cast_runs(kernel, cast_cases.HALF_INPUT)
    n = len(cast_cases.HALF_INPUT)
    finite = ~torch.isnan(x[:n])
    assert same_bytes(out["replay"][finite], x[:n][finite].to(dtype))  # the same bytes as torch's cast
    assert torch.isnan(out["replay"][~finite]).all()
    assert same_bytes(out["reference"], out["replay"]) and same_bytes(out["jit"], out["replay"])
    if dtype == torch.float16:
        values = dict(zip(cast_cases.HALF_INPUT, out["replay"].tolist()))
        assert values[65519.0] == 65504.0 and values[65520.0] == float("inf")  # round to nearest, ties to even


# -- FMA contraction: a * b + c in one rounding or in two ---------------------------------------------------------------------


@pytest.mark.parametrize("kernel, dtype, fused, plain", [(fma_bf16, BF16, "fma.rn.bf16x2", ("mul.rn.bf16x2", "add.rn.bf16x2")), (fma_f32, F32, "fma.rn.f32", ("mul.rn.f32", "add.rn.f32"))])
def test_a_times_b_plus_c_is_one_fma_with_one_rounding(kernel, dtype, fused, plain):
    a, b, c = (randn((N,), dtype, s) for s in (0, 1, 2))
    on, off = instructions(built(kernel).ptx), instructions(built(kernel, enable_fp_fusion=False).ptx)
    assert on[fused] > 0 and all(on[p] == 0 for p in plain)  # contracted by default
    assert off[fused] == 0 and all(off[p] > 0 for p in plain)  # two instructions, two roundings, without it
    two_roundings = (a * b) + c  # torch: the product is rounded, then the sum
    one_rounding = (a.double() * b.double() + c.double()).to(F32).to(dtype)  # exact product and sum, rounded once
    fused_out, plain_out = replay(kernel, a, b, c), replay(kernel, a, b, c, enable_fp_fusion=False)
    assert same_bytes(fused_out, one_rounding)
    assert same_bytes(plain_out, two_roundings)
    differ = differing(fused_out, plain_out)
    print(f"{kernel.name}: contraction changes {differ} of {N} elements ({100 * differ / N:.1f}%)")
    assert 0.15 * N < differ < 0.35 * N
    # they differ by at most the two roundings of the unfused version: about 2 * 2**-(p + 1) of |a * b| + |c|
    u = 2.0 ** -(8 if dtype == BF16 else 24)
    bound = 3 * u * (a.double().abs() * b.double().abs() + c.double().abs())
    assert ((fused_out.double() - plain_out.double()).abs() <= bound).all()


# -- division by a constant ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("c", [3.0, 5.0, 7.0, 10.0, 100.0, 768.0, 1000.0, 2560.0, 3072.0, 1e-5, 0.1, 1024.0])
def test_division_by_a_constant_is_a_multiply_by_the_rounded_reciprocal(c):
    @tilefish.kernel
    def over(x: Tensor[T, f32], *, d: const[f32] = c) -> Tensor[T, f32]:
        return x / d

    assert instructions(built(over).ptx)["div.full.f32"] > 0  # the PTX says div.full: Triton does not strength-reduce it
    x = randn((N,), F32, 0, scale=100.0)
    out = replay(over, x)
    constant = torch.tensor(c, dtype=F32)  # the f32 the binary divides by
    times_reciprocal = x * (torch.ones((), dtype=F32) / constant).to("cuda")
    correctly_rounded = (x.double() / float(constant)).float()
    assert same_bytes(out, times_reciprocal)
    # the contract of the reference is that bit-identical result
    assert same_bytes(out, tilefish.run(over, x, mode="reference"))
    wrong = differing(out, correctly_rounded)
    print(f"x / {c}: {wrong} of {N} elements differ from correctly rounded division ({100 * wrong / N:.0f}%)")
    if c == 1024.0:
        assert wrong == 0  # a power of two: exact either way
    else:
        assert wrong > 0.05 * N


def test_division_between_tensors_is_an_approximation_of_up_to_two_ulp():
    a, b = randn((N,), F32, 0, scale=100.0), randn((N,), F32, 1, scale=10.0, shift=20.0)
    out = replay(divide, a, b)
    ulp, _, count, total = error(out, a / b)
    print(f"a / b: max ULP {ulp}, {count} of {total} elements differ from correctly rounded division")
    assert 1 <= ulp <= 2 and count > 0.05 * total
    assert ((out - (a / b)).abs() / (a / b).abs().clamp_min(1e-30)).max() < 2.5e-7


def test_the_mean_of_a_row_divides_by_the_constant_row_length():
    ops = instructions(built(layernorm).ptx)
    assert ops["div.full.f32"] == 2 and ops["rcp.approx.f32"] == 0 and ops["rcp.rn.f32"] == 0  # two means, divided by D = 768
    x = randn((5, 768), BF16, 0)
    assert tilefish.run(layernorm, x, randn((768,), BF16, 1), randn((768,), BF16, 2), mode="reference").shape == (5, 768)


# -- a python float or a constexpr float with a bf16 tensor ---------------------------------------------------------------------


def test_triton_keeps_bf16_for_a_python_float_and_a_constexpr_float():
    x, y = randn((16,), BF16, 0), torch.empty(16, dtype=BF16, device="cuda")
    gpu_kernels.scalar_dtype_probe[(1,)](x, y, EPS=1e-3)  # its static_asserts say the results are bf16
    gpu_kernels.division_dtype_probe[(1,)](x, y, EPS=1e-3)  # ... except a division, which is done in f32 and returns f32
    with pytest.raises(Exception, match="static_assert|Assertion|AssertionError"):  # the checks are real: a false one fails
        gpu_kernels.scalar_dtype_wrong_probe[(1,)](x, y, EPS=1e-3)


def test_a_scalar_is_rounded_to_bf16_and_the_arithmetic_stays_bf16():
    x = randn((N,), BF16, 0)
    b = built(scalars_with_bf16)
    ops = instructions(b.ptx)
    assert ops["fma.rn.bf16x2"] > 0  # x * 0.1 + eps, contracted, in packed bf16
    assert ops["cvt.f32.bf16"] == 0 and ops["cvt.rn.bf16x2.f32"] == 0  # no promotion to f32 and back
    tenth, eps = torch.tensor(0.1, dtype=BF16, device="cuda"), torch.tensor(1e-3, dtype=BF16, device="cuda")
    # contraction off: two bf16 operations on scalars that were rounded to bf16 first, which is the reference
    plain = replay(scalars_with_bf16, x, enable_fp_fusion=False)
    assert same_bytes(plain, (x * tenth) + eps)
    assert same_bytes(plain, tilefish.run(scalars_with_bf16, x, mode="reference"))
    assert instructions(built(scalars_with_bf16, enable_fp_fusion=False).ptx)["mul.rn.bf16x2"] > 0
    # contraction on (the default): one rounding of the exact product and sum of bf16 x and the bf16-rounded scalars
    fused = replay(scalars_with_bf16, x)
    one_rounding = (x.float() * float(tenth) + float(eps)).to(BF16)
    assert same_bytes(fused, one_rounding)
    # and it is not computed in f32 with the exact python scalars (that is what promotion would give)
    promoted = (x.float() * 0.1 + 1e-3).to(BF16)
    assert differing(fused, promoted) > 0.05 * N and differing(plain, promoted) > 0.05 * N


# -- num_warps ----------------------------------------------------------------------------------------------------------------------------


def warps_kernels(d):
    dim = Dim("D", d)
    rows = Dim("R")

    @tilefish.kernel
    def softmax_rows(x: Tensor[rows, dim, f32]) -> Tensor[rows, dim, f32]:
        m = tilefish.max(x, axis=-1, keepdims=True)
        e = tilefish.exp(x - m)
        return e / tilefish.sum(e, axis=-1, keepdims=True)

    @tilefish.kernel
    def normalize_rows(x: Tensor[rows, dim, bf16], w: Tensor[dim, bf16], *, eps: const[f32] = 1e-5) -> Tensor[rows, dim, f32]:
        xf = x.to(f32)
        d_ = xf - tilefish.mean(xf, axis=-1, keepdims=True)
        var = tilefish.mean(d_ * d_, axis=-1, keepdims=True)
        return d_ * tilefish.rsqrt(var + eps) * w.to(f32)

    return softmax_rows, normalize_rows


@pytest.mark.parametrize("d, default_warps", [(96, 1), (1000, 4)])
def test_the_warp_count_changes_the_order_of_a_reduction_and_so_the_bytes(d, default_warps):
    softmax_rows, normalize_rows = warps_kernels(d)
    assert softmax_rows.lower().num_warps == default_warps  # one warp per 256 lanes of BLOCK_D
    xs = randn((129, d), F32, 0, scale=3.0)
    xb, wb = randn((129, d), BF16, 1), randn((d,), BF16, 2)
    for kernel, args in ((softmax_rows, (xs,)), (normalize_rows, (xb, wb))):
        results = {w: tilefish.run(tilefish.lower(kernel, num_warps=w), *args, mode="replay") for w in (1, 2, 4, 8, 16, 32)}
        reference = tilefish.run(kernel, *args, mode="reference")
        rows = {w: int((r.view(torch.uint8) != results[1].view(torch.uint8)).any(-1).sum()) for w, r in results.items()}
        print(f"D={d} {kernel.name}: rows (of 129) that differ from num_warps=1: {rows}")
        for w, r in results.items():
            assert (r - reference).abs().max() < 2e-6, (kernel.name, w)  # always correct ...
        assert any(rows[w] > 0 for w in (2, 4, 8, 16, 32))  # ... but not byte-identical across warp counts
        assert rows[1] == 0


def test_the_largest_buckets_warp_count_changes_the_numerics_of_a_small_bucket():
    """A bucketed row length has one ``num_warps`` for all its binaries (from the largest bucket)."""
    small, large = Dim("Dsmall", 96), Dim("Dbuckets", buckets=[96, 2048])
    rows = Dim("R")

    @tilefish.kernel
    def static_rows(x: Tensor[rows, small, bf16], w: Tensor[small, bf16], *, eps: const[f32] = 1e-5) -> Tensor[rows, small, f32]:
        xf = x.to(f32)
        d_ = xf - tilefish.mean(xf, axis=-1, keepdims=True)
        var = tilefish.mean(d_ * d_, axis=-1, keepdims=True)
        return d_ * tilefish.rsqrt(var + eps) * w.to(f32)

    @tilefish.kernel
    def bucketed_rows(x: Tensor[rows, large, bf16], w: Tensor[large, bf16], *, eps: const[f32] = 1e-5) -> Tensor[rows, large, f32]:
        xf = x.to(f32)
        d_ = xf - tilefish.mean(xf, axis=-1, keepdims=True)
        var = tilefish.mean(d_ * d_, axis=-1, keepdims=True)
        return d_ * tilefish.rsqrt(var + eps) * w.to(f32)

    alone, shared = static_rows.lower(), bucketed_rows.lower()
    assert alone.num_warps == 1 and shared.num_warps == 8  # 2048 lanes: one warp per 256
    assert {s.num_warps for s in shared.specs()} == {8}  # the D = 96 binary has 8 warps for 128 lanes
    x, w = randn((129, 96), BF16, 0), randn((96,), BF16, 1)
    one_warp = tilefish.run(static_rows, x, w, mode="replay")
    eight_warps = tilefish.run(bucketed_rows, x, w, mode="replay")  # the same computation, the D = 96 binary of the bucketed kernel
    wrong_rows = int((one_warp.view(torch.uint8) != eight_warps.view(torch.uint8)).any(-1).sum())
    print(f"D=96 with 1 warp or with the 8 warps of the largest bucket: {wrong_rows} of 129 rows differ")
    assert wrong_rows > 0
    assert (one_warp - eight_warps).abs().max() < 2e-6  # both correct, differing in the last bits


# -- FMA contraction on the goldens (data for the lead) ---------------------------------------------------------------------------------


def golden_cases(rmsnorm):
    return [
        ("softmax", softmax, (randn((257, 1000), F32, 0, scale=3.0),)),
        ("layernorm", layernorm, (randn((257, 768), BF16, 0), randn((768,), BF16, 100), randn((768,), BF16, 200))),
        ("residual_add_rmsnorm", rmsnorm, (randn((257, 2560), BF16, 0), randn((257, 2560), F32, 50), randn((2560,), BF16, 100))),
        ("gelu_exact", gelu_exact, (randn((1 << 20,), BF16, 0, scale=3.0),)),
        ("swiglu", swiglu, (randn((257, 3072), BF16, 0, scale=2.0), randn((257, 3072), BF16, 100))),
    ]


def test_fma_contraction_of_the_goldens(rmsnorm):
    """Build every golden with ``enable_fp_fusion`` True (what ``compile_specs`` asks for) and False, and compare.

    Measured on sm_103 (Triton 3.5.1): the numbers are printed; the table is in the report of the change that
    added this test. The assertions are structural: switching contraction off never adds an FMA, and the two builds
    stay within the tolerance of the reference.
    """
    for name, kernel, args in golden_cases(rmsnorm):
        on, off = built(kernel), built(kernel, enable_fp_fusion=False)

        def count(b):
            return sum(v for k, v in instructions(b.ptx).items() if k.startswith("fma."))

        out_on = replay_all(kernel, args, {})
        out_off = replay_all(kernel, args, {"enable_fp_fusion": False})
        out_ref = replay_all(kernel, args, None)
        report = []
        for label, a, b in zip(["out", *[f"inout{i}" for i in range(len(out_on) - 1)]], out_on, out_off):
            same = differing(a, b)
            report.append(f"{label}: {same} of {a.numel()} elements differ ({100 * same / a.numel():.3f}%), max ULP {error(a, b)[0]}")
        print(f"{name}: fma instructions on={count(on)} off={count(off)}; code digests {'equal' if on.sha256_code == off.sha256_code else 'differ'}; " + "; ".join(report))
        assert count(off) <= count(on)
        for a, b, r in zip(out_on, out_off, out_ref):
            for built_output in (a, b):
                ulp, absolute, _, _ = error(built_output, r)
                assert (absolute <= 3e-5) if name == "gelu_exact" else (ulp <= (64 if built_output.dtype == F32 else 2)), name


def replay_all(kernel, args, options):
    """The outputs and the in-place tensors of a call, replayed with these build options (``None``: the reference)."""
    clones = [a.clone() if torch.is_tensor(a) else a for a in args]
    if options is None:
        result = tilefish.run(kernel, *clones, mode="reference")
    else:
        with runtime.build_options(**options):
            result = tilefish.run(kernel, *clones, mode="replay")
    outs = [result] if torch.is_tensor(result) else list(result or [])
    inouts = [c for c, p in zip(clones, [p for p in kernel.signature.params if p.kind == "tensor" and p.access != "out"]) if p.access == "inout"]
    return outs + inouts
