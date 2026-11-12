"""``mode="reference"``: the traced IR interpreted with torch, on the CPU (``test_gpu_reference.py`` compares it to the binaries).

These tests check that the interpreter does what the IR and the numerics contract say, op by op, against
explicit torch expressions. They need torch, and are skipped where it is not installed.
"""

import pytest
from conftest import REPO_ROOT, make_my_kernels, readme_snippet
import cast_cases
from lower_cases import M, T, gelu_exact, layernorm, softmax, swiglu

import tilefish
from tilefish import InOut, Out, Tensor, TilefishPreconditionError, TilefishRuntimeError, TilefishTypeError, bf16, const, f32

torch = pytest.importorskip("torch")


def g(seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def randn(shape, dtype=torch.float32, seed=0, scale=1.0):
    return (torch.randn(shape, generator=g(seed)) * scale).to(dtype)


def bits(t):
    return t.contiguous().view(torch.uint8)


def same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


@pytest.fixture(scope="module")
def rmsnorm():
    if not (REPO_ROOT / "README.md").exists():
        pytest.skip("README.md is not available (not running from a source checkout)")
    ns = {"my_kernels": make_my_kernels()}
    exec(compile(readme_snippet(), "README.md", "exec"), ns)
    return ns["residual_add_rmsnorm"]


# -- the goldens, against explicit torch --------------------------------------------------------------------------------------


def test_softmax():
    x = randn((6, 1000), scale=3.0)
    y = tilefish.run(softmax, x, mode="reference")
    assert y.dtype == torch.float32 and y.shape == x.shape
    torch.testing.assert_close(y, torch.softmax(x, -1), rtol=1e-5, atol=1e-8)


def test_layernorm():
    x, w, b = randn((5, 768), torch.bfloat16), randn((768,), torch.bfloat16, 1), randn((768,), torch.bfloat16, 2)
    y = tilefish.run(layernorm, x, w, b, mode="reference")
    xf = x.float()
    d = xf - xf.mean(-1, keepdim=True)
    var = (d * d).mean(-1, keepdim=True)
    expected = (d * torch.rsqrt(var + 1e-5) * w.float() + b.float()).to(torch.bfloat16)
    assert y.dtype == torch.bfloat16 and y.shape == x.shape
    torch.testing.assert_close(y.float(), expected.float(), rtol=2**-7, atol=0)  # within one bf16 step


def test_gelu_is_the_erf_form_evaluated_left_to_right():
    x = randn((64,), torch.bfloat16, scale=3.0)
    y = tilefish.run(gelu_exact, x, mode="reference")
    xf = x.float()
    expected = (0.5 * xf * (1.0 + torch.erf(xf * 0.7071067811865476))).to(torch.bfloat16)
    assert same_bytes(y, expected)


def test_swiglu_is_silu_times_up():
    a, b = randn((4, 3072), torch.bfloat16, 0, 2.0), randn((4, 3072), torch.bfloat16, 1)
    y = tilefish.run(swiglu, a, b, mode="reference")
    af = a.float()
    expected = (af * (1.0 / (1.0 + torch.exp(-af))) * b.float()).to(torch.bfloat16)
    assert same_bytes(y, expected)


def test_the_readme_kernel_updates_its_inout_tensor_and_returns_a_fresh_one(rmsnorm):
    x, w = randn((3, 2560), torch.bfloat16), randn((2560,), torch.bfloat16, 1)
    residual = randn((3, 2560), torch.float32, 2)
    before = residual.clone()
    out = tilefish.run(rmsnorm, x, residual, w, mode="reference")
    r = before + x.float()
    assert same_bytes(residual, r)  # a single f32 add: exact
    assert out.dtype == torch.bfloat16 and out.shape == x.shape
    rstd = torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(out.float(), (r * rstd * w.float()).to(torch.bfloat16).float(), rtol=2**-7, atol=0)
    assert out.data_ptr() != residual.data_ptr() and out.data_ptr() != x.data_ptr()


def test_op_run_is_tilefish_run():
    x = randn((8, 1000))
    assert same_bytes(softmax.run(x, mode="reference"), tilefish.run(softmax, x, mode="reference"))
    assert same_bytes(softmax.lower().run(x, mode="reference"), tilefish.run(softmax, x, mode="reference"))


# -- the contract, point by point -----------------------------------------------------------------------------------------------


@tilefish.kernel
def times_tenth_plus_eps(x: Tensor[T, bf16], *, eps: const = 1e-3) -> Tensor[T, bf16]:
    return x * 0.1 + eps


def test_a_python_scalar_is_rounded_to_the_dtype_of_the_tensor_it_meets():
    x = randn((64,), torch.bfloat16)
    y = tilefish.run(times_tenth_plus_eps, x, mode="reference")
    tenth, eps = torch.tensor(0.1, dtype=torch.bfloat16), torch.tensor(1e-3, dtype=torch.bfloat16)
    assert same_bytes(y, (x * tenth) + eps)  # two bf16 roundings, with both scalars rounded to bf16 first
    promoted = (x.float() * 0.1 + 1e-3).to(torch.bfloat16)
    assert not same_bytes(y, promoted)  # not computed in f32 with exact scalars


@tilefish.kernel
def over_seven(x: Tensor[T, f32], *, c: const[f32] = 7.0) -> Tensor[T, f32]:
    return x / c


def test_division_by_a_constant_is_a_multiply_by_the_rounded_reciprocal():
    x = randn((4096,), scale=100.0)
    y = tilefish.run(over_seven, x, mode="reference")
    recip = torch.ones((), dtype=torch.float32) / torch.tensor(7.0, dtype=torch.float32)
    assert same_bytes(y, x * recip)
    assert not same_bytes(y, x / torch.full_like(x, 7.0))  # which is not IEEE division


@tilefish.kernel
def divide(a: Tensor[T, f32], b: Tensor[T, f32]) -> Tensor[T, f32]:
    return a / b


def test_division_between_tensors_is_ieee():
    a, b = randn((256,), seed=0), randn((256,), seed=1) + 5.0
    assert same_bytes(tilefish.run(divide, a, b, mode="reference"), a / b)


Rr = tilefish.Dim("Rr")


@tilefish.kernel
def row_mean(x: Tensor[Rr, tilefish.Dim("Dd", 1000), f32]) -> Tensor[Rr, 1, f32]:
    return tilefish.mean(x, axis=-1, keepdims=True)


def test_mean_is_the_sum_times_the_reciprocal_of_the_count():
    x = randn((9, 1000))
    y = tilefish.run(row_mean, x, mode="reference")
    recip = torch.ones((), dtype=torch.float32) / torch.tensor(1000.0, dtype=torch.float32)
    assert y.shape == (9, 1)
    assert same_bytes(y, x.sum(-1, keepdim=True) * recip)


def probe(values):
    return torch.tensor(values + [0.0] * (-len(values) % 16), dtype=torch.float32)


# What the binaries do (measured on an sm_103 GPU, asserted there by test_gpu_numerics.py): the tables of cast_cases.
def test_float_to_u8_clamps_to_16_bits_and_keeps_the_low_byte():
    y = tilefish.run(cast_cases.to_u8, probe(cast_cases.U8_INPUT), mode="reference")
    assert y.dtype == torch.uint8
    assert y[: len(cast_cases.U8_INPUT)].tolist() == cast_cases.U8_OUTPUT


def test_float_to_i32_saturates_and_nan_is_zero():
    y = tilefish.run(cast_cases.to_i32, probe(cast_cases.I32_INPUT), mode="reference")
    assert y[: len(cast_cases.I32_INPUT)].tolist() == cast_cases.I32_OUTPUT


def test_float_to_i64_saturates_and_nan_is_the_most_negative_value():
    y = tilefish.run(cast_cases.to_i64, probe(cast_cases.I64_INPUT), mode="reference")
    assert y[: len(cast_cases.I64_INPUT)].tolist() == cast_cases.I64_OUTPUT


@pytest.mark.parametrize("kernel, table", [(cast_cases.to_e4m3, cast_cases.E4M3_TABLE), (cast_cases.to_e5m2, cast_cases.E5M2_TABLE)])
def test_float_to_fp8_saturates_to_the_largest_finite_value(kernel, table):
    values = [v for v, _ in table]
    y = tilefish.run(kernel, probe(values), mode="reference")
    assert y.view(torch.uint8)[: len(values)].tolist() == [b for _, b in table]


@pytest.mark.parametrize("dtype, kernel", [(torch.float16, cast_cases.to_f16), (torch.bfloat16, cast_cases.to_bf16)])
def test_casts_to_the_half_types_are_round_to_nearest_even(dtype, kernel):
    x = probe(cast_cases.HALF_INPUT)
    y = tilefish.run(kernel, x, mode="reference")
    finite = ~torch.isnan(x)
    assert same_bytes(y[finite], x[finite].to(dtype))  # infinities and overflow to infinity included
    assert torch.isnan(y[~finite]).all()  # NaN stays NaN


# -- outputs, in-place tensors and stores ------------------------------------------------------------------------------------------


@tilefish.kernel
def out_and_return(x: Tensor[T, f32], y: Out[Tensor[T, f32]]) -> Tensor[T, f32]:
    y[...] = x * 2.0
    return x + 1.0


def test_a_kernel_returns_its_returned_values_then_its_out_tensors():
    x = randn((64,))
    first, second = tilefish.run(out_and_return, x, mode="reference")
    assert same_bytes(first, x + 1.0) and same_bytes(second, x * 2.0)


@tilefish.kernel
def many(x: Tensor[T, f32]):
    return x + 1.0, x * 3.0, x - 1.0


def test_several_returns_are_a_tuple():
    x = randn((64,))
    r = tilefish.run(many, x, mode="reference")
    assert isinstance(r, tuple) and len(r) == 3
    assert same_bytes(r[1], x * 3.0)


@tilefish.kernel
def stores_twice(x: Tensor[T, f32], a: InOut[Tensor[T, f32]], b: InOut[Tensor[T, f32]]):
    a[...] = x
    a[...] = x * 3.0  # the final version is what is written
    b[...] = b  # a copy onto itself writes nothing
    b[...] = b + a  # reads b (the old value) and a (the stored value)


def test_inout_tensors_get_their_final_version_and_reads_see_stores():
    x, a, b = randn((64,), seed=0), randn((64,), seed=1), randn((64,), seed=2)
    b0 = b.clone()
    assert tilefish.run(stores_twice, x, a, b, mode="reference") is None
    assert same_bytes(a, x * 3.0)
    assert same_bytes(b, b0 + x * 3.0)


@tilefish.kernel
def passthrough(x: Tensor[T, f32]) -> Tensor[T, f32]:
    return x


def test_a_returned_input_is_a_copy():
    x = randn((64,))
    y = tilefish.run(passthrough, x, mode="reference")
    assert same_bytes(y, x) and y.data_ptr() != x.data_ptr()


@tilefish.extern(triton="my_kernels:copy", grid=lambda T: (T,))
def copy(x: Tensor[T, f32], y: Out[Tensor[T, f32]], n_elem: T): ...


@tilefish.kernel
def calls_an_extern(x: Tensor[T, f32]) -> Tensor[T, f32]:
    return copy(x)


def test_a_call_to_a_declared_op_is_not_interpreted():
    with pytest.raises(TilefishRuntimeError, match="does not interpret the call to the extern op 'copy'"):
        tilefish.run(calls_an_extern, randn((64,)), mode="reference")


def test_reference_of_an_extern_has_no_ir():
    with pytest.raises(TilefishRuntimeError, match="no IR"):
        tilefish.run(copy, randn((64,)), mode="reference")


# -- what a call may say ---------------------------------------------------------------------------------------------------------------


def test_a_const_can_be_restated_but_not_changed():
    x, w, b = randn((2, 768), torch.bfloat16), randn((768,), torch.bfloat16), randn((768,), torch.bfloat16)
    same = tilefish.run(layernorm, x, w, b, eps=1e-5, mode="reference")
    assert same_bytes(same, tilefish.run(layernorm, x, w, b, mode="reference"))
    with pytest.raises(TilefishTypeError, match="compiled in as 1e-05"):
        tilefish.run(layernorm, x, w, b, eps=1e-3, mode="reference")


def test_types_are_checked_in_reference_mode_too():
    x, w, b = randn((2, 768), torch.bfloat16), randn((768,), torch.bfloat16), randn((768,), torch.bfloat16)
    with pytest.raises(TilefishPreconditionError, match="dtype") as e:
        tilefish.run(layernorm, x.float(), w, b, mode="reference")
    assert e.value.kind == "dtype"
    with pytest.raises(TilefishPreconditionError, match="requires 768") as e:
        tilefish.run(layernorm, randn((2, 700), torch.bfloat16), w, b, mode="reference")
    assert e.value.kind == "static_size"
    with pytest.raises(TilefishPreconditionError, match="not contiguous"):
        tilefish.run(layernorm, randn((768, 2), torch.bfloat16).t(), w, b, mode="reference")
    with pytest.raises(TilefishTypeError, match="must be a torch tensor"):
        tilefish.run(layernorm, x, [1.0], b, mode="reference")


def test_an_inout_tensor_cannot_alias_another_argument():
    x = randn((64,))
    with pytest.raises(TilefishPreconditionError, match="written in place but overlaps") as e:
        tilefish.run(stores_twice, x, x, randn((64,)), mode="reference")
    assert e.value.kind == "alias"


S = tilefish.Dim("S", buckets=[32, 64])


@tilefish.kernel
def bucketed(x: Tensor[M, S, f32], w: Tensor[S, f32]) -> Tensor[M, S, f32]:
    return x * w


def test_reference_has_no_binaries_so_any_size_runs_but_a_bucketed_dimension_is_still_a_type():
    x, w = randn((3, 50)), randn((50,), seed=1)
    assert same_bytes(tilefish.run(bucketed, x, w, mode="reference"), x * w)  # not a bucket, no binary needed
