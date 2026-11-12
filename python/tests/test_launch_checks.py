"""The checks that must pass before a binary may run (``tilefish._launch``), on fake tensors.

No torch and no GPU: a ``FakeTensor`` has the facts the checks read (shape, dtype, address, contiguity,
device). The same checks run on real tensors in the gpu tests, through ``tilefish.run``.
"""

import dataclasses
import math
from dataclasses import dataclass

import pytest
from lower_cases import D, M, T, gelu_exact, layernorm, swiglu

import tilefish
from tilefish import (
    Dim,
    Out,
    Tensor,
    TilefishPreconditionError,
    TilefishRuntimeError,
    TilefishTypeError,
    bf16,
    f32,
)
from tilefish._launch import (
    INT32_MAX,
    Call,
    bind_call,
    check_consts,
    check_dim_constraints,
    check_i32,
    check_no_alias,
    check_spec_preconditions,
    dims_by_name,
    eval_dim,
    eval_grid,
    select_spec,
    solve_dims,
    tensor_info,
)
from tilefish.aot import Precondition, compile_specs

BF16, F32 = "torch.bfloat16", "torch.float32"
ITEMSIZE = {BF16: 2, F32: 4}


@dataclass
class FakeDtype:
    name: str

    def __str__(self):
        return self.name


class FakeTensor:
    """What the checks read off a torch tensor."""

    def __init__(self, shape, dtype=BF16, ptr=0x7F0000000000, contiguous=True, device="cuda:0"):
        self.shape = tuple(shape)
        self.dtype = FakeDtype(dtype)
        self._ptr = ptr
        self._contiguous = contiguous
        self.device = device

    def data_ptr(self):
        return self._ptr

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return ITEMSIZE[str(self.dtype)]


def ln_args(rows=5, d=768, **kw):
    return (FakeTensor((rows, d), BF16, **kw), FakeTensor((d,), BF16, ptr=0x7F0000100000), FakeTensor((d,), BF16, ptr=0x7F0000200000))


# -- binding a call ----------------------------------------------------------------------------------------------------------


def test_bind_call_matches_arguments_and_binds_the_dimensions():
    call = bind_call("layernorm", layernorm.signature, ln_args(5), {})
    assert isinstance(call, Call)
    assert dims_by_name(call.dims) == {"M": 5}
    assert set(call.tensors) == {"x", "w", "b"} and call.consts == {}
    assert call.infos["x"].shape == (5, 768) and call.infos["x"].dtype == BF16


def test_bind_call_accepts_keywords_and_records_explicit_consts():
    x, w, b = ln_args(3)
    call = bind_call("layernorm", layernorm.signature, (x,), {"b": b, "w": w, "eps": 1e-5})
    assert call.consts == {"eps": 1e-5} and dims_by_name(call.dims) == {"M": 3}


def test_bind_call_is_a_signature_error_for_the_wrong_arguments():
    x, w, b = ln_args()
    with pytest.raises(TilefishTypeError, match="layernorm"):
        bind_call("layernorm", layernorm.signature, (x, w), {})
    with pytest.raises(TilefishTypeError, match="layernorm"):
        bind_call("layernorm", layernorm.signature, (x, w, b, 1e-5, 7), {})
    with pytest.raises(TilefishTypeError, match="layernorm"):
        bind_call("layernorm", layernorm.signature, (x, w, b), {"nope": 1})


def test_bind_call_rejects_something_that_is_not_a_tensor():
    x, w, b = ln_args()
    with pytest.raises(TilefishTypeError, match="argument 'w' must be a torch tensor"):
        bind_call("layernorm", layernorm.signature, (x, [1.0] * 768, b), {})


def raises(kind, match):
    return pytest.raises(TilefishPreconditionError, match=match)


def test_dtype_is_checked():
    x, w, b = ln_args()
    x32 = FakeTensor((5, 768), F32)
    with raises("dtype", r"argument 'x' has dtype torch.float32, but the signature requires bf16") as e:
        bind_call("layernorm", layernorm.signature, (x32, w, b), {})
    assert e.value.kind == "dtype" and e.value.arg == "x"


def test_rank_is_checked():
    x, w, b = ln_args()
    with raises("rank", "rank 1") as e:
        bind_call("layernorm", layernorm.signature, (FakeTensor((768,)), w, b), {})
    assert e.value.kind == "rank"


def test_a_static_size_is_checked():
    _, w, b = ln_args()
    with raises("static_size", r"dimension 1 is 700, but the signature requires 768") as e:
        bind_call("layernorm", layernorm.signature, (FakeTensor((5, 700)), w, b), {})
    assert e.value.kind == "static_size" and e.value.arg == "x"


def test_a_dimension_must_agree_between_tensors():
    gate = FakeTensor((4, 3072))
    with raises("shape", r"dimension 0 is 5, but the signature requires M = 4") as e:
        bind_call("swiglu", swiglu.signature, (gate, FakeTensor((5, 3072))), {})
    assert e.value.kind == "shape"


def test_contiguity_is_checked():
    x, w, b = ln_args()
    with raises("contiguous", "not contiguous") as e:
        bind_call("layernorm", layernorm.signature, (FakeTensor((5, 768), contiguous=False), w, b), {})
    assert e.value.kind == "contiguous"


def test_an_empty_tensor_is_rejected():
    _, w, b = ln_args()
    with raises("shape", "empty") as e:
        bind_call("layernorm", layernorm.signature, (FakeTensor((0, 768)), w, b), {})
    assert e.value.kind == "shape"


def test_tensor_info_reads_the_facts():
    t = FakeTensor((2, 3), F32, ptr=64)
    info = tensor_info("t", t)
    assert (info.shape, info.dtype, info.ptr, info.contiguous, info.device, info.nbytes) == ((2, 3), F32, 64, True, "cuda:0", 24)


# -- solving dimensions ---------------------------------------------------------------------------------------------------------

K = Dim("K")
B = Dim("B", buckets=[16, 32])


def solve(pattern, actual):
    return solve_dims("op", [("x", 0, pattern, actual, "Tensor[...]")])


def test_solve_binds_a_plain_dimension():
    assert solve(K, 7) == {K: 7}


def test_solve_inverts_an_affine_expression():
    assert solve(2 * K + 1, 11) == {K: 5}
    assert solve(3072 * K, 6144) == {K: 2}


@pytest.mark.parametrize("pattern, actual", [(2 * K + 1, 10), (3072 * K, 100), (2 * K, 0), (20 - K, 25)])
def test_solve_rejects_what_has_no_whole_solution(pattern, actual):
    with pytest.raises(TilefishPreconditionError, match="cannot match"):
        solve(pattern, actual)


def test_solve_is_ambiguous_when_the_unknown_is_only_inside_a_cdiv():
    with pytest.raises(TilefishTypeError, match="ambiguous"):
        solve(tilefish.cdiv(K, 4), 5)


def test_solve_checks_every_equation_once_bound():
    eqs = [("a", 0, K, 4, "t"), ("b", 1, 2 * K, 9, "t")]
    with pytest.raises(TilefishPreconditionError, match="requires 2\\*K = 8"):
        solve_dims("op", eqs)
    assert solve_dims("op", [("a", 0, 2 * K, 8, "t"), ("b", 1, K, 4, "t")]) == {K: 4}


def test_eval_dim():
    assert eval_dim(2 * K + 1, {K: 4}) == 9
    assert eval_dim(2 * K + 1, {}) is None
    assert eval_dim(tilefish.cdiv(K, 1024), {K: 2049}) == 3
    assert eval_dim(D, {}) == 768  # a static dimension is its value


def test_dimension_constraints():
    check_dim_constraints({T: 32, B: 16})
    with pytest.raises(TilefishPreconditionError, match="declared divisible_by=16") as e:
        check_dim_constraints({T: 40})
    assert e.value.kind == "divisible"
    with pytest.raises(TilefishPreconditionError, match="not one of its buckets") as e:
        check_dim_constraints({B: 20})
    assert e.value.kind == "bucket" and "no padding" in str(e.value)
    check_dim_constraints({B: 20}, buckets=False)  # reference mode has no binaries to be exact about


# -- choosing the binary ------------------------------------------------------------------------------------------------------

S = Dim("S", buckets=[32, 64, 128])


@tilefish.kernel
def rowwise(x: Tensor[M, S, f32], w: Tensor[S, f32]) -> Tensor[M, S, f32]:
    return x * w


ROWWISE = rowwise.lower()


def test_each_bucket_selects_its_own_binary():
    specs = ROWWISE.specs()
    assert [s.stem for s in specs] == ["rowwise.S32", "rowwise.S64", "rowwise.S128"]
    for size in (32, 64, 128):
        call = bind_call("rowwise", rowwise.signature, (FakeTensor((3, size), F32), FakeTensor((size,), F32)), {})
        assert select_spec("rowwise", specs, call.dims).buckets == {"S": size}


def test_a_size_between_buckets_has_no_binary_and_is_not_padded():
    specs = ROWWISE.specs()
    call = bind_call("rowwise", rowwise.signature, (FakeTensor((3, 50), F32), FakeTensor((50,), F32)), {})
    with pytest.raises(TilefishPreconditionError, match=r"no padding") as e:
        select_spec("rowwise", specs, call.dims)
    assert e.value.kind == "bucket" and e.value.arg == "S"
    assert "[32, 64, 128]" in str(e.value) and "50" in str(e.value)


def test_an_unbucketed_op_has_one_binary():
    specs = compile_specs(layernorm.lower().extern)
    call = bind_call("layernorm", layernorm.signature, ln_args(5), {})
    assert select_spec("layernorm", specs, call.dims) is specs[0]


# -- the preconditions of a spec ---------------------------------------------------------------------------------------------


def gelu_case(t_size=64, **kw):
    lowered = gelu_exact.lower()
    (spec,) = lowered.specs()
    x = FakeTensor((t_size,), BF16, **kw)
    out = FakeTensor((t_size,), BF16, ptr=0x7F0000100000)
    call = bind_call("gelu_exact", gelu_exact.signature, (x,), {})
    infos = {"x": tensor_info("x", x), "out": tensor_info("out", out)}
    return spec, infos, call


def test_a_good_call_passes_every_precondition():
    spec, infos, call = gelu_case(64)
    check_spec_preconditions(spec, infos, {"n": 64}, call.dims)
    assert {p.kind for p in spec.preconditions} == {"pointer_alignment", "divisible", "max_elements"}


def test_pointer_alignment_is_checked():
    spec, infos, call = gelu_case(64, ptr=0x7F0000000008)
    with pytest.raises(TilefishPreconditionError, match="not aligned to 16 bytes") as e:
        check_spec_preconditions(spec, infos, {"n": 64}, call.dims)
    assert e.value.kind == "pointer_alignment" and e.value.arg == "x"
    check_spec_preconditions(spec, infos, {"n": 64}, call.dims, alignment=False)


def test_divisibility_of_a_runtime_scalar_is_checked():
    spec, infos, call = gelu_case(64)
    with pytest.raises(TilefishPreconditionError, match="not a multiple of 16") as e:
        check_spec_preconditions(spec, infos, {"n": 60}, call.dims)
    assert e.value.kind == "divisible" and e.value.arg == "n"


def test_a_dimension_declared_divisible_is_checked_on_the_tensor_shape():
    with pytest.raises(TilefishPreconditionError, match="divisible_by=16"):
        call = bind_call("gelu_exact", gelu_exact.signature, (FakeTensor((60,), BF16),), {})
        check_dim_constraints(call.dims)


def test_static_sizes_are_checked_again_against_the_spec():
    lowered = layernorm.lower()
    (spec,) = lowered.specs()
    x, w, b = ln_args(5)
    out = FakeTensor((5, 768), BF16, ptr=0x7F0000300000)
    infos = {"x": tensor_info("x", x), "w": tensor_info("w", w), "b": tensor_info("b", b), "out": tensor_info("out", out)}
    call = bind_call("layernorm", layernorm.signature, (x, w, b), {})
    check_spec_preconditions(spec, infos, {}, call.dims)
    bad = dict(infos, w=tensor_info("w", FakeTensor((700,), BF16)))
    with pytest.raises(TilefishPreconditionError, match=r"dimension 0 is 700, but 'layernorm' is built for 768") as e:
        check_spec_preconditions(spec, bad, {}, call.dims)
    assert e.value.kind == "static_size"


def test_the_bucket_of_the_spec_must_be_the_actual_size():
    spec = ROWWISE.specs()[1]  # S = 64
    call = bind_call("rowwise", rowwise.signature, (FakeTensor((3, 32), F32), FakeTensor((32,), F32)), {})
    infos = {n: tensor_info(n, t) for n, t in {"x": FakeTensor((3, 32), F32), "w": FakeTensor((32,), F32), "out": FakeTensor((3, 32), F32, ptr=0x7F0000300000)}.items()}
    with pytest.raises(TilefishPreconditionError, match="built for exactly 64") as e:
        check_spec_preconditions(spec, infos, {}, call.dims)
    assert e.value.kind == "bucket"
    check_spec_preconditions(spec, infos, {}, call.dims, buckets=False)


def test_max_elements_bounds_a_tensor_without_a_static_size():
    spec, infos, call = gelu_case(64)
    big = dict(infos, x=tensor_info("x", FakeTensor((INT32_MAX + 1,), BF16)))
    with pytest.raises(TilefishPreconditionError, match="more than the 2147483647 that the i32 offsets") as e:
        check_spec_preconditions(spec, big, {"n": 64}, call.dims)
    assert e.value.kind == "max_elements" and e.value.arg == "x"
    edge = dict(infos, x=tensor_info("x", FakeTensor((INT32_MAX,), BF16)))
    check_spec_preconditions(spec, edge, {"n": 64}, call.dims)


def test_a_spec_with_a_precondition_it_does_not_know_does_not_pass_silently():
    spec, infos, call = gelu_case(64)
    future = dataclasses.replace(spec, preconditions=(*spec.preconditions, Precondition("monotone", arg="x")))
    with pytest.raises(TilefishRuntimeError, match="precondition of kind 'monotone'"):
        check_spec_preconditions(future, infos, {"n": 64}, call.dims)


def test_runtime_scalars_must_fit_an_i32():
    check_i32({"n": INT32_MAX})
    for bad in (INT32_MAX + 1, -1):
        with pytest.raises(TilefishPreconditionError, match="32-bit"):
            check_i32({"n": bad})


def test_a_tensor_written_in_place_must_not_overlap_another_argument():
    a = tensor_info("a", FakeTensor((16,), F32, ptr=0x1000))  # 64 bytes: 0x1000..0x1040
    overlapping = tensor_info("b", FakeTensor((16,), F32, ptr=0x1020))
    apart = tensor_info("c", FakeTensor((16,), F32, ptr=0x1040))
    check_no_alias({"a": a, "c": apart}, {"a"})
    check_no_alias({"a": a, "b": overlapping}, set())  # nothing is written: reading twice is fine
    with pytest.raises(TilefishPreconditionError, match="written in place but overlaps argument 'b'") as e:
        check_no_alias({"a": a, "b": overlapping}, {"a"})
    assert e.value.kind == "alias"
    with pytest.raises(TilefishPreconditionError, match="argument 'b' is written in place but overlaps argument 'a'"):
        check_no_alias({"a": a, "b": overlapping}, {"b"})


def test_a_constexpr_can_be_restated_but_not_changed():
    (spec,) = layernorm.lower().specs()
    check_consts(spec, {})
    check_consts(spec, {"eps": 1e-5})
    check_consts(spec, {"eps": 1e-5, "unrelated": 3})
    with pytest.raises(TilefishTypeError, match="compiled into 'layernorm' as 1e-05; it cannot be changed to 0.001"):
        check_consts(spec, {"eps": 1e-3})
    with pytest.raises(TilefishTypeError, match="cannot be changed"):
        check_consts(spec, {"eps": True})


def test_the_grid_is_the_declared_one_under_the_actual_dimensions():
    lowered = swiglu.lower()
    (spec,) = lowered.specs()
    call = bind_call("swiglu", swiglu.signature, (FakeTensor((7, 3072)), FakeTensor((7, 3072))), {})
    n_dim = next(p.dim for p in lowered.extern.signature.params if p.kind == "dim")
    assert eval_grid(spec, lowered.extern.grid, {**call.dims, **({n_dim: 21504} if isinstance(n_dim, Dim) else {})}) == (21,)
    rows = layernorm.lower()
    (rspec,) = rows.specs()
    assert eval_grid(rspec, rows.extern.grid, {M: 9}) == (9,)
    with pytest.raises(TilefishRuntimeError, match="no argument determines"):
        eval_grid(rspec, rows.extern.grid, {})


def test_count_of_says_where_the_element_count_comes_from():
    lowered = swiglu.lower()
    (spec,) = lowered.specs()
    scalar = next(a for a in spec.args if a.kind == "scalar")
    assert scalar.count_of == "gate" and scalar.dim == "3072*M"


# -- run(): what is refused before anything is imported -----------------------------------------------------------------------

n = Dim("n", divisible_by=16)


@tilefish.library("gemm", acc=f32)
def gemm(a: Tensor[M, 16, bf16], b: Tensor[16, 16, bf16]) -> Tensor[M, 16, bf16]: ...


@tilefish.extern(triton="my_kernels:copy", grid=lambda n: (n,))
def copy(x: Tensor[n, bf16], y: Out[Tensor[n, bf16]], n_elem: n): ...


def test_run_refuses_an_unknown_mode():
    with pytest.raises(TilefishRuntimeError, match="unknown mode 'fast'"):
        tilefish.run(layernorm, mode="fast")
    with pytest.raises(TilefishRuntimeError, match="unknown mode"):
        layernorm.run(mode="JIT")


def test_run_refuses_operators_that_have_no_runtime():
    with pytest.raises(TilefishRuntimeError, match="library op 'gemm'"):
        tilefish.run(gemm)
    with pytest.raises(TilefishRuntimeError, match="takes a kernel or an extern op"):
        tilefish.run(lambda x: x)


def test_reference_of_an_extern_says_there_is_no_ir():
    with pytest.raises(TilefishRuntimeError, match="extern op 'copy' wraps a hand-written Triton kernel and has no IR"):
        tilefish.run(copy, mode="reference")
    with pytest.raises(TilefishRuntimeError, match="no IR"):
        copy.run(mode="reference")


def test_kernels_externs_and_lowered_kernels_have_run():
    assert callable(layernorm.run) and callable(copy.run) and callable(layernorm.lower().run)
    assert tilefish.run is tilefish.runtime.run
