"""The rules of the lowering templates, the numerics table, and every rejected pattern."""

import ast
import pickle

import pytest

import tilefish
from tilefish import (
    BlockScaled,
    Dim,
    InOut,
    Out,
    Tensor,
    TilefishDeclarationError,
    TilefishLoweringError,
    bf16,
    const,
    e2m1,
    e4m3,
    f16,
    f32,
    i32,
    u8,
    ue8m0,
)
from tilefish.aot import compile_specs

M = Dim("M")
D = Dim("D", 256)
N = Dim("N")


def body(lowered):
    """The statements of the generated function after the signature, stripped."""
    lines = lowered.source.splitlines()
    start = lines.index("):") + 1
    return [line.strip() for line in lines[start:]]


def locals_of(lowered):
    return [line for line in body(lowered) if " = " in line.split("#")[0]]


def fails(kernel_op, rule, *, template=None, contains=()):
    with pytest.raises(TilefishLoweringError) as exc:
        kernel_op.lower(template=template)
    assert exc.value.rule == rule, str(exc.value)
    assert f"rule '{rule}'" in str(exc.value)
    for text in contains:
        assert text in str(exc.value), str(exc.value)
    return exc.value


# -- the numerics table --------------------------------------------------------------------------------------


def test_elementwise_ops_and_scalars_follow_the_table():
    @tilefish.kernel
    def ops(x: Tensor[M, 64, f32], y: Tensor[M, 64, f32], *, k: const[f32] = 0.1) -> Tensor[M, 64, f32]:
        a = tilefish.sqrt(x * 0.5 + 1e-6)
        b = tilefish.exp(y - a)
        c = 2.0 - b
        d = c / (x + k)
        e = tilefish.silu(d) + tilefish.gelu(d)
        return 1.0 / e

    lowered = ops.lower()
    assert lowered.template == "elementwise"
    assert body(lowered)[3:] == [
        "v0 = tl.load(x + offs, mask=mask, other=0)  # x",
        "v1 = tl.load(y + offs, mask=mask, other=0)  # y",
        "v2 = v0 * 0.5",
        "v3 = v2 + 1e-06",
        "v4 = tl.sqrt_rn(v3)",
        "v5 = v1 - v4",
        "v6 = tl.exp(v5)",
        "v7 = 2.0 - v6",
        "v8 = v0 + k",
        "v9 = v7 / v8",
        "v10 = v9 * tl.sigmoid(v9)",
        "v11 = 0.5 * v9 * (1.0 + tl.math.erf(v9 * 0.7071067811865476))",
        "v12 = v10 + v11",
        "v13 = 1.0 / v12",
        "tl.store(out + offs, v13, mask=mask)  # return value 0",
    ]
    assert "# rounding point" not in lowered.source  # no cast, so no rounding point


def test_float_literals_round_trip_and_non_finite_ones_are_written_as_calls():
    @tilefish.kernel
    def lits(x: Tensor[M, f32]) -> Tensor[M, f32]:
        return x * 0.1 + 1e-6 * x + 3 - float("inf") * x + float("-inf") + 1e300 * x

    assert body(lits.lower())[3:] == [
        "v0 = tl.load(x + offs, mask=mask, other=0)  # x",
        "v1 = v0 * 0.1",
        "v2 = 1e-06 * v0",
        "v3 = v1 + v2",
        "v4 = v3 + 3.0",  # an int literal on a float tensor is the float 3.0
        'v5 = float("inf") * v0',
        "v6 = v4 - v5",
        'v7 = v6 + float("-inf")',
        "v8 = 1e+300 * v0",
        "v9 = v7 + v8",
        "tl.store(out + offs, v9, mask=mask)  # return value 0",
    ]
    for value in (0.1, 1e-6, 1e300, 1 / 3, 5e-324, 1.7976931348623157e308):
        assert float(repr(value)) == value  # the spelling the generator uses is exact


def test_integer_tensors_get_integer_literals():
    @tilefish.kernel
    def ints(x: Tensor[M, i32]) -> Tensor[M, i32]:
        return x * 3 - 1

    assert body(ints.lower())[4:7] == ["v1 = v0 * 3", "v2 = v1 - 1", "tl.store(out + offs, v2, mask=mask)  # return value 0"]


def test_reduction_lines():
    @tilefish.kernel
    def red(x: Tensor[M, 100, f32]) -> tuple[Tensor[M, f32], Tensor[M, 1, f32], Tensor[M, 1, f32]]:
        return tilefish.sum(x, axis=-1), tilefish.mean(x, axis=-1, keepdims=True), tilefish.max(x, axis=-1, keepdims=True)

    lowered = red.lower()
    assert lowered.template == "row"
    src = body(lowered)
    assert 'v1 = tl.sum(tl.where(mask, v0, 0.0), axis=0)' in src
    assert 'v2 = tl.sum(tl.where(mask, v0, 0.0), axis=0) / D' in src
    assert 'v3 = tl.max(tl.where(mask, v0, float("-inf")), axis=0)' in src
    # per-row results are scalars: stored at `row`, with no mask
    assert "tl.store(out0 + row, v1)  # return value 0" in src
    assert "tl.store(out1 + row, v2)  # return value 1" in src
    assert "tl.store(out2 + row, v3)  # return value 2" in src
    assert [p.name for p in lowered.extern.signature.params] == ["x", "out0", "out1", "out2", "D", "BLOCK_D"]
    (spec,) = compile_specs(lowered.extern)
    assert spec.constants == {(4,): 100, (5,): 128}


def test_a_per_row_input_is_loaded_without_a_mask():
    @tilefish.kernel
    def scale_rows(x: Tensor[M, D, f32], s: Tensor[M, 1, f32]) -> Tensor[M, D, f32]:
        return x * s

    src = body(scale_rows.lower())
    assert "v1 = tl.load(s + row)  # s" in src
    assert "v2 = v0 * v1" in src


def test_rounding_points_are_marked_and_only_they_are():
    @tilefish.kernel
    def casts(x: Tensor[M, D, bf16], y: Tensor[M, D, f32]) -> tuple[Tensor[M, D, bf16], Tensor[M, D, f32], Tensor[M, D, f16]]:
        w = x.to(f32)  # widening: not a rounding point
        return (w * y).to(bf16), w * y, (w + y).to(f16)

    lowered = casts.lower()
    assert len(casts.graph.rounding_points()) == 2
    src = body(lowered)
    assert "v2 = v0.to(tl.float32)" in src  # no comment: widening is exact
    assert "v4 = v3.to(tl.bfloat16)  # rounding point" in src
    assert "v7 = v6.to(tl.float16)  # rounding point" in src
    assert [line for line in src if "rounding point" in line] == [
        "v4 = v3.to(tl.bfloat16)  # rounding point",
        "v7 = v6.to(tl.float16)  # rounding point",
    ]
    assert len([line for line in src if ".to(" in line]) == 3


# -- InOut and Out parameters ---------------------------------------------------------------------------------


def test_out_parameter_is_written_once_at_its_store():
    @tilefish.kernel
    def scale_into(x: Tensor[M, D, f32], y: Out[Tensor[M, D, f32]], *, k: const[f32] = 2.0):
        y[...] = x * k

    lowered = scale_into.lower()
    assert lowered.template == "elementwise"
    assert "tl.store(y + offs, v1, mask=mask)  # Out y, written once" in body(lowered)
    assert [p.name for p in lowered.extern.signature.params] == ["x", "y", "k", "n", "BLOCK"]
    assert lowered.extern.signature.params[1].access == "out"
    assert not [p for p in lowered.extern.signature.params if p.name.startswith("out")]  # nothing is returned


def test_inout_parameter_is_written_with_its_final_version_only():
    @tilefish.kernel
    def twice(x: InOut[Tensor[M, D, f32]]):
        a = x * 2.0
        x[...] = a
        b = x + 1.0  # reads the version just stored
        x[...] = b

    src = body(twice.lower())
    assert src.count("v0 = tl.load(x + offs, mask=mask, other=0)  # x") == 1
    assert "# InOut x: superseded by a later store" in src
    stores = [line for line in src if line.startswith("tl.store")]
    assert stores == ["tl.store(x + offs, v2, mask=mask)  # InOut x, final version"]
    assert "v1 = v0 * 2.0" in src and "v2 = v1 + 1.0" in src  # the second read is the first store's value, not a reload


def test_returning_an_inout_parameter_copies_the_final_version_to_the_output():
    @tilefish.kernel
    def bump(x: InOut[Tensor[M, D, f32]]) -> Tensor[M, D, f32]:
        x[...] = x + 1.0
        return x

    src = body(bump.lower())
    assert "tl.store(x + offs, v1, mask=mask)  # InOut x, final version" in src
    assert "tl.store(out + offs, v1, mask=mask)  # return value 0" in src


def test_returning_an_input_loads_and_copies_it():
    @tilefish.kernel
    def identity(x: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return x

    assert body(identity.lower())[-2:] == [
        "v0 = tl.load(x + offs, mask=mask, other=0)  # x",
        "tl.store(out + offs, v0, mask=mask)  # return value 0",
    ]


def test_unread_parameters_are_not_loaded_and_unwritten_ones_are_not_stored():
    @tilefish.kernel
    def first_only(x: Tensor[M, D, f32], unused: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return x + 1.0

    src = body(first_only.lower())
    assert not any("unused" in line and "tl.load" in line for line in src)
    assert [p.name for p in first_only.lower().extern.signature.params][:2] == ["x", "unused"]


# -- row template ------------------------------------------------------------------------------------------------------


def test_row_template_addressing():
    @tilefish.kernel
    def addressing(x: Tensor[M, D, f32], w: Tensor[D, f32], s: Tensor[M, 1, f32]) -> Tensor[M, D, f32]:
        return x * w + s

    lowered = addressing.lower()
    assert lowered.template == "row"
    assert body(lowered)[:4] == [
        "row = tl.program_id(0)",
        "cols = tl.arange(0, BLOCK_D)",
        "mask = cols < D",
        "offs = row * D + cols",
    ]
    src = body(lowered)
    assert "v0 = tl.load(x + offs, mask=mask, other=0)  # x" in src
    assert "v1 = tl.load(w + cols, mask=mask, other=0)  # w" in src
    assert "v2 = tl.load(s + row)  # s" in src
    assert lowered.grid == "(M,)"


@pytest.mark.parametrize(
    "dim, block, warps",
    [(1, 1, 1), (64, 64, 1), (100, 128, 1), (256, 256, 1), (257, 512, 2), (2560, 4096, 16), (16384, 16384, 16), (70000, 131072, 16)],
)
def test_row_num_warps_rule(dim, block, warps):
    @tilefish.kernel
    def k(x: Tensor[M, Dim("D", dim), f32]) -> Tensor[M, Dim("D", dim), f32]:
        return x + 1.0

    lowered = k.lower(template="row")
    assert lowered.num_warps == warps
    (spec,) = compile_specs(lowered.extern)
    assert spec.num_warps == warps and spec.constants == {(2,): dim, (3,): block}


def test_num_warps_can_be_overridden_by_the_decorator_and_by_lower():
    @tilefish.kernel(num_warps=2)
    def k(x: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return x + 1.0

    assert k.num_warps == 2
    for template in ("row", "elementwise"):
        assert k.lower(template=template).num_warps == 2
    assert compile_specs(k.lower().extern)[0].block == (64, 1, 1)
    assert k.lower(num_warps=8).num_warps == 8  # lower() wins over the decorator

    @tilefish.kernel
    def plain(x: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return x + 1.0

    assert plain.num_warps is None
    assert plain.lower(template="row").num_warps == 1  # D = 256: one warp
    assert plain.lower(template="elementwise").num_warps == 4
    assert plain.lower(num_warps=32).num_warps == 32


def test_bucketed_d_gives_one_spec_per_bucket():
    Hb = Dim("Hb", buckets=[256, 1000, 4096])

    @tilefish.kernel
    def rms(x: Tensor[M, Hb, f32], w: Tensor[Hb, f32]) -> Tensor[M, Hb, f32]:
        r = tilefish.rsqrt(tilefish.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        return x * r * w

    lowered = rms.lower()
    assert lowered.template == "row"
    specs = compile_specs(lowered.extern)
    assert [s.stem for s in specs] == ["rms.Hb256", "rms.Hb1000", "rms.Hb4096"]
    assert [s.constants for s in specs] == [{(3,): 256, (4,): 256}, {(3,): 1000, (4,): 1024}, {(3,): 4096, (4,): 4096}]
    # one source for all buckets; one num_warps for the set, from the largest bucket (4096 lanes: 16 warps)
    assert {s.num_warps for s in specs} == {16} and lowered.num_warps == 16
    assert [s.buckets for s in specs] == [{"Hb": 256}, {"Hb": 1000}, {"Hb": 4096}]
    assert [a.shape for a in specs[1].args if a.kind == "pointer"] == [("M", "1000"), ("1000",), ("M", "1000")]
    assert [(p.kind, p.dim, p.value) for p in specs[1].preconditions if p.kind == "bucket"] == [("bucket", "Hb", 1000)]
    assert "(bucketed)" in lowered.source.splitlines()[2]


def test_d_may_be_an_expression_of_a_bucketed_dim():
    F = Dim("F", buckets=[96, 128])

    @tilefish.kernel
    def wide(x: Tensor[M, 2 * F, f32]) -> Tensor[M, 2 * F, f32]:
        return x + 1.0

    specs = compile_specs(wide.lower(template="row").extern)
    assert [s.constants for s in specs] == [{(2,): 192, (3,): 256}, {(2,): 256, (3,): 256}]


def test_row_dim_may_be_static_or_an_expression():
    R = Dim("R", 8)

    @tilefish.kernel
    def fixed(x: Tensor[R, D, f32]) -> Tensor[R, D, f32]:
        return x + 1.0

    assert fixed.lower(template="row").grid == "(8,)"

    @tilefish.kernel
    def double(x: Tensor[2 * M, D, f32]) -> Tensor[2 * M, D, f32]:
        return x + 1.0

    assert double.lower(template="row").grid == "(2*M,)"


# -- rejected patterns ---------------------------------------------------------------------------------------


def test_broadcast_is_rejected_by_the_elementwise_template():
    @tilefish.kernel
    def bcast(x: Tensor[M, D, f32], w: Tensor[D, f32]) -> Tensor[M, D, f32]:
        return x * w

    err = fails(bcast, "elementwise.same-shape", template="elementwise", contains=["does not broadcast", "parameter 'w'"])
    assert "kernel 'bcast' with the elementwise template" in str(err)
    assert bcast.lower().template == "row"  # without a forced template, the row template takes it
    # a scalar is not a broadcast
    @tilefish.kernel
    def scalar(x: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return x * 2.0

    assert scalar.lower(template="elementwise").template == "elementwise"


def test_row_vector_and_per_row_values_do_not_fit_elementwise():
    @tilefish.kernel
    def per_row(x: Tensor[M, D, f32], s: Tensor[M, 1, f32]) -> Tensor[M, D, f32]:
        return x * s

    fails(per_row, "elementwise.same-shape", template="elementwise")


def test_reduction_over_a_non_last_axis_is_rejected():
    @tilefish.kernel
    def colsum(x: Tensor[M, D, f32]) -> Tensor[D, f32]:
        return tilefish.sum(x, axis=0)

    err = fails(colsum, "row.last-axis-reduction", template="row", contains=["axes (0,)", "last axis only"])
    assert "kernel 'colsum' with the row template" in str(err)
    # without a forced template both templates are tried and both are named
    with pytest.raises(TilefishLoweringError) as exc:
        colsum.lower()
    assert exc.value.rule == "no-template"
    assert [c.rule for c in exc.value.causes] == ["elementwise.reduction", "row.last-axis-reduction"]
    assert "row.last-axis-reduction" in str(exc.value) and "elementwise.reduction" in str(exc.value)


def test_reduction_over_all_axes_is_rejected():
    @tilefish.kernel
    def total(x: Tensor[M, D, f32]) -> Tensor[f32]:
        return tilefish.sum(x)

    fails(total, "row.last-axis-reduction", template="row", contains=["axes (0, 1)"])


def test_reduction_without_a_row_axis_is_rejected():
    @tilefish.kernel
    def vec(w: Tensor[D, f32], x: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return x * tilefish.sum(w, axis=0)  # a rank-0 value: M1 broadcasts it, the row template has no home for it

    fails(vec, "row.reduction-input", template="row", contains=["no row axis"])

    @tilefish.kernel
    def per_row_sum(x: Tensor[M, 1, f32], y: Tensor[M, D, f32]) -> Tensor[M, 1, f32]:
        return tilefish.sum(x, axis=-1, keepdims=True)  # reduces the size-1 axis of a per-row tensor

    fails(per_row_sum, "row.reduction-input", template="row")


def test_dynamic_d_is_rejected():
    @tilefish.kernel
    def dyn(x: Tensor[M, N, f32]) -> Tensor[M, N, f32]:
        return x + 1.0

    err = fails(dyn, "row.dynamic-d", template="row", contains=["dynamic in N", "static", "bucketed"])
    assert "kernel 'dyn' with the row template" in str(err)
    # elementwise takes a dynamic last axis fine (n is a runtime scalar): auto mode succeeds
    assert dyn.lower().template == "elementwise"

    @tilefish.kernel
    def dyn_rms(x: Tensor[M, N, f32]) -> Tensor[M, N, f32]:
        return x / tilefish.sum(x, axis=-1, keepdims=True)

    with pytest.raises(TilefishLoweringError) as exc:
        dyn_rms.lower()
    assert exc.value.rule == "no-template" and "row.dynamic-d" in str(exc.value)

    constrained = Dim("K", divisible_by=32)

    @tilefish.kernel
    def divisible_only(x: Tensor[M, constrained, f32]) -> Tensor[M, constrained, f32]:
        return x + 1.0

    fails(divisible_only, "row.dynamic-d", template="row")  # divisible_by is not a size


def test_rank_three_is_rejected_by_the_row_template():
    B = Dim("B")

    @tilefish.kernel
    def r3(x: Tensor[B, M, D, f32]) -> Tensor[B, M, D, f32]:
        return x + 1.0

    err = fails(r3, "row.rank", template="row", contains=["rank 3", "Tensor[B, M, D, f32]"])
    assert "kernel 'r3' with the row template" in str(err)
    ew = r3.lower()  # the elementwise template flattens any rank
    assert ew.template == "elementwise" and ew.grid == "(cdiv(n, 1024),)"
    assert "# template: elementwise; every tensor is [B, M, D]" in ew.source

    @tilefish.kernel
    def r3_sum(x: Tensor[B, M, D, f32]) -> Tensor[B, M, f32]:
        return tilefish.sum(x, axis=-1)

    fails(r3_sum, "row.rank", template="row")


def test_a_shape_outside_the_row_vocabulary_is_rejected():
    @tilefish.kernel
    def mismatched(x: Tensor[M, D, f32], y: Tensor[N, D, f32]) -> Tensor[M, D, f32]:
        return x + 1.0

    fails(mismatched, "row.shape", template="row", contains=["Tensor[N, D, f32]"])


def test_rank_one_is_ambiguous_when_the_row_and_column_sizes_are_the_same():
    S = Dim("S", 64)

    @tilefish.kernel
    def square(x: Tensor[S, S, f32], w: Tensor[S, f32]) -> Tensor[S, S, f32]:
        return x * w

    fails(square, "row.ambiguous", template="row", contains=["[D] or [R]"])


def test_writing_a_column_vector_from_every_row_is_rejected():
    @tilefish.kernel
    def clobber(x: Tensor[M, D, f32], w: InOut[Tensor[D, f32]]):
        w[...] = w + 1.0

    fails(clobber, "row.col-output", template="row", contains=["race"])

    @tilefish.kernel
    def col_result(x: Tensor[M, D, f32], w: Tensor[D, f32]) -> Tensor[D, f32]:
        return w * 2.0

    fails(col_result, "row.col-output", template="row", contains=["returned"])


def test_a_view_input_is_rejected():
    @tilefish.kernel
    def on_view(x: Tensor[M, D, f32].as_view()) -> Tensor[M, D, f32]:
        return x + 1.0

    err = fails(on_view, "view-input", contains=["parameter 'x'", "View[M, D, f32]"])
    assert "kernel 'on_view'" in str(err)
    fails(on_view, "view-input", template="row")
    fails(on_view, "view-input", template="elementwise")


def test_split_chunks_are_views_and_split_is_not_lowered():
    F = Dim("F", 128)

    @tilefish.kernel
    def swiglu_fused(gu: Tensor[M, 2 * F, f32]) -> Tensor[M, F, f32]:
        g, u = gu.split(F, axis=-1)
        return tilefish.silu(g) * u

    fails(swiglu_fused, "split", contains=["views", "View[M, F, f32]"])

    @tilefish.kernel
    def swiglu_rows(gu: Tensor[2, F, f32]) -> Tensor[1, F, f32]:
        g, u = gu.split(1, axis=0)  # contiguous chunks, but still not lowered
        return g * u

    fails(swiglu_rows, "split", contains=["split is not lowered"])


def test_calls_to_declared_operators_are_not_lowered():
    @tilefish.extern(triton="my_kernels:gelu", grid=lambda N: (N,))
    def ext_gelu(x: Tensor[N, f32], y: Out[Tensor[N, f32]], n_elem: N, BLOCK: const = 1024): ...

    @tilefish.kernel
    def calls(x: Tensor[M, f32]) -> Tensor[M, f32]:
        return ext_gelu(x)

    fails(calls, "declared-op-call", contains=["extern op 'ext_gelu'"])


# -- dtype rules --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule, build",
    [
        ("division-dtype", lambda x: x / x),
        ("math-dtype", lambda x: tilefish.exp(x)),
        ("math-dtype", lambda x: tilefish.rsqrt(x)),
        ("math-dtype", lambda x: tilefish.sqrt(x)),
        ("math-dtype", lambda x: tilefish.silu(x)),
        ("math-dtype", lambda x: tilefish.gelu(x)),
    ],
)
@pytest.mark.parametrize("dtype", [bf16, f16])
def test_division_and_math_functions_are_f32_only(rule, build, dtype):
    @tilefish.kernel
    def k(x: Tensor[M, D, dtype]) -> Tensor[M, D, dtype]:
        return build(x)

    fails(k, rule, contains=[".to(f32)"])


@pytest.mark.parametrize("op, dtype", [(o, d) for o in ("sum", "mean", "max") for d in (bf16, f16)] + [("sum", i32), ("max", i32)])
def test_reductions_are_f32_only(op, dtype):
    @tilefish.kernel
    def k(x: Tensor[M, D, dtype]) -> Tensor[M, 1, dtype]:
        return getattr(tilefish, op)(x, axis=-1, keepdims=True)

    fails(k, "reduction-dtype", contains=["cast with .to(f32) first"])


def test_arithmetic_dtypes():
    @tilefish.kernel
    def fine(a: Tensor[M, D, bf16], b: Tensor[M, D, bf16]) -> Tensor[M, D, bf16]:
        return a * b + a - b

    assert fine.lower().template == "elementwise"

    @tilefish.kernel
    def bytes_(a: Tensor[M, D, u8], b: Tensor[M, D, u8]) -> Tensor[M, D, u8]:
        return a + b

    fails(bytes_, "arithmetic-dtype", contains=["u8"])


def test_casts_between_supported_dtypes_are_plain_to_calls():
    @tilefish.kernel
    def widen(a: Tensor[M, D, u8], q: Tensor[M, D, e4m3]) -> tuple[Tensor[M, D, i32], Tensor[M, D, f32], Tensor[M, D, e4m3]]:
        return a.to(i32), q.to(f32), q.to(f32).to(e4m3)

    src = body(widen.lower())
    assert "v2 = v0.to(tl.int32)" in src
    assert "v3 = v1.to(tl.float32)" in src
    assert "v5 = v4.to(tl.float8e4nv)  # rounding point" in src
    assert [p.name for p in widen.lower().extern.signature.params][:2] == ["a", "q"]
    (spec,) = compile_specs(widen.lower().extern)
    assert spec.signature["q"] == "*fp8e4nv" and spec.signature["a"] == "*u8"


def test_packed_and_block_scaled_dtypes_are_rejected():
    @tilefish.kernel
    def scales(s: Tensor[M, D, ue8m0]) -> Tensor[M, D, ue8m0]:
        return s

    fails(scales, "dtype", contains=["ue8m0"])

    @tilefish.kernel
    def packed(s: Tensor[M, D, e2m1]) -> Tensor[M, D, e2m1]:
        return s

    fails(packed, "dtype", contains=["e2m1"])

    fmt = BlockScaled(elem=e4m3, scale=ue8m0, block=32, axis=-1)

    @tilefish.kernel
    def quant(w: Tensor[M, D, fmt]) -> Tensor[M, D, fmt]:
        return w

    fails(quant, "dtype", contains=["block-scaled"])


# -- names ------------------------------------------------------------------------------------------------------


def test_generated_names_step_aside_for_kernel_parameters():
    @tilefish.kernel
    def clash(
        mask: Tensor[M, D, f32], row: Tensor[M, D, f32], v0: Tensor[M, D, f32], out: Tensor[M, D, f32], cols: const = 1
    ) -> Tensor[M, D, f32]:
        return mask + row + v0 + out

    lowered = clash.lower(template="row")
    tree = ast.parse(lowered.source)
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    args = [a.arg for a in fn.args.args]
    # the kernel's own names are untouched; the generated ones are suffixed
    assert args == ["mask", "row", "v0", "out", "cols", "out_1", "D", "BLOCK_D"]
    assert [p.name for p in lowered.extern.signature.params] == args
    src = body(lowered)
    assert src[:4] == [
        "row_1 = tl.program_id(0)",
        "cols_1 = tl.arange(0, BLOCK_D)",
        "mask_1 = cols_1 < D",
        "offs = row_1 * D + cols_1",
    ]
    assert "v0_1 = tl.load(mask + offs, mask=mask_1, other=0)  # mask" in src
    assert "tl.store(out_1 + offs, v6, mask=mask_1)  # return value 0" in src
    # locals never shadow a parameter: every assignment target is new
    assigned = [t.id for n in ast.walk(fn) for t in getattr(n, "targets", []) if isinstance(t, ast.Name)]
    assert not set(assigned) & set(args)
    assert len(assigned) == len(set(assigned))
    (spec,) = compile_specs(lowered.extern)
    assert spec.constants == {(4,): 1, (6,): 256, (7,): 256}


def test_elementwise_names_step_aside_too():
    @tilefish.kernel
    def clash(
        n: Tensor[M, D, f32], BLOCK: Tensor[M, D, f32], pid: Tensor[M, D, f32], mask: const = 1
    ) -> Tensor[M, D, f32]:
        return n + BLOCK + pid

    lowered = clash.lower()
    assert lowered.template == "elementwise"
    (spec,) = compile_specs(lowered.extern)
    assert list(spec.signature) == ["n", "BLOCK", "pid", "mask", "out", "n_1", "BLOCK_1"]
    assert spec.constants == {(3,): 1, (6,): 1024}
    src = body(lowered)
    assert src[:3] == ["pid_1 = tl.program_id(0)", "offs = pid_1 * BLOCK_1 + tl.arange(0, BLOCK_1)", "mask_1 = offs < n_1"]
    assert next(a for a in spec.args if a.name == "n_1").dim == "256*M"
    assert spec.grid == "(cdiv(256*M, 1024),)"


def test_names_the_generated_code_uses_are_not_reused():
    @tilefish.kernel
    def clash(tl: Tensor[M, D, f32], float: Tensor[M, D, f32], triton: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return tl + float + triton

    lowered = clash.lower()
    ast.parse(lowered.source)
    assert "tl_1" not in lowered.source  # parameters keep their names; it is the generated code that avoids them


def test_elementwise_count_is_the_product_of_the_shape():
    n = Dim("n", divisible_by=8)

    @tilefish.kernel
    def one_dim(x: Tensor[n, f32]) -> Tensor[n, f32]:
        return x * 2.0

    (spec,) = compile_specs(one_dim.lower().extern)
    assert list(spec.signature) == ["x", "out", "n", "BLOCK"]  # the parameter is called n whatever the dimensions are called
    assert next(a for a in spec.args if a.name == "n").dim == "n"  # and it carries the kernel's dimension n
    assert spec.grid == "(cdiv(n, 1024),)"
    assert (2,) not in spec.attrs  # 8 | n only

    @tilefish.kernel
    def two_dims(x: Tensor[n, M, f32]) -> Tensor[n, M, f32]:  # not affine: a fresh dimension, which must not be called n
        return x * 2.0

    lowered = two_dims.lower()
    (spec,) = compile_specs(lowered.extern)
    assert next(a for a in spec.args if a.name == "n").dim == "n_1"
    assert spec.grid == "(cdiv(n_1, 1024),)"
    assert [a.shape for a in spec.args if a.kind == "pointer"] == [("n", "M"), ("n", "M")]
    assert "flattened to n = n * M elements" in lowered.source


def test_elementwise_count_carries_what_the_trace_proves():
    K = Dim("K", divisible_by=64)
    P = Dim("P", divisible_by=4)

    @tilefish.kernel
    def two_dims(x: Tensor[K, P, f32]) -> Tensor[K, P, f32]:  # product of two symbolic dims: 64 * 4 = 256
        return x + 1.0

    @tilefish.kernel
    def odd(x: Tensor[M, 7, f32]) -> Tensor[M, 7, f32]:
        return x + 1.0

    @tilefish.kernel
    def static(x: Tensor[2560, f32]) -> Tensor[2560, f32]:
        return x + 1.0

    def n_divisor(op):
        (spec,) = compile_specs(op.lower().extern)
        return next(a for a in spec.args if a.name == "n"), spec

    arg, spec = n_divisor(two_dims)
    assert [p.by for p in spec.preconditions if p.kind == "divisible"] == [16]
    arg, spec = n_divisor(odd)
    assert not [p for p in spec.preconditions if p.kind == "divisible"]
    assert (2,) not in spec.attrs
    arg, spec = n_divisor(static)
    assert [p.by for p in spec.preconditions if p.kind == "divisible"] == [16]  # 2560 = 512 * 5


# -- API --------------------------------------------------------------------------------------------------------------


def test_lower_takes_only_kernels():
    @tilefish.extern(triton="my_kernels:k", grid=lambda N: (N,))
    def ext(x: Tensor[N, f32], y: Out[Tensor[N, f32]]): ...

    @tilefish.graph
    def g(x: Tensor[N, f32]):
        return x + 1.0

    for bad in (ext, g, lambda x: x, None):
        with pytest.raises(TilefishLoweringError) as exc:
            tilefish.lower(bad)
        assert exc.value.rule == "not-a-kernel"


def test_lower_validates_its_options():
    @tilefish.kernel
    def k(x: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return x + 1.0

    for kwargs, rule in (
        ({"module": "not a module"}, "module"),
        ({"module": "a..b"}, "module"),
        ({"module": "a.class"}, "module"),
        ({"module": "a.é"}, "module"),
        ({"module": "a.b\n"}, "module"),
        ({"module": 3}, "module"),
        ({"template": "tile"}, "template"),
        ({"num_warps": 3}, "num-warps"),
        ({"num_warps": True}, "num-warps"),
        ({"num_warps": 4.0}, "num-warps"),
    ):
        with pytest.raises(TilefishLoweringError) as exc:
            k.lower(**kwargs)
        assert exc.value.rule == rule, kwargs
    for bad in (3, 0, 64, "4", True, 2.0):
        with pytest.raises(TilefishDeclarationError, match="num_warps"):
            tilefish.kernel(num_warps=bad)(lambda x: x)


def test_kernel_name_must_make_a_function_name():
    @tilefish.kernel(name="my-kernel")
    def k(x: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return x + 1.0

    err = fails(k, "name", contains=["tf_my-kernel"])
    assert "kernel 'my-kernel'" in str(err)

    @tilefish.kernel(name="naïve")  # a compile spec only accepts ASCII identifiers
    def accented(x: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return x + 1.0

    fails(accented, "name")



def test_const_without_a_default_cannot_be_compiled_in():
    @tilefish.kernel
    def k(x: Tensor[M, D, f32], *, c: const) -> Tensor[M, D, f32]:
        return x + 1.0

    fails(k, "const-default", contains=["'c'"])


def test_runtime_dimension_parameters_pass_through():
    @tilefish.kernel
    def k(x: Tensor[M, D, f32], m: M) -> Tensor[M, D, f32]:
        return x + 1.0

    lowered = k.lower(template="row")
    (spec,) = compile_specs(lowered.extern)
    assert list(spec.signature) == ["x", "m", "out", "D", "BLOCK_D"]
    assert spec.signature["m"] == "i32"


def test_lowering_leaves_the_kernel_and_its_specs_alone():
    @tilefish.kernel
    def k(x: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return x + 1.0

    before = str(k.graph)
    first = k.lower()
    k.lower(module="other.place")
    assert str(k.graph) == before
    assert first.extern is not k.lower().extern
    assert tilefish.lower(k, module="a.b").extern.triton == "a.b:tf_k"
    assert repr(first) == "<LoweredKernel k: elementwise template, tilefish_generated.k:tf_k>"


def test_lowered_extern_is_a_plain_extern_op():
    @tilefish.kernel
    def k(x: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return x + 1.0

    ext = k.lower(template="row").extern
    assert isinstance(ext, tilefish.ExternOp)
    assert ext.name == "k" and ext.triton_function == "tf_k" and ext.num_stages == 3
    assert ext.grid_string == "(M,)"
    import copy

    assert copy.copy(ext) is ext and copy.deepcopy(ext) is ext
    with pytest.raises(tilefish.TilefishPicklingError):
        pickle.dumps(ext)  # generated stubs are not module-level objects


def test_lowering_error_pickles_with_its_rule():
    err = TilefishLoweringError("boom", rule="row.rank")
    again = pickle.loads(pickle.dumps(err))
    assert str(again) == "boom" and again.rule == "row.rank" and again.causes == ()


def test_write_creates_nested_folders_and_overwrites(tmp_path):
    @tilefish.kernel
    def k(x: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return x + 1.0

    lowered = k.lower(module="a.b.c")
    path = lowered.write(tmp_path / "out")
    assert path == tmp_path / "out" / "a" / "b" / "c.py"
    assert path.read_text() == lowered.source
    path.write_text("stale")
    assert lowered.write(tmp_path / "out") == path and path.read_text() == lowered.source
    assert lowered.write(str(tmp_path / "out")) == path  # a str works too
