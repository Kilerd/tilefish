"""Requirements through nested calls, views, aliasing, versions, and kernel parameter use."""

import pytest
from conftest import Tracer

import tilefish
from tilefish import (
    BucketRequirement,
    Dim,
    InOut,
    Out,
    Requirement,
    Tensor,
    TilefishTypeError,
    bf16,
    const,
    f32,
    i32,
)

M = Dim("M")
N = Dim("N", divisible_by=128)
K = Dim("K", divisible_by=32)


@tilefish.library("gemm_bf16", acc=f32)
def gemm_bf16(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16]) -> Tensor[M, N, bf16]: ...


# -- requirements survive nested graph and kernel calls ---------------------------------------------------------

K2 = Dim("K2")  # no divisibility declared: the body's call to gemm_bf16 records a requirement on it
N2 = Dim("N2", divisible_by=128)


@tilefish.graph
def inner(x: Tensor[M, K2, bf16], w: Tensor[N2, K2, bf16]):
    return gemm_bf16(x, w)


@tilefish.kernel
def inner_kernel(x: Tensor[M, K2, bf16], w: Tensor[N2, K2, bf16]) -> Tensor[M, N2, bf16]:
    return gemm_bf16(x, w)


def test_the_callee_recorded_its_requirement():
    assert [str(r) for r in inner.graph.requirements] == ["K2 % 32 == 0  (required by gemm_bf16: argument 'a')"]
    assert [str(r) for r in inner_kernel.graph.requirements] == ["K2 % 32 == 0  (required by gemm_bf16: argument 'a')"]


@pytest.mark.parametrize("callee", [inner, inner_kernel], ids=["graph", "kernel"])
def test_callee_requirements_are_checked_against_static_dims(callee):
    with pytest.raises(
        TilefishTypeError,
        match=rf"{callee.name}: its body requires K2 to be divisible by 32 \(gemm_bf16: argument 'a'\), but this call binds it to 33",
    ):

        @tilefish.graph
        def outer(x: Tensor[M, 33, bf16], w: Tensor[128, 33, bf16]):
            return callee(x, w)

    @tilefish.graph
    def ok(x: Tensor[M, 64, bf16], w: Tensor[128, 64, bf16]):
        return callee(x, w)

    assert ok.graph.requirements == ()


@pytest.mark.parametrize("callee", [inner, inner_kernel], ids=["graph", "kernel"])
def test_callee_requirements_propagate_to_symbolic_dims(callee):
    K3 = Dim("K3")  # unconstrained

    @tilefish.graph
    def outer(x: Tensor[M, K3, bf16], w: Tensor[128, K3, bf16], v: Tensor[128, K3, bf16]):
        return callee(x, w), callee(x, v)

    origin = f"{callee.name} (its body: gemm_bf16: argument 'a')"
    assert outer.graph.requirements == (Requirement(K3, 32, origin),)  # once: the second call finds it known

    K64 = Dim("K64", divisible_by=64)

    @tilefish.graph
    def satisfied(x: Tensor[M, K64, bf16], w: Tensor[128, K64, bf16]):
        return callee(x, w)

    assert satisfied.graph.requirements == ()

    Fg = Dim("Fg")

    @tilefish.graph
    def affine(x: Tensor[M, 2 * Fg, bf16], w: Tensor[128, 2 * Fg, bf16]):
        return callee(x, w)

    assert [(str(r.dim), r.divisor) for r in affine.graph.requirements] == [("2*Fg", 32)]


def test_requirements_propagate_through_several_levels():
    @tilefish.graph
    def mid(x: Tensor[M, K2, bf16], w: Tensor[N2, K2, bf16]):
        return inner(x, w)

    assert [str(r) for r in mid.graph.requirements] == [
        "K2 % 32 == 0  (required by inner (its body: gemm_bf16: argument 'a'))"
    ]
    with pytest.raises(TilefishTypeError, match=r"mid: its body requires K2 to be divisible by 32.*binds it to 33"):

        @tilefish.graph
        def top(x: Tensor[M, 33, bf16], w: Tensor[128, 33, bf16]):
            return mid(x, w)

    K5 = Dim("K5")

    @tilefish.graph
    def top2(x: Tensor[M, K5, bf16], w: Tensor[128, K5, bf16]):
        return mid(x, w)

    assert [(str(r.dim), r.divisor) for r in top2.graph.requirements] == [("K5", 32)]


# -- bucket requirements ------------------------------------------------------------------------------------------

S = Dim("S", buckets=[32, 64])


@tilefish.library("per_bucket")
def per_bucket(x: Tensor[S, bf16]) -> Tensor[S, bf16]: ...


def test_a_bucketed_callee_dim_bound_to_an_unbucketed_symbolic_dim_records_a_requirement():
    Mg = Dim("Mg")

    @tilefish.graph
    def g(x: Tensor[Mg, bf16], y: Tensor[Mg, bf16]):
        return per_bucket(x), per_bucket(y)

    assert g.graph.requirements == (BucketRequirement(Mg, (32, 64), "per_bucket: argument 'x'"),)  # once
    assert "Mg in [32, 64]  (required by per_bucket: argument 'x')" in str(g.graph)

    @tilefish.graph
    def compound(x: Tensor[2 * Mg, bf16]):
        return per_bucket(x)

    assert [(str(r.dim), r.buckets) for r in compound.graph.requirements] == [("2*Mg", (32, 64))]

    covered = Dim("Covered", buckets=[32])

    @tilefish.graph
    def sub(x: Tensor[covered, bf16]):
        return per_bucket(x)

    assert sub.graph.requirements == ()


def test_bucket_requirements_propagate_through_nested_calls():
    Mg, Q = Dim("Mg"), Dim("Q")

    @tilefish.graph
    def inner_b(x: Tensor[Mg, bf16]):
        return per_bucket(x)

    @tilefish.graph
    def outer_b(x: Tensor[Q, bf16]):
        return inner_b(x)

    assert outer_b.graph.requirements == (
        BucketRequirement(Q, (32, 64), "inner_b (its body: per_bucket: argument 'x')"),
    )
    with pytest.raises(TilefishTypeError, match=r"inner_b: its body requires Mg to be one of \[32, 64\].*binds it to 48"):

        @tilefish.graph
        def static_bad(x: Tensor[48, bf16]):
            return inner_b(x)

    @tilefish.graph
    def static_ok(x: Tensor[64, bf16]):
        return inner_b(x)

    assert static_ok.graph.requirements == ()


def test_a_failed_call_leaves_no_requirements_behind():
    @tilefish.library("lk")
    def lk(a: Tensor[M, K, bf16], *, steps: const[i32] = 1) -> Tensor[M, K, bf16]: ...

    Kp = Dim("Kp")

    @tilefish.graph
    def g(x: Tensor[M, Kp, bf16]):
        try:
            lk(x, steps=1.5)  # K binds first (recording a requirement), then the const is rejected
        except TilefishTypeError:
            pass
        return x

    assert g.graph.requirements == ()

    @tilefish.graph
    def g2(x: Tensor[M, Kp, bf16]):
        try:
            lk(x, steps=1.5)
        except TilefishTypeError:
            pass
        return lk(x)

    assert [str(r.dim) for r in g2.graph.requirements] == ["Kp"]


# -- strided views --------------------------------------------------------------------------------------------------


@tilefish.library("scale")
def scale(x: Tensor[M, 4, bf16]) -> Tensor[M, 4, bf16]: ...


def test_views_cannot_be_passed_to_library_extern_kernel_or_graph_arguments():
    @tilefish.extern(triton="m:f", grid=lambda M: (M,))
    def ext(x: Tensor[M, 4, bf16], y: Out[Tensor[M, 4, bf16]]): ...

    @tilefish.kernel
    def kern(x: Tensor[M, 4, bf16]) -> Tensor[M, 4, bf16]:
        return x * 1

    @tilefish.graph
    def grph(x: Tensor[M, 4, bf16]):
        return x

    for op in (scale, ext, kern, grph):
        with pytest.raises(
            TilefishTypeError,
            match=rf"{op.name}: argument 'x' is a strided view \(View\[M, 4, bf16\]\); strided views arrive in M2, pass the unsplit tensor",
        ):

            @tilefish.graph
            def g(t: Tensor[M, 8, bf16]):
                a, _ = t.split(4, axis=1)
                return op(a)


def test_leading_axis_chunks_and_materialized_views_are_accepted():
    @tilefish.graph
    def g(t: Tensor[M, 8, bf16], u: Tensor[8, 4, bf16]):
        a, _ = t.split(4, axis=1)
        first, _ = u.split(4, axis=0)  # a leading-axis chunk of a contiguous tensor is contiguous
        return scale(a * 1), scale(a.to(f32).to(bf16)), tilefish.silu(a.to(f32)).to(bf16), first

    assert [str(v.type) for v in g.graph.outputs] == ["Tensor[M, 4, bf16]"] * 3 + ["Tensor[4, 4, bf16]"]


def test_a_graph_that_returns_a_view_has_a_view_typed_output():
    @tilefish.graph
    def g(t: Tensor[M, 8, bf16]):
        a, _ = t.split(4, axis=1)
        return a

    assert g.graph.outputs[0].type == Tensor[M, 4, bf16].as_view()
    assert g.signature.returns[0].contiguous is False


# -- aliasing and versions ------------------------------------------------------------------------------------------


@tilefish.extern(triton="m:f", grid=lambda M: (M,))
def two_in_one_out(a: Tensor[M, bf16], b: InOut[Tensor[M, bf16]], y: Out[Tensor[M, bf16]]): ...


@tilefish.extern(triton="m:g", grid=lambda M: (M,))
def two_inout(a: InOut[Tensor[M, bf16]], b: InOut[Tensor[M, bf16]]): ...


@tilefish.library("add2")
def add2(a: Tensor[M, bf16], b: Tensor[M, bf16]) -> Tensor[M, bf16]: ...


def test_the_same_tensor_cannot_be_an_input_and_an_inout_argument():
    with pytest.raises(
        TilefishTypeError,
        match=r"two_in_one_out: arguments 'a' and 'b' are the same tensor \(%x\) but one is written in place",
    ):

        @tilefish.graph
        def g(x: InOut[Tensor[M, bf16]]):
            return two_in_one_out(x, x)

    with pytest.raises(TilefishTypeError, match=r"two_inout: arguments 'a' and 'b' are the same tensor"):

        @tilefish.graph
        def g2(x: InOut[Tensor[M, bf16]]):
            two_inout(x, x)

    @tilefish.graph
    def ok(x: InOut[Tensor[M, bf16]], z: InOut[Tensor[M, bf16]]):
        two_inout(x, z)
        return add2(x, x)  # reading the same tensor twice is fine

    assert ok.graph.ops() == ["two_inout", "add2"]


def test_aliasing_is_checked_on_the_newest_version():
    with pytest.raises(TilefishTypeError, match="the same tensor"):

        @tilefish.graph
        def g(x: InOut[Tensor[M, bf16]]):
            two_inout(x.live, x)  # the parameter and its newest version are the same tensor

    with pytest.raises(TilefishTypeError, match="the same tensor"):

        @tilefish.graph
        def g2(x: InOut[Tensor[M, bf16]], z: InOut[Tensor[M, bf16]]):
            two_inout(x, z)  # x and z are updated in place: the newest versions are what aliasing is checked on
            two_in_one_out(z.live, z)


def test_an_inout_argument_gets_a_new_version_that_later_reads_see():
    @tilefish.library("accumulate")
    def accumulate(acc: InOut[Tensor[M, 64, f32]], x: Tensor[M, 64, f32]): ...

    @tilefish.graph
    def g(acc: InOut[Tensor[M, 64, f32]], x: Tensor[M, 64, f32]):
        accumulate(acc, x)
        y = acc * 2  # reads the updated acc
        accumulate(acc, y)
        return acc

    first, mul, second = g.graph.nodes
    assert first.attrs["inout"] == ("acc",) and len(first.outputs) == 1
    assert first.outputs[0].type == Tensor[M, 64, f32]
    assert mul.inputs[0] is first.outputs[0]
    assert second.inputs[0] is first.outputs[0]  # the second call updates the first call's result
    assert g.graph.outputs == (second.outputs[0],)
    assert "%0: Tensor[M, 64, f32] = accumulate(%acc, %x) {kind=library, inout=(acc,)}" in str(g.graph)


def test_extern_results_come_before_the_inout_versions():
    @tilefish.graph
    def g(a: Tensor[M, bf16], b: InOut[Tensor[M, bf16]]):
        y = two_in_one_out(a, b)
        return y, b

    (node,) = g.graph.nodes
    assert len(node.outputs) == 2  # y, then the new version of b
    assert g.graph.outputs == (node.outputs[0], node.outputs[1])
    assert node.output_types == (Tensor[M, bf16], Tensor[M, bf16])


# -- kernel parameters must be used ------------------------------------------------------------------------------------


def test_a_kernel_out_param_that_is_never_written_is_an_error():
    with pytest.raises(TilefishTypeError, match=r"kernel 'k': Out parameter 'y' is never written; assign it with y\[\.\.\.\] = value"):

        @tilefish.kernel
        def k(x: Tensor[M, bf16], y: Out[Tensor[M, bf16]]):
            pass

    with pytest.raises(TilefishTypeError, match="Out parameter 'y2' is never written"):

        @tilefish.kernel
        def k2(x: Tensor[M, bf16], y: Out[Tensor[M, bf16]], y2: Out[Tensor[M, bf16]]):
            y[...] = x


def test_a_kernel_inout_param_that_is_never_written_is_an_error():
    with pytest.raises(TilefishTypeError, match="kernel 'k': InOut parameter 'r' is never written"):

        @tilefish.kernel
        def k(x: Tensor[M, bf16], r: InOut[Tensor[M, bf16]]):
            return r + x


def test_a_kernel_inout_param_counts_as_written_by_a_store_or_an_in_place_call():
    @tilefish.library("bump")
    def bump(acc: InOut[Tensor[M, bf16]]): ...

    @tilefish.kernel
    def by_store(r: InOut[Tensor[M, bf16]]):
        r[...] = r + 1

    @tilefish.kernel
    def by_call(r: InOut[Tensor[M, bf16]]):
        bump(r)

    assert by_store.graph.ops() == ["add", "store"]
    assert by_call.graph.ops() == ["bump"]


def test_graphs_do_not_have_to_write_their_inout_and_out_params():
    @tilefish.graph
    def g(r: InOut[Tensor[M, bf16]]):
        return r

    assert g.graph.outputs == (g.graph.inputs[0],)


# -- errors raised while tracing name the graph or kernel ------------------------------------------------------------------


def test_errors_while_tracing_name_the_enclosing_graph_or_kernel():
    with pytest.raises(TilefishTypeError, match=r"^in graph 'my_graph': add \(\+\): no implicit promotion"):

        @tilefish.graph
        def my_graph(x: Tensor[M, bf16], y: Tensor[M, f32]):
            return x + y

    with pytest.raises(TilefishTypeError, match=r"^in kernel 'my_kernel': gemm_bf16: argument 'a': expected rank 2"):

        @tilefish.kernel
        def my_kernel(x: Tensor[M, bf16], w: Tensor[M, 64, bf16]):
            return gemm_bf16(x, w)

    with pytest.raises(tilefish.TilefishTypeError, match=r"^in graph 'renamed': ") as info:

        @tilefish.graph(name="renamed")
        def whatever(x: Tensor[M, bf16]):
            return x + "a"

    assert str(info.value).count("in graph") == 1


def test_errors_that_are_not_tilefish_errors_pass_through_unchanged():
    with pytest.raises(ValueError, match="invalid literal"):

        @tilefish.graph
        def g(x: Tensor[M, bf16]):
            return int("not a number")


def test_a_mismatch_between_a_dim_and_its_constrained_twin_shows_the_constraints():
    plain, constrained = Dim("Kc"), Dim("Kc", divisible_by=32)

    @tilefish.library("pair")
    def pair(a: Tensor[M, constrained, bf16], b: Tensor[M, constrained, bf16]) -> Tensor[M, bf16]: ...

    with pytest.raises(
        TilefishTypeError,
        match=r"argument 'b' \(Tensor\[M, Kc, bf16\]\): dimension 1 is Dim\('Kc'\), but the signature requires Kc = Dim\('Kc', divisible_by=32\)",
    ):

        @tilefish.graph
        def g(x: Tensor[M, constrained, bf16], y: Tensor[M, plain, bf16]):
            return pair(x, y)


# -- typed consts at call sites ---------------------------------------------------------------------------------------------


def test_typed_consts_do_not_convert_implicitly_at_call_sites():
    @tilefish.library("scale_by")
    def scale_by(x: Tensor[M, bf16], *, factor: const[f32] = 1.0) -> Tensor[M, bf16]: ...

    @tilefish.kernel
    def same(x: Tensor[M, bf16], *, f: const[f32] = 2.0) -> Tensor[M, bf16]:
        return scale_by(x, factor=f)

    @tilefish.kernel
    def untyped(x: Tensor[M, bf16], *, f: const = 2) -> Tensor[M, bf16]:
        return scale_by(x, factor=f)

    assert same.graph.nodes[0].attrs["consts"] == {"factor": 2.0}
    assert untyped.graph.nodes[0].attrs["consts"] == {"factor": 2.0}
    with pytest.raises(TilefishTypeError, match=r"argument 'factor' is const\[f32\] but 'f' is const\[i32\]"):

        @tilefish.kernel
        def mismatched(x: Tensor[M, bf16], *, f: const[i32] = 2) -> Tensor[M, bf16]:
            return scale_by(x, factor=f)


# -- values from another trace --------------------------------------------------------------------------------------------


def test_a_value_from_another_trace_is_rejected_as_a_call_argument():
    a, b = Tracer(), Tracer()
    x = a.val(Tensor[M, 64, bf16])
    w = a.val(Tensor[128, 64, bf16])
    foreign = b.val(Tensor[128, 64, bf16])
    assert gemm_bf16(x, w).type == Tensor[M, 128, bf16]
    with pytest.raises(TilefishTypeError, match="gemm_bf16: argument 'b' belongs to a different trace"):
        gemm_bf16(x, foreign)
    with pytest.raises(TilefishTypeError, match="gemm_bf16: argument 'b' belongs to a different trace"):
        gemm_bf16(foreign, w)  # the first tensor argument decides which trace the call belongs to
    assert [n.op for n in b.nodes] == []  # nothing was recorded in either trace by the failed calls
    assert [n.op for n in a.nodes] == ["gemm_bf16"]


# -- bucket sets: subset, superset, partial overlap and disjoint behave the same directly and through nesting -----------------

Wide = Dim("Wide", buckets=[32, 64, 128])  # a superset of per_bucket's [32, 64]
Partial = Dim("Partial", buckets=[64, 256])  # overlaps in 64 only
Other = Dim("Other", buckets=[128, 256])  # nothing in common
Within = Dim("Within", buckets=[32])


def test_bucket_sets_directly_and_through_a_nested_graph_are_treated_alike():
    Mg = Dim("Mg")

    @tilefish.graph
    def inner_b(x: Tensor[Mg, bf16]):
        return per_bucket(x)

    for caller_dim, runtime_checked in ((Wide, True), (Partial, True), (Within, False)):
        @tilefish.graph
        def via_direct(x: Tensor[caller_dim, bf16]):
            return per_bucket(x)

        @tilefish.graph
        def via_nested(x: Tensor[caller_dim, bf16]):
            return inner_b(x)

        for g in (via_direct, via_nested):
            reqs = g.graph.requirements
            if runtime_checked:
                (r,) = reqs
                assert isinstance(r, BucketRequirement) and r.dim == caller_dim and r.buckets == (32, 64)
            else:
                assert reqs == ()  # a subset is statically satisfied

    for g_fn in (per_bucket, inner_b):
        with pytest.raises(TilefishTypeError, match=r"is bucketed over \[32, 64\] but Other has buckets \[128, 256\] \(no bucket in common\)|body.*Other has buckets \[128, 256\] \(no bucket in common\)"):

            @tilefish.graph
            def disjoint(x: Tensor[Other, bf16]):
                return g_fn(x)


def test_the_nested_requirement_names_the_callee_and_the_original_site():
    Mg = Dim("Mg")

    @tilefish.graph
    def inner_b(x: Tensor[Mg, bf16]):
        return per_bucket(x)

    @tilefish.graph
    def outer(x: Tensor[Wide, bf16]):
        return inner_b(x)

    assert outer.graph.requirements == (
        BucketRequirement(Wide, (32, 64), "inner_b (its body: per_bucket: argument 'x')"),
    )
    assert "Wide in [32, 64]  (required by inner_b (its body: per_bucket: argument 'x'))" in str(outer.graph)


# -- a nested graph that returns a view: the view survives instantiation ----------------------------------------------------------


def test_a_view_returned_by_a_nested_graph_is_still_rejected_by_declared_operators():
    @tilefish.graph
    def returns_view(x: Tensor[M, 8, bf16]):
        a, _ = x.split(4, axis=1)
        return a

    assert returns_view.signature.returns[0].contiguous is False

    @tilefish.graph
    def through_the_call(x: Tensor[M, 8, bf16]):
        return returns_view(x)

    assert through_the_call.graph.outputs[0].type == Tensor[M, 4, bf16].as_view()

    with pytest.raises(TilefishTypeError, match=r"scale: argument 'x' is a strided view \(View\[M, 4, bf16\]\)"):

        @tilefish.graph
        def passes_it_on(x: Tensor[M, 8, bf16]):
            return scale(returns_view(x))

    @tilefish.graph
    def materialized(x: Tensor[M, 8, bf16]):
        return scale(returns_view(x) + 0)

    assert materialized.graph.outputs[0].type == Tensor[M, 4, bf16]


# -- an Out parameter cannot be read at a call site ---------------------------------------------------------------------------------


def test_an_out_param_cannot_be_passed_as_a_call_argument():
    with pytest.raises(TilefishTypeError, match=r"scale: argument 'x' reads Out parameter 'y', which is write-only"):

        @tilefish.kernel
        def before(x: Tensor[M, 4, bf16], y: Out[Tensor[M, 4, bf16]]):
            scale(y)

    with pytest.raises(TilefishTypeError, match=r"reads Out parameter 'y', which is write-only"):

        @tilefish.kernel
        def after(x: Tensor[M, 4, bf16], y: Out[Tensor[M, 4, bf16]]):
            y[...] = x
            scale(y)  # not even after it was written

    with pytest.raises(TilefishTypeError, match=r"two_in_one_out: argument 'b' reads Out parameter 'y'"):

        @tilefish.kernel
        def as_inout(x: Tensor[M, bf16], y: Out[Tensor[M, bf16]]):
            two_in_one_out(x, y)
