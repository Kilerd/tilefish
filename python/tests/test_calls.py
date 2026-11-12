"""Calls to library/extern/kernel/graph operators: unification of dimensions, requirements, results."""

import pytest

import tilefish
from tilefish import (
    BlockScaled,
    Dim,
    InOut,
    Out,
    Requirement,
    Tensor,
    TilefishDeclarationError,
    TilefishTypeError,
    bf16,
    const,
    e4m3,
    f32,
    ue8m0,
)

M = Dim("M")
N = Dim("N", divisible_by=128)
K = Dim("K", divisible_by=32)
H = Dim("H", 1024)
n = Dim("n", divisible_by=16)

mxfp8 = BlockScaled(e4m3, ue8m0, 32)


@tilefish.library("gemm", acc=f32)
def gemm(a: Tensor[M, K, bf16], b: Tensor[N, K, mxfp8]) -> Tensor[M, N, bf16]: ...


@tilefish.library("gemm_bf16", acc=f32)
def gemm_bf16(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16]) -> Tensor[M, N, bf16]: ...


# -- success -------------------------------------------------------------------------------------


def test_dims_unify_across_arguments():
    @tilefish.graph
    def g(x: Tensor[M, H, bf16], w: Tensor[4096, H, mxfp8]):
        return gemm(x, w)

    (out,) = g.graph.outputs
    assert out.type == Tensor[M, 4096, bf16]
    (node,) = g.graph.nodes
    assert node.op == "gemm"
    assert node.attrs == {"kind": "library", "acc": f32}
    assert node.inputs == tuple(g.graph.inputs)


def test_signature_dims_are_fresh_at_every_call():
    @tilefish.graph
    def g(x: Tensor[M, 64, bf16], y: Tensor[7, 128, bf16], w1: Tensor[256, 64, bf16], w2: Tensor[384, 128, bf16]):
        return gemm_bf16(x, w1), gemm_bf16(y, w2)

    a, b = g.graph.outputs
    assert a.type == Tensor[M, 256, bf16]
    assert b.type == Tensor[7, 384, bf16]  # K was 64 in the first call and 128 in the second


def test_the_same_dim_object_can_be_both_caller_symbol_and_callee_variable():
    @tilefish.library("transpose2d")
    def transpose2d(a: Tensor[M, N, bf16]) -> Tensor[N, M, bf16]: ...

    # the caller's tensor is Tensor[N, M]: callee M binds to caller N and callee N binds to caller M
    @tilefish.graph
    def g(x: Tensor[N, M, bf16]):
        return transpose2d(x)

    # the swapped binding makes callee N (divisible_by=128) stand for the caller's M: a requirement, not an error
    assert g.graph.outputs[0].type == Tensor[M, N, bf16]
    assert g.graph.requirements == (Requirement(M, 128, "transpose2d: argument 'a'"),)


def test_affine_signature_dims_are_solved():
    H2 = Dim("H2")

    @tilefish.library("halve")
    def halve(x: Tensor[M, 2 * H2, bf16]) -> Tensor[M, H2, bf16]: ...

    Fg = Dim("Fg")

    @tilefish.graph
    def g(a: Tensor[M, 6144, bf16], b: Tensor[M, 2 * Fg, bf16]):
        return halve(a), halve(b)

    x, y = g.graph.outputs
    assert x.type == Tensor[M, 3072, bf16]
    assert y.type == Tensor[M, Fg, bf16]


def test_affine_signature_dim_with_no_solution():
    H2 = Dim("H2")

    @tilefish.library("halve")
    def halve(x: Tensor[M, 2 * H2, bf16]) -> Tensor[M, H2, bf16]: ...

    with pytest.raises(TilefishTypeError, match=r"halve: argument 'x'.*dimension 1 is 7, which cannot match 2\*H2"):

        @tilefish.graph
        def odd(x: Tensor[M, 7, bf16]):
            return halve(x)

    Fg = Dim("Fg")
    with pytest.raises(TilefishTypeError, match="cannot match 2\\*H2"):

        @tilefish.graph
        def odd2(x: Tensor[M, Fg + 1, bf16]):
            return halve(x)


def test_static_signature_dims_are_literals():
    D = Dim("D", 2560)

    @tilefish.library("norm_like")
    def norm_like(x: Tensor[M, D, bf16]) -> Tensor[M, D, bf16]: ...

    @tilefish.graph
    def ok(x: Tensor[M, 2560, bf16]):
        return norm_like(x)

    assert ok.graph.outputs[0].type == Tensor[M, D, bf16]
    with pytest.raises(TilefishTypeError, match=r"norm_like: argument 'x' \(Tensor\[M, 1024, bf16\]\): dimension 1 is 1024, but the signature requires D = 2560"):

        @tilefish.graph
        def bad(x: Tensor[M, 1024, bf16]):
            return norm_like(x)

    with pytest.raises(TilefishTypeError, match="dimension 1 is M, but the signature requires D = 2560"):

        @tilefish.graph
        def bad_sym(x: Tensor[2560, M, bf16]):
            return norm_like(x)


# -- failure -------------------------------------------------------------------------------------


def test_dim_unification_failure_names_op_argument_and_binding():
    with pytest.raises(
        TilefishTypeError,
        match=r"gemm_bf16: argument 'b' \(Tensor\[256, 128, bf16\]\): dimension 1 is 128, but the signature requires K = 64 \(K was bound to 64 by argument 'a'\)",
    ):

        @tilefish.graph
        def g(x: Tensor[M, 64, bf16], w: Tensor[256, 128, bf16]):
            return gemm_bf16(x, w)


def test_symbolic_dim_mismatch_is_an_error_not_a_guess():
    Mg2 = Dim("Mg2")
    with pytest.raises(TilefishTypeError, match=r"argument 'b'.*dimension 1 is Mg2.*K = 64"):

        @tilefish.graph
        def g(x: Tensor[M, 64, bf16], w: Tensor[256, Mg2, bf16]):
            return gemm_bf16(x, w)


def test_divisibility_violation_with_static_dim():
    with pytest.raises(TilefishTypeError, match=r"gemm_bf16: argument 'a': K requires divisible_by=32 but dimension 1 is 100"):

        @tilefish.graph
        def g(x: Tensor[M, 100, bf16], w: Tensor[256, 100, bf16]):
            return gemm_bf16(x, w)
    with pytest.raises(TilefishTypeError, match=r"N requires divisible_by=128 but dimension 0 is 200"):

        @tilefish.graph
        def g2(x: Tensor[M, 64, bf16], w: Tensor[200, 64, bf16]):
            return gemm_bf16(x, w)


def test_divisibility_on_a_symbolic_dim_becomes_a_requirement():
    Kg = Dim("Kg")  # no divisibility declared

    @tilefish.graph
    def g(x: Tensor[M, Kg, bf16], w: Tensor[256, Kg, bf16], v: Tensor[256, Kg, bf16]):
        return gemm_bf16(x, w), gemm_bf16(x, v)

    # recorded once: the second call finds it already known
    assert g.graph.requirements == (Requirement(Kg, 32, "gemm_bf16: argument 'a'"),)
    assert "requires Kg % 32 == 0  (required by gemm_bf16: argument 'a')" in str(g.graph)


def test_declared_divisibility_satisfies_the_requirement():
    K64 = Dim("K64", divisible_by=64)

    @tilefish.graph
    def g(x: Tensor[M, K64, bf16], w: Tensor[256, K64, bf16]):
        return gemm_bf16(x, w)

    assert g.graph.requirements == ()
    assert g.graph.symbols == (M, K64)


def test_requirement_on_an_affine_symbolic_dim():
    Fg = Dim("Fg")

    @tilefish.graph
    def g(x: Tensor[M, 2 * Fg, bf16], w: Tensor[256, 2 * Fg, bf16]):
        return gemm_bf16(x, w)

    # 2*Fg is not provably a multiple of 32, so the call records the requirement on the expression
    assert [(str(r.dim), r.divisor) for r in g.graph.requirements] == [("2*Fg", 32)]


def test_dtype_mismatch_in_a_call():
    with pytest.raises(TilefishTypeError, match=r"gemm_bf16: argument 'a': expected dtype bf16, got f32"):

        @tilefish.graph
        def g(x: Tensor[M, 64, f32], w: Tensor[256, 64, bf16]):
            return gemm_bf16(x, w)
    with pytest.raises(TilefishTypeError, match=r"gemm: argument 'b': expected dtype blockscaled\(e4m3, ue8m0, 32\), got bf16"):

        @tilefish.graph
        def g2(x: Tensor[M, 64, bf16], w: Tensor[256, 64, bf16]):
            return gemm(x, w)


def test_blockscaled_formats_must_match():
    mxfp4 = BlockScaled(tilefish.e2m1, ue8m0, 32)
    with pytest.raises(TilefishTypeError, match="gemm: argument 'b': expected dtype"):

        @tilefish.graph
        def g(x: Tensor[M, 64, bf16], w: Tensor[256, 64, mxfp4]):
            return gemm(x, w)


def test_rank_mismatch_in_a_call():
    with pytest.raises(TilefishTypeError, match=r"gemm_bf16: argument 'a': expected rank 2 \(Tensor\[M, K, bf16\]\), got rank 1 \(Tensor\[64, bf16\]\)"):

        @tilefish.graph
        def g(x: Tensor[64, bf16], w: Tensor[256, 64, bf16]):
            return gemm_bf16(x, w)


def test_bucketed_signature_dim():
    S = Dim("S", buckets=[32, 64])
    sub = Dim("Sub", buckets=[32])
    wide = Dim("Wide", buckets=[32, 64, 128])

    @tilefish.library("per_bucket")
    def per_bucket(x: Tensor[S, bf16]) -> Tensor[S, bf16]: ...

    @tilefish.graph
    def ok(a: Tensor[64, bf16], b: Tensor[sub, bf16]):
        return per_bucket(a), per_bucket(b)

    with pytest.raises(TilefishTypeError, match=r"S is bucketed over \[32, 64\] but dimension 0 is 48"):

        @tilefish.graph
        def bad(a: Tensor[48, bf16]):
            return per_bucket(a)

    # a superset (or a partial overlap) is checked at run time: a requirement, not an error
    @tilefish.graph
    def superset(a: Tensor[wide, bf16]):
        return per_bucket(a)

    assert [str(r) for r in superset.graph.requirements] == [
        "Wide in [32, 64]  (required by per_bucket: argument 'x')"
    ]
    # disjoint bucket sets can never match
    disjoint = Dim("Disjoint", buckets=[128, 256])
    with pytest.raises(TilefishTypeError, match=r"S is bucketed over \[32, 64\] but Disjoint has buckets \[128, 256\] \(no bucket in common\)"):

        @tilefish.graph
        def bad2(a: Tensor[disjoint, bf16]):
            return per_bucket(a)


# -- argument and result handling -----------------------------------------------------------------


def test_calls_outside_a_trace_and_bad_arguments():
    with pytest.raises(TilefishTypeError, match="can only be called inside a graph or kernel body"):
        gemm_bf16(1, 2)

    with pytest.raises(TilefishTypeError, match=r"gemm_bf16: argument 'b' must be a traced tensor of type Tensor\[N, K, bf16\]"):

        @tilefish.graph
        def g(x: Tensor[M, 64, bf16]):
            return gemm_bf16(x, 3)

    with pytest.raises(TilefishTypeError, match=r"gemm_bf16: missing a required argument: 'b'"):

        @tilefish.graph
        def g2(x: Tensor[M, 64, bf16]):
            return gemm_bf16(x)

    with pytest.raises(TilefishTypeError, match="unexpected keyword argument 'c'"):

        @tilefish.graph
        def g3(x: Tensor[M, 64, bf16]):
            return gemm_bf16(x, x, c=1)


def test_out_parameters_are_results_and_dim_parameters_are_derived():
    @tilefish.extern(triton="my_kernels:gelu", grid=lambda n: (tilefish.cdiv(n, 1024),))
    def gelu(x: Tensor[n, bf16], y: Out[Tensor[n, bf16]], n_elem: n, BLOCK: const = 1024): ...

    @tilefish.graph
    def g(x: Tensor[4096, bf16]):
        return gelu(x)

    (node,) = g.graph.nodes
    assert node.op == "gelu"
    assert node.attrs == {"kind": "extern", "triton": "my_kernels:gelu", "consts": {"BLOCK": 1024}}
    assert node.inputs == (g.graph.inputs[0],)  # only the input tensor; y is a result, n_elem is derived
    assert g.graph.outputs[0].type == Tensor[4096, bf16]
    assert g.graph.outputs[0] is node.outputs[0]


def test_extern_divisibility_violation_and_requirement():
    @tilefish.extern(triton="my_kernels:gelu", grid=lambda n: (n,))
    def gelu(x: Tensor[n, bf16], y: Out[Tensor[n, bf16]], n_elem: n): ...

    with pytest.raises(TilefishTypeError, match=r"gelu: argument 'x': n requires divisible_by=16 but dimension 0 is 100"):

        @tilefish.graph
        def bad(x: Tensor[100, bf16]):
            return gelu(x)

    @tilefish.graph
    def sym(x: Tensor[M, bf16]):
        return gelu(x)

    assert sym.graph.requirements == (Requirement(M, 16, "gelu: argument 'x'"),)


def test_several_results_in_declaration_order():
    @tilefish.library("split_qk")
    def split_qk(x: Tensor[M, 128, bf16], q: Out[Tensor[M, 64, bf16]], k: Out[Tensor[M, 64, bf16]]): ...

    @tilefish.graph
    def g(x: Tensor[M, 128, bf16]):
        q, k = split_qk(x)
        return k, q

    assert [str(v.type) for v in g.graph.outputs] == ["Tensor[M, 64, bf16]"] * 2
    assert g.graph.nodes[0].outputs == (g.graph.outputs[1], g.graph.outputs[0])


def test_inout_arguments():
    @tilefish.library("accumulate")
    def accumulate(acc: InOut[Tensor[M, 64, f32]], x: Tensor[M, 64, f32]): ...

    @tilefish.graph
    def ok(acc: InOut[Tensor[M, 64, f32]], x: Tensor[M, 64, f32]):
        accumulate(acc, x)
        t = x + 1
        accumulate(t, x)  # a temporary is fresh storage and may be written

    assert [n.attrs["inout"] for n in ok.graph.nodes if n.op == "accumulate"] == [("acc",), ("acc",)]
    assert ok.graph.outputs == ()

    with pytest.raises(TilefishTypeError, match=r"accumulate: argument 'acc' is written in place but 'x' is a read-only input"):

        @tilefish.graph
        def bad(x: Tensor[M, 64, f32]):
            accumulate(x, x)


def test_const_arguments_library_vs_extern():
    @tilefish.library("scale_by")
    def scale_by(x: Tensor[M, bf16], *, factor: const[f32] = 1.0, steps: const[tilefish.i32] = 1) -> Tensor[M, bf16]: ...

    @tilefish.graph
    def g(x: Tensor[M, bf16]):
        return scale_by(x, factor=2), scale_by(x)

    assert g.graph.nodes[0].attrs["consts"] == {"factor": 2.0, "steps": 1}
    assert isinstance(g.graph.nodes[0].attrs["consts"]["factor"], float)
    assert g.graph.nodes[1].attrs["consts"] == {"factor": 1.0, "steps": 1}

    with pytest.raises(TilefishTypeError, match="argument 'steps' is const.*needs an int"):

        @tilefish.graph
        def bad(x: Tensor[M, bf16]):
            return scale_by(x, steps=1.5)

    @tilefish.extern(triton="my_kernels:norm", grid=lambda M: (M,))
    def norm(x: Tensor[M, bf16], y: Out[Tensor[M, bf16]], BLOCK: const = 1024): ...

    @tilefish.graph
    def fixed(x: Tensor[M, bf16]):
        return norm(x, BLOCK=1024)

    with pytest.raises(TilefishTypeError, match="constexpr 'BLOCK' is fixed to 1024 by the extern declaration"):

        @tilefish.graph
        def changed(x: Tensor[M, bf16]):
            return norm(x, BLOCK=2048)


def test_const_param_passes_through_to_a_call():
    @tilefish.library("scale_by")
    def scale_by(x: Tensor[M, bf16], *, factor: const[f32] = 1.0) -> Tensor[M, bf16]: ...

    @tilefish.kernel
    def k(x: Tensor[M, bf16], *, f: const[f32] = 3.0) -> Tensor[M, bf16]:
        return scale_by(x, factor=f)

    assert k.graph.nodes[0].attrs["consts"] == {"factor": 3.0}


def test_graphs_and_kernels_are_callable_operators():
    @tilefish.kernel
    def double(x: Tensor[M, 64, f32]) -> Tensor[M, 64, f32]:
        return x * 2

    @tilefish.graph
    def outer(a: Tensor[N, 64, f32]):
        return double(a)

    assert outer.graph.outputs[0].type == Tensor[N, 64, f32]
    node = outer.graph.nodes[0]
    assert node.op == "double" and node.attrs == {"kind": "kernel"}


# -- declarations --------------------------------------------------------------------------------


def test_declaration_errors():
    four = Dim("four", 4)

    with pytest.raises(TilefishDeclarationError, match="parameter 'x' needs an annotation"):

        @tilefish.library("f")
        def f(x) -> Tensor[M, bf16]: ...

    with pytest.raises(TilefishDeclarationError, match="unsupported annotation"):

        @tilefish.library("f")
        def f2(x: int) -> Tensor[M, bf16]: ...

    with pytest.raises(TilefishDeclarationError, match=r"annotated with the static dimension.*`x: const = four`"):

        @tilefish.extern(triton="m:f", grid=(1,))
        def f3(t: Tensor[M, bf16], x: four): ...

    with pytest.raises(TilefishDeclarationError, match="declare a return type"):

        @tilefish.library("f")
        def f4(x: Tensor[M, bf16]): ...

    with pytest.raises(TilefishDeclarationError, match=r"\*args/\*\*kwargs"):

        @tilefish.library("f")
        def f5(x: Tensor[M, bf16], *args) -> Tensor[M, bf16]: ...

    with pytest.raises(TilefishDeclarationError, match="return value 0 mentions dimension\\(s\\) N that no input tensor determines"):

        @tilefish.library("f")
        def f6(x: Tensor[M, bf16]) -> Tensor[N, bf16]: ...


def test_extern_declaration_errors():
    with pytest.raises(TilefishDeclarationError, match="returns nothing"):

        @tilefish.extern(triton="m:f", grid=(1,))
        def e1(x: Tensor[M, bf16]) -> Tensor[M, bf16]: ...

    with pytest.raises(TilefishDeclarationError, match="needs a default"):

        @tilefish.extern(triton="m:f", grid=(1,))
        def e2(x: Tensor[M, bf16], BLOCK: const): ...

    with pytest.raises(TilefishDeclarationError, match="module:function"):
        tilefish.extern(triton="no_colon")

    with pytest.raises(TilefishDeclarationError, match="grid parameter 'q' is not a dimension"):

        @tilefish.extern(triton="m:f", grid=lambda q: (q,))
        def e3(x: Tensor[M, bf16]): ...

    with pytest.raises(TilefishDeclarationError, match="num_warps must be a power of two"):
        tilefish.extern(triton="m:f", num_warps=0)


def test_graph_return_checks():
    with pytest.raises(TilefishTypeError, match=r"kernel 'k': return value 0 has type Tensor\[M, f32\] but the declaration says Tensor\[M, bf16\]"):

        @tilefish.kernel
        def k(x: Tensor[M, f32]) -> Tensor[M, bf16]:
            return x

    with pytest.raises(TilefishTypeError, match="declares 2 return value"):

        @tilefish.kernel
        def k2(x: Tensor[M, f32]) -> (Tensor[M, f32], Tensor[M, f32]):
            return x

    with pytest.raises(TilefishTypeError, match="the body must return a traced tensor"):

        @tilefish.graph
        def g(x: Tensor[M, f32]):
            return 1.0

    with pytest.raises(TilefishTypeError, match="cannot return Out parameter"):

        @tilefish.kernel
        def k3(x: Tensor[M, f32], y: Out[Tensor[M, f32]]):
            return y


def test_kernel_writes_an_out_parameter():
    @tilefish.kernel
    def scale(x: Tensor[M, f32], y: Out[Tensor[M, f32]]):
        y[...] = x * 2

    assert [n.op for n in scale.graph.nodes] == ["mul", "store"]
    assert scale.graph.nodes[1].inputs[0].access == "out"
