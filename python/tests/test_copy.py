"""Dims, dtypes, types, the IR and the AOT specs survive copy.deepcopy and pickle; exports do not shadow builtins."""

import builtins
import copy
import pickle

import pytest

import tilefish
from tilefish import (
    BlockScaled,
    Dim,
    DimExpr,
    InOut,
    Out,
    Tensor,
    bf16,
    cdiv,
    const,
    e2m1,
    e4m3,
    f32,
    ue8m0,
)

M = Dim("M")
K = Dim("K", divisible_by=32)
S = Dim("S", buckets=[32, 64])
D = Dim("D", 2560)
mxfp8 = BlockScaled(e4m3, ue8m0, 32)
nvfp4 = BlockScaled(e2m1, e4m3, 16, global_scale=f32)

PROTOCOLS = range(2, pickle.HIGHEST_PROTOCOL + 1)


def copies(obj):
    yield copy.deepcopy(obj)
    yield copy.copy(obj)
    for proto in PROTOCOLS:
        yield pickle.loads(pickle.dumps(obj, protocol=proto))


DIMS = [
    M,
    K,
    S,
    D,
    Dim("E", 7, divisible_by=7),
    DimExpr.const(7),
    2 * M + 3,
    M + K - 5,
    cdiv(M, 16),
    2 * cdiv(M, 16) + K - 1,
    cdiv(M + 1, K),
]


@pytest.mark.parametrize("dim", DIMS, ids=str)
def test_dims_survive_copy_and_pickle(dim):
    for other in copies(dim):
        assert other == dim
        assert hash(other) == hash(dim)
        assert str(other) == str(dim)
        assert other.is_static == dim.is_static
        assert type(other) is type(dim)
        if isinstance(dim, Dim):
            assert (other.name, other.divisible_by, other.buckets) == (dim.name, dim.divisible_by, dim.buckets)
        assert {dim: 1}[other] == 1  # usable as a dictionary key (bindings and facts are keyed by dims)


def test_a_copied_dim_still_unifies_with_the_original():
    for other in copies(M):
        assert (other + K).subst({M: 4}) == 4 + K
        assert (2 * other).free_dims() == {M}
        assert other.is_divisible_by(8, known={M: 8})
    assert Dim("E", 7).folded() == copy.deepcopy(Dim("E", 7)).folded()


def test_dimension_objects_stay_immutable_after_a_copy():
    for other in copies(K):
        with pytest.raises(AttributeError):
            other.name = "x"


@pytest.mark.parametrize("dtype", [bf16, f32, e2m1, ue8m0, mxfp8, nvfp4], ids=str)
def test_dtypes_and_formats_survive(dtype):
    for other in copies(dtype):
        assert other == dtype and hash(other) == hash(dtype) and repr(other) == repr(dtype)


@pytest.mark.parametrize(
    "ttype",
    [
        Tensor[M, D, bf16],
        Tensor[2 * M + 1, K, f32],
        Tensor[M, K, mxfp8],
        Tensor[K, nvfp4],
        Tensor[bf16],
        Tensor[4, 8, f32].as_view(),
    ],
    ids=str,
)
def test_tensor_types_survive(ttype):
    for other in copies(ttype):
        assert other == ttype and hash(other) == hash(ttype) and repr(other) == repr(ttype)
        assert other.contiguous == ttype.contiguous
        assert other.scale_shape == ttype.scale_shape
    with pytest.raises(AttributeError):
        copy.deepcopy(ttype).dtype = f32


@tilefish.library("copy_gemm", acc=f32)
def copy_gemm(a: Tensor[M, K, bf16], b: Tensor[K, K, bf16]) -> Tensor[M, K, bf16]: ...


@tilefish.kernel
def copy_kernel(x: Tensor[M, K, bf16], r: InOut[Tensor[M, K, f32]], *, eps: const[f32] = 1e-6) -> Tensor[M, K, bf16]:
    t = r + x.to(f32)
    r[...] = t
    return ((t + eps) * r).to(bf16)


Kp = Dim("Kp")  # unconstrained: the gemm call records a divisibility requirement


@tilefish.graph
def copy_graph(x: Tensor[M, 2 * Kp, bf16], w: Tensor[Kp, Kp, bf16]):
    a, b = x.split(Kp, axis=1)  # two views
    h = (a.to(f32) + b.to(f32)).to(bf16)
    return copy_gemm(h, w)


@pytest.mark.parametrize("op", [copy_kernel, copy_graph], ids=lambda o: o.name)
def test_graphs_survive_copy_and_pickle(op):
    g = op.graph
    for other in (copy.deepcopy(g), *(pickle.loads(pickle.dumps(g, protocol=p)) for p in PROTOCOLS)):
        assert str(other) == str(g)
        assert other.ops() == g.ops()
        assert [n.is_rounding_point for n in other.nodes] == [n.is_rounding_point for n in g.nodes]
        assert other.outputs[0].type == g.outputs[0].type
        assert other.requirements == g.requirements
        assert other.symbols == g.symbols
        assert other.outputs[0] is not g.outputs[0]


def test_traced_values_keep_working_after_a_copy():
    g = copy.deepcopy(copy_kernel.graph)
    r = g.inputs[1]
    # the copy is a complete trace: reads see the stored version and new nodes can still be recorded
    assert r.live is not r
    before = len(r.ctx.nodes)
    out = r.live * 2
    assert out.type == r.type and len(r.ctx.nodes) == before + 1


def test_declared_operators_are_copied_and_pickled_by_reference():
    for op in (copy_gemm, copy_kernel, copy_graph):
        assert copy.deepcopy(op) is op
        assert copy.copy(op) is op
        for proto in PROTOCOLS:
            assert pickle.loads(pickle.dumps(op, protocol=proto)) is op


def test_compile_specs_survive_copy_and_pickle():
    @tilefish.extern(triton="m:f", grid=lambda S: (S,))
    def e(x: Tensor[S, bf16], y: Out[Tensor[S, bf16]], B: const = S, E: const[f32] = float("-inf")): ...

    for spec in tilefish.aot.compile_specs(e):
        for other in copies(spec):
            assert other == spec


# -- what the package exports --------------------------------------------------------------------------------------------


def test_sum_and_max_are_not_exported_but_stay_reachable():
    assert "sum" not in tilefish.__all__ and "max" not in tilefish.__all__
    namespace: dict = {}
    exec("from tilefish import *", namespace)
    assert "sum" not in namespace and "max" not in namespace
    assert callable(tilefish.sum) and callable(tilefish.max) and callable(tilefish.mean)
    assert tilefish.sum.__name__ == "reduce_sum" and tilefish.max.__name__ == "reduce_max"


def test_no_exported_name_shadows_a_builtin():
    assert not set(tilefish.__all__) & set(dir(builtins))
    missing = [n for n in tilefish.__all__ if not hasattr(tilefish, n)]
    assert missing == []
