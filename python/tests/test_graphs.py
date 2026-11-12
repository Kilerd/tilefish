"""The graph and kernel from the design README, written out in full."""

import tilefish
from tilefish import (
    BlockScaled,
    Dim,
    InOut,
    Tensor,
    bf16,
    const,
    e4m3,
    f32,
    ue8m0,
)

M = Dim("M")
D = Dim("D", 2560)
H = Dim("H", 1024)
F = Dim("F", 3072)
K = Dim("K", divisible_by=32)
N = Dim("N", divisible_by=128)

mxfp8 = BlockScaled(elem=e4m3, scale=ue8m0, block=32, axis=-1)


@tilefish.library("gemm", acc=f32)
def gemm(a: Tensor[M, K, bf16], b: Tensor[N, K, mxfp8]) -> Tensor[M, N, bf16]: ...


@tilefish.graph
def mlp(x: Tensor[M, H, bf16], w_gu: Tensor[2 * F, H, mxfp8], w_d: Tensor[H, F, mxfp8]):
    g, u = gemm(x, w_gu).split(F, axis=-1)
    a = (tilefish.silu(g.to(f32)) * u.to(f32)).to(bf16)
    return gemm(a, w_d)


@tilefish.kernel
def residual_add_rmsnorm(
    x: Tensor[M, D, bf16],
    residual: InOut[Tensor[M, D, f32]],
    w: Tensor[D, bf16],
    *,
    eps: const[f32] = 1e-6,
) -> Tensor[M, D, bf16]:
    r = residual + x.to(f32)
    residual[...] = r
    rstd = tilefish.rsqrt(tilefish.mean(r * r, axis=-1, keepdims=True) + eps)
    return (r * rstd * w.to(f32)).to(bf16)


def test_mlp_traces_to_the_expected_node_list():
    g = mlp.graph
    assert g.kind == "graph" and g.name == "mlp"
    assert g.ops() == ["gemm", "split", "cast", "silu", "cast", "mul", "cast", "gemm"]

    gemm_up, split, cast_g, silu, cast_u, mul, cast_a, gemm_down = g.nodes
    x, w_gu, w_d = g.inputs

    assert gemm_up.inputs == (x, w_gu)
    assert gemm_up.output_types == (Tensor[M, 6144, bf16],)
    assert gemm_up.attrs["acc"] is f32

    assert split.inputs == gemm_up.outputs
    assert split.attrs == {"axis": 1, "sizes": (F, F)}
    assert split.output_types == (Tensor[M, F, bf16], Tensor[M, F, bf16])

    assert cast_g.inputs == (split.outputs[0],) and cast_g.output_types == (Tensor[M, F, f32],)
    assert silu.inputs == cast_g.outputs and silu.output_types == (Tensor[M, F, f32],)
    assert cast_u.inputs == (split.outputs[1],) and cast_u.output_types == (Tensor[M, F, f32],)
    assert mul.inputs == (silu.outputs[0], cast_u.outputs[0]) and mul.output_types == (Tensor[M, F, f32],)
    assert cast_a.inputs == mul.outputs and cast_a.output_types == (Tensor[M, F, bf16],)
    assert cast_a.attrs == {"to": bf16, "rounding": True}

    assert gemm_down.inputs == (cast_a.outputs[0], w_d)
    assert gemm_down.output_types == (Tensor[M, H, bf16],)
    assert g.outputs == gemm_down.outputs
    assert g.requirements == ()
    assert g.symbols == (M,)


def test_mlp_signature_and_text():
    assert mlp.signature.returns == (Tensor[M, H, bf16],)
    text = str(mlp.graph)
    assert text.startswith("graph mlp(%x: Tensor[M, H, bf16], %w_gu: Tensor[6144, H, blockscaled(e4m3, ue8m0, 32)]")
    assert "%1: Tensor[M, F, bf16], %2: Tensor[M, F, bf16] = split(%0) {axis=1, sizes=(F, F)}" in text
    assert "%7: Tensor[M, F, bf16] = cast(%6) {to=bf16, rounding=true}  # rounding point" in text
    assert text.endswith("return %8")


def test_residual_add_rmsnorm_has_exactly_one_rounding_point():
    g = residual_add_rmsnorm.graph
    assert g.kind == "kernel"
    points = g.rounding_points()
    assert len(points) == 1
    (final,) = points
    assert final.attrs == {"to": bf16, "rounding": True}
    assert final.outputs == g.outputs
    assert g.outputs[0].type == Tensor[M, D, bf16]
    # the two widening casts of the inputs are casts but not rounding points
    casts = [n for n in g.nodes if n.op == "cast"]
    assert len(casts) == 3 and [n.is_rounding_point for n in casts] == [False, False, True]


def test_residual_add_rmsnorm_node_list():
    g = residual_add_rmsnorm.graph
    assert g.ops() == ["cast", "add", "store", "mul", "mean", "add", "rsqrt", "mul", "cast", "mul", "cast"]
    store = g.nodes[2]
    residual = g.inputs[1]
    assert store.inputs[0] is residual and store.attrs == {"target": "residual"}
    mean = g.nodes[4]
    assert mean.attrs == {"axes": (1,), "keepdims": True}
    assert mean.output_types == (Tensor[M, 1, f32],)
    # eps is a const parameter that adopted f32
    eps_add = g.nodes[5]
    assert eps_add.inputs[1].name == "eps" and eps_add.inputs[1].dtype is f32
    assert "eps: const[f32] = 1e-06" in str(g)


def test_kernel_is_not_lowered_but_is_a_typed_operator():
    assert residual_add_rmsnorm.signature.returns == (Tensor[M, D, bf16],)
    assert [p.name for p in residual_add_rmsnorm.signature.params] == ["x", "residual", "w", "eps"]
