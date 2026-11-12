"""A generic pre-norm transformer block, typed end to end.

The hand-written kernels live in a hypothetical ``my_kernels`` module (Triton
functions); here they are only *referred to* by name, so this file needs no
GPU, no torch and no triton. Run it to trace the graphs and print the typed
IR and the Triton ahead-of-time compile specs derived from the declarations:

    uv run --project python python python/examples/transformer_block.py
"""

import tilefish
from tilefish import BlockScaled, Dim, InOut, Out, Tensor, bf16, const, e4m3, f32, ue8m0

T = Dim("T")  # tokens in the step: a runtime dimension
D = Dim("D", 2048)  # model width
F = Dim("F", 5504)  # MLP hidden width

# Library GEMMs take a K that is a multiple of 32 and an N that is a multiple of 128.
K = Dim("K", divisible_by=32)
N = Dim("N", divisible_by=128)
M = Dim("M")

mxfp8 = BlockScaled(elem=e4m3, scale=ue8m0, block=32, axis=-1)


# -- hand-written Triton kernels, given types -----------------------------------------------------


@tilefish.extern(triton="my_kernels:rmsnorm", num_warps=8, grid=lambda T: (T,))
def rmsnorm(
    x: Tensor[T, D, bf16],
    w: Tensor[D, bf16],
    y: Out[Tensor[T, D, bf16]],
    n_cols: const = D,
    BLOCK: const = 2048,
    eps: const[f32] = 1e-6,
): ...


@tilefish.extern(triton="my_kernels:add_rmsnorm", num_warps=8, grid=lambda T: (T,))
def add_rmsnorm(
    x: Tensor[T, D, bf16],
    residual: InOut[Tensor[T, D, bf16]],  # residual += x, then y = rmsnorm(residual)
    w: Tensor[D, bf16],
    y: Out[Tensor[T, D, bf16]],
    n_cols: const = D,
    BLOCK: const = 2048,
    eps: const[f32] = 1e-6,
): ...


@tilefish.extern(triton="my_kernels:gelu", num_warps=4, grid=lambda n_elem, BLOCK: (tilefish.cdiv(n_elem, BLOCK),))
def gelu(
    x: Tensor[T, F, bf16],
    y: Out[Tensor[T, F, bf16]],
    n_elem: F * T,  # a runtime i32: F is static, so this is 5504 * T, a multiple of 16
    BLOCK: const = 1024,
): ...


@tilefish.extern(triton="my_kernels:swiglu", num_warps=4, grid=lambda T, F: (T, tilefish.cdiv(F, 256)))
def swiglu(
    gate_up: Tensor[T, 2 * F, bf16],
    y: Out[Tensor[T, F, bf16]],
    BLOCK_F: const = 256,
): ...


@tilefish.extern(triton="my_kernels:attention", num_warps=4, num_stages=2, grid=lambda T: (T, 16))
def attention(
    q: Tensor[T, D, bf16],
    k: Tensor[T, D, bf16],
    v: Tensor[T, D, bf16],
    o: Out[Tensor[T, D, bf16]],
    n_tokens: T,
    HEAD_DIM: const = 128,
    sm_scale: const[f32] = 0.08838834764831845,
): ...


# -- library GEMMs: opaque bodies, typed signatures ------------------------------------------------


@tilefish.library("gemm", acc=f32)
def gemm(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16]) -> Tensor[M, N, bf16]: ...


@tilefish.library("gemm_mxfp8", acc=f32)
def gemm_mxfp8(a: Tensor[M, K, bf16], b: Tensor[N, K, mxfp8]) -> Tensor[M, N, bf16]: ...


# -- the model: ordinary Python over typed values ---------------------------------------------------


def attention_sublayer(x, w_norm, w_qkv, w_o):
    h = rmsnorm(x, w_norm)
    q, k, v = gemm(h, w_qkv).split(D, axis=-1)
    return gemm(attention(q, k, v), w_o)


@tilefish.graph
def transformer_block(
    x: InOut[Tensor[T, D, bf16]],  # the residual stream, updated in place
    w_norm1: Tensor[D, bf16],
    w_qkv: Tensor[3 * D, D, bf16],
    w_o: Tensor[D, D, bf16],
    w_norm2: Tensor[D, bf16],
    w_gate_up: Tensor[2 * F, D, bf16],
    w_down: Tensor[D, F, mxfp8],
):
    attn = attention_sublayer(x, w_norm1, w_qkv, w_o)
    h = add_rmsnorm(attn, x, w_norm2)  # x += attn; h = rmsnorm(x)
    mlp = gemm_mxfp8(swiglu(gemm(h, w_gate_up)), w_down)
    x[...] = x + mlp
    return x


@tilefish.graph
def transformer_block_gelu(
    x: InOut[Tensor[T, D, bf16]],
    w_norm1: Tensor[D, bf16],
    w_qkv: Tensor[3 * D, D, bf16],
    w_o: Tensor[D, D, bf16],
    w_norm2: Tensor[D, bf16],
    w_up: Tensor[F, D, bf16],
    w_down: Tensor[D, F, mxfp8],
):
    attn = attention_sublayer(x, w_norm1, w_qkv, w_o)
    h = add_rmsnorm(attn, x, w_norm2)
    x[...] = x + gemm_mxfp8(gelu(gemm(h, w_up)), w_down)
    return x


EXTERNS = (rmsnorm, add_rmsnorm, gelu, swiglu, attention)


if __name__ == "__main__":
    print(transformer_block.graph)
    print()
    for op in EXTERNS:
        for spec in tilefish.aot.compile_specs(op):
            print(f"{spec.stem}: {spec.signature} grid={spec.grid} block={spec.block}")
