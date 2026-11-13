"""A generic pre-norm transformer block as a tilefish graph, for the tests of launch plans (not collected: the name has no ``test_`` prefix).

bf16 activations, rows bucketed over ``M`` in {32, 64, 128}::

    h = rmsnorm(x)                          # a lowered row kernel
    qkv = gemm_bias(h, w_qkv, b_qkv)        # a pinned cuBLASLt GEMM; q, k and v are strided views of its output
    a = attention(q, k, v, n)               # a hand-written Triton extern that reads the views and masks the keys >= n (a DevScalar)
    o = gemm(a, w_o)
    r, h2 = add_rmsnorm(x, o)               # residual add and rmsnorm in one lowered kernel
    u = gemm(h2, w_up); g = gelu(u); d = gemm(g, w_down)
    y = add(r, d)                           # the residual add

No ``from __future__ import annotations``: tilefish evaluates annotations at declaration. Declaring it imports neither torch nor triton (the attention kernel is the string
``"gpu_kernels_p2:attention_kernel"``), so a plan of it can be made on any machine; the gpu tests run it.
"""

import pathlib

import tilefish
from tilefish import DevScalar, Dim, InOut, Out, StrideOf, Strided, Tensor, bf16, const, f32, i32

D_MODEL, HEADS, D_HEAD, D_FF = 256, 4, 64, 1024
BUCKETS = [32, 64, 128]
M = Dim("M", buckets=BUCKETS)
D = Dim("D", D_MODEL)
F = Dim("F", D_FF)
S = Dim("S")


@tilefish.kernel
def rmsnorm(x: Tensor[M, D, bf16], w: Tensor[D, bf16], *, eps: const[f32] = 1e-6) -> Tensor[M, D, bf16]:
    xf = x.to(f32)
    rstd = tilefish.rsqrt(tilefish.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * rstd * w.to(f32)).to(bf16)


@tilefish.kernel
def add_rmsnorm(
    x: Tensor[M, D, bf16], o: Tensor[M, D, bf16], w: Tensor[D, bf16], *, eps: const[f32] = 1e-6
) -> tuple[Tensor[M, D, bf16], Tensor[M, D, bf16]]:
    r = (x.to(f32) + o.to(f32)).to(bf16)  # the residual stream, rounded to bf16 where it is stored
    rf = r.to(f32)
    rstd = tilefish.rsqrt(tilefish.mean(rf * rf, axis=-1, keepdims=True) + eps)
    return r, (rf * rstd * w.to(f32)).to(bf16)


@tilefish.kernel
def gelu(x: Tensor[M, F, bf16]) -> Tensor[M, F, bf16]:
    return tilefish.gelu(x.to(f32)).to(bf16)


@tilefish.kernel
def add(x: Tensor[M, D, bf16], y: Tensor[M, D, bf16]) -> Tensor[M, D, bf16]:
    return (x.to(f32) + y.to(f32)).to(bf16)


def attention_reference(q, k, v, n_valid):
    """The attention kernel in torch: softmax over the keys ``j < n`` (f32), the weighted sum of the values, rounded to bf16."""
    import torch

    n = int(n_valid.reshape(()).item())
    qf, kf, vf = q.float(), k[:n].float(), v[:n].float()
    scores = torch.einsum("shd,thd->hst", qf, kf) * 0.125
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("hst,thd->shd", p, vf).to(torch.bfloat16)


@tilefish.extern(triton="gpu_kernels_p2:attention_kernel", num_warps=4, grid=lambda S: (S, HEADS), reference=attention_reference)
def attention(
    q: Strided[Tensor[S, HEADS, D_HEAD, bf16], 16],
    k: Strided[Tensor[S, HEADS, D_HEAD, bf16], 16],
    v: Strided[Tensor[S, HEADS, D_HEAD, bf16], 16],
    o: Out[Tensor[S, HEADS, D_HEAD, bf16]],
    n_valid: DevScalar[i32],
    q_s0: StrideOf("q", 0, divisible_by=8),
    q_s1: StrideOf("q", 1, divisible_by=8),
    k_s0: StrideOf("k", 0, divisible_by=8),
    k_s1: StrideOf("k", 1, divisible_by=8),
    v_s0: StrideOf("v", 0, divisible_by=8),
    v_s1: StrideOf("v", 1, divisible_by=8),
    rows: S,
    DH: const = D_HEAD,
    BLOCK: const = 128,
): ...


@tilefish.library("gemm_qkv", acc=f32, impl="cublaslt.gemm_bias")
def gemm_qkv(a: Tensor[M, D_MODEL, bf16], b: Tensor[3 * D_MODEL, D_MODEL, bf16], bias: Tensor[3 * D_MODEL, bf16]) -> Tensor[M, 3 * D_MODEL, bf16]: ...


@tilefish.library("gemm_o", acc=f32, impl="cublaslt.gemm")
def gemm_o(a: Tensor[M, D_MODEL, bf16], b: Tensor[D_MODEL, D_MODEL, bf16]) -> Tensor[M, D_MODEL, bf16]: ...


@tilefish.library("gemm_up", acc=f32, impl="cublaslt.gemm")
def gemm_up(a: Tensor[M, D_MODEL, bf16], b: Tensor[D_FF, D_MODEL, bf16]) -> Tensor[M, D_FF, bf16]: ...


@tilefish.library("gemm_down", acc=f32, impl="cublaslt.gemm")
def gemm_down(a: Tensor[M, D_FF, bf16], b: Tensor[D_MODEL, D_FF, bf16]) -> Tensor[M, D_MODEL, bf16]: ...


@tilefish.graph
def block(
    x: Tensor[M, D, bf16],
    n: DevScalar[i32],
    w_ln1: Tensor[D, bf16],
    w_qkv: Tensor[3 * D_MODEL, D_MODEL, bf16],
    b_qkv: Tensor[3 * D_MODEL, bf16],
    w_o: Tensor[D_MODEL, D_MODEL, bf16],
    w_ln2: Tensor[D, bf16],
    w_up: Tensor[D_FF, D_MODEL, bf16],
    w_down: Tensor[D_MODEL, D_FF, bf16],
) -> Tensor[M, D, bf16]:
    h = rmsnorm(x, w_ln1)
    qkv = gemm_qkv(h, w_qkv, b_qkv)
    qkv4 = qkv.reshape(M, 3, HEADS, D_HEAD)
    q, k, v = qkv4[:, 0], qkv4[:, 1], qkv4[:, 2]  # views of the fused output: nothing is copied
    a = attention(q, k, v, n)
    o = gemm_o(a.reshape(M, D_MODEL), w_o)
    r, h2 = add_rmsnorm(x, o, w_ln2)
    u = gemm_up(h2, w_up)
    g = gelu(u)
    d = gemm_down(g, w_down)
    return add(r, d)


def deep_block(layers):
    """The block repeated ``layers`` times in one graph (the layers are unrolled: a plan has no nested graphs), with the weights of the layers stacked along the first axis of one
    parameter each (``w_qkv`` is ``[layers * 768, 256]``), so a layer takes its weights as chunks of those: views of a parameter, which cost nothing."""
    L = layers

    @tilefish.graph(name=f"deep{L}")
    def deep(
        x: Tensor[M, D, bf16],
        n: DevScalar[i32],
        w_ln1: Tensor[L * D_MODEL, bf16],
        w_qkv: Tensor[L * 3 * D_MODEL, D_MODEL, bf16],
        b_qkv: Tensor[L * 3 * D_MODEL, bf16],
        w_o: Tensor[L * D_MODEL, D_MODEL, bf16],
        w_ln2: Tensor[L * D_MODEL, bf16],
        w_up: Tensor[L * D_FF, D_MODEL, bf16],
        w_down: Tensor[L * D_MODEL, D_FF, bf16],
    ) -> Tensor[M, D, bf16]:
        ln1, ln2 = w_ln1.split(D_MODEL, axis=0), w_ln2.split(D_MODEL, axis=0)
        qkv_w, qkv_b = w_qkv.split(3 * D_MODEL, axis=0), b_qkv.split(3 * D_MODEL, axis=0)
        o_w, up_w, down_w = w_o.split(D_MODEL, axis=0), w_up.split(D_FF, axis=0), w_down.split(D_MODEL, axis=0)
        for i in range(L):
            h = rmsnorm(x, ln1[i])
            qkv = gemm_qkv(h, qkv_w[i], qkv_b[i])
            qkv4 = qkv.reshape(M, 3, HEADS, D_HEAD)
            a = attention(qkv4[:, 0], qkv4[:, 1], qkv4[:, 2], n)
            o = gemm_o(a.reshape(M, D_MODEL), o_w[i])
            r, h2 = add_rmsnorm(x, o, ln2[i])
            x = add(r, gemm_down(gelu(gemm_up(h2, up_w[i])), down_w[i]))
        return x

    return deep


def stacked_weights(layers, seed=0, device="cuda"):
    """Per-layer weights (as :func:`weights` makes them, one seed per layer) and their stacked versions for :func:`deep_block`."""
    import torch

    per_layer = [weights(seed + 100 * i, device) for i in range(layers)]
    stacked = {k: torch.cat([w[k] for w in per_layer]) for k in WEIGHT_NAMES}
    return per_layer, stacked


# -- a small graph with an in-place tensor: f32, so its results are exact ------------------------------------------------------------------------------------

Q = Dim("Q", buckets=[16, 32])
W = Dim("W", 128)


@tilefish.kernel
def bump(state: InOut[Tensor[Q, W, f32]], y: Tensor[Q, W, f32]):
    state[...] = state + y


@tilefish.kernel
def twice(x: Tensor[Q, W, f32]) -> Tensor[Q, W, f32]:
    return x * 2.0


@tilefish.kernel
def plus(x: Tensor[Q, W, f32], y: Tensor[Q, W, f32]) -> Tensor[Q, W, f32]:
    return x + y


@tilefish.graph
def accumulate(a: Tensor[Q, W, f32], state: InOut[Tensor[Q, W, f32]]) -> Tensor[Q, W, f32]:
    """``state += a`` twice, an intermediate ``t = 2a`` updated in place (``t += a``), and ``t + state``: in-place calls on a parameter and on a result."""
    bump(state, a)
    bump(state, a)
    t = twice(a)
    bump(t, a)
    return plus(t.live, state)


@tilefish.kernel
def with_out(x: Tensor[Q, W, f32], y: Out[Tensor[Q, W, f32]]):
    y[...] = x * 2.0


@tilefish.kernel
def both(x: Tensor[Q, W, f32], y: Out[Tensor[Q, W, f32]]) -> Tensor[Q, W, f32]:
    y[...] = x * 3.0
    return x + 1.0


@tilefish.graph
def two_outputs(a: Tensor[Q, W, f32]) -> tuple[Tensor[Q, W, f32], Tensor[Q, W, f32]]:
    """A kernel with an ``Out`` parameter, one with an ``Out`` parameter and a returned value, and two returned values in an order of their own."""
    t = with_out(a)
    r, o = both(a)
    return plus(t, o), r


# -- a graph of prebuilt binaries: hand-written PTX, a fixed float argument, a 64-bit stride scalar ------------------------------------------------------

PTX = pathlib.Path(__file__).resolve().parent / "ptx"
B = Dim("B", buckets=[8, 24])
Nn = Dim("Nn")


@tilefish.extern(ptx=PTX / "axpy_f32.ptx", symbol="axpy_f32", block=(128,), grid=lambda Nn: (tilefish.cdiv(Nn, 128),))
def axpy(x: Tensor[Nn, f32], y: Tensor[Nn, f32], out: Out[Tensor[Nn, f32]], n_elem: Nn, alpha: const[f32] = 2.0): ...


@tilefish.extern(ptx=PTX / "strided_copy_bf16.ptx", symbol="strided_copy_bf16", block=(64,), grid=lambda B: (B,), index_bits=64)
def strided_copy(x: Strided[Tensor[B, 64, bf16]], y: Out[Tensor[B, 64, bf16]], *, cols: const[i32] = 64, x_s0: StrideOf("x", 0)): ...


@tilefish.graph
def prebuilt(x: Tensor[B, 64, f32], y: Tensor[B, 64, f32], w: Tensor[B, 128, bf16]) -> tuple[Tensor[B * 64, f32], Tensor[B, 64, bf16]]:
    z = axpy(x.reshape(B * 64), y.reshape(B * 64))  # a prebuilt PTX with a fixed float argument, on views
    left, _ = w.split(64, axis=-1)  # a column block of w: a view with a row stride of 128
    return z, strided_copy(left)  # a prebuilt PTX with a 64-bit stride scalar


# the operators of the graph, in the order they are called (what a per-op eager run of the block calls)
OPS = [rmsnorm, gemm_qkv, attention, gemm_o, add_rmsnorm, gemm_up, gelu, gemm_down, add]
WEIGHT_NAMES = ["w_ln1", "w_qkv", "b_qkv", "w_o", "w_ln2", "w_up", "w_down"]


def weights(seed=0, device="cuda"):
    """Random weights of the block, bf16, by parameter name (a generator of their own: the same on every machine)."""
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def rnd(shape, scale):
        return (torch.randn(shape, generator=gen, device=device, dtype=torch.float32) * scale).to(torch.bfloat16)

    return {
        "w_ln1": (1 + rnd((D_MODEL,), 0.1)),
        "w_qkv": rnd((3 * D_MODEL, D_MODEL), D_MODEL**-0.5),
        "b_qkv": rnd((3 * D_MODEL,), 0.1),
        "w_o": rnd((D_MODEL, D_MODEL), D_MODEL**-0.5),
        "w_ln2": (1 + rnd((D_MODEL,), 0.1)),
        "w_up": rnd((D_FF, D_MODEL), D_MODEL**-0.5),
        "w_down": rnd((D_MODEL, D_FF), D_FF**-0.5),
    }


def activations(m, seed=1, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn((m, D_MODEL), generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)
