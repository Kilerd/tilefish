"""Kernels shared by the lowering tests (not collected: the name has no ``test_`` prefix).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import tilefish
from tilefish import Dim, Tensor, bf16, const, f32

M = Dim("M")
T = Dim("T", divisible_by=16)
D = Dim("D", 768)
F = Dim("F", 3072)
V = Dim("V", 1000)


@tilefish.kernel
def layernorm(
    x: Tensor[M, D, bf16],
    w: Tensor[D, bf16],
    b: Tensor[D, bf16],
    *,
    eps: const[f32] = 1e-5,
) -> Tensor[M, D, bf16]:
    xf = x.to(f32)
    mean = tilefish.mean(xf, axis=-1, keepdims=True)
    d = xf - mean
    var = tilefish.mean(d * d, axis=-1, keepdims=True)
    rstd = tilefish.rsqrt(var + eps)
    y = d * rstd * w.to(f32) + b.to(f32)
    return y.to(bf16)


@tilefish.kernel
def gelu_exact(x: Tensor[T, bf16]) -> Tensor[T, bf16]:
    return tilefish.gelu(x.to(f32)).to(bf16)


@tilefish.kernel
def swiglu(gate: Tensor[M, F, bf16], up: Tensor[M, F, bf16]) -> Tensor[M, F, bf16]:
    return (tilefish.silu(gate.to(f32)) * up.to(f32)).to(bf16)


@tilefish.kernel
def softmax(x: Tensor[M, V, f32]) -> Tensor[M, V, f32]:
    m = tilefish.max(x, axis=-1, keepdims=True)
    e = tilefish.exp(x - m)
    s = tilefish.sum(e, axis=-1, keepdims=True)
    return e / s
