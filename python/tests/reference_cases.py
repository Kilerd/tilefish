"""The golden kernels of ``lower_cases.py`` in every float dtype (f32, bf16, f16 in and out; f32 math inside), for the tolerance matrix of the
``reference`` mode (``test_gpu_reference.py``). Not collected.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import tilefish
from tilefish import Dim, Tensor, bf16, const, f16, f32

M = Dim("M")
T = Dim("T", divisible_by=16)
D = Dim("D", 768)
F = Dim("F", 3072)
V = Dim("V", 1000)
DTYPES = {"f32": f32, "bf16": bf16, "f16": f16}


def _layernorm(dt):
    @tilefish.kernel(name=f"layernorm_{dt}")
    def layernorm(x: Tensor[M, D, dt], w: Tensor[D, dt], b: Tensor[D, dt], *, eps: const[f32] = 1e-5) -> Tensor[M, D, dt]:
        xf = x.to(f32)
        mean = tilefish.mean(xf, axis=-1, keepdims=True)
        d = xf - mean
        var = tilefish.mean(d * d, axis=-1, keepdims=True)
        y = d * tilefish.rsqrt(var + eps) * w.to(f32) + b.to(f32)
        return y.to(dt)

    return layernorm


def _gelu(dt):
    @tilefish.kernel(name=f"gelu_{dt}")
    def gelu(x: Tensor[T, dt]) -> Tensor[T, dt]:
        return tilefish.gelu(x.to(f32)).to(dt)

    return gelu


def _swiglu(dt):
    @tilefish.kernel(name=f"swiglu_{dt}")
    def swiglu(gate: Tensor[M, F, dt], up: Tensor[M, F, dt]) -> Tensor[M, F, dt]:
        return (tilefish.silu(gate.to(f32)) * up.to(f32)).to(dt)

    return swiglu


def _softmax(dt):
    @tilefish.kernel(name=f"softmax_{dt}")
    def softmax(x: Tensor[M, V, dt]) -> Tensor[M, V, dt]:
        xf = x.to(f32)
        e = tilefish.exp(xf - tilefish.max(xf, axis=-1, keepdims=True))
        return (e / tilefish.sum(e, axis=-1, keepdims=True)).to(dt)

    return softmax


KERNELS = {(name, dt): make(t) for name, make in (("layernorm", _layernorm), ("gelu", _gelu), ("swiglu", _swiglu), ("softmax", _softmax))
           for dt, t in DTYPES.items()}  # fmt: skip
