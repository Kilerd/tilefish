"""Kernels in IEEE math (``math="ieee"``, note 13 of ``tilefish.lower``), shared by the lowering tests and the gpu tests (not collected).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import tilefish
from tilefish import Dim, Tensor, f32

N = Dim("N")
R = Dim("R")
Db = Dim("Db", buckets=[96, 1000])


@tilefish.kernel(math="ieee")
def div_ieee(x: Tensor[N, f32], y: Tensor[N, f32]) -> Tensor[N, f32]:
    """IEEE 754 division: ``div.rn.f32``."""
    return x / y


@tilefish.kernel(math="ieee")
def div_by_three_ieee(x: Tensor[N, f32]) -> Tensor[N, f32]:
    """A constant divisor is divided by too (not multiplied by its reciprocal)."""
    return x / 3.0


@tilefish.kernel(math="ieee")
def math_ieee(x: Tensor[N, f32]) -> tuple[Tensor[N, f32], Tensor[N, f32], Tensor[N, f32], Tensor[N, f32], Tensor[N, f32]]:
    """Every op that IEEE math changes, on one argument: CUDA's expf, sqrtf, rsqrtf, sinf and torch's silu, none flushing a subnormal."""
    return tilefish.exp(x), tilefish.sqrt(x), tilefish.rsqrt(x), tilefish.sin(x), tilefish.silu(x)


@tilefish.kernel
def math_fast(x: Tensor[N, f32]) -> tuple[Tensor[N, f32], Tensor[N, f32], Tensor[N, f32], Tensor[N, f32], Tensor[N, f32]]:
    """The same in the default math: what the tests contrast IEEE math with."""
    return tilefish.exp(x), tilefish.sqrt(x), tilefish.rsqrt(x), tilefish.sin(x), tilefish.silu(x)


@tilefish.kernel
def per_op(x: Tensor[N, f32], y: Tensor[N, f32]) -> tuple[Tensor[N, f32], Tensor[N, f32], Tensor[N, f32], Tensor[N, f32]]:
    """A kernel of the default math whose first two ops say IEEE math at the call; the last two are the default's."""
    return tilefish.div(x, y, math="ieee"), tilefish.exp(x, math="ieee"), x / y, tilefish.exp(x)


@tilefish.kernel(math="ieee", fp_fusion=False)
def periodic(x: Tensor[R, Db, f32], a: Tensor[Db, f32]) -> Tensor[R, Db, f32]:
    """``x + (1 / (a + 1e-9)) * sin(a * x)**2``, a periodic activation with a learned frequency per channel: in IEEE math and without
    contraction every op rounds where torch's eager ops round, so the result is torch's op by op."""
    s = tilefish.sin(a * x)
    return x + (1.0 / (a + 1e-9)) * (s * s)


@tilefish.kernel(math="ieee")
def softmax_ieee(x: Tensor[R, Db, f32]) -> Tensor[R, Db, f32]:
    """A row softmax in IEEE math: ``expf`` and an IEEE division by the row's sum (the sum's tree is Triton's)."""
    e = tilefish.exp(x - tilefish.max(x, axis=-1, keepdims=True))
    return e / tilefish.sum(e, axis=-1, keepdims=True)
