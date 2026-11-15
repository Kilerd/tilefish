"""Kernels of the primitives of notes 15 to 17 of ``tilefish.lower`` (``erf``, ``sin_approx``, ``rcp``, ``neg``, ``fma`` and ordered sums), shared
by the lowering tests and the gpu tests (not collected).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import tilefish
from tilefish import Dim, Tensor, bf16, f16, f32

N = Dim("N")
R = Dim("R")
Db = Dim("Db", buckets=[1, 5, 31, 96, 100, 128, 257, 1000, 1024, 4096])


@tilefish.kernel
def erf_fast(x: Tensor[N, f32]) -> Tensor[N, f32]:
    """libdevice's erff as Triton links it (flush to zero): ``gelu``'s."""
    return tilefish.erf(x)


@tilefish.kernel(math="ieee")
def erf_ieee(x: Tensor[N, f32]) -> Tensor[N, f32]:
    """CUDA's erff without flush to zero (CUDA 12.8 to 13.1)."""
    return tilefish.erf(x)


@tilefish.kernel
def sin_approx(x: Tensor[N, f32]) -> Tensor[N, f32]:
    """``sin.approx.f32``: CUDA's ``__sinf``."""
    return tilefish.sin_approx(x)


@tilefish.kernel
def rcp(x: Tensor[N, f32]) -> Tensor[N, f32]:
    """``rcp.rn.f32``: CUDA's ``__frcp_rn`` (and ``1.0f / x``)."""
    return tilefish.rcp(x)


@tilefish.kernel
def neg_f32(x: Tensor[N, f32]) -> Tensor[N, f32]:
    """CUDA's ``-x`` of an f32."""
    return -x


@tilefish.kernel
def neg_bf16(x: Tensor[N, bf16]) -> Tensor[N, bf16]:
    """CUDA's ``-x`` of a bf16."""
    return -x


@tilefish.kernel
def neg_f16(x: Tensor[N, f16]) -> Tensor[N, f16]:
    """CUDA's ``-x`` of an f16."""
    return tilefish.neg(x)


@tilefish.kernel
def fma3(a: Tensor[N, f32], b: Tensor[N, f32], c: Tensor[N, f32]) -> Tensor[N, f32]:
    """``fma.rn.f32``: one rounding."""
    return tilefish.fma(a, b, c)


@tilefish.kernel(fp_fusion=False)
def mul_add(a: Tensor[N, f32], b: Tensor[N, f32], c: Tensor[N, f32]) -> Tensor[N, f32]:
    """``a * b + c`` without contraction: two roundings (what an explicit fma is not)."""
    return a * b + c


@tilefish.kernel(fp_fusion=False)
def fma_scalar(a: Tensor[N, f32], c: Tensor[N, f32]) -> Tensor[N, f32]:
    """An fma with a scalar operand, in a kernel without contraction: still one rounding."""
    return tilefish.fma(a, 0.1, c)


# -- ordered sums (note 17) --------------------------------------------------------------------------------------------------------------------


@tilefish.kernel
def sum_sequential(x: Tensor[R, Db, f32]) -> Tensor[R, f32]:
    """A chain over the row from +0.0: one CUDA thread per row."""
    return tilefish.sum(x, -1, order="sequential")


@tilefish.kernel
def sum_halving256(x: Tensor[R, Db, f32]) -> Tensor[R, f32]:
    """256 lanes over strided columns, then a shared-memory halving tree."""
    return tilefish.sum(x, -1, order=tilefish.tree(256))


@tilefish.kernel
def sumsq_shfl128v8(x: Tensor[R, Db, f32]) -> Tensor[R, f32]:
    """128 lanes of 8 consecutive columns, ``acc = fma(x, x, acc)``, a warp-shuffle tree and then one over the 4 warps."""
    return tilefish.sum(x * x, -1, order=tilefish.tree(128, offsets=(16, 8, 4, 2, 1, 64, 32), vector=8), fma=True)


@tilefish.kernel
def dot_halving64(x: Tensor[R, Db, f32], w: Tensor[Db, f32]) -> Tensor[R, f32]:
    """A dot product with a weight row: 64 lanes of fma chains over strided columns, then the halving tree."""
    return tilefish.sum(x * w, -1, order=tilefish.tree(64), fma=True)


@tilefish.kernel
def sum_default(x: Tensor[R, Db, f32]) -> Tensor[R, f32]:
    """Triton's own tree (it depends on num_warps): what the ordered sums are told apart from."""
    return tilefish.sum(x, -1)


@tilefish.kernel(math="ieee")
def rmsnorm_ordered(x: Tensor[R, 1024, bf16], w: Tensor[1024, bf16]) -> Tensor[R, 1024, bf16]:
    """An rmsnorm as a CUDA kernel of 128 threads writes it: the sum of squares in the shuffle tree, ``div.rn`` by the width, ``rsqrtf``."""
    v = x.to(f32)
    ss = tilefish.sum(v * v, -1, keepdims=True, order=tilefish.tree(128, offsets=(16, 8, 4, 2, 1, 64, 32), vector=8), fma=True)
    r = tilefish.rsqrt(ss / 1024.0 + 1e-6)
    return ((v * r) * w.to(f32)).to(bf16)
