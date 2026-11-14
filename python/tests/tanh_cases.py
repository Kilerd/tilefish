"""Kernels of ``tilefish.tanh`` (note 14 of ``tilefish.lower``), shared by the lowering tests and the gpu tests (not collected).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import tilefish
from tilefish import Dim, Tensor, bf16, f32

N = Dim("N")
R = Dim("R")


@tilefish.kernel
def tanh_fast(x: Tensor[N, f32]) -> Tensor[N, f32]:
    """libdevice's tanhf, built with flush to zero."""
    return tilefish.tanh(x)


@tilefish.kernel(math="ieee")
def tanh_ieee(x: Tensor[N, f32]) -> Tensor[N, f32]:
    """The same with a subnormal argument passed through: torch's tanh on every f32."""
    return tilefish.tanh(x)


@tilefish.kernel
def tanh_rows(x: Tensor[R, 64, bf16], b: Tensor[64, bf16]) -> Tensor[R, 64, bf16]:
    """A bias and a tanh of a bf16 activation in f32, one rounding back to bf16 (the row template)."""
    return tilefish.tanh(x.to(f32) + b.to(f32), math="ieee").to(bf16)
