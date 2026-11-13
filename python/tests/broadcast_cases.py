"""Elementwise kernels with an operand broadcast over the leading axes, shared by the lowering tests and the gpu tests (not collected).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import tilefish
from tilefish import Dim, InOut, Tensor, bf16, f32

K = Dim("K")
T = Dim("T")
D = Dim("D")
R = Dim("R")


@tilefish.kernel
def pos_add(y: InOut[Tensor[K, T, D, bf16]], pe: Tensor[T, D, bf16]):
    """A position table added to every chunk of a batch, in place: one f32 add, one rounding."""
    y[...] = (y.to(f32) + pe.to(f32)).to(bf16)


@tilefish.kernel
def scale_cols(x: Tensor[R, D, f32], w: Tensor[D, f32], b: Tensor[D, f32]) -> Tensor[R, D, f32]:
    """A dynamic row length: the row template cannot take it, the elementwise template broadcasts w and b."""
    return x * w + b
