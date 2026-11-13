"""Kernels with the two NaN modes of the f32 to bf16 cast, shared by the lowering tests and the gpu tests (not collected).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import tilefish
from tilefish import Dim, Tensor, bf16, f32

N = Dim("N")
R = Dim("R")
D = Dim("D", 1024)


@tilefish.kernel
def to_bf16(x: Tensor[N, f32]) -> tuple[Tensor[N, bf16], Tensor[N, bf16]]:
    return x.to(bf16), x.to(bf16, nan="preserve")


@tilefish.kernel
def to_bf16_rows(x: Tensor[R, D, f32], scale: Tensor[D, f32]) -> Tensor[R, D, bf16]:
    """A preserving cast of an input row next to arithmetic (the product's NaN is canonical either way)."""
    return tilefish.where(scale != 0.0, (x * scale).to(bf16), x.to(bf16, nan="preserve"))
