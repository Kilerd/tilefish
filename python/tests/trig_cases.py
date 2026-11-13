"""Kernels with ``sin`` and ``cos``, shared by the lowering tests and the gpu tests (not collected).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import tilefish
from tilefish import Dim, Tensor, bf16, f32, i32

N = Dim("N")
S = Dim("S")
H2 = Dim("H2", 128)  # twice the number of frequencies: each angle is stored twice, as rotary tables are


@tilefish.kernel
def sin_cos(x: Tensor[N, f32]) -> tuple[Tensor[N, f32], Tensor[N, f32]]:
    return tilefish.sin(x), tilefish.cos(x)


@tilefish.kernel
def rope_tables(inv_freq: Tensor[H2, f32], positions: Tensor[S, 1, i32]) -> tuple[Tensor[S, H2, bf16], Tensor[S, H2, bf16]]:
    """Rotary tables: the angle ``inv_freq[i] * f32(position)`` (one multiply), its cosine and sine rounded to bf16."""
    angle = inv_freq * positions.to(f32)
    return tilefish.cos(angle).to(bf16), tilefish.sin(angle).to(bf16)
