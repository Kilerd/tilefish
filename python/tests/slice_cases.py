"""Kernels that read static slices ``x[k]`` of a parameter in the row template (note 19 of ``tilefish.lower``), shared by the lowering tests and
the gpu tests (not collected).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import tilefish
from tilefish import Dim, Tensor, bf16, f32, i32, i64

R = Dim("R")  # rows: dynamic, the grid
N = Dim("N")  # rows of a table: dynamic
Nb = Dim("Nb", buckets=[64, 200])  # rows of a table: bucketed
D = Dim("D", 128)
C = Dim("C", 3000)
BOOKS = 4


@tilefish.kernel
def vq_lookup(codes: Tensor[BOOKS, R, i32], t0: Tensor[N, D, f32], t1: Tensor[N, D, f32], t2: Tensor[N, D, f32], t3: Tensor[N, D, f32]) -> Tensor[R, D, f32]:
    """A vector-quantizer lookup: per row ``r``, the sum over the books ``k`` of ``t_k[codes[k, r]]``, the codes one ``[K, R]`` plane."""
    acc = tilefish.gather_rows(t0, codes[0], oob="zero")
    for k, t in enumerate((t1, t2, t3), start=1):
        acc = acc + tilefish.gather_rows(t, codes[k], oob="zero")
    return acc


@tilefish.kernel
def vq_tables(codes: Tensor[BOOKS, R, i64], tables: Tensor[BOOKS, Nb, D, f32]) -> Tensor[R, D, f32]:
    """The same with the tables one ``[K, N, D]`` parameter: ``tables[k]`` is a table at a static offset, ``N`` bucketed."""
    acc = tilefish.gather_rows(tables[0], codes[0], oob="clamp")
    for k in range(1, BOOKS):
        acc = acc + tilefish.gather_rows(tables[k], codes.select(0, k), oob="clamp")
    return acc


@tilefish.kernel(fp_fusion=False)
def plane_rows(x: Tensor[R, D, f32], w: Tensor[3, R, 1, f32], bias: Tensor[2, D, f32], codes: Tensor[BOOKS, R, i32]) -> tuple[Tensor[R, D, f32], Tensor[R, i32]]:
    """Per-row operands from a ``[3, R, 1]`` plane, per-column ones from a ``[2, D]`` parameter, and a per-row result from a ``[K, R]`` plane
    (without contraction, so the reference rounds where the binary does)."""
    return x * w[1] + bias[0] - w[2] * bias[1], codes[3] - codes[0]


@tilefish.kernel
def slab_max(xs: Tensor[2, R, D, bf16]) -> Tensor[R, D, bf16]:
    """Two ``[R, D]`` slabs of one ``[2, R, D]`` parameter, compared element by element."""
    return tilefish.where(xs[0] > xs[1], xs[0], xs[1])


@tilefish.kernel
def pick_argmax(logits: Tensor[2, R, C, bf16]) -> Tensor[R, i64]:
    """An argmax of the second ``[R, C]`` slab, a long row walked in chunks (note 8): the slab is read chunk by chunk at its offset."""
    return tilefish.argmax(logits[1], dtype=i64)
