"""Kernels with ``gather_rows`` and ``scatter_rows``, shared by the lowering tests and the gpu tests (not collected).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import tilefish
from tilefish import DevScalar, Dim, InOut, Tensor, bf16, f32, i32, i64, u8

V = Dim("V")  # rows of a table: dynamic (a runtime i32 parameter)
Vb = Dim("Vb", buckets=[64, 200])  # bucketed (a constexpr per bucket)
R = Dim("R")
N = Dim("N")
D = Dim("D", 128)


@tilefish.kernel
def embed_lookup(table: Tensor[V, D, bf16], ids: Tensor[R, i64]) -> Tensor[R, D, bf16]:
    return tilefish.gather_rows(table, ids, oob="zero")


@tilefish.kernel
def embed_masked(table: Tensor[V, D, bf16], ids: Tensor[R, i64], n: DevScalar[i32]) -> Tensor[R, D, bf16]:
    """A lookup whose rows from ``n`` on are ``x * 0.0`` (signed zeros, NaN for inf and NaN): a gather and a mask in one launch."""
    rows = tilefish.gather_rows(table, ids, oob="zero")
    keep = tilefish.iota(rows, axis=0) < tilefish.load(n)
    return tilefish.where(keep, rows, (rows.to(f32) * 0.0).to(bf16))


@tilefish.kernel
def take_clamp(table: Tensor[Vb, D, f32], idx: Tensor[R, i32]) -> Tensor[R, D, f32]:
    return tilefish.gather_rows(table, idx, oob="clamp")


@tilefish.kernel
def take_static(table: Tensor[1000, D, bf16], idx: Tensor[R, u8]) -> Tensor[R, D, bf16]:
    return tilefish.gather_rows(table, idx, oob="zero")


@tilefish.kernel
def take_elems(x: Tensor[V, f32], idx: Tensor[N, u8]) -> tuple[Tensor[N, f32], Tensor[N, f32]]:
    return tilefish.gather_rows(x, idx, oob="clamp"), tilefish.gather_rows(x, idx, oob="zero")


@tilefish.kernel
def take_elems_i64(x: Tensor[V, bf16], idx: Tensor[N, i64]) -> Tensor[N, bf16]:
    return tilefish.gather_rows(x, idx, oob="zero")


@tilefish.kernel
def scale_rows(x: Tensor[R, D, f32], scales: Tensor[V, f32], ids: Tensor[R, 1, i32]) -> Tensor[R, D, f32]:
    """A per-row value gathered from an ``[N]`` table (one element per row program), then broadcast over the row."""
    return x * tilefish.gather_rows(scales, ids, oob="zero")


@tilefish.kernel
def index_copy(dst: InOut[Tensor[V, D, bf16]], idx: Tensor[R, i64], src: Tensor[R, D, bf16]):
    tilefish.scatter_rows(dst, idx, src, oob="skip")


@tilefish.kernel
def scatter_elems(dst: InOut[Tensor[V, f32]], idx: Tensor[N, i32], src: Tensor[N, f32]):
    tilefish.scatter_rows(dst, idx, src, oob="skip")


@tilefish.kernel
def permute_rows(x: Tensor[R, D, bf16], perm: Tensor[R, i32]) -> Tensor[R, D, bf16]:
    """A table that is also an ordinary [R, D] input: gathered and read row by row."""
    y = tilefish.gather_rows(x, perm, oob="clamp")
    return tilefish.where(y != y, x, y)
