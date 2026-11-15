"""The kernels and ops of the gpu tests of implementations per architecture with names, prebuilt binaries and ABIs of their own (not collected: the
name has no ``test_`` prefix).

Written from scratch for these tests, and public. Each op is ``copy2d``: ``y = x`` for a view ``x`` of rows of f16 (a column block of a wider tensor,
or a dense tensor), one public signature with implementations that take it in other ways:

* ``copy_tma``: a Triton kernel that moves 64x64 tiles through two TMA descriptors (``x`` a descriptor of the view, ``y`` of the result), its own stub;
* ``copy_rows``: a Triton kernel that copies a row per program through pointers, the column count and the row stride ``i32`` scalars: the op's stub;
* ``ptx/strided_copy_bf16.ptx``: a hand-written PTX copy of 16-bit elements, a row per block, the row stride a 64-bit scalar: its own stub, its C ABI.

The ops differ in which architecture runs which: ``copy2d`` runs the TMA form (named ``tma``) on sm_100 and sm_103 and the pointer form elsewhere;
``copy2d_ptx`` the PTX (named ``ptx``) on sm_100 and sm_103; ``copy2d_rows`` the pointer form on sm_103 and the TMA form elsewhere (so a device of
either kind runs both forms). Their bytes are the input's: a copy. The graph ``doubled_copy`` doubles a ``[Rb, 384]`` tensor in a lowered kernel and
copies its right half with one of them, from the arena. No ``from __future__ import annotations``: tilefish evaluates annotations.
"""

import pathlib

import triton
import triton.language as tl

import tilefish
from tilefish import Dim, Out, StrideOf, Strided, Tensor, TensorDesc, const, f16, i64

R, C = Dim("R"), Dim("C")
Rb = Dim("Rb", buckets=[64, 192])
PTX = pathlib.Path(__file__).resolve().parent / "ptx" / "strided_copy_bf16.ptx"


@triton.jit
def copy_tma(x, y, BM: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0) * BM
    n = tl.program_id(1) * BN
    y.store([m, n], x.load([m, n]))


@triton.jit
def copy_rows(x, y, n_cols, s_row, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    for c in range(0, n_cols, BLOCK):
        cols = c + tl.arange(0, BLOCK)
        keep = cols < n_cols
        tl.store(y + r * n_cols + cols, tl.load(x + r * s_row + cols, mask=keep), mask=keep)


def copy_tma_stub(
    x: TensorDesc(Strided[Tensor[R, C, f16]], block=(64, 64)),
    y: Out[TensorDesc(Tensor[R, C, f16], block=(64, 64))],
    BM: const = 64,
    BN: const = 64,
): ...


def copy_ptx_stub(x: Strided[Tensor[R, C, f16]], y: Out[Tensor[R, C, f16]], n_cols: C, x_s0: StrideOf("x", 0, dtype=i64)): ...


def tile_grid(R, C):
    return (tilefish.cdiv(R, 64), tilefish.cdiv(C, 64))


TMA = tilefish.impl(triton=copy_tma, stub=copy_tma_stub, grid=tile_grid, name="tma")
ROWS = tilefish.impl(triton=copy_rows, num_warps=4)
PTX_IMPL = tilefish.impl(ptx=PTX, symbol="strided_copy_bf16", block=(64,), stub=copy_ptx_stub, name="ptx")


def _clone(x):
    return x.clone()


@tilefish.extern(by_arch={"sm_100": TMA, "sm_103": TMA}, fallback=ROWS, grid=lambda R: (R,), reference=_clone)
def copy2d(x: Strided[Tensor[R, C, f16]], y: Out[Tensor[R, C, f16]], n_cols: C, s_row: StrideOf("x", 0), BLOCK: const = 256): ...


@tilefish.extern(by_arch={"sm_100": PTX_IMPL, "sm_103": PTX_IMPL}, fallback=ROWS, grid=lambda R: (R,), name="copy2d_ptx", reference=_clone)
def copy2d_ptx(x: Strided[Tensor[R, C, f16]], y: Out[Tensor[R, C, f16]], n_cols: C, s_row: StrideOf("x", 0), BLOCK: const = 256): ...


@tilefish.extern(by_arch={"sm_103": ROWS}, fallback=TMA, grid=lambda R: (R,), name="copy2d_rows", reference=_clone)
def copy2d_rows(x: Strided[Tensor[R, C, f16]], y: Out[Tensor[R, C, f16]], n_cols: C, s_row: StrideOf("x", 0), BLOCK: const = 256): ...


OPS = {"copy2d": copy2d, "copy2d_ptx": copy2d_ptx, "copy2d_rows": copy2d_rows}


@tilefish.kernel
def doubled(x: Tensor[Rb, 384, f16]) -> Tensor[Rb, 384, f16]:
    return (x.to(tilefish.f32) * 2.0).to(f16)


def doubled_copy(op):
    """``2x[:, 192:]`` of a ``[Rb, 384]`` tensor: a lowered step that doubles it into the arena, and a step of ``op`` that copies a view of it."""

    @tilefish.graph(name=f"doubled_{op.name}")
    def g(x: Tensor[Rb, 384, f16]) -> Tensor[Rb, 192, f16]:
        return op(doubled(x).split(192, axis=-1)[1])

    return g
