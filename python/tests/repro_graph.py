"""A launch plan with every kind of binary, for the reproducibility test on a GPU (``test_gpu_repro.py``; not collected: no ``test_`` prefix).

The test copies this file (and the PTX it declares) into two directories and exports the graph from each, in two processes: the binaries must be
the same files. It has

* a lowered kernel (``softmax_rows``, a ``@tilefish.kernel``: its Triton source is generated and written to a file at build time);
* a hand-written Triton kernel declared with ``extern`` (``scale_rows``: its source file is this module, wherever it was copied to);
* generated ops (``tilefish.im2col1d``, ``tilefish.col2im1d``, and the ``tilefish.contiguous`` that a returned transposed view becomes: their
  sources are generated and written like a lowered kernel's);
* a prebuilt PTX (``axpy_ptx``, assembled by ``ptxas`` from a scratch file at build time).

Importing it imports Triton.
"""

import pathlib

import triton
import triton.language as tl

import tilefish
from tilefish import Dim, Out, Tensor, bf16, const, f32

L = Dim("L", buckets=[32, 64])
B = Dim("B", buckets=[2])
N = Dim("N", buckets=[1024])
PTX = pathlib.Path(__file__).resolve().parent / "ptx" / "axpy_f32.ptx"


@tilefish.kernel
def softmax_rows(x: Tensor[L, 64, f32]) -> Tensor[L, 64, f32]:
    m = tilefish.max(x, axis=-1, keepdims=True)
    e = tilefish.exp(x - m)
    return e / tilefish.sum(e, axis=-1, keepdims=True)


@triton.jit
def scale_rows_kernel(x_ptr, y_ptr, N_COLS: tl.constexpr, SCALE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, N_COLS)
    x = tl.load(x_ptr + row * N_COLS + cols)
    tl.store(y_ptr + row * N_COLS + cols, x * SCALE)


@tilefish.extern(triton=scale_rows_kernel, num_warps=2, grid=lambda L: (L,))
def scale_rows(x_ptr: Tensor[L, 64, f32], y_ptr: Out[Tensor[L, 64, f32]], *, N_COLS: const = 64, SCALE: const[f32] = 0.5): ...


_N = Dim("n")


@tilefish.extern(ptx=PTX, symbol="axpy_f32", block=(128,), grid=lambda n: (tilefish.cdiv(n, 128),), name="repro_axpy")
def axpy_ptx(x: Tensor[_N, f32], y: Tensor[_N, f32], out: Out[Tensor[_N, f32]], n_elem: _N, alpha: const[f32] = 2.0): ...


@tilefish.graph
def repro(x: Tensor[L, 64, f32], c: Tensor[B, 16, L, bf16], v: Tensor[N, f32]):
    y = scale_rows(softmax_rows(x))
    cols = tilefish.im2col1d(c, 3, padding=1)
    back = tilefish.col2im1d(cols, 3, padding=1)
    return y.permute(1, 0), back, axpy_ptx(v, v)
