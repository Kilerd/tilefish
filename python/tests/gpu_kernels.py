"""Hand-written Triton kernels for the gpu tests (not collected: the name has no ``test_`` prefix).

They are written from scratch for these tests and declared with ``tilefish.extern``, like any existing
Triton kernel would be. Importing this module imports triton, so only the gpu tests do.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import triton
import triton.language as tl

import tilefish
from tilefish import Dim, InOut, Out, Tensor, bf16, const, f32

# -- an rmsnorm over rows of a static width ------------------------------------------------------------------------------------


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, EPS: tl.constexpr, N_COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS
    x = tl.load(x_ptr + row * N_COLS + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / N_COLS
    scale = 1.0 / tl.sqrt(mean_square + EPS)
    y = x * scale * w
    tl.store(y_ptr + row * N_COLS + cols, y.to(tl.bfloat16), mask=mask)


Rows = Dim("Rows")
Hidden = Dim("Hidden", 2048)


@tilefish.extern(triton=rmsnorm_kernel, num_warps=8, grid=lambda Rows: (Rows,))
def rmsnorm(
    x_ptr: Tensor[Rows, Hidden, bf16],
    w_ptr: Tensor[Hidden, bf16],
    y_ptr: Out[Tensor[Rows, Hidden, bf16]],
    *,
    EPS: const[f32] = 1e-6,
    N_COLS: const = Hidden,
    BLOCK: const = 2048,
): ...


# -- the exact (erf) gelu over a flat tensor, with a runtime element count -----------------------------------------------------------


@triton.jit
def gelu_erf_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(y_ptr + offsets, y.to(tl.bfloat16), mask=mask)


Count = Dim("Count", divisible_by=16)


@tilefish.extern(triton=gelu_erf_kernel, num_warps=4, grid=lambda Count: (tilefish.cdiv(Count, 1024),))
def gelu_erf(x_ptr: Tensor[Count, bf16], y_ptr: Out[Tensor[Count, bf16]], n_elements: Count, *, BLOCK: const = 1024): ...


# -- the dtype of a python float or a constexpr float combined with a bf16 tensor ------------------------------------------------


@triton.jit
def scalar_dtype_probe(x_ptr, y_ptr, EPS: tl.constexpr):
    offsets = tl.arange(0, 16)
    x = tl.load(x_ptr + offsets)
    tl.static_assert(x.dtype == tl.bfloat16)
    scaled = x * 0.1  # a python float
    tl.static_assert(scaled.dtype == tl.bfloat16)
    shifted = x + EPS  # a constexpr float
    tl.static_assert(shifted.dtype == tl.bfloat16)
    tl.static_assert((EPS * x).dtype == tl.bfloat16)
    tl.store(y_ptr + offsets, scaled + shifted)


@triton.jit
def division_dtype_probe(x_ptr, y_ptr, EPS: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, 16))
    tl.static_assert((x / 2.0).dtype == tl.float32)  # unlike + - *, a division of bf16 is done in f32 and returns f32
    tl.static_assert((x / x).dtype == tl.float32)
    tl.store(y_ptr + tl.arange(0, 16), x)


@triton.jit
def scalar_dtype_wrong_probe(x_ptr, y_ptr, EPS: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, 16))
    tl.static_assert((x * 0.1).dtype == tl.float32)  # false: this kernel must not compile
    tl.store(y_ptr + tl.arange(0, 16), x)


# -- a kernel that does not compile ----------------------------------------------------------------------------------------------------


@triton.jit
def broken_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    tl.store(x_ptr + tl.arange(0, BLOCK), a_name_that_does_not_exist)  # noqa: F821


@tilefish.extern(triton=broken_kernel, grid=lambda Count: (Count,))
def broken(x_ptr: InOut[Tensor[Count, f32]], n_elements: Count, *, BLOCK: const = 64): ...


# -- y += alpha * x, in place ------------------------------------------------------------------------------------------------------


@triton.jit
def axpy_kernel(x_ptr, y_ptr, n_elements, ALPHA: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, y + ALPHA * x, mask=mask)


@tilefish.extern(triton=axpy_kernel, num_warps=4, grid=lambda Count: (tilefish.cdiv(Count, 512),))
def axpy(x_ptr: Tensor[Count, f32], y_ptr: InOut[Tensor[Count, f32]], n_elements: Count, *, ALPHA: const[f32] = 0.3, BLOCK: const = 512): ...
