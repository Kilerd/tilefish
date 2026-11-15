"""Hand-written kernels for the tests of 64-bit dimension scalars, null pointer arguments and the specialisation of a lowered kernel for views that do
not start on 16 bytes (not collected: the name has no ``test_`` prefix).

Written from scratch for these tests, and public. Importing this module imports triton, so only the gpu tests do (``host_ops`` imports it with a
stand-in for Triton where there is none). No ``from __future__ import annotations``: tilefish evaluates annotations.

* ``add_one_wide``: ``y = x + 1`` over ``n`` u8 elements, the count passed as an ``i64`` (``n_elem: i64[n]``) and the offsets computed in ``i64``
  (``index_bits=64``): a kernel that takes a tensor of more than ``2**31`` elements.
* ``scale_null`` and ``scale_bias``: ``y = 2 * x (+ bias)`` over ``n`` f32 elements, one Triton kernel declared twice: with the bias a null pointer
  (``Null[f32]``, the kernel's ``HAS_BIAS`` false) and with a tensor.
* ``scale_null_ptx`` and ``scale_bias_ptx``: the same, a prebuilt PTX written by hand (``ptx/scale_bias_f32.ptx``) that reads the bias only when it
  is not null and takes a 64-bit count: both a null pointer and an ``i64`` dimension scalar of a prebuilt binary.
* ``add_rows``: a lowered kernel of two ``[R]`` int32 rows, and ``plane_sum``: a graph that adds rows ``k`` of a ``[3, R]`` int32 plane, which for an
  ``R`` that is not a multiple of 4 start off 16 bytes, so the plan runs the kernel's specialisation for them.
"""

import pathlib

import triton
import triton.language as tl

import tilefish
from tilefish import Dim, Null, Out, Tensor, const, f32, i32, i64, u8

PTX = pathlib.Path(__file__).resolve().parent / "ptx"
n = Dim("n")


# -- a 64-bit dimension scalar ---------------------------------------------------------------------------------------------------------------


@triton.jit
def add_one_wide_kernel(x, y, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)  # i64 offsets: n_elem is an i64 scalar
    mask = offs < n_elem
    tl.store(y + offs, tl.load(x + offs, mask=mask) + 1, mask=mask)


@tilefish.extern(triton=add_one_wide_kernel, num_warps=4, grid=lambda n: (tilefish.cdiv(n, 1024),), index_bits=64)
def add_one_wide(x: Tensor[n, u8], y: Out[Tensor[n, u8]], n_elem: i64[n], BLOCK: const = 1024): ...


# -- a null pointer --------------------------------------------------------------------------------------------------------------------------


@triton.jit
def scale_bias_kernel(x, bias, y, n_elem, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    v = tl.load(x + offs, mask=mask) * 2.0
    if HAS_BIAS:
        v = v + tl.load(bias + offs, mask=mask)
    tl.store(y + offs, v, mask=mask)


@tilefish.extern(triton=scale_bias_kernel, num_warps=4, grid=lambda n: (tilefish.cdiv(n, 256),), name="scale_null")
def scale_null(x: Tensor[n, f32], bias: Null[f32], y: Out[Tensor[n, f32]], n_elem: n, HAS_BIAS: const = False, BLOCK: const = 256): ...


@tilefish.extern(triton=scale_bias_kernel, num_warps=4, grid=lambda n: (tilefish.cdiv(n, 256),), name="scale_bias")
def scale_bias(x: Tensor[n, f32], bias: Tensor[n, f32], y: Out[Tensor[n, f32]], n_elem: n, HAS_BIAS: const = True, BLOCK: const = 256): ...


# -- both, in a prebuilt binary ----------------------------------------------------------------------------------------------------------------


@tilefish.extern(ptx=PTX / "scale_bias_f32.ptx", symbol="scale_bias_f32", block=(256,), grid=lambda n: (tilefish.cdiv(n, 256),), index_bits=64)
def scale_null_ptx(x: Tensor[n, f32], bias: Null[f32], out: Out[Tensor[n, f32]], n_elem: i64[n]): ...


@tilefish.extern(ptx=PTX / "scale_bias_f32.ptx", symbol="scale_bias_f32", block=(256,), grid=lambda n: (tilefish.cdiv(n, 256),), index_bits=64)
def scale_bias_ptx(x: Tensor[n, f32], bias: Tensor[n, f32], out: Out[Tensor[n, f32]], n_elem: i64[n]): ...


# -- rows of a plane that do not start on 16 bytes -------------------------------------------------------------------------------------------

R = Dim("R", buckets=[5, 8, 13])


@tilefish.kernel
def add_rows(a: Tensor[R, i32], b: Tensor[R, i32]) -> Tensor[R, i32]:
    return a + b


@tilefish.graph
def plane_sum(plane: Tensor[3, R, i32]):
    """``plane[1] + plane[0] + plane[2]``: rows 1 and 2 start ``4 * R`` and ``8 * R`` bytes in, which is not provably a multiple of 16 (it is for no
    bucket but 8), so the two calls run specialisations of ``add_rows``: the first for ``a`` (``add_rows__ua0``), the second for ``b``
    (``add_rows__ua1``)."""
    s = add_rows(plane[1], plane[0])
    return add_rows(s, plane[2])


def unaligned_ops():
    """``{name: operator}``: a specialisation of a lowered kernel (``add_rows`` for a first row that does not start on 16 bytes), for the host tests."""
    return {"add_rows_ua": add_rows.unaligned("a")}


__all__ = ["add_one_wide", "add_rows", "plane_sum", "scale_bias", "scale_bias_ptx", "scale_null", "scale_null_ptx", "f32", "i32", "i64"]
