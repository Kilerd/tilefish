"""Hand-written Triton kernels for the gpu tests of the TF32 guard (not collected: the name has no ``test_`` prefix).

Written from scratch for these tests, and public. No ``from __future__ import annotations``: tilefish evaluates annotations.

* ``dot_kernel``: ``c = a @ b`` in f32 for row tiles of ``BM`` rows, ``a: [R, 16]`` f16 (converted to f32 in the kernel) and ``b: [16, 32]`` f32, one
  ``tl.dot`` whose input precision ``PREC`` picks (0: ``"ieee"``, 1: ``"tf32"``). Triton 3.5.1 lowers the IEEE dot to ``tcgen05.mma ... kind::tf32``
  on sm_100 and sm_103 at ``BM = 64``: its MMAv5 path does not look at the input precision of a dot whose operand was loaded narrower than 32 bits.
  At ``BM = 32``, and on sm_90 and sm_120, it stays on FMAs. Declared as ``ieee64`` (refused on sm_100 and sm_103), ``ieee32`` and ``tf32_64``.
* ``mixed_kernel``: one kernel with two dots, ``c = x @ b[:16]`` as an IEEE dot (``x`` the f16 ``h`` converted to f32 when ``UP``, else the first 16
  columns of the f32 ``a``) and ``d = a @ b`` as a TF32 dot over a loop of four 16-column steps. Declared as ``mixed_f16`` (its IEEE dot goes to
  tcgen05: refused) and ``mixed_f32`` (its IEEE dot stays on FMAs, its TF32 dot is on tensor cores: built).
"""

import triton
import triton.language as tl

import tilefish
from tilefish import Dim, Out, Tensor, const, f16, f32

R = Dim("R", divisible_by=64)


@triton.jit
def dot_kernel(a, b, c, BM: tl.constexpr, PREC: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    k = tl.arange(0, 16)
    n = tl.arange(0, 32)
    x = tl.load(a + rows[:, None] * 16 + k[None, :]).to(tl.float32)
    y = tl.load(b + k[:, None] * 32 + n[None, :])
    if PREC == 0:
        z = tl.dot(x, y, input_precision="ieee")
    else:
        z = tl.dot(x, y, input_precision="tf32")
    tl.store(c + rows[:, None] * 32 + n[None, :], z)


def _dot(name, bm, prec):
    @tilefish.extern(triton=dot_kernel, num_warps=4, grid=lambda R, BM: (tilefish.cdiv(R, BM),), name=name)
    def op(a: Tensor[R, 16, f16], b: Tensor[16, 32, f32], c: Out[Tensor[R, 32, f32]], BM: const = bm, PREC: const = prec): ...

    return op


ieee64 = _dot("tf32_guard_ieee64", 64, 0)
ieee32 = _dot("tf32_guard_ieee32", 32, 0)
tf32_64 = _dot("tf32_guard_tf32_64", 64, 1)


@triton.jit
def mixed_kernel(a, h, b, c, d, UP: tl.constexpr):
    rows = tl.program_id(0) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 16)
    n = tl.arange(0, 32)
    if UP:
        x = tl.load(h + rows[:, None] * 16 + k[None, :]).to(tl.float32)
    else:
        x = tl.load(a + rows[:, None] * 64 + k[None, :])
    y = tl.load(b + k[:, None] * 32 + n[None, :])
    tl.store(c + rows[:, None] * 32 + n[None, :], tl.dot(x, y, input_precision="ieee"))
    acc = tl.zeros((64, 32), dtype=tl.float32)
    for k0 in range(0, 64, 16):
        x2 = tl.load(a + rows[:, None] * 64 + (k0 + k)[None, :])
        y2 = tl.load(b + (k0 + k)[:, None] * 32 + n[None, :])
        acc = tl.dot(x2, y2, acc, input_precision="tf32")
    tl.store(d + rows[:, None] * 32 + n[None, :], acc)


def _mixed(name, up):
    @tilefish.extern(triton=mixed_kernel, num_warps=4, grid=lambda R: (tilefish.cdiv(R, 64),), name=name)
    def op(
        a: Tensor[R, 64, f32], h: Tensor[R, 16, f16], b: Tensor[64, 32, f32], c: Out[Tensor[R, 32, f32]], d: Out[Tensor[R, 32, f32]], UP: const = up
    ): ...

    return op


mixed_f16 = _mixed("tf32_guard_mixed_f16", True)
mixed_f32 = _mixed("tf32_guard_mixed_f32", False)
