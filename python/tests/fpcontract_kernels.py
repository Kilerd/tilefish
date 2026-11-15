"""Hand-written Triton and Gluon kernels for the gpu tests of the contraction check (not collected: the name has no ``test_`` prefix).

Written from scratch for these tests, and public. Importing this module imports Triton and its Gluon, so only the gpu tests do. No ``from
__future__ import annotations``: tilefish evaluates annotations.

* ``row_axpb_kernel``: ``y = x * w + b`` of a ``[R, 1024]`` row of bf16 values widened to f32, rounded to bf16. Declared with
  ``enable_fp_fusion=False`` (``row_axpb_plain``) its binary is contracted on sm_100 and later: LLVM packs two lanes into ``mul.rn.f32x2`` and
  ``add.rn.f32x2`` and ptxas 12.9 fuses the pair (``FFMA2`` on sm_100 and sm_103, ``FFMA`` on sm_120), so the check refuses it; on sm_90 there is
  no packed f32 instruction and it builds. ``row_axpb_fused`` declares the same kernel with ``enable_fp_fusion=True`` (contraction allowed, never
  checked).
* ``row_axpb_rn_kernel``: the same with its multiply and add written as scalar inline PTX (``mul.rn.f32``, ``add.rn.f32``), which nothing packs
  or contracts: ``row_axpb_rn`` builds on every architecture and rounds twice.
* ``row_axpb_gluon_kernel``: the same row in Gluon (eight columns per thread), which LLVM packs the same way: ``row_axpb_gluon``
  (``enable_fp_fusion=False``) is refused on sm_100 and later like ``row_axpb_plain``. (An elementwise ``x * w + b`` of three loads of one length,
  f32 or widened bf16, two to eight elements per thread, is not packed, measured with Triton 3.5.1.)
"""

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import tilefish
from tilefish import Dim, Out, Tensor, bf16, const

R = Dim("R")


@triton.jit
def row_axpb_kernel(x, w, b, y, D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    v = tl.load(x + row * D + cols).to(tl.float32)
    tl.store(y + row * D + cols, (v * tl.load(w + cols).to(tl.float32) + tl.load(b + cols).to(tl.float32)).to(tl.bfloat16))


@triton.jit
def _mul_rn(a, c):
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, c], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _add_rn(a, c):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [a, c], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def row_axpb_rn_kernel(x, w, b, y, D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    v = tl.load(x + row * D + cols).to(tl.float32)
    tl.store(y + row * D + cols, _add_rn(_mul_rn(v, tl.load(w + cols).to(tl.float32)), tl.load(b + cols).to(tl.float32)).to(tl.bfloat16))


def _row(kernel, name, fusion):
    @tilefish.extern(triton=kernel, num_warps=4, grid=lambda R: (R,), enable_fp_fusion=fusion, name=name)
    def row_axpb(x: Tensor[R, 1024, bf16], w: Tensor[1024, bf16], b: Tensor[1024, bf16], y: Out[Tensor[R, 1024, bf16]], D: const = 1024): ...

    return row_axpb


row_axpb_plain = _row(row_axpb_kernel, "row_axpb_plain", False)
row_axpb_fused = _row(row_axpb_kernel, "row_axpb_fused", True)
row_axpb_rn = _row(row_axpb_rn_kernel, "row_axpb_rn", False)


@gluon.jit
def row_axpb_gluon_kernel(x, w, b, y, D: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[8], threads_per_warp=[32], warps_per_cta=[4], order=[0])
    row = gl.program_id(0)
    cols = gl.arange(0, D, layout=layout)
    v = gl.load(x + row * D + cols).to(gl.float32) * gl.load(w + cols).to(gl.float32) + gl.load(b + cols).to(gl.float32)
    gl.store(y + row * D + cols, v.to(gl.bfloat16))


@tilefish.extern(gluon=row_axpb_gluon_kernel, num_warps=4, grid=lambda R: (R,), enable_fp_fusion=False, name="row_axpb_gluon")
def row_axpb_gluon(x: Tensor[R, 1024, bf16], w: Tensor[1024, bf16], b: Tensor[1024, bf16], y: Out[Tensor[R, 1024, bf16]], D: const = 1024): ...
