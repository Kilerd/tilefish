"""Externs with epilogue hooks (``docs/rfcs/0003-epilogue-fusion.md``, milestone E1): a GEMM and a 1-D convolution written in Triton, whose epilogue
a ``@tilefish.graph`` fuses into the kernel.

Each kernel takes two hooks, which the stubs do not list (``tilefish._epilogue.ABI``):

* ``ACC`` (``value="acc"``) sees the f32 accumulator tile, before the kernel rounds it to the output's type;
* ``EPILOGUE`` (``value="out"``) sees the tile rounded to the output's type, as the kernel would store it.

A graph that calls one and computes on its result gets one kernel instead of two: under ``tilefish.fusion()`` (exact, the default) through ``EPILOGUE``,
with the bytes of the GEMM followed by the arithmetic (contract (i)); under ``tilefish.fusion(epilogues="accumulator")`` through ``ACC``, one rounding
fewer (contract (ii)). Without a fusion both hooks are the identity, and the kernel is a plain GEMM.

* :func:`gemm`: ``c = a @ b.T`` (``a: [M, K]``, ``b: [N, K]``, an f32 accumulator), TMA descriptors for ``a`` and ``b``, ``tl.dot`` (``tcgen05.mma``
  on sm_100 and sm_103, ``wgmma`` on sm_90, ``mma.sync`` on sm_120), a grouped tile order, no split-K. ``split=2`` makes the dual-accumulator GEMM a
  gated activation needs: a program holds the tile of the first half of ``b``'s rows (the gate) and the matching tile of the second half (the up
  projection), so ``silu(g) * u`` fuses into it.
* :func:`gemm_sm100` and :func:`gemm_sm120` are its configurations for sm_100/103 (``tcgen05.mma``, asserted in the SASS) and for sm_120
  (``mma.sync``, ``HMMA.16816``, asserted; tiles that fit its shared memory).
* :func:`conv1d`: a 1-D convolution of a channel-last ``[B, L, C]`` input as an implicit GEMM (its rows are the output positions ``B * Lo``, its
  columns the output channels), with the same hooks: a convolution followed by a bias and an activation is one kernel.

Importing this module imports Triton. No ``from __future__ import annotations``: tilefish evaluates annotations.
"""

import inspect

import triton
import triton.language as tl

import tilefish
from tilefish import Dim, Out, Tensor, TensorDesc, bf16, const

M, N, K = Dim("M"), Dim("N", divisible_by=16), Dim("K", divisible_by=16)
B, L = Dim("B"), Dim("L")


@triton.jit
def gemm_kernel(a, b, c, m, n, k, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, GROUP_M: tl.constexpr,
                ACC: tl.constexpr, ACC_ARGS, EPILOGUE: tl.constexpr, EPILOGUE_ARGS):  # fmt: skip
    """``c = a @ b.T``: ``a`` and ``b`` TMA descriptors of ``[BM, BK]`` and ``[BN, BK]`` blocks, ``c`` ``[m, n // SPLIT * len(tiles)]``."""
    chunk = n // SPLIT  # the columns of one tile's chunk (all of them for SPLIT = 1)
    pid = tl.program_id(0)
    num_m = tl.cdiv(m, BM)
    num_n = tl.cdiv(chunk, BN)
    width = GROUP_M * num_n
    first = (pid // width) * GROUP_M
    size = tl.minimum(num_m - first, GROUP_M)
    pid_m = first + (pid % width) % size
    pid_n = (pid % width) // size
    off_m = pid_m * BM
    off_n = pid_n * BN
    acc0 = tl.zeros((BM, BN), dtype=tl.float32)
    acc1 = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in tl.range(0, k, BK):
        x = a.load([off_m, kk])
        acc0 = tl.dot(x, b.load([off_n, kk]).T, acc0)
        if SPLIT == 2:  # the dual accumulator: the same rows of a, the rows chunk + off_n of b
            acc1 = tl.dot(x, b.load([chunk + off_n, kk]).T, acc1)
    rows = off_m + tl.arange(0, BM)
    cols = off_n + tl.arange(0, BN)
    row_mask = rows < m
    col_mask = cols < chunk
    mask = row_mask[:, None] & col_mask[None, :]
    if SPLIT == 1:
        tile = ACC(acc0, rows, cols, row_mask, col_mask, ACC_ARGS)  # the accumulator hook (value="acc")
        tile = EPILOGUE(tile.to(c.dtype.element_ty), rows, cols, row_mask, col_mask, EPILOGUE_ARGS)  # the rounded tile (value="out")
        tl.store(c + rows[:, None] * n + cols[None, :], tile, mask=mask)
    else:
        tiles = ACC((acc0, acc1), rows, cols, row_mask, col_mask, ACC_ARGS)
        if len(tiles) == 2:
            tiles = EPILOGUE((tiles[0].to(c.dtype.element_ty), tiles[1].to(c.dtype.element_ty)), rows, cols, row_mask, col_mask, EPILOGUE_ARGS)
        ldc = chunk * len(tiles)  # the hook returned one tile (a fused gated activation) or both: c's rows hold that many chunks
        for j in tl.static_range(len(tiles)):
            tl.store(c + rows[:, None] * ldc + (j * chunk + cols)[None, :], tiles[j].to(c.dtype.element_ty), mask=mask)


def _reference_gemm(dtype: object):
    def ref(a, b):
        return (a.float() @ b.float().T).to(a.dtype if dtype is None else dtype)

    return ref


def _acc_gemm(a, b):
    return a.float() @ b.float().T


def _stub(params: "list[tuple[str, object, object]]", name: str) -> object:
    """A stub function with these ``(name, annotation, default)`` parameters (what a declaration in a module writes by hand)."""
    sig = inspect.Signature([inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=a, default=d) for n, a, d in params])

    def stub(*args, **kwargs): ...

    stub.__signature__ = sig
    stub.__name__ = stub.__qualname__ = name
    stub.__module__ = __name__
    return stub


_E = inspect.Parameter.empty


def gemm(
    *,
    name: str = "gemm",
    dtype=bf16,
    out_dtype=None,
    split: int = 1,
    BM: int = 128,
    BN: int = 128,
    BK: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
    group_m: int = 8,
    sass=None,
):
    """The extern of :func:`gemm_kernel`: ``c = a @ b.T``, ``a: [M, K]``, ``b: [N, K]`` of ``dtype``, ``c: [M, N]`` of ``out_dtype`` (``dtype`` by
    default), its hooks ``ACC`` and ``EPILOGUE`` on ``c`` (``split`` tiles at once), the tile ``BM x BN`` (``BN`` the hooks' ``block_n``: a norm over
    rows of at most ``BN`` columns fuses into it), ``BK`` columns of ``K`` a step. ``M`` is a runtime size; ``N`` and ``K`` multiples of 16."""
    odt = dtype if out_dtype is None else out_dtype
    stub = _stub(
        [
            ("a", TensorDesc(Tensor[M, K, dtype], block=(BM, BK)), _E),
            ("b", TensorDesc(Tensor[N, K, dtype], block=(BN, BK)), _E),
            ("c", Out[Tensor[M, N, odt]], _E),
            ("m", M, _E),
            ("n", N, _E),
            ("k", K, _E),
            ("BM", const, BM),
            ("BN", const, BN),
            ("BK", const, BK),
            ("SPLIT", const, split),
            ("GROUP_M", const, group_m),
        ],
        name,
    )
    hooks = (
        tilefish.hook("ACC", value="acc", out="c", split=split, block_n="BN", reference=_acc_gemm),
        tilefish.hook("EPILOGUE", value="out", out="c", split=split, block_n="BN"),
    )
    return tilefish.extern(
        triton=gemm_kernel,
        name=name,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=lambda M, N, BM, BN, SPLIT: (tilefish.cdiv(M, BM) * tilefish.cdiv(N, SPLIT * BN),),
        reference=_reference_gemm(out_dtype),
        epilogue=hooks,
        sass=sass,
    )(stub)


def gemm_sm100(**kw):
    """:func:`gemm` for sm_100 and sm_103: ``tcgen05.mma`` (asserted in the PTX of every binary), tiles of 128 x 128, three stages."""
    kw.setdefault("sass", tilefish.sass(mma="tcgen05.mma kind::f16", no_tf32=True))
    return gemm(**kw)


def gemm_sm120(**kw):
    """:func:`gemm` for sm_120: ``mma.sync`` (``HMMA.16816``, asserted in the SASS of every binary), tiles of 128 x 64, two stages (sm_120 has 100 KB
    of shared memory a block)."""
    kw.setdefault("BN", 64)
    kw.setdefault("num_stages", 2)
    kw.setdefault("sass", tilefish.sass(mma="mma.sync m16n8k16", no_tf32=True))
    return gemm(**kw)


def gemm_for(arch: int, **kw):
    """The configuration of :func:`gemm` for an architecture (``103``, ``120``): :func:`gemm_sm100` on 100 and 103, :func:`gemm_sm120` on 120 and
    :func:`gemm` elsewhere."""
    if arch in (100, 103):
        return gemm_sm100(**kw)
    if arch == 120:
        return gemm_sm120(**kw)
    return gemm(**kw)


# -- a 1-D convolution as an implicit GEMM ------------------------------------------------------------------------------------------------


@triton.jit
def conv1d_kernel(x, w, c, nb, l_in, l_out, CI: tl.constexpr, CO: tl.constexpr, KW: tl.constexpr, PAD: tl.constexpr, DIL: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ACC: tl.constexpr, ACC_ARGS, EPILOGUE: tl.constexpr, EPILOGUE_ARGS):  # fmt: skip
    """``c[b, t, o] = sum_{j, i} x[b, t - PAD + j * DIL, i] * w[o, j * CI + i]``: ``x`` ``[nb, l_in, CI]``, ``w`` ``[CO, KW * CI]`` (tap-major), ``c``
    ``[nb, l_out, CO]``; positions outside ``x`` read zeros. The GEMM's rows are ``b * l_out + t``, its columns ``o``, its ``K`` ``KW * CI`` (``BK``
    divides ``CI``: a step is one tap)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    rows_total = nb * l_out
    row_mask = rows < rows_total
    col_mask = cols < CO
    bi = rows // l_out
    t = rows % l_out
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for j in tl.static_range(KW):
        pos = t - PAD + j * DIL
        ok = row_mask & (pos >= 0) & (pos < l_in)
        base = (bi * l_in + pos).to(tl.int64) * CI
        for i0 in tl.range(0, CI, BK):
            xa = tl.load(x + base[:, None] + (i0 + ks)[None, :], mask=ok[:, None], other=0.0)
            wb = tl.load(w + cols[:, None] * (KW * CI) + (j * CI + i0 + ks)[None, :], mask=col_mask[:, None], other=0.0)
            acc = tl.dot(xa, wb.T, acc)
    tile = ACC(acc, rows, cols, row_mask, col_mask, ACC_ARGS)
    tile = EPILOGUE(tile.to(c.dtype.element_ty), rows, cols, row_mask, col_mask, EPILOGUE_ARGS)
    tl.store(c + rows[:, None] * CO + cols[None, :], tile, mask=row_mask[:, None] & col_mask[None, :])


def _reference_conv(kw: int, pad: int, dil: int, ci: int, co: int):
    def ref(x, w):
        import torch.nn.functional as F

        weight = w.float().reshape(co, kw, ci).permute(0, 2, 1)  # [CO, CI, KW]
        y = F.conv1d(x.float().permute(0, 2, 1), weight, padding=pad, dilation=dil)
        return y.permute(0, 2, 1).to(x.dtype)

    return ref


def _acc_conv(kw: int, pad: int, dil: int, ci: int, co: int):
    def ref(x, w):
        import torch.nn.functional as F

        weight = w.float().reshape(co, kw, ci).permute(0, 2, 1)
        return F.conv1d(x.float().permute(0, 2, 1), weight, padding=pad, dilation=dil).permute(0, 2, 1).contiguous()

    return ref


def conv1d(
    ci: int,
    co: int,
    kw: int,
    *,
    pad: "int | None" = None,
    dil: int = 1,
    name: str = "conv1d",
    dtype=bf16,
    BM: int = 128,
    BN: int = 64,
    BK: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    sass=None,
):
    """The extern of :func:`conv1d_kernel`: a 1-D convolution of ``x: [B, L, ci]`` (channel-last) by ``w: [co, kw * ci]`` (a ``Conv1d`` weight
    ``[co, ci, kw]`` permuted to ``[co, kw, ci]`` and flattened), ``pad`` positions of zeros on each side (``"same"`` by default:
    ``dil * (kw - 1) // 2``), stride 1: ``c: [B, L + 2 * pad - dil * (kw - 1), co]``. Its hooks are :func:`gemm`'s, on ``c`` (rows ``B * Lo``)."""
    if pad is None:
        pad = dil * (kw - 1) // 2
    lo = L + (2 * pad - dil * (kw - 1))
    stub = _stub(
        [
            ("x", Tensor[B, L, ci, dtype], _E),
            ("w", Tensor[co, kw * ci, dtype], _E),
            ("c", Out[Tensor[B, lo, co, dtype]], _E),
            ("nb", B, _E),
            ("l_in", L, _E),
            ("l_out", lo, _E),
            ("CI", const, ci),
            ("CO", const, co),
            ("KW", const, kw),
            ("PAD", const, pad),
            ("DIL", const, dil),
            ("BM", const, BM),
            ("BN", const, BN),
            ("BK", const, min(BK, ci)),
        ],
        name,
    )
    hooks = (
        tilefish.hook("ACC", value="acc", out="c", block_n="BN", reference=_acc_conv(kw, pad, dil, ci, co)),
        tilefish.hook("EPILOGUE", value="out", out="c", block_n="BN"),
    )
    return tilefish.extern(
        triton=conv1d_kernel,
        name=name,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=lambda B, L, BM, BN: (tilefish.cdiv(B * (L + (2 * pad - dil * (kw - 1))), BM), tilefish.cdiv(co, BN)),
        reference=_reference_conv(kw, pad, dil, ci, co),
        epilogue=hooks,
        sass=sass,
    )(stub)
