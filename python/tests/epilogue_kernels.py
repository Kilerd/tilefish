"""Small Triton kernels with epilogue hooks for ``test_gpu_epilogue.py`` (not collected: the name has no ``test_`` prefix). Importing this module
imports Triton.

* ``scale3``: ``c = (3 * x)`` rounded to ``c``'s type, tile by tile, with an ``out`` hook as the ABI says (after the rounding);
* ``lying``: the same kernel with the hook given the f32 value before the rounding, though it declares ``value="out"``: a fusion through it is not
  the extern followed by the region, and the export-time check must refuse it.

No ``from __future__ import annotations``: tilefish evaluates annotations.
"""

import triton
import triton.language as tl

import tilefish
from tilefish import Dim, Out, Tensor, bf16, const

R, C = Dim("R"), Dim("C")


@triton.jit
def scale3_kernel(x, c, n_rows, n_cols, BM: tl.constexpr, BN: tl.constexpr, LIE: tl.constexpr, EPILOGUE: tl.constexpr, EPILOGUE_ARGS):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    row_mask = rows < n_rows
    col_mask = cols < n_cols
    mask = row_mask[:, None] & col_mask[None, :]
    acc = tl.load(x + rows[:, None] * n_cols + cols[None, :], mask=mask, other=0.0).to(tl.float32) * 3.0
    if LIE:  # the hook sees the value before the rounding the extern's own output has
        tile = EPILOGUE(acc, rows, cols, row_mask, col_mask, EPILOGUE_ARGS)
    else:
        tile = EPILOGUE(acc.to(c.dtype.element_ty), rows, cols, row_mask, col_mask, EPILOGUE_ARGS)
    tl.store(c + rows[:, None] * n_cols + cols[None, :], tile.to(c.dtype.element_ty), mask=mask)


def _declare(name, lie):
    def stub(x: Tensor[R, C, bf16], c: Out[Tensor[R, C, bf16]], n_rows: R, n_cols: C, BM: const = 32, BN: const = 64, LIE: const = int(lie)): ...

    stub.__name__ = stub.__qualname__ = name
    return tilefish.extern(
        triton=scale3_kernel,
        name=name,
        grid=lambda R, C: (tilefish.cdiv(R, 32), tilefish.cdiv(C, 64)),
        reference=lambda x: (x.float() * 3.0).to(x.dtype),
        epilogue=tilefish.hook("EPILOGUE", value="out", out="c"),
    )(stub)


scale3 = _declare("scale3", False)
lying = _declare("lying", True)
