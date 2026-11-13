"""Hand-written Triton kernels for the gpu tests of the review of M3.2 (not collected: the name has no ``test_`` prefix). Importing this module imports triton.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import triton
import triton.language as tl


@triton.jit
def fill_kernel(x_ptr, n_elements, VALUE: tl.constexpr, BLOCK: tl.constexpr):
    """``x[:n] = VALUE``: writes ``n_elements`` elements from the pointer it is given, whatever the tensor behind it holds."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(x_ptr + offsets, tl.full((BLOCK,), VALUE, tl.float32), mask=mask)
