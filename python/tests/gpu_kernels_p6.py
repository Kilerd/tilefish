"""Graphs for the gpu tests of the plan features of M3.3 (third part; not collected: the name has no ``test_`` prefix). Synthetic and public.

* ``static_affine``: a graph whose shapes are all static, which declares its roles (``inputs=("x",)``): ``x`` is an input, ``w`` and ``b`` are
  weights, which a host binds once (``tilefish.runtime.bind_weights``, Rust ``PlanRun::bind_weights``). Without the declaration all three would
  be labelled weights.

No ``from __future__ import annotations``: tilefish evaluates annotations.
"""

import tilefish
from tilefish import Tensor, f32


@tilefish.kernel
def scale_shift(x: Tensor[16, 64, f32], w: Tensor[16, 64, f32], b: Tensor[16, 64, f32]) -> Tensor[16, 64, f32]:
    return x * w + b


@tilefish.kernel
def double(x: Tensor[16, 64, f32]) -> Tensor[16, 64, f32]:
    return x * 2.0


@tilefish.graph(inputs=("x",))
def static_affine(x: Tensor[16, 64, f32], w: Tensor[16, 64, f32], b: Tensor[16, 64, f32]):
    """``2 * (x * w + b)``: two steps, the weights read by the first."""
    return double(scale_shift(x, w, b))
