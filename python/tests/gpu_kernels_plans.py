"""Graphs for the tests of the plan and arena features of M3.5 (not collected: the name has no ``test_`` prefix). Synthetic and public.

* ``strided_store``: views that are not dense stored into ``InOut`` parameters (a transpose of a result, a column block of a parameter): each a launch
  of ``tilefish.contiguous`` into the caller's tensor.

* ``zero_and_add``: ``tilefish.zeros_like``, memset steps: an accumulator zeroed in the buffer of the returned value and updated in place twice, the
  caller's ``InOut`` tensor zeroed (one memset of its bytes, no copy) then updated, and an ``i32`` tensor of zeros returned.

* ``stage_a``, ``stage_b``, ``stage_update``, ``stage_zero``: **carried buffers** (``@tilefish.graph(carried=("h",))``): ``stage_a`` writes ``h`` (the
  step that makes the value writes the arena's carried buffer: no copy), ``stage_b`` reads it (its only parameter: its dimension is bound from the shape
  ``h`` was written with), ``stage_update`` updates it in place, ``stage_zero`` zeroes it (a memset of the carried buffer).

No ``from __future__ import annotations``: tilefish evaluates annotations.
"""

import tilefish
from tilefish import Dim, InOut, Tensor, f32, i32

Rs = Dim("Rs", buckets=[8, 24])


@tilefish.kernel
def tripled(x: Tensor[Rs, 64, f32]) -> Tensor[Rs, 64, f32]:
    return x * 3.0


@tilefish.kernel
def plus_one(x: Tensor[Rs, 64, f32]) -> Tensor[Rs, 64, f32]:
    return x + 1.0


@tilefish.kernel
def accumulate(acc: InOut[Tensor[Rs, 64, f32]], x: Tensor[Rs, 64, f32]):
    acc[...] = acc + x


# -- views that are not dense, stored into InOut parameters ------------------------------------------------------------------------------------


@tilefish.graph
def strided_store(x: Tensor[Rs, 64, f32], tr: InOut[Tensor[64, Rs, f32]], cols: InOut[Tensor[Rs, 32, f32]]):
    """``tr = (3 * x).T`` and ``cols = x[:, 32:]`` stored into the caller's tensors (launches of ``tilefish.contiguous``); returns ``3 * x``."""
    t = tripled(x)
    tr[...] = t.permute(1, 0)
    cols[...] = x.split(32, axis=1)[1]
    return t


# -- memset steps --------------------------------------------------------------------------------------------------------------------------------


@tilefish.graph(outputs=("acc", "zeros"))
def zero_and_add(x: Tensor[Rs, 64, f32], state: InOut[Tensor[Rs, 64, f32]]):
    """``acc = (0 + x) + x`` in the caller's output (zeroed by a memset), ``state = 0 + x`` (the caller's tensor zeroed by a memset first), and an
    ``i32`` tensor of zeros of the shape of ``x``."""
    acc = tilefish.zeros_like(x)
    accumulate(acc, x)
    acc = acc.live
    accumulate(acc, x)
    state[...] = tilefish.zeros_like(state)
    accumulate(state, x)
    return acc.live, tilefish.zeros_like(x, dtype=i32)


@tilefish.graph(outputs=("acc",))
def memset_steps(x: Tensor[Rs, 64, f32], state: InOut[Tensor[Rs, 64, f32]]):
    """``zero_and_add`` without its tensor of zeros (the Rust gate holds no output that is all zeros): ``acc = (0 + x) + x``, ``state = 0 + x``."""
    acc = tilefish.zeros_like(x)
    accumulate(acc, x)
    acc = acc.live
    accumulate(acc, x)
    state[...] = tilefish.zeros_like(state)
    accumulate(state, x)
    return acc.live


# -- carried buffers -------------------------------------------------------------------------------------------------------------------------------


@tilefish.graph(carried=("h",), outputs=("y",))
def stage_a(x: Tensor[Rs, 64, f32], h: InOut[Tensor[Rs, 64, f32]]):
    """Writes ``h = 3 * x`` into the arena's carried buffer (the launch of ``tripled`` writes it), returns ``x + 1``."""
    h[...] = tripled(x)
    return plus_one(x)


@tilefish.graph(carried=("h",), outputs=("z",))
def stage_b(h: Tensor[Rs, 64, f32]):
    """Reads ``h`` from the arena (a call gives nothing: ``Rs`` is the size ``h`` was written with), returns ``h + 1``."""
    return plus_one(h)


@tilefish.graph(carried=("h",), outputs=("w",))
def stage_update(h: InOut[Tensor[Rs, 64, f32]], y: Tensor[Rs, 64, f32]):
    """``h += y`` in the arena (it reads ``h`` before it writes it), returns ``3 * y``."""
    accumulate(h, y)
    return tripled(y)


@tilefish.graph(carried=("h",), outputs=("w",))
def stage_zero(x: Tensor[Rs, 64, f32], h: InOut[Tensor[Rs, 64, f32]]):
    """``h = 0`` (a memset of the arena's carried buffer), returns ``3 * x``."""
    h[...] = tilefish.zeros_like(h)
    return tripled(x)


@tilefish.graph(carried=("h",), outputs=("v",))
def stage_chain(h: Tensor[Rs, 64, f32]):
    """``3 * h + 1``: an intermediate in the plan's own region of the arena, read from the carried ``h`` after it."""
    return plus_one(tripled(h))


@tilefish.graph(carried=("g",), outputs=("u",))
def stage_g(x: Tensor[Rs, 64, f32], g: InOut[Tensor[Rs, 64, f32]]):
    """Writes another carried buffer, ``g = x + 1``; returns ``3 * x``."""
    g[...] = plus_one(x)
    return tripled(x)


STAGES = (stage_a, stage_b, stage_update, stage_zero)
CARRIED_SET = (*STAGES, stage_chain, stage_g)  # the golden plan set "carried": two carried names, after a plan with a region of its own
