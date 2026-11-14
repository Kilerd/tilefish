"""Graphs for the gpu tests of tensors whose strides are products of dimensions (not collected: the name has no ``test_`` prefix). Synthetic and
public.

``attention_mask`` makes an ``[H, Lb, Lb]`` tensor in a plan, a buffer whose stride along its first axis is ``Lb*Lb`` (a product of two symbolic
sizes: a stride a view of it has, and an offset its ``[1]`` has), from two vectors per head (an outer product, both operands broadcast), masks its
keys with a ``[H, 1, Lb]`` mask (a broadcast along the middle axis) and causally (the lane's coordinates), then returns it, its transpose
(``tilefish.contiguous`` of a view whose strides are ``(Lb*Lb, 1, Lb)``, handed to the generated op as scalars), the head ``[1]`` (a dense view at
the offset ``Lb*Lb``, copied) and the maximum of every row (a row kernel over the reshape ``[H * Lb, Lb]``).

No ``from __future__ import annotations``: tilefish evaluates annotations.
"""

import broadcast_cases as bc

import tilefish
from tilefish import Dim, Tensor, f32, u8

H = bc.H
Lb = Dim("Lb", buckets=[8, 24])
Rm = Dim("Rm")


@tilefish.kernel
def row_max_lb(x: Tensor[Rm, Lb, f32]) -> Tensor[Rm, f32]:
    return tilefish.max(x, axis=-1)


@tilefish.graph(outputs=("masked", "transposed", "head1", "row_max"))
def attention_mask(q: Tensor[H, Lb, 1, f32], k: Tensor[H, 1, Lb, f32], keep: Tensor[H, 1, Lb, u8]):
    s = bc.outer_heads(q, k)  # [H, Lb, Lb], an intermediate with the stride Lb*Lb
    m = bc.causal_heads(bc.key_mask_heads(s, keep))
    return m, tilefish.contiguous(m.permute(0, 2, 1)), m[1], row_max_lb(m.reshape(H * Lb, Lb))


def attention_mask_torch(q, k, keep):
    """The same in torch: the outer product, the key mask, the causal mask, and what the graph returns."""
    import torch

    s = q * k
    s = torch.where(keep != 0, s, float("-inf"))
    n = s.shape[-1]
    i = torch.arange(n, device=s.device)
    s = torch.where(i[None, :] <= i[:, None], s, float("-inf"))
    return s, s.permute(0, 2, 1).contiguous(), s[1].contiguous(), s.reshape(H * n, n).amax(dim=-1)
