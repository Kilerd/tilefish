"""Graphs whose bodies compute (``docs/rfcs/0001-plan-fusion.md``): the use cases of fusion's first version, each with its unfused twin
(``grouping="none"``), for ``test_fusion.py`` (no GPU), ``test_gpu_fusion.py`` and the Rust gate (``rust_gate.py``).

No ``from __future__ import annotations``: tilefish evaluates annotations at declaration.
"""

import tilefish
from tilefish import DevScalar, Dim, InOut, Tensor, bf16, f16, f32, i32, u8

R = Dim("R", buckets=[8, 24])
D = Dim("D", 64)
N = Dim("N", buckets=[1000, 4096])

UNFUSED = tilefish.fusion(grouping="none")


@tilefish.kernel
def fg_scale_rows(x: Tensor[R, D, f32]) -> Tensor[R, D, f32]:
    """A declared operator between two regions (a region never spans a call)."""
    return x * 2.0


def _bias_act(x, b):
    return tilefish.silu(x.to(f32) + b.to(f32)).to(bf16)


@tilefish.graph
def bias_act(x: Tensor[R, D, bf16], b: Tensor[D, bf16]):
    """Bias and SiLU after a GEMM: ``b`` broadcasts over the rows."""
    return _bias_act(x, b)


@tilefish.graph(fusion=UNFUSED)
def bias_act_unfused(x: Tensor[R, D, bf16], b: Tensor[D, bf16]):
    return _bias_act(x, b)


@tilefish.graph(fusion=tilefish.fusion(math="ieee"))
def bias_act_ieee(x: Tensor[R, D, bf16], b: Tensor[D, bf16]):
    """The same in IEEE math (torch's silu)."""
    return _bias_act(x, b)


@tilefish.graph(fusion=tilefish.fusion(math="ieee", grouping="none"))
def bias_act_ieee_unfused(x: Tensor[R, D, bf16], b: Tensor[D, bf16]):
    return _bias_act(x, b)


def _swiglu(x):
    g, u = x.split(D, axis=-1)  # two column blocks of one tensor: views that are not dense
    return (tilefish.silu(g.to(f32)) * u.to(f32)).to(bf16)


@tilefish.graph
def swiglu(x: Tensor[R, 2 * D, bf16]):
    return _swiglu(x)


@tilefish.graph(fusion=UNFUSED)
def swiglu_unfused(x: Tensor[R, 2 * D, bf16]):
    return _swiglu(x)


def _residual(x, res):
    h = res + x.to(f32)  # the residual stream, f32
    res[...] = h  # a side output, into the caller's InOut tensor
    return (h * 0.5).to(bf16)


@tilefish.graph
def residual(x: Tensor[R, D, bf16], res: InOut[Tensor[R, D, f32]]):
    return _residual(x, res)


@tilefish.graph(fusion=UNFUSED)
def residual_unfused(x: Tensor[R, D, bf16], res: InOut[Tensor[R, D, f32]]):
    return _residual(x, res)


def _masked(s, b, n):
    keep = tilefish.iota(s, 0) < tilefish.load(n)  # the positions below a device-scalar count
    y = tilefish.where(keep, s * 2.0 + b, -1.0)
    return y, tilefish.barrier(y * y)


@tilefish.graph
def masked(s: Tensor[N, f32], b: Tensor[N, f32], n: DevScalar[i32]):
    return _masked(s, b, n)


@tilefish.graph(fusion=UNFUSED)
def masked_unfused(s: Tensor[N, f32], b: Tensor[N, f32], n: DevScalar[i32]):
    return _masked(s, b, n)


def _dequant(q, scale):
    return ((q.to(f32) - 128.0) * scale).to(f16)  # u8 codes, a scale per row ([R, 1])


@tilefish.graph
def dequant(q: Tensor[R, D, u8], scale: Tensor[R, 1, f32]):
    return _dequant(q, scale)


@tilefish.graph(fusion=UNFUSED)
def dequant_unfused(q: Tensor[R, D, u8], scale: Tensor[R, 1, f32]):
    return _dequant(q, scale)


def _periodic(x, a):
    s = tilefish.sin(a * x)  # x + sin(a x)^2 / a, a per channel
    return x + s * s / a


@tilefish.graph
def periodic(x: Tensor[R, D, f32], a: Tensor[D, f32]):
    return _periodic(x, a)


@tilefish.graph(fusion=UNFUSED)
def periodic_unfused(x: Tensor[R, D, f32], a: Tensor[D, f32]):
    return _periodic(x, a)


def _around_call(x):
    y = x * 3.0
    z = fg_scale_rows(y)
    return (y + z) * z  # a region before the call and one after it (they cannot be one: the call reads the first and the second reads the call)


@tilefish.graph
def around_call(x: Tensor[R, D, f32]):
    return _around_call(x)


@tilefish.graph(fusion=UNFUSED)
def around_call_unfused(x: Tensor[R, D, f32]):
    return _around_call(x)


def _modes(a, g):
    return a * a + 1.0, (g * 0.1 + g).to(f32) * a - a * 3.0


@tilefish.graph(fusion=tilefish.fusion(fp_contract="mul-add", intermediates="opmath"))
def modes(a: Tensor[R, D, f32], g: Tensor[R, D, bf16]):
    """``a * a + 1`` one fma; ``g * 0.1`` torch's way (in f32 with f32(0.1), rounded once to bf16); ``... - a * 3`` one fma of -a."""
    return _modes(a, g)


@tilefish.graph(fusion=tilefish.fusion(fp_contract="mul-add", intermediates="opmath", grouping="none"))
def modes_unfused(a: Tensor[R, D, f32], g: Tensor[R, D, bf16]):
    return _modes(a, g)


def _row_sum(x, w):
    s = tilefish.sum(x * x, axis=-1, keepdims=True)  # a reduction without an order: a kernel of its own (D2)
    return (x * tilefish.rsqrt(s * (1.0 / 64) + 1e-6) * w).to(bf16)


@tilefish.graph
def rms(x: Tensor[R, D, f32], w: Tensor[D, f32]):
    return _row_sum(x, w)


@tilefish.graph(fusion=UNFUSED)
def rms_unfused(x: Tensor[R, D, f32], w: Tensor[D, f32]):
    return _row_sum(x, w)


# (fused graph, its unfused twin, how to make its arguments: {parameter: (shape at a bucket of R / N, dtype name)})
CASES = {
    "bias_act": (bias_act, bias_act_unfused),
    "bias_act_ieee": (bias_act_ieee, bias_act_ieee_unfused),
    "swiglu": (swiglu, swiglu_unfused),
    "residual": (residual, residual_unfused),
    "masked": (masked, masked_unfused),
    "dequant": (dequant, dequant_unfused),
    "periodic": (periodic, periodic_unfused),
    "around_call": (around_call, around_call_unfused),
    "modes": (modes, modes_unfused),
    "rms": (rms, rms_unfused),
}
