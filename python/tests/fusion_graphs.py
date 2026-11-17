"""Graphs whose bodies compute (``docs/en/rfcs/0001-plan-fusion.md``): the use cases of fusion's first version, each with its unfused twin
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


def _views(x, y):
    t = x.permute(1, 0)  # a transpose: no axis of stride 1
    a, b = y.split(D, axis=-1)  # two column blocks
    return (t * a + b).to(bf16)


@tilefish.graph
def views(x: Tensor[D, R, f32], y: Tensor[R, 2 * D, f32]):
    """Views that are not dense, read where they are by the region (a stride per axis)."""
    return _views(x, y)


@tilefish.graph(fusion=UNFUSED)
def views_unfused(x: Tensor[D, R, f32], y: Tensor[R, 2 * D, f32]):
    return _views(x, y)


# -- row regions (fusion version 2, the default): ordered reductions fused with the ops around them -----------------------------------------------

DR = Dim("DR", buckets=[96, 1000, 2304])  # a row shorter than one round of the lanes, one just under it, one of two rounds and a quarter
DS = Dim("DS", 1024)
NT = Dim("NT", 50)
T128 = tilefish.tree(128, offsets=(16, 8, 4, 2, 1, 64, 32), vector=8)  # 128 threads of 8 columns, a warp-shuffle tree, then the 4 warps
V1 = tilefish.fusion(version=1)


def _add_rmsnorm(x, res, w):
    h = x + res  # the residual stream: returned too (a side output of the region)
    h32 = h.to(f32)
    ms = tilefish.mean(h32 * h32, -1, keepdims=True, order=T128)
    return (h32 * tilefish.rsqrt(ms + 1e-6) * w.to(f32)).to(bf16), h


@tilefish.graph
def add_rmsnorm(x: Tensor[R, DR, bf16], res: Tensor[R, DR, bf16], w: Tensor[DR, bf16]):
    """A residual add and an rmsnorm, the residual sum a second output: one row region."""
    return _add_rmsnorm(x, res, w)


@tilefish.graph(fusion=UNFUSED)
def add_rmsnorm_unfused(x: Tensor[R, DR, bf16], res: Tensor[R, DR, bf16], w: Tensor[DR, bf16]):
    return _add_rmsnorm(x, res, w)


@tilefish.graph(fusion=V1)
def add_rmsnorm_v1(x: Tensor[R, DR, bf16], res: Tensor[R, DR, bf16], w: Tensor[DR, bf16]):
    """The same in fusion version 1: the mean a region of its own."""
    return _add_rmsnorm(x, res, w)


def _rmsnorm_fma(x, w):
    v = x.to(f32)
    ss = tilefish.sum(v * v, -1, keepdims=True, order=T128, fma=True)  # acc = fma(v, v, acc): the multiply is part of the sum
    return (v * tilefish.rsqrt(ss / 1024.0 + 1e-6) * w.to(f32)).to(bf16)


@tilefish.graph
def rmsnorm_fma(x: Tensor[R, DS, bf16], w: Tensor[DS, bf16]):
    return _rmsnorm_fma(x, w)


@tilefish.graph(fusion=UNFUSED)
def rmsnorm_fma_unfused(x: Tensor[R, DS, bf16], w: Tensor[DS, bf16]):
    return _rmsnorm_fma(x, w)


def _layernorm_gelu(x, w, b, s):
    v = x.to(f32)
    mu = tilefish.mean(v, -1, keepdims=True, order=tilefish.tree(256))
    d = v - mu
    var = tilefish.mean(d * d, -1, keepdims=True, order=tilefish.tree(256))
    n = d * tilefish.rsqrt(var + 1e-5) * w + b
    return (tilefish.gelu(n) * s).to(bf16)  # an elementwise tail with a per-channel scale


@tilefish.graph
def layernorm_gelu(x: Tensor[R, DR, bf16], w: Tensor[DR, f32], b: Tensor[DR, f32], s: Tensor[DR, f32]):
    """A layer norm (two ordered means) and an elementwise tail: one row region."""
    return _layernorm_gelu(x, w, b, s)


@tilefish.graph(fusion=UNFUSED)
def layernorm_gelu_unfused(x: Tensor[R, DR, bf16], w: Tensor[DR, f32], b: Tensor[DR, f32], s: Tensor[DR, f32]):
    return _layernorm_gelu(x, w, b, s)


def _softmax(s, n):
    z = s * 0.125
    z = tilefish.where(tilefish.iota(z, -1) < tilefish.load(n), z, float("-inf"))  # the columns below a device-scalar length
    m = tilefish.max(z, -1, keepdims=True)  # order-free
    e = tilefish.exp(z - m)
    return (e / tilefish.sum(e, -1, keepdims=True, order=tilefish.tree(32))).to(bf16)


@tilefish.graph
def softmax(s: Tensor[R, DR, f32], n: DevScalar[i32]):
    """Scale, mask, max, exp, an ordered sum and a divide: one row region."""
    return _softmax(s, n)


@tilefish.graph(fusion=UNFUSED)
def softmax_unfused(s: Tensor[R, DR, f32], n: DevScalar[i32]):
    return _softmax(s, n)


def _quantise(x):
    v = x.to(f32)
    amax = tilefish.max(tilefish.where(v < 0.0, -v, v), -1, keepdims=True)
    scale = tilefish.where(amax > 0.0, amax * (1.0 / 448.0), 1.0)
    return (v / scale).to(tilefish.e4m3), scale  # satfinite; the scale per row a second output


@tilefish.graph
def quantise(x: Tensor[R, DR, bf16]):
    """Row-wise fp8 quantisation: the absolute maximum, a scale per row, the cast. One row region, two outputs."""
    return _quantise(x)


@tilefish.graph(fusion=UNFUSED)
def quantise_unfused(x: Tensor[R, DR, bf16]):
    return _quantise(x)


def _split_norm(x):
    g, u = x.split(DR, axis=-1)  # two column blocks: views a row region reads where they are
    h = tilefish.silu(g.to(f32)) * u.to(f32)
    return (h * tilefish.rsqrt(tilefish.mean(h * h, -1, keepdims=True, order=T128) + 1e-6)).to(bf16)


@tilefish.graph
def split_norm(x: Tensor[R, 2 * DR, bf16]):
    """SwiGLU of two column blocks, then a norm: one row region over two strided views, no copy."""
    return _split_norm(x)


@tilefish.graph(fusion=UNFUSED)
def split_norm_unfused(x: Tensor[R, 2 * DR, bf16]):
    return _split_norm(x)


def _embed_pick(table, idx, b):
    e = tilefish.gather_rows(table, idx, oob="zero").to(f32)  # rows of a table at device indices
    z = e * 2.0 + b
    return z, tilefish.argmax(z, -1, dtype=i32)  # the first maximum of each row


@tilefish.graph
def embed_pick(table: Tensor[NT, DR, bf16], idx: Tensor[R, i32], b: Tensor[DR, f32]):
    """A gather, an affine and an argmax: one row region."""
    return _embed_pick(table, idx, b)


@tilefish.graph(fusion=UNFUSED)
def embed_pick_unfused(table: Tensor[NT, DR, bf16], idx: Tensor[R, i32], b: Tensor[DR, f32]):
    return _embed_pick(table, idx, b)


# the row regions of fusion version 2
ROW_CASES = ("add_rmsnorm", "rmsnorm_fma", "layernorm_gelu", "softmax", "quantise", "split_norm", "embed_pick")

# (fused graph, its unfused twin, how to make its arguments: {parameter: (shape at a bucket of R / N, dtype name)})
CASES = {
    "add_rmsnorm": (add_rmsnorm, add_rmsnorm_unfused),
    "add_rmsnorm_v1": (add_rmsnorm_v1, add_rmsnorm_unfused),
    "rmsnorm_fma": (rmsnorm_fma, rmsnorm_fma_unfused),
    "layernorm_gelu": (layernorm_gelu, layernorm_gelu_unfused),
    "softmax": (softmax, softmax_unfused),
    "quantise": (quantise, quantise_unfused),
    "split_norm": (split_norm, split_norm_unfused),
    "embed_pick": (embed_pick, embed_pick_unfused),
    "views": (views, views_unfused),
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
