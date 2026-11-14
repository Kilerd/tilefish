"""Copies that tilefish writes the Triton for: ``contiguous`` (a view, materialised), ``im2col`` and ``im2col1d`` (the patches of a 2-D and of a
1-D convolution); and ``col2im1d``, the other direction (the overlap-add of a 1-D transposed convolution), which adds.

None of them is a ``@tilefish.kernel`` body: a lowered kernel takes contiguous tensors, and each reads a strided view. Each is a *generated op*:
an ``extern`` operator whose Triton source tilefish generates from a fixed template, whose input is ``Strided[...]`` (so a view goes to it as it
is, with its strides as scalar arguments, through the machinery every strided ``extern`` uses: compile specs, launches, manifests, plans, the Rust
host), and which has a torch reference. One op exists per rank and dtype (``contiguous``), per dtype, window and channel count (``im2col``),
per dtype, window, column order and channel count (``im2col1d``) or per dtype, window, output padding, column order, layout, bias and channel
count (``col2im1d``), made the first time it is asked for and kept; called in a ``@tilefish.graph``, it is a step of the plan.

All but ``col2im1d`` are raw copies: values are loaded and stored with no arithmetic, so a NaN keeps its payload and a zero its sign, and any
tiling gives the same bytes. Offsets are ``i64`` (``index_bits=64``): the strides are ``i64`` scalars, and no size limit applies.

``contiguous(v)`` copies the view ``v`` (rank 1 to 4) into a new contiguous tensor of its shape, in its logical (row-major) order. A program
copies a ``32 x 32`` tile of the last two axes for one index of the third axis from the end; a rank-4 view loops over its first axis inside the
program.

``im2col(x, kernel=(KH, KW), stride=(SH, SW), padding=(PH, PW))`` takes ``x: [N, C, H, W]`` (any strides) and gives ``[N, Ho, Wo, C * KH * KW]``
with ``Ho = (H + 2 PH - KH) // SH + 1`` (and ``Wo`` likewise): ``out[n, oh, ow, c * KH * KW + kh * KW + kw] = x[n, c, oh * SH - PH + kh, ow * SW - PW +
kw]`` where that position is inside ``x``, and ``+0.0`` (all bits clear: the load is masked) where it is in the padding. The columns are
channel-major, the order of a ``Conv2d`` weight flattened (``F.unfold``'s), so the weight is the GEMM's operand as it is. One program per output
pixel (grid ``(Wo, Ho, N)``), a loop over the columns in blocks (one block of ``next_pow2(C * KH * KW)`` up to 1024 when ``C`` is static).

``im2col1d(x, kernel=K, stride=S, dilation=D, padding=(PL, PR), order=...)`` takes ``x: [N, C, L]`` (any strides: a channel-last ``[N, L, C]``
activation permuted, for one) and gives ``[N, Lo, C * K]`` with ``Lo = (L + PL + PR - D * (K - 1) - 1) // S + 1``. Tap ``j`` of output position
``t`` is ``x[n, c, t * S - PL + j * D]`` where that position is inside ``[0, L)`` and ``+0.0`` (all bits clear) where it is not; a causal
convolution pads ``(D * (K - 1), 0)``. ``order="channel"`` puts it in column ``c * K + j`` (a ``Conv1d`` weight ``[C_out, C, K]`` flattened),
``order="tap"`` in column ``j * C + c`` (the weight permuted to ``[C_out, K, C]``): the columns are the reduction axis of the GEMM that consumes
them, so both orders exist and the weight is flattened to match. One program per output position (grid ``(Lo, N)``), a loop over the columns in
blocks as ``im2col``; in the tap order consecutive lanes walk the channels of one tap (a channel-last input is read in contiguous runs), in the
channel order the taps of one channel.

``col2im1d(cols, kernel=K, stride=S, dilation=D, padding=(PL, PR), output_padding=OP, order=..., bias=b, layout=...)`` is the overlap-add of a
1-D transposed convolution. ``cols: [N, Li, C * K]`` (any strides: a GEMM's output reshaped, or a transposed view of one, goes in as it is) holds,
for input position ``i``, the products of a ``ConvTranspose1d`` weight ``W: [C_in, C, K]``: ``cols = x^T @ W.reshape(C_in, C * K)``, tap ``j`` of
channel ``c`` in column ``c * K + j`` (``order="channel"``, the weight as it is stored) or ``j * C + c`` (``order="tap"``, the weight permuted to
``[C_in, K, C]``). The result is contiguous, ``y: [N, C, Lo]`` (``layout="ncl"``) or ``[N, Lo, C]`` (``layout="nlc"``), of the dtype of ``cols``,
with ``Lo = (Li - 1) * S - PL - PR + D * (K - 1) + OP + 1``: ``(PL, PR)`` are trimmed from the full output and ``OP`` positions added at its end
(torch's ``conv_transpose1d(padding=p, output_padding=op)`` is ``padding=(p, p), output_padding=op``). ``y[n, c, t]`` is the sum of every
``cols[n, i, col(c, j)]`` with ``t + PL = i * S + j * D`` (``0 <= i < Li``, ``0 <= j < K``), plus ``b[c]``.

The order of the adds is the contract, because it decides the bits: float addition is not associative, and three terms or more added in another
order round differently (with two, any order gives the same sum). It is the order of ATen's CUDA ``col2im`` (``col2im_device`` in
``aten/src/ATen/native/cuda/im2col.cuh``, which ``F.fold`` and torch's transposed convolution without cuDNN run on CUDA: ``val = 0``, then
``val += data_col[...]`` over the input positions in ascending order, in the accumulation type, then one cast at the store), so the bytes are
``F.fold``'s on CUDA. Precisely: an f32 accumulator starts at ``+0.0`` and adds the terms in ascending input position ``i`` (descending tap
``j``: ``i * S = t + PL - j * D``), each add one f32 rounding; a 16-bit value is widened exactly first; a position no term reaches stays ``+0.0``;
then ``+ b[c]`` (widened) in f32, and one rounding to the dtype of ``cols`` at the store (round to nearest even). There is no multiply, the ops
are compiled with ``enable_fp_fusion=False`` all the same, and nothing is reassociated. (ATen's CPU ``col2im`` adds tap by tap, ascending ``j``,
in the dtype of the data; cuDNN chooses its own algorithm.) A tap that reaches no input position adds a masked ``+0.0``, which changes nothing:
the accumulator is never ``-0.0`` (it starts at ``+0.0``, and a sum that cancels is ``+0.0`` under round to nearest). Subnormals are kept (no
flush to zero), as ATen keeps them; a NaN anywhere in a sum gives a NaN, whose payload is the device's. Each output element is the sum of one
lane, so the tiling and the warp count do not change the bits. The torch reference adds the same terms in the same order (one tensor add per
tap, from the last tap to the first, ``+0.0`` where a tap reaches nothing), on any device.

One program per output position and batch index (grid ``(Lo, N)``), a loop over the channels in blocks (``next_pow2(C)`` up to 1024 when ``C``
is static, else 1024), the taps unrolled (``tl.static_range``) from the last to the first; a static channel count is compiled in, a runtime one is
a scalar. ``Lo`` follows ``Li`` at each call; a call whose ``Lo`` would be less than 1 is refused at launch (its grid is empty), and ``col2im1d``
in a graph refuses a ``cols`` whose type allows such a length.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any

from .decorators import ExternOp
from .dims import Dim, DimExpr, as_dim, cdiv
from .dtypes import DType
from .errors import TilefishTypeError
from .lower import NUM_STAGES, LoweredKernel, _TL
from .signature import parse_signature
from .trace import Value
from .types import Out, StrideOf, Strided, Tensor, const

_lock = threading.Lock()
_ops: "dict[tuple, ExternOp]" = {}
_MODULE = "tilefish_generated"
TILE_ELEMENTS = 2048  # contiguous: a program copies a tile of about this many elements of the last two axes per step
COPY_WARPS = 4


def _generated(
    name: str,
    source: str,
    stub_source: str,
    namespace: "dict[str, Any]",
    grid: Any,
    num_warps: int,
    reference: Any,
    fp_fusion: "bool | None" = None,
) -> ExternOp:
    """An extern op over the generated Triton ``source`` (module ``tilefish_generated.<name>``, function ``tf_<name>``), declared by the stub in
    ``stub_source`` (evaluated in ``namespace``; its parameters are the Triton function's, in order), with the source attached so that every
    mode runs it. ``fp_fusion`` is Triton's ``enable_fp_fusion`` (``None``: not given, Triton's default; the copies do no arithmetic)."""
    module, fn_name = f"{_MODULE}.{name}", f"tf_{name}"
    ns = dict(namespace)
    exec(compile(stub_source, f"<tilefish generated stub {name}>", "exec"), ns)  # noqa: S102 - text generated here
    stub = ns[fn_name]
    stub.__module__ = module
    sig = parse_signature(stub, what="extern op", name=name, allow_strided=True)
    op = ExternOp(
        stub,
        name,
        sig,
        f"{module}:{fn_name}",
        num_warps,
        NUM_STAGES,
        grid,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        index_bits=64,
        reference=reference,
        fp_fusion=fp_fusion,
    )
    op.__dict__["_tf_generated"] = LoweredKernel(name, fn_name, module, "generated", source, op, num_warps, None)
    return op


def generated_source(op: ExternOp) -> "str | None":
    """The Triton source tilefish generated for ``op`` (``None`` for a hand-written extern)."""
    low = op.__dict__.get("_tf_generated")
    return None if low is None else low.source


def _cached(key: tuple, make: Any) -> ExternOp:
    with _lock:
        op = _ops.get(key)
    if op is None:
        op = make()
        with _lock:
            op = _ops.setdefault(key, op)
    return op


def _check_storable(what: str, x: Value) -> DType:
    dt = x.dtype
    if not isinstance(dt, DType) or dt.is_bool or dt.name not in _TL:
        raise TilefishTypeError(f"{what}: {x.ref} has dtype {dt}, which is not a plain storage type")
    return dt


# -- contiguous -------------------------------------------------------------------------------------------------------------------


def _tiles(rank: int, last: "int | None") -> "tuple[int, int]":
    """The tile of the last two axes a program copies: ``(rows, columns)``. The columns fit a static last axis (a power of two up to 128),
    else 64; the rows make a tile of about 2048 elements (up to 256 rows). A rank-1 view is cut in blocks of 1024."""
    if rank == 1:
        return 1, 1024
    cols = 64 if last is None else min(128, 1 << max(0, (last - 1).bit_length()))
    return min(256, max(1, TILE_ELEMENTS // cols)), cols


def _contiguous_op(rank: int, dtype: DType, last: "int | None") -> ExternOp:
    tr, tc = _tiles(rank, last)
    name = f"contiguous_r{rank}_{dtype.name}" + ("" if last is None else f"_last{last}") + f"_t{tr}x{tc}"
    a = [f"a{k}" for k in range(rank)]
    s = [f"x_s{k}" for k in range(rank)]
    if rank == 1:
        body = [
            "    c = tl.program_id(0) * TC + tl.arange(0, TC)",
            "    m = c < a0",
            "    v = tl.load(x + c.to(tl.int64) * x_s0, mask=m)",
            "    tl.store(y + c, v, mask=m)",
        ]
    else:
        # rows: the axis before the last, cut in tiles of TR; columns: the last axis, walked in chunks of TC; the axes before: the grid
        lead = [f"i{k}" for k in range(rank - 2)]
        body = ["    r = tl.program_id(0) * TR + tl.arange(0, TR)[:, None]"]
        for k, name_k in enumerate(lead):
            body.append(f"    {name_k} = tl.program_id({rank - 2 - k}).to(tl.int64)")
        src_base = " + ".join([f"{name_k} * {s[k]}" for k, name_k in enumerate(lead)] or ["0"])
        dst_rows = "0"
        for k, name_k in enumerate(lead):
            dst_rows = f"({dst_rows}) * {a[k]} + {name_k}" if dst_rows != "0" else name_k
        dst_base = f"(({dst_rows}) * {a[-2]} + r) * {a[-1]}" if lead else f"r.to(tl.int64) * {a[-1]}"
        body += [
            f"    src = {src_base} + r * {s[-2]}" if lead else f"    src = r.to(tl.int64) * {s[-2]}",
            f"    dst = {dst_base}",
            f"    rm = r < {a[-2]}",
            f"    for c0 in range(0, {a[-1]}, TC):",
            "        c = c0 + tl.arange(0, TC)[None, :]",
            f"        m = rm & (c < {a[-1]})",
            f"        v = tl.load(x + src + c * {s[-1]}, mask=m)",
            "        tl.store(y + dst + c, v, mask=m)",
        ]
    params = ["x", "y", *a, *s, *(["TR: tl.constexpr"] if rank > 1 else []), "TC: tl.constexpr"]
    source = "\n".join(
        [
            f"# Generated by tilefish (tilefish.contiguous) for a rank-{rank} {dtype.name} view, tiles of {tr} x {tc}. Do not edit.",
            "# A raw copy of the view into a contiguous tensor of its shape, in row-major order; offsets are i64.",
            "import triton",
            "import triton.language as tl",
            "",
            "",
            "@triton.jit",
            f"def tf_{name}(",
            *[f"    {q}," for q in params],
            "):",
            *body,
        ]
    ) + "\n"
    dims = [f"A{k}" for k in range(rank)]
    shape_text = ", ".join(dims[:-1] + ([str(last)] if last is not None else [dims[-1]]))
    stub = "\n".join(
        [
            f"def tf_{name}(",
            f"    x: Strided[Tensor[{shape_text}, dt]],",
            f"    y: Out[Tensor[{shape_text}, dt]],",
            *[f"    a{k}: A{k}," for k in range(rank - (1 if last is not None else 0))],
            *(["    *,", f"    a{rank - 1}: const = {last},"] if last is not None else []),  # keyword-only: a default before the strides
            *[f"    x_s{k}: StrideOf('x', {k})," for k in range(rank)],
            *([f"    TR: const = {tr},"] if rank > 1 else []),
            f"    TC: const = {tc},",
            "): ...",
        ]
    )
    shape = {d: Dim(d) for d in dims}
    if rank == 1:
        grid: "tuple[Any, ...]" = (as_dim(cdiv(shape[dims[0]] if last is None else last, tc)),)
    else:
        grid = (as_dim(cdiv(shape[dims[-2]], tr)), *[shape[dims[k]] for k in reversed(range(rank - 2))])
    namespace = {"Strided": Strided, "Tensor": Tensor, "Out": Out, "StrideOf": StrideOf, "const": const, "dt": dtype, **shape}
    if last is not None:  # a static last axis is a constexpr of the source, not a runtime parameter
        source = source.replace(f"    a{rank - 1},\n", f"    a{rank - 1}: tl.constexpr,\n")

    def reference(x: Any) -> Any:
        import torch

        return x.clone(memory_format=torch.contiguous_format)

    return _generated(name, source, stub, namespace, grid, COPY_WARPS, reference)


def contiguous(x: Any) -> Value:
    """``tilefish.contiguous(view)``: a copy of a strided view (rank 1 to 4) as a contiguous tensor of its shape, in its logical order.

    A step of a ``@tilefish.graph`` (a generated op, see :mod:`tilefish._copyops`) and a raw copy: the bytes are the view's elements as they
    are. For a consumer that needs a dense operand no view expresses (``x.permute(0, 2, 3, 1)`` before a GEMM).
    """
    if not isinstance(x, Value):
        raise TilefishTypeError(f"contiguous: takes a traced tensor (a view), got {type(x).__name__}")
    if not 1 <= x.rank <= 4:
        raise TilefishTypeError(f"contiguous: {x.ref} ({x.type}) has rank {x.rank}; views of rank 1 to 4 are supported")
    dt = _check_storable("contiguous", x)
    last_dim = x.shape[-1].folded()
    last = last_dim.value if last_dim.is_static else None
    return contiguous_op(x.rank, dt, last)(x)


# -- im2col -----------------------------------------------------------------------------------------------------------------------


def _pair(what: str, v: Any, low: int) -> "tuple[int, int]":
    pair = (v, v) if isinstance(v, int) and not isinstance(v, bool) else tuple(v) if isinstance(v, (tuple, list)) else None
    if pair is None or len(pair) != 2 or any(isinstance(e, bool) or not isinstance(e, int) or e < low for e in pair):
        raise TilefishTypeError(f"im2col: {what} must be an int or a pair of ints of at least {low}, got {v!r}")
    return pair  # type: ignore[return-value]


def _im2col_op(dtype: DType, k: "tuple[int, int]", s: "tuple[int, int]", p: "tuple[int, int]", channels: "int | None") -> ExternOp:
    (kh, kw), (sh, sw), (ph, pw) = k, s, p
    taps = kh * kw
    name = f"im2col_{dtype.name}_k{kh}x{kw}_s{sh}x{sw}_p{ph}x{pw}" + ("" if channels is None else f"_c{channels}")
    block = 1024 if channels is None else min(1024, 1 << (channels * taps - 1).bit_length())
    warps = 4 if block >= 512 else 1
    ncol = "C * " + str(taps) if channels is None else str(channels * taps)
    params = ["x", "out", *(["C"] if channels is None else []), "H", "W", "Ho", "Wo", "x_s0", "x_s1", "x_s2", "x_s3", "BLOCK: tl.constexpr"]
    source = "\n".join(
        [
            f"# Generated by tilefish (tilefish.im2col) for {dtype.name}, kernel {kh}x{kw}, stride {sh}x{sw}, padding {ph}x{pw}"
            + ("" if channels is None else f", {channels} channels")
            + ". Do not edit.",
            "# out[n, oh, ow, c*KH*KW + kh*KW + kw] = x[n, c, oh*SH - PH + kh, ow*SW - PW + kw], +0.0 in the padding: a raw copy, i64 offsets.",
            "import triton",
            "import triton.language as tl",
            "",
            "",
            "@triton.jit",
            f"def tf_{name}(",
            *[f"    {q}," for q in params],
            "):",
            "    ow = tl.program_id(0)",
            "    oh = tl.program_id(1)",
            "    n = tl.program_id(2).to(tl.int64)",
            f"    ncol = {ncol}",
            "    row = (n * Ho + oh) * Wo + ow",
            "    for c0 in range(0, ncol, BLOCK):",
            "        col = c0 + tl.arange(0, BLOCK)",
            "        cm = col < ncol",
            f"        ch = col // {taps}",
            f"        tap = col % {taps}",
            f"        ih = oh * {sh} - {ph} + tap // {kw}",
            f"        iw = ow * {sw} - {pw} + tap % {kw}",
            "        ok = cm & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)",
            "        v = tl.load(x + n * x_s0 + ch * x_s1 + ih * x_s2 + iw * x_s3, mask=ok, other=0)",
            "        tl.store(out + row * ncol + col, v, mask=cm)",
        ]
    ) + "\n"
    c_text = "CH" if channels is None else str(channels)
    stub = "\n".join(
        [
            f"def tf_{name}(",
            f"    x: Strided[Tensor[NB, {c_text}, HI, WI, dt]],",
            f"    out: Out[Tensor[NB, HO, WO, {c_text} * {taps}, dt]],",
            *(["    C: CH,"] if channels is None else []),
            "    H: HI,",
            "    W: WI,",
            "    Ho: HO,",
            "    Wo: WO,",
            "    x_s0: StrideOf('x', 0),",
            "    x_s1: StrideOf('x', 1),",
            "    x_s2: StrideOf('x', 2),",
            "    x_s3: StrideOf('x', 3),",
            f"    BLOCK: const = {block},",
            "): ...",
        ]
    )
    nb, hi, wi = Dim("N"), Dim("H"), Dim("W")
    ho = as_dim(cdiv(hi + (2 * ph - kh + 1), sh))  # (H + 2p - k) // s + 1 for H + 2p >= k
    wo = as_dim(cdiv(wi + (2 * pw - kw + 1), sw))
    namespace = {"Strided": Strided, "Tensor": Tensor, "Out": Out, "StrideOf": StrideOf, "const": const, "dt": dtype}
    namespace.update({"NB": nb, "HI": hi, "WI": wi, "HO": ho, "WO": wo, "CH": Dim("C")})

    def reference(x: Any) -> Any:
        import torch.nn.functional as F

        patches = F.pad(x, (pw, pw, ph, ph)).unfold(2, kh, sh).unfold(3, kw, sw)  # [N, C, Ho, Wo, KH, KW], +0.0 in the padding
        n_, c_, ho_, wo_ = patches.shape[:4]
        return patches.permute(0, 2, 3, 1, 4, 5).reshape(n_, ho_, wo_, c_ * kh * kw).contiguous()

    return _generated(name, source, stub, namespace, (wo, ho, nb), warps, reference)


def im2col(x: Any, kernel: Any = (3, 3), stride: Any = (1, 1), padding: Any = (0, 0)) -> Value:
    """``tilefish.im2col(x, kernel=(KH, KW), stride=(SH, SW), padding=(PH, PW))``: the patches of a 2-D convolution of ``x: [N, C, H, W]`` (any
    strides: an NHWC tensor permuted to NCHW, for one), as ``[N, Ho, Wo, C * KH * KW]`` with channel-major columns and ``+0.0`` padding.

    A step of a ``@tilefish.graph`` (a generated op, see :mod:`tilefish._copyops`) and a raw copy. Reshaped to ``[N * Ho * Wo, C * KH * KW]`` it is
    the left operand of the convolution as a GEMM against the ``[C_out, C * KH * KW]`` weight.
    """
    if not isinstance(x, Value):
        raise TilefishTypeError(f"im2col: takes a traced tensor [N, C, H, W], got {type(x).__name__}")
    if x.rank != 4:
        raise TilefishTypeError(f"im2col: {x.ref} ({x.type}) has rank {x.rank}; it takes [N, C, H, W]")
    dt = _check_storable("im2col", x)
    k, s, p = _pair("kernel", kernel, 1), _pair("stride", stride, 1), _pair("padding", padding, 0)
    for axis, size, kk, pp in zip((2, 3), x.shape[2:], k, p):
        if size.is_static and size.value + 2 * pp < kk:
            raise TilefishTypeError(f"im2col: axis {axis} of {x.type} is {size.value}, too small for a {kk}-wide window with padding {pp}")
    c = x.shape[1].folded()
    channels = c.value if c.is_static else None
    return _cached(("im2col", dt.name, k, s, p, channels), lambda: _im2col_op(dt, k, s, p, channels))(x)


# -- im2col1d ---------------------------------------------------------------------------------------------------------------------

_ORDERS = ("channel", "tap")


def _at_least(what: str, v: Any, low: int, who: str = "im2col1d") -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < low:
        raise TilefishTypeError(f"{who}: {what} must be an int of at least {low}, got {v!r}")
    return v


def _window_args(who: str, kernel: Any, stride: Any, dilation: Any, padding: Any, order: Any) -> "tuple[int, int, int, tuple[int, int], str]":
    """The window of a 1-D convolution (``im2col1d``, ``col2im1d``): taps, stride, dilation, padding ``(left, right)`` and column order."""
    k, s, d = _at_least("kernel", kernel, 1, who), _at_least("stride", stride, 1, who), _at_least("dilation", dilation, 1, who)
    pad = (padding, padding) if isinstance(padding, int) and not isinstance(padding, bool) else None
    if isinstance(padding, (tuple, list)):
        pad = tuple(padding)
    if pad is None or len(pad) != 2 or any(isinstance(e, bool) or not isinstance(e, int) or e < 0 for e in pad):
        raise TilefishTypeError(f"{who}: padding must be an int or a pair (left, right) of ints of at least 0, got {padding!r}")
    if not isinstance(order, str) or order not in _ORDERS:
        raise TilefishTypeError(f'{who}: order must be "channel" or "tap", got {order!r}')
    return k, s, d, (pad[0], pad[1]), order


def _im2col1d_args(kernel: Any, stride: Any, dilation: Any, padding: Any, order: Any) -> "tuple[int, int, int, tuple[int, int], str]":
    return _window_args("im2col1d", kernel, stride, dilation, padding, order)


def _im2col1d_op(dtype: DType, k: int, s: int, d: int, pad: "tuple[int, int]", order: str, channels: "int | None") -> ExternOp:
    left, right = pad
    name = f"im2col1d_{dtype.name}_k{k}_s{s}_d{d}_p{left}x{right}_{order}" + ("" if channels is None else f"_c{channels}")
    block = 1024 if channels is None else min(1024, 1 << (channels * k - 1).bit_length())
    warps = 4 if block >= 512 else 1
    c_src = "C" if channels is None else str(channels)
    ncol = f"C * {k}" if channels is None else str(channels * k)
    pos = ("t" if s == 1 else f"t * {s}") + (f" - {left}" if left else "") + (" + tap" if d == 1 else f" + tap * {d}")
    if order == "channel":  # column c * K + j: consecutive lanes walk the taps of one channel
        split = [f"        ch = col // {k}", f"        tap = col % {k}"]
        column = f"c * {k} + j"
    else:  # column j * C + c: consecutive lanes walk the channels of one tap (a channel-last input is read in contiguous runs)
        split = [f"        tap = col // {c_src}", f"        ch = col % {c_src}"]
        column = f"j * {c_src} + c"
    formula = f"# out[n, t, {column}] = x[n, c, {pos.replace('tap', 'j')}]"
    params = ["x", "out", *(["C"] if channels is None else []), "L", "Lo", "x_s0", "x_s1", "x_s2", "BLOCK: tl.constexpr"]
    source = "\n".join(
        [
            f"# Generated by tilefish (tilefish.im2col1d) for {dtype.name}, kernel {k}, stride {s}, dilation {d}, padding {left} before and "
            + f"{right} after, {order} order"
            + ("" if channels is None else f", {channels} channels")
            + ". Do not edit.",
            formula + ", +0.0 outside [0, L): a raw copy, i64 offsets.",
            "import triton",
            "import triton.language as tl",
            "",
            "",
            "@triton.jit",
            f"def tf_{name}(",
            *[f"    {q}," for q in params],
            "):",
            "    t = tl.program_id(0)",
            "    n = tl.program_id(1).to(tl.int64)",
            f"    ncol = {ncol}",
            "    row = n * Lo + t",
            "    for c0 in range(0, ncol, BLOCK):",
            "        col = c0 + tl.arange(0, BLOCK)",
            "        cm = col < ncol",
            *split,
            f"        pos = {pos}",
            "        ok = cm & (pos >= 0) & (pos < L)",
            "        v = tl.load(x + n * x_s0 + ch * x_s1 + pos * x_s2, mask=ok, other=0)",
            "        tl.store(out + row * ncol + col, v, mask=cm)",
        ]
    ) + "\n"
    c_text = "CH" if channels is None else str(channels)
    stub = "\n".join(
        [
            f"def tf_{name}(",
            f"    x: Strided[Tensor[NB, {c_text}, LI, dt]],",
            f"    out: Out[Tensor[NB, LO, {c_text} * {k}, dt]],",
            *(["    C: CH,"] if channels is None else []),
            "    L: LI,",
            "    Lo: LO,",
            "    x_s0: StrideOf('x', 0),",
            "    x_s1: StrideOf('x', 1),",
            "    x_s2: StrideOf('x', 2),",
            f"    BLOCK: const = {block},",
            "): ...",
        ]
    )
    nb, li = Dim("N"), Dim("L")
    lo = as_dim(cdiv(li + (left + right - d * (k - 1)), s))  # (L + left + right - d (K - 1) - 1) // s + 1 for a window that fits
    namespace = {"Strided": Strided, "Tensor": Tensor, "Out": Out, "StrideOf": StrideOf, "const": const, "dt": dtype}
    namespace.update({"NB": nb, "LI": li, "LO": lo, "CH": Dim("C")})

    def reference(x: Any) -> Any:
        import torch.nn.functional as F

        patches = F.pad(x, (left, right)).unfold(2, d * (k - 1) + 1, s)[..., ::d]  # [N, C, Lo, K], +0.0 in the padding
        n_, c_, lo_ = patches.shape[:3]
        cols = patches.permute(0, 2, 1, 3) if order == "channel" else patches.permute(0, 2, 3, 1)
        return cols.reshape(n_, lo_, c_ * k).contiguous()

    return _generated(name, source, stub, namespace, (lo, nb), warps, reference)


def im2col1d(x: Any, kernel: Any, *, stride: Any = 1, dilation: Any = 1, padding: Any = 0, order: str = "channel") -> Value:
    """``tilefish.im2col1d(x, kernel, *, stride=1, dilation=1, padding=0, order="channel")``: the patches of a 1-D convolution of
    ``x: [N, C, L]`` (any strides: a channel-last ``[N, L, C]`` activation ``.permute(0, 2, 1)``, for one), as ``[N, Lo, C * K]``.

    ``kernel`` is the number of taps ``K``, ``stride`` and ``dilation`` are ints of at least 1, and ``padding`` is an int (as many zeros on both
    sides) or a pair ``(left, right)``: a causal convolution is ``padding=(dilation * (K - 1), 0)``. ``Lo = (L + left + right - dilation * (K - 1)
    - 1) // stride + 1``. Tap ``j`` of output position ``t`` is ``x[n, c, t * stride - left + j * dilation]``, ``+0.0`` (all bits clear) where that
    position is outside ``[0, L)``; it is column ``c * K + j`` with ``order="channel"`` (the order of a ``Conv1d`` weight ``[C_out, C, K]``
    flattened) and ``j * C + c`` with ``order="tap"`` (the weight permuted to ``[C_out, K, C]`` and flattened). The columns are the reduction axis
    of the GEMM that consumes them, so the weight must be flattened in the same order.

    A step of a ``@tilefish.graph`` (a generated op, see :mod:`tilefish._copyops`) and a raw copy. Reshaped to ``[N * Lo, C * K]`` it is the left
    operand of the convolution as a GEMM against the ``[C_out, C * K]`` weight.
    """
    if not isinstance(x, Value):
        raise TilefishTypeError(f"im2col1d: takes a traced tensor [N, C, L], got {type(x).__name__}")
    if x.rank != 3:
        raise TilefishTypeError(f"im2col1d: {x.ref} ({x.type}) has rank {x.rank}; it takes [N, C, L]")
    dt = _check_storable("im2col1d", x)
    k, s, d, pad, order = _im2col1d_args(kernel, stride, dilation, padding, order)
    size = x.shape[2].folded()
    if size.is_static and size.value + pad[0] + pad[1] < d * (k - 1) + 1:
        raise TilefishTypeError(
            f"im2col1d: axis 2 of {x.type} is {size.value}, too short for {k} taps at dilation {d} (a window of {d * (k - 1) + 1}) "
            f"with padding {pad}"
        )
    c = x.shape[1].folded()
    channels = c.value if c.is_static else None
    return _cached(("im2col1d", dt.name, k, s, d, pad, order, channels), lambda: _im2col1d_op(dt, k, s, d, pad, order, channels))(x)


# -- col2im1d ---------------------------------------------------------------------------------------------------------------------

_LAYOUTS = ("ncl", "nlc")
_SUMMED = ("f32", "bf16", "f16")  # col2im1d adds: it takes the float types it accumulates in f32


def _col2im1d_args(
    kernel: Any, stride: Any, dilation: Any, padding: Any, output_padding: Any, order: Any, layout: Any
) -> "tuple[int, int, int, tuple[int, int], int, str, str]":
    k, s, d, pad, order = _window_args("col2im1d", kernel, stride, dilation, padding, order)
    extra = _at_least("output_padding", output_padding, 0, "col2im1d")
    if not isinstance(layout, str) or layout not in _LAYOUTS:
        raise TilefishTypeError(f'col2im1d: layout must be "ncl" or "nlc", got {layout!r}')
    return k, s, d, pad, extra, order, layout


def _col2im1d_length(li: int, k: int, s: int, d: int, pad: "tuple[int, int]", extra: int) -> int:
    """``Lo = (Li - 1) * s - left - right + d * (K - 1) + output_padding + 1``."""
    return (li - 1) * s - pad[0] - pad[1] + d * (k - 1) + extra + 1


def _col2im1d_op(
    dtype: DType, k: int, s: int, d: int, pad: "tuple[int, int]", extra: int, order: str, layout: str, bias: bool, channels: "int | None"
) -> ExternOp:
    left, right = pad
    name = (
        f"col2im1d_{dtype.name}_k{k}_s{s}_d{d}_p{left}x{right}_op{extra}_{order}_{layout}"
        + ("_bias" if bias else "")
        + ("" if channels is None else f"_c{channels}")
    )
    block = 1024 if channels is None else min(1024, 1 << (channels - 1).bit_length())
    warps = 4 if block >= 512 else 1
    c_src = "C" if channels is None else str(channels)
    column = f"c * {k} + j" if order == "channel" else f"j * {c_src} + c"
    # output position t is position t + left of the untrimmed output, which tap j of input position i reaches when t + left = i * s + j * d
    lands = ("t" if not left else f"t + {left}") + " = " + ("i" if s == 1 else f"i * {s}") + (" + j" if d == 1 else f" + j * {d}")
    reach = ("t" if not left else f"t + {left}") + (" - j" if d == 1 else f" - j * {d}")
    if s == 1:
        position = [f"            i = {reach}", "            ok = (i >= 0) & (i < Li)"]
    else:
        position = [f"            q = {reach}", f"            i = q // {s}", f"            ok = (q >= 0) & (q % {s} == 0) & (i < Li)"]
    widen = "" if dtype.name == "f32" else ".to(tl.float32)"
    out = f"y + (n * Lo + t) * {c_src} + c" if layout == "nlc" else f"y + (n * {c_src} + c) * Lo + t"
    result = "acc" if dtype.name == "f32" else f"acc.to(tl.{_TL[dtype.name]})"
    params = ["cols", *(["bias"] if bias else []), "y", *(["C"] if channels is None else []), "Li", "Lo"]
    params += ["cols_s0", "cols_s1", "cols_s2", "BLOCK: tl.constexpr"]
    y_index = "y[n, c, t]" if layout == "ncl" else "y[n, t, c]"
    source = "\n".join(
        [
            f"# Generated by tilefish (tilefish.col2im1d) for {dtype.name}, kernel {k}, stride {s}, dilation {d}, padding {left} before and {right} "
            + f"after, output padding {extra}, {order} order, {layout} layout"
            + (", a bias" if bias else "")
            + ("" if channels is None else f", {channels} channels")
            + ". Do not edit.",
            f"# {y_index} = +0.0 + cols[n, i, {column}] + ... over every (i, j) with {lands}, in ascending i (descending j), one f32 add each"
            + (", then + bias[c]" if bias else "")
            + f"; rounded once to {dtype.name} at the store. i64 offsets.",
            "import triton",
            "import triton.language as tl",
            "",
            "",
            "@triton.jit",
            f"def tf_{name}(",
            *[f"    {q}," for q in params],
            "):",
            "    t = tl.program_id(0)",
            "    n = tl.program_id(1).to(tl.int64)",
            "    src = cols + n * cols_s0",
            f"    for c0 in range(0, {c_src}, BLOCK):",
            "        c = c0 + tl.arange(0, BLOCK)",
            f"        cm = c < {c_src}",
            "        acc = tl.zeros([BLOCK], dtype=tl.float32)",
            f"        for r in tl.static_range({k}):",
            "            j = r  # the only tap" if k == 1 else f"            j = {k - 1} - r  # descending taps: ascending input positions",
            *position,
            f"            v = tl.load(src + i * cols_s1 + ({column}) * cols_s2, mask=cm & ok, other=0.0)",
            f"            acc = acc + v{widen}",
            *([f"        acc = acc + tl.load(bias + c, mask=cm, other=0.0){widen}"] if bias else []),
            f"        tl.store({out}, {result}, mask=cm)",
        ]
    ) + "\n"
    c_text = "CH" if channels is None else str(channels)
    out_shape = f"NB, {c_text}, LO" if layout == "ncl" else f"NB, LO, {c_text}"
    stub = "\n".join(
        [
            f"def tf_{name}(",
            f"    cols: Strided[Tensor[NB, LI, {c_text} * {k}, dt]],",
            *([f"    bias: Tensor[{c_text}, dt],"] if bias else []),
            f"    y: Out[Tensor[{out_shape}, dt]],",
            *(["    C: CH,"] if channels is None else []),
            "    Li: LI,",
            "    Lo: LO,",
            "    cols_s0: StrideOf('cols', 0),",
            "    cols_s1: StrideOf('cols', 1),",
            "    cols_s2: StrideOf('cols', 2),",
            f"    BLOCK: const = {block},",
            "): ...",
        ]
    )
    nb, li = Dim("N"), Dim("L")
    lo = as_dim(li * s + (d * (k - 1) + extra + 1 - s - left - right))  # (L - 1) s - left - right + d (K - 1) + output_padding + 1
    namespace = {"Strided": Strided, "Tensor": Tensor, "Out": Out, "StrideOf": StrideOf, "const": const, "dt": dtype}
    namespace.update({"NB": nb, "LI": li, "LO": lo, "CH": Dim("C")})

    def reference(cols: Any, bias: Any = None) -> Any:
        """The same sum in torch on any device: ``acc = +0.0``, then for each tap from the last to the first, ``acc + (that tap's terms, shifted to
        the positions they reach, +0.0 elsewhere)``, one f32 add per element and tap; then ``+ bias`` and one rounding."""
        import torch

        n_, li_, ck = cols.shape
        c_ = ck // k
        lo_ = _col2im1d_length(li_, k, s, d, pad, extra)
        taps = cols.reshape(n_, li_, c_, k) if order == "channel" else cols.reshape(n_, li_, k, c_).transpose(2, 3)  # [N, Li, C, K]
        span = (li_ - 1) * s + 1
        acc = torch.zeros((n_, c_, lo_), dtype=torch.float32, device=cols.device)
        for j in reversed(range(k)):  # descending taps: ascending input positions at every output position
            part = torch.zeros((n_, c_, max(left + lo_, j * d + span)), dtype=torch.float32, device=cols.device)
            part[:, :, j * d : j * d + span : s] = taps[:, :, :, j].transpose(1, 2).float()  # input i reaches i * s + j * d
            acc = acc + part[:, :, left : left + lo_]
        if bias is not None:
            acc = acc + bias.float()[None, :, None]
        y = acc.to(cols.dtype)
        return y.contiguous() if layout == "ncl" else y.transpose(1, 2).contiguous()

    return _generated(name, source, stub, namespace, (lo, nb), warps, reference, fp_fusion=False)


def _least(size: Any) -> int:
    """The smallest value a size of a traced tensor can take: its value; for a sum of dimensions with positive coefficients, its value at the
    smallest bucket of each (1 for one without buckets); else 1."""
    size = size.folded()
    if size.is_static:
        return int(size.value)
    if all(c > 0 and isinstance(a, Dim) for a, c in size.terms):
        least = size.subst({a: min(a.buckets) if a.buckets else 1 for a, _ in size.terms})
        return max(1, int(least.value))
    return 1


def col2im1d(
    cols: Any,
    kernel: Any,
    *,
    stride: Any = 1,
    dilation: Any = 1,
    padding: Any = 0,
    output_padding: Any = 0,
    order: str = "channel",
    bias: Any = None,
    layout: str = "ncl",
) -> Value:
    """``tilefish.col2im1d(cols, kernel, *, stride=1, dilation=1, padding=0, output_padding=0, order="channel", bias=None, layout="ncl")``: the
    overlap-add of a 1-D transposed convolution, from the products of each input position (``cols``) to the output.

    ``cols: [N, Li, C * K]`` (any strides: a GEMM's output reshaped, or a view of it, goes in as it is) holds, for input position ``i``, the
    ``C * K`` products of a ``ConvTranspose1d`` weight ``W: [C_in, C, K]``: ``cols = x.transpose(1, 2) @ W.reshape(C_in, C * K)``, tap ``j`` of
    channel ``c`` in column ``c * K + j`` (``order="channel"``, the weight as it is stored) or ``j * C + c`` (``order="tap"``, the weight
    ``.permute(0, 2, 1)``). The result is contiguous, ``[N, C, Lo]`` (``layout="ncl"``) or ``[N, Lo, C]`` (``layout="nlc"``), of the dtype of
    ``cols``, with ``Lo = (Li - 1) * stride - left - right + dilation * (K - 1) + output_padding + 1``: ``padding`` is an int or a pair
    ``(left, right)`` of positions trimmed from the full output, ``output_padding`` adds positions at the end (torch's
    ``conv_transpose1d(padding=p, output_padding=op)`` is ``padding=(p, p), output_padding=op``). ``bias`` is ``[C]`` of the dtype of ``cols``.

    Output position ``t`` of channel ``c`` adds every ``cols[n, i, col(c, j)]`` with ``t + left = i * stride + j * dilation``: in f32, from
    ``+0.0``, in ascending input position ``i`` (descending tap ``j``), one rounding per add, then ``+ bias[c]`` in f32, rounded once to the dtype
    of ``cols``. That is the order of ATen's CUDA ``col2im`` (``F.fold`` on CUDA): the bytes are ``F.fold``'s (the numerics contract is in
    :mod:`tilefish._copyops`). Supported for f32, bf16 and f16; a static channel count is compiled in.

    A step of a ``@tilefish.graph`` (a generated op, see :mod:`tilefish._copyops`).
    """
    if not isinstance(cols, Value):
        raise TilefishTypeError(f"col2im1d: takes a traced tensor [N, Li, C * K], got {type(cols).__name__}")
    if cols.rank != 3:
        raise TilefishTypeError(f"col2im1d: {cols.ref} ({cols.type}) has rank {cols.rank}; it takes [N, Li, C * K]")
    dt = cols.dtype
    if not isinstance(dt, DType) or dt.name not in _SUMMED:
        raise TilefishTypeError(f"col2im1d: {cols.ref} has dtype {dt}; it adds in f32 and takes f32, bf16 or f16")
    k, s, d, pad, extra, order, layout = _col2im1d_args(kernel, stride, dilation, padding, output_padding, order, layout)
    width = cols.shape[2].folded()
    c_expr = (DimExpr.const(width.value // k) if width.value % k == 0 else None) if width.is_static else width._div_exact_int(k)
    if c_expr is None:
        raise TilefishTypeError(f"col2im1d: axis 2 of {cols.type} is {width}, which is not C * {k} for {k} taps")
    channels = c_expr.value if c_expr.is_static else None
    if bias is not None:
        if not isinstance(bias, Value):
            raise TilefishTypeError(f"col2im1d: bias must be a traced tensor [C] or None, got {type(bias).__name__}")
        if bias.rank != 1 or bias.shape[0].folded() != c_expr or bias.dtype != dt:
            raise TilefishTypeError(f"col2im1d: bias {bias.ref} ({bias.type}) must be [C] = [{c_expr}] of the dtype of cols ({dt})")
    least = _least(cols.shape[1])
    lo = _col2im1d_length(least, k, s, d, pad, extra)
    if lo < 1:
        some = "is" if cols.shape[1].folded().is_static else "can be"
        raise TilefishTypeError(
            f"col2im1d: axis 1 of {cols.type} {some} {least}, which gives an output of {lo} positions ({k} taps at stride {s}, dilation {d}, "
            f"padding {pad}, output_padding {extra}: Lo = (Li - 1) * stride - left - right + dilation * (K - 1) + output_padding + 1)"
        )
    key = ("col2im1d", dt.name, k, s, d, pad, extra, order, layout, bias is not None, channels)
    op = _cached(key, lambda: _col2im1d_op(dt, k, s, d, pad, extra, order, layout, bias is not None, channels))
    return op(cols, bias) if bias is not None else op(cols)


def contiguous_op(rank: int, dtype: DType, last: "int | None" = None) -> ExternOp:
    """The generated op behind :func:`contiguous` for views of ``rank`` and ``dtype`` (to run it on a torch view with ``tilefish.run``); ``last``
    is a static size of the last axis (the tile is fitted to it and it is compiled in), ``None`` a runtime one."""
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 4:
        raise TilefishTypeError(f"contiguous_op: rank must be 1 to 4, got {rank!r}")
    if not isinstance(dtype, DType) or dtype.is_bool or dtype.name not in _TL:
        raise TilefishTypeError(f"contiguous_op: {dtype!r} is not a plain storage type")
    if last is not None and (isinstance(last, bool) or not isinstance(last, int) or last < 1):
        raise TilefishTypeError(f"contiguous_op: last must be a positive int or None, got {last!r}")
    return _cached(("contiguous", rank, dtype.name, last), lambda: _contiguous_op(rank, dtype, last))


def im2col_op(dtype: DType, kernel: Any = (3, 3), stride: Any = (1, 1), padding: Any = (0, 0), channels: "int | None" = None) -> ExternOp:
    """The generated op behind :func:`im2col` (``channels``: a static channel count, written into the source; ``None``: a runtime one)."""
    if not isinstance(dtype, DType) or dtype.is_bool or dtype.name not in _TL:
        raise TilefishTypeError(f"im2col_op: {dtype!r} is not a plain storage type")
    if channels is not None and (isinstance(channels, bool) or not isinstance(channels, int) or channels < 1):
        raise TilefishTypeError(f"im2col_op: channels must be a positive int or None, got {channels!r}")
    k, s, p = _pair("kernel", kernel, 1), _pair("stride", stride, 1), _pair("padding", padding, 0)
    return _cached(("im2col", dtype.name, k, s, p, channels), lambda: _im2col_op(dtype, k, s, p, channels))


def im2col1d_op(
    dtype: DType, kernel: Any, *, stride: Any = 1, dilation: Any = 1, padding: Any = 0, order: str = "channel", channels: "int | None" = None
) -> ExternOp:
    """The generated op behind :func:`im2col1d` (``channels``: a static channel count, written into the source; ``None``: a runtime one)."""
    if not isinstance(dtype, DType) or dtype.is_bool or dtype.name not in _TL:
        raise TilefishTypeError(f"im2col1d_op: {dtype!r} is not a plain storage type")
    if channels is not None and (isinstance(channels, bool) or not isinstance(channels, int) or channels < 1):
        raise TilefishTypeError(f"im2col1d_op: channels must be a positive int or None, got {channels!r}")
    k, s, d, pad, order = _im2col1d_args(kernel, stride, dilation, padding, order)
    return _cached(("im2col1d", dtype.name, k, s, d, pad, order, channels), lambda: _im2col1d_op(dtype, k, s, d, pad, order, channels))


def col2im1d_op(
    dtype: DType,
    kernel: Any,
    *,
    stride: Any = 1,
    dilation: Any = 1,
    padding: Any = 0,
    output_padding: Any = 0,
    order: str = "channel",
    layout: str = "ncl",
    bias: bool = False,
    channels: "int | None" = None,
) -> ExternOp:
    """The generated op behind :func:`col2im1d` (``bias``: whether it takes one; ``channels``: a static channel count, written into the source;
    ``None``: a runtime one). It takes ``cols`` (and ``bias``) and gives ``y``; ``Lo`` is computed from ``Li`` at each call, and a call whose
    ``Lo`` is less than 1 is refused at launch (its grid would be empty)."""
    if not isinstance(dtype, DType) or dtype.name not in _SUMMED:
        raise TilefishTypeError(f"col2im1d_op: {dtype!r} is not f32, bf16 or f16 (col2im1d adds in f32)")
    if not isinstance(bias, bool):
        raise TilefishTypeError(f"col2im1d_op: bias must be True or False, got {bias!r}")
    if channels is not None and (isinstance(channels, bool) or not isinstance(channels, int) or channels < 1):
        raise TilefishTypeError(f"col2im1d_op: channels must be a positive int or None, got {channels!r}")
    k, s, d, pad, extra, order, layout = _col2im1d_args(kernel, stride, dilation, padding, output_padding, order, layout)
    key = ("col2im1d", dtype.name, k, s, d, pad, extra, order, layout, bias, channels)
    return _cached(key, lambda: _col2im1d_op(dtype, k, s, d, pad, extra, order, layout, bias, channels))


__all__ = ["col2im1d", "col2im1d_op", "contiguous", "contiguous_op", "generated_source", "im2col", "im2col1d", "im2col1d_op", "im2col_op"]
