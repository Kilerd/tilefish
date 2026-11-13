"""Copies that tilefish writes the Triton for: ``contiguous`` (a view, materialised) and ``im2col`` (the patches of a convolution).

Neither is a ``@tilefish.kernel`` body: a lowered kernel takes contiguous tensors, and both read a strided view. Each is a *generated op*: an
``extern`` operator whose Triton source tilefish generates from a fixed template, whose input is ``Strided[...]`` (so a view goes to it as it is,
with its strides as scalar arguments, through the machinery every strided ``extern`` uses: compile specs, launches, manifests, plans, the Rust
host), and which has a torch reference. One op exists per rank and dtype (``contiguous``) or per dtype, window and channel count (``im2col``),
made the first time it is asked for and kept; called in a ``@tilefish.graph``, it is a step of the plan.

Both are raw copies: values are loaded and stored with no arithmetic, so a NaN keeps its payload and a zero its sign, and any tiling gives the
same bytes. Offsets are ``i64`` (``index_bits=64``): the strides are ``i64`` scalars, and no size limit applies.

``contiguous(v)`` copies the view ``v`` (rank 1 to 4) into a new contiguous tensor of its shape, in its logical (row-major) order. A program
copies a ``32 x 32`` tile of the last two axes for one index of the third axis from the end; a rank-4 view loops over its first axis inside the
program.

``im2col(x, kernel=(KH, KW), stride=(SH, SW), padding=(PH, PW))`` takes ``x: [N, C, H, W]`` (any strides) and gives ``[N, Ho, Wo, C * KH * KW]``
with ``Ho = (H + 2 PH - KH) // SH + 1`` (and ``Wo`` likewise): ``out[n, oh, ow, c * KH * KW + kh * KW + kw] = x[n, c, oh * SH - PH + kh, ow * SW - PW +
kw]`` where that position is inside ``x``, and ``+0.0`` (all bits clear: the load is masked) where it is in the padding. The columns are
channel-major, the order of a ``Conv2d`` weight flattened (``F.unfold``'s), so the weight is the GEMM's operand as it is. One program per output
pixel (grid ``(Wo, Ho, N)``), a loop over the columns in blocks (one block of ``next_pow2(C * KH * KW)`` up to 1024 when ``C`` is static).
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any

from .decorators import ExternOp
from .dims import Dim, as_dim, cdiv
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


def _generated(name: str, source: str, stub_source: str, namespace: "dict[str, Any]", grid: Any, num_warps: int, reference: Any) -> ExternOp:
    """An extern op over the generated Triton ``source`` (module ``tilefish_generated.<name>``, function ``tf_<name>``), declared by the stub in
    ``stub_source`` (evaluated in ``namespace``; its parameters are the Triton function's, in order), with the source attached so that every
    mode runs it."""
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


__all__ = ["contiguous", "contiguous_op", "generated_source", "im2col", "im2col_op"]
