"""The ``reference`` mode: the traced kernel IR interpreted with torch.

It is an oracle for the Triton binaries, written to follow the numerics contract of :mod:`tilefish.lower`
node by node:

* a cast happens exactly where the IR has one (``.to(dtype)``, nowhere else); a cast to ``e4m3`` or ``e5m2``
  is ``satfinite`` like the binaries (saturating at 448 and 57344, where torch gives NaN or infinity), and the
  half types round to nearest even like torch. f32 to bf16 is computed on the bits, in the NaN mode the cast declares: a NaN is
  ``0x7FFF`` for ``nan="canonical"`` (the default, what the binary's ``cvt.rn.bf16.f32`` gives) and keeps its sign and payload for
  ``nan="preserve"``; torch's own cast is neither everywhere (``0x7FFF`` on CUDA, ``0x7FC0`` or ``0xFFFF`` on the CPU), so it is not
  used for that one. A float to integer cast is computed in its declared rounding and saturation modes (note 18 of
  :mod:`tilefish.lower`), exactly, in float64 and int64: not torch's cast, which neither rounds nor saturates as declared;
* every other operation runs in the dtype the IR gave its result, and a python scalar or ``const`` is a 0-d
  tensor of that dtype (a bf16 tensor combined with a float scalar rounds the scalar to bf16 first, as
  Triton does);
* ``x / c`` with a scalar or ``const`` divisor is ``x * f32(1 / c)``, which is what the binary computes
  (``div.full.f32`` of a constant divisor is a multiply by the correctly rounded f32 reciprocal), and so
  is ``mean`` (``sum / D``);
* reductions are torch's, but an ordered sum (``order=``, note 17), which adds in its order with one rounding per add (and an ``fma`` per step
  with ``fma=True``): exact; ``argmax`` and ``argmin`` follow the contract (the first NaN, else the first extreme value), written out in torch;
* ``rcp`` is ``torch.reciprocal``, ``neg`` CUDA's negation (a NaN canonical) and ``fma`` one rounding (an f64 product and sum rounded to odd, then to
  f32): exact (note 16); ``sin_approx`` is torch's sine, approximately;
* comparisons, masks, ``where`` and ``iota`` are torch's, with a scalar rounded to the IR dtype as above: exact, like the binary;
* ``gather_rows`` and ``scatter_rows`` are torch indexing with the out-of-range policy of the call: raw copies, exact (where a scatter's
  indices in range are distinct; with duplicates neither the binary nor the reference says which row lands);
* a static slice ``x[k]`` of a parameter is torch's ``select``, the row the binary reads at its offset.

What stays different, and what a tolerance has to cover, is everything the binary does approximately:
``/`` between tensors (``div.full.f32``), ``exp`` (``ex2.approx``), ``rsqrt``, ``erf`` (libdevice), the
order of a reduction, and FMA contraction (the binaries fuse ``a * b + c`` into one rounding, the
reference rounds twice). The gpu tests measure the resulting error per kernel.

In IEEE math (``@tilefish.kernel(math="ieee")``, or ``math="ieee"`` at an op: note 13 of :mod:`tilefish.lower`) the reference is torch's
own CUDA op: ``/`` is ``torch.div`` (a scalar divisor too, as a 0-d tensor: no reciprocal), ``exp``, ``sqrt``, ``rsqrt``, ``sin`` and ``tanh`` torch's,
and ``silu`` ``torch.nn.functional.silu``; on a GPU these are the binary's bytes.

An ``InOut`` or ``Out`` parameter is written once, with the final version stored into it, after the whole
body has been evaluated, like the lowered kernel does: reads never see a store.
"""

from __future__ import annotations

from typing import Any

from ._launch import TORCH_DTYPE_NAMES, eval_dim
from .decorators import KernelOp
from .errors import TilefishRuntimeError
from .ir import Scalar
from .trace import Value

_E4M3_MAX = 448.0
_E5M2_MAX = 57344.0
_COMPARE = {"eq": "eq", "ne": "ne", "lt": "lt", "le": "le", "gt": "gt", "ge": "ge"}  # IR op -> torch function


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:
        raise TilefishRuntimeError("running a kernel needs torch: pip install 'tilefish[gpu]'") from e
    return torch


def torch_dtype(torch: Any, dt: Any) -> Any:
    if dt.name == "bool":  # a mask: a value inside the body only
        return torch.bool
    return getattr(torch, TORCH_DTYPE_NAMES[dt.name].split(".")[1])


def reciprocal(torch: Any, value: float, dtype: Any, device: Any) -> Any:
    """``1 / value`` correctly rounded in ``dtype``, as a 0-d tensor (computed on the CPU, where division is IEEE)."""
    one = torch.ones((), dtype=dtype)
    return (one / torch.tensor(value, dtype=dtype)).to(device)


def f32_to_bf16(torch: Any, x: Any, nan: str) -> Any:
    """The f32 to bf16 cast of the binary, bit for bit: round to nearest even on the bit pattern, and a NaN either canonical (``0x7FFF``,
    what ``cvt.rn.bf16.f32`` gives) or with its sign and top payload bits kept, quiet (``nan="preserve"``).

    Written on the bits rather than as ``x.to(torch.bfloat16)`` because torch's own cast of a NaN depends on where it runs: ``0x7FFF`` on
    CUDA, ``0x7FC0`` or ``0xFFFF`` on the CPU (scalar or vectorized path). On every other input they are all the same rounding.
    """
    b = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
    kept = (b >> 16) | 0x40
    canonical = torch.full_like(b, 0x7FFF)
    word = torch.where(torch.isnan(x), kept if nan == "preserve" else canonical, rounded) & 0xFFFF
    return torch.where(word >= 0x8000, word - 0x10000, word).to(torch.int16).view(torch.bfloat16)


_ROUND = {"rn": "round", "rz": "trunc", "rd": "floor", "ru": "ceil"}  # torch's rounding of a float64 to an integral value (round: ties to even)


def float_to_int(torch: Any, x: Any, dst: Any, rounding: str, saturate: bool) -> Any:
    """A float to integer cast in the declared mode (note 18 of :mod:`tilefish.lower`), computed exactly rather than with torch's cast.

    The value is widened to float64 (exact for every float type) and rounded to an integral value there (``rn`` to nearest even, ``rz``
    toward zero, ``rd`` down, ``ru`` up); a NaN is 0 and the value is clamped to the ``i64`` range (an infinity to its bound), which is
    exact in float64. ``saturate=True`` then clamps it to the range of ``dst``; ``saturate=False`` keeps its low ``bits`` bits (two's
    complement). torch's own casts are none of these everywhere (``x.to(torch.int64)`` gives ``-2**63`` for a NaN on CUDA).
    """
    v = getattr(torch, _ROUND[rounding])(x.to(torch.float32).to(torch.float64))
    nan = torch.isnan(v)
    over = v >= 2.0**63
    under = v < -(2.0**63)
    i = torch.where(nan | over | under, torch.zeros_like(v), v).to(torch.int64)  # integral and in range: exact
    i = torch.where(over, torch.full_like(i, 2**63 - 1), torch.where(under, torch.full_like(i, -(2**63)), i))
    target = torch_dtype(torch, dst)
    if dst.name == "i64":  # the two modes are one
        return i
    if saturate:
        lo, hi = dst.int_range()
        return i.clamp(lo, hi).to(target)
    bits = dst.bits
    w = i & ((1 << bits) - 1)
    if dst.signed:
        w = torch.where(w >= 1 << (bits - 1), w - (1 << bits), w)
    return w.to(target)


def cast(torch: Any, x: Any, src: Any, dst: Any, nan: "str | None" = None, rounding: "str | None" = None, saturate: bool = True) -> Any:
    """``x.to(dst)``, with ``satfinite`` for the fp8 types and the binary's own f32 to bf16 rounding.

    ``src`` and ``dst`` are tilefish dtypes. Float to ``e4m3`` / ``e5m2``: round to nearest even, saturating at
    the largest finite value (448, 57344), infinities included, instead of overflowing to NaN or infinity as
    torch does; NaN stays NaN. f32 to bf16 is :func:`f32_to_bf16` in the NaN mode ``nan`` of the cast (``canonical`` by
    default). A float to integer cast is :func:`float_to_int` in its declared ``rounding`` and ``saturate`` modes. Every other cast is
    torch's.
    """
    target = torch_dtype(torch, dst)
    if src.is_float and dst.is_int:
        if rounding is None:
            raise TilefishRuntimeError(f"reference mode: a cast of {src} to {dst} without a rounding mode (a float to integer cast says it)")
        return float_to_int(torch, x, dst, rounding, saturate)
    if src.name == "f32" and dst.name == "bf16":
        return f32_to_bf16(torch, x, nan or "canonical")
    if src.is_float and dst.name == "e4m3":
        return x.to(torch.float32).clamp(-_E4M3_MAX, _E4M3_MAX).to(target)
    if src.is_float and dst.name == "e5m2":
        return x.to(torch.float32).clamp(-_E5M2_MAX, _E5M2_MAX).to(target)
    return x.to(target)


def canonical_nan(torch: Any, x: Any) -> Any:
    """``x`` with every NaN made the canonical NaN ``0x7FFFFFFF`` (f32), what the GPU's f32 arithmetic gives for a NaN result."""
    nan = torch.tensor(0x7FFFFFFF, dtype=torch.int32, device=x.device).view(torch.float32)
    return torch.where(torch.isnan(x), nan, x)


def fma_f32(torch: Any, a: Any, b: Any, c: Any) -> Any:
    """``fma.rn.f32``: ``a * b + c`` with one rounding, computed in f64 and rounded to odd, then to f32 (note 16 of :mod:`tilefish.lower`).

    The product of two f32 is exact in f64 (48 bits, and neither overflows nor underflows there); the sum is rounded to f64 and the error of
    that rounding found exactly (TwoSum). Rounding the f64 sum to odd (towards the error when it is inexact and its last bit is even) and
    then to f32 gives the correctly rounded f32 of the exact value, because f64 has more than two bits beyond f32's; the f32 conversion
    rounds a subnormal and an overflow as the instruction does. A NaN result is the canonical NaN, as the GPU's.
    """
    a64, b64, c64 = (t.to(torch.float64) for t in torch.broadcast_tensors(a, b, c))
    p = a64 * b64
    s = p + c64
    bb = s - p
    err = (p - (s - bb)) + (c64 - bb)
    inexact = torch.isfinite(s) & torch.isfinite(err) & (err != 0)
    even = (s.view(torch.int64) & 1) == 0
    toward = torch.nextafter(s, torch.where(err > 0, torch.full_like(s, float("inf")), torch.full_like(s, float("-inf"))))
    odd = torch.where(inexact & even, toward, s)
    return canonical_nan(torch, odd.to(torch.float32))


def neg(torch: Any, x: Any) -> Any:
    """CUDA's ``-x`` (note 16 of :mod:`tilefish.lower`): a float's sign flipped, zeros and infinities included, a NaN the canonical NaN of its
    dtype; an integer negated in two's complement."""
    if not x.is_floating_point():
        return torch.neg(x)
    r = torch.neg(x)
    if x.dtype == torch.float32:
        return canonical_nan(torch, r)
    word = torch.tensor(0x7FFF, dtype=torch.int16, device=x.device).view(x.dtype)  # the canonical NaN of bf16 and f16 alike
    return torch.where(torch.isnan(x), word, r)


def ordered_sum(torch: Any, terms: "tuple[Any, ...]", order: Any) -> Any:
    """The sum of the last axis in ``order`` (note 17 of :mod:`tilefish.lower`), every step one rounding: ``terms`` is ``(x,)`` (each step adds
    ``x``) or ``(a, b)`` (each step is ``fma(a, b, acc)``), broadcast to one ``[..., D]`` shape. Lane ``j`` of ``order.lanes`` starts from ``+0.0``
    and takes the columns ``c * lanes * vector + j * vector + v`` below ``D`` in the order ``(c, v)``; then ``p[j] += p[j + o]`` for each offset
    ``o``; the result is lane 0's. A NaN is the canonical one, as the GPU's adds give."""
    terms = torch.broadcast_tensors(*terms)
    d = terms[0].shape[-1]
    lanes, vector, width = order.lanes, order.vector, order.lanes * order.vector
    lane = torch.arange(lanes, device=terms[0].device)
    p = torch.zeros((*terms[0].shape[:-1], lanes), dtype=torch.float32, device=terms[0].device)
    for c in range(-(-d // width)):
        for v in range(vector):
            col = c * width + lane * vector + v
            valid = col < d
            if not bool(valid.any()):
                continue
            k = col.clamp(max=d - 1)
            got = [t[..., k] for t in terms]
            step = canonical_nan(torch, p + got[0]) if len(got) == 1 else fma_f32(torch, got[0], got[1], p)
            p = torch.where(valid, step, p)
    for o in order.offsets:  # p[j] += p[j + o] for every lane j at once (the lanes past the end take nothing: they are not read again)
        shifted = torch.cat([p[..., o:], torch.zeros_like(p[..., :o])], dim=-1)
        p = canonical_nan(torch, p + shifted)
    return p[..., 0]


def arg_reduce(torch: Any, x: Any, axis: int, keepdim: bool, largest: bool, dtype: Any) -> Any:
    """``argmax`` (``largest``) or ``argmin`` along ``axis`` by the contract (note 8 of :mod:`tilefish.lower`), written out in torch rather
    than trusted to ``torch.argmax``: the first NaN wins when there is one, else the first position of the extreme value (``-0.0 == +0.0``)."""
    n = x.shape[axis]
    position = torch.arange(n, device=x.device).reshape([n if i == axis else 1 for i in range(x.dim())])
    if x.is_floating_point():
        v = x.float()  # exact for bf16 and f16
        nan = torch.isnan(v)
        fill = float("-inf") if largest else float("inf")
        clean = torch.where(nan, torch.full_like(v, fill), v)
        best = clean.amax(dim=axis, keepdim=True) if largest else clean.amin(dim=axis, keepdim=True)
        hit = torch.where(nan.any(dim=axis, keepdim=True), nan, v == best)
    else:
        best = x.amax(dim=axis, keepdim=True) if largest else x.amin(dim=axis, keepdim=True)
        hit = x == best
    first = torch.where(hit, position, torch.full_like(position, n)).amin(dim=axis, keepdim=keepdim)
    return first.to(dtype)


def interpret(kernel: KernelOp, tensors: "dict[str, Any]", dims: "dict[Any, int] | None" = None) -> "list[Any]":
    """Evaluate the kernel body on ``tensors`` (name -> tensor, every parameter; ``Out`` ones are filled in place).

    ``dims`` are the dimensions the call bound (``iota`` counts up to one of them). Returns the returned values as fresh tensors, in
    order. ``InOut`` and ``Out`` tensors are written in place.
    """
    torch = _torch()
    graph = kernel.graph
    math = getattr(kernel, "math", "fast")  # an op that says its own math (math= at the call) keeps it
    device = next(iter(tensors.values())).device
    env: dict[Value, Any] = {}
    for p in graph.inputs:
        if p.access != "out":
            env[p] = tensors[p.name]

    def operand(x: "Value | Scalar") -> Any:
        if isinstance(x, Value):
            return env[x]
        return torch.tensor(x.value, dtype=torch_dtype(torch, x.dtype), device=device)  # the scalar, rounded to the IR dtype

    producer = {o: n for n in graph.nodes for o in n.outputs}  # an fma sum reads the operands of its product
    final: dict[str, Any] = {}
    with torch.no_grad():
        for n in graph.nodes:
            op = n.op
            if op == "store":
                value = env[n.inputs[1]]  # type: ignore[index]
                if n.outputs:  # an InOut parameter: reading it from here on reads the stored value
                    env[n.outputs[0]] = value
                if n.inputs[0] is not n.inputs[1]:  # `x[...] = x` writes nothing
                    final[n.attrs["target"]] = value
                continue
            if op == "scatter_rows":  # rows copied to the index, the others kept; an index out of range is skipped
                dst, idx, src = (env[i] for i in n.inputs)  # type: ignore[index]
                idx = idx.long()
                ok = (idx >= 0) & (idx < dst.shape[0])
                value = dst.clone()
                value[idx[ok]] = src[ok]
                env[n.outputs[0]] = value
                final[n.attrs["target"]] = value
                continue
            if "kind" in n.attrs:
                raise TilefishRuntimeError(
                    f"reference mode does not interpret the call to the {n.attrs['kind']} op '{op}' inside kernel "
                    f"'{kernel.name}'; only the built-in ops are interpreted"
                )
            a = [operand(i) for i in n.inputs]
            (out,) = n.outputs
            if op == "load":  # the one element of a DevScalar, as a 0-d tensor (it broadcasts like the kernel's scalar does)
                r = a[0].reshape(())
            elif op == "select":  # x[k]: a static slice of a parameter (note 19 of tilefish.lower), a view of it
                r = a[0].select(n.attrs["axis"], n.attrs["index"])
            elif op in _COMPARE:  # IEEE comparisons (a NaN is unordered: only != is true), the scalar rounded to the IR dtype
                r = getattr(torch, _COMPARE[op])(a[0], a[1])
            elif op in ("and", "or", "xor"):
                r = getattr(torch, f"logical_{op}")(a[0], a[1])
            elif op == "not":
                r = torch.logical_not(a[0])
            elif op == "where":  # a select: the chosen operand's bits as they are
                r = torch.where(a[0], a[1], a[2])
            elif op == "iota":
                size = eval_dim(n.attrs["dim"], dims or {})
                if size is None:
                    raise TilefishRuntimeError(f"reference mode: the size {n.attrs['dim']} that iota counts along is not determined by the arguments")
                shape = [size] + [1] * (out.type.rank - 1)
                r = torch.arange(size, dtype=torch_dtype(torch, out.type.dtype), device=device).reshape(shape)
            elif op == "gather_rows":  # a raw copy of rows; out of range: a row of zeros (all bits clear), or the nearest row
                table, idx = a[0], a[1].long()
                rows = table.shape[0]
                if n.attrs["oob"] == "clamp":
                    r = table[idx.clamp(0, rows - 1)]
                else:
                    ok = (idx >= 0) & (idx < rows)
                    r = table[torch.where(ok, idx, torch.zeros_like(idx))]
                    r = torch.where(ok.reshape(*ok.shape, *([1] * (table.dim() - 1))), r, torch.zeros((), dtype=r.dtype, device=r.device))
            elif op in ("argmax", "argmin"):
                (ax,) = n.attrs["axes"]
                r = arg_reduce(torch, a[0], ax, n.attrs["keepdims"], op == "argmax", torch_dtype(torch, n.attrs["dtype"]))
            elif op == "floordiv":  # Python's semantics: toward minus infinity
                r = torch.div(a[0], a[1], rounding_mode="floor")
            elif op == "mod":  # Python's semantics: in [0, c) for a positive c
                r = torch.remainder(a[0], a[1])
            elif op == "add":
                r = a[0] + a[1]
            elif op == "sub":
                r = a[0] - a[1]
            elif op == "mul":
                r = a[0] * a[1]
            elif op == "div" and n.attrs.get("math", math) == "ieee":  # IEEE division, a scalar divisor too (a 0-d tensor: torch divides)
                r = torch.div(a[0], a[1])
            elif op == "div":
                divisor = n.inputs[1]
                if isinstance(divisor, Scalar) and divisor.dtype.name == "f32":  # a constant divisor: a multiply by 1/c
                    r = a[0] * reciprocal(torch, float(divisor.value), torch.float32, device)
                else:
                    r = a[0] / a[1]
            elif op == "rsqrt":
                r = torch.rsqrt(a[0])
            elif op == "exp":
                r = torch.exp(a[0])
            elif op == "sqrt":
                r = torch.sqrt(a[0])
            elif op == "cos":
                r = torch.cos(a[0])
            elif op == "erf":  # CUDA's erff in IEEE math (torch's on CUDA 12.8 to 13.1); libdevice's, approximately, in the default math
                r = torch.erf(a[0])
            elif op == "sin_approx":  # the hardware's approximate sine (a subnormal argument is a zero of its sign): torch's sine, approximately
                r = torch.where(a[0].abs() < 2.0**-126, a[0] * 0.0, torch.sin(a[0]))
            elif op == "rcp":  # rcp.rn.f32: correctly rounded
                r = torch.reciprocal(a[0])
            elif op == "neg":
                r = neg(torch, a[0])
            elif op == "fma":
                r = fma_f32(torch, a[0], a[1], a[2])
            elif op in ("sum", "mean") and "order" in n.attrs:  # an ordered sum (note 17): exact, the binary's order
                (ax,) = n.attrs["axes"]
                if n.attrs.get("fma"):  # each step fma(a, b, acc) of the product's operands
                    product = producer[n.inputs[0]]  # type: ignore[index]
                    terms = tuple(operand(i) for i in product.inputs)
                else:
                    terms = (a[0],)
                shape = torch.broadcast_shapes(*(t.shape for t in terms))
                terms = tuple(t.expand(shape).movedim(ax, -1) for t in terms)
                r = ordered_sum(torch, terms, n.attrs["order"])
                if op == "mean":  # times 1 / D correctly rounded (rcp.rn), one rounding (mul.rn); a NaN the canonical one, as the GPU's multiply gives
                    r = canonical_nan(torch, r * reciprocal(torch, float(shape[ax]), torch.float32, device))
                if n.attrs["keepdims"]:
                    r = r.unsqueeze(ax)
            elif op == "sin" and n.attrs.get("math", math) == "ieee":  # without flush to zero: torch's
                r = torch.sin(a[0])
            elif op == "sin":  # libdevice built with FTZ: a subnormal argument is a zero of its sign, and sin(±0) = ±0
                r = torch.where(a[0].abs() < 2.0**-126, a[0] * 0.0, torch.sin(a[0]))
            elif op == "tanh" and n.attrs.get("math", math) == "ieee":  # without flush to zero: torch's
                r = torch.tanh(a[0])
            elif op == "tanh":  # libdevice built with FTZ: a subnormal argument is a zero of its sign (note 14)
                r = torch.where(a[0].abs() < 2.0**-126, a[0] * 0.0, torch.tanh(a[0]))
            elif op == "silu" and n.attrs.get("math", math) == "ieee":  # torch's: x / (1 + expf(-x))
                r = torch.nn.functional.silu(a[0])
            elif op == "silu":  # x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
                r = a[0] * (1.0 / (1.0 + torch.exp(-a[0])))
            elif op == "gelu":  # exact erf form, evaluated left to right
                r = 0.5 * a[0] * (1.0 + torch.erf(a[0] * 0.7071067811865476))
            elif op == "cast":
                src = n.inputs[0].type.dtype  # type: ignore[union-attr]
                r = cast(torch, a[0], src, n.attrs["to"], n.attrs.get("nan"), n.attrs.get("round"), n.attrs.get("saturate", True))
            elif op in ("sum", "mean", "max"):
                axes = tuple(n.attrs["axes"])
                keep = bool(n.attrs["keepdims"])
                if op == "max":
                    r = torch.amax(a[0], dim=axes, keepdim=keep)
                else:
                    r = torch.sum(a[0], dim=axes, keepdim=keep)
                    if op == "mean":  # the contract: sum / D, with D a constant
                        count = 1
                        for ax in axes:
                            count *= a[0].shape[ax]
                        if r.dtype == torch.float32:
                            r = r * reciprocal(torch, float(count), torch.float32, device)
                        else:
                            r = r / count
            else:
                raise TilefishRuntimeError(f"reference mode does not interpret the op '{op}' (kernel '{kernel.name}')")
            env[out] = r
        results = []
        for v in graph.outputs:
            r = env[v]
            results.append(r.clone() if (v.is_param or any(r is t for t in tensors.values())) else r)
        for name, value in final.items():
            tensors[name].copy_(value)
    return results
