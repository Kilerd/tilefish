"""Traced built-in operations: elementwise arithmetic, unary math, casts, reductions, split, store.

Every function here takes traced :class:`~tilefish.trace.Value` operands,
type-checks them, appends a node to the trace and returns new ``Value``s.
There is no implicit promotion between tensor dtypes; python scalars and
``const`` parameters adopt the dtype of the tensor they are combined with.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .dims import DimExpr, as_dim, cdiv, describe_pair
from .dtypes import BlockScaled, DType
from .errors import TilefishError, TilefishTypeError
from .ir import Scalar
from .trace import Const, TraceContext, Value
from .types import TensorType

_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


# -- operand checking ---------------------------------------------------------


def _require_value(op: str, label: str, x: Any) -> Value:
    """Check that ``x`` is a readable traced tensor and return its newest version."""
    if not isinstance(x, Value):
        raise TilefishTypeError(f"{op}: {label} must be a traced tensor value, got {type(x).__name__} ({x!r})")
    if x.access == "out":
        raise TilefishTypeError(f"{op}: {label} reads Out parameter '{x.name}', which is write-only")
    return x.live


def _require_computable(op: str, label: str, x: Value, *, floating: bool = False) -> DType:
    dt = x.dtype
    if isinstance(dt, BlockScaled):
        raise TilefishTypeError(
            f"{op}: {label} has block-scaled dtype {dt!r}; it can only be passed to operators that "
            "declare it (library/extern), not combined directly"
        )
    if not dt.computable:
        raise TilefishTypeError(
            f"{op}: {label} has storage dtype {dt} which has no arithmetic; cast it first with .to(f32) or .to(bf16)"
        )
    if floating and not dt.is_float:
        raise TilefishTypeError(f"{op}: {label} must be a floating-point tensor, got {x.type}")
    return dt


def _scalar_operand(op: str, label: str, x: Any, dtype: DType) -> Scalar:
    name = None
    if isinstance(x, Const):
        name, value = x.name, x.value
        if value is None:
            raise TilefishTypeError(f"{op}: {label} const parameter '{x.name}' has no value at trace time; give it a default")
        if x.dtype is not None and x.dtype != dtype:
            raise TilefishTypeError(
                f"{op}: {label} const '{x.name}' is const[{x.dtype}] but the other operand has dtype {dtype}; "
                f"there is no implicit conversion: cast the tensor with .to({x.dtype}) or declare the const as "
                f"const[{dtype}] (an untyped const adopts the tensor dtype)"
            )
    elif isinstance(x, DimExpr) and x.is_static:
        value = x.value
    else:
        value = x
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TilefishTypeError(
            f"{op}: {label} must be a traced tensor, a python number or a const parameter, got {type(x).__name__} ({x!r})"
        )
    if dtype.is_float:
        value = float(value)
    else:
        if not isinstance(value, int):
            raise TilefishTypeError(
                f"{op}: {label} {value!r} is a float scalar but the other operand has integer dtype {dtype}; "
                "cast the tensor to a float dtype first"
            )
        lo, hi = dtype.int_range()
        if not lo <= value <= hi:
            raise TilefishTypeError(f"{op}: {label} {value} does not fit in {dtype}")
    return Scalar(value, dtype, name)


def broadcast_shapes(op: str, a: TensorType, b: TensorType) -> tuple[DimExpr, ...]:
    """numpy broadcasting over trailing dims; only a static 1 broadcasts."""
    out: list[DimExpr] = []
    for i in range(1, (a.rank if a.rank > b.rank else b.rank) + 1):
        da = a.shape[-i] if i <= a.rank else None
        db = b.shape[-i] if i <= b.rank else None
        if da is None:
            out.append(db)  # type: ignore[arg-type]
        elif db is None or da == db:
            out.append(da)
        elif da.is_static and da.value == 1:
            out.append(db)
        elif db.is_static and db.value == 1:
            out.append(da)
        else:
            ta, tb = describe_pair(da, db)
            raise TilefishTypeError(
                f"{op}: cannot broadcast {a} with {b}: dimension {ta} vs {tb} at axis -{i} "
                "(shapes align on trailing dimensions; only a size-1 dimension broadcasts)"
            )
    out.reverse()
    return tuple(out)


# -- elementwise arithmetic ---------------------------------------------------


def binary(op: str, left: Any, right: Any) -> Value:
    """``left <op> right`` for ``op`` in ``add``, ``sub``, ``mul``, ``div``."""
    named = f"{op} ({_SYMBOLS[op]})"
    sides = [("left operand", left), ("right operand", right)]
    ctx: TraceContext | None = None
    resolved: list[Any] = []
    for lab, x in sides:
        if isinstance(x, Value):
            if ctx is None:
                ctx = x.ctx
            elif x.ctx is not ctx:
                raise TilefishTypeError(f"{named}: {lab} belongs to a different trace")
            x = _require_value(named, lab, x)
        resolved.append(x)
    if ctx is None:
        raise TilefishTypeError(f"{named}: at least one operand must be a traced tensor value")
    tensors = [(lab, x) for (lab, _), x in zip(sides, resolved) if isinstance(x, Value)]
    dtypes = [_require_computable(named, lab, x, floating=(op == "div")) for lab, x in tensors]
    if len(tensors) == 2 and tensors[0][1].dtype != tensors[1][1].dtype:
        raise TilefishTypeError(
            f"{named}: no implicit promotion between tensor dtypes: left operand is {tensors[0][1].type}, "
            f"right operand is {tensors[1][1].type}; cast one side explicitly with .to(dtype)"
        )
    dtype = dtypes[0]
    operands = [
        x if isinstance(x, Value) else _scalar_operand(named, lab, x, dtype)
        for (lab, _), x in zip(sides, resolved)
    ]
    shapes = [o.type for o in operands if isinstance(o, Value)]
    shape = shapes[0].shape if len(shapes) == 1 else broadcast_shapes(named, shapes[0], shapes[1])
    (out,) = ctx.emit(op, operands, {}, [TensorType(shape, dtype)])
    return out


# -- unary math -----------------------------------------------------------------


def _unary(op: str, x: Any) -> Value:
    x = _require_value(op, "argument 'x'", x)
    _require_computable(op, "argument 'x'", x, floating=True)
    (out,) = x.ctx.emit(op, [x], {}, [x.type.materialized()])
    return out


def silu(x: Value) -> Value:
    """``x * sigmoid(x)``."""
    return _unary("silu", x)


def gelu(x: Value) -> Value:
    """Gaussian error linear unit."""
    return _unary("gelu", x)


def rsqrt(x: Value) -> Value:
    """``1 / sqrt(x)``."""
    return _unary("rsqrt", x)


def exp(x: Value) -> Value:
    return _unary("exp", x)


def sqrt(x: Value) -> Value:
    return _unary("sqrt", x)


# -- casts ------------------------------------------------------------------------


def cast(x: Value, dtype: DType) -> Value:
    """Cast to ``dtype``; a narrowing cast (one that can round) is flagged as a rounding point."""
    x = _require_value("to", "the tensor", x)
    if isinstance(dtype, BlockScaled) or isinstance(x.dtype, BlockScaled):
        raise TilefishTypeError(
            f"to: casting {x.type} to {dtype!r} involves a block-scaled format; quantize/dequantize is not part of M1"
        )
    if not isinstance(dtype, DType):
        raise TilefishTypeError(f"to: expected a dtype such as f32 or bf16, got {dtype!r}")
    if dtype == x.dtype:
        return x
    rounding = x.dtype.is_narrowing_to(dtype)
    (out,) = x.ctx.emit("cast", [x], {"to": dtype, "rounding": rounding}, [x.type.with_dtype(dtype)])
    return out


# -- reductions ---------------------------------------------------------------------


def _normalize_axes(op: str, x: Value, axis: Any) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(x.rank))
    axes = tuple(axis) if isinstance(axis, (tuple, list)) else (axis,)
    out = []
    for a in axes:
        if isinstance(a, bool) or not isinstance(a, int):
            raise TilefishTypeError(f"{op}: axis must be an int or a tuple of ints, got {axis!r}")
        if not -x.rank <= a < x.rank:
            raise TilefishTypeError(f"{op}: axis {a} is out of range for {x.type} (rank {x.rank})")
        out.append(a % x.rank)
    if len(set(out)) != len(out):
        raise TilefishTypeError(f"{op}: duplicate axes in {axis!r}")
    return tuple(sorted(out))


def _reduce(op: str, x: Any, axis: Any, keepdims: bool) -> Value:
    x = _require_value(op, "argument 'x'", x)
    _require_computable(op, "argument 'x'", x, floating=(op == "mean"))
    axes = _normalize_axes(op, x, axis)
    shape: list[DimExpr] = []
    for i, d in enumerate(x.shape):
        if i in axes:
            if keepdims:
                shape.append(as_dim(1))
        else:
            shape.append(d)
    (out,) = x.ctx.emit(op, [x], {"axes": axes, "keepdims": bool(keepdims)}, [TensorType(shape, x.dtype)])
    return out


def reduce_sum(x: Value, axis: Any = None, keepdims: bool = False) -> Value:
    """Sum over ``axis`` (an int, a tuple, or ``None`` for all axes) with numpy semantics."""
    return _reduce("sum", x, axis, keepdims)


def reduce_mean(x: Value, axis: Any = None, keepdims: bool = False) -> Value:
    """Mean over ``axis`` with numpy semantics (floating-point tensors only)."""
    return _reduce("mean", x, axis, keepdims)


def reduce_max(x: Value, axis: Any = None, keepdims: bool = False) -> Value:
    """Maximum over ``axis`` with numpy semantics."""
    return _reduce("max", x, axis, keepdims)


# -- split ---------------------------------------------------------------------------


def split(x: Any, size: Any, axis: int) -> tuple[Value, ...]:
    """Split ``axis`` into equal chunks of ``size``, or into chunks of the sizes in a list.

    Splitting along a non-leading axis gives *views* (slabs at an offset with
    the parent's strides), typed non-contiguous: they can feed traced
    elementwise and reduction ops but cannot be passed to a declared operator.
    """
    x = _require_value("split", "the tensor", x)
    if isinstance(axis, bool) or not isinstance(axis, int) or not -x.rank <= axis < x.rank:
        raise TilefishTypeError(f"split: axis {axis!r} is invalid for {x.type} (rank {x.rank})")
    ax = axis % x.rank
    dim = x.shape[ax]

    def as_size(s: Any) -> DimExpr:
        try:
            return as_dim(s)
        except TilefishTypeError:
            raise TilefishTypeError(f"split: sizes must be ints or dimensions, got {s!r}") from None

    if isinstance(size, (list, tuple)):
        sizes = [as_size(s) for s in size]
        total = DimExpr.const(0)
        for s in sizes:
            total = total + s
        if total != dim:
            raise TilefishTypeError(
                f"split: sizes {[str(s) for s in sizes]} add up to {total} but axis {axis} of {x.type} has size {dim}"
            )
    else:
        chunk = as_size(size)
        count = dim.exact_div(chunk)
        if count is None or count < 1:
            raise TilefishTypeError(
                f"split: axis {axis} of {x.type} has size {dim}, which is not a whole number of chunks of size {chunk}"
            )
        sizes = [chunk] * count
    # A slab along axis ax is contiguous only when everything before that axis has size 1, and it is a
    # contiguous *aligned* tensor only when it starts on a 16-byte boundary (the contract of every pointer).
    dense = x.type.contiguous and all(d.is_static and d.value == 1 for d in x.shape[:ax])
    out_types = []
    for s in sizes:
        shape = list(x.shape)
        shape[ax] = s
        try:
            out_types.append(TensorType(shape, x.dtype, known=x.ctx.known, contiguous=dense))
        except TilefishError as e:
            raise TilefishTypeError(f"split: chunk type is invalid: {e}") from None
    if dense:
        aligned = _aligned_chunks(x, ax, sizes)
        out_types = [t if ok else t.as_view() for t, ok in zip(out_types, aligned)]
    return x.ctx.emit("split", [x], {"axis": ax, "sizes": tuple(sizes)}, out_types)


def _mul(a: "DimExpr | None", b: "DimExpr | None") -> "DimExpr | None":
    """Product of two affine dimensions, or ``None`` when it is not affine (two symbolic factors)."""
    if a is None or b is None:
        return None
    try:
        return a * b
    except TilefishTypeError:
        return None


def _product(dims: "Sequence[DimExpr]") -> "DimExpr | None":
    out: DimExpr | None = DimExpr.const(1)
    for d in dims:
        out = _mul(out, d)
    return out


def _starts_aligned(offset: "DimExpr | None", bits: int, known: Any) -> bool:
    """Whether ``offset`` elements of ``bits`` bits provably span a multiple of 16 bytes."""
    if offset is None:
        return False
    return offset.is_divisible_by(128 // math.gcd(bits, 128), known)


def _aligned_chunks(x: Value, ax: int, sizes: "Sequence[DimExpr]") -> list[bool]:
    """For a split of a dense tensor, whether each chunk starts on a 16-byte boundary.

    A chunk starts ``(sizes before it) * (product of the dims after the axis)`` elements in. For a
    block-scaled tensor the scale tensor's chunk must be aligned as well.
    """
    t = x.type
    dtype = t.dtype
    data_bits = dtype.elem.bits if isinstance(dtype, BlockScaled) else dtype.bits
    known = x.ctx.known
    inner = _product(t.shape[ax + 1 :])
    scale_inner = None
    blocked = None
    if isinstance(dtype, BlockScaled):
        blocked = dtype.normalized_axis(t.rank)
        scale_inner = _product(dtype.scale_shape(t.shape, known)[ax + 1 :])
    flags: list[bool] = []
    before = DimExpr.const(0)
    for size in sizes:
        ok = before.is_static and before.value == 0  # the first chunk starts at the tensor's own (aligned) pointer
        if not ok:
            ok = _starts_aligned(_mul(before, inner), data_bits, known)
            if ok and isinstance(dtype, BlockScaled):
                scale_before = as_dim(cdiv(before, dtype.block)) if ax == blocked else before
                ok = _starts_aligned(_mul(scale_before, scale_inner), dtype.scale.bits, known)
        flags.append(ok)
        before = before + size
    return flags


# -- stores ----------------------------------------------------------------------------


def store(target: Any, key: Any, value: Any) -> None:
    """``param[...] = value``: copy ``value`` into an ``InOut``/``Out`` parameter.

    A store has copy semantics. For an ``InOut`` parameter the store node
    yields a fresh value, the parameter's new version, which later reads of
    the parameter see; ``value`` itself is untouched, and later in-place
    updates of ``value`` or of the parameter do not affect each other. A
    read-only input can never become writable through a store.

    The store node takes the parameter's *current* version as its first input and the stored value as
    its second, so consecutive versions are linked by dataflow (``store(%v1, %src)`` produces ``%v2``)
    and their order never rests on node order alone. Storing a parameter into itself (``x[...] = x``)
    is a copy onto itself: it is recorded like any store, but it is not a write, so it does not satisfy
    a kernel's check that every ``Out``/``InOut`` parameter is written.
    """
    if not isinstance(target, Value):
        raise TilefishTypeError(f"store: target must be a traced tensor, got {type(target).__name__}")
    if not (key is Ellipsis or (isinstance(key, slice) and key == slice(None))):
        raise TilefishTypeError(f"store into '{target.name}': only whole-tensor assignment `param[...] = value` is supported")
    if target.access not in ("inout", "out"):
        raise TilefishTypeError(
            f"store into '{target.name}': parameter is read-only; declare it InOut[...] or Out[...] to write it"
        )
    value = _require_value("store", "the stored value", value)
    if value.ctx is not target.ctx:
        raise TilefishTypeError(f"store into '{target.name}': the stored value belongs to a different trace")
    if value.type != target.type:
        hint = ""
        if not value.type.contiguous:
            hint = "; a strided view cannot be stored (strided views arrive in M2): combine it in an elementwise op first"
        elif value.dtype != target.dtype:
            hint = "; stores do not cast implicitly, use .to(dtype)"
        raise TilefishTypeError(f"store into '{target.name}': expected {target.type}, got {value.type}{hint}")
    current = target.live  # the version this store replaces (for an Out parameter, which has no versions: itself)
    versions = [target.type] if target.access == "inout" else []
    outs = target.ctx.emit("store", [current, value], {"target": target.name}, versions)
    if value is not current:  # `x[...] = x` copies a tensor onto itself: a node, but not a write
        target.ctx.written.add(target)
    if outs:
        current._latest = outs[0]
        target._latest = outs[0]
