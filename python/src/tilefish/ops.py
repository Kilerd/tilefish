"""Traced built-in operations: elementwise arithmetic, unary math, casts, reductions, split, store.

Every function here takes traced :class:`~tilefish.trace.Value` operands,
type-checks them, appends a node to the trace and returns new ``Value``s.
There is no implicit promotion between tensor dtypes; python scalars and
``const`` parameters adopt the dtype of the tensor they are combined with.
"""

from __future__ import annotations

from typing import Any

from .dims import DimExpr, as_dim
from .dtypes import BlockScaled, DType
from .errors import TilefishTypeError
from .ir import Scalar
from .trace import Const, TraceContext, Value
from .types import TensorType

_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


# -- operand checking ---------------------------------------------------------


def _require_value(op: str, label: str, x: Any) -> Value:
    if not isinstance(x, Value):
        raise TilefishTypeError(f"{op}: {label} must be a traced tensor value, got {type(x).__name__} ({x!r})")
    if x.access == "out":
        raise TilefishTypeError(f"{op}: {label} reads Out parameter '{x.name}', which is write-only")
    return x


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
            raise TilefishTypeError(
                f"{op}: cannot broadcast {a} with {b}: dimension {da} vs {db} at axis -{i} "
                "(shapes align on trailing dimensions; only a size-1 dimension broadcasts)"
            )
    out.reverse()
    return tuple(out)


# -- elementwise arithmetic ---------------------------------------------------


def binary(op: str, left: Any, right: Any) -> Value:
    """``left <op> right`` for ``op`` in ``add``, ``sub``, ``mul``, ``div``."""
    sym = _SYMBOLS[op]
    named = f"{op} ({sym})"
    vals = [(lab, x) for lab, x in (("left operand", left), ("right operand", right)) if isinstance(x, Value)]
    if not vals:
        raise TilefishTypeError(f"{named}: at least one operand must be a traced tensor value")
    ctx: TraceContext = vals[0][1].ctx
    for lab, x in vals:
        _require_value(named, lab, x)
        if x.ctx is not ctx:
            raise TilefishTypeError(f"{named}: {lab} belongs to a different trace")
    tensor_dtypes = []
    for lab, x in vals:
        tensor_dtypes.append(_require_computable(named, lab, x, floating=(op == "div")))
    if len(vals) == 2 and vals[0][1].dtype != vals[1][1].dtype:
        raise TilefishTypeError(
            f"{named}: no implicit promotion between tensor dtypes: left operand is {vals[0][1].type}, "
            f"right operand is {vals[1][1].type}; cast one side explicitly with .to(dtype)"
        )
    dtype = tensor_dtypes[0]
    operands = []
    for lab, x in (("left operand", left), ("right operand", right)):
        operands.append(x if isinstance(x, Value) else _scalar_operand(named, lab, x, dtype))
    shapes = [o.type for o in operands if isinstance(o, Value)]
    shape = shapes[0].shape if len(shapes) == 1 else broadcast_shapes(named, shapes[0], shapes[1])
    (out,) = ctx.emit(op, operands, {}, [TensorType(shape, dtype)])
    return out


# -- unary math -----------------------------------------------------------------


def _unary(op: str, x: Any) -> Value:
    _require_value(op, "argument 'x'", x)
    _require_computable(op, "argument 'x'", x, floating=True)
    (out,) = x.ctx.emit(op, [x], {}, [x.type])
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
    _require_value("to", "the tensor", x)
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
    _require_value(op, "argument 'x'", x)
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
    """Split ``axis`` into equal chunks of ``size``, or into chunks of the sizes in a list."""
    _require_value("split", "the tensor", x)
    if isinstance(axis, bool) or not isinstance(axis, int) or not -x.rank <= axis < x.rank:
        raise TilefishTypeError(f"split: axis {axis!r} is invalid for {x.type} (rank {x.rank})")
    ax = axis % x.rank
    dim = x.shape[ax]
    if isinstance(size, (list, tuple)):
        sizes = [as_dim(s) for s in size]
        total = DimExpr.const(0)
        for s in sizes:
            total = total + s
        if total != dim:
            raise TilefishTypeError(
                f"split: sizes {[str(s) for s in sizes]} add up to {total} but axis {axis} of {x.type} has size {dim}"
            )
    else:
        chunk = as_dim(size)
        count = dim.exact_div(chunk)
        if count is None or count < 1:
            raise TilefishTypeError(
                f"split: axis {axis} of {x.type} has size {dim}, which is not a whole number of chunks of size {chunk}"
            )
        sizes = [chunk] * count
    out_types = []
    for s in sizes:
        shape = list(x.shape)
        shape[ax] = s
        try:
            out_types.append(TensorType(shape, x.dtype, known=x.ctx.known))
        except TilefishTypeError as e:
            raise TilefishTypeError(f"split: chunk type is invalid: {e}") from None
    return x.ctx.emit("split", [x], {"axis": ax, "sizes": tuple(sizes)}, out_types)


# -- stores ----------------------------------------------------------------------------


def store(target: Any, key: Any, value: Any) -> None:
    """``param[...] = value``: write ``value`` into an ``InOut``/``Out`` parameter."""
    if not isinstance(target, Value):
        raise TilefishTypeError(f"store: target must be a traced tensor, got {type(target).__name__}")
    if not (key is Ellipsis or (isinstance(key, slice) and key == slice(None))):
        raise TilefishTypeError(f"store into '{target.name}': only whole-tensor assignment `param[...] = value` is supported")
    if target.access not in ("inout", "out"):
        raise TilefishTypeError(
            f"store into '{target.name}': parameter is read-only; declare it InOut[...] or Out[...] to write it"
        )
    _require_value("store", "the stored value", value)
    if value.ctx is not target.ctx:
        raise TilefishTypeError(f"store into '{target.name}': the stored value belongs to a different trace")
    if value.type != target.type:
        raise TilefishTypeError(
            f"store into '{target.name}': expected {target.type}, got {value.type}"
            + ("; stores do not cast implicitly, use .to(dtype)" if value.dtype != target.dtype else "")
        )
    target.ctx.emit("store", [target, value], {"target": target.name}, [])
