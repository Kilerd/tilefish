"""Traced built-in operations: elementwise arithmetic, comparisons and masks (``where``, ``iota``), unary math, casts, reductions (and
``argmax`` / ``argmin``), gathers and scatters of rows, views (split, reshape, permute, select), loads of a DevScalar, store.

Every function here takes traced :class:`~tilefish.trace.Value` operands,
type-checks them, appends a node to the trace and returns new ``Value``s.
There is no implicit promotion between tensor dtypes; python scalars and
``const`` parameters adopt the dtype of the tensor they are combined with.

The view operations (:func:`split`, :func:`reshape`, :func:`permute`, :func:`select`) copy nothing: the result is a
:class:`~tilefish.types.TensorType` with the strides and the element offset of the view, in the dimension algebra. A view
that is dense and provably starts on a 16-byte boundary is a contiguous tensor (it can be passed anywhere a ``Tensor[...]``
is wanted); any other view is a ``View[...]``, and only a ``Strided[...]`` parameter of an extern takes it. A stride or an
offset that is not an affine expression (a product of two symbolic dimensions) is unknown, and such a view is opaque.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .dims import DimExpr, as_dim, cdiv, describe_pair, dim_bounds
from .dtypes import BlockScaled, DType, bool_, i32, i64
from .errors import TilefishError, TilefishTypeError
from .ir import Scalar
from .trace import Const, TraceContext, Value
from .types import TensorType, _mul, product as _product, starts_aligned as _starts_aligned

_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_INDEX_DEFAULT = i32
_INDEX_DTYPES = (i32, i64)


# -- operand checking ---------------------------------------------------------


def _require_value(op: str, label: str, x: Any, *, devscalar: bool = False) -> Value:
    """Check that ``x`` is a readable traced tensor and return the version a read sees now.

    A parameter reads its newest version; a value that an in-place call or a store consumed cannot be read. A ``DevScalar``
    (one element in device memory) is not an operand of any operation until ``tilefish.load`` has read it (``devscalar=True`` is
    for ``load`` itself).
    """
    if not isinstance(x, Value):
        raise TilefishTypeError(f"{op}: {label} must be a traced tensor value, got {type(x).__name__} ({x!r})")
    if x.access == "out":
        raise TilefishTypeError(f"{op}: {label} reads Out parameter '{x.name}', which is write-only")
    if x.type.devscalar and not devscalar:
        raise TilefishTypeError(
            f"{op}: {label} is the DevScalar {x.ref} ({x.type}), one element in device memory; read it first with tilefish.load({x.name})"
        )
    return x._current(f"{op}: {label}")


def _require_computable(op: str, label: str, x: Value, *, floating: bool = False) -> DType:
    dt = x.dtype
    if isinstance(dt, BlockScaled):
        raise TilefishTypeError(
            f"{op}: {label} has block-scaled dtype {dt!r}; it can only be passed to operators that "
            "declare it (library/extern), not combined directly"
        )
    if dt.is_bool:
        raise TilefishTypeError(
            f"{op}: {label} is the mask {x.ref} ({x.type}), which has no arithmetic and no order: combine masks with & | ^ ~, select "
            "with tilefish.where(mask, a, b), or make it a number with .to(u8) (0 or 1)"
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


def _operands(
    named: str,
    sides: "list[tuple[str, Any]]",
    *,
    floating: bool = False,
    dtype: "DType | None" = None,
    masks: bool = False,
    ctx: "TraceContext | None" = None,
) -> "tuple[TraceContext, list[Value | Scalar], DType, tuple[DimExpr, ...]]":
    """Resolve the operands of an elementwise op: the trace, the operands (scalars typed), their common dtype and the broadcast shape.

    Tensors must have one dtype (no implicit promotion); a python number or a ``const`` takes that dtype and is rounded to it. ``dtype``
    is the dtype the caller requires (``tilefish.where(..., dtype=...)``), and the only source of it when every operand is a scalar.
    ``masks`` lets the operands be masks (``where`` selects between masks too); ``ctx`` is the trace when it is known otherwise (the mask
    of a ``where`` whose operands are scalars).
    """
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
    dtypes = [
        x.dtype if (masks and isinstance(x.dtype, DType) and x.dtype.is_bool) else _require_computable(named, lab, x, floating=floating)
        for lab, x in tensors
    ]
    for (lab, x), dt in zip(tensors, dtypes):
        if dt != dtypes[0]:
            raise TilefishTypeError(
                f"{named}: no implicit promotion between tensor dtypes: {tensors[0][0]} is {tensors[0][1].type}, "
                f"{lab} is {x.type}; cast one side explicitly with .to(dtype)"
            )
        if dtype is not None and dt != dtype:
            raise TilefishTypeError(f"{named}: {lab} is {x.type}, but dtype={dtype} was asked for; cast it with .to({dtype})")
    if not dtypes and dtype is None:
        raise TilefishTypeError(f"{named}: every operand is a scalar, so the result has no dtype; pass dtype=... (f32, bf16, i32, ...)")
    common = dtypes[0] if dtypes else dtype
    assert common is not None
    if common.is_bool and len(tensors) < len(sides):
        raise TilefishTypeError(f"{named}: masks combine with masks, not with python numbers or consts; compare first (x > 0)")
    operands = [x if isinstance(x, Value) else _scalar_operand(named, lab, x, common) for (lab, _), x in zip(sides, resolved)]
    acc: "TensorType | None" = None  # the broadcast of the tensor operands so far (the first operand's own type until there is a second)
    for o in operands:
        if isinstance(o, Value):
            acc = o.type if acc is None else TensorType(broadcast_shapes(named, acc, o.type), common)
    return ctx, operands, common, acc.shape if acc is not None else ()


def binary(op: str, left: Any, right: Any) -> Value:
    """``left <op> right`` for ``op`` in ``add``, ``sub``, ``mul``, ``div``."""
    named = f"{op} ({_SYMBOLS[op]})"
    ctx, operands, dtype, shape = _operands(named, [("left operand", left), ("right operand", right)], floating=(op == "div"))
    (out,) = ctx.emit(op, operands, {}, [TensorType(shape, dtype)])
    return out


_INT_DIVISION = {"floordiv": "//", "mod": "%"}


def int_division(op: str, x: Any, divisor: Any) -> Value:
    """``x // c`` (``floordiv``) or ``x % c`` (``mod``) of an integer tensor by a positive constant, with Python's semantics: the quotient
    rounds toward minus infinity and the remainder is in ``[0, c)`` (not C's, which truncate: ``-7 // 2`` is ``-4`` and ``-7 % 2`` is ``1``).

    The divisor is a python int, an integer ``const`` or a static dimension, and positive: a tensor divisor could be zero or negative, so it
    is refused. Exact (no overflow is possible). For index arithmetic: ``iota(x, -1) % H``, ``(p // W, p % W)``.
    """
    named = f"{op} ({_INT_DIVISION[op]})"
    if not isinstance(x, Value):
        raise TilefishTypeError(
            f"{named}: the divisor {divisor!r} is a traced tensor; a divisor is a positive constant (a python int, an integer const or a static "
            "dimension), because a tensor could be zero or negative"
        )
    x = _require_value(named, "the dividend", x)
    dt = _require_computable(named, "the dividend", x)
    if not dt.is_int:
        raise TilefishTypeError(f"{named}: {x.ref} is {x.type}; // and % are integer index arithmetic here (use / on floating-point tensors)")
    if isinstance(divisor, Value):
        raise TilefishTypeError(
            f"{named}: the divisor {divisor.ref} is a traced tensor; a divisor is a positive constant (a python int, an integer const or a "
            "static dimension), because a tensor could be zero or negative"
        )
    c = _scalar_operand(named, "the divisor", divisor, dt)
    if c.value < 1:
        raise TilefishTypeError(f"{named}: the divisor must be positive, got {c.value}")
    (out,) = x.ctx.emit(op, [x, c], {}, [x.type.materialized()])
    return out


# -- comparisons, masks, where, iota -------------------------------------------------------------------------------------------

COMPARISONS = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
LOGICAL = {"and": "&", "or": "|", "xor": "^"}


def compare(op: str, left: Any, right: Any) -> Value:
    """``left <op> right`` for ``op`` in ``eq ne lt le gt ge``: a mask (:data:`~tilefish.dtypes.bool_`) of the broadcast shape.

    The operands have one dtype (no implicit promotion; a python number or a ``const`` takes the tensor's dtype and is rounded to it,
    as in arithmetic). Floats compare as IEEE 754 does: ``-0.0 == +0.0``; a NaN is unordered, so every comparison with a NaN is
    false except ``!=``, which is true (``x != x`` is the NaN test). Integers compare by value (``u8`` unsigned).
    """
    named = f"comparison ({COMPARISONS[op]})"
    ctx, operands, _, shape = _operands(named, [("left operand", left), ("right operand", right)])
    (out,) = ctx.emit(op, operands, {}, [TensorType(shape, bool_)])
    return out


def _require_mask(named: str, label: str, x: Any) -> Value:
    if not isinstance(x, Value):
        raise TilefishTypeError(
            f"{named}: {label} must be a mask (a comparison of traced tensors, such as x > 0), got {type(x).__name__} ({x!r})"
        )
    x = _require_value(named, label, x)
    if not (isinstance(x.dtype, DType) and x.dtype.is_bool):
        raise TilefishTypeError(
            f"{named}: {label} is {x.type}, not a mask: a mask comes from a comparison (x != 0 reads a stored u8 mask back)"
        )
    return x


def logical(op: str, left: Any, right: Any) -> Value:
    """``left & right``, ``left | right``, ``left ^ right`` of two masks (logical and, or, exclusive or), broadcast."""
    named = f"{op} ({LOGICAL[op]})"
    a = _require_mask(named, "left operand", left)
    b = _require_mask(named, "right operand", right)
    if a.ctx is not b.ctx:
        raise TilefishTypeError(f"{named}: the operands belong to different traces")
    (out,) = a.ctx.emit(op, [a, b], {}, [TensorType(broadcast_shapes(named, a.type, b.type), bool_)])
    return out


def logical_not(x: Any) -> Value:
    """``~mask``: the logical negation of a mask."""
    x = _require_mask("not (~)", "the operand", x)
    (out,) = x.ctx.emit("not", [x], {}, [x.type.materialized()])
    return out


def where(mask: Any, a: Any, b: Any, *, dtype: "DType | None" = None) -> Value:
    """``tilefish.where(mask, a, b)``: ``a`` where ``mask`` is true, else ``b``, element by element, broadcast over the three.

    A select, not arithmetic: the bits of the chosen operand pass through unchanged (a NaN keeps its payload, ``-0.0`` its sign).
    ``a`` and ``b`` have one dtype (no implicit promotion); a python number or a ``const`` takes the dtype of the tensor operand and is
    rounded to it. When both are scalars, ``dtype=`` gives the result's dtype (``tilefish.where(j < n, 0.0, -1e4, dtype=bf16)``).
    """
    named = "where"
    if dtype is not None and not isinstance(dtype, DType):
        raise TilefishTypeError(f"where: dtype must be a dtype such as f32 or bf16, got {dtype!r}")
    m = _require_mask(named, "the mask", mask)
    _, operands, dt, shape = _operands(named, [("argument 'a'", a), ("argument 'b'", b)], dtype=dtype, masks=True, ctx=m.ctx)
    out_shape = broadcast_shapes(named, m.type, TensorType(shape, dt))
    (out,) = m.ctx.emit("where", [m, *operands], {}, [TensorType(out_shape, dt)])
    return out


def iota(x: Any, axis: int = -1, *, dtype: "DType | None" = None) -> Value:
    """The index of every position of ``x`` along ``axis``, as an ``i32`` tensor (``dtype=i64`` for ``i64``).

    Only the shape of ``x`` is used (nothing is read; ``x`` may be an ``Out`` parameter). The result varies along ``axis`` only, so it
    has the broadcastable shape ``x.shape[axis:]`` with size 1 after ``axis``: for ``x: [R, D]``, ``iota(x, axis=0)`` is ``[R, 1]`` (the row
    index) and ``iota(x, axis=-1)`` is ``[D]`` (the column index); for ``x: [n]`` it is ``[n]``. ``tilefish.iota(x, 0) < tilefish.load(n)`` is
    the mask of the rows below a device-scalar count.
    """
    if not isinstance(x, Value):
        raise TilefishTypeError(f"iota: takes a traced tensor (for its shape), got {type(x).__name__} ({x!r})")
    if x.type.rank == 0:
        raise TilefishTypeError(f"iota: {x.ref} ({x.type}) has rank 0, so it has no axis to count along")
    if isinstance(axis, bool) or not isinstance(axis, int) or not -x.rank <= axis < x.rank:
        raise TilefishTypeError(f"iota: axis {axis!r} is invalid for {x.type} (rank {x.rank})")
    if dtype is None:
        dtype = _INDEX_DEFAULT
    if dtype not in _INDEX_DTYPES:
        raise TilefishTypeError(f"iota: dtype must be i32 or i64, got {dtype!r}")
    ax = axis % x.rank
    shape = (x.shape[ax], *(as_dim(1) for _ in x.shape[ax + 1 :]))
    (out,) = x.ctx.emit("iota", [], {"dim": x.shape[ax]}, [TensorType(shape, dtype)])
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


def sin(x: Value) -> Value:
    """The sine (libdevice's ``sinf`` in a lowered kernel; see note 10 of :mod:`tilefish.lower`)."""
    return _unary("sin", x)


def cos(x: Value) -> Value:
    """The cosine (libdevice's ``cosf`` in a lowered kernel; see note 10 of :mod:`tilefish.lower`)."""
    return _unary("cos", x)


# -- casts ------------------------------------------------------------------------


NAN_MODES = ("canonical", "preserve")


def cast(x: Value, dtype: DType, nan: "str | None" = None) -> Value:
    """Cast to ``dtype``; a narrowing cast (one that can round) is flagged as a rounding point.

    ``nan`` says what the f32 to bf16 cast does with a NaN (note 1 of :mod:`tilefish.lower`): ``"canonical"`` (the default, PTX
    ``cvt.rn.bf16.f32``) gives the canonical NaN ``0x7FFF``; ``"preserve"`` is the integer round to nearest even on the bit pattern, which
    keeps a NaN's sign and the top of its payload (quiet bit set): ``0xFFC00000`` becomes ``0xFFC0``. The two agree on every other input.
    Only that cast has the option.

    A cast to the dtype the tensor already has emits no node and behaves exactly like reading the
    tensor at that point: it returns the version that is current at the time of the cast, and that
    version does not follow later updates: a later in-place call or store into the parameter consumes
    it, and reading it then raises (the updated tensor is its ``.live``). This holds before a
    parameter's first store as well as after it.
    """
    x = _require_value("to", "the tensor", x)
    if isinstance(dtype, BlockScaled) or isinstance(x.dtype, BlockScaled):
        raise TilefishTypeError(
            f"to: casting {x.type} to {dtype!r} involves a block-scaled format; quantize/dequantize is not part of M1"
        )
    if not isinstance(dtype, DType):
        raise TilefishTypeError(f"to: expected a dtype such as f32 or bf16, got {dtype!r}")
    if dtype.is_bool and dtype != x.dtype:
        raise TilefishTypeError(
            f"to: {x.ref} ({x.type}) cannot be cast to a mask: a mask comes from a comparison, so write the test it stands for "
            "(x != 0 is what a cast to bool would mean, and is true for a NaN)"
        )
    if nan is not None:
        if nan not in NAN_MODES:
            raise TilefishTypeError(f"to: nan must be 'canonical' or 'preserve', got {nan!r}")
        if not (x.dtype.name == "f32" and dtype.name == "bf16"):
            raise TilefishTypeError(
                f"to: nan= is a mode of the f32 to bf16 cast; casting {x.type} to {dtype} has one NaN rule (see note 1 of the numerics contract)"
            )
    if dtype == x.dtype:
        return x._reference()
    rounding = x.dtype.is_narrowing_to(dtype)
    attrs: dict[str, Any] = {"to": dtype, "rounding": rounding}
    if nan == "preserve":  # the default, "canonical", is the plain cast and leaves the node as it always was
        attrs["nan"] = nan
    (out,) = x.ctx.emit("cast", [x], attrs, [x.type.with_dtype(dtype)])
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


class _Required:
    def __repr__(self) -> str:
        return "<required>"


_REQUIRED: Any = _Required()


def _arg_reduce(op: str, x: Any, axis: Any, dtype: Any, keepdims: Any) -> Value:
    x = _require_value(op, "argument 'x'", x)
    _require_computable(op, "argument 'x'", x)
    if x.rank == 0:
        raise TilefishTypeError(f"{op}: {x.ref} ({x.type}) has rank 0, so it has no axis to reduce")
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TilefishTypeError(f"{op}: axis must be one int (the index of one axis is returned), got {axis!r}")
    (ax,) = _normalize_axes(op, x, axis)
    if dtype is _REQUIRED:
        raise TilefishTypeError(f"{op}: say the index type: dtype=i32 or dtype=i64 (it is part of the result's type)")
    if dtype not in _INDEX_DTYPES:
        raise TilefishTypeError(f"{op}: dtype must be i32 or i64, got {dtype!r}")
    if not isinstance(keepdims, bool):
        raise TilefishTypeError(f"{op}: keepdims must be True or False, got {keepdims!r}")
    shape = [as_dim(1) if i == ax else d for i, d in enumerate(x.shape)] if keepdims else [d for i, d in enumerate(x.shape) if i != ax]
    (out,) = x.ctx.emit(op, [x], {"axes": (ax,), "keepdims": keepdims, "dtype": dtype}, [TensorType(shape, dtype)])
    return out


def argmax(x: Any, axis: Any = -1, *, dtype: Any = _REQUIRED, keepdims: Any = False) -> Value:
    """The index of the maximum along ``axis``, as ``dtype`` (``i32`` or ``i64``, required).

    Exact, and the same as torch's ``argmax``: the first maximum wins (``-0.0`` and ``+0.0`` are equal, so the first of them); a NaN is
    larger than every number, so the first NaN wins; a row of ``-inf`` gives 0. A bf16 or f16 value is compared as its (exact) f32 value.
    """
    return _arg_reduce("argmax", x, axis, dtype, keepdims)


def argmin(x: Any, axis: Any = -1, *, dtype: Any = _REQUIRED, keepdims: Any = False) -> Value:
    """The index of the minimum along ``axis``, as ``dtype`` (``i32`` or ``i64``, required).

    Exact, and the same as torch's ``argmin``: the first minimum wins (``-0.0`` and ``+0.0`` are equal); a NaN wins here too (torch's
    rule: a NaN propagates through min and max alike), so the first NaN; a row of ``+inf`` gives 0.
    """
    return _arg_reduce("argmin", x, axis, dtype, keepdims)


# -- split ---------------------------------------------------------------------------


def split(x: Any, size: Any, axis: int) -> tuple[Value, ...]:
    """Split ``axis`` into equal chunks of ``size``, or into chunks of the sizes in a list.

    The chunks are views of the parent, with their strides and offset in the type. Splitting along the leading axis (or after
    size-1 axes) gives dense slabs: a slab that provably starts on a 16-byte boundary is a contiguous tensor and can be passed to
    any operator, one that does not is a view. Splitting along another axis gives strided slabs (the parent's strides at an
    offset), typed ``View[...]``: they can feed traced elementwise and reduction ops, and be passed to an extern parameter written
    ``Strided[...]``, but not to a parameter that takes a contiguous tensor.

    A chunk is a value, even where its type says "view": an in-place call on a chunk (a call that takes
    it as an ``InOut`` argument) consumes that chunk and produces a new version of the chunk. It does
    **not** change the parent, which keeps its own version, and the siblings are untouched. Lowering
    copies; there is no write-through to the parent.
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
    if isinstance(x.dtype, BlockScaled):
        out_types = _blockscaled_chunks(x, ax, sizes)
    else:
        out_types = _chunks(x, ax, sizes)
    return x.ctx.emit("split", [x], {"axis": ax, "sizes": tuple(sizes)}, out_types)


def _chunks(x: Value, ax: int, sizes: "Sequence[DimExpr]") -> "list[TensorType]":
    """The types of the chunks of a plain tensor: the strides of the parent, and an offset that is the parent's plus the chunk's start.

    A chunk starts ``(sizes before it) * (stride of the axis)`` elements in. A slab along axis ``ax`` of a dense tensor is dense
    again only when everything before that axis has size 1 (or the axis is the first one); then its strides are the default and
    only the offset says where it is. Any other chunk is a strided view with the parent's strides. Whether a chunk is a contiguous
    tensor (dense and provably starting on a 16-byte boundary: the contract of every pointer) follows from its type.
    """
    t = x.type
    strides = t.layout_strides()
    dense_chunks = t.is_dense and all(d.is_static and d.value == 1 for d in x.shape[:ax])
    out_types: list[TensorType] = []
    before = DimExpr.const(0)
    for s in sizes:
        shape = list(x.shape)
        shape[ax] = s
        step = _mul(before, strides[ax])
        offset = None if (t.offset is None or step is None) else t.offset + step
        try:
            out_types.append(
                TensorType(
                    shape,
                    x.dtype,
                    known=x.ctx.known,
                    strides=None if dense_chunks else strides,
                    offset=offset,
                )
            )
        except TilefishError as e:
            raise TilefishTypeError(f"split: chunk type is invalid: {e}") from None
        before = before + s
    return out_types


def _blockscaled_chunks(x: Value, ax: int, sizes: "Sequence[DimExpr]") -> "list[TensorType]":
    """A chunk of a block-scaled tensor is contiguous only when it is a dense slab whose data *and scales* start on 16-byte boundaries.

    Anything else is a view of unknown layout (data and scales are two buffers: a strided view of both is not expressed yet).
    """
    dense = x.type.contiguous and all(d.is_static and d.value == 1 for d in x.shape[:ax])
    out_types = []
    for s in sizes:
        shape = list(x.shape)
        shape[ax] = s
        try:
            out_types.append(TensorType(shape, x.dtype, known=x.ctx.known, aligned=dense))
        except TilefishError as e:
            raise TilefishTypeError(f"split: chunk type is invalid: {e}") from None
    if dense:
        aligned = _aligned_chunks(x, ax, sizes)
        out_types = [t if ok else t.as_view() for t, ok in zip(out_types, aligned)]
    else:
        out_types = [t.as_view() for t in out_types]
    return out_types


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


# -- views: reshape, permute, select --------------------------------------------------------


def _viewable(op: str, x: Any) -> Value:
    """The readable tensor a view operation takes: not block-scaled (data and scales are two buffers), not a loaded DevScalar."""
    x = _require_value(op, "the tensor", x)
    if isinstance(x.dtype, BlockScaled):
        raise TilefishTypeError(
            f"{op}: {x.ref} is {x.type} with the block-scaled dtype {x.dtype!r}; a view of a block-scaled tensor (data plus "
            "scales) is not supported yet, only split is"
        )
    return x


def _view_type(op: str, x: Value, shape: "Sequence[DimExpr]", strides: "Sequence[DimExpr | None] | None", offset: "DimExpr | None") -> TensorType:
    try:
        return TensorType(shape, x.dtype, known=x.ctx.known, strides=strides, offset=offset)
    except TilefishError as e:
        raise TilefishTypeError(f"{op}: the type of the view is invalid: {e}") from None


def _monomial(dims: "Sequence[DimExpr]") -> "tuple[int, tuple]":
    """A product of dimensions as ``(integer factor, sorted irreducible factors)``: two products with the same monomial are equal."""
    factor = 1
    atoms: list = []
    for d in dims:
        d = d.folded()
        if d.is_static:
            factor *= d.value
            continue
        g = math.gcd(*(abs(c) for _, c in d.terms), abs(d.offset)) if d.terms else 1
        if g > 1 and (d.offset == 0 or d.offset % g == 0):
            factor *= g
            d = DimExpr._make({a: c // g for a, c in d.terms}, d.offset // g)
        atoms.append(d._key)
    return factor, tuple(sorted(atoms, key=repr))


def _same_count(a: "Sequence[DimExpr]", b: "Sequence[DimExpr]") -> bool:
    """Whether two shapes provably have the same number of elements."""
    pa, pb = _product(a), _product(b)
    if pa is not None and pb is not None:
        return pa == pb
    return _monomial(a) == _monomial(b)


def reshape(x: Any, *dims: Any) -> Value:
    """A view of the same elements in another shape; the input must be dense (a reshape of a strided view would have to copy).

    ``x.reshape(M, 3, H, D)`` or ``x.reshape((M, 3, H, D))``. The number of elements must provably stay the same (affine products of
    dimensions are compared; a product of two symbolic dimensions is compared factor by factor). The result starts where the input
    does; whether it is a contiguous tensor follows from that offset.
    """
    x = _viewable("reshape", x)
    if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
        dims = tuple(dims[0])
    if not dims:
        raise TilefishTypeError("reshape: needs the new shape, e.g. x.reshape(M, 3, H, D)")
    new: list[DimExpr] = []
    for d in dims:
        try:
            dim = as_dim(d)
        except TilefishTypeError:
            raise TilefishTypeError(f"reshape: the shape must be ints or dimensions, got {d!r}") from None
        if dim.is_static and dim.value < 1:
            raise TilefishTypeError(f"reshape: sizes must be positive (there is no -1 inference), got {dim.value}")
        new.append(dim)
    t = x.type
    if not t.is_dense:
        raise TilefishTypeError(
            f"reshape: {x.ref} is the view {t} ({t.layout_text()}); only a dense tensor can be reshaped without a copy. "
            "Compute from it first (x + 0 makes a contiguous copy), or reshape the tensor it is a view of"
        )
    if not _same_count(t.shape, new):
        raise TilefishTypeError(
            f"reshape: {x.ref} has shape {[str(d) for d in t.shape]}, which cannot be shown to have as many elements as "
            f"{[str(d) for d in new]}"
        )
    out_type = _view_type("reshape", x, new, None, t.offset)
    (out,) = x.ctx.emit("reshape", [x], {"shape": tuple(new)}, [out_type])
    return out


def permute(x: Any, *axes: Any) -> Value:
    """A view with the axes in another order: the strides move with them and nothing is copied. ``x.permute(1, 0, 2)``."""
    x = _viewable("permute", x)
    if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
        axes = tuple(axes[0])
    rank = x.rank
    if len(axes) != rank or any(isinstance(a, bool) or not isinstance(a, int) for a in axes):
        raise TilefishTypeError(f"permute: needs {rank} integer axes for {x.type}, got {list(axes)!r}")
    order = []
    for a in axes:
        if not -rank <= a < rank:
            raise TilefishTypeError(f"permute: axis {a} is out of range for {x.type} (rank {rank})")
        order.append(a % rank)
    if sorted(order) != list(range(rank)):
        raise TilefishTypeError(f"permute: {list(axes)!r} is not a permutation of the axes of {x.type}")
    t = x.type
    if order == list(range(rank)):
        out_type = t
    else:
        strides = t.layout_strides()
        out_type = _view_type("permute", x, [t.shape[a] for a in order], [strides[a] for a in order], t.offset)
    (out,) = x.ctx.emit("permute", [x], {"axes": tuple(order)}, [out_type])
    return out


def _lower_bound(d: DimExpr) -> int:
    return dim_bounds(d)[0]


def select(x: Any, axis: Any, index: Any) -> Value:
    """A view without ``axis``: the slice at the integer ``index`` (``x[i]`` is ``x.select(0, i)``). The offset moves by ``index * stride``."""
    x = _viewable("select", x)
    t = x.type
    if isinstance(axis, bool) or not isinstance(axis, int) or not -t.rank <= axis < t.rank:
        raise TilefishTypeError(f"select: axis {axis!r} is invalid for {t} (rank {t.rank})")
    if isinstance(index, bool) or not isinstance(index, int):
        raise TilefishTypeError(f"select: the index must be a python int (a traced tensor has no value at trace time), got {index!r}")
    ax = axis % t.rank
    size = t.shape[ax]
    i = index
    if size.is_static:
        if not -size.value <= i < size.value:
            raise TilefishTypeError(f"select: index {index} is out of range for axis {axis} of {t} (size {size.value})")
        i %= size.value
    else:
        if i < 0:
            raise TilefishTypeError(f"select: a negative index needs a static size; axis {axis} of {t} has size {size}")
        if i >= _lower_bound(size):
            raise TilefishTypeError(
                f"select: index {index} is not provably in range for axis {axis} of {t}: its size {size} can be as small as "
                f"{_lower_bound(size)}"
            )
    strides = t.layout_strides()
    step = _mul(DimExpr.const(i), strides[ax])
    offset = None if (t.offset is None or step is None) else t.offset + step
    keep = [k for k in range(t.rank) if k != ax]
    out_type = _view_type("select", x, [t.shape[k] for k in keep], [strides[k] for k in keep], offset)
    (out,) = x.ctx.emit("select", [x], {"axis": ax, "index": i}, [out_type])
    return out


def getitem(x: Any, key: Any) -> Value:
    """``x[i]``, ``x[i, j]``, ``x[:, i]``, ``x[..., i]``: integer indices select (a view without that axis); ``:`` keeps an axis.

    Only integers, ``:`` and ``...`` are indices; a range is taken with ``split``.
    """
    x = _viewable("indexing", x)
    items = key if isinstance(key, tuple) else (key,)
    if items.count(Ellipsis) > 1:
        raise TilefishTypeError("indexing: at most one ... is allowed")
    consumed = sum(1 for k in items if k is not Ellipsis)
    if consumed > x.rank:
        raise TilefishTypeError(f"indexing: {consumed} indices for {x.type} (rank {x.rank})")
    expanded: list[Any] = []
    for k in items:
        if k is Ellipsis:
            expanded.extend([slice(None)] * (x.rank - consumed))
        else:
            expanded.append(k)
    out = x
    axis = 0
    for k in expanded:
        if isinstance(k, slice):
            if k != slice(None):
                raise TilefishTypeError(
                    f"indexing {x.ref}: only integers, ':' and '...' are indices; take a range with .split(size, axis=...)"
                )
            axis += 1
        elif isinstance(k, int) and not isinstance(k, bool):
            out = select(out, axis, k)
        else:
            raise TilefishTypeError(f"indexing {x.ref}: an index must be a python int, ':' or '...', got {k!r}")
    if out is x:  # x[:] and x[...] name the tensor itself
        return x._reference()
    return out


# -- gather and scatter of rows -----------------------------------------------------------------------

GATHER_OOB = ("zero", "clamp")
SCATTER_OOB = ("skip",)


def _index_operand(op: str, idx: Any) -> Value:
    idx = _require_value(op, "the index", idx)
    dt = idx.dtype
    if not (isinstance(dt, DType) and dt.is_int):
        raise TilefishTypeError(f"{op}: the index {idx.ref} is {idx.type}; an index is an integer tensor (i32, i64 or u8)")
    if idx.rank == 0:
        raise TilefishTypeError(f"{op}: the index {idx.ref} has rank 0; give it a shape ([R] for R rows)")
    return idx


def _oob(op: str, oob: Any, allowed: "tuple[str, ...]") -> str:
    if oob is _REQUIRED:
        raise TilefishTypeError(
            f"{op}: say what an index out of range does: oob={' or '.join(repr(a) for a in allowed)} (an index is device data, which "
            "nothing checks before the kernel reads it)"
        )
    if oob not in allowed:
        raise TilefishTypeError(f"{op}: oob must be {' or '.join(repr(a) for a in allowed)}, got {oob!r}")
    return str(oob)


def gather_rows(x: Any, idx: Any, *, oob: Any = _REQUIRED) -> Value:
    """``x[idx]`` along the first axis: ``out[i..., :] = x[idx[i...], :]``, a raw copy of the rows (NaN payloads, signed zeros as they are).

    ``x`` is ``[N, ...]`` (in a lowered kernel a read-only parameter, ``[N, D]`` or ``[N]``) and ``idx`` an integer tensor; the result has
    the shape ``idx.shape + x.shape[1:]`` and the dtype of ``x``. An index is device data, so what an index outside ``[0, N)`` does is part of
    the call: ``oob="zero"`` gives a row of zeros (``+0.0``, all bits clear), ``oob="clamp"`` the nearest row (0 or ``N - 1``). An embedding
    lookup is ``gather_rows(table, ids, oob="zero")``.
    """
    named = "gather_rows"
    x = _require_value(named, "the table", x)
    if isinstance(x.dtype, BlockScaled):
        raise TilefishTypeError(f"{named}: {x.ref} is block-scaled ({x.type}); a gather of data and scales is not supported yet")
    if x.rank == 0:
        raise TilefishTypeError(f"{named}: the table {x.ref} has rank 0; it has no rows")
    if isinstance(x.dtype, DType) and x.dtype.is_bool:
        raise TilefishTypeError(f"{named}: the table {x.ref} is a mask, which is not in memory")
    idx = _index_operand(named, idx)
    if idx.ctx is not x.ctx:
        raise TilefishTypeError(f"{named}: the table and the index belong to different traces")
    policy = _oob(named, oob, GATHER_OOB)
    (out,) = x.ctx.emit(named, [x, idx], {"oob": policy}, [TensorType((*idx.shape, *x.shape[1:]), x.dtype)])
    return out


def scatter_rows(dst: Any, idx: Any, src: Any, *, oob: Any = _REQUIRED) -> None:
    """``dst[idx] = src`` along the first axis, in place: ``dst[idx[i...], :] = src[i..., :]``, a raw copy; the rows of ``dst`` that no index
    names keep their bytes (an index copy).

    ``dst`` is an ``InOut`` parameter ``[N, ...]``, ``idx`` an integer tensor and ``src`` has the shape ``idx.shape + dst.shape[1:]`` and the dtype
    of ``dst``. ``oob="skip"`` (required) says that a source row whose index is outside ``[0, N)`` is not written. The indices in range must
    be distinct: two sources for one row race, and which bytes land is not defined (as in torch's ``index_copy_``). Like a store, the call
    replaces the version of ``dst`` it writes; the parameter's name then reads the new one.
    """
    named = "scatter_rows"
    if not isinstance(dst, Value):
        raise TilefishTypeError(f"{named}: the target must be a traced tensor, got {type(dst).__name__}")
    if not dst.is_param or dst.access != "inout":
        what = "an Out parameter, whose rows that no index names would be left unwritten" if dst.access == "out" else "not an InOut parameter"
        raise TilefishTypeError(f"{named}: the target {dst.ref} ({dst.type}) is {what}; declare it InOut[...] (the rows not named keep their bytes)")
    idx = _index_operand(named, idx)
    src = _require_value(named, "the source", src)
    if src.ctx is not dst.ctx or idx.ctx is not dst.ctx:
        raise TilefishTypeError(f"{named}: the target, the index and the source belong to different traces")
    want = TensorType((*idx.shape, *dst.shape[1:]), dst.dtype)
    if src.type != want:
        raise TilefishTypeError(f"{named}: the source is {src.type}, but rows of {dst.type} indexed by {idx.type} are {want}")
    policy = _oob(named, oob, SCATTER_OOB)
    current = dst.live  # the version this scatter replaces
    (new,) = dst.ctx.emit(named, [current, idx, src], {"target": dst.name, "oob": policy}, [dst.type])
    dst.ctx.written.add(dst)
    current._consumed_by = (dst.ctx.nodes[-1], dst.name)
    current._latest = new
    dst._latest = new


# -- DevScalar ---------------------------------------------------------------------------------


def load(x: Any) -> Value:
    """Read a ``DevScalar`` parameter: a rank-0 tensor of its dtype. A lowered kernel reads it with one ``tl.load(ptr)``, broadcast."""
    x = _require_value("load", "argument 's'", x, devscalar=True)
    if not x.type.devscalar:
        raise TilefishTypeError(f"load: takes a DevScalar parameter (one element in device memory), got {x.ref} ({x.type})")
    (out,) = x.ctx.emit("load", [x], {}, [TensorType((), x.dtype)])
    return out


# -- copies ----------------------------------------------------------------------------


def copy(x: Any) -> Value:
    """``tilefish.copy(x)``: a new contiguous tensor with the elements of ``x`` (its shape and dtype), for a graph body.

    Values are immutable, so a copy is only needed where *storage* matters: a launch plan makes it a **copy step**, a device-to-device copy of
    the dense bytes of ``x`` (a tensor, or a dense view of one: a ``select`` of the first axis, a chunk of a ``split`` along it, a reshape) into a
    buffer of its own. The copy is then an intermediate that an in-place call may update (``InOut``) or write over (``MayAlias``), where ``x``,
    a parameter of the graph, is the caller's and is never written. ``param[...] = x`` into an ``InOut`` parameter is a copy step too.
    A view that is not dense (a permutation, a column block) cannot be copied this way: a plan refuses it (compute from it with an operator).
    A lowered kernel has no copy: in a kernel body a value is already its own.
    """
    x = _require_value("copy", "the tensor", x)
    t = x.type
    if isinstance(t.dtype, BlockScaled):
        raise TilefishTypeError(f"copy: {x.ref} is block-scaled ({t}); block-scaled tensors are not copied yet")
    (out,) = x.ctx.emit("copy", [x], {}, [TensorType(t.shape, t.dtype)])
    return out


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

    A store overwrites the buffer of the version it replaces, so that version is *consumed*, exactly as
    by an in-place call: reading an older version of the parameter afterwards (a ``.live`` taken
    earlier, a no-op cast of the parameter) raises and points to ``.live``, instead of needing a hidden
    copy at lowering time. The name of the parameter follows its newest version. The stored ``value`` is
    only copied and is not consumed (``x[...] = x`` is the one case where it is the replaced version).
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
            hint = "; a strided view cannot be stored (a store copies a contiguous tensor): combine it in an elementwise op first (x + 0 is a copy)"
        elif value.dtype != target.dtype:
            hint = "; stores do not cast implicitly, use .to(dtype)"
        raise TilefishTypeError(f"store into '{target.name}': expected {target.type}, got {value.type}{hint}")
    current = target.live  # the version this store replaces (for an Out parameter, which has no versions: itself)
    versions = [target.type] if target.access == "inout" else []
    outs = target.ctx.emit("store", [current, value], {"target": target.name}, versions)
    if value is not current:  # `x[...] = x` copies a tensor onto itself: a node, but not a write
        target.ctx.written.add(target)
    if outs:
        # the store overwrites the buffer of the version it replaces: that version is consumed, like by an
        # in-place call (a stored value is only copied, not consumed). The name of the parameter follows.
        current._consumed_by = (target.ctx.nodes[-1], target.name)
        current._latest = outs[0]
        target._latest = outs[0]
