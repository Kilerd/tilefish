"""Traced built-in operations: elementwise arithmetic, unary math, casts, reductions, views (split, reshape, permute, select), loads of a DevScalar, store.

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
from .dtypes import BlockScaled, DType
from .errors import TilefishError, TilefishTypeError
from .ir import Scalar
from .trace import Const, TraceContext, Value
from .types import TensorType, _mul, product as _product, starts_aligned as _starts_aligned

_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


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
    """Cast to ``dtype``; a narrowing cast (one that can round) is flagged as a rounding point.

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
    if dtype == x.dtype:
        return x._reference()
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
