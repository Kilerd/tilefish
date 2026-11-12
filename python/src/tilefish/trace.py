"""Tracing: ``Value`` proxies that record a typed SSA graph as Python runs over them."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from .dims import Dim, DimExpr
from .dtypes import DType
from .errors import TilefishAttributeError, TilefishTypeError
from .ir import BucketRequirement, Node, Requirement
from .types import AnyDType, TensorType

if TYPE_CHECKING:
    from .ir import Scalar


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


class TraceContext:
    """Collects the nodes, the SSA numbering and the dimension facts of one trace."""

    def __init__(self, name: str, kind: str) -> None:
        self.name = name
        self.kind = kind
        self.nodes: list[Node] = []
        self.requirements: list[Requirement | BucketRequirement] = []
        # Divisors established by requirements, keyed by dimension expression.
        self.known: dict[DimExpr, int] = {}
        # Parameters (Out/InOut) that a store or an in-place call has written.
        self.written: set[Value] = set()
        self._next_id = 0

    def new_value(
        self,
        type: TensorType,
        *,
        name: str | None = None,
        access: str = "in",
        is_param: bool = False,
    ) -> "Value":
        if name is None:
            name = str(self._next_id)
            self._next_id += 1
        return Value(self, name, type, access, is_param)

    def emit(
        self,
        op: str,
        inputs: "Iterable[Value | Scalar]",
        attrs: Mapping[str, Any],
        out_types: Sequence[TensorType],
    ) -> "tuple[Value, ...]":
        outs = tuple(self.new_value(t) for t in out_types)
        self.nodes.append(Node(op, inputs, attrs, outs))
        return outs

    # -- facts about dimensions ---------------------------------------------

    def snapshot(self) -> tuple:
        """The dimension facts, to roll back when a call fails half way."""
        return (len(self.requirements), dict(self.known))

    def restore(self, snap: tuple) -> None:
        count, known = snap
        del self.requirements[count:]
        self.known = known

    def is_divisible(self, expr: DimExpr, k: int) -> bool:
        return expr.is_divisible_by(k, self.known)

    def require_divisible(self, expr: DimExpr, k: int, origin: str) -> None:
        """Record that ``expr`` must be a multiple of ``k`` and remember it as known."""
        if self.is_divisible(expr, k):
            return
        self.requirements.append(Requirement(expr, k, origin))
        prev = self.known.get(expr, 1)
        self.known[expr] = prev * k // math.gcd(prev, k)

    def require_bucket(self, expr: DimExpr, buckets: tuple[int, ...], origin: str) -> None:
        """Record that ``expr`` must be one of ``buckets`` (a bucketed callee dimension bound to it)."""
        if isinstance(expr, Dim) and expr.buckets and set(expr.buckets) <= set(buckets):
            return
        req = BucketRequirement(expr, tuple(buckets), origin)
        if not any(isinstance(r, BucketRequirement) and r.dim == expr and r.buckets == req.buckets for r in self.requirements):
            self.requirements.append(req)


class Const:
    """A ``const`` parameter as seen inside a traced body.

    It combines with tensors. An untyped ``const`` adopts the dtype of the
    tensor it meets; a typed one (``const[f32]``) must match it, like a tensor
    would. ``value`` is the declared default (``None`` when there is none).
    """

    __slots__ = ("name", "value", "dtype")
    _tf_operand = True

    def __init__(self, name: str, value: Any, dtype: DType | None) -> None:
        self.name = name
        self.value = value
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"Const({self.name}={self.value!r})"

    def _binop(self, other: Any) -> Any:
        if isinstance(other, Value):
            return NotImplemented  # let the tensor's reflected operator handle it
        raise TilefishTypeError(
            f"const parameter '{self.name}' can only be combined with a tensor value, got {type(other).__name__}"
        )

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _binop


# Operators a traced tensor does not support in M1, with what to do instead.
_UNSUPPORTED = {
    "__neg__": ("negation (-x)", "write 0 - x or x * -1"),
    "__pos__": ("unary plus (+x)", "just use x"),
    "__abs__": ("abs()", "there is no abs op yet"),
    "__pow__": ("exponentiation (**)", "multiply explicitly, or use exp, sqrt or rsqrt"),
    "__rpow__": ("exponentiation (**)", "multiply explicitly, or use exp, sqrt or rsqrt"),
    "__matmul__": ("matrix multiplication (@)", "declare a library op such as gemm and call it"),
    "__rmatmul__": ("matrix multiplication (@)", "declare a library op such as gemm and call it"),
    "__floordiv__": ("floor division (//)", "use / on floating-point tensors"),
    "__rfloordiv__": ("floor division (//)", "use / on floating-point tensors"),
    "__mod__": ("modulo (%)", "there is no modulo op yet"),
    "__rmod__": ("modulo (%)", "there is no modulo op yet"),
    "__lt__": ("comparison (<)", "comparisons and masks are not part of the traced ops yet"),
    "__le__": ("comparison (<=)", "comparisons and masks are not part of the traced ops yet"),
    "__gt__": ("comparison (>)", "comparisons and masks are not part of the traced ops yet"),
    "__ge__": ("comparison (>=)", "comparisons and masks are not part of the traced ops yet"),
    "__and__": ("bitwise and (&)", "there are no bitwise ops yet"),
    "__or__": ("bitwise or (|)", "there are no bitwise ops yet"),
    "__xor__": ("bitwise xor (^)", "there are no bitwise ops yet"),
    "__invert__": ("bitwise not (~)", "there are no bitwise ops yet"),
    "__lshift__": ("shift (<<)", "there are no bitwise ops yet"),
    "__rshift__": ("shift (>>)", "there are no bitwise ops yet"),
    "__len__": ("len()", "use .shape"),
    "__int__": ("int()", "a traced tensor has no value at trace time"),
    "__float__": ("float()", "a traced tensor has no value at trace time"),
    "__index__": ("use as an index or in range()", "a traced tensor has no value at trace time"),
}


class Value:
    """A traced tensor: carries a :class:`TensorType` and records every operation on it.

    After ``param[...] = v`` (or an in-place call), reading ``param`` reads the
    new version, so a body sees what it has stored. ``Out`` parameters are
    write-only.
    """

    __slots__ = ("ctx", "name", "type", "access", "is_param", "_latest")
    _tf_operand = True

    def __init__(self, ctx: TraceContext, name: str, type: TensorType, access: str, is_param: bool) -> None:
        self.ctx = ctx
        self.name = name
        self.type = type
        self.access = access
        self.is_param = is_param
        self._latest: Value | None = None

    # -- inspection ---------------------------------------------------------

    @property
    def ref(self) -> str:
        return f"%{self.name}"

    @property
    def shape(self) -> tuple[DimExpr, ...]:
        return self.type.shape

    @property
    def dtype(self) -> AnyDType:
        return self.type.dtype

    @property
    def rank(self) -> int:
        return self.type.rank

    @property
    def live(self) -> "Value":
        """The newest version of this value (itself unless it was stored into or updated in place)."""
        v = self
        while v._latest is not None:
            v = v._latest
        return v

    @property
    def producer(self) -> Node | None:
        for n in self.ctx.nodes:
            if any(o is self for o in n.outputs):
                return n
        return None

    def __repr__(self) -> str:
        return f"Value({self.ref}: {self.type})"

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):  # protocol probing (copy, pickle, ...) must see a plain AttributeError
            raise AttributeError(name)
        raise TilefishAttributeError(
            f"traced tensor {self.ref} ({self.type}) has no attribute '{name}': in M1 a tensor has "
            ".shape, .dtype, .rank, .to(dtype), .split(size, axis), .sum/.mean/.max(axis, keepdims) "
            "and the operators + - * /"
        )

    # -- methods ------------------------------------------------------------

    def to(self, dtype: Any = _MISSING) -> "Value":
        """Cast to ``dtype``. A narrowing cast is flagged as a rounding point in the IR."""
        from . import ops

        if dtype is _MISSING:
            raise TilefishTypeError("to: needs a dtype, e.g. x.to(bf16)")
        return ops.cast(self, dtype)

    def split(self, size: Any = _MISSING, axis: Any = _MISSING) -> "tuple[Value, ...]":
        """Split ``axis`` into equal chunks of ``size`` (or into chunks of the given sizes)."""
        from . import ops

        if size is _MISSING or axis is _MISSING:
            raise TilefishTypeError("split: needs a size and an axis, e.g. x.split(size, axis=-1)")
        return ops.split(self, size, axis)

    def sum(self, axis: Any = None, keepdims: bool = False) -> "Value":
        from . import ops

        return ops.reduce_sum(self, axis, keepdims)

    def mean(self, axis: Any = None, keepdims: bool = False) -> "Value":
        from . import ops

        return ops.reduce_mean(self, axis, keepdims)

    def max(self, axis: Any = None, keepdims: bool = False) -> "Value":
        from . import ops

        return ops.reduce_max(self, axis, keepdims)

    # -- operators ----------------------------------------------------------

    def __add__(self, other: Any) -> "Value":
        from . import ops

        return ops.binary("add", self, other)

    def __radd__(self, other: Any) -> "Value":
        from . import ops

        return ops.binary("add", other, self)

    def __sub__(self, other: Any) -> "Value":
        from . import ops

        return ops.binary("sub", self, other)

    def __rsub__(self, other: Any) -> "Value":
        from . import ops

        return ops.binary("sub", other, self)

    def __mul__(self, other: Any) -> "Value":
        from . import ops

        return ops.binary("mul", self, other)

    def __rmul__(self, other: Any) -> "Value":
        from . import ops

        return ops.binary("mul", other, self)

    def __truediv__(self, other: Any) -> "Value":
        from . import ops

        return ops.binary("div", self, other)

    def __rtruediv__(self, other: Any) -> "Value":
        from . import ops

        return ops.binary("div", other, self)

    def __setitem__(self, key: Any, value: Any) -> None:
        from . import ops

        ops.store(self, key, value)

    def __getitem__(self, key: Any) -> Any:
        raise TilefishTypeError(f"indexing {self.ref} is not supported (use .split(size, axis) to take pieces)")

    def __iter__(self) -> Any:
        raise TilefishTypeError(
            f"{self.ref} ({self.type}) is a single tensor and cannot be unpacked or iterated; "
            "use .split(size, axis) to take pieces"
        )

    def __bool__(self) -> bool:
        raise TilefishTypeError(f"the truth value of the traced tensor {self.ref} is not defined")

    __hash__ = object.__hash__


def _unsupported_method(dunder: str, what: str, hint: str) -> Any:
    def method(self: Value, *args: Any) -> Any:
        raise TilefishTypeError(f"{what} on tensor {self.ref} ({self.type}) is not supported in M1; {hint}")

    method.__name__ = dunder
    return method


for _dunder, (_what, _hint) in _UNSUPPORTED.items():
    setattr(Value, _dunder, _unsupported_method(_dunder, _what, _hint))
