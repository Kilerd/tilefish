"""Tracing: ``Value`` proxies that record a typed SSA graph as Python runs over them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from .dims import DimExpr
from .dtypes import DType
from .errors import TilefishTypeError
from .ir import Node, Requirement
from .types import AnyDType, TensorType

if TYPE_CHECKING:
    from .ir import Scalar


class TraceContext:
    """Collects the nodes, the SSA numbering and the dimension facts of one trace."""

    def __init__(self, name: str, kind: str) -> None:
        self.name = name
        self.kind = kind
        self.nodes: list[Node] = []
        self.requirements: list[Requirement] = []
        # Divisors established by requirements, keyed by dimension expression.
        self.known: dict[DimExpr, int] = {}
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

    def is_divisible(self, expr: DimExpr, k: int) -> bool:
        return expr.is_divisible_by(k, self.known)

    def require_divisible(self, expr: DimExpr, k: int, origin: str) -> None:
        """Record that ``expr`` must be a multiple of ``k`` and remember it as known."""
        if self.is_divisible(expr, k):
            return
        self.requirements.append(Requirement(expr, k, origin))
        prev = self.known.get(expr, 1)
        self.known[expr] = prev * k // _gcd(prev, k)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


class Const:
    """A ``const`` parameter as seen inside a traced body.

    It combines with tensors and adopts their dtype. ``value`` is the declared
    default (``None`` when there is none).
    """

    __slots__ = ("name", "value", "dtype")

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


class Value:
    """A traced tensor: carries a :class:`TensorType` and records every operation on it."""

    __slots__ = ("ctx", "name", "type", "access", "is_param")

    def __init__(self, ctx: TraceContext, name: str, type: TensorType, access: str, is_param: bool) -> None:
        self.ctx = ctx
        self.name = name
        self.type = type
        self.access = access
        self.is_param = is_param

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
    def producer(self) -> Node | None:
        for n in self.ctx.nodes:
            if any(o is self for o in n.outputs):
                return n
        return None

    def __repr__(self) -> str:
        return f"Value({self.ref}: {self.type})"

    # -- methods ------------------------------------------------------------

    def to(self, dtype: DType) -> "Value":
        """Cast to ``dtype``. A narrowing cast is flagged as a rounding point in the IR."""
        from . import ops

        return ops.cast(self, dtype)

    def split(self, size: Any, axis: int) -> "tuple[Value, ...]":
        """Split ``axis`` into equal chunks of ``size`` (or into chunks of the given sizes)."""
        from . import ops

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
