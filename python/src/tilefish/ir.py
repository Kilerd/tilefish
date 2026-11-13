"""The typed IR: a ``Graph`` of ``Node``s in SSA form."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from .dims import DimExpr
from .dtypes import DType

if TYPE_CHECKING:
    from .trace import Value
    from .types import TensorType


@dataclass(frozen=True)
class Scalar:
    """A scalar operand: a python literal or a ``const`` parameter, already typed.

    ``dtype`` is the dtype the scalar adopted from the tensor it is combined with.
    ``name`` is the ``const`` parameter it came from, if any.
    """

    value: "int | float"
    dtype: DType
    name: str | None = None

    def __str__(self) -> str:
        if self.name is not None:
            return f"{self.name}:{self.dtype}"
        return f"{self.value!r}:{self.dtype}"


@dataclass(frozen=True)
class Requirement:
    """A precondition on a symbolic dimension discovered while type checking.

    It becomes a runtime or build-time check in the host that launches the graph.
    """

    dim: DimExpr
    divisor: int
    origin: str

    def __str__(self) -> str:
        # `%` binds tighter than `+`: a sum or a scaled dimension needs parentheses to read as intended
        dim = str(self.dim) if self.dim.is_atom else f"({self.dim})"
        return f"{dim} % {self.divisor} == 0  (required by {self.origin})"


@dataclass(frozen=True)
class BucketRequirement:
    """A symbolic dimension must be one of the bucket values of a bucketed callee dimension."""

    dim: DimExpr
    buckets: tuple[int, ...]
    origin: str

    def __str__(self) -> str:
        return f"{self.dim} in {list(self.buckets)}  (required by {self.origin})"


def _fmt_attr(v: Any) -> str:
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(_fmt_attr(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt_attr(x)}" for k, x in v.items()) + "}"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


class Node:
    """One operation: ``outputs = op(inputs) {attrs}``.

    ``inputs`` are :class:`~tilefish.trace.Value` or :class:`Scalar`. For a cast,
    ``attrs["rounding"]`` says whether the cast can lose information (a
    *rounding point*); calls to declared operators carry ``attrs["kind"]``
    (``library``/``extern``/``kernel``/``graph``).

    A call's ``outputs`` are the declared return values, then the ``Out``
    parameters, then one new version of each ``InOut`` argument. The call
    *consumes* the version of each ``InOut`` argument it takes as an input and
    replaces it with the new one: after the call, the argument's name (for a
    parameter) reads the new version, and any other value that was consumed
    can no longer be read. The consumed input is the argument's *current*
    version, so consecutive updates form a chain through the inputs and
    outputs of the nodes that write it. Updating a split chunk in place
    produces a new version of the chunk only; the parent is not modified.

    A ``store`` into a parameter has the target's current version as its first
    input and the stored value as its second (``store(%current, %value)``). For
    an ``InOut`` parameter it has one output, the parameter's new version; like
    an in-place call it consumes the version it replaces (the stored value is
    only copied, not consumed); for an ``Out`` parameter, which has no
    versions, it has no output.
    """

    __slots__ = ("op", "inputs", "attrs", "outputs", "callee")

    def __init__(
        self,
        op: str,
        inputs: "Iterable[Value | Scalar]",
        attrs: Mapping[str, Any],
        outputs: "Iterable[Value]",
    ) -> None:
        self.op = op
        self.inputs = tuple(inputs)
        self.attrs = dict(attrs)
        self.outputs = tuple(outputs)
        # The declared operator a call node calls (a kernel, extern, library or graph op), so that a plan can find what to launch;
        # ``None`` for a built-in op (arithmetic, a cast, a view, a store). It is not in ``attrs``: it is not printed and not compared.
        self.callee: Any = None

    @property
    def output_types(self) -> "tuple[TensorType, ...]":
        return tuple(v.type for v in self.outputs)

    @property
    def is_rounding_point(self) -> bool:
        return self.op == "cast" and bool(self.attrs.get("rounding"))

    def __str__(self) -> str:
        args = ", ".join(i.ref if hasattr(i, "ref") else str(i) for i in self.inputs)
        attrs = ""
        if self.attrs:
            attrs = " {" + ", ".join(f"{k}={_fmt_attr(v)}" for k, v in self.attrs.items()) + "}"
        lhs = ", ".join(f"{v.ref}: {v.type}" for v in self.outputs)
        text = f"{lhs} = " if lhs else ""
        text += f"{self.op}({args}){attrs}"
        if self.is_rounding_point:
            text += "  # rounding point"
        return text

    def __repr__(self) -> str:
        return f"Node({self})"


class Graph:
    """A traced, type-checked body: inputs, nodes in program order, and outputs."""

    def __init__(
        self,
        name: str,
        kind: str,
        inputs: "Iterable[Value]",
        consts: Mapping[str, Any],
        nodes: Iterable[Node],
        outputs: "Iterable[Value]",
        requirements: "Iterable[Requirement | BucketRequirement]",
        finals: "Mapping[str, Value] | None" = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.inputs = tuple(inputs)
        self.consts = dict(consts)
        self.nodes = tuple(nodes)
        self.outputs = tuple(outputs)
        self.requirements = tuple(requirements)
        # The last version of every InOut parameter the body modified, by parameter name.
        self.finals: dict[str, "Value"] = dict(finals or {})

    @property
    def symbols(self) -> tuple:
        """The graph's symbolic (dynamic) dimensions, in order of first appearance in the inputs."""
        seen: dict = {}
        for v in self.inputs:
            for d in v.type.shape:
                for free in sorted(d.free_dims(), key=lambda x: x.name):
                    seen.setdefault(free, None)
        return tuple(seen)

    def rounding_points(self) -> tuple[Node, ...]:
        """Every cast that can lose information."""
        return tuple(n for n in self.nodes if n.is_rounding_point)

    def ops(self) -> list[str]:
        return [n.op for n in self.nodes]

    def __str__(self) -> str:
        params = [f"{v.ref}: {v.type}" + (f" [{v.access}]" if v.access != "in" else "") for v in self.inputs]
        params += [
            f"{k}: {'const' if c.dtype is None else f'const[{c.dtype}]'}" + ("" if c.value is None else f" = {c.value!r}")
            for k, c in self.consts.items()
        ]
        outs = ", ".join(str(v.type) for v in self.outputs)
        lines = [f"{self.kind} {self.name}({', '.join(params)}) -> ({outs}):"]
        lines += [f"  {n}" for n in self.nodes]
        lines.append("  return " + ", ".join(v.ref for v in self.outputs))
        for name, v in self.finals.items():
            lines.append(f"  final {name} = {v.ref}")
        for r in self.requirements:
            lines.append(f"  requires {r}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<Graph {self.kind} {self.name}: {len(self.nodes)} nodes>"
