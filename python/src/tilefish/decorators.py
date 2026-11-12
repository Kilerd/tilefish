"""Operator declarations: ``@graph``, ``@kernel``, ``@library`` and ``@extern``."""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable

from .dims import DimExpr, as_dim
from .dtypes import DType
from .errors import TilefishDeclarationError, TilefishTypeError
from .ir import Graph
from .signature import MISSING, Param, Signature, call_op, parse_signature
from .trace import Const, TraceContext, Value


class Op:
    """A declared operator: a name, a typed signature, and (for graph/kernel) a traced body.

    Calling it inside a ``graph`` or ``kernel`` body type-checks the call by
    unifying the signature with the argument types and emits one node.
    """

    kind = "op"

    def __init__(self, fn: Callable[..., Any], name: str, signature: Signature) -> None:
        self.fn = fn
        self.name = name
        self.signature = signature
        self.__doc__ = getattr(fn, "__doc__", None)
        self.__wrapped__ = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return call_op(self, args, kwargs)

    def node_attrs(self) -> dict[str, Any]:
        """Extra attributes recorded on every call node."""
        return {}

    def check_const(self, param: Param, value: Any) -> Any:
        """Validate a ``const`` argument supplied at a call site."""
        return value

    def __str__(self) -> str:
        return f"{self.kind} {self.name}{self.signature}"

    def __repr__(self) -> str:
        return f"<{self.kind} {self.name}>"


def _trace_body(fn: Callable[..., Any], signature: Signature, name: str, kind: str) -> Graph:
    ctx = TraceContext(name, kind)
    kwargs: dict[str, Any] = {}
    inputs: list[Value] = []
    consts: dict[str, Any] = {}
    for p in signature.params:
        if p.kind == "tensor":
            assert p.type is not None
            v = ctx.new_value(p.type, name=p.name, access=p.access, is_param=True)
            kwargs[p.name] = v
            inputs.append(v)
        elif p.kind == "const":
            value = None if p.default is MISSING else p.default
            const_param = Const(p.name, value, p.const_dtype)
            kwargs[p.name] = const_param
            consts[p.name] = const_param
        else:
            kwargs[p.name] = p.dim
    result = fn(**kwargs)
    if result is None:
        outputs: tuple[Value, ...] = ()
    elif isinstance(result, Value):
        outputs = (result,)
    elif isinstance(result, (tuple, list)) and all(isinstance(r, Value) for r in result):
        outputs = tuple(result)
    else:
        raise TilefishTypeError(
            f"{kind} '{name}': the body must return a traced tensor, a tuple of them or None, got {result!r}"
        )
    for r in outputs:
        if r.ctx is not ctx:
            raise TilefishTypeError(f"{kind} '{name}': returned value {r.ref} belongs to a different trace")
        if r.access == "out":
            raise TilefishTypeError(f"{kind} '{name}': cannot return Out parameter '{r.name}' (it is write-only)")
    return Graph(name, kind, inputs, consts, ctx.nodes, outputs, ctx.requirements)


def _with_traced_returns(signature: Signature, graph: Graph, kind: str, name: str) -> Signature:
    """Check declared return types against the traced body; adopt the traced ones when none are declared."""
    actual = tuple(v.type for v in graph.outputs)
    declared = signature.returns
    if not declared:
        return Signature(signature.params, actual)
    if len(declared) != len(actual):
        raise TilefishTypeError(
            f"{kind} '{name}': declares {len(declared)} return value(s) but the body returns {len(actual)}"
        )
    for i, (d, a) in enumerate(zip(declared, actual)):
        if d != a:
            raise TilefishTypeError(
                f"{kind} '{name}': return value {i} has type {a} but the declaration says {d}"
            )
    return signature


# -- library --------------------------------------------------------------------------------


class LibraryOp(Op):
    """An operator with an opaque body (CUTLASS, cuBLASLt, ...): a typed signature only."""

    kind = "library"

    def __init__(self, fn: Callable[..., Any], name: str, signature: Signature, acc: DType | None) -> None:
        super().__init__(fn, name, signature)
        self.acc = acc

    def node_attrs(self) -> dict[str, Any]:
        return {"acc": self.acc} if self.acc is not None else {}


def library(name: Any = None, *, acc: DType | None = None) -> Any:
    """``@tilefish.library("gemm", acc=f32)``: declare an operator by signature only."""
    if acc is not None and not isinstance(acc, DType):
        raise TilefishDeclarationError(f"library: acc must be a dtype such as f32, got {acc!r}")

    def make(fn: Callable[..., Any], op_name: str | None) -> LibraryOp:
        nm = op_name or fn.__name__
        sig = parse_signature(fn, what="library op", name=nm)
        if not sig.returns and not sig.tensor_outputs and not any(p.access == "inout" for p in sig.params):
            raise TilefishDeclarationError(
                f"library op '{nm}': declare a return type (-> Tensor[...]) or Out/InOut parameters"
            )
        return LibraryOp(fn, nm, sig, acc)

    if callable(name) and not isinstance(name, str):
        return make(name, None)
    if name is not None and not isinstance(name, str):
        raise TilefishDeclarationError(f"library: the name must be a string, got {name!r}")
    return lambda fn: make(fn, name)


# -- graph and kernel -----------------------------------------------------------------------


class GraphOp(Op):
    """A model: ordinary Python over typed values, traced once at declaration."""

    kind = "graph"

    def __init__(self, fn: Callable[..., Any], name: str, signature: Signature, graph: Graph) -> None:
        super().__init__(fn, name, signature)
        self.graph = graph


class KernelOp(GraphOp):
    """A kernel whose body tilefish understands (traced and type-checked; not lowered yet)."""

    kind = "kernel"


def _declare_traced(fn: Callable[..., Any], name: str | None, cls: type, kind: str) -> Any:
    nm = name or fn.__name__
    sig = parse_signature(fn, what=kind, name=nm, check_derivable=(kind == "kernel"))
    graph = _trace_body(fn, sig, nm, kind)
    sig = _with_traced_returns(sig, graph, kind, nm)
    return cls(fn, nm, sig, graph)


def graph(fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    """``@tilefish.graph``: trace and type-check a model body."""
    if fn is None:
        return lambda f: _declare_traced(f, name, GraphOp, "graph")
    return _declare_traced(fn, name, GraphOp, "graph")


def kernel(fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    """``@tilefish.kernel``: trace and type-check a kernel body (no lowering in M1)."""
    if fn is None:
        return lambda f: _declare_traced(f, name, KernelOp, "kernel")
    return _declare_traced(fn, name, KernelOp, "kernel")


# -- extern -----------------------------------------------------------------------------------

_TRITON_REF = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


def _triton_ref(obj: Any) -> str:
    if isinstance(obj, str):
        if not _TRITON_REF.match(obj):
            raise TilefishDeclarationError(f"extern: triton={obj!r} must look like 'module:function'")
        return obj
    for cand in (obj, getattr(obj, "fn", None)):  # triton.JITFunction keeps the python function in .fn
        module = getattr(cand, "__module__", None)
        fname = getattr(cand, "__name__", None)
        if isinstance(module, str) and isinstance(fname, str):
            return f"{module}:{fname}"
    raise TilefishDeclarationError(
        f"extern: triton= takes a 'module:function' string or a function object, got {obj!r}"
    )


class ExternOp(Op):
    """The typed signature of an existing Triton kernel.

    The stub's parameters, in declaration order, are the Triton kernel's
    parameters: tensors are pointers, a parameter annotated with a runtime
    ``Dim`` is an ``i32`` scalar, a ``const`` parameter is a ``constexpr``.
    """

    kind = "extern"

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str,
        signature: Signature,
        triton: str,
        num_warps: int,
        num_stages: int,
        grid: Any,
    ) -> None:
        super().__init__(fn, name, signature)
        self.triton = triton
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.grid: tuple[DimExpr, ...] | None = self._eval_grid(grid)

    @property
    def triton_module(self) -> str:
        return self.triton.split(":", 1)[0]

    @property
    def triton_function(self) -> str:
        return self.triton.split(":", 1)[1]

    @property
    def grid_string(self) -> str | None:
        """The launch grid as a string expression over dimension names, e.g. ``(cdiv(n, 1024),)``."""
        if self.grid is None:
            return None
        body = ", ".join(str(g) for g in self.grid)
        return f"({body},)" if len(self.grid) == 1 else f"({body})"

    def node_attrs(self) -> dict[str, Any]:
        return {"triton": self.triton}

    def check_const(self, param: Param, value: Any) -> Any:
        if value != param.default:
            raise TilefishTypeError(
                f"{self.name}: constexpr '{param.name}' is fixed to {param.default} by the extern declaration "
                f"(it is compiled into the binary) and cannot be changed to {value!r} at a call"
            )
        return value

    def _grid_names(self) -> dict[str, Any]:
        table: dict[str, Any] = {}
        for p in self.signature.params:
            if p.kind == "const" and p.default is not MISSING:
                table[p.name] = p.default
            elif p.kind == "dim":
                table[p.name] = p.dim
        # dimension names take priority over parameter names
        for p in self.signature.params:
            if p.kind == "tensor":
                for d in p.type.shape:  # type: ignore[union-attr]
                    table.update(d.named_dims())
            elif p.kind == "dim":
                table.update(p.dim.named_dims())  # type: ignore[union-attr]
            elif isinstance(p.default, DimExpr):
                table.update(p.default.named_dims())
        return table

    def _eval_grid(self, grid: Any) -> "tuple[DimExpr, ...] | None":
        if grid is None:
            return None
        if callable(grid):
            table = self._grid_names()
            kwargs = {}
            for pname, gp in inspect.signature(grid).parameters.items():
                if gp.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                    raise TilefishDeclarationError(f"extern '{self.name}': grid callable may only take plain named parameters")
                if pname in table:
                    kwargs[pname] = table[pname]
                elif gp.default is not inspect.Parameter.empty:
                    continue
                else:
                    raise TilefishDeclarationError(
                        f"extern '{self.name}': grid parameter '{pname}' is not a dimension or constant of the "
                        f"signature (available: {', '.join(sorted(table)) or 'none'})"
                    )
            grid = grid(**kwargs)
        if not isinstance(grid, (tuple, list)) or not 1 <= len(grid) <= 3:
            raise TilefishDeclarationError(
                f"extern '{self.name}': grid must be a tuple of 1 to 3 sizes, got {grid!r}"
            )
        try:
            return tuple(as_dim(g) for g in grid)
        except TilefishTypeError:
            raise TilefishDeclarationError(
                f"extern '{self.name}': grid entries must be ints or dimension expressions, got {grid!r}"
            ) from None


def extern(
    *,
    triton: Any,
    num_warps: int = 4,
    num_stages: int = 3,
    grid: Any = None,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], ExternOp]:
    """``@tilefish.extern(triton="module:fn", num_warps=4, grid=...)``: type an existing Triton kernel."""
    ref = _triton_ref(triton)
    for label, val in (("num_warps", num_warps), ("num_stages", num_stages)):
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise TilefishDeclarationError(f"extern: {label} must be a positive int, got {val!r}")

    def deco(fn: Callable[..., Any]) -> ExternOp:
        nm = name or fn.__name__
        sig = parse_signature(fn, what="extern op", name=nm, allow_dim_const_defaults=True)
        if sig.returns:
            raise TilefishDeclarationError(
                f"extern op '{nm}': a Triton kernel returns nothing; declare results as Out[...] parameters"
            )
        for p in sig.consts:
            if p.default is MISSING:
                raise TilefishDeclarationError(
                    f"extern op '{nm}': const parameter '{p.name}' needs a default, the value compiled in as constexpr"
                )
        return ExternOp(fn, nm, sig, ref, num_warps, num_stages, grid)

    return deco
