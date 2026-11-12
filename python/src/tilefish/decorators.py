"""Operator declarations: ``@graph``, ``@kernel``, ``@library`` and ``@extern``."""

from __future__ import annotations

import inspect
import re
import sys
from typing import Any, Callable

from .dims import DimExpr, as_dim
from .dtypes import DType
from .errors import TilefishDeclarationError, TilefishError, TilefishPicklingError, TilefishTypeError
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
        self.__module__ = getattr(fn, "__module__", None)  # type: ignore[assignment]
        self.__qualname__ = getattr(fn, "__qualname__", name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return call_op(self, args, kwargs)

    def run(self, /, *args: Any, mode: str = "jit", **kwargs: Any) -> Any:
        """``tilefish.run(self, *args, mode=mode, **kwargs)``: run this kernel or extern on torch tensors.

        ``mode`` is ``"jit"``, ``"replay"`` or ``"reference"``; see :mod:`tilefish.runtime`.
        """
        from .runtime import run

        return run(self, *args, mode=mode, **kwargs)

    def node_attrs(self) -> dict[str, Any]:
        """Extra attributes recorded on every call node."""
        return {}

    def check_const(self, param: Param, value: Any) -> Any:
        """Validate a ``const`` argument supplied at a call site."""
        return value

    def __copy__(self) -> "Op":
        return self  # a declared operator is immutable

    def __deepcopy__(self, memo: Any) -> "Op":
        return self

    def __reduce__(self) -> Any:
        # Declared operators are module-level objects: pickle them by reference.
        obj: Any = sys.modules.get(self.__module__)
        for part in self.__qualname__.split("."):
            obj = getattr(obj, part, None)
        if "<locals>" in self.__qualname__ or obj is not self:
            raise TilefishPicklingError(
                f"cannot pickle {self.kind} op '{self.name}': operators are pickled by reference, so it must be a "
                f"module-level object reachable as {self.__module__}.{self.__qualname__} "
                "(it is defined inside a function or was rebound)"
            )
        return self.__qualname__

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
    try:
        result = fn(**kwargs)
    except TilefishError as e:
        _name_scope(e, kind, name)
        raise
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
    # a returned InOut parameter is the value stored into it; a value an in-place call consumed cannot be returned
    outputs = tuple(r._current(f"{kind} '{name}': the returned value") for r in outputs)
    if kind == "kernel":
        for v in inputs:
            if v.access in ("out", "inout") and v not in ctx.written:
                label = "Out" if v.access == "out" else "InOut"
                raise TilefishTypeError(
                    f"kernel '{name}': {label} parameter '{v.name}' is never written; assign it with "
                    f"{v.name}[...] = value (or pass it to an in-place operator)"
                )
    finals = {v.name: v.live for v in inputs if v.access == "inout" and v.live is not v}
    return Graph(name, kind, inputs, consts, ctx.nodes, outputs, ctx.requirements, finals)


def _name_scope(e: TilefishError, kind: str, name: str) -> None:
    """Prefix an error raised while tracing a body with the graph or kernel it happened in (once)."""
    if getattr(e, "_tf_scoped", False):
        return
    e._tf_scoped = True  # type: ignore[attr-defined]
    first = str(e.args[0]) if e.args else str(e)
    e.args = (f"in {kind} '{name}': {first}", *e.args[1:])


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
    """A kernel whose body tilefish understands: traced, type-checked, and lowerable to Triton source.

    ``num_warps`` (``None`` for the template's default rule) is the launch warp count the
    lowering uses, for every bucket; ``fp_fusion`` says whether ``a * b + c`` may be contracted
    into one FMA with one rounding (``True``, Triton's default) or must round twice (``False``);
    see :func:`tilefish.lower`.
    """

    kind = "kernel"

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str,
        signature: Signature,
        graph: Graph,
        num_warps: "int | None" = None,
        fp_fusion: bool = True,
    ) -> None:
        super().__init__(fn, name, signature, graph)
        self.num_warps = num_warps
        self.fp_fusion = fp_fusion

    def lower(self, **options: Any) -> Any:
        """Lower this kernel to Triton source: ``tilefish.lower(self, **options)``."""
        from .lower import lower

        return lower(self, **options)


def _declare_traced(fn: Callable[..., Any], name: str | None, cls: type, kind: str, **options: Any) -> Any:
    nm = name or fn.__name__
    sig = parse_signature(fn, what=kind, name=nm, check_derivable=(kind == "kernel"))
    graph = _trace_body(fn, sig, nm, kind)
    sig = _with_traced_returns(sig, graph, kind, nm)
    return cls(fn, nm, sig, graph, **options)


def _check_decorator_arg(deco: str, fn: Any) -> None:
    if fn is not None and not callable(fn):
        raise TilefishDeclarationError(
            f"@tilefish.{deco} decorates a function: write @tilefish.{deco} or @tilefish.{deco}(name=...), got {fn!r}"
        )


def graph(fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    """``@tilefish.graph``: trace and type-check a model body."""
    _check_decorator_arg("graph", fn)
    if fn is None:
        return lambda f: _declare_traced(f, name, GraphOp, "graph")
    return _declare_traced(fn, name, GraphOp, "graph")


def kernel(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    num_warps: int | None = None,
    fp_fusion: bool = True,
) -> Any:
    """``@tilefish.kernel``: trace and type-check a kernel body; ``.lower()`` turns it into Triton source.

    ``num_warps`` pins the warp count of every binary (a power of two from 1 to 32); without it the
    lowering's rule gives each binary its own. ``fp_fusion=False`` compiles without FMA contraction, so that
    ``a * b + c`` rounds the product and then the sum (the default, ``True``, lets the compiler round once).
    """
    _check_decorator_arg("kernel", fn)
    if num_warps is not None and (
        isinstance(num_warps, bool) or not isinstance(num_warps, int) or num_warps not in _NUM_WARPS
    ):
        raise TilefishDeclarationError(f"kernel: num_warps must be a power of two from 1 to 32, got {num_warps!r}")
    if not isinstance(fp_fusion, bool):
        raise TilefishDeclarationError(f"kernel: fp_fusion must be True or False, got {fp_fusion!r}")
    if fn is None:
        return lambda f: _declare_traced(f, name, KernelOp, "kernel", num_warps=num_warps, fp_fusion=fp_fusion)
    return _declare_traced(fn, name, KernelOp, "kernel", num_warps=num_warps, fp_fusion=fp_fusion)


# -- extern -----------------------------------------------------------------------------------

_TRITON_REF = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*\Z")
_OP_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")  # ASCII only: an op name is part of a stem, which is part of a file name
_NUM_WARPS = (1, 2, 4, 8, 16, 32)


class _TritonTarget:
    """What ``extern(triton=...)`` says about the kernel: a reference and, when known, its parameters."""

    def __init__(self, ref: str, arg_names: "list[str] | None" = None, constexpr: "list[bool] | None" = None) -> None:
        self.ref = ref
        self.arg_names = arg_names
        self.constexpr = constexpr


def _triton_target(obj: Any) -> _TritonTarget:
    """Resolve ``triton=``: a ``"module:function"`` string or a kernel object.

    For an object, the innermost ``.fn`` that has a ``__module__`` and a
    ``__name__`` names the kernel (a ``triton.JITFunction`` wraps the Python
    function, an autotuner wraps the JIT function). If the object
    exposes the kernel's parameter list (duck-typed, nothing is imported:
    ``arg_names``, and ``params[i].is_constexpr`` when present) it is returned
    so the stub can be checked against it.
    """
    if isinstance(obj, str):
        if not _TRITON_REF.match(obj):
            raise TilefishDeclarationError(f"extern: triton={obj!r} must look like 'module:function'")
        return _TritonTarget(obj)
    chain = [obj]  # an autotuner wraps the JIT function, which wraps the python function
    while len(chain) < 4 and getattr(chain[-1], "fn", None) is not None:
        chain.append(chain[-1].fn)
    ref = None
    for cand in reversed(chain):  # the innermost function that has a module and a name
        module = getattr(cand, "__module__", None)
        fname = getattr(cand, "__name__", None)
        if isinstance(module, str) and isinstance(fname, str):
            ref = f"{module}:{fname}"
            break
    if ref is None:
        raise TilefishDeclarationError(
            f"extern: triton= takes a 'module:function' string or a kernel object, got {obj!r}"
        )
    # Take the parameter list from wherever it exists in the chain; wrappers that disagree are an error.
    names_at: list[tuple[str, list[str]]] = []
    flags_at: list[tuple[str, list[bool]]] = []
    try:
        for cand in chain:
            label = type(cand).__name__
            names = getattr(cand, "arg_names", None)
            if names is not None:
                names_at.append((label, [str(n) for n in names]))
            params = getattr(cand, "params", None)
            if params is not None and all(hasattr(pr, "is_constexpr") for pr in params):
                flags_at.append((label, [bool(pr.is_constexpr) for pr in params]))
                if all(hasattr(pr, "name") for pr in params):
                    names_at.append((f"{label}.params", [str(pr.name) for pr in params]))
    except TypeError:
        raise TilefishDeclarationError(
            f"extern: the kernel object for '{ref}' has an arg_names or params attribute that is not a list"
        ) from None
    names = _consensus(ref, "parameter names", names_at)
    flags = _consensus(ref, "constexpr flags", flags_at)
    if names is not None and flags is not None and len(names) != len(flags):
        raise TilefishDeclarationError(
            f"extern: the kernel object for '{ref}' lists {len(names)} parameter names but {len(flags)} "
            "constexpr flags"
        )
    return _TritonTarget(ref, names, flags)


def _consensus(ref: str, what: str, found: "list[tuple[str, list]]") -> "list | None":
    if not found:
        return None
    first_label, first = found[0]
    for label, other in found[1:]:
        if other != first:
            raise TilefishDeclarationError(
                f"extern: the wrappers around the Triton kernel '{ref}' disagree about its {what}: "
                f"{first_label} says {first}, {label} says {other}"
            )
    return first


def _check_against_kernel(name: str, sig: Signature, target: _TritonTarget) -> None:
    """The stub must list the Triton kernel's parameters, same names, same order, const <=> constexpr."""
    if target.arg_names is None:
        return
    stub = [p.name for p in sig.params]
    if stub != target.arg_names:
        diff = next((i for i, (a, b) in enumerate(zip(stub, target.arg_names)) if a != b), min(len(stub), len(target.arg_names)))
        raise TilefishDeclarationError(
            f"extern op '{name}': the stub's parameters {stub} do not match the parameters of the Triton kernel "
            f"'{target.ref}' {target.arg_names} (first difference at position {diff}); the stub lists the kernel's "
            "parameters in order"
        )
    if target.constexpr is not None:
        for p, is_constexpr in zip(sig.params, target.constexpr):
            if (p.kind == "const") != is_constexpr:
                raise TilefishDeclarationError(
                    f"extern op '{name}': parameter '{p.name}' is "
                    f"{'a const in the stub' if p.kind == 'const' else 'not a const in the stub'} but "
                    f"{'a constexpr' if is_constexpr else 'not a constexpr'} in the Triton kernel '{target.ref}'"
                )


class ExternOp(Op):
    """The typed signature of an existing Triton kernel.

    The stub's parameters, in declaration order, are the Triton kernel's
    parameters: tensors are pointers, a parameter annotated with a runtime
    ``Dim`` is an ``i32`` scalar, a ``const`` parameter is a ``constexpr``.

    Five optional facts exist for kernels that tilefish generated (``@tilefish.extern`` never sets them,
    so the specs of a hand-written kernel say nothing about its indexing, its contraction, a warp count
    per bucket or its source):

    * ``max_elements`` maps tensor parameters to the largest element count the kernel can index (its
      offsets are ``i32``); every spec of the op then carries a ``max_elements`` precondition on those
      arguments, which a host checks at launch.
    * ``element_counts`` maps runtime scalar parameters to the tensor parameter whose element count (the
      product of its shape) the scalar carries; the spec records how a host computes the value.
    * ``fp_fusion`` (``True`` or ``False``) puts ``enable_fp_fusion`` into the compile options of every
      spec; ``None`` leaves the option out, which is Triton's default (contraction on).
    * ``bucket_num_warps`` maps a bucket's stem suffix (``".D96"``) to its warp count, so that each
      binary of a bucketed op has the warps for its own size; a bucket without an entry has ``num_warps``.
    * ``source_sha256`` is the sha256 of the generated Triton source (deterministic for one IR). It is in the
      declaration digest and in every spec, so a binary built from another body of the same kernel is not the
      binary of this declaration. A hand-written kernel's source is a Python function that is not part of the
      declaration; a manifest records Triton's own digest of it instead (see ``tilefish.aot.build``).
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
        *,
        max_elements: "dict[str, int] | None" = None,
        element_counts: "dict[str, str] | None" = None,
        fp_fusion: "bool | None" = None,
        bucket_num_warps: "dict[str, int] | None" = None,
        source_sha256: "str | None" = None,
    ) -> None:
        super().__init__(fn, name, signature)
        self.triton = triton
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.grid: tuple[DimExpr, ...] | None = self._eval_grid(grid)
        by_name = {p.name: p for p in signature.params}
        self.max_elements: dict[str, int] = dict(max_elements or {})
        for k, v in self.max_elements.items():
            if k not in by_name or by_name[k].kind != "tensor":
                raise TilefishDeclarationError(f"extern '{name}': max_elements: '{k}' is not a tensor parameter")
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise TilefishDeclarationError(f"extern '{name}': max_elements['{k}'] must be a positive int, got {v!r}")
        self.element_counts: dict[str, str] = dict(element_counts or {})
        for k, v in self.element_counts.items():
            if k not in by_name or by_name[k].kind != "dim":
                raise TilefishDeclarationError(f"extern '{name}': element_counts: '{k}' is not a runtime scalar parameter")
            if v not in by_name or by_name[v].kind != "tensor":
                raise TilefishDeclarationError(f"extern '{name}': element_counts['{k}'] must name a tensor parameter, got {v!r}")
        if fp_fusion is not None and not isinstance(fp_fusion, bool):
            raise TilefishDeclarationError(f"extern '{name}': fp_fusion must be True, False or None, got {fp_fusion!r}")
        self.fp_fusion: "bool | None" = fp_fusion
        if source_sha256 is not None and not (isinstance(source_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", source_sha256)):
            raise TilefishDeclarationError(f"extern '{name}': source_sha256 must be 64 lowercase hex digits, got {source_sha256!r}")
        self.source_sha256: "str | None" = source_sha256
        self.bucket_num_warps: dict[str, int] = dict(bucket_num_warps or {})
        for k, v in self.bucket_num_warps.items():
            if not isinstance(k, str) or not re.fullmatch(r"(\.[A-Za-z_][A-Za-z0-9_]*[0-9]+)+", k):
                raise TilefishDeclarationError(
                    f"extern '{name}': bucket_num_warps keys are stem suffixes such as '.D96', got {k!r}"
                )
            if isinstance(v, bool) or not isinstance(v, int) or v not in _NUM_WARPS:
                raise TilefishDeclarationError(
                    f"extern '{name}': bucket_num_warps['{k}'] must be a power of two from 1 to 32, got {v!r}"
                )

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
        return grid_to_string(self.grid)

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
                table[p.name] = p.default.folded() if isinstance(p.default, DimExpr) else p.default
            elif p.kind == "dim":
                table[p.name] = p.dim
        # dimension names take priority over parameter names
        for p in self.signature.params:
            if p.kind == "tensor":
                dims = [d.named_dims() for d in p.type.shape]  # type: ignore[union-attr]
            elif p.kind == "dim":
                dims = [p.dim.named_dims()]  # type: ignore[union-attr]
            elif isinstance(p.default, DimExpr):
                dims = [p.default.named_dims()]
            else:
                continue
            for named in dims:
                table.update({k: v.folded() for k, v in named.items()})  # a static Dim is its value
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
            try:
                grid = grid(**kwargs)
            except TilefishError as e:
                raise TilefishDeclarationError(f"extern '{self.name}': the grid callable failed: {e}") from None
        if not isinstance(grid, (tuple, list)) or not 1 <= len(grid) <= 3:
            raise TilefishDeclarationError(
                f"extern '{self.name}': grid must be a tuple of 1 to 3 sizes, got {grid!r}"
            )
        try:
            entries = tuple(as_dim(g).folded() for g in grid)
        except TilefishTypeError:
            raise TilefishDeclarationError(
                f"extern '{self.name}': grid entries must be ints or dimension expressions, got {grid!r}"
            ) from None
        for g in entries:
            if g.is_static and g.value < 1:
                raise TilefishDeclarationError(f"extern '{self.name}': grid sizes must be at least 1, got {grid!r}")
        return entries


def grid_to_string(grid: "tuple[DimExpr, ...]") -> str:
    """Render a grid as a tuple expression: ``(M,)``, ``(S, 32)``."""
    body = ", ".join(str(g) for g in grid)
    return f"({body},)" if len(grid) == 1 else f"({body})"


def extern(
    *args: Any,
    triton: Any = None,
    num_warps: int = 4,
    num_stages: int = 3,
    grid: Any = None,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], ExternOp]:
    """``@tilefish.extern(triton="module:fn", num_warps=4, grid=...)``: type an existing Triton kernel.

    It always takes arguments (``@tilefish.extern`` without parentheses is an
    error). ``triton`` is a ``"module:function"`` string or a kernel object;
    ``grid`` is a callable over dimension and constant names, or a tuple.

    The decorated stub lists **the Triton kernel's parameters, in order**:
    a ``Tensor``/``InOut``/``Out`` is a pointer, a parameter annotated with a
    runtime ``Dim`` is an ``i32``, a ``const`` is a ``constexpr`` (it needs a
    default: the compiled-in value). Python forbids a parameter without a
    default after one with a default, so when a constexpr comes before other
    parameters, put a bare ``*`` before it::

        @tilefish.extern(triton="my_kernels:k", grid=lambda n: (n,))
        def k(x: Tensor[n, bf16], *, BLOCK: const = 64, y: Out[Tensor[n, bf16]], n_elem: n): ...

    Declaration order is still the kernel's argument order (``x``, ``BLOCK``,
    ``y``, ``n_elem``); arguments after the ``*`` are passed by keyword at call
    sites. When ``triton`` is a kernel object that exposes ``arg_names`` (and
    ``params[i].is_constexpr``), the stub is checked against it.
    """
    if args or triton is None:
        raise TilefishDeclarationError(
            '@tilefish.extern takes arguments: write @tilefish.extern(triton="module:function", grid=...), '
            "not a bare @tilefish.extern"
        )
    target = _triton_target(triton)
    if isinstance(num_warps, bool) or not isinstance(num_warps, int) or num_warps not in _NUM_WARPS:
        raise TilefishDeclarationError(f"extern: num_warps must be a power of two from 1 to 32, got {num_warps!r}")
    if isinstance(num_stages, bool) or not isinstance(num_stages, int) or num_stages < 1:
        raise TilefishDeclarationError(f"extern: num_stages must be a positive int, got {num_stages!r}")

    if name is not None and (not isinstance(name, str) or not _OP_NAME.fullmatch(name)):
        raise TilefishDeclarationError(
            f"extern: name must be a plain identifier (ASCII letters, digits and underscores, not starting with a digit): "
            f"it becomes a file name in a manifest directory, so {name!r} is not allowed"
        )

    def deco(fn: Callable[..., Any]) -> ExternOp:
        nm = name or fn.__name__
        if not _OP_NAME.fullmatch(nm):
            raise TilefishDeclarationError(f"extern: the function name {nm!r} is not a plain identifier; pass name=... to give the op one")
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
        _check_against_kernel(nm, sig, target)
        return ExternOp(fn, nm, sig, target.ref, num_warps, num_stages, grid)

    return deco
