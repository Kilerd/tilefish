"""Operator declarations: ``@graph``, ``@kernel``, ``@library`` and ``@extern``."""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable

from .dims import DimExpr, as_dim
from ._specvalid import MAX_ELEMENTS
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
    ctx.dims = frozenset(d for v in inputs for s in v.type.shape for d in s.free_dims())
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
        if isinstance(r.dtype, DType) and r.dtype.is_bool:
            raise TilefishTypeError(
                f"{kind} '{name}': the returned value {r.ref} ({r.type}) is a mask, which has no storage type; return it as a "
                "number with .to(u8) (0 or 1) and read it back with a comparison (m != 0)"
            )
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
    """An operator with an opaque body (CUTLASS, cuBLASLt, ...): a typed signature only.

    With ``impl="cublaslt.gemm"`` (or ``"cublaslt.gemm_bias"``) it is also a pinned cuBLASLt GEMM that ``tilefish.run`` runs and
    ``tilefish.runtime.export`` writes to the ``library`` section of a manifest: ``out[M, N] = a[M, K] @ b[N, K]^T`` (``+ bias[N]``), bf16
    with an f32 accumulator and one rounding, the algorithm the heuristic ranks first under a ``workspace_bytes`` limit and a ``heuristic``
    posture (or the one an imported table, ``algos``, gives), recorded with the cuBLASLt version (see :mod:`tilefish._library`). Without
    ``impl`` it is a signature only.
    """

    kind = "library"

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str,
        signature: Signature,
        acc: DType | None,
        impl: "str | None" = None,
        workspace_bytes: "int | None" = None,
        heuristic: str = "tilefish",
        allow_bf16_reduction: bool = False,
        algos: "tuple[Any, ...] | None" = None,
    ) -> None:
        super().__init__(fn, name, signature)
        self.acc = acc
        self.impl = impl
        self.workspace_bytes = workspace_bytes
        self.heuristic = heuristic
        self.allow_bf16_reduction = allow_bf16_reduction
        self.algos = algos

    def node_attrs(self) -> dict[str, Any]:
        return {"acc": self.acc} if self.acc is not None else {}


def library(
    name: Any = None,
    *,
    acc: DType | None = None,
    impl: "str | None" = None,
    workspace_bytes: "int | None" = None,
    heuristic: "str | None" = None,
    allow_bf16_reduction: "bool | None" = None,
    algos: Any = None,
) -> Any:
    """``@tilefish.library("gemm", acc=f32)``: declare an operator by signature only.

    ``impl="cublaslt.gemm"`` (or ``"cublaslt.gemm_bias"``) says a pinned cuBLASLt GEMM runs it: the signature must be
    ``(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16]) -> Tensor[M, N, bf16]`` (with ``bias: Tensor[N, bf16]`` for the second), ``M``, ``N`` and
    ``K`` static or bucketed (of any size: the library judges the leading dimensions an algorithm can take). How its algorithm is chosen is
    part of the declaration:

    * ``heuristic``: the posture cuBLASLt's heuristic is asked under, ``"tilefish"`` (the default: 16-byte alignment minimums) or ``"torch"``
      (what torch's ``addmm`` asks: the workspace limit alone, so the heuristic answers what ``torch.addmm`` / ``torch.mm`` run);
    * ``workspace_bytes``: the workspace limit (default 8 519 680, torch's ``CUBLAS_WORKSPACE_CONFIG``; under ``"torch"`` 1 048 576 for
      ``gemm_bias``, torch's ``CUBLASLT_WORKSPACE_SIZE``);
    * ``allow_bf16_reduction=True``: split-K may add its partial sums in bf16 (torch's ``allow_bf16_reduced_precision_reduction``, its
      default), which rounds more than once; without it they are added in f32 and the result is rounded once;
    * ``algos``: a :class:`~tilefish.AlgoTable` (or a list of them, one per device) whose algorithms are pinned instead of asking the
      heuristic, on the devices they are for.
    """
    if acc is not None and not isinstance(acc, DType):
        raise TilefishDeclarationError(f"library: acc must be a dtype such as f32, got {acc!r}")
    if impl is None and workspace_bytes is not None:
        raise TilefishDeclarationError("library: workspace_bytes needs an impl")
    if impl is None and (heuristic is not None or allow_bf16_reduction is not None or algos is not None):
        raise TilefishDeclarationError("library: heuristic, allow_bf16_reduction and algos need an impl")

    def make(fn: Callable[..., Any], op_name: str | None) -> LibraryOp:
        nm = op_name or fn.__name__
        sig = parse_signature(fn, what="library op", name=nm, allow_devscalar=False)
        if not sig.returns and not sig.tensor_outputs and not any(p.access == "inout" for p in sig.params):
            raise TilefishDeclarationError(
                f"library op '{nm}': declare a return type (-> Tensor[...]) or Out/InOut parameters"
            )
        ws = workspace_bytes
        if impl is None:
            return LibraryOp(fn, nm, sig, acc)
        from . import _library

        posture = "tilefish" if heuristic is None else heuristic
        bf16_reduction = False if allow_bf16_reduction is None else allow_bf16_reduction
        tables = _library.check_options(nm, posture, bf16_reduction, algos)
        if impl not in _library.IMPLS:
            raise TilefishDeclarationError(f"library op '{nm}': impl must be one of {sorted(_library.IMPLS)}, got {impl!r}")
        ws = _library.default_workspace_bytes(impl, posture) if workspace_bytes is None else workspace_bytes
        _library.check_declaration(nm, impl, sig, acc, ws)
        op = LibraryOp(fn, nm, sig, acc, impl, ws, posture, bf16_reduction, tables)
        _library.check_coverage(op)
        return op

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

    ``index_bits`` is how the kernel indexes its tensors: ``32`` (the default, and what every generated kernel does) or
    ``64``. A kernel whose offsets are ``i32`` reads and writes the wrong memory past ``2**31 - 1`` elements, silently, so
    ``@tilefish.extern`` declares ``max_elements`` for every tensor parameter when ``index_bits`` is 32 and a host checks it at
    launch; ``index_bits=64`` is the kernel's word that it indexes with ``i64``, and declares none. It is in the declaration
    digest (as ``max_elements`` is), so a binary built under the other declaration is not the binary of this one.

    Four more optional facts exist for kernels that tilefish generated (``@tilefish.extern`` sets only the contraction, and only
    when it is asked to with ``enable_fp_fusion=``, so the specs of a hand-written kernel say nothing about a warp count per bucket
    or its source):

    * ``max_elements`` maps tensor parameters to the largest element count the kernel can index (its
      offsets are ``i32``); every spec of the op then carries a ``max_elements`` precondition on those
      arguments, which a host checks at launch. ``@tilefish.extern`` sets it for every tensor parameter unless
      ``index_bits=64``; a generated kernel names the tensors whose size no static bound keeps within ``i32``.
    * ``element_counts`` maps runtime scalar parameters to the tensor parameter whose element count (the
      product of its shape) the scalar carries; the spec records how a host computes the value.
    * ``fp_fusion`` (``True`` or ``False``) puts ``enable_fp_fusion`` into the compile options of every
      spec; ``None`` leaves the option out, which is Triton's default (contraction on). A generated kernel always
      has it; a hand-written one has it when ``@tilefish.extern(enable_fp_fusion=...)`` says so.
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
        index_bits: int = 32,
        binary: "BinaryTarget | None" = None,
        reference: "Callable[..., Any] | None" = None,
    ) -> None:
        super().__init__(fn, name, signature)
        if type(index_bits) is not int or index_bits not in (32, 64):
            raise TilefishDeclarationError(f"extern '{name}': index_bits must be 32 or 64, got {index_bits!r}")
        if reference is not None and not callable(reference):
            raise TilefishDeclarationError(f"extern '{name}': reference must be a callable, got {reference!r}")
        # An oracle in torch for ``mode="reference"`` of a graph that calls this op (see ``tilefish.run``): called with the op's tensor inputs
        # as keyword arguments (by parameter name) it returns its ``Out`` tensors, a tensor or a tuple in order. It is not part of the
        # declaration: it is not in a digest, a compile spec or a manifest, and no binary depends on it.
        self.reference: "Callable[..., Any] | None" = reference
        self.index_bits: int = index_bits
        self.binary: "BinaryTarget | None" = binary  # a prebuilt PTX or cubin instead of a Triton kernel (triton is "" then)
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
    def is_binary(self) -> bool:
        """Whether this is a prebuilt PTX or cubin (``extern(ptx=...)`` / ``extern(cubin=...)``) rather than a Triton kernel."""
        return self.binary is not None

    @property
    def grid_string(self) -> str | None:
        """The launch grid as a string expression over dimension names, e.g. ``(cdiv(n, 1024),)``."""
        if self.grid is None:
            return None
        return grid_to_string(self.grid)

    def node_attrs(self) -> dict[str, Any]:
        if self.binary is not None:
            return {"binary": self.binary.symbol}
        return {"triton": self.triton}

    def check_const(self, param: Param, value: Any) -> Any:
        if value != param.default:
            if self.binary is not None:
                raise TilefishTypeError(
                    f"{self.name}: argument '{param.name}' is fixed to {param.default} by the extern declaration "
                    f"(its value is in the manifest) and cannot be changed to {value!r} at a call"
                )
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


_UNSET: Any = object()  # num_warps= / num_stages= not passed (None is a value, and an invalid one)


@dataclass(frozen=True)
class BinaryTarget:
    """A prebuilt kernel that Triton does not compile: PTX that ``ptxas`` assembles for the target architecture, or a cubin per architecture.

    ``sources`` holds the bytes of the file(s) as they were when the operator was declared (the identity of the binary is the sha256 of
    those bytes, ``source_digest`` of kind ``"tilefish.binary.v1"``): ``{"ptx": ...}`` for PTX, ``{"sm_103": ...}`` per architecture
    for cubins. ``symbol`` is the kernel's entry point, ``block`` its thread block and ``shared`` its dynamic shared memory in bytes
    (a prebuilt kernel's ABI says no more than its C parameters, so the declaration says the rest).
    """

    kind: str  # "ptx" | "cubin"
    symbol: str
    block: "tuple[int, int, int]"
    shared: int
    sources: "dict[str, bytes]"

    @property
    def digests(self) -> "dict[str, str]":
        return {k: hashlib.sha256(v).hexdigest() for k, v in sorted(self.sources.items())}


_SYMBOL = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_ARCH_NAME = re.compile(r"sm_[0-9]{2,3}")
_MAX_BINARY = 1 << 26  # 64 MiB: a kernel file is not larger than this


def _read_binary_file(label: str, where: str, path: Any, base: "pathlib.Path | None") -> bytes:
    """The bytes of a PTX or cubin file named in a declaration (a relative path is relative to the file that declares the operator)."""
    if not isinstance(path, (str, os.PathLike)):
        raise TilefishDeclarationError(f"extern: {label} must be a file path, got {path!r}")
    p = pathlib.Path(path)
    if not p.is_absolute() and base is not None:
        p = base / p
    try:
        size = p.stat().st_size
        if size > _MAX_BINARY:
            raise TilefishDeclarationError(f"extern '{where}': {label} {str(p)!r} is {size} bytes, more than the {_MAX_BINARY} a kernel file may have")
        data = p.read_bytes()
    except OSError as e:
        raise TilefishDeclarationError(f"extern '{where}': cannot read {label} {str(p)!r}: {e}") from None
    if not data:
        raise TilefishDeclarationError(f"extern '{where}': {label} {str(p)!r} is empty")
    return data


def _block_of(block: Any) -> "tuple[int, int, int]":
    if not isinstance(block, (tuple, list)) or not 1 <= len(block) <= 3 or any(isinstance(b, bool) or not isinstance(b, int) for b in block):
        raise TilefishDeclarationError(f"extern: block must be a tuple of 1 to 3 ints (threads per block), got {block!r}")
    padded = tuple(block) + (1,) * (3 - len(block))
    if any(not 1 <= b <= 1024 for b in padded) or math.prod(padded) > 1024:
        raise TilefishDeclarationError(f"extern: block {tuple(block)} must have sizes from 1 to 1024 and at most 1024 threads in all")
    return padded  # type: ignore[return-value]


def _binary_target(
    kind: str, files: Any, symbol: Any, block: Any, shared: Any, base: "pathlib.Path | None", where: str
) -> BinaryTarget:
    if not isinstance(symbol, str) or not _SYMBOL.fullmatch(symbol):
        raise TilefishDeclarationError(f"extern: symbol= names the kernel's entry point (a PTX/ELF symbol), got {symbol!r}")
    if isinstance(shared, bool) or not isinstance(shared, int) or not 0 <= shared <= 1 << 30:
        raise TilefishDeclarationError(f"extern: shared must be a non-negative int (bytes of dynamic shared memory), got {shared!r}")
    sources: dict[str, bytes] = {}
    if kind == "ptx":
        sources["ptx"] = _read_binary_file("ptx=", where, files, base)
    else:
        if not isinstance(files, dict) or not files:
            raise TilefishDeclarationError(f"extern: cubin= maps architectures to files, {{'sm_103': 'k.cubin'}}, got {files!r}")
        for arch, path in files.items():
            if not isinstance(arch, str) or not _ARCH_NAME.fullmatch(arch):
                raise TilefishDeclarationError(f"extern: cubin= keys are architectures such as 'sm_103', got {arch!r}")
            sources[arch] = _read_binary_file(f"cubin[{arch!r}]", where, path, base)
    return BinaryTarget(kind, symbol, _block_of(block), shared, sources)


def extern(
    *args: Any,
    triton: Any = None,
    ptx: Any = None,
    cubin: Any = None,
    symbol: "str | None" = None,
    block: Any = None,
    shared: int = 0,
    num_warps: Any = _UNSET,
    num_stages: Any = _UNSET,
    grid: Any = None,
    name: str | None = None,
    index_bits: int = 32,
    reference: "Callable[..., Any] | None" = None,
    enable_fp_fusion: "bool | None" = None,
) -> Callable[[Callable[..., Any]], ExternOp]:
    """``@tilefish.extern(triton="module:fn", num_warps=4, grid=...)``: type an existing Triton kernel.

    It always takes arguments (``@tilefish.extern`` without parentheses is an
    error). ``triton`` is a ``"module:function"`` string or a kernel object;
    ``grid`` is a callable over dimension and constant names, or a tuple.

    **A prebuilt binary that Triton does not compile** is declared with ``ptx="file.ptx"`` or ``cubin={"sm_103": "file.cubin", ...}``
    instead of ``triton=``, together with ``symbol="entry_name"``, ``block=(x, y, z)`` and ``grid=...`` (and ``shared=0``, the bytes of
    dynamic shared memory the kernel addresses)::

        @tilefish.extern(ptx="scale.ptx", symbol="scale_bf16", block=(256,), grid=lambda n: (tilefish.cdiv(n, 256),))
        def scale(x: Tensor[n, bf16], y: Out[Tensor[n, bf16]], n_elem: n, alpha: const[f32] = 2.0): ...

    The stub lists the kernel's **C parameters, in order**: a tensor is a pointer (``Strided[...]`` and ``DevScalar[...]`` too), a
    parameter annotated with a runtime ``Dim`` is an ``i32``, a ``StrideOf`` an ``i32`` (``i64`` for ``index_bits=64``), and a typed
    ``const[i32]``, ``const[i64]`` or ``const[f32]`` with a default is a **fixed argument**: a scalar whose value is written in the
    manifest and passed at every launch (the way a prebuilt kernel takes its constants and its unused scratch slots; there are no
    constexprs, since nothing is compiled). A prebuilt binary has **no Triton scratch slots**: it is launched with exactly these
    parameters. A relative path is relative to the file that declares the operator, and the bytes are read once, there: the identity of
    the binary is their sha256 (``source_digest`` of kind ``tilefish.binary.v1``). PTX is assembled by ``ptxas`` for the target
    architecture with fixed, recorded flags; a cubin is used as it is for the architecture it names and refused for another.

    ``index_bits`` says how the kernel computes the offsets it loads and stores at. The default, ``32``, is the safe
    claim: every tensor parameter gets a ``max_elements`` precondition of ``2**31 - 1`` elements, which Python's ``replay`` and
    a Rust host check at launch, because a kernel that indexes with ``i32`` (``row * N_COLS + col`` on ``tl.int32`` values) goes
    through memory it was not given, silently, on a tensor that is any larger. A kernel that computes its offsets in ``i64``
    (``row.to(tl.int64) * N_COLS``) declares ``index_bits=64`` and has no such limit. The choice is in the declaration digest.

    ``enable_fp_fusion=`` is Triton's compile option of the same name, for a Triton kernel only: ``False`` compiles ``a * b + c`` as a multiply
    and an add, each rounded (no FMA contraction), ``True`` asks for the contraction explicitly (Triton's default, the same machine code as
    leaving it out). ``None``, the default, leaves the option out of the compile options, as before. When it is given it is in every compile
    spec (``enable_fp_fusion``), in the compile options a manifest entry records (``options``) and in the declaration digest, so a binary
    built with the other setting is not the binary of this declaration and ``replay`` refuses it. A prebuilt binary is not compiled by
    Triton and takes no ``enable_fp_fusion=``: its PTX says how each instruction rounds.

    ``reference=`` is an oracle in torch for ``tilefish.run(graph, ..., mode="reference")``, where a graph that calls this op has nothing to
    interpret (a hand-written kernel has no IR): a callable that takes the op's tensor inputs as keyword arguments, by parameter name (a
    ``DevScalar`` as a tensor of one element), and returns its ``Out`` tensors, one tensor or a tuple in order. It is not part of the declaration
    (not in a digest, a compile spec or a manifest), so a binary does not depend on it.

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
    given = [k for k, v in (("triton", triton), ("ptx", ptx), ("cubin", cubin)) if v is not None]
    if args or not given:
        raise TilefishDeclarationError(
            '@tilefish.extern takes arguments: write @tilefish.extern(triton="module:function", grid=...) (or ptx=/cubin= for a prebuilt '
            "binary), not a bare @tilefish.extern"
        )
    if len(given) > 1:
        raise TilefishDeclarationError(f"extern: a kernel is one of triton=, ptx= and cubin=, got {' and '.join(given)}")
    prebuilt = triton is None
    if enable_fp_fusion is not None and not isinstance(enable_fp_fusion, bool):
        raise TilefishDeclarationError(f"extern: enable_fp_fusion must be True, False or None (Triton's default), got {enable_fp_fusion!r}")
    if prebuilt:
        for key, value in (("num_warps", num_warps), ("num_stages", num_stages)):
            if value is not _UNSET:
                raise TilefishDeclarationError(f"extern: {key}= is a Triton compile option; a prebuilt binary has its block= (and nothing is compiled)")
        if enable_fp_fusion is not None:
            raise TilefishDeclarationError(
                "extern: enable_fp_fusion= is a Triton compile option; a prebuilt binary is not compiled by Triton (its PTX says how each "
                "instruction rounds)"
            )
        if symbol is None or block is None:
            raise TilefishDeclarationError(f"extern: {given[0]}= needs symbol=<the kernel's entry point> and block=(x, y, z)")
        target = None
        num_warps, num_stages = 0, 0
    else:
        for key, value in (("symbol", symbol), ("block", block)):
            if value is not None:
                raise TilefishDeclarationError(f"extern: {key}= is for a prebuilt binary (ptx= or cubin=), not for a Triton kernel")
        if shared != 0:
            raise TilefishDeclarationError("extern: shared= is for a prebuilt binary; Triton records the shared memory of the kernel it compiles")
        target = _triton_target(triton)
        num_warps = 4 if num_warps is _UNSET else num_warps
        num_stages = 3 if num_stages is _UNSET else num_stages
        if isinstance(num_warps, bool) or not isinstance(num_warps, int) or num_warps not in _NUM_WARPS:
            raise TilefishDeclarationError(f"extern: num_warps must be a power of two from 1 to 32, got {num_warps!r}")
        if isinstance(num_stages, bool) or not isinstance(num_stages, int) or num_stages < 1:
            raise TilefishDeclarationError(f"extern: num_stages must be a positive int, got {num_stages!r}")

    if type(index_bits) is not int or index_bits not in (32, 64):
        raise TilefishDeclarationError(f"extern: index_bits must be 32 (the kernel indexes with i32) or 64 (with i64), got {index_bits!r}")

    if name is not None and (not isinstance(name, str) or not _OP_NAME.fullmatch(name)):
        raise TilefishDeclarationError(
            f"extern: name must be a plain identifier (ASCII letters, digits and underscores, not starting with a digit): "
            f"it becomes a file name in a manifest directory, so {name!r} is not allowed"
        )

    def deco(fn: Callable[..., Any]) -> ExternOp:
        nm = name or fn.__name__
        if not _OP_NAME.fullmatch(nm):
            raise TilefishDeclarationError(f"extern: the function name {nm!r} is not a plain identifier; pass name=... to give the op one")
        sig = parse_signature(fn, what="extern op", name=nm, allow_dim_const_defaults=not prebuilt, allow_strided=True, allow_output_dims=True)
        if sig.returns:
            raise TilefishDeclarationError(
                f"extern op '{nm}': {'a kernel' if prebuilt else 'a Triton kernel'} returns nothing; declare results as Out[...] parameters"
            )
        for p in sig.consts:
            if p.default is MISSING:
                raise TilefishDeclarationError(
                    f"extern op '{nm}': const parameter '{p.name}' needs a default, "
                    + ("the value written in the manifest and passed at every launch" if prebuilt else "the value compiled in as constexpr")
                )
        binary = None
        if prebuilt:
            for p in sig.consts:  # a fixed argument: a scalar of a C type, with its value
                dt = p.const_dtype
                ok = dt is not None and dt.name in ("i32", "i64", "f32") and isinstance(p.default, (int, float)) and not isinstance(p.default, bool)
                if ok and dt.is_int:  # type: ignore[union-attr]
                    lo, hi = dt.int_range()  # type: ignore[union-attr]
                    ok = isinstance(p.default, int) and lo <= p.default <= hi
                if not ok:
                    raise TilefishDeclarationError(
                        f"extern op '{nm}': const parameter '{p.name}' of a prebuilt binary is an argument of its C ABI: write "
                        f"`{p.name}: const[i32] = 0` (or const[i64], const[f32]) with a default that fits"
                    )
            base = pathlib.Path(fn.__code__.co_filename).resolve().parent if getattr(fn, "__code__", None) is not None else None
            if base is not None and not base.is_dir():
                base = None
            binary = _binary_target("ptx" if ptx is not None else "cubin", ptx if ptx is not None else cubin, symbol, block, shared, base, nm)
        else:
            assert target is not None
            _check_against_kernel(nm, sig, target)
        limits: dict[str, int] = {}
        if index_bits == 32:
            for p in sig.params:
                if p.kind != "tensor" or p.is_devscalar:  # one element: nothing to overflow
                    continue
                if p.strided:  # the limit is on the largest offset the view reaches (a max_extent precondition), known at launch only
                    limits[p.name] = MAX_ELEMENTS
                    continue
                shape = p.type.shape  # type: ignore[union-attr]
                count = math.prod(d.value for d in shape) if all(d.is_static for d in shape) else 0  # 0: not statically known
                if count > MAX_ELEMENTS:
                    raise TilefishDeclarationError(
                        f"extern op '{nm}': parameter '{p.name}' has {count} elements, more than the {MAX_ELEMENTS} that i32 offsets "
                        "reach; declare index_bits=64 if the kernel computes its offsets in i64"
                    )
                limits[p.name] = MAX_ELEMENTS
        return ExternOp(
            fn,
            nm,
            sig,
            target.ref if target is not None else "",
            num_warps,
            num_stages,
            grid,
            max_elements=limits,
            fp_fusion=enable_fp_fusion,
            index_bits=index_bits,
            binary=binary,
            reference=reference,
        )

    return deco
