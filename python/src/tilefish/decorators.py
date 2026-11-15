"""Operator declarations: ``@graph``, ``@kernel``, ``@library`` and ``@extern``."""

from __future__ import annotations

import hashlib
import inspect
import itertools
import math
import os
import pathlib
import re
import sys
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from .dims import Dim, DimExpr, as_dim, products
from ._specvalid import IMPLEMENTATION_FALLBACK as FALLBACK, MAX_ELEMENTS, is_implementation_name
from .dtypes import BlockScaled, DType
from .errors import TilefishDeclarationError, TilefishError, TilefishPicklingError, TilefishRuntimeError, TilefishTypeError
from .ir import Graph
from .signature import MISSING, Param, Signature, _check_const_default, call_op, parse_signature
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
    ``tilefish.runtime.export`` writes to the ``library`` section of a manifest: ``out[M, N] = a[M, K] @ b[N, K]^T`` (``b[K, N]`` without the
    transpose for ``layout="NN"`` and ``"NT"``, ``a[K, M]`` transposed for ``"TT"`` and ``"NT"``; ``+ bias[N]``), bf16, f16 or f32 operands with an f32 accumulator and one rounding to the result's type (theirs,
    or f32; f32 operands are multiplied as f32, TF32 off), the algorithm the heuristic ranks first under a ``workspace_bytes`` limit and a ``heuristic`` posture (or the one an imported table,
    ``algos``, gives), recorded with the cuBLASLt version (see :mod:`tilefish._library`). With ``impl="cudnn.conv_fwd"`` it is a pinned cuDNN
    forward convolution (``conv`` holds its options: see :mod:`tilefish._conv`). Without ``impl`` it is a signature only.
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
        layout: str = "TN",
        allow_f16_reduction: bool = False,
        sm_count_target: "int | None" = None,
        allow_f32_reduction: bool = False,
        conv: Any = None,
    ) -> None:
        super().__init__(fn, name, signature)
        self.acc = acc
        self.impl = impl
        self.workspace_bytes = workspace_bytes
        self.heuristic = heuristic
        self.allow_bf16_reduction = allow_bf16_reduction
        self.algos = algos
        self.layout = layout
        self.allow_f16_reduction = allow_f16_reduction
        self.sm_count_target = sm_count_target
        self.allow_f32_reduction = allow_f32_reduction
        self.conv = conv  # tilefish._conv.ConvOptions of a cudnn.conv_fwd, else None

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
    layout: "str | None" = None,
    allow_f16_reduction: "bool | None" = None,
    sm_count_target: "int | None" = None,
    allow_f32_reduction: "bool | None" = None,
    stride: Any = None,
    padding: Any = None,
    dilation: Any = None,
    groups: "int | None" = None,
    math: "str | None" = None,
    api: "str | None" = None,
    alignment: "int | None" = None,
) -> Any:
    """``@tilefish.library("gemm", acc=f32)``: declare an operator by signature only.

    ``impl="cublaslt.gemm"`` (or ``"cublaslt.gemm_bias"``) says a pinned cuBLASLt GEMM runs it: the signature must be
    ``(a: Tensor[M, K, T], b: Tensor[N, K, T]) -> Tensor[M, N, D]`` (with ``bias: Tensor[N, D]`` for the second), where ``T`` is ``bf16``,
    ``f16`` or ``f32`` and ``D`` is ``T`` or ``f32`` (``f32`` for ``f32`` operands); ``M``, ``N`` and ``K`` static or bucketed (of any size: the
    library judges the leading dimensions an algorithm can take). The result is ``a @ b^T`` with an f32 accumulator, rounded once to ``D``; f32
    operands are multiplied as f32, with TF32 off (``CUBLAS_COMPUTE_32F``, and an algorithm that reads its inputs as TF32 or in 16-bit pieces is
    refused). The layout and how its algorithm is chosen are part of the declaration:

    * ``layout``: ``"TN"`` (the default: ``b`` is ``[N, K]``, as a weight of ``torch.nn.Linear``), ``"NN"`` (``b`` is ``Tensor[K, N, T]``, and
      the result is ``a @ b``), ``"TT"`` or ``"NT"`` (those ``b``, and ``a`` is ``Tensor[K, M, T]``: the result is ``a^T @ b^T`` or ``a^T @ b``),
      named by the transposes cuBLASLt is called with (``transa`` for ``b``, ``transb`` for ``a``);
    * ``heuristic``: the posture cuBLASLt's heuristic is asked under, ``"tilefish"`` (the default: 16-byte alignment minimums) or ``"torch"``
      (what torch's ``addmm`` asks: the workspace limit alone, so the heuristic answers what ``torch.addmm`` / ``torch.mm`` run);
    * ``workspace_bytes``: the workspace limit (default 8 519 680, torch's ``CUBLAS_WORKSPACE_CONFIG``; under ``"torch"`` 1 048 576 for
      ``gemm_bias``, torch's ``CUBLASLT_WORKSPACE_SIZE``); a launch provides this many bytes, so it is per op (64 MiB is ``64 << 20``);
    * ``allow_bf16_reduction=True`` (bf16 operands), ``allow_f16_reduction=True`` (f16 operands) or ``allow_f32_reduction=True`` (f32 operands):
      split-K may add its partial sums in the output type (torch's ``allow_bf16_reduced_precision_reduction`` and
      ``allow_fp16_reduced_precision_reduction``, its defaults; torch has no flag for f32 and never restricts the reduction of an f32 GEMM), which
      may round more than once; without it they are added in f32 and the result is rounded once;
    * ``algos``: a :class:`~tilefish.AlgoTable` (or a list of them, one per device) whose algorithms are pinned instead of asking the
      heuristic, on the devices they are for;
    * ``sm_count_target``: the SM count the heuristic chooses for (``CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET``, at least 8 and at most the
      device's), for a GEMM that runs on a partition of the device's SMs (a green context); every check and run of the pin sets it too
      (``docs/manifest-schema-3.md``, section 9.3).

    ``impl="cudnn.conv_fwd"`` says a pinned cuDNN forward convolution runs it (:mod:`tilefish._conv`, ``docs/manifest-schema-3.md``, section 9.4): the
    signature is ``(x, w) -> y`` in the order of ``layout`` (``"NCHW"``, the default: ``x [N, C, *S]``, ``w [K, C/G, *R]``; ``"NHWC"``: ``x [N, *S, C]``,
    ``w [K, *R, C/G]``) with 1 to 3 spatial axes, one type ``f16``, ``bf16`` or ``f32``, and the convolution is ``stride``, ``padding`` (symmetric,
    zeros), ``dilation`` (an int, or one per spatial axis) and ``groups``; ``math`` is ``"tensor_op"`` (the default of 16-bit data) or ``"fma"``, and
    for f32 data ``"fma"`` (TF32 off) or ``"tf32"``, said explicitly; ``api`` is ``"backend"`` (the default: an engine configuration is pinned, chosen as
    torch chooses it, ``heuristic="instant"`` or ``"mode_b"``) or ``"legacy"`` (the algorithm enum of ``cudnnConvolutionForward``, ``heuristic="v7"``);
    ``algos`` imports the pins (:class:`~tilefish.CudnnTable`); ``workspace_bytes`` (default 64 MiB), ``sm_count_target`` and ``alignment`` (16 bytes:
    the alignment of every pointer, ``CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT``) are the rest of the pin.
    """
    if acc is not None and not isinstance(acc, DType):
        raise TilefishDeclarationError(f"library: acc must be a dtype such as f32, got {acc!r}")
    if impl is None and workspace_bytes is not None:
        raise TilefishDeclarationError("library: workspace_bytes needs an impl")
    if impl is None and (
        heuristic is not None
        or allow_bf16_reduction is not None
        or algos is not None
        or layout is not None
        or allow_f16_reduction is not None
        or sm_count_target is not None
        or allow_f32_reduction is not None
    ):
        raise TilefishDeclarationError(
            "library: heuristic, allow_bf16_reduction, allow_f16_reduction, allow_f32_reduction, algos, layout and sm_count_target need an impl"
        )
    if impl is None and any(v is not None for v in (stride, padding, dilation, groups, math, api, alignment)):
        raise TilefishDeclarationError("library: stride, padding, dilation, groups, math, api and alignment need impl='cudnn.conv_fwd'")

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
        from . import _conv, _library

        conv_options = {"stride": stride, "padding": padding, "dilation": dilation, "groups": groups, "math": math, "api": api, "alignment": alignment}
        if impl == _conv.IMPL:
            gemm_only = {
                "allow_bf16_reduction": allow_bf16_reduction, "allow_f16_reduction": allow_f16_reduction, "allow_f32_reduction": allow_f32_reduction,
            }  # fmt: skip
            given = sorted(k for k, v in gemm_only.items() if v is not None)
            if given:
                raise TilefishDeclarationError(f"library op '{nm}': {', '.join(given)} are options of a cuBLASLt GEMM, not of {_conv.IMPL}")
            options = _conv.check_options(
                nm, sig, layout="NCHW" if layout is None else layout, stride=1 if stride is None else stride, padding=0 if padding is None else padding,
                dilation=1 if dilation is None else dilation, groups=1 if groups is None else groups, math=math, api="backend" if api is None else api,
                heuristic=heuristic, alignment=16 if alignment is None else alignment,
            )  # fmt: skip
            tables = None if algos is None else _conv.check_tables(nm, algos, options.api)
            ws = _conv.DEFAULT_WORKSPACE_BYTES if workspace_bytes is None else workspace_bytes
            _conv.check_declaration(nm, sig, acc, ws, options, sm_count_target)
            op = LibraryOp(fn, nm, sig, acc, impl, ws, options.heuristic, False, tables, options.layout, False, sm_count_target, False, options)
            _conv.check_coverage(op)
            return op
        given = sorted(k for k, v in conv_options.items() if v is not None)
        if given:
            raise TilefishDeclarationError(f"library op '{nm}': {', '.join(given)} are options of {_conv.IMPL}, not of a cuBLASLt GEMM")

        posture = "tilefish" if heuristic is None else heuristic
        bf16_reduction = False if allow_bf16_reduction is None else allow_bf16_reduction
        f16_reduction = False if allow_f16_reduction is None else allow_f16_reduction
        f32_reduction = False if allow_f32_reduction is None else allow_f32_reduction
        lay = "TN" if layout is None else layout
        tables = _library.check_options(nm, posture, bf16_reduction, algos, f16_reduction, lay, sm_count_target, f32_reduction)
        if impl not in _library.IMPLS:
            raise TilefishDeclarationError(f"library op '{nm}': impl must be one of {sorted([*_library.IMPLS, _conv.IMPL])}, got {impl!r}")
        ws = _library.default_workspace_bytes(impl, posture) if workspace_bytes is None else workspace_bytes
        _library.check_declaration(nm, impl, sig, acc, ws, lay, bf16_reduction, f16_reduction, f32_reduction)
        op = LibraryOp(fn, nm, sig, acc, impl, ws, posture, bf16_reduction, tables, lay, f16_reduction, sm_count_target, f32_reduction)
        _library.check_coverage(op)
        return op

    if callable(name) and not isinstance(name, str):
        return make(name, None)
    if name is not None and not isinstance(name, str):
        raise TilefishDeclarationError(f"library: the name must be a string, got {name!r}")
    return lambda fn: make(fn, name)


# -- graph and kernel -----------------------------------------------------------------------


class GraphOp(Op):
    """A model: ordinary Python over typed values, traced once at declaration.

    ``output_names`` are the names of its returned values, in order (``@tilefish.graph(outputs=("y", "residual"))``), or ``None``: a launch plan
    names their buffers so (``out0``, ``out1``, ... otherwise), and a host binds the outputs by them.

    ``param_roles`` maps every read-only tensor parameter (not ``InOut``, not ``DevScalar``) to ``"input"`` (given at every call) or ``"weight"``
    (bound once by a host, at load), as ``@tilefish.graph(inputs=..., weights=...)`` declares them, or is ``None``: a launch plan then labels a
    parameter by its shape (``weight`` when it mentions no dimension), which a host cannot rely on.

    ``carried`` are the tensor parameters (``In`` or ``InOut``) whose tensor the plan's arena holds instead of the caller
    (``@tilefish.graph(carried=("h",))``): a run takes the other parameters, and the arena keeps the carried buffer between executions, so one plan
    of a shared arena hands it to the next (``docs/plan-schema-1.md``, section 15.2).

    ``fusion`` is how the arithmetic traced in the body is planned (``@tilefish.graph(fusion=tilefish.fusion(...))``, :mod:`tilefish._fusion`), or
    ``None``: the default, :func:`tilefish.fusion` with no arguments (exact, ``version=1``)."""

    kind = "graph"

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str,
        signature: Signature,
        graph: Graph,
        output_names: "tuple[str, ...] | None" = None,
        param_roles: "dict[str, str] | None" = None,
        carried: "tuple[str, ...]" = (),
        fusion: Any = None,
    ) -> None:
        super().__init__(fn, name, signature)
        self.graph = graph
        self.output_names = output_names
        self.param_roles = param_roles
        self.carried = carried
        self.fusion = fusion


class KernelOp(GraphOp):
    """A kernel whose body tilefish understands: traced, type-checked, and lowerable to Triton source.

    ``num_warps`` (``None`` for the template's default rule) is the launch warp count the
    lowering uses, for every bucket; ``fp_fusion`` says whether ``a * b + c`` may be contracted
    into one FMA with one rounding (``True``, Triton's default) or must round twice (``False``);
    ``math`` is the f32 math of the body (``"fast"``, the default, or ``"ieee"``: note 13 of
    :mod:`tilefish.lower`); see :func:`tilefish.lower`.

    **Pointers that are not 16-byte aligned.** Every pointer of a lowered kernel is compiled as 16-byte aligned (``tt.divisibility=16``,
    and a ``pointer_alignment`` of 16 bytes a launch checks). :meth:`unaligned` is the same kernel compiled for some of its read-only tensor
    parameters aligned to their element only: a *specialisation*, an operator of its own (named ``<kernel>__ua<indices>``, the indices of
    those parameters: ``vq_sum__ua1``), whose binaries have no divisibility hint for those pointers and promise only their element's
    alignment. It is keyed apart from the kernel everywhere a binary is: its stems, its declaration digest (the parameters' ``align``), its
    Triton IR (the hint is in it, so its build cache key) and its manifest entries. The generated source is the kernel's. A graph that
    passes a dense view that does not provably start on 16 bytes (``plane[k]``, a row of a ``[K, R]`` int32 plane with an odd ``R``) to a
    read-only parameter calls the specialisation for it, and ``tilefish.run`` of the kernel on a tensor whose address is not 16-byte aligned
    runs it (``replay`` needs its entries in the manifest: a graph's export has them, and ``export(kernel.unaligned("plane"), ...)`` writes
    them for the kernel alone).
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
        math: str = "fast",
    ) -> None:
        super().__init__(fn, name, signature, graph)
        self.num_warps = num_warps
        self.fp_fusion = fp_fusion
        self.math = math
        # a specialisation for pointers aligned to their element only (unaligned()): the kernel it is of, and the parameters it is for
        self.base: "KernelOp | None" = None
        self.unaligned_params: "frozenset[str]" = frozenset()
        # narrowing float casts inline and a negation NaN-canonical (note 20 of tilefish.lower): set on the kernels of a graph's fused regions
        self.context_free = False

    def lower(self, **options: Any) -> Any:
        """Lower this kernel to Triton source: ``tilefish.lower(self, **options)``."""
        from .lower import lower

        return lower(self, **options)

    def unaligned(self, *names: str) -> "KernelOp":
        """The specialisation of this kernel for the read-only tensor parameters ``names`` aligned to their element only (see the class
        docstring). The same names give the same operator; no name gives the kernel itself."""
        base = self.base if self.base is not None else self
        wanted = frozenset(names) | self.unaligned_params
        if not wanted:
            return base
        by_name = {p.name: p for p in base.signature.params}
        for nm in sorted(wanted):
            p = by_name.get(nm)
            if p is None or p.kind != "tensor" or p.access != "in" or p.is_devscalar or isinstance(p.type.dtype, BlockScaled):  # type: ignore[union-attr]
                raise TilefishDeclarationError(
                    f"kernel '{base.name}': {nm!r} is not a read-only tensor parameter of a plain dtype, so it has no specialisation for a pointer "
                    "aligned to its element only (only an input is a view of another tensor; an Out or InOut tensor is the kernel's own)"
                )
        variants = base.__dict__.setdefault("_tf_unaligned", {})
        found = variants.get(wanted)
        if found is None:
            indices = sorted(by_name[nm].index for nm in wanted)
            found = KernelOp(
                base.fn, f"{base.name}__ua{'_'.join(str(i) for i in indices)}", base.signature, base.graph, base.num_warps, base.fp_fusion, base.math
            )
            found.base, found.unaligned_params = base, wanted
            found.context_free = base.context_free
            found.__doc__ = base.__doc__
            found = variants.setdefault(wanted, found)
        return found

    def __reduce__(self) -> Any:
        if self.base is not None:  # a specialisation: by its kernel (which pickles by reference) and its parameters
            return (_unaligned_of, (self.base, tuple(sorted(self.unaligned_params))))
        return super().__reduce__()


def _unaligned_of(kernel: KernelOp, names: "tuple[str, ...]") -> KernelOp:
    """The specialisation ``kernel.unaligned(*names)`` (how one is unpickled: by its kernel, which pickles by reference)."""
    return kernel.unaligned(*names)


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


_OUTPUT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")  # ASCII: the name of a buffer of a plan


def _role_names(nm: str, key: str, value: Any) -> "tuple[str, ...] | None":
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TilefishDeclarationError(f"graph '{nm}': {key}= is a list of parameter names, got {value!r}")
    names = tuple(value)
    if any(not isinstance(n, str) for n in names):
        raise TilefishDeclarationError(f"graph '{nm}': {key}= is a list of parameter names, got {value!r}")
    if len(set(names)) != len(names):
        raise TilefishDeclarationError(f"graph '{nm}': a name is given twice in {key}={list(names)}")
    return names


def _param_roles(op: GraphOp, inputs: "tuple[str, ...] | None", weights: "tuple[str, ...] | None") -> "dict[str, str] | None":
    """The roles ``@tilefish.graph(inputs=..., weights=...)`` declares: every read-only tensor parameter is an ``input`` (given at every call) or a
    ``weight`` (bound once, at load). The names given must be such parameters; one list alone makes the rest of them the other role, two lists must
    name every one of them once."""
    if inputs is None and weights is None:
        return None
    params = {p.name: p for p in op.signature.params}
    eligible = [p.name for p in op.signature.params if p.kind == "tensor" and p.access == "in" and not p.is_devscalar and p.name not in op.carried]
    for key, names in (("inputs", inputs), ("weights", weights)):
        for n in names or ():
            p = params.get(n)
            if p is None:
                raise TilefishDeclarationError(f"graph '{op.name}': {key}= names '{n}', which is not a parameter (the parameters are {', '.join(params)})")
            if n not in eligible:
                what = (
                    "a DevScalar (a value the host writes at every call)"
                    if p.kind == "tensor" and p.is_devscalar
                    else "carried (the arena holds its tensor)"
                    if n in op.carried
                    else f"an {p.access.capitalize()} parameter (the caller's tensor at every call)"
                    if p.kind == "tensor"
                    else "not a tensor"
                )
                raise TilefishDeclarationError(
                    f"graph '{op.name}': {key}= names '{n}', which is {what}: inputs= and weights= sort the read-only tensor parameters only"
                )
    if inputs is not None and weights is not None:
        both = sorted(set(inputs) & set(weights))
        if both:
            raise TilefishDeclarationError(f"graph '{op.name}': {both} are named both in inputs= and in weights=")
        missing = [n for n in eligible if n not in inputs and n not in weights]
        if missing:
            raise TilefishDeclarationError(
                f"graph '{op.name}': inputs= and weights= together name every read-only tensor parameter; {', '.join(missing)} is in neither "
                "(or give one of the two lists, and the other is the rest)"
            )
    if weights is None:
        return {n: ("input" if n in inputs else "weight") for n in eligible}  # type: ignore[operator]
    return {n: ("weight" if n in weights else "input") for n in eligible}


def _carried_params(op: GraphOp, names: "tuple[str, ...] | None") -> "tuple[str, ...]":
    """The parameters ``@tilefish.graph(carried=...)`` names: tensor parameters, ``In`` or ``InOut``, not a ``DevScalar``, each once."""
    if not names:
        return ()
    params = {p.name: p for p in op.signature.params}
    for n in names:
        p = params.get(n)
        if p is None:
            raise TilefishDeclarationError(f"graph '{op.name}': carried= names '{n}', which is not a parameter (the parameters are {', '.join(params)})")
        if p.kind != "tensor" or p.is_devscalar or p.access not in ("in", "inout"):
            if p.kind != "tensor":
                what = "not a tensor"
            elif p.is_devscalar:
                what = "a DevScalar (a value the host writes at every call)"
            else:
                what = f"an {p.access.capitalize()} parameter"
            raise TilefishDeclarationError(
                f"graph '{op.name}': carried= names '{n}', which is {what}: a carried buffer is a tensor parameter, In (read) or InOut (written), "
                "that the plan's arena holds"
            )
    return tuple(names)


def _declare_graph(
    fn: Callable[..., Any], name: str | None, outputs: Any, inputs: Any = None, weights: Any = None, carried: Any = None, fusion: Any = None
) -> GraphOp:
    from ._fusion import Fusion

    names = None
    nm = name or getattr(fn, "__name__", "?")
    if fusion is not None and not isinstance(fusion, Fusion):
        raise TilefishDeclarationError(f"graph '{nm}': fusion= takes tilefish.fusion(...), got {fusion!r}")
    input_names, weight_names = _role_names(nm, "inputs", inputs), _role_names(nm, "weights", weights)
    carried_names = _role_names(nm, "carried", carried)
    if outputs is not None:
        if isinstance(outputs, (str, bytes)) or not isinstance(outputs, (list, tuple)):
            raise TilefishDeclarationError(f"graph '{nm}': outputs= is a list of the names of the returned values, in order, got {outputs!r}")
        names = tuple(outputs)
        bad = [n for n in names if not isinstance(n, str) or not _OUTPUT_NAME.match(n)]
        if bad:
            raise TilefishDeclarationError(f"graph '{nm}': the output names must be ASCII identifiers (the names of a plan's buffers), got {bad!r}")
        if len(set(names)) != len(names):
            raise TilefishDeclarationError(f"graph '{nm}': an output name is given twice in {list(names)}")
    op = _declare_traced(fn, name, GraphOp, "graph")
    if names is not None:
        returned = len(op.graph.outputs)
        if len(names) != returned:
            raise TilefishDeclarationError(f"graph '{op.name}': outputs= names {len(names)} value(s) ({', '.join(names)}), but the body returns {returned}")
        taken = sorted(set(names) & {p.name for p in op.signature.params})
        if taken:
            raise TilefishDeclarationError(
                f"graph '{op.name}': the output name(s) {taken} are names of parameters: a host binds the tensors of a call by name, so a returned value is "
                "named apart from them"
            )
        op.output_names = names
    op.carried = _carried_params(op, carried_names)
    op.param_roles = _param_roles(op, input_names, weight_names)
    op.fusion = fusion
    return op


def graph(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    outputs: "Sequence[str] | None" = None,
    inputs: "Sequence[str] | None" = None,
    weights: "Sequence[str] | None" = None,
    carried: "Sequence[str] | None" = None,
    fusion: Any = None,
) -> Any:
    """``@tilefish.graph``: trace and type-check a model body.

    ``outputs=("y", "residual")`` names the returned values, in order: a launch plan gives their buffers those names (instead of ``out0``,
    ``out1``, ...), so that a host without Python binds them by name (``docs/plan-schema-1.md``, section 6).

    ``inputs=("x",)`` and/or ``weights=("w1", "w2")`` declare which read-only tensor parameters a call gives every time (the inputs) and which a
    host binds once, when it loads the model (the weights). One list makes the rest of those parameters the other role; two must name every one
    of them once. A launch plan records the roles (``"roles": "declared"``), so a host can bind the weights once per instantiation and only the
    inputs per launch or capture (``tilefish.runtime.bind_weights``; Rust ``PlanRun::bind_weights``). Without either list a plan labels a
    parameter by its shape (``weight`` when it mentions no dimension), as before, and those labels are not a contract.

    ``carried=("h",)`` names tensor parameters (``In`` or ``InOut``) whose tensor the plan's **arena** holds, not the caller: a run takes the other
    parameters, a plan that writes ``h`` leaves it in the arena, and a later execution on the same arena (of this plan, or of another plan of a
    shared arena that names ``h`` too) reads it there: stage A hands an intermediate to stage B without a tensor of the caller's
    (``docs/plan-schema-1.md``, section 15.2). ``reference`` mode, which has no arena, takes it as an argument like any other.

    ``fusion=tilefish.fusion(...)`` says how the arithmetic written in the body (``x * scale + bias``, casts, comparisons, math functions,
    reductions) is planned: grouped into regions, each a kernel tilefish generates, byte for byte the composition of one kernel per op
    (``docs/rfcs/0001-plan-fusion.md``; :mod:`tilefish._fusion`). Without it the default applies, ``tilefish.fusion()``."""
    _check_decorator_arg("graph", fn)
    if fn is None:
        return lambda f: _declare_graph(f, name, outputs, inputs, weights, carried, fusion)
    return _declare_graph(fn, name, outputs, inputs, weights, carried, fusion)


def kernel(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    num_warps: int | None = None,
    fp_fusion: bool = True,
    math: str = "fast",
) -> Any:
    """``@tilefish.kernel``: trace and type-check a kernel body; ``.lower()`` turns it into Triton source.

    ``num_warps`` pins the warp count of every binary (a power of two from 1 to 32); without it the
    lowering's rule gives each binary its own. ``fp_fusion=False`` compiles without FMA contraction, so that
    ``a * b + c`` rounds the product and then the sum (the default, ``True``, lets the compiler round once).
    ``math="ieee"`` makes the f32 math of the body CUDA's without fast math and without flush to zero: ``/`` and
    ``sqrt`` are IEEE 754 (``div.rn.f32``, ``sqrt.rn.f32``), ``exp`` is ``expf``, ``silu`` is ``x / (1 + expf(-x))``
    (torch's), ``rsqrt`` is ``rsqrtf``, and ``sin`` and ``tanh`` keep a subnormal (note 13 of :mod:`tilefish.lower`);
    the default, ``"fast"``, is the contract of before. One op can say it alone: ``tilefish.exp(x, math="ieee")``.
    """
    _check_decorator_arg("kernel", fn)
    if num_warps is not None and (
        isinstance(num_warps, bool) or not isinstance(num_warps, int) or num_warps not in _NUM_WARPS
    ):
        raise TilefishDeclarationError(f"kernel: num_warps must be a power of two from 1 to 32, got {num_warps!r}")
    if not isinstance(fp_fusion, bool):
        raise TilefishDeclarationError(f"kernel: fp_fusion must be True or False, got {fp_fusion!r}")
    if math not in _MATH_MODES:
        raise TilefishDeclarationError(f"kernel: math must be one of {list(_MATH_MODES)}, got {math!r}")
    options = {"num_warps": num_warps, "fp_fusion": fp_fusion, "math": math}
    if fn is None:
        return lambda f: _declare_traced(f, name, KernelOp, "kernel", **options)
    return _declare_traced(fn, name, KernelOp, "kernel", **options)


# -- extern -----------------------------------------------------------------------------------

_TRITON_REF = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*\Z")
LANGUAGES = ("triton", "gluon")  # the frontends of Triton a hand-written kernel is compiled by: triton.jit, and Gluon's gluon.jit
_OP_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")  # ASCII only: an op name is part of a stem, which is part of a file name
_NUM_WARPS = (1, 2, 4, 8, 16, 32)
_MATH_MODES = ("fast", "ieee")  # the f32 math of a kernel body (note 13 of tilefish.lower)
_NUM_CTAS = (1, 2, 4, 8)  # Triton's num_ctas: the CTAs of one program, a thread-block cluster along x (8: the portable cluster size)
MAX_CLUSTER = 8  # the CTAs of a cluster a launch asks for without CUDA's non-portable cluster size
MAX_PROGRAMS_PER_SM = 32  # programs_per_sm of a persistent grid: at most the resident blocks an SM has
MAX_GLOBAL_SCRATCH = 1 << 20  # bytes of global scratch per CTA a declaration may state


@dataclass(frozen=True)
class Persistent:
    """A **persistent grid**, made by :func:`persistent`: the host launches ``min(tiles, programs_per_sm * SMs // ctas)`` programs, where
    ``tiles`` is the declared grid (one entry: the work items), ``SMs`` the multiprocessor count of the device (or of the partition the host runs
    on) and ``ctas`` the CTAs of one program (a cluster's, else 1), resolved when the binary is loaded. The kernel walks the tiles with a stride of
    the programs launched (``for t in range(tl.program_id(0), tiles, tl.num_programs(0))``), so one binary serves every device shape.

    ``sm_count=None`` (the default) is the kernel's promise that **its bytes do not depend on how many programs run**: each tile is computed by one
    program from its inputs alone, whichever program and however many there are. A kernel whose bytes depend on it (a persistent reduction whose
    programs each sum their own tiles, then combine) states the SM count it is declared for (``sm_count=30``): its entry records it, and a host on a
    device or partition of another count refuses it (declare another op, or another value, for another device shape)."""

    programs_per_sm: int = 1
    sm_count: "int | None" = None

    def to_json(self) -> "dict[str, int]":
        """The manifest's ``persistent`` object (manifest schema 3, section 4.3)."""
        out = {"programs_per_sm": self.programs_per_sm}
        if self.sm_count is not None:
            out["sm_count"] = self.sm_count
        return out


def persistent(*, programs_per_sm: int = 1, sm_count: "int | None" = None) -> Persistent:
    """A persistent grid for ``@tilefish.extern(..., grid=lambda M, N: (tilefish.cdiv(M, 128) * tilefish.cdiv(N, 128),), persistent=tilefish.persistent())``:
    the grid is the number of tiles, and the host launches at most ``programs_per_sm`` programs per SM of the device it loads the binary on (see
    :class:`Persistent`; ``sm_count=`` for a kernel whose bytes depend on the number of programs)."""
    if isinstance(programs_per_sm, bool) or not isinstance(programs_per_sm, int) or not 1 <= programs_per_sm <= MAX_PROGRAMS_PER_SM:
        raise TilefishDeclarationError(f"tilefish.persistent: programs_per_sm must be an int from 1 to {MAX_PROGRAMS_PER_SM}, got {programs_per_sm!r}")
    if sm_count is not None and (isinstance(sm_count, bool) or not isinstance(sm_count, int) or not 1 <= sm_count <= 1 << 20):
        raise TilefishDeclarationError(
            f"tilefish.persistent: sm_count is None (the kernel's bytes do not depend on how many programs run) or the SM count the kernel is declared "
            f"for (an int from 1), got {sm_count!r}"
        )
    return Persistent(programs_per_sm, sm_count)


@dataclass(frozen=True)
class Config:
    """The compile-time configuration of one bucket of an extern, made by :func:`config`: ``None`` keeps the declaration's ``num_warps``,
    ``num_stages`` or ``num_ctas``, and ``consts`` gives const parameters of the stub the values compiled into this bucket's binary."""

    num_warps: "int | None" = None
    num_stages: "int | None" = None
    num_ctas: "int | None" = None
    consts: "tuple[tuple[str, Any], ...]" = ()


def config(*, num_warps: "int | None" = None, num_stages: "int | None" = None, num_ctas: "int | None" = None, **consts: Any) -> Config:
    """One bucket's configuration for ``@tilefish.extern(..., by_bucket=lambda T: tilefish.config(num_warps=8, BLOCK_T=128) if T > 256 else
    tilefish.config(num_warps=2, BLOCK_T=32))``: the warps, stages and CTAs of the binary of that bucket, and the values of const parameters
    (tile sizes, any constexpr of the kernel) compiled into it. What it leaves out is the declaration's."""
    if num_warps is not None and (isinstance(num_warps, bool) or not isinstance(num_warps, int) or num_warps not in _NUM_WARPS):
        raise TilefishDeclarationError(f"tilefish.config: num_warps must be a power of two from 1 to 32, got {num_warps!r}")
    if num_stages is not None and (isinstance(num_stages, bool) or not isinstance(num_stages, int) or num_stages < 1):
        raise TilefishDeclarationError(f"tilefish.config: num_stages must be a positive int, got {num_stages!r}")
    if num_ctas is not None and (isinstance(num_ctas, bool) or not isinstance(num_ctas, int) or num_ctas not in _NUM_CTAS):
        raise TilefishDeclarationError(f"tilefish.config: num_ctas must be one of {list(_NUM_CTAS)}, got {num_ctas!r}")
    return Config(num_warps, num_stages, num_ctas, tuple(sorted(consts.items())))


class _TritonTarget:
    """What ``extern(triton=...)`` or ``extern(gluon=...)`` says about the kernel: a reference and, when known, its parameters and whether it is
    a Gluon kernel (``None``: the object does not say, or the reference is a string)."""

    def __init__(
        self, ref: str, arg_names: "list[str] | None" = None, constexpr: "list[bool] | None" = None, is_gluon: "bool | None" = None
    ) -> None:
        self.ref = ref
        self.arg_names = arg_names
        self.constexpr = constexpr
        self.is_gluon = is_gluon


def _is_gluon(chain: "list[Any]") -> "bool | None":
    """Whether the kernel in ``chain`` (an object and the ``.fn`` it wraps, outermost first) is a Gluon kernel: what the first object that has
    Triton's ``is_gluon()`` says (a ``triton.JITFunction`` has it, ``False``; a Gluon ``GluonJITFunction``, ``True``), or ``None``."""
    for cand in chain:
        method = getattr(cand, "is_gluon", None)
        if callable(method):
            try:
                answer = method()
            except Exception:  # noqa: BLE001 - an object that only looks like a kernel: it says nothing
                return None
            return answer if isinstance(answer, bool) else None
    return None


def _triton_target(obj: Any, language: str = "triton") -> _TritonTarget:
    """Resolve ``triton=`` (or ``gluon=``, ``language="gluon"``): a ``"module:function"`` string or a kernel object.

    For an object, the innermost ``.fn`` that has a ``__module__`` and a
    ``__name__`` names the kernel (a ``triton.JITFunction`` wraps the Python
    function, an autotuner wraps the JIT function). If the object
    exposes the kernel's parameter list (duck-typed, nothing is imported:
    ``arg_names``, and ``params[i].is_constexpr`` when present) it is returned
    so the stub can be checked against it, and so is what its ``is_gluon()`` says (``triton.JITFunction.is_gluon``).
    """
    if isinstance(obj, str):
        if not _TRITON_REF.match(obj):
            raise TilefishDeclarationError(f"extern: {language}={obj!r} must look like 'module:function'")
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
            f"extern: {language}= takes a 'module:function' string or a kernel object, got {obj!r}"
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
    gluon = _is_gluon(chain)
    if language == "triton" and gluon is True:
        raise TilefishDeclarationError(
            f"extern: '{ref}' is a Gluon kernel (@gluon.jit, triton.experimental.gluon): declare it with gluon=..., not triton=. Its "
            "identity and its compile go through Gluon's frontend (the layouts are in its IR), which triton= does not use"
        )
    if language == "gluon" and gluon is False:
        raise TilefishDeclarationError(
            f"extern: gluon= takes a Gluon kernel (@gluon.jit, triton.experimental.gluon), but '{ref}' is a triton.jit kernel: declare it "
            "with triton=..."
        )
    return _TritonTarget(ref, names, flags, gluon)


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

    ``language`` is the frontend that compiles the kernel: ``"triton"`` (a ``triton.jit`` function, ``@tilefish.extern(triton=...)``) or
    ``"gluon"`` (a Gluon ``gluon.jit`` function, ``@tilefish.extern(gluon=...)``, whose layouts are explicit). It is in the declaration
    digest and in every compile spec, and it decides which IR identifies the binary (:mod:`tilefish._ir`).

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
        language: str = "triton",
        implementation: "str | None" = None,
        implementations: "dict[str, ExternOp] | None" = None,
        selection: "dict[str, str] | None" = None,
        fallback_key: "str | None" = None,
        persistent: "Persistent | None" = None,
        num_ctas: int = 1,
        global_scratch: int = 0,
        by_bucket: "Callable[..., Any] | None" = None,
    ) -> None:
        super().__init__(fn, name, signature)
        # extern(by_arch=...): the op that a graph calls and a caller runs has ``implementations`` (its key -> an ExternOp of its own: the
        # architecture's, "fallback", or the name of a named implementation), and each of those has ``implementation``, its key, which is in its
        # declaration digest and its specs. ``selection`` maps each architecture of the map to the key of its implementation and ``fallback_key``
        # is the key of the fallback (None: no fallback)
        self.implementation: "str | None" = implementation
        self.implementations: "dict[str, ExternOp] | None" = implementations
        if implementations is not None and selection is None:
            selection = {k: k for k in implementations if k != FALLBACK}
            fallback_key = FALLBACK if FALLBACK in implementations else None
        self.selection: "dict[str, str] | None" = selection
        self.fallback_key: "str | None" = fallback_key
        if type(index_bits) is not int or index_bits not in (32, 64):
            raise TilefishDeclarationError(f"extern '{name}': index_bits must be 32 or 64, got {index_bits!r}")
        if language not in LANGUAGES:
            raise TilefishDeclarationError(f"extern '{name}': language must be one of {', '.join(LANGUAGES)}, got {language!r}")
        if binary is not None and language != "triton":
            raise TilefishDeclarationError(f"extern '{name}': a prebuilt binary is not compiled by a frontend, so it has no language {language!r}")
        # the frontend that compiles the kernel: "triton" (triton.jit) or "gluon" (gluon.jit: explicit layouts). Not a fact of a prebuilt binary
        self.language: str = language
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
        self._grid_source = grid  # the grid as declared (a callable is evaluated again for a bucket whose by_bucket= config gives its consts other values)
        self.grid: tuple[DimExpr, ...] | None = self._eval_grid(grid)
        if persistent is not None and not isinstance(persistent, Persistent):
            raise TilefishDeclarationError(f"extern '{name}': persistent= is a tilefish.persistent(...) or None, got {persistent!r}")
        # a persistent grid (tilefish.persistent): the grid is the number of tiles, and the host launches at most programs_per_sm per SM of the device
        self.persistent: "Persistent | None" = persistent
        if isinstance(num_ctas, bool) or not isinstance(num_ctas, int) or num_ctas not in _NUM_CTAS:
            raise TilefishDeclarationError(f"extern '{name}': num_ctas must be one of {list(_NUM_CTAS)}, got {num_ctas!r}")
        if num_ctas != 1 and (binary is not None or not triton):
            raise TilefishDeclarationError(f"extern '{name}': num_ctas= is a Triton compile option; a prebuilt binary declares its cluster= instead")
        # Triton's num_ctas: the CTAs of one program, launched as a thread-block cluster of (num_ctas, 1, 1) (cuLaunchKernelEx)
        self.num_ctas: int = num_ctas
        if isinstance(global_scratch, bool) or not isinstance(global_scratch, int) or not 0 <= global_scratch <= MAX_GLOBAL_SCRATCH:
            raise TilefishDeclarationError(f"extern '{name}': global_scratch must be an int from 0 to {MAX_GLOBAL_SCRATCH} (bytes per CTA), got {global_scratch!r}")
        if global_scratch and (binary is not None or not triton):
            raise TilefishDeclarationError(f"extern '{name}': global_scratch= is the global scratch of a Triton kernel (its launch ABI has the slot); a prebuilt binary has none")
        # the bytes of global scratch each CTA of the compiled kernel addresses (Triton's metadata.global_scratch_size, checked by the build): a host
        # gives a launch global_scratch * (CTAs of the grid) bytes, a plan from its arena
        self.global_scratch: int = global_scratch
        # by_bucket=: one compile-time configuration per bucket combination of the dimensions it names (_bucket_configs), and the grid of each
        self.by_bucket_dims: "tuple[Dim, ...]" = ()
        self.bucket_configs: "dict[tuple[tuple[str, int], ...], dict[str, Any]]" = {}
        if by_bucket is not None:
            self._bucket_configs(by_bucket)
        if self.persistent is not None:
            for g in [self.grid, *(c["grid"] for c in self.bucket_configs.values())]:
                if g is not None and len(g) != 1:
                    raise TilefishDeclarationError(
                        f"extern '{name}': a persistent grid has one entry, the number of tiles (the host launches at most programs_per_sm programs per "
                        f"SM along x), got {grid_to_string(g)}"
                    )
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
        """Whether this is a prebuilt PTX or cubin (``extern(ptx=...)`` / ``extern(cubin=...)``) rather than a Triton kernel; for an op declared with
        ``by_arch=``, whether every implementation is one (then its calls give the values of its fixed arguments per call)."""
        if self.implementations is not None:
            return all(one.binary is not None for one in self.implementations.values())
        return self.binary is not None

    @property
    def grid_string(self) -> str | None:
        """The launch grid as a string expression over dimension names, e.g. ``(cdiv(n, 1024),)``."""
        if self.grid is None:
            return None
        return grid_to_string(self.grid)

    def for_arch(self, arch: "int | str") -> "ExternOp":
        """The implementation this op runs on ``arch`` (``103``, ``"sm_103"``): for an op declared with ``by_arch=``, the one declared for that
        architecture (a named one may be declared for several), else its ``fallback``, else a :class:`~tilefish.errors.TilefishRuntimeError`; any
        other op is its own."""
        if self.implementations is None:
            return self
        from ._build import parse_arch

        name = f"sm_{parse_arch(arch)}"
        assert self.selection is not None
        key = self.selection.get(name, self.fallback_key)
        if key is None:
            archs = ", ".join(self.selection)
            raise TilefishRuntimeError(
                f"extern op '{self.name}' has no implementation for {name}: it declares one for {archs} and no fallback. Declare one for "
                f"{name} in by_arch= (or a fallback=) once it is known to be right there"
            )
        return self.implementations[key]

    def node_attrs(self) -> dict[str, Any]:
        if self.implementations is not None:
            return {"by_arch": {k: v.triton or f"{v.binary.kind}:{v.binary.symbol}" for k, v in self.implementations.items()}}  # type: ignore[union-attr]
        if self.binary is not None:
            return {"binary": self.binary.symbol}
        return {self.language: self.triton}

    def __reduce__(self) -> Any:
        if self.implementation is not None and self.__dict__.get("_tf_of") is not None:  # one implementation: by its op and its key
            return (_implementation_of, (self.__dict__["_tf_of"], self.implementation))
        return super().__reduce__()

    def check_const(self, param: Param, value: Any) -> Any:
        if any(param.name in c["consts"] for c in self.bucket_configs.values()):
            raise TilefishTypeError(
                f"{self.name}: constexpr '{param.name}' is compiled into each bucket's binary with the value by_bucket= chooses for it, so a call "
                "does not give it"
            )
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

    def _bucket_configs(self, by_bucket: Any) -> None:
        """Evaluate ``by_bucket=`` (a callable over names of bucketed dimensions of the signature, returning :func:`config` or ``None``) at every
        bucket combination of the dimensions it names, and keep each combination's configuration, made whole with the declaration's values (warps,
        stages, CTAs, the value of every const it sets, and the grid under those values)."""
        nm = self.name
        if self.binary is not None or not self.triton:
            raise TilefishDeclarationError(
                f"extern '{nm}': by_bucket= chooses the compile-time configuration of a Triton or Gluon kernel per bucket; a prebuilt binary is compiled "
                "already (declare one op per binary)"
            )
        if not callable(by_bucket):
            raise TilefishDeclarationError(
                f"extern '{nm}': by_bucket= is a callable over bucketed dimension names that returns a tilefish.config(...) (or None), such as "
                f"lambda T: tilefish.config(num_warps=8) if T > 256 else None; got {by_bucket!r}"
            )
        bucketed: "dict[str, Dim]" = {}
        for p in self.signature.params:
            exprs = list(p.type.shape) if p.kind == "tensor" else [p.dim] if p.kind == "dim" else []  # type: ignore[union-attr]
            if p.kind == "const" and isinstance(p.default, DimExpr):
                exprs.append(p.default)
            for e in exprs:
                for d in e.free_dims():  # type: ignore[union-attr]
                    if d.buckets:
                        bucketed.setdefault(d.name, d)
        names: "list[str]" = []
        for pname, gp in inspect.signature(by_bucket).parameters.items():
            if gp.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                raise TilefishDeclarationError(f"extern '{nm}': the by_bucket= callable may only take plain named parameters")
            if pname not in bucketed:
                raise TilefishDeclarationError(
                    f"extern '{nm}': by_bucket= takes '{pname}', which is not a bucketed dimension of the signature (bucketed: "
                    f"{', '.join(sorted(bucketed)) or 'none'}); a configuration is chosen per bucket of a Dim(..., buckets=[...])"
                )
            names.append(pname)
        if not names:
            raise TilefishDeclarationError(f"extern '{nm}': the by_bucket= callable names no bucketed dimension; it takes the ones it chooses by")
        consts = {p.name: p for p in self.signature.consts}
        dims = tuple(bucketed[n] for n in names)
        for combo in itertools.product(*(d.buckets for d in dims)):  # type: ignore[misc]
            values = dict(zip(names, combo))
            try:
                got = by_bucket(**values)
            except TilefishError as e:
                raise TilefishDeclarationError(f"extern '{nm}': by_bucket= failed at {values}: {e}") from None
            if got is None:
                got = Config()
            if not isinstance(got, Config):
                raise TilefishDeclarationError(f"extern '{nm}': by_bucket= returns a tilefish.config(...) or None, got {got!r} at {values}")
            chosen: "dict[str, Any]" = {}
            for cname, value in got.consts:
                p = consts.get(cname)
                if p is None:
                    raise TilefishDeclarationError(
                        f"extern '{nm}': by_bucket= sets '{cname}' at {values}, which is not a const parameter of the stub ({sorted(consts) or 'none'})"
                    )
                chosen[cname] = _check_const_default(f"extern '{nm}' (by_bucket= at {values})", cname, p.const_dtype, value, allow_dim=False)
            try:
                bucket_grid = self._eval_grid(self._grid_source, chosen) if chosen else self.grid
            except TilefishDeclarationError as e:
                raise TilefishDeclarationError(f"{e} (the grid under the consts by_bucket= gives at {values})") from None
            num_ctas = self.num_ctas if got.num_ctas is None else got.num_ctas
            self.bucket_configs[tuple(sorted(values.items()))] = {
                "num_warps": self.num_warps if got.num_warps is None else got.num_warps,
                "num_stages": self.num_stages if got.num_stages is None else got.num_stages,
                "num_ctas": num_ctas,
                "consts": chosen,
                "grid": bucket_grid,
            }
        self.by_bucket_dims = dims

    def bucket_config(self, buckets: "Mapping[str, int]") -> "dict[str, Any] | None":
        """The ``by_bucket=`` configuration of the binary of these bucket values (by dimension name; ``None``: the op has no ``by_bucket=``)."""
        if not self.by_bucket_dims:
            return None
        return self.bucket_configs[tuple(sorted((d.name, buckets[d.name]) for d in self.by_bucket_dims))]

    def grid_for(self, buckets: "Mapping[str, int]") -> "tuple[DimExpr, ...] | None":
        """The launch grid of the binary of these bucket values: the declared one, or the one ``by_bucket=`` gives it (its consts' values)."""
        chosen = self.bucket_config(buckets)
        return self.grid if chosen is None else chosen["grid"]

    def _grid_names(self, overrides: "Mapping[str, Any] | None" = None) -> dict[str, Any]:
        table: dict[str, Any] = {}
        for p in self.signature.params:
            if p.kind == "const" and overrides and p.name in overrides:
                table[p.name] = overrides[p.name]
            elif p.kind == "const" and p.default is not MISSING:
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

    def _eval_grid(self, grid: Any, overrides: "Mapping[str, Any] | None" = None) -> "tuple[DimExpr, ...] | None":
        if grid is None:
            return None
        if callable(grid):
            table = self._grid_names(overrides)
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
                with products():  # a grid may multiply two dimensions (cdiv(T * C, 1024)): a host computes it, it never solves it
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


# -- per-architecture implementations: extern(by_arch=...) ------------------------------------------------------------------

# FALLBACK ("fallback") is the key of the implementation an op declared with by_arch= runs on an architecture the map does not name (a named
# implementation's key is its name, whichever architectures it is declared for).
#
# What every implementation of one op has in common is its public signature, what a caller, a plan and a typed host launcher see: the tensors (by
# name, in order, with their access, dtype, declared shape and whether a view is accepted), the fixed arguments per call of a prebuilt binary, and
# the binaries (stems) with the dimensions of their buckets. Everything else is the implementation's own ABI and how it is built and launched: a
# tensor passed as a pointer or as a TMA descriptor, its runtime scalars, constexprs or fixed arguments, scratch slots, kind, preconditions, the
# dimensions its entries name, its kernel, compile options and grid. Each manifest entry carries its own ABI, and every host binds per entry.


@dataclass(frozen=True)
class _Prebuilt:
    """The prebuilt binary of an implementation (``tilefish.impl(ptx=...)`` / ``tilefish.impl(cubin=...)``) as :func:`impl` was given it: its files
    are read when the op is declared, a relative path relative to the file that declares the op (as for ``extern(ptx=...)``)."""

    kind: str  # "ptx" | "cubin"
    files: Any  # the PTX file, or {architecture: cubin file}
    symbol: str
    block: "tuple[int, int, int]"
    shared: int


@dataclass(frozen=True, eq=False)
class Implementation:
    """One implementation of an op declared with ``@tilefish.extern(by_arch=...)``, made by :func:`impl`: a Triton or Gluon kernel and how it is
    compiled and launched, or a prebuilt binary (``prebuilt``: PTX or cubins, the entry point, block and dynamic shared memory). ``stub`` is the
    function whose parameters are its kernel's (its ABI), or ``None``: the op's stub; ``consts`` gives const parameters of that stub other values
    (compiled in, or a prebuilt binary's fixed arguments); ``grid`` is its launch grid (``None``: the op's ``grid=``); ``name`` is the key that
    every architecture mapped to it shares (``None``: each architecture's own key)."""

    target: "_TritonTarget | None"
    language: str  # "triton" | "gluon": the frontend that compiles it; "ptx" | "cubin": a prebuilt binary
    num_warps: int
    num_stages: int
    fp_fusion: "bool | None"
    grid: Any
    consts: "dict[str, Any]"
    prebuilt: "_Prebuilt | None" = None
    stub: "Callable[..., Any] | None" = None
    name: "str | None" = None
    # how it is launched besides its grid: a persistent grid, the CTAs of a program, its global scratch, a configuration per bucket (each None
    # or 0 when the implementation does not say: the op's persistent= applies, one CTA, no scratch, one configuration)
    persistent: "Persistent | None" = None
    num_ctas: int = 1
    global_scratch: int = 0
    by_bucket: Any = None

    @property
    def ref(self) -> str:
        return self.target.ref if self.target is not None else ""

    @property
    def is_binary(self) -> bool:
        return self.prebuilt is not None

    def __repr__(self) -> str:
        named = "" if self.name is None else f" name={self.name!r}"
        if self.prebuilt is not None:
            return f"<impl {self.language} symbol={self.prebuilt.symbol} block={self.prebuilt.block}{named}>"
        return f"<impl {self.language}={self.ref} num_warps={self.num_warps}{named}>"


def _check_options(num_warps: Any, num_stages: Any, enable_fp_fusion: Any) -> None:
    if isinstance(num_warps, bool) or not isinstance(num_warps, int) or num_warps not in _NUM_WARPS:
        raise TilefishDeclarationError(f"extern: num_warps must be a power of two from 1 to 32, got {num_warps!r}")
    if isinstance(num_stages, bool) or not isinstance(num_stages, int) or num_stages < 1:
        raise TilefishDeclarationError(f"extern: num_stages must be a positive int, got {num_stages!r}")
    if enable_fp_fusion is not None and not isinstance(enable_fp_fusion, bool):
        raise TilefishDeclarationError(f"extern: enable_fp_fusion must be True, False or None (Triton's default), got {enable_fp_fusion!r}")


def impl(
    *,
    triton: Any = None,
    gluon: Any = None,
    ptx: Any = None,
    cubin: Any = None,
    symbol: "str | None" = None,
    block: Any = None,
    shared: int = 0,
    num_warps: Any = _UNSET,
    num_stages: Any = _UNSET,
    enable_fp_fusion: "bool | None" = None,
    grid: Any = None,
    consts: "dict[str, Any] | None" = None,
    stub: "Callable[..., Any] | None" = None,
    name: "str | None" = None,
    persistent: "Persistent | None" = None,
    num_ctas: int = 1,
    global_scratch: int = 0,
    by_bucket: Any = None,
) -> Implementation:
    """One implementation of an op with an implementation per architecture: ``@tilefish.extern(by_arch={"sm_103": tilefish.impl(triton=k_a,
    num_warps=8), "sm_120": tilefish.impl(triton=k_b)}, grid=...)``.

    The kernel is one of ``triton=`` (a ``triton.jit`` kernel) and ``gluon=`` (a Gluon kernel), a kernel object or a ``"module:function"`` string as for
    :func:`extern`, with its compile options ``num_warps``, ``num_stages`` and ``enable_fp_fusion``; or a **prebuilt binary**, ``ptx="k.ptx"`` or
    ``cubin={"sm_120": "k.cubin"}`` with ``symbol=``, ``block=`` and ``shared=`` (a relative path is relative to the file that declares the op, as
    for ``extern(ptx=...)``; a cubin implementation has a cubin for every architecture that runs it). ``grid`` is its launch grid when it is not the
    op's (a callable over the names of its stub, or a tuple); ``consts`` other values for const parameters of its stub (``{"BLOCK": 64}``): compiled
    in, or the values of a prebuilt binary's fixed arguments.

    ``stub=`` is a function annotated like the stub of an extern whose parameters are this kernel's, in order (its ABI): ``x: TensorDesc(Tensor[R,
    C, f16], block=(64, 64))`` where the op's stub has ``x: Tensor[R, C, f16]``, its own scalars, constexprs, fixed arguments. Without it the kernel
    takes the op's stub. Either way it takes the op's tensors (names, order, access, dtype, shape, ``Strided`` or not) and its fixed arguments per
    call: what a caller, a plan and a host launcher see is one signature, and each implementation's manifest entries have its own ABI.

    ``name="tcgen05"`` makes the implementation one for every architecture that maps to it: its key, in its declaration digest, its compile specs
    and its manifest entries (``"implementation": "tcgen05"``), is its name, so those architectures share one declaration digest and one set of
    compile specs. A name is an ASCII identifier of at most 64 characters that is not ``"fallback"`` and does not look like an architecture (it does
    not begin with ``sm_`` and a digit). Without a name an implementation's key is the architecture it is declared for (``"fallback"`` for the
    fallback), as for an op declared before names existed.

    ``persistent``, ``num_ctas``, ``global_scratch`` and ``by_bucket`` are those of :func:`extern`, for this implementation (``persistent=None``
    takes the op's); a prebuilt binary has none of the last three (nothing is compiled, and it has no scratch slot).
    """
    given = [k for k, v in (("triton", triton), ("gluon", gluon), ("ptx", ptx), ("cubin", cubin)) if v is not None]
    if len(given) != 1:
        raise TilefishDeclarationError(
            "tilefish.impl takes one kernel, triton= (a triton.jit kernel) or gluon= (a Gluon kernel), or a prebuilt binary, ptx= or cubin=, "
            f"got {' and '.join(given) or 'none'}"
        )
    language = given[0]
    if consts is None:
        consts = {}
    if not isinstance(consts, dict) or not all(isinstance(k, str) for k in consts):
        raise TilefishDeclarationError(f"tilefish.impl: consts= maps const parameter names to values ({{'BLOCK': 64}}), got {consts!r}")
    if stub is not None and (isinstance(stub, (Op, type)) or not callable(stub) or getattr(stub, "__code__", None) is None):
        raise TilefishDeclarationError(
            f"tilefish.impl: stub= is a function annotated like the stub of an extern, whose parameters are the kernel's in order, got {stub!r}"
        )
    if name is not None and not is_implementation_name(name):
        raise TilefishDeclarationError(
            "tilefish.impl: name= is the key of the implementation in its manifest entries: an ASCII identifier of at most 64 characters that is not "
            f"'fallback' and does not look like an architecture (begin with sm_ and a digit), got {name!r}"
        )
    if language in ("ptx", "cubin"):
        for key, value in (("num_warps", num_warps), ("num_stages", num_stages)):
            if value is not _UNSET:
                raise TilefishDeclarationError(f"tilefish.impl: {key}= is a Triton compile option; a prebuilt binary has its block= (and nothing is compiled)")
        if enable_fp_fusion is not None:
            raise TilefishDeclarationError(
                "tilefish.impl: enable_fp_fusion= is a Triton compile option; a prebuilt binary is not compiled by Triton (its PTX says how each "
                "instruction rounds)"
            )
        if symbol is None or block is None:
            raise TilefishDeclarationError(f"tilefish.impl: {language}= needs symbol=<the kernel's entry point> and block=(x, y, z)")
        files = ptx if language == "ptx" else cubin
        padded = _check_prebuilt(language, files, symbol, block, shared)
        prebuilt = _Prebuilt(language, files if language == "ptx" else dict(files), symbol, padded, shared)  # type: ignore[arg-type]
        return Implementation(None, language, 0, 0, None, grid, dict(consts), prebuilt, stub, name, persistent, num_ctas, global_scratch, by_bucket)
    for key, value in (("symbol", symbol), ("block", block)):
        if value is not None:
            raise TilefishDeclarationError(f"tilefish.impl: {key}= is for a prebuilt binary (ptx= or cubin=), not for a {language.title()} kernel")
    if shared != 0:
        raise TilefishDeclarationError("tilefish.impl: shared= is for a prebuilt binary; Triton records the shared memory of the kernel it compiles")
    target = _triton_target(triton if triton is not None else gluon, language)
    num_warps = 4 if num_warps is _UNSET else num_warps
    num_stages = 3 if num_stages is _UNSET else num_stages
    _check_options(num_warps, num_stages, enable_fp_fusion)
    return Implementation(
        target, language, num_warps, num_stages, enable_fp_fusion, grid, dict(consts), None, stub, name, persistent, num_ctas, global_scratch, by_bucket
    )


def _label(key: str) -> str:
    """How a message names the implementation of ``key``: ``the fallback``, its architecture (``sm_103``) or its name (``'tcgen05'``)."""
    if key == FALLBACK:
        return "the fallback"
    return key if _ARCH_NAME.fullmatch(key) else repr(key)


def _implementation_of(op: "ExternOp", key: str) -> "ExternOp":
    """The implementation ``key`` of an op declared with ``by_arch=`` (how one is unpickled: by its op, which pickles by reference)."""
    assert op.implementations is not None
    return op.implementations[key]


def _with_consts(nm: str, key: str, sig: Signature, consts: "dict[str, Any]", allow_dim: bool = True) -> Signature:
    """The stub's signature with the const values of one implementation (the same parameters, kinds and types: only compiled-in values differ)."""
    if not consts:
        return sig
    by_name = {p.name: p for p in sig.consts}
    unknown = sorted(set(consts) - set(by_name))
    if unknown:
        raise TilefishDeclarationError(
            f"extern op '{nm}': the implementation for {_label(key)} sets {unknown}, which are not const parameters of the stub ({sorted(by_name)})"
        )
    params = []
    for p in sig.params:
        if p.kind == "const" and p.name in consts:
            value = _check_const_default(f"extern op '{nm}' ({_label(key)})", p.name, p.const_dtype, consts[p.name], allow_dim=allow_dim)
            p = replace(p, default=value)
        params.append(p)
    return Signature(params, sig.returns)


def _public_facts(sig: Signature) -> "dict[str, Any]":
    """What a caller, a plan and a typed host launcher see of a stub: its tensors in order (name, access, dtype, declared shape, whether a view is
    accepted, a device scalar, the input an ``Out`` may be written over), the constraints of the dimensions their shapes name, and the fixed
    arguments per call (a prebuilt binary's ``const[i32]`` with no default)."""
    tensors: "list[dict[str, Any]]" = []
    dims: "dict[str, Any]" = {}
    for p in sig.params:
        if p.kind != "tensor":
            continue
        t = p.type
        assert t is not None
        tensors.append(
            {
                "name": p.name,
                "access": p.access,
                "dtype": str(t.dtype),
                "shape": [str(d.folded()) for d in t.shape],
                "view": bool(p.strided),
                "devscalar": bool(t.devscalar),
                "alias": p.alias,
            }
        )
        for d in t.shape:
            for a in d.free_dims():
                dims[a.name] = {"divisible_by": a.divisible_by, "buckets": None if a.buckets is None else list(a.buckets)}
    given = [{"name": p.name, "type": None if p.const_dtype is None else p.const_dtype.name} for p in sig.consts if p.default is MISSING]
    return {"tensors": tensors, "dims": dict(sorted(dims.items())), "given": given}


def _check_implementations(nm: str, sig: Signature, impls: "dict[str, ExternOp]") -> None:
    """Every implementation has the op's public signature (:func:`_public_facts` of its stub is the op's stub's) and the binaries (stems) of the
    others, each with the same buckets: a caller, a plan and a typed host launcher call any of them the same way, whichever architecture runs it.
    How an implementation passes its tensors, and the rest of its ABI, is its own: each manifest entry has its ABI, and every host binds per entry."""
    from .aot import _first_difference, _show, compile_specs

    want = _public_facts(sig)
    for key, op in impls.items():
        diff = _first_difference(want, _public_facts(op.signature), "")
        if diff is not None:
            path, a, b = diff
            raise TilefishDeclarationError(
                f"extern op '{nm}': the implementation for {_label(key)} does not have the op's signature: its stub differs at {path} (the op's has "
                f"{_show(a)}, its {_show(b)}). Every implementation takes the op's tensors (their names, order, access, dtype and shape, and whether a "
                "view is accepted) and its fixed arguments per call; how it passes a tensor (a pointer or a TMA descriptor), its scalars, constexprs, "
                "fixed arguments and preconditions are its own"
            )
    found: "dict[str, dict[str, dict[str, Any]]]" = {}
    for key, op in impls.items():
        try:
            specs = compile_specs(op)
        except TilefishError as e:
            raise TilefishDeclarationError(f"extern op '{nm}': the implementation for {_label(key)}: {e}") from None
        found[key] = {s.stem: {"buckets": dict(s.buckets), "dims": {d: s.dims.get(d) for d in s.buckets}} for s in specs}
    scratch = {key: op.global_scratch > 0 for key, op in impls.items()}
    if len(set(scratch.values())) > 1:  # a plan of the op is one document for every architecture: its steps have the workspace, or none has
        raise TilefishDeclarationError(
            f"extern op '{nm}': the implementations for {', '.join(_label(k) for k, v in scratch.items() if v)} have a global scratch and the ones for "
            f"{', '.join(_label(k) for k, v in scratch.items() if not v)} do not: every implementation has one (of any size), or none has"
        )
    (first_key, first), *rest = found.items()
    for key, got in rest:
        if list(got) != list(first):
            raise TilefishDeclarationError(
                f"extern op '{nm}': the implementation for {_label(key)} has the binaries {list(got)}, the one for {_label(first_key)} {list(first)}: "
                "every implementation has the same buckets (a const of one that depends on a bucketed dimension is one of the others too)"
            )
        for stem, wanted in first.items():
            diff = _first_difference(wanted, got[stem], "")
            if diff is not None:
                path, a, b = diff
                raise TilefishDeclarationError(
                    f"extern op '{nm}': the implementations for {_label(first_key)} and {_label(key)} do not have the same buckets: '{stem}' differs at "
                    f"{path.lstrip('.')} ({_show(a)} and {_show(b)}). Every implementation has the op's binaries, and a binary is selected by its buckets"
                )


def _check_prebuilt_archs(nm: str, key: str, binary: "BinaryTarget", archs: "list[str]") -> None:
    """A prebuilt implementation builds for every architecture that ``by_arch=`` sends to it: a cubin for each, PTX whose target each can run.
    (The architectures that fall back to it are not known here: the build refuses them, ``_binary.build_binary``.)"""
    if binary.kind == "cubin":
        missing = [a for a in archs if a not in binary.sources]
        if missing:
            raise TilefishDeclarationError(
                f"extern op '{nm}': the implementation for {_label(key)} is a cubin for {', '.join(sorted(binary.sources))}, but by_arch= sends "
                f"{', '.join(missing)} to it: a cubin is used as it is, for the architecture it was built for; give one for each "
                f"(cubin={{'{missing[0]}': ...}}) or PTX (ptx=...)"
            )
        return
    from ._binary import ptx_gpu_name
    from ._build import parse_arch

    try:
        text = binary.sources["ptx"].decode("utf-8")
    except UnicodeDecodeError:
        raise TilefishDeclarationError(f"extern op '{nm}': the PTX of the implementation for {_label(key)} is not text: PTX is UTF-8 text") from None
    for a in archs:
        try:
            ptx_gpu_name(text, parse_arch(a), f"the PTX of the implementation for {_label(key)}")
        except TilefishError as e:
            raise TilefishDeclarationError(f"extern op '{nm}': {e}, and by_arch= sends {a} to it") from None


def _declaring_dir(fn: Callable[..., Any]) -> "pathlib.Path | None":
    """The directory of the file that declares ``fn`` (what a relative path of a prebuilt binary is relative to), or ``None``."""
    base = pathlib.Path(fn.__code__.co_filename).resolve().parent if getattr(fn, "__code__", None) is not None else None
    if base is not None and not base.is_dir():
        base = None
    return base


def _implementation_op(
    fn: Callable[..., Any],
    nm: str,
    sig: Signature,
    limits: "dict[str, int]",
    key: str,
    im: Implementation,
    grid: Any,
    index_bits: int,
    reference: Any,
    base: "pathlib.Path | None",
    persistent: Any = None,
) -> "ExternOp":
    """The ``ExternOp`` of the implementation ``im`` under ``key``: its stub (its own, or the op's ``sig``) with its consts, its kernel or binary."""
    where = f"extern op '{nm}': the implementation for {_label(key)}"
    if im.stub is not None:
        try:
            _, sig, limits = _parse_extern_stub(im.stub, nm, index_bits, prebuilt=im.is_binary)
        except TilefishDeclarationError as e:
            raise TilefishDeclarationError(f"{where}: its stub: {e}") from None
    own = _with_consts(nm, key, sig, im.consts, allow_dim=not im.is_binary)
    binary = None
    if im.prebuilt is not None:
        _check_fixed_args(nm, own)
        pb = im.prebuilt
        binary = _binary_target(pb.kind, pb.files, pb.symbol, pb.block, pb.shared, base, nm)
    else:
        assert im.target is not None
        _check_against_kernel(nm, own, im.target)
    try:
        return ExternOp(
            fn,
            nm,
            own,
            im.ref,
            im.num_warps,
            im.num_stages,
            im.grid if im.grid is not None else grid,
            max_elements=limits,
            fp_fusion=im.fp_fusion,
            index_bits=index_bits,
            binary=binary,
            reference=reference,
            language="triton" if binary is not None else im.language,
            implementation=key,
            persistent=im.persistent if im.persistent is not None else persistent,
            num_ctas=im.num_ctas,
            global_scratch=im.global_scratch,
            by_bucket=im.by_bucket,
        )
    except TilefishDeclarationError as e:  # its grid, under its constexpr values
        raise TilefishDeclarationError(f"{where}: {e}") from None


def _extern_by_arch(
    by_arch: Any, fallback: Any, grid: Any, name: "str | None", index_bits: int, reference: Any, others: "dict[str, Any]", persistent: Any = None
) -> Callable[[Callable[..., Any]], ExternOp]:
    """``@tilefish.extern(by_arch={...}, fallback=..., grid=..., persistent=...)``: one op, one public signature, an implementation per architecture."""
    misplaced = [k for k, v in others.items() if v is not None and v is not _UNSET and not (k in ("shared", "global_scratch") and v == 0)]
    if misplaced:
        raise TilefishDeclarationError(
            f"extern: with by_arch= each implementation names its kernel and its compile options (tilefish.impl(triton=..., num_warps=...)), "
            f"so {', '.join(k + '=' for k in misplaced)} is not an argument of the op"
        )
    if not isinstance(by_arch, dict) or not by_arch:
        raise TilefishDeclarationError(
            f"extern: by_arch= maps architectures to implementations, {{'sm_103': tilefish.impl(triton=...), ...}}, got {by_arch!r}"
        )
    for arch, value in by_arch.items():
        if not isinstance(arch, str) or not _ARCH_NAME.fullmatch(arch) or arch != f"sm_{int(arch[3:])}":
            raise TilefishDeclarationError(f"extern: by_arch= keys are architectures such as 'sm_103' (no leading zero), got {arch!r}")
        if not isinstance(value, Implementation):
            raise TilefishDeclarationError(f"extern: by_arch[{arch!r}] must be a tilefish.impl(...) (a Triton or Gluon kernel, or a prebuilt PTX or cubin), got {value!r}")
    if fallback is not None and not isinstance(fallback, Implementation):
        raise TilefishDeclarationError(f"extern: fallback= is a tilefish.impl(...) (the implementation of every other architecture) or None, got {fallback!r}")
    if type(index_bits) is not int or index_bits not in (32, 64):
        raise TilefishDeclarationError(f"extern: index_bits must be 32 (the kernel indexes with i32) or 64 (with i64), got {index_bits!r}")
    _check_op_name(name)
    # the key of each architecture's implementation (its name, or the architecture), the fallback's, and the implementations under each key
    selection = {arch: (im.name if im.name is not None else arch) for arch, im in by_arch.items()}
    fallback_key = None if fallback is None else (fallback.name if fallback.name is not None else FALLBACK)
    chosen: "dict[str, list[Implementation]]" = {}
    for key, im in [*((selection[a], im) for a, im in by_arch.items()), *([(fallback_key, fallback)] if fallback is not None else [])]:
        group = chosen.setdefault(key, [])  # type: ignore[arg-type]
        if all(im is not other for other in group):
            group.append(im)
    everyone = [im for group in chosen.values() for im in group]
    binary_on_stub = any(im.is_binary and im.stub is None for im in everyone)  # the op's stub is then a prebuilt binary's C ABI
    all_binary = all(im.is_binary for im in everyone)  # only then may the op have fixed arguments per call (const[i32] with no default)

    def deco(fn: Callable[..., Any]) -> ExternOp:
        from .aot import _canonical_declaration, _first_difference, _show

        nm, sig, limits = _parse_extern_stub(fn, name, index_bits, prebuilt=binary_on_stub, givens=all_binary)
        base = _declaring_dir(fn)
        impls: "dict[str, ExternOp]" = {}
        for key, group in chosen.items():
            made = [_implementation_op(fn, nm, sig, limits, key, im, grid, index_bits, reference, base, persistent) for im in group]
            for other in made[1:]:  # two tilefish.impl(..., name=key) objects: one implementation only if they declare the same
                try:
                    diff = _first_difference(_canonical_declaration(made[0]), _canonical_declaration(other), "")
                except TilefishError as e:
                    raise TilefishDeclarationError(f"extern op '{nm}': the implementation {key!r}: {e}") from None
                if diff is not None:
                    path, a, b = diff
                    raise TilefishDeclarationError(
                        f"extern op '{nm}': two different implementations are named {key!r} (they differ at {path}: {_show(a)} and {_show(b)}): a name "
                        "is the key of one implementation, which every architecture mapped to it shares; name the other one apart"
                    )
            impls[key] = made[0]
        for key, one in impls.items():
            if one.binary is not None:
                _check_prebuilt_archs(nm, key, one.binary, [a for a, k in selection.items() if k == key])
        _check_implementations(nm, sig, impls)
        op = ExternOp(
            fn,
            nm,
            sig,
            "",
            0,
            0,
            None,
            max_elements=limits,
            index_bits=index_bits,
            reference=reference,
            implementations=impls,
            selection=selection,
            fallback_key=fallback_key,
        )
        for one in impls.values():
            one.__dict__["_tf_of"] = op
        return op

    return deco


@dataclass(frozen=True)
class BinaryTarget:
    """A prebuilt kernel that Triton does not compile: PTX that ``ptxas`` assembles for the target architecture, or a cubin per architecture.

    ``sources`` holds the bytes of the file(s) as they were when the operator was declared (the identity of the binary is the sha256 of
    those bytes, ``source_digest`` of kind ``"tilefish.binary.v1"``): ``{"ptx": ...}`` for PTX, ``{"sm_103": ...}`` per architecture
    for cubins. ``symbol`` is the kernel's entry point, ``block`` its thread block and ``shared`` its dynamic shared memory in bytes
    (a prebuilt kernel's ABI says no more than its C parameters, so the declaration says the rest). ``cluster`` is the thread-block cluster a launch
    groups its blocks in (``(1, 1, 1)``: none; the grid then counts clusters, and a launch has ``grid * cluster`` blocks).
    """

    kind: str  # "ptx" | "cubin"
    symbol: str
    block: "tuple[int, int, int]"
    shared: int
    sources: "dict[str, bytes]"
    cluster: "tuple[int, int, int]" = (1, 1, 1)

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


def _cluster_of(cluster: Any) -> "tuple[int, int, int]":
    if cluster is None:
        return (1, 1, 1)
    if not isinstance(cluster, (tuple, list)) or not 1 <= len(cluster) <= 3 or any(isinstance(c, bool) or not isinstance(c, int) for c in cluster):
        raise TilefishDeclarationError(f"extern: cluster must be a tuple of 1 to 3 ints (blocks per cluster along x, y, z), got {cluster!r}")
    padded = tuple(cluster) + (1,) * (3 - len(cluster))
    if any(c < 1 for c in padded) or math.prod(padded) > MAX_CLUSTER:
        raise TilefishDeclarationError(
            f"extern: cluster {tuple(cluster)} must have sizes of at least 1 and at most {MAX_CLUSTER} blocks in all (the portable cluster size)"
        )
    return padded  # type: ignore[return-value]



def _check_prebuilt(kind: str, files: Any, symbol: Any, block: Any, shared: Any) -> "tuple[int, int, int]":
    """What a prebuilt binary's declaration says before its files are read: the entry point, the dynamic shared memory, the architectures of its
    cubins; the block, padded to three sizes."""
    if not isinstance(symbol, str) or not _SYMBOL.fullmatch(symbol):
        raise TilefishDeclarationError(f"extern: symbol= names the kernel's entry point (a PTX/ELF symbol), got {symbol!r}")
    if isinstance(shared, bool) or not isinstance(shared, int) or not 0 <= shared <= 1 << 30:
        raise TilefishDeclarationError(f"extern: shared must be a non-negative int (bytes of dynamic shared memory), got {shared!r}")
    if kind == "ptx":
        if not isinstance(files, (str, os.PathLike)):
            raise TilefishDeclarationError(f"extern: ptx= must be a file path, got {files!r}")
    else:
        if not isinstance(files, dict) or not files:
            raise TilefishDeclarationError(f"extern: cubin= maps architectures to files, {{'sm_103': 'k.cubin'}}, got {files!r}")
        for arch in files:
            if not isinstance(arch, str) or not _ARCH_NAME.fullmatch(arch):
                raise TilefishDeclarationError(f"extern: cubin= keys are architectures such as 'sm_103', got {arch!r}")
    return _block_of(block)


def _binary_target(
    kind: str, files: Any, symbol: Any, block: Any, shared: Any, base: "pathlib.Path | None", where: str, cluster: Any = None
) -> BinaryTarget:
    padded = _check_prebuilt(kind, files, symbol, block, shared)
    sources: dict[str, bytes] = {}
    if kind == "ptx":
        sources["ptx"] = _read_binary_file("ptx=", where, files, base)
    else:
        for arch, path in files.items():
            sources[arch] = _read_binary_file(f"cubin[{arch!r}]", where, path, base)
    return BinaryTarget(kind, symbol, padded, shared, sources, _cluster_of(cluster))


def extern(
    *args: Any,
    triton: Any = None,
    gluon: Any = None,
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
    by_arch: Any = None,
    fallback: Any = None,
    persistent: "Persistent | None" = None,
    num_ctas: Any = _UNSET,
    cluster: Any = None,
    global_scratch: int = 0,
    by_bucket: Any = None,
) -> Callable[[Callable[..., Any]], ExternOp]:
    """``@tilefish.extern(triton="module:fn", num_warps=4, grid=...)``: type an existing Triton kernel.

    It always takes arguments (``@tilefish.extern`` without parentheses is an
    error). ``triton`` is a ``"module:function"`` string or a kernel object;
    ``grid`` is a callable over dimension and constant names, or a tuple.

    **A Gluon kernel** (``triton.experimental.gluon``: ``@gluon.jit``, explicit layouts) is declared with ``gluon=`` in place of ``triton=``,
    with everything else the same (``num_warps``, ``num_stages``, ``grid``, ``index_bits``, ``enable_fp_fusion``, the stub)::

        @tilefish.extern(gluon=my_gluon_kernels.row_sum, num_warps=4, grid=lambda R: (tilefish.cdiv(R, 4),))
        def row_sum(x: Strided[Tensor[R, 256, f32]], y: Out[Tensor[R, f32]], s_row: StrideOf("x", 0), n_rows: R, C: const = 256): ...

    The declaration says which frontend compiles the kernel because a compile spec is the whole recipe of its binary, made from the
    declaration alone (a ``"module:function"`` string is not imported to make it): Gluon's frontend builds TTGIR, with the layouts and
    ``num_warps`` in it, where Triton's builds TTIR, and that IR is what identifies the binary (``source_digest_kind``
    ``"tilefish.gluon-ir.v1"``, see :mod:`tilefish._ir`). The language is in the declaration digest and in every spec. A kernel object
    that says what it is (``is_gluon()``) is checked here: a Gluon kernel given as ``triton=``, or a ``triton.jit`` one as ``gluon=``, is
    refused with the argument to use; a string reference is checked when the kernel is built. The layouts of a Gluon kernel are written for
    a number of warps, so ``num_warps`` must be that number: Gluon's frontend refuses another one when the kernel is built.

    **A warp-specialised kernel** (Gluon's ``gl.warp_specialize``, or Triton's own warp specialisation of a ``tl.range(..., warp_specialize=True)``
    loop) is declared the same way, and ``num_warps`` is the warps of its **default partition**: the worker partitions add warps of their own (a 1-warp
    TMA producer and a 1-warp MMA issuer beside 4 compute warps make 8 in Triton 3.5.1, and beside 8 make 12). The build takes the compiled kernel's
    total when its TTGIR has warp-specialised regions, and launches with that block; it refuses any other kernel whose warps Triton changed. The manifest
    entry records the total (``total_num_warps``, manifest schema 3 section 4.2).

    **A prebuilt binary that Triton does not compile** is declared with ``ptx="file.ptx"`` or ``cubin={"sm_103": "file.cubin", ...}``
    instead of ``triton=``, together with ``symbol="entry_name"``, ``block=(x, y, z)`` and ``grid=...`` (and ``shared=0``, the bytes of
    dynamic shared memory the kernel addresses)::

        @tilefish.extern(ptx="scale.ptx", symbol="scale_bf16", block=(256,), grid=lambda n: (tilefish.cdiv(n, 256),))
        def scale(x: Tensor[n, bf16], y: Out[Tensor[n, bf16]], n_elem: n, alpha: const[f32] = 2.0): ...

    The stub lists the kernel's **C parameters, in order**: a tensor is a pointer (``Strided[...]`` and ``DevScalar[...]`` too), a
    parameter annotated with a runtime ``Dim`` is an ``i32`` (``i64[n]``: an ``i64``), a ``StrideOf`` an ``i32`` (``i64`` for ``index_bits=64``, or its ``dtype=``), and a typed
    ``const[i32]``, ``const[i64]`` or ``const[f32]`` with a default is a **fixed argument**: a scalar whose value is written in the
    manifest and passed at every launch (the way a prebuilt kernel takes its constants and its unused scratch slots; there are no
    constexprs, since nothing is compiled). A prebuilt binary has **no Triton scratch slots**: it is launched with exactly these
    parameters. A relative path is relative to the file that declares the operator, and the bytes are read once, there: the identity of
    the binary is their sha256 (``source_digest`` of kind ``tilefish.binary.v1``). PTX is assembled by ``ptxas`` for the target
    architecture with fixed, recorded flags; a cubin is used as it is for the architecture it names and refused for another.

    **An implementation per architecture.** ``by_arch=`` maps architectures to implementations (:func:`impl`: a Triton or Gluon kernel with its
    own compile options, constexpr values and grid, or a prebuilt PTX or cubin) of one op, and ``fallback=`` names the one of every other
    architecture::

        @tilefish.extern(by_arch={"sm_103": tilefish.impl(triton=k.sum_wide, num_warps=8), "sm_120": tilefish.impl(triton=k.sum_narrow)},
                         grid=lambda R: (R,))
        def row_sum(x: Tensor[R, 1024, f32], y: Out[Tensor[R, f32]], n_rows: R): ...

    The op is one public signature for a graph, a plan and a caller; ``export``, ``jit`` and ``replay`` run the implementation of the target's (or
    the device's) architecture, and refuse an architecture with none. Every implementation takes the stub's tensors (names, order, access, dtype,
    shape, ``Strided`` or not) and its fixed arguments per call, and has its buckets (checked here). An implementation may bring its own stub
    (``tilefish.impl(..., stub=f)``: the kernel's parameters in order), so its ABI is its own: a tensor passed as a TMA descriptor (``TensorDesc``)
    where another implementation takes a pointer, other scalars, constexprs or fixed arguments, a prebuilt binary beside a Triton kernel. Each is an
    op of its own (``op.for_arch("sm_103")``), whose declaration digest, compile specs and manifest entries carry its key (``"implementation":
    "sm_103"``, ``"fallback"``, or the name of a named implementation, ``tilefish.impl(..., name="tcgen05")``, which the architectures mapped to it
    share), so a replay refuses a binary of another implementation, or of one that changed. ``reference=``, ``index_bits=`` and ``name=`` are the
    op's.

    ``index_bits`` says how the kernel computes the offsets it loads and stores at. The default, ``32``, is the safe
    claim: every tensor parameter gets a ``max_elements`` precondition of ``2**31 - 1`` elements, which Python's ``replay`` and
    a Rust host check at launch, because a kernel that indexes with ``i32`` (``row * N_COLS + col`` on ``tl.int32`` values) goes
    through memory it was not given, silently, on a tensor that is any larger. A kernel that computes its offsets in ``i64``
    (``row.to(tl.int64) * N_COLS``) declares ``index_bits=64`` and has no such limit. The choice is in the declaration digest.

    ``enable_fp_fusion=`` is Triton's compile option of the same name, for a Triton or Gluon kernel only: ``False`` compiles ``a * b + c`` as a multiply
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
    default: the compiled-in value).

    **A null pointer.** A pointer the kernel does not use in the declared use (one it reads only when it is not null, or only under a constexpr
    flag the declaration turns off) is ``bias: Null[f32]``: a call does not pass it, it binds nothing, and every launch passes the null pointer
    (manifest kind ``null``). A Triton kernel and a prebuilt binary take one alike.

    **A 64-bit dimension scalar.** ``n_elem: i64[n]`` is a runtime scalar that carries the dimension ``n`` (any dimension expression) as an
    ``i64``, where ``n_elem: n`` is an ``i32``: a kernel whose element count or size parameter is 64 bits wide (a prebuilt kernel's ``long n``,
    a Triton kernel's ``i64`` scalar). Its value is derived from the shapes of the call like any dimension scalar's (it is never given by a
    call, unlike a fixed argument per call), and a launch refuses one that does not fit (``0`` to ``2**63 - 1``). The type is in the
    declaration digest (``"scalar_type": "i64"`` on the parameter; a declaration without one digests as before) and in the manifest (a
    ``scalar`` of type ``i64``). Python forbids a parameter without a
    default after one with a default, so when a constexpr comes before other
    parameters, put a bare ``*`` before it::

        @tilefish.extern(triton="my_kernels:k", grid=lambda n: (n,))
        def k(x: Tensor[n, bf16], *, BLOCK: const = 64, y: Out[Tensor[n, bf16]], n_elem: n): ...

    Declaration order is still the kernel's argument order (``x``, ``BLOCK``,
    ``y``, ``n_elem``); arguments after the ``*`` are passed by keyword at call
    sites. When ``triton`` is a kernel object that exposes ``arg_names`` (and
    ``params[i].is_constexpr``), the stub is checked against it.

    **Launch shapes** (manifest schema 3, sections 4.3 to 4.6):

    * **A grid may multiply dimensions**: ``grid=lambda T, C: (tilefish.cdiv(T * C, 1024),)`` is one program per 1024 elements of a ``[T, C]`` tensor,
      whatever ``T`` and ``C`` are. A host computes a grid; it never solves one, so a product is fine there (a shape stays affine).
    * ``persistent=tilefish.persistent(programs_per_sm=1)``: the grid (one entry) is the number of tiles, and the host launches at most
      ``programs_per_sm`` programs per SM of the device (or partition) it loads the binary on, so one binary serves every device shape; the kernel
      walks the tiles with a stride of ``tl.num_programs(0)``. Its bytes must not depend on how many programs run, unless it says so with
      ``sm_count=`` (see :class:`Persistent`).
    * ``num_ctas=2`` (Triton's compile option, 1, 2, 4 or 8): a program is a thread-block cluster of that many CTAs, launched with
      ``cuLaunchKernelEx`` (the grid counts programs: a launch has ``grid * num_ctas`` CTAs along x). A prebuilt binary declares ``cluster=(x, y, z)``
      instead: its grid counts clusters, and a launch has ``grid * cluster`` blocks.
    * ``global_scratch=256``: the bytes of global scratch each CTA of the compiled kernel addresses (Triton's ``global_scratch_size``: device-side TMA
      descriptors, ``tl.make_tensor_descriptor``, write theirs there). The build checks it against Triton's metadata; a host gives every launch
      ``global_scratch * CTAs`` bytes (a plan from its arena).
    * ``by_bucket=lambda T: tilefish.config(num_warps=8, BLOCK=128) if T > 256 else tilefish.config(num_warps=2, BLOCK=32)``: one compile-time
      configuration per bucket of the dimensions the callable names (each a ``Dim(..., buckets=[...])`` of the signature): the warps, stages and
      CTAs of that bucket's binary and the values of const parameters compiled into it (tile sizes), the grid evaluated with them. Each binary is
      built, keyed and digested with its own configuration; the declaration digest covers the configuration of every bucket.
    """
    if by_arch is not None or fallback is not None:
        if args:
            raise TilefishDeclarationError("@tilefish.extern takes keyword arguments only, not a bare @tilefish.extern")
        if by_arch is None:
            raise TilefishDeclarationError("extern: fallback= is the implementation of an op declared with by_arch= for the architectures it does not name")
        others = dict(
            triton=triton, gluon=gluon, ptx=ptx, cubin=cubin, symbol=symbol, block=block, shared=shared, num_warps=num_warps,
            num_stages=num_stages, enable_fp_fusion=enable_fp_fusion, num_ctas=num_ctas, cluster=cluster, global_scratch=global_scratch,
            by_bucket=by_bucket,
        )  # fmt: skip
        return _extern_by_arch(by_arch, fallback, grid, name, index_bits, reference, others, persistent)
    given = [k for k, v in (("triton", triton), ("gluon", gluon), ("ptx", ptx), ("cubin", cubin)) if v is not None]
    if args or not given:
        raise TilefishDeclarationError(
            '@tilefish.extern takes arguments: write @tilefish.extern(triton="module:function", grid=...) (gluon= for a Gluon kernel, '
            "ptx=/cubin= for a prebuilt binary), not a bare @tilefish.extern"
        )
    if len(given) > 1:
        raise TilefishDeclarationError(f"extern: a kernel is one of triton=, gluon=, ptx= and cubin=, got {' and '.join(given)}")
    prebuilt = triton is None and gluon is None
    language = "gluon" if gluon is not None else "triton"
    if enable_fp_fusion is not None and not isinstance(enable_fp_fusion, bool):
        raise TilefishDeclarationError(f"extern: enable_fp_fusion must be True, False or None (Triton's default), got {enable_fp_fusion!r}")
    if prebuilt:
        for key, value in (("num_warps", num_warps), ("num_stages", num_stages), ("num_ctas", num_ctas)):
            if value is not _UNSET:
                raise TilefishDeclarationError(
                    f"extern: {key}= is a Triton compile option; a prebuilt binary has its block= and cluster= (and nothing is compiled)"
                )
        num_ctas = 1
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
                raise TilefishDeclarationError(f"extern: {key}= is for a prebuilt binary (ptx= or cubin=), not for a {language.title()} kernel")
        if shared != 0:
            raise TilefishDeclarationError("extern: shared= is for a prebuilt binary; Triton records the shared memory of the kernel it compiles")
        if cluster is not None:
            raise TilefishDeclarationError("extern: cluster= is the thread-block cluster of a prebuilt binary; a Triton or Gluon kernel says num_ctas=")
        num_ctas = 1 if num_ctas is _UNSET else num_ctas
        target = _triton_target(gluon if gluon is not None else triton, language)
        num_warps = 4 if num_warps is _UNSET else num_warps
        num_stages = 3 if num_stages is _UNSET else num_stages
        _check_options(num_warps, num_stages, None)

    if type(index_bits) is not int or index_bits not in (32, 64):
        raise TilefishDeclarationError(f"extern: index_bits must be 32 (the kernel indexes with i32) or 64 (with i64), got {index_bits!r}")
    _check_op_name(name)

    def deco(fn: Callable[..., Any]) -> ExternOp:
        nm, sig, limits = _parse_extern_stub(fn, name, index_bits, prebuilt)
        binary = None
        if prebuilt:
            _check_fixed_args(nm, sig)
            base = _declaring_dir(fn)
            binary = _binary_target("ptx" if ptx is not None else "cubin", ptx if ptx is not None else cubin, symbol, block, shared, base, nm, cluster)
        else:
            assert target is not None
            _check_against_kernel(nm, sig, target)
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
            language=language,
            persistent=persistent,
            num_ctas=num_ctas,
            global_scratch=global_scratch,
            by_bucket=by_bucket,
        )

    return deco


def _check_op_name(name: Any) -> None:
    if name is not None and (not isinstance(name, str) or not _OP_NAME.fullmatch(name)):
        raise TilefishDeclarationError(
            f"extern: name must be a plain identifier (ASCII letters, digits and underscores, not starting with a digit): "
            f"it becomes a file name in a manifest directory, so {name!r} is not allowed"
        )


def _check_fixed_args(nm: str, sig: Signature) -> None:
    """Every const parameter of a prebuilt binary's stub with a default is a fixed argument: a scalar of a C type, with its value (one without a
    default is a fixed argument per call: each call gives it)."""
    for p in sig.consts:
        if p.default is MISSING:
            continue
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


def _parse_extern_stub(
    fn: Callable[..., Any], name: "str | None", index_bits: int, prebuilt: bool, givens: "bool | None" = None
) -> "tuple[str, Signature, dict[str, int]]":
    """The op name, the typed signature of an extern's stub and the ``max_elements`` limits its ``index_bits`` gives its tensors. ``prebuilt``: the
    stub is a prebuilt binary's C ABI (no TMA descriptor, no constexpr that a dimension gives); ``givens``: a const parameter with no default is a
    fixed argument per call (``const[i32]`` or ``const[i64]``), as it is for a prebuilt binary (the default) and for the stub of an op declared with
    ``by_arch=`` whose implementations all are prebuilt binaries."""
    if givens is None:
        givens = prebuilt
    nm = name or fn.__name__
    if not _OP_NAME.fullmatch(nm):
        raise TilefishDeclarationError(f"extern: the function name {nm!r} is not a plain identifier; pass name=... to give the op one")
    sig = parse_signature(
        fn,
        what="extern op",
        name=nm,
        allow_dim_const_defaults=not prebuilt,
        allow_strided=True,
        allow_output_dims=True,
        allow_alias=True,
        allow_tensor_map=not prebuilt,
        allow_wide_dims=True,
        allow_null=True,
    )
    if sig.returns:
        raise TilefishDeclarationError(
            f"extern op '{nm}': {'a kernel' if prebuilt else 'a Triton kernel'} returns nothing; declare results as Out[...] parameters"
        )
    for p in sig.consts:
        if p.default is MISSING and not givens:
            raise TilefishDeclarationError(f"extern op '{nm}': const parameter '{p.name}' needs a default, the value compiled in as constexpr")
        if p.default is MISSING and (p.const_dtype is None or p.const_dtype.name not in ("i32", "i64")):
            raise TilefishDeclarationError(
                f"extern op '{nm}': const parameter '{p.name}' of a prebuilt binary has no default, so every call gives its value (a fixed "
                f"argument per call): declare it `{p.name}: const[i32]` or const[i64] (a per-call value is an integer; an f32 is fixed by "
                "its default, written in the manifest)"
            )
    limits: dict[str, int] = {}
    if index_bits == 32:
        for p in sig.params:
            if p.kind != "tensor" or p.is_devscalar:  # one element: nothing to overflow
                continue
            if p.tensor_map is not None:  # a descriptor: the kernel addresses it by coordinates through TMA (each axis bounded by its i32 shape slot)
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
    return nm, sig, limits
