"""Operator signatures: parsing a stub's annotations and checking calls by unification.

In a signature a dynamic :class:`~tilefish.dims.Dim` is a type variable bound
afresh at every call, a static ``Dim`` is a literal that must match. Binding a
variable that has ``divisible_by=k`` to a static dimension checks the
requirement; binding it to a symbolic dimension records the requirement on the
trace so it propagates (see :class:`~tilefish.ir.Requirement`).
"""

from __future__ import annotations

import inspect
import math
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .dims import Dim, DimExpr, describe_pair, dim_text
from .dtypes import DType
from .errors import TilefishDeclarationError, TilefishError, TilefishTypeError
from .ir import BucketRequirement, Requirement
from .trace import Const, TraceContext, Value
from .types import ConstType, Marked, StrideOf, StridedType, TensorType, const

if TYPE_CHECKING:
    from .decorators import Op


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()


@dataclass(frozen=True)
class Param:
    """One parameter of an operator, in declaration order.

    ``kind`` is ``"tensor"`` (``type`` and ``access`` set), ``"const"``
    (compile-time scalar; ``const_dtype`` and ``default``) or ``"dim"`` (a
    runtime scalar carrying the value of the dimension expression ``dim``).

    A tensor parameter declared ``Strided[...]`` has ``strided=True``: its ``type`` carries the strides (each the dimension of
    the ``StrideOf`` parameter of that axis, or ``1`` for the last axis when it has none) and an offset dimension, which a call
    binds like it binds the shape; ``align`` is ``16`` when the declaration says the pointer is compiled as 16-byte aligned. A
    ``"dim"`` parameter declared ``StrideOf(arg, axis)`` has ``stride_of=(arg, axis)``: its ``dim`` is the stride.
    """

    name: str
    index: int
    kind: str
    access: str = "in"  # tensors: "in" | "inout" | "out"
    type: TensorType | None = None
    const_dtype: DType | None = None
    default: Any = MISSING
    dim: DimExpr | None = None
    keyword_only: bool = False
    strided: bool = False
    align: "int | None" = None
    stride_of: "tuple[str, int] | None" = None

    @property
    def is_callable_arg(self) -> bool:
        """Whether a caller supplies this parameter (``Out`` and dim parameters are derived)."""
        return self.kind == "const" or (self.kind == "tensor" and self.access != "out")

    @property
    def is_devscalar(self) -> bool:
        return self.kind == "tensor" and self.type is not None and self.type.devscalar

    def __str__(self) -> str:
        if self.kind == "tensor":
            ann = repr(self.type)
            if self.strided:
                assert self.type is not None
                ann = f"Strided[Tensor[{', '.join([*(str(d) for d in self.type.shape), str(self.type.dtype)])}]" + (f", {self.align}" if self.align else "") + "]"
            if self.access == "inout":
                ann = f"InOut[{ann}]"
            elif self.access == "out":
                ann = f"Out[{ann}]"
        elif self.kind == "dim":
            ann = f"{self.dim}" if self.stride_of is None else repr(StrideOf(*self.stride_of))
        else:
            ann = "const" if self.const_dtype is None else f"const[{self.const_dtype}]"
        text = f"{self.name}: {ann}"
        if self.default is not MISSING:
            text += f" = {self.default}"
        return text


class Signature:
    """The ordered parameters and the declared return types of an operator."""

    def __init__(self, params: Sequence[Param], returns: Sequence[TensorType]) -> None:
        self.params = tuple(params)
        self.returns = tuple(returns)
        py_params = []
        for p in self.params:
            if not p.is_callable_arg:
                continue
            kind = inspect.Parameter.KEYWORD_ONLY if p.keyword_only else inspect.Parameter.POSITIONAL_OR_KEYWORD
            default = inspect.Parameter.empty if p.default is MISSING else p.default
            py_params.append(inspect.Parameter(p.name, kind, default=default))
        self.call_signature = inspect.Signature(py_params)

    @property
    def tensor_inputs(self) -> tuple[Param, ...]:
        return tuple(p for p in self.params if p.kind == "tensor" and p.access != "out")

    @property
    def tensor_outputs(self) -> tuple[Param, ...]:
        return tuple(p for p in self.params if p.kind == "tensor" and p.access == "out")

    @property
    def consts(self) -> tuple[Param, ...]:
        return tuple(p for p in self.params if p.kind == "const")

    @property
    def dim_params(self) -> tuple[Param, ...]:
        return tuple(p for p in self.params if p.kind == "dim")

    def __str__(self) -> str:
        text = ", ".join(str(p) for p in self.params)
        if self.returns:
            rets = ", ".join(repr(r) for r in self.returns)
            text += f") -> {rets if len(self.returns) == 1 else '(' + rets + ')'}"
        else:
            text += ")"
        return "(" + text


# -- parsing -------------------------------------------------------------------------


def _resolve(ann: Any, fn: Callable[..., Any], where: str) -> Any:
    if isinstance(ann, str):
        try:
            return eval(ann, getattr(fn, "__globals__", {}), {})  # noqa: S307 - the user's own annotation
        except Exception as e:  # pragma: no cover - depends on user code
            raise TilefishDeclarationError(
                f"{where}: cannot evaluate string annotation {ann!r} ({e}); avoid "
                "`from __future__ import annotations` for tilefish declarations"
            ) from None
    return ann


def _check_const_default(where: str, p_name: str, dtype: DType | None, default: Any, allow_dim: bool) -> Any:
    if isinstance(default, DimExpr):
        if default.is_static:
            default = default.value
        elif allow_dim:
            return default
        else:
            raise TilefishDeclarationError(
                f"{where}: const parameter '{p_name}' defaults to the non-static dimension {default}; "
                "only static dims can be const values here"
            )
    if isinstance(default, bool):
        if dtype is not None:
            raise TilefishDeclarationError(f"{where}: const[{dtype}] parameter '{p_name}' cannot default to a bool")
        return default
    if not isinstance(default, (int, float)):
        raise TilefishDeclarationError(
            f"{where}: const parameter '{p_name}' must default to a number, a bool or a static Dim, got {default!r}"
        )
    if dtype is not None:
        if dtype.is_float:
            return float(default)
        if not isinstance(default, int):
            raise TilefishDeclarationError(f"{where}: const[{dtype}] parameter '{p_name}' needs an int, got {default!r}")
    return default


def parse_signature(
    fn: Callable[..., Any],
    *,
    what: str,
    name: str,
    allow_dim_const_defaults: bool = False,
    check_derivable: bool = True,
    allow_strided: bool = False,
    allow_devscalar: bool = True,
) -> Signature:
    """Read ``fn``'s annotations into a :class:`Signature`.

    ``allow_strided`` lets a tensor parameter be ``Strided[...]`` and a scalar parameter ``StrideOf(...)`` (an extern op does;
    see :class:`~tilefish.types.Strided`).
    """
    where = f"{what} '{name}'"
    sig = inspect.signature(fn)
    params: list[Param] = []
    for index, (pname, p) in enumerate(sig.parameters.items()):
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TilefishDeclarationError(f"{where}: *args/**kwargs are not supported (parameter '{pname}')")
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TilefishDeclarationError(f"{where}: positional-only parameters are not supported ('{pname}')")
        if p.annotation is inspect.Parameter.empty:
            raise TilefishDeclarationError(
                f"{where}: parameter '{pname}' needs an annotation (Tensor[...], InOut[...], Out[...], const, or a Dim)"
            )
        ann = _resolve(p.annotation, fn, where)
        kw = p.kind is inspect.Parameter.KEYWORD_ONLY
        has_default = p.default is not inspect.Parameter.empty
        if isinstance(ann, TensorType):
            if ann.devscalar and not allow_devscalar:
                raise TilefishDeclarationError(f"{where}: parameter '{pname}' is a DevScalar, which a {what} cannot take")
            _require_contiguous(where, what, f"parameter '{pname}'", ann)
            param = Param(pname, index, "tensor", "in", type=ann, keyword_only=kw)
        elif isinstance(ann, StridedType):
            if not allow_strided:
                raise TilefishDeclarationError(
                    f"{where}: parameter '{pname}' is Strided[...], which only an extern op can take (a {what} takes contiguous tensors)"
                )
            require_storable(where, f"parameter '{pname}'", ann.type)
            param = Param(pname, index, "tensor", "in", type=ann.type, keyword_only=kw, strided=True, align=ann.align)
        elif isinstance(ann, StrideOf):
            if not allow_strided:
                raise TilefishDeclarationError(f"{where}: parameter '{pname}' is {ann!r}, which only an extern op can take")
            if has_default:
                raise TilefishDeclarationError(f"{where}: stride parameter '{pname}' cannot have a default")
            param = Param(
                pname, index, "dim", dim=Dim(pname, divisible_by=ann.divisible_by), keyword_only=kw, stride_of=(ann.arg, ann.axis)
            )
        elif isinstance(ann, Marked):
            _require_contiguous(where, what, f"parameter '{pname}'", ann.type)
            param = Param(pname, index, "tensor", ann.mode, type=ann.type, keyword_only=kw)
        elif ann is const or isinstance(ann, ConstType):
            dtype = None if ann is const else ann.dtype
            default = (
                _check_const_default(where, pname, dtype, p.default, allow_dim_const_defaults)
                if has_default
                else MISSING
            )
            param = Param(pname, index, "const", const_dtype=dtype, default=default, keyword_only=kw)
        elif isinstance(ann, DimExpr):
            if ann.is_static:
                raise TilefishDeclarationError(
                    f"{where}: parameter '{pname}' is annotated with the static dimension {ann!r}; "
                    f"a runtime scalar needs a dynamic Dim. For a compile-time constant write "
                    f"`{pname}: const = {ann}`"
                )
            if has_default:
                raise TilefishDeclarationError(f"{where}: runtime dimension parameter '{pname}' cannot have a default")
            param = Param(pname, index, "dim", dim=ann, keyword_only=kw)
        else:
            raise TilefishDeclarationError(
                f"{where}: parameter '{pname}' has unsupported annotation {ann!r}; "
                "use Tensor[...], InOut[...], Out[...], DevScalar[...], const, const[dtype] or a Dim"
            )
        if param.kind == "tensor" and has_default:
            raise TilefishDeclarationError(f"{where}: tensor parameter '{pname}' cannot have a default")
        params.append(param)
    params = _resolve_strides(where, params)

    ret_ann = _resolve(sig.return_annotation, fn, where)
    returns: tuple[TensorType, ...]
    if ret_ann is inspect.Signature.empty or ret_ann is None:
        returns = ()
    elif isinstance(ret_ann, TensorType):
        returns = (ret_ann,)
    else:
        items = typing.get_args(ret_ann) if typing.get_origin(ret_ann) is tuple else ret_ann
        if not isinstance(items, (tuple, list)) or not all(isinstance(t, TensorType) for t in items):
            raise TilefishDeclarationError(
                f"{where}: the return annotation must be a Tensor[...] or a tuple of them, got {ret_ann!r}"
            )
        returns = tuple(items)
    for i, r in enumerate(returns):
        _require_contiguous(where, what, f"return value {i}", r)

    result = Signature(params, returns)
    if check_derivable:
        _check_derivable(where, result)
    return result


def require_storable(where: str, label: str, t: TensorType) -> None:
    """A tensor that lives in memory (a parameter or a result) has a storage dtype: a mask (``bool_``) is a value inside a body only."""
    if isinstance(t.dtype, DType) and t.dtype.is_bool:
        raise TilefishDeclarationError(
            f"{where}: {label} is {t!r}, a mask: a mask is a value inside a kernel body and has no storage type here. Store it as a "
            "number with .to(u8) (0 or 1) and read it back with a comparison (m != 0)"
        )


def _require_contiguous(where: str, what: str, label: str, t: TensorType) -> None:
    """A tensor annotation that is not ``Strided[...]`` is a contiguous tensor: row-major strides, offset 0, 16-byte aligned.

    That is the contract of a pointer a launch checks (``_launch.check_tensor`` refuses a tensor that is not contiguous), what a plan
    assumes of a graph's parameters, and what the Rust host checks. A ``TensorType`` with other strides, an offset or no alignment would
    say otherwise and be believed by nothing, so it is refused here: a parameter that takes a view is ``Strided[...]``.
    """
    require_storable(where, label, t)
    offset = t.offset
    dense = t.strides is None and offset is not None and offset.is_static and offset.value == 0
    if dense and t.contiguous:
        return
    raise TilefishDeclarationError(
        f"{where}: {label} is annotated with {t!r} ({t.layout_text()}), which is not a contiguous tensor: a tensor parameter or result "
        f"of a {what} is contiguous (row-major strides, offset 0), and a launch checks that it is. A parameter that takes a strided view "
        "(other strides, an offset) is declared Strided[Tensor[...]] with StrideOf parameters (an extern op)"
    )


def _resolve_strides(where: str, params: "list[Param]") -> "list[Param]":
    """Give every ``Strided[...]`` tensor parameter the strides and the offset its ``StrideOf`` parameters declare.

    Axis ``a`` of ``x`` has the stride dimension of the parameter declared ``StrideOf("x", a)``; the last axis, when it has none,
    has stride 1 (the kernel loads along it contiguously). The offset is the dimension ``<x>_offset``: when the declaration says
    ``Strided[..., 16]`` it is ``divisible_by`` the number of elements in 16 bytes, so that 16-byte alignment of the pointer is
    *provable* from the type, and ``tt.divisibility`` follows from that proof alone.
    """
    by_name = {p.name: p for p in params}
    stride_params: dict[tuple[str, int], Param] = {}
    for p in params:
        if p.stride_of is None:
            continue
        arg, axis = p.stride_of
        target = by_name.get(arg)
        if target is None or target.kind != "tensor" or not target.strided:
            raise TilefishDeclarationError(
                f"{where}: {p.name} is StrideOf({arg!r}, {axis}), but '{arg}' is not a Strided[...] tensor parameter of this op"
            )
        assert target.type is not None
        if axis >= target.type.rank:
            raise TilefishDeclarationError(f"{where}: {p.name} is the stride of axis {axis} of '{arg}', which has rank {target.type.rank}")
        if (arg, axis) in stride_params:
            raise TilefishDeclarationError(
                f"{where}: the stride of axis {axis} of '{arg}' is declared twice ({stride_params[(arg, axis)].name} and {p.name})"
            )
        stride_params[(arg, axis)] = p
    out: list[Param] = []
    for p in params:
        if not (p.kind == "tensor" and p.strided):
            out.append(p)
            continue
        assert p.type is not None
        t = p.type
        if t.rank < 1:
            raise TilefishDeclarationError(f"{where}: Strided parameter '{p.name}' needs at least one axis")
        strides: list[Any] = []
        for axis in range(t.rank):
            sp = stride_params.get((p.name, axis))
            if sp is not None:
                strides.append(sp.dim)
            elif axis == t.rank - 1:
                strides.append(1)
            else:
                raise TilefishDeclarationError(
                    f"{where}: Strided parameter '{p.name}' has no StrideOf({p.name!r}, {axis}) parameter: only the last axis may "
                    "go without one (its stride is then 1); declare "
                    f"`{p.name}_s{axis}: StrideOf({p.name!r}, {axis})` in the kernel's argument order"
                )
        offset_name = f"{p.name}_offset"
        if offset_name in by_name:
            raise TilefishDeclarationError(f"{where}: the parameter '{offset_name}' clashes with the offset dimension of Strided parameter '{p.name}'")
        elems16 = 128 // math.gcd(t.dtype.bits, 128)  # type: ignore[union-attr]  # elements in 16 bytes
        offset = Dim(offset_name, divisible_by=elems16 if p.align == 16 else None)
        typed = TensorType(t.shape, t.dtype, strides=strides, offset=offset, aligned=False, normalize=False)
        out.append(
            Param(p.name, p.index, "tensor", p.access, type=typed, keyword_only=p.keyword_only, strided=True, align=p.align)
        )
    return out


def _check_derivable(where: str, sig: Signature) -> None:
    """Outputs and dim parameters may only mention dims that the input tensors determine at a launch: those of their shapes and, for a
    ``Strided[...]`` parameter, of its strides. **Not the offset dimension** (``x_offset``) of a strided parameter: a launch does not bind it, in
    Python (``_launch.bind_infos``) or in Rust (manifest schema 3, section 5.1, step 7: the check of the address is the one made), so a result
    shaped by it, or a runtime scalar that carries it, could be traced but never launched."""
    determined: set[Dim] = set()
    for p in sig.tensor_inputs:
        t = p.type
        assert t is not None
        for d in t.shape:
            determined |= d.free_dims()
        for s in t.strides or ():
            if s is not None:
                determined |= s.free_dims()
    targets: list[tuple[str, Any]] = []
    for p in sig.tensor_outputs:
        targets.append((f"Out parameter '{p.name}'", p.type))
    for i, r in enumerate(sig.returns):
        targets.append((f"return value {i}", r))
    for p in sig.dim_params:
        targets.append((f"runtime parameter '{p.name}'", p.dim))
    offsets = {p.type.offset for p in sig.tensor_inputs if p.strided and p.type is not None}
    for label, t in targets:
        free = t.free_dims() - determined
        missing = sorted(d.name for d in free)
        if missing:
            note = (
                " (the offset dimension of a Strided[...] parameter is not bound at a launch, which checks the address of the view instead: it cannot "
                "size a result or be passed as a scalar)"
                if free & offsets
                else ""
            )
            raise TilefishDeclarationError(
                f"{where}: {label} mentions dimension(s) {', '.join(missing)} that no input tensor determines{note}"
            )


# -- calls ------------------------------------------------------------------------------


def _dtype_equal(a: TensorType, b: TensorType) -> bool:
    return a._key()[1] == b._key()[1]


class _Unifier:
    def __init__(self, op: "Op", ctx: TraceContext) -> None:
        self.op = op
        self.ctx = ctx
        self.bindings: dict[Dim, DimExpr] = {}
        self.bound_by: dict[Dim, str] = {}
        self.equations: list[tuple[str, int, DimExpr, DimExpr, TensorType, str]] = []

    def add(
        self, pname: str, axis: int, pattern: DimExpr, actual: DimExpr, actual_type: TensorType, what: str = "dimension"
    ) -> None:
        """One equation between a pattern of the signature and what the argument has: a ``dimension`` of its shape, a ``stride`` or its ``offset``."""
        self.equations.append((pname, axis, pattern, actual, actual_type, what))

    def solve(self) -> None:
        pending = self.equations
        while pending:
            rest = [eq for eq in pending if not self._try(eq)]
            if len(rest) == len(pending):
                pname, axis, pattern, actual, _, what = rest[0]
                raise TilefishTypeError(
                    f"{self.op.name}: argument '{pname}': cannot infer the dimensions in {pattern} "
                    f"({self._where(what, axis)}, argument value {actual}); the signature is ambiguous at this call"
                )
            pending = rest

    def _origin(self, pname: str) -> str:
        return f"{self.op.name}: argument '{pname}'"

    @staticmethod
    def _where(what: str, axis: int) -> str:
        return {"dimension": f"dimension {axis}", "stride": f"the stride of axis {axis}", "offset": "the offset"}[what]

    def _try(self, eq: tuple[str, int, DimExpr, DimExpr, TensorType, str]) -> bool:
        pname, axis, pattern, actual, atype, what = eq
        unbound = [d for d in pattern.free_dims() if d not in self.bindings]
        if not unbound:
            expected = pattern.subst(self.bindings)
            if expected != actual:
                self._mismatch(pname, axis, pattern, expected, actual, atype, what)
            return True
        if len(unbound) != 1:
            return False
        var = unbound[0]
        coef = pattern.coeff(var)
        if coef == 0:  # the variable only occurs inside a cdiv
            return False
        rest = pattern - var * coef
        if any(d not in self.bindings for d in rest.free_dims()):
            return False
        if not rest.terms and rest.offset == 0:
            target = actual  # keep the actual dimension as is (and its name)
        else:
            target = actual - rest.subst(self.bindings)
        cannot = f"{self._origin(pname)}: {self._where(what, axis)} is {actual}, which cannot match {pattern}"
        if coef == 1:
            value = target  # keeps the caller's dimension (and its name) as is
        elif target.is_static:
            if target.value % coef:
                raise TilefishTypeError(f"{cannot} (would need {var} = {target.value}/{coef})")
            value = DimExpr.const(target.value // coef)
        else:
            q = target._div_exact_int(coef)
            if q is None:
                raise TilefishTypeError(f"{cannot} (cannot solve for {var})")
            value = q
        if value.is_static and value.value < (1 if what == "dimension" else 0):  # a stride or an offset can be 0
            raise TilefishTypeError(f"{cannot} (would need {var} = {value.value})")
        self._bind(var, value, pname, axis, what)
        return True

    def _bind(self, var: Dim, value: DimExpr, pname: str, axis: int, what: str = "dimension") -> None:
        origin = self._origin(pname)
        k = var.divisible_by
        if k:
            if value.is_static:
                if value.value % k:
                    if what == "offset":
                        raise TilefishTypeError(
                            f"{origin}: the offset of the view is {value.value} elements, which is not a multiple of {k} (16 bytes); "
                            "this parameter is Strided[..., 16] and its kernel is compiled for a 16-byte aligned pointer"
                        )
                    raise TilefishTypeError(
                        f"{origin}: {var} requires divisible_by={k} but {self._where(what, axis)} is {value.value}"
                    )
            else:
                self.ctx.require_divisible(value, k, origin)
        if var.buckets:
            if value.is_static:
                if value.value not in var.buckets:
                    raise TilefishTypeError(
                        f"{origin}: {var} is bucketed over {list(var.buckets)} but dimension {axis} is {value.value}"
                    )
            else:
                _check_buckets_overlap(origin, str(var), var.buckets, value)
                self.ctx.require_bucket(value, var.buckets, origin)
        self.bindings[var] = value
        self.bound_by[var] = f"argument '{pname}'"

    def _mismatch(
        self, pname: str, axis: int, pattern: DimExpr, expected: DimExpr, actual: DimExpr, atype: TensorType, what: str = "dimension"
    ) -> None:
        actual_text, expected_text = describe_pair(actual.folded(), expected.folded())
        required = str(pattern)
        if required != expected_text:
            required += f" = {expected_text}"
        text = f"{self._origin(pname)} ({atype}): {self._where(what, axis)} is {actual_text}, but the signature requires {required}"
        if what == "stride":
            text += " (the kernel loads along that axis contiguously: pass a view whose last axis has stride 1)"
        notes = [
            f"{v} was bound to {dim_text(self.bindings[v])} by {self.bound_by[v]}"
            for v in pattern.free_dims()
            if v in self.bindings
        ]
        if notes:
            text += " (" + "; ".join(sorted(notes)) + ")"
        raise TilefishTypeError(text)

    def instantiate(self, t: TensorType, what: str) -> TensorType:
        free = [d for d in t.free_dims() if d not in self.bindings]
        if free:
            raise TilefishTypeError(
                f"{self.op.name}: cannot infer dimension(s) {', '.join(sorted(d.name for d in free))} of {what}"
            )
        shape = [d.subst(self.bindings) for d in t.shape]
        if t.contiguous:
            return TensorType(shape, t.dtype, known=self.ctx.known, devscalar=t.devscalar)
        # a view: its layout, with the bindings of this call substituted
        strides = None if t.strides is None else [None if s is None else s.subst(self.bindings) for s in t.strides]
        offset = None if t.offset is None else t.offset.subst(self.bindings)
        return TensorType(shape, t.dtype, known=self.ctx.known, strides=strides, offset=offset, aligned=False if strides is not None else None)


def _check_buckets_overlap(origin: str, name: str, buckets: "tuple[int, ...]", value: DimExpr) -> None:
    """A bucketed dim passed to a bucketed callee dim: the bucket sets must at least overlap.

    A subset needs nothing; a superset or a partial overlap becomes a bucket requirement (a runtime
    check that the value is one of the callee's buckets); disjoint sets can never match.
    """
    if isinstance(value, Dim) and value.buckets and not set(value.buckets) & set(buckets):
        raise TilefishTypeError(
            f"{origin}: {name} is bucketed over {list(buckets)} but {value} has buckets "
            f"{list(value.buckets)} (no bucket in common)"
        )


def _check_const_arg(op: "Op", p: Param, value: Any) -> Any:
    if isinstance(value, Const):
        if value.value is None:
            raise TilefishTypeError(f"{op.name}: argument '{p.name}': const '{value.name}' has no value at trace time")
        if value.dtype is not None and p.const_dtype is not None and value.dtype != p.const_dtype:
            raise TilefishTypeError(
                f"{op.name}: argument '{p.name}' is const[{p.const_dtype}] but '{value.name}' is const[{value.dtype}]; "
                "there is no implicit conversion between typed constants"
            )
        value = value.value
    if isinstance(value, DimExpr):
        if not value.is_static:
            raise TilefishTypeError(f"{op.name}: argument '{p.name}' must be a compile-time constant, got dimension {value}")
        value = value.value
    if isinstance(value, bool):
        if p.const_dtype is not None:
            raise TilefishTypeError(f"{op.name}: argument '{p.name}' is const[{p.const_dtype}] and cannot take a bool")
        return value
    if not isinstance(value, (int, float)):
        raise TilefishTypeError(
            f"{op.name}: argument '{p.name}' must be a compile-time number, got {type(value).__name__} ({value!r})"
        )
    if p.const_dtype is not None:
        if p.const_dtype.is_float:
            return float(value)
        if not isinstance(value, int):
            raise TilefishTypeError(f"{op.name}: argument '{p.name}' is const[{p.const_dtype}] and needs an int, got {value!r}")
    return value


def _apply_callee_requirement(
    op: "Op", r: "Requirement | BucketRequirement", bindings: dict[Dim, DimExpr], ctx: TraceContext
) -> None:
    """A graph or kernel body recorded a requirement on its own dims: restate it at this call site."""
    expr = r.dim.subst(bindings)
    origin = f"{op.name} (its body: {r.origin})"
    if isinstance(r, Requirement):
        if expr.is_static:
            if expr.value % r.divisor:
                raise TilefishTypeError(
                    f"{op.name}: its body requires {r.dim} to be divisible by {r.divisor} ({r.origin}), "
                    f"but this call binds it to {expr.value}"
                )
        else:
            ctx.require_divisible(expr, r.divisor, origin)
    else:
        if expr.is_static:
            if expr.value not in r.buckets:
                raise TilefishTypeError(
                    f"{op.name}: its body requires {r.dim} to be one of {list(r.buckets)} ({r.origin}), "
                    f"but this call binds it to {expr.value}"
                )
        else:
            _check_buckets_overlap(f"{op.name}: its body ({r.origin})", str(r.dim), r.buckets, expr)
            ctx.require_bucket(expr, r.buckets, origin)


def call_op(op: "Op", args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Type-check a call to a declared operator inside a trace and emit its node."""
    sig = op.signature
    try:
        bound = sig.call_signature.bind(*args, **kwargs)
    except TypeError as e:
        raise TilefishTypeError(f"{op.name}: {e}; signature {sig}") from None
    bound.apply_defaults()

    ctx: TraceContext | None = None
    for p in sig.tensor_inputs:
        arg = bound.arguments[p.name]
        if isinstance(arg, Value):
            ctx = arg.ctx
            break
    if ctx is None:
        raise TilefishTypeError(
            f"{op.name}: can only be called inside a graph or kernel body, with traced tensor arguments"
        )
    snapshot = ctx.snapshot()
    try:
        return _call(op, bound.arguments, ctx)
    except TilefishError:
        ctx.restore(snapshot)  # a failed call leaves no requirements behind
        raise


def _add_layout_equations(op: "Op", p: Param, pt: TensorType, at: TensorType, unifier: _Unifier) -> None:
    """The strides and the offset of a view passed to a ``Strided[...]`` parameter, as equations on the stride and offset dimensions.

    Each free stride binds to the stride the argument has (a dense argument has the row-major strides); an axis the kernel
    loads contiguously (stride ``1``) must have stride 1, unless the argument's size along it is 1. The offset dimension binds to
    the offset of the argument; for ``Strided[..., 16]`` it is ``divisible_by`` the elements in 16 bytes, so a view whose offset
    is not provably a multiple of 16 bytes is refused here (or, when it depends on dimensions, recorded as a requirement).
    """
    assert pt.strides is not None
    actual = at.layout_strides()
    for axis, (pattern, got) in enumerate(zip(pt.strides, actual)):
        assert pattern is not None
        if got is None:
            raise TilefishTypeError(
                f"{op.name}: argument '{p.name}' is the view {at} whose stride along axis {axis} is not an affine expression "
                "(a product of two symbolic dimensions), so it cannot be passed as a stride; make one of the factors static"
            )
        if pattern.is_static and at.shape[axis].is_static and at.shape[axis].value == 1:
            continue  # the stride of a size-1 axis is never used
        unifier.add(p.name, axis, pattern, got, at, "stride")
    assert pt.offset is not None
    if at.offset is None:
        if p.align:
            raise TilefishTypeError(
                f"{op.name}: argument '{p.name}': the offset of the view {at} is not an affine expression, so it cannot be shown "
                f"to be a multiple of {p.align} bytes, which this Strided[..., {p.align}] parameter needs"
            )
        return
    unifier.add(p.name, -1, pt.offset, at.offset, at, "offset")


def _call(op: "Op", arguments: dict[str, Any], ctx: TraceContext) -> Any:
    sig = op.signature
    unifier = _Unifier(op, ctx)
    inputs: list[Value] = []
    seen: dict[int, tuple[str, str]] = {}  # id of the live value -> (parameter, access)
    inout_args: list[tuple[Param, Value, Value]] = []  # parameter, the argument as given, the version it reads
    for p in sig.tensor_inputs:
        arg = arguments[p.name]
        pt = p.type
        assert pt is not None
        if not isinstance(arg, Value):
            raise TilefishTypeError(
                f"{op.name}: argument '{p.name}' must be a traced tensor of type {pt}, got "
                f"{type(arg).__name__} ({arg!r})"
            )
        if arg.ctx is not ctx:
            raise TilefishTypeError(f"{op.name}: argument '{p.name}' belongs to a different trace")
        if arg.access == "out":
            raise TilefishTypeError(f"{op.name}: argument '{p.name}' reads Out parameter '{arg.name}', which is write-only")
        if p.access == "inout" and arg.is_param and arg.access == "in":
            raise TilefishTypeError(
                f"{op.name}: argument '{p.name}' is written in place but '{arg.name}' is a read-only input; "
                "declare it InOut[...] in the enclosing signature"
            )
        live = arg._current(f"{op.name}: argument '{p.name}'")  # raises for a value an in-place call consumed
        other = seen.get(id(live))
        if other is not None and "inout" in (other[1], p.access):
            raise TilefishTypeError(
                f"{op.name}: arguments '{other[0]}' and '{p.name}' are the same tensor ({live.ref}) but one is "
                "written in place: an InOut parameter cannot alias another argument"
            )
        seen.setdefault(id(live), (p.name, p.access))
        at = live.type
        if at.devscalar != pt.devscalar:
            raise TilefishTypeError(
                f"{op.name}: argument '{p.name}': expected {pt}, got {at}: a DevScalar (one element in device memory) is only "
                "passed to a DevScalar parameter, and a DevScalar parameter takes nothing else"
            )
        if not at.contiguous and not p.strided:
            raise TilefishTypeError(
                f"{op.name}: argument '{p.name}' is a strided view ({at}); this parameter takes a contiguous tensor: pass the "
                "unsplit tensor, compute from the view (x + 0 is a contiguous copy), or have an extern declare the parameter "
                "Strided[...] (a chunk that does not start on a 16-byte boundary is a view too)"
            )
        if at.rank != pt.rank:
            raise TilefishTypeError(
                f"{op.name}: argument '{p.name}': expected rank {pt.rank} ({pt}), got rank {at.rank} ({at})"
            )
        if not _dtype_equal(pt, at):
            raise TilefishTypeError(
                f"{op.name}: argument '{p.name}': expected dtype {pt.dtype}, got {at.dtype} "
                f"(parameter is {pt}, argument is {at})"
            )
        for axis, (pd, ad) in enumerate(zip(pt.shape, at.shape)):
            unifier.add(p.name, axis, pd, ad, at)
        if p.strided:
            _add_layout_equations(op, p, pt, at, unifier)
        inputs.append(live)
        if p.access == "inout":
            inout_args.append((p, arg, live))
    unifier.solve()

    consts: dict[str, Any] = {}
    for p in sig.consts:
        value = arguments[p.name]
        if value is p.default:
            consts[p.name] = value  # the declared default, validated at declaration
        else:
            consts[p.name] = op.check_const(p, _check_const_arg(op, p, value))

    callee = getattr(op, "graph", None)
    if callee is not None:
        for r in callee.requirements:
            _apply_callee_requirement(op, r, unifier.bindings, ctx)

    out_types: list[TensorType] = []
    for i, r in enumerate(sig.returns):
        out_types.append(unifier.instantiate(r, f"return value {i}"))
    for p in sig.tensor_outputs:
        out_types.append(unifier.instantiate(p.type, f"Out parameter '{p.name}'"))  # type: ignore[arg-type]
    n_results = len(out_types)
    for p, _, _ in inout_args:
        out_types.append(unifier.instantiate(p.type, f"InOut parameter '{p.name}'"))  # type: ignore[arg-type]

    attrs: dict[str, Any] = {"kind": op.kind, **op.node_attrs()}
    if consts:
        attrs["consts"] = consts
    if inout_args:
        attrs["inout"] = tuple(p.name for p, _, _ in inout_args)
    outs = ctx.emit(op.name, inputs, attrs, out_types)
    node = ctx.nodes[-1]
    node.callee = op
    for (p, arg, read), version in zip(inout_args, outs[n_results:]):
        # The call consumes the version it read and replaces it: a temporary, a chunk or an older version
        # is a value, so reading it again raises; a parameter's name follows its newest version.
        read._consumed_by = (node, p.name)
        read._latest = version
        if arg is not read and arg._pinned is None:
            arg._latest = version  # the parameter's name now reads the updated tensor
        ctx.written.update((arg, read))
    results = outs[:n_results]
    if not results:
        return None
    return results[0] if len(results) == 1 else results
