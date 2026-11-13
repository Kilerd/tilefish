"""Matching call arguments to a signature, and the checks that must pass before a binary may run.

Everything here works on plain facts about tensors (:class:`TensorInfo`: shape, dtype, address,
contiguity, device) and never imports torch, so the rules can be tested without a GPU. A binary is
compiled for assumptions that the types and the compile spec state: dtypes, static sizes, which
dimension is divisible by what, 16-byte aligned pointers, contiguous row-major tensors, and (for a
bucketed dimension) a size that equals the bucket exactly, with no padding. A launch that breaks one
raises :class:`~tilefish.errors.TilefishPreconditionError` before anything runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .dims import Dim, DimExpr
from .dtypes import BlockScaled
from .errors import TilefishPreconditionError, TilefishRuntimeError, TilefishTypeError

if TYPE_CHECKING:
    from .aot import CompileSpec
    from .signature import Param, Signature
    from .types import TensorType

# the str() of each torch dtype a tilefish dtype is stored as
TORCH_DTYPE_NAMES = {
    "bf16": "torch.bfloat16",
    "f16": "torch.float16",
    "f32": "torch.float32",
    "i32": "torch.int32",
    "i64": "torch.int64",
    "u8": "torch.uint8",
    "e4m3": "torch.float8_e4m3fn",
    "e5m2": "torch.float8_e5m2",
}
INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class TensorInfo:
    """What the checks need to know about one tensor."""

    shape: "tuple[int, ...]"
    dtype: str  # str(tensor.dtype), e.g. "torch.bfloat16"
    ptr: int
    contiguous: bool
    device: str
    nbytes: int


def tensor_info(name: str, t: Any) -> TensorInfo:
    """Read the facts off a tensor (anything with torch's ``shape``, ``dtype``, ``data_ptr()``, ...)."""
    try:
        return TensorInfo(
            shape=tuple(int(d) for d in t.shape),
            dtype=str(t.dtype),
            ptr=int(t.data_ptr()),
            contiguous=bool(t.is_contiguous()),
            device=str(t.device),
            nbytes=int(t.numel()) * int(t.element_size()),
        )
    except (AttributeError, TypeError):
        raise TilefishTypeError(
            f"argument '{name}' must be a torch tensor, got {type(t).__name__} ({t!r})"
        ) from None


def tf_dtype_name(dtype: Any, where: str) -> str:
    if isinstance(dtype, BlockScaled):
        raise TilefishRuntimeError(f"{where} is block-scaled ({dtype!r}); block-scaled tensors cannot be run yet")
    if dtype.name not in TORCH_DTYPE_NAMES:
        raise TilefishRuntimeError(f"{where} has dtype {dtype}, which has no torch storage type here (packed formats cannot be run yet)")
    return dtype.name


# -- binding the call -----------------------------------------------------------------------------------------------------


@dataclass
class Call:
    """A call matched to a signature: the supplied tensors, the explicit consts and the facts about them."""

    tensors: "dict[str, Any]"  # name -> tensor, for the supplied (In and InOut) tensor parameters
    infos: "dict[str, TensorInfo]"
    consts: "dict[str, Any]"  # only the consts the caller wrote out
    dims: "dict[Dim, int]"


def bind_call(op_name: str, sig: "Signature", args: Sequence[Any], kwargs: Mapping[str, Any]) -> Call:
    """Match positional and keyword arguments to the callable parameters of ``sig`` (like a call in a trace).

    ``Out`` and runtime-dimension parameters are not arguments: they are derived. Dtype, rank and
    contiguity of every tensor are checked here; the shapes are unified into dimension values.
    """
    try:
        bound = sig.call_signature.bind(*args, **kwargs)
    except TypeError as e:
        raise TilefishTypeError(f"{op_name}: {e}; signature {sig}") from None
    explicit = set(bound.arguments)
    bound.apply_defaults()
    tensors: dict[str, Any] = {}
    infos: dict[str, TensorInfo] = {}
    consts: dict[str, Any] = {}
    equations: list[tuple[str, int, DimExpr, int, str]] = []
    for p in sig.params:
        if p.kind == "const":
            if p.name in explicit:
                consts[p.name] = bound.arguments[p.name]
            continue
        if p.kind != "tensor" or p.access == "out":
            continue
        t = bound.arguments[p.name]
        info = tensor_info(p.name, t)
        check_tensor(p, info)
        tensors[p.name] = t
        infos[p.name] = info
        assert p.type is not None
        for axis, (pattern, actual) in enumerate(zip(p.type.shape, info.shape)):
            equations.append((p.name, axis, pattern, actual, str(p.type)))
    dims = solve_dims(op_name, equations)
    return Call(tensors, infos, consts, dims)


def check_tensor(p: "Param", info: TensorInfo) -> None:
    """Dtype, rank and contiguity of one supplied tensor against its declared type."""
    ptype: "TensorType" = p.type  # type: ignore[assignment]
    want = TORCH_DTYPE_NAMES[tf_dtype_name(ptype.dtype, f"parameter '{p.name}'")]
    if info.dtype != want:
        raise TilefishPreconditionError(
            f"argument '{p.name}' has dtype {info.dtype}, but the signature requires {ptype.dtype} ({want})",
            kind="dtype",
            arg=p.name,
        )
    if len(info.shape) != ptype.rank:
        raise TilefishPreconditionError(
            f"argument '{p.name}' has rank {len(info.shape)} (shape {list(info.shape)}), but the signature requires "
            f"{ptype} (rank {ptype.rank})",
            kind="rank",
            arg=p.name,
        )
    if 0 in info.shape:
        raise TilefishPreconditionError(
            f"argument '{p.name}' is empty (shape {list(info.shape)}); a binary needs every dimension to be at least 1",
            kind="shape",
            arg=p.name,
        )
    if ptype.contiguous and not info.contiguous:
        raise TilefishPreconditionError(
            f"argument '{p.name}' is not contiguous (row-major, no gaps); strided tensors are not supported yet, "
            "pass .contiguous()",
            kind="contiguous",
            arg=p.name,
        )


def eval_dim(expr: DimExpr, dims: "Mapping[Dim, int]") -> "int | None":
    """The value of a dimension expression under ``dims``, or ``None`` while it mentions an unbound dimension."""
    value = expr.subst(dims)
    return value.value if value.is_static else None


def solve_dims(op_name: str, equations: "Sequence[tuple[str, int, DimExpr, int, str]]") -> "dict[Dim, int]":
    """Bind every dimension that the actual shapes determine, and check every shape equation.

    An equation is ``(argument, axis, pattern, actual size, type text)``. A pattern with one unbound
    dimension is solved (``3072*M`` against 6144 gives ``M = 2``, and fails when it is not a whole
    number); a pattern that is fully bound is checked. Whatever is left over is ambiguous.
    """
    dims: dict[Dim, int] = {}
    pending = list(equations)
    while pending:
        rest = [eq for eq in pending if not _solve_one(op_name, eq, dims)]
        if len(rest) == len(pending):
            pname, axis, pattern, actual, _ = rest[0]
            raise TilefishTypeError(
                f"{op_name}: argument '{pname}': cannot infer the dimensions in {pattern} (dimension {axis}, size "
                f"{actual}); the signature is ambiguous at this call"
            )
        pending = rest
    return dims


def _solve_one(op_name: str, eq: "tuple[str, int, DimExpr, int, str]", dims: "dict[Dim, int]") -> bool:
    pname, axis, pattern, actual, ptype = eq
    unbound = [d for d in pattern.free_dims() if d not in dims]
    if not unbound:
        expected = eval_dim(pattern, dims)
        if expected != actual:
            static = pattern.folded().is_static
            what = f"{pattern} = {expected}" if not static else f"{expected}"
            raise TilefishPreconditionError(
                f"{op_name}: argument '{pname}' ({ptype}): dimension {axis} is {actual}, but the signature requires {what}",
                kind="static_size" if static else "shape",
                arg=pname,
            )
        return True
    if len(unbound) != 1:
        return False
    var = unbound[0]
    coef = pattern.coeff(var)
    if coef == 0:  # the variable only occurs inside a cdiv
        return False
    rest = pattern - var * coef
    base = eval_dim(rest, dims)
    if base is None:
        return False
    target = actual - base
    cannot = f"{op_name}: argument '{pname}' ({ptype}): dimension {axis} is {actual}, which cannot match {pattern}"
    if target % coef:
        raise TilefishPreconditionError(f"{cannot} (would need {var} = {target}/{coef})", kind="shape", arg=pname)
    value = target // coef
    if value < 1:
        raise TilefishPreconditionError(f"{cannot} (would need {var} = {value})", kind="shape", arg=pname)
    dims[var] = value
    return True


def check_dim_constraints(dims: "Mapping[Dim, int]", *, buckets: bool = True) -> None:
    """The constraints a ``Dim`` declares: ``divisible_by`` and (with ``buckets``) the bucket list."""
    for d, v in dims.items():
        if d.divisible_by and v % d.divisible_by:
            raise TilefishPreconditionError(
                f"dimension {d.name} is {v}, but it is declared divisible_by={d.divisible_by}", kind="divisible", arg=d.name
            )
        if buckets and d.buckets and v not in d.buckets:
            raise TilefishPreconditionError(
                f"dimension {d.name} is {v}, which is not one of its buckets {list(d.buckets)}; a bucketed dimension "
                "must equal a bucket exactly (there is no padding: pad or slice in the host)",
                kind="bucket",
                arg=d.name,
            )


def dims_by_name(dims: "Mapping[Dim, int]") -> "dict[str, int]":
    return {d.name: v for d, v in dims.items()}


# -- choosing and checking the binary ---------------------------------------------------------------------------------------


def select_spec(op_name: str, specs: "Sequence[CompileSpec]", dims: "Mapping[Dim, int]") -> "CompileSpec":
    """The spec whose bucket values equal the actual sizes (the only spec of an unbucketed operator)."""
    by_name = dims_by_name(dims)
    for spec in specs:
        if all(by_name.get(name) == value for name, value in spec.buckets.items()):
            return spec
    wanted = sorted({name for s in specs for name in s.buckets})
    got = {n: by_name.get(n) for n in wanted}
    listing = {n: sorted({s.buckets[n] for s in specs}) for n in wanted}
    raise TilefishPreconditionError(
        f"{op_name}: no binary was built for {got}: a bucketed dimension must equal a bucket exactly, and the "
        f"buckets are {listing}. There is no padding: pad or slice in the host to a bucket size",
        kind="bucket",
        arg=wanted[0] if wanted else None,
    )


def check_spec_preconditions(
    spec: "CompileSpec",
    infos: "Mapping[str, TensorInfo]",
    scalars: "Mapping[str, int]",
    dims: "Mapping[Dim, int]",
    *,
    alignment: bool = True,
    buckets: bool = True,
) -> None:
    """Every precondition the compile spec lists, against the actual tensors and runtime scalars."""
    by_name = dims_by_name(dims)
    for pre in spec.preconditions:
        if pre.kind == "pointer_alignment":
            if alignment and infos[pre.arg].ptr % pre.bytes:  # type: ignore[index,operator]
                raise TilefishPreconditionError(
                    f"argument '{pre.arg}' is at address {infos[pre.arg].ptr:#x}, which is not aligned to "  # type: ignore[index]
                    f"{pre.bytes} bytes; every pointer of '{spec.stem}' is compiled as {pre.bytes}-byte aligned "
                    "(allocate it aligned, or copy it)",
                    kind="pointer_alignment",
                    arg=pre.arg,
                )
        elif pre.kind == "divisible":
            value = scalars[pre.arg]  # type: ignore[index]
            if value % pre.by:  # type: ignore[operator]
                raise TilefishPreconditionError(
                    f"runtime argument '{pre.arg}' ({pre.dim}) is {value}, which is not a multiple of {pre.by}; "
                    f"'{spec.stem}' is compiled with that assumption",
                    kind="divisible",
                    arg=pre.arg,
                )
        elif pre.kind == "static_size":
            actual = infos[pre.arg].shape[pre.axis]  # type: ignore[index]
            if actual != pre.value:
                raise TilefishPreconditionError(
                    f"argument '{pre.arg}': dimension {pre.axis} is {actual}, but '{spec.stem}' is built for {pre.value}",
                    kind="static_size",
                    arg=pre.arg,
                )
        elif pre.kind == "max_elements":  # the kernel indexes with i32 (a generated one, or an extern that did not say index_bits=64)
            count = math.prod(infos[pre.arg].shape)  # type: ignore[index]
            if count > pre.value:  # type: ignore[operator]
                raise TilefishPreconditionError(
                    f"argument '{pre.arg}' has {count} elements (shape {list(infos[pre.arg].shape)}), more than the "  # type: ignore[index]
                    f"{pre.value} that the i32 offsets of '{spec.stem}' reach",
                    kind="max_elements",
                    arg=pre.arg,
                )
        elif pre.kind == "bucket":
            if buckets and by_name.get(pre.dim) != pre.value:  # type: ignore[arg-type]
                raise TilefishPreconditionError(
                    f"dimension {pre.dim} is {by_name.get(pre.dim)}, but '{spec.stem}' is built for exactly "
                    f"{pre.value} (buckets {list(pre.buckets or ())}); there is no padding",
                    kind="bucket",
                    arg=pre.dim,
                )
        else:  # a precondition this runtime cannot check must not be skipped
            raise TilefishRuntimeError(f"'{spec.stem}' has a precondition of kind {pre.kind!r}, which this runtime does not know")


def check_i32(scalars: "Mapping[str, int]") -> None:
    for name, v in scalars.items():
        if not 0 <= v <= INT32_MAX:
            raise TilefishPreconditionError(
                f"runtime argument '{name}' is {v}, which does not fit the 32-bit signed integer the binary takes",
                kind="shape",
                arg=name,
            )


def check_no_alias(infos: "Mapping[str, TensorInfo]", written: "set[str]") -> None:
    """A tensor that is written in place must not overlap any other argument."""
    names = list(infos)
    for w in names:
        if w not in written:
            continue
        a = infos[w]
        for o in names:
            if o == w:
                continue
            b = infos[o]
            if a.ptr < b.ptr + b.nbytes and b.ptr < a.ptr + a.nbytes:
                raise TilefishPreconditionError(
                    f"argument '{w}' is written in place but overlaps argument '{o}': an InOut or Out tensor cannot "
                    "alias another argument",
                    kind="alias",
                    arg=w,
                )


def check_consts(spec: "CompileSpec", explicit: "Mapping[str, Any]") -> None:
    """A constexpr is compiled into the binary: a call may restate its value, not change it."""
    compiled = {a.name: a.value for a in spec.args if a.kind == "constexpr"}
    for name, value in explicit.items():
        if name not in compiled:
            continue
        want = compiled[name]
        if isinstance(value, DimExpr):
            value = value.value if value.is_static else value
        same = value == want and isinstance(value, bool) == isinstance(want, bool)
        if not same:
            raise TilefishTypeError(
                f"constexpr '{name}' is compiled into '{spec.stem}' as {want!r}; it cannot be changed to {value!r} at a "
                "call (change the default in the declaration and rebuild)"
            )


def eval_grid(spec: "CompileSpec", grid: "Sequence[DimExpr]", dims: "Mapping[Dim, int]") -> "tuple[int, ...]":
    """The launch grid for this call: the declared grid expressions under the bucket values and the actual dimensions."""
    out = []
    for g in grid:
        v = eval_dim(g, dims)
        if v is None:
            free = sorted(d.name for d in g.free_dims() if d not in dims)
            raise TilefishRuntimeError(f"the grid {spec.grid} of '{spec.stem}' mentions {free}, which no argument determines")
        if v < 1:
            raise TilefishRuntimeError(f"the grid {spec.grid} of '{spec.stem}' is {v} here; every grid size must be at least 1")
        out.append(v)
    return tuple(out)
