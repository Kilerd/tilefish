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
INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class TensorInfo:
    """What the checks need to know about one tensor.

    ``ptr`` is the address of its first element (``data_ptr()``: the allocation base plus the offset of a view), ``strides`` are the
    element strides, and ``nbytes`` the bytes from ``ptr`` to the end of the last element it reaches (the element count times the
    element size for a dense tensor: the extent of a strided view, which a ``numel`` would under-count, is what aliasing is
    checked with).
    """

    shape: "tuple[int, ...]"
    dtype: str  # str(tensor.dtype), e.g. "torch.bfloat16"
    ptr: int
    contiguous: bool
    device: str
    nbytes: int
    strides: "tuple[int, ...] | None" = None
    itemsize: int = 0


def extent_elements(shape: "Sequence[int]", strides: "Sequence[int]") -> int:
    """One more than the largest element offset a tensor of this shape and these strides reaches (0 for an empty tensor)."""
    if any(n == 0 for n in shape):
        return 0
    return 1 + sum((n - 1) * s for n, s in zip(shape, strides))


def tensor_info(name: str, t: Any) -> TensorInfo:
    """Read the facts off a tensor (anything with torch's ``shape``, ``dtype``, ``data_ptr()``, ...).

    A kernel reads and writes the bytes of the tensor's memory, so a tensor whose memory does not hold what it is is refused here
    (:func:`check_memory`): a negated or conjugated view, whose bytes are not its values, and a tensor whose storage cannot hold the
    elements it reaches (a storage resized or freed under it). What a stand-in tensor (a test's fake) does not have is not checked.
    """
    try:
        shape = tuple(int(d) for d in t.shape)
        itemsize = int(t.element_size())
        try:
            strides: "tuple[int, ...] | None" = tuple(int(s) for s in t.stride())
        except (AttributeError, TypeError):
            strides = None
        nbytes = int(t.numel()) * itemsize if strides is None else extent_elements(shape, strides) * itemsize
        info = TensorInfo(
            shape=shape,
            dtype=str(t.dtype),
            ptr=int(t.data_ptr()),
            contiguous=bool(t.is_contiguous()),
            device=str(t.device),
            nbytes=nbytes,
            strides=strides,
            itemsize=itemsize,
        )
    except (AttributeError, TypeError):
        raise TilefishTypeError(
            f"argument '{name}' must be a torch tensor, got {type(t).__name__} ({t!r})"
        ) from None
    check_memory(name, t, info)
    return info


def check_memory(name: str, t: Any, info: TensorInfo) -> None:
    """Refuse a tensor whose memory is not what a kernel may read and write as its elements.

    * A **negated** or **conjugated** view (``t.is_neg()``, ``t.is_conj()``: ``torch._neg_view(x)``, ``x.conj()`` of a complex tensor, the
      ``.imag`` of one) shares the memory of its base and has torch apply the sign or the conjugation when it is read: its bytes are not its
      values, and a kernel, which reads the bytes, would compute with the wrong ones (and write a result torch then negates again).
      ``resolve_neg()`` / ``resolve_conj()`` give a tensor that holds its values.
    * A tensor with elements whose data pointer is 0: its storage was resized to nothing or freed (``untyped_storage().resize_(0)``), and 0
      passes every alignment check.
    * A tensor whose storage is too small for it: the storage offset and the furthest element it reaches (its extent, by its strides), in
      bytes, beyond ``untyped_storage().nbytes()``. Torch does not check that a tensor fits its storage once the storage is resized under it,
      and a kernel would read and write past its end.

    Each check needs the methods torch has (``is_neg``, ``is_conj``; ``untyped_storage`` and ``storage_offset`` for the last two, which every
    tensor of the torch tilefish runs with has): a stand-in without them (a test's fake, whose address is any number) is not held to that check.
    An empty tensor has no memory to check.
    """
    for flag, what, fix in (("is_neg", "a negated view", "resolve_neg"), ("is_conj", "a conjugated view", "resolve_conj")):
        test = getattr(t, flag, None)
        if callable(test) and bool(test()):
            raise TilefishTypeError(
                f"argument '{name}' is {what} ({flag}() is True): torch applies the {'sign' if flag == 'is_neg' else 'conjugation'} when the "
                f"tensor is read, so its memory does not hold its values, and a kernel reads the memory; pass {name}.{fix}()"
            )
    storage, offset = getattr(t, "untyped_storage", None), getattr(t, "storage_offset", None)
    if 0 in info.shape or not (callable(storage) and callable(offset)):
        return
    if info.ptr == 0:
        raise TilefishTypeError(
            f"argument '{name}' has {math.prod(info.shape)} elements (shape {list(info.shape)}) but its data pointer is 0: its storage was "
            "resized to nothing or freed, and a kernel would read and write at address 0"
        )
    try:
        held, start = int(storage().nbytes()), int(offset())
    except (RuntimeError, NotImplementedError):  # (a tensor subclass with no storage of its own: the pointer is what there is)
        return
    reached = start * info.itemsize + info.nbytes
    if reached > held:
        raise TilefishTypeError(
            f"argument '{name}' reaches {reached} bytes into its storage (from storage offset {start}, {info.nbytes} bytes of elements of "
            f"{info.itemsize} bytes, shape {list(info.shape)}), but its storage holds {held} bytes: the storage was resized under it, and a "
            "kernel would read and write past its end"
        )


def canonical_strides(shape: "Sequence[int]", strides: "Sequence[int] | None") -> "tuple[int, ...]":
    """The strides of a tensor with those of its size-1 axes made canonical (the stride of a size-1 axis is never used).

    A size-1 axis gets the stride a dense tensor would have there (``1`` for the last axis, else the size of the next axis times
    its stride), so that two views that differ only there bind the same values, and an ``unsqueeze`` or an ``expand`` does not pass a
    stride that no kernel reads (and that might not fit an ``i32``). ``None`` strides (a fake tensor) are the dense row-major ones.
    """
    n = len(shape)
    if strides is None:
        strides, acc = [0] * n, 1
        for i in range(n - 1, -1, -1):
            strides[i] = acc
            acc *= shape[i]
    out = list(strides)
    for i in range(n - 1, -1, -1):
        if shape[i] == 1:
            out[i] = 1 if i == n - 1 else shape[i + 1] * out[i + 1]
    return tuple(out)


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
    for p in sig.params:
        if p.kind == "const":
            if p.name in explicit:
                consts[p.name] = bound.arguments[p.name]
            continue
        if p.kind != "tensor" or p.access == "out":
            continue
        t = bound.arguments[p.name]
        info = tensor_info(p.name, t)
        check_tensor(p, info)  # in argument order: the first argument that is wrong is the one reported
        tensors[p.name] = t
        infos[p.name] = info
    return Call(tensors, infos, consts, bind_infos(op_name, sig, infos))


def bind_infos(op_name: str, sig: "Signature", infos: "Mapping[str, TensorInfo]") -> "dict[Dim, int]":
    """Check the supplied tensors (``infos``: every ``In`` and ``InOut`` tensor parameter of ``sig``, by name) against their declared types
    and unify their shapes (and, for a ``Strided[...]`` parameter, their strides) into dimension values.

    This is the part of :func:`bind_call` that needs no tensor, only facts about one: a plan binds the dimensions of a step from the shapes
    and strides of the views it launches with, which are not torch tensors.
    """
    equations: list[tuple[str, int, DimExpr, int, str, str]] = []
    for p in sig.params:
        if p.kind != "tensor" or p.access == "out":
            continue
        info = infos[p.name]
        check_tensor(p, info)
        assert p.type is not None
        for axis, (pattern, actual) in enumerate(zip(p.type.shape, info.shape)):
            equations.append((p.name, axis, pattern, actual, str(p.type), "dimension"))
        if p.strided:  # the strides of the view bind the stride dimensions (and must be the fixed ones where the signature fixes them)
            assert p.type.strides is not None
            actual_strides = canonical_strides(info.shape, info.strides)
            for axis, (pattern, actual) in enumerate(zip(p.type.strides, actual_strides)):
                if pattern is not None:
                    equations.append((p.name, axis, pattern, actual, str(p), "stride"))
    return solve_dims(op_name, equations)


def check_tensor(p: "Param", info: TensorInfo) -> None:
    """Dtype, rank and contiguity of one supplied tensor against its declared type.

    A ``Strided[...]`` parameter takes any strides (the view's strides are bound like dimensions); a ``DevScalar`` takes a tensor of
    one element, of shape ``()`` or ``(1,)``.
    """
    ptype: "TensorType" = p.type  # type: ignore[assignment]
    want = TORCH_DTYPE_NAMES[tf_dtype_name(ptype.dtype, f"parameter '{p.name}'")]
    if info.dtype != want:
        raise TilefishPreconditionError(
            f"argument '{p.name}' has dtype {info.dtype}, but the signature requires {ptype.dtype} ({want})",
            kind="dtype",
            arg=p.name,
        )
    if ptype.devscalar:
        if info.shape not in ((), (1,)):
            raise TilefishPreconditionError(
                f"argument '{p.name}' has shape {list(info.shape)}, but the signature requires {ptype}: one element in device memory "
                "(a tensor of shape [] or [1])",
                kind="shape",
                arg=p.name,
            )
        return
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
    if p.strided:
        if info.strides is not None and any(s < 0 for s in info.strides):
            raise TilefishPreconditionError(
                f"argument '{p.name}' has a negative stride (strides {list(info.strides)}); a view must walk forward in memory",
                kind="stride",
                arg=p.name,
            )
        return
    if not info.contiguous:  # every tensor that is not Strided[...] nor a DevScalar, whatever its type says (a declaration refuses any other type)
        raise TilefishPreconditionError(
            f"argument '{p.name}' is not contiguous (row-major, no gaps); this parameter takes a contiguous tensor: pass "
            ".contiguous(), or have the extern declare it Strided[...]",
            kind="contiguous",
            arg=p.name,
        )


def eval_dim(expr: DimExpr, dims: "Mapping[Dim, int]") -> "int | None":
    """The value of a dimension expression under ``dims``, or ``None`` while it mentions an unbound dimension."""
    value = expr.subst(dims)
    return value.value if value.is_static else None


def solve_dims(op_name: str, equations: "Sequence[tuple[Any, ...]]") -> "dict[Dim, int]":
    """Bind every dimension that the actual shapes determine, and check every shape equation.

    An equation is ``(argument, axis, pattern, actual size, type text)`` and, for the strides of a ``Strided[...]``
    argument, ``(argument, axis, pattern, actual stride, type text, "stride")``. A pattern with one unbound
    dimension is solved (``3072*M`` against 6144 gives ``M = 2``, and fails when it is not a whole
    number); a pattern that is fully bound is checked. Whatever is left over is ambiguous.
    """
    dims: dict[Dim, int] = {}
    pending = [(*eq, "dimension") if len(eq) == 5 else tuple(eq) for eq in equations]
    while pending:
        rest = [eq for eq in pending if not _solve_one(op_name, eq, dims)]
        if len(rest) == len(pending):
            pname, axis, pattern, actual, _, what = rest[0]
            raise TilefishTypeError(
                f"{op_name}: argument '{pname}': cannot infer the dimensions in {pattern} ({_what(what, axis)}, value "
                f"{actual}); the signature is ambiguous at this call"
            )
        pending = rest
    return dims


def _what(what: str, axis: int) -> str:
    return f"dimension {axis}" if what == "dimension" else f"the stride of axis {axis}"


def _solve_one(op_name: str, eq: "tuple[Any, ...]", dims: "dict[Dim, int]") -> bool:
    pname, axis, pattern, actual, ptype, what = eq
    unbound = [d for d in pattern.free_dims() if d not in dims]
    if not unbound:
        expected = eval_dim(pattern, dims)
        if expected != actual:
            static = pattern.folded().is_static
            shown = f"{pattern} = {expected}" if not static else f"{expected}"
            note = " (the kernel loads along that axis contiguously: pass a view whose last axis has stride 1)" if what == "stride" else ""
            raise TilefishPreconditionError(
                f"{op_name}: argument '{pname}' ({ptype}): {_what(what, axis)} is {actual}, but the signature requires {shown}{note}",
                kind=("stride" if what == "stride" else "static_size") if static else "shape",
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
    cannot = f"{op_name}: argument '{pname}' ({ptype}): {_what(what, axis)} is {actual}, which cannot match {pattern}"
    if target % coef:
        raise TilefishPreconditionError(f"{cannot} (would need {var} = {target}/{coef})", kind="shape", arg=pname)
    value = target // coef
    if value < (1 if what == "dimension" else 0):  # a stride can be 0 (an expanded view)
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
        elif pre.kind == "max_extent":  # a strided pointer of a kernel that indexes with i32: the largest offset it reaches
            info = infos[pre.arg]  # type: ignore[index]
            extent = extent_elements(info.shape, canonical_strides(info.shape, info.strides))
            if extent > pre.value:  # type: ignore[operator]  # like max_elements: the view may span at most that many elements
                raise TilefishPreconditionError(
                    f"argument '{pre.arg}' (shape {list(info.shape)}, strides {list(canonical_strides(info.shape, info.strides))}) spans "
                    f"{extent} elements, more than the {pre.value} that the i32 offsets of '{spec.stem}' reach",
                    kind="max_extent",
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


def check_scalar_ranges(spec: "CompileSpec", scalars: "Mapping[str, int]") -> None:
    """Every runtime scalar must fit the type the binary takes it as: ``i32`` (or ``i64`` for a stride of a kernel with ``index_bits=64``), and not be negative."""
    types = {a.name: a.type for a in spec.args if a.kind == "scalar"}
    for name, v in scalars.items():
        wide = types.get(name) == "i64"
        if not 0 <= v <= (INT64_MAX if wide else INT32_MAX):
            raise TilefishPreconditionError(
                f"runtime argument '{name}' is {v}, which does not fit the {'64' if wide else '32'}-bit signed integer the binary takes",
                kind="shape",
                arg=name,
            )


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
    compiled = {a.name: a.value for a in spec.args if a.kind in ("constexpr", "fixed")}  # a fixed argument of a prebuilt binary likewise
    fixed = {a.name for a in spec.args if a.kind == "fixed"}
    for name, value in explicit.items():
        if name not in compiled:
            continue
        want = compiled[name]
        if isinstance(value, DimExpr):
            value = value.value if value.is_static else value
        same = value == want and isinstance(value, bool) == isinstance(want, bool)
        if not same and name in fixed:
            raise TilefishTypeError(
                f"argument '{name}' of '{spec.stem}' is fixed to {want!r} (its value is in the manifest and passed at every launch); it "
                f"cannot be changed to {value!r} at a call (change the default in the declaration and export again)"
            )
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
