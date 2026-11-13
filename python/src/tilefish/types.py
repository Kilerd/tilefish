"""Tensor types and the annotation vocabulary: ``Tensor``, ``InOut``, ``Out``, ``const``, ``Strided``, ``StrideOf``, ``DevScalar``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union

from .dims import DimExpr, as_dim
from .dtypes import BlockScaled, DType
from .errors import TilefishDeclarationError, TilefishTypeError

AnyDType = Union[DType, BlockScaled]


def _mul(a: "DimExpr | None", b: "DimExpr | None") -> "DimExpr | None":
    """Product of two affine dimensions, or ``None`` when it is not affine (two symbolic factors) or either is unknown."""
    if a is None or b is None:
        return None
    try:
        return a * b
    except TilefishTypeError:
        return None


def product(dims: "Sequence[DimExpr]") -> "DimExpr | None":
    """The product of dimensions, or ``None`` when it is not an affine expression."""
    out: "DimExpr | None" = DimExpr.const(1)
    for d in dims:
        out = _mul(out, d)
    return out


def row_major_strides(shape: "Sequence[DimExpr]") -> "tuple[DimExpr | None, ...]":
    """The element strides of a dense row-major tensor of ``shape``; an entry is ``None`` where it is not an affine expression."""
    out: "list[DimExpr | None]" = []
    running: "DimExpr | None" = DimExpr.const(1)
    for d in reversed(tuple(shape)):
        out.append(running)
        running = _mul(running, d)
    out.reverse()
    return tuple(out)


def starts_aligned(offset: "DimExpr | None", bits: int, known: "Mapping[DimExpr, int] | None" = None) -> bool:
    """Whether ``offset`` elements of ``bits`` bits provably span a multiple of 16 bytes."""
    if offset is None:
        return False
    return offset.is_divisible_by(128 // math.gcd(bits, 128), known)


def _is_one(d: DimExpr) -> bool:
    return d.is_static and d.value == 1


def _dense(shape: "tuple[DimExpr, ...]", strides: "tuple[DimExpr | None, ...]") -> bool:
    """Whether explicit ``strides`` are those of a dense row-major tensor of ``shape`` (the stride of a size-1 axis does not matter)."""
    want = row_major_strides(shape)
    for d, got, expected in zip(shape, strides, want):
        if _is_one(d):
            continue
        if got is None or expected is None or got != expected:
            return False
    return True


class TensorType:
    """The type of a tensor: a shape of dimension expressions, a dtype and a layout.

    The layout is ``strides`` (element strides, dimension expressions) and ``offset`` (the element offset of the first element
    from the start of the allocation the tensor lives in). A tensor that is dense and row-major has ``strides is None``, which is
    the default; ``offset`` is ``0`` for a tensor that owns its allocation. A *view* (``split``, ``permute``, ``select``,
    ``reshape``) shares the allocation of its parent and has the layout the view operation says. A stride or an offset that is not
    an affine expression (a product of two symbolic dimensions) is *unknown* (``None``), and a type with an unknown layout can
    flow into traced elementwise and reduction ops but cannot be passed to a declared operator.

    ``contiguous`` says the tensor is dense **and** provably starts on a 16-byte boundary: it is what every declared operator's
    pointer arguments are promised to be (the contract of a pointer), so such a tensor can be passed to a parameter written
    ``Tensor[...]``; anything else is a view and can only be passed to an operator that opts in with ``Strided[...]`` (extern
    operators only). Equality is by shape, dtype and whether the type is contiguous: two views of the same shape and dtype
    are equal whatever their strides and offsets (a contiguous tensor can be treated as a fresh allocation, and no operation
    takes a view of one particular layout except through ``Strided[...]``, which binds the layout like it binds the shape).
    :meth:`same_layout` compares layouts, an unknown stride or offset (``as_view()``) agreeing with anything. All comparison
    goes through :meth:`_key`, so a layout can be added to it without touching callers.

    ``devscalar`` marks the type of ``DevScalar[dtype]``: one element in device memory, rank 0, read with ``tilefish.load``.
    """

    __slots__ = ("shape", "dtype", "strides", "offset", "contiguous", "devscalar")

    shape: tuple[DimExpr, ...]
    dtype: AnyDType
    strides: "tuple[DimExpr | None, ...] | None"
    offset: "DimExpr | None"
    contiguous: bool
    devscalar: bool

    def __init__(
        self,
        shape: Sequence[Any],
        dtype: AnyDType,
        *,
        known: "Mapping[DimExpr, int] | None" = None,
        contiguous: bool = True,
        strides: "Sequence[Any] | None" = None,
        offset: Any = 0,
        aligned: "bool | None" = None,
        devscalar: bool = False,
        normalize: bool = True,
    ) -> None:
        if not isinstance(dtype, (DType, BlockScaled)):
            raise TilefishDeclarationError(
                f"the last entry of Tensor[...] must be a dtype (bf16, f32, BlockScaled(...), ...), got {dtype!r}"
            )
        dims = []
        for i, d in enumerate(shape):
            try:
                dim = as_dim(d)
            except TilefishTypeError:
                raise TilefishDeclarationError(
                    f"Tensor shape entry {i} must be an int or a Dim expression, got {d!r}"
                ) from None
            if dim.is_static and dim.value < 1:
                raise TilefishDeclarationError(f"Tensor shape entry {i} must be positive, got {dim.value}")
            dims.append(dim)
        shape_t = tuple(dims)
        if devscalar and (shape_t or isinstance(dtype, BlockScaled)):
            raise TilefishDeclarationError("DevScalar[...] is one element of a plain dtype: it has rank 0")
        explicit: "tuple[DimExpr | None, ...] | None" = None
        off: "DimExpr | None"
        if not contiguous and strides is None and isinstance(offset, int) and offset == 0 and aligned is None:
            # TensorType(..., contiguous=False): a view whose layout is not known
            explicit = tuple(None for _ in shape_t)
            off = None
        else:
            if strides is not None:
                if len(strides) != len(shape_t):
                    raise TilefishDeclarationError(
                        f"strides {list(strides)!r} has {len(strides)} entries for a tensor of rank {len(shape_t)}"
                    )
                explicit = tuple(None if s is None else as_dim(s).folded() for s in strides)
                if normalize and all(s is not None for s in explicit) and _dense(shape_t, explicit):
                    explicit = None  # the strides of a dense row-major tensor are the default
            off = None if offset is None else as_dim(offset).folded()
        if aligned is None:
            if off is not None and off.is_static and off.value == 0:
                aligned = True
            elif isinstance(dtype, BlockScaled) or off is None:
                aligned = False  # a block-scaled chunk is aligned by the caller (data and scales both), which knows both offsets
            else:
                aligned = starts_aligned(off, dtype.bits, known)
        object.__setattr__(self, "shape", shape_t)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "strides", explicit)
        object.__setattr__(self, "offset", off)
        object.__setattr__(self, "contiguous", explicit is None and bool(aligned))
        object.__setattr__(self, "devscalar", bool(devscalar))
        if isinstance(dtype, BlockScaled):
            dtype.check_shape(self.shape, known)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("TensorType is immutable")

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def is_dense(self) -> bool:
        """Whether the strides are those of a dense row-major tensor (a dense tensor may still start at an offset)."""
        return self.strides is None

    @property
    def is_opaque(self) -> bool:
        """Whether nothing is known of the layout (what :meth:`as_view` makes)."""
        return self.strides is not None and all(s is None for s in self.strides) and self.offset is None

    def layout_strides(self) -> "tuple[DimExpr | None, ...]":
        """The element strides, spelled out (the row-major ones for a dense tensor); an entry is ``None`` when unknown."""
        return self.strides if self.strides is not None else row_major_strides(self.shape)

    @property
    def scale_shape(self) -> "tuple[DimExpr, ...] | None":
        """Shape of the scale tensor for a block-scaled dtype, else ``None``."""
        if isinstance(self.dtype, BlockScaled):
            return self.dtype.scale_shape(self.shape)
        return None

    @property
    def scale_type(self) -> "TensorType | None":
        """Type of the scale tensor for a block-scaled dtype, else ``None``."""
        if isinstance(self.dtype, BlockScaled):
            return TensorType(self.dtype.scale_shape(self.shape), self.dtype.scale)
        return None

    def with_dtype(self, dtype: AnyDType) -> "TensorType":
        """The same shape with another dtype, as a fresh contiguous tensor."""
        return TensorType(self.shape, dtype)

    def with_shape(self, shape: Sequence[Any]) -> "TensorType":
        """Another shape of the same dtype, as a fresh contiguous tensor."""
        return TensorType(shape, self.dtype)

    def as_view(self) -> "TensorType":
        """The same shape and dtype as a view of unknown layout (it agrees with any view of that shape in comparisons)."""
        return TensorType(self.shape, self.dtype, contiguous=False)

    def materialized(self) -> "TensorType":
        """The same shape and dtype as a fresh contiguous tensor."""
        return self if self.contiguous and not self.devscalar else TensorType(self.shape, self.dtype)

    def free_dims(self) -> frozenset:
        out: frozenset = frozenset()
        for d in self.shape:
            out |= d.free_dims()
        for s in self.strides or ():
            if s is not None:
                out |= s.free_dims()
        if self.offset is not None:
            out |= self.offset.free_dims()
        return out

    def layout_text(self) -> str:
        """``strides (...), offset ...``, for messages (``?`` where unknown)."""
        strides = ", ".join("?" if s is None else str(s) for s in self.layout_strides())
        offset = "?" if self.offset is None else str(self.offset)
        return f"strides ({strides}), offset {offset}"

    def _key(self) -> tuple:
        dtype = self.dtype.normalized(self.rank) if isinstance(self.dtype, BlockScaled) else self.dtype
        return (self.shape, dtype, self.contiguous, self.devscalar)

    def same_layout(self, other: "TensorType") -> bool:
        """Whether two types can describe the same strides and offset (an unknown stride or offset agrees with anything)."""
        if self.is_opaque or other.is_opaque:
            return True
        for a, b in zip(self.layout_strides(), other.layout_strides()):
            if a is not None and b is not None and a != b:
                return False
        return not (self.offset is not None and other.offset is not None and self.offset != other.offset)

    def __reduce__(self) -> Any:
        return (_rebuild_tensor_type, (self.shape, self.dtype, self.strides, self.offset, self.contiguous, self.devscalar))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TensorType):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        if self.devscalar:
            return f"DevScalar[{self.dtype}]"
        head = "Tensor" if self.contiguous else "View"
        return head + "[" + ", ".join([*(str(d) for d in self.shape), str(self.dtype)]) + "]"

    __str__ = __repr__


def _rebuild_tensor_type(
    shape: Any, dtype: AnyDType, strides: Any, offset: Any, contiguous: bool, devscalar: bool = False
) -> TensorType:
    return TensorType(shape, dtype, strides=strides, offset=offset, aligned=contiguous, devscalar=devscalar, normalize=False)


class Tensor:
    """``Tensor[d0, ..., dtype]`` builds a :class:`TensorType`."""

    def __class_getitem__(cls, item: Any) -> TensorType:
        items = item if isinstance(item, tuple) else (item,)
        return TensorType(items[:-1], items[-1])


class DevScalar:
    """``DevScalar[dtype]``: one element of ``dtype`` in device memory, e.g. a request-dependent token count ``n_tok: DevScalar[i32]``.

    It is a pointer to one element (rank 0): in an ABI a pointer parameter with shape ``[]``, and an argument is a tensor of one
    element. A traced body reads it with ``tilefish.load(s)``, which is a rank-0 tensor of ``dtype`` (a lowered kernel reads it with one
    ``tl.load(ptr)``, broadcast). A ``DevScalar`` is read-only and cannot be combined with tensors until it is loaded.
    """

    def __class_getitem__(cls, item: Any) -> TensorType:
        if not isinstance(item, DType):
            raise TilefishDeclarationError(f"DevScalar[...] takes a scalar dtype such as i32 or f32, got {item!r}")
        return TensorType((), item, devscalar=True)


@dataclass(frozen=True)
class StridedType:
    """What ``Strided[Tensor[...]]`` evaluates to: a tensor parameter that accepts a strided view (see :class:`Strided`)."""

    type: TensorType
    align: "int | None" = None  # 16: the pointer of the view is compiled as 16-byte aligned; None: only its element alignment is assumed

    def __repr__(self) -> str:
        return f"Strided[{self.type!r}" + (f", {self.align}" if self.align else "") + "]"


class Strided:
    """``Strided[Tensor[...]]``: an extern parameter that accepts a strided view, not only a contiguous tensor.

    Without it a parameter takes contiguous, 16-byte aligned tensors only. With it the kernel is handed the pointer of the view,
    ``base + offset * itemsize``, and the strides of the view as ordinary scalar parameters that the stub declares, in the
    kernel's argument order, with :class:`StrideOf`::

        @tilefish.extern(triton=head_copy, grid=lambda S, H: (S, H))
        def head_copy(x: Strided[Tensor[S, H, D, bf16]], y: Out[Tensor[S, H, D, bf16]],
                      x_s0: StrideOf("x", 0), x_s1: StrideOf("x", 1), *, D_: const = 128): ...

    An axis that has no ``StrideOf`` parameter must be the last one, and then its stride is 1 (the kernel loads along it
    contiguously); a view that does not have unit stride there is refused. Every other axis is free: the strides are bound at the
    call like dimensions are, and passed as ``i32`` (``i64`` for a kernel that declares ``index_bits=64``), range-checked.

    ``Strided[Tensor[...], 16]`` says the kernel is compiled for a 16-byte aligned pointer (``tt.divisibility=16``): a view whose
    offset in bytes is not provably a multiple of 16 is then refused at a trace call, and at a launch the address is checked. Without
    the ``16`` the pointer is assumed to be aligned to its element only and Triton gets no divisibility hint.

    Only input tensors can be ``Strided`` in this version (an ``Out`` is allocated by the host and is contiguous).
    """

    def __class_getitem__(cls, item: Any) -> StridedType:
        items = item if isinstance(item, tuple) else (item,)
        if len(items) not in (1, 2) or not isinstance(items[0], TensorType):
            raise TilefishDeclarationError(f"Strided[...] wraps a tensor type, such as Strided[Tensor[M, D, bf16]], got {item!r}")
        t = items[0]
        if t.devscalar or isinstance(t.dtype, BlockScaled):
            raise TilefishDeclarationError(f"Strided[{t!r}]: only plain tensors can be strided (not a DevScalar, not a block-scaled tensor)")
        align = None
        if len(items) == 2:
            if isinstance(items[1], bool) or items[1] != 16:
                raise TilefishDeclarationError(f"Strided[..., align]: the alignment must be 16 (bytes), got {items[1]!r}")
            align = 16
        return StridedType(t, align)


@dataclass(frozen=True)
class StrideOf:
    """``StrideOf("x", 0)``: a runtime scalar parameter that carries the element stride of axis 0 of the tensor parameter ``x``.

    The scalar is ``i32`` (``i64`` for an op that declares ``index_bits=64``); the stride is range-checked at a launch. ``x`` must be
    declared ``Strided[...]``. ``divisible_by=k`` says the stride is a multiple of ``k`` elements (it is checked, and a multiple of 16 is
    given to Triton as ``tt.divisibility``).
    """

    arg: str
    axis: int
    divisible_by: "int | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.arg, str) or not self.arg.isidentifier():
            raise TilefishDeclarationError(f"StrideOf: the first argument names a tensor parameter, got {self.arg!r}")
        if isinstance(self.axis, bool) or not isinstance(self.axis, int) or self.axis < 0:
            raise TilefishDeclarationError(f"StrideOf({self.arg!r}, ...): the axis must be a non-negative int, got {self.axis!r}")
        if self.divisible_by is not None and (isinstance(self.divisible_by, bool) or not isinstance(self.divisible_by, int) or self.divisible_by < 1):
            raise TilefishDeclarationError(f"StrideOf: divisible_by must be a positive int, got {self.divisible_by!r}")

    def __repr__(self) -> str:
        extra = f", divisible_by={self.divisible_by}" if self.divisible_by else ""
        return f"StrideOf({self.arg!r}, {self.axis}{extra})"


@dataclass(frozen=True)
class Marked:
    """A tensor annotation with an access mode: ``InOut[...]`` or ``Out[...]``."""

    mode: str  # "inout" | "out"
    type: TensorType

    def __repr__(self) -> str:
        return f"{'InOut' if self.mode == 'inout' else 'Out'}[{self.type!r}]"


class _AccessMarker:
    mode = ""

    def __class_getitem__(cls, item: Any) -> Marked:
        if isinstance(item, StridedType):
            raise TilefishDeclarationError(
                f"{cls.__name__}[Strided[...]]: only input tensors can be strided in this version; an {cls.__name__} tensor is contiguous"
            )
        if not isinstance(item, TensorType):
            raise TilefishDeclarationError(
                f"{cls.__name__}[...] wraps a tensor type such as Tensor[M, D, bf16], got {item!r}"
            )
        if item.devscalar:
            raise TilefishDeclarationError(f"{cls.__name__}[{item!r}]: a DevScalar is read-only")
        return Marked(cls.mode, item)


class InOut(_AccessMarker):
    """``InOut[Tensor[...]]``: the operator reads and writes this tensor in place.

    Tensors have value semantics in a traced body. A call that passes a tensor as an ``InOut`` argument
    *consumes* the version it reads and produces a new one (the call node's last outputs, one per
    ``InOut`` argument; see :class:`~tilefish.ir.Node`):

    * Reading the name of an ``InOut`` *parameter* afterwards reads the new version, so a body sees
      what it has stored or updated.
    * Any other tensor (a temporary, a split chunk, an older version) is an immutable value: reading
      it after the call raises :class:`~tilefish.errors.TilefishTypeError` naming the call; the updated
      tensor is its ``.live`` version.
    * An in-place update of a **split chunk** does not change the chunk's parent. The type calls a chunk
      along a non-leading axis a "view", but it is a value: the parent keeps its own version and the
      update exists only as the chunk's new version (lowering copies; there is no write-through).
    * Passing an ``InOut`` argument that is a read-only input of the enclosing graph is an error.
    """

    mode = "inout"


class Out(_AccessMarker):
    """``Out[Tensor[...]]``: the operator writes this tensor; in a call it is a result."""

    mode = "out"


@dataclass(frozen=True)
class ConstType:
    """A compile-time scalar parameter, optionally with a dtype: ``const`` / ``const[f32]``."""

    dtype: DType | None = None

    def __repr__(self) -> str:
        return "const" if self.dtype is None else f"const[{self.dtype}]"


class const:
    """Annotation for compile-time scalar parameters: ``const`` or ``const[f32]``."""

    def __class_getitem__(cls, item: Any) -> ConstType:
        if not isinstance(item, DType):
            raise TilefishDeclarationError(f"const[...] takes a scalar dtype such as f32 or i32, got {item!r}")
        return ConstType(item)
