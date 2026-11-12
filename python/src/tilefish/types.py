"""Tensor types and the annotation vocabulary: ``Tensor``, ``InOut``, ``Out``, ``const``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union

from .dims import DimExpr, as_dim
from .dtypes import BlockScaled, DType
from .errors import TilefishDeclarationError, TilefishTypeError

AnyDType = Union[DType, BlockScaled]


class TensorType:
    """The type of a tensor: a shape of dimension expressions and a dtype.

    Tensors are contiguous and row-major; the type has no stride component
    yet. All comparison goes through :meth:`_key`, so a layout can be added to
    the key later without touching callers.
    """

    __slots__ = ("shape", "dtype")

    shape: tuple[DimExpr, ...]
    dtype: AnyDType

    def __init__(
        self,
        shape: Sequence[Any],
        dtype: AnyDType,
        *,
        known: "Mapping[DimExpr, int] | None" = None,
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
        object.__setattr__(self, "shape", tuple(dims))
        object.__setattr__(self, "dtype", dtype)
        if isinstance(dtype, BlockScaled):
            dtype.check_shape(self.shape, known)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("TensorType is immutable")

    @property
    def rank(self) -> int:
        return len(self.shape)

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
        return TensorType(self.shape, dtype)

    def with_shape(self, shape: Sequence[Any]) -> "TensorType":
        return TensorType(shape, self.dtype)

    def free_dims(self) -> frozenset:
        out: frozenset = frozenset()
        for d in self.shape:
            out |= d.free_dims()
        return out

    def _key(self) -> tuple:
        dtype = self.dtype.normalized(self.rank) if isinstance(self.dtype, BlockScaled) else self.dtype
        return (self.shape, dtype)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TensorType):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return "Tensor[" + ", ".join([*(str(d) for d in self.shape), str(self.dtype)]) + "]"

    __str__ = __repr__


class Tensor:
    """``Tensor[d0, ..., dtype]`` builds a :class:`TensorType`."""

    def __class_getitem__(cls, item: Any) -> TensorType:
        items = item if isinstance(item, tuple) else (item,)
        return TensorType(items[:-1], items[-1])


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
        if not isinstance(item, TensorType):
            raise TilefishDeclarationError(
                f"{cls.__name__}[...] wraps a tensor type such as Tensor[M, D, bf16], got {item!r}"
            )
        return Marked(cls.mode, item)


class InOut(_AccessMarker):
    """``InOut[Tensor[...]]``: the operator reads and writes this tensor in place."""

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
