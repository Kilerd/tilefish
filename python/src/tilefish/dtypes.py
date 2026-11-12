"""Element types and block-scaled quantized formats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from .errors import TilefishDeclarationError, TilefishTypeError

if TYPE_CHECKING:
    from .dims import DimExpr


@dataclass(frozen=True)
class DType:
    """A scalar element type.

    ``triton_ptr`` is the spelling of a pointer to this type in a Triton
    signature, ``triton_scalar`` the spelling of a by-value scalar (``None``
    when Triton has no scalar of this type). ``bits`` is the width of one
    element; sub-byte types (``e2m1``) are stored packed, ``elems_per_byte``
    elements to a byte, and their pointer type is the pointer to the storage
    byte (``*u8``).
    """

    name: str
    bits: int
    kind: str  # "float" | "int" | "uint"
    triton_ptr: str
    triton_scalar: str | None = None
    exp_bits: int = 0  # floats only
    man_bits: int = 0  # floats only (explicit mantissa bits)
    signed: bool = True
    computable: bool = True  # arithmetic is allowed; storage formats must be cast first

    @property
    def is_float(self) -> bool:
        return self.kind == "float"

    @property
    def is_int(self) -> bool:
        return self.kind in ("int", "uint")

    @property
    def elems_per_byte(self) -> int:
        return 8 // self.bits if self.bits < 8 else 1

    @property
    def storage_bits(self) -> int:
        return max(self.bits, 8)

    def int_range(self) -> tuple[int, int]:
        if not self.is_int:
            raise TilefishTypeError(f"{self.name} is not an integer type")
        if self.signed:
            return -(1 << (self.bits - 1)), (1 << (self.bits - 1)) - 1
        return 0, (1 << self.bits) - 1

    def can_represent(self, src: "DType") -> bool:
        """True when every value of ``src`` is exactly representable in ``self``."""
        if src == self:
            return True
        if self.is_float:
            if src.is_float:
                return (
                    (self.signed or not src.signed)
                    and self.exp_bits >= src.exp_bits
                    and self.man_bits >= src.man_bits
                )
            magnitude_bits = src.bits - (1 if src.signed else 0)
            return (self.signed or not src.signed) and magnitude_bits <= self.man_bits + 1
        # integer destination
        if src.is_float:
            return False
        if src.signed == self.signed:
            return self.bits >= src.bits
        if not src.signed and self.signed:
            return self.bits > src.bits
        return False

    def is_narrowing_to(self, dst: "DType") -> bool:
        """True when casting ``self`` -> ``dst`` can round, saturate or truncate."""
        return dst != self and not dst.can_represent(self)

    def __repr__(self) -> str:
        return self.name

    __str__ = __repr__


bf16 = DType("bf16", 16, "float", "*bf16", "bf16", exp_bits=8, man_bits=7)
f16 = DType("f16", 16, "float", "*fp16", "fp16", exp_bits=5, man_bits=10)
f32 = DType("f32", 32, "float", "*fp32", "fp32", exp_bits=8, man_bits=23)
i32 = DType("i32", 32, "int", "*i32", "i32")
i64 = DType("i64", 64, "int", "*i64", "i64")
u8 = DType("u8", 8, "uint", "*u8", "u8", signed=False)
e4m3 = DType("e4m3", 8, "float", "*fp8e4nv", "fp8e4nv", exp_bits=4, man_bits=3, computable=False)
e5m2 = DType("e5m2", 8, "float", "*fp8e5", "fp8e5", exp_bits=5, man_bits=2, computable=False)
# 4-bit float, two elements packed per byte; Triton sees the storage bytes.
e2m1 = DType("e2m1", 4, "float", "*u8", None, exp_bits=2, man_bits=1, computable=False)
# 8-bit unsigned power-of-two scale (exponent only).
ue8m0 = DType("ue8m0", 8, "float", "*u8", None, exp_bits=8, man_bits=0, signed=False, computable=False)

ALL_DTYPES: Mapping[str, DType] = {
    d.name: d for d in (bf16, f16, f32, i32, i64, u8, e4m3, e5m2, e2m1, ue8m0)
}


@dataclass(frozen=True, repr=False)
class BlockScaled:
    """A block-scaled quantized format, usable as a tensor dtype.

    ``block`` consecutive elements along ``axis`` share one ``scale`` value;
    ``global_scale`` (optional) is the dtype of one extra per-tensor scale.
    The scale tensor has the shape of the data tensor with the blocked axis
    divided by ``block``.
    """

    elem: DType
    scale: DType
    block: int
    axis: int = -1
    global_scale: DType | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.elem, DType):
            raise TilefishDeclarationError(f"BlockScaled: elem must be a DType, got {self.elem!r}")
        if not isinstance(self.scale, DType):
            raise TilefishDeclarationError(f"BlockScaled: scale must be a DType, got {self.scale!r}")
        if self.global_scale is not None and not isinstance(self.global_scale, DType):
            raise TilefishDeclarationError(
                f"BlockScaled: global_scale must be a DType or None, got {self.global_scale!r}"
            )
        if isinstance(self.block, bool) or not isinstance(self.block, int) or self.block < 1:
            raise TilefishDeclarationError(f"BlockScaled: block must be a positive int, got {self.block!r}")
        if isinstance(self.axis, bool) or not isinstance(self.axis, int):
            raise TilefishDeclarationError(f"BlockScaled: axis must be an int, got {self.axis!r}")
        if self.block % self.elem.elems_per_byte:
            raise TilefishDeclarationError(
                f"BlockScaled: block {self.block} must be a multiple of "
                f"{self.elem.elems_per_byte} (elements packed per byte of {self.elem})"
            )

    def normalized(self, rank: int) -> "BlockScaled":
        """Return the same format with ``axis`` made non-negative for a tensor of ``rank``."""
        axis = self.normalized_axis(rank)
        if axis == self.axis:
            return self
        return BlockScaled(self.elem, self.scale, self.block, axis, self.global_scale)

    def normalized_axis(self, rank: int) -> int:
        if not -rank <= self.axis < rank:
            raise TilefishTypeError(
                f"{self!r}: axis {self.axis} is out of range for a rank-{rank} tensor"
            )
        return self.axis % rank

    def check_shape(self, shape: "tuple[DimExpr, ...]", known: "Mapping[DimExpr, int] | None" = None) -> None:
        """Raise unless the blocked axis of ``shape`` is provably divisible by ``block``."""
        axis = self.normalized_axis(len(shape))
        dim = shape[axis]
        if not dim.is_divisible_by(self.block, known):
            hint = (
                f"static size {dim.value} is not a multiple of {self.block}"
                if dim.is_static
                else f"declare it with Dim(..., divisible_by={self.block})"
            )
            raise TilefishTypeError(
                f"{self!r}: blocked axis {axis} has size {dim}, which is not divisible by "
                f"block={self.block} ({hint})"
            )

    def scale_shape(self, shape: "tuple[DimExpr, ...]", known: "Mapping[DimExpr, int] | None" = None) -> "tuple[DimExpr, ...]":
        """Shape of the scale tensor: the blocked axis divided by ``block``."""
        from .dims import cdiv

        self.check_shape(shape, known)
        axis = self.normalized_axis(len(shape))
        out = list(shape)
        out[axis] = cdiv(shape[axis], self.block)  # exact: divisibility was just checked
        return tuple(out)

    def __repr__(self) -> str:
        parts = f"elem={self.elem}, scale={self.scale}, block={self.block}, axis={self.axis}"
        if self.global_scale is not None:
            parts += f", global_scale={self.global_scale}"
        return f"BlockScaled({parts})"

    def __str__(self) -> str:
        """Compact form used when printing tensor types."""
        parts = f"{self.elem}, {self.scale}, {self.block}"
        if self.axis != -1:
            parts += f", axis={self.axis}"
        if self.global_scale is not None:
            parts += f", global_scale={self.global_scale}"
        return f"blockscaled({parts})"
