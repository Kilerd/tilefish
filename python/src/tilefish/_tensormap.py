"""TMA tensor descriptors: what a ``TensorDesc`` parameter is, how a host encodes it, and the rules it is checked by. Pure Python (no torch, no Triton).

A Hopper or Blackwell kernel written in Triton or Gluon reads and writes tiles through the Tensor Memory Accelerator when one of its parameters is a
*host tensor descriptor* (``triton.tools.tensor_descriptor.TensorDescriptor``, Gluon's ``nvidia.hopper.TensorDescriptor``): a ``CUtensorMap`` the
host encodes with ``cuTensorMapEncodeTiled`` and passes **by value** (``.param .align 64 .b8 [128]``, a ``__grid_constant__``). Triton 3.5.1
expands such a parameter, in the order of the kernel's parameters, into

1. the 128-byte ``CUtensorMap``;
2. the global shape, one ``i32`` per axis (outermost first);
3. the element strides, one ``i64`` per axis (outermost first);

(``backends/nvidia/driver.py``, ``make_tensordesc_arg``), and its compiled kernel records, per descriptor, what the map must say
(``metadata.tensordesc_meta``: the swizzle, the element type and size, the box). The box is the descriptor's *block* (the tile one ``load`` or
``store`` of the kernel moves, which is in its type), with the inner dimension cut to the swizzle span: a [128, 128] f16 block with a 128-byte
swizzle is copied as two [128, 64] boxes. tilefish declares all of it with the parameter (:class:`TensorMapSpec`), checks at the build that the
compiler expects exactly that, writes it into the manifest (``"kind": "tensordesc"``, manifest schema 3 section 5.5), and both hosts encode the map
from the tensor of each launch: its address, its shape and its strides, with the declared box, swizzle, L2 promotion and out-of-bounds fill,
interleave ``NONE`` and element strides 1 (Triton's ``fillTMADescriptor`` makes the same call). This module has the rules both hosts hold a launch to
before they encode (:func:`check_tensor_map`), and the arguments of the encode (:func:`encode_arguments`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import TilefishDeclarationError, TilefishPreconditionError

MAX_RANK = 5  # cuTensorMapEncodeTiled takes 1 to 5 dimensions
MAX_BOX = 256  # each box dimension is 1 to 256 elements
MAX_BLOCK = 1 << 16  # a block dimension (the inner one may be wider than the box: the kernel copies it as several boxes)
SWIZZLES = (0, 32, 64, 128)  # bytes of the swizzle span; 0: none
L2_PROMOTIONS = (0, 64, 128, 256)  # bytes; 0: none
OOB_FILLS = ("zero", "nan")
MAX_STRIDE_BYTES = 1 << 40  # a global stride is less than 2**40 bytes
STRIDE_ALIGNMENT = 16  # every global stride (but the innermost, which is the element) is a multiple of 16 bytes
DESCRIPTOR_BYTES = 128  # sizeof(CUtensorMap), passed by value, 64-byte aligned
INT32_MAX = (1 << 31) - 1  # a global dimension: the driver takes up to 2**32, the kernel gets it as an i32

# The CUtensorMapDataType of each tilefish dtype a descriptor may describe (the name of the CUDA enumerator), and Triton's device-side code
# of it (``tensordesc_meta[i]["elem_type"]``: the same numbers, but bf16 is 10 on the device and 9 on the host, Triton's TMA_DTYPE_DEVICE_TO_HOST).
DATA_TYPES = {
    "u8": ("CU_TENSOR_MAP_DATA_TYPE_UINT8", 0),
    "e4m3": ("CU_TENSOR_MAP_DATA_TYPE_UINT8", 0),
    "e5m2": ("CU_TENSOR_MAP_DATA_TYPE_UINT8", 0),
    "i32": ("CU_TENSOR_MAP_DATA_TYPE_INT32", 3),
    "i64": ("CU_TENSOR_MAP_DATA_TYPE_INT64", 5),
    "f16": ("CU_TENSOR_MAP_DATA_TYPE_FLOAT16", 6),
    "f32": ("CU_TENSOR_MAP_DATA_TYPE_FLOAT32", 7),
    "bf16": ("CU_TENSOR_MAP_DATA_TYPE_BFLOAT16", 10),
}
DATA_TYPE_VALUES = {  # the enumerators' values (cuda.h)
    "CU_TENSOR_MAP_DATA_TYPE_UINT8": 0,
    "CU_TENSOR_MAP_DATA_TYPE_INT32": 3,
    "CU_TENSOR_MAP_DATA_TYPE_INT64": 5,
    "CU_TENSOR_MAP_DATA_TYPE_FLOAT16": 6,
    "CU_TENSOR_MAP_DATA_TYPE_FLOAT32": 7,
    "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16": 9,
}
ITEMSIZE = {"u8": 1, "e4m3": 1, "e5m2": 1, "i32": 4, "i64": 8, "f16": 2, "f32": 4, "bf16": 2}
FLOATS = ("f16", "bf16", "f32")  # the dtypes whose out-of-bounds elements may be filled with NaN
SWIZZLE_NAMES = {0: "CU_TENSOR_MAP_SWIZZLE_NONE", 32: "CU_TENSOR_MAP_SWIZZLE_32B", 64: "CU_TENSOR_MAP_SWIZZLE_64B", 128: "CU_TENSOR_MAP_SWIZZLE_128B"}
SWIZZLE_CODES = {0: 0, 32: 1, 64: 2, 128: 3}  # CUtensorMapSwizzle, and Triton's ``tensordesc_meta`` "swizzle"
L2_NAMES = {
    0: "CU_TENSOR_MAP_L2_PROMOTION_NONE",
    64: "CU_TENSOR_MAP_L2_PROMOTION_L2_64B",
    128: "CU_TENSOR_MAP_L2_PROMOTION_L2_128B",
    256: "CU_TENSOR_MAP_L2_PROMOTION_L2_256B",
}
L2_CODES = {0: 0, 64: 1, 128: 2, 256: 3}
OOB_NAMES = {"zero": "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE", "nan": "CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA"}
OOB_CODES = {"zero": 0, "nan": 1}
INTERLEAVE_NONE = "CU_TENSOR_MAP_INTERLEAVE_NONE"
MAP_KEYS = ("data_type", "block", "box", "element_strides", "interleave", "swizzle", "l2_promotion", "oob_fill")


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def default_swizzle(block: "Sequence[int]", bits: int) -> int:
    """The swizzle a descriptor of this block gets when the declaration names none: the largest of 128, 64 and 32 bytes that divides the bytes of
    the block's inner dimension, and none for a block of rank 1 or of fewer than 8 rows in all (Gluon's ``NVMMASharedLayout.get_default_for``; the
    choice of Triton 3.5.1's compiler for every block the tests compile). The build checks it against the compiler's choice either way."""
    inner = block[-1] * bits // 8
    swizzle = next((s for s in (128, 64, 32) if inner % s == 0), 0)
    if len(block) < 2 or math.prod(block[:-1]) < 8:
        return 0
    return swizzle


def box_of(block: "Sequence[int]", swizzle: int, bits: int) -> "tuple[int, ...]":
    """The box of the ``CUtensorMap`` of a descriptor of ``block`` with ``swizzle``: the block, with its inner dimension cut to the swizzle span (a
    kernel copies a block whose rows are wider than the span as several boxes: Triton 3.5.1 records ``block_size`` [128, 64] for a [128, 128] f16
    block with a 128-byte swizzle)."""
    if not swizzle:
        return tuple(block)
    return tuple(block[:-1]) + (min(block[-1], swizzle * 8 // bits),)


@dataclass(frozen=True)
class TensorMapSpec:
    """What a ``CUtensorMap`` of a ``TensorDesc`` parameter says besides its tensor: the block of the descriptor (the tile a ``load`` or ``store`` of
    the kernel moves, outermost first, which is in its type), the box of the map (the block, or its rows cut to the swizzle span: :func:`box_of`), the
    swizzle (bytes: 0, 32, 64 or 128), the L2 promotion (bytes: 0, 64, 128 or 256) and the out-of-bounds fill (``"zero"``: an element outside the
    tensor reads as zero and a store there is dropped; ``"nan"``: a float reads as NaN). Interleave is ``NONE`` and the element strides are 1,
    always (what Triton's kernels are compiled for)."""

    block: "tuple[int, ...]"
    box: "tuple[int, ...]"
    swizzle: int
    l2_promotion: int = 128
    oob_fill: str = "zero"

    @property
    def rank(self) -> int:
        return len(self.block)

    def to_json(self, dtype: str) -> "dict[str, Any]":
        """The ``tensor_map`` object of the manifest (and of a compile spec): the block of the descriptor, and every argument of
        ``cuTensorMapEncodeTiled`` that is not the tensor's, by the names of the CUDA enumerators."""
        return {
            "data_type": DATA_TYPES[dtype][0],
            "block": list(self.block),
            "box": list(self.box),
            "element_strides": [1] * self.rank,
            "interleave": INTERLEAVE_NONE,
            "swizzle": SWIZZLE_NAMES[self.swizzle],
            "l2_promotion": L2_NAMES[self.l2_promotion],
            "oob_fill": OOB_NAMES[self.oob_fill],
        }

    def canonical(self) -> "dict[str, Any]":
        """The declaration digest's part of it: what the declaration says (the box follows from the block and the swizzle)."""
        return {"block": list(self.block), "swizzle": self.swizzle, "l2_promotion": self.l2_promotion, "oob_fill": self.oob_fill}

    def triton_meta(self, dtype: str) -> "dict[str, Any]":
        """What Triton records for the descriptor in ``metadata.tensordesc_meta`` when the compiled kernel expects this map."""
        return {
            "swizzle": SWIZZLE_CODES[self.swizzle],
            "elem_size": ITEMSIZE[dtype],
            "elem_type": DATA_TYPES[dtype][1],
            "block_size": list(self.box),
            "fp4_padded": False,
        }


def check_map(where: str, dtype: str, block: "Sequence[int]", box: "Sequence[int]", swizzle: int, l2: int, oob: str) -> None:
    """The rules of a map, whoever wrote it (a declaration, a reader of a manifest): what ``cuTensorMapEncodeTiled`` refuses (a rank of 1 to 5; a box
    of 1 to 256 elements per axis whose inner dimension is a multiple of 16 bytes and, swizzled, within the swizzle span; a NaN fill of a float
    only), what Triton refuses (a block of powers of two), and a box that tiles the block. A :class:`~tilefish.errors.TilefishDeclarationError`."""
    rank = len(block)
    if not 1 <= rank <= MAX_RANK or len(box) != rank:
        raise TilefishDeclarationError(f"{where}: a tensor descriptor describes a tensor of rank 1 to {MAX_RANK}, with a box of its rank (block {list(block)}, box {list(box)})")
    item = ITEMSIZE[dtype]
    for axis, b in enumerate(block):
        if not 1 <= b <= MAX_BLOCK or b & (b - 1):
            raise TilefishDeclarationError(f"{where}: block[{axis}] is {b}: a block dimension is a power of two (Triton's rule) from 1 to {MAX_BLOCK}")
    for axis, (n, b) in enumerate(zip(box, block)):
        if not 1 <= n <= MAX_BOX:
            raise TilefishDeclarationError(
                f"{where}: the box of the descriptor is {list(box)}, but a box dimension is 1 to {MAX_BOX} elements (cuTensorMapEncodeTiled refuses axis "
                f"{axis}): take a smaller block"
            )
        if b % n:
            raise TilefishDeclarationError(f"{where}: the box {list(box)} does not tile the block {list(block)} (axis {axis})")
    inner = box[-1] * item
    if inner % 16:
        raise TilefishDeclarationError(
            f"{where}: the inner dimension of the box is {box[-1]} elements of {item} bytes, {inner} bytes, which is not a multiple of 16 "
            "(cuTensorMapEncodeTiled refuses it)"
        )
    if swizzle not in SWIZZLES:
        raise TilefishDeclarationError(f"{where}: swizzle= is the swizzle span in bytes, one of {SWIZZLES} (0: none), got {swizzle!r}")
    if swizzle and inner > swizzle:
        raise TilefishDeclarationError(
            f"{where}: the inner dimension of the box is {inner} bytes, more than the {swizzle}-byte swizzle span (cuTensorMapEncodeTiled refuses it)"
        )
    if l2 not in L2_PROMOTIONS:
        raise TilefishDeclarationError(f"{where}: l2_promotion= is in bytes, one of {L2_PROMOTIONS} (0: none), got {l2!r}")
    if oob not in OOB_FILLS:
        raise TilefishDeclarationError(f"{where}: oob_fill= is 'zero' or 'nan', got {oob!r}")
    if oob == "nan" and dtype not in FLOATS:
        raise TilefishDeclarationError(f"{where}: oob_fill='nan' fills with a float NaN, and {dtype} is not a float type of a descriptor ({', '.join(FLOATS)})")


def make_spec(where: str, dtype: str, rank: int, block: Any, swizzle: Any, l2_promotion: Any, oob_fill: Any, bits: int) -> TensorMapSpec:
    """Check a declaration (``TensorDesc(..., block=, swizzle=, l2_promotion=, oob_fill=)``) and make its :class:`TensorMapSpec`: the swizzle by
    default is :func:`default_swizzle`'s, the box follows from the block and the swizzle (:func:`box_of`), and every rule of :func:`check_map` holds."""
    if dtype not in DATA_TYPES:
        raise TilefishDeclarationError(
            f"{where}: a tensor descriptor of {dtype} is not supported (the dtypes a TMA descriptor describes here are {', '.join(sorted(DATA_TYPES))})"
        )
    if not 1 <= rank <= MAX_RANK:
        raise TilefishDeclarationError(f"{where}: a tensor descriptor describes a tensor of rank 1 to {MAX_RANK}, this one has rank {rank}")
    items = None
    if isinstance(block, (list, tuple)):
        items = [getattr(b, "value", b) if getattr(b, "is_static", None) is True else b for b in block]  # a static Dim is its size
    if items is None or len(items) != rank or not all(_is_int(b) for b in items):
        raise TilefishDeclarationError(
            f"{where}: block= is the tile of the descriptor, one int per axis of the tensor ({rank}), outermost first, got {block!r}"
        )
    block_t = tuple(items)
    for axis, b in enumerate(block_t):
        if not 1 <= b <= MAX_BLOCK or b & (b - 1):
            raise TilefishDeclarationError(f"{where}: block[{axis}] is {b}: a block dimension is a power of two (Triton's rule) from 1 to {MAX_BLOCK}")
    if swizzle is None:
        swizzle = default_swizzle(block_t, bits)
    if not _is_int(swizzle) or swizzle not in SWIZZLES:
        raise TilefishDeclarationError(f"{where}: swizzle= is the swizzle span in bytes, one of {SWIZZLES} (0: none), got {swizzle!r}")
    if not _is_int(l2_promotion) or l2_promotion not in L2_PROMOTIONS:
        raise TilefishDeclarationError(f"{where}: l2_promotion= is in bytes, one of {L2_PROMOTIONS} (0: none), got {l2_promotion!r}")
    box = box_of(block_t, swizzle, bits)
    check_map(where, dtype, block_t, box, swizzle, l2_promotion, oob_fill)
    return TensorMapSpec(block_t, box, swizzle, l2_promotion, oob_fill)


def signature_type(triton_scalar: str, spec: TensorMapSpec, bits: int, language: str) -> str:
    """The Triton signature entry of the parameter: ``tensordesc<fp16[128, 64]>`` (the block); for a Gluon kernel the shared layout of the tile follows
    (``,NVMMASharedLayout(...)``, as Gluon's ``TensorDescriptor`` spells it), since Gluon's frontend takes the layout from the type."""
    inner = f"{triton_scalar}{list(spec.block)}"
    if language != "gluon":
        return f"tensordesc<{inner}>"
    rank = spec.rank
    layout = (
        f"NVMMASharedLayout(swizzle_byte_width={spec.swizzle}, element_bitwidth={bits}, rank={rank}, transposed=False, fp4_padded=False, "
        f"ctas_per_cga={[1] * rank}, cta_split_num={[1] * rank}, cta_order={list(reversed(range(rank)))})"
    )
    return f"tensordesc<{inner},{layout}>"


def read_json(m: Any, dtype: Any, rank: int) -> "str | None":
    """What is wrong with the ``tensor_map`` object of a parameter of ``dtype`` and ``rank`` (``None``: nothing). Exactly the keys of
    :data:`MAP_KEYS`; the data type of the dtype; a block and a box of ``rank`` ints by the rules of :func:`check_map`; element strides of 1;
    interleave ``NONE``; a swizzle, an L2 promotion and an out-of-bounds fill among the enumerators. (The box is not held to :func:`box_of`: that is
    how a declaration makes it, and the build checks it against the compiled kernel; a reader holds a map to what the driver and Triton require.)"""
    if not isinstance(m, dict):
        return f"tensor_map must be an object, got {m!r}"
    keys = set(m)
    if keys != set(MAP_KEYS):
        return f"tensor_map must have exactly the fields {list(MAP_KEYS)}, has {sorted(str(k) for k in keys)}"
    if dtype not in DATA_TYPES:
        return f"a tensor descriptor of {dtype!r} is not supported"
    if m["data_type"] != DATA_TYPES[dtype][0]:
        return f"tensor_map.data_type of a {dtype} tensor is {DATA_TYPES[dtype][0]}, got {m['data_type']!r}"
    for key in ("block", "box"):
        v = m[key]
        if not isinstance(v, list) or len(v) != rank or not all(_is_int(b) for b in v):
            return f"tensor_map.{key} must be a list of {rank} ints (one per axis of the shape), got {v!r}"
    es = m["element_strides"]
    if not isinstance(es, list) or es != [1] * rank or not all(_is_int(e) for e in es):
        return f"tensor_map.element_strides must be {[1] * rank} (Triton's kernels take element strides of 1), got {es!r}"
    if m["interleave"] != INTERLEAVE_NONE:
        return f"tensor_map.interleave must be {INTERLEAVE_NONE}, got {m['interleave']!r}"
    swizzle = {v: k for k, v in SWIZZLE_NAMES.items()}.get(m["swizzle"]) if isinstance(m["swizzle"], str) else None
    l2 = {v: k for k, v in L2_NAMES.items()}.get(m["l2_promotion"]) if isinstance(m["l2_promotion"], str) else None
    oob = {v: k for k, v in OOB_NAMES.items()}.get(m["oob_fill"]) if isinstance(m["oob_fill"], str) else None
    if swizzle is None:
        return f"tensor_map.swizzle must be one of {list(SWIZZLE_NAMES.values())}, got {m['swizzle']!r}"
    if l2 is None:
        return f"tensor_map.l2_promotion must be one of {list(L2_NAMES.values())}, got {m['l2_promotion']!r}"
    if oob is None:
        return f"tensor_map.oob_fill must be one of {list(OOB_NAMES.values())}, got {m['oob_fill']!r}"
    try:
        check_map("tensor_map", dtype, m["block"], m["box"], swizzle, l2, oob)
    except TilefishDeclarationError as e:
        return str(e).removeprefix("tensor_map: ")
    return None


def spec_of_json(m: "Mapping[str, Any]") -> TensorMapSpec:
    """The :class:`TensorMapSpec` of a ``tensor_map`` object that :func:`read_json` accepted."""
    return TensorMapSpec(
        tuple(m["block"]),
        tuple(m["box"]),
        {v: k for k, v in SWIZZLE_NAMES.items()}[m["swizzle"]],
        {v: k for k, v in L2_NAMES.items()}[m["l2_promotion"]],
        {v: k for k, v in OOB_NAMES.items()}[m["oob_fill"]],
    )


def check_tensor_map(stem: str, name: str, dtype: str, shape: "Sequence[int]", strides: "Sequence[int]") -> None:
    """The rules ``cuTensorMapEncodeTiled`` holds a tensor to, checked before it is called so that a launch is refused with a kind, not with the
    driver's ``CUDA_ERROR_INVALID_VALUE`` (the address is the ``pointer_alignment`` precondition of 16 bytes, checked before): ``strides`` are the
    canonical element strides of the tensor (manifest schema 3, section 5.1), outermost first. In this order:

    * every dimension is at most ``2**31 - 1`` (the driver takes ``2**32``; the shape is passed to the kernel as ``i32``);
    * the innermost stride is 1 (a dense tensor's always is; a view's is the fixed unit stride of its declaration);
    * every other stride is a multiple of 16 bytes and less than ``2**40`` bytes (axis by axis, outermost first).

    Each is a refusal of kind ``tensor_map``."""
    item = ITEMSIZE[dtype]
    rank = len(shape)
    for axis, n in enumerate(shape):
        if n > INT32_MAX:
            raise TilefishPreconditionError(
                f"'{stem}': argument '{name}' has {n} elements along axis {axis}: the dimensions of a tensor descriptor are passed to the kernel as "
                f"i32, so at most {INT32_MAX}",
                kind="tensor_map",
                arg=name,
            )
    if strides[rank - 1] != 1:
        raise TilefishPreconditionError(
            f"'{stem}': argument '{name}' has stride {strides[rank - 1]} along its last axis: a tensor descriptor reads the last axis contiguously "
            "(stride 1)",
            kind="tensor_map",
            arg=name,
        )
    for axis in range(rank - 1):
        nbytes = strides[axis] * item
        if nbytes % STRIDE_ALIGNMENT or nbytes >= MAX_STRIDE_BYTES:
            raise TilefishPreconditionError(
                f"'{stem}': argument '{name}' (shape {list(shape)}, strides {list(strides)}) has a stride of {nbytes} bytes along axis {axis}: the "
                f"strides of a tensor descriptor are multiples of {STRIDE_ALIGNMENT} bytes and less than 2**40 bytes (pad the rows of the tensor to a "
                "multiple of 16 bytes)",
                kind="tensor_map",
                arg=name,
            )


@dataclass(frozen=True)
class EncodeArguments:
    """The arguments of one ``cuTensorMapEncodeTiled`` call, in the driver's order (innermost dimension first), as numbers: what both hosts pass.
    ``global_strides`` has ``rank`` entries, as Triton's ``fillTMADescriptor`` passes it (the driver reads the first ``rank - 1``; the last is the
    bytes of the whole tensor, which Triton writes there, and an array of ``rank - 1`` entries is an empty one for a tensor of rank 1)."""

    data_type: int
    rank: int
    address: int
    global_dim: "tuple[int, ...]"
    global_strides: "tuple[int, ...]"
    box_dim: "tuple[int, ...]"
    element_strides: "tuple[int, ...]"
    interleave: int
    swizzle: int
    l2_promotion: int
    oob_fill: int


def encode_arguments(dtype: str, spec: TensorMapSpec, address: int, shape: "Sequence[int]", strides: "Sequence[int]") -> EncodeArguments:
    """The arguments of ``cuTensorMapEncodeTiled`` for the tensor at ``address`` with ``shape`` and canonical element ``strides`` (outermost first):
    the dimensions and the box reversed (innermost first), the strides of the outer axes in bytes, innermost first, then the bytes of the tensor."""
    item = ITEMSIZE[dtype]
    rank = len(shape)
    dims = tuple(int(n) for n in reversed(shape))
    outer = [int(strides[axis]) * item for axis in range(rank - 2, -1, -1)]  # the stride of axis rank-2 first
    last = dims[rank - 1] * (outer[-1] if outer else item)
    return EncodeArguments(
        data_type=DATA_TYPE_VALUES[DATA_TYPES[dtype][0]],
        rank=rank,
        address=int(address),
        global_dim=dims,
        global_strides=tuple(outer) + (last,),
        box_dim=tuple(reversed(spec.box)),
        element_strides=(1,) * rank,
        interleave=0,
        swizzle=SWIZZLE_CODES[spec.swizzle],
        l2_promotion=L2_CODES[spec.l2_promotion],
        oob_fill=OOB_CODES[spec.oob_fill],
    )


@dataclass(frozen=True)
class TensorMapArg:
    """One ``TensorDesc`` argument of a launch: the 128 bytes of its encoded ``CUtensorMap``, and the shape (``i32`` each) and the element strides
    (``i64`` each) Triton passes after it, outermost first."""

    descriptor: bytes
    shape: "tuple[int, ...]"
    strides: "tuple[int, ...]"
