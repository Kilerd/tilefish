"""Tensors and a ``torch`` module with only what ``tilefish.runtime._prepare`` reads (not collected: the name has no ``test_`` prefix).

``_prepare`` is the part of ``replay`` that checks a call and decides the binary, the scalars and the grid; it needs facts about
tensors (shape, dtype, address, strides, contiguity, device) and, for an ``Out`` tensor, an allocator. No GPU is needed for that: a
``FakeTensor`` has the facts, and ``FakeTorch.empty`` hands out addresses. ``prepare(op, args)`` runs the real ``_prepare``
with them. It is what the cross-language vectors of the launch checks are made with (``export_rust_vectors.py``): Python's
verdict on a call is the oracle for what the Rust host decides.

A ``FakeTensor`` is dense unless it is given ``strides`` (element strides, as ``t.stride()`` reports them): then it is a view whose ``ptr`` is the address of its first
element (``data_ptr()``), like a torch view with a storage offset.
"""

import math
import struct
from dataclasses import dataclass

from tilefish import runtime
from tilefish.errors import TilefishError, TilefishPreconditionError

ITEMSIZE = {"bf16": 2, "f16": 2, "f32": 4, "i32": 4, "i64": 8, "u8": 1}
TORCH_NAME = {"bf16": "bfloat16", "f16": "float16", "f32": "float32", "i32": "int32", "i64": "int64", "u8": "uint8"}
SHORT = {v: k for k, v in TORCH_NAME.items()}


@dataclass(frozen=True)
class FakeDtype:
    short: str

    def __str__(self):
        return f"torch.{TORCH_NAME[self.short]}"


class FakeDevice:
    type = "cuda"

    def __str__(self):
        return "cuda:0"


def dense_strides(shape):
    """The row-major element strides of a tensor of this shape."""
    out, acc = [0] * len(shape), 1
    for i in range(len(shape) - 1, -1, -1):
        out[i] = acc
        acc *= shape[i]
    return tuple(out)


class FakeTensor:
    """What the checks read off a torch tensor: a tensor on ``cuda:0`` whose first element is at ``ptr`` (dense, or with ``strides``)."""

    def __init__(self, shape, dtype, ptr, strides=None):
        self.shape = tuple(shape)
        self.dtype = dtype if isinstance(dtype, FakeDtype) else FakeDtype(dtype)
        self._ptr = ptr
        self.strides = None if strides is None else tuple(strides)
        assert self.strides is None or len(self.strides) == len(self.shape), (self.shape, self.strides)
        self.device = FakeDevice()

    def data_ptr(self):
        return self._ptr

    def stride(self):
        return dense_strides(self.shape) if self.strides is None else self.strides

    def is_contiguous(self):
        """torch's rule: the strides of the axes of size 1 do not matter."""
        if math.prod(self.shape) == 0:
            return True
        expected = 1
        for size, stride in zip(reversed(self.shape), reversed(self.stride())):
            if size != 1:
                if stride != expected:
                    return False
                expected *= size
        return True

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return ITEMSIZE[self.dtype.short]


class FakeTorch:
    """``torch.empty`` and the dtypes: the allocator gives each tensor its own slab of address space, 64 GiB apart."""

    SLAB = 1 << 36

    def __init__(self, next_slab):
        self.next_slab = next_slab
        for short, name in TORCH_NAME.items():
            setattr(self, name, FakeDtype(short))

    def empty(self, shape, dtype, device):
        ptr = self.next_slab
        self.next_slab += self.SLAB
        return FakeTensor(shape, dtype, ptr)


def prepare(op, args, *, next_slab):
    """``runtime._prepare`` of ``op`` on ``args`` (fake tensors): ``(prepared, None)``, or ``(None, error)``."""
    saved = runtime._torch
    runtime._torch = lambda: FakeTorch(next_slab)
    try:
        return runtime._prepare(runtime._target(op), tuple(args), {}), None
    except TilefishError as e:
        return None, e
    finally:
        runtime._torch = saved


def f32_bits(value):
    """The IEEE 754 binary32 bits of a number, as an unsigned integer: what a ``float`` kernel parameter holds."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def verdict(prepared, error):
    """What Python decided, as plain data: the binary, the grid and the ``kernelParams`` values, or the kind of the refusal.

    ``params`` are the values of the ``kernelParams`` slots: a pointer is its address, an ``i32`` or ``i64`` scalar its integer, a fixed argument
    of a prebuilt binary its value (an integer, or the bits of an ``f32``) and, for a Triton kernel, the two null scratch pointers at the end (a
    prebuilt binary has none).
    """
    if error is not None:
        return {"ok": False, "kind": error.kind if isinstance(error, TilefishPreconditionError) else "other"}
    spec = prepared.spec
    params = []
    for a in spec.args:
        if a.kind == "pointer":
            params.append(prepared.tensors[a.name].data_ptr())
        elif a.kind == "scalar":
            params.append(prepared.scalars[a.name])
        elif a.kind == "fixed":
            params.append(f32_bits(a.value) if a.type == "f32" else int(a.value))
    scratch = [] if spec.binary is not None else [0, 0]
    return {"ok": True, "stem": spec.stem, "grid": list(prepared.grid) + [1] * (3 - len(prepared.grid)), "params": params + scratch}
