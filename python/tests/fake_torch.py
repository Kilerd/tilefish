"""Tensors and a ``torch`` module with only what ``tilefish.runtime._prepare`` reads (not collected: the name has no ``test_`` prefix).

``_prepare`` is the part of ``replay`` that checks a call and decides the binary, the scalars and the grid; it needs facts about
tensors (shape, dtype, address, contiguity, device) and, for an ``Out`` tensor, an allocator. No GPU is needed for that: a
``FakeTensor`` has the facts, and ``FakeTorch.empty`` hands out addresses. ``prepare(op, args)`` runs the real ``_prepare``
with them. It is what the cross-language vectors of the launch checks are made with (``export_rust_vectors.py``): Python's
verdict on a call is the oracle for what the Rust host decides.
"""

import math
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


class FakeTensor:
    """What the checks read off a torch tensor: a contiguous tensor on ``cuda:0`` at ``ptr``."""

    def __init__(self, shape, dtype, ptr):
        self.shape = tuple(shape)
        self.dtype = dtype if isinstance(dtype, FakeDtype) else FakeDtype(dtype)
        self._ptr = ptr
        self.device = FakeDevice()

    def data_ptr(self):
        return self._ptr

    def is_contiguous(self):
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


def verdict(prepared, error):
    """What Python decided, as plain data: the binary, the grid and the ``kernelParams`` values, or the kind of the refusal."""
    if error is not None:
        return {"ok": False, "kind": error.kind if isinstance(error, TilefishPreconditionError) else "other"}
    spec = prepared.spec
    params = []
    for a in spec.args:
        if a.kind == "pointer":
            params.append(prepared.tensors[a.name].data_ptr())
        elif a.kind == "scalar":
            params.append(prepared.scalars[a.name])
    return {"ok": True, "stem": spec.stem, "grid": list(prepared.grid) + [1] * (3 - len(prepared.grid)), "params": params + [0, 0]}
