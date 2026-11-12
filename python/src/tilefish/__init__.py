"""tilefish: typed tensor language for GPU inference.

M1: the type system (dtypes, dims, tensor types, block-scaled formats), the
operator declarations (``extern``, ``library``, ``kernel``, ``graph``),
tracing into a typed IR with type checking, and Triton ahead-of-time compile
specs for ``extern`` kernels (:mod:`tilefish.aot`). Kernel bodies are checked
but not lowered yet.
"""

from . import aot
from . import ops as _ops
from .decorators import ExternOp, GraphOp, KernelOp, LibraryOp, Op, extern, graph, kernel, library
from .dims import Dim, DimExpr, cdiv
from .dtypes import (
    BlockScaled,
    DType,
    bf16,
    e2m1,
    e4m3,
    e5m2,
    f16,
    f32,
    i32,
    i64,
    u8,
    ue8m0,
)
from .errors import (
    TilefishAttributeError,
    TilefishDeclarationError,
    TilefishError,
    TilefishPicklingError,
    TilefishSpecError,
    TilefishTypeError,
)
from .ir import BucketRequirement, Graph, Node, Requirement, Scalar
from .ops import exp, gelu, rsqrt, silu, sqrt
from .ops import reduce_mean as mean
from .trace import Const, Value
from .types import InOut, Out, Tensor, TensorType, const

__version__ = "0.0.1"

# Reductions with numpy semantics. They share a name with a builtin, so they are not in
# __all__ (`from tilefish import *` must not shadow the builtins); use tilefish.sum etc.
sum = _ops.reduce_sum  # noqa: A001
max = _ops.reduce_max  # noqa: A001

# `sum` and `max` are deliberately not in __all__: `from tilefish import *` must not shadow
# the builtins. They stay reachable as tilefish.sum and tilefish.max.
__all__ = [
    "BlockScaled",
    "BucketRequirement",
    "Const",
    "DType",
    "Dim",
    "DimExpr",
    "ExternOp",
    "Graph",
    "GraphOp",
    "InOut",
    "KernelOp",
    "LibraryOp",
    "Node",
    "Op",
    "Out",
    "Requirement",
    "Scalar",
    "Tensor",
    "TensorType",
    "TilefishAttributeError",
    "TilefishDeclarationError",
    "TilefishError",
    "TilefishPicklingError",
    "TilefishSpecError",
    "TilefishTypeError",
    "Value",
    "aot",
    "bf16",
    "cdiv",
    "const",
    "e2m1",
    "e4m3",
    "e5m2",
    "exp",
    "extern",
    "f16",
    "f32",
    "gelu",
    "graph",
    "i32",
    "i64",
    "kernel",
    "library",
    "mean",
    "rsqrt",
    "silu",
    "sqrt",
    "u8",
    "ue8m0",
]
