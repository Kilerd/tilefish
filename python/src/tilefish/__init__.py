"""tilefish: typed tensor language for GPU inference.

M1: the type system (dtypes, dims, tensor types, block-scaled formats), the
operator declarations (``extern``, ``library``, ``kernel``, ``graph``),
tracing into a typed IR with type checking, and Triton ahead-of-time compile
specs for ``extern`` kernels (:mod:`tilefish.aot`). Kernel bodies are checked
but not lowered yet.
"""

from . import aot
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
    TilefishDeclarationError,
    TilefishError,
    TilefishSpecError,
    TilefishTypeError,
)
from .ir import Graph, Node, Requirement, Scalar
from .ops import exp, gelu, rsqrt, silu, sqrt
from .ops import reduce_max as max  # noqa: A001 - tilefish.max(x, axis, keepdims) reduces
from .ops import reduce_mean as mean
from .ops import reduce_sum as sum  # noqa: A001 - tilefish.sum(x, axis, keepdims) reduces
from .trace import Const, Value
from .types import InOut, Out, Tensor, TensorType, const

__version__ = "0.0.0"

__all__ = [
    "BlockScaled",
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
    "TilefishDeclarationError",
    "TilefishError",
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
    "max",
    "mean",
    "rsqrt",
    "silu",
    "sqrt",
    "sum",
    "u8",
    "ue8m0",
]
