"""tilefish: typed tensor language for GPU inference.

M1: the type system (dtypes, dims, tensor types, block-scaled formats), the
operator declarations (``extern``, ``library``, ``kernel``, ``graph``),
tracing into a typed IR with type checking, and Triton ahead-of-time compile
specs for ``extern`` kernels (:mod:`tilefish.aot`).

M2: a ``kernel`` lowers to Triton source (:func:`lower`); a compile spec builds into a
cubin with a manifest (:func:`tilefish.aot.build`, :func:`tilefish.aot.write_manifest`); and
:func:`run` runs a kernel or an extern on torch tensors in the modes ``jit``, ``replay`` (the built
cubin through the CUDA driver, what a C++ or Rust host runs) and ``reference`` (the IR interpreted
with torch). Importing tilefish imports neither torch nor Triton; the ``gpu`` extra installs them, and it changes nothing about
how Python imports modules. The identity of a binary is the Triton IR its kernel compiles to (:mod:`tilefish._ir`).

M3.2 (P1): strided views (``reshape``, ``permute``, ``select``, ``split`` give views; an ``extern`` parameter ``Strided[...]`` takes one, with its
strides as ``StrideOf`` scalars), device scalars (``DevScalar[dtype]``, read with :func:`load`), prebuilt PTX and cubin externs
(``extern(ptx=...)``, ``extern(cubin=...)``), a cuBLASLt GEMM as a ``library`` op with a pinned algorithm, and manifest schema 3
(``docs/manifest-schema-3.md``).

M3.2 (P2): :func:`plan` compiles a graph into a static launch plan with an arena memory plan per shape bucket (:mod:`tilefish._plan`), :func:`tilefish.runtime.export` writes it as
``plan.json`` next to the manifest (``docs/plan-schema-1.md``), and :func:`run` of a graph executes it in ``jit``, ``replay`` and ``reference`` modes, capturable in a CUDA graph.

M3.3: a lowered kernel compares (a mask, :data:`bool_`), selects (:func:`where`), counts positions (:func:`iota`, integer ``//`` and ``%``),
reduces to an index (:func:`argmax`, :func:`argmin`), gathers and scatters rows (:func:`gather_rows`, :func:`scatter_rows`), takes
:func:`sin` and :func:`cos`, keeps a NaN's payload in a cast (``x.to(bf16, nan="preserve")``) and broadcasts an operand over the leading
axes; a graph materialises a view (:func:`contiguous`) and the patches of a convolution (:func:`im2col`). The numerics contract of each is in
:mod:`tilefish.lower` (notes 7 to 11) and :mod:`tilefish._copyops`.
"""

from . import aot
from . import ops as _ops
from .decorators import ExternOp, GraphOp, KernelOp, LibraryOp, Op, extern, graph, kernel, library
from .dims import Dim, DimExpr, cdiv
from .dtypes import (
    BlockScaled,
    DType,
    bf16,
    bool_,
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
    TilefishBuildError,
    TilefishCacheWarning,
    TilefishDeclarationError,
    TilefishError,
    TilefishLoweringError,
    TilefishManifestError,
    TilefishPicklingError,
    TilefishPlanError,
    TilefishPreconditionError,
    TilefishRuntimeError,
    TilefishSpecError,
    TilefishTypeError,
)
from ._manifest import Manifest
from ._plan import Plan, plan
from .ir import BucketRequirement, Graph, Node, Requirement, Scalar
from .lower import LoweredKernel, lower
from .ops import argmax, argmin, cos, exp, gather_rows, gelu, iota, load, rsqrt, scatter_rows, silu, sin, sqrt, where
from ._copyops import contiguous, im2col
from .ops import reduce_mean as mean
from .runtime import capture, run
from ._planrun import CapturedPlan
from .trace import Const, Value
from .types import DevScalar, InOut, Out, StrideOf, Strided, Tensor, TensorType, const

__version__ = "0.0.4"

# Reductions with numpy semantics. They share a name with a builtin, so they are not in
# __all__ (`from tilefish import *` must not shadow the builtins); use tilefish.sum etc.
sum = _ops.reduce_sum  # noqa: A001
max = _ops.reduce_max  # noqa: A001

# `sum` and `max` are deliberately not in __all__: `from tilefish import *` must not shadow
# the builtins. They stay reachable as tilefish.sum and tilefish.max.
__all__ = [
    "BlockScaled",
    "BucketRequirement",
    "CapturedPlan",
    "Const",
    "DType",
    "DevScalar",
    "Dim",
    "DimExpr",
    "ExternOp",
    "Graph",
    "GraphOp",
    "InOut",
    "KernelOp",
    "LibraryOp",
    "LoweredKernel",
    "Manifest",
    "Node",
    "Op",
    "Out",
    "Plan",
    "Requirement",
    "Scalar",
    "StrideOf",
    "Strided",
    "Tensor",
    "TensorType",
    "TilefishAttributeError",
    "TilefishBuildError",
    "TilefishCacheWarning",
    "TilefishDeclarationError",
    "TilefishError",
    "TilefishLoweringError",
    "TilefishManifestError",
    "TilefishPicklingError",
    "TilefishPlanError",
    "TilefishPreconditionError",
    "TilefishRuntimeError",
    "TilefishSpecError",
    "TilefishTypeError",
    "Value",
    "aot",
    "argmax",
    "argmin",
    "bf16",
    "bool_",
    "capture",
    "cdiv",
    "const",
    "contiguous",
    "cos",
    "e2m1",
    "e4m3",
    "e5m2",
    "exp",
    "extern",
    "f16",
    "f32",
    "gather_rows",
    "gelu",
    "graph",
    "i32",
    "i64",
    "im2col",
    "iota",
    "kernel",
    "library",
    "load",
    "lower",
    "mean",
    "plan",
    "rsqrt",
    "run",
    "scatter_rows",
    "silu",
    "sin",
    "sqrt",
    "u8",
    "ue8m0",
    "where",
]
