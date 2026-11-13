"""tilefish: typed tensor language for GPU inference.

M1: the type system (dtypes, dims, tensor types, block-scaled formats), the
operator declarations (``extern``, ``library``, ``kernel``, ``graph``),
tracing into a typed IR with type checking, and Triton ahead-of-time compile
specs for ``extern`` kernels (:mod:`tilefish.aot`).

M2: a ``kernel`` lowers to Triton source (:func:`lower`); a compile spec builds into a
cubin with a manifest (:func:`tilefish.aot.build`, :func:`tilefish.aot.write_manifest`); and
:func:`run` runs a kernel or an extern on torch tensors in the modes ``jit``, ``replay`` (the built
cubin through the CUDA driver, what a C++ or Rust host runs) and ``reference`` (the IR interpreted
with torch). Importing tilefish imports neither torch nor Triton; the ``gpu`` extra installs them. It does install an import
watch (:mod:`tilefish._source`) that records the files of the modules imported from then on, so that a build can refuse a module
that was edited after the process imported it.
"""

from . import _source

_source.install_import_watch()  # records the modules imported from here on, so a build can tell a file that was edited after its import

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
    TilefishBuildError,
    TilefishCacheWarning,
    TilefishDeclarationError,
    TilefishError,
    TilefishLoweringError,
    TilefishManifestError,
    TilefishPicklingError,
    TilefishPreconditionError,
    TilefishRuntimeError,
    TilefishSpecError,
    TilefishTypeError,
)
from ._manifest import Manifest
from .ir import BucketRequirement, Graph, Node, Requirement, Scalar
from .lower import LoweredKernel, lower
from .ops import exp, gelu, rsqrt, silu, sqrt
from .ops import reduce_mean as mean
from .runtime import run
from .trace import Const, Value
from .types import InOut, Out, Tensor, TensorType, const

__version__ = "0.0.2"

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
    "LoweredKernel",
    "Manifest",
    "Node",
    "Op",
    "Out",
    "Requirement",
    "Scalar",
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
    "TilefishPreconditionError",
    "TilefishRuntimeError",
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
    "lower",
    "mean",
    "rsqrt",
    "run",
    "silu",
    "sqrt",
    "u8",
    "ue8m0",
]
