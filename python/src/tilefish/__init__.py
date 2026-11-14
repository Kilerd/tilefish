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
axes; a graph materialises a view (:func:`contiguous`), the patches of a 2-D or 1-D convolution (:func:`im2col`, :func:`im2col1d`) and the
overlap-add of a 1-D transposed convolution (:func:`col2im1d`, in ATen's CUDA order of adds). The numerics contract of each is in
:mod:`tilefish.lower` (notes 7 to 11) and :mod:`tilefish._copyops`. An ``extern`` takes a Gluon kernel
(``extern(gluon=...)``: Triton's ``triton.experimental.gluon``, layouts written out), identified by the TTGIR Gluon's frontend builds, and a
``StrideOf`` says the type of its scalar (``dtype=i32`` or ``i64``) when it is not the width of the kernel's offsets.

M3.4: an extern of a Triton or Gluon kernel takes TMA tensor descriptors (:func:`TensorDesc`): a call passes the tensor, every launch encodes its
``CUtensorMap`` (``cuTensorMapEncodeTiled``) and passes it by value as Triton launches a host ``TensorDescriptor``; the manifest records the map
(manifest schema 3, section 5.5) and both hosts check and encode it, in single launches, in plans and in CUDA graphs. A stride or an offset may be
a product of symbolic sizes (``L*L`` for an ``[H, L, L]`` tensor: :func:`tilefish.dims.mul`), which views, ``Strided`` arguments and plans take;
the elementwise template broadcasts along any axis (note 12); a kernel or an op says its f32 math, ``math="ieee"`` for CUDA's without fast math
or flush to zero (:func:`div`, note 13); and :func:`tanh` is libdevice's ``tanhf`` (note 14).
"""

from . import aot
from . import ops as _ops
from ._library import AlgoTable
from .decorators import ExternOp, GraphOp, Implementation, KernelOp, LibraryOp, Op, extern, graph, impl, kernel, library
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
from .ops import copy as copy  # tilefish.copy (a copy step of a graph): not in __all__, `from tilefish import *` must not shadow the module copy
from .ops import argmax, argmin, cos, div, exp, gather_rows, gelu, iota, load, rsqrt, scatter_rows, silu, sin, sqrt, tanh, where
from ._copyops import col2im1d, contiguous, im2col, im2col1d
from .ops import reduce_mean as mean
from .runtime import capture, run
from ._planrun import CapturedPlan
from .trace import Const, Value
from .types import DevScalar, InOut, MayAlias, Out, StrideOf, Strided, Tensor, TensorDesc, TensorType, const

__version__ = "0.0.5"

# Reductions with numpy semantics. They share a name with a builtin, so they are not in
# __all__ (`from tilefish import *` must not shadow the builtins); use tilefish.sum etc.
sum = _ops.reduce_sum  # noqa: A001
max = _ops.reduce_max  # noqa: A001

# `sum` and `max` are deliberately not in __all__: `from tilefish import *` must not shadow
# the builtins. They stay reachable as tilefish.sum and tilefish.max.
__all__ = [
    "AlgoTable",
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
    "Implementation",
    "GraphOp",
    "InOut",
    "KernelOp",
    "LibraryOp",
    "LoweredKernel",
    "Manifest",
    "MayAlias",
    "Node",
    "Op",
    "Out",
    "Plan",
    "Requirement",
    "Scalar",
    "StrideOf",
    "Strided",
    "Tensor",
    "TensorDesc",
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
    "col2im1d",
    "const",
    "contiguous",
    "cos",
    "div",
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
    "im2col1d",
    "impl",
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
    "tanh",
    "u8",
    "ue8m0",
    "where",
]
