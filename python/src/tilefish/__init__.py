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

M3.5: the primitives a CUDA kernel is reproduced with: :func:`erf` (CUDA's ``erff`` in IEEE math), :func:`sin_approx`, :func:`rcp`, :func:`fma`
and :func:`neg` (``-x``), and sums in a fixed order (``tilefish.sum(x, -1, order="sequential" | tree(...), fma=...)``, :func:`tree`); notes 15 to 17.
"""

from . import aot
from . import ops as _ops
from ._conv import CudnnTable
from ._library import AlgoTable
from .decorators import Config, ExternOp, GraphOp, Implementation, KernelOp, LibraryOp, Op, Persistent, config, extern, graph, impl, kernel, library, persistent
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
    TilefishCudaMathWarning,
    TilefishDeclarationError,
    TilefishError,
    TilefishLoweringError,
    TilefishManifestError,
    TilefishOrderWarning,
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
from .ops import argmax, argmin, cos, div, erf, exp, fma, gather_rows, gelu, iota, load, neg, rcp, rsqrt, scatter_rows, silu, sin, sin_approx
from .ops import barrier, sqrt, tanh, tree, where, zeros_like
from ._fusion import Fusion, TilefishFusionError, TilefishFusionWarning, fusion
from ._sass import SassChecks, sass
from ._copyops import col2im1d, contiguous, im2col, im2col1d
from .ops import reduce_mean as mean
from .runtime import capture, run
from ._planrun import CapturedPlan
from .trace import Const, Value
from .types import DevScalar, InOut, MayAlias, Null, Out, StrideOf, Strided, Tensor, TensorDesc, TensorType, const

__version__ = "0.0.10"


def fusion_report(graph: GraphOp) -> dict:
    """The fusion report of a ``@tilefish.graph`` (what ``export`` writes as ``<graph>.fusion.json``): every fused region, the ops of the body it
    took, what it reads and writes, every rounding it keeps and the rewrites of the graph's ``tilefish.fusion`` mode (``docs/rfcs/0001-plan-fusion.md``)."""
    from ._fusion import report

    return report(graph)


def verify_fusion(graph: GraphOp, buckets: "dict | None" = None) -> dict:
    """Run every fused region of ``graph`` of more than one op against its ops one by one, ``jit`` on torch's current CUDA device, on seeded
    inputs with special values, at every bucket combination of its plan (``buckets=`` as for :func:`tilefish.plan`), and compare the bytes; raises
    :class:`TilefishFusionError` at a difference. Returns the fusion report with what was checked (what ``export`` does by default)."""
    from ._fusion import report, verify
    from ._plan import bucket_combinations, plan as make_plan

    combos = bucket_combinations(make_plan(graph, buckets).dims)
    return report(graph, verify(graph, combos))


def advise(directory: "str | object", graph: "str | None" = None, **kwargs: object) -> object:
    """The fusion advice for the plan of ``graph`` in an export directory (``tilefish-explain DIR --graph G --advise``; ``docs/advise.md``): every
    candidate (steps joined by intermediates one kernel would not store), ranked by the time it would save, with why it is not one kernel today and what
    to do; the time by class; the graph-body arithmetic that was not fused. Keywords: ``bucket=``, ``timing=`` (what :func:`profile_plan` wrote),
    ``machine=``, ``bandwidth=``, ``keep=``, ``gop=``. Returns an ``Advice`` (``.text()``, ``.doc``). No GPU."""
    from ._advise import advise as _advise

    return _advise(directory, graph, **kwargs)  # type: ignore[arg-type]


def profile_plan(graph: GraphOp, plan: object = None, **kwargs: object) -> dict:
    """The device time of every step of the plan of ``graph`` at one bucket (``tilefish-explain DIR --graph G --profile OUT``; ``docs/advise.md``), as a
    ``tilefish.step-timing.v1`` document that :func:`advise` reads. A runtime path of its own: it changes no artefact, and puts back what its runs change
    that outlives them (the caller's ``InOut`` tensors, the carried buffers). Keywords:
    ``bucket=``, ``reps=``, ``warmup=``, ``method=`` (``"graph"`` or ``"events"``), ``mode=``, ``pin=``, ``trust_directory=``, ``externals=``,
    ``devscalars=``, ``seed=``, ``out=``. Needs a CUDA device."""
    from ._profile import profile_plan as _profile_plan

    return _profile_plan(graph, plan, **kwargs)  # type: ignore[arg-type]


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
    "Config",
    "Const",
    "CudnnTable",
    "DType",
    "DevScalar",
    "Dim",
    "DimExpr",
    "ExternOp",
    "Fusion",
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
    "Null",
    "Op",
    "Out",
    "Persistent",
    "Plan",
    "Requirement",
    "SassChecks",
    "Scalar",
    "StrideOf",
    "Strided",
    "Tensor",
    "TensorDesc",
    "TensorType",
    "TilefishAttributeError",
    "TilefishBuildError",
    "TilefishCacheWarning",
    "TilefishCudaMathWarning",
    "TilefishDeclarationError",
    "TilefishError",
    "TilefishLoweringError",
    "TilefishManifestError",
    "TilefishOrderWarning",
    "TilefishPicklingError",
    "TilefishPlanError",
    "TilefishPreconditionError",
    "TilefishRuntimeError",
    "TilefishSpecError",
    "TilefishTypeError",
    "TilefishFusionError",
    "TilefishFusionWarning",
    "Value",
    "advise",
    "aot",
    "argmax",
    "argmin",
    "barrier",
    "bf16",
    "bool_",
    "capture",
    "cdiv",
    "col2im1d",
    "config",
    "const",
    "contiguous",
    "cos",
    "div",
    "e2m1",
    "e4m3",
    "e5m2",
    "erf",
    "exp",
    "extern",
    "f16",
    "f32",
    "fma",
    "fusion",
    "fusion_report",
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
    "neg",
    "plan",
    "profile_plan",
    "persistent",
    "rcp",
    "rsqrt",
    "run",
    "sass",
    "scatter_rows",
    "silu",
    "sin",
    "sin_approx",
    "sqrt",
    "tanh",
    "tree",
    "u8",
    "ue8m0",
    "verify_fusion",
    "where",
    "zeros_like",
]
