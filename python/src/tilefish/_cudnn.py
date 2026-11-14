"""A ctypes binding of cuDNN for one operation: a forward convolution, run with a pinned choice of cuDNN.

The operation, in tilefish terms (every tensor dense in its layout, every pointer aligned to the problem's ``alignment``)::

    y[n, k, p...] = sum over c, r... of x[n, g * C/G + c, p * stride - padding + r * dilation ...] * w[k, c, r...]     (cross-correlation)

with ``G`` groups (``k`` in group ``g``), zero padding of ``padding`` on both sides of each spatial axis, an f32 accumulator (``CUDNN_DATA_FLOAT``) and
one rounding to the data type at the end. ``x``, ``w`` and ``y`` are of one type, ``f16``, ``bf16`` or ``f32``. The layout is ``"NCHW"`` (``x [N, C,
*S]``, ``w [K, C/G, *R]``, ``y [N, K, *P]``) or ``"NHWC"`` (``x [N, *S, C]``, ``w [K, *R, C/G]``, ``y [N, *P, K]``) for one, two or three spatial axes.
``alpha = 1``, ``beta = 0``: ``y`` is written, never read.

**How cuDNN is called.** A problem of one spatial axis is posed as one of two with a first spatial axis of size 1 (padding 0, stride 1, dilation 1):
cuDNN's way, and torch's (``conv1d`` views ``[N, C, L]`` as ``[N, C, 1, L]``) and what the legacy API's 4-D descriptors take. The tensors are given to cuDNN
in its order, ``[N, C, (D,) H, W]`` dims with the strides of the layout (:meth:`Problem.x_dims`, :meth:`Problem.x_strides`, ...; a stride of an axis of
size 1 is the product of the sizes of the axes inside it, which is what a dense tensor has and what torch's ``fixSizeOneDimStride`` writes). The
convolution is ``CUDNN_CROSS_CORRELATION`` with the compute type ``CUDNN_DATA_FLOAT``. Two APIs run it, and a pin is of one of them:

* **legacy** (``cudnnConvolutionForward``): ``cudnnSetTensorNdDescriptorEx`` with the layout's format for ``x`` and ``y``, ``cudnnSetFilterNdDescriptor``
  with that format for ``w``, ``cudnnSetConvolutionNdDescriptor``, ``cudnnSetConvolutionGroupCount`` and ``cudnnSetConvolutionMathType``
  (:data:`MATH_TYPES`: ``CUDNN_TENSOR_OP_MATH`` for ``"tensor_op"``, ``CUDNN_FMA_MATH`` for ``"fma"``, ``CUDNN_DEFAULT_MATH`` for ``"tf32"``). The pin is
  the algorithm enum (``cudnnConvolutionFwdAlgo_t``, 0 to 7). It names a family, not a kernel: which kernel of it runs is cuDNN's choice inside the
  call. The workspace size the call is given is the pin's ``workspace_bytes`` (the limit it was chosen under), at every launch;
* **backend** (``cudnnBackendExecute``): backend tensors (``x`` uid 120, ``w`` 119, ``y`` 121, as torch names them ``'x'``, ``'w'``, ``'y'``; the
  dims and strides above; ``CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT`` the problem's ``alignment``), a convolution descriptor (pre and post paddings both
  ``padding``), a ``CONVOLUTION_FORWARD`` operation (``alpha`` and ``beta`` as ``CUDNN_TYPE_FLOAT``) and an operation graph of it. The pin is an engine
  configuration: the engine's global index and every knob choice of it (:class:`EnginePin`), which an execution plan is finalized from; it names a
  kernel. ``sm_count_target`` sets ``CUDNN_ATTR_ENGINE_SM_COUNT_TARGET`` on the engine (and ``CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET`` on the heuristic).

**Pinning.** :func:`pin_conv` asks cuDNN under a *posture* (:data:`HEURISTICS`):

* ``"v7"`` (legacy): ``cudnnGetConvolutionForwardAlgorithm_v7``, all of its answers, and the first whose status is ``CUDNN_STATUS_SUCCESS``, whose
  determinism is ``CUDNN_DETERMINISTIC`` and whose ``cudnnGetConvolutionForwardWorkspaceSize`` is at most the limit;
* ``"instant"`` (backend): what torch 2.9 asks with ``torch.backends.cudnn.deterministic = True`` and ``benchmark = False`` (``Conv_v8.cpp``,
  ``get_configs_from_heuristics``): the engine configurations of the heuristic in ``CUDNN_HEUR_MODE_INSTANT``, then those of
  ``CUDNN_HEUR_MODE_FALLBACK``, each list filtered as torch filters it (no engine with the numerical note ``NONDETERMINISTIC`` or
  ``DOWN_CONVERT_INPUTS``, and for the math ``"fma"`` none with ``TENSOR_CORE``: torch's ``allow_tf32 = False`` for f32), and the first whose
  execution plan finalizes and needs at most the workspace limit;
* ``"mode_b"`` (backend): the same with ``CUDNN_HEUR_MODE_B`` first (torch's ``TORCH_CUDNN_USE_HEURISTIC_MODE_B``).

A pin is checked before it is recorded and before it runs (:func:`check_pin`): a legacy algorithm must be an answer of
``cudnnGetConvolutionForwardAlgorithm_v7`` for the problem with ``CUDNN_STATUS_SUCCESS`` and ``CUDNN_DETERMINISTIC`` and need at most the
workspace limit; an engine configuration must be of an engine whose numerical notes the math allows (as above), whose knob choices are the knobs of
the engine (each once, in its range and on its stride: ``CUDNN_ATTR_KNOB_INFO_*``), and whose plan finalizes and needs at most the limit.

**The process.** The cuDNN version (``cudnnGetVersion()``: ``91002`` for 9.10.2) is what a pin is compared on: cuDNN exposes no build number. And
``NVIDIA_TF32_OVERRIDE`` changes what cuDNN picks and runs behind the declared math (measured, with any value), so every
pin, check and run refuses a process that has it set (:func:`refuse_tf32_override`).

The library is the ``libcudnn`` the process already has mapped (torch's, from the ``nvidia-cudnn`` wheel): it is found in ``/proc/self/maps`` and opened
by that path; when none is mapped, ``import torch`` maps it (``libtorch_cuda`` needs it), and only then is the one of the ``nvidia.cudnn`` wheel opened.
The handle is one per device, created once under a lock; a call sets the handle's stream and runs under the handle's lock (the stream is state of the
handle). Descriptors, plans and legacy descriptors of a problem are made once per device and kept.
"""

from __future__ import annotations

import contextlib
import ctypes
import glob
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Iterator

from .errors import TilefishRuntimeError

INT32_MAX = (1 << 31) - 1
INT64_MAX = (1 << 63) - 1
UINT64_MAX = (1 << 64) - 1
WORKSPACE_ALIGNMENT = 256  # of a workspace the caller gives (as the cuBLASLt op's: a plan's arena buffer is)
ALIGNMENTS = (16, 32, 64, 128, 256)  # the byte alignment a problem may declare (CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, and the pointers' precondition)
DTYPES = ("f16", "bf16", "f32")
LAYOUTS = ("NCHW", "NHWC")
MATHS = ("tensor_op", "fma", "tf32")
APIS = ("legacy", "backend")
HEURISTICS = {"v7": "legacy", "instant": "backend", "mode_b": "backend"}  # posture -> the API it asks
MIN_SM_COUNT_TARGET = 1
TF32_OVERRIDE = "NVIDIA_TF32_OVERRIDE"
UIDS = {"x": 120, "w": 119, "y": 121}  # torch's: 'x', 'w', 'y'

# -- the values of the header ------------------------------------------------------------------------------------------------
# All of them are read from the cuDNN 9.10.2 headers of the nvidia-cudnn-cu12 wheel (cudnn_graph_v9.h, cudnn_ops_v9.h, cudnn_cnn_v9.h); none is a guess.

CUDNN_STATUS_SUCCESS = 0
# cudnnDataType_t
CUDNN_DATA_FLOAT = 0
CUDNN_DATA_HALF = 2
CUDNN_DATA_BFLOAT16 = 9
DATA_TYPES = {"f32": CUDNN_DATA_FLOAT, "f16": CUDNN_DATA_HALF, "bf16": CUDNN_DATA_BFLOAT16}
DATA_TYPE_NAMES = {"f32": "CUDNN_DATA_FLOAT", "f16": "CUDNN_DATA_HALF", "bf16": "CUDNN_DATA_BFLOAT16"}
# cudnnMathType_t
CUDNN_DEFAULT_MATH = 0
CUDNN_TENSOR_OP_MATH = 1
CUDNN_FMA_MATH = 3
MATH_TYPES = {"tensor_op": CUDNN_TENSOR_OP_MATH, "fma": CUDNN_FMA_MATH, "tf32": CUDNN_DEFAULT_MATH}
MATH_TYPE_NAMES = {"tensor_op": "CUDNN_TENSOR_OP_MATH", "fma": "CUDNN_FMA_MATH", "tf32": "CUDNN_DEFAULT_MATH"}
# cudnnTensorFormat_t
CUDNN_TENSOR_NCHW = 0
CUDNN_TENSOR_NHWC = 1
FORMATS = {"NCHW": CUDNN_TENSOR_NCHW, "NHWC": CUDNN_TENSOR_NHWC}
# cudnnConvolutionMode_t
CUDNN_CROSS_CORRELATION = 1
# cudnnDeterminism_t
CUDNN_DETERMINISTIC = 1
# cudnnConvolutionFwdAlgo_t
ALGO_NAMES = (
    "IMPLICIT_GEMM",
    "IMPLICIT_PRECOMP_GEMM",
    "GEMM",
    "DIRECT",
    "FFT",
    "FFT_TILING",
    "WINOGRAD",
    "WINOGRAD_NONFUSED",
)
ALGO_COUNT = 8
# cudnnBackendDescriptorType_t
D_CONVOLUTION = 1
D_ENGINE = 2
D_ENGINECFG = 3
D_ENGINEHEUR = 4
D_EXECUTION_PLAN = 5
D_KNOB_CHOICE = 7
D_KNOB_INFO = 8
D_OPERATION_CONVOLUTION_FORWARD = 10
D_OPERATIONGRAPH = 15
D_VARIANT_PACK = 16
D_TENSOR = 17
# cudnnBackendAttributeName_t
A_CONVOLUTION_COMP_TYPE = 100
A_CONVOLUTION_CONV_MODE = 101
A_CONVOLUTION_DILATIONS = 102
A_CONVOLUTION_FILTER_STRIDES = 103
A_CONVOLUTION_POST_PADDINGS = 104
A_CONVOLUTION_PRE_PADDINGS = 105
A_CONVOLUTION_SPATIAL_DIMS = 106
A_ENGINEHEUR_MODE = 200
A_ENGINEHEUR_OPERATION_GRAPH = 201
A_ENGINEHEUR_RESULTS = 202
A_ENGINEHEUR_SM_COUNT_TARGET = 203
A_ENGINECFG_ENGINE = 300
A_ENGINECFG_KNOB_CHOICES = 302
A_EXECUTION_PLAN_HANDLE = 400
A_EXECUTION_PLAN_ENGINE_CONFIG = 401
A_EXECUTION_PLAN_WORKSPACE_SIZE = 402
A_EXECUTION_PLAN_JSON_REPRESENTATION = 405
A_KNOB_CHOICE_KNOB_TYPE = 600
A_KNOB_CHOICE_KNOB_VALUE = 601
A_OPERATION_CONVOLUTION_FORWARD_ALPHA = 700
A_OPERATION_CONVOLUTION_FORWARD_BETA = 701
A_OPERATION_CONVOLUTION_FORWARD_CONV_DESC = 702
A_OPERATION_CONVOLUTION_FORWARD_W = 703
A_OPERATION_CONVOLUTION_FORWARD_X = 704
A_OPERATION_CONVOLUTION_FORWARD_Y = 705
A_OPERATIONGRAPH_HANDLE = 800
A_OPERATIONGRAPH_OPS = 801
A_TENSOR_BYTE_ALIGNMENT = 900
A_TENSOR_DATA_TYPE = 901
A_TENSOR_DIMENSIONS = 902
A_TENSOR_STRIDES = 903
A_TENSOR_UNIQUE_ID = 906
A_VARIANT_PACK_UNIQUE_IDS = 1000
A_VARIANT_PACK_DATA_POINTERS = 1001
A_VARIANT_PACK_WORKSPACE = 1003
A_KNOB_INFO_TYPE = 1200
A_KNOB_INFO_MAXIMUM_VALUE = 1201
A_KNOB_INFO_MINIMUM_VALUE = 1202
A_KNOB_INFO_STRIDE = 1203
A_ENGINE_OPERATION_GRAPH = 1300
A_ENGINE_GLOBAL_INDEX = 1301
A_ENGINE_KNOB_INFO = 1302
A_ENGINE_NUMERICAL_NOTE = 1303
A_ENGINE_BEHAVIOR_NOTE = 1305
A_ENGINE_SM_COUNT_TARGET = 1306
# cudnnBackendAttributeType_t
T_HANDLE = 0
T_DATA_TYPE = 1
T_INT64 = 3
T_FLOAT = 4
T_VOID_PTR = 6
T_CONVOLUTION_MODE = 7
T_HEUR_MODE = 8
T_KNOB_TYPE = 9
T_NUMERICAL_NOTE = 11
T_BACKEND_DESCRIPTOR = 15
T_BEHAVIOR_NOTE = 19
T_INT32 = 23
T_CHAR = 24
# cudnnBackendHeurMode_t
HEUR_MODE_INSTANT = 0
HEUR_MODE_B = 1
HEUR_MODE_FALLBACK = 2
HEUR_MODES = {"instant": (HEUR_MODE_INSTANT, HEUR_MODE_FALLBACK), "mode_b": (HEUR_MODE_B, HEUR_MODE_FALLBACK)}
# cudnnBackendNumericalNote_t
NOTE_TENSOR_CORE = 0
NOTE_DOWN_CONVERT_INPUTS = 1
NOTE_REDUCED_PRECISION_REDUCTION = 2
NOTE_FFT = 3
NOTE_NONDETERMINISTIC = 4
NOTE_WINOGRAD = 5
NOTE_NAMES = {
    0: "TENSOR_CORE",
    1: "DOWN_CONVERT_INPUTS",
    2: "REDUCED_PRECISION_REDUCTION",
    3: "FFT",
    4: "NONDETERMINISTIC",
    5: "WINOGRAD",
    6: "WINOGRAD_TILE_4x4",
    7: "WINOGRAD_TILE_6x6",
    8: "WINOGRAD_TILE_13x13",
    9: "STRICT_NAN_PROP",
}
# cudnnBackendBehaviorNote_t
BEHAVIOR_NAMES = {0: "RUNTIME_COMPILATION", 1: "REQUIRES_FILTER_INT8x32_REORDER", 2: "REQUIRES_BIAS_INT8x32_REORDER", 3: "SUPPORTS_CUDA_GRAPH_NATIVE_API"}
KNOB_TYPE_COUNT = 43  # CUDNN_KNOB_TYPE_COUNTS in 9.10.2: knob types are 0 to 42
MAX_NOTES = 16


# -- the problem -------------------------------------------------------------------------------------------------------------


def _int(name: str, value: object, low: int, high: int = INT32_MAX) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TilefishRuntimeError(f"cuDNN convolution: {name} must be an int, got {value!r}")
    if not low <= value <= high:
        raise TilefishRuntimeError(f"cuDNN convolution: {name} is {value}, outside {low}..{high}")
    return value


def _ints(name: str, values: object, count: int, low: int) -> "tuple[int, ...]":
    if not isinstance(values, (tuple, list)) or len(values) != count:
        raise TilefishRuntimeError(f"cuDNN convolution: {name} must be {count} ints (one per spatial axis), got {values!r}")
    return tuple(_int(f"{name}[{i}]", v, low) for i, v in enumerate(values))


def output_size(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    """The output size of one spatial axis: ``(size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1`` (at most 0: no output). Pure."""
    span = size + 2 * padding - dilation * (kernel - 1) - 1
    return span // stride + 1 if span >= 0 else 0


def dense_strides(dims: "tuple[int, ...]", layout: str) -> "tuple[int, ...]":
    """The element strides of a dense tensor of cuDNN dims ``[N, C, *spatial]`` stored in ``layout`` (``NCHW``: row-major in that order; ``NHWC``:
    ``C`` innermost, then the spatial axes from the last, then ``N``). An axis of size 1 has the product of the sizes inside it, as every other. Pure."""
    rank = len(dims)
    order = list(range(rank - 1, -1, -1)) if layout == "NCHW" else [1, *range(rank - 1, 1, -1), 0]  # innermost first
    strides = [0] * rank
    z = 1
    for d in order:
        strides[d] = z
        z *= dims[d]
    return tuple(strides)


@dataclass(frozen=True)
class Problem:
    """A forward convolution as tilefish declares it: the data type of ``x``, ``w`` and ``y``, the layout, the batch ``n``, the input channels ``c``,
    the output channels ``k``, the groups, the spatial sizes of the input and of the filter and, per spatial axis, the stride, the (symmetric) zero
    padding and the dilation, and the math (:data:`MATHS`) and the byte alignment its tensors are declared with (:data:`ALIGNMENTS`)."""

    dtype: str
    layout: str
    n: int
    c: int
    k: int
    groups: int
    input: "tuple[int, ...]"
    filter: "tuple[int, ...]"
    stride: "tuple[int, ...]"
    padding: "tuple[int, ...]"
    dilation: "tuple[int, ...]"
    math: str
    alignment: int = 16

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise TilefishRuntimeError(f"cuDNN convolution: the data type is one of {list(DTYPES)}, got {self.dtype!r}")
        if self.layout not in LAYOUTS:
            raise TilefishRuntimeError(f"cuDNN convolution: the layout is one of {list(LAYOUTS)}, got {self.layout!r}")
        if self.math not in MATHS:
            raise TilefishRuntimeError(f"cuDNN convolution: the math is one of {list(MATHS)}, got {self.math!r}")
        problem = math_problem(self.dtype, self.math)
        if problem is not None:
            raise TilefishRuntimeError(f"cuDNN convolution: {problem}")
        if self.alignment not in ALIGNMENTS:
            raise TilefishRuntimeError(f"cuDNN convolution: the alignment is one of {list(ALIGNMENTS)} bytes, got {self.alignment!r}")
        for name in ("n", "c", "k", "groups"):
            _int(name, getattr(self, name), 1)
        if not isinstance(self.input, tuple) or not 1 <= len(self.input) <= 3:
            raise TilefishRuntimeError(f"cuDNN convolution: input is the sizes of 1 to 3 spatial axes, got {self.input!r}")
        sp = len(self.input)
        _ints("input", self.input, sp, 1)
        _ints("filter", self.filter, sp, 1)
        _ints("stride", self.stride, sp, 1)
        _ints("padding", self.padding, sp, 0)
        _ints("dilation", self.dilation, sp, 1)
        if self.c % self.groups or self.k % self.groups:
            raise TilefishRuntimeError(f"cuDNN convolution: the groups ({self.groups}) divide c ({self.c}) and k ({self.k})")
        for axis, size in enumerate(self.output):
            if size < 1:
                raise TilefishRuntimeError(
                    f"cuDNN convolution: spatial axis {axis} has no output (input {self.input[axis]}, filter {self.filter[axis]}, padding "
                    f"{self.padding[axis]}, dilation {self.dilation[axis]})"
                )
        for name, dims in (("x", self.x_dims), ("w", self.w_dims), ("y", self.y_dims)):
            count = 1
            for d in dims:
                count *= d
            if count > INT32_MAX:
                raise TilefishRuntimeError(f"cuDNN convolution: {name} has {count} elements, more than an int32 counts")

    @property
    def spatial(self) -> int:
        return len(self.input)

    @property
    def output(self) -> "tuple[int, ...]":
        return tuple(output_size(*a) for a in zip(self.input, self.filter, self.stride, self.padding, self.dilation))

    def _posed(self, values: "tuple[int, ...]", one: int) -> "tuple[int, ...]":
        """A spatial tuple as cuDNN is given it: a first axis of ``one`` before a single one."""
        return (one, *values) if self.spatial == 1 else values

    @property
    def x_dims(self) -> "tuple[int, ...]":
        return (self.n, self.c, *self._posed(self.input, 1))

    @property
    def w_dims(self) -> "tuple[int, ...]":
        return (self.k, self.c // self.groups, *self._posed(self.filter, 1))

    @property
    def y_dims(self) -> "tuple[int, ...]":
        return (self.n, self.k, *self._posed(self.output, 1))

    @property
    def x_strides(self) -> "tuple[int, ...]":
        return dense_strides(self.x_dims, self.layout)

    @property
    def w_strides(self) -> "tuple[int, ...]":
        return dense_strides(self.w_dims, self.layout)

    @property
    def y_strides(self) -> "tuple[int, ...]":
        return dense_strides(self.y_dims, self.layout)

    @property
    def conv_padding(self) -> "tuple[int, ...]":
        return self._posed(self.padding, 0)

    @property
    def conv_stride(self) -> "tuple[int, ...]":
        return self._posed(self.stride, 1)

    @property
    def conv_dilation(self) -> "tuple[int, ...]":
        return self._posed(self.dilation, 1)

    @property
    def what(self) -> str:
        """The problem in a message: ``"the f16 NHWC convolution x [1, 646, 64] by w [64, 7, 64] (stride 1, padding 6, dilation 1)"``."""
        x = self.tilefish_shape("x")
        w = self.tilefish_shape("w")
        groups = f", {self.groups} groups" if self.groups != 1 else ""
        return (
            f"the {self.dtype} {self.layout} convolution x {list(x)} by w {list(w)} (stride {_axes(self.stride)}, padding {_axes(self.padding)}, "
            f"dilation {_axes(self.dilation)}{groups}, math {self.math})"
        )

    def tilefish_shape(self, name: str) -> "tuple[int, ...]":
        """The shape tilefish gives the tensor ``name`` (``x``, ``w``, ``y``) in the layout: ``[N, C, *S]`` or ``[N, *S, C]``."""
        batch, channels, spatial = {
            "x": (self.n, self.c, self.input),
            "w": (self.k, self.c // self.groups, self.filter),
            "y": (self.n, self.k, self.output),
        }[name]
        return (batch, channels, *spatial) if self.layout == "NCHW" else (batch, *spatial, channels)


def _axes(values: "tuple[int, ...]") -> str:
    return str(values[0]) if len(values) == 1 else str(list(values))


def math_problem(dtype: str, math: str) -> "str | None":
    """What is wrong with ``math`` for data of ``dtype`` (``None``: nothing). 16-bit data: ``"tensor_op"`` or ``"fma"``; f32 data: ``"fma"`` (TF32 off)
    or ``"tf32"`` (TF32 on), said explicitly. Pure."""
    if dtype == "f32" and math not in ("fma", "tf32"):
        return f"f32 data needs the math said explicitly, 'fma' (TF32 off) or 'tf32' (TF32 on), got {math!r}"
    if dtype != "f32" and math not in ("tensor_op", "fma"):
        return f"{dtype} data has the math 'tensor_op' (tensor cores, f32 accumulation) or 'fma', got {math!r}"
    return None


def describe(problem: Problem) -> "dict[str, Any]":
    """The exact cuDNN-level problem a :class:`Problem` is posed as (pure: no GPU, no library). The Rust host's ``describe`` gives the same
    (``crates/tilefish/tests/python_conv.rs``)."""
    return {
        "library": "cudnn",
        "operation": "convolution_forward",
        "data_type": DATA_TYPE_NAMES[problem.dtype],
        "compute_type": "CUDNN_DATA_FLOAT",
        "format": f"CUDNN_TENSOR_{problem.layout}",
        "math_type": MATH_TYPE_NAMES[problem.math],
        "conv_mode": "CUDNN_CROSS_CORRELATION",
        "x_dims": list(problem.x_dims),
        "x_strides": list(problem.x_strides),
        "w_dims": list(problem.w_dims),
        "w_strides": list(problem.w_strides),
        "y_dims": list(problem.y_dims),
        "y_strides": list(problem.y_strides),
        "padding": list(problem.conv_padding),
        "stride": list(problem.conv_stride),
        "dilation": list(problem.conv_dilation),
        "groups": problem.groups,
        "alignment": problem.alignment,
        "uids": dict(UIDS),
        "alpha": 1.0,
        "beta": 0.0,
    }


# -- pins ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LegacyPin:
    """A legacy pin: the algorithm enum ``cudnnConvolutionFwdAlgo_t`` (0 to 7), run with the math type of the problem and a workspace of the pin's
    limit."""

    algo: int

    def __post_init__(self) -> None:
        if isinstance(self.algo, bool) or not isinstance(self.algo, int) or not 0 <= self.algo < ALGO_COUNT:
            raise TilefishRuntimeError(f"a legacy cuDNN pin is an algorithm enum from 0 to {ALGO_COUNT - 1}, got {self.algo!r}")

    @property
    def api(self) -> str:
        return "legacy"

    def to_json(self) -> "dict[str, Any]":
        return {"api": "legacy", "algo": self.algo}

    @property
    def what(self) -> str:
        return f"algorithm {self.algo} (CUDNN_CONVOLUTION_FWD_ALGO_{ALGO_NAMES[self.algo]})"


@dataclass(frozen=True)
class EnginePin:
    """A backend pin: an engine configuration, the engine's global index (``CUDNN_ATTR_ENGINE_GLOBAL_INDEX``) and its knob choices, ``(knob type,
    value)`` pairs sorted by type, each type once (``CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE`` and ``_KNOB_VALUE``)."""

    engine: int
    knobs: "tuple[tuple[int, int], ...]" = ()

    def __post_init__(self) -> None:
        if isinstance(self.engine, bool) or not isinstance(self.engine, int) or not 0 <= self.engine <= INT64_MAX:
            raise TilefishRuntimeError(f"a backend cuDNN pin has an engine index from 0, got {self.engine!r}")
        knobs = self.knobs
        if not isinstance(knobs, tuple) or any(not isinstance(kv, tuple) or len(kv) != 2 for kv in knobs):
            raise TilefishRuntimeError(f"the knobs of a backend cuDNN pin are (type, value) pairs, got {knobs!r}")
        for t, v in knobs:
            if isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < KNOB_TYPE_COUNT:
                raise TilefishRuntimeError(f"a knob type of a backend cuDNN pin is an int from 0 to {KNOB_TYPE_COUNT - 1}, got {t!r}")
            if isinstance(v, bool) or not isinstance(v, int) or not -(1 << 63) <= v <= INT64_MAX:
                raise TilefishRuntimeError(f"a knob value of a backend cuDNN pin is an int64, got {v!r}")
        types = [t for t, _ in knobs]
        if types != sorted(set(types)):
            raise TilefishRuntimeError(f"the knobs of a backend cuDNN pin are sorted by type, each type once, got {list(knobs)}")

    @property
    def api(self) -> str:
        return "backend"

    def to_json(self) -> "dict[str, Any]":
        return {"api": "backend", "engine": self.engine, "knobs": [[t, v] for t, v in self.knobs]}

    @property
    def what(self) -> str:
        return f"engine {self.engine} with knobs {[list(kv) for kv in self.knobs]}"


def pin_of(value: object, what: str = "a cuDNN pin") -> "LegacyPin | EnginePin":
    """A pin from what an entry or a table holds: ``{"api": "legacy", "algo": a}``, ``{"api": "backend", "engine": e, "knobs": [[t, v], ...]}``; for
    a table also an int (a legacy algorithm) or ``{"engine": e, "knobs": {t: v} or [[t, v], ...]}``. Pure."""
    if isinstance(value, (LegacyPin, EnginePin)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return LegacyPin(value)
    if not isinstance(value, dict):
        raise TilefishRuntimeError(f"{what} is {{'api': 'legacy', 'algo': ...}} or {{'api': 'backend', 'engine': ..., 'knobs': [...]}}, got {value!r}")
    api = value.get("api", "backend" if "engine" in value else None)
    if api == "legacy":
        if set(value) != {"api", "algo"}:
            raise TilefishRuntimeError(f"{what}: a legacy pin has exactly the keys api and algo, got {sorted(value)}")
        return LegacyPin(value["algo"])
    if api == "backend":
        if set(value) - {"api"} != {"engine", "knobs"}:
            raise TilefishRuntimeError(f"{what}: a backend pin has exactly the keys engine and knobs (and api), got {sorted(value)}")
        knobs = value["knobs"]
        if isinstance(knobs, dict):
            pairs = tuple(sorted((k, v) for k, v in knobs.items()))
        elif isinstance(knobs, (list, tuple)):
            pairs = tuple(tuple(kv) if isinstance(kv, (list, tuple)) else kv for kv in knobs)  # type: ignore[misc]
        else:
            raise TilefishRuntimeError(f"{what}: the knobs are a list of [type, value] pairs, got {knobs!r}")
        return EnginePin(value["engine"], pairs)  # type: ignore[arg-type]
    raise TilefishRuntimeError(f"{what}: api is 'legacy' or 'backend', got {api!r}")


def notes_problem(notes: "list[int]", math: str) -> "str | None":
    """What is wrong with an engine of these numerical notes for ``math`` (``None``: nothing): torch's filter (``filterEngineConfigs`` with
    ``deterministic = True``; ``allow_tf32 = False`` is the math ``"fma"``), which every pin is held to. Pure: the Rust host's ``notes_problem`` has the
    same words."""
    if NOTE_NONDETERMINISTIC in notes:
        return "it is not deterministic (the numerical note NONDETERMINISTIC)"
    if NOTE_DOWN_CONVERT_INPUTS in notes:
        return "it converts its inputs down (the numerical note DOWN_CONVERT_INPUTS)"
    if math == "fma" and NOTE_TENSOR_CORE in notes:
        return "it uses tensor cores (the numerical note TENSOR_CORE), and the math is 'fma'"
    return None


def knobs_problem(knobs: "tuple[tuple[int, int], ...]", infos: "dict[int, tuple[int, int, int]]") -> "str | None":
    """What is wrong with knob choices for an engine whose knobs are ``infos`` (type -> (minimum, maximum, stride)): every choice is of a knob of the
    engine, in its range and on its stride. Pure."""
    for t, v in knobs:
        if t not in infos:
            return f"knob type {t} is not a knob of the engine (it has {sorted(infos)})"
        lo, hi, step = infos[t]
        if not lo <= v <= hi or (step > 0 and (v - lo) % step):
            return f"knob {t} = {v} is outside the engine's {lo}..{hi} (stride {step})"
    return None


def refuse_tf32_override(environ: "Any" = None) -> None:
    """Raise when ``NVIDIA_TF32_OVERRIDE`` is set in the process's environment (any value, empty included)."""
    env = os.environ if environ is None else environ
    if TF32_OVERRIDE in env:
        raise TilefishRuntimeError(
            f"{TF32_OVERRIDE} is set (to {env[TF32_OVERRIDE]!r}) in this process: it changes the algorithms cuDNN chooses and runs behind the declared "
            "math (TF32 on or off), so a pinned cuDNN convolution is neither pinned nor run here; unset it"
        )


# -- the structs of the header -----------------------------------------------------------------------------------------------


class _AlgoPerf(ctypes.Structure):
    """``cudnnConvolutionFwdAlgoPerf_t``: algo, status, time, memory, determinism, mathType, reserved[3]."""

    _fields_ = [
        ("algo", ctypes.c_int),
        ("status", ctypes.c_int),
        ("time", ctypes.c_float),
        ("memory", ctypes.c_size_t),
        ("determinism", ctypes.c_int),
        ("mathType", ctypes.c_int),
        ("reserved", ctypes.c_int * 3),
    ]


assert ctypes.sizeof(_AlgoPerf) == 48


def _functions() -> "dict[str, tuple[Any, list[Any]]]":
    c = ctypes
    vp, i32, i64 = c.c_void_p, c.c_int, c.c_int64
    pvp = c.POINTER(c.c_void_p)
    pi32 = c.POINTER(c.c_int)
    return {
        "cudnnGetVersion": (c.c_size_t, []),
        "cudnnGetErrorString": (c.c_char_p, [i32]),
        "cudnnGetLastErrorString": (None, [c.c_char_p, c.c_size_t]),
        "cudnnCreate": (i32, [pvp]),
        "cudnnDestroy": (i32, [vp]),
        "cudnnSetStream": (i32, [vp, vp]),
        "cudnnCreateTensorDescriptor": (i32, [pvp]),
        "cudnnSetTensorNdDescriptorEx": (i32, [vp, i32, i32, i32, pi32]),
        "cudnnDestroyTensorDescriptor": (i32, [vp]),
        "cudnnCreateFilterDescriptor": (i32, [pvp]),
        "cudnnSetFilterNdDescriptor": (i32, [vp, i32, i32, i32, pi32]),
        "cudnnDestroyFilterDescriptor": (i32, [vp]),
        "cudnnCreateConvolutionDescriptor": (i32, [pvp]),
        "cudnnSetConvolutionNdDescriptor": (i32, [vp, i32, pi32, pi32, pi32, i32, i32]),
        "cudnnSetConvolutionGroupCount": (i32, [vp, i32]),
        "cudnnSetConvolutionMathType": (i32, [vp, i32]),
        "cudnnDestroyConvolutionDescriptor": (i32, [vp]),
        "cudnnGetConvolutionNdForwardOutputDim": (i32, [vp, vp, vp, i32, pi32]),
        "cudnnGetConvolutionForwardAlgorithmMaxCount": (i32, [vp, pi32]),
        "cudnnGetConvolutionForwardAlgorithm_v7": (i32, [vp, vp, vp, vp, vp, i32, pi32, c.POINTER(_AlgoPerf)]),
        "cudnnGetConvolutionForwardWorkspaceSize": (i32, [vp, vp, vp, vp, vp, i32, c.POINTER(c.c_size_t)]),
        "cudnnConvolutionForward": (i32, [vp, vp, vp, vp, vp, vp, vp, i32, vp, c.c_size_t, vp, vp, vp]),
        "cudnnBackendCreateDescriptor": (i32, [i32, pvp]),
        "cudnnBackendDestroyDescriptor": (i32, [vp]),
        "cudnnBackendFinalize": (i32, [vp]),
        "cudnnBackendSetAttribute": (i32, [vp, i32, i32, i64, vp]),
        "cudnnBackendGetAttribute": (i32, [vp, i32, i32, i64, c.POINTER(i64), vp]),
        "cudnnBackendExecute": (i32, [vp, vp, vp]),
    }


# -- finding and loading the library -----------------------------------------------------------------------------------------

_MAPS = "/proc/self/maps"
_LIB_NAME = re.compile(r"^libcudnn\.so(\.\d+)*$")
_NOT_FOUND = (
    "cuDNN is not available: no libcudnn is mapped into this process, importing torch mapped none, and the nvidia-cudnn wheel (the 'nvidia.cudnn' "
    "package) is not installed; install torch with CUDA or 'pip install nvidia-cudnn-cu12'"
)

_lib_lock = threading.Lock()
_library: "_Library | None" = None


def _mapped_paths() -> "list[str]":
    """The distinct real paths of the libcudnn files (the main library, not its sub-libraries) this process has mapped."""
    try:
        with open(_MAPS, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return []
    found: set[str] = set()
    for line in text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        path = parts[5].removesuffix(" (deleted)")
        if _LIB_NAME.match(os.path.basename(path)):
            found.add(os.path.realpath(path))
    return sorted(found)


def _wheel_path() -> "str | None":
    import importlib.util

    try:
        spec = importlib.util.find_spec("nvidia.cudnn.lib")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for directory in spec.submodule_search_locations:
        candidates = sorted(p for p in glob.glob(os.path.join(directory, "libcudnn.so*")) if _LIB_NAME.match(os.path.basename(p)))
        if candidates:
            return os.path.realpath(candidates[-1])
    return None


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:
        raise TilefishRuntimeError("the cuDNN convolution needs torch with CUDA (for the device and the workspace): pip install 'tilefish[gpu]'") from e
    return torch


def library_path() -> str:
    """The path of the ``libcudnn`` this binding uses: the one this process has mapped, else the one of the nvidia-cudnn wheel."""
    if _library is not None:
        return _library.path
    mapped = _mapped_paths()
    if not mapped:
        with contextlib.suppress(TilefishRuntimeError):
            _torch()  # libtorch_cuda needs libcudnn: importing torch maps it
        mapped = _mapped_paths()
    if len(mapped) > 1:
        raise TilefishRuntimeError(
            f"this process has mapped {len(mapped)} different libcudnn files ({', '.join(mapped)}); cuDNN is bound to the one torch uses, and which "
            "that is cannot be told"
        )
    if mapped:
        return mapped[0]
    wheel = _wheel_path()
    if wheel is None:
        raise TilefishRuntimeError(_NOT_FOUND)
    return wheel


class _Library:
    """The loaded ``libcudnn`` with the prototypes of every function this module calls."""

    def __init__(self, path: str) -> None:
        real = os.path.realpath(path)
        try:
            self.cdll = ctypes.CDLL(path)
        except OSError as e:
            raise TilefishRuntimeError(f"cuDNN ({path}) cannot be loaded: {e}") from e
        if os.path.exists(_MAPS):
            after = _mapped_paths()
            if after != [real]:
                raise TilefishRuntimeError(f"opening {path} must not map a second libcudnn, but this process now has {after or 'none'} mapped (expected {[real]})")
        self.path = real
        for name, (restype, argtypes) in _functions().items():
            try:
                fn = getattr(self.cdll, name)
            except AttributeError as e:
                raise TilefishRuntimeError(f"cuDNN ({path}) has no {name}: it is too old for this binding (cuDNN 9 is needed)") from e
            fn.restype = restype
            fn.argtypes = argtypes
            setattr(self, name, fn)

    def status_name(self, status: int) -> str:
        try:
            raw = self.cudnnGetErrorString(status)  # type: ignore[attr-defined]
            name = raw.decode("ascii", "replace") if raw else None
        except Exception:  # noqa: BLE001 - the name is a nicety
            name = None
        return f"{name} ({status})" if name else f"status {status}"

    def last_error(self) -> str:
        buffer = ctypes.create_string_buffer(1024)
        try:
            self.cudnnGetLastErrorString(buffer, len(buffer))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - a nicety
            return ""
        return buffer.value.decode("utf-8", "replace").strip()

    def call(self, name: str, *args: Any) -> None:
        status = getattr(self, name)(*args)
        if status != CUDNN_STATUS_SUCCESS:
            detail = self.last_error()
            raise TilefishRuntimeError(f"cuDNN call {name} failed: {self.status_name(status)}" + (f" ({detail})" if detail else ""))

    def attempt(self, name: str, *args: Any) -> int:
        """A call whose failure is an answer (a configuration that does not finalize): its status."""
        return int(getattr(self, name)(*args))


def _lib() -> _Library:
    global _library
    lib = _library
    if lib is None:
        with _lib_lock:
            if _library is None:
                _library = _Library(library_path())
            lib = _library
    return lib


def version() -> int:
    """``cudnnGetVersion()``: the version of the cuDNN this process uses (``91002`` for 9.10.2: ``major * 10000 + minor * 100 + patch``). This is
    all a pin is compared on: cuDNN exposes no build number."""
    return int(_lib().cudnnGetVersion())  # type: ignore[attr-defined]


def available() -> bool:
    """Whether a libcudnn can be found (it is loaded when it is)."""
    try:
        _lib()
    except TilefishRuntimeError:
        return False
    return True


# -- the device and the handle -----------------------------------------------------------------------------------------------


@dataclass
class _Handle:
    handle: ctypes.c_void_p
    lock: threading.Lock


_handle_lock = threading.Lock()
_handles: "dict[int, _Handle]" = {}


def _device(device_index: "int | None") -> int:
    torch = _torch()
    if not torch.cuda.is_available():
        raise TilefishRuntimeError("the cuDNN convolution needs a CUDA device, and torch sees none")
    if device_index is None:
        return int(torch.cuda.current_device())
    index = _int("device_index", device_index, 0)
    if index >= torch.cuda.device_count():
        raise TilefishRuntimeError(f"cuDNN convolution: device_index {index} is out of range, torch sees {torch.cuda.device_count()} device(s)")
    return index


@contextlib.contextmanager
def _on_device(index: int) -> Iterator[None]:
    with _torch().cuda.device(index):
        yield


def _handle(lib: _Library, index: int) -> _Handle:
    """The cuDNN handle of a device, created on first use (one per device, under a lock)."""
    h = _handles.get(index)
    if h is None:
        with _handle_lock:
            h = _handles.get(index)
            if h is None:
                raw = ctypes.c_void_p()
                with _on_device(index):
                    lib.call("cudnnCreate", ctypes.byref(raw))
                h = _handles[index] = _Handle(raw, threading.Lock())
    return h


def warm(device_index: "int | None" = None) -> None:
    """Load cuDNN and create the handle of a device (a launch plan does it before its first launch: creating a handle allocates device memory)."""
    index = _device(device_index)
    _handle(_lib(), index)


def sm_count(index: int) -> int:
    return int(_torch().cuda.get_device_properties(index).multi_processor_count)


# -- legacy descriptors ------------------------------------------------------------------------------------------------------


def _int_array(values: "tuple[int, ...] | list[int]") -> Any:
    return (ctypes.c_int * len(values))(*values)


class _Legacy:
    """The legacy descriptors of a problem: ``x``, ``w``, the convolution and ``y``. Destroyed by :meth:`close`."""

    def __init__(self, lib: _Library, problem: Problem) -> None:
        self.lib = lib
        self.problem = problem
        self._owned: "list[tuple[str, ctypes.c_void_p]]" = []
        dtype, fmt = DATA_TYPES[problem.dtype], FORMATS[problem.layout]
        try:
            self.x = self._create("cudnnCreateTensorDescriptor", "cudnnDestroyTensorDescriptor")
            lib.call("cudnnSetTensorNdDescriptorEx", self.x, fmt, dtype, len(problem.x_dims), _int_array(problem.x_dims))
            self.w = self._create("cudnnCreateFilterDescriptor", "cudnnDestroyFilterDescriptor")
            lib.call("cudnnSetFilterNdDescriptor", self.w, dtype, fmt, len(problem.w_dims), _int_array(problem.w_dims))
            self.conv = self._create("cudnnCreateConvolutionDescriptor", "cudnnDestroyConvolutionDescriptor")
            n = len(problem.conv_padding)
            lib.call(
                "cudnnSetConvolutionNdDescriptor", self.conv, n, _int_array(problem.conv_padding), _int_array(problem.conv_stride),
                _int_array(problem.conv_dilation), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT,
            )  # fmt: skip
            lib.call("cudnnSetConvolutionGroupCount", self.conv, problem.groups)
            lib.call("cudnnSetConvolutionMathType", self.conv, MATH_TYPES[problem.math])
            self.y = self._create("cudnnCreateTensorDescriptor", "cudnnDestroyTensorDescriptor")
            lib.call("cudnnSetTensorNdDescriptorEx", self.y, fmt, dtype, len(problem.y_dims), _int_array(problem.y_dims))
            dims = (ctypes.c_int * len(problem.y_dims))()
            lib.call("cudnnGetConvolutionNdForwardOutputDim", self.conv, self.x, self.w, len(problem.y_dims), dims)
            if tuple(dims) != problem.y_dims:  # pragma: no cover - a bug of the problem's arithmetic or of the library
                raise TilefishRuntimeError(f"cuDNN says the output of {problem.what} is {list(dims)}, not {list(problem.y_dims)}")
        except BaseException:
            self.close()
            raise

    def _create(self, create: str, destroy: str) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        self.lib.call(create, ctypes.byref(handle))
        self._owned.append((destroy, handle))
        return handle

    def close(self) -> None:
        owned, self._owned = self._owned, []
        for destroy, handle in reversed(owned):
            with contextlib.suppress(TilefishRuntimeError):
                self.lib.call(destroy, handle)

    def candidates(self, handle: ctypes.c_void_p) -> "list[dict[str, int]]":
        """``cudnnGetConvolutionForwardAlgorithm_v7``: every answer, in its order."""
        count = ctypes.c_int(0)
        self.lib.call("cudnnGetConvolutionForwardAlgorithmMaxCount", handle, ctypes.byref(count))
        perf = (_AlgoPerf * max(count.value, 1))()
        returned = ctypes.c_int(0)
        self.lib.call("cudnnGetConvolutionForwardAlgorithm_v7", handle, self.x, self.w, self.conv, self.y, count.value, ctypes.byref(returned), perf)
        return [
            {"algo": p.algo, "status": p.status, "memory": int(p.memory), "determinism": p.determinism, "math_type": p.mathType}
            for p in perf[: returned.value]
        ]

    def workspace(self, handle: ctypes.c_void_p, algo: int) -> "int | None":
        """``cudnnGetConvolutionForwardWorkspaceSize`` of ``algo``, or ``None`` when the library refuses the algorithm for the problem."""
        size = ctypes.c_size_t(0)
        status = self.lib.attempt("cudnnGetConvolutionForwardWorkspaceSize", handle, self.x, self.w, self.conv, self.y, algo, ctypes.byref(size))
        return int(size.value) if status == CUDNN_STATUS_SUCCESS else None


# -- backend descriptors -----------------------------------------------------------------------------------------------------


class _Backend:
    """Backend descriptors, each destroyed by :meth:`close` (in the reverse order of their making)."""

    def __init__(self, lib: _Library) -> None:
        self.lib = lib
        self._owned: "list[ctypes.c_void_p]" = []

    def make(self, kind: int) -> ctypes.c_void_p:
        desc = ctypes.c_void_p()
        self.lib.call("cudnnBackendCreateDescriptor", kind, ctypes.byref(desc))
        self._owned.append(desc)
        return desc

    def set(self, desc: ctypes.c_void_p, attr: int, kind: int, value: Any) -> None:
        """Set one value (a ctypes scalar of the attribute's type)."""
        self.lib.call("cudnnBackendSetAttribute", desc, attr, kind, 1, ctypes.byref(value))

    def set_int64s(self, desc: ctypes.c_void_p, attr: int, values: "tuple[int, ...] | list[int]") -> None:
        array = (ctypes.c_int64 * len(values))(*values)
        self.lib.call("cudnnBackendSetAttribute", desc, attr, T_INT64, len(values), array)

    def set_descs(self, desc: ctypes.c_void_p, attr: int, values: "list[ctypes.c_void_p]") -> None:
        array = (ctypes.c_void_p * len(values))(*[v.value for v in values])
        self.lib.call("cudnnBackendSetAttribute", desc, attr, T_BACKEND_DESCRIPTOR, len(values), array)

    def finalize(self, desc: ctypes.c_void_p) -> None:
        self.lib.call("cudnnBackendFinalize", desc)

    def get_int64s(self, desc: ctypes.c_void_p, attr: int, most: int = 16) -> "list[int]":
        array = (ctypes.c_int64 * most)()
        count = ctypes.c_int64(0)
        self.lib.call("cudnnBackendGetAttribute", desc, attr, T_INT64, most, ctypes.byref(count), array)
        return list(array[: count.value])

    def get_enums(self, desc: ctypes.c_void_p, attr: int, kind: int, most: int = MAX_NOTES) -> "list[int]":
        array = (ctypes.c_int * most)()
        count = ctypes.c_int64(0)
        self.lib.call("cudnnBackendGetAttribute", desc, attr, kind, most, ctypes.byref(count), array)
        return list(array[: count.value])

    def get_descs(self, desc: ctypes.c_void_p, attr: int, kind: int, most: int) -> "list[ctypes.c_void_p]":
        """A descriptor-valued attribute: ``most`` descriptors of ``kind`` are made to receive it (the library fills descriptors it is given)."""
        made = [self.make(kind) for _ in range(most)]
        array = (ctypes.c_void_p * most)(*[d.value for d in made])
        count = ctypes.c_int64(0)
        self.lib.call("cudnnBackendGetAttribute", desc, attr, T_BACKEND_DESCRIPTOR, most, ctypes.byref(count), array)
        return made[: count.value]

    def count_of(self, desc: ctypes.c_void_p, attr: int, kind: int) -> int:
        count = ctypes.c_int64(0)
        self.lib.call("cudnnBackendGetAttribute", desc, attr, kind, 0, ctypes.byref(count), None)
        return int(count.value)

    def json_of(self, plan: ctypes.c_void_p) -> str:
        """``CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION`` of a finalized plan (empty when the library gives none)."""
        count = ctypes.c_int64(0)
        if self.lib.attempt("cudnnBackendGetAttribute", plan, A_EXECUTION_PLAN_JSON_REPRESENTATION, T_CHAR, 0, ctypes.byref(count), None):
            return ""
        buffer = ctypes.create_string_buffer(max(int(count.value), 1) + 1)
        if self.lib.attempt("cudnnBackendGetAttribute", plan, A_EXECUTION_PLAN_JSON_REPRESENTATION, T_CHAR, len(buffer), ctypes.byref(count), buffer):
            return ""
        return buffer.value.decode("utf-8", "replace")

    def close(self) -> None:
        owned, self._owned = self._owned, []
        for desc in reversed(owned):
            with contextlib.suppress(TilefishRuntimeError):
                self.lib.call("cudnnBackendDestroyDescriptor", desc)


class _Graph:
    """The operation graph of a problem (its tensors, convolution and operation), on a handle. Destroyed by :meth:`close`."""

    def __init__(self, lib: _Library, handle: ctypes.c_void_p, problem: Problem) -> None:
        self.b = b = _Backend(lib)
        self.problem = problem
        self.handle = handle
        try:
            tensors = {}
            for name, dims, strides in (
                ("x", problem.x_dims, problem.x_strides),
                ("w", problem.w_dims, problem.w_strides),
                ("y", problem.y_dims, problem.y_strides),
            ):
                t = b.make(D_TENSOR)
                b.set(t, A_TENSOR_DATA_TYPE, T_DATA_TYPE, ctypes.c_int(DATA_TYPES[problem.dtype]))
                b.set_int64s(t, A_TENSOR_DIMENSIONS, dims)
                b.set_int64s(t, A_TENSOR_STRIDES, strides)
                b.set(t, A_TENSOR_UNIQUE_ID, T_INT64, ctypes.c_int64(UIDS[name]))
                b.set(t, A_TENSOR_BYTE_ALIGNMENT, T_INT64, ctypes.c_int64(problem.alignment))
                b.finalize(t)
                tensors[name] = t
            conv = b.make(D_CONVOLUTION)
            b.set(conv, A_CONVOLUTION_COMP_TYPE, T_DATA_TYPE, ctypes.c_int(CUDNN_DATA_FLOAT))
            b.set(conv, A_CONVOLUTION_CONV_MODE, T_CONVOLUTION_MODE, ctypes.c_int(CUDNN_CROSS_CORRELATION))
            b.set(conv, A_CONVOLUTION_SPATIAL_DIMS, T_INT64, ctypes.c_int64(len(problem.conv_padding)))
            b.set_int64s(conv, A_CONVOLUTION_DILATIONS, problem.conv_dilation)
            b.set_int64s(conv, A_CONVOLUTION_FILTER_STRIDES, problem.conv_stride)
            b.set_int64s(conv, A_CONVOLUTION_PRE_PADDINGS, problem.conv_padding)
            b.set_int64s(conv, A_CONVOLUTION_POST_PADDINGS, problem.conv_padding)
            b.finalize(conv)
            op = b.make(D_OPERATION_CONVOLUTION_FORWARD)
            b.set_descs(op, A_OPERATION_CONVOLUTION_FORWARD_X, [tensors["x"]])
            b.set_descs(op, A_OPERATION_CONVOLUTION_FORWARD_W, [tensors["w"]])
            b.set_descs(op, A_OPERATION_CONVOLUTION_FORWARD_Y, [tensors["y"]])
            b.set_descs(op, A_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, [conv])
            b.set(op, A_OPERATION_CONVOLUTION_FORWARD_ALPHA, T_FLOAT, ctypes.c_float(1.0))
            b.set(op, A_OPERATION_CONVOLUTION_FORWARD_BETA, T_FLOAT, ctypes.c_float(0.0))
            b.finalize(op)
            self.graph = graph = b.make(D_OPERATIONGRAPH)
            b.set(graph, A_OPERATIONGRAPH_HANDLE, T_HANDLE, ctypes.c_void_p(handle.value))
            b.set_descs(graph, A_OPERATIONGRAPH_OPS, [op])
            b.finalize(graph)
        except BaseException:
            b.close()
            raise

    def close(self) -> None:
        self.b.close()

    def heuristic(self, mode: int, sm_count_target: "int | None") -> "list[ctypes.c_void_p]":
        """The engine configurations the heuristic answers in ``mode``, in its order."""
        b = self.b
        heur = b.make(D_ENGINEHEUR)
        b.set_descs(heur, A_ENGINEHEUR_OPERATION_GRAPH, [self.graph])
        b.set(heur, A_ENGINEHEUR_MODE, T_HEUR_MODE, ctypes.c_int(mode))
        if sm_count_target is not None:
            b.set(heur, A_ENGINEHEUR_SM_COUNT_TARGET, T_INT32, ctypes.c_int32(sm_count_target))
        b.finalize(heur)
        count = b.count_of(heur, A_ENGINEHEUR_RESULTS, T_BACKEND_DESCRIPTOR)
        return b.get_descs(heur, A_ENGINEHEUR_RESULTS, D_ENGINECFG, count) if count else []

    def config_engine(self, config: ctypes.c_void_p) -> ctypes.c_void_p:
        return self.b.get_descs(config, A_ENGINECFG_ENGINE, D_ENGINE, 1)[0]

    def engine_facts(self, engine: ctypes.c_void_p) -> "dict[str, Any]":
        """The global index, numerical notes, behavior notes and knobs (type -> (minimum, maximum, stride)) of an engine."""
        b = self.b
        index = b.get_int64s(engine, A_ENGINE_GLOBAL_INDEX, 1)[0]
        notes = b.get_enums(engine, A_ENGINE_NUMERICAL_NOTE, T_NUMERICAL_NOTE)
        behavior = b.get_enums(engine, A_ENGINE_BEHAVIOR_NOTE, T_BEHAVIOR_NOTE)
        knobs: "dict[int, tuple[int, int, int]]" = {}
        # (a count query, a request of 0, answers 0 for the descriptor lists of an engine and of a configuration: measured, cuDNN 9.10.2; so as many as
        # there are knob types are asked for)
        for info in b.get_descs(engine, A_ENGINE_KNOB_INFO, D_KNOB_INFO, KNOB_TYPE_COUNT):
            t = b.get_enums(info, A_KNOB_INFO_TYPE, T_KNOB_TYPE, 1)[0]
            lo = b.get_int64s(info, A_KNOB_INFO_MINIMUM_VALUE, 1)[0]
            hi = b.get_int64s(info, A_KNOB_INFO_MAXIMUM_VALUE, 1)[0]
            step = b.get_int64s(info, A_KNOB_INFO_STRIDE, 1)[0]
            knobs[t] = (lo, hi, step)
        return {"engine": index, "notes": notes, "behavior": behavior, "knobs": knobs}

    def config_knobs(self, config: ctypes.c_void_p) -> "tuple[tuple[int, int], ...]":
        b = self.b
        out = []
        for choice in b.get_descs(config, A_ENGINECFG_KNOB_CHOICES, D_KNOB_CHOICE, KNOB_TYPE_COUNT):
            t = b.get_enums(choice, A_KNOB_CHOICE_KNOB_TYPE, T_KNOB_TYPE, 1)[0]
            v = b.get_int64s(choice, A_KNOB_CHOICE_KNOB_VALUE, 1)[0]
            out.append((t, v))
        return tuple(sorted(out))

    def plan_of(self, config: ctypes.c_void_p) -> "ctypes.c_void_p | None":
        """The execution plan of an engine configuration, or ``None`` when it does not finalize (an answer: the next one is tried)."""
        b = self.b
        plan = b.make(D_EXECUTION_PLAN)
        b.set(plan, A_EXECUTION_PLAN_HANDLE, T_HANDLE, ctypes.c_void_p(self.handle.value))
        b.set_descs(plan, A_EXECUTION_PLAN_ENGINE_CONFIG, [config])
        if b.lib.attempt("cudnnBackendFinalize", plan) != CUDNN_STATUS_SUCCESS:
            return None
        return plan

    def workspace(self, plan: ctypes.c_void_p) -> int:
        return self.b.get_int64s(plan, A_EXECUTION_PLAN_WORKSPACE_SIZE, 1)[0]

    def config_of(self, pin: EnginePin, sm_count_target: "int | None") -> "tuple[ctypes.c_void_p, ctypes.c_void_p]":
        """The engine and the engine configuration of a pin (finalized). Raises when the engine index is not one of the graph's, or the knob choices
        do not make a configuration of it."""
        engine = self.engine_of(pin.engine, sm_count_target)
        return engine, self.config_with(engine, pin)

    def engine_of(self, index: int, sm_count_target: "int | None") -> ctypes.c_void_p:
        """The engine of a global index (with an SM count target), finalized. Raises when it is not an engine of the graph."""
        b = self.b
        engine = b.make(D_ENGINE)
        b.set_descs(engine, A_ENGINE_OPERATION_GRAPH, [self.graph])
        b.set(engine, A_ENGINE_GLOBAL_INDEX, T_INT64, ctypes.c_int64(index))
        if sm_count_target is not None:
            b.set(engine, A_ENGINE_SM_COUNT_TARGET, T_INT32, ctypes.c_int32(sm_count_target))
        status = b.lib.attempt("cudnnBackendFinalize", engine)
        if status != CUDNN_STATUS_SUCCESS:
            raise TilefishRuntimeError(f"engine {index} is not an engine of the problem ({b.lib.status_name(status)})")
        return engine

    def config_with(self, engine: ctypes.c_void_p, pin: EnginePin) -> ctypes.c_void_p:
        """The engine configuration of a finalized engine and the knob choices of a pin, finalized."""
        b = self.b
        choices = []
        for t, v in pin.knobs:
            choice = b.make(D_KNOB_CHOICE)
            b.set(choice, A_KNOB_CHOICE_KNOB_TYPE, T_KNOB_TYPE, ctypes.c_int(t))
            b.set(choice, A_KNOB_CHOICE_KNOB_VALUE, T_INT64, ctypes.c_int64(v))
            b.finalize(choice)
            choices.append(choice)
        config = b.make(D_ENGINECFG)
        b.set_descs(config, A_ENGINECFG_ENGINE, [engine])
        if choices:
            b.set_descs(config, A_ENGINECFG_KNOB_CHOICES, choices)
        status = b.lib.attempt("cudnnBackendFinalize", config)
        if status != CUDNN_STATUS_SUCCESS:
            raise TilefishRuntimeError(f"the knob choices {[list(kv) for kv in pin.knobs]} do not make a configuration of engine {pin.engine} ({b.lib.status_name(status)})")
        return config


# -- pinning, checking and running -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pinned:
    """A pin and what it needs: the workspace of the problem (at most the limit it was chosen under), and the limit."""

    pin: "LegacyPin | EnginePin"
    workspace_required: int
    workspace_limit: int
    json: str = ""  # the execution plan's JSON representation, for a backend pin (provenance, for a person)


def _check_args(problem: object, workspace_bytes: object, sm_count_target: object, api: "str | None" = None) -> "tuple[Problem, int, int | None]":
    if not isinstance(problem, Problem):
        raise TilefishRuntimeError(f"cuDNN convolution: problem must be a tilefish._cudnn.Problem, got {problem!r}")
    limit = _int("workspace_bytes", workspace_bytes, 0, UINT64_MAX)
    target = None if sm_count_target is None else _int("sm_count_target", sm_count_target, MIN_SM_COUNT_TARGET)
    if target is not None and api == "legacy":
        raise TilefishRuntimeError("cuDNN convolution: the legacy API takes no SM count target (only an engine of the backend API has one)")
    return problem, limit, target


def pin_conv(
    problem: Problem,
    *,
    heuristic: str,
    workspace_bytes: int,
    device_index: "int | None" = None,
    sm_count_target: "int | None" = None,
) -> Pinned:
    """Ask cuDNN for the pin of ``problem`` under the posture ``heuristic`` (:data:`HEURISTICS`) and a workspace limit, and return it checked."""
    if heuristic not in HEURISTICS:
        raise TilefishRuntimeError(f"cuDNN convolution: heuristic must be one of {list(HEURISTICS)}, got {heuristic!r}")
    problem, limit, target = _check_args(problem, workspace_bytes, sm_count_target, HEURISTICS[heuristic])
    refuse_tf32_override()
    lib = _lib()
    index = _device(device_index)
    h = _handle(lib, index)
    with _on_device(index):
        if heuristic == "v7":
            legacy = _Legacy(lib, problem)
            try:
                tried = []
                for cand in legacy.candidates(h.handle):
                    if cand["status"] != CUDNN_STATUS_SUCCESS or cand["determinism"] != CUDNN_DETERMINISTIC:
                        continue
                    need = legacy.workspace(h.handle, cand["algo"])
                    tried.append((cand["algo"], need))
                    if need is not None and need <= limit:
                        return Pinned(LegacyPin(cand["algo"]), need, limit)
            finally:
                legacy.close()
            raise TilefishRuntimeError(
                f"cudnnGetConvolutionForwardAlgorithm_v7 has no deterministic algorithm for {problem.what} within a workspace of {limit} bytes "
                f"(deterministic answers and their workspaces: {tried})"
            )
        graph = _Graph(lib, h.handle, problem)
        try:
            seen = 0
            for mode in HEUR_MODES[heuristic]:
                for config in graph.heuristic(mode, target):
                    seen += 1
                    facts = graph.engine_facts(graph.config_engine(config))
                    if notes_problem(facts["notes"], problem.math) is not None:
                        continue
                    plan = graph.plan_of(config)
                    if plan is None:
                        continue
                    need = graph.workspace(plan)
                    if need > limit:
                        continue
                    pin = EnginePin(facts["engine"], graph.config_knobs(config))
                    json_text = graph.b.json_of(plan)
                    checked = _check_backend(graph, pin, problem, limit, target)
                    if checked[1] and json_text and checked[1] != json_text:
                        raise TilefishRuntimeError(
                            f"the engine configuration the heuristic answered for {problem.what} ({pin.what}) is not the one its engine and knobs make "
                            "again (their execution plans differ): it cannot be pinned"
                        )
                    return Pinned(pin, checked[0], limit, json_text)
        finally:
            graph.close()
        raise TilefishRuntimeError(
            f"cuDNN's heuristic ({heuristic}) has no engine configuration for {problem.what} that is deterministic, allowed by the math, finalizes and "
            f"needs at most {limit} bytes of workspace ({seen} answers)"
        )


def _check_backend(graph: _Graph, pin: EnginePin, problem: Problem, limit: int, target: "int | None") -> "tuple[int, str]":
    """Check an engine pin on a graph: the workspace of its plan and the plan's JSON. Raises :class:`TilefishRuntimeError` naming the problem."""
    head = f"the pinned cuDNN {pin.what} is not valid for {problem.what}"
    try:
        engine = graph.engine_of(pin.engine, target)
    except TilefishRuntimeError as e:
        raise TilefishRuntimeError(f"{head}: {e}") from None
    facts = graph.engine_facts(engine)
    problems = [p for p in (notes_problem(facts["notes"], problem.math), knobs_problem(pin.knobs, facts["knobs"])) if p]
    if problems:
        raise TilefishRuntimeError(f"{head}: {'; '.join(problems)}")
    try:
        config = graph.config_with(engine, pin)
    except TilefishRuntimeError as e:
        raise TilefishRuntimeError(f"{head}: {e}") from None
    plan = graph.plan_of(config)
    if plan is None:
        raise TilefishRuntimeError(f"{head}: its execution plan does not finalize")
    need = graph.workspace(plan)
    if need > limit:
        raise TilefishRuntimeError(f"{head}: it needs {need} bytes of workspace, more than the {limit} recorded")
    return need, graph.b.json_of(plan)


def _check_legacy(legacy: _Legacy, handle: ctypes.c_void_p, pin: LegacyPin, problem: Problem, limit: int) -> int:
    head = f"the pinned cuDNN {pin.what} is not valid for {problem.what}"
    found = [c for c in legacy.candidates(handle) if c["algo"] == pin.algo]
    if not found or found[0]["status"] != CUDNN_STATUS_SUCCESS:
        raise TilefishRuntimeError(f"{head}: cudnnGetConvolutionForwardAlgorithm_v7 does not answer it with CUDNN_STATUS_SUCCESS")
    if found[0]["determinism"] != CUDNN_DETERMINISTIC:
        raise TilefishRuntimeError(f"{head}: cudnnGetConvolutionForwardAlgorithm_v7 says it is not deterministic")
    need = legacy.workspace(handle, pin.algo)
    if need is None:
        raise TilefishRuntimeError(f"{head}: cudnnGetConvolutionForwardWorkspaceSize refuses it")
    if need > limit:
        raise TilefishRuntimeError(f"{head}: it needs {need} bytes of workspace, more than the {limit} recorded")
    return need


class _Prepared:
    """What a pin runs with on one device, kept for the process: the legacy descriptors and the algorithm, or the graph and the execution plan."""

    def __init__(self, lib: _Library, handle: _Handle, problem: Problem, pin: "LegacyPin | EnginePin", limit: int, target: "int | None") -> None:
        self.lib, self.h, self.problem, self.pin, self.limit = lib, handle, problem, pin, limit
        self.legacy: "_Legacy | None" = None
        self.graph: "_Graph | None" = None
        self.plan: "ctypes.c_void_p | None" = None
        if isinstance(pin, LegacyPin):
            self.legacy = _Legacy(lib, problem)
            try:
                self.required = _check_legacy(self.legacy, handle.handle, pin, problem, limit)
            except BaseException:
                self.legacy.close()
                raise
        else:
            self.graph = _Graph(lib, handle.handle, problem)
            try:
                self.required, _ = _check_backend(self.graph, pin, problem, limit, target)
                _, config = self.graph.config_of(pin, target)
                self.plan = self.graph.plan_of(config)
            except BaseException:
                self.graph.close()
                raise

    def run(self, x: int, w: int, y: int, workspace: "int | None", stream: int) -> None:
        lib = self.lib
        with self.h.lock:
            lib.call("cudnnSetStream", self.h.handle, ctypes.c_void_p(stream))
            if self.legacy is not None:
                alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
                size = self.limit if workspace else 0
                lib.call(
                    "cudnnConvolutionForward", self.h.handle, ctypes.byref(alpha), self.legacy.x, ctypes.c_void_p(x), self.legacy.w, ctypes.c_void_p(w),
                    self.legacy.conv, self.pin.algo, ctypes.c_void_p(workspace or 0), size, ctypes.byref(beta), self.legacy.y, ctypes.c_void_p(y),  # type: ignore[union-attr]
                )  # fmt: skip
                return
            assert self.graph is not None and self.plan is not None
            b = _Backend(lib)
            try:
                pack = b.make(D_VARIANT_PACK)
                uids = (ctypes.c_int64 * 3)(UIDS["x"], UIDS["y"], UIDS["w"])
                ptrs = (ctypes.c_void_p * 3)(x, y, w)
                lib.call("cudnnBackendSetAttribute", pack, A_VARIANT_PACK_UNIQUE_IDS, T_INT64, 3, uids)
                lib.call("cudnnBackendSetAttribute", pack, A_VARIANT_PACK_DATA_POINTERS, T_VOID_PTR, 3, ptrs)
                ws = ctypes.c_void_p(workspace or 0)
                lib.call("cudnnBackendSetAttribute", pack, A_VARIANT_PACK_WORKSPACE, T_VOID_PTR, 1, ctypes.byref(ws))
                b.finalize(pack)
                lib.call("cudnnBackendExecute", self.h.handle, self.plan, pack)
            finally:
                b.close()


_prepared_lock = threading.Lock()
_prepared: "dict[tuple, _Prepared]" = {}


def _prepare(problem: Problem, pin: "LegacyPin | EnginePin", limit: int, target: "int | None", index: int) -> _Prepared:
    key = (index, problem, pin, limit, target)
    found = _prepared.get(key)
    if found is None:
        lib = _lib()
        h = _handle(lib, index)
        with _prepared_lock:
            found = _prepared.get(key)
            if found is None:
                with _on_device(index):
                    found = _prepared[key] = _Prepared(lib, h, problem, pin, limit, target)
    return found


def check_pin(
    problem: Problem,
    pin: "LegacyPin | EnginePin",
    *,
    workspace_bytes: int,
    device_index: "int | None" = None,
    sm_count_target: "int | None" = None,
) -> int:
    """The workspace the pin needs for ``problem`` on this device, after the checks of the module docstring (refused: a
    :class:`TilefishRuntimeError`)."""
    pin = pin_of(pin)
    problem, limit, target = _check_args(problem, workspace_bytes, sm_count_target, pin.api)
    refuse_tf32_override()
    index = _device(device_index)
    return _prepare(problem, pin, limit, target, index).required


def _pointer(name: str, value: object, alignment: int) -> int:
    value = _int(f"{name} pointer", value, 1, UINT64_MAX)
    if value % alignment:
        raise TilefishRuntimeError(f"cuDNN convolution: the {name} pointer {value:#x} is not {alignment}-byte aligned")
    return value


def convolution(
    x_ptr: int,
    w_ptr: int,
    y_ptr: int,
    problem: Problem,
    pin: "LegacyPin | EnginePin",
    *,
    workspace_bytes: int,
    stream: int,
    device_index: "int | None" = None,
    workspace_ptr: "int | None" = None,
    sm_count_target: "int | None" = None,
) -> None:
    """``y = conv(x, w)`` with the pin, enqueued on ``stream`` (a ``CUstream`` as an int). The pin is checked for the problem first (once per
    process and device: :func:`check_pin`). With ``workspace_ptr`` (``workspace_bytes`` bytes, 256-byte aligned: a launch plan's arena buffer) the
    workspace is that memory and nothing is allocated; without it a workspace of ``workspace_bytes`` comes from torch's caching allocator on ``stream``."""
    pin = pin_of(pin)
    problem, limit, target = _check_args(problem, workspace_bytes, sm_count_target, pin.api)
    x = _pointer("x", x_ptr, problem.alignment)
    w = _pointer("w", w_ptr, problem.alignment)
    y = _pointer("y", y_ptr, problem.alignment)
    stream = _int("stream", stream, 0, UINT64_MAX)
    given = None
    if workspace_ptr is not None:
        given = _int("workspace pointer", workspace_ptr, 1, UINT64_MAX)
        if given % WORKSPACE_ALIGNMENT:
            raise TilefishRuntimeError(f"cuDNN convolution: the workspace pointer {given:#x} is not {WORKSPACE_ALIGNMENT}-byte aligned")
    refuse_tf32_override()
    index = _device(device_index)
    prepared = _prepare(problem, pin, limit, target, index)
    workspace = None
    address = given
    # a legacy call is given the pin's whole limit (the same call at every launch); a plan needs its own workspace
    size = limit if isinstance(pin, LegacyPin) else prepared.required
    if size and given is None:
        torch = _torch()
        with torch.cuda.stream(torch.cuda.ExternalStream(stream, device=index)):
            workspace = torch.empty(size, dtype=torch.uint8, device=f"cuda:{index}")
        address = workspace.data_ptr()
    with _on_device(index):
        prepared.run(x, w, y, address if size else None, stream)
    del workspace


def engine_list(problem: Problem, *, heuristic: str = "instant", device_index: "int | None" = None, sm_count_target: "int | None" = None) -> "list[dict[str, Any]]":
    """What the backend heuristic answers for ``problem``, every configuration of every mode of the posture, with its engine, notes, knobs,
    whether its plan finalizes and its workspace (for a person choosing or importing a pin, and the tests)."""
    if HEURISTICS.get(heuristic) != "backend":
        raise TilefishRuntimeError(f"cuDNN convolution: engine_list takes a backend posture ('instant', 'mode_b'), got {heuristic!r}")
    lib = _lib()
    index = _device(device_index)
    h = _handle(lib, index)
    out = []
    with _on_device(index):
        graph = _Graph(lib, h.handle, problem)
        try:
            for mode in HEUR_MODES[heuristic]:
                for rank, config in enumerate(graph.heuristic(mode, sm_count_target)):
                    facts = graph.engine_facts(graph.config_engine(config))
                    plan = graph.plan_of(config)
                    out.append(
                        {
                            "mode": mode,
                            "rank": rank,
                            "engine": facts["engine"],
                            "notes": [NOTE_NAMES.get(n, str(n)) for n in facts["notes"]],
                            "behavior": [BEHAVIOR_NAMES.get(n, str(n)) for n in facts["behavior"]],
                            "knobs": [list(kv) for kv in graph.config_knobs(config)],
                            "finalizes": plan is not None,
                            "workspace": None if plan is None else graph.workspace(plan),
                            "allowed": notes_problem(facts["notes"], problem.math) is None,
                        }
                    )
        finally:
            graph.close()
    return out


def legacy_list(problem: Problem, *, device_index: "int | None" = None) -> "list[dict[str, int]]":
    """``cudnnGetConvolutionForwardAlgorithm_v7``'s answers for ``problem``, with each one's ``cudnnGetConvolutionForwardWorkspaceSize``."""
    lib = _lib()
    index = _device(device_index)
    h = _handle(lib, index)
    with _on_device(index):
        legacy = _Legacy(lib, problem)
        try:
            return [{**c, "workspace": legacy.workspace(h.handle, c["algo"])} for c in legacy.candidates(h.handle)]
        finally:
            legacy.close()


__all__ = [
    "EnginePin",
    "LegacyPin",
    "Pinned",
    "Problem",
    "check_pin",
    "convolution",
    "describe",
    "engine_list",
    "legacy_list",
    "pin_conv",
    "version",
]
