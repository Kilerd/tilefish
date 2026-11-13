"""A ctypes binding of cuBLASLt for one operation, run with a pinned algorithm.

The operation, in tilefish terms (row-major, contiguous, every pointer 16-byte aligned)::

    D[M, N] = A[M, K] @ B[N, K]^T          (+ bias[N])

``A``, ``B``, ``bias`` and ``D`` are bf16, the accumulator is f32, and the result is rounded to bf16 once, at the end
(the bias is added in f32 to the accumulator before that rounding). ``alpha = 1``, ``beta = 0``, host pointer mode.

cuBLASLt is column-major, so the problem is posed as the usual ``TN`` gemm, in cuBLAS terms ``m = N``, ``n = M``, ``k = K``:

* the "A operand" is tilefish's ``B`` (``[N, K]`` row-major is ``K x N`` column-major, ``lda = K``) with ``transa = T``,
* the "B operand" is tilefish's ``A`` (``[M, K]`` row-major is ``K x M`` column-major, ``ldb = K``) with ``transb = N``,
* ``D`` is tilefish's ``D`` (``[M, N]`` row-major is ``N x M`` column-major, ``ldd = N``); the ``C`` layout is the ``D`` layout.

The library is the ``libcublasLt`` the process already has mapped (torch's, from the ``nvidia-cublas`` wheel): it is
found in ``/proc/self/maps`` and opened by that path, so the dynamic loader hands back the mapping that is there and no
second copy is loaded (this is asserted after the load). When none is mapped yet, torch is made to load its own with a tiny
bf16 matmul; only when that maps none either (no torch, or a torch that links cuBLASLt statically) is the copy inside the
``nvidia.cublas`` wheel opened. A machine with none of these raises :class:`~tilefish.errors.TilefishRuntimeError`.

Only the calls of this one operation are wrapped: handle, matmul descriptor, matrix layouts, matmul preference,
``cublasLtMatmulAlgoGetHeuristic``, ``cublasLtMatmulAlgoCheck``, ``cublasLtMatmul``, ``cublasLtGetVersion``, and, to describe
and to vet a pinned algorithm, ``cublasLtMatmulAlgoInit``, ``cublasLtMatmulAlgoConfig{Get,Set}Attribute`` and
``cublasLtMatmulAlgoCapGetAttribute``. Every call checks its ``cublasStatus_t`` and raises
:class:`~tilefish.errors.TilefishRuntimeError` naming the call. Arguments are validated before the library is touched.

**Pinning.** :func:`pin_algo` asks the heuristic under a workspace limit and takes its first answer. How it asks is a
*heuristic posture* (:data:`HEURISTICS`):

* ``"tilefish"`` (the default): a few results are asked for, with the preferences ``MIN_ALIGNMENT_{A,B,C,D}_BYTES = 16``, so
  that what the heuristic returns does not depend on the values of the pointers (the bias pointer, which the heuristic reads
  for its alignment, is a fixed 16-byte aligned dummy, :data:`HEURISTIC_BIAS_POINTER`);
* ``"torch"``: what torch 2.9 asks for ``torch.addmm(bias, a, b.t())`` (``at::cuda::blas::gemm_and_bias``): one result, under the
  workspace limit alone (the alignment minimums it sets are those of its own pointers, 256 bytes from its allocator, which is
  the library's default, so none is set here, and the bias pointer is a 256-byte aligned dummy). With the limit torch uses
  (:data:`TORCH_LT_WORKSPACE_BYTES` for ``addmm``, :data:`DEFAULT_WORKSPACE_BYTES` for ``mm``, which runs ``cublasGemmEx`` and
  is matched by this posture with its workspace) the heuristic answers what torch runs.

In both, the reduction-scheme mask lets split-K add its partial sums only in f32 (a bf16 partial sum rounds more than once),
which is what torch does with ``allow_bf16_reduced_precision_reduction = False``, and a GEMM with a bias gets no split-K at all (with
the bias epilogue a split-K was measured to round twice even with f32 partial sums, see :data:`NO_SPLIT_K_MASK`); with
``allow_bf16_reduction=True`` (torch's default) no mask is set, and an algorithm whose split-K reduces in bf16 (``INPLACE``,
``OUTPUT_TYPE``) is accepted.

An algorithm can also be imported: :func:`encode_algo` turns its nine configuration attributes into the 64 bytes cuBLASLt
writes for them. The pin is the 64-byte ``cublasLtMatmulAlgo_t`` as a 128-character hex string, and the workspace it needs.
:func:`check_algo` and :func:`matmul` rebuild the algorithm from that string and have ``cublasLtMatmulAlgoCheck`` approve it for
the problem before it runs; an algorithm that is not valid for the problem, or that needs more workspace than recorded, is
refused. ``cublasLtMatmulAlgoCheck`` is a weak judge of bytes (it accepts several corruptions and crashes the process on some,
see :func:`_validate_algo`), so the string is first held to what cuBLASLt itself would have encoded. The pin belongs to one
cuBLASLt build (:func:`version`) and one device (compute capability, and the SM count of a MIG slice).

Descriptors are built per call and destroyed at its end, so nothing mutable is shared between threads; the one thing that is
shared is the cuBLASLt handle (one per device, created once under a lock, never destroyed before the process ends), which holds
no per-call state (the workspace is an argument of the call).
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

DEFAULT_WORKSPACE_BYTES = 8_519_680  # torch's default cuBLAS workspace config ":4096:2:16:8": 4096 KiB * 2 + 16 KiB * 8
TORCH_LT_WORKSPACE_BYTES = 1_048_576  # torch's default cuBLASLt workspace (CUBLASLT_WORKSPACE_SIZE unset: 1024 KiB), what addmm offers its heuristic
ALGO_BYTES = 64  # sizeof(cublasLtMatmulAlgo_t): uint64_t data[8]
INT32_MAX = (1 << 31) - 1
UINT64_MAX = (1 << 64) - 1
ALIGNMENT_BYTES = 16
WORKSPACE_ALIGNMENT = 256  # of a workspace the caller gives (cuBLASLt asks for 16 bytes at least and recommends 256; the Rust host requires 256, a plan's arena buffer is)
HEURISTICS = ("tilefish", "torch")  # the heuristic postures (see the module docstring)
REQUESTED_ALGO_COUNT = 8  # how many results the "tilefish" posture asks the heuristic for; the first is the pin ("torch" asks for one, as torch does)
HEURISTIC_BIAS_POINTER = 16  # a 16-byte (and no more) aligned dummy, so that the heuristic never sees an address of the caller's
TORCH_HEURISTIC_BIAS_POINTER = 256  # the "torch" posture's dummy: aligned as torch's allocations are (to 256 bytes at least)

# -- the values of the header ------------------------------------------------------------------------------------------------
# All of them are read from the cuBLAS 12.8 headers of the nvidia-cublas wheel (nvidia/cublas/include/cublasLt.h and
# cublas_api.h) and library_types.h of nvidia-cuda-runtime; none is a guess.

# cublas_api.h, cublasStatus_t
CUBLAS_STATUS_SUCCESS = 0
CUBLAS_STATUS_NOT_SUPPORTED = 15
_STATUS_NAMES = {
    0: "CUBLAS_STATUS_SUCCESS",
    1: "CUBLAS_STATUS_NOT_INITIALIZED",
    3: "CUBLAS_STATUS_ALLOC_FAILED",
    7: "CUBLAS_STATUS_INVALID_VALUE",
    8: "CUBLAS_STATUS_ARCH_MISMATCH",
    11: "CUBLAS_STATUS_MAPPING_ERROR",
    13: "CUBLAS_STATUS_EXECUTION_FAILED",
    14: "CUBLAS_STATUS_INTERNAL_ERROR",
    15: "CUBLAS_STATUS_NOT_SUPPORTED",
    16: "CUBLAS_STATUS_LICENSE_ERROR",
}
# cublas_api.h, cublasOperation_t
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
# cublas_api.h, cublasComputeType_t
CUBLAS_COMPUTE_32F = 68
# library_types.h, cudaDataType_t
CUDA_R_32F = 0
CUDA_R_16BF = 14
# cublasLt.h, cublasLtMatmulDescAttributes_t (types: TRANSA/TRANSB/BIAS_DATA_TYPE int32_t, EPILOGUE uint32_t, BIAS_POINTER const void*)
CUBLASLT_MATMUL_DESC_TRANSA = 3
CUBLASLT_MATMUL_DESC_TRANSB = 4
CUBLASLT_MATMUL_DESC_EPILOGUE = 7
CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8
CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 26
# cublasLt.h, cublasLtEpilogue_t
CUBLASLT_EPILOGUE_DEFAULT = 1
CUBLASLT_EPILOGUE_BIAS = 4
# cublasLt.h, cublasLtMatmulPreferenceAttributes_t (MAX_WORKSPACE_BYTES uint64_t, the others uint32_t)
CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1
CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = 3
CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES = 5
CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES = 6
CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES = 7
CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES = 8
# cublasLt.h, cublasLtReductionScheme_t
CUBLASLT_REDUCTION_SCHEME_NONE = 0
CUBLASLT_REDUCTION_SCHEME_INPLACE = 1
CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE = 2
CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE = 4
# Split-K may combine its partial sums in the output type (bf16: INPLACE and OUTPUT_TYPE), which would round more than once; the
# pin allows only no split-K (NONE) and f32 partial sums in the workspace (COMPUTE_TYPE), unless the declaration opts in to bf16
# partial sums (allow_bf16_reduction). The mask is the bits of the schemes that may be used (the header: "mask value of 0x03 will
# allow only INPLACE and COMPUTE_TYPE"); NONE has no bit, and algorithms without split-K are returned under the mask 0x2 (measured).
#
# A GEMM with a bias gets no split-K at all without the opt-in (the mask 0: only NONE, measured to return algorithms without split-K):
# with the bias epilogue, a split-K with f32 partial sums was measured to add the bias to a sum already rounded to bf16 (cuBLASLt 12.8.4,
# RTX 5090, sm_120, algorithm 21: 32x1024x3072, 8409 of 32768 elements not the correctly rounded result and errors past what one rounding
# and an f32 accumulation allow; the same configuration without the bias, and without split-K, keep the bound).
REDUCTION_SCHEME_MASK = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE
NO_SPLIT_K_MASK = 0
ALLOWED_REDUCTION_SCHEMES = (CUBLASLT_REDUCTION_SCHEME_NONE, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)
BF16_REDUCTION_SCHEMES = (CUBLASLT_REDUCTION_SCHEME_INPLACE, CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE)  # allowed too with allow_bf16_reduction


def allowed_reduction_schemes(allow_bf16_reduction: bool, bias: bool = False) -> "tuple[int, ...]":
    """The ``cublasLtReductionScheme_t`` values a pinned algorithm may have: no split-K and f32 partial sums (only no split-K with a bias), and with
    the opt-in also the two that add the partial sums in bf16 (in the output buffer, or in a workspace of the output type)."""
    if allow_bf16_reduction:
        return ALLOWED_REDUCTION_SCHEMES + BF16_REDUCTION_SCHEMES
    return (CUBLASLT_REDUCTION_SCHEME_NONE,) if bias else ALLOWED_REDUCTION_SCHEMES


def reduction_problem(splitk_num: int, reduction_scheme: int, allow_bf16_reduction: bool, bias: bool = False) -> "str | None":
    """What is wrong with this split-K count and reduction scheme for a pin of a GEMM with or without a bias (``None``: nothing). Pure: the rule
    both hosts hold an algorithm to (the Rust crate's ``reduction_problem`` has the same words)."""
    if splitk_num < 1:
        return f"split-K count {splitk_num} is below 1"
    if (splitk_num > 1) != (reduction_scheme != CUBLASLT_REDUCTION_SCHEME_NONE):
        return f"split-K count {splitk_num} does not go with reduction scheme {reduction_scheme} (split-K needs one, no split-K has none)"
    if reduction_scheme in BF16_REDUCTION_SCHEMES and not allow_bf16_reduction:
        return (
            f"reduction scheme {reduction_scheme} adds the split-K partial sums in bf16, which rounds more than once (only no split-K, 0, and f32 "
            "partial sums, 2, keep the one rounding); a declaration may opt in with allow_bf16_reduction=True"
        )
    if reduction_scheme not in ALLOWED_REDUCTION_SCHEMES + BF16_REDUCTION_SCHEMES:
        return f"reduction scheme {reduction_scheme} is not one cuBLASLt defines (0, 1, 2 or 4)"
    if bias and splitk_num > 1 and not allow_bf16_reduction:
        return (
            f"split-K count {splitk_num} with the bias epilogue may add the bias to a sum already rounded to bf16 (measured with f32 partial "
            "sums too), which rounds twice; a GEMM with a bias has no split-K unless its declaration opts in with allow_bf16_reduction=True"
        )
    return None
# cublasLt.h, cublasLtMatmulAlgoConfigAttributes_t
_ALGO_CONFIG = {  # name -> (attribute, ctypes type)
    "id": (0, ctypes.c_int32),  # CUBLASLT_ALGO_CONFIG_ID
    "tile_id": (1, ctypes.c_uint32),  # CUBLASLT_ALGO_CONFIG_TILE_ID
    "splitk_num": (2, ctypes.c_int32),  # CUBLASLT_ALGO_CONFIG_SPLITK_NUM
    "reduction_scheme": (3, ctypes.c_uint32),  # CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME
    "cta_swizzling": (4, ctypes.c_uint32),  # CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING
    "custom_option": (5, ctypes.c_uint32),  # CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION
    "stages_id": (6, ctypes.c_uint32),  # CUBLASLT_ALGO_CONFIG_STAGES_ID
    "inner_shape_id": (7, ctypes.c_uint16),  # CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID
    "cluster_shape_id": (8, ctypes.c_uint16),  # CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID
}
# cublasLt.h, cublasLtMatmulTile_t and cublasLtMatmulStages_t
CUBLASLT_MATMUL_TILE_UNDEFINED = 0
CUBLASLT_MATMUL_STAGES_UNDEFINED = 0
# cublasLt.h, cublasLtMatmulAlgoCapAttributes_t
_CAP_CTA_SWIZZLING_SUPPORT = 2  # uint32_t: 0 no support, 1 supported value of 1, other values reserved
_CAP_TILE_IDS = 6  # array of uint32_t
_CAP_CUSTOM_OPTION_MAX = 7  # int32_t
_CAP_STAGES_IDS = 13  # array of uint32_t
_CAP_NUMERICAL_IMPL_FLAGS = 15  # uint64_t
_CAP_MIN_ALIGNMENT_BYTES = (16, 17, 18, 19)  # uint32_t, of A, B, C, D
# cublasLt.h, cublasLtNumericalImplFlags_t
_IMPL_ACCUMULATOR_TYPE_MASK = 0xFF << 8  # CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK
_IMPL_ACCUMULATOR_32F = 0x02 << 8  # CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F


# -- the structs of the header -----------------------------------------------------------------------------------------------


class _Algo(ctypes.Structure):
    """``cublasLtMatmulAlgo_t``: ``uint64_t data[8]``, trivially serializable (the header says so)."""

    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _HeuristicResult(ctypes.Structure):
    """``cublasLtMatmulHeuristicResult_t``: algo, workspaceSize, state, wavesCount, reserved[4]."""

    _fields_ = [
        ("algo", _Algo),
        ("workspaceSize", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("wavesCount", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


assert ctypes.sizeof(_Algo) == ALGO_BYTES
assert ctypes.sizeof(_HeuristicResult) == 96


def _functions() -> "dict[str, tuple[Any, list[Any]]]":
    c = ctypes
    vp, i32 = c.c_void_p, c.c_int
    pvp = c.POINTER(c.c_void_p)
    return {
        # name: (restype, argtypes); every handle, descriptor and pointer is a c_void_p
        "cublasLtGetVersion": (c.c_size_t, []),
        "cublasLtGetStatusName": (c.c_char_p, [i32]),
        "cublasLtCreate": (i32, [pvp]),
        "cublasLtDestroy": (i32, [vp]),
        "cublasLtMatmulDescCreate": (i32, [pvp, i32, i32]),
        "cublasLtMatmulDescDestroy": (i32, [vp]),
        "cublasLtMatmulDescSetAttribute": (i32, [vp, i32, vp, c.c_size_t]),
        "cublasLtMatrixLayoutCreate": (i32, [pvp, i32, c.c_uint64, c.c_uint64, c.c_int64]),
        "cublasLtMatrixLayoutDestroy": (i32, [vp]),
        "cublasLtMatmulPreferenceCreate": (i32, [pvp]),
        "cublasLtMatmulPreferenceDestroy": (i32, [vp]),
        "cublasLtMatmulPreferenceSetAttribute": (i32, [vp, i32, vp, c.c_size_t]),
        "cublasLtMatmulAlgoGetHeuristic": (
            i32,
            [vp, vp, vp, vp, vp, vp, vp, i32, c.POINTER(_HeuristicResult), c.POINTER(i32)],
        ),
        "cublasLtMatmulAlgoCheck": (i32, [vp, vp, vp, vp, vp, vp, c.POINTER(_Algo), c.POINTER(_HeuristicResult)]),
        "cublasLtMatmul": (i32, [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, c.POINTER(_Algo), vp, c.c_size_t, vp]),
        "cublasLtMatmulAlgoConfigGetAttribute": (i32, [c.POINTER(_Algo), i32, vp, c.c_size_t, c.POINTER(c.c_size_t)]),
        "cublasLtMatmulAlgoCapGetAttribute": (i32, [c.POINTER(_Algo), i32, vp, c.c_size_t, c.POINTER(c.c_size_t)]),
        "cublasLtMatmulAlgoConfigSetAttribute": (i32, [c.POINTER(_Algo), i32, vp, c.c_size_t]),
        "cublasLtMatmulAlgoInit": (i32, [vp, i32, i32, i32, i32, i32, i32, i32, c.POINTER(_Algo)]),
    }


# -- finding and loading the library -----------------------------------------------------------------------------------------

_MAPS = "/proc/self/maps"
_LIB_NAME = re.compile(r"^libcublasLt\.so(\.\d+)*$")
_NOT_FOUND = (
    "cuBLASLt is not available: no libcublasLt is mapped into this process, torch could not be made to load one, and the "
    "nvidia-cublas wheel (the 'nvidia.cublas' package) is not installed; install torch with CUDA or "
    "'pip install nvidia-cublas-cu12'"
)

_lib_lock = threading.Lock()
_library: "_Library | None" = None


def _mapped_paths() -> "list[str]":
    """The distinct real paths of the libcublasLt files this process has mapped (sorted; empty without ``/proc/self/maps``)."""
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
        path = parts[5]
        if path.endswith(" (deleted)"):
            path = path[: -len(" (deleted)")]
        if _LIB_NAME.match(os.path.basename(path)):
            found.add(os.path.realpath(path))
    return sorted(found)


def _wheel_path() -> "str | None":
    """The libcublasLt of the ``nvidia.cublas`` wheel, when it is installed (found without importing it)."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("nvidia.cublas.lib")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for directory in spec.submodule_search_locations:
        candidates = sorted(p for p in glob.glob(os.path.join(directory, "libcublasLt.so*")) if _LIB_NAME.match(os.path.basename(p)))
        if candidates:
            return os.path.realpath(candidates[-1])
    return None


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:
        raise TilefishRuntimeError(
            "the cuBLASLt operation needs torch with CUDA (for the device and the workspace): pip install 'tilefish[gpu]'"
        ) from e
    return torch


def _trigger_torch_load() -> None:
    """Make torch load its own cuBLASLt, with the smallest bf16 matmul it can run. Best effort: a failure leaves nothing mapped."""
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            return
        a = torch.ones(16, 16, dtype=torch.bfloat16, device="cuda")
        torch.mm(a, a)
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - the caller looks at what is mapped afterwards, and falls back to the wheel
        return


def library_path() -> str:
    """The path of the ``libcublasLt`` this binding uses: the one this process has mapped, else the one of the nvidia-cublas wheel.

    Raises :class:`~tilefish.errors.TilefishRuntimeError` when there is none, or when the process has mapped two different ones
    (which one torch calls is then unknown, so the pin could not be tied to it).
    """
    if _library is not None:
        return _library.path  # already opened: this is the one
    mapped = _mapped_paths()
    if not mapped:
        _trigger_torch_load()
        mapped = _mapped_paths()
    if len(mapped) > 1:
        raise TilefishRuntimeError(
            f"this process has mapped {len(mapped)} different libcublasLt files ({', '.join(mapped)}); cuBLASLt is bound "
            "to the one torch uses, and which that is cannot be told"
        )
    if mapped:
        return mapped[0]
    wheel = _wheel_path()
    if wheel is None:
        raise TilefishRuntimeError(_NOT_FOUND)
    return wheel


class _Library:
    """The loaded ``libcublasLt`` with the prototypes of every function this module calls."""

    def __init__(self, path: str) -> None:
        real = os.path.realpath(path)
        try:
            self.cdll = ctypes.CDLL(path)  # a file that is mapped already is not mapped again
        except OSError as e:
            raise TilefishRuntimeError(f"cuBLASLt ({path}) cannot be loaded: {e}") from e
        if os.path.exists(_MAPS):
            after = _mapped_paths()
            if after != [real]:
                raise TilefishRuntimeError(
                    f"opening {path} must not map a second libcublasLt, but this process now has {after or 'none'} mapped (expected {[real]})"
                )
        self.path = real
        for name, (restype, argtypes) in _functions().items():
            try:
                fn = getattr(self.cdll, name)
            except AttributeError as e:
                raise TilefishRuntimeError(f"cuBLASLt ({path}) has no {name}: it is too old for this binding") from e
            fn.restype = restype
            fn.argtypes = argtypes
            setattr(self, name, fn)

    def status_name(self, status: int) -> str:
        name = _STATUS_NAMES.get(status)
        if name is None:
            try:
                raw = self.cublasLtGetStatusName(status)  # type: ignore[attr-defined]
                name = raw.decode("ascii", "replace") if raw else None
            except Exception:  # noqa: BLE001 - the name is a nicety
                name = None
        return f"{name} ({status})" if name else f"status {status}"

    def call(self, name: str, *args: Any, allow: "tuple[int, ...]" = ()) -> int:
        """Call a function that returns a ``cublasStatus_t``; anything but success (or a status of ``allow``, which is returned) is a
        :class:`TilefishRuntimeError` naming it."""
        status = getattr(self, name)(*args)
        if status != CUBLAS_STATUS_SUCCESS and status not in allow:
            raise TilefishRuntimeError(f"cuBLASLt call {name} failed: {self.status_name(status)}")
        return int(status)


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
    """``cublasLtGetVersion()``: the version of the cuBLASLt this process uses (for 12.8.4 it is ``120804``).

    This is all a pin is compared on: cuBLASLt exposes no build number. ``cublasLtGetVersion`` is ``major * 10000 + minor * 100 + patch``, and
    ``cublasLtGetProperty`` answers the same three (``MAJOR_VERSION``, ``MINOR_VERSION``, ``PATCH_LEVEL``; checked on 12.8.4, whose library has no
    other version or build entry point), so two builds of one version are not told apart. The library that ran the pin is the one torch maps, and a
    different build of the same version is not refused by this check (``docs/manifest-schema-3.md``, section 9)."""
    return int(_lib().cublasLtGetVersion())  # type: ignore[attr-defined]


# -- the problem -------------------------------------------------------------------------------------------------------------


def _int_arg(name: str, value: object, *, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TilefishRuntimeError(f"cuBLASLt matmul: {name} must be an int, got {value!r}")
    if not low <= value <= high:
        raise TilefishRuntimeError(f"cuBLASLt matmul: {name} is {value}, outside {low}..{high}")
    return value


def _check_problem(m: object, n: object, k: object, bias: object) -> "tuple[int, int, int, bool]":
    """Positive ``int32`` sizes (the leading dimensions are ``k`` and ``n``, so they are in range with them) and a real bool."""
    m = _int_arg("m", m, low=1, high=INT32_MAX)
    n = _int_arg("n", n, low=1, high=INT32_MAX)
    k = _int_arg("k", k, low=1, high=INT32_MAX)
    if not isinstance(bias, bool):
        raise TilefishRuntimeError(f"cuBLASLt matmul: bias must be a bool, got {bias!r}")
    return m, n, k, bias


def _check_pointer(name: str, value: object) -> int:
    value = _int_arg(f"{name} pointer", value, low=1, high=UINT64_MAX)
    if value % ALIGNMENT_BYTES:
        raise TilefishRuntimeError(f"cuBLASLt matmul: the {name} pointer {value:#x} is not {ALIGNMENT_BYTES}-byte aligned")
    return value


def _check_workspace_pointer(value: object) -> int:
    """A workspace the caller gives: a non-null address aligned to ``WORKSPACE_ALIGNMENT`` bytes (what the Rust host requires too)."""
    value = _int_arg("workspace pointer", value, low=1, high=UINT64_MAX)
    if value % WORKSPACE_ALIGNMENT:
        raise TilefishRuntimeError(f"cuBLASLt matmul: the workspace pointer {value:#x} is not {WORKSPACE_ALIGNMENT}-byte aligned")
    return value


def _check_workspace(workspace_bytes: object) -> int:
    return _int_arg("workspace_bytes", workspace_bytes, low=0, high=UINT64_MAX)


def _check_heuristic(heuristic: object) -> str:
    if heuristic not in HEURISTICS:
        raise TilefishRuntimeError(f"cuBLASLt matmul: heuristic must be one of {list(HEURISTICS)}, got {heuristic!r}")
    return str(heuristic)


def _check_flag(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TilefishRuntimeError(f"cuBLASLt matmul: {name} must be a bool, got {value!r}")
    return value


def describe(m: int, n: int, k: int, bias: bool, *, allow_bf16_reduction: bool = False) -> "dict[str, Any]":
    """The exact cuBLAS-level problem that tilefish's ``D[m, n] = A[m, k] @ B[n, k]^T (+ bias[n])`` is posed as (pure: no GPU, no library).

    The keys are what is passed to cuBLASLt: the transposes, the compute and scale types, the sizes ``cublas_m = n``,
    ``cublas_n = m``, ``cublas_k = k`` and the leading dimensions of the column-major operands.
    """
    m, n, k, bias = _check_problem(m, n, k, bias)
    allow_bf16_reduction = _check_flag("allow_bf16_reduction", allow_bf16_reduction)
    return {
        "library": "cublasLt",
        "operation": "matmul",
        "dtype": "bf16",
        "a_type": "CUDA_R_16BF",
        "b_type": "CUDA_R_16BF",
        "c_type": "CUDA_R_16BF",
        "d_type": "CUDA_R_16BF",
        "compute_type": "CUBLAS_COMPUTE_32F",
        "scale_type": "CUDA_R_32F",
        "order": "col",
        "transa": "T",
        "transb": "N",
        "cublas_m": n,
        "cublas_n": m,
        "cublas_k": k,
        "lda": k,
        "ldb": k,
        "ldc": n,
        "ldd": n,
        "a_operand": "B[N, K]",
        "b_operand": "A[M, K]",
        "alpha": 1.0,
        "beta": 0.0,
        "pointer_mode": "host",
        "epilogue": "bias" if bias else "none",
        "bias_type": "CUDA_R_16BF" if bias else None,
        "bias_length": n if bias else None,  # one value per row of the column-major D, which is one per column of tilefish's D
        "min_alignment_bytes": ALIGNMENT_BYTES,
        # cublasLtReductionScheme_t values a split-K may use: NONE, COMPUTE_TYPE, and with the opt-in INPLACE and OUTPUT_TYPE (bf16 partial sums)
        "reduction_schemes": list(allowed_reduction_schemes(allow_bf16_reduction, bias)),
    }


def preferences(heuristic: str, workspace_bytes: int, allow_bf16_reduction: bool, bias: bool = False) -> "list[tuple[int, int]]":
    """The matmul preference attributes a posture asks the heuristic with, in the order they are set, as ``(attribute, value)``: the workspace
    limit (``uint64_t``) and then ``uint32_t`` ones (the reduction-scheme mask: none with the opt-in, else f32 partial sums only, and no split-K
    for a GEMM with a bias). Pure."""
    prefs = [(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspace_bytes)]
    if not allow_bf16_reduction:
        prefs.append((CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, NO_SPLIT_K_MASK if bias else REDUCTION_SCHEME_MASK))
    if heuristic == "tilefish":
        prefs += [
            (CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, ALIGNMENT_BYTES),
            (CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, ALIGNMENT_BYTES),
            (CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, ALIGNMENT_BYTES),
            (CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, ALIGNMENT_BYTES),
        ]
    return prefs


class _Problem:
    """The descriptors of one call: matmul descriptor, the layouts of the two operands and of ``D`` (``C`` is the same), and
    optionally a preference (the attributes of :func:`preferences`). Everything is destroyed by :meth:`close`; nothing of it is shared
    with another call."""

    def __init__(
        self,
        lib: _Library,
        m: int,
        n: int,
        k: int,
        bias: bool,
        bias_ptr: int,
        preference: "list[tuple[int, int]] | None",
    ) -> None:
        self.lib = lib
        self.bias = bias
        self._owned: "list[tuple[str, ctypes.c_void_p]]" = []
        try:
            self.desc = self._create("cublasLtMatmulDescDestroy", "cublasLtMatmulDescCreate", CUBLAS_COMPUTE_32F, CUDA_R_32F)
            self._set(CUBLASLT_MATMUL_DESC_TRANSA, ctypes.c_int32(CUBLAS_OP_T))
            self._set(CUBLASLT_MATMUL_DESC_TRANSB, ctypes.c_int32(CUBLAS_OP_N))
            if bias:
                self._set(CUBLASLT_MATMUL_DESC_EPILOGUE, ctypes.c_uint32(CUBLASLT_EPILOGUE_BIAS))
                self._set(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, ctypes.c_int32(CUDA_R_16BF))
                self._set(CUBLASLT_MATMUL_DESC_BIAS_POINTER, ctypes.c_void_p(bias_ptr))
            else:
                self._set(CUBLASLT_MATMUL_DESC_EPILOGUE, ctypes.c_uint32(CUBLASLT_EPILOGUE_DEFAULT))
            # the "A operand" is tilefish's B[n, k]: stored k x n column-major, lda = k, used transposed (n x k)
            self.a = self._create("cublasLtMatrixLayoutDestroy", "cublasLtMatrixLayoutCreate", CUDA_R_16BF, k, n, k)
            # the "B operand" is tilefish's A[m, k]: stored k x m column-major, ldb = k, used as it is
            self.b = self._create("cublasLtMatrixLayoutDestroy", "cublasLtMatrixLayoutCreate", CUDA_R_16BF, k, m, k)
            # D[m, n] row-major is n x m column-major, ldd = n; C has the layout of D
            self.d = self._create("cublasLtMatrixLayoutDestroy", "cublasLtMatrixLayoutCreate", CUDA_R_16BF, n, m, n)
            self.preference = None
            if preference is not None:
                self.preference = self._create("cublasLtMatmulPreferenceDestroy", "cublasLtMatmulPreferenceCreate")
                for attr, value in preference:
                    # MAX_WORKSPACE_BYTES is a uint64_t, every other preference tilefish sets a uint32_t
                    number = ctypes.c_uint64(value) if attr == CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES else ctypes.c_uint32(value)
                    lib.call("cublasLtMatmulPreferenceSetAttribute", self.preference, attr, ctypes.byref(number), ctypes.sizeof(number))
        except BaseException:
            self.close(raising=False)
            raise

    def _create(self, destroy: str, create: str, *args: Any) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        self.lib.call(create, ctypes.byref(handle), *args)
        self._owned.append((destroy, handle))
        return handle

    def _set(self, attr: int, value: Any) -> None:
        self.lib.call("cublasLtMatmulDescSetAttribute", self.desc, attr, ctypes.byref(value), ctypes.sizeof(value))

    def close(self, *, raising: bool = True) -> None:
        owned, self._owned = self._owned, []
        failure: "TilefishRuntimeError | None" = None
        for destroy, handle in reversed(owned):
            try:
                self.lib.call(destroy, handle)
            except TilefishRuntimeError as e:
                failure = failure or e
        if failure is not None and raising:
            raise failure

    def __enter__(self) -> "_Problem":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close(raising=exc_type is None)


# -- the device and the handle -----------------------------------------------------------------------------------------------

_handle_lock = threading.Lock()
_handles: "dict[int, ctypes.c_void_p]" = {}


def _check_device_arg(device_index: object) -> "int | None":
    return None if device_index is None else _int_arg("device_index", device_index, low=0, high=INT32_MAX)


def _device(device_index: "int | None") -> int:
    """The device a call runs on: ``device_index``, or torch's current device."""
    torch = _torch()
    if not torch.cuda.is_available():
        raise TilefishRuntimeError("the cuBLASLt operation needs a CUDA device, and torch sees none")
    count = torch.cuda.device_count()
    if device_index is None:
        return int(torch.cuda.current_device())
    index = _check_device_arg(device_index)
    assert index is not None
    if index >= count:
        raise TilefishRuntimeError(f"cuBLASLt matmul: device_index {index} is out of range, torch sees {count} device(s)")
    return index


@contextlib.contextmanager
def _on_device(index: int) -> Iterator[None]:
    with _torch().cuda.device(index):
        yield


def _handle(lib: _Library, index: int) -> ctypes.c_void_p:
    """The cuBLASLt handle of a device, created on first use (one per device, under a lock); calls on it need no lock."""
    handle = _handles.get(index)
    if handle is None:
        with _handle_lock:
            handle = _handles.get(index)
            if handle is None:
                handle = ctypes.c_void_p()
                with _on_device(index):
                    lib.call("cublasLtCreate", ctypes.byref(handle))
                _handles[index] = handle
    return handle


def _destroy_handles() -> None:
    """Destroy every cached handle (for tests); a later call creates new ones."""
    lib = _lib()
    with _handle_lock:
        handles = list(_handles.items())
        _handles.clear()
        for index, handle in handles:
            with _on_device(index):
                lib.call("cublasLtDestroy", handle)


# -- the algorithm as a string -----------------------------------------------------------------------------------------------

_HEX = re.compile(r"[0-9a-f]{128}")


def algo_to_hex(raw: bytes) -> str:
    """The 64 bytes of a ``cublasLtMatmulAlgo_t`` as 128 lowercase hex characters (strict: exactly 64 bytes)."""
    if not isinstance(raw, (bytes, bytearray)):
        raise TilefishRuntimeError(f"a cuBLASLt algorithm is 64 bytes, got a {type(raw).__name__}")
    if len(raw) != ALGO_BYTES:
        raise TilefishRuntimeError(f"a cuBLASLt algorithm is exactly {ALGO_BYTES} bytes, got {len(raw)}")
    return bytes(raw).hex()


def algo_from_hex(text: str) -> bytes:
    """The 64 bytes a pin's hex string stands for (strict: exactly 128 lowercase hex characters, nothing else)."""
    if not isinstance(text, str):
        raise TilefishRuntimeError(f"a cuBLASLt algorithm is a string of {2 * ALGO_BYTES} lowercase hex characters, got a {type(text).__name__}")
    if not _HEX.fullmatch(text):
        raise TilefishRuntimeError(
            f"a cuBLASLt algorithm is exactly {2 * ALGO_BYTES} lowercase hex characters (0-9, a-f), got {len(text)} characters: {text[:40]!r}"
        )
    return bytes.fromhex(text)


def _algo_struct(algo_hex: str) -> _Algo:
    return _Algo.from_buffer_copy(algo_from_hex(algo_hex))


def algo_info(algo_hex: str) -> "dict[str, int]":
    """What a pinned algorithm is made of, read back from the library (``cublasLtMatmulAlgoConfigGetAttribute``): its id, tile,
    stages, split-K count, reduction scheme, swizzling, custom option, inner shape and cluster shape."""
    return _algo_config(_lib(), _algo_struct(algo_hex))


def _algo_config(lib: _Library, algo: _Algo) -> "dict[str, int]":
    info: dict[str, int] = {}
    for name, (attr, ctype) in _ALGO_CONFIG.items():
        value = ctype()
        written = ctypes.c_size_t()
        lib.call("cublasLtMatmulAlgoConfigGetAttribute", ctypes.byref(algo), attr, ctypes.byref(value), ctypes.sizeof(value), ctypes.byref(written))
        info[name] = int(value.value)
    return info


_validated: "dict[tuple[int, str, bool, bool], dict[str, int]]" = {}  # (device, hex, allow_bf16_reduction, bias) -> the configuration that passed _validate_algo


def _cap(lib: _Library, algo: _Algo, attr: int, ctype: Any, *, array: bool = False) -> Any:
    written = ctypes.c_size_t()
    if array:
        lib.call("cublasLtMatmulAlgoCapGetAttribute", ctypes.byref(algo), attr, None, 0, ctypes.byref(written))
        values = (ctype * (written.value // ctypes.sizeof(ctype)))()
        lib.call("cublasLtMatmulAlgoCapGetAttribute", ctypes.byref(algo), attr, values, ctypes.sizeof(values), ctypes.byref(written))
        return list(values)
    value = ctype()
    lib.call("cublasLtMatmulAlgoCapGetAttribute", ctypes.byref(algo), attr, ctypes.byref(value), ctypes.sizeof(value), ctypes.byref(written))
    return value.value


def _validate_algo(
    lib: _Library, handle: ctypes.c_void_p, index: int, algo_hex: str, allow_bf16_reduction: bool = False, bias: bool = False
) -> "dict[str, int]":
    """Refuse an algorithm that cuBLASLt itself would not have produced, before ``cublasLtMatmulAlgoCheck`` sees it.

    Measured (cuBLASLt 12.8.4 on a B300): ``cublasLtMatmulAlgoCheck`` accepts an algorithm with a changed version field, changed
    padding, a reserved CTA swizzling or an out-of-range custom option, and it crashes the process (SIGSEGV) on a tile id of
    2**31 or more. So the bytes are held to what the library encodes for the configuration they read back as: the algorithm
    is rebuilt with ``cublasLtMatmulAlgoInit`` (which also writes the data types, the compute type and the version) and
    ``cublasLtMatmulAlgoConfigSetAttribute``, and must be byte for byte the same; the configuration must be one that the
    capabilities of its id list (tile, stages, swizzling, custom option); it must accumulate in f32 and need no more than
    16-byte alignment, and its split-K must go with a reduction scheme that does not round in bf16, and a GEMM with a bias has no split-K
    (or, with ``allow_bf16_reduction``, any reduction scheme cuBLASLt defines: :func:`reduction_problem`). What this cannot see is a flip to
    another valid configuration (another tile, say): that is a different, legitimate algorithm.
    """
    key = (index, algo_hex, allow_bf16_reduction, bias)
    cached = _validated.get(key)
    if cached is not None:
        return cached
    head = f"the pinned cuBLASLt algorithm {algo_hex[:16]}... is not one cuBLASLt produces for bf16 with f32 accumulation"
    try:
        cfg = _validate_algo_bytes(lib, handle, algo_hex, head, allow_bf16_reduction, bias)
    except TilefishRuntimeError as e:
        if str(e).startswith(head):
            raise
        raise TilefishRuntimeError(f"{head}: {e}") from e  # a call of the library that rejects the bytes
    _validated[key] = cfg
    return cfg


def _validate_algo_bytes(
    lib: _Library, handle: ctypes.c_void_p, algo_hex: str, head: str, allow_bf16_reduction: bool = False, bias: bool = False
) -> "dict[str, int]":
    algo = _algo_struct(algo_hex)
    cfg = _algo_config(lib, algo)
    fresh = _Algo()
    try:
        lib.call(
            "cublasLtMatmulAlgoInit",
            handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, cfg["id"], ctypes.byref(fresh),
        )
    except TilefishRuntimeError as e:
        raise TilefishRuntimeError(f"{head}: algorithm id {cfg['id']} is unknown ({e})") from e
    problems: list[str] = []
    tiles = _cap(lib, fresh, _CAP_TILE_IDS, ctypes.c_uint32, array=True)
    stages = _cap(lib, fresh, _CAP_STAGES_IDS, ctypes.c_uint32, array=True)
    # an algorithm that has no tile (or stages) ids lists none and uses the _UNDEFINED value (0)
    if cfg["tile_id"] not in (tiles or [CUBLASLT_MATMUL_TILE_UNDEFINED]):
        problems.append(f"tile id {cfg['tile_id']} is not one of the {len(tiles)} that algorithm {cfg['id']} has")
    if cfg["stages_id"] not in (stages or [CUBLASLT_MATMUL_STAGES_UNDEFINED]):
        problems.append(f"stages id {cfg['stages_id']} is not one of the {len(stages)} that algorithm {cfg['id']} has")
    swizzling = _cap(lib, fresh, _CAP_CTA_SWIZZLING_SUPPORT, ctypes.c_uint32)
    if cfg["cta_swizzling"] > swizzling:
        problems.append(f"CTA swizzling {cfg['cta_swizzling']} is not supported (algorithm {cfg['id']} takes up to {swizzling})")
    custom_max = _cap(lib, fresh, _CAP_CUSTOM_OPTION_MAX, ctypes.c_int32)
    if cfg["custom_option"] > custom_max:
        problems.append(f"custom option {cfg['custom_option']} is above the maximum {custom_max} of algorithm {cfg['id']}")
    reduction = reduction_problem(cfg["splitk_num"], cfg["reduction_scheme"], allow_bf16_reduction, bias)
    if reduction is not None:
        problems.append(reduction)
    if _cap(lib, fresh, _CAP_NUMERICAL_IMPL_FLAGS, ctypes.c_uint64) & _IMPL_ACCUMULATOR_TYPE_MASK != _IMPL_ACCUMULATOR_32F:
        problems.append("it does not accumulate in f32")
    for name, attr in zip("ABCD", _CAP_MIN_ALIGNMENT_BYTES):
        needed = _cap(lib, fresh, attr, ctypes.c_uint32)
        if needed > ALIGNMENT_BYTES:
            problems.append(f"it needs {needed}-byte alignment of {name}, more than the {ALIGNMENT_BYTES} bytes tilefish guarantees")
    if not problems:
        try:
            for name, (attr, ctype) in _ALGO_CONFIG.items():
                if name != "id":
                    value = ctype(cfg[name])
                    lib.call("cublasLtMatmulAlgoConfigSetAttribute", ctypes.byref(fresh), attr, ctypes.byref(value), ctypes.sizeof(value))
        except TilefishRuntimeError as e:
            problems.append(f"its configuration cannot be set on a fresh algorithm ({e})")
        else:
            if bytes(fresh) != bytes(algo):
                problems.append("its bytes are not those cuBLASLt encodes for its configuration (version, data types or padding differ)")
    if problems:
        raise TilefishRuntimeError(f"{head}: {'; '.join(problems)}")
    return cfg


# -- pinning, checking and running -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pinned:
    """The pin of one problem: the algorithm (:func:`algo_to_hex`), the workspace it needs, and the limit it was chosen under."""

    algo_hex: str
    workspace_required: int
    workspace_limit: int


def pin_algo(
    m: int,
    n: int,
    k: int,
    bias: bool = False,
    *,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
    device_index: "int | None" = None,
    heuristic: str = "tilefish",
    allow_bf16_reduction: bool = False,
) -> Pinned:
    """Ask the heuristic for the algorithm of ``D[m, n] = A[m, k] @ B[n, k]^T (+ bias[n])`` under a workspace limit, and pin the first.

    ``workspace_bytes`` is the limit the heuristic works under (:data:`DEFAULT_WORKSPACE_BYTES` is torch's default workspace
    config). ``heuristic`` is the posture (:data:`HEURISTICS`, see the module docstring): ``"tilefish"`` asks for a few results with
    16-byte alignment minimums (the pointers are not known here, and 16-byte alignment is all that is relied on), ``"torch"`` for one
    result under the workspace limit alone, as torch's ``addmm`` does. ``allow_bf16_reduction`` lets split-K add its partial sums in bf16
    (no reduction-scheme mask); without it they are added in f32.
    """
    m, n, k, bias = _check_problem(m, n, k, bias)
    limit = _check_workspace(workspace_bytes)
    device_index = _check_device_arg(device_index)
    heuristic = _check_heuristic(heuristic)
    allow_bf16_reduction = _check_flag("allow_bf16_reduction", allow_bf16_reduction)
    count = REQUESTED_ALGO_COUNT if heuristic == "tilefish" else 1
    dummy = HEURISTIC_BIAS_POINTER if heuristic == "tilefish" else TORCH_HEURISTIC_BIAS_POINTER
    lib = _lib()
    index = _device(device_index)
    handle = _handle(lib, index)
    no_split_k = "; a GEMM with a bias is asked for one without split-K, unless it opts in with allow_bf16_reduction=True" if bias and not allow_bf16_reduction else ""
    with _on_device(index), _Problem(lib, m, n, k, bias, dummy, preferences(heuristic, limit, allow_bf16_reduction, bias)) as p:
        results = (_HeuristicResult * count)()
        returned = ctypes.c_int(0)
        status = lib.call(
            "cublasLtMatmulAlgoGetHeuristic",
            handle, p.desc, p.a, p.b, p.d, p.d, p.preference, count, results, ctypes.byref(returned),
            allow=(CUBLAS_STATUS_NOT_SUPPORTED,),  # what the library says when no algorithm fits the preferences (measured)
        )  # fmt: skip
        if status == CUBLAS_STATUS_NOT_SUPPORTED:
            returned.value = 0
        if returned.value < 1:
            raise TilefishRuntimeError(
                f"cuBLASLt returned no algorithm for the matmul m={m} n={n} k={k} bias={bias} under a workspace limit of {limit} bytes{no_split_k}"
            )
        first = results[0]
        if first.state != CUBLAS_STATUS_SUCCESS:
            raise TilefishRuntimeError(
                f"cuBLASLt's first algorithm for the matmul m={m} n={n} k={k} bias={bias} is unusable: {lib.status_name(first.state)}"
            )
        required = int(first.workspaceSize)
        if required > limit:
            raise TilefishRuntimeError(
                f"cuBLASLt's first algorithm for the matmul m={m} n={n} k={k} bias={bias} needs {required} bytes of workspace, over the limit of {limit}"
            )
        # The pin is the algorithm cuBLASLt encodes for the configuration the heuristic chose, not the 64 bytes it returned: those may hold
        # bytes that are no part of the configuration (measured, cuBLASLt 12.8.4 on an RTX 5090: two bytes that change from process to
        # process, in split-K algorithms of id 21), which would make the pin differ from run to run and fail the check of its bytes. The
        # configuration is all a pin is held to and all a replay rebuilds (_validate_algo), as the heuristic's own algorithm would be.
        canonical = _encode(lib, handle, _algo_config(lib, first.algo))
        return Pinned(algo_hex=algo_to_hex(bytes(canonical)), workspace_required=required, workspace_limit=limit)


ALGO_CONFIG_FIELDS = tuple(_ALGO_CONFIG)  # the nine configuration attributes of an algorithm, by the names algo_info and encode_algo use
_CONFIG_RANGES = {  # the C type of each attribute, as a range
    "id": (-(1 << 31), INT32_MAX),
    "tile_id": (0, (1 << 32) - 1),
    "splitk_num": (-(1 << 31), INT32_MAX),
    "reduction_scheme": (0, (1 << 32) - 1),
    "cta_swizzling": (0, (1 << 32) - 1),
    "custom_option": (0, (1 << 32) - 1),
    "stages_id": (0, (1 << 32) - 1),
    "inner_shape_id": (0, (1 << 16) - 1),
    "cluster_shape_id": (0, (1 << 16) - 1),
}


def check_config(config: object, what: str = "an algorithm configuration") -> "dict[str, int]":
    """``config`` as a dict of exactly the nine attributes of :data:`ALGO_CONFIG_FIELDS`, each an int (not a bool) in the range of its C type (pure).

    Raises :class:`~tilefish.errors.TilefishRuntimeError` naming what is wrong."""
    if not isinstance(config, dict):
        raise TilefishRuntimeError(f"{what} is a dict of {list(ALGO_CONFIG_FIELDS)}, got {config!r}")
    missing = [f for f in ALGO_CONFIG_FIELDS if f not in config]
    unknown = sorted(str(f) for f in config if f not in ALGO_CONFIG_FIELDS)
    if missing or unknown:
        raise TilefishRuntimeError(
            f"{what} has exactly the fields {list(ALGO_CONFIG_FIELDS)}" + (f"; missing {missing}" if missing else "") + (f"; unknown {unknown}" if unknown else "")
        )
    out = {}
    for f in ALGO_CONFIG_FIELDS:
        v = config[f]
        low, high = _CONFIG_RANGES[f]
        if isinstance(v, bool) or not isinstance(v, int) or not low <= v <= high:
            raise TilefishRuntimeError(f"{what}: {f} must be an int from {low} to {high}, got {v!r}")
        out[f] = v
    return out


def encode_algo(config: "dict[str, int]", *, device_index: "int | None" = None) -> str:
    """The pin (:func:`algo_to_hex`) of an algorithm given by its nine configuration attributes (:data:`ALGO_CONFIG_FIELDS`, the names
    :func:`algo_info` reads them back as): ``cublasLtMatmulAlgoInit`` for its id with bf16 operands and results and an f32 accumulator and
    scale, then ``cublasLtMatmulAlgoConfigSetAttribute`` for the other eight, which is how cuBLASLt itself encodes them (the heuristic's own
    algorithms come out byte for byte the same this way). Nothing is checked here beyond the types: :func:`check_algo` holds the result
    to the capabilities of its id and has ``cublasLtMatmulAlgoCheck`` approve it for a problem."""
    cfg = check_config(config)
    device_index = _check_device_arg(device_index)
    lib = _lib()
    index = _device(device_index)
    handle = _handle(lib, index)
    with _on_device(index):
        try:
            algo = _encode(lib, handle, cfg)
        except TilefishRuntimeError as e:
            raise TilefishRuntimeError(f"the algorithm configuration {cfg} cannot be encoded for bf16 with f32 accumulation: {e}") from e
    return algo_to_hex(bytes(algo))


def _encode(lib: _Library, handle: ctypes.c_void_p, cfg: "dict[str, int]") -> _Algo:
    """``cublasLtMatmulAlgoInit`` for ``cfg["id"]`` (bf16 operands and results, f32 accumulator and scale), then the other eight attributes."""
    algo = _Algo()
    lib.call(
        "cublasLtMatmulAlgoInit",
        handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, cfg["id"], ctypes.byref(algo),
    )  # fmt: skip
    for name, (attr, ctype) in _ALGO_CONFIG.items():
        if name != "id":
            value = ctype(cfg[name])
            lib.call("cublasLtMatmulAlgoConfigSetAttribute", ctypes.byref(algo), attr, ctypes.byref(value), ctypes.sizeof(value))
    return algo


def _check(
    lib: _Library,
    handle: ctypes.c_void_p,
    index: int,
    p: _Problem,
    algo_hex: str,
    workspace_bytes: int,
    what: str,
    allow_bf16_reduction: bool = False,
) -> int:
    _validate_algo(lib, handle, index, algo_hex, allow_bf16_reduction, p.bias)
    algo = _algo_struct(algo_hex)
    result = _HeuristicResult()
    try:
        lib.call("cublasLtMatmulAlgoCheck", handle, p.desc, p.a, p.b, p.d, p.d, ctypes.byref(algo), ctypes.byref(result))
    except TilefishRuntimeError as e:
        raise TilefishRuntimeError(f"the pinned cuBLASLt algorithm {algo_hex[:16]}... is not valid for {what}: {e}") from e
    if result.state != CUBLAS_STATUS_SUCCESS:
        raise TilefishRuntimeError(
            f"the pinned cuBLASLt algorithm {algo_hex[:16]}... is not valid for {what}: cublasLtMatmulAlgoCheck says {lib.status_name(result.state)}"
        )
    required = int(result.workspaceSize)
    if required > workspace_bytes:
        raise TilefishRuntimeError(
            f"the pinned cuBLASLt algorithm {algo_hex[:16]}... needs {required} bytes of workspace for {what}, more than the {workspace_bytes} recorded"
        )
    return required


def check_algo(
    m: int,
    n: int,
    k: int,
    bias: bool,
    algo_hex: str,
    *,
    workspace_bytes: int,
    device_index: "int | None" = None,
    allow_bf16_reduction: bool = False,
) -> int:
    """The workspace the pinned algorithm needs for this problem, after ``cublasLtMatmulAlgoCheck`` approved it.

    Raises :class:`~tilefish.errors.TilefishRuntimeError` when the algorithm is not valid for the problem on this device, or needs
    more than ``workspace_bytes`` (the workspace recorded with the pin), or adds split-K partial sums in bf16 without
    ``allow_bf16_reduction``.
    """
    m, n, k, bias = _check_problem(m, n, k, bias)
    budget = _check_workspace(workspace_bytes)
    device_index = _check_device_arg(device_index)
    allow_bf16_reduction = _check_flag("allow_bf16_reduction", allow_bf16_reduction)
    algo_from_hex(algo_hex)  # the format, before anything reaches the library
    lib = _lib()
    index = _device(device_index)
    handle = _handle(lib, index)
    with _on_device(index), _Problem(lib, m, n, k, bias, HEURISTIC_BIAS_POINTER, None) as p:
        return _check(lib, handle, index, p, algo_hex, budget, f"the matmul m={m} n={n} k={k} bias={bias}", allow_bf16_reduction)


def warm(device_index: "int | None" = None) -> None:
    """Load cuBLASLt and create the handle of a device, which the first matmul on it does: a launch plan does it before the first launch, because
    creating a handle may allocate device memory, which is not allowed while a CUDA graph is being captured."""
    index = _device(_check_device_arg(device_index))
    _handle(_lib(), index)


def matmul(
    a_ptr: int,
    b_ptr: int,
    d_ptr: int,
    bias_ptr: "int | None",
    m: int,
    n: int,
    k: int,
    algo_hex: str,
    *,
    workspace_bytes: int,
    stream: int,
    device_index: "int | None" = None,
    workspace_ptr: "int | None" = None,
    allow_bf16_reduction: bool = False,
) -> None:
    """``D[m, n] = A[m, k] @ B[n, k]^T (+ bias[n])`` with the pinned algorithm, enqueued on ``stream`` (a ``CUstream`` as an int).

    ``a_ptr``, ``b_ptr``, ``d_ptr`` and ``bias_ptr`` (``None`` for no bias) are device addresses, 16-byte aligned. The
    algorithm is checked for this problem first (see :func:`check_algo`); ``workspace_bytes`` is the workspace recorded with
    the pin, an upper bound of what the algorithm may need. The workspace is taken from torch's caching allocator, on
    ``stream``'s device and ordered on ``stream``, and lives only as long as this call (the allocator keeps it from being reused
    before the work already queued on ``stream`` is done).

    With ``workspace_ptr`` (the address of at least ``workspace_bytes`` bytes of device memory, aligned to 256 bytes) the workspace is
    that memory and nothing is allocated: this is how a launch plan runs it (the workspace is a buffer of the plan's arena, so the call
    allocates nothing and can be captured in a CUDA graph). Besides the matmul itself the call touches the device not at all.

    An algorithm whose split-K adds its partial sums in bf16 runs only with ``allow_bf16_reduction=True`` (the declaration's opt-in).
    """
    bias = bias_ptr is not None
    m, n, k, bias = _check_problem(m, n, k, bias)
    a = _check_pointer("A", a_ptr)
    b = _check_pointer("B", b_ptr)
    d = _check_pointer("D", d_ptr)
    bias_address = _check_pointer("bias", bias_ptr) if bias else 0
    budget = _check_workspace(workspace_bytes)
    stream = _int_arg("stream", stream, low=0, high=UINT64_MAX)
    device_index = _check_device_arg(device_index)
    algo = _algo_struct(algo_hex)
    given = None if workspace_ptr is None else _check_workspace_pointer(workspace_ptr)
    allow_bf16_reduction = _check_flag("allow_bf16_reduction", allow_bf16_reduction)
    lib = _lib()
    index = _device(device_index)
    handle = _handle(lib, index)
    with _on_device(index), _Problem(lib, m, n, k, bias, bias_address, None) as p:
        required = _check(lib, handle, index, p, algo_hex, budget, f"the matmul m={m} n={n} k={k} bias={bias}", allow_bf16_reduction)
        workspace = None
        address = given
        if required and given is None:
            torch = _torch()
            with torch.cuda.stream(torch.cuda.ExternalStream(stream, device=index)):
                workspace = torch.empty(required, dtype=torch.uint8, device=f"cuda:{index}")
            address = workspace.data_ptr()
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        # cuBLAS A operand = tilefish B, cuBLAS B operand = tilefish A; C is D (beta = 0, so it is never read)
        lib.call(
            "cublasLtMatmul",
            handle, p.desc, ctypes.byref(alpha), b, p.a, a, p.b, ctypes.byref(beta), d, p.d, d, p.d,
            ctypes.byref(algo), address if required else None, required, stream,
        )
        del workspace
