"""Library operations with a pinned implementation: a cuBLASLt GEMM.

``@tilefish.library("gemm", acc=f32, impl="cublaslt.gemm")`` declares an operation by its signature and says which library
routine runs it. For ``impl="cublaslt.gemm"`` the signature is ``(a: Tensor[M, K, T], b: Tensor[N, K, T]) -> Tensor[M, N, D]``
and for ``"cublaslt.gemm_bias"`` it has a third parameter ``bias: Tensor[N, D]``, where ``T`` is ``bf16``, ``f16`` or ``f32`` and ``D`` is ``T``
or ``f32`` (``f32`` for ``f32`` operands). With ``layout="NN"`` (the default is ``"TN"``) ``b`` is ``Tensor[K, N, T]``, and with ``"TT"`` and
``"NT"`` ``a`` is ``Tensor[K, M, T]`` (stored transposed) with the ``b`` of ``"TN"`` and ``"NN"``. Semantics, in tilefish's row-major terms:

    out[M, N] = a[M, K] @ b[N, K]^T        (+ bias[N])          layout "TN"
    out[M, N] = a[M, K] @ b[K, N]          (+ bias[N])          layout "NN"
    out[M, N] = a[K, M]^T @ b[N, K]^T      (+ bias[N])          layout "TT"
    out[M, N] = a[K, M]^T @ b[K, N]        (+ bias[N])          layout "NT"

with an f32 accumulator and **one** rounding, to ``D``, at the end; the bias goes through ``CUBLASLT_EPILOGUE_BIAS``, so it is added in
f32 to the accumulator before that rounding (cuBLASLt takes a bias of the result's type only: measured, :mod:`tilefish._cublaslt`). f32 operands
are multiplied as f32, with TF32 off: the compute type is ``CUBLAS_COMPUTE_32F``, an algorithm is refused where cuBLASLt would run it as TF32 or
as a bf16x9 emulation, and its numerical-implementation flags, where reported, must say f32 inputs (:func:`tilefish._cublaslt.numerics_problems`). Every
operand is contiguous with a 16-byte aligned pointer (a strided operand is not supported in this version: pass ``.contiguous()``). ``K`` and
``N`` may be any size: when a row of ``k`` or ``n`` elements does not start on 16 bytes (``K`` or ``N`` not a multiple of 8 for 16-bit
operands), an algorithm that needs aligned rows is not valid for the problem, which ``cublasLtMatmulAlgoCheck`` says (it sees the leading
dimensions) when the algorithm is pinned and before every run.

**A binary is a pinned algorithm.** For every ``(M, N, K, bias)`` (``M``, ``N`` and ``K`` are static or bucketed: one entry per bucket
combination, like a kernel) the heuristic of cuBLASLt is asked, with a fixed workspace limit (``workspace_bytes``), for its best
algorithm; result 0 is the pin, the 64 bytes of its ``cublasLtMatmulAlgo_t`` written as 128 lowercase hex digits. A manifest records it
in its ``library`` section (see ``docs/en/manifest-schema-3.md``) with the cuBLASLt version that chose it, and ``jit`` pins in memory the same
way, so ``jit`` and ``replay`` run the same algorithm by construction. ``replay`` refuses a cuBLASLt of another version (an algorithm id
means other kernels there) and an entry that is not what the declaration asks for.

How the heuristic is asked is the declaration's ``heuristic=`` (:mod:`tilefish._cublaslt`): ``"tilefish"`` (the default; workspace limit
8 519 680, torch's default ``CUBLAS_WORKSPACE_CONFIG``, and 16-byte alignment minimums) or ``"torch"`` (what torch's ``addmm`` asks: one
result under the workspace limit alone; the limit is 1 048 576 for ``gemm_bias``, torch's default ``CUBLASLT_WORKSPACE_SIZE``, and
8 519 680 for ``gemm``, the workspace of the ``cublasGemmEx`` that ``torch.mm`` runs). Split-K adds its partial sums in f32 unless the
declaration opts in with the flag of its operands' type, ``allow_bf16_reduction=True``, ``allow_f16_reduction=True`` (torch's
``allow_bf16_reduced_precision_reduction`` and ``allow_fp16_reduced_precision_reduction``) or ``allow_f32_reduction=True`` (torch has no flag for
f32: it never masks the reduction scheme of an f32 GEMM, so its choice may need this opt-in). And ``algos=`` (:class:`AlgoTable`, one per device)
imports the algorithms instead: on a device a table is for, the heuristic is not asked, and each entry runs the algorithm the table gives for its
``(M, N, K, bias)`` (and layout).

Nothing here imports torch or loads cuBLASLt until a build or a run needs them (:mod:`tilefish._cublaslt`).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import _specvalid as sv
from .dims import Dim, DimExpr
from .errors import TilefishDeclarationError, TilefishManifestError, TilefishRuntimeError, TilefishSpecError

if TYPE_CHECKING:
    from .decorators import LibraryOp
    from .signature import Signature

IMPL_GEMM = "cublaslt.gemm"
IMPL_GEMM_BIAS = "cublaslt.gemm_bias"
IMPLS = {IMPL_GEMM: False, IMPL_GEMM_BIAS: True}  # impl -> whether it has a bias (the two impls of tilefish 0.0.4, unchanged)
# A GEMM with a declared cuBLASLt epilogue (RFC 0003, milestone E2; docs/en/manifest-schema-3.md, section 9.5): out = epilogue(acc [+ c] [+ bias]), the
# library's bytes under the pin (contract (iii)), each pin verified at the build against contract (ii) and recorded "measured-exact" or
# "binary-defined", with a canary.
IMPL_MATMUL = "cublaslt.matmul"
MATMUL_EPILOGUES = ("none", "bias", "relu", "relu_bias")  # cuBLASLt's DEFAULT, BIAS, RELU, RELU_BIAS
MATMUL_WITH_BIAS = ("bias", "relu_bias")
MATMUL_DTYPES = ("bf16", "f16")  # of a and b, and of the result, a bias and c (the same type): an f32 accumulator, one rounding to it
CONTRACTS = ("measured-exact", "binary-defined")  # what an entry's verification found (contract (iii) either way)
CUBLASLT_IMPLS = (IMPL_GEMM, IMPL_GEMM_BIAS, IMPL_MATMUL)


def formula(epilogue: str, beta: int) -> str:
    """The arithmetic of a ``cublaslt.matmul`` on the f32 accumulator ``acc``, before its one rounding, in the order the library adds (measured: X1 of
    RFC 0003, ``C`` before the bias): ``"relu((acc + c) + bias)"``. What a pin is verified against (contract (ii)) and what an entry's
    ``verification.formula`` says."""
    x = "acc + c" if beta else "acc"
    if epilogue in MATMUL_WITH_BIAS:
        x = f"({x}) + bias" if beta else f"{x} + bias"
    return f"relu({x})" if epilogue.startswith("relu") else x


def bias_of(impl: str, epilogue: "str | None") -> bool:
    """Whether an entry of ``impl`` (with ``epilogue``, for ``cublaslt.matmul``) has a ``bias`` operand."""
    return epilogue in MATMUL_WITH_BIAS if impl == IMPL_MATMUL else IMPLS[impl]
DEFAULT_WORKSPACE_BYTES = 8_519_680  # torch's default CUBLAS_WORKSPACE_CONFIG (":4096:2:16:8": 4096 KiB * 2 + 16 KiB * 8)
TORCH_LT_WORKSPACE_BYTES = 1_048_576  # torch's default CUBLASLT_WORKSPACE_SIZE (1024 KiB): the limit its addmm asks the heuristic under
HEURISTICS = ("tilefish", "torch")
MAX_WORKSPACE_BYTES = 1 << 40
ALIGNMENT = 16
DIGEST_FORMAT = 2  # of the canonical declaration below (2: the digest of an entry covers the SM count of its device)

# what every entry of this implementation says about the problem, as cuBLASLt is called (see the module docstring and describe())
COMPUTE_TYPE = "CUBLAS_COMPUTE_32F"
SCALE_TYPE = "CUDA_R_32F"
TRANSA, TRANSB = "T", "N"  # cuBLAS is column-major: its A operand is tilefish's b (op T), its B operand is tilefish's a (op N); the layout TN
# transa then transb as cuBLASLt is called: b is [N, K] for transa T (lda = k), [K, N] for transa N (lda = n); a is [M, K] for transb N (ldb = k),
# [K, M] for transb T (ldb = m)
LAYOUTS = ("TN", "NN", "TT", "NT")
OPERAND_DTYPES = ("bf16", "f16", "f32")  # of a and b; the result (and a bias) is of their type or f32 (f32 for f32 operands)

_ALGO = re.compile(r"^[0-9a-f]{128}\Z")
_ARCH = re.compile(r"^sm_[0-9]{2,3}\Z")
ENTRY_KEYS = (
    "stem op arch impl abi buckets dims preconditions dtypes compute_type scale_type transa transb m n k lda ldb ldc ldd "
    "epilogue workspace_bytes algo cublaslt_version sm_count declaration_digest"
).split()
# Fields an entry has only when its declaration is not the default, each with the one value it may then have (absent: the default, so an entry
# of a default declaration is what tilefish 0.0.4 wrote): the heuristic posture, an imported algorithm, the opt-in to split-K partial sums in the
# output type (named by the operands' type: allow_bf16_reduction on bf16 operands, allow_f16_reduction on f16 ones, allow_f32_reduction on f32 ones).
OPTIONAL_ENTRY_FIELDS = {
    "heuristic": "torch",
    "algo_source": "imported",
    "allow_bf16_reduction": True,
    "allow_f16_reduction": True,
    "allow_f32_reduction": True,
}
OPT_INS = ("allow_bf16_reduction", "allow_f16_reduction", "allow_f32_reduction")  # the opt-in of each operand type, in the order of OPERAND_DTYPES
# An optional field with a count: present only when the declaration targets an SM count (sm_count_target=), an int from 8 to the entry's sm_count.
OPTIONAL_COUNT_FIELDS = ("sm_count_target",)
# An entry of cublaslt.matmul (section 9.5): the fields of a GEMM's and its beta, the contract its verification found and the record of that
# verification; optionally the options of section 9.1 to 9.3 (not algo_source: it imports no algorithm) and a canary.
MATMUL_ENTRY_KEYS = [*ENTRY_KEYS, "beta", "contract", "verification"]
MATMUL_OPTIONAL_FIELDS = ("heuristic", "allow_bf16_reduction", "allow_f16_reduction", "allow_f32_reduction", "sm_count_target", "canary")
VERIFICATION_KEYS = ("formula", "accumulator", "operand_sets", "elements", "differ", "max_ulp")
ACCUMULATORS = ("f32-form", "plain-f32-pin")  # where the reference's accumulator came from: the pin's configuration with an f32 D, or the f32 pin
CANARY_KEYS = ("seed", "out_digest")
_HEX32 = re.compile(r"^[0-9a-f]{32}\Z")


def has_bias(impl: str) -> bool:
    return IMPLS[impl]


def table_key(m: int, n: int, k: int, bias: bool, layout: str = "TN") -> "tuple[Any, ...]":
    """The key of an :class:`AlgoTable` row: ``(m, n, k, bias)`` for the layout ``TN`` (tilefish 0.0.5's only one), ``(m, n, k, bias, layout)``
    for another."""
    return (m, n, k, bias) if layout == "TN" else (m, n, k, bias, layout)


def a_dims(layout: str) -> str:
    """How ``a`` is stored in ``layout``, as a message says it: ``"M, K"`` (``TN``, ``NN``) or ``"K, M"`` (``TT``, ``NT``)."""
    return "M, K" if layout[1] == "N" else "K, M"


def b_dims(layout: str) -> str:
    """How ``b`` is stored in ``layout``, as a message says it: ``"N, K"`` (``TN``, ``TT``) or ``"K, N"`` (``NN``, ``NT``)."""
    return "N, K" if layout[0] == "T" else "K, N"


def operand_shapes(layout: str, m: Any, n: Any, k: Any) -> "tuple[tuple[Any, Any], tuple[Any, Any]]":
    """The shapes of ``a`` and of ``b`` of the problem ``m``, ``n``, ``k`` in ``layout``."""
    a = (m, k) if layout[1] == "N" else (k, m)
    b = (n, k) if layout[0] == "T" else (k, n)
    return a, b


def leading_dimensions(layout: str, m: int, n: int, k: int) -> "tuple[int, int, int, int]":
    """``lda``, ``ldb``, ``ldc``, ``ldd`` as cuBLASLt is called: ``lda`` is ``k`` for ``b [n, k]`` and ``n`` for ``b [k, n]``, ``ldb`` is ``k`` for
    ``a [m, k]`` and ``m`` for ``a [k, m]``, and the result is ``n x m`` column-major."""
    return (k if layout[0] == "T" else n, k if layout[1] == "N" else m, n, n)


def leading_dimensions_rule(layout: str) -> str:
    """The rule of :func:`leading_dimensions` in the words a refusal uses (tilefish 0.0.5's for ``TN``, 0.0.6's for ``NN``)."""
    if layout == "TN":
        return "the leading dimensions must be lda = ldb = k and ldc = ldd = n (cuBLAS sees b as its A, a as its B and the result as C and D)"
    lda = "k" if layout[0] == "T" else "n"
    ldb = "k" if layout[1] == "N" else "m"
    b = "b" if layout[0] == "T" else "b [k, n]"
    a = "a" if layout[1] == "N" else "a [k, m]"
    return (
        f"the leading dimensions of the layout {layout} must be lda = {lda}, ldb = {ldb} and ldc = ldd = n (cuBLAS sees {b} as its A, {a} as its B "
        "and the result as C and D)"
    )


def default_workspace_bytes(impl: str, heuristic: str, epilogue: "str | None" = None) -> int:
    """The workspace limit of a declaration that gives none: torch's ``addmm`` limit for a biased GEMM under the ``"torch"`` posture (and for a
    ``cublaslt.matmul`` with an epilogue: torch runs ``addmm`` with a bias and ``_addmm_activation`` through cuBLASLt), torch's cuBLAS workspace
    otherwise."""
    if impl == IMPL_MATMUL:
        return TORCH_LT_WORKSPACE_BYTES if heuristic == "torch" and epilogue != "none" else DEFAULT_WORKSPACE_BYTES
    return TORCH_LT_WORKSPACE_BYTES if heuristic == "torch" and IMPLS[impl] else DEFAULT_WORKSPACE_BYTES


def check_matmul_options(name: str, epilogue: Any, beta: Any, canary: Any, contract: Any, algos: Any) -> "tuple[str, int, bool, str | None]":
    """Check the options of a ``cublaslt.matmul`` declaration: ``epilogue`` (one of :data:`MATMUL_EPILOGUES`, required), ``beta`` (0 or 1),
    ``canary`` (a bool) and ``contract`` (``None``, or ``"measured-exact"``: a pin that is not is refused at the build); ``algos=`` is not taken yet.
    Returns them with their defaults (beta 0, canary on)."""
    where = f"library op '{name}'"
    if epilogue not in MATMUL_EPILOGUES:
        raise TilefishDeclarationError(
            f"{where}: {IMPL_MATMUL} needs epilogue= one of {list(MATMUL_EPILOGUES)} (cuBLASLt's DEFAULT, BIAS, RELU, RELU_BIAS; GELU is not offered "
            f"yet), got {epilogue!r}"
        )
    beta = 0 if beta is None else beta
    if isinstance(beta, bool) or not isinstance(beta, int) or beta not in (0, 1):
        raise TilefishDeclarationError(f"{where}: beta must be 0 (no c) or 1 (out = epilogue(acc + c), with c: Tensor[M, N, D]), got {beta!r}")
    canary = True if canary is None else canary
    if not isinstance(canary, bool):
        raise TilefishDeclarationError(f"{where}: canary must be True (the default: export records one, a load runs it) or False, got {canary!r}")
    if contract not in (None, "measured-exact"):
        raise TilefishDeclarationError(
            f"{where}: contract must be None (the library's bytes, measured-exact or binary-defined, recorded per pin) or 'measured-exact' (a pin "
            f"whose output is not contract (ii) is refused at the build), got {contract!r}"
        )
    if algos is not None:
        raise TilefishDeclarationError(
            f"{where}: {IMPL_MATMUL} does not import algorithms (algos=) yet: its pins are the heuristic's, under heuristic='tilefish' or 'torch'"
        )
    return str(epilogue), int(beta), canary, contract


# -- imported algorithms ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AlgoTable:
    """cuBLASLt algorithms to pin instead of asking the heuristic, for one device: an architecture (``"sm_120"``), the SM count of the device
    (``CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT``: a MIG slice has its own) and the ``cublasLtGetVersion()`` they belong to.

    ``algos`` maps ``(m, n, k, bias)`` (tilefish's sizes, ``a [m, k]``, ``b [n, k]``, and whether the GEMM has a bias) to an algorithm: a dict
    of its nine configuration attributes (``id``, ``tile_id``, ``splitk_num``, ``reduction_scheme``, ``cta_swizzling``, ``custom_option``,
    ``stages_id``, ``inner_shape_id``, ``cluster_shape_id``: what ``cublasLtMatmulAlgoConfigGetAttribute`` reads back, and what
    ``tilefish._cublaslt.algo_info`` returns), which is encoded as cuBLASLt itself encodes it (with the operand and result types of the
    declaration that uses it: a configuration has no types), or the 128 hex digits of a pin (which have them: the checks refuse a pin of other
    types). A key of another layout than ``TN`` says it: ``(m, n, k, bias, "NN")``, ``"TT"`` or ``"NT"`` (``(m, n, k, bias, "TN")`` is
    ``(m, n, k, bias)``). A declaration
    with ``algos=`` runs, on a device a table is for, the table's algorithm for each of its entries, after the same checks as a pin of the
    heuristic (the bytes are what cuBLASLt encodes, the capabilities of the id, ``cublasLtMatmulAlgoCheck`` for the problem, the workspace limit,
    and a split-K reduction in the output type only with the opt-in of the operands' type); it is recorded in the entry as
    ``"algo_source": "imported"``. Rows for other shapes are allowed (one table can serve several declarations) and do not reach a declaration's
    digest."""

    arch: str
    sm_count: int
    cublaslt_version: int
    algos: "dict[tuple[Any, ...], Any]"

    def __post_init__(self) -> None:
        from . import _cublaslt

        where = "AlgoTable"
        if not isinstance(self.arch, str) or not _ARCH.match(self.arch):
            raise TilefishDeclarationError(f"{where}: arch must look like 'sm_120', got {self.arch!r}")
        for name in ("sm_count", "cublaslt_version"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise TilefishDeclarationError(f"{where}: {name} must be a positive int, got {v!r}")
        if not isinstance(self.algos, dict) or not self.algos:
            raise TilefishDeclarationError(f"{where}: algos must be a non-empty dict of (m, n, k, bias) -> algorithm, got {self.algos!r}")
        rows: "dict[tuple[Any, ...], Any]" = {}
        for key, algo in self.algos.items():
            if (
                not isinstance(key, tuple)
                or len(key) not in (4, 5)
                or any(isinstance(x, bool) or not isinstance(x, int) or not 1 <= x <= _cublaslt.INT32_MAX for x in key[:3])
                or not isinstance(key[3], bool)
                or (len(key) == 5 and key[4] not in LAYOUTS)
            ):
                raise TilefishDeclarationError(
                    f"{where}: a key is (m, n, k, bias), three ints from 1 to 2**31 - 1 and a bool, or (m, n, k, bias, layout) with a layout of "
                    f"{list(LAYOUTS)}, got {key!r}"
                )
            row = table_key(*key)
            if row in rows:
                raise TilefishDeclarationError(f"{where}: two rows are for {row} (a key of the layout TN may leave it out, but not both)")
            what = f"{where} ({self.arch}, {self.sm_count} SMs) algorithm for {key}"
            try:
                if isinstance(algo, str):
                    _cublaslt.algo_from_hex(algo)  # 128 lowercase hex digits
                    rows[row] = algo
                else:
                    rows[row] = _cublaslt.check_config(algo, what)
            except TilefishRuntimeError as e:  # the binding's pure checks; about a declaration here
                raise TilefishDeclarationError(str(e)) from None
        object.__setattr__(self, "algos", rows)

    @property
    def device(self) -> "tuple[str, int, int]":
        return (self.arch, self.sm_count, self.cublaslt_version)

    def canonical(self, keys: "list[tuple[Any, ...]]") -> "dict[str, Any]":
        """The table restricted to ``keys`` (of one declaration, so of one layout, which its canonical form says), as canonical JSON-able data
        (what a declaration's digest covers): a row is ``[m, n, k, bias, algorithm]``."""
        return {
            "arch": self.arch,
            "sm_count": self.sm_count,
            "cublaslt_version": self.cublaslt_version,
            "algos": [[*key[:4], self.algos[key]] for key in sorted(keys)],
        }

    def __repr__(self) -> str:
        return f"AlgoTable({self.arch!r}, sm_count={self.sm_count}, cublaslt_version={self.cublaslt_version}, {len(self.algos)} algorithms)"


def _check_tables(where: str, algos: Any) -> "tuple[AlgoTable, ...]":
    tables = (algos,) if isinstance(algos, AlgoTable) else algos
    if not isinstance(tables, (list, tuple)) or not tables or any(not isinstance(t, AlgoTable) for t in tables):
        raise TilefishDeclarationError(f"{where}: algos must be an AlgoTable or a non-empty list of them, got {algos!r}")
    seen: "set[tuple[str, int, int]]" = set()
    for t in tables:
        if t.device in seen:
            raise TilefishDeclarationError(
                f"{where}: two algorithm tables are for {t.arch} with {t.sm_count} SMs and cuBLASLt {t.cublaslt_version}; give one per device"
            )
        seen.add(t.device)
    return tuple(tables)


def table_for(tables: "tuple[AlgoTable, ...]", arch: str, sm_count: int, version: int) -> "AlgoTable | None":
    """The table of ``tables`` for a device of ``arch`` with ``sm_count`` SMs and cuBLASLt ``version``, or ``None``."""
    for t in tables:
        if t.device == (arch, sm_count, version):
            return t
    return None


# -- the declaration -------------------------------------------------------------------------------------------------------


def check_options(
    name: str,
    heuristic: Any,
    allow_bf16_reduction: Any,
    algos: Any,
    allow_f16_reduction: Any = False,
    layout: Any = "TN",
    sm_count_target: Any = None,
    allow_f32_reduction: Any = False,
) -> "tuple[AlgoTable, ...] | None":
    """Check the options of a ``@library(impl=...)`` declaration that say how its algorithms are chosen (the posture, the opt-ins to split-K
    partial sums in the output type, the imported tables) and its layout; return the tables (``None`` without). Raises a
    ``TilefishDeclarationError``. (Which opt-in a declaration may set depends on its operands: :func:`check_declaration`.)"""
    where = f"library op '{name}'"
    if heuristic not in HEURISTICS:
        raise TilefishDeclarationError(f"{where}: heuristic must be one of {list(HEURISTICS)}, got {heuristic!r}")
    for flag, value in zip(OPT_INS, (allow_bf16_reduction, allow_f16_reduction, allow_f32_reduction)):
        if not isinstance(value, bool):
            raise TilefishDeclarationError(f"{where}: {flag} must be a bool, got {value!r}")
    if layout not in LAYOUTS:
        raise TilefishDeclarationError(
            f"{where}: layout must be one of {list(LAYOUTS)} (TN: b is [N, K], out = a @ b^T; NN: b is [K, N], out = a @ b; TT and NT: a is [K, M], "
            f"out = a^T @ b^T and a^T @ b), got {layout!r}"
        )
    if sm_count_target is not None:
        from ._cublaslt import INT32_MAX, MIN_SM_COUNT_TARGET

        v = sm_count_target
        if isinstance(v, bool) or not isinstance(v, int) or not MIN_SM_COUNT_TARGET <= v <= INT32_MAX:
            raise TilefishDeclarationError(
                f"{where}: sm_count_target must be None or an int of at least {MIN_SM_COUNT_TARGET} (the SMs of the partition the algorithm is "
                f"chosen for; cuBLASLt 12.8.4 kills the process for a target below 8), got {v!r}"
            )
    return None if algos is None else _check_tables(where, algos)


def check_declaration(
    name: str,
    impl: Any,
    sig: "Signature",
    acc: Any,
    workspace_bytes: Any,
    layout: str = "TN",
    allow_bf16_reduction: bool = False,
    allow_f16_reduction: bool = False,
    allow_f32_reduction: bool = False,
    epilogue: "str | None" = None,
    beta: int = 0,
) -> None:
    """Check that a ``@library(impl=...)`` declaration has the signature of its implementation in its layout, and that its opt-in is the one of
    its operands' type; raise a ``TilefishDeclarationError``. ``epilogue`` and ``beta`` are those of a ``cublaslt.matmul``
    (:func:`check_matmul_options`)."""
    where = f"library op '{name}'"
    if impl not in CUBLASLT_IMPLS:
        raise TilefishDeclarationError(f"{where}: impl must be one of {sorted(IMPLS)}, got {impl!r}")
    if acc is not None and getattr(acc, "name", None) != "f32":
        raise TilefishDeclarationError(f"{where}: {impl} accumulates in f32 (acc=f32), got acc={acc!r}")
    if isinstance(workspace_bytes, bool) or not isinstance(workspace_bytes, int) or not 1 <= workspace_bytes <= MAX_WORKSPACE_BYTES:
        raise TilefishDeclarationError(f"{where}: workspace_bytes must be an int from 1 to 2**40, got {workspace_bytes!r}")
    matmul = impl == IMPL_MATMUL
    bias = bias_of(impl, epilogue)
    with_c = matmul and beta == 1
    params = list(sig.params)
    want = 2 + bias + with_c
    b_text = f"b: Tensor[{b_dims(layout)}, T]"
    if len(params) != want or any(p.kind != "tensor" or p.access != "in" or p.strided or p.is_devscalar for p in params):
        if matmul:
            raise TilefishDeclarationError(
                f"{where}: {impl} with epilogue={epilogue!r} and beta={beta} takes {want} input tensors (a: Tensor[{a_dims(layout)}, T], {b_text}"
                f"{', bias: Tensor[N, T]' if bias else ''}{', c: Tensor[M, N, T]' if with_c else ''}), and returns Tensor[M, N, T], where T is bf16 or f16 "
                f"(layout {layout}); the signature is {sig}"
            )
        raise TilefishDeclarationError(
            f"{where}: {impl} takes {want} input tensors "
            f"(a: Tensor[{a_dims(layout)}, T], {b_text}{', bias: Tensor[N, D]' if bias else ''}), and returns Tensor[M, N, D], where T is bf16, f16 or "
            f"f32 and D is T or f32 (layout {layout}); the signature is {sig}"
        )
    if len(sig.returns) != 1 or sig.tensor_outputs:
        raise TilefishDeclarationError(f"{where}: {impl} returns exactly one tensor, Tensor[M, N, D] (not an Out parameter)")
    a, b = params[0].type, params[1].type
    out = sig.returns[0]
    for label, t in (("a", a), ("b", b), ("the result", out)):
        assert t is not None
        if t.rank != 2 or not t.contiguous:  # type: ignore[union-attr]
            raise TilefishDeclarationError(f"{where}: {label} must be a contiguous tensor of rank 2, got {t}")
    dtype = a.dtype.name  # type: ignore[union-attr]
    if dtype not in OPERAND_DTYPES or b.dtype.name != dtype:  # type: ignore[union-attr]
        raise TilefishDeclarationError(f"{where}: a and b must both be bf16, both be f16 or both be f32 tensors, got {a} and {b}")
    if dtype == "f32" and out.dtype.name != "f32":  # type: ignore[union-attr]
        raise TilefishDeclarationError(
            f"{where}: the result of f32 operands must be an f32 tensor (they are multiplied as f32, TF32 off, and rounded once to the result), got {out}"
        )
    if out.dtype.name not in (dtype, "f32"):  # type: ignore[union-attr]
        raise TilefishDeclarationError(f"{where}: the result of {dtype} operands must be a {dtype} or an f32 tensor, got {out}")
    if matmul and (dtype not in MATMUL_DTYPES or out.dtype.name != dtype):  # type: ignore[union-attr]
        raise TilefishDeclarationError(
            f"{where}: {impl} takes bf16 or f16 operands and returns their type (an f32 accumulator, rounded once after the epilogue), got {a}, {b} "
            f"and {out}"
        )
    if with_c:
        ct = params[2 + bias].type
        assert ct is not None
        if ct.rank != 2 or not ct.contiguous or ct.dtype.name != out.dtype.name:  # type: ignore[union-attr]
            raise TilefishDeclarationError(f"{where}: c must be a contiguous {out.dtype.name} tensor of rank 2, the type of the result, got {ct}")  # type: ignore[union-attr]
    if bias:
        bt = params[2].type
        assert bt is not None
        if bt.rank != 1 or not bt.contiguous or bt.dtype.name != out.dtype.name:  # type: ignore[union-attr]
            raise TilefishDeclarationError(
                f"{where}: bias must be a {out.dtype.name} tensor of rank 1, the type of the result (cuBLASLt takes no other bias type), got {bt}"  # type: ignore[union-attr]
            )
    m, k = (a.shape[0], a.shape[1]) if layout[1] == "N" else (a.shape[1], a.shape[0])  # type: ignore[union-attr]
    n, b_k = (b.shape[0], b.shape[1]) if layout[0] == "T" else (b.shape[1], b.shape[0])  # type: ignore[union-attr]
    if b_k != k:
        if layout == "TN":
            raise TilefishDeclarationError(f"{where}: a is [M, K] and b is [N, K]: their last axes must be the same ({k} vs {b_k})")
        a_axis, b_axis = ("last" if layout[1] == "N" else "first"), ("last" if layout[0] == "T" else "first")
        axes = f"their {a_axis} axes" if a_axis == b_axis else f"the {a_axis} axis of a and the {b_axis} of b"
        raise TilefishDeclarationError(
            f"{where}: a is [{a_dims(layout)}] and b is [{b_dims(layout)}] (layout {layout}): {axes} must be the same ({k} vs {b_k})"
        )
    if tuple(out.shape) != (m, n) or (bias and params[2].type.shape != (n,)):  # type: ignore[union-attr]
        raise TilefishDeclarationError(f"{where}: the result must be [M, N] = [{m}, {n}]" + (f" and bias [N] = [{n}]" if bias else ""))
    if with_c and tuple(params[2 + bias].type.shape) != (m, n):  # type: ignore[union-attr]
        raise TilefishDeclarationError(f"{where}: c must be [M, N] = [{m}, {n}], the shape of the result")
    for flag, value in zip(OPT_INS, (allow_bf16_reduction, allow_f16_reduction, allow_f32_reduction)):
        if value and flag != f"allow_{dtype}_reduction":
            raise TilefishDeclarationError(
                f"{where}: {flag} is the opt-in of {flag.split('_')[1]} operands; a GEMM of {dtype} operands opts in with allow_{dtype}_reduction=True"
            )
    for label, d in (("M", m), ("N", n), ("K", k)):
        free = d.free_dims()
        loose = sorted(f.name for f in free if not f.buckets)
        if loose:
            raise TilefishDeclarationError(
                f"{where}: {label} = {d} depends on the dynamic dimension {', '.join(loose)} without buckets; a pinned algorithm is for "
                "one size, so give it Dim(name, buckets=[...]) or a static value"
            )
    # N and K may be any size: the leading dimensions (k and n) are the library's to judge for an algorithm (cublasLtMatmulAlgoCheck, at the pin
    # and before every run), and every pointer is still 16-byte aligned, which is more than any algorithm tilefish pins needs (the check of
    # its bytes refuses one that needs more)


def check_coverage(op: "LibraryOp") -> None:
    """Every table of a declaration with ``algos=`` has an algorithm for every ``(m, n, k, bias)`` the declaration has an entry for (one per
    bucket combination): a table is for a device, and on that device every entry runs one of its algorithms."""
    tables = getattr(op, "algos", None)
    if not tables:
        return
    keys = _problem_keys(op)
    for t in tables:
        missing = [key for key in keys if key not in t.algos]
        if missing:
            named = "(m, n, k, bias)" if layout_of(op) == "TN" else "(m, n, k, bias, layout)"
            raise TilefishDeclarationError(
                f"library op '{op.name}': the algorithm table for {t.arch} with {t.sm_count} SMs (cuBLASLt {t.cublaslt_version}) has no algorithm for "
                f"{named} = {', '.join(map(str, missing[:4]))}{' ...' if len(missing) > 4 else ''}: a table gives one for every entry of the declaration"
            )


def layout_of(op: "LibraryOp") -> str:
    return getattr(op, "layout", "TN")


def _mnk_exprs(op: "LibraryOp") -> "tuple[DimExpr, DimExpr, DimExpr]":
    """The dimension expressions of ``m``, ``n`` and ``k`` of a library op: ``a`` is ``[M, K]`` (TN, NN) or ``[K, M]`` (TT, NT), ``b`` is ``[N, K]``
    (TN, TT) or ``[K, N]`` (NN, NT)."""
    sig = op.signature
    layout = layout_of(op)
    a_shape = sig.params[0].type.shape  # type: ignore[union-attr]
    m_expr, k_expr = (a_shape[0], a_shape[1]) if layout[1] == "N" else (a_shape[1], a_shape[0])
    b_shape = sig.params[1].type.shape  # type: ignore[union-attr]
    n_expr = b_shape[0] if layout[0] == "T" else b_shape[1]
    return m_expr, n_expr, k_expr


def form_of(op: "LibraryOp") -> Any:
    """The :class:`tilefish._cublaslt.Form` of a library op: the type of its operands and of its result, and its layout."""
    from ._cublaslt import Form

    sig = op.signature
    return Form(sig.params[0].type.dtype.name, sig.returns[0].dtype.name, layout_of(op))  # type: ignore[union-attr]


def _problem_keys(op: "LibraryOp") -> "list[tuple[Any, ...]]":
    """The table key (:func:`table_key`: ``(m, n, k, bias)``, and the layout when it is not ``TN``) of every entry of a library op (one per
    bucket combination), sorted, without repeats."""
    m_expr, n_expr, k_expr = _mnk_exprs(op)
    free: "dict[str, Dim]" = {}
    for d in (m_expr, n_expr, k_expr):
        for f in d.free_dims():
            free.setdefault(f.name, f)
    dims = [free[name] for name in sorted(free)]
    keys = set()
    for combo in itertools.product(*(d.buckets for d in dims)) if dims else [()]:  # type: ignore[arg-type]
        mapping = dict(zip(dims, combo))
        m, n, k = (int(e.subst(mapping).folded().value) for e in (m_expr, n_expr, k_expr))
        keys.add(table_key(m, n, k, bias_of(op.impl, getattr(op, "epilogue", None)), layout_of(op)))  # type: ignore[arg-type]
    return sorted(keys)


def _canonical(op: "LibraryOp") -> "dict[str, Any]":
    """The canonical declaration the digests are taken over. The options a declaration has only when it is not the default (the posture
    ``"torch"``, the opt-in to split-K partial sums in the output type, imported algorithms: the rows of each table that the declaration uses, the
    layout ``NN``) are keys only then, so the digest of a declaration without them is the one tilefish 0.0.4 computed. (The types of the operands
    and of the result are in the signature, which is always there.)"""
    from .aot import _canonical_dim

    sig = op.signature
    params = []
    for p in sig.params:
        t = p.type
        assert t is not None
        params.append({"name": p.name, "dtype": t.dtype.name, "shape": [_canonical_dim(d) for d in t.shape]})  # type: ignore[union-attr]
    out = {
        "format": DIGEST_FORMAT,
        "op": op.name,
        "impl": op.impl,
        "acc": "f32",
        "workspace_bytes": op.workspace_bytes,
        "params": params,
        "returns": [{"dtype": r.dtype.name, "shape": [_canonical_dim(d) for d in r.shape]} for r in sig.returns],  # type: ignore[union-attr]
    }
    if getattr(op, "heuristic", "tilefish") != "tilefish":
        out["heuristic"] = op.heuristic
    if getattr(op, "allow_bf16_reduction", False):
        out["allow_bf16_reduction"] = True
    if getattr(op, "allow_f16_reduction", False):
        out["allow_f16_reduction"] = True
    if getattr(op, "allow_f32_reduction", False):
        out["allow_f32_reduction"] = True
    if layout_of(op) != "TN":
        out["layout"] = layout_of(op)  # b is [K, N]: with K = N the signature alone would not tell the layouts apart
    if getattr(op, "sm_count_target", None) is not None:
        out["sm_count_target"] = op.sm_count_target
    if op.impl == IMPL_MATMUL:  # (only on the new impl: the digests of gemm and gemm_bias are what they were)
        out["epilogue"] = op.epilogue
        out["beta"] = op.beta
        out["canary"] = op.canary
        if op.contract is not None:
            out["contract"] = op.contract
    tables = getattr(op, "algos", None)
    if tables:
        keys = _problem_keys(op)
        out["algos"] = sorted((t.canonical(keys) for t in tables), key=lambda c: (c["arch"], c["sm_count"], c["cublaslt_version"]))
    return out


def _digest(canonical: "dict[str, Any]") -> str:
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def declaration_digest(op: "LibraryOp", sm_count: "int | None" = None) -> str:
    """A sha256 over the canonical declaration of a library op: its name, implementation, workspace limit and signature (dimensions with
    their ``divisible_by`` and buckets), the same for every bucket and every device. With ``sm_count``: over that and the number of
    multiprocessors of the device an algorithm is pinned on, which is the ``declaration_digest`` of a library entry (:meth:`LibrarySpec.entry`)."""
    canonical = _canonical(op)
    return _digest(canonical if sm_count is None else {**canonical, "sm_count": sm_count})


# -- the specs: one per bucket combination ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LibrarySpec:
    """Everything about one pinned library operation that does not depend on the device or on the algorithm: what an entry of the
    manifest's ``library`` section says (see :meth:`entry`) before the build adds ``arch``, ``algo``, ``cublaslt_version`` and ``sm_count``.

    ``declaration_digest`` is the digest of the declaration alone (the same on every device); the ``declaration_digest`` of an entry also covers
    the SM count of the device it is pinned on (:meth:`pinned_digest`, over ``canonical``)."""

    stem: str
    op: str
    impl: str
    m: int
    n: int
    k: int
    workspace_bytes: int
    abi: "dict[str, Any]"
    buckets: "dict[str, int]"
    dims: "dict[str, dict[str, Any]]"
    preconditions: "tuple[Any, ...]"  # tilefish.aot.Precondition: pointer_alignment, static_size, bucket
    declaration_digest: str
    canonical: "dict[str, Any]"  # the canonical declaration (_canonical) the digests are taken over
    heuristic: str = "tilefish"  # the posture the heuristic is asked under (tilefish._cublaslt.HEURISTICS)
    allow_bf16_reduction: bool = False  # whether split-K may add its partial sums in bf16 (the opt-in of bf16 operands)
    algos: "tuple[AlgoTable, ...] | None" = None  # imported algorithms, one table per device
    dtype: str = "bf16"  # of a and b
    out_dtype: str = "bf16"  # of the result and of a bias
    layout: str = "TN"  # TN: b is [n, k]; NN: b is [k, n]; TT and NT: those, and a is [k, m]
    allow_f16_reduction: bool = False  # whether split-K may add its partial sums in the output type (the opt-in of f16 operands)
    sm_count_target: "int | None" = None  # the SM count the heuristic is asked for (CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET), None: the device's
    allow_f32_reduction: bool = False  # whether split-K may add its partial sums in the output (the opt-in of f32 operands)
    epilogue: "str | None" = None  # of a cublaslt.matmul (MATMUL_EPILOGUES); None for gemm and gemm_bias
    beta: int = 0  # of a cublaslt.matmul: 1 when it reads c (out = epilogue(acc + c))
    canary: bool = False  # whether export records a canary (a cublaslt.matmul's declaration: on by default)
    contract: "str | None" = None  # "measured-exact": a pin that is not is refused at the build; None: either is recorded

    @property
    def bias(self) -> bool:
        return bias_of(self.impl, self.epilogue)

    @property
    def matmul(self) -> bool:
        """Whether the spec is a ``cublaslt.matmul`` (an epilogue, ``beta``, a verification per pin and a canary)."""
        return self.impl == IMPL_MATMUL

    @property
    def call_epilogue(self) -> "str | None":
        """The ``epilogue=`` the binding is called with: the declaration's for a ``cublaslt.matmul``, ``None`` (by the bias) for the others."""
        return self.epilogue if self.matmul else None

    @property
    def form(self) -> Any:
        """The :class:`tilefish._cublaslt.Form`: the types and the layout."""
        from ._cublaslt import Form

        return Form(self.dtype, self.out_dtype, self.layout)

    @property
    def opt_ins(self) -> "dict[str, bool]":
        """The opt-ins as the binding takes them (``allow_bf16_reduction=``, ``allow_f16_reduction=``, ``allow_f32_reduction=``)."""
        return {
            "allow_bf16_reduction": self.allow_bf16_reduction,
            "allow_f16_reduction": self.allow_f16_reduction,
            "allow_f32_reduction": self.allow_f32_reduction,
        }

    @property
    def key(self) -> "tuple[Any, ...]":
        """The ``(m, n, k, bias)`` of the problem (with the layout when it is not ``TN``): what an :class:`AlgoTable` is indexed by."""
        return table_key(self.m, self.n, self.k, self.bias, self.layout)

    def pinned_digest(self, sm_count: int) -> str:
        """The ``declaration_digest`` of an entry of this spec pinned on a device of ``sm_count`` multiprocessors."""
        return _digest({**self.canonical, "sm_count": sm_count})

    def entry(self, sm_count: int) -> "dict[str, Any]":
        """The fields of the library entry that the declaration and the SM count of the device determine (a replay checks the manifest's entry
        against them); the build adds ``arch``, ``algo`` and ``cublaslt_version`` (and, for a ``cublaslt.matmul``, ``contract``, ``verification`` and
        ``canary``)."""
        dtypes = {
            "a": self.dtype,
            "b": self.dtype,
            **({"bias": self.out_dtype} if self.bias else {}),
            **({"c": self.out_dtype} if self.beta else {}),
            "out": self.out_dtype,
        }
        extra = {"beta": self.beta} if self.matmul else {}
        return {
            "stem": self.stem,
            "op": self.op,
            "impl": self.impl,
            "abi": json.loads(json.dumps(self.abi)),
            "buckets": dict(self.buckets),
            "dims": json.loads(json.dumps(self.dims)),
            "preconditions": [p.to_dict() for p in self.preconditions],
            "dtypes": dtypes,
            "compute_type": COMPUTE_TYPE,
            "scale_type": SCALE_TYPE,
            "transa": self.layout[0],
            "transb": self.layout[1],
            "m": self.m,
            "n": self.n,
            "k": self.k,
            **dict(zip(("lda", "ldb", "ldc", "ldd"), leading_dimensions(self.layout, self.m, self.n, self.k))),
            "epilogue": self.epilogue if self.matmul else ("bias" if self.bias else "none"),
            **extra,
            "workspace_bytes": self.workspace_bytes,
            "sm_count": sm_count,
            "declaration_digest": self.pinned_digest(sm_count),
            **self.options(),
        }

    def options(self) -> "dict[str, Any]":
        """The optional fields of an entry (:data:`OPTIONAL_ENTRY_FIELDS`) this declaration has: only those that are not the default."""
        out: "dict[str, Any]" = {}
        if self.heuristic != "tilefish":
            out["heuristic"] = self.heuristic
        if self.algos:
            out["algo_source"] = "imported"
        if self.allow_bf16_reduction:
            out["allow_bf16_reduction"] = True
        if self.allow_f16_reduction:
            out["allow_f16_reduction"] = True
        if self.allow_f32_reduction:
            out["allow_f32_reduction"] = True
        if self.sm_count_target is not None:
            out["sm_count_target"] = self.sm_count_target
        return out


def _shape_entry(d: DimExpr) -> "str | int":
    folded = d.folded()
    return folded.value if folded.is_static else str(folded)


def library_specs(op: "LibraryOp") -> "list[LibrarySpec]":
    """One :class:`LibrarySpec` per bucket combination of a library op that has an implementation (one for an unbucketed op)."""
    from .aot import Precondition, _dim_constraints

    if getattr(op, "impl", None) is None:
        raise TilefishSpecError(f"library op '{op.name}' has no implementation (impl=...): there is nothing to build")
    sig = op.signature
    tensors = [(p.name, p.type, "in") for p in sig.params] + [("out", sig.returns[0], "out")]
    m_expr, n_expr, k_expr = _mnk_exprs(op)
    form = form_of(op)
    free: "dict[str, Dim]" = {}
    for d in (m_expr, n_expr, k_expr):
        for f in d.free_dims():
            free.setdefault(f.name, f)
    dims = [free[name] for name in sorted(free)]
    canonical = _canonical(op)
    digest = _digest(canonical)
    specs = []
    for combo in itertools.product(*(d.buckets for d in dims)) if dims else [()]:  # type: ignore[arg-type]
        mapping = dict(zip(dims, combo))
        m, n, k = (int(e.subst(mapping).folded().value) for e in (m_expr, n_expr, k_expr))
        table: "dict[str, dict[str, Any]]" = {}

        def note(expr: DimExpr) -> None:
            for f in expr.free_dims():
                found = _dim_constraints(f)
                if table.setdefault(f.name, found) != found:
                    raise TilefishSpecError(
                        f"library op '{op.name}': the dimension {f.name!r} is declared twice with different constraints "
                        f"({table[f.name]} and {found})"
                    )

        params: "list[dict[str, Any]]" = []
        pres: "list[Any]" = []
        for i, (name, t, access) in enumerate(tensors):
            assert t is not None
            for d in t.shape:
                note(d)
            params.append(
                {
                    "index": i,
                    "name": name,
                    "kind": "pointer",
                    "dtype": t.dtype.name,  # type: ignore[union-attr]
                    "access": access,
                    "shape": [_shape_entry(d) for d in t.shape],
                }
            )
            pres.append(Precondition("pointer_alignment", arg=name, index=i, bytes=ALIGNMENT))
            for axis, d in enumerate(t.shape):
                if d.is_static:
                    pres.append(Precondition("static_size", arg=name, index=i, axis=axis, value=d.folded().value))
        for d, v in mapping.items():
            pres.append(Precondition("bucket", dim=d.name, value=v, buckets=d.buckets))
            table[d.name]["bucket"] = v
        stem = op.name + "".join(f".{d.name}{v}" for d, v in mapping.items())
        specs.append(
            LibrarySpec(
                stem=stem,
                op=op.name,
                impl=op.impl,
                m=m,
                n=n,
                k=k,
                workspace_bytes=op.workspace_bytes,
                abi={"params": params},
                buckets={d.name: v for d, v in mapping.items()},
                dims=dict(sorted(table.items())),
                preconditions=tuple(pres),
                declaration_digest=digest,
                canonical=canonical,
                heuristic=getattr(op, "heuristic", "tilefish"),
                allow_bf16_reduction=getattr(op, "allow_bf16_reduction", False),
                algos=getattr(op, "algos", None),
                dtype=form.dtype,
                out_dtype=form.out,
                layout=form.layout,
                allow_f16_reduction=getattr(op, "allow_f16_reduction", False),
                sm_count_target=getattr(op, "sm_count_target", None),
                allow_f32_reduction=getattr(op, "allow_f32_reduction", False),
                epilogue=getattr(op, "epilogue", None) if op.impl == IMPL_MATMUL else None,
                beta=getattr(op, "beta", 0) if op.impl == IMPL_MATMUL else 0,
                canary=bool(getattr(op, "canary", False)) if op.impl == IMPL_MATMUL else False,
                contract=getattr(op, "contract", None) if op.impl == IMPL_MATMUL else None,
            )
        )
    return specs


# -- the build: pin the algorithm ----------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BuiltLibrary:
    """One pinned library operation for one device: the spec, the algorithm cuBLASLt chose, the version that chose it, and the architecture and
    the number of multiprocessors (SMs) of the device it chose it on (a MIG slice has fewer than the whole GPU, and an algorithm that partitions its
    work by the SM count, a stream-K one, is chosen for that count)."""

    spec: LibrarySpec
    arch: str
    algo: str  # 128 lowercase hex digits: the 64 bytes of the cublasLtMatmulAlgo_t
    cublaslt_version: int
    sm_count: int
    # of a cublaslt.matmul: what the verification of the pin found (CONTRACTS), its record, and the canary export made ({"seed", "out_digest"})
    contract: "str | None" = None
    verification: "dict[str, Any] | None" = None
    canary: "dict[str, Any] | None" = None

    @property
    def stem(self) -> str:
        return self.spec.stem

    @property
    def workspace_bytes(self) -> int:
        return self.spec.workspace_bytes

    def entry(self) -> "dict[str, Any]":
        """The ``library`` entry of the manifest."""
        out = self.spec.entry(self.sm_count)
        out["arch"] = self.arch
        out["algo"] = self.algo
        out["cublaslt_version"] = self.cublaslt_version
        if self.spec.matmul:
            assert self.contract is not None and self.verification is not None
            out["contract"] = self.contract
            out["verification"] = dict(self.verification)
            if self.canary is not None:
                out["canary"] = dict(self.canary)
        return out


def build_library(spec: LibrarySpec, *, arch: "int | str", device_index: "int | None" = None, canary: bool = False) -> BuiltLibrary:
    """Pin the algorithm of ``spec`` for ``arch``: ask cuBLASLt's heuristic, under the spec's posture and workspace limit, for its best algorithm,
    or, for a declaration with ``algos=``, take the algorithm its table for this device gives (refused when it has none for this device). Either
    way the algorithm must pass the checks a run makes (``tilefish._cublaslt.check_algo``) before it is pinned.

    The heuristic is asked on a device of that architecture (a pin is per architecture), so the device must be one.

    A ``cublaslt.matmul`` is pinned for its epilogue, and its pin is then verified against contract (ii) (:func:`tilefish._matmul.verify`): the
    result, ``"measured-exact"`` or ``"binary-defined"``, is part of the pin (a declaration with ``contract="measured-exact"`` refuses the other).
    With ``canary`` (what ``export`` does) and a declaration that has one (the default), its canary is run and recorded too.
    """
    from . import _cublaslt
    from ._build import parse_arch
    from .errors import TilefishBuildError

    cap = parse_arch(arch)
    index = _device_index(device_index)
    have = _capability(index)
    if have != cap:
        raise TilefishBuildError(
            f"cannot pin '{spec.stem}' for sm_{cap}: cuBLASLt chooses an algorithm for the device it runs on, and device {index} is sm_{have}"
        )
    version, sms = _cublaslt.version(), sm_count(index)
    if spec.sm_count_target is not None and spec.sm_count_target > sms:
        raise TilefishBuildError(
            f"cannot pin '{spec.stem}' on sm_{cap} with {sms} SMs: its declaration targets {spec.sm_count_target} SMs (sm_count_target=), more than "
            "the device has"
        )
    if spec.algos:
        algo = imported_algo(spec, f"sm_{cap}", sms, version, index)
        if algo is None:
            have_tables = ", ".join(f"{t.arch} with {t.sm_count} SMs and cuBLASLt {t.cublaslt_version}" for t in spec.algos)
            raise TilefishBuildError(
                f"cannot pin '{spec.stem}' on sm_{cap} with {sms} SMs and cuBLASLt {version}: its declaration imports its algorithms (algos=), "
                f"and has a table only for {have_tables}; give one for this device, or declare it without algos= to ask the heuristic"
            )
    else:
        algo = _cublaslt.pin_algo(
            spec.m, spec.n, spec.k, spec.bias, workspace_bytes=spec.workspace_bytes, device_index=index, heuristic=spec.heuristic,
            form=spec.form, sm_count_target=spec.sm_count_target, **spec.opt_ins, **_epilogue_kw(spec),
        ).algo_hex  # fmt: skip
    try:
        _cublaslt.check_algo(
            spec.m, spec.n, spec.k, spec.bias, algo, workspace_bytes=spec.workspace_bytes, device_index=index, form=spec.form,
            sm_count_target=spec.sm_count_target, **spec.opt_ins, **_epilogue_kw(spec, beta=True),
        )  # fmt: skip
    except TilefishRuntimeError as e:
        source = "the imported" if spec.algos else "the heuristic's"
        raise TilefishBuildError(f"cannot pin '{spec.stem}' on sm_{cap}: {source} algorithm is refused: {e}") from e
    if not spec.matmul:
        return BuiltLibrary(spec, f"sm_{cap}", algo, version, sms)
    from . import _matmul

    contract, verification = _matmul.verify(spec, algo, index)
    if spec.contract == "measured-exact" and contract != "measured-exact":
        raise TilefishBuildError(
            f"cannot pin '{spec.stem}' on sm_{cap}: the declaration asks for contract='measured-exact', and the pinned algorithm "
            f"{_matmul.algo_text(algo)} with cuBLASLt {version} computes {spec.epilogue}{' with beta=1' if spec.beta else ''} as no rounding of "
            f"{verification['formula']} does: {_matmul.verification_text(verification)}"
        )
    made = {"seed": _matmul.CANARY_SEED, "out_digest": _matmul.canary_digest(spec, algo, _matmul.CANARY_SEED, index)} if canary and spec.canary else None
    return BuiltLibrary(spec, f"sm_{cap}", algo, version, sms, contract, verification, made)


def _epilogue_kw(spec: LibrarySpec, beta: bool = False) -> "dict[str, Any]":
    """The binding's ``epilogue=`` (and ``beta=``) of a ``cublaslt.matmul`` spec; nothing for ``gemm`` and ``gemm_bias`` (their calls are 0.0.4's)."""
    if not spec.matmul:
        return {}
    return {"epilogue": spec.epilogue, "beta": spec.beta} if beta else {"epilogue": spec.epilogue}


def imported_algo(spec: LibrarySpec, arch: str, sms: int, version: int, device_index: "int | None") -> "str | None":
    """The pin (128 hex digits) the tables of ``spec`` give its problem on a device of ``arch`` with ``sms`` SMs and cuBLASLt ``version`` (a
    configuration is encoded by the library: ``tilefish._cublaslt.encode_algo``), or ``None`` when no table is for that device."""
    from . import _cublaslt

    assert spec.algos
    table = table_for(spec.algos, arch, sms, version)
    if table is None:
        return None
    algo = table.algos[spec.key]  # every key of the declaration is in every table (check_coverage)
    return algo if isinstance(algo, str) else _cublaslt.encode_algo(algo, device_index=device_index, form=spec.form)


def _device_index(index: "int | None") -> int:
    import torch

    return torch.cuda.current_device() if index is None else int(index)


def _capability(index: int) -> int:
    import torch

    major, minor = torch.cuda.get_device_capability(index)
    return major * 10 + minor


def sm_count(index: int) -> int:
    """The number of multiprocessors of device ``index`` (``CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT``; a MIG slice has its own count)."""
    import torch

    return int(torch.cuda.get_device_properties(index).multi_processor_count)


def built_from_manifest(manifest: Any, spec: LibrarySpec, arch: str, sm_count: int, device_index: "int | None" = None) -> BuiltLibrary:
    """The pinned library operation of ``spec`` for a device of ``arch`` with ``sm_count`` multiprocessors, from a manifest: the entry must have
    been pinned on a device of that SM count, be what the declaration asks for, and cuBLASLt must be the version that chose its algorithm.

    Raises :class:`~tilefish.errors.TilefishManifestError` for a missing entry, an entry pinned on a device of another SM count, an entry that
    differs from the spec in any field (the first one is named), and a cuBLASLt of another version than the entry's.
    """
    from . import _cublaslt
    from .aot import _first_difference, _show

    entry = manifest.library_entry(spec.stem, arch)
    if entry is None:
        have = sorted(a for (s, a) in ((e["stem"], e["arch"]) for e in manifest.library_entries) if s == spec.stem)
        raise TilefishManifestError(
            f"the manifest in {str(manifest.directory)!r} has no library entry for '{spec.stem}' on {arch}"
            + (f" (it has {', '.join(have)})" if have else f" (it has {sorted({e['stem'] for e in manifest.library_entries})})")
        )
    pinned_on = entry["sm_count"]
    if pinned_on != sm_count:
        raise TilefishManifestError(
            f"the manifest pinned the cuBLASLt algorithm of '{spec.stem}' ({arch}) on a device with {pinned_on} SMs, but this device has {sm_count}: "
            "an algorithm is chosen for the number of multiprocessors it runs on (a MIG slice has fewer than the whole GPU), so the pin is refused; "
            "re-export the manifest on this device (or on one with the same SM count)"
        )
    if entry.get("sm_count_target") != spec.sm_count_target:
        def chosen(t: "int | None") -> str:
            return "the whole device" if t is None else f"{t} SMs (sm_count_target={t})"

        raise TilefishManifestError(
            f"the manifest pinned the cuBLASLt algorithm of '{spec.stem}' ({arch}) for {chosen(entry.get('sm_count_target'))}, but the declaration "
            f"asks for {chosen(spec.sm_count_target)}: re-export the manifest from this declaration"
        )
    expected = json.loads(json.dumps(spec.entry(sm_count)))
    got = {k: entry.get(k) for k in expected}
    found = _first_difference(expected, got, "")
    if found is not None:
        path, want, have_value = found
        raise TilefishManifestError(
            f"the library entry for '{spec.stem}' ({arch}) is not what the declaration asks for: first difference at {path}: the manifest "
            f"has {_show(have_value)}, the declaration gives {_show(want)}"
        )
    pinned_by = entry["cublaslt_version"]
    here = _cublaslt.version()
    if here != pinned_by:
        raise TilefishManifestError(
            f"the manifest pinned the cuBLASLt algorithm of '{spec.stem}' ({arch}) with cuBLASLt {pinned_by}, but this process has cuBLASLt "
            f"{here}: an algorithm id names other kernels in another version, so the pin is refused (export the manifest again with this "
            "cuBLASLt, or run with the version that made it)"
        )
    if spec.algos:  # the digest covers the tables; this is the algorithm itself, against the table of this device
        want = imported_algo(spec, arch, sm_count, pinned_by, device_index)
        if want != entry["algo"]:
            raise TilefishManifestError(
                f"the library entry for '{spec.stem}' ({arch}) has the algorithm {entry['algo'][:16]}..., but the declaration imports "
                + (f"{want[:16]}... for this device" if want else f"none for {arch} with {sm_count} SMs and cuBLASLt {pinned_by}")
                + ": it is not what the declaration asks for"
            )
    if not spec.matmul:
        return BuiltLibrary(spec, arch, entry["algo"], pinned_by, pinned_on)
    if spec.contract == "measured-exact" and entry["contract"] != "measured-exact":
        raise TilefishManifestError(
            f"the library entry for '{spec.stem}' ({arch}) is {entry['contract']}, and the declaration asks for contract='measured-exact': its pin was "
            "not verified equal to its formula, so it is refused (export it again from this declaration, which refuses such a pin)"
        )
    if spec.canary and "canary" not in entry:
        raise TilefishManifestError(
            f"the library entry for '{spec.stem}' ({arch}) has no canary, and the declaration has one (canary=True, the default): the pin cannot be "
            "told from another build of this cuBLASLt version, so it is refused (export it again, or declare canary=False)"
        )
    built = BuiltLibrary(spec, arch, entry["algo"], pinned_by, pinned_on, entry["contract"], entry["verification"], entry.get("canary"))
    if built.canary is not None:
        from . import _matmul

        _matmul.check_canary(built, _device_index(device_index))
    return built


# -- reading an entry back ---------------------------------------------------------------------------------------------------------


def _bad(where: str, message: str) -> Any:
    raise TilefishManifestError(f"library entry {where}: {message}")


def _eval_static(text: str, buckets: "dict[str, int]", where: str) -> int:
    """The value of the dimension expression ``text`` when the bucketed dimensions have their bucket values."""
    p = sv._ExprParser(text, where)
    value = p.expr()
    p.done()
    if value is None:
        return _bad(where, f"{text!r} is not an affine expression")
    value = value.subst({Dim(name): v for name, v in buckets.items()})
    if not value.is_static:
        return _bad(where, f"{text!r} mentions a dimension that has no bucket")
    return value.value


def _validate_matmul_fields(entry: "dict[str, Any]", where: str) -> None:
    """The fields of a ``cublaslt.matmul`` entry that a GEMM's has not (section 9.5): ``epilogue``, ``beta``, ``contract``, ``verification`` and the
    optional ``canary``, each exactly typed, and the redundant ones agreeing (the formula is the one of the epilogue and ``beta``; the contract is
    ``measured-exact`` exactly when no element differed)."""
    epilogue = entry["epilogue"]
    if epilogue not in MATMUL_EPILOGUES:
        _bad(where, f"the epilogue of {IMPL_MATMUL} must be one of {list(MATMUL_EPILOGUES)}, got {epilogue!r}")
    beta = entry["beta"]
    if not sv.is_int(beta) or beta not in (0, 1):
        _bad(where, f"beta must be the int 0 or 1, got {beta!r}")
    contract = entry["contract"]
    if contract not in CONTRACTS:
        _bad(where, f"contract must be one of {list(CONTRACTS)}, got {contract!r}")
    v = entry["verification"]
    if not isinstance(v, dict) or sorted(v) != sorted(VERIFICATION_KEYS):
        _bad(where, f"verification must be an object of exactly {list(VERIFICATION_KEYS)}, got {v!r}")
    want = formula(epilogue, beta)
    if v["formula"] != want:
        _bad(where, f"verification.formula must be {want!r} (the arithmetic of the epilogue {epilogue!r} with beta {beta}), got {v['formula']!r}")
    if v["accumulator"] not in ACCUMULATORS:
        _bad(where, f"verification.accumulator must be one of {list(ACCUMULATORS)}, got {v['accumulator']!r}")
    for key in ("operand_sets", "elements"):
        if not sv.is_int(v[key]) or v[key] < 1:
            _bad(where, f"verification.{key} must be a positive int, got {v[key]!r}")
    for key in ("differ", "max_ulp"):
        if not sv.is_int(v[key]) or v[key] < 0:
            _bad(where, f"verification.{key} must be an int of at least 0, got {v[key]!r}")
    if v["differ"] > v["elements"]:
        _bad(where, f"verification.differ ({v['differ']}) is more than verification.elements ({v['elements']})")
    m, n = entry["m"], entry["n"]
    if sv.is_int(m) and sv.is_int(n) and v["elements"] != v["operand_sets"] * m * n:
        _bad(where, f"verification.elements must be operand_sets * m * n = {v['operand_sets'] * m * n}, got {v['elements']}")
    if (contract == "measured-exact") != (v["differ"] == 0):
        _bad(where, f"contract is {contract!r}, but the verification found {v['differ']} elements that differ (measured-exact is exactly none)")
    if v["differ"] == 0 and v["max_ulp"] != 0:
        _bad(where, f"verification.max_ulp is {v['max_ulp']}, but no element differs")
    if "canary" in entry:
        c = entry["canary"]
        if (
            not isinstance(c, dict)
            or sorted(c) != sorted(CANARY_KEYS)
            or not sv.is_int(c["seed"])
            or not 0 <= c["seed"] < 1 << 32
            or not isinstance(c["out_digest"], str)
            or not _HEX32.match(c["out_digest"])
        ):
            _bad(where, f"canary must be {{\"seed\": an int from 0 to 2**32 - 1, \"out_digest\": 32 lowercase hex digits}}, got {c!r}")


def validate_entry(entry: Any, index: int = 0) -> None:
    """Strict reading of one entry of the manifest's ``library`` section (raises :class:`~tilefish.errors.TilefishManifestError`).

    Every field must be present and exactly typed, and the redundant ones must agree: the dtypes are a pair of operand types (``bf16``, ``f16``
    or ``f32``) with a result of their type or ``f32`` (and a bias of the result's type); ``transa`` and ``transb`` say the layout (``transa`` ``T``:
    ``b`` is ``[n, k]`` and ``lda = k``, ``N``: ``b`` is ``[k, n]`` and ``lda = n``; ``transb`` ``N``: ``a`` is ``[m, k]`` and ``ldb = k``, ``T``:
    ``a`` is ``[k, m]`` and ``ldb = m``); the other leading dimensions and the shapes of the operands follow from ``impl``, ``m``, ``n`` and ``k``;
    the opt-in, if any, is the one of the operands' type.

    An entry of ``impl`` ``"cudnn.conv_fwd"`` is a cuDNN convolution, read by :func:`tilefish._conv.validate_entry`.
    """
    where = f"#{index}"
    if isinstance(entry, dict) and entry.get("impl") == "cudnn.conv_fwd":
        from . import _conv

        return _conv.validate_entry(entry, index)
    if not isinstance(entry, dict):
        _bad(where, f"is not an object: {entry!r}")
    where = f"#{index} ({entry.get('stem')!r})"
    matmul = entry.get("impl") == IMPL_MATMUL
    keys = MATMUL_ENTRY_KEYS if matmul else ENTRY_KEYS
    optional = [*MATMUL_OPTIONAL_FIELDS] if matmul else [*OPTIONAL_ENTRY_FIELDS, *OPTIONAL_COUNT_FIELDS]
    missing = [k for k in keys if k not in entry]
    unknown = sorted(str(k) for k in set(entry) - set(keys) - set(optional))
    if missing or unknown:
        _bad(
            where,
            f"must have exactly the fields {keys} and may have {optional}"
            + (f"; missing {missing}" if missing else "")
            + (f"; unknown {unknown}" if unknown else ""),
        )
    for key, only in OPTIONAL_ENTRY_FIELDS.items():  # present only when not the default, and then with its one value (true is not 1)
        if key in entry and (type(entry[key]) is not type(only) or entry[key] != only):
            _bad(where, f"{key} is absent (the default) or {json.dumps(only)}, got {entry[key]!r}")
    if matmul:
        _validate_matmul_fields(entry, where)
    try:
        stem = entry["stem"]
        if not isinstance(stem, str) or not sv._STEM.match(stem):
            _bad(where, f"stem must look like 'gemm' or 'gemm.M64', got {stem!r}")
        sv._ident(entry["op"], "op")
        if not isinstance(entry["arch"], str) or not _ARCH.match(entry["arch"]):
            _bad(where, f"arch must look like 'sm_103', got {entry['arch']!r}")
        impl = entry["impl"]
        if impl not in CUBLASLT_IMPLS:
            _bad(where, f"impl must be one of {sorted([*CUBLASLT_IMPLS, 'cudnn.conv_fwd'])}, got {impl!r}")
        bias = bias_of(impl, entry["epilogue"])
        with_c = matmul and entry["beta"] == 1
        for key in ("m", "n", "k", "lda", "ldb", "ldc", "ldd"):
            v = entry[key]
            if not sv.is_int(v) or not 1 <= v <= sv.MAX_ELEMENTS:
                _bad(where, f"{key} must be an int from 1 to {sv.MAX_ELEMENTS}, got {v!r}")
        m, n, k = entry["m"], entry["n"], entry["k"]
        layout = f"{entry['transa']}{entry['transb']}" if isinstance(entry["transa"], str) and isinstance(entry["transb"], str) else None
        if layout not in LAYOUTS:
            _bad(
                where,
                f"transa and transb must each be 'T' or 'N' (transa 'T': b is [n, k], 'N': b is [k, n]; transb 'N': a is [m, k], 'T': a is [k, m]), "
                f"got {entry['transa']!r} and {entry['transb']!r}",
            )
        if (entry["lda"], entry["ldb"], entry["ldc"], entry["ldd"]) != leading_dimensions(layout, m, n, k):
            _bad(where, leading_dimensions_rule(layout))
        if entry["compute_type"] != COMPUTE_TYPE or entry["scale_type"] != SCALE_TYPE:
            _bad(where, f"compute_type and scale_type must be {COMPUTE_TYPE!r} and {SCALE_TYPE!r}")
        if not matmul and entry["epilogue"] != ("bias" if bias else "none"):
            _bad(where, f"the epilogue of {impl} must be {'bias' if bias else 'none'!r}, got {entry['epilogue']!r}")
        dtypes = entry["dtypes"]
        dtype = dtypes.get("a") if isinstance(dtypes, dict) else None
        out_dtype = dtypes.get("out") if isinstance(dtypes, dict) else None
        if matmul and (dtype not in MATMUL_DTYPES or out_dtype != dtype):
            dtype, out_dtype = "bf16", "bf16"  # the message names the default form
        if dtype not in OPERAND_DTYPES or out_dtype not in (dtype, "f32"):  # (f32 operands: dtype and "f32" are one type)
            dtype, out_dtype = "bf16", "bf16"  # the message names the default form
        want_dtypes = {"a": dtype, "b": dtype, **({"bias": out_dtype} if bias else {}), **({"c": out_dtype} if with_c else {}), "out": out_dtype}
        if dtypes != want_dtypes and matmul:
            _bad(where, f"dtypes must be {want_dtypes}, got {dtypes!r} (a and b are both bf16 or both f16, the result, a bias and c are of their type)")
        if dtypes != want_dtypes:
            _bad(
                where,
                f"dtypes must be {want_dtypes}, got {dtypes!r} (a and b are both bf16, both f16 or both f32, the result is of their type or f32, a "
                "bias of the result's type)",
            )
        for flag in OPT_INS:
            if flag in entry and flag != f"allow_{dtype}_reduction":
                _bad(where, f"{flag} is the opt-in of {flag.split('_')[1]} operands, and the operands of this entry are {dtype}")
        ws = entry["workspace_bytes"]
        if not sv.is_int(ws) or not 1 <= ws <= MAX_WORKSPACE_BYTES:
            _bad(where, f"workspace_bytes must be an int from 1 to 2**40, got {ws!r}")
        if not isinstance(entry["algo"], str) or not _ALGO.match(entry["algo"]):
            _bad(where, f"algo must be 128 lowercase hex digits (the 64 bytes of a cublasLtMatmulAlgo_t), got {entry['algo']!r}")
        if not sv.is_int(entry["cublaslt_version"]) or entry["cublaslt_version"] < 1:
            _bad(where, f"cublaslt_version must be a positive int, got {entry['cublaslt_version']!r}")
        if not sv.is_int(entry["sm_count"]) or entry["sm_count"] < 1:
            _bad(where, f"sm_count must be a positive int (the multiprocessors of the device the algorithm was pinned on), got {entry['sm_count']!r}")
        if "sm_count_target" in entry:  # absent: the heuristic was asked for the device's count; present: a count from 8 to the device's
            target = entry["sm_count_target"]
            if not sv.is_int(target) or not 8 <= target <= entry["sm_count"]:
                _bad(where, f"sm_count_target is absent or an int from 8 to sm_count ({entry['sm_count']}), got {target!r}")
        if not (isinstance(entry["declaration_digest"], str) and sv._DIGEST.match(entry["declaration_digest"])):
            _bad(where, f"declaration_digest must be 64 lowercase hex digits (a sha256), got {entry['declaration_digest']!r}")
        buckets = entry["buckets"]
        if not isinstance(buckets, dict):
            _bad(where, "buckets must be an object")
        for name, value in buckets.items():
            sv._ident(name, "bucket dimension name")
            sv._int(value, f"bucket value of {name}", 1)
        # the operands: a [M, K] (TN, NN) or [K, M] (TT, NT), b [N, K] (TN, TT) or [K, N] (NN, NT), [bias [N]], out [M, N], in this order, each of
        # its type in dtypes
        abi = entry["abi"]
        sv._keys(abi, "abi", ["params"])
        params = abi["params"]
        names = ["a", "b"] + (["bias"] if bias else []) + (["c"] if with_c else []) + ["out"]
        expected_shapes = [*operand_shapes(layout, m, n, k)] + ([(n,)] if bias else []) + ([(m, n)] if with_c else []) + [(m, n)]
        expected_dtypes = [dtype, dtype] + ([out_dtype] if bias else []) + ([out_dtype] if with_c else []) + [out_dtype]
        if not isinstance(params, list) or len(params) != len(names):
            _bad(where, f"abi.params must list {len(names)} pointers ({', '.join(names)} in this order)")
        seen: "set[str]" = set()
        for i, (p, shape, pdtype) in enumerate(zip(params, expected_shapes, expected_dtypes)):
            sv._keys(p, f"abi.params[{i}]", ["index", "name", "kind", "dtype", "access", "shape"])
            sv._ident(p["name"], f"abi.params[{i}].name")
            if not sv.is_int(p["index"]) or p["index"] != i or p["kind"] != "pointer" or p["dtype"] != pdtype or p["access"] != ("out" if i == len(params) - 1 else "in"):  # 0.0 and true are not 0 and 1
                _bad(where, f"abi.params[{i}] must be the {pdtype} pointer #{i} with access {'out' if i == len(params) - 1 else 'in'}")
            if p["name"] in seen:
                _bad(where, f"the name {p['name']!r} is used twice")
            seen.add(p["name"])
            if not isinstance(p["shape"], list) or len(p["shape"]) != len(shape):
                _bad(where, f"abi.params[{i}].shape must have {len(shape)} entries")
            for axis, (entry_dim, size) in enumerate(zip(p["shape"], shape)):
                if sv.is_int(entry_dim):
                    value = entry_dim
                elif isinstance(entry_dim, str):
                    value = _eval_static(entry_dim, buckets, f"{where} abi.params[{i}].shape[{axis}]")
                else:
                    return _bad(where, f"abi.params[{i}].shape[{axis}] must be an int or a dimension expression, got {entry_dim!r}")
                if value != size:
                    _bad(where, f"abi.params[{i}].shape[{axis}] is {entry_dim!r} = {value}, but m, n and k say {size}")
        # the preconditions and the table of dimensions: as in a kernel entry
        pres = entry["preconditions"]
        if not isinstance(pres, list):
            _bad(where, "preconditions must be a list")
        aligned: "set[int]" = set()
        bucket_pres: "dict[str, int]" = {}
        for pre in pres:
            sv.validate_precondition(pre)
            kind = pre["kind"]
            if kind == "bucket":
                if pre["dim"] in bucket_pres or buckets.get(pre["dim"]) != pre["value"] or pre["value"] not in pre["buckets"]:
                    _bad(where, f"the bucket precondition for {pre['dim']} disagrees with buckets")
                bucket_pres[pre["dim"]] = pre["value"]
            elif kind == "pointer_alignment":
                if pre["index"] in aligned or pre["bytes"] != ALIGNMENT or not 0 <= pre["index"] < len(params) or params[pre["index"]]["name"] != pre["arg"]:
                    _bad(where, f"pointer_alignment must be {ALIGNMENT} bytes, once per pointer, on the pointer it names")
                aligned.add(pre["index"])
            elif kind == "static_size":
                p = params[pre["index"]] if 0 <= pre["index"] < len(params) else None
                if p is None or p["name"] != pre["arg"] or pre["axis"] >= len(p["shape"]) or p["shape"][pre["axis"]] != pre["value"]:
                    _bad(where, "static_size does not match the shape of the pointer it names")
            else:
                _bad(where, f"a {kind} precondition does not belong in a library entry")
        if aligned != set(range(len(params))):
            _bad(where, "every pointer needs exactly one pointer_alignment precondition")
        if bucket_pres.keys() != buckets.keys():
            _bad(where, "every bucket needs exactly one bucket precondition")
        table = entry["dims"]
        if not isinstance(table, dict):
            _bad(where, "dims must be an object")
        for name, c in table.items():
            sv._ident(name, "dims name")
            sv._keys(c, f"dims.{name}", [], ["divisible_by", "buckets", "bucket"])
            if "divisible_by" in c:
                sv._int(c["divisible_by"], f"dims.{name}.divisible_by", 1)
            if "buckets" in c:
                b = c["buckets"]
                if not isinstance(b, list) or not b or any(not sv.is_int(x) or x < 1 for x in b) or b != sorted(set(b)):
                    _bad(where, f"dims.{name}.buckets must be a sorted list of distinct positive ints")
            if "bucket" in c and ("buckets" not in c or not sv.is_int(c["bucket"]) or c["bucket"] not in c["buckets"] or buckets.get(name) != c["bucket"]):
                _bad(where, f"dims.{name}.bucket must be one of its buckets and the bucket of this entry")
        for name in buckets:
            if table.get(name, {}).get("bucket") != buckets[name]:
                _bad(where, f"every bucketed dimension needs its bucket in dims: {name} has none")
        if stem != entry["op"] + "".join(f".{k_}{v}" for k_, v in buckets.items()):
            _bad(where, f"stem {stem!r} does not match op {entry['op']!r} and buckets {buckets}")
    except TilefishSpecError as e:  # the shared validators raise spec errors; here they are about a manifest
        raise TilefishManifestError(f"library entry {where}: {str(e).removeprefix('malformed compile spec: ')}") from None
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError, RecursionError) as e:  # an entry is untrusted input
        raise TilefishManifestError(f"library entry {where} is malformed: {type(e).__name__}: {e}") from None


# -- running -----------------------------------------------------------------------------------------------------------------------


def run_gemm(spec: LibrarySpec, algo: str, tensors: "dict[str, Any]", out: Any, stream: int, device_index: int) -> None:
    """Launch ``cublasLtMatmul`` with the pinned ``algo`` on ``stream``: ``tensors`` are ``a``, ``b`` (and ``bias``, and ``c`` of a ``cublaslt.matmul``
    with ``beta=1``) by the op's parameter names in order, ``out`` the result tensor; all are contiguous, 16-byte aligned and have the shapes of
    ``spec`` (the caller checked)."""
    a, b, *rest = tensors.values()
    bias = int(rest[0].data_ptr()) if spec.bias else None
    c = int(rest[-1].data_ptr()) if spec.beta else None
    run_gemm_ptrs(spec, algo, int(a.data_ptr()), int(b.data_ptr()), bias, int(out.data_ptr()), stream, device_index, c_ptr=c)


def run_gemm_ptrs(
    spec: LibrarySpec,
    algo: str,
    a_ptr: int,
    b_ptr: int,
    bias_ptr: "int | None",
    out_ptr: int,
    stream: int,
    device_index: int,
    workspace_ptr: "int | None" = None,
    c_ptr: "int | None" = None,
) -> None:
    """:func:`run_gemm` on device addresses. With ``workspace_ptr`` (``spec.workspace_bytes`` bytes, 256-byte aligned: a buffer of a launch plan's
    arena) the workspace is that memory and the call allocates nothing; without it the workspace comes from torch's caching allocator. ``c_ptr`` is
    the ``c`` of a ``cublaslt.matmul`` with ``beta=1`` (and only of one)."""
    from . import _cublaslt

    if (c_ptr is not None) != bool(spec.beta):
        raise TilefishRuntimeError(f"'{spec.stem}': {'a c is given to a GEMM without beta=1' if c_ptr is not None else 'its c (beta=1) is not given'}")
    _cublaslt.matmul(
        a_ptr,
        b_ptr,
        out_ptr,
        bias_ptr,
        spec.m,
        spec.n,
        spec.k,
        algo,
        workspace_bytes=spec.workspace_bytes,
        stream=stream,
        device_index=device_index,
        workspace_ptr=workspace_ptr,
        form=spec.form,
        sm_count_target=spec.sm_count_target,
        **spec.opt_ins,
        **({"epilogue": spec.epilogue, "c_ptr": c_ptr} if spec.matmul else {}),
    )


_TORCH_DTYPES = {"bf16": "bfloat16", "f16": "float16", "f32": "float32"}


def reference_gemm(tensors: "dict[str, Any]", layout: str = "TN", out_dtype: "str | None" = None) -> Any:
    """The result in torch, as the semantics say: an f32 matmul of the operands (``b`` transposed for the layouts ``TN`` and ``TT``, ``a`` for ``TT``
    and ``NT``), the bias added in f32, one rounding to ``out_dtype`` (the operands' type when ``None``)."""
    import torch

    a, b, *rest = tensors.values()
    lhs = a.to(torch.float32) if layout[1] == "N" else a.to(torch.float32).t()
    acc = lhs @ (b.to(torch.float32).t() if layout[0] == "T" else b.to(torch.float32))
    if rest:
        acc = acc + rest[0].to(torch.float32)
    return acc.to(a.dtype if out_dtype is None else getattr(torch, _TORCH_DTYPES[out_dtype]))


def reference_matmul(tensors: "dict[str, Any]", layout: str, epilogue: str, beta: int) -> Any:
    """The result of a ``cublaslt.matmul`` in torch, as its formula says (:func:`formula`): an f32 matmul of the operands, ``c`` (with ``beta=1``)
    and then the bias added in f32, ReLU, one rounding to the operands' type."""
    import torch

    a, b, *rest = tensors.values()
    lhs = a.to(torch.float32) if layout[1] == "N" else a.to(torch.float32).t()
    acc = lhs @ (b.to(torch.float32).t() if layout[0] == "T" else b.to(torch.float32))
    if beta:
        acc = acc + rest[-1].to(torch.float32)
    if epilogue in MATMUL_WITH_BIAS:
        acc = acc + rest[0].to(torch.float32)
    if epilogue.startswith("relu"):
        acc = torch.relu(acc)
    return acc.to(a.dtype)


def reference_of(op: "LibraryOp", tensors: "dict[str, Any]") -> Any:
    """:func:`reference_gemm` of a library op: its layout and the type of its result (:func:`reference_matmul` of a ``cublaslt.matmul``)."""
    if op.impl == IMPL_MATMUL:
        return reference_matmul(tensors, layout_of(op), op.epilogue, op.beta)  # type: ignore[attr-defined]
    return reference_gemm(tensors, layout_of(op), op.signature.returns[0].dtype.name)  # type: ignore[union-attr]


__all__ = [
    "AlgoTable",
    "BuiltLibrary",
    "IMPLS",
    "LibrarySpec",
    "build_library",
    "built_from_manifest",
    "check_declaration",
    "declaration_digest",
    "library_specs",
    "validate_entry",
]
