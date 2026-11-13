"""Library operations with a pinned implementation: a cuBLASLt GEMM.

``@tilefish.library("gemm", acc=f32, impl="cublaslt.gemm")`` declares an operation by its signature and says which library
routine runs it. For ``impl="cublaslt.gemm"`` the signature is ``(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16]) -> Tensor[M, N, bf16]``
and for ``"cublaslt.gemm_bias"`` it has a third parameter ``bias: Tensor[N, bf16]``. Semantics, in tilefish's row-major terms:

    out[M, N] = a[M, K] @ b[N, K]^T        (+ bias[N])

with an f32 accumulator and **one** rounding, to bf16, at the end; the bias goes through ``CUBLASLT_EPILOGUE_BIAS``, so it is added in
f32 to the accumulator before that rounding. Every operand is contiguous with a 16-byte aligned pointer (a strided operand is not
supported in this version: pass ``.contiguous()``); ``K`` and ``N`` are multiples of 8, so every row starts on 16 bytes.

**A binary is a pinned algorithm.** For every ``(M, N, K, bias)`` (``M``, ``N`` and ``K`` are static or bucketed: one entry per bucket
combination, like a kernel) the heuristic of cuBLASLt is asked, with a fixed workspace limit (``workspace_bytes``, by default
8 519 680, torch's default ``CUBLAS_WORKSPACE_CONFIG``), for its best algorithm; result 0 is the pin, the 64 bytes of its
``cublasLtMatmulAlgo_t`` written as 128 lowercase hex digits. A manifest records it in its ``library`` section (see
``docs/manifest-schema-3.md``) with the cuBLASLt version that chose it, and ``jit`` pins in memory the same way, so ``jit`` and
``replay`` run the same algorithm by construction. ``replay`` refuses a cuBLASLt of another version (an algorithm id means other
kernels there) and an entry that is not what the declaration asks for.

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
from .errors import TilefishDeclarationError, TilefishManifestError, TilefishSpecError

if TYPE_CHECKING:
    from .decorators import LibraryOp
    from .signature import Signature

IMPL_GEMM = "cublaslt.gemm"
IMPL_GEMM_BIAS = "cublaslt.gemm_bias"
IMPLS = {IMPL_GEMM: False, IMPL_GEMM_BIAS: True}  # impl -> whether it has a bias
DEFAULT_WORKSPACE_BYTES = 8_519_680  # torch's default CUBLAS_WORKSPACE_CONFIG (":4096:2:16:8": 4096 KiB * 2 + 16 KiB * 8)
MAX_WORKSPACE_BYTES = 1 << 40
ALIGNMENT = 16
DIGEST_FORMAT = 2  # of the canonical declaration below (2: the digest of an entry covers the SM count of its device)

# what every entry of this implementation says about the problem, as cuBLASLt is called (see the module docstring and describe())
COMPUTE_TYPE = "CUBLAS_COMPUTE_32F"
SCALE_TYPE = "CUDA_R_32F"
TRANSA, TRANSB = "T", "N"  # cuBLAS is column-major: its A operand is tilefish's b (op T), its B operand is tilefish's a (op N)

_ALGO = re.compile(r"^[0-9a-f]{128}\Z")
_ARCH = re.compile(r"^sm_[0-9]{2,3}\Z")
ENTRY_KEYS = (
    "stem op arch impl abi buckets dims preconditions dtypes compute_type scale_type transa transb m n k lda ldb ldc ldd "
    "epilogue workspace_bytes algo cublaslt_version sm_count declaration_digest"
).split()


def has_bias(impl: str) -> bool:
    return IMPLS[impl]


# -- the declaration -------------------------------------------------------------------------------------------------------


def check_declaration(name: str, impl: Any, sig: "Signature", acc: Any, workspace_bytes: Any) -> None:
    """Check that a ``@library(impl=...)`` declaration has the signature of its implementation; raise a ``TilefishDeclarationError``."""
    where = f"library op '{name}'"
    if impl not in IMPLS:
        raise TilefishDeclarationError(f"{where}: impl must be one of {sorted(IMPLS)}, got {impl!r}")
    if acc is not None and getattr(acc, "name", None) != "f32":
        raise TilefishDeclarationError(f"{where}: {impl} accumulates in f32 (acc=f32), got acc={acc!r}")
    if isinstance(workspace_bytes, bool) or not isinstance(workspace_bytes, int) or not 1 <= workspace_bytes <= MAX_WORKSPACE_BYTES:
        raise TilefishDeclarationError(f"{where}: workspace_bytes must be an int from 1 to 2**40, got {workspace_bytes!r}")
    bias = IMPLS[impl]
    params = list(sig.params)
    want = 3 if bias else 2
    if len(params) != want or any(p.kind != "tensor" or p.access != "in" or p.strided or p.is_devscalar for p in params):
        raise TilefishDeclarationError(
            f"{where}: {impl} takes {want} input tensors "
            f"({'a: Tensor[M, K, bf16], b: Tensor[N, K, bf16], bias: Tensor[N, bf16]' if bias else 'a: Tensor[M, K, bf16], b: Tensor[N, K, bf16]'}), "
            f"and returns Tensor[M, N, bf16]; the signature is {sig}"
        )
    if len(sig.returns) != 1 or sig.tensor_outputs:
        raise TilefishDeclarationError(f"{where}: {impl} returns exactly one tensor, Tensor[M, N, bf16] (not an Out parameter)")
    a, b = params[0].type, params[1].type
    out = sig.returns[0]
    for label, t, rank in (("a", a, 2), ("b", b, 2), ("the result", out, 2)) + ((("bias", params[2].type, 1),) if bias else ()):
        assert t is not None
        if t.rank != rank or t.dtype.name != "bf16" or not t.contiguous:  # type: ignore[union-attr]
            raise TilefishDeclarationError(f"{where}: {label} must be a bf16 tensor of rank {rank}, got {t}")
    m, k, n = a.shape[0], a.shape[1], b.shape[0]  # type: ignore[union-attr]
    if b.shape[1] != k:  # type: ignore[union-attr]
        raise TilefishDeclarationError(f"{where}: a is [M, K] and b is [N, K]: their last axes must be the same ({a.shape[1]} vs {b.shape[1]})")  # type: ignore[union-attr]
    if tuple(out.shape) != (m, n) or (bias and params[2].type.shape != (n,)):  # type: ignore[union-attr]
        raise TilefishDeclarationError(f"{where}: the result must be [M, N] = [{m}, {n}]" + (f" and bias [N] = [{n}]" if bias else ""))
    for label, d in (("M", m), ("N", n), ("K", k)):
        free = d.free_dims()
        loose = sorted(f.name for f in free if not f.buckets)
        if loose:
            raise TilefishDeclarationError(
                f"{where}: {label} = {d} depends on the dynamic dimension {', '.join(loose)} without buckets; a pinned algorithm is for "
                "one size, so give it Dim(name, buckets=[...]) or a static value"
            )
    for label, d in (("N", n), ("K", k)):  # every row of a, b and the result starts on 16 bytes
        if not _multiple_of_8(d):
            raise TilefishDeclarationError(
                f"{where}: {label} = {d} must be a multiple of 8 (bf16 rows start on 16 bytes); give it divisible_by=8, buckets that are "
                "multiples of 8, or a static size that is"
            )


def _multiple_of_8(d: DimExpr) -> bool:
    """Whether a dimension is a multiple of 8 for every bucket of the dimensions it mentions (all of them have buckets, or it is static)."""
    if d.is_divisible_by(8):
        return True
    free = sorted(d.free_dims(), key=lambda f: f.name)
    return all(d.subst(dict(zip(free, combo))).value % 8 == 0 for combo in itertools.product(*(f.buckets for f in free)))  # type: ignore[arg-type]


def _canonical(op: "LibraryOp") -> "dict[str, Any]":
    from .aot import _canonical_dim

    sig = op.signature
    params = []
    for p in sig.params:
        t = p.type
        assert t is not None
        params.append({"name": p.name, "dtype": t.dtype.name, "shape": [_canonical_dim(d) for d in t.shape]})  # type: ignore[union-attr]
    return {
        "format": DIGEST_FORMAT,
        "op": op.name,
        "impl": op.impl,
        "acc": "f32",
        "workspace_bytes": op.workspace_bytes,
        "params": params,
        "returns": [{"dtype": r.dtype.name, "shape": [_canonical_dim(d) for d in r.shape]} for r in sig.returns],  # type: ignore[union-attr]
    }


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

    @property
    def bias(self) -> bool:
        return IMPLS[self.impl]

    def pinned_digest(self, sm_count: int) -> str:
        """The ``declaration_digest`` of an entry of this spec pinned on a device of ``sm_count`` multiprocessors."""
        return _digest({**self.canonical, "sm_count": sm_count})

    def entry(self, sm_count: int) -> "dict[str, Any]":
        """The fields of the library entry that the declaration and the SM count of the device determine (a replay checks the manifest's entry
        against them); the build adds ``arch``, ``algo`` and ``cublaslt_version``."""
        dtypes = {"a": "bf16", "b": "bf16", **({"bias": "bf16"} if self.bias else {}), "out": "bf16"}
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
            "transa": TRANSA,
            "transb": TRANSB,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "lda": self.k,
            "ldb": self.k,
            "ldc": self.n,
            "ldd": self.n,
            "epilogue": "bias" if self.bias else "none",
            "workspace_bytes": self.workspace_bytes,
            "sm_count": sm_count,
            "declaration_digest": self.pinned_digest(sm_count),
        }


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
    m_expr, k_expr = sig.params[0].type.shape  # type: ignore[union-attr]
    n_expr = sig.params[1].type.shape[0]  # type: ignore[union-attr]
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
                    "dtype": "bf16",
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
        return out


def build_library(spec: LibrarySpec, *, arch: "int | str", device_index: "int | None" = None) -> BuiltLibrary:
    """Pin the algorithm of ``spec`` for ``arch``: ask cuBLASLt's heuristic, with the spec's workspace limit, for its best algorithm.

    The heuristic is asked on a device of that architecture (a pin is per architecture), so the device must be one.
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
    pinned = _cublaslt.pin_algo(spec.m, spec.n, spec.k, spec.bias, workspace_bytes=spec.workspace_bytes, device_index=index)
    return BuiltLibrary(spec, f"sm_{cap}", pinned.algo_hex, _cublaslt.version(), sm_count(index))


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


def built_from_manifest(manifest: Any, spec: LibrarySpec, arch: str, sm_count: int) -> BuiltLibrary:
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
    return BuiltLibrary(spec, arch, entry["algo"], pinned_by, pinned_on)


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


def validate_entry(entry: Any, index: int = 0) -> None:
    """Strict reading of one entry of the manifest's ``library`` section (raises :class:`~tilefish.errors.TilefishManifestError`).

    Every field must be present and exactly typed, and the redundant ones must agree: the dtypes, the cuBLAS-level problem (the T/N
    transposes, the leading dimensions) and the shapes of the operands follow from ``impl``, ``m``, ``n`` and ``k``.
    """
    where = f"#{index}"
    if not isinstance(entry, dict):
        _bad(where, f"is not an object: {entry!r}")
    where = f"#{index} ({entry.get('stem')!r})"
    missing = [k for k in ENTRY_KEYS if k not in entry]
    unknown = sorted(str(k) for k in set(entry) - set(ENTRY_KEYS))
    if missing or unknown:
        _bad(where, f"must have exactly the fields {ENTRY_KEYS}" + (f"; missing {missing}" if missing else "") + (f"; unknown {unknown}" if unknown else ""))
    try:
        stem = entry["stem"]
        if not isinstance(stem, str) or not sv._STEM.match(stem):
            _bad(where, f"stem must look like 'gemm' or 'gemm.M64', got {stem!r}")
        sv._ident(entry["op"], "op")
        if not isinstance(entry["arch"], str) or not _ARCH.match(entry["arch"]):
            _bad(where, f"arch must look like 'sm_103', got {entry['arch']!r}")
        impl = entry["impl"]
        if impl not in IMPLS:
            _bad(where, f"impl must be one of {sorted(IMPLS)}, got {impl!r}")
        bias = IMPLS[impl]
        for key in ("m", "n", "k", "lda", "ldb", "ldc", "ldd"):
            v = entry[key]
            if not sv.is_int(v) or not 1 <= v <= sv.MAX_ELEMENTS:
                _bad(where, f"{key} must be an int from 1 to {sv.MAX_ELEMENTS}, got {v!r}")
        m, n, k = entry["m"], entry["n"], entry["k"]
        if (entry["lda"], entry["ldb"], entry["ldc"], entry["ldd"]) != (k, k, n, n):
            _bad(where, "the leading dimensions must be lda = ldb = k and ldc = ldd = n (cuBLAS sees b as its A, a as its B and the result as C and D)")
        if (entry["transa"], entry["transb"]) != (TRANSA, TRANSB):
            _bad(where, f"transa and transb must be {TRANSA!r} and {TRANSB!r}, got {entry['transa']!r} and {entry['transb']!r}")
        if entry["compute_type"] != COMPUTE_TYPE or entry["scale_type"] != SCALE_TYPE:
            _bad(where, f"compute_type and scale_type must be {COMPUTE_TYPE!r} and {SCALE_TYPE!r}")
        if entry["epilogue"] != ("bias" if bias else "none"):
            _bad(where, f"the epilogue of {impl} must be {'bias' if bias else 'none'!r}, got {entry['epilogue']!r}")
        want_dtypes = {"a": "bf16", "b": "bf16", **({"bias": "bf16"} if bias else {}), "out": "bf16"}
        if entry["dtypes"] != want_dtypes:
            _bad(where, f"dtypes must be {want_dtypes}, got {entry['dtypes']!r}")
        ws = entry["workspace_bytes"]
        if not sv.is_int(ws) or not 1 <= ws <= MAX_WORKSPACE_BYTES:
            _bad(where, f"workspace_bytes must be an int from 1 to 2**40, got {ws!r}")
        if not isinstance(entry["algo"], str) or not _ALGO.match(entry["algo"]):
            _bad(where, f"algo must be 128 lowercase hex digits (the 64 bytes of a cublasLtMatmulAlgo_t), got {entry['algo']!r}")
        if not sv.is_int(entry["cublaslt_version"]) or entry["cublaslt_version"] < 1:
            _bad(where, f"cublaslt_version must be a positive int, got {entry['cublaslt_version']!r}")
        if not sv.is_int(entry["sm_count"]) or entry["sm_count"] < 1:
            _bad(where, f"sm_count must be a positive int (the multiprocessors of the device the algorithm was pinned on), got {entry['sm_count']!r}")
        if not (isinstance(entry["declaration_digest"], str) and sv._DIGEST.match(entry["declaration_digest"])):
            _bad(where, f"declaration_digest must be 64 lowercase hex digits (a sha256), got {entry['declaration_digest']!r}")
        buckets = entry["buckets"]
        if not isinstance(buckets, dict):
            _bad(where, "buckets must be an object")
        for name, value in buckets.items():
            sv._ident(name, "bucket dimension name")
            sv._int(value, f"bucket value of {name}", 1)
        # the operands: a [M, K], b [N, K], [bias [N]], out [M, N], in this order
        abi = entry["abi"]
        sv._keys(abi, "abi", ["params"])
        params = abi["params"]
        names = ["a", "b"] + (["bias"] if bias else []) + ["out"]
        expected_shapes = [(m, k), (n, k)] + ([(n,)] if bias else []) + [(m, n)]
        if not isinstance(params, list) or len(params) != len(names):
            _bad(where, f"abi.params must list {len(names)} pointers ({', '.join(names)} in this order)")
        seen: "set[str]" = set()
        for i, (p, shape) in enumerate(zip(params, expected_shapes)):
            sv._keys(p, f"abi.params[{i}]", ["index", "name", "kind", "dtype", "access", "shape"])
            sv._ident(p["name"], f"abi.params[{i}].name")
            if p["index"] != i or p["kind"] != "pointer" or p["dtype"] != "bf16" or p["access"] != ("out" if i == len(params) - 1 else "in"):
                _bad(where, f"abi.params[{i}] must be the bf16 pointer #{i} with access {'out' if i == len(params) - 1 else 'in'}")
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
            if "bucket" in c and ("buckets" not in c or c["bucket"] not in c["buckets"] or buckets.get(name) != c["bucket"]):
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
    """Launch ``cublasLtMatmul`` with the pinned ``algo`` on ``stream``: ``tensors`` are ``a``, ``b`` (and ``bias``) by the op's parameter
    names in order, ``out`` the result tensor; all are contiguous, 16-byte aligned and have the shapes of ``spec`` (the caller checked)."""
    a, b, *rest = tensors.values()
    run_gemm_ptrs(
        spec, algo, int(a.data_ptr()), int(b.data_ptr()), int(rest[0].data_ptr()) if rest else None, int(out.data_ptr()), stream, device_index
    )


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
) -> None:
    """:func:`run_gemm` on device addresses. With ``workspace_ptr`` (``spec.workspace_bytes`` bytes, 256-byte aligned: a buffer of a launch plan's
    arena) the workspace is that memory and the call allocates nothing; without it the workspace comes from torch's caching allocator."""
    from . import _cublaslt

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
    )


def reference_gemm(tensors: "dict[str, Any]") -> Any:
    """The result in torch, as the semantics say: an f32 matmul of the bf16 operands, the bias added in f32, one rounding to bf16."""
    import torch

    a, b, *rest = tensors.values()
    acc = a.to(torch.float32) @ b.to(torch.float32).t()
    if rest:
        acc = acc + rest[0].to(torch.float32)
    return acc.to(torch.bfloat16)


__all__ = [
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
