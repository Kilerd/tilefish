"""A library operation with a pinned implementation: a cuDNN forward convolution (``impl="cudnn.conv_fwd"``).

``@tilefish.library("conv", impl="cudnn.conv_fwd", layout="NHWC", stride=1, padding=3, dilation=1, groups=1, api="backend")`` declares a convolution
by its signature, ``(x, w) -> y``, in the layout's order:

    layout "NCHW":   x: Tensor[N, C, *S, T],   w: Tensor[K, C/G, *R, T]  ->  Tensor[N, K, *P, T]
    layout "NHWC":   x: Tensor[N, *S, C, T],   w: Tensor[K, *R, C/G, T]  ->  Tensor[N, *P, K, T]

with one, two or three spatial axes ``S`` (filter ``R``, output ``P``), ``T`` one of ``f16``, ``bf16``, ``f32``, ``G`` groups, and per spatial axis a
stride, a symmetric zero padding and a dilation; each output size is ``(S + 2 * padding - dilation * (R - 1) - 1) // stride + 1``, which the declared
result must say (an expression of the bucketed dimensions, checked at every bucket: ``cdiv(L + 2 * p - d * (R - 1), s)`` is that size). It is
the cross-correlation torch's ``conv1d`` / ``conv2d`` / ``conv3d`` compute, with an f32 accumulator and one rounding to ``T``; there is no bias: torch's
cuDNN convolution adds its bias with a separate ``add_``, and a tilefish kernel does the same.

**A binary is a pinned choice of cuDNN** (:mod:`tilefish._cudnn`). ``api="legacy"`` pins the algorithm enum of ``cudnnConvolutionForward`` (chosen
by ``cudnnGetConvolutionForwardAlgorithm_v7``'s first deterministic answer that fits the workspace); ``api="backend"`` (the default) pins an engine
configuration of the backend API, an engine and its knobs, which names a kernel (chosen as torch 2.9 chooses with ``deterministic = True`` and
``benchmark = False``: ``heuristic="instant"``, or ``"mode_b"``). ``algos=`` imports the pins instead (:class:`CudnnTable`, one per device). The entry
records the cuDNN version, the SM count of the device (and ``sm_count_target`` for a partition) and the math; a replay refuses another cuDNN version,
another device and a process with ``NVIDIA_TF32_OVERRIDE`` set.

**A canary.** ``cudnnGetVersion()`` does not tell two builds of one version apart, and they may map other engine libraries. So ``export`` records a
canary in each entry, ``{"seed": s, "y_digest": d}``: the digest of the ``y`` the pin computes from operands made of the seed at the entry's shape, both
made on the device (:mod:`tilefish._canary`: a PTX module, the same in both hosts). A replay runs it once per process and device before the pin is
used, on a stream of its own (never the NULL stream, which a green context does not order against another stream), and refuses the pin when the digest
differs. The field is optional: an entry without one loads as before.

The math is part of the declaration: 16-bit data takes ``"tensor_op"`` (the default) or ``"fma"``; f32 data must say ``"fma"`` (TF32 off) or
``"tf32"`` (TF32 on). See ``docs/en/manifest-schema-3.md``, section 9.4.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import _specvalid as sv
from .dims import Dim, DimExpr
from .errors import TilefishDeclarationError, TilefishManifestError, TilefishRuntimeError, TilefishSpecError

if TYPE_CHECKING:
    from .decorators import LibraryOp
    from .signature import Signature

IMPL = "cudnn.conv_fwd"
DEFAULT_WORKSPACE_BYTES = 64 << 20  # a launch provides this many bytes (and a legacy call is given them all)
MAX_WORKSPACE_BYTES = 1 << 40
DIGEST_FORMAT = 1
COMPUTE_TYPE = "CUDNN_DATA_FLOAT"
CHOICES = ("v7", "instant", "mode_b", "imported")  # how the pin was chosen: a posture (tilefish._cudnn.HEURISTICS), or imported
DEFAULT_HEURISTIC = {"legacy": "v7", "backend": "instant"}
ENTRY_KEYS = (
    "stem op arch impl abi buckets dims preconditions dtypes compute_type math layout n c k groups input filter output stride padding dilation "
    "alignment workspace_bytes choice pin cudnn_version sm_count declaration_digest"
).split()
OPTIONAL_COUNT_FIELDS = ("sm_count_target",)
OPTIONAL_FIELDS = (*OPTIONAL_COUNT_FIELDS, "canary")
CANARY_SEED = 1  # the seed export records
MAX_CANARY_SEED = (1 << 32) - 1
_DIGEST32 = re.compile(r"^[0-9a-f]{32}\Z")
_ARCH_LEN = (2, 3)
_ELEMENT_BYTES = {"f16": 2, "bf16": 2, "f32": 4}


def canary_bytes(dtype: str, seed: int, tag: int, elements: int) -> bytes:
    """The first ``elements`` elements of a canary operand (tag 0: ``x``, 1: ``w``), made on the host: what the device makes (:mod:`tilefish._canary`).
    A reference for the tests: slow for large operands."""
    from . import _canary

    return _canary.canary_bytes(dtype, seed, tag, elements)


def canary_operands(problem: Any, seed: int) -> "tuple[bytes, bytes]":
    """The canary's ``x`` and ``w`` for ``problem`` (a :class:`tilefish._cudnn.Problem`), dense in its layout, made on the host (a reference)."""
    return (
        canary_bytes(problem.dtype, seed, 0, math.prod(problem.x_dims)),
        canary_bytes(problem.dtype, seed, 1, math.prod(problem.w_dims)),
    )


def is_conv(op: Any) -> bool:
    return getattr(op, "impl", None) == IMPL


# -- the options of a declaration ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvOptions:
    """What a ``cudnn.conv_fwd`` declaration says besides its signature (the spatial tuples have one entry per spatial axis)."""

    layout: str
    stride: "tuple[int, ...]"
    padding: "tuple[int, ...]"
    dilation: "tuple[int, ...]"
    groups: int
    math: str
    api: str
    heuristic: str
    alignment: int


def _spatial(name: str, where: str, value: Any, count: int, low: int) -> "tuple[int, ...]":
    values = (value,) * count if isinstance(value, int) and not isinstance(value, bool) else value
    if not isinstance(values, (tuple, list)) or len(values) != count or any(isinstance(v, bool) or not isinstance(v, int) or v < low for v in values):
        raise TilefishDeclarationError(f"{where}: {name} is an int of at least {low}, or one per spatial axis ({count}), got {value!r}")
    return tuple(values)


def check_options(
    name: str,
    sig: "Signature",
    *,
    layout: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
    math: Any,
    api: Any,
    heuristic: Any,
    alignment: Any,
) -> ConvOptions:
    """The options of a ``cudnn.conv_fwd`` declaration, checked against its signature's rank (raises :class:`TilefishDeclarationError`)."""
    from . import _cudnn

    where = f"library op '{name}'"
    if layout not in _cudnn.LAYOUTS:
        raise TilefishDeclarationError(f"{where}: the layout of a cudnn.conv_fwd is one of {list(_cudnn.LAYOUTS)} (the order of x, w and the result), got {layout!r}")
    params = list(sig.params)
    rank = params[0].type.rank if params and params[0].type is not None else 0  # type: ignore[union-attr]
    count = rank - 2
    if not 1 <= count <= 3:
        raise TilefishDeclarationError(f"{where}: x of a cudnn.conv_fwd has 1 to 3 spatial axes (rank 3 to 5), got rank {rank}")
    if api not in _cudnn.APIS:
        raise TilefishDeclarationError(f"{where}: api is one of {list(_cudnn.APIS)} (the pin: a legacy algorithm, or a backend engine configuration), got {api!r}")
    posture = DEFAULT_HEURISTIC[api] if heuristic is None else heuristic
    if _cudnn.HEURISTICS.get(posture) != api:
        allowed = [h for h, a in _cudnn.HEURISTICS.items() if a == api]
        raise TilefishDeclarationError(f"{where}: the heuristic of the {api} API is one of {allowed}, got {heuristic!r}")
    if isinstance(groups, bool) or not isinstance(groups, int) or groups < 1:
        raise TilefishDeclarationError(f"{where}: groups is a positive int, got {groups!r}")
    if alignment not in _cudnn.ALIGNMENTS:
        raise TilefishDeclarationError(f"{where}: alignment is one of {list(_cudnn.ALIGNMENTS)} bytes, got {alignment!r}")
    dtype = getattr(getattr(params[0].type, "dtype", None), "name", None) if params else None
    if math is None:
        if dtype == "f32":
            raise TilefishDeclarationError(f"{where}: f32 data needs math='fma' (TF32 off) or math='tf32' (TF32 on), said explicitly")
        math = "tensor_op"
    if math not in _cudnn.MATHS:
        raise TilefishDeclarationError(f"{where}: math is one of {list(_cudnn.MATHS)}, got {math!r}")
    if dtype in _cudnn.DTYPES:
        problem = _cudnn.math_problem(dtype, math)
        if problem is not None:
            raise TilefishDeclarationError(f"{where}: {problem}")
    return ConvOptions(
        layout,
        _spatial("stride", where, stride, count, 1),
        _spatial("padding", where, padding, count, 0),
        _spatial("dilation", where, dilation, count, 1),
        groups,
        math,
        api,
        posture,
        alignment,
    )


# -- imported pins -----------------------------------------------------------------------------------------------------------------


def table_key(n: int, c: int, k: int, input: "tuple[int, ...]", filter: "tuple[int, ...]", stride: "tuple[int, ...]", padding: "tuple[int, ...]", dilation: "tuple[int, ...]", groups: int) -> "tuple[Any, ...]":
    """The key of a :class:`CudnnTable` row: the problem's sizes and its convolution, spatial values as tuples."""
    return (n, c, k, tuple(input), tuple(filter), tuple(stride), tuple(padding), tuple(dilation), groups)


def key_of(problem: Any) -> "tuple[Any, ...]":
    return table_key(problem.n, problem.c, problem.k, problem.input, problem.filter, problem.stride, problem.padding, problem.dilation, problem.groups)


def _check_key(key: Any, where: str) -> "tuple[Any, ...]":
    ok = isinstance(key, tuple) and len(key) == 9
    if ok:
        n, c, k, inp, flt, st, pad, dil, g = key
        ints = [n, c, k, g]
        tuples = [inp, flt, st, pad, dil]
        ok = all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in ints) and all(
            isinstance(t, (tuple, list)) and 1 <= len(t) <= 3 and len(t) == len(inp) and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in t)
            for t in tuples
        )
    if not ok:
        raise TilefishDeclarationError(
            f"{where}: a key is (n, c, k, input, filter, stride, padding, dilation, groups), ints and a tuple of ints per spatial axis, got {key!r}"
        )
    return table_key(*key)


@dataclass(frozen=True, eq=False)
class CudnnTable:
    """cuDNN pins to take instead of asking cuDNN, for one device: an architecture (``"sm_103"``), the SM count of the device and the
    ``cudnnGetVersion()`` they belong to.

    ``pins`` maps a problem, ``(n, c, k, input, filter, stride, padding, dilation, groups)`` (the spatial values as tuples, one entry per spatial
    axis of the declaration: ``(1, 64, 64, (646,), (7,), (1,), (6,), (1,), 1)``), to a pin: an int, the legacy algorithm enum, or ``{"engine": e,
    "knobs": {type: value}}`` (or ``[[type, value], ...]``), a backend engine configuration; the form of every pin of a declaration is its ``api``'s.
    A knob of the value -1 is cuDNN's 'not set' (how a plan's JSON shows a knob left at the engine's default): it is dropped, so the pin is the same with
    it or without it (and so is the declaration's digest).
    A declaration with ``algos=`` runs, on a device a table is for, the table's pin for each of its entries after the checks of a pin of the heuristic
    (``tilefish._cudnn.check_pin``), recorded as ``"choice": "imported"``. Rows for other problems are allowed and do not reach a declaration's
    digest."""

    arch: str
    sm_count: int
    cudnn_version: int
    pins: "dict[tuple[Any, ...], Any]"

    def __post_init__(self) -> None:
        from . import _cudnn

        where = "CudnnTable"
        if not isinstance(self.arch, str) or not self.arch.startswith("sm_") or not self.arch[3:].isdigit() or len(self.arch[3:]) not in _ARCH_LEN:
            raise TilefishDeclarationError(f"{where}: arch must look like 'sm_120', got {self.arch!r}")
        for name in ("sm_count", "cudnn_version"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise TilefishDeclarationError(f"{where}: {name} must be a positive int, got {v!r}")
        if not isinstance(self.pins, dict) or not self.pins:
            raise TilefishDeclarationError(f"{where}: pins must be a non-empty dict of problem -> pin, got {self.pins!r}")
        rows: "dict[tuple[Any, ...], Any]" = {}
        for key, pin in self.pins.items():
            row = _check_key(key, where)
            if row in rows:
                raise TilefishDeclarationError(f"{where}: two rows are for {row}")
            try:
                rows[row] = _cudnn.pin_of(pin, f"{where} ({self.arch}, {self.sm_count} SMs) pin for {key}", imported=True)
            except TilefishRuntimeError as e:
                raise TilefishDeclarationError(str(e)) from None
        object.__setattr__(self, "pins", rows)

    @property
    def device(self) -> "tuple[str, int, int]":
        return (self.arch, self.sm_count, self.cudnn_version)

    def canonical(self, keys: "list[tuple[Any, ...]]") -> "dict[str, Any]":
        return {
            "arch": self.arch,
            "sm_count": self.sm_count,
            "cudnn_version": self.cudnn_version,
            "pins": [[_jsonable(key), self.pins[key].to_json()] for key in sorted(keys)],
        }

    def __repr__(self) -> str:
        return f"CudnnTable({self.arch!r}, sm_count={self.sm_count}, cudnn_version={self.cudnn_version}, {len(self.pins)} pins)"


def _jsonable(key: "tuple[Any, ...]") -> "list[Any]":
    return [list(v) if isinstance(v, tuple) else v for v in key]


def check_tables(name: str, algos: Any, api: str) -> "tuple[CudnnTable, ...]":
    where = f"library op '{name}'"
    tables = (algos,) if isinstance(algos, CudnnTable) else algos
    if not isinstance(tables, (list, tuple)) or not tables or any(not isinstance(t, CudnnTable) for t in tables):
        raise TilefishDeclarationError(f"{where}: algos of a cudnn.conv_fwd is a CudnnTable or a non-empty list of them, got {algos!r}")
    seen: "set[tuple[str, int, int]]" = set()
    for t in tables:
        if t.device in seen:
            raise TilefishDeclarationError(f"{where}: two pin tables are for {t.arch} with {t.sm_count} SMs and cuDNN {t.cudnn_version}; give one per device")
        seen.add(t.device)
        for key, pin in t.pins.items():
            if pin.api != api:
                raise TilefishDeclarationError(f"{where}: the declaration's api is {api!r}, and the table for {t.arch} has a {pin.api} pin for {key}")
    return tuple(tables)


def table_for(tables: "tuple[CudnnTable, ...]", arch: str, sm_count: int, version: int) -> "CudnnTable | None":
    for t in tables:
        if t.device == (arch, sm_count, version):
            return t
    return None


# -- the declaration -------------------------------------------------------------------------------------------------------


def _layout_axes(layout: str, rank: int) -> "tuple[int, int, list[int]]":
    """The axis of the batch (or of ``k`` for ``w``), of the channels and the spatial axes, of a tensor of ``rank`` in ``layout``."""
    if layout == "NCHW":
        return 0, 1, list(range(2, rank))
    return 0, rank - 1, list(range(1, rank - 1))


def check_declaration(name: str, sig: "Signature", acc: Any, workspace_bytes: Any, options: ConvOptions, sm_count_target: Any) -> None:
    """Check that a ``cudnn.conv_fwd`` declaration has the signature of its layout (raises :class:`TilefishDeclarationError`)."""
    from . import _cudnn

    where = f"library op '{name}'"
    if acc is not None and getattr(acc, "name", None) != "f32":
        raise TilefishDeclarationError(f"{where}: {IMPL} accumulates in f32 (acc=f32), got acc={acc!r}")
    if isinstance(workspace_bytes, bool) or not isinstance(workspace_bytes, int) or not 1 <= workspace_bytes <= MAX_WORKSPACE_BYTES:
        raise TilefishDeclarationError(f"{where}: workspace_bytes must be an int from 1 to 2**40, got {workspace_bytes!r}")
    params = list(sig.params)
    shapes = "x: Tensor[N, C, *S, T], w: Tensor[K, C/G, *R, T]" if options.layout == "NCHW" else "x: Tensor[N, *S, C, T], w: Tensor[K, *R, C/G, T]"
    if len(params) != 2 or any(p.kind != "tensor" or p.access != "in" or p.strided or p.is_devscalar for p in params):
        raise TilefishDeclarationError(f"{where}: {IMPL} takes 2 input tensors ({shapes}, layout {options.layout}) and returns one; the signature is {sig}")
    if len(sig.returns) != 1 or sig.tensor_outputs:
        raise TilefishDeclarationError(f"{where}: {IMPL} returns exactly one tensor (not an Out parameter)")
    x, w, y = params[0].type, params[1].type, sig.returns[0]
    rank = x.rank  # type: ignore[union-attr]
    for label, t in (("x", x), ("w", w), ("the result", y)):
        if t.rank != rank or not t.contiguous:  # type: ignore[union-attr]
            raise TilefishDeclarationError(f"{where}: {label} must be a contiguous tensor of rank {rank} (as x), got {t}")
    dtype = x.dtype.name  # type: ignore[union-attr]
    if dtype not in _cudnn.DTYPES or w.dtype.name != dtype or y.dtype.name != dtype:  # type: ignore[union-attr]
        raise TilefishDeclarationError(f"{where}: x, w and the result are of one type, f16, bf16 or f32, got {x}, {w} and {y}")
    if sm_count_target is not None:
        if isinstance(sm_count_target, bool) or not isinstance(sm_count_target, int) or sm_count_target < 1:
            raise TilefishDeclarationError(f"{where}: sm_count_target must be None or a positive int, got {sm_count_target!r}")
    b, ch, sp = _layout_axes(options.layout, rank)
    if x.shape[b] != y.shape[b]:  # type: ignore[union-attr]
        raise TilefishDeclarationError(f"{where}: x and the result have one batch size N ({x.shape[b]} vs {y.shape[b]})")  # type: ignore[union-attr]
    if w.shape[b] != y.shape[ch]:  # type: ignore[union-attr]
        raise TilefishDeclarationError(f"{where}: the output channels K of w and of the result must be the same ({w.shape[b]} vs {y.shape[ch]})")  # type: ignore[union-attr]
    for label, d in (("x", x), ("w", w), ("the result", y)):
        for axis, e in enumerate(d.shape):  # type: ignore[union-attr]
            loose = sorted(f.name for f in e.free_dims() if not f.buckets)
            if loose:
                raise TilefishDeclarationError(
                    f"{where}: axis {axis} of {label} = {e} depends on the dynamic dimension {', '.join(loose)} without buckets; a pinned convolution is "
                    "for one size, so give it Dim(name, buckets=[...]) or a static value"
                )
    for combo, mapping in _combos(sig):
        problem_at(name, sig, options, mapping, where)  # every bucket combination is a problem cuDNN takes, and the declared result its output


def _combos(sig: "Signature") -> "list[tuple[tuple[int, ...], dict[Dim, int]]]":
    free: "dict[str, Dim]" = {}
    for t in [p.type for p in sig.params] + list(sig.returns):
        for d in t.shape:  # type: ignore[union-attr]
            for f in d.free_dims():
                free.setdefault(f.name, f)
    dims = [free[n] for n in sorted(free)]
    out = []
    for combo in itertools.product(*(d.buckets for d in dims)) if dims else [()]:  # type: ignore[arg-type]
        out.append((combo, dict(zip(dims, combo))))
    return out


def _value(e: DimExpr, mapping: "dict[Dim, int]") -> int:
    return int(e.subst(mapping).folded().value)


def problem_at(name: str, sig: "Signature", options: ConvOptions, mapping: "dict[Dim, int]", where: "str | None" = None) -> Any:
    """The :class:`tilefish._cudnn.Problem` of a declaration at one bucket combination, after the checks that the shapes agree with it."""
    from . import _cudnn

    where = where or f"library op '{name}'"
    x, w, y = sig.params[0].type, sig.params[1].type, sig.returns[0]
    rank = x.rank  # type: ignore[union-attr]
    b, ch, sp = _layout_axes(options.layout, rank)
    xs = [_value(e, mapping) for e in x.shape]  # type: ignore[union-attr]
    ws = [_value(e, mapping) for e in w.shape]  # type: ignore[union-attr]
    ys = [_value(e, mapping) for e in y.shape]  # type: ignore[union-attr]
    at = f" (at {', '.join(f'{d.name}={v}' for d, v in mapping.items())})" if mapping else ""
    c, k = xs[ch], ws[b]
    if c % options.groups or k % options.groups:
        raise TilefishDeclarationError(f"{where}: the groups ({options.groups}) divide the input channels C ({c}) and the output channels K ({k}){at}")
    if ws[ch] != c // options.groups:
        raise TilefishDeclarationError(f"{where}: the channel axis of w is C/G = {c}/{options.groups} = {c // options.groups}, got {ws[ch]}{at}")
    try:
        problem = _cudnn.Problem(
            x.dtype.name, options.layout, xs[b], c, k, options.groups, tuple(xs[i] for i in sp), tuple(ws[i] for i in sp), options.stride,  # type: ignore[union-attr]
            options.padding, options.dilation, options.math, options.alignment,
        )  # fmt: skip
    except TilefishRuntimeError as e:
        raise TilefishDeclarationError(f"{where}: {str(e).removeprefix('cuDNN convolution: ')}{at}") from None
    want = list(problem.tilefish_shape("y"))
    if ys != want:
        raise TilefishDeclarationError(
            f"{where}: the result of the convolution is {want}{at}, and the declaration says {ys} (each output size is (S + 2 * padding - dilation * "
            "(R - 1) - 1) // stride + 1)"
        )
    return problem


def _canonical(op: "LibraryOp") -> "dict[str, Any]":
    from .aot import _canonical_dim

    sig = op.signature
    o: ConvOptions = op.conv  # type: ignore[attr-defined]
    out: "dict[str, Any]" = {
        "format": DIGEST_FORMAT,
        "op": op.name,
        "impl": IMPL,
        "acc": "f32",
        "workspace_bytes": op.workspace_bytes,
        "params": [{"name": p.name, "dtype": p.type.dtype.name, "shape": [_canonical_dim(d) for d in p.type.shape]} for p in sig.params],  # type: ignore[union-attr]
        "returns": [{"dtype": r.dtype.name, "shape": [_canonical_dim(d) for d in r.shape]} for r in sig.returns],  # type: ignore[union-attr]
        "layout": o.layout,
        "stride": list(o.stride),
        "padding": list(o.padding),
        "dilation": list(o.dilation),
        "groups": o.groups,
        "math": o.math,
        "api": o.api,
        "heuristic": o.heuristic,
        "alignment": o.alignment,
    }
    if getattr(op, "sm_count_target", None) is not None:
        out["sm_count_target"] = op.sm_count_target
    tables = getattr(op, "algos", None)
    if tables:
        keys = [key_of(problem) for _, problem in _problems(op)]
        out["algos"] = sorted((t.canonical(keys) for t in tables), key=lambda c: (c["arch"], c["sm_count"], c["cudnn_version"]))
    return out


def _problems(op: "LibraryOp") -> "list[tuple[dict[Dim, int], Any]]":
    return [(mapping, problem_at(op.name, op.signature, op.conv, mapping)) for _, mapping in _combos(op.signature)]  # type: ignore[attr-defined]


def check_coverage(op: "LibraryOp") -> None:
    tables = getattr(op, "algos", None)
    if not tables:
        return
    keys = [key_of(p) for _, p in _problems(op)]
    for t in tables:
        missing = [k for k in keys if k not in t.pins]
        if missing:
            raise TilefishDeclarationError(
                f"library op '{op.name}': the pin table for {t.arch} with {t.sm_count} SMs (cuDNN {t.cudnn_version}) has no pin for "
                f"{', '.join(map(str, missing[:3]))}{' ...' if len(missing) > 3 else ''}: a table gives one for every entry of the declaration"
            )


def _digest(canonical: "dict[str, Any]") -> str:
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def declaration_digest(op: "LibraryOp", sm_count: "int | None" = None) -> str:
    canonical = _canonical(op)
    return _digest(canonical if sm_count is None else {**canonical, "sm_count": sm_count})


# -- the specs: one per bucket combination ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ConvSpec:
    """Everything about one pinned convolution that does not depend on the device or on the pin: what an entry of the manifest's ``library``
    section says (:meth:`entry`) before the build adds ``arch``, ``pin``, ``cudnn_version`` and ``choice``."""

    stem: str
    op: str
    problem: Any  # tilefish._cudnn.Problem
    workspace_bytes: int
    abi: "dict[str, Any]"
    buckets: "dict[str, int]"
    dims: "dict[str, dict[str, Any]]"
    preconditions: "tuple[Any, ...]"
    declaration_digest: str
    canonical: "dict[str, Any]"
    api: str
    heuristic: str
    algos: "tuple[CudnnTable, ...] | None" = None
    sm_count_target: "int | None" = None

    impl = IMPL

    @property
    def key(self) -> "tuple[Any, ...]":
        return key_of(self.problem)

    @property
    def choice(self) -> str:
        return "imported" if self.algos else self.heuristic

    def pinned_digest(self, sm_count: int) -> str:
        return _digest({**self.canonical, "sm_count": sm_count})

    def entry(self, sm_count: int) -> "dict[str, Any]":
        """The fields of the entry that the declaration and the SM count of the device determine; the build adds ``arch``, ``pin`` and
        ``cudnn_version``."""
        p = self.problem
        out = {
            "stem": self.stem,
            "op": self.op,
            "impl": IMPL,
            "abi": json.loads(json.dumps(self.abi)),
            "buckets": dict(self.buckets),
            "dims": json.loads(json.dumps(self.dims)),
            "preconditions": [pre.to_dict() for pre in self.preconditions],
            "dtypes": {"x": p.dtype, "w": p.dtype, "y": p.dtype},
            "compute_type": COMPUTE_TYPE,
            "math": p.math,
            "layout": p.layout,
            "n": p.n,
            "c": p.c,
            "k": p.k,
            "groups": p.groups,
            "input": list(p.input),
            "filter": list(p.filter),
            "output": list(p.output),
            "stride": list(p.stride),
            "padding": list(p.padding),
            "dilation": list(p.dilation),
            "alignment": p.alignment,
            "workspace_bytes": self.workspace_bytes,
            "choice": self.choice,
            "sm_count": sm_count,
            "declaration_digest": self.pinned_digest(sm_count),
        }
        if self.sm_count_target is not None:
            out["sm_count_target"] = self.sm_count_target
        return out


def _shape_entry(d: DimExpr) -> "str | int":
    folded = d.folded()
    return folded.value if folded.is_static else str(folded)


def conv_specs(op: "LibraryOp") -> "list[ConvSpec]":
    """One :class:`ConvSpec` per bucket combination of a ``cudnn.conv_fwd`` op."""
    from .aot import Precondition, _dim_constraints

    sig = op.signature
    o: ConvOptions = op.conv  # type: ignore[attr-defined]
    tensors = [(p.name, p.type, "in") for p in sig.params] + [("out", sig.returns[0], "out")]
    canonical = _canonical(op)
    digest = _digest(canonical)
    specs = []
    for mapping, problem in _problems(op):
        table: "dict[str, dict[str, Any]]" = {}
        params: "list[dict[str, Any]]" = []
        pres: "list[Any]" = []
        for i, (name, t, access) in enumerate(tensors):
            for d in t.shape:  # type: ignore[union-attr]
                for f in d.free_dims():
                    found = _dim_constraints(f)
                    if table.setdefault(f.name, found) != found:
                        raise TilefishSpecError(f"library op '{op.name}': the dimension {f.name!r} is declared twice with different constraints")
            params.append({"index": i, "name": name, "kind": "pointer", "dtype": t.dtype.name, "access": access, "shape": [_shape_entry(d) for d in t.shape]})  # type: ignore[union-attr]
            pres.append(Precondition("pointer_alignment", arg=name, index=i, bytes=o.alignment))
            for axis, d in enumerate(t.shape):  # type: ignore[union-attr]
                if d.is_static:
                    pres.append(Precondition("static_size", arg=name, index=i, axis=axis, value=d.folded().value))
        for d, v in mapping.items():
            pres.append(Precondition("bucket", dim=d.name, value=v, buckets=d.buckets))
            table[d.name]["bucket"] = v
        specs.append(
            ConvSpec(
                stem=op.name + "".join(f".{d.name}{v}" for d, v in mapping.items()),
                op=op.name,
                problem=problem,
                workspace_bytes=op.workspace_bytes,  # type: ignore[arg-type]
                abi={"params": params},
                buckets={d.name: v for d, v in mapping.items()},
                dims=dict(sorted(table.items())),
                preconditions=tuple(pres),
                declaration_digest=digest,
                canonical=canonical,
                api=o.api,
                heuristic=o.heuristic,
                algos=getattr(op, "algos", None),
                sm_count_target=getattr(op, "sm_count_target", None),
            )
        )
    return specs


# -- the build: pin it ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BuiltConv:
    """One pinned convolution for one device: the spec, the pin, the cuDNN version that chose it, and the architecture and the SM count of the device."""

    spec: ConvSpec
    arch: str
    pin: Any  # tilefish._cudnn.LegacyPin | EnginePin
    cudnn_version: int
    sm_count: int
    canary: "dict[str, Any] | None" = None  # {"seed": s, "y_digest": d}: made by export, checked by a replay

    @property
    def stem(self) -> str:
        return self.spec.stem

    @property
    def workspace_bytes(self) -> int:
        return self.spec.workspace_bytes

    @property
    def algo(self) -> Any:
        """The pin (what a plan step runs: a GEMM's is its 128 hex digits)."""
        return self.pin

    def entry(self) -> "dict[str, Any]":
        out = self.spec.entry(self.sm_count)
        out["arch"] = self.arch
        out["pin"] = self.pin.to_json()
        out["cudnn_version"] = self.cudnn_version
        if self.canary is not None:
            out["canary"] = dict(self.canary)
        return out


def _imported(spec: ConvSpec, arch: str, sms: int, version: int) -> Any:
    assert spec.algos
    table = table_for(spec.algos, arch, sms, version)
    return None if table is None else table.pins[spec.key]


def build_conv(spec: ConvSpec, *, arch: "int | str", device_index: "int | None" = None, canary: bool = False) -> BuiltConv:
    """Pin ``spec`` for ``arch``: ask cuDNN under the spec's posture, or take its table's pin for this device; either way the pin passes the checks of a
    run (``tilefish._cudnn.check_pin``) before it is recorded. With ``canary`` (what ``export`` does) the pin's canary is run and recorded too."""
    from . import _cudnn
    from ._build import parse_arch
    from ._library import _capability, _device_index
    from .errors import TilefishBuildError

    cap = parse_arch(arch)
    index = _device_index(device_index)
    have = _capability(index)
    if have != cap:
        raise TilefishBuildError(f"cannot pin '{spec.stem}' for sm_{cap}: cuDNN chooses for the device it runs on, and device {index} is sm_{have}")
    try:
        _cudnn.refuse_tf32_override()
    except TilefishRuntimeError as e:
        raise TilefishBuildError(f"cannot pin '{spec.stem}': {e}") from None
    version, sms = _cudnn.version(), _cudnn.sm_count(index)
    target = spec.sm_count_target
    if target is not None and target > sms:
        raise TilefishBuildError(f"cannot pin '{spec.stem}' on sm_{cap} with {sms} SMs: its declaration targets {target} SMs, more than the device has")
    if spec.algos:
        pin = _imported(spec, f"sm_{cap}", sms, version)
        if pin is None:
            have_tables = ", ".join(f"{t.arch} with {t.sm_count} SMs and cuDNN {t.cudnn_version}" for t in spec.algos)
            raise TilefishBuildError(
                f"cannot pin '{spec.stem}' on sm_{cap} with {sms} SMs and cuDNN {version}: its declaration imports its pins (algos=) and has a table only for "
                f"{have_tables}; give one for this device, or declare it without algos= to ask cuDNN"
            )
    else:
        if spec.api == "legacy" and target is not None:
            raise TilefishBuildError(
                f"cannot pin '{spec.stem}': the legacy API takes no SM count target, so its heuristic cannot choose for {target} SMs; import the algorithm "
                "chosen on the partition (algos=) with sm_count_target="
            )
        try:
            pin = _cudnn.pin_conv(spec.problem, heuristic=spec.heuristic, workspace_bytes=spec.workspace_bytes, device_index=index, sm_count_target=target).pin
        except TilefishRuntimeError as e:
            raise TilefishBuildError(f"cannot pin '{spec.stem}' on sm_{cap}: {e}") from e
    try:
        _cudnn.check_pin(spec.problem, pin, workspace_bytes=spec.workspace_bytes, device_index=index, sm_count_target=_run_target(spec, pin))
    except TilefishRuntimeError as e:
        source = "the imported" if spec.algos else "cuDNN's"
        raise TilefishBuildError(f"cannot pin '{spec.stem}' on sm_{cap}: {source} pin is refused: {e}") from e
    made = None
    if canary:
        made = {"seed": CANARY_SEED, "y_digest": canary_digest(spec, pin, CANARY_SEED, index)}
    return BuiltConv(spec, f"sm_{cap}", pin, version, sms, made)


def _run_target(spec: ConvSpec, pin: Any) -> "int | None":
    """The SM count target a run sets: the declaration's, on an engine of the backend API (a legacy call has none to set)."""
    return spec.sm_count_target if pin.api == "backend" else None


def canary_digest(spec: ConvSpec, pin: Any, seed: int, device_index: int) -> str:
    """The digest of the ``y`` the pin computes from the canary operands of ``seed`` at the spec's shape, on device ``device_index``
    (:mod:`tilefish._canary`).

    Everything is on one stream made for it (torch's pool: a non-blocking stream, ordered with no other, the NULL stream included): the operands made
    by the fill kernel, the zeroed ``y`` (a whole number of words), the workspace, the convolution, the digest kernel, and the copy of its 16 bytes back,
    which waits for that stream alone. Nothing of the operands or of ``y`` crosses to the host."""
    from . import _canary, _cudnn

    torch = _cudnn._torch()
    p = spec.problem
    size = _ELEMENT_BYTES[p.dtype]
    x_elements, w_elements = math.prod(p.x_dims), math.prod(p.w_dims)
    y_bytes = math.prod(p.y_dims) * size
    device = f"cuda:{device_index}"
    with torch.cuda.device(device_index):
        stream = torch.cuda.Stream(device=device_index)
        with torch.cuda.stream(stream):
            x = torch.empty(x_elements * size, dtype=torch.uint8, device=device)
            w = torch.empty(w_elements * size, dtype=torch.uint8, device=device)
            _canary.fill(x.data_ptr(), x_elements, p.dtype, seed, 0, stream.cuda_stream, device_index)
            _canary.fill(w.data_ptr(), w_elements, p.dtype, seed, 1, stream.cuda_stream, device_index)
            y = torch.zeros(-(-y_bytes // 4) * 4, dtype=torch.uint8, device=device)
            _cudnn.convolution(
                x.data_ptr(), w.data_ptr(), y.data_ptr(), p, pin, workspace_bytes=spec.workspace_bytes, stream=stream.cuda_stream,
                device_index=device_index, sm_count_target=_run_target(spec, pin),
            )  # fmt: skip
            sums = torch.zeros(2, dtype=torch.int64, device=device)
            _canary.digest(y.data_ptr(), y.numel() // 4, sums.data_ptr(), stream.cuda_stream, device_index)
            s1, s2 = (int(v) for v in sums.cpu())  # a copy on this stream, which it then waits for
    return _canary.hex_of(s1, s2)


_canary_lock = threading.Lock()
_canaries_passed: "set[tuple[Any, ...]]" = set()


def check_canary(built: BuiltConv, device_index: int) -> None:
    """Run the canary of a pin read from a manifest, once per process and device (refused: :class:`TilefishManifestError`; none recorded: nothing)."""
    canary = built.canary
    if canary is None:
        return
    spec = built.spec
    pin_json = json.dumps(built.pin.to_json(), sort_keys=True)
    key = (device_index, built.stem, built.arch, pin_json, built.cudnn_version, spec.workspace_bytes, canary["seed"], canary["y_digest"])
    with _canary_lock:
        if key in _canaries_passed:
            return
    from . import _cudnn

    torch = _cudnn._torch()
    with torch.cuda.device(device_index):
        if torch.cuda.is_current_stream_capturing():
            raise TilefishRuntimeError(
                f"the cuDNN convolution '{built.stem}' ({built.arch}) runs its canary at its first replay in the process, and this one is inside a CUDA graph "
                "capture: run it once before the capture"
            )
    got = canary_digest(spec, built.pin, canary["seed"], device_index)
    if got != canary["y_digest"]:
        raise TilefishManifestError(
            f"the canary of the cuDNN convolution '{built.stem}' ({built.arch}, {built.pin.what}) gives y with digest {got} here, and the manifest recorded "
            f"{canary['y_digest']} (seed {canary['seed']}): this process's cuDNN (version {built.cudnn_version} too) or this device runs other kernels for "
            "the pin, so it is refused; export the manifest again with this cuDNN build, or run with the build that made it"
        )
    with _canary_lock:
        _canaries_passed.add(key)


def built_from_manifest(manifest: Any, spec: ConvSpec, arch: str, sm_count: int, device_index: "int | None" = None) -> BuiltConv:
    """The pinned convolution of ``spec`` from a manifest, for a device of ``arch`` with ``sm_count`` SMs, after the checks of a replay: the entry is
    there, was pinned on a device of this SM count, is what the declaration asks for, has a pin of the declaration's API (the table's, when it imports),
    and the process has the cuDNN version that chose it and no ``NVIDIA_TF32_OVERRIDE``."""
    from . import _cudnn
    from .aot import _first_difference, _show

    entry = manifest.library_entry(spec.stem, arch)
    if entry is None:
        have = sorted(a for (s, a) in ((e["stem"], e["arch"]) for e in manifest.library_entries) if s == spec.stem)
        raise TilefishManifestError(
            f"the manifest in {str(manifest.directory)!r} has no library entry for '{spec.stem}' on {arch}" + (f" (it has {', '.join(have)})" if have else "")
        )
    if entry.get("impl") != IMPL:
        raise TilefishManifestError(f"the library entry for '{spec.stem}' ({arch}) is a {entry.get('impl')}, not a {IMPL}")
    pinned_on = entry["sm_count"]
    if pinned_on != sm_count:
        raise TilefishManifestError(
            f"the manifest pinned the cuDNN convolution '{spec.stem}' ({arch}) on a device with {pinned_on} SMs, but this device has {sm_count}: cuDNN chooses "
            "for the multiprocessors it runs on, so the pin is refused; re-export the manifest on this device (or on one with the same SM count)"
        )
    expected = json.loads(json.dumps(spec.entry(sm_count)))
    got = {k: entry.get(k) for k in expected}
    if "sm_count_target" in entry and "sm_count_target" not in expected:
        got["sm_count_target"], expected["sm_count_target"] = entry["sm_count_target"], None
    found = _first_difference(expected, got, "")
    if found is not None:
        path, want, have_value = found
        raise TilefishManifestError(
            f"the library entry for '{spec.stem}' ({arch}) is not what the declaration asks for: first difference at {path}: the manifest has "
            f"{_show(have_value)}, the declaration gives {_show(want)}"
        )
    pin = _cudnn.pin_of(entry["pin"])
    if pin.api != spec.api:
        raise TilefishManifestError(f"the library entry for '{spec.stem}' ({arch}) has a {pin.api} pin, and the declaration's api is {spec.api!r}")
    pinned_by = entry["cudnn_version"]
    here = _cudnn.version()
    if here != pinned_by:
        raise TilefishManifestError(
            f"the manifest pinned the cuDNN convolution '{spec.stem}' ({arch}) with cuDNN {pinned_by}, but this process has cuDNN {here}: a pin names other "
            "kernels in another version, so it is refused (export the manifest again with this cuDNN, or run with the version that made it)"
        )
    try:
        _cudnn.refuse_tf32_override()
    except TilefishRuntimeError as e:
        raise TilefishManifestError(f"the cuDNN convolution '{spec.stem}' ({arch}) is not run here: {e}") from None
    if spec.algos:
        want = _imported(spec, arch, sm_count, pinned_by)
        if want != pin:
            raise TilefishManifestError(
                f"the library entry for '{spec.stem}' ({arch}) has the pin {pin.what}, but the declaration imports "
                + (want.what if want is not None else f"none for {arch} with {sm_count} SMs and cuDNN {pinned_by}")
            )
    built = BuiltConv(spec, arch, pin, pinned_by, pinned_on, entry.get("canary"))
    if built.canary is not None:
        from ._library import _device_index

        check_canary(built, _device_index(device_index))
    return built


def check_built(built: BuiltConv, device_index: int) -> int:
    """The workspace the pin of ``built`` needs on the device (refused: :class:`TilefishRuntimeError`): once per plan bucket, before a capture."""
    from . import _cudnn

    spec = built.spec
    return _cudnn.check_pin(spec.problem, built.pin, workspace_bytes=spec.workspace_bytes, device_index=device_index, sm_count_target=_run_target(spec, built.pin))


# -- reading an entry back ---------------------------------------------------------------------------------------------------------


def _bad(where: str, message: str) -> Any:
    raise TilefishManifestError(f"library entry {where}: {message}")


def _ints(entry: "dict[str, Any]", key: str, count: int, low: int, where: str) -> "list[int]":
    v = entry[key]
    if not isinstance(v, list) or len(v) != count or any(not sv.is_int(x) or not low <= x <= sv.MAX_ELEMENTS for x in v):
        _bad(where, f"{key} must be a list of {count} ints from {low} to {sv.MAX_ELEMENTS} (one per spatial axis), got {v!r}")
    return v


def problem_of_entry(entry: "dict[str, Any]", where: str = "#0") -> Any:
    """The :class:`tilefish._cudnn.Problem` an entry says (the entry was validated: :func:`validate_entry`)."""
    from . import _cudnn

    return _cudnn.Problem(
        entry["dtypes"]["x"], entry["layout"], entry["n"], entry["c"], entry["k"], entry["groups"], tuple(entry["input"]), tuple(entry["filter"]),
        tuple(entry["stride"]), tuple(entry["padding"]), tuple(entry["dilation"]), entry["math"], entry["alignment"],
    )  # fmt: skip


def validate_entry(entry: Any, index: int = 0) -> None:
    """Strict reading of one ``cudnn.conv_fwd`` entry of the manifest's ``library`` section (raises :class:`TilefishManifestError`).

    Every field must be present and exactly typed, and the redundant ones must agree: the problem (``dtypes``, ``layout``, ``n``, ``c``, ``k``,
    ``groups``, the spatial lists, ``math``, ``alignment``) is one cuDNN takes, ``output`` is its output, the shapes of ``x``, ``w`` and ``out`` are its
    shapes in the layout under the buckets, every pointer is aligned to ``alignment``, the pin is of the API its ``choice`` asks (``v7``: legacy;
    ``instant``, ``mode_b``: backend), an ``sm_count_target`` is at most ``sm_count`` (on a legacy pin only when it was imported), and a ``canary`` is
    ``{"seed": s, "y_digest": d}`` (``s`` from 0 to 2**32 - 1, ``d`` 32 lowercase hex digits)."""
    from . import _cudnn

    where = f"#{index}"
    if not isinstance(entry, dict):
        _bad(where, f"is not an object: {entry!r}")
    where = f"#{index} ({entry.get('stem')!r})"
    missing = [k for k in ENTRY_KEYS if k not in entry]
    unknown = sorted(str(k) for k in set(entry) - set(ENTRY_KEYS) - set(OPTIONAL_FIELDS))
    if missing or unknown:
        _bad(
            where,
            f"a {IMPL} entry must have exactly the fields {ENTRY_KEYS} and may have {list(OPTIONAL_FIELDS)}"
            + (f"; missing {missing}" if missing else "")
            + (f"; unknown {unknown}" if unknown else ""),
        )
    try:
        stem = entry["stem"]
        if not isinstance(stem, str) or not sv._STEM.match(stem):
            _bad(where, f"stem must look like 'conv' or 'conv.T646', got {stem!r}")
        sv._ident(entry["op"], "op")
        arch = entry["arch"]
        if not isinstance(arch, str) or not arch.startswith("sm_") or not arch[3:].isdigit() or len(arch[3:]) not in _ARCH_LEN:
            _bad(where, f"arch must look like 'sm_103', got {arch!r}")
        if entry["impl"] != IMPL:
            _bad(where, f"impl must be {IMPL!r}, got {entry['impl']!r}")
        dtypes = entry["dtypes"]
        dtype = dtypes.get("x") if isinstance(dtypes, dict) else None
        if dtype not in _cudnn.DTYPES or dtypes != {"x": dtype, "w": dtype, "y": dtype}:
            _bad(where, f"dtypes must be {{'x': T, 'w': T, 'y': T}} with T one of {list(_cudnn.DTYPES)}, got {dtypes!r}")
        if entry["compute_type"] != COMPUTE_TYPE:
            _bad(where, f"compute_type must be {COMPUTE_TYPE!r}, got {entry['compute_type']!r}")
        if entry["math"] not in _cudnn.MATHS:
            _bad(where, f"math must be one of {list(_cudnn.MATHS)}, got {entry['math']!r}")
        if entry["layout"] not in _cudnn.LAYOUTS:
            _bad(where, f"layout must be one of {list(_cudnn.LAYOUTS)}, got {entry['layout']!r}")
        for key in ("n", "c", "k", "groups"):
            if not sv.is_int(entry[key]) or not 1 <= entry[key] <= sv.MAX_ELEMENTS:
                _bad(where, f"{key} must be an int from 1 to {sv.MAX_ELEMENTS}, got {entry[key]!r}")
        inp = entry["input"]
        if not isinstance(inp, list) or not 1 <= len(inp) <= 3:
            _bad(where, f"input must be a list of 1 to 3 spatial sizes, got {inp!r}")
        count = len(inp)
        _ints(entry, "input", count, 1, where)
        _ints(entry, "filter", count, 1, where)
        _ints(entry, "output", count, 1, where)
        _ints(entry, "stride", count, 1, where)
        _ints(entry, "padding", count, 0, where)
        _ints(entry, "dilation", count, 1, where)
        if not sv.is_int(entry["alignment"]) or entry["alignment"] not in _cudnn.ALIGNMENTS:
            _bad(where, f"alignment must be one of {list(_cudnn.ALIGNMENTS)}, got {entry['alignment']!r}")
        try:
            problem = problem_of_entry(entry, where)
        except TilefishRuntimeError as e:
            return _bad(where, f"not a convolution cuDNN takes: {str(e).removeprefix('cuDNN convolution: ')}")
        if list(problem.output) != entry["output"]:
            _bad(where, f"output must be the output of the convolution, {list(problem.output)}, got {entry['output']!r}")
        ws = entry["workspace_bytes"]
        if not sv.is_int(ws) or not 1 <= ws <= MAX_WORKSPACE_BYTES:
            _bad(where, f"workspace_bytes must be an int from 1 to 2**40, got {ws!r}")
        if entry["choice"] not in CHOICES:
            _bad(where, f"choice must be one of {list(CHOICES)}, got {entry['choice']!r}")
        pin = entry["pin"]
        if not isinstance(pin, dict) or (pin.get("api"), set(pin)) not in (("legacy", {"api", "algo"}), ("backend", {"api", "engine", "knobs"})):
            _bad(where, f"pin must be {{'api': 'legacy', 'algo': a}} or {{'api': 'backend', 'engine': e, 'knobs': [[t, v], ...]}}, got {pin!r}")
        if pin["api"] == "legacy" and not sv.is_int(pin["algo"]):
            _bad(where, f"the algorithm of a legacy pin must be an int, got {pin['algo']!r}")
        if pin["api"] == "backend" and (
            not sv.is_int(pin["engine"]) or not isinstance(pin["knobs"], list) or any(not isinstance(kv, list) or len(kv) != 2 or not all(sv.is_int(x) for x in kv) for kv in pin["knobs"])
        ):
            _bad(where, f"a backend pin has an int engine and knobs [[type, value], ...] of ints, got {pin!r}")
        try:
            parsed = _cudnn.pin_of(pin)
        except TilefishRuntimeError as e:
            return _bad(where, str(e))
        wanted_api = _cudnn.HEURISTICS.get(entry["choice"])
        if wanted_api is not None and parsed.api != wanted_api:
            _bad(where, f"the choice {entry['choice']!r} pins a {wanted_api} pin, got a {parsed.api} one")
        if not sv.is_int(entry["cudnn_version"]) or entry["cudnn_version"] < 1:
            _bad(where, f"cudnn_version must be a positive int, got {entry['cudnn_version']!r}")
        if not sv.is_int(entry["sm_count"]) or entry["sm_count"] < 1:
            _bad(where, f"sm_count must be a positive int (the multiprocessors of the device the pin was chosen on), got {entry['sm_count']!r}")
        if "sm_count_target" in entry:
            target = entry["sm_count_target"]
            if not sv.is_int(target) or not 1 <= target <= entry["sm_count"]:
                _bad(where, f"sm_count_target is absent or an int from 1 to sm_count ({entry['sm_count']}), got {target!r}")
            if parsed.api == "legacy" and entry["choice"] != "imported":
                _bad(where, "a legacy pin has an sm_count_target only when it was imported (the legacy API takes no target)")
        if "canary" in entry:
            canary = entry["canary"]
            if isinstance(canary, dict) and "y_sha256" in canary:
                _bad(where, "a canary of tilefish's first form ('y_sha256': host-made operands, a host sha256) is no longer read: export the manifest again")
            if (
                not isinstance(canary, dict)
                or set(canary) != {"seed", "y_digest"}
                or not sv.is_int(canary["seed"])
                or not 0 <= canary["seed"] <= MAX_CANARY_SEED
                or not isinstance(canary["y_digest"], str)
                or not _DIGEST32.match(canary["y_digest"])
            ):
                _bad(where, f"canary is absent or {{'seed': an int from 0 to 2**32 - 1, 'y_digest': 32 lowercase hex digits}}, got {canary!r}")
        if not (isinstance(entry["declaration_digest"], str) and sv._DIGEST.match(entry["declaration_digest"])):
            _bad(where, f"declaration_digest must be 64 lowercase hex digits (a sha256), got {entry['declaration_digest']!r}")
        buckets = entry["buckets"]
        if not isinstance(buckets, dict):
            _bad(where, "buckets must be an object")
        for name, value in buckets.items():
            sv._ident(name, "bucket dimension name")
            sv._int(value, f"bucket value of {name}", 1)
        abi = entry["abi"]
        sv._keys(abi, "abi", ["params"])
        params = abi["params"]
        shapes = [problem.tilefish_shape("x"), problem.tilefish_shape("w"), problem.tilefish_shape("y")]
        if not isinstance(params, list) or len(params) != 3:
            _bad(where, "abi.params must list 3 pointers (x, w, out in this order)")
        seen: "set[str]" = set()
        from ._library import _eval_static

        for i, (p, shape) in enumerate(zip(params, shapes)):
            sv._keys(p, f"abi.params[{i}]", ["index", "name", "kind", "dtype", "access", "shape"])
            sv._ident(p["name"], f"abi.params[{i}].name")
            access = "out" if i == 2 else "in"
            if not sv.is_int(p["index"]) or p["index"] != i or p["kind"] != "pointer" or p["dtype"] != dtype or p["access"] != access:
                _bad(where, f"abi.params[{i}] must be the {dtype} pointer #{i} with access {access}")
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
                    _bad(where, f"abi.params[{i}].shape[{axis}] is {entry_dim!r} = {value}, but the problem says {size}")
        _validate_preconditions(entry, params, buckets, where)
        if stem != entry["op"] + "".join(f".{k_}{v}" for k_, v in buckets.items()):
            _bad(where, f"stem {stem!r} does not match op {entry['op']!r} and buckets {buckets}")
    except TilefishSpecError as e:
        raise TilefishManifestError(f"library entry {where}: {str(e).removeprefix('malformed compile spec: ')}") from None
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError, RecursionError) as e:
        raise TilefishManifestError(f"library entry {where} is malformed: {type(e).__name__}: {e}") from None


def _validate_preconditions(entry: "dict[str, Any]", params: "list[Any]", buckets: "dict[str, int]", where: str) -> None:
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
            if pre["index"] in aligned or pre["bytes"] != entry["alignment"] or not 0 <= pre["index"] < len(params) or params[pre["index"]]["name"] != pre["arg"]:
                _bad(where, f"pointer_alignment must be the entry's alignment ({entry['alignment']} bytes), once per pointer, on the pointer it names")
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


# -- running -----------------------------------------------------------------------------------------------------------------------


def run_conv(built: BuiltConv, tensors: "dict[str, Any]", out: Any, stream: int, device_index: int) -> None:
    x, w = tensors.values()
    run_conv_ptrs(built, int(x.data_ptr()), int(w.data_ptr()), int(out.data_ptr()), stream, device_index)


def run_conv_ptrs(built: BuiltConv, x_ptr: int, w_ptr: int, out_ptr: int, stream: int, device_index: int, workspace_ptr: "int | None" = None) -> None:
    """:func:`run_conv` on device addresses; with ``workspace_ptr`` (``workspace_bytes`` bytes, 256-byte aligned) the call allocates nothing."""
    from . import _cudnn

    spec = built.spec
    _cudnn.convolution(
        x_ptr, w_ptr, out_ptr, spec.problem, built.pin, workspace_bytes=spec.workspace_bytes, stream=stream, device_index=device_index,
        workspace_ptr=workspace_ptr, sm_count_target=_run_target(spec, built.pin),
    )  # fmt: skip


def reference_conv(op: "LibraryOp", tensors: "dict[str, Any]") -> Any:
    """The result in torch, as the semantics say: the convolution in f64 (``torch.nn.functional.conv*d`` of the operands in the NCHW order), rounded once
    to the data type, back in the declared layout."""
    import torch.nn.functional as F

    o: ConvOptions = op.conv  # type: ignore[attr-defined]
    x, w = tensors.values()
    spatial = x.dim() - 2
    if o.layout == "NHWC":
        to_nchw = (0, spatial + 1, *range(1, spatial + 1))
        x, w = x.permute(*to_nchw), w.permute(*to_nchw)
    fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[spatial]
    y = fn(x.double().cpu(), w.double().cpu(), None, o.stride, o.padding, o.dilation, o.groups)
    if o.layout == "NHWC":
        y = y.permute(0, *range(2, spatial + 2), 1)
    return y.to(x.dtype).contiguous()


__all__ = [
    "BuiltConv",
    "ConvSpec",
    "CudnnTable",
    "build_conv",
    "built_from_manifest",
    "canary_bytes",
    "canary_operands",
    "check_canary",
    "conv_specs",
    "validate_entry",
]
