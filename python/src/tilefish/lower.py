"""Lowering of ``@tilefish.kernel`` bodies to Triton source (M2a).

``lower(kernel_op)`` turns the typed SSA IR of a kernel into one ``@triton.jit`` function in a
complete Python module, and describes that function as an :class:`~tilefish.decorators.ExternOp`
so :func:`tilefish.aot.compile_specs` yields the ahead-of-time compile inputs for it unchanged.
Nothing here imports triton or compiles anything. The output is deterministic: the same IR
gives the same bytes.

Two templates exist; a kernel that fits neither raises :class:`~tilefish.errors.TilefishLoweringError`
naming the rule that failed (``rule`` attribute). Without ``template=`` the elementwise template
is tried first, then the row template.

**row** (``R`` rows, one program per row, grid ``(R,)``). Every tensor is ``[R, D]``, ``[D]``
(shared by all rows), ``[R]`` or ``[R, 1]`` (one value per row). Reductions (``sum``, ``mean``,
``max``) run over the last axis of an ``[R, D]`` tensor only. ``D`` is static, or bucketed (one
compile spec per bucket); ``R`` is dynamic and only appears in the grid. Prologue::

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    offs = row * D + cols

``D`` and ``BLOCK_D = next_pow2(D)`` are ``tl.constexpr`` parameters. ``[R, D]`` tensors are
addressed with ``offs``, ``[D]`` with ``cols``, per-row tensors with ``row``. ``[D]`` tensors are
read-only (storing one from every row program would race). ``num_warps`` is one warp per 256
lanes of ``BLOCK_D``: ``clamp(BLOCK_D // 256, 1, 16)`` (a bucketed ``D`` uses its largest
bucket, because a spec set has one ``num_warps``); ``@tilefish.kernel(num_warps=...)`` or
``lower(num_warps=...)`` overrides it.

**elementwise** (grid ``(cdiv(n, 1024),)``). Every tensor has the same shape, flattened to
``n = prod(shape)``; no broadcasting except python and ``const`` scalars; no reductions. ``n`` is a
runtime ``i32`` parameter whose dimension is that product (``3072*M``), so a spec says how to
compute it and ``compile_specs`` emits ``tt.divisibility`` when the product is provably a multiple
of 16. When the product is not affine (two symbolic dimensions), ``n`` gets a fresh dimension that
carries the largest power of two (up to 1024) the factors prove between them.
``BLOCK: tl.constexpr = 1024`` and ``num_warps = 4``. Prologue::

    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

Both templates: offsets are ``i32``, so a tensor may have at most ``2**31 - 1`` elements. A tensor
that provably has more is rejected (rule ``index-range``; the largest bucket of a bucketed
dimension counts, and every dimension is at least 1). A tensor whose size has no static bound
(a dynamic dimension) gets a ``max_elements`` precondition in every spec, holding the argument,
its shape and the bound, which a host checks at launch; a tensor that is statically small gets
none. The generated code is the same either way. No loops, no strides (a View, a ``split`` or a
call to a declared operator is rejected), ``num_stages = 3`` in the spec (irrelevant without loops).

Parameters of the generated function, in this order: the kernel's own parameters under their own
names (tensors are pointers, ``const`` parameters are ``tl.constexpr`` with the declared default
as the spec's constant, runtime dimension parameters are ``i32``), one output pointer per returned
value (``out``, or ``out0``, ``out1``, ... for several), then the template's parameters. A
generated name that collides with a kernel parameter is suffixed (``mask_1``). Constexpr values come
from the spec (``compile_specs(lowered.extern)``), not from defaults in the source.

A kernel parameter whose name the generated module itself uses (``tl``, ``triton``, ``float``) or
that Triton's JIT launcher takes as a launch option (``grid``, ``warmup``, ``debug``, ``num_warps``,
``num_stages``, ``num_ctas``, ``maxnreg``) is renamed with a numeric suffix (``tl_1``), in the
generated function, the extern stub and every spec alike: a spec's signature must name the
function's own parameters, so the spec cannot keep the original name. The module header says which
parameter became which; every other parameter keeps its name.

The elementwise ``n`` is recorded in the spec as ``count_of``: the tensor argument whose element
count (the product of its shape entries) the scalar carries, so a host computes it without knowing
the kernel. When the product is affine the scalar's ``dim`` is that product (the spec validator
checks it); otherwise ``dim`` is a fresh name and ``count_of`` is the only definition.

Numerics contract
-----------------
Every choice below is part of the contract; changing one later is a breaking change. Every PTX fact in
it was checked on the PTX that Triton 3.5.1 and ptxas 12.9.86 produce for sm_103, and is asserted on a GPU
by ``tests/test_gpu_numerics.py`` (the numbers in the notes are measured there).

==========================  ==========================================================  =====================================
IR                          Generated source                                            Numerics
==========================  ==========================================================  =====================================
load of a tensor            ``tl.load(p + offs, mask=mask, other=0)``                   storage dtype, never pre-cast; masked
                                                                                         lanes read 0. ``[R]``/``[R, 1]``:
                                                                                         ``tl.load(p + row)``, no mask.
``x.to(dt)``                ``v.to(tl.<dt>)``, exactly where the IR has the cast        Triton's default conversion, with the
                                                                                         out-of-range behavior of note 1; no
                                                                                         ``fp_downcast_rounding`` argument.
                                                                                         A narrowing cast gets ``# rounding
                                                                                         point``. Nothing else is ever cast.
store                       ``tl.store(p + offs, v, mask=mask)``                        the IR value as is (its dtype equals
                                                                                         the target's). ``[R]``: no mask.
                                                                                         An ``Out`` or ``InOut`` parameter
                                                                                         is written once, with its final
                                                                                         version; earlier stores are elided
                                                                                         and a self-store (``x[...] = x``)
                                                                                         writes nothing.
                                                                                         A returned value is written to its
                                                                                         own output pointer (a copy).
python scalar / ``const``   a float literal written with ``repr()`` (``inf`` and           the scalar takes the dtype of the
                            ``nan`` as ``float("inf")``), an int literal, or the            tensor it meets and is rounded to it:
                            ``const`` parameter's name                                      no promotion (note 3).
``a + b``, ``a - b``,       ``a + b``, ``a - b``, ``a * b``                                 dtypes f32, f16, bf16, i32, i64.
``a * b``                                                                                    ``a * b + c`` is one FMA, one
                                                                                         rounding (note 2).
``a / b``                   ``a / b``                                                       f32 only (note 4): ``div.full.f32``,
                                                                                         not IEEE division.
``rsqrt(x)``                ``tl.rsqrt(x)``                                                 f32 only; approximate (note 5).
``exp(x)``                  ``tl.exp(x)``                                                   f32 only; fast exp (note 5).
``sqrt(x)``                 ``tl.sqrt_rn(x)``                                               f32 only; IEEE round to nearest, but
                                                                                         subnormals flush to zero (note 5).
``silu(x)``                 ``x * tl.sigmoid(x)``                                           f32 only; ``tl.sigmoid`` is
                                                                                         ``1 / (1 + tl.exp(-x))``.
``gelu(x)``                 ``0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))``       f32 only; exact erf form, evaluated
                                                                                         left to right (note 5).
``sum(x, axis=-1)``         ``tl.sum(tl.where(mask, x, 0.0), axis=0)``                      f32 only. The reduction is over all
                                                                                         ``BLOCK_D`` lanes; masked lanes are
                                                                                         set to 0 explicitly (intermediates
                                                                                         such as ``exp(x)`` or ``x - mean``
                                                                                         are not 0 there). The order is
                                                                                         Triton's, and depends on
                                                                                         ``num_warps`` (note 6).
``mean(x, axis=-1)``        ``tl.sum(tl.where(mask, x, 0.0), axis=0) / D``                  f32 only; ``D`` is the constexpr:
                                                                                         ``sum * f32(1 / D)`` (note 4).
``max(x, axis=-1)``         ``tl.max(tl.where(mask, x, float("-inf")), axis=0)``            f32 only; masked lanes are ``-inf``
                                                                                         (``float("-inf")`` compiles: JIT and
                                                                                         AOT, ``0fFF800000`` in the PTX).
==========================  ==========================================================  =====================================

Reductions are f32 only because Triton accumulates a narrower dtype in the narrow dtype, which
the IR does not say; cast to f32 first (the README kernel does). Every SSA value is one local
(``v0``, ``v1``, ... in IR order: the loaded parameters first, then one per node), and a comment
marks each rounding point.

Notes (the facts behind the table):

1. **Casts.** Float to ``f16``/``bf16`` is ``cvt.rn.f16x2.f32`` / ``cvt.rn.bf16x2.f32``: round to nearest even,
   overflow to infinity (65520 is the first f32 that becomes ``inf`` in f16), NaN stays NaN; the same bytes as
   torch's cast. Float to integer truncates toward zero and, out of range, **saturates**: ``i32`` is
   ``cvt.rzi.s32.f32`` (NaN is 0), ``i64`` is ``cvt.rzi.s64.f32`` (NaN is ``-2**63``, not 0). Float to ``u8`` is
   ``cvt.rzi.u16.f32`` followed by keeping the **low byte**: the value is truncated and clamped to [0, 65535]
   (negative and NaN give 0), then reduced modulo 256, so ``-1.5`` is 0, ``255.5`` is 255, ``256`` is 0,
   ``300`` is 44, ``1000`` is 232 and anything from 65535 up (infinity included) is 255. It neither
   saturates at 255 nor wraps like C. Float to ``e4m3`` / ``e5m2`` is ``cvt.rn.satfinite.e4m3x2.f32`` /
   ``cvt.rn.satfinite.e5m2x2.f32``: round to nearest even and **saturate** at the largest finite value (448,
   57344), infinities included; NaN stays NaN. torch's ``.to`` differs on out-of-range values (``-1.5``
   to ``u8`` is 255, 1000 to ``e4m3`` is NaN, 1e5 to ``e5m2`` is ``inf``), so the ``reference`` mode implements
   this table instead of torch's cast.
2. **FMA contraction.** Triton compiles with ``enable_fp_fusion=True`` (``compile_specs`` does not change that), so
   ``a * b + c`` is one FMA with one rounding: ``fma.rn.f32`` for f32 and ``fma.rn.bf16x2``, a native packed bf16
   FMA, for bf16. Without contraction it is ``mul.rn`` then ``add.rn`` (two roundings). The two results differ
   in 23% of random elements (22.9% in bf16, 23.4% in f32), and the contracted one is exactly the correctly
   rounded value of the exact product and sum. Contraction also changes the bytes of the goldens that have a
   contractable ``a * b + c`` (layernorm: 3 to 8 of 197376 elements, one bf16 ULP; the README rmsnorm: 1 to 2 of 657920);
   softmax, gelu and swiglu come out byte-identical either way.
3. **Scalars.** Triton 3.5.1 keeps the dtype of the tensor: a python float or a constexpr float combined with
   a bf16 tensor is bf16 (``tl.static_assert`` in a compiled kernel says so), and the scalar is rounded to
   bf16 first (0.1 becomes 0.10009765625). Nothing is promoted to f32: ``x * 0.1 + eps`` on bf16 is a single
   ``fma.rn.bf16x2`` (or ``mul.rn.bf16x2`` and ``add.rn.bf16x2`` without contraction), with no ``cvt`` to f32.
   The one exception is ``/``, which Triton does in f32 for bf16 and f16 and returns as f32; that is why a
   division of those dtypes is rejected.
4. **Division.** ``a / b`` is ``div.full.f32``, never ``div.rn.f32``: the numerator times an approximate
   reciprocal, at most 2 ULP from the correctly rounded quotient (measured: 2 ULP, 29% of random elements
   differ, relative error below 1.8e-7). With a constant divisor (``x / D`` with ``D`` a constexpr or a
   literal) the PTX is the same ``div.full.f32`` (Triton does not strength-reduce it), but the machine code
   behind it is a multiply by the f32 reciprocal: the result is bit-identical to ``x * f32(1 / c)`` for
   every constant tried (4M inputs each), and differs from correctly rounded division in 15% to 58%
   of random inputs (33% for 768, 58% for 1000); only a power of two is exact. ``mean`` is ``sum * f32(1 / D)``
   for the same reason.
5. **Math functions.** ``tl.sqrt_rn`` is ``sqrt.rn.ftz.f32``: the correctly rounded square root for every normal
   input (the same bytes as torch's), but a subnormal input is treated as zero, so ``sqrt(1e-40)`` is 0, not
   1e-20; it is not ``sqrt.rn.f32``. ``tl.rsqrt`` is ``rsqrt.approx.ftz.f32`` (up to 2 ULP, 22% of inputs off the
   correctly rounded value; a subnormal input is zero). ``tl.exp`` is ``ex2.approx.f32`` of
   ``x * 1.4426950216293335`` (log2 e as an f32): the product is rounded before the exponentiation, so the
   error grows with ``|x|`` (3 ULP for ``|x| < 2``, 27 ULP up to ``|x| = 40``). ``tl.math.erf`` (also
   ``tl.erf``) exists and links: libdevice's ``__nv_erff`` is inlined (no ``call`` is left in the PTX) as a
   polynomial of ``fma.rn.ftz.f32`` and ``ex2.approx.ftz.f32``, which flush subnormals; its FMAs are explicit, so
   they do not depend on ``enable_fp_fusion``. The exact gelu cancels for large negative inputs (``1 + erf`` is
   a difference of two numbers near 1), so its relative error there is large (128 bf16 ULP on values near
   zero), though its absolute error stays below 1.6e-5.
6. **Reduction order and ``num_warps``.** The tree a reduction adds in depends on how many warps the block has,
   so the same kernel on the same data is not byte-identical across ``num_warps`` (129 rows with D = 96: 17 to 59
   rows differ from the 1-warp result in f32, depending on the kernel; with D = 1000: 28 to 125), though
   always within a few ULP of the others and of torch's sum. A bucketed ``D`` has one ``num_warps`` for all its
   binaries, from its largest bucket, so a small bucket runs with more warps than its own size would get (a
   ``D = 96`` bucket of a kernel whose largest bucket is 2048 runs with 8 warps instead of 1: 25 of 129 rows
   differ in the last bits). The result is correct either way, only a different, equally valid rounding. A
   spec fixes ``num_warps``, so ``jit``, ``replay`` and a generated host all agree.
"""

from __future__ import annotations

import itertools
import keyword
import math
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Iterable

from ._specvalid import MAX_ELEMENTS
from .aot import CompileSpec, compile_specs
from .decorators import ExternOp, KernelOp
from .dims import Dim, DimExpr, as_dim, cdiv, next_pow2
from .dtypes import BlockScaled
from .errors import TilefishLoweringError, TilefishTypeError
from .ir import Graph, Node, Scalar
from .signature import MISSING, Param, Signature
from .trace import Value

ELEMENTWISE_BLOCK = 1024
ELEMENTWISE_NUM_WARPS = 4
ROW_LANES_PER_WARP = 256
ROW_MAX_WARPS = 16
MAX_BLOCK = 1 << 20  # Triton's limit on the number of elements of a tensor
NUM_STAGES = 3
NUM_WARPS = (1, 2, 4, 8, 16, 32)
DEFAULT_MODULE = "tilefish_generated"
TEMPLATES = ("elementwise", "row")
INDEX_MAX = MAX_ELEMENTS  # offsets are i32: a tensor has at most 2**31 - 1 elements
# Names a kernel parameter cannot keep in the generated function: what the module itself uses, and what
# Triton's JIT launcher takes as launch options (a parameter of that name could not be passed by keyword).
_RESERVED = frozenset({"tl", "triton", "float", "grid", "warmup", "debug", "num_warps", "num_stages", "num_ctas", "maxnreg"})

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")  # ASCII only, like the identifiers a compile spec accepts
_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\Z")

# the Triton spelling of each dtype the lowering handles
_TL = {
    "bf16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
    "e4m3": "float8e4nv",
    "e5m2": "float8e5",
}
_ARITHMETIC = frozenset({"f32", "bf16", "f16", "i32", "i64"})
_MATH = frozenset({"f32"})
_BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_UNARY = ("rsqrt", "exp", "sqrt", "silu", "gelu")
_REDUCTIONS = ("sum", "mean", "max")

_FULL, _COL, _ROW = "full", "col", "row"  # row template: [R, D], [D], [R] / [R, 1]


# -- the result ---------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LoweredKernel:
    """The generated Triton source of one kernel, and a typed stub describing its parameters.

    ``extern`` is an :class:`~tilefish.decorators.ExternOp` for the generated function: pass it to
    :func:`tilefish.aot.compile_specs` to get its compile specs (one per bucket of a bucketed
    ``D``). Its triton reference is ``module:fn_name``. ``kernel`` is the ``@tilefish.kernel`` it was
    lowered from, which the ``reference`` mode of :func:`tilefish.run` interprets.
    """

    name: str
    fn_name: str
    module: str
    template: str  # "row" | "elementwise"
    source: str
    extern: ExternOp
    num_warps: int
    kernel: "KernelOp | None" = None

    def run(self, /, *args: Any, mode: str = "jit", **kwargs: Any) -> Any:
        """``tilefish.run(self, *args, mode=mode, **kwargs)``."""
        from .runtime import run

        return run(self, *args, mode=mode, **kwargs)

    @property
    def ref(self) -> str:
        """``"module:function"``, the Triton reference of the generated kernel."""
        return f"{self.module}:{self.fn_name}"

    @property
    def grid(self) -> str:
        """The launch grid as an expression over dimension names, e.g. ``(M,)``."""
        return self.extern.grid_string or ""

    @property
    def filename(self) -> str:
        """The module's path relative to a source root: ``tilefish_generated/rmsnorm.py``."""
        return self.module.replace(".", "/") + ".py"

    def specs(self) -> "list[CompileSpec]":
        """``compile_specs(self.extern)``."""
        return compile_specs(self.extern)

    def write(self, directory: "str | pathlib.Path") -> pathlib.Path:
        """Write the source to ``<directory>/<module path>.py`` (creating folders) and return the path.

        With ``directory`` on ``sys.path`` the module imports as ``self.module``.
        """
        path = pathlib.Path(directory) / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.source.encode("utf-8"))
        return path

    def __repr__(self) -> str:
        return f"<LoweredKernel {self.name}: {self.template} template, {self.ref}>"


# -- errors -------------------------------------------------------------------------------------


def _fail(kernel: str, template: "str | None", rule: str, detail: str) -> TilefishLoweringError:
    where = f"kernel '{kernel}'" + (f" with the {template} template" if template else "")
    return TilefishLoweringError(f"cannot lower {where}: rule '{rule}' failed: {detail}", rule=rule)


def _label(v: Value) -> str:
    return f"parameter '{v.name}' ({v.type})" if v.is_param else f"{v.ref} ({v.type})"


def _all_values(graph: Graph) -> "list[Value]":
    """Every tensor of the kernel once, in IR order: parameters, then node outputs, then outputs."""
    seen: dict[Value, None] = {}
    for v in graph.inputs:
        seen.setdefault(v, None)
    for n in graph.nodes:
        for v in n.outputs:
            seen.setdefault(v, None)
    for v in graph.outputs:
        seen.setdefault(v, None)
    return list(seen)


def _is_one(d: DimExpr) -> bool:
    return d.is_static and d.value == 1


def _shape_text(shape: "Iterable[DimExpr]") -> str:
    return "[" + ", ".join(str(d) for d in shape) + "]"


# -- how many elements a tensor can have ---------------------------------------------------------------


def _atom_bounds(atom: DimExpr) -> "tuple[int, int | None]":
    """Smallest and largest value of a symbolic dimension (``None``: no upper bound). Dimensions are positive."""
    if isinstance(atom, Dim):
        if atom.buckets:
            return min(atom.buckets), max(atom.buckets)  # a binary exists for these sizes only
        return (atom.divisible_by or 1), None
    num, den = getattr(atom, "num", None), getattr(atom, "den", None)
    if num is None or den is None:  # pragma: no cover - the other atoms never appear in a shape
        return 1, None
    n_lo, n_hi = _bounds(num)
    d_lo, d_hi = _bounds(den)
    lo = max(1, -(-n_lo // d_hi)) if d_hi else 1
    return lo, (None if n_hi is None else -(-n_hi // max(d_lo, 1)))  # cdiv


def _bounds(d: DimExpr) -> "tuple[int, int | None]":
    """Interval of an affine dimension expression: ``(lowest, highest or None)``."""
    d = d.folded()
    lo: "int | None" = d.offset
    hi: "int | None" = d.offset
    for atom, coef in d.terms:
        a_lo, a_hi = _atom_bounds(atom)
        if coef > 0:
            lo_t, hi_t = coef * a_lo, None if a_hi is None else coef * a_hi
        else:
            lo_t, hi_t = (None if a_hi is None else coef * a_hi), coef * a_lo
        lo = None if lo is None or lo_t is None else lo + lo_t
        hi = None if hi is None or hi_t is None else hi + hi_t
    return (1 if lo is None else max(1, lo)), hi


def _numel_bounds(shape: "Iterable[DimExpr]") -> "tuple[int, int | None]":
    """Fewest and most elements a tensor of this shape can have (``None``: unbounded)."""
    lo = 1
    hi: "int | None" = 1
    for d in shape:
        d_lo, d_hi = _bounds(d)
        lo *= d_lo
        hi = None if hi is None or d_hi is None else hi * d_hi
    return lo, hi


def _check_index_range(name: str, values: "list[Value]") -> None:
    """Reject a tensor that has more elements than i32 offsets reach, whatever the dynamic sizes are."""
    for v in values:
        lo, hi = _numel_bounds(v.type.shape)
        if lo > INDEX_MAX or (hi is not None and hi > INDEX_MAX):
            what = f"{lo} elements" if lo == hi else f"at least {lo} elements" if lo > INDEX_MAX else f"up to {hi} elements (its largest bucket)"
            raise _fail(
                name,
                None,
                "index-range",
                f"{_label(v)} has {what}, more than the {INDEX_MAX} that the i32 offsets of a generated kernel reach",
            )


# -- checks that do not depend on the template --------------------------------------------------


def _check_dtype(kernel: str, what: str, dtype: Any) -> None:
    if isinstance(dtype, BlockScaled):
        raise _fail(kernel, None, "dtype", f"{what} has block-scaled dtype {dtype!r}; block-scaled tensors are not lowered")
    if dtype.name not in _TL:
        raise _fail(
            kernel,
            None,
            "dtype",
            f"{what} has dtype {dtype}, which has no plain Triton element type here "
            f"(supported: {', '.join(_TL)}); packed sub-byte formats and scales are not lowered",
        )


def _check_common(op: KernelOp) -> None:
    """Rules every template shares: parameters, supported ops, and the dtypes each op is defined for."""
    graph, name = op.graph, op.name
    for p in op.signature.params:
        if p.kind == "const" and p.default is MISSING:
            raise _fail(
                name, None, "const-default", f"const parameter '{p.name}' has no default; the value is compiled in as a constexpr"
            )
    for v in graph.inputs:
        if not v.type.contiguous:
            raise _fail(
                name,
                None,
                "view-input",
                f"parameter '{v.name}' is a View ({v.type}); strided views are not lowered, pass a contiguous tensor",
            )
        _check_dtype(name, f"parameter '{v.name}'", v.type.dtype)
    for n in graph.nodes:
        if "kind" in n.attrs:
            raise _fail(
                name,
                None,
                "declared-op-call",
                f"the body calls the {n.attrs['kind']} op '{n.op}'; only the built-in ops (+ - * /, silu, gelu, rsqrt, "
                "exp, sqrt, .to, sum, mean, max and stores) are lowered",
            )
        if n.op == "split":
            views = [str(t) for t in n.output_types if not t.contiguous]
            note = f" (its chunks are views: {views[0]})" if views else ""
            raise _fail(name, None, "split", f"split is not lowered{note}; strided views and slicing arrive later")
        for v in n.outputs:
            if not v.type.contiguous:
                raise _fail(name, None, "view-value", f"{_label(v)} is a View; strided views are not lowered")
        _check_node_dtypes(name, n)
    for v in graph.outputs:
        if not v.type.contiguous:
            raise _fail(name, None, "view-value", f"the returned {_label(v)} is a View; strided views are not lowered")
    _check_index_range(name, _all_values(graph))


def _check_node_dtypes(kernel: str, n: Node) -> None:
    if n.op in _BINARY:
        dt = n.outputs[0].type.dtype
        if n.op == "div":
            if dt.name != "f32":
                raise _fail(
                    kernel,
                    None,
                    "division-dtype",
                    f"{n}: division of {dt} tensors is not lowered; Triton divides f16/bf16 in f32 and returns f32, "
                    "an implicit promotion. Cast the operands with .to(f32) first",
                )
        elif dt.name not in _ARITHMETIC:
            raise _fail(
                kernel,
                None,
                "arithmetic-dtype",
                f"{n}: arithmetic on {dt} is not lowered (supported: {', '.join(sorted(_ARITHMETIC))}); cast with .to() first",
            )
    elif n.op in _UNARY:
        dt = n.outputs[0].type.dtype
        if dt.name not in _MATH:
            raise _fail(
                kernel,
                None,
                "math-dtype",
                f"{n}: {n.op} of a {dt} tensor is not lowered (Triton's math functions are f32 here); cast with .to(f32) first",
            )
    elif n.op in _REDUCTIONS:
        dt = n.inputs[0].type.dtype  # type: ignore[union-attr]
        if dt.name != "f32":
            raise _fail(
                kernel,
                None,
                "reduction-dtype",
                f"{n}: {n.op} of a {dt} tensor is not lowered (Triton would accumulate in {dt}); cast with .to(f32) first",
            )
    elif n.op == "cast":
        src = n.inputs[0].type.dtype  # type: ignore[union-attr]
        dst = n.outputs[0].type.dtype
        _check_dtype(kernel, f"the source of {n}", src)
        _check_dtype(kernel, f"the target of {n}", dst)
    elif n.op != "store":
        raise _fail(kernel, None, "unsupported-op", f"the op '{n.op}' is not lowered")


# -- names --------------------------------------------------------------------------------------


class _Names:
    """Hands out identifiers that collide with nothing already used (suffix ``_1``, ``_2``, ...)."""

    def __init__(self, taken: "Iterable[str]") -> None:
        self.taken = set(taken)

    def fresh(self, base: str) -> str:
        name, i = base, 0
        while name in self.taken or keyword.iskeyword(name):
            i += 1
            name = f"{base}_{i}"
        self.taken.add(name)
        return name


# -- the two templates ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Extra:
    """A parameter the template adds after the kernel's own and the outputs."""

    name: str
    constexpr: bool
    param: Param  # index is filled in when the signature is assembled


class _Layout:
    """What differs between the templates; the emitter below is shared."""

    template = ""
    grid: "tuple[DimExpr, ...]"

    def extras(self, names: _Names, taken_dims: "set[str]") -> "list[_Extra]":
        raise NotImplementedError

    def prologue(self) -> "list[str]":
        raise NotImplementedError

    def default_num_warps(self) -> int:
        raise NotImplementedError

    def header(self) -> "list[str]":
        raise NotImplementedError

    def load(self, p: str, v: Value) -> str:
        raise NotImplementedError

    def store(self, p: str, v: Value, value: str) -> str:
        raise NotImplementedError

    def reduce(self, op: str, a: str) -> str:
        raise NotImplementedError


def _const_param(name: str, default: Any) -> Param:
    return Param(name, 0, "const", default=default, keyword_only=True)


class _RowLayout(_Layout):
    template = "row"

    def __init__(self, op: KernelOp) -> None:
        graph, name = op.graph, op.name
        values = _all_values(graph)
        for v in values:
            if v.type.rank > 2:
                raise _fail(
                    name,
                    "row",
                    "row.rank",
                    f"{_label(v)} has rank {v.type.rank}; the row template takes [R, D], [D], [R] and [R, 1] (rank 1 or 2)",
                )
        two = [v for v in values if v.type.rank == 2]
        if not two:
            raise _fail(name, "row", "row.no-rows", "the row template needs a rank-2 tensor [R, D]; the kernel has none")
        pick = next((v for v in two if not _is_one(v.type.shape[1])), two[0])
        self.R, self.D = pick.type.shape[0], pick.type.shape[1].folded()
        D = self.D
        if not D.is_static and not all(d.buckets for d in D.free_dims()):
            dynamic = ", ".join(sorted(d.name for d in D.free_dims() if not d.buckets))
            raise _fail(
                name,
                "row",
                "row.dynamic-d",
                f"the row length D = {D} (last axis of {_label(pick)}) is dynamic in {dynamic}; it is baked into the "
                "binary, so it must be static (Dim('D', 2560)) or bucketed (Dim('D', buckets=[...]))",
            )
        for n in graph.nodes:
            if n.op in _REDUCTIONS:
                src = n.inputs[0]
                assert isinstance(src, Value)
                axes = n.attrs["axes"]
                if axes != (src.type.rank - 1,):
                    raise _fail(
                        name,
                        "row",
                        "row.last-axis-reduction",
                        f"{n.op} over axes {axes} of {_label(src)}: reductions run over the last axis only",
                    )
                if src.type.rank != 2:
                    raise _fail(
                        name,
                        "row",
                        "row.reduction-input",
                        f"{n.op} of {_label(src)}: a reduction needs an [R, D] input, this tensor has no row axis "
                        "to reduce along",
                    )
        for v in values:
            if v.type.rank == 0:
                raise _fail(
                    name,
                    "row",
                    "row.rank",
                    f"{_label(v)} has rank 0; the row template takes [R, D], [D], [R] and [R, 1] (rank 1 or 2)",
                )
        self.block_max = _max_block(name, D)
        self.cls = {}
        for v in values:
            shape = v.type.shape
            if len(shape) == 2:
                if shape[0] == self.R and shape[1] == D:
                    self.cls[v] = _FULL
                elif shape[0] == self.R and _is_one(shape[1]):
                    self.cls[v] = _ROW
                else:
                    raise self._shape_error(name, v)
            else:
                if shape[0] == D and shape[0] == self.R:
                    raise _fail(
                        name,
                        "row",
                        "row.ambiguous",
                        f"{_label(v)}: R and D are both {D}, so a rank-1 tensor could be [D] or [R]; give them different sizes",
                    )
                if shape[0] == D:
                    self.cls[v] = _COL
                elif shape[0] == self.R:
                    self.cls[v] = _ROW
                else:
                    raise self._shape_error(name, v)
        for n in graph.nodes:
            if n.op in _REDUCTIONS:
                src = n.inputs[0]
                assert isinstance(src, Value)
                if self.cls[src] != _FULL:
                    raise _fail(
                        name,
                        "row",
                        "row.reduction-input",
                        f"{n.op} of {_label(src)}: a reduction needs an [R, D] input, this tensor is one value per row",
                    )
        for p in graph.inputs:
            if p.access != "in" and self.cls[p] == _COL:
                raise _fail(
                    name,
                    "row",
                    "row.col-output",
                    f"{p.access} parameter '{p.name}' ({p.type}) has shape [D]; every row program would write it "
                    "(a race), so only [R, D] and per-row tensors can be written",
                )
        for v in graph.outputs:
            if self.cls[v] == _COL:
                raise _fail(
                    name,
                    "row",
                    "row.col-output",
                    f"the returned {_label(v)} has shape [D]; every row program would write it (a race), so only "
                    "[R, D] and per-row tensors can be returned",
                )
        self.grid = (self.R.folded(),)

    def _shape_error(self, name: str, v: Value) -> TilefishLoweringError:
        return _fail(
            name,
            "row",
            "row.shape",
            f"{_label(v)} does not fit: with R = {self.R} and D = {self.D} the row template takes [{self.R}, {self.D}], "
            f"[{self.D}], [{self.R}] and [{self.R}, 1]",
        )

    # -- layout protocol ------------------------------------------------------------------------

    def extras(self, names: _Names, taken_dims: "set[str]") -> "list[_Extra]":
        self.D_name = names.fresh("D")
        self.BLOCK_name = names.fresh("BLOCK_D")
        self.row, self.cols, self.mask, self.offs = (names.fresh(x) for x in ("row", "cols", "mask", "offs"))
        if self.D.is_static:
            d_default: Any = self.D.value
            block_default: Any = next_pow2(self.D.value)
        else:
            d_default = self.D
            block_default = next_pow2(self.D)
        return [
            _Extra(self.D_name, True, _const_param(self.D_name, d_default)),
            _Extra(self.BLOCK_name, True, _const_param(self.BLOCK_name, block_default)),
        ]

    def prologue(self) -> "list[str]":
        return [
            f"{self.row} = tl.program_id(0)",
            f"{self.cols} = tl.arange(0, {self.BLOCK_name})",
            f"{self.mask} = {self.cols} < {self.D_name}",
            f"{self.offs} = {self.row} * {self.D_name} + {self.cols}",
        ]

    def default_num_warps(self) -> int:
        return max(1, min(ROW_MAX_WARPS, self.block_max // ROW_LANES_PER_WARP))

    def header(self) -> "list[str]":
        d = f"{self.D} (static)" if self.D.is_static else f"{self.D} (bucketed)"
        return [f"template: row; one program per row, grid {_grid_text(self.grid)}", f"D = {d}, BLOCK_D = next_pow2(D)"]

    def load(self, p: str, v: Value) -> str:
        c = self.cls[v]
        if c == _FULL:
            return f"tl.load({p} + {self.offs}, mask={self.mask}, other=0)"
        if c == _COL:
            return f"tl.load({p} + {self.cols}, mask={self.mask}, other=0)"
        return f"tl.load({p} + {self.row})"

    def store(self, p: str, v: Value, value: str) -> str:
        if self.cls[v] == _FULL:
            return f"tl.store({p} + {self.offs}, {value}, mask={self.mask})"
        return f"tl.store({p} + {self.row}, {value})"

    def reduce(self, op: str, a: str) -> str:
        if op == "max":
            return f"tl.max(tl.where({self.mask}, {a}, float(\"-inf\")), axis=0)"
        total = f"tl.sum(tl.where({self.mask}, {a}, 0.0), axis=0)"
        return total if op == "sum" else f"{total} / {self.D_name}"


def _max_block(kernel: str, d: DimExpr) -> int:
    """The largest ``next_pow2(D)`` over the buckets of ``D`` (its value when static)."""
    free = sorted(d.free_dims(), key=lambda x: x.name)
    best = 0
    for combo in itertools.product(*(x.buckets for x in free)):  # type: ignore[misc]
        value = d.subst(dict(zip(free, combo))).value
        if value < 1:
            raise _fail(kernel, "row", "row.dynamic-d", f"D = {d} is {value} in the bucket {dict(zip((x.name for x in free), combo))}")
        best = max(best, next_pow2(value))
    if best > MAX_BLOCK:
        raise _fail(
            kernel, "row", "row.block-size", f"a row of {best} lanes is more than Triton's {MAX_BLOCK}-element tensor limit"
        )
    return best


class _ElementwiseLayout(_Layout):
    template = "elementwise"

    def __init__(self, op: KernelOp) -> None:
        graph, name = op.graph, op.name
        values = _all_values(graph)
        if not values:
            raise _fail(name, "elementwise", "elementwise.no-tensor", "the kernel has no tensors")
        for n in graph.nodes:
            if n.op in _REDUCTIONS:
                raise _fail(
                    name, "elementwise", "elementwise.reduction", f"{n.op} is a reduction; the elementwise template has none"
                )
        self.shape = values[0].type.shape
        for v in values:
            if v.type.shape != self.shape:
                raise _fail(
                    name,
                    "elementwise",
                    "elementwise.same-shape",
                    f"{_label(v)} does not have the shape {_shape_text(self.shape)} of {_label(values[0])}: the "
                    "elementwise template does not broadcast (only python and const scalars combine with a tensor)",
                )
        self.total = _product(self.shape)

    def extras(self, names: _Names, taken_dims: "set[str]") -> "list[_Extra]":
        self.n_name = names.fresh("n")
        self.BLOCK_name = names.fresh("BLOCK")
        self.pid, self.offs, self.mask = (names.fresh(x) for x in ("pid", "offs", "mask"))
        if self.total is not None:
            # the count is the product of the shape, so a spec says how to compute it and derives its divisibility
            self.n_dim = self.total
        else:
            # two symbolic factors: not an affine expression, so a fresh dimension that carries what the factors prove
            dim_name, i = self.n_name, 0
            while dim_name in taken_dims:
                i += 1
                dim_name = f"{self.n_name}_{i}"
            k = min(ELEMENTWISE_BLOCK, math.prod(_pow2_divisor(d) for d in self.shape))
            self.n_dim = Dim(dim_name, divisible_by=k if k > 1 else None)
        self.grid = (as_dim(cdiv(self.n_dim, ELEMENTWISE_BLOCK)),)
        return [
            _Extra(self.n_name, False, Param(self.n_name, 0, "dim", dim=self.n_dim)),
            _Extra(self.BLOCK_name, True, _const_param(self.BLOCK_name, ELEMENTWISE_BLOCK)),
        ]

    def prologue(self) -> "list[str]":
        return [
            f"{self.pid} = tl.program_id(0)",
            f"{self.offs} = {self.pid} * {self.BLOCK_name} + tl.arange(0, {self.BLOCK_name})",
            f"{self.mask} = {self.offs} < {self.n_name}",
        ]

    def default_num_warps(self) -> int:
        return ELEMENTWISE_NUM_WARPS

    def header(self) -> "list[str]":
        n = str(self.total) if self.total is not None else " * ".join(str(d) for d in self.shape)
        return [
            f"template: elementwise; every tensor is {_shape_text(self.shape)}, flattened to n = {n} elements",
            f"grid {_grid_text(self.grid)}, BLOCK = {ELEMENTWISE_BLOCK}",
        ]

    def load(self, p: str, v: Value) -> str:
        return f"tl.load({p} + {self.offs}, mask={self.mask}, other=0)"

    def store(self, p: str, v: Value, value: str) -> str:
        return f"tl.store({p} + {self.offs}, {value}, mask={self.mask})"

    def reduce(self, op: str, a: str) -> str:  # pragma: no cover - rejected in __init__
        raise AssertionError("the elementwise template has no reductions")


def _product(shape: "tuple[DimExpr, ...]") -> "DimExpr | None":
    """``prod(shape)`` as a dimension expression, or ``None`` when it is not affine (two symbolic factors)."""
    total = DimExpr.const(1)
    try:
        for d in shape:
            total = total * d
    except TilefishTypeError:
        return None
    return total.folded()


def _pow2_divisor(d: DimExpr) -> int:
    """The largest power of two, up to the elementwise block size, that provably divides ``d``."""
    k = 1
    while k * 2 <= ELEMENTWISE_BLOCK and d.is_divisible_by(k * 2):
        k *= 2
    return k


def _grid_text(grid: "tuple[DimExpr, ...]") -> str:
    from .decorators import grid_to_string

    return grid_to_string(grid)


# -- code generation ------------------------------------------------------------------------------


def _scalar_text(s: Scalar, rename: "dict[str, str]") -> str:
    if s.name is not None:
        return rename.get(s.name, s.name)
    v = s.value
    if isinstance(v, float):
        if math.isnan(v):
            return 'float("nan")'
        if math.isinf(v):
            return 'float("inf")' if v > 0 else 'float("-inf")'
    return repr(v)


def _is_self_store(n: Node) -> bool:
    """``x[...] = x``: the stored value is the version the store replaces."""
    return n.inputs[0] is n.inputs[1]


class _Body:
    """Emits the statements of the kernel body, one local per SSA value."""

    def __init__(
        self, op: KernelOp, layout: _Layout, names: _Names, out_names: "list[str]", rename: "dict[str, str]"
    ) -> None:
        self.graph = op.graph
        self.layout = layout
        self.names = names
        self.out_names = out_names
        self.rename = rename  # kernel parameter name -> its name in the generated function, where they differ
        self.params = {p.name: p for p in op.graph.inputs}
        self.var: dict[Value, str] = {}
        self.lines: list[str] = []
        self._count = 0

    def local(self) -> str:
        name = self.names.fresh(f"v{self._count}")
        self._count += 1
        return name

    def emit(self, text: str, comment: str = "") -> None:
        self.lines.append(f"    {text}" + (f"  # {comment}" if comment else ""))

    def operand(self, x: "Value | Scalar") -> str:
        return self.var[x] if isinstance(x, Value) else _scalar_text(x, self.rename)

    def run(self) -> "list[str]":
        graph, layout = self.graph, self.layout
        for line in layout.prologue():
            self.emit(line)
        reads: set[Value] = set()
        last_store: dict[str, Node] = {}
        for n in graph.nodes:
            if n.op == "store":
                # a store's first input is the version it replaces, so the parameter is named by the node
                if not _is_self_store(n):  # copying a tensor onto itself writes nothing
                    last_store[n.attrs["target"]] = n
                reads.add(n.inputs[1])  # type: ignore[arg-type]
            else:
                reads.update(i for i in n.inputs if isinstance(i, Value))
        reads.update(graph.outputs)
        for p in graph.inputs:  # inputs first, in declaration order; an Out parameter is never read
            if p.access != "out" and p in reads:
                v = self.local()
                self.var[p] = v
                self.emit(f"{v} = {layout.load(self.rename.get(p.name, p.name), p)}", p.name)
        for n in graph.nodes:
            if n.op == "store":
                self.store(n, last_store)
            else:
                (out,) = n.outputs
                v = self.local()
                self.var[out] = v
                rounding = "rounding point" if n.is_rounding_point else ""
                self.emit(f"{v} = {self.expression(n)}", rounding)
        for k, v in enumerate(graph.outputs):
            self.emit(layout.store(self.out_names[k], v, self.var[v]), f"return value {k}")
        return self.lines

    def store(self, n: Node, last_store: "dict[str, Node]") -> None:
        _, value = n.inputs
        assert isinstance(value, Value)
        if n.outputs:  # an InOut parameter: reading it from here on reads the stored value
            self.var[n.outputs[0]] = self.var[value]
        name = n.attrs["target"]
        param = self.params[name]
        label = "InOut" if param.access == "inout" else "Out"
        if _is_self_store(n):
            self.emit(f"# {label} {name}: stored onto itself, nothing to write")
        elif last_store[name] is n:
            what = "final version" if param.access == "inout" else "written once"
            self.emit(self.layout.store(self.rename.get(name, name), param, self.var[value]), f"{label} {name}, {what}")
        else:
            self.emit(f"# {label} {name}: superseded by a later store")

    def expression(self, n: Node) -> str:
        op = n.op
        a = [self.operand(i) for i in n.inputs]
        if op in _BINARY:
            return f"{a[0]} {_BINARY[op]} {a[1]}"
        if op == "rsqrt":
            return f"tl.rsqrt({a[0]})"
        if op == "exp":
            return f"tl.exp({a[0]})"
        if op == "sqrt":
            return f"tl.sqrt_rn({a[0]})"
        if op == "silu":
            return f"{a[0]} * tl.sigmoid({a[0]})"
        if op == "gelu":
            return f"0.5 * {a[0]} * (1.0 + tl.math.erf({a[0]} * 0.7071067811865476))"
        if op == "cast":
            return f"{a[0]}.to(tl.{_TL[n.attrs['to'].name]})"
        if op in _REDUCTIONS:
            return self.layout.reduce(op, a[0])
        raise AssertionError(f"unreachable op {op}")  # pragma: no cover - _check_common rejects it


# -- assembly -------------------------------------------------------------------------------------


def _stub(fn_name: str, module: str) -> Any:
    def stub() -> None: ...

    stub.__name__ = stub.__qualname__ = fn_name
    stub.__module__ = module
    stub.__doc__ = f"The generated Triton kernel {module}:{fn_name}."
    return stub


def _check_names(op: Any, module: "str | None") -> "tuple[str, str]":
    """Validate the kernel and the module name; return ``(module, function name)``."""
    if not isinstance(op, KernelOp):
        raise TilefishLoweringError(
            f"lower takes a @tilefish.kernel, got {op!r}; only kernels have a body to lower", rule="not-a-kernel"
        )
    fn_name = f"tf_{op.name}"
    if not _IDENT.match(fn_name):
        raise _fail(op.name, None, "name", f"'{fn_name}' is not a plain ASCII function name; give the kernel a name of letters, digits and underscores")
    if module is None:
        module = f"{DEFAULT_MODULE}.{op.name}"
    if not isinstance(module, str) or not _MODULE.match(module) or any(keyword.iskeyword(p) for p in module.split(".")):
        raise TilefishLoweringError(f"lower: module={module!r} must be a dotted Python module name", rule="module")
    return module, fn_name


def lower(
    op: KernelOp,
    *,
    module: "str | None" = None,
    template: "str | None" = None,
    num_warps: "int | None" = None,
) -> LoweredKernel:
    """Lower a ``@tilefish.kernel`` to Triton source.

    ``module`` is the Python module the generated source will live in (default
    ``"tilefish_generated.<kernel name>"``); the Triton reference is ``module:tf_<name>``.
    ``template`` forces ``"row"`` or ``"elementwise"``; by default the elementwise template is
    tried first and then the row template. ``num_warps`` overrides the kernel's own
    ``num_warps=`` and the template's rule. Raises :class:`~tilefish.errors.TilefishLoweringError`
    when the body fits no template.
    """
    module, fn_name = _check_names(op, module)
    if template is not None and template not in TEMPLATES:
        raise TilefishLoweringError(f"lower: template must be 'row' or 'elementwise', got {template!r}", rule="template")
    if num_warps is None:
        num_warps = op.num_warps
    if num_warps is not None and (isinstance(num_warps, bool) or not isinstance(num_warps, int) or num_warps not in NUM_WARPS):
        raise TilefishLoweringError(
            f"lower: num_warps must be a power of two from 1 to 32, got {num_warps!r}", rule="num-warps"
        )
    _check_common(op)
    errors: list[TilefishLoweringError] = []
    for t in (template,) if template else TEMPLATES:
        try:
            layout = _ElementwiseLayout(op) if t == "elementwise" else _RowLayout(op)
        except TilefishLoweringError as e:
            errors.append(e)
            continue
        return _assemble(op, layout, module, fn_name, num_warps)
    if len(errors) == 1:
        raise errors[0]
    detail = "; ".join(f"{e.rule}: {str(e).split('failed: ', 1)[-1]}" for e in errors)
    raise TilefishLoweringError(
        f"cannot lower kernel '{op.name}': it fits no template. {detail}", rule="no-template", causes=tuple(errors)
    )


def _assemble(op: KernelOp, layout: _Layout, module: str, fn_name: str, num_warps: "int | None") -> LoweredKernel:
    graph = op.graph
    user = list(op.signature.params)
    names = _Names({p.name for p in user} | _RESERVED)
    # a kernel parameter that has the name of something the generated module uses would shadow it
    rename = {p.name: names.fresh(p.name) for p in user if p.name in _RESERVED}
    out_names = (
        [names.fresh("out")] if len(graph.outputs) == 1 else [names.fresh(f"out{k}") for k in range(len(graph.outputs))]
    )
    taken_dims: set[str] = set()
    for p in user:
        if p.kind == "tensor":
            assert p.type is not None
            for d in p.type.shape:
                taken_dims.update(d.named_dims())
        elif p.kind == "dim":
            assert p.dim is not None
            taken_dims.update(p.dim.named_dims())
    extras = layout.extras(names, taken_dims)
    if num_warps is None:
        num_warps = layout.default_num_warps()

    # the parameters of the generated function and of its typed stub, in order
    decls: list[str] = []
    params: list[Param] = []
    max_elements: dict[str, int] = {}  # tensors whose size no static bound keeps within i32 offsets
    for p in user:
        new = rename.get(p.name, p.name)
        params.append(_reindexed(p, len(params), new))
        decls.append(f"{new}: tl.constexpr" if p.kind == "const" else new)
        if p.kind == "tensor" and _numel_bounds(p.type.shape)[1] is None:  # type: ignore[union-attr]
            max_elements[new] = INDEX_MAX
    for k, v in enumerate(graph.outputs):
        params.append(Param(out_names[k], len(params), "tensor", "out", type=v.type))
        decls.append(out_names[k])
        if _numel_bounds(v.type.shape)[1] is None:
            max_elements[out_names[k]] = INDEX_MAX
    for e in extras:
        params.append(_reindexed(e.param, len(params), e.param.name))
        decls.append(f"{e.name}: tl.constexpr" if e.constexpr else e.name)
    element_counts: dict[str, str] = {}
    if layout.template == "elementwise":  # n is the element count of every tensor: say which one a host reads it from
        element_counts[layout.n_name] = next(p.name for p in params if p.kind == "tensor")  # type: ignore[attr-defined]

    body = _Body(op, layout, names, out_names, rename).run()
    header = [
        f"Generated by tilefish.lower from the kernel '{op.name}'. Do not edit.",
        *layout.header(),
        f"num_warps: {num_warps}",
        *[f"parameter '{old}' of the kernel is '{new}' here (the name is used by this module)" for old, new in rename.items()],
        "Numerics contract: see the docstring of tilefish.lower.",
    ]
    lines = [f"# {h}" for h in header]
    lines += ["import triton", "import triton.language as tl", "", "", "@triton.jit", f"def {fn_name}("]
    lines += [f"    {d}," for d in decls]
    lines += ["):", *body]
    source = "\n".join(lines) + "\n"

    extern = ExternOp(
        _stub(fn_name, module),
        op.name,
        Signature(params, ()),
        f"{module}:{fn_name}",
        num_warps,
        NUM_STAGES,
        layout.grid,
        max_elements=max_elements,
        element_counts=element_counts,
    )
    return LoweredKernel(op.name, fn_name, module, layout.template, source, extern, num_warps, op)


def _reindexed(p: Param, index: int, name: str) -> Param:
    return Param(
        name,
        index,
        p.kind,
        p.access,
        p.type,
        p.const_dtype,
        p.default,
        p.dim,
        p.keyword_only,
    )
