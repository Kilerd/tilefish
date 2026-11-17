"""Plan fusion: the arithmetic written in a ``@tilefish.graph`` body, planned as generated kernels (``docs/rfcs/0001-plan-fusion.md``).

A graph body may compute: ``silu(h.to(f32) + b.to(f32)).to(bf16)`` between two GEMMs is traced like any kernel body. A launch plan launches
declared operators, so the planner groups the traced arithmetic into **regions** and makes each region a ``@tilefish.kernel`` of its own (a
*synthetic* kernel, named ``fused_<hash>`` by its content), lowered by :mod:`tilefish.lower` like every kernel. A region is then one ordinary
``kernel`` step of the plan: nothing in the plan document, the manifest or either host is new.

**Exactness.** Fusion never changes bytes by itself. The *unfused composition* of a graph is its plan with ``grouping="none"``: every op a
kernel of its own, every intermediate stored in the dtype the IR gives it (a mask as ``u8``, 0 or 1). A fused plan writes the same bytes as the
unfused one, for every input, because

* every region kernel is lowered with ``fp_fusion=False`` and ``context_free=True`` (note 20 of :mod:`tilefish.lower`): every float ``+ - *``
  is one scalar ``.rn`` instruction of inline PTX (ptxas contracts a packed ``mul.rn.f32x2`` and ``add.rn.f32x2`` into ``FFMA2`` even with
  ``--fmad=false``), every narrowing float cast is inline PTX (out of the reach of the folds that would keep an input NaN's payload through
  ``x.to(f32).to(bf16)``), and a negation gives the canonical NaN explicitly (ptxas folds ``neg.f32`` into a select as a sign flip);
  ``tilefish.fma``, or the declared rewrite ``fp_contract="mul-add"``, asks for one rounding;
* the lowering writes each op's code from the op alone, and a per-element op's result depends on nothing but its operands;
* a reduction joins a region only when its bytes cannot depend on the kernel around it: an ordered ``sum`` or ``mean`` (``order=``), the
  order-free ``max``, ``argmax`` and ``argmin`` (and ``gather_rows``, a copy) are fused into row regions by version 2 (the default), lowered by
  revision 2 of the lowering (note 21 of :mod:`tilefish.lower`); a Triton tree reduction (a ``sum`` or ``mean`` without ``order=``) is a region of
  its own in every version, its operand stored, because its bytes depend on the rest of its kernel (measured: 357 to 399 of 1024 rows differ when
  other loads share it). Version 1 runs every reduction alone, as tilefish 0.0.9 and 0.0.10 did, and is kept unchanged (D7).

``export`` checks it: every region of more than one op is run against its ops one by one, on seeded inputs with special values, at every
bucket, and a difference refuses the export (``verify=False`` opts out). The fusion report, ``<graph>.fusion.json`` next to the plan, says which
ops each region took and every rounding it keeps.

**Modes** (``tilefish.fusion(...)``), each exact: ``intermediates="declared"`` (every op rounds to its IR dtype; the default) or ``"opmath"``
(torch's eager rule for a 16-bit op with a python scalar: the op in f32 with the scalar in f32, rounded once to the 16-bit dtype);
``fp_contract="none"`` (the default) or ``"mul-add"`` (an f32 ``a * b + c`` whose product has no other use becomes ``tilefish.fma(a, b, c)``,
one rounding); ``math="fast"`` or ``"ieee"`` (the default math of the body's ops, as ``@tilefish.kernel(math=...)``); ``grouping="auto"`` or
``"none"``. Each rewrite is made on the IR before the regions are formed, so its rounding points never depend on where the regions fall, and a
region's source (so its digest, so the plan's pin) carries the mode.

``tilefish.barrier(x)`` makes ``x`` a region boundary (it gets a buffer of the plan, which a tap can read); it never changes a byte.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import threading
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .dims import Dim, DimExpr, as_dim
from .dtypes import DType, bool_, f32, u8
from .errors import TilefishError, TilefishLoweringError, TilefishPlanError, TilefishRuntimeError
from .ir import Graph, Node, Scalar
from .trace import Value
from .types import TensorType

VERSIONS = (1, 2)
DEFAULT_VERSION = 2  # what tilefish.fusion() and a graph that declares nothing get: the latest version (D7: every earlier one is kept)
GROUPINGS = ("auto", "none")
INTERMEDIATES = ("declared", "opmath")
FP_CONTRACTS = ("none", "mul-add")
MATHS = ("fast", "ieee")
# docs/rfcs/0003-epilogue-fusion.md: how the arithmetic after a call is fused into it. "exact" (the default): through the epilogue hooks of externs,
# which see the call's rounded result (contract (i), the bytes of the unfused composition); nothing else changes. "accumulator": the declared rewrite
# R_acc (every widening of the result of an extern with an accumulator hook is the extern's f32 result) and hooks that see the accumulator (contract
# (ii)); a library call is left as it is (R_acc of a library call is not planned). "library": the hooks of "exact", and the arithmetic after a cuBLASLt
# GEMM rewritten into one of its epilogues (cublaslt.matmul, contract (iii), verified per pin against contract (ii)) where it matches one. A graph
# declares one, a tilefish.fused scope another for the ops traced in it, so a graph may mix extern hooks of either contract and library epilogues
EPILOGUES = ("exact", "accumulator", "library")
MAX_REGION_NODES = 64  # ops of one region (versions 1 and 2)
MAX_REGION_TENSORS = 32  # inputs and outputs of one region (versions 1 and 2)
LOWERING_OF_VERSION = {1: 1, 2: 2}  # the revision of tilefish.lower each version's region kernels are lowered with (note 21), frozen with it
REPORT_SCHEMA = "tilefish.fusion-report.v1"

_VIEW_OPS = frozenset({"reshape", "permute", "select", "split"})
_MOVE_OPS = frozenset({"copy", "store", "zeros"})
_FREE_OPS = frozenset({"load", "iota"})  # no arithmetic on data: copied into every region that reads them, never stored
_SOLO_OPS = frozenset({"sum", "mean", "max", "argmax", "argmin", "gather_rows"})  # a region of their own (version 1)


def is_solo(op: str, attrs: Mapping[str, Any], version: int) -> bool:
    """Whether an op of a graph body is a region of its own: in version 1 every reduction and ``gather_rows``; in version 2 a ``sum`` or ``mean``
    without ``order=`` only (D2: a Triton tree reduction's bytes depend on the rest of its kernel), every other reduction (an ordered sum or mean,
    the order-free ``max``, ``argmax``, ``argmin``) and ``gather_rows`` joining row regions."""
    if version < 2:
        return op in _SOLO_OPS
    return op in ("sum", "mean") and "order" not in attrs
_REFUSED_OPS = {"scatter_rows": "a scatter into an InOut parameter in a graph body is not planned yet: put it in a @tilefish.kernel"}
_ROUNDING_OPS = frozenset({"add", "sub", "mul", "div", "fma", "rsqrt", "exp", "sqrt", "silu", "gelu", "sin", "cos", "tanh", "erf", "sin_approx", "rcp", "sum", "mean"})
_HALF = frozenset({"bf16", "f16"})


class TilefishFusionWarning(UserWarning):
    """An export could not verify its fused regions against the unfused composition (no CUDA device of the export's architecture), so the plan
    was written without that check; ``tilefish.verify_fusion`` runs it where such a device is."""


class TilefishFusionError(TilefishPlanError):
    """A fused region does not write the bytes of its ops one by one (the export-time verification found a difference). This is a bug of
    tilefish or of the toolchain, never of the graph: report it with the region named in the message; ``verify=False`` exports anyway."""


# -- the declaration ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Fusion:
    """How the arithmetic of a graph body is planned (:func:`fusion`)."""

    version: int = DEFAULT_VERSION
    grouping: str = "auto"
    math: str = "fast"
    intermediates: str = "declared"
    fp_contract: str = "none"
    verify: bool = True
    sass: Any = None  # the SASS assertions of every region's kernel (tilefish.sass), or None
    epilogues: str = "exact"  # docs/rfcs/0003-epilogue-fusion.md: "exact", "accumulator" (R_acc) or "library" (cuBLASLt's epilogues)

    def canonical(self) -> "dict[str, Any]":
        """What a region's kernel depends on (``verify`` is a check, not a choice of numerics, and is left out; ``sass`` only when given, so
        every region of a graph without it is named as it was; ``epilogues`` is not: a mode changes a region's nodes, through ``R_acc``, or which ops
        become a library call, or nothing, and a region it leaves is the same kernel in every mode, so the same operator)."""
        out = {"version": self.version, "grouping": self.grouping, "math": self.math, "intermediates": self.intermediates, "fp_contract": self.fp_contract}
        if self.sass is not None:
            out["sass"] = self.sass.canonical()
        return out

    def but(self, **changes: Any) -> "Fusion":
        """These settings with ``changes``."""
        return replace(self, **changes)

    def described(self) -> "dict[str, Any]":
        """The settings as the fusion report says them: :meth:`canonical`, and ``epilogues`` when it is not the default."""
        out = self.canonical()
        if self.epilogues != "exact":
            out["epilogues"] = self.epilogues
        return out


def fusion(
    *,
    version: int = DEFAULT_VERSION,
    grouping: str = "auto",
    math: str = "fast",
    intermediates: str = "declared",
    fp_contract: str = "none",
    verify: bool = True,
    sass: Any = None,
    epilogues: str = "exact",
) -> Fusion:
    """``@tilefish.graph(fusion=tilefish.fusion(...))``: how the arithmetic written in the graph body is planned. Every choice but ``epilogues=
    "accumulator"`` and ``epilogues="library"`` is exact: the plan writes the bytes of its unfused composition (each op a kernel of its own;
    ``grouping="none"`` is that plan).

    * ``version``: the grouping rules, limits and rewrites, frozen (D7). ``2``, the default: row regions, an ordered ``sum`` or ``mean``, ``max``,
      ``argmax``, ``argmin`` and ``gather_rows`` fused with the ops around them (one kernel of an rmsnorm and its residual add), lowered by
      revision 2 of the lowering (note 21 of :mod:`tilefish.lower`: faster ordered sums, views read where they are by the row template). ``1``:
      every reduction a region of its own: the regions of fusion's first version (tilefish 0.0.9 and 0.0.10), unchanged.
    * ``grouping``: ``"auto"`` (regions as large as the rules allow) or ``"none"`` (one kernel per op: the unfused composition, an oracle).
    * ``math``: ``"fast"`` or ``"ieee"``, the default f32 math of the body's ops (note 13 of :mod:`tilefish.lower`); an op's own ``math=`` wins.
    * ``intermediates``: ``"declared"`` (every op rounds to the dtype the IR gives it) or ``"opmath"`` (torch's eager rule: a bf16 or f16
      ``+ - *`` with a python scalar is computed in f32 with the scalar in f32 and rounded once to its dtype).
    * ``fp_contract``: ``"none"`` (nothing is contracted; ``tilefish.fma`` asks for one rounding) or ``"mul-add"`` (an f32 ``a * b + c`` or
      ``a * b - c`` whose product nothing else reads becomes ``tilefish.fma``, one rounding; of two such products of one add, the left one).
    * ``verify``: whether ``export`` checks every region of more than one op against its ops one by one (``True``; ``export(..., verify=False)``
      or ``False`` here skips it).
    * ``sass``: SASS assertions (:func:`tilefish.sass`) every region's kernel is built under (``tilefish.sass(no_spills=True, max_registers=64)``),
      checked at every build; in the regions' declaration digests and names (only when given).
    * ``epilogues`` (``docs/rfcs/0003-epilogue-fusion.md``): how the arithmetic that reads a call's result is fused into the call.

      * ``"exact"``, the default: into an extern that declares an epilogue hook (``tilefish.hook``), through a hook that sees the extern's rounded
        result, so the plan writes the bytes of the extern followed by the region (contract (i)); nothing changes a byte.
      * ``"accumulator"``: the declared rewrite ``R_acc`` (every widening ``y.to(f32)`` of such an extern's result is the extern's f32 result, one
        rounding fewer) and a hook that sees the accumulator (contract (ii)): the bytes of the unfused composition of the rewritten program, not of
        the extern and the region. A library call is left as it is (``R_acc`` of a ``cublaslt.gemm`` is not planned): the fusion report's
        ``library_epilogues`` says so of every cuBLASLt GEMM whose result the graph widens there.
      * ``"library"``: the hooks of ``"exact"``, and the arithmetic after a cuBLASLt GEMM (``cublaslt.gemm``, ``cublaslt.gemm_bias``) of a 16-bit
        result as one of its epilogues: ``y.to(f32)``, then ``+ c.to(f32)`` (a ``[M, N]`` residual, ``beta * C``), then ``+ bias.to(f32)`` (a ``[N]``
        bias, of a ``cublaslt.gemm``), then ``tilefish.where(h <= 0.0, 0.0, h)`` (ReLU: ``torch.relu``'s bits), then ``.to()`` the GEMM's type, each
        step optional but one, becomes one call of a ``cublaslt.matmul`` (``BIAS``, ``RELU``, ``RELU_BIAS``, ``beta * C``) on the accumulator: the
        rounding of ``y`` is removed (contract (ii) of RFC 0003), and the bytes are the library's, required ``measured-exact`` against the formula at
        every pin (``export`` and ``jit`` refuse a pin that is not). Where the pattern does not match (another order, an f32 bias, a value read
        elsewhere) nothing changes; the fusion report lists every rewrite.

      Also per scope (:func:`tilefish.fused`), whose mode wins over the graph's for the ops traced in it: a graph may mix them (``"library"`` for
      its cuBLASLt GEMMs and an ``"accumulator"`` scope around the arithmetic after an extern, say).
    """
    if version not in VERSIONS or isinstance(version, bool):
        raise TilefishPlanError(f"fusion: version must be one of {list(VERSIONS)}, got {version!r}")
    for key, value, allowed in (
        ("grouping", grouping, GROUPINGS),
        ("math", math, MATHS),
        ("intermediates", intermediates, INTERMEDIATES),
        ("fp_contract", fp_contract, FP_CONTRACTS),
        ("epilogues", epilogues, EPILOGUES),
    ):
        if value not in allowed:
            raise TilefishPlanError(f"fusion: {key} must be one of {list(allowed)}, got {value!r}")
    if not isinstance(verify, bool):
        raise TilefishPlanError(f"fusion: verify must be True or False, got {verify!r}")
    from ._sass import checks_or_none

    return Fusion(version, grouping, math, intermediates, fp_contract, verify, checks_or_none(sass, "fusion"), epilogues)


def fused_scope(*, epilogues: "str | None" = None) -> Any:
    """``with tilefish.fused(epilogues="accumulator"):`` in a graph body: the ops traced in the block are planned with these settings instead of the
    graph's (``docs/rfcs/0003-epilogue-fusion.md``, Q3: a mode is declared per graph or per scope, never per call). ``epilogues`` is a value of
    ``tilefish.fusion(epilogues=...)``: a widening ``y.to(f32)`` of an extern's result written in an ``"accumulator"`` scope is ``R_acc``'s, in an
    ``"exact"`` or ``"library"`` scope it is not, whatever the graph declares; the arithmetic after a cuBLASLt GEMM is rewritten into one of its
    epilogues where every op of the pattern (from ``y.to(f32)`` to the last ``.to()``, the widenings of ``c`` and the bias included) is in a
    ``"library"`` mode. Scopes nest (the innermost's
    settings win). A kernel body has no scope."""
    from .trace import current_context, scope

    settings: "dict[str, Any]" = {}
    if epilogues is not None:
        if epilogues not in EPILOGUES:
            raise TilefishPlanError(f"tilefish.fused: epilogues must be one of {list(EPILOGUES)}, got {epilogues!r}")
        settings["epilogues"] = epilogues
    if not settings:
        raise TilefishPlanError(
            "tilefish.fused: say what the scope sets (epilogues=...); a scope that names a region (name=) is not planned yet (RFC 0001, section 2.3)"
        )
    ctx = current_context()
    if ctx is None or ctx.kind != "graph":
        raise TilefishPlanError("tilefish.fused is a scope of a @tilefish.graph body (it sets how the arithmetic traced in it is planned)")
    return scope(settings)


def epilogue_mode(n: Node, settings: Fusion) -> str:
    """The ``epilogues`` mode of a node of a graph body: its ``tilefish.fused`` scope's, else the graph's."""
    sc = getattr(n, "scope", None) or {}
    return str(sc.get("epilogues", settings.epilogues))


def settings_of(gop: Any) -> Fusion:
    found = getattr(gop, "fusion", None)
    return found if isinstance(found, Fusion) else Fusion()


# -- the result of the pass -----------------------------------------------------------------------------------------------------------


@dataclass(eq=False)
class Region:
    """One fused region: the ops it took (``nodes``, indices into :attr:`Fused.program`), the copies of free ops it holds, its inputs and
    outputs (values of the program), and the kernel that computes it."""

    index: int
    nodes: "list[int]"
    free: "list[int]" = field(default_factory=list)
    inputs: "list[Value]" = field(default_factory=list)
    outputs: "list[Value]" = field(default_factory=list)
    kernel: Any = None
    call: Any = None  # the KernelOp the call node names (a specialisation for unaligned inputs, or the kernel)
    reads: "dict[Value, str]" = field(default_factory=dict)  # input -> "contiguous" | "unaligned" | "materialised"


@dataclass(eq=False)
class Hooked:
    """A region fused into the extern call that makes its input, through the extern's epilogue hook (``docs/rfcs/0003-epilogue-fusion.md``): one call
    of ``op`` (the extern with the region in its hook) in place of the call ``call`` (a program node), the region and the views of the call's result
    only they read."""

    call: int
    region: Region
    hook: Any
    extern: Any
    op: Any
    contract: str  # "exact" (contract (i)) | "accumulator" (contract (ii))
    inputs: "list[Value]" = field(default_factory=list)
    outputs: "list[Value]" = field(default_factory=list)
    views: "list[int]" = field(default_factory=list)  # program nodes (splits of the result) the unit takes
    # the splits whose chunks the hook holds; one of them that something else reads too stays a view of the stored result, after the unit
    held_views: "list[int]" = field(default_factory=list)
    unit: "set[int]" = field(default_factory=set)  # the call, the region's nodes and the held splits (the cycle check's unit)
    main: "Value | None" = None  # the region's output the kernel stores into the hook's output (None: the result itself)
    result_read: bool = True  # whether anything but the region reads the result (else, with no main, the kernel stores it into a buffer nothing reads)
    tiles: "list[Value]" = field(default_factory=list)  # the values the hook holds (the result, its chunks, its rounding)
    sources: "dict[str, Any]" = field(default_factory=dict)


@dataclass(eq=False)
class Fused:
    """What :func:`fused` makes of a graph: the graph the plan walks (each region one call), the regions, and the program they came from."""

    graph: Graph
    regions: "list[Region]"
    program: "list[Node]"
    origin: "list[int]"  # program node -> the index of the node of the traced graph it came from
    rewrites: "list[dict[str, Any]]"
    settings: Fusion
    source: Graph  # the traced graph
    hooked: "list[Hooked]" = field(default_factory=list)  # regions fused into an extern's epilogue hook (docs/rfcs/0003-epilogue-fusion.md)
    # an extern call whose result a region reads that no hook took: {"call": program node, "region": index, "reason", "detail"}
    refused: "list[dict[str, Any]]" = field(default_factory=list)
    renamed: "dict[Value, Value]" = field(default_factory=dict)  # R_acc: a returned value the rewrite replaced -> its replacement

    @property
    def fused_regions(self) -> "list[Region]":
        """The regions of more than one op that a plan launches as kernels of their own: what an export verifies against their ops one by one (a
        region in an epilogue hook is verified with its extern, :attr:`hooked`)."""
        taken = {id(h.region) for h in self.hooked}
        return [r for r in self.regions if len(r.nodes) > 1 and id(r) not in taken]

    def hooked_region(self, r: Region) -> "Hooked | None":
        return next((h for h in self.hooked if h.region is r), None)


_lock = threading.Lock()
_kernels: "dict[str, Any]" = {}  # content hash -> the synthetic KernelOp (the same region in two graphs is one operator)


def fused(gop: Any, grouping: "str | None" = None) -> Fused:
    """The fusion of a graph (cached on it): ``grouping`` overrides the declaration's (``"none"``: the unfused composition)."""
    settings = settings_of(gop)
    if grouping is not None and grouping != settings.grouping:
        settings = settings.but(grouping=grouping)
    key = ("_tf_fused", settings)
    cache = gop.__dict__.setdefault("_tf_fused", {})
    found = cache.get(key)
    if found is None:
        found = _Pass(gop.graph, settings).run()
        found = cache.setdefault(key, found)
    return found


def with_version(gop: Any, version: int) -> Any:
    """The graph ``gop`` declared with fusion ``version`` instead of its own (its other settings kept): a copy, for a stale check to name the
    version a plan was made with."""
    s = settings_of(gop)
    other = object.__new__(type(gop))  # (not copy.copy: an operator pickles, so copies, by reference to itself)
    other.__dict__.update({k: v for k, v in gop.__dict__.items() if not k.startswith("_tf_")})
    other.fusion = s.but(version=version)
    return other


def other_version_of_plan(gop: Any, doc: Mapping[str, Any], derive: Any) -> "str | None":
    """When the plan document ``doc`` is not the plan of ``gop`` but is the plan of ``gop`` declared with another fusion version (D7: a plan made
    with another version is refused, naming it), what to say; ``None`` otherwise (or when the graph has no arithmetic). ``derive(graph)`` plans a
    graph for the document's buckets."""
    if not has_arithmetic(gop):
        return None
    mine = settings_of(gop).version
    for v in VERSIONS:
        if v == mine:
            continue
        try:
            if derive(with_version(gop, v)).stale_reason(doc) is None:
                return (
                    f"the plan was made with tilefish.fusion(version={v}) and the graph is planned with version {mine}"
                    + (" (the default)" if mine == DEFAULT_VERSION else "")
                    + f": declare @tilefish.graph(fusion=tilefish.fusion(version={v}, ...)) to keep that plan (every version is kept, D7), or export it "
                    "again (and pin the new plan)"
                )
        except Exception:  # noqa: BLE001 - another version that cannot plan the graph names nothing; the stale error stands
            continue
    return None


def planned_graph(gop: Any) -> Graph:
    """The graph a plan walks: the traced graph when its body has no arithmetic (the very object), else its fusion."""
    if not any(_is_arith(n) or n.op == "barrier" for n in gop.graph.nodes):
        return gop.graph
    return fused(gop).graph


def has_arithmetic(gop: Any) -> bool:
    return any(_is_arith(n) for n in gop.graph.nodes)


def _is_arith(n: Node) -> bool:
    return n.attrs.get("kind") is None and n.op not in _VIEW_OPS and n.op not in _MOVE_OPS and n.op != "barrier"


def _clean(attrs: Mapping[str, Any]) -> "dict[str, Any]":
    return {k: v for k, v in attrs.items() if k not in ("region", "loc")}


def _is_split_of(n: Node, y: Value, parts: int) -> bool:
    """Whether ``n`` splits ``y`` along its last axis into ``parts`` chunks of one size (the tiles of a hook that sees ``parts`` at once)."""
    if n.op != "split" or n.attrs.get("kind") is not None or n.inputs[0] is not y or "start" in n.attrs:
        return False
    sizes = list(n.attrs.get("sizes") or ())
    return n.attrs.get("axis") == y.type.rank - 1 and len(sizes) == parts and len(set(sizes)) == 1


# -- the pass ----------------------------------------------------------------------------------------------------------------------------


class _Pass:
    def __init__(self, graph: Graph, settings: Fusion) -> None:
        self.source = graph
        self.settings = settings
        self.name = graph.name
        self.ctx = self._context(graph)
        self.rewrites: "list[dict[str, Any]]" = []

    @staticmethod
    def _context(graph: Graph) -> Any:
        for v in graph.inputs:
            return v.ctx
        for n in graph.nodes:
            for o in n.outputs:
                return o.ctx
        return None

    def fail(self, message: str) -> Any:
        raise TilefishPlanError(f"graph '{self.name}': {message}")

    def new(self, t: TensorType) -> Value:
        return self.ctx.new_value(t)

    # -- 1. the numerics rewrites (on the IR, before any region is formed) -----------------------------------------------------------

    def rewrite(self) -> "tuple[list[Node], list[int]]":
        nodes = list(self.source.nodes)
        origin = list(range(len(nodes)))
        self.renamed: "dict[Value, Value]" = {}
        if any(epilogue_mode(n, self.settings) == "library" for n in nodes):  # first: the pattern is the graph's own arithmetic, as written
            nodes, origin = self._library_epilogues(nodes, origin)
        if self.settings.intermediates == "opmath":
            nodes, origin = self._opmath(nodes, origin)
        if self.settings.fp_contract == "mul-add":
            nodes, origin = self._mul_add(nodes, origin)
        nodes, origin = self._r_acc(nodes, origin)
        for n, o in zip(nodes, origin):  # an op a rewrite made is where the op it was made of was written (tilefish.explain)
            if n.loc is None:
                n.loc = self.source.nodes[o].loc
        return nodes, origin

    def _r_acc(self, nodes: "list[Node]", origin: "list[int]") -> "tuple[list[Node], list[int]]":
        """``R_acc`` (``docs/rfcs/0003-epilogue-fusion.md``, contract (ii)): a call of an extern with an ``acc`` hook whose result ``y`` (a 16-bit
        float) is widened, ``y.to(f32)``, in an ``"accumulator"`` mode (the graph's or a ``tilefish.fused`` scope's), becomes a call of its
        accumulator form, which also writes the f32 result ``acc``; every such widening is ``acc`` (of a split hook's chunk: that chunk of ``acc``),
        one rounding fewer. What reads ``y`` in its own type still reads it, rounded by the kernel. Defined on the IR, before any region is formed."""
        from ._epilogue import acc_form, hook_of

        if not any(n.attrs.get("kind") == "extern" and hook_of(n.callee, "acc") is not None for n in nodes):
            return nodes, origin
        consumers: "dict[Value, list[int]]" = {}
        for k, n in enumerate(nodes):
            for i in n.inputs:
                if isinstance(i, Value):
                    consumers.setdefault(i, []).append(k)
        # a widening a call updates in place stays its own buffer (the accumulator it would become is every widening's)
        in_place = {v for c in nodes if c.attrs.get("inout") and c.callee is not None
                    for p, v in zip(c.callee.signature.tensor_inputs, c.inputs) if p.access == "inout"}  # fmt: skip
        replace: "dict[Value, Value]" = {}
        insert: "dict[int, list[Node]]" = {}  # after node k: the nodes the rewrite adds
        swap: "dict[int, Node]" = {}
        drop: "set[int]" = set()  # the splits of a result that only the replaced widenings read, and those widenings
        escaping = {*self.source.outputs, *self.source.finals.values()}
        for k, n in enumerate(nodes):
            h = hook_of(n.callee, "acc") if n.attrs.get("kind") == "extern" and n.callee is not None else None
            if h is None:
                continue
            sig = n.callee.signature
            n_ret, outs = len(sig.returns), [p.name for p in sig.tensor_outputs]
            y = n.outputs[n_ret + outs.index(h.out)]
            if not isinstance(y.type.dtype, DType) or y.type.dtype.name not in _HALF:
                continue  # an f32 result: no rounding to remove

            def widening(m: int) -> bool:
                c = nodes[m]
                return (
                    c.op == "cast" and c.attrs.get("kind") is None and c.attrs.get("to") == f32 and "round" not in c.attrs
                    and epilogue_mode(c, self.settings) == "accumulator" and c.outputs[0] not in in_place
                )  # fmt: skip

            direct = [m for m in consumers.get(y, []) if widening(m)]
            chunks: "list[tuple[int, int, int]]" = []  # (split node, chunk, widening)
            if h.split > 1:
                for m in consumers.get(y, []):
                    s = nodes[m]
                    if _is_split_of(s, y, h.split):
                        for j, c in enumerate(s.outputs):
                            chunks += [(m, j, w) for w in consumers.get(c, []) if widening(w)]
            if not direct and not chunks:
                continue
            acc = acc_form(n.callee, h)
            y32 = self.new(y.type.materialized().with_dtype(f32))
            n_out = len(outs)
            call = Node(acc.name, n.inputs, {**{a: v for a, v in n.attrs.items() if a not in ("triton", "gluon")}, **acc.node_attrs()},
                        [*n.outputs[: n_ret + n_out], y32, *n.outputs[n_ret + n_out :]])  # fmt: skip
            call.callee, call.loc, call.scope = acc, n.loc, n.scope
            swap[k] = call
            removed: "list[int]" = []
            for w in direct:
                replace[nodes[w].outputs[0]] = y32
                removed.append(w)
            views: "dict[int, list[Value]]" = {}
            for m, j, w in chunks:
                if m not in views:
                    from .ops import _chunk_types

                    s = nodes[m]
                    outs32 = [self.new(t) for t in _chunk_types(y32, s.attrs["axis"], list(s.attrs["sizes"]))]
                    node = Node(s.op, [y32], dict(s.attrs), outs32)
                    node.loc, node.scope = s.loc, s.scope
                    insert.setdefault(k, []).append(node)
                    views[m] = outs32
                replace[nodes[w].outputs[0]] = views[m][j]
                removed.append(w)
            # a split of y whose chunks only the replaced widenings read goes with them: left in the program, the view would read y for nothing
            # and keep the rounded result live, so a hook that takes the region would still store it (the dual GEMM's pair writing [M, 2F] as
            # well as the hook's output); y itself is still written by the call, for whatever else reads it
            for m in sorted(views):
                readers = {c for o in nodes[m].outputs for c in consumers.get(o, [])}
                if readers <= set(removed) and not any(o in escaping for o in nodes[m].outputs):
                    drop.add(m)
                    drop.update(readers)
            self.rewrites.append({
                "kind": "accumulator", "node": origin[k], "op": n.op, "value": y.ref, "dtype": str(y.type.dtype),
                "widenings": sorted(origin[w] for w in removed),
            })  # fmt: skip
        if not swap:
            return nodes, origin
        out_nodes: "list[Node]" = []
        out_origin: "list[int]" = []
        for k, (n, o) in enumerate(zip(nodes, origin)):
            if k in drop:
                continue
            n = swap.get(k, n)
            if any(isinstance(i, Value) and i in replace for i in n.inputs):
                m = Node(n.op, [replace.get(i, i) if isinstance(i, Value) else i for i in n.inputs], n.attrs, n.outputs)
                m.callee, m.loc, m.scope, m.origin = n.callee, n.loc, n.scope, n.origin
                n = m
            out_nodes.append(n)
            out_origin.append(o)
            for extra in insert.get(k, []):
                out_nodes.append(extra)
                out_origin.append(o)
        # a widening the graph returns (or leaves in an InOut parameter) is the accumulator now
        self.renamed.update({v: replace[v] for v in (*self.source.outputs, *self.source.finals.values()) if v in replace})
        return out_nodes, out_origin

    def _opmath(self, nodes: "list[Node]", origin: "list[int]") -> "tuple[list[Node], list[int]]":
        """torch's eager rule for a 16-bit op with a python scalar: ``op(a.to(f32), f32(c)).to(dt)``, the scalar's own value rounded to f32."""
        out_nodes: "list[Node]" = []
        out_origin: "list[int]" = []
        for n, o in zip(nodes, origin):
            dt = n.outputs[0].type.dtype if n.outputs else None
            if (
                n.attrs.get("kind") is None
                and n.op in ("add", "sub", "mul")
                and isinstance(dt, DType)
                and dt.name in _HALF
                and any(isinstance(i, Scalar) for i in n.inputs)
            ):
                ins: "list[Any]" = []
                made: "list[Node]" = []
                for i in n.inputs:
                    if isinstance(i, Scalar):
                        ins.append(Scalar(float(i.value), f32, i.name))
                    else:
                        wide = self.new(i.type.materialized().with_dtype(f32))
                        made.append(Node("cast", [i], {"to": f32, "rounding": False}, [wide]))
                        ins.append(wide)
                r = self.new(n.outputs[0].type.with_dtype(f32))
                made.append(Node(n.op, ins, _clean(n.attrs), [r]))
                made.append(Node("cast", [r], {"to": dt, "rounding": True}, [n.outputs[0]]))
                for m in made:  # in the op's tilefish.fused scope: its widenings are R_acc's under that scope's epilogues
                    m.scope = n.scope
                out_nodes += made
                out_origin += [o] * len(made)
                self.rewrites.append({"kind": "opmath", "node": o, "op": n.op})
                continue
            out_nodes.append(n)
            out_origin.append(o)
        return out_nodes, out_origin

    def _mul_add(self, nodes: "list[Node]", origin: "list[int]") -> "tuple[list[Node], list[int]]":
        """``fp_contract="mul-add"``: an f32 add or sub of a product that nothing else reads becomes one ``fma`` (one rounding)."""
        uses: "dict[Value, int]" = {}
        for n in nodes:
            for i in n.inputs:
                if isinstance(i, Value):
                    uses[i] = uses.get(i, 0) + 1
        for v in (*self.source.outputs, *self.source.finals.values()):
            uses[v] = uses.get(v, 0) + 1
        producer = {o: n for n in nodes for o in n.outputs}

        def product(x: Any) -> "Node | None":
            if not isinstance(x, Value):
                return None
            p = producer.get(x)
            if p is None or p.op != "mul" or p.attrs.get("kind") is not None or uses.get(x, 0) != 1:
                return None
            if x.type.dtype != f32 or any(isinstance(i, Value) and i.type.dtype != f32 for i in p.inputs):
                return None
            return p

        removed: "set[int]" = set()
        replaced: "dict[int, list[Node]]" = {}
        index_of = {id(n): k for k, n in enumerate(nodes)}
        for k, n in enumerate(nodes):
            if n.attrs.get("kind") is not None or n.op not in ("add", "sub") or n.outputs[0].type.dtype != f32:
                continue
            left, right = n.inputs
            pl, pr = product(left), product(right)
            new_nodes: "list[Node]" = []
            if n.op == "add" and (pl is not None or pr is not None):
                p, other = (pl, right) if pl is not None else (pr, left)
                a, b = p.inputs  # type: ignore[union-attr]
                new_nodes.append(Node("fma", [a, b, other], {}, [n.outputs[0]]))
            elif n.op == "sub" and pl is not None:  # a * b - c = fma(a, b, -c)
                a, b = pl.inputs
                c = self._negated(right, new_nodes)
                new_nodes.append(Node("fma", [a, b, c], {}, [n.outputs[0]]))
                p = pl
            elif n.op == "sub" and pr is not None:  # c - a * b = fma(-a, b, c)
                a, b = pr.inputs
                na = self._negated(a, new_nodes)
                new_nodes.append(Node("fma", [na, b, left], {}, [n.outputs[0]]))
                p = pr
            else:
                continue
            pi = index_of[id(p)]
            if pi in removed:  # pragma: no cover - a product has one use
                continue
            removed.add(pi)
            for m in new_nodes:  # in the op's tilefish.fused scope, as the ops _opmath makes
                m.scope = n.scope
            replaced[k] = new_nodes
            self.rewrites.append({"kind": "mul-add", "node": origin[k], "product": origin[pi]})
        out_nodes: "list[Node]" = []
        out_origin: "list[int]" = []
        for k, (n, o) in enumerate(zip(nodes, origin)):
            if k in removed:
                continue
            for m in replaced.get(k, [n]):
                out_nodes.append(m)
                out_origin.append(o)
        return out_nodes, out_origin

    def _library_epilogues(self, nodes: "list[Node]", origin: "list[int]") -> "tuple[list[Node], list[int]]":
        """``epilogues="library"``: a cuBLASLt GEMM and the arithmetic after it that one of its epilogues computes become one call of a
        ``cublaslt.matmul`` (:func:`match_library_epilogue`); the other ops are left as they are (a cast that only the epilogue read is dead
        code then, which the regions drop)."""
        from . import _matmul

        uses: "dict[Value, list[int]]" = {}
        for k, n in enumerate(nodes):
            for i in n.inputs:
                if isinstance(i, Value):
                    uses.setdefault(i, []).append(k)
        escapes = set(self.source.outputs) | set(self.source.finals.values())
        barriered = {i for n in nodes if n.op == "barrier" and n.attrs.get("kind") is None for i in n.inputs if isinstance(i, Value)}
        producer = {o: k for k, n in enumerate(nodes) for o in n.outputs}
        removed: "set[int]" = set()
        replaced: "dict[int, Node]" = {}
        for k, n in enumerate(nodes):
            if n.attrs.get("kind") != "library" or getattr(n.callee, "impl", None) not in ("cublaslt.gemm", "cublaslt.gemm_bias"):
                continue
            found = match_library_epilogue(nodes, k, uses, escapes | barriered, producer)
            if isinstance(found, str):
                self.rewrites.append({
                    "kind": "library-epilogue", "node": origin[k], "call": n.callee.name, "matched": False, "applied": False, "reason": found,
                    **_accumulator_note(nodes, k, uses, self.settings),
                })  # fmt: skip
                continue
            chain, epilogue, beta, bias, c, z = found
            # the pattern's ops: the chain and the widenings of c and the bias it reads (a widening in an "accumulator" scope is R_acc's)
            ops = set(chain) | {producer[i] for j in chain for i in nodes[j].inputs if isinstance(i, Value) and i in producer and _is_widening(nodes[producer[i]])}
            modes = sorted({epilogue_mode(nodes[j], self.settings) for j in ops})
            if modes != ["library"]:  # (a tilefish.fused scope: the pattern's ops are not all in a "library" mode, so none is rewritten)
                self.rewrites.append({
                    "kind": "library-epilogue", "node": origin[k], "call": n.callee.name, "matched": True, "applied": False, "epilogue": epilogue,
                    "beta": beta, "nodes": sorted(origin[j] for j in chain),
                    "rounding_removed": {"value": n.outputs[0].ref, "dtype": str(n.outputs[0].type.dtype)}, "contract": "measured-exact",
                    "mode": "/".join(modes),
                    "note": _ACC_LEAVES_LIBRARY if modes == ["accumulator"] else (
                        f"the ops of the pattern are in the modes {modes} (tilefish.fused scopes): it is rewritten only where all are in 'library'"),
                })  # fmt: skip
                continue
            op = _matmul.epilogue_op(n.callee, epilogue, beta)
            ins = [n.inputs[0], n.inputs[1]] + ([bias] if bias is not None else []) + ([c] if c is not None else [])
            call = Node(op.name, ins, {"kind": "library", **op.node_attrs()}, [z])
            call.callee = op
            call.loc, call.scope = n.loc, n.scope
            replaced[k] = call
            removed.update(chain)
            self.rewrites.append({
                "kind": "library-epilogue",
                "node": origin[k],
                "call": n.callee.name,
                "matched": True,
                "applied": True,
                "into": op.name,
                "epilogue": epilogue,
                "beta": beta,
                "nodes": sorted(origin[j] for j in chain),
                "rounding_removed": {"value": n.outputs[0].ref, "dtype": str(n.outputs[0].type.dtype)},
                "contract": "measured-exact",
            })  # fmt: skip
        out_nodes: "list[Node]" = []
        out_origin: "list[int]" = []
        for k, (n, o) in enumerate(zip(nodes, origin)):
            if k in removed:
                continue
            out_nodes.append(replaced.get(k, n))
            out_origin.append(o)
        return out_nodes, out_origin

    def _negated(self, x: Any, into: "list[Node]") -> Any:
        if isinstance(x, Scalar):
            return Scalar(-float(x.value), x.dtype, None)  # the sign of an f32 constant flipped: exact
        neg = self.new(x.type.materialized())
        into.append(Node("neg", [x], {}, [neg]))
        return neg

    # -- 2. the regions ------------------------------------------------------------------------------------------------------------

    def prepare(self, prog: "list[Node]", origin: "list[int]", renamed: "Mapping[Value, Value] | None" = None) -> None:
        """The tables every step of the pass reads: producers, consumers, what escapes the arithmetic, the edges, the free ops."""
        self.prog, self.origin = prog, origin
        if renamed is not None:
            self.renamed = dict(renamed)
        self.renamed = getattr(self, "renamed", {})
        self.producer: "dict[Value, int]" = {o: k for k, n in enumerate(prog) for o in n.outputs}
        self.consumers: "dict[Value, list[int]]" = {}
        for k, n in enumerate(prog):
            for i in n.inputs:
                if isinstance(i, Value):
                    self.consumers.setdefault(i, []).append(k)
        self.escapes: "set[Value]" = {self.renamed.get(v, v) for v in (*self.source.outputs, *self.source.finals.values())}
        self.succ = self._edges()
        self.free = {k for k, n in enumerate(prog) if _is_arith(n) and n.op in _FREE_OPS}
        self.masks_u8: "dict[Value, Value]" = {}
        self.dead: "set[int]" = set()
        # tilefish.barrier(v): v is a boundary of the regions (its consumers read it from its buffer)
        self.barriered = {i for n in prog if n.op == "barrier" and n.attrs.get("kind") is None for i in n.inputs if isinstance(i, Value)}

    def run(self) -> Fused:
        prog, origin = self.rewrite()
        for k, n in enumerate(prog):
            if n.attrs.get("kind") is None and n.op in _REFUSED_OPS:
                self.fail(f"the op '{n.op}' (node #{origin[k]}): {_REFUSED_OPS[n.op]}")
        self.prepare(prog, origin)
        nodes = self.select()
        regions = self.regions(self.partition(nodes, self.settings.grouping))
        self.hooked: "list[Hooked]" = []
        self.refused: "list[dict[str, Any]]" = []
        if self.settings.grouping == "auto":  # (the unfused composition calls every extern as it is)
            self.epilogues(regions)
        graph = self._graph(regions)
        return Fused(graph, regions, prog, origin, list(self.rewrites), self.settings, self.source, self.hooked, self.refused, dict(self.renamed))

    def select(self) -> "list[int]":
        """The ops the regions are made of (after :meth:`prepare`): every arithmetic op that is neither free (copied into each region that reads it)
        nor dead (nothing reads its value)."""
        prog, origin = self.prog, self.origin
        # a free op whose value something outside the arithmetic reads (a call, a move, a barrier, the graph's return) is a region of its own
        for k in sorted(self.free):
            v = prog[k].outputs[0]
            outside = [c for c in self.consumers.get(v, []) if not _is_arith(prog[c])]
            if outside or v in self.escapes:
                if prog[k].op == "load":
                    self.fail(
                        f"the value {v.ref} read by tilefish.load (node #{origin[k]}) is used outside the arithmetic (returned, stored or passed to a "
                        "call), which a scalar cannot be: compute a tensor from it"
                    )
                self.free.discard(k)
        # an op whose value nothing reads (no output, no call, no move, no barrier: dead code) is not computed, as in the unfused composition
        live: "set[Value]" = set(self.escapes)
        for n in prog:
            if not _is_arith(n):
                live.update(i for i in n.inputs if isinstance(i, Value))
        self.dead: "set[int]" = set()
        for k in range(len(prog) - 1, -1, -1):
            n = prog[k]
            if not _is_arith(n):
                continue
            if any(o in live for o in n.outputs):
                live.update(i for i in n.inputs if isinstance(i, Value))
            else:
                self.dead.add(k)
        self.free -= self.dead
        return [k for k, n in enumerate(prog) if _is_arith(n) and k not in self.free and k not in self.dead]

    def partition(self, nodes: "list[int]", grouping: str) -> "list[list[int]]":
        """The regions of ``nodes`` (arithmetic ops that are not free): one per op, merged producer into consumer in IR order (``"auto"``); then a
        region whose outputs have a dimension no input of it has (``iota(x, 0) < load(n)`` alone) takes its first consumer, which reads a tensor
        of that shape, so that every region is a kernel that can be launched (exact: the ops of such a chain compare and count)."""
        prog = self.prog
        self.member_of: "dict[int, int]" = {k: k for k in nodes}
        self.groups: "dict[int, list[int]]" = {k: [k] for k in nodes}
        # an ordered sum with fma=True adds fma(a, b, acc) of its product's operands: the product's multiply is part of the sum (it is never rounded
        # on its own for the sum), so the two are one unit in every grouping, the unfused composition's too (a kernel of the multiply and the sum)
        for k in nodes:
            n = prog[k]
            if n.op == "sum" and n.attrs.get("fma") and isinstance(n.inputs[0], Value):
                p = self.producer.get(n.inputs[0])
                if p is not None and p in self.member_of and self.member_of[p] != self.member_of[k]:
                    self._join(self.member_of[p], self.member_of[k])
        if grouping == "auto":
            for k in nodes:
                if self.solo_region(self.member_of[k]):
                    continue
                for i in prog[k].inputs:
                    if not isinstance(i, Value) or i in self.barriered:
                        continue
                    p = self.producer.get(i)
                    if p is None or p not in self.member_of or self.solo_region(self.member_of[p]):
                        continue
                    a, b = self.member_of[p], self.member_of[k]
                    if a != b:
                        self._try_merge(a, b)
        changed = True
        while changed:
            changed = False
            for root in sorted(self.groups, key=lambda r: min(self.groups[r])):
                members = sorted(self.groups[root])
                region = self._region(-1, members)
                have = frozenset(d for v in region.inputs if not v.type.devscalar for s in v.type.shape for d in s.free_dims())
                need = frozenset(d for o in region.outputs for s in o.type.shape for d in s.free_dims())
                if need <= have:
                    continue
                consumers = sorted(
                    c for o in region.outputs for c in self.consumers.get(o, []) if c in self.member_of and self.member_of[c] != root
                )
                if not consumers or not self._try_merge(root, self.member_of[consumers[0]], force=True):
                    missing = ", ".join(sorted(d.name for d in need - have))
                    self.fail(
                        f"the ops {[prog[k].op for k in members]} (nodes {sorted({self.origin[k] for k in members})}) make a tensor whose dimension "
                        f"{missing} none of their inputs has, and cannot be joined to an op that reads one: combine the result with a tensor of that shape"
                    )
                changed = True
                break
        return [sorted(self.groups[r]) for r in sorted(self.groups, key=lambda r: min(self.groups[r]))]

    def regions(self, parts: "list[list[int]]") -> "list[Region]":
        regions = [self._region(ri, members) for ri, members in enumerate(parts)]
        for r in regions:  # a mask leaves its region as u8 (0 or 1): every consumer region reads that value
            for o in r.outputs:
                if isinstance(o.type.dtype, DType) and o.type.dtype.is_bool:
                    self.masks_u8[o] = self.new(TensorType(o.type.shape, u8))
        for r in regions:
            self._kernel(r)
        return regions

    def _edges(self) -> "list[set[int]]":
        """The successors of every program node: data edges and the order of the accesses to a storage that something overwrites (a store into a
        parameter, an in-place call): every read of a storage before the write comes before it, every read after comes after. All go forward."""
        prog = self.prog
        succ: "list[set[int]]" = [set() for _ in prog]
        root: "dict[Value, Value]" = {}

        def of(v: Value) -> Value:
            return root.get(v, v)

        consumed_by: "dict[Value, list[int]]" = {}  # storage root -> the nodes that overwrite it
        readers: "dict[Value, list[int]]" = {}
        for k, n in enumerate(prog):
            for i in n.inputs:
                if isinstance(i, Value):
                    p = self.producer.get(i)
                    if p is not None:
                        succ[p].add(k)
                    readers.setdefault(of(i), []).append(k)
            kind = n.attrs.get("kind")
            if kind is None and n.op in _VIEW_OPS:
                for o in n.outputs:
                    root[o] = of(n.inputs[0])  # type: ignore[arg-type]
            elif kind is None and n.op == "store" and n.outputs:
                for o in n.outputs:
                    root[o] = of(n.inputs[0])  # type: ignore[arg-type]
                consumed_by.setdefault(of(n.inputs[0]), []).append(k)  # type: ignore[arg-type]
            elif kind in ("kernel", "extern", "library") and n.attrs.get("inout"):
                inout = list(n.attrs["inout"])
                params = [p for p in n.callee.signature.tensor_inputs] if n.callee is not None else []
                consumed = [v for p, v in zip(params, n.inputs) if p.name in inout]
                n_new = len(n.outputs) - len(consumed)
                for version, c in zip(n.outputs[n_new:], consumed):
                    root[version] = of(c)  # type: ignore[arg-type]
                    consumed_by.setdefault(of(c), []).append(k)  # type: ignore[arg-type]
        for r, writers in consumed_by.items():
            for w in writers:
                for k in readers.get(r, []):
                    if k < w:
                        succ[k].add(w)
                    elif k > w:
                        succ[w].add(k)
        return succ

    def merge_problem(self, a: int, b: int, force: bool = False) -> "tuple[str, str] | None":
        """The first rule of 3.5 the union of the regions whose first ops are ``a`` and ``b`` breaks, as ``(reason, what)``: ``"limit"`` (more ops or
        tensors than a region has), ``"cycle"`` (a path leaves the union and comes back: through a call, a view, a move or another region), or, for
        a path through a reduction that is a region of its own in this version (:func:`is_solo`), the reduction's reason (:func:`reduction_reason`),
        ``"template"`` (the union does not lower); ``None`` when the union is legal. The pass merges only when it is ``None``."""
        union = sorted(self.groups[a] + self.groups[b])
        if len(union) > MAX_REGION_NODES and not force:
            return "limit", f"together they are {len(union)} ops, more than the {MAX_REGION_NODES} of a region"
        members = set(union)
        last = union[-1]
        # a cycle: a path that leaves the union and comes back (every edge goes forward, so look only before its last node)
        stack = [s for k in union for s in self.succ[k] if s not in members and s < last]
        seen: "set[int]" = set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            for s in self.succ[x]:
                if s in members:
                    n = self.prog[x]
                    what = f"the call to '{n.op}'" if n.attrs.get("kind") else f"the op '{n.op}'"
                    detail = f"one region of both would make a cycle through {what} (node #{self.origin[x]}), which reads one and feeds the other"
                    reason = reduction_reason(n.op, "order" in n.attrs, self.settings.version) if _is_arith(n) else None
                    return (reason or "cycle"), detail  # (through a reduction of version 1: a row region of version 2 takes all three)
                if s < last and s not in seen:
                    stack.append(s)
        try:
            region = self._region(-1, union, trial=True)
            tensors = len(region.inputs) + len(region.outputs)
            if tensors > MAX_REGION_TENSORS and not force:
                return "limit", f"together they read and write {tensors} tensors, more than the {MAX_REGION_TENSORS} of a region"
            self._kernel(region, trial=True)
        except (TilefishLoweringError, TilefishPlanError) as e:
            first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            return "template", f"one kernel of both does not lower: {first}"
        return None

    def _try_merge(self, a: int, b: int, force: bool = False) -> bool:
        if self.merge_problem(a, b, force) is not None:
            return False
        self._join(a, b)
        return True

    def _join(self, a: int, b: int) -> None:
        union = sorted(self.groups[a] + self.groups[b])
        keep, gone = (a, b) if a < b else (b, a)
        self.groups[keep] = union
        del self.groups[gone]
        for k in union:
            self.member_of[k] = keep

    def solo(self, k: int) -> bool:
        """Whether the program node ``k`` is a region of its own in this version (:func:`is_solo`)."""
        return is_solo(self.prog[k].op, self.prog[k].attrs, self.settings.version)

    def solo_region(self, root: int) -> bool:
        """Whether the region whose first op is ``root`` holds an op that is a region of its own (with, for an fma sum, its product): it takes no
        other op."""
        return any(self.solo(k) for k in self.groups[root])

    def _region(self, index: int, nodes: "list[int]", trial: bool = False) -> Region:
        prog = self.prog
        members = set(nodes)
        free: "list[int]" = []
        inputs: "list[Value]" = []
        seen: "set[Value]" = set()

        def take(v: Value) -> None:
            p = self.producer.get(v)
            if p is not None and p in self.free:
                if p not in free:
                    free.append(p)
                    for i in prog[p].inputs:
                        if isinstance(i, Value):
                            take(i)
                return
            if v not in seen:
                seen.add(v)
                inputs.append(v)

        for k in nodes:
            for i in prog[k].inputs:
                if isinstance(i, Value) and self.producer.get(i) not in members:
                    take(i)
        outputs: "list[Value]" = []
        for k in nodes:
            for o in prog[k].outputs:
                outside = [c for c in self.consumers.get(o, []) if c not in members and c not in self.dead]
                if outside or o in self.escapes:
                    outputs.append(o)
        if not outputs:
            if trial:
                raise TilefishPlanError("no output")
            outputs = [prog[nodes[-1]].outputs[0]]  # a value nothing reads: still computed, as in the unfused composition
        r = Region(index, nodes, free, inputs, outputs)
        return r

    # -- 3. a region's kernel ------------------------------------------------------------------------------------------------------

    def _params(self, r: Region, strided: bool) -> "tuple[list[tuple[str, TensorType, Value, bool]], set[Value], list[str], dict[Value, str], list[Value]]":
        """The parameters of a region's kernel: ``(name, type, value, strided)`` per input, the inputs that are masks (``u8`` here), the
        parameters compiled for an element-aligned pointer, how each input is read, and what the call passes. A view that is not dense is read
        where it is (``strided``: a ``Strided[...]`` parameter with a stride per axis) or materialised by ``tilefish.contiguous`` first."""
        params: "list[tuple[str, TensorType, Value, bool]]" = []
        mask_inputs: "set[Value]" = set()
        unaligned: "list[str]" = []
        reads: "dict[Value, str]" = {}
        call_inputs: "list[Value]" = []
        for k, v in enumerate(r.inputs):
            name = f"a{k}"
            t = v.type
            if isinstance(t.dtype, DType) and t.dtype.is_bool:
                mask_inputs.add(v)
                u = self.masks_u8.get(v)
                params.append((name, TensorType(t.shape, u8), v, False))
                call_inputs.append(u if u is not None else v)
                reads[v] = "mask"
                continue
            if t.devscalar:
                params.append((name, t, v, False))
                call_inputs.append(v)
                reads[v] = "devscalar"
                continue
            view = False
            if t.contiguous:
                reads[v] = "contiguous"
            elif t.strides is None and t.offset is not None:
                reads[v] = "unaligned"
                unaligned.append(name)
            else:
                if t.offset is None or any(s is None for s in t.layout_strides()):
                    self.fail(f"{v.ref} ({t}) is a view whose layout is unknown, which a fused region cannot read (pass a tensor)")
                if not 1 <= t.rank <= 4:
                    self.fail(f"{v.ref} ({t}) is a view of rank {t.rank} that is not dense; a fused region reads one of rank 1 to 4")
                view = strided
                reads[v] = "strided" if strided else "materialised"
            params.append((name, TensorType(t.shape, t.dtype), v, view))
            call_inputs.append(v)
        return params, mask_inputs, unaligned, reads, call_inputs

    def _kernel(self, r: Region, trial: bool = False) -> None:
        out_types: "list[TensorType]" = []
        for o in r.outputs:
            t = o.type
            out_types.append(TensorType(t.shape, u8) if isinstance(t.dtype, DType) and t.dtype.is_bool else TensorType(t.shape, t.dtype))
        body = [self.prog[k] for k in sorted(r.free)] + [self.prog[k] for k in r.nodes]
        params, mask_inputs, unaligned, reads, call_inputs = self._params(r, strided=True)
        try:
            kernel = _synthetic(params, mask_inputs, body, r.outputs, out_types, self.settings, self.name, trial)
        except TilefishLoweringError:
            if not any(s for *_, s in params):
                raise
            # the template cannot read the view where it is (a row region, a view of another shape): a copy of it first
            params, mask_inputs, unaligned, reads, call_inputs = self._params(r, strided=False)
            kernel = _synthetic(params, mask_inputs, body, r.outputs, out_types, self.settings, self.name, trial)
        if trial:
            return
        r.kernel = kernel
        r.call = kernel.unaligned(*unaligned) if unaligned else kernel
        r.reads = reads
        r.call_inputs = call_inputs  # type: ignore[attr-defined]

    # -- 4. epilogue hooks (docs/rfcs/0003-epilogue-fusion.md) ------------------------------------------------------------------------

    def epilogues(self, regions: "list[Region]") -> None:
        """Fuse each region that reads the result of an extern call with an epilogue hook into that call, where it fits the hook: one call of the
        extern with the region in its hook (``_epilogue.hooked``) in place of the call, the region and the views of the result only they read. A
        call of an extern's accumulator form (``R_acc``) fuses through its ``acc`` hook (contract (ii)), any other through its ``out`` hook (contract
        (i)). One region per call, the first that fits; every region that reads a hooked extern's result and is not fused is in ``self.refused``,
        with why."""
        from ._epilogue import hook_of, hooks_of_op

        by_root = {r.nodes[0]: r for r in regions}
        taken: "dict[int, Hooked]" = {}
        # the units of the graph so far (a region, a call with the region in its hook, one node): what a merge's cycle check walks
        self.unit_key: "dict[int, Any]" = {x: ("r", r.index) for r in regions for x in r.nodes}
        self.unit_nodes: "dict[Any, list[int]]" = {("r", r.index): list(r.nodes) for r in regions}
        for k, n in enumerate(self.prog):
            if n.attrs.get("kind") != "extern" or n.callee is None:
                continue
            acc = getattr(n.callee, "acc_of", None)
            if acc is not None:
                extern, hook = acc
            elif hooks_of_op(n.callee):
                extern, hook = n.callee, hook_of(n.callee, "out")
            else:
                continue
            values, views = self._tile_values(n, extern, hook if hook is not None else hook_of(extern, "acc"), acc is not None)
            readers = sorted({self.member_of[c] for v in [*values, *self._result_values(n, extern)] for c in self.consumers.get(v, []) if c in self.member_of})
            done: "Hooked | None" = None
            for root in readers:
                r = by_root[root]
                if hook is None:
                    mode = epilogue_mode(self.prog[r.nodes[0]], self.settings)
                    self._refuse(k, r, "epilogue-acc-only", f"'{extern.name}' has an epilogue hook on its accumulator only, which this region does not use (epilogues='{mode}' here: only a "
                                 "widening y.to(f32) of the result in an 'accumulator' mode is R_acc's)")
                    continue
                if done is not None:
                    self._refuse(k, r, "epilogue-taken", f"region {done.region.index} is fused into the call to '{extern.name}' already (one region per call)")
                    continue
                other = taken.get(id(r))
                if other is not None:  # a region that reads the results of two calls fuses into the first's hook; the second's is stored
                    self._refuse(k, r, "epilogue-taken", f"region {r.index} is fused into the call to '{other.extern.name}' (node #{self.origin[other.call]}) already (one call per region)")
                    continue
                found = self._hook(k, n, extern, hook, acc is not None, r, values, views)
                if isinstance(found, Hooked):
                    self.hooked.append(found)
                    taken[id(r)] = found
                    done = found
                    key = ("h", k)
                    self.unit_nodes[key] = sorted(found.unit)
                    self.unit_nodes.pop(("r", r.index), None)
                    for x in found.unit:
                        self.unit_key[x] = key
                else:
                    self._refuse(k, r, *found)

    def _refuse(self, k: int, r: Region, reason: str, detail: str) -> None:
        self.refused.append({"call": k, "region": r.index, "reason": reason, "detail": detail})

    def _result_values(self, n: Node, extern: Any) -> "list[Value]":
        """The outputs of the extern call ``n`` and every view of them."""
        sig = extern.signature
        return self._with_views(n.outputs[len(sig.returns) : len(sig.returns) + len(sig.tensor_outputs) + (1 if getattr(n.callee, "acc_of", None) else 0)])

    def _with_views(self, values: "Sequence[Value]") -> "list[Value]":
        """``values`` and every view of them."""
        out = list(values)
        todo = list(out)
        while todo:
            v = todo.pop()
            for c in self.consumers.get(v, []):
                m = self.prog[c]
                if m.attrs.get("kind") is None and m.op in _VIEW_OPS:
                    out += m.outputs
                    todo += m.outputs
        return out

    def _tile_values(self, n: Node, extern: Any, hook: Any, is_acc: bool) -> "tuple[dict[Value, tuple[str, int | None]], dict[int, list[Value]]]":
        """What the hook of the call ``n`` holds: value -> ``("tile", chunk)`` or ``("rounded", chunk)`` (an ``acc`` hook's tile rounded as the
        kernel rounds it), and the split nodes whose chunks they are."""
        sig = extern.signature
        n_ret = len(sig.returns)
        outs = [p.name for p in sig.tensor_outputs]
        y = n.outputs[n_ret + outs.index(hook.out)]
        values: "dict[Value, tuple[str, int | None]]" = {}
        if is_acc:
            values[n.outputs[n_ret + len(outs)]] = ("tile", None)
            values[y] = ("rounded", None)
        else:
            values[y] = ("tile", None)
        views: "dict[int, list[Value]]" = {}
        if hook.split > 1:  # a hook that holds several tiles: the chunks of a split of the result along its last axis
            whole = dict(values)
            values = {}
            for v, (what, _) in whole.items():
                for m in self.consumers.get(v, []):
                    if _is_split_of(self.prog[m], v, hook.split):
                        views[m] = list(self.prog[m].outputs)
                        for j, c in enumerate(self.prog[m].outputs):
                            values[c] = (what, j)
        return values, views

    def _cycle(self, unit: "set[int]") -> "int | None":
        """A node outside ``unit`` on a path from it back into it through the units of the graph so far (``self.unit_key``: a region, a call with a
        region in its hook, else one node), or None. The units' graph, not the nodes': a path may leave by one node of a region and come back from
        another."""

        def key(x: int) -> Any:
            return None if x in unit else self.unit_key.get(x, x)

        stack = [key(s) for u in unit for s in self.succ[u] if s not in unit]
        seen: "set[Any]" = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            for x in self.unit_nodes.get(u, ()) if isinstance(u, tuple) else (u,):
                for s in self.succ[x]:
                    ks = key(s)
                    if ks is None:
                        return x
                    if ks not in seen:
                        stack.append(ks)
        return None

    def _hook(self, k: int, n: Node, extern: Any, hook: Any, is_acc: bool, r: Region, values: "Mapping[Value, Any]", views: "Mapping[int, list[Value]]") -> Any:
        """The :class:`Hooked` of the call ``k`` and the region ``r``, or ``(reason, detail)``: why the hook does not take it."""
        from . import _epilogue
        from .aot import _bucket_dims, compile_specs

        sig = extern.signature
        n_ret, n_out = len(sig.returns), len(sig.tensor_outputs)
        members = set(r.nodes)
        made = set(self._result_values(n, extern))
        for v in r.inputs:
            if v in made and v not in values:
                return "epilogue-view", f"region {r.index} reads {v.ref}, a part of the result of '{extern.name}' the hook does not hold (a hook holds its tile" + (
                    f"s, the {hook.split} chunks of a split of the last axis)" if hook.split > 1 else ")")
        # what the call updates in place (InOut): the kernel writes it, so the hook can read neither the version it writes nor the one it replaces
        n_callee_out = len(n.callee.signature.tensor_outputs)
        updated = set(self._with_views([*n.outputs[n_ret + n_callee_out :], *(v for p, v in zip(sig.tensor_inputs, n.inputs) if p.access == "inout")]))
        for v in r.inputs:
            if v in updated:
                return "epilogue-inout", f"region {r.index} reads {v.ref}, which '{extern.name}' updates in place (InOut): its kernel would read what it writes"
        used = [v for v in r.inputs if v in values]
        held = set(used)  # (a set: == of two values is an op of the trace)
        if any(v in self.barriered for v in (*values, *made)):  # tilefish.barrier keeps the result stored and stops any epilogue fusion of it
            return "barrier", f"the result of '{extern.name}' passes tilefish.barrier, which keeps it a boundary of the regions and of the call's hook"
        unit_views = [m for m, cs in views.items() if any(c in held for c in cs)]
        unit = {k, *members, *unit_views}

        def needed(v: Value) -> bool:
            return v in self.escapes or v in self.barriered or any(c not in unit and c not in self.dead for c in self.consumers.get(v, []))

        y = n.outputs[n_ret + [p.name for p in sig.tensor_outputs].index(hook.out)]
        rounded = [y, *(c for m in views for c in self.prog[m].outputs if values.get(c, ("", 0))[0] == ("rounded" if is_acc else "tile"))]
        y_needed = any(needed(v) for v in rounded)
        if is_acc:
            accs = [v for v, (what, _) in values.items() if what == "tile"]
            acc_value = n.outputs[n_ret + n_out]
            if any(needed(v) for v in (acc_value, *accs)):
                return "epilogue-acc-shared", f"the f32 result of '{extern.name}' (R_acc) is read outside region {r.index}, which a hook cannot store as well"
        # a path from the unit out of it and back into it, through the units the graph has so far (a call, a view, a move, another region, a call
        # with a region in its hook): one kernel would wait for itself
        x = self._cycle(unit)
        if x is not None:
            what = f"the call to '{self.prog[x].op}'" if self.prog[x].attrs.get("kind") else f"the op '{self.prog[x].op}'"
            return "epilogue-cycle", f"{what} (node #{self.origin[x]}) is on a path from the call to '{extern.name}' into region {r.index}"
        tile_value = next(iter(values))
        tile_type = tile_value.type
        if any(self.prog[m].op in ("sum", "mean", "max", "argmax", "argmin") for m in r.nodes):  # a row region: whole rows in a tile
            if extern.language == "gluon":
                return "epilogue-row", f"region {r.index} reduces rows, which a hook fuses into a Triton kernel only (a Gluon kernel's tile has its layout)"
            if hook.block_n is None:
                return "epilogue-row", f"region {r.index} reduces rows, which a hook takes only where a tile holds whole rows: declare the tile's width (tilefish.hook(..., block_n=...))"
            d = tile_type.shape[-1].folded()
            if not d.is_static:
                return "epilogue-row", f"region {r.index} reduces rows of {d} columns; a hook takes rows of a static length"
            q = next(p for p in sig.params if p.name == hook.block_n)
            widths = [c["consts"].get(hook.block_n, q.default) for c in extern.bucket_configs.values()] or [q.default]
            narrow = [w for w in widths if not isinstance(w, int) or w < d.value]
            if narrow:
                return "epilogue-row", f"region {r.index} reduces rows of {d.value} columns, but a tile of '{extern.name}' is {narrow[0]} columns wide ({hook.block_n}): a hook takes a norm only where a tile holds whole rows"
        # the region's types are the graph's: the extern's dimensions in their place, where a call binds one to the other
        mapping: "dict[Dim, DimExpr]" = {}
        for p, v in [*zip(sig.tensor_inputs, n.inputs), *zip(sig.tensor_outputs, n.outputs[n_ret:])]:
            for s, g in zip(p.type.shape, v.type.shape):
                g = g.folded()
                if isinstance(g, Dim) and not g.is_static and g not in mapping:
                    mapping[g] = s
        stub_dims = {d.name: d for p in sig.params for d in (p.type.free_dims() if p.type is not None else p.dim.free_dims() if p.dim is not None else ())}

        def translate(d: Any) -> Any:
            d = as_dim(d).folded()
            if d.is_static:
                return d
            if all(x in mapping for x in d.free_dims()):
                return d.subst(mapping)
            return d

        out_param = next(p for p in sig.params if p.name == hook.out)
        main: "int | None" = None
        out_dtype = None
        if not y_needed:
            for i, o in enumerate(r.outputs):
                dt = o.type.dtype
                if tuple(o.type.shape) != tuple(tile_type.shape) or not isinstance(dt, DType) or dt.name not in ("f32", "bf16", "f16"):
                    continue
                if hook.value == "out" and dt != y.type.dtype:
                    continue
                if out_param.tensor_map is not None and dt != out_param.type.dtype:
                    continue
                main, out_dtype = i, dt
                break
            # (no output of the result's shape and type: the kernel stores the result all the same, into a buffer nothing reads, and every output of
            # the region is a side output: still one launch and one read of the result fewer than the region after the call)
        out_type = None
        if main is not None:  # the hook's output holds the region's output instead: its dtype, and one chunk of the columns for a split hook
            shape = list(out_param.type.shape)
            if hook.split > 1:
                shape[-1] = translate(tile_type.shape[-1])
            out_type = TensorType(tuple(shape), out_dtype)
        sources = {f"a{i}": values[v] for i, v in enumerate(r.inputs) if v in values}
        try:
            op = _epilogue.hooked(extern, hook, r.kernel, sources=sources, main=main, out_type=out_type, translate=translate)
            for p in op.signature.params:
                for d in p.type.free_dims() if p.type is not None else p.dim.free_dims() if p.dim is not None else ():
                    if d.name in stub_dims and stub_dims[d.name] is not d and d not in mapping.values():
                        return "epilogue-dims", f"region {r.index} has a dimension '{d.name}' that is not the extern's dimension of that name"
            if {d.name for d in _bucket_dims(op)} - {d.name for d in _bucket_dims(extern)}:
                extra = sorted({d.name for d in _bucket_dims(op)} - {d.name for d in _bucket_dims(extern)})
                return "epilogue-dims", f"region {r.index} has bucketed dimensions {extra} that are none of the extern's (a binary per bucket of each)"
            compile_specs(op)
        except TilefishLoweringError as e:
            return "epilogue-template", f"region {r.index} does not lower to a hook: {str(e).strip().splitlines()[0]}"
        except TilefishError as e:
            return "epilogue-hook", f"the hook of '{extern.name}' cannot take region {r.index}: {str(e).strip().splitlines()[0]}"
        h = Hooked(k, r, hook, extern, op, "accumulator" if is_acc else "exact")
        h.result_read = y_needed
        h.views = [m for m in unit_views if not any(needed(c) for c in self.prog[m].outputs)]  # (a split whose chunks something else reads stays)
        h.held_views, h.unit = list(unit_views), set(unit)
        h.main = r.outputs[main] if main is not None else None
        h.tiles = used
        h.sources = sources
        tiles = set(values)
        h.inputs = list(n.inputs)
        h.region_inputs = [(v, given) for v, given in zip(r.inputs, r.call_inputs) if v not in tiles]  # type: ignore[attr-defined]
        outs = list(n.outputs[n_ret : n_ret + n_out])
        if h.main is not None:
            outs[[p.name for p in sig.tensor_outputs].index(hook.out)] = h.main
        side = [self.masks_u8.get(o, o) for i, o in enumerate(r.outputs) if i != main]
        n_callee_out = len(n.callee.signature.tensor_outputs)
        h.outputs = [*n.outputs[:n_ret], *outs, *side, *n.outputs[n_ret + n_callee_out :]]
        return h

    # -- 5. the graph the plan walks -----------------------------------------------------------------------------------------------

    def _graph(self, regions: "list[Region]") -> Graph:
        prog = self.prog
        region_of: "dict[int, Region]" = {k: r for r in regions for k in r.nodes}
        hooked_at: "dict[int, Hooked]" = {}  # a call, its region and the views only they read: one unit, a call of the extern with the region in its hook
        for h in getattr(self, "hooked", ()):
            for k in (h.call, *h.region.nodes, *h.views):
                hooked_at[k] = h
        dropped = set(self.free) | set(getattr(self, "dead", ())) | {k for k, n in enumerate(prog) if n.op == "barrier"}
        # units: a region, or one node that stays (a call, a view, a move)
        unit_of: "dict[int, Any]" = {}
        units: "list[Any]" = []
        for k, n in enumerate(prog):
            if k in dropped:
                continue
            if k in hooked_at:
                h = hooked_at[k]
                if h.call == k:
                    units.append(h)
                unit_of[k] = h
            elif k in region_of:
                r = region_of[k]
                if r.nodes[0] == k:
                    units.append(r)
                unit_of[k] = r
            else:
                units.append(k)
                unit_of[k] = k
        first = {id(u): (u.call if isinstance(u, Hooked) else u.nodes[0] if isinstance(u, Region) else u) for u in units}
        succ: "dict[int, set[int]]" = {id(u): set() for u in units}
        indeg: "dict[int, int]" = {id(u): 0 for u in units}
        by_id = {id(u): u for u in units}
        # a split whose chunks a hook holds, and something else reads as well, stays a view of the stored result after the unit: the hook has those
        # chunks as tiles, not through it, so it is no input of the unit
        held = {m: id(h) for h in getattr(self, "hooked", ()) for m in h.held_views if m not in h.views}
        for k in range(len(prog)):
            if k not in unit_of:
                continue
            for s in self.succ[k]:
                if s not in unit_of:
                    continue
                a, b = id(unit_of[k]), id(unit_of[s])
                if held.get(k) == b:
                    continue
                if a != b and b not in succ[a]:
                    succ[a].add(b)
                    indeg[b] += 1
        import heapq

        ready = [(first[i], i) for i, d in indeg.items() if d == 0]
        heapq.heapify(ready)
        order: "list[Any]" = []
        while ready:
            _, i = heapq.heappop(ready)
            order.append(by_id[i])
            for s in succ[i]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    heapq.heappush(ready, (first[s], s))
        if len(order) != len(units):  # pragma: no cover - every merge was checked for a cycle
            self.fail("the fused regions would make a cycle (a bug of tilefish's fusion; plan with tilefish.fusion(grouping='none'))")
        from ._copyops import contiguous_op

        nodes: "list[Node]" = []
        materialised: "dict[Value, Value]" = {}

        def given_of(r: Region, v: Value, given: Value, copied: "tuple[str, ...]" = ("materialised",)) -> Value:
            if r.reads.get(v) not in copied:
                return given
            c = materialised.get(v)
            if c is None:
                t = v.type
                last = t.shape[-1].folded()
                op = contiguous_op(t.rank, t.dtype, last.value if last.is_static else None)
                c = self.new(TensorType(t.shape, t.dtype))
                node = Node(op.name, [v], {"kind": "extern"}, [c])
                node.callee = op
                nodes.append(node)
                materialised[v] = c
            return c

        for u in order:
            if isinstance(u, Hooked):  # the extern with the region in its epilogue hook (docs/rfcs/0003-epilogue-fusion.md)
                call = prog[u.call]
                # (a dense view at an offset not a multiple of 16 bytes is copied first too: the hook's parameter is compiled for an aligned pointer)
                ins = [*u.inputs, *(given_of(u.region, v, given, ("materialised", "unaligned")) for v, given in u.region_inputs)]  # type: ignore[attr-defined]
                attrs = {a: v for a, v in call.attrs.items() if a not in ("triton", "gluon")}
                attrs.update(u.op.node_attrs())
                node = Node(u.op.name, ins, attrs, u.outputs)
                node.callee, node.loc = u.op, call.loc
                nodes.append(node)
                continue
            if not isinstance(u, Region):
                nodes.append(prog[u])
                continue
            ins = [given_of(u, v, given) for v, given in zip(u.inputs, u.call_inputs)]  # type: ignore[attr-defined]
            outs = [self.masks_u8.get(o, o) for o in u.outputs]
            attrs: "dict[str, Any]" = {"kind": "kernel"}
            if u.call is not u.kernel:
                attrs["unaligned"] = tuple(sorted(u.call.unaligned_params))
            node = Node(u.kernel.name, ins, attrs, outs)
            node.callee = u.call
            nodes.append(node)
        g = self.source
        renamed = getattr(self, "renamed", {})
        outputs = tuple(renamed.get(v, v) for v in g.outputs)
        finals = {k: renamed.get(v, v) for k, v in g.finals.items()}
        return Graph(g.name, g.kind, g.inputs, g.consts, nodes, outputs, g.requirements, finals)


# -- library epilogues (RFC 0003, E2) --------------------------------------------------------------------------------------------------------


def _is_widening(n: Node, to: Any = f32) -> bool:
    """A plain cast to ``to`` that cannot round (``x.to(f32)`` of a bf16 or f16 tensor)."""
    return n.op == "cast" and n.attrs.get("kind") is None and set(_clean(n.attrs)) == {"to", "rounding"} and n.attrs["to"] == to and not n.attrs["rounding"]


def _is_zero(x: Any) -> bool:
    """The scalar ``+0.0`` of f32 (not ``-0.0``: a ReLU gives ``+0``)."""
    import math

    return isinstance(x, Scalar) and x.name is None and x.dtype == f32 and float(x.value) == 0.0 and math.copysign(1.0, float(x.value)) > 0


def match_library_epilogue(
    nodes: "Sequence[Node]", k: int, uses: "Mapping[Value, list[int]]", kept: "set[Value]", producer: "Mapping[Value, int]"
) -> "tuple[list[int], str, int, Any, Any, Value] | str":
    """Whether the cuBLASLt GEMM called by ``nodes[k]`` and the ops after it are one of its epilogues (``epilogues="library"``): ``(chain, epilogue,
    beta, bias, c, z)`` (the nodes the call replaces, the epilogue, ``beta``, the bias and ``c`` values it reads, the value it makes), or why not.

    The pattern, each step optional but one, every value read by the next step alone (not returned, not barriered, not read elsewhere)::

        y = gemm(a, b)                      # cublaslt.gemm (or cublaslt.gemm_bias, then ReLU only) of a bf16 or f16 result T
        h = y.to(f32)
        h = h + c.to(f32)                   # c [M, N] of T, defined before the call: beta * C
        h = h + bias.to(f32)                # bias [N] of T, defined before the call (a cublaslt.gemm): BIAS
        h = tilefish.where(h <= 0.0, 0.0, h)  # torch.relu's bits: NaN stays NaN, -0.0 becomes +0.0
        z = h.to(T)

    That is the library's order (``c`` before the bias, measured: RFC 0003, X1); ``(acc + b) + c`` rounds otherwise in f32 and is not matched."""
    n = nodes[k]
    op = n.callee
    if getattr(op, "algos", None):
        return f"'{op.name}' imports its algorithms (algos=), and cublaslt.matmul imports none yet"
    sig = op.signature
    out_t = sig.returns[0]
    t = out_t.dtype
    if getattr(t, "name", None) not in _HALF or sig.params[0].type.dtype != t:
        return f"'{op.name}' has a result of {out_t.dtype} from {sig.params[0].type.dtype} operands: library epilogues are planned for a 16-bit result of the operands' type"
    y = n.outputs[0]

    def only_use(v: Value) -> "int | None":
        found = uses.get(v, [])
        return found[0] if len(found) == 1 and v not in kept else None

    def defined_before(v: Value) -> bool:
        p = producer.get(v)
        return p is None or p < k

    def dense(v: Value) -> bool:
        return v.type.contiguous and v.type.strides is None and not v.type.devscalar

    j = only_use(y)
    if j is None or not _is_widening(nodes[j]):
        return "the GEMM's result is read by something else than one y.to(f32) (or returned, or barriered)"
    chain = [j]
    h = nodes[j].outputs[0]
    bias = c = None
    with_bias = op.impl == "cublaslt.gemm_bias"
    relu = False
    while True:
        j = only_use(h) if h is not None else None
        nxt = nodes[j] if j is not None else None
        if nxt is not None and nxt.op == "add" and nxt.attrs.get("kind") is None and not relu:
            other = nxt.inputs[1] if nxt.inputs[0] is h else nxt.inputs[0]
            p = producer.get(other) if isinstance(other, Value) else None
            if p is None or not _is_widening(nodes[p]):
                return "an add after the GEMM is not of a widened 16-bit tensor (other.to(f32))"
            src = nodes[p].inputs[0]
            if not isinstance(src, Value) or src.type.dtype != t or not dense(src) or not defined_before(src):
                return f"an add after the GEMM reads {getattr(src, 'ref', src)}, which is not a dense tensor of {t} made before the GEMM"
            if tuple(src.type.shape) == tuple(out_t.shape) and tuple(src.type.shape) == tuple(y.type.shape):
                if c is not None or bias is not None or with_bias:
                    return "the residual is added after the bias: the library adds c before the bias ((acc + c) + bias), so write (y + c) + bias"
                c = src
            elif len(src.type.shape) == 1 and src.type.shape[0] == y.type.shape[-1]:
                if bias is not None or with_bias:
                    return "a second bias after the GEMM"
                bias = src
            else:
                return f"an add after the GEMM reads {src.ref} of shape {list(src.type.shape)}, neither a bias [N] nor a residual [M, N]"
            chain.append(j)
            h = nxt.outputs[0]
            continue
        found = uses.get(h, [])
        if not relu and len(found) == 2 and h not in kept:
            le_k, wh_k = sorted(found)
            le, wh = nodes[le_k], nodes[wh_k]
            if (
                le.op == "le" and le.attrs.get("kind") is None and le.inputs[0] is h and _is_zero(le.inputs[1])
                and only_use(le.outputs[0]) == wh_k
                and wh.op == "where" and wh.attrs.get("kind") is None and wh.inputs[0] is le.outputs[0] and _is_zero(wh.inputs[1]) and wh.inputs[2] is h
            ):  # fmt: skip
                relu = True
                chain += [le_k, wh_k]
                h = wh.outputs[0]
                continue
        break
    j = only_use(h)
    last = nodes[j] if j is not None else None
    if last is None or last.op != "cast" or last.attrs.get("kind") is not None or set(_clean(last.attrs)) != {"to", "rounding"} or last.attrs["to"] != t:
        return f"the arithmetic after the GEMM does not end in one plain .to({t}) of a value read by it alone"
    chain.append(j)
    if bias is None and c is None and not relu:
        return "nothing between y.to(f32) and .to() for an epilogue to compute"
    epilogue = ("relu_" if relu else "") + ("bias" if bias is not None or with_bias else "")
    epilogue = epilogue.rstrip("_") or "none"
    if with_bias:
        bias = n.inputs[2]
    return chain, epilogue, int(c is not None), bias, c, last.outputs[0]


# -- the synthetic kernel of a region ------------------------------------------------------------------------------------------------------


def _dim_key(d: DimExpr) -> "list[Any]":
    atoms = sorted(d.free_dims(), key=lambda x: x.name)
    return [str(d.folded()), [[a.name, a.divisible_by, list(a.buckets) if a.buckets else None] for a in atoms]]


def _type_key(t: TensorType, strided: bool = False) -> "list[Any]":
    return [str(t.dtype), bool(t.devscalar), [_dim_key(d) for d in t.shape], strided]


def _declare(name: str, params: "list[tuple[str, TensorType, Value, bool]]", mask_inputs: "set[Value]", body: "list[Node]", outputs: "list[Value]",
             out_types: "list[TensorType]", settings: Fusion) -> Any:
    """The ``@tilefish.kernel`` of a region's ops, named ``name``: a parameter per input (``Strided[...]`` with a ``StrideOf`` per axis for a view
    read where it is), lowered without contraction and context-free (note 20 of :mod:`tilefish.lower`)."""
    from .decorators import KernelOp, _trace_body, _with_traced_returns
    from .signature import parse_signature
    from .types import StrideOf, Strided

    def replay(**kw: Any) -> Any:
        values = [kw[p] for p, _, _, _ in params]
        ctx = values[0].ctx
        env: "dict[Value, Any]" = {}
        for (pname, _, v, _), val in zip(params, values):
            if v in mask_inputs:  # a mask that came from another region as u8: 0 or 1, read back as != 0 (exact, note 7)
                (env[v],) = ctx.emit("ne", [val, Scalar(0, u8)], {}, [TensorType(v.type.shape, bool_)])
            else:
                env[v] = val
        for position, n in enumerate(body):
            # a const of the graph is its value here (a region's kernel has no such parameter)
            ins = [env[i] if isinstance(i, Value) else Scalar(i.value, i.dtype) if isinstance(i, Scalar) and i.name is not None else i for i in n.inputs]
            outs = ctx.emit(n.op, ins, _clean(n.attrs), [o.type for o in n.outputs])
            ctx.nodes[-1].loc, ctx.nodes[-1].origin = n.loc, position  # where the op is in the graph's body (tilefish.explain); nothing depends on it
            for o, new in zip(n.outputs, outs):
                env[o] = new
        results = []
        for o in outputs:
            val = env[o]
            if isinstance(o.type.dtype, DType) and o.type.dtype.is_bool:
                (val,) = ctx.emit("cast", [val], {"to": u8, "rounding": False}, [TensorType(o.type.shape, u8)])
            results.append(val)
        return results[0] if len(results) == 1 else tuple(results)

    ret: Any = out_types[0] if len(out_types) == 1 else tuple.__class_getitem__(tuple(out_types))
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    sig_params = []
    for pname, ptype, _, view in params:
        sig_params.append(inspect.Parameter(pname, positional, annotation=Strided[ptype] if view else ptype))
        if view:  # its strides, in the kernel's argument order right after it: every axis (a view of a transpose has no axis of stride 1)
            sig_params += [inspect.Parameter(f"{pname}_s{k}", positional, annotation=StrideOf(pname, k)) for k in range(ptype.rank)]
    replay.__signature__ = inspect.Signature(sig_params, return_annotation=ret)  # type: ignore[attr-defined]
    replay.__name__ = replay.__qualname__ = name
    replay.__module__ = "tilefish_generated"
    sig = parse_signature(replay, what="kernel", name=name, check_derivable=True, allow_strided=True)
    graph = _trace_body(replay, sig, name, "kernel")
    op = KernelOp(replay, name, _with_traced_returns(sig, graph, "kernel", name), graph, None, False, settings.math, settings.sass)
    op.context_free = True
    op.lowering = LOWERING_OF_VERSION[settings.version]  # version 2: revision 2 of the lowering (note 21 of tilefish.lower), frozen with it
    return op


def _synthetic(params: Any, mask_inputs: Any, body: Any, outputs: Any, out_types: Any, settings: Fusion, graph: str, trial: bool = False) -> Any:
    """The ``@tilefish.kernel`` of a region, named by its content: the source it lowers to, the types of its parameters (dimensions with their
    constraints) and the fusion settings. The same content anywhere is the same operator."""
    if not params:
        raise TilefishPlanError(f"graph '{graph}': a fused region with no input (an iota alone that a call reads) is not planned yet")
    try:
        draft = _declare("fused", params, mask_inputs, body, outputs, out_types, settings)
        source = draft.lower().source
    except TilefishLoweringError:
        raise
    except TilefishError as e:
        raise TilefishPlanError(f"graph '{graph}': a fused region cannot be made: {e}") from e
    if trial:  # a candidate union of regions: only whether it lowers
        return draft
    content = json.dumps(
        {"fusion": settings.canonical(), "source": source, "params": [_type_key(t, s) for _, t, _, s in params], "outputs": [_type_key(t) for t in out_types]},
        sort_keys=True,
        separators=(",", ":"),
    )
    name = "fused_" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    with _lock:
        found = _kernels.get(name)
    if found is not None:
        return found
    op = _declare(name, params, mask_inputs, body, outputs, out_types, settings)
    with _lock:
        return _kernels.setdefault(name, op)


# -- the report ------------------------------------------------------------------------------------------------------------------------------


def report(gop: Any, verification: "Mapping[int, Any] | None" = None) -> "dict[str, Any]":
    """The fusion report of a graph (``<graph>.fusion.json``): every region, the ops of the traced graph it took, what it reads and writes, the
    roundings it keeps and the rewrites of its mode; informative (no host reads it) and derived (made again from the graph)."""
    f = fused(gop)
    src = f.source.nodes
    regions = []
    for r in f.regions:
        nodes = sorted({f.origin[k] for k in r.nodes})
        low = r.kernel.lower()
        roundings = []
        for k in r.nodes:
            n = f.program[k]
            if n.op in _ROUNDING_OPS or n.is_rounding_point:
                entry: "dict[str, Any]" = {"node": f.origin[k], "op": n.op, "dtype": str(n.outputs[0].type.dtype)}
                if "order" in n.attrs:
                    entry["order"] = str(n.attrs["order"])
                roundings.append(entry)
        item: "dict[str, Any]" = {
            "op": r.kernel.name,
            "template": low.template,
            "nodes": [{"index": i, "op": src[i].op} for i in nodes],
            "inputs": [{"value": v.ref, "type": str(v.type), "read": r.reads.get(v, "contiguous")} for v in r.inputs],
            "outputs": [{"value": o.ref, "dtype": "u8 (a mask, 0 or 1)" if o.type.dtype == bool_ else str(o.type.dtype)} for o in r.outputs],
            "roundings": roundings,
            "rewrites": [w for w in f.rewrites if w["node"] in nodes],
            "exact": True,
        }
        h = f.hooked_region(r)
        if h is not None:  # fused into an extern's epilogue hook (docs/rfcs/0003-epilogue-fusion.md): one step, the extern's kernel with the region
            removed = [w for w in f.rewrites if w["kind"] == "accumulator" and w["node"] == f.origin[h.call]]
            item.update({
                "op": h.op.name, "region": r.kernel.name, "template": h.op.__dict__["_tf_generated"].template, "route": "hook",
                "call": {"op": h.extern.name, "node": f.origin[h.call]}, "value": h.hook.value, "contract": h.contract,
                "stored": h.main.ref if h.main is not None else "the result",
            })  # fmt: skip
            if h.main is None and not h.result_read:
                item["stored"] = "the result, which nothing reads (no output of the region has its shape and type)"
            if removed:
                item["rounding_points_removed"] = [
                    {"value": w["value"], "dtype": w["dtype"], "widenings": w["widenings"], "reason": "R_acc: widened by the epilogue"} for w in removed
                ]
                item["exact"] = False  # (exact against U(R_acc(P)), the declared program; not against U(P))
        refused = [x for x in f.refused if x["region"] == r.index]
        if refused:
            item["epilogue_refused"] = [{"call": {"op": _extern_of(f.program[x["call"]]).name, "node": f.origin[x["call"]]}, "reason": x["reason"], "detail": x["detail"]} for x in refused]
        if verification is not None and r.index in verification:
            item["verified"] = verification[r.index]
        regions.append(item)
    doc: "dict[str, Any]" = {"schema": REPORT_SCHEMA, "graph": f.source.name, "fusion": f.settings.described(), "regions": regions}
    epilogues = [w for w in f.rewrites if w["kind"] == "library-epilogue"]
    if not epilogues:  # (no op is in a "library" mode: what the mode would do)
        epilogues = library_epilogue_candidates(f.source, f.settings)
    if epilogues:  # (only where a GEMM's result is widened by the graph, or a "library" mode looked: every other report is what it was)
        doc["library_epilogues"] = epilogues
    hooks = _hook_externs(f)
    if hooks:  # (only a graph that calls an extern with an epilogue hook has it: every other report is as it was)
        doc["hooks"] = hooks
    return doc


def _extern_of(n: Node) -> Any:
    """The extern a call node calls (the one whose accumulator form it calls, for ``R_acc``'s call)."""
    acc = getattr(n.callee, "acc_of", None)
    return acc[0] if acc is not None else n.callee


def _hook_externs(f: Fused) -> "dict[str, list[dict[str, Any]]]":
    """The externs with epilogue hooks the graph calls: name -> their hooks (what the advisor reads)."""
    from ._epilogue import hooks_of_op

    out: "dict[str, list[dict[str, Any]]]" = {}
    for n in f.program:
        if n.attrs.get("kind") == "extern" and n.callee is not None:
            ext = _extern_of(n)
            if hooks_of_op(ext):
                out[ext.name] = [h.canonical() for h in hooks_of_op(ext)]
    return out


_ACC_LEAVES_LIBRARY = (
    "epilogues='accumulator' leaves a library call as it is (R_acc of a cuBLASLt GEMM is not planned: the GEMM keeps its rounding); declare "
    "epilogues='library' (the graph's or a tilefish.fused scope's) for its epilogue"
)


def _accumulator_note(nodes: "Sequence[Node]", k: int, uses: "Mapping[Value, list[int]]", settings: "Fusion | None") -> "dict[str, Any]":
    """``{"mode": "accumulator", "note": ...}`` when the result of the cuBLASLt GEMM ``nodes[k]`` is widened in an ``"accumulator"`` mode (which leaves
    a library call as it is), else ``{}``: what a ``library_epilogues`` entry says of it on either path of the report."""
    if settings is None:
        return {}
    widenings = [j for j in uses.get(nodes[k].outputs[0], []) if _is_widening(nodes[j])]
    return {"mode": "accumulator", "note": _ACC_LEAVES_LIBRARY} if any(epilogue_mode(nodes[j], settings) == "accumulator" for j in widenings) else {}


def library_epilogue_candidates(graph: Graph, settings: "Fusion | None" = None) -> "list[dict[str, Any]]":
    """What ``epilogues="library"`` would do with the cuBLASLt GEMMs of a graph planned without it (the fusion report's ``library_epilogues``, which
    the advisor reads): each call whose result the graph widens to f32, with the epilogue it would become (``"applied": False``) or why it would not
    (the order of the adds, a value read elsewhere, ...). A GEMM whose result is not widened is not listed. One widened in an ``"accumulator"`` mode
    says that the mode leaves it as it is (``"mode"``, ``"note"``)."""
    nodes = list(graph.nodes)
    uses: "dict[Value, list[int]]" = {}
    for k, n in enumerate(nodes):
        for i in n.inputs:
            if isinstance(i, Value):
                uses.setdefault(i, []).append(k)
    kept = set(graph.outputs) | set(graph.finals.values())
    kept |= {i for n in nodes if n.op == "barrier" and n.attrs.get("kind") is None for i in n.inputs if isinstance(i, Value)}
    producer = {o: k for k, n in enumerate(nodes) for o in n.outputs}
    out: "list[dict[str, Any]]" = []
    for k, n in enumerate(nodes):
        if n.attrs.get("kind") != "library" or getattr(n.callee, "impl", None) not in ("cublaslt.gemm", "cublaslt.gemm_bias"):
            continue
        widenings = [j for j in uses.get(n.outputs[0], []) if _is_widening(nodes[j])]
        if not widenings:
            continue
        mode = _accumulator_note(nodes, k, uses, settings)
        found = match_library_epilogue(nodes, k, uses, kept, producer)
        if isinstance(found, str):
            out.append({"kind": "library-epilogue", "node": k, "call": n.callee.name, "matched": False, "applied": False, "reason": found, **mode})
            continue
        chain, epilogue, beta, _, _, _ = found
        out.append({
            "kind": "library-epilogue", "node": k, "call": n.callee.name, "matched": True, "applied": False, "epilogue": epilogue, "beta": beta,
            "nodes": sorted(chain), "rounding_removed": {"value": n.outputs[0].ref, "dtype": str(n.outputs[0].type.dtype)},
            "contract": "measured-exact", **mode,
        })  # fmt: skip
    return out


# -- why the regions are where they are (tilefish.advise) ------------------------------------------------------------------------------------

# reason -> (what to do, the phase of docs/rfcs/0001-plan-fusion.md that does it)
BOUNDARY_FIXES: "dict[str, tuple[str, str]]" = {
    "barrier": ("remove tilefish.barrier: a tap of the unfused plan (grouping='none') shows the same bytes", "1"),
    "view": (
        "a view between two ops ends a region (rule 6): compute on one shape (a dense reshape inside a region, phase 2 work, is not planned yet; "
        "a permute or a select would still end one)",
        "2",
    ),
    "reduction-unordered": (
        "add order= to the reduction (D2): exact and layout-independent, and fusion version 2 (the default) fuses it with the ops around it; "
        "without an order it runs alone in every version",
        "2",
    ),
    "reduction": (
        "the graph declares tilefish.fusion(version=1), where every reduction is a region of its own: version 2 (the default) fuses ordered sums "
        "and means, max, argmax and argmin into row regions",
        "2",
    ),
    "gather": ("the graph declares tilefish.fusion(version=1), where gather_rows is a region of its own: version 2 (the default) fuses it", "2"),
    "cycle": ("move the call out from between them (or accept two kernels)", "1"),
    "limit": (f"a region has at most {MAX_REGION_NODES} ops and {MAX_REGION_TENSORS} tensors: a split is a speed choice only", "1"),
    "template": (
        "the templates cannot compute both in one kernel: another iteration space (rows of another length, a reduction over another axis, rows "
        "of rank 3) or, in fusion version 1, a row region and an elementwise one (version 2 makes row regions)",
        "2",
    ),
    "grouping": ("the greedy grouping of this version did not make this union (a later fusion version may)", "1"),
    "unfused": ("the graph declares tilefish.fusion(grouping='none'), one kernel per op: drop it (the default groups them)", "1"),
    "materialised": (
        "a view that is not dense, copied by tilefish.contiguous first (a copy or a store of a view is a launch of it; fusion version 1's row regions "
        "read no view where it is): version 2 (the default) reads views of rank 1 and 2 where they are in the row template",
        "2",
    ),
    "move": (
        "a copy, a store or a memset step makes the value (zeros are read from memory, never folded into a region: H2): return or store a value where "
        "it is made",
        "1",
    ),
    "epilogue-library": (
        "a pinned cuBLASLt GEMM's or cuDNN convolution's epilogue (RFC 0003): cuBLASLt's bias, ReLU and beta * C are an epilogue of cublaslt.matmul "
        "(E2) under tilefish.fusion(epilogues='library') when written as its pattern (docs/advise.md); another epilogue of a cuBLASLt GEMM (GELU, ...) "
        "and a generated GEMM or convolution with hooks are E3, a cuDNN fused graph E6, future work. Today: call an extern with an epilogue hook "
        "instead (E1), whose fusion is exact (contract (i))",
        "future",
    ),
    "epilogue-library-cublaslt": (
        "declare tilefish.fusion(epilogues='library'): the region is a cuBLASLt epilogue of the GEMM (cublaslt.matmul, RFC 0003 E2), one step; the "
        "bytes are the library's under the pin (contract (iii)), verified equal to the region's arithmetic on the f32 accumulator at every pin "
        "(measured-exact against contract (ii)): one rounding fewer than now, so they change",
        "E2",
    ),
    "epilogue-extern": (
        "declare an epilogue hook on the extern (tilefish.hook, RFC 0003 E1): value='out' fuses the region exactly (contract (i)), value='acc' on the "
        "accumulator under tilefish.fusion(epilogues='accumulator') (contract (ii))",
        "4",
    ),
    "epilogue-acc-only": (
        "the extern hooks its accumulator only: declare tilefish.fusion(epilogues='accumulator') or a tilefish.fused(epilogues='accumulator') scope "
        "(contract (ii): one rounding fewer, other bytes than the extern followed by the region), or give the kernel a value='out' hook (contract (i))",
        "4",
    ),
    "epilogue-taken": ("a hook takes one region of a call's result: the other reads the stored result (write both in one region to fuse both)", "4"),
    "epilogue-view": ("read the extern's result itself (or the chunks of a split hook's tiles), not another view of it, in the region the hook takes", "4"),
    "epilogue-acc-shared": ("read the extern's f32 result (R_acc) in one region only: a hook stores one result", "4"),
    "epilogue-cycle": ("move the call out from between the extern and the region (it is on a path from the extern's result into the region)", "4"),
    "epilogue-inout": ("read what the extern updates in place (InOut) outside the region its hook takes: its kernel cannot read what it writes", "4"),
    "epilogue-row": (
        "a region that reduces rows fuses into a hook only where a tile of the extern holds whole rows (block_n >= the row, a Triton kernel; RFC 0003 "
        "Q10): a wider tile, or the split over tiles of E4",
        "4",
    ),
    "epilogue-template": ("the region does not lower to a hook (a gather, a scatter, a static slice, another iteration space than the result's)", "4"),
    "epilogue-hook": ("the extern's hook cannot take the region (its message says why)", "4"),
    "epilogue-dims": ("the region's sizes are not the extern's (a dimension of its own, bucketed): give it the extern's", "4"),
    "epilogue-kernel": ("write the @tilefish.kernel's body in the graph body so phase 1 fuses it (inlining kernel calls: RFC 0001, 8.2)", "1"),
    "prologue-library": ("GEMM / convolution prologue: future work (RFC 0001, section 11.3)", "future"),
    "prologue-extern": ("a prologue hook of the extern: future work (RFC 0001, section 11.3)", "future"),
    "prologue-kernel": ("write the @tilefish.kernel's body in the graph body so phase 1 fuses it (inlining kernel calls: RFC 0001, 8.2)", "1"),
}


def reduction_reason(op: str, ordered: bool, version: int = 1) -> "str | None":
    """Why an op of a graph body is a region of its own in fusion ``version``: ``"reduction-unordered"`` (a ``sum`` or ``mean`` without
    ``order=``, D2, in every version), and in version 1 ``"reduction"`` (an ordered one, ``max``, ``argmax``, ``argmin``) and ``"gather"``
    (``gather_rows``); ``None`` for any other op (in version 2 those join row regions)."""
    if not is_solo(op, {"order": True} if ordered else {}, version):
        return None
    if op == "gather_rows":
        return "gather"
    return "reduction-unordered" if op in ("sum", "mean") and not ordered else "reduction"


def _node_ref(f: Fused, k: int, region: "Region | None" = None) -> "dict[str, Any]":
    n = f.program[k]
    out: "dict[str, Any]" = {"node": f.origin[k], "op": n.op, "loc": f"{n.loc[0]}:{n.loc[1]}" if n.loc else None}
    if region is not None:
        out["region"] = region.index
        out["kernel"] = region.kernel.name
    elif n.attrs.get("kind"):
        out["call"] = n.attrs["kind"]
    return out


def boundaries(gop: Any) -> "list[dict[str, Any]]":
    """Why the arithmetic of a graph body is in the regions it is in (what ``tilefish.advise`` reports): one entry per value that crosses a boundary of a
    region, from a region (or a call) to another region (or a call), and per view a region copies first. Each entry is ``{"reason", "from", "to", "value",
    "detail", "fix", "phase"}``; ``from`` and ``to`` name the op of the traced graph (``node``, ``op``, ``loc``, and ``region`` and ``kernel`` for a region, or
    ``call`` for a call). Reasons: ``barrier``, ``view``, ``move``, ``reduction-unordered`` (D2), ``reduction``, ``gather``, ``cycle``, ``limit``,
    ``template``, ``grouping``, ``materialised``, ``epilogue-<library|extern|kernel>`` (a call's result read by a region) and ``prologue-...`` (a region's
    value read by a call). Derived from the graph like the fusion report; no GPU."""
    f = fused(gop)
    if not f.regions:
        return []
    p = _Pass(f.source, f.settings)
    p.prepare(f.program, f.origin, f.renamed)
    p.select()
    p.member_of = {k: r.nodes[0] for r in f.regions for k in r.nodes}
    p.groups = {r.nodes[0]: list(r.nodes) for r in f.regions}
    region_at = {r.nodes[0]: r for r in f.regions}
    prog = f.program
    out: "list[dict[str, Any]]" = []
    seen: "set[tuple[Any, ...]]" = set()

    def source(v: Value) -> "tuple[int | None, list[int]]":
        """The node that makes the storage ``v`` reads (through views), and the views on the way."""
        via: "list[int]" = []
        while True:
            k = p.producer.get(v)
            if k is not None and prog[k].attrs.get("kind") is None and prog[k].op in _VIEW_OPS and isinstance(prog[k].inputs[0], Value):
                via.append(k)
                v = prog[k].inputs[0]  # type: ignore[assignment]
                continue
            return k, via

    def add(reason: str, a: "dict[str, Any]", b: "dict[str, Any]", value: Value, detail: str) -> None:
        key = (reason, a.get("region", a["node"]), b.get("region", b["node"]), value.ref)
        if key in seen:
            return
        seen.add(key)
        fix, phase = BOUNDARY_FIXES[reason]
        out.append({"reason": reason, "from": a, "to": b, "value": value.ref, "detail": detail, "fix": fix, "phase": phase})

    version = f.settings.version

    def solo_node(k: int) -> "int | None":
        """The op of ``k``'s region that makes it a region of its own (``k`` itself, or the fma sum whose product ``k`` is), or ``None``."""
        root = p.member_of.get(k)
        for m in p.groups.get(root, [k]) if root is not None else [k]:
            if reduction_reason(prog[m].op, "order" in prog[m].attrs, version) is not None:
                return m
        return None

    def solo(k: int) -> "str | None":
        m = solo_node(k)
        return None if m is None else reduction_reason(prog[m].op, "order" in prog[m].attrs, version)

    for r in f.regions:
        root = r.nodes[0]
        for k in r.nodes:
            for i in prog[k].inputs:
                if not isinstance(i, Value):
                    continue
                pk, via = source(i)
                if pk is None or pk in r.nodes or pk in p.free:
                    continue
                pn = prog[pk]
                to = _node_ref(f, k, r)
                kind = pn.attrs.get("kind")
                if kind in ("kernel", "extern", "library"):
                    h = f.hooked_region(r)
                    if h is not None and h.call == pk:  # fused into the call's epilogue hook (docs/rfcs/0003-epilogue-fusion.md): no boundary
                        continue
                    refusal = next((x for x in f.refused if x["call"] == pk and x["region"] == r.index), None)
                    if refusal is not None:  # the call's extern has a hook, which does not take this region: why
                        add(refusal["reason"], _node_ref(f, pk), to, i, refusal["detail"])
                        continue
                    from ._epilogue import hooks_of_op

                    if kind == "extern" and hooks_of_op(_extern_of(pn)) and f.settings.grouping == "none":
                        add("unfused", _node_ref(f, pk), to, i, "the graph declares grouping='none': the extern is called as it is, every op a kernel of its own")
                        continue
                    what = "the extern declares no epilogue hook" if kind == "extern" else "no fusion across a call"
                    add(f"epilogue-{kind}", _node_ref(f, pk), to, i, f"the result of the call to '{pn.op}' is read by region {r.index}: {what}")
                    continue
                if kind is None and pn.op in _MOVE_OPS:
                    add("move", _node_ref(f, pk), to, i, f"'{pn.op}' (node #{f.origin[pk]}) is between them")
                    continue
                other = p.member_of.get(pk)
                if other is None or other == root:
                    continue
                frm = _node_ref(f, pk, region_at[other])
                if i in p.barriered or any(v in p.barriered for v in (prog[v].inputs[0] for v in via)):
                    add("barrier", frm, to, i, f"{i.ref} passes tilefish.barrier")
                elif via:
                    add("view", frm, to, i, f"the {', '.join(prog[v].op for v in via)} of {i.ref} is between them")
                elif solo(pk) or solo(k):
                    which = solo_node(pk) if solo(pk) else solo_node(k)
                    assert which is not None
                    reason = solo(which) or "reduction"
                    n = prog[which]
                    order = f" with order={n.attrs['order']}" if "order" in n.attrs else (" without order=" if reason == "reduction-unordered" else "")
                    add(reason, frm, to, i, f"'{n.op}'{order} (node #{f.origin[which]}) is a region of its own in fusion version {version}")
                elif f.settings.grouping == "none":
                    add("unfused", frm, to, i, "the graph declares grouping='none': every op is a kernel of its own")
                else:
                    problem = p.merge_problem(other, root)
                    if problem is None:
                        add("grouping", frm, to, i, f"the union of the two regions is legal, but the greedy merge of version {version} did not make it")
                    else:
                        add(problem[0], frm, to, i, problem[1])
        for v, how in r.reads.items():
            if how == "materialised":
                pk, _ = source(v)
                a = _node_ref(f, pk) if pk is not None else {"node": None, "op": "parameter", "loc": None}
                add("materialised", a, _node_ref(f, r.nodes[0], r), v, f"{v.ref} ({v.type}) is not dense, and region {r.index} is lowered by the row template")
    for k, n in enumerate(prog):  # a region's value read by a call
        kind = n.attrs.get("kind")
        if kind not in ("kernel", "extern", "library"):
            continue
        for i in n.inputs:
            if not isinstance(i, Value):
                continue
            pk, _ = source(i)
            other = p.member_of.get(pk) if pk is not None else None
            if other is None:
                continue
            add(f"prologue-{kind}", _node_ref(f, pk, region_at[other]), _node_ref(f, k), i, f"region {region_at[other].index}'s value is read by the call to '{n.op}': no fusion across a call")
    return out


# -- verification: each region against its ops one by one ------------------------------------------------------------------------------------


def _unfused_region(f: Fused, r: Region) -> "list[tuple[Any, list[Value], list[Value], dict[Value, str]]]":
    """The ops of a region as kernels of their own (as ``grouping="none"`` makes them): ``(kernel, inputs, outputs, how each input is read)`` in
    order (a view one of them copies first, ``"materialised"``, is copied by ``tilefish.contiguous`` in the unfused plan)."""
    p = _Pass(f.source, f.settings)
    p.prepare(f.program, f.origin, f.renamed)
    p.free = set(r.free)
    return [(one.kernel, list(one.inputs), list(one.outputs), dict(one.reads)) for one in p.regions(p.partition(list(r.nodes), "none"))]


def _sample_operand(torch: Any, t: TensorType, sizes: "tuple[int, ...]", seed: int, device: Any) -> Any:
    """An operand of an extern (a GEMM's, a convolution's) for the check of an epilogue hook: normal values with the special values (both zeros,
    both infinities, NaNs, the largest finite values, a subnormal) at its first elements, so that the result has rows and columns of special values
    and finite values everywhere else. An integer or device-scalar operand is zero (an index or a count the kernel trusts)."""
    from ._reference import torch_dtype

    dt = torch_dtype(torch, t.dtype)
    n = 1
    for s in sizes:
        n *= s
    if t.dtype.name not in ("f32", "bf16", "f16"):
        return torch.zeros(sizes, dtype=dt, device=device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = (torch.randn(n, generator=gen) * 0.5).to(dt)
    finfo = torch.finfo(dt)
    special = [0.0, -0.0, float("inf"), float("-inf"), float("nan"), finfo.max, -finfo.max, finfo.tiny / 4]
    m = min(n, len(special))
    x[:m] = torch.tensor(special[:m], dtype=torch.float32).to(dt)
    return x.reshape(sizes).to(device)


def _verify_hooked(f: Fused, h: Hooked, combos: "Sequence[Mapping[str, int]]", seeds: "Sequence[int]", device: Any) -> "dict[str, Any]":
    """An epilogue hook against its reference, byte for byte, at every bucket of ``combos`` and every seed: the fused op (the extern with the region in
    its hook) against the extern (contract (i), ``U(P)``) or its accumulator form (contract (ii), ``U(R_acc(P))``) followed by the region's ops one by
    one, every intermediate stored. Raises :class:`TilefishFusionError` at the first difference."""
    from . import _epilogue, runtime
    from ._launch import eval_dim

    torch = runtime._torch()
    call = f.program[h.call]
    r = h.region
    held = set(h.tiles)
    steps = _unfused_region(f, r)
    reference = _epilogue.acc_form(h.extern, h.hook) if h.contract == "accumulator" else h.extern
    sig = h.extern.signature
    n_ret, outs = len(sig.returns), [p.name for p in sig.tensor_outputs]
    k_out = outs.index(h.hook.out)
    region_in = [v for v in r.inputs if v not in held]
    checked = 0
    for combo in combos:
        values = [*call.inputs, *call.outputs, *r.inputs, *r.outputs]
        dims = {d: combo[d.name] for v in values for s in v.type.shape for d in s.free_dims() if d.name in combo}
        for seed in seeds:
            env: "dict[Value, Any]" = {}
            for k, v in enumerate(call.inputs):
                sizes = tuple(int(eval_dim(s, dims)) for s in v.type.shape)  # type: ignore[arg-type]
                env[v] = _sample_operand(torch, v.type, sizes, seed * 7919 + k * 31 + sum(sizes), device)
            for k, v in enumerate(region_in):
                t = v.type
                sizes = tuple(int(eval_dim(s, dims)) for s in t.shape)  # type: ignore[arg-type]
                if isinstance(t.dtype, DType) and t.dtype.is_bool:
                    env[v] = torch.randint(0, 2, sizes, generator=torch.Generator(device="cpu").manual_seed(seed * 31 + k), dtype=torch.int64).to(torch.uint8).to(device)
                elif t.devscalar:
                    env[v] = _sample(torch, TensorType((), t.dtype), (1,), seed * 31 + k, device).reshape(())
                elif r.reads.get(v) == "strided":
                    strides = tuple(int(eval_dim(s, dims)) for s in t.layout_strides())  # type: ignore[arg-type]
                    offset = int(eval_dim(t.offset, dims))  # type: ignore[arg-type]
                    extent = offset + 1 + sum((n - 1) * s for n, s in zip(sizes, strides))
                    env[v] = _sample(torch, TensorType((), t.dtype), (extent,), seed * 1009 + k * 31 + extent, device).as_strided(sizes, strides, offset)
                else:
                    env[v] = _sample(torch, t, sizes, seed * 1009 + k * 31 + sum(sizes), device)
            extra = [p.name for p in h.op.signature.tensor_inputs][len(call.inputs) :]  # (keyword-only parameters of the fused op)
            # (each run on copies of the extern's operands: an InOut one is updated in place)
            got = runtime.run(h.op, *[env[v].clone() for v in call.inputs], **{nm: env[v] for nm, v in zip(extra, region_in)})
            got = list(got) if isinstance(got, (tuple, list)) else [got]
            # the reference: the extern (or its accumulator form), every result stored, then the region's ops one kernel each
            res = runtime.run(reference, *[env[v].clone() for v in call.inputs])
            res = list(res) if isinstance(res, (tuple, list)) else [res]
            denv = dict(env)
            for o, val in zip(call.outputs[n_ret:], res):
                denv[o] = val
            results_of_call = set(call.outputs)
            for m in [m for m, s in enumerate(f.program) if s.op == "split" and s.attrs.get("kind") is None and s.inputs and s.inputs[0] in results_of_call]:
                s = f.program[m]  # the chunks of a split hook's tiles: views of the stored result
                parts = torch.split(denv[s.inputs[0]], [int(eval_dim(z, dims)) for z in s.attrs["sizes"]], dim=s.attrs["axis"])
                for o, val in zip(s.outputs, parts):
                    denv[o] = val
            pending = list(steps)
            while pending:
                step = next((s for s in pending if all(v in denv for v in s[1])), pending[0])
                pending.remove(step)
                kern, ins, outs_, how = step
                out = runtime.run(kern, *[denv[v].contiguous() if how.get(v) == "materialised" else denv[v] for v in ins])
                out = list(out) if isinstance(out, tuple) else [out]
                for o, val in zip(outs_, out):
                    denv[o] = val
            want = [res[i] for i in range(len(outs))]
            if h.main is not None:
                want[k_out] = denv[h.main]
            want += [denv[o] for o in r.outputs if o is not h.main]
            where = f"the epilogue of '{h.extern.name}' ({h.op.name}: region {r.kernel.name}, graph '{f.source.name}', nodes {sorted({f.origin[k] for k in r.nodes})})"
            for i, (a, b) in enumerate(zip(got, want)):
                if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape) or not _same_bytes(torch, a, b):
                    detail = f"{_count_diff(torch, a, b)} of {a.numel()} elements differ" if a.shape == b.shape and a.dtype == b.dtype else f"{tuple(a.shape)} {a.dtype} against {tuple(b.shape)} {b.dtype}"
                    against = "U(R_acc(P)), the accumulator form of the extern" if h.contract == "accumulator" else "U(P), the extern"
                    raise TilefishFusionError(
                        f"{where} at {dict(combo) or 'its only size'}, seed {seed}: output {i} of the fused op is not the bytes of {against} followed by the "
                        f"region's ops one by one ({detail}). This is a bug of tilefish, of the toolchain or of the extern's hook (it must apply the hook to the "
                        "tile it stores); export with verify=False to write the plan anyway"
                    )
            checked += 1
    return {"runs": checked, "buckets": len(combos), "seeds": len(seeds), "against": "U(R_acc(P))" if h.contract == "accumulator" else "U(P)"}


def _sample(torch: Any, t: TensorType, sizes: "tuple[int, ...]", seed: int, device: Any) -> Any:
    """Seeded inputs that reach every corner: values of every magnitude, random bit patterns (NaNs with payloads, infinities, subnormals) and
    the special values themselves at the start."""
    from ._reference import torch_dtype

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    n = 1
    for s in sizes:
        n *= s
    name = t.dtype.name
    dt = torch_dtype(torch, t.dtype)
    if name in ("f32", "bf16", "f16"):
        x = torch.randn(n, generator=gen) * torch.pow(2.0, torch.randint(-24, 25, (n,), generator=gen).float())
        x = x.to(dt)
        bits = {"f32": torch.int32, "bf16": torch.int16, "f16": torch.int16}[name]
        width = 32 if name == "f32" else 16
        rnd = torch.randint(-(2 ** (width - 1)), 2 ** (width - 1), (n,), generator=gen, dtype=torch.int64).to(bits).view(dt)
        pick = torch.rand(n, generator=gen) < 0.125
        x = torch.where(pick, rnd, x)
        if name == "f32":
            special = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFC12345, 0x00000001, 0x007FFFFF, 0x7F7FFFFF, 0x00800000]
            sp = torch.tensor([s - (1 << 32) if s >= 1 << 31 else s for s in special], dtype=torch.int32).view(torch.float32)
        else:
            special = [0, 0x8000, 0x7F80 if name == "bf16" else 0x7C00, 0xFF80 if name == "bf16" else 0xFC00, 0x7FC0 if name == "bf16" else 0x7E00,
                       0x7F81 if name == "bf16" else 0x7C01, 0xFFC1 if name == "bf16" else 0xFE01, 0x0001, 0x007F if name == "bf16" else 0x03FF]
            sp = torch.tensor([s - (1 << 16) if s >= 1 << 15 else s for s in special], dtype=torch.int16).view(dt)
        m = min(n, sp.numel())
        x[:m] = sp[:m]
    elif name in ("e4m3", "e5m2"):
        x = torch.randint(0, 256, (n,), generator=gen, dtype=torch.int64).to(torch.uint8).view(dt)
    elif name == "u8":
        x = torch.randint(0, 256, (n,), generator=gen, dtype=torch.int64).to(torch.uint8)
    else:
        x = torch.randint(-(1 << 20), 1 << 20, (n,), generator=gen, dtype=torch.int64).to(dt)
    return x.reshape(sizes).to(device)


def verify(gop: Any, combos: "Sequence[Mapping[str, int]]", *, seeds: "Sequence[int]" = (0, 1), device: Any = None) -> "dict[int, Any]":
    """Run every region of more than one op of ``gop`` against its ops one by one (``jit``, on ``device``: torch's current one by default), on
    seeded inputs, at every bucket combination of ``combos``, and compare the bytes; raises :class:`TilefishFusionError` at the first difference.
    Returns, per region index, what was checked."""
    from . import runtime
    from ._launch import eval_dim

    torch = runtime._torch()
    f = fused(gop)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    results: "dict[int, Any]" = {}
    for r in f.fused_regions:
        steps = _unfused_region(f, r)
        checked = 0
        for combo in combos:
            dims = {d: combo[d.name] for v in r.inputs for s in v.type.shape for d in s.free_dims() if d.name in combo}
            for seed in seeds:
                env: "dict[Value, Any]" = {}
                for k, v in enumerate(r.inputs):
                    sizes = tuple(int(eval_dim(s, dims)) for s in v.type.shape)  # type: ignore[arg-type]
                    t = v.type
                    if isinstance(t.dtype, DType) and t.dtype.is_bool:
                        env[v] = torch.randint(0, 2, sizes, generator=torch.Generator(device="cpu").manual_seed(seed * 31 + k), dtype=torch.int64).to(torch.uint8).to(device)
                    elif t.devscalar:
                        env[v] = _sample(torch, TensorType((), t.dtype), (1,), seed * 31 + k, device).reshape(())
                    elif r.reads.get(v) == "strided":  # a view of the graph's layout at this bucket, over a storage of its extent
                        strides = tuple(int(eval_dim(s, dims)) for s in t.layout_strides())  # type: ignore[arg-type]
                        offset = int(eval_dim(t.offset, dims))  # type: ignore[arg-type]
                        extent = offset + 1 + sum((n - 1) * s for n, s in zip(sizes, strides))
                        base = _sample(torch, TensorType((), t.dtype), (extent,), seed * 1009 + k * 31 + extent, device)
                        env[v] = base.as_strided(sizes, strides, offset)
                    else:
                        env[v] = _sample(torch, t, sizes, seed * 1009 + k * 31 + sum(sizes), device)
                got = runtime.run(r.kernel, *[env[v] for v in r.inputs])
                got = list(got) if isinstance(got, tuple) else [got]
                # the same ops one kernel each, every intermediate stored (a mask as u8)
                denv = dict(env)
                pending = list(steps)
                while pending:  # in an order their values allow (a region joined to its consumer comes where its inputs are)
                    step = next((s for s in pending if all(v in denv for v in s[1])), pending[0])
                    pending.remove(step)
                    kern, ins, outs, how = step
                    res = runtime.run(kern, *[denv[v].contiguous() if how.get(v) == "materialised" else denv[v] for v in ins])
                    res = list(res) if isinstance(res, tuple) else [res]
                    for o, val in zip(outs, res):
                        denv[o] = val
                for o, a in zip(r.outputs, got):
                    b = denv[o]
                    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape) or not _same_bytes(torch, a, b):
                        where = f"region {r.kernel.name} (graph '{f.source.name}', nodes {sorted({f.origin[k] for k in r.nodes})}), output {o.ref}"
                        raise TilefishFusionError(
                            f"{where} at {dict(combo) or 'its only size'}, seed {seed}: the fused kernel's bytes are not those of its ops one by one "
                            f"({_count_diff(torch, a, b)} of {a.numel()} elements differ). This is a bug of tilefish or of the toolchain; export with "
                            "verify=False to write the plan anyway"
                        )
                checked += 1
        results[r.index] = {"runs": checked, "buckets": len(combos), "seeds": len(seeds)}
    for h in f.hooked:  # a region in an extern's epilogue hook: the pair against the extern and the region's ops (docs/rfcs/0003-epilogue-fusion.md)
        results[h.region.index] = _verify_hooked(f, h, combos, seeds, device)
    return results


def _same_bytes(torch: Any, a: Any, b: Any) -> bool:
    if a.numel() == 0:
        return True
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def _count_diff(torch: Any, a: Any, b: Any) -> int:
    width = a.element_size()
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[width]
    return int((a.contiguous().view(view) != b.contiguous().view(view)).sum().item())


def verify_for_export(gop: Any, combos: "Sequence[Mapping[str, int]]", cap: int, verify_flag: "bool | None") -> "dict[int, Any] | None":
    """What ``export`` does: verify the fused regions of ``gop`` when the graph or the call asks for it and a CUDA device of architecture ``cap`` is
    here; returns the results (``None``: nothing to verify, or not asked), or a ``{"skipped": why}`` per region with a warning when no such device is."""
    if not has_arithmetic(gop):
        return None
    want = settings_of(gop).verify if verify_flag is None else verify_flag
    f = fused(gop)
    if not want or not (f.fused_regions or f.hooked):
        return None
    from . import runtime

    why = None
    try:
        torch = runtime._torch()
        if not torch.cuda.is_available():
            why = "no CUDA device"
        else:
            have = runtime._capability(torch, torch.cuda.current_device())
            if have != cap:
                why = f"the device is sm_{have}, the export is for sm_{cap}"
    except TilefishRuntimeError as e:
        why = str(e)
    if why is not None:
        warnings.warn(
            f"graph '{gop.name}': the fused regions were not verified against their unfused composition ({why}); run tilefish.verify_fusion on a "
            "device of the export's architecture, or export with verify=False to say so",
            TilefishFusionWarning,
            stacklevel=3,
        )
        return {r.index: {"skipped": why} for r in [*f.fused_regions, *(h.region for h in f.hooked)]}
    return verify(gop, combos)


__all__ = ["Fusion", "TilefishFusionError", "TilefishFusionWarning", "fusion", "fused", "planned_graph", "report", "verify"]
