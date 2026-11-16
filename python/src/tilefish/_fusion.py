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
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .dims import DimExpr
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

    def canonical(self) -> "dict[str, Any]":
        """What a region's kernel depends on (``verify`` is a check, not a choice of numerics, and is left out; ``sass`` only when given, so
        every region of a graph without it is named as it was)."""
        out = {"version": self.version, "grouping": self.grouping, "math": self.math, "intermediates": self.intermediates, "fp_contract": self.fp_contract}
        if self.sass is not None:
            out["sass"] = self.sass.canonical()
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
) -> Fusion:
    """``@tilefish.graph(fusion=tilefish.fusion(...))``: how the arithmetic written in the graph body is planned. Every choice is exact: the plan
    writes the bytes of its unfused composition (each op a kernel of its own; ``grouping="none"`` is that plan).

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
    """
    if version not in VERSIONS or isinstance(version, bool):
        raise TilefishPlanError(f"fusion: version must be one of {list(VERSIONS)}, got {version!r}")
    for key, value, allowed in (
        ("grouping", grouping, GROUPINGS),
        ("math", math, MATHS),
        ("intermediates", intermediates, INTERMEDIATES),
        ("fp_contract", fp_contract, FP_CONTRACTS),
    ):
        if value not in allowed:
            raise TilefishPlanError(f"fusion: {key} must be one of {list(allowed)}, got {value!r}")
    if not isinstance(verify, bool):
        raise TilefishPlanError(f"fusion: verify must be True or False, got {verify!r}")
    from ._sass import checks_or_none

    return Fusion(version, grouping, math, intermediates, fp_contract, verify, checks_or_none(sass, "fusion"))


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
class Fused:
    """What :func:`fused` makes of a graph: the graph the plan walks (each region one call), the regions, and the program they came from."""

    graph: Graph
    regions: "list[Region]"
    program: "list[Node]"
    origin: "list[int]"  # program node -> the index of the node of the traced graph it came from
    rewrites: "list[dict[str, Any]]"
    settings: Fusion
    source: Graph  # the traced graph

    @property
    def fused_regions(self) -> "list[Region]":
        """The regions of more than one op: what an export verifies."""
        return [r for r in self.regions if len(r.nodes) > 1]


_lock = threading.Lock()
_kernels: "dict[str, Any]" = {}  # content hash -> the synthetic KernelOp (the same region in two graphs is one operator)


def fused(gop: Any, grouping: "str | None" = None) -> Fused:
    """The fusion of a graph (cached on it): ``grouping`` overrides the declaration's (``"none"``: the unfused composition)."""
    settings = settings_of(gop)
    if grouping is not None and grouping != settings.grouping:
        settings = Fusion(settings.version, grouping, settings.math, settings.intermediates, settings.fp_contract, settings.verify, settings.sass)
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
    other.fusion = Fusion(version, s.grouping, s.math, s.intermediates, s.fp_contract, s.verify, s.sass)
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
        if self.settings.intermediates == "opmath":
            nodes, origin = self._opmath(nodes, origin)
        if self.settings.fp_contract == "mul-add":
            nodes, origin = self._mul_add(nodes, origin)
        for n, o in zip(nodes, origin):  # an op a rewrite made is where the op it was made of was written (tilefish.explain)
            if n.loc is None:
                n.loc = self.source.nodes[o].loc
        return nodes, origin

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
                for i in n.inputs:
                    if isinstance(i, Scalar):
                        ins.append(Scalar(float(i.value), f32, i.name))
                    else:
                        wide = self.new(i.type.materialized().with_dtype(f32))
                        out_nodes.append(Node("cast", [i], {"to": f32, "rounding": False}, [wide]))
                        out_origin.append(o)
                        ins.append(wide)
                r = self.new(n.outputs[0].type.with_dtype(f32))
                out_nodes.append(Node(n.op, ins, _clean(n.attrs), [r]))
                out_origin.append(o)
                out_nodes.append(Node("cast", [r], {"to": dt, "rounding": True}, [n.outputs[0]]))
                out_origin.append(o)
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

    def _negated(self, x: Any, into: "list[Node]") -> Any:
        if isinstance(x, Scalar):
            return Scalar(-float(x.value), x.dtype, None)  # the sign of an f32 constant flipped: exact
        neg = self.new(x.type.materialized())
        into.append(Node("neg", [x], {}, [neg]))
        return neg

    # -- 2. the regions ------------------------------------------------------------------------------------------------------------

    def prepare(self, prog: "list[Node]", origin: "list[int]") -> None:
        """The tables every step of the pass reads: producers, consumers, what escapes the arithmetic, the edges, the free ops."""
        self.prog, self.origin = prog, origin
        self.producer: "dict[Value, int]" = {o: k for k, n in enumerate(prog) for o in n.outputs}
        self.consumers: "dict[Value, list[int]]" = {}
        for k, n in enumerate(prog):
            for i in n.inputs:
                if isinstance(i, Value):
                    self.consumers.setdefault(i, []).append(k)
        self.escapes: "set[Value]" = set(self.source.outputs) | set(self.source.finals.values())
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
        graph = self._graph(regions)
        return Fused(graph, regions, prog, origin, list(self.rewrites), self.settings, self.source)

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

    # -- 4. the graph the plan walks -----------------------------------------------------------------------------------------------

    def _graph(self, regions: "list[Region]") -> Graph:
        prog = self.prog
        region_of: "dict[int, Region]" = {k: r for r in regions for k in r.nodes}
        dropped = set(self.free) | set(getattr(self, "dead", ())) | {k for k, n in enumerate(prog) if n.op == "barrier"}
        # units: a region, or one node that stays (a call, a view, a move)
        unit_of: "dict[int, Any]" = {}
        units: "list[Any]" = []
        for k, n in enumerate(prog):
            if k in dropped:
                continue
            if k in region_of:
                r = region_of[k]
                if r.nodes[0] == k:
                    units.append(r)
                unit_of[k] = r
            else:
                units.append(k)
                unit_of[k] = k
        first = {id(u): (u.nodes[0] if isinstance(u, Region) else u) for u in units}
        succ: "dict[int, set[int]]" = {id(u): set() for u in units}
        indeg: "dict[int, int]" = {id(u): 0 for u in units}
        by_id = {id(u): u for u in units}
        for k in range(len(prog)):
            if k not in unit_of:
                continue
            for s in self.succ[k]:
                if s not in unit_of:
                    continue
                a, b = id(unit_of[k]), id(unit_of[s])
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
        for u in order:
            if not isinstance(u, Region):
                nodes.append(prog[u])
                continue
            ins: "list[Value]" = []
            for v, given in zip(u.inputs, u.call_inputs):  # type: ignore[attr-defined]
                if u.reads.get(v) == "materialised":
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
                    ins.append(c)
                else:
                    ins.append(given)
            outs = [self.masks_u8.get(o, o) for o in u.outputs]
            attrs: "dict[str, Any]" = {"kind": "kernel"}
            if u.call is not u.kernel:
                attrs["unaligned"] = tuple(sorted(u.call.unaligned_params))
            node = Node(u.kernel.name, ins, attrs, outs)
            node.callee = u.call
            nodes.append(node)
        g = self.source
        return Graph(g.name, g.kind, g.inputs, g.consts, nodes, g.outputs, g.requirements, g.finals)


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
        if verification is not None and r.index in verification:
            item["verified"] = verification[r.index]
        regions.append(item)
    return {"schema": REPORT_SCHEMA, "graph": f.source.name, "fusion": f.settings.canonical(), "regions": regions}


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
    "epilogue-library": ("GEMM / convolution epilogue: future work (RFC 0001, section 7 route (b) and section 11.2)", "future"),
    "epilogue-extern": ("an epilogue hook of the extern (RFC 0001, section 7 route (c)): phase 4", "4"),
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
    p.prepare(f.program, f.origin)
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
                    add(f"epilogue-{kind}", _node_ref(f, pk), to, i, f"the result of the call to '{pn.op}' is read by region {r.index}: no fusion across a call")
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
    p.prepare(f.program, f.origin)
    p.free = set(r.free)
    return [(one.kernel, list(one.inputs), list(one.outputs), dict(one.reads)) for one in p.regions(p.partition(list(r.nodes), "none"))]


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
    if not want or not f.fused_regions:
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
        return {r.index: {"skipped": why} for r in f.fused_regions}
    return verify(gop, combos)


__all__ = ["Fusion", "TilefishFusionError", "TilefishFusionWarning", "fusion", "fused", "planned_graph", "report", "verify"]
