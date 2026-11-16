"""The plan-level fusion advisor (``tilefish-explain DIR --graph G --advise``, :func:`tilefish.advise`): which steps of a launch plan could be one
kernel, how many bytes and microseconds that would save, and what to do about it (``docs/advise.md``).

It reads an export directory and needs no GPU: the plan (the steps of each bucket, their buffers and views), the manifest (what each step launches: a
pinned GEMM or convolution and its sizes, a kernel and the access of each of its pointers, whether tilefish generated it and with which template), the
fusion report (``<graph>.fusion.json``) and, when the export has one, the trace (``trace/graphs/<graph>.json``: the ops of the graph body each fused region
took, and why the regions are where they are). A timing file of the steps (:func:`tilefish.profile_plan`) replaces the estimates by measured times.

1. **Each step**: the bytes it reads and writes at the bucket (from the shapes of the buffers and views it binds, and the access of each pointer); the
   FLOPs of a GEMM or a convolution (from its sizes); a class: ``library``, ``fused`` (a region of the graph body), ``elementwise`` and ``row`` (a
   ``@tilefish.kernel`` lowered by that template), ``copy`` (a copy or memset step, a generated ``tilefish.contiguous`` or ``im2col``) or ``opaque`` (a
   hand-written Triton or Gluon extern, a prebuilt binary); memory-bound or compute-bound by a roofline against the machine balance of the architecture
   (:data:`MACHINES`).
2. **Producer and consumer**: a buffer that one step writes and exactly one other step reads, and that is not returned, carried, updated in place, a
   workspace or a buffer the caller keeps (a tap), is an intermediate that one kernel of both would never store. Steps joined by such intermediates
   form a candidate, which a library step bounds (a GEMM or a convolution is never fused with another; its epilogue and prologue are future work).
3. **Why it is not one kernel today, and what to do**: a hand-written extern (write its arithmetic in the graph body so phase 1 fuses it), a
   ``@tilefish.kernel`` (write its body in the graph body), a row kernel or a reduction (phase 2), a reduction without ``order=`` (D2), a
   ``tilefish.barrier``, a view, a call between regions, a GEMM epilogue (future work).
4. **Ranked** by the time it would save: measured, when a timing file is given (each step's time above the launch floor shrinks with the bytes it
   no longer moves, and each kernel that disappears saves the floor), else the bytes saved over the bandwidth of the machine.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .errors import TilefishError

ADVICE_SCHEMA = "tilefish.advice.v1"
TIMING_SCHEMA = "tilefish.step-timing.v1"


class TilefishAdviseError(TilefishError, ValueError):
    """What the advisor was asked for is not in the export (no plan of the graph, no such bucket), or a timing file is not of its plan."""


# -- the machines ---------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Machine:
    """The roofline of an architecture: memory bandwidth (GB/s) and dense peaks (TFLOPS) of a whole GPU of it, as the vendor states them (rounded)."""

    arch: str
    example: str
    bandwidth: float  # GB/s
    tensor16: float  # dense f16 / bf16 MMA with an f32 accumulator, TFLOPS
    tf32: float  # dense TF32 MMA, TFLOPS
    fp32: float  # f32 FMA (no tensor core), TFLOPS
    sms: int  # the SMs of the part the peaks are of

    @property
    def balance(self) -> float:
        """FLOPs per byte at which a 16-bit GEMM stops being memory-bound."""
        return self.tensor16 * 1e12 / (self.bandwidth * 1e9)

    def peak(self, kind: str) -> float:
        return {"tensor16": self.tensor16, "tf32": self.tf32, "fp32": self.fp32}[kind]

    def scaled(self, bandwidth: "float | None" = None, sms: "int | None" = None) -> "Machine":
        """The same architecture with another bandwidth (measured, or a partition's) and, for a partition of ``sms`` SMs, its share of the peaks."""
        f = min(1.0, sms / self.sms) if sms else 1.0
        return Machine(self.arch, self.example, bandwidth or self.bandwidth, self.tensor16 * f, self.tf32 * f, self.fp32 * f, sms or self.sms)


# Whole-GPU numbers of one part of each architecture (vendor data sheets, dense, rounded; approximate). A partition (a MIG slice, a green context)
# has a share of them: a timing file records the device's SMs and the copy bandwidth it measured, and the advisor scales by those. The ranking by
# bytes does not depend on the table; the microseconds and the roofline's memory- or compute-bound do.
MACHINES: "dict[str, Machine]" = {
    "sm_90": Machine("sm_90", "H100 SXM", 3350.0, 989.0, 495.0, 67.0, 132),
    "sm_100": Machine("sm_100", "B200", 8000.0, 2250.0, 1100.0, 80.0, 148),
    "sm_103": Machine("sm_103", "B300", 8000.0, 2250.0, 1100.0, 75.0, 148),
    "sm_120": Machine("sm_120", "RTX 5090", 1792.0, 209.5, 104.8, 104.8, 170),
}

_ITEMSIZE = {"bf16": 2, "f16": 2, "f32": 4, "i32": 4, "i64": 8, "u8": 1, "e4m3": 1, "e5m2": 1}
_GENERATED_COPY = re.compile(r"^(contiguous|im2col1d|im2col|col2im1d)_")
_MMA = re.compile(r"\b(HMMA|IMMA|QMMA|OMMA|DMMA|HGMMA|IGMMA|QGMMA|UTCHMMA|UTCQMMA|UTCIMMA|UTCOMMA)\b")
_PHASES = ("1", "2", "4", "future")
STREAMING = 0.25  # a measured step moving its bytes (above the launch floor) at this share of the bandwidth or more is taken to be bound by them
CLASSES = ("library", "fused", "elementwise", "row", "copy", "opaque")

# blocker -> (what to do, the phase of docs/rfcs/0001-plan-fusion.md that does it). The reasons of the fusion pass (tilefish._fusion.boundaries) are
# merged in at import below.
BLOCKERS: "dict[str, tuple[str, str]]" = {
    "opaque-extern": ("write this extern's arithmetic in the graph body so fusion takes it (an elementwise one or, with order= on its sums, a row one)", "1"),
    "opaque-unknown": (
        "a hand-written kernel (export with trace=True to tell whether it computes MMAs): elementwise or row arithmetic, write it in the graph body so "
        "phase 1 (or 2) fuses it; a GEMM, an attention or a convolution: an epilogue hook of it (phase 4) where it makes the intermediate, a prologue hook (future work) where it reads it",
        "1",
    ),
    "opaque-compute-epilogue": (
        "the extern is bound by its arithmetic (MMAs in its SASS, or measured far below the bandwidth: a GEMM, an attention, a convolution, a search): "
        "an epilogue hook of it (RFC 0001, section 7 route (c)), phase 4",
        "4",
    ),
    "opaque-compute-prologue": ("the extern is bound by its arithmetic: a prologue hook of it, future work (RFC 0001, section 11.3)", "future"),
    "dsl-kernel": ("write the @tilefish.kernel's body in the graph body so phase 1 fuses it (exact as it is when it declares fp_fusion=False; RFC 0001, 8.2)", "1"),
    "dsl-kernel-contracts": (
        "write the @tilefish.kernel's body in the graph body so phase 1 fuses it; it contracts (fp_fusion is not False): write tilefish.fma where it "
        "meant one rounding, or its bytes change (RFC 0001, 8.2)",
        "1",
    ),
    "row-kernel": ("a row @tilefish.kernel: write its body in the graph body, with order= on its sums, and fusion version 2 (the default) fuses it into a row region", "2"),
    "generated-movement": ("a generated data-movement op (im2col, col2im) around a convolution: a convolution prologue or epilogue, future work", "future"),
    "unknown-boundary": ("two regions of the graph body: export with trace=True (or pass the graph) to say why (a barrier, a view, a limit, another iteration space)", "1"),
    "copy-step": ("a copy step: return or store the value where it is made (the step that makes it can write the destination)", "1"),
    "library-in-the-middle": (
        "a library op sits between them (the consumer also reads what it computes from the producer's value): one kernel of both would need the GEMM "
        "or convolution inside it, future work (RFC 0001, section 7)",
        "future",
    ),
    "strided-output": ("the producer writes the intermediate into part of a buffer (a column block, a head): a strided store, phase 2 (RFC 0001, 3.5 rule 7)", "2"),
    "memset-step": ("a memset step: its zeros are read from memory, never folded into a region (RFC 0001, H2)", "future"),
    "library-library": ("two library ops: a GEMM or a convolution is never fused with another", "future"),
    "step-in-the-middle": ("another step sits between them (a path from the producer to the consumer through it): one kernel would need it inside too", "1"),
}


def _merge_fusion_fixes() -> None:
    from ._fusion import BOUNDARY_FIXES

    for k, v in BOUNDARY_FIXES.items():
        BLOCKERS.setdefault(k, v)


_merge_fusion_fixes()


def _phase_rank(p: str) -> int:
    return _PHASES.index(p) if p in _PHASES else len(_PHASES)


# -- reading an export ----------------------------------------------------------------------------------------------------------------------------


def _read_json(path: pathlib.Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise TilefishAdviseError(f"cannot read {what} {str(path)!r}: {e}") from e


def graphs_of(directory: "str | pathlib.Path") -> "list[str]":
    """The graphs whose plans an export directory holds (one ``plan.json``, or a plan set's ``plans.json``)."""
    d = pathlib.Path(directory)
    if (d / "plans.json").is_file():
        index = _read_json(d / "plans.json", "the plan set index")
        return sorted((index.get("plans") or {}).keys()) if isinstance(index, dict) else []
    if (d / "plan.json").is_file():
        doc = _read_json(d / "plan.json", "the plan")
        return [doc.get("graph")] if isinstance(doc, dict) and doc.get("graph") else []
    return []


def _plan_doc(directory: pathlib.Path, graph: "str | None") -> "dict[str, Any]":
    names = graphs_of(directory)
    if graph is None:
        if len(names) != 1:
            raise TilefishAdviseError(f"{str(directory)!r} holds the plans of {names or 'no graph'}: say which with --graph")
        graph = names[0]
    for name in (f"{graph}.plan.json", "plan.json"):
        path = directory / name
        if path.is_file():
            doc = _read_json(path, "the plan")
            if isinstance(doc, dict) and doc.get("graph") == graph:
                return doc
    raise TilefishAdviseError(f"{str(directory)!r} has no plan of the graph '{graph}' (plan.json or {graph}.plan.json; it has {names or 'none'})")


def _parse_bucket(bucket: "str | Mapping[str, int] | None") -> "dict[str, int] | None":
    if bucket is None or isinstance(bucket, Mapping):
        return None if bucket is None else {str(k): int(v) for k, v in bucket.items()}
    out = {}
    for part in str(bucket).split(","):
        name, eq, value = part.partition("=")
        if not eq or not value.strip().lstrip("-").isdigit():
            raise TilefishAdviseError(f"a bucket is NAME=VALUE[,NAME=VALUE...] (D=96), got {bucket!r}")
        out[name.strip()] = int(value)
    return out


# -- the steps ------------------------------------------------------------------------------------------------------------------------------------


@dataclass
class _Arg:
    name: str
    binding: str
    access: str  # "in" | "out" | "inout"
    view: "dict[str, Any] | None"


@dataclass
class StepInfo:
    """What a step is, the same in every bucket: its kind, what it launches, its class, and how each argument is accessed."""

    index: int
    kind: str
    entry: "str | None"
    op: "str | None"
    cls: str
    sub: str
    label: str
    args: "list[_Arg]"
    contracts: bool = False  # a @tilefish.kernel that lets the compiler contract (fp_fusion not False)
    mma: "bool | None" = None  # an opaque kernel: whether its SASS has MMAs (None: unknown, no trace)
    arith_bound: bool = False  # measured: it moves its bytes (above the launch floor) at less than STREAMING of the bandwidth

    def blocker(self, role: str) -> "str | None":
        """The blocker this step is for a fusion with a neighbour (``role``: ``"producer"`` or ``"consumer"`` of the intermediate)."""
        if self.cls == "opaque":
            if self.mma or self.arith_bound:
                return "opaque-compute-epilogue" if role == "producer" else "opaque-compute-prologue"
            return "opaque-extern" if self.mma is False else "opaque-unknown"
        if self.cls == "elementwise":
            return "dsl-kernel-contracts" if self.contracts else "dsl-kernel"
        if self.cls == "row":
            return "row-kernel"
        if self.cls == "copy":
            if self.sub.startswith("generated:contiguous"):
                return "materialised"
            if self.sub.startswith("generated"):
                return "generated-movement"
            return "memset-step" if self.sub == "memset" else "copy-step"
        return None


def _flops(entry: "Mapping[str, Any]") -> "tuple[int | None, str | None]":
    impl = entry.get("impl")
    if impl in ("cublaslt.gemm", "cublaslt.gemm_bias"):
        m, n, k = int(entry.get("m", 0)), int(entry.get("n", 0)), int(entry.get("k", 0))
        dt = (entry.get("dtypes") or {}).get("a", "bf16")
        peak = "tensor16" if dt in ("bf16", "f16") else ("tf32" if "TF32" in str(entry.get("compute_type", "")) else "fp32")
        return 2 * m * n * k, peak
    if impl == "cudnn.conv_fwd":
        out = math.prod(int(x) for x in entry.get("output") or [1])
        flt = math.prod(int(x) for x in entry.get("filter") or [1])
        n, c, k, g = (int(entry.get(x, 1)) for x in ("n", "c", "k", "groups"))
        dt = (entry.get("dtypes") or {}).get("x", "f16")
        peak = "tensor16" if dt in ("bf16", "f16") else ("tf32" if entry.get("math") == "tf32" else "fp32")
        return 2 * n * k * out * (c // max(g, 1)) * flt, peak
    return None, None


def _library_label(entry: "Mapping[str, Any]") -> str:
    impl = entry.get("impl", "")
    if impl.startswith("cublaslt"):
        dt = (entry.get("dtypes") or {}).get("a", "?")
        return f"GEMM {entry.get('m')}x{entry.get('n')}x{entry.get('k')} {dt}" + (" + bias" if impl.endswith("_bias") else "")
    if impl == "cudnn.conv_fwd":
        dt = (entry.get("dtypes") or {}).get("x", "?")
        return f"conv n{entry.get('n')} c{entry.get('c')} k{entry.get('k')} out{entry.get('output')} filter{entry.get('filter')} g{entry.get('groups')} {dt}"
    return str(impl)


class _Export:
    """The parts of an export the advisor reads, indexed: the manifest's entries, the fusion report, the trace of the graph."""

    def __init__(self, manifest: "Mapping[str, Any]", arch: "str | None", report: "Mapping[str, Any] | None", gdoc: "Mapping[str, Any] | None",
                 boundaries: "Sequence[Mapping[str, Any]] | None", trace: Any = None) -> None:
        self.kernels: "dict[str, dict[str, Any]]" = {}
        self.library: "dict[str, dict[str, Any]]" = {}
        for e in manifest.get("kernels") or []:
            if arch is None or e.get("arch") == arch:
                self.kernels.setdefault(e.get("stem"), e)
        for e in manifest.get("library") or []:
            if arch is None or e.get("arch") == arch:
                self.library.setdefault(e.get("stem"), e)
        self.arch = arch
        self.report = report
        self.gdoc = gdoc
        self.boundaries = list(boundaries) if boundaries is not None else (list(gdoc.get("boundaries") or []) if gdoc and "boundaries" in gdoc else None)
        self.trace = trace  # the export's trace (tilefish.explain.Trace), or None
        # the fusion version of the graph (the report's settings): what is a region of its own depends on it (tilefish._fusion.is_solo)
        version = ((report or {}).get("fusion") or {}).get("version", 1)
        self.version = version if isinstance(version, int) and not isinstance(version, bool) else 1
        self.regions: "dict[str, dict[str, Any]]" = {}  # region kernel name -> {"index", "template", "ops", "body", "version"}
        for i, r in enumerate((report or {}).get("regions") or []):
            self.regions[r.get("op")] = {"index": i, "template": r.get("template"), "nodes": r.get("nodes") or [], "roundings": r.get("roundings") or [],
                                         "inputs": r.get("inputs") or [], "version": self.version}
        for name, r in ((gdoc or {}).get("regions") or {}).items():
            have = self.regions.setdefault(r.get("kernel", name), {"index": r.get("region"), "version": self.version})
            have.setdefault("template", r.get("template"))
            have["body"] = r.get("body")
            if name != r.get("kernel"):
                self.regions[name] = have
        self._mma: "dict[str, bool | None]" = {}

    def region_of(self, op: "str | None") -> "dict[str, Any] | None":
        if not op:
            return None
        return self.regions.get(op) or self.regions.get(op.split("__ua", 1)[0])

    def mma(self, stem: str) -> "bool | None":
        """Whether the SASS of a kernel has MMA instructions, from the trace (its entry of this binary, checked as ``tilefish-explain`` checks it);
        ``None`` when the trace has no SASS of it."""
        if stem in self._mma:
            return self._mma[stem]
        found: "bool | None" = None
        entry = self.kernels.get(stem) or {}
        if self.trace is not None and self.arch and entry.get("sha256"):
            try:
                t = self.trace.kernel(stem, self.arch, entry["sha256"])
                path = self.trace.root / f"{stem}.{self.arch}" / "sass.txt"
                if t is not None and "error" not in (t.get("sass") or {"error": ""}) and path.is_file():
                    found = bool(_MMA.search(path.read_text(encoding="utf-8", errors="replace")))
            except (TilefishError, OSError):
                found = None
        self._mma[stem] = found
        return found


def _access_table(entry: "Mapping[str, Any] | None", kind: str) -> "dict[str, str]":
    if kind == "copy":
        return {"src": "in", "dst": "out"}
    if kind == "memset":
        return {"dst": "out"}
    if entry is None:
        return {}
    if kind == "library":
        return {p.get("name"): ("out" if p.get("name") == "out" else "in") for p in (entry.get("abi") or {}).get("params") or []}
    return {p.get("name"): p.get("access", "in") for p in (entry.get("abi") or {}).get("params") or [] if p.get("kind") in ("pointer", "tensordesc")}


def _step_info(k: int, st: "Mapping[str, Any]", ex: _Export) -> StepInfo:
    kind = st.get("kind", "kernel")
    entry_name = st.get("entry")
    if kind in ("copy", "memset"):
        args = [_Arg(a["name"], a["binding"], "in" if a["name"] == "src" else "out", a.get("view")) for a in st.get("args") or []]
        return StepInfo(k, kind, None, None, "copy", kind, f"{kind} step", args)
    entry = ex.library.get(entry_name) if kind == "library" else ex.kernels.get(entry_name)
    access = _access_table(entry, kind)
    args = []
    for a in st.get("args") or []:
        acc = access.get(a["name"]) or ("out" if a["name"].startswith("out") else "in")
        args.append(_Arg(a["name"], a["binding"], acc, a.get("view")))
    op = (entry or {}).get("op") or (entry_name.split(".", 1)[0] if entry_name else None)
    if kind == "library":
        info = StepInfo(k, kind, entry_name, op, "library", "conv" if (entry or {}).get("impl") == "cudnn.conv_fwd" else "gemm",
                        f"{op} [{_library_label(entry or {})}]", args)
        return info
    region = ex.region_of(op)
    generated = bool((entry or {}).get("source_sha256"))
    if region is not None or (op or "").startswith("fused_"):
        template = (region or {}).get("template") or "elementwise"
        nodes = (region or {}).get("nodes") or (region or {}).get("body") or []
        sub = "reduction" if _solo_reason(region or {}) is not None else template
        shown = (region or {}).get("body") or nodes  # the trace's: with the line of each op
        body = ", ".join(f"#{n.get('index', n.get('node'))} {n.get('op')}" + (f" ({n['loc']})" if n.get("loc") else "") for n in shown)
        where = f"region {region['index']}" if region and region.get("index") is not None else "region"
        info = StepInfo(k, kind, entry_name, op, "fused", sub, f"{op} [fused {where}, {sub}" + (f": {body}]" if body else "]"), args)
        return info
    if generated and _GENERATED_COPY.match(op or ""):
        name = _GENERATED_COPY.match(op or "").group(1)  # type: ignore[union-attr]
        return StepInfo(k, kind, entry_name, op, "copy", f"generated:{name}", f"{op} [generated {name}]", args)
    if generated:
        consts = {c.get("name") for c in (entry or {}).get("constexprs") or []}
        template = "row" if "BLOCK_D" in consts else "elementwise"
        info = StepInfo(k, kind, entry_name, op, template, "kernel", f"{op} [@tilefish.kernel, {template} template]", args)
        info.contracts = ((entry or {}).get("options") or {}).get("enable_fp_fusion") is not False
        return info
    lang = "binary" if (entry or {}).get("kind") == "binary" else ("gluon" if (entry or {}).get("source_digest_kind") == "tilefish.gluon-ir.v1" else "triton")
    info = StepInfo(k, kind, entry_name, op, "opaque", lang, "", args)
    info.mma = ex.mma(entry_name) if entry_name else None
    what = {"binary": "prebuilt binary", "gluon": "Gluon extern", "triton": "Triton extern"}[lang]
    info.label = f"{op} [{what}" + ({True: ", MMA", False: ", no MMA", None: ""}[info.mma]) + "]"
    return info


def _elements(arg: _Arg, shapes: "Mapping[str, tuple[str, tuple[int, ...]]]") -> int:
    if arg.view is not None:
        return math.prod(int(x) for x in arg.view.get("shape") or [1])
    return math.prod(shapes[arg.binding][1]) if arg.binding in shapes else 0


def _itemsize(binding: str, shapes: "Mapping[str, tuple[str, tuple[int, ...]]]") -> int:
    return _ITEMSIZE.get(shapes[binding][0], 4) if binding in shapes else 4


@dataclass
class _Cost:
    """A step at one bucket: bytes per buffer read and written, FLOPs, the roofline."""

    reads: "dict[str, int]"
    writes: "dict[str, int]"
    flops: "int | None"
    est_us: float
    bound: str
    intensity: "float | None"

    @property
    def nbytes(self) -> int:
        return sum(self.reads.values()) + sum(self.writes.values())


def _cost(info: StepInfo, shapes: "Mapping[str, tuple[str, tuple[int, ...]]]", ex: _Export, machine: Machine, step: "Mapping[str, Any]") -> _Cost:
    """A step at one bucket: ``step`` is the bucket's own (its entry and its views' numbers are the bucket's; the bindings are every bucket's)."""
    reads: "dict[str, int]" = {}
    writes: "dict[str, int]" = {}
    entry_name = step.get("entry")
    views = [x.get("view") for x in step.get("args") or []]
    for i, a in enumerate(info.args):
        if a.binding not in shapes:
            continue
        here = _Arg(a.name, a.binding, a.access, views[i] if i < len(views) else a.view)
        b = _elements(here, shapes) * _itemsize(a.binding, shapes)
        if a.access in ("in", "inout"):
            reads[a.binding] = reads.get(a.binding, 0) + b
        if a.access in ("out", "inout"):
            writes[a.binding] = writes.get(a.binding, 0) + b
    flops = None
    peak = None
    if info.kind == "library":
        flops, peak = _flops(ex.library.get(entry_name or "") or {})
    nbytes = sum(reads.values()) + sum(writes.values())
    t_mem = nbytes / (machine.bandwidth * 1e9) * 1e6
    if flops:
        t_flop = flops / (machine.peak(peak or "tensor16") * 1e12) * 1e6
        intensity = flops / nbytes if nbytes else float("inf")
        balance = machine.peak(peak or "tensor16") * 1e12 / (machine.bandwidth * 1e9)
        return _Cost(reads, writes, flops, max(t_mem, t_flop), "compute" if intensity > balance else "memory", intensity)
    bound = "unknown" if (info.cls == "opaque" and info.mma is not False) else "memory"
    return _Cost(reads, writes, None, t_mem, bound, None)


# -- producer and consumer ------------------------------------------------------------------------------------------------------------------------


@dataclass
class _Value:
    buffer: str
    writer: "int | None"
    readers: "list[int]"


def _values(infos: "Sequence[StepInfo]") -> "tuple[list[_Value], dict[str, int]]":
    """Every version of every buffer: who wrote it and who read it (in step order). And how many times each buffer is written."""
    current: "dict[str, _Value]" = {}
    out: "list[_Value]" = []
    writes: "dict[str, int]" = {}
    for info in infos:
        for a in info.args:
            if a.access in ("in", "inout"):
                v = current.get(a.binding)
                if v is None:
                    v = current[a.binding] = _Value(a.binding, None, [])
                    out.append(v)
                if info.index not in v.readers:
                    v.readers.append(info.index)
        for a in info.args:
            if a.access in ("out", "inout"):
                writes[a.binding] = writes.get(a.binding, 0) + 1
                v = _Value(a.binding, info.index, [])
                current[a.binding] = v
                out.append(v)
    return out, writes


def _reachability(infos: "Sequence[StepInfo]") -> "tuple[list[int], list[int]]":
    """For every step, the steps that depend on it (``reach``, a bit set) and those it depends on (``coreach``), through the data (a step reads a
    version another wrote) and the order of the accesses to a buffer (a write after reads of the version before it, a write after a write)."""
    n = len(infos)
    succ: "list[set[int]]" = [set() for _ in range(n)]
    writer: "dict[str, int]" = {}
    readers: "dict[str, list[int]]" = {}
    for info in infos:
        k = info.index
        for a in info.args:
            if a.access in ("in", "inout"):
                w = writer.get(a.binding)
                if w is not None and w != k:
                    succ[w].add(k)
                readers.setdefault(a.binding, []).append(k)
        for a in info.args:
            if a.access in ("out", "inout"):
                w = writer.get(a.binding)
                if w is not None and w != k:
                    succ[w].add(k)
                for r in readers.get(a.binding, []):
                    if r != k:
                        succ[r].add(k)
                writer[a.binding] = k
                readers[a.binding] = []
    reach = [0] * n
    for k in range(n - 1, -1, -1):  # the steps are in an order that respects these edges (every edge goes forward)
        m = 0
        for s in succ[k]:
            m |= (1 << s) | reach[s]
        reach[k] = m
    coreach = [0] * n
    for k in range(n):
        for s in succ[k]:
            coreach[s] |= (1 << k) | coreach[k]
    return reach, coreach


@dataclass
class Edge:
    """An intermediate that one kernel of its producer and consumer would not store."""

    buffer: str
    producer: int
    consumer: int
    blockers: "list[str]"
    detail: "list[str]"


def _classify_values(doc: "Mapping[str, Any]", infos: "Sequence[StepInfo]", keep: "Iterable[str]") -> "tuple[list[Edge], list[dict[str, Any]], list[dict[str, Any]]]":
    """The fusible intermediates (single producer, single consumer), the fan-outs (one producer, several consumers), and the buffers kept, with why."""
    roles = {b["name"]: b["role"] for b in doc.get("buffers") or []}
    keep = set(keep)
    values, writes = _values(infos)
    edges: "list[Edge]" = []
    fanouts: "list[dict[str, Any]]" = []
    kept: "list[dict[str, Any]]" = []
    for v in values:
        if v.writer is None:
            continue
        role = roles.get(v.buffer)
        readers = v.readers  # (a step that reads and writes a buffer reads the version before the one it writes)
        why = None
        if role in ("output", "carried", "inout"):
            why = {"output": "returned", "carried": "carried to a later plan", "inout": "updated in place (an InOut parameter)"}[role]
        elif role != "intermediate":
            continue
        elif v.buffer in keep:
            why = "kept (a tap)"
        elif writes.get(v.buffer, 0) > 1:
            why = "written more than once (an in-place update, or a result written over its input)"
        elif not readers:
            why = "read by no step"
        if why is not None:
            if readers:
                kept.append({"buffer": v.buffer, "producer": v.writer, "consumers": readers, "why": why})
            continue
        if len(readers) == 1:
            edges.append(Edge(v.buffer, v.writer, readers[0], [], []))
        else:
            fanouts.append({"buffer": v.buffer, "producer": v.writer, "consumers": readers})
    return edges, fanouts, kept


# -- why a pair is not one kernel -----------------------------------------------------------------------------------------------------------------


def _solo_reason(r: "Mapping[str, Any]") -> "tuple[str, str] | None":
    """What the fusion report says of a region that is one op a region of its own in its fusion version (a reduction without an order in every
    version; in version 1 any reduction, or a gather): ``(reason, what)``."""
    from ._fusion import reduction_reason

    ops = [n.get("op") for n in r.get("nodes") or []]
    if len(ops) == 2 and ops == ["mul", "sum"]:  # an fma sum with its product (one unit in every grouping)
        ops = ["sum"]
    if len(ops) != 1:
        return None
    ordered = any("order" in x for x in r.get("roundings") or [])
    reason = reduction_reason(ops[0], ordered, r.get("version", 1))
    if reason is None:
        return None
    return reason, f"region {r.get('index')} is a '{ops[0]}'" + (" with order=" if ordered else " without order=" if reason == "reduction-unordered" else "")


def _fallback_reasons(a: StepInfo, b: StepInfo, ex: _Export) -> "list[tuple[str, str]]":
    """Without the trace's boundaries: what the fusion report says of two regions."""
    out = [found for found in (_solo_reason(ex.region_of(s.op) or {}) for s in (a, b)) if found is not None]
    if not out:
        if "row" in (a.sub, b.sub) and ex.version < 2:  # (version 2 makes row regions of elementwise ops and reductions alike)
            out.append(("template", "one of them is a row region"))
        else:
            out.append(("unknown-boundary", "the trace (or the graph) says why"))
    return out


def _region_reasons(a: StepInfo, b: StepInfo, ex: _Export) -> "list[tuple[str, str]]":
    if ex.boundaries is None:
        return _fallback_reasons(a, b, ex)
    ka, kb = (ex.region_of(a.op) or {}).get("index"), (ex.region_of(b.op) or {}).get("index")
    out = []
    for bd in ex.boundaries:
        fa, tb = bd.get("from") or {}, bd.get("to") or {}
        names_a = {a.op, (a.op or "").split("__ua", 1)[0]}
        names_b = {b.op, (b.op or "").split("__ua", 1)[0]}
        if (fa.get("kernel") in names_a or (ka is not None and fa.get("region") == ka)) and (tb.get("kernel") in names_b or (kb is not None and tb.get("region") == kb)):
            out.append((bd.get("reason", "unknown-boundary"), bd.get("detail", "")))
    return out or _fallback_reasons(a, b, ex)


def _whole(view: "Mapping[str, Any] | None", shape: "Sequence[int]") -> bool:
    """Whether an argument's view is its whole buffer (or a dense reshape of all of it): what a region reads and writes in phase 1."""
    if view is None:
        return True
    vshape = [int(x) for x in view.get("shape") or []]
    if view.get("offset", 0) != 0 or math.prod(vshape) != math.prod(shape):
        return False
    want, stride = [], 1
    for size in reversed(vshape):
        want.append(stride)
        stride *= size
    return all(s == w or n == 1 for s, w, n in zip(view.get("strides") or [], reversed(want), vshape))


def _edge_blockers(e: Edge, infos: "Sequence[StepInfo]", ex: _Export, shapes: "Mapping[str, tuple[str, tuple[int, ...]]]") -> None:
    a, b = infos[e.producer], infos[e.consumer]
    shape = shapes.get(e.buffer, ("", ()))[1]
    # the producer writes the intermediate into part of its buffer (a column block, a head): a strided store, phase 2 (RFC 0001, 3.5 rule 7);
    # a row step reads it through a view that is not dense: phase 2 too (rule 5); an elementwise one reads it where it is in phase 1, and so does a
    # row region of fusion version 2
    if any(x.binding == e.buffer and x.access in ("out", "inout") and not _whole(x.view, shape) for x in a.args):
        e.blockers.append("strided-output")
        e.detail.append(f"step {a.index} writes {e.buffer} through a view of part of it")
    row_reader = b.cls == "row" or (b.sub in ("row", "reduction") and ex.version < 2)
    if row_reader and any(x.binding == e.buffer and x.access in ("in", "inout") and not _whole(x.view, shape) for x in b.args):
        e.blockers.append("materialised")
        e.detail.append(f"step {b.index} (a row step) reads {e.buffer} through a view that is not dense")
    if a.cls == "library":
        e.blockers.append("epilogue-library")
        e.detail.append(f"step {a.index} is a pinned {a.sub}: its epilogue is future work")
    if b.cls == "library":
        e.blockers.append("prologue-library")
        e.detail.append(f"step {b.index} is a pinned {b.sub}: its prologue is future work")
    if a.cls == "fused" and b.cls == "fused":
        for reason, detail in _region_reasons(a, b, ex):
            if reason not in e.blockers:
                e.blockers.append(reason)
                e.detail.append(detail)
    for s, role in ((a, "producer"), (b, "consumer")):
        if s.cls == "library":
            continue
        bl = s.blocker(role)
        if s.cls == "fused" and s.sub in ("row", "reduction") and not (a.cls == "fused" and b.cls == "fused"):
            # a row region next to a step that is not a region: in fusion version 1 it would stay a kernel of its own even with the other written in
            # the graph body (a reduction alone, or a row region the elementwise one cannot join); version 2 joins them but a reduction without an order
            if ex.version >= 2:
                solo = _solo_reason(ex.region_of(s.op) or {})
                bl = solo[0] if solo is not None else None
            else:
                bl = "reduction" if s.sub == "reduction" else "template"
        if bl is not None and bl not in e.blockers:
            e.blockers.append(bl)
            anchored = "library" in (a.cls, b.cls)
            e.detail.append(f"step {s.index} ({s.op or s.kind}) is {_what(s)}" + (" (which an epilogue or prologue of the library op would need in the graph body too)" if anchored else ""))


def _what(s: StepInfo) -> str:
    return {
        "opaque": "a hand-written kernel" + (" with MMAs" if s.mma else ""),
        "elementwise": "a @tilefish.kernel (elementwise template)",
        "row": "a @tilefish.kernel (row template)",
        "copy": {"copy": "a copy step", "memset": "a memset step"}.get(s.sub, f"a generated {s.sub.split(':', 1)[-1]}"),
        "fused": f"a fused region ({s.sub})",
        "library": f"a pinned {s.sub}",
    }.get(s.cls, s.cls)


# -- the advice -----------------------------------------------------------------------------------------------------------------------------------


def _timing_steps(timing: "Mapping[str, Any]", doc: "Mapping[str, Any]", bucket: "Mapping[str, int]") -> "dict[int, float]":
    if timing.get("schema") != TIMING_SCHEMA:
        raise TilefishAdviseError(f"the timing file is not a {TIMING_SCHEMA} document (tilefish.profile_plan writes one)")
    if timing.get("graph") != doc.get("graph"):
        raise TilefishAdviseError(f"the timing file is of the graph '{timing.get('graph')}', not '{doc.get('graph')}'")
    digest = timing.get("plan_digest")
    if not isinstance(digest, str) or (doc.get("plan_digest") and digest != doc["plan_digest"]):
        raise TilefishAdviseError(f"the timing file is of the plan {str(digest)[:12]}..., not of this export's ({str(doc.get('plan_digest'))[:12]}...)")
    steps = next((b["steps"] for b in doc.get("buckets") or [] if b.get("values") == dict(bucket)), None)
    if steps is None:
        raise TilefishAdviseError(f"the plan of '{doc.get('graph')}' has no bucket {dict(bucket)}, which the timing file is of")
    listed = timing.get("steps")
    if not isinstance(listed, list) or len(listed) != len(steps):
        raise TilefishAdviseError(f"the timing file times {len(listed) if isinstance(listed, list) else 'no'} steps, but the plan has {len(steps)} at {dict(bucket)}")
    out: "dict[int, float]" = {}
    for s in listed:
        k = s.get("index") if isinstance(s, Mapping) else None
        if type(k) is not int or not 0 <= k < len(steps) or k in out or steps[k].get("entry") != s.get("entry"):
            raise TilefishAdviseError(f"the timing file's step {k} is not step {k} of the plan at {dict(bucket)} (or is listed twice)")
        us = s.get("us")
        # the median less the cost of the events around it (what profile_plan measured of them), when the file says it
        value = (us["net"] if "net" in us else us.get("median")) if isinstance(us, Mapping) else us
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise TilefishAdviseError(f"the timing file's step {k} has no time in microseconds (us: {us!r})")
        out[k] = float(value)
    return out


def _device_number(timing: "Mapping[str, Any]", key: str) -> "float | None":
    """A number the timing file records of its device (``copy_GBps``, ``sm_count``), or ``None``; refused when it is not a positive number."""
    device = timing.get("device") or {}
    v = device.get(key) if isinstance(device, Mapping) else None
    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0):
        raise TilefishAdviseError(f"the timing file's device {key} is not a positive number ({v!r})")
    return v


def analyse(
    doc: "Mapping[str, Any]",
    manifest: "Mapping[str, Any]",
    *,
    report: "Mapping[str, Any] | None" = None,
    gdoc: "Mapping[str, Any] | None" = None,
    boundaries: "Sequence[Mapping[str, Any]] | None" = None,
    timing: "Mapping[str, Any] | None" = None,
    bucket: "str | Mapping[str, int] | None" = None,
    machine: "Machine | str | None" = None,
    bandwidth: "float | None" = None,
    keep: "Iterable[str]" = (),
    trace: Any = None,
) -> "Advice":
    """The advice for a plan document and its manifest (the documents themselves; :func:`advise` reads them from a directory)."""
    from ._plan import buffer_shapes

    buckets = list(doc.get("buckets") or [])
    if not buckets or not buckets[0].get("steps"):
        raise TilefishAdviseError(f"the plan of '{doc.get('graph')}' has no steps")
    arch = doc.get("arch")
    if isinstance(machine, Machine):
        m = machine
    else:
        key = machine or arch or "sm_103"
        key = key if str(key).startswith("sm_") else f"sm_{key}"
        if key not in MACHINES:
            raise TilefishAdviseError(f"no machine balance for {key} (known: {', '.join(MACHINES)}): pass machine= or --bandwidth")
        m = MACHINES[key]
    measured_bw = sms = None
    if timing is not None:
        if not isinstance(timing, Mapping):
            raise TilefishAdviseError("the timing file is not a JSON object (tilefish.profile_plan writes one)")
        measured_bw = _device_number(timing, "copy_GBps")
        sms = _device_number(timing, "sm_count")
        sms = int(sms) if sms else None
    if bandwidth or measured_bw or sms:
        m = m.scaled(bandwidth=bandwidth or measured_bw, sms=sms)
    ex = _Export(manifest, arch, report, gdoc, boundaries, trace)
    infos = [_step_info(k, st, ex) for k, st in enumerate(buckets[0]["steps"])]
    wanted = _parse_bucket(bucket)
    if timing is not None:
        tbk = timing.get("bucket") or {}
        if not isinstance(tbk, Mapping) or not all(type(v) is int for v in tbk.values()):
            raise TilefishAdviseError(f"the timing file's bucket is not an object of dimension sizes ({tbk!r})")
        tb = {str(k): v for k, v in tbk.items()}
        if wanted is not None and wanted != tb:
            raise TilefishAdviseError(f"the timing file is of the bucket {tb}, not {wanted}")
        wanted = tb
    if wanted is not None:
        chosen = [b for b in buckets if b.get("values") == wanted]
        if not chosen:
            raise TilefishAdviseError(f"the plan of '{doc.get('graph')}' has no bucket {wanted} ({[b.get('values') for b in buckets][:8]}...)")
        analysed = chosen
    else:
        analysed = buckets
    costs: "dict[str, list[_Cost]]" = {}
    for b in analysed:
        shapes = buffer_shapes(doc, b.get("values") or {})
        costs[_key(b)] = [_cost(info, shapes, ex, m, st) for info, st in zip(infos, b["steps"])]
    rank_bucket = max(analysed, key=lambda b: sum(c.nbytes for c in costs[_key(b)]))
    if rank_bucket is not buckets[0]:  # the labels and entries of the bucket it is ranked at (a bucketed GEMM has an entry per bucket)
        infos = [_step_info(k, st, ex) for k, st in enumerate(rank_bucket["steps"])]
    measured = _timing_steps(timing, doc, rank_bucket["values"]) if timing is not None else None
    if measured is not None:  # a hand-written kernel far slower than its bytes is bound by its arithmetic (an attention of FMAs, a search), not glue
        floor = _launch_floor(infos, measured)
        rc = costs[_key(rank_bucket)]
        for info in infos:
            if info.cls == "opaque" and info.index in measured:
                busy = max(measured[info.index] - floor, 1e-3)
                info.arith_bound = rc[info.index].nbytes / (busy * 1e-6) < STREAMING * m.bandwidth * 1e9
    shapes0 = buffer_shapes(doc, rank_bucket.get("values") or {})  # (the views of `infos` are the ranked bucket's)
    edges, fanouts, kept = _classify_values(doc, infos, keep)
    for e in edges:
        _edge_blockers(e, infos, ex, shapes0)
    return Advice(doc, infos, costs, analysed, rank_bucket, m, edges, fanouts, kept, measured, ex, timing)


def _launch_floor(infos: "Sequence[StepInfo]", measured: "Mapping[int, float] | None") -> float:
    """The launch floor (µs): the shortest measured kernel step, what a kernel costs whatever it moves (0 without a timing file)."""
    if not measured:
        return 0.0
    ks = [t for k, t in measured.items() if infos[k].kind == "kernel" and t > 0]
    return min(ks) if ks else 0.0


def _key(b: "Mapping[str, Any]") -> str:
    return ",".join(f"{k}={v}" for k, v in sorted((b.get("values") or {}).items())) or "-"


class Advice:
    """The advice for one plan: its steps, the candidates ranked, the time by class, the unfused arithmetic of the graph body. ``doc`` is the JSON."""

    def __init__(self, plan: Any, infos: Any, costs: Any, analysed: Any, rank_bucket: Any, machine: Machine, edges: "list[Edge]", fanouts: Any, kept: Any,
                 measured: "dict[int, float] | None", ex: _Export, timing: Any) -> None:
        self.plan, self.infos, self.costs, self.analysed = plan, infos, costs, analysed
        self.rank_bucket, self.machine, self.edges, self.fanouts, self.kept = rank_bucket, machine, edges, fanouts, kept
        self.measured, self.ex, self.timing = measured, ex, timing
        self.floor = _launch_floor(self.infos, self.measured)
        self.candidates = self._candidates()
        self.doc = self._doc()

    # -- the model of the time a fusion saves ----------------------------------------------------------------------------------------------------

    def _saved_us(self, steps: "Sequence[int]", saved: "Mapping[int, int]", bkey: str) -> "tuple[float, int]":
        """The time one kernel of ``steps`` would save at a bucket, and the launches it removes. ``saved``: the bytes each step would no longer move."""
        costs = self.costs[bkey]
        has_lib = any(self.infos[k].cls == "library" for k in steps)
        kernels = [k for k in steps if self.infos[k].cls != "library"]
        removed = len(kernels) - (0 if has_lib else 1)
        bw = self.machine.bandwidth * 1e9
        measured = self.measured if (self.measured is not None and bkey == _key(self.rank_bucket)) else None
        total = 0.0
        for k in steps:
            c = costs[k]
            info = self.infos[k]
            s = saved.get(k, 0)
            if not s or (info.cls == "library" and c.bound == "compute"):
                continue
            # a step whose time is its bytes' (it may be: elementwise, row, a copy, a memory-bound GEMM, a kernel without MMAs; and it moves them, above
            # the launch floor, at a quarter of the bandwidth or more): its time above the floor shrinks with the bytes it no longer moves. Any other (an
            # attention of FMAs or MMAs, a kernel whose MMAs are not known, a slow kernel) saves what the bytes take at the bandwidth
            may = info.cls in ("elementwise", "row", "fused", "copy", "library") or (info.cls == "opaque" and info.mma is False)
            busy = max((measured or {}).get(k, 0.0) - self.floor, 1e-3)
            achieved = min(bw, c.nbytes / (busy * 1e-6)) if c.nbytes else bw
            if measured is None or not may or achieved < STREAMING * bw:
                total += s / bw * 1e6
            else:
                total += min(busy, s / achieved * 1e6)
        if measured is not None:
            total += max(removed, 0) * self.floor
            span = sum(measured.get(k, 0.0) for k in steps)
            total = min(total, max(span - self.floor, 0.0))
        return total, max(removed, 0)

    # -- candidates ----------------------------------------------------------------------------------------------------------------------------

    def _middle(self, members: "set[int]") -> "set[int]":
        """The steps outside ``members`` on a path from one of them to another: one kernel of ``members`` would need them inside it too (rule 2 of
        RFC 0001, 3.5: no path may leave a region and come back)."""
        down = 0
        up = 0
        for g in members:
            down |= self.reach[g]
            up |= self.coreach[g]
        mask = down & up
        return {k for k in range(len(self.infos)) if mask >> k & 1 and k not in members}

    def _candidates(self) -> "list[dict[str, Any]]":
        infos = self.infos
        self.reach, self.coreach = _reachability(infos)
        rk = _key(self.rank_bucket)
        costs = self.costs[rk]

        def weight(e: Edge) -> int:
            return costs[e.producer].writes.get(e.buffer, 0) + costs[e.consumer].reads.get(e.buffer, 0)

        lib = {k for k, info in enumerate(infos) if info.cls == "library"}
        inner = [e for e in self.edges if e.producer not in lib and e.consumer not in lib]
        anchored = [e for e in self.edges if (e.producer in lib) != (e.consumer in lib)]
        self.blocked: "list[dict[str, Any]]" = []
        for e in self.edges:
            if e.producer in lib and e.consumer in lib:  # (two library ops are never one kernel; listed, so that every intermediate is accounted for)
                self._block(e, set(), "library-library")
        # the largest intermediates first: an edge joins two groups only when one kernel of the union has no step between its members. An edge
        # refused is tried again after the others joined their groups (its middle step may have joined one), until nothing changes
        root = {k: k for k in range(len(infos))}
        members: "dict[int, set[int]]" = {}
        edges_of: "dict[int, list[Edge]]" = {}

        def find(x: int) -> int:
            while root[x] != x:
                root[x] = root[root[x]]
                x = root[x]
            return x

        def join(e: Edge) -> bool:
            a, b = find(e.producer), find(e.consumer)
            if a == b:
                edges_of[a].append(e)
                return True
            union = members.get(a, {a}) | members.get(b, {b})
            if self._middle(union):
                return False
            root[b] = a
            members[a] = union
            members.pop(b, None)
            edges_of[a] = edges_of.get(a, []) + edges_of.pop(b, []) + [e]
            return True

        pending = [e for e in sorted(inner, key=lambda e: (-weight(e), e.producer, e.consumer)) if not join(e)]
        changed = True
        while pending and changed:
            left = [e for e in pending if not join(e)]
            changed = len(left) < len(pending)
            pending = left
        for e in pending:
            union = members.get(find(e.producer), {e.producer}) | members.get(find(e.consumer), {e.consumer})
            self._block(e, self._middle(union))
        out = []
        for es in edges_of.values():
            steps = {x for e in es for x in (e.producer, e.consumer)}
            linear = all(sum(1 for e in es if e.producer == s) <= 1 and sum(1 for e in es if e.consumer == s) <= 1 for s in steps)
            out.append(self._candidate(es, "pair" if len(steps) == 2 else ("chain" if linear else "group")))
        for e in anchored:
            middle = self._middle({e.producer, e.consumer})
            if middle:
                self._block(e, middle)
                continue
            out.append(self._candidate([e], "epilogue" if e.producer in lib else "prologue"))
        out.sort(key=lambda c: (-c["saved_us"][rk], -c["saved_bytes"][rk], c["steps"][0]))
        for i, c in enumerate(out, 1):
            c["rank"] = i
        return out

    def _block(self, e: Edge, middle: "set[int]", reason: "str | None" = None) -> None:
        """An intermediate one kernel of its producer and consumer cannot save: another step sits between them (it reads what the producer made
        and the consumer reads what it makes; a library op in the middle makes it future work, any other step one kernel of all of them), or both
        are library ops."""
        infos = self.infos
        lib = sorted(k for k in middle if infos[k].cls == "library")
        reason = reason or ("library-in-the-middle" if lib else "step-in-the-middle")
        shown = (lib or sorted(middle))[:3]
        rk = _key(self.rank_bucket)
        saved = self.costs[rk][e.producer].writes.get(e.buffer, 0) + self.costs[rk][e.consumer].reads.get(e.buffer, 0)
        if reason == "library-library":
            why = f"steps {e.producer} and {e.consumer} are both library ops"
        else:
            why = f"step {e.consumer} also depends on step {e.producer} through " + ", ".join(f"step {k} ({infos[k].op or infos[k].kind})" for k in shown)
            why += f" and {len(middle) - len(shown)} more" if len(middle) > len(shown) else ""
        self.blocked.append({
            "buffer": e.buffer, "producer": e.producer, "consumer": e.consumer, "reason": reason, "middle": sorted(middle),
            "labels": [infos[e.producer].label, infos[e.consumer].label], "why": why,
            "action": BLOCKERS[reason][0], "phase": BLOCKERS[reason][1], "saved_bytes": saved,
        })

    def _candidate(self, es: "list[Edge]", shape: str) -> "dict[str, Any]":
        infos = self.infos
        steps = sorted({x for e in es for x in (e.producer, e.consumer)})
        blockers: "list[str]" = []
        details: "list[str]" = []
        for e in es:
            for bl, d in zip(e.blockers, e.detail):
                if bl not in blockers:
                    blockers.append(bl)
                    details.append(d)
        saved_bytes, saved_us = {}, {}
        removed = 0
        for b in self.analysed:
            bk = _key(b)
            costs = self.costs[bk]
            per: "dict[int, int]" = {}
            for e in es:
                per[e.producer] = per.get(e.producer, 0) + costs[e.producer].writes.get(e.buffer, 0)
                per[e.consumer] = per.get(e.consumer, 0) + costs[e.consumer].reads.get(e.buffer, 0)
            saved_bytes[bk] = sum(per.values())
            saved_us[bk], removed = self._saved_us(steps, per, bk)
        phase = max((BLOCKERS.get(b, ("", "future"))[1] for b in blockers), key=_phase_rank, default="1")
        # the blocker that decides when it can be one kernel: the one of the latest phase (the others are fixed sooner, or by the user)
        gate = max(blockers, key=lambda b: (_phase_rank(BLOCKERS.get(b, ("", "future"))[1]), -blockers.index(b)), default=None)
        return {
            "shape": shape,
            "steps": steps,
            "labels": [infos[k].label for k in steps],
            "classes": [infos[k].cls + (f":{infos[k].sub}" if infos[k].sub else "") for k in steps],
            "buffers": [e.buffer for e in es],
            "saved_bytes": saved_bytes,
            "saved_us": saved_us,
            "launches_removed": removed,
            "blockers": blockers,
            "why": details,
            "actions": [BLOCKERS.get(b, (b, ""))[0] for b in blockers],
            "phase": phase,
            "gate": gate,
        }

    # -- the document --------------------------------------------------------------------------------------------------------------------------

    def _doc(self) -> "dict[str, Any]":
        rk = _key(self.rank_bucket)
        costs = self.costs[rk]
        steps = []
        for info, c in zip(self.infos, costs):
            item: "dict[str, Any]" = {
                "index": info.index, "kind": info.kind, "entry": info.entry, "op": info.op, "class": info.cls, "sub": info.sub, "label": info.label,
                "bytes_read": sum(c.reads.values()), "bytes_written": sum(c.writes.values()), "flops": c.flops,
                "intensity": None if c.intensity is None or math.isinf(c.intensity) else round(c.intensity, 2), "bound": c.bound,
                "est_us": round(c.est_us, 4),
            }
            if self.measured is not None:
                item["us"] = self.measured.get(info.index)
                busy = (item["us"] or 0.0) - self.floor  # the bandwidth it moves its bytes at above the launch floor (none for a step at the floor)
                item["achieved_GBps"] = round(c.nbytes / busy / 1e3, 1) if busy >= 0.1 else None
                if info.cls == "opaque":
                    item["bound"] = "compute (measured)" if info.arith_bound else "memory (measured)"
            steps.append(item)
        by_class: "dict[str, dict[str, Any]]" = {}
        for info, c in zip(self.infos, costs):
            d = by_class.setdefault(info.cls, {"steps": 0, "est_us": 0.0, "us": 0.0 if self.measured is not None else None, "bytes": 0})
            d["steps"] += 1
            d["est_us"] += c.est_us
            d["bytes"] += c.nbytes
            if self.measured is not None:
                d["us"] += self.measured.get(info.index, 0.0)
        by_op: "dict[str, dict[str, Any]]" = {}
        for info, c in zip(self.infos, costs):
            o = by_op.setdefault(info.op or info.kind, {"op": info.op or info.kind, "class": info.cls, "steps": 0, "est_us": 0.0, "bytes": 0,
                                                       "us": 0.0 if self.measured is not None else None})
            o["steps"] += 1
            o["est_us"] += c.est_us
            o["bytes"] += c.nbytes
            if self.measured is not None:
                o["us"] += self.measured.get(info.index, 0.0)
        for o in by_op.values():
            o["est_us"] = round(o["est_us"], 3)
            if o["us"] is not None:
                o["us"] = round(o["us"], 3)
        total_est = sum(d["est_us"] for d in by_class.values()) or 1.0
        total_us = sum(d["us"] for d in by_class.values()) if self.measured is not None else None
        for d in by_class.values():
            d["est_us"] = round(d["est_us"], 3)
            d["est_share"] = round(d["est_us"] / total_est, 4)
            if self.measured is not None:
                d["us"] = round(d["us"], 3)
                d["share"] = round(d["us"] / (total_us or 1.0), 4)
        patterns: "dict[str, dict[str, Any]]" = {}
        for c in self.candidates:  # each candidate counted once, under the blocker that decides when it can be one kernel
            bl = c["gate"]
            if bl is None:
                continue
            p = patterns.setdefault(bl, {"blocker": bl, "action": BLOCKERS.get(bl, (bl, ""))[0], "phase": BLOCKERS.get(bl, ("", "?"))[1],
                                         "candidates": 0, "saved_us": 0.0, "saved_bytes": 0, "also": {}})
            p["candidates"] += 1
            p["saved_us"] += c["saved_us"][rk]
            p["saved_bytes"] += c["saved_bytes"][rk]
            for other in c["blockers"]:
                if other != bl:
                    p["also"][other] = p["also"].get(other, 0) + 1
        for p in patterns.values():
            p["saved_us"] = round(p["saved_us"], 3)
        # the same candidate repeated (a layer of a model): the ops of its steps, in order, with what all of them would save
        recurring: "dict[str, dict[str, Any]]" = {}
        for c in self.candidates:
            ops = " -> ".join(self.infos[k].op or self.infos[k].kind for k in c["steps"])
            r = recurring.setdefault(f"{c['shape']}: {ops}", {"pattern": ops, "shape": c["shape"], "gate": c["gate"], "phase": c["phase"], "count": 0,
                                                              "saved_us": 0.0, "saved_bytes": 0, "ranks": []})
            r["count"] += 1
            r["saved_us"] += c["saved_us"][rk]
            r["saved_bytes"] += c["saved_bytes"][rk]
            r["ranks"].append(c["rank"])
        for r in recurring.values():
            r["saved_us"] = round(r["saved_us"], 3)
        unfused = self._unfused()
        m = self.machine
        return {
            "schema": ADVICE_SCHEMA,
            "graph": self.plan.get("graph"),
            "arch": self.plan.get("arch"),
            "plan_digest": self.plan.get("plan_digest"),
            "bucket": dict(self.rank_bucket.get("values") or {}),
            "buckets": [dict(b.get("values") or {}) for b in self.analysed],
            "machine": {"arch": m.arch, "example": m.example, "bandwidth_GBps": m.bandwidth, "tensor16_TFLOPS": round(m.tensor16, 1), "sms": m.sms, "balance": round(m.balance, 1)},
            "timing": None if self.timing is None else {"method": self.timing.get("method"), "reps": self.timing.get("reps"), "floor_us": round(self.floor, 3)},
            "estimate": "measured" if self.measured is not None else "bytes / bandwidth",
            "steps": steps,
            "by_class": dict(sorted(by_class.items(), key=lambda kv: CLASSES.index(kv[0]) if kv[0] in CLASSES else 99)),
            "by_op": sorted(by_op.values(), key=lambda o: (-(o["us"] if o["us"] is not None else o["est_us"]), o["op"])),
            "total_us": None if total_us is None else round(total_us, 3),
            "total_est_us": round(total_est, 3),
            "candidates": [{**c, "saved_us": {k: round(v, 3) for k, v in c["saved_us"].items()}} for c in self.candidates],
            "patterns": sorted(patterns.values(), key=lambda p: (-p["saved_us"], -p["saved_bytes"])),
            "recurring": sorted(recurring.values(), key=lambda r: (-r["saved_us"], -r["saved_bytes"])),
            "blocked": sorted(self.blocked, key=lambda b: (-b["saved_bytes"], b["producer"])),
            "fan_outs": self.fanouts,
            "kept": self.kept,
            "unfused_arithmetic": unfused,
        }

    def _unfused(self) -> "list[dict[str, Any]]":
        """The boundaries of the graph body's regions (from the trace or the graph), each with the steps it maps to."""
        ex = self.ex
        out = []
        steps_of: "dict[str, list[int]]" = {}
        for info in self.infos:
            if info.op:
                steps_of.setdefault(info.op.split("__ua", 1)[0], []).append(info.index)
        if ex.boundaries is not None:
            for b in ex.boundaries:
                fa, tb = b.get("from") or {}, b.get("to") or {}
                out.append({**b, "from_steps": steps_of.get(fa.get("kernel") or fa.get("op") or "", []), "to_steps": steps_of.get(tb.get("kernel") or tb.get("op") or "", [])})
            return out
        done: "set[int]" = set()
        for name, r in ex.regions.items():  # no trace: what the fusion report says alone
            if r.get("index") is None or id(r) in done:
                continue
            done.add(id(r))
            solo = _solo_reason(r)
            if solo is not None:
                reason, what = solo
                out.append({"reason": reason, "region": r["index"], "kernel": name, "detail": what, "fix": BLOCKERS[reason][0], "phase": BLOCKERS[reason][1],
                            "steps": steps_of.get(name, [])})
            for i in r.get("inputs") or []:
                if i.get("read") == "materialised":
                    out.append({"reason": "materialised", "region": r["index"], "kernel": name, "detail": f"{i.get('value')} ({i.get('type')}) is copied first",
                                "fix": BLOCKERS["materialised"][0], "phase": "2", "steps": steps_of.get(name, [])})
        return out

    # -- text ----------------------------------------------------------------------------------------------------------------------------------

    def text(self, top: int = 20) -> str:
        d = self.doc
        rk = _key(self.rank_bucket)
        m = d["machine"]
        lines = []
        nb = len(self.analysed)
        which = f"bucket {d['bucket']}" + (f" (the largest of the {nb} analysed; --bucket to choose)" if nb > 1 else "")
        lines.append(f"tilefish advise: the plan of '{d['graph']}' on {d['arch'] or 'no architecture'}, {len(self.infos)} steps, ranked at {which}")
        est = "measured per step, net of the events around it (" + f"{d['timing']['method']}, {d['timing']['reps']} reps; launch floor {d['timing']['floor_us']} us)" if self.measured is not None else "estimated (bytes / bandwidth; --timing FILE for measured times)"
        lines.append(f"  machine {m['arch']} ({m['example']}): {m['bandwidth_GBps']:.0f} GB/s, {m['tensor16_TFLOPS']:.0f} TFLOPS f16/bf16, balance {m['balance']} FLOP/B; times {est}")
        lines.append("")
        lines.append("time by class" + (" (measured | roofline estimate)" if self.measured is not None else " (roofline estimate)") + ":")
        for cls, c in d["by_class"].items():
            meas = f"{c['us']:10.1f} us {100 * c['share']:5.1f}%  | " if self.measured is not None else ""
            lines.append(f"  {cls:<12}{c['steps']:5d} steps  {meas}{c['est_us']:10.1f} us {100 * c['est_share']:5.1f}%  {_mb(c['bytes'])}")
        if self.measured is not None:
            lines.append(f"  {'total':<12}{len(self.infos):5d} steps  {d['total_us']:10.1f} us")
        lines.append("")
        lines.append("time by op (the ten longest):")
        for o in d["by_op"][:10]:
            t = o["us"] if o["us"] is not None else o["est_us"]
            lines.append(f"  {t:10.1f} us  {o['steps']:4d} steps  {o['class']:<12} {o['op']}")
        lines.append("")
        cands = self.candidates
        lines.append(f"fusion candidates ({len(cands)}), ranked by the time they would save at {d['bucket']}:")
        for c in cands[:top]:
            us = c["saved_us"][rk]
            launches = f", {c['launches_removed']} launch{'es' if c['launches_removed'] != 1 else ''} fewer" if c["launches_removed"] else ""
            sep = " -> " if c["shape"] in ("pair", "chain", "epilogue", "prologue") else ", "
            ids = c["steps"] if len(c["steps"]) <= 8 else [*c["steps"][:5], "...", *c["steps"][-2:]]
            lines.append(f" #{c['rank']:<3} steps {sep.join(str(s) for s in ids)} ({c['shape']}): saves {_mb(c['saved_bytes'][rk])}, ~{us:.2f} us{launches}; phase {c['phase']}")
            shown = list(zip(c["steps"], c["labels"]))
            for k, label in shown if len(shown) <= 8 else shown[:5]:
                lines.append(f"        step {k}: {label}")
            if len(shown) > 8:
                lines.append(f"        ... {len(shown) - 5} more steps")
            for w in c["why"]:
                lines.append(f"        why not today: {w}")
            for a in c["actions"]:
                lines.append(f"        do: {a}")
        if len(cands) > top:
            lines.append(f"  ... {len(cands) - top} more (--json for all)")
        if d["patterns"]:
            lines.append("")
            lines.append("what would help most (each candidate under the blocker of the latest phase it needs removed; candidates overlap: upper bounds):")
            for p in d["patterns"]:
                also = "; also needs " + ", ".join(f"{k} ({n})" for k, n in sorted(p["also"].items(), key=lambda kv: -kv[1])) if p["also"] else ""
                lines.append(f"  {p['saved_us']:9.2f} us  {_mb(p['saved_bytes']):>10}  {p['candidates']:4d} candidates  phase {p['phase']:<6} {p['blocker']}{also}")
                lines.append(f"        {p['action']}")
        rec = [r for r in d["recurring"] if r["count"] > 1]
        if rec:
            lines.append("")
            lines.append("recurring candidates (the same ops in the same order, summed: upper bounds):")
            for r in rec[:top]:
                lines.append(f"  {r['saved_us']:9.2f} us  {_mb(r['saved_bytes']):>10}  {r['count']:4d}x  phase {r['phase']:<6} {r['shape']}: {r['pattern']}  ({r['gate']})")
        if d["unfused_arithmetic"]:
            lines.append("")
            lines.append("graph-body arithmetic that was not fused:")
            for u in d["unfused_arithmetic"][:top]:
                if "from" in u:
                    fa, tb = u["from"], u["to"]
                    lines.append(f"  {_end(fa)} -> {_end(tb)}: {u['reason']}: {u['detail']}")
                else:
                    lines.append(f"  region {u.get('region')} ({u.get('kernel')}): {u['reason']}: {u['detail']}")
                lines.append(f"      fix: {u['fix']} (phase {u['phase']})")
            if len(d["unfused_arithmetic"]) > top:
                lines.append(f"  ... {len(d['unfused_arithmetic']) - top} more (--json for all)")
        if self.blocked:
            lines.append("")
            by: "dict[str, list[dict[str, Any]]]" = {}
            for b in d["blocked"]:
                by.setdefault(b["reason"], []).append(b)
            lines.append("intermediates one kernel of their producer and consumer alone cannot save (a step sits between them):")
            for reason, bs in sorted(by.items(), key=lambda kv: -sum(b["saved_bytes"] for b in kv[1])):
                lines.append(f"  {len(bs):4d}  {_mb(sum(b['saved_bytes'] for b in bs)):>10}  {reason}: {BLOCKERS[reason][0]} (phase {BLOCKERS[reason][1]})")
                for b in bs[:3]:
                    lines.append(f"        {b['buffer']}: steps {b['producer']} -> {b['consumer']}: {b['why']}")
        if self.fanouts or self.kept:
            lines.append("")
            why: "dict[str, int]" = {}
            for k in self.kept:
                why[k["why"]] = why.get(k["why"], 0) + 1
            parts = [f"{n} {w}" for w, n in sorted(why.items(), key=lambda kv: -kv[1])]
            if self.fanouts:
                parts.append(f"{len(self.fanouts)} read by several steps (one region of all of them, or recomputation, which version 1 does not do)")
            lines.append("not candidates: buffers " + "; ".join(parts))
        return "\n".join(lines) + "\n"


def _end(r: "Mapping[str, Any]") -> str:
    where = f" ({r['loc']})" if r.get("loc") else ""
    if r.get("region") is not None:
        return f"region {r['region']} #{r.get('node')} {r.get('op')}{where}"
    if r.get("call"):
        return f"call #{r.get('node')} {r.get('op')}{where}"
    return f"#{r.get('node')} {r.get('op')}{where}"


def _mb(n: float) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KiB"
    return f"{int(n)} B"


# -- the entry point ------------------------------------------------------------------------------------------------------------------------------


def advise(
    directory: "str | pathlib.Path",
    graph: "str | None" = None,
    *,
    bucket: "str | Mapping[str, int] | None" = None,
    timing: "str | pathlib.Path | Mapping[str, Any] | None" = None,
    machine: "Machine | str | None" = None,
    bandwidth: "float | None" = None,
    keep: "Iterable[str]" = (),
    gop: Any = None,
) -> Advice:
    """The fusion advice for the plan of ``graph`` in the export ``directory`` (see the module's doc and ``docs/advise.md``).

    ``bucket``: one bucket (``"M=64"`` or ``{"M": 64}``), else every bucket, ranked at the one that moves the most bytes; ``timing``: the file (or the
    document) :func:`tilefish.profile_plan` wrote for one bucket, whose times replace the estimates; ``machine``: an architecture of :data:`MACHINES` or a
    :class:`Machine` (default: the plan's); ``bandwidth``: GB/s to estimate with (a partition's, measured); ``keep``: buffers the caller taps, which stay
    stored; ``gop``: the graph itself, when the export has no trace, so that the boundaries of its fused regions are known."""
    from .explain import Trace

    d = pathlib.Path(directory)
    doc = _plan_doc(d, graph)
    name = doc["graph"]
    manifest = _read_json(d / "manifest.json", "the manifest")
    from ._manifest import manifest_digest

    if not isinstance(manifest, dict) or (doc.get("manifest_digest") and manifest_digest(manifest) != doc["manifest_digest"]):
        raise TilefishAdviseError(f"the plan of '{name}' in {str(d)!r} was made against another manifest than the manifest.json next to it")
    report_path = d / f"{name}.fusion.json"
    report = _read_json(report_path, "the fusion report") if report_path.is_file() else None
    trace = Trace.load(d)
    gdoc = trace.graph(name, doc.get("plan_digest")) if trace is not None else None
    bounds = None
    if gop is not None:
        from . import _fusion

        if getattr(gop, "name", None) != name:
            raise TilefishAdviseError(f"gop= is the graph '{getattr(gop, 'name', gop)}', not '{name}'")
        bounds = _fusion.boundaries(gop) if _fusion.has_arithmetic(gop) else []
        if _fusion.has_arithmetic(gop):
            ops = {str(e.get("op")).split("__ua", 1)[0] for e in manifest.get("kernels") or []}  # (an unaligned specialisation is its region's)
            missing = sorted({r.kernel.name for r in _fusion.fused(gop).regions} - ops)
            if missing:
                raise TilefishAdviseError(f"gop= is not the graph '{name}' was exported from: its fused regions {missing[:3]} are not in the manifest")
            if report is None:
                report = _fusion.report(gop)
    if isinstance(timing, (str, pathlib.Path)):
        timing = _read_json(pathlib.Path(timing), "the timing file")
    return analyse(doc, manifest, report=report, gdoc=gdoc, boundaries=bounds, timing=timing, bucket=bucket, machine=machine, bandwidth=bandwidth,
                   keep=keep, trace=trace)


__all__ = ["ADVICE_SCHEMA", "Advice", "BLOCKERS", "MACHINES", "Machine", "TIMING_SCHEMA", "TilefishAdviseError", "advise", "analyse", "graphs_of"]
