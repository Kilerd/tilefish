"""The records of instrumented launches, collected and explained (``docs/instrument.md``): :class:`Probes` collects them from ``tilefish.run``,
:class:`ProbeTrace` holds them (and reads and writes ``tilefish.probe-trace.v1``, which the Rust host writes too), and renders phase durations
with their distributions, timelines per SM, CTA and warp role (text), and a Chrome trace that Perfetto opens.

**Phases.** The records of a warp are in the order it made them. The time between two consecutive records is the phase ``"a -> b"`` (``a`` and
``b`` the probes' names; ``@begin``, ``@end`` and the kernel's own). Its duration is measured twice: in nanoseconds of ``%globaltimer`` (one
clock for the whole GPU, in steps of its resolution: 32 ns on a B300) and in cycles of the SM's ``%clock``, which is fine-grained; the
durations reported are the cycles converted at the clock rate the launch measured (cycles over nanoseconds of its warps' spans), and the start of
a phase is its ``%globaltimer``, so that the warps of every SM are on one time axis. ``@calib`` follows ``@begin`` immediately: the cycles
between them are what a probe costs that warp, the overhead the report gives (its records times that cost), and the first phase of a warp runs
from ``@calib``.

Nothing here needs torch, Triton or a GPU but :meth:`Probes.collect` (which reads a buffer back from the device).
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import _probe
from ._probe import BEGIN, CALIB, IMPLICIT, WarpRecords
from .errors import TilefishError

TRACE_SCHEMA = "tilefish.probe-trace.v1"
_CLOCK_SPAN_NS = 2000  # a warp's span this long or longer measures the clock rate


class TilefishProbeError(TilefishError, ValueError):
    """A probe trace that is not one (another schema, a malformed launch), or a recorder used where it cannot be."""


def _percentile(values: "Sequence[float]", q: float) -> float:
    """The ``q`` quantile (0 to 1) of sorted ``values``, linearly interpolated."""
    if not values:
        return float("nan")
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


@dataclass(frozen=True)
class Phase:
    """The time between two consecutive records of one warp."""

    cta: int
    warp: int
    sm: int
    role: str
    start: str  # the name of the probe that begins it
    end: str
    start_ns: float  # its start, from the launch's first record (globaltimer)
    ns: float  # its duration: its cycles at the launch's clock rate (or its globaltimer difference when that is not known)
    cycles: int
    timer_ns: int  # its duration in globaltimer nanoseconds

    @property
    def name(self) -> str:
        return f"{self.start} -> {self.end}"


@dataclass
class Launch:
    """The records of one launch of an instrumented binary.

    ``grid`` and ``block`` are the CUDA grid and block of the launch (a cluster's CTAs counted, a persistent grid as launched); ``probes`` the
    manifest's table of the binary's probes (``instrument.probes``: ids, names, where each is written and the partition it runs in); ``warps`` what
    each warp recorded; ``missing`` the ``(cta, warp)`` slots without an intact header. ``step``, ``graph`` and ``bucket`` say which step of which
    plan it was, for a plan's run."""

    stem: str
    arch: str
    grid: "tuple[int, int, int]"
    block: "tuple[int, int, int]"
    capacity: int
    probes: "list[dict[str, Any]]"
    warps: "list[WarpRecords]"
    missing: "list[tuple[int, int]]" = field(default_factory=list)
    op: str = ""
    graph: "str | None" = None
    step: "int | None" = None
    bucket: "dict[str, int] | None" = None
    manifest_digest: "str | None" = None

    # -- names and roles -------------------------------------------------------------------------------------------------------------------

    def names(self) -> "dict[int, str]":
        """The name of every probe id of the binary (``@begin``, ``@end``, ``@calib`` and the kernel's)."""
        out = dict(IMPLICIT)
        out.update({int(p["id"]): str(p["name"]) for p in self.probes})
        return out

    def sites(self) -> "dict[str, list[dict[str, Any]]]":
        """Where each of the kernel's probes is written (its ``sites``), by name."""
        return {str(p["name"]): list(p.get("sites", [])) for p in self.probes}

    def role_of(self, w: WarpRecords) -> str:
        """The partition a warp ran (``default``, ``worker 0``, ...: what the sites of the probes it recorded say), ``""`` for a kernel that is not
        warp-specialised, and ``"-"`` for a warp that recorded none of the kernel's probes (a partition without probes, or a warp Triton adds to
        round a warp-specialised kernel up to whole warp groups)."""
        by_id = {int(p["id"]): {s.get("role", "") for s in p.get("sites", [])} for p in self.probes}
        roles: "set[str]" = set()
        for pid, _, _, _ in w.records:
            roles |= by_id.get(pid, set())
        if not roles:
            return "-" if self.probes else ""
        return "/".join(sorted(roles))

    # -- time ------------------------------------------------------------------------------------------------------------------------------

    @property
    def t0(self) -> int:
        """The earliest globaltimer of the launch's records (0 when it has none)."""
        return min((r[2] for w in self.warps for r in w.records), default=0)

    def clock_ghz(self) -> "float | None":
        """The SM clock rate the launch measured (GHz): the median, over the warps that recorded for 2 us or more, of their cycles over their
        globaltimer nanoseconds; ``None`` when no warp did."""
        rates = []
        for w in self.warps:
            if len(w.records) < 2:
                continue
            first, last = w.records[0], w.records[-1]
            ns = last[2] - first[2]
            cycles = (last[3] - first[3]) & 0xFFFFFFFF
            if ns >= _CLOCK_SPAN_NS and cycles:
                rates.append(cycles / ns)
        return statistics.median(rates) if rates else None

    def probe_cycles(self) -> "list[int]":
        """What one probe costs a warp, in cycles, per warp 0 of each CTA (the warps that calibrate): from its first ``@calib`` to its second (the
        same instructions back to back, the second with its code in the instruction cache). It is the latency of a probe in the warp; where the
        warp would wait anyway (an ``mbarrier`` wait) a probe costs less."""
        out = []
        for w in self.warps:
            r = w.records
            if len(r) >= 3 and r[0][0] == BEGIN and r[1][0] == CALIB and r[2][0] == CALIB:
                out.append((r[2][3] - r[1][3]) & 0xFFFFFFFF)
        return out

    @staticmethod
    def _work(records: "Sequence[tuple[int, int, int, int]]") -> "list[tuple[int, int, int, int]]":
        """A warp's records from where its work starts: ``@begin`` and the ``@calib`` records become one ``@begin`` at the last ``@calib``."""
        recs = list(records)
        k = 1
        while k < len(recs) and recs[k][0] == CALIB:
            k += 1
        if recs and recs[0][0] == BEGIN and k > 1:
            return [(BEGIN,) + recs[k - 1][1:]] + recs[k:]
        return recs

    def phases(self) -> "list[Phase]":
        """Every phase of every warp, in the order of the warps and of their records (the first from ``@calib``, labelled ``@begin``)."""
        names = self.names()
        t0 = self.t0
        ghz = self.clock_ghz()
        out: "list[Phase]" = []
        for w in self.warps:
            role = self.role_of(w)
            recs = self._work(w.records)  # (the warp starts its work after @calib)
            for a, b in zip(recs, recs[1:]):
                cycles = (b[3] - a[3]) & 0xFFFFFFFF
                timer = b[2] - a[2]
                out.append(
                    Phase(
                        cta=w.cta,
                        warp=w.warp,
                        sm=a[1],
                        role=role,
                        start=names.get(a[0], f"#{a[0]}"),
                        end=names.get(b[0], f"#{b[0]}"),
                        start_ns=float(a[2] - t0),
                        ns=cycles / ghz if ghz else float(timer),
                        cycles=cycles,
                        timer_ns=timer,
                    )
                )
        return out

    def span_ns(self) -> int:
        """From the launch's first record to its last (globaltimer)."""
        times = [r[2] for w in self.warps for r in w.records]
        return (max(times) - min(times)) if times else 0

    # -- what it says ----------------------------------------------------------------------------------------------------------------------

    def summary(self) -> "dict[str, Any]":
        """The launch in numbers: its span, CTAs, SMs and warps, the records kept and dropped, the clock rate, what a probe costs and the overhead
        of the records (their count times that cost, over the warps' time), and per phase name (and role) its count and duration distribution."""
        ghz = self.clock_ghz()
        cost = sorted(self.probe_cycles())
        cost_cycles = statistics.median(cost) if cost else None
        records = sum(len(w.records) for w in self.warps)
        dropped = sum(w.dropped for w in self.warps)
        warp_ns = []
        overheads = []
        idle = 0
        for w in self.warps:
            if self.role_of(w) == "-":  # a warp that ran none of the kernel's probes: a partition without them, or a warp Triton adds
                idle += 1
                continue
            work = self._work(w.records)
            if len(work) >= 2:
                ns = work[-1][2] - work[0][2]
                cycles = (work[-1][3] - work[0][3]) & 0xFFFFFFFF
                warp_ns.append(ns)
                if cost_cycles and cycles:  # the probes after @calib (each costs the warp what the second @calib did)
                    overheads.append(min(1.0, cost_cycles * (len(work) - 1 + w.dropped) / cycles))
        groups: "dict[tuple[str, str], list[Phase]]" = {}
        for ph in self.phases():
            if ph.role != "-":
                groups.setdefault((ph.name, ph.role), []).append(ph)
        total_ns = sum(ph.ns for phs in groups.values() for ph in phs) or 1.0
        rows = []
        for (name, role), phs in groups.items():
            ns = sorted(ph.ns for ph in phs)
            cyc = sorted(ph.cycles for ph in phs)
            rows.append(
                {
                    "phase": name,
                    "role": role,
                    "count": len(phs),
                    "ns": {
                        "min": ns[0],
                        "p10": _percentile(ns, 0.1),
                        "median": _percentile(ns, 0.5),
                        "mean": sum(ns) / len(ns),
                        "p90": _percentile(ns, 0.9),
                        "max": ns[-1],
                        "total": sum(ns),
                    },
                    "cycles_median": _percentile(cyc, 0.5),
                    "share": sum(ns) / total_ns,
                }
            )
        rows.sort(key=lambda r: -r["ns"]["total"])
        ctas = sorted({w.cta for w in self.warps})
        sms = sorted({r[1] for w in self.warps for r in w.records})
        return {
            "stem": self.stem,
            "arch": self.arch,
            "graph": self.graph,
            "step": self.step,
            "grid": list(self.grid),
            "block": list(self.block),
            "span_ns": self.span_ns(),
            "ctas": len(ctas),
            "sms": len(sms),
            "warps": len(self.warps),
            "idle_warps": idle,
            "missing": len(self.missing),
            "records": records,
            "dropped": dropped,
            "capacity": self.capacity,
            "clock_ghz": ghz,
            "probe_cycles": cost_cycles,
            "probe_ns": (cost_cycles / ghz) if (cost_cycles and ghz) else None,
            "overhead": statistics.median(overheads) if overheads else None,
            "warp_span_ns_median": statistics.median(warp_ns) if warp_ns else None,
            "phases": rows,
        }

    def ctas(self) -> "list[dict[str, Any]]":
        """Per CTA: its SM, its first and last record (from the launch's first record, ns) and its warps with their roles."""
        t0 = self.t0
        out: "dict[int, dict[str, Any]]" = {}
        for w in self.warps:
            if not w.records:
                continue
            c = out.setdefault(w.cta, {"cta": w.cta, "sm": w.records[0][1], "start": math.inf, "end": -math.inf, "warps": []})
            c["start"] = min(c["start"], w.records[0][2] - t0)
            c["end"] = max(c["end"], w.records[-1][2] - t0)
            c["warps"].append((w.warp, self.role_of(w), w.count, w.dropped))
        return [out[k] for k in sorted(out)]

    def to_json(self) -> "dict[str, Any]":
        """The launch as ``tilefish.probe-trace.v1`` writes it (times from ``t0``, a decimal string: a globaltimer does not fit a JSON number)."""
        t0 = self.t0
        return {
            "stem": self.stem,
            "arch": self.arch,
            "op": self.op,
            "graph": self.graph,
            "step": self.step,
            "bucket": self.bucket,
            "manifest_digest": self.manifest_digest,
            "grid": list(self.grid),
            "block": list(self.block),
            "capacity": self.capacity,
            "probes": self.probes,
            "t0": str(t0),
            "warps": [[w.cta, w.warp, w.count, [[r[0], r[1], r[2] - t0, r[3]] for r in w.records]] for w in self.warps],
            "missing": [list(m) for m in self.missing],
        }

    @classmethod
    def from_json(cls, doc: Any, where: str = "a launch") -> "Launch":
        """The inverse of :meth:`to_json`. Raises :class:`TilefishProbeError` for a launch that is not one."""
        try:
            t0 = int(doc["t0"])
            warps = [
                WarpRecords(int(c), int(w), int(n), tuple((int(r[0]), int(r[1]), t0 + int(r[2]), int(r[3])) for r in recs))
                for c, w, n, recs in doc["warps"]
            ]
            grid = tuple(int(g) for g in doc["grid"])
            block = tuple(int(b) for b in doc["block"])
            if len(grid) != 3 or len(block) != 3:
                raise ValueError("grid and block have 3 axes")
            probes = list(doc["probes"])
            problem = _probe.check_field(where, {"format": _probe.FORMAT, "capacity": int(doc["capacity"]), "probes": probes})
            if problem is not None:
                raise ValueError(problem)
            return cls(
                stem=str(doc["stem"]),
                arch=str(doc["arch"]),
                grid=grid,  # type: ignore[arg-type]
                block=block,  # type: ignore[arg-type]
                capacity=int(doc["capacity"]),
                probes=probes,
                warps=warps,
                missing=[(int(a), int(b)) for a, b in doc.get("missing", [])],
                op=str(doc.get("op") or ""),
                graph=doc.get("graph"),
                step=doc.get("step"),
                bucket=doc.get("bucket"),
                manifest_digest=doc.get("manifest_digest"),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise TilefishProbeError(f"{where} of the probe trace is not a launch of {TRACE_SCHEMA}: {type(e).__name__}: {e}") from e


def _us(ns: float) -> str:
    return f"{ns / 1000:.3f}"


def _fmt_ns(ns: "float | None") -> str:
    if ns is None or (isinstance(ns, float) and math.isnan(ns)):
        return "-"
    return f"{ns / 1000:.2f} us" if ns >= 10000 else f"{ns:.0f} ns"


@dataclass
class ProbeTrace:
    """The records of one or more instrumented launches (a run, a plan's steps), with the views of ``tilefish-explain --probes``."""

    launches: "list[Launch]" = field(default_factory=list)

    # -- files ---------------------------------------------------------------------------------------------------------------------------------

    def to_json(self) -> "dict[str, Any]":
        return {"schema": TRACE_SCHEMA, "launches": [launch.to_json() for launch in self.launches]}

    def save(self, path: "str | pathlib.Path") -> pathlib.Path:
        """Write the trace as ``tilefish.probe-trace.v1`` JSON (what ``tilefish-explain DIR --probes FILE`` reads)."""
        path = pathlib.Path(path)
        path.write_text(json.dumps(self.to_json(), separators=(",", ":")) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: "str | pathlib.Path") -> "ProbeTrace":
        """Read a ``tilefish.probe-trace.v1`` file (Python's :meth:`save`, the Rust host's ``ProbeTrace::save``)."""
        try:
            doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise TilefishProbeError(f"cannot read the probe trace {str(path)!r}: {e}") from e
        return cls.from_json(doc)

    @classmethod
    def from_json(cls, doc: Any) -> "ProbeTrace":
        if not isinstance(doc, dict) or doc.get("schema") != TRACE_SCHEMA or not isinstance(doc.get("launches"), list):
            raise TilefishProbeError(f"not a probe trace of {TRACE_SCHEMA}")
        return cls([Launch.from_json(d, f"launch {k}") for k, d in enumerate(doc["launches"])])

    # -- views ---------------------------------------------------------------------------------------------------------------------------------

    def summaries(self) -> "list[dict[str, Any]]":
        return [launch.summary() for launch in self.launches]

    def summary_text(self, sources: "Any" = None, top: int = 40) -> str:
        """Per launch: its span, the clock rate, what a probe costs and the overhead, then the phases by total time, with their distribution."""
        out: "list[str]" = []
        for k, launch in enumerate(self.launches):
            s = launch.summary()
            where = f" (step {launch.step} of the plan of '{launch.graph}')" if launch.graph is not None and launch.step is not None else ""
            out.append(f"launch {k}: {launch.stem} on {launch.arch}{where}, grid {tuple(launch.grid)}, block {tuple(launch.block)}")
            ghz = f"{s['clock_ghz']:.3f} GHz" if s["clock_ghz"] else "not measured (no warp recorded for 2 us)"
            cost = f"{s['probe_cycles']:.0f} cycles" + (f" ({s['probe_ns']:.0f} ns)" if s["probe_ns"] else "") if s["probe_cycles"] else "-"
            over = f"{100 * s['overhead']:.1f}%" if s["overhead"] is not None else "-"
            out.append(
                f"  span {_fmt_ns(s['span_ns'])} (first record to last), {s['ctas']} CTAs on {s['sms']} SMs, {s['warps']} warps"
                + (f" ({s['idle_warps']} ran none of the kernel's probes: not in the phases)" if s["idle_warps"] else "")
                + (f", {s['missing']} warp slots without records" if s["missing"] else "")
            )
            out.append(
                f"  records {s['records']} kept, {s['dropped']} dropped (capacity {s['capacity']} per warp); SM clock {ghz}; a probe costs a warp "
                f"{cost}; the probes' share of a warp's time at most (records times that cost, median over the warps): {over}"
            )
            if s["dropped"]:
                out.append(f"  ! {s['dropped']} records were dropped: export with tilefish.instrument(capacity=...) larger to keep them")
            out.append("  phase (role)                                   count     median       mean        p10        p90        max   share")
            for row in s["phases"][:top]:
                ns = row["ns"]
                label = row["phase"] + (f" ({row['role']})" if row["role"] else "")
                out.append(
                    f"  {label[:46]:46s} {row['count']:6d} {_fmt_ns(ns['median']):>10s} {_fmt_ns(ns['mean']):>10s} {_fmt_ns(ns['p10']):>10s} "
                    f"{_fmt_ns(ns['p90']):>10s} {_fmt_ns(ns['max']):>10s} {100 * row['share']:6.1f}%"
                )
            if len(s["phases"]) > top:
                out.append(f"  ... ({len(s['phases']) - top} more phases)")
            sites = launch.sites()
            if sites:
                out.append("  probes:")
                for name, ss in sites.items():
                    for site in ss or [{"loc": "", "role": "", "function": ""}]:
                        text = sources(site["loc"]) if (sources is not None and site.get("loc")) else None
                        role = f" [{site['role']}]" if site.get("role") else ""
                        out.append(f"    {name:24s} {site.get('loc') or '?'}{role}" + (f"   {text}" if text else ""))
            out.append("")
        return "\n".join(out)

    def timeline_text(self, by: str = "sm", limit: int = 16) -> str:
        """The launches over time: per SM (``by="sm"``) its CTAs with their start and end; per CTA (``"cta"``) each warp's phases; per role
        (``"role"``) the phases of each warp role, one warp per CTA. Times in microseconds from the launch's first record; ``limit`` SMs or CTAs."""
        out: "list[str]" = []
        for k, launch in enumerate(self.launches):
            out.append(f"launch {k}: {launch.stem} on {launch.arch}, span {_fmt_ns(launch.span_ns())} (times in us from its first record)")
            ctas = launch.ctas()
            if by == "sm":
                per_sm: "dict[int, list[dict[str, Any]]]" = {}
                for c in ctas:
                    per_sm.setdefault(c["sm"], []).append(c)
                for sm in sorted(per_sm)[:limit]:
                    runs = ", ".join(f"CTA {c['cta']} {_us(c['start'])}-{_us(c['end'])}" for c in sorted(per_sm[sm], key=lambda c: c["start"]))
                    out.append(f"  SM {sm:3d}: {runs}")
                if len(per_sm) > limit:
                    out.append(f"  ... ({len(per_sm) - limit} more SMs)")
            else:
                phases = launch.phases()
                by_warp: "dict[tuple[int, int], list[Phase]]" = {}
                for ph in phases:
                    by_warp.setdefault((ph.cta, ph.warp), []).append(ph)
                shown_ctas = sorted({c for c, _ in by_warp})[:limit]
                for cta in shown_ctas:
                    info = next(c for c in ctas if c["cta"] == cta)
                    out.append(f"  CTA {cta} on SM {info['sm']}: {_us(info['start'])}-{_us(info['end'])}")
                    seen_roles: "set[str]" = set()
                    for (c, w), phs in sorted(by_warp.items()):
                        if c != cta:
                            continue
                        role = phs[0].role
                        if by == "role" and role in seen_roles:
                            continue
                        seen_roles.add(role)
                        steps = " | ".join(f"{p.end} +{_fmt_ns(p.ns)}" for p in phs[:12]) + (" | ..." if len(phs) > 12 else "")
                        out.append(f"    warp {w:2d}{(' ' + role) if role else '':12s} {_us(phs[0].start_ns)}: {steps}")
                if len({c for c, _ in by_warp}) > limit:
                    out.append(f"  ... ({len({c for c, _ in by_warp}) - limit} more CTAs)")
            out.append("")
        return "\n".join(out)

    def chrome(self) -> "dict[str, Any]":
        """A Chrome trace (``chrome://tracing``, ``ui.perfetto.dev``): a process per launch and SM, a thread per CTA and warp, an event per phase
        (its start on the launch's globaltimer axis, its duration from cycles), with its cycles and the role in its arguments."""
        events: "list[dict[str, Any]]" = []
        offset = 0.0
        base = min((launch.t0 for launch in self.launches if launch.warps), default=0)
        for k, launch in enumerate(self.launches):
            label = f"{k}: {launch.stem}" + (f" (step {launch.step})" if launch.step is not None else "")
            shift = (launch.t0 - base) / 1000.0 if base else offset
            pids: "dict[int, int]" = {}
            tids: "set[tuple[int, int]]" = set()
            for ph in launch.phases():
                pid = pids.setdefault(ph.sm, k * 100000 + ph.sm)
                tid = ph.cta * 64 + ph.warp
                tids.add((pid, tid))
                events.append(
                    {
                        "name": ph.name,
                        "cat": launch.stem,
                        "ph": "X",
                        "pid": pid,
                        "tid": tid,
                        "ts": shift + ph.start_ns / 1000.0,
                        "dur": ph.ns / 1000.0,
                        "args": {"cycles": ph.cycles, "timer_ns": ph.timer_ns, "role": ph.role, "cta": ph.cta, "warp": ph.warp},
                    }
                )
            for sm, pid in pids.items():
                events.append({"name": "process_name", "ph": "M", "pid": pid, "args": {"name": f"{label} SM {sm}"}})
            roles = {(w.cta, w.warp): launch.role_of(w) for w in launch.warps}
            for pid, tid in sorted(tids):
                cta, warp = divmod(tid, 64)
                role = roles.get((cta, warp), "")
                events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid, "args": {"name": f"CTA {cta} warp {warp}" + (f" ({role})" if role else "")}})
            offset += launch.span_ns() / 1000.0 + 1.0
        return {"traceEvents": events, "displayTimeUnit": "ns", "otherData": {"schema": TRACE_SCHEMA, "units": "us"}}


# -- collecting: tilefish.run(..., probes=Probes()) ------------------------------------------------------------------------------------------


class Probes:
    """Collects the records of instrumented launches: ``tilefish.run(op, ..., probes=rec)`` (``jit``: the instrumented variant is built in
    memory; ``replay``: the manifest must be an instrumented export) and ``tilefish.run(graph, ..., probes=rec)`` (every instrumented step of the
    plan). After each launch its buffer is read back (the stream synchronised) and decoded into :attr:`launches`; :meth:`trace` returns them.

    ``instrument`` is what a ``jit`` build records (``tilefish.instrument(capacity=..., ops=...)``; an exported variant has its own). ``buffer``
    is a device buffer of the caller's (a CUDA ``uint8`` tensor) that every launch records into, instead of one allocated per launch (a plan's
    steps take consecutive regions of it): it must hold the largest launch's records (:func:`tilefish._probe.buffer_bytes`)."""

    def __init__(self, instrument: Any = True, *, buffer: Any = None) -> None:
        cfg = _probe.config(instrument)
        if cfg is None:
            raise TilefishProbeError("tilefish.Probes(instrument=...) records the probes of an instrumented build: give True or tilefish.instrument(...)")
        self.instrument = cfg
        self.buffer = buffer
        self.launches: "list[Launch]" = []

    def trace(self) -> ProbeTrace:
        """The launches collected so far."""
        return ProbeTrace(list(self.launches))

    def clear(self) -> None:
        self.launches.clear()

    def region(self, nbytes: int, device_index: int, offset: int = 0) -> "tuple[Any, int]":
        """``(tensor, address)`` of ``nbytes`` of device memory for a launch's records: ``offset`` bytes into the caller's buffer, or a new tensor."""
        if self.buffer is not None:
            t = self.buffer
            have = int(t.numel()) * int(t.element_size())
            if offset + nbytes > have:
                raise TilefishProbeError(
                    f"the probe buffer of tilefish.Probes(buffer=...) has {have} bytes, but the launch's records need {nbytes} at offset {offset}: give a "
                    "larger buffer (or none: one is allocated per launch)"
                )
            return t, int(t.data_ptr()) + offset
        import torch

        t = torch.empty(max(nbytes, 1), dtype=torch.uint8, device=f"cuda:{device_index}")
        return t, int(t.data_ptr())

    def collect(
        self,
        tensor: Any,
        address: int,
        *,
        built: Any,
        grid: "Sequence[int]",
        block: "Sequence[int]",
        stem: str,
        arch: str,
        op: str = "",
        graph: "str | None" = None,
        step: "int | None" = None,
        bucket: "Mapping[str, int] | None" = None,
        manifest_digest: "str | None" = None,
    ) -> Launch:
        """Read a launch's records back from ``address`` (inside ``tensor``) and decode them (synchronises the device with the stream of the copy)."""
        field_ = built.instrument
        capacity = int(field_["capacity"])
        warps = max(1, -(-int(block[0]) // 32))
        ctas = math.prod(int(g) for g in grid)
        nbytes = _probe.buffer_bytes(capacity, warps, ctas)
        start = address - int(tensor.data_ptr())
        data = tensor.view(-1)[start : start + nbytes].cpu().numpy().tobytes() if hasattr(tensor, "view") else bytes(tensor)[start : start + nbytes]
        recs, missing = _probe.decode(data, capacity, warps, ctas)
        launch = Launch(
            stem=stem,
            arch=arch,
            grid=tuple(int(g) for g in grid),  # type: ignore[arg-type]
            block=tuple(int(b) for b in block),  # type: ignore[arg-type]
            capacity=capacity,
            probes=[dict(p) for p in field_["probes"]],
            warps=recs,
            missing=missing,
            op=op,
            graph=graph,
            step=step,
            bucket=dict(bucket) if bucket is not None else None,
            manifest_digest=manifest_digest,
        )
        self.launches.append(launch)
        return launch


def merge(traces: "Iterable[ProbeTrace]") -> ProbeTrace:
    """One trace of the launches of several."""
    out = ProbeTrace()
    for t in traces:
        out.launches.extend(t.launches)
    return out


__all__ = ["Launch", "Phase", "ProbeTrace", "Probes", "TRACE_SCHEMA", "TilefishProbeError", "merge"]
