"""``tilefish-explain``: the SASS of a kernel of an export, explained line by line (``python -m tilefish.explain``, or :func:`explain`).

For a kernel (``tilefish-explain DIR rmsnorm``), a step of a plan (``--graph block --step 3``) or a fused region (``--graph block --region 2``)
of an export directory:

* **annotate** (the default view): the SASS, each run of instructions under the source line its line information names: the line of the
  Triton or Gluon kernel, or of the generated source and, through what the lowering recorded, the op of the kernel's graph and the line of the
  ``@tilefish.kernel`` body it was written on; for a fused region the op of the graph body (its index in the traced graph, as the fusion report
  numbers it) and its line. ``--ptx`` adds the PTX line each instruction was assembled from.
* **summary**: the instruction mix (classes, the MMAs and their kinds and shapes, the PTX MMAs, TMA, local-memory traffic, barriers and fences),
  the registers, stack frame, shared memory and named barriers, and an estimate of the occupancy.
* **lines**: instructions per source line, by class.
* **diff**: ``--diff OTHER_DIR`` (the same kernel in another export) or ``--diff-arch sm_120`` (the same kernel for another architecture): what
  differs in the summary, and the source lines whose instructions differ.

It reads the trace of the export (``trace/``, written by ``export(..., trace=True)``: :mod:`tilefish._trace`) when there is one, checked
against the manifest (each binary's sha256) and its own digest, and needs no CUDA tool then: it runs anywhere, on any machine. Without a trace
it disassembles the cubin of the manifest (an ``nvdisasm`` that decodes the architecture is needed; no line of a generated source is mapped to
an op then). The module-relative source names of the line information (``kernels/mm.py``, ``tilefish_generated/rmsnorm.py``) are resolved in
the trace's copies first, then under each ``--source-root`` (default: the working directory), so it works from any directory.

``--graph G --advise`` is the plan-level fusion advisor (:mod:`tilefish._advise`, ``docs/advise.md``): which steps of the plan could be one kernel,
what that would save and what to do; ``--profile OUT`` times every step of the plan on the device first (:mod:`tilefish._profile`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import _sass
from ._trace import INDEX_NAME, TRACE_DIR, TRACE_SCHEMA, _safe_name, index_digest
from .errors import TilefishError

VIEWS = ("annotate", "summary", "lines", "diff")


class TilefishExplainError(TilefishError, ValueError):
    """What ``tilefish-explain`` was asked for is not in the export (no such kernel, step or region), or its trace is not the export's."""


# -- the trace --------------------------------------------------------------------------------------------------------------------------------


def _read_json(path: pathlib.Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise TilefishExplainError(f"cannot read {what} {str(path)!r}: {e}") from e


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Trace:
    """The trace of an export directory (``trace/index.json``), its digest checked."""

    root: pathlib.Path
    index: "dict[str, Any]"

    @classmethod
    def load(cls, directory: pathlib.Path) -> "Trace | None":
        """The trace of ``directory``, or ``None`` when it has none. Raises :class:`TilefishExplainError` for an index that is not intact."""
        root = directory / TRACE_DIR
        path = root / INDEX_NAME
        if not path.is_file():
            return None
        index = _read_json(path, "the trace index")
        if not isinstance(index, dict) or index.get("schema") != TRACE_SCHEMA:
            raise TilefishExplainError(f"{str(path)!r} is not a trace index of {TRACE_SCHEMA}")
        if index.get("trace_digest") != index_digest(index):
            raise TilefishExplainError(f"{str(path)!r} does not have the digest it records: it was changed after it was written")
        return cls(root, index)

    def kernel(self, stem: str, arch: str, sha256: str) -> "dict[str, Any] | None":
        """The entry of ``stem`` on ``arch``, when it is of the binary whose sha256 is ``sha256`` (the manifest's) and its files are intact."""
        entry = (self.index.get("kernels") or {}).get(f"{stem}.{arch}")
        if not isinstance(entry, dict) or entry.get("sha256") != sha256:
            return None
        base = self.root / f"{stem}.{arch}"
        for rel, digest in (entry.get("files") or {}).items():
            path = base / rel
            if not _safe_name(rel) or not path.is_file() or _sha256(path) != digest:  # (a name inside the trace: no "..", not absolute)
                raise TilefishExplainError(f"the trace file {str(path)!r} is not the one its index records (changed or missing)")
        return entry

    def graph(self, name: str, plan_digest: "str | None" = None) -> "dict[str, Any] | None":
        """The trace of the graph ``name`` (``graphs/<name>.json``), when it is of the plan whose digest is ``plan_digest`` (``None``: any) and of
        the manifest the index is of; ``None`` otherwise (a later export of another plan left it behind)."""
        found = (self.index.get("graphs") or {}).get(name)
        if not isinstance(found, dict) or not isinstance(found.get("file"), str):
            return None
        rel = found["file"]
        path = self.root / rel
        if not _safe_name(rel) or not path.is_file() or _sha256(path) != found.get("sha256"):
            raise TilefishExplainError(f"the trace file {str(path)!r} is not the one its index records (changed or missing)")
        for other, digest in (found.get("files") or {}).items():
            if not _safe_name(other) or not (self.root / other).is_file() or _sha256(self.root / other) != digest:
                raise TilefishExplainError(f"the trace file {str(self.root / other)!r} is not the one its index records (changed or missing)")
        doc = _read_json(path, "the graph trace")
        if not isinstance(doc, dict) or doc.get("manifest_digest") != self.index.get("manifest_digest"):
            return None
        if plan_digest is not None and doc.get("plan_digest") != plan_digest:
            return None
        return doc


# -- what is explained ------------------------------------------------------------------------------------------------------------------------


@dataclass
class Explained:
    """One binary of an export, ready to be explained: its listing, its PTX, what the lowering recorded about its generated lines, its sources,
    what the ELF file says, and (for a plan step or a fused region) the graph context."""

    stem: str
    arch: str
    symbol: str
    listing: "_sass.Listing"
    sass_label: str
    attributes: "dict[str, Any]"
    launch: "dict[str, Any]"
    ptx: "str | None" = None
    origin: str = ""
    language: str = ""
    nodes: "dict[str, Any] | None" = None  # nodes.json: generated file, line -> what it was written for
    sources: "dict[str, pathlib.Path]" = field(default_factory=dict)
    source_roots: "tuple[pathlib.Path, ...]" = ()
    context: "dict[str, Any]" = field(default_factory=dict)  # a plan step or a region: the graph, the step, the region's body
    traced: bool = False

    # -- sources -----------------------------------------------------------------------------------------------------------------------

    def _source_lines(self, name: "str | None") -> "list[str] | None":
        if not name:
            return None
        cache = self.__dict__.setdefault("_lines", {})
        if name in cache:
            return cache[name]
        found = None
        candidates = [self.sources.get(name)] + [root / name for root in self.source_roots] + ([pathlib.Path(name)] if pathlib.Path(name).is_absolute() else [])
        for path in candidates:
            if path is not None and path.is_file():
                try:  # (split on newlines only: a form feed or a line separator in a source must not move its line numbers)
                    found = path.read_text(encoding="utf-8", errors="replace").split("\n")
                except OSError:
                    continue
                break
        cache[name] = found
        return found

    def source_text(self, name: "str | None", line: "int | None") -> str:
        """The text of line ``line`` of the source ``name`` (stripped), or ``""`` when it cannot be found."""
        lines = self._source_lines(name)
        if lines is None or not line or line > len(lines):
            return ""
        return lines[line - 1].strip()

    def written_for(self, name: "str | None", line: "int | None") -> "dict[str, Any] | None":
        """What line ``line`` of the generated source ``name`` was written for (``nodes.json``), with the graph op of a fused region."""
        if not self.nodes or name != self.nodes.get("file") or not line:
            return None
        found = (self.nodes.get("lines") or {}).get(str(line))
        if found is None:
            return None
        out = dict(found)
        body = self.context.get("region_body")
        if body is not None and found.get("origin") is not None and found["origin"] < len(body):
            out["graph_op"] = body[found["origin"]]
        return out

    # -- views ---------------------------------------------------------------------------------------------------------------------------

    @property
    def instructions(self) -> "tuple[_sass.Instruction, ...]":
        """The binary's instructions: its kernel's function, then any other function of its code (:func:`tilefish._sass.kernel_instructions`)."""
        return _sass.kernel_instructions(self.listing, self.symbol)

    def summary(self) -> "dict[str, Any]":
        """The instruction mix and the attributes (:func:`tilefish._sass.summary`), with what is explained and where its SASS comes from."""
        arch = int(self.arch[3:]) if self.arch.startswith("sm_") and self.arch[3:].isdigit() else None
        block = self.launch.get("block") or [0, 1, 1]
        threads = int(block[0]) * int(block[1]) * int(block[2]) if block else 0
        out = _sass.summary(
            self.instructions, ptx=self.ptx, attributes=self.attributes, threads=threads, shared=int(self.launch.get("shared") or 0), arch=arch
        )
        out = {"kernel": self.stem, "arch": self.arch, "origin": self.origin, "sass": self.sass_label, **out}
        if self.context:
            out["context"] = {k: v for k, v in self.context.items() if k != "region_body"}
        return out

    def _label(self, name: "str | None", line: "int | None") -> str:
        """The header of a run of instructions: the source line, its text, and what it was written for."""
        if name is None:
            return "(no line information)"
        head = f"{name}:{line}"
        text = self.source_text(name, line)
        out = f"{head}  {text}" if text else head
        made = self.written_for(name, line)
        if made is not None:
            out += f"    [{self.describe(made)}]"
        return out

    def _at(self, loc: str) -> str:
        """``file:line: text`` of a ``"file:line"`` location (the text when the source is found)."""
        file, _, ln = loc.rpartition(":")
        body = self.source_text(file, int(ln)) if ln.isdigit() else ""
        return f"{loc}: {body}" if body else loc

    def describe(self, made: "Mapping[str, Any]") -> str:
        """What a generated line was written for, in words: ``node 2 mul (helper _tf_mul_rn_f32), written at model.py:17: y = x * w``, or for a fused
        region ``graph op #12 mul at model.py:17: ...`` (the op of the graph body the region took)."""
        if "what" in made:
            return str(made["what"])
        desc = f"node {made.get('node')} {made.get('op')}" + (f" (helper {made['helper']})" if made.get("helper") else "")
        g = made.get("graph_op")
        if g is not None:
            return desc + f"; graph op #{g.get('node')} {g.get('op')}" + (f" at {self._at(g['loc'])}" if g.get("loc") else "")
        if made.get("loc"):
            desc += f", written at {self._at(made['loc'])}"
        return desc

    def header(self) -> "list[str]":
        s = self.summary()
        out = [f"{self.stem} on {self.arch} ({self.origin or self.language}): {self.sass_label}"]
        attrs = [f"{k} {s[k]}" for k in ("registers", "frame_size", "static_shared", "num_barriers") if k in s]
        if attrs:
            out.append("  " + ", ".join(attrs) + (f"; dynamic shared {self.launch.get('shared')}" if self.launch.get("shared") else ""))
        if self.context.get("label"):
            out.append("  " + self.context["label"])
        if not self.traced:
            out.append("  (no trace: the cubin of the manifest was disassembled; export with trace=True for the ops of generated lines)")
        return out

    def annotate(self, *, ptx: bool = False) -> str:
        """The SASS, each run of instructions under its source line (see the module docstring)."""
        ptx_lines = self.ptx.split("\n") if (ptx and self.ptx) else None
        out = self.header() + [""]
        last: "tuple[Any, Any] | None" = None
        function = self.symbol
        for i in self.instructions:
            if i.function != function:  # another function of the binary's code (a noinline function, a subroutine of ptxas)
                function, last = i.function, None
                out += ["", f"== function {function}"]
            key = (i.file, i.line)
            if key != last:
                out.append(self._label(i.file, i.line))
                last = key
            extra = f"  [{_sass.class_of(i)}]"
            if ptx_lines is not None and i.ptx_line and i.ptx_line <= len(ptx_lines):
                extra += f"  ptx {i.ptx_line}: {ptx_lines[i.ptx_line - 1].strip()}"
            elif i.ptx_line:
                extra += f"  ptx {i.ptx_line}"
            notes = f"  ({', '.join(i.notes)})" if i.notes else ""
            out.append(f"    /*{i.address:04x}*/ {i.text} ;{extra}{notes}")
        return "\n".join(out) + "\n"

    def lines(self) -> "list[dict[str, Any]]":
        """Instructions per source line (:func:`tilefish._sass.by_line`), each with its source text and what it was written for."""
        rows = _sass.by_line(self.instructions)
        for r in rows:
            r["text"] = self.source_text(r["file"], r["line"])
            made = self.written_for(r["file"], r["line"])
            if made is not None:
                r["written_for"] = made
        return rows

    def lines_text(self) -> str:
        cols = [name for name, _ in _sass.LINE_COLUMNS] + ["alu"]
        out = self.header() + ["", f"{'count':>6} " + " ".join(f"{c:>5}" for c in cols) + "  line"]
        for r in self.lines():
            where = f"{r['file']}:{r['line']}" if r["file"] else "(no line information)"
            made = r.get("written_for")
            label = ""
            if made is not None:
                g = made.get("graph_op")
                loc = (g or {}).get("loc") or made.get("loc")
                op = f"#{g['node']} {g['op']}" if g else str(made.get("op"))
                label = f" [{made['what']}]" if "what" in made else f" [{op}{' @ ' + loc if loc else ''}]"
            text = f"  {r['text']}" if r["text"] else ""
            out.append(f"{r['count']:>6} " + " ".join(f"{r['columns'].get(c, 0) or '.':>5}" for c in cols) + f"  {where}{label}{text}")
        return "\n".join(out) + "\n"

    def summary_text(self) -> str:
        s = self.summary()
        out = self.header() + [""]
        out.append(f"instructions: {s['instructions']}")
        out.append("classes: " + ", ".join(f"{k} {v}" for k, v in s["classes"].items()))
        if s["mma"]:
            out.append("SASS MMAs:")
            for e in s["mma"]:
                d = e.get("descriptor")
                desc = f"  (descriptor: kind::{d['kind']}, {d['a']} x {d['b']} -> {d['d']}, M{d['m']} N{d['n']})" if d else ""
                out.append(f"  {e['count']:>5} x {e['sass']}{desc}")
        if s.get("ptx_mma"):
            out.append("PTX MMAs:")
            out += [f"  {e['count']:>5} x {e['ptx']}" for e in s["ptx_mma"]]
        out.append(f"TF32 MMAs: {s['tf32_mma']}")
        sp = s["spills"]
        out.append(f"local memory: {sp['loads']} loads, {sp['stores']} stores" + (f" ({sp['noted']} marked as spills by nvdisasm)" if sp["noted"] else ""))
        for key, title in (("tma", "TMA"), ("barriers", "barriers"), ("fences", "fences")):
            if s.get(key):
                out.append(f"{title}: " + ", ".join(f"{k} {v}" for k, v in s[key].items()))
        if s.get("occupancy"):
            o = s["occupancy"]
            out.append(f"occupancy (estimate from the attributes): {o['blocks_per_sm']} blocks, {o['warps_per_sm']} warps per SM ({o['occupancy']:.0%}), limited by {', '.join(o['limited_by'])}")
        return "\n".join(out) + "\n"


def diff(a: Explained, b: Explained) -> str:
    """What differs between two binaries of one kernel (two exports, or two architectures): the summary, then the source lines whose
    instructions differ (by count and class), largest change first."""
    out = [f"diff: {a.stem} {a.arch} ({a.sass_label})", f"  vs  {b.stem} {b.arch} ({b.sass_label})", ""]
    changes = _sass.diff_summaries(a.summary(), b.summary())
    out += ["summary:"] + ([f"  {c}" for c in changes] or ["  (the same)"])
    ra = {(r["file"], r["line"]): r for r in a.lines()}
    rb = {(r["file"], r["line"]): r for r in b.lines()}
    rows = []
    for key in set(ra) | set(rb):
        x, y = ra.get(key), rb.get(key)
        cx, cy = (x or {}).get("count", 0), (y or {}).get("count", 0)
        colx, coly = (x or {}).get("columns", {}), (y or {}).get("columns", {})
        if cx != cy or colx != coly:
            delta = ", ".join(f"{c} {colx.get(c, 0)}->{coly.get(c, 0)}" for c in sorted(set(colx) | set(coly)) if colx.get(c, 0) != coly.get(c, 0))
            where = f"{key[0]}:{key[1]}" if key[0] else "(no line information)"
            text = (x or y or {}).get("text", "")
            rows.append((abs(cy - cx), f"  {cx:>5} -> {cy:<5} {where}  {delta}" + (f"  | {text}" if text else "")))
    out += ["", "source lines whose instructions differ:"] + ([r for _, r in sorted(rows, key=lambda t: -t[0])] or ["  (none)"])
    return "\n".join(out) + "\n"


# -- loading ----------------------------------------------------------------------------------------------------------------------------------


def _plan_doc(directory: pathlib.Path, graph: str) -> "dict[str, Any]":
    for name in (f"{graph}.plan.json", "plan.json"):
        path = directory / name
        if path.is_file():
            doc = _read_json(path, "the plan")
            if isinstance(doc, dict) and doc.get("graph") == graph:
                return doc
    raise TilefishExplainError(f"{str(directory)!r} has no plan of the graph '{graph}' (plan.json or {graph}.plan.json)")


def _parse_bucket(bucket: "str | Mapping[str, int] | None") -> "dict[str, int] | None":
    if bucket is None or isinstance(bucket, Mapping):
        return None if bucket is None else {str(k): int(v) for k, v in bucket.items()}
    out = {}
    for part in str(bucket).split(","):
        name, eq, value = part.partition("=")
        if not eq or not value.strip().lstrip("-").isdigit():
            raise TilefishExplainError(f"a bucket is NAME=VALUE[,NAME=VALUE...] (D=96), got {bucket!r}")
        out[name.strip()] = int(value)
    return out


def _pick_bucket(doc: "dict[str, Any]", bucket: "dict[str, int] | None") -> "dict[str, Any]":
    buckets = doc.get("buckets") or []
    if bucket is None:
        if len(buckets) == 1:
            return buckets[0]
        raise TilefishExplainError(f"the plan of '{doc.get('graph')}' has {len(buckets)} buckets: say which with --bucket ({[b.get('values') for b in buckets]})")
    for b in buckets:
        if b.get("values") == bucket:
            return b
    raise TilefishExplainError(f"the plan of '{doc.get('graph')}' has no bucket {bucket} ({[b.get('values') for b in buckets]})")


def _select(manifest: Any, kernel: "str | None", arch: "str | None") -> "tuple[str, str]":
    stems = sorted({a.stem for a in manifest.artifacts})
    if kernel is None:
        raise TilefishExplainError(f"say which kernel to explain (a stem or an op name): {stems}")
    chosen = [s for s in stems if s == kernel] or [s for s in stems if s.split(".", 1)[0] == kernel]
    if not chosen:
        raise TilefishExplainError(f"the manifest has no kernel {kernel!r}: {stems}")
    if len(chosen) > 1:
        raise TilefishExplainError(f"{kernel!r} has a binary per bucket: say which stem ({chosen})")
    stem = chosen[0]
    archs = sorted(a.arch for a in manifest.artifacts if a.stem == stem)
    if arch is None:
        if len(archs) != 1:
            raise TilefishExplainError(f"'{stem}' is built for {archs}: say which with --arch")
        arch = archs[0]
    arch = arch if arch.startswith("sm_") else f"sm_{arch}"
    if arch not in archs:
        raise TilefishExplainError(f"'{stem}' is not built for {arch} (it is for {archs})")
    return stem, arch


def load(
    directory: "str | pathlib.Path",
    kernel: "str | None" = None,
    *,
    arch: "str | None" = None,
    graph: "str | None" = None,
    step: "int | None" = None,
    bucket: "str | Mapping[str, int] | None" = None,
    region: "str | int | None" = None,
    source_roots: "Sequence[str | pathlib.Path]" = (),
) -> Explained:
    """The binary of an export directory to explain: ``kernel`` (a stem, or an op name with one binary), or a step of the plan of ``graph``
    (``step``, at ``bucket``: ``"D=96"`` or ``{"D": 96}``, needed when the plan has several), or a fused region of ``graph`` (``region``: its
    kernel's name, ``fused_...``, or its index in the fusion report), on ``arch`` (needed when the binary is built for several). ``source_roots``
    are where module-relative source names are looked up after the trace's own copies (default: the working directory)."""
    from ._manifest import Manifest

    directory = pathlib.Path(directory)
    manifest = Manifest.load(directory)
    trace = Trace.load(directory)
    context: "dict[str, Any]" = {}
    plan_digest: "str | None" = None
    if graph is not None:
        plan_digest = _plan_doc(directory, graph).get("plan_digest")
    gdoc = trace.graph(graph, plan_digest) if (trace is not None and graph is not None) else None
    if graph is not None and step is not None:
        doc = _plan_doc(directory, graph)
        chosen_bucket = _pick_bucket(doc, _parse_bucket(bucket))
        steps = chosen_bucket.get("steps") or []
        if not 0 <= step < len(steps):
            raise TilefishExplainError(f"the plan of '{graph}' at {chosen_bucket.get('values')} has steps 0 to {len(steps) - 1}, not {step}")
        st = steps[step]
        if st.get("kind") != "kernel":
            raise TilefishExplainError(
                f"step {step} of '{graph}' at {chosen_bucket.get('values')} is a {st.get('kind')} step ({st.get('entry') or 'no entry'}): it launches no "
                "kernel of tilefish (a library step is a pinned cuBLASLt or cuDNN call; copy and memset steps are the driver's)"
            )
        kernel = st["entry"]
        arch = arch or doc.get("arch")
        context.update({"graph": graph, "step": step, "bucket": chosen_bucket.get("values"), "entry": kernel})
        context["label"] = f"step {step} of the plan of '{graph}' at {chosen_bucket.get('values') or 'its only bucket'}: launches '{kernel}'"
    elif graph is not None and region is not None:
        report = directory / f"{graph}.fusion.json"
        names = []
        if report.is_file():
            rep = _read_json(report, "the fusion report")
            regions = rep.get("regions") or []
            names = [r.get("op") for r in regions]
            if isinstance(region, int) or str(region).isdigit():
                k = int(region)
                if not 0 <= k < len(regions):
                    raise TilefishExplainError(f"the fusion report of '{graph}' has regions 0 to {len(regions) - 1}, not {region}")
                region = regions[k]["op"]
            found = next((r for r in regions if r.get("op") == region), None)
            if found is not None:
                context["region_ops"] = found.get("nodes")
        if not str(region).startswith("fused_"):
            raise TilefishExplainError(f"a region is the name of its kernel (fused_...) or its index in {graph}.fusion.json ({names or 'no report'})")
        stems = sorted({a.stem for a in manifest.artifacts if a.stem.split(".", 1)[0].split("__ua", 1)[0] == region})
        if not stems:
            raise TilefishExplainError(f"the manifest has no binary of the region {region!r}")
        b = _parse_bucket(bucket)
        if len(stems) > 1 and b is not None:
            stems = [s for s in stems if all(f".{k}{v}" in s for k, v in b.items())] or stems
        if len(stems) > 1:
            raise TilefishExplainError(f"the region {region!r} has several binaries: name one as the kernel ({stems})")
        kernel = stems[0]
        context.update({"graph": graph, "region": region})
        context["label"] = f"fused region {region} of the graph '{graph}'"
    stem, arch = _select(manifest, kernel, arch)
    if gdoc is not None:
        info = (gdoc.get("regions") or {}).get(stem.split(".", 1)[0])
        if info is not None:
            context["region_body"] = info.get("body")
            ops = ", ".join(f"#{o['node']} {o['op']}" + (f" ({o['loc']})" if o.get("loc") else "") for o in info.get("body") or [])
            context["label"] = (context.get("label", f"'{stem}'") + f"; fused region {info.get('region')} of the ops {ops}").strip()
    artifact = manifest.get(stem, arch)
    assert artifact is not None
    roots = tuple(pathlib.Path(r) for r in (source_roots or [pathlib.Path.cwd()]))
    entry = trace.kernel(stem, arch, artifact.entry.get("sha256")) if trace is not None else None
    if entry is not None and "error" not in (entry.get("sass") or {"error": ""}):
        base = trace.root / f"{stem}.{arch}"  # type: ignore[union-attr]
        text = (base / "sass.txt").read_text(encoding="utf-8")
        table = _read_json(base / "lines.json", "the line table")
        listing = _with_lines(_sass.parse_listing(text), table)
        ptx = (base / "ptx.ptx").read_text(encoding="utf-8") if (base / "ptx.ptx").is_file() else None
        nodes = _read_json(base / "nodes.json", "the line map") if (base / "nodes.json").is_file() else None
        sources = {name: base / rel for name, rel in (entry.get("sources") or {}).items() if _safe_name(rel)}
        if gdoc is not None:
            sources.update({name: trace.root / rel for name, rel in (gdoc.get("sources") or {}).items() if name not in sources and _safe_name(rel)})  # type: ignore[union-attr]
        return Explained(
            stem, arch, entry["symbol"], listing, entry["sass"].get("label", ""), entry.get("attributes") or {}, entry.get("launch") or {},
            ptx, entry.get("origin", ""), entry.get("language", ""), nodes, sources, roots, context, traced=True,
        )
    symbol = artifact.entry.get("symbol") or ""
    try:
        d = _sass.disassemble(artifact.cubin)
    except TilefishError as e:
        if entry is not None:
            why = f"its trace has no SASS ({entry['sass'].get('error')}: export it again where an nvdisasm reads it)"
        elif trace is not None:
            why = "its trace is of another binary or has none of it"
        else:
            why = "export it with trace=True"
        raise TilefishExplainError(f"'{stem}' on {arch} has no SASS in the trace of {str(directory)!r} ({why}), and its cubin cannot be disassembled here: {e}") from e
    launch = {"block": artifact.entry.get("block"), "shared": artifact.entry.get("shared", 0)}
    return Explained(
        stem, arch, symbol, d.listing, d.label, _sass.kernel_attributes(artifact.cubin, symbol), launch, None, "", artifact.entry.get("kind", ""),
        None, {}, roots, context, traced=False,
    )


def _with_lines(listing: "_sass.Listing", table: "Mapping[str, Any]") -> "_sass.Listing":
    """``listing`` with the PTX line of each instruction from the trace's line table (``lines.json``)."""
    import dataclasses

    ptx = {(fn, row[0]): row[3] for fn, rows in (table.get("functions") or {}).items() for row in rows}
    return _sass.Listing(
        listing.target,
        {fn: tuple(dataclasses.replace(i, ptx_line=ptx.get((fn, i.address)) or None) for i in ins) for fn, ins in listing.functions.items()},
    )


def explain(directory: "str | pathlib.Path", kernel: "str | None" = None, *, view: str = "annotate", ptx: bool = False, **kwargs: Any) -> str:
    """The text of one view (``"annotate"``, ``"summary"``, ``"lines"``) of a binary of an export: :func:`load` with ``kwargs``, then the view."""
    e = load(directory, kernel, **kwargs)
    if view == "annotate":
        return e.annotate(ptx=ptx)
    if view == "summary":
        return e.summary_text()
    if view == "lines":
        return e.lines_text()
    raise TilefishExplainError(f"view is one of annotate, summary, lines (a diff is explain.diff(load(...), load(...))), got {view!r}")


# -- the command line -------------------------------------------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tilefish-explain",
        description="The SASS of a kernel of a tilefish export, annotated with the source lines and graph ops that produced it.",
    )
    p.add_argument("directory", help="an export directory (manifest.json; trace/ when exported with trace=True)")
    p.add_argument("kernel", nargs="?", help="a stem of the manifest, or an op name with one binary")
    p.add_argument("--arch", help="the architecture (sm_103), when the binary is built for several")
    p.add_argument("--graph", help="a graph whose plan is in the directory (with --step, --region, --advise or --profile)")
    p.add_argument("--step", type=int, help="a step of the graph's plan (its index in the bucket's steps)")
    p.add_argument("--bucket", help="the bucket of the plan (D=96,T=128), when it has several")
    p.add_argument("--region", help="a fused region of the graph: its kernel's name (fused_...) or its index in <graph>.fusion.json")
    p.add_argument("--source-root", action="append", default=[], help="where module-relative source names are looked up (repeatable; default: .)")
    p.add_argument("--view", choices=VIEWS[:3], default="annotate", help="annotate (default), summary, lines")
    p.add_argument("--ptx", action="store_true", help="annotate: show the PTX line of each instruction")
    p.add_argument("--diff", metavar="OTHER_DIR", help="compare with the same kernel in another export directory")
    p.add_argument("--diff-arch", metavar="ARCH", help="compare with the same kernel for another architecture")
    p.add_argument("--json", action="store_true", help="print the view as JSON (summary, lines, the advice)")
    a = p.add_argument_group("fusion advice (docs/advise.md)")
    a.add_argument("--advise", action="store_true", help="rank the fusion candidates of the plan of --graph: the steps, bytes and time saved, what to do")
    a.add_argument("--timing", metavar="FILE", help="--advise with the measured times of one bucket (what --profile or tilefish.profile_plan writes)")
    a.add_argument("--machine", metavar="ARCH", help="--advise with the roofline of this architecture (default: the plan's; sm_90, sm_100, sm_103, sm_120)")
    a.add_argument("--bandwidth", type=float, metavar="GBPS", help="--advise with this memory bandwidth (GB/s: a partition's, a measured one)")
    a.add_argument("--keep", action="append", default=[], metavar="BUFFER", help="--advise: a buffer the caller taps, which stays stored (repeatable)")
    a.add_argument("--top", type=int, default=20, help="--advise: how many candidates the text lists (default 20)")
    a.add_argument("--profile", metavar="OUT", help="measure the device time of every step of --graph's plan at --bucket and write it to OUT (needs a GPU and --load)")
    a.add_argument("--load", metavar="MODULE:ATTR", help="--profile: the @tilefish.graph to replay the plan with (as every Python host does), imported from MODULE")
    a.add_argument("--reps", type=int, default=50, help="--profile: timed runs (default 50)")
    a.add_argument("--warmup", type=int, default=5, help="--profile: runs before the timed ones (default 5)")
    a.add_argument("--method", choices=("graph", "events"), default="graph", help="--profile: a CUDA graph with event nodes (default), or launches with events")
    a.add_argument("--devscalar", action="append", default=[], metavar="NAME=VALUE", help="--profile: the value of a device scalar (repeatable; every one is needed)")
    a.add_argument("--pin", help="--profile: the plan's (or plan set's) digest the replay is checked against")
    a.add_argument("--trust-directory", action="store_true", help="--profile: replay whatever plan the directory holds, without a pin")
    return p


def _load_object(spec: str) -> Any:
    import importlib

    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise TilefishExplainError(f"--load takes MODULE:ATTR (models.block:graph), got {spec!r}")
    if "" not in sys.path and str(pathlib.Path.cwd()) not in sys.path:
        sys.path.insert(0, str(pathlib.Path.cwd()))
    obj: Any = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _advise_main(args: Any) -> int:
    from . import _advise

    timing: Any = args.timing
    if args.profile:
        from ._profile import profile_plan

        if not args.load:
            raise TilefishExplainError("--profile replays the plan with its graph: say where it is with --load MODULE:ATTR")
        gop = _load_object(args.load)
        if args.graph is not None and getattr(gop, "name", None) != args.graph:
            raise TilefishExplainError(f"--load gives the graph '{getattr(gop, 'name', gop)}', not '{args.graph}'")
        scalars = {}
        for item in args.devscalar:
            name, eq, value = item.partition("=")
            if not eq:
                raise TilefishExplainError(f"--devscalar takes NAME=VALUE, got {item!r}")
            scalars[name.strip()] = int(value)
        timing = profile_plan(gop, args.directory, bucket=args.bucket, reps=args.reps, warmup=args.warmup, method=args.method, pin=args.pin,
                              trust_directory=args.trust_directory, devscalars=scalars, out=args.profile)
        if not args.advise:
            if args.json:
                print(json.dumps(timing, indent=1))
            else:
                print(f"{args.profile}: {len(timing['steps'])} steps of '{timing['graph']}' at {timing['bucket']}, {timing['method']}, {timing['reps']} reps: "
                      f"sum of the steps {timing['sum_us']} us, the plan alone {timing['plan_us']['median']} us (events {100 * (timing['overhead'] or 0):+.1f}%)")
            return 0
    adv = _advise.advise(args.directory, args.graph, bucket=args.bucket, timing=timing, machine=args.machine, bandwidth=args.bandwidth, keep=args.keep)
    if args.json:
        print(json.dumps(adv.doc, indent=1, default=str))
    else:
        sys.stdout.write(adv.text(top=args.top))
    return 0


def main(argv: "Sequence[str] | None" = None) -> int:
    """``tilefish-explain``; returns the exit status (2 for a request the export cannot answer)."""
    args = _parser().parse_args(argv)
    if args.advise or args.profile:
        from ._advise import TilefishAdviseError

        try:
            return _advise_main(args)
        except (TilefishError, TilefishAdviseError) as err:
            print(f"tilefish-explain: {err}", file=sys.stderr)
            return 2
    region: "str | int | None" = args.region
    kw = dict(arch=args.arch, graph=args.graph, step=args.step, bucket=args.bucket, region=region, source_roots=args.source_root)
    try:
        e = load(args.directory, args.kernel, **kw)
        if args.diff or args.diff_arch:
            other_dir = args.diff or args.directory
            other = load(other_dir, e.stem, arch=args.diff_arch or e.arch, source_roots=args.source_root)
            if args.json:
                print(json.dumps({"a": e.summary(), "b": other.summary(), "changes": _sass.diff_summaries(e.summary(), other.summary())}, indent=1))
            else:
                sys.stdout.write(diff(e, other))
            return 0
        if args.json:
            doc: Any = e.summary() if args.view == "summary" else e.lines() if args.view == "lines" else {"summary": e.summary(), "lines": e.lines()}
            print(json.dumps(doc, indent=1, default=str))
        elif args.view == "summary":
            sys.stdout.write(e.summary_text())
        elif args.view == "lines":
            sys.stdout.write(e.lines_text())
        else:
            sys.stdout.write(e.annotate(ptx=args.ptx))
    except TilefishError as err:
        print(f"tilefish-explain: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["Explained", "Trace", "TilefishExplainError", "diff", "explain", "load", "main"]
