"""The trace of an export: every level of every kernel, kept next to the manifest but outside it (``export(..., trace=True)``).

Every kernel goes DSL / Triton / Gluon -> TTIR -> TTGIR -> LLVM IR -> PTX -> ``ptxas`` -> SASS. Tracing a hot spot or a bit difference back through
that chain meant dumping every level by hand. An export with ``trace=True`` (or with ``TILEFISH_TRACE=1`` in the environment, for an export run
by a script one does not edit) writes, for each binary it builds, ``trace/<stem>.<arch>/``:

* ``source/<name>``: the source files the binary's line information names (the generated source of a lowered kernel or a fused region, the
  file of a Triton or Gluon kernel, Triton's own ``triton/language/*.py``), under the module-relative names the debug information gives them
  (:func:`tilefish._source.source_name`), so that the trace is read from any directory;
* ``frontend.mlir`` (the IR the frontend built, with its locations), ``ttir.mlir``, ``ttgir.mlir``, ``llir.ll``, ``ptx.ptx``: what Triton made at
  each stage (a Gluon kernel has no TTIR; a prebuilt binary has its PTX, when it is one);
* ``sass.txt``: the SASS, ``nvdisasm -c -g -gi`` (line information in it), and ``lines.json``: the line table, for each instruction its address,
  source file and line, and PTX line (``nvdisasm -gp``); where no ``nvdisasm`` decodes the architecture (sm_103 with Triton's bundled CUDA
  12.8 tools) the SASS of the PTX assembled for the base of its family, which the index says (``"proxy": true``, and ``"matches"``: whether
  its registers, stack frames and code sizes are the binary's);
* ``nodes.json`` for a lowered kernel or a fused region: what each line of the generated source was written for (the node of the kernel's
  graph, its op, the line of the body it was written on; for a fused region the op's position in the region).

``trace/graphs/<graph>.json`` says, for a graph, what each step of its plan launches and, for a fused region, the ops of the graph body it
took and their lines and its template, and why the regions are where they are (``boundaries``: :func:`tilefish._fusion.boundaries`, which the
fusion advisor reads). ``trace/index.json`` (``tilefish.trace.v1``) lists every file with its sha256, the manifest digest the trace is of and
the sha256 of each binary, and its own digest (``trace_digest``) over all of it.

**It holds source code.** The kernels' own files, the graph body's and Triton's are copied in (only files this process has loaded: nothing is
searched for, and a prebuilt binary's are never read). Do not ship ``trace/`` with an export: it is for the people who build it.

**Outside the pinned artefact.** Nothing in the manifest, a plan or a plan set names the trace, and no digest covers it: an export with or
without it writes the same manifest and plans, byte for byte (a traced export compiles every Triton kernel instead of reading the build cache,
which holds binaries, not their IR; the build is reproducible, so the binaries are the same). A host never reads it. ``tilefish-explain``
(:mod:`tilefish.explain`) does, and checks it against the manifest (each binary's sha256) before it trusts a line of it.

**Memory.** The levels of a binary are written after every binary of the export is built; until then they are files of a temporary directory
(``TMPDIR``), one per distinct text, which goes when the export ends (:func:`tilefish._build.export_session`), so an export of many binaries
does not hold every level of every one of them in memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ._build import write_file
from .errors import TilefishBuildError

TRACE_DIR = "trace"
INDEX_NAME = "index.json"
TRACE_SCHEMA = "tilefish.trace.v1"
TRACE_ENV = "TILEFISH_TRACE"
_TAG = b"tilefish.trace-digest.v1\0"
_STAGE_FILES = (("source", "frontend.mlir"), ("ttir", "ttir.mlir"), ("ttgir", "ttgir.mlir"), ("llir", "llir.ll"), ("ptx", "ptx.ptx"))
_PTX_FILE = re.compile(r'^\s*\.file\s+\d+\s+"([^"]*)"', re.M)


class TilefishTraceWarning(UserWarning):
    """A traced export could not write part of a trace (no ``nvdisasm`` reads the binary, a source file was not found): the rest is written, and the
    index says what is missing."""


def wanted(trace: "bool | None") -> bool:
    """Whether an export writes a trace: ``trace=`` when given, else ``TILEFISH_TRACE`` (``1``: yes; unset, empty or ``0``: no)."""
    if trace is not None:
        if not isinstance(trace, bool):
            raise TilefishBuildError(f"trace= is True, False or None (TILEFISH_TRACE decides), got {trace!r}")
        return trace
    value = os.environ.get(TRACE_ENV, "")
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    raise TilefishBuildError(f"{TRACE_ENV}={value!r}: set it to 1 to trace every export (or 0, or leave it unset)")


@dataclass
class Item:
    """One binary of a traced export, with what the trace needs besides it: what it is (``origin``: ``"kernel"``, ``"fused"``, ``"generated"``,
    ``"extern"``, ``"binary"``), its lowering (a generated source and its line map), its jit function (a hand-written kernel's file), and a
    prebuilt PTX's text."""

    built: Any
    origin: str
    lowered: Any = None
    jit: Any = None
    ptx: "str | None" = None


def item_of(built: Any, target: Any, jit: Any) -> Item:
    """The :class:`Item` of a binary built for ``target`` (a ``runtime._Target``) from ``jit``."""
    extern = target.launch_extern() if target is not None else None
    lowered = getattr(target, "lowered", None)
    ptx = None
    if extern is not None and extern.binary is not None:
        origin = "binary"
        if extern.binary.kind == "ptx":
            ptx = extern.binary.sources["ptx"].decode("utf-8", "replace")
    elif getattr(target, "kernel", None) is not None:
        origin = "fused" if getattr(target.kernel, "context_free", False) else "kernel"
    elif lowered is not None:
        origin = "generated"
    else:
        origin = "extern"
    return Item(built, origin, lowered, jit, ptx)


def _safe_name(name: str) -> bool:
    parts = name.replace("\\", "/").split("/")
    return bool(name) and not name.startswith(("/", "<")) and ".." not in parts and "" not in parts and ":" not in parts[0]


def _source_of(name: str, item: Item) -> "bytes | None":
    """The bytes of the source file the debug information names ``name``: the generated source, the jit function's file, or the file of a module
    this process has loaded (``__file__``) whose module-relative name it is. Nothing is searched for (not ``sys.path``, not the working directory),
    and a prebuilt binary's PTX names files of another build: none of them is read."""
    from ._source import function_file, module_of, source_name

    if item.origin == "binary":
        return None
    lowered = item.lowered
    if lowered is not None and getattr(lowered, "filename", None) == name:
        return lowered.source.encode("utf-8")
    paths = []
    path = function_file(item.jit) if item.jit is not None else None
    if path is not None and source_name(path, module_of(item.jit)) == name:
        paths.append(path)
    for module_name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if isinstance(file, str) and file.endswith(".py") and source_name(file, module_name) == name:
            paths.append(file)
    for p in paths:
        try:
            return pathlib.Path(p).read_bytes()
        except OSError:
            continue
    return None


def _json(doc: Any) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("utf-8")


def _line_table(listing: Any) -> "dict[str, Any]":
    files: "list[str]" = []
    index: "dict[str, int]" = {}
    functions: "dict[str, list[list[int]]]" = {}
    for name, instructions in listing.functions.items():
        rows = []
        for i in instructions:
            f = -1
            if i.file is not None:
                f = index.setdefault(i.file, len(files))
                if f == len(files):
                    files.append(i.file)
            rows.append([i.address, f, i.line or 0, i.ptx_line or 0])
        functions[name] = rows
    return {"files": files, "functions": functions, "columns": ["address", "file", "line", "ptx_line"]}


def write_kernel(root: pathlib.Path, item: Item) -> "dict[str, Any]":
    """Write ``trace/<stem>.<arch>/`` of one binary; returns its entry of the index."""
    from . import _sass

    b = item.built
    key = f"{b.stem}.{b.arch}"
    where = root / key
    if where.exists():
        shutil.rmtree(where)
    where.mkdir(parents=True)
    files: "dict[str, str]" = {}
    problems: "list[str]" = []

    def put(rel: str, data: bytes) -> None:
        path = where / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        write_file(path, data)
        files[rel] = hashlib.sha256(data).hexdigest()

    stages = dict(b.stages or {})
    ptx = stages.get("ptx") or b.ptx or item.ptx or ""
    if item.ptx and "ptx" not in stages:
        stages["ptx"] = item.ptx
    for stage, name in _STAGE_FILES:
        text = stages.get(stage)
        if text:
            put(name, text.encode("utf-8"))
    names = list(dict.fromkeys(_PTX_FILE.findall(ptx)))
    if item.lowered is not None and item.lowered.filename not in names:
        names.insert(0, item.lowered.filename)
    for m in getattr(item.lowered, "line_map", ()) or ():  # the files of the body the generated lines were written for (a kernel's, a graph's)
        loc = m.get("loc") if isinstance(m, dict) else None
        if loc and loc.rsplit(":", 1)[0] not in names:
            names.append(loc.rsplit(":", 1)[0])
    sources: "dict[str, str]" = {}
    for name in names:
        if not _safe_name(name):
            problems.append(f"the source {name!r} is not a module-relative name (TILEFISH_DEBUG_PATHS=absolute?): not copied")
            continue
        data = _source_of(name, item)
        if data is None:
            problems.append(f"the source {name!r} was not found on sys.path: not copied")
            continue
        put(f"source/{name}", data)
        sources[name] = f"source/{name}"
    if item.lowered is not None and getattr(item.lowered, "line_map", ()):
        lines = {str(n): m for n, m in enumerate(item.lowered.line_map, 1) if m is not None}
        put("nodes.json", _json({"file": item.lowered.filename, "kernel": item.lowered.name, "lines": lines}))
    entry: "dict[str, Any]" = {
        "stem": b.stem,
        "arch": b.arch,
        "op": b.spec.op,
        "origin": item.origin,
        "language": "binary" if b.spec.binary is not None else b.spec.language,
        "symbol": b.symbol,
        "sha256": b.sha256,
        "sha256_code": b.sha256_code,
        "ptx_target": b.ptx_target,
        "sources": sources,
    }
    if b.triton_version:
        entry["triton"] = b.triton_version
    if b.ptxas_version:
        entry["ptxas"] = b.ptxas_version
    try:
        d = _sass.disassemble(b.cubin, ptx=ptx or None, options=b.options, prebuilt=item.origin == "binary")
    except TilefishBuildError as e:
        problems.append(f"no SASS: {e}")
        entry["sass"] = {"error": str(e)}
    else:
        put("sass.txt", d.text.encode("utf-8"))
        put("lines.json", _json(_line_table(d.listing)))
        entry["sass"] = {"target": d.target, "proxy": d.proxy, "nvdisasm": d.version, "label": d.label}
        if d.proxy:  # a re-assembly for the family base: whether its registers, stack frames and code sizes are the binary's
            entry["sass"]["matches"] = bool(d.matches)
            if d.mismatch:
                problems.append(f"the SASS is a re-assembly that is not the binary ({d.mismatch})")
    if b.sass_checked:
        entry["sass_checked"] = b.sass_checked  # how the build judged the declaration's SASS assertions
    entry["attributes"] = _sass.kernel_attributes(b.cubin, b.symbol)
    entry["launch"] = {"block": list(b.block), "shared": b.shared, "num_warps": b.num_warps}
    entry["files"] = dict(sorted(files.items()))
    if problems:
        entry["problems"] = problems
        warnings.warn(f"the trace of '{key}' is not complete: {'; '.join(problems)}", TilefishTraceWarning, stacklevel=4)
    return entry


def graph_doc(graph: Any, plan: Any, plan_file: str) -> "dict[str, Any]":
    """``trace/graphs/<graph>.json``: per bucket, the steps of the plan (the entry each launches); per fused region, the ops of the graph body it took
    (their index, op and line) in the order of its kernel's body."""
    from . import _fusion

    doc: "dict[str, Any]" = {
        "graph": graph.graph.name,
        "plan": plan_file,
        "plan_digest": plan.digest,  # the plan (and the manifest) this file describes: a reader ignores it for any other
        "manifest_digest": plan.manifest_digest,
        "buckets": [],
        "regions": {},
    }
    for b in plan.doc.get("buckets", []):
        steps = [{"index": k, "kind": s.get("kind"), "entry": s.get("entry")} for k, s in enumerate(b.get("steps", []))]
        doc["buckets"].append({"values": b.get("values", {}), "steps": steps})
    if _fusion.has_arithmetic(graph):
        f = _fusion.fused(graph)
        src = f.source.nodes
        for r in f.regions:
            body = [*sorted(r.free), *r.nodes]
            ops = []
            for k in body:
                o = f.origin[k]
                n = src[o]
                ops.append({"node": o, "op": n.op, "loc": f"{n.loc[0]}:{n.loc[1]}" if n.loc else None})
            template = r.kernel.lower().template
            for name in {r.kernel.name, r.call.name}:
                doc["regions"][name] = {"kernel": r.kernel.name, "region": r.index, "template": template, "body": ops}
        try:  # why the regions are where they are (tilefish.advise); a trace is never refused for it
            doc["boundaries"] = _fusion.boundaries(graph)
        except Exception as e:  # noqa: BLE001
            doc["boundaries_error"] = f"{type(e).__name__}: {e}"
    return doc


def index_digest(index: "Mapping[str, Any]") -> str:
    """The digest of a trace index: the sha256 of its canonical JSON without ``trace_digest``."""
    body = {k: v for k, v in index.items() if k != "trace_digest"}
    return hashlib.sha256(_TAG + json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def write(directory: "str | os.PathLike[str]", items: "Sequence[Item]", manifest: Any, graphs: "Iterable[tuple[Any, Any, str]]" = ()) -> "dict[str, Any]":
    """Write the trace of an export into ``<directory>/trace``: a directory per binary of ``items``, a file per graph of ``graphs`` (``(graph,
    plan, plan file name)``), and the index (merged with an index already there: an entry of another binary stays while the manifest still has
    that binary). ``manifest`` is the loaded manifest the export wrote. Returns the index."""
    from . import __version__

    root = pathlib.Path(directory) / TRACE_DIR
    root.mkdir(parents=True, exist_ok=True)
    have = {(a.stem, a.arch): a.entry.get("sha256") for a in manifest.artifacts}
    previous: "dict[str, Any]" = {}
    old = root / INDEX_NAME
    if old.is_file():
        try:
            doc = json.loads(old.read_text(encoding="utf-8"))
            previous = doc if isinstance(doc, dict) and doc.get("schema") == TRACE_SCHEMA else {}
        except (OSError, ValueError):
            previous = {}
    kernels: "dict[str, Any]" = {}
    for key, entry in (previous.get("kernels") or {}).items():
        if isinstance(entry, dict) and have.get((entry.get("stem"), entry.get("arch"))) == entry.get("sha256"):
            kernels[key] = entry
    for item in items:
        entry = write_kernel(root, item)
        kernels[f"{entry['stem']}.{entry['arch']}"] = entry
    graph_files = dict(previous.get("graphs") or {})
    for graph, plan, plan_file in graphs:
        doc = graph_doc(graph, plan, plan_file)
        (root / "graphs").mkdir(exist_ok=True)
        sources: "dict[str, str]" = {}
        hashes: "dict[str, str]" = {}
        for region in doc["regions"].values():  # the files of the graph body the regions' ops were written on
            for op in region["body"]:
                name = op["loc"].rsplit(":", 1)[0] if op.get("loc") else None
                if name is None or name in sources or not _safe_name(name):
                    continue
                data = _source_of(name, Item(None, "graph"))
                if data is not None:
                    rel = f"graphs/source/{name}"
                    (root / rel).parent.mkdir(parents=True, exist_ok=True)
                    write_file(root / rel, data)
                    sources[name] = rel
                    hashes[rel] = hashlib.sha256(data).hexdigest()
        doc["sources"] = sources
        data = _json(doc)
        rel = f"graphs/{graph.graph.name}.json"
        write_file(root / rel, data)
        graph_files[graph.graph.name] = {"file": rel, "sha256": hashlib.sha256(data).hexdigest(), "files": dict(sorted(hashes.items()))}
    index: "dict[str, Any]" = {
        "schema": TRACE_SCHEMA,
        "tilefish": __version__,
        "manifest_digest": manifest.digest,
        "kernels": dict(sorted(kernels.items())),
        "graphs": dict(sorted(graph_files.items())),
    }
    index["trace_digest"] = index_digest(index)
    write_file(root / INDEX_NAME, _json(index))
    return index


__all__ = ["Item", "TRACE_DIR", "TRACE_ENV", "TRACE_SCHEMA", "TilefishTraceWarning", "index_digest", "item_of", "wanted", "write"]
