"""The trace of an export (:mod:`tilefish._trace`, ``export(..., trace=True)``) and ``tilefish-explain`` (:mod:`tilefish.explain`), with no Triton,
no GPU and no CUDA tool: ``fake_compile`` builds, and the disassembler is replaced by a listing written here from the generated source's lines (as
``nvdisasm -c -g`` prints one). The real thing, on sm_103: ``test_gpu_sass.py``.
"""

import hashlib
import json
import pathlib
import sys

import fake_compile
import pytest

import tilefish
from tilefish import Dim, Tensor, TilefishBuildError, _fusion, _sass, _trace, bf16, explain, f32, runtime

ARCH = 103
M = Dim("M", divisible_by=16)
R = Dim("R", buckets=[64])
HERE = pathlib.Path(__file__).resolve().parent


@pytest.fixture
def fake(tmp_path, monkeypatch):
    compiler = fake_compile.install(monkeypatch)
    monkeypatch.setattr(runtime, "_modules", {})
    monkeypatch.setattr(runtime, "_gen_root", [])
    monkeypatch.setattr(runtime, "_warned_gen_roots", set())
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.delenv(_trace.TRACE_ENV, raising=False)
    return compiler


@tilefish.kernel
def ex_scale(x: Tensor[M, f32]) -> Tensor[M, f32]:
    return x * 2.0  # (the line the mul of the SASS below is traced back to)


SCALE_LINE = 1 + next(i for i, line in enumerate((HERE / "test_explain.py").read_text().splitlines()) if "return x * 2.0  #" in line)


@tilefish.graph
def ex_graph(x: Tensor[R, 256, bf16], w: Tensor[R, 256, bf16]) -> Tensor[R, 256, bf16]:
    h = x.to(f32) * w.to(f32)
    return (h + 1.0).to(bf16)


def line_of(low, op):
    """The line of the generated source of ``low`` written for the first node of ``op`` (and not a helper's)."""
    return next(k for k, m in enumerate(low.line_map, 1) if m and m.get("op") == op and "helper" not in m)


def listing(low, extra=()):
    """A listing of ``tf_k`` (the fake compiler's symbol) whose line information names ``low``'s generated source: a load, a multiply, a store."""
    file = low.filename
    load = next(k for k, m in enumerate(low.line_map, 1) if m and m.get("what", "").startswith("load"))
    store = next(k for k, m in enumerate(low.line_map, 1) if m and m.get("what", "").startswith("return value"))
    body = [
        (load, "LDG.E R2, desc[UR4][R2.64]"),
        (line_of(low, "mul"), "FMUL R5, R2, 2"),
        *extra,
        (store, "STG.E desc[UR4][R6.64], R5"),
        (store, "EXIT"),
    ]
    out = ["\t.target\tsm_103a", '\t.section\t.text.tf_k,"ax",@progbits']
    for k, (line, ins) in enumerate(body):
        out.append(f'\t//## File "{file}", line {line}')
        out.append(f"        /*{k * 16:04x}*/                   {ins} ;")
    return "\n".join(out) + "\n"


def use_listing(monkeypatch, text, calls=None):
    def disassemble(cubin, *, ptx=None, options=None, prebuilt=False):
        if calls is not None:
            calls.append(hashlib.sha256(cubin).hexdigest())
        return _sass.Disassembly(text, "", _sass.parse_listing(text), "sm_103a", False, "/x/nvdisasm", "13.5.37", "sm_103a")

    monkeypatch.setattr(_sass, "disassemble", disassemble)


def test_a_traced_export_writes_every_level_next_to_the_same_manifest(fake, tmp_path, monkeypatch):
    use_listing(monkeypatch, listing(ex_scale.lower()))
    plain = runtime.export(ex_scale, tmp_path / "plain", arch=ARCH)
    traced = runtime.export(ex_scale, tmp_path / "traced", arch=ARCH, trace=True)
    assert plain.digest == traced.digest  # outside the pinned artefact: the manifest is the same, byte for byte
    for name in ("manifest.json", "ex_scale.sm_103.cubin"):
        assert (plain.directory / name).read_bytes() == (traced.directory / name).read_bytes()
    assert not (plain.directory / "trace").exists()
    index = json.loads((traced.directory / "trace" / "index.json").read_text())
    assert index["schema"] == "tilefish.trace.v1" and index["manifest_digest"] == traced.digest
    assert index["trace_digest"] == _trace.index_digest(index)
    entry = index["kernels"]["ex_scale.sm_103"]
    assert entry["origin"] == "kernel" and entry["symbol"] == "tf_k" and entry["sha256"] == traced.get("ex_scale", "sm_103").entry["sha256"]
    assert entry["sass"] == {"target": "sm_103a", "proxy": False, "nvdisasm": "13.5.37", "label": "its SASS, read by nvdisasm 13.5.37"}
    low = ex_scale.lower()
    assert set(entry["files"]) >= {"ttgir.mlir", "ptx.ptx", "sass.txt", "lines.json", "nodes.json", f"source/{low.filename}", "source/test_explain.py"}
    base = traced.directory / "trace" / "ex_scale.sm_103"
    for rel, digest in entry["files"].items():
        assert hashlib.sha256((base / rel).read_bytes()).hexdigest() == digest
    assert (base / "source" / low.filename).read_text() == low.source
    nodes = json.loads((base / "nodes.json").read_text())
    assert nodes["lines"][str(line_of(low, "mul"))] == {"node": 0, "op": "mul", "loc": f"test_explain.py:{SCALE_LINE}"}
    table = json.loads((base / "lines.json").read_text())
    assert table["files"] == [low.filename] and table["functions"]["tf_k"][1] == [16, 0, line_of(low, "mul"), 0]


def test_explain_maps_each_instruction_to_the_generated_line_the_op_and_the_line_of_the_body(fake, tmp_path, monkeypatch):
    low = ex_scale.lower()
    use_listing(monkeypatch, listing(low))
    runtime.export(ex_scale, tmp_path / "out", arch=ARCH, trace=True)
    monkeypatch.setattr(_sass, "disassemble", None)  # explain reads the trace: no disassembler is needed
    text = explain.explain(tmp_path / "out", "ex_scale", source_roots=[tmp_path / "nowhere"])
    assert text.startswith("ex_scale on sm_103 (kernel): its SASS, read by nvdisasm 13.5.37\n")
    mul = f"{low.filename}:{line_of(low, 'mul')}  v1 = v0 * 2.0    [node 0 mul, written at test_explain.py:{SCALE_LINE}: return x * 2.0  # (the line"
    assert mul in text
    assert "    /*0010*/ FMUL R5, R2, 2 ;  [float]" in text
    assert "[load x]" in text and "[return value 0]" in text
    rows = explain.load(tmp_path / "out", "ex_scale").lines()
    assert [r["count"] for r in rows] == [1, 1, 2] and rows[1]["written_for"]["op"] == "mul" and rows[2]["columns"] == {"gmem": 1, "alu": 1}
    s = explain.load(tmp_path / "out", "ex_scale").summary()
    assert s["classes"] == {"control": 1, "float": 1, "global_load": 1, "global_store": 1} and s["kernel"] == "ex_scale"
    assert "instructions: 4" in explain.explain(tmp_path / "out", "ex_scale", view="summary")
    assert f"[mul @ test_explain.py:{SCALE_LINE}]" in explain.explain(tmp_path / "out", "ex_scale", view="lines")


def test_the_command_line(fake, tmp_path, monkeypatch, capsys):
    use_listing(monkeypatch, listing(ex_scale.lower()))
    runtime.export(ex_scale, tmp_path / "out", arch=ARCH, trace=True)
    assert explain.main([str(tmp_path / "out"), "ex_scale", "--view", "summary", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["instructions"] == 4 and doc["sass"] == "its SASS, read by nvdisasm 13.5.37"
    assert explain.main([str(tmp_path / "out"), "nope"]) == 2
    assert "the manifest has no kernel 'nope': ['ex_scale']" in capsys.readouterr().err
    assert explain.main([str(tmp_path / "out"), "ex_scale", "--arch", "sm_120"]) == 2
    assert "'ex_scale' is not built for sm_120" in capsys.readouterr().err


def test_a_changed_trace_is_refused_and_a_trace_of_another_binary_is_not_used(fake, tmp_path, monkeypatch):
    use_listing(monkeypatch, listing(ex_scale.lower()))
    out = tmp_path / "out"
    runtime.export(ex_scale, out, arch=ARCH, trace=True)
    sass = out / "trace" / "ex_scale.sm_103" / "sass.txt"
    original = sass.read_text()
    sass.write_text(original.replace("FMUL", "FADD"))
    with pytest.raises(explain.TilefishExplainError, match="is not the one its index records"):
        explain.load(out, "ex_scale")
    sass.write_text(original)
    index = out / "trace" / "index.json"
    doc = json.loads(index.read_text())
    doc["manifest_digest"] = "0" * 64
    index.write_text(json.dumps(doc))
    with pytest.raises(explain.TilefishExplainError, match="does not have the digest it records"):
        explain.load(out, "ex_scale")
    # the binary is exported again without a trace: the old trace is of another binary, and the cubin itself is read
    runtime.export(ex_scale, out, arch=ARCH, trace=True)

    @tilefish.kernel(name="ex_scale")
    def other(x: Tensor[M, f32]) -> Tensor[M, f32]:
        return x * 3.0

    calls = []
    use_listing(monkeypatch, listing(other.lower()), calls)
    runtime.export(other, out, arch=ARCH, overwrite=True)
    e = explain.load(out, "ex_scale")
    assert not e.traced and len(calls) == 1 and "(no trace: the cubin of the manifest was disassembled" in e.annotate()


def test_the_environment_asks_for_a_trace(fake, tmp_path, monkeypatch):
    use_listing(monkeypatch, listing(ex_scale.lower()))
    monkeypatch.setenv(_trace.TRACE_ENV, "1")
    runtime.export(ex_scale, tmp_path / "a", arch=ARCH)
    assert (tmp_path / "a" / "trace" / "index.json").is_file()
    runtime.export(ex_scale, tmp_path / "b", arch=ARCH, trace=False)
    assert not (tmp_path / "b" / "trace").exists()
    monkeypatch.setenv(_trace.TRACE_ENV, "yes")
    with pytest.raises(TilefishBuildError, match="TILEFISH_TRACE='yes'"):
        runtime.export(ex_scale, tmp_path / "c", arch=ARCH)


def test_a_plan_step_and_a_fused_region_are_traced_back_to_the_ops_of_the_graph_body(fake, tmp_path, monkeypatch):
    (region,) = _fusion.fused(ex_graph).regions
    low = region.kernel.lower()
    use_listing(monkeypatch, listing(low))
    plan = runtime.export(ex_graph, tmp_path / "g", arch=ARCH, trace=True, verify=False)
    gdoc = json.loads((tmp_path / "g" / "trace" / "graphs" / "ex_graph.json").read_text())
    assert gdoc["buckets"][0]["steps"][0]["entry"] == region.kernel.name
    body = gdoc["regions"][region.kernel.name]["body"]
    assert [o["op"] for o in body] == ["cast", "cast", "mul", "add", "cast"] and body[2]["loc"].startswith("test_explain.py:")
    assert plan.digest == runtime.export(ex_graph, tmp_path / "plain", arch=ARCH, verify=False).digest
    text = explain.explain(tmp_path / "g", graph="ex_graph", step=0)
    assert f"step 0 of the plan of 'ex_graph' at {{'R': 64}}: launches '{region.kernel.name}'" in text
    assert "fused region 0 of the ops #0 cast (test_explain.py:" in text
    mul = [line for line in text.splitlines() if line.startswith(f"{low.filename}:{line_of(low, 'mul')}")]
    assert mul and "graph op #2 mul at test_explain.py:" in mul[0] and "h = x.to(f32) * w.to(f32)" in mul[0]
    assert "fused region" in explain.explain(tmp_path / "g", graph="ex_graph", region=0)
    with pytest.raises(explain.TilefishExplainError, match="has steps 0 to 0, not 3"):
        explain.load(tmp_path / "g", graph="ex_graph", step=3)


def test_the_diff_of_two_exports(fake, tmp_path, monkeypatch):
    low = ex_scale.lower()
    use_listing(monkeypatch, listing(low))
    runtime.export(ex_scale, tmp_path / "a", arch=ARCH, trace=True)
    use_listing(monkeypatch, listing(low, extra=[(line_of(low, "mul"), "FMUL R5, R5, 2"), (line_of(low, "mul"), "STL [R1], R5")]))
    runtime.export(ex_scale, tmp_path / "b", arch=ARCH, trace=True)
    text = explain.diff(explain.load(tmp_path / "a", "ex_scale"), explain.load(tmp_path / "b", "ex_scale"))
    assert "instructions: 4 -> 6" in text and "classes.local: 0 -> 1" in text and "spills.stores: 0 -> 1" in text
    assert f"      1 -> 3     {low.filename}:{line_of(low, 'mul')}  alu 1->2, local 0->1  | v1 = v0 * 2.0" in text


def test_the_lines_of_a_row_walked_in_chunks_are_mapped():
    import argmax_cases as ac

    low = ac.argmax_bf16.lower()
    lines = low.source.split("\n")
    made = {lines[k - 1].strip(): m for k, m in enumerate(low.line_map, 1) if m}
    store = next(text for text in made if text.startswith("tl.store(") and "return value 0" in text)
    assert made[store] == {"what": "return value 0"}
    combine = next(text for text in made if text.startswith("v0_bv, v0_bi = _tf_argmax_float("))
    assert made[combine]["op"] == "argmax"


def test_the_trace_copies_only_the_files_of_loaded_modules(tmp_path, monkeypatch):
    (tmp_path / "ex_not_loaded.py").write_text("X = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    item = _trace.Item(None, "extern")
    assert _trace._source_of("ex_not_loaded.py", item) is None  # on sys.path and in the working directory, but never imported: not searched for
    assert _trace._source_of("test_explain.py", item) == (HERE / "test_explain.py").read_bytes()  # a module this process loaded
    assert _trace._source_of("test_explain.py", _trace.Item(None, "binary")) is None  # a prebuilt PTX's files are another build's


def test_a_graph_trace_of_another_plan_or_outside_the_trace_is_not_used(fake, tmp_path, monkeypatch):
    (region,) = _fusion.fused(ex_graph).regions
    use_listing(monkeypatch, listing(region.kernel.lower()))
    plan = runtime.export(ex_graph, tmp_path / "g", arch=ARCH, trace=True, verify=False)
    trace = explain.Trace.load(tmp_path / "g")
    assert trace.graph("ex_graph", plan.digest)["plan_digest"] == plan.digest
    assert trace.graph("ex_graph", "0" * 64) is None  # (a later export of another plan of the graph)
    index_path = tmp_path / "g" / "trace" / "index.json"
    index = json.loads(index_path.read_text())
    index["graphs"]["ex_graph"]["file"] = "../ex_graph.json"
    index["trace_digest"] = _trace.index_digest(index)
    index_path.write_text(json.dumps(index))
    with pytest.raises(explain.TilefishExplainError, match="is not the one its index records"):
        explain.Trace.load(tmp_path / "g").graph("ex_graph")
