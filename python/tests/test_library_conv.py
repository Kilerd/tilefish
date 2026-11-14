"""A pinned cuDNN convolution (``impl="cudnn.conv_fwd"``): declared, specified, written to a manifest and read back, planned and emitted, with no GPU.

``@tilefish.library(..., impl="cudnn.conv_fwd", layout=, stride=, padding=, dilation=, groups=, math=, api=, heuristic=, algos=, alignment=,
workspace_bytes=, sm_count_target=)`` over ``(x, w) -> y`` in the layout's order. The declaration is checked against its signature at every bucket
(the declared result is the convolution's output); an entry says the problem, the math, the pin and how it was chosen, the cuDNN version and the
device; the strict reader holds them together; the binding's pure parts (the problem as cuDNN is given it, the pins, the rules a pin is held to,
the refusal of ``NVIDIA_TF32_OVERRIDE``) are checked here. What it does on a GPU is ``test_gpu_conv.py``.
"""

import json

import pytest

import tilefish
from tilefish import CudnnTable, Dim, Manifest, Tensor, TilefishDeclarationError, TilefishManifestError, TilefishRuntimeError, bf16, cdiv, f16, f32
from tilefish import _conv, _cudnn
from tilefish._library import validate_entry
from tilefish.aot import write_manifest

T = Dim("T", buckets=[64, 646])
VERSION = 91002
SMS = 30
ENGINE = _cudnn.EnginePin(28, ((2, 4), (3, 0)))


def conv(name="c", **kw):
    return tilefish.library(name, impl="cudnn.conv_fwd", **kw)


def nhwc_k7(x: Tensor[1, T, 64, f16], w: Tensor[64, 7, 64, f16]) -> Tensor[1, T + 6, 64, f16]: ...


def nchw_down(x: Tensor[2, 32, 100, bf16], w: Tensor[64, 32, 4, bf16]) -> Tensor[2, 64, 50, bf16]: ...


def img_f32(x: Tensor[1, 16, 24, 24, f32], w: Tensor[32, 16, 3, 3, f32]) -> Tensor[1, 32, 12, 12, f32]: ...


def built(op, pin=None, sm_count=SMS):
    out = []
    for spec in _conv.conv_specs(op):
        p = pin or (_cudnn.LegacyPin(1) if spec.api == "legacy" else ENGINE)
        out.append(_conv.BuiltConv(spec, "sm_103", p, VERSION, sm_count))
    return out


# -- the declaration ------------------------------------------------------------------------------------------------------------------------


def test_a_convolution_is_declared_with_its_defaults():
    op = conv(layout="NHWC", padding=6)(nhwc_k7)
    o = op.conv
    assert (op.impl, op.workspace_bytes, op.layout) == ("cudnn.conv_fwd", 64 << 20, "NHWC")
    assert (o.stride, o.padding, o.dilation, o.groups, o.math, o.api, o.heuristic, o.alignment) == ((1,), (6,), (1,), 1, "tensor_op", "backend", "instant", 16)
    legacy = conv(layout="NHWC", padding=6, api="legacy")(nhwc_k7)
    assert (legacy.conv.api, legacy.conv.heuristic) == ("legacy", "v7")
    down = conv(stride=2, padding=1)(nchw_down)
    assert [s.problem.output for s in _conv.conv_specs(down)] == [(50,)]
    img = conv(stride=(2, 2), padding=1, math="fma")(img_f32)
    assert img.conv.padding == (1, 1) and _conv.conv_specs(img)[0].problem.x_dims == (1, 16, 24, 24)


def _bad_out(x: Tensor[1, T, 64, f16], w: Tensor[64, 7, 64, f16]) -> Tensor[1, T, 64, f16]: ...


def _mixed(x: Tensor[1, T, 64, f16], w: Tensor[64, 7, 64, bf16]) -> Tensor[1, T + 6, 64, f16]: ...


def _rank2(x: Tensor[T, 64, f16], w: Tensor[64, 64, f16]) -> Tensor[T, 64, f16]: ...


def _loose(x: Tensor[1, Dim("L"), 64, f16], w: Tensor[64, 7, 64, f16]) -> Tensor[1, Dim("L") + 6, 64, f16]: ...


def _down_cdiv(x: Tensor[1, T, 64, f16], w: Tensor[128, 4, 64, f16]) -> Tensor[1, cdiv(T, 2), 128, f16]: ...


def _gemm(a: Tensor[T, 64, f16], b: Tensor[64, 64, f16]) -> Tensor[T, 64, f16]: ...


def test_the_declaration_is_checked_against_its_signature_at_every_bucket():
    with pytest.raises(TilefishDeclarationError, match=r"the result of the convolution is \[1, 70, 64\] \(at T=64\), and the declaration says \[1, 64, 64\]"):
        conv(layout="NHWC", padding=6)(_bad_out)
    with pytest.raises(TilefishDeclarationError, match="x, w and the result are of one type"):
        conv(layout="NHWC", padding=6)(_mixed)
    with pytest.raises(TilefishDeclarationError, match="1 to 3 spatial axes"):
        conv()(_rank2)
    with pytest.raises(TilefishDeclarationError, match="depends on the dynamic dimension L without buckets"):
        conv(layout="NHWC", padding=6)(_loose)
    assert [s.problem.output for s in _conv.conv_specs(conv(layout="NHWC", stride=2, padding=1)(_down_cdiv))] == [(32,), (323,)]
    with pytest.raises(TilefishDeclarationError, match="f32 data needs math='fma'"):
        conv(stride=2, padding=1)(img_f32)
    with pytest.raises(TilefishDeclarationError, match="f32 data needs the math said explicitly"):
        conv(stride=2, padding=1, math="tensor_op")(img_f32)
    with pytest.raises(TilefishDeclarationError, match="bf16 data has the math 'tensor_op'"):
        conv(stride=2, padding=1, math="tf32")(nchw_down)
    with pytest.raises(TilefishDeclarationError, match=r"stride is an int of at least 1, or one per spatial axis \(2\)"):
        conv(stride=(2, 2, 2), padding=1, math="fma")(img_f32)
    with pytest.raises(TilefishDeclarationError, match=r"the groups \(3\) divide the input channels"):
        conv(stride=2, padding=1, groups=3)(nchw_down)
    with pytest.raises(TilefishDeclarationError, match="the heuristic of the legacy API is one of \\['v7'\\]"):
        conv(stride=2, padding=1, api="legacy", heuristic="instant")(nchw_down)
    with pytest.raises(TilefishDeclarationError, match="the layout of a cudnn.conv_fwd is one of"):
        conv(layout="TN")(nchw_down)
    with pytest.raises(TilefishDeclarationError, match="alignment is one of"):
        conv(stride=2, padding=1, alignment=8)(nchw_down)
    with pytest.raises(TilefishDeclarationError, match="allow_f16_reduction are options of a cuBLASLt GEMM"):
        conv(layout="NHWC", padding=6, allow_f16_reduction=True)(nhwc_k7)
    with pytest.raises(TilefishDeclarationError, match="stride are options of cudnn.conv_fwd, not of a cuBLASLt GEMM"):
        tilefish.library("g", acc=f32, impl="cublaslt.gemm", stride=2)(_gemm)
    with pytest.raises(TilefishDeclarationError, match="need impl='cudnn.conv_fwd'"):
        tilefish.library("x", padding=1)
    with pytest.raises(TilefishDeclarationError, match=r"impl must be one of \['cublaslt.gemm', 'cublaslt.gemm_bias', 'cudnn.conv_fwd'\]"):
        tilefish.library("x", acc=f32, impl="cudnn.conv_bwd")(nhwc_k7)


def test_the_options_reach_the_digest_of_the_declaration():
    base = _conv.declaration_digest(conv(layout="NHWC", padding=6)(nhwc_k7))
    for kw in ({"api": "legacy"}, {"heuristic": "mode_b"}, {"alignment": 32}, {"workspace_bytes": 1 << 20}, {"math": "fma"}, {"sm_count_target": 16}):
        assert _conv.declaration_digest(conv(layout="NHWC", padding=6, **kw)(nhwc_k7)) != base, kw
    assert _conv.declaration_digest(conv(layout="NHWC", padding=6)(nhwc_k7)) == base


# -- imported pins --------------------------------------------------------------------------------------------------------------------------


def test_a_table_imports_the_pins_of_every_entry():
    rows = {(1, 64, 64, (t,), (7,), (1,), (6,), (1,), 1): {"engine": 1, "knobs": {}} for t in (64, 646)}
    table = CudnnTable("sm_103", SMS, VERSION, rows)
    op = conv(layout="NHWC", padding=6, algos=table)(nhwc_k7)
    assert [s.choice for s in _conv.conv_specs(op)] == ["imported", "imported"]
    assert table.pins[(1, 64, 64, (64,), (7,), (1,), (6,), (1,), 1)] == _cudnn.EnginePin(1)
    assert "algos" in _conv._canonical(op)
    with pytest.raises(TilefishDeclarationError, match="has no pin for"):
        conv(layout="NHWC", padding=6, algos=CudnnTable("sm_103", SMS, VERSION, {next(iter(rows)): {"engine": 1, "knobs": {}}}))(nhwc_k7)
    with pytest.raises(TilefishDeclarationError, match="the declaration's api is 'legacy', and the table for sm_103 has a backend pin"):
        conv(layout="NHWC", padding=6, api="legacy", algos=table)(nhwc_k7)
    legacy = CudnnTable("sm_103", SMS, VERSION, {k: 1 for k in rows})
    assert conv(layout="NHWC", padding=6, api="legacy", algos=legacy)(nhwc_k7).algos == (legacy,)
    with pytest.raises(TilefishDeclarationError, match="a key is"):
        CudnnTable("sm_103", SMS, VERSION, {(1, 64, 64, 646, 7): 1})
    with pytest.raises(TilefishDeclarationError, match="sorted by type"):
        CudnnTable("sm_103", SMS, VERSION, {next(iter(rows)): {"engine": 1, "knobs": [[3, 0], [2, 4]]}})


# -- the entry ------------------------------------------------------------------------------------------------------------------------------


def test_the_entry_says_the_problem_the_pin_and_the_device():
    op = conv(layout="NHWC", padding=6)(nhwc_k7)
    e = built(op)[1].entry()
    assert (e["impl"], e["stem"], e["layout"], e["math"], e["compute_type"]) == ("cudnn.conv_fwd", "c.T646", "NHWC", "tensor_op", "CUDNN_DATA_FLOAT")
    assert (e["n"], e["c"], e["k"], e["groups"], e["input"], e["filter"], e["output"]) == (1, 64, 64, 1, [646], [7], [652])
    assert (e["stride"], e["padding"], e["dilation"], e["alignment"], e["workspace_bytes"]) == ([1], [6], [1], 16, 64 << 20)
    assert (e["choice"], e["pin"], e["cudnn_version"], e["sm_count"]) == ("instant", ENGINE.to_json(), VERSION, SMS)
    assert [(p["name"], p["shape"]) for p in e["abi"]["params"]] == [("x", [1, "T", 64]), ("w", [64, 7, 64]), ("out", [1, "T + 6", 64])]
    assert {pre["bytes"] for pre in e["preconditions"] if pre["kind"] == "pointer_alignment"} == {16}
    validate_entry(e)
    e32 = built(conv(layout="NHWC", padding=6, alignment=32)(nhwc_k7))[0].entry()
    assert e32["alignment"] == 32 and {pre["bytes"] for pre in e32["preconditions"] if pre["kind"] == "pointer_alignment"} == {32}
    validate_entry(e32)


def refuses(entry, change, message):
    e = json.loads(json.dumps(entry))
    change(e)
    with pytest.raises(TilefishManifestError, match=message):
        validate_entry(e)


def test_the_reader_holds_the_fields_together():
    e = built(conv(layout="NHWC", padding=6)(nhwc_k7))[1].entry()
    refuses(e, lambda e: e.update(output=[646]), r"output must be the output of the convolution, \[652\]")
    refuses(e, lambda e: e.update(choice="v7"), "the choice 'v7' pins a legacy pin, got a backend one")
    refuses(e, lambda e: e.update(math="tf32"), "not a convolution cuDNN takes: f16 data has the math")
    refuses(e, lambda e: e.update(groups=3), r"the groups \(3\) divide")
    refuses(e, lambda e: e.update(alignment=32), "pointer_alignment must be the entry's alignment")
    refuses(e, lambda e: e["pin"].update(knobs=[[3, 0], [2, 4]]), "sorted by type")
    refuses(e, lambda e: e.update(pin={"api": "legacy", "algo": 8}), "the choice 'instant' pins a backend pin|from 0 to 7")
    refuses(e, lambda e: e.update(sm_count_target=31), "sm_count_target is absent or an int from 1 to sm_count")
    refuses(e, lambda e: e["dtypes"].update(w="f32"), "dtypes must be")
    refuses(e, lambda e: e["abi"]["params"][2].update(shape=[1, "T", 64]), "but the problem says 652")
    refuses(e, lambda e: e.update(extra=1), "must have exactly the fields")
    legacy = built(conv(layout="NHWC", padding=6, api="legacy")(nhwc_k7))[0].entry()
    validate_entry(legacy)
    refuses(legacy, lambda e: e.update(sm_count_target=16), "a legacy pin has an sm_count_target only when it was imported")
    validate_entry({**legacy, "choice": "imported", "sm_count_target": 16})


def test_a_manifest_holds_convolutions_and_a_replay_checks_them(tmp_path, monkeypatch):
    op = conv(layout="NHWC", padding=6)(nhwc_k7)
    write_manifest([], tmp_path / "m" / "manifest.json", cubins=True, library=built(op))
    m = Manifest.load(tmp_path / "m")
    assert sorted(e["stem"] for e in m.library_entries) == ["c.T64", "c.T646"]
    monkeypatch.setattr(_cudnn, "version", lambda: VERSION)
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    for spec in _conv.conv_specs(op):
        got = _conv.built_from_manifest(m, spec, "sm_103", SMS)
        assert got.pin == ENGINE and got.cudnn_version == VERSION
    spec = _conv.conv_specs(op)[0]
    with pytest.raises(TilefishManifestError, match="on a device with 30 SMs, but this device has 148"):
        _conv.built_from_manifest(m, spec, "sm_103", 148)
    monkeypatch.setattr(_cudnn, "version", lambda: VERSION + 1)
    with pytest.raises(TilefishManifestError, match="with cuDNN 91002, but this process has cuDNN 91003"):
        _conv.built_from_manifest(m, spec, "sm_103", SMS)
    monkeypatch.setattr(_cudnn, "version", lambda: VERSION)
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    with pytest.raises(TilefishManifestError, match="NVIDIA_TF32_OVERRIDE is set \\(to '1'\\)"):
        _conv.built_from_manifest(m, spec, "sm_103", SMS)
    other = conv(layout="NHWC", padding=6, math="fma")(nhwc_k7)
    with pytest.raises(TilefishManifestError, match="first difference at math: the manifest has 'tensor_op', the declaration gives 'fma'"):
        _conv.built_from_manifest(m, _conv.conv_specs(other)[0], "sm_103", SMS)


def test_a_plan_launches_convolutions_as_library_steps():
    from tilefish import _plan

    @tilefish.library("k7", impl="cudnn.conv_fwd", layout="NHWC", padding=3)
    def k7(x: Tensor[1, T, 64, f16], w: Tensor[64, 7, 64, f16]) -> Tensor[1, T, 64, f16]: ...

    @tilefish.library("k3", impl="cudnn.conv_fwd", layout="NHWC", padding=1, api="legacy", workspace_bytes=1 << 20)
    def k3(x: Tensor[1, T, 64, f16], w: Tensor[64, 3, 64, f16]) -> Tensor[1, T, 64, f16]: ...

    @tilefish.graph(inputs=("x",), weights=("w1", "w2"))
    def chain(x: Tensor[1, T, 64, f16], w1: Tensor[64, 7, 64, f16], w2: Tensor[64, 3, 64, f16]):
        return k3(k7(x, w1), w2)

    p = _plan.plan(chain, None)
    steps = p.doc["buckets"][0]["steps"]
    assert [(s["kind"], s["entry"]) for s in steps] == [("library", "k7.T64"), ("library", "k3.T64")]
    workspace = next(b for b in p.doc["buffers"] if b["role"] == "workspace")
    assert workspace["bytes"] == 64 << 20 and all(s["workspace"] == workspace["name"] for s in steps)


def test_a_rust_launcher_of_a_convolution(tmp_path):
    from tilefish.hosts import rust

    ops = [conv("k7", layout="NHWC", padding=6)(nhwc_k7), conv("down", stride=2, padding=1, api="legacy")(nchw_down)]
    write_manifest([], tmp_path / "m" / "manifest.json", cubins=True, library=[b for op in ops for b in built(op)])
    m = Manifest.load(tmp_path / "m")
    source = rust.render(m.check_pin(m.digest), ops=ops)
    uses = next(line for line in source.splitlines() if line.startswith("use tilefish::"))
    assert ", Conv, " in uses and "Gemm" not in source
    assert "pub struct K7T646 {\n    conv: Conv,\n}" in source
    assert "manifest.conv(ctx, Self::STEM, Self::BUCKETS, Arch::of_context(ctx)?)?" in source
    assert "`out = conv(x, w)` (NHWC, 1 spatial axis, stride 1, padding 6, dilation 1)" in source
    assert "sm_103: cuDNN 9.10.2 (91002) on 30 SMs, legacy algorithm 1" in source
    assert "engine 28 with knobs [[2, 4], [3, 0]]" in source


def test_the_committed_conv_vectors_are_what_python_decides_now():
    """``crates/tilefish/tests/data/python_conv.json`` is what the strict reader decides now; ``tests/python_conv.rs`` holds the Rust reader to it."""
    from conftest import REPO_ROOT
    from export_rust_vectors import conv_entries

    committed = json.loads((REPO_ROOT / "crates" / "tilefish" / "tests" / "data" / "python_conv.json").read_text())
    assert committed == json.loads(json.dumps(conv_entries()))


# -- the binding's pure parts ---------------------------------------------------------------------------------------------------------------


def test_a_problem_is_posed_as_cudnn_takes_it():
    p = _cudnn.Problem("f16", "NHWC", 1, 64, 64, 1, (646,), (7,), (1,), (6,), (1,), "tensor_op")
    d = _cudnn.describe(p)
    assert (d["x_dims"], d["x_strides"], d["w_dims"], d["w_strides"]) == ([1, 64, 1, 646], [41344, 1, 41344, 64], [64, 64, 1, 7], [448, 1, 448, 64])
    assert (d["y_dims"], d["padding"], d["stride"], d["dilation"]) == ([1, 64, 1, 652], [0, 6], [1, 1], [1, 1])
    assert (d["data_type"], d["format"], d["math_type"], d["uids"]) == ("CUDNN_DATA_HALF", "CUDNN_TENSOR_NHWC", "CUDNN_TENSOR_OP_MATH", {"x": 120, "w": 119, "y": 121})
    v = _cudnn.Problem("f32", "NCHW", 1, 8, 16, 2, (6, 6, 6), (3, 3, 3), (1, 2, 1), (1, 1, 1), (1, 1, 2), "tf32")
    dv = _cudnn.describe(v)
    assert (dv["x_dims"], dv["w_dims"], dv["y_dims"], dv["math_type"]) == ([1, 8, 6, 6, 6], [16, 4, 3, 3, 3], [1, 16, 6, 3, 4], "CUDNN_DEFAULT_MATH")
    assert _cudnn.dense_strides((2, 3, 4, 5), "NHWC") == (60, 1, 15, 3) and _cudnn.dense_strides((2, 3, 4, 5), "NCHW") == (60, 20, 5, 1)
    assert _cudnn.output_size(646, 4, 2, 1, 1) == 323 and _cudnn.output_size(3, 7, 1, 0, 1) == 0
    for args, message in [
        (("f16", "NHWC", 1, 64, 64, 1, (6,), (7,), (1,), (0,), (1,), "tensor_op"), "spatial axis 0 has no output"),
        (("f16", "NHWC", 1, 64, 64, 3, (64,), (7,), (1,), (3,), (1,), "tensor_op"), r"the groups \(3\) divide"),
        (("f8", "NHWC", 1, 64, 64, 1, (64,), (7,), (1,), (3,), (1,), "tensor_op"), "the data type is one of"),
        (("f32", "NHWC", 1, 64, 64, 1, (64,), (7,), (1,), (3,), (1,), "tensor_op"), "f32 data needs the math said explicitly"),
        (("f16", "NHWC", 1, 64, 64, 1, (64,), (7, 1), (1,), (3,), (1,), "tensor_op"), "filter must be 1 ints"),
    ]:
        with pytest.raises(TilefishRuntimeError, match=message):
            _cudnn.Problem(*args)


def test_the_pins_and_the_rules_they_are_held_to():
    assert _cudnn.pin_of(1) == _cudnn.LegacyPin(1)
    assert _cudnn.pin_of({"engine": 5, "knobs": {3: 0, 2: 4}}) == _cudnn.EnginePin(5, ((2, 4), (3, 0)))
    assert _cudnn.pin_of({"api": "backend", "engine": 5, "knobs": [[2, 4]]}).to_json() == {"api": "backend", "engine": 5, "knobs": [[2, 4]]}
    for bad, message in [(8, "from 0 to 7"), ({"api": "legacy", "algo": 1, "x": 0}, "exactly the keys"), ({"engine": -1, "knobs": []}, "from 0"), (True, "is {")]:
        with pytest.raises(TilefishRuntimeError, match=message):
            _cudnn.pin_of(bad)
    assert _cudnn.notes_problem([0, 4], "tensor_op") == "it is not deterministic (the numerical note NONDETERMINISTIC)"
    assert _cudnn.notes_problem([1], "tensor_op") == "it converts its inputs down (the numerical note DOWN_CONVERT_INPUTS)"
    assert _cudnn.notes_problem([0], "fma") == "it uses tensor cores (the numerical note TENSOR_CORE), and the math is 'fma'"
    assert _cudnn.notes_problem([0, 5], "tf32") is None
    assert _cudnn.knobs_problem(((2, 4),), {2: (0, 4, 1), 3: (0, 1, 1)}) is None
    assert _cudnn.knobs_problem(((2, 5),), {2: (0, 4, 1)}) == "knob 2 = 5 is outside the engine's 0..4 (stride 1)"
    assert _cudnn.knobs_problem(((2, 3),), {2: (0, 4, 2)}) == "knob 2 = 3 is outside the engine's 0..4 (stride 2)"
    assert _cudnn.knobs_problem(((7, 0),), {3: (0, 1, 1), 2: (0, 4, 1)}) == "knob type 7 is not a knob of the engine (it has [2, 3])"
    _cudnn.refuse_tf32_override({})
    with pytest.raises(TilefishRuntimeError, match="NVIDIA_TF32_OVERRIDE is set \\(to ''\\) in this process"):
        _cudnn.refuse_tf32_override({"NVIDIA_TF32_OVERRIDE": ""})


# -- the canary -------------------------------------------------------------------------------------------------------------------------------


def _values(dtype, raw):
    """The floats of canary bytes (little-endian)."""
    import struct

    if dtype == "f16":
        return struct.unpack(f"<{len(raw) // 2}e", raw)
    if dtype == "f32":
        return struct.unpack(f"<{len(raw) // 4}f", raw)
    halves = struct.unpack(f"<{len(raw) // 2}H", raw)
    return struct.unpack(f"<{len(halves)}f", struct.pack(f"<{len(halves)}I", *(h << 16 for h in halves)))


def test_the_canary_operands_are_normal_numbers_of_many_magnitudes_made_of_the_seed():
    import math

    for dtype, low, high in (("f16", 2.0**-8, 1.0), ("bf16", 2.0**-17, 0.5), ("f32", 2.0**-17, 0.5)):
        raw = _conv.canary_bytes(dtype, 1, 0, 4096)
        assert len(raw) == 4096 * {"f16": 2, "bf16": 2, "f32": 4}[dtype]
        values = _values(dtype, raw)
        assert all(math.isfinite(v) and low <= abs(v) < high for v in values), dtype
        assert len({math.frexp(abs(v))[1] for v in values}) == (8 if dtype == "f16" else 16), dtype  # every exponent of the range
        assert 0.4 < sum(v < 0 for v in values) / len(values) < 0.6, dtype
        assert raw == _conv.canary_bytes(dtype, 1, 0, 4096)
        size = len(raw) // 4096
        assert raw[: 100 * size] == _conv.canary_bytes(dtype, 1, 0, 100)  # a prefix of the stream: the size does not change the first elements
        assert raw != _conv.canary_bytes(dtype, 2, 0, 4096) and raw != _conv.canary_bytes(dtype, 1, 1, 4096)
    p = _cudnn.Problem("f16", "NHWC", 1, 64, 64, 1, (646,), (7,), (1,), (3,), (1,), "tensor_op")
    x, w = _conv.canary_operands(p, 5)
    assert (len(x), len(w)) == (646 * 64 * 2, 64 * 7 * 64 * 2)
    assert x == _conv.canary_bytes("f16", 5, 0, 646 * 64) and w == _conv.canary_bytes("f16", 5, 1, 64 * 7 * 64)


def test_an_entry_may_have_a_canary_and_the_reader_holds_its_form():
    op = conv(layout="NHWC", padding=6)(nhwc_k7)
    spec = _conv.conv_specs(op)[0]
    plain = _conv.BuiltConv(spec, "sm_103", ENGINE, VERSION, SMS).entry()
    assert "canary" not in plain
    validate_entry(plain)
    e = _conv.BuiltConv(spec, "sm_103", ENGINE, VERSION, SMS, {"seed": 1, "y_digest": "0f" * 16}).entry()
    assert e["canary"] == {"seed": 1, "y_digest": "0f" * 16}
    validate_entry(e)
    validate_entry({**e, "canary": {"seed": (1 << 32) - 1, "y_digest": "0f" * 16}})
    for bad in (None, [], {"seed": 1}, {"seed": -1, "y_digest": "0f" * 16}, {"seed": 1 << 32, "y_digest": "0f" * 16}, {"seed": True, "y_digest": "0f" * 16},
                {"seed": 1, "y_digest": "0F" * 16}, {"seed": 1, "y_digest": "0f" * 16, "x": 1}):  # fmt: skip
        refuses(e, lambda e, bad=bad: e.update(canary=bad), "canary is absent or")
    refuses(e, lambda e: e.update(canary={"seed": 1, "y_sha256": "0f" * 32}), "a canary of tilefish's first form .* is no longer read: export the manifest again")


def test_the_canary_ptx_is_the_crates():
    """The PTX of the canary's kernels is one text in both hosts: the Python package's constant and the crate's ``src/canary.ptx``."""
    from conftest import REPO_ROOT

    from tilefish import _canary

    assert _canary.CANARY_PTX == (REPO_ROOT / "crates" / "tilefish" / "src" / "canary.ptx").read_text()
    assert ".entry tilefish_canary_fill(" in _canary.CANARY_PTX and ".entry tilefish_canary_digest(" in _canary.CANARY_PTX


def test_the_canary_digest_is_a_sum_of_words_that_any_one_change_changes():
    from tilefish import _canary

    data = _conv.canary_bytes("f16", 3, 0, 1000)
    d = _canary.digest_of(data)
    assert len(d) == 32 and d == _canary.digest_of(data + b"\x00\x00")  # zero padding to a whole word
    for at in (0, 1, 2, 3, 999, 1998, 1999):  # one changed byte, wherever
        changed = bytearray(data)
        changed[at] ^= 0x01
        assert _canary.digest_of(bytes(changed)) != d, at
    words = data[:400]
    split = _canary.digest_of(words[:200] + b"\x00" * 200)
    rest = _canary.digest_of(b"\x00" * 200 + words[200:])
    s1 = (int(split[:16], 16) + int(rest[:16], 16)) % 2**64
    s2 = (int(split[16:], 16) + int(rest[16:], 16)) % 2**64
    assert _canary.hex_of(s1, s2) == _canary.digest_of(words)  # a sum: the device adds it in any order


def test_two_canaries_that_differ_are_two_binaries_and_an_entry_without_one_is_replaced():
    from tilefish._build import _conflicts

    op = conv(layout="NHWC", padding=6)(nhwc_k7)
    spec = _conv.conv_specs(op)[0]
    old = _conv.BuiltConv(spec, "sm_103", ENGINE, VERSION, SMS).entry()
    new = _conv.BuiltConv(spec, "sm_103", ENGINE, VERSION, SMS, {"seed": 1, "y_digest": "0f" * 16}).entry()
    assert _conflicts(old, new) is None and _conflicts(new, new) is None
    assert _conflicts(new, {**new, "canary": {"seed": 1, "y_digest": "1f" * 16}}) == "canary"


def test_a_replay_runs_the_canary_once_and_refuses_other_bytes(tmp_path, monkeypatch):
    import contextlib
    import types

    from tilefish import _library

    op = conv(layout="NHWC", padding=6)(nhwc_k7)
    specs = _conv.conv_specs(op)
    good = "0f" * 16
    library = [_conv.BuiltConv(specs[0], "sm_103", ENGINE, VERSION, SMS, {"seed": 3, "y_digest": good}), _conv.BuiltConv(specs[1], "sm_103", ENGINE, VERSION, SMS)]
    write_manifest([], tmp_path / "m" / "manifest.json", cubins=True, library=library)
    m = Manifest.load(tmp_path / "m")
    capturing = [False]
    fake_cuda = types.SimpleNamespace(device=lambda i: contextlib.nullcontext(), is_current_stream_capturing=lambda: capturing[0])
    monkeypatch.setattr(_cudnn, "_torch", lambda: types.SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(_cudnn, "version", lambda: VERSION)
    monkeypatch.setattr(_library, "_device_index", lambda i: 0 if i is None else i)
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    monkeypatch.setattr(_conv, "_canaries_passed", set())
    runs = []
    answer = [good]
    monkeypatch.setattr(_conv, "canary_digest", lambda spec, pin, seed, index: runs.append((spec.stem, seed, index)) or answer[0])
    capturing[0] = True
    with pytest.raises(TilefishRuntimeError, match="runs its canary at its first replay in the process, and this one is inside a CUDA graph capture"):
        _conv.built_from_manifest(m, specs[0], "sm_103", SMS)
    capturing[0] = False
    for _ in range(3):
        got = _conv.built_from_manifest(m, specs[0], "sm_103", SMS)
        assert got.canary == {"seed": 3, "y_digest": good}
        assert _conv.built_from_manifest(m, specs[1], "sm_103", SMS).canary is None  # none recorded: nothing runs
    assert runs == [("c.T64", 3, 0)]  # once per process and device
    _conv.built_from_manifest(m, specs[0], "sm_103", SMS, device_index=1)
    assert runs[-1] == ("c.T64", 3, 1)
    answer[0] = "1f" * 16
    with pytest.raises(TilefishManifestError, match=f"the canary of the cuDNN convolution 'c.T64' .* gives y with digest {'1f' * 16} here, and the manifest recorded {good} \\(seed 3\\)"):
        _conv.built_from_manifest(m, specs[0], "sm_103", SMS, device_index=2)
