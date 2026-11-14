"""Pinned cuDNN convolutions (``impl="cudnn.conv_fwd"``) on a real GPU.

* the backend posture ``"instant"`` pins what torch runs: the result is ``torch.nn.functional.conv1d`` / ``conv2d`` / ``conv3d`` byte for byte with
  ``torch.backends.cudnn.deterministic = True``, ``benchmark = False`` and TF32 off (NHWC through ``conv2d`` of channels-last tensors), over 1-D,
  2-D and 3-D problems of f16, bf16 and f32 with strides, paddings, dilations and groups, three operand sets each (asserted), where torch runs cuDNN
  (``torch._C._select_conv_backend``); the pin is an engine configuration whose execution plan is the heuristic's own (their JSON is compared when it is
  pinned); a problem torch runs otherwise, or 16-bit data with the math ``"fma"`` (no torch setting), is held to the f64 result instead;
* a legacy pin of ``CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`` is what torch's legacy path runs (``TORCH_CUDNN_V8_API_DISABLED=1``, a process
  of its own: its deterministic default), f16 and f32;
* a pin gives the same bytes run after run and in another process;
* ``jit`` and ``replay`` agree byte for byte for both APIs, the reference (f64, one rounding) is close, and a plan of two convolutions runs the same
  in ``jit``, ``replay``, a CUDA graph capture and alone;
* an imported pin (``CudnnTable``) is the heuristic's; refusals: another cuDNN version, another SM count, ``NVIDIA_TF32_OVERRIDE`` set, a pin of
  the other API, an engine the math does not allow, knobs out of range, a legacy algorithm cuDNN does not answer as deterministic;
* export records a canary per entry, a replay runs it (on a stream of its own: it passes with the NULL stream busy, which is still busy after it) and
  refuses other bytes, and an entry without one runs; a call of either API captured in a CUDA graph in the global mode on a non-blocking stream writes
  nothing until the replay, which is the eager bytes (nothing of it on the NULL stream or another), and so do a replay of the op and of a plan.
"""

import json
import os
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import conv_kernels as ck  # noqa: E402
from gpu_util import differing, randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import CudnnTable, TilefishBuildError, TilefishManifestError, TilefishRuntimeError, runtime  # noqa: E402
from tilefish import _conv, _cudnn  # noqa: E402

F = torch.nn.functional
DT = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True, scope="module")
def _torch_deterministic():
    torch.zeros(1, device="cuda")
    before = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32 = True, False, False
    yield
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32 = before


def tensors(p, seed):
    fan = p.c // p.groups
    for r in p.filter:
        fan *= r
    x = randn(p.tilefish_shape("x"), DT[p.dtype], seed=seed)
    w = randn(p.tilefish_shape("w"), DT[p.dtype], seed=seed + 1, scale=1 / fan**0.5)
    return x, w


def special(x, kind, seed):
    """``x`` made into values of many exponents (``"wide"``) or into powers of two with 3% of them +-2**8 (``"traps"``: sums that cancel)."""
    if kind == "normal":
        return x
    g = torch.Generator().manual_seed(seed)
    v = x.float().cpu()
    sign = torch.where(v < 0, -1.0, 1.0)
    if kind == "wide":
        v = v * torch.pow(2.0, torch.randint(-8, 5, v.shape, generator=g).float())
    else:
        v = sign * torch.pow(2.0, torch.randint(-6, 2, v.shape, generator=g).float())
        v = torch.where(torch.rand(v.shape, generator=g) < 0.03, sign * 2.0**8, v)
    return v.to(x.dtype).to("cuda")


def torch_conv(p, x, w):
    """torch's convolution of the tilefish tensors: conv1d/2d/3d of NCHW ones; for NHWC conv2d/conv3d of channels-last views (a 1-D one as
    [N, C, 1, L], as torch itself poses conv1d)."""
    if p.layout == "NCHW":
        fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[p.spatial]
        return fn(x, w, None, p.stride, p.padding, p.dilation, p.groups)
    if p.spatial == 1:
        xt, wt = x.unsqueeze(1).permute(0, 3, 1, 2), w.unsqueeze(1).permute(0, 3, 1, 2)
        y = F.conv2d(xt, wt, None, (1, p.stride[0]), (0, p.padding[0]), (1, p.dilation[0]), p.groups)
        return y.permute(0, 2, 3, 1).squeeze(1)
    perm, back = ((0, 3, 1, 2), (0, 2, 3, 1)) if p.spatial == 2 else ((0, 4, 1, 2, 3), (0, 2, 3, 4, 1))
    fn = F.conv2d if p.spatial == 2 else F.conv3d
    return fn(x.permute(*perm), w.permute(*perm), None, p.stride, p.padding, p.dilation, p.groups).permute(*back)


def ours(p, pin, x, w, workspace_bytes=64 << 20, target=None):
    y = torch.empty(p.tilefish_shape("y"), dtype=DT[p.dtype], device="cuda")
    _cudnn.convolution(
        x.data_ptr(), w.data_ptr(), y.data_ptr(), p, pin, workspace_bytes=workspace_bytes, stream=torch.cuda.current_stream().cuda_stream,
        sm_count_target=target,
    )  # fmt: skip
    return y


# (dtype, layout, n, c, k, groups, input, filter, stride, padding, dilation, math)
PROBLEMS = [
    ("f16", "NHWC", 1, 64, 64, 1, (646,), (7,), (1,), (3,), (1,), "tensor_op"),
    ("f16", "NHWC", 1, 64, 64, 1, (646,), (7,), (1,), (9,), (3,), "tensor_op"),
    ("f16", "NHWC", 1, 64, 64, 1, (2048,), (7,), (1,), (27,), (9,), "tensor_op"),
    ("f16", "NHWC", 1, 64, 128, 1, (646,), (4,), (2,), (1,), (1,), "tensor_op"),
    ("f16", "NHWC", 1, 512, 1024, 1, (82,), (16,), (8,), (4,), (1,), "tensor_op"),
    ("f16", "NHWC", 1, 1, 64, 1, (2048,), (7,), (1,), (6,), (1,), "tensor_op"),
    ("f16", "NHWC", 1, 1024, 1024, 1, (81,), (3,), (1,), (2,), (1,), "tensor_op"),
    ("f16", "NCHW", 2, 64, 64, 1, (1000,), (7,), (1,), (3,), (1,), "tensor_op"),
    ("f16", "NCHW", 1, 64, 64, 64, (300,), (7,), (1,), (3,), (1,), "tensor_op"),
    ("bf16", "NCHW", 1, 96, 192, 1, (500,), (3,), (1,), (1,), (1,), "tensor_op"),
    ("f16", "NHWC", 2, 32, 64, 1, (40, 40), (3, 3), (1, 1), (1, 1), (1, 1), "tensor_op"),
    ("f16", "NCHW", 2, 32, 64, 2, (33, 47), (3, 5), (2, 1), (1, 2), (1, 1), "tensor_op"),
    ("f32", "NCHW", 1, 32, 64, 1, (777,), (5,), (1,), (2,), (1,), "fma"),
    ("f32", "NHWC", 1, 16, 32, 1, (24, 24), (3, 3), (2, 2), (1, 1), (1, 1), "fma"),
    ("f32", "NCHW", 1, 16, 16, 2, (10, 10, 10), (3, 3, 3), (2, 1, 1), (1, 0, 1), (1, 1, 2), "fma"),
]


# problems with no torch bytes of cuDNN to compare with: torch runs them without cuDNN (torch._C._select_conv_backend: a bf16 depthwise one is
# CudaDepthwise2d, an f16 3-D one SlowDilated3d on sm_103), or 16-bit data with the math "fma", which torch has no setting for (it filters tensor-core
# engines for f32 only: measured with cuDNN 9.13.0 on sm_103, where the first engine of f16 is a tensor-core one). A pin of each runs, repeats its
# bytes and is the f64 result within one accumulation and one rounding
OTHER_PROBLEMS = [
    ("f16", "NHWC", 1, 64, 64, 1, (646,), (7,), (1,), (3,), (1,), "fma"),
    ("bf16", "NCHW", 2, 96, 96, 96, (500,), (7,), (1,), (3,), (1,), "tensor_op"),
    ("f16", "NCHW", 1, 32, 32, 1, (12, 12, 12), (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), "tensor_op"),
]


def problem_id(spec):
    return f"{spec[0]}-{spec[1]}-c{spec[3]}k{spec[4]}g{spec[5]}-{'x'.join(map(str, spec[6]))}-r{'x'.join(map(str, spec[7]))}-s{spec[8][0]}d{spec[10][0]}-{spec[11]}"


def torch_backend(p, x, w):
    """The backend torch's convolution of these tensors takes (``torch._C._select_conv_backend``): only a cuDNN one is a comparison of pins."""
    if p.layout == "NCHW":
        xt, wt, st, pad, dil = x, w, list(p.stride), list(p.padding), list(p.dilation)
        if p.spatial == 1:  # conv1d is posed as conv2d of [N, C, 1, L], as torch itself does
            xt, wt, st, pad, dil = x.unsqueeze(2), w.unsqueeze(2), [1, *st], [0, *pad], [1, *dil]
    elif p.spatial == 1:
        xt, wt = x.unsqueeze(1).permute(0, 3, 1, 2), w.unsqueeze(1).permute(0, 3, 1, 2)
        st, pad, dil = [1, p.stride[0]], [0, p.padding[0]], [1, p.dilation[0]]
    else:
        perm = (0, 3, 1, 2) if p.spatial == 2 else (0, 4, 1, 2, 3)
        xt, wt, st, pad, dil = x.permute(*perm), w.permute(*perm), list(p.stride), list(p.padding), list(p.dilation)
    return torch._C._select_conv_backend(xt, wt, None, st, pad, dil, False, [0] * len(st), p.groups, None)


@pytest.mark.parametrize("spec", PROBLEMS, ids=problem_id)
def test_the_instant_posture_pins_what_torch_runs(spec):
    p = _cudnn.Problem(*spec)
    x0, w0 = tensors(p, 1)
    backend = torch_backend(p, x0, w0)
    assert str(backend).endswith("Cudnn"), f"torch runs {p.what} with {backend}, not cuDNN: no comparison of pins"
    pinned = _cudnn.pin_conv(p, heuristic="instant", workspace_bytes=64 << 20)
    assert pinned.pin.api == "backend" and pinned.workspace_required <= 64 << 20
    misses = []
    for seed, kind in enumerate(("normal", "wide", "traps")):
        x, w = tensors(p, seed * 7 + 1)
        x, w = special(x, kind, seed), special(w, kind, seed + 1)
        y, theirs = ours(p, pinned.pin, x, w), torch_conv(p, x, w)
        torch.cuda.synchronize()
        if not same_bytes(y, theirs.contiguous()):
            misses.append((kind, differing(y, theirs.contiguous())))
    assert not misses, f"{p.what}, {pinned.pin.what}: not torch's bytes {misses}"


@pytest.mark.parametrize("spec", OTHER_PROBLEMS, ids=problem_id)
def test_a_pin_of_a_problem_torch_runs_without_cudnn_is_the_f64_result_within_one_accumulation(spec):
    p = _cudnn.Problem(*spec)
    x, w = tensors(p, 2)
    fma16 = p.math == "fma" and p.dtype != "f32"
    assert fma16 or not str(torch_backend(p, x, w)).endswith("Cudnn"), "torch runs it with cuDNN: it belongs with the comparisons of pins"
    for heuristic in ("instant", "v7"):
        pin = _cudnn.pin_conv(p, heuristic=heuristic, workspace_bytes=64 << 20).pin
        y, again = ours(p, pin, x, w), ours(p, pin, x, w)
        torch.cuda.synchronize()
        assert same_bytes(y, again), (heuristic, pin.what)
        fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[p.spatial]
        to_nchw = (0, p.spatial + 1, *range(1, p.spatial + 1)) if p.layout == "NHWC" else tuple(range(p.spatial + 2))
        x64, w64 = x.double().cpu().permute(*to_nchw), w.double().cpu().permute(*to_nchw)
        y = y.permute(*to_nchw)
        exact = fn(x64, w64, None, p.stride, p.padding, p.dilation, p.groups)
        absolute = fn(x64.abs(), w64.abs(), None, p.stride, p.padding, p.dilation, p.groups)
        terms = p.c // p.groups
        for r in p.filter:
            terms *= r
        gamma = (terms + 1) * 2.0**-24 / (1 - (terms + 1) * 2.0**-24)
        unit = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24}[p.dtype]
        if p.math == "tf32":
            gamma = terms * 2.0**-10
        excess = ((y.double().cpu() - exact).abs() - gamma * absolute - unit * exact.abs() * 1.0001 - 2.0**-24).max().item()
        assert excess <= 0, (heuristic, pin.what, excess)


SRC = os.path.dirname(os.path.dirname(os.path.abspath(tilefish.__file__)))  # the tilefish a subprocess imports: this one


def test_a_pin_repeats_its_bytes_in_the_process_and_in_another():
    p = _cudnn.Problem(*PROBLEMS[0])
    x, w = tensors(p, 3)
    for heuristic in ("instant", "v7"):
        pin = _cudnn.pin_conv(p, heuristic=heuristic, workspace_bytes=64 << 20).pin
        first, second = ours(p, pin, x, w), ours(p, pin, x, w)
        torch.cuda.synchronize()
        assert same_bytes(first, second), heuristic
        code = (
            "import sys, json, torch; sys.path[:0] = [%r, %r]; from tilefish import _cudnn as C; from test_gpu_conv import tensors, ours, PROBLEMS;"
            "p = C.Problem(*PROBLEMS[0]); x, w = tensors(p, 3); y = ours(p, C.pin_of(json.loads(sys.argv[1])), x, w); torch.cuda.synchronize();"
            "sys.stdout.write(y.cpu().view(torch.int16).numpy().tobytes().hex())"
        ) % (SRC, HERE)
        done = subprocess.run([sys.executable, "-c", code, json.dumps(pin.to_json())], capture_output=True, text=True, cwd=HERE)
        assert done.returncode == 0, done.stderr[-3000:]
        assert done.stdout == first.cpu().view(torch.int16).numpy().tobytes().hex(), heuristic


LEGACY_TORCH = [s for s in PROBLEMS if s[0] != "bf16"]  # torch's legacy path runs bf16 with CUDNN_DEFAULT_MATH, which no declaration says


def test_a_legacy_pin_of_implicit_precomp_gemm_is_what_torchs_legacy_path_runs():
    """torch's legacy convolution with ``deterministic = True`` and ``benchmark = False`` runs ``CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`` with the
    math type of its data (``onlyDefaultAlgorithm``, ``Conv_v7.cpp``): a legacy pin of it is torch's bytes. torch's legacy path is
    ``TORCH_CUDNN_V8_API_DISABLED=1``, read once per process, so torch runs in a process of its own and writes its results."""
    code = (
        "import sys, torch; sys.path[:0] = [%r, %r]; torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False;"
        "torch.backends.cudnn.allow_tf32 = False; from tilefish import _cudnn as C; from test_gpu_conv import tensors, torch_conv, LEGACY_TORCH;"
        "out = [];\n"
        "for spec in LEGACY_TORCH:\n"
        "    p = C.Problem(*spec); x, w = tensors(p, 5); y = torch_conv(p, x, w).contiguous(); torch.cuda.synchronize(); out.append(y.cpu().view(torch.uint8).numpy().tobytes().hex())\n"
        "sys.stdout.write('\\n'.join(out))"
    ) % (SRC, HERE)
    env = {**os.environ, "TORCH_CUDNN_V8_API_DISABLED": "1"}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE, env=env)
    assert done.returncode == 0, done.stderr[-3000:]
    lines = done.stdout.split("\n")
    misses = []
    for spec, theirs in zip(LEGACY_TORCH, lines):
        p = _cudnn.Problem(*spec)
        x, w = tensors(p, 5)
        y = ours(p, _cudnn.LegacyPin(1), x, w)
        torch.cuda.synchronize()
        if y.contiguous().cpu().view(torch.uint8).numpy().tobytes().hex() != theirs:
            misses.append(p.what)
    assert len(lines) == len(LEGACY_TORCH) and not misses, misses


def test_the_v7_posture_takes_the_first_deterministic_answer_that_fits():
    for spec in PROBLEMS[:6]:
        p = _cudnn.Problem(*spec)
        answers = _cudnn.legacy_list(p)
        first = next(a for a in answers if a["status"] == 0 and a["determinism"] == 1 and a["workspace"] is not None and a["workspace"] <= 64 << 20)
        assert _cudnn.pin_conv(p, heuristic="v7", workspace_bytes=64 << 20).pin == _cudnn.LegacyPin(first["algo"]), p.what


# -- the library op ---------------------------------------------------------------------------------------------------------------------------


def test_jit_and_replay_agree_for_both_apis_and_the_reference_is_close(tmp_path):
    manifest = runtime.export(list(ck.CONVS.values()), tmp_path)
    entries = {e["stem"]: e for e in manifest.library_entries}
    assert entries["conv_k7.T646"]["pin"]["api"] == "backend" and entries["conv_k7.T646"]["choice"] == "instant"
    assert entries["conv_k7_dil.T64"]["pin"]["api"] == "legacy" and entries["conv_k7_dil.T64"]["choice"] == "v7"
    assert entries["conv_in.T646"]["output"] == [652] and entries["conv_down.T646"]["output"] == [323]
    for e in entries.values():
        assert e["cudnn_version"] == _cudnn.version() and e["sm_count"] == _cudnn.sm_count(0)
    for op in ck.CONVS.values():
        bucketed = any(d.free_dims() for p in op.signature.params for d in p.type.shape)
        for values in ({"T": 64}, {"T": 646}) if bucketed else ({},):
            x, w = ck.operands(op, values, seed=len(op.name))
            jit = tilefish.run(op, x, w)
            replay = tilefish.run(op, x, w, mode="replay", manifest=tmp_path, pin=manifest.digest)
            ref = tilefish.run(op, x, w, mode="reference")
            assert same_bytes(jit, replay), (op.name, values)
            scale = ref.double().abs().max().item()
            assert (jit.double() - ref.double()).abs().max().item() <= 2**-6 * scale + 1e-3, (op.name, values)


def test_a_plan_of_two_convolutions_runs_the_same_in_jit_replay_a_capture_and_alone(tmp_path):
    plan = runtime.export(ck.conv_chain, tmp_path)
    workspace = next(b for b in plan.doc["buffers"] if b["role"] == "workspace")
    assert workspace["bytes"] >= 64 << 20
    for t in (64, 646):
        x, w1 = ck.operands(ck.conv_k7, {"T": t}, seed=t)
        _, w2 = ck.operands(ck.conv_k7_dil, {"T": t}, seed=t + 9)
        jit = tilefish.run(ck.conv_chain, x, w1, w2)
        replay = tilefish.run(ck.conv_chain, x, w1, w2, mode="replay", plan=tmp_path, pin=plan.digest)
        alone = tilefish.run(ck.conv_k7_dil, tilefish.run(ck.conv_k7, x, w1), w2)
        assert same_bytes(jit, replay) and same_bytes(jit, alone), t
        out = torch.empty_like(jit)
        runtime.prepare(ck.conv_chain, mode="replay", plan=tmp_path, pin=plan.digest)
        graph = tilefish.capture(ck.conv_chain, x, w1, w2, mode="replay", plan=tmp_path, pin=plan.digest, outputs=out)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert same_bytes(out, replay), t


def test_an_imported_pin_is_the_heuristics_and_its_table_reaches_the_digest(tmp_path):
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major * 10 + minor}"
    rows = {}
    for t in (64, 646):
        p = _conv.problem_at("conv_k7", ck.conv_k7.signature, ck.conv_k7.conv, {ck.T: t})
        rows[_conv.key_of(p)] = _cudnn.pin_conv(p, heuristic="instant", workspace_bytes=64 << 20).pin.to_json()
    table = CudnnTable(arch, props.multi_processor_count, _cudnn.version(), rows)

    @tilefish.library("conv_k7", impl="cudnn.conv_fwd", layout="NHWC", padding=3, algos=table)
    def imported(x: tilefish.Tensor[1, ck.T, 64, tilefish.f16], w: tilefish.Tensor[64, 7, 64, tilefish.f16]) -> tilefish.Tensor[1, ck.T, 64, tilefish.f16]: ...

    m_imported = runtime.export([imported], tmp_path / "imported")
    m_plain = runtime.export([ck.conv_k7], tmp_path / "plain")
    assert _conv.declaration_digest(imported) != _conv.declaration_digest(ck.conv_k7)
    for t in (64, 646):
        stem = f"conv_k7.T{t}"
        assert m_imported.library_entry(stem, arch)["pin"] == m_plain.library_entry(stem, arch)["pin"]
        assert m_imported.library_entry(stem, arch)["choice"] == "imported"
        x, w = ck.operands(ck.conv_k7, {"T": t}, seed=t)
        assert same_bytes(
            tilefish.run(imported, x, w, mode="replay", manifest=tmp_path / "imported", pin=m_imported.digest),
            tilefish.run(ck.conv_k7, x, w, mode="replay", manifest=tmp_path / "plain", pin=m_plain.digest),
        )


def _patched(tmp_path, change, name="patched"):
    """The manifest of ``conv_k7`` exported to ``name`` with its entries changed by ``change``; ``(None, its digest)``."""
    from tilefish._manifest import manifest_digest

    runtime.export([ck.conv_k7], tmp_path / name)
    path = tmp_path / name / "manifest.json"
    doc = json.loads(path.read_text())
    for e in doc["library"]:
        change(e)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return None, manifest_digest(doc)


def test_a_replay_refuses_another_cudnn_another_device_and_the_tf32_override(tmp_path, monkeypatch):
    x, w = ck.operands(ck.conv_k7, {"T": 64})
    run = lambda d, digest: tilefish.run(ck.conv_k7, x, w, mode="replay", manifest=tmp_path / d, pin=digest)  # noqa: E731
    _, digest = _patched(tmp_path, lambda e: e.update(cudnn_version=e["cudnn_version"] + 1), "version")
    with pytest.raises(TilefishManifestError, match=r"with cuDNN \d+, but this process has cuDNN \d+: a pin names other kernels"):
        run("version", digest)
    _, digest = _patched(tmp_path, lambda e: e.update(sm_count=e["sm_count"] + 1), "sms")
    with pytest.raises(TilefishManifestError, match="on a device with .* SMs, but this device has"):
        run("sms", digest)
    _, digest = _patched(tmp_path, lambda e: e.update(pin={"api": "legacy", "algo": 1}), "api")
    with pytest.raises(TilefishManifestError, match="the choice 'instant' pins a backend pin, got a legacy one"):
        run("api", digest)
    _, digest = _patched(tmp_path, lambda e: None, "ok")
    assert run("ok", digest).shape == (1, 64, 64)
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    with pytest.raises(TilefishManifestError, match="NVIDIA_TF32_OVERRIDE is set \\(to '0'\\)"):
        run("ok", digest)
    with pytest.raises(TilefishRuntimeError, match="NVIDIA_TF32_OVERRIDE is set"):
        _cudnn.pin_conv(_cudnn.Problem(*PROBLEMS[0]), heuristic="instant", workspace_bytes=1 << 20)
    with pytest.raises(TilefishBuildError, match="NVIDIA_TF32_OVERRIDE is set"):
        runtime.export([ck.conv_k7], tmp_path / "tf32")


def test_a_pin_the_math_or_the_engine_does_not_allow_is_refused():
    p = _cudnn.Problem(*PROBLEMS[0])
    engines = _cudnn.engine_list(p)
    tensor_core = next((e for e in engines if "TENSOR_CORE" in e["notes"] and e["finalizes"] and "NONDETERMINISTIC" not in e["notes"]), None)
    if tensor_core is not None:  # an engine of tensor cores is not one of the math 'fma'
        fma = _cudnn.Problem(*PROBLEMS[0][:-1], "fma")
        pin = _cudnn.EnginePin(tensor_core["engine"], tuple(tuple(kv) for kv in tensor_core["knobs"]))
        with pytest.raises(TilefishRuntimeError, match="uses tensor cores .* and the math is 'fma'"):
            _cudnn.check_pin(fma, pin, workspace_bytes=64 << 20)
    good = _cudnn.pin_conv(p, heuristic="instant", workspace_bytes=64 << 20)
    infos = next(e for e in engines if e["engine"] == good.pin.engine)
    with pytest.raises(TilefishRuntimeError, match="knob type 42 is not a knob of the engine"):
        _cudnn.check_pin(p, _cudnn.EnginePin(good.pin.engine, ((42, 0),)), workspace_bytes=64 << 20)
    with pytest.raises(TilefishRuntimeError, match="not an engine of the problem"):
        _cudnn.check_pin(p, _cudnn.EnginePin(10_000), workspace_bytes=64 << 20)
    if good.workspace_required:
        with pytest.raises(TilefishRuntimeError, match="bytes of workspace, more than the"):
            _cudnn.check_pin(p, good.pin, workspace_bytes=good.workspace_required - 1)
    assert infos["allowed"]
    nondeterministic = [a for a in _cudnn.legacy_list(p) if a["status"] == 0 and a["determinism"] != 1]
    for a in nondeterministic[:1]:
        with pytest.raises(TilefishRuntimeError, match="says it is not deterministic"):
            _cudnn.check_pin(p, _cudnn.LegacyPin(a["algo"]), workspace_bytes=1 << 30)


def test_an_sm_count_target_is_set_on_the_engine_and_recorded(tmp_path):
    """A target of the device's count is as none; a smaller one is set on the heuristic and on the engine, and where cuDNN takes none for the problem
    (measured with cuDNN 9.10.2 on a B300 MIG slice, sm_103: no engine of these convolutions finalizes with a target below the device's 30 SMs) the
    pin is refused with what was asked, never run without it."""
    props = torch.cuda.get_device_properties(0)
    target = max(1, props.multi_processor_count // 2)
    p = _cudnn.Problem(*PROBLEMS[0])
    plain = _cudnn.pin_conv(p, heuristic="instant", workspace_bytes=64 << 20)
    whole = _cudnn.pin_conv(p, heuristic="instant", workspace_bytes=64 << 20, sm_count_target=props.multi_processor_count)
    assert whole.pin == plain.pin
    try:
        aimed = _cudnn.pin_conv(p, heuristic="instant", workspace_bytes=64 << 20, sm_count_target=target)
    except TilefishRuntimeError as e:
        assert "has no engine configuration" in str(e), e
        with pytest.raises(TilefishRuntimeError, match=f"engine {plain.pin.engine} is not an engine of the problem"):
            _cudnn.check_pin(p, plain.pin, workspace_bytes=64 << 20, sm_count_target=target)
        print(f"\nconv sm_count_target={target}: cuDNN {_cudnn.version()} takes no target for {p.what}")
        return
    x, w = tensors(p, 11)
    same = same_bytes(ours(p, plain.pin, x, w), ours(p, aimed.pin, x, w, target=target))
    print(f"\nconv sm_count_target={target}: plain {plain.pin.what}, aimed {aimed.pin.what}, same bytes: {same}")

    @tilefish.library("aimed", impl="cudnn.conv_fwd", layout="NHWC", padding=3, sm_count_target=target)
    def aimed_op(x: tilefish.Tensor[1, 646, 64, tilefish.f16], w: tilefish.Tensor[64, 7, 64, tilefish.f16]) -> tilefish.Tensor[1, 646, 64, tilefish.f16]: ...

    m = runtime.export([aimed_op], tmp_path)
    assert m.library_entries[0]["sm_count_target"] == target and m.library_entries[0]["pin"] == aimed.pin.to_json()
    xs, ws = tensors(p, 12)
    assert same_bytes(tilefish.run(aimed_op, xs, ws), tilefish.run(aimed_op, xs, ws, mode="replay", manifest=tmp_path, pin=m.digest))

    @tilefish.library("aimed_legacy", impl="cudnn.conv_fwd", layout="NHWC", padding=3, api="legacy", sm_count_target=target)
    def aimed_legacy(x: tilefish.Tensor[1, 646, 64, tilefish.f16], w: tilefish.Tensor[64, 7, 64, tilefish.f16]) -> tilefish.Tensor[1, 646, 64, tilefish.f16]: ...

    with pytest.raises(TilefishBuildError, match="the legacy API takes no SM count target"):
        runtime.export([aimed_legacy], tmp_path / "legacy")


# -- the canary, and the streams the calls are on -----------------------------------------------------------------------------------------------


def _arch():
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major * 10 + minor}"


def test_export_records_a_canary_a_replay_runs_it_and_refuses_other_bytes(tmp_path, monkeypatch):
    manifest = runtime.export([ck.conv_k7, ck.conv_k7_dil], tmp_path / "m")
    assert len(manifest.library_entries) == 4
    for op in (ck.conv_k7, ck.conv_k7_dil):
        for spec in _conv.conv_specs(op):
            e = manifest.library_entry(spec.stem, _arch())
            assert e["canary"]["seed"] == _conv.CANARY_SEED, spec.stem
            # the pin's bytes from the operands of the seed: computed again (another stream, another allocation), the same
            assert _conv.canary_digest(spec, _cudnn.pin_of(e["pin"]), e["canary"]["seed"], 0) == e["canary"]["y_digest"], spec.stem
    assert len({e["canary"]["y_digest"] for e in manifest.library_entries}) == 4
    x, w = ck.operands(ck.conv_k7, {"T": 64})
    jit = tilefish.run(ck.conv_k7, x, w)
    monkeypatch.setattr(_conv, "_canaries_passed", set())
    # a canary another build of cuDNN (or another device) computed: refused before the pin runs
    _, digest = _patched(tmp_path, lambda e: e["canary"].update(y_digest="0" * 32), "canary")
    with pytest.raises(TilefishManifestError, match="the canary of the cuDNN convolution 'conv_k7.T64' .* gives y with digest [0-9a-f]{32} here, and the manifest recorded 0{32}"):
        tilefish.run(ck.conv_k7, x, w, mode="replay", manifest=tmp_path / "canary", pin=digest)
    # the field is optional: an entry exported without one runs as before
    _, digest = _patched(tmp_path, lambda e: e.pop("canary"), "none")
    assert same_bytes(tilefish.run(ck.conv_k7, x, w, mode="replay", manifest=tmp_path / "none", pin=digest), jit)
    _, digest = _patched(tmp_path, lambda e: None, "ok")
    assert same_bytes(tilefish.run(ck.conv_k7, x, w, mode="replay", manifest=tmp_path / "ok", pin=digest), jit)


def _sleep_cycles(seconds):
    """The cycles of ``torch.cuda._sleep`` that take about ``seconds`` on this device."""
    s = torch.cuda.Stream()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(s):
        start.record()
        torch.cuda._sleep(20_000_000)
        end.record()
    end.synchronize()
    return int(20_000_000 * seconds * 1000 / start.elapsed_time(end))


def test_the_canary_runs_on_a_stream_of_its_own_and_never_on_the_null_stream(tmp_path, monkeypatch):
    """With the NULL stream (the legacy default stream) busy for two seconds, the canaries of a manifest run and pass, and the NULL stream is still busy when
    they return: no copy, memset, allocation or wait of a canary was on it (an operand copied there would be copied after the convolution read it, and a
    copy back there would wait for the sleep). Inside a green context the NULL stream does not order against another stream, so a canary on it would
    race."""
    manifest = runtime.export([ck.conv_k7, ck.conv_k7_dil, ck.conv_in], tmp_path)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    built = [
        _conv.built_from_manifest(manifest, spec, _arch(), sms, 0) for op in (ck.conv_k7, ck.conv_k7_dil, ck.conv_in) for spec in _conv.conv_specs(op)
    ]  # the pins prepared, their canaries passed once
    cycles = _sleep_cycles(2.0)
    monkeypatch.setattr(_conv, "_canaries_passed", set())
    torch.cuda.synchronize()
    null = torch.cuda.default_stream()
    assert null.cuda_stream == 0, "torch's default stream is the legacy NULL stream"
    with torch.cuda.stream(null):
        torch.cuda._sleep(cycles)
    began = time.monotonic()
    for b in built:
        _conv.check_canary(b, 0)
    busy = not null.query()
    took = time.monotonic() - began
    torch.cuda.synchronize()
    assert len(_conv._canaries_passed) == len(built) == 6
    assert busy and took < 1.5, f"the canaries took {took:.2f} s and the NULL stream is {'busy' if busy else 'idle'}: one of them waited for it"


CAPTURED = [PROBLEMS[0], PROBLEMS[6], PROBLEMS[12]]


@pytest.mark.parametrize("spec", CAPTURED, ids=lambda s: f"{s[0]}-{s[1]}-c{s[3]}-k{s[4]}")
def test_a_global_capture_on_a_non_blocking_stream_holds_every_call_of_a_pin(spec):
    """Captured with ``capture_error_mode="global"`` on a non-blocking stream, a call of either API with the workspace given (a plan's arena) or allocated
    by the call writes nothing until the graph is replayed, and the replay is the eager bytes: nothing of the call is on the NULL stream or another stream
    (it would run during the capture, or be refused by it), the handle's stream being set before the call."""
    p = _cudnn.Problem(*spec)
    x, w = tensors(p, 5)
    limit = 64 << 20
    for heuristic in ("instant", "v7"):
        pin = _cudnn.pin_conv(p, heuristic=heuristic, workspace_bytes=limit).pin
        eager = ours(p, pin, x, w, limit)
        for given in (True, False):
            y = torch.zeros(p.tilefish_shape("y"), dtype=DT[p.dtype], device="cuda")
            workspace = torch.empty(limit, dtype=torch.uint8, device="cuda") if given else None
            s = torch.cuda.Stream()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s, capture_error_mode="global"):
                _cudnn.convolution(
                    x.data_ptr(), w.data_ptr(), y.data_ptr(), p, pin, workspace_bytes=limit, stream=s.cuda_stream,
                    workspace_ptr=workspace.data_ptr() if given else None,
                )  # fmt: skip
            torch.cuda.synchronize()
            assert not y.view(torch.uint8).any(), (heuristic, given, "the capture ran something")
            g.replay()
            torch.cuda.synchronize()
            assert same_bytes(y, eager), (heuristic, given)


def test_a_global_capture_holds_a_replay_of_the_op_and_of_a_plan(tmp_path):
    """``tilefish.run`` in replay, the op and a plan of two convolutions, captured with ``torch.cuda.graph`` in its global mode after a first run (where the
    canaries run): the replay of the graph is the eager bytes."""
    manifest = runtime.export([ck.conv_k7, ck.conv_k7_dil], tmp_path / "ops")
    plan = runtime.export(ck.conv_chain, tmp_path / "plan")
    x, w1 = ck.operands(ck.conv_k7, {"T": 646}, seed=3)
    _, w2 = ck.operands(ck.conv_k7_dil, {"T": 646}, seed=4)
    runs = {
        "op": lambda: tilefish.run(ck.conv_k7_dil, x, w2, mode="replay", manifest=tmp_path / "ops", pin=manifest.digest),
        "plan": lambda: tilefish.run(ck.conv_chain, x, w1, w2, mode="replay", plan=tmp_path / "plan", pin=plan.digest),
    }
    for name, run in runs.items():
        eager = run()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode="global"):
            out = run()
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert same_bytes(out, eager), name


def test_the_canary_kernels_are_the_host_reference():
    """The fill kernel writes the host reference's bytes (each data type, sizes around a block and past one grid), and the digest kernel adds the
    host's digest (odd and even sizes)."""
    import ctypes
    import random

    from tilefish import _canary

    s = torch.cuda.Stream()
    for dtype in ("f16", "bf16", "f32"):
        for n in (1, 255, 256, 257, 70001):
            x = torch.empty(n * _canary.ELEMENT_BYTES[dtype], dtype=torch.uint8, device="cuda")
            _canary.fill(x.data_ptr(), n, dtype, 7, 1, s.cuda_stream, 0)
            s.synchronize()
            on_host = x.cpu()
            assert ctypes.string_at(on_host.data_ptr(), on_host.numel()) == _canary.canary_bytes(dtype, 7, 1, n), (dtype, n)
    rng = random.Random(3)
    for nbytes in (2, 4, 6, 1022, 4096 * 4 + 2, 300001 * 2):
        data = bytes(rng.getrandbits(8) for _ in range(nbytes))
        padded = data + b"\x00" * (-nbytes % 4)
        y = torch.frombuffer(bytearray(padded), dtype=torch.uint8).cuda()
        sums = torch.zeros(2, dtype=torch.int64, device="cuda")
        torch.cuda.synchronize()
        _canary.digest(y.data_ptr(), len(padded) // 4, sums.data_ptr(), s.cuda_stream, 0)
        s.synchronize()
        assert _canary.hex_of(*(int(v) for v in sums.cpu())) == _canary.digest_of(data), nbytes


def test_a_knob_the_heuristic_leaves_unset_is_left_out_of_the_pin():
    """cuDNN shows a knob it leaves at the engine's default as -1 among a configuration's choices (measured with cuDNN 9.13.0: engine 23's SPLIT_K_BUF for
    a stride-8 k16 convolution, 512 to 1024 channels, 646 long). The pin leaves it out: its plan is the heuristic's (the JSON check) and its bytes are
    torch's. No knob of these engines takes -1 as a value (every KNOB_INFO minimum is 0 or more)."""
    p = _cudnn.Problem("f16", "NHWC", 1, 512, 1024, 1, (646,), (16,), (8,), (4,), (1,), "tensor_op")
    lib = _cudnn._lib()
    h = _cudnn._handle(lib, 0)
    with h.lock:
        graph = _cudnn._Graph(lib, h.handle, p)
        try:
            configs = [cfg for mode in _cudnn.HEUR_MODES["instant"] for cfg in graph.heuristic(mode, None)]
            raw = [graph.config_knobs(cfg) for cfg in configs]
            minimums = [lo for cfg in configs for lo, _, _ in graph.engine_facts(graph.config_engine(cfg))["knobs"].values()]
        finally:
            graph.close()
    assert min(minimums, default=0) >= 0
    pinned = _cudnn.pin_conv(p, heuristic="instant", workspace_bytes=1 << 30)
    assert all(v != -1 for _, v in pinned.pin.knobs)
    unset = [t for t, v in raw[0] if v == -1] if raw else []
    if unset:
        assert set(dict(raw[0])) - set(dict(pinned.pin.knobs)) == set(unset)
    print(f"\nthe heuristic's first configuration leaves knobs {unset} unset (-1); the pin is {pinned.pin.what}")
    x, w = tensors(p, 4)
    assert same_bytes(ours(p, pinned.pin, x, w, 1 << 30), torch_conv(p, x, w))
