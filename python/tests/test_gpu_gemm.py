"""A cuBLASLt GEMM as a library operation with a pinned algorithm, on a real GPU.

``@tilefish.library("gemm", acc=f32, impl="cublaslt.gemm")`` over ``(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16]) -> Tensor[M, N, bf16]``, which
computes ``a @ b.T`` with an f32 accumulator and one rounding to bf16 (``impl="cublaslt.gemm_bias"`` adds ``bias: Tensor[N, bf16]`` through
``CUBLASLT_EPILOGUE_BIAS``: added in f32 before the rounding). For each ``(M, N, K, bias)`` a binary is the 64-byte algorithm that
cuBLASLt's heuristic ranks first under a fixed workspace limit (8 519 680 bytes, torch's default), recorded in the ``library`` section of a
manifest with the cuBLASLt version. ``jit`` pins in memory the same way, so ``jit`` and ``replay`` run the same algorithm.

Checked here: ``jit`` and ``replay`` agree byte for byte; the pinned replay is byte-identical across two processes (and the pin itself is); the
result is the f64 reference within what one f32 accumulation and one rounding can do; a cuBLASLt of another version is refused; the
preconditions (contiguous, aligned, an exact bucket) are. How often the result is torch's byte for byte is *reported* (informational: it is not
asserted).
"""

import hashlib
import json
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Manifest, Tensor, TilefishManifestError, TilefishPreconditionError, bf16, f32, runtime  # noqa: E402
from tilefish import _cublaslt  # noqa: E402

BF16 = torch.bfloat16
BUCKETS = [32, 64, 320]
SHAPES = [(1024, 1024), (3072, 1024), (1024, 3072), (4096, 3072), (3072, 4096)]  # (N, K)


def make_gemm(name, n, k, bias):
    M = Dim("M", buckets=BUCKETS)
    if bias:

        @tilefish.library(name, acc=f32, impl="cublaslt.gemm_bias")
        def gemm_bias(a: Tensor[M, k, bf16], b: Tensor[n, k, bf16], bias: Tensor[n, bf16]) -> Tensor[M, n, bf16]: ...

        return gemm_bias

    @tilefish.library(name, acc=f32, impl="cublaslt.gemm")
    def gemm(a: Tensor[M, k, bf16], b: Tensor[n, k, bf16]) -> Tensor[M, n, bf16]: ...

    return gemm


def operands(m, n, k, bias, seed=0):
    a = randn((m, k), BF16, seed=seed)
    b = randn((n, k), BF16, seed=seed + 1, scale=1 / k**0.5)
    return (a, b, randn((n,), BF16, seed=seed + 2)) if bias else (a, b)


def torch_result(args):
    a, b, *rest = args
    return torch.addmm(rest[0], a, b.t()) if rest else torch.mm(a, b.t())


@pytest.fixture(scope="module")
def ops():
    return {(n, k, bias): make_gemm(f"gemm_{n}_{k}{'_bias' if bias else ''}", n, k, bias) for n, k in SHAPES for bias in (False, True)}


@pytest.fixture(scope="module")
def exported(ops, tmp_path_factory):
    directory = tmp_path_factory.mktemp("gemm-manifest")
    return directory, runtime.export(list(ops.values()), directory)


def test_the_manifest_has_one_library_entry_per_bucket_and_no_kernel(ops, exported):
    directory, manifest = exported
    doc = json.loads((directory / "manifest.json").read_text())
    assert doc["schema"] == 3 and doc["kernels"] == [] and len(doc["library"]) == len(ops) * len(BUCKETS)
    assert [p.name for p in directory.iterdir()] == ["manifest.json"]  # a library entry has no file
    version = _cublaslt.version()
    for e in doc["library"]:
        n, k, bias = e["n"], e["k"], e["impl"].endswith("bias")
        assert e["m"] in BUCKETS and e["buckets"] == {"M": e["m"]} and e["stem"] == f"{e['op']}.M{e['m']}"
        assert e["arch"] == f"sm_{torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]}"
        assert (e["transa"], e["transb"], e["compute_type"], e["scale_type"]) == ("T", "N", "CUBLAS_COMPUTE_32F", "CUDA_R_32F")
        assert (e["lda"], e["ldb"], e["ldc"], e["ldd"]) == (k, k, n, n) and e["workspace_bytes"] == 8_519_680
        assert e["epilogue"] == ("bias" if bias else "none") and e["cublaslt_version"] == version
        assert len(e["algo"]) == 128 and set(e["algo"]) <= set("0123456789abcdef")
        # what the library binding says it calls cuBLASLt with
        d = _cublaslt.describe(e["m"], n, k, bias)
        assert (d["transa"], d["transb"], d["lda"], d["ldb"], d["ldc"], d["ldd"], d["epilogue"]) == (
            e["transa"], e["transb"], e["lda"], e["ldb"], e["ldc"], e["ldd"], e["epilogue"]
        )
    assert Manifest.load(directory, pin=manifest.digest).digest == manifest.digest


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("n, k", SHAPES)
def test_jit_and_replay_agree_byte_for_byte_and_equal_the_f64_reference(ops, exported, n, k, bias):
    directory, manifest = exported
    op = ops[(n, k, bias)]
    for m in BUCKETS:
        args = operands(m, n, k, bias, seed=m)
        jit = tilefish.run(op, *args)
        replay = tilefish.run(op, *args, mode="replay", manifest=directory, pin=manifest.digest)
        assert jit.shape == (m, n) and jit.dtype == BF16
        assert same_bytes(jit, replay), (m, n, k, bias)
        a, b, *rest = args
        exact = a.double() @ b.double().t() + (rest[0].double() if rest else 0)
        assert (jit.double() - exact).abs().max() <= 2**-8 * exact.abs().max() * 1.01, (m, n, k, bias)  # one rounding, within half a bf16 step


def test_the_pin_is_deterministic_in_one_process(ops):
    op = ops[(3072, 1024, False)]
    from tilefish import _library

    first = [(s.stem, _library.build_library(s, arch=_arch()).algo) for s in _library.library_specs(op)]
    second = [(s.stem, _library.build_library(s, arch=_arch()).algo) for s in _library.library_specs(op)]
    assert first == second and len(first) == len(BUCKETS)


def _arch():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


SCRIPT = textwrap.dedent(
    """
    import hashlib, json, sys
    import torch
    import tilefish
    from tilefish import Dim, Tensor, bf16, f32, runtime
    from gpu_util import randn

    directory, pin, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n, k = 3072, 1024
    M = Dim("M", buckets=[32, 64, 320])

    @tilefish.library("gemm_3072_1024_bias", acc=f32, impl="cublaslt.gemm_bias")
    def gemm_bias(a: Tensor[M, k, bf16], b: Tensor[n, k, bf16], bias: Tensor[n, bf16]) -> Tensor[M, n, bf16]: ...

    digests = {}
    for m in (32, 64, 320):
        a = randn((m, k), torch.bfloat16, seed=m)
        b = randn((n, k), torch.bfloat16, seed=m + 1, scale=1 / k**0.5)
        bias = randn((n,), torch.bfloat16, seed=m + 2)
        y = tilefish.run(gemm_bias, a, b, bias, mode="replay", manifest=directory, pin=pin)
        digests[m] = hashlib.sha256(y.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    # and the pin itself, made again in this process
    other = out + "/again"
    runtime.export(gemm_bias, other)
    doc = json.loads(open(other + "/manifest.json").read())
    json.dump({"digests": digests, "algos": {e["stem"]: e["algo"] for e in doc["library"]}}, open(out + "/result.json", "w"))
    """
)


def test_a_pinned_replay_and_the_pin_are_byte_identical_across_two_processes(ops, exported, tmp_path):
    import os

    directory, manifest = exported
    op = ops[(3072, 1024, True)]
    here = {}
    for m in BUCKETS:
        args = operands(m, 3072, 1024, True, seed=m)
        # the script's operands: seeds m, m + 1, m + 2 (the same as operands(...) makes)
        y = tilefish.run(op, *args, mode="replay", manifest=directory, pin=manifest.digest)
        here[str(m)] = hashlib.sha256(y.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    script = tmp_path / "second.py"
    script.write_text(SCRIPT)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "TILEFISH_CACHE_DIR": str(tmp_path / "cache")}
    out = tmp_path / "out"
    out.mkdir()
    done = subprocess.run([sys.executable, str(script), str(directory), manifest.digest, str(out)], capture_output=True, text=True, env=env, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    result = json.loads((out / "result.json").read_text())
    assert {str(k): v for k, v in result["digests"].items()} == here
    ours = {e["stem"]: e["algo"] for e in json.loads((directory / "manifest.json").read_text())["library"] if e["op"] == op.name}
    assert result["algos"] == ours  # the other process pinned the same algorithms


def test_a_cublaslt_of_another_version_is_refused(ops, exported, monkeypatch, tmp_path):
    directory, manifest = exported
    op = ops[(1024, 1024, False)]
    args = operands(32, 1024, 1024, False)
    version = _cublaslt.version()
    assert same_bytes(tilefish.run(op, *args, mode="replay", manifest=directory, pin=manifest.digest), tilefish.run(op, *args))
    monkeypatch.setattr(_cublaslt, "version", lambda: version + 100)  # this process has another cuBLASLt
    runtime._pins.clear()
    with pytest.raises(TilefishManifestError, match=rf"pinned the cuBLASLt algorithm of 'gemm_1024_1024.M32' \(sm_\d+\) with cuBLASLt {version}, but this process has cuBLASLt {version + 100}"):
        tilefish.run(op, *args, mode="replay", manifest=directory, pin=manifest.digest)
    monkeypatch.undo()
    # the manifest says another version (re-pinned: the digest is what vouches for the file): the same refusal
    doc = json.loads((directory / "manifest.json").read_text())
    for e in doc["library"]:
        e["cublaslt_version"] = version + 1
    other = tmp_path / "other"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="this process has cuBLASLt"):
        tilefish.run(op, *args, mode="replay", manifest=other, trust_directory=True)


def test_an_entry_that_is_not_what_the_declaration_asks_for_is_refused(ops, exported, tmp_path):
    directory, _ = exported
    op = ops[(1024, 1024, False)]
    args = operands(32, 1024, 1024, False)
    doc = json.loads((directory / "manifest.json").read_text())
    for e in doc["library"]:
        if e["stem"] == "gemm_1024_1024.M32":
            e["workspace_bytes"] = 1 << 20
    (tmp_path / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="first difference at workspace_bytes"):
        tilefish.run(op, *args, mode="replay", manifest=tmp_path, trust_directory=True)


def test_the_preconditions_of_a_gemm_are_checked(ops):
    op = ops[(1024, 1024, False)]
    a, b = operands(32, 1024, 1024, False)
    with pytest.raises(TilefishPreconditionError, match="not contiguous"):
        tilefish.run(op, a, b.t().contiguous().t())
    with pytest.raises(TilefishPreconditionError, match="is not one of its buckets"):
        tilefish.run(op, randn((48, 1024), BF16, seed=1), b)
    with pytest.raises(TilefishPreconditionError, match="dtype"):
        tilefish.run(op, a.float(), b)
    flat = randn((32 * 1024 + 8,), BF16, seed=2)
    with pytest.raises(TilefishPreconditionError, match="not aligned to 16 bytes"):
        tilefish.run(op, flat[1 : 1 + 32 * 1024].view(32, 1024), b)


def test_the_reference_mode_is_torch_in_f32(ops):
    op = ops[(1024, 1024, True)]
    args = operands(64, 1024, 1024, True, seed=3)
    got = tilefish.run(op, *args, mode="reference")
    a, b, bias = args
    assert same_bytes(got, (a.float() @ b.float().t() + bias.float()).to(BF16))


def test_how_often_the_pinned_result_is_the_result_of_torch_byte_for_byte(ops, exported, capsys):
    """Informational: nothing here fails on the rate. (The algorithm is cuBLASLt's top heuristic result under the same workspace limit as torch's.)"""
    rows = {"gemm": [0, 0, 0], "gemm_bias": [0, 0, 0]}
    for (n, k, bias), op in ops.items():
        for m in BUCKETS:
            args = operands(m, n, k, bias, seed=1000 + m)
            ours, ref = tilefish.run(op, *args), torch_result(args)
            row = rows["gemm_bias" if bias else "gemm"]
            row[0] += int(same_bytes(ours, ref))
            row[1] += 1
            if not same_bytes(ours, ref):
                a = ours.view(torch.int16).to(torch.int64)
                b = ref.view(torch.int16).to(torch.int64)
                row[2] = max(row[2], int((a - b).abs().max()))
    with capsys.disabled():
        for name, (same, total, ulp) in rows.items():
            print(f"\n[gemm torch match] {name}: {same} of {total} shapes byte-identical to torch" + (f" (largest bf16 distance {ulp} steps)" if same != total else ""))
    assert sum(r[1] for r in rows.values()) == len(ops) * len(BUCKETS)
