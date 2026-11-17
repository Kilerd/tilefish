"""Locked exports (``docs/guide/locked-export.md``) with no GPU: a lock made from an export directory (``tilefish.runtime.write_lock``), and an export
from it (``export(..., lock=...)``) that asks no heuristic, runs nothing on a device and writes the same bytes; and every refusal of either. The builds are
the fake compiler's (``fake_compile``) and the pins of the "GPU" export are made here; the real thing, a lock made on a GPU and the locked export on a
machine without one, is ``test_gpu_locked_export.py`` and ``rust_gate.py prepare --locked``.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import hashlib
import json
import pathlib
import shutil
import sys
import tempfile

import fake_compile
import pytest
from binary_fakes import fake_binary_build, make_ptx_op, write_ptx

import tilefish
from tilefish import Dim, Tensor, TilefishLockError, bf16, f16, f32, runtime
from tilefish import _binary, _conv, _cublaslt, _cudnn, _fusion, _library, _lock, _planrun, _trace

ARCH = 103
SMS = 30  # a MIG slice of a B300
CUBLASLT = 120804
CUDNN = 91002
PTXAS = "Cuda compilation tools, release 12.9, V12.9.86"

M = Dim("M", buckets=[32, 64])
T = Dim("T", buckets=[64, 128])


@tilefish.kernel
def lt_norm(x: Tensor[M, 256, bf16], w: Tensor[256, bf16]) -> Tensor[M, 256, bf16]:
    xf = x.to(f32)
    return (xf * tilefish.rsqrt(tilefish.mean(xf * xf, axis=-1, keepdims=True) + 1e-6) * w.to(f32)).to(bf16)


@tilefish.library("lt_up", acc=f32, impl="cublaslt.gemm_bias")
def lt_up(a: Tensor[M, 256, bf16], b: Tensor[512, 256, bf16], bias: Tensor[512, bf16]) -> Tensor[M, 512, bf16]: ...


@tilefish.library("lt_down", acc=f32, impl="cublaslt.matmul", epilogue="relu_bias", beta=1)
def lt_down(a: Tensor[M, 512, bf16], b: Tensor[256, 512, bf16], bias: Tensor[256, bf16], c: Tensor[M, 256, bf16]) -> Tensor[M, 256, bf16]: ...


@tilefish.library("lt_conv1d", impl="cudnn.conv_fwd", layout="NHWC", padding=1)
def lt_conv1d(x: Tensor[1, T, 128, f16], w: Tensor[128, 3, 128, f16]) -> Tensor[1, T, 128, f16]: ...


lt_axpy = make_ptx_op(write_ptx(tempfile.mkdtemp(prefix="tilefish-lt-")), name="lt_axpy")


@tilefish.graph
def lt_mlp(x: Tensor[M, 256, bf16], w_n: Tensor[256, bf16], w_up: Tensor[512, 256, bf16], b_up: Tensor[512, bf16], w_down: Tensor[256, 512, bf16], b_down: Tensor[256, bf16]):
    u = lt_up(lt_norm(x, w_n), w_up, b_up)
    return lt_down(tilefish.silu(u.to(f32)).to(bf16), w_down, b_down, x)


@tilefish.graph
def lt_conv(x: Tensor[1, T, 128, f16], w: Tensor[128, 3, 128, f16]):
    z = lt_conv1d(x, w).reshape(T, 128).to(f32) * 0.5
    return lt_axpy(z.reshape(T * 128))


GRAPHS = [lt_mlp, lt_conv]


def fake_pin(spec, cap):
    """What an export on the device pins for ``spec`` (the heuristic's algorithm, a verification and a canary), made up."""
    algo = hashlib.sha512(spec.stem.encode()).hexdigest()
    if isinstance(spec, _conv.ConvSpec):
        pin = _cudnn.pin_of({"api": "backend", "engine": 7, "knobs": [[1, 2], [3, 0]]})
        return _conv.BuiltConv(spec, f"sm_{cap}", pin, CUDNN, SMS, {"seed": 1, "y_digest": algo[:32]})
    if not spec.matmul:
        return _library.BuiltLibrary(spec, f"sm_{cap}", algo, CUBLASLT, SMS)
    verification = {
        "formula": _library.formula(spec.epilogue, spec.beta), "accumulator": "f32-form", "operand_sets": 2, "elements": 2 * spec.m * spec.n, "differ": 0,
        "max_ulp": 0,
    }  # fmt: skip
    return _library.BuiltLibrary(spec, f"sm_{cap}", algo, CUBLASLT, SMS, "measured-exact", verification, {"seed": 1, "out_digest": algo[32:64]})


def fake_verify(graph, plan, cap, verify):
    """What an export on the device's verification of the fused regions finds: every region checked at every bucket, two seeds."""
    if not _fusion.has_arithmetic(graph) or verify is False:
        return None
    return {r.index: {"runs": 4, "buckets": 2, "seeds": 2} for r in _fusion.fused(graph).fused_regions}


def no_device(*a, **k):
    raise AssertionError("a locked export asked a heuristic, a library or a device")


@pytest.fixture
def fake(monkeypatch):
    compiler = fake_compile.install(monkeypatch)
    monkeypatch.setattr(runtime, "_modules", {})
    monkeypatch.setattr(runtime, "_gen_root", [])
    monkeypatch.setattr(runtime, "_warned_gen_roots", set())
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.delenv(_trace.TRACE_ENV, raising=False)
    monkeypatch.setattr(_binary, "build_binary", lambda spec, target, *, arch: fake_binary_build(spec, target, arch=f"sm_{arch}", ptxas_version=PTXAS))
    monkeypatch.setattr(_binary, "find_ptxas", lambda: "/fake/ptxas")
    return compiler


@pytest.fixture
def gpu(fake, monkeypatch):
    """An export "on the GPU": the pins and the verification of the fused regions made up as a device would make them."""
    monkeypatch.setattr(_planrun, "_pin_library", fake_pin)
    monkeypatch.setattr(_planrun, "_verify_fusion", fake_verify)
    return monkeypatch


def lock_on_gpu(monkeypatch, tmp_path, ops=None, **kw):
    """Export ``ops`` (default: the plan set of :data:`GRAPHS`) "on the GPU" into ``tmp_path/gpu``, write its lock, then take the device away: every
    heuristic, library, verification and device query raises from then on."""
    with monkeypatch.context() as m:
        m.setattr(_planrun, "_pin_library", fake_pin)
        m.setattr(_planrun, "_verify_fusion", fake_verify)
        m.setattr(_library, "build_library", lambda spec, *, arch, canary=False, device_index=None: fake_pin(spec, arch))  # (an export of library ops)
        m.setattr(_conv, "build_conv", lambda spec, *, arch, canary=False, device_index=None: fake_pin(spec, arch))
        out = tmp_path / "gpu"
        runtime.export(GRAPHS if ops is None else ops, out, arch=ARCH, **kw)
        doc = runtime.write_lock(out, tmp_path / "sm_103.lock.json")
    for module, name in (
        (_planrun, "_pin_library"), (_planrun, "_verify_fusion"), (_library, "build_library"), (_conv, "build_conv"), (_cublaslt, "pin_algo"),
        (_cublaslt, "check_algo"), (_cudnn, "pin_conv"), (_cudnn, "check_pin"), (_fusion, "verify"), (_fusion, "verify_for_export"), (runtime, "_torch"),
    ):  # fmt: skip
        monkeypatch.setattr(module, name, no_device)
    return out, tmp_path / "sm_103.lock.json", doc


def files_of(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(pathlib.Path(directory).rglob("*")) if p.is_file()}


def edited(lock_path, tmp_path, edit, name="edited.lock.json"):
    doc = json.loads(lock_path.read_text())
    edit(doc)
    out = tmp_path / name
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return out


# -- the lock and the locked export ---------------------------------------------------------------------------------------------------------


def test_a_locked_export_writes_the_bytes_of_the_export_on_the_gpu_and_asks_no_heuristic(fake, monkeypatch, tmp_path):
    gpu_dir, lock, doc = lock_on_gpu(monkeypatch, tmp_path)
    got = runtime.export(GRAPHS, tmp_path / "cpu", lock=lock)
    assert files_of(tmp_path / "cpu") == files_of(gpu_dir)  # every file, byte for byte
    assert got.digest == doc["digests"]["set"]
    # the lock: the device, the toolchain, every pin with what the device measured, the verification of the regions, the digests, every file
    assert doc["schema"] == "tilefish.lock.v1" and doc["tilefish"] == tilefish.__version__
    assert doc["device"] == {"arch": "sm_103", "sm_count": SMS}
    assert doc["toolchain"] == {"cublaslt": CUBLASLT, "cudnn": CUDNN, "ptxas": PTXAS, "triton": "3.5.1"}
    assert sorted(doc["library"]) == ["lt_conv1d.T128", "lt_conv1d.T64", "lt_down.M32", "lt_down.M64", "lt_up.M32", "lt_up.M64"]
    assert set(doc["library"]["lt_up.M32"]) == {"impl", "declaration_digest", "workspace_bytes", "algo"}
    assert set(doc["library"]["lt_down.M32"]) == {"impl", "declaration_digest", "workspace_bytes", "algo", "contract", "verification", "canary"}
    assert set(doc["library"]["lt_conv1d.T64"]) == {"impl", "declaration_digest", "workspace_bytes", "pin", "canary"}
    assert sorted(doc["fusion"]) == ["lt_conv", "lt_mlp"] and all(r["verified"] == {"runs": 4, "buckets": 2, "seeds": 2} for rs in doc["fusion"].values() for r in rs)
    assert set(doc["digests"]["plans"]) == {"lt_mlp", "lt_conv"} and doc["digests"]["manifest"] == tilefish.Manifest.load(gpu_dir).digest
    assert doc["files"] == {name: hashlib.sha256(data).hexdigest() for name, data in files_of(gpu_dir).items()}
    assert {"manifest.json", "plans.json", "lt_mlp.plan.json", "lt_conv.plan.json", "lt_mlp.fusion.json", "lt_conv.fusion.json"} <= set(doc["files"])
    # canonical: the same directory gives the same bytes, and a locked export gives the same lock
    first = lock.read_bytes()
    runtime.write_lock(tmp_path / "cpu", tmp_path / "again.lock.json")
    assert (tmp_path / "again.lock.json").read_bytes() == first == _lock.canonical(doc)
    # a locked export into the directory it wrote writes the same files again (a merge of the same entries)
    runtime.export(GRAPHS, tmp_path / "cpu", lock=lock)
    assert files_of(tmp_path / "cpu") == files_of(gpu_dir)


def test_kernels_and_externs_and_one_graph_export_locked_too(fake, monkeypatch, tmp_path):
    gpu_dir, lock, doc = lock_on_gpu(monkeypatch, tmp_path, ops=[lt_norm, lt_axpy, lt_up, lt_conv1d])
    m = runtime.export([lt_norm, lt_axpy, lt_up, lt_conv1d], tmp_path / "cpu", lock=lock)
    assert files_of(tmp_path / "cpu") == files_of(gpu_dir) and m.digest == doc["digests"]["manifest"] and doc["digests"]["plans"] == {}
    gpu_dir, lock, doc = lock_on_gpu(monkeypatch, tmp_path / "one", ops=lt_mlp)
    p = runtime.export(lt_mlp, tmp_path / "one" / "cpu", lock=lock)
    assert files_of(tmp_path / "one" / "cpu") == files_of(gpu_dir) and p.digest == doc["digests"]["plans"]["lt_mlp"] and "set" not in doc["digests"]


def test_a_lock_of_an_export_without_verification_exports_without_it(fake, monkeypatch, tmp_path):
    gpu_dir, lock, doc = lock_on_gpu(monkeypatch, tmp_path, verify=False)
    assert doc["fusion"] == {}
    runtime.export(GRAPHS, tmp_path / "cpu", lock=lock, verify=False)
    assert files_of(tmp_path / "cpu") == files_of(gpu_dir)
    with pytest.raises(TilefishLockError, match=r"no verification of the fused region 0 .* re-lock on a GPU .*or export with verify=False"):
        runtime.export(GRAPHS, tmp_path / "verified", lock=lock)


# -- what a locked export refuses -------------------------------------------------------------------------------------------------------------


def test_a_tampered_hash_is_refused_naming_the_file(fake, monkeypatch, tmp_path):
    _, lock, doc = lock_on_gpu(monkeypatch, tmp_path)
    cubin = next(iter(doc["kernels"]))
    bad = edited(lock, tmp_path, lambda d: d["kernels"][cubin].update(sha256="0" * 64))
    with pytest.raises(TilefishLockError, match=rf"the cubin of '{cubin}' \(sm_103\) is not the lock's: .*its machine code \(sha256_code\) is the lock's"):
        runtime.export(GRAPHS, tmp_path / "a", lock=bad)
    assert not (tmp_path / "a").exists()  # refused before anything was written
    bad = edited(lock, tmp_path, lambda d: d["files"].update({"manifest.json": "0" * 64}))
    with pytest.raises(TilefishLockError, match=r"the manifest the export wrote, 'manifest.json', is not the lock's: sha256 [0-9a-f]{64} here, 0{64} in the lock"):
        runtime.export(GRAPHS, tmp_path / "b", lock=bad)
    bad = edited(lock, tmp_path, lambda d: d["files"].update({"lt_mlp.plan.json": "1" * 64}))
    with pytest.raises(TilefishLockError, match=r"a file the export wrote, 'lt_mlp.plan.json', is not the lock's"):
        runtime.export(GRAPHS, tmp_path / "c", lock=bad)
    bad = edited(lock, tmp_path, lambda d: d["files"].update({"extra.bin": "1" * 64}))
    with pytest.raises(TilefishLockError, match=r"the export did not write 'extra.bin', which the lock .* has"):
        runtime.export(GRAPHS, tmp_path / "d", lock=bad)
    (tmp_path / "e").mkdir()
    (tmp_path / "e" / "notes.txt").write_text("left over")
    with pytest.raises(TilefishLockError, match=r"has 'notes.txt', a file the lock .* does not have: export into an empty directory"):
        runtime.export(GRAPHS, tmp_path / "e", lock=lock)


def test_a_missing_pin_or_verification_is_refused_with_a_re_lock(fake, monkeypatch, tmp_path):
    _, lock, _ = lock_on_gpu(monkeypatch, tmp_path)
    for stem in ("lt_up.M64", "lt_down.M32", "lt_conv1d.T128"):
        bad = edited(lock, tmp_path, lambda d: d["library"].pop(stem))
        with pytest.raises(TilefishLockError, match=rf"has no pin for '{stem}' \(.*\) on sm_103: .*never asks a heuristic; re-lock on a GPU of the target device"):
            runtime.export(GRAPHS, tmp_path / stem, lock=bad)
    bad = edited(lock, tmp_path, lambda d: d["library"]["lt_down.M32"].pop("canary"))
    with pytest.raises(TilefishLockError, match=r"has no canary for 'lt_down.M32', and its declaration has one .*re-lock"):
        runtime.export(GRAPHS, tmp_path / "canary", lock=bad)
    bad = edited(lock, tmp_path, lambda d: d["fusion"].pop("lt_conv"))
    with pytest.raises(TilefishLockError, match=r"no verification of the fused region \d+ \('.*'\) of graph 'lt_conv'.*re-lock on a GPU"):
        runtime.export(GRAPHS, tmp_path / "fusion", lock=bad)
    bad = edited(lock, tmp_path, lambda d: d["kernels"].pop(next(iter(d["kernels"]))))
    with pytest.raises(TilefishLockError, match=r"the export builds '.*' \(sm_103\), and the lock .* has no binary of that name.*re-lock"):
        runtime.export(GRAPHS, tmp_path / "kernel", lock=bad)
    bad = edited(lock, tmp_path, lambda d: d["library"]["lt_up.M32"].update(declaration_digest="a" * 64))
    with pytest.raises(TilefishLockError, match=r"pins 'lt_up.M32' \(cublaslt.gemm_bias\) for another declaration: .*changed since the lock was made\); re-lock"):
        runtime.export(GRAPHS, tmp_path / "declaration", lock=bad)


def test_a_toolchain_or_a_tilefish_of_another_version_is_refused(fake, monkeypatch, tmp_path):
    _, lock, _ = lock_on_gpu(monkeypatch, tmp_path)
    bad = edited(lock, tmp_path, lambda d: d["toolchain"].update(ptxas="Cuda compilation tools, release 12.8, V12.8.93"))
    with pytest.raises(TilefishLockError, match=r"was made with the ptxas 'Cuda compilation tools, release 12.8, V12.8.93', and this process compiles with '.*V12.9.86'"):
        runtime.export(GRAPHS, tmp_path / "a", lock=bad)
    bad = edited(lock, tmp_path, lambda d: d["toolchain"].update(triton="3.6.0"))
    with pytest.raises(TilefishLockError, match=r"was made with Triton 3.6.0, and this process has Triton 3.5.1: .*install Triton 3.6.0, or re-lock"):
        runtime.export(GRAPHS, tmp_path / "b", lock=bad)
    bad = edited(lock, tmp_path, lambda d: d.update(tilefish="0.0.1"))
    with pytest.raises(TilefishLockError, match=rf"was made by tilefish 0.0.1, and this is tilefish {tilefish.__version__}"):
        runtime.export(GRAPHS, tmp_path / "c", lock=bad)
    # a binary built by another ptxas than the one the lock records (TILEFISH_PTXAS_PATH apart from Triton's)
    monkeypatch.setattr(_binary, "build_binary", lambda spec, target, *, arch: fake_binary_build(spec, target, arch=f"sm_{arch}", ptxas_version="ptxas 13.0"))
    with pytest.raises(TilefishLockError, match=r"'lt_axpy' \(sm_103\) was built with the ptxas 'ptxas 13.0', and the lock .* was made with '.*V12.9.86': install the lock's, or re-lock with this toolchain on a GPU"):
        runtime.export(GRAPHS, tmp_path / "d", lock=lock)


def test_a_lock_of_another_device_is_refused(fake, monkeypatch, tmp_path):
    _, lock, _ = lock_on_gpu(monkeypatch, tmp_path)
    with pytest.raises(TilefishLockError, match=r"is of a device of sm_103 with 30 SMs, and this export is for sm_120: a lock is of one device; use the lock of the sm_120 device"):
        runtime.export(GRAPHS, tmp_path / "a", lock=lock, arch="sm_120")
    runtime.export(GRAPHS, tmp_path / "b", lock=lock, arch=103)  # the lock's own architecture, said again


def test_a_malformed_lock_is_refused(fake, monkeypatch, tmp_path):
    _, lock, doc = lock_on_gpu(monkeypatch, tmp_path)
    text = lock.read_text()
    for edit, message in (
        (lambda d: d.update(schema="tilefish.lock.v2"), "schema must be 'tilefish.lock.v1'"),
        (lambda d: d.pop("files"), r"must have the fields .*missing \['files'\]"),
        (lambda d: d["library"]["lt_up.M32"].update(algo="xyz"), "library.lt_up.M32.algo must be 128 lowercase hex digits"),
        (lambda d: d["library"]["lt_conv1d.T64"].update(pin={"api": "backend", "engine": -1, "knobs": []}), "engine index from 0"),
        (lambda d: d["device"].update(sm_count=None), r"library.* pin, so toolchain.* and device.sm_count must be there"),
        (lambda d: d["files"].update({"../outside": "0" * 64}), "not a relative path inside the directory"),
        (lambda d: d["toolchain"].update(nvcc="12.9"), r"unknown \['nvcc'\]"),
    ):
        with pytest.raises(TilefishLockError, match=message):
            runtime.export(GRAPHS, tmp_path / "x", lock=edited(lock, tmp_path, edit))
    (tmp_path / "dup.lock.json").write_text(text.replace('"schema":', '"tilefish": "0", "schema":', 1))
    with pytest.raises(TilefishLockError, match="not valid JSON"):
        runtime.export(GRAPHS, tmp_path / "x", lock=tmp_path / "dup.lock.json")
    with pytest.raises(TilefishLockError, match="cannot read the lock"):
        runtime.export(GRAPHS, tmp_path / "x", lock=tmp_path / "missing.lock.json")
    with pytest.raises(TilefishLockError, match="lock= is the path of a lock file"):
        runtime.export(GRAPHS, tmp_path / "x", lock=doc)


# -- what a lock is not made of ---------------------------------------------------------------------------------------------------------------


def test_write_lock_takes_one_export_of_one_device_whose_regions_were_verified(fake, gpu, tmp_path):
    out = tmp_path / "out"
    runtime.export(GRAPHS, out, arch=ARCH)
    shutil.copytree(out, tmp_path / "junk")
    (tmp_path / "junk" / "notes.txt").write_text("not an export's")
    with pytest.raises(TilefishLockError, match=r"has files an export does not write \(notes.txt\)"):
        runtime.write_lock(tmp_path / "junk", tmp_path / "x.lock.json")
    runtime.export([lt_norm], out, arch=90)  # another architecture merged into the manifest
    with pytest.raises(TilefishLockError, match=r"has entries for \['sm_103', 'sm_90'\]: a lock is of one device"):
        runtime.write_lock(out, tmp_path / "x.lock.json")
    with pytest.raises(TilefishLockError, match="has no manifest.json"):
        runtime.write_lock(tmp_path / "nothing", tmp_path / "x.lock.json")
    # an export whose regions were not verified (no device of the architecture: the export warns and records why)
    gpu.setattr(_planrun, "_verify_fusion", lambda g, p, cap, verify: {r.index: {"skipped": "no CUDA device"} for r in _fusion.fused(g).fused_regions} if _fusion.has_arithmetic(g) else None)
    runtime.export(GRAPHS, tmp_path / "skipped", arch=ARCH)
    with pytest.raises(TilefishLockError, match=r"the fused regions of 'lt_.*' were not verified by the export \(no CUDA device\)"):
        runtime.write_lock(tmp_path / "skipped", tmp_path / "x.lock.json")
    assert not (tmp_path / "x.lock.json").exists()
