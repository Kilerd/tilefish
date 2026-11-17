"""The locked export (``docs/guide/locked-export.md``) on a real GPU: the plan set of ``locked_graphs`` (a generated kernel, Triton, Gluon, a prebuilt
PTX and cubin, a pinned cuBLASLt GEMM, a cuBLASLt matmul with an epilogue, a cuDNN convolution, fused regions, a global scratch in the workspace,
buckets, canaries) exported on the device and locked; the locked export of it, in a process that sees no CUDA device and cannot import torch, writes
the same files byte for byte; its plans replay (with the canaries run at the load) to the bytes ``jit`` gives; and every refusal, made by a locked
export in this process, on the device, asks no heuristic. The same artefacts in the Rust and C++ hosts, and a locked export on a machine without a
GPU: ``rust_gate.py prepare --locked DIR --lock FILE`` (the guide page).
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from gpu_util import same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishLockError, runtime  # noqa: E402
from tilefish import _conv, _cublaslt, _cudnn, _fusion, _library, _lock, _planrun  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def sm90_or_newer():
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available() or not sm90_or_newer(), reason="the locked set has TMA descriptors (sm_90 or newer)")]


@pytest.fixture(scope="module")
def locked(tmp_path_factory):
    """The set exported on this device into ``gpu/`` and its lock."""
    import locked_graphs as lg

    root = tmp_path_factory.mktemp("locked")
    runtime.export(list(lg.GRAPHS), root / "gpu")
    doc = runtime.write_lock(root / "gpu", root / "device.lock.json")
    return root, root / "device.lock.json", doc


def files_of(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(pathlib.Path(directory).rglob("*")) if p.is_file()}


NO_DEVICE = """
import sys
sys.modules["torch"] = None  # no torch: `import torch` raises ImportError
sys.path.insert(0, sys.argv[1])
import locked_graphs as lg
from tilefish import runtime
s = runtime.export(list(lg.GRAPHS), sys.argv[2], lock=sys.argv[3])
print(s.digest)
"""


def test_a_locked_export_without_a_device_writes_the_files_of_the_export_on_it(locked, tmp_path):
    root, lock, doc = locked
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TILEFISH_CACHE_DIR": str(tmp_path / "cache")}  # no device, and every binary compiled anew
    done = subprocess.run([sys.executable, "-c", NO_DEVICE, str(HERE), str(tmp_path / "cpu"), str(lock)], env=env, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    assert done.stdout.split()[-1] == doc["digests"]["set"]
    assert files_of(tmp_path / "cpu") == files_of(root / "gpu")
    # every kind of callee is in it
    kinds = {e.get("source_digest_kind") for e in json.loads((root / "gpu" / "manifest.json").read_text())["kernels"]}
    assert {"tilefish.triton-ir.v1", "tilefish.gluon-ir.v1", "tilefish.binary.v1"} <= kinds
    assert {r["impl"] for r in doc["library"].values()} == {"cublaslt.gemm_bias", "cublaslt.matmul", "cudnn.conv_fwd"}
    assert all("canary" in r for r in doc["library"].values() if r["impl"] != "cublaslt.gemm_bias") and sorted(doc["fusion"]) == ["lk_conv", "lk_mlp"]


def test_the_locked_plans_replay_with_their_canaries_to_the_bytes_of_jit(locked, tmp_path):
    import locked_graphs as lg

    root, lock, doc = locked
    out = tmp_path / "locked"
    runtime.export(list(lg.GRAPHS), out, lock=lock)  # on this machine, with its device: nothing of it is asked
    for g in lg.GRAPHS:
        for label, args in lg.cases(g):
            want = tilefish.run(g, *args.values())
            got = tilefish.run(g, *args.values(), mode="replay", plan=out, pin=doc["digests"]["set"])
            want, got = (list(v) if isinstance(v, tuple) else [v] for v in (want, got))
            assert len(want) == len(got) and all(same_bytes(a, b) for a, b in zip(want, got)), (g.name, label)


def test_every_refusal_asks_no_heuristic(locked, tmp_path, monkeypatch):
    import locked_graphs as lg

    root, lock, doc = locked

    def no_device(*a, **k):
        raise AssertionError("a locked export asked a heuristic, a library or a device")

    for module, name in ((_planrun, "_pin_library"), (_library, "build_library"), (_conv, "build_conv"), (_cublaslt, "pin_algo"), (_cudnn, "pin_conv"), (_fusion, "verify")):
        monkeypatch.setattr(module, name, no_device)
    graphs = list(lg.GRAPHS)

    def edited(edit):
        d = json.loads(lock.read_text())
        edit(d)
        path = tmp_path / f"edited{len(list(tmp_path.glob('edited*')))}.lock.json"
        path.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
        return path

    cubin = next(iter(doc["kernels"]))
    with pytest.raises(TilefishLockError, match=rf"the cubin of '{cubin}' \(sm_\d+\) is not the lock's"):
        runtime.export(graphs, tmp_path / "a", lock=edited(lambda d: d["kernels"][cubin].update(sha256="0" * 64)))
    with pytest.raises(TilefishLockError, match=r"the manifest the export wrote, 'manifest.json', is not the lock's"):
        runtime.export(graphs, tmp_path / "b", lock=edited(lambda d: d["files"].update({"manifest.json": "0" * 64})))
    stem = next(s for s, r in doc["library"].items() if r["impl"] == "cublaslt.matmul")
    with pytest.raises(TilefishLockError, match=rf"has no pin for '{stem}' \(cublaslt.matmul\) on sm_\d+: .*re-lock on a GPU"):
        runtime.export(graphs, tmp_path / "c", lock=edited(lambda d: d["library"].pop(stem)))
    with pytest.raises(TilefishLockError, match=r"was made with the ptxas 'ptxas 0.0', and this process compiles with"):
        runtime.export(graphs, tmp_path / "d", lock=edited(lambda d: d["toolchain"].update(ptxas="ptxas 0.0")))
    other = "sm_120" if doc["device"]["arch"] != "sm_120" else "sm_103"
    with pytest.raises(TilefishLockError, match=rf"is of a device of {doc['device']['arch']} with \d+ SMs, and this export is for {other}"):
        runtime.export(graphs, tmp_path / "e", lock=lock, arch=other)
    assert not any((tmp_path / name).exists() for name in "acde")  # refused before anything was written (a file's sha256 is held to the lock after)
