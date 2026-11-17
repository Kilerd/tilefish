"""CuTe DSL kernels as externs on a real GPU (``@tilefish.extern(cute=...)``, ``docs/en/cute.md``), with public kernels of the CUTLASS examples.

Needs, besides the ``gpu`` extra, the CuTe DSL (``nvidia-cutlass-dsl``) and ``TILEFISH_CUTLASS_EXAMPLES``, the ``examples/python/CuTeDSL`` directory of a
CUTLASS checkout (``cute_kernels.py``); skipped otherwise. What is held:

* **The bytes of the DSL's own launch.** tilefish launches the DSL's cubin with the parameters it took from the captured launch (pointers, TMA
  descriptors whose bytes from 64 on it passes as zeros, constant bytes): the output is byte for byte the output of the compiled host function the DSL
  launches itself, for the tcgen05 GEMM (three TMA descriptors, a cluster of 2) and the elementwise kernels.
* **Correct.** The elementwise ``add`` is torch's ``a + b``, byte for byte; the GEMM is ``a @ b.T`` within the tolerance of an f16 result.
* **Pinned and reproducible.** An export is the same files and the same manifest digest in another process, from another directory; a replay (pinned)
  is the bytes of ``jit``; a replay refuses a binary of another declaration and one whose PTX the DSL no longer makes, pinned or not.
* **Plans and graphs.** A graph of a CuTe GEMM and a CuTe add exports a plan whose replay, and whose CUDA graph capture, are the eager calls' bytes.
* **SASS assertions and traces** apply as for any prebuilt binary.
* **Speed** of the GEMM against the pinned cuBLASLt GEMM of the same problem (printed, not asserted).

The GEMM is the sm_100/sm_103 tcgen05 kernel or, on sm_120, the warp-MMA kernel of the examples (one operator, an implementation per architecture).
"""

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("cuda.bindings.driver")
pytest.importorskip("cutlass")
if not os.environ.get("TILEFISH_CUTLASS_EXAMPLES"):
    pytest.skip("TILEFISH_CUTLASS_EXAMPLES (the examples/python/CuTeDSL directory of a CUTLASS checkout) is not set", allow_module_level=True)

from gpu_util import randn, same_bytes  # noqa: E402

import cute_kernels as ck  # noqa: E402
import tilefish  # noqa: E402
from tilefish import TilefishBuildError, TilefishManifestError, runtime  # noqa: E402
from tilefish import _cute  # noqa: E402
from tilefish.aot import compile_specs  # noqa: E402

F16 = torch.float16
HERE = pathlib.Path(__file__).resolve().parent


def capability():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


needs_gemm = pytest.mark.skipif(capability() not in (100, 103, 120), reason="the GEMMs of the examples are for sm_100, sm_103 and sm_120")


def gemm_args(m, seed):
    return randn((m, ck.K), F16, seed=seed), randn((ck.N, ck.K), F16, seed=seed + 1)


def dsl_launch(op, tensors, outputs):
    """What the DSL computes itself: the host function compiled and called (on the operator's spec of these shapes) on ``tensors`` + ``outputs``."""
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda_driver

    extern = runtime._target(op).at(capability()).launch_extern()
    shapes = [tuple(t.shape) for t in [*tensors, *outputs]]
    spec = next(s for s in compile_specs(extern) if [tuple(int(x) for x in a.shape) for a in s.args] == shapes)
    target = extern.binary
    args = _cute._host_args(target, [*tensors, *outputs], spec.stem)
    stream = torch.cuda.current_stream()
    with tempfile.TemporaryDirectory() as dump:  # (the DSL writes what it keeps to the working directory otherwise)
        compiled = cute.compile(target.host, *args, cuda_driver.CUstream(stream.cuda_stream), options=f"--dump-dir={dump} {target.options}".strip())
    compiled(*_cute._runtime_args(compiled, [*args, cuda_driver.CUstream(stream.cuda_stream)]))
    torch.cuda.synchronize()


# -- the elementwise kernels --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [64, 4096])
def test_add_is_torchs_sum_byte_for_byte(rows):
    a, b = randn((rows, ck.C), F16, seed=rows), randn((rows, ck.C), F16, seed=rows + 1)
    got = tilefish.run(ck.add, a, b, mode="jit")
    assert same_bytes(got, a + b)


@pytest.mark.parametrize("op, inputs, dtype", [(ck.add, 2, F16), (ck.leaky_relu, 1, torch.float32)])
def test_an_elementwise_launch_is_the_dsls_own_launch(op, inputs, dtype):
    xs = [randn((4096, ck.C), dtype, seed=11 + i) for i in range(inputs)]
    got = tilefish.run(op, *xs, mode="jit")
    own = torch.empty_like(xs[0])
    dsl_launch(op, xs, [own])
    assert same_bytes(got, own)


# -- the GEMM -----------------------------------------------------------------------------------------------------------------------------------


@needs_gemm
@pytest.mark.parametrize("m", [256, 1024, 4096])
def test_the_gemm_is_a_times_b_transposed(m):
    a, b = gemm_args(m, seed=m)
    got = tilefish.run(ck.gemm, a, b, mode="jit")
    want = (a.float() @ b.float().T).to(F16)
    error = (got.float() - want.float()).abs().max().item()
    assert error <= 0.125, error  # an f16 result of an f32 accumulation over K = 1024: within an ulp of |c| ~ 32 here


@needs_gemm
@pytest.mark.parametrize("m", [256, 4096])
def test_a_gemm_launch_is_the_dsls_own_launch(m):
    a, b = gemm_args(m, seed=3 * m)
    got = tilefish.run(ck.gemm, a, b, mode="jit")
    own = torch.empty((m, ck.N), dtype=F16, device="cuda")
    dsl_launch(ck.gemm, [a, b], [own])
    assert same_bytes(got, own)


needs_tcgen05 = pytest.mark.skipif(capability() not in (100, 103), reason="the tcgen05 GEMMs of the examples are for sm_100 and sm_103")


def built_of(op, m):
    target = runtime._target(op).at(capability())
    spec = next(s for s in target.compile_specs() if s.buckets == {"M": m})
    return runtime.jit_built(target, spec, torch.cuda.current_device())


@needs_tcgen05
@pytest.mark.parametrize("m", [256, 4096])
def test_the_dense_tcgen05_gemm_is_correct_and_the_dsls_own_launch(m):
    a, b = gemm_args(m, seed=5 * m)
    got = tilefish.run(ck.gemm_dense, a, b, mode="jit")
    own = torch.empty((m, ck.N), dtype=F16, device="cuda")
    dsl_launch(ck.gemm_dense, [a, b], [own])
    assert same_bytes(got, own) and same_bytes(got, tilefish.run(ck.gemm, a, b, mode="jit"))  # the persistent one adds in the same order


@needs_tcgen05
def test_the_dense_tcgen05_gemm_takes_three_tma_descriptors_and_a_cluster_per_tile():
    built = built_of(ck.gemm_dense, 1024)
    kinds = [(p["kind"], p["name"]) for p in built.abi()["params"]]
    assert [k for k in kinds if k[0] != "bytes"] == [("tensormap", "a"), ("tensormap", "b"), ("tensormap", "c")]
    assert built.launch.cluster_dims == (2, 1, 1) and built.block == (128, 1, 1) and built.shared > 48 * 1024
    assert built.grid == (1024 // 256, ck.N // 128, 1)  # 2-CTA tiles of 256 rows, clusters of 2 CTAs along M
    assert built.binary_build == {"source": "cute", "cutlass_dsl": _cute.dsl_version(), "options": ""}
    assert "tcgen05" in built.ptx and built.ptx_target == f"sm_{capability()}a"


@needs_tcgen05
def test_the_persistent_gemm_passes_its_tile_scheduler_without_the_padding_it_does_not_read():
    built = built_of(ck.gemm, 4096)
    params = built.abi()["params"]
    assert [(p["kind"], p["name"]) for p in params if p["kind"] != "bytes"] == [("tensormap", "a"), ("tensormap", "b"), ("tensormap", "c")]
    divisors = [p for p in params if p["kind"] == "bytes" and p["size"] == 12]
    assert divisors and all(p["value"].endswith("0000") for p in divisors)  # {divisor, multiplier, shift, shift}, then 2 bytes of padding: zero
    clusters = torch.cuda.get_device_properties(0).multi_processor_count // 2
    assert built.launch.cluster_dims == (2, 1, 1) and 1 <= built.grid[0] * built.grid[1] * built.grid[2] <= clusters


# -- pinned, reproducible, stale --------------------------------------------------------------------------------------------------------------


def _ops():
    return list(ck.ops(torch.cuda.get_device_capability()).values())


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cute-export")
    return directory, runtime.export(_ops(), directory)


def test_an_export_is_the_same_files_in_another_process_and_directory(exported, tmp_path):
    directory, manifest = exported
    other = tmp_path / "again"
    code = (
        "import sys; sys.path.insert(0, sys.argv[2]); import cute_kernels as ck, torch, tilefish; from tilefish import runtime; "
        "ops = list(ck.ops(torch.cuda.get_device_capability()).values()); "
        "print(runtime.export(ops, sys.argv[1]).digest)"
    )
    done = subprocess.run([sys.executable, "-c", code, str(other), str(HERE)], capture_output=True, text=True, cwd=tmp_path, timeout=1200, env=os.environ)
    assert done.returncode == 0, done.stderr[-4000:]
    assert done.stdout.strip().splitlines()[-1] == manifest.digest
    files = sorted(p.name for p in directory.iterdir() if p.is_file())
    assert files == sorted(p.name for p in other.iterdir() if p.is_file())
    for name in files:
        assert (directory / name).read_bytes() == (other / name).read_bytes(), name


def test_a_pinned_replay_is_the_bytes_of_jit(exported):
    directory, manifest = exported
    a, b = randn((64, ck.C), F16, seed=5), randn((64, ck.C), F16, seed=6)
    assert same_bytes(tilefish.run(ck.add, a, b, mode="replay", manifest=directory, pin=manifest.digest), tilefish.run(ck.add, a, b, mode="jit"))
    if capability() in (100, 103, 120):
        a, b = gemm_args(1024, seed=9)
        replay = tilefish.run(ck.gemm, a, b, mode="replay", manifest=directory, pin=manifest.digest)
        assert same_bytes(replay, tilefish.run(ck.gemm, a, b, mode="jit"))


def test_the_entries_say_the_dsl_and_its_tools(exported):
    directory, manifest = exported
    doc = json.loads((directory / "manifest.json").read_text())
    for e in doc["kernels"]:
        assert e["kind"] == "binary" and e["binary"]["source"] == "cute" and e["source_digest_kind"] == "tilefish.cute.v1"
        assert e["toolchain"]["cutlass_dsl"] == _cute.dsl_version()
        assert e["toolchain"]["nvvm"].startswith("Cuda compilation tools") and e["toolchain"]["ptxas"].startswith("Cuda compilation tools")


def test_a_replay_refuses_a_binary_whose_ptx_the_dsl_no_longer_makes(exported, monkeypatch):
    directory, manifest = exported
    a, b = randn((64, ck.C), F16, seed=1), randn((64, ck.C), F16, seed=2)
    runtime._cute_digests.clear()
    monkeypatch.setattr(_cute, "ptx_digest", lambda spec, target, index: "f" * 64)
    with pytest.raises(TilefishManifestError, match="now compiles to PTX with sha256 f{64}"):
        tilefish.run(ck.add, a, b, mode="replay", manifest=directory, pin=manifest.digest)
    runtime._cute_digests.clear()


def test_a_replay_refuses_the_binary_of_another_declaration(exported):
    directory, manifest = exported

    @tilefish.extern(cute=ck.elementwise.elementwise_apply, cute_args=lambda a, b, c: (__import__("operator").add, [a, b], c), cute_options="--opt-level 2", name="cute_add")
    def add_o2(a: tilefish.Tensor[ck.R, ck.C, tilefish.f16], b: tilefish.Tensor[ck.R, ck.C, tilefish.f16], c: tilefish.Out[tilefish.Tensor[ck.R, ck.C, tilefish.f16]]): ...

    a, b = randn((64, ck.C), F16, seed=1), randn((64, ck.C), F16, seed=2)
    with pytest.raises(TilefishManifestError, match="first difference at declaration_digest"):
        tilefish.run(add_o2, a, b, mode="replay", manifest=directory, pin=manifest.digest)


# -- plans --------------------------------------------------------------------------------------------------------------------------------------


@needs_gemm
def test_a_plan_of_cute_kernels_replays_and_captures_with_the_bytes_of_the_eager_calls(tmp_path):
    plan = runtime.export(ck.gemm_then_add, tmp_path)
    for m in (256, 1024):
        a, b = gemm_args(m, seed=m + 7)
        r = randn((m, ck.N), F16, seed=m + 8)
        eager = tilefish.run(ck.add_mn, tilefish.run(ck.gemm, a, b, mode="jit"), r, mode="jit")
        replay = tilefish.run(ck.gemm_then_add, a, b, r, mode="replay", plan=tmp_path, pin=plan.digest)
        assert same_bytes(replay, eager)
        cap = tilefish.capture(ck.gemm_then_add, a, b, r, mode="replay", plan=tmp_path, pin=plan.digest)
        for _ in range(2):
            cap.replay()
            torch.cuda.synchronize()
            assert same_bytes(cap.outputs[0] if isinstance(cap.outputs, (list, tuple)) else cap.outputs, eager)


# -- SASS assertions and traces ---------------------------------------------------------------------------------------------------------------


def test_sass_assertions_are_checked_on_a_cute_binary():
    def declare(sass):
        @tilefish.extern(cute=ck.elementwise.elementwise_apply, cute_args=lambda a, b, c: (__import__("operator").add, [a, b], c), name="add_sass", sass=sass)
        def add_sass(a: tilefish.Tensor[64, ck.C, tilefish.f16], b: tilefish.Tensor[64, ck.C, tilefish.f16], c: tilefish.Out[tilefish.Tensor[64, ck.C, tilefish.f16]]): ...

        return add_sass

    a, b = randn((64, ck.C), F16, seed=1), randn((64, ck.C), F16, seed=2)
    assert same_bytes(tilefish.run(declare(tilefish.sass(no_spills=True, no_tf32=True)), a, b, mode="jit"), a + b)
    with pytest.raises(TilefishBuildError, match="max_registers"):
        tilefish.run(declare(tilefish.sass(max_registers=2)), a, b, mode="jit")


def test_a_traced_export_keeps_the_ptx_of_a_cute_kernel(tmp_path):
    manifest = runtime.export([ck.add], tmp_path / "traced", trace=True)
    entry = next(e for e in json.loads((tmp_path / "traced" / "manifest.json").read_text())["kernels"] if e["stem"] == "cute_add.R64")
    ptx = tmp_path / "traced" / "trace" / f"cute_add.R64.{entry['arch']}" / "ptx.ptx"
    assert hashlib.sha256(ptx.read_bytes()).hexdigest() == entry["source_digest"]  # the PTX the DSL generated, which the entry is identified by
    assert (tmp_path / "traced" / "trace" / f"cute_add.R64.{entry['arch']}" / "sass.txt").exists()
    assert manifest.digest == runtime.export([ck.add], tmp_path / "plain").digest  # the trace is outside the pinned artefact


# -- speed ------------------------------------------------------------------------------------------------------------------------------------


def _graph_time(fn, reps=50):
    """The device time of one call of ``fn`` in microseconds: ``reps`` calls captured in one CUDA graph (so that no host overhead is timed), replayed."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / (5 * reps)


@needs_gemm
def test_the_gemm_against_the_pinned_cublaslt_gemm(tmp_path, capsys):
    from tilefish import Out, Tensor, f16, f32  # noqa: F401

    @tilefish.library("gemm_cublaslt", acc=f32, impl="cublaslt.gemm")
    def lt(a: Tensor[ck.M, ck.K, f16], b: Tensor[ck.N, ck.K, f16]) -> Tensor[ck.M, ck.N, f16]: ...

    gemms = [("persistent", ck.gemm)] + ([("dense", ck.gemm_dense)] if capability() in (100, 103) else [])
    manifest = runtime.export([op for _, op in gemms] + [lt], tmp_path)
    rows = []
    for m in (256, 1024, 4096):
        a, b = gemm_args(m, seed=m)
        c = torch.empty((m, ck.N), dtype=F16, device="cuda")
        lt_us = _graph_time(lambda: tilefish.run(lt, a, b, mode="replay", manifest=tmp_path, pin=manifest.digest))
        torch_us = _graph_time(lambda: torch.mm(a, b.T))
        ref = tilefish.run(lt, a, b, mode="replay", manifest=tmp_path, pin=manifest.digest)
        for label, op in gemms:
            us = _graph_time(lambda: tilefish.run(op, a, b, mode="replay", manifest=tmp_path, pin=manifest.digest, outputs=c))
            got = tilefish.run(op, a, b, mode="replay", manifest=tmp_path, pin=manifest.digest)
            rows.append((m, label, us, lt_us, torch_us, same_bytes(got, ref)))
            assert (got.float() - ref.float()).abs().max().item() <= 0.125
    with capsys.disabled():
        print(f"\nCuTe DSL GEMMs vs cuBLASLt on sm_{capability()}, {torch.cuda.get_device_properties(0).multi_processor_count} SMs (N={ck.N}, K={ck.K}, f16, device time per call in a CUDA graph, µs):")
        for m, label, us, lt_, th, same in rows:
            flops = 2 * m * ck.N * ck.K
            print(f"  M={m:5d}  cute {label:10s} {us:8.1f} ({flops / us / 1e6:5.0f} TFLOP/s, x{lt_ / us:4.2f})  cublaslt {lt_:8.1f}  torch.mm {th:8.1f}  bytes == cublaslt: {same}")
