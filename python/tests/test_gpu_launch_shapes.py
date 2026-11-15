"""Launch shapes on a real GPU (sm_90 or newer): the kernels of ``launch_kernels.py``. Manifest schema 3, sections 4.3 to 4.6.

Held here, each in ``jit``, in ``replay`` of an exported manifest (with a pin) and, for the plans, in a plan replay, a ``tilefish.capture`` replay and a
``torch.cuda.graph`` replay, every result equal in bytes to torch's:

* a grid that multiplies two dimensions (``cdiv(T * C, 1024)``): one binary for every ``T`` and ``C``;
* a persistent grid: the programs are cut to one per SM of the device, or of the partition ``tilefish.runtime.sm_partition`` says, and the bytes are
  the same for every SM count; a kernel declared for 30 SMs (``sm_count=30``) runs on 30 and is refused on another count;
* a configuration per bucket (``by_bucket=``): each bucket's binary has its warps and its tile, recorded in its manifest entry;
* thread-block clusters: a Triton kernel of ``num_ctas=2`` (launched without the cluster it gives wrong bytes) and a prebuilt PTX whose blocks read each
  other's shared memory in a cluster of two (it requires a cluster launch);
* a global scratch: device-side TMA descriptors (``tl.make_tensor_descriptor``) written into the scratch the host gives each launch (a plan from its
  arena's workspace).
"""

import ctypes

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:  # pragma: no cover - clusters and TMA need sm_90
    pytest.skip("thread-block clusters and TMA descriptors need sm_90 or newer", allow_module_level=True)

import json  # noqa: E402

import launch_kernels as lk  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishRuntimeError, runtime  # noqa: E402
from tilefish import _cuda  # noqa: E402

F32 = torch.float32
CAP = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
ARCH = f"sm_{CAP[0] * 10 + CAP[1]}"


def sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    ops = [op for op in lk.ops().values() if op is not lk.persistent_pinned or sms() == 30]
    directory = tmp_path_factory.mktemp("launch_shapes")
    manifest = runtime.export(ops, directory)
    return directory, manifest


def run_both(op, exported, *args):
    directory, manifest = exported
    jit = tilefish.run(op, *args)
    replay = tilefish.run(op, *args, mode="replay", manifest=directory, pin=manifest.digest)
    assert same_bytes(jit, replay), f"{op.name}: jit and replay differ"
    return replay


def entry(exported, stem):
    directory, _ = exported
    doc = json.loads((directory / "manifest.json").read_text())
    return next(e for e in doc["kernels"] if e["stem"] == stem and e["arch"] == ARCH)


# -- a grid that multiplies -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("t, c", [(1, 1), (7, 300), (33, 1024), (1000, 3), (513, 515)])
def test_one_binary_serves_every_product_of_the_dimensions(exported, t, c):
    x = randn((t, c), F32, seed=t * 7 + c)
    y = run_both(lk.product_scale, exported, x)
    assert same_bytes(y, x * 2 + 1)
    assert entry(exported, "product_scale")["grid"] == "(cdiv(C*T, 1024),)"


# -- persistent grids -----------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 1000, 1025, 30 * 1024 + 1, 1 << 22])
def test_a_persistent_grid_gives_the_same_bytes_on_every_sm_count(exported, n):
    x = randn((n,), F32, seed=n % 97)
    want = x * 2 + 0.25
    assert same_bytes(run_both(lk.persistent_scale, exported, x), want)
    for partition in (1, 8, 16, sms()):
        with runtime.sm_partition(partition):
            assert same_bytes(run_both(lk.persistent_scale, exported, x), want), partition
    assert entry(exported, "persistent_scale")["persistent"] == {"programs_per_sm": 1}


def test_a_persistent_kernel_is_launched_with_at_most_one_program_per_sm(exported):
    directory, manifest = exported
    built = manifest.built_kernel(runtime._target(lk.persistent_scale).compile_specs()[0], ARCH, pinned=True)
    loaded = _cuda.load(built, 0)
    assert loaded.sms == sms() and loaded.programs((4096,)) == (sms(),) and loaded.programs((5,)) == (5,)
    with runtime.sm_partition(8):
        assert _cuda.load(built, 0).programs((4096,)) == (8,)  # another load: the grid of a partition of 8 SMs


def test_a_persistent_kernel_declared_for_an_sm_count_runs_on_that_count_only(exported):
    if sms() != 30:
        pytest.skip("persistent_pinned is declared for 30 SMs (a B300 MIG slice of 1g)")
    x = randn((70000,), F32, seed=3)
    assert same_bytes(run_both(lk.persistent_pinned, exported, x), x * 2 + 0.25)
    assert entry(exported, "persistent_pinned")["persistent"] == {"programs_per_sm": 1, "sm_count": 30}
    with runtime.sm_partition(16), pytest.raises(TilefishRuntimeError, match="declared for 30 SMs"):
        tilefish.run(lk.persistent_pinned, x)


# -- a configuration per bucket ------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [64, 4096])
def test_each_bucket_runs_the_binary_of_its_configuration(exported, rows):
    x = randn((rows, 128), F32, seed=rows)
    y = run_both(lk.rows_by_bucket, exported, x)
    assert same_bytes(y, x - x.max(dim=1, keepdim=True).values)
    e = entry(exported, f"rows_by_bucket.TB{rows}")
    warps, block_t = (2, 16) if rows == 64 else (8, 64)
    assert (e["num_warps"], e["block"], e["grid"]) == (warps, [32 * warps, 1, 1], f"({rows // block_t},)")
    assert next(c["value"] for c in e["constexprs"] if c["name"] == "BLOCK_T") == block_t


# -- thread-block clusters ------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 1000, 2048, 65537])
def test_a_triton_kernel_of_two_ctas_per_program_runs_as_clusters(exported, n):
    x = randn((n,), F32, seed=n % 89)
    assert same_bytes(run_both(lk.cluster_scale, exported, x), x * 2 + 1)
    e = entry(exported, "cluster_scale")
    assert (e["options"]["num_ctas"], e["launch"]["cluster_dims"], e["launch"]["num_ctas"]) == (2, [2, 1, 1], 2)


def test_a_kernel_of_clusters_launched_without_its_cluster_gives_other_bytes(exported):
    """What makes the cluster launch necessary: the same binary through a plain ``cuLaunchKernel`` of the grid of programs computes something else."""
    directory, manifest = exported
    spec = runtime._target(lk.cluster_scale).compile_specs()[0]
    built = manifest.built_kernel(spec, ARCH, pinned=True)
    loaded = _cuda.load(built, 0)
    n = 5000
    x = randn((n,), F32, seed=1)
    y = torch.full((n,), float("nan"), device="cuda")
    drv = _cuda._driver()
    vals = [ctypes.c_uint64(x.data_ptr()), ctypes.c_uint64(y.data_ptr()), ctypes.c_int32(n), ctypes.c_uint64(0), ctypes.c_uint64(0)]
    slots = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])
    stream = torch.cuda.current_stream().cuda_stream
    _cuda._check(drv.cuLaunchKernel(loaded.function, 5, 1, 1, 128, 1, 1, 0, drv.CUstream(stream), ctypes.addressof(slots), 0), "cuLaunchKernel")
    torch.cuda.synchronize()
    assert not same_bytes(y, x * 2 + 1)


@pytest.mark.parametrize("n", [1, 200, 256, 1000, 70000])
def test_a_prebuilt_kernel_of_clusters_reads_its_partners_shared_memory(exported, n):
    x = randn((n,), F32, seed=n % 83)
    y = run_both(lk.cluster_swap, exported, x)
    assert same_bytes(y, lk.swap_reference(x))
    e = entry(exported, "cluster_swap")
    assert (e["launch"]["cluster_dims"], e["launch"]["num_ctas"]) == ([2, 1, 1], 2)


# -- a global scratch ----------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 64, 100, 1000])
def test_device_side_descriptors_write_their_maps_into_the_global_scratch_of_the_launch(exported, rows):
    x = randn((rows, 64), F32, seed=rows)
    assert same_bytes(run_both(lk.desc_scale, exported, x), x * 2)
    e = entry(exported, "desc_scale")
    assert e["launch"]["global_scratch_size"] == 256 and e["launch"]["global_scratch_align"] in (64, 128, 256)


# -- plans ---------------------------------------------------------------------------------------------------------------------------------------------

PLAN_CASES = {
    "grid_plan": (lambda b: randn((b["Rb"], b["Cb"]), F32, seed=b["Rb"] + b["Cb"]), lambda x: (x * 2 + 1) * 2 + 1),
    "persistent_plan": (lambda b: randn((b["Nb"],), F32, seed=b["Nb"] % 101), lambda x: (x * 2 + 0.25) * 2 + 0.25),
    "bucket_plan": (lambda b: randn((b["TB"], 128), F32, seed=b["TB"]), lambda x: x - x.max(dim=1, keepdim=True).values),
    "cluster_plan": (lambda b: randn((b["Nb"],), F32, seed=b["Nb"] % 103), lambda x: lk.swap_reference(x * 2 + 1)),
    "scratch_plan": (lambda b: randn((b["Mb"], 64), F32, seed=b["Mb"]), lambda x: x * 4),
}


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    out = {}
    for name, g in lk.graphs().items():
        directory = tmp_path_factory.mktemp(name)
        out[name] = (g, directory, runtime.export(g, directory))
    return out


@pytest.mark.parametrize("name", list(PLAN_CASES))
def test_a_plan_of_each_launch_shape_runs_and_captures_with_torchs_bytes(plans, name):
    g, directory, plan = plans[name]
    make, expect = PLAN_CASES[name]
    for bucket in plan.buckets():
        x = make(bucket)
        want = expect(x)
        jit = tilefish.run(g, x)
        replay = tilefish.run(g, x, mode="replay", plan=directory, pin=plan.digest)
        assert same_bytes(jit, want) and same_bytes(replay, want), (name, bucket)
        out = torch.empty_like(want)
        captured = tilefish.capture(g, x, mode="replay", plan=directory, pin=plan.digest, outputs=out)
        captured.replay()
        torch.cuda.synchronize()
        assert same_bytes(out, want), (name, bucket, "capture")
        graph_out = torch.empty_like(want)
        runtime.prepare(g, mode="replay", plan=directory, pin=plan.digest)
        torch.cuda.synchronize()
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            tilefish.run(g, x, mode="replay", plan=directory, pin=plan.digest, outputs=graph_out)
        cuda_graph.replay()
        torch.cuda.synchronize()
        assert same_bytes(graph_out, want), (name, bucket, "torch.cuda.graph")


def test_a_plan_gives_its_scratch_kernels_the_workspace_of_the_arena(plans):
    g, directory, plan = plans["scratch_plan"]
    ws = next(b for b in plan.doc["buffers"] if b["role"] == "workspace")
    assert ws["bytes"] == 256 * 16  # cdiv(1000, 64) programs of the largest bucket, 256 bytes each
    assert all(st["workspace"] == ws["name"] for b in plan.doc["buckets"] for st in b["steps"])


def test_a_plan_of_a_persistent_kernel_runs_on_a_partition(plans):
    g, directory, plan = plans["persistent_plan"]
    x = randn((70000,), F32, seed=5)
    with runtime.sm_partition(4):
        y = tilefish.run(g, x, mode="replay", plan=directory, pin=plan.digest)
    assert same_bytes(y, (x * 2 + 0.25) * 2 + 0.25)
