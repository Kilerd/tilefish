"""The gate of launch plans, on a real GPU: a pre-norm transformer block planned, exported, replayed, captured in a CUDA graph.

``plan_model.block`` is a generic block with bf16 activations and the rows bucketed over M in {32, 64, 128}: a lowered rmsnorm, a pinned cuBLASLt GEMM with a bias (the QKV
projection), q, k and v as strided views of its fused output, a hand-written Triton attention that reads the views and masks the keys at or beyond a device scalar ``n``, a GEMM, a
lowered residual add and rmsnorm in one kernel, a GEMM, a lowered gelu, a GEMM and a lowered residual add. Checked here, from an empty tilefish and Triton cache:

(a) plan ``replay``, plan ``jit`` and the per-op eager composition of the same ops (``tilefish.run`` of each op on its own tensors) give the same bytes, for every bucket and two values of ``n``;
(b) ``replay`` against ``reference`` (the ops' references composed in torch): the error per output is reported and bounded;
(c) a replay under ``torch.cuda.graph`` captures, allocates nothing, and the graph replays byte-identically (also after the inputs and ``n`` change);
(d) a tampered ``plan.json``, a wrong pin and a plan built against another manifest are refused, and refusing launches nothing;
(e) the arena is reported with and without liveness reuse, and reuse changes no byte of any result.
"""

import json
import pathlib
import warnings

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import plan_model as pm  # noqa: E402
from gpu_util import bits, error, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishManifestError, TilefishPreconditionError, TilefishRuntimeError, runtime  # noqa: E402
from tilefish import _plan  # noqa: E402

BF16 = torch.bfloat16
N_VALUES = {32: (32, 5), 64: (64, 17), 128: (128, 77)}  # all the keys, and some of them


def args_of(m, n, seed=1, weights=None):
    x = pm.activations(m, seed=seed)
    n_t = torch.tensor([n], dtype=torch.int32, device="cuda")
    w = weights if weights is not None else pm.weights()
    return (x, n_t, *[w[k] for k in pm.WEIGHT_NAMES])


@pytest.fixture(scope="module")
def weights():
    return pm.weights()


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("plan")
    return directory, runtime.export(pm.block, directory)


@pytest.fixture(scope="module")
def exported_reuse(tmp_path_factory):
    directory = tmp_path_factory.mktemp("plan-reuse")
    return directory, runtime.export(pm.block, directory, reuse=True)


def replay(p, directory, *args, **kw):
    return tilefish.run(pm.block, *args, mode="replay", plan=directory, pin=p.digest, **kw)


def eager(x, n_t, w_ln1, w_qkv, b_qkv, w_o, w_ln2, w_up, w_down):
    """The block as eight ordinary calls, one per op, each on tensors of its own (what the plan is held to)."""
    m = x.shape[0]
    h = tilefish.run(pm.rmsnorm, x, w_ln1)
    qkv = tilefish.run(pm.gemm_qkv, h, w_qkv, b_qkv)
    fused = qkv.view(m, 3, pm.HEADS, pm.D_HEAD)
    a = tilefish.run(pm.attention, fused[:, 0], fused[:, 1], fused[:, 2], n_t)
    o = tilefish.run(pm.gemm_o, a.view(m, pm.D_MODEL), w_o)
    r, h2 = tilefish.run(pm.add_rmsnorm, x, o, w_ln2)
    u = tilefish.run(pm.gemm_up, h2, w_up)
    g = tilefish.run(pm.gelu, u)
    d = tilefish.run(pm.gemm_down, g, w_down)
    return tilefish.run(pm.add, r, d)


# -- the manifest and the plan that export wrote --------------------------------------------------------------------------------------------------------------


def test_export_writes_a_schema_3_manifest_and_a_plan_that_pins_it(exported):
    directory, p = exported
    manifest = json.loads((directory / "manifest.json").read_text())
    plan_doc = json.loads((directory / "plan.json").read_text())
    assert manifest["schema"] == 3 and len(manifest["library"]) == 4 * 3 and {e["stem"] for e in manifest["kernels"]} == {"rmsnorm", "attention", "add_rmsnorm", "gelu", "add"}
    assert plan_doc["schema"] == "tilefish.plan.v1" and plan_doc["manifest_digest"] == tilefish.Manifest.load(directory).digest == p.manifest_digest
    assert plan_doc["plan_digest"] == p.digest == _plan.plan_digest(plan_doc)
    major, minor = torch.cuda.get_device_capability()
    assert plan_doc["arch"] == f"sm_{major * 10 + minor}"
    assert [b["values"] for b in plan_doc["buckets"]] == [{"M": 32}, {"M": 64}, {"M": 128}]


def test_the_plan_digest_is_the_same_for_two_exports_of_the_same_graph(exported, tmp_path):
    _, p = exported
    again = runtime.export(pm.block, tmp_path)
    assert again.manifest_digest == p.manifest_digest and again.digest == p.digest  # the binaries, the pins and the plan are all deterministic


# -- (a) replay == jit == per-op eager ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("m", pm.BUCKETS)
def test_plan_replay_plan_jit_and_per_op_eager_are_byte_identical(exported, weights, m):
    directory, p = exported
    outputs = []
    for n in N_VALUES[m]:
        args = args_of(m, n, seed=m + n, weights=weights)
        jit = tilefish.run(pm.block, *args)
        rep = replay(p, directory, *args)
        ref = eager(*args)
        assert jit.shape == (m, pm.D_MODEL) and jit.dtype == BF16
        assert same_bytes(jit, rep), (m, n, "jit vs replay")
        assert same_bytes(rep, ref), (m, n, "replay vs per-op eager")
        assert torch.isfinite(rep.float()).all()
        outputs.append(rep)
    assert not same_bytes(outputs[0], outputs[1])  # the device scalar changes the result: n is read from memory by the one binary


def test_replay_takes_the_plan_export_returned_or_its_directory_and_runs_a_directory_it_is_told_to_trust(exported, weights):
    directory, p = exported
    args = args_of(32, 32, seed=41, weights=weights)
    expected = replay(p, directory, *args)
    assert same_bytes(tilefish.run(pm.block, *args, mode="replay", plan=p, pin=p.digest), expected)  # the Plan export returned knows its directory
    assert same_bytes(tilefish.run(pm.block, *args, mode="replay", plan=directory / "plan.json", pin=p.digest), expected)
    assert same_bytes(tilefish.run(pm.block, *args, mode="replay", plan=directory, trust_directory=True), expected)
    with pytest.raises(TilefishRuntimeError, match="was only made, not exported"):
        tilefish.run(pm.block, *args, mode="replay", plan=tilefish.plan(pm.block), pin=p.digest)
    with pytest.raises(TilefishRuntimeError, match="replay runs the plan of an exported directory"):
        tilefish.run(pm.block, *args, mode="replay", pin=p.digest)
    with pytest.raises(TilefishRuntimeError, match="for mode='replay'"):
        tilefish.run(pm.block, *args, pin=p.digest)  # a pin vouches for a directory: jit has none
    with pytest.raises(TilefishRuntimeError, match="manifest= is for a kernel or an extern"):
        tilefish.run(pm.block, *args, mode="replay", manifest=directory, pin=p.digest)


def test_jit_runs_the_plan_it_is_given_with_its_buckets_and_refuses_a_size_it_has_none_for(weights):
    only64 = tilefish.plan(pm.block, buckets={"M": [64]})
    args = args_of(64, 64, seed=42, weights=weights)
    assert same_bytes(tilefish.run(pm.block, *args, plan=only64), eager(*args))
    with pytest.raises(TilefishPreconditionError, match=r"not one of the buckets \[64\] of the plan") as e:
        tilefish.run(pm.block, *args_of(32, 32, weights=weights), plan=only64)
    assert e.value.kind == "bucket"
    with pytest.raises(TilefishRuntimeError, match="plan of graph 'accumulate', not of 'block'"):
        tilefish.run(pm.block, *args, plan=tilefish.plan(pm.accumulate))


def test_an_in_place_graph_updates_the_callers_tensor_in_the_order_of_the_ir_in_every_mode(tmp_path):
    from gpu_util import randn

    p = runtime.export(pm.accumulate, tmp_path)
    reused = tilefish.plan(pm.accumulate, reuse=True)
    for q in (16, 32):
        a = randn((q, pm.W.value), torch.float32, seed=q)
        state0 = randn((q, pm.W.value), torch.float32, seed=q + 1)
        final_state = state0 + a + a  # two in-place updates of the parameter
        expected = (a * 2 + a) + final_state  # an in-place update of a result, then a read of the newest version of both
        for how in ("jit", "reused", "replay", "reference"):
            state = state0.clone()
            if how == "jit":
                out = tilefish.run(pm.accumulate, a, state)
            elif how == "reused":
                out = tilefish.run(pm.accumulate, a, state, plan=reused)
            elif how == "replay":
                out = tilefish.run(pm.accumulate, a, state, mode="replay", plan=tmp_path, pin=p.digest)
            else:
                out = tilefish.run(pm.accumulate, a, state, mode="reference")
            assert same_bytes(out, expected), (q, how)
            assert same_bytes(state, final_state), (q, how)  # the caller's tensor was updated in place
        with pytest.raises(TilefishPreconditionError, match="overlaps 'a'"):
            tilefish.run(pm.accumulate, a, a)  # the in-place tensor must not also be an input


PTX = pathlib.Path(__file__).resolve().parent / "ptx"
B = tilefish.Dim("B", buckets=[8, 24])
Nn = tilefish.Dim("Nn")


@tilefish.extern(ptx=PTX / "axpy_f32.ptx", symbol="axpy_f32", block=(128,), grid=lambda Nn: (tilefish.cdiv(Nn, 128),))
def axpy(x: tilefish.Tensor[Nn, tilefish.f32], y: tilefish.Tensor[Nn, tilefish.f32], out: tilefish.Out[tilefish.Tensor[Nn, tilefish.f32]], n_elem: Nn, alpha: tilefish.const[tilefish.f32] = 2.0): ...


@tilefish.extern(ptx=PTX / "strided_copy_bf16.ptx", symbol="strided_copy_bf16", block=(64,), grid=lambda B: (B,), index_bits=64)
def strided_copy(
    x: tilefish.Strided[tilefish.Tensor[B, 64, tilefish.bf16]], y: tilefish.Out[tilefish.Tensor[B, 64, tilefish.bf16]], *, cols: tilefish.const[tilefish.i32] = 64, x_s0: tilefish.StrideOf("x", 0)
): ...


@tilefish.graph
def prebuilt(
    x: tilefish.Tensor[B, 64, tilefish.f32], y: tilefish.Tensor[B, 64, tilefish.f32], w: tilefish.Tensor[B, 128, tilefish.bf16]
) -> tuple[tilefish.Tensor[B * 64, tilefish.f32], tilefish.Tensor[B, 64, tilefish.bf16]]:
    z = axpy(x.reshape(B * 64), y.reshape(B * 64))  # a prebuilt PTX with a fixed float argument, on views
    left, _ = w.split(64, axis=-1)  # a column block of w: a view with a row stride of 128
    return z, strided_copy(left)  # a prebuilt PTX with a 64-bit stride scalar


def test_a_graph_of_prebuilt_binaries_plans_exports_and_replays_like_one_of_triton_kernels(tmp_path):
    from gpu_util import randn

    p = runtime.export(prebuilt, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert {e["kind"] for e in manifest["kernels"]} == {"binary"} and {e["stem"] for e in manifest["kernels"]} == {"axpy", "strided_copy"}
    assert [s["entry"] for s in p.steps(B=24)] == ["axpy", "strided_copy"]
    for b in (8, 24):
        x, y = randn((b, 64), torch.float32, seed=b), randn((b, 64), torch.float32, seed=b + 1)
        w = randn((b, 128), BF16, seed=b + 2)
        jit = tilefish.run(prebuilt, x, y, w)
        rep = tilefish.run(prebuilt, x, y, w, mode="replay", plan=tmp_path, pin=p.digest)
        z, c = tilefish.run(axpy, x.view(-1), y.view(-1)), tilefish.run(strided_copy, w[:, :64])
        assert same_bytes(jit[0], rep[0]) and same_bytes(jit[1], rep[1])
        assert same_bytes(rep[0], z) and same_bytes(rep[1], c)
        assert same_bytes(rep[0], 2.0 * x.view(-1) + y.view(-1)) and same_bytes(rep[1], w[:, :64])


def test_reference_mode_takes_a_device_scalar_as_an_int_and_a_missing_reference_is_named(weights):
    m = 32
    x = pm.activations(m, seed=9)
    w = [weights[k] for k in pm.WEIGHT_NAMES]
    from_int = tilefish.run(pm.block, x, 7, *w, mode="reference")
    from_tensor = tilefish.run(pm.block, x, torch.tensor([7], dtype=torch.int32, device="cuda"), *w, mode="reference")
    assert same_bytes(from_int, from_tensor)

    @tilefish.extern(triton="gpu_kernels_p2:attention_kernel", num_warps=4, grid=lambda S: (S, pm.HEADS), name="attention_without_a_reference")
    def no_reference(
        q: tilefish.Strided[tilefish.Tensor[pm.S, pm.HEADS, pm.D_HEAD, tilefish.bf16], 16], k: tilefish.Strided[tilefish.Tensor[pm.S, pm.HEADS, pm.D_HEAD, tilefish.bf16], 16],
        v: tilefish.Strided[tilefish.Tensor[pm.S, pm.HEADS, pm.D_HEAD, tilefish.bf16], 16], o: tilefish.Out[tilefish.Tensor[pm.S, pm.HEADS, pm.D_HEAD, tilefish.bf16]],
        n_valid: tilefish.DevScalar[tilefish.i32], q_s0: tilefish.StrideOf("q", 0, divisible_by=8), q_s1: tilefish.StrideOf("q", 1, divisible_by=8),
        k_s0: tilefish.StrideOf("k", 0, divisible_by=8), k_s1: tilefish.StrideOf("k", 1, divisible_by=8), v_s0: tilefish.StrideOf("v", 0, divisible_by=8),
        v_s1: tilefish.StrideOf("v", 1, divisible_by=8), rows: pm.S, DH: tilefish.const = pm.D_HEAD, BLOCK: tilefish.const = 128,
    ): ...

    @tilefish.graph
    def with_it(
        x: tilefish.Tensor[pm.M, pm.D, tilefish.bf16], n: tilefish.DevScalar[tilefish.i32], w_qkv: tilefish.Tensor[3 * pm.D_MODEL, pm.D_MODEL, tilefish.bf16], b_qkv: tilefish.Tensor[3 * pm.D_MODEL, tilefish.bf16]
    ):
        fused = pm.gemm_qkv(x, w_qkv, b_qkv).reshape(pm.M, 3, pm.HEADS, pm.D_HEAD)
        return no_reference(fused[:, 0], fused[:, 1], fused[:, 2], n)

    with pytest.raises(TilefishRuntimeError, match="needs a reference for the extern op 'attention_without_a_reference'"):
        tilefish.run(with_it, x, 7, w[1], w[2], mode="reference")


def test_results_of_every_kind_land_in_the_right_buffers_and_come_back_in_the_order_the_graph_returns_them(tmp_path):
    from gpu_util import randn

    p = runtime.export(pm.two_outputs, tmp_path)
    for q in (16, 32):
        a = randn((q, pm.W.value), torch.float32, seed=q)
        expected = (a * 2.0 + a * 3.0, a + 1.0)
        for how in ("jit", "replay", "reference"):
            kw = dict(mode="replay", plan=tmp_path, pin=p.digest) if how == "replay" else dict(mode=how)
            first, second = tilefish.run(pm.two_outputs, a, **kw)
            assert same_bytes(first, expected[0]) and same_bytes(second, expected[1]), (q, how)
        given = (torch.empty_like(a), torch.empty_like(a))
        out = tilefish.run(pm.two_outputs, a, outputs=given)
        assert out[0] is given[0] and out[1] is given[1] and same_bytes(given[0], expected[0]) and same_bytes(given[1], expected[1])


def test_a_deep_forward_is_one_plan_whose_reuse_changes_no_byte_and_that_a_cuda_graph_replays(tmp_path):
    """Four unrolled layers, 36 steps, the weights of every layer a view of a stacked parameter: replay == jit == the layers composed per op, with and without reuse, and captured."""
    layers = 4
    deep = pm.deep_block(layers)
    per_layer, stacked = pm.stacked_weights(layers)
    exported = runtime.export(deep, tmp_path / "plain")
    exported_reuse = runtime.export(deep, tmp_path / "reuse", reuse=True)
    assert exported_reuse.arena_bytes < exported.arena_bytes and len(exported.steps(M=64)) == 9 * layers
    print(f"\n[arena] {layers} layers: without reuse {exported.arena_bytes} bytes, with reuse {exported_reuse.arena_bytes} bytes")
    names = ["w_ln1", "w_qkv", "b_qkv", "w_o", "w_ln2", "w_up", "w_down"]
    for m in pm.BUCKETS:
        x = pm.activations(m, seed=60 + m)
        n_t = torch.tensor([m - 3], dtype=torch.int32, device="cuda")
        expected = x
        for layer in per_layer:  # the same block, one layer at a time, each op on tensors of its own
            expected = eager(expected, n_t, *[layer[k] for k in names])
        args = (x, n_t, *[stacked[k] for k in names])
        jit = tilefish.run(deep, *args)
        plain = tilefish.run(deep, *args, mode="replay", plan=tmp_path / "plain", pin=exported.digest)
        shared = tilefish.run(deep, *args, mode="replay", plan=tmp_path / "reuse", pin=exported_reuse.digest)
        assert same_bytes(jit, expected) and same_bytes(plain, expected) and same_bytes(shared, expected), m
    m = 64
    x = pm.activations(m, seed=7)
    n_t = torch.tensor([20], dtype=torch.int32, device="cuda")
    args = (x, n_t, *[stacked[k] for k in names])
    out = torch.empty((m, pm.D_MODEL), dtype=BF16, device="cuda")
    runtime.prepare(deep, mode="replay", plan=tmp_path / "reuse", pin=exported_reuse.digest)
    graph = capture(lambda: tilefish.run(deep, *args, mode="replay", plan=tmp_path / "reuse", pin=exported_reuse.digest, outputs=out))
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert same_bytes(out, tilefish.run(deep, *args))
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    print(f"[graph] 4 layers, 36 launches: {start.elapsed_time(stop) / 20 * 1000:.0f} us per replay of the CUDA graph")


def test_n_is_a_device_scalar_the_one_binary_reads_so_the_same_arena_serves_every_value(exported, weights):
    directory, p = exported
    m = 64
    x = pm.activations(m, seed=3)
    results = {}
    for n in (1, 2, 17, 64):
        args = (x, n, *[weights[k] for k in pm.WEIGHT_NAMES])  # an int: written into the slot of the arena before the launches
        results[n] = replay(p, directory, *args).clone()
        n_t = torch.tensor([n], dtype=torch.int32, device="cuda")
        assert same_bytes(results[n], replay(p, directory, x, n_t, *[weights[k] for k in pm.WEIGHT_NAMES]))  # and from a tensor on the device
    assert len({bits(results[n]).cpu().numpy().tobytes() for n in results}) == 4  # four values of n, four different results


def test_two_runs_back_to_back_on_one_stream_do_not_disturb_each_other_through_the_shared_arena(exported, weights):
    directory, p = exported
    a1, a2 = args_of(64, 64, seed=11, weights=weights), args_of(64, 9, seed=12, weights=weights)
    expected1, expected2 = eager(*a1), eager(*a2)
    out1, out2 = torch.empty_like(expected1), torch.empty_like(expected2)
    replay(p, directory, *a1, outputs=out1)
    replay(p, directory, *a2, outputs=out2)  # enqueued behind the first: no synchronization in between
    assert same_bytes(out1, expected1) and same_bytes(out2, expected2)


def test_replay_launches_the_binaries_of_the_manifest_the_same_cubins_jit_builds(exported):
    """The identity of what runs: every kernel entry of the manifest has the machine code digest that a jit build of the same spec has."""
    directory, p = exported
    manifest = tilefish.Manifest.load(directory)
    for stem, callee in p.callees.items():
        if callee.kind != "kernel":
            continue
        built = runtime.jit_built(callee.target, callee.by_stem[stem], 0)
        assert manifest.get(stem, p.arch).sha256_code == built.sha256_code, stem


# -- (b) replay vs reference -----------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("m", pm.BUCKETS)
def test_replay_is_within_a_bf16_step_of_the_reference_and_the_error_is_reported(exported, weights, m, record_property):
    directory, p = exported
    for n in N_VALUES[m]:
        args = args_of(m, n, seed=100 + m + n, weights=weights)
        rep = replay(p, directory, *args)
        ref = tilefish.run(pm.block, *args, mode="reference")
        assert ref.dtype == BF16 and ref.shape == rep.shape
        ulp, absolute, differing, count = error(rep, ref)
        scale = float(ref.float().abs().max())
        tag = f"M={m} n={n}"
        print(f"\n[reference] {tag}: max |abs error| {absolute:.5f} (output scale {scale:.3f}, one bf16 step at the scale {scale * 2**-8:.5f}), max ulp {ulp}, "
              f"{differing}/{count} elements differ")
        record_property(f"error {tag}", f"abs {absolute:.5f} ulp {ulp} differing {differing}/{count}")
        # a result is rounded to bf16 after every op, so one flipped rounding anywhere moves what follows by a step: the bound is a couple of steps of the scale, on few elements
        assert absolute <= 4 * 2**-8 * scale, (tag, absolute, scale)
        assert differing <= 0.2 * count, (tag, differing)


# -- (c) CUDA graphs --------------------------------------------------------------------------------------------------------------------------------------------


def capture(fn):
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "the CUDA graph is empty" when a capture is refused before its first launch
        with torch.cuda.graph(graph):
            fn()
    return graph


@pytest.mark.parametrize("m", pm.BUCKETS)
def test_a_replay_under_cuda_graph_capture_allocates_nothing_and_the_graph_replays_byte_identically(exported, weights, m):
    directory, p = exported
    x = pm.activations(m, seed=5)
    n_t = torch.tensor([m // 2], dtype=torch.int32, device="cuda")
    args = (x, n_t, *[weights[k] for k in pm.WEIGHT_NAMES])
    out = torch.empty((m, pm.D_MODEL), dtype=BF16, device="cuda")
    runtime.prepare(pm.block, mode="replay", plan=directory, pin=p.digest)  # the arena, the binaries, the handle: nothing is made while capturing
    allocated = []

    def run_once():
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        replay(p, directory, *args, outputs=out)
        allocated.append(torch.cuda.memory_stats()["allocation.all.allocated"] - before)

    graph = capture(run_once)
    assert allocated == [0]  # nothing was allocated by the plan, by cuBLASLt or by the launches
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    expected = eager(*args)
    assert same_bytes(out, expected)
    # a graph holds addresses, not values: write the inputs and n in place and replay again
    x.copy_(pm.activations(m, seed=6))
    n_t.fill_(3)
    expected = eager(*args)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert same_bytes(out, expected)
    graph.replay()  # and again: the arena holds nothing from the last replay that the next needs
    torch.cuda.synchronize()
    assert same_bytes(out, expected)


def test_a_jit_plan_captures_the_same_way(weights):
    m = 32
    args = args_of(m, 20, seed=8, weights=weights)
    out = torch.empty((m, pm.D_MODEL), dtype=BF16, device="cuda")
    tilefish.run(pm.block, *args, outputs=out)  # once outside the capture: builds, loads, allocates
    graph = capture(lambda: tilefish.run(pm.block, *args, outputs=out))
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert same_bytes(out, eager(*args))


def test_what_a_capture_cannot_do_is_refused_with_the_reason(exported, weights):
    directory, p = exported
    x = pm.activations(64, seed=5)
    out = torch.empty((64, pm.D_MODEL), dtype=BF16, device="cuda")
    w = [weights[k] for k in pm.WEIGHT_NAMES]
    from tilefish import _planrun

    _planrun.clear_caches()  # a fresh runner: nothing is prepared
    x32, out32 = pm.activations(32), torch.empty((32, pm.D_MODEL), dtype=BF16, device="cuda")
    n_t = torch.tensor([5], dtype=torch.int32, device="cuda")
    # a plan that was never run or prepared: its binaries would be loaded, and its arena made, while capturing
    with pytest.raises(TilefishRuntimeError, match="has not been prepared for {'M': 64}"):
        capture(lambda: replay(p, directory, x, n_t, *w, outputs=out))
    # a bucket that was never run or prepared, next to one that was
    runtime.prepare(pm.block, mode="replay", plan=directory, pin=p.digest, buckets=[{"M": 64}])
    with pytest.raises(TilefishRuntimeError, match="has not been prepared for {'M': 32}"):
        capture(lambda: replay(p, directory, x32, n_t, *w, outputs=out32))
    # a device scalar written from the host cannot be captured
    with pytest.raises(TilefishRuntimeError, match="host value"):
        capture(lambda: replay(p, directory, x, 5, *w, outputs=out))
    # and everything still works afterwards, outside a capture, and then inside one
    assert replay(p, directory, x, 5, *w, outputs=out) is out
    graph = capture(lambda: replay(p, directory, x, n_t, *w, outputs=out))
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert same_bytes(out, eager(x, n_t, *w))
    _planrun.clear_caches()


# -- (d) tampering --------------------------------------------------------------------------------------------------------------------------------------------


def copied(directory, tmp_path, name="copy"):
    target = tmp_path / name
    target.mkdir()
    for f in directory.iterdir():
        (target / f.name).write_bytes(f.read_bytes())
    return target


def test_a_wrong_pin_a_tampered_plan_and_a_plan_of_another_manifest_are_refused_and_launch_nothing(exported, weights, tmp_path):
    directory, p = exported
    m = 64
    args = args_of(m, m, seed=21, weights=weights)
    sentinel = torch.full((m, pm.D_MODEL), 1.0, dtype=BF16, device="cuda")
    kept = sentinel.clone()

    def refused(d, fragment, **kw):
        with pytest.raises(TilefishManifestError, match=fragment):
            tilefish.run(pm.block, *args, mode="replay", plan=d, outputs=sentinel, **kw)
        torch.cuda.synchronize()
        assert same_bytes(sentinel, kept)  # nothing was written: not by a step, not by the write of the device scalar

    refused(directory, "but the pin is", pin="1" * 64)
    refused(directory, "but the pin is", pin=p.manifest_digest)  # the digest of the manifest is not the plan's
    with pytest.raises(TilefishRuntimeError, match="needs a trust anchor"):
        tilefish.run(pm.block, *args, mode="replay", plan=directory, outputs=sentinel)
    with pytest.raises(TilefishRuntimeError, match="not both"):
        tilefish.run(pm.block, *args, mode="replay", plan=directory, outputs=sentinel, pin=p.digest, trust_directory=True)
    # a byte of plan.json changed
    d = copied(directory, tmp_path)
    text = (d / "plan.json").read_text()
    (d / "plan.json").write_text(text.replace('"offset": 256', '"offset": 512', 1))
    refused(d, "but the pin is", pin=p.digest)
    refused(d, "was edited: its plan_digest says", trust_directory=True)
    # an edit that rewrites the digest too: refused by the pin, and by the manifest/graph checks when only the directory is trusted
    d = copied(directory, tmp_path, "rewritten")
    doc = json.loads((d / "plan.json").read_text())
    doc["manifest_digest"] = "0" * 64
    doc["plan_digest"] = _plan.plan_digest(doc)
    (d / "plan.json").write_text(json.dumps(doc))
    refused(d, "but the pin is", pin=p.digest)
    refused(d, "built against the manifest with digest", pin=doc["plan_digest"])
    refused(d, "built against the manifest with digest", trust_directory=True)
    # the manifest changed under the plan
    d = copied(directory, tmp_path, "manifest-changed")
    manifest = json.loads((d / "manifest.json").read_text())
    manifest["kernels"][0]["shared"] += 16
    (d / "manifest.json").write_text(json.dumps(manifest))
    refused(d, "built against the manifest with digest", pin=p.digest)
    # a cubin changed under both digests
    d = copied(directory, tmp_path, "cubin-changed")
    cubin = next(d.glob("rmsnorm.*.cubin"))
    data = bytearray(cubin.read_bytes())
    data[len(data) // 2] ^= 0xFF
    cubin.write_bytes(bytes(data))
    refused(d, "sha256|machine code|valid cubin", pin=p.digest)


def test_a_precondition_that_fails_on_the_device_launches_nothing(exported, weights):
    directory, p = exported
    m = 64
    args = list(args_of(m, m, seed=22, weights=weights))
    sentinel = torch.full((m, pm.D_MODEL), 1.0, dtype=BF16, device="cuda")
    kept = sentinel.clone()
    runtime.prepare(pm.block, mode="replay", plan=directory, pin=p.digest)
    # a weight that does not start on 16 bytes
    big = torch.empty(pm.D_FF * pm.D_MODEL + 8, dtype=BF16, device="cuda")
    shifted = big[1 : 1 + pm.D_FF * pm.D_MODEL].view(pm.D_FF, pm.D_MODEL)
    shifted.copy_(weights["w_up"])
    bad = list(args)
    bad[2 + pm.WEIGHT_NAMES.index("w_up")] = shifted
    with pytest.raises(TilefishPreconditionError, match="not aligned to 16 bytes") as e:
        replay(p, directory, *bad, outputs=sentinel)
    assert e.value.kind == "pointer_alignment"
    torch.cuda.synchronize()
    assert same_bytes(sentinel, kept)
    # a number of rows that is not a bucket
    with pytest.raises(TilefishPreconditionError, match="not one of the buckets") as e:
        replay(p, directory, *args_of(48, 48, weights=weights))
    assert e.value.kind == "bucket"
    # an output that overlaps an input
    with pytest.raises(TilefishPreconditionError, match="overlaps"):
        replay(p, directory, *args, outputs=args[0])


def test_a_changed_graph_makes_the_exported_plan_stale(exported):
    directory, p = exported

    @tilefish.graph(name="block")
    def block(x: tilefish.Tensor[pm.M, pm.D, tilefish.bf16], w: tilefish.Tensor[pm.D, tilefish.bf16]) -> tilefish.Tensor[pm.M, pm.D, tilefish.bf16]:
        return pm.rmsnorm(x, w)

    x = pm.activations(32)
    with pytest.raises(TilefishManifestError, match="is not the plan of graph 'block' as this process holds it"):
        tilefish.run(block, x, pm.weights()["w_ln1"], mode="replay", plan=directory, pin=p.digest)


# -- (e) the arena ------------------------------------------------------------------------------------------------------------------------------------------------


def test_the_arena_is_reported_with_and_without_reuse_and_reuse_changes_no_result(exported, exported_reuse, weights, record_property):
    directory, p = exported
    directory_r, pr = exported_reuse
    assert pr.reuse is True and p.reuse is False
    assert pr.arena_bytes < p.arena_bytes
    workspace = next(b for b in p.buffers if b["role"] == "workspace")["bytes"]
    print(
        f"\n[arena] without reuse {p.arena_bytes} bytes ({p.arena_bytes / 2**20:.2f} MiB), with reuse {pr.arena_bytes} bytes ({pr.arena_bytes / 2**20:.2f} MiB); "
        f"the cuBLASLt workspace is {workspace} bytes of it; intermediates {p.arena_bytes - workspace - 256} bytes without reuse, {pr.arena_bytes - workspace - 256} with"
    )
    record_property("arena_bytes_without_reuse", p.arena_bytes)
    record_property("arena_bytes_with_reuse", pr.arena_bytes)
    assert (p.arena_bytes, pr.arena_bytes) == (p.arena_bytes_without_reuse, pr.arena_bytes_with_reuse)
    for m in pm.BUCKETS:
        for n in N_VALUES[m]:
            args = args_of(m, n, seed=m * 7 + n, weights=weights)
            plain = replay(p, directory, *args)
            shared = replay(pr, directory_r, *args)
            jit_shared = tilefish.run(pm.block, *args, plan=tilefish.plan(pm.block, reuse=True))
            assert same_bytes(plain, shared) and same_bytes(plain, jit_shared), (m, n)


def test_the_offsets_are_the_same_in_every_bucket_so_one_arena_serves_all(exported, weights):
    directory, p = exported
    runtime.prepare(pm.block, mode="replay", plan=directory, pin=p.digest)
    from tilefish import _planrun

    runner = next(iter(_planrun._loaded.values()))[2]
    arena = runner._arenas[torch.cuda.current_device()]
    assert arena.nbytes == p.arena_bytes and arena.ptr % 256 == 0
    # run every bucket and read an intermediate back from the arena at its one offset: the qkv of every bucket is where the plan says
    off = {b["name"]: b["offset"] for b in p.buffers}
    for m in pm.BUCKETS:
        args = args_of(m, m, seed=31, weights=weights)
        replay(p, directory, *args)
        torch.cuda.synchronize()
        h = tilefish.run(pm.rmsnorm, args[0], args[2])
        qkv = tilefish.run(pm.gemm_qkv, h, args[3], args[4])
        in_arena = arena.handle[off["t1"] : off["t1"] + m * 768 * 2].view(BF16).view(m, 768)
        assert same_bytes(in_arena, qkv), m
