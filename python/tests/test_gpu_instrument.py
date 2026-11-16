"""The instrumented variant on a GPU (``docs/instrument.md``): the kernels of ``probe_kernels.py``, a warp-specialised Gluon producer and consumer, a
tcgen05 MMA pipeline (sm_100 and sm_103), a persistent Triton kernel and an elementwise ``@tilefish.kernel``, built with their probes.

Held here:

* a production build of a kernel with probes has none (no field, no instruction: the machine code of the kernel without them), and the instrumented
  export has binaries and a digest of its own, which a host refuses unless asked;
* the outputs of an instrumented launch are the production bytes (jit and replay), and a null probe buffer records nothing and changes no byte;
* the records say what the kernels do: the partitions of a warp-specialised kernel are told apart, the producer and the consumer of a pipeline run at
  one rate, the epilogue of a matmul waits as long as its main loop runs, each program of a persistent grid walks its own tiles on an SM of its own;
* a plan records every instrumented step (jit, replay of an instrumented export, a captured replay), and ``tilefish-explain --probes`` reads a real
  trace with the lines of the probes;
* what the probes cost: each kernel captured in a CUDA graph with and without probes, replayed and timed; the numbers are printed (``-s``) and bounded.

Ran on sm_103 (a B300 MIG slice); written to run on sm_120 too (the tcgen05 matmul is skipped there).
"""

import json
import math
import statistics

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("triton.experimental.gluon")
pytest.importorskip("cuda.bindings.driver")

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:  # pragma: no cover - TMA needs sm_90
    pytest.skip("the warp-specialised TMA kernels need sm_90 or newer", allow_module_level=True)

import probe_kernels as pk  # noqa: E402
import ws_kernels as wk  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Manifest, TilefishManifestError, explain, runtime  # noqa: E402
from tilefish._probe import BEGIN, CALIB  # noqa: E402
from tilefish.aot import build, compile_specs  # noqa: E402

F16 = torch.float16
CAP = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
ARCH = f"sm_{CAP[0] * 10 + CAP[1]}"
needs_tcgen05 = pytest.mark.skipif(not pk.tcgen05(CAP), reason="tcgen05 (sm_100, sm_103) only")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("instrument")
    ops = pk.ops(CAP)
    prod = runtime.export(ops, d / "prod")
    inst = runtime.export(ops, d / "inst", instrument=tilefish.instrument(capacity=1024), trace=True)
    return d, prod, inst


def probed(op, *args, capacity=1024, **kw):
    """``op`` run with probes: its output and the trace of its launch."""
    rec = tilefish.Probes(tilefish.instrument(capacity=capacity))
    out = tilefish.run(op, *args, probes=rec, **kw)
    (launch,) = rec.trace().launches
    return out, launch


def names(launch, w):
    by_id = launch.names()
    return [by_id[r[0]] for r in w.records]


def body(launch, w):
    """The names of a warp's records between its ``@begin`` (and the two ``@calib`` of warp 0 of each CTA, which calibrates) and its ``@end``."""
    got = names(launch, w)
    assert got[0] == "@begin" and got[-1] == "@end", got[:4]
    k = 1
    while got[k] == "@calib":
        k += 1
    assert k == (3 if w.warp == 0 else 1), (w.warp, got[:4])
    return got[k:-1]


def median_of(launch, phase, role=None):
    values = sorted(p.ns for p in launch.phases() if p.name == phase and (role is None or p.role == role))
    assert values, f"no phase {phase!r} ({role})"
    return statistics.median(values)


# -- the variant -------------------------------------------------------------------------------------------------------------------------------


def test_a_production_build_of_a_kernel_with_probes_has_none(exported):
    _, prod, _ = exported
    assert not prod.instrumented and all("instrument" not in e for e in prod.doc["kernels"])
    (spec,) = compile_specs(pk.ws_scale_p)
    with_probes = build(spec, arch=ARCH, fn=pk.ws_scale_kernel, cache=False)
    assert "tilefish.probe" not in with_probes.ptx and "%tfp_" not in with_probes.ptx and with_probes.instrument is None
    (plain_spec,) = compile_specs(wk.ws_scale)
    plain = build(plain_spec, arch=ARCH, fn=wk.ws_scale_kernel, cache=False)
    assert with_probes.sha256_code == plain.sha256_code  # the probes compiled to nothing: the machine code of the kernel written without them


def test_the_instrumented_export_is_a_variant_the_hosts_refuse_unless_asked(exported):
    d, prod, inst = exported
    assert inst.instrumented and inst.digest != prod.digest
    entries = {e["op"]: e for e in inst.doc["kernels"]}
    assert [p["name"] for p in entries["ws_scale_p"]["instrument"]["probes"]] == ["compute.wait", "compute.work", "load.wait", "load.issue"]
    roles = {p["name"]: p["sites"][0]["role"] for p in entries["ws_scale_p"]["instrument"]["probes"]}
    assert roles == {"compute.wait": "default", "compute.work": "default", "load.wait": "worker 0", "load.issue": "worker 0"}
    for e in inst.doc["kernels"]:
        p = next(x for x in prod.doc["kernels"] if x["stem"] == e["stem"])
        assert e["sha256_code"] != p["sha256_code"] and e["declaration_digest"] == p["declaration_digest"] and e["abi"] == p["abi"]
    with pytest.raises(TilefishManifestError, match="is an instrumented export"):
        Manifest.load(d / "inst")
    x = randn((128, 512), F16, seed=1)
    with pytest.raises(TilefishManifestError, match="is an instrumented export"):
        tilefish.run(pk.ws_scale_p, x, mode="replay", manifest=d / "inst", pin=inst.digest)
    with pytest.raises(TilefishManifestError, match="is a production one"):
        tilefish.run(pk.ws_scale_p, x, mode="replay", manifest=d / "prod", pin=prod.digest, probes=tilefish.Probes())


def inputs():
    """Arguments of every op of the tests (seeded)."""
    out = {
        pk.ws_scale_p: (randn((256, 4096), F16, seed=2),),
        pk.persistent_scale_p: (randn(1 << 23, seed=3),),
        pk.scale_add_p: (randn(1 << 22, seed=4), randn(1 << 22, seed=5)),
    }
    if pk.tcgen05(CAP):
        out[pk.ws_matmul_p] = (randn((512, 8192), F16, seed=6, scale=0.5), randn((512, 8192), F16, seed=7, scale=0.5))
    return out


def test_the_outputs_of_an_instrumented_launch_are_the_production_bytes(exported):
    d, prod, inst = exported
    for op, args in inputs().items():
        want = tilefish.run(op, *args, mode="replay", manifest=d / "prod", pin=prod.digest)
        assert same_bytes(tilefish.run(op, *args), want), op.name
        rec = tilefish.Probes()
        got = tilefish.run(op, *args, mode="replay", manifest=d / "inst", pin=inst.digest, probes=rec)
        assert same_bytes(got, want), op.name
        jit, _ = probed(op, *args)
        assert same_bytes(jit, want), op.name
        assert rec.launches and rec.launches[0].manifest_digest == inst.digest


def test_a_null_probe_buffer_records_nothing_and_changes_no_byte():
    x = randn((256, 4096), F16, seed=8)
    want = tilefish.run(pk.ws_scale_p, x)
    target = runtime._target(pk.ws_scale_p)
    prep = runtime._prepare(target, (x,), {}, None)
    index = torch.cuda.current_device()
    built = runtime.jit_built(target, prep.spec, index, tilefish.instrument())
    assert built.instrument is not None
    runtime._launch_built(prep, built, index)  # no probes=: the profile scratch is null, and every probe skips its store
    torch.cuda.synchronize()
    assert same_bytes(prep.results[0], want)


# -- what the records say ---------------------------------------------------------------------------------------------------------------------


def test_a_producer_and_a_consumer_partition():
    x = randn((256, 4096), F16, seed=9)
    y, launch = probed(pk.ws_scale_p, x)
    assert same_bytes(y, wk.scaled(x))
    tiles = 4096 // 64
    assert launch.grid == (4, 1, 1) and launch.block == (256, 1, 1) and not launch.missing and len(launch.warps) == 4 * 8
    for cta in range(4):
        warps = [w for w in launch.warps if w.cta == cta]
        roles = {w.warp: launch.role_of(w) for w in warps}
        assert sorted(k for k, r in roles.items() if r == "default") == [0, 1, 2, 3]
        assert len([k for k, r in roles.items() if r == "worker 0"]) == 1 and sum(r == "-" for r in roles.values()) == 3
        for w in warps:
            got = body(launch, w)
            assert w.dropped == 0
            if roles[w.warp] == "default":
                assert got == ["compute.wait", "compute.work"] * tiles
            elif roles[w.warp] == "worker 0":
                assert got == ["load.wait", "load.issue"] * tiles
            else:
                assert got == []
            assert len({r[1] for r in w.records}) == 1  # one SM for the warp's lifetime
            assert all(b[2] >= a[2] for a, b in zip(w.records, w.records[1:]))  # globaltimer never goes back
    # a pipeline in its steady state: the producer and the consumer go round at one rate, and the one that waits is the one that is faster
    produce = median_of(launch, "load.wait -> load.issue", "worker 0") + median_of(launch, "load.issue -> load.wait", "worker 0")
    consume = median_of(launch, "compute.wait -> compute.work", "default") + median_of(launch, "compute.work -> compute.wait", "default")
    assert produce == pytest.approx(consume, rel=0.15)
    s = launch.summary()
    print(f"\nws_scale_p: producer {produce:.0f} ns per tile, consumer {consume:.0f} ns per tile, a probe costs {s['probe_cycles']} cycles; {s}")
    assert s["idle_warps"] == 12 and s["probe_cycles"] and s["clock_ghz"]


@needs_tcgen05
def test_a_tcgen05_mma_pipeline():
    a, b = inputs()[pk.ws_matmul_p]
    c, launch = probed(pk.ws_matmul_p, a, b)
    want = (a.double() @ b.double().T).to(F16)
    assert torch.allclose(c.float(), want.float(), rtol=2e-2, atol=2e-2)
    steps = 8192 // 64
    assert launch.grid == (4, 4, 1) and not launch.missing and all(w.dropped == 0 for w in launch.warps)
    for cta in range(16):
        warps = {launch.role_of(w): w for w in launch.warps if w.cta == cta and launch.role_of(w) != "-"}
        assert set(warps) == {"default", "worker 0", "worker 1"}
        assert body(launch, warps["worker 0"]) == ["load.wait", "load.issue"] * steps
        assert body(launch, warps["worker 1"]) == ["mma.wait", "mma.issue"] * steps + ["mma.committed"]
        epi = [w for w in launch.warps if w.cta == cta and launch.role_of(w) == "default"]
        assert len(epi) == 4 and all(body(launch, w) == ["epi.wait", "epi.store", "epi.done"] for w in epi)
        # the epilogue waits for the accumulator as long as the main loop runs: from the issuer's first wait for a tile to its commit, and a little
        # more (the last MMAs complete after the commit is issued)
        issuer = [r for r in warps["worker 1"].records if r[0] not in (BEGIN, CALIB)]
        loop = issuer[-2][2] - issuer[0][2]
        wait = next(p for p in launch.phases() if p.cta == cta and p.warp == epi[0].warp and p.name == "epi.wait -> epi.store")
        assert 0.9 * loop <= wait.timer_ns <= 1.3 * loop + 2000, (cta, loop, wait.timer_ns)
    # the producer and the issuer go round at one rate; the one that waits longer is the faster one
    produce = median_of(launch, "load.wait -> load.issue", "worker 0") + median_of(launch, "load.issue -> load.wait", "worker 0")
    issue = median_of(launch, "mma.wait -> mma.issue", "worker 1") + median_of(launch, "mma.issue -> mma.wait", "worker 1")
    assert produce == pytest.approx(issue, rel=0.25)
    print(
        f"\nws_matmul_p: per K step producer {produce:.0f} ns (waits {median_of(launch, 'load.wait -> load.issue', 'worker 0'):.0f}), issuer "
        f"{issue:.0f} ns (waits {median_of(launch, 'mma.wait -> mma.issue', 'worker 1'):.0f}); epilogue wait {median_of(launch, 'epi.wait -> epi.store', 'default'):.0f} ns"
    )


def test_a_persistent_kernel():
    (x,) = inputs()[pk.persistent_scale_p]
    y, launch = probed(pk.persistent_scale_p, x)
    assert same_bytes(y, x * 2 + 0.25)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    programs = launch.grid[0]
    tiles = math.ceil(x.numel() / 1024)
    assert programs == min(tiles, sms)  # one program per SM (tilefish.persistent())
    by_cta = {}
    for w in launch.warps:
        assert w.dropped == 0
        mine = len(range(w.cta, tiles, programs))
        assert body(launch, w) == ["tile", "stored"] * mine
        by_cta.setdefault(w.cta, set()).add(w.records[0][1])
    assert len({next(iter(v)) for v in by_cta.values()}) == programs  # each program on an SM of its own
    starts = [min(w.records[0][2] for w in launch.warps if w.cta == c) for c in by_cta]
    ends = [max(w.records[-1][2] for w in launch.warps if w.cta == c) for c in by_cta]
    assert max(starts) < min(ends)  # the programs run at once (a persistent grid is one wave)
    print(f"\npersistent_scale_p: {programs} programs, tile {median_of(launch, 'tile -> stored'):.0f} ns, span {launch.span_ns()} ns")


def test_an_elementwise_kernel_of_the_dsl():
    x, y = inputs()[pk.scale_add_p]
    out, launch = probed(pk.scale_add_p, x, y)
    assert same_bytes(out, x * 2 + y)
    ctas = (1 << 22) // 1024
    assert launch.grid == (ctas, 1, 1) and len(launch.warps) == ctas * 4
    for w in launch.warps[:64]:
        assert body(launch, w) == ["scaled", "added"]
    s = launch.summary()
    assert {r["phase"] for r in s["phases"]} == {"@begin -> scaled", "scaled -> added", "added -> @end"}
    per_sm = {}
    for c in launch.ctas():
        per_sm.setdefault(c["sm"], []).append(c)
    assert len(per_sm) == torch.cuda.get_device_properties(0).multi_processor_count  # every SM ran CTAs, in waves
    sites = launch.sites()
    assert sites["scaled"][0]["loc"].startswith("tilefish_generated/scale_add_p.py:") and sites["scaled"][0]["role"] == ""
    print(f"\nscale_add_p: {s['ctas']} CTAs on {s['sms']} SMs, span {s['span_ns']} ns, phases {[(r['phase'], round(r['ns']['median'])) for r in s['phases']]}")


def test_a_plan_records_every_instrumented_step(tmp_path):
    x = randn((256, 1024), F16, seed=10)
    want = tilefish.run(pk.probe_chain, x)
    rec = tilefish.Probes()
    got = tilefish.run(pk.probe_chain, x, probes=rec)
    assert same_bytes(got, want)
    assert [(launch.step, launch.op, launch.graph) for launch in rec.launches] == [(0, "bump_p", "probe_chain"), (1, "ws_scale_p", "probe_chain")]
    assert rec.launches[0].bucket == {"Rb": 256} and all(not launch.missing for launch in rec.launches)
    plan = runtime.export(pk.probe_chain, tmp_path / "inst", instrument=True)
    prod = runtime.export(pk.probe_chain, tmp_path / "prod")
    assert plan.digest != prod.digest
    with pytest.raises(TilefishManifestError, match="instrumented export"):
        tilefish.run(pk.probe_chain, x, mode="replay", plan=tmp_path / "inst", pin=plan.digest)
    rec = tilefish.Probes()
    assert same_bytes(tilefish.run(pk.probe_chain, x, mode="replay", plan=tmp_path / "inst", pin=plan.digest, probes=rec), want)
    assert [launch.stem.split(".")[0] for launch in rec.launches] == ["bump_p", "ws_scale_p"] and rec.launches[1].manifest_digest
    captured = tilefish.capture(pk.probe_chain, x, mode="replay", plan=tmp_path / "inst", pin=plan.digest, probes=rec)
    rec.clear()
    captured.replay()
    captured.collect()
    assert len(rec.launches) == 2 and same_bytes(captured.outputs, want)
    t0 = [launch.t0 for launch in rec.launches]
    assert t0[0] < t0[1]  # the second step after the first
    rec.clear()
    captured.replay()  # every replay writes its records again
    captured.collect()
    assert len(rec.launches) == 2 and rec.launches[0].t0 > t0[1]


def test_explain_reads_a_real_trace(exported, tmp_path, capsys):
    d, _, inst = exported
    x = randn((256, 4096), F16, seed=11)
    rec = tilefish.Probes()
    tilefish.run(pk.ws_scale_p, x, mode="replay", manifest=d / "inst", pin=inst.digest, probes=rec)
    path = rec.trace().save(tmp_path / "ws.json")
    assert explain.main([str(d / "inst"), "--probes", str(path), "--chrome", str(tmp_path / "ws.chrome.json")]) == 0
    out = capsys.readouterr().out
    assert "load.wait -> load.issue (worker 0)" in out and 'tilefish.probe("load.wait")' in out and "a probe costs a warp" in out
    events = json.loads((tmp_path / "ws.chrome.json").read_text())["traceEvents"]
    assert any(e.get("name") == "compute.wait -> compute.work" for e in events)


# -- what the probes cost -------------------------------------------------------------------------------------------------------------------


def replay_us(captured, reps=50):
    for _ in range(5):
        captured.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(7):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            captured.replay()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / reps)
    return statistics.median(times)


def graphs():
    out = [
        (pk.ws_scale_g, (randn((1024, 4096), F16, seed=12),)),
        (pk.persistent_g, (randn(1 << 23, seed=13),)),
        (pk.scale_g, (randn(1 << 22, seed=14), randn(1 << 22, seed=15))),
    ]
    if pk.tcgen05(CAP):
        out.append((pk.mm_g, (randn((512, 8192), F16, seed=16, scale=0.5), randn((512, 8192), F16, seed=17, scale=0.5))))
    return out


def test_what_the_probes_cost():
    report = []
    for graph, args in graphs():
        plain = tilefish.capture(graph, *args)
        rec = tilefish.Probes(tilefish.instrument(capacity=1024))
        inst = tilefish.capture(graph, *args, probes=rec)
        t_plain, t_inst = replay_us(plain), replay_us(inst)
        inst.collect()
        launch = rec.launches[-1]
        s = launch.summary()
        per_warp = max(len(w.records) for w in launch.warps)
        report.append(
            f"{graph.name}: {t_plain:.1f} us -> {t_inst:.1f} us ({100 * (t_inst / t_plain - 1):+.1f}%), up to {per_warp} records per warp, a probe "
            f"costs a warp {s['probe_cycles']} cycles, the probes' share of a warp's time {100 * (s['overhead'] or 0):.1f}%"
        )
        assert same_bytes(inst.outputs, plain.outputs)
        assert t_inst < 2.0 * t_plain + 10.0, report[-1]
    print("\n" + "\n".join(report))
