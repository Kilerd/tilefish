"""The per-step profiler (:func:`tilefish.profile_plan`, ``tilefish-explain --profile``) on a device, and the advisor with what it measured: the plan's
steps are timed, the export and the bytes a run writes are not changed, and the times are sane (positive, their sum close to the plan's own time).

Arch-agnostic: it ran on sm_103 (a B300 MIG slice); the sm_120 and sm_90 runs are a later gate's. No ``from __future__ import annotations``.
"""

import hashlib
import json
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import plan_model as pm  # noqa: E402
from gpu_util import same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishRuntimeError, explain, runtime  # noqa: E402

pytestmark = pytest.mark.gpu


def args_of(m, n, seed=1, weights=None):
    x = pm.activations(m, seed=seed)
    n_t = torch.tensor([n], dtype=torch.int32, device="cuda")
    w = weights if weights is not None else pm.weights()
    return (x, n_t, *[w[k] for k in pm.WEIGHT_NAMES])


def files_of(directory):
    root = pathlib.Path(directory)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("advise-plan")
    return directory, runtime.export(pm.block, directory)


def test_every_step_is_timed_and_no_byte_of_the_export_or_of_a_run_changes(exported, tmp_path):
    directory, p = exported
    before = files_of(directory)
    args = args_of(128, 77)
    y0 = tilefish.run(pm.block, *args, mode="replay", plan=directory, pin=p.digest)
    timing = tilefish.profile_plan(pm.block, directory, pin=p.digest, bucket="M=128", reps=20, warmup=3, devscalars={"n": 77}, out=tmp_path / "t.json")
    assert json.loads((tmp_path / "t.json").read_text()) == timing
    assert timing["schema"] == "tilefish.step-timing.v1" and timing["graph"] == "block" and timing["plan_digest"] == p.digest
    assert timing["bucket"] == {"M": 128} and timing["method"] == "graph" and timing["reps"] == 20
    steps = p.steps(M=128)
    assert [s["entry"] for s in timing["steps"]] == [s.get("entry") for s in steps]
    assert all(0 < s["us"]["min"] <= s["us"]["median"] <= s["us"]["p90"] < 5000 for s in timing["steps"])
    assert timing["device"]["copy_GBps"] > 50 and timing["device"]["sm_count"] >= 1
    # the events between the steps cost little: the steps' sum is close to the plan captured without them
    assert 0.5 < timing["sum_us"] / timing["plan_us"]["trimmed"] < 1.6, timing
    assert sum(s["us"]["net"] for s in timing["steps"]) == pytest.approx(timing["plan_us"]["trimmed"], rel=0.02)
    y1 = tilefish.run(pm.block, *args, mode="replay", plan=directory, pin=p.digest)
    assert same_bytes(y0, y1)
    assert files_of(directory) == before


def test_the_two_methods_measure_the_same_steps(exported):
    directory, p = exported
    g = tilefish.profile_plan(pm.block, directory, pin=p.digest, bucket="M=64", reps=20, devscalars={"n": 17})
    e = tilefish.profile_plan(pm.block, directory, pin=p.digest, bucket="M=64", reps=20, devscalars={"n": 17}, method="events")
    assert e["method"] == "events" and len(e["steps"]) == len(g["steps"])
    assert 0.4 < e["sum_us"] / g["sum_us"] < 2.5, (e["sum_us"], g["sum_us"])
    # the longest step is the same kind of step in both (the work dominates, not the way it is launched)
    longest = lambda t: max(t["steps"], key=lambda s: s["us"]["median"])["entry"].split(".")[0]  # noqa: E731
    assert longest(g) == longest(e) or abs(g["sum_us"] - e["sum_us"]) < 30


def test_jit_needs_every_device_scalar_and_a_bucket():
    with pytest.raises(TilefishRuntimeError, match="devscalars"):
        tilefish.profile_plan(pm.block, bucket="M=32", reps=1)
    with pytest.raises(TilefishRuntimeError, match="say which with bucket="):
        tilefish.profile_plan(pm.block, reps=1, devscalars={"n": 5})
    t = tilefish.profile_plan(pm.block, bucket={"M": 32}, reps=3, warmup=1, devscalars={"n": 5})
    assert t["bucket"] == {"M": 32} and len(t["steps"]) == 9


def test_the_profile_leaves_an_inout_tensor_and_a_carried_buffer_as_it_found_them():
    import gpu_kernels_plans as kp

    # the caller's InOut tensor: every run updates it in place; after the profile it holds its bytes again
    a = torch.randn(16, 128, device="cuda")
    state = torch.randn(16, 128, device="cuda")
    before = state.clone()
    t = tilefish.profile_plan(pm.accumulate, bucket="Q=16", reps=3, warmup=1, externals={"a": a, "state": state})
    assert len(t["steps"]) == 5 and same_bytes(state, before)
    # a carried buffer the plan writes: a run of stage_a leaves h = 3x in its arena; the profile's runs (of other inputs) overwrite it, and it puts
    # back the bytes and what the arena notes of them (read here from the arena itself, at h's region)
    x = torch.randn(8, 64, device="cuda")
    tilefish.run(kp.stage_a, x)
    runner = tilefish._planrun._jit_runner(tilefish._planrun._default_plan(kp.stage_a))
    arena = runner.arena_of(torch.cuda.current_device())
    offset = runner._carried.regions["h"][0]
    h = lambda: arena.handle[offset:offset + 8 * 64 * 4].clone()  # noqa: E731
    held = h()
    tilefish.profile_plan(kp.stage_a, bucket="Rs=8", reps=2, warmup=1, method="events")
    assert torch.equal(h(), held) and arena.shapes.get("h") == (8, 64)


def test_the_advice_with_measured_times_and_the_command_line(exported, tmp_path, capsys):
    directory, p = exported
    out = tmp_path / "timing.json"
    assert explain.main([str(directory), "--graph", "block", "--profile", str(out), "--load", "plan_model:block", "--bucket", "M=128", "--reps", "10",
                         "--devscalar", "n=77", "--pin", p.digest]) == 0
    assert "steps of 'block' at {'M': 128}" in capsys.readouterr().out
    assert explain.main([str(directory), "--graph", "block", "--advise", "--timing", str(out), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["estimate"] == "measured" and doc["bucket"] == {"M": 128} and doc["timing"]["floor_us"] > 0
    assert doc["total_us"] == pytest.approx(sum(s["us"] for s in doc["steps"]), rel=1e-6)
    assert doc["candidates"] and all(c["saved_us"]["M=128"] >= 0 for c in doc["candidates"])
    assert doc["machine"]["bandwidth_GBps"] == json.loads(out.read_text())["device"]["copy_GBps"]
    assert explain.main([str(directory), "--graph", "block", "--advise", "--timing", str(out), "--bucket", "M=32"]) == 2
    assert "not {'M': 32}" in capsys.readouterr().err
