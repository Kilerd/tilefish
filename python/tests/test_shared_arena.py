"""An arena shared by the plans of a set (``tilefish.runtime.share_arena``, ``docs/plan-schema-1.md`` section 15.1), with no GPU: runners on a fake backend.

The three stages of ``plan_model`` (the transformer block cut into ``stage_attn``, ``stage_mix`` and ``stage_ffn``, each stage's result a returned value the
next one reads) run one after the other on one arena: it has the bytes of the largest of their arenas, each plan's buffers at the offsets of its own
document from the one base; every run, capture, replay and ``run_step`` of any of them waits for and records the one order event of the arena under its
one lock; a tensor of the caller anywhere in the arena (not only in the part of the plan that runs) is refused; and the documents, their digests and the
set's are what they were (the pin covers the layout). The real launches are ``test_gpu_shared_arena.py``.
"""

import json

import plan_fakes
import plan_model as pm
import pytest
from fake_torch import FakeTensor
from test_planrun import ARENA, FakeBackend, FakeProvider

import tilefish
from tilefish import TilefishManifestError, TilefishPreconditionError, TilefishRuntimeError, _plan, _planrun, runtime
from tilefish._planrun import PlanRunner, SharedArena

SIZES = {"stage_attn": 8_782_080, "stage_mix": 8_585_216, "stage_ffn": 9_109_504}


class Tensors:
    """Fake tensors of the stages, each at an address of its own (far from the arena and from what the fake backend allocates)."""

    def __init__(self, m, base=1 << 36):
        self.m, self.next = m, base
        self.x = self.make((m, 256))
        self.n = FakeTensor((1,), "i32", self.take())
        shapes = {"w_ln1": (256,), "w_qkv": (768, 256), "b_qkv": (768,), "w_o": (256, 256), "w_ln2": (256,), "w_up": (1024, 256), "w_down": (256, 1024)}
        self.w = {k: self.make(s) for k, s in shapes.items()}

    def take(self):
        ptr = self.next
        self.next += 1 << 24
        return ptr

    def make(self, shape, dtype="bf16", ptr=None):
        return FakeTensor(shape, dtype, self.take() if ptr is None else ptr)

    def weights(self, stage):
        return [self.w[k] for k in pm.STAGE_WEIGHTS[stage.name]]


def shared(backend=None, graphs=None):
    be = backend or FakeBackend()
    plans = [tilefish.plan(g) for g in (graphs or pm.STAGES)]
    return SharedArena([PlanRunner(p, FakeProvider(), be) for p in plans]), be


def offsets(plan):
    return {b["name"]: b["offset"] for b in plan.buffers if b["offset"] is not None}


def pipeline(s, t):
    """The three stages in a row on the shared arena: each stage's returned values are the next one's arguments."""
    a = s.run(pm.stage_attn, t.x, t.n, *t.weights(pm.stage_attn))
    r, h2 = s.run(pm.stage_mix, a, t.x, *t.weights(pm.stage_mix))
    return a, r, h2, s.run(pm.stage_ffn, r, h2, *t.weights(pm.stage_ffn))


def test_the_shared_arena_has_the_bytes_of_the_largest_plan_and_each_plan_its_own_offsets_from_one_base():
    s, be = shared()
    assert s.sizes == SIZES and s.arena_bytes == max(SIZES.values()) == 9_109_504 and sum(SIZES.values()) == 26_476_800
    t = Tensors(64)
    pipeline(s, t)
    assert be.log.count(("arena", s.arena_bytes)) == 1 and be.arenas == 1  # one arena for the three plans, of the largest plan's bytes
    plans = s.plans
    gemms = {e[1]: e[6] for e in be.launches() if e[0] == "gemm"}  # the workspace of each GEMM
    assert gemms == {
        "gemm_qkv.M64": ARENA + offsets(plans["stage_attn"])["workspace"],
        "gemm_o.M64": ARENA + offsets(plans["stage_mix"])["workspace"],
        "gemm_up.M64": ARENA + offsets(plans["stage_ffn"])["workspace"],
        "gemm_down.M64": ARENA + offsets(plans["stage_ffn"])["workspace"],
    }
    assert ("devscalar", offsets(plans["stage_attn"])["n"], "tensor", True) in be.log  # the attention's slot, at its own offset
    gelu = next(e for e in be.launches() if e[1] == "gelu")
    assert [v for k, v in gelu[3] if k == "pointer"] == [ARENA + offsets(plans["stage_ffn"])["t0"], ARENA + offsets(plans["stage_ffn"])["t1"]]
    # the ffn's intermediates lie over the attention's device-scalar slot: the bytes are shared, which only the order of the runs makes sound
    assert offsets(plans["stage_ffn"])["t0"] <= offsets(plans["stage_attn"])["n"] < offsets(plans["stage_ffn"])["t0"] + 262_144


def test_runs_of_different_plans_on_the_shared_arena_wait_for_and_record_one_order_event():
    """A run of stage A on one stream and of stage B on another must not overlap on the device: they share every byte. Each waits for the most recent
    record of the arena's one event before its first write and records it after its last launch, under the arena's one lock, as the runs of one plan do."""
    s, be = shared()
    streams = iter([11, 22, 33, 11, 22, 33])
    be.stream = lambda index: next(streams)
    t = Tensors(32)
    pipeline(s, t)
    pipeline(s, t)
    assert be.log.count(("event", 0)) == 1  # one event, made with the one arena
    waits = [e for e in be.log if e[0] in ("wait", "record")]
    assert waits == [(kind, "event1", stream, False) for stream in (11, 22, 33, 11, 22, 33) for kind in ("wait", "record")]
    locks = {id(r.arena_of(0).lock) for r in (s.runner(g) for g in pm.STAGES)}
    assert len(locks) == 1  # one lock for the three


def test_a_captured_plan_on_the_shared_arena_is_ordered_with_the_runs_of_the_other_plans():
    """``SharedArena.capture``: the CapturedPlan's replay takes the shared arena's lock, waits for and records its event, and writes its device scalar again
    (another plan's run has written over the slot's bytes in between)."""
    s, be = shared()
    t = Tensors(64)
    a, r, h2, _ = pipeline(s, t)  # prepares the buckets and the arena
    cap = s.capture(pm.stage_attn, t.x, *t.weights(pm.stage_attn), devscalars={"n": 5})
    start = len(be.log)
    s.run(pm.stage_ffn, r, h2, *t.weights(pm.stage_ffn))
    cap.replay(stream=44)
    ops = [e for e in be.log[start:] if e[0] in ("wait", "record", "devscalar", "graph")]
    assert ops[0] == ("wait", "event1", 7, False) and ops[1] == ("record", "event1", 7, False)  # the ffn's run
    assert ops[2:] == [
        ("wait", "event1", 44, False),
        ("devscalar", offsets(s.plans["stage_attn"])["n"], 5, False),
        ("graph", "graph1", 44),
        ("record", "event1", 44, False),
    ]
    assert cap._arena is s.arena_of(0)


def test_a_tensor_of_the_caller_anywhere_in_the_shared_arena_is_refused():
    """What would be overwritten is refused: an output of stage_mix in the shared arena past the end of stage_mix's own arena (the ffn's bytes) is still
    in the arena, and the next run of the ffn would write over it. Refused before anything is launched; just past the shared arena it is fine."""
    s, be = shared()
    t = Tensors(64)
    a, *_ = pipeline(s, t)
    mix = s.plans["stage_mix"]
    assert mix.arena_bytes < s.arena_bytes
    launched = len(be.launches())
    inside = t.make((64, 256), ptr=ARENA + mix.arena_bytes + 4096)  # outside the mix's own part, inside the shared arena
    with pytest.raises(TilefishPreconditionError, match="'r' overlaps the arena of the plan.*shared with other plans") as e:
        s.run(pm.stage_mix, a, t.x, *t.weights(pm.stage_mix), outputs={"r": inside, "h2": t.make((64, 256))})
    assert e.value.kind == "alias" and len(be.launches()) == launched
    with pytest.raises(TilefishPreconditionError, match="overlaps a tensor of the call or the plan's arena"):  # a tap
        s.run(pm.stage_mix, a, t.x, *t.weights(pm.stage_mix), taps={"t0": inside})
    with pytest.raises(TilefishPreconditionError, match="overlaps the arena of the plan"):  # one step alone
        s.run_step(pm.stage_mix, 1, {"x": t.x, "t0": t.make((64, 256)), "w_ln2": t.w["w_ln2"], "r": inside, "h2": t.make((64, 256))})
    assert len(be.launches()) == launched
    past = t.make((64, 256), ptr=ARENA + s.arena_bytes)
    s.run(pm.stage_mix, a, t.x, *t.weights(pm.stage_mix), outputs={"r": past, "h2": t.make((64, 256))})
    assert len(be.launches()) == launched + 2


def test_two_plans_of_the_shared_arena_in_one_capture_on_two_streams_are_refused():
    """In one ``torch.cuda.graph`` capture, a run of stage B on a forked stream would not wait for the run of stage A inside the graph: refused, as two runs
    of one plan are, before anything of it is enqueued."""
    s, be = shared()
    t = Tensors(64)
    a, r, h2, _ = pipeline(s, t)
    be.capture, be.capture_ident = True, 41
    be.current = 11
    s.run(pm.stage_attn, t.x, t.n, *t.weights(pm.stage_attn), outputs=a)
    before = list(be.log)
    be.current = 22
    with pytest.raises(TilefishRuntimeError, match="a plan of the shared arena of graph 'stage_mix' was already run in this CUDA graph capture on another stream"):
        s.run(pm.stage_mix, a, t.x, *t.weights(pm.stage_mix), outputs=(r, h2))
    assert be.log == before
    be.current = 11
    s.run(pm.stage_mix, a, t.x, *t.weights(pm.stage_mix), outputs=(r, h2))  # on the capture's stream: ordered by the graph
    assert [e for e in be.log if e[0] == "record"][-1] == ("record", "event1", 11, True)


def test_one_step_alone_and_weights_bound_once_are_on_the_shared_arena():
    s, be = shared()
    t = Tensors(64)
    a, r, h2, _ = pipeline(s, t)
    start = len(be.log)
    s.run_step(pm.stage_ffn, 0, {"h2": h2, "w_up": t.w["w_up"], "t0": t.make((64, 1024))})
    gemm = [e for e in be.log[start:] if e[0] == "gemm"]
    assert gemm[0][6] == ARENA + offsets(s.plans["stage_ffn"])["workspace"]
    assert [e[:2] for e in be.log[start:] if e[0] in ("wait", "record")] == [("wait", "event1"), ("record", "event1")]
    w = s.bind_weights(pm.stage_mix, {"w_o": t.w["w_o"], "w_ln2": t.w["w_ln2"]})
    w.run(a, t.x)
    assert w._runner is s.runner(pm.stage_mix) and be.arenas == 1


def test_every_shared_arena_of_the_same_plans_is_one_arena_and_a_plan_alone_or_another_set_has_its_own():
    """Arenas are kept per layout, as for one plan: two SharedArenas of the same plans (jit or replay: the layouts are the same) are one arena; a plan run
    alone, and a shared arena of other plans, have arenas of their own (they are not ordered with this one, and share no memory with it)."""
    be = FakeBackend()
    s1, _ = shared(be)
    s2, _ = shared(be)
    t = Tensors(32)
    pipeline(s1, t)
    pipeline(s2, t)
    assert be.arenas == 1 and s1.arena_of(0) is s2.arena_of(0) and s1.key == s2.key
    alone = PlanRunner(tilefish.plan(pm.stage_mix), FakeProvider(), be)
    alone.run((t.x, t.x, *t.weights(pm.stage_mix)), {})
    assert be.arenas == 2 and alone.arena_of(0) is not s1.arena_of(0) and alone.arena_of(0).nbytes == SIZES["stage_mix"]
    pair, _ = shared(be, [pm.stage_mix, pm.stage_ffn])
    assert pair.key != s1.key and pair.arena_bytes == SIZES["stage_ffn"]


def test_what_a_shared_arena_takes_is_checked():
    be = FakeBackend()
    runners = [PlanRunner(tilefish.plan(g), FakeProvider(), be) for g in pm.STAGES]
    with pytest.raises(TilefishRuntimeError, match="the arena of one plan or more"):
        SharedArena([])
    with pytest.raises(TilefishRuntimeError, match="stage_mix is given twice"):
        SharedArena([runners[1], runners[1]])
    with pytest.raises(TilefishRuntimeError, match="run on one backend"):
        SharedArena([runners[0], PlanRunner(runners[1].plan, FakeProvider(), FakeBackend())])
    s = SharedArena(runners)
    with pytest.raises(TilefishRuntimeError, match="is not a graph of this shared arena: its plans are \\['stage_attn', 'stage_mix', 'stage_ffn'\\]"):
        s.run(pm.block, Tensors(32).x)
    with pytest.raises(TilefishRuntimeError, match="not a graph of this shared arena"):
        s.runner("block")
    assert s.runner("stage_mix") is s.runner(pm.stage_mix)
    for bad, message in [
        (pm.stage_attn, "takes a list of graphs"),
        ([], "the arena of one plan or more"),
        ([pm.rmsnorm], "takes @tilefish.graph objects"),
        ([pm.stage_attn, pm.stage_attn], "stage_attn is given twice"),
    ]:
        with pytest.raises(TilefishRuntimeError, match=message):
            runtime.share_arena(bad)
    with pytest.raises(TilefishRuntimeError, match="plan= is the directory of a replay"):
        runtime.share_arena(pm.STAGES, plan=tilefish.plan(pm.stage_attn))
    with pytest.raises(TilefishRuntimeError, match="pin= and trust_directory= are for mode='replay'"):
        runtime.share_arena(pm.STAGES, pin="0" * 64)
    with pytest.raises(TilefishRuntimeError, match="mode is 'jit' or 'replay'"):
        runtime.share_arena(pm.STAGES, mode="reference")
    jit = runtime.share_arena([tilefish.plan(pm.stage_attn), pm.stage_mix])  # jit: a plan made by tilefish.plan, or a graph (its default plan)
    assert jit.names == ["stage_attn", "stage_mix"] and jit.arena_bytes == SIZES["stage_attn"]


@pytest.fixture
def exported_stages(tmp_path):
    directory = tmp_path / "stages"
    return directory, plan_fakes.export_set([tilefish.plan(g) for g in pm.STAGES], directory)


def test_share_arena_in_replay_takes_the_plans_of_a_set_through_its_pin_and_changes_no_document(exported_stages):
    """The layout of a shared arena is the plans' own (the offsets and ``arena_bytes`` of their documents), so the set's pin, which covers every plan's
    digest, covers it: nothing is added to a document, and the digests of the plans and of the set are the ones the export wrote."""
    directory, planset = exported_stages
    index = json.loads((directory / "plans.json").read_text())
    s = runtime.share_arena(pm.STAGES, mode="replay", plan=directory, pin=planset.digest)
    assert s.sizes == SIZES and s.arena_bytes == max(SIZES.values())
    assert {n: p.digest for n, p in s.plans.items()} == index["plans"] and _plan.set_digest(index) == planset.digest == index["set_digest"]
    assert all(p.set_digest == planset.digest for p in s.plans.values())
    jit = runtime.share_arena(pm.STAGES)
    assert jit.key == s.key  # the same layouts: one arena for the jit and the replay of the same plans
    trusted = runtime.share_arena(pm.STAGES[:2], mode="replay", plan=directory, trust_directory=True)
    assert trusted.names == ["stage_attn", "stage_mix"] and trusted.arena_bytes == SIZES["stage_attn"]
    with pytest.raises(TilefishManifestError, match="but the pin is"):
        runtime.share_arena(pm.STAGES, mode="replay", plan=directory, pin="0" * 64)
    with pytest.raises(TilefishManifestError, match="lists no plan of graph 'block'"):
        runtime.share_arena([pm.stage_attn, pm.block], mode="replay", plan=directory, pin=planset.digest)
    # a plan whose layout was edited (a larger arena) is not the plan the set's pin covers: refused, so the arena cannot be sized from it
    doc = json.loads((directory / "stage_mix.plan.json").read_text())
    doc["arena_bytes"] += 256
    (directory / "stage_mix.plan.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="has digest .* but the pin is|was edited"):
        runtime.share_arena(pm.STAGES, mode="replay", plan=directory, pin=planset.digest)


def test_clear_caches_frees_the_shared_arena_after_its_last_run_and_a_capture_on_it_is_not_replayed_after(monkeypatch):
    be = FakeBackend()
    monkeypatch.setattr(_planrun, "_torch_backend", [be])
    s, _ = shared(be)
    t = Tensors(64)
    pipeline(s, t)
    cap = s.capture(pm.stage_attn, t.x, *t.weights(pm.stage_attn), devscalars={"n": 3})
    _planrun.clear_caches()
    assert ("synchronize", "event1", 0) in be.log and ("destroy", "event1", 0) in be.log and s.arena_of(0) is None
    with pytest.raises(TilefishRuntimeError, match="cannot be replayed"):
        cap.replay()
    pipeline(s, t)  # the next run allocates the arena again
    assert be.arenas == 2 and s.arena_of(0).nbytes == s.arena_bytes
