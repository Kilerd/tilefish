"""On a GPU: the plans of a set run one after the other on one shared arena (``tilefish.runtime.share_arena``, ``docs/plan-schema-1.md`` section 15.1).

* The three stages of ``plan_model`` (the transformer block cut into ``stage_attn``, ``stage_mix`` and ``stage_ffn``), exported as a plan set and replayed
  through its pin on one arena of the largest stage's bytes (9 109 504, where three arenas take 26 476 800): every bucket and two values of the device
  scalar give the bytes of the same stages on separate arenas, and of the block in one plan; captured (``SharedArena.capture``) and replayed, with new
  inputs and device scalars between replays; in one ``torch.cuda.graph`` on one stream; and a tensor of the caller inside the arena is refused.
* The order: two plans whose intermediates share bytes (a long chain ``three_x`` whose ``t`` is read by its last step, and a short ``tripled`` whose
  first intermediate lies over ``t``) run at once on two streams, and from two threads with replays of a captured graph. With the arena's order event
  every result is right; with the order taken away (the negative control) the same rounds come out wrong.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import threading
import warnings

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import plan_model as pm  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, TilefishPreconditionError, TilefishRuntimeError, _planrun, f32, runtime  # noqa: E402

BF16 = torch.bfloat16


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("stages")
    return directory, runtime.export(pm.STAGES, directory)


@pytest.fixture(scope="module")
def weights():
    return pm.weights(seed=7)


def stage_weights(w, stage):
    return [w[k] for k in pm.STAGE_WEIGHTS[stage.name]]


def separate(directory, pin, w, x, n):
    """The three stages, each replayed on its own arena (``tilefish.run`` of the set's plans)."""
    opts = dict(mode="replay", plan=directory, pin=pin)
    a = tilefish.run(pm.stage_attn, x, n, *stage_weights(w, pm.stage_attn), **opts)
    r, h2 = tilefish.run(pm.stage_mix, a, x, *stage_weights(w, pm.stage_mix), **opts)
    return a, r, h2, tilefish.run(pm.stage_ffn, r, h2, *stage_weights(w, pm.stage_ffn), **opts)


def on_shared(s, w, x, n):
    a = s.run(pm.stage_attn, x, n, *stage_weights(w, pm.stage_attn))
    r, h2 = s.run(pm.stage_mix, a, x, *stage_weights(w, pm.stage_mix))
    return a, r, h2, s.run(pm.stage_ffn, r, h2, *stage_weights(w, pm.stage_ffn))


def test_the_stages_on_one_shared_arena_give_the_bytes_of_separate_arenas_and_of_the_block(exported, weights):
    directory, planset = exported
    s = runtime.share_arena(pm.STAGES, mode="replay", plan=directory, pin=planset.digest)
    s.prepare()
    index = torch.cuda.current_device()
    assert s.arena_of(index).nbytes == s.arena_bytes == 9_109_504 and sum(s.sizes.values()) == 26_476_800
    cases = 0
    for m in pm.BUCKETS:
        x = pm.activations(m, seed=m)
        for n in (m, m // 2 + 1):
            nt = torch.tensor([n], dtype=torch.int32, device="cuda")
            want = separate(directory, planset.digest, weights, x, nt)
            got = on_shared(s, weights, x, nt)
            block = tilefish.run(pm.block, x, nt, *[weights[k] for k in pm.WEIGHT_NAMES])
            torch.cuda.synchronize()
            for name, g, e in zip(("a", "r", "h2", "y"), got, want):
                assert same_bytes(g, e), (m, n, name)
            assert same_bytes(got[3], block), (m, n)
            cases += 1
    print(f"\n[shared arena] {cases} cases of the three stages on one arena of {s.arena_bytes} bytes: the bytes of three arenas ({sum(s.sizes.values())} bytes) and of the block")


def test_the_stages_captured_on_the_shared_arena_replay_the_bytes_of_separate_arenas(exported, weights):
    """``SharedArena.capture`` of each stage, the next stage's inputs the previous stage's outputs (tensors of the caller's): the graphs, replayed in a
    row, give the bytes of separate arenas, with new inputs and a new device scalar between replays."""
    directory, planset = exported
    s = runtime.share_arena(pm.STAGES, mode="replay", plan=directory, pin=planset.digest)
    s.prepare()
    m = 64
    x = pm.activations(m, seed=11).clone()
    a, r, h2, y = (torch.empty((m, 256), dtype=BF16, device="cuda") for _ in range(4))
    caps = [
        s.capture(pm.stage_attn, x, *stage_weights(weights, pm.stage_attn), devscalars={"n": 40}, outputs=a),
        s.capture(pm.stage_mix, a, x, *stage_weights(weights, pm.stage_mix), outputs={"r": r, "h2": h2}),
        s.capture(pm.stage_ffn, r, h2, *stage_weights(weights, pm.stage_ffn), outputs=y),
    ]
    for round_, (seed, n) in enumerate([(11, 40), (12, 64), (13, 9)]):
        x.copy_(pm.activations(m, seed=seed))
        caps[0].set_devscalar("n", n)
        for t in (a, r, h2, y):
            t.zero_()
        for cap in caps:
            cap.replay()
        torch.cuda.synchronize()
        want = separate(directory, planset.digest, weights, x, torch.tensor([n], dtype=torch.int32, device="cuda"))
        for name, g, e in zip(("a", "r", "h2", "y"), (a, r, h2, y), want):
            assert same_bytes(g, e), (round_, name)


def test_the_stages_in_one_torch_graph_on_one_stream_replay_and_a_forked_stream_is_refused(exported, weights):
    directory, planset = exported
    s = runtime.share_arena(pm.STAGES, mode="replay", plan=directory, pin=planset.digest)
    s.prepare()
    m = 32
    x = pm.activations(m, seed=21)
    nt = torch.tensor([m], dtype=torch.int32, device="cuda")
    a, r, h2, y = (torch.empty((m, 256), dtype=BF16, device="cuda") for _ in range(4))
    graph = torch.cuda.CUDAGraph()
    refused = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.cuda.graph(graph):
            s.run(pm.stage_attn, x, nt, *stage_weights(weights, pm.stage_attn), outputs=a)
            s.run(pm.stage_mix, a, x, *stage_weights(weights, pm.stage_mix), outputs=(r, h2))
            s.run(pm.stage_ffn, r, h2, *stage_weights(weights, pm.stage_ffn), outputs=y)
            main = torch.cuda.current_stream()
            forked = torch.cuda.Stream()
            forked.wait_stream(main)
            with torch.cuda.stream(forked):
                try:
                    s.run(pm.stage_mix, a, x, *stage_weights(weights, pm.stage_mix), outputs=(r, h2))
                except TilefishRuntimeError as e:
                    refused.append(str(e))
            main.wait_stream(forked)
    assert len(refused) == 1 and "a plan of the shared arena of graph 'stage_mix' was already run in this CUDA graph capture on another stream" in refused[0]
    y.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert same_bytes(y, separate(directory, planset.digest, weights, x, nt)[3])


def test_a_tensor_of_the_caller_inside_the_shared_arena_is_refused_and_nothing_is_written(exported, weights):
    """An output of ``stage_mix`` past the end of its own arena but inside the shared one (bytes of the ffn's intermediates) would be written over by the
    next run of the ffn: refused, and nothing is launched."""
    directory, planset = exported
    s = runtime.share_arena(pm.STAGES, mode="replay", plan=directory, pin=planset.digest)
    s.prepare()
    arena = s.arena_of(torch.cuda.current_device())
    m = 64
    start = s.sizes["stage_mix"] + 256
    inside = arena.handle[start : start + m * 256 * 2].view(BF16).view(m, 256)
    x = pm.activations(m, seed=5)
    a = s.run(pm.stage_attn, x, torch.tensor([m], dtype=torch.int32, device="cuda"), *stage_weights(weights, pm.stage_attn))
    h2 = torch.full((m, 256), 7.0, dtype=BF16, device="cuda")
    torch.cuda.synchronize()
    with pytest.raises(TilefishPreconditionError, match="'r' overlaps the arena of the plan.*shared with other plans"):
        s.run(pm.stage_mix, a, x, *stage_weights(weights, pm.stage_mix), outputs={"r": inside, "h2": h2})
    torch.cuda.synchronize()
    assert bool((h2 == 7.0).all())


# -- the order: two plans whose intermediates share bytes, on two streams and two threads --------------------------------------------------------------

ROWS = Dim("ROWS", buckets=[2048])
COLS = Dim("COLS", 4096)
STEPS = 40


@tilefish.kernel
def twice(x: Tensor[ROWS, COLS, f32]) -> Tensor[ROWS, COLS, f32]:
    return x * 2.0


@tilefish.kernel
def negate(x: Tensor[ROWS, COLS, f32]) -> Tensor[ROWS, COLS, f32]:
    return x * -1.0


@tilefish.kernel
def plus(x: Tensor[ROWS, COLS, f32], y: Tensor[ROWS, COLS, f32]) -> Tensor[ROWS, COLS, f32]:
    return x + y


@tilefish.graph(inputs=("x",))
def three_x(x: Tensor[ROWS, COLS, f32]) -> Tensor[ROWS, COLS, f32]:
    """``2x`` into ``t`` (the arena's first bytes), 40 negations of ``x``, then ``t + x``: ``t`` is read by the last step, long after it was written."""
    t = twice(x)
    w = x
    for _ in range(STEPS):
        w = negate(w)
    return plus(t, w)


@tilefish.graph(inputs=("z",))
def tripled(z: Tensor[ROWS, COLS, f32]) -> Tensor[ROWS, COLS, f32]:
    """``2z + z``: two launches, the first writing the arena's first bytes (the ``t`` of ``three_x`` in a shared arena)."""
    return plus(twice(z), z)


@pytest.fixture(scope="module")
def pair():
    s = runtime.share_arena([tilefish.plan(three_x, reuse=True), tilefish.plan(tripled)])
    s.prepare()
    offsets = {n: {b["name"]: b["offset"] for b in p.buffers if b["offset"] is not None} for n, p in s.plans.items()}
    assert offsets["three_x"]["t0"] == offsets["tripled"]["t0"] == 0  # three_x's t and tripled's first result: the same bytes
    x1, z1 = randn((2048, 4096), seed=1), randn((2048, 4096), seed=2)
    return s, x1, z1


def rounds_on_two_streams(s, x1, z1, rounds, spin):
    """``three_x`` on one stream and ``tripled`` on another, every round enqueued back to back with no synchronization until the last: the number of
    rounds whose results are wrong. With ``spin``, both streams start with a spin of the same length, behind which the host enqueues every round, so when
    the spins end the two streams' work is released together and overlaps however fast the GPU is (a large GPU finishes one run before the host has
    enqueued the next, and without the spin would never overlap them; a small one overlaps them without it)."""
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    for stream in (s1, s2) if spin else ():
        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)  # tens of milliseconds of one thread spinning: far longer than enqueueing every round
    results = []
    for _ in range(rounds):
        with torch.cuda.stream(s1):
            y1 = s.run(three_x, x1)
        with torch.cuda.stream(s2):
            y2 = s.run(tripled, z1)
        results.append((y1, y2))
    torch.cuda.synchronize()
    return sum(int(not same_bytes(y1, x1 * 3.0) or not same_bytes(y2, z1 * 3.0)) for y1, y2 in results)


def test_two_plans_of_a_shared_arena_on_two_streams_run_one_after_the_other(pair, monkeypatch):
    s, x1, z1 = pair
    rounds = 20

    def both():  # the rounds enqueued behind a spin and without one: each provokes the race on some GPU
        return rounds_on_two_streams(s, x1, z1, rounds, spin=True) + rounds_on_two_streams(s, x1, z1, rounds, spin=False)

    ordered = both()

    def unordered(be, arena, stream, capturing, work):  # the negative control: no wait for the order event, no record
        work()

    monkeypatch.setattr(_planrun, "_ordered", unordered)
    raced = both()
    monkeypatch.undo()
    print(f"\n[shared arena] two plans on two streams, 2 x {rounds} rounds: {ordered} wrong with the arena's order event, {raced} wrong without it")
    assert ordered == 0
    assert raced > 0, "the negative control saw no race: the test would not see one either"


def test_replays_of_one_plan_and_runs_of_another_from_two_threads_are_ordered(pair):
    """A thread replays a ``CapturedPlan`` of ``three_x`` (on the shared arena) while another runs ``tripled``, each on a stream of its own, 12 times: every
    result is right, so the replay of plan A is ordered with the runs of plan B."""
    s, x1, z1 = pair
    out = torch.empty_like(x1)
    cap = s.capture(three_x, x1, outputs=out)
    rounds = 12
    got_a, got_b, errors = [], [], []
    start = threading.Barrier(2)

    def replayer():
        stream = torch.cuda.Stream()
        try:
            start.wait()
            with torch.cuda.stream(stream):
                for _ in range(rounds):
                    cap.replay()
                    got_a.append(out.clone())
            stream.synchronize()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def runner():
        stream = torch.cuda.Stream()
        try:
            start.wait()
            with torch.cuda.stream(stream):
                for _ in range(rounds):
                    got_b.append(s.run(tripled, z1))
            stream.synchronize()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=replayer), threading.Thread(target=runner)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize()
    assert errors == []
    wrong = sum(not same_bytes(g, x1 * 3.0) for g in got_a) + sum(not same_bytes(g, z1 * 3.0) for g in got_b)
    print(f"\n[shared arena] {rounds} replays of three_x and {rounds} runs of tripled from two threads: {wrong} wrong")
    assert wrong == 0
