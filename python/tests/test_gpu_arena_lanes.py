"""On a GPU: several arenas of one plan (``share_arena(..., name=...)``), and the replays of a CUDA graph of the caller's ordered by
``tilefish.runtime.replay_graph``.

* **An arena per stream.** Executions on one arena run one after the other on the device, whatever their streams (the arena's order event). A run on
  another arena of the same plan does not wait for them: behind a stream that sleeps 600 ms before a run on arena ``lane0``, a run on ``lane1`` on
  another stream is done at once, and a run on ``lane0`` itself waits for the sleep. Runs on two arenas from two threads give their own bytes.
* **A graph of the caller's.** ``tilefish.run`` inside ``torch.cuda.graph`` captures a graph whose ``replay()`` takes no lock of tilefish, so a run of the
  same plan on another thread can overlap it (``docs/en/plan-schema-1.md``, section 10, item 7: 3 of 4 000 runs wrong on a B300). The second thread that
  uses such an arena gets a ``TilefishOrderWarning``; replays through ``tilefish.runtime.replay_graph(g)`` take the arena's lock, and not one of the
  runs racing them is wrong.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import threading
import time
import warnings

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, TilefishOrderWarning, f32, runtime  # noqa: E402

ROWS = Dim("LROWS", buckets=[2048])
COLS = Dim("LCOLS", 4096)
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


@tilefish.graph
def three_x(x: Tensor[ROWS, COLS, f32]) -> Tensor[ROWS, COLS, f32]:
    """``3x`` through an intermediate (``2x``) that the arena holds until the last of 42 steps: an overlap of two runs on one arena breaks it."""
    t = twice(x)
    w = x
    for _ in range(STEPS):
        w = negate(w)
    return plus(t, w)


def expected(x):
    return x * 3.0


@pytest.fixture(scope="module")
def inputs():
    x1, x2 = randn((2048, 4096), seed=31), randn((2048, 4096), seed=32)
    assert same_bytes(tilefish.run(three_x, x1), expected(x1))  # prepared: built, loaded
    return x1, x2


@pytest.fixture(scope="module")
def cycles_per_ms():
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(10_000_000)
    stop.record()
    torch.cuda.synchronize()
    return int(10_000_000 / start.elapsed_time(stop))


def behind_a_sleep(first, second, x, cycles_per_ms, sleep_ms=600):
    """``first.run`` on a stream that sleeps ``sleep_ms`` first, then ``second.run`` on another stream: how long the host waits for the second
    stream (seconds), and both results once everything is done."""
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(s1):
        torch.cuda._sleep(int(cycles_per_ms * sleep_ms))
        y1 = first.run(three_x, x)
    with torch.cuda.stream(s2):
        y2 = second.run(three_x, x)
    t0 = time.perf_counter()
    s2.synchronize()
    waited = time.perf_counter() - t0
    torch.cuda.synchronize()
    return waited, y1, y2


def test_a_run_on_another_arena_does_not_wait_for_the_runs_of_the_first(inputs, cycles_per_ms):
    x, _ = inputs
    lane0, lane1 = runtime.share_arena([three_x], name="lane0"), runtime.share_arena([three_x], name="lane1")
    for lane in (lane0, lane1):
        assert same_bytes(lane.run(three_x, x), expected(x))  # each arena allocated, outside the timing
    assert lane0.arena_of(0).ptr != lane1.arena_of(0).ptr and lane0.arena_of(0).event is not lane1.arena_of(0).event
    apart, y1, y2 = behind_a_sleep(lane0, lane1, x, cycles_per_ms)
    assert same_bytes(y1, expected(x)) and same_bytes(y2, expected(x))
    together, y1, y2 = behind_a_sleep(lane0, runtime.share_arena([three_x], name="lane0"), x, cycles_per_ms)  # the same name: the same arena
    assert same_bytes(y1, expected(x)) and same_bytes(y2, expected(x))
    print(f"\n[lanes] behind a 600 ms sleep on another stream: a run on another arena waited {apart * 1e3:.1f} ms, on the same arena {together * 1e3:.1f} ms")
    assert apart < 0.2 < together


def test_runs_on_two_arenas_from_two_threads_give_their_own_bytes(inputs):
    x1, x2 = inputs
    lanes = [runtime.share_arena([three_x], name=f"thread{i}") for i in range(2)]
    results = {0: [], 1: []}
    errors = []
    barrier = threading.Barrier(2)

    def work(i, x):
        try:
            stream = torch.cuda.Stream()
            barrier.wait()
            with torch.cuda.stream(stream):
                for _ in range(8):
                    results[i].append(lanes[i].run(three_x, x))
            stream.synchronize()
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(0, x1)), threading.Thread(target=work, args=(1, x2))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize()
    assert errors == []
    want = (expected(x1), expected(x2))
    assert [(i, r) for i in (0, 1) for r, y in enumerate(results[i]) if not same_bytes(y, want[i])] == []


def caller_graph(lane, x, out):
    lane.run(three_x, x)  # prepared outside the capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        lane.run(three_x, x, outputs=out)
    return g


def test_a_second_thread_on_an_arena_a_caller_graph_holds_gets_a_warning(inputs):
    x1, x2 = inputs
    lane = runtime.share_arena([three_x], name="warned")
    out = torch.empty_like(x1)
    caller_graph(lane, x1.clone(), out)
    caught = []

    def other():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lane.run(three_x, x2)
        caught.extend(w)

    t = threading.Thread(target=other)
    t.start()
    t.join()
    torch.cuda.synchronize()
    order = [w for w in caught if issubclass(w.category, TilefishOrderWarning)]
    assert len(order) == 1 and "replay_graph" in str(order[0].message)


RUNS = 3000


def test_replays_of_a_caller_graph_through_replay_graph_never_overlap_the_runs_of_another_thread(inputs):
    x1, x2 = inputs
    lane = runtime.share_arena([three_x], name="ordered")
    x_cap = x1.clone()
    out = torch.empty_like(x_cap)
    g = caller_graph(lane, x_cap, out)
    want_cap, want2 = expected(x_cap), expected(x2)
    stop = threading.Event()
    counts = {"runs": 0, "runs_wrong": 0, "replays": 0, "replays_wrong": 0}
    errors = []

    def runs():
        try:
            s1 = torch.cuda.Stream()
            with torch.cuda.stream(s1):
                for _ in range(RUNS):
                    y = lane.run(three_x, x2)
                    s1.synchronize()
                    counts["runs"] += 1
                    counts["runs_wrong"] += not same_bytes(y, want2)
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)
        finally:
            stop.set()

    def replays():
        try:
            s2 = torch.cuda.Stream()
            while not stop.is_set():
                runtime.replay_graph(g, stream=s2)
                s2.synchronize()
                counts["replays"] += 1
                counts["replays_wrong"] += not same_bytes(out, want_cap)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    replay_thread = threading.Thread(target=replays)
    replay_thread.start()
    while counts["replays"] == 0 and not errors:  # the replays (through replay_graph) start first
        time.sleep(0.001)
    run_thread = threading.Thread(target=runs)
    run_thread.start()
    for t in (run_thread, replay_thread):
        t.join()
    torch.cuda.synchronize()
    print(f"\n[replay_graph] {counts['runs']} runs ({counts['runs_wrong']} wrong) and {counts['replays']} replays ({counts['replays_wrong']} wrong)")
    assert errors == [] and counts["runs"] == RUNS and counts["replays"] > 0
    assert counts["runs_wrong"] == 0 and counts["replays_wrong"] == 0
    arena = lane.arena_of(0)
    assert arena.caller_graphs and arena.ordered and not arena.warned  # the caller orders its replays: no warning was given
