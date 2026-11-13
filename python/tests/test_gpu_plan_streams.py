"""On a GPU: runs of one plan on several streams share its arena, and are ordered on the device by the arena's order event.

The reviewer's repro: a graph that computes ``3x`` (``t = twice(x)``, then 40 steps of ``x * -1.0``, then ``plus(t, w)``) in f32 on ``[2048, 4096]``, run with the default
``jit`` path on a stream ``s1`` for ``x1`` and, behind a ``torch.cuda._sleep`` on a stream ``s2``, for ``x2``. The arena (and ``t`` in it) is the same for both runs, and the
lock of the arena ordered only the enqueueing: in 5 of 5 rounds ``y1`` came out as ``2*x2 + x1``, with no error. Each run now waits on its stream for the most recent
record of the arena's event before its first write and records the event after its last launch (``_planrun._Arena``), and a captured graph does the same through external
event nodes. Held here: 0 of 20 rounds corrupted, two threads with two streams, and replays of a captured graph with runs on other streams in between.

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
from tilefish import Dim, Tensor, f32  # noqa: E402

ROWS = Dim("ROWS", buckets=[2048])
COLS = Dim("COLS", 4096)
STEPS = 40
ROUNDS = 20


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
    t = twice(x)  # an intermediate in the arena, read again only by the last step
    w = x
    for _ in range(STEPS):
        w = negate(w)  # an even number of negations: w is x again
    return plus(t, w)


def expected(x):
    return x * 3.0  # 2x and x are exact, their sum is rounded once: the bytes of x * 3


@pytest.fixture(scope="module")
def inputs():
    return randn((2048, 4096), seed=1), randn((2048, 4096), seed=2)


@pytest.fixture(scope="module")
def cycles_per_ms():
    """How many cycles of ``torch.cuda._sleep`` take a millisecond here (to put the second run in the middle of the first)."""
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(10_000_000)
    stop.record()
    torch.cuda.synchronize()
    return int(10_000_000 / start.elapsed_time(stop))


@pytest.fixture(scope="module")
def run_ms(inputs):
    """How long one run of the plan takes (it is prepared by the first: built, loaded, the arena allocated)."""
    x1, _ = inputs
    assert same_bytes(tilefish.run(three_x, x1), expected(x1))
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    tilefish.run(three_x, x1)
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop)


def test_the_reviewers_two_runs_on_two_streams_are_never_corrupted(inputs, cycles_per_ms, run_ms):
    x1, x2 = inputs
    want1, want2 = expected(x1), expected(x2)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    sleep = max(1, int(cycles_per_ms * run_ms / 2))  # the second run would start in the middle of the first
    corrupted = []
    for i in range(ROUNDS):
        torch.cuda.synchronize()
        with torch.cuda.stream(s1):
            y1 = tilefish.run(three_x, x1)
        with torch.cuda.stream(s2):
            torch.cuda._sleep(sleep)
            y2 = tilefish.run(three_x, x2)
        torch.cuda.synchronize()
        if not (same_bytes(y1, want1) and same_bytes(y2, want2)):
            corrupted.append(i)
    print(f"\n[streams] one run {run_ms:.2f} ms, the second stream sleeps {sleep / cycles_per_ms:.2f} ms: {len(corrupted)} of {ROUNDS} rounds corrupted")
    assert corrupted == []


def test_runs_from_two_threads_each_on_its_own_stream_are_all_correct(inputs, cycles_per_ms, run_ms):
    x1, x2 = inputs
    results = {0: [], 1: []}
    errors = []
    barrier = threading.Barrier(2)

    def work(k, x):
        try:
            stream = torch.cuda.Stream()
            barrier.wait()
            with torch.cuda.stream(stream):
                for i in range(6):
                    torch.cuda._sleep(int(cycles_per_ms * run_ms * (0.25 + 0.25 * ((i + k) % 3))))
                    results[k].append(tilefish.run(three_x, x))
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
    wrong = [(k, i) for k in (0, 1) for i, y in enumerate(results[k]) if not same_bytes(y, want[k])]
    assert len(results[0]) == len(results[1]) == 6 and wrong == []


REPLAY_RUNS = 3000  # eager runs racing the replays of a captured plan (the reviewer's repro: 2 to 4 of about 4000 came out wrong with a raw CUDAGraph.replay)


def test_the_replays_of_tilefish_capture_from_another_thread_never_overlap_the_runs_of_the_plan(inputs):
    """The reviewer's repro of the race between a raw ``CUDAGraph.replay()`` (which takes no lock) and a run between its wait and its record: thread A runs the
    plan on ``s1`` in a loop and checks each result; thread B replays the plan's ``tilefish.capture`` on ``s2`` in a loop and checks each. ``CapturedPlan.replay``
    takes the arena's lock and waits for and records the order event around the graph, so not one result is wrong."""
    x1, x2 = inputs
    x_cap = x1.clone()
    out = torch.empty_like(x_cap)
    tilefish.run(three_x, x2)  # prepared
    cap = tilefish.capture(three_x, x_cap, outputs=out)
    want_cap, want2 = expected(x_cap), expected(x2)
    stop = threading.Event()
    counts = {"runs": 0, "runs_wrong": 0, "replays": 0, "replays_wrong": 0}
    errors = []

    def runs():
        try:
            s1 = torch.cuda.Stream()
            with torch.cuda.stream(s1):
                for _ in range(REPLAY_RUNS):
                    y = tilefish.run(three_x, x2)
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
                cap.replay(stream=s2)
                s2.synchronize()
                counts["replays"] += 1
                counts["replays_wrong"] += not same_bytes(out, want_cap)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=runs), threading.Thread(target=replays)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize()
    print(f"\n[capture] {counts['runs']} runs ({counts['runs_wrong']} wrong) and {counts['replays']} replays ({counts['replays_wrong']} wrong)")
    assert errors == [] and counts["runs"] == REPLAY_RUNS and counts["replays"] > 0
    assert counts["runs_wrong"] == 0 and counts["replays_wrong"] == 0


def test_clear_caches_while_another_thread_runs_the_plan_frees_nothing_under_a_run(inputs):
    """``clear_caches`` waits for a run in progress on another thread before it frees the arena, destroys its event and unloads the modules; the runs that
    start after it build, load and allocate again (before, a run kept going on a destroyed event and a freed arena)."""
    from tilefish import runtime

    x1, _ = inputs
    want = expected(x1)
    stop = threading.Event()
    counts = {"runs": 0, "wrong": 0}
    errors = []

    def runs():
        try:
            s1 = torch.cuda.Stream()
            with torch.cuda.stream(s1):
                while not stop.is_set():
                    y = tilefish.run(three_x, x1)
                    s1.synchronize()
                    counts["runs"] += 1
                    counts["wrong"] += not same_bytes(y, want)
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    worker = threading.Thread(target=runs)
    worker.start()
    try:
        for _ in range(8):
            before = counts["runs"]
            while counts["runs"] < before + 3 and worker.is_alive():  # a few runs on the new arena, then clear again
                time.sleep(0.005)
            runtime.clear_caches()
    finally:
        stop.set()
        worker.join()
    torch.cuda.synchronize()
    print(f"\n[clear_caches] 8 clears during {counts['runs']} runs on another thread: {counts['wrong']} wrong, {len(errors)} errors")
    assert errors == [] and counts["wrong"] == 0 and counts["runs"] >= 24
    assert same_bytes(tilefish.run(three_x, x1), want)


def test_a_second_run_of_a_plan_on_a_forked_stream_of_one_capture_is_refused_and_the_graph_still_replays(inputs):
    """Two runs of one plan on two forked streams of one ``torch.cuda.graph`` capture would have external wait nodes that do not wait for each other's record
    inside the graph, so a replay would run them at once: the second is refused before it enqueues anything, and the graph of the first replays correctly."""
    x1, x2 = inputs
    out1, out2 = torch.empty_like(x1), torch.empty_like(x2)
    tilefish.runtime.prepare(three_x)
    graph = torch.cuda.CUDAGraph()
    refused = []
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.cuda.graph(graph):
            main = torch.cuda.current_stream()
            s1.wait_stream(main)
            s2.wait_stream(main)
            with torch.cuda.stream(s1):
                tilefish.run(three_x, x1, outputs=out1)
            with torch.cuda.stream(s2):
                try:
                    tilefish.run(three_x, x2, outputs=out2)
                except tilefish.TilefishRuntimeError as e:
                    refused.append(str(e))
            main.wait_stream(s1)
            main.wait_stream(s2)
    assert len(refused) == 1 and "already run in this CUDA graph capture on another stream" in refused[0]
    out1.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert same_bytes(out1, expected(x1))


def _capture(fn):
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.cuda.graph(graph):
            fn()
    return graph


def test_replays_of_a_captured_graph_and_runs_on_other_streams_are_ordered_through_the_arena(inputs, cycles_per_ms, run_ms):
    """A captured graph's arena is the plan's arena: its replays wait for the runs issued before them and the runs issued after wait for them (external event nodes)."""
    x1, x2 = inputs
    x_cap = x1.clone()
    out = torch.empty_like(x_cap)
    tilefish.runtime.prepare(three_x)
    graph = _capture(lambda: tilefish.run(three_x, x_cap, outputs=out))
    want_cap, want2 = expected(x_cap), expected(x2)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    sleep = max(1, int(cycles_per_ms * run_ms / 2))
    corrupted = []
    for i in range(ROUNDS):
        torch.cuda.synchronize()
        out.zero_()
        if i % 2 == 0:  # a replay, then a run on another stream that would start in its middle
            with torch.cuda.stream(s1):
                graph.replay()
            with torch.cuda.stream(s2):
                torch.cuda._sleep(sleep)
                y2 = tilefish.run(three_x, x2)
        else:  # a run, then a replay on another stream that would start in its middle
            with torch.cuda.stream(s2):
                y2 = tilefish.run(three_x, x2)
            with torch.cuda.stream(s1):
                torch.cuda._sleep(sleep)
                graph.replay()
        torch.cuda.synchronize()
        if not (same_bytes(out, want_cap) and same_bytes(y2, want2)):
            corrupted.append(i)
    assert corrupted == []
