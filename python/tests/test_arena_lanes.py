"""Several arenas of the same plans, and the warning for a CUDA graph of the caller's used from several threads, with no GPU (a fake backend).

* ``share_arena(graphs, name=...)`` (``SharedArena(..., name=...)``): every arena of the same plans and the same name on a device is one, another name
  is another arena, with its own memory, lock, order event and carried buffers; so executions that must run concurrently (one per stream) take an
  arena each (``docs/en/plan-schema-1.md``, sections 9 and 15.1). A plan alone is a shared arena of one plan.
* ``tilefish.run`` inside ``torch.cuda.graph`` captures a graph whose replay takes no lock of tilefish (section 10, item 7): an arena that such a graph
  holds and that a second thread uses gets one ``TilefishOrderWarning``; ``tilefish.runtime.replay_graph`` (which takes the locks) silences it.

The real launches are ``test_gpu_arena_lanes.py``.
"""

import threading
import warnings

import gpu_kernels_plans as k
import pytest
from fake_torch import FakeTensor
from test_planrun import FakeBackend, FakeProvider

import tilefish
from tilefish import TilefishOrderWarning, TilefishRuntimeError
from tilefish._planrun import PlanRunner, SharedArena



class Backend(FakeBackend):
    def memset(self, dst, nbytes, stream):
        self.log.append(("memset", dst, nbytes, stream))

    def current_device(self):
        return 0


def lane(name, be, graphs=(k.stage_a, k.stage_b)):
    return SharedArena([PlanRunner(tilefish.plan(g), FakeProvider(), be) for g in graphs], name=name)


def x_of(rows, at=1 << 36):
    return FakeTensor((rows, 64), "f32", at)


def test_a_name_is_another_arena_and_the_same_name_the_same_one():
    be = Backend()
    first, second, again, unnamed = lane("lane0", be), lane("lane1", be), lane("lane0", be), lane(None, be)
    assert len({first.key, second.key, unnamed.key}) == 3 and again.key == first.key
    assert "name='lane0'" in repr(first) and "name" not in repr(unnamed)
    for s in (first, second, unnamed):
        s.run(k.stage_a, x_of(8))
    assert be.arenas == 3  # one arena per name (and one without a name): lane0's twice is one
    arenas = [s.arena_of(0) for s in (first, second, again, unnamed)]
    assert arenas[0] is arenas[2] and len({id(a) for a in arenas}) == 3
    assert len({id(a.lock) for a in arenas}) == 3 and len({id(a.event) for a in arenas}) == 3 and all(a.event is not None for a in arenas)


def test_each_arena_has_its_own_order_event_and_its_own_carried_buffers():
    be = Backend()
    first, second = lane("lane0", be), lane("lane1", be)
    first.run(k.stage_a, x_of(8))
    with pytest.raises(tilefish.TilefishPreconditionError, match="not determined"):
        second.run(k.stage_b)  # lane1's h was never written: the carried buffers are the arena's
    second.run(k.stage_a, x_of(24))
    assert first.run(k.stage_b).shape == (8, 64) and second.run(k.stage_b).shape == (24, 64)
    records = [e for e in be.log if e[0] == "record"]
    assert len({r[1] for r in records}) == 2  # two order events: an execution on lane0 never waits for one on lane1
    assert not [e for e in be.log if e[0] == "wait"]  # all on one stream: each event's last record is on it, so no run waits (section 9)
    be.current, before = be.current + 1, len(be.log)  # on another stream, each run waits for its own arena's event only
    first.run(k.stage_b), second.run(k.stage_b)
    waits = [e for e in be.log[before:] if e[0] == "wait"]
    assert [w[1] for w in waits] == [first.arena_of(0).event.name, second.arena_of(0).event.name]


def test_a_name_is_a_non_empty_string():
    be = Backend()
    for bad in ("", 3, b"x"):
        with pytest.raises(TilefishRuntimeError, match="name= of a shared arena is a non-empty string"):
            lane(bad, be)


def test_a_caller_graph_used_from_two_threads_warns_once_and_one_thread_does_not():
    be = Backend()
    s = lane(None, be, graphs=(k.stage_a,))
    s.run(k.stage_a, x_of(8))  # outside the capture: the arena and the binaries
    be.capture = True
    with warnings.catch_warnings():
        warnings.simplefilter("error", TilefishOrderWarning)
        s.run(k.stage_a, x_of(8))  # captured in a graph of the caller's, on this thread only: no warning
        be.capture = False
        s.run(k.stage_a, x_of(8))
    caught = []

    def other():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.run(k.stage_a, x_of(8))
            s.run(k.stage_a, x_of(8))
        caught.extend(w)

    t = threading.Thread(target=other)
    t.start()
    t.join()
    order = [w for w in caught if issubclass(w.category, TilefishOrderWarning)]
    assert len(order) == 1 and "tilefish.runtime.replay_graph" in str(order[0].message) and "tilefish.capture" in str(order[0].message)


def test_as_an_error_the_warning_refuses_every_use_from_another_thread_and_enqueues_nothing():
    be = Backend()
    s = lane(None, be, graphs=(k.stage_a,))
    s.run(k.stage_a, x_of(8))
    be.capture = True
    s.run(k.stage_a, x_of(8))  # a graph of the caller's holds the arena
    be.capture = False
    outcomes = []

    def other():
        with warnings.catch_warnings():
            warnings.simplefilter("error", TilefishOrderWarning)
            for _ in range(3):
                before = len(be.launches())
                try:
                    s.run(k.stage_a, x_of(8))
                    outcomes.append("ran")
                except TilefishOrderWarning:
                    outcomes.append("refused" if len(be.launches()) == before else "refused after enqueueing")

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert outcomes == ["refused"] * 3 and not s.arena_of(0).warned


def test_a_run_refused_inside_a_caller_capture_does_not_count_as_a_run_of_the_capture():
    be = Backend()
    s = lane(None, be, graphs=(k.stage_a, k.stage_update))
    s.run(k.stage_a, x_of(8))  # h holds 8 rows
    be.capture = True
    be.current = 11
    with pytest.raises(tilefish.TilefishPreconditionError, match="carried buffer"):
        s.run(k.stage_update, x_of(24))  # refused on stream 11: the capture has no run of the arena on it
    be.current = 12
    s.run(k.stage_a, x_of(8))  # so a run on another stream of the same capture is the first, not a second on another stream
    be.capture = False


def test_an_arena_without_a_caller_graph_does_not_warn_from_many_threads():
    be = Backend()
    s = lane(None, be, graphs=(k.stage_a,))
    errors = []

    def go():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", TilefishOrderWarning)
                s.run(k.stage_a, x_of(8))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_the_warning_is_not_given_once_the_caller_replays_through_replay_graph():
    be = Backend()
    s = lane(None, be, graphs=(k.stage_a,))
    s.run(k.stage_a, x_of(8))
    be.capture = True
    s.run(k.stage_a, x_of(8))
    be.capture = False
    s.arena_of(0).ordered = True  # what tilefish.runtime.replay_graph sets (it needs torch: test_gpu_arena_lanes.py)
    caught = []

    def other():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.run(k.stage_a, x_of(8))
        caught.extend(w)

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert not [w for w in caught if issubclass(w.category, TilefishOrderWarning)]
