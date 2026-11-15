"""Carried buffers on a real GPU (``@tilefish.graph(carried=("h",))``, ``docs/plan-schema-1.md`` section 15.2).

The stages of ``gpu_kernels_plans`` hand ``h`` to each other through the arena they share, with no tensor of the caller's: ``stage_a`` writes it
(``3x``), ``stage_b`` reads it (``h + 1``; it is given nothing), ``stage_update`` adds to it in place, ``stage_zero`` zeroes it (a memset of the carried
buffer), ``stage_chain`` reads it through an intermediate of its own region, and ``stage_g`` writes the other carried buffer ``g``. Held here, in replay
and in jit: every call gives the bytes torch gives for the same sequence; graphs captured with ``tilefish.capture`` read and write ``h`` at every replay
in the order they are replayed, between runs on other streams; a ``torch.cuda.graph`` of ``stage_a`` then ``stage_b`` replays both; two arenas of the
same plans (``share_arena(..., name=...)``) on two streams from two threads keep two ``h``; and a call that reads ``h`` before any call wrote it is refused.
"""

import threading

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels_plans as k  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishPreconditionError, runtime  # noqa: E402

F32 = torch.float32


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("carried")
    planset = runtime.export(list(k.CARRIED_SET), directory)
    return {"mode": "replay", "plan": directory, "pin": planset.digest}


def modes(exported):
    return [("replay", exported), ("jit", {})]


@pytest.mark.parametrize("mode", ["replay", "jit"])
def test_the_stages_hand_h_to_each_other_and_give_torchs_bytes(exported, mode):
    opts = exported if mode == "replay" else {}
    s = runtime.share_arena(list(k.CARRIED_SET), name=f"bytes-{mode}", **opts)
    x8, y8, x24, y24 = (randn(shape, F32, seed=seed) for shape, seed in (((8, 64), 1), ((8, 64), 2), ((24, 64), 3), ((24, 64), 4)))
    y = s.run(k.stage_a, x8)
    h = x8 * 3.0
    assert same_bytes(y, x8 + 1.0)
    assert same_bytes(s.run(k.stage_b), h + 1.0)
    assert same_bytes(s.run(k.stage_update, y8), y8 * 3.0)
    h = h + y8
    assert same_bytes(s.run(k.stage_chain), h * 3.0 + 1.0)
    assert same_bytes(s.run(k.stage_g, x24), x24 * 3.0)  # g = x24 + 1: h is untouched
    assert same_bytes(s.run(k.stage_b), h + 1.0)
    assert same_bytes(s.run(k.stage_zero, x24), x24 * 3.0)
    h = torch.zeros((24, 64), dtype=F32, device="cuda")
    assert same_bytes(s.run(k.stage_b), h + 1.0)  # 24 rows now: the size h was written with
    s.run(k.stage_update, y24)
    assert same_bytes(s.run(k.stage_chain), (h + y24) * 3.0 + 1.0)
    # the reference mode takes the carried parameter as an argument, as it has no arena
    href = x8 * 3.0
    w = tilefish.run(k.stage_update, href, y8, mode="reference")
    assert same_bytes(w, y8 * 3.0) and same_bytes(href, x8 * 3.0 + y8)


def test_a_call_that_reads_h_before_any_call_wrote_it_is_refused(exported):
    s = runtime.share_arena(list(k.CARRIED_SET), name="refusals", **exported)
    with pytest.raises(TilefishPreconditionError, match="is not determined"):
        s.run(k.stage_b)
    with pytest.raises(TilefishPreconditionError, match="no execution on this arena has written it"):
        s.run(k.stage_update, randn((8, 64), F32, seed=5))
    s.run(k.stage_a, randn((8, 64), F32, seed=6))
    with pytest.raises(TilefishPreconditionError, match="carried buffer"):
        s.run(k.stage_update, randn((24, 64), F32, seed=7))  # h holds 8 rows
    # each plan alone has an arena of its own, with a carried region of its own: stage_b alone finds no h
    tilefish.run(k.stage_a, randn((8, 64), F32, seed=8), **exported)
    with pytest.raises(TilefishPreconditionError, match="is not determined"):
        tilefish.run(k.stage_b, **exported)


def test_captured_stages_read_and_write_h_in_the_order_they_are_replayed(exported):
    s = runtime.share_arena(list(k.CARRIED_SET), name="captured", **exported)
    x, y = randn((24, 64), F32, seed=10), randn((24, 64), F32, seed=11)
    s.run(k.stage_a, x)  # h exists (24 rows) before the captures: nothing runs at a capture, and a replay checks h
    cap_a = s.capture(k.stage_a, x)
    cap_b = s.capture(k.stage_b)
    cap_update = s.capture(k.stage_update, y)
    other = torch.cuda.Stream()
    h = x * 3.0
    for round_ in range(4):
        cap_a.replay()
        cap_update.replay(stream=other)  # another stream: ordered through the arena
        cap_b.replay()
        torch.cuda.synchronize()
        assert same_bytes(cap_b.outputs, h + y + 1.0), round_
    s.run(k.stage_zero, x)
    cap_b.replay()
    torch.cuda.synchronize()
    assert same_bytes(cap_b.outputs, torch.ones((24, 64), dtype=F32, device="cuda"))


def test_a_torch_graph_of_two_stages_replays_both(exported):
    s = runtime.share_arena(list(k.CARRIED_SET), name="torch-graph", **exported)
    x = randn((8, 64), F32, seed=12)
    y, z = torch.empty_like(x), torch.empty_like(x)
    s.run(k.stage_a, x)  # prepared: the arena, the binaries
    s.run(k.stage_b)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        s.run(k.stage_a, x, outputs=y)
        s.run(k.stage_b, outputs=z)
    for seed in (13, 14):
        x.copy_(randn((8, 64), F32, seed=seed))
        g.replay()
        torch.cuda.synchronize()
        assert same_bytes(y, x + 1.0) and same_bytes(z, x * 3.0 + 1.0)


ROUNDS = 40


def test_two_arenas_of_the_same_plans_keep_two_carried_buffers_from_two_threads(exported):
    lanes = [runtime.share_arena(list(k.CARRIED_SET), name=f"lane{i}", **exported) for i in range(2)]
    assert lanes[0].arena_bytes == lanes[1].arena_bytes
    xs = [randn((24, 64), F32, seed=20 + i) for i in range(2)]
    for s, x in zip(lanes, xs):
        s.run(k.stage_a, x)
    torch.cuda.synchronize()
    assert lanes[0].arena_of(0).ptr != lanes[1].arena_of(0).ptr
    results = {0: [], 1: []}
    errors = []
    barrier = threading.Barrier(2)

    def lane(i):
        try:
            stream = torch.cuda.Stream()
            barrier.wait()
            with torch.cuda.stream(stream):
                for _ in range(ROUNDS):
                    lanes[i].run(k.stage_a, xs[i])
                    results[i].append(lanes[i].run(k.stage_b))
            stream.synchronize()
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    threads = [threading.Thread(target=lane, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize()
    assert errors == []
    wrong = [(i, r) for i in range(2) for r, z in enumerate(results[i]) if not same_bytes(z, xs[i] * 3.0 + 1.0)]
    assert len(results[0]) == len(results[1]) == ROUNDS and wrong == []
