"""An ``Out`` that may be written over an input (``MayAlias``), on a real GPU.

``gpu_kernels_p5.add_into`` writes ``r_out = r + x`` and ``y = 0.5 * r_out``; its ``r_out`` may be ``r``. A single launch given ``r`` as ``r_out``
(``outputs=``) writes the result over it, in ``jit`` and ``replay``, with the bytes of a launch into a tensor of its own; ``r_out`` over ``r`` shifted
by a row is refused before anything runs. The graph ``residual_chain`` makes the residual in the arena and updates it twice: the plan writes both
updates over it (one buffer for the three versions, the one the graph returns), and plan replay, plan jit, the per-op eager composition, the
reference mode and a CUDA graph capture give the same bytes.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels_p5 as k  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishPreconditionError, runtime  # noqa: E402

F32 = torch.float32


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("alias")
    manifest = runtime.export([k.add_into], directory)
    return directory, manifest.digest


@pytest.mark.parametrize("rows", [8, 24])
def test_a_launch_writes_the_result_over_the_input_it_may_alias(exported, rows):
    directory, digest = exported
    x, r = randn((rows, 64), F32, seed=rows), randn((rows, 64), F32, seed=rows + 1)
    want_r, want_y = r + x, (r + x) * 0.5
    apart = tilefish.run(k.add_into, x, r)  # r_out a tensor of its own
    assert same_bytes(apart[1], want_r) and same_bytes(apart[0], want_y)
    for mode, extra in (("jit", {}), ("replay", {"manifest": directory, "pin": digest})):
        r_in = r.clone()
        y = torch.empty_like(r)
        got = tilefish.run(k.add_into, x, r_in, mode=mode, outputs=(y, r_in), **extra)
        assert got[1] is r_in and got[0] is y
        assert same_bytes(r_in, apart[1]) and same_bytes(y, apart[0]), mode


def test_an_out_over_its_input_shifted_is_refused_and_nothing_runs(exported):
    directory, digest = exported
    x = randn((8, 64), F32, seed=3)
    big = randn((9, 64), F32, seed=4)
    r, r_out = big[:8], big[1:]  # overlapping, not one tensor
    before = big.clone()
    for mode, extra in (("jit", {}), ("replay", {"manifest": directory, "pin": digest})):
        with pytest.raises(TilefishPreconditionError, match="only as exactly that tensor") as e:
            tilefish.run(k.add_into, x, r, mode=mode, outputs=(torch.empty_like(x), r_out), **extra)
        assert e.value.kind == "alias"
    assert same_bytes(big, before)


@pytest.fixture(scope="module")
def exported_plan(tmp_path_factory):
    directory = tmp_path_factory.mktemp("residual-chain")
    return directory, runtime.export(k.residual_chain, directory)


def bindings(plan, rb):
    return [{a["name"]: a["binding"] for a in s["args"]} for s in plan.steps(Rb=rb)]


@pytest.mark.parametrize("rb", [8, 24])
def test_the_plan_updates_the_residual_in_place_and_gives_the_bytes_of_the_eager_calls(exported_plan, rb):
    directory, plan = exported_plan
    steps = bindings(plan, rb)
    assert steps[1]["r"] == steps[1]["r_out"] == steps[2]["r"] == steps[2]["r_out"] == steps[0]["out"]
    x = randn((rb, 64), F32, seed=rb)
    replay = tilefish.run(k.residual_chain, x, mode="replay", plan=directory, pin=plan.digest)
    jit = tilefish.run(k.residual_chain, x)
    reference = tilefish.run(k.residual_chain, x, mode="reference")
    r = tilefish.run(k.scaled, x)  # the per-op eager composition, each result a tensor of its own
    y1, r1 = tilefish.run(k.add_into, x, r)
    y2, r2 = tilefish.run(k.add_into, y1, r1)
    for got in (replay, jit, reference):
        assert same_bytes(got[0], y2) and same_bytes(got[1], r2)
    assert same_bytes(replay[1], (x * 2.0 + x) + (x * 2.0 + x) * 0.5)


def test_the_in_place_plan_captures_in_a_cuda_graph(exported_plan):
    directory, plan = exported_plan
    x = randn((24, 64), F32, seed=7)
    cap = tilefish.capture(k.residual_chain, x, mode="replay", plan=directory, pin=plan.digest)
    want = tilefish.run(k.residual_chain, x, mode="replay", plan=directory, pin=plan.digest)
    for _ in range(2):  # a replay writes the residual over itself again from x: the same bytes every time
        cap.replay()
        torch.cuda.synchronize()
        assert same_bytes(cap.outputs[0], want[0]) and same_bytes(cap.outputs[1], want[1])
