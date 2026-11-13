"""Copy steps of a launch plan, on a real GPU.

``gpu_kernels_p5.copy_steps`` copies ``x[1]`` (a dense view at an offset) into the arena with ``tilefish.copy``, updates the copy in place with
``add_into`` (which writes its ``r_out`` over the copy) and stores it into the caller's ``state`` (``state[...] = r``, an ``InOut`` parameter). Plan
replay, plan jit, the reference mode and the per-op eager composition give the same bytes, the caller's ``x`` is untouched, and the plan captures in a
CUDA graph (the copies are memcpy nodes) that replays them again.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels_p5 as k  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import runtime  # noqa: E402

F32 = torch.float32


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("copy-steps")
    return directory, runtime.export(k.copy_steps, directory)


def eager(x):
    y, r = tilefish.run(k.add_into, x[0].contiguous(), x[1].contiguous())
    return y, r


@pytest.mark.parametrize("rb", [8, 24])
def test_the_copies_give_the_bytes_of_the_eager_calls_in_every_mode(exported, rb):
    directory, plan = exported
    assert [s["kind"] for s in plan.steps(Rb=rb)] == ["copy", "kernel", "copy"]
    x = randn((2, rb, 64), F32, seed=rb)
    before = x.clone()
    want_y, want_state = eager(x)
    for mode, extra in (("replay", {"plan": directory, "pin": plan.digest}), ("jit", {}), ("reference", {})):
        state = torch.full((rb, 64), 7.0, dtype=F32, device="cuda")
        y = tilefish.run(k.copy_steps, x, state, mode=mode, **extra)
        assert same_bytes(y, want_y) and same_bytes(state, want_state), mode
        assert same_bytes(x, before), f"{mode}: the caller's x is never written"


def test_the_plan_with_copies_captures_in_a_cuda_graph(exported):
    directory, plan = exported
    x = randn((2, 24, 64), F32, seed=5)
    state = torch.zeros((24, 64), dtype=F32, device="cuda")
    cap = tilefish.capture(k.copy_steps, x, state, mode="replay", plan=directory, pin=plan.digest)
    want_y, want_state = eager(x)
    cap.replay()
    torch.cuda.synchronize()
    assert same_bytes(cap.outputs, want_y) and same_bytes(state, want_state)
    x.copy_(randn((2, 24, 64), F32, seed=6))  # new contents at the addresses the graph holds
    state.zero_()
    cap.replay()
    torch.cuda.synchronize()
    want_y, want_state = eager(x)
    assert same_bytes(cap.outputs, want_y) and same_bytes(state, want_state)
