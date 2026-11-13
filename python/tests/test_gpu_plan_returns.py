"""Returned views of a launch plan, on a real GPU.

``gpu_kernels_p5.view_returns`` returns a flat result as ``[Rb, 64]`` (the caller's output is the buffer the kernel writes, through a flat view),
``x[0]`` (a dense view of a parameter, copied at the end) and the result again (copied). Plan replay, plan jit, the reference mode and torch give the
same bytes, the outputs are three tensors of their own, and the plan captures in a CUDA graph.
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
    directory = tmp_path_factory.mktemp("view-returns")
    return directory, runtime.export(k.view_returns, directory)


@pytest.mark.parametrize("rb", [8, 24])
def test_a_reshape_a_dense_view_and_a_value_returned_twice_come_back_in_every_mode(exported, rb):
    directory, plan = exported
    assert [s["kind"] for s in plan.steps(Rb=rb)] == ["kernel", "copy", "copy"]
    x = randn((2, rb, 64), F32, seed=rb)
    want = (x[1] * 3.0, x[0])
    for mode, extra in (("replay", {"plan": directory, "pin": plan.digest}), ("jit", {}), ("reference", {})):
        a, b, c = tilefish.run(k.view_returns, x, mode=mode, **extra)
        assert a.shape == (rb, 64) and same_bytes(a, want[0]) and same_bytes(b, want[1]) and same_bytes(c, want[0]), mode
        assert len({t.data_ptr() for t in (a, b, c)}) == 3 and x.data_ptr() not in {a.data_ptr(), b.data_ptr()}, mode


def test_the_plan_of_returned_views_captures_in_a_cuda_graph(exported):
    directory, plan = exported
    x = randn((2, 24, 64), F32, seed=3)
    cap = tilefish.capture(k.view_returns, x, mode="replay", plan=directory, pin=plan.digest)
    cap.replay()
    torch.cuda.synchronize()
    a, b, c = cap.outputs
    assert same_bytes(a, x[1] * 3.0) and same_bytes(b, x[0]) and same_bytes(c, x[1] * 3.0)
