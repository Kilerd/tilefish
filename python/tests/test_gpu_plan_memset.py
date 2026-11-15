"""Memset steps of a launch plan (``tilefish.zeros_like``), on a real GPU.

``gpu_kernels_plans.zero_and_add`` zeroes the buffer of its returned accumulator and updates it in place twice (``(0 + x) + x``), zeroes the caller's
``InOut`` tensor before it updates it (one memset of its bytes), and returns an ``i32`` tensor of zeros. The caller's tensors are filled with NaN or
garbage first, so a missing memset shows. Plan replay, plan jit and the reference mode give the same bytes; the memsets capture as memset nodes, in a
graph of the plan's own and in a ``torch.cuda.graph``, whose replays zero again.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels_plans as k  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import runtime  # noqa: E402

F32, I32 = torch.float32, torch.int32


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("memset")
    return directory, runtime.export(k.zero_and_add, directory)


def garbage(shape, dtype):
    """A tensor whose bytes are a NaN's (not zero, whatever the dtype: f32 and i32 are both four bytes)."""
    return torch.full(shape, float("nan"), dtype=F32, device="cuda").view(dtype)


@pytest.mark.parametrize("rs", [8, 24])
def test_the_memsets_zero_the_callers_tensors_in_every_mode(exported, rs):
    directory, plan = exported
    assert [s["kind"] for s in plan.steps(Rs=rs)] == ["memset", "kernel", "kernel", "memset", "kernel", "memset"]
    x = randn((rs, 64), F32, seed=rs)
    for mode, extra in (("replay", {"plan": directory, "pin": plan.digest}), ("jit", {})):
        state = garbage((rs, 64), F32)
        acc, zeros = garbage((rs, 64), F32), garbage((rs, 64), I32)
        tilefish.run(k.zero_and_add, x, state, outputs=(acc, zeros), mode=mode, **extra)
        assert same_bytes(acc, x + x) and same_bytes(state, x) and same_bytes(zeros, torch.zeros((rs, 64), dtype=I32, device="cuda")), mode
    state = garbage((rs, 64), F32)
    acc, zeros = tilefish.run(k.zero_and_add, x, state, mode="reference")
    assert same_bytes(acc, x + x) and same_bytes(state, x) and not zeros.any()


def test_the_memsets_capture_as_memset_nodes_and_zero_again_at_every_replay(exported):
    directory, plan = exported
    opts = {"mode": "replay", "plan": directory, "pin": plan.digest}
    x = randn((24, 64), F32, seed=7)
    state = garbage((24, 64), F32)
    acc, zeros = garbage((24, 64), F32), garbage((24, 64), I32)
    cap = tilefish.capture(k.zero_and_add, x, state, outputs=(acc, zeros), **opts)
    for seed in (8, 9):
        x.copy_(randn((24, 64), F32, seed=seed))
        for t in (state, acc):
            t.fill_(float("nan"))
        zeros.fill_(-1)
        cap.replay()
        torch.cuda.synchronize()
        assert same_bytes(acc, x + x) and same_bytes(state, x) and not zeros.any()
    runtime.prepare(k.zero_and_add, **opts)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        tilefish.run(k.zero_and_add, x, state, outputs=(acc, zeros), **opts)
    for t in (state, acc):
        t.fill_(float("nan"))
    zeros.fill_(-1)
    g.replay()
    torch.cuda.synchronize()
    assert same_bytes(acc, x + x) and same_bytes(state, x) and not zeros.any()
