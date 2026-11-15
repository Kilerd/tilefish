"""Views that are not dense stored into ``InOut`` parameters of a graph, on a real GPU: launches of ``tilefish.contiguous`` into the caller's tensors.

``gpu_kernels_plans.strided_store`` stores a transpose of a result (``tr[...] = t.permute(1, 0)``) and a column block of its input (``cols[...] =
x[:, 32:]``) into two ``InOut`` parameters. Plan replay, plan jit, the reference mode and torch give the same bytes; the plan captures in a graph of its
own (``tilefish.capture``) and in a ``torch.cuda.graph``, whose replays write the caller's tensors again.
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

F32 = torch.float32


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("strided-store")
    return directory, runtime.export(k.strided_store, directory)


def want_of(x):
    t = x * 3.0
    return t, t.t().contiguous(), x[:, 32:].contiguous()


@pytest.mark.parametrize("rs", [8, 24])
def test_the_views_are_stored_into_the_callers_tensors_in_every_mode(exported, rs):
    directory, plan = exported
    assert [s.get("entry") for s in plan.steps(Rs=rs)][1:] == ["contiguous_r2_f32_t32x64", "contiguous_r2_f32_last32_t64x32"]
    x = randn((rs, 64), F32, seed=rs)
    want_t, want_tr, want_cols = want_of(x)
    for mode, extra in (("replay", {"plan": directory, "pin": plan.digest}), ("jit", {}), ("reference", {})):
        tr = torch.full((64, rs), float("nan"), dtype=F32, device="cuda")
        cols = torch.full((rs, 32), float("nan"), dtype=F32, device="cuda")
        t = tilefish.run(k.strided_store, x, tr, cols, mode=mode, **extra)
        assert same_bytes(t, want_t) and same_bytes(tr, want_tr) and same_bytes(cols, want_cols), mode


def test_the_stores_capture_in_a_graph_of_their_own_and_in_a_torch_graph(exported):
    directory, plan = exported
    opts = {"mode": "replay", "plan": directory, "pin": plan.digest}
    x = randn((24, 64), F32, seed=3)
    tr, cols = torch.zeros((64, 24), dtype=F32, device="cuda"), torch.zeros((24, 32), dtype=F32, device="cuda")
    cap = tilefish.capture(k.strided_store, x, tr, cols, **opts)
    for seed in (4, 5):
        x.copy_(randn((24, 64), F32, seed=seed))
        tr.zero_()
        cols.zero_()
        cap.replay()
        torch.cuda.synchronize()
        want_t, want_tr, want_cols = want_of(x)
        assert same_bytes(cap.outputs, want_t) and same_bytes(tr, want_tr) and same_bytes(cols, want_cols)
    out = torch.empty((24, 64), dtype=F32, device="cuda")
    runtime.prepare(k.strided_store, **opts)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        tilefish.run(k.strided_store, x, tr, cols, outputs=out, **opts)
    x.copy_(randn((24, 64), F32, seed=6))
    tr.zero_()
    cols.zero_()
    g.replay()
    torch.cuda.synchronize()
    want_t, want_tr, want_cols = want_of(x)
    assert same_bytes(out, want_t) and same_bytes(tr, want_tr) and same_bytes(cols, want_cols)
