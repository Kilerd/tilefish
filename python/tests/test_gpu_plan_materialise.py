"""Views that are not dense, returned or copied in a plan, on a real GPU: launches of ``tilefish.contiguous``.

``gpu_kernels_p6.strided_returns`` returns a transpose of a result and a ``tilefish.copy`` of a column block of its input (both materialised by the
generated op) and a dense row of the result (a copy step). Plan replay, plan jit, the reference mode and torch give the same bytes, and the plan captures.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels_p6 as k  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import runtime  # noqa: E402

F32 = torch.float32


@pytest.mark.parametrize("rg", [8, 24])
def test_a_transpose_a_column_block_and_a_row_come_back_in_every_mode(tmp_path, rg):
    plan = runtime.export(k.strided_returns, tmp_path)
    assert [s.get("entry", "copy") for s in plan.steps(Rg=rg)][1:3] == ["contiguous_r2_f32_last32_t64x32", "contiguous_r2_f32_t32x64"]
    x = randn((rg, 64), F32, seed=rg)
    t = x * 3.0
    want = (t.t().contiguous(), x[:, 32:].contiguous(), t[2].contiguous())
    for mode, extra in (("replay", {"plan": tmp_path, "pin": plan.digest}), ("jit", {}), ("reference", {})):
        got = tilefish.run(k.strided_returns, x, mode=mode, **extra)
        assert all(same_bytes(g, w) for g, w in zip(got, want)), mode
        assert all(g.is_contiguous() for g in got[:2]) or mode == "reference", mode
    cap = tilefish.capture(k.strided_returns, x, mode="replay", plan=tmp_path, pin=plan.digest)
    cap.replay()
    torch.cuda.synchronize()
    assert all(same_bytes(g, w) for g, w in zip(cap.outputs, want))
