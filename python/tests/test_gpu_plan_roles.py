"""Declared roles and weights bound once, on a real GPU.

``gpu_kernels_p6.static_affine`` is a static graph that declares ``x`` its input (``w`` and ``b`` are weights). ``bind_weights`` binds the weights
once; its runs take ``x`` only and give the bytes of a run given everything, in ``jit`` and ``replay``; a capture through it replays with a new
``x`` written into the captured tensor. A plan whose roles were chosen from the shapes is refused.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels_p6 as k  # noqa: E402
import plan_model as pm  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishPreconditionError, TilefishRuntimeError, runtime  # noqa: E402

F32 = torch.float32


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("roles")
    return directory, runtime.export(k.static_affine, directory)


def test_the_weights_bound_once_give_the_bytes_of_a_run_given_everything(exported):
    directory, plan = exported
    assert plan.roles_declared and plan.inputs == ["x"] and plan.weights == ["w", "b"]
    w, b = randn((16, 64), F32, seed=1), randn((16, 64), F32, seed=2)
    for mode, extra in (("jit", {}), ("replay", {"plan": directory, "pin": plan.digest})):
        bound = runtime.bind_weights(k.static_affine, {"w": w, "b": b}, mode=mode, **extra)
        for seed in range(3):
            x = randn((16, 64), F32, seed=10 + seed)
            got = bound.run(x)
            want = tilefish.run(k.static_affine, x, w, b, mode=mode, **extra)
            assert same_bytes(got, want), (mode, seed)
            assert torch.allclose(got, 2.0 * (x * w + b), rtol=1e-6, atol=1e-6), (mode, seed)  # (one FMA rounding or two: allclose)
        out = torch.empty((16, 64), dtype=F32, device="cuda")
        assert bound.run(x=x, outputs=out) is out and same_bytes(out, want)
        with pytest.raises(TilefishPreconditionError, match="w is a weight, bound once"):
            bound.run(x, w=w)


def test_a_capture_with_the_weights_bound_replays_a_new_input(exported):
    directory, plan = exported
    w, b = randn((16, 64), F32, seed=3), randn((16, 64), F32, seed=4)
    bound = runtime.bind_weights(k.static_affine, {"w": w, "b": b}, mode="replay", plan=directory, pin=plan.digest)
    x = randn((16, 64), F32, seed=5)
    cap = bound.capture(x)
    for seed in range(3):
        x.copy_(randn((16, 64), F32, seed=20 + seed))
        cap.replay()
        torch.cuda.synchronize()
        assert same_bytes(cap.outputs, tilefish.run(k.static_affine, x, w, b, mode="replay", plan=directory, pin=plan.digest))


def test_a_plan_whose_roles_were_guessed_is_refused():
    with pytest.raises(TilefishRuntimeError, match="does not declare which parameters are weights"):
        runtime.bind_weights(pm.block, pm.weights())
