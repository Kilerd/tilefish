"""One step of an exported plan replayed under the plan's pin (``tilefish.runtime.run_step``), on a real GPU.

Every step of the transformer block of ``plan_model``, launched alone on the per-op eager composition's tensors (its inputs the eager results it reads,
its outputs fresh tensors), writes the eager result of its op byte for byte, in ``replay`` (the binary of the pinned plan's manifest) and in ``jit``.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import plan_model as pm  # noqa: E402
from gpu_util import same_bytes  # noqa: E402
from test_gpu_plan_taps import intermediates  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishRuntimeError, runtime  # noqa: E402


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("steps")
    return directory, runtime.export(pm.block, directory)


@pytest.mark.parametrize("m", [32, 128])
def test_every_step_alone_writes_the_eager_result_of_its_op(exported, m):
    directory, plan = exported
    w = pm.weights()
    x = pm.activations(m, seed=m + 3)
    n_t = torch.tensor([m - 5], dtype=torch.int32, device="cuda")
    eager = intermediates(x, n_t, w)
    y = tilefish.run(pm.add, eager[4], eager[8])
    tensors = {"x": x, "n": n_t, **{k: w[k] for k in pm.WEIGHT_NAMES}, **{f"t{i}": t for i, t in enumerate(eager)}, "out0": y}
    written = {0: ["t0"], 1: ["t1"], 2: ["t2"], 3: ["t3"], 4: ["t4", "t5"], 5: ["t6"], 6: ["t7"], 7: ["t8"], 8: ["out0"]}  # what each step of the block makes
    for mode, extra in (("replay", {"plan": directory, "pin": plan.digest}), ("jit", {})):
        for i, step in enumerate(plan.steps(M=m)):
            bound = list(dict.fromkeys(a["binding"] for a in step["args"]))
            assert set(written[i]) <= set(bound)
            given = {n: tensors[n] for n in bound}
            fresh = {n: torch.full_like(tensors[n], 7.0) for n in written[i]}
            given.update(fresh)
            runtime.run_step(pm.block, i, given, mode=mode, **extra)
            torch.cuda.synchronize()
            for n, t in fresh.items():
                assert same_bytes(t, tensors[n].reshape(t.shape)), f"{mode}: step {i} ({step['entry']}) wrote '{n}' other than its op"


def test_a_step_needs_a_trust_anchor_in_replay(exported):
    directory, plan = exported
    with pytest.raises(TilefishRuntimeError, match="trust anchor"):
        runtime.run_step(pm.block, 0, {}, mode="replay", plan=directory)
