"""Debug taps of a launch plan, on a real GPU.

The transformer block of ``plan_model`` planned with liveness reuse (its intermediates share bytes): a tap of every intermediate, after the last step that
writes it, is byte-identical to the result of that op called on its own (the per-op eager composition), in ``replay`` and ``jit``, although later steps
write over those bytes; a tap after a step given sees what that step left; the outputs are the same with and without taps, and a captured plan copies
its taps at every replay.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import plan_model as pm  # noqa: E402
from gpu_util import same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import runtime  # noqa: E402

TORCH = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("taps")
    return directory, runtime.export(pm.block, directory, reuse=True)


def intermediates(x, n_t, w):
    """What the block's ops make, each called on its own: the intermediates t0 ... t8 of its plan, in the order its steps make them."""
    m = x.shape[0]
    h = tilefish.run(pm.rmsnorm, x, w["w_ln1"])
    qkv = tilefish.run(pm.gemm_qkv, h, w["w_qkv"], w["b_qkv"])
    fused = qkv.view(m, 3, pm.HEADS, pm.D_HEAD)
    a = tilefish.run(pm.attention, fused[:, 0], fused[:, 1], fused[:, 2], n_t)
    o = tilefish.run(pm.gemm_o, a.view(m, pm.D_MODEL), w["w_o"])
    r, h2 = tilefish.run(pm.add_rmsnorm, x, o, w["w_ln2"])
    u = tilefish.run(pm.gemm_up, h2, w["w_up"])
    g = tilefish.run(pm.gelu, u)
    d = tilefish.run(pm.gemm_down, g, w["w_down"])
    return [h, qkv, a, o, r, h2, u, g, d]


def taps_for(plan, m):
    names = [b["name"] for b in plan.buffers if b["role"] == "intermediate"]
    dtype = {b["name"]: TORCH[b["dtype"]] for b in plan.buffers if b["name"] in names}
    return names, {n: torch.full(plan.shape_of(n, M=m), 7.0, dtype=dtype[n], device="cuda") for n in names}


@pytest.mark.parametrize("m", pm.BUCKETS)
def test_every_intermediate_tapped_is_the_result_of_its_op(exported, m):
    directory, plan = exported
    w = pm.weights()
    x = pm.activations(m, seed=m)
    n_t = torch.tensor([m // 2 + 1], dtype=torch.int32, device="cuda")
    args = (x, n_t, *[w[k] for k in pm.WEIGHT_NAMES])
    want = intermediates(x, n_t, w)
    plain = tilefish.run(pm.block, *args, mode="replay", plan=directory, pin=plan.digest)
    for mode, extra in (("replay", {"plan": directory, "pin": plan.digest}), ("jit", {"plan": tilefish.plan(pm.block, reuse=True)})):
        names, taps = taps_for(plan, m)
        y = tilefish.run(pm.block, *args, mode=mode, taps=taps, **extra)
        torch.cuda.synchronize()
        assert same_bytes(y, plain), f"{mode}: the output with taps"
        assert len(names) == len(want) == 9
        for name, expect in zip(names, want):
            assert same_bytes(taps[name], expect.reshape(taps[name].shape)), f"{mode}: tap {name}"


def test_a_tap_after_a_step_given_and_in_a_captured_plan(exported):
    directory, plan = exported
    w = pm.weights()
    x = pm.activations(64, seed=5)
    n_t = torch.tensor([40], dtype=torch.int32, device="cuda")
    want = intermediates(x, n_t, w)
    qkv_after_attention = torch.empty(plan.shape_of("t1", M=64), dtype=torch.bfloat16, device="cuda")
    tilefish.run(pm.block, x, n_t, *[w[k] for k in pm.WEIGHT_NAMES], mode="replay", plan=directory, pin=plan.digest, taps={("t1", 2): qkv_after_attention})
    torch.cuda.synchronize()
    assert same_bytes(qkv_after_attention, want[1])  # step 2 (the attention) reads t1 and writes t2: t1 is what step 1 left
    gelu = torch.empty(plan.shape_of("t7", M=64), dtype=torch.bfloat16, device="cuda")
    cap = tilefish.capture(pm.block, x, *[w[k] for k in pm.WEIGHT_NAMES], mode="replay", plan=directory, pin=plan.digest, devscalars={"n": 40}, taps={"t7": gelu})
    for _ in range(2):
        gelu.zero_()
        cap.replay()
        torch.cuda.synchronize()
        assert same_bytes(gelu, want[7])
