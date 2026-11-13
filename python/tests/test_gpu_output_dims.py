"""A dimension that only an ``Out`` tensor determines, on a real GPU.

``gpu_kernels_p4.key_mask`` writes a ``[B]`` bf16 mask (``0`` for the keys ``j < n``, ``-inf`` after) from a device scalar ``n`` alone: no input has
``B``. Run with ``outputs=`` (the mask tensor, which binds ``B``), ``jit`` and ``replay`` agree byte for byte and equal torch, for the buckets of the
bucketed ``key_mask`` and any size of ``key_mask_any``; a mask of another bucket and a call without ``outputs=`` are refused before anything runs. The
graph ``mask_rows`` makes the mask in a step whose ``B`` is the graph's, and adds it to the rows of ``x``: plan replay, plan jit, the per-op eager
composition and the reference mode (whose ``reference=`` of ``key_mask`` is given ``b``) give the same bytes, and the plan captures in a CUDA graph.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels_p4 as k  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishPreconditionError, TilefishRuntimeError, runtime  # noqa: E402

BF16, I32 = torch.bfloat16, torch.int32


def want_mask(b, n):
    keys = torch.arange(b, device="cuda")
    return torch.where(keys < n, 0.0, float("-inf")).to(BF16)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("output-dims")
    manifest = runtime.export([k.key_mask, k.key_mask_any, k.apply_mask], directory)
    return directory, manifest.digest


def mask_both(op, exported, b, n):
    directory, digest = exported
    n_t = torch.tensor(n, dtype=I32, device="cuda")
    jit = tilefish.run(op, n_t, outputs=torch.full((b,), 7.0, dtype=BF16, device="cuda"))  # every element is written
    out = torch.full((b,), 7.0, dtype=BF16, device="cuda")
    replay = tilefish.run(op, n_t, mode="replay", manifest=directory, pin=digest, outputs=out)
    assert replay is out, "the result is the tensor given"
    assert same_bytes(jit, replay), f"{op.name}: jit and replay differ"
    return replay


@pytest.mark.parametrize("b", [8, 24])
@pytest.mark.parametrize("n", [0, 5, 8, 24, 100])
def test_the_mask_of_a_bucket_is_bound_from_the_out_tensor(exported, b, n):
    assert same_bytes(mask_both(k.key_mask, exported, b, n), want_mask(b, n))


@pytest.mark.parametrize("b", [1, 130, 1000, 4097])
def test_an_unbucketed_mask_takes_any_size(exported, b):
    for n in (0, b // 2, b):
        assert same_bytes(mask_both(k.key_mask_any, exported, b, n), want_mask(b, n))


def test_a_mask_of_another_size_or_none_is_refused_and_nothing_runs(exported):
    directory, digest = exported
    n_t = torch.tensor(3, dtype=I32, device="cuda")
    out = torch.full((16,), 7.0, dtype=BF16, device="cuda")
    for mode, extra in (("jit", {}), ("replay", {"manifest": directory, "pin": digest})):
        with pytest.raises(TilefishPreconditionError, match="dimension Rb is 16") as e:
            tilefish.run(k.key_mask, n_t, mode=mode, outputs=out, **extra)
        assert e.value.kind == "bucket"
        with pytest.raises(TilefishRuntimeError, match="give it with outputs="):
            tilefish.run(k.key_mask, n_t, mode=mode, **extra)
    assert bool((out == 7.0).all()), "a refused call writes nothing"


@pytest.fixture(scope="module")
def exported_plan(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mask-rows-plan")
    return directory, runtime.export(k.mask_rows, directory)


@pytest.mark.parametrize("rb", [8, 24])
@pytest.mark.parametrize("n", [0, 5, 24])
def test_a_plan_makes_the_mask_in_a_step_whose_dimension_is_the_graphs(exported_plan, rb, n):
    directory, plan = exported_plan
    x = randn((rb, 64), torch.float32, seed=rb + n)
    n_t = torch.tensor([n], dtype=I32, device="cuda")
    replay = tilefish.run(k.mask_rows, x, n_t, mode="replay", plan=directory, pin=plan.digest)
    jit = tilefish.run(k.mask_rows, x, n_t)
    reference = tilefish.run(k.mask_rows, x, n_t, mode="reference")
    mask = tilefish.run(k.key_mask, n_t, outputs=torch.empty((rb,), dtype=BF16, device="cuda"))  # the per-op eager composition
    eager = tilefish.run(k.apply_mask, x, mask)
    assert same_bytes(replay, jit) and same_bytes(replay, eager) and same_bytes(replay, reference)
    assert same_bytes(replay, x + want_mask(rb, n).float()[:, None])


def test_the_plan_of_the_mask_captures_in_a_cuda_graph(exported_plan):
    directory, plan = exported_plan
    x = randn((24, 64), torch.float32, seed=1)
    cap = tilefish.capture(k.mask_rows, x, mode="replay", plan=directory, pin=plan.digest, devscalars={"n": 5})
    cap.replay()
    torch.cuda.synchronize()
    assert same_bytes(cap.outputs, x + want_mask(24, 5).float()[:, None])
    cap.set_devscalar("n", 17)
    cap.replay()
    torch.cuda.synchronize()
    assert same_bytes(cap.outputs, x + want_mask(24, 17).float()[:, None])
