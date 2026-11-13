"""Strides that several views share, on a real GPU: ``StrideOf(("q", "k", "v"), axis)``.

``gpu_kernels_p4.qkv_sum`` reads three views with one stride pair: ``o = (q + k) + v`` in f32, rounded once to bf16. Given the q, k and v of one fused
``[R, 3, 4, 64]`` tensor (offsets 0, 256 and 512 elements, strides ``(768, 64, 1)``) or three tensors of their own with the same strides, ``jit`` and ``replay``
agree byte for byte and equal torch; a ``k`` or ``v`` of other strides is refused by both before anything is written. ``fused_qkv`` is a graph that makes
the fused tensor in the arena with a lowered kernel and reads its three views in one step: plan replay, plan jit and the per-op eager composition give the
same bytes, and so does the reference mode.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels_p4 as k  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishPreconditionError, runtime  # noqa: E402

BF16 = torch.bfloat16


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("shared-strides")
    manifest = runtime.export([k.qkv_sum, k.qkv_sum16], directory)
    return directory, manifest.digest


def run_both(op, exported, *args):
    directory, digest = exported
    jit = tilefish.run(op, *args)
    replay = tilefish.run(op, *args, mode="replay", manifest=directory, pin=digest)
    assert same_bytes(jit, replay), f"{op.name}: jit and replay differ"
    return replay


def want(q, kk, v):
    return ((q.float() + kk.float()) + v.float()).to(BF16)


@pytest.mark.parametrize("rows", [1, 7, 40])
def test_the_q_k_and_v_of_one_fused_tensor_are_read_with_one_stride_pair(exported, rows):
    fused = randn((rows, 3, 4, 64), BF16, seed=rows)
    q, kk, v = fused[:, 0], fused[:, 1], fused[:, 2]
    assert q.stride() == kk.stride() == v.stride() == (768, 64, 1)
    assert [t.storage_offset() for t in (q, kk, v)] == [0, 256, 512]
    for op in (k.qkv_sum, k.qkv_sum16):
        assert same_bytes(run_both(op, exported, q, kk, v), want(q, kk, v)), op.name


@pytest.mark.parametrize("rows", [1, 7, 40])
def test_three_tensors_with_the_same_strides_are_fine(exported, rows):
    q, kk, v = (randn((rows, 4, 64), BF16, seed=rows + i) for i in range(3))
    for op in (k.qkv_sum, k.qkv_sum16):
        assert same_bytes(run_both(op, exported, q, kk, v), want(q, kk, v)), op.name


@pytest.mark.parametrize("which", [1, 2])
def test_a_view_without_the_shared_stride_is_refused_by_jit_and_replay(exported, which):
    directory, digest = exported
    fused = randn((7, 3, 4, 64), BF16, seed=3)
    views = [fused[:, 0], fused[:, 1], fused[:, 2]]
    views[which] = randn((7, 4, 64), BF16, seed=4)  # its own tensor: row stride 256, where q bound 768
    name = "qkv"[which]
    for mode, extra in (("jit", {}), ("replay", {"manifest": directory, "pin": digest})):
        with pytest.raises(TilefishPreconditionError, match=rf"argument '{name}'.*the stride of axis 0 is 256, but the signature requires s_row = 768") as e:
            tilefish.run(k.qkv_sum, *views, mode=mode, **extra)
        assert e.value.kind == "stride"


@pytest.fixture(scope="module")
def exported_plan(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fused-qkv-plan")
    return directory, runtime.export(k.fused_qkv, directory)


@pytest.mark.parametrize("rb", [8, 24])
def test_a_plan_reads_the_three_views_of_an_intermediate_in_one_step(exported_plan, rb):
    directory, plan = exported_plan
    x = randn((rb, 768), BF16, seed=rb)
    replay = tilefish.run(k.fused_qkv, x, mode="replay", plan=directory, pin=plan.digest)
    jit = tilefish.run(k.fused_qkv, x)
    reference = tilefish.run(k.fused_qkv, x, mode="reference")
    fused = tilefish.run(k.doubled, x).view(rb, 3, 4, 64)  # the per-op eager composition
    eager = tilefish.run(k.qkv_sum16, fused[:, 0], fused[:, 1], fused[:, 2])
    assert same_bytes(replay, jit) and same_bytes(replay, eager) and same_bytes(replay, reference)
    assert same_bytes(replay, want(fused[:, 0], fused[:, 1], fused[:, 2]))
    (step,) = [s for s in plan.steps(Rb=rb) if s["entry"] == "qkv_sum16"]
    assert [a["view"]["offset"] for a in step["args"][:3]] == [0, 256, 512]
    assert all(a["binding"] == step["args"][0]["binding"] and a["view"]["strides"] == [768, 64, 1] for a in step["args"][:3])
