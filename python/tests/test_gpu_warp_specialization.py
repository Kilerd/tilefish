"""Warp-specialised kernels on a real GPU: the kernels of ``ws_kernels.py`` (Gluon's ``gl.warp_specialize`` and Triton's own warp specialisation),
over TMA descriptors, sm_90 or newer (the tcgen05 matmuls on sm_100 and sm_103 only).

The declaration's ``num_warps`` is the warps of the default partition; Triton 3.5.1 reports the warps of all the partitions as the kernel's
``num_warps``, and tilefish launches with that block (``total_num_warps`` in the manifest entry). Held here:

* the manifest entry of each kernel records the compiled warps (4 + 1 TMA producer: 8; 4 + 1 + 1: 8; 8 + 1 + 1: 12; Triton's partitions of a
  ``tl.range(..., warp_specialize=True)`` loop: 8 on sm_100 and up) and the block they make, and keeps the declared ``num_warps``;
* ``jit``, ``replay``, a ``torch.cuda.graph`` replay of a ``replay`` and Triton's own launch of the same kernel give the same bytes, and the bytes are
  the reference's (exactly for the scaling, within the accumulation's rounding for the matmuls);
* a plan whose steps have blocks of different warps (a warp-specialised step and a plain one) gives the bytes of its ``jit``, its ``tilefish.capture``
  replay and a ``torch.cuda.graph`` replay;
* the build still refuses a kernel that is not warp-specialised and whose warps Triton changed, and a warp-specialised one whose partitions add none.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("triton.experimental.gluon")
pytest.importorskip("cuda.bindings.driver")

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:  # pragma: no cover - TMA needs sm_90
    pytest.skip("warp-specialised TMA kernels need sm_90 or newer", allow_module_level=True)

import ws_kernels as wk  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishBuildError, runtime  # noqa: E402
from tilefish.aot import build, compile_specs  # noqa: E402

F16 = torch.float16
CAP = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
ARCH = f"sm_{CAP[0] * 10 + CAP[1]}"
OPS = wk.ops(CAP)
needs_tcgen05 = pytest.mark.skipif(not wk.tcgen05(CAP), reason="tcgen05 (sm_100, sm_103) only")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ws")
    manifest = runtime.export(list(OPS.values()), directory)
    return directory, manifest


def entry_of(exported, op):
    directory, manifest = exported
    (spec,) = compile_specs(op)
    return manifest.get(spec.stem, ARCH).entry


def run_all(op, exported, *args):
    """``jit``, ``replay`` and a ``torch.cuda.graph`` replay of the ``replay`` (outputs of the caller, inputs at the addresses the graph holds): the
    same bytes, returned."""
    directory, manifest = exported
    kw = {"mode": "replay", "manifest": directory, "pin": manifest.digest}
    jit = tilefish.run(op, *args)
    replay = tilefish.run(op, *args, **kw)
    assert same_bytes(jit, replay), f"{op.name}: jit and replay differ"
    out = torch.full_like(replay, 7.0)
    tilefish.run(op, *args, outputs=out, **kw)  # once outside the capture: loaded, and its descriptors encoded once
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    captured = torch.full_like(replay, 7.0)
    with torch.cuda.graph(graph):
        tilefish.run(op, *args, outputs=captured, **kw)
    graph.replay()
    torch.cuda.synchronize()
    assert same_bytes(captured, replay), f"{op.name}: a CUDA graph replay and the replay differ"
    return replay


def gluon_desc(t, block):
    from triton.experimental.gluon.language._layouts import NVMMASharedLayout
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    layout = NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    return TensorDescriptor(t, list(t.shape), list(t.stride()), list(block), layout)


def host_desc(t, block):
    from triton.tools.tensor_descriptor import TensorDescriptor

    return TensorDescriptor(t, list(t.shape), list(t.stride()), list(block))


def test_the_manifest_records_the_warps_of_all_the_partitions(exported):
    e = entry_of(exported, wk.ws_scale)
    assert (e["num_warps"], e["options"]["num_warps"], e["total_num_warps"], e["block"]) == (4, 4, 8, [256, 1, 1])
    e = entry_of(exported, wk.tl_ws_matmul)
    if CAP[0] >= 10:  # Triton warp-specialises the loop on sm_100 and up
        assert (e["num_warps"], e["total_num_warps"], e["block"]) == (4, 8, [256, 1, 1])
    else:
        assert "total_num_warps" not in e and e["block"] == [128, 1, 1]
    if wk.tcgen05(CAP):
        e = entry_of(exported, wk.ws_matmul)
        assert (e["num_warps"], e["total_num_warps"], e["block"]) == (4, 8, [256, 1, 1])
        e = entry_of(exported, wk.ws_matmul_wide)
        assert (e["num_warps"], e["total_num_warps"], e["block"]) == (8, 12, [384, 1, 1])


@pytest.mark.parametrize("shape", [(64, 64), (130, 200), (256, 512), (7, 72)])
def test_a_producer_partition_feeds_the_compute_partition(exported, shape):
    x = randn(shape, F16, seed=shape[0] + shape[1])
    y = run_all(wk.ws_scale, exported, x)
    assert same_bytes(y, wk.scaled(x))
    mine = torch.full_like(x, 7.0)  # Triton's own launch: its descriptors, its block (32 * the metadata's warps)
    wk.ws_scale_kernel[(-(-shape[0] // 64),)](gluon_desc(x, (64, 64)), gluon_desc(mine, (64, 64)), shape[1], 64, 64, 3, 4, num_warps=4)
    torch.cuda.synchronize()
    assert same_bytes(mine, y)


MNK = [(128, 128, 64), (256, 384, 192), (200, 136, 72), (64, 64, 512)]


def check_matmul(c, a, b):
    want = (a.double() @ b.double().T).to(F16)
    assert torch.allclose(c.float(), want.float(), rtol=2e-2, atol=2e-2)


@needs_tcgen05
@pytest.mark.parametrize("mnk", MNK)
@pytest.mark.parametrize("wide", [False, True])
def test_a_producer_an_issuer_and_an_epilogue_partition_multiply(exported, mnk, wide):
    m, n, k = mnk
    a, b = randn((m, k), F16, seed=m, scale=0.5), randn((n, k), F16, seed=n + 1, scale=0.5)
    op, nw = (wk.ws_matmul_wide, 8) if wide else (wk.ws_matmul, 4)
    c = run_all(op, exported, a, b)
    check_matmul(c, a, b)
    mine = torch.full((m, n), 7.0, dtype=F16, device="cuda")
    descs = (gluon_desc(a, (128, 64)), gluon_desc(b, (128, 64)), gluon_desc(mine, (128, 128)))
    wk.ws_matmul_kernel[(-(-m // 128), -(-n // 128))](*descs, k, 128, 128, 64, 3, nw, num_warps=nw)
    torch.cuda.synchronize()
    assert same_bytes(mine, c)


@pytest.mark.parametrize("mnk", MNK)
def test_a_loop_triton_warp_specialises_itself(exported, mnk):
    m, n, k = mnk
    a, b = randn((m, k), F16, seed=m + 2, scale=0.5), randn((n, k), F16, seed=n + 3, scale=0.5)
    c = run_all(wk.tl_ws_matmul, exported, a, b)
    check_matmul(c, a, b)
    mine = torch.full((m, n), 7.0, dtype=F16, device="cuda")
    descs = (host_desc(a, (128, 64)), host_desc(b, (128, 64)), host_desc(mine, (128, 128)))
    wk.tl_ws_matmul_kernel[(-(-m // 128), -(-n // 128))](*descs, k, BM=128, BN=128, BK=64, num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    assert same_bytes(mine, c)


# -- what the build still refuses ------------------------------------------------------------------------------------------------------------


REAL_COMPILE = triton.compile


def lying_compile(change):
    """``triton.compile`` that returns the real result with its metadata changed by ``change``."""

    def compile_(*a, **k):
        compiled = REAL_COMPILE(*a, **k)
        compiled.metadata = change(compiled.metadata)
        return compiled

    return compile_


def test_the_build_refuses_warps_triton_changed_without_warp_specialisation_and_partitions_that_add_none(monkeypatch):
    import tma_kernels as tk

    cap = CAP[0] * 10 + CAP[1]
    (plain,) = compile_specs(tk.tile_copy)
    monkeypatch.setattr(triton, "compile", lying_compile(lambda md: md._replace(num_warps=8)))
    with pytest.raises(TilefishBuildError, match=r"the compile option num_warps=4 was not applied \(Triton used 8\)"):
        build(plain, arch=cap, cache=False)
    (ws,) = compile_specs(wk.ws_scale)
    monkeypatch.setattr(triton, "compile", lying_compile(lambda md: md._replace(num_warps=4)))
    with pytest.raises(TilefishBuildError, match="add warps to the 4 of its default partition"):
        build(ws, arch=cap, cache=False)
    monkeypatch.setattr(triton, "compile", REAL_COMPILE)
    built = build(ws, arch=cap, cache=False)
    assert (built.num_warps, built.warp_specialized, built.block) == (8, True, (256, 1, 1))


# -- plans ----------------------------------------------------------------------------------------------------------------------------------


def graphs():
    return [wk.ws_chain] + ([wk.ws_mm_chain] if wk.tcgen05(CAP) else [])


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    out = {}
    for graph in graphs():
        directory = tmp_path_factory.mktemp(graph.name)
        out[graph.name] = (directory, runtime.export(graph, directory))
    return out


def chain_args(graph, rows, seed):
    if graph is wk.ws_chain:
        return (randn((rows, 256), F16, seed=seed),)
    return (randn((rows, 192), F16, seed=seed, scale=0.5), randn((128, 192), F16, seed=seed + 1, scale=0.5))


def expected(graph, args):
    if graph is wk.ws_chain:
        return wk.scaled(args[0])
    return None  # the matmul's bytes are the jit's (its rounding is checked by the single launches)


CASES = [("ws_chain", rows) for rows in (64, 192)] + [("ws_mm_chain", rows) for rows in (128, 256)]


@pytest.mark.parametrize("name, rows", CASES)
def test_a_plan_with_steps_of_other_blocks_replays_and_captures_byte_for_byte(plans, name, rows):
    if name not in plans:
        pytest.skip("tcgen05 (sm_100, sm_103) only")
    graph = getattr(wk, name)
    directory, p = plans[name]
    kw = {"mode": "replay", "plan": directory, "pin": p.digest}
    args = chain_args(graph, rows, seed=rows)
    jit = tilefish.run(graph, *args)
    eager = tilefish.run(graph, *args, **kw)
    assert same_bytes(jit, eager)
    want = expected(graph, args)
    if want is not None:
        assert same_bytes(eager, want)
    out = torch.empty_like(eager)
    cap = tilefish.capture(graph, *args, outputs=out, **kw)
    cap.replay()
    torch.cuda.synchronize()
    assert same_bytes(out, eager)
    fresh = chain_args(graph, rows, seed=rows + 1)
    for held, new in zip(args, fresh):
        held.copy_(new)  # new values at the addresses the graph holds
    cap.replay()
    torch.cuda.synchronize()
    assert same_bytes(out, tilefish.run(graph, *args, **kw))
    runtime.prepare(graph, **kw)
    mine = torch.empty_like(eager)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        tilefish.run(graph, *args, outputs=mine, **kw)
    g.replay()
    torch.cuda.synchronize()
    assert same_bytes(mine, out)


def test_the_plans_steps_take_the_blocks_of_their_binaries(plans):
    directory, p = plans["ws_chain"]
    manifest = tilefish.Manifest.load(directory, pin=tilefish.Manifest.load(directory).digest)
    blocks = sorted({tuple(a.entry["block"]) for a in manifest.artifacts if a.entry["kind"] == "triton"})
    assert (128, 1, 1) in blocks and (256, 1, 1) in blocks  # the plain copy and the warp-specialised scaling, in one plan
