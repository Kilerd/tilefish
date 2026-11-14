"""TMA tensor descriptors (``tilefish.TensorDesc``) on a real GPU: the kernels of ``tma_kernels.py`` (Triton and Gluon), sm_90 or newer.

A ``TensorDesc`` parameter takes a tensor; every launch encodes a ``CUtensorMap`` of it (``cuTensorMapEncodeTiled``: its address, shape and strides,
the declared box, swizzle, L2 promotion and fill) and passes it by value, then the tensor's shape and strides, as Triton's own launcher passes a host
``TensorDescriptor``. Held here:

* ``jit`` and ``replay`` give the same bytes, and those are the bytes of Triton's own launch of the same kernel with ``TensorDescriptor`` arguments, for a
  tiled copy (edge tiles past the tensor), a negative row coordinate (TMA's zero fill), a block wider than its box, a ``tl.dot`` matmul whose operands
  are swizzled tiles, a descriptor of a view (a column block of a wider tensor), and a Gluon kernel;
* the manifest records each descriptor (``kind`` ``tensordesc``, its ``tensor_map``) and the build refuses a declared swizzle the compiled kernel does not
  expect;
* a launch refuses, before anything runs, a tensor whose address is not 16-byte aligned (``pointer_alignment``) and one whose rows are not a multiple of
  16 bytes (``tensor_map``); a declaration refuses a box the driver refuses;
* a plan encodes the descriptors of every step from the buffers it binds (arena buffers, the caller's outputs, a view of the caller's input): its
  replay, its jit, a ``tilefish.capture`` replay and a ``torch.cuda.graph`` replay give the same bytes, and a graph captured with other outputs
  writes those (a captured descriptor is of the addresses the graph holds).
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("triton.experimental.gluon")
pytest.importorskip("cuda.bindings.driver")

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:  # pragma: no cover - TMA needs sm_90
    pytest.skip("TMA descriptors need sm_90 or newer", allow_module_level=True)

import tma_kernels as tk  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Out, Tensor, TensorDesc, TilefishBuildError, TilefishDeclarationError, TilefishPreconditionError, const, f16, runtime  # noqa: E402
from tilefish.aot import build, compile_specs  # noqa: E402

F16, BF16 = torch.float16, torch.bfloat16
CAP = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
ARCH = f"sm_{CAP[0] * 10 + CAP[1]}"
OPS = list(tk.ops().values())


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tma")
    manifest = runtime.export(OPS, directory)
    return directory, manifest


def run_both(op, exported, *args):
    directory, manifest = exported
    jit = tilefish.run(op, *args)
    replay = tilefish.run(op, *args, mode="replay", manifest=directory, pin=manifest.digest)
    jits, replays = (jit, replay) if isinstance(jit, tuple) else ((jit,), (replay,))
    for a, b in zip(jits, replays):
        assert same_bytes(a, b), f"{op.name}: jit and replay differ"
    return replay


def triton_launch(kernel, grid, descs, *rest, **meta):
    """Triton's own launch of the kernel, with ``TensorDescriptor`` arguments it encodes itself."""
    kernel[grid](*descs, *rest, **meta)
    torch.cuda.synchronize()


def host_desc(t, block):
    from triton.tools.tensor_descriptor import TensorDescriptor

    return TensorDescriptor(t, list(t.shape), list(t.stride()), list(block))


def gluon_desc(t, block):
    from triton.experimental.gluon.language._layouts import NVMMASharedLayout
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    layout = NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    return TensorDescriptor(t, list(t.shape), list(t.stride()), list(block), layout)


@pytest.mark.parametrize("shape", [(64, 64), (128, 192), (100, 72), (7, 8), (300, 520)])
def test_a_tiled_copy_is_the_tensor_and_triton_launches_the_same_bytes(exported, shape):
    x = randn(shape, F16, seed=shape[0])
    y = run_both(tk.tile_copy, exported, x)
    assert same_bytes(y, x)
    mine = torch.full_like(x, 7.0)
    triton_launch(tk.tile_copy_kernel, (-(-shape[0] // 64), -(-shape[1] // 64)), (host_desc(x, (64, 64)), host_desc(mine, (64, 64))), BM=64, BN=64)
    assert same_bytes(mine, y)


@pytest.mark.parametrize("rows", [1, 3, 64, 65, 200])
def test_a_negative_row_coordinate_reads_zeros(exported, rows):
    x = randn((rows, 128), F16, seed=rows)
    y = run_both(tk.shift_rows, exported, x)
    assert same_bytes(y, tk._shifted(x))


@pytest.mark.parametrize("shape", [(32, 256), (96, 512), (33, 264)])
def test_a_block_wider_than_its_box_is_copied_as_several_boxes(exported, shape):
    directory, manifest = exported
    (spec,) = compile_specs(tk.wide_copy)
    (x_param,) = [p for p in manifest.get(spec.stem, ARCH).entry["abi"]["params"] if p["name"] == "x"]
    assert x_param["tensor_map"]["block"] == [32, 256] and x_param["tensor_map"]["box"] == [32, 64]
    x = randn(shape, BF16, seed=shape[1])
    assert same_bytes(run_both(tk.wide_copy, exported, x), x)


@pytest.mark.parametrize("mnk", [(128, 128, 64), (256, 384, 192), (200, 136, 72), (64, 64, 512)])
def test_a_matmul_of_swizzled_tiles_is_tritons_bytes(exported, mnk):
    m, n, k = mnk
    a, b = randn((m, k), F16, seed=m), randn((n, k), F16, seed=n + 1)
    c = run_both(tk.tma_matmul, exported, a, b)
    want = (a.double() @ b.double().T).to(F16)
    assert torch.allclose(c.float(), want.float(), rtol=2e-2, atol=2e-2)
    mine = torch.empty((m, n), dtype=F16, device="cuda")
    triton_launch(
        tk.tma_matmul_kernel,
        (-(-m // 128), -(-n // 128)),
        (host_desc(a, (128, 64)), host_desc(b, (128, 64)), host_desc(mine, (128, 128))),
        k,
        BM=128,
        BN=128,
        BK=64,
        num_warps=4,
        num_stages=3,
    )
    assert same_bytes(mine, c)


@pytest.mark.parametrize("first", [0, 8, 64, 136])
def test_a_descriptor_of_a_view_reads_the_view(exported, first):
    wide = randn((96, 512), F16, seed=first)
    x = wide[:, first : first + 192]  # a column block: row stride 512 elements, a start on 16 bytes
    assert x.stride() == (512, 1)
    y = run_both(tk.tile_copy_view, exported, x)
    assert same_bytes(y, x.contiguous())


@pytest.mark.parametrize("shape", [(64, 64), (130, 200)])
def test_a_gluon_kernel_copies_through_its_descriptors(exported, shape):
    directory, manifest = exported
    (spec,) = compile_specs(tk.gluon_tile_copy)
    assert manifest.get(spec.stem, ARCH).entry["source_digest_kind"] == "tilefish.gluon-ir.v1"
    x = randn(shape, F16, seed=shape[1])
    y = run_both(tk.gluon_tile_copy, exported, x)
    assert same_bytes(y, x)
    mine = torch.zeros_like(x)
    triton_launch(tk.gluon_tile_copy_kernel, (-(-shape[0] // 64), -(-shape[1] // 64)), (gluon_desc(x, (64, 64)), gluon_desc(mine, (64, 64))), BM=64, BN=64, num_warps=4)
    assert same_bytes(mine, y)


def test_the_manifest_records_each_descriptor(exported):
    directory, manifest = exported
    (spec,) = compile_specs(tk.tma_matmul)
    params = manifest.get(spec.stem, ARCH).entry["abi"]["params"]
    assert [(p["name"], p["kind"], p["slot"]) for p in params] == [("a", "tensordesc", 0), ("b", "tensordesc", 5), ("c", "tensordesc", 10), ("k_len", "scalar", 15)]
    a = params[0]
    assert (a["type"], a["ptx_type"], a["size"]) == ("tensordesc", "b8[128]", 128)
    assert a["tensor_map"] == {
        "data_type": "CU_TENSOR_MAP_DATA_TYPE_FLOAT16",
        "block": [128, 64],
        "box": [128, 64],
        "element_strides": [1, 1],
        "interleave": "CU_TENSOR_MAP_INTERLEAVE_NONE",
        "swizzle": "CU_TENSOR_MAP_SWIZZLE_128B",
        "l2_promotion": "CU_TENSOR_MAP_L2_PROMOTION_L2_128B",
        "oob_fill": "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE",
    }
    assert params[2]["tensor_map"]["block"] == [128, 128] and params[2]["tensor_map"]["box"] == [128, 64]


def test_a_misaligned_tensor_and_rows_that_are_not_16_bytes_are_refused_before_anything_runs(exported):
    directory, manifest = exported
    flat = randn((64 * 64 + 8,), F16, seed=3)
    for mode in ("jit", "replay"):
        kw = {} if mode == "jit" else {"mode": "replay", "manifest": directory, "pin": manifest.digest}
        with pytest.raises(TilefishPreconditionError) as e:
            tilefish.run(tk.tile_copy, flat[1 : 1 + 64 * 64].view(64, 64), **kw)  # 2 bytes past a 16-byte boundary
        assert e.value.kind == "pointer_alignment"
        with pytest.raises(TilefishPreconditionError) as e:
            tilefish.run(tk.tile_copy, randn((64, 36), F16, seed=4), **kw)  # rows of 72 bytes
        assert e.value.kind == "tensor_map" and "multiples of 16 bytes" in str(e.value)
        wide = randn((64, 260), F16, seed=5)
        with pytest.raises(TilefishPreconditionError) as e:
            tilefish.run(tk.tile_copy_view, wide[:, :64], **kw)  # a view with a row stride of 520 bytes
        assert e.value.kind == "tensor_map"
        with pytest.raises(TilefishPreconditionError) as e:
            tilefish.run(tk.tile_copy_view, wide.reshape(-1)[4 : 4 + 64 * 64].view(64, 64), **kw)  # a view 8 bytes off a 16-byte boundary
        assert e.value.kind == "pointer_alignment"


def test_a_box_the_driver_refuses_is_refused_where_it_is_declared():
    R = Dim("R")
    for block, why in (((512, 64), "1 to 256"), ((64, 4), "multiple of 16"), ((64, 48), "power of two")):
        with pytest.raises(TilefishDeclarationError, match=why):
            TensorDesc(Tensor[R, 64, f16], block=block, swizzle=0)
    assert TensorDesc(Tensor[R, 64, f16], block=(64, 64), swizzle=64).tensor_map.box == (64, 32)  # rows of 128 bytes in a 64-byte span: two boxes
    with pytest.raises(TilefishDeclarationError, match="float NaN"):
        TensorDesc(Tensor[R, 64, tilefish.i32], block=(64, 64), oob_fill="nan")


def test_a_swizzle_the_compiled_kernel_does_not_expect_is_refused_at_the_build():
    R, C = Dim("R"), Dim("C")

    @tilefish.extern(triton=tk.tile_copy_kernel, num_warps=4, grid=lambda R, C: (tilefish.cdiv(R, 64), tilefish.cdiv(C, 64)), name="tile_copy_sw64")
    def tile_copy_sw64(
        x: TensorDesc(Tensor[R, C, f16], block=(64, 64), swizzle=64),
        y: Out[TensorDesc(Tensor[R, C, f16], block=(64, 64), swizzle=64)],
        BM: const = 64,
        BN: const = 64,
    ): ...

    assert tile_copy_sw64.signature.params[0].tensor_map.box == (64, 32)
    (spec,) = compile_specs(tile_copy_sw64)
    with pytest.raises(TilefishBuildError, match="declare swizzle=128"):
        build(spec, arch=CAP[0] * 10 + CAP[1], cache=False)


# -- plans ----------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    out = {}
    for graph in (tk.tma_chain, tk.tma_view):
        directory = tmp_path_factory.mktemp(graph.name)
        out[graph.name] = (directory, runtime.export(graph, directory))
    return out


def expected_chain(x):
    return tk._shifted((x.float() * 2.0).to(F16))


@pytest.mark.parametrize("rows", [64, 192])
def test_a_plan_encodes_the_descriptors_of_its_buffers(plans, rows):
    directory, p = plans["tma_chain"]
    x = randn((rows, 256), F16, seed=rows)
    jit = tilefish.run(tk.tma_chain, x)
    replay = tilefish.run(tk.tma_chain, x, mode="replay", plan=directory, pin=p.digest)
    assert same_bytes(jit, replay) and same_bytes(replay, expected_chain(x))
    directory2, p2 = plans["tma_view"]
    wide = randn((rows, 512), F16, seed=rows + 1)
    got = tilefish.run(tk.tma_view, wide, mode="replay", plan=directory2, pin=p2.digest)
    assert same_bytes(got, wide[:, 256:].contiguous()) and same_bytes(got, tilefish.run(tk.tma_view, wide))


@pytest.mark.parametrize("rows", [64, 192])
def test_a_captured_plan_replays_its_descriptors_byte_for_byte(plans, rows):
    directory, p = plans["tma_chain"]
    kw = {"mode": "replay", "plan": directory, "pin": p.digest}
    x = randn((rows, 256), F16, seed=rows + 7)
    eager = tilefish.run(tk.tma_chain, x, **kw)
    out = torch.empty_like(x)
    cap = tilefish.capture(tk.tma_chain, x, outputs=out, **kw)
    cap.replay()
    torch.cuda.synchronize()
    assert same_bytes(out, eager)
    x.copy_(randn((rows, 256), F16, seed=rows + 8))  # new values at the addresses the graph holds
    cap.replay()
    torch.cuda.synchronize()
    assert same_bytes(out, expected_chain(x))
    # another capture with other outputs: its descriptors are of those (a graph holds the addresses it was captured with)
    other = torch.empty_like(x)
    cap2 = tilefish.capture(tk.tma_chain, x, outputs=other, **kw)
    cap2.replay()
    torch.cuda.synchronize()
    assert same_bytes(other, out)
    # a run inside the caller's torch.cuda.graph
    runtime.prepare(tk.tma_chain, **kw)
    mine = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tilefish.run(tk.tma_chain, x, outputs=mine, **kw)
    graph.replay()
    torch.cuda.synchronize()
    assert same_bytes(mine, out)


def test_one_step_alone_encodes_the_descriptors_of_the_callers_tensors(plans):
    directory, p = plans["tma_chain"]
    x = randn((64, 256), F16, seed=11)
    steps = p.steps(Rb=64)
    shift = next(i for i, s in enumerate(steps) if s["entry"] == "shift_rows")
    names = [a["binding"] for a in steps[shift]["args"]]
    src = randn((64, 256), F16, seed=12)
    dst = torch.empty_like(src)
    runtime.run_step(tk.tma_chain, shift, {names[0]: src, names[1]: dst}, mode="replay", plan=directory, pin=p.digest)
    torch.cuda.synchronize()
    assert same_bytes(dst, tk._shifted(src))
    del x
