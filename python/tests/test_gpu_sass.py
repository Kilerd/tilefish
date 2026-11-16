"""SASS traceability on a real GPU and a real Triton: traced exports, ``tilefish-explain`` and the declared SASS assertions (``sass_kernels.py``).

On the device's architecture (sm_103 is where it was written; sm_90, sm_100 and sm_120 run the same tests):

* a traced export (``trace=True``) of a DSL kernel, a graph's fused region, a Triton extern and a Gluon extern writes every level and the same
  manifest and plan as an export without a trace;
* ``tilefish-explain`` maps the MMAs of ``tl.dot`` (``tcgen05.mma`` on sm_100 and sm_103) and of Gluon's ``tcgen05_mma`` to the source line that
  wrote them, and the multiply of a DSL kernel to the op and the line of its body;
* each assertion builds a kernel that meets it and refuses one that does not, naming the instructions and their lines; a cache hit is judged again;
* where no nvdisasm decodes the architecture (sm_103 with Triton's bundled CUDA 12.8 tools) the SASS is read from the PTX assembled for the base of
  its family, and said so; with ``TILEFISH_NVDISASM_PATH`` naming CUDA 13's, the binary itself.
"""

import pathlib
import re

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import mm_kernels  # noqa: E402
import sass_kernels as sk  # noqa: E402
from conftest import require_ptxas_for  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, TilefishBuildError, _fpcontract, _sass, bf16, explain, f32, runtime  # noqa: E402
from tilefish.aot import build, compile_specs  # noqa: E402

CAP = torch.cuda.get_device_capability()
CC = CAP[0] * 10 + CAP[1]
TCGEN05 = CC in (100, 103)
HERE = pathlib.Path(__file__).resolve().parent


def line_of(path, text, start=0):
    lines = path.read_text().splitlines()
    return 1 + next(i for i, line in enumerate(lines) if i >= start and text in line)


MM_DOT = line_of(HERE / "mm_kernels.py", "acc = tl.dot(a, b, acc)")
N = Dim("N", divisible_by=16)
R = Dim("R", buckets=[64])


@tilefish.kernel
def gs_scale(x: Tensor[N, f32]) -> Tensor[N, f32]:
    return x * 3.0  # the multiply the trace maps back here


@tilefish.graph
def gs_graph(x: Tensor[R, 256, bf16], w: Tensor[R, 256, bf16]) -> Tensor[R, 256, bf16]:
    h = x.to(f32) * w.to(f32)
    return (h + 1.0).to(bf16)


@pytest.fixture(autouse=True)
def _fresh_disassemblies(monkeypatch):
    monkeypatch.setattr(_sass, "_disassemblies", {})
    monkeypatch.setattr(_fpcontract, "_decodes", {})


def mma_rows(e):
    return [r for r in e.lines() if r["columns"].get("mma")]


# -- traced exports and explain ---------------------------------------------------------------------------------------------------------------


def test_a_traced_export_of_a_triton_extern_maps_its_mmas_to_the_dot(tmp_path):
    require_ptxas_for(CC)
    plain = runtime.export(mm_kernels.mm, tmp_path / "plain")
    traced = runtime.export(mm_kernels.mm, tmp_path / "traced", trace=True)
    assert plain.digest == traced.digest
    assert (plain.directory / "manifest.json").read_bytes() == (traced.directory / "manifest.json").read_bytes()
    base = traced.directory / "trace" / f"mm.sm_{CC}"
    for name in ("frontend.mlir", "ttir.mlir", "ttgir.mlir", "llir.ll", "ptx.ptx", "sass.txt", "lines.json", "source/mm_kernels.py"):
        assert (base / name).is_file(), name
    e = explain.load(traced.directory, "mm")
    assert e.traced
    rows = mma_rows(e)
    total = sum(r["columns"]["mma"] for r in rows)
    on_dot = sum(r["columns"]["mma"] for r in rows if (r["file"], r["line"]) == ("mm_kernels.py", MM_DOT))
    assert total > 0 and (on_dot == total if CC != 90 else on_dot > total // 2), rows  # (sm_90: a wgmma may sit under the store that waits for it)
    s = e.summary()
    if TCGEN05:
        assert {m["sass"] for m in s["mma"]} == {"UTCHMMA"} and all(m["descriptor"]["kind"] == "f16" for m in s["mma"])
        assert {p["ptx"] for p in s["ptx_mma"]} == {"tcgen05.mma.cta_group::1.kind::f16"}
    assert s["tf32_mma"] == 0 and s["registers"] > 0
    text = e.annotate()
    assert f"mm_kernels.py:{MM_DOT}  acc = tl.dot(a, b, acc)" in text


def test_a_traced_export_of_a_gluon_extern(tmp_path):
    require_ptxas_for(CC)
    if TCGEN05:
        import ws_kernels

        line = line_of(HERE / "ws_kernels.py", "tcgen05_mma(a_bufs.index(s)")
        out = runtime.export(ws_kernels.ws_matmul, tmp_path / "ws", trace=True)
        e = explain.load(out.directory, "ws_matmul")
        rows = mma_rows(e)
        assert rows and {(r["file"], r["line"]) for r in rows} == {("ws_kernels.py", line)}, rows
        assert {p["ptx"] for p in e.summary()["ptx_mma"]} == {"tcgen05.mma.cta_group::1.kind::f16"}
        assert not (out.directory / "trace" / f"ws_matmul.sm_{CC}" / "ttir.mlir").exists()  # a Gluon kernel's frontend builds TTGIR
    else:
        import gluon_kernels

        out = runtime.export(gluon_kernels.row_sum4, tmp_path / "rs", trace=True)
        e = explain.load(out.directory, "row_sum4")
        assert any(r["file"] == "gluon_kernels.py" for r in e.lines())
    assert (out.directory / "trace" / "index.json").is_file()


def test_a_traced_export_of_a_dsl_kernel_and_of_a_fused_region(tmp_path):
    require_ptxas_for(CC)
    out = runtime.export(gs_scale, tmp_path / "k", trace=True)
    e = explain.load(out.directory, "gs_scale")
    here = line_of(HERE / "test_gpu_sass.py", "return x * 3.0  # the multiply")
    mul = [r for r in e.lines() if (r.get("written_for") or {}).get("op") == "mul"]
    assert mul and mul[0]["written_for"]["loc"] == f"test_gpu_sass.py:{here}"
    assert any(r["columns"].get("alu") for r in mul)  # FMUL
    plan = runtime.export(gs_graph, tmp_path / "g", trace=True)
    assert plan.digest == runtime.export(gs_graph, tmp_path / "g_plain").digest
    text = explain.explain(tmp_path / "g", graph="gs_graph", step=0)
    assert "step 0 of the plan of 'gs_graph'" in text and "fused region 0 of the ops #0 cast (test_gpu_sass.py:" in text
    assert re.search(r"\[node \d+ mul \(helper _tf_mul_rn_f32\); graph op #2 mul at test_gpu_sass\.py:\d+: h = x\.to\(f32\) \* w\.to\(f32\)\]", text), text
    assert re.search(r"FMUL R\d+, R\d+, R\d+ ;", text)


def test_the_sass_of_sm_103_is_read_from_the_family_base_where_no_nvdisasm_decodes_it(tmp_path, monkeypatch):
    if CC != 103:
        pytest.skip("the family base is sm_103's question")
    require_ptxas_for(CC)
    bundled = str(triton.knobs.nvidia.nvdisasm.path)
    monkeypatch.setattr(_fpcontract, "nvdisasm_candidates", lambda: [bundled])  # Triton 3.5.1's: CUDA 12.8, which cannot decode sm_103
    out = runtime.export(mm_kernels.mm, tmp_path / "proxy", trace=True)
    e = explain.load(out.directory, "mm")
    assert e.sass_label.startswith("the SASS of its PTX assembled again for sm_100a, the base of its family, read by nvdisasm 12.8")
    assert mma_rows(e) and {(r["file"], r["line"]) for r in mma_rows(e)} == {("mm_kernels.py", MM_DOT)}


# -- the assertions ---------------------------------------------------------------------------------------------------------------------------


def built(op, **options):
    (spec,) = compile_specs(op)
    return build(spec, arch=CC, fn=spec_fn(spec), options=options or None)


def spec_fn(spec):
    from tilefish._build import resolve_triton

    return resolve_triton(spec.triton)


def test_a_kernel_that_meets_its_assertions_is_built():
    require_ptxas_for(CC)
    assert built(sk.mm_with(sk.MM_CHECKS)).symbol == "mm_kernel"
    assert built(sk.spill_with(tilefish.sass(no_tf32=True, max_registers=255))).symbol == "spill_kernel"
    assert built(sk.count_with(tilefish.sass(no_spills=True, forbid="fp64"))).symbol == "count_kernel"


@pytest.mark.parametrize(
    "make,checks,message",
    [
        (sk.mm_tf32_with, tilefish.sass(no_tf32=True), r"no_tf32: TF32 tensor-core MMAs.*\(\d+ in its PTX.*PTX line \d+: \S*tf32\S* \(sass_kernels\.py:\d+\)"),
        (sk.mm_with, tilefish.sass(mma="tcgen05.mma kind::f8f6f4", by_arch={"sm_120": tilefish.sass(mma="HMMA.16816.F32.BF16"),
                                                                           "sm_90": tilefish.sass(mma="HGMMA.F32.BF16")}),
         r"mma: it declares the (PTX|SASS) MMA"),
        (sk.spill_with, tilefish.sass(no_spills=True), r"no_spills: \d+ local-memory instructions.*/\*[0-9a-f]{4}\*/ (STL|LDL)\S* .*\(sass_kernels\.py:\d+\)"),
        (sk.mm_with, tilefish.sass(max_registers=16), r"max_registers: the binary uses \d+ registers per thread, more than the 16 declared"),
        (sk.count_with, tilefish.sass(forbid="atomic"), r"forbid: \d+ instructions? of the class 'atomic'.*(RED|ATOM)\S* .*\(sass_kernels\.py:\d+\)"),
    ],
)  # fmt: skip
def test_a_kernel_that_violates_an_assertion_is_refused_with_its_instructions(make, checks, message):
    require_ptxas_for(CC)
    with pytest.raises(TilefishBuildError, match=message) as e:
        built(make(checks, name=f"refused_{make.__name__}"))
    assert f"for sm_{CC} violates the SASS assertions of its declaration" in str(e.value)


def test_a_cache_hit_is_judged_again(monkeypatch):
    require_ptxas_for(CC)
    built(sk.mm_with(tilefish.sass(max_registers=255), name="mm_hit"))  # compiled and stored
    monkeypatch.setattr(triton, "compile", lambda *a, **k: pytest.fail("a cache hit compiles nothing"))
    with pytest.raises(TilefishBuildError, match="more than the 16 declared"):
        built(sk.mm_with(tilefish.sass(max_registers=16), name="mm_hit"))  # the same IR: served by the cache, and refused
    assert built(sk.mm_with(tilefish.sass(max_registers=255), name="mm_hit")).symbol == "mm_kernel"


def test_the_registers_the_elf_file_records_are_the_driver_s():
    require_ptxas_for(CC)
    from cuda.bindings import driver as drv

    from tilefish import _cuda

    torch.cuda.init()
    for op in (sk.mm_with(None, name="mm_regs"), sk.spill_with(None, name="spill_regs")):
        b = built(op)
        loaded = _cuda.load(b, torch.cuda.current_device())
        err, regs = drv.cuFuncGetAttribute(drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, loaded.function)
        err2, local = drv.cuFuncGetAttribute(drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, loaded.function)
        attrs = _sass.kernel_attributes(b.cubin, b.symbol)
        assert err == drv.CUresult.CUDA_SUCCESS and err2 == drv.CUresult.CUDA_SUCCESS
        assert regs == attrs["registers"] and local == attrs.get("frame_size", 0), (b.stem, regs, local, attrs)
