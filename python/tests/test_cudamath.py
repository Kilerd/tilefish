"""The CUDA math guard (``tilefish._cudamath``; manifest schema 3, section 4.7; ``docs/en/numerics.md``) with no GPU: which functions a generated
source is made of, the ``cuda_math`` field and the strict reading of it (``crates/tilefish/tests/cuda_math.rs`` has the same refusals), and the
warnings of a run whose torch is built with a CUDA release the functions were not measured on. On a GPU: ``test_gpu_cuda_math.py``."""

import json
import warnings

import ieee_cases as ic
import primitive_cases as pc
import pytest
import tanh_cases as tc
from built_fakes import fake_build, scale

import tilefish
from tilefish import Dim, Tensor, TilefishCudaMathWarning, TilefishManifestError, f32
from tilefish import _cudamath as cm
from tilefish._manifest import manifest_digest
from tilefish.aot import Manifest, compile_specs, write_manifest

N = Dim("N")
LIBDEVICE = "5c2fae37c86e68c3a38605a95f512d7d12d5f3db986310be47f57304aa72a5ee"
UNARY = ("exp", "sqrt", "rsqrt", "sin", "cos", "tanh", "silu", "gelu", "erf", "sin_approx", "rcp")


def kernel_of(op, math):
    @tilefish.kernel(math=math, name=f"k_{op}_{math}")
    def k(x: Tensor[N, f32]) -> Tensor[N, f32]:
        return getattr(tilefish, op)(x)

    return k


@pytest.mark.parametrize("math", ["fast", "ieee"])
@pytest.mark.parametrize("op", UNARY)
def test_what_a_generated_source_is_made_of_is_what_its_ops_are(op, math):
    """The markers the build reads from the source say what the ops of the kernel are, op by op and math by math (they cannot drift apart)."""
    k = kernel_of(op, math)
    assert cm.of_source(k.lower().source) == cm.of_ops((n.op, math) for n in k.graph.nodes)


def test_the_functions_of_the_kernels_of_the_contract():
    assert cm.of_source(ic.math_ieee.lower().source) == (("expf", "rsqrtf", "sinf"), True)
    assert cm.of_source(ic.math_fast.lower().source) == (("sinf",), True)  # exp, sqrt, rsqrt and silu of the default math are Triton's own
    assert cm.of_source(tc.tanh_fast.lower().source) == (("tanhf",), True)
    assert cm.of_source(pc.erf_ieee.lower().source) == (("erff",), False)  # inline PTX: no libdevice
    assert cm.of_source(pc.erf_fast.lower().source) == ((), True)  # libdevice's erff: no CUDA function claimed
    assert cm.of_source(pc.fma3.lower().source) == ((), False)
    assert cm.of_source(ic.per_op.lower().source) == (("expf",), False)  # an op's own math: exp in IEEE math


def test_the_field_lists_the_releases_every_function_was_measured_on():
    source = ic.math_ieee.lower().source
    field = cm.entry_field(source, {"extern_libs": {"libdevice": __file__}})
    assert field["functions"] == ["expf", "rsqrtf", "sinf"]
    assert field["cuda"] == ["12.8", "12.9", "13.0", "13.1", "13.2"]
    assert len(field["libdevice"]) == 64  # the sha256 of the file the options name
    assert cm.entry_field(pc.erf_ieee.lower().source)["cuda"] == ["12.8", "12.9", "13.0", "13.1"]  # CUDA 13.2 changed erff
    assert cm.entry_field(pc.fma3.lower().source) is None
    assert cm.verified(["erff", "expf"]) == ("12.8", "12.9", "13.0", "13.1") and cm.verified([]) == ()
    assert cm.check_field("'k' (sm_103)", field) is None


@pytest.mark.parametrize(
    "field, why",
    [
        ("13.0", "cuda_math must be an object"),
        ({"functions": ["erff"], "cuda": ["13.0"], "extra": 1}, "unknown fields ['extra']"),
        ({"cuda": ["13.0"]}, "cuda_math.functions must be a list of function names"),
        ({"functions": ["tanhf", "erff"], "cuda": []}, "cuda_math.functions must be sorted, each once"),
        ({"functions": ["erff", "erff"], "cuda": []}, "cuda_math.functions must be sorted, each once"),
        ({"functions": ["Erff"], "cuda": []}, "cuda_math.functions must be a list of function names"),
        ({"functions": ["erff"], "cuda": ["13"]}, "cuda_math.cuda must be a list of CUDA releases"),
        ({"functions": ["erff"], "cuda": ["13.01"]}, "cuda_math.cuda must be a list of CUDA releases"),
        ({"functions": ["erff"], "cuda": ["13.1", "13.0"]}, "cuda_math.cuda must be sorted by release, each once"),
        ({"functions": ["erff"], "cuda": ["12.10", "12.9"]}, "cuda_math.cuda must be sorted by release, each once"),
        ({"functions": [], "cuda": ["13.0"], "libdevice": LIBDEVICE}, "cuda_math.cuda lists releases"),
        ({"functions": ["erff"], "cuda": [], "libdevice": "ABC"}, "cuda_math.libdevice must be a sha256"),
        ({"functions": [], "cuda": []}, "cuda_math names no function and no libdevice"),
    ],
)
def test_a_malformed_field_is_refused(field, why):
    assert why in cm.check_field("'k' (sm_103)", field)


def generated_manifest(tmp_path, field):
    (spec,) = compile_specs(pc.erf_ieee.lower().extern)
    write_manifest([fake_build(spec, cuda_math=field)], tmp_path / "m" / "manifest.json", cubins=True)
    return spec, tmp_path / "m"


def test_the_manifest_entry_of_a_generated_kernel_carries_it(tmp_path):
    field = {"functions": ["erff"], "cuda": ["12.8", "12.9", "13.0", "13.1"]}
    spec, directory = generated_manifest(tmp_path, field)
    doc = json.loads((directory / "manifest.json").read_text())
    assert doc["kernels"][0]["cuda_math"] == field
    m = Manifest.load(directory)
    assert m.built_kernel(spec, "sm_90", pinned=True).cuda_math == field
    none_spec, none_dir = generated_manifest(tmp_path / "none", None)
    assert "cuda_math" not in json.loads((none_dir / "manifest.json").read_text())["kernels"][0]  # an entry without it is as before
    assert Manifest.load(none_dir).digest != m.digest


def test_the_reader_refuses_a_malformed_field_and_one_on_another_entry(tmp_path):
    spec, directory = generated_manifest(tmp_path, {"functions": ["erff"], "cuda": ["13.0"]})
    path = directory / "manifest.json"
    doc = json.loads(path.read_text())
    doc["kernels"][0]["cuda_math"] = {"functions": ["erff"], "cuda": ["13.1", "13.0"]}
    path.write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="cuda_math.cuda must be sorted by release"):
        Manifest.load(directory, pin=manifest_digest(doc))
    (ext,) = [s for s in compile_specs(scale) if s.stem == "scale"]
    write_manifest([fake_build(ext, cuda_math={"functions": ["erff"], "cuda": ["13.0"]})], tmp_path / "x" / "manifest.json", cubins=True)
    with pytest.raises(TilefishManifestError, match="cuda_math is only on the entry of a kernel tilefish generated"):
        Manifest.load(tmp_path / "x")


def test_a_run_warns_where_torch_is_built_with_another_release(monkeypatch):
    field = {"functions": ["erff"], "cuda": ["12.8", "12.9", "13.0", "13.1"], "libdevice": LIBDEVICE}
    monkeypatch.setattr(cm, "torch_release", lambda: "13.2")
    with pytest.warns(TilefishCudaMathWarning, match=r"'k' \(sm_103\): its erff is the bits of CUDA's math library on CUDA 12.8, 12.9, 13.0, 13.1, and torch here is built with CUDA 13.2"):
        cm.check("'k' (sm_103)", field)
    monkeypatch.setattr(cm, "torch_release", lambda: "13.0")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cm.check("'k' (sm_103)", field, libdevice_here=LIBDEVICE)  # a measured release and the same libdevice: nothing
        cm.check("'k' (sm_103)", {"functions": [], "cuda": [], "libdevice": LIBDEVICE})
        cm.check("'k' (sm_103)", None)
    with pytest.warns(TilefishCudaMathWarning, match="a build here would link another"):
        cm.check("'k' (sm_103)", field, libdevice_here="0" * 64)
    monkeypatch.setattr(cm, "torch_release", lambda: None)  # no torch: nothing to compare with
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cm.check("'k' (sm_103)", field)


def test_the_reference_mode_warns_for_the_ops_it_runs_with_torch(monkeypatch):
    monkeypatch.setattr(cm, "torch_release", lambda: "13.2")
    with pytest.warns(TilefishCudaMathWarning, match="erff"):
        cm.check_graph("kernel 'erf_ieee'", pc.erf_ieee.graph, "ieee")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cm.check_graph("kernel 'tanh_ieee'", tc.tanh_ieee.graph, "ieee")  # tanhf was measured on 13.2
        cm.check_graph("kernel 'erf_fast'", pc.erf_fast.graph, "fast")  # libdevice's erf claims no release
