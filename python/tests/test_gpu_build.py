"""``aot.build`` with real Triton: cubins, digests, the manifest, and reproducibility across build directories.

The central test here is reproducibility: the same compile spec built from two different directories, with
two different Triton cache directories, gives byte-different cubins (the debug information holds the path of
the Python source) with the same ``sha256_code``, the digest of the machine code.
"""

import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels  # noqa: E402
from conftest import REPO_ROOT, make_my_kernels, readme_snippet  # noqa: E402
from lower_cases import gelu_exact, layernorm, softmax, swiglu  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishBuildError, TilefishRuntimeError, _cuda, runtime  # noqa: E402
from tilefish._elf import code_digest, parse_elf  # noqa: E402
from tilefish.aot import build, compile_specs, write_manifest  # noqa: E402

CAP = torch.cuda.get_device_capability()
ARCH = f"sm_{CAP[0] * 10 + CAP[1]}"


@pytest.fixture(scope="module")
def readme_kernel():
    if not (REPO_ROOT / "README.md").exists():
        pytest.skip("README.md is not available (not running from a source checkout)")
    ns = {"my_kernels": make_my_kernels()}
    exec(compile(readme_snippet(), "README.md", "exec"), ns)
    return ns["residual_add_rmsnorm"]


Db = tilefish.Dim("Db", buckets=[96, 512])


@tilefish.kernel
def bucketed(x: tilefish.Tensor[tilefish.Dim("Mb"), Db, tilefish.f32]) -> tilefish.Tensor[tilefish.Dim("Mb"), Db, tilefish.f32]:
    return x * 2.0


@pytest.fixture(scope="module")
def lowered(readme_kernel):
    ks = {"softmax": softmax, "layernorm": layernorm, "gelu_exact": gelu_exact, "swiglu": swiglu, "residual_add_rmsnorm": readme_kernel, "bucketed": bucketed}
    # an elementwise kernel would be tried first; the row template is the one with a binary per bucket
    return {name: k.lower(template="row" if name == "bucketed" else None) for name, k in ks.items()}


def build_all(lowered_kernels, **kw):
    out = []
    for low in lowered_kernels.values():
        fn = runtime.load_lowered(low)
        out += [build(spec, arch=ARCH, fn=fn, **kw) for spec in low.specs()]
    return out


@pytest.fixture(scope="module")
def built(lowered):
    return build_all(lowered)


def test_a_build_is_a_cubin_with_everything_a_host_needs(built, lowered):
    assert len(built) == 5 + 2  # five goldens, and the bucketed kernel has one binary per bucket
    for b in built:
        assert b.arch == ARCH and b.ptx_target.startswith("sm_") and f".target {b.ptx_target}" in b.ptx
        assert b.cubin[:4] == b"\x7fELF" and parse_elf(b.cubin).section(f".text.{b.symbol}") is not None
        assert b.sha256 == hashlib.sha256(b.cubin).hexdigest()
        assert b.sha256_code == code_digest(b.cubin) and b.sha256_code != b.sha256
        assert b.symbol == f"tf_{b.spec.op}"
        assert b.num_warps == b.spec.num_warps and b.shared >= 0
        assert b.triton_version == triton.__version__
        assert "release" in b.ptxas_version and "V" in b.ptxas_version
        assert b.options == {"num_warps": b.spec.num_warps, "num_stages": b.spec.num_stages}
        assert b.launch.replay_blockers() == []  # nothing the plain driver launch would have to set up
        assert b.ptx.count(".entry") == 1
        assert [p["size"] for p in b.abi()["scratch"]] == [8, 8]


def test_the_abi_matches_the_ptx_entry_of_every_build(built):
    import re

    for b in built:
        entry = re.search(r"\.entry\s+(\w+)\s*\((.*?)\)", b.ptx, re.S)
        types = re.findall(r"\.param\s+\.(\w+)", entry.group(2))
        abi = b.abi()
        assert types == [p["ptx_type"] for p in abi["params"]] + ["u64", "u64"]


def test_building_twice_gives_the_same_bytes(lowered):
    low = lowered["layernorm"]
    fn = runtime.load_lowered(low)
    (spec,) = low.specs()
    a, b = build(spec, arch=ARCH, fn=fn), build(spec, arch=ARCH, fn=fn)
    assert a.cubin == b.cubin and a.sha256 == b.sha256 and a.sha256_code == b.sha256_code


CHILD = r"""
import importlib, json, os, sys
import conftest
import lower_cases
import tilefish
from tilefish.aot import build

arch, source_dir = sys.argv[1], sys.argv[2]
ns = {"my_kernels": conftest.make_my_kernels()}
exec(compile(conftest.readme_snippet(), "README.md", "exec"), ns)
kernels = [lower_cases.softmax, lower_cases.layernorm, lower_cases.gelu_exact, lower_cases.swiglu, ns["residual_add_rmsnorm"]]
sys.path.insert(0, source_dir)
out = {}
for k in kernels:
    low = k.lower()
    low.write(source_dir)
    fn = getattr(importlib.import_module(low.module), low.fn_name)
    for spec in low.specs():
        b = build(spec, arch=arch, fn=fn)
        out[spec.stem] = {"sha256": b.sha256, "sha256_code": b.sha256_code, "size": len(b.cubin), "cwd": os.getcwd(), "file": fn.fn.__code__.co_filename}
print(json.dumps(out))
"""


def build_in_a_fresh_place(root: pathlib.Path):
    """Build the five goldens in a child process: its own working directory, source directory and Triton cache."""
    (root / "cwd").mkdir(parents=True)
    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = str(root / "cache")
    tests = pathlib.Path(__file__).resolve().parent
    src = pathlib.Path(tilefish.__file__).resolve().parent.parent
    env["PYTHONPATH"] = os.pathsep.join([str(src), str(tests), env.get("PYTHONPATH", "")])
    r = subprocess.run(
        [sys.executable, "-c", CHILD, ARCH, str(root / "generated")],
        cwd=root / "cwd",
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_same_spec_built_from_two_directories_has_the_same_code_digest(tmp_path):
    a = build_in_a_fresh_place(tmp_path / "first")
    b = build_in_a_fresh_place(tmp_path / "second" / "a" / "much" / "deeper" / "directory")
    assert sorted(a) == sorted(b) == sorted(["softmax", "layernorm", "gelu_exact", "swiglu", "residual_add_rmsnorm"])
    for stem in a:
        assert a[stem]["cwd"] != b[stem]["cwd"] and a[stem]["file"] != b[stem]["file"]
        # the debug information holds the source path, so the files differ, and so does the digest of the whole file ...
        assert a[stem]["sha256"] != b[stem]["sha256"], stem
        assert a[stem]["size"] != b[stem]["size"], stem
        # ... while the machine code is the same
        assert a[stem]["sha256_code"] == b[stem]["sha256_code"], stem


def test_the_code_digest_follows_the_code(lowered):
    """A real change of the code changes it: the same kernel without FMA contraction, and another architecture."""
    low = lowered["layernorm"]
    fn = runtime.load_lowered(low)
    (spec,) = low.specs()
    base = build(spec, arch=ARCH, fn=fn)
    unfused = build(spec, arch=ARCH, fn=fn, options={"enable_fp_fusion": False})
    assert unfused.options["enable_fp_fusion"] is False and "enable_fp_fusion" not in base.options
    assert unfused.sha256_code != base.sha256_code
    assert base.ptx.count("fma.rn.f32") > unfused.ptx.count("fma.rn.f32")
    other = "sm_80" if ARCH != "sm_80" else "sm_90"
    elsewhere = build(spec, arch=other, fn=fn)  # building needs no GPU of that architecture
    assert elsewhere.arch == other and elsewhere.sha256_code != base.sha256_code
    assert parse_elf(elsewhere.cubin).machine == 190 and parse_elf(elsewhere.cubin).flags != parse_elf(base.cubin).flags


def test_the_digest_covers_how_ptxas_was_called_not_only_the_instructions(lowered):
    """``.note.nv.tkinfo`` records the ptxas version and command line, and it is not a debug section."""
    low = lowered["softmax"]
    fn = runtime.load_lowered(low)
    (spec,) = low.specs()
    fused = build(spec, arch=ARCH, fn=fn)
    unfused = build(spec, arch=ARCH, fn=fn, options={"enable_fp_fusion": False})  # softmax has nothing to contract
    a, b = parse_elf(fused.cubin), parse_elf(unfused.cubin)
    code = f".text.{fused.symbol}"
    assert a.data(a.section(code)) == b.data(b.section(code))  # the same instructions ...
    assert b"-fmad false" in b.data(b.section(".note.nv.tkinfo")) and b"-fmad false" not in a.data(a.section(".note.nv.tkinfo"))
    assert fused.sha256_code != unfused.sha256_code  # ... but the cubin says it was built differently


@pytest.mark.parametrize("arch", ["sm_80", "sm_90", "sm_100", "sm_120"])
def test_specs_build_for_other_architectures(lowered, arch):
    low = lowered["softmax"]
    (spec,) = low.specs()
    b = build(spec, arch=arch, fn=runtime.load_lowered(low))
    assert b.arch == arch and b.ptx_target.startswith(arch) and b.symbol == "tf_softmax"
    assert b.launch.replay_blockers() == []


def test_build_rejects_what_it_cannot_trust(lowered):
    low = lowered["softmax"]
    fn = runtime.load_lowered(low)
    (spec,) = low.specs()
    with pytest.raises(TilefishBuildError, match="does not know the compile option 'enable_fp_fusionn'"):
        build(spec, arch=ARCH, fn=fn, options={"enable_fp_fusionn": False})
    with pytest.raises(TilefishBuildError, match="not the ones its compile spec"):
        build(spec, arch=ARCH, fn=runtime.load_lowered(lowered["swiglu"]))
    with pytest.raises(TilefishBuildError, match="not a triton.jit function"):
        build(spec, arch=ARCH, fn=lambda x: x)
    with pytest.raises(TilefishBuildError, match="cannot import the module"):
        build(spec, arch=ARCH)  # tilefish_generated.softmax is not importable: the source was not written anywhere
    (broken,) = compile_specs(gpu_kernels.broken)
    with pytest.raises(TilefishBuildError, match=r"Triton could not compile 'broken' \(gpu_kernels:broken_kernel\)") as e:
        build(broken, arch=ARCH)
    assert "a_name_that_does_not_exist" in str(e.value)


def test_the_externs_resolve_their_triton_function_by_reference():
    (spec,) = compile_specs(gpu_kernels.rmsnorm)
    assert spec.triton == "gpu_kernels:rmsnorm_kernel"
    b = build(spec, arch=ARCH)  # fn=None: imported from the reference
    assert b.symbol == "rmsnorm_kernel" and b.num_warps == 8


def test_the_manifest_describes_what_was_built(built, tmp_path):
    path = write_manifest(built, tmp_path / "out" / "manifest.json", cubins=True)
    doc = json.loads(path.read_text())
    assert doc["schema"] == 1 and [k["stem"] for k in doc["kernels"]] == [b.stem for b in built]
    for entry, b in zip(doc["kernels"], built):
        cubin = (tmp_path / "out" / entry["cubin"]).read_bytes()
        assert cubin == b.cubin
        assert entry["sha256"] == hashlib.sha256(cubin).hexdigest() and entry["sha256_code"] == code_digest(cubin)
        assert entry["symbol"] == b.symbol and entry["grid"] == b.spec.grid and entry["shared"] == b.shared
        assert entry["block"] == [32 * b.num_warps, 1, 1]
        assert entry["toolchain"] == {"triton": triton.__version__, "ptxas": b.ptxas_version}
        assert entry["launch"]["global_scratch_size"] == 0
        assert [p["name"] for p in entry["abi"]["params"]] == [a.name for a in b.spec.args if a.kind != "constexpr"]
        assert [s["name"] for s in entry["abi"]["scratch"]] == ["global_scratch", "profile_scratch"]
    swiglu_entry = next(k for k in doc["kernels"] if k["stem"] == "swiglu")
    n = next(p for p in swiglu_entry["abi"]["params"] if p["kind"] == "scalar")
    assert n["count_of"] == "gate" and n["dim"] == "3072*M"  # how a host computes the element count
    assert any(p["kind"] == "max_elements" for p in swiglu_entry["preconditions"])


def test_build_for_device_caches_by_spec_arch_and_options(lowered):
    low = lowered["softmax"]
    fn = runtime.load_lowered(low)
    (spec,) = low.specs()
    runtime.clear_caches()
    first = runtime.build_for_device(spec, fn, 0)
    assert runtime.build_for_device(spec, fn, 0) is first
    assert first.arch == ARCH
    with runtime.build_options(enable_fp_fusion=False):
        other = runtime.build_for_device(spec, fn, 0)
    assert other is not first and other.options["enable_fp_fusion"] is False
    assert runtime.build_for_device(spec, fn, 0) is first


def test_the_loaded_function_is_checked_against_the_abi(lowered):
    from cuda.bindings import driver

    assert hasattr(driver, "cuFuncGetParamInfo")  # otherwise the parameter check below would be skipped
    low = lowered["softmax"]
    fn = runtime.load_lowered(low)
    (spec,) = low.specs()
    softmax_build = build(spec, arch=ARCH, fn=fn)
    torch.zeros(1, device="cuda")
    _cuda.LoadedKernel(softmax_build, 0)  # the promised parameters are the ones of the cubin
    other = compile_specs(gpu_kernels.gelu_erf)[0]  # another spec: three parameters and the scratch slots
    wrong = dataclasses.replace(softmax_build, spec=other)
    with pytest.raises(TilefishRuntimeError, match="the ABI in the manifest says"):
        _cuda.LoadedKernel(wrong, 0)


def test_replay_refuses_a_binary_that_needs_more_than_a_plain_launch(lowered):
    low = lowered["softmax"]
    (spec,) = low.specs()
    b = build(spec, arch=ARCH, fn=runtime.load_lowered(low))
    needy = dataclasses.replace(b, launch=dataclasses.replace(b.launch, global_scratch_size=128, launch_pdl=True))
    with pytest.raises(TilefishRuntimeError, match="global_scratch_size=128, launch_pdl=True"):
        _cuda.load(needy, 0)
