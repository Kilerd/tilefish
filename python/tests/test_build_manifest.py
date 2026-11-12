"""``aot.build`` without a compiler, the ``BuiltKernel`` and the manifest.

Nothing here needs triton or a GPU: a ``BuiltKernel`` is made by hand around a small ELF file. The real
build, with Triton, is checked in ``test_gpu_build.py``.
"""

import dataclasses
import hashlib
import json
import subprocess
import sys

import pytest
from built_fakes import fake_build, rowop, scale

import tilefish
from tilefish import TilefishBuildError, TilefishRuntimeError, TilefishSpecError
from tilefish import _cuda
from tilefish._build import LaunchAttributes, _check_abi, parse_arch
from tilefish.aot import CompileSpec, build, compile_specs, manifest_dict, write_manifest


# -- arch -------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arch, cap",
    [(103, 103), ("103", 103), ("sm_103", 103), ("sm103", 103), ("sm_90", 90), ("sm_90a", 90), (80, 80), (120, 120)],
)
def test_parse_arch(arch, cap):
    assert parse_arch(arch) == cap


@pytest.mark.parametrize("arch", [True, None, 3.5, "", "sm_", "gfx90a", "sm_9x", 5, 7, 10000, "sm_1000", "-90"])
def test_parse_arch_rejects(arch):
    with pytest.raises(TilefishBuildError):
        parse_arch(arch)


# -- build() before it needs Triton -----------------------------------------------------------------------------------------


def test_build_needs_a_verified_spec_and_says_so_first():
    (spec,) = compile_specs(scale)
    edited = CompileSpec.from_dict(spec.to_dict())  # read back from JSON: not verified
    with pytest.raises(TilefishSpecError, match="not been verified"):
        build(edited, arch="sm_90")


def test_build_without_triton_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)  # makes `import triton` fail
    (spec,) = compile_specs(scale)
    with pytest.raises(TilefishBuildError, match=r"tilefish\[gpu\]"):
        build(spec, arch="sm_90")


def test_build_checks_the_arch_before_anything_else():
    (spec,) = compile_specs(scale)
    with pytest.raises(TilefishBuildError, match="arch"):
        build(spec, arch="gfx90a")


def test_importing_tilefish_imports_neither_torch_nor_triton_nor_cuda():
    code = (
        "import sys, tilefish\n"
        "import tilefish.aot, tilefish.runtime\n"
        "bad = [m for m in ('torch', 'triton', 'cuda', 'numpy') if m in sys.modules]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# -- the ABI promised by a spec versus the PTX it is built into --------------------------------------------------------------

PTX_OK = """
.visible .entry tf_k(
\t.param .u64 .ptr .global .align 1 tf_k_param_0,
\t.param .u64 .ptr .global .align 1 tf_k_param_1,
\t.param .u32 tf_k_param_2,
\t.param .u64 .ptr .global .align 1 tf_k_param_3,
\t.param .u64 .ptr .global .align 1 tf_k_param_4
)
.reqntid 128
{
"""


def test_check_abi_accepts_the_promised_parameters():
    (spec,) = compile_specs(scale)
    _check_abi(spec, PTX_OK, "tf_k")


@pytest.mark.parametrize(
    "ptx, message",
    [
        (PTX_OK.replace("\t.param .u64 .ptr .global .align 1 tf_k_param_4\n", "").replace("tf_k_param_3,", "tf_k_param_3"), "does not have the ABI"),
        (PTX_OK.replace(".param .u32 tf_k_param_2", ".param .u64 tf_k_param_2"), "does not have the ABI"),
        (PTX_OK.replace("tf_k(", "other("), "no entry point 'tf_k'"),
        (PTX_OK.replace("\t.param .u32 tf_k_param_2,\n", ""), "does not have the ABI"),
    ],
)
def test_check_abi_rejects_another_signature(ptx, message):
    (spec,) = compile_specs(scale)
    with pytest.raises(TilefishBuildError, match=message):
        _check_abi(spec, ptx, "tf_k")


# -- the BuiltKernel and its manifest ------------------------------------------------------------------------------------------


def test_abi_lists_the_launch_parameters_then_the_scratch_slots():
    (spec,) = compile_specs(scale)
    abi = fake_build(spec).abi()
    assert [(p["slot"], p["name"], p["type"], p["ptx_type"], p["size"]) for p in abi["params"]] == [
        (0, "x", "*bf16", "u64", 8),
        (1, "y", "*bf16", "u64", 8),
        (2, "n_elem", "i32", "u32", 4),
    ]
    assert abi["params"][0]["access"] == "in" and abi["params"][1]["access"] == "out"
    assert abi["params"][2]["dim"] == "n"
    assert abi["scratch"] == [
        {"slot": 3, "name": "global_scratch", "type": "u64", "ptx_type": "u64", "size": 8, "value": 0},
        {"slot": 4, "name": "profile_scratch", "type": "u64", "ptx_type": "u64", "size": 8, "value": 0},
    ]
    # the constexprs are not parameters of the binary
    assert [p["name"] for p in abi["params"]] == [a.name for a in spec.args if a.kind != "constexpr"]


def test_manifest_content():
    specs = compile_specs(rowop)
    built = [fake_build(s) for s in specs]
    doc = manifest_dict(built)
    assert doc["schema"] == 1 and doc["tilefish"] == tilefish.__version__
    first, second = doc["kernels"]
    assert [first["stem"], second["stem"]] == ["rowop.S32", "rowop.S64"]
    assert first["symbol"] == "tf_k" and first["arch"] == "sm_90" and first["ptx_target"] == "sm_90a"
    assert first["cubin"] == "rowop.S32.sm_90.cubin"
    assert first["grid"] == "(M,)" and first["block"] == [64, 1, 1] and first["shared"] == 16
    assert first["buckets"] == {"S": 32} and second["buckets"] == {"S": 64}
    assert first["constexprs"] == [
        {"index": 2, "name": "EPS", "value": 1e-05, "dtype": "f32"},
        {"index": 3, "name": "BLOCK", "value": 32},
    ]
    assert second["constexprs"][1] == {"index": 3, "name": "BLOCK", "value": 64}
    assert first["sha256"] == built[0].sha256 and first["sha256_code"] == built[0].sha256_code
    assert first["sha256"] != first["sha256_code"]
    assert first["toolchain"] == {"triton": "3.5.1", "ptxas": "Cuda compilation tools, release 12.9, V12.9.86"}
    assert first["declaration_digest"] == specs[0].declaration_digest
    assert first["launch"] == {
        "global_scratch_size": 0,
        "profile_scratch_size": 0,
        "launch_pdl": False,
        "cluster_dims": [1, 1, 1],
        "num_ctas": 1,
        "cooperative": False,
    }
    kinds = [p["kind"] for p in first["preconditions"]]
    assert kinds.count("bucket") == 1 and "pointer_alignment" in kinds
    assert first["options"] == {"num_warps": 2, "num_stages": 3}


def test_manifest_is_strict_json_even_for_infinite_constexprs(tmp_path):
    (spec,) = compile_specs(scale)
    path = write_manifest([fake_build(spec)], tmp_path / "out" / "manifest.json")
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    doc = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    alpha = next(c for c in doc["kernels"][0]["constexprs"] if c["name"] == "ALPHA")
    assert alpha["value"] == {"float": "inf"}


def test_manifest_is_deterministic(tmp_path):
    (spec,) = compile_specs(scale)
    a = write_manifest([fake_build(spec)], tmp_path / "a.json").read_bytes()
    b = write_manifest([fake_build(spec)], tmp_path / "b.json").read_bytes()
    assert a == b


def test_manifest_can_write_the_cubins_it_names(tmp_path):
    built = [fake_build(s) for s in compile_specs(rowop)]
    path = write_manifest(built, tmp_path / "m.json", cubins=True)
    doc = json.loads(path.read_text())
    for entry, b in zip(doc["kernels"], built):
        assert (tmp_path / entry["cubin"]).read_bytes() == b.cubin
        assert hashlib.sha256((tmp_path / entry["cubin"]).read_bytes()).hexdigest() == entry["sha256"]
    assert sorted(p.name for p in tmp_path.glob("*.cubin")) == ["rowop.S32.sm_90.cubin", "rowop.S64.sm_90.cubin"]
    fresh = tmp_path / "fresh"
    write_manifest(built, fresh / "m.json")  # without cubins=True only the manifest is written
    assert [p.name for p in fresh.iterdir()] == ["m.json"]


def test_manifest_refuses_the_same_binary_twice(tmp_path):
    (spec,) = compile_specs(scale)
    b = fake_build(spec)
    with pytest.raises(TilefishBuildError, match="twice"):
        write_manifest([b, b], tmp_path / "m.json")
    write_manifest([b, fake_build(spec, arch="sm_100")], tmp_path / "m.json")  # another arch is another binary


def test_write_cubin(tmp_path):
    (spec,) = compile_specs(scale)
    b = fake_build(spec)
    path = b.write_cubin(tmp_path / "deep" / "dir")
    assert path.name == "scale.sm_90.cubin" and path.read_bytes() == b.cubin
    assert "scale" in repr(b) and "sm_90" in repr(b)


# -- what the replay refuses ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "launch, text",
    [
        (LaunchAttributes(64, 0, False, (1, 1, 1), 1, False), "global_scratch_size=64"),
        (LaunchAttributes(0, 8, False, (1, 1, 1), 1, False), "profile_scratch_size=8"),
        (LaunchAttributes(0, 0, True, (1, 1, 1), 1, False), "launch_pdl=True"),
        (LaunchAttributes(0, 0, False, (2, 1, 1), 2, False), "cluster_dims=(2, 1, 1)"),
        (LaunchAttributes(0, 0, False, (1, 1, 1), 2, False), "num_ctas=2"),
        (LaunchAttributes(0, 0, False, (1, 1, 1), 1, True), "launch_cooperative_grid=True"),
    ],
)
def test_replay_refuses_what_it_does_not_set_up(launch, text):
    (spec,) = compile_specs(scale)
    with pytest.raises(TilefishRuntimeError, match=text.replace("(", r"\(").replace(")", r"\)")) as e:
        _cuda.refuse_unsupported(fake_build(spec, launch=launch))
    assert "scale" in str(e.value)


def test_replay_names_every_blocker_at_once():
    (spec,) = compile_specs(scale)
    launch = LaunchAttributes(16, 8, True, (2, 1, 1), 2, True)
    assert len(launch.replay_blockers()) == 5
    with pytest.raises(TilefishRuntimeError) as e:
        _cuda.refuse_unsupported(fake_build(spec, launch=launch))
    for text in ("global_scratch_size=16", "profile_scratch_size=8", "launch_pdl", "cluster", "cooperative"):
        assert text in str(e.value)


def test_a_plain_binary_is_accepted():
    (spec,) = compile_specs(scale)
    assert LaunchAttributes(0, 0, False, (1, 1, 1), 1, False).replay_blockers() == []
    _cuda.refuse_unsupported(fake_build(spec))


def test_built_kernels_are_frozen():
    (spec,) = compile_specs(scale)
    b = fake_build(spec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.symbol = "other"
