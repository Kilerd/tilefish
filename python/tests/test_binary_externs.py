"""Prebuilt binaries (``extern(ptx=...)``, ``extern(cubin=...)``): declarations, compile specs, builds and manifest entries, with no GPU.

A prebuilt kernel is declared by its C ABI. Its stub lists the pointers and scalars in order (a typed ``const`` is a fixed argument whose value is written
in the manifest), there are no constexprs and no Triton scratch slots, its block is declared, and its identity is the sha256 of the bytes of its PTX or
cubin (``source_digest`` of kind ``tilefish.binary.v1``). PTX is assembled by ptxas with fixed, recorded flags; a cubin is used as it is for its own
architecture. The ptxas here is a script that copies a prepared cubin: what runs on a GPU is ``test_gpu_binary.py``.
"""

import copy
import hashlib
import importlib.util
import json
import pathlib
import re
import sys

import pytest
from binary_fakes import PTX_TEXT, SM90, SM103A, fake_binary_build, fake_ptxas, make_ptx_op, write_ptx
from elf_builder import cubin_like

import tilefish
from tilefish import (
    Dim,
    Out,
    Tensor,
    TilefishBuildError,
    TilefishDeclarationError,
    TilefishManifestError,
    TilefishSpecError,
    bf16,
    const,
    f32,
    i32,
    i64,
)
from tilefish import _binary, _elf
from tilefish._build import BINARY_DIGEST_KIND, abi_of, spec_entry
from tilefish._manifest import manifest_digest
from tilefish.aot import CompileSpec, Manifest, compile_specs, declaration_digest, write_manifest

n = Dim("n", divisible_by=16)
R = Dim("R")
SHA = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


@pytest.fixture
def ptx(tmp_path):
    return write_ptx(tmp_path)


@pytest.fixture
def op(ptx):
    return make_ptx_op(ptx)


def spec_of(op_):
    (spec,) = compile_specs(op_)
    return spec


# -- declarations -----------------------------------------------------------------------------------------------------------------------


def test_a_ptx_extern_is_declared_by_its_c_abi(op, ptx):
    assert op.binary.kind == "ptx" and op.binary.symbol == "tf_k" and op.binary.block == (128, 1, 1) and op.binary.shared == 0
    assert op.binary.digests == {"ptx": SHA(ptx.read_bytes())} and op.triton == "" and op.num_warps == 0 and op.is_binary
    assert [p.name for p in op.signature.params] == ["x", "y", "n_elem", "alpha", "scratch"]
    assert op.node_attrs() == {"binary": "tf_k"}
    assert str(op).startswith("extern axpy(x: Tensor[n, f32]")


def test_a_cubin_extern_holds_the_bytes_of_every_architecture_it_names(tmp_path):
    a, b = tmp_path / "k90.cubin", tmp_path / "k103.cubin"
    a.write_bytes(cubin_like(**SM90))
    b.write_bytes(cubin_like(**SM103A))

    @tilefish.extern(cubin={"sm_90": a, "sm_103": b}, symbol="tf_k", block=(64, 2), grid=(1,), shared=1024, name="k")
    def k(x: Tensor[16, f32], y: Out[Tensor[16, f32]]): ...

    assert k.binary.kind == "cubin" and k.binary.block == (64, 2, 1) and k.binary.shared == 1024
    assert k.binary.digests == {"sm_103": SHA(b.read_bytes()), "sm_90": SHA(a.read_bytes())}


def test_a_relative_path_is_relative_to_the_file_that_declares_the_op(tmp_path):
    write_ptx(tmp_path, "rel.ptx")
    module = tmp_path / "declares_rel.py"
    module.write_text(
        "import tilefish\n"
        "from tilefish import Tensor, Out, f32\n"
        "@tilefish.extern(ptx='rel.ptx', symbol='tf_k', block=(32,), grid=(1,))\n"
        "def rel(x: Tensor[16, f32], y: Out[Tensor[16, f32]]): ...\n"
    )
    spec = importlib.util.spec_from_file_location("declares_rel", module)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["declares_rel"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        del sys.modules["declares_rel"]
    assert mod.rel.binary.digests == {"ptx": SHA((tmp_path / "rel.ptx").read_bytes())}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(triton="m:f", ptx="x.ptx", symbol="s", block=(1,)), "a kernel is one of triton=, gluon=, ptx= and cubin="),
        (dict(ptx="x.ptx", block=(1,)), "needs symbol=<the kernel's entry point> and block"),
        (dict(ptx="x.ptx", symbol="s"), "needs symbol=<the kernel's entry point> and block"),
        (dict(ptx="x.ptx", symbol="s", block=(1,), num_warps=4), "num_warps= is a Triton compile option"),
        (dict(ptx="x.ptx", symbol="s", block=(1,), num_stages=3), "num_stages= is a Triton compile option"),
        (dict(triton="m:f", symbol="s"), "symbol= is for a prebuilt binary"),
        (dict(triton="m:f", block=(32,)), "block= is for a prebuilt binary"),
        (dict(triton="m:f", shared=16), "shared= is for a prebuilt binary"),
        (dict(ptx="x.ptx", symbol="not a symbol", block=(1,)), "names the kernel's entry point"),
        (dict(ptx="x.ptx", symbol="s", block=(0,)), "must have sizes from 1 to 1024"),
        (dict(ptx="x.ptx", symbol="s", block=(32, 64)), "at most 1024 threads in all"),
        (dict(ptx="x.ptx", symbol="s", block=(2.5,)), "block must be a tuple of 1 to 3 ints"),
        (dict(ptx="x.ptx", symbol="s", block=(1, 1, 1, 1)), "block must be a tuple of 1 to 3 ints"),
        (dict(ptx="x.ptx", symbol="s", block=(1,), shared=-1), "shared must be a non-negative int"),
        (dict(ptx="no-such-file.ptx", symbol="s", block=(1,)), "cannot read ptx="),
        (dict(cubin="k.cubin", symbol="s", block=(1,)), "cubin= maps architectures to files"),
        (dict(cubin={"sm_90a": "k"}, symbol="s", block=(1,)), "keys are architectures such as 'sm_103'"),
        (dict(ptx=7, symbol="s", block=(1,)), "ptx= must be a file path"),
    ],
)
def test_a_prebuilt_declaration_is_checked(tmp_path, kwargs, message):
    kwargs = {k: (str(write_ptx(tmp_path, "x.ptx")) if v == "x.ptx" else v) for k, v in kwargs.items()}
    with pytest.raises(TilefishDeclarationError, match=message):

        @tilefish.extern(grid=(1,), **kwargs)
        def f(x: Tensor[16, f32]): ...


def test_a_missing_or_empty_file_is_refused_when_the_op_is_declared(tmp_path):
    empty = tmp_path / "empty.ptx"
    empty.write_bytes(b"")
    with pytest.raises(TilefishDeclarationError, match="is empty"):

        @tilefish.extern(ptx=empty, symbol="s", block=(1,), grid=(1,))
        def f(x: Tensor[16, f32]): ...


def test_the_const_of_a_prebuilt_binary_is_a_typed_argument_with_a_value(ptx):
    for ann, default in ((const, 1), (const[bf16], 1.0), (const[i32], 2.5), (const[f32], "x")):
        with pytest.raises(TilefishDeclarationError, match=r"is an argument of its C ABI: write `a: const\[i32\] = 0`|must default to a number|needs an int"):

            @tilefish.extern(ptx=ptx, symbol="tf_k", block=(1,), grid=(1,))
            def f(x: Tensor[16, f32], a: ann = default): ...

    with pytest.raises(TilefishDeclarationError, match="const\\[i32\\] = 0"):

        @tilefish.extern(ptx=ptx, symbol="tf_k", block=(1,), grid=(1,))
        def g(x: Tensor[16, f32], a: const[i32] = 2**31): ...  # does not fit

    with pytest.raises(TilefishDeclarationError, match="needs a default, the value written in the manifest"):

        @tilefish.extern(ptx=ptx, symbol="tf_k", block=(1,), grid=(1,))
        def h(x: Tensor[16, f32], *, a: const[i32]): ...


def test_a_triton_extern_is_not_a_prebuilt_one_and_still_has_its_warps():
    @tilefish.extern(triton="m:f", grid=(1,))
    def f(x: Tensor[16, f32]): ...

    assert f.binary is None and not f.is_binary and f.num_warps == 4 and f.num_stages == 3 and f.node_attrs() == {"triton": "m:f"}
    with pytest.raises(TilefishDeclarationError, match="num_warps must be a power of two"):
        tilefish.extern(triton="m:f", num_warps=None, grid=(1,))


# -- the compile spec -----------------------------------------------------------------------------------------------------------------------


def test_a_prebuilt_spec_has_the_c_abi_and_no_triton(op, ptx):
    spec = spec_of(op)
    assert spec.binary == {"kind": "ptx", "symbol": "tf_k", "shared": 0, "digests": {"ptx": SHA(ptx.read_bytes())}}
    assert [(a.name, a.kind, a.type) for a in spec.args] == [
        ("x", "pointer", "*fp32"),
        ("y", "pointer", "*fp32"),
        ("n_elem", "scalar", "i32"),
        ("alpha", "fixed", "f32"),
        ("scratch", "fixed", "i64"),
    ]
    assert spec.args[3].value == 2.0 and spec.args[4].value == 0 and spec.args[3].dtype == "f32"
    assert spec.signature == {"x": "*fp32", "y": "*fp32", "n_elem": "i32", "alpha": "f32", "scratch": "i64"}
    assert spec.constants == {} and spec.attrs == {}  # no constexprs, and no divisibility attributes: Triton compiles nothing
    assert spec.triton == "" and spec.num_warps == 0 and spec.num_stages == 3 - 3 + 0 and spec.block == (128, 1, 1)
    assert [p.name for p in spec.launch_params] == ["x", "y", "n_elem", "alpha", "scratch"]  # exactly the C parameters: no scratch slots
    assert spec.schema == 3 and spec.uses_strided_schema
    # the preconditions are the same promises as a kernel's: aligned pointers, a multiple of 16 for n
    assert {(p.kind, p.arg) for p in spec.preconditions} >= {("pointer_alignment", "x"), ("divisible", "n_elem"), ("max_elements", "x")}


def test_a_prebuilt_spec_round_trips_and_verifies(op):
    spec = spec_of(op)
    doc = spec.to_dict()
    assert "triton" not in doc and "num_warps" not in doc and "num_stages" not in doc and doc["binary"]["kind"] == "ptx"
    again = CompileSpec.from_json(spec.to_json())
    assert again == spec and again.binary == spec.binary and again.num_warps == 0
    again.verify_against(op)
    assert again.verified
    with pytest.raises(TilefishSpecError, match="prebuilt ptx binary: Triton does not compile it"):
        again.ast_source_inputs()


def test_a_declaration_digest_covers_the_bytes_the_entry_point_the_block_and_the_shared_memory(tmp_path, ptx):
    base = declaration_digest(make_ptx_op(ptx))
    assert base == declaration_digest(make_ptx_op(ptx))
    assert declaration_digest(make_ptx_op(write_ptx(tmp_path, "other.ptx", PTX_TEXT + "// another file\n"))) != base
    assert declaration_digest(make_ptx_op(ptx, shared=16)) != base

    @tilefish.extern(ptx=ptx, symbol="tf_k", block=(64,), grid=lambda n: (tilefish.cdiv(n, 128),), name="axpy")
    def smaller(x: Tensor[n, f32], y: Out[Tensor[n, f32]], n_elem: n, alpha: const[f32] = 2.0, scratch: const[i64] = 0): ...

    assert declaration_digest(smaller) != base


def test_a_fixed_argument_cannot_be_restated_with_another_value():
    from tilefish._launch import check_consts

    spec = spec_of(make_ptx_op(pathlib.Path(__file__).parent / "ptx" / "axpy_f32.ptx"))
    check_consts(spec, {"alpha": 2.0, "scratch": 0})
    with pytest.raises(tilefish.TilefishTypeError, match=r"argument 'alpha' of 'axpy' is fixed to 2.0 \(its value is in the manifest"):
        check_consts(spec, {"alpha": 3.0})


def test_the_abi_of_a_prebuilt_binary_has_fixed_arguments_and_no_scratch(op):
    abi = abi_of(spec_of(op))
    assert [(p["name"], p["kind"], p["type"], p["ptx_type"], p["size"]) for p in abi["params"]] == [
        ("x", "pointer", "*fp32", "u64", 8),
        ("y", "pointer", "*fp32", "u64", 8),
        ("n_elem", "scalar", "i32", "u32", 4),
        ("alpha", "fixed", "f32", "f32", 4),
        ("scratch", "fixed", "i64", "u64", 8),
    ]
    assert abi["params"][3]["value"] == 2.0 and abi["params"][4]["value"] == 0 and "dim" not in abi["params"][3]
    assert abi["scratch"] == [] and abi["scratch_slots"] == 0


# -- the strict reading of a prebuilt spec ------------------------------------------------------------------------------------------------


def doc_of(op_):
    return copy.deepcopy(spec_of(op_).to_dict())


def rejects(change, message, op_):
    doc = doc_of(op_)
    change(doc)
    with pytest.raises(TilefishSpecError, match=message):
        CompileSpec.from_dict(doc)


def test_the_binary_section_of_a_spec_is_checked(op):
    rejects(lambda d: d["binary"].update(kind="elf"), "binary kind must be 'ptx' or 'cubin'", op)
    rejects(lambda d: d["binary"].update(symbol="a b"), "binary symbol must be a kernel symbol", op)
    rejects(lambda d: d["binary"].update(shared=-1), "binary shared must be an int >= 0", op)
    rejects(lambda d: d["binary"].update(digests={}), "binary digests must be a non-empty object", op)
    rejects(lambda d: d["binary"].update(digests={"ptx": "zz"}), "must be 64 lowercase hex digits", op)
    rejects(lambda d: d["binary"].update(digests={"sm_90": "0" * 64}), "a ptx binary has one digest, keyed 'ptx'", op)
    rejects(lambda d: d["binary"].update(kind="cubin", digests={"ptx": "0" * 64}), "a cubin's digests are keyed by architecture", op)
    rejects(lambda d: d["binary"].pop("shared"), "binary is missing", op)
    rejects(lambda d: d["binary"].update(extra=1), "binary has unknown fields", op)
    rejects(lambda d: d.update(triton={"module": "m", "function": "f", "ref": "m:f"}), "unknown fields", op)
    rejects(lambda d: d.update(schema=2), "spec is missing", op)  # schema 2 has no prebuilt binary: it is a spec of a Triton kernel, and lacks its reference


def test_a_prebuilt_spec_has_no_constexprs_no_attrs_and_no_scratch_slots(op):
    rejects(lambda d: d["attrs"].update({"0": [["tt.divisibility", 16]]}), "a prebuilt binary has no Triton attributes", op)
    rejects(lambda d: d["launch_params"].append({"name": "global_scratch", "type": "u64", "value": 0}), "launch_params must be every arg in order", op)
    rejects(lambda d: d["launch_params"].pop(), "launch_params must be every arg in order", op)
    rejects(lambda d: d["block"].append(1), "block must be three sizes", op)
    rejects(lambda d: d.update(block=[32, 64, 1]), "whose product is at most 1024", op)
    rejects(lambda d: d.update(block=[0, 1, 1]), "block must be three sizes", op)
    rejects(lambda d: d["args"].append({"index": 5, "name": "C", "kind": "constexpr", "type": "constexpr", "value": 1}), "has no constexprs", op)


def test_a_fixed_argument_is_checked(op):
    rejects(lambda d: d["args"][3].update(value=1), "must be a float", op)  # an f32 takes a float
    rejects(lambda d: d["args"][4].update(value=1.5), "must be an int", op)
    rejects(lambda d: d["args"][4].update(value=2**63), "does not fit i64", op)
    rejects(lambda d: d["args"][4].update(value=True), "must be an int", op)
    rejects(lambda d: d["args"][3].update(type="u8"), "a fixed argument is one of", op)
    rejects(lambda d: d["args"][3].update(dim="n"), "must have exactly the fields", op)


# -- the ptx and the cubin ------------------------------------------------------------------------------------------------------------------


def test_the_gpu_name_of_a_ptx_follows_its_target():
    assert _binary.ptx_gpu_name(".version 8.8\n.target sm_103a\n", 103, "k") == "sm_103a"
    assert _binary.ptx_gpu_name(".version 7.0\n.target sm_80\n", 103, "k") == "sm_103"  # a generic target builds for any later architecture
    assert _binary.ptx_gpu_name(".target sm_90", 90, "k") == "sm_90"
    with pytest.raises(TilefishBuildError, match="architecture-specific target sm_103a, which only runs on sm_103; it cannot be built for sm_90"):
        _binary.ptx_gpu_name(".target sm_103a\n", 90, "k")
    with pytest.raises(TilefishBuildError, match=r"is PTX for sm_103, which is newer than the sm_90"):
        _binary.ptx_gpu_name(".target sm_103\n", 90, "k")
    with pytest.raises(TilefishBuildError, match="has no .target directive"):
        _binary.ptx_gpu_name("not ptx", 90, "k")


def test_the_entry_of_a_ptx_must_take_the_c_parameters_the_stub_declares(op):
    spec = spec_of(op)
    _binary.check_ptx_abi(spec, PTX_TEXT, "tf_k")
    with pytest.raises(TilefishBuildError, match=r"has no entry point 'other' \(it has \['tf_k'\]\)"):
        _binary.check_ptx_abi(spec, PTX_TEXT, "other")
    for bad in (PTX_TEXT.replace(".param .f32 alpha", ".param .u32 alpha"), PTX_TEXT.replace(".param .u32 n", ".param .u64 n"), PTX_TEXT.replace(",\n\t.param .u64 scratch", "")):
        with pytest.raises(TilefishBuildError, match="does not have the C ABI its stub declares"):
            _binary.check_ptx_abi(spec, bad, "tf_k")
    # the signed and the bit types carry the same values as the unsigned ones
    _binary.check_ptx_abi(spec, PTX_TEXT.replace(".param .u32 n", ".param .s32 n").replace(".param .u64 x", ".param .b64 x"), "tf_k")


def test_the_architecture_of_a_cubin_is_read_from_its_header():
    assert _binary.cubin_sm(cubin_like(**SM90)) == 90
    assert _binary.cubin_sm(cubin_like(**SM103A)) == 103
    assert _binary.cubin_sm(cubin_like(osabi=0x41, flags=0x0600780A)) == 120
    assert _binary.cubin_sm(cubin_like(osabi=0x33, flags=0x00500550)) == 80
    with pytest.raises(TilefishBuildError, match="not one of a CUDA cubin"):
        _binary.cubin_sm(cubin_like(osabi=0x00))


def test_ptx_is_assembled_by_the_configured_ptxas_with_fixed_recorded_flags(tmp_path, monkeypatch, ptx):
    cubin = tmp_path / "made.cubin"
    cubin.write_bytes(cubin_like(**SM90))
    monkeypatch.setenv("TILEFISH_PTXAS_PATH", fake_ptxas(tmp_path, cubin))
    op = make_ptx_op(ptx)
    built = _binary.build_binary(spec_of(op), op.binary, arch="sm_90")
    assert built.arch == "sm_90" and built.symbol == "tf_k" and built.shared == 0 and built.num_warps == 0 and built.options == {}
    assert built.binary_build == {"source": "ptx", "ptxas_flags": ["-lineinfo", "-v", "--gpu-name=sm_90"]}
    assert built.binary_digest == op.binary.digests["ptx"] and built.ir_digest is None and built.triton_version == "" and built.ptx_target == "sm_90"
    assert built.ptxas_version == "Cuda compilation tools, release 12.9, V12.9.86" and built.cubin == cubin.read_bytes()
    assert (tmp_path / "ptxas.args").read_text().split()[:3] == ["-lineinfo", "-v", "--gpu-name=sm_90"]
    # an architecture the PTX cannot be built for, a symbol the code does not have, a ptxas that fails
    with pytest.raises(TilefishBuildError, match="has no code section '.text.tf_k'"):
        cubin.write_bytes(cubin_like(**SM90).replace(b".text.tf_k", b".text.tf_j"))
        _binary.build_binary(spec_of(op), op.binary, arch=90)
    monkeypatch.setenv("TILEFISH_PTXAS_PATH", "/nonexistent/ptxas")
    _binary._ptxas_version.cache_clear() if hasattr(_binary, "_ptxas_version") and hasattr(_binary._ptxas_version, "cache_clear") else None
    with pytest.raises(TilefishBuildError, match="cannot read the version of ptxas"):
        _binary.build_binary(spec_of(op), op.binary, arch=90)


def test_a_cubin_is_used_as_it_is_for_its_architecture_and_refused_for_another(tmp_path):
    k90, k103 = tmp_path / "k90.cubin", tmp_path / "k103.cubin"
    k90.write_bytes(cubin_like(**SM90))
    k103.write_bytes(cubin_like(**SM103A))

    @tilefish.extern(cubin={"sm_90": k90, "sm_103": k103}, symbol="tf_k", block=(64,), grid=(1,), name="k")
    def k(x: Tensor[16, f32], y: Out[Tensor[16, f32]]): ...

    spec = spec_of(k)
    built = _binary.build_binary(spec, k.binary, arch="sm_90")
    assert built.cubin == k90.read_bytes() and built.binary_build == {"source": "cubin"} and built.ptxas_version == "" and built.binary_digest == SHA(k90.read_bytes())
    assert _binary.build_binary(spec, k.binary, arch=103).cubin == k103.read_bytes()
    with pytest.raises(TilefishBuildError, match=r"'k' has no cubin for sm_80 \(it has \['sm_103', 'sm_90'\]\)"):
        _binary.build_binary(spec, k.binary, arch=80)

    @tilefish.extern(cubin={"sm_80": k90}, symbol="tf_k", block=(64,), grid=(1,), name="liar")  # a cubin of sm_90 under the key sm_80
    def liar(x: Tensor[16, f32], y: Out[Tensor[16, f32]]): ...

    with pytest.raises(TilefishBuildError, match=r"declared for sm_80 of 'liar' is for sm_90 \(its ELF header says so\)"):
        _binary.build_binary(spec_of(liar), liar.binary, arch=80)


def test_a_declaration_that_is_not_the_one_of_the_spec_is_refused_by_a_build(tmp_path, ptx):
    other = make_ptx_op(write_ptx(tmp_path, "o.ptx", PTX_TEXT + "// o\n"))
    with pytest.raises(TilefishBuildError, match="is not the one the compile spec 'axpy' was made from"):
        _binary.build_binary(spec_of(make_ptx_op(ptx)), other.binary, arch=90)


# -- the manifest ---------------------------------------------------------------------------------------------------------------------------


def manifest_of(tmp_path, op_, **kw):
    spec = spec_of(op_)
    built = fake_binary_build(spec, op_.binary, **kw)
    write_manifest([built], tmp_path / "m" / "manifest.json", cubins=True)
    return Manifest.load(tmp_path / "m"), spec


def test_a_prebuilt_entry_records_its_source_and_how_it_was_made(tmp_path, op, ptx):
    m, spec = manifest_of(tmp_path, op)
    entry = json.loads((tmp_path / "m" / "manifest.json").read_text())["kernels"][0]
    assert entry["kind"] == "binary" and entry["symbol"] == "tf_k" and entry["shared"] == 0 and entry["block"] == [128, 1, 1]
    assert "num_warps" not in entry and "num_stages" not in entry and "options" not in entry
    assert entry["source_digest"] == SHA(ptx.read_bytes()) and entry["source_digest_kind"] == BINARY_DIGEST_KIND == "tilefish.binary.v1"
    assert entry["binary"] == {"source": "ptx", "ptxas_flags": ["-lineinfo", "-v", "--gpu-name=sm_90"]}
    assert entry["toolchain"] == {"ptxas": "Cuda compilation tools, release 12.9, V12.9.86"}
    assert entry["abi"]["scratch_slots"] == 0 and entry["abi"]["scratch"] == [] and entry["constexprs"] == []
    assert entry["launch"] == {"global_scratch_size": 0, "profile_scratch_size": 0, "launch_pdl": False, "cluster_dims": [1, 1, 1], "num_ctas": 1, "cooperative": False}
    assert m.digest == manifest_digest(json.loads((tmp_path / "m" / "manifest.json").read_text()))
    built = m.built_kernel(spec, "sm_90")  # no Triton, no pin needed: the sha256 of the bytes is what is compared
    assert built.binary_digest == SHA(ptx.read_bytes()) and built.num_warps == 0 and built.options == {} and built.binary_build["source"] == "ptx"


def test_a_cubin_entry_has_no_toolchain(tmp_path):
    k = tmp_path / "k.cubin"
    k.write_bytes(cubin_like(**SM90))

    @tilefish.extern(cubin={"sm_90": k}, symbol="tf_k", block=(64,), grid=(1,), name="k")
    def kern(x: Tensor[16, f32], y: Out[Tensor[16, f32]]): ...

    m, spec = manifest_of(tmp_path, kern)
    doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
    entry = doc["kernels"][0]
    assert entry["binary"] == {"source": "cubin"} and entry["toolchain"] == {} and entry["source_digest"] == SHA(k.read_bytes()) == entry["sha256"]
    assert m.built_kernel(spec, "sm_90").binary_build == {"source": "cubin"}
    # a cubin used as it is is its own source: a source_digest that is not the sha256 of the file is refused when the manifest is read (as Rust does),
    # whatever the declaration says (the replay's comparison with the declaration comes later, and only for its own operator)
    entry["source_digest"] = "ab" * 32
    (tmp_path / "m" / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match=r"'k' \(sm_90\): a cubin used as it is is its own source, so its source_digest is the sha256 of the cubin file"):
        Manifest.load(tmp_path / "m")


def test_a_prebuilt_cubin_must_be_for_the_architecture_of_its_entry_and_say_which_it_is(tmp_path, op):
    """As the Rust reader (``verify_cubin``): the ELF header of a prebuilt binary names its architecture, and it is the entry's."""
    manifest_of(tmp_path, op)
    path = tmp_path / "m" / "manifest.json"
    for cubin, message in (
        (cubin_like(shared=None, **SM103A), r"'axpy' \(sm_90\): 'axpy.sm_90.cubin' is for sm_103 \(its ELF header says so\), but the manifest says sm_90"),
        (cubin_like(shared=None, osabi=0x07), r"'axpy' \(sm_90\): the ELF header of 'axpy.sm_90.cubin' is not that of a CUDA cubin of a known architecture"),
    ):
        doc = json.loads(path.read_text())
        (tmp_path / "m" / doc["kernels"][0]["cubin"]).write_bytes(cubin)
        doc["kernels"][0].update(sha256=hashlib.sha256(cubin).hexdigest(), sha256_code=_elf.code_digest(cubin))
        path.write_text(json.dumps(doc))
        with pytest.raises(TilefishManifestError, match=message):
            Manifest.load(tmp_path / "m", pin=manifest_digest(doc))


def test_a_replay_refuses_an_entry_built_from_other_bytes_or_for_another_kind(tmp_path, op, ptx):
    m, spec = manifest_of(tmp_path, op)
    other = make_ptx_op(write_ptx(tmp_path, "edited.ptx", PTX_TEXT + "// the same code, another file\n"))
    with pytest.raises(TilefishManifestError, match="the file changed since the manifest was written, so its binary is stale"):
        m.built_kernel(spec_of(other), "sm_90")
    # the entry of a Triton kernel is not a prebuilt binary's
    doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
    doc["kernels"][0]["kind"] = "triton"
    doc["kernels"][0]["options"] = {}
    doc["kernels"][0]["num_warps"] = 4
    doc["kernels"][0]["num_stages"] = 3
    doc["kernels"][0]["abi"]["scratch_slots"] = 2
    (tmp_path / "m" / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="parameter alpha: a fixed argument belongs to a prebuilt binary, and this is a triton kernel"):
        Manifest.load(tmp_path / "m")  # its fixed arguments say what it is, to the reader (as to the Rust reader)
    params = doc["kernels"][0]["abi"]["params"]
    doc["kernels"][0]["abi"]["params"] = [p for p in params if p["kind"] != "fixed"]
    (tmp_path / "m" / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="is a triton kernel, but the compile spec is for a prebuilt binary"):
        Manifest.load(tmp_path / "m").built_kernel(spec, "sm_90")


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda e: e.update(kind="elf"), "has kind 'elf'"),
        (lambda e: e.pop("kind"), "has kind None"),
        (lambda e: e.update(num_warps=4), "prebuilt binary, which has no \\['num_warps'\\]"),
        (lambda e: e.update(options={}), "prebuilt binary, which has no \\['options'\\]"),
        (lambda e: e.pop("binary"), "is missing \\['binary'\\]"),
        (lambda e: e.update(binary={"source": "elf"}), "binary must be"),
        (lambda e: e.update(binary={"source": "cubin", "ptxas_flags": []}), "binary must be"),
        (lambda e: e.update(binary={"source": "ptx"}), "binary must be"),
        (lambda e: e.update(binary={"source": "ptx", "ptxas_flags": [1]}), "binary must be"),
        (lambda e: e.update(source_digest_kind="tilefish.triton-ir.v1"), "records the sha256 of its PTX or cubin as source_digest"),
        (lambda e: e.pop("source_digest"), "source_digest and source_digest_kind come together|records the sha256"),
        (lambda e: e.update(source_sha256="0" * 64), "has no source_sha256"),
        (lambda e: e["abi"].update(scratch_slots=2), "a binary kernel has 0 scratch slots, but abi.scratch_slots is 2"),
        (lambda e: e["abi"].update(scratch_slots="0"), "scratch slots, but abi.scratch_slots is '0'"),
    ],
)
def test_the_entry_of_a_prebuilt_binary_is_read_strictly(tmp_path, op, change, message):
    manifest_of(tmp_path, op)
    doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
    change(doc["kernels"][0])
    (tmp_path / "m" / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match=message):
        Manifest.load(tmp_path / "m")


def test_a_triton_entry_has_two_scratch_slots_and_a_kind():
    from built_fakes import fake_build, scale

    (spec,) = compile_specs(scale)
    entry = spec_entry(spec)
    assert entry["kind"] == "triton" and entry["abi"]["scratch_slots"] == 2 and len(entry["abi"]["scratch"]) == 2
    assert "binary" not in entry and "symbol" not in entry and fake_build(spec).binary_digest is None


def test_the_digest_of_a_manifest_covers_the_binary_section_and_the_flags(tmp_path, op):
    m, _ = manifest_of(tmp_path, op)
    doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
    for change in (
        lambda e: e["binary"]["ptxas_flags"].append("-O0"),
        lambda e: e.update(kind="triton"),
        lambda e: e["abi"].update(scratch_slots=2),
        lambda e: e["abi"]["params"][3].update(value=3.0),
        lambda e: e.update(source_digest="1" * 64),
        lambda e: e["toolchain"].update(ptxas="other"),
    ):
        other = copy.deepcopy(doc)
        change(other["kernels"][0])
        assert manifest_digest(other) != m.digest


def test_the_rust_emitter_writes_a_launcher_for_a_prebuilt_binary(tmp_path, op):
    """The launcher of a prebuilt binary has no scratch slots to pass and no parameter for a fixed argument (``test_rust_host_schema3.py`` has the rest)."""
    from tilefish.hosts import rust

    m, _ = manifest_of(tmp_path, op)
    source = rust.render(m.check_pin(m.digest), ops=[op])
    text = re.sub(r"\n\s*///[ ]?", " ", source)  # the doc comments, a wrapped paragraph as one run of words
    assert "pub struct Axpy " in source and "`alpha` = 2.0 (f32)" in text and "`scratch` = 0 (i64)" in text
    assert "x: TensorRef<'a, F32>" in source and "alpha" not in source.split("pub fn launch<'a>")[1].split("Result<(), Error>")[0]
