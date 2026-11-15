"""Implementations per architecture with names, prebuilt binaries and ABIs of their own (``tilefish.impl(..., name=, ptx=, cubin=, stub=)``), with no
Triton and no GPU.

* **A named implementation** (``tilefish.impl(..., name="tcgen05")``) is one implementation for every architecture mapped to it: its key is its name,
  in its declaration digest, its compile specs and its manifest entries (``"implementation": "tcgen05"``), so those architectures share one digest
  and one set of specs. Without a name everything is as it was (a key, a digest and specs per architecture, the digests of 0.0.6).
* **A prebuilt binary** (``tilefish.impl(ptx=..., symbol=..., block=...)`` or ``cubin={...}``) is an implementation: its entries are ``binary``
  entries with ``implementation``; a cubin implementation has a cubin for every architecture mapped to it.
* **An ABI per implementation.** An implementation may bring its own stub (``stub=``), so a tensor it takes as a TMA descriptor may be a pointer in
  another, with other scalars, constexprs or fixed arguments. What every implementation shares is the op's public signature: the tensors (names,
  order, access, dtype, shape, ``Strided`` or not), the fixed arguments per call, the buckets. A call, a plan step and a Rust launcher bind that;
  each launch has the ABI of the implementation it runs, and a plan is checked against every implementation.

``test_gpu_per_arch_impls.py`` runs real kernels.
"""

import hashlib
import importlib
import json
import linecache
import os
import pathlib
import subprocess
import sys

import fake_compile
import pytest
from binary_fakes import PTX_TEXT, fake_ptxas, write_ptx
from built_fakes import fake_build
from conftest import REPO_ROOT
from elf_builder import arch_header, cubin_like
from fake_torch import FakeTensor, prepare
from test_rust_host import needs_new_crate

import tilefish
from tilefish import (
    Dim,
    Manifest,
    Out,
    StrideOf,
    Strided,
    Tensor,
    TensorDesc,
    TilefishBuildError,
    TilefishDeclarationError,
    TilefishManifestError,
    TilefishPreconditionError,
    TilefishRuntimeError,
    TilefishTypeError,
    const,
    f16,
    f32,
    i32,
    i64,
    runtime,
)
from tilefish._build import build, write_manifest
from tilefish.aot import CompileSpec, _canonical_declaration, compile_specs, declaration_digest

R, Cols = Dim("R"), Dim("C")
n = Dim("n", divisible_by=16)
Rb = Dim("Rb", buckets=[64, 128])
MOD = "pai_kernels"
TESTS = pathlib.Path(__file__).resolve().parent
SLAB = 1 << 40


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def prose(text):
    """The doc comments of generated Rust as one line of text (a sentence may be wrapped over several comment lines)."""
    return " ".join(line.strip().removeprefix("///").strip() for line in text.splitlines())


# -- the row reduction of test_per_arch.py, with names --------------------------------------------------------------------------------------------


def wide(**kw):
    return tilefish.impl(triton=f"{MOD}:row_sum_wide", num_warps=kw.pop("num_warps", 8), consts={"ROWS": 4}, **kw)


def narrow(**kw):
    return tilefish.impl(triton=f"{MOD}:row_sum_narrow", **kw)


def declare(by_arch, fallback=None, name="row_sum"):
    @tilefish.extern(by_arch=by_arch, fallback=fallback, grid=lambda R, ROWS: (tilefish.cdiv(R, ROWS),), name=name)
    def row_sum(x: Tensor[R, 256, f32], y: Out[Tensor[R, f32]], n_rows: R, ROWS: const = 1, C: const = 256): ...

    return row_sum


# the declaration digest of each implementation of the op below and the sha256 of the JSON of its compile spec, as tilefish 0.0.6 computes them:
# an implementation without a name is what it was
DIGESTS_0_0_6 = {
    "fallback": ("3842636316f8ad42bf99c6bb5027ac4ac3e70638c261965ebfe255e99eee553a", "1cc707f3b9640a8679ce033a3a74e9a45fa2a158cddddcfd3a6c3967ed7840c6"),
    "sm_103": ("e690ac8a493431ef9bea4973810fd58d63d159244025a712929aa78ce33314ac", "22880c147d62ea6bd57126eb770801ebb39d99039cadf542683afd733c52a630"),
    "sm_120": ("b1fd013324a06127475bf93f7f09ae0e40f8087a81a66ca1449ee47cf2aa1f30", "8492495ab9407be7479ddf8969ccbf49dce95e52e499ec99e041a29767f094bf"),
}


def test_an_implementation_without_a_name_digests_as_it_did():
    narrow_ref = "pa_kernels:row_sum_narrow"
    op = declare(
        {"sm_103": tilefish.impl(triton="pa_kernels:row_sum_wide", num_warps=8, consts={"ROWS": 4}), "sm_120": tilefish.impl(triton=narrow_ref)},
        fallback=tilefish.impl(triton=narrow_ref, num_warps=2),
    )
    assert list(op.implementations) == ["sm_103", "sm_120", "fallback"]
    for key, (digest, spec_sha) in DIGESTS_0_0_6.items():
        one = op.implementations[key]
        (spec,) = compile_specs(one)
        assert (declaration_digest(one), sha(spec.to_json())) == (digest, spec_sha), key


def test_a_named_implementation_is_one_implementation_for_every_architecture_mapped_to_it():
    tc = wide(name="tcgen05")
    op = declare({"sm_100": tc, "sm_103": tc, "sm_120": narrow()})
    assert list(op.implementations) == ["tcgen05", "sm_120"]
    assert op.selection == {"sm_100": "tcgen05", "sm_103": "tcgen05", "sm_120": "sm_120"} and op.fallback_key is None
    assert op.for_arch(100) is op.for_arch("sm_103") is op.implementations["tcgen05"]
    a, b = compile_specs(op, arch=100), compile_specs(op, arch=103)
    assert a == b and a[0].implementation == "tcgen05" and a[0].to_dict()["implementation"] == "tcgen05"
    assert a[0].declaration_digest == declaration_digest(op.for_arch(100))
    # one implementation object without a name for two architectures: a key, a digest and specs for each (as before)
    plain = wide()
    unnamed = declare({"sm_100": plain, "sm_103": plain, "sm_120": narrow()})
    assert list(unnamed.implementations) == ["sm_100", "sm_103", "sm_120"] and unnamed.for_arch(100) is not unnamed.for_arch(103)
    digests = {declaration_digest(unnamed.for_arch(100)), declaration_digest(unnamed.for_arch(103)), a[0].declaration_digest}
    assert len(digests) == 3
    # which architectures map to it is not in its digest: one more leaves it as it was (a manifest is per architecture)
    more = declare({"sm_100": tc, "sm_103": tc, "sm_110": tc, "sm_120": narrow()})
    assert declaration_digest(more.for_arch(110)) == a[0].declaration_digest
    with pytest.raises(TilefishRuntimeError, match=r"no implementation for sm_90: it declares one for sm_100, sm_103, sm_120 and no fallback"):
        op.for_arch(90)
    assert op.node_attrs() == {"by_arch": {"tcgen05": f"{MOD}:row_sum_wide", "sm_120": f"{MOD}:row_sum_narrow"}}


def test_a_fallback_may_be_named_and_be_an_architectures_implementation_too():
    generic = narrow(name="generic")
    op = declare({"sm_103": wide(), "sm_120": generic}, fallback=generic)
    assert list(op.implementations) == ["sm_103", "generic"] and op.fallback_key == "generic"
    assert op.for_arch(90) is op.for_arch(120) is op.implementations["generic"]
    assert compile_specs(op, arch=80)[0].implementation == "generic"
    alone = declare({"sm_103": wide()}, fallback=narrow(name="generic"))
    assert alone.for_arch(80).implementation == "generic" and list(alone.implementations) == ["sm_103", "generic"]


@pytest.mark.parametrize("bad", ["fallback", "Fallback", "FALLBACK", "sm_103", "SM_90", "sm_9x", "a-b", "", "2nd", "x" * 65, "té", 3, True])
def test_a_name_is_an_identifier_that_is_not_fallback_and_not_an_architecture(bad):
    with pytest.raises(TilefishDeclarationError, match="name= is the key of the implementation in its manifest entries"):
        narrow(name=bad)


@pytest.mark.parametrize("good", ["tcgen05", "sm_", "smem", "_", "x" * 64, "fallback2", "Sm90ish"])
def test_a_name_may_be_any_other_identifier(good):
    assert narrow(name=good).name == good
    assert declare({"sm_103": narrow(name=good)}).for_arch(103).implementation == good


def test_two_different_implementations_may_not_share_a_name():
    with pytest.raises(TilefishDeclarationError, match=r"two different implementations are named 'tc' \(they differ at num_warps: 8 and 4\)"):
        declare({"sm_100": wide(name="tc"), "sm_103": wide(name="tc", num_warps=4)})
    with pytest.raises(TilefishDeclarationError, match=r"two different implementations are named 'tc' \(they differ at triton"):
        declare({"sm_100": wide(name="tc")}, fallback=narrow(name="tc"))
    # two tilefish.impl objects that declare the same implementation under one name are one implementation
    op = declare({"sm_100": wide(name="tc"), "sm_103": wide(name="tc")})
    assert list(op.implementations) == ["tc"] and op.for_arch(100) is op.for_arch(103)


def test_a_named_spec_round_trips_and_is_verified_by_its_name():
    op = declare({"sm_100": wide(name="tc"), "sm_103": wide(name="tc"), "sm_120": narrow()})
    (spec,) = compile_specs(op, arch=100)
    back = CompileSpec.from_json(spec.to_json())
    assert back.implementation == "tc" and not back.verified
    back.verify_against(op)
    assert back.verified and back._selection == spec._selection == ((("sm_100", "tc"), ("sm_103", "tc"), ("sm_120", "sm_120")), None)
    doc = spec.to_dict()
    for bad in ("sm_9", "Fallback", "a-b", "x" * 65, 103, None, ""):
        doc["implementation"] = bad
        with pytest.raises(tilefish.TilefishSpecError, match="implementation"):
            CompileSpec.from_dict(doc)
    doc["implementation"] = "other"  # a name the declaration does not have
    with pytest.raises(tilefish.TilefishSpecError, match=r"first difference at implementation: the spec has 'other'"):
        CompileSpec.from_dict(doc).verify_against(op)


# -- builds, exports and replays (the fake frontend and compiler) ---------------------------------------------------------------------------------

KERNELS = '''from fake_jit import jit


@jit
def row_sum_wide(x, y, n_rows, ROWS, C):
    return (x + n_rows, "wide", ROWS, C)


@jit
def row_sum_narrow(x, y, n_rows, ROWS, C):
    return (x + n_rows, "narrow", ROWS, C)


@jit
def axpy(x, y, n_elem, alpha, scratch):
    return (x * alpha + n_elem, scratch)


@jit
def copy_rows(x, y, n_cols, s_row, BLOCK):
    return (x, n_cols, s_row, BLOCK)


@jit
def copy_tma(x, y, BM, BN):
    return (x, BM, BN)


@jit
def scale(x, y, n):
    return (x * 2.0, n)
'''


@pytest.fixture
def world(tmp_path, monkeypatch):
    compiler = fake_compile.install(monkeypatch)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    directory = tmp_path / "project"
    directory.mkdir()
    (directory / f"{MOD}.py").write_text(KERNELS)
    monkeypatch.syspath_prepend(str(directory))
    sys.modules.pop(MOD, None)
    importlib.invalidate_caches()
    linecache.clearcache()
    yield directory, compiler
    sys.modules.pop(MOD, None)


def test_a_named_implementation_is_built_only_for_the_architectures_that_run_it(world):
    tc = wide(name="tc")
    op = declare({"sm_100": tc, "sm_103": tc, "sm_120": narrow()}, fallback=narrow(name="generic", num_warps=2))
    (spec,) = compile_specs(op, arch=103)
    assert build(spec, arch=100).spec.implementation == "tc" and build(spec, arch=103).arch == "sm_103"
    with pytest.raises(
        TilefishBuildError, match=r"'row_sum' is the implementation 'tc' \(declared for sm_100, sm_103\), not for sm_120: build the implementation of sm_120"
    ):
        build(spec, arch=120)
    (fb,) = compile_specs(op, arch=90)
    assert build(fb, arch=90).spec.implementation == "generic"
    with pytest.raises(TilefishBuildError, match=r"is the fallback 'generic', which the declaration does not run on sm_103: it runs its implementation 'tc' there"):
        build(fb, arch=103)
    back = CompileSpec.from_json(spec.to_json())  # a spec read back and verified knows which architectures run it too
    back.verify_against(op)
    with pytest.raises(TilefishBuildError, match="not for sm_120"):
        build(back, arch=120)
    # the fallback is not built for an architecture that has an implementation of its own
    plain = declare({"sm_103": wide()}, fallback=narrow())
    (fb,) = compile_specs(plain, arch=90)
    with pytest.raises(TilefishBuildError, match=r"'row_sum' is the fallback, which the declaration does not run on sm_103: it runs its implementation 'sm_103' there"):
        build(fb, arch=103)


def test_the_entry_of_a_named_implementation_says_its_name_and_a_replay_holds_the_map_to_it(world, tmp_path):
    tc = wide(name="tc")
    op = declare({"sm_100": tc, "sm_103": tc, "sm_120": narrow()})
    out = tmp_path / "build"
    runtime.export(op, out, arch=100)
    m = runtime.export(op, out, arch=103)
    entries = {e["arch"]: e for e in m.doc["kernels"]}
    assert {a: e["implementation"] for a, e in entries.items()} == {"sm_100": "tc", "sm_103": "tc"}
    assert entries["sm_100"]["declaration_digest"] == entries["sm_103"]["declaration_digest"] == declaration_digest(op.for_arch(103))
    (spec,) = compile_specs(op, arch=103)
    m.built_kernel(spec, "sm_103", pinned=True)
    # another architecture mapped to the name leaves the entries valid
    more = declare({"sm_100": tc, "sm_103": tc, "sm_110": tc, "sm_120": narrow()})
    m.built_kernel(compile_specs(more, arch=103)[0], "sm_103", pinned=True)
    # sm_103 runs another implementation now (its own, or another name): the entry is stale
    for changed in (declare({"sm_100": tc, "sm_103": wide(), "sm_120": narrow()}), declare({"sm_100": tc, "sm_103": wide(name="tc2"), "sm_120": narrow()})):
        (s,) = compile_specs(changed, arch=103)
        with pytest.raises(
            TilefishManifestError, match=rf"from the implementation 'tc' of an op declared with by_arch=, but the declaration now runs its implementation '{s.implementation}' on sm_103"
        ):
            m.built_kernel(s, "sm_103", pinned=True)
    # the named implementation changed (its options): its declaration digest, on every architecture mapped to it
    changed = declare({"sm_100": wide(name="tc", num_stages=2), "sm_103": wide(name="tc", num_stages=2), "sm_120": narrow()})
    for arch in (100, 103):
        with pytest.raises(TilefishManifestError, match="first difference at"):
            m.built_kernel(compile_specs(changed, arch=arch)[0], f"sm_{arch}", pinned=True)


def test_the_typed_rust_launchers_name_each_architectures_implementation(world, tmp_path):
    from tilefish.hosts import rust

    tc = wide(name="tc")
    op = declare({"sm_100": tc, "sm_103": tc, "sm_120": narrow()})
    out = tmp_path / "build"
    for arch in (100, 103, 120):
        manifest = runtime.export(op, out, arch=arch)
    text = prose(rust.render(manifest, ops=[op]))
    assert "sm_100: the implementation `tc`, grid `(cdiv(R, 4),)`" in text and "sm_103: the implementation `tc`" in text
    assert "sm_120: the implementation `sm_120`, grid `(R,)`" in text
    with pytest.raises(TilefishManifestError, match=r"binary 'row_sum' was built from \(on sm_100\)"):
        rust.render(manifest, ops=[declare({"sm_100": wide(), "sm_103": tc, "sm_120": narrow()})])


def block(op):
    Mb = Dim("Mb", buckets=[8, 16])

    @tilefish.extern(triton=f"{MOD}:scale", grid=lambda Mb: (Mb,), name="scale")
    def scale(x: Tensor[Mb, 256, f32], y: Out[Tensor[Mb, 256, f32]], n: Mb): ...

    @tilefish.graph
    def summed(x: Tensor[Mb, 256, f32]) -> Tensor[Mb, f32]:
        return op(scale(x))

    return summed


def test_a_plan_is_the_same_document_whatever_the_names():
    tc = wide(name="tc")
    named = tilefish.plan(block(declare({"sm_100": tc, "sm_103": tc, "sm_120": narrow()})))
    plain = tilefish.plan(block(declare({"sm_103": wide(), "sm_120": narrow()})))
    assert named.digest == plain.digest
    assert set(named.callees["row_sum"].implementations) == {"tc", "sm_120"}
    assert named.callees["row_sum"].at(100) is named.callees["row_sum"].at(103)


# -- prebuilt binaries as implementations ----------------------------------------------------------------------------------------------------------


def declare_axpy(by_arch, fallback=None):
    """``axpy(x, y: Out, n_elem, alpha, scratch)``: the C ABI of binary_fakes.PTX_TEXT, and the parameters of the Triton kernel ``axpy``."""

    @tilefish.extern(by_arch=by_arch, fallback=fallback, grid=lambda n: (tilefish.cdiv(n, 128),), name="axpy")
    def axpy(x: Tensor[n, f32], y: Out[Tensor[n, f32]], n_elem: n, alpha: const[f32] = 2.0, scratch: const[i64] = 0): ...

    return axpy


def ptx_impl(path, **kw):
    return tilefish.impl(ptx=path, symbol="tf_k", block=(128,), **kw)


def test_a_prebuilt_binary_is_an_implementation(tmp_path):
    from binary_fakes import make_ptx_op

    ptx = write_ptx(tmp_path)
    op = declare_axpy({"sm_103": ptx_impl(ptx), "sm_120": tilefish.impl(triton=f"{MOD}:axpy")})
    one = op.for_arch(103)
    assert one.binary.kind == "ptx" and one.implementation == "sm_103" and op.for_arch(120).binary is None and not op.is_binary
    (spec,) = compile_specs(op, arch=103)
    assert spec.binary["kind"] == "ptx" and spec.implementation == "sm_103" and spec.to_dict()["implementation"] == "sm_103"
    assert [a.kind for a in spec.args] == ["pointer", "pointer", "scalar", "fixed", "fixed"] and spec.launch_params[-1].name == "scratch"
    (other,) = compile_specs(op, arch=120)
    assert [a.kind for a in other.args] == ["pointer", "pointer", "scalar", "constexpr", "constexpr"]
    back = CompileSpec.from_json(spec.to_json())
    back.verify_against(op)
    assert back.verified
    # the same PTX declared as an op of its own is another declaration: the implementation's key is in its digest, and nothing else differs
    plain = make_ptx_op(ptx)
    assert declaration_digest(plain) != declaration_digest(one)
    mine, theirs = _canonical_declaration(one), _canonical_declaration(plain)
    assert mine.pop("implementation") == "sm_103" and mine == theirs
    named = declare_axpy({"sm_100": ptx_impl(ptx, name="ptx"), "sm_103": ptx_impl(ptx, name="ptx")})
    assert compile_specs(named, arch=100) == compile_specs(named, arch=103) and named.is_binary


def test_a_relative_path_is_relative_to_the_file_that_declares_the_op():
    @tilefish.extern(by_arch={"sm_103": tilefish.impl(ptx="ptx/axpy_f32.ptx", symbol="axpy_f32", block=(128,))}, grid=lambda n: (tilefish.cdiv(n, 128),))
    def axpy(x: Tensor[n, f32], y: Tensor[n, f32], out: Out[Tensor[n, f32]], n_elem: n, alpha: const[f32] = 2.0): ...

    assert axpy.for_arch(103).binary.digests["ptx"] == sha((TESTS / "ptx" / "axpy_f32.ptx").read_bytes())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(ptx="k.ptx", symbol="k", block=(32,), num_warps=4), r"num_warps= is a Triton compile option"),
        (dict(ptx="k.ptx", symbol="k", block=(32,), num_stages=2), r"num_stages= is a Triton compile option"),
        (dict(ptx="k.ptx", symbol="k", block=(32,), enable_fp_fusion=False), r"enable_fp_fusion= is a Triton compile option"),
        (dict(ptx="k.ptx", symbol="k"), r"ptx= needs symbol=<the kernel's entry point> and block="),
        (dict(cubin={"sm_103": "k.cubin"}, block=(32,)), r"cubin= needs symbol="),
        (dict(triton="m:f", symbol="k"), r"symbol= is for a prebuilt binary"),
        (dict(triton="m:f", block=(32,)), r"block= is for a prebuilt binary"),
        (dict(gluon="m:f", shared=16), r"shared= is for a prebuilt binary"),
        (dict(ptx="k.ptx", symbol="1k", block=(32,)), r"symbol= names the kernel's entry point"),
        (dict(ptx="k.ptx", symbol="k", block=(2048,)), r"block .* must have sizes from 1 to 1024"),
        (dict(ptx="k.ptx", symbol="k", block=(32,), shared=-1), r"shared must be a non-negative int"),
        (dict(ptx=3, symbol="k", block=(32,)), r"ptx= must be a file path"),
        (dict(cubin=["k.cubin"], symbol="k", block=(32,)), r"cubin= maps architectures to files"),
        (dict(cubin={"103": "k.cubin"}, symbol="k", block=(32,)), r"cubin= keys are architectures such as 'sm_103'"),
        (dict(triton="m:f", ptx="k.ptx"), r"takes one kernel.*got triton and ptx"),
        (dict(ptx="k.ptx", cubin={"sm_90": "k"}), r"takes one kernel.*got ptx and cubin"),
        (dict(triton="m:f", stub="m:stub"), r"stub= is a function annotated like the stub of an extern"),
        (dict(triton="m:f", stub=print), r"stub= is a function annotated like the stub of an extern"),
    ],
)
def test_an_implementation_of_a_prebuilt_binary_is_checked(kwargs, message):
    with pytest.raises(TilefishDeclarationError, match=message):
        tilefish.impl(**kwargs)


def test_the_files_of_a_prebuilt_implementation_are_read_when_the_op_is_declared(tmp_path):
    missing = ptx_impl(tmp_path / "none.ptx")  # (nothing is read yet)
    with pytest.raises(TilefishDeclarationError, match=r"extern 'axpy': cannot read ptx="):
        declare_axpy({"sm_103": missing})


def test_a_cubin_implementation_has_a_cubin_for_every_architecture_mapped_to_it(tmp_path):
    k103 = tmp_path / "k103.cubin"
    k103.write_bytes(cubin_like(shared=None, **arch_header("sm_103")))
    cubin = dict(cubin={"sm_103": k103}, symbol="tf_k", block=(128,))
    with pytest.raises(
        TilefishDeclarationError, match=r"the implementation for sm_120 is a cubin for sm_103, but by_arch= sends sm_120 to it: a cubin is used as it is"
    ):
        declare_axpy({"sm_120": tilefish.impl(**cubin)})
    with pytest.raises(TilefishDeclarationError, match=r"the implementation for 'cu' is a cubin for sm_103, but by_arch= sends sm_100 to it"):
        declare_axpy({"sm_100": tilefish.impl(**cubin, name="cu"), "sm_103": tilefish.impl(**cubin, name="cu")})
    op = declare_axpy({"sm_103": tilefish.impl(**cubin)})
    assert op.for_arch(103).binary.sources["sm_103"] == k103.read_bytes()
    # as the fallback, the architectures it runs on are not known when it is declared: the build refuses one without a cubin
    fb = declare_axpy({"sm_120": tilefish.impl(triton=f"{MOD}:axpy")}, fallback=tilefish.impl(**cubin))
    from tilefish import _binary

    (spec,) = compile_specs(fb, arch=90)
    with pytest.raises(TilefishBuildError, match=r"'axpy' has no cubin for sm_90 \(it has \['sm_103'\]\)"):
        _binary.build_binary(spec, fb.for_arch(90).binary, arch=90)
    assert _binary.build_binary(compile_specs(fb, arch=103)[0], fb.for_arch(103).binary, arch=103).cubin == k103.read_bytes()
    with pytest.raises(TilefishBuildError, match=r"is the fallback, which the declaration does not run on sm_120"):
        _binary.build_binary(spec, fb.for_arch(90).binary, arch=120)


def test_ptx_for_one_architecture_is_not_an_implementation_for_another(tmp_path):
    specific = write_ptx(tmp_path, "a.ptx", PTX_TEXT.replace(".target sm_90", ".target sm_103a"))
    with pytest.raises(
        TilefishDeclarationError, match=r"is PTX for the architecture-specific target sm_103a, which only runs on sm_103; it cannot be built for sm_120, and by_arch= sends sm_120 to it"
    ):
        declare_axpy({"sm_103": ptx_impl(specific), "sm_120": ptx_impl(specific)})
    declare_axpy({"sm_103": ptx_impl(specific)})
    newer = write_ptx(tmp_path, "b.ptx", PTX_TEXT.replace(".target sm_90", ".target sm_100"))
    with pytest.raises(TilefishDeclarationError, match=r"is PTX for sm_100, which is newer than the sm_90"):
        declare_axpy({"sm_90": ptx_impl(newer), "sm_103": ptx_impl(newer)})


def test_a_prebuilt_implementation_is_exported_replayed_and_launched_with_its_own_abi(world, tmp_path, monkeypatch):
    from tilefish.hosts import rust

    cubin = tmp_path / "made.cubin"
    cubin.write_bytes(cubin_like(shared=None, **arch_header("sm_103")))
    monkeypatch.setenv("TILEFISH_PTXAS_PATH", fake_ptxas(tmp_path, cubin))
    ptx = write_ptx(tmp_path)
    op = declare_axpy({"sm_103": ptx_impl(ptx, name="ptx")}, fallback=tilefish.impl(triton=f"{MOD}:axpy"))
    out = tmp_path / "build"
    runtime.export(op, out, arch=103)
    m = runtime.export(op, out, arch=90)
    entries = {e["arch"]: e for e in m.doc["kernels"]}
    b, t = entries["sm_103"], entries["sm_90"]
    assert (b["kind"], b["implementation"], b["abi"]["scratch_slots"], b["binary"]["source"]) == ("binary", "ptx", 0, "ptx")
    assert (t["kind"], t["implementation"], t["abi"]["scratch_slots"]) == ("triton", "fallback", 2)
    assert b["source_digest"] == sha(ptx.read_bytes())
    (spec,) = compile_specs(op, arch=103)
    assert m.built_kernel(spec, "sm_103").binary_build["source"] == "ptx"  # a prebuilt binary is told by its bytes: no pin needed
    # its PTX changed since: stale
    edited = write_ptx(tmp_path, "k2.ptx", PTX_TEXT + "// edited\n")
    changed = declare_axpy({"sm_103": ptx_impl(edited, name="ptx")}, fallback=tilefish.impl(triton=f"{MOD}:axpy"))
    with pytest.raises(TilefishManifestError, match="first difference at|stale"):
        m.built_kernel(compile_specs(changed, arch=103)[0], "sm_103", pinned=True)
    # a launch of each architecture's: the binary has no scratch slots and its fixed arguments, the Triton kernel constexprs
    x = FakeTensor((256,), "f32", SLAB)
    for cap, kinds in ((103, ["pointer", "pointer", "scalar"]), (90, ["pointer", "pointer", "scalar", "constexpr", "constexpr"])):
        monkeypatch.setattr(runtime, "_capability", lambda torch, index, cap=cap: cap)
        monkeypatch.setattr(runtime, "_device_index", lambda torch, device: 0)
        prepared, error = prepare(op, [x], next_slab=1 << 44)
        assert error is None, error
        assert [k for k, _, _ in runtime._launch_values(prepared)] == kinds and prepared.grid == (2,)
    # the Rust launcher: one signature, and the ABI of each architecture's implementation in its docs
    text = prose(rust.render(m, ops=[op]))
    assert "x: TensorRef<'a, F32>" in text and "y: TensorMut<'a, F32>" in text
    assert "- sm_103: the implementation `ptx`, a prebuilt binary (PTX that ptxas assembled" in text
    assert "- sm_90: the implementation `fallback`, a Triton kernel" in text
    assert "fixed arguments: `alpha` = 2.0 (f32), `scratch` = 0 (i64)" in text and "compiled in: `alpha` = 2.0 (f32), `scratch` = 0 (i64)" in text


def test_a_cubin_implementation_beside_a_triton_one_in_a_manifest(world, tmp_path):
    k103 = tmp_path / "k103.cubin"
    k103.write_bytes(cubin_like(shared=None, **arch_header("sm_103")))
    op = declare_axpy({"sm_103": tilefish.impl(cubin={"sm_103": k103}, symbol="tf_k", block=(128,))}, fallback=tilefish.impl(triton=f"{MOD}:axpy"))
    m = runtime.export(op, tmp_path / "build", arch=103)
    entry = m.get("axpy", "sm_103").entry
    assert (entry["kind"], entry["implementation"], entry["binary"], entry["toolchain"]) == ("binary", "sm_103", {"source": "cubin"}, {})
    assert entry["source_digest"] == entry["sha256"] == sha(k103.read_bytes())


# -- an ABI per implementation ---------------------------------------------------------------------------------------------------------------------


def copy_tma_stub(
    x: TensorDesc(Strided[Tensor[R, Cols, f16]], block=(64, 64)),
    y: Out[TensorDesc(Tensor[R, Cols, f16], block=(64, 64))],
    BM: const = 64,
    BN: const = 64,
): ...


def tma(**kw):
    """The TMA form: the view and the result as descriptors, a 64x64 tile per program, no scalar."""
    return tilefish.impl(triton=f"{MOD}:copy_tma", stub=copy_tma_stub, grid=lambda R, C: (tilefish.cdiv(R, 64), tilefish.cdiv(C, 64)), **kw)


def rows(**kw):
    """The pointer form: the op's stub, a row per program, the row stride a scalar."""
    return tilefish.impl(triton=f"{MOD}:copy_rows", **kw)


def declare_copy(by_arch=None, fallback=None, name="copy2d"):
    by_arch = {"sm_100": tma(name="tma"), "sm_103": tma(name="tma")} if by_arch is None else by_arch

    @tilefish.extern(by_arch=by_arch, fallback=fallback, grid=lambda R: (R,), name=name)
    def copy2d(x: Strided[Tensor[R, Cols, f16]], y: Out[Tensor[R, Cols, f16]], n_cols: Cols, s_row: StrideOf("x", 0), BLOCK: const = 256): ...

    return copy2d


def test_an_implementation_may_take_a_tensor_as_a_tma_descriptor_where_another_takes_a_pointer():
    op = declare_copy(fallback=rows())
    assert [p.name for p in op.signature.params] == ["x", "y", "n_cols", "s_row", "BLOCK"]  # the public signature: the op's stub
    (t,), (p,) = compile_specs(op, arch=103), compile_specs(op, arch=90)
    assert t.stem == p.stem == "copy2d" and (t.implementation, p.implementation) == ("tma", "fallback")
    assert [(a.name, a.kind) for a in t.args] == [("x", "tensordesc"), ("y", "tensordesc"), ("BM", "constexpr"), ("BN", "constexpr")]
    assert [(a.name, a.kind) for a in p.args] == [("x", "pointer"), ("y", "pointer"), ("n_cols", "scalar"), ("s_row", "scalar"), ("BLOCK", "constexpr")]
    # the dimensions each names: the descriptor of the view brings its own stride dimension, the pointer form a scalar's
    assert set(t.dims) == {"R", "C", "x_s0", "x_offset"} and set(p.dims) == {"R", "C", "s_row", "x_offset"}
    assert t.dims["x_offset"] == {"divisible_by": 8} and p.dims["x_offset"] == {}  # a descriptor's address is 16-byte aligned
    assert (t.grid, p.grid) == ("(cdiv(R, 64), cdiv(C, 64))", "(R,)")
    back = CompileSpec.from_json(t.to_json())
    back.verify_against(op)


def stub_with(x=None, y=None, extra=False):
    """A TMA stub with one thing changed."""
    xt = x if x is not None else TensorDesc(Strided[Tensor[R, Cols, f16]], block=(64, 64))
    yt = y if y is not None else Out[TensorDesc(Tensor[R, Cols, f16], block=(64, 64))]
    if extra:

        def stub(x: xt, y: yt, z: Out[Tensor[R, Cols, f16]], BM: const = 64, BN: const = 64): ...

    else:

        def stub(x: xt, y: yt, BM: const = 64, BN: const = 64): ...

    return stub


@pytest.mark.parametrize(
    "stub, where",
    [
        (stub_with(y=Out[TensorDesc(Tensor[R, Cols, f32], block=(64, 32))]), r"tensors\[1\].dtype \(the op's has 'f16', its 'f32'\)"),
        (stub_with(x=TensorDesc(Tensor[R, Cols, f16], block=(64, 64))), r"tensors\[0\].view \(the op's has True, its False\)"),
        (stub_with(y=Out[TensorDesc(Tensor[R, 64, f16], block=(64, 64))]), r"tensors\[1\].shape\[1\] \(the op's has 'C', its '64'\)"),
        (
            stub_with(
                x=TensorDesc(Strided[Tensor[Dim("R", divisible_by=64), Cols, f16]], block=(64, 64)),
                y=Out[TensorDesc(Tensor[Dim("R", divisible_by=64), Cols, f16], block=(64, 64))],
            ),
            r"dims.R.divisible_by \(the op's has None, its 64\)",
        ),
        (stub_with(extra=True), r"tensors \(length\)"),
    ],
)
def test_every_implementation_has_the_ops_public_signature(stub, where):
    with pytest.raises(TilefishDeclarationError, match=rf"the implementation for 'tma' does not have the op's signature: its stub differs at {where}"):
        declare_copy(by_arch={"sm_103": tilefish.impl(triton=f"{MOD}:copy_tma", stub=stub, grid=(1,), name="tma")})


def test_the_order_and_the_access_of_the_tensors_are_the_ops():
    def swapped(y: Out[TensorDesc(Tensor[R, Cols, f16], block=(64, 64))], x: TensorDesc(Strided[Tensor[R, Cols, f16]], block=(64, 64))): ...

    with pytest.raises(TilefishDeclarationError, match=r"its stub differs at tensors\[0\].name \(the op's has 'x', its 'y'\)"):
        declare_copy(by_arch={"sm_103": tilefish.impl(triton=f"{MOD}:copy_tma", stub=swapped, grid=(1,))})

    def written(x: TensorDesc(Strided[Tensor[R, Cols, f16]], block=(64, 64)), y: TensorDesc(Tensor[R, Cols, f16], block=(64, 64))): ...

    with pytest.raises(TilefishDeclarationError, match=r"differs at tensors\[1\].access \(the op's has 'out', its 'in'\)"):
        declare_copy(by_arch={"sm_103": tilefish.impl(triton=f"{MOD}:copy_tma", stub=written, grid=(1,))})


def test_the_stub_of_an_implementation_is_parsed_as_its_kernels():
    def bad(x: TensorDesc(Strided[Tensor[R, Cols, f16]], block=(64, 64)), y: Out[TensorDesc(Tensor[R, Cols, f16], block=(64, 64))], BM: const): ...

    with pytest.raises(TilefishDeclarationError, match=r"the implementation for sm_103: its stub: extern op 'copy2d': const parameter 'BM' needs a default"):
        declare_copy(by_arch={"sm_103": tilefish.impl(triton=f"{MOD}:copy_tma", stub=bad, grid=(1,))})
    # a prebuilt binary's stub is its C ABI: no descriptor
    with pytest.raises(TilefishDeclarationError, match=r"its stub: .*TMA tensor descriptor"):
        declare_copy(by_arch={"sm_103": tilefish.impl(ptx=TESTS / "ptx" / "axpy_f32.ptx", symbol="axpy_f32", block=(64,), stub=copy_tma_stub)})


def test_a_call_launches_the_implementation_of_its_device_with_that_implementations_abi(monkeypatch):
    op = declare_copy(fallback=rows())
    monkeypatch.setattr(runtime, "_device_index", lambda torch, device: 0)

    def run_on(cap, x, **kw):
        monkeypatch.setattr(runtime, "_capability", lambda torch, index: cap)
        saved = runtime._torch
        try:
            from fake_torch import FakeTorch

            runtime._torch = lambda: FakeTorch(1 << 44)
            return runtime._prepare(runtime._target(op), (x,), kw), None
        except tilefish.TilefishError as e:
            return None, e
        finally:
            runtime._torch = saved

    dense = FakeTensor((100, 192), "f16", SLAB)
    p103, e = run_on(103, dense)
    assert e is None, e
    assert p103.spec.implementation == "tma" and p103.grid == (2, 3) and p103.scalars == {}
    assert [(k, name) for k, name, _ in runtime._launch_values(p103)] == [("tensordesc", "x"), ("tensordesc", "y"), ("constexpr", "BM"), ("constexpr", "BN")]
    p90, e = run_on(90, dense)
    assert e is None, e
    assert p90.spec.implementation == "fallback" and p90.grid == (100,) and p90.scalars == {"n_cols": 192, "s_row": 192}
    # a view whose row stride is 386 bytes: the pointer form takes it, a TMA descriptor cannot be made of it (on that architecture only)
    odd = FakeTensor((100, 192), "f16", SLAB, strides=(193, 1))
    assert run_on(90, odd)[0].scalars == {"n_cols": 192, "s_row": 193}
    _, e = run_on(103, odd)
    assert isinstance(e, TilefishPreconditionError) and e.kind == "tensor_map", e
    # a view that starts 2 bytes into its buffer: a descriptor's address is 16-byte aligned, the pointer form's view the element's
    shifted = FakeTensor((100, 192), "f16", SLAB + 2, strides=(256, 1))
    assert run_on(90, shifted)[1] is None
    _, e = run_on(103, shifted)
    assert isinstance(e, TilefishPreconditionError) and e.kind == "pointer_alignment", e
    # a const of the op's stub: restated, it is held to the op's value, and to the implementation's where it has one
    assert run_on(103, dense, BLOCK=256)[1] is None and run_on(90, dense, BLOCK=256)[1] is None
    for cap in (90, 103):
        _, e = run_on(cap, dense, BLOCK=128)
        assert isinstance(e, TilefishTypeError) and "compiled in as 256" in str(e), (cap, e)


class Device:
    """The fake backend of ``test_planrun`` on a device of an architecture, that encodes tensor maps (bytes made of the address)."""

    def __new__(cls, cap):
        from test_planrun import FakeBackend

        class Backend(FakeBackend):
            def capability(self, index):
                return cap

            def encode_tensor_map(self, arguments):
                self.log.append(("encode", arguments.address, arguments.global_dim))
                return arguments.address.to_bytes(8, "little") * 16

        return Backend()


def copy_graph(op, width, part):
    """A graph that copies one column block of a ``[Rb, width]`` parameter (``x.split(width // 2)``, block ``part``) with ``op``."""

    @tilefish.graph
    def g(x: Tensor[Rb, width, f16]) -> Tensor[Rb, width // 2, f16]:
        return op(x.split(width // 2, axis=-1)[part])

    return g


def test_a_plan_binds_every_implementation_and_runs_the_one_of_its_device():
    from tilefish._planrun import PlanRunner
    from test_planrun import FakeProvider

    op = declare_copy(fallback=rows())
    p = tilefish.plan(copy_graph(op, 384, 1))
    (step,) = p.steps(Rb=64)
    assert step["entry"] == "copy2d" and [a["name"] for a in step["args"]] == ["x", "y"]
    assert step["args"][0]["view"] == {"shape": [64, 192], "strides": [384, 1], "offset": 192}
    x = FakeTensor((64, 384), "f16", 1 << 36)
    launched = {}
    for cap in (103, 90):
        be = Device(cap)
        PlanRunner(p, FakeProvider(), be).run((x,), {})
        (launch,) = [e for e in be.log if e[0] == "launch"]
        launched[cap] = (launch[2], [k for k, _ in launch[3]])
    assert launched == {103: ((1, 3), ["tensordesc", "tensordesc"]), 90: ((64,), ["pointer", "pointer", "scalar", "scalar"])}
    # the plan of the same graph over an op that has the pointer form only is the same document: a plan does not say which runs
    assert tilefish.plan(copy_graph(declare_copy(by_arch={"sm_103": rows()}), 384, 1)).digest == p.digest


def test_a_plan_is_checked_against_every_implementation_not_only_this_devices():
    """The view of the right half of a ``[Rb, 196]`` f16 tensor starts 196 bytes in and has rows of 392 bytes: the pointer form takes it, a TMA
    descriptor cannot be made of it. A plan is for every architecture, so it is refused even where the pointer form is what runs."""
    pointer_here = declare_copy(by_arch={"sm_103": rows()}, fallback=tma(name="tma"))
    with pytest.raises(TilefishPreconditionError, match=r"its implementation 'tma' \(a plan is for every architecture\)") as e:
        tilefish.plan(copy_graph(pointer_here, 196, 1))
    assert e.value.kind in ("pointer_alignment", "tensor_map")
    tilefish.plan(copy_graph(declare_copy(by_arch={"sm_103": rows()}), 196, 1))  # with the pointer form alone it is a plan


def test_the_rust_launcher_of_implementations_with_their_own_abis(tmp_path):
    from tilefish.hosts import rust

    op = declare_copy(fallback=rows())
    (t,), (p,) = compile_specs(op, arch=103), compile_specs(op, arch=90)
    path = write_manifest([fake_build(t, arch="sm_103"), fake_build(p, arch="sm_90")], tmp_path / "manifest.json", cubins=True)
    m = Manifest.load(tmp_path, pin=Manifest.load(tmp_path).digest)
    text = prose(rust.render(m, ops=[op]))
    assert "x: TensorRef<'a, F16>," in text and "y: TensorMut<'a, F16>," in text
    assert "- `x`: in, f16, `[R, C]`, a strided view - `y`: out, f16, `[R, C]`" in text  # (each implementation its own strides, below)
    assert "the binary of each architecture takes them as its ABI says (in stride scalars, or in a TMA descriptor)" in text
    assert "- sm_103: the implementation `tma`, a Triton kernel; grid `(cdiv(R, 64), cdiv(C, 64))`" in text
    assert "passed as TMA descriptors: `x` (box `[64, 64]`, SWIZZLE_128B), `y` (box `[64, 64]`, SWIZZLE_128B); its views: `x` (strides `[x_s0, 1]`, 16-byte aligned)" in text
    assert "- sm_90: the implementation `fallback`, a Triton kernel; grid `(R,)`" in text
    assert "runtime scalars, derived by `launch`: `n_cols` (i32) is `C`, `s_row` (i32) is the stride of axis 0 of `x`, in elements; its views: `x` (strides `[s_row, 1]`, 2-byte aligned)" in text
    # the entries of one stem must share what a launcher is typed from
    doc = json.loads(path.read_text())
    pointer = next(e for e in doc["kernels"] if e["arch"] == "sm_90")
    pointer["abi"]["params"][1].update(dtype="f32", type="*fp32")
    path.write_text(json.dumps(doc))
    edited = Manifest.load(tmp_path, pin=Manifest.load(tmp_path).digest)
    with pytest.raises(TilefishManifestError, match=r"the binaries of 'copy2d' for sm_103 and sm_90 disagree about the tensors and the fixed arguments per call"):
        rust.render(edited)


# -- fixed arguments per call are the op's ------------------------------------------------------------------------------------------------------


def declare_given(own=None):
    ptx = TESTS / "ptx" / "strided_copy_bf16.ptx"

    def own_stub(x: Strided[Tensor[R, 64, f16]], y: Out[Tensor[R, 64, f16]], cols: const[i32], x_s0: StrideOf("x", 0, dtype=i64)): ...

    impl = tilefish.impl(ptx=ptx, symbol="strided_copy_bf16", block=(64,), stub=own if own is not None else own_stub)

    @tilefish.extern(by_arch={"sm_103": impl}, fallback=tilefish.impl(ptx=ptx, symbol="strided_copy_bf16", block=(64,)), grid=lambda R: (R,), name="gcopy")
    def gcopy(x: Strided[Tensor[R, 64, f16]], y: Out[Tensor[R, 64, f16]], cols: const[i32], s: StrideOf("x", 0, dtype=i64)): ...

    return gcopy


def test_the_fixed_arguments_per_call_are_the_ops_and_a_call_gives_them_whichever_runs(monkeypatch):
    op = declare_given()
    assert op.is_binary  # every implementation is a prebuilt binary: a call gives the values of its fixed arguments per call
    for arch in (103, 90):
        (spec,) = compile_specs(op, arch=arch)
        assert [(a.name, a.kind) for a in spec.args if a.kind == "given"] == [("cols", "given")]

    def without(x: Strided[Tensor[R, 64, f16]], y: Out[Tensor[R, 64, f16]], *, cols: const[i32] = 64, x_s0: StrideOf("x", 0, dtype=i64)): ...

    with pytest.raises(TilefishDeclarationError, match=r"its stub differs at given \(length\)"):
        declare_given(own=without)
    # a Triton implementation has no fixed argument per call, so an op that has one cannot have it
    with pytest.raises(TilefishDeclarationError, match=r"const parameter 'cols' needs a default"):

        @tilefish.extern(by_arch={"sm_103": tilefish.impl(triton=f"{MOD}:copy_rows")}, grid=lambda R: (R,), name="gcopy")
        def gcopy(x: Strided[Tensor[R, 64, f16]], y: Out[Tensor[R, 64, f16]], cols: const[i32], s: StrideOf("x", 0, dtype=i64)): ...

    monkeypatch.setattr(runtime, "_device_index", lambda torch, device: 0)
    x = FakeTensor((8, 64), "f16", SLAB, strides=(80, 1))
    for cap in (103, 90):
        monkeypatch.setattr(runtime, "_capability", lambda torch, index, cap=cap: cap)
        saved = runtime._torch
        from fake_torch import FakeTorch

        runtime._torch = lambda: FakeTorch(1 << 44)
        try:
            prepared = runtime._prepare(runtime._target(op), (x,), {"cols": 64})
        finally:
            runtime._torch = saved
        assert prepared.scalars == {**({"x_s0": 80} if cap == 103 else {"s": 80}), "cols": 64}

    @tilefish.graph
    def g(x: Tensor[Rb, 128, f16]) -> Tensor[Rb, 64, f16]:
        return op(x.split(64, axis=-1)[0], cols=64)

    (step,) = tilefish.plan(g).steps(Rb=64)
    assert step["given"] == {"cols": 64}  # one plan whichever runs: both implementations take the value


# -- the launchers compile against the crate ---------------------------------------------------------------------------------------------------------


@needs_new_crate
def test_the_launchers_of_implementations_with_their_own_abis_compile_against_the_crate(tmp_path):
    """``cargo check`` with no warnings of the module of a stem whose entries have an ABI per architecture (a TMA descriptor form, a pointer form)."""
    from tilefish.hosts import rust

    op = declare_copy(fallback=rows())
    (t,), (p,) = compile_specs(op, arch=103), compile_specs(op, arch=90)
    write_manifest([fake_build(t, arch="sm_103"), fake_build(p, arch="sm_90")], tmp_path / "m" / "manifest.json", cubins=True)
    m = Manifest.load(tmp_path / "m", pin=Manifest.load(tmp_path / "m").digest)
    crate = tmp_path / "launchers-crate"
    (crate / "src").mkdir(parents=True)
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "launchers-check-by-arch"\nversion = "0.0.0"\nedition = "2024"\n\n'
        f'[dependencies]\ntilefish = {{ path = {json.dumps(str(REPO_ROOT / "crates" / "tilefish"))} }}\n\n[workspace]\n'
    )
    rust.emit(m, crate / "src" / "copy.rs", ops=[op])
    (crate / "src" / "lib.rs").write_text("pub mod copy;\n")
    env = {**os.environ, "RUSTFLAGS": "-D warnings", "CARGO_TARGET_DIR": str(REPO_ROOT / "target" / "launchers-check-schema3")}
    done = subprocess.run(["cargo", "check", "--manifest-path", str(crate / "Cargo.toml"), "--color", "never"], capture_output=True, text=True, env=env, timeout=900)
    assert done.returncode == 0, done.stderr[-4000:]
