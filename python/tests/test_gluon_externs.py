"""Gluon kernels as externs (``@tilefish.extern(gluon=...)``), with no Triton and no GPU.

A Gluon kernel (``triton.experimental.gluon``: ``@gluon.jit``, layouts written out) is compiled by Gluon's frontend, ``GluonASTSource``,
whose IR is TTGIR: the module carries ``ttg.num-warps`` and ``ttg.target``, which Triton's own frontend does not set, so that frontend
cannot build it. The declaration says which frontend compiles a kernel (``gluon=`` in place of ``triton=``): it is in the declaration
digest and in every compile spec (``"language": "gluon"``), the build gives the spec's frontend to ``triton.compile``, and the identity of
the binary is the TTGIR, under its own tag and kind (``source_digest_kind`` ``"tilefish.gluon-ir.v1"``), in the manifest entry, the build
cache key and every replay. Everything else of a declaration (strided views, shared strides, consts, buckets, ``enable_fp_fusion``,
``index_bits``) is what it is for a ``triton.jit`` kernel.

``fake_compile`` stands in for both frontends and the compiler (``test_build_cache.py`` says how); ``test_gpu_gluon.py`` runs a real Gluon
kernel. The type of a stride scalar (``StrideOf(..., dtype=...)``) is tested in ``test_stride_dtype.py``.
"""

import importlib
import json
import linecache
import sys
import warnings

import fake_compile
import pytest
from fake_jit import FakeGluonJit, FakeJit

import tilefish
from tilefish import (
    Dim,
    Manifest,
    Out,
    StrideOf,
    Strided,
    Tensor,
    TilefishBuildError,
    TilefishCacheWarning,
    TilefishDeclarationError,
    TilefishManifestError,
    TilefishSpecError,
    _cache,
    _ir,
    const,
    f32,
    i32,
    runtime,
)
from tilefish.aot import CompileSpec, build, compile_specs, declaration_digest, write_manifest

R = Dim("R")
Cb = Dim("Cb", buckets=[128, 256])
ARCH = 103
REF = "gl_kernels:row_dot"


def declare(ref=REF, *, kind="gluon", name="row_dot", **kw):
    """``y[r] = SCALE * sum(a[r, :] * b[r, :])`` over two views read with one row stride: the declaration of a layout-pinned row reduction."""
    options = dict(num_warps=4, grid=lambda R: (tilefish.cdiv(R, 4),), name=name, index_bits=64, enable_fp_fusion=False)
    options.update(kw)

    @tilefish.extern(**{kind: ref}, **options)
    def row_dot(
        a: Strided[Tensor[R, Cb, f32]],
        b: Strided[Tensor[R, Cb, f32]],
        y: Out[Tensor[R, f32]],
        s_row: StrideOf(("a", "b"), 0, dtype=i32),
        n_rows: R,
        C: const = Cb,
        SCALE: const[f32] = 0.5,
    ): ...

    return row_dot


# -- the declaration ----------------------------------------------------------------------------------------------------------------------------


def test_a_gluon_declaration_is_the_triton_one_with_its_language():
    gluon, triton = declare(), declare(kind="triton")
    assert (gluon.language, triton.language) == ("gluon", "triton") and gluon.triton == triton.triton == REF
    assert declaration_digest(gluon) != declaration_digest(triton)  # which frontend compiles it is part of what the declaration says
    pairs = list(zip(compile_specs(gluon), compile_specs(triton)))
    assert [s.stem for s, _ in pairs] == ["row_dot.Cb128", "row_dot.Cb256"]  # one binary per bucket
    for g, t in pairs:
        assert g.language == "gluon" and g.ast_source_inputs().language == "gluon" and t.ast_source_inputs().language == "triton"
        dg, dt = g.to_dict(), t.to_dict()
        assert dg.pop("language") == "gluon" and "language" not in dt  # a triton.jit kernel's spec is as it was
        assert dg.pop("declaration_digest") != dt.pop("declaration_digest")
        assert dg == dt  # everything else is the same: views, the shared stride, the consts, the bucket, the options
    g = pairs[1][0]
    assert g.signature == {"a": "*fp32", "b": "*fp32", "y": "*fp32", "s_row": "i32", "n_rows": "i32", "C": "constexpr", "SCALE": "constexpr"}
    assert g.constants == {(5,): 256, (6,): 0.5} and g.ast_source_inputs().options == {"num_warps": 4, "num_stages": 3, "enable_fp_fusion": False}
    (stride,) = [a for a in g.args if a.name == "s_row"]
    assert stride.stride_of == (("a", 0), ("b", 0)) and stride.type == "i32"  # an i32 stride, although the kernel indexes in i64
    assert not [p for p in g.preconditions if p.kind in ("max_elements", "max_extent")]  # index_bits=64: no limit on the offsets


def test_a_gluon_spec_round_trips_and_is_verified_against_its_declaration_only():
    gluon, triton = declare(), declare(kind="triton")
    for spec in compile_specs(gluon):
        back = CompileSpec.from_json(spec.to_json())
        assert back.language == "gluon" and not back.verified
        with pytest.raises(TilefishSpecError):
            back.verify_against(triton)
        back.verify_against(gluon)
        assert back.verified and back.ast_source_inputs().language == "gluon"


@pytest.mark.parametrize("value", ["triton", "Gluon", 1, None, True])
def test_a_spec_says_gluon_or_nothing(value):
    doc = compile_specs(declare())[0].to_dict()
    doc["language"] = value
    with pytest.raises(TilefishSpecError, match="language"):
        CompileSpec.from_dict(doc)


def test_a_prebuilt_spec_has_no_language(tmp_path):
    ptx = tmp_path / "k.ptx"
    ptx.write_text(".version 8.8\n.target sm_90\n.address_size 64\n.visible .entry k(\n\t.param .u64 x\n)\n{\n\tret;\n}\n")

    @tilefish.extern(ptx=ptx, symbol="k", block=(32,), grid=lambda R: (R,))
    def k(x: Tensor[R, f32]): ...

    doc = compile_specs(k)[0].to_dict()
    doc["language"] = "gluon"
    with pytest.raises(TilefishSpecError):
        CompileSpec.from_dict(doc)


def gluon_object(name="row_dot"):
    def row_dot(a, b, y, s_row, n_rows, C, SCALE):
        return a

    row_dot.__module__ = "gl_kernels"
    row_dot.__name__ = name
    return row_dot


def test_a_kernel_object_says_what_it_is_and_the_wrong_argument_is_refused():
    fn = gluon_object()
    assert declare(FakeGluonJit(fn)).language == "gluon"  # the object's parameters are checked as they are for triton=
    with pytest.raises(TilefishDeclarationError, match=r"is a Gluon kernel .*declare it with gluon="):
        declare(FakeGluonJit(fn), kind="triton")
    with pytest.raises(TilefishDeclarationError, match=r"gluon= takes a Gluon kernel .*declare it with triton="):
        declare(FakeJit(fn))
    assert declare(FakeJit(fn), kind="triton").language == "triton"

    def swapped(a, b, y, n_rows, s_row, C, SCALE):
        return a

    swapped.__module__ = "gl_kernels"
    with pytest.raises(TilefishDeclarationError, match="do not match the parameters"):  # the stub is checked against a Gluon kernel too
        declare(FakeGluonJit(swapped))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(gluon="m:f", triton="m:f"), "one of triton=, gluon=, ptx= and cubin=, got triton and gluon"),
        (dict(gluon="m:f", ptx="x.ptx"), "one of triton=, gluon=, ptx= and cubin=, got gluon and ptx"),
        (dict(gluon="m:f", symbol="k"), "symbol= is for a prebuilt binary"),
        (dict(gluon="m:f", block=(32,)), "block= is for a prebuilt binary"),
        (dict(gluon="m:f", shared=16), "shared= is for a prebuilt binary"),
        (dict(gluon="not a ref"), r"gluon='not a ref' must look like 'module:function'"),
        (dict(gluon=3), "gluon= takes a 'module:function' string or a kernel object"),
        (dict(gluon="m:f", num_warps=3), "num_warps must be a power of two"),
    ],
)
def test_a_gluon_declaration_is_checked(kwargs, message):
    with pytest.raises(TilefishDeclarationError, match=message):
        tilefish.extern(grid=lambda R: (R,), **kwargs)


def test_the_graph_node_names_the_frontend():
    assert declare().node_attrs() == {"gluon": REF} and declare(kind="triton").node_attrs() == {"triton": REF}


def test_a_plan_takes_a_gluon_step():
    Rb = Dim("Rb", buckets=[8, 24])

    @tilefish.extern(gluon=REF, num_warps=4, grid=lambda Rb: (tilefish.cdiv(Rb, 4),), index_bits=64)
    def row_dot(
        a: Strided[Tensor[Rb, 128, f32]],
        b: Strided[Tensor[Rb, 128, f32]],
        y: Out[Tensor[Rb, f32]],
        s_row: StrideOf(("a", "b"), 0, dtype=i32),
        n_rows: Rb,
        C: const = 128,
        SCALE: const[f32] = 1.0,
    ): ...

    @tilefish.graph
    def halves(x: Tensor[Rb, 256, f32]) -> Tensor[Rb, f32]:
        pair = x.reshape(Rb, 2, 128)
        return row_dot(pair[:, 0], pair[:, 1])

    (step,) = tilefish.plan(halves).steps(Rb=24)
    assert step["entry"] == "row_dot" and step["kind"] == "kernel"
    assert [(a["name"], a.get("view")) for a in step["args"] if a["name"] in ("a", "b")] == [
        ("a", {"shape": [24, 128], "strides": [256, 1], "offset": 0}),
        ("b", {"shape": [24, 128], "strides": [256, 1], "offset": 128}),
    ]


# -- the build, the cache and the manifest (the fake frontends and compiler) ------------------------------------------------------------------

KERNELS = '''from fake_jit import gluon_jit, jit

LAYOUT = [1, 4]


@gluon_jit
def row_dot(a, b, y, s_row, n_rows, C, SCALE):
    return (a * s_row + b, LAYOUT, C, SCALE)


@jit
def row_dot_triton(a, b, y, s_row, n_rows, C, SCALE):
    return (a * s_row + b, LAYOUT, C, SCALE)
'''


@pytest.fixture
def world(tmp_path, monkeypatch):
    compiler = fake_compile.install(monkeypatch)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    directory = tmp_path / "project"
    directory.mkdir()
    (directory / "gl_kernels.py").write_text(KERNELS)
    monkeypatch.syspath_prepend(str(directory))
    sys.modules.pop("gl_kernels", None)
    yield directory, compiler
    sys.modules.pop("gl_kernels", None)


def load(directory, attr="row_dot"):
    """The kernel as a new process imports it from the file as it is now."""
    sys.modules.pop("gl_kernels", None)
    importlib.invalidate_caches()
    linecache.clearcache()
    return getattr(importlib.import_module("gl_kernels"), attr)


def frontend(fn, spec):
    inputs = spec.ast_source_inputs()
    return _ir.ttir_text(fn, inputs, arch=ARCH, options=inputs.options)


def test_a_gluon_kernel_is_built_by_gluons_frontend_and_identified_by_its_ttgir(world, tmp_path):
    directory, compiler = world
    spec = compile_specs(declare())[0]
    built = build(spec, arch=ARCH)
    assert compiler.languages == ["gluon"]  # GluonASTSource went to triton.compile
    text = frontend(load(directory), spec)
    assert text.startswith('module attributes {"ttg.num-warps" = 4, ttg.target = "cuda:103"} ttgir ')  # num_warps is in a Gluon kernel's IR
    assert built.ir_digest == _ir.ir_digest(text, "gluon") != _ir.ir_digest(text, "triton")  # its own tag: never equal to a TTIR digest
    write_manifest([built], tmp_path / "out" / "manifest.json", cubins=True)
    entry = json.loads((tmp_path / "out" / "manifest.json").read_text())["kernels"][0]
    assert entry["kind"] == "triton" and entry["source_digest"] == built.ir_digest  # Triton compiles it: launched like any Triton kernel
    assert entry["source_digest_kind"] == _ir.GLUON_IR_DIGEST_KIND == "tilefish.gluon-ir.v1"
    assert entry["abi"]["scratch_slots"] == 2 and [p["type"] for p in entry["abi"]["params"]][3] == "i32"


def test_the_frontend_of_another_language_is_refused_before_anything_runs(world):
    directory, compiler = world
    gluon_of_triton = compile_specs(declare("gl_kernels:row_dot_triton"))[0]
    with pytest.raises(TilefishBuildError, match=r"'row_dot.Cb128' \(gl_kernels:row_dot_triton\) is declared a Gluon kernel .* is a triton.jit function"):
        build(gluon_of_triton, arch=ARCH)
    triton_of_gluon = compile_specs(declare(kind="triton"))[0]
    gluon_but_triton = r"is a Gluon kernel \(@gluon.jit\), but it is declared a triton.jit kernel: declare it with @tilefish.extern\(gluon="
    with pytest.raises(TilefishBuildError, match=gluon_but_triton):
        build(triton_of_gluon, arch=ARCH)
    with pytest.raises(TilefishBuildError, match=gluon_but_triton):  # a jit run and a replay compute the digest first: they refuse it there
        runtime._ir_digest_of(triton_of_gluon, load(directory), ARCH, {})
    assert compiler.count == 0 and compiler.frontends == 0  # neither frontend ran, nothing was compiled or cached


def test_the_cache_keys_a_gluon_build_by_its_kind_and_checks_it(world):
    directory, compiler = world
    spec = compile_specs(declare())[0]
    first = build(spec, arch=ARCH)
    assert build(spec, arch=ARCH).sha256_code == first.sha256_code and compiler.count == 1  # a hit
    key = dict(arch=ARCH, ir_digest=first.ir_digest, triton_version="3.5.1", ptxas_version="v", options={}, environment={}, files={})
    assert _cache.cache_key(**key, language="gluon") != _cache.cache_key(**key) == _cache.cache_key(**key, language="triton")
    (path,) = list(_cache.default_cache().directory.rglob("*.json"))
    doc = json.loads(path.read_text())
    assert doc["ir_kind"] == "tilefish.gluon-ir.v1" and doc["ir_digest"] == first.ir_digest
    doc["ir_kind"] = _ir.IR_DIGEST_KIND
    path.write_text(json.dumps(doc))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build(spec, arch=ARCH)
    assert compiler.count == 2 and any("of kind 'tilefish.triton-ir.v1'" in str(w.message) for w in caught if w.category is TilefishCacheWarning)


def export(directory, tmp_path, spec, name="export", attr="row_dot"):
    built = build(spec, arch=ARCH, fn=load(directory, attr))
    out = tmp_path / name
    write_manifest([built], out / "manifest.json", cubins=True)
    manifest = Manifest.load(out)
    return manifest, built


def test_a_changed_layout_is_a_stale_binary_pinned_or_not(world, tmp_path):
    """The edit, in-process with the fake frontend (``test_gpu_gluon.py`` does it in real processes): the layout of the kernel changes in its
    file; a process that imports the edited file computes another TTGIR digest, and the manifest of the old kernel is refused, with a pin too."""
    directory, _ = world
    spec = compile_specs(declare())[0]
    manifest, built = export(directory, tmp_path, spec)
    held = runtime._ir_digest_of(spec, load(directory), ARCH, {})
    assert held == built.ir_digest
    assert manifest.built_kernel(spec, "sm_103", ir_digest=held, ir_triton="3.5.1").sha256_code == built.sha256_code
    text = (directory / "gl_kernels.py").read_text()
    (directory / "gl_kernels.py").write_text(text.replace("LAYOUT = [1, 4]", "LAYOUT = [1, 2]"))
    edited = runtime._ir_digest_of(spec, load(directory), ARCH, {})
    assert edited != held
    for pinned in (False, True):
        with pytest.raises(TilefishManifestError, match=r"built from Gluon IR with digest .*its layouts.*stale"):
            manifest.built_kernel(spec, "sm_103", ir_digest=edited, ir_triton="3.5.1", pinned=pinned)
    with pytest.raises(TilefishManifestError, match="cannot tell whether the binary"):  # nothing to compare with: only a pin vouches for it
        manifest.built_kernel(spec, "sm_103", ir_digest=None, ir_triton=None)
    assert manifest.built_kernel(spec, "sm_103", ir_digest=None, ir_triton=None, pinned=True).ir_digest == held


def test_an_entry_of_the_other_frontend_is_refused_pinned_or_not(world, tmp_path):
    directory, _ = world
    gluon, triton = compile_specs(declare())[0], compile_specs(declare("gl_kernels:row_dot_triton", kind="triton"))[0]
    manifest, built = export(directory, tmp_path, gluon)
    for pinned in (False, True):
        with pytest.raises(TilefishManifestError, match=r"built from a Gluon kernel .* now declares a Triton kernel .* stale"):
            manifest.built_kernel(triton, "sm_103", ir_digest=built.ir_digest, ir_triton="3.5.1", pinned=pinned)
    manifest, built = export(directory, tmp_path, triton, "export-triton", "row_dot_triton")
    assert json.loads((tmp_path / "export-triton" / "manifest.json").read_text())["kernels"][0]["source_digest_kind"] == _ir.IR_DIGEST_KIND
    with pytest.raises(TilefishManifestError, match=r"built from a Triton kernel .* now declares a Gluon kernel"):
        manifest.built_kernel(gluon, "sm_103", ir_digest=None, ir_triton=None, pinned=True)
