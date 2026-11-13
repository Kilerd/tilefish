"""The Rust emitter on manifest schema 3: strided views, device scalars, prebuilt binaries with fixed arguments, and pinned cuBLASLt GEMMs.

``tilefish.hosts.rust.emit`` writes one launcher per stem for everything the manifest can describe (``docs/manifest-schema-3.md``): the
snapshots are in ``tests/golden/`` (regenerate with ``UPDATE_GOLDEN=1 pytest tests/test_rust_host_schema3.py`` and review the diff). The manifests here are
built from fake binaries, so nothing needs a GPU, Triton or the Rust crate; ``cargo check`` of the emitted modules against the crate is the last
test (it runs once the crate in the checkout reads schema 3 and has ``Gemm``). What the emitter refuses is checked field by field, like
``test_rust_host.py`` does for the fields of a schema 2 entry: nothing in a manifest reaches the generated source unvalidated, and an entry that the
reader of the crate refuses (``fixed`` on a Triton kernel, ``strides`` on a pointer that is written ...) is refused here too.
"""

import copy
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess

import pytest
from binary_fakes import PTX_TEXT, SM90, fake_binary_build, make_ptx_op, write_ptx
from built_fakes import fake_build
from conftest import REPO_ROOT
from elf_builder import arch_header, cubin_like
from host_ops import gpu_kernels_p1, gpu_kernels_p4
from test_library_ops import ALGO, SMS, VERSION, gemm, gemm_bias
from test_rust_host import check_golden, needs_new_crate, pinned

import tilefish
from tilefish import Dim, Out, StrideOf, Strided, Tensor, TilefishBuildError, TilefishManifestError, bf16, const, f32, i32, i64, runtime
from tilefish._elf import code_digest
from tilefish._library import BuiltLibrary, library_specs
from tilefish.aot import Manifest, write_manifest
from tilefish.hosts import rust

CRATE = REPO_ROOT / "crates" / "tilefish"

R = Dim("R")
EVIL = 'x\n}\n}\npub fn pwned() { std::process::exit(0) }\nstatic S: &str = "'
ALGO2 = "fedcba9876543210" * 8


# -- manifests of fake builds ---------------------------------------------------------------------------------------------------------------


def build_manifest(directory, ops=(), library=(), arch="sm_90"):
    """A pinned manifest directory of fake builds (and library entries) of ``ops``: Triton kernels and prebuilt binaries."""
    built = []
    for op in ops:
        target = runtime._target(op)
        extern = target.launch_extern()
        for spec in target.compile_specs():
            built.append(fake_binary_build(spec, extern.binary, arch=arch) if extern.binary is not None else fake_build(spec, arch=arch))
    write_manifest(built, pathlib.Path(directory) / "manifest.json", cubins=True, library=list(library))
    return pinned(Manifest.load(directory))


def pins(op, arch="sm_103", algo=ALGO, version=VERSION, sm_count=SMS):
    """The library entries of every bucket of ``op``, as pinned on ``arch`` (a device of ``sm_count`` SMs) with ``algo`` by cuBLASLt ``version``."""
    return [BuiltLibrary(spec, arch, algo, version, sm_count) for spec in library_specs(op)]


def docs(source):
    """The doc comments of a module as one text: a paragraph that was wrapped over several lines is one run of words again."""
    return re.sub(r"\n\s*///[ ]?", " ", source)


@pytest.fixture(scope="module")
def p1():
    return gpu_kernels_p1()


@pytest.fixture
def prebuilt(tmp_path):
    """Three prebuilt externs over the stand-in PTX (and a cubin): a float fixed argument, a 64-bit stride with a fixed i32, a 2-D block with dynamic shared memory."""
    ptx = write_ptx(tmp_path)
    cubin = tmp_path / "k.sm_90.cubin"
    cubin.write_bytes(cubin_like(shared=None, **SM90))
    axpy = make_ptx_op(ptx)

    @tilefish.extern(ptx=ptx, symbol="tf_k", block=(64,), grid=lambda R: (R,), index_bits=64, name="strided_copy")
    def strided_copy(x: Strided[Tensor[R, 64, bf16]], y: Out[Tensor[R, 64, bf16]], *, cols: const[i32] = 64, x_s0: StrideOf("x", 0)): ...

    @tilefish.extern(cubin={"sm_90": cubin}, symbol="tf_k", block=(32, 4), shared=1024, grid=lambda R: (R, 2), name="tile_cubin")
    def tile_cubin(x: Tensor[R, 128, f32], y: Out[Tensor[R, 128, f32]], scale: const[f32] = 0.5, stride: const[i64] = 1 << 40): ...

    return [axpy, strided_copy, tile_cubin]


# -- snapshots ------------------------------------------------------------------------------------------------------------------------------


def test_launchers_for_strided_views_and_device_scalars(p1, tmp_path):
    ops = [p1.head_copy, p1.head_copy16, p1.head_copy64, p1.softmax3d, p1.masked_rows, p1.scale_by, p1.scale_flat]
    manifest = build_manifest(tmp_path / "m", ops)
    source = rust.render(manifest, ops=ops)
    check_golden("rust_strided.rs", source)
    # a strided pointer and a device scalar are parameters like any other: the stride scalars and the offset are the crate's to derive
    assert "x: TensorRef<'a, Bf16>,\n        y: TensorMut<'a, Bf16>,\n    ) -> Result<(), Error> {\n        self.kernel.launch(stream, &mut [x.into_arg(), y.into_arg()])" in source
    assert "n_tok: TensorRef<'a, I32>" in source and "s: TensorRef<'a, F32>" in source and "b: TensorRef<'a, F32>" in source
    for launch in source.split("pub fn launch<'a>(")[1:]:
        params = launch.split(") -> Result")[0]
        assert "_s0" not in params and "_s1" not in params and "offset" not in params
    # the shape of a device scalar is empty
    assert 'pub const SHAPES: &\'static [&\'static [&\'static str]] = &[&["R", "128"], &["R", "128"], &[]];' in source
    # the strided pointer is documented as a view, with the alignment its binary was compiled for
    text = docs(source)
    assert "- `x`: in, bf16, `[S, 8, 64]`, a strided view (strides `[x_s0, x_s1, 1]`)" in text
    assert "- `x`: 2 bytes" in text and "- `x`: 16 bytes" in text and "- `x`: 4 bytes" in text  # Strided[...], Strided[..., 16], and the f32 of softmax3d
    assert "(`max_extent`: the binary indexes with 32 bits)" in text and "(the binary indexes with 64 bits: its views have no `max_extent`)" in text
    assert "- `x_s0` is the stride of axis 0 of `x`, in elements (i64)" in text  # head_copy64's strides are 64-bit
    assert "A device scalar is a `TensorRef` over one element" in text and "a device scalar (one element: the shape `[]` or `[1]`)" in text
    assert "TensorRef::from_cuda_strided" in text


def test_launchers_for_prebuilt_binaries(prebuilt, tmp_path):
    manifest = build_manifest(tmp_path / "m", prebuilt)
    source = rust.render(manifest, ops=prebuilt)
    check_golden("rust_binary.rs", source)
    text = docs(source)
    assert "pub struct Axpy " in source and "pub struct StridedCopy " in source and "pub struct TileCubin " in source
    assert "PTX that ptxas assembled, not compiled by Triton" in text and "a cubin used as it is, not compiled by Triton" in text
    assert "there are no Triton scratch slots" in text
    # a fixed argument is documented, with its type, and is not a parameter of the launch
    assert "`alpha` = 2.0 (f32), `scratch` = 0 (i64)" in text and "`cols` = 64 (i32)" in text and "`scale` = 0.5 (f32), `stride` = 1099511627776 (i64)" in text
    assert "they are not parameters of `launch`" in text
    for launch in source.split("pub fn launch<'a>(")[1:]:
        params = launch.split(") -> Result")[0]
        assert "alpha" not in params and "cols" not in params and "scale" not in params and "stride" not in params and "scratch" not in params
    # the dynamic shared memory and the 2-D block of a hand-written kernel are in the doc
    assert "Grid `(R, 2)`, block 32x4x1 threads, 1024 bytes of dynamic shared memory." in text
    assert "Grid `(cdiv(n, 128),)`, block 128 threads, 0 bytes of dynamic shared memory." in text
    # the 64-bit stride is a runtime scalar of the binary, which the launch derives
    assert "- `x_s0` is the stride of axis 0 of `x`, in elements (i64)" in text
    assert "the binary indexes with 64 bits" in text  # strided_copy declared index_bits=64
    # a launch passes exactly what the ABI says: the generated code never writes a scratch slot (the crate does not append any to a prebuilt binary)
    assert "scratch_slots" not in source and "global_scratch" not in source


def test_launchers_for_pinned_gemms(tmp_path):
    library = [*pins(gemm), *pins(gemm_bias), BuiltLibrary(library_specs(gemm)[0], "sm_90", ALGO2, VERSION - 100, 132)]
    manifest = build_manifest(tmp_path / "m", (), library)
    source = rust.render(manifest, ops=[gemm, gemm_bias])
    check_golden("rust_library.rs", source)
    assert "use tilefish::{Arch, Bf16, CUstream, CudaContext, CudaStream, Error, Gemm, Manifest, TensorMut, TensorRef, U8};" in source  # no Kernel: there is none
    for struct in ("GemmM32", "GemmM64", "GemmBiasKb1024M32", "GemmBiasKb1024M64", "GemmBiasKb2048M32", "GemmBiasKb2048M64"):
        assert f"pub struct {struct} {{\n    gemm: Gemm,\n}}" in source
    # the tensors, in the order a, b, [bias], out, then the workspace; both kinds
    assert "a: TensorRef<'a, Bf16>,\n        b: TensorRef<'a, Bf16>,\n        out: TensorMut<'a, Bf16>,\n        workspace: TensorMut<'a, U8>,\n" in source
    assert "a: TensorRef<'a, Bf16>,\n        b: TensorRef<'a, Bf16>,\n        bias: TensorRef<'a, Bf16>,\n        out: TensorMut<'a, Bf16>,\n        workspace: TensorMut<'a, U8>,\n" in source
    assert "self.gemm.launch(stream, &mut [a.into_arg(), b.into_arg(), bias.into_arg(), out.into_arg(), workspace.into_arg()])" in source
    assert "unsafe { self.gemm.launch_raw(stream, &[a.into_arg(), b.into_arg(), out.into_arg(), workspace.into_arg()]) }" in source
    assert "let gemm = manifest.gemm(ctx, Self::STEM, Self::BUCKETS, Arch::of_context(ctx)?)?;" in source
    # what the entry says: the problem, the workspace the algorithm was pinned under and the cuBLASLt version of each architecture's pin
    assert "pub const M: usize = 32;" in source and "pub const N: usize = 2048;" in source and "pub const K: usize = 1024;" in source and "pub const BIAS: bool = false;" in source
    assert "pub const WORKSPACE_BYTES: usize = 8519680;" in source and "pub const WORKSPACE_BYTES: usize = 1048576;" in source
    assert 'pub const PINNED_WITH: &\'static [(&\'static str, u64)] = &[("sm_103", 120804), ("sm_90", 120704)];' in source  # one struct for the two architectures
    assert "Pinned with: sm_103: cuBLASLt 12.8.4 (120804) on 30 SMs, sm_90: cuBLASLt 12.7.4 (120704) on 132 SMs." in docs(source)
    assert 'pub const SHAPES: &\'static [&\'static [&\'static str]] = &[&["M", "Kb"], &["2048", "Kb"], &["2048"], &["M", "2048"]];' in source


def test_a_manifest_of_kernels_and_library_entries_embeds_the_cubins_and_no_library_file(p1, prebuilt, tmp_path):
    ops = [p1.head_copy, prebuilt[0]]
    manifest = build_manifest(tmp_path / "m", ops, pins(gemm)[:1])
    out = tmp_path / "launchers" / "launchers.rs"
    source = rust.emit(manifest, out, ops=[*ops, gemm], embed=True).read_text()
    check_golden("rust_schema3_embed.rs", source)
    cubins = source.split("pub static CUBINS")[1].split("];")[0]
    assert cubins.count("include_bytes!") == 2 and "axpy.sm_90.cubin" in cubins and "head_copy.sm_90.cubin" in cubins and "gemm" not in cubins
    assert "Manifest::from_embedded(MANIFEST_JSON, CUBINS, Pin::Digest(MANIFEST_DIGEST))" in source
    assert "use tilefish::{Arch, Bf16, CUstream, CudaContext, CudaStream, Error, F32, Gemm, Kernel, Manifest, Pin, TensorMut, TensorRef, U8};" in source


def test_a_library_only_embedded_manifest_has_no_cubins(tmp_path):
    manifest = build_manifest(tmp_path / "m", (), pins(gemm))
    source = rust.emit(manifest, tmp_path / "out.rs", embed=True).read_text()
    assert "pub static CUBINS: &[(&str, &[u8])] = &[\n];" in source and "Kernel" not in source.split("pub static")[0].split("use tilefish::")[1]


# -- determinism ------------------------------------------------------------------------------------------------------------------------------


def test_the_output_does_not_depend_on_the_order_of_the_entries_of_either_list(p1, prebuilt, tmp_path):
    ops = [p1.head_copy, p1.masked_rows, *prebuilt]
    manifest = build_manifest(tmp_path / "m", ops, [*pins(gemm), *pins(gemm_bias)])
    first = rust.render(manifest, ops=[*ops, gemm, gemm_bias])
    assert rust.render(manifest, ops=[gemm_bias, *reversed(ops), gemm]) == first  # nor on the order of the declarations
    doc = json.loads(manifest.path.read_text())
    doc["kernels"].reverse()
    doc["library"].reverse()
    other = tmp_path / "reordered"
    shutil.copytree(manifest.directory, other)
    (other / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n")
    reordered = pinned(Manifest.load(other))
    assert reordered.digest == manifest.digest  # the digest does not depend on the order either
    assert rust.render(reordered, ops=[*ops, gemm, gemm_bias]) == first
    a, b = tmp_path / "a.rs", tmp_path / "b.rs"
    rust.emit(manifest, a, ops=[*ops, gemm, gemm_bias])
    rust.emit(manifest, b, ops=[*ops, gemm, gemm_bias])
    assert a.read_bytes() == b.read_bytes() == first.encode("utf-8")


# -- declarations -----------------------------------------------------------------------------------------------------------------------------


def test_a_library_declaration_that_is_not_what_the_manifest_was_pinned_from_is_refused(tmp_path):
    manifest = build_manifest(tmp_path / "m", (), pins(gemm))
    wide = tilefish.library("gemm", acc=f32, impl="cublaslt.gemm", workspace_bytes=4096)(gemm.fn)
    with pytest.raises(TilefishManifestError, match=r"the declaration of 'gemm' is not what the manifest's library entry 'gemm.M32' \(sm_103\) was built from: first difference at workspace_bytes"):
        rust.render(manifest, ops=[wide])
    assert "pub struct GemmM32 " in rust.render(manifest, ops=[gemm_bias])  # a declaration of an operator the manifest does not have is not an error


def test_a_prebuilt_declaration_that_is_not_what_the_manifest_was_built_from_is_refused(prebuilt, tmp_path):
    manifest = build_manifest(tmp_path / "m", prebuilt)
    other = make_ptx_op(write_ptx(tmp_path, "other.ptx", PTX_TEXT + "// the same code, another file\n"))
    with pytest.raises(TilefishManifestError, match="is not what the manifest's binary 'axpy' was built from"):
        rust.render(manifest, ops=[other])


def test_a_stem_is_a_kernel_or_a_library_entry_not_both(tmp_path):
    """The namespace of ``(stem, arch)`` is shared: the loader refuses the same stem and architecture twice, and the emitter a stem that is both for two."""

    @tilefish.extern(triton="kernels:copy", grid=lambda R: (R,), name="g")
    def kernel(x: Tensor[R, 4, f32], y: Out[Tensor[R, 4, f32]]): ...

    @tilefish.library("g", acc=f32, impl="cublaslt.gemm")
    def g(a: Tensor[32, 1024, bf16], b: Tensor[2048, 1024, bf16]) -> Tensor[32, 2048, bf16]: ...

    (kspec,) = runtime._target(kernel).compile_specs()
    (lspec,) = library_specs(g)
    assert kspec.stem == lspec.stem == "g"
    write_manifest([fake_build(kspec, arch="sm_90")], tmp_path / "m" / "manifest.json", cubins=True, library=[BuiltLibrary(lspec, "sm_103", ALGO, VERSION, SMS)])
    manifest = pinned(Manifest.load(tmp_path / "m"))
    with pytest.raises(TilefishManifestError, match=r"\['g'\] are both kernels and library entries"):
        rust.render(manifest)
    with pytest.raises(TilefishBuildError, match="one build per .stem, arch., kernels and library entries together"):  # nor can a manifest be written so
        write_manifest([fake_build(kspec, arch="sm_103")], tmp_path / "same" / "manifest.json", cubins=True, library=[BuiltLibrary(lspec, "sm_103", ALGO, VERSION, SMS)])
    shutil.copytree(manifest.directory, tmp_path / "edited")
    doc = json.loads((tmp_path / "edited" / "manifest.json").read_text())
    doc["kernels"][0]["arch"] = "sm_103"  # the kernel and the library entry are now for the same architecture
    sm103 = cubin_like(**arch_header("sm_103"))  # (and the cubin is one for it: the reader holds a cubin to the architecture of its entry)
    (tmp_path / "edited" / doc["kernels"][0]["cubin"]).write_bytes(sm103)
    doc["kernels"][0].update(sha256=hashlib.sha256(sm103).hexdigest(), sha256_code=code_digest(sm103))
    (tmp_path / "edited" / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="lists 'g' for sm_103 twice"):
        Manifest.load(tmp_path / "edited")


def test_a_struct_named_like_an_import_gets_op_after_it():
    # `Gemm` is imported from the crate, so a library entry (or a kernel) whose stem would give that struct name gets `Op` after it
    assert rust._camel("gemm") == "GemmOp" and rust._camel("gemm.M32") == "GemmM32" and rust._camel("gemm_bias.Kb1024.M32") == "GemmBiasKb1024M32"
    # the names of the gate's GEMMs (host_ops.library_ops): the struct of a stem is its words with their first letters in capitals
    assert rust._camel("gemm_1024_1024.M320") == "Gemm10241024M320" and rust._camel("gemm_3072_4096_bias.M32") == "Gemm30724096BiasM32"


def test_a_tensor_called_workspace_would_clash_with_the_launchers_workspace(tmp_path):
    manifest = build_manifest(tmp_path / "m", (), pins(gemm_bias)[:1])
    entry = manifest.library_entries[0]
    entry["abi"]["params"][2]["name"] = "workspace"  # the bias, renamed in the ABI and in the precondition that is about it
    for pre in entry["preconditions"]:
        if pre.get("arg") == "bias":
            pre["arg"] = "workspace"
    with pytest.raises(TilefishManifestError, match="one of them is 'workspace'"):
        rust.render(manifest)


def test_the_library_entries_of_a_stem_must_agree_across_architectures(tmp_path):
    spec = library_specs(gemm)[0]
    library = [BuiltLibrary(spec, "sm_103", ALGO, VERSION, SMS), BuiltLibrary(spec, "sm_90", ALGO2, VERSION - 100, 132)]
    manifest = build_manifest(tmp_path / "m", (), library)
    assert rust.render(manifest).count("pub struct GemmM32 ") == 1
    doc = json.loads(manifest.path.read_text())
    doc["library"][0]["workspace_bytes"] = 4096
    (tmp_path / "m" / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="disagree about 'workspace_bytes'"):
        rust.render(pinned(Manifest.load(tmp_path / "m")))


# -- what the emitter refuses -----------------------------------------------------------------------------------------------------------------


def _entry(manifest, stem):
    return next(a.entry for a in manifest.artifacts if a.stem == stem)


def _param(entry, name):
    return next(p for p in entry["abi"]["params"] if p["name"] == name)


def _drop(entry, **match):
    entry["preconditions"] = [p for p in entry["preconditions"] if not all(p.get(k) == v for k, v in match.items())]


@pytest.fixture(scope="module")
def schema3_manifest(tmp_path_factory):
    """A pinned manifest with a strided kernel, a 64-bit one, one whose views share their strides, a device scalar, a prebuilt binary with fixed
    arguments and a GEMM (for sm_90 and sm_103)."""
    tmp = tmp_path_factory.mktemp("schema3")
    p1 = gpu_kernels_p1()
    ptx = write_ptx(tmp)
    axpy = make_ptx_op(ptx)
    ops = [p1.head_copy, p1.head_copy16, p1.head_copy64, gpu_kernels_p4().qkv_sum, p1.masked_rows, axpy]
    return build_manifest(tmp / "m", ops, pins(gemm)[:1])


KERNEL_FIELDS = {
    # strided pointers
    "strides without offset": ("head_copy", lambda e: _param(e, "x").pop("offset")),
    "offset without strides": ("head_copy", lambda e: _param(e, "x").pop("strides")),
    "strides on a pointer that is written": ("head_copy", lambda e: _param(e, "y").update(strides=["y_s0", "y_s1", 1], offset="y_offset")),
    "strides of the wrong length": ("head_copy", lambda e: _param(e, "x").update(strides=["x_s0", 1])),
    "no strides": ("head_copy", lambda e: _param(e, "x").update(strides=[])),
    "strides that are not a list": ("head_copy", lambda e: _param(e, "x").update(strides="x_s0")),
    "a fixed stride that is not 1": ("head_copy", lambda e: _param(e, "x").update(strides=["x_s0", "x_s1", 2])),
    "a fixed stride on another axis than the last": ("head_copy", lambda e: _param(e, "x").update(strides=["x_s0", 1, "x_s1"])),
    "a stride entry that is a bool": ("head_copy", lambda e: _param(e, "x").update(strides=["x_s0", "x_s1", True])),
    "a stride dimension that is not a name": ("head_copy", lambda e: _param(e, "x").update(strides=[EVIL, "x_s1", 1])),
    "a stride dimension with a unicode letter": ("head_copy", lambda e: _param(e, "x").update(strides=["x_sé", "x_s1", 1])),
    "an offset that is not a name": ("head_copy", lambda e: _param(e, "x").update(offset=EVIL)),
    "an offset that is a number": ("head_copy", lambda e: _param(e, "x").update(offset=0)),
    "a stride that no scalar carries": ("head_copy", lambda e: e["abi"]["params"].remove(_param(e, "x_s1"))),
    "a stride_of that names no pointer": ("head_copy", lambda e: _param(e, "x_s0")["stride_of"].update(arg="nothing")),
    "a stride_of that names a pointer that is not strided": ("head_copy", lambda e: _param(e, "x_s0")["stride_of"].update(arg="y")),
    "a stride_of axis out of range": ("head_copy", lambda e: _param(e, "x_s0")["stride_of"].update(axis=3)),
    "a negative stride_of axis": ("head_copy", lambda e: _param(e, "x_s0")["stride_of"].update(axis=-1)),
    "a stride_of axis that is a bool": ("head_copy", lambda e: _param(e, "x_s0")["stride_of"].update(axis=False)),
    "a stride_of with another key": ("head_copy", lambda e: _param(e, "x_s0")["stride_of"].update(extra=1)),
    "a stride_of that is not an object": ("head_copy", lambda e: _param(e, "x_s0").update(stride_of="x")),
    "a stride_of with a dimension that is not the stride's": ("head_copy", lambda e: _param(e, "x_s0").update(dim="x_s1")),
    "two scalars for one stride": ("head_copy", lambda e: _param(e, "x_s1")["stride_of"].update(axis=0)),
    "a stride_of together with a count_of": ("head_copy", lambda e: _param(e, "x_s0").update(count_of="x")),
    # a stride that several views share: stride_of is a list of two or more of them, each once, whose strides name the scalar
    "a shared stride_of of one view": ("qkv_sum", lambda e: _param(e, "s_row").update(stride_of=[{"arg": "q", "axis": 0}])),
    "a shared stride_of with a view twice": (
        "qkv_sum",
        lambda e: _param(e, "s_row").update(stride_of=[{"arg": "q", "axis": 0}, {"arg": "q", "axis": 0}, {"arg": "k", "axis": 0}, {"arg": "v", "axis": 0}]),
    ),
    "a shared stride_of with an item that is not an object": (
        "qkv_sum",
        lambda e: _param(e, "s_row").update(stride_of=[{"arg": "q", "axis": 0}, "k", {"arg": "v", "axis": 0}]),
    ),
    "a shared stride_of that leaves a view out": ("qkv_sum", lambda e: _param(e, "s_row").update(stride_of=[{"arg": "q", "axis": 0}, {"arg": "k", "axis": 0}])),
    "a shared stride_of of another axis": (
        "qkv_sum",
        lambda e: _param(e, "s_row").update(stride_of=[{"arg": "q", "axis": 0}, {"arg": "k", "axis": 1}, {"arg": "v", "axis": 0}]),
    ),
    "a shared stride_of that names the output": (
        "qkv_sum",
        lambda e: _param(e, "s_row").update(stride_of=[{"arg": "q", "axis": 0}, {"arg": "k", "axis": 0}, {"arg": "v", "axis": 0}, {"arg": "o", "axis": 0}]),
    ),
    "a scalar of a type that is not i32 or i64": ("head_copy", lambda e: _param(e, "x_s0").update(type="i16")),
    "a scalar type with a newline": ("head_copy", lambda e: _param(e, "x_s0").update(type="i32\n")),
    # preconditions
    "max_extent on a pointer that is not strided": ("head_copy", lambda e: e["preconditions"].append({"kind": "max_extent", "arg": "y", "index": 1, "value": 2**31 - 1})),
    "max_extent twice": ("head_copy", lambda e: e["preconditions"].append({"kind": "max_extent", "arg": "x", "index": 0, "value": 2**31 - 1})),
    "max_extent with another value": ("head_copy", lambda e: next(p for p in e["preconditions"] if p["kind"] == "max_extent").update(value=5)),
    "max_elements on a strided pointer": ("head_copy", lambda e: e["preconditions"].append({"kind": "max_elements", "arg": "x", "index": 0, "value": 2**31 - 1, "shape": ["S"]})),
    "an alignment that is neither 16 nor the element size": ("head_copy", lambda e: next(p for p in e["preconditions"] if p["kind"] == "pointer_alignment").update(bytes=8)),
    "an alignment of a dense pointer that is not 16": ("head_copy", lambda e: next(p for p in e["preconditions"] if p["kind"] == "pointer_alignment" and p["arg"] == "y").update(bytes=2)),
    "no alignment": ("head_copy", lambda e: _drop(e, kind="pointer_alignment", arg="y")),
    "an alignment twice": ("head_copy", lambda e: e["preconditions"].append({"kind": "pointer_alignment", "arg": "x", "index": 0, "bytes": 2})),
    "an alignment with the wrong index": ("head_copy", lambda e: next(p for p in e["preconditions"] if p["kind"] == "pointer_alignment").update(index=7)),
    "an alignment of a pointer that does not exist": ("head_copy", lambda e: e["preconditions"].append({"kind": "pointer_alignment", "arg": "z", "index": 9, "bytes": 16})),
    "a precondition of a kind nobody knows": ("head_copy", lambda e: e["preconditions"].append({"kind": "other"})),
    "a precondition kind with a newline": ("head_copy", lambda e: e["preconditions"].append({"kind": "bucket\n"})),
    # device scalars
    "a device scalar that is written": ("masked_rows", lambda e: _param(e, "n_tok").update(access="inout")),
    "a device scalar with a static size": ("masked_rows", lambda e: e["preconditions"].append({"kind": "static_size", "arg": "n_tok", "index": 2, "axis": 0, "value": 1})),
    "a device scalar with max_elements": ("masked_rows", lambda e: e["preconditions"].append({"kind": "max_elements", "arg": "n_tok", "index": 2, "value": 2**31 - 1, "shape": []})),
    "a device scalar that is strided": ("masked_rows", lambda e: _param(e, "n_tok").update(strides=[], offset="n_tok_offset")),
    "a device scalar that is aligned to 4": ("masked_rows", lambda e: next(p for p in e["preconditions"] if p["arg"] == "n_tok").update(bytes=4)),
    # scratch slots and kinds
    "a kind that is neither": ("head_copy", lambda e: e.update(kind="python")),
    "no kind": ("head_copy", lambda e: e.pop("kind")),
    "a Triton kernel with no scratch slots": ("head_copy", lambda e: (e["abi"].update(scratch=[]), e["abi"].update(scratch_slots=0))),
    "a Triton kernel whose slots are not the number": ("head_copy", lambda e: e["abi"].update(scratch_slots=3)),
    "scratch slots that are a bool": ("head_copy", lambda e: e["abi"].update(scratch_slots=True)),
    "no scratch_slots": ("head_copy", lambda e: e["abi"].pop("scratch_slots")),
    "a prebuilt binary with scratch slots": ("axpy", lambda e: (e["abi"].update(scratch=[{"name": "s", "type": "*i8", "value": 0}] * 2), e["abi"].update(scratch_slots=2))),
    "a prebuilt binary with a scratch list but no slots": ("axpy", lambda e: e["abi"].update(scratch=[{"name": "s"}])),
    "a prebuilt binary that says it is Triton": ("axpy", lambda e: e.update(kind="triton")),
    # fixed arguments
    "a fixed argument on a Triton kernel": ("head_copy", lambda e: e["abi"]["params"].append({"slot": 4, "index": 4, "name": "k", "kind": "fixed", "type": "i32", "ptx_type": "u32", "size": 4, "value": 1})),
    "a fixed argument of a type that is not allowed": ("axpy", lambda e: _param(e, "scratch").update(type="u8")),
    "a fixed i64 with a float": ("axpy", lambda e: _param(e, "scratch").update(value=1.0)),
    "a fixed i64 with a bool": ("axpy", lambda e: _param(e, "scratch").update(value=True)),
    "a fixed i64 out of range": ("axpy", lambda e: _param(e, "scratch").update(value=2**63)),
    "a fixed i32 out of range": ("axpy", lambda e: _param(e, "scratch").update(type="i32", value=2**31)),
    "a fixed i32 below its range": ("axpy", lambda e: _param(e, "scratch").update(type="i32", value=-(2**31) - 1)),
    "a fixed f32 with an integer": ("axpy", lambda e: _param(e, "alpha").update(value=2)),
    "a fixed f32 with a string": ("axpy", lambda e: _param(e, "alpha").update(value=EVIL)),
    "a fixed f32 with a tagged float that is not one": ("axpy", lambda e: _param(e, "alpha").update(value={"float": "inf\n"})),
    "a fixed f32 with a tagged float and more": ("axpy", lambda e: _param(e, "alpha").update(value={"float": "inf", "more": 1})),
    "a fixed f32 that is nan": ("axpy", lambda e: _param(e, "alpha").update(value=float("nan"))),
    "a fixed argument with no value": ("axpy", lambda e: _param(e, "alpha").pop("value")),
    "a fixed argument with a dim": ("axpy", lambda e: _param(e, "alpha").update(dim="n")),
    "a fixed argument with a name that is not one": ("axpy", lambda e: _param(e, "alpha").update(name=EVIL)),
    "two parameters of one name": ("axpy", lambda e: _param(e, "alpha").update(name="x")),
    "a parameter of a kind that is none": ("axpy", lambda e: _param(e, "alpha").update(kind="const")),
    # the binary section
    "no binary section": ("axpy", lambda e: e.pop("binary")),
    "a binary section of another source": ("axpy", lambda e: e.update(binary={"source": "elf"})),
    "a ptx section with no flags": ("axpy", lambda e: e.update(binary={"source": "ptx"})),
    "a ptx section with a flag that is not a string": ("axpy", lambda e: e.update(binary={"source": "ptx", "ptxas_flags": [1]})),
    "a cubin section with flags": ("axpy", lambda e: e.update(binary={"source": "cubin", "ptxas_flags": []})),
    "a source digest that is not a digest": ("axpy", lambda e: e.update(source_digest="the sha256\n")),
    "no source digest": ("axpy", lambda e: e.pop("source_digest")),
    "a source digest kind of another tool": ("axpy", lambda e: e.update(source_digest_kind="tilefish.triton-ir.v1")),
    "options on a prebuilt binary": ("axpy", lambda e: e.update(options={})),
    "num_warps on a prebuilt binary": ("axpy", lambda e: e.update(num_warps=4)),
    "a block that is not three sizes": ("axpy", lambda e: e.update(block=[128])),
}


@pytest.mark.parametrize("name", sorted(KERNEL_FIELDS))
def test_a_kernel_entry_the_reader_refuses_is_refused_and_writes_nothing(name, schema3_manifest, tmp_path):
    stem, change = KERNEL_FIELDS[name]
    manifest = copy.deepcopy(schema3_manifest)
    change(_entry(manifest, stem))
    out = tmp_path / "launchers.rs"
    for embed in (False, True):
        with pytest.raises(TilefishManifestError, match="cannot write Rust launchers for"):
            rust.emit(manifest, out, embed=embed)
    assert not out.exists()


def test_the_unmutated_manifest_is_accepted_and_has_no_injected_code(schema3_manifest):
    source = rust.render(schema3_manifest)
    assert "pwned" not in source and "pub struct HeadCopy " in source and "pub struct MaskedRows " in source and "pub struct Axpy " in source and "pub struct GemmM32 " in source
    assert "pub struct QkvSum " in source and "the stride of axis 1 of `q`, `k` and `v`, in elements (one stride, which each of them must have)" in source


LIBRARY_FIELDS = {
    "the stem": lambda e: e.update(stem=EVIL),
    "the operator": lambda e: e.update(op=EVIL),
    "the architecture": lambda e: e.update(arch="sm_103\n"),
    "the implementation": lambda e: e.update(impl="cublaslt.gemm\npub fn pwned() {}"),
    "an implementation that is not one": lambda e: e.update(impl="cublaslt.nothing"),
    "a field that is not in the section": lambda e: e.update(extra=1),
    "a field that is missing": lambda e: e.pop("algo"),
    "the name of a tensor": lambda e: e["abi"]["params"][0].update(name=EVIL),
    "the name of a tensor with a unicode letter": lambda e: e["abi"]["params"][0].update(name="aé"),
    "two tensors of one name": lambda e: e["abi"]["params"][1].update(name="a"),
    "the shape of a tensor": lambda e: e["abi"]["params"][0].update(shape=[EVIL, 1024]),
    "a shape that is not the problem's": lambda e: e["abi"]["params"][0].update(shape=["M", 512]),
    "the dtype of a tensor": lambda e: e["abi"]["params"][0].update(dtype=EVIL),
    "a tensor that is not bf16": lambda e: e["abi"]["params"][0].update(dtype="f32"),
    "a tensor that is strided": lambda e: e["abi"]["params"][0].update(strides=["a_s0", 1], offset="a_offset"),
    "a result that is not written": lambda e: e["abi"]["params"][-1].update(access="in"),
    "m that is negative": lambda e: e.update(m=-1),
    "m that is a bool": lambda e: e.update(m=True),
    "n that is a string": lambda e: e.update(n="2048\n"),
    "k that is huge": lambda e: e.update(k=2**64),
    "a leading dimension that is not k": lambda e: e.update(lda=1),
    "the workspace": lambda e: e.update(workspace_bytes="1\npub fn pwned() {}"),
    "a workspace of zero": lambda e: e.update(workspace_bytes=0),
    "the version of cuBLASLt": lambda e: e.update(cublaslt_version="12.8\n"),
    "a version of zero": lambda e: e.update(cublaslt_version=0),
    "the SM count": lambda e: e.update(sm_count="30\npub fn pwned() {}"),
    "an SM count of zero": lambda e: e.update(sm_count=0),
    "an SM count that is a bool": lambda e: e.update(sm_count=True),
    "no SM count": lambda e: e.pop("sm_count"),
    "the algorithm": lambda e: e.update(algo="z" * 128),
    "an algorithm that is short": lambda e: e.update(algo="0" * 127),
    "the epilogue": lambda e: e.update(epilogue="bias"),
    "the compute type": lambda e: e.update(compute_type="CUBLAS_COMPUTE_16F"),
    "a bucket value": lambda e: e["buckets"].update(M=EVIL),
    "the name of a bucket": lambda e: e["buckets"].update({EVIL: 1}),
    "a table of dimensions with a name that is not one": lambda e: e["dims"].update({EVIL: {}}),
    "a bucket that is not the one of the dimension": lambda e: e["dims"]["M"].update(bucket=64),
    "a precondition of another kind": lambda e: e["preconditions"].append({"kind": "max_elements", "arg": "a", "index": 0, "value": 2**31 - 1, "shape": ["M", "1024"]}),
    "a missing alignment": lambda e: e["preconditions"].pop(0),
    "an alignment that is not 16": lambda e: e["preconditions"][0].update(bytes=2),
}


@pytest.mark.parametrize("name", sorted(LIBRARY_FIELDS))
def test_a_library_entry_the_reader_refuses_is_refused_and_writes_nothing(name, tmp_path):
    manifest = build_manifest(tmp_path / "m", (), pins(gemm)[:1])
    LIBRARY_FIELDS[name](manifest.library_entries[0])
    out = tmp_path / "launchers.rs"
    with pytest.raises(TilefishManifestError, match="cannot write Rust launchers for"):
        rust.emit(manifest, out)
    assert not out.exists()


def test_the_doc_text_a_library_entry_carries_is_the_validated_text(tmp_path):
    """The version, the sizes and the algorithm architectures are numbers and names by the time they are text: nothing else is in the doc."""
    manifest = build_manifest(tmp_path / "m", (), pins(gemm_bias)[:1])
    source = rust.render(manifest, ops=[gemm_bias])
    assert "cuBLASLt 12.8.4 (120804)" in source and "It computes `out[32, 2048] = a[32, 1024] @ b[2048, 1024]^T + bias[2048]`" in source
    assert "pwned" not in source


# -- the launchers compile against the crate, once it reads schema 3 ---------------------------------------------------------------------------------


@needs_new_crate
def test_the_launchers_of_schema_3_compile_against_the_crate(p1, prebuilt, tmp_path_factory):
    """``cargo check`` with no warnings of the modules this file snapshots: they need ``Gemm``, ``TensorRef::from_cuda_strided`` and the rest of the crate that reads schema 3."""
    tmp = tmp_path_factory.mktemp("schema3-crate")
    crate = tmp / "launchers-crate"
    (crate / "src").mkdir(parents=True)
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "launchers-check"\nversion = "0.0.0"\nedition = "2024"\n\n'
        f'[dependencies]\ntilefish = {{ path = {json.dumps(str(CRATE))} }}\n\n[workspace]\n'
    )
    ops = [p1.head_copy, p1.head_copy16, p1.head_copy64, p1.softmax3d, p1.masked_rows, p1.scale_by, p1.scale_flat]
    rust.emit(build_manifest(tmp / "m1", ops), crate / "src" / "strided.rs", ops=ops)
    rust.emit(build_manifest(tmp / "m2", prebuilt), crate / "src" / "binary.rs", ops=prebuilt)
    rust.emit(build_manifest(tmp / "m3", (), [*pins(gemm), *pins(gemm_bias)]), crate / "src" / "library.rs", ops=[gemm, gemm_bias])
    everything = build_manifest(tmp / "m4", [ops[0], prebuilt[0]], pins(gemm)[:1])
    rust.emit(everything, crate / "src" / "embedded.rs", ops=[ops[0], prebuilt[0], gemm], embed=True)
    import gpu_kernels_p6 as p6  # a prebuilt binary with fixed arguments given per call: `launch` takes them after the tensors

    given = rust.emit(build_manifest(tmp / "m5", [p6.scale_add]), crate / "src" / "given.rs", ops=[p6.scale_add]).read_text()
    assert "k: i32,\n        c: i64,\n" in given and 'launch_with(stream, &mut [x.into_arg(), y.into_arg()], &[("k", i64::from(k)), ("c", c)])' in given
    (crate / "src" / "lib.rs").write_text("pub mod binary;\npub mod embedded;\npub mod given;\npub mod library;\npub mod strided;\n")
    env = {**os.environ, "RUSTFLAGS": "-D warnings", "CARGO_TARGET_DIR": str(REPO_ROOT / "target" / "launchers-check-schema3")}
    done = subprocess.run(["cargo", "check", "--manifest-path", str(crate / "Cargo.toml"), "--color", "never"], capture_output=True, text=True, env=env, timeout=900)
    assert done.returncode == 0, done.stderr[-4000:]
