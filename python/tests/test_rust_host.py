"""The Rust host's Python side: the emitter of typed launchers, and the fixtures the crate is tested with.

``tilefish.hosts.rust.emit`` writes one Rust module per manifest. Here it is checked on small fake manifests (snapshots in
``tests/golden/``: regenerate with ``UPDATE_GOLDEN=1 pytest tests/test_rust_host.py`` and review the diff) and on the real
fixtures the crate is tested with (``crates/tilefish/tests/fixtures``, manifests of schema 3 built on a B300 by ``export_rust_fixtures.py``); the
output is deterministic and independent of the order of the entries; a declaration that is not what the manifest was
built from is refused; and the emitted code is compiled by ``cargo check`` (with the ``tilefish`` crate, where cargo is
installed and the crate reads schema 3), with ``embed=True`` too. Nothing here needs a GPU: the digests of the fixtures are recomputed by Python from the
files and must be the ones ``expected.json`` (which the Rust tests check against) holds. What schema 3 added (strided views, device scalars, prebuilt binaries,
GEMMs) is ``test_rust_host_schema3.py``.
"""

import copy
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

import pytest
from built_fakes import fake_build, rowop, scale
from conftest import REPO_ROOT
from elf_builder import arch_header, cubin_like

import tilefish
from tilefish import Dim, Out, Tensor, TilefishBuildError, TilefishManifestError, bf16, const, e2m1, f32
from tilefish._elf import code_digest
from tilefish.aot import Manifest, compile_specs, write_manifest
from tilefish.hosts import rust

GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "golden"
FIXTURES = REPO_ROOT / "crates" / "tilefish" / "tests" / "fixtures"
needs_crate = pytest.mark.skipif(not FIXTURES.exists(), reason="the Rust crate is not available (not running from a source checkout)")
CRATE_SRC = REPO_ROOT / "crates" / "tilefish" / "src"
# a launcher of a GEMM needs `Gemm` of the crate, and the fixtures have GEMMs: the crate of an older checkout reads schema 2 only
needs_new_crate = pytest.mark.skipif(
    shutil.which("cargo") is None or not CRATE_SRC.is_dir() or not any("pub struct Gemm" in f.read_text() for f in CRATE_SRC.rglob("*.rs")),
    reason="cargo is not installed, or the crate of this checkout does not read schema 3 yet (it has no `Gemm`)",
)


def check_golden(name: str, source: str) -> None:
    path = GOLDEN_DIR / name
    if os.environ.get("UPDATE_GOLDEN") == "1":
        path.write_bytes(source.encode("utf-8"))
        return
    assert path.exists(), f"no golden {name}: run UPDATE_GOLDEN=1 pytest tests/test_rust_host.py and review {path}"
    assert source == path.read_bytes().decode("utf-8"), f"{name} no longer matches {path}; if the change is intended, rerun with UPDATE_GOLDEN=1"


def pinned(manifest):
    """The manifest, pinned to its own digest: the emitter takes a pinned manifest only, and these tests are about what it writes."""
    return manifest.check_pin(manifest.digest)


def expected_digest(name="sm_103"):
    """The digest of a fixture that ``expected.json`` holds (which the Rust tests check too): computed by Python."""
    return json.loads((FIXTURES / "expected.json").read_text())["fixtures"][name]["manifest_digest"]


# The SM count of the device the GEMMs of the committed sm_103 fixture were pinned on: the B300 MIG slice (1g) the fixtures are built on has 30 SMs, and
# re-pinning the fixture's twelve problems there gives the twelve algorithms it holds (checked on that device when the field was added).
FIXTURE_SM_COUNT = 30


def fixture_predates_sm_count(name="sm_103"):
    """Whether the committed fixture's library entries lack ``sm_count`` (the field a library entry gained after the fixtures were built): the Rust lane
    regenerates the fixtures with it; until then Python, which requires the field, reads a copy that has it (:func:`fixture_dir`)."""
    doc = json.loads((FIXTURES / name / "manifest.json").read_text())
    return any("sm_count" not in e for e in doc["library"])


predates_sm_count = pytest.mark.skipif(
    FIXTURES.exists() and fixture_predates_sm_count(),
    reason="the committed Rust fixtures predate the sm_count of a library entry: generated.rs and expected.json are regenerated with them (crates/tilefish/tests/fixtures)",
)


def fixture_dir(tmp_path, name="sm_103"):
    """The directory of a fixture of the crate as a reader of today's schema takes it, and the digest to pin it with: the committed one, or, while that predates
    ``sm_count``, a copy of it migrated as ``export_rust_fixtures.py --migrate-sm-count`` migrates the committed one (``with_sm_count``)."""
    if not fixture_predates_sm_count(name):
        return FIXTURES / name, expected_digest(name)
    from export_rust_fixtures import with_sm_count

    from tilefish._manifest import manifest_digest

    directory = tmp_path / f"fixture-{name}"
    shutil.copytree(FIXTURES / name, directory)
    doc = with_sm_count(json.loads((directory / "manifest.json").read_text()), FIXTURE_SM_COUNT)
    (directory / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n")
    return directory, manifest_digest(doc)


def fixture(tmp_path, name="sm_103"):
    """A fixture of the crate (manifest schema 3, built by ``export_rust_fixtures.py``), loaded with the pin ``expected.json`` holds for it (or, while the committed
    one predates ``sm_count``, the copy of :func:`fixture_dir` with its own digest)."""
    directory, pin = fixture_dir(tmp_path, name)
    return Manifest.load(directory, pin=pin)


def committed_ops():
    """The declarations the committed fixture was emitted from: what can be declared without a GPU (``host_ops.gpu_ops`` cannot: it needs the Triton
    builds the PTX of two prebuilt externs comes from), and the GEMMs of the committed shapes."""
    import host_ops

    return [*host_ops.ops().values(), *host_ops.library_ops(host_ops.GEMM_SHAPES_COMMITTED).values()]


@pytest.fixture
def fake_manifest(tmp_path):
    """A pinned manifest directory of four fake binaries: ``scale``, and ``rowop`` for its two buckets, for sm_90."""
    built = [fake_build(s) for s in compile_specs(scale) + compile_specs(rowop)]
    write_manifest(built, tmp_path / "manifest" / "manifest.json", cubins=True)
    return pinned(Manifest.load(tmp_path / "manifest"))


# -- snapshots ------------------------------------------------------------------------------------------------------------


def test_launchers_typed_from_declarations(fake_manifest):
    source = rust.render(fake_manifest, ops=[scale, rowop])
    check_golden("rust_fakes.rs", source)
    assert f'pub const MANIFEST_DIGEST: &str = "{fake_manifest.digest}";' in source
    # one struct per stem, so one per bucket, with the shapes of the declaration
    for struct in ("pub struct Scale ", "pub struct RowopS32 ", "pub struct RowopS64 "):
        assert struct in source
    assert 'pub const SHAPES: &\'static [&\'static [&\'static str]] = &[&["M", "S"], &["M", "S"]];' in source
    assert "x: TensorRef<'a, F32>" in source and "y: TensorMut<'a, F32>" in source


def test_launchers_typed_from_the_manifest_alone(fake_manifest):
    """Schema 2 records the declared shapes and the dimensions: the declarations only add the signature to the docs."""
    source = rust.render(fake_manifest)
    check_golden("rust_fakes_manifest_only.rs", source)
    with_ops = rust.render(fake_manifest, ops=[scale, rowop])
    assert "`scale(x: Tensor[n, bf16]" in with_ops and "`scale(" not in source  # the declarations add the signature, nothing else
    assert with_ops.replace("\n///\n", "\n").count("pub fn launch") == source.count("pub fn launch")
    assert "SHAPES" in source and 'pub const SHAPES: &\'static [&\'static [&\'static str]] = &[&["M", "S"], &["M", "S"]];' in source
    # nothing but the tensors: there is no way to tell a launch what a dimension is
    assert "scalars: " not in source and "dims: " not in source and "self.kernel.launch(stream, &mut [x.into_arg(), y.into_arg()])" in source
    assert "Constraints: `S` is exactly 32" in source and "Constraints: `n` is a multiple of 16" in source


def test_embedded_launchers(fake_manifest, tmp_path):
    out = tmp_path / "launchers" / "launchers.rs"
    source = rust.emit(fake_manifest, out, ops=[scale, rowop], embed=True).read_text()
    check_golden("rust_fakes_embed.rs", source)
    assert 'include_bytes!("../manifest/manifest.json")' in source
    assert 'include_bytes!("../manifest/scale.sm_90.cubin")' in source
    assert "Manifest::from_embedded(MANIFEST_JSON, CUBINS, Pin::Digest(MANIFEST_DIGEST))" in source


@needs_crate
def test_baked_in_sizes_are_constants_and_documented(tmp_path):
    source = rust.render(fixture(tmp_path), ops=committed_ops())
    assert "pub const X_PTR_AXIS1: usize = 2048;" in source  # the extern rmsnorm: Tensor[Rows, Hidden = 2048]
    assert "Baked into the binary: axis 1 of `x_ptr` is 2048;" in source
    assert "Constraints: `Db` is exactly 96 (a bucket of [96, 512, 1000]" in source


# -- determinism ------------------------------------------------------------------------------------------------------------


def test_the_output_is_deterministic_and_does_not_depend_on_the_order_of_the_entries(fake_manifest, tmp_path):
    first = rust.render(fake_manifest, ops=[scale, rowop])
    assert rust.render(fake_manifest, ops=[rowop, scale]) == first  # nor on the order of the declarations
    doc = json.loads(fake_manifest.path.read_text())
    doc["kernels"].reverse()
    other = tmp_path / "reordered"
    shutil.copytree(fake_manifest.directory, other)
    (other / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n")
    reordered = pinned(Manifest.load(other))
    assert reordered.digest == fake_manifest.digest  # the digest does not depend on the order either
    assert rust.render(reordered, ops=[scale, rowop]) == first
    # and writing it twice gives the same bytes
    a, b = tmp_path / "a.rs", tmp_path / "b.rs"
    rust.emit(fake_manifest, a, ops=[scale, rowop])
    rust.emit(fake_manifest, b, ops=[scale, rowop])
    assert a.read_bytes() == b.read_bytes() == first.encode("utf-8")


def test_a_symbolic_link_where_the_file_goes_is_refused(fake_manifest, tmp_path):
    target = tmp_path / "elsewhere.rs"
    target.write_text("// mine\n")
    link = tmp_path / "launchers.rs"
    link.symlink_to(target)
    with pytest.raises(TilefishBuildError, match="symbolic link"):
        rust.emit(fake_manifest, link, ops=[scale, rowop])
    assert target.read_text() == "// mine\n"


# -- what the emitter refuses --------------------------------------------------------------------------------------------------


def test_a_declaration_that_is_not_what_the_manifest_was_built_from_is_refused(fake_manifest):
    n = Dim("n", divisible_by=16)

    @tilefish.extern(triton="my_kernels:scale", num_warps=4, grid=lambda n: (tilefish.cdiv(n, 128),))  # not the grid of the build
    def scale(x: Tensor[n, bf16], y: Out[Tensor[n, bf16]], n_elem: n, *, ALPHA: const[f32] = float("inf"), BLOCK: const = 256): ...

    with pytest.raises(TilefishManifestError, match="is not what the manifest's binary 'scale' was built from") as e:
        rust.render(fake_manifest, ops=[scale])
    assert "first difference at grid" in str(e.value)
    # a declaration of an operator that the manifest does not have is not an error: it has nothing to type
    assert "pub struct RowopS32 " in rust.render(fake_manifest, ops=[rowop])


def test_the_binaries_of_a_stem_must_agree_across_architectures(tmp_path):
    (spec,) = compile_specs(scale)
    a = fake_build(spec, arch="sm_90")
    b = fake_build(spec, arch="sm_100", cubin=cubin_like(code=b"\x09\x08\x07\x06" * 16, **arch_header("sm_100")))
    write_manifest([a, b], tmp_path / "manifest.json", cubins=True)
    manifest = pinned(Manifest.load(tmp_path))
    source = rust.render(manifest, ops=[scale])
    assert source.count("pub struct Scale ") == 1  # one stem for two architectures is one launcher
    assert "built for sm_100, sm_90" in source
    doc = json.loads((tmp_path / "manifest.json").read_text())
    doc["kernels"][1]["grid"] = "(cdiv(n, 128),)"
    (tmp_path / "manifest.json").write_text(json.dumps(doc, indent=2))
    with pytest.raises(TilefishManifestError, match="disagree about 'grid'"):
        rust.render(pinned(Manifest.load(tmp_path)))


def test_a_pointer_of_a_packed_dtype_has_no_launcher(tmp_path):
    n = Dim("n", divisible_by=16)

    @tilefish.extern(triton="my_kernels:packed", num_warps=4, grid=lambda n: (tilefish.cdiv(n, 256),))
    def packed(x: Tensor[n, e2m1], y: Out[Tensor[n, bf16]], n_elem: n, *, BLOCK: const = 256): ...

    (spec,) = compile_specs(packed)
    write_manifest([fake_build(spec)], tmp_path / "manifest.json", cubins=True)
    with pytest.raises(TilefishManifestError, match="no element type in Rust"):
        rust.render(pinned(Manifest.load(tmp_path)))


def test_identifiers_are_made_safe(fake_manifest):
    assert rust._camel("bucketed_rms.Db96") == "BucketedRmsDb96"
    assert rust._camel("a.b_c") == "ABC"
    assert rust._camel("_9x") == "K9x"
    assert rust._snake_ident("type") == "type_" and rust._snake_ident("stream") == "stream_" and rust._snake_ident("x") == "x"
    with pytest.raises(TilefishManifestError, match="not printable ASCII"):
        rust._rs_str("café")


# -- the fixtures of the crate ------------------------------------------------------------------------------------------------------


@needs_crate
def test_the_fixture_digests_are_the_ones_the_rust_tests_expect(tmp_path):
    expected = json.loads((FIXTURES / "expected.json").read_text())["fixtures"]
    assert sorted(expected) == ["sm_103", "sm_120"]
    for name, want in expected.items():
        directory, pin = fixture_dir(tmp_path, name)
        manifest = Manifest.load(directory, pin=pin)  # verifies every cubin against the manifest; schema 3, which Rust reads too
        if directory == FIXTURES / name:  # (a copy that stands in for a fixture that predates sm_count has a digest of its own)
            assert manifest.digest == want["manifest_digest"], name
        assert len(manifest.artifacts) == len(want["kernels"]) == 31 and len(manifest.library_entries) == len(want["library"])
        # an extern declared with enable_fp_fusion= records the option it was compiled with
        for stem, fused in (("fma_plain", False), ("fma_fused", True)):
            assert manifest.get(stem, name).entry["options"] == {"num_warps": 4, "num_stages": 3, "enable_fp_fusion": fused}
        # a stride that three views share is one scalar whose stride_of lists them
        for stem in ("qkv_sum", "qkv_sum16"):
            params = {p["name"]: p for p in manifest.get(stem, name).entry["abi"]["params"]}
            assert [p["stride_of"] for p in (params["s_row"], params["s_head"])] == [[{"arg": v, "axis": axis} for v in "qkv"] for axis in (0, 1)]
        for k in want["kernels"]:
            artifact = manifest.get(k["stem"], k["arch"])
            assert artifact.entry["cubin"] == k["cubin"] and artifact.entry["kind"] == k["kind"]
            assert hashlib.sha256((FIXTURES / name / k["cubin"]).read_bytes()).hexdigest() == k["sha256"]
            assert code_digest(artifact.cubin) == k["sha256_code"]
        for e in want["library"]:
            entry = manifest.library_entry(e["stem"], e["arch"])
            assert (entry["algo"], entry["cublaslt_version"], entry["workspace_bytes"], entry["impl"], entry["sm_count"]) == (
                e["algo"],
                e["cublaslt_version"],
                e["workspace_bytes"],
                e["impl"],
                e.get("sm_count", FIXTURE_SM_COUNT),
            )
    # a pin is made on a device of its architecture: the native fixture has the GEMMs (two shapes, with and without a bias, over three buckets of M)
    # and the one that was cross-compiled has none
    assert len(expected["sm_103"]["library"]) == 12 and expected["sm_120"]["library"] == []
    assert {e["stem"].split(".")[0] for e in expected["sm_103"]["library"]} == {"gemm_1024_1024", "gemm_1024_1024_bias", "gemm_3072_1024", "gemm_3072_1024_bias"}
    # the prebuilt binaries are in both: hand-written PTX, the PTX of a Triton build and a cubin
    for name in expected:
        kinds = {k["stem"]: k["kind"] for k in expected[name]["kernels"]}
        assert {s for s, kind in kinds.items() if kind == "binary"} == {"axpy_cubin", "axpy_ptx", "gelu_ptx", "softmax_ptx", "strided_copy"}, name
    # the same binary through both doors has the same machine code (the PTX of the Triton build, assembled with Triton's flags)
    by = {k["stem"]: k for k in expected["sm_103"]["kernels"]}
    assert by["gelu_ptx"]["sha256_code"] == by["gelu_exact"]["sha256_code"] and by["softmax_ptx"]["sha256_code"] == by["softmax"]["sha256_code"]
    assert by["axpy_cubin"]["sha256_code"] == by["axpy_ptx"]["sha256_code"]
    # one kernel compiled with and without FMA contraction is two binaries
    assert by["fma_plain"]["sha256_code"] != by["fma_fused"]["sha256_code"]


@needs_crate
def test_a_library_entry_without_its_sm_count_is_refused(tmp_path):
    """The committed fixture while it predates the field, and a copy of it with the field removed once it has it: Python requires sm_count (so does Rust)."""
    if fixture_predates_sm_count():
        with pytest.raises(TilefishManifestError, match=r"library entry #0 \('gemm_1024_1024.M32'\): must have exactly the fields .*missing \['sm_count'\]"):
            Manifest.load(FIXTURES / "sm_103")
    directory, _ = fixture_dir(tmp_path)
    shutil.copytree(directory, tmp_path / "m")
    doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
    del doc["library"][3]["sm_count"]
    (tmp_path / "m" / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match=r"library entry #3 .*missing \['sm_count'\]"):
        Manifest.load(tmp_path / "m")


@needs_crate
@predates_sm_count
def test_the_committed_launchers_are_what_the_emitter_writes_for_the_fixture(tmp_path):
    emitted = rust.render(fixture(tmp_path), ops=committed_ops())
    assert emitted == (FIXTURES / "generated.rs").read_text(), (
        "crates/tilefish/tests/fixtures/generated.rs is not what tilefish.hosts.rust.emit writes for the sm_103 fixture: "
        "regenerate it with python/tests/export_rust_fixtures.py (or, if only the emitter changed, with emit)"
    )
    assert f'pub const MANIFEST_DIGEST: &str = "{expected_digest()}";' in emitted


def test_a_schema_2_manifest_is_refused_and_becomes_schema_3_by_three_additions(fake_manifest, tmp_path):
    """A schema 2 manifest (no ``kind``, no ``scratch_slots``, no ``library``): Python refuses it, and ``migrate_doc`` of ``schema2_fixtures`` is what changes it."""
    from schema2_fixtures import migrate_doc, migrated_copy

    from tilefish._manifest import LEGACY_DIGEST_TAG, manifest_digest

    doc = json.loads(fake_manifest.path.read_text())
    old = copy.deepcopy(doc)
    old["schema"] = 2
    del old["library"]
    for entry in old["kernels"]:
        del entry["kind"]
        del entry["abi"]["scratch_slots"]
    directory = tmp_path / "old"
    shutil.copytree(fake_manifest.directory, directory)
    (directory / "manifest.json").write_text(json.dumps(old, indent=2) + "\n")
    with pytest.raises(TilefishManifestError, match="schema 2, which an older tilefish wrote.*Re-export it"):
        Manifest.load(directory)  # Python reads schema 3 only
    assert migrate_doc(old) == doc
    assert migrated_copy(directory, tmp_path / "new") == doc  # the directory too: the manifest rewritten, the cubins as they are
    migrated = Manifest.load(tmp_path / "new")
    assert migrated.doc == doc and migrated.digest == fake_manifest.digest
    # the schema 2 document is still digested, in its own form (tag v2, no library), which is what the crate 0.0.1 computed
    assert manifest_digest(old) != fake_manifest.digest and LEGACY_DIGEST_TAG == b"tilefish.manifest-digest.v2\0"


# -- cargo check --------------------------------------------------------------------------------------------------------------------


@needs_new_crate
@needs_crate
def test_the_launchers_compile_against_the_crate(tmp_path_factory, fake_manifest):
    """``cargo check`` of three emitted modules (typed from declarations, from the manifest alone, and embedded) with no warnings."""
    crate = tmp_path_factory.mktemp("launchers-crate")
    (crate / "src").mkdir()
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "launchers-check"\nversion = "0.0.0"\nedition = "2024"\n\n'
        f'[dependencies]\ntilefish = {{ path = {json.dumps(str(REPO_ROOT / "crates" / "tilefish"))} }}\n\n[workspace]\n'
    )
    sm_103 = fixture(tmp_path_factory.mktemp("fixture"))
    declared = committed_ops()
    rust.emit(sm_103, crate / "src" / "typed.rs", ops=declared)
    rust.emit(sm_103, crate / "src" / "manifest_only.rs")
    rust.emit(sm_103, crate / "src" / "embedded.rs", ops=declared, embed=True)
    rust.emit(fake_manifest, crate / "src" / "fakes.rs", ops=[scale, rowop])
    (crate / "src" / "lib.rs").write_text("pub mod embedded;\npub mod fakes;\npub mod manifest_only;\npub mod typed;\n")
    env = {**os.environ, "RUSTFLAGS": "-D warnings", "CARGO_TARGET_DIR": str(REPO_ROOT / "target" / "launchers-check")}
    done = subprocess.run(
        ["cargo", "check", "--manifest-path", str(crate / "Cargo.toml"), "--color", "never"],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert done.returncode == 0, done.stderr[-4000:]


# -- Python refuses what the Rust tests refuse -----------------------------------------------------------------------------------------


def _mutations():
    """``(name, mutate(directory))``: what ``crates/tilefish/tests/strict_reading.rs`` does to a copy of the fixture, and Python must refuse too."""

    def text(d, f):
        path = d / "manifest.json"
        path.write_text(f(path.read_text()))

    def doc(d, f):
        path = d / "manifest.json"
        value = json.loads(path.read_text())
        f(value)
        path.write_text(json.dumps(value, indent=2))

    def entry(value, stem):
        return next(k for k in value["kernels"] if k["stem"] == stem)

    def flip_code(d):
        from tilefish._elf import parse_elf

        path = d / "gelu_exact.sm_103.cubin"
        data = bytearray(path.read_bytes())
        old = hashlib.sha256(data).hexdigest()
        section = parse_elf(bytes(data)).section(".text.tf_gelu_exact")
        data[section.offset + 8] ^= 0x10
        path.write_bytes(bytes(data))
        text(d, lambda t: t.replace(old, hashlib.sha256(data).hexdigest()))  # the file digest is rewritten, the code digest is not

    def flip_last(d):
        path = d / "gelu_exact.sm_103.cubin"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(bytes(data))

    def duplicate_entry(value):
        value["kernels"].append(dict(value["kernels"][0]))

    return [
        ("a key twice", lambda d: text(d, lambda t: t.replace('"stem": "softmax",', '"stem": "softmax", "stem": "softmax",', 1))),
        ("NaN", lambda d: text(d, lambda t: t.replace('"shared": 16', '"shared": NaN', 1))),
        ("Infinity", lambda d: text(d, lambda t: t.replace('"shared": 16', '"shared": Infinity', 1))),
        ("1e999", lambda d: text(d, lambda t: t.replace('"num_stages": 3', '"num_stages": 1e999', 1))),
        ("a trailing comma", lambda d: text(d, lambda t: t.replace('"schema": 3,', '"schema": 3,,', 1))),
        ("another schema", lambda d: text(d, lambda t: t.replace('"schema": 3,', '"schema": 4,', 1))),
        ("the schema before the one with shapes", lambda d: text(d, lambda t: t.replace('"schema": 3,', '"schema": 1,', 1))),
        ("the schema of the Rust host", lambda d: text(d, lambda t: t.replace('"schema": 3,', '"schema": 2,', 1))),
        ("not JSON", lambda d: text(d, lambda t: "not json")),
        ("a cubin that leaves the directory", lambda d: text(d, lambda t: t.replace('"cubin": "softmax.sm_103.cubin"', '"cubin": "../sm_120/softmax.sm_120.cubin"', 1))),
        ("an absolute cubin name", lambda d: text(d, lambda t: t.replace('"cubin": "softmax.sm_103.cubin"', '"cubin": "/etc/passwd"', 1))),
        ("a missing cubin", lambda d: (d / "swiglu.sm_103.cubin").unlink()),
        ("a tampered cubin", flip_last),
        ("a tampered cubin with its file digest rewritten", flip_code),
        ("a kernel that uses shared memory with shared = 0", lambda d: doc(d, lambda v: entry(v, "softmax").update(shared=0))),
        ("an entry twice", lambda d: doc(d, duplicate_entry)),
        ("a missing key", lambda d: text(d, lambda t: t.replace('"symbol": "tf_softmax",', "", 1))),
        # what schema 3 added
        ("the schema 3.0, which is 3 to Python and not to a strict host", lambda d: text(d, lambda t: t.replace('"schema": 3,', '"schema": 3.0,', 1))),
        ("the schema as a string", lambda d: text(d, lambda t: t.replace('"schema": 3,', '"schema": "3",', 1))),
        ("the schema as true", lambda d: text(d, lambda t: t.replace('"schema": 3,', '"schema": true,', 1))),
        ("no library section", lambda d: doc(d, lambda v: v.pop("library"))),
        ("a library section that is not a list", lambda d: doc(d, lambda v: v.update(library={}))),
        ("a kind that is neither triton nor binary", lambda d: doc(d, lambda v: entry(v, "softmax").update(kind="python"))),
        ("no kind", lambda d: doc(d, lambda v: entry(v, "softmax").pop("kind"))),
        ("scratch slots that are not the two of a Triton kernel", lambda d: doc(d, lambda v: entry(v, "softmax")["abi"].update(scratch_slots=3))),
        ("a prebuilt binary with the settings of Triton", lambda d: doc(d, lambda v: entry(v, "axpy_ptx").update(num_warps=4))),
        ("a prebuilt binary without its binary section", lambda d: doc(d, lambda v: entry(v, "axpy_ptx").pop("binary"))),
        ("a binary section that is neither a cubin nor PTX", lambda d: doc(d, lambda v: entry(v, "axpy_ptx").update(binary={"source": "elf"}))),
        ("a prebuilt binary without the digest of its source", lambda d: doc(d, lambda v: entry(v, "axpy_ptx").pop("source_digest"))),
        ("a prebuilt binary with the scratch slots of Triton", lambda d: doc(d, lambda v: entry(v, "axpy_ptx")["abi"].update(scratch_slots=2))),
        ("a library entry with a field more", lambda d: doc(d, lambda v: v["library"][0].update(extra=1))),
        ("a library entry without its algorithm", lambda d: doc(d, lambda v: v["library"][0].pop("algo"))),
        ("a library entry with an algorithm that is not 64 bytes of hex", lambda d: doc(d, lambda v: v["library"][0].update(algo="0" * 126))),
        ("a library entry with leading dimensions that are not the problem's", lambda d: doc(d, lambda v: v["library"][0].update(lda=1))),
        ("a library entry twice", lambda d: doc(d, lambda v: v["library"].append(dict(v["library"][0])))),
        ("a library entry and a kernel of one stem and architecture", lambda d: doc(d, lambda v: entry(v, "softmax").update(stem=v["library"][0]["stem"], arch=v["library"][0]["arch"]))),
    ]


@needs_crate
@pytest.mark.parametrize("name, mutate", _mutations(), ids=[n for n, _ in _mutations()])
def test_python_refuses_what_the_rust_tests_refuse(name, mutate, tmp_path):
    directory, _ = fixture_dir(tmp_path)  # the committed fixture (as a reader of sm_count takes it), which Python accepts as it is
    Manifest.load(directory)
    shutil.copytree(directory, tmp_path / "m")
    mutate(tmp_path / "m")
    with pytest.raises(TilefishManifestError):
        Manifest.load(tmp_path / "m")


# -- what the emitter trusts, and what no field of a manifest can do to the file it writes -------------------------------------------------

# a payload that would end a doc comment, an item and a string literal, and write code
EVIL = 'x\n}\n}\npub fn pwned() { std::process::exit(0) }\nstatic S: &str = "'


def edited(fake_manifest, tmp_path, change):
    """A copy of the directory of ``fake_manifest`` whose entry ``scale`` was changed by ``change(entry, directory)``; loaded, not pinned."""
    other = tmp_path / "edited"
    shutil.copytree(fake_manifest.directory, other)
    doc = json.loads((other / "manifest.json").read_text())
    entry = next(k for k in doc["kernels"] if k["stem"] == "scale")
    change(entry, other)
    (other / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n")
    return Manifest.load(other)


def edited_in_memory(fake_manifest, tmp_path, change):
    """A copy of the directory of ``fake_manifest``, loaded, whose entry ``scale`` is then changed by ``change(entry, directory)`` in memory.

    The reader refuses most of the files :func:`edited` writes (as the Rust reader does); a :class:`Manifest` is an object its holder can still
    change, and what it holds is what the emitter validates again before any of it reaches the source it writes.
    """
    other = tmp_path / "in-memory"
    shutil.copytree(fake_manifest.directory, other)
    manifest = Manifest.load(other)
    change(next(a.entry for a in manifest.artifacts if a.stem == "scale"), other)
    return manifest


def param(entry, name):
    return next(p for p in entry["abi"]["params"] if p["name"] == name)


def rename_cubin(entry, directory, name):
    (directory / name).write_bytes((directory / entry["cubin"]).read_bytes())  # the file is there under the name it is given
    entry["cubin"] = name


FIELDS = {
    "the stem": lambda e, d: e.update(stem=EVIL),
    "a stem with a dot at the end": lambda e, d: e.update(stem="scale."),
    "the operator": lambda e, d: e.update(op=EVIL),
    "the architecture": lambda e, d: e.update(arch="sm_90\n"),
    "an architecture that is not one": lambda e, d: e.update(arch="sm_90; drop"),
    "a cubin name with a quote": lambda e, d: rename_cubin(e, d, 'a".sm_90.cubin'),
    "a cubin name with a newline": lambda e, d: rename_cubin(e, d, "a\n.sm_90.cubin"),
    "a tensor name": lambda e, d: param(e, "x").update(name=EVIL),
    "a tensor name with a unicode letter": lambda e, d: param(e, "x").update(name="xé"),
    "a tensor name that starts with a digit": lambda e, d: param(e, "x").update(name="1x"),
    "the dtype of a tensor": lambda e, d: param(e, "x").update(dtype=EVIL),
    "the access of a tensor": lambda e, d: param(e, "x").update(access="in\n"),
    "a tensor's dimension expression": lambda e, d: param(e, "x").update(shape=[EVIL]),
    "a tensor's dimension expression with a quote": lambda e, d: param(e, "x").update(shape=['n"; drop']),
    "a static size that is not an integer": lambda e, d: param(e, "x").update(shape=[1.5]),
    "a static size that is negative": lambda e, d: param(e, "x").update(shape=[-1]),
    "a static size beyond 64 bits": lambda e, d: param(e, "x").update(shape=[2**64]),
    "the shape of a tensor": lambda e, d: param(e, "x").update(shape=EVIL),
    "a scalar's dimension expression": lambda e, d: param(e, "n_elem").update(dim=EVIL),
    "a scalar's kind": lambda e, d: param(e, "n_elem").update(kind="pointer\n"),
    "a name in dims": lambda e, d: e["dims"].update({EVIL: {}}),
    "the divisor of a dimension": lambda e, d: e["dims"]["n"].update(divisible_by="16\npub fn pwned() {}"),
    "a dimension's constraints": lambda e, d: e["dims"].update(n="16"),
    "a bucket value": lambda e, d: e["dims"].update(n={"buckets": [16], "bucket": "16\n"}),
    "the name in buckets": lambda e, d: e["buckets"].update({EVIL: 1}),
    "a constexpr's name": lambda e, d: e["constexprs"][1].update(name=EVIL),
    "a constexpr's value": lambda e, d: e["constexprs"][1].update(value=EVIL),
    "a constexpr's tagged float": lambda e, d: e["constexprs"][0].update(value={"float": "inf\n"}),
    "a constexpr's dtype": lambda e, d: e["constexprs"][0].update(dtype=EVIL),
    "the grid": lambda e, d: e.update(grid=EVIL),
    "a grid that is a name": lambda e, d: e.update(grid="(cdiv(n, 256),)\n"),
    "the block": lambda e, d: e.update(block=[128, "1", 1]),
    "a block of three words": lambda e, d: e.update(block="1 1 1"),
}


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_a_field_that_is_not_what_it_must_be_is_refused_and_writes_nothing(name, fake_manifest, tmp_path):
    out = tmp_path / "launchers.rs"
    try:  # the file: the reader refuses it, or the emitter does
        manifests = [edited(fake_manifest, tmp_path, FIELDS[name])]
    except TilefishManifestError:
        manifests = []
    manifests.append(edited_in_memory(fake_manifest, tmp_path, FIELDS[name]))  # and the emitter's own guard, whatever the reader let through
    for manifest in manifests:
        for embed in (False, True):
            with pytest.raises(TilefishManifestError, match="cannot write Rust launchers for"):
                rust.emit(manifest, out, pin=manifest.digest, embed=embed)
    assert not out.exists()  # and the file is not written half way


def test_the_reviewers_grid_with_a_newline_cannot_inject_items(fake_manifest, tmp_path):
    """A newline in the grid ended the doc comment and wrote ``pub fn`` items into the generated file."""
    with pytest.raises(TilefishManifestError, match=r"grid .* has a character the dimension grammar does not allow"):
        edited(fake_manifest, tmp_path, lambda e, d: e.update(grid="(n,)\npub fn pwned() {}\n// "))  # the reader refuses the file
    manifest = edited_in_memory(fake_manifest, tmp_path, lambda e, d: e.update(grid="(n,)\npub fn pwned() {}\n// "))
    with pytest.raises(TilefishManifestError, match=r"the grid is not a grid expression"):
        rust.render(manifest, pin=manifest.digest)


def test_every_piece_of_text_that_reaches_a_doc_comment_stays_in_the_comment():
    # the guard below the validation: whatever a field carries, a line break starts another comment line, never code
    for text in ["a\npub fn pwned() {}", "b\r\npub fn pwned() {}", "c\rpub fn pwned() {}", "d pub fn pwned() {}", "e\x0bpub fn pwned() {}", "\x00\x1b[31m", "", "\n"]:
        out = rust._doc_comment(["first", text, "last"], "    ")
        lines = out.split("\n")
        assert lines[-1] == "" and all(line.startswith("    ///") for line in lines[:-1]), repr(out)
        assert "\r" not in out and "\x0b" not in out and " " not in out and "\x00" not in out and "\x1b" not in out, repr(out)
    assert rust._doc_comment(["a", "", "b"]) == "/// a\n///\n/// b\n"


def test_a_string_literal_cannot_be_ended_from_inside():
    assert rust._rs_str('a"b') == '"a\\"b"' and rust._rs_str("a\\b") == '"a\\\\b"'
    for bad in ["a\nb", "a\rb", "a\tb", "a\x00b", "café", " ", 5, None]:
        with pytest.raises(TilefishManifestError, match="not printable ASCII"):
            rust._rs_str(bad)


def test_the_names_that_clash_with_the_module_or_the_language_are_renamed(fake_manifest, tmp_path):
    """A tensor called ``stream`` is a second parameter of that name, and a stem that gives ``Self`` or ``Error`` a struct of that name."""

    def clash(entry, directory):
        entry["stem"] = entry["op"] = "kernel"  # (a stem is its operator and its buckets)
        param(entry, "x").update(name="stream")
        param(entry, "y").update(name="type")
        for pre in entry["preconditions"]:  # the preconditions name the pointers they are about
            pre["arg"] = {"x": "stream", "y": "type"}.get(pre.get("arg"), pre.get("arg"))

    manifest = edited(fake_manifest, tmp_path, clash)
    source = rust.render(manifest, pin=manifest.digest)
    assert "pub struct KernelOp " in source and "pub struct Kernel " not in source  # `Kernel` is imported from the crate
    assert "stream_: TensorRef<'a, Bf16>" in source and "type_: TensorMut<'a, Bf16>" in source
    for stem, struct in [("self", "SelfOp"), ("error", "ErrorOp"), ("manifest", "ManifestOp"), ("kernel", "KernelOp"), ("bf16", "Bf16Op"), ("tensor_ref", "TensorRefOp"), ("none", "NoneOp")]:
        assert rust._camel(stem) == struct, stem
    # and the struct names that are not imports are as they were
    assert rust._camel("scale") == "Scale" and rust._camel("rowop.S32") == "RowopS32"


# -- what the emitter loads, and from whom ------------------------------------------------------------------------------------------------


def test_a_directory_is_loaded_only_with_a_pin_and_only_the_pinned_manifest_is_taken(fake_manifest, tmp_path):
    out = tmp_path / "out.rs"
    directory, digest = fake_manifest.directory, fake_manifest.digest
    with pytest.raises(TilefishManifestError, match="without a trust anchor.*pin="):
        rust.emit(directory, out)
    with pytest.raises(TilefishManifestError, match=r"has digest .* but the pin is " + "0" * 64):
        rust.emit(directory, out, pin="0" * 64)
    with pytest.raises(TilefishManifestError, match="64 lowercase hex digits"):
        rust.emit(directory, out, pin="abc")
    assert not out.exists()
    rust.emit(directory, out, pin=digest, ops=[scale, rowop])
    assert out.read_text() == rust.render(fake_manifest, ops=[scale, rowop])  # a directory with its pin is the pinned manifest
    # a Manifest that was loaded without a pin is not trusted, and one that was is
    loose = Manifest.load(directory)
    assert not loose.pinned and fake_manifest.pinned and Manifest.load(directory, pin=digest).pinned
    with pytest.raises(TilefishManifestError, match="is not pinned"):
        rust.render(loose)
    with pytest.raises(TilefishManifestError, match="is not pinned"):
        rust.emit(loose, out)
    assert rust.render(loose, pin=digest) == rust.render(fake_manifest)  # a pin given here pins it
    assert loose.pinned  # and it is pinned from then on
    with pytest.raises(TilefishManifestError, match="but the pin is"):
        rust.render(Manifest.load(directory), pin="1" * 64)
    with pytest.raises(TilefishManifestError, match="takes a manifest directory"):
        rust.render(object())


def test_a_manifest_that_was_changed_after_it_was_pinned_is_not_the_one_that_was_pinned(fake_manifest, tmp_path):
    other = tmp_path / "changed"
    shutil.copytree(fake_manifest.directory, other)
    doc = json.loads((other / "manifest.json").read_text())
    doc["kernels"][0]["grid"] = "(n,)"
    (other / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n")
    with pytest.raises(TilefishManifestError, match="but the pin is " + fake_manifest.digest):
        rust.emit(other, tmp_path / "out.rs", pin=fake_manifest.digest)
    with pytest.raises(TilefishManifestError, match="but the pin is"):
        Manifest.load(other, pin=fake_manifest.digest)
    assert not (tmp_path / "out.rs").exists()


def test_a_pin_is_checked_before_a_cubin_is_read(fake_manifest, tmp_path):
    other = tmp_path / "no-cubins"
    shutil.copytree(fake_manifest.directory, other)
    for cubin in other.glob("*.cubin"):
        cubin.unlink()
    with pytest.raises(TilefishManifestError, match="but the pin is"):  # not "cannot read the cubin"
        Manifest.load(other, pin="2" * 64)
    with pytest.raises(TilefishManifestError, match="cannot read the cubin"):
        Manifest.load(other, pin=fake_manifest.digest)
