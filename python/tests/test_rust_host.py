"""The Rust host's Python side: the emitter of typed launchers, and the fixtures the crate is tested with.

``tilefish.hosts.rust.emit`` writes one Rust module per manifest. Here it is checked on small fake manifests (snapshots in
``tests/golden/``: regenerate with ``UPDATE_GOLDEN=1 pytest tests/test_rust_host.py`` and review the diff) and on the real
fixture the crate is tested with (``crates/tilefish/tests/fixtures``, built on a B300 by ``export_rust_fixtures.py``); the
output is deterministic and independent of the order of the entries; a declaration that is not what the manifest was
built from is refused; and the emitted code is compiled by ``cargo check`` (with the ``tilefish`` crate, where cargo is
installed), with ``embed=True`` too. Nothing here needs a GPU: the digests of the fixtures are recomputed by Python from the files
and must be the ones ``expected.json`` (which the Rust tests check against) holds.
"""

import hashlib
import json
import os
import pathlib
import shutil
import subprocess

import pytest
from built_fakes import fake_build, rowop, scale
from conftest import REPO_ROOT
from elf_builder import cubin_like

import tilefish
from tilefish import Dim, Out, Tensor, TilefishBuildError, TilefishManifestError, bf16, const, e2m1, f32
from tilefish._elf import code_digest
from tilefish.aot import Manifest, compile_specs, write_manifest
from tilefish.hosts import rust

GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "golden"
FIXTURES = REPO_ROOT / "crates" / "tilefish" / "tests" / "fixtures"
needs_crate = pytest.mark.skipif(not FIXTURES.exists(), reason="the Rust crate is not available (not running from a source checkout)")


def check_golden(name: str, source: str) -> None:
    path = GOLDEN_DIR / name
    if os.environ.get("UPDATE_GOLDEN") == "1":
        path.write_bytes(source.encode("utf-8"))
        return
    assert path.exists(), f"no golden {name}: run UPDATE_GOLDEN=1 pytest tests/test_rust_host.py and review {path}"
    assert source == path.read_bytes().decode("utf-8"), f"{name} no longer matches {path}; if the change is intended, rerun with UPDATE_GOLDEN=1"


@pytest.fixture
def fake_manifest(tmp_path):
    """A manifest directory of four fake binaries: ``scale``, and ``rowop`` for its two buckets, for sm_90."""
    built = [fake_build(s) for s in compile_specs(scale) + compile_specs(rowop)]
    write_manifest(built, tmp_path / "manifest" / "manifest.json", cubins=True)
    return Manifest.load(tmp_path / "manifest")


# -- snapshots ------------------------------------------------------------------------------------------------------------


def test_launchers_typed_from_declarations(fake_manifest):
    source = rust.render(fake_manifest, ops=[scale, rowop])
    check_golden("rust_fakes.rs", source)
    assert f'pub const MANIFEST_DIGEST: &str = "{fake_manifest.digest}";' in source
    # one struct per stem, so one per bucket, with the shapes of the declaration
    for struct in ("pub struct Scale ", "pub struct RowopS32 ", "pub struct RowopS64 "):
        assert struct in source
    assert 'pub const SHAPES: &\'static [&\'static [&\'static str]] = &[&["M", "S"], &["M", "S"]];' in source
    assert "x: TensorRef<'_, F32>" in source and "y: TensorMut<'_, F32>" in source


def test_launchers_typed_from_the_manifest_alone(fake_manifest):
    """Schema 2 records the declared shapes and the dimensions: the declarations only add the signature to the docs."""
    source = rust.render(fake_manifest)
    check_golden("rust_fakes_manifest_only.rs", source)
    with_ops = rust.render(fake_manifest, ops=[scale, rowop])
    assert "`scale(x: Tensor[n, bf16]" in with_ops and "`scale(" not in source  # the declarations add the signature, nothing else
    assert with_ops.replace("\n///\n", "\n").count("pub fn launch") == source.count("pub fn launch")
    assert "SHAPES" in source and 'pub const SHAPES: &\'static [&\'static [&\'static str]] = &[&["M", "S"], &["M", "S"]];' in source
    # nothing but the tensors: there is no way to tell a launch what a dimension is
    assert "scalars: " not in source and "dims: " not in source and "self.kernel.launch(stream, &[x.arg(), y.arg()])" in source
    assert "Constraints: `S` is exactly 32" in source and "Constraints: `n` is a multiple of 16" in source


def test_embedded_launchers(fake_manifest, tmp_path):
    out = tmp_path / "launchers" / "launchers.rs"
    source = rust.emit(fake_manifest, out, ops=[scale, rowop], embed=True).read_text()
    check_golden("rust_fakes_embed.rs", source)
    assert 'include_bytes!("../manifest/manifest.json")' in source
    assert 'include_bytes!("../manifest/scale.sm_90.cubin")' in source
    assert "Manifest::from_embedded(MANIFEST_JSON, CUBINS, Pin::Digest(MANIFEST_DIGEST))" in source


@needs_crate
def test_baked_in_sizes_are_constants_and_documented():
    import host_ops

    source = rust.render(Manifest.load(FIXTURES / "sm_103"), ops=host_ops.ops().values())
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
    reordered = Manifest.load(other)
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
    b = fake_build(spec, arch="sm_100", cubin=cubin_like(code=b"\x09\x08\x07\x06" * 16))
    write_manifest([a, b], tmp_path / "manifest.json", cubins=True)
    manifest = Manifest.load(tmp_path)
    source = rust.render(manifest, ops=[scale])
    assert source.count("pub struct Scale ") == 1  # one stem for two architectures is one launcher
    assert "built for sm_100, sm_90" in source
    doc = json.loads((tmp_path / "manifest.json").read_text())
    doc["kernels"][1]["grid"] = "(cdiv(n, 128),)"
    (tmp_path / "manifest.json").write_text(json.dumps(doc, indent=2))
    with pytest.raises(TilefishManifestError, match="disagree about 'grid'"):
        rust.render(Manifest.load(tmp_path))


def test_a_pointer_of_a_packed_dtype_has_no_launcher(tmp_path):
    n = Dim("n", divisible_by=16)

    @tilefish.extern(triton="my_kernels:packed", num_warps=4, grid=lambda n: (tilefish.cdiv(n, 256),))
    def packed(x: Tensor[n, e2m1], y: Out[Tensor[n, bf16]], n_elem: n, *, BLOCK: const = 256): ...

    (spec,) = compile_specs(packed)
    write_manifest([fake_build(spec)], tmp_path / "manifest.json", cubins=True)
    with pytest.raises(TilefishManifestError, match="no element type in Rust"):
        rust.render(Manifest.load(tmp_path))


def test_identifiers_are_made_safe(fake_manifest):
    assert rust._camel("bucketed_rms.Db96") == "BucketedRmsDb96"
    assert rust._camel("a.b_c") == "ABC"
    assert rust._camel("_9x") == "K9x"
    assert rust._snake_ident("type") == "type_" and rust._snake_ident("stream") == "stream_" and rust._snake_ident("x") == "x"
    with pytest.raises(TilefishManifestError, match="not printable ASCII"):
        rust._rs_str("café")


# -- the fixtures of the crate ------------------------------------------------------------------------------------------------------


@needs_crate
def test_the_fixture_digests_are_the_ones_the_rust_tests_expect():
    expected = json.loads((FIXTURES / "expected.json").read_text())["fixtures"]
    assert sorted(expected) == ["sm_103", "sm_120"]
    for name, want in expected.items():
        manifest = Manifest.load(FIXTURES / name)  # verifies every cubin against the manifest
        assert manifest.digest == want["manifest_digest"], name
        assert len(manifest.artifacts) == len(want["kernels"]) == 12
        for k in want["kernels"]:
            artifact = manifest.get(k["stem"], k["arch"])
            assert artifact.entry["cubin"] == k["cubin"]
            assert hashlib.sha256((FIXTURES / name / k["cubin"]).read_bytes()).hexdigest() == k["sha256"]
            assert code_digest(artifact.cubin) == k["sha256_code"]


@needs_crate
def test_the_committed_launchers_are_what_the_emitter_writes_for_the_fixture():
    import host_ops

    manifest = Manifest.load(FIXTURES / "sm_103")
    emitted = rust.render(manifest, ops=host_ops.ops().values())
    assert emitted == (FIXTURES / "generated.rs").read_text(), (
        "crates/tilefish/tests/fixtures/generated.rs is not what tilefish.hosts.rust.emit writes for the sm_103 fixture: "
        "regenerate it with python/tests/export_rust_fixtures.py (or, if only the emitter changed, with emit)"
    )


# -- cargo check --------------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
@needs_crate
def test_the_launchers_compile_against_the_crate(tmp_path_factory, fake_manifest):
    """``cargo check`` of three emitted modules (typed from declarations, from the manifest alone, and embedded) with no warnings."""
    import host_ops

    crate = tmp_path_factory.mktemp("launchers-crate")
    (crate / "src").mkdir()
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "launchers-check"\nversion = "0.0.0"\nedition = "2024"\n\n'
        f'[dependencies]\ntilefish = {{ path = {json.dumps(str(REPO_ROOT / "crates" / "tilefish"))} }}\n\n[workspace]\n'
    )
    fixture = Manifest.load(FIXTURES / "sm_103")
    declared = list(host_ops.ops().values())
    rust.emit(fixture, crate / "src" / "typed.rs", ops=declared)
    rust.emit(fixture, crate / "src" / "manifest_only.rs")
    rust.emit(fixture, crate / "src" / "embedded.rs", ops=declared, embed=True)
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
        ("a trailing comma", lambda d: text(d, lambda t: t.replace('"schema": 2,', '"schema": 2,,', 1))),
        ("another schema", lambda d: text(d, lambda t: t.replace('"schema": 2,', '"schema": 3,', 1))),
        ("the schema before the one with shapes", lambda d: text(d, lambda t: t.replace('"schema": 2,', '"schema": 1,', 1))),
        ("not JSON", lambda d: text(d, lambda t: "not json")),
        ("a cubin that leaves the directory", lambda d: text(d, lambda t: t.replace('"cubin": "softmax.sm_103.cubin"', '"cubin": "../sm_120/softmax.sm_120.cubin"', 1))),
        ("an absolute cubin name", lambda d: text(d, lambda t: t.replace('"cubin": "softmax.sm_103.cubin"', '"cubin": "/etc/passwd"', 1))),
        ("a missing cubin", lambda d: (d / "swiglu.sm_103.cubin").unlink()),
        ("a tampered cubin", flip_last),
        ("a tampered cubin with its file digest rewritten", flip_code),
        ("a kernel that uses shared memory with shared = 0", lambda d: doc(d, lambda v: entry(v, "softmax").update(shared=0))),
        ("an entry twice", lambda d: doc(d, duplicate_entry)),
        ("a missing key", lambda d: text(d, lambda t: t.replace('"symbol": "tf_softmax",', "", 1))),
    ]


@needs_crate
@pytest.mark.parametrize("name, mutate", _mutations(), ids=[n for n, _ in _mutations()])
def test_python_refuses_what_the_rust_tests_refuse(name, mutate, tmp_path):
    shutil.copytree(FIXTURES / "sm_103", tmp_path / "m")
    mutate(tmp_path / "m")
    with pytest.raises(TilefishManifestError):
        Manifest.load(tmp_path / "m")
