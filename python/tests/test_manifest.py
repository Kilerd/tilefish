"""Loading a manifest directory back (``tilefish.Manifest``): what ``replay`` runs, and what it refuses.

A manifest is written by ``write_manifest(built, path, cubins=True)`` and read by ``Manifest.load``, which checks
every cubin against the digests the manifest records; ``Manifest.built_kernel(spec, arch)`` then checks that the
entry is what a compile spec asks for, and that it was built from the current source of the kernel. No triton and no
GPU: the cubins are small ELF files made by hand. On a GPU, ``test_gpu_byte_identity.py`` and ``test_gpu_build.py``
replay real ones.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import dataclasses
import hashlib
import json
import os

import pytest
from built_fakes import FAKE_SOURCE_DIGEST, fake_build, rowop, scale
from elf_builder import S, SHT_PROGBITS, arch_header, build_elf
from elf_builder import cubin_like as any_cubin_like

import tilefish
from tilefish import Dim, Manifest, Out, Tensor, TilefishBuildError, TilefishManifestError, TilefishRuntimeError, TilefishSpecError, bf16, const, f32
from tilefish._elf import code_digest, parse_elf
from tilefish._manifest import manifest_digest
from tilefish._ir import IR_DIGEST_KIND
from tilefish.aot import compile_specs, write_manifest
from tilefish.runtime import _load_manifest

SRC = FAKE_SOURCE_DIGEST
TRITON = "3.5.1"  # the Triton version fake_build says built it


def cubin_like(arch="sm_90", **kw):
    """A small cubin for ``arch`` (sm_90, the architecture of the manifests here): its ELF header names it, as a reader requires."""
    return any_cubin_like(**{**arch_header(arch), **kw})


@pytest.fixture
def directory(tmp_path):
    """A manifest directory with ``scale`` and the two buckets of ``rowop``, built for sm_90."""
    built = [fake_build(s) for s in compile_specs(scale)] + [fake_build(s) for s in compile_specs(rowop)]
    write_manifest(built, tmp_path / "manifest.json", cubins=True)
    return tmp_path


def edit_manifest(directory, change):
    path = directory / "manifest.json"
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc))


def cubin_of(directory, stem="scale"):
    return directory / f"{stem}.sm_90.cubin"


# -- loading ---------------------------------------------------------------------------------------------------------------------


def test_load_reads_every_binary_and_verifies_it(directory):
    m = Manifest.load(directory)
    assert sorted(a.stem for a in m.artifacts) == ["rowop.S32", "rowop.S64", "scale"]
    assert m.get("scale", "sm_90").cubin == cubin_like() and m.get("scale", "sm_90").sha256_code == code_digest(cubin_like())
    assert ("rowop.S64", "sm_90") in m and ("rowop.S64", "sm_100") not in m and m.get("scale", "sm_100") is None
    assert Manifest.load(directory / "manifest.json").directory == directory  # the file itself works too
    assert m.path == directory / "manifest.json" and "3 binaries" in repr(m)


def test_load_is_the_same_through_tilefish_and_aot(directory):
    assert tilefish.Manifest is tilefish.aot.Manifest is Manifest


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d.update(schema=4), "not a tilefish manifest of schema 3"),
        (lambda d: d.update(schema=2), "schema 2, which an older tilefish wrote.*Re-export it"),  # what the Rust host 0.0.1 reads
        (lambda d: d.update(schema=1), "schema 1, which an older tilefish wrote.*Re-export it"),
        (lambda d: d.update(schema=True), "not a tilefish manifest of schema 3"),
        (lambda d: d.update(schema=3.0), "not a tilefish manifest of schema 3"),  # 3.0 == 3 in Python; a strict host refuses it
        (lambda d: d.update(schema=2.5), "not a tilefish manifest of schema 3"),
        (lambda d: d.update(schema=1.0), "not a tilefish manifest of schema 3"),
        (lambda d: d.update(schema="3"), "not a tilefish manifest of schema 3"),
        (lambda d: d.update(schema=[3]), "not a tilefish manifest of schema 3"),
        (lambda d: d.update(schema=None), "not a tilefish manifest of schema 3"),
        (lambda d: d.update(schema={"v": 3}), "not a tilefish manifest of schema 3"),
        (lambda d: d.pop("schema"), "not a tilefish manifest of schema 3"),
        (lambda d: d.update(kernels="nope"), "not a tilefish manifest"),
        (lambda d: d.pop("library"), "not a tilefish manifest"),
        (lambda d: d.update(library={}), "not a tilefish manifest"),
        (lambda d: d["kernels"].append(d["kernels"][0]), "twice"),
        (lambda d: d["kernels"][0].pop("sha256_code"), "is missing \\['sha256_code'\\]"),
        (lambda d: d["kernels"].__setitem__(0, 7), "is not an object"),
        (lambda d: d["kernels"][0].update(cubin="../outside.cubin"), "plain file name"),
        (lambda d: d["kernels"][0].update(cubin="sub/dir.cubin"), "plain file name"),
        (lambda d: d["kernels"][0].update(cubin="/etc/passwd"), "plain file name"),
        (lambda d: d["kernels"][0].update(cubin=".."), "plain file name"),
        (lambda d: d["kernels"][0].update(cubin=""), "plain file name"),
        (lambda d: d["kernels"][0].update(cubin="missing.cubin"), "cannot read the cubin 'missing.cubin'"),
        (lambda d: d["kernels"][0].update(symbol="tf_other"), "no code section '.text.tf_other'"),
        (lambda d: d["kernels"][0].update(shared=-1), "shared must be a non-negative integer"),
        (lambda d: d["kernels"][0].update(shared=True), "shared must be a non-negative integer"),
        (lambda d: d["kernels"][0].update(shared="16"), "shared must be a non-negative integer"),
    ],
)
def test_load_refuses_a_malformed_manifest(directory, change, message):
    edit_manifest(directory, change)
    with pytest.raises(TilefishManifestError, match=message):
        Manifest.load(directory)


def test_load_refuses_a_missing_or_unreadable_manifest(tmp_path):
    with pytest.raises(TilefishManifestError, match="cannot read the manifest"):
        Manifest.load(tmp_path)
    (tmp_path / "manifest.json").write_text("{not json")
    with pytest.raises(TilefishManifestError, match="not valid JSON"):
        Manifest.load(tmp_path)
    (tmp_path / "manifest.json").write_text('{"schema": 1, "kernels": [], "x": NaN}')
    with pytest.raises(TilefishManifestError, match="not strict JSON: it contains the constant NaN"):
        Manifest.load(tmp_path)


def test_load_refuses_a_cubin_that_is_not_the_file_that_was_built(directory):
    path = cubin_of(directory)
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(TilefishManifestError, match=r"'scale' \(sm_90\): the file 'scale.sm_90.cubin' has sha256 .* the manifest says"):
        Manifest.load(directory)


def test_load_refuses_changed_machine_code_even_when_the_file_digest_was_fixed_up(directory):
    path = cubin_of(directory)
    blob = bytearray(path.read_bytes())
    text = parse_elf(bytes(blob)).section(".text.tf_k")
    blob[text.offset + 3] ^= 0xFF  # one byte of machine code
    path.write_bytes(bytes(blob))
    edit_manifest(directory, lambda d: d["kernels"][0].update(sha256=hashlib.sha256(bytes(blob)).hexdigest()))
    with pytest.raises(TilefishManifestError, match=r"the machine code of 'scale.sm_90.cubin' has sha256_code .* the binary was changed after it was built"):
        Manifest.load(directory)


def test_a_change_to_the_debug_information_alone_is_not_a_change_of_the_code(directory):
    """``sha256_code`` ignores debug sections, so a rebuilt-elsewhere cubin passes once its file digest is updated."""
    path = cubin_of(directory)
    other = cubin_like(debug="/some/other/build/directory/with/a/longer/path/kernel.py")
    assert other != path.read_bytes() and code_digest(other) == code_digest(path.read_bytes())
    path.write_bytes(other)
    edit_manifest(directory, lambda d: d["kernels"][0].update(sha256=hashlib.sha256(other).hexdigest()))
    assert Manifest.load(directory).get("scale", "sm_90").cubin == other


def test_load_refuses_a_file_that_is_not_an_elf_cubin(directory):
    junk = b"this is not an ELF file at all" * 10
    cubin_of(directory).write_bytes(junk)
    edit_manifest(directory, lambda d: d["kernels"][0].update(sha256=hashlib.sha256(junk).hexdigest()))
    with pytest.raises(TilefishManifestError, match="is not a valid cubin: malformed ELF file"):
        Manifest.load(directory)


# -- shared memory --------------------------------------------------------------------------------------------------------------------


def test_a_kernel_that_uses_shared_memory_needs_shared_in_the_manifest(directory):
    """With ``shared = 0`` the launch would give the kernel no dynamic shared memory and it would fault."""
    edit_manifest(directory, lambda d: d["kernels"][0].update(shared=0))
    with pytest.raises(TilefishManifestError, match=r"has a shared memory section \('.nv.shared.tf_k'\).* shared = 0"):
        Manifest.load(directory)


def test_the_section_is_what_says_a_kernel_uses_shared_memory_not_its_size(tmp_path):
    """On sm_80 the section of a kernel that uses shared memory has size 0; the section's presence is the signal."""
    (spec,) = compile_specs(scale)
    for shared_section in (1024, 0):
        write_manifest([fake_build(spec, cubin=cubin_like(shared=shared_section), shared=0)], tmp_path / "manifest.json", cubins=True)
        with pytest.raises(TilefishManifestError, match="has a shared memory section"):
            Manifest.load(tmp_path)
        write_manifest([fake_build(spec, cubin=cubin_like(shared=shared_section), shared=16)], tmp_path / "manifest.json", cubins=True)
        assert Manifest.load(tmp_path).built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON).shared == 16


def test_a_kernel_without_shared_memory_may_have_shared_zero(tmp_path):
    (spec,) = compile_specs(scale)
    plain = cubin_like(shared=None)
    assert parse_elf(plain).section(".nv.shared.tf_k") is None
    write_manifest([fake_build(spec, cubin=plain, shared=0)], tmp_path / "manifest.json", cubins=True)
    assert Manifest.load(tmp_path).built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON).shared == 0


def test_shared_memory_of_the_manifest_is_what_the_binary_has(directory):
    spec, _ = specs()
    assert Manifest.load(directory).built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON).shared == 16


# -- the binary for a spec ---------------------------------------------------------------------------------------------------------


def specs():
    return compile_specs(scale)[0], compile_specs(rowop)


def test_built_kernel_is_the_binary_for_a_spec(directory):
    spec, (s32, s64) = specs()
    m = Manifest.load(directory)
    b = m.built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON)
    assert b.spec is spec and b.arch == "sm_90" and b.symbol == "tf_k" and b.cubin == cubin_like()
    assert b.ptx == "" and b.sha256 == hashlib.sha256(b.cubin).hexdigest() and b.sha256_code == code_digest(b.cubin)
    assert b.num_warps == 4 and b.shared == 16 and b.options == {"num_warps": 4, "num_stages": 3} and b.ir_digest == SRC
    assert b.launch.replay_blockers() == [] and b.triton_version == "3.5.1" and "V12.9.86" in b.ptxas_version
    assert b.abi() == fake_build(spec).abi()
    assert m.built_kernel(s64, "sm_90", ir_digest=SRC, ir_triton=TRITON).stem == "rowop.S64"


def test_built_kernel_needs_the_architecture(directory):
    spec, (s32, _) = specs()
    m = Manifest.load(directory)
    with pytest.raises(TilefishManifestError, match=r"no binary for 'scale' on sm_100 \(it has sm_90\)"):
        m.built_kernel(spec, "sm_100", ir_digest=SRC, ir_triton=TRITON)
    with pytest.raises(TilefishManifestError, match=r"no binary for 'other' on sm_90 \(it has \['rowop.S32', 'rowop.S64', 'scale'\]\)"):
        renamed = dataclasses.replace(spec, stem="other")
        renamed._mark_verified()
        m.built_kernel(renamed, "sm_90", ir_digest=SRC, ir_triton=TRITON)


ENTRY_CHANGES = [
    (lambda e: e.update(grid="(n,)"), "grid"),
    (lambda e: e.update(block=[64, 1, 1]), "block"),
    (lambda e: e.update(num_warps=8), "num_warps"),
    (lambda e: e["constexprs"][1].update(value=512), "constexprs[1].value"),
    (lambda e: e["abi"]["params"][0].update(dtype="f32", type="*fp32"), "abi.params[0].type"),  # (a dtype that is not its type the reader refuses)
    (lambda e: e["abi"]["scratch"].pop(), "abi.scratch"),
    (lambda e: e["preconditions"].pop(), "preconditions"),
    (lambda e: e.update(declaration_digest="0" * 64), "declaration_digest"),
    (lambda e: e["options"].update(enable_fp_fusion=False), "options.enable_fp_fusion"),
    (lambda e: e["options"].update(num_stages=2), "options.num_stages"),
]


@pytest.mark.parametrize("change, difference", ENTRY_CHANGES)
def test_built_kernel_refuses_an_entry_that_is_not_what_the_spec_asks_for(directory, change, difference):
    edit_manifest(directory, lambda d: change(d["kernels"][0]))
    spec, _ = specs()
    m = Manifest.load(directory)  # still a perfectly good manifest: the cubins are what was built
    with pytest.raises(TilefishManifestError, match=r"not what the compile spec asks for: first difference at " + difference.replace("[", r"\[").replace("]", r"\]")):
        m.built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON)


@pytest.mark.parametrize("key", ["op", "abi", "grid", "block", "num_warps", "num_stages", "constexprs", "buckets", "preconditions", "declaration_digest", "dims"])
def test_an_entry_that_lacks_a_field_of_the_spec_is_not_taken_to_agree_with_it(directory, key):
    """A missing field must not read as "whatever the spec says": the entry has to say it.

    What an entry says of itself (its operator and buckets, which make its stem, its ABI, grid, dimensions, preconditions and declaration
    digest) the reader requires, as the Rust reader does; the rest is told missing when the entry is compared with the spec.
    """
    edit_manifest(directory, lambda d: d["kernels"][0].pop(key))
    spec, _ = specs()
    if key in ("op", "abi", "grid", "buckets", "preconditions", "declaration_digest", "dims"):
        with pytest.raises(TilefishManifestError, match=rf"manifest kernel 0 \('scale'\) is missing \['{key}'\]"):
            Manifest.load(directory)
        return
    with pytest.raises(TilefishManifestError, match=r"not what the compile spec asks for: first difference at " + key):
        Manifest.load(directory).built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON)


def test_built_kernel_refuses_a_binary_of_another_declaration(directory):
    """The same stem, another extern (here with a different constexpr default): the declaration digest and the constants differ."""
    n = Dim("n", divisible_by=16)

    @tilefish.extern(triton="my_kernels:scale", num_warps=4, grid=lambda n: (tilefish.cdiv(n, 256),), name="scale")
    def scale_with_another_alpha(
        x: Tensor[n, bf16], y: Out[Tensor[n, bf16]], n_elem: n, *, ALPHA: const[f32] = 2.0, BLOCK: const = 256
    ): ...

    (spec,) = compile_specs(scale_with_another_alpha)
    assert spec.stem == "scale"
    with pytest.raises(TilefishManifestError, match="not what the compile spec asks for"):
        Manifest.load(directory).built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON)


def test_launch_attributes_that_replay_cannot_set_up_are_kept_for_the_driver_check(directory):
    edit_manifest(directory, lambda d: d["kernels"][0]["launch"].update(global_scratch_size=64, launch_pdl=True))
    spec, _ = specs()
    b = Manifest.load(directory).built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON)
    assert b.launch.replay_blockers() == ["global_scratch_size=64", "launch_pdl=True"]  # refused when it is loaded into the driver
    edit_manifest(directory, lambda d: d["kernels"][0].update(launch={"global_scratch_size": 0}))
    with pytest.raises(TilefishManifestError, match="launch attributes are malformed"):
        Manifest.load(directory).built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON)


# -- the source: a stale binary is refused ----------------------------------------------------------------------------------------------


def rows_body(factor):
    """The same kernel (name, signature) with another body: the p1 scenario."""
    rows = Dim("R")
    d = Dim("D", 64)

    @tilefish.kernel(name="scale_k")
    def scale_k(x: Tensor[rows, d, f32]) -> Tensor[rows, d, f32]:
        return x * factor

    return scale_k


def test_a_generated_kernel_with_another_body_is_not_the_manifests_binary(tmp_path):
    old, new = rows_body(2.0).lower(), rows_body(3.0).lower()
    assert old.specs()[0].stem == new.specs()[0].stem == "scale_k"  # the same name, the same signature ...
    assert old.source_sha256 != new.source_sha256  # ... another source
    assert old.specs()[0].declaration_digest != new.specs()[0].declaration_digest
    write_manifest([fake_build(s, symbol="tf_k") for s in old.specs()], tmp_path / "manifest.json", cubins=True)
    m = Manifest.load(tmp_path)
    (old_spec,), (new_spec,) = old.specs(), new.specs()
    assert m.built_kernel(old_spec, "sm_90").spec is old_spec  # the manifest is fine for the kernel it was built from
    with pytest.raises(TilefishManifestError) as e:
        m.built_kernel(new_spec, "sm_90")
    message = str(e.value)
    assert old.source_sha256 in message and new.source_sha256 in message  # both hashes are named
    assert "'scale_k' (sm_90)" in message and "changed since the manifest was written" in message and "stale" in message


def test_a_generated_kernels_entry_records_its_source(tmp_path):
    low = rows_body(2.0).lower()
    write_manifest([fake_build(s) for s in low.specs()], tmp_path / "manifest.json", cubins=True)
    entry = json.loads((tmp_path / "manifest.json").read_text())["kernels"][0]
    assert entry["source_sha256"] == low.source_sha256 == hashlib.sha256(low.source.encode()).hexdigest()
    assert entry["declaration_digest"] == low.specs()[0].declaration_digest


def test_an_entry_without_the_source_of_a_generated_kernel_is_refused(tmp_path):
    low = rows_body(2.0).lower()
    write_manifest([fake_build(s) for s in low.specs()], tmp_path / "manifest.json", cubins=True)
    edit_manifest(tmp_path, lambda d: d["kernels"][0].pop("source_sha256"))
    with pytest.raises(TilefishManifestError, match="was built from kernel source with sha256 None"):
        Manifest.load(tmp_path).built_kernel(low.specs()[0], "sm_90")


def test_a_hand_written_kernels_source_digest_must_match(directory):
    spec, _ = specs()
    m = Manifest.load(directory)
    assert m.built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON).stem == "scale"
    other = hashlib.sha256(b"an edited kernel").hexdigest()
    with pytest.raises(TilefishManifestError) as e:
        m.built_kernel(spec, "sm_90", ir_digest=other, ir_triton=TRITON)
    assert SRC in str(e.value) and other in str(e.value) and "my_kernels:scale" in str(e.value) and "stale" in str(e.value)
    with pytest.raises(TilefishManifestError, match="stale"):
        m.built_kernel(spec, "sm_90", ir_digest=other, ir_triton=TRITON, pinned=True)  # a pin does not excuse a digest that is known to differ


def test_without_a_recomputed_source_digest_a_hand_written_kernel_needs_a_pin(directory):
    spec, _ = specs()
    m = Manifest.load(directory)
    with pytest.raises(TilefishManifestError, match=r"cannot tell whether the binary for 'scale' \(sm_90\) was built from the kernel 'my_kernels:scale' as the process holds it: the IR of 'my_kernels:scale' cannot be recomputed here"):
        m.built_kernel(spec, "sm_90")
    assert m.built_kernel(spec, "sm_90", pinned=True).stem == "scale"  # the caller vouches for the manifest


def test_an_entry_without_a_source_digest_needs_a_pin(tmp_path):
    (spec,) = compile_specs(scale)
    write_manifest([fake_build(spec, ir_digest=None)], tmp_path / "manifest.json", cubins=True)
    entry = json.loads((tmp_path / "manifest.json").read_text())["kernels"][0]
    assert "source_digest" not in entry and "source_digest_kind" not in entry
    m = Manifest.load(tmp_path)
    with pytest.raises(TilefishManifestError, match="the manifest entry records no IR digest"):
        m.built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON)
    assert m.built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON, pinned=True).ir_digest is None


def test_an_ir_digest_made_by_another_triton_cannot_be_compared_and_needs_a_pin(directory):
    """The same kernel may print differently in another Triton: not a mismatch, not a match, only a pin vouches."""
    spec, _ = specs()
    m = Manifest.load(directory)
    other = hashlib.sha256(b"printed differently").hexdigest()
    with pytest.raises(TilefishManifestError, match=r"the entry was built with Triton 3.5.1, this process has Triton 3.6.0, .*cannot be compared"):
        m.built_kernel(spec, "sm_90", ir_digest=other, ir_triton="3.6.0")
    assert m.built_kernel(spec, "sm_90", ir_digest=other, ir_triton="3.6.0", pinned=True).stem == "scale"
    # a replay that did not even run the frontend of the other Triton says the same: no digest, and the version it has
    with pytest.raises(TilefishManifestError, match=r"the entry was built with Triton 3.5.1, this process has Triton 3.6.0, .*cannot be compared"):
        m.built_kernel(spec, "sm_90", ir_digest=None, ir_triton="3.6.0")
    assert m.built_kernel(spec, "sm_90", ir_digest=None, ir_triton="3.6.0", pinned=True).stem == "scale"
    with pytest.raises(TilefishManifestError, match="cannot be recomputed here"):  # and with no Triton at all, the other message
        m.built_kernel(spec, "sm_90", ir_digest=None, ir_triton=None)


def test_an_entry_that_does_not_record_its_triton_says_so_and_never_says_none(directory):
    """The toolchain of an entry can be missing (an entry written by hand, or an older tilefish): the message does not read 'built with Triton None'."""
    spec, _ = specs()
    edit_manifest(directory, lambda d: d["kernels"][0]["toolchain"].pop("triton"))
    m = Manifest.load(directory)
    other = hashlib.sha256(b"printed differently").hexdigest()
    for digest in (other, None):
        with pytest.raises(TilefishManifestError) as e:
            m.built_kernel(spec, "sm_90", ir_digest=digest, ir_triton="3.6.0")
        message = str(e.value)
        assert "does not record which Triton built it, and this process has Triton 3.6.0" in message and "cannot be compared" in message
        assert "Triton None" not in message
    assert m.built_kernel(spec, "sm_90", ir_digest=other, ir_triton="3.6.0", pinned=True).stem == "scale"


def test_a_generated_kernels_entry_records_its_ir_and_is_told_by_its_source_without_triton(tmp_path):
    low = rows_body(2.0).lower()
    (spec,) = low.specs()
    write_manifest([fake_build(spec)], tmp_path / "manifest.json", cubins=True)
    entry = json.loads((tmp_path / "manifest.json").read_text())["kernels"][0]
    assert entry["source_sha256"] == low.source_sha256 and entry["source_digest"] == SRC and entry["source_digest_kind"] == IR_DIGEST_KIND
    m = Manifest.load(tmp_path)
    assert m.built_kernel(spec, "sm_90").stem == spec.stem  # no Triton, no pin: its source is what is checked
    other = hashlib.sha256(b"another kernel").hexdigest()
    with pytest.raises(TilefishManifestError, match="stale"):  # with Triton, a different IR is stale too
        m.built_kernel(spec, "sm_90", ir_digest=other, ir_triton=TRITON)
    assert m.built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON).ir_digest == SRC
    assert m.built_kernel(spec, "sm_90", ir_digest=other, ir_triton="3.6.0").stem == spec.stem  # another Triton: not comparable, not an error here


# -- the digests of the source are read as the Rust reader reads them (crates/tilefish/src/manifest.rs) -------------------------------------


@pytest.mark.parametrize("field", ["source_digest", "source_sha256"])
@pytest.mark.parametrize("bad", ["A" * 64, "a" * 63, "a" * 65, "g" * 64, "", " " + "a" * 63, "a" * 63 + "é", 5, True, ["a" * 64], {"a": 1}])
def test_a_source_digest_that_is_not_64_lowercase_hex_digits_is_refused_when_the_manifest_is_loaded(directory, field, bad):
    edit_manifest(directory, lambda d: d["kernels"][0].update({field: bad}))
    with pytest.raises(TilefishManifestError, match=rf"'scale' \(sm_90\): {field} must be 64 lowercase hex digits"):
        Manifest.load(directory)


def test_a_source_digest_and_its_kind_come_together_and_a_json_null_is_as_absent(directory):
    for change in (lambda e: e.pop("source_digest_kind"), lambda e: e.pop("source_digest"), lambda e: e.update(source_digest_kind=None), lambda e: e.update(source_digest=None)):
        edit_manifest(directory, lambda d: d["kernels"][0].update(source_digest=SRC, source_digest_kind=IR_DIGEST_KIND))
        edit_manifest(directory, lambda d, change=change: change(d["kernels"][0]))
        with pytest.raises(TilefishManifestError, match="source_digest and source_digest_kind come together"):
            Manifest.load(directory)
    edit_manifest(directory, lambda d: d["kernels"][0].update(source_digest=None, source_digest_kind=None))  # both null: both absent
    assert Manifest.load(directory).get("scale", "sm_90").entry["source_digest"] is None
    edit_manifest(directory, lambda d: (d["kernels"][0].pop("source_digest"), d["kernels"][0].pop("source_digest_kind")))
    assert Manifest.load(directory).get("scale", "sm_90") is not None  # and both missing: an entry that records none


def test_the_kind_of_a_source_digest_is_a_string_and_any_string_is_read(directory):
    edit_manifest(directory, lambda d: d["kernels"][0].update(source_digest_kind=7))
    with pytest.raises(TilefishManifestError, match="source_digest_kind must be a string"):
        Manifest.load(directory)
    edit_manifest(directory, lambda d: d["kernels"][0].update(source_digest_kind="another.definition.v9"))  # Rust does not judge the kind either
    assert Manifest.load(directory).get("scale", "sm_90").entry["source_digest_kind"] == "another.definition.v9"


def test_the_source_digest_of_an_entry_is_labelled(directory):
    entry = json.loads((directory / "manifest.json").read_text())["kernels"][0]
    assert entry["source_digest"] == SRC and entry["source_digest_kind"] == IR_DIGEST_KIND == "tilefish.triton-ir.v1"


def test_a_source_digest_of_another_kind_says_nothing_about_the_ir(tmp_path):
    """An older tilefish recorded Triton's ``cache_key`` (or a digest of files) there; it cannot be compared with a digest of IR."""
    (spec,) = compile_specs(scale)
    write_manifest([fake_build(spec)], tmp_path / "manifest.json", cubins=True)
    edit_manifest(tmp_path, lambda d: d["kernels"][0].update(source_digest_kind="triton.JITFunction.cache_key"))
    m = Manifest.load(tmp_path)
    with pytest.raises(TilefishManifestError, match=r"source digest is of kind 'triton.JITFunction.cache_key', not 'tilefish.triton-ir.v1'"):
        m.built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON)  # the same hash, the other kind: not taken as a match
    assert m.built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON, pinned=True).stem == "scale"  # only a pin vouches for it
    edit_manifest(tmp_path, lambda d: d["kernels"][0].pop("source_digest_kind"))
    with pytest.raises(TilefishManifestError, match="source_digest and source_digest_kind come together"):  # the reader of Rust refuses it as well
        Manifest.load(tmp_path)


# -- what an entry says of itself is read as the Rust reader reads it (crates/tilefish/src/manifest.rs) -----------------------------------
#
# Each document below is a valid manifest changed in one place, and loaded with its digest pinned again: the pin vouches for the document, so
# the refusal can only be the check of the reader (the Rust reader refuses each of them, with the same message).


def repinned(directory, change):
    """Change the manifest in ``directory`` with ``change(doc)`` and load it pinned to the digest of the changed document."""
    edit_manifest(directory, change)
    return Manifest.load(directory, pin=manifest_digest(json.loads((directory / "manifest.json").read_text())))


def kernel(doc, stem="scale"):
    return next(e for e in doc["kernels"] if e["stem"] == stem)


def param_of(entry, name):
    return next(p for p in entry["abi"]["params"] if p["name"] == name)


@pytest.mark.parametrize("field, value", [("dtype", "i32"), ("access", "in"), ("shape", []), ("strides", [1]), ("offset", "n_offset"), ("value", 7)])
def test_a_scalar_parameter_with_a_field_of_a_pointer_or_a_fixed_argument_is_refused(directory, field, value):
    with pytest.raises(TilefishManifestError, match=rf"'scale' \(sm_90\): parameter n_elem is a scalar: it has a dim and no dtype, access, shape, strides or value \(it has \['{field}'\]\)"):
        repinned(directory, lambda d: param_of(kernel(d), "n_elem").update({field: value}))


def test_a_field_of_a_parameter_that_is_null_is_as_absent_and_one_no_parameter_has_is_refused(directory):
    assert repinned(directory, lambda d: param_of(kernel(d), "n_elem").update(dtype=None, access=None)).get("scale", "sm_90") is not None
    with pytest.raises(TilefishManifestError, match=r"parameter x is not a well-formed 64-bit pointer"):  # and a pointer has no dim
        repinned(directory, lambda d: param_of(kernel(d), "x").update(dim="n"))
    with pytest.raises(TilefishManifestError, match=r"abi.params\[0\] \('x'\) has unknown fields \['note'\]"):
        repinned(directory, lambda d: param_of(kernel(d), "x").update(note="anything"))


@pytest.mark.parametrize(
    "stem, change, message",
    [
        ("scale", lambda e: e.update(buckets={"S": 32}), r"stem 'scale' does not match op 'scale' and buckets \{'S': 32\}: .* 'scale.S32'"),
        ("rowop.S32", lambda e: e.update(stem="rowop.S64"), r"stem 'rowop.S64' does not match op 'rowop' and buckets \{'S': 32\}"),
        ("rowop.S32", lambda e: e.update(stem="rowop.S32x"), r"stem 'rowop.S32x' does not match op 'rowop'"),
        ("scale", lambda e: e.update(op="scaled"), r"stem 'scale' does not match op 'scaled' and buckets \{\}"),
        ("rowop.S32", lambda e: e.update(buckets={"S": True}), r"bucket value of S must be an int >= 1, got True"),
        ("rowop.S32", lambda e: e.update(buckets={"S": 0}), r"bucket value of S must be an int >= 1, got 0"),
        ("scale", lambda e: e.update(op="scale.x"), r"op must be an identifier, got 'scale.x'"),
    ],
)
def test_a_stem_that_is_not_its_operator_and_its_buckets_is_refused(directory, stem, change, message):
    """The stem of a binary is its operator and the value of each of its buckets (``rowop.S32``), the name a host chooses a binary by."""
    with pytest.raises(TilefishManifestError, match=message):
        repinned(directory, lambda d: change(kernel(d, stem)))


@pytest.mark.parametrize("value", [None, "A" * 64, "a" * 63, 5])
def test_a_declaration_digest_that_is_null_or_not_a_sha256_is_refused(directory, value):
    with pytest.raises(TilefishManifestError, match=rf"'scale' \(sm_90\): declaration_digest must be 64 lowercase hex digits \(a sha256\), got {value!r}"):
        repinned(directory, lambda d: kernel(d).update(declaration_digest=value))


def test_an_entry_whose_architecture_is_not_the_one_its_cubin_is_for_is_refused(directory):
    """A cubin is used as it is, for the architecture its ELF header names: an entry relabelled ``sm_90`` that holds an sm_103 cubin is refused."""
    sm103 = cubin_like("sm_103")
    cubin_of(directory).write_bytes(sm103)
    with pytest.raises(TilefishManifestError, match=r"'scale' \(sm_90\): 'scale.sm_90.cubin' is for sm_103 \(its ELF header says so\), but the manifest says sm_90"):
        repinned(directory, lambda d: kernel(d).update(sha256=hashlib.sha256(sm103).hexdigest(), sha256_code=code_digest(sm103)))
    with pytest.raises(TilefishManifestError, match=r"arch must look like 'sm_103', got 'sm_103\\n'"):
        repinned(directory, lambda d: kernel(d).update(arch="sm_103\n"))
    assert repinned(directory, lambda d: kernel(d).update(arch="sm_103")).get("scale", "sm_103").cubin == sm103  # the label that is the cubin's
    for arch in ("sm_9", "sm_1030", "sm_90a", "SM_90", "sm_٩٠"):
        with pytest.raises(TilefishManifestError, match=r"arch must look like 'sm_103'"):
            repinned(directory, lambda d, arch=arch: kernel(d, "rowop.S32").update(arch=arch))


def test_a_triton_cubin_whose_header_names_no_architecture_is_left_to_the_driver(directory):
    """As the Rust reader: only a prebuilt binary must name its architecture (``test_binary_externs.py`` has that refusal)."""
    other = cubin_like(osabi=0x07)
    cubin_of(directory).write_bytes(other)
    m = repinned(directory, lambda d: kernel(d).update(sha256=hashlib.sha256(other).hexdigest(), sha256_code=code_digest(other)))
    assert m.get("scale", "sm_90").cubin == other


# -- the dimension names of an entry, and the 128-bit fold of its expressions -------------------------------------------------------------


def test_an_entry_names_at_most_32_dimensions_and_every_one_it_uses_is_in_dims(directory):
    """The Rust host binds at most 32 dimension names per entry (``expr::MAX_DIMS``): the names of ``dims`` and of every expression together."""
    from tilefish._manifest import MAX_DIMS

    assert MAX_DIMS == 32
    names = {f"d{i}": {} for i in range(31)}  # and n: 32
    assert repinned(directory, lambda d: kernel(d)["dims"].update(names)).get("scale", "sm_90") is not None
    with pytest.raises(TilefishManifestError, match=r"'scale' \(sm_90\): a kernel may use at most 32 dimension names, and this one has 33"):
        repinned(directory, lambda d: kernel(d)["dims"].update(d31={}))
    with pytest.raises(TilefishManifestError, match=r"at most 32 dimension names, and this one has 33"):  # the 33rd in an expression only
        repinned(directory, lambda d: (kernel(d)["dims"].pop("d31"), kernel(d).update(grid="(cdiv(n + m - m, 256),)")))
    edit_manifest(directory, lambda d: kernel(d)["dims"].pop("d30"))
    with pytest.raises(TilefishManifestError, match=r"dims does not name \['m'\], which the declared shapes"):
        repinned(directory, lambda d: kernel(d).update(grid="(cdiv(n + m - m, 256),)"))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda e: e.update(grid="(n)"), r"grid '\(n\)': a single entry needs a trailing comma"),
        (lambda e: e.update(grid="(1234567890123,)"), r"grid '\(1234567890123,\)': expected \), got '3'"),
        (lambda e: e.update(grid="(n - n,)"), r"entry 0 is the static size 0"),
        (lambda e: param_of(e, "x").update(shape=["n // 0"]), r"parameter x shape\[0\] 'n // 0': division by 0"),
        (lambda e: param_of(e, "n_elem").update(dim="n.real"), r"parameter n_elem dim 'n.real' has a character the dimension grammar does not allow"),
        (lambda e: param_of(e, "n_elem").update(dim=5), r"parameter n_elem: dim must be a str, got 5"),
    ],
)
def test_the_dimension_expressions_of_an_entry_are_in_the_grammar(directory, change, message):
    with pytest.raises(TilefishManifestError, match=message):
        repinned(directory, lambda d: change(kernel(d)))


X = "549755813888*549755813888*549755813888*1024"  # 2**127, the first integer beyond i128 (four 12-digit numbers)
Y = "549755813888*549755813888*549755813888*512"  # 2**126


def test_expressions_fold_in_128_bits_as_the_rust_reader_folds_them():
    """A constant or coefficient beyond i128 has no folded value in either reader: a static check that needs one is not made, in both."""
    from tilefish._specvalid import FOLD_RANGE, _value_of, parse_grid

    assert FOLD_RANGE == range(-(2**127), 2**127)
    assert _value_of(X, "e") is None and _value_of(Y, "e") == 2**126 and _value_of("-" + X, "e") == -(2**127)  # -2**127 is the least i128
    assert _value_of(f"0 - (-{X})", "e") is None  # negating -2**127 overflows
    assert _value_of(f"-1 - (-{X}) - {Y} - {Y}", "e") is None  # even when the result would fit: Rust subtracts by adding the negation
    assert parse_grid(f"({X} - {X},)") == set()  # 0 only in unbounded arithmetic: not the static grid (0,) here, nor in Rust
    with pytest.raises(TilefishSpecError, match="entry 0 is the static size 0"):
        parse_grid(f"({Y} - {Y},)")  # within 128 bits it is
    # a cdiv by a symbolic divisor whose ratio is 2**127 stays the cdiv atom (Rust's checked ratio), so that `* 0` folds it to 0
    assert str(_value_of(f"cdiv(-{X}*n, -n)", "e")).startswith("cdiv(") and _value_of("cdiv(2*n, -n)", "e") == -2
    with pytest.raises(TilefishSpecError, match="entry 0 is the static size 0"):
        parse_grid(f"(cdiv(-{X}*n, -n)*0,)")
    assert parse_grid(f"(cdiv(-{X}*n, -n) - {Y} - {Y},)") == {"n"}


def test_a_manifest_whose_expressions_leave_128_bits_is_read_as_the_rust_reader_reads_it(directory):
    assert repinned(directory, lambda d: kernel(d).update(grid=f"(cdiv(n, 256) + {X} - {X},)")).get("scale", "sm_90") is not None
    with pytest.raises(TilefishManifestError, match="entry 0 is the static size 0"):
        repinned(directory, lambda d: kernel(d).update(grid=f"({Y} - {Y},)"))
    with pytest.raises(TilefishManifestError, match=r"parameter x shape\[0\] .*: cdiv by 0 \(the divisor must be positive\)"):
        repinned(directory, lambda d: param_of(kernel(d), "x").update(shape=[f"cdiv(n, {Y} - {Y})"]))


# -- the digest of the manifest --------------------------------------------------------------------------------------------------------


def test_the_manifest_has_a_digest_of_everything_in_it(directory):
    m = Manifest.load(directory)
    assert len(m.digest) == 64 and set(m.digest) <= set("0123456789abcdef")
    assert m.digest == manifest_digest(json.loads((directory / "manifest.json").read_text())) == Manifest.load(directory).digest


def test_the_digest_does_not_depend_on_formatting_order_or_the_tilefish_version(directory):
    m = Manifest.load(directory)
    doc = json.loads((directory / "manifest.json").read_text())
    reordered = {
        "library": doc["library"],
        "kernels": list(reversed([dict(reversed(list(k.items()))) for k in doc["kernels"]])),
        "tilefish": "9.9.9",
        "schema": doc["schema"],
    }
    assert manifest_digest(reordered) == m.digest
    (directory / "manifest.json").write_text(json.dumps(doc, indent=4, sort_keys=True))
    assert Manifest.load(directory).digest == m.digest


DIGEST_CHANGES = [
    ("shared", lambda e: e.update(shared=32)),
    ("block", lambda e: e.update(block=[64, 1, 1])),
    ("launch", lambda e: e["launch"].update(cluster_dims=[2, 1, 1])),
    ("launch pdl", lambda e: e["launch"].update(launch_pdl=True)),
    ("toolchain", lambda e: e["toolchain"].update(ptxas="another ptxas")),
    ("symbol", lambda e: e.update(symbol="tf_k")),  # (the same: not a change)
    ("sha256", lambda e: e.update(sha256="0" * 64)),
    ("sha256_code", lambda e: e.update(sha256_code="0" * 64)),
    ("source_digest", lambda e: e.update(source_digest="1" * 64)),
    ("declaration_digest", lambda e: e.update(declaration_digest="2" * 64)),
    ("options", lambda e: e["options"].update(num_stages=2)),
    ("abi", lambda e: e["abi"]["params"][0].update(access="out")),
    ("grid", lambda e: e.update(grid="(n,)")),
    ("constexprs", lambda e: e["constexprs"][0].update(value=1.0)),
    ("preconditions", lambda e: e["preconditions"].pop()),
    ("arch", lambda e: e.update(arch="sm_100")),
    ("an entry", lambda e: None),
]


@pytest.mark.parametrize("label, change", DIGEST_CHANGES)
def test_the_digest_changes_with_every_field_of_every_entry(directory, label, change):
    doc = json.loads((directory / "manifest.json").read_text())
    before = manifest_digest(doc)
    if label == "an entry":
        doc["kernels"].pop()
    else:
        change(doc["kernels"][0])
    if label == "symbol":
        assert manifest_digest(doc) == before  # the entry already had this symbol
    else:
        assert manifest_digest(doc) != before


def test_the_digest_of_a_different_set_of_binaries_differs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_manifest([fake_build(s) for s in compile_specs(scale)], a / "manifest.json", cubins=True)
    write_manifest([fake_build(s, cubin=cubin_like(code=b"\x07" * 64)) for s in compile_specs(scale)], b / "manifest.json", cubins=True)
    assert Manifest.load(a).digest != Manifest.load(b).digest


def test_a_pinned_replay_refuses_another_digest(directory):
    pin = Manifest.load(directory).digest
    wrong = "f" * 64 if pin[0] != "f" else "e" * 64
    with pytest.raises(TilefishManifestError, match=f"has digest {pin}, but pin= says {wrong}"):
        tilefish.run(scale, mode="replay", manifest=directory, pin=wrong)
    edit_manifest(directory, lambda d: d["kernels"][0].update(shared=32))  # a change the file digests do not see
    changed = Manifest.load(directory)
    assert changed.digest != pin
    with pytest.raises(TilefishManifestError, match="it is not the manifest that was pinned"):
        tilefish.run(scale, mode="replay", manifest=directory, pin=pin)


def test_pin_is_a_digest_and_belongs_to_replay(directory):
    for bad in ("abc", 5, "G" * 64, "A" * 64, ""):
        with pytest.raises(TilefishRuntimeError, match="pin= is the 64 lowercase hex digits"):
            tilefish.run(scale, mode="replay", manifest=directory, pin=bad)
    for mode in ("jit", "reference"):
        with pytest.raises(TilefishRuntimeError, match="pin= is for mode='replay'"):
            tilefish.run(scale, mode=mode, pin="a" * 64)


# -- writing: merge, overwrite and safe names ------------------------------------------------------------------------------------------


def test_merging_keeps_the_other_architectures_and_kernels(directory):
    (spec,) = compile_specs(scale)
    other_arch = fake_build(spec, arch="sm_100", cubin=cubin_like("sm_100", code=b"\x09" * 64))
    path = write_manifest([other_arch], directory / "manifest.json", cubins=True, merge=True)
    m = Manifest.load(path)
    assert sorted((a.stem, a.arch) for a in m.artifacts) == [("rowop.S32", "sm_90"), ("rowop.S64", "sm_90"), ("scale", "sm_100"), ("scale", "sm_90")]
    assert m.get("scale", "sm_90").cubin == cubin_like() and m.get("scale", "sm_100").cubin == other_arch.cubin


def test_without_merge_the_manifest_is_replaced(directory):
    (spec,) = compile_specs(scale)
    write_manifest([fake_build(spec, arch="sm_100")], directory / "manifest.json", cubins=True)
    assert [(a.stem, a.arch) for a in Manifest.load(directory).artifacts] == [("scale", "sm_100")]


def test_a_new_build_of_the_same_binary_replaces_its_entry(directory):
    (spec,) = compile_specs(scale)
    rebuilt = fake_build(spec, cubin=cubin_like(debug="/a/longer/other/path.py"))  # same machine code, another build path
    assert rebuilt.sha256 != Manifest.load(directory).get("scale", "sm_90").entry["sha256"]
    write_manifest([rebuilt], directory / "manifest.json", cubins=True, merge=True)
    m = Manifest.load(directory)
    assert len(m.artifacts) == 3 and m.get("scale", "sm_90").entry["sha256"] == rebuilt.sha256


@pytest.mark.parametrize(
    "change, what",
    [
        ({"cubin": cubin_like(code=b"\x05" * 64)}, "sha256_code"),
        ({"shared": 32}, "shared"),
        ({"ir_digest": hashlib.sha256(b"edited").hexdigest()}, "source_digest"),
    ],
)
def test_a_conflicting_binary_replaces_an_entry_only_with_overwrite(directory, change, what):
    (spec,) = compile_specs(scale)
    different = fake_build(spec, **change)
    before = Manifest.load(directory).digest
    with pytest.raises(TilefishManifestError, match=rf"already has a different binary for 'scale' on sm_90 \({what} differs\); pass overwrite=True"):
        write_manifest([different], directory / "manifest.json", cubins=True, merge=True)
    assert Manifest.load(directory).digest == before  # nothing was written
    write_manifest([different], directory / "manifest.json", cubins=True, merge=True, overwrite=True)
    assert Manifest.load(directory).digest != before


def test_a_merge_into_a_manifest_that_does_not_verify_is_an_error(directory):
    cubin_of(directory).write_bytes(b"damaged")
    (spec,) = compile_specs(scale)
    with pytest.raises(TilefishManifestError, match="sha256"):
        write_manifest([fake_build(spec, arch="sm_100")], directory / "manifest.json", cubins=True, merge=True)


def test_the_cubins_are_written_inside_the_directory(tmp_path):
    (spec,) = compile_specs(scale)
    for stem in ("../evil", "a/b", "a\\b", ".hidden", "x\0y"):
        evil = dataclasses.replace(spec, stem=stem)
        evil._mark_verified()
        with pytest.raises(TilefishBuildError, match="not a plain file name"):
            fake_build(evil).write_cubin(tmp_path / "out")
        with pytest.raises(TilefishBuildError, match="not a plain file name"):
            write_manifest([fake_build(evil)], tmp_path / "m" / "manifest.json", cubins=True)
    for arch in ("../sm_90", "sm/90"):
        with pytest.raises(TilefishBuildError, match="not a plain file name"):
            fake_build(spec, arch=arch).write_cubin(tmp_path / "out")
    assert not (tmp_path / "evil.sm_90.cubin").exists() and not list((tmp_path / "out").glob("*"))
    assert fake_build(spec).write_cubin(tmp_path / "out").parent == tmp_path / "out"


def test_a_directory_that_is_a_symlink_elsewhere_is_still_the_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    (spec,) = compile_specs(scale)
    assert fake_build(spec).write_cubin(link).resolve().parent == real.resolve()


def test_a_cubin_name_that_is_a_symlink_out_of_the_directory_is_not_written_through(tmp_path):
    """A plain file name can still be a symlink that is already in the directory; writing would follow it."""
    (spec,) = compile_specs(scale)
    out = tmp_path / "out"
    out.mkdir()
    elsewhere = tmp_path / "elsewhere.bin"
    elsewhere.write_bytes(b"precious")
    build = fake_build(spec)
    (out / build.cubin_name).symlink_to(elsewhere)
    with pytest.raises(TilefishBuildError, match="is a symbolic link; refusing to write through it"):
        build.write_cubin(out)
    assert elsewhere.read_bytes() == b"precious" and (out / build.cubin_name).is_symlink()


def test_an_extern_name_is_a_plain_identifier():
    for bad in ("../evil", "a/b", "a b", "a\nb", "a.b", "", "1abc", "naïve", "a\x00b", "-x", "x;y"):
        with pytest.raises(tilefish.TilefishDeclarationError, match="name must be a plain identifier"):

            @tilefish.extern(triton="my_kernels:scale", grid=lambda n: (n,), name=bad)
            def stub(x: Tensor[Dim("n", divisible_by=16), bf16], n_elem: Dim("n", divisible_by=16)): ...

    with pytest.raises(tilefish.TilefishDeclarationError, match="name must be a plain identifier"):
        tilefish.extern(triton="my_kernels:scale", grid=lambda n: (n,), name=3)

    @tilefish.extern(triton="my_kernels:scale", grid=lambda n: (n,), name="_ok_Name_9")
    def stub(x: Tensor[Dim("n", divisible_by=16), bf16], n_elem: Dim("n", divisible_by=16)): ...

    assert stub.name == "_ok_Name_9"


def test_an_extern_function_name_that_is_not_ascii_needs_a_name():
    """Without ``name=`` the function's name is the op name, and Python allows identifiers that are no file name here."""

    def naïve(x: Tensor[Dim("n", divisible_by=16), bf16], n_elem: Dim("n", divisible_by=16)): ...

    with pytest.raises(tilefish.TilefishDeclarationError, match=r"function name 'naïve' is not a plain identifier"):
        tilefish.extern(triton="my_kernels:scale", grid=lambda n: (n,))(naïve)
    assert tilefish.extern(triton="my_kernels:scale", grid=lambda n: (n,), name="naive")(naïve).name == "naive"


# -- run(): the manifest argument --------------------------------------------------------------------------------------------------------


def test_replay_needs_a_manifest_and_only_replay_takes_one(directory):
    with pytest.raises(TilefishRuntimeError, match="replay runs prebuilt cubins from a manifest directory: pass manifest="):
        tilefish.run(scale, mode="replay")
    for mode in ("jit", "reference"):
        with pytest.raises(TilefishRuntimeError, match="manifest= is for mode='replay'"):
            tilefish.run(scale, mode=mode, manifest=directory)
    with pytest.raises(TilefishRuntimeError, match="manifest= is for mode='replay'"):
        scale.run(manifest=directory)  # jit is the default mode


def test_a_replay_needs_a_trust_anchor_and_does_not_read_the_directory_without_one(directory, monkeypatch):
    """``manifest=`` alone is not enough: the digests sit next to the files they describe, so the caller says which to trust."""
    from tilefish import runtime

    def read(manifest):
        raise AssertionError("the manifest was read before the caller said whether to trust it")

    monkeypatch.setattr(runtime, "_load_manifest", read)
    anchor = "replay needs a trust anchor.*pin=<the manifest's digest>.*or trust_directory=True"
    with pytest.raises(TilefishRuntimeError, match=anchor):
        tilefish.run(scale, mode="replay", manifest=directory)
    with pytest.raises(TilefishRuntimeError, match=anchor):
        scale.run(mode="replay", manifest=directory)  # the method passes it through
    with pytest.raises(TilefishRuntimeError, match=anchor):
        tilefish.run(scale, mode="replay", manifest=directory, trust_directory=False)
    with pytest.raises(TilefishRuntimeError, match=anchor):
        tilefish.run(scale, mode="replay", manifest=Manifest.load(directory))  # a loaded manifest is no anchor either


def test_the_trust_anchor_is_one_of_a_pin_and_trust_directory(directory, monkeypatch):
    from tilefish import runtime

    seen = []

    def load(manifest):
        seen.append(manifest)
        raise TilefishManifestError("stop here: the anchor was accepted")

    monkeypatch.setattr(runtime, "_load_manifest", load)
    with pytest.raises(TilefishManifestError, match="the anchor was accepted"):
        tilefish.run(scale, mode="replay", manifest=directory, trust_directory=True)
    with pytest.raises(TilefishManifestError, match="the anchor was accepted"):
        tilefish.run(scale, mode="replay", manifest=directory, pin="a" * 64)
    assert len(seen) == 2
    with pytest.raises(TilefishRuntimeError, match=r"pass either pin= .* or trust_directory=True .*, not both"):
        tilefish.run(scale, mode="replay", manifest=directory, pin="a" * 64, trust_directory=True)
    assert len(seen) == 2


@pytest.mark.parametrize("bad", [1, "yes", None, 0, "True"])
def test_trust_directory_is_a_bool(directory, bad):
    with pytest.raises(TilefishRuntimeError, match="trust_directory= is True or False"):
        tilefish.run(scale, mode="replay", manifest=directory, trust_directory=bad)


def test_trust_directory_belongs_to_replay(directory):
    for mode in ("jit", "reference"):
        with pytest.raises(TilefishRuntimeError, match="trust_directory= is for mode='replay'"):
            tilefish.run(scale, mode=mode, trust_directory=True)
    with pytest.raises(TilefishRuntimeError, match="trust_directory= is for mode='replay'"):
        scale.run(trust_directory=True)  # jit is the default mode


def test_a_manifest_directory_is_loaded_once_and_again_when_a_file_changes(directory):
    first = _load_manifest(directory)
    assert _load_manifest(directory) is first and _load_manifest(str(directory)) is first and _load_manifest(first) is first
    assert _load_manifest(directory / "manifest.json") is not None
    path = cubin_of(directory)
    other = cubin_like(debug="/a/b/c/d/e/f/longer/debug/path.py")
    path.write_bytes(other)
    edit_manifest(directory, lambda d: d["kernels"][0].update(sha256=hashlib.sha256(other).hexdigest()))
    second = _load_manifest(directory)
    assert second is not first and second.get("scale", "sm_90").cubin == other
    path.write_bytes(other + b"x")  # a changed cubin is seen on the next call, and refused
    with pytest.raises(TilefishManifestError, match="sha256"):
        _load_manifest(directory)


def test_a_manifest_must_be_a_directory_a_file_or_a_manifest():
    with pytest.raises(TilefishRuntimeError, match="manifest= takes a directory or a Manifest"):
        _load_manifest(12)
    with pytest.raises(TilefishManifestError, match="cannot read the manifest directory"):
        _load_manifest("/no/such/directory/at/all/manifest.json")


# -- what the manifest says ---------------------------------------------------------------------------------------------------------------


def test_a_generated_kernels_manifest_names_its_contraction_and_its_warps_per_bucket(tmp_path):
    """The entries of a bucketed lowered kernel carry their own num_warps and the contraction in the options."""
    import lower_cases

    Hb = tilefish.Dim("Hb", buckets=[64, 4096])

    @tilefish.kernel
    def rows(x: tilefish.Tensor[lower_cases.M, Hb, tilefish.f32]) -> tilefish.Tensor[lower_cases.M, Hb, tilefish.f32]:
        return x - tilefish.mean(x, axis=-1, keepdims=True)

    specs_ = rows.lower().specs()
    write_manifest([fake_build(s) for s in specs_], tmp_path / "manifest.json", cubins=True)
    doc = json.loads((tmp_path / "manifest.json").read_text())
    assert [(k["stem"], k["num_warps"], k["block"]) for k in doc["kernels"]] == [("rows.Hb64", 1, [32, 1, 1]), ("rows.Hb4096", 16, [512, 1, 1])]
    assert all(k["options"] == {"num_warps": k["num_warps"], "num_stages": 3, "enable_fp_fusion": True} for k in doc["kernels"])
    assert all(k["source_sha256"] == rows.lower().source_sha256 for k in doc["kernels"])
    m = Manifest.load(tmp_path)
    assert [m.built_kernel(s, "sm_90").num_warps for s in specs_] == [1, 16]
    strict = rows.lower(fp_fusion=False).specs()
    with pytest.raises(TilefishManifestError, match="stale|first difference at"):
        m.built_kernel(strict[0], "sm_90")


def test_a_minimal_cubin_with_its_code_section_loads(tmp_path):
    only_debug = build_elf([S(".text.tf_k", SHT_PROGBITS, data=b"\1\2\3\4"), S(".debug_line", data=b"x")], **arch_header("sm_90"))
    (spec,) = compile_specs(scale)
    write_manifest([fake_build(spec, cubin=only_debug, shared=0)], tmp_path / "manifest.json", cubins=True)
    assert Manifest.load(tmp_path).get("scale", "sm_90").sha256_code == code_digest(only_debug)


# -- strict JSON and typed fields ---------------------------------------------------------------------------------------------------------


def manifest_text(directory):
    return (directory / "manifest.json").read_text()


def test_an_object_with_the_same_key_twice_is_refused(directory):
    """Parsers disagree about which of two equal keys wins: an attacker picks the one a check reads and a host does not."""
    text = manifest_text(directory)
    (directory / "manifest.json").write_text(text.replace('"shared": 16', '"shared": 16, "shared": 0', 1))
    with pytest.raises(TilefishManifestError, match="an object has the key 'shared' twice"):
        Manifest.load(directory)
    (directory / "manifest.json").write_text('{"schema": 1, "schema": 1, "kernels": []}')
    with pytest.raises(TilefishManifestError, match="the key 'schema' twice"):
        Manifest.load(directory)


@pytest.mark.parametrize("number", ["1e999", "-1e999", "1E400", "123456789e999999"])
def test_a_number_that_does_not_fit_a_float_is_refused_when_the_manifest_is_read(directory, number):
    (directory / "manifest.json").write_text(manifest_text(directory).replace('"shared": 16', f'"shared": {number}', 1))
    with pytest.raises(TilefishManifestError, match=r"not strict JSON: the number .* does not fit a float"):
        Manifest.load(directory)


def test_the_digest_of_a_document_with_an_infinite_number_is_a_manifest_error_not_a_value_error(directory):
    doc = json.loads(manifest_text(directory))
    doc["kernels"][0]["shared"] = float("inf")
    with pytest.raises(TilefishManifestError, match="cannot be digested"):
        manifest_digest(doc)
    m = Manifest.load(directory)
    m.doc["kernels"][0]["constexprs"][0]["value"] = float("nan")  # a loaded manifest whose document was edited in memory
    with pytest.raises(TilefishManifestError, match="cannot be digested"):
        m.digest
    with pytest.raises(TilefishManifestError, match="cannot be digested"):
        manifest_digest({"schema": 1})  # no kernels at all


@pytest.mark.parametrize("field", ["toolchain", "options", "launch"])
@pytest.mark.parametrize("value", [None, [], "triton 3.5.1", 3])
def test_a_field_that_must_be_an_object_is_one(directory, field, value):
    edit_manifest(directory, lambda d: d["kernels"][0].update({field: value}))
    with pytest.raises(TilefishManifestError, match=rf"'scale' \(sm_90\): '{field}' must be an object"):
        Manifest.load(directory)


def test_a_manifest_that_is_not_utf8_or_is_nested_too_deeply_is_a_manifest_error(tmp_path):
    (tmp_path / "manifest.json").write_bytes(b'{"schema": 1, "kernels": [], "x": "\xff\xfe"}')
    with pytest.raises(TilefishManifestError, match="cannot read the manifest"):
        Manifest.load(tmp_path)
    (tmp_path / "manifest.json").write_text("[" * 100000 + "]" * 100000)
    with pytest.raises(TilefishManifestError, match="nested too deeply|not valid JSON"):
        Manifest.load(tmp_path)


def test_a_malformed_entry_is_a_manifest_error_whatever_raised_inside(directory):
    spec, _ = specs()
    m = Manifest.load(directory)
    m.get("scale", "sm_90").entry["shared"] = "sixteen"  # a loaded manifest whose entry was edited in memory
    with pytest.raises(TilefishManifestError, match="the manifest entry is malformed: ValueError"):
        m.built_kernel(spec, "sm_90", ir_digest=SRC, ir_triton=TRITON)


# -- writing: no link is followed, a file appears whole -----------------------------------------------------------------------------------


def test_a_manifest_that_is_a_symlink_is_not_written_through(tmp_path):
    (spec,) = compile_specs(scale)
    outside = tmp_path / "outside.json"
    outside.write_text("precious")
    out = tmp_path / "out"
    out.mkdir()
    (out / "manifest.json").symlink_to(outside)
    with pytest.raises(TilefishBuildError, match="manifest.json' is a symbolic link; refusing to write through it"):
        write_manifest([fake_build(spec)], out / "manifest.json", cubins=True)
    assert outside.read_text() == "precious" and not list(out.glob("*.cubin"))  # and not a cubin either: nothing is written before the checks pass


def test_a_dangling_manifest_symlink_is_refused_too(tmp_path):
    (spec,) = compile_specs(scale)
    out = tmp_path / "out"
    out.mkdir()
    (out / "manifest.json").symlink_to(tmp_path / "nowhere" / "created.json")
    for merge in (False, True):
        with pytest.raises(TilefishBuildError, match="is a symbolic link"):
            write_manifest([fake_build(spec)], out / "manifest.json", cubins=True, merge=merge)
    assert not (tmp_path / "nowhere").exists()


def test_a_symlinked_cubin_stops_the_whole_write_before_any_file_is_written(directory, tmp_path):
    before = manifest_text(directory)
    elsewhere = tmp_path / "elsewhere.bin"
    elsewhere.write_bytes(b"precious")
    victim = directory / "scale.sm_90.cubin"
    victim.unlink()
    victim.symlink_to(elsewhere)
    built = [fake_build(s, cubin=cubin_like(code=b"\x0b" * 64)) for s in compile_specs(rowop)]
    built.append(fake_build(compile_specs(scale)[0], cubin=cubin_like(code=b"\x0c" * 64)))
    with pytest.raises(TilefishBuildError, match="scale.sm_90.cubin' is a symbolic link"):
        write_manifest(built, directory / "manifest.json", cubins=True)
    assert elsewhere.read_bytes() == b"precious" and manifest_text(directory) == before
    assert cubin_of(directory, "rowop.S32").read_bytes() == cubin_like()  # the rowop cubins that came first in the list were not rewritten


def test_a_file_is_written_whole_or_not_at_all(directory, monkeypatch):
    before = manifest_text(directory)
    (spec,) = compile_specs(scale)
    other = fake_build(spec, arch="sm_100")

    def fail(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        write_manifest([other], directory / "manifest.json", cubins=True, merge=True)
    monkeypatch.undo()
    assert manifest_text(directory) == before  # the old manifest is intact ...
    assert sorted(p.name for p in directory.iterdir()) == sorted(["manifest.json", "scale.sm_90.cubin", "rowop.S32.sm_90.cubin", "rowop.S64.sm_90.cubin"])  # ... and no temporary file is left


def test_written_files_are_world_readable_and_have_no_temporary_siblings(tmp_path):
    (spec,) = compile_specs(scale)
    write_manifest([fake_build(spec)], tmp_path / "manifest.json", cubins=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json", "scale.sm_90.cubin"]
    for p in tmp_path.iterdir():
        assert p.stat().st_mode & 0o777 == 0o644


def test_the_document_a_write_returns_is_the_one_whose_digest_a_load_finds(directory):
    from tilefish._build import write_manifest_doc

    (spec,) = compile_specs(scale)
    path, written = write_manifest_doc([fake_build(spec, arch="sm_100")], directory / "manifest.json", cubins=True, merge=True)
    assert path == directory / "manifest.json" and len(written["kernels"]) == 4  # the merged document, not only the new entry
    assert manifest_digest(written) == Manifest.load(directory).digest
