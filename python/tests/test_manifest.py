"""Loading a manifest directory back (``tilefish.Manifest``): what ``replay`` runs, and what it refuses.

A manifest is written by ``write_manifest(built, path, cubins=True)`` and read by ``Manifest.load``, which checks
every cubin against the digests the manifest records; ``Manifest.built_kernel(spec, arch)`` then checks that the
entry is what a compile spec asks for. No triton and no GPU: the cubins are small ELF files made by hand. On a GPU,
``test_gpu_byte_identity.py`` and ``test_gpu_build.py`` replay real ones.
"""

import dataclasses
import hashlib
import json

import pytest
from built_fakes import fake_build, rowop, scale
from elf_builder import S, SHT_PROGBITS, build_elf, cubin_like

import tilefish
from tilefish import Manifest, TilefishManifestError, TilefishRuntimeError
from tilefish._elf import code_digest, parse_elf
from tilefish.aot import compile_specs, write_manifest
from tilefish.runtime import _load_manifest


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
    assert "3 binaries" in repr(m)


def test_load_is_the_same_through_tilefish_and_aot(directory):
    assert tilefish.Manifest is tilefish.aot.Manifest is Manifest


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d.update(schema=2), "not a tilefish manifest of schema 1"),
        (lambda d: d.update(kernels="nope"), "not a tilefish manifest"),
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


# -- the binary for a spec ---------------------------------------------------------------------------------------------------------


def specs():
    return compile_specs(scale)[0], compile_specs(rowop)


def test_built_kernel_is_the_binary_for_a_spec(directory):
    spec, (s32, s64) = specs()
    m = Manifest.load(directory)
    b = m.built_kernel(spec, "sm_90")
    assert b.spec is spec and b.arch == "sm_90" and b.symbol == "tf_k" and b.cubin == cubin_like()
    assert b.ptx == "" and b.sha256 == hashlib.sha256(b.cubin).hexdigest() and b.sha256_code == code_digest(b.cubin)
    assert b.num_warps == 4 and b.shared == 0 and b.options == {"num_warps": 4, "num_stages": 3}
    assert b.launch.replay_blockers() == [] and b.triton_version == "3.5.1" and "V12.9.86" in b.ptxas_version
    assert b.abi() == fake_build(spec).abi()
    assert m.built_kernel(s64, "sm_90").stem == "rowop.S64"


def test_built_kernel_needs_the_architecture(directory):
    spec, (s32, _) = specs()
    m = Manifest.load(directory)
    with pytest.raises(TilefishManifestError, match=r"no binary for 'scale' on sm_100 \(it has sm_90\)"):
        m.built_kernel(spec, "sm_100")
    with pytest.raises(TilefishManifestError, match=r"no binary for 'other' on sm_90 \(it has \['rowop.S32', 'rowop.S64', 'scale'\]\)"):
        renamed = dataclasses.replace(spec, stem="other")
        renamed._mark_verified()
        m.built_kernel(renamed, "sm_90")


@pytest.mark.parametrize(
    "change, difference",
    [
        (lambda e: e.update(grid="(n,)"), "grid"),
        (lambda e: e.update(block=[64, 1, 1]), "block"),
        (lambda e: e.update(num_warps=8), "num_warps"),
        (lambda e: e["constexprs"][1].update(value=512), "constexprs[1].value"),
        (lambda e: e["abi"]["params"][0].update(dtype="f32"), "abi.params[0].dtype"),
        (lambda e: e["abi"]["scratch"].pop(), "abi.scratch"),
        (lambda e: e["preconditions"].pop(), "preconditions"),
        (lambda e: e.update(declaration_digest="0" * 64), "declaration_digest"),
        (lambda e: e["options"].update(enable_fp_fusion=False), "options.enable_fp_fusion"),
        (lambda e: e["options"].update(num_stages=2), "options.num_stages"),
        (lambda e: e.update(buckets={"S": 32}), "buckets.S"),
    ],
)
def test_built_kernel_refuses_an_entry_that_is_not_what_the_spec_asks_for(directory, change, difference):
    edit_manifest(directory, lambda d: change(d["kernels"][0]))
    spec, _ = specs()
    m = Manifest.load(directory)  # still a perfectly good manifest: the cubins are what was built
    with pytest.raises(TilefishManifestError, match=r"not what the compile spec asks for: first difference at " + difference.replace("[", r"\[").replace("]", r"\]")):
        m.built_kernel(spec, "sm_90")


def test_built_kernel_refuses_a_binary_of_another_declaration(directory):
    """The same stem, another extern (here with a different constexpr default): the declaration digest and the constants differ."""
    import tilefish as tf
    from tilefish import Dim, Out, Tensor, bf16, const, f32

    n = Dim("n", divisible_by=16)

    @tf.extern(triton="my_kernels:scale", num_warps=4, grid=lambda n: (tf.cdiv(n, 256),), name="scale")
    def scale_with_another_alpha(
        x: Tensor[n, bf16], y: Out[Tensor[n, bf16]], n_elem: n, *, ALPHA: const[f32] = 2.0, BLOCK: const = 256
    ): ...

    (spec,) = compile_specs(scale_with_another_alpha)
    assert spec.stem == "scale"
    with pytest.raises(TilefishManifestError, match="not what the compile spec asks for"):
        Manifest.load(directory).built_kernel(spec, "sm_90")


def test_launch_attributes_that_replay_cannot_set_up_are_kept_for_the_driver_check(directory):
    edit_manifest(directory, lambda d: d["kernels"][0]["launch"].update(global_scratch_size=64, launch_pdl=True))
    spec, _ = specs()
    b = Manifest.load(directory).built_kernel(spec, "sm_90")
    assert b.launch.replay_blockers() == ["global_scratch_size=64", "launch_pdl=True"]  # refused when it is loaded into the driver
    edit_manifest(directory, lambda d: d["kernels"][0].update(launch={"global_scratch_size": 0}))
    with pytest.raises(TilefishManifestError, match="launch attributes are malformed"):
        Manifest.load(directory).built_kernel(spec, "sm_90")


# -- run(): the manifest argument --------------------------------------------------------------------------------------------------------


def test_replay_needs_a_manifest_and_only_replay_takes_one(directory):
    with pytest.raises(TilefishRuntimeError, match="replay runs prebuilt cubins from a manifest directory: pass manifest="):
        tilefish.run(scale, mode="replay")
    for mode in ("jit", "reference"):
        with pytest.raises(TilefishRuntimeError, match="manifest= is for mode='replay'"):
            tilefish.run(scale, mode=mode, manifest=directory)
    with pytest.raises(TilefishRuntimeError, match="manifest= is for mode='replay'"):
        scale.run(manifest=directory)  # jit is the default mode


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
    m = Manifest.load(tmp_path)
    assert [m.built_kernel(s, "sm_90").num_warps for s in specs_] == [1, 16]
    strict = rows.lower(fp_fusion=False).specs()
    with pytest.raises(TilefishManifestError, match="first difference at options.enable_fp_fusion"):
        m.built_kernel(strict[0], "sm_90")


def test_a_minimal_cubin_with_its_code_section_loads(tmp_path):
    only_debug = build_elf([S(".text.tf_k", SHT_PROGBITS, data=b"\1\2\3\4"), S(".debug_line", data=b"x")])
    (spec,) = compile_specs(scale)
    write_manifest([fake_build(spec, cubin=only_debug)], tmp_path / "manifest.json", cubins=True)
    assert Manifest.load(tmp_path).get("scale", "sm_90").sha256_code == code_digest(only_debug)
