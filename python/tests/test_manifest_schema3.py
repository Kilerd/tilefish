"""Manifest schema 3: what it adds to schema 2, and what the Rust host (which reads schema 2) is told it cannot launch yet.

Schema 3 is schema 2 plus: ``kind`` on every kernel entry, ``abi.scratch_slots``, the ``library`` section and a digest of its own (tag ``v3``); for strided
views, ``strides`` and ``offset`` on a pointer of the ABI, ``stride_of`` on a scalar and the ``max_extent`` precondition; for device scalars, a pointer with
the shape ``[]``; for prebuilt binaries, an entry of kind ``binary`` and ``fixed`` arguments. ``docs/manifest-schema-3.md`` specifies it, field by field.
"""

import copy
import json

import pytest
from built_fakes import fake_build, rowop, scale
from schema2_fixtures import migrate_doc

import tilefish
from tilefish import Dim, Out, Strided, StrideOf, Tensor, TilefishManifestError, bf16, const, f32, i32
from tilefish._manifest import DIGEST_TAG, LEGACY_DIGEST_TAG, manifest_digest
from tilefish.aot import Manifest, compile_specs, write_manifest
from tilefish.hosts import rust

S = Dim("S")


@tilefish.extern(triton="kernels:head_copy", num_warps=1, grid=lambda S: (S, 8))
def head_copy(x: Strided[Tensor[S, 8, 64, bf16]], y: Out[Tensor[S, 8, 64, bf16]], x_s0: StrideOf("x", 0), x_s1: StrideOf("x", 1), D: const = 64): ...


@tilefish.extern(triton="kernels:masked", grid=lambda S: (S,))
def masked(x: Tensor[S, 128, f32], y: Out[Tensor[S, 128, f32]], n_tok: tilefish.DevScalar[i32], D: const = 128): ...


@tilefish.extern(triton="kernels:head_copy64", num_warps=1, grid=lambda S: (S, 8), index_bits=64)
def head_copy64(x: Strided[Tensor[S, 8, 64, bf16]], y: Out[Tensor[S, 8, 64, bf16]], x_s0: StrideOf("x", 0), x_s1: StrideOf("x", 1)): ...


def manifest_of(tmp_path, *ops):
    built = [fake_build(s) for op in ops for s in compile_specs(op)]
    write_manifest(built, tmp_path / "m" / "manifest.json", cubins=True)
    return Manifest.load(tmp_path / "m")


def test_a_triton_entry_is_schema_2_plus_a_kind_and_the_scratch_slots(tmp_path):
    m = manifest_of(tmp_path, scale, rowop)
    doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
    assert set(doc) == {"schema", "tilefish", "kernels", "library"} and doc["schema"] == 3 and doc["library"] == []
    for e in doc["kernels"]:
        assert e["kind"] == "triton" and e["abi"]["scratch_slots"] == 2 and len(e["abi"]["scratch"]) == 2
        assert {"num_warps", "num_stages", "options", "toolchain"} <= set(e) and "binary" not in e
        assert set(e["abi"]) == {"params", "scratch", "scratch_slots"}
    # a schema 2 document becomes a schema 3 one by exactly those three additions, and its digests change with them
    old = copy.deepcopy(doc)
    old["schema"] = 2
    del old["library"]
    for e in old["kernels"]:
        del e["kind"]
        del e["abi"]["scratch_slots"]
    assert migrate_doc(old) == doc
    assert manifest_digest(old) != m.digest


def test_the_two_digests_have_their_own_tags_and_schema_3_has_the_library_in_it(tmp_path):
    import hashlib

    doc = {"schema": 3, "kernels": [], "library": []}
    assert manifest_digest(doc) == "f66955caeb4e878dbe95483da2607d19ea4f734977e86282fff6f92f1a87fe6e"  # the vector of docs/manifest-schema-3.md
    assert manifest_digest({"schema": 2, "kernels": []}) == "a3593eed6b463f7408711b92645197eda2e7bc5f2830b138b40e33ceedc1cfc8"
    text = json.dumps({"schema": 3, "kernels": [], "library": []}, sort_keys=True, separators=(",", ":"))
    assert DIGEST_TAG == b"tilefish.manifest-digest.v3\0" and manifest_digest(doc) == hashlib.sha256(DIGEST_TAG + text.encode()).hexdigest()
    old = {"schema": 2, "kernels": []}
    text2 = json.dumps(old, sort_keys=True, separators=(",", ":"))
    assert LEGACY_DIGEST_TAG == b"tilefish.manifest-digest.v2\0" and manifest_digest(old) == hashlib.sha256(LEGACY_DIGEST_TAG + text2.encode()).hexdigest()
    assert manifest_digest({"schema": 4, "kernels": [], "library": []}) == manifest_digest({"schema": 4, "kernels": []})  # any other schema: the old form
    for bad in ({"schema": 3, "kernels": []}, {"schema": 3, "library": []}, {"kernels": [], "library": []}):
        with pytest.raises(TilefishManifestError):
            manifest_digest(bad)


def test_the_digest_of_the_entries_in_two_orders_is_the_same(tmp_path):
    m = manifest_of(tmp_path, scale, rowop)
    doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
    doc["kernels"].reverse()
    assert manifest_digest(doc) == m.digest


def test_a_strided_pointer_and_its_stride_scalars_are_in_the_manifest(tmp_path):
    manifest_of(tmp_path, head_copy)
    e = json.loads((tmp_path / "m" / "manifest.json").read_text())["kernels"][0]
    x, y, s0, s1 = e["abi"]["params"]
    assert x["strides"] == ["x_s0", "x_s1", 1] and x["offset"] == "x_offset" and "strides" not in y
    assert s0["stride_of"] == {"arg": "x", "axis": 0} and s0["dim"] == "x_s0" and s0["type"] == "i32"
    assert {"kind": "pointer_alignment", "arg": "x", "index": 0, "bytes": 2} in e["preconditions"]
    assert {"kind": "max_extent", "arg": "x", "index": 0, "value": 2**31 - 1} in e["preconditions"]
    assert e["dims"] == {"S": {}, "x_offset": {}, "x_s0": {}, "x_s1": {}}
    assert e["grid"] == "(S, 8)"


def test_a_device_scalar_is_a_pointer_with_no_axes_in_the_manifest(tmp_path):
    manifest_of(tmp_path, masked)
    e = json.loads((tmp_path / "m" / "manifest.json").read_text())["kernels"][0]
    n = e["abi"]["params"][2]
    assert (n["kind"], n["dtype"], n["shape"], n["access"], n["name"]) == ("pointer", "i32", [], "in", "n_tok")


def test_the_entry_must_match_the_compile_spec_in_the_new_fields_too(tmp_path):
    m = manifest_of(tmp_path, head_copy)
    (spec,) = compile_specs(head_copy)
    m.built_kernel(spec, "sm_90", ir_digest=None, ir_triton=None, pinned=True)
    for change, path in (
        (lambda e: e["abi"]["params"][0].update(strides=["x_s0", "x_s1", 2]), "abi.params[0].strides[2]"),
        (lambda e: e["abi"]["params"][0].update(offset="other"), "abi.params[0].offset"),
        (lambda e: e["abi"]["params"][2]["stride_of"].update(axis=1), "abi.params[2].stride_of.axis"),
        (lambda e: e["abi"]["params"][0].pop("strides"), "abi.params[0].strides"),
        (lambda e: e["abi"].update(scratch_slots=0), "scratch_slots"),
        (lambda e: e.update(kind="binary"), "kind"),
    ):
        doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
        change(doc["kernels"][0])
        (tmp_path / "m2").mkdir(exist_ok=True)
        (tmp_path / "m2" / "manifest.json").write_text(json.dumps(doc))
        (tmp_path / "m2" / doc["kernels"][0]["cubin"]).write_bytes((tmp_path / "m" / doc["kernels"][0]["cubin"]).read_bytes())
        with pytest.raises(TilefishManifestError):  # at load (the kind and the scratch slots) or when compared with the spec (the rest)
            Manifest.load(tmp_path / "m2").built_kernel(spec, "sm_90", pinned=True)


# -- what the Rust host launches (test_rust_host_schema3.py has the details) ----------------------------------------------------------------


@pytest.mark.parametrize(
    "op, snippets",
    [
        (head_copy, ["x: TensorRef<'a, Bf16>", "a strided view (strides `[x_s0, x_s1, 1]`)", "`x_s0` is the stride of axis 0 of `x`, in elements"]),
        (masked, ["n_tok: TensorRef<'a, I32>", "a device scalar (one element: the shape `[]` or `[1]`)", '&["S", "128"], &["S", "128"], &[]']),
        (head_copy64, ["Runtime scalars of the binary, derived by `launch` from the tensors", "`x_s0` is the stride of axis 0 of `x`, in elements (i64)"]),
    ],
)
def test_the_rust_emitter_writes_launchers_for_strided_views_and_device_scalars(tmp_path, op, snippets):
    m = manifest_of(tmp_path, op)
    source = rust.render(m.check_pin(m.digest), ops=[op])
    for snippet in snippets:
        assert snippet in source, snippet
    assert "        x_s0:" not in source  # a stride is not an argument of a launch (a parameter of `launch` is indented by eight spaces)


def test_the_rust_emitter_still_writes_what_schema_2_could_say(tmp_path):
    ok = manifest_of(tmp_path / "ok", scale, rowop)
    source = rust.render(ok.check_pin(ok.digest), ops=[scale, rowop])
    assert f'pub const MANIFEST_DIGEST: &str = "{ok.digest}";' in source and "pub struct Scale " in source


def test_a_manifest_with_a_wrong_number_of_scratch_slots_is_not_launched_by_the_rust_emitter(tmp_path):
    m = manifest_of(tmp_path, scale)
    doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
    doc["kernels"][0]["abi"]["scratch_slots"] = 1
    (tmp_path / "m" / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="a triton kernel has 2 scratch slots, but abi.scratch_slots is 1"):
        Manifest.load(tmp_path / "m")
    del m
