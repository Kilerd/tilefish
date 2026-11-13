"""The plan document (``plan.json``, ``tilefish.plan.v1``): what is written, how it is digested, and everything a reader refuses (``docs/plan-schema-1.md``).

These tests export the plan of a transformer block into a real manifest directory with fake builds (``plan_fakes``: no GPU), then read it back as a replay does:
the digest, the pin, the manifest it names, every rule of the format, and that the document is the plan of the graph in this process. A Rust reader implements the
same rules; the tests are their examples.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import copy
import hashlib
import json

import plan_fakes
import plan_model as pm
import pytest

import tilefish
from tilefish import Dim, Manifest, Tensor, TilefishManifestError, TilefishPlanError, TilefishRuntimeError, f32
from tilefish import _plan
from tilefish._build import write_manifest_doc

WRONG = "0" * 64


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("plan")
    return directory, plan_fakes.export(tilefish.plan(pm.block), directory)


@pytest.fixture
def directory(exported, tmp_path):
    """A private copy of the exported directory, for the tests that damage it."""
    source, _ = exported
    target = tmp_path / "copy"
    target.mkdir()
    for f in source.iterdir():
        (target / f.name).write_bytes(f.read_bytes())
    return target


def doc_of(directory):
    return json.loads((directory / "plan.json").read_text())


def write_doc(directory, doc, *, redigest=True):
    if redigest:
        doc["plan_digest"] = _plan.plan_digest(doc)
    (directory / "plan.json").write_text(json.dumps(doc, indent=2) + "\n")
    return doc["plan_digest"]


def load(directory, **kw):
    return _plan.load_plan(directory, pm.block, **kw)


# -- what is written -----------------------------------------------------------------------------------------------------------------------


def test_export_writes_the_manifest_the_plan_and_the_plan_is_the_one_made(exported):
    directory, p = exported
    assert sorted(f.name for f in directory.iterdir() if f.suffix == ".json") == ["manifest.json", "plan.json"]
    made = tilefish.plan(pm.block)
    doc = doc_of(directory)
    assert doc["arch"] == "sm_103" and doc["schema"] == "tilefish.plan.v1" and doc["graph"] == "block"
    assert doc["manifest_digest"] == Manifest.load(directory).digest == p.manifest_digest
    # everything but the architecture, the manifest and the digests is the plan that was made without them
    for key in ("buffers", "buckets", "dims", "outputs", "arena_bytes", "reuse", "arena_alignment"):
        assert doc[key] == made.doc[key], key
    assert p.digest == doc["plan_digest"] and p.directory == directory and p.manifest.digest == doc["manifest_digest"]
    assert list(doc) == _plan.TOP_KEYS or sorted(doc) == sorted(_plan.TOP_KEYS)


def test_the_digest_is_sha256_of_a_tag_and_the_canonical_json_without_the_digest_and_the_version(exported):
    directory, p = exported
    doc = doc_of(directory)
    body = {k: v for k, v in doc.items() if k not in ("plan_digest", "tilefish")}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    assert hashlib.sha256(b"tilefish.plan-digest.v1\0" + text.encode()).hexdigest() == doc["plan_digest"] == p.digest
    assert _plan.DIGEST_TAG == b"tilefish.plan-digest.v1\0"
    # the plan includes the digest of the manifest: another manifest is another plan digest
    other = copy.deepcopy(doc)
    other["manifest_digest"] = WRONG
    assert _plan.plan_digest(other) != p.digest


def test_the_digest_of_a_tiny_document_is_a_test_vector():
    """For an implementer in another language: the digest of this document is this, whatever it is made of (the version and the digest itself are left out)."""
    doc = {"schema": "tilefish.plan.v1", "tilefish": "x", "graph": "g", "plan_digest": "ignored"}
    assert _plan.plan_digest(doc) == "f59e1096fcfb737bcb745fc0a11258a25a1e2eb2fd3039b8958c777d4dadc71e"
    assert _plan.plan_digest({}) == "345611870fcf5832cd7533b55c9bcb061b78c17c4f53635c9bcf8f7cffca6893"  # sha256(tag + "{}")


def test_a_pinned_load_returns_the_plan_and_the_verified_manifest(exported):
    directory, p = exported
    loaded = load(directory, pin=p.digest)
    assert loaded.digest == p.digest and loaded.arch == "sm_103" and loaded.manifest.digest == p.manifest_digest and loaded.manifest.pinned
    assert loaded.doc == p.doc
    trusted = load(directory, trust_directory=True)
    assert trusted.digest == p.digest and not trusted.manifest.pinned


def test_a_plan_needs_exactly_one_trust_anchor(exported):
    directory, p = exported
    with pytest.raises(TilefishRuntimeError, match="needs a trust anchor"):
        load(directory)
    with pytest.raises(TilefishRuntimeError, match="not both"):
        load(directory, pin=p.digest, trust_directory=True)


# -- what is refused: the pin, the digest, the manifest ---------------------------------------------------------------------------------------


def test_a_wrong_pin_is_refused_before_anything_else(exported):
    directory, p = exported
    with pytest.raises(TilefishManifestError, match="has digest .*but the pin is"):
        load(directory, pin=WRONG)
    with pytest.raises(TilefishManifestError, match="64 lowercase hex digits"):
        load(directory, pin="abc")
    with pytest.raises(TilefishManifestError, match="64 lowercase hex digits"):
        load(directory, pin=p.digest.upper())
    # the digest of the manifest is not the digest of the plan
    with pytest.raises(TilefishManifestError, match="but the pin is"):
        load(directory, pin=p.manifest_digest)


def test_an_edit_of_plan_json_is_refused_by_the_pin_and_by_the_digest_it_carries(directory, exported):
    _, p = exported
    text = (directory / "plan.json").read_text()
    edited = text.replace('"offset": 256', '"offset": 512', 1)
    assert edited != text
    (directory / "plan.json").write_text(edited)
    with pytest.raises(TilefishManifestError, match="but the pin is"):
        load(directory, pin=p.digest)
    with pytest.raises(TilefishManifestError, match="was edited: its plan_digest says"):
        load(directory, trust_directory=True)


def test_an_edit_that_also_rewrites_the_digest_is_refused_by_the_pin_but_not_by_trusting_the_directory(directory, exported):
    _, p = exported
    doc = doc_of(directory)
    doc["reuse"] = True  # a plan with other offsets that is not the one of the pin: overlapping regions with reuse=false are refused, true needs liveness
    digest = write_doc(directory, doc)
    with pytest.raises(TilefishManifestError, match="but the pin is"):
        load(directory, pin=p.digest)
    assert digest != p.digest
    # trusting the directory says whatever it holds is fine: a document that is wrong is still refused by the rules of the format and by the graph
    with pytest.raises(TilefishManifestError):
        load(directory, trust_directory=True)  # the regions of this plan do not overlap, so it is a valid one: but it is not the plan of this graph with reuse=true


def test_a_plan_built_against_another_manifest_digest_is_refused(directory):
    doc = doc_of(directory)
    doc["manifest_digest"] = WRONG
    digest = write_doc(directory, doc)  # self-consistent, and even pinned: the plan names a manifest that is not the one next to it
    with pytest.raises(TilefishManifestError, match=r"built against the manifest with digest 0{64}, but the manifest in .* has digest"):
        load(directory, pin=digest)


def test_a_manifest_that_changed_after_the_export_makes_the_plan_stale(directory, exported):
    _, p = exported
    doc = json.loads((directory / "manifest.json").read_text())
    doc["tilefish"] = "changed but not digested"
    (directory / "manifest.json").write_text(json.dumps(doc))
    assert load(directory, pin=p.digest).digest == p.digest  # the version string is not part of the manifest's digest
    doc["kernels"][0]["shared"] += 16
    (directory / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="built against the manifest with digest"):
        load(directory, pin=p.digest)


def test_exporting_another_graph_next_to_it_changes_the_manifest_and_refuses_the_old_plan(directory, exported):
    _, p = exported
    spec = tilefish.plan(pm.block).callees["gelu"].by_stem["gelu"]
    extra = plan_fakes.fake_build(spec, arch="sm_103")
    other_stem = tilefish.plan(pm.block).callees["add"].by_stem["add"]
    write_manifest_doc([plan_fakes.fake_build(other_stem, arch="sm_103", shared=32)], directory / "manifest.json", cubins=True, merge=True, overwrite=True)
    with pytest.raises(TilefishManifestError, match="built against the manifest with digest"):
        load(directory, pin=p.digest)
    del extra


def test_a_plan_of_another_graph_or_a_graph_that_changed_is_refused(directory, exported):
    _, p = exported

    @tilefish.graph(name="block")
    def block(x: Tensor[pm.M, pm.D, tilefish.bf16], w: Tensor[pm.D, tilefish.bf16]) -> Tensor[pm.M, pm.D, tilefish.bf16]:
        return pm.rmsnorm(x, w)

    with pytest.raises(TilefishManifestError, match=r"is not the plan of graph 'block' as this process holds it: first difference at"):
        _plan.load_plan(directory, block, pin=p.digest)

    @tilefish.graph(name="other")
    def other(x: Tensor[pm.M, pm.D, tilefish.bf16], w: Tensor[pm.D, tilefish.bf16]) -> Tensor[pm.M, pm.D, tilefish.bf16]:
        return pm.rmsnorm(x, w)

    with pytest.raises(TilefishManifestError, match="is the plan of graph 'block', not of 'other'"):
        _plan.load_plan(directory, other, pin=p.digest)


def test_a_file_that_is_not_a_plan_is_refused(directory):
    (directory / "plan.json").write_text("{not json")
    with pytest.raises(TilefishManifestError, match="not valid JSON"):
        load(directory, trust_directory=True)
    (directory / "plan.json").write_text('{"schema": "tilefish.plan.v2"}')
    with pytest.raises(TilefishManifestError, match="is not a tilefish plan of schema"):
        load(directory, trust_directory=True)
    (directory / "plan.json").write_text('{"schema": "tilefish.plan.v1", "schema": "tilefish.plan.v1"}')
    with pytest.raises(TilefishManifestError, match="the key 'schema' twice"):
        load(directory, trust_directory=True)
    (directory / "plan.json").write_text('{"schema": "tilefish.plan.v1", "x": NaN}')
    with pytest.raises(TilefishManifestError, match="not strict JSON"):
        load(directory, trust_directory=True)
    (directory / "plan.json").unlink()
    with pytest.raises(TilefishManifestError, match="cannot read the plan"):
        load(directory, trust_directory=True)


def test_a_symbolic_link_where_plan_json_goes_is_refused(tmp_path):
    p = tilefish.plan(pm.block)
    target = tmp_path / "elsewhere.json"
    target.write_text("{}")
    d = tmp_path / "d"
    d.mkdir()
    (d / "plan.json").symlink_to(target)
    with pytest.raises(Exception, match="symbolic link"):
        plan_fakes.export(p, d, overwrite=True)
    assert target.read_text() == "{}"


def test_a_directory_holds_one_plan_and_another_graph_needs_overwrite(exported, tmp_path):
    directory, _ = exported
    d = tmp_path / "d"
    d.mkdir()
    for f in directory.iterdir():
        (d / f.name).write_bytes(f.read_bytes())

    @tilefish.graph(name="smaller")
    def smaller(x: Tensor[pm.M, pm.D, tilefish.bf16], w: Tensor[pm.D, tilefish.bf16]) -> Tensor[pm.M, pm.D, tilefish.bf16]:
        return pm.rmsnorm(x, w)

    other = tilefish.plan(smaller)
    with pytest.raises(TilefishManifestError, match="is the plan of graph 'block'"):
        plan_fakes.export(other, d)
    again = plan_fakes.export(tilefish.plan(pm.block), d)  # the same graph again is a re-export
    assert again.digest == doc_of(d)["plan_digest"]
    replaced = plan_fakes.export(other, d, overwrite=True)
    assert doc_of(d)["graph"] == "smaller" and replaced.digest == doc_of(d)["plan_digest"]


# -- the rules of the format -----------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def good():
    return tilefish.plan(pm.block).with_manifest("sm_103", "a" * 64).doc


@pytest.fixture(scope="module")
def good_reuse():
    return tilefish.plan(pm.block, reuse=True).with_manifest("sm_103", "a" * 64).doc


def buffer(doc, name):
    return next(b for b in doc["buffers"] if b["name"] == name)


def test_a_valid_document_passes_every_rule(good, good_reuse):
    _plan.validate_doc(good)
    _plan.validate_doc(good_reuse)


def mutations():
    """(name, a function that breaks a copy of the document, a fragment of the refusal)."""

    def setter(path, value):
        def apply(doc):
            node = doc
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value

        return apply

    def deleter(path):
        def apply(doc):
            node = doc
            for key in path[:-1]:
                node = node[key]
            del node[path[-1]]

        return apply

    out = [
        ("schema", setter(["schema"], "tilefish.plan.v2"), "schema must be"),
        ("unknown key", setter(["extra"], 1), "unknown"),
        ("missing key", deleter(["arena_bytes"]), "missing"),
        ("version type", setter(["tilefish"], 1), "tilefish must be a string"),
        ("graph", setter(["graph"], "a b"), "graph must be an identifier"),
        ("arch", setter(["arch"], "sm_9"), "arch must look like"),
        ("arch null", setter(["arch"], None), "arch must look like"),
        ("manifest digest", setter(["manifest_digest"], "A" * 64), "manifest_digest must be 64 lowercase hex"),
        ("manifest digest null", setter(["manifest_digest"], None), "manifest_digest must be 64 lowercase hex"),
        ("reuse", setter(["reuse"], 1), "reuse must be true or false"),
        ("alignment", setter(["arena_alignment"], 128), "arena_alignment must be 256"),
        ("arena bytes", setter(["arena_bytes"], 256), "arena_bytes must be the end of the last region"),
        ("arena bytes float", setter(["arena_bytes"], 9634048.0), "arena_bytes must be"),
        ("dims not sorted", setter(["dims", "M"], [64, 32, 128]), "strictly increasing"),
        ("dims duplicate", setter(["dims", "M"], [32, 32, 64]), "strictly increasing"),
        ("dims zero", setter(["dims", "M"], [0, 32]), "strictly increasing"),
        ("dims empty", setter(["dims", "M"], []), "non-empty"),
        ("dims name", setter(["dims"], {"1M": [32]}), "not an identifier"),
        ("buffers empty", setter(["buffers"], []), "non-empty list"),
        ("buffer key", setter(["buffers", 0, "extra"], 1), "unknown"),
        ("buffer name", setter(["buffers", 1, "name"], "x"), "unique identifier"),
        ("buffer role", setter(["buffers", 0, "role"], "weights"), "role must be one of"),
        ("buffer dtype", setter(["buffers", 0, "dtype"], "f64"), "dtype must be one of"),
        ("buffer shape", setter(["buffers", 0, "shape"], [0, 256]), "shape must be a list of positive ints"),
        ("buffer shape expr", setter(["buffers", 0, "shape"], ["M*M", 256]), "no value|malformed|shape"),
        ("buffer bytes", setter(["buffers", 0, "bytes"], 1), "bytes must be the largest size"),
        ("external offset", setter(["buffers", 0, "offset"], 0), "must be null"),
        ("arena offset null", setter(["buffers", 9, "offset"], None), "lives in the arena"),
        ("arena offset unaligned", setter(["buffers", 9, "offset"], 100), "multiple of 256"),
        ("arena offset negative", setter(["buffers", 9, "offset"], -256), "non-negative multiple of 256"),
        ("devscalar shape", setter(["buffers", 1, "shape"], [1]), "devscalar"),
        ("workspace dtype", setter(["buffers", -1, "dtype"], "bf16"), "workspace"),
        ("outputs missing", setter(["outputs"], []), "outputs must list the buffers of role output"),
        ("outputs not a buffer", setter(["outputs"], ["x"]), "outputs must list the buffers of role output"),
        ("outputs twice", setter(["outputs"], ["out0", "out0"]), "outputs must list the buffers of role output"),
        ("buckets short", setter(["buckets"], []), "one entry per combination"),
        ("bucket values", setter(["buckets", 0, "values"], {"M": 33}), "follow the combinations"),
        ("bucket order", lambda d: d["buckets"].reverse(), "follow the combinations"),
        ("bucket key", setter(["buckets", 0, "extra"], 1), "unknown"),
        ("step key", setter(["buckets", 0, "steps", 0, "extra"], 1), "unknown"),
        ("step entry", setter(["buckets", 0, "steps", 0, "entry"], "a b"), "manifest stem"),
        ("step kind", setter(["buckets", 0, "steps", 0, "kind"], "launch"), "kind must be one of"),
        ("step kind swapped", setter(["buckets", 0, "steps", 0, "kind"], "library"), "exactly the steps of kind 'library' have a workspace"),
        ("workspace on a kernel", setter(["buckets", 0, "steps", 0, "workspace"], "workspace"), "exactly the steps of kind 'library' have a workspace"),
        ("workspace missing", deleter(["buckets", 0, "steps", 1, "workspace"]), "exactly the steps of kind 'library' have a workspace"),
        ("workspace is not one", setter(["buckets", 0, "steps", 1, "workspace"], "x"), "must name the buffer of role workspace"),
        ("args empty", setter(["buckets", 0, "steps", 0, "args"], []), "non-empty list"),
        ("arg key", setter(["buckets", 0, "steps", 0, "args", 0, "extra"], 1), "unknown"),
        ("arg binding", setter(["buckets", 0, "steps", 0, "args", 0, "binding"], "nothing"), "is not a buffer"),
        ("arg names twice", setter(["buckets", 0, "steps", 0, "args", 1, "name"], "x"), "argument names must be distinct"),
        ("arg is the workspace", setter(["buckets", 0, "steps", 0, "args", 0, "binding"], "workspace"), "the workspace is not an argument"),
        ("view key", setter(["buckets", 0, "steps", 2, "args", 0, "view", "extra"], 1), "unknown"),
        ("view shape", setter(["buckets", 0, "steps", 2, "args", 0, "view", "shape"], [0, 4, 64]), "view must be"),
        ("view strides length", setter(["buckets", 0, "steps", 2, "args", 0, "view", "strides"], [768, 64]), "view must be"),
        ("view negative stride", setter(["buckets", 0, "steps", 2, "args", 0, "view", "strides"], [768, -64, 1]), "view must be"),
        ("view offset", setter(["buckets", 0, "steps", 2, "args", 0, "view", "offset"], -1), "view must be"),
        ("view past the buffer", setter(["buckets", 0, "steps", 2, "args", 0, "view", "offset"], 10**6), "reaches past the end"),
        ("view of a devscalar", setter(["buckets", 0, "steps", 2, "args", 4, "view"], {"shape": [1], "strides": [1], "offset": 0}), "bound whole"),
        ("kinds differ across buckets", setter(["buckets", 1, "steps", 0, "kind"], "library"), "exactly the steps of kind 'library' have a workspace|other kinds"),
        ("bindings differ across buckets", setter(["buckets", 1, "steps", 0, "args", 0, "binding"], "w_ln1"), "other kinds, workspaces, argument names or bindings"),
        ("step counts differ", lambda d: d["buckets"][1]["steps"].pop(), "same number of steps"),
        ("overlap without reuse", setter(["buffers", 10, "offset"], 0), "overlap, which only reuse=true allows"),
    ]
    return out


@pytest.mark.parametrize("name, mutate, fragment", mutations(), ids=[m[0] for m in mutations()])
def test_every_rule_of_the_format_is_enforced(good, name, mutate, fragment):
    doc = copy.deepcopy(good)
    mutate(doc)
    with pytest.raises(TilefishManifestError, match=fragment):
        _plan.validate_doc(doc)


def test_regions_may_overlap_only_with_reuse_and_only_where_no_step_uses_both(good_reuse):
    doc = copy.deepcopy(good_reuse)
    _plan.validate_doc(doc)
    names = {b["name"]: b for b in doc["buffers"] if b["role"] == "intermediate"}
    # t1 (the fused qkv) lives from step 1 to step 2 and t2 (the attention output) from step 2 to step 3: they are both used by step 2
    t1, t2 = names["t1"], names["t2"]
    assert t1["offset"] != t2["offset"]
    t2["offset"] = t1["offset"]
    with pytest.raises(TilefishManifestError, match="no step uses together"):
        _plan.validate_doc(doc)
    # a device scalar and the workspace are never shared, even where reuse is on
    doc = copy.deepcopy(good_reuse)
    buffer(doc, "t0")["offset"] = buffer(doc, "n")["offset"]
    with pytest.raises(TilefishManifestError, match="no step uses together|overlap"):
        _plan.validate_doc(doc)


def test_there_is_a_workspace_exactly_when_a_step_launches_a_gemm():
    doc = tilefish.plan(pm.accumulate).with_manifest("sm_103", "a" * 64).doc  # no GEMM, no workspace
    _plan.validate_doc(doc)
    doc["buffers"].append({"name": "workspace", "role": "workspace", "dtype": "u8", "shape": [256], "offset": 0, "bytes": 256})
    doc["arena_bytes"] = 16384
    with pytest.raises(TilefishManifestError, match="a buffer of role workspace exactly when some step has kind library"):
        _plan.validate_doc(doc)


def leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, path + (i,))
    else:
        yield path


@pytest.mark.parametrize("seed", range(40))
def test_a_damaged_document_is_refused_with_a_manifest_error_or_still_valid_never_another_exception(good, good_reuse, exported, seed):
    """Fuzz: change, delete, retype or duplicate one random part of a valid document, 25 times per seed; a reader either accepts it or raises TilefishManifestError."""
    import random

    rng = random.Random(seed)
    _, p = exported
    manifest = Manifest.load(exported[0])
    base = (good, good_reuse)[seed % 2]
    junk = [None, True, False, 0, -1, 1, 2**70, 1.5, "", "x", "M", "M*M", "3*M", "(((", [], {}, [1], {"a": 1}, "sm_103", "0" * 64]
    refused = accepted = 0
    for _ in range(25):
        doc = copy.deepcopy(base)
        path = rng.choice(list(leaves(doc)))
        node = doc
        for key in path[:-1]:
            node = node[key]
        how = rng.choice(["set", "set", "delete", "dup"])
        if how == "set":
            node[path[-1]] = rng.choice(junk)
        elif how == "delete":
            del node[path[-1]]
        elif isinstance(node, list):
            node.insert(rng.randrange(len(node) + 1), copy.deepcopy(node[path[-1]]))
        else:
            node[path[-1] + "_"] = node[path[-1]]
        try:
            _plan.validate_doc(doc)
            doc["manifest_digest"] = manifest.digest  # the manifest check is next: it needs the digest to be the manifest's
            _plan.check_against_manifest(doc, manifest)
            accepted += 1
        except TilefishManifestError:
            refused += 1
    assert refused + accepted == 25 and refused > 10


def test_a_document_cannot_make_a_reader_do_work_out_of_proportion_to_its_size(good):
    import time

    doc = copy.deepcopy(good)
    doc["dims"] = {f"D{i}": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] for i in range(30)}  # 10**30 combinations: counted, not listed
    start = time.perf_counter()
    with pytest.raises(TilefishManifestError, match="more than the 4096 a plan may have"):
        _plan.validate_doc(doc)
    assert time.perf_counter() - start < 1
    doc = copy.deepcopy(good)
    doc["buffers"] = doc["buffers"] * (_plan.MAX_BUFFERS // len(doc["buffers"]) + 1)
    with pytest.raises(TilefishManifestError, match="at most 32768 buffers"):
        _plan.validate_doc(doc)
    doc = copy.deepcopy(good)
    doc["buckets"][0]["steps"][0]["args"] = doc["buckets"][0]["steps"][0]["args"] * 100
    with pytest.raises(TilefishManifestError, match="at most 256 arguments"):
        _plan.validate_doc(doc)
    doc = copy.deepcopy(good)
    doc["dims"]["M"] = list(range(1, 100001))
    with pytest.raises(TilefishManifestError) as e:
        _plan.validate_doc(doc)
    assert len(str(e.value)) < 1200  # a refusal quotes what it refuses, cut

    @tilefish.kernel
    def two_dims(x: Tensor[Dim("R"), Dim("C"), f32]) -> Tensor[Dim("R"), Dim("C"), f32]:
        return x * 2.0

    R, C = Dim("R"), Dim("C")

    @tilefish.graph
    def wide(a: Tensor[R, C, f32]) -> Tensor[R, C, f32]:
        return two_dims(a)

    with pytest.raises(TilefishPlanError, match="10000 bucket combinations are planned, more than the 4096 a plan may have"):
        tilefish.plan(wide, buckets={"R": list(range(1, 101)), "C": list(range(1, 101))})
    with pytest.raises(TilefishPlanError) as e:
        tilefish.plan(pm.block, buckets={"M": list(range(1, 4098))})
    assert len(str(e.value)) < 600 and "in all" in str(e.value)


def test_a_deeply_nested_or_huge_document_is_a_manifest_error(tmp_path):
    (tmp_path / "plan.json").write_text('{"schema": "tilefish.plan.v1", "x": ' + "[" * 990 + "]" * 990 + "}")
    with pytest.raises(TilefishManifestError):
        _plan.read_plan(tmp_path)  # json parses it, the encoder cannot walk it: a manifest error, not a RecursionError
    from tilefish._manifest import manifest_digest

    # the digest of a deep document is a manifest error or a digest, never a RecursionError (the encoder of schema 3 walks it without recursing)
    try:
        d = manifest_digest({"schema": 3, "kernels": [{"stem": "a", "arch": "b", "x": __import__("functools").reduce(lambda a, _: [a], range(1500), [])}], "library": []})
    except TilefishManifestError:
        pass
    else:
        assert isinstance(d, str) and len(d) == 64 and int(d, 16) >= 0
    # an integer a double cannot hold (every language reads a plan number as one) is not strict JSON, wherever it is
    text = json.dumps(tilefish.plan(pm.block).with_manifest("sm_103", "a" * 64).doc)
    (tmp_path / "plan.json").write_text(text.replace('"arena_bytes": 9634048', '"arena_bytes": ' + str(256 * 10**400)))
    with pytest.raises(TilefishManifestError, match="does not fit a double|not valid JSON"):
        _plan.read_plan(tmp_path)
    (tmp_path / "plan.json").write_text(text.replace('"arena_bytes": 9634048', '"arena_bytes": ' + str(2**53 + 2)))
    with pytest.raises(TilefishManifestError, match="does not fit a double"):
        _plan.read_plan(tmp_path)


def test_validating_a_plan_with_thousands_of_buffers_that_share_regions_is_fast():
    import time

    doc = tilefish.plan(pm.deep_block(150), reuse=True).with_manifest("sm_103", "a" * 64).doc
    assert len(doc["buffers"]) > 1500
    start = time.perf_counter()
    _plan.validate_doc(doc)
    assert time.perf_counter() - start < 2  # a sweep: it was quadratic in the buffers that share one region


def test_the_overlap_check_is_a_sweep_that_finds_every_overlapping_pair(good_reuse):
    """Many buffers sharing one region is what reuse makes; a pair in the middle of them that does not have disjoint live ranges is still found."""
    doc = copy.deepcopy(good_reuse)
    names = [b["name"] for b in doc["buffers"] if b["role"] == "intermediate"]
    shared = [n for n in names if sum(1 for m in names if buffer(doc, m)["offset"] == buffer(doc, n)["offset"]) > 1]
    assert len(shared) >= 3
    _plan.validate_doc(doc)
    victim = next(n for n in names if buffer(doc, n)["offset"] not in {buffer(doc, s)["offset"] for s in shared})
    buffer(doc, victim)["offset"] = buffer(doc, shared[1])["offset"]  # lands in the middle of the shared region: used together with at least one of them
    with pytest.raises(TilefishManifestError, match="no step uses together|overlap"):
        _plan.validate_doc(doc)


def _live(doc):
    """The live range of every buffer that a step of the first bucket binds: (first step, last step)."""
    first, last = {}, {}
    for si, st in enumerate(doc["buckets"][0]["steps"]):
        for a in st["args"]:
            first.setdefault(a["binding"], si)
            last[a["binding"]] = si
    return {n: (first[n], last[n]) for n in first}


def _regions(doc):
    return {b["name"]: (b["offset"], b["offset"] + _plan.align_up(b["bytes"])) for b in doc["buffers"] if b["role"] in _plan.ARENA_ROLES}


def _resize(doc):
    """``arena_bytes`` as the moved regions make it (the overlap rules are what is tested, not the size)."""
    doc["arena_bytes"] = _plan.align_up(max(end for _, end in _regions(doc).values()))


def test_a_region_that_may_not_share_is_refused_inside_any_active_region_not_only_one_that_may_not_share(good_reuse):
    """The sweep refuses a non-sharing region (here the device scalar ``n``) that starts while *any* region is active, a sharing one included (a sweep that
    compared it with the active non-sharing region alone would let it into an intermediate)."""
    doc = copy.deepcopy(good_reuse)
    regions = _regions(doc)
    host = next(
        n
        for n, (lo, hi) in regions.items()
        if buffer(doc, n)["role"] == "intermediate" and hi - lo >= 1024 and not any(lo + 256 <= o < lo + 512 for m, (o, _) in regions.items() if m != n)
    )
    buffer(doc, "n")["offset"] = regions[host][0] + 256  # inside the intermediate, past its start: the intermediate is active and may share
    _resize(doc)
    with pytest.raises(TilefishManifestError, match=rf"the regions of 'n' and '{host}' overlap"):
        _plan.validate_doc(doc)


def test_only_an_intermediate_may_share_a_device_scalar_never_even_with_disjoint_live_ranges(good_reuse):
    """``n`` (read by step 2 only) put on an intermediate that no step uses with it: disjoint live ranges, which would let two intermediates share, but a
    device scalar never shares (a sweep that let any role share would accept it)."""
    doc = copy.deepcopy(good_reuse)
    live, regions = _live(doc), _regions(doc)
    n_first, n_last = live["n"]

    def apart(m):
        lo, hi = live.get(m, (-1, -1))
        return hi < n_first or n_last < lo

    host = next(
        x
        for x, (lo, hi) in regions.items()
        if buffer(doc, x)["role"] == "intermediate"
        and all(apart(m) for m, (o, e) in regions.items() if m not in ("n", x) and o < lo + 256 and lo < e)  # everything under n's new region is apart from n
        and apart(x)
    )
    buffer(doc, "n")["offset"] = regions[host][0]
    _resize(doc)
    with pytest.raises(TilefishManifestError, match=r"the regions of '(n|%s)' and '(n|%s)' overlap" % (host, host)):
        _plan.validate_doc(doc)


def test_a_region_is_active_up_to_its_last_byte(good):
    """Two regions of a plan without reuse that overlap by the last 256 bytes of the first: refused (a sweep that retired a region 256 bytes early would not see it)."""
    doc = copy.deepcopy(good)
    regions = sorted(_regions(doc).items(), key=lambda kv: kv[1])
    (a, (lo_a, hi_a)), (b, (lo_b, _hi_b)) = regions[2], regions[3]
    assert hi_a == lo_b and hi_a - lo_a >= 512  # consecutive, the first larger than 256 bytes
    buffer(doc, b)["offset"] = hi_a - 256
    _plan.validate_doc(copy.deepcopy(good))
    with pytest.raises(TilefishManifestError, match=rf"the regions of '{b}' and '{a}' overlap"):
        _plan.validate_doc(doc)
    buffer(doc, b)["offset"] = hi_a  # back where it ends: fine
    _plan.validate_doc(doc)


def _view_args(doc, bucket=1):
    """The attention step of a bucket (M = 64) and its view of v in the fused qkv ``t1``: offset 512, the last element of t1 is its last."""
    st = next(s for s in doc["buckets"][bucket]["steps"] if s["entry"] == "attention")
    v = next(a for a in st["args"] if a["name"] == "v")
    assert v["binding"] == "t1" and v["view"] == {"shape": [64, 4, 64], "strides": [768, 64, 1], "offset": 512}
    return st, v


def test_a_view_that_ends_at_the_last_element_of_its_buffer_is_inside_and_one_more_is_outside(good):
    """``offset + 1 + sum((shape - 1) * stride) <= elements``: v ends exactly at the end of t1 (64 * 768 elements); one element further is refused, by the
    reader (validate_doc) and by what binds a step (resolve_step)."""
    import plan_model as pm

    doc = copy.deepcopy(good)
    st, v = _view_args(doc)
    assert v["view"]["offset"] + 1 + 63 * 768 + 3 * 64 + 63 == 64 * 768
    _plan.validate_doc(doc)
    v["view"]["offset"] = 513
    with pytest.raises(TilefishManifestError, match=r"the view reaches past the end of buffer 't1' at \{'M': 64\}"):
        _plan.validate_doc(doc)
    v["view"]["offset"] = 512
    v["view"]["shape"] = [64, 4, 65]  # the same end, by the shape: one element past
    with pytest.raises(TilefishManifestError, match=r"the view reaches past the end of buffer 't1'"):
        _plan.validate_doc(doc)
    # resolve_step, which a runner binds every step with
    made = tilefish.plan(pm.block)
    shapes = _plan.buffer_shapes(made._doc, {"M": 64})
    callee = made.callees["attention"]
    st = copy.deepcopy(_view_args(made._doc)[0])
    _plan.resolve_step(callee, st["args"], shapes, "the attention")  # as planned: the last element of v is the last of t1
    for change in ({"offset": 513}, {"shape": [64, 4, 65]}):
        bad = copy.deepcopy(st)
        next(a for a in bad["args"] if a["name"] == "v")["view"].update(change)
        with pytest.raises(TilefishPlanError, match=r"the view of 't1' for argument 'v' .* reaches past the 49152 elements of the buffer"):
            _plan.resolve_step(callee, bad["args"], shapes, "the attention")


def test_a_document_that_is_not_an_object_or_has_odd_types_is_a_refusal_not_a_crash(good):
    for bad in (None, [], "plan", 1, {"schema": "tilefish.plan.v1"}):
        with pytest.raises(TilefishManifestError):
            _plan.validate_doc(bad)
    doc = copy.deepcopy(good)
    doc["buckets"][0]["steps"][0]["args"][0] = 5
    with pytest.raises(TilefishManifestError):
        _plan.validate_doc(doc)
    doc = copy.deepcopy(good)
    doc["buffers"][0]["shape"] = ["(((("]
    with pytest.raises(TilefishManifestError):
        _plan.validate_doc(doc)


# -- the plan against its manifest --------------------------------------------------------------------------------------------------------------


@pytest.fixture
def against(exported):
    directory, p = exported
    return doc_of(directory), Manifest.load(directory)


def refused(doc, manifest, fragment):
    with pytest.raises(TilefishManifestError, match=fragment):
        _plan.check_against_manifest(doc, manifest)


def test_every_step_must_be_what_the_manifests_entries_say(against):
    doc, manifest = against
    _plan.check_against_manifest(doc, manifest)

    def broken(mutate):
        d = copy.deepcopy(doc)
        mutate(d)
        return d

    steps = lambda d: d["buckets"][0]["steps"]  # noqa: E731
    refused(broken(lambda d: steps(d)[0].update(entry="nothing")), manifest, "has no kernel 'nothing' for sm_103")
    refused(broken(lambda d: steps(d)[1].update(entry="gemm_o.M32")), manifest, "binary takes 4 pointers|argument 'a'|binds")
    refused(broken(lambda d: steps(d)[1].update(entry="nothing.M32")), manifest, "has no library entry")
    refused(broken(lambda d: d.update(arch="sm_90")), manifest, "has no kernel 'rmsnorm' for sm_90")
    refused(broken(lambda d: steps(d)[0]["args"][0].update(name="y")), manifest, "is bound where the binary has 'x'")
    refused(broken(lambda d: steps(d)[0]["args"].pop()), manifest, "takes 3 pointers")
    refused(broken(lambda d: steps(d)[0]["args"][1].update(binding="x")), manifest, "has rank 2, but the binary takes rank 1")  # w bound to the activations
    refused(broken(lambda d: steps(d)[0]["args"][0].update(binding="w_ln1")), manifest, "has rank")
    refused(broken(lambda d: buffer(d, "x").update(dtype="f32")), manifest, "takes bf16")
    refused(broken(lambda d: steps(d)[4]["args"][3].update(binding="w_o")), manifest, "is written by the binary but bound to 'w_o', whose role is weight")
    refused(broken(lambda d: steps(d)[0]["args"][2].update(binding="x")), manifest, "is written by the binary but bound to 'x', whose role is input")
    refused(broken(lambda d: buffer(d, "n").update(role="input", offset=None)), manifest, "a device scalar .a pointer of shape \\[\\]. is bound to the buffer of role devscalar")
    refused(broken(lambda d: steps(d)[0]["args"][0].update(binding="n")), manifest, "is bound to .n. .i32., but the binary takes bf16")


def test_a_buffer_is_written_before_it_is_read(against):
    doc, manifest = against
    d = copy.deepcopy(doc)
    first, second = d["buckets"][0]["steps"][0], d["buckets"][0]["steps"][1]
    first["args"][2]["binding"], second["args"][0]["binding"] = "t5", "t5"  # step 0 writes t5 (an intermediate no one reads yet), step 1 reads it: fine
    _plan.check_against_manifest(d, manifest)
    d = copy.deepcopy(doc)
    d["buckets"][0]["steps"][1]["args"][0]["binding"] = "t4"  # reads t4, which only step 4 writes
    refused(d, manifest, "reads 't4' before any step has written it")


def test_an_in_place_pointer_reads_its_buffer_so_the_buffer_is_written_before(tmp_path):
    """An ``inout`` pointer reads what it updates: bound to an intermediate no step has written yet, it is refused like an ``in`` pointer (a check of the
    ``in`` pointers alone would let it read garbage). ``accumulate``: step 2 (``twice``) writes t0 and step 3 (``bump``) updates it in place."""
    exported = plan_fakes.export(tilefish.plan(pm.accumulate), tmp_path)
    doc, manifest = exported.doc, exported.manifest
    steps = doc["buckets"][0]["steps"]
    assert [s["entry"] for s in steps] == ["bump", "bump", "twice", "bump", "plus"] and steps[3]["args"][0] == {"name": "state", "binding": "t0"}
    _plan.check_against_manifest(doc, manifest)
    pointers = {p["name"]: p["access"] for p in manifest.get("bump", "sm_103").entry["abi"]["params"] if p["kind"] == "pointer"}
    assert pointers["state"] == "inout"
    d = copy.deepcopy(doc)
    d["buckets"][0]["steps"][2]["args"][1]["binding"] = "out0"  # twice writes the output instead: t0 is first touched by the in-place bump
    refused(d, manifest, r"step #3 \(bump\) of bucket \{'Q': 16\}: argument 'state' reads 't0' before any step has written it")


def test_every_output_is_written_by_a_step(against):
    """A returned value that no step writes would come back as whatever its tensor held: refused (the last step's result rebound to an intermediate)."""
    doc, manifest = against
    d = copy.deepcopy(doc)
    last = d["buckets"][0]["steps"][-1]
    assert last["args"][-1] == {"name": "out", "binding": "out0"}
    last["args"][-1]["binding"] = "t0"  # an intermediate of the same dtype and shape, which an earlier step wrote: every other rule holds
    refused(d, manifest, r"bucket \{'M': 32\}: no step writes the output 'out0'")
    d = copy.deepcopy(doc)
    d["buckets"][2]["steps"][-1]["args"][-1]["binding"] = "t0"  # in one bucket only
    refused(d, manifest, r"bucket \{'M': 128\}: no step writes the output 'out0'")


def test_a_library_step_needs_the_workspace_the_entry_says(against):
    doc, manifest = against
    d = copy.deepcopy(doc)
    buffer(d, "workspace")["bytes"] = 1000
    refused(d, manifest, "needs 8519680")


def test_every_bucket_is_checked_not_only_the_first(against):
    doc, manifest = against
    d = copy.deepcopy(doc)
    d["buckets"][2]["steps"][0]["entry"] = "nothing"
    refused(d, manifest, r"has no kernel 'nothing'")


def test_a_plan_directory_loads_through_the_runner_cache_with_the_same_refusals(exported, directory):
    from tilefish import _planrun

    _, p = exported
    runner = _planrun._replay_runner(pm.block, directory, p.digest, False)
    assert runner.plan.digest == p.digest
    assert _planrun._replay_runner(pm.block, directory, p.digest, False) is runner  # cached until a file in the directory changes
    with pytest.raises(TilefishManifestError, match="but the pin is"):
        _planrun._replay_runner(pm.block, directory, WRONG, False)
    with pytest.raises(TilefishRuntimeError, match="not both"):
        _planrun._replay_runner(pm.block, directory, p.digest, True)
    (directory / "plan.json").write_text((directory / "plan.json").read_text().replace('"offset": 256', '"offset": 512', 1))
    with pytest.raises(TilefishManifestError, match="but the pin is"):
        _planrun._replay_runner(pm.block, directory, p.digest, False)  # the directory changed: it is read again, and refused
    _planrun.clear_caches()
