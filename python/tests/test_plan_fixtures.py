"""The golden plan fixtures (``golden/plans``): a plan directory and the digests a reader in another language must reproduce from the files alone.

``export_plan_fixtures.py`` writes them (the transformer block of ``plan_model``, with and without reuse, on fake builds: the format is real, the machine code is not). Here the
committed files are required to be what the code makes now, byte for byte (but the ``tilefish`` version, which is not digested), each directory is loaded as a replay loads
it, and the digests are recomputed with nothing but ``json`` and ``hashlib``, the way an implementation of ``docs/plan-schema-1.md`` does.
"""

import hashlib
import json

import export_plan_fixtures as fx
import plan_model as pm
import pytest

from tilefish import _plan
from tilefish._manifest import Manifest, manifest_digest


@pytest.fixture(scope="module")
def committed():
    if not (fx.DEFAULT / "expected.json").exists():
        pytest.skip("the golden plan fixtures are not available (not running from a source checkout)")
    return json.loads((fx.DEFAULT / "expected.json").read_text())


def without_version(path):
    doc = json.loads(path.read_text())
    doc.pop("tilefish", None)  # the version of the package that wrote the file: not digested, so not compared
    return doc


def test_the_committed_fixtures_are_what_the_code_makes(committed, tmp_path):
    made = fx.export_all(tmp_path)
    assert made == committed
    for name in [*fx.PLANS, fx.SET]:
        assert sorted(f.name for f in (tmp_path / name).iterdir()) == sorted(f.name for f in (fx.DEFAULT / name).iterdir())
        for f in sorted((fx.DEFAULT / name).iterdir()):
            if f.suffix == ".json":
                assert without_version(f) == without_version(tmp_path / name / f.name), (name, f.name)
            else:
                assert f.read_bytes() == (tmp_path / name / f.name).read_bytes(), (name, f.name)


@pytest.mark.parametrize("name", list(fx.PLANS))
def test_each_directory_loads_as_a_replay_loads_it(committed, name):
    e = committed[name]
    graph = getattr(pm, fx.PLANS[name][0])
    loaded = _plan.load_plan(fx.DEFAULT / name, graph, pin=e["plan_digest"])
    assert loaded.digest == e["plan_digest"] and loaded.manifest_digest == e["manifest_digest"] and loaded.manifest.digest == e["manifest_digest"]
    assert loaded.arch == e["arch"] == "sm_103" and loaded.reuse is e["reuse"] and loaded.arena_bytes == e["arena_bytes"]
    assert {b["name"]: b["offset"] for b in loaded.buffers if b["offset"] is not None} == e["offsets"]
    assert loaded.buckets() == e["buckets"] and len(loaded.steps(**loaded.buckets()[-1])) == e["steps_per_bucket"] == {"steps": 4, "given": 2}.get(name, 9)
    assert loaded.outputs == e["outputs"] == (["y", "half"] if name == "steps" else ["out0"])


def test_the_given_plan_launches_one_binary_with_the_values_of_each_call(committed):
    loaded = _plan.load_plan(fx.DEFAULT / "given", pm.two_scales, pin=committed["given"]["plan_digest"])
    for rg in (8, 24):
        first, second = loaded.steps(Rg=rg)
        assert first["entry"] == second["entry"] == "scale_add"
        assert (first["given"], second["given"]) == ({"k": 3, "c": -2}, {"k": -1, "c": rg})
    entry = loaded.manifest.get("scale_add", "sm_103").entry
    assert [p["kind"] for p in entry["abi"]["params"]][-2:] == ["given", "given"]


def test_the_steps_plan_has_what_m3_3_added(committed):
    loaded = _plan.load_plan(fx.DEFAULT / "steps", pm.steps, pin=committed["steps"]["plan_digest"])
    steps = loaded.steps(Rf=24)
    assert [s["kind"] for s in steps] == ["copy", "kernel", "copy", "kernel"]
    over = {a["name"]: a["binding"] for a in steps[1]["args"]}
    assert over["r"] == over["r_out"] == "t0"  # the result written over its input
    assert [b["shape"] for b in loaded.buffers if b["name"] == "half"] == [["Rf", 64]] and "view" in steps[3]["args"][1]  # the returned reshape
    assert loaded.output_buffer("out0") == "y" and loaded.output_buffer("out1") == "half" and loaded.output_buffer("out2") is None


@pytest.mark.parametrize("name", list(fx.PLANS))
def test_the_digests_are_recomputed_from_the_files_with_json_and_hashlib_alone(committed, name):
    e = committed[name]
    directory = fx.DEFAULT / name
    plan = json.loads((directory / "plan.json").read_text())
    body = {k: v for k, v in plan.items() if k not in ("plan_digest", "tilefish")}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    assert hashlib.sha256(b"tilefish.plan-digest.v1\0" + text.encode()).hexdigest() == plan["plan_digest"] == e["plan_digest"]
    manifest = json.loads((directory / "manifest.json").read_text())
    key = lambda entry: (entry["stem"], entry["arch"])  # noqa: E731
    body = {"schema": 3, "kernels": sorted(manifest["kernels"], key=key), "library": sorted(manifest["library"], key=key)}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(b"tilefish.manifest-digest.v3\0" + text.encode()).hexdigest() == plan["manifest_digest"] == e["manifest_digest"]
    assert manifest_digest(manifest) == e["manifest_digest"] == Manifest.load(directory).digest
    # the cubins are the files the manifest names, with the digests it records
    for entry in manifest["kernels"]:
        assert hashlib.sha256((directory / entry["cubin"]).read_bytes()).hexdigest() == entry["sha256"]


def test_the_two_plans_share_a_manifest_and_differ_in_where_buffers_are(committed):
    plain, reused = committed["block"], committed["block-reuse"]
    assert plain["manifest_digest"] == reused["manifest_digest"] and plain["plan_digest"] != reused["plan_digest"]
    assert reused["arena_bytes"] < plain["arena_bytes"] and plain["arena_bytes"] == 9634048 and reused["arena_bytes"] == 9240832
    assert plain["offsets"]["n"] == reused["offsets"]["n"] == 0 and plain["offsets"] != reused["offsets"]
