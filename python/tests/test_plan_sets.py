"""Plan sets: several plans exported to one directory with one manifest.

A stage whose launch count depends on a key (``gpu_kernels_p6.tile_chain``: ``tiles`` launches of one binary) is one plan per key; exported together
(``tilefish.runtime.export([g1, g2, ...], directory)``) they share **one manifest**, in which each binary has one entry whatever the number of plans,
and an index (``plans.json``) lists the digest of every plan, so one pin (the set's) covers every plan, their manifest and every binary. Each plan is
``<graph>.plan.json`` and is also pinned by its own digest. A replay of one graph takes ``plan=directory`` and either pin.
"""

import json

import gpu_kernels_p6 as k
import plan_fakes
import pytest

import tilefish
from tilefish import TilefishManifestError, TilefishRuntimeError, _plan

GRAPHS = [k.TILES[t] for t in (1, 2, 3)]


@pytest.fixture
def exported(tmp_path):
    plans = [tilefish.plan(g) for g in GRAPHS]
    return tmp_path / "set", plan_fakes.export_set(plans, tmp_path / "set")


def test_the_plans_of_a_set_share_one_manifest_with_one_entry_per_binary(exported):
    directory, s = exported
    files = sorted(f.name for f in directory.iterdir() if f.suffix == ".json")
    assert files == ["manifest.json", "plans.json", "tiles1.plan.json", "tiles2.plan.json", "tiles3.plan.json"]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert sorted(e["stem"] for e in manifest["kernels"]) == ["halve", "scale_add"]  # one entry each, three plans
    assert [len(s[f"tiles{t}"].steps()) for t in (1, 2, 3)] == [2, 3, 4]  # the launch count depends on the key
    assert {p.manifest_digest for p in s.plans.values()} == {s.manifest_digest}
    index = json.loads((directory / "plans.json").read_text())
    assert index["plans"] == {name: p.digest for name, p in s.plans.items()} and index["set_digest"] == s.digest == _plan.set_digest(index)
    assert index["schema"] == "tilefish.plan-set.v1" and index["arch"] == "sm_103"


def test_a_plan_of_the_set_loads_with_its_own_pin_or_the_sets(exported):
    directory, s = exported
    for g in GRAPHS:
        by_set = _plan.load_plan(directory, g, pin=s.digest)
        by_own = _plan.load_plan(directory, g, pin=s[g.name].digest)
        assert by_set.digest == by_own.digest == s[g.name].digest and by_set.set_digest == s.digest and by_own.set_digest is None
        assert _plan.load_plan(directory / f"{g.name}.plan.json", g, trust_directory=True).digest == by_own.digest
    with pytest.raises(TilefishManifestError, match="has digest .* but the pin is"):
        _plan.load_plan(directory, GRAPHS[0], pin=s[GRAPHS[1].name].digest)


def test_an_edited_index_a_plan_it_does_not_list_and_a_plan_that_is_not_the_listed_one_are_refused(exported, tmp_path):
    directory, s = exported
    index = json.loads((directory / "plans.json").read_text())
    # the index edited without its digest: refused, pinned or not
    edited = dict(index, arch="sm_90")
    (directory / "plans.json").write_text(json.dumps(edited))
    with pytest.raises(TilefishManifestError, match="was edited"):
        _plan.read_plan_set(directory)
    # an index that lists another plan digest (rewritten digest: a pin of the old set refuses it, the plan is refused against the new one)
    other = dict(index, plans={**index["plans"], "tiles1": index["plans"]["tiles2"]})
    other["set_digest"] = _plan.set_digest(other)
    (directory / "plans.json").write_text(json.dumps(other))
    with pytest.raises(TilefishManifestError, match="has digest .* but the pin is"):
        _plan.read_plan_set(directory, pin=s.digest)
    with pytest.raises(TilefishManifestError, match="has digest .* but the pin is"):
        _plan.load_plan(directory, GRAPHS[0], pin=other["set_digest"])  # tiles1.plan.json is not the plan the index lists
    # a set that does not list the graph
    fewer = dict(index, plans={"tiles2": index["plans"]["tiles2"]})
    fewer["set_digest"] = _plan.set_digest(fewer)
    (directory / "plans.json").write_text(json.dumps(fewer))
    with pytest.raises(TilefishManifestError, match="lists no plan of graph 'tiles1'"):
        _plan.load_plan(directory, GRAPHS[0], pin=fewer["set_digest"])


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d.update(schema="tilefish.plan-set.v2"), "not a tilefish plan set"),
        (lambda d: d.update(extra=1), "unknown \\['extra'\\]"),
        (lambda d: d.pop("arch"), "missing \\['arch'\\]"),
        (lambda d: d.update(arch="sm_9"), "arch must look like"),
        (lambda d: d.update(manifest_digest="A" * 64), "manifest_digest must be"),
        (lambda d: d.update(plans={}), "plans must be a non-empty object"),
        (lambda d: d.update(plans=[["tiles1", "0" * 64]]), "plans must be a non-empty object"),
        (lambda d: d["plans"].update({"1x": "0" * 64}), "plans maps graph names"),
        (lambda d: d["plans"].update(tiles1="x"), "plans maps graph names"),
        (lambda d: d.update(tilefish=1), "tilefish must be a string"),
    ],
)
def test_the_rules_of_the_index(exported, change, message):
    directory, s = exported
    index = json.loads((directory / "plans.json").read_text())
    change(index)
    if "set_digest" in index:
        index["set_digest"] = _plan.set_digest(index)
    (directory / "plans.json").write_text(json.dumps(index))
    with pytest.raises(TilefishManifestError, match=message):
        _plan.read_plan_set(directory)


def test_a_set_and_a_single_plan_do_not_share_a_directory(exported, tmp_path):
    directory, s = exported
    with pytest.raises(TilefishManifestError, match="holds a plan set"):
        plan_fakes.export(tilefish.plan(GRAPHS[0]), directory)
    single = tmp_path / "single"
    plan_fakes.export(tilefish.plan(GRAPHS[0]), single)
    with pytest.raises(TilefishManifestError, match="holds the plan of one graph"):
        plan_fakes.export_set([tilefish.plan(g) for g in GRAPHS], single)


def test_export_takes_graphs_or_operators_and_distinct_names(tmp_path):
    from tilefish import runtime

    with pytest.raises(TilefishRuntimeError, match="takes graphs, or kernels and externs, not both"):
        runtime.export([GRAPHS[0], k.halve], tmp_path)
    with pytest.raises(TilefishRuntimeError, match="distinct names"):
        runtime.export([GRAPHS[0], GRAPHS[0]], tmp_path, arch=103)
