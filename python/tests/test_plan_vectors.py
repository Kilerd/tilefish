"""The plan documents the Rust reader is held to (``crates/tilefish/tests/data/python_plans.json``, written by ``export_plan_vectors.py``) are what the
code decides now: a change to the format, or to a rule of ``validate_doc`` or ``check_against_manifest``, shows up here until the file is written again.
"""

import json
import pathlib

import export_plan_vectors as ev
import pytest

COMMITTED = pathlib.Path(__file__).resolve().parents[2] / "crates" / "tilefish" / "tests" / "data" / "python_plans.json"


@pytest.fixture(scope="module")
def committed():
    if not COMMITTED.exists() or not (ev.GOLDEN / "expected.json").exists():
        pytest.skip("the Rust crate and the golden plans are not available (not running from a source checkout)")
    return json.loads(COMMITTED.read_text())["documents"]


def test_the_committed_vectors_are_what_python_decides_now(committed):
    made = ev.documents()
    assert len(made) == len(committed)
    changed = [m["name"] for m, c in zip(made, committed) if m != c]
    assert not changed, f"{len(changed)} vectors differ (first: {changed[:5]}); regenerate with python/tests/export_plan_vectors.py"


def test_the_vectors_hold_both_verdicts_and_every_rule(committed):
    accepted = [v for v in committed if v["accepted"]]
    refused = [v for v in committed if not v["accepted"]]
    assert len(accepted) >= 50 and len(refused) >= 700
    # every named mutation is there for both plans, and the file as it is is accepted
    names = {v["name"] for v in committed}
    assert {name for name, _ in ev.mutations()} <= names
    assert all(v["accepted"] for v in committed if v["name"] == "the file as it is")
    # a shape expression Python's dimension algebra folds is accepted, one it cannot is refused
    verdicts = {(v["base"], v["name"]): v["accepted"] for v in committed}
    assert verdicts[("block", "shape '2*M//2'")] and verdicts[("block", "shape 'M+N-N'")]
    assert not verdicts[("block", "shape 'M*M//M'")] and not verdicts[("block", "shape 'M // 2 * 2'")]
