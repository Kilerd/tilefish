"""Python's decision on every call, as the vectors the Rust host is held to (``crates/tilefish/tests/data/python_launches.json``).

``export_rust_vectors.launches()`` builds calls on the binaries of ``host_ops.ops()`` (a valid call per size and every single change of
one input) and asks the real ``runtime._prepare`` what to do with each, on fake tensors: the grid and the ``kernelParams`` values, or the
kind of the refusal. ``tests/python_launches.rs`` runs the same calls through ``Plan::prepare`` on the committed manifest, which has the
declared shapes and the constraints of the dimensions and nothing else, and requires the same decision for each. Here: the committed
file is what Python decides now, and it covers every kind of refusal a host can make.
"""

import collections
import json
import pathlib

import pytest
from conftest import REPO_ROOT
from export_rust_vectors import launches, launches_json

VECTORS = REPO_ROOT / "crates" / "tilefish" / "tests" / "data" / "python_launches.json"


@pytest.fixture(scope="module")
def made():
    return launches()


def test_every_kind_of_refusal_a_host_makes_is_in_the_vectors(made):
    records, binaries = made
    kinds = collections.Counter(r["expect"].get("kind", "accepted") for r in records)
    assert set(kinds) == {
        "accepted",
        "alias",
        "bucket",
        "divisible",
        "dtype",
        "max_elements",
        "pointer_alignment",
        "rank",
        "shape",
        "static_size",
    }
    assert binaries == 12 and len(records) > 400 and kinds["accepted"] > 80
    # what is accepted has the grid and the parameters of a launch: the pointers and scalars of the ABI, then the two null scratch slots
    for r in records:
        if r["expect"]["ok"]:
            assert len(r["expect"]["grid"]) == 3 and r["expect"]["params"][-2:] == [0, 0]
    print(f"python decided {len(records)} launches of {binaries} binaries: {dict(sorted(kinds.items()))}")


@pytest.mark.skipif(not VECTORS.exists(), reason="the Rust crate is not available (not running from a source checkout)")
def test_the_committed_vectors_are_what_python_decides_now(made):
    records, _ = made
    committed = json.loads(VECTORS.read_text())
    assert len(committed) == len(records)
    differences = [i for i, (a, b) in enumerate(zip(committed, records)) if a != b]
    assert not differences, f"{len(differences)} of {len(records)} launches are decided differently now; regenerate with python/tests/export_rust_vectors.py"
    assert VECTORS.read_text() == launches_json()
    print(f"compared {len(records)} launches of {pathlib.Path(VECTORS).name} with Python: 0 differences")
