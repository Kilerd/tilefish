"""Python's decision on every call, as the vectors the Rust host is held to (``crates/tilefish/tests/data/python_launches.json``).

``export_rust_vectors.launches()`` builds calls on the binaries of ``host_ops.ops()`` (a valid call per size and every single change of
one input) and asks the real ``runtime._prepare`` what to do with each, on fake tensors: the grid and the ``kernelParams`` values, or the
kind of the refusal. The calls cover manifest schema 3: strided views (a record has the ``strides`` of a view, in elements, and the address of its
first element), device scalars and prebuilt binaries (a fixed argument has its slot in the ``kernelParams``, and no scratch slots follow).
``tests/python_launches.rs`` runs the same calls through ``Plan::prepare`` on the committed manifest, which has the declared shapes and the
constraints of the dimensions and nothing else, and requires the same decision for each. Here: the committed file is what Python decides now, and
it covers every kind of refusal a host can make.
"""

import collections
import json
import pathlib

import pytest
from conftest import REPO_ROOT
from export_rust_vectors import launches, launches_json

VECTORS = REPO_ROOT / "crates" / "tilefish" / "tests" / "data" / "python_launches.json"
# a Triton kernel is launched with its parameters and the two null scratch slots; these stems are prebuilt binaries, which are not: their
# number of kernelParams is the number of parameters of their ABI (the pointers, the scalars and the fixed arguments)
PREBUILT = {"axpy_ptx": 5, "strided_copy": 4}


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
        "contiguous",
        "divisible",
        "dtype",
        "max_elements",
        "max_extent",
        "pointer_alignment",
        "rank",
        "shape",
        "static_size",
        "stride",
    }
    assert binaries == 25 and len(records) > 900 and kinds["accepted"] > 200
    # what is accepted has the grid and the parameters of a launch: the pointers, scalars and fixed arguments of the ABI, then (for a Triton
    # kernel only) the two null scratch slots
    for r in records:
        if r["expect"]["ok"]:
            assert len(r["expect"]["grid"]) == 3
            if r["stem"] in PREBUILT:
                assert len(r["expect"]["params"]) == PREBUILT[r["stem"]]
            else:
                assert r["expect"]["params"][-2:] == [0, 0]
    print(f"python decided {len(records)} launches of {binaries} binaries: {dict(sorted(kinds.items()))}")


def test_a_view_is_a_record_with_strides_and_a_dense_tensor_is_not(made):
    records, _ = made
    views = [t for r in records for t in r["tensors"] if "strides" in t]
    assert views and all(len(t["strides"]) == len(t["shape"]) for t in views)
    softmax = [r for r in records if r["stem"] == "softmax"]
    assert not any("strides" in t for r in softmax if r["expect"]["ok"] for t in r["tensors"])  # a dense pointer is given a dense tensor
    gaps = [r for r in softmax if any("strides" in t for t in r["tensors"])]
    assert gaps and all(r["expect"] == {"ok": False, "kind": "contiguous"} for r in gaps)  # or a tensor with gaps, which it refuses


def test_the_fixed_arguments_of_a_prebuilt_binary_are_in_the_parameters_and_no_scratch_slot_follows(made):
    records, _ = made
    axpy = next(r for r in records if r["stem"] == "axpy_ptx" and r["expect"]["ok"])
    x, y, out = (t["ptr"] for t in axpy["tensors"])
    n = axpy["tensors"][0]["shape"][0]
    assert axpy["expect"]["params"] == [x, y, out, n, 0x40000000]  # alpha = 2.0f: its IEEE bits, after the dimension scalar; nothing after it
    copy = next(r for r in records if r["stem"] == "strided_copy" and r["expect"]["ok"] and r["tensors"][0]["shape"][0] > 1)
    x, y = (t["ptr"] for t in copy["tensors"])
    assert copy["expect"]["params"] == [x, y, 64, copy["tensors"][0]["strides"][0]]  # cols = 64 (an i32), then the 64-bit row stride


def test_the_stride_of_a_size_one_axis_is_the_one_a_dense_tensor_would_have(made):
    records, _ = made
    one_row = [
        r
        for r in records
        if r["stem"] == "head_copy" and r["expect"]["ok"] and r["tensors"][0]["shape"] == [1, 8, 64] and r["tensors"][0]["strides"][0] != 512
    ]
    assert one_row, "a view of one row whose stride along that axis is another number"
    for r in one_row:
        assert r["expect"]["params"][2:4] == [512, 64]  # canonical: 8 * 64 elements, not the 1536 or 2^40 that was given


@pytest.mark.skipif(not VECTORS.exists(), reason="the Rust crate is not available (not running from a source checkout)")
def test_the_committed_vectors_are_what_python_decides_now(made):
    records, _ = made
    committed = json.loads(VECTORS.read_text())
    assert len(committed) == len(records)
    differences = [i for i, (a, b) in enumerate(zip(committed, records)) if a != b]
    assert not differences, f"{len(differences)} of {len(records)} launches are decided differently now; regenerate with python/tests/export_rust_vectors.py"
    assert VECTORS.read_text() == launches_json()
    print(f"compared {len(records)} launches of {pathlib.Path(VECTORS).name} with Python: 0 differences")
