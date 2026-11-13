"""The committed fixtures of the Rust crate (``crates/tilefish/tests/fixtures``) are manifests of schema 2, which is what the crate reads.

Python reads schema 3 only (``docs/manifest-schema-3.md``). A schema 2 manifest of Triton kernels becomes a schema 3 manifest by
the three additive changes below, and nothing else: so the tests that need a *real* manifest (built on a B300, with its cubins) migrate a copy
of a fixture instead of rebuilding it, and the digest the Rust tests expect (``expected.json``) is still computed by Python, from the
schema 2 document, with ``manifest_digest`` (which digests both schemas). Not collected: the name has no ``test_`` prefix.

* every kernel entry gets ``"kind": "triton"``;
* every ABI gets ``"scratch_slots": 2`` (a Triton kernel's two scratch slots, which it already lists in ``scratch``);
* the manifest gets ``"library": []`` and ``"schema": 3``.
"""

from __future__ import annotations

import copy
import json
import pathlib
import shutil
from typing import Any


def migrate_doc(doc: "dict[str, Any]") -> "dict[str, Any]":
    """The schema 3 document of a schema 2 manifest of Triton kernels."""
    assert doc["schema"] == 2, doc["schema"]
    out = copy.deepcopy(doc)
    out["schema"] = 3
    for entry in out["kernels"]:
        entry["kind"] = "triton"
        entry["abi"]["scratch_slots"] = len(entry["abi"]["scratch"])
        assert entry["abi"]["scratch_slots"] == 2
    out["library"] = []
    return out


def migrated_copy(source: "pathlib.Path", destination: "pathlib.Path") -> "dict[str, Any]":
    """Copy the fixture directory ``source`` to ``destination`` with its manifest migrated to schema 3; returns the schema 3 document."""
    shutil.copytree(source, destination)
    doc = migrate_doc(json.loads((destination / "manifest.json").read_text()))
    (destination / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n")
    return doc
