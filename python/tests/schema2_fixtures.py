"""Migrating a manifest of schema 2 to schema 3 (``docs/manifest-schema-3.md``, section 11). Not collected: the name has no ``test_`` prefix.

The committed fixtures of the Rust crate (``crates/tilefish/tests/fixtures``) were manifests of schema 2, which the crate 0.0.1 read; they are manifests of
schema 3 now, written by ``export_rust_fixtures.py``. Python reads schema 3 only: a schema 2 manifest of Triton kernels becomes a schema 3 manifest by the
three additive changes below, and nothing else, so a manifest an older tilefish wrote can be carried over without building its cubins again (and its schema 2
digest is still computed by ``manifest_digest``, which digests both). ``test_rust_host.py: test_a_schema_2_manifest_is_refused_and_becomes_schema_3_by_three_additions``
and ``test_manifest_schema3.py`` check it on a synthetic schema 2 document (a manifest of fake binaries, with the three additions taken out).

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
    """Copy the manifest directory ``source`` to ``destination`` with its manifest migrated to schema 3; returns the schema 3 document."""
    shutil.copytree(source, destination)
    doc = migrate_doc(json.loads((destination / "manifest.json").read_text()))
    (destination / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n")
    return doc
