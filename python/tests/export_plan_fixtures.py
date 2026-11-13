"""Writes the golden plan fixtures (``python/tests/golden/plans``). Not collected: the name has no ``test_`` prefix. Needs no GPU.

    python export_plan_fixtures.py [OUTPUT_DIR]        # default: golden/plans next to this file

Three directories, ``block`` and ``block-reuse`` (the transformer block of ``plan_model`` planned without and with liveness reuse) and ``steps`` (the graph
``plan_model.steps``: a result written over its input, two copy steps, a returned reshape and named outputs), each exported to ``sm_103`` the way
``tilefish.runtime.export`` does, but with the *fake* builds of ``plan_fakes`` (a tiny ELF file per kernel entry, a made-up 64-byte algorithm per GEMM): what is real is
the format, the digests and the plan; the machine code is not, so the fixtures validate a reader and cannot be launched. Each directory has ``manifest.json`` (schema 3),
the cubins and ``plan.json``; ``expected.json`` has, for each, the plan digest and the manifest digest that Python computes, the arena size and every buffer's offset, which a
reader in another language must reproduce from the files alone (``docs/plan-schema-1.md``). ``test_plan_fixtures.py`` regenerates all of it and requires the committed files
to be the same bytes (the ``tilefish`` version in a file is not compared: it is not digested), and loads each directory as a replay does.
"""

import json
import pathlib
import shutil
import sys

import plan_fakes
import plan_model as pm

import tilefish
from tilefish import Manifest

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT = HERE / "golden" / "plans"
PLANS = {"block": ("block", False), "block-reuse": ("block", True), "steps": ("steps", False), "given": ("two_scales", False)}  # directory: (the graph of plan_model, reuse)
SET = "set"  # a plan set: the plans of gpu_kernels_p6.tile_chain for 1, 2 and 3 tiles over one manifest (plans.json, <graph>.plan.json)


def export_all(output: pathlib.Path) -> dict:
    expected = {}
    for name, (graph, reuse) in PLANS.items():
        directory = output / name
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        exported = plan_fakes.export(tilefish.plan(getattr(pm, graph), reuse=reuse), directory)
        manifest = Manifest.load(directory)
        expected[name] = {
            "graph": exported.name,
            "arch": exported.arch,
            "reuse": reuse,
            "plan_digest": exported.digest,
            "manifest_digest": manifest.digest,
            "arena_bytes": exported.arena_bytes,
            "offsets": {b["name"]: b["offset"] for b in exported.buffers if b["offset"] is not None},
            "buckets": exported.buckets(),
            "steps_per_bucket": len(exported.steps(**exported.buckets()[-1])),
            "outputs": exported.outputs,
            "roles_declared": exported.roles_declared,
            "weights": exported.weights,
        }
    import gpu_kernels_p6 as p6

    directory = output / SET
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    exported_set = plan_fakes.export_set([tilefish.plan(p6.TILES[t]) for t in (1, 2, 3)], directory)
    expected[SET] = {
        "arch": exported_set.doc["arch"],
        "set_digest": exported_set.digest,
        "manifest_digest": exported_set.manifest_digest,
        "plans": {name: p.digest for name, p in exported_set.plans.items()},
        "steps": {name: len(p.steps()) for name, p in exported_set.plans.items()},
        # the arena the three plans share (docs/plan-schema-1.md, section 15.1): the largest of their arenas, each plan at its own offsets
        "arena_bytes": {name: p.arena_bytes for name, p in exported_set.plans.items()},
        "shared_arena_bytes": max(p.arena_bytes for p in exported_set.plans.values()),
    }
    (output / "expected.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    return expected


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    target.mkdir(parents=True, exist_ok=True)
    for name, e in export_all(target).items():
        print(name, e.get("plan_digest", e.get("set_digest")), e["manifest_digest"], e.get("arena_bytes", ""))
