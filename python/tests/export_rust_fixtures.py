"""Builds the fixtures of the Rust crate (``crates/tilefish/tests/fixtures``). Not collected: the name has no ``test_`` prefix.

Run on a machine with a GPU and the ``gpu`` extra (the cubins are built by Triton, the library entries pinned by cuBLASLt), from
``python/tests``::

    python export_rust_fixtures.py OUTPUT_DIR [--archs 103,120] [--gemm-shapes committed|all]

For each architecture it exports to ``OUTPUT_DIR/sm_<arch>/`` with ``tilefish.runtime.export`` (manifest schema 3: ``manifest.json``, the cubins) the operators
of ``host_ops.ops()`` (the five lowered goldens, a bucketed row kernel, the hand-written Triton externs, the strided views and device scalars
of ``gpu_kernels_p1.py`` and two prebuilt hand-written PTX externs), the prebuilt externs of ``host_ops.gpu_ops()`` (the PTX of two Triton
builds, and a cubin: their source files are written to ``OUTPUT_DIR/sources/sm_<arch>/``, which ``rust_gate.py --manifest`` reuses) and, **for
the architecture of the device only**, the pinned cuBLASLt GEMMs of ``host_ops.library_ops()``: a pin is made on a device of its architecture, so a
fixture that is cross-compiled (any other architecture, which needs nothing but ptxas) has no ``library`` entries. The committed fixtures hold two
``(N, K)`` shapes of ``gemm`` and ``gemm_bias`` (``--gemm-shapes committed``, 12 entries: buckets 32, 64 and 320 of ``M``); the gate exports every one
(``rust_gate.py prepare``). Then it writes

* ``expected.json``: for each fixture the digest of the manifest, for every binary its ``kind``, ``sha256`` and ``sha256_code`` and for every library entry its
  algorithm, the cuBLASLt version that chose it, the SM count of the device it chose it on and its workspace, all computed by Python (``Manifest.digest``, ``code_digest``). The Rust tests
  (``tests/cross_language.rs``) reproduce them from the files with no GPU and no Python; ``test_rust_host.py`` checks them against Python on every ``pytest`` run.
* ``generated.rs``: the launchers ``tilefish.hosts.rust.emit`` writes for the fixture of the device's architecture (the first architecture when the
  device is none of them), from the declarations that need no GPU (``host_ops.ops()`` and the GEMMs: ``test_rust_host.py`` writes the same file without one; the
  docs of the prebuilt externs of ``gpu_ops`` have no tilefish signature). The launchers of another architecture differ only in ``MANIFEST_DIGEST``, and have no GEMM.

The fixtures are committed; they only change when a binary does (another Triton, ptxas, cuBLASLt, or kernel), and then the files
of ``crates/tilefish/tests/fixtures`` are replaced together.

Fixtures built before a library entry recorded the SM count of its device (``sm_count``) are migrated without a GPU, keeping every binary and algorithm::

    python export_rust_fixtures.py ../../crates/tilefish/tests/fixtures --migrate-sm-count 30

An operator added to ``host_ops.ops()`` is added to the committed fixtures without building the others again (their cubins, and so their digests, stay
as they are; a fresh export would give the same machine code in other files, since a cubin names the directory it was built in)::

    python export_rust_fixtures.py ../../crates/tilefish/tests/fixtures --add-missing [--archs 103,120]

which exports, for each architecture, the operators of ``ops()`` that the fixture has no binary of into its directory (``tilefish.runtime.export`` merges
into a manifest that is there), then writes ``expected.json`` and ``generated.rs`` again.
"""

import argparse
import json
import pathlib
import shutil
import sys

from host_ops import GEMM_SHAPES, GEMM_SHAPES_COMMITTED, gpu_ops, library_ops, ops

from tilefish import runtime
from tilefish._elf import code_digest
from tilefish.hosts import rust


def expected(manifest):
    return {
        "manifest_digest": manifest.digest,
        "kernels": [
            {
                "stem": a.stem,
                "arch": a.arch,
                "kind": a.entry["kind"],
                "cubin": a.entry["cubin"],
                "sha256": a.entry["sha256"],
                "sha256_code": code_digest(a.cubin),
            }
            for a in sorted(manifest.artifacts, key=lambda a: (a.stem, a.arch))
        ],
        "library": [
            {
                "stem": e["stem"],
                "arch": e["arch"],
                "impl": e["impl"],
                "algo": e["algo"],
                "cublaslt_version": e["cublaslt_version"],
                "sm_count": e["sm_count"],
                "workspace_bytes": e["workspace_bytes"],
            }
            for e in sorted(manifest.library_entries, key=lambda e: (e["stem"], e["arch"]))
        ],
    }


def native_capability():
    """The compute capability of the device (``103``), or ``None`` when there is no torch or no device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    except ImportError:
        return None


def with_sm_count(doc, sm_count):
    """A manifest document whose library entries predate ``sm_count`` (the multiprocessor count of the device a pin was made on), as it is with the field:
    ``sm_count`` placed after ``cublaslt_version``, and the declaration digest of the committed declaration on a device of that SM count (an entry's
    declaration digest covers it). The cubins and the kernels are untouched."""
    import copy

    from tilefish._library import library_specs

    specs = {s.stem: s for op in library_ops(GEMM_SHAPES).values() for s in library_specs(op)}
    out = copy.deepcopy(doc)
    for i, e in enumerate(out["library"]):
        items = [(k, v) for k, v in e.items() if k != "sm_count"]
        at = [k for k, _ in items].index("cublaslt_version") + 1
        out["library"][i] = dict(items[:at] + [("sm_count", sm_count)] + items[at:])
        out["library"][i]["declaration_digest"] = specs[e["stem"]].pinned_digest(sm_count)
    return out


def migrate_sm_count(output, sm_count):
    """Give the library entries of the committed fixtures in ``output`` the ``sm_count`` of the device they were pinned on (no GPU: the algorithms stay), then write
    ``expected.json`` and ``generated.rs`` again from them. For the fixtures built before library entries had the field."""
    from tilefish import Manifest

    summary = json.loads((output / "expected.json").read_text())["fixtures"]
    for name in summary:
        path = output / name / "manifest.json"
        doc = with_sm_count(json.loads(path.read_text()), sm_count)
        path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest = Manifest.load(output / name)
        manifest = manifest.check_pin(manifest.digest)
        summary[name] = expected(manifest)
        if manifest.library_entries:
            rust.emit(manifest, output / "generated.rs", ops=list(ops().values()) + list(library_ops(GEMM_SHAPES_COMMITTED).values()))
        print(f"{name}: manifest digest {manifest.digest}", file=sys.stderr)
    (output / "expected.json").write_text(json.dumps({"fixtures": summary}, indent=2) + "\n", encoding="utf-8")


def add_missing(output, caps):
    """Export the operators of ``ops()`` that a committed fixture has no binary of into its directory, for each of ``caps`` (merged into the manifest
    that is there: every other binary and library entry stays), then write ``expected.json`` and ``generated.rs`` again. Needs Triton and ptxas, not a GPU
    of the architecture (a library entry is not added: those are pinned on a device)."""
    from tilefish import Manifest

    summary = json.loads((output / "expected.json").read_text())["fixtures"]
    native = native_capability()
    emit_cap = native if native in caps else caps[0]
    plain = list(ops().values())
    gemms = list(library_ops(GEMM_SHAPES_COMMITTED).values())
    for cap in caps:
        name = f"sm_{cap}"
        directory = output / name
        have = Manifest.load(directory, pin=summary[name]["manifest_digest"])
        stems = {a.stem for a in have.artifacts}
        missing = [op for op in plain if any(s.stem not in stems for s in runtime._target(op).compile_specs())]
        manifest = runtime.export(missing, directory, arch=cap) if missing else have
        if cap == emit_cap:
            rust.emit(manifest, output / "generated.rs", ops=plain + gemms)
        summary[name] = expected(manifest)
        print(f"{name}: added {[op.name for op in missing]}, manifest digest {manifest.digest}", file=sys.stderr)
    (output / "expected.json").write_text(json.dumps({"fixtures": summary}, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--archs", default="103,120", help="compute capabilities to build for (default: 103,120)")
    parser.add_argument("--gemm-shapes", choices=["committed", "all"], default="committed", help="the (N, K) shapes of the GEMMs of the device's fixture")
    parser.add_argument(
        "--migrate-sm-count",
        type=int,
        metavar="N",
        help="no build: give the library entries of the fixtures in OUTPUT the sm_count N of the device they were pinned on, and write expected.json and generated.rs again",
    )
    parser.add_argument(
        "--add-missing",
        action="store_true",
        help="build only the operators of ops() that the fixtures in OUTPUT have no binary of, merge them in, and write expected.json and generated.rs again",
    )
    args = parser.parse_args(argv)
    if args.migrate_sm_count is not None:
        migrate_sm_count(args.output, args.migrate_sm_count)
        return
    caps = [int(c) for c in args.archs.split(",")]
    if args.add_missing:
        add_missing(args.output, caps)
        return
    native = native_capability()
    emit_cap = native if native in caps else caps[0]
    shapes = GEMM_SHAPES_COMMITTED if args.gemm_shapes == "committed" else GEMM_SHAPES
    summary = {}
    for cap in caps:
        directory = args.output / f"sm_{cap}"
        if directory.exists():
            shutil.rmtree(directory)
        sources = args.output / "sources" / f"sm_{cap}"
        if sources.exists():
            shutil.rmtree(sources)
        plain = list(ops().values())  # declared without a GPU
        declared = plain + list(gpu_ops(sources, cap).values())
        gemms = list(library_ops(shapes).values())
        if cap == native:
            declared += gemms
        manifest = runtime.export(declared, directory, arch=cap)
        if cap == emit_cap:
            # the declarations that need no GPU (and those of the GEMMs, which the manifest may not have): what ``test_rust_host.py`` emits the committed file from,
            # so the three prebuilt externs of ``gpu_ops`` have no tilefish signature in their docs. The manifest that export returned is pinned.
            rust.emit(manifest, args.output / "generated.rs", ops=plain + gemms)
        summary[f"sm_{cap}"] = expected(manifest)
        print(
            f"sm_{cap}: {len(manifest.artifacts)} binaries and {len(manifest.library_entries)} library entries, manifest digest {manifest.digest}",
            file=sys.stderr,
        )
    (args.output / "expected.json").write_text(json.dumps({"fixtures": summary}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
