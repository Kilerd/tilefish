"""Builds the fixtures of the Rust crate (``crates/tilefish/tests/fixtures``). Not collected: the name has no ``test_`` prefix.

Run on a machine with a GPU and the ``gpu`` extra (the cubins are built by Triton), from ``python/tests``::

    python export_rust_fixtures.py OUTPUT_DIR [--archs 103,120]

For each architecture it exports every operator of ``host_ops.ops()`` (the five lowered goldens, a bucketed row kernel and
the hand-written externs) to ``OUTPUT_DIR/sm_<arch>/`` with ``tilefish.runtime.export``: ``manifest.json`` and the cubins.
An architecture other than the device's is cross-compiled, which needs nothing but Triton's ``ptxas``. Then it writes

* ``expected.json``: for each fixture the digest of the manifest and, for every binary, its ``sha256`` and ``sha256_code``,
  all computed by Python (``Manifest.digest``, ``code_digest``). The Rust tests (``tests/cross_language.rs``) reproduce them
  from the files with no GPU and no Python; ``test_rust_fixtures.py`` checks them against Python on every ``pytest`` run.
* ``generated.rs``: the launchers ``tilefish.hosts.rust.emit`` writes for the first architecture, from the declarations (the
  launchers of another architecture differ only in ``MANIFEST_DIGEST``).

The fixtures are committed; they only change when a binary does (another Triton, ptxas, or kernel), and then the files
of ``crates/tilefish/tests/fixtures`` are replaced together.
"""

import argparse
import json
import pathlib
import shutil
import sys

from host_ops import ops

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
                "cubin": a.entry["cubin"],
                "sha256": a.entry["sha256"],
                "sha256_code": code_digest(a.cubin),
            }
            for a in sorted(manifest.artifacts, key=lambda a: (a.stem, a.arch))
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--archs", default="103,120", help="compute capabilities to build for (default: 103,120)")
    args = parser.parse_args(argv)
    declared = list(ops().values())
    summary = {}
    for position, cap in enumerate(int(c) for c in args.archs.split(",")):
        directory = args.output / f"sm_{cap}"
        if directory.exists():
            shutil.rmtree(directory)
        manifest = runtime.export(declared, directory, arch=cap)
        if position == 0:
            rust.emit(manifest, args.output / "generated.rs", ops=declared)  # the manifest export returned is pinned
        summary[f"sm_{cap}"] = expected(manifest)
        print(f"sm_{cap}: {len(manifest.artifacts)} binaries, manifest digest {manifest.digest}", file=sys.stderr)
    (args.output / "expected.json").write_text(json.dumps({"fixtures": summary}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
