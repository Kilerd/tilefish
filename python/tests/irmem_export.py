"""The peak memory and the time of exporting the extern of :mod:`irmem_kernels` (132 bucket entries, 3 configurations), on a machine with
Triton (not collected: the name has no ``test_`` prefix)::

    python irmem_export.py OUT [--steps 64] [--arch sm_103] [--trace]

prints one line of JSON: the entries, the distinct IR digests, the seconds of the export and the peak resident memory of the process
(``ru_maxrss``) before and after it. ``TILEFISH_CACHE_DIR`` chooses the build cache, as for any export (an empty directory: a cold cache).
"""

import argparse
import json
import pathlib
import resource
import sys
import time


def _peak_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024  # bytes on macOS, KiB on Linux


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--arch", default="sm_103")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import irmem_kernels

    import tilefish

    op = irmem_kernels.chain_op(args.steps)
    before = _peak_mib()
    start = time.perf_counter()
    manifest = tilefish.runtime.export(op, args.out, arch=args.arch, trace=args.trace)
    seconds = time.perf_counter() - start
    doc = json.loads(pathlib.Path(manifest.path).read_text())
    digests = {e["source_digest"] for e in doc["kernels"]}
    print(
        json.dumps(
            {
                "entries": len(doc["kernels"]),
                "ir_digests": len(digests),
                "seconds": round(seconds, 2),
                "peak_mib_before": round(before, 1),
                "peak_mib": round(_peak_mib(), 1),
                "manifest_digest": manifest.digest,
                "trace": args.trace,
                "steps": args.steps,
            }
        )
    )


if __name__ == "__main__":
    main()
