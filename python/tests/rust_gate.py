"""The byte-identity gate between ``tilefish.run(..., mode="replay")`` and the Rust host, on a GPU. Not collected: the name
has no ``test_`` prefix.

The claim is that the Rust host runs the exact machine code Python's ``replay`` runs, so the same inputs give the same
bytes. The gate has three steps, on a machine with a GPU, the ``gpu`` extra and a Rust toolchain::

    python rust_gate.py prepare WORK                  # 1. Python: export (or take --manifest DIR), emit, run replay
    TILEFISH_GATE_GENERATED=WORK/generated.rs TILEFISH_GATE_DIR=WORK \\
        cargo test -p tilefish --features gpu-gate --test gpu_gate -- --test-threads=1 --nocapture
    python rust_gate.py compare WORK                  # 3. Python: compare the bytes, tensor by tensor

``prepare`` exports every operator of ``host_ops.ops()`` (or reads ``--manifest``, for example the committed fixture),
writes the generated launchers to ``WORK/generated.rs`` (with the manifest and the cubins embedded), and for each case (the
five goldens, a bucketed row kernel over each of its buckets, the hand-written rmsnorm and exact gelu, an in-place axpy and a
``tl.dot`` matrix product with more than 48 KiB of dynamic shared memory, over several sizes and seeds) generates
inputs, writes them as raw bytes, runs ``replay`` *with a pin* and writes what it left in every tensor, outputs and
arguments, so that a tensor that is updated in place is compared too. ``compare`` then requires the files the Rust test
wrote to be equal, byte for byte, to those.

**Manifest schema 3.** Python now writes and reads schema 3 only (``docs/manifest-schema-3.md``), and the crate 0.0.1 reads schema 2: until a
crate that reads schema 3 exists, ``prepare`` writes manifests the Rust test cannot load (and ``--manifest`` of the committed schema 2 fixtures is
refused by Python). The Rust lane that implements schema 3 regenerates the fixtures and runs this gate again.
"""

import argparse
import json
import pathlib
import shutil
import sys

import torch
from gpu_util import randn
from host_ops import abi_of, ops

import tilefish
from tilefish import runtime
from tilefish.aot import Manifest
from tilefish.hosts import rust

BF16, F16, F32 = torch.bfloat16, torch.float16, torch.float32
SEEDS = (0, 1, 2)
ROW_SIZES = (1, 7, 64, 129)
FLAT_SIZES = (16, 1040, 65552, 1 << 20)
MM_SIZES = ((128, 128, 64), (256, 256, 256), (384, 128, 192), (128, 512, 128))  # (M, N, K): M and N multiples of 128, K of 64


def args_softmax(rows, seed):
    return (randn((rows, 1000), F32, seed, scale=3.0),)


def args_layernorm(rows, seed):
    return (randn((rows, 768), BF16, seed), randn((768,), BF16, seed + 100), randn((768,), BF16, seed + 200, scale=0.1))


def args_residual_add_rmsnorm(rows, seed):
    return (randn((rows, 2560), BF16, seed), randn((rows, 2560), F32, seed + 50), randn((2560,), BF16, seed + 100))


def args_gelu(n, seed):
    return (randn((n,), BF16, seed, scale=3.0),)


def args_swiglu(rows, seed):
    return (randn((rows, 3072), BF16, seed, scale=2.0), randn((rows, 3072), BF16, seed + 100))


def args_bucketed(d):
    return lambda rows, seed: (randn((rows, d), BF16, seed), randn((d,), BF16, seed + 100))


def args_rmsnorm(rows, seed):
    return (randn((rows, 2048), BF16, seed), randn((2048,), BF16, seed + 100))


def args_axpy(n, seed):
    return (randn((n,), F32, seed), randn((n,), F32, seed + 100))


def args_mm(size, seed):
    m, n, k = size
    return (randn((m, k), F16, seed, scale=0.5), randn((k, n), F16, seed + 100, scale=0.5))


# (the operator in host_ops.ops(), the stem of its binary, the inputs, the sizes)
CASES = [
    ("softmax", "softmax", args_softmax, ROW_SIZES),
    ("layernorm", "layernorm", args_layernorm, ROW_SIZES),
    ("residual_add_rmsnorm", "residual_add_rmsnorm", args_residual_add_rmsnorm, (1, 3, 33)),
    ("gelu_exact", "gelu_exact", args_gelu, FLAT_SIZES),
    ("swiglu", "swiglu", args_swiglu, (1, 3, 33, 130)),
    *[("bucketed_rms", f"bucketed_rms.Db{d}", args_bucketed(d), (1, 5, 40)) for d in (96, 512, 1000)],
    ("rmsnorm", "rmsnorm", args_rmsnorm, (1, 5, 64)),
    ("gelu_erf", "gelu_erf", args_gelu, FLAT_SIZES),
    ("axpy", "axpy", args_axpy, FLAT_SIZES),
    ("mm", "mm", args_mm, MM_SIZES),  # a tl.dot kernel with more than 48 KiB of dynamic shared memory and a 2-D grid
]

DTYPES = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}


def raw(t):
    return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def expected_pin(manifest_dir):
    """The digest ``expected.json`` (next to the fixture directories, which the Rust tests check as well) has for a fixture directory."""
    expected = pathlib.Path(manifest_dir).resolve().parent / "expected.json"
    try:
        return json.loads(expected.read_text())["fixtures"][pathlib.Path(manifest_dir).resolve().name]["manifest_digest"]
    except (OSError, KeyError, ValueError):
        sys.exit(f"--manifest needs --pin <the manifest's digest> ({expected} has none for {manifest_dir})")


def prepare(work, manifest_dir=None, pin=None):
    work = pathlib.Path(work).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    declared = ops()
    if manifest_dir is None:
        manifest = runtime.export(list(declared.values()), work / "manifest")
        manifest_ref = "manifest"
    else:
        manifest = Manifest.load(manifest_dir, pin=pin or expected_pin(manifest_dir))  # the emitter and the replay both take a pinned manifest
        manifest_ref = str(pathlib.Path(manifest_dir).resolve())
    digest = manifest.digest
    # embedded: the generated module also carries the manifest and the cubins (include_bytes!), which the Rust test runs too
    rust.emit(manifest, work / "generated.rs", ops=list(declared.values()), embed=True)
    cap = torch.cuda.get_device_capability()
    arch = f"sm_{cap[0] * 10 + cap[1]}"
    cases = []
    for key, stem, make, sizes in CASES:
        op = declared[key]
        tensors, inputs, outputs = abi_of(op)
        for size in sizes:
            for seed in SEEDS:
                case_id = f"{stem}-{'x'.join(map(str, size)) if isinstance(size, tuple) else size}-seed{seed}"
                (work / "cases" / case_id).mkdir(parents=True)
                args = make(size, seed)
                clones = [a.clone() for a in args]
                result = tilefish.run(op, *clones, mode="replay", manifest=manifest_dir or work / "manifest", pin=digest)
                results = [] if result is None else (list(result) if isinstance(result, tuple) else [result])
                assert len(results) == len(outputs), (case_id, len(results), outputs)
                post = {**dict(zip(inputs, clones)), **dict(zip(outputs, results))}
                entries = []
                for name, access in tensors:
                    t = post[name]
                    entry = {
                        "name": name,
                        "role": access,
                        "dtype": DTYPES[t.dtype],
                        "shape": list(t.shape),
                        "nbytes": t.numel() * t.element_size(),
                        "expect": f"cases/{case_id}/{name}.expect.bin",
                    }
                    (work / entry["expect"]).write_bytes(raw(t))
                    if access != "out":
                        entry["input"] = f"cases/{case_id}/{name}.in.bin"
                        (work / entry["input"]).write_bytes(raw(args[inputs.index(name)]))
                    entries.append(entry)
                cases.append({"id": case_id, "launcher": rust._camel(stem), "tensors": entries})
    doc = {"arch": arch, "digest": digest, "manifest": manifest_ref, "cases": cases}
    (work / "cases.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"prepared {len(cases)} cases for {arch}, manifest digest {digest}", file=sys.stderr)


def compare(work):
    """Every file the Rust test wrote, from the manifest directory and from the embedded manifest, against what replay left."""
    work = pathlib.Path(work).resolve()
    doc = json.loads((work / "cases.json").read_text(encoding="utf-8"))
    failed, tensors, total = [], 0, 0
    passed = {}
    for subdir in ("rust", "rust_embedded"):
        passed[subdir] = 0
        for case in doc["cases"]:
            bad = []
            for t in case["tensors"]:
                expect = (work / t["expect"]).read_bytes()
                path = work / subdir / case["id"] / f"{t['name']}.bin"
                got = path.read_bytes() if path.exists() else None
                tensors += 1
                total += len(expect)
                if got != expect:
                    bad.append(f"{t['name']} ({'missing' if got is None else sum(a != b for a, b in zip(got, expect))} bytes differ)")
                elif t["role"] == "out" and not any(expect):
                    bad.append(f"{t['name']} is all zeros: the kernel wrote nothing")
            if bad:
                failed.append((f"{subdir}/{case['id']}", bad))
            else:
                passed[subdir] += 1
    for case_id, bad in failed:
        print(f"FAIL {case_id}: {'; '.join(bad)}")
    n = len(doc["cases"])
    print(
        f"rust vs replay: {passed['rust']} of {n} cases byte-identical with the manifest directory, "
        f"{passed['rust_embedded']} of {n} with the embedded manifest ({tensors} tensors, {total} bytes compared)"
    )
    return 0 if not failed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("work", type=pathlib.Path)
    p.add_argument("--manifest", type=pathlib.Path, help="use this manifest directory instead of exporting one")
    p.add_argument("--pin", help="the digest of --manifest (default: the one expected.json next to a fixture directory has)")
    c = sub.add_parser("compare")
    c.add_argument("work", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.work, args.manifest, args.pin)
        return 0
    return compare(args.work)


if __name__ == "__main__":
    sys.exit(main())
