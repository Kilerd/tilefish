"""The byte-identity gate between ``tilefish.run(..., mode="replay")`` and the Rust host, on a GPU. Not collected: the name
has no ``test_`` prefix.

The claim is that the Rust host runs the exact machine code Python's ``replay`` runs, so the same inputs give the same
bytes. The gate has three steps, on a machine with a GPU, the ``gpu`` extra and a Rust toolchain::

    python rust_gate.py prepare WORK                  # 1. Python: export (or take --manifest DIR), emit, run replay
    TILEFISH_GATE_GENERATED=WORK/generated.rs TILEFISH_GATE_DIR=WORK \\
        cargo test -p tilefish --features gpu-gate --test gpu_gate -- --test-threads=1 --nocapture
    python rust_gate.py compare WORK                  # 3. Python: compare the bytes, tensor by tensor

``prepare`` exports every operator of ``host_ops.ops()`` (or reads ``--manifest``, for example the committed fixture),
writes the generated launchers to ``WORK/generated.rs``, and for each case (the five goldens, a bucketed row kernel over each
of its buckets, the hand-written rmsnorm and exact gelu and an in-place axpy, over several sizes and seeds) generates
inputs, writes them as raw bytes, runs ``replay`` *with a pin* and writes what it left in every tensor, outputs and
arguments, so that a tensor that is updated in place is compared too. ``compare`` then requires the files the Rust test
wrote to be equal, byte for byte, to those.
"""

import argparse
import json
import pathlib
import shutil
import sys

import torch
from gpu_util import randn
from host_ops import ops

import tilefish
from tilefish import runtime
from tilefish.aot import Manifest
from tilefish.hosts import rust

BF16, F32 = torch.bfloat16, torch.float32
SEEDS = (0, 1, 2)
ROW_SIZES = (1, 7, 64, 129)
FLAT_SIZES = (16, 1040, 65552, 1 << 20)


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
]

DTYPES = {torch.bfloat16: "bf16", torch.float32: "f32"}


def raw(t):
    return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def abi_of(op):
    """``(tensors, inputs, outputs)``: the tensor parameters of the launched kernel in ABI order with their access, the
    names of the arguments of a call (``In`` and ``InOut``) and of the results of a call, in order."""
    target = runtime._target(op)
    extern = target.launch_extern()
    names = runtime._extern_names(target, extern)
    tensors = [(p.name, p.access) for p in extern.signature.params if p.kind == "tensor"]
    if target.kernel is None:
        inputs = [n for n, a in tensors if a != "out"]
        outputs = [n for n, a in tensors if a == "out"]
    else:
        user = len(target.kernel.signature.params)
        returned = [p.name for p in extern.signature.params[user:] if p.kind == "tensor" and p.access == "out"]
        outputs = returned + [names[p.name] for p in target.kernel.signature.tensor_outputs]
        inputs = [names[p.name] for p in target.kernel.signature.tensor_inputs]
    return tensors, inputs, outputs


def prepare(work, manifest_dir=None):
    work = pathlib.Path(work).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    declared = ops()
    if manifest_dir is None:
        manifest = runtime.export(list(declared.values()), work / "manifest")
        manifest_ref = "manifest"
    else:
        manifest = Manifest.load(manifest_dir)
        manifest_ref = str(pathlib.Path(manifest_dir).resolve())
    digest = manifest.digest
    rust.emit(manifest, work / "generated.rs", ops=list(declared.values()))
    cap = torch.cuda.get_device_capability()
    arch = f"sm_{cap[0] * 10 + cap[1]}"
    cases = []
    for key, stem, make, sizes in CASES:
        op = declared[key]
        tensors, inputs, outputs = abi_of(op)
        for size in sizes:
            for seed in SEEDS:
                case_id = f"{stem}-{size}-seed{seed}"
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
    work = pathlib.Path(work).resolve()
    doc = json.loads((work / "cases.json").read_text(encoding="utf-8"))
    passed, failed, tensors, total = 0, [], 0, 0
    for case in doc["cases"]:
        bad = []
        for t in case["tensors"]:
            expect = (work / t["expect"]).read_bytes()
            path = work / "rust" / case["id"] / f"{t['name']}.bin"
            got = path.read_bytes() if path.exists() else None
            tensors += 1
            total += len(expect)
            if got != expect:
                bad.append(f"{t['name']} ({'missing' if got is None else sum(a != b for a, b in zip(got, expect))} bytes differ)")
            elif t["role"] == "out" and not any(expect):
                bad.append(f"{t['name']} is all zeros: the kernel wrote nothing")
        if bad:
            failed.append((case["id"], bad))
        else:
            passed += 1
    for case_id, bad in failed:
        print(f"FAIL {case_id}: {'; '.join(bad)}")
    print(f"rust vs replay: {passed} of {len(doc['cases'])} cases byte-identical ({tensors} tensors, {total} bytes compared)")
    return 0 if not failed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("work", type=pathlib.Path)
    p.add_argument("--manifest", type=pathlib.Path, help="use this manifest directory instead of exporting one")
    c = sub.add_parser("compare")
    c.add_argument("work", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.work, args.manifest)
        return 0
    return compare(args.work)


if __name__ == "__main__":
    sys.exit(main())
