"""A pinned GEMM whose algorithm needs a workspace, for the Rust gate (not collected: the name has no ``test_`` prefix).

The algorithm the heuristic ranks first for the shapes of the gate needs no workspace on the devices it was run on, so the gate's GEMMs never exercise
the workspace of a launch. This writes a manifest (``WORK/workspace_case/manifest``) whose library entry pins the first algorithm among the heuristic's
results that *does* need one, under a workspace limit of 64 MiB, the inputs and what Python's ``replay`` computes with it (the workspace taken from
torch's allocator), for ``gpu_gate.rs`` to run through ``Manifest::gemm`` with the caller's workspace, and to capture in a CUDA graph::

    python rust_workspace_case.py WORK

Run on a machine with a GPU and the ``gpu`` extra, from ``python/tests``. ``rust_gate.py prepare WORK`` runs it at its end (prepare empties WORK first).
"""

import ctypes
import json
import pathlib
import shutil
import sys

import torch
from gpu_util import randn

import tilefish
from tilefish import Dim, Tensor, _cublaslt, _library, bf16, f32
from tilefish._build import write_manifest

SHAPES = [(64, 3072, 1024), (64, 128, 16384), (32, 256, 32768), (64, 256, 8192), (16, 128, 32768), (128, 128, 32768)]  # (M, N, K): the first one the heuristic gives a workspace for
LIMIT = 64 << 20


def candidates(m, n, k, bias, limit, count=16):
    """The heuristic's results for the problem under a workspace limit: ``(the 64 bytes as hex, the workspace it needs)``, best first."""
    lib = _cublaslt._lib()
    index = torch.cuda.current_device()
    handle = _cublaslt._handle(lib, index)
    with _cublaslt._on_device(index), _cublaslt._Problem(lib, m, n, k, bias, _cublaslt.HEURISTIC_BIAS_POINTER, limit) as p:
        results = (_cublaslt._HeuristicResult * count)()
        returned = ctypes.c_int(0)
        lib.call("cublasLtMatmulAlgoGetHeuristic", handle, p.desc, p.a, p.b, p.d, p.d, p.preference, count, results, ctypes.byref(returned))
        return [(_cublaslt.algo_to_hex(bytes(r.algo)), int(r.workspaceSize)) for r in list(results)[: returned.value] if r.state == 0]


def main(work):
    work = pathlib.Path(work).resolve()
    out = work / "workspace_case"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    pick = None
    for M, N, K in SHAPES:
        found = candidates(M, N, K, True, LIMIT)
        print(f"heuristic results for m={M} n={N} k={K} (workspace bytes):", [w for _, w in found], file=sys.stderr)
        pick = next(((a, w) for a, w in found if w > 0), None)
        if pick is not None:
            break
    if pick is None:
        sys.exit(f"no algorithm of the heuristic needs a workspace for the shapes {SHAPES}: choose other sizes")
    algo, needs = pick
    Mb = Dim("M", buckets=[M])

    @tilefish.library("gemm_ws", acc=f32, impl="cublaslt.gemm_bias", workspace_bytes=LIMIT)
    def gemm_ws(a: Tensor[Mb, K, bf16], b: Tensor[N, K, bf16], bias: Tensor[N, bf16]) -> Tensor[Mb, N, bf16]: ...

    (spec,) = _library.library_specs(gemm_ws)
    cap = torch.cuda.get_device_capability()
    built = _library.BuiltLibrary(spec, f"sm_{cap[0] * 10 + cap[1]}", algo, _cublaslt.version())
    write_manifest([], out / "manifest" / "manifest.json", library=[built])
    a = randn((M, K), torch.bfloat16, seed=1)
    b = randn((N, K), torch.bfloat16, seed=2, scale=1 / K**0.5)
    bias = randn((N,), torch.bfloat16, seed=3)
    result = tilefish.run(gemm_ws, a, b, bias, mode="replay", manifest=out / "manifest", trust_directory=True)
    for name, t in (("a", a), ("b", b), ("bias", bias), ("expect", result)):
        (out / f"{name}.bin").write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    case = {"stem": spec.stem, "arch": built.arch, "m": M, "n": N, "k": K, "algo": algo, "workspace_required": needs, "workspace_bytes": LIMIT}
    (out / "case.json").write_text(json.dumps(case, indent=1) + "\n")
    print(f"wrote {out}: '{spec.stem}' pinned with an algorithm that needs {needs} bytes of workspace (limit {LIMIT})", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
