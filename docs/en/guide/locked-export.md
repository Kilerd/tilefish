# Locked exports: an export without a GPU

An export learns some facts from the device it runs on, and everything else is a function of the sources, the declarations and the toolchain. A
**lock** records those facts once, on a GPU of the target device; a **locked export** then writes the same files byte for byte on a machine with no
GPU, no CUDA driver and no torch, such as a CI runner or an image build, and fails when a file is not the lock's.

```python
# on a GPU of the target device, into an empty directory; once per device, and again whenever an implementation changes
tilefish.runtime.export([encoder, decoder], "build/sm_103")
tilefish.runtime.write_lock("build/sm_103", "locks/sm_103.lock.json")          # commit the lock

# in CI and the image build, on a machine without a GPU
s = tilefish.runtime.export([encoder, decoder], "plans/sm_103", lock="locks/sm_103.lock.json")
s.digest   # the lock's digests.set: the pin the hosts load the plans with
```

`write_lock(directory, path)` reads files only (the manifest, the plans, the fusion reports and the sha256 of every file); it does not export.
`export(..., lock=path)` takes the same operators and options as the export it locks (`buckets=`, `reuse=`, `verify=`), and the architecture from the
lock (`arch=`, when given, must be the lock's). A plan set, one graph, and kernels and externs alone are all locked the same way. Both raise
`tilefish.TilefishLockError`, which names the first difference and what to do.

## What a device contributes

| fact | made on the device by | where an export writes it | where the lock keeps it |
|---|---|---|---|
| the architecture | the device | every entry's `arch`, each plan's `arch` | `device.arch` |
| the SM count | the device | each library entry's `sm_count` and `declaration_digest` | `device.sm_count` (`null` without library entries) |
| a cuBLASLt algorithm (`cublaslt.gemm`, `gemm_bias`, `matmul`) | the heuristic, or an imported table, checked by `cublasLtMatmulAlgoCheck` | the entry's `algo` | `library.<stem>.algo` |
| the verification of a `cublaslt.matmul` pin | the pin, run on special operands | the entry's `contract` and `verification` | `library.<stem>.contract`, `.verification` |
| a cuDNN pin (an engine and its knobs, or a legacy algorithm) | cuDNN's heuristic, or an imported table, checked | the entry's `pin` | `library.<stem>.pin` |
| a canary (`cublaslt.matmul`, `cudnn.conv_fwd`) | the pin, run on operands made of a seed | the entry's `canary` | `library.<stem>.canary` |
| the cuBLASLt and cuDNN versions | the libraries | each entry's `cublaslt_version`, `cudnn_version` | `toolchain.cublaslt`, `toolchain.cudnn` |
| the verification of the fused regions | each region run against its ops one by one | `<graph>.fusion.json`, each region's `verified` | `fusion.<graph>` |

Nothing else needs a device. Triton compiles ahead of time for an architecture (Triton and Gluon kernels, generated kernels and fused regions), `ptxas`
assembles a prebuilt PTX, a prebuilt cubin is used as it is; the plans (steps, buffers, the arena, the workspace: the declarations' `workspace_bytes` and
the global scratch Triton reports), the warps of each bucket (`by_bucket=`), the TF32 guard and the SASS assertions (`nvdisasm`, the `tilefish[sass]` extra)
are all made from the declarations and the binaries. Measured: the plan set of `python/tests/locked_graphs.py` (a generated kernel, a Triton extern, a Gluon
extern, a prebuilt PTX and cubin, a pinned cuBLASLt GEMM, a `cublaslt.matmul` with an epilogue, a cuDNN convolution, two fused regions, a kernel whose
global scratch is the plan's workspace, two buckets per graph, three canaries), exported on a B300 MIG slice (sm_103, 30 SMs, Python 3.11) and locked
there, then exported from the lock on a CPU-only machine (Python 3.12, no torch, no CUDA driver) with the same Triton 3.5.1 and ptxas 12.9.86: 14 files, 0
differences, in 1.3 s from a cold build cache.

## The lock file (`tilefish.lock.v1`)

Canonical JSON (sorted keys, one space of indent, a newline at the end), so a re-lock that changes nothing changes no byte, and its diff says what
changed. Every field is required:

| field | what |
|---|---|
| `schema` | `"tilefish.lock.v1"` |
| `tilefish` | the version that wrote the export (the manifest and the plans name it, so a locked export needs the same) |
| `device` | `{"arch": "sm_103", "sm_count": 30}`: the device the export ran on (`sm_count` is `null` when no library entry recorded it) |
| `toolchain` | the versions the export recorded, each only when it has binaries or entries of it: `triton` and `ptxas` (the manifest's `toolchain` texts), `cublaslt` and `cudnn` (integers) |
| `library` | per stem: `impl`, `declaration_digest` (the entry's: the declaration and the SM count), `workspace_bytes`, and the pin, `algo` (cuBLASLt, 128 hex digits) or `pin` (cuDNN); a `cublaslt.matmul` also has `contract` and `verification`; a canary is `canary` (`{"seed", "out_digest"}` or `{"seed", "y_digest"}`) |
| `fusion` | per graph, the regions an export verified: `[{"region": the index of the region in the fusion report, "op": its kernel, "verified": what the report records}]` |
| `kernels` | per stem: `cubin` (the file), `sha256` and `sha256_code` |
| `digests` | `manifest` (the manifest's digest), `plans` (each plan's digest, by graph) and, for a plan set, `set`: the pins the hosts take |
| `files` | the sha256 of every file of the directory, by its path (`trace/` is left out, as it is outside every digest) |

`write_lock` refuses a directory that is not one export of one device: entries of two architectures, two toolchains or two SM counts, a file an export
does not write, entries or a fusion report no plan of the directory has (left by an export of another graph into it), a fusion report whose regions were
not verified (an export made off the device warns and records `skipped`; export on the device, or with `verify=False`), and a CuTe DSL kernel (below).

## What a locked export checks, in order

1. The lock reads strictly (no duplicate key, every field typed); `arch=` is the lock's; the tilefish, Triton and ptxas versions are the lock's (the
   ptxas Triton compiles with, `TRITON_PTXAS_PATH`, or for an export of prebuilt PTX only, `TILEFISH_PTXAS_PATH`).
2. Each library step takes the lock's record of its stem, whose `declaration_digest` must be the declaration's on a device of the lock's SM count (an
   operator, option or bucket that changed is refused); a `cublaslt.matmul` declared `contract="measured-exact"` is refused a `binary-defined` pin,
   and one with a canary (the default) needs the lock's canary, as an export does; the entry it makes must pass every rule a manifest reader has
   (section 9 of `docs/manifest-schema-3.md`). **No heuristic is asked, no library is loaded, nothing runs on a device, with a device or without one.**
3. Each fused region the export would verify (the graph's `verify`, or `verify=`) takes the lock's record of it.
4. Each binary is built and held to `kernels.<stem>` (its toolchain, then its `sha256`, saying whether the machine code differs) before anything is written.
5. The files are written as an export writes them, and every file of the directory is held to `files`: a missing file, a file of other bytes (the cubins
   first, then the manifest, then the rest) and a file the lock does not have (export into an empty directory) are refused. The files a refusal at this
   step leaves are there to be compared with the export on the GPU; do not ship them.
6. The directory, locked again, gives the lock itself: the digests a host is told to pin, the binaries' `sha256_code`, the pins and the regions'
   records are what the files say, so a lock edited by hand where no file shows it is refused too.

A stem, a region or a binary the lock does not have is refused with **re-lock on a GPU of the target device**: there is no fallback to a heuristic.

## What is checked where

| check | export on the GPU | locked export (no GPU) | load on the GPU (Python, Rust, C++) |
|---|---|---|---|
| each binary is its source's, for its architecture | built | built, and held to the lock's `sha256` | each cubin against the manifest's `sha256` and `sha256_code`; the manifest and the plans against the pin |
| the pin is what the declaration asks for | the heuristic under the declaration's posture | the lock's record of this declaration (`declaration_digest`) | Python's replay compares the entry with the declaration (an imported table's algorithm too); Rust and C++ hold the plan's pin |
| the algorithm is valid: `cublasLtMatmulAlgoCheck`, the re-encoding, the reduction scheme and its opt-in, no TF32 or emulated f32, the workspace | checked | not checked (no library) | checked at every load (Rust and C++: `Manifest::gemm`; Python: when a plan is prepared and before every run) |
| the cuBLASLt and cuDNN versions, the SM count, the architecture | recorded | the lock's | another one is refused |
| a cuDNN pin is deterministic, of allowed numerical notes, its knobs in range, its plan finalizes; no `NVIDIA_TF32_OVERRIDE` | checked | not checked | checked at every load |
| the canary | computed | the lock's | run at the first load, another digest refused |
| a `cublaslt.matmul` pin against its formula (contract (ii)) | measured | the lock's record | the record (Python refuses a `binary-defined` pin a declaration says `measured-exact` of); the canary ties the bytes to the run that measured it |
| fused regions against their ops one by one | measured | the lock's record | not repeated |

So nothing an export checks goes unchecked when it runs without a GPU: every check a load makes is made as before, and what only an export measures
(the heuristic's choice, the verification of a matmul pin and of the fused regions) was measured on the GPU that made the lock, of the binaries that
the locked export is held to byte for byte. A lock is a trust anchor like a pin: a lock edited by hand, with the pins and the hashes changed together,
writes what it says. Review a lock's diff as a change of the artefacts it describes, and commit it next to the code.

## When to re-lock

When an implementation changes (a kernel, a declaration, an option, a bucket, the operators of the export), when tilefish, Triton or ptxas is upgraded,
when cuBLASLt or cuDNN is (a pin is of a library version: a host refuses another), and for each new target device. A locked export that fails says what
differs; re-lock on a GPU of that device and review the diff.

## Limitations

* **CuTe DSL kernels** are compiled by the DSL on a GPU, and their launch (grid, cluster, the parameters of the compiled host function) is captured on
  it (`docs/cute.md`): a locked export refuses them, and so does `write_lock`. Export a set with a CuTe DSL kernel on a GPU.
* **The same toolchain on both machines**: the tilefish, Triton and ptxas versions are checked; anything else that changes a binary (a Triton setting,
  `TILEFISH_DEBUG_PATHS`) changes its `sha256`, and the locked export refuses it. The Python version does not reach a binary (3.11 and 3.12 above).
* **One lock per device.** The architecture is checked; two devices of one architecture with other SM counts (a MIG slice, the whole GPU) are not told
  apart without a GPU: a locked export of the wrong lock succeeds, and the host refuses its library entries at the load ("pinned on a device with 30
  SMs, but this device has 148").
* **One export per directory**: a lock holds every file of its directory, so manifests of several architectures merged into one directory are not locked.
* A lock is not an imported table (`algos=`, `AlgoTable`, `CudnnTable`): those are part of a declaration and of its digest, and an entry made from one
  says `"algo_source": "imported"`, so they cannot reproduce an export that asked the heuristic. A lock records what the export pinned, whichever way
  it pinned it, `cublaslt.matmul` included (which takes no `algos=`).

The gate runs the hosts on locked artefacts: `python/tests/rust_gate.py prepare WORK --locked DIR --lock FILE` takes the plan set of
`locked_graphs.py` from `DIR` (a locked export, made on a machine without a GPU, held to the lock and replayed under its set digest) instead of
exporting it, so Python's replay, the Rust host and the C++ host load it, run its canaries and its cases, and `compare` holds their bytes to Python's.
