# Changelog

Versions are the Python package `tilefish` on PyPI and the Rust crate `tilefish` on crates.io (and, from 0.0.8, the C ABI crate
`tilefish-capi`, which is built from this repository). The manifest is schema 3 and the
plan is `tilefish.plan.v1` in both releases below; `docs/manifest-schema-3.md` and `docs/plan-schema-1.md` specify every field, and
each section names the release that added it.

## Unreleased

### Documentation

- **The README is an overview** (about 200 lines): what tilefish is, an example, how it works, the install, a quick start for each host, the
  features at a glance and the status. Its reference sections moved, with their text unchanged, to `docs/guide/`: `build.md` (how a build works),
  `operators.md` (views, device scalars, prebuilt binaries, pinned GEMMs and convolutions), `lowering.md` (masks, indices, gathers and copies),
  `plans.md` (launch plans), `fusion.md` (fusion, epilogue hooks, cuBLASLt epilogues), `rewrites.md` (rewrites that know about rounding),
  `sass.md`, `rust.md` and `cpp.md`. A README section that an entry below names is the page of that title; `docs/README.md` lists them.

## tilefish 0.0.13, crate 0.0.11, tilefish-capi 0.0.2 (2026-11-17)

The last release that adds features ahead of a need (`docs/scope.md`). Manifest schema 3 gains one form of kernel entry, a CuTe DSL kernel
(`binary.source` `"cute"` and the parameter kinds `tensormap` and `bytes`, section 4.9): additive, so every manifest, plan and pin of 0.0.12 is read
and written as it was, and a reader of 0.0.12 or crate 0.0.10 refuses such an entry. Plan schema 1 and the C ABI (1.1) are unchanged. An export
without a CuTe DSL kernel writes the manifests and plans of 0.0.12, but for a graph that declares `epilogues="accumulator"` around a hook of a dual
accumulator: its fused pair no longer stores the rounded result as well (below); its outputs are the same bytes.

The crate reads and launches CuTe DSL entries (0.0.11). `tilefish-capi`'s sources are unchanged, so it stays 0.0.2 (ABI 1.1), built from the release
commit with cargo or CMake; built from this release it launches CuTe DSL entries through the crate (`tilefish_version()` says `"0.0.11"`).

### Scope

- **`docs/scope.md`** says what tilefish is (types, the four declarations, build once and pin, thin hosts, numerics written down) and what it is not
  (a competitor of the libraries on kernel speed, a general optimising compiler, a benchmark suite). 0.0.13 is the last release that adds features
  ahead of a need: from now on a feature is added when a real model migration needs it. The README's opening is shorter and points to it and to the
  documents' index, `docs/README.md`.

### Example: a persistent GEMM with epilogue hooks (`python/examples/persistent_gemm.py`)

- **An example kernel, not a supported feature** (`docs/scope.md`): it shows E1's epilogue hooks in a persistent, warp-specialised GEMM. `c = a @ b`
  with bf16 or f16 operands and an f32 accumulator, in the layouts `TN`, `NN`, `TT` and `NT`, with the hooks `ACC` and `EPILOGUE` and the dual
  accumulator of a gated activation (`split=2`): in Gluon on sm_100 and sm_103 (a TMA producer warp, a `tcgen05` issuer warp, and epilogue partitions
  that drain one tensor-memory buffer while the next tile accumulates), in Triton on sm_120 (`mma.sync`, asserted in the SASS).
  `persistent_gemm.gemm(arch, M=..., n=..., k=..., layout=..., split=...)` declares it with a configuration per bucket of `M`. The whole `k` of a tile
  is one program's, so no configuration changes the bytes: on sm_103 they are the cuBLASLt pin's wherever the pin does not split `k`. What it measured
  as RFC 0003's experiment X8 is recorded there ("X8: results").
- **Tests**: `tests/test_gpu_persistent_gemm.py` (its bytes against the pin at every bucket, every configuration the same bytes, the Triton kernel the
  Gluon one's bytes, every elementwise pattern fused against the unfused plan in both modes at every bucket with special values, a norm refused by the
  Gluon kernel and fused into the Triton one, the fused binary's MMAs, the sm_120 builds, the probes); the Rust and C++ gates run its fused bias +
  GELU and SwiGLU plans (`fused_epi_pgemm_*`).

### CuTe DSL kernels as externs (issue #5, section 3; `docs/cute.md`)

- **`@tilefish.extern(cute=host, cute_args=..., cute_options=...)`** declares a kernel written in CUTLASS's CuTe DSL (`nvidia-cutlass-dsl`): a
  `@cute.jit` host function, or an object whose `__call__` is one (the kernel classes of the CUTLASS examples), and a stub of the operator's tensors of
  static shapes (numbers or bucketed dimensions: one binary per bucket). `tilefish.impl(cute=...)` is one implementation of an operator declared with
  `by_arch=` (the tcgen05 GEMM of sm_100/sm_103 beside the sm_120 one). `cute_args=` makes the host function's arguments from the tensors; tilefish
  appends the stream.
- **The build** compiles it with the DSL for the device's architecture (the DSL's own cubin and PTX, its file cache off) and takes the launch from the
  compiled host function without running the kernel: it calls it three times on a stream captured into a CUDA graph, each time with the tensors at other
  addresses, and reads the kernel node (grid, block, cluster, dynamic shared memory, every parameter's bytes). Every byte the kernel reads (its PTX's
  `ld.param` loads; all of a parameter whose address it takes) must be constant or a tensor's address: a parameter is a `pointer`, a `tensormap` (a TMA
  descriptor the DSL encoded for a static shape: a template with the address in its first 8 bytes) or `bytes`. Padding the kernel does not read (the fast
  divisors of a persistent tile scheduler) and the 64 bytes after a descriptor that the DSL does not write are passed as zeros. Anything else (a dynamic
  shape, a pointer inside a structure, a pointer the host function allocates) is refused with what it is. SASS assertions and traces apply.
- **Manifest schema 3, additively** (section 4.9): a `binary` entry of `binary.source` `"cute"` (`{"source", "cutlass_dsl", "options"}`), its toolchain
  `{"cutlass_dsl", "nvvm", "ptxas"}`, `source_digest_kind` `"tilefish.cute.v1"` (the sha256 of its PTX), a grid of integers, and the ABI kinds `tensormap`
  and `bytes` (on such an entry only). Both readers refuse every malformed form; a reader of 0.0.12 or crate 0.0.10 refuses such an entry. Manifests without one, and
  their digests, are unchanged.
- **Pins.** The declaration digest has the host function, the sha256 of its file, the DSL's version and the options; Python's replay compiles the
  declaration again and compares the PTX (refused if it differs, pinned or not; accepted only pinned where it cannot compile it). The Rust crate and the
  C ABI launch the entry from the manifest alone (`tensormap`: the template with the tensor's address; `bytes`: as written), in a single launch, a plan
  and a captured graph; the Rust and C++ emitters type it by its tensors.
- **Measured on sm_103** (a 30-SM partition, nvidia-cutlass-dsl 4.8.1 with its CUDA 12.9 libraries): tilefish's launch of the examples' tcgen05 GEMMs
  (persistent and not) and elementwise kernels is byte for byte the DSL's own launch; two exports in different processes and directories are the same
  files and digest; `cute_add` is torch's `a + b` byte for byte. The persistent tcgen05 GEMM (2-CTA 256x128, cluster 2x1) is byte for byte the pinned
  cuBLASLt GEMM's output and, at `N = 2048, K = 1024`, 0.71x, 0.92x and 1.17x its speed at `M` = 256, 1024 and 4096 (256x256 tiles: 1.25x at 4096; the
  non-persistent GEMM 0.49x to 0.59x).
- **Tests**: `test_cute_externs.py` (no GPU, no DSL), `test_gpu_cute.py` (the DSL and a CUTLASS checkout's examples in `TILEFISH_CUTLASS_EXAMPLES`), the
  cases of `cute_kernels.py` in the Rust and C++ gates, and the crate's strict-reading and packing tests. The examples are BSD-3-Clause and are not
  copied here; the DSL is NVIDIA software under its own licence, which tilefish neither ships nor depends on.
- **Pin the DSL**: tilefish is tested with `nvidia-cutlass-dsl==4.8.1`; install that version and pin it where you export and replay (the DSL is in
  beta; another version is another declaration, whose binaries a replay refuses). The cubins and PTX an export writes are the DSL's output: shipping
  or redistributing them is the user's responsibility under NVIDIA's licence (`docs/cute.md`, section 6).
- Messages that changed: the kinds an extern takes (`a kernel is one of triton=, gluon=, ptx=, cubin= and cute=`), the kinds of a parameter (`... given,
  null, tensormap or bytes`) and of a spec's `binary` (`'ptx', 'cubin' or 'cute'`).

### Fixes

- **`epilogues="accumulator"` and a hook of a dual accumulator** (`split=2`, a gated activation): `R_acc` replaced the widenings of the split's chunks
  by the chunks of the accumulator and left the split of the rounded result in the program. Nothing read it, but a view is never dead code, so the
  result stayed live and the fused pair stored all of it (`[M, 2F]`) besides the hook's output. `R_acc` now removes a split of the result that only
  the widenings it replaced read, with those widenings; a chunk read in its own type, or returned, keeps it. The outputs are the same bytes (the
  dual GEMM + SwiGLU of the persistent GEMM example, and the gate's `fused_epi_swiglu_acc`, compared with 0.0.12's); the fused entry, which stores
  one chunk now, and the plan digest are new. On a B300 slice that example at 2048 rows takes 89.7 us under `"accumulator"` (124.5 us before), and
  90.2 us under `"exact"`.
- **An epilogue hook on a CuTe DSL extern** (`@tilefish.extern(cute=..., epilogue=...)`) is refused: the DSL compiles the kernel, which has no hook.
  The declaration dropped the hook without a word.

### Tests

- The crate's CuTe DSL fixture (`crates/tilefish/tests/fixtures/cute_sm_103`) is written by `export_rust_fixtures.py` (every run, or `--cute-only`,
  which needs no GPU) from `tests/cute_fixture.py`, with its digest and its cubins' `sha256` in `expected.json` (`"cute"`), which `cute_externs.rs`
  reads. It was written once by hand, its digest a constant of the Rust test, and went stale with each edit of `test_cute_externs.py` (a declaration
  digest has the sha256 of the host function's file); `test_cute_externs.py` now requires the committed files to be what the script writes.

## tilefish 0.0.12, crate 0.0.10, tilefish-capi 0.0.2 (2026-11-17)

Manifest schema 3 gains a library impl, `cublaslt.matmul` (section 9.5): additive, so every manifest, plan and pin of 0.0.11 is read and written as
it was (the entries of `cublaslt.gemm` and `cublaslt.gemm_bias` and their digests are unchanged), and a reader of 0.0.11 or crate 0.0.9 refuses an
entry of the new impl. The C ABI is 1.1 (additive); plan schema 1 is unchanged. A fused pair of an extern and a region is a generated `triton` entry,
a kernel step like any. Nothing changes bytes without a declaration: a graph planned as before (`epilogues="exact"`, the default) has the plan it had,
and an export that calls no extern with an epilogue hook writes the manifest and plans of 0.0.11 (its fusion report gains `library_epilogues` where the
body widens a cuBLASLt GEMM's result).

The crate reads and runs `cublaslt.matmul` (0.0.10). `tilefish-capi` is 0.0.2, ABI 1.1 (`tilefish_gemm_epilogue`, additive: a program built against
ABI 1.0 runs as it did), built from the release commit with cargo or CMake; it is not on crates.io.


### `epilogues=`: three modes, per graph or per scope (RFC 0003)

- `tilefish.fusion(epilogues=...)` and the new scope `with tilefish.fused(epilogues=...):` take `"exact"` (the default), `"accumulator"` (milestone
  E1) or `"library"` (milestone E2). A scope's mode wins over the graph's for the ops traced in it (scopes nest; never per call), so one graph may mix
  extern hooks of either contract and cuBLASLt epilogues.
- `"exact"`: epilogue hooks of externs with `value="out"` (contract (i)); nothing changes a byte.
- `"accumulator"`: the rewrite `R_acc` and the `value="acc"` hooks of externs (contract (ii)). A cuBLASLt GEMM is left as it is: `R_acc` of a library
  call is not planned, and the fusion report's `library_epilogues` entry of every such GEMM whose result is widened there says so (`"mode":
  "accumulator"` and a `note`).
- `"library"`: the extern hooks of `"exact"`, and the cuBLASLt rewrite below. A pattern is rewritten only where all its ops (from `y.to(f32)` to the
  last `.to()`, and the widenings of `c` and the bias) are in a `"library"` mode; otherwise the report lists it as matched but not applied, with the modes it is in.

### Epilogue hooks on externs (`docs/rfcs/0003-epilogue-fusion.md`, milestone E1)

- **`tilefish.hook(name, value="out" | "acc", out=..., split=1, block_n=None)`**, given to `@tilefish.extern(..., epilogue=...)` (one hook or two of
  different values): the kernel takes a `tl.constexpr` jit function `<name>` and its arguments `<name>_ARGS`, which the stub does not list, and calls
  `tile = <name>(tile, rows, cols, row_mask, col_mask, <name>_ARGS)` on each output tile where it would store it (`tilefish._epilogue.ABI`). Without a
  fusion the hook is the identity: an extern with hooks is launched through a small generated module (`tilefish_generated.<extern>__hooks`) that calls
  its kernel with the identity, whose source is in the declaration digest. A Triton or a Gluon kernel of one implementation.
- **A region of the graph body that reads the extern's result and fits its tile is fused into the kernel**: one step, `<extern>_epi_<hash>` (the
  extern with the region's code, lowered by `tilefish.lower.lower_hook`, note 22, in its hook; the region's other inputs and side outputs extra
  parameters, keyword-only). What a hook takes: elementwise regions over the result (a bias and an activation, a periodic activation, a residual add,
  masks, casts; a side output stored by the hook), the tiles of a dual accumulator (`split=2`: a gated activation of the gate and up halves), and an
  ordered norm where a tile holds whole rows (`block_n=`, a Triton kernel). Anything else stays a region after the call, and the fusion report, the
  boundaries and the advisor say why (`epilogue-*` reasons: among them a cycle through another region or fused pair, `epilogue-cycle`, and a
  tensor the extern updates in place, `epilogue-inout`). `tilefish.barrier(y)` keeps a result stored and stops its epilogue fusion.
- **Contracts**: `tilefish.fusion(epilogues="exact")`, the default, fuses through `value="out"` hooks, which see the tile the kernel rounded: the
  bytes of the extern followed by the region (contract (i)). `epilogues="accumulator"`, declared on the graph or in a new scope, `with
  tilefish.fused(epilogues="accumulator"):`, is the rewrite `R_acc` on the IR: every widening of such a result is the extern's f32 result, written by
  the extern's accumulator form (`<extern>__acc`) where no hook takes the region, and the region goes into the `value="acc"` hook: the bytes of the
  unfused plan of the rewritten program (contract (ii)), one rounding fewer.
- **Checked at every export**: each fused pair against the extern (or its accumulator form) followed by the region's ops one by one, at every bucket,
  on seeded inputs with special values in every side input and in the extern's operands; a difference refuses the export (`verify=False` opts out).
  `tilefish.verify_fusion` does the same on demand.
- **The fusion report** gains, only for a graph that calls an extern with hooks: per fused region `route: "hook"`, `call`, `value`, `contract`,
  `stored`, `rounding_points_removed` (under `R_acc`); per region a hook did not take, `epilogue_refused`; `hooks` (the externs with hooks the graph
  calls); `fusion.epilogues` when it is not `"exact"`. The trace's regions name the fused pair.
- **The advisor and `boundaries()`**: an extern with hooks is an `opaque` step whose epilogue candidate is actionable (`epilogue-hook-consumer`, or the
  reason its hook did not take the region), with `routes` and the contract of each; an epilogue of a pinned GEMM or convolution points to RFC 0003's
  E2, E3 (with its break-even slowdown) or E6, and to an extern with a hook today. The fixes of `epilogue-library`, `epilogue-extern`,
  `opaque-unknown` and `opaque-compute-epilogue` are reworded.
- **Examples** (`python/examples/epilogue_gemm.py`): a Triton GEMM with both hooks (`tcgen05.mma` on sm_100 and sm_103, `mma.sync` on sm_120, each
  asserted in the SASS; its epilogue on sub-tiles of the columns), its dual-accumulator form, a Gluon `tcgen05` GEMM, and a 1-D convolution as an
  implicit GEMM.
- **Speed** (B300 slice, the example GEMM): bias + GELU after a 2048 x 4096 x 1024 GEMM 1.21 times faster fused than the GEMM and its region after
  it, a gated activation 1.14 times, parity for a periodic activation and a norm at 2048 rows; slower where the epilogue is heavy for the kernel's
  warps (a periodic activation after a convolution: 0.77 times). The hook's arithmetic runs on the extern's warps after its main loop; `tilefish.barrier`
  keeps a region after the call (RFC 0003, E1 notes).

### cuBLASLt epilogues: `cublaslt.matmul` (RFC 0003, milestone E2)

- **A GEMM with a declared epilogue**: `@tilefish.library(..., impl="cublaslt.matmul", epilogue=..., beta=...)` with `epilogue` one of `"none"`,
  `"bias"`, `"relu"`, `"relu_bias"` (cuBLASLt's `DEFAULT`, `BIAS`, `RELU`, `RELU_BIAS`) and `beta=1` to add `c [M, N]`: `out = epilogue(acc [+ c] [+
  bias])` on the f32 accumulator, `c` before the bias (the library's order), rounded once; bf16 or f16 operands and result, the four layouts, both
  postures, the opt-ins and `sm_count_target`. The bytes are the library's under the pin (contract (iii)). An epilogue gets no split-K without the opt-in,
  as a bias does. `algos=` is not taken yet.
- **Every pin is verified when it is made** (`jit` and `export`): two seeded operand sets with special values, compared bit for bit with the formula
  computed in f32 on the pin's own accumulator (its configuration re-encoded with an f32 result) and rounded once. The entry records `contract`
  (`"measured-exact"` or `"binary-defined"`) and `verification` (the formula, where the accumulator came from, the elements that differ and their largest
  distance); `contract="measured-exact"` refuses a pin that is not. Measured on a B300 MIG slice over 14 shapes, bf16 and f16, both postures and seven
  cases (392 pins per version): every pin measured-exact with cuBLASLt 12.8.4; with 13.1.0, 372 of 392, the others the pins of algorithm 21 at
  32 x 1024 x 3072, which adds a bias to `c` in neither order and whose ReLU maps a NaN to 0 (`docs/rfcs/0003-epilogue-fusion.md`, section 7.3).
  On an RTX 5090 (sm_120) both versions answer every epilogue with algorithm 21: 140 of 392 pins measured-exact (`bias`, `c`, half of `bias` with
  `c`), every ReLU binary-defined (a NaN to 0), so `epilogues="library"` refuses a ReLU there.
- **A canary, on by default** (`canary=True`): `export` records the digest of the pin's result on operands made on the device from a seed (the PTX
  module of the cuDNN canaries); a replay runs it once per process and device and refuses another digest (and a first replay inside a capture), as the
  Rust and C++ hosts do at load.

### `tilefish.fusion(epilogues="library")`

- **The arithmetic after a cuBLASLt GEMM as one of its epilogues**, declared per graph or per scope (`tilefish.fused`): `y.to(f32)`, `+ c.to(f32)`, `+ bias.to(f32)`,
  `tilefish.where(h <= 0.0, 0.0, h)` (ReLU, `torch.relu`'s bits) and `.to()` the GEMM's type, after a `cublaslt.gemm` of a 16-bit result (or the ReLU after
  a `gemm_bias`), become one step of a `cublaslt.matmul` (`<gemm>__<epilogue>[_c]`) declared `contract="measured-exact"`: the rounding of `y` is removed
  (contract (ii)), and every pin is verified equal to the region's arithmetic on its accumulator or the build is refused (the pin is chosen for the
  epilogue, RFC 0003 section 3.1.1: its accumulator is the GEMM's where the two pins compute the same sum). Another order, an f32 bias, a value read elsewhere or barriered is left as it is. `mode="reference"` interprets the
  rewritten program. On a B300 slice the rewritten plans take 0 to 35% less time than the GEMM and its region (`README.md`, the table).
- The fusion report gains `library_epilogues`: the rewrites, and why a widened GEMM result was not rewritten; a graph planned `"exact"` lists what the
  mode would do (so a 0.0.11 export of a graph whose body widens a cuBLASLt GEMM's result gets that list in its informative report; no digest has it).
  The report's `fusion` says `epilogues` when it is declared; the regions' names do not depend on the mode.
- **The advisor** reports a region that the mode would make a cuBLASLt epilogue as the blocker `epilogue-library-cublaslt`, of the new phase `E2` (a
  mode to declare), with the epilogue and the contract; one that is not is `epilogue-library` with the reason (such as the order of the adds).

### The Rust crate (`tilefish`) and the C ABI (`tilefish-capi`)

- **`cublaslt.matmul` is read and run by the crate**: `LibraryEntry` gains `epilogue` (`GemmEpilogue::None`, `Bias`, `Relu`, `ReluBias`; `formula`,
  `cublaslt_value`), `beta` (`has_bias()`, `has_c()`), `contract` (`MatmulContract`), `verification` (`MatmulVerification`, `MatmulAccumulator`) and
  `canary` (`MatmulCanary`), each held to the rules of section 10 with Python's words; `LibraryImpl::Matmul`. Python's verdict on 576 vectors of the new
  impl (each form, and each changed in one field) is the crate's (`tests/python_library.rs`). The entries of `cublaslt.gemm` and `cublaslt.gemm_bias` are
  read as before (`GemmEpilogue::None` / `Bias`, `beta` 0), and the fields of a matmul on them are refused by name.
- **`Manifest::gemm` loads a matmul with its epilogue**: every check (`cublasLtMatmulAlgoCheck`) and run sets the epilogue on the matmul descriptor (`BIAS_DATA_TYPE`
  and `BIAS_POINTER` for the ones with a bias), and `beta = 1.0` with the call's `c` as `C` (the layout of `D`) when the entry has `c`. The vetting refuses
  a split-K of every epilogue but `none` without the opt-in, in the words of Python's `reduction_problem`. A call is `a`, `b`, [`bias`], [`c`], `out`, workspace
  (`Gemm::launch`, `launch_raw`, `check`, a launch plan's library step); `Gemm::epilogue`, `beta`, `has_c`, `contract`, `canary` and `implementation` say
  what was loaded.
- **The canary at load**, on by default: an entry with a `canary` runs it once per context before the pin is used, exactly as Python computes it (the PTX of
  `canary.ptx`, `a` tag 0, `b` 1, `bias` 2, `c` 3, a zeroed `out` and workspace, on a non-blocking stream of its own, never the NULL stream), and another
  digest is refused ("the canary of the cuBLASLt matmul '<stem>' (<arch>, algorithm id ..., epilogue ...) gives out with digest ... here, and the manifest
  recorded ..."); `Gemm::check_canary` runs it again. The canary's kernels are one module per context, shared with the cuDNN convolutions (`src/canary.rs`;
  `canary_bytes` and `canary_digest` are re-exported from there, unchanged).
- **C ABI 1.1** (additive): `tilefish_gemm_epilogue(gemm, int32_t* epilogue, bool* has_c)` with `TILEFISH_EPILOGUE_NONE`, `_BIAS`, `_RELU`, `_RELU_BIAS`
  (0 to 3, tilefish's values); `tilefish_gemm_launch` takes `c` before `out`; `tilefish::Gemm::epilogue()`, `has_c()` and `has_bias()` in `tilefish.hpp`.
- **The emitters** (`tilefish.hosts.rust`, `tilefish.hosts.cpp`) write typed launchers of a matmul entry (`a`, `b`, [`bias`], [`c`], `out`, `workspace`; `EPILOGUE`
  and `BETA`; docs of the formula with the launch's names, the epilogue, each architecture's contract and canary). The launchers of `gemm` and `gemm_bias`
  are byte for byte what they were.
- The Rust and C++ gates run 30 cases of ten matmul forms (every epilogue with `beta` 0 and 1, bf16 and f16, the four layouts, an opt-in, an SM count
  target, three buckets of `M`, a canary and none) and the plan `epilogue_chain` byte for byte against Python's replay, capture each in a CUDA graph, run
  every canary again with the NULL stream held, and refuse a forged canary (under a recomputed manifest pin in Rust, trusting the directory in C++).

### Smaller changes

- `tilefish._cublaslt`: `pin_algo`, `check_algo`, `describe` and `matmul` take `epilogue=` (and `beta=`, `c_ptr=`); `reduction_problem` takes the
  epilogue's name for its words. Called without them, every call is 0.0.11's.
- A plan backend's `gemm` is passed `c=` for a `cublaslt.matmul` step that reads `c` (only then). A re-export whose `contract` differs from the manifest's
  entry is another binary (`overwrite=True` to replace it).

### Tests

- The C++ GPU gate's own buffers (`crates/tilefish-capi/tests/gpu/driver.hpp`) finish their writes before the call returns. Their zeros and uploads
  run on the legacy stream, which the gate's non-blocking stream does not wait for, and a cuDNN convolution's workspace zeroed there while the
  convolution used it gave all-zero outputs now and then (the gate's `conv_k7.T646` and others: 77 of 1000 runs of its convolutions on a busy B300
  slice, none of 900 on an RTX 5090; none of 300 with the fix). Test code only: the crate and the C ABI order their own memory on the launch stream.
  The new gate test `buffers` checks it.

## tilefish 0.0.11, crate 0.0.9, tilefish-capi 0.0.1 (2026-11-17)

Manifest schema 3 and plan schema 1 are unchanged but for one optional entry field, `instrument`, which only the instrumented variant writes (below).
An export whose graphs have no arithmetic in their bodies and whose kernels write no `-0.0` scalar writes the same manifest, plans and plan sets as
0.0.10 (checked: every kernel and plan of the test suite and the Rust and C++ gates but those below, and the plan sets of three ported models
re-exported); a graph with arithmetic in its body is planned by fusion version 2 now (Breaking). An export of many buckets or plans takes less memory
and time, every byte the same. The crate gains the readers and launches of the instrumented variant; the C ABI is unchanged and refuses an
instrumented export.

The crate's sources change for the first time since 0.0.6 (the instrumented variant). `tilefish-capi`'s sources are unchanged (only a test of its GPU
gate reads two more dtypes), so it stays 0.0.1, built from the release commit.

### Breaking

- **Fusion version 2 is the default.** A graph whose body has arithmetic and declares no version is now planned by `tilefish.fusion(version=2)`
  (below): its fused regions have new names (the version is in each region's content hash), new sources (revision 2 of the lowering) and warp
  counts, so its manifest entries, plan and pin are new, and an ordered reduction, `max`, `argmax`, `argmin` or `gather_rows` is fused with its
  neighbours (fewer steps; an intermediate fused away has no buffer, so a `taps=` of it is refused: tap the unfused plan or add `tilefish.barrier`).
  The bytes it writes are those of 0.0.10 (every region is its unfused composition, and so was 0.0.10's) but where a `-0.0` scalar meets a zero
  (Fixed, below). A plan exported by 0.0.10 or 0.0.9 and loaded by the graph as it is is refused, and the message names version 1. **To keep
  0.0.10's plans and pins**, declare `@tilefish.graph(fusion=tilefish.fusion(version=1))`: version 1 plans as 0.0.10 did, byte for byte (plan
  digests of the use cases checked), but where it was wrong (Fixed).
- **What else of 0.0.10 changes:**
  - `tilefish._fusion.boundaries()`, the trace's `boundaries` and the advisor (`tilefish.advise`, `--advise`) report by the graph's fusion
    version: for version 2 no `reduction`, `gather` or row-region `materialised` and `template` reasons, and the fixes of the reasons `reduction`,
    `reduction-unordered`, `gather`, `materialised`, `template`, `limit`, `grouping`, `view` and the blockers `row-kernel` and `opaque-extern`
    are reworded.
  - A kernel or region that writes a `-0.0` scalar has a new source, binary and pin, and writes `-0.0` where 0.0.10 wrote `+0.0` (Fixed).
    In the test suite and the gates only the kernels written to test it; version-1 plan digests change for such graphs only.
  - A region that gathers rows from a view that is not dense copies it first (`tilefish.contiguous`, a new step) in both versions (Fixed).

### Fusion phase 2: row regions (`docs/rfcs/0001-plan-fusion.md`, "Phase 2")

- **`tilefish.fusion(version=2)`, the default.** An ordered `sum` or `mean` (`order=`), the order-free `max`, `argmax`, `argmin` and `gather_rows`
  join the ops around them in **row regions**: a residual add and an rmsnorm with the residual sum as a second output, a layer norm with an
  elementwise tail, softmax (a scale, a mask from a device scalar, `max`, `exp`, an ordered sum, a divide), a row-wise fp8 quantisation with its scale
  per row, a norm over two column blocks of one tensor, a gather and an argmax: one kernel each. Under the same exact contract as version 1: the
  fused plan writes the bytes of its unfused composition (`grouping="none"`), `export` verifies it at every bucket, and the version is frozen into
  every region's name and source. A `sum` or `mean` without `order=` still runs alone (D2). Version 1 is kept and plans as before (its regions,
  sources, names and plan digests checked against 0.0.10's, which are 0.0.9's).
- **Faster ordered sums: revision 2 of the lowering** (note 21 of `tilefish.lower`; `lower(..., lowering=2)`), which the regions of version 2
  are lowered with. Revision 1 took each step of a lane's chain with a `tl.gather` of the row, through shared memory: an rmsnorm of 4096 rows of
  1024 columns took 85 us on a B300 slice, its three unfused kernels with a Triton tree sum 61. Revision 2 makes the columns past the row's end
  `-0.0` (which `add.rn` leaves every value of a chain as it is), reshapes the row to `[lanes, steps]` and takes the chains by halving splits (data
  movement), the tree over the lanes through `tl.gather`: the same adds in the same order, checked against the CUDA kernels of each order at every
  length and at every warp count, and 1.6 to 37 times faster than revision 1. A row of ordered sums gets the warps that suit it (one warp up to
  `BLOCK_D = 2048`, then one per 1024 columns), and a fused row region reads views of rank 1 and 2 where they are (no `tilefish.contiguous` copy).
  An fp8 cast of a region is inline `cvt.rn.satfinite`, `.to()`'s bits on every input. Hand-written kernels keep revision 1 and their sources.
- **`tilefish.mean(x, axis, order=...)`**: the ordered sum times `1 / D` correctly rounded to f32, one rounding (`mul.rn`); the `reference`
  mode computes the same.
- **An ordered sum with `fma=True` in a graph body plans** (it raised a `KeyError` in 0.0.10): its product's multiply is part of the sum (each step
  `fma(a, b, acc)`), so the two are one region in every grouping, the unfused composition's too.
- **The advisor and `boundaries()`** follow the graph's fusion version (the fusion report's `fusion.version`): in version 2 an ordered reduction, a
  `max`, an `argmax` or a gather is no longer reported, a row region no longer blocks its neighbours, and a fused row region reading a view is no
  longer `materialised`; a graph of version 1 is told to use version 2 (the reasons `reduction`, `gather`, `materialised`).
- **Speed** on a B300 MIG slice, rows 32 to 4096 and 512 to 4096 columns (the RFC's phase 2 table): the fused row region is the fastest in
  every cell, 2.6 to 8 times faster than the same body with Triton's tree reductions run as three to five kernels (D2); an add + rmsnorm of 4096
  rows of 1024 columns takes 16 us (tree composition 65, version 1 121, one kernel per op 103).
- **`@tilefish.kernel(lowering=2)`** asks for revision 2 for a hand-written kernel (the same bytes; other binaries and pins).

### Intra-kernel instrumentation (`docs/instrument.md`)

- **`tilefish.probe("name")`** marks a phase boundary in a `@tilefish.kernel` body (between the ops written before and after it), a `@triton.jit`
  or a `@gluon.jit` kernel (a builtin Triton's code generator calls; importing tilefish still imports neither Triton nor torch). A production build
  compiles it to nothing: the IR and the machine code of the kernel without it (on sm_103 the production binary of a probed Gluon kernel has the
  `sha256_code` of the same kernel written without probes).
- **The instrumented variant**: `tilefish.runtime.export(ops, directory, instrument=True)` (or `tilefish.instrument(capacity=..., ops=...)`; graphs
  and plan sets too) builds every Triton, Gluon and lowered binary with its probes as inline PTX, and rewrites the PTX of each entry between Triton's
  `ptx` and `cubin` stages to add `@begin`, `@calib` (twice, in warp 0) and `@end`: lane 0 of every warp records `%globaltimer`, `%clock`, the probe
  and `%smid` into its slot of a probe buffer the host passes in the profile scratch slot (null in production, so the ABI is unchanged; a null buffer
  records nothing). Its entries carry `instrument` (manifest schema 3, section 4.8: the format, the capacity and the table of the kernel's probes
  with their lines, the partition of a warp-specialised kernel each runs in and the jit function), so its binaries, its manifest digest and the
  digests of its plans are its own. It is compiled every time (never from or into the build cache); the SASS assertions of a declaration are not
  judged on it.
- **It is never taken for the production artefact**: Python's `Manifest.load` refuses it unless `instrumented=True`, `tilefish.run(..., mode="replay")`
  and `tilefish.capture` refuse it unless `probes=` (and refuse `probes=` on a production export), `tilefish.hosts.rust.emit` and
  `tilefish.hosts.cpp.emit` refuse it, and an export refuses to write one variant into a directory of the other.
- **Probes and fused regions.** A probe is in a kernel body only (a graph body refuses it), so no fused region of either fusion version has one: in
  the instrumented variant a region's kernel records `@begin`, `@calib` and `@end` only. A `@tilefish.kernel` with probes that a graph calls stays a step
  of its own (a call is never fused), with its probes, and revision 2 of the lowering puts a kernel's probes where revision 1 does.
- **Collecting**: `rec = tilefish.Probes()`; `tilefish.run(op, ..., probes=rec)` (jit builds the variant in memory; replay takes an instrumented export),
  `tilefish.run(graph, ..., probes=rec)` (each bucket of an instrumented plan has a probe region, allocated with its binaries outside any capture, or
  the caller's `Probes(buffer=...)`), `tilefish.capture(..., probes=rec)` and `CapturedPlan.collect()`; `rec.trace()` is a `ProbeTrace`
  (`summary_text()`, `timeline_text(by="sm" | "cta" | "role")`, `chrome()`, `save()`), whose file `tilefish.probe-trace.v1` the Rust host writes too.
- **`tilefish-explain DIR --probes FILE`**: per launch its span, the SM clock it measured, what a probe costs a warp (from `@calib`), the records kept
  and dropped, and the phases between consecutive probes of a warp with their count and distribution per warp role (the partitions of
  `gl.warp_specialize`, from the frontend's IR), with each probe's source line from the export's trace; `--timeline sm | cta | role`, `--chrome OUT`
  (Perfetto), `--json`.
- **The Rust host** (crate): `Manifest::load_instrumented` (and `LaunchPlan::load_instrumented`, `PlanSet::load_instrumented`) reads an
  instrumented export, which `Manifest::load`, `from_embedded` and the plain plan loaders refuse; `Entry::instrument`; `Kernel::probe_bytes`,
  `ProbeBuffer::new`, `Kernel::launch_probed` (the buffer in the profile scratch slot) and `ProbeBuffer::collect`; `LaunchPlan::instantiate_probed`
  (a probe region with the run, per bucket; captured graphs hold its addresses) and `PlanRun::collect_probes`; `ProbeTrace::save` writes
  `tilefish.probe-trace.v1`. The C ABI has no probe calls and refuses an instrumented export.
- Measured on sm_103 (B300 slice): a probe costs a warp 45 to 80 cycles where it does not wait anyway; the instrumented kernels of the tests took
  -6% to +8% of their production time (a Gluon producer and consumer, a persistent kernel of 552 records per warp, an elementwise kernel of 4096
  short CTAs, a tcgen05 matmul); no shared memory, no stack frame. Triton's Proton was evaluated first and not used (`docs/instrument.md`, section 5:
  a shared-memory buffer, 131 to 197 KB on the tcgen05 matmul; the SM clock only; process-wide state).

### The memory of an export of many buckets

None of it changes a digest, `sha256`, `sha256_code`, manifest, plan or trace: an export writes the same bytes.

- **The frontend runs once per distinct input in an export.** An export (`tilefish.runtime.export`, a graph's and a plan set's too) and the
  `prepare` of a plan run Triton's frontend once per distinct kernel input (the function, signature, attributes, constants, options, architecture
  and variant: an instrumented build's IR has its probes), not once per bucket: the buckets of one `by_bucket=` configuration share one IR digest
  (`tilefish._ir.frontend_scope`). A `jit` call and a replay call outside them run it every time, as before.
- **Modules are freed.** Triton 3.5.1's Python binding does not own a module, so every module its frontend built and every module a compile lowered
  stayed allocated for the life of the process (about twice the size of the IR text, at every run): tilefish now erases the body of each module it
  is done with (`tilefish._ir.release_module`), after the digest and after the compile.
- **The IR is not kept.** A build keeps the digest of its frontend's IR and its dots (what the TF32 guard reads: `tilefish._ir.Frontend`), not the
  text, and hashes the text a megabyte at a time.
- **One copy of each binary.** In an export the entries that have the same cubin or PTX share one copy until the manifest is written, and a traced
  export (`trace=True`, `TILEFISH_TRACE=1`) keeps the stages of its builds in a temporary directory, one file per distinct text, until it writes the
  trace (`tilefish._build.export_session`).
- **A plan set is read back against one manifest.** An export of a plan set read each plan back with a manifest of its own (`_plan.load_plan`
  loaded the directory's manifest and every binary again for every plan, and each plan kept its copy until the export returned): a set of 1319
  plans over 147 binaries and 1199 GEMM pins peaked at 54 GB of resident memory and took 10 minutes on an RTX 5090, and was killed at a pod's
  32 GB on a B300. The plans now share the manifest the export loaded and checked (`load_plan(..., manifest=)`, the same checks): 1.3 GB and
  4 minutes on the RTX 5090, 1.3 GB and 150 s on the B300, the same files.
- An export of 132 buckets in three configurations of a Gluon kernel with a large IR (`python/tests/irmem_export.py`, B300): 77 s and 1.27 GB peak
  resident memory before, 9 s and 0.40 GB now (warm cache: 69 s and 1.17 GB before, 2.7 s and 0.25 GB now; traced: 371 s and 2.66 GB before,
  282 s and 0.43 GB now, the same trace).

### Epilogue and prologue fusion: RFC 0003 (`docs/rfcs/0003-epilogue-fusion.md`)

- A design, nothing implemented: fusion into a GEMM's or a convolution's call, which the advisor found is where nearly every fusible byte of the
  ported models sits. It keeps three numerics contracts apart: (i) exact, the bytes of the unfused composition; (ii) accumulator, a declared rewrite
  that drops the rounding of the result an epilogue widens to f32; (iii) library, a cuBLASLt epilogue or a cuDNN fused graph under a pin. The
  phase 0 experiments behind them are in it (a plain Triton `tcgen05` GEMM writes the bits of the pinned cuBLASLt GEMM on sm_103 wherever the pin
  does not split `k`; cuBLASLt's bias, ReLU and `beta * C` epilogues equal contract (ii); its GELU is defined by the binary). The decisions of
  2026-11-16: `epilogues="exact"` is the default, the other two apply only when declared; hooks on externs (E1) and cuBLASLt epilogues (E2) first.

### Fixed

- **A scalar `-0.0` keeps its sign** (note 3 of `tilefish.lower`): Triton 3.5.1 writes a literal `-0.0` as `+0.0`, so `where(x > 0, x, -0.0)`,
  `x * -0.0` and the like gave `+0.0` in a `@tilefish.kernel` and in a fused region alike, against the declared op and the `reference` mode. Every
  kernel (both revisions of the lowering, so also version 1's regions) now writes a `-0.0` scalar by its bits. Old and new binaries of the kernels
  it changes differ only there: on 3 x 2**20 random inputs with special values and every bf16 pattern, every differing element is one where the
  `-0.0` is selected, `+0.0` before and `-0.0` now. A `const` parameter whose value is `-0.0` is still `+0.0` (a `tl.constexpr`; open).
- **A view gathered from is read at its rows.** A fused region that gathers rows from a view that is not dense read it as if it were dense (the
  elementwise template of both versions, a rank-1 table: the wrong elements, fused and unfused alike); the view is now copied first.
- **The warps of a row whose gather table has bucketed rows** name every bucket of its binaries (`compile_specs` refused such a kernel).

## tilefish 0.0.10, crate 0.0.8, tilefish-capi 0.0.1 (2026-11-16)

Manifest schema 3, plan schema 1, both hosts and every digest are unchanged: an export writes the same manifest, plans, plan sets and fusion
reports as 0.0.9. The trace's graph file gains two keys (it is outside every digest). The crate's sources are those of 0.0.6 (only its version
moves, in step with the Python package); `tilefish-capi` is unchanged and stays 0.0.1, built from the release commit.

### The fusion advisor (`docs/advise.md`)

- **`tilefish-explain DIR --graph G --advise`** (`tilefish.advise(directory, graph, ...)`, an `Advice` with `.text()` and `.doc`, JSON
  `tilefish.advice.v1`): for a graph's plan, every fusion candidate (steps joined by intermediates that one producer writes and one consumer alone
  reads, which one kernel would not store; returned, carried, `InOut`, twice-written and tapped buffers excluded; a GEMM or convolution bounds a
  candidate as its epilogue or prologue), with its steps and op labels (the fusion report's and the trace's ops and lines), classes, bytes and
  microseconds saved per bucket, the launches it removes, why it is not one kernel today and the action that would make it one (write an extern's
  or a `@tilefish.kernel`'s arithmetic in the graph body, add `order=`, remove `tilefish.barrier`, phase 2 row fusion, a GEMM epilogue: future work),
  ranked by the time it would save; the time of the plan by class (library, fused, elementwise, row, copy, opaque); per step the bytes read and
  written, a GEMM's or convolution's FLOPs and a roofline against a per-architecture machine table (sm_90, sm_100, sm_103, sm_120); the blockers
  that would unlock the most (which phase-2 patterns help most); and the graph-body arithmetic that was not fused, with the reason and the fix.
  No GPU: it reads the export (the manifest, the plan, the fusion report, the trace). `--bucket`, `--timing`, `--machine`, `--bandwidth`, `--keep`,
  `--top`, `--json`.
- **`tilefish.profile_plan(graph, plan, bucket=, reps=, method=, devscalars=, out=)`** and `tilefish-explain DIR --graph G --profile OUT --load
  MODULE:ATTR`: the device time of every step of a plan at one bucket, a runtime path of its own that changes no artefact (the plan run as a run
  runs it, with timing events between its steps: event nodes of a CUDA graph, or events between launches behind a spinning kernel, which every run
  checks was long enough for the host to enqueue the steps; the caller's `InOut` tensors and the carried buffers the runs write are put back as
  they were), as a `tilefish.step-timing.v1` file the advisor ranks by. It records the device's SMs and copy bandwidth, the plan's time without
  the events and what the events cost each step (about 3 us on a B300 slice), and each step's time net of it; `docs/advise.md` compares it with
  Nsight Systems.
- **Why the regions are where they are**: `tilefish._fusion.boundaries(graph)` gives, for every value that crosses a boundary of a fused region,
  the reason (`barrier`, `reduction-unordered` (D2), `reduction`, `view`, `materialised`, `cycle`, `limit`, `template`, `move`, a call's epilogue or
  prologue, `grouping`), the ops on both sides and their lines, and the fix. It repeats the pass's rules (now one function, `merge_problem`, which
  the pass itself calls) and changes no region. A traced export writes it into `trace/graphs/<graph>.json` (`boundaries`), with each region's
  template.

## tilefish 0.0.9, crate 0.0.7, tilefish-capi 0.0.1 (2026-11-16)

Manifest schema 3, plan schema 1 and both hosts are unchanged, and so is every digest of an export that declares no SASS assertion
(`docs/explain.md`, issue #5 section 1): an export of 0.0.8 loads, pins and replays as before, and exporting it again writes the same manifest and
plans. The crate's sources are those of 0.0.6 (only its version moves, in step with the Python package); `tilefish-capi` is unchanged and stays
0.0.1, built from the release commit.

### SASS traceability and checks

- **Traced exports.** `export(..., trace=True)` (or `TILEFISH_TRACE=1` in the environment) writes `trace/` next to the manifest: for every binary
  it builds the source files under their module-relative names, the frontend's IR, TTIR, TTGIR, LLVM IR, PTX, the SASS with line information and
  its line table (each instruction's source line and PTX line), and for a lowered kernel or a fused region what each line of its generated source
  was written for (the op of the kernel's graph and the line of the body); for a graph, the steps of its plan and the ops of the body each fused
  region took. It has its own index and digest (`tilefish.trace.v1`) and is outside the pinned artefact: the manifest, the plans and every digest
  are the same with or without it (the whole GPU gate, traced, writes the bytes of an untraced export). A traced export compiles every Triton
  kernel (the build cache holds no IR).
- **`tilefish-explain`** (a console script; `python -m tilefish.explain`; `tilefish.explain.load`) for a kernel, a plan step or a fused region of
  an export: the SASS annotated with the source line of each instruction (a Triton or Gluon kernel's line; a generated line, its op and the line of
  the `@tilefish.kernel` body; a fused region's op of the graph body), the instruction mix (classes, MMA kinds and shapes, decoded from a
  `tcgen05` instruction descriptor where the SASS does not spell them, the PTX MMAs, TMA, local-memory traffic, barriers and fences, registers,
  stack frame, shared memory, an occupancy estimate), instructions per source line by class, and a diff of two exports or two architectures.
  It reads the trace with no CUDA tool, from any directory (`--source-root`), and checks it against the manifest; without one it disassembles
  the manifest's cubin.
- **SASS assertions**: `tilefish.sass(no_tf32=, mma=, no_spills=, max_registers=, forbid=, by_arch=)` on `@tilefish.kernel(sass=...)`,
  `@tilefish.extern(sass=...)` and `tilefish.fusion(sass=...)`. In the declaration digest when given (a fused region's name too), checked at
  every build, a cache hit and a prebuilt binary included, from the PTX (MMA kinds), the ELF file (registers) and the SASS; a violation refuses
  the build with the instructions and their source lines. `mma=` takes PTX patterns (`"tcgen05.mma kind::f16"`) and SASS opcode patterns
  (`"HMMA.16816.F32"`); `forbid=` instruction classes (`atomic`, `fp64`, `local`, ...) and opcode patterns.
- **Disassembly** goes through the `nvdisasm` of the contraction check (`TILEFISH_NVDISASM_PATH`, an `nvidia-cuda-nvdisasm` wheel, Triton's, the
  `PATH`); the new extra `tilefish[sass]` installs CUDA 13's, which is preferred. Where none decodes the architecture (Triton 3.5.1's CUDA 12.8 tools
  on sm_103) the PTX is assembled for the family base (`sm_100a`) by the binary's own ptxas and flags, and its SASS is judged only where its registers,
  stack frames and code sizes are the binary's (else a build with assertions that need the SASS is refused with the fix); `BuiltKernel.sass_checked`
  records what was read. `no_spills` and `max_registers` are decided from the binary's ELF file. A listing must cover every instruction of the code
  it judges (the entry and the functions it calls, never a sibling entry), or nothing is judged. The parser reads CUDA 12.8 and 13.x listings of
  sm_90, sm_100, sm_103 and sm_120 (recorded fixtures in `python/tests/sass_fixtures`).
- **The trace holds source code** (only the files of modules the exporting process loaded; nothing is searched for): it is not to be shipped.
- **Graph nodes record where they were written** (`Node.loc`, the module-relative file and line of the body) and a lowered kernel what each
  line of its source was written for (`LoweredKernel.line_map`); neither is printed, compared or in any digest or source.

### Fixed

- **The `nvidia-cuda-nvdisasm` wheel is found wherever pip installed it.** The contraction check (and now the SASS tooling) looked for its `nvdisasm`
  only under the directory `import nvidia` resolves to; with a regular `nvidia` package elsewhere on the path (the `nvidia/__init__.py` of CUDA 12
  wheels), a wheel installed with `--user`, `--target` or into a virtual environment over the system's packages was not found, and Triton's CUDA 12.8
  `nvdisasm` was used instead. Its installed files are now read first (`importlib.metadata`).

## tilefish 0.0.8, crate 0.0.6, tilefish-capi 0.0.1 (2026-11-15)

`tilefish-capi` 0.0.1 (`libtilefish_capi`, ABI 1.0, `tilefish.h`, `tilefish.hpp` and the CMake package) is built from the release commit with cargo or CMake; it is
not on crates.io. Manifest schema 3 and plan schema 1 are unchanged: a manifest or plan of 0.0.7 loads as before, but a lowered kernel with
`fp_fusion=False`, a float negation or a narrowing cast of a widened value has another source and declaration digest, so a replay refuses its
0.0.7 export (export it again; see "Fixed").

### Plan fusion: arithmetic in a graph body (`docs/rfcs/0001-plan-fusion.md`, phase 1)

- **A graph body may compute.** Arithmetic traced in a `@tilefish.graph` body (casts, `+ - * /`, comparisons and `where`, math functions,
  `iota` and `load`, reductions) is planned instead of refused: the planner groups it into regions, each a `@tilefish.kernel` that tilefish
  generates (`fused_<hash>`, named by its content, so the same region in two graphs or two plans of a set is one manifest entry) and one kernel
  step of the plan. Plan schema 1, manifest schema 3 and the hosts are unchanged; a graph without arithmetic plans exactly as before.
- **Exact.** A fused plan writes the bytes of its unfused composition (each op a kernel of its own, every intermediate stored in its dtype, a
  mask as `u8`): region kernels contract nothing and are lowered with `context_free=True` (note 20 of `tilefish.lower`): every float `+ - *` one
  scalar `.rn` instruction of inline PTX, every narrowing float cast inline PTX, a negation's NaN canonical. A reduction is a kernel of its own
  (version 1). A view that is not dense (a column block, a head, a transpose) is read where it is by a region of the elementwise template (a
  `Strided[...]` parameter of its kernel, a stride per axis) and copied by `tilefish.contiguous` first for one of the row template; an op whose
  value nothing reads is not computed.
- **`tilefish.fusion(version=1, grouping="auto" | "none", math="fast" | "ieee", intermediates="declared" | "opmath", fp_contract="none" |
  "mul-add", verify=True)`**, given as `@tilefish.graph(fusion=...)`: every choice exact; `"opmath"` (torch's eager rule for a 16-bit op with a
  python scalar) and `"mul-add"` (an f32 product that nothing else reads, contracted into `tilefish.fma`) are rewrites of the IR made before
  the regions are formed. `tilefish.barrier(x)` makes `x` a region boundary with a buffer of the plan (a tap reads it).
- **Verified at export.** `export` (and `export([...])` of a plan set) runs every region of more than one op against its ops one by one on the
  device at every bucket, on seeded inputs with special values, and raises `TilefishFusionError` before writing anything at a difference;
  `export(..., verify=False)` opts out, and a device of another architecture skips the check with a `TilefishFusionWarning`. It writes
  `<graph>.fusion.json`, the fusion report (the ops each region took, what it reads and writes, the roundings it keeps, the rewrites of the mode).
  `tilefish.fusion_report(graph)` and `tilefish.verify_fusion(graph)` give them on demand. `reference` mode of a graph interprets its arithmetic.
- **Found on the way** (sm_103, ptxas 12.9; `tests/test_gpu_fusion.py`): ptxas contracts a packed `mul.rn.f32x2` and `add.rn.f32x2`, which LLVM
  makes of two lanes' f32 ops, into `FFMA2` even with `--fmad=false`, so `@tilefish.kernel(fp_fusion=False)` does not guarantee two roundings
  on sm_100 and later where the ops are packed; and `neg.f32` folded into a select flips a NaN's sign where alone it gives the canonical NaN.
  Fused regions avoid both (and so does every kernel now: "Fixed" below).

### The C++ host (`docs/rfcs/0002-cpp-host.md`)

- **`crates/tilefish-capi`: the C ABI of the Rust host.** `libtilefish_capi` (shared, with the SONAME `libtilefish_capi.so.1`, and static)
  exposes loading and verifying manifests (from a directory or from memory, with a pin; trusting a directory is a function of its own), plans
  and plan sets; kernel, pinned GEMM and pinned convolution launches; plan runs on their own arena, on a shared arena and as a set run, with
  taps, weights bound once, carried buffers and memset steps; captures and replays; and the cuDNN handle of a context made early. Every
  check is the `tilefish` crate's. Tensors are regions the host vouches for; contexts (a device's primary context, or one the host lends)
  and streams (`CUstream`, `cudaStream_t`) are the host's; every call leaves the thread's current context as it found it.
- **`tilefish.h`**, generated by cbindgen from the crate and committed: a test generates it again and requires the same bytes, and another
  that every export is declared in it. It compiles as C99 and C++17 and needs no CUDA header. `tilefish_abi_version()` and
  `TILEFISH_ABI_MAJOR`/`MINOR`: a minor version only adds. Errors are a `tilefish_status` per call (one per variant of `tilefish::Error`,
  and `INVALID_ARGUMENT`, `NO_DRIVER` and `INTERNAL`, a panic caught at the boundary) and the thread's last error (`tilefish_last_error`).
- **`tilefish.hpp`**, a header-only C++17 wrapper: move-only RAII handles, `tilefish::Error` (a `std::runtime_error` with the status, the
  precondition, the argument and the `CUresult`), and typed tensors `TensorRef<T>` and `TensorMut<T>`.
- **A CMake package**: `crates/tilefish-capi/CMakeLists.txt` builds the library with cargo and installs `tilefishConfig.cmake` with the
  targets `tilefish::tilefish` and `tilefish::tilefish_static`; ctest runs C99 and C++17 programs against both libraries (no GPU needed)
  and a consumer of the installed package.
- **`tilefish.hosts.cpp`**: `emit` writes typed C++ launchers from a pinned manifest (one class per binary, GEMM and convolution; with
  `embed=True`, the manifest and its cubins in a companion source file and `load_embedded()`), and `emit_plan` a typed runner of a pinned
  plan, with the validation of `tilefish.hosts.rust`.
- **The C++ GPU gate** (`crates/tilefish-capi/tests/gpu/cpp_gate.cpp`, built with `-DTILEFISH_BUILD_GPU_GATE=ON`) runs the cases of
  `python/tests/rust_gate.py` from the same directory, and `rust_gate.py compare WORK --host cpp` holds them to Python's replay byte for
  byte; `rust_gate.py prepare` writes the C++ launchers and plan runners too (the block's, and every fused plan's, `fused_*`). `tests/gpu_overhead.rs` reports the host time a launch takes
  through the C ABI and through the crate.

### Fixed

- **`fp_fusion=False` could still contract packed mul+add into FFMA2 on sm_100+.** On sm_100 and later LLVM packs two lanes' f32 operations
  into `mul.rn.f32x2` / `add.rn.f32x2`, and ptxas 12.9 contracts such a pair whatever `--fmad` says (into `FFMA2` on sm_100 and sm_103, into
  `FFMA` on sm_120), so `@tilefish.kernel(fp_fusion=False)` did not always round twice where its contract (note 2 of `tilefish.lower`) says it
  does. A kernel with `fp_fusion=False` now has the code of a fused region: every float `+ - *` (f32, bf16, f16), and the multiplies of `silu`
  and `gelu`, is one scalar `.rn` instruction of inline PTX, and every narrowing float cast is inline `cvt.rn`. Its outputs are the old ones
  wherever the old binary kept two roundings; its source and declaration digest change (and its `sha256_code` where Triton's code was
  packed or folded), so a manifest exported before is refused by a replay of the new declaration (export again). A product that an ordered sum without `fma` adds (note 17) is `mul.rn` in
  every kernel.
- **neg inside a select flipped the NaN sign.** `-x` (`tilefish.neg`) gives the canonical NaN (`0x7FFFFFFF`, `0x7FFF`) for every NaN in every
  context now: the canonicalisation is written out (`testp.notanumber` / `setp.nan` and `selp` around `neg`), because ptxas folds a lone
  `neg.f32` into a select as a negated operand, which flips the NaN's sign. Every kernel with a float negation gets another binary.
- **A narrowing cast of a widened value kept an input NaN's payload.** `x.to(f32).to(bf16)` (and through a `where`) is inline `cvt.rn` in every
  kernel, out of the reach of the MLIR and LLVM folds that turned it into `x`; it gives the canonical NaN, as the cast does alone (note 1).

### New

- **The extern SASS contraction check** (`tilefish._fpcontract`, `docs/numerics.md` section 4). Every binary built with
  `enable_fp_fusion=False` (a hand-written Triton or Gluon extern that declares it, a lowered kernel with `fp_fusion=False`, a fused region, a
  generated op) is disassembled at build, cache hits included, and compared with its PTX assembled again with every float multiply taken out; one
  with more fused multiply-add lanes (`FFMA`, `FFMA2`, `HFMA2`) is refused with a `TilefishBuildError` that names the kernel, shows the
  instructions and gives the fix (scalar `.rn` inline PTX, or `enable_fp_fusion=True`). `nvdisasm` is `$TILEFISH_NVDISASM_PATH`, Triton's, the
  one next to its `ptxas`, an `nvidia-cuda-nvdisasm` wheel's or the `PATH`'s; where none decodes the architecture (Triton 3.5.1's CUDA 12.8
  `nvdisasm` and sm_103) the PTX is assembled for the base of its family (`sm_100a`) and read. An extern that ptxas contracted is refused where
  it used to build.

## tilefish 0.0.7, crate 0.0.5 (2026-11-15)

### Breaking changes

- A float to integer cast in a kernel body says its rounding: `x.to(i32)` of a float tensor raises a `TilefishTypeError` at the
  declaration (it used to trace and then be refused by the lowering, rule `cast.float-to-int`). Write
  `x.to(i32, rounding="rz")` for a truncation.
- The build cache recipe is 5: entries stored by 0.0.6 named their build directory, so they are not read, and kernels are built again once.
- Readers of 0.0.6 and crate 0.0.4 refuse what this adds (`null` parameters, an element-aligned dense pointer); manifests without them load and digest as
  before. The crate's `ParamKind` gained the variant `Null`.
- Readers of tilefish 0.0.6 and crate 0.0.4 refuse a plan with a memset step or a carried buffer; every other plan, and its digest, is unchanged.
- The crate's public enums gained variants (`BufferRole::Carried`, `StepKind::Memset`), and `SharedArena::bytes_for` counts the carried regions.
- Readers of 0.0.6 and crate 0.0.4 refuse what this adds (`persistent`, `options.num_ctas`, `global_scratch_align`, a kernel step with a `workspace`,
  clusters and global scratch at launch). Manifests and plans without them are unchanged, digests included. A 0.0.6 manifest whose kernel has a
  global scratch still loads, and is refused at launch (no alignment recorded): export it again.
- The crate: `Entry::persistent`, `Persistent`, `LaunchAttributes::global_scratch_align` and `cluster()`, `Plan::prepare_for`, `Plan::cluster()`,
  `Plan::global_scratch()`, `Prepared::cluster`, `Prepared::scratch_bytes` and `Kernel::sm_count()` are new; `Prepared::grid` is the grid of blocks
  (programs times the cluster).
- An export now writes `cuda_math` (manifest schema 3, section 4.7) on the entry of every generated kernel whose ops copy a CUDA math function
  or that links libdevice. The crate 0.0.4 refuses an entry with a field it does not know, so a Rust host loads a new export with such kernels
  only with crate 0.0.5; manifests of 0.0.6 and earlier load as before.
- The Rust host refuses a plan launch on a stream that is capturing a CUDA graph (`PlanRun::capture` makes a plan's graph), a capture no longer
  holds the arena's lock to its end, and `PlanSet::load` of a set the process verified already, while a clone of its manifest is alive, takes
  the verified manifest without reading the files again (`Pin::TrustDirectory` included).
- Python's strict manifest reader refuses the table of dimensions of a kernel entry that the Rust reader refuses (a `divisible_by` below 1,
  an unknown field, unsorted `buckets`, a `bucket` that is not the entry's), where it used to leave that to the comparison with the spec at
  replay; the Rust reader checks the `static_size` preconditions in their order, so the one it names is the same in every run.

### Reproducible exports and builds

- **A cubin names its sources by module path.** Triton records the file of a kernel and of every jit function it calls in the IR's
  locations, and `ptxas` writes them into the cubin's debug sections. These were absolute paths: the checkout, the virtual environment
  (`triton/language/standard.py`), and the random `/tmp/tilefish-gen-XXXXXXXX` directory of a lowered kernel's or a generated op's source.
  So every export of the same operators had other cubin bytes, another `sha256` and another manifest and plan digest (the `sha256_code`
  did not change). A build now names each file by a path that does not depend on the host: `kernels/rms.py` for the module
  `kernels.rms`, `tilefish_generated/<name>.py` for a generated source, otherwise the base name. Two exports with the same toolchain
  for the same device are the same files and digests, from any directory, process or machine, and a recorded pin checks a rebuild.
  `ptxas` runs in an empty directory, so that it does not find a source file by such a relative name and record its modification time
  and size (a build run from the directory of the sources would have).
  The machine code is unchanged: every `sha256_code` is what 0.0.6 built. Manifest schema 3, the digest and both readers are unchanged,
  and manifests of 0.0.6 or earlier load and replay as before (`docs/manifest-schema-3.md`, section 3.1, says why the digest still covers
  the files rather than `sha256_code`).
- **A generated source has a content-addressed file.** `runtime.load_lowered` writes the source of a lowered kernel or a generated op
  (`tilefish.contiguous`, `im2col`, `im2col1d`, `col2im1d`) to `<cache>/generated/<sha256>/tilefish_generated/<name>.py` in tilefish's
  cache directory, through a temporary file and an atomic rename, and uses a file there only when it holds exactly the source. The
  module runs from the checked text, never from a byte code cache. When the directory is switched off or cannot be written, the
  source goes to a private scratch directory of the process (one `TilefishCacheWarning`); otherwise no scratch directory is made.
- `TILEFISH_DEBUG_PATHS=absolute` keeps the absolute source paths in a cubin's debug sections, for a profiler that opens the files where
  they are. Off by default; it breaks reproducibility (that export's bytes and digests name where it was built; its `sha256_code`s do not
  change), and its binaries are cached under a key of their own. Any other value is refused.
- **Hosts.** The Rust crate embeds no path of its own. A host that embeds a manifest (`emit(..., embed=True)`) is reproducible once
  the manifest is (export one made by 0.0.6 or earlier again); the paths rustc writes for panic locations are remapped with
  `--remap-path-prefix` (crate README, "Reproducible builds of a host").

### Types, arguments and ops

- **64-bit dimension scalars.** An extern's runtime scalar that carries a dimension may be an `i64`: `n_elem: i64[n]` (`i32[n]` is `n`, as before).
  Its value is derived from the shapes and range-checked as an `i64` by both hosts, where an `i64` count used to be a per-call `given` value nothing
  checked. The declaration digest has `"scalar_type": "i64"` on the parameter; the manifest has a `scalar` of type `i64` (schema 3, section 5.7).
- **Null pointers.** A pointer parameter of an extern's kernel that the declaration does not use is `Null[dtype]`: a call passes nothing for it, and both
  hosts write the 64-bit zero at its slot (a Triton kernel's and a prebuilt binary's alike; manifest kind `null`, section 5.6), where a one-element dummy
  tensor was needed before. The Rust crate reads it into `ParamKind::Null`.
- **Lowered kernels on views that do not start on 16 bytes.** A graph that passes a dense view not provably aligned to 16 bytes (a row of a `[K, R]` int32
  plane with an odd `R`) to a read-only parameter of a lowered kernel runs the kernel's specialisation for it, `kernel.unaligned("a")`
  (`<kernel>__ua<indices>`), instead of refusing the call; so does `tilefish.run` of an address that is not 16-byte aligned. Its pointers for those
  parameters have no divisibility hint and promise their element's alignment (a `pointer_alignment` a dense pointer may now have, section 6); its stems,
  declaration digest, IR and build cache key are its own, its source the kernel's.
- Float to integer casts are lowered: `x.to(i32 | i64 | u8, rounding="rn" | "rz" | "rd" | "ru", saturate=True | False)` (from f32, bf16,
  f16, e4m3 or e5m2). `saturate=True`, the default, clamps to the type's range with a NaN 0; `saturate=False` keeps the low bits of the
  `i64`. Each is a helper of inline PTX (`cvt.rni.s32.f32`, ...), and the `reference` mode computes the declared mode exactly: the two are
  byte-identical on every f32 bit pattern. Note 18 of `tilefish.lower` says what torch's casts are in these terms.
- The row template reads a static slice `x[k]` (`x.select(0, k)`) of a read-only parameter at its offset: `codes[k]` of a `[K, R]` codes
  plane is a per-row operand, such as the index of `gather_rows`, and `tables[k]` of a `[K, N, D]` parameter is a table. A dynamic
  dimension of the offset is a runtime `i32` parameter (`size_R`) that a host derives from the shapes. Every other view is still refused
  (rule `view`). Note 19 of `tilefish.lower`.
- `x.split(16, axis=1)` of a dimension declared only `divisible_by=16` (or whose buckets are all multiples of 16) returns
  `tilefish.ops.Chunks`, a sequence whose `chunks[0]` and `chunks[-1]` (and any chunk the dimension's smallest size proves) are views,
  where it used to raise "not a whole number of chunks"; `len()`, iteration and unpacking raise. A split whose count is a number still
  returns a tuple. Plans and the graph `reference` mode take these chunks.

### Implementations per architecture (manifest schema 3, section 4.1)

- **Named implementations.** `tilefish.impl(..., name="tcgen05")` is one implementation of every architecture `by_arch=` maps to it (and of the
  fallback, which may be named too): its key is its name, so those architectures share one declaration digest and one set of compile specs, and
  their manifest entries say `"implementation": "tcgen05"`. A name is an ASCII identifier of at most 64 characters that is not `fallback` and does
  not begin with `sm_` and a digit; two different implementations may not share one. Without a name nothing changes: keys are architectures, and
  digests are those of 0.0.6.
- **Prebuilt binaries as implementations.** `tilefish.impl(ptx="k.ptx", symbol=..., block=..., shared=...)` and `tilefish.impl(cubin={"sm_120":
  "k.cubin"}, ...)` (paths relative to the file that declares the op). Their entries are `binary` entries with `implementation`. A cubin
  implementation must have a cubin for every architecture that runs it: refused at the declaration for the architectures `by_arch=` names, at the
  build for those that fall back to it (PTX whose `.target` an architecture cannot build is refused likewise).
- **An ABI per implementation.** `tilefish.impl(..., stub=f)` gives an implementation its own stub, the kernel's parameters in order: a TMA
  descriptor (`TensorDesc`) where the op's stub has a pointer, other scalars, constexprs or fixed arguments. Every implementation must share the op's
  public signature: its tensors (names, order, access, dtype, shape, `Strided` or not, `MayAlias`), the constraints of their dimensions, its fixed
  arguments per call and its buckets. `jit` and `replay` launch the implementation of the device with its own ABI; the planner binds every step
  with every implementation, preconditions included, so a plan never relies on an architecture; the Rust emitter types `launch` from the shared
  tensors and documents each architecture's ABI.
- The build refuses a spec of one implementation for an architecture the declaration does not run it on, which now includes the fallback on an
  architecture that has an implementation of its own (that build's entry was always refused by `replay`).
- A call that restates a const of an op declared with `by_arch=` is held to the value of the op's stub, as a graph's call is, and to the launched
  implementation's where it has a constexpr or fixed argument of that name.
- Both manifest readers accept, as an entry's `implementation`, its architecture, `"fallback"` or a name of that form, on a `triton` or a `binary`
  entry. Readers of 0.0.6 and crate 0.0.4 refuse a name and an `implementation` on a `binary` entry; manifests without them, and their digests, are
  unchanged.

### Launch shapes (manifest schema 3, sections 4.3 to 4.6; plan schema 1, sections 6 and 7)

- A grid may multiply dimensions: `grid=lambda T, C: (tilefish.cdiv(T * C, 1024),)` (written `(cdiv(C*T, 1024),)`), one binary for every `T` and
  `C`, in both hosts, plans and captures. The readers always took such a grid; a shape stays affine.
- Persistent grids: `@tilefish.extern(..., persistent=tilefish.persistent(programs_per_sm=1))`. The grid is the tiles, and the host launches at most
  `programs_per_sm * SMs // CTAs per program` programs, the SMs resolved when the binary is loaded (the device's, or a partition's:
  `Manifest::with_sm_partition` in Rust, `tilefish.runtime.sm_partition(n)` in Python). The kernel's bytes must not depend on the number of programs,
  unless it declares the SM count it is for (`sm_count=30`), which a host on another count refuses. New optional entry field `persistent`.
- A configuration per bucket: `@tilefish.extern(..., by_bucket=lambda TB: tilefish.config(num_warps=..., num_stages=..., num_ctas=..., BLOCK=...))`,
  each bucket's binary built with its own warps, stages, CTAs and constexpr values and launched with the grid under them; the declaration digest covers
  every bucket's configuration. Nothing new in the manifest.
- Thread-block clusters: `num_ctas=2` (Triton's compile option, 1 to 8) and, for a prebuilt binary, `cluster=(x, y, z)`. Both hosts launch them with
  `cuLaunchKernelEx` and the cluster attribute (the grid counts clusters), in single launches, plans and CUDA graphs; a Triton entry records
  `options.num_ctas`.
- A global scratch: `global_scratch=N` (bytes per CTA, checked against Triton's metadata) for kernels with device-side TMA descriptors
  (`tl.make_tensor_descriptor`). Python allocates it per launch; the Rust crate takes it as the last tensor of a launch (`GLOBAL_SCRATCH`,
  `Prepared::scratch_bytes`); a plan's kernel step names the workspace as its scratch. New optional `launch.global_scratch_align`.

### Plans and arenas

- **Carried buffers.** `@tilefish.graph(carried=("h",))` names parameters (`In` or `InOut`) whose tensor the plan's arena holds instead of the
  caller: buffer role `carried` (offset `null`), placed by the host after the plans' own regions, one region per carried name in code point order
  (`_plan.carried_layout`, Rust `CarriedLayout`), so one stage of a shared arena hands an intermediate to the next with no allocation of the
  caller's. A run gives every parameter but the carried ones; a plan that reads a carried buffer before it writes it binds its dimensions from the
  shape the buffer was written with, and is refused before anything is enqueued when no execution on the arena wrote it, or wrote another shape
  (both hosts, launches and host-owned replays). `h[...] = op(x)` is written by the launch of `op` (no copy). `docs/plan-schema-1.md`, section 15.2.
- **Memset steps.** `tilefish.zeros_like(x, dtype=None)` in a graph body is a step of kind `memset` (`cuMemsetD8Async` of a dense buffer, a memset
  node under capture), in both hosts; `state[...] = tilefish.zeros_like(state)` is one memset of the caller's tensor. Additive: no schema bump, and
  a plan without one is unchanged (section 7.6).
- **Several arenas of the same plans.** `tilefish.runtime.share_arena(graphs, name="lane1")`: arenas are kept per plan layout, device and name, so
  a lane that runs concurrently (one per stream) takes an arena of its own; documented for both hosts (the Rust host already made one per
  instantiation and per `SharedArena`) in section 9.
- **A view that is not dense stored into an `InOut` parameter** of a graph (`state[...] = t.permute(1, 0)`) is accepted and planned as a launch of
  `tilefish.contiguous` into the caller's tensor, as a returned view already was. A kernel body still refuses it.
- **Graphs of the caller's.** The second thread that uses an arena a `torch.cuda.graph` capture of `tilefish.run` holds gets one
  `tilefish.TilefishOrderWarning` (such a graph's `replay()` takes no lock), and `tilefish.runtime.replay_graph(g)` replays one under the arena's
  lock, ordered with every thread (0 of 3 000 runs wrong against it). Section 10, item 7, says why it cannot be made safe behind the caller's back.
- A `shared` that is nonzero but too small stays undetectable without a pin: measured, a Triton cubin does not record the dynamic shared memory it
  addresses (`test_gpu_shared_size.py`; `docs/manifest-schema-3.md`, section 4).
- A shared arena of runners of which some have run and some not is no longer refused for "another backend".

### Numerics in the DSL (notes 15 to 17 of `tilefish.lower`)

- `tilefish.erf(x)`: libdevice's `erff` by default (what `gelu` uses), CUDA's `erff` without flush to zero with `math="ieee"` (the bits of
  CUDA 12.8 to 13.1; CUDA 13.2 changed the function).
- `tilefish.sin_approx(x)` (`sin.approx.f32`, CUDA's `__sinf`), `tilefish.rcp(x)` (`rcp.rn.f32`, `__frcp_rn`), `tilefish.fma(a, b, c)` (`fma.rn.f32`,
  one rounding whatever `fp_fusion` says) and `-x` / `tilefish.neg(x)` (CUDA's negation: `neg.f32`, `neg.bf16`, `neg.f16`, or `0 - x` of an
  integer). `-x` was refused before.
- Ordered sums: `tilefish.sum(x, axis=-1, order="sequential" | tilefish.tree(lanes, offsets, vector), fma=False)` (row template, f32): lanes of
  chains from `+0.0` and a fixed tree over the lanes, every add `add.rn`; `fma=True` fuses the multiply of `a * b` into each step. Byte for byte
  the CUDA kernels of those orders, and the same bytes at every `num_warps`.
- `num_warps` stays part of every kernel's identity; the contract now says which ops depend on it (Triton's own reductions) and which do not
  (per-element ops, `argmax` / `argmin`, ordered sums), measured.

### The CUDA math guard (manifest schema 3, section 4.7; `docs/numerics.md`)

- The manifest entry of a generated kernel whose ops copy a CUDA math library function (`expf`, `rsqrtf`, `erff` in IEEE math; `sinf`, `cosf`,
  `tanhf` from libdevice) or that links libdevice has `cuda_math`: the functions, the CUDA releases tilefish measured each on over every f32
  (12.8 to 13.2, `erff` 12.8 to 13.1), and the sha256 of the libdevice. Optional and additive: an entry without it reads as before, and no
  generated source or source digest changes.
- `jit`, `replay` and `reference` warn with `tilefish.TilefishCudaMathWarning` where torch is built with a release the functions were not measured on,
  or where a replay's process would link another libdevice. Rust: `Entry::cuda_math` (`CudaMath`) and `Manifest::check_cuda_math(release)`.
- `pytest -m cuda_math` runs the exhaustive comparisons with CUDA's math library again; `TILEFISH_NVRTC_PATHS` adds the NVRTC of more releases
  to the table's check. `docs/numerics.md` is the procedure on a CUDA or torch change, the `num_warps` decision, and a design for device-global
  state (counters), which waits for a memset step in plans.

### Pinned GEMMs

- `test_gpu_gemm.py` asserts that `heuristic="torch"` (with torch's default reduction) gives `torch.mm` / `torch.addmm` byte for byte on the 30
  shapes whose default-posture rate it reports, with cuBLASLt 12.8.4 and 13.1.0. Why the default posture matches `addmm` on only 13 of the 15
  biased shapes under cuBLASLt 13.1 is documented: it gives a GEMM with a bias no split-K, a rule of tilefish's, where `addmm` runs the split-K of
  3 with f32 partial sums that the heuristic ranks first for 32 and 64 x 1024 x 3072. No pin changes. The rule stays: it is what lets the
  default posture promise one rounding for every pin without measuring each split-K algorithm (`docs/numerics.md`, section 4).

### The reference mode

- The tolerance matrix of `test_gpu_reference.py` covers f32, bf16 and f16 inputs and outputs of the layernorm, gelu, swiglu and softmax goldens
  on five more distributions (uniform, heavy-tailed, sparse outliers, values of every magnitude, rows of one value), measured and asserted.

### Hosts

- **The order of the executions on an arena costs less host time.** A run, replay or step alone on the stream where the arena's
  order event was last recorded skips the wait for it (stream order already puts it after that record) and still records it, so
  executions that stay on one stream cost one `cuEventRecord` each, and one on another stream one `cuStreamWaitEvent` more; the most
  recent record still stands for every execution before it, so cross-stream ordering holds. Python never skips inside a capture, on
  the per-thread default stream, or on an arena a `torch.cuda.graph` captured, and now records the event when the arena is allocated,
  on the stream it was allocated on. The Rust host orders an arena by an order event of its own instead of as a slice owned by the
  stream it was allocated on (no work is enqueued on that stream any more), copies the device scalars with no cudarc tracking of the
  arena and one event per staging entry, and a capture holds the arena's lock only to begin (executions and captures on other streams
  go on meanwhile). Measured on a B300 MIG slice, four plans per request: Python makes 0 waits per request instead of 4 on one
  stream (1 instead of 4 alternating two streams), and a `CapturedPlan.replay` takes 7.8 to 8.4 us of host instead of 9.8; Rust
  replays take 5.9 to 7.7 us of host per request instead of 7.3 to 9.2 on one stream, 7.1 to 7.7 instead of 9.8 to 11.5 alternating.
- **Start-up of a large plan set (Rust).** `PlanSet::load` verifies a set once per process: a second load of the same set digest,
  while a clone of the first one's manifest is alive, reads only the index (131 -> 1.1 ms for a synthetic set of 1280 plans, 96
  kernels and 1024 GEMMs). `PlanSet::plans(names)` loads plans on up to 8 threads (238 -> 35 ms), and `PlanSet::instantiate` loads
  and instantiates its plans on threads (the binaries and GEMMs on one thread: the driver's and cuBLASLt's loads do not get faster on
  threads); loading and instantiating that set took 413 to 452 ms instead of 708 to 723, the whole start-up with its 2560 captures
  1154 to 1166 ms instead of 1424 to 1436. Captures on several threads measured no faster (the driver serializes them).
- **`cudnnCreate` waits for the whole device**, every stream, not only the NULL stream (measured, cuDNN 9.10.2), so the first
  convolution loaded while work runs stalls behind it. `tilefish::CudnnHandle::new(&ctx)` (crate) and
  `tilefish.runtime.warm_cudnn(device)` (Python) make the handle at start-up and keep it, so that no load makes one (the first run
  of a cuDNN engine, a load's canary, may also wait while CUDA loads its kernels lazily: load convolutions at start-up too).
- The crate no longer turns on serde_json's `preserve_order` for the whole host process, and reads the same with or without it: the
  order of an entry's `buckets` and `dims` and of a plan's `dims` comes from the file; `tilefish::json::parse_strict` returns every
  object in code point order.

### Tooling and tests

- A build that fails because the `ptxas` Triton compiles with cannot target the architecture (Triton 3.5.1 bundles CUDA 12.8's,
  which has no `sm_103a`) says so before Triton's message, with that `ptxas` and its version, and says to set `TRITON_PTXAS_PATH`
  (for sm_103, the `ptxas` of CUDA 12.9 or newer). The GPU tests that build for architectures other than the device's skip when
  that `ptxas` cannot target one of them, naming it, its version and `TRITON_PTXAS_PATH` (`conftest.ptxas_problem`, which reads the
  targets from `ptxas --help`). README documents the requirement.
- The allocation-free plan gate test (`gpu_plan_gate.rs`) compares only what this process allocates: the used memory of the device's
  memory pool and of its graph memory, their high watermarks over the launches and the replays, and the pool's reserved memory, which
  may shrink at a synchronize but must not grow. The device's free memory, which other processes on a shared GPU change, is only
  reported.
- The committed launchers of the crate's fixture (`crates/tilefish/tests/fixtures/generated.rs`) embed the fixture (`embed=True`),
  so `cargo clippy --all-targets --features gpu-gate` builds with only committed files (`TILEFISH_GATE_GENERATED` set to that file;
  see the header of `gpu_gate.rs`). `export_rust_fixtures.py --generated-only` writes the file again from the committed fixture with
  no build. The published crate is unchanged (`tests/` is not packaged).
- The cuBLASLt process-stability tests use neutral shapes (N = 1024 with K = 8192 and 16384) instead of `(N, K) = (1024, 7680)`;
  on an sm_120 device the test asserts that one of its problems still gets a split-K configuration.

## tilefish 0.0.6, crate 0.0.4

### Breaking changes

- **The TF32 guard.** Every build (`jit`, `export`, a plan, and a build-cache hit, which is checked again from its PTX) refuses a
  Triton or Gluon kernel whose IR asks for an IEEE f32 `tl.dot` but whose binary has TF32 tensor-core MMAs made of that dot.
  Triton 3.5.1 compiles some of them to `tcgen05.mma ... kind::tf32` on sm_100 and sm_103 (an operand loaded narrower than 32 bits,
  or from `tl.join`, in a tile with `M % 64 == 0` and `N % 16 == 0`). A kernel that built with 0.0.5 can be refused now; the
  error names the dot and the fixes (fewer than 64 rows, a loop of `tl.fma`, or `input_precision="tf32"`). README, "An IEEE f32
  dot is not built into TF32 MMAs".
- The build cache recipe is 4: entries stored by 0.0.5 are not read, and kernels are built again once.
- Readers of 0.0.5 and crate 0.0.3 refuse what this release adds (unknown fields and values are refused): manifests with
  `tensordesc` parameters, `total_num_warps`, GEMMs of another form than bf16 `TN`, `sm_count_target`, `allow_f16_reduction`,
  `allow_f32_reduction` or `cudnn.conv_fwd` entries. Entries of 0.0.5 are unchanged, digests included.
- The crate's public types gained fields and variants (for example `ParamKind::Pointer::tensor_map`,
  `Entry::total_num_warps`, `LibraryEntry::dtype` / `out_dtype` / `layout`).

### Pinned GEMMs (manifest schema 3, sections 9.2 and 9.3)

- `@tilefish.library(impl="cublaslt.gemm" | "cublaslt.gemm_bias")` takes bf16, f16 or f32 operands (both of one type), a result of
  their type or f32 (f32 operands: an f32 result), a bias of the result's type, and `layout="TN"` (the default), `"NN"` (`b`
  stored `[K, N]`), `"TT"` and `"NT"` (`a` stored `[K, M]`).
- f32 GEMMs are multiplied as f32 with TF32 off: `CUBLAS_COMPUTE_32F`, never `_FAST_TF32` or cuBLAS 13's bf16x9 emulation. Both hosts
  refuse an f32 pin where a fresh encoding is not `CUBLAS_COMPUTE_32F` (`NVIDIA_TF32_OVERRIDE=1`,
  `CUBLAS_EMULATE_SINGLE_PRECISION=1`).
- Split-K partial sums in the output type are opt-in per operand type: `allow_f16_reduction`, `allow_f32_reduction`.
- `AlgoTable` rows of the `NN` layout are keyed `(m, n, k, bias, "NN")`; a configuration is encoded with the declaration's types.
- `sm_count_target=T` (8 to the device's SM count) pins the algorithm cuBLASLt's heuristic gives for a partition of `T` SMs (a green
  context); the Rust host says which partition it runs on with `Manifest::with_sm_partition(Some(T))` and refuses entries chosen
  for another count.

### A pinned cuDNN convolution (manifest schema 3, section 9.4)

- `@tilefish.library(..., impl="cudnn.conv_fwd")`: a forward convolution of one to three spatial axes, `NCHW` or `NHWC`, f16, bf16
  or f32 data with an f32 accumulator, stride, padding, dilation, groups and an explicit math (`"tensor_op"`, `"fma"`, or `"tf32"`
  for f32 data).
- What cuDNN runs is pinned in one of two forms: a legacy algorithm enum, or a backend engine configuration (an engine and every
  knob choice), chosen as torch 2.9 chooses with `deterministic=True`, or imported with `tilefish.CudnnTable`. A knob of -1 is
  cuDNN's "not set": an imported pin drops it, and a pin or an entry that has one is refused.
- Replay and the Rust host refuse another cuDNN version, another SM count, an SM partition the pin was not chosen for and any
  `NVIDIA_TF32_OVERRIDE`, and vet the pin.
- An optional canary per entry, `{seed, y_digest}`: the digest of the pin's output on operands made from the seed, made and
  digested on the device by one PTX module in both hosts, checked at load on a stream of its own (never the NULL stream). It tells
  two builds of one cuDNN version apart. Manifests with the canary form of earlier development builds, `{seed, y_sha256}`, are
  refused: export them again.
- The Rust host shares one cuDNN handle per CUDA context between all its convolutions, and opens libcudnn only for a manifest that
  has one; plans run convolution steps with the arena's workspace and capture them.

### Kernels and ops

- TMA tensor descriptors: an extern of a Triton or Gluon kernel declares `TensorDesc(Tensor[...] | Strided[...], block=...,
  swizzle=..., l2_promotion=..., oob_fill=...)`; both hosts check the tensor and encode its `CUtensorMap` at every launch and at every
  execution of a plan (manifest kind `tensordesc`, section 5.5). `global_scratch` stays refused.
- Warp-specialised kernels (Gluon's `gl.warp_specialize`, Triton's own warp specialisation): `num_warps` is the warps of the
  default partition, the manifest records `total_num_warps`, and every launch takes the block of all the partitions (section 4.2).
- `tilefish.im2col1d(x, kernel, stride=, dilation=, padding=, order="channel" | "tap")`: the patches of a 1-D convolution, with
  causal and asymmetric padding, as a generated op.
- `tilefish.col2im1d(cols, kernel, ...)`: the overlap-add of a 1-D transposed convolution, in the order of adds of ATen's CUDA
  col2im (F.fold's bytes), with a bias and a channel-last output.
- Strides and offsets that are products of symbolic sizes (`tilefish.dims.mul`, `L*L` for an `[H, L, L]` tensor), and an elementwise
  broadcast along any axis (`tilefish.iota` along any axis; numerics note 12).
- `math="ieee"` on a `@tilefish.kernel` or on one op (`tilefish.div`, `exp`, `sqrt`, `rsqrt`, `sin`, `silu`): CUDA's f32 math without
  fast math or flush to zero, equal to torch's CUDA ops on every f32 input (numerics note 13).
- `tilefish.tanh`: libdevice's `tanhf`, with its numerics contract (note 14).

### Rust host

- Reads and runs every new entry form: GEMMs of every form and layout (`GemmLayout`), `sm_count_target`, `cudnn.conv_fwd`
  (`Manifest::conv`, `Conv`, `ConvEntry`, the canary), `tensordesc` parameters and `total_num_warps` (`Plan::block()` is public).
- `tilefish.hosts.rust.emit` writes launchers for these entries; the binaries of one stem no longer need one block.

## tilefish 0.0.5, crate 0.0.3

Lowered kernels that compare, select, gather and scatter rows and reduce to an index; Gluon externs and an implementation per
architecture (`by_arch=`); GEMM algorithms chosen as torch chooses them or imported (`AlgoTable`); plan sets that share one manifest
and one arena (`PlanSet`, `SharedArena`).

## tilefish 0.0.4, crate 0.0.2

Manifest schema 3: strided views, device scalars, prebuilt binaries, pinned cuBLASLt GEMMs, launch plans and CUDA graph capture, and
the Rust host's `LaunchPlan` and `CapturedPlan`.
