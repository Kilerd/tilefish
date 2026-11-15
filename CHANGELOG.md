# Changelog

Versions are the Python package `tilefish` on PyPI and the Rust crate `tilefish` on crates.io. The manifest is schema 3 and the
plan is `tilefish.plan.v1` in both releases below; `docs/manifest-schema-3.md` and `docs/plan-schema-1.md` specify every field, and
each section names the release that added it.

## Unreleased

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

## tilefish 0.0.7, crate 0.0.5 (unreleased)

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

### Plan fusion: arithmetic in a graph body (`docs/rfcs/0001-plan-fusion.md`, phase 1)

- **A graph body may compute.** Arithmetic traced in a `@tilefish.graph` body (casts, `+ - * /`, comparisons and `where`, math functions,
  `iota` and `load`, reductions) is planned instead of refused: the planner groups it into regions, each a `@tilefish.kernel` that tilefish
  generates (`fused_<hash>`, named by its content, so the same region in two graphs or two plans of a set is one manifest entry) and one kernel
  step of the plan. Plan schema 1, manifest schema 3 and both hosts are unchanged; a graph without arithmetic plans exactly as before.
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
  Fused regions avoid both (and so does every kernel now: "Unreleased", "Fixed").

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
