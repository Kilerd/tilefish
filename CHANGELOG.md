# Changelog

Versions are the Python package `tilefish` on PyPI and the Rust crate `tilefish` on crates.io. The manifest is schema 3 and the
plan is `tilefish.plan.v1` in both releases below; `docs/manifest-schema-3.md` and `docs/plan-schema-1.md` specify every field, and
each section names the release that added it.

## Unreleased

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
  `tilefish.runtime.warm_cudnn(device)` (Python) make the handle at start-up and keep it, so that no load makes one.
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
