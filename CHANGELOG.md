# Changelog

Versions are the Python package `tilefish` on PyPI and the Rust crate `tilefish` on crates.io. The manifest is schema 3 and the
plan is `tilefish.plan.v1` in both releases below; `docs/manifest-schema-3.md` and `docs/plan-schema-1.md` specify every field, and
each section names the release that added it.

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
