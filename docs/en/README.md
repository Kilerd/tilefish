# Documentation

**English** | [简体中文](zh-CN/README.md)

* [`scope.md`](scope.md): what tilefish is and is not, and how features are added.
* [`../README.md`](../README.md): the overview, the install and a quick start for each host.
* [`zh-CN/README.md`](zh-CN/README.md): this index in Simplified Chinese, which lists the Chinese pages.

## Guide

The reference of each stage, from a declaration to a launch from a host:

* [`guide/build.md`](guide/build.md): how a build works: its steps, the `jit`, `reference` and `replay` modes, pins and reproducible exports,
  what identifies a binary (its Triton IR), the build cache, what `replay` refuses, the TF32 guard, trust, and what CI checks on a GPU.
* [`guide/operators.md`](guide/operators.md): views, device scalars, prebuilt binaries and pinned library calls: strided views and shared strides,
  views that do not start on 16 bytes, device scalars, a dimension only an `Out` tensor has, prebuilt PTX and cubin, null pointers and 64-bit
  dimension scalars, Gluon kernels, one implementation per architecture, TMA descriptors, warp-specialised kernels, launch shapes, pinned cuBLASLt
  GEMMs (f16, f32 with TF32 off, the four layouts, SM partitions, cuBLASLt epilogues), pinned cuDNN convolutions and CuTe DSL kernels.
* [`guide/lowering.md`](guide/lowering.md): masks, indices, gathers and copies in a lowered `@kernel`, with their numerics: comparisons and
  selects, `iota`, `argmax`, `gather_rows` and `scatter_rows`, integer casts, IEEE math, the primitives for reproducing a CUDA kernel, the CUDA
  math guard, ordered sums, broadcasts, `contiguous`, `im2col`, `im2col1d` and `col2im1d`.
* [`guide/plans.md`](guide/plans.md): launch plans: a graph as a static list of launches over one arena, CUDA graphs, an `Out` written over its
  input, copy and memset steps, returned views, named outputs, inputs and weights, plan sets, an arena shared by the plans of a set, carried
  buffers, debug taps and one step alone.
* [`guide/fusion.md`](guide/fusion.md): fusion of the arithmetic in a graph body, its modes, epilogue hooks on externs and cuBLASLt epilogues.
* [`guide/rewrites.md`](guide/rewrites.md): rewrites that know about rounding: exact and numerics-changing rules, `num_warps` and FMA contraction.
* [`guide/sass.md`](guide/sass.md): SASS explained and asserted: traces, `tilefish-explain`, SASS assertions, the advisor and probes in short.
* [`guide/rust.md`](guide/rust.md): the Rust host, `tilefish.hosts.rust.emit` and the byte-identity gate.
* [`guide/cpp.md`](guide/cpp.md): the C++ host (the C ABI and `tilefish.hpp`), `tilefish.hosts.cpp` and the C++ half of the gate.

## Reference

* The formats a host reads:
  * [`manifest-schema-3.md`](manifest-schema-3.md): the manifest (kernel entries, the ABI, prebuilt binaries, pinned library calls);
  * [`plan-schema-1.md`](plan-schema-1.md): launch plans and plan sets.
* [`numerics.md`](numerics.md): the numerics contract's dependencies, and how to check it again after a CUDA or torch change.
* [`cute.md`](cute.md): CuTe DSL kernels as externs (the `nvidia-cutlass-dsl` pin and its licence).
* Tools:
  * [`explain.md`](explain.md): traces, `tilefish-explain` and SASS assertions;
  * [`instrument.md`](instrument.md): probes and the instrumented variant;
  * [`advise.md`](advise.md): the fusion advisor.
* The Rust crate's own page: [`../crates/tilefish/README.md`](../crates/tilefish/README.md).

## Design records and history

* [`rfcs/`](rfcs): [0001, plan fusion](rfcs/0001-plan-fusion.md); [0002, the C++ host](rfcs/0002-cpp-host.md);
  [0003, epilogue and prologue fusion](rfcs/0003-epilogue-fusion.md).
* [`../CHANGELOG.md`](../CHANGELOG.md): what each release changed.
