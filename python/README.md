# tilefish

**English** | [简体中文](https://github.com/Kilerd/tilefish/blob/main/python/README.zh-CN.md)

A typed tensor language for GPU inference: shapes, dtypes, layouts and
quantization formats in the type, one build into GPU binaries, and generated
Python, C++ and Rust hosts.

**Status: early preview.** This release type-checks: tensor types and
block-scaled formats (MXFP8, MXFP4, NVFP4), `extern` / `library` / `kernel` /
`graph` declarations, tracing into a typed IR, and Triton ahead-of-time
compile specs for `extern` kernels. Pure Python, no dependencies. A `@kernel`
also lowers to Triton source (`tilefish.lower`), and with the `gpu` extra
(`pip install "tilefish[gpu]"`: torch, Triton, cuda-python) a spec builds into
a cubin (`tilefish.aot.build`), a manifest directory holds the cubins
(`tilefish.runtime.export`), and a kernel runs in `jit` (built through its
spec, in memory), `replay` (the manifest's cubins, checked against its
digests; a stale binary is refused, and it needs a trust anchor: the manifest's
pinned digest, or an explicit `trust_directory=True`) and `reference` mode
(`tilefish.run`). A binary is identified by the Triton IR it compiles. A
library GEMM runs a pinned cuBLASLt algorithm, and a `@tilefish.graph`
compiles into a launch plan over one arena (`tilefish.plan`, written to
`plan.json` by the export) that runs in the same three modes and captures in
a CUDA graph (`tilefish.capture`). Lowered kernels also compare, select,
gather and scatter rows, reduce to an index, broadcast along any axis and may
use IEEE f32 math (`math="ieee"`) and `tanh`; `im2col1d` and `col2im1d` are
generated ops for 1-D convolutions; an extern may be a Gluon kernel, have one
implementation per architecture, take TMA tensor descriptors (`TensorDesc`)
and be warp-specialised; a pinned GEMM takes bf16, f16 or f32 operands, an f32
result, the layouts `TN`, `NN`, `TT` and `NT` and an SM partition, and its
algorithm is chosen as torch chooses it or imported; a pinned cuDNN forward
convolution (`cudnn.conv_fwd`) is a library op too; no build makes an IEEE f32
dot into TF32 tensor-core MMAs; plans exported together share one manifest and
can share one arena. The `tilefish` crate on crates.io runs the
same manifest and plan from Rust, and `tilefish.hosts.rust.emit` generates
typed Rust launchers for it. The API will change.

What tilefish is and is not, and how features are added from 0.0.13 on, is in
[`docs/scope.md`](https://github.com/Kilerd/tilefish/blob/main/docs/scope.md).

## Install

```sh
pip install tilefish                  # types, tracing, lowering, compile specs: pure Python, no dependencies
pip install "tilefish[gpu]"           # torch, Triton and cuda-python: build, jit, replay, plans
pip install "tilefish[sass]"          # CUDA 13's nvdisasm, for tilefish-explain and SASS assertions on sm_103
```

## Documentation

- [The README](https://github.com/Kilerd/tilefish#readme): an example, how it works, a quick start for each host
  and the features at a glance.
- [The documents' index](https://github.com/Kilerd/tilefish/blob/main/docs/README.md), with the guide: [the build](https://github.com/Kilerd/tilefish/blob/main/docs/guide/build.md),
  [operators](https://github.com/Kilerd/tilefish/blob/main/docs/guide/operators.md), [lowered kernels](https://github.com/Kilerd/tilefish/blob/main/docs/guide/lowering.md),
  [launch plans](https://github.com/Kilerd/tilefish/blob/main/docs/guide/plans.md), [fusion](https://github.com/Kilerd/tilefish/blob/main/docs/guide/fusion.md),
  [rewrites](https://github.com/Kilerd/tilefish/blob/main/docs/guide/rewrites.md), [SASS](https://github.com/Kilerd/tilefish/blob/main/docs/guide/sass.md), [the Rust host](https://github.com/Kilerd/tilefish/blob/main/docs/guide/rust.md)
  and [the C++ host](https://github.com/Kilerd/tilefish/blob/main/docs/guide/cpp.md).
- [The changelog](https://github.com/Kilerd/tilefish/blob/main/CHANGELOG.md): what each release changed.

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
