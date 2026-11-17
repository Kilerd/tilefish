# What tilefish is, and what it is not

tilefish is a typed DSL for GPU inference. You describe tensors and operators with types (shape, dtype, layout,
quantization format), build a model once into GPU binaries plus a manifest, and run that same artifact from Python,
Rust or C/C++.

## The core

1. **Types.** Every tensor carries its shape, dtype, layout and quantization format; operators have typed signatures.
2. **Four declarations.**
   - `@tilefish.kernel`: small kernels that tilefish lowers itself.
   - `@tilefish.extern`: kernels written elsewhere. Triton, Gluon, CuTe DSL, and prebuilt PTX or cubin all come in this way.
   - `@tilefish.library`: pinned library calls (cuBLASLt GEMMs, cuDNN convolutions).
   - `@tilefish.graph`: a model step as a plan over these operators.
3. **Build once, pin the result.** Cubins, the manifest and the plans are exported once and checked by digests. An
   export is reproducible from any directory, and on another machine with the same toolchain and GPU architecture.
4. **Thin hosts.** Python, Rust and C/C++ load the same artifact and launch it the same way. No host re-derives anything.
5. **Numerics written down.** Rounding points, accumulation types, reduction orders and what "exact" means are part
   of the contract. This is what lets a model be migrated byte for byte from an existing implementation.

Supporting tools sit around the core: `tilefish-explain` with SASS checks, probes, the fusion advisor, and the exact
fusion of graph-body arithmetic.

## Not goals

- **Competing with cuBLASLt, cuDNN or CUTLASS on kernel speed.** Write high-performance kernels in Triton, Gluon or
  CuTe DSL and bring them in as externs. tilefish types them, pins them and runs them. Example kernels in
  `python/examples/` (for instance the persistent GEMM) are examples, not supported features.
- **A general optimizing compiler.** Fusion is limited to what can be proven exact, or to what a graph explicitly
  declares.
- **Benchmark suites.** Correctness gates (bytes and digests) are required for every change. Speed is measured end to
  end on a real model when a decision depends on it.

## How features are added

From 0.0.13 on, new features are added when a real model migration needs them, not ahead of that need.
