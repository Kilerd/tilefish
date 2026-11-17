# tilefish

**English** | [简体中文](README.zh-CN.md)

A typed tensor language for GPU inference. The shape, dtype, layout and
quantization format of every tensor live in its type, operators come from a
typed library, and a model is built once into GPU binaries plus a manifest.
Thin Python, C++ and Rust hosts are generated from the same signatures.

**Status: early preview (0.0.13 on PyPI, 0.0.11 on crates.io; the C++ host builds from this repository).** tilefish types GPU kernels
written in Triton, Gluon, CuTe DSL or PTX, pinned cuBLASLt and cuDNN calls and kernels it lowers itself, builds a model's graphs into launch plans
over one arena, and runs the same pinned artefact from Python, Rust and C++. What it is and is not, and how features are added from now on, is in
[`docs/en/scope.md`](docs/en/scope.md); the other documents are listed in [`docs/en/README.md`](docs/en/README.md).

## Why

In inference code the contract of a kernel is usually written three times and
checked by none of them:

- the shape, dtype, alignment and aliasing rules sit in a docstring, a few
  `assert`s and a comment above a C ABI;
- a block-scaled weight format (MXFP8, MXFP4, NVFP4) is described in prose,
  together with the per-architecture byte layout of its scales;
- the same kernel is ported by hand from Triton to CUDA C++ to Rust, and parity
  tests chase the last ULP (FMA contraction, flush-to-zero, reduction order)
  after the fact;
- which implementation runs at which `(M, N, K)` on which GPU is a table
  someone measured once.

tilefish moves all of that into types, so the compiler can check it, generate
it and reason about it.

## What it looks like

```python
import tilefish
from tilefish import Dim, Tensor, InOut, Out, BlockScaled, const
from tilefish import bf16, f32, e4m3, e2m1, ue8m0

M = Dim("M")                          # runtime dimension
D = Dim("D", 2560)                    # static: baked into the binary
H = Dim("H", 1024)
F = Dim("F", 3072)
K = Dim("K", divisible_by=32)         # constraint, checked at build time
N = Dim("N", divisible_by=128)

mxfp8 = BlockScaled(elem=e4m3, scale=ue8m0, block=32, axis=-1)
mxfp4 = BlockScaled(elem=e2m1, scale=ue8m0, block=32, axis=-1)
nvfp4 = BlockScaled(elem=e2m1, scale=e4m3, block=16, global_scale=f32)

# A kernel whose body tilefish understands: it can be fused, and it is
# lowered to Triton.
@tilefish.kernel
def residual_add_rmsnorm(
    x: Tensor[M, D, bf16],
    residual: InOut[Tensor[M, D, f32]],
    w: Tensor[D, bf16],
    *, eps: const[f32] = 1e-6,
) -> Tensor[M, D, bf16]:
    r = residual + x.to(f32)
    residual[...] = r
    rstd = tilefish.rsqrt(tilefish.mean(r * r, axis=-1, keepdims=True) + eps)
    return (r * rstd * w.to(f32)).to(bf16)      # the only rounding point

# An existing hand-written kernel, unchanged, given a type.
n = Dim("n", divisible_by=16)

# (index_bits=32 is the default: a kernel that offsets with i32 gets a max_elements
# limit of 2**31 - 1 on every tensor, which both hosts check; a kernel that
# indexes in i64 says index_bits=64.)
@tilefish.extern(triton=my_kernels.gelu, num_warps=4,
                 grid=lambda n: (tilefish.cdiv(n, 1024),))
def gelu(x: Tensor[n, bf16], y: Out[Tensor[n, bf16]],
         n_elem: n, BLOCK: const = 1024): ...

# A library operator: opaque body, typed signature, pluggable epilogue.
@tilefish.library("gemm", acc=f32)
def gemm(a: Tensor[M, K, bf16], b: Tensor[N, K, mxfp8]) -> Tensor[M, N, bf16]: ...

# A model is ordinary Python over typed values.
@tilefish.graph
def mlp(x: Tensor[M, H, bf16], w_gu: Tensor[2 * F, H, mxfp8], w_d: Tensor[H, F, mxfp8]):
    g, u = gemm(x, w_gu).split(F, axis=-1)
    a = (tilefish.silu(g.to(f32)) * u.to(f32)).to(bf16)
    return gemm(a, w_d)
```

## How it works

1. **Declare.** Types (`Dim`, `Tensor`, `BlockScaled`, `DevScalar`, views) and four declarations: `@tilefish.kernel` (a body
   tilefish lowers to Triton itself), `@tilefish.extern` (a Triton, Gluon or CuTe DSL kernel, or prebuilt PTX or cubin, given a
   type), `@tilefish.library` (a pinned cuBLASLt GEMM or cuDNN convolution) and `@tilefish.graph` (a model step over them).
2. **Build once.** `tilefish.runtime.export` traces and checks the graph, compiles every kernel ahead of time, pins every library
   algorithm, and writes the cubins, `manifest.json` and the launch plans (`plan.json`: every launch of a step over one arena).
   The manifest's digest covers every binary and pin, the plan's digest the plan and its manifest; a host pins one, and an
   export is reproducible from any directory and on another machine with the same toolchain and GPU architecture.
3. **Run the same artefact.** Python (`replay`), the Rust crate and the C/C++ library load the directory, verify it against
   the pin and check every precondition before a launch. `jit` runs the same machine code built in memory, `reference` the
   same bodies in torch.
4. **Look inside.** `tilefish-explain` maps each SASS instruction to its source line, and SASS assertions are checked at every
   build; probes time the phases inside a kernel; `tilefish-explain --advise` ranks the steps of a plan that could be one kernel.

The details of each stage are in the [guide](docs/en/README.md#guide).

## Install

```sh
pip install tilefish                  # types, tracing, lowering, compile specs: pure Python, no dependencies
pip install "tilefish[gpu]"           # torch, Triton and cuda-python: build, jit, replay, plans
pip install "tilefish[sass]"          # CUDA 13's nvdisasm, for tilefish-explain and SASS assertions on sm_103
cargo add tilefish                    # the Rust host
cmake -S crates/tilefish-capi -B build && cmake --build build   # the C/C++ host, from this repository
```

Building for sm_103 needs a newer `ptxas` than the one Triton 3.5.1 bundles
(CUDA 12.8's, which cannot target `sm_103a`): set `TRITON_PTXAS_PATH` to the
`ptxas` of CUDA 12.9 or newer. A build that fails for this reason says so, and
the GPU tests that build for an architecture the `ptxas` cannot target skip,
naming it.

## Quick start

### The Python host

```python
# x, residual, w: torch tensors on the GPU (residual, an InOut, is updated in place)
y = tilefish.run(residual_add_rmsnorm, x, residual, w)                        # jit: built in memory
m = tilefish.runtime.export(residual_add_rmsnorm, "build/")                    # cubins and manifest.json
y = tilefish.run(residual_add_rmsnorm, x, residual, w, mode="replay", manifest="build/", pin=m.digest)
y = tilefish.run(residual_add_rmsnorm, x, residual, w, mode="reference")      # the same body in torch
```

A `@tilefish.graph` exports its launch plan next to the manifest, and `tilefish.run(graph, ..., mode="replay", plan="build/",
pin=p.digest)` replays it ([build](docs/en/guide/build.md), [plans](docs/en/guide/plans.md)).

### The Rust host

```rust
let manifest = Manifest::load("build/", Pin::Digest("<manifest digest>"))?;
let kernel = manifest.kernel(&ctx, "gelu_erf", &[], Arch::of_context(&ctx)?)?;
kernel.launch(&stream, &mut [x.into_arg(), y.into_arg()])?;     // every precondition is checked first

let plan = LaunchPlan::load("build/", Pin::Digest("<plan digest>"))?;
let run = plan.instantiate_all(&stream)?;
run.launch(&stream, &mut [("x", x.into_arg()), ("out0", y.into_arg())], &[("n", DevScalar::Int(64))])?;
```

`tilefish.hosts.rust.emit` writes typed launchers for a pinned manifest ([guide](docs/en/guide/rust.md),
[`crates/tilefish/README.md`](crates/tilefish/README.md)).

### The C++ host

```cpp
#include "tilefish.hpp"

auto ctx = tilefish::Context::primary(0);
auto run = tilefish::LaunchPlan::load("build/", "<the plan's digest>").instantiate(ctx, stream);
run.launch(stream, {{"x", xt}, {"out0", yt}}, {{"n", 64}});
```

`find_package(tilefish)` after `cmake --install`; `tilefish.hosts.cpp.emit` writes typed launchers and plan runners
([guide](docs/en/guide/cpp.md)).

## Features at a glance

| feature | documented in |
|---|---|
| What tilefish is and is not, and how features are added | [`docs/en/scope.md`](docs/en/scope.md) |
| The build: the steps, `jit`, `reference` and `replay`, pins, reproducible exports, what identifies a binary, the build cache, the TF32 guard, trust | [`docs/en/guide/build.md`](docs/en/guide/build.md) |
| Strided views, device scalars, prebuilt PTX and cubin, Gluon kernels, one implementation per architecture, TMA descriptors, warp-specialised kernels, launch shapes | [`docs/en/guide/operators.md`](docs/en/guide/operators.md) |
| Pinned cuBLASLt GEMMs (bf16, f16, f32 with TF32 off; four layouts; SM partitions; epilogues) and pinned cuDNN convolutions | [`docs/en/guide/operators.md`](docs/en/guide/operators.md) |
| CuTe DSL kernels as externs | [`docs/en/cute.md`](docs/en/cute.md) |
| Lowered kernels: masks, indices, `argmax`, gathers and scatters, integer casts, IEEE math, ordered sums, broadcasts, `contiguous`, `im2col`, `col2im1d` | [`docs/en/guide/lowering.md`](docs/en/guide/lowering.md) |
| Launch plans: one arena, CUDA graphs, copy and memset steps, weights bound once, plan sets, shared arenas, carried buffers, debug taps | [`docs/en/guide/plans.md`](docs/en/guide/plans.md) |
| Fusion of a graph body's arithmetic, epilogue hooks on externs, cuBLASLt epilogues | [`docs/en/guide/fusion.md`](docs/en/guide/fusion.md) |
| Rounding points, exact and numerics-changing rewrites, `num_warps` and FMA contraction | [`docs/en/guide/rewrites.md`](docs/en/guide/rewrites.md) |
| The CUDA math guard and what to check again on a CUDA or torch change | [`docs/en/numerics.md`](docs/en/numerics.md) |
| Traces, `tilefish-explain` and SASS assertions | [`docs/en/guide/sass.md`](docs/en/guide/sass.md), [`docs/en/explain.md`](docs/en/explain.md) |
| The fusion advisor and the plan profiler | [`docs/en/advise.md`](docs/en/advise.md) |
| Probes and the instrumented variant | [`docs/en/instrument.md`](docs/en/instrument.md) |
| The Rust host and generated Rust launchers | [`docs/en/guide/rust.md`](docs/en/guide/rust.md), [`crates/tilefish/README.md`](crates/tilefish/README.md) |
| The C ABI, the C++ wrapper and generated C++ launchers | [`docs/en/guide/cpp.md`](docs/en/guide/cpp.md), [RFC 0002](docs/en/rfcs/0002-cpp-host.md) |
| The files a host reads | [`docs/en/manifest-schema-3.md`](docs/en/manifest-schema-3.md), [`docs/en/plan-schema-1.md`](docs/en/plan-schema-1.md) |

## What tilefish is not

- **Not a kernel compiler.** Triton, CuTe DSL, cuTile, Helion and TileLang
  write kernels. tilefish types them, builds them and hosts them.
- **Not a serving runtime.** Scheduling, KV cache, batching and request
  handling stay in the host program. tilefish owns one forward step over
  buffers it has planned.
- **Not a promise to generate every kernel.** Hand-tuned attention and GEMMs
  enter as `extern` or library operators and keep their own numerics
  contract.

## Status and versions

Early preview: the API will change.

| package | version | from |
|---|---|---|
| `tilefish` (Python) | 0.0.13 | PyPI |
| `tilefish` (the Rust host) | 0.0.11 | crates.io |
| `tilefish-capi` (C ABI 1.1, `tilefish.h` and `tilefish.hpp`) | 0.0.2 | this repository |

The manifest is schema 3 and the plan `tilefish.plan.v1`. There is no list of features to come. A feature is added when a real
model migration needs it (`docs/en/scope.md`); the open items are tracked in issue [#4](https://github.com/Kilerd/tilefish/issues/4),
and [#5](https://github.com/Kilerd/tilefish/issues/5) (SASS traceability, probes, CuTe DSL externs) is closed. What each release
changed is in [`CHANGELOG.md`](CHANGELOG.md), and every document is listed in [`docs/en/README.md`](docs/en/README.md).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
