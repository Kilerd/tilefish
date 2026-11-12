# tilefish

A typed tensor language for GPU inference. The shape, dtype, layout and
quantization format of every tensor live in its type, operators come from a
typed library, and a model is built once into GPU binaries plus a manifest.
Thin Python, C++ and Rust hosts are generated from the same signatures.

**Status: early preview (0.0.1 on PyPI).** The Python package type-checks:
tensor types and block-scaled formats, the four declarations below, tracing
into a typed IR, and Triton ahead-of-time compile specs for `extern` kernels,
in pure Python with no dependencies. On `main` (after 0.0.1) a `@kernel` also
lowers to Triton source, as a row kernel or an elementwise kernel
(`tilefish.lower`), with its numerics contract written down in the module.
With the `gpu` extra (`pip install "tilefish[gpu]"`: torch, Triton,
cuda-python) a compile spec builds into a cubin (`tilefish.aot.build`, with a
manifest and a digest of the machine code) and a kernel runs from Python in
the three modes below (`tilefish.run`). There are no generated hosts, and the
`tilefish build` command is still the plan. The Rust crate is a name
reservation.

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

## How a build works

```sh
tilefish build --arch sm_90,sm_100,sm_120 --host python,rust,cpp
```

1. Trace the graph into a typed IR and check it: shapes, dtypes, divisibility,
   aliasing, and which architectures each operator supports.
2. Run the rewrite rules (below) and settle the final list of kernels.
3. Lower each `@kernel` to Triton source, written to disk for review.
4. Derive Triton's compile inputs from the types: pointer dtypes and 16-byte
   alignment, `constexpr` values from static dimensions, divisibility hints
   from `divisible_by`, and one binary per bucket for bucketed dimensions.
   Compile ahead of time to one cubin per architecture.
5. Write a manifest: symbol, parameter ABI in order, grid rule, block size,
   shared memory, launch attributes, toolchain versions, and a digest of the
   machine code.
6. Generate the hosts from the manifest and the types: a `torch.library`
   custom op for Python, a header for C++, a `cudarc` crate for Rust. Every
   precondition that selected a binary (alignment, divisibility, static sizes)
   becomes a check in the generated launcher, so no host can run a binary
   under assumptions it was not compiled for.

Library operators put a different code object behind the same kind of
manifest entry: a CUTLASS instantiation with a C entry point, a CuTe DSL
ahead-of-time export, or a pinned cuBLASLt algorithm recorded with its
workspace size and replayed as is.

The Python side has three modes:

| mode | what runs | used for |
|---|---|---|
| `jit` | the current source, built through its compile spec (the path of the build) in memory and launched through the CUDA driver | iterating on a kernel |
| `reference` | the same bodies interpreted with torch | an oracle, compared within a tolerance |
| `replay` | the prebuilt cubins of a manifest directory, checked against its digests, through the CUDA driver; the directory is trusted unless you pin its digest | the exact machine code the C++ and Rust hosts run |

```python
y = tilefish.run(residual_add_rmsnorm, x, residual, w)                        # jit
m = tilefish.runtime.export(residual_add_rmsnorm, "build/")                    # cubins and manifest.json
y = tilefish.run(residual_add_rmsnorm, x, residual, w, mode="replay", manifest="build/")
y = tilefish.run(residual_add_rmsnorm, x, residual, w, mode="replay", manifest="build/", pin=m.digest)
```

Arguments are torch tensors, matched to the signature the way a call is
matched in a trace: inputs and `InOut` tensors are passed (an `InOut` tensor
is updated in place), `Out` tensors are allocated and returned. Before a
launch every precondition of the binary is checked (dtypes, static sizes,
`divisible_by`, 16-byte aligned pointers, contiguity), and a bucketed
dimension must equal a bucket exactly: there is no padding.

`jit` and `replay` launch the same way, so they run the same machine code: a
cubin that `jit` builds has the `sha256_code` of the one in the manifest.
`replay` refuses a cubin whose digests are not the manifest's, a manifest
entry that is not what the operator's compile spec asks for, and a stale
binary: a kernel redeclared with another body but the same name and signature
does not replay its old binary. A generated kernel's spec carries the sha256 of
its Triton source. A hand-written `extern` has no source in its declaration,
so its manifest entry records a *source digest*: a sha256 over the text of the
file of the kernel's module, of every user module file reachable from that
module's globals (the modules it imports, and the modules of the functions and
objects it holds, recursively; not `triton`, `torch`, `tilefish` or the
standard library) and the Triton version. `replay` recomputes it, which needs
the kernel to be importable, or a pin. Triton's own `cache_key` is not used: it
does not see a `tl.constexpr` read as a module attribute (`cfg.SCALE`), a dtype
alias or a constant in a helper module, so none of them would make a stale
binary look stale. For the same reason every `jit` and `export` build compiles
from scratch and never reads Triton's disk cache (about 35 to 130 ms per binary
on a B300, once per process for `jit`).

**Any edit to a file a hand-written kernel reaches, a comment included, means
exporting again and pinning the new manifest.** The digest is of the files on
disk, so a process that edited a file after importing it must reload before it
builds (a build refuses a function whose imported lines are not the file's), and
it does not see a value that a program assigned to a module attribute at run
time. `build` also refuses source that is not its spec's: a stale
`LoweredKernel.write` output on `sys.path`, or another function passed as `fn`.

`replay` also refuses `shared` memory the cubin and the device cannot have (the
driver's static shared memory of the function plus the manifest's dynamic size
must fit the device's opt-in maximum), an argument list that does not fit the
ABI, and a grid beyond the device's limits. The manifest is read as strict JSON
(no duplicate keys, no `NaN`, no number that does not fit a float), files are
written through a temporary file and renamed, and a symbolic link where a file
is to go is refused.

**Trust.** Without `pin=` a replay trusts the directory: the digests sit next
to the files they describe, so whoever can write the directory can rewrite
both. `export` returns the manifest with its `digest`, a sha256 over every
entry, and checks that the directory it wrote has that digest; keep it somewhere
you trust and pass it as `pin=` to refuse a directory that changed. A pin
vouches for the manifest, not for a source digest that is known to differ, and
not for a `shared` that is nonzero but too small for the kernel (nothing but the
manifest says how much it needs). Triton's own JIT launcher, which specializes
on argument values, is not a mode. Neither `jit` nor `replay` needs a C
compiler, and a `replay` of a generated kernel needs neither Triton nor `ptxas`.

CI asserts, on a GPU (the tests marked `gpu`), that a `jit`-built cubin has the
digest of the manifest's, that `jit` and `replay` agree byte for byte, and that
`replay` stays within tolerance of `reference` (tolerances per input
distribution, in the test). An experiment that passes in `replay` is, kernel
for kernel, what production runs.

## Rewrites that know about rounding

Types make fusion legality mechanical: elementwise chains into GEMM
epilogues, a norm into the preceding residual add, quantization into the next
GEMM's prologue, permutations into strides instead of copy kernels, in-place
reuse of buffers. Bytes and FLOPs per operator follow from shapes and dtype
widths, so a build can report a roofline per kernel and count launches per
step.

Fusion can also change numerics. Every cast to a narrower type is a rounding
point in the IR, and rules come in two kinds:

- **exact**: every rounding point is kept and only kernel boundaries move.
  Applied automatically.
- **numerics-changing**: a reduction order, an accumulator dtype or a rounding
  point changes. Applied only on request, and gated against the reference.

Whether a legal fusion is worth doing is still measured, not assumed. The
measurements live in the repository next to the rules.

## What tilefish is not

- **Not a kernel compiler.** Triton, CuTe DSL, cuTile, Helion and TileLang
  write kernels. tilefish types them, builds them and hosts them.
- **Not a serving runtime.** Scheduling, KV cache, batching and request
  handling stay in the host program. tilefish owns one forward step over
  buffers it has planned.
- **Not a promise to generate every kernel.** Hand-tuned attention and GEMMs
  enter as `extern` or library operators and keep their own numerics
  contract.

## Roadmap

- **V0: one real model, end to end.** The whole GPU forward of a production
  model of about 0.6B parameters (a convolutional front end, a bidirectional
  encoder and a causal GQA decoder) is written in tilefish. The generated Rust
  host replaces the hand-written launch code of an existing engine and
  reproduces its output byte for byte on every architecture it ships for.
  Starting from a naive version of the model, the rewrite rules rediscover the
  fusions that were made by hand. V0 needs typed signatures and manifests,
  `extern` for Triton functions and prebuilt PTX or cubins, strided views,
  device-side scalars, gather / scatter / im2col style index kernels, a static
  launch and memory plan per shape bucket, pinned GEMMs, and a numerics
  contract on every operator.
- **V1: quantization as types.** MXFP8, MXFP4 and NVFP4, with generated
  quantize and dequantize references and per-architecture scale layouts.
- **V2: more of the forward generated.** Row reductions and elementwise
  families lowered natively, and GEMM epilogue fusion through CUTLASS or
  CuTe DSL.
- **V3: C++ and Rust as first-class deploy targets**, with the Python
  frontend used only at build time.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
