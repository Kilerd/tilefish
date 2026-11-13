# tilefish (Rust host)

The Rust host of a [tilefish](https://github.com/Kilerd/tilefish) manifest.
tilefish builds kernels once into CUDA binaries plus a `manifest.json`. This
crate loads that directory and launches its cubins from Rust, with no Python
and no Triton, making every check Python's `replay` makes. The machine code is
the exact bytes Python's `replay` runs, and `tilefish.hosts.rust.emit`
generates typed launchers for a manifest.

**Status: early preview.** The API will change.

```rust
use tilefish::cudarc::driver::CudaContext;
use tilefish::{Arch, Bf16, Manifest, Pin, TensorMut, TensorRef};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !tilefish::driver_available() {
        return Ok(()); // CudaContext::new panics without an NVIDIA driver
    }
    // Production pins the digest that `tilefish.runtime.export` returned.
    let manifest = Manifest::load("build/", Pin::Digest("<manifest digest>"))?;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;
    let kernel = manifest.kernel(&ctx, "gelu_erf", &[], Arch::of_context(&ctx)?)?;

    let (x, mut y) = (stream.alloc_zeros::<u8>(4096 * 2)?, stream.alloc_zeros::<u8>(4096 * 2)?);
    let shape = [4096usize];
    let x = TensorRef::<Bf16>::from_cuda(&x, &shape)?;
    let y = TensorMut::<Bf16>::from_cuda(&mut y, &shape)?;
    kernel.launch(&stream, &mut [x.into_arg(), y.into_arg()])?; // every precondition is checked first
    Ok(())
}
```

These checks happen before anything reaches the driver:

- **At load:** the manifest is strict JSON of schema 3 (an older one is refused:
  export it again), its digest is the pinned one, every cubin matches its
  `sha256` and machine-code digest and is for the architecture its entry says,
  and the loaded function takes the parameter sizes of the ABI.
- **At launch:** the shapes bind the dimensions (and the strides of a view bind
  its stride dimensions), those of the tensors the kernel reads first, then the
  shape of an `Out` tensor for a dimension only it has (a `[B]` mask made from a
  device scalar); dtypes, buckets, `divisible_by`, `max_elements` and
  `max_extent`, aliasing, the alignment each pointer promises, contiguity,
  device and context, and the grid limits are all checked.

Everything schema 3 describes is launched: Triton kernels and prebuilt PTX and
cubin binaries (launched with exactly their parameters, `fixed` arguments
written at their slots), strided views (`TensorRef::from_cuda_strided`), device
scalars (a tensor of one element), `i64` strides, and pinned cuBLASLt GEMMs
(`Manifest::gemm`, `Gemm::launch`: the cuBLASLt of the process must be the version
that chose the algorithm, and the workspace is the caller's last tensor, so
nothing is allocated in a launch and it can be captured in a CUDA graph). An
entry says how its algorithm was chosen (`LibraryEntry::heuristic`,
`algo_source`: the heuristic under tilefish's or torch's posture, or imported),
and an algorithm whose split-K adds its partial sums in bf16 is run only for an
entry that opts in (`allow_bf16_reduction`).
cudarc opens `libcublasLt.so.12` by name: where cuBLASLt comes from a Python
wheel (`nvidia-cublas`, torch's), put that wheel's `nvidia/cublas/lib` on
`LD_LIBRARY_PATH`. A missing library is an error, and another build of it is
refused by the version check.

A whole forward pass runs from a **launch plan** (`plan.json`, which
`tilefish.runtime.export` writes next to the manifest for a `@tilefish.graph`):
`LaunchPlan::load` verifies it (a pin, the digests, every rule of
`docs/plan-schema-1.md`, the manifest it was built against, every step against
its entry), `instantiate` loads its binaries and allocates its one arena (the
intermediates, the device scalars, the cuBLASLt workspace), and
`PlanRun::launch` binds the caller's tensors by name, checks every step before
the first launch and launches them in order, allocating nothing.
`PlanRun::capture` captures it in a CUDA graph: the `CapturedPlan` owns the
arena and borrows the tensors it captured, so its safe `replay` cannot run on
memory that was freed; new inputs are written through it between replays.

```rust
let plan = LaunchPlan::load("build/", Pin::Digest("<plan digest>"))?;
let run = plan.instantiate_all(&stream)?;
run.launch(&stream, &mut [("x", x.into_arg()), /* the weights, */ ("out0", y.into_arg())], &[("n", DevScalar::Int(64))])?;
```

The externals are given by the names of the plan's buffers: a parameter by its name, a returned value by the name the graph gives it
(`@tilefish.graph(outputs=("y", ...))` in Python, `LaunchPlan::outputs`) or by `out0`, `out1`, ... in the order the graph returns them
(`LaunchPlan::output_buffer`), so a host written before graphs named their outputs keeps working. A plan whose graph declares its roles
(`@tilefish.graph(inputs=..., weights=...)`, `LaunchPlan::roles_declared`) has its weights bound once: `PlanRun::bind_weights` checks them and
returns a `WeightedRun`, whose `launch`, `launch_tapped`, `check` and `capture` take the inputs, the `inout` tensors and the outputs only (the
weights are borrowed for the life of the `WeightedRun`). `PlanRun::launch_tapped` is a launch with debug
taps: each `Tap` copies a buffer of the plan into a `TensorMut` right after a step, with the plan and its digest unchanged. `PlanRun::launch_step`
launches one step of a bucket on the caller's tensors, with the binary and the binding of the pinned plan.

Safe code cannot free a slice under a running kernel or GEMM, and a launch of
slices on a stream that is capturing a CUDA graph is refused, in every cudarc
context. The per-thread default stream (`ctx.per_thread_stream()`) is refused
by every safe launch: its one handle names another stream in every thread, so
a launch on it cannot be ordered against other threads' uses of a slice; use
`ctx.new_stream()` or `ctx.default_stream()`. `unsafe` is needed in exactly two places: vouching for a raw device
region, which is also how CUDA graphs are captured (with `launch_raw`), and
launching on a raw `CUstream`. See the crate docs for the exact contracts.

**In a workspace that uses cudarc itself.** The crate needs cudarc 0.19.10 or a
later 0.19 release: it hands the driver the function a `CudaFunction` holds
(`CudaFunction::cu_function`, to check its parameter sizes, to load it eagerly
and to launch it on a raw stream) and instantiates captured graphs with no flags
(`CUgraphInstantiate_flags(0)`), both of which cudarc 0.19.10 added; 0.19.9 does
not compile it. cudarc builds its bindings for exactly one `cuda-*` feature,
and this crate enables `cuda-13020` by default: a workspace that picks its own
must depend on the crate with the same one, or cudarc is built with two and does
not compile:

```toml
cudarc = { version = "0.19.10", default-features = false, features = ["driver", "cuda-13000"] }
tilefish = { version = "0.0.2", default-features = false, features = ["cuda-13000"] }
```

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
