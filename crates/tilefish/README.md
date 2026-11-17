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
written at their slots, and the values of the `given` ones, fixed per call,
passed by `Kernel::launch_with(stream, tensors, &[("h_out", 5)])` or a plan
step), null pointers (kind `null`: 0 at their slot, no tensor), strided views
(`TensorRef::from_cuda_strided`), device scalars (a tensor of one element), `i64`
strides and dimension scalars, dense pointers that promise only their element's
alignment (a lowered kernel's specialisation for views), and pinned cuBLASLt GEMMs
(`Manifest::gemm`, `Gemm::launch`: the cuBLASLt of the process must be the version
that chose the algorithm, and the workspace is the caller's last tensor, so
nothing is allocated in a launch and it can be captured in a CUDA graph). An
entry says how its algorithm was chosen (`LibraryEntry::heuristic`,
`algo_source`: the heuristic under tilefish's or torch's posture, or imported),
and an algorithm whose split-K adds its partial sums in bf16 is run only for an
entry that opts in (`allow_bf16_reduction`). A GEMM may have f16 operands, an
f32 result, `b` stored `[K, N]` and `a` stored `[K, M]` (`LibraryEntry::dtype`,
`out_dtype`, `layout`: `GemmLayout::Tn`, `Nn`, `Tt` or `Nt`); its tensors are checked against those
types and shapes, and an f16 entry opts in to f16 partial sums with
`allow_f16_reduction`. A GEMM of f32 operands (and an f32 result) is multiplied
as f32, TF32 off: the crate refuses its algorithm where the library would run
it as TF32 or as a bf16x9 emulation (an `NVIDIA_TF32_OVERRIDE=1` or
`CUBLAS_EMULATE_SINGLE_PRECISION=1` process), and an f32 entry opts in to
split-K partial sums in the output with `allow_f32_reduction`
(`LibraryEntry::allow_f32_reduction`). An entry pinned for a partition of the device's SMs
(`sm_count_target`) sets that target on every launch, and a host that runs on a
partition says so with `Manifest::with_sm_partition(Some(sms))`: entries chosen
for another count are refused. A GEMM may have a cuBLASLt epilogue on its f32
accumulator (`impl: "cublaslt.matmul"`: `LibraryEntry::epilogue`, a
`GemmEpilogue` of `none`, `bias`, `relu` or `relu_bias`, and `beta` 1 adds a
`c [M, N]` before it): a launch takes `a`, `b`, [`bias`], [`c`], `out` and the
workspace, every check and run sets the epilogue, an epilogue gets no split-K
without the opt-in, the entry says what the verification of its pin found
(`LibraryEntry::contract`, `verification`), and the load runs its canary
(`LibraryEntry::canary`: the pin's output on operands made of a seed on the
device, on a stream of its own) and refuses other bytes, since cuBLASLt has no
build number to compare (`Gemm::epilogue`, `has_c`, `contract`,
`check_canary`). A library entry may also be a pinned cuDNN
forward convolution (`impl: "cudnn.conv_fwd"`, `Manifest::conv`, `Conv`,
`ConvEntry`): the pin is a legacy algorithm enum or a backend engine
configuration (an engine and its knobs), with the cuDNN version that chose it;
the crate refuses another cuDNN version, another SM count and a process with
`NVIDIA_TF32_OVERRIDE` set, vets the pin as Python does, runs the entry's
canary (its output on operands made of a seed, on a stream of its own) and
refuses other bytes, shares one cuDNN handle per CUDA context between all
its convolutions, and opens libcudnn only for a manifest that has a
convolution. Making that handle (`cudnnCreate`) waits for all the work queued
on the device, on every stream: a host that loads convolutions while work runs
makes `CudnnHandle::new(&ctx)` at start-up, before it queues any, and keeps it,
so that no load makes a handle (the first run of a cuDNN engine, a load's
canary, may wait too while CUDA loads its kernels: load them at start-up as
well). The entry of an operator declared
with an implementation per architecture (`@tilefish.extern(by_arch=...)`) says
which one it is (`Entry::implementation`: its own architecture, `"fallback"`,
or the name of an implementation that several architectures share; a Triton
kernel or a prebuilt binary); the crate launches the entry of the device's
architecture with that entry's own ABI (the entries of one operator on other
architectures may pass a tensor as a TMA descriptor instead of a pointer, or
take other scalars), as for any operator. A parameter that is a TMA descriptor of a tensor (manifest
kind `tensordesc`, `ParamKind::Pointer` with a `tensor_map`) is given its
tensor like a pointer: every launch (and every step of a plan, at every
execution) checks what the driver would refuse (`PreconditionKind::TensorMap`)
and encodes the `CUtensorMap` of that tensor with `cuTensorMapEncodeTiled`
(cudarc's `sys`), then passes it by value with the tensor's shape and strides,
as Triton launches a host `TensorDescriptor`; a captured graph holds the map
of the addresses it captured. A CuTe DSL kernel (a `binary` entry of source
`cute`, `BinarySource::Cute`) is launched as its compiled host function
launches it: the entry's grid of integers, block, cluster and shared memory,
its tensors as pointers or as the TMA descriptors the DSL encoded (kind
`tensormap`, `ParamKind::Pointer` with a `template`: a tensor like a pointer,
whose launch passes the template with the tensor's address in its first eight
bytes, 128 bytes by value; an address of 2^57 or more is refused) and its
constant bytes (kind `bytes`, `ParamKind::Bytes`, passed as the manifest has
them), in a single launch, a plan step and a captured graph alike. A warp-specialised kernel (Gluon's
`gl.warp_specialize`, or Triton's own warp specialisation) has the warps of all
its partitions in `Entry::total_num_warps` (next to `Entry::num_warps`, the
declared warps of its default partition), and every launch, step of a plan and
captured graph takes the block they make (`Plan::block()`: the entry's
`block`, `32 * total_num_warps`).
cudarc opens cuBLASLt by name (`libcublasLt.so`, then `libcublasLt.so.13` for the
default `cuda-13020` bindings, then older majors): where cuBLASLt comes from a Python
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
weights are borrowed for the life of the `WeightedRun`). A plan set (several plans exported together with one manifest, `plans.json` its index)
loads with one pin: `PlanSet::load(dir, Pin::Digest(set_digest))`, then `set.plan(name)` for each `LaunchPlan`, which share the set's manifest
(`Manifest` is cheap to clone, and its clones share the modules they load); a set this process verified already, and still holds, is not read again
by a second `PlanSet::load` (one per worker, say), and `set.plans(&names)` loads many plans on several threads. Plans of a set that a host runs one at a time share one arena:
`set.instantiate(&stream, &["a", "b", "c"])` is a `SetRun` whose `run("a")` is a `PlanRun` on one `SharedArena` of the largest plan's bytes (each
plan at its own offsets; `SharedArena::new` and `LaunchPlan::instantiate_in` for a chosen set of buckets), loaded and instantiated on several threads.
Every launch and replay of any of them is ordered with the others by the arena's order event (a launch on the stream of its last record does not
wait for it: one `cuEventRecord` per launch while they stay on one stream), so they never overlap on the device; every `SetRun`, `SharedArena` and
instantiation is an arena of its own, so lanes that run concurrently take one each. Nothing in the plans' own regions outlives a launch: a stage's output is the next stage's external, in
the caller's memory (a tensor in the arena is refused), or a **carried buffer** (`BufferRole::Carried`, a parameter the graphs declare
`carried=("h",)`), which the arena holds after the plans' regions (`CarriedLayout`, `SharedArena::carried`): a stage writes it, the next reads it
without being given it, and a launch or a replay that reads it before any launch on the arena wrote it, or in another shape, is refused. Memset steps
(`StepKind::Memset`, `tilefish.zeros_like` in a graph) zero a buffer with `cuMemsetD8Async`, a memset node in a graph. `PlanRun::launch_tapped` is a launch with debug
taps: each `Tap` copies a buffer of the plan into a `TensorMut` right after a step, with the plan and its digest unchanged (`PlanRun::capture_tapped`
captures them as memcpy nodes, copied again at every replay). `PlanRun::launch_step` launches one step of a bucket on the caller's tensors, with the
binary and the binding of the pinned plan (not on a capturing stream).

An **instrumented export** (`tilefish.runtime.export(..., instrument=True)`, `docs/instrument.md`) is a diagnostic variant whose binaries record the
`tilefish.probe("name")` phase boundaries of their kernels into a probe buffer given in their profile scratch parameter; it has a digest of its own,
and `Manifest::load`, `Manifest::from_embedded`, `LaunchPlan::load` and `PlanSet::load` refuse it. `Manifest::load_instrumented` (and
`LaunchPlan::load_instrumented`, `PlanSet::load_instrumented`) read it; `Kernel::launch_probed` records one launch into a `ProbeBuffer`
(`Kernel::probe_bytes` says how large; `ProbeBuffer::collect` reads the records back), `LaunchPlan::instantiate_probed` gives every instrumented step
of a plan its region of a probe buffer of the instantiation (launched or captured), read with `PlanRun::collect_probes`, and `ProbeTrace::save`
writes `tilefish.probe-trace.v1` for `tilefish-explain DIR --probes FILE`.

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
tilefish = { version = "0.0.11", default-features = false, features = ["cuda-13000"] }
```

**Reproducible builds of a host.** The crate embeds no path of its own: it has no
build script, and uses `env!` and `file!` only in its tests. Two things made
builds of the same host on two machines differ:

- **An embedded manifest** (`tilefish.hosts.rust.emit(..., embed=True)`, whose
  `include_bytes!` paths are relative to the generated file). Up to tilefish 0.0.6
  a cubin's debug sections named the directory it was built in, so each export had
  other bytes and another `MANIFEST_DIGEST`. tilefish 0.0.7 names the sources
  by module path, so the same operators exported with the same toolchain for the same
  device are the same files on any machine. Export a manifest of 0.0.6 or earlier
  again to embed it reproducibly.
- **The paths rustc writes into the binary**: the source file of every panic
  location (`std::panic::Location`), including those of this crate, cudarc and
  serde. A workspace member's are relative to the workspace root, but a crate from
  a registry is under `$CARGO_HOME` (by default `~/.cargo`), which names the user.
  Build with the same toolchain, `Cargo.lock`, target and profile, and remap the
  prefixes:

  ```sh
  RUSTFLAGS="--remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo --remap-path-prefix=$PWD=/build --remap-path-prefix=$(rustc --print sysroot)=/rustc" \
    cargo build --release --locked
  ```

  The sysroot prefix matters when the `rust-src` component is installed: rustc
  then writes the local paths of std's sources. Cargo's `trim-paths` profile option
  does the same, but cargo 1.100 still needs nightly for it. A path dependency outside the
  workspace root (`tilefish = { path = "../tilefish" }` from another workspace)
  cannot be remapped: cargo hashes its absolute path into the crate's metadata, and
  that changes symbol hashes and the code layout. Keep a vendored copy inside the
  workspace, or depend on the crate from a registry.

Checked on Linux (cargo 1.100) and macOS (cargo 1.99): a host that embeds a
manifest, built in two checkouts at different paths with different `CARGO_HOME`s,
has the same binary sha256 with these flags, and a different one without them.

The crate turns on no feature of serde_json, so it does not change the key order
of the host's own JSON (cargo would turn `preserve_order` on for the whole
build). A host that turns it on itself gets the same reading of a manifest or a
plan: where the order of an object's keys means something (the buckets of an
entry make its stem in their order), the crate takes it from the file.

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
