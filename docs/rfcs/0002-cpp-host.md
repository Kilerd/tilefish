# RFC 0002: a C++ host

Status: accepted and implemented, 2026-11-15 (the defaults of section 12 were accepted). It addresses the open item "C++ host: a header generated
from the same manifest and plan" (carried over from issue #3 into issue #4). The sections below describe what was built; where the implementation
settled a detail the proposal left open, the text says what it is.

## 1. Summary

A C++ program gets the same host the Rust program has, with no second implementation of it:

* **`crates/tilefish-capi`**: a C ABI over the `tilefish` crate (a `cdylib` and a `staticlib`), with a versioned header, `tilefish.h`. It covers loading
  and verifying manifests, plans and plan sets, launching kernels, pinned GEMMs and pinned convolutions, running plans on arenas and shared arenas, and
  capturing and replaying plans. Every check of the crate runs unchanged behind it.
* **`tilefish.hpp`**: a header-only C++17 wrapper over `tilefish.h`, with RAII handles, typed tensors and exceptions.
* **`tilefish.hosts.cpp`**: the Python emitter writes typed C++ launchers (`emit`) and typed plan runners (`emit_plan`) from a pinned manifest and plan.
  It works like `tilefish.hosts.rust`.

No manifest or plan schema changes: the C++ host reads them through the crate.

## 2. What a host must do

The Rust host is the reference. It reads a manifest strictly and checks its digest against a pin. It checks every cubin's `sha256` and `sha256_code`
and every entry's consistency. When it loads a binary, it checks the function's parameter sizes. When it loads a GEMM, it checks the cuBLASLt version
and vets the pinned algorithm. When it loads a convolution, it checks the cuDNN version and runs the canary. Before every launch it checks the call
against the manifest: shapes, dtypes, buckets, `divisible_by`, alignment, aliasing, the extent of a view, the grid, and TMA descriptors. It runs plans
on one arena with its own ordering, and it captures plans in CUDA graphs. That is about 20,000 lines of Rust. A vector suite (`crates/tilefish/tests/data`)
holds it to the Python reader, and a GPU gate (`python/tests/rust_gate.py`) holds it byte for byte to Python's `replay`. Every release adds to it. In the
current cycle that includes product and persistent grids, clusters, global scratch, null pointers, i64 dimensions, memset steps, carried buffers and a
cuDNN handle per context.

## 3. Options

| Option | For | Against |
|---|---|---|
| **A. Re-implement the runtime in C++** | No Rust toolchain for C++ users. | A third strict reader and a second launcher would have to stay in step with Python and Rust. Every schema change would land in four places, and the vector suite and the GPU gate would grow a third column. The safety argument (every check before the first launch) would have to be made again in a language that cannot express the borrows the crate relies on. |
| **B. A C ABI over the crate, a C++ wrapper and generated launchers** (chosen) | One implementation of every check. A schema change lands in Python and Rust only. The bytes come from the same code. The C ABI also serves C and any language with a C FFI. | A Rust toolchain is needed to build `libtilefish_capi` until prebuilt libraries exist (section 9). The C boundary replaces the crate's compile-time borrows with documented contracts, which Python's host also has. |
| **C. Generated self-contained C++** (a header that calls `cuLaunchKernel` with baked parameters) | No runtime library. | The manifest would not be verified at load (digests, pins, ELF checks, the cuBLASLt and cuDNN pins), or the verification would have to be generated as C++. GEMMs, convolutions and plans need a runtime anyway. |
| **D. A `cxx` bridge** | Typed C++ and Rust in one build. | Every consumer needs a Rust toolchain and `cxx`'s build integration, and the result is not a stable ABI. |

The decision is **B**. Option C can be a later extension for single kernels, built on the same emitter.

## 4. Layers

```text
generated.hpp, runner.hpp   (tilefish.hosts.cpp: one class per stem and per graph, typed tensors, MANIFEST_DIGEST)
tilefish.hpp                (header-only C++17: RAII, tilefish::Error, TensorRef<T>/TensorMut<T>)
tilefish.h                  (the C ABI, versioned; C99 and C++)
libtilefish_capi            (crates/tilefish-capi: extern "C" over the tilefish crate)
tilefish crate -> cudarc    (libcuda, libcublasLt, libcudnn opened at run time)
```

## 5. The C ABI (`crates/tilefish-capi`, `tilefish.h`)

### 5.1 Conventions

* **Naming.** Every symbol starts with `tilefish_` and every macro with `TILEFISH_`. The short prefix `tf_` is avoided because TensorFlow's C API uses it.
* **Return values.** Every function that can fail returns `tilefish_status` (an `int32_t`; `TILEFISH_OK` is 0) and writes its result through an out
  parameter. Output handles are set to NULL on failure. Accessors that cannot fail return their value directly.
* **Handles.** Handles are opaque pointers to heap objects made by `tilefish_*_create` or `tilefish_*_load`, and are freed by `tilefish_*_destroy`.
  `destroy(NULL)` is a no-op.
* **Strings.** Input strings are NUL-terminated UTF-8. Strings the library returns are owned by the handle they came from, or by the thread's error
  state (section 5.8), and stay valid as long as that owner does.
* **Argument structs.** These are plain C structs passed by pointer, such as `tilefish_tensor`, `tilefish_named_tensor`, `tilefish_devscalar`,
  `tilefish_bucket` and `tilefish_given`. Arrays are passed as pointer plus count. Nothing is passed or returned by value except scalars.
* **Panics.** Every entry point runs inside `std::panic::catch_unwind`. A panic, which is a bug, becomes `TILEFISH_ERROR_INTERNAL` with its message, and
  never unwinds into C. The library is built with `panic = "unwind"`.
* **No driver.** cudarc panics when it is first called on a machine without a driver. Before any call that needs the driver, the ABI checks
  `tilefish::driver_available()` and returns `TILEFISH_ERROR_NO_DRIVER`. Loading and verifying manifests and plans needs no driver, so it also works on a
  machine without a GPU.
* **No macro exports.** cbindgen reads the source as written and expands no macro, so every export is a plain `#[unsafe(no_mangle)] pub unsafe extern
  "C" fn`; a test fails on a `macro_rules!` in the crate's source and on an export the header does not declare.

### 5.2 Handles

| Handle | Wraps | Made by |
|---|---|---|
| `tilefish_context` | `Arc<CudaContext>` and a stream of tilefish's own (section 5.4) | `tilefish_context_primary(device)`, `tilefish_context_borrow(CUcontext)` |
| `tilefish_cudnn_handle` | `CudnnHandle` | `tilefish_cudnn_handle_create(ctx)` |
| `tilefish_manifest` | `Manifest` | `tilefish_manifest_load(dir, pin)`, `..._load_trusting_directory(dir)`, `..._from_embedded(bytes, files, pin)`, `..._with_sm_partition(m, sms)` |
| `tilefish_kernel`, `tilefish_gemm`, `tilefish_conv` | `Kernel`, `Gemm`, `Conv` | `tilefish_kernel_load(manifest, ctx, stem, buckets, arch)`, and the same for GEMMs and convolutions |
| `tilefish_plan` | `LaunchPlan` | `tilefish_plan_load(path, pin)`, `..._load_with_manifest`, `..._from_embedded`, `tilefish_plan_set_plan(set, name)`, `tilefish_plan_set_plans(set, names, n, out)` (several, loaded on threads: `PlanSet::plans`) |
| `tilefish_plan_set` | `PlanSet` | `tilefish_plan_set_load(dir, pin)` |
| `tilefish_shared_arena` | `SharedArena` | `tilefish_shared_arena_create(ctx, stream, plans, n)` |
| `tilefish_plan_run` | `PlanRun` | `tilefish_plan_instantiate(plan, ctx, stream, buckets)`, `..._instantiate_in(plan, arena, buckets)`, `tilefish_set_run_plan(set_run, name)` |
| `tilefish_set_run` | `SetRun` | `tilefish_plan_set_instantiate(set, ctx, stream, names, n)` |
| `tilefish_weighted_run` | `WeightedRun<'static>` | `tilefish_plan_run_bind_weights(run, weights, n)` |
| `tilefish_captured_plan` | `CapturedPlan<'static>` | `tilefish_plan_run_capture(run, stream, externals, devscalars)`, and the same on a weighted run |

For `arch` (a string, `"sm_103"`), NULL means the architecture of the context (`Arch::of_context`). For `buckets`, NULL with a count of 0 means none (a
single binary) or, for plans, every bucket (`instantiate_all`). Handles are independent of each other: a kernel keeps its context and module alive, and a
run keeps its plan and arena alive (Rust `Arc`s), so handles can be destroyed in any order. A captured plan made through a weighted run holds that
weighted run. The exceptions are the borrows in section 5.10. Read-only accessors cover what a generic host and the tests need: the digest, label and
version; for a kernel, the names of its tensors in launch order; for a plan, its buffers (name, role, dtype, rank, bytes), outputs and dimensions with
their buckets, and the arena size; for a plan set, its digest and graph names; for a GEMM or convolution, its sizes and workspace bytes; for a shared
arena, its bytes and address; and the cuBLASLt and cuDNN versions of the process.

### 5.3 Tensors are regions

```c
typedef struct tilefish_tensor {
    uint64_t address;         /* a CUdeviceptr: the start of the region */
    uint64_t bytes;           /* the length of the region */
    int32_t device;           /* the device ordinal of the region */
    int32_t dtype;            /* tilefish_dtype: BF16, F16, F32, I32, I64, U8, E4M3, E5M2 */
    int32_t access;           /* tilefish_access: IN, OUT, INOUT */
    uint32_t rank;
    const uint64_t *shape;    /* rank sizes */
    const uint64_t *strides;  /* NULL: dense row-major; otherwise rank element strides (a view) */
    uint64_t offset;          /* a view: the element offset of its first element in the region */
} tilefish_tensor;
```

Each tensor maps to `TensorArg::new` or `TensorArg::new_strided` over `DevicePtr::from_raw(address, bytes, device)`. Its contract is the contract of
`from_raw`, stated in `tilefish.h`. The memory is allocated device memory of that device (`cuMemAlloc`, `cudaMalloc`, `cudaMallocAsync`, `cuMemMap`, or a
framework's allocator). It stays allocated until every launch that used it has completed, and for a captured plan until the plan is destroyed. The caller
orders it against its own streams. Everything else is still checked before anything is enqueued, as in Rust: the tensor fits its region, plus every
precondition of section 2. The C ABI does not take cudarc slices. Its memory model is the model of Python's host over torch tensors, which the hosts already
share. A dense tensor (`strides` NULL) has offset 0: its address is that of its first element. The shape and strides are read in place for a launch; a
captured plan and a weighted run keep copies of them for as long as they live.

Named tensors (`{const char *name; tilefish_tensor tensor;}`) are the externals and weights of plans, and a tap is
`{const char *buffer; int64_t after; tilefish_tensor into;}` (`after` negative: after the last step that writes the buffer). A device scalar is
`{const char *name; int32_t kind; float f32_value; int64_t int_value; uint64_t bits;}` and maps to `DevScalar::{Int, F32, Bits}`. A given value is
`{const char *name; int64_t value;}` and maps to `launch_with`. The sizes and field offsets of every argument struct are fixed (a Rust test and a C test
check the same numbers: `tilefish_tensor` is 56 bytes).

### 5.4 Contexts and streams; interop with the driver and runtime APIs

* **The caller owns contexts and streams.** `tilefish_context_primary(device)` retains the device's primary context (cudarc's `CudaContext::new`,
  `cuDevicePrimaryCtxRetain`), and `destroy` releases it. The CUDA runtime API uses the primary context, so this is the context of a host built on
  `cudaSetDevice` and `cudaStream_t`. `tilefish_context_borrow(CUcontext)` takes a context the caller made (`cuCtxCreate`, or `cuCtxFromGreenCtx` for an
  SM partition). If that context is its device's primary context, `borrow` is `primary` and is retained and released. Otherwise tilefish never destroys it.
  cudarc's `from_raw_context` takes ownership and would destroy it on drop, so the ABI keeps one reference to that wrapper for the life of the process: a
  small fixed allocation per borrow. The caller keeps a borrowed context alive until every handle made in it is destroyed.
* **Streams are parameters.** Every launching function takes a `CUstream`, which is the same type as `cudaStream_t`. The stream is borrowed for the
  call and must be a stream of the handle's context: a single launch passes it to the crate's `launch_raw`, a plan wraps it with cudarc's
  `wrap_borrowed_stream` (in cudarc 0.19.10, the crate's oldest). The crate's rules apply as they are. The legacy NULL stream is accepted. The
  per-thread handle (`CU_STREAM_PER_THREAD`, `cudaStreamPerThread`), which names a different stream in every thread, is refused where the crate refuses
  it: by a plan's instantiation, launch, capture and replay.
* **A stream of tilefish's own per context handle.** cudarc orders the memory it allocates (the crate's: an arena, a workspace in it, the staging of
  device scalars) between streams only in a context that has made a stream of its own, and borrowed streams do not count. So each context handle makes
  one stream (`CudaContext::new_stream`, never used for work), and every handle made in the context holds it, so the context keeps that mode while any of
  them lives. Its creation skips cudarc's one-time `cuCtxSynchronize`, which is there for memory a new `CudaContext` value has not allocated yet.
* **Arena streams.** A run (`tilefish_plan_instantiate`, `tilefish_plan_set_instantiate`, `tilefish_shared_arena_create`) allocates its arena on the
  stream it is given and frees it there when the last handle that holds the arena is destroyed. That stream is borrowed until then.
* **The current context.** Every call makes the handle's context current for its duration and then puts back what was current, also on failure. This is
  what the crate's `launch_raw` already does. A thread needs no current context to call tilefish, and tilefish leaves none behind.
* **CUDA graphs of single launches.** `tilefish_kernel_launch`, `tilefish_gemm_launch` and `tilefish_conv_launch` are `launch_raw` over regions. On a
  stream that the caller is capturing, they are captured like any kernel node, and the caller owns the graph and the lifetime of its memory, as in Rust.
* **CUDA graphs of plans.** Plans are captured with `tilefish_plan_run_capture`. The resulting `tilefish_captured_plan` owns the graph and orders the
  arena against every launch and replay of the run, as the crate's `CapturedPlan` does. A plan launch on a capturing stream is refused, as in Rust,
  because the arena's ordering cannot be added to a caller's capture. The stream a plan is captured on is borrowed for the call only: cudarc's graph keeps
  its handle, but uses it only to launch a replay on that same handle.
* **SM partitions.** `tilefish_manifest_with_sm_partition(m, sms)` is `Manifest::with_sm_partition`, for persistent grids and GEMM SM targets on a
  partition.

### 5.5 Loading and verification

`tilefish_manifest_load(dir, pin)` requires a pin (64 lowercase hex digits). Development uses the separately named
`tilefish_manifest_load_trusting_directory(dir)`, so `Pin::TrustDirectory` cannot be reached by passing NULL by mistake. The same holds for plans and plan
sets. `..._from_embedded` takes the manifest bytes and an array of `{file name, bytes, length}`, as `Manifest::from_embedded` does. Every verification
error of the crate surfaces with its own status (section 5.8).

### 5.6 Launches

```c
tilefish_status tilefish_kernel_launch(const tilefish_kernel *k, CUstream stream, const tilefish_tensor *tensors, size_t n,
                                       const tilefish_given *given, size_t n_given);
tilefish_status tilefish_kernel_check(const tilefish_kernel *k, const tilefish_tensor *tensors, size_t n,
                                      const tilefish_given *given, size_t n_given);
```

A GEMM takes `a`, `b`, `bias` if it has one, `c` if it has one (a `cublaslt.matmul` with `beta` 1), `out` and the caller's workspace, which is a `U8`
tensor of at least `tilefish_gemm_workspace_required` bytes (`tilefish_gemm_workspace_bytes` is what the entry was pinned with); `tilefish_gemm_epilogue`
(ABI 1.1) says its epilogue and whether it takes `c`. A convolution takes `x`, `w`, `y` and its workspace. A kernel with a global
scratch takes it last, as the crate's `GLOBAL_SCRATCH` tensor (`tilefish_kernel_tensor_name` names it). What M3.5 added to launches needs nothing from
the ABI: a `null` parameter is not a call argument (the crate passes 0 at its slot), an `i64` dimension is derived from the shapes, a persistent grid is
sized from the device's SM count (or the partition of `tilefish_manifest_with_sm_partition`), and a cluster launch goes through `cuLaunchKernelEx` in the
crate. `tilefish_cudnn_handle_create(ctx)` makes the context's cuDNN handle early and keeps it while the handle lives (the crate's `CudnnHandle::new`:
`cudnnCreate` waits for the whole device).

### 5.7 Plans, arenas and captures

`tilefish_plan_run_launch(run, stream, externals, n, devscalars, n_dev)`, `..._check` and `..._launch_tapped(..., taps, n_taps)` take named tensors.
A weighted run takes the inputs, in-place tensors and outputs only. `tilefish_plan_run_capture` returns a captured plan whose functions are
`tilefish_captured_plan_replay(cp, stream)` and `tilefish_captured_plan_set_devscalar(cp, value)`; `tilefish_plan_run_capture_tapped` and
`tilefish_weighted_run_capture` capture with taps and through a weighted run. Between replays, the caller writes new inputs into its own memory on the
replay stream. Carried buffers and memset steps need nothing from the ABI, because the run handles them: a carried buffer is never given as an external,
and a plan that reads one binds its dimensions from the shape the arena recorded, as in Rust. The crate orders every execution on an arena with its own
event (it skips the wait when the arena's last user was the same stream), so the ABI adds no ordering of its own.

### 5.8 Errors

`tilefish_status` maps the crate's `Error` one to one: `IO`, `MANIFEST`, `BAD_PIN`, `PIN_MISMATCH`, `PLAN`, `PLAN_PIN_MISMATCH`, `NO_BINARY`,
`PRECONDITION`, `UNSUPPORTED`, `LAUNCH`, `CUBLAS`, `CUDNN` and `DRIVER`, each prefixed with `TILEFISH_ERROR_`. The boundary adds three codes:
`INVALID_ARGUMENT` (a NULL pointer, a bad enum value, a string that is not UTF-8, a size that does not fit `usize`), `NO_DRIVER` and `INTERNAL` (a caught
panic). The details of the last failure are kept per thread, as `errno`, `dlerror` and `cudaGetLastError` do:

```c
typedef struct tilefish_error_info {
    tilefish_status status;
    int32_t precondition;    /* tilefish_precondition (DTYPE, RANK, SHAPE, ..., TENSOR_MAP) for TILEFISH_ERROR_PRECONDITION, else -1 */
    int32_t cu_result;       /* the CUresult for TILEFISH_ERROR_DRIVER, else 0 */
    const char *message;     /* the crate's message, the same text Rust prints */
    const char *argument;    /* the argument or dimension the precondition is about, or NULL */
    const char *call;        /* the driver, cuBLASLt or cuDNN function, or NULL */
} tilefish_error_info;
void tilefish_last_error(tilefish_error_info *out);   /* valid until the next failing tilefish call on this thread */
const char *tilefish_status_name(tilefish_status status);
```

Callers treat an unknown status as an error, and new codes may be added in a minor version (section 5.11).

### 5.9 Thread safety

Every handle except `tilefish_captured_plan` can be used from several threads at once: all of the crate's types behind them are `Send + Sync`, which a
compile-time test of the C ABI crate asserts, and the crate's arena lock orders concurrent runs of one arena. A `tilefish_captured_plan` can be used
from any thread, but by one thread at a time, because replay and `set_devscalar` mutate it. The crate's `CapturedPlan` is not `Send` because it holds
raw graph handles and may hold borrows of cudarc slices; with region externals only, using it from another thread is sound (a C caller holds a pointer,
so Rust's marker does not reach it). A `WeightedRun` is not `Send` or `Sync` for the same reason (a `TensorArg` may borrow a slice): the ABI's holds
regions only and asserts both with a documented `unsafe impl`. `destroy` must not race with another use of the same handle. Error details are
thread-local. Concurrent runs on different streams need one arena each (a run or a shared arena per stream), as in Rust.

### 5.10 Borrows the caller keeps

| What | Until |
|---|---|
| a borrowed `CUcontext` | every handle made in it is destroyed |
| the stream a run or shared arena was made with | the arena is freed (its last run and captured plan are destroyed) |
| the stream a plan was captured on | the call returns |
| the memory of a launch's tensors | the launch has completed on its stream |
| the memory of a captured plan's externals and taps | the captured plan is destroyed (a replay launches on those addresses) |
| the memory of a weighted run's weights | the weighted run (and every capture of it) is destroyed |
| the arrays and strings of an argument | the call returns (nothing is kept) |

### 5.11 ABI versioning

* `tilefish.h` defines `TILEFISH_ABI_MAJOR` and `TILEFISH_ABI_MINOR`. `uint32_t tilefish_abi_version(void)` returns `major << 16 | minor` for the loaded
  library. The header's inline `tilefish_abi_compatible()` requires the same major and a library minor at least the header's. `tilefish.hpp` checks this
  when it makes its first handle. `tilefish_version()` is the crate version, and `tilefish_manifest_schema()` and `tilefish_plan_schema()` name the
  schemas the library reads.
* **A minor version only adds.** It may add functions and enum values. It never changes a signature, a struct layout or the meaning of a value. A new
  field means a new struct and a new function (`..._v2`). Anything else is a new major version.
* **Shared library.** It is `libtilefish_capi.so.1` with that SONAME (set by the crate's build script; a test holds the build script, the CMake file and
  `TILEFISH_ABI_MAJOR` to one number), and on macOS its install name is `@rpath/libtilefish_capi.dylib`. It exports only the `tilefish_*` symbols
  (`#[unsafe(no_mangle)] extern "C"`). The first ABI is 1.0; 1.1 adds `tilefish_gemm_epilogue` and the `TILEFISH_EPILOGUE_*` values (a GEMM with a
  cuBLASLt epilogue, manifest schema 3 section 9.5).
* **Header and exports stay in sync.** `tilefish.h` is generated from the crate's `extern "C"` items by cbindgen (`=0.29.4`, a dev-dependency, with
  `cbindgen.toml`) and committed. A test regenerates it and fails on any difference (`TILEFISH_BLESS=1` writes it), and another fails on an export the
  header does not declare. A C99 program and a C++17 program compile the header (twice: the include guard), and the C one checks `sizeof` and
  `offsetof` of the argument structs against the numbers a Rust test checks too.

## 6. `tilefish.hpp`

`tilefish.hpp` is a header-only C++17 wrapper with no dependency beyond `tilefish.h`. CUDA headers are optional: `tilefish.h` declares `CUcontext` and
`CUstream` as the same incomplete struct pointers that `cuda.h` declares, guarded so that including `cuda.h` first or later is fine.

* **RAII.** Move-only classes own one handle each: `Context` (`primary`, `borrow`), `CudnnHandle`, `Manifest`, `Kernel`, `Gemm`, `Conv`, `LaunchPlan`,
  `PlanSet`, `SharedArena` (`make_shared_arena`), `PlanRun`, `SetRun`, `WeightedRun` and `CapturedPlan`. A handle is shared with `std::shared_ptr` if
  needed.
* **Errors.** `tilefish::Error` derives from `std::runtime_error` and carries `status()`, `status_name()`, `precondition()`, `argument()`, `call()`,
  `cu_result()` and the message. Every failing call throws it. A program built with `-fno-exceptions` uses `tilefish.h` directly. An
  `expected`-returning variant can be added later without an ABI change.
* **Typed tensors.** `TensorRef<T>` (access `IN`) and `TensorMut<T>` (access `INOUT`), for element tags `Bf16`, `F16`, `F32`, `I32`, `I64`, `U8`,
  `E4m3` and `E5m2`, mirror the crate's types; both are a `Tensor`, the untyped form. They are built from a `Region{address, bytes, device}` with a
  shape, and optionally strides and an offset. A typed launcher therefore cannot receive an `f32` tensor where a `bf16` one is declared. Up to 8 sizes
  and 16 tensors of a call are stored inline, so a launch does not allocate.
* **Plans.** `run.launch(stream, {{"x", x}, {"out0", y}}, {{"n", 64}})` takes `Named` tensors and `DevScalar` values (an integer, an `f32`, or
  `DevScalar::bits`), and `CapturedPlan cp = run.capture(stream, ...); cp.replay(stream);` captures and replays; a `Tap` list is the last argument of
  `launch` and `capture`.

## 7. Generated C++ (`tilefish.hosts.cpp`)

`emit(manifest, "launchers.hpp", embed=False, namespace="tilefish_launchers")` and `emit_plan(plan, "runner.hpp", pin=..., namespace="tilefish_plans")`
mirror `tilefish.hosts.rust`. They follow the same rule: only a pinned manifest or plan is accepted, and every field that reaches the output is
validated. A manifest therefore cannot break out of a comment, an identifier or a string literal. The validation code is the Rust emitter's, imported;
names that are C++ keywords or that the generated code uses get a `_`, and names C++ reserves are refused.

* **Launchers.** There is one class per stem of a kernel, GEMM or convolution, with `STEM`, `BUCKETS` (a `std::array<tilefish_bucket, N>`), the
  declared shape of each tensor and the sizes baked in (and `M`, `N`, `K`, `BIAS`, `WORKSPACE_BYTES`, `PINNED_ARCHS` and `PINNED_WITH` for a GEMM or
  convolution). `static Cls load(const Manifest&, const Context&)` refuses a manifest whose digest is not `MANIFEST_DIGEST`.
  `void launch(CUstream, const TensorRef<Bf16>& x, const TensorMut<Bf16>& y) const` takes one typed parameter per tensor, then a global scratch and the
  values given per call when the binary has them; `check(...)` takes the same without the stream. Documentation comments carry what the manifest says,
  as in Rust.
* **Embedding.** `embed=True` writes the manifest and cubins as byte arrays into a companion source file, `launchers_embedded.cpp`, and declares
  `load_embedded()` in the header. Large arrays then compile once, and not in every translation unit that includes the header. C23 and C++26 `#embed`
  can replace the arrays later. The emitter checks the files against the pin before it embeds them.
* **Plan runners.** There is one class per graph, with `PLAN_DIGEST`, `GRAPH`, `ARCH` and `ARENA_BYTES`, `load(directory, ctx, stream)` (pinned to the
  plan's digest, every bucket instantiated), and `launch` and `capture` that take an `Externals` aggregate (one typed member per input, weight, in-place
  tensor and output, in the plan's order) and a `Scalars` aggregate (one member per device scalar). The two structs are at namespace scope
  (`<Graph>Externals`, `<Graph>Scalars`) with aliases in the class.
* **Output.** The output is deterministic (its include guard has a prefix of the digest, so two runners of one graph name can be included in different
  namespaces) and is checked by unit tests (golden files of the committed fixture and of a golden plan) and by compiling it in the C++ tests.

## 8. How cuBLASLt and cuDNN are loaded

`libtilefish_capi` links nothing from CUDA at build time (cudarc's `dynamic-loading`). It opens `libcuda.so.1` at the first driver call, cuBLASLt when
the first GEMM is loaded, and cuDNN when the first convolution is loaded. A manifest without them never opens them. The libraries are opened by name, in
the order the crate's README lists, so they are found through `LD_LIBRARY_PATH`, the system search path, or a library the process already loaded under
the same SONAME. A C++ host that links cuBLASLt or cuDNN itself, or opens it first, therefore makes tilefish use that copy. The load then checks that
copy's version against the pin. A GEMM pinned under another cuBLASLt, or a convolution pinned under another cuDNN, is refused with
`TILEFISH_ERROR_MANIFEST` and a message that names both versions. A library that cannot be opened gives `TILEFISH_ERROR_UNSUPPORTED`. One cuDNN handle
per context is shared by every convolution of that context, including those loaded through Rust in the same process. `tilefish_cudnn_handle_create`
moves the handle's creation, which waits for the device, to start-up. An explicit library path (an environment variable or a function) is a possible later addition, if a host needs to choose among copies.

## 9. Build and distribution

* **cargo first.** `cargo build -p tilefish-capi --release` gives `libtilefish_capi.{so,a}` (`.dylib` on macOS). The crate has the same `cuda-*`
  features as `tilefish`: the bindings decide the oldest driver it runs on.
* **CMake package.** `crates/tilefish-capi/CMakeLists.txt` (CMake 3.19 or newer) builds the crate with cargo through a custom command
  (`TILEFISH_CARGO_PROFILE`, `TILEFISH_CARGO_FEATURES` for another `cuda-*` binding), and installs the headers, both libraries and
  `tilefishConfig.cmake`. That config provides the imported targets `tilefish::tilefish` (shared) and `tilefish::tilefish_static` (static, with the link
  interface Rust's standard library needs: `pthread`, `dl`, `m`, `rt` on Linux, `iconv` on macOS). A consumer writes
  `find_package(tilefish CONFIG REQUIRED)` and `target_link_libraries(app PRIVATE tilefish::tilefish)`. `add_subdirectory` of the crate directory gives
  the same targets.
* **Prebuilt libraries** are a release step after this RFC lands: per platform (Linux x86_64 and aarch64, with a glibc baseline) and per driver baseline
  (12.4 bindings for 12.x drivers, and 13.x bindings), attached to a GitHub release with their sha256. Until then a C++ host needs a Rust toolchain to
  build.
* **Linking more than one Rust static library** into one binary can give duplicate symbols from Rust's standard library. Such a binary uses the shared
  library.
* **macOS.** The crate and the C++ unit tests build and run without CUDA: `tilefish_driver_available()` is false, and loading and verification work.

## 10. Tests and gates

* **Mac, no GPU.**
  * `cargo test -p tilefish-capi`: the status for every `Error` variant and every precondition, NULL and invalid arguments, a caught panic,
    `TILEFISH_ERROR_NO_DRIVER`, the ABI version, the layout of the argument structs, `Send + Sync` of what the handles wrap, the header being current and
    declaring every export, and loading the committed fixture manifest with its pin, with a wrong pin, with a bad pin, trusting the directory, from
    embedded bytes and from a tampered copy (each refusal's status and message), a golden plan with what it says of itself, and a golden plan set.
  * CMake and ctest: a C99 program and a C++17 program that use only the headers and link the static and the shared library, which repeat the loading
    checks; the C++ one compiles the committed `generated.hpp` and `block_runner.hpp`. A third test installs the package to a scratch prefix and builds a
    consumer with `find_package(tilefish)`.
  * `uv run pytest`: emitter unit tests (validation, refusal of an unpinned manifest, determinism, golden output; the rendered headers compiled with
    `-Wall -Wextra -Werror`; an embedded manifest linked against the library and verified).
  * The existing gates are unchanged.
* **GPU: the C++ gate.** `rust_gate.py prepare WORK` writes `generated.hpp`, `generated_embedded.cpp`, `block_runner.hpp` and `fused_runners.hpp` (the
  runner of every fused plan, `fused_*`, and a call of each from a case) next to `generated.rs`, with
  the same inputs and the same replay bytes. A C++ program (`crates/tilefish-capi/tests/gpu/cpp_gate.cpp`, built by CMake) runs every case of
  `cases.json` and `plans.json` through the C ABI, with the typed launchers where it has an arm. That covers kernels (views, device scalars, prebuilt
  binaries, TMA descriptors, warp-specialised kernels, clusters, persistent grids, global scratch, given values), GEMMs (and the workspace case) and
  convolutions, with the manifest directory and with the embedded manifest; plans launched and captured and replayed twice; a plan set; the stages of a
  set on one shared arena; carried buffers handed from plan to plan; weights bound once; the generated runner of the plan `block`; and the plans whose
  graph body computes (`fused_*`, plan fusion), untyped and through their generated runners. It writes the outputs to `cpp/`, `cpp_embedded/`,
  `cpp_plans/`, `cpp_plans_graph/`, `cpp_shared/`, `cpp_shared_graph/`, `cpp_carried/`, `cpp_carried_graph/` and `cpp_fused_runners/`, and
  `rust_gate.py compare WORK --host cpp` holds them byte for byte to Python's replay, as it does for Rust. The gate also checks refusals by status: a
  precondition of each kind it can make (shape, access, dtype), a tensor larger than its region, a wrong pin, a tampered cubin, another architecture, a
  plan launched on a capturing stream, and on a fresh arena a carried buffer nothing wrote. It checks that a kernel and a GEMM captured in the host's own
  CUDA graph replay to the same bytes, and that a borrowed primary context launches and leaves the thread's current context as it was. The gate reads
  JSON with a small reader in the test directory and opens `libcuda.so.1` for its own memory, so it needs no CUDA toolkit to build.
* **The cost of the boundary.** `tests/gpu_overhead.rs` (`--features gpu-gate`) launches a kernel and a plan through the crate and through the C ABI in
  one process and reports the host time of each.
* **What the gate proves.** The C++ host runs the Rust code, so its kernels are Rust's by construction. The gate tests the boundary: tensor and stride
  translation, offsets, device-scalar encoding, external names, capture lifetimes and error mapping.

## 11. What the crate needs

Nothing: the `tilefish` crate is unchanged. Kernels, GEMMs and convolutions use `launch_raw` with regions; plans use `wrap_borrowed_stream`, which is in
cudarc 0.19.10, the crate's oldest supported version; the per-context cuDNN handle is `CudnnHandle`, and parallel plan loading `PlanSet::plans`. The cost
of the boundary per call (the argument conversion, `catch_unwind`, putting back the current context and, for a plan, the stream wrapper) is measured by
`tests/gpu_overhead.rs`. On a B300 MIG slice, in a release build, a kernel launch took 1.77 µs of host time through the crate and 1.93 µs through
`tilefish_kernel_launch`, and a launch of the gate's plan `block` (nine steps) 33.1 µs and 31.8 µs (the difference is in the noise), so no cache of
stream wrappers was needed.

## 12. Risks and open questions

* **Contracts in place of borrows.** The C ABI cannot check lifetimes. A replay after the caller frees an external writes into freed memory, as with any
  CUDA graph. The headers state each contract (section 5.10), and every other check still runs.
* **Two toolchains** until prebuilt libraries ship.
* **Plans in a caller's CUDA graph** are refused, as in Rust. A host that needs one graph for everything captures single launches itself. Exporting a
  captured plan's graph handle, for embedding as a child node, is a possible later addition.
* **Not in this version**: an explicit cuBLASLt or cuDNN path, an `expected`-returning C++ API, a self-contained generated launcher (option C), and
  prebuilt libraries.
* **Decisions** (accepted on 2026-11-15):
  * the prefix `tilefish_` and the library name `libtilefish_capi`;
  * C++17 with exceptions only (`-fno-exceptions` callers use the status-code C API);
  * a header generated by cbindgen and kept in sync by a test, not written by hand;
  * prebuilt libraries after this implementation, not in it.
