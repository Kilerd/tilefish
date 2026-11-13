//! The Rust host of a tilefish manifest.
//!
//! tilefish builds a model once into GPU binaries plus a `manifest.json`; this crate loads that directory and
//! launches its cubins from Rust, with no Python and no Triton, and every check Python's `replay` makes. Together with
//! the typed launchers that `tilefish.hosts.rust.emit` generates from a manifest, it is the "migrate to a
//! high-performance host with zero changes" half of the design, at the level of one kernel: the machine code is the
//! exact bytes Python's `replay` runs.
//!
//! ```no_run
//! use tilefish::cudarc::driver::CudaContext;
//! use tilefish::{Arch, Bf16, Manifest, Pin, TensorMut, TensorRef};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // 0. cudarc loads the CUDA driver when it is first called, and `CudaContext::new` PANICS when there is none (this crate builds,
//! //    and its tests that need no GPU run, on a machine without one): ask first.
//! if !tilefish::driver_available() {
//!     eprintln!("no NVIDIA driver on this machine");
//!     return Ok(());
//! }
//!
//! // 1. Read and verify the directory. Production pins the digest of the manifest; `Pin::TrustDirectory` trusts the
//! //    directory's own digests, which whoever can write the directory can rewrite.
//! let manifest = Manifest::load("build/", Pin::Digest("<the digest `tilefish.runtime.export` returned>"))?;
//!
//! // 2. Load the binary for the device of a context, which the kernel then keeps alive.
//! let ctx = CudaContext::new(0)?;
//! let stream = ctx.new_stream()?;
//! let kernel = manifest.kernel(&ctx, "gelu_erf", &[], Arch::of_context(&ctx)?)?;
//!
//! // 3. Launch on a stream of that context. Every precondition of the binary is checked first. The tensors are made of cudarc
//! //    slices, whose length is known: no `unsafe`, and a shape that does not fit its slice is an error, not a launch. A tensor
//! //    borrows its slice (the slice cannot be freed under it), making it records nothing on the slice, and the launch gives the
//! //    slice to cudarc on the stream it launches on and makes the slice's own stream wait for the kernel (see `Kernel::launch`).
//! let (x, mut y) = (stream.alloc_zeros::<u8>(4096 * 2)?, stream.alloc_zeros::<u8>(4096 * 2)?);
//! let shape = [4096usize];
//! let x_tensor = TensorRef::<Bf16>::from_cuda(&x, &shape)?;
//! let y_tensor = TensorMut::<Bf16>::from_cuda(&mut y, &shape)?;
//! kernel.launch(&stream, &mut [x_tensor.into_arg(), y_tensor.into_arg()])?; // the dimensions and the scalar come from the shapes
//! # Ok(())
//! # }
//! ```
//!
//! # What a manifest can describe
//!
//! Manifest schema 3 (`docs/manifest-schema-3.md`) is read in full, and everything it describes is launched:
//!
//! * a **Triton kernel**, or a **prebuilt binary** (`kind: "binary"`: PTX that `ptxas` assembled, or a cubin used as it is, for its own
//!   architecture), which is launched with exactly its parameters and no scratch slots ([`EntryKind`], [`BinarySource`]);
//! * a pointer that takes a **strided view** of a tensor (`Strided[...]` in Python): any element strides and any starting offset, the
//!   kernel is given the address of the view's first element and the strides as scalars. A view is made with
//!   [`TensorRef::from_cuda_strided`] (or `new_strided`, over a region): its shape, its element strides and the element offset of its
//!   first element, and it is refused when it reaches past its memory. The axis a kernel loads along has the fixed stride 1, the strides
//!   are bound from the *canonical* strides of the view (the stride of a size-1 axis is never read, so an `unsqueeze` or an `expand` does
//!   not pass one), a stride must fit the `i32` or `i64` the binary takes it as, and a binary that indexes with `i32` bounds the view's
//!   extent (`max_extent`) as it bounds the elements of a dense tensor. A dense tensor is a view too (the dense strides), and a view is
//!   given to a dense parameter only when it is contiguous;
//! * a pointer with the empty shape, a **device scalar**: one element of device memory (a tensor of shape `[]` or `[1]`) that the kernel
//!   reads with `tl.load`, so that one binary runs for every value;
//! * **`i64` scalars** and **`fixed` arguments** (a value the manifest holds, written at its slot: an `i32`, an `i64` or an `f32`);
//! * a **library operation**: a bf16 GEMM through cuBLASLt with the algorithm the manifest pinned ([`Manifest::gemm`], [`Gemm`]).
//!
//! ```no_run
//! use tilefish::cudarc::driver::CudaContext;
//! use tilefish::{Arch, Bf16, I32, Manifest, Pin, TensorMut, TensorRef, U8};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # if !tilefish::driver_available() { return Ok(()); }
//! # let manifest = Manifest::load("build/", Pin::TrustDirectory)?;
//! # let ctx = CudaContext::new(0)?;
//! # let stream = ctx.new_stream()?;
//! # let arch = Arch::of_context(&ctx)?;
//! // the q of a fused qkv of 40 tokens: a [40, 8, 64] view with the row stride 1536, starting 512 elements in (`qkv[:, 0]` has offset 0,
//! // `qkv[:, 1]` has offset 512): the kernel is given the address of its first element and the strides. `u8` is any slice of bytes.
//! let qkv = stream.alloc_zeros::<u8>(40 * 3 * 8 * 64 * 2)?;
//! let mut out = stream.alloc_zeros::<u8>(40 * 8 * 64 * 2)?;
//! let (shape, strides) = ([40usize, 8, 64], [1536usize, 64, 1]);
//! let view = TensorRef::<Bf16>::from_cuda_strided(&qkv, &shape, &strides, 512)?;
//! let y = TensorMut::<Bf16>::from_cuda(&mut out, &shape)?;
//! manifest.kernel(&ctx, "head_copy", &[], arch)?.launch(&stream, &mut [view.into_arg(), y.into_arg()])?;
//!
//! // a device scalar is a tensor of one element
//! let n_tok = stream.alloc_zeros::<u8>(4)?;
//! let scalar = TensorRef::<I32>::from_cuda(&n_tok, &[])?;
//! # let _ = scalar;
//!
//! // a pinned GEMM: out[64, 1024] = a[64, 512] @ b[1024, 512]^T + bias[1024]. The workspace is the caller's, the last tensor: nothing is
//! // allocated inside a launch (so it can be captured in a CUDA graph with `launch_raw` and regions).
//! let gemm = manifest.gemm(&ctx, "gemm_bias", &[("M", 64)], arch)?;
//! let mut workspace = gemm.alloc_workspace(&stream)?;
//! # let (a, b, bias) = (stream.alloc_zeros::<u8>(64 * 512 * 2)?, stream.alloc_zeros::<u8>(1024 * 512 * 2)?, stream.alloc_zeros::<u8>(1024 * 2)?);
//! # let mut gemm_out = stream.alloc_zeros::<u8>(64 * 1024 * 2)?;
//! let (a_shape, b_shape, bias_shape, out_shape, ws_shape) = ([64usize, 512], [1024usize, 512], [1024usize], [64usize, 1024], [workspace.len()]);
//! gemm.launch(
//!     &stream,
//!     &mut [
//!         TensorRef::<Bf16>::from_cuda(&a, &a_shape)?.into_arg(),
//!         TensorRef::<Bf16>::from_cuda(&b, &b_shape)?.into_arg(),
//!         TensorRef::<Bf16>::from_cuda(&bias, &bias_shape)?.into_arg(),
//!         TensorMut::<Bf16>::from_cuda(&mut gemm_out, &out_shape)?.into_arg(),
//!         TensorMut::<U8>::from_cuda(&mut workspace, &ws_shape)?.into_arg(),
//!     ],
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! # What is checked
//!
//! * **At load** ([`Manifest::load`]): the manifest is strict JSON of schema 3 (no duplicate keys, no `NaN`, no unknown
//!   field, every field of its type, the `library` list a list); its digest is the pinned one (the tag `v3`, over the kernels and the
//!   library); every cubin is a regular file in the directory, whose `sha256` and `sha256_code` (the digest of the machine code,
//!   [`elf`]) are the manifest's and which is for the architecture its entry says (its ELF header, for a prebuilt binary always); the
//!   entries are consistent (the ABI with its strides, device scalars and fixed arguments, the declared shapes, the dimensions, the
//!   preconditions, the grid, the library entries' problem and pinned algorithm). Digests are computed exactly as Python computes
//!   them. A manifest of an older schema is refused: export it again. When a binary is loaded ([`Manifest::kernel`]) the function the driver
//!   reports must take the parameter sizes of the ABI (`cuFuncGetParamInfo`, as Python checks at load), so a prebuilt binary whose
//!   declared C ABI is not the PTX's is refused; when a GEMM is loaded ([`Manifest::gemm`]) the cuBLASLt of the process must be the
//!   version that chose its algorithm, the algorithm's 64 bytes are vetted the way Python's binding vets them and the library must approve
//!   them for the problem.
//! * **At launch** ([`Kernel::launch`], [`Gemm::launch`]): the manifest declares the shape of every pointer (`["Rows", 1024]`) and the
//!   constraints of every named dimension, so a launch needs nothing but the tensors. It binds the dimensions from their sizes and
//!   the stride dimensions from the canonical strides of the views, checks that they agree across arguments and with the static sizes,
//!   the exact bucket and `divisible_by` of each dimension, the dtype and rank of every tensor (a device scalar is one element), that a
//!   tensor given for a parameter that is not strided is contiguous, that every tensor is on the kernel's device *and in its CUDA context*
//!   and that a tensor that is only read is not given for a parameter that is written, `max_elements` (and, for a view, `max_extent`),
//!   that the runtime scalars (derived from the dimensions, from the element count of the tensor they count, or from a stride) fit the
//!   32 or 64 bits they are passed as, that nothing written aliases anything else (a view by its extent), the alignment each pointer
//!   promises (16 bytes, or for a strided pointer its element's), and the grid against the device's limits. Nothing is launched when a check
//!   fails. There is no way to tell a launch what a dimension is: that would be a way to run a binary under assumptions the tensors do
//!   not satisfy.
//!
//! # Memory and streams
//!
//! A tensor is made of a cudarc `CudaSlice` or view ([`TensorRef::from_cuda`], [`TensorMut::from_cuda`]: sealed to those types
//! ([`CudaRead`], [`CudaWrite`]), because a user type that implements cudarc's `DevicePtr` can name any address and length), or of a
//! region the host vouches for ([`DevicePtr`], `unsafe` to make).
//!
//! * A tensor over a slice holds the borrow of the slice, and has no stream. The tensor, and the [`TensorArg`] it turns into, cannot
//!   outlive the slice, and an address is only a number ([`TensorRef::addr`]): there is nothing to rebuild a tensor from without
//!   `unsafe`. **Making a tensor records nothing on the slice** (and neither does throwing it away, or a launch that is refused):
//!   cudarc keeps one `read` event per slice, and recording on it replaces the record of a launch that is still queued on another stream.
//! * [`Kernel::launch`] (and [`Gemm::launch`], through the guards of cudarc's `device_ptr`, which wait and record the same way) hands the
//!   slice to cudarc on the stream it launches on (a view of a tensor whose first element is not at the start of the slice is handed as the
//!   bytes of the slice from there, so that the kernel is given the view's address and cudarc still tracks the slice), which orders the launch
//!   after the slice's earlier writes (and reads, for a write) and records it on the slice, and the crate adds, in every context (cudarc tracks
//!   only one that has made a stream of its own): the slice's own stream (the one it is freed on) waits for an event recorded right
//!   after the kernel, when that is not the launch stream, so freeing the slice, or any later work on its stream, runs after the
//!   kernel, however many other launches read the slice on other streams; and a kernel that writes the slice, from a stream that is not
//!   its own, first waits for everything enqueued on the slice's own stream, which includes the end of every earlier launch that used
//!   the slice, so a writer runs after all earlier readers, not only the last one cudarc recorded. What is not guaranteed: a reader
//!   after a writer on another stream is cudarc's tracking, which needs a context that has made a stream of its own; and uses of a
//!   slice that are not launches of this crate are cudarc's or the host's. An `Err` from a launch does not promise that nothing was
//!   launched (a driver error can come after the kernel was enqueued); the crate then still orders the slices' owners behind it
//!   (best effort) before it returns.
//! * A region has a lifetime ([`DevicePtr<'a>`]); the host that makes one says what it borrows from, and nothing tracks it: not its
//!   ordering against other streams, and not its being freed.
//! * **A CUDA graph is captured with [`Kernel::launch_raw`] or [`Gemm::launch_raw`] and regions** (`unsafe` [`DevicePtr::from_raw`]);
//!   [`Kernel::launch`] with regions on a cudarc stream captures too. A graph holds the addresses it captured and nothing that keeps the memory alive, so the
//!   contract of `from_raw` includes that the graph is not replayed after the memory is gone. **A launch of tensors made of slices is
//!   refused while the launch stream, or a stream a slice is freed on, is capturing** (an `Error::Launch`; in every context: cudarc's
//!   own waits would only refuse it in one that tracks events, and a kernel launched on a stream of a context that
//!   tracks nothing would otherwise be captured with the slices' addresses, and a replay after the free would write into reused memory). There is nothing to switch
//!   off: do not turn cudarc's event tracking off to capture slices, it leaves exactly the graph that outlives its memory. **A whole launch
//!   plan is captured safely** with [`PlanRun::capture`] (below): the graph it returns borrows what it captured.
//! * **The per-thread default stream is refused** by every safe launch (`ctx.per_thread_stream()`, the handle `CU_STREAM_PER_THREAD`), as
//!   the launch stream and as the stream a slice is owned by: its handle is the same in every thread and names another stream in each, so
//!   the ordering above cannot be made across threads (a slice of one thread's per-thread stream launched on another's would look like
//!   a slice of the launch stream, and could be freed under the kernel). Use `ctx.new_stream()` or `ctx.default_stream()`.
//!
//! # Launch plans
//!
//! A *launch plan* (`plan.json`, `docs/plan-schema-1.md`) is a `@tilefish.graph` compiled by Python into, for every bucket of its dimensions,
//! the ordered launches of manifest entries over the caller's tensors (the *externals*) and one *arena* (the intermediates, the device scalars,
//! the cuBLASLt workspace). [`LaunchPlan::load`] reads and verifies it (a pin, the digests, every rule of the format, the manifest it was built
//! against, every step against its entry); [`LaunchPlan::instantiate`] loads its kernels and GEMMs and allocates the arena, once;
//! [`PlanRun::launch`] runs a whole forward pass, checking every step before the first launch; [`PlanRun::capture`] captures it in a CUDA
//! graph, a [`CapturedPlan`] that owns the instantiation and borrows the externals, so its safe [`CapturedPlan::replay`] can never run on
//! memory that is gone.
//!
//! ```no_run
//! use tilefish::cudarc::driver::CudaContext;
//! use tilefish::{Bf16, DevScalar, LaunchPlan, Pin, TensorMut, TensorRef};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # if !tilefish::driver_available() { return Ok(()); }
//! let plan = LaunchPlan::load("build/", Pin::Digest("<the digest of the plan, which tilefish.runtime.export returned>"))?;
//! let ctx = CudaContext::new(0)?;
//! let stream = ctx.new_stream()?;
//! let run = plan.instantiate_all(&stream)?; // every bucket: the binaries, the arena; nothing is allocated after this
//! // (a graph of one input `x`, one returned value `out0` and one device scalar `n`; the weights would be more `TensorRef`s)
//! let (x_shape, y_shape) = ([64usize, 256], [64usize, 256]);
//! let (mut x, mut y) = (stream.alloc_zeros::<u8>(64 * 256 * 2)?, stream.alloc_zeros::<u8>(64 * 256 * 2)?);
//! // one forward pass: the bucket is the one the shapes select (M = 64), the device scalar is written into its slot first
//! run.launch(
//!     &stream,
//!     &mut [
//!         ("x", TensorRef::<Bf16>::from_cuda(&x, &x_shape)?.into_arg()),
//!         ("out0", TensorMut::<Bf16>::from_cuda(&mut y, &y_shape)?.into_arg()),
//!     ],
//!     &[("n", DevScalar::Int(64))],
//! )?;
//! // the same, captured: the graph borrows x and y, so they cannot be freed while it can be replayed
//! let mut graph = run.capture(
//!     &stream,
//!     vec![
//!         ("x", TensorMut::<Bf16>::from_cuda(&mut x, &x_shape)?.into_arg()), // a TensorMut: rewritten between replays
//!         ("out0", TensorMut::<Bf16>::from_cuda(&mut y, &y_shape)?.into_arg()),
//!     ],
//!     &[("n", DevScalar::Int(64))],
//! )?;
//! graph.replay(&stream)?;
//! stream.memcpy_htod(&vec![0u8; 64 * 256 * 2], &mut graph.external_mut("x")?)?; // new contents, the same address
//! graph.set_devscalar("n", 17)?;
//! graph.replay(&stream)?;
//! # Ok(())
//! # }
//! ```
//!
//! # `unsafe`
//!
//! The CUDA driver is cudarc's safe layer (contexts, modules, functions, attributes, streams, the launch builder, graphs), with four calls to
//! `cudarc::driver::sys` in this crate, the probe of [`driver_available`], the version of the driver (`cuDriverGetVersion`: a driver older than
//! 12.4 is refused before a function it lacks is called, which would panic in cudarc), the query of the parameter sizes of a loaded function
//! (`cuFuncGetParamInfo`) and the eager load of a function of a plan (`cuFuncLoad`), and one to `cudarc::driver::result`, the launch of a
//! captured graph on another stream than its own; cuBLASLt is cudarc's `cublaslt::result` and `sys` (its safe `CudaBlasLT::matmul` always runs
//! the heuristic and cannot take a pinned algorithm). There is no hand-written driver FFI. A kernel or a GEMM holds the context it was loaded in
//! ([`Manifest::kernel`] and [`Manifest::gemm`] take an `&Arc<CudaContext>`), and so keeps it, its module (or its cuBLASLt handle) and its
//! function alive. The host takes on obligations in exactly two places (and the items of `src/region.rs`, `src/ffi.rs`,
//! `src/ffi/cublaslt.rs` and `src/ffi/plan.rs`, the only ones with `unsafe`, each say why they are sound):
//!
//! * [`DevicePtr::from_raw`] (`unsafe`): the host vouches for a region of device memory, by address, byte length, device and lifetime:
//!   that it is allocated there, and stays allocated (and is not raced on) until every launch that was given a tensor made of it has
//!   completed (a captured launch is launched again at every replay of its graph). A tensor is *refused* when its shape needs more bytes than its region has, and the sub-regions (`offset`, `slice`) are
//!   checked, so a launch cannot reach past what was vouched for. A tensor over a cudarc `CudaSlice` or `CudaView` needs no
//!   `unsafe` ([`TensorRef::from_cuda`]): its length is known, the borrow keeps it alive, and the launch orders its free after the
//!   kernel (see above).
//! * [`Kernel::launch_raw`] and [`Gemm::launch_raw`] (`unsafe`): the host vouches for a raw `CUstream` (a valid handle of the kernel's
//!   context). The context is made current for the call and what was current is put back. The safe [`Kernel::launch`] and
//!   [`Gemm::launch`] take cudarc's [`CudaStream`] and check that it is of the context they were loaded in.
//!
//! Everything else is safe: a launch is checked against the manifest (shapes, dtypes, alignment, aliasing, buckets, grid) before it
//! reaches the driver, and nothing is launched when a check fails.

// `unsafe` is denied everywhere; the few items that need it (in `ffi`, its `cublaslt` submodule and `region`) allow it one by one, with the reason
// beside them.
#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod elf;
mod error;
mod expr;
mod ffi;
mod gemm;
pub mod json;
mod kernel;
mod launch_plan;
mod library;
mod manifest;
mod memory;
mod plan;
mod plan_run;
mod region;
mod tensor;

pub use cudarc;
pub use error::{Error, PreconditionKind, Result};
pub use ffi::driver_available;
pub use gemm::{Gemm, cublaslt_version};
pub use kernel::{Arch, CUstream, CudaContext, CudaStream, Kernel};
pub use launch_plan::{
    ARENA_ALIGNMENT, BufferRole, LaunchPlan, PLAN_NAME, PLAN_SCHEMA, PlanArg, PlanBucket, PlanBuffer, PlanStep,
    PlanView, StepKind, plan_digest,
};
pub use library::{LibraryEntry, LibraryImpl};
pub use manifest::{
    Access, BinarySource, Constexpr, DimInfo, Entry, EntryKind, FixedValue, LaunchAttributes, MANIFEST_NAME,
    MANIFEST_SCHEMA, Manifest, Param, ParamKind, Pin, Precondition, ScalarType, ShapeDim, StrideEntry, StrideOf, View,
    manifest_digest,
};
pub use memory::{CudaRead, CudaWrite};
pub use plan::{DeviceInfo, Plan, Prepared};
pub use plan_run::{CapturedPlan, DevScalar, PlanRun};
pub use region::DevicePtr;
pub use tensor::{Bf16, DType, E4m3, E5m2, Element, F16, F32, I32, I64, Readable, TensorArg, TensorMut, TensorRef, U8};

/// What the compiler refuses, as tests (`cargo test --doc`): each `compile_fail` block is paired with a `no_run` block that is the
/// same program without the one line that must not compile, so a failure of the first is about that line.
#[cfg(doctest)]
mod safety_doctests {
    /// A tensor made of a slice has an address that is a number, and nothing to make a region of again. (The method that gave the
    /// region back, `ptr()`, is why a launch could be given a region of a slice that was freed and its address reused.)
    ///
    /// ```no_run
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{Bf16, TensorRef};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.new_stream()?;
    /// let x = stream.alloc_zeros::<u8>(64)?;
    /// let shape = [32usize];
    /// let tensor = TensorRef::<Bf16>::from_cuda(&x, &shape)?;
    /// let _address: u64 = tensor.addr();
    /// # Ok(()) }
    /// ```
    ///
    /// ```compile_fail
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{Bf16, TensorRef};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.new_stream()?;
    /// let x = stream.alloc_zeros::<u8>(64)?;
    /// let shape = [32usize];
    /// let tensor = TensorRef::<Bf16>::from_cuda(&x, &shape)?;
    /// let _region = tensor.ptr(); // there is no such method
    /// # Ok(()) }
    /// ```
    fn a_tensor_has_no_region_to_give_back() {}

    /// A tensor borrows its slice: the slice cannot be dropped (freed) while the tensor is alive.
    ///
    /// ```no_run
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{Bf16, TensorRef};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.new_stream()?;
    /// let x = stream.alloc_zeros::<u8>(64)?;
    /// let shape = [32usize];
    /// let tensor = TensorRef::<Bf16>::from_cuda(&x, &shape)?;
    /// let _ = tensor.addr();
    /// drop(x); // after the last use of the tensor
    /// # Ok(()) }
    /// ```
    ///
    /// ```compile_fail
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{Bf16, TensorRef};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.new_stream()?;
    /// let x = stream.alloc_zeros::<u8>(64)?;
    /// let shape = [32usize];
    /// let tensor = TensorRef::<Bf16>::from_cuda(&x, &shape)?;
    /// drop(x);
    /// let _ = tensor.addr(); // the tensor is used after its slice is gone
    /// # Ok(()) }
    /// ```
    fn a_slice_cannot_be_freed_under_its_tensor() {}

    /// What a tensor turns into for a launch borrows the slice too: it cannot be kept past it.
    ///
    /// ```no_run
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{Bf16, TensorRef};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.new_stream()?;
    /// let x = stream.alloc_zeros::<u8>(64)?;
    /// let shape = [32usize];
    /// let arg = TensorRef::<Bf16>::from_cuda(&x, &shape)?.into_arg();
    /// let _ = arg.addr();
    /// # Ok(()) }
    /// ```
    ///
    /// ```compile_fail
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{Bf16, TensorRef};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.new_stream()?;
    /// let shape = [32usize];
    /// let arg = {
    ///     let x = stream.alloc_zeros::<u8>(64)?;
    ///     TensorRef::<Bf16>::from_cuda(&x, &shape)?.into_arg()
    /// };
    /// let _ = arg.addr(); // the slice was freed at the end of the block
    /// # Ok(()) }
    /// ```
    fn a_launch_argument_cannot_outlive_its_slice() {}

    /// A view borrows its slice as a dense tensor does (and its strides, which a launch reads): the slice cannot be freed under it, and the
    /// strides cannot go out of scope first.
    ///
    /// ```no_run
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{Bf16, TensorRef};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.new_stream()?;
    /// let x = stream.alloc_zeros::<u8>(256)?;
    /// let (shape, strides) = ([4usize, 8], [16usize, 1]);
    /// let view = TensorRef::<Bf16>::from_cuda_strided(&x, &shape, &strides, 2)?;
    /// let _ = view.addr();
    /// drop(x); // after the last use of the view
    /// # Ok(()) }
    /// ```
    ///
    /// ```compile_fail
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{Bf16, TensorRef};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.new_stream()?;
    /// let x = stream.alloc_zeros::<u8>(256)?;
    /// let (shape, strides) = ([4usize, 8], [16usize, 1]);
    /// let view = TensorRef::<Bf16>::from_cuda_strided(&x, &shape, &strides, 2)?;
    /// drop(x);
    /// let _ = view.addr(); // the view is used after its slice is gone
    /// # Ok(()) }
    /// ```
    fn a_view_cannot_outlive_its_slice() {}

    /// The reviewer's repro, whole: the pointers of two tensors, the slices dropped (and their memory reused), a launch with tensors
    /// made again from the pointers. It cannot be written: there is no pointer to take, and a tensor is gone before its slice.
    ///
    /// ```compile_fail
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{F32, Kernel, TensorMut, TensorRef};
    /// # fn run(kernel: &Kernel) -> Result<(), Box<dyn std::error::Error>> {
    /// let ctx = CudaContext::new(0)?;
    /// let s = ctx.new_stream()?;
    /// let (x, mut y) = (s.alloc_zeros::<u8>(64)?, s.alloc_zeros::<u8>(64)?);
    /// let shape = [16usize];
    /// let px = TensorRef::<F32>::from_cuda(&x, &shape)?.ptr();
    /// let py = TensorMut::<F32>::from_cuda(&mut y, &shape)?.ptr();
    /// drop(x);
    /// drop(y);
    /// kernel.launch(&s, &mut [TensorRef::<F32>::new(px, &shape)?.into_arg(), TensorMut::<F32>::new(py, &shape)?.into_arg()])?;
    /// # Ok(()) }
    /// ```
    fn the_use_after_free_of_the_review_does_not_compile() {}

    /// A region that a host vouches for has the lifetime the host gives it: made from a borrow of what owns the memory, a tensor of it
    /// cannot outlive that owner.
    ///
    /// ```no_run
    /// use tilefish::{Bf16, DevicePtr, TensorRef};
    /// struct Owner(u64); // memory that the host owns, by address
    /// impl Owner {
    ///     fn region(&self) -> DevicePtr<'_> {
    ///         // SAFETY: (a doctest that is never run) the owner's memory is 64 bytes on device 0 for as long as the owner lives.
    ///         unsafe { DevicePtr::from_raw(self.0, 64, 0) }
    ///     }
    /// }
    /// let owner = Owner(0x7000_0000_0000);
    /// let shape = [32usize];
    /// let tensor = TensorRef::<Bf16>::new(owner.region(), &shape).unwrap();
    /// let _ = tensor.addr();
    /// drop(owner); // after the last use of the tensor
    /// ```
    ///
    /// ```compile_fail
    /// use tilefish::{Bf16, DevicePtr, TensorRef};
    /// struct Owner(u64);
    /// impl Owner {
    ///     fn region(&self) -> DevicePtr<'_> {
    ///         // SAFETY: (a doctest that is never run) as above.
    ///         unsafe { DevicePtr::from_raw(self.0, 64, 0) }
    ///     }
    /// }
    /// let owner = Owner(0x7000_0000_0000);
    /// let shape = [32usize];
    /// let tensor = TensorRef::<Bf16>::new(owner.region(), &shape).unwrap();
    /// drop(owner);
    /// let _ = tensor.addr(); // the tensor is used after its owner is gone
    /// ```
    fn a_region_is_bound_to_what_it_borrows_from() {}

    /// What `from_cuda` takes is a cudarc `CudaSlice`, `CudaView` or `CudaViewMut` and nothing else: a user type that implements
    /// cudarc's `DevicePtr` (and so can say any address and any length) is not one, and the traits that say what is cannot be
    /// implemented outside this crate.
    ///
    /// ```no_run
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{Bf16, TensorMut, TensorRef};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.new_stream()?;
    /// let mut x = stream.alloc_zeros::<u8>(64)?;
    /// let shape = [32usize];
    /// let _read = TensorRef::<Bf16>::from_cuda(&x, &shape)?;
    /// let _write = TensorMut::<Bf16>::from_cuda(&mut x, &shape)?;
    /// let view = x.as_view();
    /// let _of_a_view = TensorRef::<Bf16>::from_cuda(&view, &shape)?;
    /// # Ok(()) }
    /// ```
    ///
    /// ```compile_fail
    /// use std::sync::Arc;
    /// use tilefish::cudarc::driver::{CudaStream, DevicePtr, DeviceSlice, sys};
    /// use tilefish::{Bf16, TensorRef};
    ///
    /// struct Lying(Arc<CudaStream>);
    /// impl DeviceSlice<u8> for Lying {
    ///     fn len(&self) -> usize { usize::MAX / 2 } // any length at all
    ///     fn stream(&self) -> &Arc<CudaStream> { &self.0 }
    /// }
    /// impl DevicePtr<u8> for Lying {
    ///     fn device_ptr<'a>(&'a self, _: &'a CudaStream) -> (sys::CUdeviceptr, tilefish::cudarc::driver::SyncOnDrop<'a>) {
    ///         unimplemented!()
    ///     }
    /// }
    /// # fn run(lying: &Lying) -> Result<(), Box<dyn std::error::Error>> {
    /// let shape = [1usize << 20];
    /// let _ = TensorRef::<Bf16>::from_cuda(lying, &shape)?; // `Lying` is not a CudaRead
    /// # Ok(()) }
    /// ```
    ///
    /// ```compile_fail
    /// use tilefish::CudaRead;
    /// struct Mine;
    /// impl CudaRead for Mine {} // a CudaRead is a type of this crate's list: the supertrait cannot be named, let alone implemented
    /// ```
    fn from_cuda_takes_cudarcs_memory_and_nothing_else() {}

    /// A captured plan borrows the externals it captured: the graph cannot outlive one of them, so it can never be replayed on memory that
    /// was freed (and reused).
    ///
    /// ```no_run
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{Bf16, LaunchPlan, Pin, TensorMut, TensorRef};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let plan = LaunchPlan::load("build/", Pin::TrustDirectory)?;
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.new_stream()?;
    /// let run = plan.instantiate_all(&stream)?;
    /// let shape = [64usize, 256];
    /// let (x, mut y) = (stream.alloc_zeros::<u8>(64 * 256 * 2)?, stream.alloc_zeros::<u8>(64 * 256 * 2)?);
    /// let externals = vec![
    ///     ("x", TensorRef::<Bf16>::from_cuda(&x, &shape)?.into_arg()),
    ///     ("out0", TensorMut::<Bf16>::from_cuda(&mut y, &shape)?.into_arg()),
    /// ];
    /// let mut graph = run.capture(&stream, externals, &[])?;
    /// graph.replay(&stream)?;
    /// drop(graph);
    /// drop(x); // after the graph is gone
    /// # Ok(()) }
    /// ```
    ///
    /// ```compile_fail
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{Bf16, LaunchPlan, Pin, TensorMut, TensorRef};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let plan = LaunchPlan::load("build/", Pin::TrustDirectory)?;
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.new_stream()?;
    /// let run = plan.instantiate_all(&stream)?;
    /// let shape = [64usize, 256];
    /// let (x, mut y) = (stream.alloc_zeros::<u8>(64 * 256 * 2)?, stream.alloc_zeros::<u8>(64 * 256 * 2)?);
    /// let externals = vec![
    ///     ("x", TensorRef::<Bf16>::from_cuda(&x, &shape)?.into_arg()),
    ///     ("out0", TensorMut::<Bf16>::from_cuda(&mut y, &shape)?.into_arg()),
    /// ];
    /// let mut graph = run.capture(&stream, externals, &[])?;
    /// drop(x); // the graph holds its address
    /// graph.replay(&stream)?;
    /// # Ok(()) }
    /// ```
    ///
    /// ```compile_fail
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{Bf16, LaunchPlan, Pin, TensorMut, TensorRef};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let plan = LaunchPlan::load("build/", Pin::TrustDirectory)?;
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.new_stream()?;
    /// let run = plan.instantiate_all(&stream)?;
    /// let shape = [64usize, 256];
    /// let mut y = stream.alloc_zeros::<u8>(64 * 256 * 2)?;
    /// let mut graph = {
    ///     let x = stream.alloc_zeros::<u8>(64 * 256 * 2)?;
    ///     let externals = vec![
    ///         ("x", TensorRef::<Bf16>::from_cuda(&x, &shape)?.into_arg()),
    ///         ("out0", TensorMut::<Bf16>::from_cuda(&mut y, &shape)?.into_arg()),
    ///     ];
    ///     run.capture(&stream, externals, &[])?
    /// }; // x is freed here, and the graph would outlive it
    /// graph.replay(&stream)?;
    /// # Ok(()) }
    /// ```
    fn a_captured_plan_cannot_outlive_an_external() {}

    /// An external that a captured plan holds is written between replays through the captured plan, and only so: the host's own exclusive
    /// borrow of it is the graph's for as long as the graph lives.
    ///
    /// ```no_run
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{Bf16, LaunchPlan, Pin, TensorMut};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let plan = LaunchPlan::load("build/", Pin::TrustDirectory)?;
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.new_stream()?;
    /// let run = plan.instantiate_all(&stream)?;
    /// let shape = [64usize, 256];
    /// let (mut x, mut y) = (stream.alloc_zeros::<u8>(64 * 256 * 2)?, stream.alloc_zeros::<u8>(64 * 256 * 2)?);
    /// let externals = vec![
    ///     ("x", TensorMut::<Bf16>::from_cuda(&mut x, &shape)?.into_arg()),
    ///     ("out0", TensorMut::<Bf16>::from_cuda(&mut y, &shape)?.into_arg()),
    /// ];
    /// let mut graph = run.capture(&stream, externals, &[])?;
    /// stream.memcpy_htod(&vec![0u8; 64 * 256 * 2], &mut graph.external_mut("x")?)?;
    /// graph.replay(&stream)?;
    /// # Ok(()) }
    /// ```
    ///
    /// ```compile_fail
    /// use tilefish::cudarc::driver::CudaContext;
    /// use tilefish::{Bf16, LaunchPlan, Pin, TensorMut};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let plan = LaunchPlan::load("build/", Pin::TrustDirectory)?;
    /// let ctx = CudaContext::new(0)?;
    /// let stream = ctx.new_stream()?;
    /// let run = plan.instantiate_all(&stream)?;
    /// let shape = [64usize, 256];
    /// let (mut x, mut y) = (stream.alloc_zeros::<u8>(64 * 256 * 2)?, stream.alloc_zeros::<u8>(64 * 256 * 2)?);
    /// let externals = vec![
    ///     ("x", TensorMut::<Bf16>::from_cuda(&mut x, &shape)?.into_arg()),
    ///     ("out0", TensorMut::<Bf16>::from_cuda(&mut y, &shape)?.into_arg()),
    /// ];
    /// let mut graph = run.capture(&stream, externals, &[])?;
    /// stream.memcpy_htod(&vec![0u8; 64 * 256 * 2], &mut x)?; // the graph holds the exclusive borrow of x
    /// graph.replay(&stream)?;
    /// # Ok(()) }
    /// ```
    fn an_external_of_a_captured_plan_is_written_through_it() {}
}
