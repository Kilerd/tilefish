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
//! # What is checked
//!
//! * **At load** ([`Manifest::load`]): the manifest is strict JSON of schema 2 (no duplicate keys, no `NaN`, no unknown
//!   field, every field of its type); its digest is the pinned one; every cubin is a regular file in the directory,
//!   whose `sha256` and `sha256_code` (the digest of the machine code, [`elf`]) are the manifest's; the entries are
//!   consistent (the ABI, the declared shapes, the dimensions, the preconditions, the grid). Digests are computed exactly as
//!   Python computes them. A manifest of an older schema is refused: export it again.
//! * **At launch** ([`Kernel::launch`]): the manifest declares the shape of every pointer (`["Rows", 1024]`) and the constraints
//!   of every named dimension, so a launch needs nothing but the tensors. It binds the dimensions from their sizes, checks
//!   that they agree across arguments and with the static sizes, the exact bucket and `divisible_by` of each dimension,
//!   the dtype and rank of every tensor, that every tensor is on the kernel's device *and in its CUDA context* and that a
//!   tensor that is only read is not given for a parameter that is written, `max_elements`, that the runtime scalars (derived from
//!   the dimensions, or from the element count of the tensor they count) fit 32 bits, that nothing written aliases anything else,
//!   16-byte alignment, and the grid against the device's limits. Nothing is launched when a check fails. There is no way to tell a
//!   launch what a dimension is: that would be a way to run a binary under assumptions the tensors do not satisfy.
//!
//! The one thing Python's `replay` asks the driver at load that this crate does not is the parameter sizes of the loaded function
//! (`cuFuncGetParamInfo`, against the ABI): cudarc has no wrapper for it, and it adds nothing here. The cubin is the one whose
//! `sha256` and `sha256_code` the manifest pinned, the ABI is the one in the manifest's digest, and the PTX entry of every binary was
//! checked against its compile spec when it was built; a pair that disagreed could only be made by someone who can rewrite the
//! pin too.
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
//! * [`Kernel::launch`] hands the slice to cudarc's launch builder on the stream it launches on, which orders the launch after the
//!   slice's earlier writes (and reads, for a write) and records it on the slice, **and** makes the slice's own stream (the one it is
//!   freed on) wait for an event recorded right after the kernel, when that is not the launch stream. So freeing the slice, or any
//!   later work on its stream, runs after the kernel, however many other launches read the slice on other streams. What is not
//!   guaranteed is more than cudarc's one-event-per-slice tracking gives between launches on different streams that write: a
//!   writer waits for the last reader recorded, not for all of them. Order such work on one stream.
//! * A region has a lifetime ([`DevicePtr<'a>`]); the host that makes one says what it borrows from, and nothing tracks it: not its
//!   ordering against other streams, and not its being freed.
//! * **A CUDA graph is captured with [`Kernel::launch_raw`] and regions** (`unsafe` [`DevicePtr::from_raw`]). A graph holds the
//!   addresses it captured and nothing that keeps the memory alive, so the contract of `from_raw` includes that the graph is not
//!   replayed after the memory is gone. The safe [`Kernel::launch`] on a capturing stream with a tensor made of a slice is an
//!   error (the driver refuses cudarc's waits on events recorded outside the capture, `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`), not a
//!   graph. There is nothing to switch off: do not turn cudarc's event tracking off to capture slices, it leaves exactly the graph
//!   that outlives its memory.
//!
//! # `unsafe`
//!
//! The CUDA driver is cudarc's safe layer (contexts, modules, functions, attributes, streams, the launch builder), with one call to
//! `cudarc::driver::sys` in this crate, the probe of [`driver_available`]. A kernel holds the context it was loaded in
//! ([`Manifest::kernel`] takes an `&Arc<CudaContext>`), and so keeps it, its module and its function alive. The host takes on
//! obligations in exactly two places (and the five items of `src/region.rs` and `src/ffi.rs`, the only ones with `unsafe`, each say
//! why they are sound):
//!
//! * [`DevicePtr::from_raw`] (`unsafe`): the host vouches for a region of device memory, by address, byte length, device and lifetime:
//!   that it is allocated there, and stays allocated (and is not raced on) until every launch that was given a tensor made of it has
//!   completed (a captured launch is launched again at every replay of its graph). A tensor is *refused* when its shape needs more bytes than its region has, and the sub-regions (`offset`, `slice`) are
//!   checked, so a launch cannot reach past what was vouched for. A tensor over a cudarc `CudaSlice` or `CudaView` needs no
//!   `unsafe` ([`TensorRef::from_cuda`]): its length is known, the borrow keeps it alive, and the launch orders its free after the
//!   kernel (see above).
//! * [`Kernel::launch_raw`] (`unsafe`): the host vouches for a raw `CUstream` (a valid handle of the kernel's context). The kernel's
//!   context is made current for the call and what was current is put back. The safe [`Kernel::launch`] takes cudarc's
//!   [`CudaStream`] and checks that it is of the kernel's context.
//!
//! Everything else is safe: a launch is checked against the manifest (shapes, dtypes, alignment, aliasing, buckets, grid) before it
//! reaches the driver, and nothing is launched when a check fails.

// `unsafe` is denied everywhere; the few items that need it (in `ffi` and `region`) allow it one by one, with the reason beside them.
#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod elf;
mod error;
mod expr;
mod ffi;
pub mod json;
mod kernel;
mod manifest;
mod memory;
mod plan;
mod region;
mod tensor;

pub use cudarc;
pub use error::{Error, PreconditionKind, Result};
pub use ffi::driver_available;
pub use kernel::{Arch, CUstream, CudaContext, CudaStream, Kernel};
pub use manifest::{
    Access, Constexpr, Entry, LaunchAttributes, MANIFEST_NAME, MANIFEST_SCHEMA, Manifest, Param, ParamKind, Pin,
    Precondition, manifest_digest,
};
pub use memory::{CudaRead, CudaWrite};
pub use plan::{DeviceInfo, Plan, Prepared};
pub use region::DevicePtr;
pub use tensor::{Bf16, DType, E4m3, E5m2, Element, F16, F32, I32, I64, TensorArg, TensorMut, TensorRef, U8};

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
}
