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
//! //    slices, whose length is known: no `unsafe`, and a shape that does not fit its slice is an error, not a launch.
//! let (x, mut y) = (stream.alloc_zeros::<u8>(4096 * 2)?, stream.alloc_zeros::<u8>(4096 * 2)?);
//! let shape = [4096usize];
//! // (keep them until the launch is enqueued: a tensor made of a cudarc slice holds cudarc's event tracking)
//! let x_tensor = TensorRef::<Bf16>::from_cuda(&x, &stream, &shape)?;
//! let y_tensor = TensorMut::<Bf16>::from_cuda(&mut y, &stream, &shape)?;
//! kernel.launch(&stream, &[x_tensor.arg(), y_tensor.arg()])?; // the dimensions and the runtime scalar come from the shapes
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
//!   the dtype and rank of every tensor, `max_elements`, that the runtime scalars (derived from the dimensions, or from the
//!   element count of the tensor they count) fit 32 bits, that nothing written aliases anything else, 16-byte alignment, and
//!   the grid against the device's limits. Nothing is launched when a check fails. There is no way to tell a launch what a
//!   dimension is: that would be a way to run a binary under assumptions the tensors do not satisfy.
//!
//! # `unsafe`
//!
//! The CUDA driver is cudarc's safe layer (contexts, modules, functions, attributes, streams, the launch builder), with no call to
//! `cudarc::driver::sys` in this crate. A kernel holds the context it was loaded in ([`Manifest::kernel`] takes an
//! `&Arc<CudaContext>`), and so keeps it, its module and its function alive. The host takes on obligations in exactly two places
//! (and the five items of `src/region.rs` and `src/ffi.rs`, the only ones with `unsafe`, each say why they are sound):
//!
//! * [`DevicePtr::from_raw`] (`unsafe`): the host vouches for a region of device memory, by address, byte length and device: that it
//!   is allocated there, and stays allocated (and is not raced on) until every launch that was given a tensor made of it has completed.
//!   A tensor is *refused* when its shape needs more bytes than its region has, and the sub-regions (`offset`, `slice`) are checked, so
//!   a launch cannot reach past what was vouched for. A tensor over a cudarc `CudaSlice` or `CudaView` needs no `unsafe`
//!   ([`TensorRef::from_cuda`]): its length is known and the borrow keeps it alive.
//! * [`Kernel::launch_raw`] (`unsafe`): the host vouches for a raw `CUstream` (a valid handle of the kernel's context). The safe
//!   [`Kernel::launch`] takes cudarc's [`CudaStream`] and checks that it is of the kernel's context.
//!
//! Everything else is safe: a launch is checked against the manifest (shapes, dtypes, alignment, aliasing, buckets, grid) before it
//! reaches the driver, and nothing is launched when a check fails.

// `unsafe` is denied everywhere; the few items that need it (in `ffi`) allow it one by one, with the reason beside them.
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
mod plan;
mod region;
mod tensor;

pub use cudarc;
pub use error::{Error, PreconditionKind, Result};
pub use kernel::{Arch, CUstream, CudaContext, CudaStream, Kernel};
pub use manifest::{
    Access, Constexpr, Entry, LaunchAttributes, MANIFEST_NAME, MANIFEST_SCHEMA, Manifest, Param, ParamKind, Pin,
    Precondition, manifest_digest,
};
pub use plan::{DeviceInfo, Plan, Prepared};
pub use region::DevicePtr;
pub use tensor::{Bf16, DType, E4m3, E5m2, Element, F16, F32, I32, I64, TensorArg, TensorMut, TensorRef, U8};
