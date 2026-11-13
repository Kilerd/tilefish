//! The Rust host of a tilefish manifest.
//!
//! tilefish builds a model once into GPU binaries plus a `manifest.json`; this crate loads that directory and
//! launches its cubins from Rust, with no Python and no Triton, and every check Python's `replay` makes. Together with
//! the typed launchers that `tilefish.hosts.rust.emit` generates from a manifest, it is the "migrate to a
//! high-performance host with zero changes" half of the design, at the level of one kernel: the machine code is the
//! exact bytes Python's `replay` runs.
//!
//! ```no_run
//! use tilefish::{Arch, Args, Manifest, Pin, TensorArg, DType, DevicePtr};
//!
//! # fn main() -> tilefish::Result<()> {
//! // 1. Read and verify the directory. Production pins the digest of the manifest; `Pin::TrustDirectory` trusts the
//! //    directory's own digests, which whoever can write the directory can rewrite.
//! let manifest = Manifest::load("build/", Pin::Digest("<the digest `tilefish.runtime.export` returned>"))?;
//!
//! // 2. Load the binary for the current device. (A CUDA context must be current: cudarc's `CudaContext::new` binds one.)
//! let kernel = manifest.kernel("gelu_erf", &[], Arch::current()?)?;
//!
//! // 3. Launch on your own stream. Every precondition of the binary is checked first.
//! # let (x, y, stream): (DevicePtr, DevicePtr, tilefish::CUstream) = unimplemented!();
//! let shape = [4096usize];
//! let tensors = [TensorArg::new(x, &shape, DType::Bf16, 0), TensorArg::new(y, &shape, DType::Bf16, 0)];
//! kernel.launch(stream, &Args::new(&tensors).scalars(&[("n_elements", 4096)]))?;
//! # Ok(())
//! # }
//! ```
//!
//! # What is checked
//!
//! * **At load** ([`Manifest::load`]): the manifest is strict JSON of schema 1 (no duplicate keys, no `NaN`, no unknown
//!   field, every field of its type); its digest is the pinned one; every cubin is a regular file in the directory,
//!   whose `sha256` and `sha256_code` (the digest of the machine code, [`elf`]) are the manifest's; the entries are
//!   consistent (the ABI, the preconditions, the buckets, the grid). Digests are computed exactly as Python computes them.
//! * **At launch** ([`Kernel::launch`]): the dtype of every tensor, static sizes, `divisible_by`, 16-byte alignment, the
//!   exact bucket, `max_elements`, that the 32-bit scalars fit, that nothing written aliases anything else, and the grid
//!   against the device's limits. Nothing is launched when a check fails.
//!
//! # `unsafe`
//!
//! Only the [`DevicePtr`] constructor is `unsafe`: it is where the host says that an address is device memory that will
//! stay valid. The driver calls are in one module that is not part of the public API.

#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod elf;
mod error;
mod expr;
#[allow(unsafe_code)]
mod ffi;
pub mod json;
mod kernel;
mod manifest;
mod plan;
mod tensor;

pub use error::{Error, PreconditionKind, Result};
pub use ffi::DevicePtr;
pub use kernel::{Arch, CUstream, Kernel};
pub use manifest::{
    Access, Constexpr, Entry, LaunchAttributes, MANIFEST_NAME, MANIFEST_SCHEMA, Manifest, Param, ParamKind, Pin,
    Precondition, manifest_digest,
};
pub use plan::{Args, Plan, Prepared};
pub use tensor::{Bf16, DType, E4m3, E5m2, Element, F16, F32, I32, I64, TensorArg, TensorMut, TensorRef, U8};
