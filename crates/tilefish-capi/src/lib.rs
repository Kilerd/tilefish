//! The C ABI of the tilefish Rust host: `libtilefish_capi` and `include/tilefish.h` (generated from this crate by cbindgen), with the header-only
//! C++17 wrapper `include/tilefish.hpp`. See `docs/rfcs/0002-cpp-host.md`.
//!
//! Every entry point is a thin layer over the `tilefish` crate: a manifest, a plan and a plan set are loaded and verified by the crate, a
//! kernel, a GEMM and a convolution are launched with its `launch_raw`, and a plan is run, captured and replayed by its `PlanRun` and
//! `CapturedPlan`. Every check of the crate runs before anything is enqueued. What the C ABI adds:
//!
//! * **Tensors are regions** (`tilefish_tensor`): an address, a length and a device that the caller vouches for, as for
//!   `tilefish::DevicePtr::from_raw`; the caller owns and orders its memory, as Python's host does with torch tensors.
//! * **Contexts and streams are the caller's.** A context handle is a device's primary context or a context the caller lends; a stream is a
//!   `CUstream` (a `cudaStream_t`), borrowed for a call. Every call leaves the thread's current context as it found it.
//! * **Errors** are a status per call (`TILEFISH_ERROR_*`, one per variant of `tilefish::Error`, and `INVALID_ARGUMENT`, `NO_DRIVER` and
//!   `INTERNAL`) and the thread's last error (`tilefish_last_error`). No panic unwinds into C.
//! * **Threads.** Every handle may be used from several threads at once, except a captured plan (one thread at a time, any thread). A handle is
//!   not destroyed while another thread uses it.
//!
//! The functions are `unsafe extern "C"`: a C caller passes valid pointers (each function says which may be NULL) and the regions it vouches
//! for. Their contracts are in their docs, which the header carries.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod context;
mod error;
mod launch;
mod manifest;
mod plan;
mod types;

use std::ffi::c_char;

pub use context::*;
pub use error::*;
pub use launch::*;
pub use manifest::*;
pub use plan::*;
pub use types::*;

/// The major version of the ABI this header describes: a library of another major version is not compatible (`tilefish_abi_compatible`).
pub const TILEFISH_ABI_MAJOR: u32 = 1;
/// The minor version of the ABI this header describes: a library of the same major version and at least this minor version has everything
/// the header declares (a minor version only adds functions and values).
pub const TILEFISH_ABI_MINOR: u32 = 0;

/// The ABI version of the library loaded: `TILEFISH_ABI_MAJOR << 16 | TILEFISH_ABI_MINOR` of the header it was built with.
#[unsafe(no_mangle)]
pub extern "C" fn tilefish_abi_version() -> u32 {
    (TILEFISH_ABI_MAJOR << 16) | TILEFISH_ABI_MINOR
}

/// The version of the tilefish crate the library is built from (`"0.0.6"`), a static string.
#[unsafe(no_mangle)]
pub extern "C" fn tilefish_version() -> *const c_char {
    static VERSION: &str = concat!(env!("TILEFISH_CRATE_VERSION"), "\0");
    VERSION.as_ptr().cast()
}

/// The manifest schema the library reads (3).
#[unsafe(no_mangle)]
pub extern "C" fn tilefish_manifest_schema() -> u32 {
    tilefish::MANIFEST_SCHEMA as u32
}

/// The plan schema the library reads (`"tilefish.plan.v1"`), a static string.
#[unsafe(no_mangle)]
pub extern "C" fn tilefish_plan_schema() -> *const c_char {
    static SCHEMA: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
    SCHEMA.get_or_init(|| error::c_string(tilefish::PLAN_SCHEMA)).as_ptr()
}

/// The version of cuBLASLt this process has (`130100` for 13.1.0), or 0 when it cannot be loaded. Opens the library.
#[unsafe(no_mangle)]
pub extern "C" fn tilefish_cublaslt_version() -> u64 {
    error::quiet(0, || tilefish::cublaslt_version().unwrap_or(0))
}

/// The version of cuDNN this process has (`91002` for 9.10.2), or 0 when it cannot be loaded. Opens the library.
#[unsafe(no_mangle)]
pub extern "C" fn tilefish_cudnn_version() -> u64 {
    error::quiet(0, || tilefish::cudnn_version().unwrap_or(0))
}
