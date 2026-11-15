//! Manifests: loaded from a directory or from memory, verified, pinned.

use std::ffi::{CString, c_char};

use tilefish::{Manifest, Pin};

use crate::context::destroy_keeping_current;
use crate::error::{Outcome, c_string, call, invalid, quiet, tilefish_status};
use crate::types::{handle, out_arg, put, slice_arg, str_arg, tilefish_file};

/// A manifest directory (or the files of one, in memory), read strictly and verified: its digest against the pin, every file against its
/// digests, every entry. Cheap to copy in the crate; loads through copies share their modules.
#[allow(non_camel_case_types)]
pub struct tilefish_manifest {
    pub(crate) manifest: Manifest,
    digest: CString,
    label: CString,
}

impl tilefish_manifest {
    pub(crate) fn new(manifest: Manifest) -> Self {
        let digest = c_string(manifest.digest());
        let label = c_string(manifest.label());
        Self { manifest, digest, label }
    }
}

/// Load and verify the manifest directory `directory` (or its `manifest.json`), pinned to `pin`, the 64 lowercase hex digits of its digest
/// (`tilefish.runtime.export` returns it). A manifest with another digest is `TILEFISH_ERROR_PIN_MISMATCH`. Needs no driver.
///
/// # Safety
///
/// `directory` and `pin` are NUL-terminated strings; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_manifest_load(
    directory: *const c_char,
    pin: *const c_char,
    out: *mut *mut tilefish_manifest,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract.
        let out = unsafe { out_arg(out) }?;
        // SAFETY: as above.
        let (directory, pin) = unsafe { (str_arg(directory, "the directory")?, str_arg(pin, "the pin")?) };
        let manifest = Manifest::load(directory, Pin::Digest(pin))?;
        // SAFETY: as above.
        unsafe { put(out, tilefish_manifest::new(manifest)) }
    })
}

/// Load and verify a manifest directory trusting its own digests (whoever can write the directory can rewrite them): for development.
/// Production pins the digest (`tilefish_manifest_load`).
///
/// # Safety
///
/// `directory` is a NUL-terminated string; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_manifest_load_trusting_directory(
    directory: *const c_char,
    out: *mut *mut tilefish_manifest,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract.
        let out = unsafe { out_arg(out) }?;
        // SAFETY: as above.
        let directory = unsafe { str_arg(directory, "the directory") }?;
        let manifest = Manifest::load(directory, Pin::TrustDirectory)?;
        // SAFETY: as above.
        unsafe { put(out, tilefish_manifest::new(manifest)) }
    })
}

/// A manifest whose files are in memory (what a generated header embeds): `manifest` is the contents of `manifest.json` (`manifest_len`
/// bytes), and `files` its cubins by file name. Verified exactly like a directory, and pinned to `pin`. The bytes are copied: they need not
/// outlive the call.
///
/// # Safety
///
/// `manifest` points at `manifest_len` bytes; `files` at `n_files` files whose names and bytes are valid (NULL when `n_files` is 0); `pin` is
/// a NUL-terminated string; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_manifest_from_embedded(
    manifest: *const u8,
    manifest_len: usize,
    files: *const tilefish_file,
    n_files: usize,
    pin: *const c_char,
    out: *mut *mut tilefish_manifest,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract.
        let out = unsafe { out_arg(out) }?;
        if manifest.is_null() {
            return invalid("the manifest bytes are NULL");
        }
        // SAFETY: as above.
        let bytes = unsafe { std::slice::from_raw_parts(manifest, manifest_len) };
        // SAFETY: as above.
        let files = unsafe { slice_arg(files, n_files, "files") }?;
        let cubins = files
            .iter()
            .map(|f| {
                // SAFETY: as above.
                let name = unsafe { str_arg(f.name, "the name of a file") }?;
                // SAFETY: as above.
                let data = unsafe { slice_arg(f.bytes, f.len, &format!("the bytes of '{name}'")) }?;
                Ok((name, data))
            })
            .collect::<Outcome<Vec<(&str, &[u8])>>>()?;
        // SAFETY: as above.
        let pin = unsafe { str_arg(pin, "the pin") }?;
        let manifest = Manifest::from_embedded(bytes, &cubins, Pin::Digest(pin))?;
        // SAFETY: as above.
        unsafe { put(out, tilefish_manifest::new(manifest)) }
    })
}

/// A copy of `manifest` for a host that runs on a partition of `sms` of the device's SMs (`tilefish::Manifest::with_sm_partition`; 0 for the
/// whole device): persistent grids are sized for it, and library entries chosen for another SM count are refused at load.
///
/// # Safety
///
/// `manifest` is a live handle; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_manifest_with_sm_partition(
    manifest: *const tilefish_manifest,
    sms: u64,
    out: *mut *mut tilefish_manifest,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract.
        let out = unsafe { out_arg(out) }?;
        // SAFETY: as above.
        let manifest = unsafe { handle(manifest, "the manifest") }?;
        let copy = manifest.manifest.with_sm_partition((sms != 0).then_some(sms));
        // SAFETY: as above.
        unsafe { put(out, tilefish_manifest::new(copy)) }
    })
}

/// Refuse a manifest whose IEEE math functions were measured under other CUDA math libraries than release `cuda` (`"13.0"`): see
/// `tilefish::Manifest::check_cuda_math`. Entries without the field pass.
///
/// # Safety
///
/// `manifest` is a live handle; `cuda` is a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_manifest_check_cuda_math(
    manifest: *const tilefish_manifest,
    cuda: *const c_char,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract.
        let manifest = unsafe { handle(manifest, "the manifest") }?;
        // SAFETY: as above.
        let cuda = unsafe { str_arg(cuda, "the CUDA release") }?;
        Ok(manifest.manifest.check_cuda_math(cuda)?)
    })
}

/// The digest of a manifest (64 lowercase hex digits), owned by the handle (NULL for NULL).
///
/// # Safety
///
/// `manifest` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_manifest_digest(manifest: *const tilefish_manifest) -> *const c_char {
    // SAFETY: the caller's contract.
    quiet(std::ptr::null(), || unsafe { manifest.as_ref() }.map_or(std::ptr::null(), |m| m.digest.as_ptr()))
}

/// Where a manifest was read from (its directory, or `"<embedded>"`), owned by the handle (NULL for NULL).
///
/// # Safety
///
/// `manifest` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_manifest_label(manifest: *const tilefish_manifest) -> *const c_char {
    // SAFETY: the caller's contract.
    quiet(std::ptr::null(), || unsafe { manifest.as_ref() }.map_or(std::ptr::null(), |m| m.label.as_ptr()))
}

/// Destroy a manifest handle (NULL is a no-op). What was loaded from it lives on.
///
/// # Safety
///
/// `manifest` is NULL or a live handle, not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_manifest_destroy(manifest: *mut tilefish_manifest) {
    // SAFETY: the caller's contract.
    quiet((), || unsafe { destroy_keeping_current(manifest) })
}
