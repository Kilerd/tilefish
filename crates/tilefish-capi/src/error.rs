//! Status codes, the thread's last error, and the boundary every entry point runs behind (no panic unwinds into C).

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use tilefish::{Error, PreconditionKind};

/// The result of a call: `TILEFISH_OK` (0), or one of the `TILEFISH_ERROR_*` codes. `tilefish_last_error` has the details of a failure.
/// A minor version of the ABI may add codes: treat an unknown one as an error.
#[allow(non_camel_case_types)]
pub type tilefish_status = i32;

/// The call succeeded.
pub const TILEFISH_OK: tilefish_status = 0;
/// A file of a manifest or plan directory cannot be read.
pub const TILEFISH_ERROR_IO: tilefish_status = 1;
/// The manifest is refused: not strict JSON of its schema, a digest that does not match a file, an inconsistent entry, a library of another
/// version than the one a GEMM or convolution was pinned with.
pub const TILEFISH_ERROR_MANIFEST: tilefish_status = 2;
/// The pin is not 64 lowercase hex digits.
pub const TILEFISH_ERROR_BAD_PIN: tilefish_status = 3;
/// The manifest has another digest than the pin.
pub const TILEFISH_ERROR_PIN_MISMATCH: tilefish_status = 4;
/// A plan or plan set is refused (a rule of `docs/en/plan-schema-1.md`, another manifest, a step that is not its entry).
pub const TILEFISH_ERROR_PLAN: tilefish_status = 5;
/// The plan or plan set has another digest than the pin.
pub const TILEFISH_ERROR_PLAN_PIN_MISMATCH: tilefish_status = 6;
/// The manifest has no binary for the stem, the buckets or the architecture.
pub const TILEFISH_ERROR_NO_BINARY: tilefish_status = 7;
/// An argument breaks a precondition of the binary (`tilefish_error_info.precondition` says which); nothing was launched.
pub const TILEFISH_ERROR_PRECONDITION: tilefish_status = 8;
/// The binary, the device or the machine cannot run this (a library that cannot be loaded, a grid beyond the device's limits, ...).
pub const TILEFISH_ERROR_UNSUPPORTED: tilefish_status = 9;
/// The call does not fit the signature (a missing or unknown tensor, a value the signature determines), or the stream cannot be used.
pub const TILEFISH_ERROR_LAUNCH: tilefish_status = 10;
/// A cuBLASLt call failed (`tilefish_error_info.call` names it).
pub const TILEFISH_ERROR_CUBLAS: tilefish_status = 11;
/// A cuDNN call failed (`tilefish_error_info.call` names it).
pub const TILEFISH_ERROR_CUDNN: tilefish_status = 12;
/// A CUDA driver call failed (`tilefish_error_info.call` names it, `tilefish_error_info.cu_result` is its `CUresult`).
pub const TILEFISH_ERROR_DRIVER: tilefish_status = 13;
/// An argument of the C call is invalid: a NULL pointer where one is required, an unknown enum value, a string that is not UTF-8, a size
/// that does not fit.
pub const TILEFISH_ERROR_INVALID_ARGUMENT: tilefish_status = 14;
/// There is no CUDA driver (`libcuda`) on this machine.
pub const TILEFISH_ERROR_NO_DRIVER: tilefish_status = 15;
/// A bug in tilefish: a panic, caught at the boundary (the message is the panic's).
pub const TILEFISH_ERROR_INTERNAL: tilefish_status = 16;

/// `tilefish_error_info.precondition` of an error that is not `TILEFISH_ERROR_PRECONDITION`.
pub const TILEFISH_PRECONDITION_NONE: i32 = -1;
/// The tensor is not of the element type the binary was compiled for.
pub const TILEFISH_PRECONDITION_DTYPE: i32 = 0;
/// The tensor has another number of dimensions than the signature says.
pub const TILEFISH_PRECONDITION_RANK: i32 = 1;
/// The shapes of the call cannot be one binding of the signature's dimensions.
pub const TILEFISH_PRECONDITION_SHAPE: i32 = 2;
/// A size the binary has baked in is not the size of the tensor.
pub const TILEFISH_PRECONDITION_STATIC_SIZE: i32 = 3;
/// A pointer is not aligned as the binary was compiled for.
pub const TILEFISH_PRECONDITION_POINTER_ALIGNMENT: i32 = 4;
/// A dimension is not a multiple of what the binary assumes.
pub const TILEFISH_PRECONDITION_DIVISIBLE: i32 = 5;
/// A bucketed dimension is not the bucket the binary was built for (there is no padding).
pub const TILEFISH_PRECONDITION_BUCKET: i32 = 6;
/// A tensor has more elements than the 32-bit offsets of the kernel reach.
pub const TILEFISH_PRECONDITION_MAX_ELEMENTS: i32 = 7;
/// A view reaches further than the 32-bit offsets of the kernel.
pub const TILEFISH_PRECONDITION_MAX_EXTENT: i32 = 8;
/// A view has a stride the binary cannot take.
pub const TILEFISH_PRECONDITION_STRIDE: i32 = 9;
/// A tensor that is not contiguous is given for a parameter that is not declared strided.
pub const TILEFISH_PRECONDITION_CONTIGUOUS: i32 = 10;
/// A tensor that is written overlaps another argument.
pub const TILEFISH_PRECONDITION_ALIAS: i32 = 11;
/// A tensor is not on the device the binary was loaded on.
pub const TILEFISH_PRECONDITION_DEVICE: i32 = 12;
/// A tensor given as `TILEFISH_ACCESS_IN` is given for a parameter the kernel writes.
pub const TILEFISH_PRECONDITION_ACCESS: i32 = 13;
/// A tensor given for a TMA descriptor is one the driver refuses to encode.
pub const TILEFISH_PRECONDITION_TENSOR_MAP: i32 = 14;

/// The details of the last call that failed on this thread (`tilefish_last_error`). The strings stay valid until the next call that fails on
/// this thread.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct tilefish_error_info {
    /// The status the call returned (`TILEFISH_OK` when no call has failed on this thread).
    pub status: tilefish_status,
    /// For `TILEFISH_ERROR_PRECONDITION`, which precondition (`TILEFISH_PRECONDITION_*`); otherwise `TILEFISH_PRECONDITION_NONE`.
    pub precondition: i32,
    /// For `TILEFISH_ERROR_DRIVER`, the `CUresult` of the driver call; otherwise 0.
    pub cu_result: i32,
    /// What went wrong: the message of the tilefish crate, the same text the Rust host prints. Never NULL ("" when no call has failed).
    pub message: *const c_char,
    /// For a precondition, the argument or dimension it is about, when there is one; otherwise NULL.
    pub argument: *const c_char,
    /// For a driver, cuBLASLt or cuDNN error, the function that failed (or what the host was doing); otherwise NULL.
    pub call: *const c_char,
}

/// Why a call failed, before it is recorded.
pub(crate) enum Failure {
    Tilefish(Error),
    Invalid(String),
    NoDriver,
    Internal(String),
}

impl From<Error> for Failure {
    fn from(e: Error) -> Self {
        Self::Tilefish(e)
    }
}

pub(crate) type Outcome<T = ()> = std::result::Result<T, Failure>;

/// An invalid argument of the C call.
pub(crate) fn invalid<T>(message: impl Into<String>) -> Outcome<T> {
    Err(Failure::Invalid(message.into()))
}

struct Last {
    status: tilefish_status,
    precondition: i32,
    cu_result: i32,
    message: CString,
    argument: Option<CString>,
    call: Option<CString>,
}

thread_local! {
    static LAST: RefCell<Last> = RefCell::new(Last {
        status: TILEFISH_OK,
        precondition: TILEFISH_PRECONDITION_NONE,
        cu_result: 0,
        message: CString::default(),
        argument: None,
        call: None,
    });
}

/// A C string of `text` (an interior NUL, which no message of the crate has, becomes a space).
pub(crate) fn c_string(text: &str) -> CString {
    CString::new(text.replace('\0', " ")).expect("no NUL is left")
}

pub(crate) fn precondition_code(kind: PreconditionKind) -> i32 {
    match kind {
        PreconditionKind::Dtype => TILEFISH_PRECONDITION_DTYPE,
        PreconditionKind::Rank => TILEFISH_PRECONDITION_RANK,
        PreconditionKind::Shape => TILEFISH_PRECONDITION_SHAPE,
        PreconditionKind::StaticSize => TILEFISH_PRECONDITION_STATIC_SIZE,
        PreconditionKind::PointerAlignment => TILEFISH_PRECONDITION_POINTER_ALIGNMENT,
        PreconditionKind::Divisible => TILEFISH_PRECONDITION_DIVISIBLE,
        PreconditionKind::Bucket => TILEFISH_PRECONDITION_BUCKET,
        PreconditionKind::MaxElements => TILEFISH_PRECONDITION_MAX_ELEMENTS,
        PreconditionKind::MaxExtent => TILEFISH_PRECONDITION_MAX_EXTENT,
        PreconditionKind::Stride => TILEFISH_PRECONDITION_STRIDE,
        PreconditionKind::Contiguous => TILEFISH_PRECONDITION_CONTIGUOUS,
        PreconditionKind::Alias => TILEFISH_PRECONDITION_ALIAS,
        PreconditionKind::Device => TILEFISH_PRECONDITION_DEVICE,
        PreconditionKind::Access => TILEFISH_PRECONDITION_ACCESS,
        PreconditionKind::TensorMap => TILEFISH_PRECONDITION_TENSOR_MAP,
    }
}

/// The status of a crate error: one code per variant of `tilefish::Error`.
pub(crate) fn status_of(e: &Error) -> tilefish_status {
    match e {
        Error::Io { .. } => TILEFISH_ERROR_IO,
        Error::Manifest(_) => TILEFISH_ERROR_MANIFEST,
        Error::BadPin(_) => TILEFISH_ERROR_BAD_PIN,
        Error::PinMismatch { .. } => TILEFISH_ERROR_PIN_MISMATCH,
        Error::Plan(_) => TILEFISH_ERROR_PLAN,
        Error::PlanPinMismatch { .. } => TILEFISH_ERROR_PLAN_PIN_MISMATCH,
        Error::NoBinary(_) => TILEFISH_ERROR_NO_BINARY,
        Error::Precondition { .. } => TILEFISH_ERROR_PRECONDITION,
        Error::Unsupported(_) => TILEFISH_ERROR_UNSUPPORTED,
        Error::Launch(_) => TILEFISH_ERROR_LAUNCH,
        Error::Cublas { .. } => TILEFISH_ERROR_CUBLAS,
        Error::Cudnn { .. } => TILEFISH_ERROR_CUDNN,
        Error::Driver { .. } => TILEFISH_ERROR_DRIVER,
    }
}

fn record(failure: Failure) -> tilefish_status {
    let mut last = Last {
        status: TILEFISH_ERROR_INTERNAL,
        precondition: TILEFISH_PRECONDITION_NONE,
        cu_result: 0,
        message: CString::default(),
        argument: None,
        call: None,
    };
    match failure {
        Failure::Tilefish(e) => {
            last.status = status_of(&e);
            last.message = c_string(&e.to_string());
            match &e {
                Error::Precondition { kind, arg, .. } => {
                    last.precondition = precondition_code(*kind);
                    last.argument = arg.as_deref().map(c_string);
                }
                Error::Driver { call, source } => {
                    last.cu_result = source.0 as i32;
                    last.call = Some(c_string(call));
                }
                Error::Cublas { call, .. } | Error::Cudnn { call, .. } => last.call = Some(c_string(call)),
                _ => {}
            }
        }
        Failure::Invalid(message) => {
            last.status = TILEFISH_ERROR_INVALID_ARGUMENT;
            last.message = c_string(&message);
        }
        Failure::NoDriver => {
            last.status = TILEFISH_ERROR_NO_DRIVER;
            last.message = c_string(
                "there is no CUDA driver on this machine (libcuda cannot be loaded): loading and verifying manifests and plans works without one, \
                 contexts and launches do not",
            );
        }
        Failure::Internal(message) => {
            last.message = c_string(&format!("a bug in tilefish (a panic, caught at the C boundary): {message}"));
        }
    }
    let status = last.status;
    LAST.with(|cell| *cell.borrow_mut() = last);
    status
}

/// Run the body of an entry point: its failure (or its panic) is recorded as the thread's last error and returned as a status.
pub(crate) fn call(body: impl FnOnce() -> Outcome) -> tilefish_status {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => TILEFISH_OK,
        Ok(Err(failure)) => record(failure),
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic without a message".to_owned());
            record(Failure::Internal(message))
        }
    }
}

/// Run an accessor that cannot fail but must not unwind into C: a panic gives `fallback`.
pub(crate) fn quiet<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(fallback)
}

/// The details of the last call that failed on this thread. `out` must not be NULL. The strings it points at stay valid until the next call
/// that fails on this thread. A call that succeeds changes nothing here.
///
/// # Safety
///
/// `out` points at a writable `tilefish_error_info`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_last_error(out: *mut tilefish_error_info) {
    if out.is_null() {
        return;
    }
    let info = LAST.with(|cell| {
        let last = cell.borrow();
        tilefish_error_info {
            status: last.status,
            precondition: last.precondition,
            cu_result: last.cu_result,
            message: last.message.as_ptr(),
            argument: last.argument.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            call: last.call.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
        }
    });
    // SAFETY: the caller's contract: `out` is a writable `tilefish_error_info` (checked not NULL above).
    unsafe { out.write(info) };
}

/// The name of a status (`"TILEFISH_ERROR_PRECONDITION"`), or `"TILEFISH_ERROR_UNKNOWN"` for a code this library does not have. A static string.
#[unsafe(no_mangle)]
pub extern "C" fn tilefish_status_name(status: tilefish_status) -> *const c_char {
    let name: &'static [u8] = match status {
        TILEFISH_OK => b"TILEFISH_OK\0",
        TILEFISH_ERROR_IO => b"TILEFISH_ERROR_IO\0",
        TILEFISH_ERROR_MANIFEST => b"TILEFISH_ERROR_MANIFEST\0",
        TILEFISH_ERROR_BAD_PIN => b"TILEFISH_ERROR_BAD_PIN\0",
        TILEFISH_ERROR_PIN_MISMATCH => b"TILEFISH_ERROR_PIN_MISMATCH\0",
        TILEFISH_ERROR_PLAN => b"TILEFISH_ERROR_PLAN\0",
        TILEFISH_ERROR_PLAN_PIN_MISMATCH => b"TILEFISH_ERROR_PLAN_PIN_MISMATCH\0",
        TILEFISH_ERROR_NO_BINARY => b"TILEFISH_ERROR_NO_BINARY\0",
        TILEFISH_ERROR_PRECONDITION => b"TILEFISH_ERROR_PRECONDITION\0",
        TILEFISH_ERROR_UNSUPPORTED => b"TILEFISH_ERROR_UNSUPPORTED\0",
        TILEFISH_ERROR_LAUNCH => b"TILEFISH_ERROR_LAUNCH\0",
        TILEFISH_ERROR_CUBLAS => b"TILEFISH_ERROR_CUBLAS\0",
        TILEFISH_ERROR_CUDNN => b"TILEFISH_ERROR_CUDNN\0",
        TILEFISH_ERROR_DRIVER => b"TILEFISH_ERROR_DRIVER\0",
        TILEFISH_ERROR_INVALID_ARGUMENT => b"TILEFISH_ERROR_INVALID_ARGUMENT\0",
        TILEFISH_ERROR_NO_DRIVER => b"TILEFISH_ERROR_NO_DRIVER\0",
        TILEFISH_ERROR_INTERNAL => b"TILEFISH_ERROR_INTERNAL\0",
        _ => b"TILEFISH_ERROR_UNKNOWN\0",
    };
    name.as_ptr().cast()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn last() -> (tilefish_status, i32, String, Option<String>) {
        LAST.with(|cell| {
            let l = cell.borrow();
            (
                l.status,
                l.precondition,
                l.message.to_str().unwrap().to_owned(),
                l.argument.as_ref().map(|a| a.to_str().unwrap().to_owned()),
            )
        })
    }

    /// One status per variant of the crate's error, each distinct.
    #[test]
    fn every_error_of_the_crate_has_its_own_status() {
        let io = Error::Io { path: "p".into(), source: std::io::Error::other("x") };
        let errors = [
            (io, TILEFISH_ERROR_IO),
            (Error::Manifest("m".into()), TILEFISH_ERROR_MANIFEST),
            (Error::BadPin("b".into()), TILEFISH_ERROR_BAD_PIN),
            (
                Error::PinMismatch { directory: "d".into(), found: "f".into(), pinned: "p".into() },
                TILEFISH_ERROR_PIN_MISMATCH,
            ),
            (Error::Plan("p".into()), TILEFISH_ERROR_PLAN),
            (
                Error::PlanPinMismatch { path: "p".into(), found: "f".into(), pinned: "p".into() },
                TILEFISH_ERROR_PLAN_PIN_MISMATCH,
            ),
            (Error::NoBinary("n".into()), TILEFISH_ERROR_NO_BINARY),
            (
                Error::Precondition { kind: PreconditionKind::Alias, arg: Some("y".into()), message: "m".into() },
                TILEFISH_ERROR_PRECONDITION,
            ),
            (Error::Unsupported("u".into()), TILEFISH_ERROR_UNSUPPORTED),
            (Error::Launch("l".into()), TILEFISH_ERROR_LAUNCH),
            (Error::Cublas { call: "cublasLtMatmul", status: "s".into() }, TILEFISH_ERROR_CUBLAS),
            (Error::Cudnn { call: "cudnnCreate", status: "s".into() }, TILEFISH_ERROR_CUDNN),
        ];
        let mut seen = std::collections::BTreeSet::new();
        for (e, status) in errors {
            assert_eq!(status_of(&e), status, "{e:?}");
            assert!(seen.insert(status));
            assert_eq!(call(|| Err(Failure::Tilefish(e))), status);
        }
        assert_eq!(seen.len(), 12);
        let e = Error::Precondition {
            kind: PreconditionKind::Alias,
            arg: Some("y".into()),
            message: "y overlaps x".into(),
        };
        call(|| Err(e.into()));
        assert_eq!(
            last(),
            (TILEFISH_ERROR_PRECONDITION, TILEFISH_PRECONDITION_ALIAS, "y overlaps x".to_owned(), Some("y".to_owned()))
        );
    }

    #[test]
    fn every_precondition_has_its_own_code() {
        use PreconditionKind::*;
        let kinds = [
            Dtype,
            Rank,
            Shape,
            StaticSize,
            PointerAlignment,
            Divisible,
            Bucket,
            MaxElements,
            MaxExtent,
            Stride,
            Contiguous,
            Alias,
            Device,
            Access,
            TensorMap,
        ];
        let codes: std::collections::BTreeSet<i32> = kinds.iter().map(|k| precondition_code(*k)).collect();
        assert_eq!(codes, (0..15).collect());
    }

    /// A panic in an entry point is a status, not an unwind into C; a success leaves the last error as it was.
    #[test]
    fn a_panic_is_caught_and_a_success_changes_nothing() {
        let status = call(|| panic!("boom"));
        assert_eq!(status, TILEFISH_ERROR_INTERNAL);
        assert!(last().2.contains("boom"), "{:?}", last());
        assert_eq!(call(|| Ok(())), TILEFISH_OK);
        assert_eq!(last().0, TILEFISH_ERROR_INTERNAL);
        assert_eq!(quiet(7, || panic!("quiet")), 7);
        // an interior NUL does not lose the message
        call(|| invalid("a\0b"));
        assert_eq!(last().2, "a b");
    }
}
