//! CUDA contexts: the device's primary context (the one the CUDA runtime API uses), or a context the caller made and lends.

use std::sync::{Arc, OnceLock};

use tilefish::cudarc::driver::sys::{self, CUcontext, CUstream};
use tilefish::cudarc::driver::{CudaContext, CudaStream, DriverError, result};
use tilefish::{Arch, CudnnHandle, Error};

use crate::error::{Failure, Outcome, call, quiet, tilefish_status};
use crate::types::{KeepCurrent, destroy, handle, out_arg, put, put_value};

fn driver(call: &'static str) -> impl FnOnce(DriverError) -> Failure {
    move |source| Failure::Tilefish(Error::Driver { call, source })
}

/// What a context handle shares with every handle made in it: the crate's context, and a stream of tilefish's own.
///
/// The stream is never used for work. It is there because cudarc orders its own memory between streams (the memory the crate allocates: an
/// arena's, a GEMM's workspace in it, the staging of device scalars) only in a context that has made a stream of its own; the caller's streams
/// are borrowed, which does not count. Every handle made in the context holds this value, so the context keeps that mode for as long as any of
/// them lives.
pub(crate) struct Ctx {
    _tracking: Arc<CudaStream>,
    pub(crate) ctx: Arc<CudaContext>,
}

impl Ctx {
    fn new(ctx: Arc<CudaContext>) -> Outcome<Arc<Self>> {
        // `new_stream` synchronizes the context when it is the first stream of this (new) `CudaContext`, for the memory cudarc tracked before
        // it: there is none (nothing was allocated through this value yet), so that wait is skipped.
        // SAFETY: no `CudaSlice` is made between the two calls; the streams made meanwhile track nothing themselves.
        unsafe { ctx.disable_event_tracking() };
        let tracking = ctx.new_stream();
        // SAFETY: as above.
        unsafe { ctx.enable_event_tracking() };
        let tracking = tracking.map_err(driver("cuStreamCreate"))?;
        Ok(Arc::new(Self { _tracking: tracking, ctx }))
    }

    /// The caller's stream, borrowed, as the crate's `CudaStream` of this context.
    ///
    /// # Safety
    ///
    /// `stream` is a valid stream of this context (or NULL, the legacy default stream), for as long as the returned value is used.
    pub(crate) unsafe fn borrow(&self, stream: CUstream) -> Arc<CudaStream> {
        // SAFETY: the caller's contract.
        unsafe { self.ctx.wrap_borrowed_stream(stream) }
    }
}

/// A CUDA context: the primary context of a device (`tilefish_context_primary`), or a context the caller lends (`tilefish_context_borrow`).
/// Every handle made in it keeps what it needs of it alive, so a context handle may be destroyed before them.
#[allow(non_camel_case_types)]
pub struct tilefish_context {
    pub(crate) inner: Arc<Ctx>,
}

/// The cuDNN handle of a context, made early (`tilefish_cudnn_handle_create`).
#[allow(non_camel_case_types)]
pub struct tilefish_cudnn_handle {
    _handle: CudnnHandle,
    _ctx: Arc<Ctx>,
}

static DRIVER: OnceLock<bool> = OnceLock::new();

pub(crate) fn require_driver() -> Outcome {
    if *DRIVER.get_or_init(tilefish::driver_available) { Ok(()) } else { Err(Failure::NoDriver) }
}

/// Destroy a handle, putting back the thread's current context afterwards (dropping a module, a cuBLASLt or cuDNN handle, an arena or a graph
/// binds its context).
///
/// # Safety
///
/// As for [`destroy`].
pub(crate) unsafe fn destroy_keeping_current<T>(p: *mut T) {
    let keep = if DRIVER.get().copied() == Some(true) { KeepCurrent::new().ok() } else { None };
    // SAFETY: the caller's contract.
    unsafe { destroy(p) };
    drop(keep);
}

/// Whether the CUDA driver (`libcuda`) can be loaded on this machine. Loading and verifying manifests and plans needs no driver; a context and
/// everything made in one do (`TILEFISH_ERROR_NO_DRIVER` otherwise). The first call opens the library.
#[unsafe(no_mangle)]
pub extern "C" fn tilefish_driver_available() -> bool {
    quiet(false, || require_driver().is_ok())
}

/// The context handle of the primary context of device `device` (an ordinal): retained now, and released when the last handle made in it is
/// destroyed. This is the context the CUDA runtime API uses (`cudaSetDevice`), so a host built on the runtime API passes its `cudaStream_t`s
/// to launches in it.
///
/// # Safety
///
/// `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_context_primary(device: i32, out: *mut *mut tilefish_context) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract.
        let out = unsafe { out_arg(out) }?;
        require_driver()?;
        result::init().map_err(driver("cuInit"))?; // before anything else asks the driver (cuCtxGetCurrent too)
        let _keep = KeepCurrent::new()?;
        let count = result::device::get_count().map_err(driver("cuDeviceGetCount"))?;
        if device < 0 || device >= count {
            return Err(Failure::Invalid(format!("there is no device {device}: this machine has {count}")));
        }
        let ctx = CudaContext::new(device as usize).map_err(driver("cuDevicePrimaryCtxRetain"))?;
        // SAFETY: as above.
        unsafe { put(out, tilefish_context { inner: Ctx::new(ctx)? }) }
    })
}

/// The context handle of a context the caller made (`cuCtxCreate`, or `cuCtxFromGreenCtx` for a partition of the device's SMs), or of the
/// primary context of a device (which is then retained and released as by `tilefish_context_primary`).
///
/// tilefish never destroys a context it is lent: the caller keeps `context` alive until every handle made in it is destroyed. (The crate's
/// cudarc would destroy such a context with its last reference, so the handle keeps one reference for the life of the process: a small
/// allocation per call.)
///
/// # Safety
///
/// `context` is a valid `CUcontext`, and stays valid as said above; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_context_borrow(
    context: CUcontext,
    out: *mut *mut tilefish_context,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract.
        let out = unsafe { out_arg(out) }?;
        if context.is_null() {
            return Err(Failure::Invalid("the context is NULL".to_owned()));
        }
        require_driver()?;
        result::init().map_err(driver("cuInit"))?; // before anything else asks the driver (cuCtxGetCurrent too)
        let _keep = KeepCurrent::new()?;
        // SAFETY: the caller vouches for the context; it is current only for this call (`KeepCurrent` puts back what was).
        unsafe { result::ctx::set_current(context) }.map_err(driver("cuCtxSetCurrent"))?;
        let mut device: sys::CUdevice = 0;
        // SAFETY: a live out-pointer; the context is current.
        unsafe { sys::cuCtxGetDevice(&mut device) }.result().map_err(driver("cuCtxGetDevice"))?;
        let count = result::device::get_count().map_err(driver("cuDeviceGetCount"))?;
        let ordinal = (0..count)
            .find(|&i| result::device::get(i).is_ok_and(|d| d == device))
            .ok_or_else(|| Failure::Invalid("the device of the context is none of this machine's".to_owned()))?;
        let (mut flags, mut active) = (0u32, 0i32);
        // SAFETY: live out-pointers, a device of this machine. (Unlike retaining it, asking for its state never creates a primary context.)
        unsafe { sys::cuDevicePrimaryCtxGetState(device, &mut flags, &mut active) }
            .result()
            .map_err(driver("cuDevicePrimaryCtxGetState"))?;
        let primary = active != 0 && {
            // SAFETY: the primary context is active, so retaining it creates nothing; it is released right after.
            let primary = unsafe { result::primary_ctx::retain(device) }.map_err(driver("cuDevicePrimaryCtxRetain"))?;
            // SAFETY: the retain above.
            unsafe { result::primary_ctx::release(device) }.map_err(driver("cuDevicePrimaryCtxRelease"))?;
            primary == context
        };
        let ctx = if primary {
            CudaContext::new(ordinal as usize).map_err(driver("cuDevicePrimaryCtxRetain"))?
        } else {
            // SAFETY: a valid context of `device` (the caller's contract). cudarc destroys it with the last reference of the value it returns,
            // so one reference is kept for ever: tilefish never destroys a context it is lent.
            let ctx = unsafe { CudaContext::from_raw_context(ordinal as usize, device, context) }
                .map_err(driver("cuDeviceGetAttribute"))?;
            std::mem::forget(Arc::clone(&ctx));
            ctx
        };
        // SAFETY: as above.
        unsafe { put(out, tilefish_context { inner: Ctx::new(ctx)? }) }
    })
}

/// Destroy a context handle (NULL is a no-op). Handles made in it keep what they need of it.
///
/// # Safety
///
/// `context` is NULL or a live handle, used by no other thread meanwhile and not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_context_destroy(context: *mut tilefish_context) {
    // SAFETY: the caller's contract.
    quiet((), || unsafe { destroy_keeping_current(context) })
}

/// The `CUcontext` of a context handle (NULL for NULL).
///
/// # Safety
///
/// `context` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_context_handle(context: *const tilefish_context) -> CUcontext {
    // SAFETY: the caller's contract.
    quiet(std::ptr::null_mut(), || unsafe { context.as_ref() }.map_or(std::ptr::null_mut(), |c| c.inner.ctx.cu_ctx()))
}

/// The device ordinal of a context handle (-1 for NULL).
///
/// # Safety
///
/// `context` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_context_device(context: *const tilefish_context) -> i32 {
    // SAFETY: the caller's contract.
    quiet(-1, || unsafe { context.as_ref() }.map_or(-1, |c| c.inner.ctx.ordinal() as i32))
}

/// The architecture of the device of a context, as its compute capability times ten plus the minor (`103` for `sm_103`).
///
/// # Safety
///
/// `context` is a live handle; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_context_arch(context: *const tilefish_context, out: *mut u32) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract.
        let context = unsafe { handle(context, "the context") }?;
        let _keep = KeepCurrent::new()?;
        let arch = Arch::of_context(&context.inner.ctx)?;
        // SAFETY: as above.
        unsafe { put_value(out, arch.capability()) }
    })
}

/// Make the cuDNN handle of a context now, and keep it for as long as the returned handle lives (`tilefish::CudnnHandle`).
///
/// `cudnnCreate` waits for all the work queued on the device. A context has one cuDNN handle, made by the first convolution loaded in it and
/// shared by every later one; with this handle alive, no load in the context makes one. A host makes it at start-up, before it queues work.
/// `TILEFISH_ERROR_UNSUPPORTED` when cuDNN cannot be loaded.
///
/// # Safety
///
/// `context` is a live handle; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_cudnn_handle_create(
    context: *const tilefish_context,
    out: *mut *mut tilefish_cudnn_handle,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract.
        let out = unsafe { out_arg(out) }?;
        // SAFETY: as above.
        let context = unsafe { handle(context, "the context") }?;
        let _keep = KeepCurrent::new()?;
        let made = CudnnHandle::new(&context.inner.ctx)?;
        // SAFETY: as above.
        unsafe { put(out, tilefish_cudnn_handle { _handle: made, _ctx: Arc::clone(&context.inner) }) }
    })
}

/// Destroy a cuDNN handle (NULL is a no-op): the context's cuDNN handle lives on while a convolution loaded in it does.
///
/// # Safety
///
/// `handle` is NULL or a live handle, not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_cudnn_handle_destroy(handle: *mut tilefish_cudnn_handle) {
    // SAFETY: the caller's contract.
    quiet((), || unsafe { destroy_keeping_current(handle) })
}
