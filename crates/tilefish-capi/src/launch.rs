//! Single launches: a kernel (Triton, Gluon or prebuilt), a pinned cuBLASLt GEMM, a pinned cuDNN convolution.
//!
//! Each launch is the crate's `launch_raw` over regions: every precondition is checked before anything is enqueued, the binary's context is
//! current for the call, and the caller's stream is used as it is. On a stream the caller is capturing, the launch is captured like any kernel
//! node, and the graph is the caller's.

use std::ffi::{CString, c_char};
use std::sync::Arc;

use tilefish::cudarc::driver::sys::CUstream;
use tilefish::{Arch, Conv, Gemm, GemmEpilogue, Kernel, TensorArg};

use crate::context::{Ctx, destroy_keeping_current, tilefish_context};
use crate::error::{Outcome, c_string, call, quiet, tilefish_status};
use crate::manifest::tilefish_manifest;
use crate::types::{
    KeepCurrent, buckets_arg, given_arg, handle, out_arg, put, put_value, slice_arg, str_arg, tensor_arg,
    tilefish_bucket, tilefish_given, tilefish_tensor,
};

/// A kernel loaded for the device of a context: a Triton or Gluon kernel or a prebuilt binary of the manifest.
#[allow(non_camel_case_types)]
pub struct tilefish_kernel {
    kernel: Kernel,
    stem: CString,
    names: Vec<CString>,
    _ctx: Arc<Ctx>,
}

/// A pinned cuBLASLt GEMM loaded in a context.
#[allow(non_camel_case_types)]
pub struct tilefish_gemm {
    gemm: Gemm,
    stem: CString,
    _ctx: Arc<Ctx>,
}

/// A pinned cuDNN forward convolution loaded in a context (its canary checked at load).
#[allow(non_camel_case_types)]
pub struct tilefish_conv {
    conv: Conv,
    stem: CString,
    _ctx: Arc<Ctx>,
}

/// The architecture a load asks for: NULL for the context's.
///
/// # Safety
///
/// `arch` is NULL or a NUL-terminated string.
unsafe fn arch_arg(arch: *const c_char, ctx: &Ctx) -> Outcome<Arch> {
    if arch.is_null() {
        return Ok(Arch::of_context(&ctx.ctx)?);
    }
    // SAFETY: the caller's contract.
    Ok(Arch::parse(unsafe { str_arg(arch, "the architecture") }?)?)
}

/// The tensors of a launch, over the caller's arrays.
///
/// # Safety
///
/// `p` points at `n` tensors valid for the call (NULL when `n` is 0), each a region the caller vouches for.
unsafe fn tensors_arg<'a>(p: *const tilefish_tensor, n: usize) -> Outcome<Vec<TensorArg<'a>>> {
    // SAFETY: the caller's contract.
    let items = unsafe { slice_arg(p, n, "tensors") }?;
    // SAFETY: as above.
    items.iter().enumerate().map(|(i, t)| unsafe { tensor_arg(t, &format!("tensor {i}")) }).collect()
}

/// The stem of a kernel in the manifest, owned by the handle (NULL for NULL).
///
/// # Safety
///
/// The handle is NULL or live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_kernel_stem(h: *const tilefish_kernel) -> *const c_char {
    // SAFETY: the caller's contract.
    quiet(std::ptr::null(), || unsafe { h.as_ref() }.map_or(std::ptr::null(), |h| h.stem.as_ptr()))
}

/// Destroy a kernel handle (NULL is a no-op), after the launches that used it are enqueued (the driver keeps what they need).
///
/// # Safety
///
/// The handle is NULL or live, and not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_kernel_destroy(h: *mut tilefish_kernel) {
    // SAFETY: the caller's contract.
    quiet((), || unsafe { destroy_keeping_current(h) })
}

/// The stem of a GEMM in the manifest, owned by the handle (NULL for NULL).
///
/// # Safety
///
/// The handle is NULL or live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_gemm_stem(h: *const tilefish_gemm) -> *const c_char {
    // SAFETY: the caller's contract.
    quiet(std::ptr::null(), || unsafe { h.as_ref() }.map_or(std::ptr::null(), |h| h.stem.as_ptr()))
}

/// Destroy a GEMM handle (NULL is a no-op), after the launches that used it are enqueued (the driver keeps what they need).
///
/// # Safety
///
/// The handle is NULL or live, and not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_gemm_destroy(h: *mut tilefish_gemm) {
    // SAFETY: the caller's contract.
    quiet((), || unsafe { destroy_keeping_current(h) })
}

/// The stem of a convolution in the manifest, owned by the handle (NULL for NULL).
///
/// # Safety
///
/// The handle is NULL or live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_conv_stem(h: *const tilefish_conv) -> *const c_char {
    // SAFETY: the caller's contract.
    quiet(std::ptr::null(), || unsafe { h.as_ref() }.map_or(std::ptr::null(), |h| h.stem.as_ptr()))
}

/// Destroy a convolution handle (NULL is a no-op), after the launches that used it are enqueued (the driver keeps what they need).
///
/// # Safety
///
/// The handle is NULL or live, and not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_conv_destroy(h: *mut tilefish_conv) {
    // SAFETY: the caller's contract.
    quiet((), || unsafe { destroy_keeping_current(h) })
}

/// Load the binary `stem` (with its bucket values, `n_buckets` of them, for a bucketed kernel) of `manifest` for `arch` (`"sm_103"`, or NULL for
/// the architecture of the context's device) in `context`. The function's parameters are checked against the manifest's ABI.
///
/// # Safety
///
/// `manifest` and `context` are live handles; `stem` (and `arch` when not NULL) are NUL-terminated strings; `buckets` points at `n_buckets`
/// buckets (NULL when 0); `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_kernel_load(
    manifest: *const tilefish_manifest,
    context: *const tilefish_context,
    stem: *const c_char,
    buckets: *const tilefish_bucket,
    n_buckets: usize,
    arch: *const c_char,
    out: *mut *mut tilefish_kernel,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let manifest = unsafe { handle(manifest, "the manifest") }?;
        let context = unsafe { handle(context, "the context") }?;
        let stem = unsafe { str_arg(stem, "the stem") }?;
        let buckets = unsafe { buckets_arg(buckets, n_buckets) }?;
        let _keep = KeepCurrent::new()?;
        let arch = unsafe { arch_arg(arch, &context.inner) }?;
        let kernel = manifest.manifest.kernel(&context.inner.ctx, stem, &buckets, arch)?;
        let names = kernel.plan().tensor_names().map(c_string).collect();
        let stem = c_string(kernel.stem());
        unsafe { put(out, tilefish_kernel { kernel, stem, names, _ctx: Arc::clone(&context.inner) }) }
    })
}

/// Check a call of a kernel as `tilefish_kernel_launch` does, and launch nothing. `given` are the values of the binary's fixed arguments that
/// a call gives (`n_given` of them; none for most kernels).
///
/// # Safety
///
/// `kernel` is a live handle; `tensors` points at `n_tensors` tensors (and `given` at `n_given` values), valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_kernel_check(
    kernel: *const tilefish_kernel,
    tensors: *const tilefish_tensor,
    n_tensors: usize,
    given: *const tilefish_given,
    n_given: usize,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let kernel = unsafe { handle(kernel, "the kernel") }?;
        let args = unsafe { tensors_arg(tensors, n_tensors) }?;
        let given = unsafe { given_arg(given, n_given) }?;
        kernel.kernel.check_with(&args, &given)?;
        Ok(())
    })
}

/// Check the call and launch the kernel on `stream` (a stream of the kernel's context, or NULL for the legacy default stream). The tensors are
/// the kernel's tensor parameters in the order of its ABI (`tilefish_kernel_tensor_name`), a kernel with a global scratch taking it last.
/// Nothing is launched when a check fails. An error from the driver does not promise that nothing was launched.
///
/// # Safety
///
/// `kernel` is a live handle; `stream` is a valid stream of its context (or NULL) for the call; `tensors` points at `n_tensors` tensors (and
/// `given` at `n_given` values), valid for the call, each tensor's region as `tilefish_tensor` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_kernel_launch(
    kernel: *const tilefish_kernel,
    stream: CUstream,
    tensors: *const tilefish_tensor,
    n_tensors: usize,
    given: *const tilefish_given,
    n_given: usize,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let kernel = unsafe { handle(kernel, "the kernel") }?;
        let args = unsafe { tensors_arg(tensors, n_tensors) }?;
        let given = unsafe { given_arg(given, n_given) }?;
        let _keep = KeepCurrent::new()?;
        Ok(unsafe { kernel.kernel.launch_raw_with(stream, &args, &given) }?)
    })
}

/// The number of tensors a launch of the kernel takes.
///
/// # Safety
///
/// `kernel` is NULL (0) or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_kernel_tensor_count(kernel: *const tilefish_kernel) -> usize {
    // SAFETY: the caller's contract.
    quiet(0, || unsafe { kernel.as_ref() }.map_or(0, |k| k.names.len()))
}

/// The name of tensor `index` of a launch of the kernel (`"global_scratch"` for the scratch of a kernel that has one), owned by the handle, or
/// NULL past the last.
///
/// # Safety
///
/// `kernel` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_kernel_tensor_name(kernel: *const tilefish_kernel, index: usize) -> *const c_char {
    // SAFETY: the caller's contract.
    quiet(std::ptr::null(), || {
        unsafe { kernel.as_ref() }.and_then(|k| k.names.get(index)).map_or(std::ptr::null(), |n| n.as_ptr())
    })
}

/// Load the pinned GEMM `stem` of `manifest` in `context`: the cuBLASLt of the process must be the version that chose its algorithm, the
/// algorithm is vetted and approved by the library for the problem (with its epilogue), and the entry's canary, when it has one (a
/// `cublaslt.matmul`), runs and must give its bytes.
///
/// # Safety
///
/// As for `tilefish_kernel_load`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_gemm_load(
    manifest: *const tilefish_manifest,
    context: *const tilefish_context,
    stem: *const c_char,
    buckets: *const tilefish_bucket,
    n_buckets: usize,
    arch: *const c_char,
    out: *mut *mut tilefish_gemm,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let manifest = unsafe { handle(manifest, "the manifest") }?;
        let context = unsafe { handle(context, "the context") }?;
        let stem = unsafe { str_arg(stem, "the stem") }?;
        let buckets = unsafe { buckets_arg(buckets, n_buckets) }?;
        let _keep = KeepCurrent::new()?;
        let arch = unsafe { arch_arg(arch, &context.inner) }?;
        let gemm = manifest.manifest.gemm(&context.inner.ctx, stem, &buckets, arch)?;
        let stem = c_string(gemm.stem());
        unsafe { put(out, tilefish_gemm { gemm, stem, _ctx: Arc::clone(&context.inner) }) }
    })
}

/// Check a GEMM call as `tilefish_gemm_launch` does, and launch nothing.
///
/// # Safety
///
/// `gemm` is a live handle; `tensors` points at `n_tensors` tensors valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_gemm_check(
    gemm: *const tilefish_gemm,
    tensors: *const tilefish_tensor,
    n_tensors: usize,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let gemm = unsafe { handle(gemm, "the GEMM") }?;
        let args = unsafe { tensors_arg(tensors, n_tensors) }?;
        Ok(gemm.gemm.check(&args)?)
    })
}

/// Check the call and run the pinned GEMM on `stream`: the tensors are `a`, `b`, `bias` when the GEMM has one, `c` when it has one (a
/// `cublaslt.matmul` with `beta` 1: `tilefish_gemm_epilogue`), `out`, and the caller's workspace, a `u8` tensor of at least
/// `tilefish_gemm_workspace_required` bytes.
///
/// # Safety
///
/// As for `tilefish_kernel_launch`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_gemm_launch(
    gemm: *const tilefish_gemm,
    stream: CUstream,
    tensors: *const tilefish_tensor,
    n_tensors: usize,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let gemm = unsafe { handle(gemm, "the GEMM") }?;
        let args = unsafe { tensors_arg(tensors, n_tensors) }?;
        let _keep = KeepCurrent::new()?;
        Ok(unsafe { gemm.gemm.launch_raw(stream, &args) }?)
    })
}

/// The workspace bytes the GEMM was pinned with (the manifest's `workspace_bytes`; 0 for NULL).
///
/// # Safety
///
/// `gemm` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_gemm_workspace_bytes(gemm: *const tilefish_gemm) -> u64 {
    // SAFETY: the caller's contract.
    quiet(0, || unsafe { gemm.as_ref() }.map_or(0, |g| g.gemm.workspace_bytes()))
}

/// The bytes of workspace a launch needs at least (0 for NULL).
///
/// # Safety
///
/// `gemm` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_gemm_workspace_required(gemm: *const tilefish_gemm) -> u64 {
    // SAFETY: the caller's contract.
    quiet(0, || unsafe { gemm.as_ref() }.map_or(0, |g| g.gemm.workspace_required()))
}

/// The sizes `(M, N, K)` of a GEMM, and whether it has a bias (any out-pointer may be NULL).
///
/// # Safety
///
/// `gemm` is a live handle; each out-pointer is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_gemm_sizes(
    gemm: *const tilefish_gemm,
    m: *mut u64,
    n: *mut u64,
    k: *mut u64,
    has_bias: *mut bool,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let gemm = unsafe { handle(gemm, "the GEMM") }?;
        let (sm, sn, sk) = gemm.gemm.sizes();
        for (p, v) in [(m, sm), (n, sn), (k, sk)] {
            if !p.is_null() {
                unsafe { put_value(p, v) }?;
            }
        }
        if !has_bias.is_null() {
            unsafe { put_value(has_bias, gemm.gemm.has_bias()) }?;
        }
        Ok(())
    })
}

/// No epilogue: `out = acc` (`cublaslt.gemm`, and a `cublaslt.matmul` of `epilogue: "none"`). ABI 1.1.
pub const TILEFISH_EPILOGUE_NONE: i32 = 0;
/// The bias epilogue: `out = acc + bias` (`cublaslt.gemm_bias`, and `"bias"`). ABI 1.1.
pub const TILEFISH_EPILOGUE_BIAS: i32 = 1;
/// The ReLU epilogue: `out = relu(acc)` (`"relu"`). ABI 1.1.
pub const TILEFISH_EPILOGUE_RELU: i32 = 2;
/// The ReLU of the bias epilogue: `out = relu(acc + bias)` (`"relu_bias"`). ABI 1.1.
pub const TILEFISH_EPILOGUE_RELU_BIAS: i32 = 3;

/// The epilogue of a GEMM, one of `TILEFISH_EPILOGUE_*` (tilefish's values, not cuBLASLt's `cublasLtEpilogue_t`): what cuBLASLt does to the f32
/// accumulator before the one rounding, and whether a launch takes `c` (a `cublaslt.matmul` with `beta` 1: the epilogue acts on `acc + c`, and
/// `c`, of the shape and type of `out`, is the tensor before `out`). Either out-pointer may be NULL. ABI 1.1.
///
/// # Safety
///
/// `gemm` is a live handle; each out-pointer is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_gemm_epilogue(
    gemm: *const tilefish_gemm,
    epilogue: *mut i32,
    has_c: *mut bool,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let gemm = unsafe { handle(gemm, "the GEMM") }?;
        if !epilogue.is_null() {
            let value = match gemm.gemm.epilogue() {
                GemmEpilogue::None => TILEFISH_EPILOGUE_NONE,
                GemmEpilogue::Bias => TILEFISH_EPILOGUE_BIAS,
                GemmEpilogue::Relu => TILEFISH_EPILOGUE_RELU,
                GemmEpilogue::ReluBias => TILEFISH_EPILOGUE_RELU_BIAS,
            };
            unsafe { put_value(epilogue, value) }?;
        }
        if !has_c.is_null() {
            unsafe { put_value(has_c, gemm.gemm.has_c()) }?;
        }
        Ok(())
    })
}

/// Load the pinned convolution `stem` of `manifest` in `context`: the cuDNN of the process must be the version that chose its pin, the pin is
/// vetted, and the entry's canary runs and must give its bytes. The context's cuDNN handle is shared by its convolutions.
///
/// # Safety
///
/// As for `tilefish_kernel_load`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_conv_load(
    manifest: *const tilefish_manifest,
    context: *const tilefish_context,
    stem: *const c_char,
    buckets: *const tilefish_bucket,
    n_buckets: usize,
    arch: *const c_char,
    out: *mut *mut tilefish_conv,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let manifest = unsafe { handle(manifest, "the manifest") }?;
        let context = unsafe { handle(context, "the context") }?;
        let stem = unsafe { str_arg(stem, "the stem") }?;
        let buckets = unsafe { buckets_arg(buckets, n_buckets) }?;
        let _keep = KeepCurrent::new()?;
        let arch = unsafe { arch_arg(arch, &context.inner) }?;
        let conv = manifest.manifest.conv(&context.inner.ctx, stem, &buckets, arch)?;
        let stem = c_string(conv.stem());
        unsafe { put(out, tilefish_conv { conv, stem, _ctx: Arc::clone(&context.inner) }) }
    })
}

/// Check a convolution call as `tilefish_conv_launch` does, and launch nothing.
///
/// # Safety
///
/// `conv` is a live handle; `tensors` points at `n_tensors` tensors valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_conv_check(
    conv: *const tilefish_conv,
    tensors: *const tilefish_tensor,
    n_tensors: usize,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let conv = unsafe { handle(conv, "the convolution") }?;
        let args = unsafe { tensors_arg(tensors, n_tensors) }?;
        Ok(conv.conv.check(&args)?)
    })
}

/// Check the call and run the pinned convolution on `stream`: the tensors are `x`, `w`, `y` and the caller's workspace, a `u8` tensor of at
/// least `tilefish_conv_workspace_required` bytes.
///
/// # Safety
///
/// As for `tilefish_kernel_launch`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_conv_launch(
    conv: *const tilefish_conv,
    stream: CUstream,
    tensors: *const tilefish_tensor,
    n_tensors: usize,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let conv = unsafe { handle(conv, "the convolution") }?;
        let args = unsafe { tensors_arg(tensors, n_tensors) }?;
        let _keep = KeepCurrent::new()?;
        Ok(unsafe { conv.conv.launch_raw(stream, &args) }?)
    })
}

/// The workspace bytes the convolution was pinned with (0 for NULL).
///
/// # Safety
///
/// `conv` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_conv_workspace_bytes(conv: *const tilefish_conv) -> u64 {
    // SAFETY: the caller's contract.
    quiet(0, || unsafe { conv.as_ref() }.map_or(0, |c| c.conv.workspace_bytes()))
}

/// The bytes of workspace a launch needs at least (0 for NULL).
///
/// # Safety
///
/// `conv` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_conv_workspace_required(conv: *const tilefish_conv) -> u64 {
    // SAFETY: the caller's contract.
    quiet(0, || unsafe { conv.as_ref() }.map_or(0, |c| c.conv.workspace_required()))
}

/// Run the convolution's canary again (as its load did) and refuse other bytes.
///
/// # Safety
///
/// `conv` is a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_conv_check_canary(conv: *const tilefish_conv) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract.
        let conv = unsafe { handle(conv, "the convolution") }?;
        let _keep = KeepCurrent::new()?;
        Ok(conv.conv.check_canary()?)
    })
}
