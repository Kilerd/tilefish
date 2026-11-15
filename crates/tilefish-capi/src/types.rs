//! The argument structs of the C ABI and their conversion into the crate's types.
//!
//! Every tensor is a *region* the caller vouches for (`tilefish::DevicePtr::from_raw`): its address, its length in bytes and its device. The
//! crate checks everything else before anything is enqueued (the tensor fits its region, and every precondition of the binary or plan).

use std::ffi::{CStr, c_char};

use tilefish::cudarc::driver::result;
use tilefish::cudarc::driver::sys::CUcontext;
use tilefish::{Access, DType, DevScalar, DevicePtr, TensorArg};

use crate::error::{Failure, Outcome, invalid};

// The shapes and strides of the C ABI are `uint64_t` arrays, which the crate reads as `usize` slices in place.
const _: () = assert!(std::mem::size_of::<usize>() == 8, "tilefish-capi is for 64-bit targets");

/// `bf16`, 2 bytes.
pub const TILEFISH_DTYPE_BF16: i32 = 0;
/// `f16`, 2 bytes.
pub const TILEFISH_DTYPE_F16: i32 = 1;
/// `f32`, 4 bytes.
pub const TILEFISH_DTYPE_F32: i32 = 2;
/// `i32`, 4 bytes.
pub const TILEFISH_DTYPE_I32: i32 = 3;
/// `i64`, 8 bytes.
pub const TILEFISH_DTYPE_I64: i32 = 4;
/// `u8`, 1 byte.
pub const TILEFISH_DTYPE_U8: i32 = 5;
/// `e4m3` (an fp8), 1 byte.
pub const TILEFISH_DTYPE_E4M3: i32 = 6;
/// `e5m2` (an fp8), 1 byte.
pub const TILEFISH_DTYPE_E5M2: i32 = 7;

/// The kernel only reads the tensor.
pub const TILEFISH_ACCESS_IN: i32 = 0;
/// The kernel only writes the tensor.
pub const TILEFISH_ACCESS_OUT: i32 = 1;
/// The kernel reads and writes the tensor.
pub const TILEFISH_ACCESS_INOUT: i32 = 2;

/// A device scalar given as an integer (`tilefish_devscalar.int_value`), for an integer dtype; it must fit the dtype.
pub const TILEFISH_DEVSCALAR_INT: i32 = 0;
/// A device scalar given as an f32 (`tilefish_devscalar.f32_value`), for dtype `f32`.
pub const TILEFISH_DEVSCALAR_F32: i32 = 1;
/// A device scalar given as the bits of one element of its dtype, in the low bytes (`tilefish_devscalar.bits`).
pub const TILEFISH_DEVSCALAR_BITS: i32 = 2;

/// A tensor: a region of device memory the caller vouches for, and the tensor in it.
///
/// The caller vouches that `address .. address + bytes` is allocated memory of device `device` (`cuMemAlloc`, `cudaMalloc`,
/// `cudaMallocAsync`, `cuMemMap`, a framework's allocator); that it stays allocated until every launch given the tensor has completed on its
/// stream (and, for a captured plan, until the captured plan is destroyed: a replay launches on the address again); and that the caller orders
/// it against its other uses of the memory (tilefish orders nothing it does not own). tilefish refuses a tensor whose shape, strides and
/// offset reach past `bytes`.
#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct tilefish_tensor {
    /// The start of the region (a `CUdeviceptr`).
    pub address: u64,
    /// The length of the region in bytes.
    pub bytes: u64,
    /// The device ordinal of the region.
    pub device: i32,
    /// The element type (`TILEFISH_DTYPE_*`).
    pub dtype: i32,
    /// How the kernel may use it (`TILEFISH_ACCESS_*`): a tensor given as `IN` for a parameter that is written is refused.
    pub access: i32,
    /// The number of dimensions (0 for a device scalar of shape `[]`).
    pub rank: u32,
    /// `rank` sizes (may be NULL when `rank` is 0).
    pub shape: *const u64,
    /// NULL for a dense row-major tensor at `address`; otherwise `rank` element strides: a view whose first element is element `offset`
    /// of the region.
    pub strides: *const u64,
    /// The element offset of a view's first element in the region (0 for a dense tensor).
    pub offset: u64,
}

/// A tensor with a name: an external, a weight or a tap target of a plan.
#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct tilefish_named_tensor {
    /// The name of the plan's buffer (a parameter, a returned value by its name or as `out0`, `out1`, ...).
    pub name: *const c_char,
    /// The caller's tensor.
    pub tensor: tilefish_tensor,
}

/// A debug tap of a plan launch or capture: the plan's buffer `buffer` copied into `into` right after step `after` of the bucket (or, when
/// `after` is negative, after the last step that writes it).
#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct tilefish_tap {
    /// The buffer of the plan (by name, or `out<i>`).
    pub buffer: *const c_char,
    /// The index of the step the copy follows, or a negative number for the last step that writes the buffer.
    pub after: i64,
    /// Where the copy goes: a dense, writable tensor of the buffer's dtype and shape at the bucket.
    pub into: tilefish_tensor,
}

/// The value of a device scalar of a plan, by name.
#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct tilefish_devscalar {
    /// The name of the device scalar.
    pub name: *const c_char,
    /// Which field holds the value (`TILEFISH_DEVSCALAR_*`).
    pub kind: i32,
    /// The value of `TILEFISH_DEVSCALAR_F32`.
    pub f32_value: f32,
    /// The value of `TILEFISH_DEVSCALAR_INT`.
    pub int_value: i64,
    /// The value of `TILEFISH_DEVSCALAR_BITS`.
    pub bits: u64,
}

/// A bucket value: the size a bucketed dimension is built for.
#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct tilefish_bucket {
    /// The dimension.
    pub dim: *const c_char,
    /// Its value.
    pub value: u64,
}

/// A value of a fixed argument of a binary that the call gives (a `given` parameter of the manifest).
#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct tilefish_given {
    /// The argument.
    pub name: *const c_char,
    /// Its value (an `i32` argument must fit 32 bits).
    pub value: i64,
}

/// A file of an embedded manifest: a cubin, by the file name the manifest gives it.
#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct tilefish_file {
    /// The file name.
    pub name: *const c_char,
    /// The bytes.
    pub bytes: *const u8,
    /// The number of bytes.
    pub len: usize,
}

/// A required string argument.
///
/// # Safety
///
/// `p` is NULL or a NUL-terminated string that outlives `'a`.
pub(crate) unsafe fn str_arg<'a>(p: *const c_char, what: &str) -> Outcome<&'a str> {
    if p.is_null() {
        return invalid(format!("{what} is NULL"));
    }
    // SAFETY: the caller's contract.
    let text = unsafe { CStr::from_ptr(p) };
    text.to_str().map_err(|_| Failure::Invalid(format!("{what} is not UTF-8")))
}

/// An array argument of `n` items (NULL is allowed when `n` is 0).
///
/// # Safety
///
/// `p` is NULL with `n == 0`, or points at `n` items that outlive `'a`.
pub(crate) unsafe fn slice_arg<'a, T>(p: *const T, n: usize, what: &str) -> Outcome<&'a [T]> {
    if n == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return invalid(format!("{what} is NULL but its count is {n}"));
    }
    // SAFETY: the caller's contract.
    Ok(unsafe { std::slice::from_raw_parts(p, n) })
}

/// A required handle argument.
///
/// # Safety
///
/// `p` is NULL or a live handle of type `T` that outlives `'a`.
pub(crate) unsafe fn handle<'a, T>(p: *const T, what: &str) -> Outcome<&'a T> {
    // SAFETY: the caller's contract.
    unsafe { p.as_ref() }.ok_or_else(|| Failure::Invalid(format!("{what} is NULL")))
}

/// A required mutable handle argument.
///
/// # Safety
///
/// `p` is NULL or a live handle of type `T` that outlives `'a`, used by no other thread meanwhile.
pub(crate) unsafe fn handle_mut<'a, T>(p: *mut T, what: &str) -> Outcome<&'a mut T> {
    // SAFETY: the caller's contract.
    unsafe { p.as_mut() }.ok_or_else(|| Failure::Invalid(format!("{what} is NULL")))
}

/// The out-parameter of a handle: checked not NULL and set to NULL, so that a failure leaves NULL there.
///
/// # Safety
///
/// `out` is NULL or a writable pointer.
pub(crate) unsafe fn out_arg<T>(out: *mut *mut T) -> Outcome<*mut *mut T> {
    if out.is_null() {
        return invalid("the out-parameter is NULL");
    }
    // SAFETY: the caller's contract, and `out` is not NULL.
    unsafe { out.write(std::ptr::null_mut()) };
    Ok(out)
}

/// Store a new handle in an out-parameter that `out_arg` checked.
///
/// # Safety
///
/// `out` is what `out_arg` returned.
pub(crate) unsafe fn put<T>(out: *mut *mut T, value: T) -> Outcome {
    // SAFETY: the caller's contract.
    unsafe { out.write(Box::into_raw(Box::new(value))) };
    Ok(())
}

/// A scalar out-parameter.
///
/// # Safety
///
/// `out` is NULL or a writable pointer.
pub(crate) unsafe fn put_value<T>(out: *mut T, value: T) -> Outcome {
    if out.is_null() {
        return invalid("the out-parameter is NULL");
    }
    // SAFETY: the caller's contract, and `out` is not NULL.
    unsafe { out.write(value) };
    Ok(())
}

/// Destroy a handle made by `put` (NULL is a no-op).
///
/// # Safety
///
/// `p` is NULL or a handle of type `T` that `put` made and nothing destroyed yet, used by no other thread.
pub(crate) unsafe fn destroy<T>(p: *mut T) {
    if !p.is_null() {
        // SAFETY: the caller's contract: the box `put` made.
        drop(unsafe { Box::from_raw(p) });
    }
}

pub(crate) fn dtype_of(code: i32) -> Outcome<DType> {
    Ok(match code {
        TILEFISH_DTYPE_BF16 => DType::Bf16,
        TILEFISH_DTYPE_F16 => DType::F16,
        TILEFISH_DTYPE_F32 => DType::F32,
        TILEFISH_DTYPE_I32 => DType::I32,
        TILEFISH_DTYPE_I64 => DType::I64,
        TILEFISH_DTYPE_U8 => DType::U8,
        TILEFISH_DTYPE_E4M3 => DType::E4m3,
        TILEFISH_DTYPE_E5M2 => DType::E5m2,
        other => return invalid(format!("{other} is not a TILEFISH_DTYPE_* value")),
    })
}

pub(crate) fn dtype_code(dtype: DType) -> i32 {
    match dtype {
        DType::Bf16 => TILEFISH_DTYPE_BF16,
        DType::F16 => TILEFISH_DTYPE_F16,
        DType::F32 => TILEFISH_DTYPE_F32,
        DType::I32 => TILEFISH_DTYPE_I32,
        DType::I64 => TILEFISH_DTYPE_I64,
        DType::U8 => TILEFISH_DTYPE_U8,
        DType::E4m3 => TILEFISH_DTYPE_E4M3,
        DType::E5m2 => TILEFISH_DTYPE_E5M2,
    }
}

fn access_of(code: i32) -> Outcome<Access> {
    Ok(match code {
        TILEFISH_ACCESS_IN => Access::In,
        TILEFISH_ACCESS_OUT => Access::Out,
        TILEFISH_ACCESS_INOUT => Access::InOut,
        other => return invalid(format!("{other} is not a TILEFISH_ACCESS_* value")),
    })
}

/// The shape and strides of a tensor, read in place from the caller's arrays.
///
/// # Safety
///
/// As for [`tensor_arg`].
unsafe fn dims_of<'a>(t: &tilefish_tensor, what: &str) -> Outcome<(&'a [usize], Option<&'a [usize]>)> {
    let rank = t.rank as usize;
    // SAFETY: the caller's contract: `shape` has `rank` items; a u64 is a usize here (64-bit targets, asserted above).
    let shape = unsafe { slice_arg(t.shape.cast::<usize>(), rank, &format!("the shape of {what}")) }?;
    let strides = if t.strides.is_null() {
        None
    } else if rank == 0 {
        Some(&[][..])
    } else {
        // SAFETY: as for the shape.
        Some(unsafe { std::slice::from_raw_parts(t.strides.cast::<usize>(), rank) })
    };
    Ok((shape, strides))
}

/// The crate's tensor of a C tensor, over the caller's shape and strides.
///
/// # Safety
///
/// The region is what `tilefish_tensor` says the caller vouches for, and `shape` (and `strides`, when not NULL) point at `rank` items that
/// outlive `'a`.
pub(crate) unsafe fn tensor_arg<'a>(t: &tilefish_tensor, what: &str) -> Outcome<TensorArg<'a>> {
    // SAFETY: the caller's contract.
    let (shape, strides) = unsafe { dims_of(t, what) }?;
    // SAFETY: as above.
    unsafe { tensor_over(t, shape, strides, what) }
}

/// The crate's tensor of a C tensor over the given shape and strides (the caller's, or copies a handle owns).
///
/// # Safety
///
/// The region is what `tilefish_tensor` says the caller vouches for.
pub(crate) unsafe fn tensor_over<'a>(
    t: &tilefish_tensor,
    shape: &'a [usize],
    strides: Option<&'a [usize]>,
    what: &str,
) -> Outcome<TensorArg<'a>> {
    let dtype = dtype_of(t.dtype)
        .map_err(|_| Failure::Invalid(format!("{what}: {} is not a TILEFISH_DTYPE_* value", t.dtype)))?;
    let access = access_of(t.access)
        .map_err(|_| Failure::Invalid(format!("{what}: {} is not a TILEFISH_ACCESS_* value", t.access)))?;
    // SAFETY: the caller vouches for the region (the contract of `tilefish_tensor`), for as long as the launches that use it.
    let region = unsafe { DevicePtr::from_raw(t.address, t.bytes, t.device) };
    let offset = usize::try_from(t.offset).map_err(|_| Failure::Invalid(format!("{what}: the offset does not fit")))?;
    Ok(match strides {
        Some(strides) => TensorArg::new_strided(region, shape, strides, offset, dtype, access)?,
        None if offset != 0 => {
            return invalid(format!(
                "{what}: a dense tensor (strides NULL) has offset 0: give the address of its first element, or strides for a view"
            ));
        }
        None => TensorArg::new(region, shape, dtype, access)?,
    })
}

/// A tensor whose shape and strides a handle owns (a captured plan's externals and taps, a weighted run's weights): the crate keeps the tensor
/// for as long as the handle lives.
pub(crate) struct OwnedDims {
    shape: Box<[usize]>,
    strides: Option<Box<[usize]>>,
}

impl OwnedDims {
    /// Copies of the shape and strides of `t`.
    ///
    /// # Safety
    ///
    /// As for [`tensor_arg`] (only for the duration of the call).
    pub(crate) unsafe fn of(t: &tilefish_tensor, what: &str) -> Outcome<Self> {
        // SAFETY: the caller's contract.
        let (shape, strides) = unsafe { dims_of(t, what) }?;
        Ok(Self { shape: shape.into(), strides: strides.map(Into::into) })
    }

    /// The crate's tensor over these copies, for any lifetime.
    ///
    /// # Safety
    ///
    /// The region is what `tilefish_tensor` says the caller vouches for, and `self` outlives every use of the tensor (its boxes do not move
    /// when `self` does, and are never written).
    pub(crate) unsafe fn tensor<'a>(&self, t: &tilefish_tensor, what: &str) -> Outcome<TensorArg<'a>> {
        // SAFETY: the boxes live as long as `self`, which the caller keeps for as long as the tensor is used, and are never written.
        let shape: &'a [usize] = unsafe { &*std::ptr::from_ref::<[usize]>(&self.shape) };
        let strides: Option<&'a [usize]> =
            self.strides.as_deref().map(|s| unsafe { &*std::ptr::from_ref::<[usize]>(s) });
        // SAFETY: the caller's contract.
        unsafe { tensor_over(t, shape, strides, what) }
    }
}

/// The bucket values of a load.
///
/// # Safety
///
/// `p` points at `n` buckets whose strings outlive `'a` (NULL when `n` is 0).
pub(crate) unsafe fn buckets_arg<'a>(p: *const tilefish_bucket, n: usize) -> Outcome<Vec<(&'a str, u64)>> {
    // SAFETY: the caller's contract.
    let items = unsafe { slice_arg(p, n, "buckets") }?;
    // SAFETY: as above.
    items.iter().map(|b| Ok((unsafe { str_arg(b.dim, "the dimension of a bucket") }?, b.value))).collect()
}

/// The values given per call.
///
/// # Safety
///
/// `p` points at `n` items whose strings outlive `'a` (NULL when `n` is 0).
pub(crate) unsafe fn given_arg<'a>(p: *const tilefish_given, n: usize) -> Outcome<Vec<(&'a str, i64)>> {
    // SAFETY: the caller's contract.
    let items = unsafe { slice_arg(p, n, "given") }?;
    // SAFETY: as above.
    items.iter().map(|g| Ok((unsafe { str_arg(g.name, "the name of a given value") }?, g.value))).collect()
}

/// The value of a device scalar.
pub(crate) fn devscalar_of(d: &tilefish_devscalar, name: &str) -> Outcome<DevScalar> {
    Ok(match d.kind {
        TILEFISH_DEVSCALAR_INT => DevScalar::Int(d.int_value),
        TILEFISH_DEVSCALAR_F32 => DevScalar::F32(d.f32_value),
        TILEFISH_DEVSCALAR_BITS => DevScalar::Bits(d.bits),
        other => return invalid(format!("the device scalar '{name}': {other} is not a TILEFISH_DEVSCALAR_* value")),
    })
}

/// The device scalars of a call.
///
/// # Safety
///
/// `p` points at `n` items whose strings outlive `'a` (NULL when `n` is 0).
pub(crate) unsafe fn devscalars_arg<'a>(p: *const tilefish_devscalar, n: usize) -> Outcome<Vec<(&'a str, DevScalar)>> {
    // SAFETY: the caller's contract.
    let items = unsafe { slice_arg(p, n, "devscalars") }?;
    items
        .iter()
        .map(|d| {
            // SAFETY: as above.
            let name = unsafe { str_arg(d.name, "the name of a device scalar") }?;
            Ok((name, devscalar_of(d, name)?))
        })
        .collect()
}

/// The named tensors of a call, over the caller's arrays.
///
/// # Safety
///
/// `p` points at `n` items whose strings, shapes and strides outlive `'a` (NULL when `n` is 0), and each region is what `tilefish_tensor`
/// says the caller vouches for.
pub(crate) unsafe fn named_arg<'a>(
    p: *const tilefish_named_tensor,
    n: usize,
    what: &str,
) -> Outcome<Vec<(&'a str, TensorArg<'a>)>> {
    // SAFETY: the caller's contract.
    let items = unsafe { slice_arg(p, n, what) }?;
    items
        .iter()
        .map(|t| {
            // SAFETY: as above.
            let name = unsafe { str_arg(t.name, &format!("the name of one of the {what}")) }?;
            // SAFETY: as above.
            Ok((name, unsafe { tensor_arg(&t.tensor, &format!("'{name}'")) }?))
        })
        .collect()
}

/// Named tensors whose shapes a handle owns, with their names.
pub(crate) struct OwnedNamed {
    pub(crate) items: Vec<(String, tilefish_tensor, OwnedDims)>,
}

impl OwnedNamed {
    /// Copies of the names, shapes and strides of `n` named tensors.
    ///
    /// # Safety
    ///
    /// `p` points at `n` items (NULL when `n` is 0) that are valid for the call.
    pub(crate) unsafe fn of(p: *const tilefish_named_tensor, n: usize, what: &str) -> Outcome<Self> {
        // SAFETY: the caller's contract.
        let items = unsafe { slice_arg(p, n, what) }?;
        let items = items
            .iter()
            .map(|t| {
                // SAFETY: as above.
                let name = unsafe { str_arg(t.name, &format!("the name of one of the {what}")) }?.to_owned();
                // SAFETY: as above.
                let dims = unsafe { OwnedDims::of(&t.tensor, &format!("'{name}'")) }?;
                Ok((name, t.tensor, dims))
            })
            .collect::<Outcome<Vec<_>>>()?;
        Ok(Self { items })
    }

    /// The crate's tensors over the copies, with their names (borrowed from `self`: the crate copies them).
    ///
    /// # Safety
    ///
    /// As for [`OwnedDims::tensor`]: `self` outlives every use of the tensors.
    pub(crate) unsafe fn tensors_static(&self) -> Outcome<Vec<(&str, TensorArg<'static>)>> {
        self.items
            .iter()
            // SAFETY: the caller's contract.
            .map(|(name, t, dims)| {
                let tensor: TensorArg<'static> = unsafe { dims.tensor(t, &format!("'{name}'")) }?;
                Ok((name.as_str(), tensor))
            })
            .collect()
    }
}

/// The context that was current on this thread when a call began, put back when it ends (also on an error or a panic): a call of the C ABI
/// leaves the thread's current context as it found it.
pub(crate) struct KeepCurrent(Option<CUcontext>);

impl KeepCurrent {
    /// Remember the current context. Only for calls that have a context, which means the driver is loaded.
    pub(crate) fn new() -> Outcome<Self> {
        let current = result::ctx::get_current()
            .map_err(|source| Failure::Tilefish(tilefish::Error::Driver { call: "cuCtxGetCurrent", source }))?;
        Ok(Self(current))
    }
}

impl Drop for KeepCurrent {
    fn drop(&mut self) {
        if let Ok(now) = result::ctx::get_current()
            && now != self.0
        {
            // SAFETY: the context that was current when the call began (or none), which the thread had then.
            let _ = unsafe { result::ctx::set_current(self.0.unwrap_or(std::ptr::null_mut())) };
        }
    }
}
