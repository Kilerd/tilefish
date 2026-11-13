//! The CUDA driver API, the only place in the crate with `unsafe` code.
//!
//! What a host needs from the driver is small, and it is exactly what Python's `_cuda.py` does through `cuda-python`:
//! `cuModuleLoadData` on the verified cubin bytes, `cuModuleGetFunction` on the manifest's symbol,
//! `cuFuncGetAttribute` / `cuFuncSetAttribute` for the shared memory, `cuFuncGetParamInfo` to check the loaded
//! function against the manifest's ABI, `cuDeviceGetAttribute` for the limits of the device, and `cuLaunchKernel` with
//! `kernelParams` laid out as the ABI says. The bindings are `cudarc::driver::sys`, loaded at run time.
//!
//! The obligations a host takes on are in two places, and nowhere else: [`DevicePtr::from_raw`] (the host vouches for a region of
//! device memory: where it is, how many bytes, on which device, and that it outlives the launches that use it) and
//! [`Kernel::launch_raw`] (the host vouches for a raw `CUstream` handle). Everything built from them is checked: a tensor is
//! refused when its shape needs more bytes than its pointer vouches for, and the safe [`Kernel::launch`] takes cudarc's
//! `CudaStream`, which owns its handle. The other functions here are safe to call: a launch is made of values that the
//! callers have already checked.

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::OnceLock;

use cudarc::driver::sys;
use cudarc::driver::{CudaStream, DevicePtr as CudarcDevicePtr, DevicePtrMut, SyncOnDrop};

use crate::error::{Error, Result};
use crate::kernel::Kernel;
use crate::tensor::TensorArg;

#[cfg(target_endian = "big")]
compile_error!("tilefish packs a 32-bit kernel argument into the low bytes of a 64-bit slot: little-endian hosts only");

/// A region of device memory: its address, how many bytes of it the holder vouches for, and the device it is on.
///
/// A tensor is made from one (`TensorRef::new`, `TensorMut::new`, [`TensorArg::new`]) and is refused when its shape needs more bytes
/// than the region has, so no launch can be made to read or write past what the host vouched for. The host vouches by calling
/// [`DevicePtr::from_raw`], which is `unsafe`; [`DevicePtr::offset`] and [`DevicePtr::slice`] make smaller regions of a region, checked.
/// A region from a cudarc `CudaSlice` or `CudaView` needs no `unsafe`: its length is known (`TensorRef::from_cuda`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePtr {
    addr: u64,
    len: u64,
    device: i32,
}

impl DevicePtr {
    /// The region of `len_bytes` bytes at `addr` (a `CUdeviceptr`) in the memory of the device with ordinal `device_ordinal`.
    ///
    /// # Safety
    ///
    /// The caller vouches that
    ///
    /// * `addr .. addr + len_bytes` is allocated device memory of the device `device_ordinal` (not host memory, not a freed or
    ///   not yet allocated address), writable if a [`TensorMut`](crate::TensorMut) is made of it;
    /// * it stays allocated, and is not reused, until every launch that was given a tensor made of it has completed on the device
    ///   (a launch is asynchronous: its stream orders it, it is not finished when `launch` returns);
    /// * nothing else writes it while a launch that reads it is in flight, or reads it while one that writes it is.
    ///
    /// Tilefish checks the rest: that a tensor fits the region, that the tensors of a call do not overlap where one is written, and
    /// the alignment the binary was compiled for.
    pub unsafe fn from_raw(addr: u64, len_bytes: u64, device_ordinal: i32) -> Self {
        Self { addr, len: len_bytes, device: device_ordinal }
    }

    /// The raw address.
    pub fn addr(self) -> u64 {
        self.addr
    }

    /// How many bytes of memory the region has.
    pub fn len_bytes(self) -> u64 {
        self.len
    }

    /// The ordinal of the device the memory is on.
    pub fn device(self) -> i32 {
        self.device
    }

    /// The region that starts `bytes` bytes into this one and runs to its end; an error when it starts past the end.
    pub fn offset(self, bytes: u64) -> Result<Self> {
        if bytes > self.len {
            return Err(Error::Launch(format!(
                "an offset of {bytes} bytes is past the end of a region of {} bytes",
                self.len
            )));
        }
        let addr = self
            .addr
            .checked_add(bytes)
            .ok_or_else(|| Error::Launch("an offset past the end of the address space".to_owned()))?;
        Ok(Self { addr, len: self.len - bytes, device: self.device })
    }

    /// The region of `len_bytes` bytes that starts `bytes` bytes into this one; an error when it does not fit.
    pub fn slice(self, bytes: u64, len_bytes: u64) -> Result<Self> {
        let end = bytes.checked_add(len_bytes).filter(|end| *end <= self.len);
        if end.is_none() {
            return Err(Error::Launch(format!(
                "bytes {bytes}..{bytes}+{len_bytes} are not within a region of {} bytes",
                self.len
            )));
        }
        let addr = self
            .addr
            .checked_add(bytes)
            .ok_or_else(|| Error::Launch("an offset past the end of the address space".to_owned()))?;
        Ok(Self { addr, len: len_bytes, device: self.device })
    }
}

/// The region a cudarc slice (or view) owns, and the guard that cudarc's event tracking wants kept until the launch is enqueued.
///
/// SAFETY of the `from_raw`: `slice` owns `num_bytes()` bytes at its device address, in the context of its stream, and the borrow the
/// caller holds keeps it alive; the guard makes cudarc record the use on `stream`, so that freeing the slice waits for the launch.
pub(crate) fn region_of<'a, E, S: CudarcDevicePtr<E>>(
    slice: &'a S,
    stream: &'a CudaStream,
) -> (DevicePtr, SyncOnDrop<'a>) {
    let (addr, guard) = slice.device_ptr(stream);
    let len = slice.num_bytes() as u64;
    let device = slice.stream().context().ordinal() as i32;
    // SAFETY: see above.
    (unsafe { DevicePtr::from_raw(addr, len, device) }, guard)
}

/// As [`region_of`], for a region that will be written.
pub(crate) fn region_of_mut<'a, E, S: DevicePtrMut<E>>(
    slice: &'a mut S,
    stream: &'a CudaStream,
) -> (DevicePtr, SyncOnDrop<'a>) {
    let len = slice.num_bytes() as u64;
    let device = slice.stream().context().ordinal() as i32;
    let (addr, guard) = slice.device_ptr_mut(stream);
    // SAFETY: as in `region_of`.
    (unsafe { DevicePtr::from_raw(addr, len, device) }, guard)
}

impl Kernel {
    /// Check the call and launch the kernel on a raw `CUstream`.
    ///
    /// What [`Kernel::launch`] does for a stream that cudarc owns, for a handle that nothing owns: another library's stream, or the
    /// stream that a CUDA graph is being captured on (the call is captured like any kernel node).
    ///
    /// # Safety
    ///
    /// * `stream` must be a valid `CUstream` of the CUDA context the kernel was loaded in (a handle from another context, a destroyed
    ///   stream or a pointer that is not a stream is undefined behavior in the driver), and it must stay valid until this returns;
    /// * that context must be current on this thread (it is checked, and refused if not);
    /// * the tensors keep the obligations of [`DevicePtr::from_raw`]: their memory must stay allocated until the launch has completed
    ///   on `stream`, which this call does not wait for.
    pub unsafe fn launch_raw(&self, stream: sys::CUstream, tensors: &[TensorArg<'_>]) -> Result<()> {
        if !is_current(self.context()) {
            return Err(Error::Launch(format!(
                "'{}' was loaded in another CUDA context than the one that is current on this thread: make its context current first",
                self.stem()
            )));
        }
        self.launch_on(stream, tensors)
    }
}

/// The widest `kernelParams` array: the arguments of a kernel plus its two scratch slots.
pub(crate) const MAX_PARAMS: usize = 64;

/// Whether the CUDA driver library can be loaded here.
pub(crate) fn driver_available() -> bool {
    static PRESENT: OnceLock<bool> = OnceLock::new();
    // SAFETY: `is_culib_present` only tries to open the library by its usual names.
    *PRESENT.get_or_init(|| unsafe { sys::is_culib_present() })
}

fn require_driver() -> Result<()> {
    if driver_available() { Ok(()) } else { Err(Error::NoDriver) }
}

fn describe(result: sys::CUresult, f: unsafe fn(sys::CUresult, *mut *const c_char) -> sys::CUresult) -> Option<String> {
    let mut text: *const c_char = std::ptr::null();
    // SAFETY: `text` is a valid out pointer; on success the driver gives a NUL-terminated static string.
    unsafe {
        if f(result, &mut text) != sys::CUresult::CUDA_SUCCESS || text.is_null() {
            return None;
        }
        Some(CStr::from_ptr(text).to_string_lossy().into_owned())
    }
}

fn check(call: &'static str, result: sys::CUresult) -> Result<()> {
    if result == sys::CUresult::CUDA_SUCCESS {
        return Ok(());
    }
    Err(Error::Driver {
        call,
        code: result as u32,
        name: describe(result, sys::cuGetErrorName).unwrap_or_else(|| "unknown error".to_owned()),
        detail: describe(result, sys::cuGetErrorString).map(|t| format!(": {t}")).unwrap_or_default(),
    })
}

/// The device and context that are current on this thread.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Current {
    pub ordinal: i32,
    pub context: usize,
}

pub(crate) fn current() -> Result<Current> {
    require_driver()?;
    let mut ctx: sys::CUcontext = std::ptr::null_mut();
    // SAFETY: valid out pointers.
    unsafe { check("cuCtxGetCurrent", sys::cuCtxGetCurrent(&mut ctx))? };
    if ctx.is_null() {
        return Err(Error::Launch(
            "no CUDA context is current on this thread: create or bind one first (a cudarc `CudaContext::new` does)"
                .to_owned(),
        ));
    }
    let mut device: sys::CUdevice = 0;
    // SAFETY: valid out pointer, and a context is current.
    unsafe { check("cuCtxGetDevice", sys::cuCtxGetDevice(&mut device))? };
    Ok(Current { ordinal: device, context: ctx as usize })
}

fn device_attribute(ordinal: i32, attribute: sys::CUdevice_attribute, call: &'static str) -> Result<i32> {
    let mut value = 0;
    // SAFETY: valid out pointer.
    unsafe { check(call, sys::cuDeviceGetAttribute(&mut value, attribute, ordinal))? };
    Ok(value)
}

/// `(major, minor)` of the compute capability of a device.
pub(crate) fn compute_capability(ordinal: i32) -> Result<(i32, i32)> {
    require_driver()?;
    Ok((
        device_attribute(
            ordinal,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)",
        )?,
        device_attribute(
            ordinal,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)",
        )?,
    ))
}

/// What a device allows of a launch.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Limits {
    pub grid: [i64; 3],
    pub shared_optin: i64,
}

pub(crate) fn limits(ordinal: i32) -> Result<Limits> {
    use sys::CUdevice_attribute as A;
    let grid = [
        device_attribute(ordinal, A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, "cuDeviceGetAttribute(MAX_GRID_DIM_X)")?,
        device_attribute(ordinal, A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, "cuDeviceGetAttribute(MAX_GRID_DIM_Y)")?,
        device_attribute(ordinal, A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, "cuDeviceGetAttribute(MAX_GRID_DIM_Z)")?,
    ];
    let shared_optin = device_attribute(
        ordinal,
        A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)",
    )?;
    Ok(Limits { grid: grid.map(i64::from), shared_optin: i64::from(shared_optin) })
}

/// A module loaded from a cubin; unloaded when dropped.
#[derive(Debug)]
pub(crate) struct Module {
    raw: sys::CUmodule,
    context: sys::CUcontext,
}

// SAFETY: a CUmodule is a handle to driver-owned state; the driver API is thread safe.
unsafe impl Send for Module {}
// SAFETY: as above.
unsafe impl Sync for Module {}

impl Module {
    /// `cuModuleLoadData` on `image`, in the context that is current (the one `context` names).
    pub(crate) fn load(image: &[u8], context: usize) -> Result<Self> {
        require_driver()?;
        let mut raw: sys::CUmodule = std::ptr::null_mut();
        // SAFETY: `image` is a valid cubin image for the duration of the call, `raw` a valid out pointer.
        unsafe { check("cuModuleLoadData", sys::cuModuleLoadData(&mut raw, image.as_ptr().cast::<c_void>()))? };
        Ok(Self { raw, context: context as sys::CUcontext })
    }

    pub(crate) fn function(&self, symbol: &str) -> Result<Function> {
        let name =
            CString::new(symbol).map_err(|_| Error::Unsupported(format!("the symbol {symbol:?} has a NUL byte")))?;
        let mut raw: sys::CUfunction = std::ptr::null_mut();
        // SAFETY: `self.raw` is a loaded module, `name` is NUL terminated, `raw` a valid out pointer.
        unsafe { check("cuModuleGetFunction", sys::cuModuleGetFunction(&mut raw, self.raw, name.as_ptr()))? };
        Ok(Function { raw })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // The module belongs to the context it was loaded in; make that current for the unload, and put back what was.
        // Every failure (the context is gone, the process is exiting) is ignored: there is nothing to do about it.
        // SAFETY: handles came from the driver; the pushed context is popped again.
        unsafe {
            if sys::cuCtxPushCurrent_v2(self.context) == sys::CUresult::CUDA_SUCCESS {
                let _ = sys::cuModuleUnload(self.raw);
                let mut popped: sys::CUcontext = std::ptr::null_mut();
                let _ = sys::cuCtxPopCurrent_v2(&mut popped);
            }
        }
    }
}

/// A function of a loaded module. It does not outlive the module: [`crate::kernel::Kernel`] keeps both.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Function {
    raw: sys::CUfunction,
}

// SAFETY: a CUfunction is a handle to driver-owned state; the driver API is thread safe.
unsafe impl Send for Function {}
// SAFETY: as above.
unsafe impl Sync for Function {}

impl Function {
    /// The static shared memory of the function in bytes, as the driver reports it once the module is loaded.
    pub(crate) fn static_shared(&self) -> Result<i64> {
        let mut value = 0;
        // SAFETY: valid function and out pointer.
        unsafe {
            check(
                "cuFuncGetAttribute(SHARED_SIZE_BYTES)",
                sys::cuFuncGetAttribute(
                    &mut value,
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                    self.raw,
                ),
            )?;
        }
        Ok(i64::from(value))
    }

    /// Allow `bytes` of dynamic shared memory (needed above the default 48 KiB).
    pub(crate) fn set_max_dynamic_shared(&self, bytes: i32) -> Result<()> {
        // SAFETY: valid function.
        unsafe {
            check(
                "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
                sys::cuFuncSetAttribute(
                    self.raw,
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes,
                ),
            )
        }
    }

    /// The sizes of the parameters of the loaded function, or `None` when these bindings cannot ask (CUDA before 12.4).
    #[cfg(any(feature = "cuda-12040", feature = "cuda-12080", feature = "cuda-13000", feature = "cuda-13020"))]
    pub(crate) fn param_sizes(&self) -> Result<Option<Vec<usize>>> {
        let mut sizes = Vec::new();
        loop {
            let (mut offset, mut size) = (0usize, 0usize);
            // SAFETY: valid function and out pointers; an index past the last parameter is an error, not undefined.
            let result = unsafe { sys::cuFuncGetParamInfo(self.raw, sizes.len(), &mut offset, &mut size) };
            if result != sys::CUresult::CUDA_SUCCESS {
                return Ok(Some(sizes));
            }
            sizes.push(size);
            if sizes.len() > 4096 {
                return Err(Error::Unsupported("a kernel with more than 4096 parameters".to_owned()));
            }
        }
    }

    /// The sizes of the parameters of the loaded function, or `None` when these bindings cannot ask (CUDA before 12.4).
    #[cfg(not(any(feature = "cuda-12040", feature = "cuda-12080", feature = "cuda-13000", feature = "cuda-13020")))]
    pub(crate) fn param_sizes(&self) -> Result<Option<Vec<usize>>> {
        Ok(None)
    }

    /// `cuLaunchKernel` on `stream` with `values[..count]` as the kernel's arguments, in the 64-bit slots a launch uses.
    pub(crate) fn launch(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        stream: sys::CUstream,
        values: &mut [u64; MAX_PARAMS],
        count: usize,
    ) -> Result<()> {
        let mut params: [*mut c_void; MAX_PARAMS] = [std::ptr::null_mut(); MAX_PARAMS];
        for (slot, value) in params.iter_mut().zip(values.iter_mut()).take(count) {
            *slot = std::ptr::from_mut(value).cast::<c_void>();
        }
        // SAFETY: `params[..count]` point at live 64-bit slots, which the driver reads (and copies) before it returns;
        // the grid, block and shared memory were checked against the device by the caller; the stream is the caller's.
        unsafe {
            check(
                "cuLaunchKernel",
                sys::cuLaunchKernel(
                    self.raw,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared,
                    stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )
        }
    }
}

/// Whether `context` (an address) is the context that is current on this thread.
pub(crate) fn is_current(context: usize) -> bool {
    let mut ctx: sys::CUcontext = std::ptr::null_mut();
    // SAFETY: valid out pointer.
    let result = unsafe { sys::cuCtxGetCurrent(&mut ctx) };
    result == sys::CUresult::CUDA_SUCCESS && ctx as usize == context
}
