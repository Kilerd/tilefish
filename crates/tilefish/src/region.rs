//! Regions of device memory: the address, the length and the device that a host vouches for, from which every tensor is made.
//!
//! This is where the host takes on its first obligation, in three items that each say why they are sound: [`DevicePtr::from_raw`]
//! (the host vouches for a region: where it is, how many bytes, on which device, and that it outlives the launches that use
//! it) and the two regions made of a cudarc slice, `region_of` and `region_of_mut`. Everything built from them is checked: a tensor
//! is refused when its shape needs more bytes than its pointer vouches for. The other obligation, a raw `CUstream`, is in `ffi`.

use cudarc::driver::{CudaStream, DevicePtr as CudarcDevicePtr, DevicePtrMut, SyncOnDrop};

use crate::error::{Error, Result};

/// A region of device memory: its address, how many bytes of it the holder vouches for, and the device it is on.
///
/// A tensor is made from one (`TensorRef::new`, `TensorMut::new`, [`crate::TensorArg::new`]) and is refused when its shape needs more bytes
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
    #[allow(unsafe_code)] // the declaration of an `unsafe fn`: the obligations are the ones above, and nothing is done here
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
#[allow(unsafe_code)]
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
#[allow(unsafe_code)]
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
