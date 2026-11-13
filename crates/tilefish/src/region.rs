//! Regions of device memory that a host vouches for: the address, the length and the device, for a tensor that is not made of a
//! cudarc slice.
//!
//! This is where the host takes on its first obligation, in one item that says why it is sound: [`DevicePtr::from_raw`], where the host
//! vouches for a region (where it is, how many bytes, on which device, and that it outlives the launches that use it). A region has a
//! lifetime, `DevicePtr<'a>`, so that a host that can say what the memory borrows from says it, and everything made of the region
//! (a tensor, a launch argument) cannot outlive it. A tensor made of a cudarc `CudaSlice` or view has no region at all: it holds the
//! borrow of the slice itself (see `memory`), and the launch hands the slice back to cudarc. Everything built from a region is
//! checked: a tensor is refused when its shape needs more bytes than its pointer vouches for. The other obligation, a raw
//! `CUstream`, is in `ffi`.

use std::marker::PhantomData;

use crate::error::{Error, Result};

/// A region of device memory: its address, how many bytes of it the holder vouches for, and the device it is on, for as long as `'a`.
///
/// A tensor is made from one (`TensorRef::new`, `TensorMut::new`, [`crate::TensorArg::new`]) and is refused when its shape needs more bytes
/// than the region has, so no launch can be made to read or write past what the host vouched for. The host vouches by calling
/// [`DevicePtr::from_raw`], which is `unsafe` and returns a region of whatever lifetime the caller says (give it the lifetime of
/// the borrow the memory comes from, if there is one); [`DevicePtr::offset`] and [`DevicePtr::slice`] make smaller regions of a
/// region, checked, with its lifetime. A tensor over a cudarc `CudaSlice` or `CudaView` needs no region and no `unsafe`:
/// `TensorRef::from_cuda`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePtr<'a> {
    addr: u64,
    len: u64,
    device: i32,
    _borrow: PhantomData<&'a ()>,
}

impl DevicePtr<'_> {
    /// The region of `len_bytes` bytes at `addr` (a `CUdeviceptr`) in the memory of the device with ordinal `device_ordinal`, for the
    /// lifetime the caller chooses.
    ///
    /// # Safety
    ///
    /// The caller vouches that
    ///
    /// * `addr .. addr + len_bytes` is allocated device memory of the device `device_ordinal` (not host memory, not a freed or
    ///   not yet allocated address), writable if a [`TensorMut`](crate::TensorMut) is made of it;
    /// * it stays allocated, and is not reused, for the whole lifetime `'a` of the region and until every launch that was given a
    ///   tensor made of it has completed on the device (a launch is asynchronous: its stream orders it, it is not finished when
    ///   `launch` returns). The lifetime is the caller's to bind: make it the lifetime of a borrow of whatever owns the memory, and
    ///   the compiler refuses a tensor that outlives it. A launch that is *captured in a CUDA graph* is launched again at every replay
    ///   of the graph, which holds the address and nothing else: the memory must stay allocated for as long as the graph can be
    ///   replayed, and the lifetime cannot say so;
    /// * nothing else writes it while a launch that reads it is in flight, or reads it while one that writes it is. Nothing orders a
    ///   launch on a raw region against other streams' use of the memory (cudarc's event tracking is for its own slices, which
    ///   `TensorRef::from_cuda` hands back to it): that is the caller's too.
    ///
    /// Tilefish checks the rest: that a tensor fits the region, that the tensors of a call do not overlap where one is written, and
    /// the alignment the binary was compiled for.
    #[allow(unsafe_code)] // the declaration of an `unsafe fn`: the obligations are the ones above, and nothing is done here
    pub unsafe fn from_raw(addr: u64, len_bytes: u64, device_ordinal: i32) -> Self {
        Self { addr, len: len_bytes, device: device_ordinal, _borrow: PhantomData }
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
        Ok(Self { addr, len: self.len - bytes, device: self.device, _borrow: PhantomData })
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
        Ok(Self { addr, len: len_bytes, device: self.device, _borrow: PhantomData })
    }
}
