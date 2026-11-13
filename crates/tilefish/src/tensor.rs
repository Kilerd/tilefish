//! Element types and the tensors a launch is given.
//!
//! A tensor of a launch is memory and a shape that must fit it. The memory is either a cudarc `CudaSlice` or view, borrowed
//! (`from_cuda`: no `unsafe`; making the tensor records nothing on the slice, and the launch gives the slice to cudarc's launch
//! builder on the launch stream, and makes the slice's own stream wait for the kernel: what that guarantees is in `memory`), or a
//! region of device memory that the host vouches for (a [`DevicePtr`]: an address, the bytes, the device, and a lifetime; `new`,
//! which needs `unsafe` to make the region, and nothing tracks it). [`TensorRef`] (read only) and [`TensorMut`] (written by the
//! kernel: `Out` and `InOut`) carry the element type in their type, so a generated launcher cannot be given a `bf16` tensor where
//! it wants `f32`. A tensor is a contiguous row-major tensor, or (`from_cuda_strided`, `new_strided`) a *view*: a shape, element strides and
//! the element offset of its first element from the start of its memory, for a parameter the manifest declares strided (`Strided[...]`
//! in Python); the view must fit its memory (its *extent*, one more than the largest element offset it reaches, from its first element on).
//! A tensor borrows what it is made of, for as long as it lives, and what it turns into for a launch ([`TensorArg`]) does too.

use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::driver::CudaStream;

use crate::error::{Error, Result};
use crate::manifest::Access;
use crate::memory::{CudaRead, CudaWrite, sealed};
use crate::region::DevicePtr;

/// An element type a binary can be given (the dtypes of Python's `TORCH_DTYPE_NAMES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// `bf16`, 2 bytes.
    Bf16,
    /// `f16`, 2 bytes.
    F16,
    /// `f32`, 4 bytes.
    F32,
    /// `i32`, 4 bytes.
    I32,
    /// `i64`, 8 bytes.
    I64,
    /// `u8`, 1 byte.
    U8,
    /// `e4m3` (Triton's `fp8e4nv`), 1 byte.
    E4m3,
    /// `e5m2` (Triton's `fp8e5`), 1 byte.
    E5m2,
}

impl DType {
    /// The name a manifest and a tilefish signature use (`"bf16"`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::E4m3 => "e4m3",
            Self::E5m2 => "e5m2",
        }
    }

    /// The dtype called `name` in a manifest, if it is one a binary can be run on.
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Bf16, Self::F16, Self::F32, Self::I32, Self::I64, Self::U8, Self::E4m3, Self::E5m2]
            .into_iter()
            .find(|d| d.name() == name)
    }

    /// The bytes of one element.
    pub fn size(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::I64 => 8,
            Self::U8 | Self::E4m3 | Self::E5m2 => 1,
        }
    }

    /// How Triton spells a pointer to it (the `type` of a pointer parameter of a manifest).
    pub fn pointer_type(self) -> &'static str {
        match self {
            Self::Bf16 => "*bf16",
            Self::F16 => "*fp16",
            Self::F32 => "*fp32",
            Self::I32 => "*i32",
            Self::I64 => "*i64",
            Self::U8 => "*u8",
            Self::E4m3 => "*fp8e4nv",
            Self::E5m2 => "*fp8e5",
        }
    }
}

mod element_sealed {
    pub trait Sealed {}
}

/// The element type of a [`TensorRef`] or [`TensorMut`], as a type.
pub trait Element: element_sealed::Sealed + Copy + 'static {
    /// The runtime tag of the type.
    const DTYPE: DType;
}

macro_rules! elements {
    ($($(#[$doc:meta])* $name:ident => $tag:ident),* $(,)?) => {$(
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name;
        impl element_sealed::Sealed for $name {}
        impl Element for $name {
            const DTYPE: DType = DType::$tag;
        }
    )*};
}

elements! {
    /// Marker of `bf16` tensors.
    Bf16 => Bf16,
    /// Marker of `f16` tensors.
    F16 => F16,
    /// Marker of `f32` tensors.
    F32 => F32,
    /// Marker of `i32` tensors.
    I32 => I32,
    /// Marker of `i64` tensors.
    I64 => I64,
    /// Marker of `u8` tensors.
    U8 => U8,
    /// Marker of `e4m3` (`fp8e4nv`) tensors.
    E4m3 => E4m3,
    /// Marker of `e5m2` (`fp8e5`) tensors.
    E5m2 => E5m2,
}

/// How many elements a tensor reaches from its first one: the number of elements of a dense tensor, and for a view its *extent*, one more
/// than the largest element offset (`1 + sum((size - 1) * stride)`; 0 when a size is 0). Saturating: a number that does not fit is far
/// more than any memory has.
pub(crate) fn reach(shape: &[usize], strides: Option<&[usize]>) -> u128 {
    if shape.contains(&0) {
        return 0;
    }
    match strides {
        None => shape.iter().fold(1u128, |acc, d| acc.saturating_mul(*d as u128)),
        Some(strides) => shape
            .iter()
            .zip(strides)
            .fold(1u128, |acc, (n, s)| acc.saturating_add((*n as u128 - 1).saturating_mul(*s as u128))),
    }
}

/// The bytes from the start of the memory to the end of the last element a tensor reaches (`None` when they do not fit 64 bits).
fn bytes_of(shape: &[usize], strides: Option<&[usize]>, offset: usize, dtype: DType) -> Option<u64> {
    let elements = (offset as u128).checked_add(reach(shape, strides))?;
    u64::try_from(elements.checked_mul(dtype.size() as u128)?).ok()
}

/// The error of a tensor that needs more bytes than its memory has.
fn too_big(shape: &[usize], strides: Option<&[usize]>, offset: usize, dtype: DType, len_bytes: u64) -> Error {
    let needed = bytes_of(shape, strides, offset, dtype).map_or_else(|| "more than 2^64".to_owned(), |b| b.to_string());
    match strides {
        None => Error::Launch(format!(
            "a tensor of shape {shape:?} and dtype {} needs {needed} bytes, but its region has {len_bytes} bytes",
            dtype.name()
        )),
        Some(strides) => Error::Launch(format!(
            "a view of shape {shape:?} and strides {strides:?}, whose first element is element {offset} of its region (dtype {}), reaches {needed} bytes \
             from the start of its region, but the region has {len_bytes} bytes",
            dtype.name()
        )),
    }
}

fn check_fits(shape: &[usize], strides: Option<&[usize]>, offset: usize, dtype: DType, len_bytes: u64) -> Result<()> {
    if let Some(strides) = strides
        && strides.len() != shape.len()
    {
        return Err(Error::Launch(format!(
            "a view needs one stride per axis: the shape {shape:?} has {} axes, the strides {strides:?} have {}",
            shape.len(),
            strides.len()
        )));
    }
    match bytes_of(shape, strides, offset, dtype) {
        Some(needed) if needed <= len_bytes => Ok(()),
        _ => Err(too_big(shape, strides, offset, dtype, len_bytes)),
    }
}

/// What a tensor is made of, for a launch.
pub(crate) enum Backing<'a> {
    /// A region the host vouched for: the launch passes its address.
    Raw,
    /// A cudarc slice or view that is read: the launch gives it to cudarc's launch builder.
    Read(&'a dyn sealed::Read),
    /// A cudarc slice or view that is written.
    Write(&'a mut dyn sealed::Write),
}

/// The facts about one tensor that a launch checks, without the element type in the Rust type (what [`TensorRef`] and
/// [`TensorMut`] turn into with `into_arg`). Build one with [`TensorArg::new`] to launch a kernel without a generated launcher.
///
/// It borrows the memory it is made of for `'a` (so it cannot outlive a `CudaSlice`, nor the region of an `unsafe` constructor whose
/// lifetime was bound), it is used once (it is not `Copy` or `Clone`: it may hold the exclusive borrow of a slice that is written),
/// and it says what the kernel may do to it ([`Access`]), which a launch checks against the parameter it is given for.
pub struct TensorArg<'a> {
    /// The address of the first element (for a view, `base + offset * itemsize`).
    pub(crate) ptr: u64,
    pub(crate) shape: &'a [usize],
    /// The element strides of a view; `None` is a dense, row-major tensor.
    pub(crate) strides: Option<&'a [usize]>,
    /// Bytes from the start of the cudarc slice to the first element (0 unless it is a view): the launch hands cudarc a view of the bytes
    /// from there, so that the kernel is given this tensor's address and cudarc still tracks the slice.
    pub(crate) byte_offset: u64,
    pub(crate) dtype: DType,
    pub(crate) device: i32,
    /// The `CUcontext` the memory was allocated in, when it is known (it is for cudarc memory): a tensor of another context on the
    /// same device is refused.
    pub(crate) context: Option<usize>,
    /// The stream cudarc memory was allocated on (and will be freed on), when the tensor is cudarc memory: a launch on another stream
    /// makes it wait for the kernel (see `memory`).
    pub(crate) owner: Option<Arc<CudaStream>>,
    pub(crate) access: Access,
    pub(crate) backing: Backing<'a>,
}

impl std::fmt::Debug for TensorArg<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorArg")
            .field("addr", &format_args!("{:#x}", self.ptr))
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .field("access", &self.access)
            .field(
                "backing",
                &match self.backing {
                    Backing::Raw => "a region the host vouched for",
                    Backing::Read(_) => "a cudarc slice, read",
                    Backing::Write(_) => "a cudarc slice, written",
                },
            )
            .finish_non_exhaustive()
    }
}

impl<'a> TensorArg<'a> {
    /// A contiguous row-major tensor of `dtype` and `shape` in the region `ptr`, on the device the region is on, which the kernel
    /// may use as `access` says: [`Access::In`] it reads, [`Access::Out`] it writes, [`Access::InOut`] it does both. A launch refuses a
    /// tensor that is `In` for a parameter that is `Out` or `InOut`.
    ///
    /// Refused when the shape needs more bytes than the region has: a launch can then never touch memory the host did not vouch for.
    pub fn new(ptr: DevicePtr<'a>, shape: &'a [usize], dtype: DType, access: Access) -> Result<Self> {
        check_fits(shape, None, 0, dtype, ptr.len_bytes())?;
        Ok(Self {
            ptr: ptr.addr(),
            shape,
            strides: None,
            byte_offset: 0,
            dtype,
            device: ptr.device(),
            context: None,
            owner: None,
            access,
            backing: Backing::Raw,
        })
    }

    /// A view of `dtype` and `shape` with element `strides` in the region `ptr`, whose first element is element `offset` of the region
    /// (the address of the view is `ptr + offset * itemsize`): for a parameter the manifest declares strided. A tensor made with
    /// [`TensorArg::new`] is the view with the dense row-major strides.
    ///
    /// Refused when the view reaches further than the region: the view's extent (one more than the largest element offset it reaches, which
    /// a stride of 0, as of an expanded view, does not add to) counted from its first element must fit the bytes the region has, so a
    /// launch can never touch memory the host did not vouch for. The strides are unsigned: a view that walks backwards is not
    /// representable.
    pub fn new_strided(
        ptr: DevicePtr<'a>,
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
        dtype: DType,
        access: Access,
    ) -> Result<Self> {
        check_fits(shape, Some(strides), offset, dtype, ptr.len_bytes())?;
        let byte_offset = byte_offset_of(offset, dtype, ptr.len_bytes())?;
        let addr = ptr
            .addr()
            .checked_add(byte_offset)
            .ok_or_else(|| Error::Launch("an offset past the end of the address space".to_owned()))?;
        Ok(Self {
            ptr: addr,
            shape,
            strides: Some(strides),
            byte_offset: 0, // a region has nothing cudarc tracks: the address is all
            dtype,
            device: ptr.device(),
            context: None,
            owner: None,
            access,
            backing: Backing::Raw,
        })
    }

    /// The device address.
    pub fn addr(&self) -> u64 {
        self.ptr
    }

    /// The shape.
    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    /// The element strides of a view (`None`: a dense, row-major tensor).
    pub fn strides(&self) -> Option<&'a [usize]> {
        self.strides
    }

    /// The element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The device ordinal.
    pub fn device(&self) -> i32 {
        self.device
    }

    /// What the kernel may do to the tensor.
    pub fn access(&self) -> Access {
        self.access
    }
}

macro_rules! tensor_type {
    ($(#[$doc:meta])* $name:ident, $access:expr) => {
        $(#[$doc])*
        ///
        /// It is made from a cudarc slice or view (`from_cuda`, no `unsafe`: the length is known, the borrow keeps the memory alive, making
        /// it records nothing on the slice, and the launch gives the slice to cudarc and makes the slice's own stream wait for the
        /// kernel: see [`Kernel::launch`](crate::Kernel::launch)) or from a region the host vouches for (`new`, from a
        /// [`DevicePtr`], which nothing tracks); either way it is refused when the shape needs more bytes than the memory has. It is not `Copy`, and a
        /// launcher takes it by value. It has no address to give back as a region: only [`addr`](Self::addr), a number.
        pub struct $name<'a, T: Element> {
            arg: TensorArg<'a>,
            _marker: PhantomData<T>,
        }

        impl<T: Element> std::fmt::Debug for $name<'_, T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_tuple(stringify!($name)).field(&self.arg).finish()
            }
        }

        impl<'a, T: Element> $name<'a, T> {
            /// A contiguous row-major tensor of `shape` in the region `ptr`.
            pub fn new(ptr: DevicePtr<'a>, shape: &'a [usize]) -> Result<Self> {
                Ok(Self { arg: TensorArg::new(ptr, shape, T::DTYPE, $access)?, _marker: PhantomData })
            }

            /// A view of `shape` with element `strides` in the region `ptr`, whose first element is element `offset` of the region (see
            /// [`TensorArg::new_strided`]): refused when it reaches further than the region.
            pub fn new_strided(
                ptr: DevicePtr<'a>,
                shape: &'a [usize],
                strides: &'a [usize],
                offset: usize,
            ) -> Result<Self> {
                Ok(Self { arg: TensorArg::new_strided(ptr, shape, strides, offset, T::DTYPE, $access)?, _marker: PhantomData })
            }

            /// The device address, a plain number: it cannot be turned back into a tensor without `unsafe`
            /// ([`DevicePtr::from_raw`]).
            pub fn addr(&self) -> u64 {
                self.arg.ptr
            }

            /// The shape.
            pub fn shape(&self) -> &'a [usize] {
                self.arg.shape
            }

            /// The element strides of a view (`None`: a dense, row-major tensor).
            pub fn strides(&self) -> Option<&'a [usize]> {
                self.arg.strides
            }

            /// The device ordinal.
            pub fn device(&self) -> i32 {
                self.arg.device
            }

            /// The number of elements.
            pub fn elements(&self) -> usize {
                self.arg.shape.iter().product()
            }

            /// The same tensor without its element type in the Rust type, for [`Kernel::launch`](crate::Kernel::launch). It keeps
            /// the borrow of what it is made of.
            pub fn into_arg(self) -> TensorArg<'a> {
                self.arg
            }
        }
    };
}

tensor_type!(
    /// A tensor a kernel only reads (an `In` parameter).
    TensorRef,
    Access::In
);
tensor_type!(
    /// A tensor a kernel writes (an `Out` or `InOut` parameter).
    TensorMut,
    Access::InOut
);

impl<'a, T: Element> TensorRef<'a, T> {
    /// A tensor over a cudarc `CudaSlice` or `CudaView`, whose length is known. The slice may hold any type: its bytes are the
    /// tensor's elements, as `T` says. The tensor borrows the slice (so it is gone before the slice is), making it records nothing on
    /// the slice (it only waits, on the slice's own stream, for the slice's last write), and a launch gives the slice to cudarc as a
    /// read, on the stream it launches on, and makes the slice's own stream wait for the kernel.
    pub fn from_cuda<S: CudaRead + 'a>(slice: &'a S, shape: &'a [usize]) -> Result<Self> {
        Self::over(slice, shape, None, 0)
    }

    /// A view over a cudarc `CudaSlice` or `CudaView`: `shape`, element `strides`, and the element `offset` of its first element from the
    /// start of the slice (the kernel is given the address `slice + offset * itemsize`). For a parameter the manifest declares strided
    /// (`Strided[...]` in Python); any element strides and any offset are accepted by the tensor, a launch checks them against the
    /// parameter (a stride that the binary fixes, the alignment it was compiled for, the extent it indexes with `i32`).
    ///
    /// Refused when the view reaches further than the slice (its extent, counted from its first element, must fit) and when the strides do
    /// not have one entry per axis. The view may start at any byte (a `bf16` view of a `CudaSlice<f32>` may start at an odd element).
    /// Everything else is as for [`TensorRef::from_cuda`]: the tensor borrows the slice, records nothing on it, and a launch hands cudarc the
    /// slice's bytes from the first element on, on the stream it launches on, so cudarc's ordering of the slice is what it is for a dense
    /// tensor.
    pub fn from_cuda_strided<S: CudaRead + 'a>(
        slice: &'a S,
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Result<Self> {
        Self::over(slice, shape, Some(strides), offset)
    }

    fn over<S: CudaRead + 'a>(
        slice: &'a S,
        shape: &'a [usize],
        strides: Option<&'a [usize]>,
        offset: usize,
    ) -> Result<Self> {
        let region = sealed::Memory::region(slice);
        check_fits(shape, strides, offset, T::DTYPE, region.len_bytes)?;
        let byte_offset = byte_offset_of(offset, T::DTYPE, region.len_bytes)?;
        if byte_offset != 0 && slice.bytes_from(byte_offset).is_none() {
            return Err(unaligned_view(offset, byte_offset));
        }
        let arg = TensorArg {
            ptr: region.addr + byte_offset,
            shape,
            strides,
            byte_offset,
            dtype: T::DTYPE,
            device: region.ordinal,
            context: Some(region.context),
            owner: Some(region.owner),
            access: Access::In,
            backing: Backing::Read(slice),
        };
        Ok(Self { arg, _marker: PhantomData })
    }
}

impl<'a, T: Element> TensorMut<'a, T> {
    /// A tensor over a cudarc `CudaSlice` or `CudaViewMut` that the kernel writes; see [`TensorRef::from_cuda`]. It holds the
    /// exclusive borrow of the slice, and a launch gives the slice to cudarc as a write.
    pub fn from_cuda<S: CudaWrite + 'a>(slice: &'a mut S, shape: &'a [usize]) -> Result<Self> {
        Self::over(slice, shape, None, 0)
    }

    /// A view over a cudarc `CudaSlice` or `CudaViewMut` that the kernel writes: see [`TensorRef::from_cuda_strided`]. (The binaries of a
    /// manifest only take a strided pointer that is read, so a launch refuses it for a parameter that is written and not contiguous.)
    pub fn from_cuda_strided<S: CudaWrite + 'a>(
        slice: &'a mut S,
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Result<Self> {
        Self::over(slice, shape, Some(strides), offset)
    }

    fn over<S: CudaWrite + 'a>(
        slice: &'a mut S,
        shape: &'a [usize],
        strides: Option<&'a [usize]>,
        offset: usize,
    ) -> Result<Self> {
        let region = sealed::Memory::region(&*slice);
        check_fits(shape, strides, offset, T::DTYPE, region.len_bytes)?;
        let byte_offset = byte_offset_of(offset, T::DTYPE, region.len_bytes)?;
        if byte_offset != 0 && slice.bytes_from_mut(byte_offset).is_none() {
            return Err(unaligned_view(offset, byte_offset));
        }
        let arg = TensorArg {
            ptr: region.addr + byte_offset,
            shape,
            strides,
            byte_offset,
            dtype: T::DTYPE,
            device: region.ordinal,
            context: Some(region.context),
            owner: Some(region.owner),
            access: Access::InOut,
            backing: Backing::Write(slice),
        };
        Ok(Self { arg, _marker: PhantomData })
    }
}

/// The bytes from the start of the slice to a view's first element: `offset` elements of `dtype` (already known to fit `len_bytes`).
fn byte_offset_of(offset: usize, dtype: DType, len_bytes: u64) -> Result<u64> {
    (offset as u64).checked_mul(dtype.size() as u64).filter(|b| *b <= len_bytes).ok_or_else(|| {
        Error::Launch(format!("the element offset {offset} is past the end of a region of {len_bytes} bytes"))
    })
}

fn unaligned_view(offset: usize, byte_offset: u64) -> Error {
    Error::Launch(format!("a view that starts at element {offset} ({byte_offset} bytes) has no bytes in this slice"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtypes_round_trip_through_their_names() {
        for d in [DType::Bf16, DType::F16, DType::F32, DType::I32, DType::I64, DType::U8, DType::E4m3, DType::E5m2] {
            assert_eq!(DType::from_name(d.name()), Some(d));
        }
        assert_eq!(DType::from_name("e2m1"), None);
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::E4m3.pointer_type(), "*fp8e4nv");
    }
}
