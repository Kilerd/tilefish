//! Element types and the tensors a launch is given.
//!
//! A tensor of a launch is a region of device memory (a [`DevicePtr`]: an address, the bytes the host vouches for, the device it
//! is on) and a shape that must fit it; the host owns the memory. [`TensorRef`] (read only) and [`TensorMut`] (written by the kernel: `Out` and `InOut`) carry the element type
//! in their type, so a generated launcher cannot be given a `bf16` tensor where it wants `f32`. Both are contiguous
//! row-major tensors: the binaries take no strides.

use std::marker::PhantomData;

use cudarc::driver::{CudaStream, SyncOnDrop};

use crate::error::{Error, Result};
use crate::region::{self, DevicePtr};

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

mod sealed {
    pub trait Sealed {}
}

/// The element type of a [`TensorRef`] or [`TensorMut`], as a type.
pub trait Element: sealed::Sealed + Copy + 'static {
    /// The runtime tag of the type.
    const DTYPE: DType;
}

macro_rules! elements {
    ($($(#[$doc:meta])* $name:ident => $tag:ident),* $(,)?) => {$(
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name;
        impl sealed::Sealed for $name {}
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

/// The bytes a tensor of `dtype` and `shape` covers (`None` when they do not fit 64 bits).
fn bytes_of(shape: &[usize], dtype: DType) -> Option<u64> {
    let numel = shape.iter().try_fold(1u128, |acc, d| acc.checked_mul(*d as u128))?;
    u64::try_from(numel.checked_mul(dtype.size() as u128)?).ok()
}

/// The error of a tensor that needs more bytes than its region has.
fn too_big(shape: &[usize], dtype: DType, region: DevicePtr) -> Error {
    Error::Launch(format!(
        "a tensor of shape {shape:?} and dtype {} needs {} bytes, but its region has {} bytes",
        dtype.name(),
        bytes_of(shape, dtype).map_or_else(|| "more than 2^64".to_owned(), |b| b.to_string()),
        region.len_bytes()
    ))
}

/// The facts about one tensor that a launch checks, without the element type in the Rust type (what [`TensorRef`] and
/// [`TensorMut`] turn into). Build one with [`TensorArg::new`] to launch a kernel without a generated launcher.
#[derive(Debug, Clone, Copy)]
pub struct TensorArg<'a> {
    pub(crate) ptr: u64,
    pub(crate) shape: &'a [usize],
    pub(crate) dtype: DType,
    pub(crate) device: i32,
}

impl<'a> TensorArg<'a> {
    /// A contiguous row-major tensor of `dtype` and `shape` in the region `ptr`, on the device the region is on.
    ///
    /// Refused when the shape needs more bytes than the region has: a launch can then never touch memory the host did not vouch for.
    pub fn new(ptr: DevicePtr, shape: &'a [usize], dtype: DType) -> Result<Self> {
        match bytes_of(shape, dtype) {
            Some(needed) if needed <= ptr.len_bytes() => {
                Ok(Self { ptr: ptr.addr(), shape, dtype, device: ptr.device() })
            }
            _ => Err(too_big(shape, dtype, ptr)),
        }
    }

    /// The device address.
    pub fn addr(&self) -> u64 {
        self.ptr
    }

    /// The shape.
    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    /// The element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The device ordinal.
    pub fn device(&self) -> i32 {
        self.device
    }
}

macro_rules! tensor_type {
    ($(#[$doc:meta])* $name:ident, $marker:ty) => {
        $(#[$doc])*
        ///
        /// It is made from a region the host vouches for (`new`, from a [`DevicePtr`]) or from a cudarc slice or view whose length is
        /// known (`from_cuda`, no `unsafe`); either way it is refused when the shape needs more bytes than the region has. One from a
        /// cudarc slice keeps cudarc's event tracking until it is dropped, which is after the launch it is given to has been enqueued:
        /// so it is not `Copy`, and a launcher takes it by value.
        #[derive(Debug)]
        pub struct $name<'a, T: Element> {
            ptr: DevicePtr,
            shape: &'a [usize],
            _guard: Option<SyncOnDrop<'a>>,
            _marker: PhantomData<$marker>,
        }

        impl<'a, T: Element> $name<'a, T> {
            /// A contiguous row-major tensor of `shape` in the region `ptr`.
            pub fn new(ptr: DevicePtr, shape: &'a [usize]) -> Result<Self> {
                match bytes_of(shape, T::DTYPE) {
                    Some(needed) if needed <= ptr.len_bytes() => Ok(Self { ptr, shape, _guard: None, _marker: PhantomData }),
                    _ => Err(too_big(shape, T::DTYPE, ptr)),
                }
            }

            /// The device pointer (the region).
            pub fn ptr(&self) -> DevicePtr {
                self.ptr
            }

            /// The shape.
            pub fn shape(&self) -> &'a [usize] {
                self.shape
            }

            /// The device ordinal.
            pub fn device(&self) -> i32 {
                self.ptr.device()
            }

            /// The number of elements.
            pub fn elements(&self) -> usize {
                self.shape.iter().product()
            }

            /// The same tensor without its element type in the Rust type, for [`Kernel::launch`](crate::Kernel::launch).
            pub fn arg(&self) -> TensorArg<'a> {
                TensorArg { ptr: self.ptr.addr(), shape: self.shape, dtype: T::DTYPE, device: self.ptr.device() }
            }
        }
    };
}

tensor_type!(
    /// A tensor a kernel only reads (an `In` parameter).
    TensorRef,
    &'a T
);
tensor_type!(
    /// A tensor a kernel writes (an `Out` or `InOut` parameter).
    TensorMut,
    &'a mut T
);

impl<'a, T: Element> TensorRef<'a, T> {
    /// A tensor over a cudarc `CudaSlice` or `CudaView`, whose length is known, as seen from `stream` (the stream the launch is
    /// made on: cudarc orders the slice's earlier writes before it, and records the use on it). The slice may hold any type: its
    /// bytes are the tensor's elements, as `T` says.
    pub fn from_cuda<E, S: cudarc::driver::DevicePtr<E>>(
        slice: &'a S,
        stream: &'a CudaStream,
        shape: &'a [usize],
    ) -> Result<Self> {
        let (ptr, guard) = region::region_of(slice, stream);
        let mut tensor = Self::new(ptr, shape)?;
        tensor._guard = Some(guard);
        Ok(tensor)
    }
}

impl<'a, T: Element> TensorMut<'a, T> {
    /// A tensor over a cudarc `CudaSlice` or `CudaViewMut` that the kernel writes, as seen from `stream`; see
    /// [`TensorRef::from_cuda`].
    pub fn from_cuda<E, S: cudarc::driver::DevicePtrMut<E>>(
        slice: &'a mut S,
        stream: &'a CudaStream,
        shape: &'a [usize],
    ) -> Result<Self> {
        let (ptr, guard) = region::region_of_mut(slice, stream);
        let mut tensor = Self::new(ptr, shape)?;
        tensor._guard = Some(guard);
        Ok(tensor)
    }
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
