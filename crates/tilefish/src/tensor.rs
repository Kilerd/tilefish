//! Element types and the tensors a launch is given.
//!
//! A tensor of a launch is memory and a shape that must fit it. The memory is either a cudarc `CudaSlice` or view, borrowed
//! (`from_cuda`: no `unsafe`, and the launch hands the slice back to cudarc on its own stream, which orders it against every other use
//! of it, see `memory`), or a region of device memory that the host vouches for (a [`DevicePtr`]: an address, the bytes, the device,
//! and a lifetime; `new`, which needs `unsafe` to make the region). [`TensorRef`] (read only) and [`TensorMut`] (written by the
//! kernel: `Out` and `InOut`) carry the element type in their type, so a generated launcher cannot be given a `bf16` tensor where
//! it wants `f32`. They are contiguous row-major tensors: the binaries take no strides. A tensor borrows what it is made of, for
//! as long as it lives, and what it turns into for a launch ([`TensorArg`]) does too.

use std::marker::PhantomData;

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

/// The bytes a tensor of `dtype` and `shape` covers (`None` when they do not fit 64 bits).
fn bytes_of(shape: &[usize], dtype: DType) -> Option<u64> {
    let numel = shape.iter().try_fold(1u128, |acc, d| acc.checked_mul(*d as u128))?;
    u64::try_from(numel.checked_mul(dtype.size() as u128)?).ok()
}

/// The error of a tensor that needs more bytes than its memory has.
fn too_big(shape: &[usize], dtype: DType, len_bytes: u64) -> Error {
    Error::Launch(format!(
        "a tensor of shape {shape:?} and dtype {} needs {} bytes, but its region has {len_bytes} bytes",
        dtype.name(),
        bytes_of(shape, dtype).map_or_else(|| "more than 2^64".to_owned(), |b| b.to_string()),
    ))
}

fn check_fits(shape: &[usize], dtype: DType, len_bytes: u64) -> Result<()> {
    match bytes_of(shape, dtype) {
        Some(needed) if needed <= len_bytes => Ok(()),
        _ => Err(too_big(shape, dtype, len_bytes)),
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
    pub(crate) ptr: u64,
    pub(crate) shape: &'a [usize],
    pub(crate) dtype: DType,
    pub(crate) device: i32,
    /// The `CUcontext` the memory was allocated in, when it is known (it is for cudarc memory): a tensor of another context on the
    /// same device is refused.
    pub(crate) context: Option<usize>,
    pub(crate) access: Access,
    pub(crate) backing: Backing<'a>,
}

impl std::fmt::Debug for TensorArg<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorArg")
            .field("addr", &format_args!("{:#x}", self.ptr))
            .field("shape", &self.shape)
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
        check_fits(shape, dtype, ptr.len_bytes())?;
        Ok(Self { ptr: ptr.addr(), shape, dtype, device: ptr.device(), context: None, access, backing: Backing::Raw })
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

    /// What the kernel may do to the tensor.
    pub fn access(&self) -> Access {
        self.access
    }
}

macro_rules! tensor_type {
    ($(#[$doc:meta])* $name:ident, $access:expr) => {
        $(#[$doc])*
        ///
        /// It is made from a cudarc slice or view (`from_cuda`, no `unsafe`: the length is known, the borrow keeps the memory alive,
        /// and the launch gives the slice to cudarc, which orders it) or from a region the host vouches for (`new`, from a
        /// [`DevicePtr`]); either way it is refused when the shape needs more bytes than the memory has. It is not `Copy`, and a
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

            /// The device address, a plain number: it cannot be turned back into a tensor without `unsafe`
            /// ([`DevicePtr::from_raw`]).
            pub fn addr(&self) -> u64 {
                self.arg.ptr
            }

            /// The shape.
            pub fn shape(&self) -> &'a [usize] {
                self.arg.shape
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
    /// tensor's elements, as `T` says. The tensor borrows the slice (so it is gone before the slice is) and a launch gives the slice
    /// to cudarc as a read, on the stream it launches on.
    pub fn from_cuda<S: CudaRead + 'a>(slice: &'a S, shape: &'a [usize]) -> Result<Self> {
        let region = sealed::Memory::region(slice);
        check_fits(shape, T::DTYPE, region.len_bytes)?;
        let arg = TensorArg {
            ptr: region.addr,
            shape,
            dtype: T::DTYPE,
            device: region.ordinal,
            context: Some(region.context),
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
        let region = sealed::Memory::region(&*slice);
        check_fits(shape, T::DTYPE, region.len_bytes)?;
        let arg = TensorArg {
            ptr: region.addr,
            shape,
            dtype: T::DTYPE,
            device: region.ordinal,
            context: Some(region.context),
            access: Access::InOut,
            backing: Backing::Write(slice),
        };
        Ok(Self { arg, _marker: PhantomData })
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
