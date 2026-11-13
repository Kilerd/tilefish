//! The cudarc memory a tensor can be made of, and how a launch hands it back to cudarc.
//!
//! cudarc tracks the use of its slices with events: a launch that reads or writes a `CudaSlice` (or a view of one) must be given
//! to cudarc's own launch builder *on the stream that launches*, which then orders the launch after the slice's earlier writes (and,
//! for a write, reads), and records the launch on the slice, so that dropping the slice (`cuMemFreeAsync` on the stream that
//! allocated it) waits for it. A tensor made with `TensorRef::from_cuda` or `TensorMut::from_cuda` therefore holds the borrow of
//! the slice, not a copy of its address: [`Kernel::launch`](crate::Kernel::launch) pushes the slice to the builder it launches
//! with, so the stream the tensor was made for and the stream of the launch cannot disagree (there is no stream in `from_cuda`),
//! and the tensor cannot outlive the slice (the borrow). Nothing a user can hold is an address that could be given back
//! without cudarc's tracking.
//!
//! What a tensor can be made of is a closed set, [`CudaRead`] (`CudaSlice`, `CudaView`) and [`CudaWrite`] (`CudaSlice`,
//! `CudaViewMut`): the two traits are sealed, because their implementations say what a launch reads and writes and where, and an
//! implementation of cudarc's own `DevicePtr` trait by a user type, which can name any address and length, is not one of them.

use cudarc::driver::{CudaSlice, CudaView, CudaViewMut, DevicePtr as _, DeviceSlice as _, LaunchArgs, PushKernelArg};

pub(crate) mod sealed {
    use super::LaunchArgs;

    /// Where a piece of cudarc memory is: what a launch checks a tensor of it against.
    #[derive(Debug, Clone, Copy)]
    pub struct Region {
        pub addr: u64,
        pub len_bytes: u64,
        pub ordinal: i32,
        /// The address of the `CUcontext` the memory was allocated in.
        pub context: usize,
    }

    /// Memory that cudarc owns and tracks.
    pub trait Memory {
        fn region(&self) -> Region;
    }

    /// Memory a launch reads.
    pub trait Read: Memory {
        /// Hand the memory to `launch` as a read, on the stream `launch` is for.
        fn push<'b>(&'b self, launch: &mut LaunchArgs<'b>);
    }

    /// Memory a launch writes.
    pub trait Write: Memory {
        /// Hand the memory to `launch` as a write, on the stream `launch` is for.
        fn push<'b>(&'b mut self, launch: &mut LaunchArgs<'b>);
    }
}

use sealed::Region;

/// Memory of cudarc's a tensor can be read from: a [`CudaSlice`] or a [`CudaView`]. Sealed.
pub trait CudaRead: sealed::Read {}
impl<S: sealed::Read> CudaRead for S {}

/// Memory of cudarc's a tensor can be written to: a [`CudaSlice`] or a [`CudaViewMut`]. Sealed.
pub trait CudaWrite: sealed::Write {}
impl<S: sealed::Write> CudaWrite for S {}

/// The address of cudarc memory, which cudarc only gives with a stream: its own, so that nothing is ordered against another stream
/// (the guard records a read on the slice's own stream when it drops, which only makes a later free wait for a point that has passed).
macro_rules! region_of {
    ($self:ident, $ordinal:expr, $context:expr) => {{
        let (addr, _guard) = $self.device_ptr($self.stream());
        Region { addr, len_bytes: $self.num_bytes() as u64, ordinal: $ordinal, context: $context }
    }};
}

impl<E> sealed::Memory for CudaSlice<E> {
    fn region(&self) -> Region {
        region_of!(self, self.ordinal() as i32, self.context().cu_ctx() as usize)
    }
}

impl<E> sealed::Memory for CudaView<'_, E> {
    fn region(&self) -> Region {
        region_of!(self, self.stream().context().ordinal() as i32, self.stream().context().cu_ctx() as usize)
    }
}

impl<E> sealed::Memory for CudaViewMut<'_, E> {
    fn region(&self) -> Region {
        region_of!(self, self.stream().context().ordinal() as i32, self.stream().context().cu_ctx() as usize)
    }
}

impl<E> sealed::Read for CudaSlice<E> {
    fn push<'b>(&'b self, launch: &mut LaunchArgs<'b>) {
        launch.arg(self);
    }
}

impl<E> sealed::Read for CudaView<'_, E> {
    fn push<'b>(&'b self, launch: &mut LaunchArgs<'b>) {
        launch.arg(self);
    }
}

impl<E> sealed::Write for CudaSlice<E> {
    fn push<'b>(&'b mut self, launch: &mut LaunchArgs<'b>) {
        launch.arg(self);
    }
}

impl<E> sealed::Write for CudaViewMut<'_, E> {
    fn push<'b>(&'b mut self, launch: &mut LaunchArgs<'b>) {
        launch.arg(self);
    }
}
