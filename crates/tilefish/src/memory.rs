//! The cudarc memory a tensor can be made of, and how a launch hands it back to cudarc.
//!
//! cudarc tracks the use of its slices with events: a launch that reads or writes a `CudaSlice` (or a view of one) must be given
//! to cudarc's own launch builder *on the stream that launches*, which then orders the launch after the slice's earlier writes (and,
//! for a write, reads) and records the launch on the slice (one `read` and one `write` event per slice), and dropping the slice
//! (`cuMemFreeAsync` on the stream that allocated it) waits for those events. A tensor made with `TensorRef::from_cuda` or
//! `TensorMut::from_cuda` therefore holds the borrow of the slice, not a copy of its address: [`Kernel::launch`](crate::Kernel::launch)
//! pushes the slice to the builder it launches with, so there is no stream in `from_cuda` for the launch stream to disagree with, and
//! the tensor cannot outlive the slice (the borrow).
//!
//! What this crate does about cudarc's tracking, exactly:
//!
//! * **Making a tensor records nothing.** The address of a slice is only given by cudarc's `device_ptr(stream)`, which waits (on the
//!   slice's own stream) for the slice's last write, and returns a guard that, dropped, *records the use on the slice's `read` event*.
//!   `cuEventRecord` overwrites: a record made while building a tensor replaces the record of a launch that is still queued on
//!   another stream, and the free of the slice would then wait for the wrong point. So the guard is forgotten, not dropped: the
//!   only thing left of `device_ptr` is a wait for earlier writes, which adds ordering and removes none (so a tensor made over a
//!   slice that a launch on another stream is still writing makes the slice's own stream wait for that launch, as cudarc's own
//!   `device_ptr` does). Building a tensor, and throwing it away, or having it refused, changes nothing about when the slice may be
//!   freed or written.
//! * **A launch is tracked by cudarc on the launch stream** (the slices are given to its builder), which keeps cudarc's own ordering
//!   of the launch after the slice's earlier writes (and reads, for a write) on any stream.
//! * **And by this crate, on the stream that owns the slice**: cudarc keeps one `read` event per slice, so two launches that read a
//!   slice on two streams leave only the last one for the free to wait for. After the launch, for every slice whose owning stream (the
//!   one it will be freed on) is not the launch stream, the owning stream waits for an event recorded right after the kernel. A free
//!   (or a use) on the owning stream after that is ordered after the kernel, however many other launches read the slice. This is
//!   not done for a region the host vouched for: nothing tracks those.
//!
//! What is *not* guaranteed: ordering between two launches on two different streams that both write, or one that writes and others
//! that read, beyond what cudarc's one-event-per-slice tracking gives (a writer waits for the last reader that was recorded, not for
//! all of them). Use one stream for work that must be ordered, or order streams yourself.
//!
//! What a tensor can be made of is a closed set, [`CudaRead`] (`CudaSlice`, `CudaView`) and [`CudaWrite`] (`CudaSlice`,
//! `CudaViewMut`): the two traits are sealed, because their implementations say what a launch reads and writes and where, and an
//! implementation of cudarc's own `DevicePtr` trait by a user type, which can name any address and length, is not one of them.

use std::sync::Arc;

use cudarc::driver::{
    CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr as _, DeviceSlice as _, LaunchArgs, PushKernelArg,
};

pub(crate) mod sealed {
    use std::sync::Arc;

    use super::{CudaStream, LaunchArgs};

    /// Where a piece of cudarc memory is: what a launch checks a tensor of it against.
    #[derive(Debug, Clone)]
    pub struct Region {
        pub addr: u64,
        pub len_bytes: u64,
        pub ordinal: i32,
        /// The address of the `CUcontext` the memory was allocated in.
        pub context: usize,
        /// The stream the memory was allocated on, and will be freed on.
        pub owner: Arc<CudaStream>,
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

/// The address of cudarc memory, which cudarc only gives with a stream: the slice's own. The guard it returns is forgotten, not
/// dropped: dropped, it records a read on the slice's single `read` event, which replaces the record of a launch that is queued on
/// another stream (see the module docs). What is left is the wait for the slice's last write, on the slice's own stream.
macro_rules! region_of {
    ($self:ident, $ordinal:expr, $context:expr) => {{
        let owner = Arc::clone($self.stream());
        let (addr, guard) = $self.device_ptr(&owner);
        std::mem::forget(guard);
        Region { addr, len_bytes: $self.num_bytes() as u64, ordinal: $ordinal, context: $context, owner }
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
