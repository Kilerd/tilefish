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
//!   of the launch after the slice's earlier writes (and reads, for a write) on any stream, in a context that tracks (below).
//! * **And by this crate, on the stream that owns the slice**: cudarc keeps one `read` event per slice, so two launches that read a
//!   slice on two streams leave only the last one for the free to wait for. After the launch, for every slice whose owning stream (the
//!   one it will be freed on) is not the launch stream, the owning stream waits for an event recorded right after the kernel. A free
//!   (or a use) on the owning stream after that is ordered after the kernel, however many other launches read the slice. This is
//!   not done for a region the host vouched for: nothing tracks those. It is done in every context, whether cudarc tracks it or not.
//! * **A writer waits for everything before it.** cudarc's writer waits for the slice's last `read` and last `write` record, which is
//!   one launch: a writer launched after two readers on two streams would wait for the later of them only. So, before the kernel, for
//!   every slice that the kernel writes and whose owning stream is not the launch stream, the launch stream waits for an event recorded
//!   on the owning stream at that moment. The owning stream has waited for the end of every earlier launch of this crate that used the
//!   slice (previous item), so the writer runs after all of them, readers and writers, on whatever streams.
//! * **A launch of slices is refused while a stream is capturing a CUDA graph** (the launch stream, or a stream a slice is freed on):
//!   a graph holds addresses and nothing that keeps the slice alive, and `CudaGraph::launch` is safe. cudarc alone does not refuse
//!   it: a context that never made a stream of its own (it uses `per_thread_stream`, or the default stream) has no events to wait on,
//!   and the kernel would be captured. Capture uses regions (`unsafe` [`DevicePtr::from_raw`](crate::DevicePtr::from_raw)) and
//!   [`Kernel::launch_raw`](crate::Kernel::launch_raw).
//! * **An error from a launch** is not a promise that the kernel was not enqueued: a driver error can come after it (cudarc's launch
//!   records an event after the kernel, and on the slices). When a slice's owning stream is not the launch stream, the crate then
//!   makes every owner wait for what it can, and blocks the host until the launch stream is idle (best effort), before it returns the
//!   error, so no free of a slice is left unordered behind a kernel that may be running.
//!
//! What is *not* guaranteed: the ordering of a reader after a writer of the slice on another stream is cudarc's, which tracks it only in
//! a context that has made a stream of its own (`CudaContext::new_stream`) and has event tracking on, and it is the tracking of one
//! `write` event, which each writer's wait for the one before it makes enough; the uses of a slice that are not launches of this crate
//! (cudarc's copies, another library) are cudarc's or the host's own; and calls from two threads are as ordered as the host orders
//! them. Do not turn cudarc's event tracking off (`unsafe`) to get a graph: use regions.
//!
//! What a tensor can be made of is a closed set, [`CudaRead`] (`CudaSlice`, `CudaView`) and [`CudaWrite`] (`CudaSlice`,
//! `CudaViewMut`): the two traits are sealed, because their implementations say what a launch reads and writes and where, and an
//! implementation of cudarc's own `DevicePtr` trait by a user type, which can name any address and length, is not one of them.

use std::sync::Arc;

use cudarc::driver::{
    CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr as _, DevicePtrMut as _, DeviceSlice as _, LaunchArgs,
    PushKernelArg, SyncOnDrop,
};

pub(crate) mod sealed {
    use std::sync::Arc;

    use cudarc::driver::SyncOnDrop;

    use super::{CudaStream, CudaView, CudaViewMut, LaunchArgs};

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

        /// A view of the bytes of the memory from byte `from` on (`None` when there are none): for a view of a tensor whose first element
        /// is not the first byte of the slice, which `launch` hands to cudarc in place of the slice, so that the kernel is given the
        /// address of the first element and cudarc still orders and records the use of the slice.
        fn bytes_from(&self, from: u64) -> Option<CudaView<'_, u8>>;

        /// What cudarc's `device_ptr(stream)` does besides giving the address: `stream` waits for the memory's last write, and the guard,
        /// dropped, records the read on the memory's `read` event on `stream`. (For the calls that are not a kernel launch.)
        fn sync_read<'b>(&'b self, stream: &'b CudaStream) -> SyncOnDrop<'b>;
    }

    /// Memory a launch writes.
    pub trait Write: Memory {
        /// Hand the memory to `launch` as a write, on the stream `launch` is for.
        fn push<'b>(&'b mut self, launch: &mut LaunchArgs<'b>);

        /// As [`Read::bytes_from`], for memory that is written.
        fn bytes_from_mut(&mut self, from: u64) -> Option<CudaViewMut<'_, u8>>;

        /// What cudarc's `device_ptr_mut(stream)` does besides giving the address: `stream` waits for the memory's last read and write, and
        /// the guard, dropped, records the write on the memory's `write` event on `stream`.
        fn sync_write<'b>(&'b mut self, stream: &'b CudaStream) -> SyncOnDrop<'b>;

        /// All the bytes of the memory, as a view that is read (it shares the events of the memory: cudarc orders a use of it as a use of
        /// the memory).
        fn bytes(&self) -> Option<CudaView<'_, u8>>;

        /// All the bytes of the memory, as a view that may be written (it shares the events of the memory).
        fn bytes_mut(&mut self) -> Option<CudaViewMut<'_, u8>>;
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

    fn bytes_from(&self, from: u64) -> Option<CudaView<'_, u8>> {
        crate::ffi::bytes_of_slice(self, from)
    }

    fn sync_read<'b>(&'b self, stream: &'b CudaStream) -> SyncOnDrop<'b> {
        self.device_ptr(stream).1
    }
}

impl<E> sealed::Read for CudaView<'_, E> {
    fn push<'b>(&'b self, launch: &mut LaunchArgs<'b>) {
        launch.arg(self);
    }

    fn bytes_from(&self, from: u64) -> Option<CudaView<'_, u8>> {
        crate::ffi::bytes_of_view(self, from)
    }

    fn sync_read<'b>(&'b self, stream: &'b CudaStream) -> SyncOnDrop<'b> {
        self.device_ptr(stream).1
    }
}

impl<E> sealed::Write for CudaSlice<E> {
    fn push<'b>(&'b mut self, launch: &mut LaunchArgs<'b>) {
        launch.arg(self);
    }

    fn bytes_from_mut(&mut self, from: u64) -> Option<CudaViewMut<'_, u8>> {
        crate::ffi::bytes_of_slice_mut(self, from)
    }

    fn sync_write<'b>(&'b mut self, stream: &'b CudaStream) -> SyncOnDrop<'b> {
        self.device_ptr_mut(stream).1
    }

    fn bytes(&self) -> Option<CudaView<'_, u8>> {
        crate::ffi::bytes_of_slice(self, 0)
    }

    fn bytes_mut(&mut self) -> Option<CudaViewMut<'_, u8>> {
        crate::ffi::all_bytes_of_slice_mut(self)
    }
}

impl<E> sealed::Write for CudaViewMut<'_, E> {
    fn push<'b>(&'b mut self, launch: &mut LaunchArgs<'b>) {
        launch.arg(self);
    }

    fn bytes_from_mut(&mut self, from: u64) -> Option<CudaViewMut<'_, u8>> {
        crate::ffi::bytes_of_view_mut(self, from)
    }

    fn sync_write<'b>(&'b mut self, stream: &'b CudaStream) -> SyncOnDrop<'b> {
        self.device_ptr_mut(stream).1
    }

    fn bytes(&self) -> Option<CudaView<'_, u8>> {
        crate::ffi::all_bytes_of_view_mut(self)
    }

    fn bytes_mut(&mut self) -> Option<CudaViewMut<'_, u8>> {
        crate::ffi::all_bytes_of_view_mut_mut(self)
    }
}
