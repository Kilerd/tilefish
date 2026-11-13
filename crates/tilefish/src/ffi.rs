//! The calls that enqueue a launch, and the second place where the host takes on an obligation.
//!
//! The CUDA driver is cudarc's safe layer: contexts, modules, functions, attributes and streams are cudarc's, and so is the launch
//! builder that [`Kernel::launch`] goes through; [`Kernel::launch_raw`] uses cudarc's `result::launch_kernel` for a stream that
//! cudarc does not own. The one call to `cudarc::driver::sys` in this crate is the probe of [`driver_available`]. The `unsafe`
//! code of the crate is in four items here (`enqueue`, [`Kernel::launch_raw`], the restore of the current context, the probe) and one
//! in `region` ([`crate::DevicePtr::from_raw`]), each with the reason it is sound beside it.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg, result, sys};

use crate::error::{Error, Result};
use crate::kernel::Kernel;
use crate::plan::{MAX_PARAMS, NOT_A_TENSOR, Prepared};
use crate::tensor::{Backing, TensorArg};

#[cfg(target_endian = "big")]
compile_error!("tilefish packs a 32-bit kernel argument into the low bytes of a 64-bit slot: little-endian hosts only");

/// Whether the CUDA driver library (`libcuda`) can be loaded on this machine.
///
/// cudarc loads the driver when it is first called, and **`CudaContext::new` panics inside cudarc when there is no driver**: this
/// crate builds, and its tests that need no GPU run, on a machine without one, so a program that may run on such a machine asks
/// first.
///
/// The probe opens the library (`dlopen`) and keeps it open when it is there. Opening a library runs its initializers, so on a
/// machine that has one this has the effect of loading the driver into the process, as the first call of cudarc would; it calls no
/// function of the driver.
#[allow(unsafe_code)]
pub fn driver_available() -> bool {
    // SAFETY: `is_culib_present` only tries to open the library by its usual names (and keeps it open when it is there): loading
    // the driver, which runs its initializers as any use of it does, and calling no function in it.
    unsafe { sys::is_culib_present() }
}

/// Enqueue `function` on `stream` with the arguments of `prepared` in the order of the ABI of the manifest: a tensor's pointer, an
/// `i32`, then the two scratch pointers, which are null.
///
/// This is cudarc's launch builder (`CudaStream::launch_builder`, `.arg`, `.launch`): it binds the stream's context to the thread
/// and calls `cuLaunchKernel`, which copies the arguments before it returns. A tensor that is cudarc's own memory is given to the
/// builder as the slice or view itself, which makes cudarc order the launch on `stream` after the slice's earlier writes (and reads,
/// for a write) and record the launch on the slice's `read` or `write` event. A tensor that is a region the host vouched for is given
/// as its address, and nothing tracks it. A scalar is read through the first four bytes of its slot (little-endian, which the crate
/// insists on).
///
/// What this adds to cudarc's tracking, which keeps one `read` and one `write` event per slice:
///
/// * a launch of slices is refused while `stream` (or a stream a slice is freed on) is capturing a CUDA graph, whatever mode cudarc's
///   context is in: a graph would hold the slices' addresses with nothing to keep them alive ([`refuse_capture`]);
/// * before the kernel, for a slice that the kernel writes and whose owning stream (the one it is freed on) is not `stream`, `stream`
///   waits for everything enqueued on the owning stream so far. The owning stream has waited for every earlier launch that used the
///   slice from another stream (next item), so the writer is ordered after all earlier readers and writers, not only the last one that
///   cudarc recorded;
/// * after the kernel, for every slice whose owning stream is not `stream`, the owning stream waits for an event recorded right after
///   the kernel (`record_kernel_launch`, timing disabled): whatever is enqueued on the owning stream afterwards (the free, a use) runs
///   after this kernel, however many other launches read the slice. Nothing is waited for when every slice is on `stream` (stream
///   order already says it);
/// * an error after the kernel may have been enqueued does not leave a slice's owner unordered ([`settle`]).
#[allow(unsafe_code)]
pub(crate) fn enqueue(
    function: &CudaFunction,
    stream: &CudaStream,
    prepared: &Prepared,
    block: [u32; 3],
    shared: u32,
    tensors: &mut [TensorArg<'_>],
) -> Result<()> {
    refuse_capture(stream, tensors)?;
    let across = |t: &TensorArg<'_>| t.owner.as_ref().is_some_and(|owner| owner.cu_stream() != stream.cu_stream());
    let owners = distinct_owners(tensors.iter().filter(|t| across(t)));
    // before the kernel: a writer waits for what is on the owning stream (see above); nothing is enqueued for the kernel yet, so an
    // error here is a launch that did not happen
    for owner in distinct_owners(tensors.iter().filter(|t| across(t) && matches!(t.backing, Backing::Write(_)))) {
        let behind = owner.record_event(None).map_err(Error::driver("cuEventRecord"))?;
        stream.wait(&behind).map_err(Error::driver("cuStreamWaitEvent"))?;
    }
    let mut launch = stream.launch_builder(function);
    if !owners.is_empty() {
        launch.record_kernel_launch(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING);
    }
    let mut pushed = tensors.iter_mut();
    for (slot, value) in prepared.params().iter().enumerate() {
        if prepared.tensor_of[slot] == NOT_A_TENSOR {
            launch.arg(value);
            continue;
        }
        // the tensors are in the order of the slots: `Plan::new` numbers them as it meets them
        let Some(tensor) = pushed.next() else {
            return Err(Error::Launch("fewer tensors than the ABI has".to_owned()));
        };
        match &mut tensor.backing {
            Backing::Raw => {
                launch.arg(value);
            }
            Backing::Read(memory) => {
                let memory: &dyn crate::memory::sealed::Read = *memory;
                memory.push(&mut launch);
            }
            Backing::Write(memory) => {
                let memory: &mut dyn crate::memory::sealed::Write = &mut **memory;
                memory.push(&mut launch);
            }
        }
    }
    let grid = prepared.grid;
    let config = LaunchConfig {
        grid_dim: (grid[0], grid[1], grid[2]),
        block_dim: (block[0], block[1], block[2]),
        shared_mem_bytes: shared,
    };
    // SAFETY: cudarc asks the caller to vouch for the arguments and for what the kernel does with them. The arguments are the slots of
    // a `Prepared`, made by `Plan::prepare` from the manifest's ABI (a pointer, an `i32`, two null pointers for the binary's scratch,
    // in the order of its parameters), and the cubin is the one whose digests the manifest pinned. The memory the pointers name is
    // what the tensors borrow: a cudarc slice that the tensor holds the borrow of, for the whole call, and that this launch hands to
    // cudarc on `stream` (so it is ordered and recorded there, and the slice's own stream is made to wait for the kernel below), or
    // a region whose lifetime and contents the host vouched for (`DevicePtr::from_raw`); the tensors were checked to fit their memory,
    // not to overlap where one is written, to be in this kernel's context and device, to be writable where the kernel writes, and to
    // be aligned as the binary was compiled; the grid was checked against the device's limits and the shared memory against its
    // opt-in maximum when the kernel was loaded.
    let launched = unsafe { launch.launch(config) }.map_err(Error::driver("cuLaunchKernel"));
    settle(
        launched.map(|events| events.map(|(_, end)| end)),
        &owners,
        |owner, end| owner.wait(end).map_err(Error::driver("cuStreamWaitEvent")),
        || {
            // best effort: if this fails too, there is nothing more to do
            let _ = stream.synchronize();
        },
    )
}

/// The owners of `tensors` (the streams their memory is freed on), each once.
fn distinct_owners<'t, 'a: 't>(tensors: impl Iterator<Item = &'t TensorArg<'a>>) -> Vec<Arc<CudaStream>> {
    let mut owners: Vec<Arc<CudaStream>> = Vec::new();
    for owner in tensors.filter_map(|t| t.owner.as_ref()) {
        if !owners.iter().any(|seen| seen.cu_stream() == owner.cu_stream()) {
            owners.push(Arc::clone(owner));
        }
    }
    owners
}

/// What follows cudarc's `launch`, whose result is `launched` (the event recorded after the kernel, when it was asked for): each of
/// `owners`, the streams that slices of the launch are freed on and that are not the launch stream, waits for that event.
///
/// A call of cudarc's `launch` that returns an error may have enqueued the kernel already (the error can be that of the event
/// recorded after it, or of the records on the slices), and then there is no event to wait for; and the wait of one owner may fail
/// while another's does not. In either case the kernel may be running with nothing ordering the free of a slice behind it, so every
/// owner is still asked to wait, and then `synchronize` blocks the host until the launch stream has finished everything it has,
/// the kernel included, before the first error is returned: whatever the host enqueues on an owner after that is ordered after it.
/// With no `owners` (every slice is on the launch stream) stream order says it and the result is `launched`.
///
/// An `Err` from a launch is therefore not a promise that nothing was launched: a precondition that fails (an `Error::Launch` or a
/// precondition error) enqueues nothing, a driver error may have.
fn settle<Owner, Event>(
    launched: Result<Option<Event>>,
    owners: &[Owner],
    mut wait: impl FnMut(&Owner, &Event) -> Result<()>,
    synchronize: impl FnOnce(),
) -> Result<()> {
    if owners.is_empty() {
        return launched.map(drop);
    }
    let mut first_error = None;
    match launched {
        Ok(Some(end)) => {
            for owner in owners {
                if let Err(error) = wait(owner, &end) {
                    first_error.get_or_insert(error);
                }
            }
        }
        Ok(None) => {
            first_error = Some(Error::Launch("the driver gave no event for the end of the launch".to_owned()));
        }
        Err(error) => first_error = Some(error),
    }
    match first_error {
        None => Ok(()),
        Some(error) => {
            synchronize();
            Err(error)
        }
    }
}

/// Refuse a launch of tensors made of cudarc slices while `stream`, or a stream a slice is freed on, is capturing a CUDA graph.
///
/// A graph holds the addresses of the memory its kernels use, and nothing that keeps that memory alive: `CudaGraph::launch` is safe
/// and runs the kernel again with those addresses wherever the graph is launched, after the slices are freed and their memory is
/// used by something else. Whether a launch of slices is refused during a capture is not something cudarc's tracking decides: it
/// waits on events when its context has more than one stream and tracks events, and then the driver refuses the launch on a capturing
/// stream (`CUDA_ERROR_STREAM_CAPTURE_ISOLATION`), but a context that never made a stream of its own (`per_thread_stream`,
/// `default_stream`) has no such events, and the kernel would be captured. So the capture is looked for here, in every mode. Regions
/// the host vouched for (`DevicePtr::from_raw`) are not looked at: capture is what they are for, and their contract says that the
/// graph is not replayed after the memory is gone.
///
/// The check is made when the call is: a capture begun on `stream` by another thread while this call runs is not excluded by this
/// crate (CUDA does not define the order of a capture and a launch on one stream from two threads either).
fn refuse_capture(stream: &CudaStream, tensors: &[TensorArg<'_>]) -> Result<()> {
    if tensors.iter().all(|t| matches!(t.backing, Backing::Raw)) {
        return Ok(());
    }
    let owners = distinct_owners(tensors.iter());
    for candidate in std::iter::once(stream).chain(owners.iter().map(|owner| &**owner)) {
        let status = candidate.capture_status().map_err(Error::driver("cuStreamIsCapturing"))?;
        if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err(Error::Launch(CAPTURE_REFUSED.to_owned()));
        }
    }
    Ok(())
}

/// The message of a launch of slices that is refused because a stream is capturing.
const CAPTURE_REFUSED: &str = "a tensor is made of a cudarc slice and the stream is capturing a CUDA graph: a graph would hold the \
    slice's address and nothing that keeps the slice alive; capture tensors made of regions (`unsafe DevicePtr::from_raw`), \
    with `launch_raw`";

/// A context made current on this thread for as long as the guard lives, with what was current before put back when it is
/// dropped (on an error as well).
struct Current {
    changed: bool,
    previous: Option<sys::CUcontext>,
}

impl Current {
    fn make(context: &CudaContext) -> Result<Self> {
        let previous = result::ctx::get_current().map_err(Error::driver("cuCtxGetCurrent"))?;
        if previous == Some(context.cu_ctx()) {
            return Ok(Self { changed: false, previous });
        }
        context.bind_to_thread().map_err(Error::driver("cuCtxSetCurrent"))?;
        Ok(Self { changed: true, previous })
    }
}

impl Drop for Current {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        if self.changed {
            // SAFETY: `previous` was this thread's current context when the guard was made (a null pointer: there was none), and
            // setting it again is what the thread had. A failure is ignored: there is nothing to do about it in a destructor.
            let _ = unsafe { result::ctx::set_current(self.previous.unwrap_or(std::ptr::null_mut())) };
        }
    }
}

impl Kernel {
    /// Check the call and launch the kernel on a raw `CUstream`.
    ///
    /// What [`Kernel::launch`] does for a stream that cudarc owns, for a handle that nothing owns: another library's stream, or the
    /// stream that a CUDA graph is being captured on (the call is captured like any kernel node). The kernel's context is made
    /// current on this thread for the call, and what was current is put back afterwards (also when the launch fails).
    ///
    /// **This is how a CUDA graph is captured**, with tensors made of regions ([`DevicePtr::from_raw`](crate::DevicePtr::from_raw)):
    /// a graph holds the addresses it captured and nothing that keeps the memory alive, so it is the host's (see the contract of
    /// `from_raw`) that the graph is not replayed after the memory is gone.
    ///
    /// A tensor made of a cudarc slice is used by its address here: cudarc's event tracking is for its own streams, so nothing
    /// orders this launch against other streams' use of the memory, or a later free of it against this launch. That is the caller's,
    /// as it is for a region. A launch with such a tensor on a stream that is capturing is refused (an `Error::Launch`), as in
    /// [`Kernel::launch`]: a captured launch of a slice would outlive the borrow that keeps the slice alive.
    ///
    /// # Safety
    ///
    /// * `stream` must be a valid `CUstream` of the CUDA context the kernel was loaded in ([`Kernel::context`]): a handle from another
    ///   context, a destroyed stream or a pointer that is not a stream is undefined behavior in the driver, and it must stay valid
    ///   until this returns;
    /// * the tensors' memory must stay allocated until the launch has completed on `stream`, which this call does not wait for, and
    ///   it must be ordered against the other streams that use it (nothing here is).
    #[allow(unsafe_code)]
    pub unsafe fn launch_raw(&self, stream: sys::CUstream, tensors: &[TensorArg<'_>]) -> Result<()> {
        let mut prepared = self.check(tensors)?;
        let _current = Current::make(self.context())?;
        if tensors.iter().any(|t| !matches!(t.backing, Backing::Raw)) {
            // SAFETY: `stream` is a valid `CUstream`, as this function's contract says; the call only reads its capture status.
            let status =
                unsafe { result::stream::is_capturing(stream) }.map_err(Error::driver("cuStreamIsCapturing"))?;
            if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
                return Err(Error::Launch(CAPTURE_REFUSED.to_owned()));
            }
        }
        let mut params: [*mut c_void; MAX_PARAMS] = [std::ptr::null_mut(); MAX_PARAMS];
        let count = prepared.count;
        for (slot, value) in params.iter_mut().zip(prepared.values.iter_mut()).take(count) {
            *slot = std::ptr::from_mut(value).cast::<c_void>();
        }
        let (grid, block) = (prepared.grid, self.plan().block());
        // SAFETY: `params[..count]` point at the live 64-bit slots of `prepared`, which the driver reads before it returns, in the
        // order of the manifest's ABI; the rest is as in `enqueue`, and the stream is the caller's, as the contract above says.
        unsafe {
            result::launch_kernel(
                self.function().cu_function(),
                (grid[0], grid[1], grid[2]),
                (block[0], block[1], block[2]),
                self.plan().shared(),
                stream,
                &mut params[..count],
            )
        }
        .map_err(Error::driver("cuLaunchKernel"))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use super::*;

    fn boom(what: &str) -> Error {
        Error::Launch(what.to_owned())
    }

    /// `settle` with owners numbered 1, 2, 3, of which `failing` fail to wait: what they waited for, how often the host was
    /// synchronized, and the result.
    fn run(
        launched: Result<Option<&'static str>>,
        owners: &[u32],
        failing: &[u32],
    ) -> (Vec<(u32, &'static str)>, u32, Result<()>) {
        let waited = RefCell::new(Vec::new());
        let synchronized = Cell::new(0);
        let result = settle(
            launched,
            owners,
            |owner, end| {
                waited.borrow_mut().push((*owner, *end));
                if failing.contains(owner) { Err(boom(&format!("wait {owner}"))) } else { Ok(()) }
            },
            || synchronized.set(synchronized.get() + 1),
        );
        (waited.into_inner(), synchronized.get(), result)
    }

    #[test]
    fn every_owner_waits_for_the_end_of_the_kernel_and_nothing_else_is_done() {
        let (waited, synchronized, result) = run(Ok(Some("end")), &[1, 2, 3], &[]);
        assert_eq!(waited, [(1, "end"), (2, "end"), (3, "end")]);
        assert_eq!(synchronized, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn a_failing_owner_does_not_stop_the_others_and_the_host_waits_for_the_stream() {
        let (waited, synchronized, result) = run(Ok(Some("end")), &[1, 2, 3], &[1, 3]);
        assert_eq!(waited, [(1, "end"), (2, "end"), (3, "end")], "every owner was asked, although the first failed");
        assert_eq!(synchronized, 1, "the launch stream is synchronized once, so that no free is left unordered");
        assert_eq!(result.unwrap_err().to_string(), boom("wait 1").to_string(), "the first error is the one returned");
    }

    #[test]
    fn a_launch_that_failed_after_the_kernel_may_have_been_enqueued_has_no_event_and_synchronizes() {
        let (waited, synchronized, result) = run(Err(boom("end event")), &[1, 2], &[]);
        assert!(waited.is_empty(), "there is no event to wait for");
        assert_eq!(synchronized, 1);
        assert_eq!(result.unwrap_err().to_string(), boom("end event").to_string());
    }

    #[test]
    fn a_launch_that_gave_no_event_is_an_error_and_synchronizes() {
        let (waited, synchronized, result) = run(Ok(None), &[1], &[]);
        assert!(waited.is_empty());
        assert_eq!(synchronized, 1);
        assert!(result.is_err());
    }

    #[test]
    fn with_every_slice_on_the_launch_stream_the_result_is_the_launch_and_nothing_is_synchronized() {
        let (waited, synchronized, result) = run(Ok(None), &[], &[]);
        assert!(waited.is_empty() && result.is_ok());
        assert_eq!(synchronized, 0);
        let (_, synchronized, result) = run(Err(boom("launch")), &[], &[]);
        assert_eq!(synchronized, 0, "stream order orders the free after the kernel, enqueued or not");
        assert_eq!(result.unwrap_err().to_string(), boom("launch").to_string());
    }
}
