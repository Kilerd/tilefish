//! The calls that enqueue a launch, and the second place where the host takes on an obligation.
//!
//! The CUDA driver is cudarc's safe layer: contexts, modules, functions, attributes and streams are cudarc's, and so is the launch
//! builder that [`Kernel::launch`] goes through; [`Kernel::launch_raw`] uses cudarc's `result::launch_kernel` for a stream that
//! cudarc does not own. The one call to `cudarc::driver::sys` in this crate is the probe of [`driver_available`]. The `unsafe`
//! code of the crate is in four items here (`enqueue`, [`Kernel::launch_raw`], the restore of the current context, the probe) and one
//! in `region` ([`crate::DevicePtr::from_raw`]), each with the reason it is sound beside it.

use std::ffi::c_void;

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
/// cudarc keeps one `read` event per slice, so with two launches that read a slice on two streams the free of the slice (on the stream
/// it was allocated on) waits only for the last one. So, when a slice's owning stream is not `stream`, the builder is asked for the
/// events around the kernel (`record_kernel_launch`, timing disabled) and, once the kernel is enqueued, the owning stream waits for
/// the one recorded after it: whatever is enqueued on the owning stream afterwards (the free, a use) runs after this kernel, however
/// many other launches read the slice. Nothing is waited for when every slice is on `stream` (stream order already says it).
#[allow(unsafe_code)]
pub(crate) fn enqueue(
    function: &CudaFunction,
    stream: &CudaStream,
    prepared: &Prepared,
    block: [u32; 3],
    shared: u32,
    tensors: &mut [TensorArg<'_>],
) -> Result<()> {
    let across = |t: &TensorArg<'_>| t.owner.as_ref().is_some_and(|owner| owner.cu_stream() != stream.cu_stream());
    let any_across = tensors.iter().any(across);
    let mut launch = stream.launch_builder(function);
    if any_across {
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
    let events = unsafe { launch.launch(config) }.map_err(Error::driver("cuLaunchKernel"))?;
    if let Some((_, end)) = events {
        for owner in tensors.iter().filter(|t| across(t)).filter_map(|t| t.owner.as_ref()) {
            owner.wait(&end).map_err(Error::driver("cuStreamWaitEvent"))?;
        }
    }
    Ok(())
}

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
    /// A tensor made of a cudarc slice is used by its address here: cudarc's event tracking is for its own streams, so nothing
    /// orders this launch against other streams' use of the memory, or a later free of it against this launch. That is the caller's,
    /// as it is for a region from [`DevicePtr::from_raw`](crate::DevicePtr::from_raw).
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
