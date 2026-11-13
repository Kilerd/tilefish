//! The two calls that enqueue a launch, and the second place where the host takes on an obligation.
//!
//! The CUDA driver is cudarc's safe layer: contexts, modules, functions, attributes and streams are cudarc's, and so is the launch
//! builder that [`Kernel::launch`] goes through; [`Kernel::launch_raw`] uses cudarc's `result::launch_kernel` for a stream that
//! cudarc does not own. There is no call to `cudarc::driver::sys` in this crate. The `unsafe` code of the crate is in two items
//! here (`enqueue`, [`Kernel::launch_raw`]) and three in `region`, each with the reason it is sound beside it.

use std::ffi::c_void;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg, result, sys};

use crate::error::{Error, Result};
use crate::kernel::Kernel;
use crate::plan::MAX_PARAMS;
use crate::tensor::TensorArg;

#[cfg(target_endian = "big")]
compile_error!("tilefish packs a 32-bit kernel argument into the low bytes of a 64-bit slot: little-endian hosts only");

/// Enqueue `function` on `stream` with `slots` as its arguments, one 64-bit slot each, in the order of the ABI of the manifest: a
/// pointer, an `i32` in the low four bytes, then the two scratch pointers, which are null.
///
/// This is cudarc's launch builder (`CudaStream::launch_builder`, `.arg`, `.launch`): it binds the stream's context to the thread
/// and calls `cuLaunchKernel`, which copies the arguments before it returns, so the slots need not outlive the call. A kernel
/// argument is a pointer to its value, so a slot that is an `i32` is read through its first four bytes (little-endian, which
/// the crate insists on).
#[allow(unsafe_code)]
pub(crate) fn enqueue(
    function: &CudaFunction,
    stream: &CudaStream,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    slots: &[u64],
) -> Result<()> {
    let mut launch = stream.launch_builder(function);
    for slot in slots {
        launch.arg(slot);
    }
    let config = LaunchConfig {
        grid_dim: (grid[0], grid[1], grid[2]),
        block_dim: (block[0], block[1], block[2]),
        shared_mem_bytes: shared,
    };
    // SAFETY: cudarc asks the caller to vouch for the arguments and for what the kernel does with them. The arguments are the
    // slots of a `Prepared`, made by `Plan::prepare` from the manifest's ABI (a pointer, an `i32`, two null pointers for the
    // binary's scratch, in the order of its parameters), and the cubin is the one whose digests the manifest pinned. The memory the
    // pointers name is what the tensors vouched for (`DevicePtr::from_raw`, or a cudarc slice that `from_cuda` borrowed), checked to
    // fit the shapes, not to overlap where one is written, and to be aligned as the binary was compiled; the grid was checked against
    // the device's limits and the shared memory against its opt-in maximum when the kernel was loaded.
    unsafe { launch.launch(config) }.map(drop).map_err(Error::driver("cuLaunchKernel"))
}

impl Kernel {
    /// Check the call and launch the kernel on a raw `CUstream`.
    ///
    /// What [`Kernel::launch`] does for a stream that cudarc owns, for a handle that nothing owns: another library's stream, or the
    /// stream that a CUDA graph is being captured on (the call is captured like any kernel node). The kernel's context is made
    /// current on this thread, as cudarc does for its own calls.
    ///
    /// # Safety
    ///
    /// * `stream` must be a valid `CUstream` of the CUDA context the kernel was loaded in ([`Kernel::context`]): a handle from another
    ///   context, a destroyed stream or a pointer that is not a stream is undefined behavior in the driver, and it must stay valid
    ///   until this returns;
    /// * the tensors keep the obligations of [`crate::DevicePtr::from_raw`]: their memory must stay allocated until the launch has completed
    ///   on `stream`, which this call does not wait for.
    #[allow(unsafe_code)]
    pub unsafe fn launch_raw(&self, stream: sys::CUstream, tensors: &[TensorArg<'_>]) -> Result<()> {
        let mut prepared = self.check(tensors)?;
        self.context().bind_to_thread().map_err(Error::driver("cuCtxSetCurrent"))?;
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
