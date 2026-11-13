//! The `unsafe` of a launch plan, each item with the reason it is sound: the regions its steps are checked with, the launches of the
//! checked steps, the launch of a captured graph on a stream it was not captured on, and the eager loading of a function.
//!
//! What decides that a step may be launched (the binding of the externals, the checks of every step before the first launch, the ordering of
//! the externals and of the arena against other streams, the lifetimes a captured graph is held to) is `crate::plan_run`, which has no
//! `unsafe`; it calls these.

use cudarc::driver::{CudaFunction, CudaGraph, CudaStream, LaunchConfig, PushKernelArg, result, sys};

use crate::error::{Error, Result};
use crate::plan_run::Ready;
use crate::region::DevicePtr;

/// The region `addr .. addr + len` of device `device`, for the checks of a step ([`crate::Plan::prepare`], `Gemm::check_call`) and nothing
/// else.
///
/// The region is made from an address the crate computed (an external's first byte, or the arena's address plus a buffer's offset) or
/// made up (the distinct addresses an instantiation checks the address-independent preconditions with). It is never handed to a launch: the
/// tensors made of it are given to the checks, which read their numbers (address, shape, strides, dtype) and return the grid and the
/// parameters to launch with; the launch is [`enqueue_steps`], whose own argument says why those addresses may be used.
#[allow(unsafe_code)]
pub(crate) fn check_region<'a>(addr: u64, len: u64, device: i32) -> DevicePtr<'a> {
    // SAFETY: the contract of `from_raw` is about launches made of the region; this one only describes numbers to the checks (see above).
    unsafe { DevicePtr::from_raw(addr, len, device) }
}

/// Enqueue the steps of a plan, checked, in order, on `stream` (which may be capturing: every call here is a launch, so it is captured as a
/// node; nothing is allocated, waited for or loaded).
///
/// A kernel step is cudarc's launch builder with the parameter values of its `Prepared` (addresses, scalars, fixed values, scratch slots);
/// a library step is the pinned cuBLASLt matmul of its `Call`, whose workspace is in the arena.
///
/// Why the addresses may be used: every one was checked by `Plan::prepare` (or `Gemm::check_call`) against a region of the memory it is in,
/// so no step reaches past an external's bytes or a buffer's region of the arena; the externals were checked to be on this device and in
/// this context, not to overlap the arena, and not to overlap each other where one is written; the arena is owned by the instantiation, which
/// every `PlanRun` and `CapturedPlan` that can launch it keeps alive; and the use is ordered by the caller: a launch through `ffi::ordered`
/// (the externals' slices borrowed for the call and handed to cudarc's tracking on `stream`, their owning streams and the arena's waiting for
/// the end of the plan), a capture with every external borrowed for as long as the `CapturedPlan` that holds the graph, whose replays are
/// ordered the same way.
#[allow(unsafe_code)]
pub(crate) fn enqueue_steps(stream: &CudaStream, steps: &[Ready<'_>]) -> Result<()> {
    for step in steps {
        match step {
            Ready::Kernel(kernel, prepared) => {
                let mut launch = stream.launch_builder(kernel.function());
                for value in prepared.params() {
                    launch.arg(value);
                }
                let (grid, block) = (prepared.grid, kernel.plan().block());
                let config = LaunchConfig {
                    grid_dim: (grid[0], grid[1], grid[2]),
                    block_dim: (block[0], block[1], block[2]),
                    shared_mem_bytes: kernel.plan().shared(),
                };
                // SAFETY: the arguments are the slots of a `Prepared` that `Plan::prepare` made from the manifest's ABI for this very kernel (a
                // pointer, a scalar or a fixed value in the width of its type, then a Triton kernel's null scratch slots), whose loaded function
                // was checked to take those parameter sizes; the grid was checked against the device and the shared memory when the kernel was
                // loaded; the addresses are as the doc comment of this function says.
                unsafe { launch.launch(config) }.map_err(Error::driver("cuLaunchKernel"))?;
            }
            Ready::Gemm(gemm, call) => {
                stream.context().bind_to_thread().map_err(Error::driver("cuCtxSetCurrent"))?;
                // SAFETY: `stream` is a cudarc stream of the GEMM's context (the run checks it), now current on this thread; the addresses of
                // `call` were checked by `Gemm::check_call` (shapes of the problem, bf16, alignment, the workspace in the arena with what the
                // algorithm needs) and are used as the doc comment of this function says.
                unsafe { gemm.run(stream.cu_stream(), call) }?;
            }
        }
    }
    Ok(())
}

/// Launch a captured graph on `stream`: cudarc's `CudaGraph::launch` when `stream` is the one it was captured on (`captured_on`), and
/// `cuGraphLaunch` on the handle of `stream` otherwise (cudarc launches a graph on its capture stream only).
#[allow(unsafe_code)]
pub(crate) fn launch_graph(graph: &CudaGraph, captured_on: &CudaStream, stream: &CudaStream) -> Result<()> {
    if stream.cu_stream() == captured_on.cu_stream() {
        return graph.launch().map_err(Error::driver("cuGraphLaunch"));
    }
    stream.context().bind_to_thread().map_err(Error::driver("cuCtxSetCurrent"))?;
    // SAFETY: the executable graph is owned by `graph`, which is borrowed for the call (so it is alive and not destroyed meanwhile, and the
    // `&mut` of the replay that calls this serializes the uses of it); `stream` is a live cudarc stream of the same context (checked by the
    // caller), current on this thread. What the graph does with its addresses is argued where it is captured (`enqueue_steps`).
    unsafe { result::graph::launch(graph.cu_graph_exec(), stream.cu_stream()) }.map_err(Error::driver("cuGraphLaunch"))
}

/// Load a function of a module now (`cuFuncLoad`), and not at its first launch: under lazy loading (`CUDA_MODULE_LOADING=LAZY`, the
/// default) the first launch loads the function, which would be inside a CUDA graph capture, and would take device memory after the plan
/// was instantiated. cudarc has no wrapper of `cuFuncLoad`.
#[allow(unsafe_code)]
pub(crate) fn load_function(function: &CudaFunction) -> Result<()> {
    // SAFETY: `function` is a live function of a loaded module (the kernel that holds it keeps the module and its context alive); the call
    // only loads its code. The function exists in every driver this crate runs on: `Manifest::kernel` has already called
    // `cuFuncGetParamInfo`, which came with it (CUDA 12.4).
    let status = unsafe { sys::cuFuncLoad(function.cu_function()) };
    status.result().map_err(Error::driver("cuFuncLoad"))
}
