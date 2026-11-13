//! Loading a binary into a CUDA context and launching it, through cudarc's safe layer.
//!
//! What Python's `_cuda.py` does through `cuda-python`: load the verified cubin bytes into a context
//! (`CudaContext::load_module(Ptx::from_binary(..))`), get the manifest's symbol (`CudaModule::load_function`), opt in to dynamic
//! shared memory above the default 48 KiB (`CudaFunction::set_attribute`), and launch with `kernelParams` laid out as the ABI says (the
//! arguments in declaration order: a pointer as 64 bits, a scalar as its `i32` or `i64`, a `fixed` argument as its value in its type;
//! then, for a Triton kernel, two null 64-bit scratch pointers, and for a prebuilt binary nothing). The loaded function must take the
//! parameter sizes of that ABI (`cuFuncGetParamInfo`).
//!
//! A kernel belongs to the context it was loaded in, which is passed explicitly ([`Manifest::kernel`] takes an
//! `&Arc<CudaContext>`) and kept alive by the kernel, with the module and the function: nothing relies on which context is current on
//! a thread. A load is checked against the device like Python's: the function's static shared memory (as the driver reports it) plus
//! the manifest's dynamic shared memory must fit the device's opt-in maximum per block. A launch is checked by [`Plan::prepare`],
//! and then against the device's maximum grid.
//!
//! A launch goes on the stream the caller gives (cudarc's `CudaStream`, checked to be of the kernel's context; or, for a handle that
//! nothing owns, `unsafe` [`Kernel::launch_raw`]). A CUDA graph is captured by launching tensors made of regions
//! ([`DevicePtr::from_raw`](crate::DevicePtr::from_raw)) on the capturing stream: the driver copies `kernelParams` at the call. A
//! launch of tensors made of cudarc slices on a capturing stream is refused.

use std::fmt;
use std::sync::{Arc, Weak};

use cudarc::driver::CudaFunction;
use cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute};
use cudarc::nvrtc::Ptx;

pub use cudarc::driver::sys::CUstream;
pub use cudarc::driver::{CudaContext, CudaStream};

use crate::error::{Error, Result};
use crate::ffi;
use crate::manifest::{Manifest, Param};
use crate::plan::{DeviceInfo, Plan, Prepared};
use crate::tensor::TensorArg;

/// Dynamic shared memory above this needs `cuFuncSetAttribute`.
const DEFAULT_SHARED_LIMIT: u32 = 48 * 1024;

/// A compute capability: `Arch::sm(103)` is `sm_103`, the `arch` of a manifest entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Arch(u32);

impl Arch {
    /// `sm_<capability>`, with the capability written as major * 10 + minor (`90`, `100`, `103`, `120`).
    pub const fn sm(capability: u32) -> Self {
        Self(capability)
    }

    /// Parse `"sm_103"`, `"sm103"` or `"103"`.
    pub fn parse(text: &str) -> Result<Self> {
        let digits = text.strip_prefix("sm_").or_else(|| text.strip_prefix("sm")).unwrap_or(text);
        match digits.parse::<u32>() {
            Ok(n) if (70..=999).contains(&n) => Ok(Self(n)),
            _ => Err(Error::Unsupported(format!("arch must look like 'sm_90' or 90, got {text:?}"))),
        }
    }

    /// The compute capability of the device of `ctx` (the one a binary is loaded for).
    pub fn of_context(ctx: &CudaContext) -> Result<Self> {
        let (major, minor) =
            ctx.compute_capability().map_err(Error::driver("cuDeviceGetAttribute(COMPUTE_CAPABILITY)"))?;
        Ok(Self((major * 10 + minor) as u32))
    }

    /// `major * 10 + minor`.
    pub fn capability(self) -> u32 {
        self.0
    }
}

impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sm_{}", self.0)
    }
}

struct Loaded {
    // the function first: it holds its module, which holds the context, and is gone before this struct's own handle on it
    function: CudaFunction,
    context: Arc<CudaContext>,
    device: DeviceInfo,
    stem: String,
    arch: String,
}

impl fmt::Debug for Loaded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Loaded")
            .field("stem", &self.stem)
            .field("arch", &self.arch)
            .field("device", &self.device.ordinal)
            .finish_non_exhaustive()
    }
}

/// What a manifest remembers of a kernel it loaded: the module is shared by the kernels of the same context while any of
/// them is alive, and unloaded with the last (a context that is gone does not leave a stale module behind).
#[derive(Debug)]
pub(crate) struct Cached {
    loaded: Weak<Loaded>,
    plan: Arc<Plan>,
}

/// A binary of a manifest, loaded in a context and ready to launch. Cheap to clone; it can be shared between threads. It keeps
/// its context, its module and its function alive.
#[derive(Debug, Clone)]
pub struct Kernel {
    loaded: Arc<Loaded>,
    plan: Arc<Plan>,
}

impl Kernel {
    /// The stem of the binary (`rowop.S32`).
    pub fn stem(&self) -> &str {
        &self.loaded.stem
    }

    /// The architecture the binary is for.
    pub fn arch(&self) -> &str {
        &self.loaded.arch
    }

    /// The ordinal of the device it is loaded on.
    pub fn device(&self) -> i32 {
        self.loaded.device.ordinal
    }

    /// The context the kernel was loaded in: a launch is on a stream of this context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.loaded.context
    }

    /// What a call is checked against.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    pub(crate) fn function(&self) -> &CudaFunction {
        &self.loaded.function
    }

    /// Check a call without launching it: every precondition of the manifest, and the grid against the device's limits.
    ///
    /// `tensors` are in the order of the manifest's ABI (see [`Plan::tensor_names`]); the dimensions and the runtime
    /// scalars all come from their shapes.
    pub fn check(&self, tensors: &[TensorArg<'_>]) -> Result<Prepared> {
        self.plan.prepare(tensors, Some(&self.loaded.device))
    }

    /// Check the call and launch the kernel on `stream`.
    ///
    /// The stream must be a stream of the context the kernel was loaded in (checked, with cudarc's own comparison of contexts); that
    /// context is made current on this thread, as cudarc does for its own calls. The launch is asynchronous, as any launch is: `stream`
    /// orders it with the host's other work. `tensors` are used by the call (a written slice is borrowed exclusively).
    ///
    /// What is guaranteed for a tensor made of a cudarc slice or view (`from_cuda`), in every context (whether cudarc tracks it or not):
    ///
    /// * the slice's own stream (the one it will be freed on) waits for the end of this kernel, when that is not `stream`: freeing the
    ///   slice, or any later work on its stream, runs after the kernel, however many other launches read the slice on other streams.
    ///   Making the tensor, and throwing it away or having the call refused, change nothing about that;
    /// * when the kernel writes the slice and its own stream is not `stream`, `stream` first waits for everything that was enqueued
    ///   on the slice's own stream, which includes the end of every earlier launch that used the slice from another stream: a writer
    ///   runs after all earlier readers and writers, not only the last one that cudarc recorded;
    /// * plus cudarc's own tracking, on `stream`: the launch is ordered after the slice's earlier writes (and, when it writes, its
    ///   reads), on whatever stream they were, in a context that has made a stream of its own and tracks events.
    ///
    /// A tensor made of a region the host vouched for ([`DevicePtr::from_raw`](crate::DevicePtr::from_raw)) is tracked by nothing:
    /// ordering it against other streams, and keeping it allocated, is the host's. Uses of a slice that are not launches of this
    /// crate are cudarc's or the host's (see the `memory` module docs).
    ///
    /// **An `Err` is not a promise that nothing was launched.** What refuses a call (a precondition, a stream of another context,
    /// a capture) enqueues nothing; a driver error can come after the kernel was enqueued (cudarc records events after it). When a
    /// slice's own stream is not `stream`, the call then still makes every such stream wait for what it can and blocks the host until
    /// `stream` is idle (best effort) before it returns the error, so freeing a slice after an error is still ordered after a kernel
    /// that may have run.
    ///
    /// **A CUDA graph is captured with [`Kernel::launch_raw`] and regions, not slices.** A launch of tensors made of cudarc slices
    /// is refused (an `Error::Launch`, nothing is enqueued) when `stream`, or a stream a slice is freed on, is capturing, in every
    /// context, whether cudarc tracks it or not: a graph holds the slices' addresses with nothing to keep the slices alive, and
    /// replaying it (which is safe) after they are freed would write into memory that something else uses. The regions are made with
    /// `unsafe` [`DevicePtr::from_raw`](crate::DevicePtr::from_raw), whose contract includes that the graph is not replayed after the
    /// memory it captured is gone; the launch itself may be this method or [`Kernel::launch_raw`] (regions are not looked at). The
    /// check is made when the call is made: a capture begun on `stream` by another thread while the call runs is not excluded by this
    /// crate. For a raw handle that cudarc does not own, see [`Kernel::launch_raw`].
    pub fn launch(&self, stream: &CudaStream, tensors: &mut [TensorArg<'_>]) -> Result<()> {
        if **stream.context() != *self.loaded.context {
            return Err(Error::Launch(format!(
                "the stream is in another CUDA context than the one '{}' was loaded in: launch it on a stream of that context",
                self.loaded.stem
            )));
        }
        let prepared = self.check(tensors)?;
        ffi::enqueue(&self.loaded.function, stream, &prepared, self.plan.block(), self.plan.shared(), tensors)
    }
}

impl Manifest {
    /// The binary for `name` (a stem such as `rowop.S32`, or an operator) with exactly these bucket values, for `arch`,
    /// loaded in `ctx`.
    ///
    /// Bucket semantics are exact: the bucket values name the binary (each bucketed dimension once: a dimension named twice is refused
    /// with [`Error::Launch`]), there is no padding and no rounding up. `arch` is
    /// the compute capability of the device of `ctx` (`Arch::of_context(&ctx)?`); a manifest that has no binary for it is refused,
    /// as is an `arch` that is not the device's. The module is loaded once per context and shared by later calls, and the kernel
    /// keeps `ctx` alive.
    pub fn kernel(&self, ctx: &Arc<CudaContext>, name: &str, buckets: &[(&str, u64)], arch: Arch) -> Result<Kernel> {
        let index = self.select(name, buckets, &arch.to_string())?;
        let entry = &self.entries[index];
        let plan = Plan::new(entry)?;
        let device_arch = Arch::of_context(ctx)?;
        if device_arch != arch {
            return Err(Error::Unsupported(format!(
                "'{}' is asked for {arch}, but the CUDA device {} of the context is {device_arch}",
                entry.stem,
                ctx.ordinal()
            )));
        }
        let key = (index, ctx.cu_ctx() as usize);
        let mut cache = self.kernels.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = cache.get(&key)
            && let Some(loaded) = cached.loaded.upgrade()
        {
            return Ok(Kernel { loaded, plan: Arc::clone(&cached.plan) });
        }
        let attribute =
            |a: CUdevice_attribute, call: &'static str| ctx.attribute(a).map(i64::from).map_err(Error::driver(call));
        let max_grid = [
            attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, "cuDeviceGetAttribute(MAX_GRID_DIM_X)")?,
            attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, "cuDeviceGetAttribute(MAX_GRID_DIM_Y)")?,
            attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, "cuDeviceGetAttribute(MAX_GRID_DIM_Z)")?,
        ];
        let shared_optin = attribute(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)",
        )?;
        if entry.symbol.contains('\0') {
            return Err(Error::Unsupported(format!("the symbol {:?} has a NUL byte", entry.symbol)));
        }
        let module =
            ctx.load_module(Ptx::from_binary(entry.cubin.to_vec())).map_err(Error::driver("cuModuleLoadData"))?;
        let function = module.load_function(&entry.symbol).map_err(Error::driver("cuModuleGetFunction"))?;
        let who = format!("'{}' ({})", entry.stem, entry.arch);
        // the loaded function must take the parameters of the manifest's ABI, with the sizes it says (Python's `_check_params`): a prebuilt
        // binary's ABI is declared by its author, and a function that read more than it was given would read what follows the array
        let reported = ffi::param_sizes(&function);
        let expected: Vec<usize> =
            entry.params.iter().map(Param::size).chain(std::iter::repeat_n(8, entry.scratch_slots)).collect();
        if reported != expected {
            return Err(Error::Manifest(format!(
                "{who}: the loaded function '{}' takes parameters of {reported:?} bytes, but the ABI in the manifest says {expected:?}",
                entry.symbol
            )));
        }
        let static_shared =
            i64::from(function.shared_size_bytes().map_err(Error::driver("cuFuncGetAttribute(SHARED_SIZE_BYTES)"))?);
        if static_shared + i64::from(entry.shared) > shared_optin {
            return Err(Error::Unsupported(format!(
                "{who} needs {static_shared} bytes of static and {} bytes of dynamic shared memory, more than the {shared_optin} bytes a block may use on this device",
                entry.shared
            )));
        }
        if entry.shared > DEFAULT_SHARED_LIMIT {
            let bytes = i32::try_from(entry.shared)
                .map_err(|_| Error::Unsupported(format!("{who}: too much shared memory")))?;
            set_max_dynamic_shared(ctx, &function, bytes)?;
        }
        let kernel = Kernel {
            loaded: Arc::new(Loaded {
                function,
                context: Arc::clone(ctx),
                device: DeviceInfo { ordinal: ctx.ordinal() as i32, context: Some(ctx.cu_ctx() as usize), max_grid },
                stem: entry.stem.clone(),
                arch: entry.arch.clone(),
            }),
            plan: Arc::new(plan),
        };
        cache.insert(key, Cached { loaded: Arc::downgrade(&kernel.loaded), plan: Arc::clone(&kernel.plan) });
        Ok(kernel)
    }

    /// The plan of a binary without loading it: what [`Manifest::kernel`] checks calls with, for a machine with no GPU
    /// (tests, a dry run in CI).
    pub fn plan(&self, name: &str, buckets: &[(&str, u64)], arch: Arch) -> Result<Plan> {
        let index = self.select(name, buckets, &arch.to_string())?;
        Plan::new(&self.entries[index])
    }
}

/// Allow `bytes` of dynamic shared memory (needed above the default 48 KiB) for `function`, which is in `ctx`.
fn set_max_dynamic_shared(ctx: &CudaContext, function: &CudaFunction, bytes: i32) -> Result<()> {
    let call = "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)";
    // cudarc's `set_attribute` does not bind the context, as its `get_attribute` does
    ctx.bind_to_thread().map_err(Error::driver(call))?;
    function
        .set_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes)
        .map_err(Error::driver(call))
}
