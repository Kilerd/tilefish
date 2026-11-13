//! Loading a binary into the CUDA driver and launching it.
//!
//! What Python's `_cuda.py` does through `cuda-python`, through `cudarc`'s driver bindings: `cuModuleLoadData` on the
//! verified bytes in memory, `cuModuleGetFunction` on the manifest's symbol, `cuFuncSetAttribute` for dynamic shared
//! memory above the default 48 KiB, and `cuLaunchKernel` with `kernelParams` laid out as the ABI says (the arguments in
//! declaration order, a pointer as 64 bits and an `i32` as 32, then two null 64-bit scratch pointers).
//!
//! A load is checked against the device like Python's: the function's static shared memory (as the driver reports it)
//! plus the manifest's dynamic shared memory must fit the device's opt-in maximum per block, and the loaded function must
//! take the parameters the ABI says. A launch is checked by [`Plan::prepare`], and then against the device's maximum grid.
//!
//! A launch goes on the stream the caller gives (a raw `CUstream`), so it also works inside a CUDA graph capture: the
//! driver copies `kernelParams` at the call.

use std::fmt;
use std::sync::{Arc, Weak};

pub use cudarc::driver::sys::CUstream;

use crate::error::{Error, Result};
use crate::ffi;
use crate::manifest::{Manifest, ParamKind};
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

    /// The compute capability of the device with this ordinal.
    pub fn of_device(ordinal: i32) -> Result<Self> {
        let (major, minor) = ffi::compute_capability(ordinal)?;
        Ok(Self((major * 10 + minor) as u32))
    }

    /// The compute capability of the device that is current on this thread (the one a binary is loaded for).
    pub fn current() -> Result<Self> {
        Self::of_device(ffi::current()?.ordinal)
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
    // the function first: it is gone before the module is unloaded
    function: ffi::Function,
    _module: ffi::Module, // held for its Drop, which unloads the module
    device: DeviceInfo,
    context: usize,
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

/// A binary of a manifest, loaded on a device and ready to launch. Cheap to clone; it can be shared between threads
/// (each thread that launches needs the kernel's CUDA context to be current).
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

    /// What a call is checked against.
    pub fn plan(&self) -> &Plan {
        &self.plan
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
    /// The launch is asynchronous, as any launch is: `stream` orders it with the host's other work, and the tensors must
    /// stay allocated until it has run. `stream` may be capturing a CUDA graph: the call is captured like any kernel node.
    /// The CUDA context the kernel was loaded in must be current on this thread.
    pub fn launch(&self, stream: CUstream, tensors: &[TensorArg<'_>]) -> Result<()> {
        let mut prepared = self.check(tensors)?;
        if !ffi::is_current(self.loaded.context) {
            return Err(Error::Launch(format!(
                "'{}' was loaded in another CUDA context than the one that is current on this thread: make its context current first",
                self.loaded.stem
            )));
        }
        let (block, shared) = (self.plan.block(), self.plan.shared());
        self.loaded.function.launch(prepared.grid, block, shared, stream, &mut prepared.values, prepared.count)
    }
}

impl Manifest {
    /// The binary for `name` (a stem such as `rowop.S32`, or an operator) with exactly these bucket values, for `arch`,
    /// loaded in the CUDA context that is current on this thread.
    ///
    /// Bucket semantics are exact: the bucket values name the binary, there is no padding and no rounding up. `arch` is
    /// the compute capability of the device (`Arch::current()?`); a manifest that has no binary for it is refused, as is
    /// an `arch` that is not the current device's. The module is loaded once per context and shared by later calls.
    pub fn kernel(&self, name: &str, buckets: &[(&str, u64)], arch: Arch) -> Result<Kernel> {
        let index = self.select(name, buckets, &arch.to_string())?;
        let entry = &self.entries[index];
        let plan = Plan::new(entry)?;
        let current = ffi::current()?;
        let device_arch = Arch::of_device(current.ordinal)?;
        if device_arch != arch {
            return Err(Error::Unsupported(format!(
                "'{}' is asked for {arch}, but the current CUDA device {} is {device_arch}",
                entry.stem, current.ordinal
            )));
        }
        let key = (index, current.context);
        let mut cache = self.kernels.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = cache.get(&key)
            && let Some(loaded) = cached.loaded.upgrade()
        {
            return Ok(Kernel { loaded, plan: Arc::clone(&cached.plan) });
        }
        let limits = ffi::limits(current.ordinal)?;
        let module = ffi::Module::load(&entry.cubin, current.context)?;
        let function = module.function(&entry.symbol)?;
        let static_shared = function.static_shared()?;
        let who = format!("'{}' ({})", entry.stem, entry.arch);
        if static_shared + i64::from(entry.shared) > limits.shared_optin {
            return Err(Error::Unsupported(format!(
                "{who} needs {static_shared} bytes of static and {} bytes of dynamic shared memory, more than the {} bytes a block may use on this device",
                entry.shared, limits.shared_optin
            )));
        }
        if entry.shared > DEFAULT_SHARED_LIMIT {
            function.set_max_dynamic_shared(
                i32::try_from(entry.shared)
                    .map_err(|_| Error::Unsupported(format!("{who}: too much shared memory")))?,
            )?;
        }
        if let Some(sizes) = function.param_sizes()? {
            let want: Vec<usize> = entry
                .params
                .iter()
                .map(|p| if matches!(p.kind, ParamKind::Pointer { .. }) { 8 } else { 4 })
                .chain([8, 8])
                .collect();
            if sizes != want {
                return Err(Error::Unsupported(format!(
                    "the loaded function '{}' takes parameters of {sizes:?} bytes, but the ABI in the manifest says {want:?}",
                    entry.symbol
                )));
            }
        }
        let kernel = Kernel {
            loaded: Arc::new(Loaded {
                function,
                _module: module,
                device: DeviceInfo { ordinal: current.ordinal, max_grid: limits.grid },
                context: current.context,
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
