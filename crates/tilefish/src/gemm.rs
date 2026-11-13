//! A pinned cuBLASLt GEMM of a manifest's `library` section, loaded and launched with the checks of a kernel.
//!
//! `out[M, N] = a[M, K] @ b[N, K]^T (+ bias[N])` in tilefish's row-major terms: bf16 operands, an f32 accumulator, one rounding to bf16 at
//! the end, the bias added in f32 before it (`docs/manifest-schema-3.md`, section 9). The entry pins the 64 bytes of the algorithm that
//! cuBLASLt's heuristic ranked first under a workspace limit, with the cuBLASLt version that chose it and the SM count of the device it
//! was chosen on; [`Manifest::gemm`] refuses a device of another SM count (an algorithm is chosen for the multiprocessors it runs on: a MIG
//! slice has fewer than the whole GPU), another version (an algorithm id names other kernels in another version), vets the 64 bytes the way Python's binding does (a weak check by the
//! library itself would accept several corruptions and crash the process on some), and has the library approve the algorithm for the
//! problem. A launch is then [`Gemm::launch`] on a cudarc stream, which checks everything [`Kernel::launch`](crate::Kernel::launch)
//! does of its tensors (shapes bound to the entry's dimensions, the exact bucket, dtypes, contiguity, 16-byte alignment, the device and
//! context, aliasing) and orders the slices with the same events and the same refusal of a capture; or `unsafe` [`Gemm::launch_raw`], with
//! regions, which is how it is captured in a CUDA graph.
//!
//! **The workspace is the caller's.** cuBLASLt needs scratch memory the algorithm names (`workspace_required`, never more than the entry's
//! `workspace_bytes`, which is what Python provides), and an allocation inside a launch is an allocation inside a graph capture. So the
//! last of a launch's tensors is the workspace, a `TensorMut<U8>` of at least [`Gemm::workspace_required`] bytes (a cudarc slice, or a
//! region), 256-byte aligned, which the call writes: it must not overlap an operand, and two launches that share one are ordered by
//! cudarc's write tracking and the crate's writer-waits-for-readers rule like any other written tensor (on one stream, by stream order).
//! [`Gemm::alloc_workspace`] allocates [`Gemm::workspace_bytes`], the entry's size.

use std::sync::{Arc, Weak};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use crate::error::{Error, PreconditionKind, Result};
use crate::ffi::cublaslt::{self, Algo, AlgoConfig, Handle};
use crate::kernel::Arch;
use crate::library::LibraryEntry;
use crate::manifest::{Access, Manifest};
use crate::plan::{DeviceInfo, Plan};
use crate::tensor::{DType, TensorArg};

/// The alignment of the workspace pointer (cuBLASLt asks for 16 bytes at least, and recommends 256: cudarc's allocations are).
const WORKSPACE_ALIGNMENT: u64 = 256;
const ACCUMULATOR_TYPE_MASK: u64 = 0xFF << 8;
const ACCUMULATOR_32F: u64 = 0x02 << 8;
const REDUCTION_NONE: u32 = 0;
const REDUCTION_COMPUTE_TYPE: u32 = 2;
/// The alignment of every operand tilefish guarantees (and the library is told: `MIN_ALIGNMENT_{A,B,C,D}_BYTES = 16`).
const OPERAND_ALIGNMENT: u32 = 16;

/// What a manifest remembers of a GEMM it loaded: the handle and the checked algorithm are shared by the GEMMs of the same context while
/// any of them is alive.
#[derive(Debug)]
pub(crate) struct CachedGemm {
    loaded: Weak<LoadedGemm>,
    plan: Arc<Plan>,
}

#[derive(Debug)]
struct LoadedGemm {
    // the handle first: it is destroyed before this struct's own handle on the context
    handle: Handle,
    context: Arc<CudaContext>,
    device: DeviceInfo,
    algo: Algo,
    /// What the library says the algorithm needs for the problem.
    required: usize,
    stem: String,
    arch: String,
    m: u64,
    n: u64,
    k: u64,
    bias: bool,
    workspace_bytes: u64,
    sm_count: u64,
}

/// A pinned GEMM of a manifest, loaded in a context and ready to launch. Cheap to clone; it can be shared between threads. It keeps its
/// context and its cuBLASLt handle alive.
#[derive(Debug, Clone)]
pub struct Gemm {
    loaded: Arc<LoadedGemm>,
    plan: Arc<Plan>,
}

/// The addresses of a checked call.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Call {
    pub a: u64,
    pub b: u64,
    pub bias: Option<u64>,
    pub out: u64,
    /// The workspace, or 0 when the algorithm needs none.
    pub workspace: u64,
}

fn refuse(who: &str, why: impl std::fmt::Display) -> Error {
    Error::Manifest(format!("{who}: {why}"))
}

/// Refuse an algorithm that cuBLASLt itself would not have produced, before `cublasLtMatmulAlgoCheck` sees it.
///
/// Measured by Python's binding (cuBLASLt 12.8.4 on a B300): `cublasLtMatmulAlgoCheck` accepts an algorithm with a changed version field,
/// changed padding, a reserved CTA swizzling or an out-of-range custom option, and it crashes the process on a tile id of 2^31 or more. So
/// the bytes are held to what the library encodes for the configuration they read back as: the algorithm is rebuilt with
/// `cublasLtMatmulAlgoInit` (which also writes the data types, the compute type and the version) and the configuration attributes, and must
/// be byte for byte the same; the configuration must be one that the capabilities of its id list (tile, stages, swizzling, custom option);
/// it must accumulate in f32 and need no more than 16-byte alignment, and its split-K must go with a reduction scheme that does not round in
/// bf16 (none, or f32 partial sums). What this cannot see is a flip to another valid configuration (another tile, say): that is a
/// different, legitimate algorithm, and only the digest of the manifest vouches for the file.
fn vet(handle: &Handle, algo: &Algo, who: &str) -> Result<AlgoConfig> {
    let bytes = algo.to_bytes();
    let hex: String = bytes[..8].iter().map(|b| format!("{b:02x}")).collect();
    let head = format!(
        "{who}: the pinned cuBLASLt algorithm {hex}... is not one cuBLASLt produces for bf16 with f32 accumulation"
    );
    let library = |e: Error| refuse(&head, e);
    let config = algo.config().map_err(library)?;
    let fresh = Algo::init(handle, config.id)
        .map_err(|e| Error::Manifest(format!("{head}: algorithm id {} is unknown ({e})", config.id)))?;
    let caps = fresh.caps().map_err(library)?;
    let mut problems: Vec<String> = Vec::new();
    // an algorithm that has no tile (or stages) ids lists none and uses the _UNDEFINED value (0)
    let listed = |ids: &[u32], id: u32| if ids.is_empty() { id == 0 } else { ids.contains(&id) };
    if !listed(&caps.tile_ids, config.tile_id) {
        problems.push(format!(
            "tile id {} is not one of the {} that algorithm {} has",
            config.tile_id,
            caps.tile_ids.len(),
            config.id
        ));
    }
    if !listed(&caps.stages_ids, config.stages_id) {
        problems.push(format!(
            "stages id {} is not one of the {} that algorithm {} has",
            config.stages_id,
            caps.stages_ids.len(),
            config.id
        ));
    }
    if config.cta_swizzling > caps.cta_swizzling {
        problems.push(format!(
            "CTA swizzling {} is not supported (algorithm {} takes up to {})",
            config.cta_swizzling, config.id, caps.cta_swizzling
        ));
    }
    if i64::from(config.custom_option) > i64::from(caps.custom_option_max) {
        problems.push(format!(
            "custom option {} is above the maximum {} of algorithm {}",
            config.custom_option, caps.custom_option_max, config.id
        ));
    }
    if config.splitk_num < 1 {
        problems.push(format!("split-K count {} is below 1", config.splitk_num));
    }
    if (config.splitk_num > 1) != (config.reduction_scheme != REDUCTION_NONE) {
        problems.push(format!(
            "split-K count {} does not go with reduction scheme {} (split-K needs one, no split-K has none)",
            config.splitk_num, config.reduction_scheme
        ));
    }
    if ![REDUCTION_NONE, REDUCTION_COMPUTE_TYPE].contains(&config.reduction_scheme) {
        problems.push(format!(
            "reduction scheme {} does not keep the one rounding (only no split-K, 0, and f32 partial sums, 2, do; bf16 partial sums round more than once)",
            config.reduction_scheme
        ));
    }
    if caps.numerical_impl_flags & ACCUMULATOR_TYPE_MASK != ACCUMULATOR_32F {
        problems.push("it does not accumulate in f32".to_owned());
    }
    for (name, needed) in ["A", "B", "C", "D"].into_iter().zip(caps.min_alignment) {
        if needed > OPERAND_ALIGNMENT {
            problems.push(format!(
                "it needs {needed}-byte alignment of {name}, more than the {OPERAND_ALIGNMENT} bytes tilefish guarantees"
            ));
        }
    }
    if problems.is_empty() {
        let mut rebuilt = fresh;
        match rebuilt.set_config(&config) {
            Ok(()) if rebuilt.to_bytes() != bytes => problems
                .push("its bytes are not those cuBLASLt encodes for its configuration (version, data types or padding differ)".to_owned()),
            Ok(()) => {}
            Err(e) => problems.push(format!("its configuration cannot be set on a fresh algorithm ({e})")),
        }
    }
    if problems.is_empty() { Ok(config) } else { Err(Error::Manifest(format!("{head}: {}", problems.join("; ")))) }
}

impl Manifest {
    /// The pinned GEMM for `name` (a stem such as `gemm.M64`, or an operator) with exactly these bucket values, for `arch`, loaded in `ctx`.
    ///
    /// As [`Manifest::kernel`]: bucket semantics are exact, `arch` is the compute capability of the device of `ctx` (a pin belongs to one
    /// architecture), and the GEMM is loaded once per context and shared while any clone of it is alive. It is refused when cuBLASLt is not
    /// on this machine, when its version is not the one that chose the entry's algorithm, when the algorithm is not one cuBLASLt would
    /// have produced (see the module docs), when the library does not approve it for the problem, or when it needs more workspace than
    /// the entry's `workspace_bytes`.
    pub fn gemm(&self, ctx: &Arc<CudaContext>, name: &str, buckets: &[(&str, u64)], arch: Arch) -> Result<Gemm> {
        let index = self.select_library(name, buckets, &arch.to_string())?;
        let entry = &self.library[index];
        let plan = Plan::for_library(entry)?;
        let device_arch = Arch::of_context(ctx)?;
        if device_arch != arch {
            return Err(Error::Unsupported(format!(
                "'{}' is asked for {arch}, but the CUDA device {} of the context is {device_arch}",
                entry.stem,
                ctx.ordinal()
            )));
        }
        let key = (index, ctx.cu_ctx() as usize);
        let mut cache = self.gemms.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = cache.get(&key)
            && let Some(loaded) = cached.loaded.upgrade()
        {
            return Ok(Gemm { loaded, plan: Arc::clone(&cached.plan) });
        }
        let loaded = load(entry, ctx)?;
        let gemm = Gemm { loaded: Arc::new(loaded), plan: Arc::new(plan) };
        cache.insert(key, CachedGemm { loaded: Arc::downgrade(&gemm.loaded), plan: Arc::clone(&gemm.plan) });
        Ok(gemm)
    }
}

impl Manifest {
    /// The plan of a library operation without loading it: what [`Manifest::gemm`] checks the operands of a call with (`a`, `b`, `bias` and
    /// `out`; the workspace is checked on its own), for a machine with no GPU (tests, a dry run in CI).
    pub fn library_plan(&self, name: &str, buckets: &[(&str, u64)], arch: Arch) -> Result<Plan> {
        let index = self.select_library(name, buckets, &arch.to_string())?;
        Plan::for_library(&self.library[index])
    }
}

/// Refuse an entry pinned on a device of another multiprocessor count than this one (`device`), before any other field of the entry is
/// compared (as Python does): an entry is found by its stem and its architecture, and two devices of one architecture may differ in their
/// SM count (a MIG slice and the whole GPU), for which another algorithm is chosen.
fn require_sm_count(entry: &LibraryEntry, device: u64) -> Result<()> {
    sm_count_matches(&entry.stem, &entry.arch, entry.sm_count, device)
}

fn sm_count_matches(stem: &str, arch: &str, pinned: u64, device: u64) -> Result<()> {
    if pinned != device {
        return Err(Error::Manifest(format!(
            "the manifest pinned the cuBLASLt algorithm of '{}' ({}) on a device with {} SMs, but this device has {device}: an algorithm is \
             chosen for the multiprocessors it runs on, so the pin is refused (re-export the manifest on this device)",
            stem, arch, pinned
        )));
    }
    Ok(())
}

fn load(entry: &LibraryEntry, ctx: &Arc<CudaContext>) -> Result<LoadedGemm> {
    let who = format!("'{}' ({})", entry.stem, entry.arch);
    let sm_count = ctx
        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(Error::driver("cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)"))?;
    require_sm_count(entry, u64::try_from(sm_count).unwrap_or(0))?;
    if !cublaslt::available() {
        return Err(Error::Unsupported(format!(
            "{who} is a cuBLASLt GEMM, and libcublasLt cannot be loaded on this machine (with LD_LIBRARY_PATH or the like, give the process the \
             cuBLASLt of version {}, the one that chose its algorithm)",
            entry.cublaslt_version
        )));
    }
    let here = cublaslt::version();
    if here != entry.cublaslt_version {
        return Err(Error::Manifest(format!(
            "the manifest pinned the cuBLASLt algorithm of '{}' ({}) with cuBLASLt {}, but this process has cuBLASLt {here}: an algorithm id names \
             other kernels in another version, so the pin is refused (export the manifest again with this cuBLASLt, or run with the version that made it)",
            entry.stem, entry.arch, entry.cublaslt_version
        )));
    }
    let handle = Handle::new(ctx)?;
    let algo = Algo::from_bytes(&entry.algo);
    vet(&handle, &algo, &who)?;
    let problem = cublaslt::Problem::new(entry.m, entry.n, entry.k, entry.implementation.has_bias().then_some(16))?;
    let checked = cublaslt::check(&handle, &problem, &algo).map_err(|e| {
        Error::Manifest(format!(
            "{who}: the pinned cuBLASLt algorithm is not valid for the matmul m={} n={} k={}: {e}",
            entry.m, entry.n, entry.k
        ))
    })?;
    if checked.workspace as u64 > entry.workspace_bytes {
        return Err(Error::Manifest(format!(
            "{who}: the pinned cuBLASLt algorithm needs {} bytes of workspace for the matmul m={} n={} k={}, more than the {} recorded",
            checked.workspace, entry.m, entry.n, entry.k, entry.workspace_bytes
        )));
    }
    Ok(LoadedGemm {
        handle,
        context: Arc::clone(ctx),
        device: DeviceInfo {
            ordinal: ctx.ordinal() as i32,
            context: Some(ctx.cu_ctx() as usize),
            max_grid: [i64::MAX; 3],
        },
        algo,
        required: checked.workspace,
        stem: entry.stem.clone(),
        arch: entry.arch.clone(),
        m: entry.m,
        n: entry.n,
        k: entry.k,
        bias: entry.implementation.has_bias(),
        workspace_bytes: entry.workspace_bytes,
        sm_count: entry.sm_count,
    })
}

impl Gemm {
    /// The stem of the library entry (`gemm.M64`).
    pub fn stem(&self) -> &str {
        &self.loaded.stem
    }

    /// The architecture the algorithm was pinned for.
    pub fn arch(&self) -> &str {
        &self.loaded.arch
    }

    /// The ordinal of the device it is loaded on.
    pub fn device(&self) -> i32 {
        self.loaded.device.ordinal
    }

    /// The context the GEMM was loaded in: a launch is on a stream of this context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.loaded.context
    }

    /// What the operands of a call are checked against: the entry's operands `a`, `b`, `bias` (for `gemm_bias`) and `out` (the workspace
    /// is checked on its own).
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// The sizes of the problem in tilefish's row-major terms: `a` is `[m, k]`, `b` is `[n, k]`, `out` is `[m, n]`.
    pub fn sizes(&self) -> (u64, u64, u64) {
        (self.loaded.m, self.loaded.n, self.loaded.k)
    }

    /// Whether the operation has a `bias` operand (`cublaslt.gemm_bias`).
    pub fn has_bias(&self) -> bool {
        self.loaded.bias
    }

    /// The multiprocessor count of the device the algorithm was pinned on (the device it is loaded on has it: [`Manifest::gemm`] refuses another).
    pub fn sm_count(&self) -> u64 {
        self.loaded.sm_count
    }

    /// The workspace limit of the entry: the limit the heuristic was asked under, and the size of the workspace Python provides. The size
    /// [`Gemm::alloc_workspace`] allocates.
    pub fn workspace_bytes(&self) -> u64 {
        self.loaded.workspace_bytes
    }

    /// The bytes of workspace the pinned algorithm needs for this problem (`cublasLtMatmulAlgoCheck`'s answer): at most
    /// [`Gemm::workspace_bytes`]. The workspace of a launch has at least this many.
    pub fn workspace_required(&self) -> u64 {
        self.loaded.required as u64
    }

    /// A workspace of [`Gemm::workspace_bytes`] bytes (zeroed), allocated on `stream`: a `CudaSlice<u8>` to make the last tensor of a launch of
    /// (`TensorMut::<U8>::from_cuda`). Allocate it once, outside any graph capture, and keep it as long as launches use it.
    pub fn alloc_workspace(&self, stream: &Arc<CudaStream>) -> Result<CudaSlice<u8>> {
        let bytes = usize::try_from(self.loaded.workspace_bytes)
            .map_err(|_| Error::Launch("the workspace of the entry does not fit the address space".to_owned()))?;
        stream.alloc_zeros::<u8>(bytes).map_err(Error::driver("cuMemAllocAsync"))
    }

    /// Everything a launch checks, with nothing launched. `tensors` are `a`, `b`, `bias` (for `gemm_bias`), `out` and the workspace.
    pub fn check(&self, tensors: &[TensorArg<'_>]) -> Result<()> {
        self.check_call(tensors).map(drop)
    }

    /// The engine of a launch: what `ffi` calls cuBLASLt with.
    pub(crate) fn engine(&self) -> (&Handle, &Algo, usize, [u64; 3]) {
        (&self.loaded.handle, &self.loaded.algo, self.loaded.required, [self.loaded.m, self.loaded.n, self.loaded.k])
    }

    /// Check a call (see [`Gemm::launch`]) and return the addresses to run it with.
    pub(crate) fn check_call(&self, tensors: &[TensorArg<'_>]) -> Result<Call> {
        let stem = self.loaded.stem.as_str();
        let operands = self.plan.tensor_names().count();
        if tensors.len() != operands + 1 {
            return Err(Error::Launch(format!(
                "'{stem}' takes {} tensors ({}, workspace), got {}",
                operands + 1,
                self.plan.tensor_names().collect::<Vec<_>>().join(", "),
                tensors.len()
            )));
        }
        let (operand_args, workspace) = tensors.split_at(operands);
        let workspace = &workspace[0];
        // the operands: shapes bound to the entry's dimensions, the exact bucket, dtype, contiguity, alignment, device and context, aliasing
        self.plan.prepare(operand_args, Some(&self.loaded.device))?;
        // (the entry's shapes agree with m, n, k, and its dimensions are bucketed to them: a defense for the memory the library reads)
        let (m, n, k) = (self.loaded.m as usize, self.loaded.n as usize, self.loaded.k as usize);
        let mut wanted: Vec<Vec<usize>> = vec![vec![m, k], vec![n, k]];
        if self.loaded.bias {
            wanted.push(vec![n]);
        }
        wanted.push(vec![m, n]);
        for (arg, want) in operand_args.iter().zip(&wanted) {
            if arg.shape != want.as_slice() {
                return Err(Error::precondition(
                    PreconditionKind::Shape,
                    None,
                    format!(
                        "'{stem}': an operand has shape {:?}, but the problem is a [{m}, {k}] by [{n}, {k}] GEMM",
                        arg.shape
                    ),
                ));
            }
        }
        self.check_workspace(workspace, operand_args)?;
        let address = |i: usize| operand_args[i].ptr;
        let (a, b) = (address(0), address(1));
        let (bias, out) = if self.loaded.bias { (Some(address(2)), address(3)) } else { (None, address(2)) };
        Ok(Call { a, b, bias, out, workspace: if self.loaded.required == 0 { 0 } else { workspace.ptr } })
    }

    /// The workspace is a tensor that is written: bytes, enough of them, aligned, in this context, overlapping no operand.
    fn check_workspace(&self, workspace: &TensorArg<'_>, operands: &[TensorArg<'_>]) -> Result<()> {
        let stem = self.loaded.stem.as_str();
        let name = Some("workspace");
        let device = &self.loaded.device;
        if workspace.device != device.ordinal {
            return Err(Error::precondition(
                PreconditionKind::Device,
                name,
                format!(
                    "the workspace is on device {}, but '{stem}' is loaded on device {}",
                    workspace.device, device.ordinal
                ),
            ));
        }
        if workspace.context.is_some_and(|c| Some(c) != device.context) {
            return Err(Error::precondition(
                PreconditionKind::Device,
                name,
                format!("the workspace is memory of another CUDA context than the one '{stem}' is loaded in"),
            ));
        }
        if workspace.access == Access::In {
            return Err(Error::precondition(
                PreconditionKind::Access,
                name,
                format!(
                    "the workspace is a read-only tensor, but '{stem}' writes it: give it a tensor that may be written"
                ),
            ));
        }
        if workspace.dtype != DType::U8 {
            return Err(Error::precondition(
                PreconditionKind::Dtype,
                name,
                format!("the workspace has dtype {}, but it is bytes (u8)", workspace.dtype.name()),
            ));
        }
        let contiguous = workspace.strides.is_none_or(|s| s == [1] || workspace.shape == [1]);
        if workspace.shape.len() != 1 || !contiguous {
            return Err(Error::precondition(
                PreconditionKind::Rank,
                name,
                format!(
                    "the workspace is a contiguous tensor of rank 1 (shape [bytes]), got shape {:?} and strides {:?}",
                    workspace.shape, workspace.strides
                ),
            ));
        }
        if (workspace.shape[0] as u64) < self.loaded.required as u64 {
            return Err(Error::precondition(
                PreconditionKind::Shape,
                name,
                format!(
                    "the workspace has {} bytes, but the pinned algorithm of '{stem}' needs {} (the entry's limit is {})",
                    workspace.shape[0], self.loaded.required, self.loaded.workspace_bytes
                ),
            ));
        }
        if !workspace.ptr.is_multiple_of(WORKSPACE_ALIGNMENT) {
            return Err(Error::precondition(
                PreconditionKind::PointerAlignment,
                name,
                format!(
                    "the workspace is at address {:#x}, which is not aligned to {WORKSPACE_ALIGNMENT} bytes",
                    workspace.ptr
                ),
            ));
        }
        let (start, end) = (u128::from(workspace.ptr), u128::from(workspace.ptr) + workspace.shape[0] as u128);
        for (arg, plan_name) in operands.iter().zip(self.plan.tensor_names()) {
            let bytes = crate::tensor::reach(arg.shape, arg.strides).saturating_mul(arg.dtype.size() as u128);
            let (a_start, a_end) = (u128::from(arg.ptr), u128::from(arg.ptr).saturating_add(bytes));
            if start < a_end && a_start < end {
                return Err(Error::precondition(
                    PreconditionKind::Alias,
                    name,
                    format!(
                        "the workspace is written in place but overlaps argument '{plan_name}': it cannot alias an operand"
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Check the call and run the GEMM on `stream`.
    ///
    /// `tensors` are `a` `[m, k]`, `b` `[n, k]`, `bias` `[n]` (for `gemm_bias`), `out` `[m, n]` (written) and the workspace (written), in this
    /// order: bf16 tensors but the workspace, which is bytes, contiguous, 16-byte aligned (the workspace 256), with the entry's static sizes
    /// and bucket exactly. The stream must be a stream of the context the GEMM was loaded in (checked); that context is made current on
    /// this thread, as cudarc does for its own calls. The launch is asynchronous: `stream` orders it with the host's other work.
    ///
    /// The slices are ordered exactly as for [`Kernel::launch`](crate::Kernel::launch): cudarc's launch tracking on `stream`, the slices'
    /// own streams waiting for the end of the GEMM (and a written slice waiting for everything before it), a refusal while a stream is
    /// capturing (`launch_raw` and regions capture a GEMM), and an `Err` that is not a promise that nothing was enqueued.
    pub fn launch(&self, stream: &CudaStream, tensors: &mut [TensorArg<'_>]) -> Result<()> {
        if **stream.context() != *self.loaded.context {
            return Err(Error::Launch(format!(
                "the stream is in another CUDA context than the one '{}' was loaded in: launch it on a stream of that context",
                self.loaded.stem
            )));
        }
        let call = self.check_call(tensors)?;
        crate::ffi::enqueue_gemm(self, stream, &call, tensors)
    }
}

/// The version of cuBLASLt this process has (`120804` for 12.8.4), or `None` when it cannot be loaded: what a pinned algorithm is compared with.
pub fn cublaslt_version() -> Option<u64> {
    cublaslt::available().then(cublaslt::version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pin_of_a_device_with_another_sm_count_is_refused_and_says_to_export_again() {
        assert!(sm_count_matches("gemm.M64", "sm_103", 30, 30).is_ok());
        // a whole B300 (148 SMs) and a 1g MIG slice of it (30 SMs) are one architecture, and another algorithm is chosen for each
        let e = sm_count_matches("gemm.M64", "sm_103", 30, 148).unwrap_err();
        assert!(
            matches!(&e, Error::Manifest(m) if m.contains("on a device with 30 SMs, but this device has 148")),
            "{e}"
        );
        assert!(e.to_string().contains("re-export the manifest on this device"), "{e}");
    }
}
