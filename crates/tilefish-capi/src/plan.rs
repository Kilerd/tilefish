//! Launch plans, plan sets, arenas, runs, weighted runs and captured plans.
//!
//! A run owns its arena, or shares one (`tilefish_shared_arena`, `tilefish_set_run`); the crate orders every launch, capture and replay on an
//! arena with the others, whatever the stream, and checks every step before the first launch. The externals are the caller's regions, which the
//! caller orders. A plan is captured with `tilefish_plan_run_capture` (a plan launch on a stream that is capturing is refused): the captured
//! plan owns the graph.

use std::ffi::{CString, c_char};
use std::sync::Arc;

use tilefish::cudarc::driver::sys::CUstream;
use tilefish::{
    BufferRole, CapturedPlan, LaunchPlan, Pin, PlanRun, PlanSet, SetRun, SharedArena, Tap, TensorArg, WeightedRun,
};

use crate::context::{Ctx, destroy_keeping_current, tilefish_context};
use crate::error::{Outcome, c_string, call, invalid, quiet, tilefish_status};
use crate::manifest::tilefish_manifest;
use crate::types::{
    KeepCurrent, OwnedDims, OwnedNamed, buckets_arg, devscalar_of, devscalars_arg, dtype_code, handle, handle_mut,
    named_arg, out_arg, put, put_value, slice_arg, str_arg, tensor_arg, tilefish_bucket, tilefish_devscalar,
    tilefish_named_tensor, tilefish_tap, tilefish_tensor,
};

/// A graph parameter that a call gives every time (an activation), read.
pub const TILEFISH_ROLE_INPUT: i32 = 0;
/// A returned value, written.
pub const TILEFISH_ROLE_OUTPUT: i32 = 1;
/// An `InOut` parameter, updated in place.
pub const TILEFISH_ROLE_INOUT: i32 = 2;
/// A parameter bound once (`tilefish_plan_run_bind_weights`), read.
pub const TILEFISH_ROLE_WEIGHT: i32 = 3;
/// An intermediate, in the arena.
pub const TILEFISH_ROLE_INTERMEDIATE: i32 = 4;
/// A device scalar, in the arena (its value is given by name at a call).
pub const TILEFISH_ROLE_DEVSCALAR: i32 = 5;
/// The cuBLASLt (and global scratch) workspace, in the arena.
pub const TILEFISH_ROLE_WORKSPACE: i32 = 6;
/// A carried buffer: in the arena's carried region, handed from plan to plan of a shared arena (never given by the caller).
pub const TILEFISH_ROLE_CARRIED: i32 = 7;

fn role_code(role: BufferRole) -> i32 {
    match role {
        BufferRole::Input => TILEFISH_ROLE_INPUT,
        BufferRole::Output => TILEFISH_ROLE_OUTPUT,
        BufferRole::Inout => TILEFISH_ROLE_INOUT,
        BufferRole::Weight => TILEFISH_ROLE_WEIGHT,
        BufferRole::Intermediate => TILEFISH_ROLE_INTERMEDIATE,
        BufferRole::Devscalar => TILEFISH_ROLE_DEVSCALAR,
        BufferRole::Workspace => TILEFISH_ROLE_WORKSPACE,
        BufferRole::Carried => TILEFISH_ROLE_CARRIED,
    }
}

/// A buffer of a plan (`tilefish_plan_buffer`).
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct tilefish_buffer_info {
    /// The name, owned by the plan handle.
    pub name: *const c_char,
    /// The role (`TILEFISH_ROLE_*`).
    pub role: i32,
    /// The element type (`TILEFISH_DTYPE_*`).
    pub dtype: i32,
    /// The number of dimensions.
    pub rank: u32,
    /// The bytes of the buffer at its largest bucket.
    pub bytes: u64,
}

/// A launch plan (`plan.json`), verified against its pin, the manifest next to it and every rule of `docs/plan-schema-1.md`.
#[allow(non_camel_case_types)]
pub struct tilefish_plan {
    pub(crate) plan: LaunchPlan,
    digest: CString,
    manifest_digest: CString,
    graph: CString,
    arch: CString,
    label: CString,
    buffers: Vec<CString>,
    outputs: Vec<CString>,
    dims: Vec<(CString, Vec<u64>)>,
}

impl tilefish_plan {
    fn new(plan: LaunchPlan) -> Self {
        Self {
            digest: c_string(plan.digest()),
            manifest_digest: c_string(plan.manifest_digest()),
            graph: c_string(plan.graph()),
            arch: c_string(plan.arch()),
            label: c_string(plan.label()),
            buffers: plan.buffers().iter().map(|b| c_string(&b.name)).collect(),
            outputs: plan.outputs().iter().map(|o| c_string(o)).collect(),
            dims: plan.dims().iter().map(|(d, v)| (c_string(d), v.clone())).collect(),
            plan,
        }
    }
}

/// Plans exported together over one manifest (`plans.json`), loaded with one pin.
#[allow(non_camel_case_types)]
pub struct tilefish_plan_set {
    set: PlanSet,
    digest: CString,
    names: Vec<CString>,
}

/// An arena shared by plans that run one at a time.
#[allow(non_camel_case_types)]
pub struct tilefish_shared_arena {
    arena: SharedArena,
    ctx: Arc<Ctx>,
}

/// Plans of a set instantiated on one shared arena.
#[allow(non_camel_case_types)]
pub struct tilefish_set_run {
    run: SetRun,
    ctx: Arc<Ctx>,
}

/// A plan instantiated in a context: its binaries loaded, its arena allocated (or shared).
#[allow(non_camel_case_types)]
pub struct tilefish_plan_run {
    run: PlanRun,
    ctx: Arc<Ctx>,
}

/// A plan run with its weights bound once.
#[allow(non_camel_case_types)]
pub struct tilefish_weighted_run {
    inner: Arc<Weighted>,
}

struct Weighted {
    // first: it borrows the shapes of `_weights`
    run: WeightedRun<'static>,
    _weights: OwnedNamed,
    ctx: Arc<Ctx>,
}

// SAFETY: a `WeightedRun` is not `Send` or `Sync` only because a `TensorArg` may hold a borrow of a cudarc slice (a trait object of the crate's
// that is neither). Every tensor here is a region the C caller vouches for (`DevicePtr::from_raw`: plain numbers, and the shapes in
// `_weights`, which are never written); the rest is a `PlanRun`, which is `Send + Sync`, and the crate orders concurrent executions itself.
unsafe impl Send for Weighted {}
// SAFETY: as above.
unsafe impl Sync for Weighted {}

/// A tap of a capture: its buffer, the step it follows, its tensor.
type TapItem = (String, Option<usize>, tilefish_tensor);

/// A plan captured in a CUDA graph for one bucket: it owns the graph and holds the run (and its arena).
#[allow(non_camel_case_types)]
pub struct tilefish_captured_plan {
    // first: it borrows the shapes below and, for a weighted run's capture, the weighted run
    plan: CapturedPlan<'static>,
    _externals: OwnedNamed,
    _taps: Vec<OwnedDims>,
    _weighted: Option<Arc<Weighted>>,
    ctx: Arc<Ctx>,
}

// -- plans --------------------------------------------------------------------------------------------------------------------------------

/// Load and verify the plan in `path` (a plan directory or its `plan.json`; the manifest next to it is loaded and checked too), pinned to
/// `pin`, the plan's digest. Needs no driver.
///
/// # Safety
///
/// `path` and `pin` are NUL-terminated strings; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_load(
    path: *const c_char,
    pin: *const c_char,
    out: *mut *mut tilefish_plan,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let (path, pin) = unsafe { (str_arg(path, "the path")?, str_arg(pin, "the pin")?) };
        let plan = LaunchPlan::load(path, Pin::Digest(pin))?;
        unsafe { put(out, tilefish_plan::new(plan)) }
    })
}

/// Load and verify a plan trusting its directory's own digests: for development.
///
/// # Safety
///
/// `path` is a NUL-terminated string; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_load_trusting_directory(
    path: *const c_char,
    out: *mut *mut tilefish_plan,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let path = unsafe { str_arg(path, "the path") }?;
        let plan = LaunchPlan::load(path, Pin::TrustDirectory)?;
        unsafe { put(out, tilefish_plan::new(plan)) }
    })
}

/// Load and verify the plan in `path` against a manifest already loaded (which must be the one the plan was built against), pinned to `pin`.
///
/// # Safety
///
/// `path` and `pin` are NUL-terminated strings; `manifest` is a live handle; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_load_with_manifest(
    path: *const c_char,
    manifest: *const tilefish_manifest,
    pin: *const c_char,
    out: *mut *mut tilefish_plan,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let manifest = unsafe { handle(manifest, "the manifest") }?;
        let (path, pin) = unsafe { (str_arg(path, "the path")?, str_arg(pin, "the pin")?) };
        let plan = LaunchPlan::load_with_manifest(path, &manifest.manifest, Pin::Digest(pin))?;
        unsafe { put(out, tilefish_plan::new(plan)) }
    })
}

/// A plan whose `plan.json` is in memory (`len` bytes), over a manifest already loaded, pinned to `pin`. The bytes are copied.
///
/// # Safety
///
/// `plan` points at `len` bytes; `manifest` is a live handle; `pin` is a NUL-terminated string; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_from_embedded(
    plan: *const u8,
    len: usize,
    manifest: *const tilefish_manifest,
    pin: *const c_char,
    out: *mut *mut tilefish_plan,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        if plan.is_null() {
            return invalid("the plan bytes are NULL");
        }
        let bytes = unsafe { std::slice::from_raw_parts(plan, len) };
        let manifest = unsafe { handle(manifest, "the manifest") }?;
        let pin = unsafe { str_arg(pin, "the pin") }?;
        let plan = LaunchPlan::from_embedded(bytes, manifest.manifest.clone(), Pin::Digest(pin))?;
        unsafe { put(out, tilefish_plan::new(plan)) }
    })
}

macro_rules! plan_string {
    ($name:ident, $field:ident, $doc:literal) => {
        #[doc = $doc]
        ///
        /// # Safety
        ///
        /// `plan` is NULL or a live handle.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(plan: *const tilefish_plan) -> *const c_char {
            // SAFETY: the caller's contract.
            quiet(std::ptr::null(), || unsafe { plan.as_ref() }.map_or(std::ptr::null(), |p| p.$field.as_ptr()))
        }
    };
}

plan_string!(tilefish_plan_digest, digest, "The digest of a plan, owned by the handle (NULL for NULL).");
plan_string!(
    tilefish_plan_manifest_digest,
    manifest_digest,
    "The digest of the plan's manifest, owned by the handle (NULL for NULL)."
);
plan_string!(tilefish_plan_graph, graph, "The name of the plan's graph, owned by the handle (NULL for NULL).");
plan_string!(
    tilefish_plan_arch,
    arch,
    "The architecture the plan is for (`\"sm_103\"`), owned by the handle (NULL for NULL)."
);
plan_string!(tilefish_plan_label, label, "Where the plan was read from, owned by the handle (NULL for NULL).");

/// The bytes of the plan's own arena (0 for NULL).
///
/// # Safety
///
/// `plan` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_arena_bytes(plan: *const tilefish_plan) -> u64 {
    // SAFETY: the caller's contract.
    quiet(0, || unsafe { plan.as_ref() }.map_or(0, |p| p.plan.arena_bytes()))
}

/// The number of buffers of the plan (0 for NULL).
///
/// # Safety
///
/// `plan` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_buffer_count(plan: *const tilefish_plan) -> usize {
    // SAFETY: the caller's contract.
    quiet(0, || unsafe { plan.as_ref() }.map_or(0, |p| p.buffers.len()))
}

/// Buffer `index` of the plan.
///
/// # Safety
///
/// `plan` is a live handle; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_buffer(
    plan: *const tilefish_plan,
    index: usize,
    out: *mut tilefish_buffer_info,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let plan = unsafe { handle(plan, "the plan") }?;
        let Some(b) = plan.plan.buffers().get(index) else {
            return invalid(format!("the plan has {} buffers, not {}", plan.buffers.len(), index + 1));
        };
        let info = tilefish_buffer_info {
            name: plan.buffers[index].as_ptr(),
            role: role_code(b.role),
            dtype: dtype_code(b.dtype),
            rank: b.shape.len() as u32,
            bytes: b.bytes,
        };
        unsafe { put_value(out, info) }
    })
}

/// The number of values the plan's graph returns (0 for NULL).
///
/// # Safety
///
/// `plan` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_output_count(plan: *const tilefish_plan) -> usize {
    // SAFETY: the caller's contract.
    quiet(0, || unsafe { plan.as_ref() }.map_or(0, |p| p.outputs.len()))
}

/// The name of returned value `index` (also given as `out<index>`), owned by the handle, or NULL past the last.
///
/// # Safety
///
/// `plan` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_output(plan: *const tilefish_plan, index: usize) -> *const c_char {
    // SAFETY: the caller's contract.
    quiet(std::ptr::null(), || {
        unsafe { plan.as_ref() }.and_then(|p| p.outputs.get(index)).map_or(std::ptr::null(), |n| n.as_ptr())
    })
}

/// The number of dimensions of the plan (0 for NULL).
///
/// # Safety
///
/// `plan` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_dim_count(plan: *const tilefish_plan) -> usize {
    // SAFETY: the caller's contract.
    quiet(0, || unsafe { plan.as_ref() }.map_or(0, |p| p.dims.len()))
}

/// Dimension `index` of the plan: its name and the sizes planned for it (both owned by the handle).
///
/// # Safety
///
/// `plan` is a live handle; the out-pointers are writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_dim(
    plan: *const tilefish_plan,
    index: usize,
    name: *mut *const c_char,
    values: *mut *const u64,
    n_values: *mut usize,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let plan = unsafe { handle(plan, "the plan") }?;
        let Some((d, v)) = plan.dims.get(index) else {
            return invalid(format!("the plan has {} dimensions, not {}", plan.dims.len(), index + 1));
        };
        unsafe { put_value(name, d.as_ptr()) }?;
        unsafe { put_value(values, v.as_ptr()) }?;
        unsafe { put_value(n_values, v.len()) }
    })
}

/// The manifest of a plan, as a new handle.
///
/// # Safety
///
/// `plan` is a live handle; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_manifest(
    plan: *const tilefish_plan,
    out: *mut *mut tilefish_manifest,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let plan = unsafe { handle(plan, "the plan") }?;
        unsafe { put(out, tilefish_manifest::new(plan.plan.manifest().clone())) }
    })
}

/// Destroy a plan handle (NULL is a no-op). Runs of it live on.
///
/// # Safety
///
/// `plan` is NULL or a live handle, not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_destroy(plan: *mut tilefish_plan) {
    // SAFETY: the caller's contract.
    quiet((), || unsafe { destroy_keeping_current(plan) })
}

// -- plan sets ----------------------------------------------------------------------------------------------------------------------------

impl tilefish_plan_set {
    fn new(set: PlanSet) -> Self {
        Self { digest: c_string(set.digest()), names: set.names().map(c_string).collect(), set }
    }
}

/// Load and verify the plan set in `directory` (its `plans.json` and the manifest), pinned to `pin`, the set's digest.
///
/// # Safety
///
/// `directory` and `pin` are NUL-terminated strings; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_set_load(
    directory: *const c_char,
    pin: *const c_char,
    out: *mut *mut tilefish_plan_set,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let (directory, pin) = unsafe { (str_arg(directory, "the directory")?, str_arg(pin, "the pin")?) };
        let set = PlanSet::load(directory, Pin::Digest(pin))?;
        unsafe { put(out, tilefish_plan_set::new(set)) }
    })
}

/// Load and verify a plan set trusting its directory's own digests: for development.
///
/// # Safety
///
/// `directory` is a NUL-terminated string; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_set_load_trusting_directory(
    directory: *const c_char,
    out: *mut *mut tilefish_plan_set,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let directory = unsafe { str_arg(directory, "the directory") }?;
        let set = PlanSet::load(directory, Pin::TrustDirectory)?;
        unsafe { put(out, tilefish_plan_set::new(set)) }
    })
}

/// The digest of a plan set, owned by the handle (NULL for NULL).
///
/// # Safety
///
/// `set` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_set_digest(set: *const tilefish_plan_set) -> *const c_char {
    // SAFETY: the caller's contract.
    quiet(std::ptr::null(), || unsafe { set.as_ref() }.map_or(std::ptr::null(), |s| s.digest.as_ptr()))
}

/// The number of plans of a set (0 for NULL).
///
/// # Safety
///
/// `set` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_set_name_count(set: *const tilefish_plan_set) -> usize {
    // SAFETY: the caller's contract.
    quiet(0, || unsafe { set.as_ref() }.map_or(0, |s| s.names.len()))
}

/// The graph name of plan `index` of a set, owned by the handle, or NULL past the last.
///
/// # Safety
///
/// `set` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_set_name(set: *const tilefish_plan_set, index: usize) -> *const c_char {
    // SAFETY: the caller's contract.
    quiet(std::ptr::null(), || {
        unsafe { set.as_ref() }.and_then(|s| s.names.get(index)).map_or(std::ptr::null(), |n| n.as_ptr())
    })
}

/// The plan of graph `name` of a set (it shares the set's manifest).
///
/// # Safety
///
/// `set` is a live handle; `name` is a NUL-terminated string; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_set_plan(
    set: *const tilefish_plan_set,
    name: *const c_char,
    out: *mut *mut tilefish_plan,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let set = unsafe { handle(set, "the plan set") }?;
        let name = unsafe { str_arg(name, "the graph") }?;
        let plan = set.set.plan(name)?;
        unsafe { put(out, tilefish_plan::new(plan)) }
    })
}

/// The plans of graphs `names` of a set, loaded on several threads (`tilefish::PlanSet::plans`), into `out[0 .. n]` (all NULL on a failure).
///
/// # Safety
///
/// `set` is a live handle; `names` points at `n` NUL-terminated strings; `out` points at `n` writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_set_plans(
    set: *const tilefish_plan_set,
    names: *const *const c_char,
    n: usize,
    out: *mut *mut tilefish_plan,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let set = unsafe { handle(set, "the plan set") }?;
        let names = unsafe { names_arg(names, n) }?;
        if n > 0 && out.is_null() {
            return invalid("the out-array is NULL");
        }
        for i in 0..n {
            unsafe { out.add(i).write(std::ptr::null_mut()) };
        }
        let plans = set.set.plans(&names)?;
        for (i, plan) in plans.into_iter().enumerate() {
            unsafe { out.add(i).write(Box::into_raw(Box::new(tilefish_plan::new(plan)))) };
        }
        Ok(())
    })
}

/// # Safety
///
/// `names` points at `n` NUL-terminated strings valid for `'a` (NULL when `n` is 0).
unsafe fn names_arg<'a>(names: *const *const c_char, n: usize) -> Outcome<Vec<&'a str>> {
    // SAFETY: the caller's contract.
    let items = unsafe { slice_arg(names, n, "names") }?;
    // SAFETY: as above.
    items.iter().map(|p| unsafe { str_arg(*p, "a graph name") }).collect()
}

/// The manifest of a plan set, as a new handle.
///
/// # Safety
///
/// `set` is a live handle; `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_set_manifest(
    set: *const tilefish_plan_set,
    out: *mut *mut tilefish_manifest,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let set = unsafe { handle(set, "the plan set") }?;
        unsafe { put(out, tilefish_manifest::new(set.set.manifest().clone())) }
    })
}

/// The bytes of one arena shared by the plans `names` of a set (the largest of theirs, and their carried buffers).
///
/// # Safety
///
/// `set` is a live handle; `names` points at `n` NUL-terminated strings; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_set_shared_arena_bytes(
    set: *const tilefish_plan_set,
    names: *const *const c_char,
    n: usize,
    out: *mut u64,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let set = unsafe { handle(set, "the plan set") }?;
        let names = unsafe { names_arg(names, n) }?;
        let bytes = set.set.shared_arena_bytes(&names)?;
        unsafe { put_value(out, bytes) }
    })
}

/// Instantiate the plans `names` of a set in `context` on one shared arena, allocated on `stream` (and freed on it when the last handle that
/// holds the arena is destroyed: the stream stays valid until then).
///
/// # Safety
///
/// `set` and `context` are live handles; `stream` is a valid stream of the context (or NULL) for as long as said above; `names` points at `n`
/// NUL-terminated strings; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_set_instantiate(
    set: *const tilefish_plan_set,
    context: *const tilefish_context,
    stream: CUstream,
    names: *const *const c_char,
    n: usize,
    out: *mut *mut tilefish_set_run,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let set = unsafe { handle(set, "the plan set") }?;
        let context = unsafe { handle(context, "the context") }?;
        let names = unsafe { names_arg(names, n) }?;
        let _keep = KeepCurrent::new()?;
        let home = unsafe { context.inner.borrow(stream) };
        let run = set.set.instantiate(&home, &names)?;
        unsafe { put(out, tilefish_set_run { run, ctx: Arc::clone(&context.inner) }) }
    })
}

/// Destroy a plan set handle (NULL is a no-op). Plans and runs of it live on.
///
/// # Safety
///
/// `set` is NULL or a live handle, not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_set_destroy(set: *mut tilefish_plan_set) {
    // SAFETY: the caller's contract.
    quiet((), || unsafe { destroy_keeping_current(set) })
}

/// The run of graph `name` of a set run, on the shared arena, as a new handle.
///
/// # Safety
///
/// `set_run` is a live handle; `name` is a NUL-terminated string; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_set_run_plan(
    set_run: *const tilefish_set_run,
    name: *const c_char,
    out: *mut *mut tilefish_plan_run,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let set_run = unsafe { handle(set_run, "the set run") }?;
        let name = unsafe { str_arg(name, "the graph") }?;
        let run = set_run.run.run(name)?.clone();
        unsafe { put(out, tilefish_plan_run { run, ctx: Arc::clone(&set_run.ctx) }) }
    })
}

/// The shared arena of a set run, as a new handle.
///
/// # Safety
///
/// `set_run` is a live handle; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_set_run_arena(
    set_run: *const tilefish_set_run,
    out: *mut *mut tilefish_shared_arena,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let set_run = unsafe { handle(set_run, "the set run") }?;
        unsafe { put(out, tilefish_shared_arena { arena: set_run.run.arena().clone(), ctx: Arc::clone(&set_run.ctx) }) }
    })
}

/// Destroy a set run handle (NULL is a no-op). The arena is freed with the last handle that holds it, after its last execution.
///
/// # Safety
///
/// `set_run` is NULL or a live handle, not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_set_run_destroy(set_run: *mut tilefish_set_run) {
    // SAFETY: the caller's contract.
    quiet((), || unsafe { destroy_keeping_current(set_run) })
}

// -- shared arenas ------------------------------------------------------------------------------------------------------------------------

/// Allocate one arena for the plans `plans` (`n` of them) in `context`, on `stream` (which stays valid until the arena is freed with the last
/// handle that holds it). Instantiate each plan on it with `tilefish_plan_instantiate_in`.
///
/// # Safety
///
/// `context` is a live handle; `stream` is a valid stream of it (or NULL) as said above; `plans` points at `n` live plan handles; `out` is
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_shared_arena_create(
    context: *const tilefish_context,
    stream: CUstream,
    plans: *const *const tilefish_plan,
    n: usize,
    out: *mut *mut tilefish_shared_arena,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let context = unsafe { handle(context, "the context") }?;
        let items = unsafe { slice_arg(plans, n, "plans") }?;
        let plans =
            items.iter().map(|p| unsafe { handle(*p, "a plan") }.map(|p| &p.plan)).collect::<Outcome<Vec<_>>>()?;
        let _keep = KeepCurrent::new()?;
        let home = unsafe { context.inner.borrow(stream) };
        let arena = SharedArena::new(&home, &plans)?;
        unsafe { put(out, tilefish_shared_arena { arena, ctx: Arc::clone(&context.inner) }) }
    })
}

/// The bytes of a shared arena (0 for NULL).
///
/// # Safety
///
/// `arena` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_shared_arena_bytes(arena: *const tilefish_shared_arena) -> u64 {
    // SAFETY: the caller's contract.
    quiet(0, || unsafe { arena.as_ref() }.map_or(0, |a| a.arena.bytes()))
}

/// The device address of a shared arena (0 for NULL).
///
/// # Safety
///
/// `arena` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_shared_arena_address(arena: *const tilefish_shared_arena) -> u64 {
    // SAFETY: the caller's contract.
    quiet(0, || unsafe { arena.as_ref() }.map_or(0, |a| a.arena.address()))
}

/// Destroy a shared arena handle (NULL is a no-op). The arena is freed with the last handle that holds it.
///
/// # Safety
///
/// `arena` is NULL or a live handle, not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_shared_arena_destroy(arena: *mut tilefish_shared_arena) {
    // SAFETY: the caller's contract.
    quiet((), || unsafe { destroy_keeping_current(arena) })
}

// -- runs ---------------------------------------------------------------------------------------------------------------------------------

/// Instantiate a plan in `context`: load its binaries and allocate its arena on `stream` (which stays valid until the arena is freed with the
/// last handle that holds it). `buckets` (`n_buckets` of them) is one bucket combination; none instantiates every bucket.
///
/// # Safety
///
/// `plan` and `context` are live handles; `stream` is a valid stream of the context (or NULL) as said above; `buckets` points at `n_buckets`
/// buckets (NULL when 0); `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_instantiate(
    plan: *const tilefish_plan,
    context: *const tilefish_context,
    stream: CUstream,
    buckets: *const tilefish_bucket,
    n_buckets: usize,
    out: *mut *mut tilefish_plan_run,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let plan = unsafe { handle(plan, "the plan") }?;
        let context = unsafe { handle(context, "the context") }?;
        let buckets = unsafe { buckets_arg(buckets, n_buckets) }?;
        let _keep = KeepCurrent::new()?;
        let home = unsafe { context.inner.borrow(stream) };
        let run = if buckets.is_empty() {
            plan.plan.instantiate_all(&home)?
        } else {
            plan.plan.instantiate(&home, &buckets)?
        };
        unsafe { put(out, tilefish_plan_run { run, ctx: Arc::clone(&context.inner) }) }
    })
}

/// Instantiate a plan on a shared arena (made for it): every execution of the run is ordered with those of the other plans on the arena.
///
/// # Safety
///
/// As for `tilefish_plan_instantiate`, with a live arena handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_instantiate_in(
    plan: *const tilefish_plan,
    arena: *const tilefish_shared_arena,
    buckets: *const tilefish_bucket,
    n_buckets: usize,
    out: *mut *mut tilefish_plan_run,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let plan = unsafe { handle(plan, "the plan") }?;
        let arena = unsafe { handle(arena, "the arena") }?;
        let buckets = unsafe { buckets_arg(buckets, n_buckets) }?;
        let _keep = KeepCurrent::new()?;
        let run = if buckets.is_empty() {
            plan.plan.instantiate_all_in(&arena.arena)?
        } else {
            plan.plan.instantiate_in(&arena.arena, &buckets)?
        };
        unsafe { put(out, tilefish_plan_run { run, ctx: Arc::clone(&arena.ctx) }) }
    })
}

/// The bytes of a run's arena (0 for NULL).
///
/// # Safety
///
/// `run` is NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_run_arena_bytes(run: *const tilefish_plan_run) -> u64 {
    // SAFETY: the caller's contract.
    quiet(0, || unsafe { run.as_ref() }.map_or(0, |r| r.run.arena_bytes()))
}

/// The taps of a launch, over the caller's arrays.
///
/// # Safety
///
/// `p` points at `n` taps valid for `'a` (NULL when 0), each target a region the caller vouches for.
unsafe fn taps_arg<'a>(p: *const tilefish_tap, n: usize) -> Outcome<Vec<Tap<'a>>> {
    // SAFETY: the caller's contract.
    let items = unsafe { slice_arg(p, n, "taps") }?;
    items
        .iter()
        .map(|t| {
            // SAFETY: as above.
            let buffer = unsafe { str_arg(t.buffer, "the buffer of a tap") }?;
            // SAFETY: as above.
            let into = unsafe { tensor_arg(&t.into, &format!("the tap of '{buffer}'")) }?;
            Ok(Tap { buffer, after: usize::try_from(t.after).ok(), into })
        })
        .collect()
}

/// Check a call of a run as `tilefish_plan_run_launch` does, and launch nothing.
///
/// # Safety
///
/// `run` is a live handle; `externals` points at `n_externals` named tensors and `devscalars` at `n_devscalars` values, valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_run_check(
    run: *const tilefish_plan_run,
    externals: *const tilefish_named_tensor,
    n_externals: usize,
    devscalars: *const tilefish_devscalar,
    n_devscalars: usize,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let run = unsafe { handle(run, "the run") }?;
        let externals = unsafe { named_arg(externals, n_externals, "externals") }?;
        let devscalars = unsafe { devscalars_arg(devscalars, n_devscalars) }?;
        Ok(run.run.check(&externals, &devscalars)?)
    })
}

/// Run the plan on `stream`: the externals by name (every input, weight, in-place tensor and output; never a carried buffer) and a value for
/// every device scalar. Everything is checked before anything is enqueued (a refusal launches and writes nothing); then the device scalars are
/// written and the steps launched in order, allocating nothing. Refused on a stream that is capturing (capture with `tilefish_plan_run_capture`).
///
/// # Safety
///
/// `run` is a live handle; `stream` is a valid stream of its context (or NULL) for the call; `externals` points at `n_externals` named tensors
/// (each a region the caller vouches for) and `devscalars` at `n_devscalars` values, valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_run_launch(
    run: *const tilefish_plan_run,
    stream: CUstream,
    externals: *const tilefish_named_tensor,
    n_externals: usize,
    devscalars: *const tilefish_devscalar,
    n_devscalars: usize,
) -> tilefish_status {
    // SAFETY: the caller's contract (no taps).
    unsafe {
        tilefish_plan_run_launch_tapped(
            run,
            stream,
            externals,
            n_externals,
            devscalars,
            n_devscalars,
            std::ptr::null(),
            0,
        )
    }
}

/// `tilefish_plan_run_launch` with debug taps: each tap copies a buffer of the plan into the caller's tensor right after a step.
///
/// # Safety
///
/// As for `tilefish_plan_run_launch`, and `taps` points at `n_taps` taps valid for the call.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tilefish_plan_run_launch_tapped(
    run: *const tilefish_plan_run,
    stream: CUstream,
    externals: *const tilefish_named_tensor,
    n_externals: usize,
    devscalars: *const tilefish_devscalar,
    n_devscalars: usize,
    taps: *const tilefish_tap,
    n_taps: usize,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let run = unsafe { handle(run, "the run") }?;
        let mut externals = unsafe { named_arg(externals, n_externals, "externals") }?;
        let devscalars = unsafe { devscalars_arg(devscalars, n_devscalars) }?;
        let mut taps = unsafe { taps_arg(taps, n_taps) }?;
        let _keep = KeepCurrent::new()?;
        let stream = unsafe { run.ctx.borrow(stream) };
        Ok(run.run.launch_tapped(&stream, &mut externals, &devscalars, &mut taps)?)
    })
}

/// Copies of the taps of a capture: the crate keeps their tensors for the life of the graph.
///
/// # Safety
///
/// As for `taps_arg` (for the call), and the returned dims outlive the tensors.
unsafe fn owned_taps(p: *const tilefish_tap, n: usize) -> Outcome<(Vec<OwnedDims>, Vec<TapItem>)> {
    // SAFETY: the caller's contract.
    let items = unsafe { slice_arg(p, n, "taps") }?;
    let mut dims = Vec::with_capacity(n);
    let mut taps = Vec::with_capacity(n);
    for t in items {
        // SAFETY: as above.
        let buffer = unsafe { str_arg(t.buffer, "the buffer of a tap") }?.to_owned();
        // SAFETY: as above.
        dims.push(unsafe { OwnedDims::of(&t.into, &format!("the tap of '{buffer}'")) }?);
        taps.push((buffer, usize::try_from(t.after).ok(), t.into));
    }
    Ok((dims, taps))
}

/// Capture one execution through `capture`, with the externals and taps copied into the handle.
///
/// # Safety
///
/// The arguments are valid for the call, every region is one the caller vouches for until the captured plan is destroyed, and `weighted` (when
/// given) is the weighted run `capture` borrows.
#[allow(clippy::too_many_arguments)]
unsafe fn capture_into(
    ctx: &Arc<Ctx>,
    externals: *const tilefish_named_tensor,
    n_externals: usize,
    devscalars: *const tilefish_devscalar,
    n_devscalars: usize,
    taps: *const tilefish_tap,
    n_taps: usize,
    weighted: Option<Arc<Weighted>>,
    out: *mut *mut tilefish_captured_plan,
    capture: impl FnOnce(
        Vec<(&str, TensorArg<'static>)>,
        &[(&str, tilefish::DevScalar)],
        Vec<Tap<'static>>,
    ) -> Outcome<CapturedPlan<'static>>,
) -> Outcome {
    // SAFETY: the caller's contract, for every argument below.
    let out = unsafe { out_arg(out) }?;
    let owned = unsafe { OwnedNamed::of(externals, n_externals, "externals") }?;
    let devscalars = unsafe { devscalars_arg(devscalars, n_devscalars) }?;
    let (tap_dims, tap_items) = unsafe { owned_taps(taps, n_taps) }?;
    // the tensors over the copies, which the handle keeps (and drops after the graph)
    let tensors = unsafe { owned.tensors_static() }?;
    let mut tap_args: Vec<Tap<'static>> = Vec::with_capacity(tap_items.len());
    for ((buffer, after, t), dims) in tap_items.iter().zip(&tap_dims) {
        let into: TensorArg<'static> = unsafe { dims.tensor(t, &format!("the tap of '{buffer}'")) }?;
        // `Tap` borrows its buffer name for as long as its tensor, but the crate copies the name: `tap_items` outlives the call, which is
        // all the name needs
        let buffer: &'static str = unsafe { &*std::ptr::from_ref::<str>(buffer.as_str()) };
        tap_args.push(Tap { buffer, after: *after, into });
    }
    let _keep = KeepCurrent::new()?;
    let plan = capture(tensors, &devscalars, tap_args)?;
    drop(tap_items);
    unsafe {
        put(
            out,
            tilefish_captured_plan {
                plan,
                _externals: owned,
                _taps: tap_dims,
                _weighted: weighted,
                ctx: Arc::clone(ctx),
            },
        )
    }
}

/// Capture one execution of the plan in a CUDA graph on `stream` (which must not be capturing already), for the bucket the externals select:
/// everything is checked first, nothing runs, and nothing is allocated. `devscalars` are the values the first replay reads. The captured plan
/// owns the graph and holds the run; the externals' memory must stay allocated until it is destroyed (a replay launches on their addresses).
/// The stream is used for the call only.
///
/// # Safety
///
/// As for `tilefish_plan_run_launch`, the regions vouched for until the captured plan is destroyed; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_run_capture(
    run: *const tilefish_plan_run,
    stream: CUstream,
    externals: *const tilefish_named_tensor,
    n_externals: usize,
    devscalars: *const tilefish_devscalar,
    n_devscalars: usize,
    out: *mut *mut tilefish_captured_plan,
) -> tilefish_status {
    // SAFETY: the caller's contract (no taps).
    unsafe {
        tilefish_plan_run_capture_tapped(
            run,
            stream,
            externals,
            n_externals,
            devscalars,
            n_devscalars,
            std::ptr::null(),
            0,
            out,
        )
    }
}

/// `tilefish_plan_run_capture` with debug taps: each tap's copy is a node of the graph, made again at every replay.
///
/// # Safety
///
/// As for `tilefish_plan_run_capture`, and `taps` points at `n_taps` taps whose targets are vouched for until the captured plan is destroyed.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tilefish_plan_run_capture_tapped(
    run: *const tilefish_plan_run,
    stream: CUstream,
    externals: *const tilefish_named_tensor,
    n_externals: usize,
    devscalars: *const tilefish_devscalar,
    n_devscalars: usize,
    taps: *const tilefish_tap,
    n_taps: usize,
    out: *mut *mut tilefish_captured_plan,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let run = unsafe { handle(run, "the run") }?;
        let stream = unsafe { run.ctx.borrow(stream) };
        unsafe {
            capture_into(
                &run.ctx,
                externals,
                n_externals,
                devscalars,
                n_devscalars,
                taps,
                n_taps,
                None,
                out,
                |ext, dev, taps| Ok(run.run.capture_tapped(&stream, ext, dev, taps)?),
            )
        }
    })
}

/// Bind the weights of a plan once (by name; a plan whose graph declares its roles): the weighted run's launches and captures take the other
/// externals only. The weights' memory must stay allocated until the weighted run and every capture of it are destroyed.
///
/// # Safety
///
/// `run` is a live handle; `weights` points at `n_weights` named tensors whose regions are vouched for as said above; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_run_bind_weights(
    run: *const tilefish_plan_run,
    weights: *const tilefish_named_tensor,
    n_weights: usize,
    out: *mut *mut tilefish_weighted_run,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let out = unsafe { out_arg(out) }?;
        let run = unsafe { handle(run, "the run") }?;
        let owned = unsafe { OwnedNamed::of(weights, n_weights, "weights") }?;
        let tensors = unsafe { owned.tensors_static() }?;
        let _keep = KeepCurrent::new()?;
        let weighted = run.run.bind_weights(tensors)?;
        let inner = Arc::new(Weighted { run: weighted, _weights: owned, ctx: Arc::clone(&run.ctx) });
        unsafe { put(out, tilefish_weighted_run { inner }) }
    })
}

/// Destroy a run handle (NULL is a no-op). The arena is freed with the last handle that holds it, after its last execution.
///
/// # Safety
///
/// `run` is NULL or a live handle, not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_plan_run_destroy(run: *mut tilefish_plan_run) {
    // SAFETY: the caller's contract.
    quiet((), || unsafe { destroy_keeping_current(run) })
}

// -- weighted runs ------------------------------------------------------------------------------------------------------------------------

/// `tilefish_plan_run_check` of a weighted run (without the weights).
///
/// # Safety
///
/// As for `tilefish_plan_run_check`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_weighted_run_check(
    run: *const tilefish_weighted_run,
    externals: *const tilefish_named_tensor,
    n_externals: usize,
    devscalars: *const tilefish_devscalar,
    n_devscalars: usize,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let run = unsafe { handle(run, "the weighted run") }?;
        let externals = unsafe { named_arg(externals, n_externals, "externals") }?;
        let devscalars = unsafe { devscalars_arg(devscalars, n_devscalars) }?;
        Ok(run.inner.run.check(&externals, &devscalars)?)
    })
}

/// `tilefish_plan_run_launch_tapped` of a weighted run (without the weights; `taps` may be NULL with `n_taps` 0).
///
/// # Safety
///
/// As for `tilefish_plan_run_launch_tapped`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tilefish_weighted_run_launch(
    run: *const tilefish_weighted_run,
    stream: CUstream,
    externals: *const tilefish_named_tensor,
    n_externals: usize,
    devscalars: *const tilefish_devscalar,
    n_devscalars: usize,
    taps: *const tilefish_tap,
    n_taps: usize,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let run = unsafe { handle(run, "the weighted run") }?;
        let mut externals = unsafe { named_arg(externals, n_externals, "externals") }?;
        let devscalars = unsafe { devscalars_arg(devscalars, n_devscalars) }?;
        let mut taps = unsafe { taps_arg(taps, n_taps) }?;
        let _keep = KeepCurrent::new()?;
        let stream = unsafe { run.inner.ctx.borrow(stream) };
        Ok(run.inner.run.launch_tapped(&stream, &mut externals, &devscalars, &mut taps)?)
    })
}

/// `tilefish_plan_run_capture_tapped` of a weighted run (without the weights; `taps` may be NULL with `n_taps` 0). The captured plan holds the
/// weighted run.
///
/// # Safety
///
/// As for `tilefish_plan_run_capture_tapped`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tilefish_weighted_run_capture(
    run: *const tilefish_weighted_run,
    stream: CUstream,
    externals: *const tilefish_named_tensor,
    n_externals: usize,
    devscalars: *const tilefish_devscalar,
    n_devscalars: usize,
    taps: *const tilefish_tap,
    n_taps: usize,
    out: *mut *mut tilefish_captured_plan,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let run = unsafe { handle(run, "the weighted run") }?;
        let inner = Arc::clone(&run.inner);
        // the captured plan holds `inner` (it is dropped after the graph), so the weighted run it borrows lives as long as it does
        let weighted: &'static WeightedRun<'static> = unsafe { &*std::ptr::from_ref(&inner.run) };
        let stream = unsafe { inner.ctx.borrow(stream) };
        let ctx = Arc::clone(&inner.ctx);
        unsafe {
            capture_into(
                &ctx,
                externals,
                n_externals,
                devscalars,
                n_devscalars,
                taps,
                n_taps,
                Some(inner),
                out,
                |ext, dev, taps| Ok(weighted.capture_tapped(&stream, ext, dev, taps)?),
            )
        }
    })
}

/// Destroy a weighted run handle (NULL is a no-op). Its captures hold what they need of it.
///
/// # Safety
///
/// `run` is NULL or a live handle, not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_weighted_run_destroy(run: *mut tilefish_weighted_run) {
    // SAFETY: the caller's contract.
    quiet((), || unsafe { destroy_keeping_current(run) })
}

// -- captured plans -----------------------------------------------------------------------------------------------------------------------

/// Replay a captured plan on `stream` (a stream of its context: the one it was captured on, or another). The device scalars are written first
/// when they changed. The arena is ordered with every other execution on it; the externals are the caller's to order (write new inputs on
/// `stream` before the replay, read the outputs after it). Refused while a stream captures.
///
/// A captured plan is used by one thread at a time (any thread).
///
/// # Safety
///
/// `captured` is a live handle, used by no other thread meanwhile; `stream` is a valid stream of its context (or NULL) for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_captured_plan_replay(
    captured: *mut tilefish_captured_plan,
    stream: CUstream,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let captured = unsafe { handle_mut(captured, "the captured plan") }?;
        let _keep = KeepCurrent::new()?;
        let stream = unsafe { captured.ctx.borrow(stream) };
        Ok(captured.plan.replay(&stream)?)
    })
}

/// Set the value of a device scalar for the next replays (written by the next replay, on its stream, before the graph).
///
/// # Safety
///
/// `captured` is a live handle, used by no other thread meanwhile; `value` points at a valid `tilefish_devscalar`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_captured_plan_set_devscalar(
    captured: *mut tilefish_captured_plan,
    value: *const tilefish_devscalar,
) -> tilefish_status {
    call(|| {
        // SAFETY: the caller's contract, for every argument below.
        let captured = unsafe { handle_mut(captured, "the captured plan") }?;
        let value = unsafe { handle(value, "the value") }?;
        let name = unsafe { str_arg(value.name, "the name of the device scalar") }?;
        let scalar = devscalar_of(value, name)?;
        Ok(captured.plan.set_devscalar(name, scalar)?)
    })
}

/// Destroy a captured plan (NULL is a no-op): its graph is destroyed (after the replays enqueued, which the driver keeps), and the run it holds
/// lives on while other handles hold it.
///
/// # Safety
///
/// `captured` is NULL or a live handle, not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tilefish_captured_plan_destroy(captured: *mut tilefish_captured_plan) {
    // SAFETY: the caller's contract.
    quiet((), || unsafe { destroy_keeping_current(captured) })
}
