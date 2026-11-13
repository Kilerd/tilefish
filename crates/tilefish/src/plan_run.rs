//! Running a launch plan: one arena, the steps in order, the device scalars, and a CUDA graph capture that cannot outlive what it captured.
//!
//! [`LaunchPlan::instantiate`] makes a [`PlanRun`]: every kernel and GEMM of the chosen bucket combinations loaded once (and each function
//! loaded eagerly, `cuFuncLoad`), the address-independent preconditions of every step checked (the binding of the dimensions, the exact
//! bucket of every entry, the grids against the device, the workspace a pinned algorithm needs), and the **arena** allocated once: a
//! `CudaSlice<u8>` of `arena_bytes` bytes on the stream the plan is instantiated with, which holds the intermediates, the slots of the device
//! scalars and the cuBLASLt workspace at the offsets of the plan. Nothing is allocated after that.
//!
//! [`PlanRun::launch`] takes the externals by name (tensors of cudarc slices, `TensorRef` for what the plan reads and `TensorMut` for what it
//! writes) and a value for every device scalar, binds the dimensions from the externals' shapes and selects the bucket (exactly one of the
//! plan's), checks every step of it with the real addresses (**before the first launch: a refusal launches nothing and writes nothing**),
//! writes the device scalars into their slots on the stream (asynchronous copies from pinned host memory the instantiation allocated, which
//! never wait for the device: a copy from pageable memory may, as CUDA documents it), and launches the steps in order.
//! The externals are ordered against other
//! streams as a single launch's tensors are ([`Kernel::launch`]): cudarc's tracking on the launch stream, each slice's own stream waiting for
//! the end of the plan, a written slice's own stream waited for first; and the arena is ordered the same way, as a slice that every launch
//! writes, owned by the stream it was allocated on: two launches of one instantiation on two streams run one after the other.
//!
//! [`PlanRun::capture`] makes a [`CapturedPlan`]: the same checks, then the steps captured in a CUDA graph on the stream. The graph holds
//! addresses: the arena's (the `CapturedPlan` holds the instantiation, so the arena outlives the graph) and the externals' (the `CapturedPlan`
//! **borrows** the externals for its lifetime `'ext`, so no slice the graph captured can be freed while the graph can be replayed: that is a
//! compile error). [`CapturedPlan::replay`] is safe, and ordered as a launch is. The externals keep their addresses; their contents may change
//! between replays: an external given as a `TensorMut` is written through [`CapturedPlan::external_mut`], and a device scalar is set with
//! [`CapturedPlan::set_devscalar`] (its slot is written on the stream right before the next replay, so a launch or another graph of the same
//! instantiation in between does not change what this graph reads).
//!
//! **The order of the runs of one instantiation** (`docs/plan-schema-1.md`, section 9, step 1: two executions that use one arena must not
//! overlap on the device). Every launch and every replay of an instantiation, whatever its stream and its thread, takes the arena's lock, makes
//! its stream wait for what the arena's own stream holds (the stream the arena was allocated on, which has waited for the end of every earlier
//! run made elsewhere; cudarc's read and write events on the arena are waited for too), and makes the arena's stream wait for an event recorded
//! after its last launch (also when a launch fails after others were enqueued). So the runs of one instantiation (launches, and replays of its
//! graphs of one bucket or several) run on the device one after the other, in the order they are issued: the contract of Python's *order
//! event* of an arena. Python captures its event's wait and record as event nodes, because a torch graph is replayed by whoever holds it; the
//! graph of a [`CapturedPlan`] is replayed only by [`CapturedPlan::replay`], which orders the replay as a launch, so it has no event node.
//! A capture ([`PlanRun::capture`]) holds the arena's lock from the beginning of the capture to its end, so no launch or replay of the
//! instantiation, on any thread, makes the arena's stream wait for an event while that stream is being captured.
//! Dropping the last handle of an instantiation waits for the arena's stream, so nothing of the plan runs when its arena is freed and its modules
//! unloaded.
//!
//! What is not this module's: the ordering of a use of an external that is not a launch of this crate (cudarc's tracking, or the host's), as
//! everywhere in the crate.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode, CUstreamCaptureStatus};
use cudarc::driver::{CudaContext, CudaEvent, CudaGraph, CudaSlice, CudaStream, CudaView, CudaViewMut};

use crate::error::{Error, PreconditionKind, Result};
use crate::expr::{Expr, MAX_DIMS, Pattern, Solved, Symbols, Unsolvable};
use crate::ffi;
use crate::gemm::{Call, Gemm};
use crate::kernel::{Arch, Kernel};
use crate::launch_plan::{BufferRole, LaunchPlan, StepKind, describe_values};
use crate::manifest::{Access, ParamKind, ShapeDim};
use crate::memory::sealed;
use crate::plan::Prepared;
use crate::tensor::{Backing, DType, TensorArg, TensorMut, U8};

/// The value of a device scalar (a buffer of role `devscalar`), which a launch writes into its slot of the arena.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DevScalar {
    /// An integer, for a device scalar of an integer dtype (`i32`, `i64`, `u8`): it must fit the dtype.
    Int(i64),
    /// A float, for a device scalar of dtype `f32`.
    F32(f32),
    /// The bits of one element of any dtype, in the low bytes (`bf16` 1.0 is `Bits(0x3f80)`): the bits above the element's size must be 0.
    Bits(u64),
}

impl From<i32> for DevScalar {
    fn from(v: i32) -> Self {
        Self::Int(i64::from(v))
    }
}

impl From<i64> for DevScalar {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<u8> for DevScalar {
    fn from(v: u8) -> Self {
        Self::Int(i64::from(v))
    }
}

impl From<f32> for DevScalar {
    fn from(v: f32) -> Self {
        Self::F32(v)
    }
}

impl DevScalar {
    /// The little-endian bytes of the value as one element of `dtype`.
    fn encode(self, dtype: DType, name: &str) -> Result<Vec<u8>> {
        let refused = |why: String| {
            Error::precondition(PreconditionKind::Shape, Some(name), format!("the device scalar '{name}' {why}"))
        };
        let range = match dtype {
            DType::I32 => Some((i64::from(i32::MIN), i64::from(i32::MAX))),
            DType::I64 => Some((i64::MIN, i64::MAX)),
            DType::U8 => Some((0, 255)),
            _ => None,
        };
        match (self, range) {
            (Self::Int(v), Some((lo, hi))) => {
                if !(lo..=hi).contains(&v) {
                    return Err(refused(format!(
                        "is {v}, which is not an integer that fits {} ({lo} to {hi})",
                        dtype.name()
                    )));
                }
                Ok(v.to_le_bytes()[..dtype.size()].to_vec())
            }
            (Self::F32(v), _) if dtype == DType::F32 => Ok(v.to_le_bytes().to_vec()),
            (Self::Bits(bits), _) => {
                let size = dtype.size();
                if size < 8 && bits >> (8 * size) != 0 {
                    return Err(refused(format!(
                        "is given the bits {bits:#x}, more than the {size} bytes of one {}",
                        dtype.name()
                    )));
                }
                Ok(bits.to_le_bytes()[..size].to_vec())
            }
            (other, _) => Err(refused(format!(
                "is {}, which is not a value of its dtype {} (an integer dtype takes DevScalar::Int, f32 DevScalar::F32, any dtype DevScalar::Bits)",
                match other {
                    Self::Int(v) => format!("the integer {v}"),
                    Self::F32(v) => format!("the f32 {v}"),
                    Self::Bits(b) => format!("the bits {b:#x}"),
                },
                dtype.name()
            ))),
        }
    }
}

/// A debug tap of a launch ([`PlanRun::launch_tapped`]): the buffer `buffer` of the plan (by name, or `out<i>` for a returned value), copied into
/// `into` right after the step `after` (by default the last step of the bucket that writes the buffer).
pub struct Tap<'a> {
    /// The buffer of the plan.
    pub buffer: &'a str,
    /// The index of the step of the bucket the copy follows, or `None` for the last step that writes the buffer.
    pub after: Option<usize>,
    /// Where the copy goes: a dense `TensorMut` of the buffer's dtype and shape at the bucket.
    pub into: TensorArg<'a>,
}

impl std::fmt::Debug for Tap<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tap").field("buffer", &self.buffer).field("after", &self.after).finish_non_exhaustive()
    }
}

/// The launch of one step, checked: what [`ffi::plan::enqueue_steps`] launches.
pub(crate) enum Ready<'r> {
    Kernel(&'r Kernel, Box<Prepared>),
    Gemm(&'r Gemm, Call),
    /// A copy step: `bytes` bytes from `src` to `dst`, checked to be inside their buffers and not to overlap.
    Copy {
        src: u64,
        dst: u64,
        bytes: usize,
    },
}

/// The value of one device scalar, at its slot of the arena.
#[derive(Debug, Clone)]
struct Slot {
    name: String,
    offset: u64,
    dtype: DType,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
enum Op {
    Kernel(Kernel),
    Gemm(Gemm),
    /// A copy step: no binary.
    Copy,
}

/// One pointer argument of a step at one bucket: the buffer, the shape, and the view (none: the whole buffer, dense).
#[derive(Debug)]
struct ArgRun {
    buffer: usize,
    shape: Vec<usize>,
    view: Option<(Vec<usize>, usize)>,
    dtype: DType,
    access: Access,
}

#[derive(Debug)]
struct StepRun {
    stem: String,
    op: Op,
    args: Vec<ArgRun>,
    /// A library step's workspace: its buffer and its shape (`[bytes]`).
    workspace: Option<(usize, [usize; 1])>,
    /// The buffers the step writes.
    written: Vec<usize>,
}

#[derive(Debug)]
struct BucketRun {
    /// The index of the combination in the plan.
    index: usize,
    steps: Vec<StepRun>,
}

#[derive(Debug)]
struct Arena {
    slice: Option<CudaSlice<u8>>,
    /// Whose device scalar values the slots hold: the id of a `CapturedPlan`, or 0 (a launch, or nothing yet).
    slots_of: u64,
    /// The pinned host memory the device scalars are copied from (a plan with device scalars only): after the slice, so that it is freed
    /// after the arena, when the instantiation has waited for its stream.
    staging: Option<ffi::plan::Staging>,
}

#[derive(Debug)]
struct Instance {
    plan: LaunchPlan,
    context: Arc<CudaContext>,
    /// The stream the arena was allocated on, and is freed on: a launch elsewhere is ordered against it, as a slice's own stream is.
    home: Arc<CudaStream>,
    device: i32,
    arena: Mutex<Arena>,
    arena_addr: u64,
    buckets: Vec<BucketRun>,
    by_name: HashMap<String, usize>,
    /// The dimensions, numbered in the order of the plan's `dims` (what the externals' shapes bind).
    syms: Symbols,
    /// The shape of every external, as patterns over the dimensions (empty for an arena buffer).
    patterns: Vec<Vec<Pattern>>,
    captures: AtomicU64,
}

impl Drop for Instance {
    fn drop(&mut self) {
        // The arena's own stream has waited for the end of every launch and replay of this instantiation made on another stream, and has the
        // ones made on it: when it is idle nothing of the plan runs, so unloading the modules, destroying the cuBLASLt handles and freeing the
        // arena (which the fields do next) cannot happen under a step. A failure cannot be reported from a destructor.
        let _ = self.home.synchronize();
    }
}

impl Instance {
    fn lock_arena(&self) -> MutexGuard<'_, Arena> {
        self.arena.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// A launch plan instantiated on a device: its kernels and GEMMs loaded, its arena allocated. Cheap to clone (the clones share the
/// instantiation and its arena); it can be shared between threads. See the module docs.
#[derive(Debug, Clone)]
pub struct PlanRun {
    inner: Arc<Instance>,
}

/// What a call binds: the bucket, the address of every buffer, the values of the device scalars.
struct Bound {
    /// The index into `Instance::buckets`.
    bucket: usize,
    bases: Vec<u64>,
    slots: Vec<Slot>,
}

impl LaunchPlan {
    /// Instantiate the plan for the bucket combination `values` (one value per dimension, `&[]` for a plan without dimensions): load every
    /// kernel and GEMM of its steps in the context of `stream`, check every step's preconditions that do not depend on an address, and
    /// allocate the arena on `stream` (which it is freed on; a launch on that stream needs no extra ordering). The device must be of the
    /// plan's architecture.
    pub fn instantiate(&self, stream: &Arc<CudaStream>, values: &[(&str, u64)]) -> Result<PlanRun> {
        let Some(index) = self.bucket_index(values) else {
            let all: Vec<String> = self.buckets().iter().map(|b| describe_values(&b.values)).collect();
            return Err(Error::precondition(
                PreconditionKind::Bucket,
                None,
                format!(
                    "the plan of graph '{}' has no bucket {values:?}: its buckets are {}",
                    self.graph(),
                    all.join(", ")
                ),
            ));
        };
        PlanRun::new(self, stream, &[index])
    }

    /// Instantiate the plan for every bucket combination: one arena serves all of them (a buffer is at the same offset in every bucket).
    pub fn instantiate_all(&self, stream: &Arc<CudaStream>) -> Result<PlanRun> {
        let all: Vec<usize> = (0..self.buckets().len()).collect();
        PlanRun::new(self, stream, &all)
    }
}

fn in_step(e: Error, what: &str) -> Error {
    match e {
        Error::Precondition { kind, arg, message } => {
            Error::Precondition { kind, arg, message: format!("{what}: {message}") }
        }
        Error::Launch(m) => Error::Launch(format!("{what}: {m}")),
        Error::Unsupported(m) => Error::Unsupported(format!("{what}: {m}")),
        other => other,
    }
}

/// Write `bytes` at byte `at` of written cudarc memory, on `stream`: an asynchronous copy from the pinned memory of `staging` (no
/// synchronization of the host, see [`ffi::plan::Staging`]), ordered after the memory's earlier uses on any stream and recorded as a write
/// by cudarc's tracking.
fn write_bytes(
    memory: &mut dyn sealed::Write,
    staging: &mut ffi::plan::Staging,
    stream: &Arc<CudaStream>,
    at: u64,
    bytes: &[u8],
) -> Result<()> {
    let missing = || Error::Launch("a slot of a device scalar is outside the arena".to_owned());
    let mut all = memory.bytes_mut().ok_or_else(missing)?;
    let at = usize::try_from(at).map_err(|_| missing())?;
    let mut part = all.try_slice_mut(at..at + bytes.len()).ok_or_else(missing)?;
    staging.copy(stream, bytes, &mut part)
}

/// The work of a launch or a replay, inside `ffi::ordered` (which refused a capturing stream and made `stream` wait for what a written slice's
/// own stream holds): the device scalars into their slots (the arena is the last tensor), cudarc's tracking of every slice on `stream` (a
/// wait for the earlier writes, and reads for a write; the use recorded when the guards are dropped, after the work), the work, and the event
/// after it when an owning stream is not `stream`.
fn tracked(
    stream: &Arc<CudaStream>,
    tensors: &mut [TensorArg<'_>],
    slots: &[Slot],
    staging: Option<&mut ffi::plan::Staging>,
    want_end: bool,
    work: impl FnOnce(&CudaStream) -> Result<()>,
) -> Result<Option<CudaEvent>> {
    if !slots.is_empty() {
        let (Some(TensorArg { backing: Backing::Write(arena), .. }), Some(staging)) = (tensors.last_mut(), staging)
        else {
            return Err(Error::Launch("a plan with device scalars has no arena".to_owned()));
        };
        for slot in slots {
            write_bytes(&mut **arena, staging, stream, slot.offset, &slot.bytes)?;
        }
    }
    let mut guards = Vec::with_capacity(tensors.len());
    for tensor in tensors.iter_mut() {
        match &mut tensor.backing {
            Backing::Raw => {}
            Backing::Read(memory) => guards.push(memory.sync_read(stream)),
            Backing::Write(memory) => guards.push(memory.sync_write(stream)),
        }
    }
    work(stream)?;
    let end = if want_end {
        Some(
            stream
                .record_event(Some(CUevent_flags::CU_EVENT_DISABLE_TIMING))
                .map_err(Error::driver("cuEventRecord"))?,
        )
    } else {
        None
    };
    drop(guards);
    Ok(end)
}

impl PlanRun {
    fn new(plan: &LaunchPlan, stream: &Arc<CudaStream>, buckets: &[usize]) -> Result<Self> {
        // the arena's stream orders every launch of the run against the others: it cannot be the per-thread default stream (`ffi::refuse_per_thread`)
        ffi::refuse_per_thread(stream, std::iter::empty())?;
        let ctx = Arc::clone(stream.context());
        let arch = Arch::of_context(&ctx)?;
        if arch.to_string() != plan.arch() {
            return Err(Error::Unsupported(format!(
                "the plan of graph '{}' was built for {}, but the CUDA device {} of the stream is {arch}",
                plan.graph(),
                plan.arch(),
                ctx.ordinal()
            )));
        }
        let status = stream.capture_status().map_err(Error::driver("cuStreamIsCapturing"))?;
        if status != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err(Error::Launch(
                "a plan is not instantiated while its stream captures a CUDA graph: its arena would be allocated inside the capture".to_owned(),
            ));
        }
        if plan.dims().len() > MAX_DIMS {
            return Err(Error::Unsupported(format!(
                "the plan of graph '{}' has {} dimensions; this host binds at most {MAX_DIMS}",
                plan.graph(),
                plan.dims().len()
            )));
        }
        let buffers = plan.buffers();
        let mut syms = Symbols::default();
        for (name, _) in plan.dims() {
            syms.intern(name).map_err(Error::Unsupported)?;
        }
        let mut patterns = Vec::with_capacity(buffers.len());
        for b in buffers {
            let mut axes = Vec::new();
            if b.role.is_external() {
                for (axis, d) in b.shape.iter().enumerate() {
                    axes.push(match d {
                        ShapeDim::Static(n) => Pattern::new(&n.to_string(), Expr::Num(i128::from(*n))),
                        ShapeDim::Expr(e) => {
                            Pattern::parse(e, &format!("the shape of '{}' axis {axis}", b.name), &mut syms)
                                .map_err(Error::Plan)?
                        }
                    });
                }
            }
            patterns.push(axes);
        }
        let by_name: HashMap<String, usize> = buffers.iter().enumerate().map(|(i, b)| (b.name.clone(), i)).collect();

        // -- every entry of the chosen buckets, loaded once (and its function loaded now, not inside a capture)
        let mut ops: HashMap<(bool, String), Op> = HashMap::new();
        let mut runs = Vec::with_capacity(buckets.len());
        for &index in buckets {
            let bucket = &plan.buckets()[index];
            let mut steps = Vec::with_capacity(bucket.steps.len());
            for st in &bucket.steps {
                let key = (st.kind == StepKind::Library, st.entry.clone());
                let op = match ops.get(&key) {
                    _ if st.kind == StepKind::Copy => Op::Copy, // a device-to-device copy: nothing to load
                    Some(op) => op.clone(),
                    None => {
                        let op = match st.kind {
                            StepKind::Kernel => {
                                let kernel = plan.manifest().kernel(&ctx, &st.entry, &[], arch)?;
                                ctx.bind_to_thread().map_err(Error::driver("cuCtxSetCurrent"))?;
                                ffi::plan::load_function(kernel.function())?;
                                Op::Kernel(kernel)
                            }
                            StepKind::Library => Op::Gemm(plan.manifest().gemm(&ctx, &st.entry, &[], arch)?),
                            StepKind::Copy => unreachable!("a copy step loads nothing"),
                        };
                        ops.insert(key, op.clone());
                        op
                    }
                };
                let args = st
                    .args
                    .iter()
                    .map(|a| {
                        let buffer = &buffers[a.buffer];
                        let (shape, view) = match &a.view {
                            Some(v) => (v.shape.clone(), Some((v.strides.clone(), v.offset))),
                            None => (bucket.shapes[a.buffer].clone(), None),
                        };
                        let access = if buffer.role.is_writable() { Access::InOut } else { Access::In };
                        ArgRun { buffer: a.buffer, shape, view, dtype: buffer.dtype, access }
                    })
                    .collect();
                let workspace = st.workspace_buffer.map(|w| (w, [bucket.shapes[w][0]]));
                // the buffers the step writes (what a tap without a step is copied after): a copy's dst, a GEMM's out, a kernel's out and
                // inout pointers (the reader checked the arguments are the entry's pointers, in order)
                let written: Vec<usize> = match st.kind {
                    StepKind::Copy => vec![st.args[1].buffer],
                    StepKind::Library => st.args.last().map(|a| a.buffer).into_iter().collect(),
                    StepKind::Kernel => {
                        let entry =
                            plan.manifest().entry(&st.entry, plan.arch()).expect("the reader checked the entry");
                        let pointers = entry.params.iter().filter_map(|p| match &p.kind {
                            ParamKind::Pointer { access, .. } => Some(*access),
                            _ => None,
                        });
                        st.args
                            .iter()
                            .zip(pointers)
                            .filter(|(_, access)| *access != Access::In)
                            .map(|(a, _)| a.buffer)
                            .collect()
                    }
                };
                steps.push(StepRun { stem: st.entry.clone(), op, args, workspace, written });
            }
            runs.push(BucketRun { index, steps });
        }

        // -- the arena, once
        let arena_bytes = usize::try_from(plan.arena_bytes())
            .map_err(|_| Error::Unsupported("the arena does not fit the address space".to_owned()))?;
        let slice = if arena_bytes > 0 {
            Some(stream.alloc_zeros::<u8>(arena_bytes).map_err(Error::driver("cuMemAllocAsync"))?)
        } else {
            None
        };
        let arena_addr = slice.as_ref().map_or(0, |s| sealed::Memory::region(s).addr);
        if !arena_addr.is_multiple_of(crate::launch_plan::ARENA_ALIGNMENT) {
            return Err(Error::Unsupported(format!(
                "the arena was allocated at {arena_addr:#x}, which is not aligned to {} bytes",
                crate::launch_plan::ARENA_ALIGNMENT
            )));
        }
        let staging = if buffers.iter().any(|b| b.role == BufferRole::Devscalar) {
            Some(ffi::plan::Staging::new(&ctx)?)
        } else {
            None
        };
        let run = Self {
            inner: Arc::new(Instance {
                plan: plan.clone(),
                context: ctx.clone(),
                home: Arc::clone(stream),
                device: ctx.ordinal() as i32,
                arena: Mutex::new(Arena { slice, slots_of: 0, staging }),
                arena_addr,
                buckets: runs,
                by_name,
                syms,
                patterns,
                captures: AtomicU64::new(0),
            }),
        };
        // -- every step's preconditions that no address can change, with made-up addresses for the externals (Python's virtual ones, distinct
        // per buffer): a refusal comes now, not at the first launch
        for (k, bucket) in run.inner.buckets.iter().enumerate() {
            let bases: Vec<u64> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| match b.offset {
                    Some(offset) => arena_addr + offset,
                    None => ((i as u64) + 1) << 44,
                })
                .collect();
            run.ready(&Bound { bucket: k, bases, slots: Vec::new() })
                .map_err(|e| in_step(e, &format!("at {}", describe_values(&plan.buckets()[bucket.index].values))))?;
        }
        Ok(run)
    }

    /// The plan.
    pub fn plan(&self) -> &LaunchPlan {
        &self.inner.plan
    }

    /// The context the plan is instantiated in: a launch is on a stream of it.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.inner.context
    }

    /// The stream the arena was allocated on (and is freed on).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.inner.home
    }

    /// The bytes of the arena.
    pub fn arena_bytes(&self) -> u64 {
        self.inner.plan.arena_bytes()
    }

    /// The bucket combinations the run was instantiated for.
    pub fn buckets(&self) -> Vec<&[(String, u64)]> {
        self.inner.buckets.iter().map(|b| self.inner.plan.buckets()[b.index].values.as_slice()).collect()
    }

    fn check_stream(&self, stream: &CudaStream) -> Result<()> {
        if **stream.context() != *self.inner.context {
            return Err(Error::Launch(format!(
                "the stream is in another CUDA context than the one the plan of graph '{}' is instantiated in: use a stream of that context",
                self.inner.plan.graph()
            )));
        }
        Ok(())
    }

    /// Bind a call: the externals by name, the device scalars by name, the dimensions and the bucket, the address of every buffer. Every check
    /// of an external is made here (and none of a step): see [`PlanRun::launch`].
    fn bind(&self, externals: &[(&str, TensorArg<'_>)], devscalars: &[(&str, DevScalar)]) -> Result<Bound> {
        let inner = &*self.inner;
        let plan = &inner.plan;
        let buffers = plan.buffers();
        let graph = plan.graph();
        let list = |keep: &dyn Fn(BufferRole) -> bool| -> String {
            buffers
                .iter()
                .filter(|b| keep(b.role))
                .map(|b| format!("{} ({})", b.name, b.role))
                .collect::<Vec<_>>()
                .join(", ")
        };
        // -- the names
        let mut given: Vec<Option<usize>> = vec![None; buffers.len()];
        for (k, (name, _)) in externals.iter().enumerate() {
            // a returned value by its name, or by out<i> (`LaunchPlan::output_buffer`)
            let index = inner
                .by_name
                .get(*name)
                .copied()
                .or_else(|| plan.output_buffer(name).and_then(|b| inner.by_name.get(b).copied()));
            match index {
                None => {
                    return Err(Error::Launch(format!(
                        "'{name}' is not a buffer of the plan of graph '{graph}'; its externals are {}",
                        list(&|r| r.is_external())
                    )));
                }
                Some(i) if buffers[i].role == BufferRole::Devscalar => {
                    return Err(Error::Launch(format!(
                        "'{name}' is a device scalar: give its value with the device scalars, not a tensor (its slot is in the arena)"
                    )));
                }
                Some(i) if !buffers[i].role.is_external() => {
                    return Err(Error::Launch(format!(
                        "'{name}' is the {} '{name}' of the plan, which lives in its arena: a call gives the externals only ({})",
                        buffers[i].role,
                        list(&|r| r.is_external())
                    )));
                }
                Some(i) if given[i].is_some() => {
                    return Err(Error::Launch(format!("the external '{name}' is given twice")));
                }
                Some(i) => given[i] = Some(k),
            }
        }
        if let Some(missing) = (0..buffers.len()).find(|i| buffers[*i].role.is_external() && given[*i].is_none()) {
            return Err(Error::Launch(format!(
                "the external '{}' ({}) of graph '{graph}' is missing: a call gives every one of {}",
                buffers[missing].name,
                buffers[missing].role,
                list(&|r| r.is_external())
            )));
        }
        let mut values: Vec<Option<DevScalar>> = vec![None; buffers.len()];
        for (name, value) in devscalars {
            match inner.by_name.get(*name).copied() {
                Some(i) if buffers[i].role == BufferRole::Devscalar => {
                    if values[i].is_some() {
                        return Err(Error::Launch(format!("the device scalar '{name}' is given twice")));
                    }
                    values[i] = Some(*value);
                }
                _ => {
                    return Err(Error::Launch(format!(
                        "'{name}' is not a device scalar of the plan of graph '{graph}'; its device scalars are [{}]",
                        list(&|r| r == BufferRole::Devscalar)
                    )));
                }
            }
        }
        let mut slots = Vec::new();
        for (i, b) in buffers.iter().enumerate().filter(|(_, b)| b.role == BufferRole::Devscalar) {
            let Some(value) = values[i] else {
                return Err(Error::Launch(format!(
                    "the device scalar '{}' of graph '{graph}' has no value: a call gives every one of [{}]",
                    b.name,
                    list(&|r| r == BufferRole::Devscalar)
                )));
            };
            let offset = b.offset.expect("a device scalar lives in the arena");
            slots.push(Slot { name: b.name.clone(), offset, dtype: b.dtype, bytes: value.encode(b.dtype, &b.name)? });
        }

        // -- each external on its own: device, context, dtype, rank, density, and that the plan may write it
        let tensor = |i: usize| &externals[given[i].expect("checked above")].1;
        for (i, b) in buffers.iter().enumerate().filter(|(_, b)| b.role.is_external()) {
            let t = tensor(i);
            let name = b.name.as_str();
            if t.device != inner.device || t.context.is_some_and(|c| c != inner.context.cu_ctx() as usize) {
                return Err(Error::precondition(
                    PreconditionKind::Device,
                    Some(name),
                    format!(
                        "the external '{name}' is memory of device {} or of another CUDA context than the plan's (device {})",
                        t.device, inner.device
                    ),
                ));
            }
            if t.dtype != b.dtype {
                return Err(Error::precondition(
                    PreconditionKind::Dtype,
                    Some(name),
                    format!("the external '{name}' has dtype {}, but the plan has {}", t.dtype.name(), b.dtype.name()),
                ));
            }
            if t.shape.len() != b.shape.len() {
                return Err(Error::precondition(
                    PreconditionKind::Rank,
                    Some(name),
                    format!(
                        "the external '{name}' has rank {} (shape {:?}), but the plan has rank {} ({})",
                        t.shape.len(),
                        t.shape,
                        b.shape.len(),
                        b.shape.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
                    ),
                ));
            }
            if let Some(strides) = t.strides
                && !dense(t.shape, strides)
            {
                return Err(Error::precondition(
                    PreconditionKind::Contiguous,
                    Some(name),
                    format!(
                        "the external '{name}' is not contiguous (shape {:?}, strides {strides:?}): a tensor of a plan is dense and row-major",
                        t.shape
                    ),
                ));
            }
            if b.role.is_writable() && t.access == Access::In {
                return Err(Error::precondition(
                    PreconditionKind::Access,
                    Some(name),
                    format!(
                        "the external '{name}' is a read-only tensor, but the plan writes it ({}): give a TensorMut",
                        b.role
                    ),
                ));
            }
        }

        // -- the dimensions, bound from the shapes of the parameters (as a single launch binds them), and the bucket
        let mut dims: Vec<Option<i128>> = vec![None; inner.syms.len()];
        let mut pending: Vec<(usize, usize, i64)> = Vec::new();
        for (i, b) in buffers.iter().enumerate().filter(|(_, b)| b.role.is_parameter()) {
            for (axis, size) in tensor(i).shape.iter().enumerate() {
                let actual = i64::try_from(*size).map_err(|_| {
                    Error::precondition(PreconditionKind::Shape, Some(&b.name), "a size does not fit 64 bits")
                })?;
                pending.push((i, axis, actual));
            }
        }
        loop {
            let mut progress = false;
            let mut rest = Vec::with_capacity(pending.len());
            for (i, axis, actual) in pending {
                let pattern = &inner.patterns[i][axis];
                let solved = match pattern.expr.static_value() {
                    // an expression that folds to a constant (`4`, `M - M + 4`) is checked as one
                    Some(expected) if expected == i128::from(actual) => Ok(Solved::Checked),
                    Some(expected) => Err(Unsolvable::Mismatch { expected }),
                    None => pattern.solve_from(i128::from(actual), &dims, 1),
                };
                match solved {
                    Ok(Solved::Checked) => progress = true,
                    Ok(Solved::Bound(var, value)) => {
                        dims[var] = Some(value);
                        progress = true;
                    }
                    Ok(Solved::Stuck) => rest.push((i, axis, actual)),
                    Err(e) => return Err(self.unsolvable(i, axis, actual, e)),
                }
            }
            pending = rest;
            if pending.is_empty() || !progress {
                break;
            }
        }
        let mut chosen: Vec<(&str, u64)> = Vec::with_capacity(plan.dims().len());
        for (d, (name, allowed)) in plan.dims().iter().enumerate() {
            let Some(value) = dims[d] else {
                return Err(Error::precondition(
                    PreconditionKind::Bucket,
                    Some(name),
                    format!("the dimension {name} of graph '{graph}' is not determined by the externals"),
                ));
            };
            let Some(bucket) = u64::try_from(value).ok().filter(|v| *v >= 1 && allowed.contains(v)) else {
                return Err(Error::precondition(
                    PreconditionKind::Bucket,
                    Some(name),
                    format!(
                        "dimension {name} is {value}, which is not one of the buckets {allowed:?} of the plan of graph '{graph}'; a bucketed \
                         dimension must equal a bucket exactly (there is no padding: pad or slice in the host)"
                    ),
                ));
            };
            chosen.push((name.as_str(), bucket));
        }
        let index = plan.bucket_index(&chosen).expect("every combination of the dims is a bucket");
        let Some(bucket) = inner.buckets.iter().position(|b| b.index == index) else {
            let have: Vec<String> =
                inner.buckets.iter().map(|b| describe_values(&plan.buckets()[b.index].values)).collect();
            return Err(Error::precondition(
                PreconditionKind::Bucket,
                chosen.first().map(|(n, _)| *n),
                format!(
                    "the externals select the bucket {} of graph '{graph}', but this run was instantiated for {} only: instantiate the plan for it \
                     (LaunchPlan::instantiate, or instantiate_all)",
                    describe_values(&plan.buckets()[index].values),
                    have.join(", ")
                ),
            ));
        };
        let shapes = &plan.buckets()[index].shapes;

        // -- every external has its shape at the bucket; none overlaps the arena, and none that is written overlaps another
        let arena = (u128::from(inner.arena_addr), u128::from(inner.arena_addr) + u128::from(plan.arena_bytes()));
        let mut ranges: Vec<(usize, u128, u128)> = Vec::new();
        for (i, b) in buffers.iter().enumerate().filter(|(_, b)| b.role.is_external()) {
            let t = tensor(i);
            if t.shape != shapes[i].as_slice() {
                return Err(Error::precondition(
                    PreconditionKind::Shape,
                    Some(&b.name),
                    format!(
                        "the external '{}' has shape {:?}, but at {} the plan has {:?}",
                        b.name,
                        t.shape,
                        describe_values(&plan.buckets()[index].values),
                        shapes[i]
                    ),
                ));
            }
            let bytes = shapes[i].iter().fold(1u128, |acc, d| acc.saturating_mul(*d as u128)) * b.dtype.size() as u128;
            let (lo, hi) = (u128::from(t.ptr), u128::from(t.ptr) + bytes);
            if plan.arena_bytes() > 0 && lo < arena.1 && arena.0 < hi {
                return Err(Error::precondition(
                    PreconditionKind::Alias,
                    Some(&b.name),
                    format!(
                        "'{}' overlaps the arena of the plan (the memory the plan keeps its intermediates, device scalars and workspace in): a \
                         tensor of the caller must not be in it",
                        b.name
                    ),
                ));
            }
            ranges.push((i, lo, hi));
        }
        for &(w, lo, hi) in ranges.iter().filter(|(i, _, _)| buffers[*i].role.is_writable()) {
            if let Some(&(other, _, _)) = ranges.iter().find(|(o, olo, ohi)| *o != w && lo < *ohi && *olo < hi) {
                return Err(Error::precondition(
                    PreconditionKind::Alias,
                    Some(&buffers[w].name),
                    format!(
                        "'{}' is written by the plan but overlaps '{}': the tensors of a plan must not alias when one of them is written",
                        buffers[w].name, buffers[other].name
                    ),
                ));
            }
        }
        let bases = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| match b.offset {
                Some(offset) => inner.arena_addr + offset,
                None => tensor(i).ptr,
            })
            .collect();
        Ok(Bound { bucket, bases, slots })
    }

    fn unsolvable(&self, buffer: usize, axis: usize, actual: i64, e: Unsolvable) -> Error {
        let b = &self.inner.plan.buffers()[buffer];
        let pattern = &self.inner.patterns[buffer][axis];
        let head = format!("the external '{}' has size {actual} on axis {axis}", b.name);
        let name = Some(b.name.as_str());
        match e {
            Unsolvable::Mismatch { expected } if pattern.is_static => Error::precondition(
                PreconditionKind::StaticSize,
                name,
                format!("{head}, but the plan has {expected} there"),
            ),
            Unsolvable::Mismatch { expected } => Error::precondition(
                PreconditionKind::Shape,
                name,
                format!("{head}, but the plan has {} = {expected} there", pattern.text),
            ),
            Unsolvable::NotWhole { var, target, coefficient } => Error::precondition(
                PreconditionKind::Shape,
                name,
                format!(
                    "{head}, which cannot be {} (it would need {} = {target}/{coefficient})",
                    pattern.text,
                    self.inner.syms.name(var)
                ),
            ),
            Unsolvable::NotPositive { var, value } => Error::precondition(
                PreconditionKind::Shape,
                name,
                format!(
                    "{head}, which cannot be {} (it would need {} = {value})",
                    pattern.text,
                    self.inner.syms.name(var)
                ),
            ),
            Unsolvable::Eval(message) => Error::Launch(format!("{head}: {message}")),
        }
    }

    /// Every step of the bound bucket checked with the addresses of the call, as a single launch is checked: the grid and the parameters of
    /// each kernel, the addresses of each GEMM. A refusal is the step's, and nothing is launched.
    fn ready(&self, bound: &Bound) -> Result<Vec<Ready<'_>>> {
        let steps = self.inner.buckets[bound.bucket].steps.len();
        (0..steps).map(|si| self.ready_step(bound.bucket, si, &bound.bases)).collect()
    }

    /// Step `si` of the instantiated bucket `bucket` checked with the addresses `bases` (one per buffer), as a single launch is checked.
    fn ready_step(&self, bucket: usize, si: usize, bases: &[u64]) -> Result<Ready<'_>> {
        let inner = &*self.inner;
        let plan = &inner.plan;
        let bucket = &inner.buckets[bucket];
        let combo = &plan.buckets()[bucket.index];
        let region_bytes =
            |i: usize| combo.shapes[i].iter().product::<usize>() as u64 * plan.buffers()[i].dtype.size() as u64;
        {
            let step = &bucket.steps[si];
            let stem = if matches!(step.op, Op::Copy) { "copy" } else { step.stem.as_str() };
            let what = format!("step #{si} ({stem}) at {}", describe_values(&combo.values));
            let mut args: Vec<TensorArg<'_>> = Vec::with_capacity(step.args.len() + 1);
            for a in &step.args {
                let region = ffi::plan::check_region(bases[a.buffer], region_bytes(a.buffer), inner.device);
                let arg = match &a.view {
                    Some((strides, offset)) => {
                        TensorArg::new_strided(region, &a.shape, strides, *offset, a.dtype, a.access)
                    }
                    None => TensorArg::new(region, &a.shape, a.dtype, a.access),
                };
                args.push(arg.map_err(|e| in_step(e, &what))?);
            }
            Ok(match &step.op {
                Op::Kernel(kernel) => {
                    Ready::Kernel(kernel, Box::new(kernel.check(&args).map_err(|e| in_step(e, &what))?))
                }
                Op::Gemm(gemm) => {
                    let (w, shape) = step.workspace.as_ref().expect("a library step has a workspace");
                    let region = ffi::plan::check_region(bases[*w], region_bytes(*w), inner.device);
                    args.push(TensorArg::new(region, shape, DType::U8, Access::InOut).map_err(|e| in_step(e, &what))?);
                    Ready::Gemm(gemm, gemm.check_call(&args).map_err(|e| in_step(e, &what))?)
                }
                Op::Copy => {
                    // the dense bytes of src into dst: each was made a tensor of its region above (so it lies inside its buffer), and the
                    // reader checked that both are dense, of one dtype and one count; with the real addresses they must not overlap
                    let bytes = args[0].shape.iter().product::<usize>() * args[0].dtype.size();
                    let (src, dst) = (args[0].ptr, args[1].ptr);
                    if src < dst + bytes as u64 && dst < src + bytes as u64 {
                        return Err(Error::precondition(
                            PreconditionKind::Alias,
                            Some("dst"),
                            format!("{what}: the source and the destination of the copy overlap"),
                        ));
                    }
                    Ready::Copy { src, dst, bytes }
                }
            })
        }
    }

    /// Launch step `step` of the bucket `bucket` (one value per dimension) on the caller's tensors: the binary of the step's entry in the plan's
    /// verified manifest (what the pin of the plan pins) and the binding the plan says, each argument the buffer and the view the step names,
    /// made of the tensor given for that buffer. A single-op replay under the plan's pin (Python: `tilefish.runtime.run_step`).
    ///
    /// `buffers` gives a tensor for every buffer the step binds, by name (a returned value also as `out<i>`), and nothing else: of the buffer's
    /// dtype and its shape at the bucket (a device scalar: one element), dense, on the plan's device and context, a `TensorMut` for a buffer the
    /// step writes. A GEMM's workspace is the plan's arena. Every check of the step is made with these addresses before the launch, and the
    /// tensors must not overlap the arena, nor each other where the step writes one. The launch is ordered as [`PlanRun::launch`] is.
    pub fn launch_step(
        &self,
        stream: &Arc<CudaStream>,
        bucket: &[(&str, u64)],
        step: usize,
        buffers: &mut [(&str, TensorArg<'_>)],
    ) -> Result<()> {
        self.check_stream(stream)?;
        let inner = &*self.inner;
        let plan = &inner.plan;
        let all = plan.buffers();
        let graph = plan.graph();
        let Some(index) = plan.bucket_index(bucket) else {
            return Err(Error::precondition(
                PreconditionKind::Bucket,
                None,
                format!("the plan of graph '{graph}' has no bucket {bucket:?}"),
            ));
        };
        let Some(b) = inner.buckets.iter().position(|r| r.index == index) else {
            return Err(Error::precondition(
                PreconditionKind::Bucket,
                None,
                format!("this run of the plan of graph '{graph}' was not instantiated for the bucket {bucket:?}"),
            ));
        };
        let run = &inner.buckets[b];
        let Some(st) = run.steps.get(step) else {
            return Err(Error::Launch(format!("the bucket has {} steps, not a step #{step}", run.steps.len())));
        };
        let shapes = &plan.buckets()[index].shapes;
        let mut given: Vec<Option<usize>> = vec![None; all.len()];
        for (k, (name, _)) in buffers.iter().enumerate() {
            let i = inner
                .by_name
                .get(*name)
                .copied()
                .or_else(|| plan.output_buffer(name).and_then(|o| inner.by_name.get(o).copied()))
                .filter(|i| all[*i].role != BufferRole::Workspace)
                .ok_or_else(|| {
                    Error::Launch(format!(
                        "'{name}' is not a buffer of the plan of graph '{graph}' (the workspace is the plan's own)"
                    ))
                })?;
            if given[i].replace(k).is_some() {
                return Err(Error::Launch(format!("the buffer '{}' is given twice", all[i].name)));
            }
        }
        let needed: Vec<usize> = st.args.iter().map(|a| a.buffer).collect();
        if let Some(missing) = needed.iter().find(|i| given[**i].is_none()) {
            return Err(Error::Launch(format!(
                "step #{step} binds the buffer '{}', which is not given",
                all[*missing].name
            )));
        }
        if let Some(extra) = (0..all.len()).find(|i| given[*i].is_some() && !needed.contains(i)) {
            return Err(Error::Launch(format!("step #{step} does not bind the buffer '{}'", all[extra].name)));
        }
        let mut bases = vec![0u64; all.len()];
        let mut ranges: Vec<(usize, u128, u128)> = Vec::new();
        for (i, k) in given.iter().enumerate().filter_map(|(i, k)| k.map(|k| (i, k))) {
            let (t, buf) = (&buffers[k].1, &all[i]);
            let name = buf.name.as_str();
            let refuse =
                |kind, why: String| Err(Error::precondition(kind, Some(name), format!("the buffer '{name}': {why}")));
            if t.device != inner.device || t.context.is_some_and(|c| c != inner.context.cu_ctx() as usize) {
                return refuse(
                    PreconditionKind::Device,
                    "its tensor is not memory of the plan's device and context".to_owned(),
                );
            }
            if t.dtype != buf.dtype {
                return refuse(
                    PreconditionKind::Dtype,
                    format!("its tensor is {}, the buffer {}", t.dtype.name(), buf.dtype.name()),
                );
            }
            let scalar = buf.role == BufferRole::Devscalar && t.shape.iter().product::<usize>() == 1;
            if t.shape != shapes[i].as_slice() && !scalar {
                return refuse(
                    PreconditionKind::Shape,
                    format!("its tensor has shape {:?}, the buffer at this bucket {:?}", t.shape, shapes[i]),
                );
            }
            if t.strides.is_some_and(|s| !dense(t.shape, s)) {
                return refuse(PreconditionKind::Contiguous, "its tensor is not contiguous".to_owned());
            }
            if st.written.contains(&i) && t.access == Access::In {
                return refuse(
                    PreconditionKind::Access,
                    "the step writes it, but its tensor is read-only: give a TensorMut".to_owned(),
                );
            }
            let bytes =
                shapes[i].iter().fold(1u128, |acc, d| acc.saturating_mul(*d as u128)) * buf.dtype.size() as u128;
            let (lo, hi) = (u128::from(t.ptr), u128::from(t.ptr) + bytes);
            let arena = (u128::from(inner.arena_addr), u128::from(inner.arena_addr) + u128::from(plan.arena_bytes()));
            if plan.arena_bytes() > 0 && lo < arena.1 && arena.0 < hi {
                return refuse(PreconditionKind::Alias, "its tensor overlaps the arena of the plan".to_owned());
            }
            if let Some(&(o, _, _)) = ranges
                .iter()
                .find(|(o, olo, ohi)| lo < *ohi && *olo < hi && (st.written.contains(&i) || st.written.contains(o)))
            {
                return refuse(
                    PreconditionKind::Alias,
                    format!("its tensor overlaps the one of '{}', and the step writes one", all[o].name),
                );
            }
            ranges.push((i, lo, hi));
            bases[i] = t.ptr;
        }
        if let Some((w, _)) = st.workspace {
            bases[w] = inner.arena_addr + all[w].offset.expect("the workspace is in the arena");
        }
        let ready = [self.ready_step(b, step, &bases)?];
        let shape = [plan.arena_bytes() as usize];
        let mut arena = inner.lock_arena();
        let Arena { slice, staging, .. } = &mut *arena;
        let mut tensors: Vec<TensorArg<'_>> = buffers.iter_mut().map(|(_, t)| t.reborrow()).collect();
        if let Some(slice) = slice.as_mut() {
            tensors.push(TensorMut::<U8>::from_cuda(slice, &shape)?.into_arg());
        }
        ffi::ordered(stream, &mut tensors, |tensors, want_end| {
            tracked(stream, tensors, &[], staging.as_mut(), want_end, |stream| ffi::plan::enqueue_steps(stream, &ready))
        })
    }

    /// Check a call as [`PlanRun::launch`] does, and launch nothing.
    pub fn check(&self, externals: &[(&str, TensorArg<'_>)], devscalars: &[(&str, DevScalar)]) -> Result<()> {
        let bound = self.bind(externals, devscalars)?;
        self.ready(&bound).map(drop)
    }

    /// Run the plan on `stream`: the externals by name (`TensorRef` for a buffer the plan reads, `TensorMut` for one it writes: an `output` or
    /// an `inout`), a value for every device scalar.
    ///
    /// The checks, all before anything is enqueued (**a refusal launches nothing and writes nothing**, not even a device scalar): every external
    /// is given once and nothing else is; each is on the plan's device and in its context, has the dtype and the rank of its buffer, is dense,
    /// and may be written where the plan writes it; the dimensions are bound from the shapes of the parameters (`3*M` against 192 gives
    /// `M = 64`), every one is bound to one of its buckets, and the bucket is one the run was instantiated for; every external has its buffer's
    /// shape at the bucket, none overlaps the arena, and none that is written overlaps another; and every step is checked with the real
    /// addresses as a single launch of its entry is. Then the device scalars are written into their slots, on `stream`, and the steps are
    /// launched in order on `stream`: nothing is allocated.
    ///
    /// Ordering: as for [`Kernel::launch`], for every external made of a cudarc slice (cudarc's tracking on `stream`, the slice's own stream
    /// waiting for the end of the plan, a written slice's own stream waited for first), and for the arena, which every launch writes: a
    /// launch on another stream than the one the run was instantiated on waits for that stream first, and that stream waits for the end of
    /// the launch, so two launches of one run, or a launch and a replay, never overlap. A launch is refused while `stream`, or a stream an
    /// external or the arena is freed on, captures a CUDA graph ([`PlanRun::capture`] is how a plan is captured). An `Err` from a driver
    /// call may come after some steps were enqueued, as for a single launch.
    pub fn launch(
        &self,
        stream: &Arc<CudaStream>,
        externals: &mut [(&str, TensorArg<'_>)],
        devscalars: &[(&str, DevScalar)],
    ) -> Result<()> {
        self.launch_tapped(stream, externals, devscalars, &mut [])
    }

    /// [`PlanRun::launch`] with debug taps: each [`Tap`] copies a buffer of the plan (all of it, at the bucket) into the caller's tensor right after
    /// a step (`after`, or by default the last step of the bucket that writes the buffer), on the launch stream, before the next step may reuse the
    /// buffer's bytes. The intermediates of the very plan a product runs, with the plan and its digest unchanged (Python: `tilefish.run(...,
    /// taps=...)`). A tap is checked with everything else before anything is enqueued: a buffer of the plan (by name, or `out<i>`), not the
    /// workspace; a step of the bucket; a dense, writable tensor (a `TensorMut`) of the buffer's dtype and shape at the bucket, on the plan's device
    /// and context, that overlaps no external, no other tap and not the arena. Its memory is ordered as an external's that the plan writes.
    pub fn launch_tapped(
        &self,
        stream: &Arc<CudaStream>,
        externals: &mut [(&str, TensorArg<'_>)],
        devscalars: &[(&str, DevScalar)],
        taps: &mut [Tap<'_>],
    ) -> Result<()> {
        self.check_stream(stream)?;
        let bound = self.bind(externals, devscalars)?;
        let mut ready = self.ready(&bound)?;
        if !taps.is_empty() {
            let copies = self.resolve_taps(&bound, externals, taps)?;
            for (after, copy) in copies.into_iter().rev() {
                ready.insert(after + 1, copy); // from the last: an index is not moved by an insertion after it
            }
        }
        let shape = [self.inner.plan.arena_bytes() as usize];
        let mut arena = self.inner.lock_arena();
        let launched = {
            let Arena { slice, staging, .. } = &mut *arena;
            let mut tensors: Vec<TensorArg<'_>> = externals.iter_mut().map(|(_, t)| t.reborrow()).collect();
            tensors.extend(taps.iter_mut().map(|t| t.into.reborrow()));
            if let Some(slice) = slice.as_mut() {
                tensors.push(TensorMut::<U8>::from_cuda(slice, &shape)?.into_arg());
            }
            let slots = bound.slots.as_slice();
            ffi::ordered(stream, &mut tensors, |tensors, want_end| {
                tracked(stream, tensors, slots, staging.as_mut(), want_end, |stream| {
                    ffi::plan::enqueue_steps(stream, &ready)
                })
            })
        };
        arena.slots_of = 0;
        launched
    }

    /// The copies of the taps of a launch, each with the index of the step it follows, in the order of the steps (see [`PlanRun::launch_tapped`]).
    fn resolve_taps(
        &self,
        bound: &Bound,
        externals: &[(&str, TensorArg<'_>)],
        taps: &[Tap<'_>],
    ) -> Result<Vec<(usize, Ready<'static>)>> {
        let inner = &*self.inner;
        let plan = &inner.plan;
        let buffers = plan.buffers();
        let bucket = &inner.buckets[bound.bucket];
        let combo = &plan.buckets()[bucket.index];
        let steps = bucket.steps.len();
        let elements = |shape: &[usize]| shape.iter().fold(1u128, |acc, d| acc.saturating_mul(*d as u128));
        let mut taken: Vec<(u128, u128, String)> = Vec::new();
        if plan.arena_bytes() > 0 {
            let start = u128::from(inner.arena_addr);
            taken.push((start, start + u128::from(plan.arena_bytes()), "the arena of the plan".to_owned()));
        }
        for (name, t) in externals {
            let bytes = elements(t.shape) * t.dtype.size() as u128;
            taken.push((u128::from(t.ptr), u128::from(t.ptr) + bytes, format!("the external '{name}'")));
        }
        let mut copies = Vec::with_capacity(taps.len());
        for tap in taps {
            let name = tap.buffer;
            let index = inner
                .by_name
                .get(name)
                .copied()
                .or_else(|| plan.output_buffer(name).and_then(|b| inner.by_name.get(b).copied()))
                .filter(|i| buffers[*i].role != BufferRole::Workspace);
            let Some(index) = index else {
                return Err(Error::Launch(format!(
                    "the tap '{name}' is not a buffer of the plan of graph '{}' (a tap is any buffer but the workspace, by name or out<i>)",
                    plan.graph()
                )));
            };
            let after = match tap.after {
                Some(step) => step,
                None => bucket.steps.iter().rposition(|s| s.written.contains(&index)).ok_or_else(|| {
                    Error::Launch(format!(
                        "the tap '{name}' names a buffer no step writes: give the step to copy it after"
                    ))
                })?,
            };
            if after >= steps {
                return Err(Error::Launch(format!(
                    "the tap '{name}' is to follow step {after}, but the bucket has {steps} steps"
                )));
            }
            let (t, b) = (&tap.into, &buffers[index]);
            let want = combo.shapes[index].as_slice();
            let refuse =
                |kind, why: String| Err(Error::precondition(kind, Some(name), format!("the tap '{name}': {why}")));
            if t.device != inner.device || t.context.is_some_and(|c| c != inner.context.cu_ctx() as usize) {
                return refuse(
                    PreconditionKind::Device,
                    "its tensor is not memory of the plan's device and context".to_owned(),
                );
            }
            if t.dtype != b.dtype {
                return refuse(
                    PreconditionKind::Dtype,
                    format!("its tensor is {}, the buffer {}", t.dtype.name(), b.dtype.name()),
                );
            }
            if t.shape != want {
                return refuse(
                    PreconditionKind::Shape,
                    format!("its tensor has shape {:?}, the buffer at this bucket {want:?}", t.shape),
                );
            }
            if t.strides.is_some_and(|s| !dense(t.shape, s)) {
                return refuse(PreconditionKind::Contiguous, "its tensor is not contiguous".to_owned());
            }
            if t.access == Access::In {
                return refuse(PreconditionKind::Access, "its tensor is read-only: give a TensorMut".to_owned());
            }
            let bytes = elements(want) * b.dtype.size() as u128;
            let (lo, hi) = (u128::from(t.ptr), u128::from(t.ptr) + bytes);
            if let Some((_, _, what)) = taken.iter().find(|(olo, ohi, _)| lo < *ohi && *olo < hi) {
                return refuse(PreconditionKind::Alias, format!("its tensor overlaps {what}"));
            }
            taken.push((lo, hi, format!("the tap '{name}'")));
            let bytes = usize::try_from(bytes)
                .map_err(|_| Error::Launch(format!("the tap '{name}' does not fit the address space")))?;
            copies.push((after, Ready::Copy { src: bound.bases[index], dst: t.ptr, bytes }));
        }
        copies.sort_by_key(|(after, _)| *after); // stable: taps after one step in the order given
        Ok(copies)
    }

    /// Capture the plan in a CUDA graph on `stream`, for the bucket the externals select: the returned [`CapturedPlan`] holds the graph and this
    /// instantiation (its arena), and **borrows the externals for `'ext`**, so the graph cannot be replayed after any memory it captured is
    /// gone: a slice an external is made of cannot be dropped, or moved, while the `CapturedPlan` lives.
    ///
    /// Everything [`PlanRun::launch`] checks is checked first (a refusal captures nothing); then the steps are captured on `stream`
    /// (`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`), which must not be capturing already. Nothing runs, and nothing is allocated: the workspace and
    /// the device scalars are in the arena, every function and handle was loaded when the plan was instantiated. `devscalars` are the values the
    /// first replay reads (see [`CapturedPlan::set_devscalar`]). The arena's lock is held from the beginning of the capture to its end, so a
    /// launch or a replay of this instantiation on another thread waits for the capture to end (it would otherwise make the arena's stream, which
    /// may be the one captured, wait for an event from outside the capture: `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`, and the capture invalidated).
    pub fn capture<'ext>(
        &self,
        stream: &Arc<CudaStream>,
        externals: Vec<(&str, TensorArg<'ext>)>,
        devscalars: &[(&str, DevScalar)],
    ) -> Result<CapturedPlan<'ext>> {
        self.check_stream(stream)?;
        ffi::refuse_per_thread(stream, externals.iter().map(|(_, t)| t))?;
        let status = stream.capture_status().map_err(Error::driver("cuStreamIsCapturing"))?;
        if status != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err(Error::Launch(
                "the stream is capturing a CUDA graph already: a plan is captured on a stream that is not".to_owned(),
            ));
        }
        let bound = self.bind(&externals, devscalars)?;
        let ready = self.ready(&bound)?;
        // The arena's lock is held from the beginning of the capture to its end: a launch or a replay of this instantiation on another thread
        // makes the arena's own stream wait for its end, which would be a wait on an event from outside the capture if that stream is the one
        // captured (`CUDA_ERROR_STREAM_CAPTURE_ISOLATION`, then the capture invalidated); they wait for the capture to end instead.
        let arena = self.inner.lock_arena();
        stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(Error::driver("cuStreamBeginCapture"))?;
        let enqueued = ffi::plan::enqueue_steps(stream, &ready);
        // the capture is ended whatever happened, so that the stream is usable again (a graph of a failed capture is dropped)
        let ended = stream.end_capture(CUgraphInstantiate_flags(0)).map_err(Error::driver("cuStreamEndCapture"));
        drop(arena);
        enqueued?;
        let graph = ended?.ok_or_else(|| Error::Launch("the capture made no graph".to_owned()))?;
        drop(ready);
        let id = self.inner.captures.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(CapturedPlan {
            graph,
            captured_on: Arc::clone(stream),
            run: self.clone(),
            bucket: bound.bucket,
            externals: externals.into_iter().map(|(n, t)| (n.to_owned(), t)).collect(),
            slots: bound.slots,
            id,
            current: false,
        })
    }
}

/// Whether a tensor of this shape and these element strides is dense and row-major, the strides of its size-1 axes not looked at (torch's
/// `is_contiguous`).
fn dense(shape: &[usize], strides: &[usize]) -> bool {
    let mut expected = 1u128;
    for (n, s) in shape.iter().zip(strides).rev() {
        if *n == 1 {
            continue;
        }
        if *s as u128 != expected {
            return false;
        }
        expected = expected.saturating_mul(*n as u128);
    }
    true
}

/// A launch plan captured in a CUDA graph, for one bucket and one set of externals: replayed with [`CapturedPlan::replay`].
///
/// It owns the graph and the instantiation it was captured from (so the arena outlives the graph), and borrows the externals for `'ext` (so
/// their memory outlives it too). A graph holds addresses, not contents: write new contents into the externals between replays (through
/// [`CapturedPlan::external_mut`], for an external given as a `TensorMut`), and set the device scalars with [`CapturedPlan::set_devscalar`].
pub struct CapturedPlan<'ext> {
    // the graph first: it is destroyed before the instantiation it captured (and the borrows end after both)
    graph: CudaGraph,
    captured_on: Arc<CudaStream>,
    run: PlanRun,
    bucket: usize,
    externals: Vec<(String, TensorArg<'ext>)>,
    slots: Vec<Slot>,
    /// Distinct among the captures of one instantiation: whose device scalar values the arena's slots hold.
    id: u64,
    /// Whether the slots hold this graph's values (set by a replay, cleared by `set_devscalar`).
    current: bool,
}

impl std::fmt::Debug for CapturedPlan<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapturedPlan")
            .field("graph", &self.run.inner.plan.graph())
            .field("bucket", &self.bucket())
            .field("externals", &self.externals.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl<'ext> CapturedPlan<'ext> {
    /// The instantiation the graph was captured from.
    pub fn run(&self) -> &PlanRun {
        &self.run
    }

    /// The bucket combination the graph runs.
    pub fn bucket(&self) -> &[(String, u64)] {
        let inner = &self.run.inner;
        &inner.plan.buckets()[inner.buckets[self.bucket].index].values
    }

    /// Replay the graph on `stream` (a stream of the plan's context; the one it was captured on, or another).
    ///
    /// The device scalars are written into their slots first, on `stream`, when the slots do not hold this graph's values (it was never
    /// replayed, a value was set, or a launch or another graph of the instantiation wrote them since). The externals and the arena are ordered
    /// exactly as for [`PlanRun::launch`]: a replay is refused while a stream captures, waits for the earlier uses of what it writes, and the
    /// streams the externals and the arena are freed on wait for its end.
    pub fn replay(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        self.run.check_stream(stream)?;
        let inner = &*self.run.inner;
        let shape = [inner.plan.arena_bytes() as usize];
        let mut arena = inner.lock_arena();
        let write = !(self.current && arena.slots_of == self.id);
        let replayed = {
            let Arena { slice, staging, .. } = &mut *arena;
            let mut tensors: Vec<TensorArg<'_>> = self.externals.iter_mut().map(|(_, t)| t.reborrow()).collect();
            if let Some(slice) = slice.as_mut() {
                tensors.push(TensorMut::<U8>::from_cuda(slice, &shape)?.into_arg());
            }
            let slots: &[Slot] = if write { &self.slots } else { &[] };
            let (graph, captured_on) = (&self.graph, &self.captured_on);
            ffi::ordered(stream, &mut tensors, |tensors, want_end| {
                tracked(stream, tensors, slots, staging.as_mut(), want_end, |stream| {
                    ffi::plan::launch_graph(graph, captured_on, stream)
                })
            })
        };
        match replayed {
            Ok(()) => {
                if write {
                    arena.slots_of = self.id;
                    self.current = true;
                }
                Ok(())
            }
            Err(e) => {
                arena.slots_of = 0; // what the slots hold is not known: the next replay writes them
                Err(e)
            }
        }
    }

    /// Set the value of the device scalar `name` for the next replays. It is written into its slot of the arena by the next replay, on that
    /// replay's stream, before the graph.
    pub fn set_devscalar(&mut self, name: &str, value: impl Into<DevScalar>) -> Result<()> {
        let Some(slot) = self.slots.iter_mut().find(|s| s.name == name) else {
            let names: Vec<&str> = self.slots.iter().map(|s| s.name.as_str()).collect();
            return Err(Error::Launch(format!(
                "'{name}' is not a device scalar of the plan; its device scalars are {names:?}"
            )));
        };
        slot.bytes = value.into().encode(slot.dtype, name)?;
        self.current = false;
        Ok(())
    }

    fn external_arg(&self, name: &str) -> Result<&TensorArg<'ext>> {
        self.externals.iter().find(|(n, _)| n == name).map(|(_, t)| t).ok_or_else(|| {
            let names: Vec<&str> = self.externals.iter().map(|(n, _)| n.as_str()).collect();
            Error::Launch(format!("'{name}' is not an external of the captured plan; its externals are {names:?}"))
        })
    }

    /// The memory the external `name` was made of (the whole cudarc slice or view, as bytes), to read: an output after a replay, say. Reading
    /// it through cudarc on any stream is ordered after the replays by cudarc's tracking (the replays record their use on the slice).
    pub fn external(&self, name: &str) -> Result<CudaView<'_, u8>> {
        let missing = || Error::Launch(format!("the memory of the external '{name}' has no bytes to view"));
        match &self.external_arg(name)?.backing {
            Backing::Read(memory) => memory.bytes_from(0).ok_or_else(missing),
            Backing::Write(memory) => memory.bytes().ok_or_else(missing),
            Backing::Raw => Err(Error::Launch(format!(
                "the external '{name}' is a region the host vouched for (DevicePtr::from_raw): its memory is the host's"
            ))),
        }
    }

    /// The memory the external `name` was made of (the whole cudarc slice or view, as bytes), to write new contents into between replays: an
    /// input, say. Only an external given as a `TensorMut` (an exclusive borrow) can be written while the graph holds it. A write through
    /// cudarc is ordered after the replays that read the memory, and before the next one, by cudarc's tracking.
    pub fn external_mut(&mut self, name: &str) -> Result<CudaViewMut<'_, u8>> {
        self.external_arg(name)?;
        let position = self.externals.iter().position(|(n, _)| n == name).expect("an external");
        match &mut self.externals[position].1.backing {
            Backing::Write(memory) => memory
                .bytes_mut()
                .ok_or_else(|| Error::Launch(format!("the memory of the external '{name}' has no bytes to view"))),
            Backing::Read(_) => Err(Error::Launch(format!(
                "the external '{name}' was given read-only (a TensorRef: a shared borrow, which other code may read too): give a TensorMut to \
                 write it between replays"
            ))),
            Backing::Raw => Err(Error::Launch(format!(
                "the external '{name}' is a region the host vouched for (DevicePtr::from_raw): its memory is the host's"
            ))),
        }
    }
}
