//! Everything a launch checks, with no driver in it.
//!
//! A [`Plan`] is one manifest entry made ready for launching: the dimension names numbered, every expression parsed,
//! every constraint attached to what it is about. [`Plan::prepare`] takes the tensors of a call and either refuses it (with
//! the same checks, in the same order, as Python's `replay`) or returns the grid and the `kernelParams` values to launch
//! with. The driver is not involved (a device is described by a [`DeviceInfo`], which a test can make up), so all of it
//! is tested without a GPU.
//!
//! Everything comes from the manifest: each pointer parameter has its declared shape (`["Rows", 1024]`: a dimension
//! expression, or the size of a static axis) and the entry has the constraints of its named dimensions. A pointer may also take a
//! *strided view* (its `strides` are fixed to 1 or are dimensions to bind, its `offset` is the address of the view's first element),
//! or be a *device scalar* (the empty shape: one element); a runtime scalar is an `i32` or an `i64`, and a `fixed` argument is a
//! value the manifest holds. A CuTe DSL kernel (manifest schema 3, section 4.9) takes its tensors as pointers or as TMA descriptors the DSL
//! encoded (kind `tensormap`: a tensor like a pointer, whose launch passes the template with the tensor's address in its first eight bytes,
//! 128 bytes by value), and constant bytes (kind `bytes`) the manifest holds.
//!
//! The checks are Python's, in the order of `runtime._prepare`, so that a call that breaks several preconditions is refused for the one
//! Python finds first: the number of tensors; for each tensor the kernel reads (`In`, `InOut`: what Python is given) its dtype and rank (a
//! device scalar is one element), that no dimension is empty, and that a tensor given for a parameter that is not strided is contiguous;
//! the binding of the dimensions from the sizes of those tensors, and from the canonical strides of their views, which must agree across
//! arguments and with the static sizes (a port of `solve_dims`: a call that leaves a dimension undetermined is ambiguous, an error that is not
//! a precondition), then of the dimensions they leave unbound from the shapes of the `Out` tensors that mention one (each checked on its own
//! first: a dimension only an `Out` parameter mentions, Python's `bind_outputs`); `divisible_by` and the buckets of each dimension, in the
//! order they were bound; the bucket of the binary (a bucketed
//! dimension is the bucket exactly: there is no padding, and it is what the tensors say, never taken from the binary); the device and the
//! context of every tensor; each tensor the kernel writes (`Out`, which Python allocates) against what Python would allocate; the runtime
//! scalars, derived from their dimension expression, from the element count of the tensor they count (which binds the dimension a scalar is,
//! when no tensor did) or from the stride they carry, then their range (`i32` or `i64`); aliasing of the tensors that are written; the
//! preconditions of the binary in the order the manifest lists them, argument after argument: the alignment each pointer promises, its
//! `max_elements` or `max_extent`, the divisor of a runtime scalar. Last, the grid is evaluated from its expression and checked against the
//! device's limits. What only this host checks (that a tensor the kernel writes may be written) is checked with the dtype of each tensor.
//!
//! A dimension, a size and an element count are exact here, as Python's integers are, so that no count is refused before the check Python
//! refuses it with (a broadcast view of `[2^62, 2^62, 127]` has its static axis checked first, an element count is too large for the scalar
//! that carries it where the scalars are checked): a dimension is an integer of 128 bits, and an element count stops at `2^128 - 1` (only a
//! view broadcast along huge axes has more), which every limit refuses. A call has nothing but its tensors: there is no way to tell a launch
//! what a dimension is.

use std::sync::Arc;

use crate::error::{Error, PreconditionKind, Result};
use crate::expr::{Expr, MAX_DIMS, Pattern, Solved, Symbols, Unsolvable, parse_grid};
use crate::library::LibraryEntry;
use crate::manifest::{
    Access, CUTE_ADDRESS_LIMIT, DESCRIPTOR_BYTES, DimInfo, Entry, LaunchAttributes, Param, ParamKind, Persistent,
    Precondition, SCRATCH_ALIGNMENT, ScalarType, ShapeDim, StrideEntry,
};
use crate::tensor::{DType, TensorArg, reach};

/// The widest `kernelParams` array: the arguments of a kernel plus its scratch slots.
pub(crate) const MAX_PARAMS: usize = 64;

/// The device a launch is checked against: its ordinal, the CUDA context the kernel is loaded in, and the largest grid it takes
/// along each axis (`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X/Y/Z`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceInfo {
    /// The ordinal of the device the kernel is loaded on: every tensor must be on it.
    pub ordinal: i32,
    /// The address of the `CUcontext` the kernel is loaded in, when there is one (a made-up device for a test has none): a tensor whose
    /// memory is cudarc's and was allocated in another context, even of the same device, is refused.
    pub context: Option<usize>,
    /// The largest grid size along x, y and z.
    pub max_grid: [i64; 3],
}

/// One axis of the strides a pointer takes: a stride dimension to bind (or an expression over them), or the fixed unit stride (the
/// constant 1, whose mismatch is a refusal of its own kind, `stride`).
#[derive(Debug, Clone)]
struct StridePlan {
    pattern: Pattern,
}

#[derive(Debug, Clone)]
struct TensorPlan {
    name: String,
    dtype: DType,
    access: Access,
    /// A device scalar: a pointer with the empty shape, one element in device memory.
    device_scalar: bool,
    shape: Vec<Pattern>,
    /// The strides of a pointer that takes a view, one per axis; `None` is a dense, row-major tensor.
    strides: Option<Vec<StridePlan>>,
    /// The alignment the pointer passed must have (16 bytes, or for a strided pointer promised only its element's alignment, that).
    alignment: u64,
    /// An `out` pointer that may be written over an input (`alias`): the index of that input, which it may be exactly (not overlap otherwise).
    alias: Option<usize>,
    /// A TMA descriptor of the tensor: what its map is encoded with besides the tensor.
    map: Option<MapPlan>,
    /// The TMA descriptor of a CuTe DSL kernel's tensor (manifest kind `tensormap`): its template, into which a launch writes the address.
    template: Option<Box<[u8; DESCRIPTOR_BYTES]>>,
}

/// The map of a TMA descriptor parameter: the numbers of `cuTensorMapEncodeTiled` that are not the tensor's.
#[derive(Debug, Clone)]
struct MapPlan {
    data_type: u32,
    /// The box, innermost first.
    box_dim: Vec<u32>,
    swizzle: u32,
    l2_promotion: u32,
    oob_fill: u32,
}

/// The arguments of one `cuTensorMapEncodeTiled` call of a launch (a TMA descriptor parameter): every number of it, the innermost
/// dimension first, as both hosts compute them (Python's `_tensormap.encode_arguments`): the tensor's address, its dimensions, the strides
/// of its outer axes in bytes (then, as Triton's `fillTMADescriptor` passes them, the bytes of the whole tensor: `rank` entries, of which
/// the driver reads `rank - 1`), the box, element strides of 1, interleave `NONE` (0), and the map's swizzle, L2 promotion and fill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMapEncode {
    /// `CUtensorMapDataType`.
    pub data_type: u32,
    /// The number of dimensions, 1 to 5.
    pub rank: u32,
    /// The address of the tensor's first element.
    pub address: u64,
    /// Its dimensions, innermost first.
    pub global_dim: Vec<u64>,
    /// The strides of its outer axes in bytes, innermost first, then its bytes in all.
    pub global_strides: Vec<u64>,
    /// The box, innermost first.
    pub box_dim: Vec<u32>,
    /// The element strides (all 1).
    pub element_strides: Vec<u32>,
    /// `CUtensorMapInterleave` (0: none).
    pub interleave: u32,
    /// `CUtensorMapSwizzle`.
    pub swizzle: u32,
    /// `CUtensorMapL2promotion`.
    pub l2_promotion: u32,
    /// `CUtensorMapFloatOOBfill`.
    pub oob_fill: u32,
}

/// One TMA descriptor of a checked launch: its slot of `kernelParams`, the tensor it is of, the arguments it is encoded with, and its bytes
/// once a loaded kernel has encoded it ([`crate::Kernel::check`]).
#[derive(Debug, Clone)]
pub(crate) struct MapSlot {
    pub(crate) slot: usize,
    pub(crate) encode: TensorMapEncode,
    pub(crate) bytes: Option<crate::ffi::TensorMapBytes>,
}

/// The 128 bytes of a CuTe DSL kernel's TMA descriptor as a launch passes them, 64-byte aligned in the host's memory as the PTX parameter is
/// (`.param .align 64 .b8 [128]`).
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(64))]
pub(crate) struct Descriptor(pub(crate) [u8; DESCRIPTOR_BYTES]);

impl std::fmt::Debug for Descriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Descriptor(")?;
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        write!(f, ")")
    }
}

/// What a launch passes by value at a slot of `kernelParams` that is no 64-bit word of [`Prepared::params`] (a parameter of a CuTe DSL kernel,
/// manifest schema 3 section 4.9).
#[derive(Debug, Clone)]
pub(crate) enum PassedBytes {
    /// A TMA descriptor (kind `tensormap`): its template with the tensor's address in bytes 0 to 7.
    Descriptor(Descriptor),
    /// Constant bytes (kind `bytes`), as the manifest has them (shared with the plan).
    Constant(Arc<[u8]>),
}

impl PassedBytes {
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Descriptor(d) => &d.0,
            Self::Constant(bytes) => bytes,
        }
    }
}

/// One parameter a checked launch passes by value from bytes of its own ([`PassedBytes`]): its slot and the bytes.
#[derive(Debug, Clone)]
pub(crate) struct PassedSlot {
    pub(crate) slot: usize,
    pub(crate) bytes: PassedBytes,
}

/// The 128 bytes a launch passes for a CuTe DSL kernel's TMA descriptor (manifest kind `tensormap`): its `template` with the tensor's `address`,
/// a little-endian `u64`, in bytes 0 to 7 (Python's `_cute.descriptor`; the template is zero there). The address is not null, 16-byte aligned
/// and below 2**57 (the DSL keeps 53 bits of `address / 16`): `None` for any other.
pub(crate) fn descriptor(template: &[u8; DESCRIPTOR_BYTES], address: u64) -> Option<Descriptor> {
    if address == 0 || !address.is_multiple_of(16) || address >= CUTE_ADDRESS_LIMIT {
        return None;
    }
    let mut bytes = *template;
    bytes[..8].copy_from_slice(&address.to_le_bytes());
    Some(Descriptor(bytes))
}

#[derive(Debug, Clone)]
struct ScalarPlan {
    name: String,
    ty: ScalarType,
    dim: Pattern,
    count_of: Option<usize>,
}

/// What the manifest says about the named dimension of one symbol.
#[derive(Debug, Clone)]
struct DimPlan {
    divisible_by: Option<i64>,
    /// The buckets of a bucketed dimension: its value must be one of them.
    buckets: Option<Vec<u64>>,
    /// The bucket this binary is built for: its value must be that one.
    bucket: Option<i64>,
}

/// A precondition of the binary that is checked where Python's `check_spec_preconditions` checks it, after the scalars and the aliasing,
/// in the order the manifest lists them (argument after argument: the alignment of a pointer and its `max_elements`, then the next one).
/// The other two kinds are checked before: a `static_size` as the dimensions are bound, a `bucket` where the binary is chosen.
#[derive(Debug, Clone, Copy)]
enum Check {
    /// The pointer of the tensor is aligned to its `alignment`.
    Alignment(usize),
    /// The tensor has at most this many elements.
    MaxElements(usize, i64),
    /// The view spans at most this many elements.
    MaxExtent(usize, i64),
    /// The runtime scalar is a multiple of this.
    Divisible(usize, i64),
}

#[derive(Debug, Clone, Copy)]
enum Slot {
    Tensor(usize),
    Scalar(usize),
    /// A `fixed` argument: its bits in the low bytes of the slot.
    Fixed(u64),
    /// A fixed argument per call: the index of its value in what the call gives.
    Given(usize),
    /// The `CUtensorMap` of a TMA descriptor of the tensor: encoded from it at the launch, passed by value.
    Map(usize),
    /// The size of an axis of the tensor of a TMA descriptor (an `i32`), after its map.
    MapShape(usize, usize),
    /// The element stride of an axis of the tensor of a TMA descriptor (an `i64`), after its shape.
    MapStride(usize, usize),
    /// The TMA descriptor of a CuTe DSL kernel's tensor: its template with the tensor's address, 128 bytes passed by value.
    Descriptor(usize),
    /// Constant bytes of a CuTe DSL kernel: the index of its bytes among the plan's `constants`, passed by value.
    Bytes(usize),
}

/// One manifest entry, ready to check a call against.
#[derive(Debug, Clone)]
pub struct Plan {
    stem: String,
    syms: Symbols,
    dims: Vec<DimPlan>,
    tensors: Vec<TensorPlan>,
    scalars: Vec<ScalarPlan>,
    /// The fixed arguments per call (`given`), in the order of the ABI: their names and how each is passed.
    given: Vec<(String, ScalarType)>,
    checks: Vec<Check>,
    slots: Vec<Slot>,
    /// The constant bytes of a CuTe DSL kernel's parameters (kind `bytes`), in the order of the ABI ([`Slot::Bytes`]).
    constants: Vec<Arc<[u8]>>,
    scratch_slots: usize,
    /// The entries of the grid, evaluated in their folded form as Python evaluates its grid.
    grid: Vec<Pattern>,
    grid_text: String,
    block: [u32; 3],
    shared: u32,
    /// A persistent grid: the grid is the tiles, cut to the programs the SMs of the device run (manifest schema 3, section 4.3).
    persistent: Option<Persistent>,
    /// The cluster a launch groups its blocks in (`cuLaunchKernelEx`), for a kernel of more than one CTA per program; the grid counts clusters.
    cluster: Option<[u32; 3]>,
    /// The CTAs of a program (1, or the cluster's).
    num_ctas: u32,
    /// The bytes of global scratch each CTA addresses (a launch is given a scratch tensor of that many bytes per CTA: the last tensor).
    scratch: Option<u64>,
}

/// The name [`Plan::tensor_names`] gives the global scratch of a kernel that has one: the last tensor of a launch.
pub const GLOBAL_SCRATCH: &str = "global_scratch";

/// What a checked call launches with.
#[derive(Debug, Clone)]
pub struct Prepared {
    /// The grid of blocks a launch has, three sizes: the entry's grid of programs (a persistent grid cut to the programs the SMs of the device
    /// run), times the cluster of a kernel of clusters.
    pub grid: [u32; 3],
    /// The cluster the blocks are launched in (`cuLaunchKernelEx`'s cluster attribute), for a kernel of more than one CTA per program.
    pub cluster: Option<[u32; 3]>,
    /// The bytes of global scratch the launch gives the kernel (its scratch tensor has at least this many; 0 for a kernel that has none).
    pub scratch_bytes: u64,
    pub(crate) values: [u64; MAX_PARAMS],
    pub(crate) count: usize,
    /// For each slot, the index of the tensor it holds (`NOT_A_TENSOR` for a scalar, a fixed argument or a scratch slot). The slot of the map
    /// of a TMA descriptor holds its tensor too (a launch orders the tensor's memory as a pointer's, and passes the map).
    pub(crate) tensor_of: [u8; MAX_PARAMS],
    /// The TMA descriptors of the call, in the order of their slots.
    pub(crate) maps: Vec<MapSlot>,
    /// The parameters a launch passes by value from bytes of their own (a CuTe DSL kernel's descriptors and constant bytes), in the order of
    /// their slots.
    pub(crate) passed: Vec<PassedSlot>,
}

/// What `Prepared::tensor_of` holds for a slot that is not a tensor.
pub(crate) const NOT_A_TENSOR: u8 = u8::MAX;

impl Prepared {
    /// The `kernelParams` values, one 64-bit slot each: an address, an `i32` or an `f32` in the low four bytes, an `i64`, then the scratch
    /// slots of a Triton kernel (null pointers; a prebuilt binary has none). The slot of the map of a TMA descriptor holds 0 here: the launch
    /// passes the 128 bytes of the map it encoded ([`Prepared::tensor_maps`]); its shape and strides follow in slots of their own. So does the
    /// slot of a CuTe DSL kernel's descriptor or constant bytes: the launch passes [`Prepared::by_value`] there.
    pub fn params(&self) -> &[u64] {
        &self.values[..self.count]
    }

    /// The bytes a launch passes by value at `slot` for a parameter of a CuTe DSL kernel that is no 64-bit word (manifest schema 3, section
    /// 4.9): the 128 bytes of a TMA descriptor (kind `tensormap`: its template with the address of the call's tensor, little-endian, in bytes 0
    /// to 7), or constant bytes (kind `bytes`, as the manifest has them). `None` for any other slot.
    pub fn by_value(&self, slot: usize) -> Option<&[u8]> {
        self.passed.iter().find(|p| p.slot == slot).map(|p| p.bytes.as_slice())
    }

    /// The `cuTensorMapEncodeTiled` arguments of each TMA descriptor of the call, in the order of their slots: what a launch encodes.
    pub fn tensor_maps(&self) -> impl Iterator<Item = &TensorMapEncode> {
        self.maps.iter().map(|m| &m.encode)
    }

    /// The map that a launch passes at `slot`, when that is the slot of a TMA descriptor (`None` for any other slot). It has its bytes once a
    /// loaded kernel has checked the call.
    pub(crate) fn map_at(&self, slot: usize) -> Option<&MapSlot> {
        self.maps.iter().find(|m| m.slot == slot)
    }
}

fn to_i64(v: u64, what: &str) -> Result<i64> {
    i64::try_from(v).map_err(|_| Error::manifest(format!("{what} {v} does not fit 64 bits")))
}

/// Everything `Plan` is made of, from a kernel entry or a library entry.
struct Source<'a> {
    stem: &'a str,
    arch: &'a str,
    params: &'a [Param],
    dims: &'a [DimInfo],
    preconditions: &'a [Precondition],
    grid: &'a str,
    block: [u32; 3],
    shared: u32,
    scratch_slots: usize,
    launch: Option<&'a LaunchAttributes>,
    persistent: Option<Persistent>,
}

impl Plan {
    /// The plan of a manifest entry. Refuses a binary that a plain `cuLaunchKernel` cannot run (it needs scratch memory,
    /// programmatic dependent launch, clusters or a cooperative launch) and a pointer of a packed element type.
    pub fn new(entry: &Entry) -> Result<Self> {
        Self::build(&Source {
            stem: &entry.stem,
            arch: &entry.arch,
            params: &entry.params,
            dims: &entry.dims,
            preconditions: &entry.preconditions,
            grid: &entry.grid,
            block: entry.block,
            shared: entry.shared,
            scratch_slots: entry.scratch_slots,
            launch: Some(&entry.launch),
            persistent: entry.persistent,
        })
    }

    /// The plan of a library entry (a pinned GEMM): what its operands are checked against. It launches nothing by itself (its grid is
    /// `(1,)` and it has no scalars): the checks are those of a kernel, on the pointers `a`, `b`, `bias`, `c` and `out` it has.
    pub(crate) fn for_library(entry: &LibraryEntry) -> Result<Self> {
        Self::build(&Source {
            stem: &entry.stem,
            arch: &entry.arch,
            params: &entry.params,
            dims: &entry.dims,
            preconditions: &entry.preconditions,
            grid: "(1,)",
            block: [1, 1, 1],
            shared: 0,
            scratch_slots: 0,
            launch: None,
            persistent: None,
        })
    }

    /// The plan of a convolution entry: what its tensors `x`, `w` and `out` are checked against (it launches nothing by itself).
    pub(crate) fn for_conv(entry: &crate::conv::ConvEntry) -> Result<Self> {
        Self::build(&Source {
            stem: &entry.stem,
            arch: &entry.arch,
            params: &entry.params,
            dims: &entry.dims,
            preconditions: &entry.preconditions,
            grid: "(1,)",
            block: [1, 1, 1],
            shared: 0,
            scratch_slots: 0,
            launch: None,
            persistent: None,
        })
    }

    fn build(source: &Source<'_>) -> Result<Self> {
        let who = format!("'{}' ({})", source.stem, source.arch);
        let blockers = source.launch.map(LaunchAttributes::blockers).unwrap_or_default();
        if !blockers.is_empty() {
            return Err(Error::Unsupported(format!(
                "{who}: the binary needs launch attributes that this host does not set up: {}",
                blockers.join(", ")
            )));
        }
        let slot_count: usize = source.params.iter().map(|p| p.slots().len()).sum();
        if slot_count + source.scratch_slots > MAX_PARAMS {
            return Err(Error::Unsupported(format!(
                "{who}: a kernel with {} parameters ({slot_count} slots of kernelParams) is not supported",
                source.params.len()
            )));
        }
        let bad = |e: String| Error::manifest(format!("{who}: {e}"));
        // the named dimensions come first, so that a symbol's number is its place in the table of the manifest
        let mut syms = Symbols::default();
        let mut dims = Vec::with_capacity(source.dims.len());
        for d in source.dims {
            syms.intern(&d.name).map_err(bad)?;
            dims.push(DimPlan {
                divisible_by: d.divisible_by.map(|v| to_i64(v, "a divisor")).transpose()?,
                buckets: d.buckets.clone(),
                bucket: d.bucket.map(|b| to_i64(b, "a bucket")).transpose()?,
            });
        }
        let mut tensors: Vec<TensorPlan> = Vec::new();
        let mut scalars: Vec<ScalarPlan> = Vec::new();
        let mut given: Vec<(String, ScalarType)> = Vec::new();
        let mut slots = Vec::new();
        let mut constants: Vec<Arc<[u8]>> = Vec::new();
        for p in source.params {
            match &p.kind {
                ParamKind::Pointer { dtype_name, dtype, access, shape, view, tensor_map, template, .. } => {
                    let Some(dtype) = *dtype else {
                        return Err(Error::Unsupported(format!(
                            "{who}: parameter '{}' has dtype {dtype_name}, which has no storage type here (packed formats cannot be run yet)",
                            p.name
                        )));
                    };
                    let mut patterns = Vec::with_capacity(shape.len());
                    for (axis, d) in shape.iter().enumerate() {
                        patterns.push(match d {
                            ShapeDim::Static(n) => Pattern::new(&n.to_string(), Expr::Num(i128::from(*n))),
                            ShapeDim::Expr(e) => {
                                Pattern::parse(e, &format!("the shape of '{}' axis {axis}", p.name), &mut syms)
                                    .map_err(bad)?
                            }
                        });
                    }
                    let strides = match view {
                        Some(view) => {
                            let mut out = Vec::with_capacity(view.strides.len());
                            for (axis, s) in view.strides.iter().enumerate() {
                                out.push(match s {
                                    StrideEntry::Unit => StridePlan { pattern: Pattern::new("1", Expr::Num(1)) },
                                    StrideEntry::Dim(e) => StridePlan {
                                        pattern: Pattern::parse(
                                            e,
                                            &format!("the stride of '{}' axis {axis}", p.name),
                                            &mut syms,
                                        )
                                        .map_err(bad)?,
                                    },
                                });
                            }
                            syms.intern(&view.offset).map_err(bad)?;
                            Some(out)
                        }
                        None => None,
                    };
                    let t = tensors.len();
                    let map = tensor_map.as_ref().map(|m| MapPlan {
                        data_type: m.data_type_value(),
                        box_dim: m.box_shape.iter().rev().copied().collect(),
                        swizzle: m.swizzle_value(),
                        l2_promotion: m.l2_promotion_value(),
                        oob_fill: u32::from(m.oob_nan),
                    });
                    if map.is_some() {
                        // the map, then the shape (i32 each) and the element strides (i64 each) of the tensor: Triton's expansion
                        slots.push(Slot::Map(t));
                        slots.extend((0..shape.len()).map(|axis| Slot::MapShape(t, axis)));
                        slots.extend((0..shape.len()).map(|axis| Slot::MapStride(t, axis)));
                    } else if template.is_some() {
                        // a CuTe DSL kernel's descriptor: its template with the tensor's address, one slot of 128 bytes
                        slots.push(Slot::Descriptor(t));
                    } else {
                        slots.push(Slot::Tensor(t));
                    }
                    tensors.push(TensorPlan {
                        name: p.name.clone(),
                        dtype,
                        access: *access,
                        device_scalar: shape.is_empty(),
                        shape: patterns,
                        strides,
                        alignment: 16,
                        alias: None,
                        map,
                        template: template.clone(),
                    });
                }
                ParamKind::Scalar { ty, dim, .. } => {
                    slots.push(Slot::Scalar(scalars.len()));
                    let dim = Pattern::parse(dim, &format!("parameter {} dim", p.name), &mut syms).map_err(bad)?;
                    scalars.push(ScalarPlan { name: p.name.clone(), ty: *ty, dim, count_of: None });
                }
                ParamKind::Fixed { value } => slots.push(Slot::Fixed(value.bits())),
                // a null pointer: the 64-bit zero at its slot, as the bits of a fixed argument are written
                ParamKind::Null { .. } => slots.push(Slot::Fixed(0)),
                ParamKind::Given { ty } => {
                    slots.push(Slot::Given(given.len()));
                    given.push((p.name.clone(), *ty));
                }
                // constant bytes of a CuTe DSL kernel: passed by value at its slot, as the manifest has them (a call gives nothing for it)
                ParamKind::Bytes { value } => {
                    slots.push(Slot::Bytes(constants.len()));
                    constants.push(Arc::from(value.as_slice()));
                }
            }
        }
        for p in source.params {
            if let ParamKind::Pointer { alias: Some(target), .. } = &p.kind
                && let (Some(w), Some(t)) =
                    (tensors.iter().position(|t| t.name == p.name), tensors.iter().position(|t| t.name == *target))
            {
                tensors[w].alias = Some(t);
            }
        }
        for p in source.params {
            if let ParamKind::Scalar { count_of: Some(target), .. } = &p.kind
                && let (Some(s), Some(t)) =
                    (scalars.iter().position(|s| s.name == p.name), tensors.iter().position(|t| t.name == *target))
            {
                scalars[s].count_of = Some(t);
            }
        }
        let mut checks = Vec::with_capacity(source.preconditions.len());
        for pre in source.preconditions {
            let tensor = |arg: &str| tensors.iter().position(|t| t.name == arg);
            match pre {
                Precondition::PointerAlignment { arg, bytes, .. } => {
                    if let Some(t) = tensor(arg) {
                        tensors[t].alignment = *bytes;
                        checks.push(Check::Alignment(t));
                    }
                }
                Precondition::MaxElements { arg, value, .. } => {
                    if let Some(t) = tensor(arg) {
                        checks.push(Check::MaxElements(t, to_i64(*value, "max_elements")?));
                    }
                }
                Precondition::MaxExtent { arg, value, .. } => {
                    if let Some(t) = tensor(arg) {
                        checks.push(Check::MaxExtent(t, to_i64(*value, "max_extent")?));
                    }
                }
                Precondition::Divisible { arg, by, .. } => {
                    if let Some(s) = scalars.iter().position(|s| s.name == *arg) {
                        checks.push(Check::Divisible(s, to_i64(*by, "a divisor")?));
                    }
                }
                // a static size is checked as the dimensions are bound (it is what the shape says), a bucket where the binary is chosen
                // (it is what `dims` says): both before these
                Precondition::StaticSize { .. } | Precondition::Bucket { .. } => {}
            }
        }
        // a pointer the manifest promises no alignment for (the reader refuses such an entry) is still held to 16 bytes, after the rest
        for t in 0..tensors.len() {
            if !checks.iter().any(|c| matches!(c, Check::Alignment(u) if *u == t)) {
                checks.push(Check::Alignment(t));
            }
        }
        let grid = parse_grid(source.grid, &mut syms)
            .map_err(bad)?
            .into_iter()
            .map(|e| Pattern::new(source.grid, e))
            .collect();
        if syms.len() != dims.len() {
            return Err(Error::manifest(format!("{who}: the entry's dims do not name every dimension it uses")));
        }
        Ok(Self {
            stem: source.stem.to_owned(),
            syms,
            dims,
            tensors,
            scalars,
            given,
            checks,
            slots,
            constants,
            scratch_slots: source.scratch_slots,
            grid,
            grid_text: source.grid.to_owned(),
            block: source.block,
            shared: source.shared,
            persistent: source.persistent,
            cluster: source.launch.and_then(LaunchAttributes::cluster),
            num_ctas: source.launch.map_or(1, |l| l.num_ctas.max(1)),
            scratch: source.launch.map(|l| l.global_scratch_size).filter(|s| *s > 0 && source.scratch_slots > 0),
        })
    }

    /// The stem of the entry.
    pub fn stem(&self) -> &str {
        &self.stem
    }

    /// The names of the tensor parameters, in the order a launch gives its tensors, and last [`GLOBAL_SCRATCH`] for a kernel with a global
    /// scratch (a writable `u8` tensor of [`Prepared::scratch_bytes`] bytes, 256-byte aligned, that overlaps no other tensor).
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|t| t.name.as_str()).chain(self.scratch.map(|_| GLOBAL_SCRATCH))
    }

    /// The bytes of global scratch each CTA of the kernel addresses (`None`: it has none): a launch is given that many per CTA of its grid.
    pub fn global_scratch(&self) -> Option<u64> {
        self.scratch
    }

    /// The scratch slots a launch passes after the parameters: 2 for a Triton kernel (the global scratch, the profile scratch), 0 for a prebuilt
    /// binary.
    pub fn scratch_slots(&self) -> usize {
        self.scratch_slots
    }

    /// The persistent grid of the entry, when it has one.
    pub fn persistent(&self) -> Option<Persistent> {
        self.persistent
    }

    /// The cluster a launch groups its blocks in, for a kernel of more than one CTA per program (`cuLaunchKernelEx`).
    pub fn cluster(&self) -> Option<[u32; 3]> {
        self.cluster
    }

    /// The names of the fixed arguments per call (a prebuilt binary's `const[i32]` or `const[i64]` with no default), in the order of the ABI:
    /// what [`Plan::prepare_with`] takes a value for.
    pub fn given_names(&self) -> impl Iterator<Item = &str> {
        self.given.iter().map(|(n, _)| n.as_str())
    }

    /// The thread block every launch of the entry takes: the entry's `block`, which for a Triton kernel is `32 *` the warps of the compiled
    /// kernel (all the partitions of a warp-specialised one, its `total_num_warps`).
    pub fn block(&self) -> [u32; 3] {
        self.block
    }

    pub(crate) fn shared(&self) -> u32 {
        self.shared
    }

    fn describe_tensor(&self, t: usize) -> String {
        let p = &self.tensors[t];
        let shape = p.shape.iter().map(|d| d.text.as_str()).collect::<Vec<_>>().join(", ");
        if p.device_scalar {
            format!("DevScalar[{}]", p.dtype.name())
        } else if p.strides.is_some() {
            format!("Strided[Tensor[{shape}, {}]]", p.dtype.name())
        } else {
            format!("Tensor[{shape}, {}]", p.dtype.name())
        }
    }

    fn unsolvable(&self, origin: Origin, pattern: &Pattern, actual: i128, e: Unsolvable) -> Error {
        let name_of = |i: usize| self.syms.name(i).to_owned();
        let (t, axis, stride) = match origin {
            Origin::Tensor { t, axis } => (t, axis, false),
            Origin::Stride { t, axis } => (t, axis, true),
        };
        let name = &self.tensors[t].name;
        let which = if stride { format!("the stride of axis {axis}") } else { format!("dimension {axis}") };
        let head = format!("'{}': argument '{name}' ({}): {which} is {actual}", self.stem, self.describe_tensor(t));
        match e {
            Unsolvable::Eval(message) => Error::Launch(format!("'{}': {message}", self.stem)),
            Unsolvable::Mismatch { expected } => {
                let what =
                    if pattern.is_static { expected.to_string() } else { format!("{} = {expected}", pattern.text) };
                let note = match (stride, pattern.is_static) {
                    (true, true) => {
                        " (the kernel loads along that axis contiguously: pass a view whose last axis has stride 1)"
                    }
                    (true, false) => " (the kernel takes one stride for the views that share it: each must have it)",
                    (false, _) => "",
                };
                // a stride that is not the one required is a refusal of kind `stride` (Python's `_solve_one`): the fixed unit stride of the
                // last axis, or a stride dimension that another view bound first (views that share a stride)
                let kind = match (pattern.is_static, stride) {
                    (_, true) => PreconditionKind::Stride,
                    (true, false) => PreconditionKind::StaticSize,
                    (false, false) => PreconditionKind::Shape,
                };
                Error::precondition(kind, Some(name), format!("{head}, but the signature requires {what}{note}"))
            }
            Unsolvable::NotWhole { var, target, coefficient } => Error::precondition(
                PreconditionKind::Shape,
                Some(name),
                format!(
                    "{head}, which cannot match {} (would need {} = {target}/{coefficient})",
                    pattern.text,
                    name_of(var)
                ),
            ),
            Unsolvable::NotPositive { var, value } => Error::precondition(
                PreconditionKind::Shape,
                Some(name),
                format!("{head}, which cannot match {} (would need {} = {value})", pattern.text, name_of(var)),
            ),
        }
    }

    /// One tensor against the parameter it is given for, on its own (Python's `check_tensor`): its dtype, the one element of a device
    /// scalar, its rank, that no dimension is empty, and that a tensor given for a parameter that is not strided is contiguous. Before
    /// those, what only this host has to check: that a tensor the kernel writes may be written.
    fn check_tensor(&self, i: usize, a: &TensorArg<'_>) -> Result<()> {
        let (t, stem) = (&self.tensors[i], self.stem.as_str());
        if t.access != Access::In && a.access == Access::In {
            return Err(Error::precondition(
                PreconditionKind::Access,
                Some(&t.name),
                format!(
                    "argument '{}' is a read-only tensor, but '{stem}' writes it ({}): give it a tensor that may be written",
                    t.name,
                    t.access.as_str()
                ),
            ));
        }
        if a.dtype != t.dtype {
            return Err(Error::precondition(
                PreconditionKind::Dtype,
                Some(&t.name),
                format!(
                    "argument '{}' has dtype {}, but the signature requires {}",
                    t.name,
                    a.dtype.name(),
                    t.dtype.name()
                ),
            ));
        }
        if t.device_scalar {
            // one element of device memory: a tensor of shape [] or [1]
            if !(a.shape.is_empty() || a.shape == [1]) {
                return Err(Error::precondition(
                    PreconditionKind::Shape,
                    Some(&t.name),
                    format!(
                        "argument '{}' has shape {:?}, but the signature requires {}: one element in device memory (a tensor of shape [] or [1])",
                        t.name,
                        a.shape,
                        self.describe_tensor(i)
                    ),
                ));
            }
            return Ok(());
        }
        if a.shape.len() != t.shape.len() {
            return Err(Error::precondition(
                PreconditionKind::Rank,
                Some(&t.name),
                format!(
                    "argument '{}' has rank {} (shape {:?}), but the signature requires {} (rank {})",
                    t.name,
                    a.shape.len(),
                    a.shape,
                    self.describe_tensor(i),
                    t.shape.len()
                ),
            ));
        }
        if a.shape.contains(&0) {
            return Err(Error::precondition(
                PreconditionKind::Shape,
                Some(&t.name),
                format!(
                    "argument '{}' is empty (shape {:?}); a binary needs every dimension to be at least 1",
                    t.name, a.shape
                ),
            ));
        }
        if t.strides.is_none()
            && let Some(strides) = a.strides
            && !is_contiguous(a.shape, strides)
        {
            return Err(Error::precondition(
                PreconditionKind::Contiguous,
                Some(&t.name),
                format!(
                    "argument '{}' is not contiguous (row-major, no gaps; shape {:?}, strides {:?}); this parameter takes a contiguous \
                     tensor: pass it contiguous, or have the binary declare it Strided[...]",
                    t.name, a.shape, strides
                ),
            ));
        }
        Ok(())
    }

    /// The equations of the tensor `t` (Python's `bind_infos`): each axis of its declared shape against its size, then each stride of a
    /// view against its canonical stride (`canonical`).
    fn equations_of<'p>(
        &'p self,
        t: usize,
        a: &TensorArg<'_>,
        canonical: &[u128],
        out: &mut Vec<Equation<'p>>,
    ) -> Result<()> {
        let plan = &self.tensors[t];
        for (axis, (pattern, size)) in plan.shape.iter().zip(a.shape).enumerate() {
            out.push(Equation { pattern, actual: *size as i128, origin: Origin::Tensor { t, axis }, minimum: 1 });
        }
        if let Some(strides) = &plan.strides {
            for (axis, (stride, value)) in strides.iter().zip(canonical).enumerate() {
                // (a view that fits its memory has canonical strides below 2^66: this cannot fail)
                let actual = i128::try_from(*value).map_err(|_| {
                    Error::Launch(format!(
                        "'{}': argument '{}': the stride of axis {axis} is {value}, more than this host can count",
                        self.stem, plan.name
                    ))
                })?;
                // a stride can be 0 (an expanded view)
                out.push(Equation { pattern: &stride.pattern, actual, origin: Origin::Stride { t, axis }, minimum: 0 });
            }
        }
        Ok(())
    }

    /// The constraints the manifest gives the dimensions `vars`, in that order (Python's `check_dim_constraints`, whose dimensions are in the
    /// order they were bound): each is a multiple of its `divisible_by`, and a bucketed one is one of its buckets.
    fn check_dims(&self, dims: &[Option<i128>], vars: &[usize]) -> Result<()> {
        for &i in vars {
            let (info, name) = (&self.dims[i], self.syms.name(i));
            let Some(v) = dims[i] else { continue };
            if let Some(by) = info.divisible_by
                && v.checked_rem(i128::from(by)).is_some_and(|r| r != 0)
            {
                return Err(Error::precondition(
                    PreconditionKind::Divisible,
                    Some(name),
                    format!("dimension {name} is {v}, but it is declared divisible_by={by}"),
                ));
            }
            if let Some(all) = &info.buckets
                && !u64::try_from(v).is_ok_and(|v| all.contains(&v))
            {
                return Err(Error::precondition(
                    PreconditionKind::Bucket,
                    Some(name),
                    format!(
                        "dimension {name} is {v}, which is not one of its buckets {all:?}; a bucketed dimension must equal a bucket exactly (there \
                         is no padding: pad or slice in the host)"
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Check a call against everything the manifest says, and make the grid and the `kernelParams` values for it.
    ///
    /// `tensors` are the tensors of the call in the order of the manifest's ABI (see [`Plan::tensor_names`]). `device` is the
    /// device the kernel is loaded on: every tensor must be on it, and the grid must fit what it takes (`None` skips both).
    ///
    /// The checks are made in the order Python's `runtime._prepare` makes them (see the module's documentation), so that a call that breaks
    /// several preconditions is refused for the one Python finds first.
    pub fn prepare(&self, tensors: &[TensorArg<'_>], device: Option<&DeviceInfo>) -> Result<Prepared> {
        self.prepare_with(tensors, &[], device)
    }

    /// [`Plan::prepare`] of a call that gives the values of the binary's fixed arguments per call (`given`, by name: every one of
    /// [`Plan::given_names`] once, and nothing else; each must fit the `i32` or `i64` it is passed as). A value is checked where Python's
    /// `_finish_launch` checks it, after the runtime scalars; a name that is missing, unknown or given twice is refused first.
    pub fn prepare_with(
        &self,
        tensors: &[TensorArg<'_>],
        given: &[(&str, i64)],
        device: Option<&DeviceInfo>,
    ) -> Result<Prepared> {
        self.prepare_for(tensors, given, device, None)
    }

    /// [`Plan::prepare_with`] for a device of `sm_count` SMs (or a partition of that many: [`Manifest::with_sm_partition`](crate::Manifest)):
    /// a persistent grid (manifest schema 3, section 4.3) is cut to the programs they run, at most `programs_per_sm` per SM (`None`: the grid is
    /// the tiles, uncut; a loaded kernel's [`Kernel::check`](crate::Kernel::check) gives its device's count).
    pub fn prepare_for(
        &self,
        tensors: &[TensorArg<'_>],
        given: &[(&str, i64)],
        device: Option<&DeviceInfo>,
        sm_count: Option<u64>,
    ) -> Result<Prepared> {
        let stem = self.stem.as_str();
        let mut given_values = [0i64; MAX_PARAMS];
        let mut seen = [false; MAX_PARAMS];
        for (name, value) in given {
            let Some(i) = self.given.iter().position(|(n, _)| n == name) else {
                return Err(Error::Launch(format!(
                    "'{stem}' has no fixed argument per call '{name}' (it has {:?})",
                    self.given_names().collect::<Vec<_>>()
                )));
            };
            if std::mem::replace(&mut seen[i], true) {
                return Err(Error::Launch(format!("the fixed argument per call '{name}' of '{stem}' is given twice")));
            }
            given_values[i] = *value;
        }
        if let Some((missing, _)) = self.given.iter().enumerate().find(|(i, _)| !seen[*i]).map(|(_, g)| g) {
            return Err(Error::Launch(format!(
                "argument '{missing}' of '{stem}' is a fixed argument per call (a value every launch gives): give its value \
                 (Kernel::launch_with)"
            )));
        }
        let wanted = self.tensors.len() + usize::from(self.scratch.is_some());
        if tensors.len() != wanted {
            return Err(Error::Launch(format!(
                "'{stem}' takes {wanted} tensors ({}), got {}",
                self.tensor_names().collect::<Vec<_>>().join(", "),
                tensors.len()
            )));
        }
        // the global scratch of a kernel that has one: the last tensor (checked once the grid says how much the launch needs)
        let (tensors, scratch) = tensors.split_at(self.tensors.len());
        let scratch = scratch.first();
        let read = |t: usize| self.tensors[t].access != Access::Out;

        // -- the tensors the kernel reads (In and InOut, what Python's `bind_call` is given), each on its own, in order
        for (t, a) in tensors.iter().enumerate().filter(|(t, _)| read(*t)) {
            self.check_tensor(t, a)?;
        }
        let canonical: Vec<Vec<u128>> = self
            .tensors
            .iter()
            .zip(tensors)
            .map(|(t, a)| if t.strides.is_some() { canonical_strides(a.shape, a.strides) } else { Vec::new() })
            .collect();

        // -- the dimensions, bound from the sizes of those tensors and the canonical strides of their views (Python's `solve_dims`)
        let mut dims: [Option<i128>; MAX_DIMS] = [None; MAX_DIMS];
        let mut bound: Vec<usize> = Vec::with_capacity(MAX_DIMS); // in the order they are bound, the order Python checks them in
        let mut equations = Vec::new();
        for (t, a) in tensors.iter().enumerate().filter(|(t, _)| read(*t)) {
            self.equations_of(t, a, &canonical[t], &mut equations)?;
        }
        self.solve(&mut dims, &mut bound, &equations)?;

        // -- the dimensions those leave unbound, from the tensors the kernel writes (Python's `bind_outputs`): a dimension only an `Out`
        // parameter's shape mentions (a `[B]` mask an extern writes from a device scalar) is what the `Out` tensor given for it says. Each `Out`
        // tensor whose shape mentions one is checked on its own first, in the order of the ABI
        let open: Vec<usize> = (0..self.tensors.len())
            .filter(|&t| !read(t) && self.tensors[t].shape.iter().any(|p| p.vars.iter().any(|v| dims[*v].is_none())))
            .collect();
        if !open.is_empty() {
            let mut more = Vec::new();
            for &t in &open {
                self.check_tensor(t, &tensors[t])?;
                self.equations_of(t, &tensors[t], &canonical[t], &mut more)?;
            }
            self.solve(&mut dims, &mut bound, &more)?;
        }

        // -- what the manifest says of each dimension bound (Python's `check_dim_constraints`): divisible_by, one of its buckets
        self.check_dims(&dims, &bound)?;

        // -- the binary is the one of these buckets (Python's `select_spec`): a bucketed dimension is the bucket exactly (there is no
        // padding), and it is what the tensors say, never the bucket the binary was built for (one that no tensor determines is refused)
        for (i, info) in self.dims.iter().enumerate() {
            let Some(value) = info.bucket else { continue };
            let (name, all) = (self.syms.name(i), info.buckets.as_deref().unwrap_or_default());
            match dims[i] {
                Some(found) if found == i128::from(value) => {}
                Some(found) => {
                    return Err(Error::precondition(
                        PreconditionKind::Bucket,
                        Some(name),
                        format!(
                            "dimension {name} is {found}, but '{stem}' is built for exactly {value} (buckets {all:?}); there is no padding: \
                             pad or slice in the host to a bucket size"
                        ),
                    ));
                }
                None => {
                    return Err(Error::precondition(
                        PreconditionKind::Bucket,
                        Some(name),
                        format!(
                            "dimension {name} is not determined by the tensors, but '{stem}' is built for exactly {value} (buckets {all:?}): \
                             a bucketed dimension is the size of the tensors, it is never taken from the binary"
                        ),
                    ));
                }
            }
        }

        // -- every tensor is on the device, and in the CUDA context, the kernel is loaded in (Python's `_device_of`, once the binary is chosen)
        for (t, a) in self.tensors.iter().zip(tensors) {
            if let Some(d) = device.filter(|d| d.ordinal != a.device) {
                return Err(Error::precondition(
                    PreconditionKind::Device,
                    Some(&t.name),
                    format!(
                        "argument '{}' is on device {}, but '{stem}' is loaded on device {}",
                        t.name, a.device, d.ordinal
                    ),
                ));
            }
            if let (Some(kernel), Some(tensor)) = (device.and_then(|d| d.context), a.context)
                && kernel != tensor
            {
                return Err(Error::precondition(
                    PreconditionKind::Device,
                    Some(&t.name),
                    format!(
                        "argument '{}' is memory of another CUDA context than the one '{stem}' is loaded in (the same device, {}, but \
                         another context): memory can only be used in the context it was allocated in",
                        t.name, a.device
                    ),
                ));
            }
        }

        // -- the tensors the kernel writes (Out), which Python allocates from the dimensions bound so far: each must be what it allocates
        for (t, a) in tensors.iter().enumerate().filter(|(t, _)| !read(*t)) {
            self.check_tensor(t, a)?;
            let mut own = Vec::new();
            self.equations_of(t, a, &canonical[t], &mut own)?;
            for eq in &own {
                match eq.pattern.eval(&dims).map_err(|e| Error::Launch(format!("'{stem}': {e}")))? {
                    Some(expected) if expected == eq.actual => {}
                    Some(expected) => {
                        return Err(self.unsolvable(
                            eq.origin,
                            eq.pattern,
                            eq.actual,
                            Unsolvable::Mismatch { expected },
                        ));
                    }
                    None => {
                        return Err(Error::Launch(format!(
                            "'{stem}': argument '{}' ({}) is shaped {}, which the tensors the kernel reads do not determine (Python allocates an \
                             Out tensor from them)",
                            self.tensors[t].name,
                            self.describe_tensor(t),
                            eq.pattern.text
                        )));
                    }
                }
            }
        }

        // -- the runtime scalars (Python's `_finish_launch`): the element count of the tensor a scalar counts (which binds the dimension the
        // scalar is, when no tensor did), or the value of its dimension expression (or of the stride it carries)
        let counted = bound.len();
        let mut scalar_values = [0i128; MAX_PARAMS];
        for (s, scalar) in self.scalars.iter().enumerate() {
            let value = scalar.dim.eval(&dims).map_err(|e| Error::Launch(format!("'{stem}': {e}")))?;
            scalar_values[s] = match scalar.count_of {
                Some(t) => {
                    let count = reach(tensors[t].shape, None);
                    if let Some(v) = value
                        && u128::try_from(v).ok() != Some(count)
                    {
                        return Err(Error::precondition(
                            PreconditionKind::Shape,
                            Some(&scalar.name),
                            format!(
                                "'{stem}': runtime argument '{}' is {} = {v}, but the tensor '{}' has {count} elements",
                                scalar.name, scalar.dim.text, self.tensors[t].name
                            ),
                        ));
                    }
                    // a count beyond 127 bits (a view broadcast along huge axes) is no dimension this host holds: refused as too large for
                    // the scalar, which is what Python refuses it as, after the constraints of the dimension it binds
                    let count = i128::try_from(count).map_err(|_| {
                        Error::precondition(
                            PreconditionKind::Shape,
                            Some(&scalar.name),
                            format!(
                                "'{stem}': runtime argument '{}' is the {count} elements of the tensor '{}', which do not fit the integer the \
                                 binary takes",
                                scalar.name, self.tensors[t].name
                            ),
                        )
                    })?;
                    if value.is_none()
                        && let Some(var) = scalar.dim.as_var()
                    {
                        dims[var] = Some(count);
                        bound.push(var);
                    }
                    count
                }
                None => value.ok_or_else(|| {
                    Error::Launch(format!(
                        "cannot determine the runtime argument '{}' ({}) from the shapes of the tensors",
                        scalar.name, scalar.dim.text
                    ))
                })?,
            };
        }
        // the constraints of the dimensions a count bound (Python checks every dimension again)
        self.check_dims(&dims, &bound[counted..])?;
        // each fits the integer the binary takes it as (Python's `check_scalar_ranges`)
        for (s, scalar) in self.scalars.iter().enumerate() {
            let value = scalar_values[s];
            if !(0..=i128::from(scalar.ty.max())).contains(&value) {
                return Err(Error::precondition(
                    PreconditionKind::Shape,
                    Some(&scalar.name),
                    format!(
                        "runtime argument '{}' is {value}, which does not fit the {}-bit signed integer the binary takes",
                        scalar.name,
                        if scalar.ty == ScalarType::I64 { 64 } else { 32 }
                    ),
                ));
            }
        }

        // each value given per call fits the integer the binary takes it as (Python's `given_values`): any sign
        for (i, (name, ty)) in self.given.iter().enumerate() {
            let value = given_values[i];
            if *ty == ScalarType::I32 && i32::try_from(value).is_err() {
                return Err(Error::precondition(
                    PreconditionKind::Shape,
                    Some(name),
                    format!(
                        "argument '{name}' of '{stem}' is {value}, which does not fit the i32 the binary takes ({} to {})",
                        i32::MIN,
                        i32::MAX
                    ),
                ));
            }
        }

        // -- a tensor that is written must not overlap another argument (Python's `check_no_alias`)
        for (w, tw) in self.tensors.iter().enumerate().filter(|(_, t)| t.access != Access::In) {
            let (a_start, a_end) = extent(&tensors[w]);
            for (o, to) in self.tensors.iter().enumerate().filter(|(o, _)| *o != w) {
                let (b_start, b_end) = extent(&tensors[o]);
                if a_start < b_end && b_start < a_end {
                    if tw.alias == Some(o) {
                        // an Out that may be written over this input (`alias`): exactly that tensor, or refused
                        if same_tensor(&tensors[w], &tensors[o]) {
                            continue;
                        }
                        return Err(Error::precondition(
                            PreconditionKind::Alias,
                            Some(&tw.name),
                            format!(
                                "argument '{}' may be written over '{}' (MayAlias), but only as exactly that tensor (one address, shape and \
                                 strides): it overlaps it otherwise",
                                tw.name, to.name
                            ),
                        ));
                    }
                    return Err(Error::precondition(
                        PreconditionKind::Alias,
                        Some(&tw.name),
                        format!(
                            "argument '{}' is written in place but overlaps argument '{}': an InOut or Out tensor cannot alias another argument",
                            tw.name, to.name
                        ),
                    ));
                }
            }
        }

        // -- the preconditions of the binary, in the order the manifest lists them (Python's `check_spec_preconditions`): the alignment each
        // pointer promises (16 bytes; a strided pointer may promise only its element's), `max_elements` and `max_extent` (a kernel that
        // indexes with i32: the number of elements of a dense tensor, and the extent of a view, must not pass 2**31 - 1), the divisor of a
        // runtime scalar
        for check in &self.checks {
            match *check {
                Check::Alignment(i) => {
                    let (t, a) = (&self.tensors[i], &tensors[i]);
                    if !a.ptr.is_multiple_of(t.alignment) {
                        return Err(Error::precondition(
                            PreconditionKind::PointerAlignment,
                            Some(&t.name),
                            format!(
                                "argument '{}' is at address {:#x}, which is not aligned to {} bytes; every pointer of '{stem}' is compiled as {}-byte \
                                 aligned (allocate it aligned, or copy it)",
                                t.name, a.ptr, t.alignment, t.alignment
                            ),
                        ));
                    }
                }
                Check::MaxElements(i, limit) => {
                    let (t, a) = (&self.tensors[i], &tensors[i]);
                    let count = reach(a.shape, None);
                    if count > limit as u128 {
                        return Err(Error::precondition(
                            PreconditionKind::MaxElements,
                            Some(&t.name),
                            format!(
                                "argument '{}' has {count} elements (shape {:?}), more than the {limit} that the i32 offsets of '{stem}' reach",
                                t.name, a.shape
                            ),
                        ));
                    }
                }
                Check::MaxExtent(i, limit) => {
                    let (t, a) = (&self.tensors[i], &tensors[i]);
                    let spans = reach_canonical(a.shape, &canonical[i]);
                    if spans > limit as u128 {
                        return Err(Error::precondition(
                            PreconditionKind::MaxExtent,
                            Some(&t.name),
                            format!(
                                "argument '{}' (shape {:?}, strides {:?}) spans {spans} elements, more than the {limit} that the i32 offsets of \
                                 '{stem}' reach",
                                t.name, a.shape, canonical[i]
                            ),
                        ));
                    }
                }
                Check::Divisible(s, by) => {
                    let (scalar, value) = (&self.scalars[s], scalar_values[s]);
                    if value.checked_rem(i128::from(by)).is_some_and(|r| r != 0) {
                        return Err(Error::precondition(
                            PreconditionKind::Divisible,
                            Some(&scalar.name),
                            format!(
                                "runtime argument '{}' ({}) is {value}, which is not a multiple of {by}; '{stem}' is compiled with that assumption",
                                scalar.name, scalar.dim.text
                            ),
                        ));
                    }
                }
            }
        }

        // -- what `cuTensorMapEncodeTiled` requires of the tensor of each TMA descriptor, in the order of the ABI (Python's `check_tensor_maps`):
        // dimensions that fit the i32 the kernel gets them as, the innermost stride 1, the other strides multiples of 16 bytes below 2**40 bytes
        // (the canonical strides; its address is its pointer_alignment, checked above)
        let mut map_strides: Vec<Vec<u128>> = vec![Vec::new(); self.tensors.len()];
        for (t, plan) in self.tensors.iter().enumerate() {
            if plan.map.is_none() {
                continue;
            }
            let a = &tensors[t];
            let strides = canonical_strides(a.shape, a.strides);
            check_tensor_map(stem, &plan.name, plan.dtype.size(), a.shape, &strides)?;
            map_strides[t] = strides;
        }

        // -- the grid, from its expression (a persistent grid's tiles cut to the programs the SMs run; a cluster's blocks), against what the
        // device takes
        let mut programs = [1i128; 3];
        for (axis, entry) in self.grid.iter().enumerate() {
            let value =
                entry.eval(&dims).map_err(|e| Error::Launch(format!("'{stem}': the grid {}: {e}", self.grid_text)))?;
            let Some(value) = value else {
                let free: Vec<&str> =
                    entry.vars.iter().filter(|v| dims[**v].is_none()).map(|v| self.syms.name(*v)).collect();
                return Err(Error::Launch(format!(
                    "the grid {} of '{stem}' mentions {free:?}, which no tensor determines",
                    self.grid_text
                )));
            };
            if value < 1 {
                return Err(Error::Launch(format!(
                    "the grid {} of '{stem}' is {value} here; every grid size must be at least 1",
                    self.grid_text
                )));
            }
            programs[axis] = value;
        }
        if let (Some(p), Some(sms)) = (self.persistent, sm_count) {
            let tiles = u64::try_from(programs[0]).unwrap_or(u64::MAX);
            programs[0] = i128::from(p.programs(tiles, sms, u64::from(self.num_ctas)));
        }
        let cluster = self.cluster.unwrap_or([1, 1, 1]);
        let mut grid = [1u32; 3];
        for (axis, programs) in programs.iter().enumerate() {
            let along = ["x", "y", "z"][axis];
            let value = programs.saturating_mul(i128::from(cluster[axis]));
            if let Some(limit) = device.map(|d| d.max_grid[axis]).filter(|limit| value > i128::from(*limit)) {
                return Err(Error::Unsupported(format!(
                    "'{stem}': the grid {} is {value} blocks along {along} here, but this device takes 1 to {limit} there",
                    self.grid_text
                )));
            }
            grid[axis] = u32::try_from(value).map_err(|_| {
                Error::Unsupported(format!(
                    "the grid {} of '{stem}' has {value} blocks along {along}, more than 2^32 - 1",
                    self.grid_text
                ))
            })?;
        }

        // -- the kernelParams values (every scalar is in the range of its type: checked above), and the bytes of a CuTe DSL kernel's descriptors
        // and constant bytes, passed by value (a descriptor's address is checked last, where Python's launch checks it)
        let mut values = [0u64; MAX_PARAMS];
        let mut tensor_of = [NOT_A_TENSOR; MAX_PARAMS];
        let mut maps = Vec::new();
        let mut passed = Vec::new();
        for (k, slot) in self.slots.iter().enumerate() {
            values[k] = match slot {
                Slot::Tensor(t) => {
                    tensor_of[k] = *t as u8; // at most MAX_PARAMS tensors: `Plan::new` refuses more parameters
                    tensors[*t].ptr
                }
                Slot::Descriptor(t) => {
                    tensor_of[k] = *t as u8;
                    let (plan, a) = (&self.tensors[*t], &tensors[*t]);
                    let template =
                        plan.template.as_ref().expect("a descriptor slot is of a CuTe DSL kernel's descriptor");
                    let Some(bytes) = descriptor(template, a.ptr) else {
                        return Err(Error::Launch(format!(
                            "'{stem}': argument '{}' is at address {:#x}, but the TMA descriptor of a CuTe DSL kernel holds the address of its tensor \
                             16-byte aligned and below 2**57 (the DSL keeps 53 bits of address / 16): give it memory there",
                            plan.name, a.ptr
                        )));
                    };
                    passed.push(PassedSlot { slot: k, bytes: PassedBytes::Descriptor(bytes) });
                    0 // the descriptor's 128 bytes are passed from `passed`
                }
                Slot::Bytes(i) => {
                    passed.push(PassedSlot { slot: k, bytes: PassedBytes::Constant(Arc::clone(&self.constants[*i])) });
                    0 // the bytes are passed from `passed`
                }
                Slot::Map(t) => {
                    tensor_of[k] = *t as u8;
                    let (plan, a) = (&self.tensors[*t], &tensors[*t]);
                    let map = plan.map.as_ref().expect("a map slot is of a descriptor");
                    maps.push(MapSlot {
                        slot: k,
                        encode: encode_of(map, plan.dtype.size(), a, &map_strides[*t]),
                        bytes: None,
                    });
                    0 // the map's 128 bytes are passed from the encoded map
                }
                // (each fits: checked above, a dimension in i32, a stride below 2**40 bytes)
                Slot::MapShape(t, axis) => u64::from(tensors[*t].shape[*axis] as u32),
                Slot::MapStride(t, axis) => map_strides[*t][*axis] as u64,
                Slot::Scalar(s) => {
                    let v = scalar_values[*s];
                    match self.scalars[*s].ty {
                        ScalarType::I32 => u64::from(v as u32),
                        ScalarType::I64 => v as u64,
                    }
                }
                Slot::Fixed(bits) => *bits,
                Slot::Given(i) => match self.given[*i].1 {
                    ScalarType::I32 => u64::from(given_values[*i] as u32),
                    ScalarType::I64 => given_values[*i] as u64,
                },
            };
        }
        // -- the global scratch: a writable u8 tensor of the bytes the launch needs (size per CTA, times the CTAs of its programs), 256-byte
        // aligned, on the kernel's device, overlapping no other tensor; its address is the first scratch slot (the profile scratch stays 0)
        let mut scratch_bytes = 0u64;
        if let (Some(per_cta), Some(s)) = (self.scratch, scratch) {
            let ctas = programs.iter().fold(i128::from(self.num_ctas), |acc, p| acc.saturating_mul(*p));
            let need = i128::from(per_cta).saturating_mul(ctas);
            scratch_bytes = u64::try_from(need).map_err(|_| {
                Error::Unsupported(format!(
                    "'{stem}': a global scratch of {need} bytes is more than this host can give"
                ))
            })?;
            self.check_scratch(s, scratch_bytes, tensors, device)?;
            let slot = self.slots.len();
            values[slot] = s.ptr;
            tensor_of[slot] = self.tensors.len() as u8; // the scratch tensor, after the others: a launch orders its memory as a written tensor's
        }
        // (the scratch slots of a Triton kernel stay 0, null pointers, but the global scratch of a kernel that has one)
        Ok(Prepared {
            grid,
            cluster: self.cluster,
            scratch_bytes,
            values,
            count: self.slots.len() + self.scratch_slots,
            tensor_of,
            maps,
            passed,
        })
    }

    /// The global scratch tensor `s` of a launch that needs `need` bytes of it: `u8`, writable, dense, at least `need` bytes, 256-byte aligned (the
    /// alignment every scratch is given at, which the manifest's `global_scratch_align` divides), on the kernel's device and context, overlapping
    /// none of `others`.
    fn check_scratch(
        &self,
        s: &TensorArg<'_>,
        need: u64,
        others: &[TensorArg<'_>],
        device: Option<&DeviceInfo>,
    ) -> Result<()> {
        let stem = self.stem.as_str();
        let name = GLOBAL_SCRATCH;
        if s.dtype != DType::U8 {
            return Err(Error::precondition(
                PreconditionKind::Dtype,
                Some(name),
                format!("the global scratch of '{stem}' is a u8 tensor, got {}", s.dtype.name()),
            ));
        }
        if s.access == Access::In {
            return Err(Error::precondition(
                PreconditionKind::Access,
                Some(name),
                format!(
                    "the global scratch of '{stem}' is written by the kernel: give it a tensor that may be written"
                ),
            ));
        }
        if let Some(strides) = s.strides
            && !is_contiguous(s.shape, strides)
        {
            return Err(Error::precondition(
                PreconditionKind::Contiguous,
                Some(name),
                format!("the global scratch of '{stem}' is one run of bytes: give it a contiguous tensor"),
            ));
        }
        let bytes = reach(s.shape, None).saturating_mul(s.dtype.size() as u128);
        if bytes < u128::from(need) {
            return Err(Error::precondition(
                PreconditionKind::Shape,
                Some(name),
                format!(
                    "'{stem}' needs {need} bytes of global scratch for this launch, but its scratch tensor has {bytes}"
                ),
            ));
        }
        if !s.ptr.is_multiple_of(SCRATCH_ALIGNMENT) {
            return Err(Error::precondition(
                PreconditionKind::PointerAlignment,
                Some(name),
                format!(
                    "the global scratch of '{stem}' is at address {:#x}, which is not aligned to {SCRATCH_ALIGNMENT} bytes",
                    s.ptr
                ),
            ));
        }
        if let Some(d) = device.filter(|d| d.ordinal != s.device) {
            return Err(Error::precondition(
                PreconditionKind::Device,
                Some(name),
                format!(
                    "the global scratch of '{stem}' is on device {}, but the kernel is loaded on device {}",
                    s.device, d.ordinal
                ),
            ));
        }
        if let (Some(kernel), Some(tensor)) = (device.and_then(|d| d.context), s.context)
            && kernel != tensor
        {
            return Err(Error::precondition(
                PreconditionKind::Device,
                Some(name),
                format!(
                    "the global scratch of '{stem}' is memory of another CUDA context than the one the kernel is loaded in"
                ),
            ));
        }
        let (start, end) = (u128::from(s.ptr), u128::from(s.ptr).saturating_add(u128::from(need.max(1))));
        for (t, o) in self.tensors.iter().zip(others) {
            let (b_start, b_end) = extent(o);
            if start < b_end && b_start < end {
                return Err(Error::precondition(
                    PreconditionKind::Alias,
                    Some(name),
                    format!(
                        "the global scratch of '{stem}' overlaps argument '{}': the kernel writes its scratch",
                        t.name
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Bind the dimensions from `equations`, as Python's `solve_dims` does: in passes over the equations in order, each binding the one
    /// unbound dimension it is affine in or checking a bound one, until all are done (`bound` gets each dimension as it is bound) or what
    /// is left is ambiguous.
    fn solve(&self, dims: &mut [Option<i128>], bound: &mut Vec<usize>, equations: &[Equation<'_>]) -> Result<()> {
        let mut done = vec![false; equations.len()];
        loop {
            let mut progress = false;
            for (k, eq) in equations.iter().enumerate() {
                if done[k] {
                    continue;
                }
                match eq.pattern.solve_from(eq.actual, dims, eq.minimum) {
                    Ok(Solved::Checked) => {
                        done[k] = true;
                        progress = true;
                    }
                    Ok(Solved::Bound(var, value)) => {
                        dims[var] = Some(value);
                        bound.push(var);
                        done[k] = true;
                        progress = true;
                    }
                    Ok(Solved::Stuck) => {}
                    Err(e) => return Err(self.unsolvable(eq.origin, eq.pattern, eq.actual, e)),
                }
            }
            if done.iter().all(|d| *d) {
                return Ok(());
            }
            if progress {
                continue;
            }
            // nothing more can be solved, and what is left is ambiguous: Python's `solve_dims` raises a type error (no precondition of the
            // binary is broken: the call does not determine the dimensions). A bucketed dimension is not taken from the binary here either
            // (a dimension that only occurs inside a `cdiv` is not bound by any equation, whatever its bucket)
            let stuck =
                equations.iter().zip(&done).find(|(_, d)| !**d).map(|(eq, _)| eq).expect("not all equations are done");
            let (t, what) = match stuck.origin {
                Origin::Tensor { t, axis } => (t, format!("dimension {axis} (size {})", stuck.actual)),
                Origin::Stride { t, axis } => (t, format!("the stride of axis {axis} (value {})", stuck.actual)),
            };
            return Err(Error::Launch(format!(
                "'{}': argument '{}' ({}): {what}: cannot infer the dimensions in {}; the signature is ambiguous at this call",
                self.stem,
                self.tensors[t].name,
                self.describe_tensor(t),
                stuck.pattern.text
            )));
        }
    }
}

#[derive(Clone, Copy)]
enum Origin {
    Tensor { t: usize, axis: usize },
    Stride { t: usize, axis: usize },
}

struct Equation<'p> {
    pattern: &'p Pattern,
    actual: i128,
    origin: Origin,
    /// The least value the dimension it binds may have: 1 for a size, 0 for a stride.
    minimum: i128,
}

/// The strides of a view with those of its size-1 axes made canonical (Python's `canonical_strides`): the stride of a size-1 axis is never
/// read by a kernel, so it is replaced by the stride a dense tensor would have there (`1` for the last axis, else the size of the next axis
/// times its stride), so that two views that differ only there bind the same values, and an `unsqueeze` or an `expand` does not pass a stride
/// that no kernel reads. `None` strides are the dense row-major ones.
fn canonical_strides(shape: &[usize], strides: Option<&[usize]>) -> Vec<u128> {
    let n = shape.len();
    let mut out: Vec<u128> = match strides {
        Some(s) => s.iter().map(|v| *v as u128).collect(),
        None => {
            let mut dense = vec![0u128; n];
            let mut acc = 1u128;
            for i in (0..n).rev() {
                dense[i] = acc;
                acc = acc.saturating_mul(shape[i] as u128);
            }
            dense
        }
    };
    for i in (0..n).rev() {
        if shape[i] == 1 {
            out[i] = if i + 1 == n { 1 } else { (shape[i + 1] as u128).saturating_mul(out[i + 1]) };
        }
    }
    out
}

/// The largest dimension of a tensor of a TMA descriptor: its shape is passed to the kernel as `i32` (the driver takes up to `2**32`).
const MAP_MAX_DIM: usize = i32::MAX as usize;
/// A stride of a tensor of a TMA descriptor is less than this many bytes.
const MAP_MAX_STRIDE_BYTES: u128 = 1 << 40;

/// What `cuTensorMapEncodeTiled` requires of the tensor of a TMA descriptor (Python's `_tensormap.check_tensor_map`, in its order), with its
/// canonical element `strides`: every dimension at most `2**31 - 1`, the innermost stride 1, every other stride a multiple of 16 bytes below
/// `2**40` bytes. A refusal of kind `tensor_map`.
fn check_tensor_map(stem: &str, name: &str, item: usize, shape: &[usize], strides: &[u128]) -> Result<()> {
    let refuse = |message: String| Err(Error::precondition(PreconditionKind::TensorMap, Some(name), message));
    let rank = shape.len();
    for (axis, n) in shape.iter().enumerate() {
        if *n > MAP_MAX_DIM {
            return refuse(format!(
                "'{stem}': argument '{name}' has {n} elements along axis {axis}: the dimensions of a tensor descriptor are passed to the kernel \
                 as i32, so at most {MAP_MAX_DIM}"
            ));
        }
    }
    if strides[rank - 1] != 1 {
        return refuse(format!(
            "'{stem}': argument '{name}' has stride {} along its last axis: a tensor descriptor reads the last axis contiguously (stride 1)",
            strides[rank - 1]
        ));
    }
    for axis in 0..rank - 1 {
        let bytes = strides[axis].saturating_mul(item as u128);
        if !bytes.is_multiple_of(16) || bytes >= MAP_MAX_STRIDE_BYTES {
            return refuse(format!(
                "'{stem}': argument '{name}' (shape {shape:?}, strides {strides:?}) has a stride of {bytes} bytes along axis {axis}: the strides \
                 of a tensor descriptor are multiples of 16 bytes and less than 2**40 bytes (pad the rows of the tensor to a multiple of 16 bytes)"
            ));
        }
    }
    Ok(())
}

/// The arguments of `cuTensorMapEncodeTiled` for the tensor `a` (its checked canonical element `strides`) of a descriptor with `map` (Python's
/// `_tensormap.encode_arguments`): the dimensions reversed (innermost first), the strides of the outer axes in bytes, innermost first, then the
/// bytes of the whole tensor, as Triton's `fillTMADescriptor` passes them.
fn encode_of(map: &MapPlan, item: usize, a: &TensorArg<'_>, strides: &[u128]) -> TensorMapEncode {
    let rank = a.shape.len();
    let item = item as u64;
    let global_dim: Vec<u64> = a.shape.iter().rev().map(|n| *n as u64).collect();
    let mut global_strides: Vec<u64> =
        (0..rank.saturating_sub(1)).rev().map(|axis| strides[axis] as u64 * item).collect();
    let outermost = global_strides.last().copied().unwrap_or(item);
    global_strides.push(global_dim[rank - 1] * outermost);
    TensorMapEncode {
        data_type: map.data_type,
        rank: rank as u32,
        address: a.ptr,
        global_dim,
        global_strides,
        box_dim: map.box_dim.clone(),
        element_strides: vec![1; rank],
        interleave: 0,
        swizzle: map.swizzle,
        l2_promotion: map.l2_promotion,
        oob_fill: map.oob_fill,
    }
}

/// One more than the largest element offset a tensor of this shape and these (canonical) strides reaches.
fn reach_canonical(shape: &[usize], strides: &[u128]) -> u128 {
    if strides.is_empty() {
        // a dense tensor: its number of elements
        return shape.iter().fold(1u128, |acc, d| acc.saturating_mul(*d as u128));
    }
    shape.iter().zip(strides).fold(1u128, |acc, (n, s)| acc.saturating_add((*n as u128 - 1).saturating_mul(*s)))
}

/// Whether a tensor is contiguous the way torch's `is_contiguous` says: row-major with no gaps, the strides of size-1 axes not looked at.
fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
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

/// Whether two arguments are exactly one tensor: one address, one dtype, one shape and the same canonical strides (Python's `same_tensor`).
fn same_tensor(a: &TensorArg<'_>, b: &TensorArg<'_>) -> bool {
    a.ptr == b.ptr
        && a.dtype == b.dtype
        && a.shape == b.shape
        && canonical_strides(a.shape, a.strides) == canonical_strides(b.shape, b.strides)
}

/// The bytes a tensor covers, `[start, end)`: the elements it reaches (a dense tensor's, a view's extent) from its first one.
fn extent(a: &TensorArg<'_>) -> (u128, u128) {
    let bytes = reach(a.shape, a.strides).saturating_mul(a.dtype.size() as u128);
    (u128::from(a.ptr), u128::from(a.ptr).saturating_add(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{Manifest, Pin};
    use crate::region::DevicePtr;

    fn gelu() -> Plan {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("sm_103");
        let manifest = Manifest::load(dir, Pin::TrustDirectory).unwrap();
        manifest.plan("gelu_erf", &[], crate::kernel::Arch::sm(103)).unwrap()
    }

    fn arg(shape: &[usize], addr: u64, context: Option<usize>) -> TensorArg<'_> {
        // SAFETY: the address is never dereferenced: `prepare` only looks at the number.
        #[allow(unsafe_code)]
        let region = unsafe { DevicePtr::from_raw(addr, 8192, 0) };
        let mut arg = TensorArg::new(region, shape, DType::Bf16, Access::InOut).unwrap();
        arg.context = context;
        arg
    }

    const SHAPE: [usize; 1] = [4096];

    #[test]
    fn memory_of_another_context_on_the_same_device_is_refused_and_the_kernels_own_is_not() {
        let plan = gelu();
        let device = |context| DeviceInfo { ordinal: 0, context, max_grid: [1 << 20, 65535, 65535] };
        let call = |kernel: Option<usize>, x: Option<usize>, y: Option<usize>| {
            plan.prepare(&[arg(&SHAPE, 0x7000_0000_0000, x), arg(&SHAPE, 0x7000_1000_0000, y)], Some(&device(kernel)))
        };
        assert!(call(Some(0xA), Some(0xA), Some(0xA)).is_ok());
        // a tensor of another context (the same device): refused, and the message says which argument
        let e = call(Some(0xA), Some(0xB), Some(0xA)).unwrap_err();
        assert_eq!(e.precondition_kind(), Some(PreconditionKind::Device), "{e}");
        assert!(e.to_string().contains("x_ptr") && e.to_string().contains("another CUDA context"), "{e}");
        let e = call(Some(0xA), Some(0xA), Some(0xB)).unwrap_err();
        assert!(e.to_string().contains("y_ptr"), "{e}");
        // memory with no context of its own (a region the host vouched for), and a kernel with none (a made-up device), are not checked
        assert!(call(Some(0xA), None, None).is_ok());
        assert!(call(None, Some(0xB), Some(0xC)).is_ok());
        assert!(
            plan.prepare(&[arg(&SHAPE, 0x7000_0000_0000, Some(1)), arg(&SHAPE, 0x7000_1000_0000, Some(2))], None)
                .is_ok()
        );
    }

    #[test]
    fn strides_are_made_canonical_where_the_size_is_one() {
        // a dense tensor
        assert_eq!(canonical_strides(&[4, 8, 64], None), [512, 64, 1]);
        // a view as torch reports it
        assert_eq!(canonical_strides(&[40, 8, 64], Some(&[1536, 64, 1])), [1536, 64, 1]);
        // a size-1 axis gets the stride of a dense tensor there, whatever the view says (an unsqueeze, an expand)
        assert_eq!(canonical_strides(&[1, 8, 8], Some(&[999, 8, 1])), [64, 8, 1]);
        assert_eq!(canonical_strides(&[4, 8, 1], Some(&[8, 1, 12345])), [8, 1, 1]);
        assert_eq!(canonical_strides(&[1], Some(&[7])), [1]);
        // a stride of 0 (an expanded view) of an axis that is not size 1 stays 0
        assert_eq!(canonical_strides(&[40, 8, 64], Some(&[0, 64, 1])), [0, 64, 1]);
    }

    #[test]
    fn contiguity_is_torchs() {
        assert!(is_contiguous(&[4, 8], &[8, 1]));
        assert!(!is_contiguous(&[4, 8], &[16, 1]));
        assert!(!is_contiguous(&[4, 8], &[1, 4]));
        assert!(is_contiguous(&[1, 8], &[777, 1]), "the stride of a size-1 axis is not looked at");
        assert!(is_contiguous(&[4, 1, 8], &[8, 5, 1]));
        assert!(!is_contiguous(&[40, 8, 64], &[1536, 64, 1]));
        assert!(is_contiguous(&[], &[]));
    }

    #[test]
    fn the_extent_of_a_view_is_one_more_than_the_largest_offset() {
        assert_eq!(reach_canonical(&[40, 8, 64], &[1536, 64, 1]), 39 * 1536 + 7 * 64 + 63 + 1);
        assert_eq!(reach_canonical(&[40, 8, 64], &[0, 64, 1]), 7 * 64 + 63 + 1, "an expanded axis adds nothing");
        assert_eq!(reach_canonical(&[4, 8], &[]), 32, "a dense tensor: its elements");
        // unbounded: the product exceeds 64 bits and is still compared as it is
        assert!(reach_canonical(&[1 << 40, 1 << 40], &[1 << 40, 1]) > u128::from(u64::MAX));
    }

    #[test]
    fn a_cute_descriptor_is_its_template_with_the_address_in_its_first_eight_bytes() {
        let mut template = [0u8; DESCRIPTOR_BYTES];
        for (i, b) in template.iter_mut().enumerate().take(64).skip(8) {
            *b = i as u8;
        }
        let d = descriptor(&template, 0x7F00_0010_0000).unwrap();
        assert_eq!(&d.0[..8], &0x7F00_0010_0000u64.to_le_bytes());
        assert_eq!(&d.0[8..], &template[8..]);
        // 64-byte aligned in the host's memory, as the kernel parameter is
        assert_eq!((std::mem::align_of::<Descriptor>(), std::mem::size_of::<Descriptor>()), (64, 128));
        assert_eq!(PassedBytes::Descriptor(d).as_slice(), &d.0[..]);
        // the addresses the DSL's descriptor holds: not null, 16-byte aligned, below 2**57 (Python's `_cute.descriptor`)
        for refused in [0, 8, 0x7F00_0010_0008, CUTE_ADDRESS_LIMIT, CUTE_ADDRESS_LIMIT + 16, !15u64] {
            assert!(descriptor(&template, refused).is_none(), "{refused:#x}");
        }
        assert!(descriptor(&template, 16).is_some());
        assert_eq!(
            &descriptor(&template, CUTE_ADDRESS_LIMIT - 16).unwrap().0[..8],
            &(CUTE_ADDRESS_LIMIT - 16).to_le_bytes()
        );
    }
}
