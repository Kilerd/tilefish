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
//! value the manifest holds. The checks, in order: the number of tensors; for each tensor its device, dtype and rank (a device scalar
//! is one element), that no dimension is empty, and that a tensor given for a parameter that is not strided is contiguous; the
//! binding of the dimensions from the sizes of the tensors, and from the canonical strides of the views, which must agree across
//! arguments and with the static sizes (a port of `solve_dims`); the bucket (a bucketed dimension is the bucket exactly: there is no
//! padding) and `divisible_by` of every dimension; the runtime scalars, derived from their dimension expression, from the element count
//! of the tensor they count or from the stride they carry, and their range (`i32` or `i64`); aliasing of the tensors that are written;
//! the alignment of every pointer the parameter promises; `max_elements` and `max_extent`. Last, the grid is evaluated from its
//! expression and checked against the device's limits. A call has nothing but its tensors: there is no way to tell a launch what a
//! dimension is.

use crate::error::{Error, PreconditionKind, Result};
use crate::expr::{Expr, MAX_DIMS, Pattern, Solved, Symbols, Unsolvable, parse_grid};
use crate::library::LibraryEntry;
use crate::manifest::{
    Access, DimInfo, Entry, LaunchAttributes, Param, ParamKind, Precondition, ScalarType, ShapeDim, StrideEntry,
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
    max_elements: Option<i64>,
    max_extent: Option<i64>,
}

#[derive(Debug, Clone)]
struct ScalarPlan {
    name: String,
    ty: ScalarType,
    dim: Pattern,
    count_of: Option<usize>,
    divisible: Option<i64>,
}

/// What the manifest says about the named dimension of one symbol.
#[derive(Debug, Clone)]
struct DimPlan {
    divisible_by: Option<i64>,
    bucket: Option<(i64, Vec<u64>)>,
}

#[derive(Debug, Clone, Copy)]
enum Slot {
    Tensor(usize),
    Scalar(usize),
    /// A `fixed` argument: its bits in the low bytes of the slot.
    Fixed(u64),
}

/// One manifest entry, ready to check a call against.
#[derive(Debug, Clone)]
pub struct Plan {
    stem: String,
    syms: Symbols,
    dims: Vec<DimPlan>,
    tensors: Vec<TensorPlan>,
    scalars: Vec<ScalarPlan>,
    slots: Vec<Slot>,
    scratch_slots: usize,
    grid: Vec<Expr>,
    grid_text: String,
    block: [u32; 3],
    shared: u32,
}

/// What a checked call launches with.
#[derive(Debug, Clone)]
pub struct Prepared {
    /// The grid, three sizes.
    pub grid: [u32; 3],
    pub(crate) values: [u64; MAX_PARAMS],
    pub(crate) count: usize,
    /// For each slot, the index of the tensor it holds (`NOT_A_TENSOR` for a scalar, a fixed argument or a scratch slot).
    pub(crate) tensor_of: [u8; MAX_PARAMS],
}

/// What `Prepared::tensor_of` holds for a slot that is not a tensor.
pub(crate) const NOT_A_TENSOR: u8 = u8::MAX;

impl Prepared {
    /// The `kernelParams` values, one 64-bit slot each: an address, an `i32` or an `f32` in the low four bytes, an `i64`, then the scratch
    /// slots of a Triton kernel (null pointers; a prebuilt binary has none).
    pub fn params(&self) -> &[u64] {
        &self.values[..self.count]
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
        })
    }

    /// The plan of a library entry (a pinned GEMM): what its operands are checked against. It launches nothing by itself (its grid is
    /// `(1,)` and it has no scalars): the checks are those of a kernel, on the pointers `a`, `b`, `bias` and `out`.
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
        if source.params.len() + source.scratch_slots > MAX_PARAMS {
            return Err(Error::Unsupported(format!(
                "{who}: a kernel with {} parameters is not supported",
                source.params.len()
            )));
        }
        let bad = |e: String| Error::manifest(format!("{who}: {e}"));
        // the named dimensions come first, so that a symbol's number is its place in the table of the manifest
        let mut syms = Symbols::default();
        let mut dims = Vec::with_capacity(source.dims.len());
        for d in source.dims {
            syms.intern(&d.name).map_err(bad)?;
            let bucket = match (d.bucket, &d.buckets) {
                (Some(b), Some(all)) => Some((to_i64(b, "a bucket")?, all.clone())),
                _ => None,
            };
            dims.push(DimPlan { divisible_by: d.divisible_by.map(|v| to_i64(v, "a divisor")).transpose()?, bucket });
        }
        let mut tensors: Vec<TensorPlan> = Vec::new();
        let mut scalars: Vec<ScalarPlan> = Vec::new();
        let mut slots = Vec::new();
        for p in source.params {
            match &p.kind {
                ParamKind::Pointer { dtype_name, dtype, access, shape, view } => {
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
                    slots.push(Slot::Tensor(tensors.len()));
                    tensors.push(TensorPlan {
                        name: p.name.clone(),
                        dtype,
                        access: *access,
                        device_scalar: shape.is_empty(),
                        shape: patterns,
                        strides,
                        alignment: 16,
                        max_elements: None,
                        max_extent: None,
                    });
                }
                ParamKind::Scalar { ty, dim, .. } => {
                    slots.push(Slot::Scalar(scalars.len()));
                    let dim = Pattern::parse(dim, &format!("parameter {} dim", p.name), &mut syms).map_err(bad)?;
                    scalars.push(ScalarPlan { name: p.name.clone(), ty: *ty, dim, count_of: None, divisible: None });
                }
                ParamKind::Fixed { value } => slots.push(Slot::Fixed(value.bits())),
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
        for pre in source.preconditions {
            match pre {
                Precondition::MaxElements { arg, value, .. } => {
                    if let Some(t) = tensors.iter_mut().find(|t| t.name == *arg) {
                        t.max_elements = Some(to_i64(*value, "max_elements")?);
                    }
                }
                Precondition::MaxExtent { arg, value, .. } => {
                    if let Some(t) = tensors.iter_mut().find(|t| t.name == *arg) {
                        t.max_extent = Some(to_i64(*value, "max_extent")?);
                    }
                }
                Precondition::PointerAlignment { arg, bytes, .. } => {
                    if let Some(t) = tensors.iter_mut().find(|t| t.name == *arg) {
                        t.alignment = *bytes;
                    }
                }
                Precondition::Divisible { arg, by, .. } => {
                    if let Some(s) = scalars.iter_mut().find(|s| s.name == *arg) {
                        s.divisible = Some(to_i64(*by, "a divisor")?);
                    }
                }
                // checked when the manifest is read: a static size is what the shape says, and a bucket is what `dims` says
                Precondition::StaticSize { .. } | Precondition::Bucket { .. } => {}
            }
        }
        let grid = parse_grid(source.grid, &mut syms).map_err(bad)?;
        if syms.len() != dims.len() {
            return Err(Error::manifest(format!("{who}: the entry's dims do not name every dimension it uses")));
        }
        Ok(Self {
            stem: source.stem.to_owned(),
            syms,
            dims,
            tensors,
            scalars,
            slots,
            scratch_slots: source.scratch_slots,
            grid,
            grid_text: source.grid.to_owned(),
            block: source.block,
            shared: source.shared,
        })
    }

    /// The stem of the entry.
    pub fn stem(&self) -> &str {
        &self.stem
    }

    /// The names of the tensor parameters, in the order a launch gives its tensors.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|t| t.name.as_str())
    }

    pub(crate) fn block(&self) -> [u32; 3] {
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

    fn unsolvable(
        &self,
        origin: Origin,
        pattern: &Pattern,
        actual: i64,
        e: Unsolvable,
        counted: Option<(usize, i64)>,
    ) -> Error {
        let name_of = |i: usize| self.syms.name(i).to_owned();
        match (origin, e) {
            (_, Unsolvable::Eval(message)) => Error::Launch(format!("'{}': {message}", self.stem)),
            (Origin::Tensor { t, axis } | Origin::Stride { t, axis }, e) => {
                let stride = matches!(origin, Origin::Stride { .. });
                let name = &self.tensors[t].name;
                let which = if stride { format!("the stride of axis {axis}") } else { format!("dimension {axis}") };
                let head =
                    format!("'{}': argument '{name}' ({}): {which} is {actual}", self.stem, self.describe_tensor(t));
                match e {
                    Unsolvable::Mismatch { expected } => {
                        let what = if pattern.is_static {
                            expected.to_string()
                        } else {
                            format!("{} = {expected}", pattern.text)
                        };
                        let note = if stride && pattern.is_static {
                            " (the kernel loads along that axis contiguously: pass a view whose last axis has stride 1)"
                        } else {
                            ""
                        };
                        let kind = match (pattern.is_static, stride) {
                            (true, true) => PreconditionKind::Stride,
                            (true, false) => PreconditionKind::StaticSize,
                            (false, _) => PreconditionKind::Shape,
                        };
                        Error::precondition(
                            kind,
                            Some(name),
                            format!("{head}, but the signature requires {what}{note}"),
                        )
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
                    Unsolvable::Eval(_) => unreachable!("handled above"),
                }
            }
            (Origin::Scalar { s }, other) => {
                let name = &self.scalars[s].name;
                let detail = match (other, counted) {
                    (Unsolvable::Mismatch { expected }, Some((t, count))) => format!(
                        "runtime argument '{name}' is {} = {expected}, but the tensor '{}' has {count} elements",
                        pattern.text, self.tensors[t].name
                    ),
                    (_, Some((t, count))) => format!(
                        "runtime argument '{name}' ({}) cannot be the {count} elements of the tensor '{}'",
                        pattern.text, self.tensors[t].name
                    ),
                    (_, None) => format!(
                        "runtime argument '{name}' is {actual}, which cannot match its dimension {}",
                        pattern.text
                    ),
                };
                Error::precondition(PreconditionKind::Shape, Some(name), format!("'{}': {detail}", self.stem))
            }
        }
    }

    /// Check a call against everything the manifest says, and make the grid and the `kernelParams` values for it.
    ///
    /// `tensors` are the tensors of the call in the order of the manifest's ABI (see [`Plan::tensor_names`]). `device` is the
    /// device the kernel is loaded on: every tensor must be on it, and the grid must fit what it takes (`None` skips both).
    pub fn prepare(&self, tensors: &[TensorArg<'_>], device: Option<&DeviceInfo>) -> Result<Prepared> {
        let stem = self.stem.as_str();
        if tensors.len() != self.tensors.len() {
            return Err(Error::Launch(format!(
                "'{stem}' takes {} tensors ({}), got {}",
                self.tensors.len(),
                self.tensor_names().collect::<Vec<_>>().join(", "),
                tensors.len()
            )));
        }
        // -- each tensor on its own
        let mut counts = [0i64; MAX_PARAMS];
        let mut canonical: Vec<Vec<u128>> = Vec::with_capacity(tensors.len());
        for (i, (t, a)) in self.tensors.iter().zip(tensors).enumerate() {
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
                counts[i] = 1;
                canonical.push(Vec::new());
                continue;
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
            let count =
                a.shape.iter().try_fold(1u128, |acc, d| acc.checked_mul(*d as u128)).filter(|c| *c <= i64::MAX as u128);
            let Some(count) = count else {
                return Err(Error::precondition(
                    PreconditionKind::Shape,
                    Some(&t.name),
                    format!("argument '{}' has more elements than fit 64 bits (shape {:?})", t.name, a.shape),
                ));
            };
            counts[i] = count as i64;
            canonical.push(if t.strides.is_some() { canonical_strides(a.shape, a.strides) } else { Vec::new() });
        }

        // -- the dimensions, bound from the sizes of the tensors, the strides of the views (and the element counts the scalars carry)
        let mut dims: [Option<i64>; MAX_DIMS] = [None; MAX_DIMS];
        let mut equations: Vec<Equation<'_>> = Vec::new();
        for (t, a) in tensors.iter().enumerate() {
            let plan = &self.tensors[t];
            for (axis, (pattern, size)) in plan.shape.iter().zip(a.shape).enumerate() {
                let actual = i64::try_from(*size).map_err(|_| {
                    Error::precondition(PreconditionKind::Shape, Some(&plan.name), "a dimension does not fit 64 bits")
                })?;
                equations.push(Equation {
                    pattern,
                    actual,
                    origin: Origin::Tensor { t, axis },
                    counted: None,
                    minimum: 1,
                });
            }
            if let Some(strides) = &plan.strides {
                for (axis, (stride, value)) in strides.iter().zip(&canonical[t]).enumerate() {
                    let Ok(actual) = i64::try_from(*value) else {
                        return Err(Error::precondition(
                            PreconditionKind::Shape,
                            Some(&plan.name),
                            format!(
                                "argument '{}': the stride of axis {axis} is {value}, which does not fit the 64-bit signed integer a stride is carried in",
                                plan.name
                            ),
                        ));
                    };
                    equations.push(Equation {
                        pattern: &stride.pattern,
                        actual,
                        origin: Origin::Stride { t, axis },
                        counted: None,
                        minimum: 0, // a stride can be 0 (an expanded view)
                    });
                }
            }
        }
        for (s, scalar) in self.scalars.iter().enumerate() {
            if let Some(t) = scalar.count_of {
                equations.push(Equation {
                    pattern: &scalar.dim,
                    actual: counts[t],
                    origin: Origin::Scalar { s },
                    counted: Some((t, counts[t])),
                    minimum: 1,
                });
            }
        }
        self.solve(&mut dims, &equations)?;

        // -- what the manifest says of each dimension: a bucket is exact (there is no padding), and divisible_by
        for (i, info) in self.dims.iter().enumerate() {
            let name = self.syms.name(i);
            if let Some((value, all)) = &info.bucket {
                match dims[i] {
                    Some(found) if found != *value => {
                        return Err(Error::precondition(
                            PreconditionKind::Bucket,
                            Some(name),
                            format!(
                                "dimension {name} is {found}, but '{stem}' is built for exactly {value} (buckets {all:?}); there is no padding: \
                                 pad or slice in the host to a bucket size"
                            ),
                        ));
                    }
                    _ => dims[i] = Some(*value),
                }
            }
            if let (Some(by), Some(v)) = (info.divisible_by, dims[i])
                && v % by != 0
            {
                return Err(Error::precondition(
                    PreconditionKind::Divisible,
                    Some(name),
                    format!("dimension {name} is {v}, but it is declared divisible_by={by}"),
                ));
            }
        }

        // -- the runtime scalars
        let mut scalar_values = [0i64; MAX_PARAMS];
        for (s, scalar) in self.scalars.iter().enumerate() {
            let value = match scalar.count_of {
                Some(t) => counts[t],
                None => match scalar.dim.expr.eval(&dims).map_err(|e| Error::Launch(format!("'{stem}': {e}")))? {
                    Some(v) => v,
                    None => {
                        return Err(Error::Launch(format!(
                            "cannot determine the runtime argument '{}' ({}) from the shapes of the tensors",
                            scalar.name, scalar.dim.text
                        )));
                    }
                },
            };
            if !(0..=scalar.ty.max()).contains(&value) {
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
            if let Some(by) = scalar.divisible.filter(|by| value % by != 0) {
                return Err(Error::precondition(
                    PreconditionKind::Divisible,
                    Some(&scalar.name),
                    format!(
                        "runtime argument '{}' ({}) is {value}, which is not a multiple of {by}; '{stem}' is compiled with that assumption",
                        scalar.name, scalar.dim.text
                    ),
                ));
            }
            scalar_values[s] = value;
        }

        // -- a tensor that is written must not overlap another argument
        for (w, tw) in self.tensors.iter().enumerate().filter(|(_, t)| t.access != Access::In) {
            let (a_start, a_end) = extent(&tensors[w]);
            for (o, to) in self.tensors.iter().enumerate().filter(|(o, _)| *o != w) {
                let (b_start, b_end) = extent(&tensors[o]);
                if a_start < b_end && b_start < a_end {
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

        // -- the alignment each pointer promises (16 bytes; a strided pointer may promise only its element's), and a generated kernel
        // indexes with i32: the number of elements of a dense tensor, and the extent of a view, must not pass 2**31 - 1
        for (t, a) in self.tensors.iter().zip(tensors) {
            if a.ptr % t.alignment != 0 {
                return Err(Error::precondition(
                    PreconditionKind::PointerAlignment,
                    Some(&t.name),
                    format!(
                        "argument '{}' is at address {:#x}, which is not aligned to {} bytes; every pointer of '{stem}' is compiled as {}-byte aligned \
                         (allocate it aligned, or copy it)",
                        t.name, a.ptr, t.alignment, t.alignment
                    ),
                ));
            }
        }
        for (i, (t, a)) in self.tensors.iter().zip(tensors).enumerate() {
            if let Some(limit) = t.max_extent {
                let spans = reach_canonical(a.shape, &canonical[i]);
                if spans > limit as u128 {
                    return Err(Error::precondition(
                        PreconditionKind::MaxExtent,
                        Some(&t.name),
                        format!(
                            "argument '{}' (shape {:?}, strides {:?}) spans {spans} elements, more than the {limit} that the i32 offsets of '{stem}' reach",
                            t.name, a.shape, canonical[i]
                        ),
                    ));
                }
            }
            if let Some(limit) = t.max_elements.filter(|limit| counts[i] > *limit) {
                return Err(Error::precondition(
                    PreconditionKind::MaxElements,
                    Some(&t.name),
                    format!(
                        "argument '{}' has {} elements (shape {:?}), more than the {limit} that the i32 offsets of '{stem}' reach",
                        t.name, counts[i], a.shape
                    ),
                ));
            }
        }

        // -- the grid, from its expression, against what the device takes
        let mut grid = [1u32; 3];
        for (axis, entry) in self.grid.iter().enumerate() {
            let along = ["x", "y", "z"][axis];
            let value =
                entry.eval(&dims).map_err(|e| Error::Launch(format!("'{stem}': the grid {}: {e}", self.grid_text)))?;
            let Some(value) = value else {
                let mut vars = Vec::new();
                entry.vars(&mut vars);
                let free: Vec<&str> =
                    vars.into_iter().filter(|v| dims[*v].is_none()).map(|v| self.syms.name(v)).collect();
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
            if let Some(limit) = device.map(|d| d.max_grid[axis]).filter(|limit| value > *limit) {
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

        // -- the kernelParams values
        let mut values = [0u64; MAX_PARAMS];
        let mut tensor_of = [NOT_A_TENSOR; MAX_PARAMS];
        for (k, slot) in self.slots.iter().enumerate() {
            values[k] = match slot {
                Slot::Tensor(t) => {
                    tensor_of[k] = *t as u8; // at most MAX_PARAMS tensors: `Plan::new` refuses more parameters
                    tensors[*t].ptr
                }
                Slot::Scalar(s) => {
                    let v = scalar_values[*s];
                    match self.scalars[*s].ty {
                        ScalarType::I32 => u64::from(v as u32),
                        ScalarType::I64 => v as u64,
                    }
                }
                Slot::Fixed(bits) => *bits,
            };
        }
        // (the scratch slots of a Triton kernel stay 0: null pointers)
        Ok(Prepared { grid, values, count: self.slots.len() + self.scratch_slots, tensor_of })
    }

    fn solve(&self, dims: &mut [Option<i64>], equations: &[Equation<'_>]) -> Result<()> {
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
                        done[k] = true;
                        progress = true;
                    }
                    Ok(Solved::Stuck) => {}
                    Err(e) => return Err(self.unsolvable(eq.origin, eq.pattern, eq.actual, e, eq.counted)),
                }
            }
            if done.iter().all(|d| *d) {
                return Ok(());
            }
            if progress {
                continue;
            }
            // nothing more can be solved: a bucketed dimension is known from the binary itself
            let mut bound_any = false;
            for (dim, info) in self.dims.iter().enumerate() {
                let Some((value, _)) = &info.bucket else { continue };
                let wanted = equations.iter().zip(&done).any(|(eq, d)| !*d && eq.pattern.vars.contains(&dim));
                if wanted && dims[dim].is_none() {
                    dims[dim] = Some(*value);
                    bound_any = true;
                }
            }
            if !bound_any {
                let stuck = equations
                    .iter()
                    .zip(&done)
                    .find(|(_, d)| !**d)
                    .map(|(eq, _)| eq)
                    .expect("not all equations are done");
                let what = match stuck.origin {
                    Origin::Tensor { t, axis } => format!(
                        "argument '{}' ({}): dimension {axis} (size {})",
                        self.tensors[t].name,
                        self.describe_tensor(t),
                        stuck.actual
                    ),
                    Origin::Stride { t, axis } => format!(
                        "argument '{}' ({}): the stride of axis {axis} (value {})",
                        self.tensors[t].name,
                        self.describe_tensor(t),
                        stuck.actual
                    ),
                    Origin::Scalar { s } => {
                        format!("runtime argument '{}' (value {})", self.scalars[s].name, stuck.actual)
                    }
                };
                return Err(Error::precondition(
                    PreconditionKind::Shape,
                    None,
                    format!(
                        "'{}': {what}: cannot infer the dimensions in {}; the signature is ambiguous at this call",
                        self.stem, stuck.pattern.text
                    ),
                ));
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Origin {
    Tensor { t: usize, axis: usize },
    Stride { t: usize, axis: usize },
    Scalar { s: usize },
}

struct Equation<'p> {
    pattern: &'p Pattern,
    actual: i64,
    origin: Origin,
    counted: Option<(usize, i64)>,
    /// The least value the dimension it binds may have: 1 for a size, 0 for a stride.
    minimum: i64,
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
}
