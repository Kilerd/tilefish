//! Everything a launch checks, with no driver in it.
//!
//! A [`Plan`] is one manifest entry made ready for launching: the dimension names numbered, every expression parsed,
//! every constraint attached to what it is about. [`Plan::prepare`] takes the tensors of a call and either refuses it (with
//! the same checks, in the same order, as Python's `replay`) or returns the grid and the `kernelParams` values to launch
//! with. The driver is not involved (a device is described by a [`DeviceInfo`], which a test can make up), so all of it
//! is tested without a GPU.
//!
//! Everything comes from the manifest: each pointer parameter has its declared shape (`["Rows", 1024]`: a dimension
//! expression, or the size of a static axis) and the entry has the constraints of its named dimensions. The checks, in order:
//! the number of tensors; for each tensor its device, dtype and rank, and that no dimension is empty; the binding of the
//! dimensions from the sizes of the tensors, which must agree across arguments and with the static sizes (a port of
//! `solve_dims`); the bucket (a bucketed dimension is the bucket exactly: there is no padding) and `divisible_by` of every
//! dimension; the runtime scalars, derived from their dimension expression or from the element count of the tensor they count,
//! and their 32-bit range; aliasing of the tensors that are written; 16-byte alignment; `max_elements`. Last, the grid is
//! evaluated from its expression and checked against the device's limits. A call has nothing but its tensors: there is no way to
//! tell a launch what a dimension is.

use crate::error::{Error, PreconditionKind, Result};
use crate::expr::{Expr, MAX_DIMS, Pattern, Solved, Symbols, Unsolvable, parse_grid};
use crate::ffi::MAX_PARAMS;
use crate::manifest::{Access, Entry, ParamKind, Precondition, ShapeDim};
use crate::tensor::{DType, TensorArg};

/// The device a launch is checked against: its ordinal, and the largest grid it takes along each axis
/// (`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X/Y/Z`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceInfo {
    /// The ordinal of the device the kernel is loaded on: every tensor must be on it.
    pub ordinal: i32,
    /// The largest grid size along x, y and z.
    pub max_grid: [i64; 3],
}

#[derive(Debug, Clone)]
struct TensorPlan {
    name: String,
    dtype: DType,
    access: Access,
    shape: Vec<Pattern>,
    max_elements: Option<i64>,
}

#[derive(Debug, Clone)]
struct ScalarPlan {
    name: String,
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
}

impl Prepared {
    /// The `kernelParams` values, one 64-bit slot each: an address, or an `i32` in the low four bytes, then the two
    /// scratch slots (null pointers).
    pub fn params(&self) -> &[u64] {
        &self.values[..self.count]
    }
}

fn to_i64(v: u64, what: &str) -> Result<i64> {
    i64::try_from(v).map_err(|_| Error::manifest(format!("{what} {v} does not fit 64 bits")))
}

impl Plan {
    /// The plan of a manifest entry. Refuses a binary that a plain `cuLaunchKernel` cannot run (it needs scratch memory,
    /// programmatic dependent launch, clusters or a cooperative launch) and a pointer of a packed element type.
    pub fn new(entry: &Entry) -> Result<Self> {
        let who = format!("'{}' ({})", entry.stem, entry.arch);
        let blockers = entry.launch.blockers();
        if !blockers.is_empty() {
            return Err(Error::Unsupported(format!(
                "{who}: the binary needs launch attributes that this host does not set up: {}",
                blockers.join(", ")
            )));
        }
        if entry.params.len() + 2 > MAX_PARAMS {
            return Err(Error::Unsupported(format!(
                "{who}: a kernel with {} parameters is not supported",
                entry.params.len()
            )));
        }
        let bad = |e: String| Error::manifest(format!("{who}: {e}"));
        // the named dimensions come first, so that a symbol's number is its place in the table of the manifest
        let mut syms = Symbols::default();
        let mut dims = Vec::with_capacity(entry.dims.len());
        for d in &entry.dims {
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
        for p in &entry.params {
            match &p.kind {
                ParamKind::Pointer { dtype_name, dtype, access, shape } => {
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
                    slots.push(Slot::Tensor(tensors.len()));
                    tensors.push(TensorPlan {
                        name: p.name.clone(),
                        dtype,
                        access: *access,
                        shape: patterns,
                        max_elements: None,
                    });
                }
                ParamKind::Scalar { dim, .. } => {
                    slots.push(Slot::Scalar(scalars.len()));
                    let dim = Pattern::parse(dim, &format!("parameter {} dim", p.name), &mut syms).map_err(bad)?;
                    scalars.push(ScalarPlan { name: p.name.clone(), dim, count_of: None, divisible: None });
                }
            }
        }
        for p in &entry.params {
            if let ParamKind::Scalar { count_of: Some(target), .. } = &p.kind
                && let (Some(s), Some(t)) =
                    (scalars.iter().position(|s| s.name == p.name), tensors.iter().position(|t| t.name == *target))
            {
                scalars[s].count_of = Some(t);
            }
        }
        for pre in &entry.preconditions {
            match pre {
                Precondition::MaxElements { arg, value, .. } => {
                    if let Some(t) = tensors.iter_mut().find(|t| t.name == *arg) {
                        t.max_elements = Some(to_i64(*value, "max_elements")?);
                    }
                }
                Precondition::Divisible { arg, by, .. } => {
                    if let Some(s) = scalars.iter_mut().find(|s| s.name == *arg) {
                        s.divisible = Some(to_i64(*by, "a divisor")?);
                    }
                }
                // checked when the manifest is read: the alignment is 16 for every pointer, a static size is what the
                // shape says, and a bucket is what `dims` says
                Precondition::PointerAlignment { .. }
                | Precondition::StaticSize { .. }
                | Precondition::Bucket { .. } => {}
            }
        }
        let grid = parse_grid(&entry.grid, &mut syms).map_err(bad)?;
        if syms.len() != dims.len() {
            return Err(Error::manifest(format!("{who}: the entry's dims do not name every dimension it uses")));
        }
        Ok(Self {
            stem: entry.stem.clone(),
            syms,
            dims,
            tensors,
            scalars,
            slots,
            grid,
            grid_text: entry.grid.clone(),
            block: entry.block,
            shared: entry.shared,
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
        format!(
            "Tensor[{}, {}]",
            p.shape.iter().map(|d| d.text.as_str()).collect::<Vec<_>>().join(", "),
            p.dtype.name()
        )
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
            (Origin::Tensor { t, axis }, Unsolvable::Mismatch { expected }) => {
                let name = &self.tensors[t].name;
                let what =
                    if pattern.is_static { expected.to_string() } else { format!("{} = {expected}", pattern.text) };
                Error::precondition(
                    if pattern.is_static { PreconditionKind::StaticSize } else { PreconditionKind::Shape },
                    Some(name),
                    format!(
                        "'{}': argument '{name}' ({}): dimension {axis} is {actual}, but the signature requires {what}",
                        self.stem,
                        self.describe_tensor(t)
                    ),
                )
            }
            (Origin::Tensor { t, axis }, Unsolvable::NotWhole { var, target, coefficient }) => {
                let name = &self.tensors[t].name;
                Error::precondition(
                    PreconditionKind::Shape,
                    Some(name),
                    format!(
                        "'{}': argument '{name}' ({}): dimension {axis} is {actual}, which cannot match {} (would need {} = {target}/{coefficient})",
                        self.stem,
                        self.describe_tensor(t),
                        pattern.text,
                        name_of(var)
                    ),
                )
            }
            (Origin::Tensor { t, axis }, Unsolvable::NotPositive { var, value }) => {
                let name = &self.tensors[t].name;
                Error::precondition(
                    PreconditionKind::Shape,
                    Some(name),
                    format!(
                        "'{}': argument '{name}' ({}): dimension {axis} is {actual}, which cannot match {} (would need {} = {value})",
                        self.stem,
                        self.describe_tensor(t),
                        pattern.text,
                        name_of(var)
                    ),
                )
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
        }

        // -- the dimensions, bound from the sizes of the tensors (and the element counts the scalars carry)
        let mut dims: [Option<i64>; MAX_DIMS] = [None; MAX_DIMS];
        let mut equations: Vec<Equation<'_>> = Vec::new();
        for (t, a) in tensors.iter().enumerate() {
            for (axis, (pattern, size)) in self.tensors[t].shape.iter().zip(a.shape).enumerate() {
                let actual = i64::try_from(*size).map_err(|_| {
                    Error::precondition(
                        PreconditionKind::Shape,
                        Some(&self.tensors[t].name),
                        "a dimension does not fit 64 bits",
                    )
                })?;
                equations.push(Equation { pattern, actual, origin: Origin::Tensor { t, axis }, counted: None });
            }
        }
        for (s, scalar) in self.scalars.iter().enumerate() {
            if let Some(t) = scalar.count_of {
                equations.push(Equation {
                    pattern: &scalar.dim,
                    actual: counts[t],
                    origin: Origin::Scalar { s },
                    counted: Some((t, counts[t])),
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
            if !(0..=i64::from(i32::MAX)).contains(&value) {
                return Err(Error::precondition(
                    PreconditionKind::Shape,
                    Some(&scalar.name),
                    format!(
                        "runtime argument '{}' is {value}, which does not fit the 32-bit signed integer the binary takes",
                        scalar.name
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

        // -- every pointer of a binary is compiled as 16-byte aligned, and a generated kernel indexes with i32
        for (t, a) in self.tensors.iter().zip(tensors) {
            if a.ptr % 16 != 0 {
                return Err(Error::precondition(
                    PreconditionKind::PointerAlignment,
                    Some(&t.name),
                    format!(
                        "argument '{}' is at address {:#x}, which is not aligned to 16 bytes; every pointer of '{stem}' is compiled as 16-byte aligned \
                         (allocate it aligned, or copy it)",
                        t.name, a.ptr
                    ),
                ));
            }
        }
        for (i, (t, a)) in self.tensors.iter().zip(tensors).enumerate() {
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
        for (k, slot) in self.slots.iter().enumerate() {
            values[k] = match slot {
                Slot::Tensor(t) => tensors[*t].ptr,
                Slot::Scalar(s) => u64::from(scalar_values[*s] as u32),
            };
        }
        Ok(Prepared { grid, values, count: self.slots.len() + 2 }) // the two scratch slots stay 0: null pointers
    }

    fn solve(&self, dims: &mut [Option<i64>], equations: &[Equation<'_>]) -> Result<()> {
        let mut done = vec![false; equations.len()];
        loop {
            let mut progress = false;
            for (k, eq) in equations.iter().enumerate() {
                if done[k] {
                    continue;
                }
                match eq.pattern.solve(eq.actual, dims) {
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
    Scalar { s: usize },
}

struct Equation<'p> {
    pattern: &'p Pattern,
    actual: i64,
    origin: Origin,
    counted: Option<(usize, i64)>,
}

/// The bytes a tensor covers, `[start, end)`.
fn extent(a: &TensorArg<'_>) -> (u128, u128) {
    let bytes =
        a.shape.iter().fold(1u128, |acc, d| acc.saturating_mul(*d as u128)).saturating_mul(a.dtype.size() as u128);
    (u128::from(a.ptr), u128::from(a.ptr).saturating_add(bytes))
}
