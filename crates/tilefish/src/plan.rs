//! Everything a launch checks, with no driver in it.
//!
//! A [`Plan`] is one manifest entry made ready for launching: the dimension names numbered, every expression parsed,
//! every precondition attached to the argument it is about. [`Plan::prepare`] takes the arguments of a call and
//! either refuses it (with the same checks, in the same order, as Python's `replay`) or returns the grid and the
//! `kernelParams` values to launch with. The driver is not involved, so all of it is tested without a GPU.
//!
//! The checks, in order: the number of tensors; for each tensor its device, dtype, rank (against its shape pattern when
//! it has one, else against its static axes) and that no dimension is empty; the binding of the dimensions from the
//! shapes, the scalars and the dimensions the caller gives (a port of `solve_dims`); the bucket (a bucketed dimension is
//! the bucket exactly: there is no padding); the 32-bit range of the runtime scalars; aliasing of the tensors that
//! are written; and the preconditions of the manifest (16-byte alignment, `divisible_by`, static sizes, `max_elements`).
//! Last, the grid is evaluated from its expression.

use std::collections::HashMap;

use crate::error::{Error, PreconditionKind, Result};
use crate::expr::{Pattern, Solved, Symbols, Unsolvable, parse_grid};
use crate::ffi::MAX_PARAMS;
use crate::manifest::{Access, Entry, ParamKind, Precondition};
use crate::tensor::{DType, TensorArg};

/// The arguments of a launch: the tensors, in the order of the manifest's ABI, and the values the signature does not
/// determine by itself.
///
/// A scalar of the ABI is derived from the shapes (a generated launcher passes none): for a kernel whose shape patterns
/// are known (see [`Kernel::with_shapes`](crate::Kernel::with_shapes)) every dimension comes from the tensors, and a
/// scalar from its dimension expression or, for a generated kernel, from the element count of the tensor it counts.
/// Without shape patterns, give what the tensors cannot: `scalars` by the name of the ABI parameter, and `dims` by the
/// name of the dimension (such as the `Rows` of a grid `(Rows,)`). Whatever is given is checked against what is derived.
#[derive(Debug, Clone, Copy, Default)]
pub struct Args<'a> {
    tensors: &'a [TensorArg<'a>],
    scalars: &'a [(&'a str, i64)],
    dims: &'a [(&'a str, i64)],
}

impl<'a> Args<'a> {
    /// A call with these tensors and nothing else.
    pub fn new(tensors: &'a [TensorArg<'a>]) -> Self {
        Self { tensors, scalars: &[], dims: &[] }
    }

    /// Also give runtime scalars by the name of their ABI parameter.
    #[must_use]
    pub fn scalars(mut self, scalars: &'a [(&'a str, i64)]) -> Self {
        self.scalars = scalars;
        self
    }

    /// Also give dimensions by name.
    #[must_use]
    pub fn dims(mut self, dims: &'a [(&'a str, i64)]) -> Self {
        self.dims = dims;
        self
    }
}

#[derive(Debug, Clone)]
struct TensorPlan {
    name: String,
    dtype: DType,
    access: Access,
    statics: Vec<(usize, i64)>,
    max_elements: Option<i64>,
    shape: Option<Vec<Pattern>>,
}

#[derive(Debug, Clone)]
struct ScalarPlan {
    name: String,
    dim: Pattern,
    count_of: Option<usize>,
    divisible: Option<i64>,
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
    tensors: Vec<TensorPlan>,
    scalars: Vec<ScalarPlan>,
    slots: Vec<Slot>,
    grid: Vec<crate::expr::Expr>,
    grid_text: String,
    buckets: Vec<(usize, i64, Vec<u64>)>,
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
        let mut syms = Symbols::default();
        let bad = |e: String| Error::manifest(format!("{who}: {e}"));
        let mut tensors: Vec<TensorPlan> = Vec::new();
        let mut scalars: Vec<ScalarPlan> = Vec::new();
        let mut slots = Vec::new();
        for p in &entry.params {
            match &p.kind {
                ParamKind::Pointer { dtype_name, dtype, access } => {
                    let Some(dtype) = *dtype else {
                        return Err(Error::Unsupported(format!(
                            "{who}: parameter '{}' has dtype {dtype_name}, which has no storage type here (packed formats cannot be run yet)",
                            p.name
                        )));
                    };
                    slots.push(Slot::Tensor(tensors.len()));
                    tensors.push(TensorPlan {
                        name: p.name.clone(),
                        dtype,
                        access: *access,
                        statics: Vec::new(),
                        max_elements: None,
                        shape: None,
                    });
                }
                ParamKind::Scalar { dim, .. } => {
                    slots.push(Slot::Scalar(scalars.len()));
                    let dim = Pattern::parse(dim, &format!("parameter {} dim", p.name), &mut syms).map_err(bad)?;
                    scalars.push(ScalarPlan { name: p.name.clone(), dim, count_of: None, divisible: None });
                }
            }
        }
        let tensor_ix: HashMap<String, usize> = tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        let scalar_ix: HashMap<String, usize> = scalars.iter().enumerate().map(|(i, s)| (s.name.clone(), i)).collect();
        let tensor_index = |name: &str| tensor_ix.get(name).copied();
        let scalar_index = |name: &str| scalar_ix.get(name).copied();
        let mut count_of = Vec::new();
        for p in &entry.params {
            if let ParamKind::Scalar { count_of: Some(target), .. } = &p.kind {
                count_of.push((scalar_index(&p.name), tensor_index(target)));
            }
        }
        for (s, t) in count_of {
            if let (Some(s), Some(t)) = (s, t) {
                scalars[s].count_of = Some(t);
            }
        }
        let mut buckets = Vec::new();
        for pre in &entry.preconditions {
            match pre {
                Precondition::StaticSize { arg, axis, value, .. } => {
                    if let Some(t) = tensor_index(arg) {
                        tensors[t].statics.push((*axis, to_i64(*value, "a static size")?));
                    }
                }
                Precondition::MaxElements { arg, value, .. } => {
                    if let Some(t) = tensor_index(arg) {
                        tensors[t].max_elements = Some(to_i64(*value, "max_elements")?);
                    }
                }
                Precondition::Divisible { arg, by, .. } => {
                    if let Some(s) = scalar_index(arg) {
                        scalars[s].divisible = Some(to_i64(*by, "a divisor")?);
                    }
                }
                Precondition::Bucket { dim, value, buckets: all } => {
                    buckets.push((syms.intern(dim).map_err(bad)?, to_i64(*value, "a bucket")?, all.clone()));
                }
                Precondition::PointerAlignment { .. } => {}
            }
        }
        let grid = parse_grid(&entry.grid, &mut syms).map_err(bad)?;
        Ok(Self {
            stem: entry.stem.clone(),
            syms,
            tensors,
            scalars,
            slots,
            grid,
            grid_text: entry.grid.clone(),
            buckets,
            block: entry.block,
            shared: entry.shared,
        })
    }

    /// Say what shape each tensor has, as dimension expressions in the order of the ABI (`["M", "1000"]`): the form the
    /// signature of the declaration has. Every dimension then comes from the tensors of a call, and the tensors are
    /// checked against it (rank, static sizes, buckets, `max_elements`). What the manifest already says about a tensor
    /// (a static size) must agree with it.
    pub fn with_shapes(mut self, shapes: &[&[&str]]) -> Result<Self> {
        if shapes.len() != self.tensors.len() {
            return Err(Error::Launch(format!(
                "'{}' has {} tensors, but {} shapes were given",
                self.stem,
                self.tensors.len(),
                shapes.len()
            )));
        }
        for (t, texts) in self.tensors.iter_mut().zip(shapes) {
            let mut patterns = Vec::with_capacity(texts.len());
            for (axis, text) in texts.iter().enumerate() {
                let what = format!("the shape of '{}' axis {axis}", t.name);
                patterns.push(Pattern::parse(text, &what, &mut self.syms).map_err(Error::Launch)?);
            }
            for (axis, value) in &t.statics {
                let agrees =
                    patterns.get(*axis).is_some_and(|p| p.is_static && p.expr.eval(&[]).ok().flatten() == Some(*value));
                if !agrees {
                    return Err(Error::Launch(format!(
                        "'{}': the shape of '{}' does not have {value} on axis {axis}, which the manifest says the binary is built for",
                        self.stem, t.name
                    )));
                }
            }
            t.shape = Some(patterns);
        }
        Ok(self)
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
        match &p.shape {
            Some(patterns) => {
                format!(
                    "Tensor[{}, {}]",
                    patterns.iter().map(|p| p.text.as_str()).collect::<Vec<_>>().join(", "),
                    p.dtype.name()
                )
            }
            None => format!("Tensor[..., {}]", p.dtype.name()),
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
                    (Unsolvable::Mismatch { expected }, None) => {
                        format!(
                            "runtime argument '{name}' is {actual}, but its dimension {} is {expected}",
                            pattern.text
                        )
                    }
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
    /// `device` is the ordinal of the device the kernel is loaded on (`None` to skip that check).
    pub fn prepare(&self, args: &Args<'_>, device: Option<i32>) -> Result<Prepared> {
        let stem = self.stem.as_str();
        if args.tensors.len() != self.tensors.len() {
            return Err(Error::Launch(format!(
                "'{stem}' takes {} tensors ({}), got {}",
                self.tensors.len(),
                self.tensor_names().collect::<Vec<_>>().join(", "),
                args.tensors.len()
            )));
        }
        // -- each tensor on its own
        let mut counts = [0i64; MAX_PARAMS];
        for (i, (t, a)) in self.tensors.iter().zip(args.tensors).enumerate() {
            if let Some(d) = device.filter(|d| *d != a.device) {
                return Err(Error::precondition(
                    PreconditionKind::Device,
                    Some(&t.name),
                    format!("argument '{}' is on device {}, but '{stem}' is loaded on device {d}", t.name, a.device),
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
            let rank = a.shape.len();
            let wanted = t.shape.as_ref().map(Vec::len);
            let enough = t.statics.iter().all(|(axis, _)| *axis < rank);
            if wanted.is_some_and(|w| w != rank) || !enough {
                return Err(Error::precondition(
                    PreconditionKind::Rank,
                    Some(&t.name),
                    format!(
                        "argument '{}' has rank {rank} (shape {:?}), but the signature requires {} (rank {})",
                        t.name,
                        a.shape,
                        self.describe_tensor(i),
                        wanted.map_or_else(
                            || format!("at least {}", t.statics.iter().map(|(a, _)| a + 1).max().unwrap_or(0)),
                            |w| w.to_string()
                        )
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

        // -- the dimensions: seeded by what the call gives, then solved from the equations
        let mut dims: [Option<i64>; crate::expr::MAX_DIMS] = [None; crate::expr::MAX_DIMS];
        for (name, value) in args.dims {
            let Some(i) = self.syms.find(name) else {
                return Err(Error::Launch(format!(
                    "'{stem}' has no dimension '{name}' (its dimensions are {})",
                    self.dim_names()
                )));
            };
            if *value < 1 {
                return Err(Error::precondition(
                    PreconditionKind::Shape,
                    Some(name),
                    format!("the dimension {name} is {value}; a dimension is at least 1"),
                ));
            }
            if dims[i].is_some_and(|v| v != *value) {
                return Err(Error::Launch(format!("the dimension '{name}' is given twice, with different values")));
            }
            dims[i] = Some(*value);
        }
        let mut given: Vec<Option<i64>> = vec![None; self.scalars.len()];
        for (name, value) in args.scalars {
            let Some(s) = self.scalars.iter().position(|s| s.name == *name) else {
                return Err(Error::Launch(format!(
                    "'{stem}' has no runtime scalar '{name}' (it has {:?})",
                    self.scalars.iter().map(|s| s.name.as_str()).collect::<Vec<_>>()
                )));
            };
            if let Some(t) = self.scalars[s].count_of {
                return Err(Error::Launch(format!(
                    "the runtime scalar '{name}' is the element count of '{}' and is derived from it: it is not an argument",
                    self.tensors[t].name
                )));
            }
            if given[s].is_some_and(|v| v != *value) {
                return Err(Error::Launch(format!(
                    "the runtime scalar '{name}' is given twice, with different values"
                )));
            }
            given[s] = Some(*value);
        }

        let mut equations: Vec<Equation<'_>> = Vec::new();
        for (t, a) in args.tensors.iter().enumerate() {
            if let Some(patterns) = &self.tensors[t].shape {
                for (axis, (pattern, size)) in patterns.iter().zip(a.shape).enumerate() {
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
        }
        for (s, scalar) in self.scalars.iter().enumerate() {
            match (scalar.count_of, given[s]) {
                (Some(t), _) => equations.push(Equation {
                    pattern: &scalar.dim,
                    actual: counts[t],
                    origin: Origin::Scalar { s },
                    counted: Some((t, counts[t])),
                }),
                (None, Some(value)) => equations.push(Equation {
                    pattern: &scalar.dim,
                    actual: value,
                    origin: Origin::Scalar { s },
                    counted: None,
                }),
                (None, None) => {}
            }
        }
        self.solve(&mut dims, &equations)?;

        // -- the bucket: exact, there is no padding
        for (dim, value, all) in &self.buckets {
            match dims[*dim] {
                Some(found) if found != *value => {
                    return Err(Error::precondition(
                        PreconditionKind::Bucket,
                        Some(self.syms.name(*dim)),
                        format!(
                            "dimension {} is {found}, but '{stem}' is built for exactly {value} (buckets {all:?}); there is no padding: \
                             pad or slice in the host to a bucket size",
                            self.syms.name(*dim)
                        ),
                    ));
                }
                _ => dims[*dim] = Some(*value),
            }
        }

        // -- the runtime scalars
        let mut scalar_values = [0i64; MAX_PARAMS];
        for (s, scalar) in self.scalars.iter().enumerate() {
            let value = match (scalar.count_of, given[s]) {
                (Some(t), _) => counts[t],
                (None, Some(v)) => v,
                (None, None) => {
                    match scalar.dim.expr.eval(&dims).map_err(|e| Error::Launch(format!("'{stem}': {e}")))? {
                        Some(v) => v,
                        None => {
                            return Err(Error::Launch(format!(
                                "cannot determine the runtime argument '{}' ({}) from the tensor shapes: give it with Args::scalars, or the dimension with Args::dims",
                                scalar.name, scalar.dim.text
                            )));
                        }
                    }
                }
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
            let (a_start, a_end) = extent(&args.tensors[w]);
            for (o, to) in self.tensors.iter().enumerate().filter(|(o, _)| *o != w) {
                let (b_start, b_end) = extent(&args.tensors[o]);
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

        // -- what the manifest lists for each tensor
        for (t, a) in self.tensors.iter().zip(args.tensors) {
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
            for (axis, value) in &t.statics {
                let actual = a.shape[*axis];
                if i64::try_from(actual) != Ok(*value) {
                    return Err(Error::precondition(
                        PreconditionKind::StaticSize,
                        Some(&t.name),
                        format!(
                            "argument '{}': dimension {axis} is {actual}, but '{stem}' is built for {value}",
                            t.name
                        ),
                    ));
                }
            }
        }
        for (i, (t, a)) in self.tensors.iter().zip(args.tensors).enumerate() {
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

        // -- the grid
        let mut grid = [1u32; 3];
        for (axis, entry) in self.grid.iter().enumerate() {
            let value =
                entry.eval(&dims).map_err(|e| Error::Launch(format!("'{stem}': the grid {}: {e}", self.grid_text)))?;
            let Some(value) = value else {
                let mut vars = Vec::new();
                entry.vars(&mut vars);
                let free: Vec<&str> =
                    vars.into_iter().filter(|v| dims[*v].is_none()).map(|v| self.syms.name(v)).collect();
                return Err(Error::Launch(format!(
                    "the grid {} of '{stem}' mentions {free:?}, which no argument determines: give them with Args::dims, or the shapes with Kernel::with_shapes",
                    self.grid_text
                )));
            };
            if value < 1 {
                return Err(Error::Launch(format!(
                    "the grid {} of '{stem}' is {value} here; every grid size must be at least 1",
                    self.grid_text
                )));
            }
            grid[axis] = u32::try_from(value).map_err(|_| {
                Error::Unsupported(format!(
                    "the grid {} of '{stem}' has {value} blocks along {}, more than 2^32 - 1",
                    self.grid_text,
                    "XYZ".as_bytes()[axis] as char
                ))
            })?;
        }

        // -- the kernelParams values
        let mut values = [0u64; MAX_PARAMS];
        for (k, slot) in self.slots.iter().enumerate() {
            values[k] = match slot {
                Slot::Tensor(t) => args.tensors[*t].ptr,
                Slot::Scalar(s) => u64::from(scalar_values[*s] as u32),
            };
        }
        Ok(Prepared { grid, values, count: self.slots.len() + 2 }) // the two scratch slots stay 0: null pointers
    }

    fn dim_names(&self) -> String {
        (0..self.syms.len()).map(|i| self.syms.name(i)).collect::<Vec<_>>().join(", ")
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
            for (dim, value, _) in &self.buckets {
                let wanted = equations.iter().zip(&done).any(|(eq, d)| !*d && eq.pattern.vars.contains(dim));
                if wanted && dims[*dim].is_none() {
                    dims[*dim] = Some(*value);
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
