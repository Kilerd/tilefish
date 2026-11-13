//! Reading a launch plan: `plan.json`, schema `tilefish.plan.v1` (`docs/plan-schema-1.md`).
//!
//! A port of `read_plan`, `validate_doc`, `plan_digest` and `check_against_manifest` of `python/src/tilefish/_plan.py`. A plan is a
//! `@tilefish.graph` compiled into, for every combination of the bucket values of its dimensions, an ordered list of *steps* (each one
//! launch of a manifest entry: a Triton kernel, a prebuilt binary or a pinned cuBLASLt GEMM) over *buffers*: the caller's tensors (the
//! *externals*) and the regions of one *arena* (the intermediates, the slots of the device scalars, the GEMMs' workspace), at offsets that
//! are the same in every bucket.
//!
//! [`LaunchPlan::load`] reads the file as strict JSON (no duplicate key, no `NaN`, no number a double cannot hold, at most 256 MiB, UTF-8),
//! computes its digest ([`plan_digest`], the canonical JSON of the manifest's digest, tagged `tilefish.plan-digest.v1`, without the keys
//! `plan_digest` and `tilefish`) and refuses it unless that digest is the pin (or the directory is trusted, [`Pin::TrustDirectory`]) and the
//! `plan_digest` the file carries; then every rule of the format, exactly as Python's `validate_doc` has them; then it loads the manifest in
//! the same directory, pinned to the plan's `manifest_digest` (so pinning the plan pins every binary and every GEMM algorithm it launches),
//! and checks every step against the manifest's entries. A host that has no graph cannot tell a plan that is stale against the graph
//! (Python's sixth check): the pin is what vouches for it.
//!
//! Nothing here needs a GPU: a plan is run by [`crate::PlanRun`].

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Map, Value};

use crate::elf::sha256_hex;
use crate::error::{Error, Result};
use crate::expr::{Symbols, parse_expr};
use crate::json::{canonical, parse_strict, type_name};
use crate::manifest::{
    Access, MANIFEST_NAME, Manifest, ParamKind, Pin, ShapeDim, is_arch, is_digest, is_ident, is_stem,
};
use crate::tensor::DType;

/// The `schema` of a plan document. Python's `PLAN_SCHEMA`.
pub const PLAN_SCHEMA: &str = "tilefish.plan.v1";
/// The file a plan directory has, next to its `manifest.json`.
pub const PLAN_NAME: &str = "plan.json";
/// The alignment of the arena and of every offset in it, in bytes.
pub const ARENA_ALIGNMENT: u64 = 256;
/// Part of what is hashed: another canonical form must change it. Python's `DIGEST_TAG`.
const DIGEST_TAG: &[u8] = b"tilefish.plan-digest.v1\0";
const MAX_PLAN_BYTES: u64 = 1 << 28;
const MAX_BUCKETS: u128 = 1 << 12;
const MAX_BUFFERS: usize = 1 << 15;
const MAX_STEPS: usize = 1 << 20;
const MAX_ARGS: usize = 256;
/// The integers a double holds exactly: a plan number is read as one in every language.
const MAX_INT: u64 = 1 << 53;
/// A refusal quotes at most this many characters of what it refuses.
const MAX_QUOTE: usize = 1000;

const TOP_KEYS: [&str; 13] = [
    "schema",
    "tilefish",
    "graph",
    "arch",
    "manifest_digest",
    "dims",
    "reuse",
    "arena_alignment",
    "arena_bytes",
    "buffers",
    "outputs",
    "buckets",
    "plan_digest",
];
const BUFFER_KEYS: [&str; 6] = ["name", "role", "dtype", "shape", "offset", "bytes"];
const STEP_KEYS: [&str; 3] = ["entry", "kind", "args"];
/// A copy step has no entry (it launches no binary) and no workspace.
const COPY_KEYS: [&str; 2] = ["kind", "args"];

/// What a buffer of a plan is, and who provides it (`docs/plan-schema-1.md`, section 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferRole {
    /// A graph parameter whose shape mentions a dimension (an activation): the caller's, read.
    Input,
    /// A returned value: the caller's, written (first), then read and written.
    Output,
    /// An `InOut` parameter: the caller's, updated in place.
    Inout,
    /// A graph parameter whose shape mentions no dimension: the caller's, read.
    Weight,
    /// A result that is not returned: in the arena.
    Intermediate,
    /// A `DevScalar` parameter: one element in the arena, which the host writes from the value the caller gives before the first launch.
    Devscalar,
    /// The cuBLASLt workspace of the GEMMs: in the arena.
    Workspace,
}

impl BufferRole {
    const ALL: [Self; 7] =
        [Self::Input, Self::Output, Self::Inout, Self::Weight, Self::Intermediate, Self::Devscalar, Self::Workspace];

    /// The word a plan uses (`"input"`, `"devscalar"`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
            Self::Inout => "inout",
            Self::Weight => "weight",
            Self::Intermediate => "intermediate",
            Self::Devscalar => "devscalar",
            Self::Workspace => "workspace",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|r| r.as_str() == name)
    }

    /// Whether the caller provides the buffer (`input`, `output`, `inout`, `weight`); the others are in the arena.
    pub fn is_external(self) -> bool {
        matches!(self, Self::Input | Self::Output | Self::Inout | Self::Weight)
    }

    /// Whether a step may write the buffer (`intermediate`, `output`, `inout`).
    pub fn is_writable(self) -> bool {
        matches!(self, Self::Intermediate | Self::Output | Self::Inout)
    }

    /// Whether the caller's tensor for the buffer is a graph parameter (`input`, `weight`, `inout`): what binds the dimensions of a call.
    pub fn is_parameter(self) -> bool {
        matches!(self, Self::Input | Self::Weight | Self::Inout)
    }
}

impl std::fmt::Display for BufferRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One buffer of a plan: a tensor it names (section 6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanBuffer {
    /// Unique in the plan: a parameter's own name, `out0`, `out1`, ... for the returned values, `t0`, `t1`, ... for the intermediates.
    pub name: String,
    /// What it is.
    pub role: BufferRole,
    /// Its element type.
    pub dtype: DType,
    /// Its shape: sizes and dimension expressions over the plan's dimensions (`[]` for a device scalar).
    pub shape: Vec<ShapeDim>,
    /// The byte offset in the arena (a multiple of 256), for a buffer that lives there; `None` for an external.
    pub offset: Option<u64>,
    /// The largest size it has over the buckets, in bytes.
    pub bytes: u64,
}

/// What a step is: a launch of an entry of one section of the manifest, or a copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind {
    /// An entry of `kernels`: a Triton kernel or a prebuilt binary.
    Kernel,
    /// An entry of `library`: a pinned cuBLASLt GEMM.
    Library,
    /// A copy of the dense bytes of one argument (`src`) into another (`dst`): `cuMemcpyDtoDAsync`, no manifest entry.
    Copy,
}

/// A view of a buffer that a step passes: element shape, element strides and the element offset of its first element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanView {
    /// The shape.
    pub shape: Vec<usize>,
    /// The element strides, one per axis.
    pub strides: Vec<usize>,
    /// The element offset of the first element in the buffer.
    pub offset: usize,
}

/// One pointer argument of a step: the ABI's parameter, the buffer it is bound to, and the view of it (none: the whole buffer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanArg {
    /// The name of the ABI's pointer parameter.
    pub name: String,
    /// The buffer.
    pub binding: String,
    /// The view of the buffer, if the argument is not the whole buffer.
    pub view: Option<PlanView>,
    pub(crate) buffer: usize,
}

/// One launch of a manifest entry (section 7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanStep {
    /// The stem of the entry (`rmsnorm`, `gemm_qkv.M64`); empty for a copy step, which launches no entry.
    pub entry: String,
    /// The section of the manifest the entry is in.
    pub kind: StepKind,
    /// The pointer arguments, in the order of the entry's ABI.
    pub args: Vec<PlanArg>,
    /// The workspace buffer of a library step.
    pub workspace: Option<String>,
    pub(crate) workspace_buffer: Option<usize>,
}

/// The steps of one bucket combination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanBucket {
    /// The value of every dimension, by name in code point order (empty for a plan with no dimension).
    pub values: Vec<(String, u64)>,
    /// The steps, in launch order.
    pub steps: Vec<PlanStep>,
    /// The shape of every buffer at this combination, in the order of the buffers.
    pub(crate) shapes: Vec<Vec<usize>>,
}

impl PlanBucket {
    /// The shape of the buffer of index `buffer` (in [`LaunchPlan::buffers`]) at this combination.
    pub fn shape_of(&self, buffer: usize) -> Option<&[usize]> {
        self.shapes.get(buffer).map(Vec::as_slice)
    }
}

/// A verified launch plan and its verified manifest: what [`LaunchPlan::instantiate`](crate::LaunchPlan::instantiate) runs. Cheap to clone.
#[derive(Debug, Clone)]
pub struct LaunchPlan {
    pub(crate) inner: Arc<Inner>,
}

#[derive(Debug)]
pub(crate) struct Inner {
    pub(crate) label: String,
    pub(crate) version: String,
    pub(crate) graph: String,
    pub(crate) arch: String,
    pub(crate) manifest_digest: String,
    pub(crate) digest: String,
    pub(crate) reuse: bool,
    pub(crate) arena_bytes: u64,
    pub(crate) dims: Vec<(String, Vec<u64>)>,
    pub(crate) buffers: Vec<PlanBuffer>,
    pub(crate) outputs: Vec<String>,
    pub(crate) buckets: Vec<PlanBucket>,
    pub(crate) manifest: Manifest,
}

/// `plan_digest(doc)` of Python: `sha256(b"tilefish.plan-digest.v1\0" + canonical_json(doc without "plan_digest" and "tilefish"))`, in lowercase
/// hex. The canonical JSON is the manifest's ([`crate::json::canonical`]). A document that is not an object is digested as it is.
pub fn plan_digest(doc: &Value) -> String {
    let body = match doc {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(k, _)| k.as_str() != "plan_digest" && k.as_str() != "tilefish")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
        other => other.clone(),
    };
    let mut bytes = DIGEST_TAG.to_vec();
    bytes.extend_from_slice(canonical(&body).as_bytes());
    sha256_hex(&bytes)
}

/// A refusal of the document, quoting at most [`MAX_QUOTE`] characters (Python's `_bad`).
fn bad(message: impl Into<String>) -> Error {
    let message = message.into();
    let count = message.chars().count();
    let text = if count > MAX_QUOTE {
        let cut: String = message.chars().take(MAX_QUOTE).collect();
        format!("{cut}... ({count} characters in all)")
    } else {
        message
    };
    Error::Plan(format!("plan.json: {text}"))
}

type Check<T> = std::result::Result<T, Error>;

fn refuse<T>(message: impl Into<String>) -> Check<T> {
    Err(bad(message))
}

/// Python's `type(v) is int`: a JSON integer (not a float, not a boolean).
fn int_of(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) if n.is_i64() || n.is_u64() => n.as_i64(),
        _ => None,
    }
}

fn brief(v: &Value) -> String {
    let text = v.to_string();
    if text.chars().count() > 200 { format!("{}...", text.chars().take(200).collect::<String>()) } else { text }
}

/// Python's `_keys`: an object with exactly `keys` (and optionally `optional`).
fn keys<'v>(obj: &'v Value, what: &str, keys: &[&str], optional: &[&str]) -> Check<&'v Map<String, Value>> {
    let Value::Object(map) = obj else {
        return refuse(format!("{what} must be an object, got {}", type_name(obj)));
    };
    let missing: Vec<&str> = keys.iter().copied().filter(|k| !map.contains_key(*k)).collect();
    let mut unknown: Vec<&str> =
        map.keys().map(String::as_str).filter(|k| !keys.contains(k) && !optional.contains(k)).collect();
    unknown.sort_unstable();
    if missing.is_empty() && unknown.is_empty() {
        return Ok(map);
    }
    let mut message = format!("{what} must have exactly the fields {keys:?}");
    if !optional.is_empty() {
        message.push_str(&format!(" (and optionally {optional:?})"));
    }
    if !missing.is_empty() {
        message.push_str(&format!("; missing {missing:?}"));
    }
    if !unknown.is_empty() {
        message.push_str(&format!("; unknown {unknown:?}"));
    }
    refuse(message)
}

fn align_up(n: u128) -> u128 {
    n.div_ceil(u128::from(ARENA_ALIGNMENT)) * u128::from(ARENA_ALIGNMENT)
}

/// Every integer of the document fits a double exactly (Python's `_parse_int` refuses the others while it parses).
fn integers_fit_a_double(v: &Value) -> Check<()> {
    match v {
        Value::Number(n) if !n.is_f64() => {
            let fits = n.as_i64().map(i64::unsigned_abs).or_else(|| n.as_u64()).is_some_and(|m| m <= MAX_INT);
            if fits {
                Ok(())
            } else {
                Err(Error::Plan(format!(
                    "the plan is not strict JSON: the integer {n} does not fit a double (at most 2**53 in magnitude)"
                )))
            }
        }
        Value::Array(items) => items.iter().try_for_each(integers_fit_a_double),
        Value::Object(map) => map.values().try_for_each(integers_fit_a_double),
        _ => Ok(()),
    }
}

/// One shape entry of a buffer, ready to be evaluated at a combination.
enum ShapeEntry {
    Static(u64),
    /// The folded form (`None`: the expression is not affine), and the names its indices are of.
    Expr {
        affine: Option<crate::expr::Affine>,
        names: Vec<String>,
        text: String,
    },
}

impl ShapeEntry {
    fn parse(e: &Value, buffer: &str) -> Check<Self> {
        if let Some(n) = int_of(e) {
            return Ok(Self::Static(n as u64)); // checked >= 1 by the caller
        }
        let Value::String(text) = e else { unreachable!("checked by the caller") };
        let mut syms = Symbols::default();
        let expr = parse_expr(text, &format!("the shape of buffer '{buffer}'"), &mut syms).map_err(bad)?;
        let names = (0..syms.len()).map(|i| syms.name(i).to_owned()).collect();
        // Python folds the expression with its dimension algebra (`DimExpr`) and has no value for one that is not affine (two symbolic
        // factors, an inexact `//`), whatever the values; `Expr::affine` is the port of that fold
        Ok(Self::Expr { affine: expr.affine(), names, text: text.clone() })
    }

    fn eval(&self, combo: &[(String, u64)], buffer: &str) -> Check<i128> {
        match self {
            Self::Static(n) => Ok(i128::from(*n)),
            Self::Expr { affine, names, text } => {
                let values: Vec<Option<i64>> =
                    names.iter().map(|n| combo.iter().find(|(d, _)| d == n).map(|(_, v)| *v as i64)).collect();
                // the folded form is evaluated, as Python substitutes into its folded `DimExpr`: a name that cancels needs no value
                let value = match affine {
                    Some(affine) => affine.eval(&values).map_err(bad)?,
                    None => None,
                };
                match value {
                    Some(v) => Ok(i128::from(v)),
                    None => refuse(format!(
                        "the shape {text:?} of buffer '{buffer}' has no value at {}",
                        describe_values(combo)
                    )),
                }
            }
        }
    }
}

pub(crate) fn describe_values(values: &[(String, u64)]) -> String {
    if values.is_empty() {
        return "its only size".to_owned();
    }
    let items: Vec<String> = values.iter().map(|(n, v)| format!("{n:?}: {v}")).collect();
    format!("{{{}}}", items.join(", "))
}

/// Every combination of bucket values: the names sorted, the first varying slowest, each list ascending (Python's `bucket_combinations`).
fn combinations(dims: &[(String, Vec<u64>)]) -> Vec<Vec<(String, u64)>> {
    let mut sorted: Vec<&(String, Vec<u64>)> = dims.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out: Vec<Vec<(String, u64)>> = vec![Vec::new()];
    for (name, values) in sorted {
        let mut next = Vec::with_capacity(out.len() * values.len());
        for prefix in &out {
            for v in values {
                let mut combo = prefix.clone();
                combo.push((name.clone(), *v));
                next.push(combo);
            }
        }
        out = next;
    }
    out
}

/// What `validate` reads out of a valid document.
struct Validated {
    version: String,
    graph: String,
    arch: String,
    manifest_digest: String,
    reuse: bool,
    arena_bytes: u64,
    dims: Vec<(String, Vec<u64>)>,
    buffers: Vec<PlanBuffer>,
    outputs: Vec<String>,
    buckets: Vec<PlanBucket>,
}

/// Every rule of sections 2 and 5 to 8, in the order of Python's `_validate` (`validate_doc`).
fn validate(doc: &Value) -> Check<Validated> {
    let top = keys(doc, "the document", &TOP_KEYS, &[])?;
    if top["schema"].as_str() != Some(PLAN_SCHEMA) {
        return refuse(format!("schema must be the string {PLAN_SCHEMA:?}, got {}", brief(&top["schema"])));
    }
    let Some(version) = top["tilefish"].as_str() else { return refuse("tilefish must be a string") };
    let graph = match top["graph"].as_str() {
        Some(g) if is_ident(g) => g.to_owned(),
        _ => return refuse(format!("graph must be an identifier, got {}", brief(&top["graph"]))),
    };
    let arch = match top["arch"].as_str() {
        Some(a) if is_arch(a) => a.to_owned(),
        _ => return refuse(format!("arch must look like 'sm_103', got {}", brief(&top["arch"]))),
    };
    for key in ["manifest_digest", "plan_digest"] {
        if !top[key].as_str().is_some_and(is_digest) {
            return refuse(format!("{key} must be 64 lowercase hex digits (a sha256), got {}", brief(&top[key])));
        }
    }
    let manifest_digest = top["manifest_digest"].as_str().unwrap_or_default().to_owned();
    let Some(reuse) = top["reuse"].as_bool() else {
        return refuse(format!("reuse must be true or false, got {}", brief(&top["reuse"])));
    };
    if int_of(&top["arena_alignment"]) != Some(ARENA_ALIGNMENT as i64) {
        return refuse(format!("arena_alignment must be {ARENA_ALIGNMENT}, got {}", brief(&top["arena_alignment"])));
    }
    let Value::Object(dims_doc) = &top["dims"] else { return refuse("dims must be an object") };
    let mut dims: Vec<(String, Vec<u64>)> = Vec::with_capacity(dims_doc.len());
    for (name, list) in dims_doc {
        if !is_ident(name) {
            return refuse(format!("the dimension name {name:?} is not an identifier"));
        }
        let values: Option<Vec<i64>> = list.as_array().map(|items| items.iter().map(int_of).collect()).unwrap_or(None);
        let strictly_increasing = |vs: &[i64]| vs.windows(2).all(|w| w[0] < w[1]);
        match values {
            Some(vs) if !vs.is_empty() && vs.iter().all(|v| *v >= 1) && strictly_increasing(&vs) => {
                dims.push((name.clone(), vs.into_iter().map(|v| v as u64).collect()));
            }
            _ => {
                return refuse(format!(
                    "dims.{name} must be a non-empty, strictly increasing list of positive ints, got {}",
                    brief(list)
                ));
            }
        }
    }

    // -- buffers
    let buffers_doc = match top["buffers"].as_array() {
        Some(list) if !list.is_empty() => list,
        _ => return refuse("buffers must be a non-empty list"),
    };
    if buffers_doc.len() > MAX_BUFFERS {
        return refuse(format!("a plan has at most {MAX_BUFFERS} buffers, this one has {}", buffers_doc.len()));
    }
    let mut buffers: Vec<PlanBuffer> = Vec::with_capacity(buffers_doc.len());
    let mut entries: Vec<Vec<ShapeEntry>> = Vec::with_capacity(buffers_doc.len());
    let mut index: HashMap<String, usize> = HashMap::with_capacity(buffers_doc.len());
    for (i, b) in buffers_doc.iter().enumerate() {
        let b = keys(b, &format!("buffers[{i}]"), &BUFFER_KEYS, &[])?;
        let name = match b["name"].as_str() {
            Some(n) if is_ident(n) && !index.contains_key(n) => n.to_owned(),
            _ => return refuse(format!("buffers[{i}].name must be a unique identifier, got {}", brief(&b["name"]))),
        };
        let Some(role) = b["role"].as_str().and_then(BufferRole::from_name) else {
            let roles: Vec<&str> = BufferRole::ALL.iter().map(|r| r.as_str()).collect();
            return refuse(format!("buffers[{i}].role must be one of {roles:?}, got {}", brief(&b["role"])));
        };
        let Some(dtype) = b["dtype"].as_str().and_then(DType::from_name) else {
            return refuse(format!(
                "buffers[{i}].dtype must be one of [\"bf16\", \"e4m3\", \"e5m2\", \"f16\", \"f32\", \"i32\", \"i64\", \"u8\"], got {}",
                brief(&b["dtype"])
            ));
        };
        let shape_ok = |e: &Value| int_of(e).is_some_and(|n| n >= 1) || e.is_string();
        let Some(shape_doc) = b["shape"].as_array().filter(|s| s.iter().all(shape_ok)) else {
            return refuse(format!("buffers[{i}].shape must be a list of positive ints and dimension expressions"));
        };
        let bytes = match int_of(&b["bytes"]) {
            Some(n) if n >= 1 => n as u64,
            _ => return refuse(format!("buffers[{i}].bytes must be a positive int, got {}", brief(&b["bytes"]))),
        };
        let offset = if role.is_external() {
            if !b["offset"].is_null() {
                return refuse(format!(
                    "buffers[{i}] ('{name}') is external (role {role}): its offset must be null, not {}",
                    brief(&b["offset"])
                ));
            }
            None
        } else {
            match int_of(&b["offset"]) {
                Some(n) if n >= 0 && (n as u64).is_multiple_of(ARENA_ALIGNMENT) => Some(n as u64),
                _ => {
                    return refuse(format!(
                        "buffers[{i}] ('{name}') lives in the arena: its offset must be a non-negative multiple of {ARENA_ALIGNMENT}, got {}",
                        brief(&b["offset"])
                    ));
                }
            }
        };
        let shape: Vec<ShapeDim> = shape_doc
            .iter()
            .map(|e| match int_of(e) {
                Some(n) => ShapeDim::Static(n as u64),
                None => ShapeDim::Expr(e.as_str().unwrap_or_default().to_owned()),
            })
            .collect();
        index.insert(name.clone(), i);
        entries.push(Vec::new()); // parsed below, when the shapes are evaluated (Python parses them there)
        buffers.push(PlanBuffer { name, role, dtype, shape, offset, bytes });
    }
    if buffers.iter().filter(|b| b.role == BufferRole::Workspace).count() > 1 {
        return refuse("at most one buffer has the role workspace");
    }
    let outputs: Vec<String> = match top["outputs"].as_array() {
        Some(list) => {
            let names: Option<Vec<String>> = list.iter().map(|o| o.as_str().map(str::to_owned)).collect();
            let wanted: HashSet<&str> =
                buffers.iter().filter(|b| b.role == BufferRole::Output).map(|b| b.name.as_str()).collect();
            match names {
                Some(names)
                    if names.iter().all(|o| index.get(o).is_some_and(|i| buffers[*i].role == BufferRole::Output))
                        && names.iter().collect::<HashSet<_>>().len() == names.len()
                        && names.iter().map(String::as_str).collect::<HashSet<_>>() == wanted =>
                {
                    names
                }
                _ => {
                    return refuse(
                        "outputs must list the buffers of role output, each once, in the order the graph returns them",
                    );
                }
            }
        }
        None => {
            return refuse(
                "outputs must list the buffers of role output, each once, in the order the graph returns them",
            );
        }
    };
    for b in &buffers {
        if b.role == BufferRole::Devscalar && !b.shape.is_empty() {
            return refuse(format!("the devscalar '{}' is one element in device memory: its shape must be []", b.name));
        }
        if b.role == BufferRole::Workspace
            && (b.dtype != DType::U8 || b.shape.len() != 1 || !matches!(b.shape[0], ShapeDim::Static(_)))
        {
            return refuse(format!("the workspace '{}' must be a u8 buffer with one static size", b.name));
        }
    }

    // -- buckets (counted before they are listed: a document must not make a reader build a list of 10**30 combinations)
    let n_combos = dims.iter().try_fold(1u128, |acc, (_, v)| acc.checked_mul(v.len() as u128)).unwrap_or(u128::MAX);
    if n_combos > MAX_BUCKETS {
        return refuse(format!(
            "dims have {n_combos} bucket combinations, more than the {MAX_BUCKETS} a plan may have"
        ));
    }
    let buckets_doc = match top["buckets"].as_array() {
        Some(list) if list.len() as u128 == n_combos => list,
        _ => return refuse(format!("buckets must have one entry per combination of the dims ({n_combos}), in order")),
    };
    let combos = combinations(&dims);
    for (i, b) in buffers_doc.iter().enumerate() {
        let shape = b["shape"].as_array().expect("checked above");
        entries[i] =
            shape.iter().map(|e| ShapeEntry::parse(e, &buffers[i].name)).collect::<Check<Vec<ShapeEntry>>>()?;
    }
    let mut sizes = vec![0u128; buffers.len()];
    let mut shapes_by_bucket: Vec<Vec<Vec<usize>>> = Vec::with_capacity(combos.len());
    for (combo, bucket) in combos.iter().zip(buckets_doc) {
        let bucket = keys(bucket, "a bucket", &["values", "steps"], &[])?;
        let follows = match &bucket["values"] {
            Value::Object(values) => {
                values.len() == combo.len()
                    && combo.iter().all(|(n, v)| values.get(n).and_then(int_of) == Some(*v as i64))
            }
            _ => false,
        };
        if !follows {
            return refuse(format!(
                "the buckets must follow the combinations of the dims in order: expected {}, got {}",
                describe_values(combo),
                brief(&bucket["values"])
            ));
        }
        let mut shapes = Vec::with_capacity(buffers.len());
        for (i, b) in buffers.iter().enumerate() {
            let mut shape = Vec::with_capacity(entries[i].len());
            for e in &entries[i] {
                shape.push(e.eval(combo, &b.name)?);
            }
            if shape.iter().any(|d| *d < 1) {
                return refuse(format!(
                    "the shape of '{}' at {} has a non-positive size {shape:?}",
                    b.name,
                    describe_values(combo)
                ));
            }
            let elements = shape.iter().try_fold(1u128, |acc, d| acc.checked_mul(*d as u128)).unwrap_or(u128::MAX);
            sizes[i] = sizes[i].max(elements.saturating_mul(b.dtype.size() as u128));
            // a size that does not fit is far more than the bytes of the buffer, which is refused next
            shapes.push(shape.into_iter().map(|d| usize::try_from(d).unwrap_or(usize::MAX)).collect());
        }
        shapes_by_bucket.push(shapes);
    }
    for (b, size) in buffers.iter().zip(&sizes) {
        if u128::from(b.bytes) != *size {
            return refuse(format!(
                "buffers '{}': bytes must be the largest size of its shape over the buckets ({size}), got {}",
                b.name, b.bytes
            ));
        }
    }

    // -- steps
    let n_steps = buckets_doc[0].get("steps").and_then(Value::as_array).map_or(0, Vec::len);
    if n_steps > MAX_STEPS {
        return refuse(format!("a bucket has at most {MAX_STEPS} steps, this one has {n_steps}"));
    }
    type Skeleton = Vec<(StepKind, Option<String>, Vec<(String, String)>)>;
    let mut skeleton: Option<Skeleton> = None;
    let mut first: HashMap<usize, usize> = HashMap::new();
    let mut last: HashMap<usize, usize> = HashMap::new();
    let mut buckets: Vec<PlanBucket> = Vec::with_capacity(combos.len());
    for (bi, ((bucket, shapes), combo)) in buckets_doc.iter().zip(shapes_by_bucket).zip(&combos).enumerate() {
        let steps_doc = match bucket["steps"].as_array() {
            Some(list) if list.len() == n_steps => list,
            _ => return refuse(format!("every bucket must have the same number of steps ({n_steps})")),
        };
        let mut this: Skeleton = Vec::with_capacity(n_steps);
        let mut steps = Vec::with_capacity(n_steps);
        for (si, st) in steps_doc.iter().enumerate() {
            // a copy step has no entry (it launches no binary) and no workspace
            let is_copy = st.get("kind").and_then(Value::as_str) == Some("copy");
            let (st, entry) = if is_copy {
                (keys(st, &format!("step #{si}"), &COPY_KEYS, &[])?, String::new())
            } else {
                let st = keys(st, &format!("step #{si}"), &STEP_KEYS, &["workspace"])?;
                match st["entry"].as_str() {
                    Some(e) if is_stem(e) => (st, e.to_owned()),
                    _ => {
                        return refuse(format!(
                            "step #{si}: entry must be a manifest stem such as 'gemm.M64', got {}",
                            brief(&st["entry"])
                        ));
                    }
                }
            };
            let kind = match st["kind"].as_str() {
                Some("kernel") => StepKind::Kernel,
                Some("library") => StepKind::Library,
                Some("copy") => StepKind::Copy,
                _ => {
                    return refuse(format!(
                        "step #{si}: kind must be one of [\"kernel\", \"library\", \"copy\"], got {}",
                        brief(&st["kind"])
                    ));
                }
            };
            if (kind == StepKind::Library) != st.contains_key("workspace") {
                return refuse(format!("step #{si}: exactly the steps of kind 'library' have a workspace"));
            }
            let (workspace, workspace_buffer) = match st.get("workspace") {
                Some(ws) => match ws.as_str().and_then(|w| index.get(w).map(|i| (w, *i))) {
                    Some((w, i)) if buffers[i].role == BufferRole::Workspace => (Some(w.to_owned()), Some(i)),
                    _ => {
                        return refuse(format!(
                            "step #{si}: workspace must name the buffer of role workspace, got {}",
                            brief(ws)
                        ));
                    }
                },
                None => (None, None),
            };
            let args_doc = match st["args"].as_array() {
                Some(list) if !list.is_empty() && list.len() <= MAX_ARGS => list,
                _ => {
                    return refuse(format!(
                        "step #{si}: args must be a non-empty list of at most {MAX_ARGS} arguments"
                    ));
                }
            };
            let mut names_here: Vec<(String, String)> = Vec::with_capacity(args_doc.len());
            let mut args = Vec::with_capacity(args_doc.len());
            for (ai, a) in args_doc.iter().enumerate() {
                let a = keys(a, &format!("step #{si} args[{ai}]"), &["name", "binding"], &["view"])?;
                let name = match a["name"].as_str() {
                    Some(n) if is_ident(n) => n.to_owned(),
                    _ => return refuse(format!("step #{si} args[{ai}]: name must be an identifier")),
                };
                let Some(buffer) = a["binding"].as_str().and_then(|b| index.get(b).copied()) else {
                    return refuse(format!("step #{si} args[{ai}]: binding {} is not a buffer", brief(&a["binding"])));
                };
                let b = &buffers[buffer];
                if b.role == BufferRole::Workspace {
                    return refuse(format!(
                        "step #{si} args[{ai}]: the workspace is not an argument, it is the workspace field of a library step"
                    ));
                }
                names_here.push((name.clone(), b.name.clone()));
                let view = match a.get("view") {
                    None => None,
                    Some(v) => {
                        let v = keys(v, &format!("step #{si} args[{ai}].view"), &["shape", "strides", "offset"], &[])?;
                        let ints = |x: &Value, minimum: i64| -> Option<Vec<usize>> {
                            x.as_array()?
                                .iter()
                                .map(|e| int_of(e).filter(|n| *n >= minimum).map(|n| n as usize))
                                .collect()
                        };
                        let (shape, strides) = (ints(&v["shape"], 1), ints(&v["strides"], 0));
                        let offset = int_of(&v["offset"]).filter(|n| *n >= 0);
                        let (Some(shape), Some(strides), Some(offset)) = (shape, strides, offset) else {
                            return refuse(format!(
                                "step #{si} args[{ai}].view must be {{shape: positive ints, strides: non-negative ints (same length), offset: non-negative int}}"
                            ));
                        };
                        if shape.len() != strides.len() {
                            return refuse(format!(
                                "step #{si} args[{ai}].view must be {{shape: positive ints, strides: non-negative ints (same length), offset: non-negative int}}"
                            ));
                        }
                        let reach = shape
                            .iter()
                            .zip(&strides)
                            .fold(1u128, |acc, (n, s)| acc.saturating_add((*n as u128 - 1).saturating_mul(*s as u128)));
                        let elements = shapes[buffer]
                            .iter()
                            .try_fold(1u128, |acc, d| acc.checked_mul(*d as u128))
                            .unwrap_or(u128::MAX);
                        if (offset as u128).saturating_add(reach) > elements {
                            return refuse(format!(
                                "step #{si} args[{ai}]: the view reaches past the end of buffer '{}' at {}",
                                b.name,
                                describe_values(combo)
                            ));
                        }
                        if b.role == BufferRole::Devscalar {
                            return refuse(format!(
                                "step #{si} args[{ai}]: a devscalar is bound whole, without a view"
                            ));
                        }
                        Some(PlanView { shape, strides, offset: offset as usize })
                    }
                };
                args.push(PlanArg { name, binding: b.name.clone(), view, buffer });
            }
            let distinct: HashSet<&str> = names_here.iter().map(|(n, _)| n.as_str()).collect();
            if distinct.len() != names_here.len() {
                let all: Vec<&str> = names_here.iter().map(|(n, _)| n.as_str()).collect();
                return refuse(format!("step #{si}: the argument names must be distinct, got {all:?}"));
            }
            if kind == StepKind::Copy {
                if let Some(problem) = copy_problem(&args, &shapes, &buffers) {
                    return refuse(format!("step #{si} at {}: {problem}", describe_values(combo)));
                }
                let dst = &buffers[args[1].buffer];
                if !dst.role.is_writable() {
                    return refuse(format!(
                        "step #{si}: a copy step writes '{}', whose role is {}",
                        dst.name, dst.role
                    ));
                }
            }
            this.push((kind, workspace.clone(), names_here));
            if bi == 0 {
                for a in &args {
                    first.entry(a.buffer).or_insert(si);
                    last.insert(a.buffer, si);
                }
            }
            steps.push(PlanStep { entry, kind, args, workspace, workspace_buffer });
        }
        match &skeleton {
            None => skeleton = Some(this),
            Some(s) if *s != this => {
                return refuse(format!(
                    "bucket {} has other kinds, workspaces, argument names or bindings than the first bucket: only the entries and the views may differ",
                    describe_values(combo)
                ));
            }
            Some(_) => {}
        }
        buckets.push(PlanBucket { values: combo.clone(), steps, shapes });
    }
    let uses_library = buckets[0].steps.iter().any(|s| s.kind == StepKind::Library);
    if uses_library != buffers.iter().any(|b| b.role == BufferRole::Workspace) {
        return refuse("there is a buffer of role workspace exactly when some step has kind library");
    }

    // -- the arena: every region inside, overlaps only where the live ranges are disjoint
    let arena: Vec<usize> = (0..buffers.len()).filter(|i| !buffers[*i].role.is_external()).collect();
    let region_end = |i: usize| u128::from(buffers[i].offset.unwrap_or(0)) + align_up(u128::from(buffers[i].bytes));
    let end = arena.iter().map(|i| region_end(*i)).max().unwrap_or(0);
    let arena_bytes = match int_of(&top["arena_bytes"]) {
        Some(n) if n >= 0 && n as u128 == align_up(end) => n as u64,
        _ => {
            return refuse(format!(
                "arena_bytes must be the end of the last region rounded up to {ARENA_ALIGNMENT} ({}), got {}",
                align_up(end),
                brief(&top["arena_bytes"])
            ));
        }
    };
    // a sweep over the regions by offset (Python's, in O(n log n)): a region that may not share (reuse is false, or it is not an
    // intermediate that some step uses) must start with no active region and have none start under it; the ones that may share must have
    // pairwise disjoint live ranges, kept as a sorted list of ranges (disjoint, so a new range is compared with its two neighbours only)
    let mut order = arena.clone();
    order.sort_by_key(|i| (buffers[*i].offset.unwrap_or(0), buffers[*i].bytes)); // stable, as Python's sort
    let mut heap: BinaryHeap<Reverse<(u128, usize)>> = BinaryHeap::new();
    type Span = (usize, usize, String); // the first and the last step that use a buffer, and its name
    let mut active: HashMap<usize, (usize, Option<Span>)> = HashMap::new(); // seq -> (buffer, its live range if it shares)
    let mut lives: Vec<Span> = Vec::new();
    let mut exclusive: Option<usize> = None;
    let overlap = "overlap, which only reuse=true allows, and only for two intermediates that no step uses together";
    for (seq, &i) in order.iter().enumerate() {
        let start = u128::from(buffers[i].offset.unwrap_or(0));
        while let Some(Reverse((end, s))) = heap.peek().copied() {
            if end > start {
                break;
            }
            heap.pop();
            let (gone, span) = active.remove(&s).expect("every region on the heap is active");
            match span {
                Some(span) => {
                    let at = lives.binary_search(&span).expect("an active range is listed");
                    lives.remove(at);
                }
                None if exclusive == Some(gone) => exclusive = None,
                None => {}
            }
        }
        let name = &buffers[i].name;
        let shares = reuse && buffers[i].role == BufferRole::Intermediate && first.contains_key(&i);
        let span = if !shares {
            if let Some(Reverse((_, s))) = heap.peek() {
                return refuse(format!("the regions of '{name}' and '{}' {overlap}", buffers[active[s].0].name));
            }
            exclusive = Some(i);
            None
        } else {
            if let Some(other) = exclusive {
                return refuse(format!("the regions of '{name}' and '{}' {overlap}", buffers[other].name));
            }
            let span = (first[&i], last[&i], name.clone());
            let at = lives.binary_search(&span).unwrap_or_else(|at| at);
            let neighbours = [at.checked_sub(1).and_then(|k| lives.get(k)), lives.get(at)];
            for neighbour in neighbours.into_iter().flatten() {
                if neighbour.0 <= span.1 && span.0 <= neighbour.1 {
                    return refuse(format!("the regions of '{name}' and '{}' {overlap}", neighbour.2));
                }
            }
            lives.insert(at, span.clone());
            Some(span)
        };
        heap.push(Reverse((region_end(i), seq)));
        active.insert(seq, (i, span));
    }
    // (that an intermediate or an output is written before it is read needs the access of each pointer, which the manifest has:
    // `check_against_manifest`)
    Ok(Validated {
        version: version.to_owned(),
        graph,
        arch,
        manifest_digest,
        reuse,
        arena_bytes,
        dims,
        buffers,
        outputs,
        buckets,
    })
}

/// `(shape, strides, offset)` in elements of an argument of a step at a bucket: its view, or the whole buffer (Python's `copy_view`).
pub(crate) fn arg_layout(a: &PlanArg, shapes: &[Vec<usize>]) -> (Vec<usize>, Vec<usize>, usize) {
    match &a.view {
        Some(v) => (v.shape.clone(), v.strides.clone(), v.offset),
        None => {
            let shape = shapes[a.buffer].clone();
            let mut strides = vec![0usize; shape.len()];
            let mut acc = 1usize;
            for i in (0..shape.len()).rev() {
                strides[i] = acc;
                acc = acc.saturating_mul(shape[i]);
            }
            (shape, strides, 0)
        }
    }
}

/// Whether a tensor of this shape and these strides is one run of elements (the strides of its axes of a size other than 1 the row-major ones).
fn is_dense(shape: &[usize], strides: &[usize]) -> bool {
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

/// What is wrong with the arguments of a copy step at a bucket (Python's `copy_problem`): exactly `src` then `dst`, each dense, of one dtype and
/// one number of elements, and in one buffer not overlapping.
fn copy_problem(args: &[PlanArg], shapes: &[Vec<usize>], buffers: &[PlanBuffer]) -> Option<String> {
    let names: Vec<&str> = args.iter().map(|a| a.name.as_str()).collect();
    if names != ["src", "dst"] {
        return Some(format!("a copy step has the arguments [\"src\", \"dst\"], in that order, got {names:?}"));
    }
    let mut runs = Vec::with_capacity(2);
    for a in args {
        let (shape, strides, offset) = arg_layout(a, shapes);
        if !is_dense(&shape, &strides) {
            return Some(format!(
                "the {} of a copy is a view of '{}' with the shape {shape:?} and the strides {strides:?}, which is not dense: a copy step copies \
                 one run of bytes",
                a.name, a.binding
            ));
        }
        let count = shape.iter().fold(1u128, |acc, d| acc.saturating_mul(*d as u128));
        runs.push((buffers[a.buffer].dtype, count, offset as u128));
    }
    let ((sd, sn, so), (dd, dn, d_off)) = (runs[0], runs[1]);
    if sd != dd || sn != dn {
        return Some(format!(
            "a copy step copies {sn} {} elements into {dn} {} elements: the dtype and the number of elements must be the same",
            sd.name(),
            dd.name()
        ));
    }
    if args[0].buffer == args[1].buffer && so < d_off + dn && d_off < so + sn {
        return Some(format!("the source and the destination of a copy step overlap in '{}'", args[0].binding));
    }
    None
}

/// Every step of every bucket against the manifest (Python's `check_against_manifest`): the digest, the entry for the plan's architecture in
/// the section the step says, the pointers of its ABI against the arguments (count, names, dtype, rank, which buffers may be written, the
/// device scalars), the workspace of a library step, that a buffer is written before it is read, and that every output is written by a step of every bucket.
fn check_against_manifest(v: &Validated, manifest: &Manifest) -> Check<()> {
    if manifest.digest() != v.manifest_digest {
        return refuse(format!(
            "the plan was built against the manifest with digest {}, but the manifest in {:?} has digest {}: the plan is refused (export it \
             again, or restore the manifest it was built with)",
            v.manifest_digest,
            manifest.label(),
            manifest.digest()
        ));
    }
    let arch = v.arch.as_str();
    for bucket in &v.buckets {
        let mut written: HashSet<usize> = (0..v.buffers.len())
            .filter(|i| {
                matches!(
                    v.buffers[*i].role,
                    BufferRole::Input | BufferRole::Weight | BufferRole::Inout | BufferRole::Devscalar
                )
            })
            .collect();
        for (si, st) in bucket.steps.iter().enumerate() {
            if st.kind == StepKind::Copy {
                // no entry: it reads src and writes dst (`validate` held it to its rules)
                let (src, dst) = (st.args[0].buffer, st.args[1].buffer);
                if !written.contains(&src) {
                    return refuse(format!(
                        "step #{si} (copy) of bucket {}: argument 'src' reads '{}' before any step has written it",
                        describe_values(&bucket.values),
                        v.buffers[src].name
                    ));
                }
                written.insert(dst);
                continue;
            }
            let stem = st.entry.as_str();
            let at = format!("step #{si} ({stem}) of bucket {}", describe_values(&bucket.values));
            let pointers: Vec<(&str, &str, Access, usize)> = match st.kind {
                StepKind::Kernel => {
                    let Some(entry) = manifest.entry(stem, arch) else {
                        return refuse(format!("{at}: the manifest has no kernel '{stem}' for {arch}"));
                    };
                    pointers_of(&entry.params)
                }
                StepKind::Library => {
                    let Some(entry) = manifest.library_entry(stem, arch) else {
                        return refuse(format!("{at}: the manifest has no library entry '{stem}' for {arch}"));
                    };
                    let ws = &v.buffers[st.workspace_buffer.expect("a library step has a workspace")];
                    if ws.bytes < entry.workspace_bytes {
                        return refuse(format!(
                            "{at}: the workspace '{}' has {} bytes, but the entry needs {}",
                            ws.name, ws.bytes, entry.workspace_bytes
                        ));
                    }
                    pointers_of(&entry.params)
                }
                StepKind::Copy => unreachable!("handled above"),
            };
            if pointers.len() != st.args.len() {
                let names: Vec<&str> = pointers.iter().map(|p| p.0).collect();
                return refuse(format!(
                    "{at}: the binary takes {} pointers ({}), the step binds {}",
                    pointers.len(),
                    names.join(", "),
                    st.args.len()
                ));
            }
            for ((pname, pdtype, access, rank), a) in pointers.iter().zip(&st.args) {
                let buf = &v.buffers[a.buffer];
                if *pname != a.name {
                    return refuse(format!("{at}: argument '{}' is bound where the binary has '{pname}'", a.name));
                }
                if *pdtype != buf.dtype.name() {
                    return refuse(format!(
                        "{at}: argument '{}' is bound to '{}' ({}), but the binary takes {pdtype}",
                        a.name,
                        buf.name,
                        buf.dtype.name()
                    ));
                }
                let bound_rank = a.view.as_ref().map_or(buf.shape.len(), |view| view.shape.len());
                if bound_rank != *rank {
                    return refuse(format!(
                        "{at}: argument '{}' has rank {bound_rank}, but the binary takes rank {rank}",
                        a.name
                    ));
                }
                if (*rank == 0) != (buf.role == BufferRole::Devscalar) {
                    return refuse(format!(
                        "{at}: argument '{}': a device scalar (a pointer of shape []) is bound to the buffer of role devscalar, and only it",
                        a.name
                    ));
                }
                let writes = matches!(access, Access::Out | Access::InOut);
                if writes && !buf.role.is_writable() {
                    return refuse(format!(
                        "{at}: argument '{}' is written by the binary but bound to '{}', whose role is {}",
                        a.name, buf.name, buf.role
                    ));
                }
                if *access != Access::Out && !written.contains(&a.buffer) {
                    return refuse(format!(
                        "{at}: argument '{}' reads '{}' before any step has written it",
                        a.name, buf.name
                    ));
                }
                if writes {
                    written.insert(a.buffer);
                }
            }
        }
        // a result no step writes would be returned as whatever the caller's tensor held: every returned value is written by a step of every
        // bucket
        let unwritten = v
            .outputs
            .iter()
            .find(|o| v.buffers.iter().position(|b| b.name == **o).is_none_or(|i| !written.contains(&i)));
        if let Some(output) = unwritten {
            return refuse(format!(
                "bucket {}: no step writes the output '{output}' (every returned value is written by a step of every bucket)",
                describe_values(&bucket.values)
            ));
        }
    }
    Ok(())
}

/// `(name, dtype, access, rank)` of the pointer parameters of an ABI, in order.
fn pointers_of(params: &[crate::manifest::Param]) -> Vec<(&str, &str, Access, usize)> {
    params
        .iter()
        .filter_map(|p| match &p.kind {
            ParamKind::Pointer { dtype_name, access, shape, .. } => {
                Some((p.name.as_str(), dtype_name.as_str(), *access, shape.len()))
            }
            _ => None,
        })
        .collect()
}

/// Read `plan.json` through no symbolic link, a regular file of at most 256 MiB.
fn read_plan_file(path: &Path) -> Result<Vec<u8>> {
    let io = |source| Error::Io { path: path.display().to_string(), source };
    let meta = std::fs::symlink_metadata(path).map_err(io)?;
    if meta.file_type().is_symlink() {
        return Err(Error::Plan(format!(
            "{:?} is a symbolic link; refusing to read through it",
            path.display().to_string()
        )));
    }
    if !meta.is_file() {
        return Err(Error::Plan(format!("{:?} is not a regular file", path.display().to_string())));
    }
    if meta.len() > MAX_PLAN_BYTES {
        return Err(Error::Plan(format!(
            "the plan {:?} is {} bytes, more than the {MAX_PLAN_BYTES} a plan may have",
            path.display().to_string(),
            meta.len()
        )));
    }
    std::fs::read(path).map_err(io)
}

/// Steps 1 to 3 of section 3: strict JSON of the schema, the digest against the pin and against the `plan_digest` the file carries, every
/// rule of the format.
fn read_document(label: &str, bytes: &[u8], pin: Pin<'_>) -> Result<(Validated, String)> {
    if let Pin::Digest(d) = pin
        && !is_digest(d)
    {
        return Err(Error::BadPin(d.to_owned()));
    }
    if bytes.len() as u64 > MAX_PLAN_BYTES {
        return Err(Error::Plan(format!(
            "the plan {label:?} is {} bytes, more than the {MAX_PLAN_BYTES} a plan may have",
            bytes.len()
        )));
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|e| Error::Plan(format!("cannot read the plan {label:?}: it is not UTF-8 text: {e}")))?;
    let doc = parse_strict(text).map_err(|e| Error::Plan(format!("the plan {label:?} is not valid JSON: {e}")))?;
    integers_fit_a_double(&doc)?;
    if doc.get("schema").and_then(Value::as_str) != Some(PLAN_SCHEMA) {
        let schema = doc.get("schema").map_or_else(|| "None".to_owned(), brief);
        return Err(Error::Plan(format!(
            "{label:?} is not a tilefish plan of schema {PLAN_SCHEMA:?} (schema {schema})"
        )));
    }
    let found = plan_digest(&doc);
    if let Pin::Digest(pinned) = pin
        && found != pinned
    {
        return Err(Error::PlanPinMismatch { path: label.to_owned(), found, pinned: pinned.to_owned() });
    }
    if doc.get("plan_digest").and_then(Value::as_str) != Some(found.as_str()) {
        let carried = doc.get("plan_digest").map_or_else(|| "None".to_owned(), brief);
        return Err(Error::Plan(format!(
            "the plan in {label:?} was edited: its plan_digest says {carried}, but its content has digest {found}"
        )));
    }
    Ok((validate(&doc)?, found))
}

impl LaunchPlan {
    /// Load `<directory>/plan.json` (or the plan file itself) and the manifest next to it, and verify both (`docs/plan-schema-1.md`, section 3).
    ///
    /// `pin` is the trust anchor, as for a manifest: [`Pin::Digest`] with the plan's digest (what Python's `tilefish.runtime.export` returned,
    /// `Plan.digest`), and a plan with any other content is refused; or [`Pin::TrustDirectory`], which accepts whatever the directory holds (its
    /// digests are only as good as its permissions). Either way the file's own `plan_digest` must be its digest (an edit that did not rewrite it
    /// is refused), every rule of the format must hold, the manifest in the same directory must have the digest the plan names (it is loaded
    /// with that pin, every cubin verified), and every step must be what the manifest's entries say. A plan of another architecture than the
    /// device is refused when it is instantiated.
    pub fn load(path: impl AsRef<Path>, pin: Pin<'_>) -> Result<Self> {
        let path = path.as_ref();
        let (directory, file): (PathBuf, PathBuf) = if path.is_dir() {
            (path.to_owned(), path.join(PLAN_NAME))
        } else {
            (path.parent().map_or_else(|| PathBuf::from("."), Path::to_owned), path.to_owned())
        };
        let label = file.display().to_string();
        let bytes = read_plan_file(&file)?;
        let (validated, digest) = read_document(&label, &bytes, pin)?;
        let manifest = match Manifest::load(directory.join(MANIFEST_NAME), Pin::Digest(&validated.manifest_digest)) {
            Ok(m) => m,
            Err(Error::PinMismatch { directory, found, pinned }) => {
                return Err(bad(format!(
                    "the plan was built against the manifest with digest {pinned}, but the manifest in {directory:?} has digest {found}: the \
                     plan is refused (export it again, or restore the manifest it was built with)"
                )));
            }
            Err(other) => return Err(other),
        };
        Self::finish(label, validated, digest, manifest)
    }

    /// A plan whose `plan.json` is in memory, run with a manifest the caller loaded and verified (`Manifest::load` or
    /// `Manifest::from_embedded`): verified as [`LaunchPlan::load`] verifies a directory, the manifest's digest included.
    pub fn from_embedded(plan: &[u8], manifest: Manifest, pin: Pin<'_>) -> Result<Self> {
        let (validated, digest) = read_document("<embedded plan.json>", plan, pin)?;
        Self::finish("<embedded plan.json>".to_owned(), validated, digest, manifest)
    }

    fn finish(label: String, validated: Validated, digest: String, manifest: Manifest) -> Result<Self> {
        check_against_manifest(&validated, &manifest)?;
        let Validated { version, graph, arch, manifest_digest, reuse, arena_bytes, dims, buffers, outputs, buckets } =
            validated;
        Ok(Self {
            inner: Arc::new(Inner {
                label,
                version,
                graph,
                arch,
                manifest_digest,
                digest,
                reuse,
                arena_bytes,
                dims,
                buffers,
                outputs,
                buckets,
                manifest,
            }),
        })
    }

    /// The digest of the plan ([`plan_digest`]): what a pin is compared with.
    pub fn digest(&self) -> &str {
        &self.inner.digest
    }

    /// The digest of the manifest the plan was built against (the manifest of [`LaunchPlan::manifest`] has it).
    pub fn manifest_digest(&self) -> &str {
        &self.inner.manifest_digest
    }

    /// The name of the graph.
    pub fn graph(&self) -> &str {
        &self.inner.graph
    }

    /// The architecture the plan's entries are taken for (`sm_103`).
    pub fn arch(&self) -> &str {
        &self.inner.arch
    }

    /// The version of tilefish that wrote the file (informative: not digested).
    pub fn version(&self) -> &str {
        &self.inner.version
    }

    /// Where the plan was read from.
    pub fn label(&self) -> &str {
        &self.inner.label
    }

    /// Whether the memory plan lets intermediates whose live ranges are disjoint share bytes.
    pub fn reuse(&self) -> bool {
        self.inner.reuse
    }

    /// The bytes of the arena.
    pub fn arena_bytes(&self) -> u64 {
        self.inner.arena_bytes
    }

    /// The dimensions and the sizes planned for each (its buckets), in the order of the file.
    pub fn dims(&self) -> &[(String, Vec<u64>)] {
        &self.inner.dims
    }

    /// Every buffer, in the order of the file (the graph's parameters first, in the order of its signature).
    pub fn buffers(&self) -> &[PlanBuffer] {
        &self.inner.buffers
    }

    /// The buffer called `name`.
    pub fn buffer(&self, name: &str) -> Option<&PlanBuffer> {
        self.inner.buffers.iter().find(|b| b.name == name)
    }

    /// The returned values (buffers of role `output`), in the order the graph returns them: the graph's names of them
    /// (`@tilefish.graph(outputs=...)` in Python), or `out0`, `out1`, ...
    pub fn outputs(&self) -> &[String] {
        &self.inner.outputs
    }

    /// The buffer of the returned value a host calls `name`: its own name, or `out<i>` for the `i`-th returned value whatever its name (the name
    /// every returned value had before a graph could name them), when no buffer of the plan is called `out<i>` (Python's `output_buffer`). A
    /// launch takes an output by either name.
    pub fn output_buffer(&self, name: &str) -> Option<&str> {
        let outputs = &self.inner.outputs;
        if let Some(found) = outputs.iter().find(|o| *o == name) {
            return Some(found);
        }
        let digits = name.strip_prefix("out")?;
        let canonical = !digits.is_empty()
            && digits.bytes().all(|b| b.is_ascii_digit())
            && (digits == "0" || !digits.starts_with('0'));
        let index: usize = digits.parse().ok().filter(|_| canonical)?;
        if index < outputs.len() && self.buffer(name).is_none() { Some(outputs[index].as_str()) } else { None }
    }

    /// Every bucket combination and its steps, in the order of the file.
    pub fn buckets(&self) -> &[PlanBucket] {
        &self.inner.buckets
    }

    /// The bucket combination with exactly these values (one per dimension, in any order).
    pub fn bucket(&self, values: &[(&str, u64)]) -> Option<&PlanBucket> {
        self.bucket_index(values).map(|i| &self.inner.buckets[i])
    }

    pub(crate) fn bucket_index(&self, values: &[(&str, u64)]) -> Option<usize> {
        if values.len() != self.inner.dims.len() {
            return None;
        }
        self.inner.buckets.iter().position(|b| {
            b.values.iter().all(|(n, v)| values.iter().filter(|(m, _)| *m == n.as_str()).map(|(_, w)| *w).eq([*v]))
        })
    }

    /// The verified manifest the plan launches the entries of.
    pub fn manifest(&self) -> &Manifest {
        &self.inner.manifest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_digest_test_vectors_of_the_spec() {
        assert_eq!(
            plan_digest(&serde_json::json!({})),
            "345611870fcf5832cd7533b55c9bcb061b78c17c4f53635c9bcf8f7cffca6893"
        );
        let doc =
            serde_json::json!({"schema": "tilefish.plan.v1", "tilefish": "x", "graph": "g", "plan_digest": "ignored"});
        assert_eq!(plan_digest(&doc), "f59e1096fcfb737bcb745fc0a11258a25a1e2eb2fd3039b8958c777d4dadc71e");
    }

    #[test]
    fn combinations_vary_the_first_name_slowest_with_the_names_sorted() {
        let dims = vec![("M".to_owned(), vec![32, 64]), ("B".to_owned(), vec![1, 2])];
        let got: Vec<String> = combinations(&dims).iter().map(|c| describe_values(c)).collect();
        assert_eq!(
            got,
            ["{\"B\": 1, \"M\": 32}", "{\"B\": 1, \"M\": 64}", "{\"B\": 2, \"M\": 32}", "{\"B\": 2, \"M\": 64}"]
        );
        assert_eq!(combinations(&[]), vec![Vec::<(String, u64)>::new()]);
    }

    #[test]
    fn a_refusal_quotes_at_most_a_thousand_characters() {
        let e = bad("x".repeat(5000)).to_string();
        assert!(e.len() < 1100 && e.contains("5000 characters in all"), "{e}");
    }
}
