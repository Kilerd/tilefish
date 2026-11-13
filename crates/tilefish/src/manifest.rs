//! Reading a manifest directory: the cubins, checked against what the manifest says.
//!
//! A port of `python/src/tilefish/_manifest.py` (`Manifest.load`) and of the schema checks of `_specvalid.py`. A
//! manifest is read as strict JSON (see [`crate::json`]), every field has the type it needs and no key is unknown, and
//! the entries are checked for being consistent with themselves: the ABI, the preconditions, the buckets, the grid. Then
//! each cubin is read, and refused unless its `sha256` (the file) and `sha256_code` (the machine code, see
//! [`crate::elf`]) are the manifest's, it has the code section of the manifest's symbol, and it does not use shared
//! memory the manifest does not account for.
//!
//! **Trust.** The digests sit next to the files they describe, so whoever can write the directory can rewrite both. The
//! digest of the whole manifest ([`Manifest::digest`], what Python's `Manifest.digest` and `export` return) is what a
//! caller holds somewhere it trusts and passes as [`Pin::Digest`]: a manifest with any other content is refused. Reading a
//! directory without a pin takes an explicit [`Pin::TrustDirectory`]; there is no default.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::elf::{Elf, sha256_hex};
use crate::error::{Error, PreconditionKind, Result};
use crate::expr::{Symbols, parse_expr, parse_grid};
use crate::json::{canonical, parse_strict};
use crate::kernel::Cached;
use crate::tensor::DType;

/// The only manifest schema this host reads. Python's `MANIFEST_SCHEMA`. Schema 2 records, for every pointer parameter, its declared
/// shape, and for every entry the constraints of its named dimensions: all that a host needs to check a launch on its own.
pub const MANIFEST_SCHEMA: u64 = 2;
/// The file a manifest directory has.
pub const MANIFEST_NAME: &str = "manifest.json";
/// Part of what is hashed: another canonical form must change it. Python's `DIGEST_TAG`.
const DIGEST_TAG: &[u8] = b"tilefish.manifest-digest.v2\0";
/// The limit of 32-bit offsets, the only bound a `max_elements` precondition carries.
const MAX_ELEMENTS: u64 = (1 << 31) - 1;
/// Every pointer of a binary is compiled as this many bytes aligned.
const ALIGNMENT: u64 = 16;
const NUM_WARPS: [u32; 6] = [1, 2, 4, 8, 16, 32];
const MAX_STAGES: u32 = 16;

/// What vouches for a manifest directory.
#[derive(Debug, Clone, Copy)]
pub enum Pin<'a> {
    /// The digest of the whole manifest (64 lowercase hex digits), held somewhere the caller trusts. A manifest with any
    /// other digest is refused. This is what production uses.
    Digest(&'a str),
    /// Trust whatever the directory holds: its digests are only as good as its permissions. For development.
    TrustDirectory,
}

/// How a pointer parameter is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Read only.
    In,
    /// Read and written.
    InOut,
    /// Written.
    Out,
}

impl Access {
    /// The word a manifest uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::InOut => "inout",
            Self::Out => "out",
        }
    }
}

/// One axis of a declared shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeDim {
    /// A size baked into the binary.
    Static(u64),
    /// A dimension expression over the named dimensions of the entry (`M`, `3072*M`, `cdiv(n, 16)`).
    Expr(String),
}

impl std::fmt::Display for ShapeDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static(n) => write!(f, "{n}"),
            Self::Expr(e) => f.write_str(e),
        }
    }
}

/// What the manifest says about one named dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimInfo {
    /// The name the shapes, the scalars and the grid use.
    pub name: String,
    /// The dimension is a multiple of this.
    pub divisible_by: Option<u64>,
    /// The buckets of a bucketed dimension.
    pub buckets: Option<Vec<u64>>,
    /// The bucket this binary is built for: the dimension must equal it exactly (there is no padding).
    pub bucket: Option<u64>,
}

/// What a launch parameter is.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamKind {
    /// A device pointer.
    Pointer {
        /// The element dtype as the manifest names it (`"bf16"`, `"e2m1"`, ...).
        dtype_name: String,
        /// The dtype, when a binary can be run on it (a packed format cannot).
        dtype: Option<DType>,
        /// How the kernel uses it.
        access: Access,
        /// The declared shape: one entry per axis. A launch binds the named dimensions from the sizes of the tensors it is given.
        shape: Vec<ShapeDim>,
    },
    /// A runtime `i32` that carries the value of a dimension expression.
    Scalar {
        /// The expression (`n`, `3072*M`).
        dim: String,
        /// The pointer parameter whose element count this scalar is, for a generated kernel.
        count_of: Option<String>,
    },
}

/// One non-constexpr parameter of the launch ABI.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    /// Position in `kernelParams`.
    pub slot: usize,
    /// Position among all the kernel's parameters, constexprs included.
    pub index: usize,
    /// The parameter's name.
    pub name: String,
    /// Pointer or scalar.
    pub kind: ParamKind,
}

/// A value compiled into the binary as `tl.constexpr`.
#[derive(Debug, Clone, PartialEq)]
pub struct Constexpr {
    /// Position among all the kernel's parameters.
    pub index: usize,
    /// The parameter's name.
    pub name: String,
    /// The value: a number, a bool, or `{"float": "inf" | "-inf" | "nan"}`.
    pub value: Value,
    /// The declared `const[dtype]`, if any.
    pub dtype: Option<String>,
}

/// A condition a host checks before it may launch the binary.
#[derive(Debug, Clone, PartialEq)]
pub enum Precondition {
    /// The pointer is aligned to `bytes`.
    PointerAlignment {
        /// The pointer parameter.
        arg: String,
        /// Its position among the kernel's parameters.
        index: usize,
        /// The alignment in bytes (always 16).
        bytes: u64,
    },
    /// The runtime scalar is a multiple of `by`.
    Divisible {
        /// The scalar parameter.
        arg: String,
        /// Its position among the kernel's parameters.
        index: usize,
        /// The dimension expression it carries.
        dim: String,
        /// The divisor.
        by: u64,
    },
    /// Axis `axis` of the tensor is `value`.
    StaticSize {
        /// The pointer parameter.
        arg: String,
        /// Its position among the kernel's parameters.
        index: usize,
        /// The axis.
        axis: usize,
        /// The size baked into the binary.
        value: u64,
    },
    /// Dimension `dim` is exactly `value` (one of `buckets`).
    Bucket {
        /// The bucketed dimension.
        dim: String,
        /// The bucket this binary is built for.
        value: u64,
        /// Every bucket of the operator.
        buckets: Vec<u64>,
    },
    /// The tensor has at most `value` elements (a generated kernel indexes with `i32`).
    MaxElements {
        /// The pointer parameter.
        arg: String,
        /// Its position among the kernel's parameters.
        index: usize,
        /// Its shape, with the bucket values in.
        shape: Vec<String>,
        /// The limit (`2**31 - 1`).
        value: u64,
    },
}

/// What the binary asks of the launch besides grid, block and shared memory.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchAttributes {
    /// Bytes of global scratch the kernel needs (a plain launch passes a null pointer).
    pub global_scratch_size: u64,
    /// Bytes of profile scratch.
    pub profile_scratch_size: u64,
    /// Programmatic dependent launch.
    pub launch_pdl: bool,
    /// Cluster dimensions.
    pub cluster_dims: [u32; 3],
    /// Number of CTAs.
    pub num_ctas: u32,
    /// A cooperative launch.
    pub cooperative: bool,
}

impl LaunchAttributes {
    /// The attributes a plain `cuLaunchKernel` does not set up, as `name=value` strings (Python's `replay_blockers`).
    pub fn blockers(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.global_scratch_size != 0 {
            out.push(format!("global_scratch_size={}", self.global_scratch_size));
        }
        if self.profile_scratch_size != 0 {
            out.push(format!("profile_scratch_size={}", self.profile_scratch_size));
        }
        if self.launch_pdl {
            out.push("launch_pdl=True".to_owned());
        }
        if self.cluster_dims != [1, 1, 1] || self.num_ctas != 1 {
            out.push(format!("cluster_dims={:?}, num_ctas={}", self.cluster_dims, self.num_ctas));
        }
        if self.cooperative {
            out.push("launch_cooperative_grid=True".to_owned());
        }
        out
    }
}

/// One binary of a manifest: its entry, and its cubin whose digests have been checked.
#[derive(Debug, Clone)]
pub struct Entry {
    /// The operator and its bucket values (`rowop.S32`); unique with `arch`.
    pub stem: String,
    /// The operator.
    pub op: String,
    /// The architecture the cubin is for (`sm_103`).
    pub arch: String,
    /// The `.target` of the PTX it was assembled from.
    pub ptx_target: String,
    /// The kernel's symbol in the cubin.
    pub symbol: String,
    /// The cubin's file name in the manifest directory.
    pub cubin_name: String,
    /// The non-constexpr parameters, in `kernelParams` order.
    pub params: Vec<Param>,
    /// The grid, an expression over dimension names (`(cdiv(n, 1024),)`).
    pub grid: String,
    /// The thread block.
    pub block: [u32; 3],
    /// The dynamic shared memory a launch passes, in bytes.
    pub shared: u32,
    /// Warps per block.
    pub num_warps: u32,
    /// Pipeline stages the kernel was compiled with.
    pub num_stages: u32,
    /// What the binary asks of the launch.
    pub launch: LaunchAttributes,
    /// The compiled-in constants.
    pub constexprs: Vec<Constexpr>,
    /// The bucket value of each bucketed dimension, in the order of the stem.
    pub buckets: Vec<(String, u64)>,
    /// The named dimensions of the signature and what is known of each.
    pub dims: Vec<DimInfo>,
    /// What a launch must check.
    pub preconditions: Vec<Precondition>,
    /// Whether FMA contraction was disabled at compile time, when the build said.
    pub enable_fp_fusion: Option<bool>,
    /// The sha256 of the declaration the binary was built from.
    pub declaration_digest: String,
    /// The sha256 of the cubin file.
    pub sha256: String,
    /// The sha256 of its machine code (see [`crate::elf`]).
    pub sha256_code: String,
    /// The Triton version that built it.
    pub triton: String,
    /// The `ptxas` that assembled it.
    pub ptxas: String,
    /// The sha256 of the Triton source of a generated kernel.
    pub source_sha256: Option<String>,
    /// The digest of the files of a hand-written kernel.
    pub source_digest: Option<String>,
    /// How `source_digest` was computed.
    pub source_digest_kind: Option<String>,
    /// The verified cubin.
    pub cubin: Arc<[u8]>,
}

impl Entry {
    /// The ABI entry of a pointer or scalar parameter by name.
    pub fn param(&self, name: &str) -> Option<&Param> {
        self.params.iter().find(|p| p.name == name)
    }
}

/// The binaries of a manifest directory, verified when it is loaded.
#[derive(Debug)]
pub struct Manifest {
    pub(crate) label: String,
    pub(crate) digest: String,
    pub(crate) entries: Vec<Entry>,
    pub(crate) kernels: Mutex<HashMap<(usize, usize), Cached>>,
}

// -- the digest -------------------------------------------------------------------------------------------------------

/// The sha256 (lowercase hex) of a manifest document: its schema and every kernel entry in full.
///
/// Entries are sorted by `(stem, arch)` and serialized as canonical JSON (see [`crate::json::canonical`]), so the
/// digest does not depend on the order of the entries or on how the file was formatted, and changes with any field of
/// any entry. The top-level `tilefish` version is not part of it. This is Python's `manifest_digest`.
pub fn manifest_digest(doc: &Value) -> std::result::Result<String, String> {
    let schema = doc.get("schema").ok_or("the manifest has no schema")?;
    let kernels = doc.get("kernels").and_then(Value::as_array).ok_or("the manifest has no list of kernels")?;
    let key = |e: &Value, field: &str| -> String { e.get(field).and_then(Value::as_str).unwrap_or("None").to_owned() };
    let mut sorted: Vec<&Value> = kernels.iter().collect();
    sorted.sort_by_key(|e| (key(e, "stem"), key(e, "arch")));
    let mut body = Map::new();
    body.insert("schema".to_owned(), schema.clone());
    body.insert("kernels".to_owned(), Value::Array(sorted.into_iter().cloned().collect()));
    let text = canonical(&Value::Object(body));
    let mut bytes = DIGEST_TAG.to_vec();
    bytes.extend_from_slice(text.as_bytes());
    Ok(sha256_hex(&bytes))
}

// -- the typed, strict reading of one entry --------------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDoc {
    schema: Value,
    #[serde(default)]
    #[allow(dead_code)]
    tilefish: Option<String>,
    kernels: Vec<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEntry {
    stem: String,
    op: String,
    arch: String,
    ptx_target: String,
    symbol: String,
    cubin: String,
    abi: RawAbi,
    grid: String,
    block: [u32; 3],
    shared: u32,
    num_warps: u32,
    num_stages: u32,
    launch: LaunchAttributes,
    constexprs: Vec<RawConstexpr>,
    buckets: Map<String, Value>,
    dims: Map<String, Value>,
    preconditions: Vec<Value>,
    options: RawOptions,
    declaration_digest: String,
    sha256: String,
    sha256_code: String,
    toolchain: RawToolchain,
    #[serde(default)]
    source_sha256: Option<String>,
    #[serde(default)]
    source_digest: Option<String>,
    #[serde(default)]
    source_digest_kind: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAbi {
    params: Vec<RawParam>,
    scratch: Vec<RawScratch>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawParam {
    slot: usize,
    index: usize,
    name: String,
    kind: String,
    #[serde(rename = "type")]
    ty: String,
    ptx_type: String,
    size: usize,
    #[serde(default)]
    dtype: Option<String>,
    #[serde(default)]
    access: Option<String>,
    #[serde(default)]
    shape: Option<Vec<Value>>,
    #[serde(default)]
    dim: Option<String>,
    #[serde(default)]
    count_of: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDim {
    #[serde(default)]
    divisible_by: Option<u64>,
    #[serde(default)]
    buckets: Option<Vec<u64>>,
    #[serde(default)]
    bucket: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawScratch {
    slot: usize,
    name: String,
    #[serde(rename = "type")]
    ty: String,
    ptx_type: String,
    size: usize,
    value: i64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConstexpr {
    index: usize,
    name: String,
    value: Value,
    #[serde(default)]
    dtype: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOptions {
    num_warps: u32,
    num_stages: u32,
    #[serde(default)]
    enable_fp_fusion: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawToolchain {
    triton: String,
    ptxas: String,
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_stem(s: &str) -> bool {
    !s.is_empty() && s.split('.').all(is_ident)
}

fn is_digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn is_arch(s: &str) -> bool {
    s.strip_prefix("sm_").is_some_and(|n| (2..=3).contains(&n.len()) && n.bytes().all(|b| b.is_ascii_digit()))
}

/// The Triton spelling of a pointer to each dtype a manifest may name (Python's `ALL_DTYPES`).
fn pointer_type_of(dtype: &str) -> Option<&'static str> {
    Some(match dtype {
        "e2m1" | "ue8m0" => "*u8",
        other => DType::from_name(other)?.pointer_type(),
    })
}

fn strictly<T>(r: std::result::Result<T, String>, who: &str) -> Result<T> {
    r.map_err(|e| Error::manifest(format!("{who}: {e}")))
}

fn want_int(v: &Value, what: &str, minimum: u64) -> std::result::Result<u64, String> {
    match v.as_u64() {
        Some(n) if n >= minimum => Ok(n),
        _ => Err(format!("{what} must be an int >= {minimum}, got {v}")),
    }
}

fn want_ident(v: &Value, what: &str) -> std::result::Result<String, String> {
    match v.as_str() {
        Some(s) if is_ident(s) => Ok(s.to_owned()),
        _ => Err(format!("{what} must be an identifier, got {v}")),
    }
}

fn want_keys(obj: &Map<String, Value>, what: &str, kind: &str, allowed: &[&str]) -> std::result::Result<(), String> {
    for k in obj.keys() {
        if k != "kind" && !allowed.contains(&k.as_str()) {
            return Err(format!("{what} has unknown fields [{k:?}]"));
        }
    }
    let missing: Vec<&&str> = allowed.iter().filter(|k| !obj.contains_key(**k)).collect();
    if missing.is_empty() { Ok(()) } else { Err(format!("{what} ({kind}) is missing {missing:?}")) }
}

fn precondition_of(v: &Value, syms: &mut Symbols) -> std::result::Result<Precondition, String> {
    let obj = v.as_object().ok_or_else(|| format!("a precondition must be an object, got {v}"))?;
    let kind = obj.get("kind").and_then(Value::as_str).unwrap_or("");
    let what = format!("{kind} precondition");
    let get = |k: &str| obj.get(k).unwrap_or(&Value::Null);
    let usize_of = |k: &str, min: u64| -> std::result::Result<usize, String> {
        usize::try_from(want_int(get(k), &format!("{what} {k}"), min)?).map_err(|_| format!("{what} {k} is too large"))
    };
    Ok(match kind {
        "pointer_alignment" => {
            want_keys(obj, &what, kind, &["arg", "index", "bytes"])?;
            Precondition::PointerAlignment {
                arg: want_ident(get("arg"), &format!("{what} arg"))?,
                index: usize_of("index", 0)?,
                bytes: want_int(get("bytes"), &format!("{what} bytes"), 1)?,
            }
        }
        "divisible" => {
            want_keys(obj, &what, kind, &["arg", "index", "dim", "by"])?;
            let dim = get("dim").as_str().ok_or_else(|| format!("{what} dim must be a string"))?.to_owned();
            parse_expr(&dim, "divisible precondition dim", syms)?;
            Precondition::Divisible {
                arg: want_ident(get("arg"), &format!("{what} arg"))?,
                index: usize_of("index", 0)?,
                dim,
                by: want_int(get("by"), &format!("{what} by"), 1)?,
            }
        }
        "static_size" => {
            want_keys(obj, &what, kind, &["arg", "index", "axis", "value"])?;
            Precondition::StaticSize {
                arg: want_ident(get("arg"), &format!("{what} arg"))?,
                index: usize_of("index", 0)?,
                axis: usize_of("axis", 0)?,
                value: want_int(get("value"), &format!("{what} value"), 1)?,
            }
        }
        "bucket" => {
            want_keys(obj, &what, kind, &["dim", "value", "buckets"])?;
            let buckets = get("buckets")
                .as_array()
                .and_then(|a| a.iter().map(|b| b.as_u64().filter(|n| *n >= 1)).collect::<Option<Vec<u64>>>())
                .filter(|b| !b.is_empty() && b.windows(2).all(|w| w[0] < w[1]))
                .ok_or_else(|| {
                    format!(
                        "bucket precondition buckets must be a sorted list of distinct positive ints, got {}",
                        get("buckets")
                    )
                })?;
            Precondition::Bucket {
                dim: want_ident(get("dim"), &format!("{what} dim"))?,
                value: want_int(get("value"), &format!("{what} value"), 1)?,
                buckets,
            }
        }
        "max_elements" => {
            want_keys(obj, &what, kind, &["arg", "index", "shape", "value"])?;
            let value = want_int(get("value"), &format!("{what} value"), 1)?;
            if value != MAX_ELEMENTS {
                return Err(format!(
                    "max_elements precondition value must be {MAX_ELEMENTS} (the limit of i32 offsets), got {value}"
                ));
            }
            let shape = get("shape")
                .as_array()
                .ok_or("max_elements precondition shape must be a list of dimension strings")?
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let text =
                        d.as_str().ok_or_else(|| format!("max_elements precondition shape[{i}] must be a string"))?;
                    parse_expr(text, &format!("max_elements precondition shape[{i}]"), syms)?;
                    Ok(text.to_owned())
                })
                .collect::<std::result::Result<Vec<_>, String>>()?;
            Precondition::MaxElements {
                arg: want_ident(get("arg"), &format!("{what} arg"))?,
                index: usize_of("index", 0)?,
                shape,
                value,
            }
        }
        other => {
            return Err(format!(
                "precondition kind must be one of [\"bucket\", \"divisible\", \"max_elements\", \"pointer_alignment\", \"static_size\"], got {other:?}"
            ));
        }
    })
}

/// The declared shape of a pointer: each axis a size (an int of at least 1) or a dimension expression.
fn parse_shape(name: &str, shape: &[Value]) -> std::result::Result<Vec<ShapeDim>, String> {
    shape
        .iter()
        .enumerate()
        .map(|(i, d)| match (d.as_u64(), d.as_str()) {
            (Some(n), _) if n >= 1 => Ok(ShapeDim::Static(n)),
            (_, Some(e)) => Ok(ShapeDim::Expr(e.to_owned())),
            _ => Err(format!(
                "parameter {name}: shape[{i}] must be an int size of at least 1 or a dimension expression, got {d}"
            )),
        })
        .collect()
}

/// The table of named dimensions, checked against the buckets of the entry.
fn parse_dims(dims: &Map<String, Value>, buckets: &Map<String, Value>) -> std::result::Result<Vec<DimInfo>, String> {
    let mut out = Vec::with_capacity(dims.len());
    for (name, value) in dims {
        want_ident(&Value::String(name.clone()), "dims name")?;
        let raw: RawDim = serde_json::from_value(value.clone()).map_err(|e| format!("dims.{name}: {e}"))?;
        if raw.divisible_by == Some(0) {
            return Err(format!("dims.{name}.divisible_by must be at least 1"));
        }
        if let Some(b) = &raw.buckets
            && (b.is_empty() || b.iter().any(|x| *x < 1) || !b.windows(2).all(|w| w[0] < w[1]))
        {
            return Err(format!("dims.{name}.buckets must be a sorted list of distinct positive ints, got {b:?}"));
        }
        let mine = buckets.get(name).and_then(Value::as_u64);
        match (raw.bucket, &raw.buckets) {
            (Some(b), Some(all)) if all.contains(&b) && mine == Some(b) => {}
            (None, _) if mine.is_none() => {}
            _ => return Err(format!("dims.{name}.bucket must be one of its buckets and the bucket of this entry")),
        }
        out.push(DimInfo {
            name: name.clone(),
            divisible_by: raw.divisible_by,
            buckets: raw.buckets,
            bucket: raw.bucket,
        });
    }
    if let Some(missing) = buckets.keys().find(|b| !dims.contains_key(*b)) {
        return Err(format!("every bucketed dimension needs its entry in dims: {missing} has none"));
    }
    Ok(out)
}

fn decode_scalar(v: &Value, what: &str) -> std::result::Result<(), String> {
    match v {
        Value::Bool(_) | Value::Number(_) => Ok(()),
        Value::Object(o)
            if o.len() == 1 && matches!(o.get("float").and_then(Value::as_str), Some("inf" | "-inf" | "nan")) =>
        {
            Ok(())
        }
        other => Err(format!("{what} must be a number, a bool or a tagged float, got {other}")),
    }
}

impl Entry {
    fn from_raw(raw: RawEntry, cubin: Arc<[u8]>) -> Result<Self> {
        let who = format!("'{}' ({})", raw.stem, raw.arch);
        let dims = strictly(parse_dims(&raw.dims, &raw.buckets), &who)?;
        let mut syms = Symbols::default();
        for d in &dims {
            strictly(syms.intern(&d.name), &who)?;
        }
        let mut params = Vec::with_capacity(raw.abi.params.len());
        for p in raw.abi.params {
            let kind = if p.kind == "pointer" {
                let dtype_name = p.dtype.unwrap_or_default();
                let access = match p.access.as_deref() {
                    Some("in") => Access::In,
                    Some("inout") => Access::InOut,
                    _ => Access::Out,
                };
                let shape = strictly(parse_shape(&p.name, p.shape.as_deref().unwrap_or_default()), &who)?;
                ParamKind::Pointer { dtype: DType::from_name(&dtype_name), dtype_name, access, shape }
            } else {
                ParamKind::Scalar { dim: p.dim.unwrap_or_default(), count_of: p.count_of }
            };
            params.push(Param { slot: p.slot, index: p.index, name: p.name, kind });
        }
        let mut preconditions = Vec::with_capacity(raw.preconditions.len());
        for p in &raw.preconditions {
            preconditions.push(strictly(precondition_of(p, &mut syms), &who)?);
        }
        let buckets = raw.buckets.iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0))).collect::<Vec<_>>();
        let entry = Self {
            stem: raw.stem,
            op: raw.op,
            arch: raw.arch,
            ptx_target: raw.ptx_target,
            symbol: raw.symbol,
            cubin_name: raw.cubin,
            params,
            grid: raw.grid,
            block: raw.block,
            shared: raw.shared,
            num_warps: raw.num_warps,
            num_stages: raw.num_stages,
            launch: raw.launch,
            constexprs: raw
                .constexprs
                .into_iter()
                .map(|c| Constexpr { index: c.index, name: c.name, value: c.value, dtype: c.dtype })
                .collect(),
            buckets,
            dims,
            preconditions,
            enable_fp_fusion: raw.options.enable_fp_fusion,
            declaration_digest: raw.declaration_digest,
            sha256: raw.sha256,
            sha256_code: raw.sha256_code,
            triton: raw.toolchain.triton,
            ptxas: raw.toolchain.ptxas,
            source_sha256: raw.source_sha256,
            source_digest: raw.source_digest,
            source_digest_kind: raw.source_digest_kind,
            cubin,
        };
        strictly(entry.check_consistent(&mut syms), &who)?;
        Ok(entry)
    }

    /// The checks that need no more than the raw fields.
    fn check_raw(raw: &RawEntry) -> std::result::Result<(), String> {
        if !is_stem(&raw.stem) {
            return Err(format!("stem must look like 'op' or 'op.S32', got {:?}", raw.stem));
        }
        if !is_ident(&raw.op) {
            return Err(format!("op must be an identifier, got {:?}", raw.op));
        }
        if !is_arch(&raw.arch) {
            return Err(format!("arch must look like 'sm_103', got {:?}", raw.arch));
        }
        if !is_ident(&raw.symbol) {
            return Err(format!("symbol must be an identifier, got {:?}", raw.symbol));
        }
        let name = raw.cubin.as_str();
        if name.is_empty() || name.starts_with('.') || name.contains(['/', '\\', '\0']) {
            return Err(format!("the cubin file name {name:?} must be a plain file name in the manifest directory"));
        }
        for (field, value) in [
            ("sha256", &raw.sha256),
            ("sha256_code", &raw.sha256_code),
            ("declaration_digest", &raw.declaration_digest),
        ] {
            if !is_digest(value) {
                return Err(format!("{field} must be 64 lowercase hex digits (a sha256), got {value:?}"));
            }
        }
        for (field, value) in [("source_sha256", &raw.source_sha256), ("source_digest", &raw.source_digest)] {
            if let Some(v) = value.as_ref().filter(|v| !is_digest(v)) {
                return Err(format!("{field} must be 64 lowercase hex digits (a sha256), got {v:?}"));
            }
        }
        if raw.source_digest.is_some() != raw.source_digest_kind.is_some() {
            return Err("source_digest and source_digest_kind come together".to_owned());
        }
        if !NUM_WARPS.contains(&raw.num_warps) {
            return Err(format!("num_warps must be one of {NUM_WARPS:?}, got {}", raw.num_warps));
        }
        if !(1..=MAX_STAGES).contains(&raw.num_stages) {
            return Err(format!("num_stages must be an int from 1 to {MAX_STAGES}, got {}", raw.num_stages));
        }
        if raw.block != [32 * raw.num_warps, 1, 1] {
            return Err(format!(
                "block must be [32 * num_warps, 1, 1] = {:?}, got {:?}",
                [32 * raw.num_warps, 1, 1],
                raw.block
            ));
        }
        if raw.options.num_warps != raw.num_warps || raw.options.num_stages != raw.num_stages {
            return Err("the compile options disagree with num_warps and num_stages".to_owned());
        }
        if raw.launch.cluster_dims.contains(&0) || raw.launch.num_ctas == 0 {
            return Err("the launch attributes have a zero cluster dimension or CTA count".to_owned());
        }
        let mut stem = raw.op.clone();
        for (dim, value) in &raw.buckets {
            want_ident(&Value::String(dim.clone()), "bucket dimension name")?;
            let n = want_int(value, &format!("bucket value of {dim}"), 1)?;
            stem.push_str(&format!(".{dim}{n}"));
        }
        if stem != raw.stem {
            return Err(format!("stem {:?} does not match op {:?} and buckets {:?}", raw.stem, raw.op, raw.buckets));
        }
        for c in &raw.constexprs {
            want_ident(&Value::String(c.name.clone()), "constexpr name")?;
            decode_scalar(&c.value, &format!("constexpr {} value", c.name))?;
            if let Some(d) = &c.dtype
                && pointer_type_of(d).is_none()
            {
                return Err(format!("constexpr {}: dtype {d:?} is not a tilefish dtype", c.name));
            }
        }
        parse_dims(&raw.dims, &raw.buckets)?;
        let n = raw.abi.params.len();
        for (position, p) in raw.abi.params.iter().enumerate() {
            want_ident(&Value::String(p.name.clone()), "parameter name")?;
            if p.slot != position {
                return Err(format!("parameter {} has slot {} but is at position {position}", p.name, p.slot));
            }
            match p.kind.as_str() {
                "pointer" => {
                    let dtype = p.dtype.as_deref().unwrap_or("");
                    if pointer_type_of(dtype) != Some(p.ty.as_str()) {
                        return Err(format!(
                            "parameter {}: pointer type {:?} does not match dtype {dtype:?}",
                            p.name, p.ty
                        ));
                    }
                    if !matches!(p.access.as_deref(), Some("in" | "inout" | "out")) {
                        return Err(format!(
                            "parameter {}: access must be in, inout or out, got {:?}",
                            p.name, p.access
                        ));
                    }
                    if p.dim.is_some() || p.count_of.is_some() || p.ptx_type != "u64" || p.size != 8 {
                        return Err(format!("parameter {} is not a well-formed 64-bit pointer", p.name));
                    }
                    let Some(shape) = &p.shape else {
                        return Err(format!("parameter {} has no shape: a pointer records its declared shape", p.name));
                    };
                    parse_shape(&p.name, shape)?;
                }
                "scalar" => {
                    if p.ty != "i32" || p.ptx_type != "u32" || p.size != 4 || p.dtype.is_some() || p.access.is_some() {
                        return Err(format!("parameter {}: a runtime scalar must be an i32 passed as a u32", p.name));
                    }
                    if p.dim.is_none() || p.shape.is_some() {
                        return Err(format!("parameter {} is a scalar: it has a dim and no shape", p.name));
                    }
                }
                other => return Err(format!("parameter {}: kind must be pointer or scalar, got {other:?}", p.name)),
            }
        }
        for (k, s) in raw.abi.scratch.iter().enumerate() {
            let name = ["global_scratch", "profile_scratch"].get(k).copied().unwrap_or("");
            if s.name != name || s.slot != n + k || s.ty != "u64" || s.ptx_type != "u64" || s.size != 8 || s.value != 0
            {
                return Err("the scratch slots must be global_scratch and profile_scratch, null 64-bit pointers after the arguments".to_owned());
            }
        }
        if raw.abi.scratch.len() != 2 {
            return Err("the ABI must end with the two scratch slots".to_owned());
        }
        // every parameter index once, parameters and constexprs together
        let mut indices: Vec<usize> =
            raw.abi.params.iter().map(|p| p.index).chain(raw.constexprs.iter().map(|c| c.index)).collect();
        indices.sort_unstable();
        if indices.iter().enumerate().any(|(i, v)| i != *v) {
            return Err("the parameters and the constexprs must have the indices 0, 1, 2, ... once each".to_owned());
        }
        if !raw.abi.params.windows(2).all(|w| w[0].index < w[1].index) {
            return Err("the parameters must be in declaration order".to_owned());
        }
        let mut names: Vec<&str> = raw
            .abi
            .params
            .iter()
            .map(|p| p.name.as_str())
            .chain(raw.constexprs.iter().map(|c| c.name.as_str()))
            .collect();
        names.sort_unstable();
        if names.windows(2).any(|w| w[0] == w[1]) {
            return Err("argument names must be unique".to_owned());
        }
        Ok(())
    }

    /// The checks of `validate_spec` that the manifest's own fields allow: what a launch relies on is all there.
    fn check_consistent(&self, syms: &mut Symbols) -> std::result::Result<(), String> {
        for p in &self.params {
            if let ParamKind::Pointer { shape, .. } = &p.kind {
                for (axis, d) in shape.iter().enumerate() {
                    if let ShapeDim::Expr(e) = d {
                        parse_expr(e, &format!("parameter {} shape[{axis}]", p.name), syms)?;
                    }
                }
            }
            if let ParamKind::Scalar { dim, count_of } = &p.kind {
                parse_expr(dim, &format!("parameter {} dim", p.name), syms)?;
                if let Some(target) = count_of {
                    match self.param(target).map(|t| &t.kind) {
                        Some(ParamKind::Pointer { .. }) => {}
                        _ => {
                            return Err(format!(
                                "parameter {} count_of {target:?} must name a pointer parameter",
                                p.name
                            ));
                        }
                    }
                }
            }
        }
        let mut aligned = HashSet::new();
        let mut divisible = HashSet::new();
        let mut limited = HashSet::new();
        let mut sized = HashMap::new();
        let mut bucketed = HashSet::new();
        for pre in &self.preconditions {
            match pre {
                Precondition::Bucket { dim, value, buckets } => {
                    let mine = self.buckets.iter().find(|(d, _)| d == dim).map(|(_, v)| *v);
                    let in_dims = self.dims.iter().find(|d| d.name == *dim).and_then(|d| d.bucket);
                    if mine != Some(*value)
                        || in_dims != Some(*value)
                        || !buckets.contains(value)
                        || !bucketed.insert(dim.clone())
                    {
                        return Err(format!("the bucket precondition for {dim} disagrees with the entry's buckets"));
                    }
                    continue;
                }
                Precondition::PointerAlignment { arg, index, .. }
                | Precondition::Divisible { arg, index, .. }
                | Precondition::StaticSize { arg, index, .. }
                | Precondition::MaxElements { arg, index, .. } => match self.param(arg) {
                    Some(p) if p.index == *index => {}
                    _ => return Err(format!("a precondition refers to {arg}@{index}, which is not an argument")),
                },
            }
            let kind_of = |name: &str| self.param(name).map(|p| &p.kind);
            match pre {
                Precondition::PointerAlignment { arg, bytes, .. } => {
                    if !matches!(kind_of(arg), Some(ParamKind::Pointer { .. }))
                        || *bytes != ALIGNMENT
                        || !aligned.insert(arg.clone())
                    {
                        return Err(format!(
                            "pointer_alignment must be {ALIGNMENT} bytes, once, on a pointer argument ({arg})"
                        ));
                    }
                }
                Precondition::Divisible { arg, dim, by, .. } => {
                    let own = match kind_of(arg) {
                        Some(ParamKind::Scalar { dim, .. }) => Some(dim),
                        _ => None,
                    };
                    if own != Some(dim) || by % ALIGNMENT != 0 || !divisible.insert(arg.clone()) {
                        return Err(format!(
                            "divisible must be by a multiple of {ALIGNMENT}, once, on a runtime scalar with its own dim ({arg})"
                        ));
                    }
                }
                Precondition::MaxElements { arg, .. } => {
                    if !matches!(kind_of(arg), Some(ParamKind::Pointer { .. })) || !limited.insert(arg.clone()) {
                        return Err(format!("max_elements must be once, on a pointer argument ({arg})"));
                    }
                }
                Precondition::StaticSize { arg, axis, value, .. } => {
                    if !matches!(kind_of(arg), Some(ParamKind::Pointer { .. }))
                        || sized.insert((arg.clone(), *axis), *value).is_some_and(|v| v != *value)
                    {
                        return Err(format!(
                            "static_size of {arg} axis {axis} is not on a pointer argument, or contradicts another"
                        ));
                    }
                }
                Precondition::Bucket { .. } => unreachable!("handled above"),
            }
        }
        if bucketed.len() != self.buckets.len() {
            return Err("every bucket needs exactly one bucket precondition".to_owned());
        }
        for p in &self.params {
            if matches!(p.kind, ParamKind::Pointer { .. }) && !aligned.contains(&p.name) {
                return Err(format!(
                    "pointer argument {} has no pointer_alignment precondition: every pointer of a binary is compiled as {ALIGNMENT}-byte aligned",
                    p.name
                ));
            }
        }
        parse_grid(&self.grid, syms)?;
        // a static_size is the size the shape has on that axis; the table of dimensions names every dimension in use
        for (arg, axis) in sized.keys() {
            let agrees = match self.param(arg).map(|p| &p.kind) {
                Some(ParamKind::Pointer { shape, .. }) => {
                    matches!(shape.get(*axis), Some(ShapeDim::Static(n)) if Some(n) == sized.get(&(arg.clone(), *axis)))
                }
                _ => false,
            };
            if !agrees {
                return Err(format!("static_size of {arg} axis {axis} is not the size the declared shape has there"));
            }
        }
        if syms.len() > self.dims.len() {
            let unnamed: Vec<&str> = (self.dims.len()..syms.len()).map(|i| syms.name(i)).collect();
            return Err(format!(
                "dims does not name {unnamed:?}, which the declared shapes, the scalars or the grid mention"
            ));
        }
        Ok(())
    }
}

// -- loading -------------------------------------------------------------------------------------------------------------

fn read_file(path: &Path) -> Result<Vec<u8>> {
    let io = |source| Error::Io { path: path.display().to_string(), source };
    let meta = std::fs::symlink_metadata(path).map_err(io)?;
    if meta.file_type().is_symlink() {
        return Err(Error::manifest(format!(
            "{:?} is a symbolic link; refusing to read through it",
            path.display().to_string()
        )));
    }
    if !meta.is_file() {
        return Err(Error::manifest(format!("{:?} is not a regular file", path.display().to_string())));
    }
    std::fs::read(path).map_err(io)
}

impl Manifest {
    /// Load `<directory>/manifest.json` (or the manifest file itself) and every cubin it names.
    ///
    /// Refuses a manifest that is not strict JSON of the schema `tilefish` writes, one whose digest is not the pinned one
    /// (before any cubin is read), a cubin that is a symbolic link or whose name is not a plain file name in the
    /// directory, and a cubin whose `sha256` or `sha256_code` is not the manifest's or that is not an ELF file with the
    /// code section of the manifest's symbol.
    pub fn load(path: impl AsRef<Path>, pin: Pin<'_>) -> Result<Self> {
        let path = path.as_ref();
        let (directory, manifest_path): (PathBuf, PathBuf) = if path.is_dir() {
            (path.to_owned(), path.join(MANIFEST_NAME))
        } else {
            (path.parent().map_or_else(|| PathBuf::from("."), Path::to_owned), path.to_owned())
        };
        let bytes = read_file(&manifest_path)?;
        Self::from_parts(&directory.display().to_string(), &bytes, pin, |name| read_file(&directory.join(name)))
    }

    /// A manifest whose files are in memory (what `include_bytes!` gives a generated module): `manifest` is the
    /// contents of `manifest.json` and `cubins` are `(file name, contents)` pairs. Verified exactly like [`Manifest::load`].
    pub fn from_embedded(manifest: &[u8], cubins: &[(&str, &[u8])], pin: Pin<'_>) -> Result<Self> {
        Self::from_parts("<embedded>", manifest, pin, |name| {
            cubins
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, bytes)| bytes.to_vec())
                .ok_or_else(|| Error::manifest(format!("the cubin {name:?} is not among the embedded files")))
        })
    }

    fn from_parts(
        label: &str,
        manifest: &[u8],
        pin: Pin<'_>,
        mut read_cubin: impl FnMut(&str) -> Result<Vec<u8>>,
    ) -> Result<Self> {
        if let Pin::Digest(d) = pin
            && !is_digest(d)
        {
            return Err(Error::BadPin(d.to_owned()));
        }
        let text = std::str::from_utf8(manifest)
            .map_err(|e| Error::manifest(format!("the manifest {label:?} is not UTF-8 text: {e}")))?;
        let doc = parse_strict(text)
            .map_err(|e| Error::manifest(format!("the manifest {label:?} is not strict JSON: {e}")))?;
        let raw: RawDoc = serde_json::from_value(doc.clone()).map_err(|e| {
            if doc.is_object() {
                Error::manifest(format!(
                    "the manifest {label:?} is not a tilefish manifest of schema {MANIFEST_SCHEMA}: {e}"
                ))
            } else {
                Error::manifest(format!("the manifest {label:?} is not a tilefish manifest: it is not a JSON object"))
            }
        })?;
        if let Some(older) = raw.schema.as_u64().filter(|n| (1..MANIFEST_SCHEMA).contains(n)) {
            return Err(Error::manifest(format!(
                "the manifest {label:?} is schema {older}, which an older tilefish wrote; this host reads schema {MANIFEST_SCHEMA}, where every \
                 pointer records its shape and every entry the constraints of its dimensions: export it again with a current tilefish"
            )));
        }
        if raw.schema.as_u64() != Some(MANIFEST_SCHEMA) {
            return Err(Error::manifest(format!(
                "the manifest {label:?} is not a tilefish manifest of schema {MANIFEST_SCHEMA} (schema {})",
                raw.schema
            )));
        }
        let mut raws = Vec::with_capacity(raw.kernels.len());
        for (i, value) in raw.kernels.iter().enumerate() {
            let entry: RawEntry = serde_json::from_value(value.clone()).map_err(|e| {
                Error::manifest(format!("manifest kernel {i} ({:?}): {e}", value.get("stem").and_then(Value::as_str)))
            })?;
            raws.push(entry);
        }
        for raw in &raws {
            // everything that needs no file: the names (a cubin name that leaves the directory is refused here, before any read)
            Entry::check_raw(raw).map_err(|e| Error::manifest(format!("'{}' ({}): {e}", raw.stem, raw.arch)))?;
        }
        let digest =
            manifest_digest(&doc).map_err(|e| Error::manifest(format!("the manifest cannot be digested: {e}")))?;
        if let Pin::Digest(pinned) = pin
            && digest != pinned
        {
            return Err(Error::PinMismatch { directory: label.to_owned(), found: digest, pinned: pinned.to_owned() });
        }
        let mut seen = HashSet::new();
        let mut entries = Vec::with_capacity(raws.len());
        for raw in raws {
            if !seen.insert((raw.stem.clone(), raw.arch.clone())) {
                return Err(Error::manifest(format!("the manifest lists '{}' for {} twice", raw.stem, raw.arch)));
            }
            let (stem, arch) = (raw.stem.clone(), raw.arch.clone());
            let cubin = verify_cubin(&raw, &stem, &arch, &mut read_cubin)?;
            entries.push(Entry::from_raw(raw, cubin.into())?);
        }
        Ok(Self { label: label.to_owned(), digest, entries, kernels: Mutex::new(HashMap::new()) })
    }

    /// The digest of the whole manifest (see [`manifest_digest`]): what a pin is compared with.
    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// Where the manifest was read from.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Every binary of the manifest, in file order.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The binary of `stem` for `arch` (`"sm_103"`).
    pub fn entry(&self, stem: &str, arch: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.stem == stem && e.arch == arch)
    }

    /// Refuse unless the digest is `pinned`: what a generated launcher does with the digest of the manifest it was
    /// generated from.
    pub fn require_digest(&self, pinned: &str) -> Result<()> {
        if !is_digest(pinned) {
            return Err(Error::BadPin(pinned.to_owned()));
        }
        if self.digest != pinned {
            return Err(Error::PinMismatch {
                directory: self.label.clone(),
                found: self.digest.clone(),
                pinned: pinned.to_owned(),
            });
        }
        Ok(())
    }

    /// The index of the entry for `name` (a stem, or an operator) with exactly these bucket values on `arch`.
    pub(crate) fn select(&self, name: &str, buckets: &[(&str, u64)], arch: &str) -> Result<usize> {
        let same = |entry: &Entry| {
            entry.buckets.len() == buckets.len()
                && buckets.iter().all(|(d, v)| entry.buckets.iter().any(|(ed, ev)| ed == d && ev == v))
        };
        let named: Vec<usize> =
            (0..self.entries.len()).filter(|i| self.entries[*i].stem == name || self.entries[*i].op == name).collect();
        if named.is_empty() {
            let mut have: Vec<&str> = self.entries.iter().map(|e| e.stem.as_str()).collect();
            have.sort_unstable();
            have.dedup();
            return Err(Error::NoBinary(format!(
                "the manifest in {:?} has no kernel '{name}' (it has {have:?})",
                self.label
            )));
        }
        let here: Vec<usize> = named.iter().copied().filter(|i| self.entries[*i].arch == arch).collect();
        if here.is_empty() {
            let mut archs: Vec<&str> = named.iter().map(|i| self.entries[*i].arch.as_str()).collect();
            archs.sort_unstable();
            archs.dedup();
            return Err(Error::NoBinary(format!(
                "the manifest in {:?} has no binary for '{name}' on {arch} (it has {})",
                self.label,
                archs.join(", ")
            )));
        }
        // a stem names its own buckets; an operator needs them given
        let found = here.iter().copied().find(|i| {
            let e = &self.entries[*i];
            if e.stem == name { buckets.is_empty() || same(e) } else { same(e) }
        });
        found.ok_or_else(|| {
            let mut wanted: Vec<&str> = here.iter().flat_map(|i| self.entries[*i].buckets.iter().map(|(d, _)| d.as_str())).collect();
            wanted.sort_unstable();
            wanted.dedup();
            let listing: Vec<String> = wanted
                .iter()
                .map(|d| {
                    let mut vs: Vec<u64> =
                        here.iter().filter_map(|i| self.entries[*i].buckets.iter().find(|(ed, _)| ed == d).map(|(_, v)| *v)).collect();
                    vs.sort_unstable();
                    vs.dedup();
                    format!("{d}: {vs:?}")
                })
                .collect();
            Error::Precondition {
                kind: PreconditionKind::Bucket,
                arg: wanted.first().map(|d| (*d).to_owned()),
                message: format!(
                    "no binary was built for {buckets:?}: a bucketed dimension must equal a bucket exactly, and the buckets are \
                     {{{}}}. There is no padding: pad or slice in the host to a bucket size",
                    listing.join(", ")
                ),
            }
        })
    }
}

fn verify_cubin(
    raw: &RawEntry,
    stem: &str,
    arch: &str,
    read_cubin: &mut impl FnMut(&str) -> Result<Vec<u8>>,
) -> Result<Vec<u8>> {
    let name = raw.cubin.as_str();
    let cubin = read_cubin(name).map_err(|e| match e {
        Error::Io { source, .. } => {
            Error::manifest(format!("'{stem}' ({arch}): cannot read the cubin {name:?}: {source}"))
        }
        other => other,
    })?;
    let found = sha256_hex(&cubin);
    if found != raw.sha256 {
        return Err(Error::manifest(format!(
            "'{stem}' ({arch}): the file {name:?} has sha256 {found}, but the manifest says {}; the cubin is not the one that was built",
            raw.sha256
        )));
    }
    let elf = Elf::parse(&cubin)
        .map_err(|e| Error::manifest(format!("'{stem}' ({arch}): {name:?} is not a valid cubin: {e}")))?;
    let code = elf.code_digest();
    if code != raw.sha256_code {
        return Err(Error::manifest(format!(
            "'{stem}' ({arch}): the machine code of {name:?} has sha256_code {code}, but the manifest says {}; the binary was changed after it was built",
            raw.sha256_code
        )));
    }
    if elf.section(&format!(".text.{}", raw.symbol)).is_none() {
        return Err(Error::manifest(format!("'{stem}' ({arch}): {name:?} has no code section '.text.{}'", raw.symbol)));
    }
    if raw.shared == 0 && elf.uses_shared(&raw.symbol) {
        return Err(Error::manifest(format!(
            "'{stem}' ({arch}): the kernel has a shared memory section ('.nv.shared.{}'), so it uses shared memory, but the manifest \
             says shared = 0; a launch without the dynamic shared memory it addresses would fault and poison the CUDA context",
            raw.symbol
        )));
    }
    Ok(cubin)
}
