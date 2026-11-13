//! The `library` section of a manifest: operations that a vendor library runs, with no file. Today: a bf16 GEMM through cuBLASLt with
//! the algorithm pinned (`docs/manifest-schema-3.md`, section 9).
//!
//! A port of `validate_entry` of `python/src/tilefish/_library.py`. An entry has exactly the fields of [`ENTRY_KEYS`], each of the type it
//! needs, and the redundant ones must agree: the dtypes, the cuBLAS-level problem (the `T`/`N` transposes, the leading dimensions)
//! and the shapes of the operands follow from `impl`, `m`, `n` and `k`. `n` and `k` are multiples of 8, as the declaration of the operation
//! asks (`check_declaration`: every bf16 row starts on 16 bytes). The pinned algorithm is 64 bytes (128 lowercase hex digits) that
//! only the library that made them can judge: running one is [`crate::gemm`].

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::expr::{Symbols, parse_expr};
use crate::manifest::{
    ALIGNMENT, Access, DimInfo, MAX_ELEMENTS, Param, ParamKind, Precondition, ShapeDim, by_name, is_arch, is_digest,
    is_ident, is_stem, parse_dims, parse_shape, precondition_of, want_ident, want_int,
};
use crate::tensor::DType;

/// The `cublasComputeType_t` every entry says (`CUBLAS_COMPUTE_32F`: an f32 accumulator).
pub(crate) const COMPUTE_TYPE: &str = "CUBLAS_COMPUTE_32F";
/// The scale type every entry says (`CUDA_R_32F`: `alpha` and `beta` are f32).
pub(crate) const SCALE_TYPE: &str = "CUDA_R_32F";
const MAX_WORKSPACE_BYTES: u64 = 1 << 40;

/// Which routine runs an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibraryImpl {
    /// `cublaslt.gemm`: `out[M, N] = a[M, K] @ b[N, K]^T`, an f32 accumulator, one rounding to bf16.
    Gemm,
    /// `cublaslt.gemm_bias`: the same with `bias[N]` added in f32 before the rounding (`CUBLASLT_EPILOGUE_BIAS`).
    GemmBias,
}

impl LibraryImpl {
    /// The word a manifest uses (`"cublaslt.gemm"`, `"cublaslt.gemm_bias"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gemm => "cublaslt.gemm",
            Self::GemmBias => "cublaslt.gemm_bias",
        }
    }

    /// Whether the operation has a `bias` operand.
    pub fn has_bias(self) -> bool {
        self == Self::GemmBias
    }
}

/// One pinned library operation of a manifest, for one architecture: what the entry says (the operands, their dimensions and
/// preconditions, the cuBLASLt problem) and the 64 bytes of the algorithm cuBLASLt chose for it, with the version that chose them.
#[derive(Debug, Clone, PartialEq)]
pub struct LibraryEntry {
    /// The operator and its bucket values (`gemm.M64`); unique with `arch`, and across the kernels of the manifest too.
    pub stem: String,
    /// The operator.
    pub op: String,
    /// The architecture the algorithm was pinned for (`sm_103`): an algorithm id means other kernels on another architecture.
    pub arch: String,
    /// Which routine runs it.
    pub implementation: LibraryImpl,
    /// The operands in launch order: `a [M, K]`, `b [N, K]`, `bias [N]` (only for `gemm_bias`), `out [M, N]`, all bf16 pointers (`out` is written).
    pub params: Vec<Param>,
    /// The bucket value of each bucketed dimension, in the order of the stem.
    pub buckets: Vec<(String, u64)>,
    /// The named dimensions and what is known of each.
    pub dims: Vec<DimInfo>,
    /// What a launch must check: the 16-byte alignment of every pointer, the static sizes and the buckets.
    pub preconditions: Vec<Precondition>,
    /// The sizes of the problem, in tilefish's row-major terms: `a` is `[m, k]`, `b` is `[n, k]`, `out` is `[m, n]`.
    pub m: u64,
    /// See `m`.
    pub n: u64,
    /// See `m`.
    pub k: u64,
    /// The workspace limit the heuristic was asked under, **and the size of the workspace a launch provides**.
    pub workspace_bytes: u64,
    /// The 64 bytes of the `cublasLtMatmulAlgo_t`, in memory order (eight `uint64_t`, as the host stores them).
    pub algo: [u8; 64],
    /// `cublasLtGetVersion()` of the library that chose the algorithm (`120804` for 12.8.4): an algorithm id names other kernels in another
    /// version, so a host with another version refuses the entry.
    pub cublaslt_version: u64,
    /// The multiprocessor count (`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`) of the device the algorithm was pinned on: an algorithm is chosen for
    /// the multiprocessors it runs on (a MIG slice has fewer than the whole GPU, and a stream-K algorithm partitions its work by the count), so a
    /// device of another count refuses the entry.
    pub sm_count: u64,
    /// The sha256 of the canonical declaration the entry was made from.
    pub declaration_digest: String,
}

impl LibraryEntry {
    /// The operand parameter called `name`.
    pub fn param(&self, name: &str) -> Option<&Param> {
        self.params.iter().find(|p| p.name == name)
    }
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub(crate) struct RawLibrary {
    stem: String,
    op: String,
    arch: String,
    #[serde(rename = "impl")]
    implementation: String,
    abi: RawLibraryAbi,
    buckets: Map<String, Value>,
    dims: Map<String, Value>,
    preconditions: Vec<Value>,
    dtypes: Map<String, Value>,
    compute_type: String,
    scale_type: String,
    transa: String,
    transb: String,
    m: Value,
    n: Value,
    k: Value,
    lda: Value,
    ldb: Value,
    ldc: Value,
    ldd: Value,
    epilogue: String,
    workspace_bytes: Value,
    algo: String,
    cublaslt_version: Value,
    sm_count: Value,
    declaration_digest: String,
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
struct RawLibraryAbi {
    params: Vec<RawLibraryParam>,
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
struct RawLibraryParam {
    index: u64,
    name: String,
    kind: String,
    dtype: String,
    access: String,
    shape: Vec<Value>,
}

// each from a JSON object only, as Python reads them (`validate_entry`, `_keys`): not a list of the values by position
by_name! {
    RawLibrary => RawLibrary::deserialize, "a library entry";
    RawLibraryAbi => RawLibraryAbi::deserialize, "abi";
    RawLibraryParam => RawLibraryParam::deserialize, "a parameter of abi.params";
}

/// An integer field from 1 to `2**31 - 1` (a size, or a leading dimension, of cuBLASLt's `int32` world).
fn size_field(v: &Value, name: &str) -> std::result::Result<u64, String> {
    match v.as_u64() {
        Some(n) if (1..=MAX_ELEMENTS).contains(&n) => Ok(n),
        _ => Err(format!("{name} must be an int from 1 to {MAX_ELEMENTS}, got {v}")),
    }
}

/// The value of the dimension expression `text` when the bucketed dimensions have their bucket values.
fn eval_static(text: &str, buckets: &[(String, u64)], where_: &str) -> std::result::Result<i64, String> {
    let mut syms = Symbols::default();
    let mut values = Vec::with_capacity(buckets.len());
    for (name, value) in buckets {
        syms.intern(name)?;
        values.push(Some(i64::try_from(*value).map_err(|_| format!("{where_}: a bucket does not fit 64 bits"))?));
    }
    let expr = parse_expr(text, where_, &mut syms)?;
    values.resize(syms.len(), None);
    match expr.eval(&values)? {
        Some(v) => Ok(v),
        None => Err(format!("{where_}: {text:?} mentions a dimension that has no bucket")),
    }
}

impl LibraryEntry {
    /// Strict reading of one entry of the manifest's `library` section: every field must be present and exactly typed, and the redundant
    /// ones must agree. `index` is the position of the entry in the section, for the messages.
    pub(crate) fn from_raw(raw: RawLibrary, index: usize) -> Result<Self> {
        let who = format!("library entry #{index} ({:?})", raw.stem);
        Self::read(raw).map_err(|e| Error::manifest(format!("{who}: {e}")))
    }

    fn read(raw: RawLibrary) -> std::result::Result<Self, String> {
        if !is_stem(&raw.stem) {
            return Err(format!("stem must look like 'gemm' or 'gemm.M64', got {:?}", raw.stem));
        }
        if !is_ident(&raw.op) {
            return Err(format!("op must be an identifier, got {:?}", raw.op));
        }
        if !is_arch(&raw.arch) {
            return Err(format!("arch must look like 'sm_103', got {:?}", raw.arch));
        }
        let implementation = match raw.implementation.as_str() {
            "cublaslt.gemm" => LibraryImpl::Gemm,
            "cublaslt.gemm_bias" => LibraryImpl::GemmBias,
            other => {
                return Err(format!("impl must be one of [\"cublaslt.gemm\", \"cublaslt.gemm_bias\"], got {other:?}"));
            }
        };
        let bias = implementation.has_bias();
        let (m, n, k) = (size_field(&raw.m, "m")?, size_field(&raw.n, "n")?, size_field(&raw.k, "k")?);
        let ld = [
            size_field(&raw.lda, "lda")?,
            size_field(&raw.ldb, "ldb")?,
            size_field(&raw.ldc, "ldc")?,
            size_field(&raw.ldd, "ldd")?,
        ];
        if ld != [k, k, n, n] {
            return Err("the leading dimensions must be lda = ldb = k and ldc = ldd = n (cuBLAS sees b as its A, a as its B and the result as C and D)".to_owned());
        }
        // what Python's declaration asks of N and K (`check_declaration`; M may be any size): every row of a, b and the result starts on 16 bytes
        for (name, size) in [("n", n), ("k", k)] {
            if size % 8 != 0 {
                return Err(format!(
                    "{name} = {size} must be a multiple of 8: a, b and the result are bf16 rows of k and n elements, and every row starts \
                     on 16 bytes (a declaration of the GEMM gives N and K divisible_by=8, buckets that are multiples of 8, or such a static size)"
                ));
            }
        }
        if (raw.transa.as_str(), raw.transb.as_str()) != ("T", "N") {
            return Err(format!(
                "transa and transb must be \"T\" and \"N\", got {:?} and {:?}",
                raw.transa, raw.transb
            ));
        }
        if raw.compute_type != COMPUTE_TYPE || raw.scale_type != SCALE_TYPE {
            return Err(format!("compute_type and scale_type must be {COMPUTE_TYPE:?} and {SCALE_TYPE:?}"));
        }
        let want_epilogue = if bias { "bias" } else { "none" };
        if raw.epilogue != want_epilogue {
            return Err(format!(
                "the epilogue of {} must be {want_epilogue:?}, got {:?}",
                raw.implementation, raw.epilogue
            ));
        }
        let mut want_dtypes = Map::new();
        for name in ["a", "b"].into_iter().chain(bias.then_some("bias")).chain(["out"]) {
            want_dtypes.insert(name.to_owned(), Value::String("bf16".to_owned()));
        }
        if raw.dtypes != want_dtypes {
            return Err(format!("dtypes must be {}, got {}", Value::Object(want_dtypes), Value::Object(raw.dtypes)));
        }
        let workspace_bytes = match raw.workspace_bytes.as_u64() {
            Some(w) if (1..=MAX_WORKSPACE_BYTES).contains(&w) => w,
            _ => return Err(format!("workspace_bytes must be an int from 1 to 2**40, got {}", raw.workspace_bytes)),
        };
        let algo = decode_algo(&raw.algo)?;
        let cublaslt_version = match raw.cublaslt_version.as_u64() {
            Some(v) if v >= 1 => v,
            _ => return Err(format!("cublaslt_version must be a positive int, got {}", raw.cublaslt_version)),
        };
        // a JSON integer >= 1: not 30.0, not true, not "30" (`as_u64` is `None` for all three); no upper bound but the 64 bits of the reader
        let sm_count = match raw.sm_count.as_u64() {
            Some(v) if v >= 1 => v,
            _ => {
                return Err(format!(
                    "sm_count must be a positive int (the multiprocessors of the device the algorithm was pinned on), got {}",
                    raw.sm_count
                ));
            }
        };
        if !is_digest(&raw.declaration_digest) {
            return Err(format!(
                "declaration_digest must be 64 lowercase hex digits (a sha256), got {:?}",
                raw.declaration_digest
            ));
        }
        let mut buckets = Vec::with_capacity(raw.buckets.len());
        let mut stem = raw.op.clone();
        for (name, value) in &raw.buckets {
            want_ident(&Value::String(name.clone()), "bucket dimension name")?;
            let v = want_int(value, &format!("bucket value of {name}"), 1)?;
            stem.push_str(&format!(".{name}{v}"));
            buckets.push((name.clone(), v));
        }
        // the operands: a [M, K], b [N, K], [bias [N]], out [M, N], in this order
        let expected_shapes: Vec<Vec<u64>> =
            [vec![m, k], vec![n, k]].into_iter().chain(bias.then(|| vec![n])).chain([vec![m, n]]).collect();
        let params_raw = &raw.abi.params;
        if params_raw.len() != expected_shapes.len() {
            return Err(format!(
                "abi.params must list {} pointers ({} in this order)",
                expected_shapes.len(),
                if bias { "a, b, bias, out" } else { "a, b, out" }
            ));
        }
        let mut params = Vec::with_capacity(params_raw.len());
        for (i, (p, size)) in params_raw.iter().zip(&expected_shapes).enumerate() {
            want_ident(&Value::String(p.name.clone()), &format!("abi.params[{i}].name"))?;
            let last = i + 1 == params_raw.len();
            let want_access = if last { "out" } else { "in" };
            if p.index != i as u64 || p.kind != "pointer" || p.dtype != "bf16" || p.access != want_access {
                return Err(format!("abi.params[{i}] must be the bf16 pointer #{i} with access {want_access}"));
            }
            if params_raw[..i].iter().any(|q| q.name == p.name) {
                return Err(format!("the name {:?} is used twice", p.name));
            }
            if p.shape.len() != size.len() {
                return Err(format!("abi.params[{i}].shape must have {} entries", size.len()));
            }
            let shape = parse_shape(&p.name, &p.shape)?;
            for (axis, (entry, want)) in shape.iter().zip(size).enumerate() {
                let value = match entry {
                    ShapeDim::Static(v) => i128::from(*v),
                    ShapeDim::Expr(text) => {
                        i128::from(eval_static(text, &buckets, &format!("abi.params[{i}].shape[{axis}]"))?)
                    }
                };
                if value != i128::from(*want) {
                    return Err(format!(
                        "abi.params[{i}].shape[{axis}] is {entry} = {value}, but m, n and k say {want}"
                    ));
                }
            }
            params.push(Param {
                slot: i,
                index: i,
                name: p.name.clone(),
                kind: ParamKind::Pointer {
                    dtype_name: "bf16".to_owned(),
                    dtype: Some(DType::Bf16),
                    access: if last { Access::Out } else { Access::In },
                    shape,
                    view: None,
                    alias: None,
                },
            });
        }
        // the preconditions and the table of dimensions: as in a kernel entry
        let mut syms = Symbols::default();
        let mut preconditions = Vec::with_capacity(raw.preconditions.len());
        let mut aligned = vec![false; params.len()];
        let mut bucketed = Vec::new();
        for value in &raw.preconditions {
            let pre = precondition_of(value, &mut syms)?;
            match &pre {
                Precondition::Bucket { dim, value, buckets: all } => {
                    let mine = buckets.iter().find(|(d, _)| d == dim).map(|(_, v)| *v);
                    if bucketed.contains(dim) || mine != Some(*value) || !all.contains(value) {
                        return Err(format!("the bucket precondition for {dim} disagrees with buckets"));
                    }
                    bucketed.push(dim.clone());
                }
                Precondition::PointerAlignment { arg, index, bytes } => {
                    let named = params.get(*index).is_some_and(|p| p.name == *arg);
                    if !named || *bytes != ALIGNMENT || std::mem::replace(&mut aligned[*index], true) {
                        return Err(format!(
                            "pointer_alignment must be {ALIGNMENT} bytes, once per pointer, on the pointer it names"
                        ));
                    }
                }
                Precondition::StaticSize { arg, index, axis, value } => {
                    let shape = match params.get(*index) {
                        Some(p) if p.name == *arg => match &p.kind {
                            ParamKind::Pointer { shape, .. } => Some(shape),
                            _ => None,
                        },
                        _ => None,
                    };
                    if !shape.is_some_and(|s| matches!(s.get(*axis), Some(ShapeDim::Static(v)) if v == value)) {
                        return Err("static_size does not match the shape of the pointer it names".to_owned());
                    }
                }
                other => {
                    let kind = match other {
                        Precondition::Divisible { .. } => "divisible",
                        Precondition::MaxElements { .. } => "max_elements",
                        _ => "max_extent",
                    };
                    return Err(format!("a {kind} precondition does not belong in a library entry"));
                }
            }
            preconditions.push(pre);
        }
        if aligned.iter().any(|a| !a) {
            return Err("every pointer needs exactly one pointer_alignment precondition".to_owned());
        }
        if bucketed.len() != buckets.len() {
            return Err("every bucket needs exactly one bucket precondition".to_owned());
        }
        let dims = parse_dims(&raw.dims, &raw.buckets)?;
        if stem != raw.stem {
            return Err(format!("stem {:?} does not match op {:?} and buckets {:?}", raw.stem, raw.op, raw.buckets));
        }
        Ok(Self {
            stem: raw.stem,
            op: raw.op,
            arch: raw.arch,
            implementation,
            params,
            buckets,
            dims,
            preconditions,
            m,
            n,
            k,
            workspace_bytes,
            algo,
            cublaslt_version,
            sm_count,
            declaration_digest: raw.declaration_digest,
        })
    }
}

/// The 64 bytes of a pinned algorithm from its 128 lowercase hex digits (strict: nothing else).
fn decode_algo(text: &str) -> std::result::Result<[u8; 64], String> {
    let bad = || {
        format!(
            "algo must be 128 lowercase hex digits (the 64 bytes of a cublasLtMatmulAlgo_t), got {:?}",
            text.chars().take(40).collect::<String>()
        )
    };
    if text.len() != 128 || !text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(bad());
    }
    let mut out = [0u8; 64];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).map_err(|_| bad())?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_algorithm_is_exactly_128_lowercase_hex_digits() {
        let good = "0123456789abcdef".repeat(8);
        assert_eq!(decode_algo(&good).unwrap()[..4], [0x01, 0x23, 0x45, 0x67]);
        for bad in [good.to_uppercase(), good[1..].to_owned(), format!("{good}0"), "g".repeat(128), String::new()] {
            assert!(decode_algo(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_dimension_expression_is_evaluated_under_the_buckets_only() {
        let buckets = vec![("M".to_owned(), 64u64)];
        assert_eq!(eval_static("M", &buckets, "x").unwrap(), 64);
        assert_eq!(eval_static("3*M + 1", &buckets, "x").unwrap(), 193);
        assert_eq!(eval_static("cdiv(M, 48)", &buckets, "x").unwrap(), 2);
        assert!(eval_static("N", &buckets, "x").unwrap_err().contains("no bucket"));
        assert!(eval_static("M +", &buckets, "x").is_err());
    }
}
