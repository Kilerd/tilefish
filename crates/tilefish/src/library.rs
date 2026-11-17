//! The `library` section of a manifest: operations that a vendor library runs, with no file. Today: a GEMM of bf16, f16 or f32 operands
//! through cuBLASLt with the algorithm pinned (`docs/manifest-schema-3.md`, section 9).
//!
//! A port of `validate_entry` of `python/src/tilefish/_library.py`. An entry has exactly the fields of [`ENTRY_KEYS`], each of the type it
//! needs, and the redundant ones must agree: the dtypes are two operands of one type (`bf16`, `f16` or `f32`) and a result of their type or
//! `f32` (a bias has the result's type); `transa` and `transb` say the layout (`transa` `T`: `b` is `[n, k]`, `N`: `b` is `[k, n]`; `transb` `N`:
//! `a` is `[m, k]`, `T`: `a` is `[k, m]`), and the leading dimensions and the
//! shapes of the operands follow from it, `impl`, `m`, `n` and `k`, which may be any size (an algorithm that cannot take the leading
//! dimensions they make is refused by `cublasLtMatmulAlgoCheck` when it is loaded). The pinned algorithm is 64 bytes (128 lowercase hex digits) that
//! only the library that made them can judge: running one is [`crate::gemm`].
//!
//! An entry of `impl: "cublaslt.matmul"` (section 9.5) is a GEMM with a cuBLASLt epilogue ([`GemmEpilogue`]) and `beta` (`out = epilogue(acc +
//! c)` when it is 1): the fields of a GEMM's with `beta`, `contract` and `verification` added and an optional `canary` (Python's `MATMUL_ENTRY_KEYS`,
//! `_validate_matmul_fields`), bf16 or f16 operands and a result of their type, a `bias` exactly for the epilogues with one and a `c` exactly for
//! `beta` 1. `cublaslt.gemm` and `cublaslt.gemm_bias` are read as they always were.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::expr::{Symbols, parse_expr};
use crate::json::Members;
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
    /// `cublaslt.gemm`: `out[M, N] = a[M, K] @ b[N, K]^T` (`b[K, N]` for [`GemmLayout::Nn`]), an f32 accumulator, one rounding to the result's
    /// type.
    Gemm,
    /// `cublaslt.gemm_bias`: the same with `bias[N]` added in f32 before the rounding (`CUBLASLT_EPILOGUE_BIAS`).
    GemmBias,
    /// `cublaslt.matmul`: the same with a cuBLASLt epilogue ([`LibraryEntry::epilogue`]) on the f32 accumulator, and `c[M, N]` added to it first
    /// when [`LibraryEntry::beta`] is 1, before the one rounding (`docs/manifest-schema-3.md`, section 9.5).
    Matmul,
}

/// What cuBLASLt does to the f32 accumulator of a GEMM before the one rounding to the result's type (`epilogue` of an entry; Python's
/// `tilefish._cublaslt.EPILOGUES`). `cublaslt.gemm` has [`GemmEpilogue::None`] and `cublaslt.gemm_bias` [`GemmEpilogue::Bias`]; a
/// `cublaslt.matmul` has any of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmEpilogue {
    /// `"none"`: `CUBLASLT_EPILOGUE_DEFAULT` (1), the accumulator as it is.
    None,
    /// `"bias"`: `CUBLASLT_EPILOGUE_BIAS` (4), `acc + bias`.
    Bias,
    /// `"relu"`: `CUBLASLT_EPILOGUE_RELU` (2), `relu(acc)`.
    Relu,
    /// `"relu_bias"`: `CUBLASLT_EPILOGUE_RELU_BIAS` (6), `relu(acc + bias)`.
    ReluBias,
}

impl GemmEpilogue {
    /// Every epilogue, in the order of the manifest schema (section 9.5).
    pub const ALL: [Self; 4] = [Self::None, Self::Bias, Self::Relu, Self::ReluBias];

    /// The word an entry uses (`"none"`, `"bias"`, `"relu"`, `"relu_bias"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Bias => "bias",
            Self::Relu => "relu",
            Self::ReluBias => "relu_bias",
        }
    }

    /// The epilogue of a word, `None` for any other.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|e| e.as_str() == name)
    }

    /// Whether the epilogue reads a `bias` operand (`"bias"` and `"relu_bias"`).
    pub fn has_bias(self) -> bool {
        matches!(self, Self::Bias | Self::ReluBias)
    }

    /// The `cublasLtEpilogue_t` value the matmul descriptor is given (`CUBLASLT_MATMUL_DESC_EPILOGUE`).
    pub fn cublaslt_value(self) -> u32 {
        match self {
            Self::None => 1,
            Self::Bias => 4,
            Self::Relu => 2,
            Self::ReluBias => 6,
        }
    }

    /// The arithmetic on the f32 accumulator `acc`, before the one rounding, in the order the library adds (`c` before the bias), as an entry's
    /// `verification.formula` says it (Python's `tilefish._library.formula`): `"relu((acc + c) + bias)"`.
    pub fn formula(self, beta: u8) -> String {
        let mut x = if beta == 1 { "acc + c".to_owned() } else { "acc".to_owned() };
        if self.has_bias() {
            x = if beta == 1 { format!("({x}) + bias") } else { format!("{x} + bias") };
        }
        if matches!(self, Self::Relu | Self::ReluBias) { format!("relu({x})") } else { x }
    }
}

impl std::fmt::Display for GemmEpilogue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What the verification of a `cublaslt.matmul` pin found when it was made (`contract` of the entry, section 9.5): its bytes are the library's
/// under the pin either way (contract (iii) of RFC 0003).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulContract {
    /// `"measured-exact"`: every element of the verification equals the formula computed in f32 on the pin's own accumulator and rounded once.
    MeasuredExact,
    /// `"binary-defined"`: some do not; the bytes are what the pinned binary computes.
    BinaryDefined,
}

impl MatmulContract {
    /// The word an entry uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MeasuredExact => "measured-exact",
            Self::BinaryDefined => "binary-defined",
        }
    }
}

/// Where the accumulator the verification of a `cublaslt.matmul` pin compared with came from (`verification.accumulator`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulAccumulator {
    /// `"f32-form"`: the f32 output of the pin's configuration re-encoded with an f32 `D`.
    F32Form,
    /// `"plain-f32-pin"`: the f32 GEMM the same posture pins (cuBLASLt does not take the configuration with an f32 `D`).
    PlainF32Pin,
}

impl MatmulAccumulator {
    /// The word an entry uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32Form => "f32-form",
            Self::PlainF32Pin => "plain-f32-pin",
        }
    }
}

/// The record of the verification of a `cublaslt.matmul` pin (`verification` of the entry; Python's `tilefish._matmul.verify`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulVerification {
    /// The formula of the epilogue and `beta` ([`GemmEpilogue::formula`]): what the pin was compared with.
    pub formula: String,
    /// Where the accumulator of the reference came from.
    pub accumulator: MatmulAccumulator,
    /// The seeded operand sets the pin ran on (2).
    pub operand_sets: u64,
    /// The elements compared (`2 * m * n`).
    pub elements: u64,
    /// The elements whose bits differ from the reference (0 exactly when the contract is [`MatmulContract::MeasuredExact`]).
    pub differ: u64,
    /// The largest distance in units in the last place over the elements where both are finite (0 when none differs).
    pub max_ulp: u64,
}

/// The canary of a `cublaslt.matmul` entry (`canary`): the digest of the `out` its pin computed at export from operands made of `seed` on the
/// device (section 9.5, with the operands and the digest of section 9.4), which [`crate::Manifest::gemm`] computes again before the pin is used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulCanary {
    /// The seed of the operands (0 to 2^32 - 1; export records 1).
    pub seed: u64,
    /// 32 lowercase hex digits: the two sums of the digest of `out` (`tilefish::canary_digest` of its bytes).
    pub out_digest: String,
}

/// How `b` and `a` are stored, named by the transposes cuBLASLt is called with (`transa`, then `transb`; cuBLAS's A operand is tilefish's `b`,
/// its B operand tilefish's `a`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmLayout {
    /// `"TN"`: `b` is `[N, K]` (a weight of `torch.nn.Linear`), `out = a @ b^T`; `lda = k`.
    Tn,
    /// `"NN"`: `b` is `[K, N]`, `out = a @ b`; `lda = n`.
    Nn,
    /// `"TT"`: `b` is `[N, K]` and `a` is `[K, M]`, `out = a^T @ b^T`; `lda = k`, `ldb = m`.
    Tt,
    /// `"NT"`: `b` is `[K, N]` and `a` is `[K, M]`, `out = a^T @ b`; `lda = n`, `ldb = m`.
    Nt,
}

impl GemmLayout {
    /// The word an entry's `transa` and `transb` spell (`"TN"`, `"NN"`, `"TT"`, `"NT"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tn => "TN",
            Self::Nn => "NN",
            Self::Tt => "TT",
            Self::Nt => "NT",
        }
    }

    /// Whether `b` is stored `[N, K]` and used transposed (`transa = T`: the layouts TN and TT).
    pub fn b_transposed(self) -> bool {
        matches!(self, Self::Tn | Self::Tt)
    }

    /// Whether `a` is stored `[K, M]` and used transposed (`transb = T`: the layouts TT and NT).
    pub fn a_transposed(self) -> bool {
        matches!(self, Self::Tt | Self::Nt)
    }

    /// The leading dimension of cuBLAS's A operand (tilefish's `b`): `k` for `[N, K]`, `n` for `[K, N]`.
    pub fn lda(self, n: u64, k: u64) -> u64 {
        if self.b_transposed() { k } else { n }
    }

    /// The leading dimension of cuBLAS's B operand (tilefish's `a`): `k` for `[M, K]`, `m` for `[K, M]`.
    pub fn ldb(self, m: u64, k: u64) -> u64 {
        if self.a_transposed() { m } else { k }
    }

    /// The shape of `a`.
    pub fn a_shape(self, m: u64, k: u64) -> [u64; 2] {
        if self.a_transposed() { [k, m] } else { [m, k] }
    }

    /// The shape of `b`.
    pub fn b_shape(self, n: u64, k: u64) -> [u64; 2] {
        if self.b_transposed() { [n, k] } else { [k, n] }
    }

    /// The rule of the leading dimensions in the words a refusal uses (Python's `leading_dimensions_rule`).
    pub(crate) fn leading_dimensions_rule(self) -> String {
        if self == Self::Tn {
            return "the leading dimensions must be lda = ldb = k and ldc = ldd = n (cuBLAS sees b as its A, a as its B and the result as C \
                    and D)"
                .to_owned();
        }
        let (lda, b) = if self.b_transposed() { ("k", "b") } else { ("n", "b [k, n]") };
        let (ldb, a) = if self.a_transposed() { ("m", "a [k, m]") } else { ("k", "a") };
        format!(
            "the leading dimensions of the layout {} must be lda = {lda}, ldb = {ldb} and ldc = ldd = n (cuBLAS sees {b} as its A, {a} as its \
             B and the result as C and D)",
            self.as_str()
        )
    }
}

/// The types and the layout of a GEMM (Python's `tilefish._cublaslt.Form`): what its algorithm is encoded for (the types) and how the
/// library is called (the layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GemmForm {
    /// The type of `a` and `b` (bf16, f16 or f32).
    pub dtype: DType,
    /// The type of the result and of a bias (`dtype`, or f32; f32 for f32 operands).
    pub out: DType,
    pub layout: GemmLayout,
}

impl GemmForm {
    /// tilefish 0.0.5's only GEMM: bf16 in and out, TN.
    #[cfg(test)]
    pub const DEFAULT: Self = Self { dtype: DType::Bf16, out: DType::Bf16, layout: GemmLayout::Tn };

    /// The types, as a message says them: `"bf16"`, `"f16"`, `"f16 operands and an f32 result"` (Python's `Form.what`).
    pub fn what(&self) -> String {
        if self.out == self.dtype {
            self.dtype.name().to_owned()
        } else {
            format!("{} operands and an {} result", self.dtype.name(), self.out.name())
        }
    }

    /// The keyword of the opt-in to split-K partial sums in the output type, named by the operands' type (`allow_f16_reduction`,
    /// `allow_f32_reduction`).
    pub fn opt_in(&self) -> String {
        format!("allow_{}_reduction", self.dtype.name())
    }
}

/// How the heuristic was asked for an entry's algorithm (`heuristic`; absent means [`Heuristic::Tilefish`]). Provenance: a host runs the
/// pinned algorithm either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Heuristic {
    /// The default: a few results under the workspace limit, 16-byte alignment minimums.
    Tilefish,
    /// `"torch"`: one result under the workspace limit alone, as torch's `addmm` asks.
    Torch,
}

/// Where an entry's algorithm comes from (`algo_source`; absent means [`AlgoSource::Heuristic`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgoSource {
    /// The heuristic's first answer.
    Heuristic,
    /// `"imported"`: a table of the declaration gave it, and the heuristic was not asked.
    Imported,
}

impl LibraryImpl {
    /// The word a manifest uses (`"cublaslt.gemm"`, `"cublaslt.gemm_bias"`, `"cublaslt.matmul"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gemm => "cublaslt.gemm",
            Self::GemmBias => "cublaslt.gemm_bias",
            Self::Matmul => "cublaslt.matmul",
        }
    }

    /// Whether every operation of the impl has a `bias` operand (`cublaslt.gemm_bias`): a `cublaslt.matmul` has one by its epilogue
    /// ([`LibraryEntry::has_bias`]).
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
    /// The type of `a` and `b` (`dtypes.a`): [`DType::Bf16`], [`DType::F16`] or [`DType::F32`] (multiplied as f32, TF32 off: the crate
    /// refuses an algorithm that would not; `docs/manifest-schema-3.md`, section 9.2).
    pub dtype: DType,
    /// The type of the result and of a bias (`dtypes.out`): `dtype`, or [`DType::F32`] (f32 for f32 operands).
    pub out_dtype: DType,
    /// How `b` and `a` are stored (`transa`, `transb`).
    pub layout: GemmLayout,
    /// The operands in launch order: `a [M, K]` (`[K, M]` for [`GemmLayout::Tt`] and [`GemmLayout::Nt`]), `b [N, K]` (`[K, N]` for
    /// [`GemmLayout::Nn`] and [`GemmLayout::Nt`]), `bias [N]` (for `gemm_bias` and an epilogue with a bias), `c [M, N]` (for `beta` 1), `out [M,
    /// N]`, pointers of their dtypes (`out` is written).
    pub params: Vec<Param>,
    /// The bucket value of each bucketed dimension, in the order of the stem.
    pub buckets: Vec<(String, u64)>,
    /// The named dimensions and what is known of each, in the order of the file.
    pub dims: Vec<DimInfo>,
    /// What a launch must check: the 16-byte alignment of every pointer, the static sizes and the buckets.
    pub preconditions: Vec<Precondition>,
    /// The sizes of the problem, in tilefish's row-major terms: `a` is `[m, k]`, `b` is `[n, k]`, `out` is `[m, n]` (in the layout TN).
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
    /// The SM count the heuristic was asked to choose for (`sm_count_target`: `CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET`, from 8 to `sm_count`), for
    /// a GEMM run on a partition of the device's SMs; every check and run sets it too. `None`: the heuristic chose for the device's count.
    /// [`LibraryEntry::sm_count_chosen_for`] is the count either way.
    pub sm_count_target: Option<u64>,
    /// The sha256 of the canonical declaration the entry was made from.
    pub declaration_digest: String,
    /// How the heuristic was asked (`heuristic`, absent: [`Heuristic::Tilefish`]).
    pub heuristic: Heuristic,
    /// Where the algorithm comes from (`algo_source`, absent: [`AlgoSource::Heuristic`]).
    pub algo_source: AlgoSource,
    /// Whether the declaration opted in to split-K partial sums in bf16 (`allow_bf16_reduction: true`, only on bf16 operands; absent: no).
    /// Without it an algorithm whose split-K reduction scheme is `INPLACE` or `OUTPUT_TYPE` (partial sums in the output type, more than one
    /// rounding) is refused.
    pub allow_bf16_reduction: bool,
    /// The same opt-in of f16 operands (`allow_f16_reduction: true`, only on f16 operands; absent: no).
    pub allow_f16_reduction: bool,
    /// The same opt-in of f32 operands (`allow_f32_reduction: true`, only on f32 operands; absent: no): split-K may add its partial sums in the
    /// output (`INPLACE`, `OUTPUT_TYPE`), not only in a workspace of the compute type.
    pub allow_f32_reduction: bool,
    /// What cuBLASLt does to the accumulator (`epilogue`): [`GemmEpilogue::None`] for `gemm`, [`GemmEpilogue::Bias`] for `gemm_bias`, any for a
    /// `cublaslt.matmul`.
    pub epilogue: GemmEpilogue,
    /// `beta` of a `cublaslt.matmul` (0, or 1: `c [M, N]` is an operand and `out = epilogue(acc + c)`); 0 for `gemm` and `gemm_bias`.
    pub beta: u8,
    /// What the verification of a `cublaslt.matmul` pin found (`contract`); `None` for `gemm` and `gemm_bias`.
    pub contract: Option<MatmulContract>,
    /// The record of that verification (`verification`); `None` for `gemm` and `gemm_bias`.
    pub verification: Option<MatmulVerification>,
    /// The canary of a `cublaslt.matmul` (`canary`), which [`crate::Manifest::gemm`] runs before the pin is used; `None` without one.
    pub canary: Option<MatmulCanary>,
}

impl LibraryEntry {
    /// The operand parameter called `name`.
    pub fn param(&self, name: &str) -> Option<&Param> {
        self.params.iter().find(|p| p.name == name)
    }

    /// Whether the GEMM has a `bias` operand (`gemm_bias`, and a `cublaslt.matmul` whose epilogue has one).
    pub fn has_bias(&self) -> bool {
        self.epilogue.has_bias()
    }

    /// Whether the GEMM has a `c` operand (a `cublaslt.matmul` with `beta` 1).
    pub fn has_c(&self) -> bool {
        self.beta == 1
    }

    /// The canary's operands of the entry as `(name, tag, elements)`, in the order of its ABI: `a` (tag 0, `m * k` elements), `b` (tag 1, `n *
    /// k`), the bias (tag 2, `n`) and `c` (tag 3, `m * n`) when the entry has them, each dense in its layout, of the operands' type (Python's
    /// `tilefish._matmul.canary_elements`).
    pub fn canary_elements(&self) -> Vec<(&'static str, u8, u64)> {
        let mut out = vec![("a", 0, self.m * self.k), ("b", 1, self.n * self.k)];
        if self.has_bias() {
            out.push(("bias", 2, self.n));
        }
        if self.has_c() {
            out.push(("c", 3, self.m * self.n));
        }
        out
    }

    /// Whether split-K may add its partial sums in the output type: the opt-in of the operands' type.
    pub fn allows_output_type_reduction(&self) -> bool {
        self.allow_bf16_reduction || self.allow_f16_reduction || self.allow_f32_reduction
    }

    /// The SM count the algorithm was chosen for: the target, or the device's count.
    pub fn sm_count_chosen_for(&self) -> u64 {
        self.sm_count_target.unwrap_or(self.sm_count)
    }

    pub(crate) fn form(&self) -> GemmForm {
        GemmForm { dtype: self.dtype, out: self.out_dtype, layout: self.layout }
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
    pub(crate) buckets: Members,
    pub(crate) dims: Members,
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
    // present only when not the default, each with its one value (`docs/manifest-schema-3.md`, section 9); a JSON null is present (and refused)
    #[serde(default, deserialize_with = "present")]
    heuristic: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    algo_source: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    allow_bf16_reduction: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    allow_f16_reduction: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    allow_f32_reduction: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    sm_count_target: Option<Value>,
    // the fields of a `cublaslt.matmul` (section 9.5): `beta`, `contract` and `verification` required of it, `canary` optional, all four refused
    // on `gemm` and `gemm_bias` (`LibraryEntry::read` holds each impl to its list)
    #[serde(default, deserialize_with = "present")]
    beta: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    contract: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    verification: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    canary: Option<Value>,
}

/// The fields every `cublaslt.gemm` and `cublaslt.gemm_bias` entry has (Python's `ENTRY_KEYS`), in its order.
const ENTRY_KEYS: [&str; 26] = [
    "stem",
    "op",
    "arch",
    "impl",
    "abi",
    "buckets",
    "dims",
    "preconditions",
    "dtypes",
    "compute_type",
    "scale_type",
    "transa",
    "transb",
    "m",
    "n",
    "k",
    "lda",
    "ldb",
    "ldc",
    "ldd",
    "epilogue",
    "workspace_bytes",
    "algo",
    "cublaslt_version",
    "sm_count",
    "declaration_digest",
];
/// The fields of a `cublaslt.matmul` entry besides those of [`ENTRY_KEYS`] (Python's `MATMUL_ENTRY_KEYS`).
const MATMUL_KEYS: [&str; 3] = ["beta", "contract", "verification"];
/// The optional fields of a `gemm` or `gemm_bias` entry (Python's `OPTIONAL_ENTRY_FIELDS` and `OPTIONAL_COUNT_FIELDS`).
const OPTIONAL_KEYS: [&str; 6] = [
    "heuristic",
    "algo_source",
    "allow_bf16_reduction",
    "allow_f16_reduction",
    "allow_f32_reduction",
    "sm_count_target",
];
/// The optional fields of a `cublaslt.matmul` entry (Python's `MATMUL_OPTIONAL_FIELDS`): not `algo_source` (it imports no algorithm), and a canary.
const MATMUL_OPTIONAL_KEYS: [&str; 6] =
    ["heuristic", "allow_bf16_reduction", "allow_f16_reduction", "allow_f32_reduction", "sm_count_target", "canary"];
/// The fields of a `verification` record, in Python's order (`VERIFICATION_KEYS`).
const VERIFICATION_KEYS: [&str; 6] = ["formula", "accumulator", "operand_sets", "elements", "differ", "max_ulp"];
const MAX_CANARY_SEED: u64 = (1 << 32) - 1;

/// A field that is there, whatever its value: `null` included (serde's `Option` would read a `null` as an absent field, which Python does not).
pub(crate) fn present<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<Value>, D::Error> {
    Value::deserialize(deserializer).map(Some)
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

/// An optional field: absent (`None`) is the default, and present it must be exactly `only` (`true` is not `1`, `"torch"` not `"Torch"`).
fn optional_field(value: &Option<Value>, name: &str, only: &Value) -> std::result::Result<bool, String> {
    match value {
        None => Ok(false),
        Some(v) if v == only => Ok(true),
        Some(v) => Err(format!("{name} is absent (the default) or {only}, got {v}")),
    }
}

/// The value of the dimension expression `text` when the bucketed dimensions have their bucket values.
pub(crate) fn eval_static(text: &str, buckets: &[(String, u64)], where_: &str) -> std::result::Result<i64, String> {
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
            "cublaslt.matmul" => LibraryImpl::Matmul,
            other => {
                return Err(format!(
                    "impl must be one of [\"cublaslt.gemm\", \"cublaslt.gemm_bias\", \"cublaslt.matmul\"], got {other:?}"
                ));
            }
        };
        let matmul = implementation == LibraryImpl::Matmul;
        fields_of(&raw, matmul)?;
        // the epilogue: the impl's for gemm and gemm_bias, any of the four for a cublaslt.matmul, with its beta, contract, verification and canary
        let (epilogue, beta, contract, verification, canary) = if matmul {
            matmul_fields(&raw)?
        } else {
            let want = if implementation.has_bias() { GemmEpilogue::Bias } else { GemmEpilogue::None };
            if raw.epilogue != want.as_str() {
                return Err(format!(
                    "the epilogue of {} must be {:?}, got {:?}",
                    raw.implementation,
                    want.as_str(),
                    raw.epilogue
                ));
            }
            (want, 0, None, None, None)
        };
        let bias = epilogue.has_bias();
        let with_c = beta == 1;
        let (m, n, k) = (size_field(&raw.m, "m")?, size_field(&raw.n, "n")?, size_field(&raw.k, "k")?);
        let ld = [
            size_field(&raw.lda, "lda")?,
            size_field(&raw.ldb, "ldb")?,
            size_field(&raw.ldc, "ldc")?,
            size_field(&raw.ldd, "ldd")?,
        ];
        let layout = match (raw.transa.as_str(), raw.transb.as_str()) {
            ("T", "N") => GemmLayout::Tn,
            ("N", "N") => GemmLayout::Nn,
            ("T", "T") => GemmLayout::Tt,
            ("N", "T") => GemmLayout::Nt,
            _ => {
                return Err(format!(
                    "transa and transb must each be \"T\" or \"N\" (transa \"T\": b is [n, k], \"N\": b is [k, n]; transb \"N\": a is [m, k], \
                     \"T\": a is [k, m]), got {:?} and {:?}",
                    raw.transa, raw.transb
                ));
            }
        };
        if ld != [layout.lda(n, k), layout.ldb(m, k), n, n] {
            return Err(layout.leading_dimensions_rule());
        }
        if raw.compute_type != COMPUTE_TYPE || raw.scale_type != SCALE_TYPE {
            return Err(format!("compute_type and scale_type must be {COMPUTE_TYPE:?} and {SCALE_TYPE:?}"));
        }
        // a and b of one type, bf16, f16 or f32; the result (and a bias) of theirs or f32; anything else is held to the default form in the message.
        // A cublaslt.matmul: bf16 or f16, and the result, a bias and c of their type.
        let named = |key: &str| raw.dtypes.get(key).and_then(Value::as_str).and_then(DType::from_name);
        let (dtype, out_dtype) = match (named("a"), named("out")) {
            (Some(a @ (DType::Bf16 | DType::F16)), Some(out)) if matmul && out == a => (a, out),
            (Some(a @ (DType::Bf16 | DType::F16 | DType::F32)), Some(out))
                if !matmul && (out == a || out == DType::F32) =>
            {
                (a, out)
            }
            _ => (DType::Bf16, DType::Bf16),
        };
        let mut want_dtypes = Map::new();
        for (name, dt) in [("a", dtype), ("b", dtype)]
            .into_iter()
            .chain(bias.then_some(("bias", out_dtype)))
            .chain(with_c.then_some(("c", out_dtype)))
            .chain([("out", out_dtype)])
        {
            want_dtypes.insert(name.to_owned(), Value::String(dt.name().to_owned()));
        }
        if raw.dtypes != want_dtypes {
            let why = if matmul {
                "a and b are both bf16 or both f16, the result, a bias and c are of their type"
            } else {
                "a and b are both bf16, both f16 or both f32, the result is of their type or f32, a bias of the result's type"
            };
            return Err(format!(
                "dtypes must be {}, got {} ({why})",
                Value::Object(want_dtypes),
                Value::Object(raw.dtypes)
            ));
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
        // absent: the heuristic chose for the device's count; present: an integer from 8 (cuBLASLt kills the process below) to the device's
        let sm_count_target = match &raw.sm_count_target {
            None => None,
            Some(v) => match v.as_u64() {
                Some(t) if (8..=sm_count).contains(&t) => Some(t),
                _ => {
                    return Err(format!(
                        "sm_count_target is absent or an int from 8 to sm_count ({sm_count}), got {v}"
                    ));
                }
            },
        };
        if !is_digest(&raw.declaration_digest) {
            return Err(format!(
                "declaration_digest must be 64 lowercase hex digits (a sha256), got {:?}",
                raw.declaration_digest
            ));
        }
        let heuristic = if optional_field(&raw.heuristic, "heuristic", &Value::from("torch"))? {
            Heuristic::Torch
        } else {
            Heuristic::Tilefish
        };
        let algo_source = if optional_field(&raw.algo_source, "algo_source", &Value::from("imported"))? {
            AlgoSource::Imported
        } else {
            AlgoSource::Heuristic
        };
        let allow_bf16_reduction =
            optional_field(&raw.allow_bf16_reduction, "allow_bf16_reduction", &Value::Bool(true))?;
        let allow_f16_reduction = optional_field(&raw.allow_f16_reduction, "allow_f16_reduction", &Value::Bool(true))?;
        let allow_f32_reduction = optional_field(&raw.allow_f32_reduction, "allow_f32_reduction", &Value::Bool(true))?;
        // the opt-in is named by the operands' type, as torch's flags are
        for (flag, set, of) in [
            ("allow_bf16_reduction", allow_bf16_reduction, DType::Bf16),
            ("allow_f16_reduction", allow_f16_reduction, DType::F16),
            ("allow_f32_reduction", allow_f32_reduction, DType::F32),
        ] {
            if set && dtype != of {
                return Err(format!(
                    "{flag} is the opt-in of {} operands, and the operands of this entry are {}",
                    of.name(),
                    dtype.name()
                ));
            }
        }
        let mut buckets = Vec::with_capacity(raw.buckets.len());
        let mut stem = raw.op.clone();
        for (name, value) in &raw.buckets {
            want_ident(&Value::String(name.clone()), "bucket dimension name")?;
            let v = want_int(value, &format!("bucket value of {name}"), 1)?;
            stem.push_str(&format!(".{name}{v}"));
            buckets.push((name.clone(), v));
        }
        // the operands: a [M, K] (TN, NN) or [K, M] (TT, NT), b [N, K] (TN, TT) or [K, N] (NN, NT), [bias [N]], [c [M, N]], out [M, N], in this
        // order, each of its type in dtypes
        let expected: Vec<(Vec<u64>, DType)> =
            [(layout.a_shape(m, k).to_vec(), dtype), (layout.b_shape(n, k).to_vec(), dtype)]
                .into_iter()
                .chain(bias.then(|| (vec![n], out_dtype)))
                .chain(with_c.then(|| (vec![m, n], out_dtype)))
                .chain([(vec![m, n], out_dtype)])
                .collect();
        let params_raw = &raw.abi.params;
        if params_raw.len() != expected.len() {
            let names: Vec<&str> = ["a", "b"]
                .into_iter()
                .chain(bias.then_some("bias"))
                .chain(with_c.then_some("c"))
                .chain(["out"])
                .collect();
            return Err(format!(
                "abi.params must list {} pointers ({} in this order)",
                expected.len(),
                names.join(", ")
            ));
        }
        let mut params = Vec::with_capacity(params_raw.len());
        for (i, (p, (size, pdtype))) in params_raw.iter().zip(&expected).enumerate() {
            want_ident(&Value::String(p.name.clone()), &format!("abi.params[{i}].name"))?;
            let last = i + 1 == params_raw.len();
            let want_access = if last { "out" } else { "in" };
            if p.index != i as u64 || p.kind != "pointer" || p.dtype != pdtype.name() || p.access != want_access {
                return Err(format!(
                    "abi.params[{i}] must be the {} pointer #{i} with access {want_access}",
                    pdtype.name()
                ));
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
                    dtype_name: pdtype.name().to_owned(),
                    dtype: Some(*pdtype),
                    access: if last { Access::Out } else { Access::In },
                    shape,
                    view: None,
                    alias: None,
                    tensor_map: None,
                    template: None,
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
            dtype,
            out_dtype,
            layout,
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
            sm_count_target,
            declaration_digest: raw.declaration_digest,
            heuristic,
            algo_source,
            allow_bf16_reduction,
            allow_f16_reduction,
            allow_f32_reduction,
            epilogue,
            beta,
            contract,
            verification,
            canary,
        })
    }
}

/// Hold an entry to the field list of its impl (Python's `ENTRY_KEYS` and `OPTIONAL_ENTRY_FIELDS` for `gemm` and `gemm_bias`, `MATMUL_ENTRY_KEYS` and
/// `MATMUL_OPTIONAL_FIELDS` for `cublaslt.matmul`), in the words of Python's refusal: serde has refused a missing field of [`ENTRY_KEYS`] and any field
/// of no list, and this refuses the fields of a `cublaslt.matmul` on the others, theirs missing on it, and `algo_source` on it.
fn fields_of(raw: &RawLibrary, matmul: bool) -> std::result::Result<(), String> {
    let matmul_only = [
        ("beta", &raw.beta),
        ("contract", &raw.contract),
        ("verification", &raw.verification),
        ("canary", &raw.canary),
    ];
    let (missing, unknown): (Vec<&str>, Vec<&str>) = if matmul {
        let missing = matmul_only[..3].iter().filter(|(_, v)| v.is_none()).map(|(k, _)| *k).collect();
        let unknown = raw.algo_source.is_some().then_some("algo_source").into_iter().collect();
        (missing, unknown)
    } else {
        (Vec::new(), matmul_only.iter().filter(|(_, v)| v.is_some()).map(|(k, _)| *k).collect())
    };
    if missing.is_empty() && unknown.is_empty() {
        return Ok(());
    }
    let (keys, optional): (Vec<&str>, &[&str]) = if matmul {
        (ENTRY_KEYS.iter().chain(&MATMUL_KEYS).copied().collect(), &MATMUL_OPTIONAL_KEYS)
    } else {
        (ENTRY_KEYS.to_vec(), &OPTIONAL_KEYS)
    };
    let mut why = format!("must have exactly the fields {keys:?} and may have {optional:?}");
    if !missing.is_empty() {
        why.push_str(&format!("; missing {missing:?}"));
    }
    if !unknown.is_empty() {
        why.push_str(&format!("; unknown {unknown:?}"));
    }
    Err(why)
}

/// The fields of a `cublaslt.matmul` entry that a GEMM's has not (Python's `_validate_matmul_fields`): its `epilogue`, `beta`, `contract`,
/// `verification` and optional `canary`, each exactly typed, and the redundant ones agreeing (the formula is the one of the epilogue and `beta`;
/// the contract is `measured-exact` exactly when no element differed).
#[allow(clippy::type_complexity)]
fn matmul_fields(
    raw: &RawLibrary,
) -> std::result::Result<
    (GemmEpilogue, u8, Option<MatmulContract>, Option<MatmulVerification>, Option<MatmulCanary>),
    String,
> {
    let names: Vec<&str> = GemmEpilogue::ALL.iter().map(|e| e.as_str()).collect();
    let epilogue = GemmEpilogue::from_name(&raw.epilogue)
        .ok_or_else(|| format!("the epilogue of cublaslt.matmul must be one of {names:?}, got {:?}", raw.epilogue))?;
    let beta_value = raw.beta.as_ref().expect("checked by fields_of");
    let beta = match beta_value.as_u64() {
        Some(b @ (0 | 1)) => b as u8,
        _ => return Err(format!("beta must be the int 0 or 1, got {beta_value}")),
    };
    let contract_value = raw.contract.as_ref().expect("checked by fields_of");
    let contract = match contract_value.as_str() {
        Some("measured-exact") => MatmulContract::MeasuredExact,
        Some("binary-defined") => MatmulContract::BinaryDefined,
        _ => {
            return Err(format!(
                "contract must be one of [\"measured-exact\", \"binary-defined\"], got {contract_value}"
            ));
        }
    };
    let v = raw.verification.as_ref().expect("checked by fields_of");
    let record = match v.as_object() {
        Some(map) if map.len() == VERIFICATION_KEYS.len() && VERIFICATION_KEYS.iter().all(|k| map.contains_key(*k)) => {
            map
        }
        _ => return Err(format!("verification must be an object of exactly {VERIFICATION_KEYS:?}, got {v}")),
    };
    let want = epilogue.formula(beta);
    if record["formula"].as_str() != Some(want.as_str()) {
        return Err(format!(
            "verification.formula must be {want:?} (the arithmetic of the epilogue {:?} with beta {beta}), got {}",
            epilogue.as_str(),
            record["formula"]
        ));
    }
    let accumulator = match record["accumulator"].as_str() {
        Some("f32-form") => MatmulAccumulator::F32Form,
        Some("plain-f32-pin") => MatmulAccumulator::PlainF32Pin,
        _ => {
            return Err(format!(
                "verification.accumulator must be one of [\"f32-form\", \"plain-f32-pin\"], got {}",
                record["accumulator"]
            ));
        }
    };
    let count = |key: &str, low: u64| match record[key].as_u64() {
        Some(c) if c >= low => Ok(c),
        _ if low == 1 => Err(format!("verification.{key} must be a positive int, got {}", record[key])),
        _ => Err(format!("verification.{key} must be an int of at least 0, got {}", record[key])),
    };
    let (operand_sets, elements) = (count("operand_sets", 1)?, count("elements", 1)?);
    let (differ, max_ulp) = (count("differ", 0)?, count("max_ulp", 0)?);
    if differ > elements {
        return Err(format!("verification.differ ({differ}) is more than verification.elements ({elements})"));
    }
    // the elements compared are those of every operand set: `operand_sets * m * n` (when `m` and `n` are ints: else they are refused below)
    if let (Some(m), Some(n)) = (raw.m.as_u64(), raw.n.as_u64()) {
        let want = u128::from(operand_sets).checked_mul(u128::from(m)).and_then(|x| x.checked_mul(u128::from(n)));
        if want != Some(u128::from(elements)) {
            let want = want.map_or_else(|| "a number beyond 128 bits".to_owned(), |w| w.to_string());
            return Err(format!("verification.elements must be operand_sets * m * n = {want}, got {elements}"));
        }
    }
    if (contract == MatmulContract::MeasuredExact) != (differ == 0) {
        return Err(format!(
            "contract is {:?}, but the verification found {differ} elements that differ (measured-exact is exactly none)",
            contract.as_str()
        ));
    }
    if differ == 0 && max_ulp != 0 {
        return Err(format!("verification.max_ulp is {max_ulp}, but no element differs"));
    }
    let canary = match &raw.canary {
        None => None,
        Some(c) => Some(matmul_canary(c)?),
    };
    let verification = MatmulVerification { formula: want, accumulator, operand_sets, elements, differ, max_ulp };
    Ok((epilogue, beta, Some(contract), Some(verification), canary))
}

/// The canary of a `cublaslt.matmul` entry: an object of exactly `seed` (an int from 0 to 2^32 - 1) and `out_digest` (32 lowercase hex digits).
fn matmul_canary(v: &Value) -> std::result::Result<MatmulCanary, String> {
    let bad = || {
        format!(
            "canary must be {{\"seed\": an int from 0 to 2**32 - 1, \"out_digest\": 32 lowercase hex digits}}, got {v}"
        )
    };
    let map = v.as_object().filter(|m| m.len() == 2).ok_or_else(bad)?;
    let seed = map.get("seed").and_then(Value::as_u64).filter(|s| *s <= MAX_CANARY_SEED).ok_or_else(bad)?;
    let digest = map
        .get("out_digest")
        .and_then(Value::as_str)
        .filter(|h| h.len() == 32 && h.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
        .ok_or_else(bad)?;
    Ok(MatmulCanary { seed, out_digest: digest.to_owned() })
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
    fn an_optional_field_is_absent_or_its_one_value() {
        let torch = Value::from("torch");
        assert_eq!(optional_field(&None, "heuristic", &torch), Ok(false));
        assert_eq!(optional_field(&Some(Value::from("torch")), "heuristic", &torch), Ok(true));
        for bad in [Value::from("tilefish"), Value::from("Torch"), Value::Null, Value::from(1)] {
            let e = optional_field(&Some(bad.clone()), "heuristic", &torch).unwrap_err();
            assert_eq!(e, format!("heuristic is absent (the default) or \"torch\", got {bad}"));
        }
        let yes = Value::Bool(true);
        assert_eq!(optional_field(&Some(Value::Bool(true)), "allow_bf16_reduction", &yes), Ok(true));
        for bad in [Value::Bool(false), Value::from(1), Value::from("true")] {
            assert!(optional_field(&Some(bad), "allow_bf16_reduction", &yes).is_err());
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
