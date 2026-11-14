//! A pinned cuDNN forward convolution of a manifest's `library` section (`impl: "cudnn.conv_fwd"`), read strictly, loaded and launched with the
//! checks of a kernel (`docs/manifest-schema-3.md`, section 9.4).
//!
//! `y = conv(x, w)`: the cross-correlation of `x` (`[N, C, *S]` in the layout NCHW, `[N, *S, C]` in NHWC) with `w` (`[K, C/G, *R]` or
//! `[K, *R, C/G]`), with `G` groups and, per spatial axis (one, two or three), a stride, a symmetric zero padding and a dilation; f16, bf16 or f32
//! data, an f32 accumulator and one rounding. The entry pins what cuDNN runs: a legacy algorithm enum ([`ConvPin::Legacy`], run by
//! `cudnnConvolutionForward` with a workspace of the entry's `workspace_bytes`) or a backend engine configuration ([`ConvPin::Backend`], an
//! engine's global index and its knob choices, from which an execution plan is finalized), with the cuDNN version that chose it and the SM count
//! of the device it was chosen on. [`Manifest::conv`] refuses another cuDNN version, another SM count, a process with `NVIDIA_TF32_OVERRIDE` set,
//! a host on a partition the pin was not chosen for, and a pin that is not valid for its problem (a legacy algorithm that
//! `cudnnGetConvolutionForwardAlgorithm_v7` does not answer as successful and deterministic; an engine whose numerical notes the math does not
//! allow, whose knobs are not the engine's, or whose plan does not finalize; a workspace above the entry's). Then it runs the entry's canary, when
//! it has one ([`ConvCanary`]: the sha256 of the bytes the pin computed at export from operands made of a seed, [`canary_operands`]), and refuses
//! other bytes: two builds of one cuDNN version answer the same `cudnnGetVersion()`. The canary runs on a stream of its own (`ctx.new_stream()`,
//! non-blocking), with its copies, its zeroed buffers and its frees, never on the NULL stream, which a CUDA green context does not order against
//! another stream. A launch is [`Conv::launch`] on a cudarc stream, with the checks of [`Kernel::launch`](crate::Kernel::launch) of its tensors and
//! the workspace as the last tensor, or `unsafe` [`Conv::launch_raw`], which is how it is captured in a CUDA graph (measured with both APIs: a
//! capture works even when nothing of cuDNN has run in the process before). Every call sets the handle's stream to its own right before it runs.

use std::sync::{Arc, Mutex, Weak};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::error::{Error, PreconditionKind, Result};
use crate::expr::Symbols;
use crate::ffi::cudnn::{self, Desc, Graph, Handle, Legacy};
use crate::kernel::Arch;
use crate::library::{eval_static, present};
use crate::manifest::{
    Access, DimInfo, MAX_ELEMENTS, Manifest, Param, ParamKind, Precondition, ShapeDim, by_name, is_arch, is_digest,
    is_ident, is_stem, parse_dims, parse_shape, precondition_of, want_ident, want_int,
};
use crate::plan::{DeviceInfo, Plan};
use crate::tensor::{DType, TensorArg};

/// The `impl` of a convolution entry.
pub const CONV_IMPL: &str = "cudnn.conv_fwd";
/// The compute type every entry says (`CUDNN_DATA_FLOAT`: an f32 accumulator).
pub const CONV_COMPUTE_TYPE: &str = "CUDNN_DATA_FLOAT";
/// The environment variable that changes what cuDNN picks and runs behind the declared math: a process that has it is refused.
pub const TF32_OVERRIDE: &str = "NVIDIA_TF32_OVERRIDE";
const MAX_WORKSPACE_BYTES: u64 = 1 << 40;
const ALIGNMENTS: [u64; 5] = [16, 32, 64, 128, 256];
/// `cudnnBackendNumericalNote_t`.
const NOTE_TENSOR_CORE: i32 = 0;
const NOTE_DOWN_CONVERT_INPUTS: i32 = 1;
const NOTE_NONDETERMINISTIC: i32 = 4;
const INT32_MAX: u64 = (1 << 31) - 1;

/// The order of the tensors' axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvLayout {
    /// `"NCHW"`: `x [N, C, *S]`, `w [K, C/G, *R]`, `y [N, K, *P]`.
    Nchw,
    /// `"NHWC"`: `x [N, *S, C]`, `w [K, *R, C/G]`, `y [N, *P, K]`.
    Nhwc,
}

impl ConvLayout {
    /// The word an entry uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nchw => "NCHW",
            Self::Nhwc => "NHWC",
        }
    }
}

/// The math of a convolution (`math`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvMath {
    /// `"tensor_op"` (16-bit data): tensor cores allowed, an f32 accumulator, no conversion of the inputs (`CUDNN_TENSOR_OP_MATH`).
    TensorOp,
    /// `"fma"`: no tensor cores (`CUDNN_FMA_MATH`; for f32 data, TF32 off).
    Fma,
    /// `"tf32"` (f32 data): TF32 tensor cores allowed (`CUDNN_DEFAULT_MATH`).
    Tf32,
}

impl ConvMath {
    /// The word an entry uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TensorOp => "tensor_op",
            Self::Fma => "fma",
            Self::Tf32 => "tf32",
        }
    }

    /// The `cudnnMathType_t` a legacy descriptor is given, by name.
    pub fn math_type_name(self) -> &'static str {
        match self {
            Self::TensorOp => "CUDNN_TENSOR_OP_MATH",
            Self::Fma => "CUDNN_FMA_MATH",
            Self::Tf32 => "CUDNN_DEFAULT_MATH",
        }
    }
}

/// How the pin of an entry was chosen (`choice`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvChoice {
    /// `"v7"`: the first deterministic answer of `cudnnGetConvolutionForwardAlgorithm_v7` that fits the workspace (a legacy pin).
    V7,
    /// `"instant"`: torch's posture, `CUDNN_HEUR_MODE_INSTANT` then `_FALLBACK`, deterministic engines only (a backend pin).
    Instant,
    /// `"mode_b"`: the same with `CUDNN_HEUR_MODE_B` first (a backend pin).
    ModeB,
    /// `"imported"`: a table of the declaration gave it.
    Imported,
}

/// What an entry pins.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConvPin {
    /// The legacy algorithm enum (`cudnnConvolutionFwdAlgo_t`, 0 to 7).
    Legacy {
        /// The enum.
        algo: u32,
    },
    /// A backend engine configuration: the engine's global index and its knob choices, `(type, value)` sorted by type.
    Backend {
        /// `CUDNN_ATTR_ENGINE_GLOBAL_INDEX`.
        engine: i64,
        /// `(CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE)` pairs.
        knobs: Vec<(i32, i64)>,
    },
}

impl ConvPin {
    /// `"legacy"` or `"backend"`.
    pub fn api(&self) -> &'static str {
        match self {
            Self::Legacy { .. } => "legacy",
            Self::Backend { .. } => "backend",
        }
    }

    fn what(&self) -> String {
        match self {
            Self::Legacy { algo } => format!("algorithm {algo}"),
            Self::Backend { engine, knobs } => format!("engine {engine} with knobs {knobs:?}"),
        }
    }
}

/// A forward convolution as an entry declares it (Python's `tilefish._cudnn.Problem`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConvProblem {
    /// The type of `x`, `w` and `y`.
    pub dtype: DType,
    /// The order of the axes.
    pub layout: ConvLayout,
    /// The batch.
    pub n: u64,
    /// The input channels.
    pub c: u64,
    /// The output channels.
    pub k: u64,
    /// The groups.
    pub groups: u64,
    /// The spatial sizes of `x` (1 to 3).
    pub input: Vec<u64>,
    /// The spatial sizes of `w`.
    pub filter: Vec<u64>,
    /// Per spatial axis.
    pub stride: Vec<u64>,
    /// Per spatial axis, on both sides.
    pub padding: Vec<u64>,
    /// Per spatial axis.
    pub dilation: Vec<u64>,
    /// The math.
    pub math: ConvMath,
    /// The byte alignment of every pointer, and `CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT` of the backend tensors.
    pub alignment: u64,
}

/// The output size of one spatial axis (0: none).
pub fn output_size(size: u64, kernel: u64, stride: u64, padding: u64, dilation: u64) -> u64 {
    let span = i128::from(size) + 2 * i128::from(padding) - i128::from(dilation) * (i128::from(kernel) - 1) - 1;
    if span < 0 || stride == 0 { 0 } else { u64::try_from(span / i128::from(stride) + 1).unwrap_or(0) }
}

/// The element strides of a dense tensor of cuDNN dims `[N, C, *spatial]` stored in `layout` (Python's `dense_strides`).
pub fn dense_strides(dims: &[u64], layout: ConvLayout) -> Vec<u64> {
    let rank = dims.len();
    let order: Vec<usize> = match layout {
        ConvLayout::Nchw => (0..rank).rev().collect(),
        ConvLayout::Nhwc => std::iter::once(1).chain((2..rank).rev()).chain(std::iter::once(0)).collect(),
    };
    let mut strides = vec![0u64; rank];
    let mut z = 1u64;
    for d in order {
        strides[d] = z;
        z = z.saturating_mul(dims[d]);
    }
    strides
}

impl ConvProblem {
    /// What is wrong with the problem (Python's `Problem.__post_init__`), or the problem.
    pub fn checked(self) -> std::result::Result<Self, String> {
        if let Some(problem) = math_problem(self.dtype, self.math) {
            return Err(problem);
        }
        if !ALIGNMENTS.contains(&self.alignment) {
            return Err(format!("the alignment is one of {ALIGNMENTS:?} bytes, got {}", self.alignment));
        }
        for (name, v) in [("n", self.n), ("c", self.c), ("k", self.k), ("groups", self.groups)] {
            if !(1..=INT32_MAX).contains(&v) {
                return Err(format!("{name} is {v}, outside 1..{INT32_MAX}"));
            }
        }
        let sp = self.input.len();
        if !(1..=3).contains(&sp) {
            return Err(format!("input is the sizes of 1 to 3 spatial axes, got {:?}", self.input));
        }
        for (name, values, low) in [
            ("input", &self.input, 1),
            ("filter", &self.filter, 1),
            ("stride", &self.stride, 1),
            ("padding", &self.padding, 0),
            ("dilation", &self.dilation, 1),
        ] {
            if values.len() != sp || values.iter().any(|v| !(low..=INT32_MAX).contains(v)) {
                return Err(format!("{name} must be {sp} ints from {low} to {INT32_MAX}, got {values:?}"));
            }
        }
        if !self.c.is_multiple_of(self.groups) || !self.k.is_multiple_of(self.groups) {
            return Err(format!("the groups ({}) divide c ({}) and k ({})", self.groups, self.c, self.k));
        }
        for (axis, size) in self.output().iter().enumerate() {
            if *size < 1 {
                return Err(format!("spatial axis {axis} has no output"));
            }
        }
        for (name, dims) in [("x", self.x_dims()), ("w", self.w_dims()), ("y", self.y_dims())] {
            let count = dims.iter().fold(1u128, |a, d| a * u128::from(*d));
            if count > u128::from(INT32_MAX) {
                return Err(format!("{name} has {count} elements, more than an int32 counts"));
            }
        }
        Ok(self)
    }

    /// The number of spatial axes.
    pub fn spatial(&self) -> usize {
        self.input.len()
    }

    /// The output sizes.
    pub fn output(&self) -> Vec<u64> {
        (0..self.spatial())
            .map(|i| output_size(self.input[i], self.filter[i], self.stride[i], self.padding[i], self.dilation[i]))
            .collect()
    }

    fn posed(&self, values: &[u64], one: u64) -> Vec<u64> {
        if self.spatial() == 1 { vec![one, values[0]] } else { values.to_vec() }
    }

    /// The dims of `x` as cuDNN is given them, `[N, C, (1,) *S]`.
    pub fn x_dims(&self) -> Vec<u64> {
        [self.n, self.c].into_iter().chain(self.posed(&self.input, 1)).collect()
    }

    /// The dims of `w`, `[K, C/G, (1,) *R]`.
    pub fn w_dims(&self) -> Vec<u64> {
        [self.k, self.c / self.groups.max(1)].into_iter().chain(self.posed(&self.filter, 1)).collect()
    }

    /// The dims of `y`, `[N, K, (1,) *P]`.
    pub fn y_dims(&self) -> Vec<u64> {
        [self.n, self.k].into_iter().chain(self.posed(&self.output(), 1)).collect()
    }

    /// The strides of `x` in the layout.
    pub fn x_strides(&self) -> Vec<u64> {
        dense_strides(&self.x_dims(), self.layout)
    }

    /// The strides of `w`.
    pub fn w_strides(&self) -> Vec<u64> {
        dense_strides(&self.w_dims(), self.layout)
    }

    /// The strides of `y`.
    pub fn y_strides(&self) -> Vec<u64> {
        dense_strides(&self.y_dims(), self.layout)
    }

    /// The padding as cuDNN is given it.
    pub fn conv_padding(&self) -> Vec<u64> {
        self.posed(&self.padding, 0)
    }

    /// The stride as cuDNN is given it.
    pub fn conv_stride(&self) -> Vec<u64> {
        self.posed(&self.stride, 1)
    }

    /// The dilation as cuDNN is given it.
    pub fn conv_dilation(&self) -> Vec<u64> {
        self.posed(&self.dilation, 1)
    }

    /// The shape tilefish gives `name` (`"x"`, `"w"`, `"y"`) in the layout.
    pub fn tilefish_shape(&self, name: &str) -> Vec<u64> {
        let (batch, channels, spatial) = match name {
            "x" => (self.n, self.c, self.input.clone()),
            "w" => (self.k, self.c / self.groups.max(1), self.filter.clone()),
            _ => (self.n, self.k, self.output()),
        };
        match self.layout {
            ConvLayout::Nchw => [batch, channels].into_iter().chain(spatial).collect(),
            ConvLayout::Nhwc => std::iter::once(batch).chain(spatial).chain(std::iter::once(channels)).collect(),
        }
    }

    /// The exact cuDNN-level problem (Python's `tilefish._cudnn.describe`, value for value: `tests/python_conv.rs` holds them together).
    pub fn describe(&self) -> Value {
        let data_type = match self.dtype {
            DType::F16 => "CUDNN_DATA_HALF",
            DType::Bf16 => "CUDNN_DATA_BFLOAT16",
            _ => "CUDNN_DATA_FLOAT",
        };
        serde_json::json!({
            "library": "cudnn",
            "operation": "convolution_forward",
            "data_type": data_type,
            "compute_type": CONV_COMPUTE_TYPE,
            "format": format!("CUDNN_TENSOR_{}", self.layout.as_str()),
            "math_type": self.math.math_type_name(),
            "conv_mode": "CUDNN_CROSS_CORRELATION",
            "x_dims": self.x_dims(),
            "x_strides": self.x_strides(),
            "w_dims": self.w_dims(),
            "w_strides": self.w_strides(),
            "y_dims": self.y_dims(),
            "y_strides": self.y_strides(),
            "padding": self.conv_padding(),
            "stride": self.conv_stride(),
            "dilation": self.conv_dilation(),
            "groups": self.groups,
            "alignment": self.alignment,
            "uids": {"x": cudnn::UID_X, "w": cudnn::UID_W, "y": cudnn::UID_Y},
            "alpha": 1.0,
            "beta": 0.0,
        })
    }

    fn what(&self) -> String {
        format!(
            "the {} {} convolution x {:?} by w {:?} (stride {:?}, padding {:?}, dilation {:?}, {} groups, math {})",
            self.dtype.name(),
            self.layout.as_str(),
            self.tilefish_shape("x"),
            self.tilefish_shape("w"),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.math.as_str()
        )
    }
}

/// What is wrong with a math for data of a type (Python's `math_problem`, word for word).
pub fn math_problem(dtype: DType, math: ConvMath) -> Option<String> {
    match (dtype, math) {
        (DType::F32, ConvMath::Fma | ConvMath::Tf32)
        | (DType::F16 | DType::Bf16, ConvMath::TensorOp | ConvMath::Fma) => None,
        (DType::F32, other) => Some(format!(
            "f32 data needs the math said explicitly, 'fma' (TF32 off) or 'tf32' (TF32 on), got '{}'",
            other.as_str()
        )),
        (dt, other) => Some(format!(
            "{} data has the math 'tensor_op' (tensor cores, f32 accumulation) or 'fma', got '{}'",
            dt.name(),
            other.as_str()
        )),
    }
}

/// What is wrong with an engine of these numerical notes for a math (Python's `notes_problem`, word for word): torch's filter with
/// `deterministic = True`.
pub fn notes_problem(notes: &[i32], math: ConvMath) -> Option<String> {
    if notes.contains(&NOTE_NONDETERMINISTIC) {
        return Some("it is not deterministic (the numerical note NONDETERMINISTIC)".to_owned());
    }
    if notes.contains(&NOTE_DOWN_CONVERT_INPUTS) {
        return Some("it converts its inputs down (the numerical note DOWN_CONVERT_INPUTS)".to_owned());
    }
    if math == ConvMath::Fma && notes.contains(&NOTE_TENSOR_CORE) {
        return Some("it uses tensor cores (the numerical note TENSOR_CORE), and the math is 'fma'".to_owned());
    }
    None
}

/// What is wrong with knob choices for an engine of these knobs, `(type, minimum, maximum, stride)` (Python's `knobs_problem`, word for word).
pub fn knobs_problem(knobs: &[(i32, i64)], infos: &[(i32, i64, i64, i64)]) -> Option<String> {
    for (t, v) in knobs {
        let Some((_, lo, hi, step)) = infos.iter().find(|(it, ..)| it == t) else {
            let mut types: Vec<i32> = infos.iter().map(|(it, ..)| *it).collect();
            types.sort_unstable();
            return Some(format!("knob type {t} is not a knob of the engine (it has {types:?})"));
        };
        if v < lo || v > hi || (*step > 0 && (v - lo) % step != 0) {
            return Some(format!("knob {t} = {v} is outside the engine's {lo}..{hi} (stride {step})"));
        }
    }
    None
}

// -- the entry ----------------------------------------------------------------------------------------------------------------

/// One pinned convolution of a manifest, for one architecture.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvEntry {
    /// The operator and its bucket values (`conv.T646`); unique with `arch`, across kernels and library entries.
    pub stem: String,
    /// The operator.
    pub op: String,
    /// The architecture the pin was chosen for.
    pub arch: String,
    /// The problem.
    pub problem: ConvProblem,
    /// `x`, `w`, `out`, pointers of the entry's dtype (`out` is written).
    pub params: Vec<Param>,
    /// The bucket value of each bucketed dimension, in the order of the stem.
    pub buckets: Vec<(String, u64)>,
    /// The named dimensions and what is known of each.
    pub dims: Vec<DimInfo>,
    /// What a launch must check: the alignment of every pointer, the static sizes and the buckets.
    pub preconditions: Vec<Precondition>,
    /// The workspace limit the pin was chosen under, the size a launch provides, and the size a legacy call is given.
    pub workspace_bytes: u64,
    /// How the pin was chosen.
    pub choice: ConvChoice,
    /// The pin.
    pub pin: ConvPin,
    /// `cudnnGetVersion()` of the library that chose the pin (`91002` for 9.10.2).
    pub cudnn_version: u64,
    /// The multiprocessor count of the device the pin was chosen on.
    pub sm_count: u64,
    /// The SM count of the partition the pin was chosen for, when it is not the device's.
    pub sm_count_target: Option<u64>,
    /// The sha256 of the canonical declaration.
    pub declaration_digest: String,
    /// What the pin computed at export from operands made of a seed ([`canary_operands`]), run again by [`Manifest::conv`] (absent in an entry exported
    /// without one: then nothing is run).
    pub canary: Option<ConvCanary>,
}

/// The canary of a convolution entry: the sha256 of the bytes of `y` its pin computed at export from the operands of `seed` ([`canary_operands`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvCanary {
    /// The seed of the operands (0 to 2^32 - 1).
    pub seed: u64,
    /// The sha256 of the bytes of `y`, 64 lowercase hex digits.
    pub y_sha256: String,
}

impl ConvEntry {
    /// The SM count the pin was chosen for: the target, or the device's count.
    pub fn sm_count_chosen_for(&self) -> u64 {
        self.sm_count_target.unwrap_or(self.sm_count)
    }

    /// The SM count target a run sets: the entry's, on an engine of the backend API (a legacy call has none to set).
    pub fn run_target(&self) -> Option<u64> {
        match self.pin {
            ConvPin::Backend { .. } => self.sm_count_target,
            ConvPin::Legacy { .. } => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub(crate) struct RawConv {
    stem: String,
    op: String,
    arch: String,
    #[serde(rename = "impl")]
    implementation: String,
    abi: RawConvAbi,
    buckets: Map<String, Value>,
    dims: Map<String, Value>,
    preconditions: Vec<Value>,
    dtypes: Map<String, Value>,
    compute_type: String,
    math: String,
    layout: String,
    n: Value,
    c: Value,
    k: Value,
    groups: Value,
    input: Vec<Value>,
    filter: Vec<Value>,
    output: Vec<Value>,
    stride: Vec<Value>,
    padding: Vec<Value>,
    dilation: Vec<Value>,
    alignment: Value,
    workspace_bytes: Value,
    choice: String,
    pin: Map<String, Value>,
    cudnn_version: Value,
    sm_count: Value,
    declaration_digest: String,
    #[serde(default, deserialize_with = "present")]
    sm_count_target: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    canary: Option<Value>,
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
struct RawConvAbi {
    params: Vec<RawConvParam>,
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
struct RawConvParam {
    index: u64,
    name: String,
    kind: String,
    dtype: String,
    access: String,
    shape: Vec<Value>,
}

by_name! {
    RawConv => RawConv::deserialize, "a cudnn.conv_fwd library entry";
    RawConvAbi => RawConvAbi::deserialize, "abi";
    RawConvParam => RawConvParam::deserialize, "a parameter of abi.params";
}

fn count_field(v: &Value, name: &str, low: u64) -> std::result::Result<u64, String> {
    match v.as_u64() {
        Some(n) if (low..=MAX_ELEMENTS).contains(&n) => Ok(n),
        _ => Err(format!("{name} must be an int from {low} to {MAX_ELEMENTS}, got {v}")),
    }
}

fn spatial_field(values: &[Value], name: &str, count: usize, low: u64) -> std::result::Result<Vec<u64>, String> {
    let parsed: Option<Vec<u64>> =
        values.iter().map(|v| v.as_u64().filter(|n| (low..=MAX_ELEMENTS).contains(n))).collect();
    match parsed {
        Some(p) if p.len() == count => Ok(p),
        _ => Err(format!(
            "{name} must be a list of {count} ints from {low} to {MAX_ELEMENTS} (one per spatial axis), got {}",
            Value::Array(values.to_vec())
        )),
    }
}

fn pin_of(raw: &Map<String, Value>) -> std::result::Result<ConvPin, String> {
    let bad = || {
        format!(
            "pin must be {{\"api\": \"legacy\", \"algo\": a}} or {{\"api\": \"backend\", \"engine\": e, \"knobs\": [[t, v], ...]}}, got {}",
            Value::Object(raw.clone())
        )
    };
    let keys: Vec<&str> = {
        let mut k: Vec<&str> = raw.keys().map(String::as_str).collect();
        k.sort_unstable();
        k
    };
    match raw.get("api").and_then(Value::as_str) {
        Some("legacy") if keys == ["algo", "api"] => {
            let algo = raw["algo"].as_u64().ok_or_else(bad)?;
            if algo > 7 {
                return Err(format!("a legacy cuDNN pin is an algorithm enum from 0 to 7, got {algo}"));
            }
            Ok(ConvPin::Legacy { algo: algo as u32 })
        }
        Some("backend") if keys == ["api", "engine", "knobs"] => {
            let engine = raw["engine"].as_i64().filter(|e| *e >= 0).ok_or_else(bad)?;
            let list = raw["knobs"].as_array().ok_or_else(bad)?;
            let mut knobs = Vec::with_capacity(list.len());
            for kv in list {
                let pair = kv.as_array().filter(|p| p.len() == 2).ok_or_else(bad)?;
                let t = pair[0].as_i64().ok_or_else(bad)?;
                let v = pair[1].as_i64().ok_or_else(bad)?;
                if !(0..cudnn::KNOB_TYPE_COUNT as i64).contains(&t) {
                    return Err(format!(
                        "a knob type of a backend cuDNN pin is an int from 0 to {}, got {t}",
                        cudnn::KNOB_TYPE_COUNT - 1
                    ));
                }
                knobs.push((t as i32, v));
            }
            if knobs.windows(2).any(|w| w[0].0 >= w[1].0) {
                return Err(format!(
                    "the knobs of a backend cuDNN pin are sorted by type, each type once, got {}",
                    raw["knobs"]
                ));
            }
            Ok(ConvPin::Backend { engine, knobs })
        }
        _ => Err(bad()),
    }
}

impl ConvEntry {
    /// Strict reading of one convolution entry (Python's `tilefish._conv.validate_entry`). `index` is its position in the section.
    pub(crate) fn from_raw(raw: RawConv, index: usize) -> Result<Self> {
        let who = format!("library entry #{index} ({:?})", raw.stem);
        Self::read(raw).map_err(|e| Error::manifest(format!("{who}: {e}")))
    }

    fn read(raw: RawConv) -> std::result::Result<Self, String> {
        if !is_stem(&raw.stem) {
            return Err(format!("stem must look like 'conv' or 'conv.T646', got {:?}", raw.stem));
        }
        if !is_ident(&raw.op) {
            return Err(format!("op must be an identifier, got {:?}", raw.op));
        }
        if !is_arch(&raw.arch) {
            return Err(format!("arch must look like 'sm_103', got {:?}", raw.arch));
        }
        if raw.implementation != CONV_IMPL {
            return Err(format!("impl must be {CONV_IMPL:?}, got {:?}", raw.implementation));
        }
        let named = |key: &str| raw.dtypes.get(key).and_then(Value::as_str).and_then(DType::from_name);
        let dtype = match named("x") {
            Some(dt @ (DType::F16 | DType::Bf16 | DType::F32))
                if raw.dtypes.len() == 3 && named("w") == Some(dt) && named("y") == Some(dt) =>
            {
                dt
            }
            _ => {
                return Err(format!(
                    "dtypes must be {{\"x\": T, \"w\": T, \"y\": T}} with T one of [\"f16\", \"bf16\", \"f32\"], got {}",
                    Value::Object(raw.dtypes.clone())
                ));
            }
        };
        if raw.compute_type != CONV_COMPUTE_TYPE {
            return Err(format!("compute_type must be {CONV_COMPUTE_TYPE:?}, got {:?}", raw.compute_type));
        }
        let math = match raw.math.as_str() {
            "tensor_op" => ConvMath::TensorOp,
            "fma" => ConvMath::Fma,
            "tf32" => ConvMath::Tf32,
            other => return Err(format!("math must be one of [\"tensor_op\", \"fma\", \"tf32\"], got {other:?}")),
        };
        let layout = match raw.layout.as_str() {
            "NCHW" => ConvLayout::Nchw,
            "NHWC" => ConvLayout::Nhwc,
            other => return Err(format!("layout must be one of [\"NCHW\", \"NHWC\"], got {other:?}")),
        };
        let (n, c, k, groups) = (
            count_field(&raw.n, "n", 1)?,
            count_field(&raw.c, "c", 1)?,
            count_field(&raw.k, "k", 1)?,
            count_field(&raw.groups, "groups", 1)?,
        );
        let count = raw.input.len();
        if !(1..=3).contains(&count) {
            return Err(format!(
                "input must be a list of 1 to 3 spatial sizes, got {}",
                Value::Array(raw.input.clone())
            ));
        }
        let input = spatial_field(&raw.input, "input", count, 1)?;
        let filter = spatial_field(&raw.filter, "filter", count, 1)?;
        let output = spatial_field(&raw.output, "output", count, 1)?;
        let stride = spatial_field(&raw.stride, "stride", count, 1)?;
        let padding = spatial_field(&raw.padding, "padding", count, 0)?;
        let dilation = spatial_field(&raw.dilation, "dilation", count, 1)?;
        let alignment = match raw.alignment.as_u64() {
            Some(a) if ALIGNMENTS.contains(&a) => a,
            _ => return Err(format!("alignment must be one of {ALIGNMENTS:?}, got {}", raw.alignment)),
        };
        let problem =
            ConvProblem { dtype, layout, n, c, k, groups, input, filter, stride, padding, dilation, math, alignment }
                .checked()
                .map_err(|e| format!("not a convolution cuDNN takes: {e}"))?;
        if problem.output() != output {
            return Err(format!(
                "output must be the output of the convolution, {:?}, got {output:?}",
                problem.output()
            ));
        }
        let workspace_bytes = match raw.workspace_bytes.as_u64() {
            Some(w) if (1..=MAX_WORKSPACE_BYTES).contains(&w) => w,
            _ => return Err(format!("workspace_bytes must be an int from 1 to 2**40, got {}", raw.workspace_bytes)),
        };
        let choice = match raw.choice.as_str() {
            "v7" => ConvChoice::V7,
            "instant" => ConvChoice::Instant,
            "mode_b" => ConvChoice::ModeB,
            "imported" => ConvChoice::Imported,
            other => {
                return Err(format!(
                    "choice must be one of [\"v7\", \"instant\", \"mode_b\", \"imported\"], got {other:?}"
                ));
            }
        };
        let pin = pin_of(&raw.pin)?;
        let wanted = match choice {
            ConvChoice::V7 => Some("legacy"),
            ConvChoice::Instant | ConvChoice::ModeB => Some("backend"),
            ConvChoice::Imported => None,
        };
        if let Some(api) = wanted
            && pin.api() != api
        {
            return Err(format!("the choice {:?} pins a {api} pin, got a {} one", raw.choice, pin.api()));
        }
        let cudnn_version = match raw.cudnn_version.as_u64() {
            Some(v) if v >= 1 => v,
            _ => return Err(format!("cudnn_version must be a positive int, got {}", raw.cudnn_version)),
        };
        let sm_count = match raw.sm_count.as_u64() {
            Some(v) if v >= 1 => v,
            _ => {
                return Err(format!(
                    "sm_count must be a positive int (the multiprocessors of the device the pin was chosen on), got {}",
                    raw.sm_count
                ));
            }
        };
        let sm_count_target = match &raw.sm_count_target {
            None => None,
            Some(v) => match v.as_u64() {
                Some(t) if (1..=sm_count).contains(&t) => Some(t),
                _ => {
                    return Err(format!(
                        "sm_count_target is absent or an int from 1 to sm_count ({sm_count}), got {v}"
                    ));
                }
            },
        };
        if sm_count_target.is_some() && pin.api() == "legacy" && choice != ConvChoice::Imported {
            return Err(
                "a legacy pin has an sm_count_target only when it was imported (the legacy API takes no target)"
                    .to_owned(),
            );
        }
        let canary = match &raw.canary {
            None => None,
            Some(v) => Some(canary_of(v).ok_or_else(|| {
                format!(
                    "canary is absent or {{\"seed\": an int from 0 to 2**32 - 1, \"y_sha256\": 64 lowercase hex digits}}, got {v}"
                )
            })?),
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
        let shapes = [problem.tilefish_shape("x"), problem.tilefish_shape("w"), problem.tilefish_shape("y")];
        if raw.abi.params.len() != 3 {
            return Err("abi.params must list 3 pointers (x, w, out in this order)".to_owned());
        }
        let mut params = Vec::with_capacity(3);
        for (i, (p, size)) in raw.abi.params.iter().zip(&shapes).enumerate() {
            want_ident(&Value::String(p.name.clone()), &format!("abi.params[{i}].name"))?;
            let last = i == 2;
            let want_access = if last { "out" } else { "in" };
            if p.index != i as u64 || p.kind != "pointer" || p.dtype != dtype.name() || p.access != want_access {
                return Err(format!(
                    "abi.params[{i}] must be the {} pointer #{i} with access {want_access}",
                    dtype.name()
                ));
            }
            if raw.abi.params[..i].iter().any(|q| q.name == p.name) {
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
                        "abi.params[{i}].shape[{axis}] is {entry} = {value}, but the problem says {want}"
                    ));
                }
            }
            params.push(Param {
                slot: i,
                index: i,
                name: p.name.clone(),
                kind: ParamKind::Pointer {
                    dtype_name: dtype.name().to_owned(),
                    dtype: Some(dtype),
                    access: if last { Access::Out } else { Access::In },
                    shape,
                    view: None,
                    alias: None,
                    tensor_map: None,
                },
            });
        }
        let mut syms = Symbols::default();
        let mut preconditions = Vec::with_capacity(raw.preconditions.len());
        let mut aligned = [false; 3];
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
                    if !named || *bytes != alignment || std::mem::replace(&mut aligned[*index], true) {
                        return Err(format!(
                            "pointer_alignment must be the entry's alignment ({alignment} bytes), once per pointer, on the pointer it names"
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
            problem,
            params,
            buckets,
            dims,
            preconditions,
            workspace_bytes,
            choice,
            pin,
            cudnn_version,
            sm_count,
            sm_count_target,
            declaration_digest: raw.declaration_digest,
            canary,
        })
    }
}

/// The canary of an entry: an object of exactly `seed` and `y_sha256`.
fn canary_of(v: &Value) -> Option<ConvCanary> {
    let map = v.as_object().filter(|m| m.len() == 2)?;
    let seed = map.get("seed")?.as_u64().filter(|s| *s <= MAX_CANARY_SEED)?;
    let y_sha256 = map.get("y_sha256")?.as_str().filter(|h| is_digest(h))?;
    Some(ConvCanary { seed, y_sha256: y_sha256.to_owned() })
}

// -- the canary's operands -------------------------------------------------------------------------------------------------------------

/// The prefix of the canary's SHA-256 blocks.
const CANARY_PREFIX: &[u8] = b"tilefish.conv.canary\0";
const MAX_CANARY_SEED: u64 = (1 << 32) - 1;

/// What the most significant byte of an element becomes (Python's `tilefish._conv._top_byte_table`): f16 (`s eeeee mm`) an exponent field of 7 to 14,
/// bf16 and f32 (`s eeeeeee`) exponents 110 to 125; the sign and the low bits stay.
fn canary_top_byte(dtype: DType, b: u8) -> u8 {
    match dtype {
        DType::F16 => (b & 0x80) | ((7 + ((b >> 2) & 7)) << 2) | (b & 3),
        _ => (b & 0x80) | (55 + (b & 7)),
    }
}

/// The bytes of a canary operand of `elements` elements of `dtype` (f16, bf16 or f32; little-endian, in memory order), as Python's
/// `tilefish._conv.canary_bytes` makes them: SHA-256 in counter mode, block `j` the digest of the prefix `tilefish.conv.canary\0`, `seed` (8 bytes
/// little-endian), `tag` (1 byte: 0 for `x`, 1 for `w`) and `j` (8 bytes little-endian), cut to the size, the most significant byte of each element
/// put in a range of exponents (finite, normal, many orders of magnitude: sums whose rounding depends on their order).
pub fn canary_bytes(dtype: DType, seed: u64, tag: u8, elements: u64) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let size = dtype.size();
    let count = elements as usize * size;
    let mut base = Sha256::new();
    base.update(CANARY_PREFIX);
    base.update(seed.to_le_bytes());
    base.update([tag]);
    let mut out = Vec::with_capacity(count.div_ceil(32) * 32);
    for j in 0..count.div_ceil(32) as u64 {
        let mut h = base.clone();
        h.update(j.to_le_bytes());
        out.extend_from_slice(&h.finalize());
    }
    out.truncate(count);
    for b in out.iter_mut().skip(size - 1).step_by(size) {
        *b = canary_top_byte(dtype, *b);
    }
    out
}

/// The bytes of the canary's `x` and `w` for `problem`, dense in its layout (Python's `tilefish._conv.canary_operands`).
pub fn canary_operands(problem: &ConvProblem, seed: u64) -> (Vec<u8>, Vec<u8>) {
    (
        canary_bytes(problem.dtype, seed, 0, problem.x_dims().iter().product()),
        canary_bytes(problem.dtype, seed, 1, problem.w_dims().iter().product()),
    )
}

// -- loading ------------------------------------------------------------------------------------------------------------------

/// What a manifest remembers of a convolution it loaded: shared by the loads of the same context while any of them is alive.
#[derive(Debug)]
pub(crate) struct CachedConv {
    loaded: Weak<LoadedConv>,
    plan: Arc<Plan>,
}

/// The engine a pin runs with: the legacy descriptors, or the graph and the execution plan of the backend.
#[derive(Debug)]
enum Engine {
    Legacy { legacy: Legacy, algo: u32 },
    Backend { plan: Desc, _graph: Graph },
}

#[derive(Debug)]
struct LoadedConv {
    // the handle and the engine under one lock: a call sets the handle's stream and runs on it (the stream is state of the handle)
    engine: Mutex<(Handle, Engine)>,
    context: Arc<CudaContext>,
    device: DeviceInfo,
    /// What the pin needs for the problem (a legacy call is given the entry's `workspace_bytes`, so that is what it requires).
    required: u64,
    stem: String,
    arch: String,
    problem: ConvProblem,
    pin: ConvPin,
    workspace_bytes: u64,
    cudnn_version: u64,
    sm_count: u64,
    sm_count_target: Option<u64>,
    canary: Option<ConvCanary>,
}

/// A pinned convolution of a manifest, loaded in a context and ready to launch. Cheap to clone; it can be shared between threads (a call
/// runs under a lock of its own: the handle's stream is its state). It keeps its context and its cuDNN handle alive.
#[derive(Debug, Clone)]
pub struct Conv {
    loaded: Arc<LoadedConv>,
    plan: Arc<Plan>,
}

impl Manifest {
    /// Every convolution of the manifest (`impl: "cudnn.conv_fwd"` in the `library` section), in file order.
    pub fn convolutions(&self) -> &[ConvEntry] {
        &self.convs
    }

    /// The convolution of `stem` for `arch` (`"sm_103"`).
    pub fn conv_entry(&self, stem: &str, arch: &str) -> Option<&ConvEntry> {
        self.convs.iter().find(|e| e.stem == stem && e.arch == arch)
    }

    /// The pinned convolution for `name` (a stem such as `conv.T646`, or an operator) with exactly these bucket values, for `arch`, loaded in `ctx`.
    ///
    /// As [`Manifest::gemm`]: bucket semantics are exact, `arch` is the compute capability of the device of `ctx`, and the convolution is loaded
    /// once per context and shared while any clone of it is alive. It is refused when cuDNN is not on this machine, when its version is not the
    /// one that chose the pin, when the device has another SM count, when the process has `NVIDIA_TF32_OVERRIDE` set, when the host runs on a
    /// partition the pin was not chosen for ([`Manifest::with_sm_partition`]), and when the pin is not valid for its problem (see the module docs).
    pub fn conv(&self, ctx: &Arc<CudaContext>, name: &str, buckets: &[(&str, u64)], arch: Arch) -> Result<Conv> {
        let index = self.select_conv(name, buckets, &arch.to_string())?;
        let entry = &self.convs[index];
        if let Some(partition) = self.sm_partition {
            partition_matches(entry, partition)?;
        }
        let plan = Plan::for_conv(entry)?;
        let device_arch = Arch::of_context(ctx)?;
        if device_arch != arch {
            return Err(Error::Unsupported(format!(
                "'{}' is asked for {arch}, but the CUDA device {} of the context is {device_arch}",
                entry.stem,
                ctx.ordinal()
            )));
        }
        let key = (index, ctx.cu_ctx() as usize);
        let mut cache = self.conv_cache.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = cache.get(&key)
            && let Some(loaded) = cached.loaded.upgrade()
        {
            return Ok(Conv { loaded, plan: Arc::clone(&cached.plan) });
        }
        let loaded = load(entry, ctx)?;
        let conv = Conv { loaded: Arc::new(loaded), plan: Arc::new(plan) };
        cache.insert(key, CachedConv { loaded: Arc::downgrade(&conv.loaded), plan: Arc::clone(&conv.plan) });
        Ok(conv)
    }

    /// The plan of a convolution without loading it: what [`Manifest::conv`] checks the tensors of a call with (`x`, `w`, `out`; the workspace
    /// is checked on its own), for a machine with no GPU.
    pub fn conv_plan(&self, name: &str, buckets: &[(&str, u64)], arch: Arch) -> Result<Plan> {
        let index = self.select_conv(name, buckets, &arch.to_string())?;
        Plan::for_conv(&self.convs[index])
    }
}

/// Refuse an entry whose pin was chosen for another SM count than the partition the host runs it on.
fn partition_matches(entry: &ConvEntry, partition: u64) -> Result<()> {
    if entry.sm_count_chosen_for() != partition {
        let what = match entry.sm_count_target {
            Some(t) => format!("{t} SMs (its sm_count_target)"),
            None => format!("the whole device ({} SMs)", entry.sm_count),
        };
        return Err(Error::Manifest(format!(
            "the cuDNN pin of '{}' ({}) was chosen for {what}, but this host runs it on a partition of {partition} SMs: declare the convolution \
             with sm_count_target={partition} and export it again, or give the host the partition the pin is for",
            entry.stem, entry.arch
        )));
    }
    Ok(())
}

/// Refuse a process that has `NVIDIA_TF32_OVERRIDE` set (any value): the words of Python's `refuse_tf32_override`.
pub(crate) fn refuse_tf32_override(value: Option<std::ffi::OsString>) -> std::result::Result<(), String> {
    match value {
        None => Ok(()),
        Some(v) => Err(format!(
            "{TF32_OVERRIDE} is set (to '{}') in this process: it changes the algorithms cuDNN chooses and runs behind the declared math (TF32 on \
             or off), so a pinned cuDNN convolution is neither pinned nor run here; unset it",
            v.to_string_lossy()
        )),
    }
}

fn load(entry: &ConvEntry, ctx: &Arc<CudaContext>) -> Result<LoadedConv> {
    let who = format!("'{}' ({})", entry.stem, entry.arch);
    let sm_count = ctx
        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(Error::driver("cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)"))?;
    let sm_count = u64::try_from(sm_count).unwrap_or(0);
    if entry.sm_count != sm_count {
        return Err(Error::Manifest(format!(
            "the manifest pinned the cuDNN convolution '{}' ({}) on a device with {} SMs, but this device has {sm_count}: cuDNN chooses for the \
             multiprocessors it runs on, so the pin is refused (re-export the manifest on this device)",
            entry.stem, entry.arch, entry.sm_count
        )));
    }
    refuse_tf32_override(std::env::var_os(TF32_OVERRIDE))
        .map_err(|e| Error::Manifest(format!("the cuDNN convolution {who} is not run here: {e}")))?;
    if !cudnn::available() {
        return Err(Error::Unsupported(format!(
            "{who} is a cuDNN convolution, and libcudnn cannot be loaded on this machine (with LD_LIBRARY_PATH or the like, give the process the \
             cuDNN of version {}, the one that chose its pin)",
            entry.cudnn_version
        )));
    }
    let here = cudnn::version();
    if here != entry.cudnn_version {
        return Err(Error::Manifest(format!(
            "the manifest pinned the cuDNN convolution '{}' ({}) with cuDNN {}, but this process has cuDNN {here}: a pin names other kernels in \
             another version, so it is refused (export the manifest again with this cuDNN, or run with the version that made it)",
            entry.stem, entry.arch, entry.cudnn_version
        )));
    }
    let handle = Handle::new(ctx)?;
    let problem = &entry.problem;
    let head = format!("{who}: the pinned cuDNN {} is not valid for {}", entry.pin.what(), problem.what());
    let refuse = |why: String| Error::Manifest(format!("{head}: {why}"));
    let (engine, required) = match &entry.pin {
        ConvPin::Legacy { algo } => {
            let legacy = Legacy::new(problem)?;
            let found = legacy.candidates(&handle)?.into_iter().find(|c| c.algo == *algo as i32);
            match found {
                Some(c) if c.status == cudnn::STATUS_SUCCESS && c.determinism == cudnn::DETERMINISTIC => {}
                Some(c) if c.status == cudnn::STATUS_SUCCESS => {
                    return Err(refuse(
                        "cudnnGetConvolutionForwardAlgorithm_v7 says it is not deterministic".to_owned(),
                    ));
                }
                _ => {
                    return Err(refuse(
                        "cudnnGetConvolutionForwardAlgorithm_v7 does not answer it with CUDNN_STATUS_SUCCESS"
                            .to_owned(),
                    ));
                }
            }
            let need = legacy
                .workspace(&handle, *algo)?
                .ok_or_else(|| refuse("cudnnGetConvolutionForwardWorkspaceSize refuses it".to_owned()))?;
            if need as u64 > entry.workspace_bytes {
                return Err(refuse(format!(
                    "it needs {need} bytes of workspace, more than the {} recorded",
                    entry.workspace_bytes
                )));
            }
            // a legacy call is given the whole limit, at every launch
            (Engine::Legacy { legacy, algo: *algo }, entry.workspace_bytes)
        }
        ConvPin::Backend { engine, knobs } => {
            let graph = Graph::new(&handle, problem)?;
            let target = entry.run_target();
            let found = graph
                .engine(*engine, target)?
                .ok_or_else(|| refuse(format!("engine {engine} is not an engine of the problem")))?;
            let facts = Graph::facts(&found)?;
            let problems: Vec<String> = [notes_problem(&facts.notes, problem.math), knobs_problem(knobs, &facts.knobs)]
                .into_iter()
                .flatten()
                .collect();
            if !problems.is_empty() {
                return Err(refuse(problems.join("; ")));
            }
            let config = Graph::config(&found, knobs)?.ok_or_else(|| {
                refuse(format!("the knob choices {knobs:?} do not make a configuration of engine {engine}"))
            })?;
            let (plan, need) = Graph::plan(&handle, &config)?
                .ok_or_else(|| refuse("its execution plan does not finalize".to_owned()))?;
            if need as u64 > entry.workspace_bytes {
                return Err(refuse(format!(
                    "it needs {need} bytes of workspace, more than the {} recorded",
                    entry.workspace_bytes
                )));
            }
            (Engine::Backend { plan, _graph: graph }, need as u64)
        }
    };
    if let Some(canary) = &entry.canary {
        let mine = CanaryOf { who: &who, pin: &entry.pin, cudnn_version: entry.cudnn_version, problem, required };
        check_canary(ctx, &handle, &engine, &mine, canary)?;
    }
    Ok(LoadedConv {
        engine: Mutex::new((handle, engine)),
        context: Arc::clone(ctx),
        device: DeviceInfo {
            ordinal: ctx.ordinal() as i32,
            context: Some(ctx.cu_ctx() as usize),
            max_grid: [i64::MAX; 3],
        },
        required,
        stem: entry.stem.clone(),
        arch: entry.arch.clone(),
        problem: entry.problem.clone(),
        pin: entry.pin.clone(),
        workspace_bytes: entry.workspace_bytes,
        cudnn_version: entry.cudnn_version,
        sm_count: entry.sm_count,
        sm_count_target: entry.sm_count_target,
        canary: entry.canary.clone(),
    })
}

/// What the check of a canary says of its convolution.
struct CanaryOf<'a> {
    who: &'a str,
    pin: &'a ConvPin,
    cudnn_version: u64,
    problem: &'a ConvProblem,
    required: u64,
}

/// Run a canary ([`run_canary`]) and refuse other bytes.
fn check_canary(
    ctx: &Arc<CudaContext>,
    handle: &Handle,
    engine: &Engine,
    of: &CanaryOf<'_>,
    canary: &ConvCanary,
) -> Result<()> {
    let got = run_canary(ctx, handle, engine, of.required, of.problem, canary.seed)?;
    if got != canary.y_sha256 {
        return Err(Error::Manifest(format!(
            "the canary of the cuDNN convolution {} ({}) gives y with sha256 {got} here, and the manifest recorded {} (seed {}): this process's \
             cuDNN (version {} too) or this device runs other kernels for the pin, so it is refused; export the manifest again with this cuDNN \
             build, or run with the build that made it",
            of.who,
            of.pin.what(),
            canary.y_sha256,
            canary.seed,
            of.cudnn_version
        )));
    }
    Ok(())
}

/// Run the canary of a loaded pin: its `y` from the operands of `seed`, and the sha256 of its bytes (hex).
///
/// Everything is on a stream made for it (`ctx.new_stream()`: non-blocking, ordered with no other stream, the NULL stream included, which a green
/// context does not order against another stream): the copies of the operands, the zeroed `y` and workspace, the convolution, the copy of `y` back
/// (which waits for that stream alone) and the frees.
#[allow(unsafe_code)]
fn run_canary(
    ctx: &Arc<CudaContext>,
    handle: &Handle,
    engine: &Engine,
    required: u64,
    problem: &ConvProblem,
    seed: u64,
) -> Result<String> {
    use cudarc::driver::DevicePtr;
    use sha2::{Digest, Sha256};
    let stream = ctx.new_stream().map_err(Error::driver("cuStreamCreate"))?;
    let (x_host, w_host) = canary_operands(problem, seed);
    let y_bytes = problem.y_dims().iter().product::<u64>() as usize * problem.dtype.size();
    let x = stream.clone_htod(&x_host).map_err(Error::driver("cuMemcpyHtoDAsync"))?;
    let w = stream.clone_htod(&w_host).map_err(Error::driver("cuMemcpyHtoDAsync"))?;
    let mut y = stream.alloc_zeros::<u8>(y_bytes).map_err(Error::driver("cuMemAllocAsync"))?;
    let mut workspace = if required > 0 {
        Some(stream.alloc_zeros::<u8>(required as usize).map_err(Error::driver("cuMemAllocAsync"))?)
    } else {
        None
    };
    {
        use cudarc::driver::DevicePtrMut;
        let (x_ptr, _x) = x.device_ptr(&stream);
        let (w_ptr, _w) = w.device_ptr(&stream);
        let (y_ptr, _y) = y.device_ptr_mut(&stream);
        let ws = workspace.as_mut().map(|s| s.device_ptr_mut(&stream));
        let call = ConvCall { x: x_ptr, w: w_ptr, out: y_ptr, workspace: ws.as_ref().map_or(0, |(p, _)| *p) };
        ctx.bind_to_thread().map_err(Error::driver("cuCtxSetCurrent"))?;
        // SAFETY: the stream is a live stream of the handle's context, current on this thread; the addresses are slices of that context made above with
        // the sizes of the problem (and the workspace with what the pin needs), alive until the copy back below has waited for the stream.
        unsafe { run_engine(handle, engine, required, stream.cu_stream(), &call) }?;
    }
    let host = stream.clone_dtoh(&y).map_err(Error::driver("cuMemcpyDtoHAsync"))?;
    drop((x, w, y, workspace));
    let digest = Sha256::digest(&host);
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

/// Run a pin on `stream`: set the handle's stream, then the legacy call (given `required` bytes of workspace, the entry's whole limit, when it has
/// one) or the execution of the backend plan.
///
/// # Safety
///
/// `stream` must be a valid `CUstream` of the handle's context, which is current on this thread, and the addresses of `call` must be device memory of
/// that context holding what the problem says (the workspace at least `required` bytes, or 0 when the pin needs none), allocated until the work has
/// completed on `stream`.
#[allow(unsafe_code)]
unsafe fn run_engine(
    handle: &Handle,
    engine: &Engine,
    required: u64,
    stream: cudarc::driver::sys::CUstream,
    call: &ConvCall,
) -> Result<()> {
    // SAFETY: the contract of this function: the stream is a valid stream of the handle's context.
    unsafe { handle.set_stream(stream) }?;
    match engine {
        // SAFETY: the contract of this function; the workspace has `required` bytes, which the call is given.
        Engine::Legacy { legacy, algo } => unsafe {
            legacy.forward(
                handle,
                *algo,
                call.x,
                call.w,
                call.out,
                call.workspace,
                if call.workspace == 0 { 0 } else { required as usize },
            )
        },
        // SAFETY: the contract of this function; the workspace has at least what the plan needs.
        Engine::Backend { plan, .. } => unsafe {
            cudnn::execute(handle, plan, call.x, call.w, call.out, call.workspace)
        },
    }
}

/// The addresses of a checked call.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvCall {
    pub x: u64,
    pub w: u64,
    pub out: u64,
    /// The workspace, or 0 when the pin needs none.
    pub workspace: u64,
}

impl Conv {
    /// The stem of the entry (`conv.T646`).
    pub fn stem(&self) -> &str {
        &self.loaded.stem
    }

    /// The architecture the pin was chosen for.
    pub fn arch(&self) -> &str {
        &self.loaded.arch
    }

    /// The ordinal of the device it is loaded on.
    pub fn device(&self) -> i32 {
        self.loaded.device.ordinal
    }

    /// The context it was loaded in.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.loaded.context
    }

    /// What the tensors of a call are checked against (`x`, `w`, `out`; the workspace on its own).
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// The problem.
    pub fn problem(&self) -> &ConvProblem {
        &self.loaded.problem
    }

    /// The pin.
    pub fn pin(&self) -> &ConvPin {
        &self.loaded.pin
    }

    /// The cuDNN version that chose the pin (this process has it: the load refuses another).
    pub fn cudnn_version(&self) -> u64 {
        self.loaded.cudnn_version
    }

    /// The SM count of the device the pin was chosen on.
    pub fn sm_count(&self) -> u64 {
        self.loaded.sm_count
    }

    /// The SM count of the partition the pin was chosen for, when it is not the device's.
    pub fn sm_count_target(&self) -> Option<u64> {
        self.loaded.sm_count_target
    }

    /// The canary of the entry, which the load ran.
    pub fn canary(&self) -> Option<&ConvCanary> {
        self.loaded.canary.as_ref()
    }

    /// Run the entry's canary again, as the load did: on a stream of its own (never the NULL stream), under the handle's lock, refused
    /// ([`Error::Manifest`]) when the bytes differ; nothing when the entry has none.
    pub fn check_canary(&self) -> Result<()> {
        let Some(canary) = &self.loaded.canary else {
            return Ok(());
        };
        let loaded = &self.loaded;
        let who = format!("'{}' ({})", loaded.stem, loaded.arch);
        let guard = loaded.engine.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let (handle, engine) = &*guard;
        let of = CanaryOf {
            who: &who,
            pin: &loaded.pin,
            cudnn_version: loaded.cudnn_version,
            problem: &loaded.problem,
            required: loaded.required,
        };
        check_canary(&loaded.context, handle, engine, &of, canary)
    }

    /// The workspace limit of the entry, the size [`Conv::alloc_workspace`] allocates.
    pub fn workspace_bytes(&self) -> u64 {
        self.loaded.workspace_bytes
    }

    /// The bytes of workspace a launch needs: what the plan needs for a backend pin, the entry's `workspace_bytes` for a legacy one (a legacy
    /// call is given the whole limit, so that it is the same call at every launch).
    pub fn workspace_required(&self) -> u64 {
        self.loaded.required
    }

    /// A workspace of [`Conv::workspace_bytes`] bytes (zeroed), allocated on `stream`.
    pub fn alloc_workspace(&self, stream: &Arc<CudaStream>) -> Result<CudaSlice<u8>> {
        let bytes = usize::try_from(self.loaded.workspace_bytes)
            .map_err(|_| Error::Launch("the workspace of the entry does not fit the address space".to_owned()))?;
        stream.alloc_zeros::<u8>(bytes).map_err(Error::driver("cuMemAllocAsync"))
    }

    /// Everything a launch checks, with nothing launched. `tensors` are `x`, `w`, `out` and the workspace.
    pub fn check(&self, tensors: &[TensorArg<'_>]) -> Result<()> {
        self.check_call(tensors).map(drop)
    }

    pub(crate) fn check_call(&self, tensors: &[TensorArg<'_>]) -> Result<ConvCall> {
        let loaded = &*self.loaded;
        ConvChecks {
            stem: &loaded.stem,
            plan: &self.plan,
            device: &loaded.device,
            problem: &loaded.problem,
            required: loaded.required,
            workspace_bytes: loaded.workspace_bytes,
        }
        .check_call(tensors)
    }

    /// Check the call and run the convolution on `stream` (a stream of the context it was loaded in: checked). `tensors` are `x`, `w`, `out`
    /// (written) and the workspace (written), with the checks of [`Gemm::launch`](crate::Gemm::launch) and the same ordering of the slices.
    pub fn launch(&self, stream: &CudaStream, tensors: &mut [TensorArg<'_>]) -> Result<()> {
        if **stream.context() != *self.loaded.context {
            return Err(Error::Launch(format!(
                "the stream is in another CUDA context than the one '{}' was loaded in: launch it on a stream of that context",
                self.loaded.stem
            )));
        }
        let call = self.check_call(tensors)?;
        crate::ffi::enqueue_conv(self, stream, &call, tensors)
    }

    /// Run the pin of a checked call on `stream`.
    ///
    /// # Safety
    ///
    /// `stream` must be a valid `CUstream` of this convolution's context, which is current on this thread, and the addresses of `call` must be
    /// device memory of that context holding what the problem says, allocated until the work has completed on `stream`: what
    /// `Conv::check_call` checks of the tensors they came from.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn run(&self, stream: cudarc::driver::sys::CUstream, call: &ConvCall) -> Result<()> {
        let guard = self.loaded.engine.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let (handle, engine) = &*guard;
        // SAFETY: the contract of this function; the workspace has what the pin requires (checked by `check_call`: a legacy call's is the entry's whole
        // workspace_bytes, which it is given).
        unsafe { run_engine(handle, engine, self.loaded.required, stream, call) }
    }
}

/// What the checks of a call need of a loaded convolution, without the library, so that they run without a GPU.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvChecks<'a> {
    pub stem: &'a str,
    pub plan: &'a Plan,
    pub device: &'a DeviceInfo,
    pub problem: &'a ConvProblem,
    /// What a launch needs.
    pub required: u64,
    pub workspace_bytes: u64,
}

impl ConvChecks<'_> {
    /// Check a call (see [`Conv::launch`]) and return the addresses to run it with.
    pub(crate) fn check_call(&self, tensors: &[TensorArg<'_>]) -> Result<ConvCall> {
        let stem = self.stem;
        if tensors.len() != 4 {
            return Err(Error::Launch(format!(
                "'{stem}' takes 4 tensors ({}, workspace), got {}",
                self.plan.tensor_names().collect::<Vec<_>>().join(", "),
                tensors.len()
            )));
        }
        let (operands, workspace) = tensors.split_at(3);
        let workspace = &workspace[0];
        self.plan.prepare(operands, Some(self.device))?;
        // (the entry's shapes agree with the problem, and its dimensions are bucketed to them: a defense for the memory the library reads)
        for (arg, name) in operands.iter().zip(["x", "w", "y"]) {
            let want: Vec<usize> = self.problem.tilefish_shape(name).iter().map(|d| *d as usize).collect();
            if arg.shape != want.as_slice() {
                return Err(Error::precondition(
                    PreconditionKind::Shape,
                    None,
                    format!("'{stem}': a tensor has shape {:?}, but the problem's {name} is {want:?}", arg.shape),
                ));
            }
        }
        crate::gemm::check_workspace(
            stem,
            self.device,
            workspace,
            operands,
            self.plan,
            self.required,
            self.workspace_bytes,
            &format!("the cuDNN pin of '{stem}'"),
        )?;
        let address = |i: usize| operands[i].ptr;
        Ok(ConvCall {
            x: address(0),
            w: address(1),
            out: address(2),
            workspace: if self.required == 0 { 0 } else { workspace.ptr },
        })
    }
}

/// The version of cuDNN this process has (`91002` for 9.10.2), or `None` when it cannot be loaded: what a pin is compared with.
pub fn cudnn_version() -> Option<u64> {
    cudnn::available().then(cudnn::version)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Pin;
    use crate::region::DevicePtr;
    use crate::tensor::{TensorMut, U8};

    const CONTEXT: usize = 0xC0_0000;
    const DEVICE: DeviceInfo = DeviceInfo { ordinal: 0, context: Some(CONTEXT), max_grid: [i64::MAX; 3] };

    /// The manifest of the entries Python wrote for `stem` in the cross-language vectors (`tests/data/python_conv.json`).
    fn manifest_of(stem: &str) -> Manifest {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join("python_conv.json");
        let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let entry = cases
            .iter()
            .find(|c| c["accepted"] == true && c["entry"]["stem"] == stem)
            .map(|c| c["entry"].clone())
            .unwrap_or_else(|| panic!("no accepted vector of {stem}"));
        let doc = serde_json::json!({"schema": 3, "kernels": [], "library": [entry]});
        Manifest::from_embedded(doc.to_string().as_bytes(), &[], Pin::TrustDirectory).unwrap()
    }

    fn region(addr: u64, bytes: u64) -> DevicePtr<'static> {
        // SAFETY: never dereferenced: the checks only look at the numbers.
        #[allow(unsafe_code)]
        unsafe {
            DevicePtr::from_raw(addr, bytes, 0)
        }
    }

    fn f16(addr: u64, shape: &[usize], access: Access) -> TensorArg<'_> {
        TensorArg::new(region(addr, shape.iter().product::<usize>() as u64 * 2), shape, DType::F16, access).unwrap()
    }

    fn workspace(addr: u64, bytes: &[usize]) -> TensorArg<'_> {
        let mut arg = TensorMut::<U8>::new(region(addr, bytes[0] as u64), bytes).unwrap().into_arg();
        arg.context = Some(CONTEXT);
        arg
    }

    /// A call of a convolution is its three tensors and the workspace: checked as a kernel's with no GPU, the shapes of the problem, and a
    /// workspace of at least what a launch needs (for a legacy pin the entry's whole `workspace_bytes`, which the call is given).
    #[test]
    fn a_call_of_a_convolution_is_checked_without_a_gpu() {
        let manifest = manifest_of("k7_dil.T646");
        let entry = manifest.conv_entry("k7_dil.T646", "sm_103").unwrap().clone();
        assert_eq!((entry.pin.api(), entry.problem.output()), ("legacy", vec![664])); // padding 18, dilation 3: T + 18
        let plan = manifest.conv_plan("k7_dil", &[("T", 646)], Arch::sm(103)).unwrap();
        let checks = |required: u64| ConvChecks {
            stem: &entry.stem,
            plan: &plan,
            device: &DEVICE,
            problem: &entry.problem,
            required,
            workspace_bytes: entry.workspace_bytes,
        };
        let (x, w, y, ws) = (0x7000_0000_0000u64, 0x7000_1000_0000u64, 0x7000_2000_0000u64, 0x7000_3000_0000u64);
        let limit = entry.workspace_bytes as usize;
        let (full, short, tiny) = ([limit], [limit - 1], [16usize]);
        let call = |ws_shape: &'static [usize], y_shape: &'static [usize]| {
            vec![
                f16(x, &[1, 646, 64], Access::In),
                f16(w, &[64, 7, 64], Access::In),
                f16(y, y_shape, Access::Out),
                workspace(ws, ws_shape),
            ]
        };
        let (full, short, tiny): (&'static [usize], &'static [usize], &'static [usize]) =
            (Box::leak(Box::new(full)), Box::leak(Box::new(short)), Box::leak(Box::new(tiny)));
        let ok = checks(entry.workspace_bytes).check_call(&call(full, &[1, 664, 64])).unwrap();
        assert_eq!((ok.x, ok.w, ok.out, ok.workspace), (x, w, y, ws));
        let e = checks(entry.workspace_bytes).check_call(&call(short, &[1, 664, 64])).unwrap_err();
        assert_eq!(e.precondition_kind(), Some(PreconditionKind::Shape));
        assert!(
            e.to_string()
                .contains("the workspace has 67108863 bytes, but the cuDNN pin of 'k7_dil.T646' needs 67108864"),
            "{e}"
        );
        let e = checks(entry.workspace_bytes).check_call(&call(full, &[1, 652, 64])).unwrap_err();
        assert!(e.precondition_kind().is_some(), "an out of another length than T + 18: {e}");
        let mut three = call(full, &[1, 664, 64]);
        three.pop();
        assert!(
            matches!(checks(0).check_call(&three).unwrap_err(), Error::Launch(m) if m.contains("takes 4 tensors (x_, w_, out, workspace), got 3"))
        );
        // a pin that needs no workspace is given none (a null pointer), whatever the call has
        assert_eq!(checks(0).check_call(&call(tiny, &[1, 664, 64])).unwrap().workspace, 0);
        // a host on a partition takes only the pins chosen for it
        assert!(partition_matches(&entry, 30).is_ok());
        assert!(
            partition_matches(&entry, 16).unwrap_err().to_string().contains("was chosen for the whole device (30 SMs)")
        );
    }

    #[test]
    fn a_one_dimensional_problem_is_posed_as_two_with_a_first_axis_of_one() {
        let p = ConvProblem {
            dtype: DType::F16,
            layout: ConvLayout::Nhwc,
            n: 1,
            c: 64,
            k: 64,
            groups: 1,
            input: vec![646],
            filter: vec![7],
            stride: vec![1],
            padding: vec![6],
            dilation: vec![1],
            math: ConvMath::TensorOp,
            alignment: 16,
        }
        .checked()
        .unwrap();
        assert_eq!(p.output(), [652]);
        assert_eq!((p.x_dims(), p.x_strides()), (vec![1, 64, 1, 646], vec![41344, 1, 41344, 64]));
        assert_eq!((p.w_dims(), p.w_strides()), (vec![64, 64, 1, 7], vec![448, 1, 448, 64]));
        assert_eq!((p.y_dims(), p.y_strides()), (vec![1, 64, 1, 652], vec![41728, 1, 41728, 64]));
        assert_eq!((p.conv_padding(), p.conv_stride(), p.conv_dilation()), (vec![0, 6], vec![1, 1], vec![1, 1]));
        assert_eq!(p.tilefish_shape("y"), [1, 652, 64]);
    }

    #[test]
    fn the_rules_speak_pythons_words() {
        assert_eq!(
            notes_problem(&[0, 4], ConvMath::TensorOp).unwrap(),
            "it is not deterministic (the numerical note NONDETERMINISTIC)"
        );
        assert!(notes_problem(&[0], ConvMath::Fma).unwrap().contains("the math is 'fma'"));
        assert_eq!(notes_problem(&[0], ConvMath::Tf32), None);
        assert_eq!(knobs_problem(&[(2, 4)], &[(2, 0, 4, 1), (3, 0, 1, 1)]), None);
        assert_eq!(
            knobs_problem(&[(2, 5)], &[(2, 0, 4, 1)]).unwrap(),
            "knob 2 = 5 is outside the engine's 0..4 (stride 1)"
        );
        assert_eq!(
            knobs_problem(&[(7, 0)], &[(3, 0, 1, 1), (2, 0, 4, 1)]).unwrap(),
            "knob type 7 is not a knob of the engine (it has [2, 3])"
        );
        assert!(
            math_problem(DType::F32, ConvMath::TensorOp)
                .unwrap()
                .starts_with("f32 data needs the math said explicitly")
        );
        assert!(
            refuse_tf32_override(Some("0".into())).unwrap_err().starts_with("NVIDIA_TF32_OVERRIDE is set (to '0')")
        );
        assert!(refuse_tf32_override(None).is_ok());
    }
}
