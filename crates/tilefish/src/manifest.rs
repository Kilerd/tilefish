//! Reading a manifest directory: the cubins, checked against what the manifest says.
//!
//! A port of `python/src/tilefish/_manifest.py` (`Manifest.load`) and of the schema checks of `_specvalid.py`, for manifest schema 3
//! (`docs/manifest-schema-3.md`). A manifest is read as strict JSON (see [`crate::json`]), every field has the type it needs (an object is
//! an object, never the list of its values) and no key is unknown, and the entries are checked for being consistent with themselves: the ABI (pointers, strided views, device scalars,
//! scalars, `fixed` arguments, scratch slots), the preconditions, the buckets, the grid. Then each cubin is read, and refused unless
//! its `sha256` (the file) and `sha256_code` (the machine code, see [`crate::elf`]) are the manifest's, it has the code section of
//! the manifest's symbol, and it is for the architecture the entry says. A Triton kernel that does not use shared memory in the
//! manifest's account must not have a shared memory section.
//!
//! The `library` section (pinned cuBLASLt GEMMs, no files) is read by [`crate::library`].
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

use crate::conv::{CONV_IMPL, CachedConv, ConvEntry, RawConv};
use crate::elf::{Elf, sha256_hex};
use crate::error::{Error, PreconditionKind, Result};
use crate::expr::{Expr, Symbols, parse_expr, parse_grid};
use crate::gemm::CachedGemm;
use crate::json::{Members, PathStep, canonical, parse_strict_keeping, type_name};
use crate::kernel::Cached;
use crate::library::{LibraryEntry, RawLibrary};
use crate::tensor::DType;

/// The only manifest schema this host reads. Python's `MANIFEST_SCHEMA`. Schema 3 adds to schema 2 (where every pointer records its declared
/// shape and every entry the constraints of its named dimensions) strided views, device scalars, 64-bit scalars, `fixed` arguments,
/// prebuilt binaries (`kind: "binary"`) and the `library` section.
pub const MANIFEST_SCHEMA: u64 = 3;
/// The file a manifest directory has.
pub const MANIFEST_NAME: &str = "manifest.json";
/// Part of what is hashed: another canonical form must change it. Python's `DIGEST_TAG`.
const DIGEST_TAG: &[u8] = b"tilefish.manifest-digest.v3\0";
/// The tag of the digest of any other document than schema 3 (the form of schema 2). Python's `LEGACY_DIGEST_TAG`.
const LEGACY_DIGEST_TAG: &[u8] = b"tilefish.manifest-digest.v2\0";
/// The limit of 32-bit offsets, the only bound a `max_elements` or `max_extent` precondition carries.
pub(crate) const MAX_ELEMENTS: u64 = (1 << 31) - 1;
/// Every pointer of a binary is compiled as this many bytes aligned (a strided pointer may promise only its element's size).
pub(crate) const ALIGNMENT: u64 = 16;
const NUM_WARPS: [u32; 6] = [1, 2, 4, 8, 16, 32];
/// The most warps a block has (1024 threads): the bound of the `total_num_warps` of a warp-specialised kernel. Python's `MAX_WARPS`.
const MAX_WARPS: u32 = 32;
const MAX_STAGES: u32 = 16;
/// The `implementation` of an entry built from the fallback of an operator declared with an implementation per architecture (the
/// implementation declared for an architecture is named by that architecture, a named implementation by its name). Python's
/// `IMPLEMENTATION_FALLBACK`.
pub const IMPLEMENTATION_FALLBACK: &str = "fallback";
/// The most characters the name of a named implementation has (Python's `_specvalid._IMPLEMENTATION_NAME`).
const MAX_IMPLEMENTATION_NAME: usize = 64;
/// The `source_digest_kind` of a prebuilt binary: its `source_digest` is the sha256 of the PTX or cubin file.
const BINARY_DIGEST_KIND: &str = "tilefish.binary.v1";

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

/// The stride a strided pointer accepts along one axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrideEntry {
    /// Fixed: 1, the unit stride of the axis the kernel loads along (only ever the last axis).
    Unit,
    /// A *stride dimension*, bound from the view's stride (the `dim` of the one scalar that carries it; a scalar that several views
    /// share is the stride dimension of each of them, which must all have the stride it binds).
    Dim(String),
}

/// What a pointer that takes a strided view says (`strides` and `offset` of the ABI, which come together).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    /// One entry per axis of the declared shape.
    pub strides: Vec<StrideEntry>,
    /// The name of the pointer's *offset dimension* (`<pointer>_offset`): not carried by a scalar, because the pointer passed to the
    /// kernel is already the address of the view's first element (`base + offset * itemsize`).
    pub offset: String,
}

/// A pointer and the axis whose stride a scalar carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrideOf {
    /// The strided pointer parameter.
    pub arg: String,
    /// Its axis.
    pub axis: usize,
}

/// The integer a runtime scalar is passed as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    /// `i32`, 4 bytes (`0 ..= 2**31 - 1` here: a dimension, an element count or a stride is not negative).
    I32,
    /// `i64`, 8 bytes (a stride of a kernel declared with `index_bits=64`).
    I64,
}

impl ScalarType {
    /// The word a manifest uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
        }
    }

    /// The largest value a runtime scalar of this type may carry.
    pub fn max(self) -> i64 {
        match self {
            Self::I32 => i64::from(i32::MAX),
            Self::I64 => i64::MAX,
        }
    }
}

/// The value of a `fixed` argument, in its type.
#[derive(Debug, Clone, Copy)]
pub enum FixedValue {
    /// An `i32`.
    I32(i32),
    /// An `i64`.
    I64(i64),
    /// An `f32` (the JSON number rounded to the nearest `f32`; compared by its bits).
    F32(f32),
}

impl PartialEq for FixedValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::I32(a), Self::I32(b)) => a == b,
            (Self::I64(a), Self::I64(b)) => a == b,
            (Self::F32(a), Self::F32(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        }
    }
}

impl FixedValue {
    /// The word a manifest uses for the type.
    pub fn type_name(self) -> &'static str {
        match self {
            Self::I32(_) => "i32",
            Self::I64(_) => "i64",
            Self::F32(_) => "f32",
        }
    }

    /// The bytes of the value in a 64-bit `kernelParams` slot: little-endian, so a 32-bit value is in the low four bytes.
    pub(crate) fn bits(self) -> u64 {
        match self {
            Self::I32(v) => u64::from(v as u32),
            Self::I64(v) => v as u64,
            Self::F32(v) => u64::from(v.to_bits()),
        }
    }
}

/// The `CUtensorMap` a TMA descriptor parameter (manifest kind `tensordesc`) is encoded with, besides its tensor: the `tensor_map` of the
/// parameter (manifest schema 3, section 5.5). The interleave is `CU_TENSOR_MAP_INTERLEAVE_NONE` and the element strides are 1, always.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMap {
    /// The `CUtensorMapDataType` (the name of its enumerator: `CU_TENSOR_MAP_DATA_TYPE_FLOAT16`, ...), the one of the parameter's dtype.
    pub data_type: String,
    /// The block of the descriptor: the tile one `load` or `store` of the kernel moves (outermost first), which is in its Triton type.
    pub block: Vec<u32>,
    /// The box of the map (outermost first): the block, with its inner dimension cut to the swizzle span (the kernel copies a wider block
    /// as several boxes). What `cuTensorMapEncodeTiled` is given (innermost first).
    pub box_shape: Vec<u32>,
    /// The swizzle span in bytes: 0 (none), 32, 64 or 128.
    pub swizzle: u32,
    /// The L2 promotion in bytes: 0 (none), 64, 128 or 256.
    pub l2_promotion: u32,
    /// The out-of-bounds fill: `true` for `CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA` (a float reads as NaN), `false` for
    /// `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` (an element outside the tensor reads as zero, and a store there is dropped).
    pub oob_nan: bool,
}

/// The dtypes a TMA descriptor describes, with the `CUtensorMapDataType` of each (its enumerator's name and value).
pub(crate) const TENSOR_MAP_TYPES: [(&str, &str, u32); 8] = [
    ("u8", "CU_TENSOR_MAP_DATA_TYPE_UINT8", 0),
    ("e4m3", "CU_TENSOR_MAP_DATA_TYPE_UINT8", 0),
    ("e5m2", "CU_TENSOR_MAP_DATA_TYPE_UINT8", 0),
    ("i32", "CU_TENSOR_MAP_DATA_TYPE_INT32", 3),
    ("i64", "CU_TENSOR_MAP_DATA_TYPE_INT64", 5),
    ("f16", "CU_TENSOR_MAP_DATA_TYPE_FLOAT16", 6),
    ("f32", "CU_TENSOR_MAP_DATA_TYPE_FLOAT32", 7),
    ("bf16", "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16", 9),
];

impl TensorMap {
    /// The `CUtensorMapDataType` value of the map (`cuda.h`).
    pub fn data_type_value(&self) -> u32 {
        TENSOR_MAP_TYPES.iter().find(|(_, name, _)| *name == self.data_type).map_or(0, |(_, _, v)| *v)
    }

    /// The `CUtensorMapSwizzle` value of the map: 0 none, 1 32 B, 2 64 B, 3 128 B.
    pub fn swizzle_value(&self) -> u32 {
        match self.swizzle {
            32 => 1,
            64 => 2,
            128 => 3,
            _ => 0,
        }
    }

    /// The `CUtensorMapL2promotion` value of the map: 0 none, 1 64 B, 2 128 B, 3 256 B.
    pub fn l2_promotion_value(&self) -> u32 {
        match self.l2_promotion {
            64 => 1,
            128 => 2,
            256 => 3,
            _ => 0,
        }
    }
}

/// What a launch parameter is.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamKind {
    /// A device pointer. With a `view` it takes a strided view of a tensor; with an empty `shape` it is a *device scalar* (one element in
    /// device memory, read by the kernel with `tl.load(ptr)`).
    ///
    /// With a `tensor_map` it is a **TMA descriptor** of the tensor (manifest kind `tensordesc`): a launch is given the tensor, as for a pointer,
    /// and passes the `CUtensorMap` it encodes of it (`cuTensorMapEncodeTiled`: the tensor's address, shape and strides, and the map's box,
    /// swizzle, L2 promotion and fill), by value, then the tensor's shape (`i32` each) and element strides (`i64` each): `1 + 2 * rank` slots
    /// of `kernelParams` ([`Param::slots`]), as Triton passes a host `TensorDescriptor`.
    Pointer {
        /// The element dtype as the manifest names it (`"bf16"`, `"e2m1"`, ...).
        dtype_name: String,
        /// The dtype, when a binary can be run on it (a packed format cannot).
        dtype: Option<DType>,
        /// How the kernel uses it.
        access: Access,
        /// The declared shape: one entry per axis. A launch binds the named dimensions from the sizes of the tensors it is given.
        shape: Vec<ShapeDim>,
        /// The strides and the offset a view of the tensor may have (a pointer without it is a dense, row-major tensor).
        view: Option<View>,
        /// An `out` pointer only: the `in` pointer it may be written over (`Out[..., MayAlias("r")]` in Python), the two given as exactly one
        /// tensor (one address, shape and strides). Any other overlap of a written tensor is refused, as always.
        alias: Option<String>,
        /// A TMA descriptor of the tensor (`TensorDesc(...)` in Python): the map it is encoded with. `None` for a pointer.
        tensor_map: Option<TensorMap>,
    },
    /// A runtime integer: the value of a dimension expression, the element count of a pointer, or one stride of a strided pointer.
    Scalar {
        /// How it is passed.
        ty: ScalarType,
        /// The expression (`n`, `3072*M`); for a stride, the name of the stride dimension.
        dim: String,
        /// The pointer parameter whose element count this scalar is, for a generated kernel.
        count_of: Option<String>,
        /// The strided pointer parameters and the axes whose stride this scalar is: one, or several views that share one stride
        /// (`StrideOf(("q", "k", "v"), 0)`: the q, k and v of a fused tensor, which a kernel reads with one stride pair). Empty for a
        /// scalar that is not a stride.
        stride_of: Vec<StrideOf>,
    },
    /// A scalar whose value is in the manifest (a prebuilt binary's constant argument): the launch writes it at its own slot.
    Fixed {
        /// Its value, in its type.
        value: FixedValue,
    },
    /// A fixed argument per call (a prebuilt binary's `const[i32]` or `const[i64]` with no default): every launch gives its value
    /// ([`crate::Kernel::launch_with`]; a step of a plan records it), which is passed at its own slot.
    Given {
        /// How it is passed.
        ty: ScalarType,
    },
    /// A pointer that every launch passes as null (Python's `Null[dtype]`: a parameter of the kernel that the declaration does not use): the
    /// 64-bit zero at its own slot. No tensor is bound to it, and it has no precondition.
    Null {
        /// The element dtype of the pointer as the manifest names it (its `type` is that dtype's pointer type).
        dtype_name: String,
    },
}

/// One non-constexpr parameter of the launch ABI.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    /// Position in `kernelParams` of its first slot (a TMA descriptor takes `1 + 2 * rank` slots from there: [`Param::slots`]).
    pub slot: usize,
    /// Position among all the kernel's parameters, constexprs included.
    pub index: usize,
    /// The parameter's name.
    pub name: String,
    /// Pointer, scalar or fixed argument.
    pub kind: ParamKind,
}

impl Param {
    /// Whether this is a device scalar: a pointer with the empty shape.
    pub fn is_device_scalar(&self) -> bool {
        matches!(&self.kind, ParamKind::Pointer { shape, .. } if shape.is_empty())
    }

    /// Whether this is a TMA descriptor of a tensor (a pointer with a `tensor_map`: manifest kind `tensordesc`).
    pub fn tensor_map(&self) -> Option<&TensorMap> {
        match &self.kind {
            ParamKind::Pointer { tensor_map, .. } => tensor_map.as_ref(),
            _ => None,
        }
    }

    /// The bytes of each `kernelParams` slot the parameter takes: one slot of [`Param::size`] bytes, or for a TMA descriptor of a tensor of
    /// rank `r` the 128 bytes of its `CUtensorMap`, then `r` of 4 (its shape) and `r` of 8 (its strides).
    pub fn slots(&self) -> Vec<usize> {
        match &self.kind {
            ParamKind::Pointer { tensor_map: Some(_), shape, .. } => std::iter::once(128)
                .chain(std::iter::repeat_n(4, shape.len()))
                .chain(std::iter::repeat_n(8, shape.len()))
                .collect(),
            _ => vec![self.size()],
        }
    }

    /// The bytes the parameter takes in `kernelParams` (its first slot, for a TMA descriptor): 8 for a pointer, 128 for a descriptor, as many
    /// as its type for a scalar or a fixed argument (the `size` of the manifest, which the loaded function is asked to report).
    pub fn size(&self) -> usize {
        match &self.kind {
            ParamKind::Pointer { tensor_map: Some(_), .. } => 128,
            ParamKind::Pointer { .. } | ParamKind::Null { .. } => 8,
            ParamKind::Scalar { ty, .. } => match ty {
                ScalarType::I32 => 4,
                ScalarType::I64 => 8,
            },
            ParamKind::Fixed { value } => match value {
                FixedValue::I32(_) | FixedValue::F32(_) => 4,
                FixedValue::I64(_) => 8,
            },
            ParamKind::Given { ty } => match ty {
                ScalarType::I32 => 4,
                ScalarType::I64 => 8,
            },
        }
    }
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
    /// The pointer passed (the address of the first element of the view) is aligned to `bytes`.
    PointerAlignment {
        /// The pointer parameter.
        arg: String,
        /// Its position among the kernel's parameters.
        index: usize,
        /// The alignment in bytes: 16, or (for a strided pointer only) the size of its element.
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
    /// The view's extent (one more than the largest element offset it reaches, `1 + sum((size - 1) * stride)` with the canonical strides) is at
    /// most `value`: what `max_elements` says of a dense tensor, for a kernel that indexes a strided pointer with `i32`.
    MaxExtent {
        /// The strided pointer parameter.
        arg: String,
        /// Its position among the kernel's parameters.
        index: usize,
        /// The limit (`2**31 - 1`).
        value: u64,
    },
}

/// What the binary asks of the launch besides grid, block and shared memory.
///
/// Read (with serde) from a JSON object with exactly these six fields, by name, and `global_scratch_align` with a global scratch (tilefish
/// 0.0.7, crate 0.0.5: manifest schema 3, section 4.6): a list of the values in their order is refused, as Python refuses it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchAttributes {
    /// Bytes of global scratch each CTA of the kernel addresses: a launch gives it this many bytes per CTA of its grid (the caller's memory, or a
    /// plan's workspace); 0 for a kernel that has none (its scratch slot is a null pointer).
    pub global_scratch_size: u64,
    /// Bytes of profile scratch.
    pub profile_scratch_size: u64,
    /// Programmatic dependent launch.
    pub launch_pdl: bool,
    /// Cluster dimensions: the CTAs of a thread-block cluster along x, y and z (`[1, 1, 1]`: no cluster).
    pub cluster_dims: [u32; 3],
    /// The CTAs of a cluster (the product of `cluster_dims`): a kernel of more than one is launched with `cuLaunchKernelEx` and the cluster
    /// attribute, and its grid counts clusters (a launch has `grid * cluster_dims` blocks).
    pub num_ctas: u32,
    /// A cooperative launch.
    pub cooperative: bool,
    /// The alignment of the global scratch (a power of two), recorded with a global scratch only; `None` for every other kernel, and for one of a
    /// manifest of tilefish 0.0.6, which did not record it (refused at load: export it again).
    pub global_scratch_align: Option<u32>,
}

/// The derived reader of [`LaunchAttributes`] (every field by name, none unknown), which `by_name!` hands an object only.
#[derive(Deserialize)]
#[serde(remote = "LaunchAttributes", deny_unknown_fields)]
struct LaunchFields {
    global_scratch_size: u64,
    profile_scratch_size: u64,
    launch_pdl: bool,
    cluster_dims: [u32; 3],
    num_ctas: u32,
    cooperative: bool,
    #[serde(default)]
    global_scratch_align: Option<u32>,
}

/// The CTAs of a portable thread-block cluster: a launch of more needs CUDA's non-portable cluster size, which this host does not ask for.
pub(crate) const MAX_CLUSTER: u32 = 8;
/// What every global scratch this host gives starts on (an arena offset, a buffer of the caller held to it): an alignment that divides it is met.
pub(crate) const SCRATCH_ALIGNMENT: u64 = 256;

impl LaunchAttributes {
    /// The attributes this host does not set up, as `name=value` strings (Python's `replay_blockers`): profile scratch, programmatic dependent
    /// launch, a cooperative launch, a cluster that is not `num_ctas` CTAs or is larger than the portable 8, and a global scratch whose alignment
    /// the manifest does not record or does not divide 256 bytes. (A cluster of `num_ctas` CTAs is launched with `cuLaunchKernelEx`, and a global
    /// scratch is given from the caller's memory or a plan's workspace: manifest schema 3, sections 4.5 and 4.6.)
    pub fn blockers(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.global_scratch_size != 0 {
            match self.global_scratch_align {
                None | Some(0) => out.push(format!(
                    "global_scratch_size={} of an alignment the manifest does not record (export it again)",
                    self.global_scratch_size
                )),
                Some(align) if !SCRATCH_ALIGNMENT.is_multiple_of(u64::from(align)) => {
                    out.push(format!("global_scratch_size={} aligned to {align} bytes", self.global_scratch_size));
                }
                Some(_) => {}
            }
        }
        if self.profile_scratch_size != 0 {
            out.push(format!("profile_scratch_size={}", self.profile_scratch_size));
        }
        if self.launch_pdl {
            out.push("launch_pdl=True".to_owned());
        }
        let product = self.cluster_dims.iter().map(|c| u64::from(*c)).product::<u64>();
        if product != u64::from(self.num_ctas) || self.num_ctas > MAX_CLUSTER || self.cluster_dims.contains(&0) {
            out.push(format!(
                "cluster_dims=({}, {}, {}), num_ctas={}",
                self.cluster_dims[0], self.cluster_dims[1], self.cluster_dims[2], self.num_ctas
            ));
        }
        if self.cooperative {
            out.push("launch_cooperative_grid=True".to_owned());
        }
        out
    }

    /// The cluster a launch groups its blocks in (`cuLaunchKernelEx`'s cluster attribute), when the kernel has one (`num_ctas` above 1).
    pub fn cluster(&self) -> Option<[u32; 3]> {
        (self.num_ctas > 1).then_some(self.cluster_dims)
    }
}

/// A persistent grid (`persistent` of an entry: manifest schema 3, section 4.3, tilefish 0.0.7, crate 0.0.5): the grid (one entry) is the number
/// of tiles, and a launch runs at most `programs_per_sm * SMs / CTAs per program` programs, with the SM count of the device (or of the partition
/// the host says it runs on, [`Manifest::with_sm_partition`]) read when the kernel is loaded. The kernel walks the tiles with a stride of the programs
/// launched, so one binary serves every device shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Persistent {
    /// The programs a launch runs per SM, at most (1 to 32).
    pub programs_per_sm: u32,
    /// For a kernel whose bytes depend on how many programs run (a persistent reduction): the SM count it was declared for, which a host on
    /// another count refuses at load. `None`: the kernel's bytes are the same for every SM count.
    pub sm_count: Option<u64>,
}

impl Persistent {
    /// The programs a launch of `tiles` runs on `sms` SMs, each program `ctas` CTAs: at most `programs_per_sm` per SM, at least one, at most
    /// `tiles` (Python's `persistent_programs`).
    pub fn programs(&self, tiles: u64, sms: u64, ctas: u64) -> u64 {
        (u64::from(self.programs_per_sm).saturating_mul(sms) / ctas.max(1)).min(tiles).max(1)
    }
}

/// The most programs per SM a persistent grid asks for, and the largest SM count one is declared for (Python's `MAX_PROGRAMS_PER_SM`, and
/// `tilefish.persistent`'s bound).
const MAX_PROGRAMS_PER_SM: u32 = 32;
const MAX_SM_COUNT: u64 = 1 << 20;

/// What a kernel entry is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// A kernel Triton compiled (`num_warps`, `num_stages` and `options` are its compile settings; two scratch slots follow its parameters).
    Triton,
    /// A prebuilt PTX (assembled by `ptxas`) or cubin: launched with exactly its parameters, no scratch slots.
    Binary,
}

impl EntryKind {
    /// The word a manifest uses (`"triton"`, `"binary"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Triton => "triton",
            Self::Binary => "binary",
        }
    }

    /// How many scratch slots a kernel of this kind is launched with after its parameters: 2 for Triton, 0 for a prebuilt binary.
    pub fn scratch_slots(self) -> usize {
        match self {
            Self::Triton => 2,
            Self::Binary => 0,
        }
    }
}

/// How a prebuilt binary was made (`binary` of its entry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinarySource {
    /// A cubin used as it is, for its own architecture.
    Cubin,
    /// PTX assembled by `ptxas` with these flags (the argument vector before the file names).
    Ptx {
        /// The flags, in their spelling (`ptxas` records them in the cubin, so the machine code depends on them).
        ptxas_flags: Vec<String>,
    },
}

/// The CUDA math library functions a generated kernel's bits are (`docs/manifest-schema-3.md` section 4.7, `docs/numerics.md`): in IEEE math
/// `exp` and `silu` are built on `expf`, `rsqrt` is `rsqrtf` and `erf` `erff`, and `sin`, `cos` and `tanh` are libdevice's `sinf`, `cosf` and
/// `tanhf`. Those functions are not the same in every CUDA release (13.2 changed `erff`), so the entry records which it uses and the releases
/// tilefish measured each on; [`Manifest::check_cuda_math`] refuses a manifest whose functions were not measured on a host's release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaMath {
    /// The functions (`["erff", "tanhf"]`), sorted; empty for a kernel that only links libdevice (`gelu`'s `erf`).
    pub functions: Vec<String>,
    /// The CUDA releases on which every listed function, compiled by that release's NVRTC with nvcc's defaults, is the kernel's bits on every
    /// f32 (`["12.8", "12.9", "13.0", "13.1"]`), sorted by release.
    pub cuda: Vec<String>,
    /// The sha256 of the libdevice the kernel was linked with, when it calls a libdevice function.
    pub libdevice: Option<String>,
}

/// One binary of a manifest: its entry, and its cubin whose digests have been checked.
#[derive(Debug, Clone)]
pub struct Entry {
    /// The operator and its bucket values (`rowop.S32`); unique with `arch` (and across the `library` section too).
    pub stem: String,
    /// The operator.
    pub op: String,
    /// A kernel Triton compiled, or a prebuilt binary.
    pub kind: EntryKind,
    /// The architecture the cubin is for (`sm_103`).
    pub arch: String,
    /// The `.target` of the PTX it was assembled from (`""` for a cubin that was used as it is).
    pub ptx_target: String,
    /// The kernel's symbol in the cubin.
    pub symbol: String,
    /// The cubin's file name in the manifest directory.
    pub cubin_name: String,
    /// The non-constexpr parameters, in `kernelParams` order.
    pub params: Vec<Param>,
    /// How many scratch slots (null 64-bit pointers) a launch passes after the parameters: 2 for a Triton kernel, 0 for a prebuilt binary.
    pub scratch_slots: usize,
    /// The grid, an expression over dimension names (`(cdiv(n, 1024),)`).
    pub grid: String,
    /// The thread block a launch takes: for a Triton kernel `32 * num_warps` threads, or `32 * total_num_warps` for a warp-specialised one;
    /// for a prebuilt binary the block it declares.
    pub block: [u32; 3],
    /// The dynamic shared memory a launch passes, in bytes.
    pub shared: u32,
    /// The warps the kernel was compiled with (a Triton kernel's compile option, equal to `options.num_warps`): the warps of a block, or of the
    /// default partition of a warp-specialised kernel.
    pub num_warps: Option<u32>,
    /// The warps of a block of a warp-specialised Triton kernel (Gluon's `gl.warp_specialize`, or Triton's own warp specialisation): its
    /// partitions add warps to the `num_warps` of its default partition, and the build recorded their total, Triton's `ttg.total-num-warps`
    /// (more than `num_warps`, at most 32). `None` for any other kernel.
    pub total_num_warps: Option<u32>,
    /// Pipeline stages the kernel was compiled with (a Triton kernel).
    pub num_stages: Option<u32>,
    /// What the binary asks of the launch.
    pub launch: LaunchAttributes,
    /// The compiled-in constants.
    pub constexprs: Vec<Constexpr>,
    /// The bucket value of each bucketed dimension, in the order of the stem.
    pub buckets: Vec<(String, u64)>,
    /// The named dimensions of the signature and what is known of each, in the order of the file.
    pub dims: Vec<DimInfo>,
    /// What a launch must check.
    pub preconditions: Vec<Precondition>,
    /// Whether FMA contraction was disabled at compile time, when the build said.
    pub enable_fp_fusion: Option<bool>,
    /// The sha256 of the declaration the binary was built from.
    pub declaration_digest: String,
    /// Which implementation of an operator declared with an implementation per architecture (`@tilefish.extern(by_arch=...)`) the binary is:
    /// the entry's own architecture (`"sm_103"`, the one declared for it), `"fallback"`, or the name of a named implementation that several
    /// architectures may share (`"tcgen05"`: an identifier that is not `"fallback"` and does not begin with `sm_` and a digit). `None` for every
    /// other operator. It is in the declaration digest; a host launches the entry of its device's architecture, with that entry's own ABI, so it
    /// needs nothing else from it. A Triton entry or a prebuilt binary may have one.
    pub implementation: Option<String>,
    /// The sha256 of the cubin file.
    pub sha256: String,
    /// The sha256 of its machine code (see [`crate::elf`]).
    pub sha256_code: String,
    /// The Triton version that built it (a Triton kernel).
    pub triton: Option<String>,
    /// The `ptxas` that assembled it (always for a Triton kernel; for a prebuilt binary only when it was made from PTX).
    pub ptxas: Option<String>,
    /// How a prebuilt binary was made.
    pub binary: Option<BinarySource>,
    /// The sha256 of the Triton source of a generated kernel.
    pub source_sha256: Option<String>,
    /// The CUDA math library functions a generated kernel's bits are, the releases they were measured on, and its libdevice (section 4.7).
    pub cuda_math: Option<CudaMath>,
    /// The identity of what was built: the digest of the Triton IR (TTIR without locations) the binary was built from, for generated and
    /// hand-written kernels alike (the TTGIR of Gluon's frontend for a Gluon kernel; Python recomputes it to tell a stale binary; a Rust
    /// host has no Triton and trusts it through the pinned manifest digest), or, for a prebuilt binary, the sha256 of its PTX or cubin file.
    pub source_digest: Option<String>,
    /// How `source_digest` was computed (`tilefish.triton-ir.v1`, `tilefish.gluon-ir.v1` for a Gluon kernel, or `tilefish.binary.v1` for a
    /// prebuilt binary). A Gluon kernel is a `triton` entry like any other: Triton compiled it, and it is launched the same way.
    pub source_digest_kind: Option<String>,
    /// A persistent grid (manifest schema 3, section 4.3): the grid is the tiles, and a launch runs at most `programs_per_sm` programs per SM of
    /// the device (or partition) the kernel was loaded for. `None` for every other kernel.
    pub persistent: Option<Persistent>,
    /// The verified cubin.
    pub cubin: Arc<[u8]>,
}

impl Entry {
    /// The ABI entry of a parameter by name.
    pub fn param(&self, name: &str) -> Option<&Param> {
        self.params.iter().find(|p| p.name == name)
    }
}

/// What a loaded kernel is cached under: its entry, the `CUcontext` it is loaded in, and the SMs a persistent kernel's grid is resolved for (the
/// partition of [`Manifest::with_sm_partition`], or the device's; `None` for any other kernel).
pub(crate) type KernelKey = (usize, usize, Option<u64>);

/// The binaries of a manifest directory, verified when it is loaded.
///
/// Cheap to clone: the clones share the entries and what was loaded from them (a module per context, a GEMM per context), so the plans of one
/// plan set (`PlanSet`, or [`LaunchPlan::load_with_manifest`](crate::LaunchPlan::load_with_manifest)) load each binary once per context.
#[derive(Debug, Clone)]
pub struct Manifest {
    pub(crate) label: String,
    pub(crate) digest: String,
    pub(crate) entries: Arc<Vec<Entry>>,
    pub(crate) library: Arc<Vec<LibraryEntry>>,
    /// The cuDNN convolutions of the `library` section (`impl: "cudnn.conv_fwd"`).
    pub(crate) convs: Arc<Vec<ConvEntry>>,
    pub(crate) kernels: Arc<Mutex<HashMap<KernelKey, Cached>>>,
    pub(crate) gemms: Arc<Mutex<HashMap<(usize, usize), CachedGemm>>>,
    pub(crate) conv_cache: Arc<Mutex<HashMap<(usize, usize), CachedConv>>>,
    /// The partition of the device's SMs the host runs its GEMMs on, when it said (`Manifest::with_sm_partition`).
    pub(crate) sm_partition: Option<u64>,
}

// -- the digest -------------------------------------------------------------------------------------------------------

/// `Python's str(value)` as far as a sort key needs it: the text of a string, `"None"` for anything else (a reader refuses an entry whose
/// stem or architecture is not a string, so no digest of a manifest that loads depends on the rest).
fn sort_key(e: &Value, field: &str) -> String {
    e.get(field).and_then(Value::as_str).unwrap_or("None").to_owned()
}

fn sorted_entries(list: &[Value], what: &str) -> std::result::Result<Vec<Value>, String> {
    if let Some(i) = list.iter().position(|e| !e.is_object()) {
        return Err(format!("{what} {i} is not an object"));
    }
    let mut sorted: Vec<&Value> = list.iter().collect();
    sorted.sort_by_key(|e| (sort_key(e, "stem"), sort_key(e, "arch")));
    Ok(sorted.into_iter().cloned().collect())
}

/// Whether `schema` is the number 3 the way Python's `schema == 3` says: the integer 3 and the float 3.0 are, `true` and `"3"` are not.
fn is_schema_3(schema: &Value) -> bool {
    match schema {
        Value::Number(n) => n.as_i64() == Some(3) || n.as_u64() == Some(3) || (n.is_f64() && n.as_f64() == Some(3.0)),
        _ => false,
    }
}

/// The objects of a manifest whose order means something, as Python's reader takes them: the `buckets` of an entry (its stem names them in
/// their order, and so does [`Entry::buckets`]) and its `dims` (the order of [`Entry::dims`]), of a kernel entry and of a library entry.
fn is_ordered_object(path: &[PathStep]) -> bool {
    matches!(path, [PathStep::Key(section), PathStep::Index(_), PathStep::Key(field)]
        if (section == "kernels" || section == "library") && (field == "buckets" || field == "dims"))
}

/// The sha256 (lowercase hex) of a manifest document: its schema, every kernel entry and every library entry in full.
///
/// Entries are sorted by `(stem, arch)` (kernels and library entries each) and serialized as canonical JSON (see
/// [`crate::json::canonical`]), so the digest does not depend on the order of the entries or on how the file was formatted, and
/// changes with any field of any entry. The top-level `tilefish` version is not part of it. This is Python's `manifest_digest`:
/// for schema 3 it is `sha256(b"tilefish.manifest-digest.v3\0" + canonical({"schema": 3, "kernels": K, "library": L}))`; any other
/// document is digested in the form of schema 2 (`{"schema": <whatever it says>, "kernels": [...]}`, the tag `v2`), which is what a
/// reader of schema 2 computed, and which a reader of schema 3 refuses to load.
pub fn manifest_digest(doc: &Value) -> std::result::Result<String, String> {
    let schema = doc.get("schema").ok_or("the manifest has no schema")?;
    let kernels = doc.get("kernels").and_then(Value::as_array).ok_or("the manifest has no list of kernels")?;
    let mut body = Map::new();
    body.insert("schema".to_owned(), schema.clone());
    body.insert("kernels".to_owned(), Value::Array(sorted_entries(kernels, "kernel entry")?));
    let tag = if is_schema_3(schema) {
        let library =
            doc.get("library").and_then(Value::as_array).ok_or("the manifest has no list of library entries")?;
        body.insert("library".to_owned(), Value::Array(sorted_entries(library, "library entry")?));
        DIGEST_TAG
    } else {
        LEGACY_DIGEST_TAG
    };
    let text = canonical(&Value::Object(body));
    let mut bytes = tag.to_vec();
    bytes.extend_from_slice(text.as_bytes());
    Ok(sha256_hex(&bytes))
}

// -- the typed, strict reading of one entry --------------------------------------------------------------------------------

/// The value a struct of the manifest is read from, refused unless it is a JSON object.
///
/// serde's derived `Deserialize` of a struct also takes a JSON array, the fields by position (`[0, 0, false, [1, 1, 1], 1, false]` for the
/// launch attributes), and Python's reader refuses one (`isinstance(entry, dict)`, `_keys`): a manifest names every field. So every struct a
/// manifest is read into derives its reader as an inherent function (`#[serde(remote = "Self")]`, or a `remote` mirror for a public type)
/// and has its `Deserialize` from `by_name!`, which hands that reader an object only, at every depth. `what` names it in the message.
pub(crate) fn object_value<'de, D: serde::Deserializer<'de>>(d: D, what: &str) -> std::result::Result<Value, D::Error> {
    let value = Value::deserialize(d)?;
    if value.is_object() {
        Ok(value)
    } else {
        Err(serde::de::Error::custom(format!(
            "{what} must be an object (its fields by name), got {}",
            type_name(&value)
        )))
    }
}

/// `Deserialize` for structs that are read from a JSON object only (see `object_value`): `Type => reader, "what it is";`, where the
/// reader is the derived one (`Type::deserialize` of a struct derived with `#[serde(remote = "Self")]`).
macro_rules! by_name {
    ($($ty:ty => $read:path, $what:expr;)*) => {$(
        impl<'de> serde::Deserialize<'de> for $ty {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
                $read($crate::manifest::object_value(d, $what)?).map_err(serde::de::Error::custom)
            }
        }
    )*};
}
pub(crate) use by_name;

by_name! {
    LaunchAttributes => LaunchFields::deserialize, "launch";
    RawPersistent => RawPersistent::deserialize, "persistent";
    RawDoc => RawDoc::deserialize, "the manifest";
    RawEntry => RawEntry::deserialize, "a kernel entry";
    RawAbi => RawAbi::deserialize, "abi";
    RawStrideOf => RawStrideOf::deserialize, "stride_of";
    RawParam => RawParam::deserialize, "a parameter of abi.params";
    RawDim => RawDim::deserialize, "an entry of dims";
    RawScratch => RawScratch::deserialize, "a slot of abi.scratch";
    RawConstexpr => RawConstexpr::deserialize, "an entry of constexprs";
    RawOptions => RawOptions::deserialize, "options";
    RawToolchain => RawToolchain::deserialize, "toolchain";
    RawCudaMath => RawCudaMath::deserialize, "cuda_math";
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
struct RawDoc {
    schema: Value,
    #[serde(default)]
    #[allow(dead_code)]
    tilefish: Option<String>,
    kernels: Vec<Value>,
    library: Vec<Value>,
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
struct RawEntry {
    stem: String,
    op: String,
    kind: String,
    arch: String,
    ptx_target: String,
    symbol: String,
    cubin: String,
    abi: RawAbi,
    grid: String,
    block: [u32; 3],
    shared: u32,
    #[serde(default)]
    num_warps: Option<u32>,
    #[serde(default)]
    total_num_warps: Option<u32>,
    #[serde(default)]
    num_stages: Option<u32>,
    launch: LaunchAttributes,
    constexprs: Vec<RawConstexpr>,
    buckets: Members,
    dims: Members,
    preconditions: Vec<Value>,
    #[serde(default)]
    options: Option<RawOptions>,
    declaration_digest: String,
    #[serde(default)]
    implementation: Option<String>,
    sha256: String,
    sha256_code: String,
    toolchain: RawToolchain,
    #[serde(default)]
    binary: Option<Value>,
    #[serde(default)]
    source_sha256: Option<String>,
    #[serde(default)]
    cuda_math: Option<RawCudaMath>,
    #[serde(default)]
    source_digest: Option<String>,
    #[serde(default)]
    source_digest_kind: Option<String>,
    #[serde(default)]
    persistent: Option<RawPersistent>,
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
struct RawPersistent {
    programs_per_sm: u32,
    #[serde(default)]
    sm_count: Option<u64>,
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
struct RawCudaMath {
    functions: Vec<String>,
    cuda: Vec<String>,
    #[serde(default)]
    libdevice: Option<String>,
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
struct RawAbi {
    params: Vec<RawParam>,
    scratch: Vec<RawScratch>,
    scratch_slots: u64,
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
struct RawStrideOf {
    arg: String,
    axis: u64,
}

/// `stride_of` as a manifest has it: the object of one view, or a list of them for the views that share the stride.
enum RawStrideOfs {
    One(RawStrideOf),
    Many(Vec<RawStrideOf>),
}

impl<'de> serde::Deserialize<'de> for RawStrideOfs {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(d)?;
        let one = <RawStrideOf as serde::Deserialize>::deserialize; // an object only, at every depth (`by_name!`)
        match value {
            Value::Object(_) => one(value).map(Self::One).map_err(serde::de::Error::custom),
            Value::Array(items) => items
                .into_iter()
                .map(one)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map(Self::Many)
                .map_err(serde::de::Error::custom),
            other => Err(serde::de::Error::custom(format!(
                "stride_of must be an object (its fields by name) or a list of them, got {}",
                type_name(&other)
            ))),
        }
    }
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
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
    #[serde(default)]
    strides: Option<Vec<Value>>,
    #[serde(default)]
    offset: Option<Value>,
    #[serde(default)]
    stride_of: Option<RawStrideOfs>,
    #[serde(default)]
    value: Option<Value>,
    #[serde(default)]
    alias: Option<Value>,
    #[serde(default)]
    tensor_map: Option<Value>,
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
struct RawDim {
    #[serde(default)]
    divisible_by: Option<u64>,
    #[serde(default)]
    buckets: Option<Vec<u64>>,
    #[serde(default)]
    bucket: Option<u64>,
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
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
#[serde(remote = "Self", deny_unknown_fields)]
struct RawConstexpr {
    index: usize,
    name: String,
    value: Value,
    #[serde(default)]
    dtype: Option<String>,
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
struct RawOptions {
    num_warps: u32,
    num_stages: u32,
    #[serde(default)]
    enable_fp_fusion: Option<bool>,
    #[serde(default)]
    num_ctas: Option<u32>,
}

#[derive(Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
struct RawToolchain {
    #[serde(default)]
    triton: Option<String>,
    #[serde(default)]
    ptxas: Option<String>,
}

pub(crate) fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The name of a named implementation (`tilefish.impl(..., name="tcgen05")`): an identifier of at most 64 characters that is not
/// `"fallback"` and does not look like an architecture (it does not begin with `sm_` and a digit), in any case. Python's
/// `_specvalid.is_implementation_name`.
pub(crate) fn is_implementation_name(s: &str) -> bool {
    let b = s.as_bytes();
    let looks_like_arch = b.len() > 3 && b[..3].eq_ignore_ascii_case(b"sm_") && b[3].is_ascii_digit();
    is_ident(s)
        && s.len() <= MAX_IMPLEMENTATION_NAME
        && !s.eq_ignore_ascii_case(IMPLEMENTATION_FALLBACK)
        && !looks_like_arch
}

pub(crate) fn is_stem(s: &str) -> bool {
    !s.is_empty() && s.split('.').all(is_ident)
}

pub(crate) fn is_digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

pub(crate) fn is_arch(s: &str) -> bool {
    s.strip_prefix("sm_").is_some_and(|n| (2..=3).contains(&n.len()) && n.bytes().all(|b| b.is_ascii_digit()))
}

/// A kernel symbol of a prebuilt binary (a PTX or ELF symbol: `$` is allowed).
fn is_symbol(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == '$')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
}

/// The `sm_103` of an architecture name as its number.
fn arch_number(arch: &str) -> Option<u32> {
    arch.strip_prefix("sm_")?.parse().ok()
}

/// The Triton spelling of a pointer to each dtype a manifest may name (Python's `ALL_DTYPES`).
pub(crate) fn pointer_type_of(dtype: &str) -> Option<&'static str> {
    Some(match dtype {
        "e2m1" | "ue8m0" => "*u8",
        other => DType::from_name(other)?.pointer_type(),
    })
}

pub(crate) fn strictly<T>(r: std::result::Result<T, String>, who: &str) -> Result<T> {
    r.map_err(|e| Error::manifest(format!("{who}: {e}")))
}

pub(crate) fn want_int(v: &Value, what: &str, minimum: u64) -> std::result::Result<u64, String> {
    match v.as_u64() {
        Some(n) if n >= minimum => Ok(n),
        _ => Err(format!("{what} must be an int >= {minimum}, got {v}")),
    }
}

pub(crate) fn want_ident(v: &Value, what: &str) -> std::result::Result<String, String> {
    match v.as_str() {
        Some(s) if is_ident(s) => Ok(s.to_owned()),
        _ => Err(format!("{what} must be an identifier, got {v}")),
    }
}

pub(crate) fn want_keys(
    obj: &Map<String, Value>,
    what: &str,
    kind: &str,
    allowed: &[&str],
) -> std::result::Result<(), String> {
    for k in obj.keys() {
        if k != "kind" && !allowed.contains(&k.as_str()) {
            return Err(format!("{what} has unknown fields [{k:?}]"));
        }
    }
    let missing: Vec<&&str> = allowed.iter().filter(|k| !obj.contains_key(**k)).collect();
    if missing.is_empty() { Ok(()) } else { Err(format!("{what} ({kind}) is missing {missing:?}")) }
}

pub(crate) fn precondition_of(v: &Value, syms: &mut Symbols) -> std::result::Result<Precondition, String> {
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
        "max_extent" => {
            want_keys(obj, &what, kind, &["arg", "index", "value"])?;
            let value = want_int(get("value"), &format!("{what} value"), 1)?;
            if value != MAX_ELEMENTS {
                return Err(format!(
                    "max_extent precondition value must be {MAX_ELEMENTS} (the limit of i32 offsets), got {value}"
                ));
            }
            Precondition::MaxExtent {
                arg: want_ident(get("arg"), &format!("{what} arg"))?,
                index: usize_of("index", 0)?,
                value,
            }
        }
        other => {
            return Err(format!(
                "precondition kind must be one of [\"bucket\", \"divisible\", \"max_elements\", \"max_extent\", \"pointer_alignment\", \"static_size\"], got {other:?}"
            ));
        }
    })
}

/// The declared shape of a pointer: each axis a size (an int of at least 1) or a dimension expression. Empty for a device scalar.
pub(crate) fn parse_shape(name: &str, shape: &[Value]) -> std::result::Result<Vec<ShapeDim>, String> {
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

/// The table of named dimensions, checked against the buckets of the entry: both in the order of the file (see [`Members`]).
pub(crate) fn parse_dims(dims: &Members, buckets: &Members) -> std::result::Result<Vec<DimInfo>, String> {
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

/// The value of a `fixed` argument of C type `ty` (`i32`, `i64`, `f32`): an integer in range for an integer type (not a bool, not a float),
/// a JSON number with a fraction or exponent (or the tagged form of a non-finite one) for `f32`.
fn fixed_value(ty: &str, v: &Value, who: &str) -> std::result::Result<FixedValue, String> {
    match ty {
        "i32" | "i64" => {
            let wide = match v {
                Value::Number(n) if !n.is_f64() => {
                    n.as_i64().ok_or_else(|| format!("{who}: the value {v} does not fit {ty}"))
                }
                _ => Err(format!("{who}: the value of a fixed {ty} argument must be an int, got {v}")),
            }?;
            if ty == "i32" {
                i32::try_from(wide)
                    .map(FixedValue::I32)
                    .map_err(|_| format!("{who}: the value {wide} does not fit i32"))
            } else {
                Ok(FixedValue::I64(wide))
            }
        }
        "f32" => match v {
            Value::Number(n) if n.is_f64() => {
                let x = n.as_f64().ok_or_else(|| format!("{who}: the value {v} is not a number"))?;
                Ok(FixedValue::F32(x as f32))
            }
            Value::Object(o) if o.len() == 1 => match o.get("float").and_then(Value::as_str) {
                Some("inf") => Ok(FixedValue::F32(f32::INFINITY)),
                Some("-inf") => Ok(FixedValue::F32(f32::NEG_INFINITY)),
                Some("nan") => Ok(FixedValue::F32(f32::NAN)),
                _ => Err(format!("{who}: {v} is not a valid tagged float")),
            },
            _ => Err(format!("{who}: the value of a fixed f32 argument must be a float, got {v}")),
        },
        other => Err(format!("{who}: a fixed argument is one of i32, i64, f32, got {other:?}")),
    }
}

fn str_field<'a>(v: &'a Option<String>, what: &str, name: &str) -> std::result::Result<&'a str, String> {
    v.as_deref().ok_or_else(|| format!("parameter {name}: {what} is missing"))
}

/// The largest dimension of the block of a TMA descriptor (its inner one may be wider than the box), Python's `_tensormap.MAX_BLOCK`.
const MAX_TENSOR_MAP_BLOCK: u64 = 1 << 16;

/// The `tensor_map` of a TMA descriptor parameter of `dtype` and `rank` (Python's `_tensormap.read_json` and `check_map`): exactly its eight
/// fields; the data type of the dtype; a block of powers of two and a box that tiles it, each of `rank` ints; element strides of 1; interleave
/// `NONE`; and what `cuTensorMapEncodeTiled` refuses of the box (1 to 256 elements per axis, an inner dimension of a multiple of 16 bytes and,
/// swizzled, within the swizzle span) and of the fill (NaN for a float only).
fn read_tensor_map(v: Option<&Value>, dtype: &str, rank: usize, name: &str) -> std::result::Result<TensorMap, String> {
    let bad = |what: String| Err(format!("parameter {name}: {what}"));
    let Some(Value::Object(m)) = v else {
        return bad(format!(
            "tensor_map must be an object, got {}",
            v.map_or("nothing".to_owned(), ToString::to_string)
        ));
    };
    const KEYS: [&str; 8] =
        ["data_type", "block", "box", "element_strides", "interleave", "swizzle", "l2_promotion", "oob_fill"];
    if m.len() != KEYS.len() || KEYS.iter().any(|k| !m.contains_key(*k)) {
        return bad(format!(
            "tensor_map must have exactly the fields {KEYS:?}, has {:?}",
            m.keys().collect::<Vec<_>>()
        ));
    }
    let Some((_, data_type, _)) = TENSOR_MAP_TYPES.iter().find(|(d, _, _)| *d == dtype) else {
        return bad(format!("a tensor descriptor of {dtype:?} is not supported"));
    };
    if m["data_type"].as_str() != Some(data_type) {
        return bad(format!("tensor_map.data_type of a {dtype} tensor is {data_type}, got {}", m["data_type"]));
    }
    let ints = |key: &str| -> std::result::Result<Vec<u64>, String> {
        let list = m[key].as_array().filter(|a| a.len() == rank);
        list.and_then(|a| a.iter().map(|x| if x.is_u64() { x.as_u64() } else { None }).collect::<Option<Vec<u64>>>())
            .ok_or_else(|| format!("parameter {name}: tensor_map.{key} must be a list of {rank} ints (one per axis of the shape), got {}", m[key]))
    };
    let block = ints("block")?;
    let box_shape = ints("box")?;
    if ints("element_strides")?.iter().any(|e| *e != 1) {
        return bad(format!(
            "tensor_map.element_strides must be all 1 (Triton's kernels take element strides of 1), got {}",
            m["element_strides"]
        ));
    }
    if m["interleave"].as_str() != Some("CU_TENSOR_MAP_INTERLEAVE_NONE") {
        return bad(format!("tensor_map.interleave must be CU_TENSOR_MAP_INTERLEAVE_NONE, got {}", m["interleave"]));
    }
    let swizzle = match m["swizzle"].as_str() {
        Some("CU_TENSOR_MAP_SWIZZLE_NONE") => 0,
        Some("CU_TENSOR_MAP_SWIZZLE_32B") => 32,
        Some("CU_TENSOR_MAP_SWIZZLE_64B") => 64,
        Some("CU_TENSOR_MAP_SWIZZLE_128B") => 128,
        _ => {
            return bad(format!(
                "tensor_map.swizzle must be one of CU_TENSOR_MAP_SWIZZLE_NONE/_32B/_64B/_128B, got {}",
                m["swizzle"]
            ));
        }
    };
    let l2_promotion = match m["l2_promotion"].as_str() {
        Some("CU_TENSOR_MAP_L2_PROMOTION_NONE") => 0,
        Some("CU_TENSOR_MAP_L2_PROMOTION_L2_64B") => 64,
        Some("CU_TENSOR_MAP_L2_PROMOTION_L2_128B") => 128,
        Some("CU_TENSOR_MAP_L2_PROMOTION_L2_256B") => 256,
        _ => {
            return bad(format!(
                "tensor_map.l2_promotion must be one of CU_TENSOR_MAP_L2_PROMOTION_NONE/_L2_64B/_L2_128B/_L2_256B, got {}",
                m["l2_promotion"]
            ));
        }
    };
    let oob_nan = match m["oob_fill"].as_str() {
        Some("CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE") => false,
        Some("CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA") => true,
        _ => {
            return bad(format!(
                "tensor_map.oob_fill must be CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE or _NAN_REQUEST_ZERO_FMA, got {}",
                m["oob_fill"]
            ));
        }
    };
    if !(1..=5).contains(&rank) {
        return bad(format!("a tensor descriptor describes a tensor of rank 1 to 5, this one has rank {rank}"));
    }
    for (axis, b) in block.iter().enumerate() {
        if !(1..=MAX_TENSOR_MAP_BLOCK).contains(b) || !b.is_power_of_two() {
            return bad(format!(
                "block[{axis}] is {b}: a block dimension is a power of two (Triton's rule) from 1 to {MAX_TENSOR_MAP_BLOCK}"
            ));
        }
    }
    for (axis, (n, b)) in box_shape.iter().zip(&block).enumerate() {
        if !(1..=256).contains(n) {
            return bad(format!(
                "the box of the descriptor is {box_shape:?}, but a box dimension is 1 to 256 elements (axis {axis})"
            ));
        }
        if b % n != 0 {
            return bad(format!("the box {box_shape:?} does not tile the block {block:?} (axis {axis})"));
        }
    }
    let item = DType::from_name(dtype).map_or(1, |d| d.size() as u64);
    let inner = box_shape[rank - 1] * item;
    if !inner.is_multiple_of(16) {
        return bad(format!(
            "the inner dimension of the box is {inner} bytes, which is not a multiple of 16 (cuTensorMapEncodeTiled refuses it)"
        ));
    }
    if swizzle != 0 && inner > swizzle {
        return bad(format!(
            "the inner dimension of the box is {inner} bytes, more than the {swizzle}-byte swizzle span"
        ));
    }
    if oob_nan && !matches!(dtype, "f16" | "bf16" | "f32") {
        return bad(format!("oob_fill NaN fills with a float NaN, and {dtype} is not a float type of a descriptor"));
    }
    let to_u32 = |v: Vec<u64>| v.into_iter().map(|x| x as u32).collect(); // (each at most 2^16: checked above)
    Ok(TensorMap {
        data_type: (*data_type).to_owned(),
        block: to_u32(block),
        box_shape: to_u32(box_shape),
        swizzle: swizzle as u32,
        l2_promotion,
        oob_nan,
    })
}

/// The ABI parameters of an entry, each checked on its own (the kinds, the types and their widths, which fields each has).
fn read_params(raw: &RawEntry, kind: EntryKind) -> std::result::Result<Vec<Param>, String> {
    let mut out = Vec::with_capacity(raw.abi.params.len());
    let mut next_slot = 0usize; // where the slots of the next parameter begin: a TMA descriptor takes 1 + 2 * rank, every other one 1
    for (position, p) in raw.abi.params.iter().enumerate() {
        want_ident(&Value::String(p.name.clone()), "parameter name")?;
        if p.slot != next_slot {
            return Err(format!(
                "parameter {} has slot {}, but its slots begin at {next_slot} (position {position}; a descriptor takes 1 + 2 * rank slots)",
                p.name, p.slot
            ));
        }
        next_slot += match (p.kind.as_str(), &p.shape) {
            ("tensordesc", Some(shape)) => 1 + 2 * shape.len(),
            _ => 1,
        };
        let name = p.name.as_str();
        if p.tensor_map.is_some() && p.kind != "tensordesc" {
            return Err(format!(
                "parameter {name} is a {} and has a tensor_map: only a tensor descriptor (kind tensordesc) has one",
                p.kind
            ));
        }
        let kind_of = match p.kind.as_str() {
            "pointer" | "tensordesc" => {
                let desc = p.kind == "tensordesc";
                let dtype = p.dtype.as_deref().unwrap_or("");
                if desc {
                    if kind != EntryKind::Triton {
                        return Err(format!(
                            "parameter {name}: a tensor descriptor belongs to a Triton (or Gluon) kernel, and this is a {} kernel",
                            kind.as_str()
                        ));
                    }
                    if p.ty != "tensordesc" || p.ptx_type != "b8[128]" || p.size != 128 {
                        return Err(format!(
                            "parameter {name}: a tensor descriptor has type 'tensordesc' and is passed as the 128 bytes of a CUtensorMap \
                             (ptx_type 'b8[128]', size 128)"
                        ));
                    }
                    if !TENSOR_MAP_TYPES.iter().any(|(d, _, _)| *d == dtype) {
                        return Err(format!(
                            "parameter {name}: a tensor descriptor describes one of {:?}, got {dtype:?}",
                            TENSOR_MAP_TYPES.iter().map(|(d, _, _)| *d).collect::<Vec<_>>()
                        ));
                    }
                } else if pointer_type_of(dtype) != Some(p.ty.as_str()) {
                    return Err(format!("parameter {name}: pointer type {:?} does not match dtype {dtype:?}", p.ty));
                }
                let access = match p.access.as_deref() {
                    Some("in") => Access::In,
                    Some("inout") => Access::InOut,
                    Some("out") => Access::Out,
                    other => return Err(format!("parameter {name}: access must be in, inout or out, got {other:?}")),
                };
                if desc
                    && (p.dim.is_some()
                        || p.count_of.is_some()
                        || p.stride_of.is_some()
                        || p.value.is_some()
                        || p.alias.is_some())
                {
                    return Err(format!(
                        "parameter {name} is a tensor descriptor: it has the fields of a pointer and a tensor_map, and no dim, count_of, \
                         stride_of, value or alias"
                    ));
                }
                if p.dim.is_some()
                    || p.count_of.is_some()
                    || p.stride_of.is_some()
                    || p.value.is_some()
                    || (!desc && (p.ptx_type != "u64" || p.size != 8))
                {
                    return Err(format!("parameter {name} is not a well-formed 64-bit pointer"));
                }
                // an out pointer that may be written over an in pointer (checked against that pointer by `check_consistent`)
                let alias = match &p.alias {
                    None => None,
                    Some(Value::String(a)) if access == Access::Out => Some(a.clone()),
                    Some(Value::String(_)) => {
                        return Err(format!(
                            "arg {name}: only an out pointer may alias an input (it is {:?})",
                            access.as_str()
                        ));
                    }
                    Some(other) => return Err(format!("parameter {name}: alias must be a str, got {other}")),
                };
                let Some(shape) = &p.shape else {
                    return Err(format!("parameter {name} has no shape: a pointer records its declared shape"));
                };
                let shape = parse_shape(name, shape)?;
                let view = match (&p.strides, &p.offset) {
                    (None, None) => None,
                    (Some(strides), Some(offset)) => {
                        if shape.is_empty() || strides.len() != shape.len() {
                            return Err(format!(
                                "parameter {name}: strides must be a list with one entry per axis of shape (at least one axis), got {} for {} axes",
                                strides.len(),
                                shape.len()
                            ));
                        }
                        if access != Access::In {
                            return Err(format!(
                                "parameter {name} is {} and takes strides, but only an `in` pointer may be strided",
                                access.as_str()
                            ));
                        }
                        let mut entries = Vec::with_capacity(strides.len());
                        for (axis, s) in strides.iter().enumerate() {
                            entries.push(match s {
                                Value::Number(n) if !n.is_f64() => {
                                    if n.as_u64() != Some(1) {
                                        return Err(format!(
                                            "the stride of axis {axis} of '{name}' is the fixed size {s}: a fixed stride is 1 (and only on the last axis)"
                                        ));
                                    }
                                    if axis + 1 != strides.len() {
                                        return Err(format!(
                                            "the stride of axis {axis} of '{name}' is fixed, but only the last axis may be"
                                        ));
                                    }
                                    StrideEntry::Unit
                                }
                                Value::String(text) => StrideEntry::Dim(text.clone()),
                                other => {
                                    return Err(format!(
                                        "parameter {name}: strides[{axis}] must be an int or a dimension expression, got {other}"
                                    ));
                                }
                            });
                        }
                        let offset =
                            want_ident(offset, &format!("the offset of '{name}' (the name of the offset dimension)"))?;
                        if desc {
                            // a descriptor of a view: its own stride dimensions (names, passed in its stride slots), the last axis's fixed 1
                            let last = entries.len() - 1;
                            for (axis, s) in entries.iter().enumerate() {
                                let ok = match s {
                                    StrideEntry::Unit => axis == last,
                                    StrideEntry::Dim(text) => axis != last && is_ident(text),
                                };
                                if !ok {
                                    return Err(format!(
                                        "the stride of axis {axis} of the descriptor '{name}' must be {}",
                                        if axis == last {
                                            "the fixed 1 (a TMA descriptor reads its inner dimension contiguously)"
                                        } else {
                                            "the name of its stride dimension"
                                        }
                                    ));
                                }
                            }
                        }
                        Some(View { strides: entries, offset })
                    }
                    _ => return Err(format!("parameter {name}: strides and offset come together")),
                };
                let tensor_map = if desc {
                    if !(1..=5).contains(&shape.len()) {
                        return Err(format!(
                            "parameter {name}: a tensor descriptor has a shape of 1 to 5 axes, got {}",
                            shape.len()
                        ));
                    }
                    Some(read_tensor_map(p.tensor_map.as_ref(), dtype, shape.len(), name)?)
                } else {
                    None
                };
                ParamKind::Pointer {
                    dtype: DType::from_name(dtype),
                    dtype_name: dtype.to_owned(),
                    access,
                    shape,
                    view,
                    alias,
                    tensor_map,
                }
            }
            "scalar" => {
                let (ty, ptx, size) = match p.ty.as_str() {
                    "i32" => (ScalarType::I32, "u32", 4),
                    "i64" => (ScalarType::I64, "u64", 8),
                    other => {
                        return Err(format!("parameter {name}: a runtime scalar must be i32 or i64, got {other:?}"));
                    }
                };
                if p.ptx_type != ptx || p.size != size {
                    return Err(format!(
                        "parameter {name}: an {} scalar is passed as a {ptx} of {size} bytes, got {} of {}",
                        p.ty, p.ptx_type, p.size
                    ));
                }
                if p.dtype.is_some()
                    || p.access.is_some()
                    || p.shape.is_some()
                    || p.strides.is_some()
                    || p.offset.is_some()
                    || p.value.is_some()
                    || p.alias.is_some()
                {
                    return Err(format!(
                        "parameter {name} is a scalar: it has a dim and no dtype, access, shape, strides or value"
                    ));
                }
                if p.count_of.is_some() && p.stride_of.is_some() {
                    return Err(format!(
                        "parameter {name}: a scalar is the element count of a pointer or the stride of one, not both"
                    ));
                }
                let dim = str_field(&p.dim, "dim", name)?.to_owned();
                let items: &[RawStrideOf] = match &p.stride_of {
                    None => &[],
                    Some(RawStrideOfs::One(s)) => std::slice::from_ref(s),
                    Some(RawStrideOfs::Many(list)) if list.len() < 2 => {
                        return Err(format!(
                            "parameter {name}: stride_of is a list of the views that share the stride, so it has two or more of them (one \
                             view is an object)"
                        ));
                    }
                    Some(RawStrideOfs::Many(list)) => list,
                };
                let mut stride_of: Vec<StrideOf> = Vec::with_capacity(items.len());
                for s in items {
                    want_ident(&Value::String(s.arg.clone()), "stride_of arg")?;
                    let axis = usize::try_from(s.axis)
                        .map_err(|_| format!("parameter {name}: stride_of axis is too large"))?;
                    let target = StrideOf { arg: s.arg.clone(), axis };
                    if stride_of.contains(&target) {
                        return Err(format!("parameter {name}: stride_of names axis {axis} of '{}' twice", s.arg));
                    }
                    stride_of.push(target);
                }
                ParamKind::Scalar { ty, dim, count_of: p.count_of.clone(), stride_of }
            }
            "fixed" => {
                if kind != EntryKind::Binary {
                    return Err(format!(
                        "parameter {name}: a fixed argument belongs to a prebuilt binary, and this is a {} kernel",
                        kind.as_str()
                    ));
                }
                let (ptx, size) = match p.ty.as_str() {
                    "i32" => ("u32", 4),
                    "i64" => ("u64", 8),
                    "f32" => ("f32", 4),
                    other => {
                        return Err(format!(
                            "parameter {name}: a fixed argument is one of i32, i64, f32, got {other:?}"
                        ));
                    }
                };
                if p.ptx_type != ptx || p.size != size {
                    return Err(format!(
                        "parameter {name}: a fixed {} is passed as a {ptx} of {size} bytes, got {} of {}",
                        p.ty, p.ptx_type, p.size
                    ));
                }
                if p.dim.is_some()
                    || p.count_of.is_some()
                    || p.stride_of.is_some()
                    || p.access.is_some()
                    || p.shape.is_some()
                    || p.strides.is_some()
                    || p.offset.is_some()
                    || p.dtype.is_some()
                    || p.alias.is_some()
                {
                    return Err(format!("parameter {name} is a fixed argument: it has a value and no dim or shape"));
                }
                let Some(value) = &p.value else {
                    return Err(format!("parameter {name} is a fixed argument and has no value"));
                };
                ParamKind::Fixed { value: fixed_value(&p.ty, value, &format!("parameter {name}"))? }
            }
            "given" => {
                // a fixed argument per call: every launch (every step of a plan) gives its value
                if kind != EntryKind::Binary {
                    return Err(format!(
                        "parameter {name}: a fixed argument per call belongs to a prebuilt binary, and this is a {} kernel",
                        kind.as_str()
                    ));
                }
                let (ty, ptx, size) = match p.ty.as_str() {
                    "i32" => (ScalarType::I32, "u32", 4),
                    "i64" => (ScalarType::I64, "u64", 8),
                    other => {
                        return Err(format!(
                            "parameter {name}: a fixed argument per call is i32 or i64, got {other:?}"
                        ));
                    }
                };
                if p.ptx_type != ptx || p.size != size {
                    return Err(format!(
                        "parameter {name}: a {} given per call is passed as a {ptx} of {size} bytes, got {} of {}",
                        p.ty, p.ptx_type, p.size
                    ));
                }
                if p.dim.is_some()
                    || p.count_of.is_some()
                    || p.stride_of.is_some()
                    || p.access.is_some()
                    || p.shape.is_some()
                    || p.strides.is_some()
                    || p.offset.is_some()
                    || p.dtype.is_some()
                    || p.alias.is_some()
                    || p.value.is_some()
                {
                    return Err(format!(
                        "parameter {name} is a fixed argument per call: it has no value, dim or shape (each call gives its value)"
                    ));
                }
                ParamKind::Given { ty }
            }
            "null" => {
                // a pointer every launch passes as null: its dtype, whose pointer type is its type, and nothing else
                let dtype = p.dtype.as_deref().unwrap_or("");
                if pointer_type_of(dtype) != Some(p.ty.as_str()) {
                    return Err(format!(
                        "parameter {name}: the type of a null pointer {:?} is not the pointer type of its dtype {dtype:?}",
                        p.ty
                    ));
                }
                if p.ptx_type != "u64" || p.size != 8 {
                    return Err(format!(
                        "parameter {name}: a null pointer is passed as a u64 of 8 bytes, got {} of {}",
                        p.ptx_type, p.size
                    ));
                }
                if p.dim.is_some()
                    || p.count_of.is_some()
                    || p.stride_of.is_some()
                    || p.access.is_some()
                    || p.shape.is_some()
                    || p.strides.is_some()
                    || p.offset.is_some()
                    || p.value.is_some()
                    || p.alias.is_some()
                {
                    return Err(format!(
                        "parameter {name} is a null pointer: it has a dtype and nothing a tensor or a scalar has"
                    ));
                }
                ParamKind::Null { dtype_name: dtype.to_owned() }
            }
            other => {
                return Err(format!(
                    "parameter {name}: kind must be pointer, tensordesc, scalar, fixed, given or null, got {other:?}"
                ));
            }
        };
        out.push(Param { slot: p.slot, index: p.index, name: p.name.clone(), kind: kind_of });
    }
    Ok(out)
}

fn read_binary(v: &Value) -> std::result::Result<BinarySource, String> {
    let bad = || {
        format!("binary must be {{\"source\": \"cubin\"}} or {{\"source\": \"ptx\", \"ptxas_flags\": [...]}}, got {v}")
    };
    let obj = v.as_object().ok_or_else(bad)?;
    match obj.get("source").and_then(Value::as_str) {
        Some("cubin") if obj.len() == 1 => Ok(BinarySource::Cubin),
        Some("ptx") if obj.len() == 2 => {
            let flags = obj
                .get("ptxas_flags")
                .and_then(Value::as_array)
                .and_then(|a| a.iter().map(|f| f.as_str().map(str::to_owned)).collect::<Option<Vec<String>>>())
                .ok_or_else(bad)?;
            Ok(BinarySource::Ptx { ptxas_flags: flags })
        }
        _ => Err(bad()),
    }
}

/// A CUDA release as `cuda_math` writes it (`"13.0"`): two decimal numbers without leading zeros, and their value.
fn cuda_release(s: &str) -> Option<(u64, u64)> {
    let (major, minor) = s.split_once('.')?;
    let number = |p: &str| {
        (!p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) && (p == "0" || !p.starts_with('0')))
            .then(|| p.parse().ok())
            .flatten()
    };
    Some((number(major)?, number(minor)?))
}

/// The rules of section 4.7 that Python's `_cudamath.check_field` holds too.
fn check_cuda_math(math: &RawCudaMath) -> std::result::Result<(), String> {
    let name_ok = |f: &String| {
        let mut chars = f.chars();
        chars.next().is_some_and(|c| c.is_ascii_lowercase() || c == '_')
            && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    };
    if !math.functions.iter().all(name_ok) {
        return Err(format!("cuda_math.functions must be a list of function names, got {:?}", math.functions));
    }
    if !math.functions.windows(2).all(|w| w[0] < w[1]) {
        return Err(format!("cuda_math.functions must be sorted, each once, got {:?}", math.functions));
    }
    let releases = math.cuda.iter().map(|r| cuda_release(r)).collect::<Option<Vec<_>>>();
    let Some(releases) = releases else {
        return Err(format!("cuda_math.cuda must be a list of CUDA releases such as \"13.0\", got {:?}", math.cuda));
    };
    if !releases.windows(2).all(|w| w[0] < w[1]) {
        return Err(format!("cuda_math.cuda must be sorted by release, each once, got {:?}", math.cuda));
    }
    if math.functions.is_empty() && !math.cuda.is_empty() {
        return Err(format!("cuda_math.cuda lists releases {:?} for no function", math.cuda));
    }
    if let Some(sha) = math.libdevice.as_ref().filter(|d| !is_digest(d)) {
        return Err(format!("cuda_math.libdevice must be a sha256 (64 lowercase hex digits), got {sha:?}"));
    }
    if math.functions.is_empty() && math.libdevice.is_none() {
        return Err(
            "cuda_math names no function and no libdevice (an entry without either has no cuda_math)".to_owned()
        );
    }
    Ok(())
}

fn entry_kind(raw: &RawEntry) -> std::result::Result<EntryKind, String> {
    match raw.kind.as_str() {
        "triton" => Ok(EntryKind::Triton),
        "binary" => Ok(EntryKind::Binary),
        other => Err(format!("kind must be 'triton' or 'binary', got {other:?}")),
    }
}

impl Entry {
    fn from_raw(raw: RawEntry, cubin: Arc<[u8]>) -> Result<Self> {
        let who = format!("'{}' ({})", raw.stem, raw.arch);
        let kind = strictly(entry_kind(&raw), &who)?;
        let dims = strictly(parse_dims(&raw.dims, &raw.buckets), &who)?;
        let mut syms = Symbols::default();
        for d in &dims {
            strictly(syms.intern(&d.name), &who)?;
        }
        let params = strictly(read_params(&raw, kind), &who)?;
        let mut preconditions = Vec::with_capacity(raw.preconditions.len());
        for p in &raw.preconditions {
            preconditions.push(strictly(precondition_of(p, &mut syms), &who)?);
        }
        let binary = match &raw.binary {
            Some(b) => Some(strictly(read_binary(b), &who)?),
            None => None,
        };
        let buckets = raw.buckets.iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0))).collect::<Vec<_>>();
        let entry = Self {
            stem: raw.stem,
            op: raw.op,
            kind,
            arch: raw.arch,
            ptx_target: raw.ptx_target,
            symbol: raw.symbol,
            cubin_name: raw.cubin,
            params,
            scratch_slots: raw.abi.scratch.len(),
            grid: raw.grid,
            block: raw.block,
            shared: raw.shared,
            num_warps: raw.num_warps,
            total_num_warps: raw.total_num_warps,
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
            enable_fp_fusion: raw.options.as_ref().and_then(|o| o.enable_fp_fusion),
            declaration_digest: raw.declaration_digest,
            implementation: raw.implementation,
            sha256: raw.sha256,
            sha256_code: raw.sha256_code,
            triton: raw.toolchain.triton,
            ptxas: raw.toolchain.ptxas,
            binary,
            source_sha256: raw.source_sha256,
            cuda_math: raw.cuda_math.map(|c| CudaMath { functions: c.functions, cuda: c.cuda, libdevice: c.libdevice }),
            source_digest: raw.source_digest,
            source_digest_kind: raw.source_digest_kind,
            persistent: raw.persistent.map(|p| Persistent { programs_per_sm: p.programs_per_sm, sm_count: p.sm_count }),
            cubin,
        };
        strictly(entry.check_consistent(&mut syms), &who)?;
        Ok(entry)
    }

    /// The checks that need no more than the raw fields.
    fn check_raw(raw: &RawEntry) -> std::result::Result<(), String> {
        let kind = entry_kind(raw)?;
        if !is_stem(&raw.stem) {
            return Err(format!("stem must look like 'op' or 'op.S32', got {:?}", raw.stem));
        }
        if !is_ident(&raw.op) {
            return Err(format!("op must be an identifier, got {:?}", raw.op));
        }
        if !is_arch(&raw.arch) {
            return Err(format!("arch must look like 'sm_103', got {:?}", raw.arch));
        }
        let symbol_ok = if kind == EntryKind::Binary { is_symbol(&raw.symbol) } else { is_ident(&raw.symbol) };
        if !symbol_ok {
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
        if let Some(chosen) = &raw.implementation
            && chosen != &raw.arch
            && chosen != IMPLEMENTATION_FALLBACK
            && !is_implementation_name(chosen)
        {
            return Err(format!(
                "implementation must be the entry's architecture {:?}, {IMPLEMENTATION_FALLBACK:?} or the name of an implementation (an identifier \
                 of at most {MAX_IMPLEMENTATION_NAME} characters that is not \"fallback\" and does not begin with sm_ and a digit) of an operator \
                 declared with by_arch=, got {chosen:?}",
                raw.arch
            ));
        }
        if let Some(math) = &raw.cuda_math {
            check_cuda_math(math)?;
            if kind != EntryKind::Triton || raw.source_sha256.is_none() {
                return Err("cuda_math is only on the entry of a kernel tilefish generated (a triton entry with a source_sha256)".to_owned());
            }
        }
        match kind {
            EntryKind::Triton => {
                let (Some(num_warps), Some(num_stages)) = (raw.num_warps, raw.num_stages) else {
                    return Err("a triton kernel has num_warps and num_stages".to_owned());
                };
                if !NUM_WARPS.contains(&num_warps) {
                    return Err(format!("num_warps must be one of {NUM_WARPS:?}, got {num_warps}"));
                }
                if !(1..=MAX_STAGES).contains(&num_stages) {
                    return Err(format!("num_stages must be an int from 1 to {MAX_STAGES}, got {num_stages}"));
                }
                match raw.total_num_warps {
                    None if raw.block != [32 * num_warps, 1, 1] => {
                        return Err(format!(
                            "block must be [32 * num_warps, 1, 1] = {:?}, got {:?}",
                            [32 * num_warps, 1, 1],
                            raw.block
                        ));
                    }
                    // a warp-specialised kernel: its partitions add warps to the num_warps of its default partition, and its block is all of them
                    Some(total) if total <= num_warps || total > MAX_WARPS => {
                        return Err(format!(
                            "total_num_warps must be more than num_warps ({num_warps}, the warps of the default partition of a warp-specialised \
                             kernel) and at most {MAX_WARPS}, got {total}"
                        ));
                    }
                    Some(total) if raw.block != [32 * total, 1, 1] => {
                        return Err(format!(
                            "the block of a warp-specialised kernel is [32 * total_num_warps, 1, 1] = {:?}, got {:?}",
                            [32 * total, 1, 1],
                            raw.block
                        ));
                    }
                    _ => {}
                }
                let Some(options) = &raw.options else {
                    return Err("a triton kernel has the compile options it was built with".to_owned());
                };
                if options.num_warps != num_warps || options.num_stages != num_stages {
                    return Err("the compile options disagree with num_warps and num_stages".to_owned());
                }
                // a program of more than one CTA (a cluster): the compile option says so, exactly then (Python's `_check_launch`)
                match options.num_ctas {
                    None if raw.launch.num_ctas != 1 => {
                        return Err(format!(
                            "options.num_ctas (present when a program is more than one CTA) must be launch.num_ctas {}, got None",
                            raw.launch.num_ctas
                        ));
                    }
                    Some(said) if said == 1 || said != raw.launch.num_ctas => {
                        return Err(format!(
                            "options.num_ctas (present when a program is more than one CTA) must be launch.num_ctas {}, got {said}",
                            raw.launch.num_ctas
                        ));
                    }
                    _ => {}
                }
                if raw.binary.is_some() {
                    return Err("a triton kernel is not a prebuilt binary: it has no `binary` section".to_owned());
                }
                if raw.toolchain.triton.is_none() || raw.toolchain.ptxas.is_none() {
                    return Err("the toolchain of a triton kernel is {\"triton\": ..., \"ptxas\": ...}".to_owned());
                }
            }
            EntryKind::Binary => {
                let extra: Vec<&str> = [
                    ("num_warps", raw.num_warps.is_some()),
                    ("total_num_warps", raw.total_num_warps.is_some()),
                    ("num_stages", raw.num_stages.is_some()),
                    ("options", raw.options.is_some()),
                ]
                .into_iter()
                .filter_map(|(k, present)| present.then_some(k))
                .collect();
                if !extra.is_empty() {
                    return Err(format!("a prebuilt binary has no {extra:?} (Triton compile settings)"));
                }
                let Some(binary) = &raw.binary else {
                    return Err("a prebuilt binary has a `binary` section: {\"source\": \"cubin\"} or {\"source\": \"ptx\", \"ptxas_flags\": [...]}".to_owned());
                };
                let source = read_binary(binary)?;
                if raw.source_digest_kind.as_deref() != Some(BINARY_DIGEST_KIND) || raw.source_digest.is_none() {
                    return Err(format!(
                        "a prebuilt binary records the sha256 of its PTX or cubin as source_digest, with source_digest_kind '{BINARY_DIGEST_KIND}'"
                    ));
                }
                if raw.source_sha256.is_some() {
                    return Err(
                        "a prebuilt binary has no source_sha256 (that is the source of a generated Triton kernel)"
                            .to_owned(),
                    );
                }
                if source == BinarySource::Cubin && raw.source_digest.as_deref() != Some(raw.sha256.as_str()) {
                    return Err("the source_digest of a cubin used as it is is the sha256 of the cubin file".to_owned());
                }
                if raw.toolchain.triton.is_some() {
                    return Err(
                        "the toolchain of a prebuilt binary has no triton: it is {} or {\"ptxas\": ...}".to_owned()
                    );
                }
                if (source == BinarySource::Cubin) != raw.toolchain.ptxas.is_none() {
                    return Err(
                        "the toolchain of a prebuilt binary is {} for a cubin and {\"ptxas\": ...} for PTX".to_owned()
                    );
                }
                if !raw.constexprs.is_empty() {
                    return Err("a prebuilt binary has no constexprs".to_owned());
                }
                if !raw.block.iter().all(|b| (1..=1024).contains(b))
                    || raw.block.iter().map(|b| u64::from(*b)).product::<u64>() > 1024
                {
                    return Err(format!(
                        "block must be three sizes from 1 to 1024 whose product is at most 1024, got {:?}",
                        raw.block
                    ));
                }
                let blockers = raw.launch.blockers();
                if !blockers.is_empty() || raw.launch.global_scratch_size != 0 {
                    return Err(format!(
                        "a prebuilt binary is launched with its parameters and, when it declares one, its cluster (at most {MAX_CLUSTER} blocks): \
                         its launch attributes are the zero ones but cluster_dims and num_ctas, got {}",
                        if blockers.is_empty() {
                            format!("global_scratch_size={}", raw.launch.global_scratch_size)
                        } else {
                            blockers.join(", ")
                        }
                    ));
                }
            }
        }
        if raw.launch.cluster_dims.contains(&0) || raw.launch.num_ctas == 0 {
            return Err("the launch attributes have a zero cluster dimension or CTA count".to_owned());
        }
        // the alignment of a global scratch: with a global scratch only, a power of two (Python's `_check_launch`)
        if let Some(align) = raw.launch.global_scratch_align
            && (raw.launch.global_scratch_size == 0 || !align.is_power_of_two())
        {
            return Err(format!(
                "launch.global_scratch_align is the alignment (a power of two) of a global scratch, only with global_scratch_size above 0; got \
                 {align} with {}",
                raw.launch.global_scratch_size
            ));
        }
        // a persistent grid: programs_per_sm from 1 to 32, an sm_count from 1 to 2**20, and a grid of one entry (the tiles)
        if let Some(p) = &raw.persistent {
            if !(1..=MAX_PROGRAMS_PER_SM).contains(&p.programs_per_sm) {
                return Err(format!(
                    "persistent.programs_per_sm must be an int from 1 to {MAX_PROGRAMS_PER_SM}, got {}",
                    p.programs_per_sm
                ));
            }
            if let Some(n) = p.sm_count.filter(|n| !(1..=MAX_SM_COUNT).contains(n)) {
                return Err(format!("persistent.sm_count must be an int from 1 to {MAX_SM_COUNT}, got {n}"));
            }
            if parse_grid(&raw.grid, &mut Symbols::default())?.len() != 1 {
                return Err(format!("a persistent grid has one entry (the number of tiles), got {:?}", raw.grid));
            }
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
        // the slots of the parameters (a TMA descriptor takes 1 + 2 * rank), after which the scratch slots come
        let n: usize = read_params(raw, kind)?.iter().map(|p| p.slots().len()).sum();
        if raw.abi.scratch_slots != kind.scratch_slots() as u64 {
            return Err(format!(
                "a {} kernel has {} scratch slots, but abi.scratch_slots is {}",
                kind.as_str(),
                kind.scratch_slots(),
                raw.abi.scratch_slots
            ));
        }
        if raw.abi.scratch.len() != kind.scratch_slots() {
            return Err(match kind {
                EntryKind::Triton => "the ABI must end with the two scratch slots".to_owned(),
                EntryKind::Binary => {
                    "a prebuilt binary is launched with exactly its parameters: its abi.scratch is empty".to_owned()
                }
            });
        }
        for (k, s) in raw.abi.scratch.iter().enumerate() {
            let name = ["global_scratch", "profile_scratch"].get(k).copied().unwrap_or("");
            if s.name != name || s.slot != n + k || s.ty != "u64" || s.ptx_type != "u64" || s.size != 8 || s.value != 0
            {
                return Err("the scratch slots must be global_scratch and profile_scratch, null 64-bit pointers after the arguments".to_owned());
            }
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

    /// An `out` pointer that may be written over an input (`alias`): the input is an `in` pointer, not strided and not a device scalar, of
    /// the same dtype and declared shape, and no other `out` pointer names it (Python's `check_alias_target`).
    fn check_aliases(&self) -> std::result::Result<(), String> {
        let mut taken: HashMap<&str, &str> = HashMap::new();
        for p in &self.params {
            let ParamKind::Pointer { alias: Some(alias), dtype_name, shape, .. } = &p.kind else { continue };
            if !is_ident(alias) {
                return Err(format!("arg {}: alias must be the name of a pointer argument, got {alias:?}", p.name));
            }
            let Some(ParamKind::Pointer {
                access,
                dtype_name: target_dtype,
                shape: target_shape,
                view,
                tensor_map,
                ..
            }) = self.param(alias).map(|t| &t.kind)
            else {
                return Err(format!("arg {} may alias '{alias}', which is not an `in` pointer argument", p.name));
            };
            if *access != Access::In || tensor_map.is_some() {
                return Err(format!("arg {} may alias '{alias}', which is not an `in` pointer argument", p.name));
            }
            if view.is_some() || target_shape.is_empty() {
                return Err(format!(
                    "arg {} may alias '{alias}', which is a strided view or a device scalar: an Out written over an input is that whole tensor",
                    p.name
                ));
            }
            if target_dtype != dtype_name || target_shape != shape {
                return Err(format!("arg {} may alias '{alias}', which has another dtype or declared shape", p.name));
            }
            if let Some(other) = taken.insert(alias.as_str(), p.name.as_str()) {
                return Err(format!(
                    "the out pointers {other} and {} both may alias '{alias}': at most one is written over an input",
                    p.name
                ));
            }
        }
        Ok(())
    }

    /// The checks of `validate_spec` that the manifest's own fields allow: what a launch relies on is all there.
    fn check_consistent(&self, syms: &mut Symbols) -> std::result::Result<(), String> {
        self.check_aliases()?;
        let mut carried: HashMap<(&str, usize), &str> = HashMap::new();
        for p in &self.params {
            match &p.kind {
                ParamKind::Pointer { shape, view, .. } => {
                    for (axis, d) in shape.iter().enumerate() {
                        if let ShapeDim::Expr(e) = d {
                            parse_expr(e, &format!("parameter {} shape[{axis}]", p.name), syms)?;
                        }
                    }
                    if let Some(view) = view {
                        for (axis, s) in view.strides.iter().enumerate() {
                            if let StrideEntry::Dim(e) = s {
                                parse_expr(e, &format!("parameter {} strides[{axis}]", p.name), syms)?;
                            }
                        }
                        syms.intern(&view.offset)?;
                    }
                }
                ParamKind::Scalar { dim, count_of, stride_of, .. } => {
                    parse_expr(dim, &format!("parameter {} dim", p.name), syms)?;
                    if let Some(target) = count_of {
                        self.check_count_of(&p.name, dim, target, syms)?;
                    }
                    for StrideOf { arg, axis } in stride_of {
                        let Some(ParamKind::Pointer { view: Some(view), tensor_map: None, .. }) =
                            self.param(arg).map(|t| &t.kind)
                        else {
                            return Err(format!(
                                "parameter {} is the stride of '{arg}', which is not a strided pointer parameter (a descriptor passes its \
                                 strides in its own slots)",
                                p.name
                            ));
                        };
                        let Some(entry) = view.strides.get(*axis) else {
                            return Err(format!(
                                "parameter {} is the stride of axis {axis} of '{arg}', which has {} axes",
                                p.name,
                                view.strides.len()
                            ));
                        };
                        if *entry != StrideEntry::Dim(dim.clone()) {
                            return Err(format!(
                                "parameter {} carries {dim:?} as the stride of axis {axis} of '{arg}', but the strides of that pointer say {entry:?}",
                                p.name
                            ));
                        }
                        if let Some(other) = carried.insert((arg.as_str(), *axis), p.name.as_str()) {
                            return Err(format!(
                                "axis {axis} of '{arg}' has two stride scalars ({other} and {})",
                                p.name
                            ));
                        }
                    }
                }
                ParamKind::Fixed { .. } | ParamKind::Given { .. } | ParamKind::Null { .. } => {}
            }
        }
        for p in &self.params {
            // (a descriptor's stride dimensions are carried by its own stride slots, not by scalars)
            if let ParamKind::Pointer { view: Some(view), tensor_map: None, .. } = &p.kind {
                for (axis, s) in view.strides.iter().enumerate() {
                    if matches!(s, StrideEntry::Dim(_)) && !carried.contains_key(&(p.name.as_str(), axis)) {
                        return Err(format!(
                            "the stride of axis {axis} of '{}' is carried by no scalar parameter (stride_of)",
                            p.name
                        ));
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
                | Precondition::MaxElements { arg, index, .. }
                | Precondition::MaxExtent { arg, index, .. } => match self.param(arg) {
                    Some(p) if p.index == *index => {}
                    _ => return Err(format!("a precondition refers to {arg}@{index}, which is not an argument")),
                },
            }
            let kind_of = |name: &str| self.param(name).map(|p| &p.kind);
            match pre {
                Precondition::PointerAlignment { arg, bytes, .. } => {
                    let allowed = match kind_of(arg) {
                        // the global address of a TMA descriptor is 16-byte aligned, a view's too
                        Some(ParamKind::Pointer { tensor_map: Some(_), .. }) => Some(vec![ALIGNMENT]),
                        // a strided pointer, and a dense one that is not a device scalar (the specialisation of a lowered kernel for views
                        // that do not start on 16 bytes), may promise only its element's alignment
                        Some(ParamKind::Pointer { view, shape, dtype_name, .. })
                            if view.is_some() || !shape.is_empty() =>
                        {
                            let element = DType::from_name(dtype_name).map_or(1, |d| d.size() as u64);
                            Some(vec![ALIGNMENT, element])
                        }
                        Some(ParamKind::Pointer { .. }) => Some(vec![ALIGNMENT]),
                        _ => None,
                    };
                    if !allowed.is_some_and(|a| a.contains(bytes)) || !aligned.insert(arg.clone()) {
                        return Err(format!(
                            "pointer_alignment must be {ALIGNMENT} bytes (or, for a pointer that is not a device scalar or a descriptor, its element size), once, on a pointer argument ({arg})"
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
                    // (not on a TMA descriptor: its kernel addresses it by coordinates)
                    let dense = matches!(kind_of(arg), Some(ParamKind::Pointer { view: None, shape, tensor_map: None, .. }) if !shape.is_empty());
                    if !dense || !limited.insert(arg.clone()) {
                        return Err(format!(
                            "max_elements must be once, on a dense pointer argument that is not a device scalar ({arg})"
                        ));
                    }
                }
                Precondition::MaxExtent { arg, .. } => {
                    if !matches!(kind_of(arg), Some(ParamKind::Pointer { view: Some(_), tensor_map: None, .. }))
                        || !limited.insert(arg.clone())
                    {
                        return Err(format!("max_extent must be once, on a strided pointer argument ({arg})"));
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
                    "pointer argument {} has no pointer_alignment precondition: every pointer of a binary is compiled for the alignment it promises",
                    p.name
                ));
            }
            if p.is_device_scalar() && !matches!(&p.kind, ParamKind::Pointer { access: Access::In, .. }) {
                return Err(format!("the device scalar {} must be read only (access in)", p.name));
            }
        }
        // a device scalar's alignment is 16 bytes: "A device scalar's is 16"
        for pre in &self.preconditions {
            if let Precondition::PointerAlignment { arg, bytes, .. } = pre
                && self.param(arg).is_some_and(Param::is_device_scalar)
                && *bytes != ALIGNMENT
            {
                return Err(format!("the pointer_alignment of the device scalar {arg} is {ALIGNMENT} bytes"));
            }
        }
        parse_grid(&self.grid, syms)?;
        // a static_size is the size the shape has on that axis (checked in the order of the preconditions, as Python's reader does, so the
        // first one refused is the same in every run); the table of dimensions names every dimension in use
        for pre in &self.preconditions {
            let Precondition::StaticSize { arg, axis, value, .. } = pre else { continue };
            let agrees = match self.param(arg).map(|p| &p.kind) {
                Some(ParamKind::Pointer { shape, .. }) => {
                    matches!(shape.get(*axis), Some(ShapeDim::Static(n)) if n == value)
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
                "dims does not name {unnamed:?}, which the declared shapes, the strides, the scalars or the grid mention"
            ));
        }
        // a max_elements precondition is on the shape of its argument, with the bucket values in
        for pre in &self.preconditions {
            if let Precondition::MaxElements { arg, shape, .. } = pre {
                self.check_max_elements_shape(arg, shape, syms)?;
            }
        }
        Ok(())
    }

    /// The value of each bucketed dimension, by the number `syms` gives its name (`None` for every other dimension).
    fn bucket_values(&self, syms: &Symbols) -> std::result::Result<Vec<Option<i64>>, String> {
        let mut values = vec![None; syms.len()];
        for (name, value) in &self.buckets {
            if let Some(i) = syms.find(name) {
                let value =
                    i64::try_from(*value).map_err(|_| format!("the bucket {name} = {value} does not fit 64 bits"))?;
                values[i] = Some(value);
            }
        }
        Ok(values)
    }

    /// Axis `axis` of the declared shape of pointer `arg` with the bucket values in: what Python's compile spec has as the axis of the
    /// argument's `shape` (the text of the folded expression), which the manifest does not carry.
    fn bucketed_axis(
        &self,
        arg: &str,
        axis: usize,
        d: &ShapeDim,
        values: &[Option<i64>],
        syms: &mut Symbols,
    ) -> std::result::Result<Expr, String> {
        Ok(match d {
            ShapeDim::Static(n) => Expr::Num(i128::from(*n)),
            ShapeDim::Expr(e) => parse_expr(e, &format!("parameter {arg} shape[{axis}]"), syms)?.subst(values),
        })
    }

    /// The declared shape of a pointer and, when the entry is bucketed, the bucket values that go into it, for a message.
    fn shape_text(&self, shape: &[ShapeDim]) -> String {
        let axes: Vec<String> = shape.iter().map(ToString::to_string).collect();
        let buckets: Vec<String> = self.buckets.iter().map(|(d, v)| format!("{d} = {v}")).collect();
        if buckets.is_empty() {
            format!("[{}]", axes.join(", "))
        } else {
            format!("[{}] (with {})", axes.join(", "), buckets.join(", "))
        }
    }

    /// Python's `_check_count_of`: a scalar that carries the element count of a pointer (`count_of`) carries the product of that pointer's
    /// shape, with the bucket values in. When the product is affine the scalar's `dim` must be that expression (as Python's `DimExpr`
    /// compares two: `3072*M` is `M*3072`); when it is not (two symbolic sizes), `dim` must be a fresh dimension name, and `count_of` is how a
    /// host gets its value.
    fn check_count_of(
        &self,
        name: &str,
        dim: &str,
        target: &str,
        syms: &mut Symbols,
    ) -> std::result::Result<(), String> {
        let Some(ParamKind::Pointer { shape, .. }) = self.param(target).map(|t| &t.kind) else {
            return Err(format!("parameter {name} count_of {target:?} must name a pointer parameter"));
        };
        let values = self.bucket_values(syms)?;
        let mut product = Expr::Num(1);
        for (axis, d) in shape.iter().enumerate() {
            let factor = self.bucketed_axis(target, axis, d, &values, syms)?;
            product = Expr::Mul(Box::new(product), Box::new(factor));
        }
        match product.folded_key() {
            Some(count) => {
                if parse_expr(dim, &format!("parameter {name} dim"), syms)?.folded_key() != Some(count) {
                    return Err(format!(
                        "parameter {name} is the element count of {target} (count_of), the product of its shape {}, but it carries {dim:?}, \
                         which is another expression: it must carry that product",
                        self.shape_text(shape)
                    ));
                }
            }
            None if !is_ident(dim) => {
                return Err(format!(
                    "parameter {name} is the element count of {target} (count_of), and the product of its shape {} is not an affine \
                     expression, so it must carry a fresh dimension name, got {dim:?}",
                    self.shape_text(shape)
                ));
            }
            None => {}
        }
        Ok(())
    }

    /// Python's `p["shape"] == a["shape"]` of a `max_elements` precondition: its shape is the declared shape of its argument with the
    /// bucket values in, axis by axis. Python compares the texts of the compile spec, which the manifest does not carry; the expressions are
    /// compared as Python's `DimExpr` compares them, which for the folded texts Python writes is the same.
    fn check_max_elements_shape(
        &self,
        arg: &str,
        limited: &[String],
        syms: &mut Symbols,
    ) -> std::result::Result<(), String> {
        let Some(ParamKind::Pointer { shape, .. }) = self.param(arg).map(|p| &p.kind) else {
            return Err(format!("a max_elements precondition refers to {arg}, which is not a pointer argument"));
        };
        let values = self.bucket_values(syms)?;
        let mut agrees = limited.len() == shape.len();
        for (axis, (d, text)) in shape.iter().zip(limited).enumerate() {
            let declared = self.bucketed_axis(arg, axis, d, &values, syms)?;
            let said = parse_expr(text, &format!("max_elements precondition shape[{axis}]"), syms)?;
            agrees &= match (declared.folded_key(), said.folded_key()) {
                (Some(a), Some(b)) => a == b,
                (None, None) => declared == said, // (no declaration has an axis that is not affine: then the same expression)
                _ => false,
            };
        }
        if !agrees {
            return Err(format!(
                "the max_elements precondition of {arg} has the shape {limited:?}, but the declared shape of {arg} is {}: a max_elements \
                 precondition is on its argument's own shape",
                self.shape_text(shape)
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

/// The integer `n` a schema field holds, for `0 < n < 3`: an older schema (Python: `isinstance(older, int)`, a float or a bool is not one).
fn older_schema(schema: &Value) -> Option<u64> {
    schema.as_u64().filter(|n| (1..MANIFEST_SCHEMA).contains(n))
}

impl Manifest {
    /// Load `<directory>/manifest.json` (or the manifest file itself) and every cubin it names.
    ///
    /// Refuses a manifest that is not strict JSON of the schema `tilefish` writes (schema 3: an older one is refused with the advice to
    /// export it again), one whose digest is not the pinned one (before any cubin is read), a cubin that is a symbolic link or whose name
    /// is not a plain file name in the directory, and a cubin whose `sha256` or `sha256_code` is not the manifest's, that is not an ELF
    /// file with the code section of the manifest's symbol, or that is for another architecture than the entry says.
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
        let (doc, order) = parse_strict_keeping(text, is_ordered_object)
            .map_err(|e| Error::manifest(format!("the manifest {label:?} is not strict JSON: {e}")))?;
        // the buckets and the dims of entry `i` of a section, in the order of the file
        let file_order = |section: &str, i: usize, field: &str| {
            order.keys(&[PathStep::Key(section.to_owned()), PathStep::Index(i), PathStep::Key(field.to_owned())])
        };
        // the schema is looked at first, so that a manifest of an older schema is told to be exported again, whatever else it lacks
        if let Some(older) = doc.get("schema").and_then(older_schema) {
            return Err(Error::manifest(format!(
                "the manifest {label:?} is schema {older}, which an older tilefish wrote; this host reads schema {MANIFEST_SCHEMA}, which adds \
                 strided views, device scalars, prebuilt binaries and library operations (and records the number of scratch slots of every ABI) \
                 and has another digest. Export it again with a current tilefish (tilefish.runtime.export) and pin the new digest"
            )));
        }
        let raw: RawDoc = serde_json::from_value(doc.clone()).map_err(|e| {
            if doc.is_object() {
                Error::manifest(format!(
                    "the manifest {label:?} is not a tilefish manifest of schema {MANIFEST_SCHEMA} with a 'kernels' list and a 'library' list: {e}"
                ))
            } else {
                Error::manifest(format!("the manifest {label:?} is not a tilefish manifest: it is not a JSON object"))
            }
        })?;
        // the schema is the integer 3: not 3.0 (which a strict host refuses), not "3", not true
        if raw.schema.as_u64() != Some(MANIFEST_SCHEMA) {
            return Err(Error::manifest(format!(
                "the manifest {label:?} is not a tilefish manifest of schema {MANIFEST_SCHEMA} (schema {})",
                raw.schema
            )));
        }
        let mut raws = Vec::with_capacity(raw.kernels.len());
        for (i, value) in raw.kernels.iter().enumerate() {
            let mut entry: RawEntry = serde_json::from_value(value.clone()).map_err(|e| {
                Error::manifest(format!("manifest kernel {i} ({:?}): {e}", value.get("stem").and_then(Value::as_str)))
            })?;
            entry.buckets.follow(file_order("kernels", i, "buckets"));
            entry.dims.follow(file_order("kernels", i, "dims"));
            raws.push(entry);
        }
        for raw in &raws {
            // everything that needs no file: the names (a cubin name that leaves the directory is refused here, before any read)
            Entry::check_raw(raw).map_err(|e| Error::manifest(format!("'{}' ({}): {e}", raw.stem, raw.arch)))?;
        }
        let mut library = Vec::with_capacity(raw.library.len());
        let mut convs = Vec::new();
        for (i, value) in raw.library.iter().enumerate() {
            let what = |e: serde_json::Error| {
                Error::manifest(format!("library entry #{i} ({:?}): {e}", value.get("stem").and_then(Value::as_str)))
            };
            // a cuDNN convolution has its own fields (`docs/manifest-schema-3.md`, section 9.4); every other entry is a cuBLASLt GEMM's
            if value.get("impl").and_then(Value::as_str) == Some(CONV_IMPL) {
                let mut entry: RawConv = serde_json::from_value(value.clone()).map_err(what)?;
                entry.buckets.follow(file_order("library", i, "buckets"));
                entry.dims.follow(file_order("library", i, "dims"));
                convs.push(ConvEntry::from_raw(entry, i)?);
            } else {
                let mut entry: RawLibrary = serde_json::from_value(value.clone()).map_err(what)?;
                entry.buckets.follow(file_order("library", i, "buckets"));
                entry.dims.follow(file_order("library", i, "dims"));
                library.push(LibraryEntry::from_raw(entry, i)?);
            }
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
        let library_keys = library.iter().map(|e| (&e.stem, &e.arch)).chain(convs.iter().map(|e| (&e.stem, &e.arch)));
        for (stem, arch) in library_keys {
            if !seen.insert((stem.clone(), arch.clone())) {
                return Err(Error::manifest(format!("the manifest lists '{stem}' for {arch} twice")));
            }
        }
        Ok(Self {
            label: label.to_owned(),
            digest,
            entries: Arc::new(entries),
            library: Arc::new(library),
            convs: Arc::new(convs),
            kernels: Arc::new(Mutex::new(HashMap::new())),
            gemms: Arc::new(Mutex::new(HashMap::new())),
            conv_cache: Arc::new(Mutex::new(HashMap::new())),
            sm_partition: None,
        })
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

    /// Every pinned cuBLASLt GEMM of the manifest's `library` section, in file order (its cuDNN convolutions are [`Manifest::convolutions`]).
    pub fn library(&self) -> &[LibraryEntry] {
        &self.library
    }

    /// The library operation of `stem` for `arch` (`"sm_103"`).
    pub fn library_entry(&self, stem: &str, arch: &str) -> Option<&LibraryEntry> {
        self.library.iter().find(|e| e.stem == stem && e.arch == arch)
    }

    /// Refuse a manifest whose generated kernels' CUDA math functions (section 4.7) were not measured on the CUDA release `cuda` (`"13.0"`): the
    /// release whose math library the rest of the host (its other CUDA kernels, the framework it is compared with) is built with. Such a
    /// binary still computes what it was exported with, but its `erff`, `tanhf`, ... may not be the host's bits; the error names each entry
    /// and the releases its functions were measured on. Entries without functions (or without `cuda_math`) pass.
    pub fn check_cuda_math(&self, cuda: &str) -> Result<()> {
        if cuda_release(cuda).is_none() {
            return Err(Error::manifest(format!("check_cuda_math: a CUDA release looks like \"13.0\", got {cuda:?}")));
        }
        let off: Vec<String> = self
            .entries
            .iter()
            .filter_map(|e| e.cuda_math.as_ref().map(|m| (e, m)))
            .filter(|(_, m)| !m.functions.is_empty() && !m.cuda.iter().any(|r| r == cuda))
            .map(|(e, m)| {
                format!("'{}' ({}): {} measured on CUDA {:?}", e.stem, e.arch, m.functions.join(", "), m.cuda)
            })
            .collect();
        if off.is_empty() {
            return Ok(());
        }
        Err(Error::manifest(format!(
            "the manifest in {:?} has kernels whose CUDA math functions were not measured on CUDA {cuda}, so their bits may not be that \
             release's: {}. Re-run the CUDA math tests on it (docs/numerics.md) and export again",
            self.label,
            off.join("; ")
        )))
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
        let items: Vec<Named<'_>> = self
            .entries
            .iter()
            .map(|e| Named { stem: &e.stem, op: &e.op, arch: &e.arch, buckets: &e.buckets })
            .collect();
        select_among(&self.label, "kernel", &items, name, buckets, arch).map_err(|e| match e {
            Error::NoBinary(m) if self.library.iter().any(|l| l.stem == name || l.op == name) => Error::NoBinary(
                format!("{m}; '{name}' is a library operation (a pinned cuBLASLt GEMM): load it with Manifest::gemm"),
            ),
            Error::NoBinary(m) if self.convs.iter().any(|l| l.stem == name || l.op == name) => {
                Error::NoBinary(format!(
                    "{m}; '{name}' is a library operation (a pinned cuDNN convolution): load it with Manifest::conv"
                ))
            }
            other => other,
        })
    }

    /// The index of the convolution entry for `name` (a stem, or an operator) with exactly these bucket values on `arch`.
    pub(crate) fn select_conv(&self, name: &str, buckets: &[(&str, u64)], arch: &str) -> Result<usize> {
        let items: Vec<Named<'_>> =
            self.convs.iter().map(|e| Named { stem: &e.stem, op: &e.op, arch: &e.arch, buckets: &e.buckets }).collect();
        select_among(&self.label, "cuDNN convolution", &items, name, buckets, arch).map_err(|e| match e {
            Error::NoBinary(m) if self.library.iter().any(|l| l.stem == name || l.op == name) => {
                Error::NoBinary(format!("{m}; '{name}' is a pinned cuBLASLt GEMM: load it with Manifest::gemm"))
            }
            Error::NoBinary(m) if self.entries.iter().any(|k| k.stem == name || k.op == name) => {
                Error::NoBinary(format!("{m}; '{name}' is a kernel: load it with Manifest::kernel"))
            }
            other => other,
        })
    }

    /// The index of the library entry for `name` (a stem, or an operator) with exactly these bucket values on `arch`.
    pub(crate) fn select_library(&self, name: &str, buckets: &[(&str, u64)], arch: &str) -> Result<usize> {
        let items: Vec<Named<'_>> = self
            .library
            .iter()
            .map(|e| Named { stem: &e.stem, op: &e.op, arch: &e.arch, buckets: &e.buckets })
            .collect();
        select_among(&self.label, "library operation", &items, name, buckets, arch).map_err(|e| match e {
            Error::NoBinary(m) if self.entries.iter().any(|k| k.stem == name || k.op == name) => {
                Error::NoBinary(format!("{m}; '{name}' is a kernel: load it with Manifest::kernel"))
            }
            Error::NoBinary(m) if self.convs.iter().any(|k| k.stem == name || k.op == name) => {
                Error::NoBinary(format!("{m}; '{name}' is a cuDNN convolution: load it with Manifest::conv"))
            }
            other => other,
        })
    }
}

/// What the selection of an entry by name needs to know about one: a kernel or a library entry.
pub(crate) struct Named<'a> {
    stem: &'a str,
    op: &'a str,
    arch: &'a str,
    buckets: &'a [(String, u64)],
}

/// The index of the item for `name` (a stem, or an operator) with exactly these bucket values on `arch`: the choice of a kernel and of
/// a library operation, with `what` (`"kernel"`, `"library operation"`) in the messages.
fn select_among(
    label: &str,
    what: &str,
    items: &[Named<'_>],
    name: &str,
    buckets: &[(&str, u64)],
    arch: &str,
) -> Result<usize> {
    // the bucket values are Python's mapping of dimension names to sizes, which cannot name a dimension twice: here a list could, and one
    // that does would match an entry with as many buckets of which it names only some
    if let Some((d, _)) =
        buckets.iter().enumerate().find_map(|(i, b)| buckets[..i].iter().any(|(e, _)| *e == b.0).then_some(*b))
    {
        let values: Vec<u64> = buckets.iter().filter(|(e, _)| *e == d).map(|(_, v)| *v).collect();
        return Err(Error::Launch(format!(
            "the bucket values for '{name}' name the dimension {d} more than once ({values:?}): give each bucketed dimension once, \
             with its value"
        )));
    }
    let same = |item: &Named<'_>| {
        item.buckets.len() == buckets.len()
            && buckets.iter().all(|(d, v)| item.buckets.iter().any(|(ed, ev)| ed == d && ev == v))
    };
    let named: Vec<usize> = (0..items.len()).filter(|i| items[*i].stem == name || items[*i].op == name).collect();
    if named.is_empty() {
        let mut have: Vec<&str> = items.iter().map(|e| e.stem).collect();
        have.sort_unstable();
        have.dedup();
        return Err(Error::NoBinary(format!("the manifest in {label:?} has no {what} '{name}' (it has {have:?})")));
    }
    let here: Vec<usize> = named.iter().copied().filter(|i| items[*i].arch == arch).collect();
    if here.is_empty() {
        let mut archs: Vec<&str> = named.iter().map(|i| items[*i].arch).collect();
        archs.sort_unstable();
        archs.dedup();
        return Err(Error::NoBinary(format!(
            "the manifest in {label:?} has no binary for '{name}' on {arch} (it has {})",
            archs.join(", ")
        )));
    }
    // a stem names its own buckets; an operator needs them given
    let found = here.iter().copied().find(|i| {
        let e = &items[*i];
        if e.stem == name { buckets.is_empty() || same(e) } else { same(e) }
    });
    found.ok_or_else(|| {
        let mut wanted: Vec<&str> = here.iter().flat_map(|i| items[*i].buckets.iter().map(|(d, _)| d.as_str())).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let listing: Vec<String> = wanted
            .iter()
            .map(|d| {
                let mut vs: Vec<u64> =
                    here.iter().filter_map(|i| items[*i].buckets.iter().find(|(ed, _)| ed == d).map(|(_, v)| *v)).collect();
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

fn verify_cubin(
    raw: &RawEntry,
    stem: &str,
    arch: &str,
    read_cubin: &mut impl FnMut(&str) -> Result<Vec<u8>>,
) -> Result<Vec<u8>> {
    let name = raw.cubin.as_str();
    let kind = entry_kind(raw).map_err(Error::manifest)?;
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
    // the architecture the cubin is for is in its ELF header, where it can be read (a prebuilt cubin is used as it is, for its own
    // architecture: Python refuses one whose header says another; a Triton cubin the driver refuses on the wrong device)
    match (elf.sm(), arch_number(arch)) {
        (Some(sm), Some(want)) if sm != want => {
            return Err(Error::manifest(format!(
                "'{stem}' ({arch}): {name:?} is for sm_{sm} (its ELF header says so), but the manifest says {arch}: a cubin is used as it is, for the architecture it was built for"
            )));
        }
        (None, _) if kind == EntryKind::Binary => {
            return Err(Error::manifest(format!(
                "'{stem}' ({arch}): the ELF header of {name:?} is not that of a CUDA cubin of a known architecture: its architecture cannot be read"
            )));
        }
        _ => {}
    }
    // (Triton gives a kernel that needs shared memory dynamic shared memory, so the section means the manifest must say how much. A prebuilt
    // binary may have static shared memory, which has the same section, so its declaration says how much dynamic memory it addresses; the
    // loader's check, static + dynamic within the device's opt-in maximum, is the guard.)
    if kind == EntryKind::Triton && raw.shared == 0 && elf.uses_shared(&raw.symbol) {
        return Err(Error::manifest(format!(
            "'{stem}' ({arch}): the kernel has a shared memory section ('.nv.shared.{}'), so it uses shared memory, but the manifest \
             says shared = 0; a launch without the dynamic shared memory it addresses would fault and poison the CUDA context",
            raw.symbol
        )));
    }
    Ok(cubin)
}
