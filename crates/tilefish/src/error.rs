//! The errors of the host. Every refusal is an [`Error`]; nothing here panics on a bad manifest or a bad call.

use std::fmt;

/// Which precondition of a binary a launch broke (the `kind` of Python's `TilefishPreconditionError`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreconditionKind {
    /// The tensor is not of the element type the binary was compiled for.
    Dtype,
    /// The tensor has another number of dimensions than the signature says.
    Rank,
    /// The shapes of the call cannot be one binding of the signature's dimensions (or a dimension is empty).
    Shape,
    /// A dimension that the binary has baked in (`Dim("D", 2048)`) is not the size of the tensor.
    StaticSize,
    /// A pointer is not aligned to the 16 bytes every pointer of a binary is compiled for.
    PointerAlignment,
    /// A runtime dimension is not a multiple of what the binary assumes.
    Divisible,
    /// A bucketed dimension is not the bucket the binary was built for (there is no padding).
    Bucket,
    /// A tensor has more elements than the 32-bit offsets of a generated kernel reach.
    MaxElements,
    /// A strided view reaches further than the 32-bit offsets of a kernel that indexes with `i32` (its *extent*, one more than the largest
    /// element offset, is more than `2**31 - 1`).
    MaxExtent,
    /// A view has a stride a binary cannot take: the axis the kernel loads along is not unit stride, or a stride does not fit the
    /// integer the binary takes it as. (Python's `stride`; a negative stride is not representable here: strides are unsigned.)
    Stride,
    /// A tensor that is not contiguous (row-major with no gaps) is given for a parameter that is not declared strided.
    Contiguous,
    /// A tensor that is written in place overlaps another argument.
    Alias,
    /// The tensors are not all on the device the kernel was loaded on, nor in the context it was loaded in.
    Device,
    /// A tensor that may only be read is given for a parameter that the kernel writes. (Only the Rust host has this one: Python's
    /// tensors have no access mode.)
    Access,
}

impl PreconditionKind {
    /// The name Python gives the same kind (`"static_size"`, `"pointer_alignment"`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dtype => "dtype",
            Self::Rank => "rank",
            Self::Shape => "shape",
            Self::StaticSize => "static_size",
            Self::PointerAlignment => "pointer_alignment",
            Self::Divisible => "divisible",
            Self::Bucket => "bucket",
            Self::MaxElements => "max_elements",
            Self::MaxExtent => "max_extent",
            Self::Stride => "stride",
            Self::Contiguous => "contiguous",
            Self::Alias => "alias",
            Self::Device => "device",
            Self::Access => "access",
        }
    }
}

impl fmt::Display for PreconditionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Everything that can go wrong in loading a manifest or launching one of its kernels.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A file of the manifest directory cannot be read.
    #[error("cannot read {path:?}: {source}")]
    Io {
        /// The file.
        path: String,
        /// Why.
        #[source]
        source: std::io::Error,
    },
    /// The manifest is not what `tilefish` writes: not strict JSON, a field of the wrong type, an unknown or missing
    /// key, a cubin whose digests differ from the manifest's, a file name that leaves the directory, ...
    #[error("{0}")]
    Manifest(String),
    /// The pin is not a sha256 in lowercase hex.
    #[error("the pin must be the 64 lowercase hex digits of a digest (of a manifest, or of a plan), got {0:?}")]
    BadPin(String),
    /// The manifest has another digest than the pinned one.
    #[error(
        "the manifest in {directory:?} has digest {found}, but the pin says {pinned}: it is not the manifest that was \
         pinned (it was changed, or it is another export)"
    )]
    PinMismatch {
        /// Where the manifest was read from.
        directory: String,
        /// The digest of what was read.
        found: String,
        /// The digest that was pinned.
        pinned: String,
    },
    /// A launch plan (`plan.json`) is refused: it is not strict JSON of schema `tilefish.plan.v1`, a rule of the format does not hold
    /// (`docs/plan-schema-1.md`), it was edited after it was written, or it was built against another manifest than the one next to it, or a
    /// step is not what the manifest's entry says.
    #[error("{0}")]
    Plan(String),
    /// The plan has another digest than the pinned one.
    #[error(
        "the plan in {path:?} has digest {found}, but the pin is {pinned}: it is not the plan that was pinned (it was changed, or it is \
         another export)"
    )]
    PlanPinMismatch {
        /// Where the plan was read from.
        path: String,
        /// The digest of what was read.
        found: String,
        /// The digest that was pinned.
        pinned: String,
    },
    /// The manifest has no binary for the kernel, the buckets or the architecture asked for.
    #[error("{0}")]
    NoBinary(String),
    /// An argument breaks a precondition of the binary; nothing was launched.
    #[error("{message}")]
    Precondition {
        /// Which precondition.
        kind: PreconditionKind,
        /// The argument or dimension it is about.
        arg: Option<String>,
        /// What is wrong.
        message: String,
    },
    /// The binary or the device cannot run this: launch attributes a plain `cuLaunchKernel` does not set up, a packed
    /// element type, more shared memory than the device has, a grid beyond its limits.
    #[error("{0}")]
    Unsupported(String),
    /// The call does not fit the kernel's signature (a missing tensor, an unknown scalar or dimension, a value that
    /// the signature determines and the call also gives, bucket values that name a dimension twice).
    #[error("{0}")]
    Launch(String),
    /// A cuBLASLt call failed: the call and the `cublasStatus_t` it returned.
    #[error("cuBLASLt call {call} failed: {status}")]
    Cublas {
        /// The function of cuBLASLt.
        call: &'static str,
        /// The status it returned (cudarc's `CublasError`, as text).
        status: String,
    },
    /// A CUDA driver call failed: cudarc's `DriverError`, which is the `CUresult` with the name and the text the driver gives it.
    #[error("CUDA driver call {call} failed: {source}")]
    Driver {
        /// What the host was doing when it failed (the driver function, or what it was for).
        call: &'static str,
        /// The driver's error.
        #[source]
        source: cudarc::driver::DriverError,
    },
}

/// The result of an operation of this crate.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// The error of a driver call, for `map_err`: `ctx.attribute(..).map_err(Error::driver("cuDeviceGetAttribute"))`.
    pub(crate) fn driver(call: &'static str) -> impl FnOnce(cudarc::driver::DriverError) -> Self {
        move |source| Self::Driver { call, source }
    }

    /// The error of a cuBLASLt call, for `map_err`.
    pub(crate) fn cublas(call: &'static str) -> impl FnOnce(cudarc::cublaslt::result::CublasError) -> Self {
        move |source| Self::Cublas { call, status: format!("{:?}", source.0) }
    }

    pub(crate) fn manifest(message: impl Into<String>) -> Self {
        Self::Manifest(message.into())
    }

    pub(crate) fn precondition(kind: PreconditionKind, arg: Option<&str>, message: impl Into<String>) -> Self {
        Self::Precondition { kind, arg: arg.map(str::to_owned), message: message.into() }
    }

    /// The kind of precondition this is, if it is one.
    pub fn precondition_kind(&self) -> Option<PreconditionKind> {
        match self {
            Self::Precondition { kind, .. } => Some(*kind),
            _ => None,
        }
    }
}
