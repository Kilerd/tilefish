//! cuDNN through cudarc's `cudnn::sys`: what one pinned convolution needs, as safe items over the `unsafe` calls.
//!
//! cudarc's safe `cudnn` layer chooses its own algorithm and has no backend engine configurations, so this crate calls the library the way
//! Python's `_cudnn.py` does: the legacy descriptors of a problem (`cudnnSetTensorNdDescriptorEx`, `cudnnSetFilterNdDescriptor`,
//! `cudnnSetConvolutionNdDescriptor`, ...), `cudnnGetConvolutionForwardAlgorithm_v7`, `cudnnGetConvolutionForwardWorkspaceSize` and
//! `cudnnConvolutionForward`; and the backend descriptors (tensors, the convolution, its operation and operation graph, an engine of a global
//! index, knob choices, an engine configuration, an execution plan, a variant pack) and `cudnnBackendExecute`. Every `unsafe` call is in this file,
//! with the reason it is sound beside it; what is decided with the answers (the vetting, the order of the calls) is `crate::conv`, which has no
//! `unsafe`.
//!
//! The library is cudarc's dynamically loaded `libcudnn`: it is opened when the first function of it is called, and cudarc **panics** when it is
//! missing, so every entry point here is behind [`available`], which the callers check first. An enum that cuDNN writes (a status of a heuristic
//! answer, a numerical note, a knob type) is read as an integer, never as a Rust enum, so a value this crate does not know is a number, not
//! undefined behavior.

use std::ffi::c_void;
use std::mem::size_of;

use cudarc::cudnn::sys;
use cudarc::driver::CudaContext;

use crate::conv::{ConvLayout, ConvMath, ConvProblem};
use crate::error::{Error, Result};
use crate::tensor::DType;

type Attr = sys::cudnnBackendAttributeName_t;
type AttrType = sys::cudnnBackendAttributeType_t;
type DescType = sys::cudnnBackendDescriptorType_t;

/// The uids of the tensors (torch's: `'x'`, `'w'`, `'y'`).
pub(crate) const UID_X: i64 = 120;
pub(crate) const UID_W: i64 = 119;
pub(crate) const UID_Y: i64 = 121;
/// `CUDNN_KNOB_TYPE_COUNTS` in cuDNN 9.10.2: as many knob descriptors as there are knob types are asked for (a count query answers 0 for the
/// descriptor lists of an engine and of a configuration: measured by Python's binding).
pub(crate) const KNOB_TYPE_COUNT: usize = 43;
const MAX_NOTES: usize = 16;
/// `CUDNN_STATUS_SUCCESS` and `CUDNN_DETERMINISTIC`, as the integers cuDNN writes into a heuristic answer.
pub(crate) const STATUS_SUCCESS: i32 = 0;
pub(crate) const DETERMINISTIC: i32 = 1;

fn status(call: &'static str, s: sys::cudnnStatus_t) -> Result<()> {
    s.result().map_err(Error::cudnn(call))
}

fn data_type(dtype: DType) -> sys::cudnnDataType_t {
    match dtype {
        DType::F16 => sys::cudnnDataType_t::CUDNN_DATA_HALF,
        DType::Bf16 => sys::cudnnDataType_t::CUDNN_DATA_BFLOAT16,
        _ => sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
    }
}

fn format(layout: ConvLayout) -> sys::cudnnTensorFormat_t {
    match layout {
        ConvLayout::Nchw => sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        ConvLayout::Nhwc => sys::cudnnTensorFormat_t::CUDNN_TENSOR_NHWC,
    }
}

fn math_type(math: ConvMath) -> sys::cudnnMathType_t {
    match math {
        ConvMath::TensorOp => sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH,
        ConvMath::Fma => sys::cudnnMathType_t::CUDNN_FMA_MATH,
        ConvMath::Tf32 => sys::cudnnMathType_t::CUDNN_DEFAULT_MATH,
    }
}

/// The `cudnnConvolutionFwdAlgo_t` of an algorithm number (the reader of an entry held it to 0..=7).
fn algo_of(algo: u32) -> Result<sys::cudnnConvolutionFwdAlgo_t> {
    use sys::cudnnConvolutionFwdAlgo_t as A;
    Ok(match algo {
        0 => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        1 => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        2 => A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        3 => A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        4 => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        5 => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        6 => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        7 => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        other => return Err(Error::manifest(format!("a legacy cuDNN algorithm is 0 to 7, got {other}"))),
    })
}

fn ints(values: &[u64]) -> Result<Vec<i32>> {
    values
        .iter()
        .map(|v| {
            i32::try_from(*v)
                .map_err(|_| Error::Launch(format!("a size of the convolution, {v}, does not fit an int32")))
        })
        .collect()
}

fn longs(values: &[u64]) -> Result<Vec<i64>> {
    values
        .iter()
        .map(|v| {
            i64::try_from(*v)
                .map_err(|_| Error::Launch(format!("a size of the convolution, {v}, does not fit an int64")))
        })
        .collect()
}

/// Whether libcudnn can be loaded on this machine (it is opened and kept open when it is there; nothing in it is called).
#[allow(unsafe_code)]
pub(crate) fn available() -> bool {
    // SAFETY: `is_culib_present` only tries to open the library by its usual names, running its initializers as any use of it does, and
    // calls no function in it.
    unsafe { sys::is_culib_present() }
}

/// `cudnnGetVersion()`: the version of the library this process has (`91002` for 9.10.2). The caller has checked [`available`].
#[allow(unsafe_code)]
pub(crate) fn version() -> u64 {
    // SAFETY: the function takes no arguments, and the library is there (the caller checked).
    unsafe { sys::cudnnGetVersion() as u64 }
}

/// A cuDNN handle, destroyed when it is dropped. Its stream is state of the handle, so a call sets it and runs under the lock of the `Conv` that
/// owns the handle.
#[derive(Debug)]
pub(crate) struct Handle {
    raw: sys::cudnnHandle_t,
}

// SAFETY: a cuDNN handle is not tied to a thread; the crate uses it from one thread at a time (every call that sets its stream or runs on it is
// made under the mutex of the `Conv` that owns it).
#[allow(unsafe_code)]
unsafe impl Send for Handle {}
#[allow(unsafe_code)]
unsafe impl Sync for Handle {}

impl Handle {
    /// A handle for the device of `ctx`, made with that context current on this thread.
    #[allow(unsafe_code)]
    pub(crate) fn new(ctx: &CudaContext) -> Result<Self> {
        ctx.bind_to_thread().map_err(Error::driver("cuCtxSetCurrent"))?;
        let mut raw: sys::cudnnHandle_t = std::ptr::null_mut();
        // SAFETY: `raw` is a live local the library writes the handle into.
        status("cudnnCreate", unsafe { sys::cudnnCreate(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Set the stream the handle's calls are enqueued on.
    ///
    /// # Safety
    ///
    /// `stream` must be a valid `CUstream` of the handle's context (or null).
    #[allow(unsafe_code)]
    pub(crate) unsafe fn set_stream(&self, stream: cudarc::driver::sys::CUstream) -> Result<()> {
        // SAFETY: the handle is live; the stream is the caller's, as the contract says.
        status("cudnnSetStream", unsafe { sys::cudnnSetStream(self.raw, stream.cast()) })
    }
}

impl Drop for Handle {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        // SAFETY: `raw` was made by `cudnnCreate` and is destroyed once, here; a failure is ignored in a destructor.
        let _ = unsafe { sys::cudnnDestroy(self.raw) };
    }
}

// -- the legacy API --------------------------------------------------------------------------------------------------------------

/// `cudnnConvolutionFwdAlgoPerf_t` with every enum read as an integer (the same layout as cudarc's struct, which it is passed as).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct RawPerf {
    algo: i32,
    status: i32,
    time: f32,
    memory: usize,
    determinism: i32,
    math_type: i32,
    reserved: [i32; 3],
}

const _: () = assert!(size_of::<RawPerf>() == size_of::<sys::cudnnConvolutionFwdAlgoPerf_t>());

/// One answer of `cudnnGetConvolutionForwardAlgorithm_v7`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Candidate {
    pub algo: i32,
    pub status: i32,
    pub determinism: i32,
}

/// The legacy descriptors of a problem: `x`, `w`, the convolution and `y`, destroyed when they are dropped.
#[derive(Debug)]
pub(crate) struct Legacy {
    x: sys::cudnnTensorDescriptor_t,
    w: sys::cudnnFilterDescriptor_t,
    conv: sys::cudnnConvolutionDescriptor_t,
    y: sys::cudnnTensorDescriptor_t,
}

// SAFETY: descriptors are plain host objects of the library, used under the mutex of the `Conv` that owns them.
#[allow(unsafe_code)]
unsafe impl Send for Legacy {}
#[allow(unsafe_code)]
unsafe impl Sync for Legacy {}

impl Legacy {
    /// The descriptors of `problem`, as Python's `_Legacy` makes them.
    #[allow(unsafe_code)]
    pub(crate) fn new(problem: &ConvProblem) -> Result<Self> {
        let mut out = Self {
            x: std::ptr::null_mut(),
            w: std::ptr::null_mut(),
            conv: std::ptr::null_mut(),
            y: std::ptr::null_mut(),
        };
        let (dt, fmt) = (data_type(problem.dtype), format(problem.layout));
        let (x_dims, w_dims, y_dims) = (ints(&problem.x_dims())?, ints(&problem.w_dims())?, ints(&problem.y_dims())?);
        let (pad, stride, dil) =
            (ints(&problem.conv_padding())?, ints(&problem.conv_stride())?, ints(&problem.conv_dilation())?);
        let groups =
            i32::try_from(problem.groups).map_err(|_| Error::Launch("the groups do not fit an int32".to_owned()))?;
        // SAFETY: every out-pointer is a field of `out`, which owns what is made (its `Drop` destroys what is not null); every array is a live
        // local of the length the call is told.
        unsafe {
            status("cudnnCreateTensorDescriptor", sys::cudnnCreateTensorDescriptor(&mut out.x))?;
            status(
                "cudnnSetTensorNdDescriptorEx",
                sys::cudnnSetTensorNdDescriptorEx(out.x, fmt, dt, x_dims.len() as i32, x_dims.as_ptr()),
            )?;
            status("cudnnCreateFilterDescriptor", sys::cudnnCreateFilterDescriptor(&mut out.w))?;
            status(
                "cudnnSetFilterNdDescriptor",
                sys::cudnnSetFilterNdDescriptor(out.w, dt, fmt, w_dims.len() as i32, w_dims.as_ptr()),
            )?;
            status("cudnnCreateConvolutionDescriptor", sys::cudnnCreateConvolutionDescriptor(&mut out.conv))?;
            status(
                "cudnnSetConvolutionNdDescriptor",
                sys::cudnnSetConvolutionNdDescriptor(
                    out.conv,
                    pad.len() as i32,
                    pad.as_ptr(),
                    stride.as_ptr(),
                    dil.as_ptr(),
                    sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                    sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
                ),
            )?;
            status("cudnnSetConvolutionGroupCount", sys::cudnnSetConvolutionGroupCount(out.conv, groups))?;
            status("cudnnSetConvolutionMathType", sys::cudnnSetConvolutionMathType(out.conv, math_type(problem.math)))?;
            status("cudnnCreateTensorDescriptor", sys::cudnnCreateTensorDescriptor(&mut out.y))?;
            status(
                "cudnnSetTensorNdDescriptorEx",
                sys::cudnnSetTensorNdDescriptorEx(out.y, fmt, dt, y_dims.len() as i32, y_dims.as_ptr()),
            )?;
        }
        Ok(out)
    }

    /// `cudnnGetConvolutionForwardAlgorithm_v7`: every answer, in its order.
    #[allow(unsafe_code)]
    pub(crate) fn candidates(&self, handle: &Handle) -> Result<Vec<Candidate>> {
        let mut count = 0i32;
        // SAFETY: the handle is live; `count` is a live local.
        status("cudnnGetConvolutionForwardAlgorithmMaxCount", unsafe {
            sys::cudnnGetConvolutionForwardAlgorithmMaxCount(handle.raw, &mut count)
        })?;
        let mut perf = vec![RawPerf::default(); usize::try_from(count.max(1)).unwrap_or(1)];
        let mut returned = 0i32;
        // SAFETY: the handle and the descriptors are live; `perf` holds `count` structs of the layout of `cudnnConvolutionFwdAlgoPerf_t` (asserted
        // above), written as integers; `returned` is a live local.
        status("cudnnGetConvolutionForwardAlgorithm_v7", unsafe {
            sys::cudnnGetConvolutionForwardAlgorithm_v7(
                handle.raw,
                self.x,
                self.w,
                self.conv,
                self.y,
                count,
                &mut returned,
                perf.as_mut_ptr().cast::<sys::cudnnConvolutionFwdAlgoPerf_t>(),
            )
        })?;
        let returned = usize::try_from(returned).unwrap_or(0).min(perf.len());
        Ok(perf[..returned]
            .iter()
            .map(|p| Candidate { algo: p.algo, status: p.status, determinism: p.determinism })
            .collect())
    }

    /// `cudnnGetConvolutionForwardWorkspaceSize` of `algo`, or `None` when the library refuses the algorithm for the problem.
    #[allow(unsafe_code)]
    pub(crate) fn workspace(&self, handle: &Handle, algo: u32) -> Result<Option<usize>> {
        let algo = algo_of(algo)?;
        let mut size = 0usize;
        // SAFETY: the handle and the descriptors are live; `size` is a live local.
        let s = unsafe {
            sys::cudnnGetConvolutionForwardWorkspaceSize(handle.raw, self.x, self.w, self.conv, self.y, algo, &mut size)
        };
        Ok((s == sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS).then_some(size))
    }

    /// Enqueue `cudnnConvolutionForward` of `algo` on the handle's stream: `y = 1.0 * conv(x, w) + 0.0 * y`, with a workspace of
    /// `workspace_size` bytes.
    ///
    /// # Safety
    ///
    /// The handle's stream must have been set to a valid stream of its context, which must be current on this thread; the addresses must be device
    /// memory of that context holding what the problem says, allocated until the work has completed, and the workspace at least `workspace_size`
    /// bytes (or null with a size of 0).
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn forward(
        &self,
        handle: &Handle,
        algo: u32,
        x: u64,
        w: u64,
        y: u64,
        workspace: u64,
        workspace_size: usize,
    ) -> Result<()> {
        let algo = algo_of(algo)?;
        let (alpha, beta) = (1.0f32, 0.0f32);
        // SAFETY: the handle and the descriptors are live; alpha and beta are live locals (float, for the f32 compute type); the rest is the
        // caller's contract.
        status("cudnnConvolutionForward", unsafe {
            sys::cudnnConvolutionForward(
                handle.raw,
                std::ptr::from_ref(&alpha).cast::<c_void>(),
                self.x,
                x as *const c_void,
                self.w,
                w as *const c_void,
                self.conv,
                algo,
                workspace as *mut c_void,
                workspace_size,
                std::ptr::from_ref(&beta).cast::<c_void>(),
                self.y,
                y as *mut c_void,
            )
        })
    }
}

impl Drop for Legacy {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        // SAFETY: each descriptor was made by its create call (or is null and skipped) and is destroyed once, here; failures are ignored.
        unsafe {
            if !self.y.is_null() {
                let _ = sys::cudnnDestroyTensorDescriptor(self.y);
            }
            if !self.conv.is_null() {
                let _ = sys::cudnnDestroyConvolutionDescriptor(self.conv);
            }
            if !self.w.is_null() {
                let _ = sys::cudnnDestroyFilterDescriptor(self.w);
            }
            if !self.x.is_null() {
                let _ = sys::cudnnDestroyTensorDescriptor(self.x);
            }
        }
    }
}

// -- the backend API -------------------------------------------------------------------------------------------------------------

/// A backend descriptor, destroyed when it is dropped.
#[derive(Debug)]
pub(crate) struct Desc(sys::cudnnBackendDescriptor_t);

// SAFETY: a backend descriptor is a host object of the library, used under the mutex of the `Conv` that owns it.
#[allow(unsafe_code)]
unsafe impl Send for Desc {}
#[allow(unsafe_code)]
unsafe impl Sync for Desc {}

impl Drop for Desc {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: made by `cudnnBackendCreateDescriptor`, destroyed once, here; a failure is ignored in a destructor.
            let _ = unsafe { sys::cudnnBackendDestroyDescriptor(self.0) };
        }
    }
}

impl Desc {
    #[allow(unsafe_code)]
    fn new(kind: DescType) -> Result<Self> {
        let mut raw: sys::cudnnBackendDescriptor_t = std::ptr::null_mut();
        // SAFETY: `raw` is a live local the library writes the descriptor into.
        status("cudnnBackendCreateDescriptor", unsafe { sys::cudnnBackendCreateDescriptor(kind, &mut raw) })?;
        Ok(Self(raw))
    }

    /// Set an attribute to `values`, `count` of them at `values` (of the attribute type's C type).
    #[allow(unsafe_code)]
    fn set_raw(&self, attr: Attr, kind: AttrType, count: usize, values: *const c_void) -> Result<()> {
        // SAFETY: the descriptor is live; `values` points at `count` live values of the type the attribute type names (the callers below pass
        // arrays of exactly that type and length).
        status("cudnnBackendSetAttribute", unsafe {
            sys::cudnnBackendSetAttribute(self.0, attr, kind, count as i64, values)
        })
    }

    fn set_i64s(&self, attr: Attr, values: &[i64]) -> Result<()> {
        self.set_raw(attr, AttrType::CUDNN_TYPE_INT64, values.len(), values.as_ptr().cast())
    }

    fn set_i64(&self, attr: Attr, value: i64) -> Result<()> {
        self.set_i64s(attr, &[value])
    }

    /// An enum-valued attribute (data type, convolution mode, heuristic mode, knob type): a C enum, which is an `int`.
    fn set_enum(&self, attr: Attr, kind: AttrType, value: i32) -> Result<()> {
        self.set_raw(attr, kind, 1, std::ptr::from_ref(&value).cast())
    }

    fn set_i32(&self, attr: Attr, value: i32) -> Result<()> {
        self.set_raw(attr, AttrType::CUDNN_TYPE_INT32, 1, std::ptr::from_ref(&value).cast())
    }

    fn set_f32(&self, attr: Attr, value: f32) -> Result<()> {
        self.set_raw(attr, AttrType::CUDNN_TYPE_FLOAT, 1, std::ptr::from_ref(&value).cast())
    }

    fn set_descs(&self, attr: Attr, values: &[&Desc]) -> Result<()> {
        let raw: Vec<sys::cudnnBackendDescriptor_t> = values.iter().map(|d| d.0).collect();
        self.set_raw(attr, AttrType::CUDNN_TYPE_BACKEND_DESCRIPTOR, raw.len(), raw.as_ptr().cast())
    }

    fn set_handle(&self, attr: Attr, handle: &Handle) -> Result<()> {
        self.set_raw(attr, AttrType::CUDNN_TYPE_HANDLE, 1, std::ptr::from_ref(&handle.raw).cast())
    }

    #[allow(unsafe_code)]
    fn finalize(&self) -> Result<()> {
        // SAFETY: the descriptor is live.
        status("cudnnBackendFinalize", unsafe { sys::cudnnBackendFinalize(self.0) })
    }

    /// Finalize, as an answer: whether the library accepts the descriptor.
    #[allow(unsafe_code)]
    fn try_finalize(&self) -> bool {
        // SAFETY: the descriptor is live.
        unsafe { sys::cudnnBackendFinalize(self.0) == sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS }
    }

    /// Up to `most` values of an attribute, into `out` (of the attribute type's C type); the count the library wrote.
    #[allow(unsafe_code)]
    fn get_raw(&self, attr: Attr, kind: AttrType, most: usize, out: *mut c_void) -> Result<usize> {
        let mut count = 0i64;
        // SAFETY: the descriptor is live; `out` has room for `most` values of the type the attribute type names (the callers pass arrays of
        // exactly that type and length); `count` is a live local.
        status("cudnnBackendGetAttribute", unsafe {
            sys::cudnnBackendGetAttribute(self.0, attr, kind, most as i64, &mut count, out)
        })?;
        Ok(usize::try_from(count).unwrap_or(0).min(most))
    }

    fn get_i64(&self, attr: Attr) -> Result<i64> {
        let mut value = [0i64; 1];
        self.get_raw(attr, AttrType::CUDNN_TYPE_INT64, 1, value.as_mut_ptr().cast())?;
        Ok(value[0])
    }

    /// An enum list (numerical or behavior notes, a knob type) read as integers.
    fn get_enums(&self, attr: Attr, kind: AttrType, most: usize) -> Result<Vec<i32>> {
        let mut values = vec![0i32; most];
        let n = self.get_raw(attr, kind, most, values.as_mut_ptr().cast())?;
        values.truncate(n);
        Ok(values)
    }

    /// A descriptor-valued attribute: `most` descriptors of `kind` are made to receive it (the library fills descriptors it is given).
    fn get_descs(&self, attr: Attr, kind: DescType, most: usize) -> Result<Vec<Desc>> {
        let mut made = Vec::with_capacity(most);
        for _ in 0..most {
            made.push(Desc::new(kind)?);
        }
        let mut raw: Vec<sys::cudnnBackendDescriptor_t> = made.iter().map(|d| d.0).collect();
        let n = self.get_raw(attr, AttrType::CUDNN_TYPE_BACKEND_DESCRIPTOR, most, raw.as_mut_ptr().cast())?;
        made.truncate(n);
        Ok(made)
    }
}

/// What an engine is: its global index, numerical notes and knobs (type, minimum, maximum, stride).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EngineFacts {
    pub engine: i64,
    pub notes: Vec<i32>,
    pub knobs: Vec<(i32, i64, i64, i64)>,
}

/// The operation graph of a problem on a handle, with the tensors, convolution and operation it was made of (kept alive with it).
#[derive(Debug)]
pub(crate) struct Graph {
    graph: Desc,
    _parts: Vec<Desc>,
}

impl Graph {
    /// The graph of `problem`, as Python's `_Graph` makes it.
    pub(crate) fn new(handle: &Handle, problem: &ConvProblem) -> Result<Self> {
        let dt = data_type(problem.dtype) as i32;
        let mut parts = Vec::new();
        let tensor = |dims: Vec<u64>, strides: Vec<u64>, uid: i64| -> Result<Desc> {
            let t = Desc::new(DescType::CUDNN_BACKEND_TENSOR_DESCRIPTOR)?;
            t.set_enum(Attr::CUDNN_ATTR_TENSOR_DATA_TYPE, AttrType::CUDNN_TYPE_DATA_TYPE, dt)?;
            t.set_i64s(Attr::CUDNN_ATTR_TENSOR_DIMENSIONS, &longs(&dims)?)?;
            t.set_i64s(Attr::CUDNN_ATTR_TENSOR_STRIDES, &longs(&strides)?)?;
            t.set_i64(Attr::CUDNN_ATTR_TENSOR_UNIQUE_ID, uid)?;
            t.set_i64(Attr::CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, problem.alignment as i64)?;
            t.finalize()?;
            Ok(t)
        };
        let x = tensor(problem.x_dims(), problem.x_strides(), UID_X)?;
        let w = tensor(problem.w_dims(), problem.w_strides(), UID_W)?;
        let y = tensor(problem.y_dims(), problem.y_strides(), UID_Y)?;
        let conv = Desc::new(DescType::CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR)?;
        conv.set_enum(
            Attr::CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
            AttrType::CUDNN_TYPE_DATA_TYPE,
            sys::cudnnDataType_t::CUDNN_DATA_FLOAT as i32,
        )?;
        conv.set_enum(
            Attr::CUDNN_ATTR_CONVOLUTION_CONV_MODE,
            AttrType::CUDNN_TYPE_CONVOLUTION_MODE,
            sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION as i32,
        )?;
        let pad = longs(&problem.conv_padding())?;
        conv.set_i64(Attr::CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, pad.len() as i64)?;
        conv.set_i64s(Attr::CUDNN_ATTR_CONVOLUTION_DILATIONS, &longs(&problem.conv_dilation())?)?;
        conv.set_i64s(Attr::CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, &longs(&problem.conv_stride())?)?;
        conv.set_i64s(Attr::CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, &pad)?;
        conv.set_i64s(Attr::CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, &pad)?;
        conv.finalize()?;
        let op = Desc::new(DescType::CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)?;
        op.set_descs(Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, &[&x])?;
        op.set_descs(Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, &[&w])?;
        op.set_descs(Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, &[&y])?;
        op.set_descs(Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, &[&conv])?;
        op.set_f32(Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, 1.0)?;
        op.set_f32(Attr::CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, 0.0)?;
        op.finalize()?;
        let graph = Desc::new(DescType::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)?;
        graph.set_handle(Attr::CUDNN_ATTR_OPERATIONGRAPH_HANDLE, handle)?;
        graph.set_descs(Attr::CUDNN_ATTR_OPERATIONGRAPH_OPS, &[&op])?;
        graph.finalize()?;
        parts.extend([x, w, y, conv, op]);
        Ok(Self { graph, _parts: parts })
    }

    /// The engine of a global index (with an SM count target), finalized, or `None` when it is not an engine of the graph.
    pub(crate) fn engine(&self, index: i64, sm_count_target: Option<u64>) -> Result<Option<Desc>> {
        let engine = Desc::new(DescType::CUDNN_BACKEND_ENGINE_DESCRIPTOR)?;
        engine.set_descs(Attr::CUDNN_ATTR_ENGINE_OPERATION_GRAPH, &[&self.graph])?;
        engine.set_i64(Attr::CUDNN_ATTR_ENGINE_GLOBAL_INDEX, index)?;
        if let Some(target) = sm_count_target {
            let target = i32::try_from(target)
                .map_err(|_| Error::Launch("the SM count target does not fit an int32".to_owned()))?;
            engine.set_i32(Attr::CUDNN_ATTR_ENGINE_SM_COUNT_TARGET, target)?;
        }
        Ok(engine.try_finalize().then_some(engine))
    }

    /// The facts of a finalized engine.
    pub(crate) fn facts(engine: &Desc) -> Result<EngineFacts> {
        let index = engine.get_i64(Attr::CUDNN_ATTR_ENGINE_GLOBAL_INDEX)?;
        let notes =
            engine.get_enums(Attr::CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, AttrType::CUDNN_TYPE_NUMERICAL_NOTE, MAX_NOTES)?;
        let mut knobs = Vec::new();
        for info in engine.get_descs(
            Attr::CUDNN_ATTR_ENGINE_KNOB_INFO,
            DescType::CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR,
            KNOB_TYPE_COUNT,
        )? {
            let t = info.get_enums(Attr::CUDNN_ATTR_KNOB_INFO_TYPE, AttrType::CUDNN_TYPE_KNOB_TYPE, 1)?;
            let lo = info.get_i64(Attr::CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE)?;
            let hi = info.get_i64(Attr::CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE)?;
            let step = info.get_i64(Attr::CUDNN_ATTR_KNOB_INFO_STRIDE)?;
            knobs.push((t.first().copied().unwrap_or(-1), lo, hi, step));
        }
        Ok(EngineFacts { engine: index, notes, knobs })
    }

    /// The engine configuration of an engine and knob choices, finalized, or `None` when the library refuses it.
    pub(crate) fn config(engine: &Desc, knobs: &[(i32, i64)]) -> Result<Option<Desc>> {
        let mut choices = Vec::with_capacity(knobs.len());
        for (t, v) in knobs {
            let choice = Desc::new(DescType::CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR)?;
            choice.set_enum(Attr::CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, AttrType::CUDNN_TYPE_KNOB_TYPE, *t)?;
            choice.set_i64(Attr::CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, *v)?;
            choice.finalize()?;
            choices.push(choice);
        }
        let config = Desc::new(DescType::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR)?;
        config.set_descs(Attr::CUDNN_ATTR_ENGINECFG_ENGINE, &[engine])?;
        if !choices.is_empty() {
            let refs: Vec<&Desc> = choices.iter().collect();
            config.set_descs(Attr::CUDNN_ATTR_ENGINECFG_KNOB_CHOICES, &refs)?;
        }
        Ok(config.try_finalize().then_some(config))
    }

    /// The execution plan of a configuration, finalized, or `None` when it does not finalize; and its workspace.
    pub(crate) fn plan(handle: &Handle, config: &Desc) -> Result<Option<(Desc, usize)>> {
        let plan = Desc::new(DescType::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)?;
        plan.set_handle(Attr::CUDNN_ATTR_EXECUTION_PLAN_HANDLE, handle)?;
        plan.set_descs(Attr::CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, &[config])?;
        if !plan.try_finalize() {
            return Ok(None);
        }
        let workspace = plan.get_i64(Attr::CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE)?;
        Ok(Some((plan, usize::try_from(workspace).unwrap_or(usize::MAX))))
    }
}

/// Enqueue `cudnnBackendExecute` of `plan` on the handle's stream, with a variant pack of the three tensors and the workspace.
///
/// # Safety
///
/// As [`Legacy::forward`]: the handle's stream is a valid stream of its context, current on this thread; the addresses hold what the problem says
/// and stay allocated until the work has completed; the workspace has at least the plan's workspace size (or is null when it needs none).
#[allow(unsafe_code)]
pub(crate) unsafe fn execute(handle: &Handle, plan: &Desc, x: u64, w: u64, y: u64, workspace: u64) -> Result<()> {
    let pack = Desc::new(DescType::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR)?;
    let uids = [UID_X, UID_Y, UID_W];
    let pointers: [*mut c_void; 3] = [x as *mut c_void, y as *mut c_void, w as *mut c_void];
    pack.set_i64s(Attr::CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, &uids)?;
    pack.set_raw(
        Attr::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
        AttrType::CUDNN_TYPE_VOID_PTR,
        3,
        pointers.as_ptr().cast(),
    )?;
    let ws = workspace as *mut c_void;
    pack.set_raw(
        Attr::CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
        AttrType::CUDNN_TYPE_VOID_PTR,
        1,
        std::ptr::from_ref(&ws).cast(),
    )?;
    pack.finalize()?;
    // SAFETY: the handle, the plan and the pack are live; the rest is the caller's contract.
    status("cudnnBackendExecute", unsafe { sys::cudnnBackendExecute(handle.raw, plan.0, pack.0) })
}
