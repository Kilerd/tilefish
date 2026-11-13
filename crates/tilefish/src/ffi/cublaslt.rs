//! cuBLASLt through cudarc's `cublaslt::result` and `cublaslt::sys`: what one pinned GEMM needs, as safe items over the `unsafe` calls.
//!
//! cudarc's safe `CudaBlasLT::matmul` always asks the heuristic for an algorithm and cannot take a pinned one, so this crate calls the
//! library the way Python's `_cublaslt.py` does: a matmul descriptor and the matrix layouts of the problem, `cublasLtMatmulAlgoInit` and
//! the configuration and capability attributes (to vet 64 bytes of an algorithm that only the library that made them can judge),
//! `cublasLtMatmulAlgoCheck` and `cublasLtMatmul`. Every `unsafe` call is in this file, with the reason it is sound beside it; what is
//! decided with the answers (the vetting, the order of the calls) is `crate::gemm`, which has no `unsafe`.
//!
//! The library is cudarc's dynamically loaded `libcublasLt`: it is opened when the first function of it is called, and cudarc **panics**
//! when it is missing, so every entry point here is behind [`available`], which the callers check first.

use std::ffi::c_void;
use std::mem::size_of;

use cudarc::cublaslt::result;
use cudarc::cublaslt::sys;
use cudarc::driver::CudaContext;

use crate::error::{Error, Result};

const COMPUTE_32F: sys::cublasComputeType_t = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F;
const R_32F: sys::cudaDataType = sys::cudaDataType::CUDA_R_32F;
const R_16BF: sys::cudaDataType = sys::cudaDataType::CUDA_R_16BF;
/// `cublasOperation_t`: `CUBLAS_OP_N` and `CUBLAS_OP_T`.
const OP_N: i32 = 0;
const OP_T: i32 = 1;
/// `cublasLtEpilogue_t`: `CUBLASLT_EPILOGUE_DEFAULT` and `CUBLASLT_EPILOGUE_BIAS`.
const EPILOGUE_DEFAULT: u32 = 1;
const EPILOGUE_BIAS: u32 = 4;

/// Whether libcublasLt can be loaded on this machine (it is opened and kept open when it is there; nothing in it is called).
#[allow(unsafe_code)]
pub(crate) fn available() -> bool {
    // SAFETY: `is_culib_present` only tries to open the library by its usual names, running its initializers as any use of it does, and
    // calls no function in it.
    unsafe { sys::is_culib_present() }
}

/// `cublasLtGetVersion()`: the version of the library this process has (`120804` for 12.8.4). The caller has checked [`available`].
#[allow(unsafe_code)]
pub(crate) fn version() -> u64 {
    // SAFETY: the function takes no arguments, and the library is there (the caller checked).
    unsafe { sys::cublasLtGetVersion() as u64 }
}

/// A cuBLASLt handle, destroyed when it is dropped. It holds no per-call state (the workspace, the descriptors and the pointers are arguments
/// of a call), so one handle serves every launch of a GEMM, from any thread.
#[derive(Debug)]
pub(crate) struct Handle {
    raw: sys::cublasLtHandle_t,
}

// SAFETY: a cuBLASLt handle is not tied to a thread, and the library documents its calls on one handle as thread safe (as cudarc's own
// `CudaBlasLT`, which holds the same raw handle, asserts); this crate's calls on it (`cublasLtMatmul`, `cublasLtMatmulAlgoCheck` and the
// algorithm queries) take everything else as arguments.
#[allow(unsafe_code)]
unsafe impl Send for Handle {}
#[allow(unsafe_code)]
unsafe impl Sync for Handle {}

impl Handle {
    /// A handle for the device of `ctx`, made with that context current on this thread.
    pub(crate) fn new(ctx: &CudaContext) -> Result<Self> {
        ctx.bind_to_thread().map_err(Error::driver("cuCtxSetCurrent"))?;
        let raw = result::create_handle().map_err(Error::cublas("cublasLtCreate"))?;
        Ok(Self { raw })
    }
}

impl Drop for Handle {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        // SAFETY: `raw` was made by `create_handle` and is destroyed once, here. A failure is ignored: there is nothing to do about it in a
        // destructor.
        let _ = unsafe { result::destroy_handle(self.raw) };
    }
}

/// The 64 bytes of a `cublasLtMatmulAlgo_t` (eight `uint64_t`, in memory order, as the host stores them).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Algo(sys::cublasLtMatmulAlgo_t);

impl Algo {
    /// The algorithm of these bytes (little-endian host: the crate refuses to build on another).
    pub(crate) fn from_bytes(bytes: &[u8; 64]) -> Self {
        let mut data = [0u64; 8];
        let (words, _) = bytes.as_chunks::<8>();
        for (word, chunk) in data.iter_mut().zip(words) {
            *word = u64::from_le_bytes(*chunk);
        }
        Self(sys::cublasLtMatmulAlgo_t { data })
    }

    /// The 64 bytes of the algorithm.
    pub(crate) fn to_bytes(self) -> [u8; 64] {
        let mut out = [0u8; 64];
        let (words, _) = out.as_chunks_mut::<8>();
        for (chunk, word) in words.iter_mut().zip(self.0.data) {
            *chunk = word.to_le_bytes();
        }
        out
    }

    /// A fresh algorithm of the library's id `id` for bf16 operands, bf16 results and an f32 accumulator and scale
    /// (`cublasLtMatmulAlgoInit`): what the library itself would write for that configuration, to compare 64 bytes with.
    #[allow(unsafe_code)]
    pub(crate) fn init(handle: &Handle, id: i32) -> Result<Self> {
        let mut algo = sys::cublasLtMatmulAlgo_t { data: [0; 8] };
        // SAFETY: `handle` is live; `algo` is a live local struct the library fills.
        let status = unsafe {
            sys::cublasLtMatmulAlgoInit(handle.raw, COMPUTE_32F, R_32F, R_16BF, R_16BF, R_16BF, R_16BF, id, &mut algo)
        };
        status.result().map_err(Error::cublas("cublasLtMatmulAlgoInit"))?;
        Ok(Self(algo))
    }
}

/// What an algorithm is made of (`cublasLtMatmulAlgoConfigGetAttribute`), each field in the integer type the header gives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AlgoConfig {
    pub id: i32,
    pub tile_id: u32,
    pub splitk_num: i32,
    pub reduction_scheme: u32,
    pub cta_swizzling: u32,
    pub custom_option: u32,
    pub stages_id: u32,
    pub inner_shape_id: u16,
    pub cluster_shape_id: u16,
}

type ConfigAttr = sys::cublasLtMatmulAlgoConfigAttributes_t;

#[allow(unsafe_code)]
fn config_get<T: Copy + Default>(algo: &Algo, attr: ConfigAttr, call: &'static str) -> Result<T> {
    let (mut value, mut written) = (T::default(), 0usize);
    // SAFETY: `algo` is a live algorithm struct; the buffer is the live local `value`, of the `size_of::<T>()` bytes it is told, a plain integer.
    let status = unsafe {
        sys::cublasLtMatmulAlgoConfigGetAttribute(
            &algo.0,
            attr,
            std::ptr::from_mut(&mut value).cast::<c_void>(),
            size_of::<T>(),
            &mut written,
        )
    };
    status.result().map_err(Error::cublas(call))?;
    Ok(value)
}

#[allow(unsafe_code)]
fn config_set<T: Copy>(algo: &mut Algo, attr: ConfigAttr, value: T, call: &'static str) -> Result<()> {
    // SAFETY: `algo` is a live algorithm struct; the buffer is the live local `value`, of the `size_of::<T>()` bytes it is told, a plain integer.
    let status = unsafe {
        sys::cublasLtMatmulAlgoConfigSetAttribute(
            &mut algo.0,
            attr,
            std::ptr::from_ref(&value).cast::<c_void>(),
            size_of::<T>(),
        )
    };
    status.result().map_err(Error::cublas(call))
}

impl Algo {
    /// The configuration the bytes read back as.
    pub(crate) fn config(&self) -> Result<AlgoConfig> {
        use sys::cublasLtMatmulAlgoConfigAttributes_t as A;
        const CALL: &str = "cublasLtMatmulAlgoConfigGetAttribute";
        Ok(AlgoConfig {
            id: config_get(self, A::CUBLASLT_ALGO_CONFIG_ID, CALL)?,
            tile_id: config_get(self, A::CUBLASLT_ALGO_CONFIG_TILE_ID, CALL)?,
            splitk_num: config_get(self, A::CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CALL)?,
            reduction_scheme: config_get(self, A::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CALL)?,
            cta_swizzling: config_get(self, A::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, CALL)?,
            custom_option: config_get(self, A::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, CALL)?,
            stages_id: config_get(self, A::CUBLASLT_ALGO_CONFIG_STAGES_ID, CALL)?,
            inner_shape_id: config_get(self, A::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, CALL)?,
            cluster_shape_id: config_get(self, A::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, CALL)?,
        })
    }

    /// Set every field of `config` but the id on this algorithm (`cublasLtMatmulAlgoConfigSetAttribute`).
    pub(crate) fn set_config(&mut self, config: &AlgoConfig) -> Result<()> {
        use sys::cublasLtMatmulAlgoConfigAttributes_t as A;
        const CALL: &str = "cublasLtMatmulAlgoConfigSetAttribute";
        config_set(self, A::CUBLASLT_ALGO_CONFIG_TILE_ID, config.tile_id, CALL)?;
        config_set(self, A::CUBLASLT_ALGO_CONFIG_SPLITK_NUM, config.splitk_num, CALL)?;
        config_set(self, A::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, config.reduction_scheme, CALL)?;
        config_set(self, A::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, config.cta_swizzling, CALL)?;
        config_set(self, A::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, config.custom_option, CALL)?;
        config_set(self, A::CUBLASLT_ALGO_CONFIG_STAGES_ID, config.stages_id, CALL)?;
        config_set(self, A::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, config.inner_shape_id, CALL)?;
        config_set(self, A::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, config.cluster_shape_id, CALL)
    }
}

type CapAttr = sys::cublasLtMatmulAlgoCapAttributes_t;

#[allow(unsafe_code)]
fn cap_get<T: Copy + Default>(algo: &Algo, attr: CapAttr) -> Result<T> {
    let (mut value, mut written) = (T::default(), 0usize);
    // SAFETY: as `config_get`.
    let status = unsafe {
        sys::cublasLtMatmulAlgoCapGetAttribute(
            &algo.0,
            attr,
            std::ptr::from_mut(&mut value).cast::<c_void>(),
            size_of::<T>(),
            &mut written,
        )
    };
    status.result().map_err(Error::cublas("cublasLtMatmulAlgoCapGetAttribute"))?;
    Ok(value)
}

/// A capability that is a list of `u32` (the first call, with no buffer, says how many bytes the list has).
#[allow(unsafe_code)]
fn cap_get_list(algo: &Algo, attr: CapAttr) -> Result<Vec<u32>> {
    const CALL: &str = "cublasLtMatmulAlgoCapGetAttribute";
    let mut written = 0usize;
    // SAFETY: a size query: a null buffer of size 0 is the documented way to ask how many bytes the list has.
    let status =
        unsafe { sys::cublasLtMatmulAlgoCapGetAttribute(&algo.0, attr, std::ptr::null_mut(), 0, &mut written) };
    status.result().map_err(Error::cublas(CALL))?;
    let mut values = vec![0u32; written / size_of::<u32>()];
    // SAFETY: the buffer is `values`, of exactly the `size_of_val(&values)` bytes it is told.
    let status = unsafe {
        sys::cublasLtMatmulAlgoCapGetAttribute(
            &algo.0,
            attr,
            values.as_mut_ptr().cast::<c_void>(),
            std::mem::size_of_val(values.as_slice()),
            &mut written,
        )
    };
    status.result().map_err(Error::cublas(CALL))?;
    Ok(values)
}

/// What an algorithm can do (`cublasLtMatmulAlgoCapGetAttribute`).
#[derive(Debug, Clone)]
pub(crate) struct Caps {
    pub tile_ids: Vec<u32>,
    pub stages_ids: Vec<u32>,
    pub cta_swizzling: u32,
    pub custom_option_max: i32,
    pub numerical_impl_flags: u64,
    /// The alignment in bytes the algorithm needs of A, B, C and D.
    pub min_alignment: [u32; 4],
}

impl Algo {
    /// The capabilities of this algorithm (its id's: ask a fresh one made by [`Algo::init`]).
    pub(crate) fn caps(&self) -> Result<Caps> {
        use sys::cublasLtMatmulAlgoCapAttributes_t as C;
        Ok(Caps {
            tile_ids: cap_get_list(self, C::CUBLASLT_ALGO_CAP_TILE_IDS)?,
            stages_ids: cap_get_list(self, C::CUBLASLT_ALGO_CAP_STAGES_IDS)?,
            cta_swizzling: cap_get(self, C::CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT)?,
            custom_option_max: cap_get(self, C::CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX)?,
            numerical_impl_flags: cap_get(self, C::CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS)?,
            min_alignment: [
                cap_get(self, C::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES)?,
                cap_get(self, C::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES)?,
                cap_get(self, C::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES)?,
                cap_get(self, C::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES)?,
            ],
        })
    }
}

/// A matrix layout, destroyed when it is dropped.
#[derive(Debug)]
struct Layout(sys::cublasLtMatrixLayout_t);

impl Layout {
    /// A column-major bf16 matrix of `rows` x `cols` with leading dimension `ld`.
    fn bf16(rows: u64, cols: u64, ld: i64) -> Result<Self> {
        Ok(Self(
            result::create_matrix_layout(R_16BF, rows, cols, ld)
                .map_err(Error::cublas("cublasLtMatrixLayoutCreate"))?,
        ))
    }
}

impl Drop for Layout {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        if self.0.is_null() {
            return; // a problem whose construction failed before this layout was made
        }
        // SAFETY: made by `create_matrix_layout`, destroyed once, here; a failure is ignored in a destructor.
        let _ = unsafe { result::destroy_matrix_layout(self.0) };
    }
}

/// The descriptors of one call: the matmul descriptor and the layouts of the two operands and of `D` (`C` is the same). Made per call and
/// destroyed at its end, so nothing mutable is shared between threads (the bias pointer is in the descriptor).
///
/// The problem is the one `docs/manifest-schema-3.md` (section 9) gives, tilefish's `out[m, n] = a[m, k] @ b[n, k]^T (+ bias[n])` posed to the
/// column-major library as the TN gemm `m_cublas = n`, `n_cublas = m`, `k_cublas = k`: the cuBLAS "A" operand is tilefish's `b` (`transa = T`,
/// a `k x n` column-major matrix with `lda = k`), the "B" operand is tilefish's `a` (`transb = N`, `k x m`, `ldb = k`), and `C` and `D` are
/// tilefish's `out` (`n x m` column-major, `ldc = ldd = n`); every layout is `CUDA_R_16BF`, the compute type `CUBLAS_COMPUTE_32F`, the scale type
/// `CUDA_R_32F`. A bias goes through `CUBLASLT_EPILOGUE_BIAS` with a `CUDA_R_16BF` bias pointer. Python's `describe` is the reference.
#[derive(Debug)]
pub(crate) struct Problem {
    desc: sys::cublasLtMatmulDesc_t,
    a: Layout,
    b: Layout,
    d: Layout,
}

#[allow(unsafe_code)]
fn desc_set<T: Copy>(
    desc: sys::cublasLtMatmulDesc_t,
    attr: sys::cublasLtMatmulDescAttributes_t,
    value: T,
) -> Result<()> {
    // SAFETY: `desc` is a live descriptor (the caller made it and has not destroyed it); the buffer is the live local `value`, of the
    // `size_of::<T>()` bytes it is told, a plain integer or pointer.
    unsafe {
        result::set_matmul_desc_attribute(desc, attr, std::ptr::from_ref(&value).cast::<c_void>(), size_of::<T>())
    }
    .map_err(Error::cublas("cublasLtMatmulDescSetAttribute"))
}

impl Problem {
    /// The problem of sizes `m`, `n`, `k` (tilefish's), with `bias` the address of the bias vector for a `gemm_bias`.
    pub(crate) fn new(m: u64, n: u64, k: u64, bias: Option<u64>) -> Result<Self> {
        use sys::cublasLtMatmulDescAttributes_t as D;
        let ld = |v: u64| {
            i64::try_from(v).map_err(|_| Error::Launch("a size of the problem does not fit 64 bits".to_owned()))
        };
        let desc = result::create_matmul_desc(COMPUTE_32F, R_32F).map_err(Error::cublas("cublasLtMatmulDescCreate"))?;
        // from here on the descriptor is owned by `problem`, so an error destroys it
        let mut problem = Self {
            desc,
            a: Layout(std::ptr::null_mut()),
            b: Layout(std::ptr::null_mut()),
            d: Layout(std::ptr::null_mut()),
        };
        desc_set(desc, D::CUBLASLT_MATMUL_DESC_TRANSA, OP_T)?;
        desc_set(desc, D::CUBLASLT_MATMUL_DESC_TRANSB, OP_N)?;
        match bias {
            Some(pointer) => {
                desc_set(desc, D::CUBLASLT_MATMUL_DESC_EPILOGUE, EPILOGUE_BIAS)?;
                desc_set(desc, D::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, R_16BF as i32)?;
                desc_set(desc, D::CUBLASLT_MATMUL_DESC_BIAS_POINTER, pointer as *const c_void)?;
            }
            None => desc_set(desc, D::CUBLASLT_MATMUL_DESC_EPILOGUE, EPILOGUE_DEFAULT)?,
        }
        // the "A operand" is tilefish's b[n, k]: stored k x n column-major, lda = k, used transposed (n x k)
        problem.a = Layout::bf16(k, n, ld(k)?)?;
        // the "B operand" is tilefish's a[m, k]: stored k x m column-major, ldb = k, used as it is
        problem.b = Layout::bf16(k, m, ld(k)?)?;
        // out[m, n] row-major is n x m column-major, ldd = n; C has the layout of D
        problem.d = Layout::bf16(n, m, ld(n)?)?;
        Ok(problem)
    }
}

impl Drop for Problem {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        // SAFETY: made by `create_matmul_desc`, destroyed once, here; a failure is ignored in a destructor.
        let _ = unsafe { result::destroy_matmul_desc(self.desc) };
    }
}

/// What `cublasLtMatmulAlgoCheck` says of an algorithm for a problem: the workspace it needs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Checked {
    /// The bytes of workspace the algorithm needs for the problem.
    pub workspace: usize,
}

/// Ask the library whether `algo` is valid for `problem` (`cublasLtMatmulAlgoCheck`), and how much workspace it needs.
#[allow(unsafe_code)]
pub(crate) fn check(handle: &Handle, problem: &Problem, algo: &Algo) -> Result<Checked> {
    let mut answer = sys::cublasLtMatmulHeuristicResult_t {
        algo: sys::cublasLtMatmulAlgo_t { data: [0; 8] },
        workspaceSize: 0,
        state: sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        wavesCount: 0.0,
        reserved: [0; 4],
    };
    // SAFETY: the handle, the descriptor and the layouts are live; `algo` and `answer` are live locals.
    let status = unsafe {
        sys::cublasLtMatmulAlgoCheck(
            handle.raw,
            problem.desc,
            problem.a.0,
            problem.b.0,
            problem.d.0,
            problem.d.0,
            &algo.0,
            &mut answer,
        )
    };
    status.result().map_err(Error::cublas("cublasLtMatmulAlgoCheck"))?;
    answer.state.result().map_err(Error::cublas("cublasLtMatmulAlgoCheck (its state)"))?;
    Ok(Checked { workspace: answer.workspaceSize })
}

/// The addresses of one call: tilefish's `a`, `b`, `out` (and, in the descriptor of the problem, the bias) and the workspace.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Addresses {
    pub a: u64,
    pub b: u64,
    pub out: u64,
    /// The workspace, or 0 (a null pointer) when the algorithm needs none.
    pub workspace: u64,
}

/// Enqueue `cublasLtMatmul` of the pinned algorithm on `stream`: `out = 1.0 * (a @ b^T) + 0.0 * out` (with the problem's epilogue), the
/// workspace being `workspace_size` bytes at `addresses.workspace`.
///
/// # Safety
///
/// `stream` must be a valid `CUstream` of the context the handle was made in, which must be current on this thread; the addresses must be
/// device memory of that context holding what `problem` says (`a` `[m, k]`, `b` `[n, k]`, `out` `[m, n]`, bf16, and the bias `[n]` of the
/// problem), allocated until the work has completed on `stream`, and the workspace at least `workspace_size` bytes.
#[allow(unsafe_code)]
pub(crate) unsafe fn matmul(
    handle: &Handle,
    problem: &Problem,
    algo: &Algo,
    addresses: &Addresses,
    workspace_size: usize,
    stream: cudarc::driver::sys::CUstream,
) -> Result<()> {
    let (alpha, beta) = (1.0f32, 0.0f32);
    // cuBLAS A operand = tilefish b, cuBLAS B operand = tilefish a; C is D (beta = 0, so it is never read)
    // SAFETY: the handle, the descriptor, the layouts and the algorithm are live; alpha and beta are live locals (host pointer mode); the
    // rest is the caller's contract.
    unsafe {
        result::matmul(
            handle.raw,
            problem.desc,
            std::ptr::from_ref(&alpha).cast::<c_void>(),
            std::ptr::from_ref(&beta).cast::<c_void>(),
            addresses.b as *const c_void,
            problem.a.0,
            addresses.a as *const c_void,
            problem.b.0,
            addresses.out as *const c_void,
            problem.d.0,
            addresses.out as *mut c_void,
            problem.d.0,
            &algo.0,
            addresses.workspace as *mut c_void,
            workspace_size,
            stream.cast(),
        )
    }
    .map_err(Error::cublas("cublasLtMatmul"))
}
