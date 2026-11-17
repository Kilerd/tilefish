//! The canary of a pinned library entry: operands made of a seed, and the digest of the result the pin computes from them, both made on the device
//! by the two kernels of `canary.ptx` (`docs/en/manifest-schema-3.md`, section 9.4; a cuDNN convolution's canary, `crate::conv`, and a cuBLASLt
//! matmul's, section 9.5, `crate::gemm`).
//!
//! The PTX is the same text as Python's `tilefish._canary.CANARY_PTX`, which the driver compiles for the device it is loaded on, once per context
//! while a load holds it ([`kernels`]). [`canary_bytes`] and [`canary_digest`] are the same computations on the host: references, for the tests.

use std::sync::{Arc, Mutex, Weak};

use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::error::{Error, Result};
use crate::tensor::DType;

/// The PTX of the canary's kernels (the operands and the digest), the same text as Python's `tilefish._canary.CANARY_PTX`: the driver compiles
/// it for the device it is loaded on.
const CANARY_PTX: &str = include_str!("canary.ptx");
/// The largest seed of a canary (a `u32`).
pub(crate) const MAX_CANARY_SEED: u64 = (1 << 32) - 1;
const CANARY_BLOCK: u32 = 256;
const CANARY_FILL_BLOCKS: u64 = 65535;
const CANARY_DIGEST_BLOCKS: u64 = 1024;
const MIX_GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const DIGEST_K1: u64 = 0x243F_6A88_85A3_08D3;
const DIGEST_K2: u64 = 0x1319_8A2E_0370_7344;

/// splitmix64's finalizer of `v + 0x9E3779B97F4A7C15` (wrapping), the mix of the canary's kernels.
fn mix64(v: u64) -> u64 {
    let mut z = v.wrapping_add(MIX_GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The base of a canary operand's elements: `mix(2 * seed + tag)`.
fn canary_base(seed: u64, tag: u8) -> u64 {
    mix64(2 * seed + u64::from(tag))
}

/// The most significant byte of a canary element: f16 an exponent field of 7 to 14, bf16 and f32 exponents 110 to 125; the sign and the low bits stay.
fn canary_top_byte(dtype: DType, b: u32) -> u32 {
    match dtype {
        DType::F16 => (b & 0x80) | ((7 + ((b >> 2) & 7)) << 2) | (b & 3),
        _ => (b & 0x80) | (55 + (b & 7)),
    }
}

/// The first `elements` elements of a canary operand of `dtype` (f16, bf16 or f32; tag 0 for a convolution's `x` and a GEMM's `a`, 1 for `w` and
/// `b`, 2 for a GEMM's `bias`, 3 for its `c`), little-endian, made on the host: what the fill kernel of `canary.ptx` makes on the device (Python's
/// `tilefish._canary.canary_bytes`). Element `i` is made of `h = mix(base + i)`, `base = mix(2 * seed + tag)`: its low 16 bits (f16, bf16) or 32
/// bits (f32), the most significant byte put in a range of exponents (finite, normal, many orders of magnitude: sums whose rounding depends on their
/// order). A reference, for the tests.
pub fn canary_bytes(dtype: DType, seed: u64, tag: u8, elements: u64) -> Vec<u8> {
    let first = canary_base(seed, tag);
    let mut out = Vec::with_capacity(elements as usize * dtype.size());
    for i in 0..elements {
        let h = mix64(first.wrapping_add(i));
        if dtype == DType::F32 {
            let r = h as u32;
            out.extend_from_slice(&((canary_top_byte(dtype, r >> 24) << 24) | (r & 0x00FF_FFFF)).to_le_bytes());
        } else {
            let r = (h & 0xFFFF) as u32;
            out.extend_from_slice(&(((canary_top_byte(dtype, r >> 8) << 8) | (r & 0xFF)) as u16).to_le_bytes());
        }
    }
    out
}

/// The canary's digest of a result's bytes, on the host: what the digest kernel of `canary.ptx` adds on the device (Python's
/// `tilefish._canary.digest_of`). The bytes as 32-bit little-endian words `w_j` (zero-padded to a whole word), `s1 = sum w_j * (mix(j ^ K1) | 1)` and
/// `s2 = sum w_j * (mix(j ^ K2) | 1)` modulo 2^64, written as 16 lowercase hex digits each: a sum that does not depend on its order, which one changed
/// word always changes.
pub fn canary_digest(bytes: &[u8]) -> String {
    let (mut s1, mut s2) = (0u64, 0u64);
    for (j, chunk) in bytes.chunks(4).enumerate() {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        let w = u64::from(u32::from_le_bytes(word));
        s1 = s1.wrapping_add(w.wrapping_mul(mix64(j as u64 ^ DIGEST_K1) | 1));
        s2 = s2.wrapping_add(w.wrapping_mul(mix64(j as u64 ^ DIGEST_K2) | 1));
    }
    format!("{s1:016x}{s2:016x}")
}

/// Whether `h` is 32 lowercase hex digits (a canary's digest).
pub(crate) fn is_digest32(h: &str) -> bool {
    h.len() == 32 && h.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The kernels of `canary.ptx` in one context: one module per context, shared by every canary run there while a load holds it.
#[derive(Debug)]
pub(crate) struct Kernels {
    fill: CudaFunction,
    digest: CudaFunction,
}

/// The canary kernels of each context that has them alive, by the address of its `CUcontext` (an entry whose kernels are gone is replaced; one that
/// is alive keeps its module, and the module its context, so no other context has that address meanwhile).
static KERNELS: Mutex<Vec<(usize, Weak<Kernels>)>> = Mutex::new(Vec::new());

/// The canary's kernels in `ctx`, loaded (the driver compiles `canary.ptx`) when the context has none alive.
pub(crate) fn kernels(ctx: &Arc<CudaContext>) -> Result<Arc<Kernels>> {
    let key = ctx.cu_ctx() as usize;
    let mut all = KERNELS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    all.retain(|(_, weak)| weak.strong_count() > 0);
    if let Some(found) = all.iter().find(|(k, _)| *k == key).and_then(|(_, weak)| weak.upgrade()) {
        return Ok(found);
    }
    let module = ctx
        .load_module(cudarc::nvrtc::Ptx::from_src(CANARY_PTX))
        .map_err(Error::driver("cuModuleLoadData(canary.ptx)"))?;
    let fill = module.load_function("tilefish_canary_fill").map_err(Error::driver("cuModuleGetFunction"))?;
    let digest = module.load_function("tilefish_canary_digest").map_err(Error::driver("cuModuleGetFunction"))?;
    let kernels = Arc::new(Kernels { fill, digest });
    all.push((key, Arc::downgrade(&kernels)));
    Ok(kernels)
}

/// The grid of a canary kernel over `count` items, at most `most` blocks (the kernels loop over what one grid does not cover).
fn blocks(count: u64, most: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (count.div_ceil(u64::from(CANARY_BLOCK)).clamp(1, most) as u32, 1, 1),
        block_dim: (CANARY_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Kernels {
    /// Enqueue the fill of `elements` elements of `dtype` at `ptr` with the canary operand of `seed` and `tag` ([`canary_bytes`]) on `stream`.
    ///
    /// # Safety
    ///
    /// `stream` must be a stream of the context the kernels were loaded in, and `ptr` the address of device memory of that context that holds at
    /// least `elements` elements of `dtype` and stays allocated until the work has completed on `stream`.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn fill(
        &self,
        stream: &CudaStream,
        ptr: u64,
        elements: u64,
        dtype: DType,
        seed: u64,
        tag: u8,
    ) -> Result<()> {
        let kind: u32 = match dtype {
            DType::F16 => 0,
            DType::Bf16 => 1,
            _ => 2,
        };
        let base = canary_base(seed, tag);
        let mut launch = stream.launch_builder(&self.fill);
        launch.arg(&ptr).arg(&elements).arg(&base).arg(&kind);
        // SAFETY: the kernel's parameters are (u64 address, u64 count, u64 base, u32 kind), as pushed; the memory is the caller's contract.
        unsafe { launch.launch(blocks(elements, CANARY_FILL_BLOCKS)) }
            .map_err(Error::driver("cuLaunchKernel(tilefish_canary_fill)"))?;
        Ok(())
    }

    /// Enqueue the digest of the `words` 32-bit words at `ptr` on `stream`: its two sums are added into the two `u64` at `sums` (zeroed first).
    ///
    /// # Safety
    ///
    /// `stream` must be a stream of the context the kernels were loaded in; `ptr` the address of at least `words` words and `sums` of two `u64` of
    /// device memory of that context, allocated until the work has completed on `stream`.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn digest(&self, stream: &CudaStream, ptr: u64, words: u64, sums: u64) -> Result<()> {
        let mut launch = stream.launch_builder(&self.digest);
        launch.arg(&ptr).arg(&words).arg(&sums);
        // SAFETY: the kernel's parameters are (u64 address, u64 count, u64 address), as pushed; the memory is the caller's contract.
        unsafe { launch.launch(blocks(words, CANARY_DIGEST_BLOCKS)) }
            .map_err(Error::driver("cuLaunchKernel(tilefish_canary_digest)"))?;
        Ok(())
    }
}
