# Masks, indices, gathers and copies

A lowered `@kernel` also compares, selects, counts positions, reduces to an index, and gathers and scatters rows at indices that are device
data; a graph materialises a view or the patches of a convolution. Each has an exact numerics contract (notes 7 to 11 of `tilefish.lower`) and
runs in `jit`, `replay` and `reference`, alone and in a plan.

```python
import tilefish
from tilefish import Dim, DevScalar, Tensor, bf16, f32, i32, i64

V, R, C = Dim("V"), Dim("R"), Dim("C", 3000)
D = Dim("D", 1024)

@tilefish.kernel
def embed(table: Tensor[V, D, bf16], ids: Tensor[R, i64], n: DevScalar[i32]) -> Tensor[R, D, bf16]:
    rows = tilefish.gather_rows(table, ids, oob="zero")               # an index outside [0, V) loads a row of +0.0
    keep = tilefish.iota(rows, axis=0) < tilefish.load(n)            # a mask: the rows below a device-scalar count
    return tilefish.where(keep, rows, (rows.to(f32) * 0.0).to(bf16))  # a select: the kept rows' bits as they are

@tilefish.kernel
def greedy(logits: Tensor[R, C, bf16]) -> Tensor[R, i64]:
    return tilefish.argmax(logits, dtype=i64)                        # torch's rule: the first NaN, else the first maximum
```

- `== != < <= > >=` give a mask (`tilefish.bool_`, a value inside the body: store it with `.to(u8)`); masks combine with `& | ^ ~`; a scalar is
  rounded to the operand's dtype before it is compared. `tilefish.where(m, a, b)` selects bits, NaN payloads and signed zeros included.
- `tilefish.iota(x, axis)` is the index along an axis (`[R, 1]` for the rows of `[R, D]`, `[D]` for its columns); `//` and `%` of integers by a
  positive constant have Python's semantics.
- `tilefish.argmax` / `argmin(x, dtype=i32 | i64)` along the last axis are exact and independent of the reduction tree (a combine of
  (value, index) pairs); a row of up to 16384 columns is walked in chunks of 1024.
- `tilefish.gather_rows(x, idx, oob="zero" | "clamp")` and `tilefish.scatter_rows(dst, idx, src, oob="skip")` copy rows (or elements) as raw
  bits, with the out-of-range policy declared at the call.
- `x.to(i32, rounding="rn" | "rz" | "rd" | "ru", saturate=True | False)` (also `i64` and `u8`): a float to integer cast says how it rounds
  (to nearest even, toward zero, down, up; there is no default) and what a value out of range becomes: clamped to the type's range with a NaN
  0 (`saturate=True`, the default, Rust's `as`), or the low bits of the `i64` (`saturate=False`: C's narrowing, torch's cast to `u8`). Each is inline PTX
  (`cvt.rni.s32.f32` and the like) and is byte-identical to the reference on every one of the 2**32 f32 bit patterns (note 18).
- In the row template, `x[k]` of a read-only parameter is read at its offset: `codes[k]` of a `[K, R]` codes plane is a per-row operand (the
  index of `gather_rows`), `tables[k]` of a `[K, N, D]` parameter a table, with the dimensions of the offset runtime scalars (note 19).
- `x.to(bf16, nan="preserve")` keeps a NaN's sign and payload (the default, `"canonical"`, gives `0x7FFF`); `tilefish.sin`, `tilefish.cos` and
  `tilefish.tanh` are libdevice's (`sinf`, `cosf`, `tanhf`), built with flush-to-zero like every libdevice function Triton links: over every f32,
  torch's CUDA bits but `sin` and `tanh` of a subnormal, which are a zero of its sign (notes 10 and 14).
- **IEEE math.** `@tilefish.kernel(math="ieee")`, or `math="ieee"` at one op (`tilefish.div(a, b, math="ieee")`, `tilefish.exp(x, math="ieee")`),
  makes the f32 math CUDA C++'s without fast math and without flush to zero, what torch's CUDA kernels are built with: `/` and `sqrt` are IEEE 754
  (`div.rn.f32`, `sqrt.rn.f32`; a constant divisor is divided by), `exp` is `expf`, `silu` torch's `x / (1 + expf(-x))`, `rsqrt` `rsqrtf`, and `sin`
  and `tanh` keep a subnormal, each written out as the inline PTX nvcc emits (Triton links libdevice with flush to zero). On every one of the 2**32 f32 bit
  patterns these are torch's CUDA bits, and `/` is torch's division of tensors; with `fp_fusion=False` too, a body of these ops is torch's eager
  composition of the same ops byte for byte. The default, `math="fast"`, is the contract of before (note 13 of `tilefish.lower`).
- **Primitives for reproducing a CUDA kernel.** `tilefish.erf(x, math="ieee")` is CUDA's `erff` without flush to zero (the default math's is
  libdevice's, `gelu`'s); `tilefish.sin_approx(x)` is `__sinf` (`sin.approx.f32`), `tilefish.rcp(x)` `__frcp_rn` (`rcp.rn.f32`),
  `tilefish.fma(a, b, c)` `fmaf` (one rounding, whatever `fp_fusion` says), and `-x` (`tilefish.neg`) is CUDA's negation, `-0.0` for `+0.0`
  where Triton's `-x` is `0 - x` and the canonical NaN for every NaN (written out, so that a negation ptxas folds into a select keeps it). Each
  is the bits of the CUDA C++ function on every f32, measured against NVRTC on the GPU (notes 15 and 16).
- **The CUDA math guard.** `expf`, `erff`, `tanhf` and the other functions those ops copy are not the same in every CUDA release (CUDA 13.2
  changed `erff`). The manifest entry of a generated kernel records the functions it uses, the releases tilefish measured each on, and the
  libdevice it linked (`cuda_math`); a run warns (`TilefishCudaMathWarning`) where torch is built with another release or a replay's process would
  link another libdevice, and the Rust host refuses a release with `Manifest::check_cuda_math`. `docs/numerics.md` has the table and the procedure
  to run on a CUDA or torch change (`pytest -m cuda_math`).
- **Ordered sums.** `tilefish.sum(x, axis=-1, order=...)` adds in the order a CUDA kernel does: `order="sequential"` is one thread's loop from
  `+0.0`, `tilefish.tree(256)` is 256 threads over strided columns and a shared-memory halving tree, and
  `tilefish.tree(128, offsets=(16, 8, 4, 2, 1, 64, 32), vector=8)` is 128 threads of 8 consecutive columns, a warp-shuffle tree and then the
  warps; `fma=True` makes each step of a chain `acc = fma(a, b, acc)` for `tilefish.sum(a * b, ...)`, as nvcc compiles `acc += a * b`. The bytes
  are those kernels' and the same at every `num_warps` (note 17):

  ```python
  @tilefish.kernel(math="ieee")
  def rmsnorm(x: Tensor[R, 1024, bf16], w: Tensor[1024, bf16]) -> Tensor[R, 1024, bf16]:
      v = x.to(f32)
      ss = tilefish.sum(v * v, -1, keepdims=True, order=tilefish.tree(128, offsets=(16, 8, 4, 2, 1, 64, 32), vector=8), fma=True)
      return ((v * tilefish.rsqrt(ss / 1024.0 + 1e-6)) * w.to(f32)).to(bf16)   # a 128-thread CUDA rmsnorm, byte for byte
  ```
- The elementwise template broadcasts an operand over the leading axes (a `[T, D]` table added to a `[K, T, D]` batch), and along any axis by
  numpy's rules (a `[H, 1, L]` mask of the keys over `[H, L, L]` scores, a `[K, T, 1]` scale per row): such an operand is read at the offset of the
  lane's coordinates, and `tilefish.iota(x, axis)` is the lane's coordinate along any axis (a causal mask is `iota(s, 2) <= iota(s, 1)`). The sizes the
  coordinates need are runtime scalars of the dimensions, so one binary serves every size (note 12).
- In a graph, `tilefish.contiguous(view)` and `tilefish.im2col(x, kernel, stride, padding)` are ops whose Triton tilefish generates, reading
  a strided view: a materialised permutation, and convolution patches with channel-major columns and `+0.0` padding.
  `tilefish.im2col1d(x, kernel, stride=, dilation=, padding=, order="channel" | "tap")` gives the patches of a 1-D convolution of `[N, C, L]` as
  `[N, Lo, C * K]`, with a dilation, a padding `(left, right)` (a causal one is `(dilation * (K - 1), 0)`) and the columns channel-major or tap-major.
- `tilefish.col2im1d(cols, kernel, stride=, dilation=, padding=, output_padding=, order="channel" | "tap", bias=, layout="ncl" | "nlc")` is the
  overlap-add of a 1-D transposed convolution: `cols: [N, Li, C * K]` (any strides: a GEMM's `x^T @ W.reshape(C_in, C * K)` reshaped, as it is)
  becomes `[N, C, Lo]` or a channel-last `[N, Lo, C]`, `Lo = (Li - 1) * stride - left - right + dilation * (K - 1) + output_padding + 1`
  (torch's `conv_transpose1d(padding=p, output_padding=op)` is `padding=(p, p), output_padding=op`). Unlike the copies it adds, so the order of
  the adds is the contract, and it is the order of ATen's CUDA `col2im` (`F.fold` on CUDA): an f32 sum from `+0.0` over the input positions in
  ascending order (the taps descending), one rounding per add and nothing contracted (`enable_fp_fusion=False`), then the bias in f32, then one
  rounding to the dtype of `cols`. Three terms or more added in another order round differently, so this order is what makes the bytes
  `F.fold`'s: measured byte for byte against `F.fold` on CUDA in f32, bf16 and f16, and against `conv_transpose1d` on small integers, where every
  order is exact.
