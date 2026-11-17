# 掩码、索引、gather 和拷贝

[English](../../en/guide/lowering.md) | **简体中文**

下沉（lowering）生成的 `@kernel` 还能做比较、选择、计数位置、归约到索引，以及在由设备数据给出的索引处 gather 和 scatter 行；graph（计算图）可以物化一个视图或卷积的 patch。每一种都有精确（exact）的数值语义契约（`tilefish.lower` 的第 7 到 11 条说明），并且都能在 `jit`、`replay` 和 `reference` 中运行，既可以单独运行，也可以在 plan（计划）中运行。

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

- `== != < <= > >=` 得到掩码（`tilefish.bool_`，函数体内部的值：用 `.to(u8)` 存储）；掩码用 `& | ^ ~` 组合；标量在比较之前先舍入到操作数的 dtype。`tilefish.where(m, a, b)` 按位选择，NaN 载荷和带符号零也包括在内。
- `tilefish.iota(x, axis)` 是沿某个轴的索引（对 `[R, D]` 的行是 `[R, 1]`，对它的列是 `[D]`）；整数除以正常数的 `//` 和 `%` 具有 Python 的语义。
- 沿最后一个轴的 `tilefish.argmax` / `argmin(x, dtype=i32 | i64)` 是精确的，且与归约树无关（对 (值, 索引) 对做合并）；最多 16384 列的一行按每块 1024 列遍历。
- `tilefish.gather_rows(x, idx, oob="zero" | "clamp")` 和 `tilefish.scatter_rows(dst, idx, src, oob="skip")` 按原始位拷贝行（或元素），越界策略在调用处声明。
- `x.to(i32, rounding="rn" | "rz" | "rd" | "ru", saturate=True | False)`（`i64` 和 `u8` 也一样）：浮点到整数的转换要说明如何舍入（就近取偶、向零、向下、向上；没有默认值），以及超出范围的值变成什么：钳制到该类型的范围、NaN 变为 0（`saturate=True`，默认值，即 Rust 的 `as`），或者取 `i64` 的低位（`saturate=False`：C 的收窄转换，torch 转换到 `u8` 的方式）。每一种都是内联 PTX（`cvt.rni.s32.f32` 之类），并在全部 2**32 个 f32 位模式上都与参考实现逐字节一致（第 18 条说明）。
- 在行模板中，只读参数的 `x[k]` 在其偏移处读取：`[K, R]` codes 平面的 `codes[k]` 是一个逐行操作数（`gather_rows` 的索引），`[K, N, D]` 参数的 `tables[k]` 是一张表，偏移涉及的维度是运行时标量（第 19 条说明）。
- `x.to(bf16, nan="preserve")` 保留 NaN 的符号和载荷（默认值 `"canonical"` 给出 `0x7FFF`）；`tilefish.sin`、`tilefish.cos` 和 `tilefish.tanh` 是 libdevice 的实现（`sinf`、`cosf`、`tanhf`），和 Triton 链接的所有 libdevice 函数一样以 flush-to-zero 构建：在所有 f32 上，它们与 torch 的 CUDA 位相同，但次正规数的 `sin` 和 `tanh` 除外，它们是与其同号的零（第 10 和 14 条说明）。
- **IEEE 数学。** `@tilefish.kernel(math="ieee")`，或在单个算子上用 `math="ieee"`（`tilefish.div(a, b, math="ieee")`、`tilefish.exp(x, math="ieee")`），让 f32 数学与不开 fast math、不开 flush to zero 的 CUDA C++ 一致，torch 的 CUDA kernel（核函数）正是这样构建的：`/` 和 `sqrt` 符合 IEEE 754（`div.rn.f32`、`sqrt.rn.f32`；除数为常数时也做除法），`exp` 是 `expf`，`silu` 是 torch 的 `x / (1 + expf(-x))`，`rsqrt` 是 `rsqrtf`，`sin` 和 `tanh` 保留次正规数，每一个都写成 nvcc 生成的那种内联 PTX（Triton 以 flush to zero 链接 libdevice）。在全部 2**32 个 f32 位模式上，这些都是 torch 的 CUDA 位，`/` 就是 torch 的张量除法；再加上 `fp_fusion=False`，由这些算子组成的函数体与 torch 对相同算子的 eager 组合逐字节一致。默认值 `math="fast"` 就是以前的契约（`tilefish.lower` 的第 13 条说明）。
- **复现 CUDA kernel 的原语。** `tilefish.erf(x, math="ieee")` 是不开 flush to zero 的 CUDA `erff`（默认数学下用的是 libdevice 的，即 `gelu` 所用的）；`tilefish.sin_approx(x)` 是 `__sinf`（`sin.approx.f32`），`tilefish.rcp(x)` 是 `__frcp_rn`（`rcp.rn.f32`），`tilefish.fma(a, b, c)` 是 `fmaf`（一次舍入，无论 `fp_fusion` 怎么设），`-x`（`tilefish.neg`）是 CUDA 的取负：对 `+0.0` 得到 `-0.0`（而 Triton 的 `-x` 是 `0 - x`），对每个 NaN 得到规范 NaN（显式写出，这样即使 ptxas 把取负折叠进 select 也能保留它）。每一个都在所有 f32 上与对应 CUDA C++ 函数的位相同，已在 GPU 上对照 NVRTC 测量（第 15 和 16 条说明）。
- **CUDA 数学守卫。** `expf`、`erff`、`tanhf` 以及这些算子复刻的其他函数在不同的 CUDA 版本中并不相同（CUDA 13.2 改了 `erff`）。生成的 kernel 的 manifest（清单）条目记录它用到的函数、tilefish 测量每个函数时所用的版本，以及它链接的 libdevice（`cuda_math`）；当 torch 用另一个版本构建，或者 replay（回放）的进程会链接另一个 libdevice 时，运行会发出警告（`TilefishCudaMathWarning`），Rust host（宿主）则用 `Manifest::check_cuda_math` 拒绝这样的版本。`docs/zh-CN/numerics.md` 给出了表格，以及在 CUDA 或 torch 变更时要执行的步骤（`pytest -m cuda_math`）。
- **有序求和。** `tilefish.sum(x, axis=-1, order=...)` 按 CUDA kernel 的顺序做加法：`order="sequential"` 是一个线程从 `+0.0` 开始的循环，`tilefish.tree(256)` 是 256 个线程按步长遍历各列，再加一棵共享内存对半归约树，`tilefish.tree(128, offsets=(16, 8, 4, 2, 1, 64, 32), vector=8)` 是 128 个线程各处理 8 个连续列，先做一棵 warp（线程束）shuffle 树，再在各 warp 之间归约；对 `tilefish.sum(a * b, ...)`，`fma=True` 让链上的每一步都是 `acc = fma(a, b, acc)`，就像 nvcc 编译 `acc += a * b` 那样。得到的字节就是那些 kernel 的字节，并且在每个 `num_warps` 下都相同（第 17 条说明）：

  ```python
  @tilefish.kernel(math="ieee")
  def rmsnorm(x: Tensor[R, 1024, bf16], w: Tensor[1024, bf16]) -> Tensor[R, 1024, bf16]:
      v = x.to(f32)
      ss = tilefish.sum(v * v, -1, keepdims=True, order=tilefish.tree(128, offsets=(16, 8, 4, 2, 1, 64, 32), vector=8), fma=True)
      return ((v * tilefish.rsqrt(ss / 1024.0 + 1e-6)) * w.to(f32)).to(bf16)   # a 128-thread CUDA rmsnorm, byte for byte
  ```
- 逐元素模板把操作数沿前导轴广播（把一张 `[T, D]` 表加到一个 `[K, T, D]` 批次上），也能按 numpy 的规则沿任意轴广播（在 `[H, L, L]` 的分数上应用 `[H, 1, L]` 的 key 掩码，每行一个的 `[K, T, 1]` 缩放）：这样的操作数在 lane（通道）坐标对应的偏移处读取，`tilefish.iota(x, axis)` 是 lane 沿任意轴的坐标（因果掩码是 `iota(s, 2) <= iota(s, 1)`）。坐标所需的尺寸是各维度的运行时标量，因此一个二进制文件适用于所有尺寸（第 12 条说明）。
- 在 graph 中，`tilefish.contiguous(view)` 和 `tilefish.im2col(x, kernel, stride, padding)` 是由 tilefish 生成 Triton 代码、读取带步长的视图的算子：前者是物化的置换，后者是列按通道主序排列、用 `+0.0` 填充的卷积 patch。`tilefish.im2col1d(x, kernel, stride=, dilation=, padding=, order="channel" | "tap")` 把 `[N, C, L]` 的 1-D 卷积的 patch 给成 `[N, Lo, C * K]`，支持膨胀、填充 `(left, right)`（因果填充是 `(dilation * (K - 1), 0)`），列可以按通道主序或按 tap 主序排列。
- `tilefish.col2im1d(cols, kernel, stride=, dilation=, padding=, output_padding=, order="channel" | "tap", bias=, layout="ncl" | "nlc")` 是 1-D 转置卷积的 overlap-add：`cols: [N, Li, C * K]`（任意步长：一个 GEMM（通用矩阵乘）的 `x^T @ W.reshape(C_in, C * K)` 结果，reshape 后原样使用）变成 `[N, C, Lo]` 或通道在后的 `[N, Lo, C]`，`Lo = (Li - 1) * stride - left - right + dilation * (K - 1) + output_padding + 1`（torch 的 `conv_transpose1d(padding=p, output_padding=op)` 对应 `padding=(p, p), output_padding=op`）。与那些拷贝不同，它要做加法，所以加法的顺序就是契约，这个顺序就是 ATen 的 CUDA `col2im`（CUDA 上的 `F.fold`）的顺序：从 `+0.0` 开始按输入位置升序（tap 降序）做 f32 求和，每次加法舍入一次，不收缩（contraction）任何运算（`enable_fp_fusion=False`），然后在 f32 中加 bias，最后舍入一次到 `cols` 的 dtype。三个或更多项按其他顺序相加，舍入结果不同，所以正是这个顺序让字节与 `F.fold` 的一致：已在 CUDA 上以 f32、bf16 和 f16 对照 `F.fold` 逐字节测量，并在小整数上对照 `conv_transpose1d` 测量，小整数上任何顺序都是精确的。
