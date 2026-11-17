# 数值语义：CUDA math 防护、CUDA 或 torch 变更时的流程，以及设计决定

[English](../numerics.md) | **简体中文**

下沉（lowering）生成的 kernel（核函数）的数值语义契约，就是 `tilefish.lower` 的 docstring（其中的表格和注释 1 到 20）：每个算子、它变成的 PTX，以及它的比特是什么，都经过实测。本文档讨论这个契约在 tilefish 之外依赖什么、这些依赖变化时如何重新检查，以及契约所依据的设计决定。

## 1. 下沉生成的 kernel 的比特取决于什么

下沉生成的 kernel 中，大多数算子是一条 PTX 指令，或 tilefish 写出的一段固定序列（`div.rn.f32`、`fma.rn.f32`、`sin.approx.f32`、有序求和的 `add.rn` 链……）。它们的比特取决于 PTX（它就在生成源码里）和硬件；tilefish 之外的任何东西都不会改变它们。有三类算子不同：

* **CUDA math 库函数的副本。** 在 IEEE math 中（`tilefish.lower` 的注释 13），`exp` 和 `silu` 构建在 CUDA 的 `expf` 之上，`rsqrt` 是 `rsqrtf`，`erf` 是 `erff`（注释 15）：即 NVRTC 为它们各自生成的内联 PTX，复制一次。二进制文件的比特不会变，但关于它们的声明“与 CUDA C++ 的 `expf` 的比特相同”（以及与用它构建的 torch 的 `exp` 相同），只对函数正是这段序列的 CUDA 版本成立。
* **libdevice 函数。** `sin`、`cos` 和 `tanh`（两种 math 下都是）、`gelu` 以及默认 math 下的 `erf` 调用 libdevice，Triton 从一个文件链接它（Triton 自带的 `libdevice.10.bc`，或 `TRITON_LIBDEVICE_PATH` 指定的文件）。换一个文件就会得到不同的比特，而关于它们的声明（注释 10：“除 subnormal 外与 torch 的比特相同”）只对 `sinf`、`cosf`、`tanhf` 与 libdevice 相同的版本成立。

* **PTX 交给 ptxas 决定的运算。** 普通 PTX 的算子（Triton 的 `a * b`、`a + b`）按 ptxas 的决定汇编：在 `enable_fp_fusion=True` 下，它可以把一次乘法和一次加法收缩（contraction）成一次舍入，这是契约允许的（注释 2）。在 `enable_fp_fusion=False` 下它不得这样做，但在 LLVM 把两个 lane（通道）打包成 `mul.rn.f32x2` / `add.rn.f32x2` 的地方，ptxas 12.9 还是会这样做（sm_100 及更新的架构：sm_100 和 sm_103 上是 `FFMA2`，sm_120 上是 `FFMA`）。所以 `fp_fusion=False` 的 kernel 把每个浮点 `+ - *` 写成一条标量 `.rn` 内联 PTX 指令（注释 2），并且收缩检查会读取每个用 `enable_fp_fusion=False` 构建的二进制文件（第 4 节）。

这些函数确实会在版本之间变化。在一块 B300（sm_103）上对每个 f32 实测，用各版本的 NVRTC 按 nvcc 的默认设置编译：

| 函数 | 与 tilefish 算子比特相同的 CUDA 版本 | |
|---|---|---|
| `expf`、`rsqrtf` | 12.8、12.9、13.0、13.1、13.2 | 在 13.5 中 PTX 文本相同 |
| `sinf`、`cosf`、`tanhf` | 12.8、12.9、13.0、13.1、13.2 | `tanhf` 的 PTX 在 13.0 中变了（`copysign`），比特没变；13.5 的 `sinf` 和 `cosf` 展开方式不同（未测：测试机器的驱动是 13.2） |
| `erff` | 12.8、12.9、13.0、13.1 | 13.2 改了它（系数和阈值不同）：在 353 987 744 个输入上不同，相差一个 ULP |

这张表就是 `tilefish._cudamath.VERIFIED`。libdevice 自己的 `erff`（`gelu` 用的那个）与这些都不同：它在 24 071 628 个输入上与 CUDA 的不同，它的契约是一个容差，而不是某个版本。

## 2. 防护

* **记录。** 含有这类算子的生成 kernel，其 manifest（清单）条目带有 `cuda_math`（manifest schema（模式）3 第 4.7 节）：这些函数、表中这些函数全部经过实测的版本，以及链接所用 libdevice 的 sha256。它包含在 manifest digest（摘要）中。构建从生成源码的代码中读出这些函数（`tilefish.lower` 为它们写出的辅助函数和调用）；源码本身不变，所以没有任何源码 digest 改变。
* **在 Python 中检查。** 这类 kernel 的 `jit`、`replay` 或 `reference` 运行（以及含有这类 kernel 的 plan（计划）的 replay（回放））会把构建 torch 所用的版本（`torch.version.cuda`）与条目中的版本比较；replay 还会把本进程中的构建将要链接的 libdevice 与条目中的比较。不匹配时发出 `tilefish.TilefishCudaMathWarning`：二进制文件照导出时的样子运行，警告说明 torch 自己的算子以及使用它们的 reference 模式在这里可能与它的比特不同。要改为拒绝：`warnings.simplefilter("error", tilefish.TilefishCudaMathWarning)`（或 `pytest -W error::tilefish.TilefishCudaMathWarning`）。
* **在 Rust 中检查。** crate 读取 `Entry::cuda_math`，但无法得知 host（宿主）其他 CUDA 代码所用的版本，所以由 host 说明：`manifest.check_cuda_math("13.0")?` 会拒绝其生成 kernel 的函数未在该版本上实测过的 manifest，并逐一指出这些函数。

它不覆盖的：调用 libdevice 的手写 `extern`（它的身份是它的 IR；它的数学由其作者负责），以及没人测过的版本（它只是不在表中，防护会把它报告为不匹配）。

## 3. CUDA 或 torch 变更时的流程

每当 torch 升级到另一个 CUDA 构建、与 tilefish kernel 对比的代码改用新的 CUDA 版本、Triton（以及随之而来的 libdevice）发生变化，或者换到新的 GPU 架构时，都要在 GPU 上运行这个流程。

1. 获取每个要检查的版本的 NVRTC：该版本的 `nvidia-cuda-nvrtc` wheel（`pip download --no-deps nvidia-cuda-nvrtc==13.2.86`，CUDA 12 则用 `nvidia-cuda-nvrtc-cu12==12.9.86`），解压到任意位置：`libnvrtc.so.13` 在 `nvidia/cu13/lib` 下（12 则在 `nvidia/cuda_nvrtc/lib` 下）。它的 `libnvrtc-builtins` 必须在它旁边。驱动必须是该版本或更新的版本（由它编译 PTX）。
2. 运行穷举比较：

   ```sh
   cd python
   TILEFISH_NVRTC_PATHS=/path/to/13.2/libnvrtc.so.13:/path/to/12.9/libnvrtc.so.12 pytest -m cuda_math -s
   ```

   `-m cuda_math` 选中在每个 f32 上与 CUDA math 库的比较：`test_gpu_cuda_math.py`（这张表，针对 torch 的 NVRTC 和每个指定的 NVRTC）、`test_gpu_ieee.py`、`test_gpu_tanh.py`、`test_gpu_trig.py`（与 torch 的算子对比）和 `test_gpu_primitives.py`（与 NVRTC 和 PTX 对比）。在一个 B300 MIG 切片上约需半分钟，每多一个版本再加几秒。
3. 阅读结果。表格过时时，`test_the_table_is_what_each_release_computes` 会失败，并说明是怎样过时的：一个已列出的版本结果不同（把它从 `VERIFIED` 中去掉：之后使用该函数的 kernel 在该版本上会发出警告），或一个未列出的版本在每个 f32 上都与该算子的比特相同（把它加进去）。当 torch 自己的算子不再与这些二进制文件一致时（torch 用一个函数发生了变化的版本构建），其他文件会失败：此时用户看到的就是第 2 节的警告，而契约中列出实测版本的注释需要加上新的版本。
4. 更新 `tilefish._cudamath.VERIFIED` 和第 1 节的表格，重新运行第 2 步直到通过，然后重新导出那些 `cuda_math` 应当列出新版本的 manifest（旧的 manifest 保留导出时的列表）。

## 4. 设计决定

* **`num_warps` 是每个 kernel 身份的一部分**（声明 digest、编译规格、manifest 条目及其 digest；replay 会拒绝不同的值）。Triton 自己的归约依赖于它：不带顺序的 `sum`、`mean` 和 `max` 按由 warp（线程束）数决定形状的树来归约（在 1024 行各种量级的值上，1 个 warp 与 2 到 16 个 warp 之间有 669 到 878 行不同，列数为 96、1000 和 4096）。逐元素算子、类型转换、`argmax` / `argmin`（全序）和有序求和（注释 17）不依赖于它：它们的字节在任何 warp 数下都相同（`tests/test_gpu_reduction_order.py`）。一条对所有 kernel 一致的规则可以被机械地检查，而针对每个 kernel 的论证则不行；每个归约都带有 `order` 的 kernel 可以为了速度改变 warp 数而不改变字节。契约中 `tilefish.lower` 的 “num_warps and the numerics identity” 一段说的是同一件事。
* **归约顺序是调用的一部分，而不是模板的一部分。** `tilefish.sum(x, -1, order=...)` 指定一个 CUDA kernel 做加法的顺序（一条顺序链，或多个 lane 的链再加上跨 lane 的固定树；`fma=True` 表示 `acc = fma(a, b, acc)` 的链），下沉时使用数据搬运和 `add.rn`，使 Triton 的布局和 `num_warps` 都不会介入。没有 `order` 时，求和保留 Triton 的树，速度更快。
* **原语照搬 CUDA 的写法**，而不是某种理想化的写法：`-x` 是 CUDA 的取负（`neg.f32` 给出规范 NaN，而翻转符号位会保留载荷），`sin_approx` 是硬件的 `sin.approx.f32`，包括它对 subnormal 的 flush，IEEE math 中的 `erf` 是 CUDA 12.8 到 13.1 的 `erff`。用 nvcc 默认设置编译的 CUDA C++ kernel 能算出什么，DSL 就能表达什么。
* **“不做收缩”既被显式写出，也在二进制文件上检查。** `@tilefish.kernel(fp_fusion=False)` 承诺每个浮点 `+`、`-` 和 `*` 都单独舍入，用 `enable_fp_fusion=False` 声明的 `extern` 也要求同样的事。编译选项（`--fmad=false`）在 sm_100 及更新的架构上守不住这个承诺（见上文），所以它由两道措施来保证：
  * **由下沉保证**：`fp_fusion=False` 的 kernel（以及每个融合区域）把每个浮点 `+ - *` 写成一条带 `.rn` 的标量内联 PTX 指令，没有什么会打包它，ptxas 也不会收缩它，并把每个窄化类型转换写成内联 `cvt.rn`（`tilefish.lower` 的注释 1 和 2）；
  * **由收缩检查保证**（`tilefish._fpcontract`，每次构建都运行）：用 `enable_fp_fusion=False` 构建的二进制文件（下沉生成的 kernel、融合区域、生成算子，或手写的 Triton 或 Gluon kernel）会被反汇编；同时，把它的 PTX 中每个浮点乘法替换为对其操作数做按位 `or`（不做任何乘法），再用同样的标志重新汇编，也反汇编。如果二进制文件的融合乘加 lane（`FFMA`、`FFMA2`、`HFMA2`、`DFMA`，不包括那些不做乘法或不做加法的惯用形式）比后者多，构建就会被拒绝，并给出 kernel 的名称、相关指令和修复方法（在舍入要紧的地方用标量 `.rn` 内联 PTX，或者用 `enable_fp_fusion=True`）。这种比较涵盖了 ptxas 自行融合的部分（显式的 `fma`，IEEE 除法、平方根或倒数的展开），而无需逐一列出。反汇编器是第一个能解码该架构的 `nvdisasm`（`$TILEFISH_NVDISASM_PATH`、`nvidia-cuda-nvdisasm` wheel、Triton 的、它的 `ptxas` 旁边的那个、`PATH`，然后是 `$CUDA_HOME` 或 `$CUDA_PATH`）；Triton 3.5.1 自带 CUDA 12.8 的版本，它无法解码 sm_103，此时 PTX 会为其架构族的基础架构 `sm_100a` 汇编后再读取（`pip install nvidia-cuda-nvdisasm` 可以直接读取 sm_103）。没有浮点乘法或没有浮点加法的二进制文件不做反汇编。`tests/test_gpu_fpcontract.py` 构建会被 ptxas 收缩的一个 Triton kernel 和一个 Gluon kernel（在 sm_100、sm_103 和 sm_120 上被拒绝，在 sm_90 上构建成功）、改用标量 `.rn` 指令的同一个 kernel（在所有架构上都能构建），以及 `fp_fusion=False` 下下沉的每个用例，在全部四个架构上运行。
  取负的 NaN 也用同样的方式显式写出（注释 16）：`neg.f32` 给出规范 NaN，但 ptxas 会把它作为取负的操作数折叠进 select，这会翻转 NaN 的符号。
* **默认 GEMM（通用矩阵乘）姿态（posture）保持它的规则：带 bias 的 GEMM 不使用 split-K（K 维切分）**，除非其声明明确选择加入（`allow_bf16_reduction`，或其操作数类型的选择加入）。正是这条规则，让默认姿态无需对每个算法做测量，就能对它做出的每个 pin（固定）、在每种设备和每个 cuBLASLt 版本上承诺 `A @ B + bias` 只有一次舍入。实测表明，带 bias epilogue（尾部）的 split-K 会把 bias 加到一个已经舍入到输出类型的和上，即使部分和是 f32（cuBLASLt 12.8.4，RTX 5090，算法 21，32 x 1024 x 3072：32768 个元素中有 8409 个超出了一次舍入加 f32 累加所允许的范围）。另一个算法保持了一次舍入（算法 66，split-K 为 3，即 sm_103 上 cuBLASLt 13.1.0 中 `addmm` 对 32 和 64 x 1024 x 3072 运行的算法），但启发式给出的答案中没有任何信息能区分这两者，而且双重舍入是悄无声息的。因此在 13.1.0 下，默认姿态在 `test_gpu_gemm.py` 的 15 个带 bias 的形状中有 2 个得不到 `addmm` 的字节（在 12.8.4 下 15 个全部如此）；想要 `torch.addmm` 字节的调用方可以声明 `heuristic="torch"`，它在实测的每个形状上都与之一致（12.8.4 和 13.1.0）。按算法放宽这条规则，需要在每种设备和每个 cuBLASLt 版本上对每个带 bias epilogue 的 split-K 算法做穷举检查；目前没有这个计划。这条规则推广到 `cublaslt.matmul` 除 `none` 之外的每个 epilogue（`docs/manifest-schema-3.md` 第 9.5 节）：作用在已经舍入过的和上的 ReLU，也会以同样的方式舍入两次。在那里，如果声明选择加入，对每个 pin 的验证（累加器上的公式，逐比特）会说明一个 split-K pin 是否保持了一次舍入：不保持时，这个 pin 会被记录为 binary-defined。

## 5. 设备全局状态：已设计，未实现

一个由 kernel 在设备上递增、由 host 在运行后读取的计数器（饱和存储截断了多少个值、丢弃了多少个 token），今天还写不出来：下沉生成的 kernel 没有原子操作，plan 也没有一种比自身写入活得更久、并具有明确重置语义的缓冲区。

在三条限制下，它与这个模型相容，这些限制保证一次运行的字节仍然是其输入的函数：

* **声明的状态缓冲区。** graph（计算图）把它声明为一种新角色的参数 `State[DevScalar[i64]]`（一个整数标量，或一个小的整数张量），带 `reset="run"`：plan 在每次运行开始时把它清零，作为第一个步骤，并把它返回（host 在运行后读取它，或把它绑定到自己的缓冲区）。它不是调用方提供的输出：它在运行开始时的值是契约的一部分（零），而不是调用方内存里碰巧有的值。
* **只允许与顺序无关的更新。** kernel 用满足交换律和结合律的整数原子操作更新它（`tilefish.atomic_add`，以及 `max`、`min`、`or`），作用于回绕整数：无论各 program 以什么顺序到达，最终值都相同，所以 `jit`、`replay`、CUDA graph（CUDA 图）和 `reference`（它以任意顺序累加各 program 的贡献）逐字节一致。浮点原子操作会被拒绝：它的结果取决于顺序。在下沉生成的 kernel 中，一个 program 只把自己的贡献加一次（对 block 做整数 `tl.sum`，是精确（exact）的），所以一个计数器的开销是每个 program 一次原子操作。
* **每个缓冲区同一时间只有一次运行。** 状态位于 plan 的 arena（内存池）中，而同一个 plan 的各次运行本来就在 arena 上串行（并发的 stream 本来就各需一个 arena），所以两次运行永远不会更新同一个计数器。

它应当等待它仍然需要的、由其他工作负责的那部分：DSL 中的原子算子及其在两种模板中的下沉（重置是一个 memset 步骤，plan schema 1 自 tilefish 0.0.7 起就有，见第 7.6 节）。有了它们，状态缓冲区就是 plan（`role: "state"` 及其重置）和 manifest ABI（一个 kernel 可以原子更新的 `inout` 设备标量）中的一个增量字段，由两种 host 读取；在此之前，计数的方法是写一个手写 `extern`，原子地更新一个由调用方清零的 `InOut` 张量。
