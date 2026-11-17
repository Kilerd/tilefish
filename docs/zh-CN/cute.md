# 作为 extern 的 CuTe DSL kernel

[English](../en/cute.md) | **简体中文**

CUTLASS 的 CuTe DSL（`nvidia-cutlass-dsl` 包，`import cutlass.cute as cute`）用 Python 编写带异步 MMA 流水线（TMA、`tcgen05`、warp（线程束）特化、cluster（线程块簇））的峰值性能 kernel（核函数）。tilefish 像对待任何其他 extern（外部 kernel）一样接纳这样的 kernel：按架构把它编译成 cubin，pin（固定）它，并从 Python、Rust 和 C ABI 启动它，可以是单次启动、启动 plan（计划）或 CUDA graph（CUDA 图），使用编译后 kernel 自带的启动属性（它的 grid、block、cluster、动态共享内存和 TMA 描述符）。manifest（清单）条目的规范见 `docs/zh-CN/manifest-schema-3.md` 第 4.9 节；代码在 `python/src/tilefish/_cute.py`。

目录：1 声明；2 如何构建；3 host（宿主）启动什么；4 身份、pin 与可复现性；5 限制；6 许可证；7 如何测试。

## 1. 声明

CuTe DSL kernel 是一个启动 `@cute.kernel` 的 `@cute.jit` host 函数。声明给出这个 host 函数（或一个 `__call__` 是这样的函数的对象：CUTLASS 示例中某个 kernel 类的实例），以及一个列出算子**张量**的桩函数（stub）：

```python
import cutlass
import tilefish
from tilefish import Dim, Out, Tensor, f16
from dense_gemm import DenseGemmKernel  # CUTLASS examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py

M = Dim("M", buckets=[256, 1024, 4096])

@tilefish.extern(
    cute=DenseGemmKernel(cutlass.Float32, True, (256, 128), (2, 1), True),   # f32 accumulator, 2-CTA MMA 256x128, a 2x1 cluster, a TMA store
    cute_args=lambda a, b, c: (a.unsqueeze(-1), b.unsqueeze(-1), c.unsqueeze(-1)),  # its (M, K, L) tensors, L = 1
)
def gemm(a: Tensor[M, 1024, f16], b: Tensor[2048, 1024, f16], c: Out[Tensor[M, 2048, f16]]): ...

c = tilefish.run(gemm, a, b)  # a @ b.T, compiled by the DSL and launched by tilefish
```

* **桩函数**只列出算子的张量（`Tensor`、`Out`、`InOut`；稠密），顺序与 kernel 接受它们的顺序相同，**形状必须是静态的**：每个轴都是一个数字或由分桶维度构成的表达式（`Dim("M", buckets=[...])`）。DSL 针对给定的布局编译 kernel，所以 tilefish 为每个 bucket（分桶）编译一个二进制文件（和其他地方一样，bucket 语义是精确匹配（exact）的）。`Strided`、`TensorDesc`、`Null`、标量或 `const` 会被拒绝：kernel 的参数由编译后的 kernel 决定。
* **`cute_args=`** 用这些张量构造 host 函数的位置参数：传入的是按桩函数顺序排列、形状为该 bucket 的形状、16 字节对齐的 torch 张量。它按 host 函数接受的形式返回参数；其中任何位置上的 torch 张量（包括输入列表中的）都会变成具有其静态布局的 CuTe 张量（`from_dlpack(t, assumed_align=16)`），其他东西原样传递（一个 `Constexpr`，例如 `operator.add`，或一个配置值）。视图必须从其张量的地址开始（增加一个轴、做一次置换）。tilefish 会在末尾追加 stream：host 函数把它作为最后一个参数。没有 `cute_args` 时，张量原样传递。
* **`cute_options=`** 是 DSL 的编译选项（`"--opt-level 3"`）。tilefish 自己传入 `--keep-cubin --keep-ptx`，针对设备的架构编译，不带行号信息；改变这些设置的选项会被拒绝。
* `sass=`、`index_bits=`、`reference=` 和 `name=` 与任何 extern 相同。grid、block、cluster、共享内存以及 Triton 的编译选项都由编译后的 kernel 决定，所以它们不是参数。
* **每个架构一个实现。** sm_100 和 sm_103 的 tcgen05 GEMM（通用矩阵乘）不能在 sm_120 上运行：`tilefish.impl(cute=..., cute_args=..., cute_options=...)` 是用 `by_arch=` 声明的算子的一个实现，可以与 Triton kernel 和预构建二进制文件并列：

  ```python
  tcgen05 = tilefish.impl(cute=DenseGemmKernel(...), cute_args=batched, name="tcgen05")
  @tilefish.extern(by_arch={"sm_100": tcgen05, "sm_103": tcgen05, "sm_120": tilefish.impl(cute=Sm120GemmKernel(...), cute_args=sm120_args)})
  def gemm(a: Tensor[M, 1024, f16], b: Tensor[2048, 1024, f16], c: Out[Tensor[M, 2048, f16]]): ...
  ```

声明会读取 DSL 的版本（`importlib.metadata`）和 host 函数所在的文件；它既不导入 DSL，也不导入 torch。DSL 必须安装为 tilefish 测试过的版本：`pip install nvidia-cutlass-dsl==4.8.1`（Linux 或 Windows，CUDA 12.9 或 13 的驱动）。在每个做导出或 replay（回放）的环境的 lock 文件中固定它的版本（第 4 节）：DSL 还处于 beta 阶段，换一个版本就是另一个声明。

## 2. 如何构建

`export`、`jit` 和 plan 的导出都**在对应架构的设备上**构建 CuTe DSL kernel（它的启动要在那里被 capture（捕获），所以交叉构建会被拒绝；cuBLASLt 的 pin 也是按设备的）：

1. **编译。** `cute.compile(host, *cute_args(tensors), stream, options="--keep-cubin --keep-ptx --dump-dir=<a temporary directory> <cute_options>")`（它不读取 DSL 的任何缓存，保留下来的 PTX 和 cubin 放进一个私有目录，而不是工作目录）：Python 到 MLIR，用 DSL 自带的 NVVM 把 MLIR 编译成 PTX，再用它自带的 `ptxas` 把 PTX 编译成 cubin，目标是设备的架构（`sm_103a`）。host 函数必须恰好编译出一个 kernel。cubin 和 PTX 都会保留；cubin 的 ELF 头必须属于该架构，并且包含该 kernel 的代码节。
2. **capture 启动，三次。** 在一个正被 capture 进 CUDA graph 的 stream 上调用编译后的 host 函数，因此 kernel 并不会运行；每次调用时每个张量都位于不同的地址。graph（计算图）必须恰好包含一个操作，即编译后 kernel 的一个 kernel 节点；tilefish 从中读取 grid、block、动态共享内存、cluster 及其调度策略、启动属性，以及每个 kernel 参数的字节（按 `cuFuncGetParamInfo` 给出的大小，必须与 PTX 入口函数的一致）。
3. **构造 ABI**（`classify_params`）。kernel 读取的参数字节，就是它的 PTX 加载的那些字节（常量偏移处的 `ld.param`：`ld.param.v2.b8 {%rs2, %rs3}, [k_param_4+8]` 读取第 8 和第 9 字节）；如果代码以其他方式取参数的地址，则是整个参数（TMA 描述符，`cp.async.bulk.tensor` 通过其地址读取它）。kernel 读取的每个字节要么在三次 capture 中都相同，要么是算子某个张量的地址（一个对齐的 8 字节字，被整体读取，并在每次 capture 中都等于该张量的地址）。于是一个参数是 `pointer`（8 字节，一个张量的地址）、`tensormap`（128 字节、按 64 对齐，第一个字是一个张量的地址：DSL 的 host 代码为静态形状编码出的 TMA 描述符；其第 64 到 127 字节那段代码不写，因此不记录，并以零传入），或 `bytes`（常量：静态布局、空结构体、为该 bucket 编译的值、持久化 tile 调度器的快速除数）。kernel 不读取的字节（结构体的填充，host 函数让它保持在其栈上原来的样子）不参与比较，并以零传入，因此重新构建会写出同样的 manifest。其他任何情况都会被拒绝，并说明是什么情况：会变化但不是地址的字节（host 函数分配的指针、动态形状）、结构体内部的地址、出现在两个参数中或没有出现在任何参数中的张量、顺序与桩函数不同的张量。
4. **检查启动。** 当 PTX 入口函数有 `.reqntid` 和 `.reqnctapercluster` 时，block 必须等于前者，cluster 必须等于后者；cluster 最多 8 个 CTA（线程块），调度策略为 spread（或默认），没有 programmatic dependent launch 或 cooperative launch，block 构成的 grid 必须是整数个 cluster（条目的 grid 以 cluster 计数，见 schema（模式）第 4.5 节），除动态共享内存大小外没有任何函数属性。
5. **SASS 断言**（`sass=tilefish.sass(...)`）与预构建二进制文件一样，在 cubin 及其 PTX 上判定（`docs/zh-CN/explain.md`）；带 trace（追踪记录）的导出（`trace=True`）会保留 DSL 生成的 PTX 和 SASS（`trace/<stem>.<arch>/ptx.ptx`、`sass.txt`）。probe（探针）不可用（编译 kernel 的是 DSL，而不是 Triton）。

在 sm_103 上用 nvidia-cutlass-dsl 4.8.1 实测（cu12 库：CUDA 13.4 的 NVVM，`ptxas` 12.9.83）：示例中的 tcgen05 GEMM 每个 bucket 编译约 0.25 秒，逐元素 kernel 更快。非持久化 GEMM 的 ABI 是一个 64 字节的常量和三个 TMA 描述符（A、B 和 C 的 TMA 存储），cluster 为 2 个 CTA，动态共享内存为 213 248 字节；持久化 GEMM 另外加上其 tile 调度器的四个快速除数（12 字节的结构体，最后 2 字节是 kernel 不读取的填充），它的 grid 是设备能同时运行的 cluster 数。

速度：通过 capture 在 CUDA graph 中的 `tilefish.run` replay 测量（每次调用的设备时间），f16 的 `C = A @ B.T`，f32 累加器，`N = 2048`，`K = 1024`，在一块 sm_103 GPU 的 30 SM（流式多处理器）分区上，与同一问题的已 pin 的 cuBLASLt GEMM（`impl="cublaslt.gemm"`）对比：

| M | 持久化 tcgen05（2-CTA 256x128，cluster 2x1） | 非持久化 tcgen05（相同的 tile） | cuBLASLt |
|---|---|---|---|
| 256 | 7.1 µs（0.71x） | 10.2 µs（0.49x） | 5.0 µs |
| 1024 | 14.2 µs（0.92x） | 25.1 µs（0.52x） | 13.0 µs |
| 4096 | 44.4 µs（1.17x，387 TFLOP/s） | 88.6 µs（0.59x） | 52.1 µs |

使用 256x256 的 tile 时，持久化 GEMM 在 `M = 4096` 下是 cuBLASLt 的 1.25x（408 TFLOP/s）。每种配置写出的字节都与 cuBLASLt 的 pin 相同（tcgen05 的累加顺序相同），因此可以替换它，而不改变模型输出的任何一个比特。

## 3. host 启动什么

该条目是一个 `binary` 条目（`binary.source` 为 `"cute"`），所以每个 host 都把它当作预构建 cubin 启动：参数恰好是它自己的参数，没有 scratch 槽；有 cluster 时用 `cuLaunchKernelEx` 带上 cluster 启动；超过 48 KiB 时设置动态共享内存属性：

* `pointer` 是张量的地址；
* `tensormap` 是它的模板，在第 0 到 7 字节填入张量的地址（小端 `u64`），按值传递（地址必须 16 字节对齐，由该张量的 `pointer_alignment` 前置条件检查，并且小于 2^57）；
* `bytes` 按 manifest 中的内容原样传递。

一次调用（以及 plan 的一个步骤）只给出张量，顺序与桩函数相同；维度从它们的形状绑定，bucket 的二进制文件与任何 extern 一样选择。grid 是条目中的整数（形状是静态的）。Python（`tilefish.run`、plan、`tilefish.capture`）、Rust crate（`Kernel::launch`、`launch_raw`、`LaunchPlan`）和 C ABI 的做法相同，只依据 manifest；CUDA graph 持有 capture 时那些地址的描述符，正如它持有这些地址本身一样。

## 4. 身份、pin 与可复现性

* **声明 digest**（摘要）包含 host 函数（`module:qualname`）、其文件的 sha256、DSL 的版本和选项：换一个文件、换一个 DSL 或换一组选项就是另一个声明，replay 会拒绝它的二进制文件（`first difference at declaration_digest`）。
* **二进制文件由其 PTX 标识**（`source_digest`，kind 为 `tilefish.cute.v1`），正如 Triton kernel 由其 IR 标识：Python 的 replay 用进程中的 DSL 编译该声明（必须是声明中的版本；每个规格的 PTX 每个进程只生成一次），并拒绝 PTX 不同的条目，无论是否已 pin。在无法编译时（没有 DSL、版本不同），只有给出 `pin=` 才接受该条目。Rust host 和 C ABI 不需要 DSL：pin 照例覆盖每个字段和每个文件。
* 在做导出和 replay 的环境中**固定 DSL 的版本**（在 lock 文件中写 `nvidia-cutlass-dsl==4.8.1`）：二进制文件是 DSL 的输出，而 DSL 还处于 beta 阶段。
* **可复现。** 在同一架构上用同一个 DSL 对同样的算子做两次导出，即使在另一个进程、另一个目录中，也会得到相同的文件和相同的 manifest digest（在 sm_103 上实测：GEMM 和逐元素 kernel，cubin 和 PTX 逐字节一致）。不带行号信息时，DSL 不会把任何路径写进二进制文件，而 tilefish 不要求行号信息。
* manifest 记录 DSL 的版本（`binary.cutlass_dsl`）以及它自带的 NVVM 和 `ptxas` 版本（`toolchain`），这些信息读自 PTX 头部和 cubin 的 `.note.nv.tkinfo`。

## 5. 限制

* **只支持静态形状。** 动态布局（`mark_layout_dynamic`、`SymInt`、host 函数的 `Int32` 形状）会把形状放在结构体里传递，或者在 host 上计算一些值（快速除数），而 tilefish 必须能从维度重现这些值：如第 2 节所述，会被拒绝。请把该维度声明为分桶维度（每个 bucket 一个二进制文件）。
* 每个 host 函数一个 kernel，在 tilefish 传入的 stream 上启动，且不入队任何其他东西（没有 memset，没有第二个 kernel，没有被 kernel 取地址的分配）。
* 张量作为参数时位于它自己的地址（偏移 0）：`cute_args` 构造的视图从该张量开始。
* 持久化 kernel 的 grid 是 host 函数在构建时所在设备上给出的（例如 `max_active_clusters`）：在 SM 数不同的设备上，二进制文件的结果仍然正确，但它的 grid 可能填不满该设备。请在它将运行的设备形状上导出。
* 描述符是 DSL 自己对静态形状的编码（DSL 在它的 host 代码中写出它们，而不是用 `cuTensorMapEncodeTiled`）。tilefish 只依赖每次构建都会重新检查的内容：第一个字是地址，其余是常量。以其他方式编码描述符的 DSL 会被构建拒绝，而不会被误读。

## 6. 许可证

CuTe DSL 是 NVIDIA 的软件，采用 NVIDIA Software License Agreement（专有许可证；它还处于 beta 阶段）。tilefish 不随附它、不依赖它，并且除非声明了 CuTe DSL kernel，否则不导入它；请自行安装并阅读其条款。导出写出的 cubin 和 PTX 是 DSL 根据你的 kernel 生成的输出，如何处理它们由你负责：在发布或再分发它们之前（放在 manifest 目录、软件包或嵌入式 host 中），请查阅该协议。测试使用的 CUTLASS 示例（`examples/python/CuTeDSL`，BSD-3-Clause）没有复制进本仓库：测试从一份 checkout 中加载它们（`TILEFISH_CUTLASS_EXAMPLES`）。

## 7. 如何测试

* `python/tests/test_cute_externs.py`（不需要 GPU，不需要 DSL：DSL 的版本被替换）：声明及其拒绝、编译规格和 digest、在合成的 capture 上运行 `classify_params` 和 `launch_of`（TMA GEMM 的、普通 kernel 的、每种拒绝）、PTX 解析器、manifest 条目、读取方的每种拒绝、replay 对 PTX digest 的检查、Rust 和 C++ emitter，以及一次启动打包的字节（通过一个记录它们的驱动）。
* `python/tests/test_gpu_cute.py`（需要 GPU、DSL 和 `TILEFISH_CUTLASS_EXAMPLES`）：CUTLASS 示例中的 tcgen05 GEMM（sm_100、sm_103）或 sm_120 GEMM，以及逐元素 kernel；tilefish 的启动与 DSL 自己的启动逐字节一致；`add` 的字节与 torch 相同，GEMM 与参考结果相差在一个 f16 ulp 以内；在另一个进程和目录中导出得到相同的文件；已 pin 的 replay 与 `jit` 的字节相同；过期的声明和其他声明被拒绝；由一个 CuTe GEMM 和一个 CuTe add 组成的 plan 在 replay 和 capture 时都得到 eager 的字节；SASS 断言和 trace；GEMM 与已 pin 的 cuBLASLt GEMM 的耗时对比（会打印出来）。
* Rust 和 C++ 门禁（`python/tests/rust_gate.py`，在有 DSL 和示例时运行 `cute_kernels.py` 中的用例）：Rust 和 C++ host 写出的字节与 Python 的 replay 相同。
* `crates/tilefish/tests/`：crate 的读取方拒绝 schema 第 4.9 节所拒绝的内容，并像 Python 一样打包描述符和字节。
