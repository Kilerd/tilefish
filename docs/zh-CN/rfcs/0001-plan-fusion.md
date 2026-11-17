# RFC 0001：plan 融合

[English](../../rfcs/0001-plan-fusion.md) | **简体中文**

* **状态：** 已评审；2026-11-15 的决定记录在下文“决定”一节。阶段 0 已测量，阶段 1 已实现（见下文“阶段 1”），阶段 2 的行区域已实现为 `tilefish.fusion(version=2)`（见下文“阶段 2”）；其余部分是设计。
* **日期：** 2026-11-15。
* **范围：** Python 规划器（`tilefish.plan`、`tilefish.runtime.export`）、下沉（lowering）（`tilefish.lower`）、graph（计算图）体的追踪器，以及 `reference` 模式。各个 host（宿主）（Python plan（计划）运行器和 Rust crate）以及两个 schema（模式）都不变。
* **基线：** 下一个版本的接口面：0.0.6 / crate 0.0.4 加上尚未发布的改动：carried buffer（携带缓冲区）、memset 步骤、按 bucket（分桶）的配置（`by_bucket`）、乘积 grid 和持久化 grid、有序求和（`tilefish.sum(order=...)`）、单指令原语（`tilefish.fma`、`rcp`、`neg`、`sin_approx`）、CUDA 的 `erff`、浮点到整数的类型转换以及静态切片。下文的 note 编号是包含这些改动的 `python/src/tilefish/lower.py` 中数值语义契约的编号（note 1 到 19）。

## 概要

目前 `@tilefish.graph` 的函数体只能调用已声明的算子和取视图；graph 函数体中的 `y * scale + bias` 会被拒绝，因此模型的每个逐元素步骤都需要一个手写的 kernel（核函数）或一个专门的 `@tilefish.kernel`。就连 README 第一个示例中的 graph（两个 GEMM（通用矩阵乘）之间的 `silu(g) * u`）也无法规划。

本 RFC 提出 **plan 融合**：规划器把 graph 函数体中追踪到的算术分组成区域，并用 `@tilefish.kernel` 已有的模板和数值语义契约，把每个区域下沉为一个*生成的算子*，就像如今生成 `tilefish.contiguous` 和 `im2col` 那样。融合区域是 plan 中一个普通的 `kernel` 步骤：plan 和 manifest（清单）的 schema 都不变，host 也不变。

数值语义规则是这个设计的要点：**融合本身永远不改变字节。** 每个 graph 都有一个声明的数值语义模式；默认模式以及每个称为*精确（exact）*的模式，都让融合后的 plan 与*未融合组合（unfused composition）*逐字节一致——未融合组合是指每个算术算子都是单独的 kernel、每个中间值都以其声明的 dtype 存储的 plan——并且 `export` 默认逐字节验证这一点。版本 1 只有精确模式；不精确的模式（编译器收缩（contraction）、区域内的 Triton 树形归约、库 epilogue（尾部））是未来工作（第 11 节）。第 5 节陈述这一论断、它所依赖的义务、设计过程中发现的隐患（其中之一是一个可能改变 NaN 载荷的编译器折叠），以及证明它的测试。

## 目录

* 决定（2026-11-15）
* 阶段 0 结果（2026-11-15，sm_103）
* 阶段 1（2026-11-15）
* 阶段 2（2026-11-16）：行区域，融合版本 2
* 0\. 背景：已有的东西
* 1\. 用例与非目标
* 2\. 表层语法与捕获
* 3\. 融合算法
* 4\. 代码生成
* 5\. 数值语义契约及其模式
* 6\. 与 plan 的交互
* 7\. 把 epilogue 和 prologue（头部）融合进 library op（库算子）与 extern（外部 kernel）
* 8\. 对 schema 的影响与迁移
* 9\. 分阶段实现计划、测试计划与开放问题
* 10\. 已有工作
* 11\. 未来工作：不精确的模式，以及其他扩展

附录 A：一个完整示例。

## 决定（2026-11-15）

草案于 2026-11-15 评审。第 9.4 节中已经定下的问题：

* **D1（Q1）：默认开启。** 函数体中有算术的 graph 会以精确的默认模式融合这些算术，不需要任何声明。已有的 plan 都不会改变：在此之前这样的 graph 会被拒绝。
* **D2（Q2）：无序归约单独运行。** 没有 `order=` 的归约是一个单独的 kernel，不融合，它的操作数被存储下来（与未融合组合中一样）；一旦函数体给它一个 `order=`，它就加入区域。
* **D3（Q3）：默认不收缩。** graph 不做任何收缩；要让 `a * b + c` 只舍入一次，就用 `tilefish.fma` 或声明的改写 `fp_contract="mul-add"`（一个精确模式）。
* **D4（Q4）：标量舍入到张量的 dtype**，与 kernel 函数体中相同；torch 的规则是需要选择加入的精确模式 `intermediates="opmath"`。
* **D5（Q5）：版本 1 只有精确模式。** 编译器收缩、区域内的 Triton 树形归约以及改写为库 epilogue 都是未来工作（第 11 节）。
* **D6（Q6）：导出时验证。** `export` 在每个 bucket 上用带种子的输入和特殊值输入运行融合后的 plan 和未融合组合，逐字节比较，有差异就拒绝；`verify=False` 可以退出这一验证。
* **D7（Q7）：冻结的分组规则。** `tilefish.fusion(version=N)` 固定分组规则、限制和改写；版本 1 是第一个，以后的 tilefish 保留每个更早的版本。用另一个版本做出的 plan 会被当作过期而拒绝，消息中会写明该版本。

D3 和 D4 是推荐的默认选项，没有人提出反对。Q4b 以及 Q8 到 Q11 仍然开放；第 9.4 节给出了每一项的推荐默认选项。

## 阶段 0 结果（2026-11-15，sm_103）

在一个 B300 MIG 切片（sm_103）上测量，使用 torch 2.9.1（CUDA 12.8）、Triton 3.5.1、ptxas 12.9 和 cuBLASLt 12.8.4。没有任何结果改变决定；E5 证实了隐患 H1，阶段 1 消除了它。

* **E1，上下文中的树形归约。** 单独的 `r * r` 的 `tl.sum` 行归约（操作数由一个逐元素 kernel 存储），对比同一个归约融合在加载 bf16 的 `x` 和 f32 的残差之后，warp（线程束）数相同：在 96 列（1 个 warp）时没有行不同；在 1000、1024 和 4096 列（4、4 和 16 个 warp）时，**1024 行中有 357 到 399 行不同**（三个种子）。有序求和（`tree(128)`）没有任何行不同。这就是 D2 的前提：Triton 树形归约的字节取决于它所在 kernel 的其余部分。
* **E2，`tl.max`。** 在 1、2、4、8 和 16 个 warp 下，并且把每行的元素打乱，对以下各种行取 `max`：`+0.0` / `-0.0` 混合、零与负数混合、数值中夹杂带随机载荷的 NaN（100、1000 和 4096 列各 2048 行）：**没有行不同**。含有 `+0.0` 的行给出 `+0.0`，NaN 被丢弃。所以在这些输入上 `max` 与顺序无关，可以加入行区域（阶段 2）而无需新的下沉；阶段 2 的测试会持续检查它。
* **E3，速度**（plan 捕获在 CUDA graph（CUDA 图）中；每次 plan 运行的微秒数；融合后的 kernel 就是区域生成的 kernel，`fp_fusion=False`）：

  | 用例 | 行数 | 融合 | 未融合（每个算子一个 kernel） | 备注 |
  |---|---|---|---|---|
  | bias + SiLU，`[M, 3072]` bf16 | 32 / 4096 | 1.45 / 58.6 | 8.61 / 208.5 | 融合与未融合逐字节一致 |
  | 连续两半上的 SwiGLU，`[M, 3072]` | 32 / 4096 | 1.34 / 51.5 | 8.74 / 278.2 | 融合与未融合逐字节一致 |
  | 残差相加 + rmsnorm，1024 列 | 32 / 4096 | 有序 `tree(128)` 2.05 / 80.2；树形 1.23 / 28.6 | D2（三个 kernel）6.07 / 61.4 | |
  | softmax，1024 列 | 32 / 4096 | 有序 2.05 / 54.5；树形 1.44 / 18.0 | D2（四个步骤）6.04 / 44.8 | |

  逐元素融合快 3.6 到 6.5 倍。D2 的耗时在 32 行时约为树形形式的 5 倍，在 4096 行时为 2 到 2.5 倍；在 32 行时有序求和能挽回大部分差距，但当前有序求和的下沉在 4096 行时比 D2 还慢，所以它的速度是阶段 2 的工作。
* **E4，cuBLASLt 的 bias epilogue。** `(gemm_f32(a, b) + bias.to(f32)).to(bf16)` 对比输出为 bf16 的 `gemm_bias(a, b, bias)`，两者都分别在 torch 姿态（posture）和默认姿态下 pin（固定），`M` 取 32、64、128 和 1024，`N x K` 取 1536 x 512、3072 x 1024 和 1024 x 3072，bias 中包含特殊值：24 次比较中**没有任何元素不同**。这是测量结果，不是证明：它为第 11.2 节提供参考。
* **E5，H1。** 在一个 kernel 内部，对每个 bf16 位模式做 `x.to(f32).to(bf16)` **会保留全部 254 个 NaN 的载荷**，而未融合组合的两个 kernel 对每个 NaN 都给出 `0x7FFF`；f16 也是如此（2046 个 NaN 中的 2046 个）。在两个加宽后的值之间做 `where` 再收窄，有 506 个输出不同，全是 NaN：折叠也能穿过 select。每个非 NaN 输出都相同。因此阶段 1 把区域中**每个**收窄的浮点类型转换（不只是紧跟在加宽类型转换之后的那些）都生成为内联 PTX，任何折叠都看不到它。

## 阶段 1（2026-11-15）

已在下一个版本的集成分支上实现，包括 `tilefish._fusion`（融合 pass）、`tilefish.lower` 的 note 20（区域的代码）、`tilefish.fusion`、`tilefish.barrier`、`tilefish.fusion_report` 和 `tilefish.verify_fusion`、导出时的验证，以及 `<graph>.fusion.json`：

* **按设计实现：** D1 到 D7；逐元素区域（凡是下沉能处理的都算：当某个操作数是按行或按列的值时，区域可以用行模板下沉）；归约作为单独的区域；无开销的算子（`load`、`iota`）复制到每个读取它们的区域中；跨区域的 mask 以 `u8` 传递；`intermediates="opmath"` 和 `fp_contract="mul-add"`（两者都从阶段 3 提前）；不计算死算子；没有与其输出形状相同的输入的区域（单独的 `iota(x, 0) < load(n)`）并入它的消费者，使每个区域都是一个可以启动的 kernel；融合报告；Rust 门禁把这些用例作为 plan 运行（`fused_*`）。
* **测试中发现并已消除：** H8 和 H9（第 5.3 节）。它们与 H1 一起使区域的代码*与上下文无关*：即 note 20 的下沉选项 `context_free=True`，每个区域 kernel 以及未融合组合的每个 kernel 都用这个选项下沉。
* 逐元素模板中的**带步长的输入**，如 3.5 的规则 5 所述：非稠密的视图是区域 kernel 的一个 `Strided[...]` 参数，每个轴一个 `StrideOf`，在 `sum(coordinate_k * stride_k)` 处读取；SwiGLU 的两半是对 GEMM 输出的两个视图进行的一个 kernel 步骤。由行模板下沉的区域仍然先用 `tilefish.contiguous` 复制视图。
* **尚未实现：** 行模板中的带步长输入；`tilefish.fused(...)` 作用域；`intermediates="f32"` 模式。
* **此后**（2026-11-15 的决定，“修正实现，信守承诺”）：H8 和 H9 适用于每个 kernel，在那里也已消除。`@tilefish.kernel(fp_fusion=False)` 具有区域的代码（每个浮点 `+ - *` 是一条标量 `.rn` 指令，每个收窄类型转换是内联 PTX；note 2），每个 kernel 的取负都显式给出规范 NaN（note 16），并且在每个 kernel 中，对加宽后的值做收窄类型转换都是内联 PTX（note 1）。每个以 `enable_fp_fusion=False` 构建的二进制文件，包括区域的和手写 extern 的，都会被收缩检查（`tilefish._fpcontract`）读回，SASS 中把乘法和加法融合在一起的二进制文件会被拒绝。
* **测试：** `tests/test_fusion.py`（无 GPU：分组、合法性、命名、模式、报告、拒绝、在另一个进程中得到相同的 plan）和 `tests/test_gpu_fusion.py`（sm_103；与架构无关）：内联类型转换在每个 f32 以及每个 bf16 和 f16 位模式上都等于 `.to()`；各用例在每个 bucket 上用特殊值对比融合与未融合，在参考实现精确的地方还对比参考实现；在每个 bf16 和 f16 位模式上的算子链；在 `2**22` 个位模式上的二元算子；120 个最多 24 个算子的随机 graph，每个也都经过 `verify_fusion`；IEEE math 下的 `opmath` 在每个位模式上对比 torch 的 eager 算子，两个张量的 16 位 `+ - *` 在每一对上对比 torch 的结果（每个算子和 dtype `2**32` 对）；导出检查拒绝做了收缩的区域、写出报告，并且 plan 可以回放和捕获。

## 阶段 2（2026-11-16）：行区域，融合版本 2

实现为 `tilefish.fusion(version=2)`，现在是默认值（D1：融合默认开启；D7：保留版本 1）。改变的是分组规则（因此版本号也变了），以及仅对版本 2 的区域而言的下沉代码（revision 2，`tilefish.lower` 的 note 21）；精确契约、导出时验证、schema 和 host 都与阶段 1 相同。

* **行区域。** 有序的 `sum` 或 `mean`（`mean(order=)` 是新增的：有序求和乘以 `f32(1 / D)`，一条 `mul.rn`）、与顺序无关的 `max`（E2）、`argmax`、`argmin` 和 `gather_rows`（一次复制）会与周围的算子合并；与阶段 1 一样，由下沉的试运行来决定，包含归约的合并结果由行模板下沉。没有 `order=` 的 `sum` 或 `mean` 仍然是单独的区域，其操作数被存储（D2）。一个区域可以写出多个值（rmsnorm 及其残差和；量化后的行及其 scale）。`fma=True` 的有序求和，与它每一步所取操作数来自的那个乘法，在每种分组中都是一个整体，未融合组合中也是如此（版本 1 遇到这种情况会抛出 `KeyError`：它的求和在没有乘积的情况下单独运行）。
* **1.1 的用例**写成 graph（`tests/fusion_graphs.py`，每个都有对应的未融合孪生版本，由 Rust 和 C++ 门禁作为 `fused_*` plan 运行）：残差相加和 rmsnorm，残差流作为第二个输出（一个 kernel；未融合组合有十个），同样的计算但用 fma 求和，带 gelu 和按通道尾部的 layer norm，带来自设备标量的 mask 的 softmax，按行的 fp8 量化，两个列块的 SwiGLU 加一个 norm，一个 gather 和一个 argmax。
* **版本 1 已冻结**，连同它的下沉一起：它的区域保持 revision 1，每个声明 `fusion(version=1)` 的用例都规划得与 0.0.9 和 0.0.10 的规划完全一样（plan 的 digest（摘要）在 `tests/test_fusion.py` 中检查）。用另一个版本做出的 plan 会被当作过期而拒绝，消息中会写明该版本（`_fusion.other_version_of_plan`，只在 plan 过期时才尝试）。
* **有序求和为什么慢（E3），以及 revision 2。** revision 1 用对整行的 `tl.gather(x, k, 0)` 取 lane（通道）链上的每一步；Triton 把源跨越多个 warp 的 gather 通过共享内存下沉，因此每一步都要存储整行、在 barrier 处等待、再加载（每步 `lanes * vector` 列：`tree(128, vector=8)` 在 1024 列时 8 步，`tree(32)` 在 4096 列时 128 步，4096 行时耗时 2.5 ms），而且模板的 warp 规则（每 256 列一个 warp）在 4096 列时给 128 个 lane 分配了 16 个 warp。revision 2 把行重塑为 `[lanes, K]`（lane `j` 的各列沿最后一个轴按其顺序排列，超出 `D` 的列置为 `-0.0`，在 `add.rn` 下它让每个链值保持原样），并通过对半拆分来求每条链，也就是在每个线程持有自己 lane 的各列时做寄存器搬移，而加载的布局正好给出这种分布。第一个版本把 lane 之间的树也用拆分来求，结果 Triton 的布局传播把行的每个值都给了每个线程（926 次 spill（寄存器溢出），4 ms）：因此树改为通过对 `[lanes]` 的 `tl.gather` 进行（warp 内的 warp shuffle），lane 0 则通过一个带 select 合并函数的 `tl.reduce` 取出。warp 数：`BLOCK_D = 2048` 及以下为一个，之后每 1024 列一个（512 个及以上 lane 时每 512 列一个），在测量网格上与 1 到 8 个 warp 中的最佳值相差不超过 2%。已在 1 到 4096 列的每个长度和每个 warp 数下对照每种顺序的 CUDA kernel 检查；比 revision 1 快 1.6 到 37 倍。
* **H10，在这里发现：Triton 3.5.1 中字面量 `-0.0` 就是 `+0.0`**（`tl.where(m, x, -0.0)`、`tl.full(..., -0.0)`，在 kernel 中有和没有 `+0.0` 两种情况下都测量过）。融合仍然精确（融合和未融合的 kernel 写的是同一个字面量），但函数体中的 `where(x > 0, x, -0.0)` 并不是它字面上的意思，在 kernel 和区域中都是如此。现在每个 kernel 都按位写出 `-0.0` 标量（一次 `i32` bitcast；note 3），revision 2 对有序求和的填充也是如此。
* **revision 2 中还有：** 秩为 1 和 2 的视图由行模板就地读取（规则 5；版本 1 的行区域先复制它们），内联的 fp8 类型转换（`cvt.rn.satfinite.e4m3x2.f32`，在每个输入上与 `.to()` 的位相同）。手写的 kernel 可以要求使用 revision 2（`@tilefish.kernel(lowering=2)`）；默认情况下它保持 revision 1 及其源码。
* **advisor（建议器）**（`docs/advise.md`）和 `boundaries()` 跟随 graph 的版本：在版本 2 中，有序归约、`max`、`argmax` 或 gather 不会被报告，行区域既不阻挡相邻算子，也不复制视图。
* **速度**（B300 MIG 切片，sm_103，30 个 SM（流式多处理器）；捕获在 CUDA graph 中的每次 plan 运行的微秒数；v2：区域，一个 kernel；tree：同样的函数体但不带 `order=`，即 D2 的三到五个使用 Triton 树形归约的 kernel，字节不同；v1：版本 1，归约以 revision 1 单独运行；unfused：每个算子一个 kernel；v1 和 unfused 写出的字节与 v2 相同，每个单元格都检查过）。在整个网格上，即 32、128、512、1024 和 4096 行与 512、1024、2048 和 4096 列，v2 在每个单元格中都最快：比树形组合快 2.6 到 8 倍。

  | 形态（pattern）| 行数 | D = 512：v2 / tree / v1 / unfused | D = 1024 | D = 2048 | D = 4096 |
  |---|---|---|---|---|---|
  | add + rmsnorm，返回残差（`mean`、`tree(128, vector=8)`） | 32 | 1.3 / 5.3 / 6.4 / 10.8 | 1.7 / 9.9 / 10.9 / 16.7 | 1.6 / 9.5 / 12.9 / 16.0 | 2.1 / 10.8 / 22.1 / 18.1 |
  | | 4096 | 8.1 / 37.1 / 66.3 / 62.6 | 15.6 / 64.7 / 120.9 / 102.7 | 39.8 / 125.8 / 300.9 / 228.3 | 81.4 / 246.8 / 887.8 / 462.9 |
  | layer norm（两个 mean，`tree(256)`） | 32 | 2.7 / 11.4 / 11.2 / 18.1 | 3.0 / 12.2 / 13.6 / 18.3 | 3.5 / 12.3 / 16.1 / 18.3 | 3.0 / 13.5 / 24.3 / 20.8 |
  | | 4096 | 31.2 / 49.5 / 58.0 / 75.0 | 40.2 / 84.8 / 135.1 / 124.7 | 56.2 / 174.0 / 369.4 / 290.4 | 131.9 / 347.2 / 903.8 / 610.9 |
  | softmax（`max`、`exp`，求和 `tree(32)`） | 32 | 1.2 / 8.6 / 11.1 / 11.2 | 1.4 / 9.1 / 13.5 / 11.8 | 1.8 / 9.4 / 18.1 / 11.8 | 2.4 / 10.9 / 35.6 / 14.0 |
  | | 4096 | 9.0 / 28.6 / 71.5 / 62.5 | 15.0 / 43.6 / 175.2 / 95.9 | 33.5 / 95.7 / 545.5 / 227.8 | 95.6 / 193.6 / 1769.6 / 477.4 |
  | fp8 量化（`max`；没有求和：没有 tree） | 32 | 1.1 / - / 8.9 / 14.2 | 1.1 / - / 8.9 / 13.8 | 1.3 / - / 9.4 / 14.0 | 1.7 / - / 10.5 / 16.0 |
  | | 4096 | 8.5 / - / 32.1 / 58.6 | 15.9 / - / 55.6 / 97.8 | 31.6 / - / 107.0 / 236.1 | 69.4 / - / 204.9 / 483.9 |

  在 4096 行、4096 列时，add + rmsnorm 在 81 us 内搬运 134 MB，达到该切片的带宽（1.6 TB/s）。
* **测试：** `tests/test_fusion.py`（版本 1 冻结、行区域、fma 单元、带步长的行输入、区域的源码等于以 revision 2 下沉的手写 kernel 的源码、warp 规则、过期消息）；`tests/test_gpu_fusion_rows.py`（每种行形态在每个 bucket 上对比融合与未融合，使用带种子的输入、特殊值组成的行以及每个 bf16 位模式；每个 warp 数；80 个由行算子组成的随机 graph，其中四分之一也经过 `verify_fusion`；内联 fp8 类型转换在每个 f32 和 16 位位模式上的测试；`-0.0`；导出、回放和捕获）；`tests/test_gpu_reduction_order.py`（revision 2 在每个 warp 数下对照每种顺序的 CUDA kernel）；`tests/test_gpu_fusion.py` 中的用例，Rust 和 C++ 门禁。每个 GPU 测试都与架构无关；在 sm_120 上运行它们属于以后的门禁，那里的速度规则测量也是如此。
* **尚未实现**（阶段 2 遗留的工作）：由区域对 `InOut` 或 carried 张量进行的存储（规则 4：它仍然是一个复制步骤）、带步长的输出（规则 7）、区域内的稠密 reshape（规则 6）、除 warp 数以外的按 bucket 配置、`tilefish.fused(name=...)` 作用域、`intermediates="f32"`、revision 2 对 sequential 顺序的拆分形式（单个 lane：用 revision 1 的辅助函数），以及对照本 RFC 验收（Q11）的参照：用 graph 函数体写出的已移植模型的 norm 和激活函数。

## 0. 背景：已有的东西

* **graph 函数体本来就追踪算术。** graph 函数体运行在与 kernel 函数体相同的 `Value` 代理之上，所以 graph 中的 `x + y`、类型转换或归约都会成为 graph IR 中的一个类型化节点。规划器（`_plan._walk`）拒绝这样的节点（“arithmetic traced in the graph body ... put it in a `@tilefish.kernel`”），graph 的 `reference` 模式也拒绝它。
* **下沉本身已经是一个融合器。** `tilefish.lower` 用两个模板之一把 `@tilefish.kernel` 的 SSA IR 变成一个 Triton 函数：*逐元素*模板（每个元素一个 lane，可沿任意轴广播，note 12）和*行*模板（每行一个 program，在最后一个轴上归约，note 6、8、17）。它生成的每个算子都有一份数值语义契约，是在 Triton 3.5.1 和 ptxas 12.9 产出的 PTX 上测量的（note 1 到 19），其中大部分是在全部 `2**32` 个 f32 位模式上测量的。
* **生成的算子已经存在。** `tilefish.contiguous`、`im2col`、`im2col1d` 和 `col2im1d` 是由 tilefish 编写 Triton 源码的 extern 算子（`_copyops.py`）：每个形状类别一个算子，第一次被请求时生成并缓存，带一个 torch 参考实现。plan 在 graph 需要的地方启动它（非稠密复制是 `tilefish.contiguous` 的一个 `kernel` 步骤），“所以文档中没有新东西”。
* **身份是机械确定的。** 生成的 kernel 的规格带有 `source_sha256`；前端构建的 TTIR 会被哈希（`source_digest`）；编译选项（`num_warps`、`num_stages`、`enable_fp_fusion`）在声明 digest 和 manifest 条目中；plan digest 覆盖 manifest digest。导出在不同机器间可复现（源码按内容寻址，调试路径与机器无关）。
* **归约的字节取决于什么是已知的。** Triton 的 `tl.sum` 和 `tl.max` 按一棵由 `num_warps` 决定形状的树形归约（不同 warp 数之间，1024 行中有 669 到 878 行不同）；逐元素算子、类型转换、`argmax` / `argmin` 和有序求和不依赖于它。
* **FMA（乘加融合）收缩是一个声明的选项。** `enable_fp_fusion=False` 让 Triton 生成 `mul.rn` / `add.rn`，并向 ptxas 传 `-fmad false`；无论该选项如何，`tilefish.fma` 都是一条 `fma.rn.f32`。

## 1. 用例与非目标

### 1.1 真实推理 graph 需要什么

下面这些算子链，是一个典型的解码器、编码器或编解码器在 GEMM 被 pin 住之后，库调用与手写的注意力和卷积 kernel 之间剩下的部分。每条链都受内存带宽限制：如果每个算子一个 kernel 地运行，每个算子都要把它的张量读写一遍。

| 用例 | graph 函数体写法（草图） | 模板 | 阶段 |
|---|---|---|---|
| GEMM 之后的 bias + 激活 | `silu(h.to(f32) + b.to(f32)).to(bf16)`，`b: [F]` 沿行广播 | 逐元素 | 1 |
| 门控激活（SwiGLU、GeGLU） | `g, u = gemm(x, w).split(F, -1)`；`(silu(g.to(f32)) * u.to(f32)).to(bf16)`：`g`、`u` 是同一个缓冲区的列块 | 逐元素，带步长的输入 | 1 |
| 残差相加 | `h = x + attn`（并返回或存储 `h`） | 逐元素 | 1 |
| 按通道的仿射和周期性激活 | `s = sin(a * x)`；`x + s * s / a`，其中 `a: [C, 1]`、`x: [N, C, T]`（f32） | 逐元素，广播 | 1 |
| 来自设备标量的 mask | `where(iota(s, 1) < load(n), s, float("-inf"))` | 逐元素 | 1 |
| 反量化 | `w.to(f32) * scale`，`scale` 按行或按列 | 逐元素 | 1 |
| norm + 残差，残差流作为旁路输出 | `h = x + r`；`var = sum(h32 * h32, -1, order=...) * (1 / D)`；`y = (h32 * rsqrt(var + eps) * w32).to(bf16)`；返回 `y, h` | 行，两个输出 | 2 |
| layer norm | 两个有序求和（均值、方差）和一个仿射尾部 | 行 | 2 |
| softmax 的各部分 | 缩放、mask、`max`、`exp(x - m)`、有序 `sum`、除法 | 行 | 2 |
| 带按行 scale 的量化 | `amax = max(where(x < 0, -x, x), -1)`；`q = (x / scale).to(e4m3)`（`satfinite`），或带 `rounding=` 和 `saturate=` 的整数类型；返回 `q, scale` | 行，两个输出 | 2 |
| 旋转位置编码 | 最后一个轴的对半拆分或成对交错，按位置 gather 的 `cos` / `sin` 行 | 行 + 数据搬移 | 3 |
| 块缩放量化（MX 格式） | 在最后一个轴的 32 元素块上归约，`ue8m0` 指数 | 分段行 | 4（开放） |
| GEMM epilogue | 把 bias、激活、残差、输出类型转换和 scale 折叠进 GEMM | 库或 extern epilogue | 4（第 7 节） |

### 1.2 非目标

* **在精确模式下，永远不做代数化简。** 没有 `x * 1 -> x`，没有 `x + 0 -> x`，没有重结合，没有强度削减，也不会让常量传播穿过未融合组合会从内存加载的值（第 5.3 节）。融合移动的是 kernel 边界；除非声明的模式说明了怎样改写，否则它不改写算术。
* 版本 1 中**不做重计算**（不把一个节点复制到两个区域中）。两个区域都需要的值只存储一次。
* 阶段 1 到 3 中**不跨调用融合**：对 kernel、extern 或 library op 的调用是区域边界。第 7 节是进入调用的唯一路线。
* **不做可能改变字节的自动调优。** 配置是形状的函数（并且对精确区域而言只是速度上的选择）。
* **不是 kernel 编译器。** GEMM、注意力和卷积仍然是 `extern` 或 library op。融合只生成受内存带宽限制的逐元素 kernel 和行 kernel。
* **没有新的动态形状机制。** bucket 和以前一样严格；在模板需要的地方，区域按 bucket 特化。
* **块缩放的张量**仍然不进入规划（与现在一样）；列出 MX 量化链只是为了完整。

## 2. 表层语法与捕获

### 2.1 graph 函数体

graph 函数体中的算术写法与 kernel 函数体中完全相同：

```python
@tilefish.graph(inputs=("x",))
def mlp(x: Tensor[M, H, bf16], w_gu: Tensor[2 * F, H, bf16], w_d: Tensor[H, F, bf16]):
    g, u = gemm_gu(x, w_gu).split(F, axis=-1)
    a = (tilefish.silu(g.to(f32)) * u.to(f32)).to(bf16)     # fused into one generated kernel
    return gemm_d(a, w_d)
```

算子、它们的 dtype 规则和广播都与 kernel 函数体相同（第 2.5 节），所以函数体可以原样在 `@tilefish.kernel` 和 graph 之间移动。graph 增加的是一个关于算术如何求值的声明。

### 2.2 声明数值语义：`tilefish.fusion(...)`

```python
@tilefish.graph(fusion=tilefish.fusion())                    # the default: exact, as below
@tilefish.graph(fusion=tilefish.fusion(math="ieee"))         # the default math of the body's ops (an op's own math= wins)
@tilefish.graph(fusion=tilefish.fusion(grouping="none"))     # the unfused composition: one kernel per op (the oracle, section 5.1)
```

| 参数 | 取值（默认值在前） | 含义 | 阶段 |
|---|---|---|---|
| `version` | `2`, `1` | 分组规则、限制和改写规则，已冻结（D7）：`2` 生成行区域（阶段 2），`1` 让每个归约单独运行 | 1, 2 |
| `grouping` | `"auto"`, `"none"` | 按第 3 节生成区域，或每个算子一个 kernel（未融合组合） | 1 |
| `math` | `"fast"`, `"ieee"` | 函数体中 f32 算子的默认 math，与 `@tilefish.kernel(math=...)` 相同 | 1 |
| `intermediates` | `"declared"`, `"opmath"`, `"f32"` | 16 位中间值如何舍入：按其 IR dtype；按 torch eager 算子的舍入方式（D4）；在调用之间加宽为 f32 | `"declared"`：1，其余：3 |
| `fp_contract` | `"none"`, `"mul-add"` | 不收缩（D3）；改写为 `tilefish.fma` 的声明式改写 | `"none"`: 1, `"mul-add"`: 3 |
| `verify` | `True`, `False` | `export` 在每个 bucket 上用带种子的输入运行融合后的 plan 和未融合组合，有差异就拒绝；`False` 退出验证（D6，第 5.6 节） | 1 |

每个取值都是精确模式（第 5.4 节）。归约只有带 `order=` 时才加入区域；不带时它是单独的 kernel（D2）。不精确模式的开关是未来工作（第 11 节）。

**默认值。** 什么都不声明但有算术的 graph 得到 `tilefish.fusion()`。这不会改变任何已有的 plan：在此之前这样的 graph 会被拒绝，所以每个已有的 plan 都没有算术，什么也不融合（D1）。

### 2.3 函数体中的控制手段

* `tilefish.barrier(x)` 返回 `x`，并使它成为区域边界：该值获得 plan 的一个缓冲区。barrier 永远不改变字节（每个模式都是精确的）；它的用途是给中间值设 tap（观测点）、为后续阶段保留缓冲区，以及手动塑造 kernel。在 kernel 函数体中它会被拒绝。
* `with tilefish.fused(name="norm"):` 要求其中追踪到的算子构成一个区域；如果做不到（没有顺序的归约、两个迭代空间、经过调用的环），plan 会被拒绝，并给出未通过的规则。`name` 作为生成的算子名的前缀，用于 profile 和报告。区域不能嵌套，也不能跨越调用。
* `tilefish.sum(..., order=...)` 已经存在；本 RFC 给 `tilefish.mean` 增加 `order=`（以该顺序 `sum`，然后 `* f32(1 / D)`，与现在的 `mean` 相同），并且如果实验 E2（第 9.2 节）表明 Triton 的 `tl.max` 在带符号零和 NaN 上与顺序有关，则提出一个全序 `max`。

### 2.4 追踪器的改动

改动很小，并且只限于 graph 的 trace（追踪记录），所以任何 kernel 的 IR、源码或 digest 都不变：

1. `@tilefish.graph(fusion=...)` 在 `GraphOp` 上存储一个 `Fusion` 值（仅 Python；它是推导 plan 的依据之一）。
2. graph 函数体的节点获得两个 kernel 函数体从来没有的属性：`region`（`tilefish.fused` 作用域，如果有）和 `loc`（文件和行号，按可复现构建的路径方式命名，用于报告和消息）。融合 pass 在构建区域的 kernel 之前把两者都去掉，所以它们都不会进入生成的源码。
3. 新节点 `barrier`（仅限 graph 的 trace）。
4. `mean(order=)`，以及在需要时的全序 `max`：新属性只在给出时才写入，所以已有的 kernel 保持它们的 IR。
5. 在 `grouping="none"` 下以及区域必须拆分时，mask 现在可以跨越区域边界：这样的 mask 以 `u8`（0 或 1）存储，读回时为 `!= 0`，这是精确的（note 7）。

追踪器的 dtype 规则不变：在每种模式下，16 位张量在 `exp` 之前仍然需要 `.to(f32)`（`intermediates="opmath"` 是否应像 torch 那样接受 16 位的一元数学函数，是开放问题 Q4b；推荐：否）。

### 2.5 graph 函数体中的 dtype 与广播语义

与 kernel 函数体相同。最后一列是 torch 的行为，因为 graph 经常是从 torch 模型移植过来的。

| 构造 | 操作数 dtype | 广播 | 数值语义（note） | torch eager 的不同之处 |
|---|---|---|---|---|
| `a + b`, `a - b`, `a * b` | 单一 dtype：f32、bf16、f16、i32、i64；**没有隐式提升**（`bf16 + f32` 会被拒绝） | numpy 的规则：按末尾轴对齐，只有静态大小 1 会广播，动态大小永远不广播 | 在该 dtype 中舍入一次；16 位算子用原生指令（`add.rn.bf16x2`）；收缩取决于 `fp_contract`（2） | torch 会提升混合 dtype；对两个 16 位张量，字节一致（torch 先做 f32 运算再舍入，得到的是正确舍入的值；论证见 5.4，在 5.6 中对每一对都做了测试） |
| python 标量或 `const` 操作数 | 取张量的 dtype，并且**先舍入到该 dtype** | 广播 | (3) | torch 以 f32 计算带标量的 16 位运算（`x_bf16 * 0.1` 使用 `f32(0.1)`）；`intermediates="opmath"` 给出 torch 的规则 |
| `a / b` | 仅 f32 | numpy 的规则 | fast math 下为 `div.full.f32`，常量除数变为乘以 `f32(1 / c)`（4）；IEEE math 下为 `div.rn.f32`（13） | IEEE math 就是 torch 的张量除法 |
| `i // c`, `i % c` | 整数，正常量 | numpy 的规则 | Python 的向下取整语义，精确（11） | 无 |
| `==`, `!=`, `<`, ... | 单一 dtype；标量舍入到该 dtype | numpy 的规则 | IEEE 比较，得到 mask（7） | 标量舍入之后无差异 |
| `&`, `\|`, `^`, `~` | mask | numpy 的规则 | 精确（7） | 无 |
| `where(m, a, b)` | `a`、`b` 为同一 dtype | numpy 的规则 | 按位选择，保留 NaN 载荷和带符号零（7） | 无 |
| `iota(x, axis)` | 得到 i32 或 i64 | （一个坐标） | 精确（12） | 无 |
| `rsqrt exp sqrt silu gelu sin cos tanh erf` | 仅 f32 | （一元） | 取决于 math 模式（5、10、13、14、15） | 对 `exp sqrt rsqrt sin tanh silu erf`，IEEE math 在每个 f32 上都等于 torch 的 CUDA 算子；`gelu` 保留 libdevice 的 `erf` |
| `fma`, `rcp`, `sin_approx`, `neg` / `-x` | f32（`neg`：所有浮点类型） | numpy 的规则 | 一条 PTX 指令，永不收缩（16） | `rcp` 就是 `torch.reciprocal`；`neg` 就是 CUDA 的 |
| `x.to(dt)` | 任意存储 dtype；浮点到整数需要 `rounding=` | （一元） | 除各算子外唯一的舍入点；fp8 为 `satfinite`（1、18） | torch 的 fp8 溢出为 NaN 或 inf；torch 的 `u8` 会回绕 |
| `sum`, `mean`, `max` | f32 | 行的最后一个轴 | Triton 的树（取决于 `num_warps`，6）或显式的 `order=`（17） | torch 自己的顺序 |
| `argmax`, `argmin` | 数值 | 最后一个轴 | 精确，与顺序无关（8） | 无 |
| `gather_rows`, `scatter_rows` | 任意 | 行 | 原始复制，越界策略需声明（9） | 无 |
| `load(devscalar)` | 任意 | 秩 0，广播 | 原始读取 | 不适用 |
| 视图（`reshape`、`permute`、`select`、`split`、`x[i]`） | 任意 | 不适用 | 不搬移数据；视图作为带步长的输入供给区域（第 3.5 节） | 不适用 |

## 3. 融合算法

### 3.1 流水线

```
graph IR G  --(numerics rewrite R_mode, section 5.4)-->  declared program P
P  --(grouping, 3.4; legality, 3.5)-->  regions
each region  --(synthetic @tilefish.kernel, 4.1; tilefish.lower)-->  a generated op
P with each region replaced by one call to its generated op  --(the existing _walk)-->  the plan
```

一切都是纯 Python 且确定的：不需要 torch、Triton 或 GPU。因此 Python 的过期检查（拥有该 graph 的 host 重新推导 plan，并要求文档就是那个 plan）照常工作，只是融合成了推导的一部分。已有的遍历不变：它从不看到算术节点，只看到调用。

### 3.2 节点类别

| 类别 | 节点 | 作用 |
|---|---|---|
| CALL | 对 `kernel`、`extern`、`library` 算子的调用 | 边界（阶段 1 到 3） |
| VIEW | `reshape`, `permute`, `select`, `split`, `getitem` | 不含数据；作为带步长的输入供给区域，或结束一个区域 |
| MOVE | `copy`、写入 `InOut` 或 carried 参数的 `store`、`zeros_like`（memset） | 一个复制或 memset 步骤；存储区域某个值的 `store` 则改为该区域自己的写入（3.5） |
| ARITH | 第 2.5 节中每个被追踪的算子 | 可融合 |
| BARRIER | `barrier` | 边界 |

### 3.3 迭代空间

每个 ARITH 节点都有一个迭代空间，它说明哪个模板能计算该节点：

* **E(S)**，在形状 `S` 上逐元素：结果形状为 `S` 的逐元素算子，以及该形状的 `iota` 或类型转换。设备标量的 `load`（秩 0）可以加入任何区域。
* **R(r, D)**，按行：对值 `[..., D]` 的最后一个轴的归约（前面的轴展平为 `r` 行）；以及作用于 `[r, D]`、`[r, 1]` / `[r]`（每行一个值）或只读 `[D]` 输入上的逐元素算子。

当 `S` 为 `[r, D]`（在展平前面的轴的意义下）时，E(S) 与 R(r, D) 兼容：逐元素节点可以加入同样行上的行区域。行区域要求 `D` 是静态的或分了 bucket 的（plan 的每个维度都有 bucket，所以这一点成立；`D` 的每个 bucket 一个二进制文件，与行模板现在的做法相同）。形状更小的广播操作数在区域内只被读取，从不被计算：区域计算出的形状为 `[D]` 的值不能供给同一区域中的 `[r, D]` 值（那样会每行计算一次：即重计算）；它会结束自己所在的区域。

### 3.4 分组

按 IR 顺序进行确定性的贪心生产者-消费者合并，在每台机器和每个 Python 版本上结果都相同：

1. 每个 ARITH 节点一开始各自是一个区域。
2. 按 IR 顺序访问 ARITH 节点。对节点 `n`，按顺序访问它的操作数；对于由 ARITH 节点 `p` 产生的操作数，如果合并后合法（3.5），就合并 `region(p)` 和 `region(n)`。`grouping="none"` 时什么都不合并。
3. 先合并 `tilefish.fused` 作用域；如果它们不能成为一个区域，就拒绝（并给出规则）。
4. 区域的**输入**是它读取但不由它产生的值（参数、调用结果、视图、其他区域的输出），按首次使用的顺序排列；它的**输出**是它产生且被外部读取的值：外部可以是调用、另一个区域、VIEW 或 MOVE 节点，或 graph 的返回。只有该区域读取的值永远不会被存储。

版本 1 不做*水平*融合（共享一个输入但互不供给的两个区域仍然是两个 kernel），也不做重计算。两者都是精确的，以后可以在新的 `version` 下加入。

### 3.5 合法性

当以下各条全部成立时，合并 `A ∪ B` 是合法的；起决定作用的是最后一项检查，即下沉的试运行。

1. **单一迭代空间。** 每个节点都是同一个 `S` 的 E(S)，或者合并结果是一个行区域 R(r, D)（3.3）。
2. **经由 graph 其余部分无环。** 没有路径离开 `A ∪ B` 后又回来（经过调用、视图、move 或另一个区域）。边包括数据边和原地更新的**反依赖**：读取缓冲区版本 `k` 的节点排在产生版本 `k + 1` 的节点之前。
3. **归约**（D2）：没有 `order=` 的归约是单独的区域，它的操作数是一个已存储的值（其生产者在另一个区域中）。这样它的字节只取决于它自己的 kernel，与未融合组合中相同（第 5.3 节，隐患 H4）。有序求和、全序 `max` 和 `argmax` / `argmin` 可以自由加入区域。
4. **原地写入。** 只有当对 `InOut` 或 carried 参数**旧**版本的每次读取都在区域内部并且是 **lane 局部的**时，区域才可以写该参数（把它的某个值 `store` 进去，或者做原本由调用完成的更新）：即读取的是同一个 program 所写的那个元素（逐元素）或那一行（行模板），绝不经过广播、gather、静态切片或沿另一个轴的归约。模板先加载后存储（`Out` 或 `InOut` 只写一次，写入其最终版本），所以这样的写入不与任何东西竞争。其他任何原地写入都会结束区域，并保持为复制步骤。
5. **视图作为输入。** 可证明起始地址按 16 字节对齐的稠密输入是普通的连续参数；不能证明的稠密输入按元素对齐编译（与 `kernel.unaligned` 的做法相同）。非稠密输入（转置、列块、融合 `qkv` 中的一个 head）是生成的算子的一个 `Strided[...]` 参数，其步长作为标量参数传入（每个带步长的 extern 和每个 plan 步骤都已有这套机制），在 `sum(coordinate_k * stride_k)` 处读取（逐元素模板在阶段 1，行模板在阶段 2）。在模板支持之前，视图先由 `tilefish.contiguous` 物化：结果正确，速度较慢，并且会被报告。
6. **区域内的视图。** 在阶段 1 中，两个 ARITH 节点之间的视图会结束区域。阶段 2 允许稠密 `reshape` 留在逐元素区域内（lane 顺序相同；广播操作数按其使用处的形状寻址）。
7. **输出。** 区域把每个输出写进一个新缓冲区、调用者的输出缓冲区（返回值，或某个返回值的整体 reshape），或者按规则 4 写进 `InOut` / carried 参数。作为更大缓冲区的非稠密视图的输出（把 `q`、`k` 和 `v` 写进同一个张量的列块）需要带步长的存储：阶段 2；在此之前它是一个缓冲区加一次复制。
8. **索引范围与限制。** 合并结果遵守模板的索引范围（偏移为 `i32`；动态大小会得到一个 `max_elements` 前置条件）以及 3.7 的限制。
9. **能够下沉。** 对合并结果的合成 kernel（4.1）调用 `tilefish.lower` 成功。下沉是判断模板能做什么的权威，所以分组永远不需要重述它的规则。

### 3.6 步骤的顺序

步骤按压缩后的图（调用、move、区域；数据边和反依赖）的拓扑顺序生成，在就绪节点中选取其首个节点 IR 索引最小的那个。没有融合时这就是 IR 顺序，所以没有算术的 graph 规划结果与以前完全一样。缓冲区的存活范围可能变长或变短；`reuse=True` 按新的范围布局 arena（内存池），与现在的做法相同。

### 3.7 限制（版本 1，属于 `version` 所固定的内容）

* 每个区域最多 64 个 ARITH 节点；
* 每个区域最多 32 个张量参数和输出（一个步骤最多有 256 个指针参数）；
* 行区域同时最多保持 8 个整行的 f32 值存活（根据 `BLOCK_D` 和 warp 数对每个 lane 寄存器用量的估计）；超过时在两个节点之间确定性地拆分；
* 带步长的输入秩最多为 4（与 `tilefish.contiguous` 相同）；
* 没有区域跨越调用、barrier、复制或 memset 步骤。

限制只会拆分区域。在精确模式下，拆分不改变任何字节；它是速度上的选择。

## 4. 代码生成

### 4.1 区域是一个合成的 `@tilefish.kernel`

融合 pass 为每个区域构建用户本来会手写的那个 `KernelOp`：

* **签名**：区域的输入，按首次使用的顺序，每个都是基于 graph 自己的 `Dim` 的 `Tensor[shape, dtype]`（带步长的输入为 `Strided[...]` 及其 `StrideOf` 标量；`DevScalar` 保持原样；规则 4 对应 `InOut`），然后每个输出一个结果；
* **函数体**：区域的节点，复制进一个新的 kernel trace，去掉 `region` 和 `loc`，并对每个没有自己 `math=` 的节点应用 graph 的默认 `math`；
* **选项**：`fp_fusion=False`（D3：声明的收缩是一个显式的 `fma` 节点）；`math` 按声明；`num_warps` 按模板的规则（4.4）。

`tilefish.lower` 把它变成 Triton 源码，方式与下沉任何 kernel 完全相同。这是这个设计的核心选择：**区域继承下沉已经写明并测量过的数值语义契约，而不是用一个需要把这一切重新测量一遍的新代码生成器。** 具有相同节点和选项的区域与手写 `@tilefish.kernel` 生成相同的源码（函数名除外），所以它们编译成相同的指令。

### 4.2 生成的算子、它的名字和源码

* 该算子是附带生成源码的 `ExternOp`，制作方式与 `contiguous` 相同（`_copyops._generated`）：它像任何生成的算子一样构建、启动、导出、回放和规划，它的 `reference=` 是 `reference` 模式对区域节点的解释执行。
* 它的**名字**是 `fused_<hash>`（在 `tilefish.fused(name=...)` 下为 `<name>_<hash>`），其中 `<hash>` 是合成 kernel 规范形式的 sha256 的前 12 个十六进制数字；规范形式包括：融合 `version`、选项、签名（dtype、以维度表达式表示的形状、步长）和节点（算子、属性、以精确十六进制浮点数表示的标量）。两个 graph 中的同一个区域，或一个 set 中两个 plan 里的同一个区域，是同一个算子、同一个 manifest 条目。由于名字是内容哈希，源码永远不会写出它来自的 graph 或节点索引；来源信息写进融合报告（5.5）。
* 源码按内容寻址写出（`<cache>/generated/<sha256>/tilefish_generated/fused_<hash>.py`），与其他生成的源码相同，所以导出保持可复现。
* 模块头部用注释写明区域的数值语义（模板、`fp_contract`、`intermediates`、`reductions`、`math`）。注释计入 `source_sha256`，但不计入 TTIR digest；注释负责说明，IR 负责决定。

### 4.3 融合需要的模板扩展

| 扩展 | 用途 | 阶段 |
|---|---|---|
| 逐元素模板中的带步长（非稠密）输入：操作数在 `sum(coordinate_k * stride_k)` 处读取，坐标按 note 12 的方式计算 | 门控激活、融合 `qkv` 的各个 head、转置的操作数 | 1 |
| 区域中每个收窄的浮点类型转换都写成内联 PTX（隐患 H1，由 E5 证实） | NaN 载荷的精确性 | 1 |
| 行模板中的带步长输入（行步长；列步长为 1 或任意） | 对视图做 norm | 2 |
| 带步长的输出（存储到列块中） | 原地写出 `q`、`k`、`v` | 2 |
| 逐元素区域内的稠密 reshape | head、reshape 后的 `[B, T, D]` | 2 |
| 行内的数据搬移：最后一个轴的对半拆分、成对交错和拼接（`tl.reshape`、`tl.split`、`tl.join`：数据搬移，精确） | 旋转位置编码 | 3 |
| 分段行归约（在最后一个轴每 `k` 个元素的块上） | 块缩放量化 | 4 |

多输出、`InOut` 参数、沿任意轴的广播、设备标量、gather 和 scatter、有序求和以及静态切片都已经存在。

### 4.4 按 bucket 的配置

* **逐元素区域**只有逐元素算子，其字节不取决于块大小、向量宽度或 `num_warps`（义务 O5）。它们的配置是速度上的选择：模板的默认值（`BLOCK = 1024`，4 个 warp），或通过 extern 的 `by_bucket` 机制给出的按 bucket 的表（小 bucket 用更小的块）。改变配置会改变二进制文件和 pin，但永远不改变输出字节。
* 只含有序求和、`argmax` / `argmin` 和逐元素算子的**行区域**同样与布局无关：按 bucket 的 `num_warps`（目前为 `clamp(BLOCK_D // 256, 1, 16)`）是速度上的选择。
* **没有顺序的归约**（D2）是单独的 kernel，在融合后的 plan 和未融合组合中是同一个 kernel。它的 warp 数按 bucket 遵循模板的规则，是它数值语义的一部分（note 6），会被记录并 pin 住，与任何 `@tilefish.kernel` 相同。
* 导出时不做任何自动调优。

### 4.5 grid 还是持久化

区域在模板的 grid 上启动：逐元素为 `(cdiv(n, BLOCK),)`，行为 `(r,)`（像 `T * C` 这样的乘积 grid 现在也可以）。每个元素或每一行都由一个 program 用相同的代码计算，所以持久化启动（`tilefish.persistent`）给出相同的字节；它是以后针对大量短行的一个速度选项，并且保持持久化启动的默认承诺：字节不取决于 SM 数。

## 5. 数值语义契约及其模式

### 5.1 声明的程序与未融合组合

* **声明的程序** `P = R_mode(G)` 是 graph 函数体经过其模式所指定的数值语义改写（5.4）之后的结果。每个改写都是一个 IR 到 IR 的函数，不取决于区域如何形成；在默认模式下它是恒等变换。`P` 的每个 ARITH 节点都有一个 dtype，其结果是按其算子的契约计算出的该 dtype 的值：算术运算符或类型转换舍入一次，数学函数则是其 note 中固定的指令序列。
* **未融合组合** `U(P)` 是 `grouping="none"` 下 `P` 的 plan：`P` 的每个 ARITH 节点都是单独生成的 kernel，用同样的模板和同样的逐节点选项下沉，每个中间值都以其 dtype 存储（mask 存为 `u8`，秩 0 的值存为单元素张量）。不对数据做算术的节点（设备标量的 `load`、`iota`）在每个消费者中重新计算而不是存储，这是精确的。
* `U(P)` 是一个真正的 plan：它能规划、导出、在 `jit` 和 `replay` 中运行、捕获，它的 tap 能显示每个中间值。它就是判定基准（oracle）。

### 5.2 论断及其依据

> **精确性。** 在精确模式下，对每个 bucket 和每个输入（包括每个位模式、NaN 载荷、带符号零、次正规数和无穷大），融合后的 plan `F(P)` 写进每个输出、每个 `InOut` 和每个 carried buffer 的字节都与 `U(P)` 相同。

当这些义务成立时，它在构造上就成立。每项义务都有测试（5.6），能做到的还有构建时的守卫检查。

| | 义务 | 为什么成立 | 检查 |
|---|---|---|---|
| O1 | 存储一个值再把它加载回来是恒等的 | 加载和存储搬移的是位（每种存储 dtype，包括 NaN 载荷）；存为 `u8` 0/1、读为 `!= 0` 的 mask 是同一个 mask | 特殊值测试 |
| O2 | 算子的代码不取决于它的相邻算子 | 下沉只根据节点本身生成每个节点的表达式；标量按节点确定类型 | 源码相等：区域的源码等于相同节点构成的 kernel 的源码 |
| O3 | 跨两个算子不发生任何收缩 | 在 sm_100 及以后，`enable_fp_fusion=False` 不够（H9）：区域把每个浮点 `+ - *` 写成一条内联 PTX 的标量 `.rn` 指令（`context_free=True`），ptxas 不会收缩它；声明的收缩是显式的 `tilefish.fma`（内联 PTX）；IEEE 辅助函数带 `.rn` | 每个条目中的选项；发现 H9 的那个用例的 SASS；随机 graph |
| O4 | 跨两个算子的折叠、重结合或 fast-math 改写都不改变任何一位 | 预期 Triton 除收缩外不设置其他 fast-math 标志（守卫检查在每次构建时检查这一点）；LLVM 只折叠它认为精确的东西，而它认为精确的某个折叠在 NaN 载荷上并不精确（H1） | 一个守卫检查，扫描每个区域的 LLVM IR 中的 `reassoc`、`nnan`、`ninf`、`nsz`、`arcp`、`afn`；NaN 载荷测试 |
| O5 | 逐元素算子的结果只取决于它在该元素处的操作数 | 按 lane 的代码；打包的 `x2` 指令对每一半的计算与标量指令相同；块大小、向量宽度和 warp 数只负责分配 lane | 同一区域在多种配置下给出相同的字节 |
| O6 | 区域内的每个归约都与布局无关 | 有序求和（note 17）、`argmax` / `argmin`（note 8）；Triton 树形归约不是（H4），所以它单独运行（D2） | E1 和 E2；多种 warp 数下的有序求和测试 |
| O7 | `F` 和 `U` 由同一个工具链、同一个 libdevice 和同一个 CUDA 数学库编译 | 同一次导出 | CUDA 数学守卫检查和 manifest 的 `toolchain` |
| O8 | 每个算子的 flush-to-zero 行为是它自己的 | 相同的编译选项，相同的逐算子代码（`.ftz` 来自算子，note 5） | 次正规数扫描 |

### 5.3 设计中发现的隐患，以及每个隐患如何消除

| | 隐患 | 影响 | 解决办法 |
|---|---|---|---|
| H1 | MLIR 的 arith 折叠器或 LLVM 的 instcombine 把 `fptrunc(fpext(x))` 折叠为 `x`（它们认为这是精确的），并且也能穿过加宽值之间的 select（由 E5 证实）。在 `U` 中，加宽和收窄是两个 kernel，收窄是 `cvt.rn.bf16.f32`，它把每个 NaN 变成规范形式（`0x7FFF`，note 1）；融合并折叠后，输入的载荷会保留下来 | 对载荷不规范的输入 NaN 做 `x.to(f32).to(bf16)` 结果不同（只涉及 NaN，并且只涉及来自输入的 NaN：GPU 算术产生的是规范 NaN） | 区域中每个收窄的浮点类型转换都以内联 PTX 生成（`cvt.rn.bf16.f32` / `cvt.rn.f16.f32`，即单独的类型转换编译出的指令），任何折叠都看不到它；在每种类型转换的每个位模式上以及在带 NaN 载荷的类型转换对上做了测试 |
| H2 | `U` 从内存加载的值在 `F` 中变成常量（折叠进区域的 memset，一个已知的零），然后 MLIR 或 LLVM 的折叠器就会生效（`x + (-0.0) -> x` 保留了 GPU `add` 会规范化掉的 NaN 载荷） | NaN 载荷和带符号零的差异 | 规则：**`U` 加载的值，`F` 要么加载，要么用相同的算子计算；永远不用常量替代它**。常量只能是 IR 自己的字面量，它们是逐节点的，所以在 `F` 和 `U` 中折叠方式相同 |
| H3 | 跨算子边界的收缩（`a * b` 在一个算子中，`+ c` 在下一个算子中） | 23% 的元素不同（note 2） | O3：除非声明，否则不收缩 |
| H4 | Triton 树形归约的操作数布局（每个线程的元素数，来自锚定布局的那些加载）取决于整个 kernel，所以当其他 dtype 的其他加载与它共处一个 kernel 时，同一个 `tl.sum` 可能按另一棵树相加 | 最后一位的差异，与数据有关 | D2：无序归约单独运行，并且它们的操作数被存储；实验 E1 测量树实际改变的频率 |
| H5 | `tl.max` 在 `+0.0` 和 `-0.0` 上以及在 NaN 上可能与顺序有关 | 零的符号 | E2：在 1 到 16 个 warp 下以及打乱顺序后，在每个测试过的行上都与顺序无关；`max` 可以加入行区域，阶段 2 的测试会持续检查它 |
| H6 | 数学算子背后的库随 CUDA 或 libdevice 版本变化（CUDA 13.2 改变了 `erff`） | 字节不同，`F` 和 `U` 一样 | 不是融合隐患（O7）；CUDA 数学守卫检查会发出警告 |
| H7 | 已在 2026-11-15 的评审中随它所针对的模式 `fp_contract="compiler"` 一起撤回，D5 把该模式留作未来工作（第 11 节） | — | — |
| H8 | （在阶段 1 中发现）ptxas 把 `neg.f32` 折叠进它的消费者，作为取负的操作数（`-x` 的 `where`：`-R` 的 `FSEL`），这会翻转 NaN 的符号，而单独的 `neg.f32` 给出规范 NaN | 融合时 NaN 为 `0xFFFFFFFF`，未融合时为 `0x7FFFFFFF`（一个随机 graph 的 1536 个元素中有 61 个） | 区域的取负显式给出规范 NaN（`testp.notanumber` 或 `setp.nan`，`selp`） |
| H9 | （在阶段 1 中发现）LLVM 把两个 lane 的 f32 运算打包成 `mul.rn.f32x2` / `add.rn.f32x2`，而 ptxas 12.9 即使在 `--fmad=false` 下也会把这一对收缩成 `FFMA2`（在为 sm_100a 汇编同一份 PTX 得到的 SASS 中读到） | 一个 ULP，一个随机 graph 的 1536 个元素中有 33 个；并且在运算被打包的地方，`@tilefish.kernel(fp_fusion=False)` 不是两次舍入 | 区域把每个浮点 `+ - *`（以及 `silu` 和 `gelu` 中的乘法）写成标量 `.rn` 内联 PTX：SASS 中是分开的 `FMUL` 和 `FADD` |

### 5.4 模式

版本 1 的每个开关都是一个 IR 改写，所以每种模式都对改写后的 `P` 保持 `F(P) == U(P)`。改写定义在 IR 上，在分组之前进行，所以它的舍入点永远不取决于区域落在哪里。

| 开关 | 取值 | 含义 | 与未融合组合的关系 |
|---|---|---|---|
| `intermediates` | `"declared"`（默认） | 恒等改写：每个算子舍入到 IR 给它的 dtype（`bf16 * bf16` 是 bf16 乘法；`x.to(f32)` 保持 f32 直到遇到类型转换） | 精确 |
| | `"opmath"` | 把 torch 的 eager 规则作为改写（D4）：带标量操作数的 16 位运算变为 `op(a.to(f32), f32(c)).to(bf16)`，其中标量的原始 Python 值舍入到 f32（IR 保留了这个值）。两个张量的 16 位 `+ - *` 不变：两个 16 位值的 f32 乘积是精确的，f32 和要么是精确的，要么离 16 位的中点足够远，使得第二次舍入不会产生差异，所以 torch 的字节就是原生运算的字节（5.6 的穷举成对测试检查这一点） | 对改写后的程序是精确的；在 `math="ieee"` 下，对于 IEEE 形式就是 torch 形式的算子（note 13、15、16），它的目标是与 torch 的 eager 组合逐字节一致（有待测量，5.6） |
| | `"f32"` | 把类似 Inductor 的加宽作为改写：每个由 ARITH 节点产生的 16 位值都以 f32 计算，只在它离开算术的地方才舍入到其声明的 dtype：调用的参数、返回值、store、barrier。函数体中写出的类型转换仍然是舍入点。定义在 IR 上（调用、返回、store、类型转换），而不是在区域上 | 对改写后的程序是精确的（区域拆分时存储 f32 值）；不等于 `"declared"` |
| `fp_contract` | `"none"`（默认） | 不收缩（D3） | 精确 |
| | `"mul-add"` | 一个改写：`mul` 没有其他用途的 f32 `add(mul(a, b), c)` 或 `add(c, mul(a, b))` 变为 `fma(a, b, c)`；`sub(mul(a, b), c)` 变为 `fma(a, b, neg(c))`，`sub(c, mul(a, b))` 变为 `fma(neg(a), b, c)`；当 `add` 的两个操作数都符合条件时，收缩左边那个。16 位收缩需要 16 位的 `fma` 原语（Q10） | 对改写后的程序是精确的（`fma` 和 `neg` 在参考实现中是精确的，note 16） |
| `math` | `"fast"` / `"ieee"` | 逐算子的 math（note 13） | 两种都精确（`F` 和 `U` 中是同样的代码） |
| `grouping` | `"auto"` / `"none"` | `"none"` 就是 `U(P)` 本身 | 精确 |

没有顺序的归约不是一种模式：它在 `F` 和 `U` 中都是单独的 kernel（D2）。

### 5.5 如何计入 digest，以及 pin 意味着什么

* 模式决定 `P`，因此决定每个生成的源码（`source_sha256`）及其 TTIR（`source_digest`）；它决定编译选项（`enable_fp_fusion`、`num_warps`），这些选项在条目的 `options` 和声明 digest 中。条目计入 manifest digest，manifest digest 计入 plan digest。**因此 plan 的 pin 就 pin 住了每个融合 kernel 的数值语义**，不需要新字段。
* 融合设置是 Python 推导 plan 的依据之一，所以模式改变了的 graph 会被当作过期而拒绝，就像算子改变了的 graph 一样。
* 在精确模式下，根据 5.2，pin 还意味着字节就是 `U(P)` 的字节；`tilefish.verify_fusion(graph, *inputs, buckets=...)` 可以随时在任意输入上检查这一点，`export` 在写出任何东西之前用带种子的输入检查它，除非 `verify=False`（D6，5.6）。
* **融合报告。** `tilefish.fusion_report(graph, buckets=...)` 和每次 `export`（作为 plan 旁边的 `<graph>.fusion.json`）列出每个区域：它生成的算子、模板、迭代空间、它吸收的 graph 节点（索引、算子、`loc`）、它的输入（以及视图是带步长读取还是被物化）、它的输出、它保留的每个舍入点、应用的每个改写（做了哪些收缩、`"f32"` 去掉了哪些舍入）、每个归约及其顺序，以及模式。它是说明性的（host 从不读取它），也是派生的（Python 根据 graph 重新生成它）。它就是移植手工融合引擎时所需要的“有文档记录的融合步骤”：哪些原始算子被合并了，数值上改变了什么。

### 5.6 测试如何证明它

* **构造（无 GPU）。** 区域的源码等于具有相同节点和选项的手写 `@tilefish.kernel` 的源码。每个改写都以 IR 输入、IR 输出的方式测试。分组是确定的（两个进程，相同的名字、源码和 plan digest），并且在重命名局部变量下保持不变。每条合法性规则都有一个拒绝测试或拆分测试（对正在被写入的 `InOut` 的广播读取、经过调用的环、在原地更新之后读取的视图）。
* **融合对比未融合（GPU，核心部分）。** 在每个 bucket 上逐字节比较 `F(P)` 和 `U(P)`：
  * 穷举：对每个算子对和类型转换对，在每个 bf16 和 f16 位模式（65 536 个输入）上测试一个和两个一元算子的链（带它们需要的类型转换）；在全部 `2**32` 个位模式上测试选定的 f32 链，与 IEEE 测试的做法相同；
  * 大网格：在 `2**24` 到 `2**26` 个随机位模式对上测试二元算子，特殊值的笛卡尔积（两种零、两种无穷大、带规范和非规范载荷的 NaN、最小和最大的次正规数与正规数），以及抵消的情形；
  * 基于性质：一个带种子的生成器，在一个调用和一个返回之间生成随机的算术 DAG（2 到 30 个节点，1 到 4 个输入，随机的 dtype、类型转换、标量、广播、mask、设备标量、带步长的视图）：在任意机器上规划并下沉 10 000 个 graph，在每种 GPU 架构上用特殊值输入对照 `U` 运行 500 个；
  * 归约：有序求和融合时对比单独运行时，行为 1 到 16 384 列，包括各种量级、带符号零、NaN、无穷大和次正规数，在 1 到 16 个 warp 下；
  * 1.1 的用例写成 graph，对照 `U`，也对照具有相同声明数值语义的手写 kernel。
* **对照参考实现。** `reference` 模式逐节点解释执行 `P`（改写后的程序）。在每个算子的参考实现都精确的地方（IEEE math、类型转换、比较、`fma`、`rcp`、`neg`、有序求和），融合后的二进制文件与参考实现逐字节一致；其他地方适用列表中的容差。IEEE math 下的 `intermediates="opmath"` 与 torch 对相同算子的 eager 组合进行比较，两个张量的 bf16 和 f16 `+`、`-` 和 `*` 在每一对位模式上与 torch 的 eager 运算比较（每个算子和 dtype `2**32` 对），正是这一点让 `opmath` 可以保持它们不变。
* **host。** Rust 门禁加入融合后的 plan：Rust 对照 Python 回放逐字节比较，包括直接启动和在 CUDA graph 中，带 carried buffer 和 shared arena（共享内存池）。
* **导出时验证**（D6）。`export` 在导出设备上的每个 bucket 上，用带种子且含特殊值的输入在 `jit` 中运行 `F` 和 `U`，有差异就拒绝；`verify=False` 退出验证。它自己的测试：一个故意破坏的区域（偷偷加入一个收缩）会被拒绝。
* **两种架构。** 以上所有内容都在 sm_103 和 sm_120 上运行。`F == U` 是按架构声称的；不声称跨架构相等（只有在算子自身的 PTX 相同的地方才成立，与现在一样）。

## 6. 与 plan 的交互

**host 不变。** 区域是一个生成条目的 `kernel` 步骤，与 `tilefish.contiguous` 步骤一样。Python 运行器和 Rust crate 像启动任何 Triton 条目一样启动它；两个 schema 中都没有新东西。

* **arena 与复用。** 融合掉的中间值永远不会获得缓冲区，所以 arena 会变小。`reuse=True` 按新的存活范围布局剩下的缓冲区。
* **carried buffer。** carried 参数和任何参数一样是区域的输入；区域可以按规则 4（3.5）写它，这把“`h[...] = op(x)` 由 op 的启动直接写入”从调用扩展到了区域。arena 对 carried 形状的记录不变。
* **memset 步骤。** `zeros_like` 仍然是 memset 步骤，永远不会作为常量折叠进区域（H2）。区域从内存中读取这些零。
* **shared arena 与 plan set（plan 集）。** 不受影响。出现在一个 set 的多个 plan 中的区域是同一个 manifest 条目（它的名字是内容哈希）。
* **捕获。** 融合后的步骤就是 kernel 启动；`tilefish.capture`、`replay_graph` 以及在调用者的 graph 中捕获都与以前一样工作。
* **tap 与 `run_step`。** 被融合掉的值没有缓冲区，所以 `taps=` 会拒绝它，并建议添加 `tilefish.barrier`，或者对未融合的 plan（`grouping="none"`）设 tap——在精确模式下，未融合 plan 的中间值正是融合 kernel 计算出的那些值。`run_step` 按融合后 plan 的步骤编号。
* **权重、输入与角色。** 不变：区域的参数是 plan 中带有各自角色的缓冲区。
* graph 的 **`reference` 模式**通过 `P` 解释执行 ARITH 节点（目前它拒绝这些节点），所以声明的模式会体现在参考实现中。
* **`jit`** 在内存中构建生成的算子；构建缓存像对待任何生成的 kernel 一样为它们计算键。
* **Rust 启动器生成**（`hosts.rust.emit`）每个区域多看到一个 Triton 条目；那里什么都不变（类型化的启动器中是否跳过生成的条目只是外观问题）。

## 7. 把 epilogue 和 prologue 融合进 library op 与 extern

RFC 0003（`docs/rfcs/0003-epilogue-fusion.md`）用一份设计及其契约和测量结果取代了本节以及 11.2 和 11.3 中的草图。

GEMM 之后的逐元素工作可以这样运行：(a) 在它之后的一个生成的 kernel，(b) cuBLASLt 或 cuDNN 的 epilogue，或 (c) extern 自己 kernel 内部的 epilogue。版本 1 只有路线 (a)（D5）；路线 (c) 是阶段 4；路线 (b) 和 prologue 是未来工作（第 11.2 节）。

| 路线 | pin 住的是什么 | 相对 `U` 是否精确？ | 状态 |
|---|---|---|---|
| (a) 调用之后的生成 kernel | 区域的机器码 | 是，根据 5.2 | 默认路线，阶段 1 到 3；对调用的输出多一次读和写 |
| (c) extern 的 epilogue hook（钩子） | 内联了 hook 的 extern 二进制文件（Triton 把 `tl.constexpr` 的 jit 函数折叠进 IR，所以 TTIR digest 覆盖了它） | 是，前提是 hook 作用于 kernel 本来要存储的值、在 kernel 自己舍入到输出 dtype 之后，并且它的算术不能与 kernel 的算术收缩（hook 的运算以 `.rn` 内联 PTX 生成，或在边界处做一次类型转换）；kernel 自身的算术不变这一点由导出时验证（D6）检查，而非证明 | 阶段 4：`@tilefish.extern(..., epilogue=tilefish.hook("EPILOGUE", out="c"))`；kernel 在存储之前调用 `c_tile = EPILOGUE(c_tile, ...)`，plan 传入区域生成的 `@triton.jit` 函数；每个 extern 加 epilogue 的组合都是一个新的二进制文件，验证拒绝的组合会回退到路线 (a) |
| (b) cuBLASLt 或 cuDNN 的 epilogue | 库的算法或引擎 pin、版本、canary（哨兵值） | 只能靠测量（第 11.2 节） | 未来工作 |

用户声明为 library op 的 epilogue（目前为 `impl="cublaslt.gemm_bias"`）是一次具有库所声明数值语义的普通调用，不是融合，不受影响。

## 8. 对 schema 的影响与迁移

### 8.1 schema

* **Plan schema 1：** 无变化。区域是 `kernel` 步骤。
* **Manifest schema 3：** 无变化。区域是一个带 `source_sha256` 的 `triton` 条目，与每个生成的算子一样。
* **仅 Python 的新增内容：** `@tilefish.graph(fusion=...)`、`tilefish.fusion`、`tilefish.barrier`、`tilefish.fused`、`mean(order=)`、`tilefish.fusion_report`、`tilefish.verify_fusion`、`export(..., verify=...)`，以及说明性的 `<graph>.fusion.json`。
* **以后可能增加（只做添加）：** 一个可选的条目字段，写明融合条目的模式，让没有 Python 的审计者能从 manifest 中读到它。两个严格读取器都必须先接受它，而旧的读取器会拒绝带有该字段的 manifest；本 RFC 不提议在版本 1 中加入它（Q8）。

### 8.2 迁移手写的算子

* **什么都不必迁移。** 每个已有的 `@tilefish.kernel`、extern 和生成的算子都保持其身份；没有算术的 graph 规划结果与以前相同。
* **移进 graph 的 `@tilefish.kernel` 函数体**，在该 kernel 本来就能精确表达时保持其字节：即声明了 `fp_fusion=False`，并且每个归约都有顺序（或者没有归约）。在 `with tilefish.fused():` 下，区域会生成该 kernel 自己的源码。依赖编译器收缩（kernel 的默认行为）或 Triton 树形归约的 kernel，在版本 1 中仍然是 `@tilefish.kernel`（对应的 graph 模式是未来工作，第 11.1 节）。更好的迁移方式是把数值语义写明：kernel 想要一次舍入的地方用 `tilefish.fma`，想要某个顺序的地方用 `order=`。
* **手写的逐元素 extern**（bias 加激活、残差相加、类型转换），一旦有测试表明 graph 声明的数值语义能复现它们（在特殊值输入上用 `verify_fusion` 对照该 extern），就可以用 graph 算术替代。单指令原语和有序求和的存在，就是为了能用 DSL 写出手工融合的 CUDA kernel 的字节。
* **对另一个 graph 的调用**仍不支持；普通的 Python 辅助函数会被内联追踪，这才是复用算术的方式。
* **把 `@tilefish.kernel` 调用内联进区域**（把 DSL kernel 调用当作它的节点）只有在其选项与区域相同、其归约有顺序时才是精确的；它会改变已有的 plan，所以它是以后的选择加入项（`fusion(inline_kernels=True)`），不属于版本 1。

## 9. 分阶段实现计划、测试计划与开放问题

### 9.1 里程碑

| 阶段 | 内容 | 里程碑（完成标准） |
|---|---|---|
| 0：实验 | 下面的 E1 到 E5，在 sm_103 上（然后在 sm_120 上） | `max` 的下沉方式已确定，H1 已证实或排除，速度预期已确立；E1 和 E4 为未来工作提供参考 |
| 1：逐元素区域 | 带默认模式（`intermediates="declared"`、`fp_contract="none"`、`math="fast"` / `"ieee"`）的 `tilefish.fusion(version=1)`，默认开启（D1），`grouping="none"`、`barrier`、`fused`；融合 pass 和合成 kernel；逐元素模板中的带步长输入；针对 H1 的类型转换生成；fast-math 标志守卫检查；graph 对 ARITH 节点的 `reference` 模式；融合报告；`verify_fusion` 和默认开启的导出时验证（D6）；归约作为单独的区域运行（行模板保持原样） | README 中的 `mlp` graph 能够规划并运行；bias + 激活、残差相加、反量化、周期性激活和 mask 能够融合；在穷举、网格和基于性质的测试集上 `F == U`；Rust 门禁通过；导出可复现 |
| 2：行区域 | 区域中的有序归约（`sum`、`mean(order=)`、与顺序无关的 `max`、`argmax`）；多输出和旁路输出；`InOut` 和 carried 写入（规则 4）；行模板中的带步长输入；带步长的输出；区域内的稠密 reshape；按 bucket 的配置 | rmsnorm + 残差、layer norm、softmax 的各部分和按行量化能够精确融合，并且运行速度至少与等价的手写 `@tilefish.kernel` 一样快 |
| 3：精确模式与数据搬移 | `intermediates="opmath"` 和 `"f32"`、`fp_contract="mul-add"`（如果 Q10 被接受，则带 16 位 `fma`）；用于旋转位置编码的行内拆分 / 交错 / 拼接 | 每种模式的测试：`opmath` + IEEE 对照 torch eager；`mul-add` 对照显式 `fma`；每种模式都 `F == U` |
| 4：extern epilogue | extern epilogue hook（路线 (c)）；水平融合和分段归约作为单独的事项 | 带融合 bias-激活 epilogue 的 extern GEMM 在每个 bucket 上等于 `U` |

一切不精确的东西都不在这些阶段之内（第 11 节）。

### 9.2 阶段 0 实验（每个都很小，在一块空闲 GPU 上进行）

* **E1，上下文中的树形归约。** 同一个 `tl.sum` 行 kernel 单独运行，以及融合在其他 dtype 的加载之后（bf16 和 f32 操作数，与 rmsnorm 一样）：在归约顺序测试集上统计不同的行数。证实 D2 的前提，并为第 11.1 节提供参考（预期：有行不同，因此树形归约永远不能精确地加入区域）。
* **E2，`tl.max` 与顺序无关。** 在 1 到 16 个 warp 下测试 `+0.0` / `-0.0` 混合以及含 NaN 的行。决定区域中 `max` 的下沉方式。
* **E3，速度。** 针对 bias + SiLU、带步长两半上的 SwiGLU、rmsnorm + 残差和 softmax，在 decode 和 prefill 的行数下，在 B300 切片和 RTX 5090 上比较融合、未融合和手写 `@tilefish.kernel`。测量 D2 的代价（单独运行的无序归约对比融合的有序归约）。
* **E4，cuBLASLt 的 bias epilogue。** `(gemm_f32 + b.to(f32)).to(bf16)` 对比输出为 bf16 的 `gemm_bias`，覆盖一个测试集中的每个 pin，bias 取特殊值。为第 11.2 节提供参考（路线 (b) 以后是否值得构建）。
* **E5，H1。** 证实 Triton 3.5.1 上的 `fptrunc(fpext(x))` 折叠，以及内联 PTX 类型转换能消除它。

### 9.3 测试计划摘要

按阶段划分的第 5.6 节：在每台机器上运行构造测试和拒绝测试（`pytest`）；在 GPU 上运行 `F == U` 测试集、参考实现比较和 Rust 门禁（`pytest -m gpu`、`rust_gate.py`），在 sm_103 和 sm_120 上；在融合后的导出上运行可复现导出测试；用例 graph 作为 golden，连同其融合报告一起提交。

### 9.4 开放问题

Q1 到 Q7 已经决定（上面的 D1 到 D7）。以下仍然开放，每一项都附有默认选项，除非评审另有意见，否则将实现该默认选项：

* **Q4b. `opmath` 下的 16 位一元数学函数。** `intermediates="opmath"` 是否应像 torch 那样接受 `exp(x_bf16)` 之类的写法？那会把追踪器的 dtype 规则与 graph 的模式耦合起来。**推荐默认选项：否**；在每种模式下，函数体都写 `.to(f32)`。
* **Q8. 记录模式的 manifest 字段。** **推荐默认选项：版本 1 中没有**：模式写在每个生成源码的头部（计入 `source_sha256`）和融合报告中。增加字段需要先更新两个严格读取器，并且旧的读取器会拒绝带有该字段的 manifest。
* **Q9. 命名。** **推荐默认选项：按提议**：`@tilefish.graph(fusion=tilefish.fusion(...))`、`tilefish.barrier`、`tilefish.fused`、`tilefish.fusion_report`、`tilefish.verify_fusion`，以 `export(..., verify=False)` 作为退出方式，生成的算子命名为 `fused_<hash>`（或 `<name>_<hash>`），以及 `<graph>.fusion.json`。
* **Q10. 16 位 `fma`。** **推荐默认选项：是，在阶段 3** 与 `fp_contract="mul-add"` 一起：用于 bf16 和 f16 的 `tilefish.fma`（`fma.rn.bf16`、`fma.rn.f16`，内联 PTX），按 f32 `fma` 的方式测量（note 16）。在此之前 `"mul-add"` 只收缩 f32。
* **Q11. 验收。** **推荐默认选项：** 阶段 2 在已经移植到 tilefish 的模型上测量，每个模型都按其原始的算子结构编写：在每个 bucket 上 `F == U`，并在 sm_103 和 sm_120 上与它们手工融合的 kernel 比较速度；1.1 的用例 graph 作为 golden 连同融合报告一起提交。

## 10. 已有工作

| 系统 | 如何融合 | 16 位链的中间值 | FMA 收缩 | 归约顺序 | 与未融合程序的关系 |
|---|---|---|---|---|---|
| torch.compile / Inductor | 下沉到循环 IR；它的调度器把逐点节点和归约节点（纵向和横向）融合进 Triton kernel；在自动调优下把 epilogue 融合进它自己的 GEMM 模板 | 默认在 kernel 内以 f32 计算（`codegen_upcast_to_fp32`），因此跳过了 eager 会做的舍入；有一个选项可以模拟 eager 的类型转换 | 后端编译器的默认行为 | Triton 的树，块大小来自启发式或自动调优 | 没有字节契约；在容差范围内与 eager 比较 |
| XLA | HLO pass 把逐元素算子融合成 loop fusion，把归约融合成 input fusion，以及 multi-output fusion；一个 GEMM 改写器把 bias 和激活折叠进 cuBLASLt epilogue | 在某些目标上可能把低精度运算规范化为 f32；代数化简会改写算术 | 由后端决定 | 由代码生成器决定；有用于多次运行间确定性的标志 | 没有字节契约；保证多次运行间的确定性，而不是与未融合程序一致 |
| TVM | 按算子形态融合（injective、归约、把逐元素融合进 dense 层这类可融合输出的算子）；调度来自调优器 | 取决于调度 | 取决于目标 | 由调优器找到的调度选择，所以可能随调优而变化 | 没有字节契约；一个可选的 pass 会换入快速近似 |
| 基于 Triton 的融合器和 kernel 库 | 从 graph 生成 Triton 的编译器，以及手工融合的 Triton kernel 库（norm、旋转位置编码、门控激活） | kernel 怎么算就是什么，通常是 f32 | Triton 的默认值（开启） | Triton 的树 | 通过容差测试确认一致 |
| tilefish（本 RFC） | 在已 pin 的库调用之间，由 DSL 自己的模板把区域下沉为生成的算子 | 声明式：`declared`、`opmath` 或 `f32`，每个都是定义在 IR 上的改写 | 除非声明，否则没有（`tilefish.fma`、`mul-add`） | 区域内有序或与顺序无关；无序归约是单独的 kernel | 与未融合组合逐字节一致：由构造保证、经过测试、并在每次导出时验证；版本 1 中没有不精确的模式 |

区别不在于融合本身，这些系统融合得更激进；区别在于它们每一个都把融合 kernel 的数值语义当作在容差范围内检查的实现细节，而 tilefish 把舍入点、收缩和归约顺序作为声明的一部分，使融合只是 kernel 边界的改变。

## 11. 未来工作：不精确的模式，以及其他扩展

版本 1 只有精确模式（D5）。以下内容是在草案中设计的，保留在这里以便以后的 RFC 以此为起点；其中没有任何一项在计划中。

### 11.1 由二进制文件定义的模式

*由二进制文件定义*的模式，是指它的字节是确定的、被 pin 住的，并且每次回放和两个 host 都能复现（digest 覆盖机器码），但不等于任何未融合组合的字节：添加 barrier 或改变分组可能改变它们，参考实现只在测得的容差内与之一致，融合报告会标记每个这样的区域。这就是如今每个带 Triton 归约或默认收缩的 `@tilefish.kernel` 所具有的契约。

* `fp_contract="compiler"`：`enable_fp_fusion=True`，于是 LLVM 和 ptxas 会收缩它们看到的东西。它们看到什么取决于区域的形状，所以字节取决于分组。这是迁移过来的、依赖收缩的 `@tilefish.kernel` 所需要的。
* `reductions="tree"`：Triton 树形归约加入区域，每个行区域的 warp 规则成为其数值语义的一部分。E1 测量在上下文中树改变的频率（隐患 H4）。
* 这类模式需要的测试：多次运行间的确定性，`jit` 等于 `replay`，Rust 等于 Python，与参考实现的差异在容差内，以及每个区域一条报告条目；`verify` 会对照一个已 pin 的早先导出而不是 `U` 进行比较。

### 11.2 库 epilogue（路线 (b)）

由 RFC 0003 继续，它测量了这些 epilogue（它的第 6 节）。

* **cuBLASLt epilogue**（bias、ReLU、GELU 及其带辅助输出的变体，作为残差的 `beta * C`，按张量的 FP8 scale 和绝对最大值输出）像 GEMM 一样被 pin 住：64 字节的算法、cuBLASLt 版本、SM 数、一个 canary。它们相对 `U` 的精确性只能靠测量：库没有 IR，它的 epilogue 的舍入也没有文档说明与任何 tilefish 算子一致。已有的 pin 显示了这一风险：带 split-K（K 维切分）和 f32 部分和的 bias epilogue 在一种架构上被测出舍入了两次，这就是为什么在没有选择加入降低精度的部分和时，带 bias 的已 pin GEMM 不使用 split-K。
* 把 graph 算术改写为 epilogue（`epilogues="cublaslt"`）只会应用于舍入点在纸面上一致的形态（`(gemm_f32(x, w) + b.to(f32)).to(bf16)` 就是输出为 bf16 的 `gemm_bias`：一次 f32 求和，一次舍入），并且只在 `export` 针对该 pin 在每个 bucket 上用带种子的输入和特殊值输入测得改写结果等于 `U` 之后才应用（E4 是第一次测量）。改变舍入点的改写（把 `round(acc) + b` 变成 `round(acc + b)`）永远不会自动进行。
* **cuDNN 运行时融合引擎**（卷积后接逐点算子）属于同一类：一个声明的 library op，没有自动改写。

### 11.3 其他扩展（精确，尚未计划）

* **prologue hook**（路线 (c')）：在 extern 加载 GEMM 操作数时对其反量化或归一化；逐元素进行，在路线 (c) 的条件下是精确的，但每个加载该元素的 tile 都要重复这项工作。
* **水平融合**，以及把廉价节点**重计算**进多个区域（一个新的 `version`）。
* 把 **`@tilefish.kernel` 调用内联**进区域（8.2），需选择加入。
* 如果 D2 的代价（E3）被证明很重要，为无序归约提供一个根据 `D` 选择的**规范默认顺序**（精确且可融合，但会让不带 `order` 的 `sum` 在 kernel 函数体中的含义之外多出第二种含义）。

## 附录 A：一个完整示例

2.1 中的 graph，针对 `M` 属于 `{32, 64}`、`F = 3072`、`H = 1024` 进行规划：

```
IR of mlp                                   regions (version 1, default mode)
#0 t = gemm_gu(x, w_gu)        CALL         -
#1 g, u = split(t, F, -1)      VIEW         -
#2 g32 = g.to(f32)             ARITH E(M,F) ┐
#3 s = silu(g32)               ARITH E(M,F) │ region fused_<hash>: elementwise over [M, F]
#4 u32 = u.to(f32)             ARITH E(M,F) │ inputs g, u: strided views of t (strides (2F, 1), offsets 0 and F)
#5 p = s * u32                 ARITH E(M,F) │ output a: [M, F] bf16
#6 a = p.to(bf16)              ARITH E(M,F) ┘
#7 y = gemm_d(a, w_d)          CALL         -
```

plan 的步骤（每个 bucket）：`gemm_gu.M32` 写 `t0`；`fused_<hash>` 读取 `t0` 的两个视图并写 `t1`；`gemm_d.M32` 写 `out0`。未融合组合在两个 GEMM 之间有五个 kernel 和四个 `[M, F]` f32 中间值；融合后的 plan 有一个 kernel，没有中间值。两者逐字节一致：加宽类型转换是精确的，`silu`（note 5 中的指令序列）和乘法在两者中对相同的 f32 值执行相同的指令，没有任何收缩，收窄类型转换在两者中都是 `cvt.rn.bf16x2.f32`，并且融合 kernel 从 `t0` 加载 `g` 和 `u`，而不是加载存储下来的类型转换结果，这不改变任何东西（O1）。融合报告条目：

```json
{"op": "fused_<hash>", "template": "elementwise", "space": ["M", "F"], "exact": true,
 "nodes": [{"index": 2, "op": "cast"}, {"index": 3, "op": "silu"}, {"index": 4, "op": "cast"}, {"index": 5, "op": "mul"}, {"index": 6, "op": "cast"}],
 "inputs": [{"value": "g", "read": "strided"}, {"value": "u", "read": "strided"}],
 "outputs": [{"value": "a", "dtype": "bf16", "buffer": "t1"}],
 "rounding_points": [{"index": 3, "dtype": "f32"}, {"index": 5, "dtype": "f32"}, {"index": 6, "dtype": "bf16"}],
 "rewrites": [], "reductions": [], "contractions": 0}
```
