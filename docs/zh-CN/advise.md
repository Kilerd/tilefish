# 融合 advisor：plan 的哪些步骤可以合成一个 kernel

[English](../advise.md) | **简体中文**

`tilefish-explain` 解释的是单个 kernel（核函数）；它不会告诉你模型的哪些部分合成更少的 kernel 后会更快。advisor（建议器）为整个 plan（计划）做这件事：它按能节省的时间，对一个 graph（计算图）的 plan 中的**融合候选**（由中间张量连接起来的步骤，合成一个 kernel 后就不必存储这些中间张量）排序，说明每个候选为什么现在还不是一个 kernel、该怎么办，并按步骤类别汇总 plan 的时间。一个 profiler 测量每个步骤的设备时间，从而可以按实测时间排序。它还会指出哪些融合模式帮助最大，这正是 plan 融合第 2 阶段（`docs/rfcs/0001-plan-fusion.md`）应当优先做的。

代码在 `tilefish._advise`（分析）和 `tilefish._profile`（profiler）。

```console
tilefish-explain DIR --graph G --advise                     # every bucket analysed, ranked at the one that moves the most bytes
tilefish-explain DIR --graph G --advise --bucket M=128      # one bucket
tilefish-explain DIR --graph G --advise --timing t.json     # ranked by measured times (the timing file says the bucket)
  --machine sm_120 | --bandwidth 1600                       # another roofline (default: the plan's architecture), a measured bandwidth
  --keep t7                                                 # a buffer you tap: it stays stored (repeatable)
  --top 40 | --json                                         # how many candidates the text lists; the whole advice as JSON
tilefish-explain DIR --graph G --profile t.json --load models.block:graph --bucket M=128 --devscalar n=77 --pin DIGEST
  --reps 50 --warmup 5 --method graph|events                # with --advise too: profile, then advise with what was measured
```

```python
adv = tilefish.advise("build/block", "block", bucket="M=128", timing="t.json")   # an Advice: adv.text(), adv.doc (JSON), adv.candidates
t = tilefish.profile_plan(block, "build/block", pin=p.digest, bucket="M=128", devscalars={"n": 77}, reps=50, out="t.json")
```

**为什么是 `tilefish-explain` 的一个选项，而不是一个独立的工具。** advisor 读取的就是 `tilefish-explain` 读取的东西（一个导出目录：manifest（清单）、plan 或 plan set（plan 集）、融合报告、trace（追踪记录）及其校验），接受同样的 `--graph`、`--bucket` 和 `--json`，回答的是用户向它提出的问题（“哪些部分可以融合以变得更快”）。“这次导出做了什么、怎样才能更快”只用一个命令，比用两个命令更容易找到；代码则是一个独立的模块。

## 1. 它读取什么

全部来自导出目录，不需要 GPU，也不需要任何 CUDA 工具：

* **plan**（`plan.json`，或 plan set 中的 `<graph>.plan.json`）：每个 bucket（分桶）的步骤，每个步骤绑定的缓冲区和视图，缓冲区的角色（input、weight、output、intermediate、carried、inout、workspace）；
* **manifest**：每个步骤启动什么。library 条目的尺寸（GEMM（通用矩阵乘）的 `m`、`n`、`k`；卷积的 `n`、`c`、`k`、输出和滤波器尺寸以及分组数）和数据类型；kernel 条目的 ABI（每个指针的访问方式：`in`、`out`、`inout`），它是否由 tilefish 生成（`source_sha256`）、用的是哪个模板（行模板下沉（lowering）生成的 kernel 有一个 `BLOCK_D` 常量），以及它的选项（`enable_fp_fusion`）；
* **融合报告**（`<graph>.fusion.json`）：graph 函数体的各个区域，每个区域包含的算子、它的模板、它如何读取输入（由 `tilefish.contiguous` `materialised` 的视图）、它的归约及其顺序；
* **trace**，在导出带有 trace 时（`trace/graphs/<graph>.json`，`docs/explain.md`）：每个区域的算子和源码行，以及新增的各区域**边界**及其原因（第 4 节），还有每个 kernel 的 SASS，advisor 据此判断一个手写 kernel 是否计算 MMA。属于另一个 plan 的 trace 会被忽略。没有 trace 时，`tilefish.advise(..., gop=graph)` 从 graph 计算边界。

## 2. 每个步骤

在每个 bucket 下，对每个步骤：

* **读写字节数**：每个参数的元素数（其视图的形状，或缓冲区在该 bucket 下的形状）乘以其数据类型的大小；指针为 `in` 或 `inout` 时计为读，为 `out` 或 `inout` 时计为写（library 步骤读取其操作数并写 `out`；copy 读 `src` 写 `dst`；memset 写 `dst`）。同一缓冲区的两个视图计两次。这是一个步骤至少要搬运的量：多次读取操作数的 GEMM 搬运得更多。
* library 步骤的 **FLOPs**：GEMM 为 `2 m n k`，卷积为 `2 n k prod(output) (c / groups) prod(filter)`。kernel 的 FLOPs 未知。
* **类别**：

  | 类别 | 含义 |
  |---|---|
  | `library` | 已 pin（固定）的 cuBLASLt GEMM 或 cuDNN 卷积 |
  | `fused` | graph 函数体中的一个区域（`fused_<hash>`）：逐元素、行，或单独的一个归约 |
  | `elementwise`、`row` | 用对应模板下沉的 `@tilefish.kernel` |
  | `copy` | copy 或 memset 步骤，或生成的 `tilefish.contiguous`、`im2col`、`im2col1d`、`col2im1d` |
  | `opaque` | 手写的 Triton 或 Gluon extern（外部 kernel），或预编译二进制文件（有 trace 时：区分有无 MMA）；带 epilogue（尾部）hook 的 extern（标签中写出它的 hook）、它的累加器形式（`<extern>__acc`），以及 hook 中带有区域的 extern（`<extern>_epi_<hash>`，子类 `hooked`，标签中写出契约） |

* **roofline**：library 步骤的每字节 FLOPs 低于机器平衡点（峰值除以带宽）时为访存受限，高于时为计算受限；elementwise、row、fused 或 copy 步骤为访存受限；opaque 步骤未知（没有 MMA 的 kernel：访存受限）。其时间估计为 `max(bytes / bandwidth, FLOPs / peak)`，是一个下界。

**机器表**（`tilefish._advise.MACHINES`）：每个架构选一款产品，取厂商公布的数字，稠密算力，已取整。

| 架构 | 产品 | 带宽（GB/s） | f16 / bf16 MMA，f32 累加器（TFLOPS） | TF32 | f32 FMA（乘加融合） | SM（流式多处理器）数 | 平衡点（FLOP/B） |
|---|---|---|---|---|---|---|---|
| sm_90 | H100 SXM | 3350 | 989 | 495 | 67 | 132 | 295 |
| sm_100 | B200 | 8000 | 2250 | 1100 | 80 | 148 | 281 |
| sm_103 | B300 | 8000 | 2250 | 1100 | 75 | 148 | 281 |
| sm_120 | RTX 5090 | 1792 | 209.5 | 104.8 | 104.8 | 170 | 117 |

一个 SM 分区（MIG 切片、green context）只占其中一部分。计时文件记录它运行时所在设备的 SM 数，以及它测得的设备到设备拷贝带宽；advisor 随后使用这个带宽，并按 SM 数缩放峰值（B300 的一个 30 SM 切片测得 1614 GB/s：456 TFLOPS，平衡点 283）。`--bandwidth` 可以手动设置带宽。按字节数的排序不依赖于这张表；微秒数以及访存受限还是计算受限则依赖于它。

## 3. 生产者与消费者

每个 bucket 中的步骤都相同（只有尺寸不同），所以谁写什么、谁读什么的分析只做一次，字节数则按 bucket 分别计算。对缓冲区的每次写入都开启它的一个新版本；读取某个版本的步骤是该版本的消费者。一个版本在满足以下条件时是**可融合的中间张量**：

* 它的缓冲区是 `intermediate`（不是输入、权重、返回值、carried buffer（携带缓冲区）、`InOut` 参数、设备标量或 workspace）；
* 该缓冲区在 plan 中只写一次（没有被 `InOut` 指针原地更新，也不是覆盖写在其输入上的结果 `MayAlias`）；
* 除写入它的步骤外，恰好只有一个步骤读取它；
* 它不是你要保留的缓冲区（`--keep`、`keep=`：一个 tap（观测点））。

把生产者和消费者合成一个 kernel 就不必存储它：这省掉了生产者对它的写和消费者对它的读（绑定到它的每个参数的字节数）。由可融合中间张量连接起来的步骤构成一个**候选**：

* 由非 library 步骤构成的**一对**、**一条链**（`i -> j -> k`）或**一组**（一个步骤读取两个可融合中间张量，或写出两个）：把它们全部合成一个 kernel，用一次启动代替多次；
* **epilogue**（一个 library 步骤，以及唯一读取其结果的步骤）或 **prologue**（头部；其结果只被一个 library 步骤读取的步骤）：library op（库算子）从不与另一个 library op 融合，而它的 epilogue 和 prologue 是将来的工作，所以每个都是一个独立的候选，以该 library op 为界。

**它们之间不能夹着任何步骤**（RFC 0001 第 3.5 节规则 2）：由一个候选的步骤合成的 kernel 不能依赖候选之外的步骤，即一个依赖于某个成员、又被某个成员依赖的步骤。候选从最大的中间张量开始往下生长；如果合并某个中间张量会在中间夹进这样的步骤，它就先等待，等其他中间张量加入各自的组之后再试（那个步骤可能已经加入了某个组），直到不再有变化。仍然夹着步骤的中间张量不计入节省：它列在 `blocked` 下，附上夹在中间的步骤，归为 **`library-in-the-middle`**（它们之间有 GEMM 或卷积：transformer 的残差流由一个 norm 写入、由下一个 norm 读取，中间隔着该层的 GEMM；将来的工作）或 **`step-in-the-middle`**（另一个无法加入它们的 kernel）；位于两个 library op 之间的中间张量列为 **`library-library`**。依赖关系取自 plan：一个步骤读取另一个步骤写出的版本，以及对同一缓冲区访问的先后顺序。

被多个步骤读取的版本是一个 **fan-out**，会被列出，但不是候选：它需要把所有读取者合成一个区域，或者重新计算，而第 1 版不做这些。被保留存储的缓冲区会连同原因一起列出。

## 4. 为什么现在还不是一个 kernel，以及该怎么办

每个候选都列出它的**阻碍项**，每个阻碍项附有消除它的操作，以及能消除它的 RFC 0001 阶段；候选的阶段取其中最晚的。步骤的类别给出一个阻碍项；其余的由融合 pass 给出。阶段 `E2` 不属于 RFC 0001：它是现在就可以声明的一种模式，即 RFC 0003 的 cuBLASLt epilogue，它会改变字节（所以 advisor 会提到它，但从不替你选择它）；它排在 `1` 之后、`2` 之前。

| 阻碍项 | 怎么办 | 阶段 |
|---|---|---|
| `opaque-extern` | 没有 MMA 的手写 kernel：把它的运算写进 graph 函数体，让融合接管它（逐元素运算，或在求和上带 `order=` 的行运算） | 1 |
| `opaque-unknown` | 同上，适用于没有 trace 可判断它是否计算 MMA 的情况（如果它计算 MMA：在它产生中间张量的地方用 epilogue hook，在它读取中间张量的地方用 prologue hook） | 1 |
| `opaque-compute-epilogue`、`-prologue` | 受自身运算限制的手写 kernel：SASS 中有 MMA，或者（有计时文件时）搬运字节的速度不到带宽的四分之一，例如由 FMA 构成的 attention 或一次搜索；在 extern 上声明 epilogue hook（`tilefish.hook`，RFC 0003 E1）；prologue hook | 4，将来 |
| `epilogue-hook-consumer` | 生产者是带 epilogue hook 的 extern，消费者不是 graph 函数体中的运算：把消费者的运算写进 graph 函数体，由 hook 接管（通过 `value="out"` 得到契约 (i)，通过 `value="acc"` 得到契约 (ii)） | 4 |
| `dsl-kernel` | 一个（逐元素的）`@tilefish.kernel`：把它的函数体写进 graph 函数体（RFC 8.2）；它声明了 `fp_fusion=False` 时，照原样写就是精确（exact）的 | 1 |
| `dsl-kernel-contracts` | 同上，适用于允许编译器做收缩的 kernel：在本意是一次舍入的地方写 `tilefish.fma`，否则它的字节会改变 | 1 |
| `row-kernel` | 一个行 `@tilefish.kernel`：把它的函数体写进 graph 函数体，并在求和上加 `order=`，融合第 2 版（默认）会把它融合进一个行区域 | 2 |
| `epilogue-library-cublaslt` | `cublaslt.gemm`（或 `gemm_bias`）之后的区域是 cuBLASLt 的某个 epilogue，如融合报告的 `library_epilogues` 所示（bias、ReLU、残差加法，按库的顺序）：声明 `tilefish.fusion(epilogues="library")`，把它们合成一个 `cublaslt.matmul` 步骤，其字节是 pin 下库给出的字节，并在每次 pin 时验证与该区域在 f32 累加器上的运算相等（RFC 0003 的契约 (ii)：比现在少一次舍入）；详情中会写明 epilogue 和契约 | E2 |
| `epilogue-library`、`prologue-library` | `epilogues="library"` 接不了的已 pin GEMM 或卷积的 epilogue（详情说明原因：加法的顺序、在别处读取的值……），或者卷积的 epilogue：RFC 0003 的 E3（带 hook 的生成 GEMM 或卷积）或 E6（cuDNN fused graph）；尚未实现。目前：改为调用带 epilogue hook 的 extern（E1）。prologue：将来的工作 | 将来 |
| `copy-step` | copy 步骤：在值产生的地方直接返回或存储它 | 1 |
| `memset-step` | memset 步骤：它的零值是从内存读取的，从不折叠进区域（H2） | 将来 |
| `strided-output` | 生产者把中间张量写进一个缓冲区的一部分（一个列块、一个 head）：带步长的存储（规则 7） | 2 |
| `materialised` | 视图的 `tilefish.contiguous` 拷贝，或通过非稠密视图读取中间张量的行步骤（规则 5；融合第 2 版的行区域会原地读取它） | 2 |
| `generated-movement` | 卷积前后的 `im2col` / `col2im`：它的 prologue 或 epilogue，将来的工作 | 将来 |
| `unknown-boundary` | 两个区域，没有 trace 或 graph 来说明原因 | 1 |

**没有被融合的 graph 函数体运算。** 对含有运算的 graph，融合 pass 会说明它的区域为什么划在那里（`tilefish._fusion.boundaries`，以 `boundaries` 写入 trace；它重复 pass 自己的规则 `merge_problem`，不改变任何区域）：每个跨越边界的值一个条目，附上两侧的算子及其行、原因和修复方法：

| 原因 | 为什么 | 修复 | 阶段 |
|---|---|---|---|
| `barrier` | 该值经过了 `tilefish.barrier` | 去掉它：在未融合的 plan（`grouping="none"`）上用 tap 能看到同样的字节 | 1 |
| `reduction-unordered` | 没有 `order=` 的 `sum` / `mean` 在每个融合版本中都单独运行（D2） | 加上 `order=`：融合第 2 版（默认）会融合它 | 2 |
| `reduction` | graph 声明了 `tilefish.fusion(version=1)`：有序归约、`max`、`argmax` 各自单独成为一个区域（经过其中之一形成的环也是） | 第 2 版（默认）把它们融合进行区域 | 2 |
| `gather` | graph 声明了 `tilefish.fusion(version=1)`：`gather_rows` 单独成为一个区域 | 第 2 版会融合它 | 2 |
| `view` | 两个算子之间的视图会结束一个区域（规则 6） | 在同一个形状上计算（区域内的稠密 reshape：第 2 阶段的工作，尚未排期） | 2 |
| `materialised` | 第 1 版的行区域读取的视图会先由 `tilefish.contiguous` 拷贝 | 第 2 版在行模板中原地读取 1 维和 2 维的视图 | 2 |
| `template` | 模板无法在一个 kernel 中同时计算两者：迭代空间不同（另一种长度的行、沿另一个轴的归约、3 维的行），或者在第 1 版中是行区域 | 如果是行区域，用第 2 版；否则在同一个形状上计算 | 2 |
| `cycle` | 把两者合成一个区域会形成经过一次调用或一次搬运的环 | 移动该调用 | 1 |
| `limit` | 一个区域中超过 64 个算子或 32 个张量 | 拆分仅是速度上的取舍 | 1 |
| `move` | 它们之间有一次拷贝或一次存储 | 在值产生的地方直接返回或存储它 | 1 |
| `epilogue-<kind>`、`prologue-<kind>` | 区域读取一次调用的结果，或一次调用读取区域的值：不跨调用融合（extern：它没有声明 epilogue hook） | library：RFC 0003 E2、E3 或 E6；extern：声明一个 hook（`tilefish.hook`，E1）；`@tilefish.kernel`：把它的函数体写进 graph 函数体 | 将来、4、1 |
| `epilogue-acc-only` | extern 只对其累加器设置了 hook，而该区域不处于 `"accumulator"` 模式（graph 或其作用域是 `"exact"` 或 `"library"`） | 声明 `epilogues="accumulator"`（在 graph 或 `tilefish.fused` 作用域上；契约 (ii)），或者给 kernel 一个 `value="out"` hook（契约 (i)） | 4 |
| `epilogue-taken` | 该调用的 hook 已经接管了另一个区域（每次调用一个），或者该区域已融合进另一次调用的 hook（每个区域一次调用） | 把两者写在一个区域里 | 4 |
| `epilogue-view` | 该区域通过 hook 不持有的视图读取结果 | 直接读取结果本身（或拆分 hook 的各个分块） | 4 |
| `epilogue-acc-shared` | f32 结果（`R_acc`）不止被该区域读取 | 只在一个区域中读取它 | 4 |
| `epilogue-cycle` | 从结果到该区域的路径上有一次调用（或一个区域，或另一对已融合的组合） | 把它从两者之间移出去 | 4 |
| `epilogue-inout` | 该区域在更新之前或之后读取了 extern 原地更新的张量（`InOut`） | 在该区域之外读取它 | 4 |
| `epilogue-row` | 该区域对行做归约，而 extern 的一个 tile 装不下一整行（`block_n`），或者该 kernel 是 Gluon kernel | 用更宽的 tile；跨 tile 拆分是 E4 | 4 |
| `epilogue-template`、`epilogue-hook`、`epilogue-dims` | 该区域无法下沉为 hook，hook 无法接管它，或者它的尺寸与 extern 的不同 | 见详情 | 4 |
| `grouping` | 合并是合法的，但该 graph 所用融合版本的贪心合并没有做出它 | 以后的融合版本 | 1 |
| `unfused` | graph 声明了 `tilefish.fusion(grouping="none")`：每个算子一个 kernel | 去掉它（默认会把它们分组） | 1 |

没有 trace 也没有 graph 时，融合报告会说明哪些区域是单独的归约（以及它是否有顺序），哪些读取的是先被拷贝过的视图；此时由中间张量连接的两个区域为 `unknown-boundary`。什么会单独成为一个区域取决于 graph 的融合版本，报告会记录它（`fusion.version`）：第 2 版中只有不带 `order=` 的归约，第 1 版中是每个归约和每个 gather。

## 5. 排序

每个候选的节省在每个被分析的 bucket 下都会计算，但只在一个 bucket 下排序：计时文件的 bucket、`--bucket` 指定的 bucket，或者其步骤搬运字节最多的 bucket。

* **没有计时文件时**：节省的字节数除以带宽，`S / BW`，计入除计算受限的 library 步骤之外每个步骤的字节（计算受限的 GEMM 不会因为少写一些就变快）。它省掉的启动只列出，不计入。
* **有计时文件时**（步骤的*净*时间，第 6 节）：**启动下限** `t0` 是最短的 kernel 步骤，即一个 kernel 无论搬运多少都要付出的开销。一个步骤用 `t - t0` 以其实际带宽 `min(BW, bytes / (t - t0))` 搬运字节。如果这段时间可以就是它搬运字节的时间（elementwise、row、fused 或 copy 步骤、GEMM、没有 MMA 的 kernel），并且实际带宽达到带宽的四分之一或以上（`STREAMING`），它节省 `min(t - t0, S_step / achieved)`；其他步骤（attention、MMA 情况未知的 kernel、远慢于其字节量的 kernel）节省 `S_step / BW`，计算受限的 GEMM 不节省。融合每省掉一个 kernel 就节省 `t0`：省掉的 kernel 数比候选的 kernel 步骤数少一个，或者在 library op 吸收它们时全部省掉。节省最多为候选的时间减去一个下限。候选之间会重叠（两个 GEMM 之间的 elementwise 步骤既是一个的 epilogue，又是另一个的 prologue），所以它们的节省不能相加。

**epilogue 的路径**（`docs/rfcs/0003-epilogue-fusion.md` 第 5 节）。生产者为 GEMM 或卷积的候选带有 `routes`：每条路径是把消费者的运算并入生产者的一种方式，附上它给出的契约、它的状态和节省（即该候选在排序所用 bucket 下的节省）。对已 pin 的 cuBLASLt GEMM：E2（库 epilogue，契约 (iii)：在融合报告把该区域匹配为 cuBLASLt 的某个 epilogue 时可用，否则说明原因）、E3（带 hook 的生成 GEMM，实测与 pin 相等时为契约 (i)，否则为 (ii)）及其**盈亏平衡点**（break-even）`1 + saving / T_gemm`，即生成 GEMM 慢到什么程度时融合就不再划算（advisor 不预测生成 GEMM 的速度），以及 E1（你自己写的带 epilogue hook 的 extern，现在可用）；对 cuDNN 卷积，用 E6（fused graph）代替 E2。对带 epilogue hook 的 extern：它的 `value="out"` hook（契约 (i)，可用）和 `value="acc"` hook（契约 (ii)，在 `epilogues="accumulator"` 下可用，否则说明要声明什么）；对受自身运算限制、没有声明 hook 的 extern，给出要声明的 hook。给出不同字节的路径永远不会被 advisor 或 planner 按时间自动选择：实测时间只为声明提供参考。

`patterns` 按每个候选需要消除的最晚阶段的阻碍项汇总候选（每个候选只计一次；它需要的其他阻碍项列在 `also` 下）：即该融合阶段能解锁什么。最晚阻碍项为 `epilogue-library` 的候选，无论做了其他什么，都需要 GEMM epilogue。`recurring` 汇总那些步骤以相同顺序运行相同算子的候选（模型中重复的一层）：即用模型的语言来说，节省最多的融合模式。

**按类别的时间**按类别汇总步骤的时间（实测值和 roofline 估计）：`library`、`fused`、`elementwise`、`row`、`copy`、`opaque`；**按算子的时间**按 manifest 算子汇总。

**JSON**（`--json`、`Advice.doc`、`tilefish.advice.v1`）：`graph`、`arch`、`plan_digest`、`bucket`（排序所用）、`buckets`（被分析的）、`machine`、`timing`、`estimate`、`steps`（每个步骤：`class`、`label`、`bytes_read`、`bytes_written`、`flops`、`intensity`、`bound`、`est_us`，有实测时还有 `us` 和 `achieved_GBps`）、`by_class`、`by_op`、`candidates`（每个候选：`shape`、`steps`、`labels`、`classes`、`buffers`、按 bucket 的 `saved_bytes` 和 `saved_us`、`launches_removed`、`blockers`、`why`、`actions`、`phase`、`gate`、`rank`，对 GEMM 或卷积的 epilogue 还有 `routes`：每条路径的 `route`、`milestone`、`contract`、`status`、`saved_us`，hook 的 `value`，E3 的 `break_even`）、`patterns`、`recurring`、`blocked`、`fan_outs`、`kept`、`unfused_arithmetic`。与融合报告一样仅供参考：没有 host（宿主）读取它。

## 6. profiler

`tilefish.profile_plan(graph, plan, ...)` 像 `tilefish.run` 那样在一个 bucket 下运行 graph 的 plan（同样经过验证的 plan 和二进制文件，同样的 arena（内存池）及其顺序），并在设备上为每个步骤计时。和每个 Python host 一样，它需要 graph（plan 要与之核对），所以命令行会导入它（`--load MODULE:ATTR`）。除非 `externals=` 给出，否则输入和权重是随机的（只有在 kernel 按值分支时，时间才取决于值）；每个设备标量都必须给出（`devscalars=`、`--devscalar`）：一个计数会改变步骤的行为。

* **`method="graph"`**（默认）：把步骤一次性 capture（捕获）进一个 CUDA graph（CUDA 图），在第一个步骤之前和每个步骤之后各放一个事件记录节点（`CU_EVENT_RECORD_EXTERNAL`），在 `warmup` 次之后回放 `reps` 次：这正是 capture plan 的 host 运行它的方式。一个步骤的时间就是它前后两个节点之间的时间。plan 还会在不带事件的情况下再 capture 一次，并作为整体计时。
* **`method="events"`**：像一次运行那样启动各步骤，在 stream 上于第一个步骤之前和每个步骤之后记录事件；先用一个自旋 kernel 在 host 入队这些步骤期间让设备保持忙碌，使它们背靠背运行，而不是以 host 启动它们的速度运行。对 `n` 个步骤，自旋最初为 `20 n + 50` 微秒，每次运行都会检查它：自旋之前的一个事件在设备上为它计时；如果某次运行中 host 入队步骤所用的时间比设备自旋的时间还长，就用更长的自旋重做（host 每个步骤可能要花 30 us，而在 RTX 5090 上，自旋的实际时长比按时钟频率算出的短约 15%）。即使自旋 2 秒 host 仍跟不上的，会被拒绝：请使用 `method="graph"`。适用于最多约一千个步骤的 plan（驱动的队列）。

**计时文件**（`tilefish.step-timing.v1`）：`graph`、`plan_digest`、`manifest_digest`、`arch`、`bucket`、`method`、`reps`、`warmup`、`device`（`name`、`arch`、`sm_count`、`clock_khz`、`copy_GBps`）、`steps`（每个步骤：`index`、`kind`、`entry`、`us`：`median`、`mean`、`trimmed`（中间 80% 的均值）、`min`、`p10`、`p90`、`net`）、`sum_us`（各截尾均值之和）、`plan_us`（不带事件的 plan）、`event_us` 和 `overhead`（`sum_us / plan_us - 1`）。advisor 会拒绝：没有 `plan_digest` 或属于另一个 plan、graph 或 bucket 的文件，没有为该 bucket 的每个步骤恰好计时一次的文件，不是有限微秒数的时间，以及不为正数的设备数字。

**它会改变什么：不会留下任何比它活得更久的改变。** 它是一条独立的运行时路径：没有任何 manifest、plan、plan set 或 digest（摘要）引用它，它不写导出中的任何文件，它所做的运行都是该 plan 的运行。这些运行中会比它们活得更久的改变，即调用方的 `InOut` 张量和 plan 写入的 carried buffer，会在第一次运行之前（按 arena 的顺序）复制到一边，并在最后一次运行之后复制回去，arena 对每个 carried buffer 记录的信息也与之前相同：profile 之后的一次运行写出的字节与 profile 之前的一次运行相同，看到的状态也与本来会看到的相同（`test_gpu_advise.py`）。先读后写 carried buffer 的 plan 会被拒绝，正如单独运行它也会被拒绝一样（它要在一个 shared arena（共享内存池）上、在写入该缓冲区的 plan 之后运行，而 profiler 不会搭建这样的环境）。

**开销与准确度**（在一个 B300 MIG 切片上实测，sm_103，30 个 SM，使用 `python/tests` 中 `plan_model` 的 plan，并在同样的 graph replay（回放）上用 Nsight Systems 2026.5.2 对照，`--cuda-graph-trace=node`）：

* **在这台设备上，事件让每个步骤多花约 3 us**（在更大的 plan 上为 3 到 4.5 us）：`plan_model.block` 在 `M = 128` 下的 9 个步骤，事件之间的时间合计 94 us，而 plan 单独运行只要 64 us，两种方法都是如此。事件节点要等它之前的步骤完成，而 graph 中相邻的 kernel 本来会让前一个的尾部与下一个的启动重叠。所以文件会记录这个开销（`event_us`：各步骤之和减去 plan 的时间，平均分摊），以及每个步骤的 **`net`**：它的截尾均值减去这个开销。各 net 相加等于 plan 的时间，advisor 按它们排序。
* **时间戳按固定步长跳动**（短步骤的中位数是 4.096 和 6.144 us），中位数会落在这些刻度上，而均值会把它们平均掉：所以用截尾均值。
* **与 nsys 对照**（在 `nsys profile --trace=cuda --cuda-graph-trace=node` 下，对 plan 的 CUDA graph 做 200 次不带事件的 replay；profiler 做 200 次带事件的 replay；单位为微秒）：

  | `plan_model.block` 的步骤，`M = 128` | profiler 中位数 | profiler net | nsys kernel | nsys kernel + 其前的间隙 |
  |---|---|---|---|---|
  | `rmsnorm` | 4.70 | 1.72 | 1.35 | 1.35 |
  | `gemm_qkv`（cuBLASLt） | 6.18 | 2.91 | 2.50 | 2.54 |
  | `attention` | 49.12 | 46.00 | 46.52 | 46.54 |
  | `gemm_o` | 5.98 | 2.74 | 2.22 | 2.25 |
  | `add_rmsnorm` | 4.29 | 1.28 | 1.20 | 1.23 |
  | `gemm_up` | 6.14 | 2.88 | 2.52 | 2.56 |
  | `gelu` | 6.14 | 2.87 | 1.65 | 1.68 |
  | `gemm_down` | 6.14 | 2.88 | 3.04 | 3.08 |
  | `add` | 4.10 | 0.83 | 1.08 | 1.09 |
  | **合计** | 92.8 | **64.1** | 62.1 | **62.3**（第一个 kernel 到最后一个：62.3） |

  profiler 不带事件单独为 plan 计时：64.1 us。对 `plan_model.deep_block(4)`（36 个步骤），各 net 合计 244.2 us，nsys 的 kernel 加间隙合计 249.4 us。**除一个步骤（1.2 us）外，每个步骤的 net 与 nsys 的 kernel 时间相差不超过 0.6 us，各 net 之和与 nsys 的相差不超过 3%**；包含事件的中位数每个都长了 3 到 4 us。短步骤请按 `net` 排序，两个步骤之间不到一微秒的差异请视为噪声。
* **在 RTX 5090 上**（sm_120，170 个 SM；同样的对照，nsys 2026.5.2）：事件让每个步骤多花约 1.1 us，时间戳以 2.048 us 为步长跳动。`plan_model.block` 在 `M = 128` 下的各 net 合计 31.6 us，nsys 的 kernel 加间隙合计 26.0 us（profiler 自己测得的不带事件的 plan 时间也是 31.6 us：比 nsys 从第一个到最后一个 kernel 的时间多约 5 us，这是一个固定开销，对这么短的 plan 影响很大）；对 `plan_model.deep_block(4)`，是 110.4 us 对 104.6 us。除一个步骤（1.9 us）外，每个步骤的 net 与 nsys 的 kernel 加间隙相差不超过 1.1 us。events 方法的各 net 合计分别为 30.7 和 114.7 us。
* **它看不到的**：步骤内部的工作（由多个 kernel 组成的 library 步骤算一个步骤），以及一个步骤与下一个步骤的重叠，`net` 是把它平均分摊而不是测量出来的。要测 kernel 内部的时间：用 probe（探针）和插桩变体，见 `docs/instrument.md`。

## 7. 如何测试

* `python/tests/test_advise.py`（任何机器）：合成的 plan（字节、FLOPs、类别和 roofline；一对步骤，GEMM 前后的 prologue 和 epilogue；链、组和 fan-out；被保留的返回值、carried、`InOut`、被 tap 的和写了两次的缓冲区；中间夹着 GEMM 或另一个 kernel；每个 bucket 的条目和视图；同一缓冲区的多个视图；用 trace 的 SASS 区分有 MMA 和没有 MMA 的 kernel；受自身运算限制的 kernel；按字节和按实测时间的排序、下限和实际带宽；属于另一个 plan、bucket 或步骤的计时文件被拒绝）；命令行上的 golden plan `block`（它的残差被 MLP 的 GEMM 阻挡）；graph 函数体的边界（barrier、无序和有序求和、调用、被 materialise 的视图）及其修复方法，以及它们不改变任何区域；假编译器做的带 trace 导出（它的 manifest、plan 和融合报告与不带 trace 时逐字节一致；trace 中的边界；带 trace、不带 trace 和带 `gop=` 时的建议）；profiler 在假后端上的运行时路径（每个步骤前后的事件、与一次运行相同的启动、两种方法、为慢 host 加长自旋）。`python/tests/test_epilogue.py`：紧挨着 kernel 步骤的带 epilogue hook 的 extern（一个可操作的候选，其 hook 的路径和契约），带和不带融合报告；hook 中带有区域的 extern 的步骤；GEMM epilogue 的各条路径（E1、E2、E3 及其盈亏平衡点）；hook 给出的每个原因都有修复方法。
* `python/tests/test_gpu_advise.py`（需要设备；与架构无关：已在 sm_103 和 sm_120 上运行，sm_90 的运行属于之后的门禁）：每个步骤都被计时，profile 不改变导出的文件和一次运行的字节，各 net 相加等于 plan 的时间，两种方法结果一致，命令行（先 `--profile`，再 `--advise --timing`）。
