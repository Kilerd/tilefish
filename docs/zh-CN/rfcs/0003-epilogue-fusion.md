# RFC 0003：epilogue 与 prologue 融合

[English](../../rfcs/0003-epilogue-fusion.md) | **简体中文**

* **状态：** 已评审；2026-11-16 的决定记录在下文“决定”一节。设计和阶段 0 实验（第 6 节）。里程碑 E1（extern（外部 kernel）上的 epilogue（尾部）hook（钩子），见下文“E1：实现说明”）和 E2（cuBLASLt epilogue，见 7.3 节：构建了什么、测量了什么、还剩下什么）已经实现，并能在同一个 graph（计算图）中协同工作（见下文“E1 与 E2 结合”）；实验 X8（一个带这些 hook 的持久化、warp（线程束）特化的 GEMM（通用矩阵乘））已经测量（见下文“X8：结果”；该 kernel（核函数）是 `python/examples/` 下的一个示例，不是受支持的功能）；其余部分是设计。
* **范围：** 跨调用的融合。*epilogue* 是读取 GEMM 或卷积结果的逐元素或行级工作；*prologue*（头部）生成其某个操作数。本 RFC 涉及规划器和融合 pass（`tilefish._plan`、`tilefish._fusion`）、library op（库算子）（`cublaslt.*`、`cudnn.conv_fwd`）、extern、一族新的生成的 GEMM 和卷积算子，以及 advisor（建议器）。只有库 epilogue 涉及 manifest（清单）和 host（宿主）。
* **基线：** tilefish 0.0.10 / crate 0.0.8，其中包含融合 advisor（`docs/advise.md`）。
* **与 RFC 0001 的关系。** RFC 0001（plan（计划）融合）融合各调用之间的 graph 函数体算术，并承诺融合本身绝不改变字节。它的第 7 节勾勒了进入调用的三条路线，第 11.2 节推迟了库 epilogue。本 RFC 取代这些草案。它是一份独立的文档，而不是 0001 的一节，原因有二。RFC 0001 已经评审，它的决定 D1 到 D7 建立在一个只有精确（exact）模式的契约之上。本 RFC 必须引入第一批字节*不*等于未融合组合（unfused composition）字节的契约，并且需要它自己的决定。RFC 0001 的每个决定都仍然成立。本 RFC 在 0001 某个阶段之上构建的地方，会说明是哪个阶段。

## 概要

融合 advisor 在一个 B300 MIG 切片上测量了三个已移植到 tilefish 的模型：两个 transformer 堆栈，以及一个卷积编码器和解码器。几乎每个可以融合的字节都紧挨着一个 GEMM 或卷积：

* 需要 GEMM 或卷积 epilogue 的候选，加起来占每个 plan 测得的 plan 时间的 3.6 到 13.3%，除一个由 attention kernel 主导的 plan 之外，其余每个 plan 上都是 6 到 13%；
* 卷积模型的 prologue 另外加起来占 10.9%，与 epilogue 有重叠；
* RFC 0001 第 1、2 阶段能够覆盖的候选加起来不到 0.2%。

所以，划算的融合是融合进计算算子。

这种融合与 tilefish 的承诺相冲突。如今，一个已 pin（固定）的 cuBLASLt 或 cuDNN 算子写出的字节与参考栈写出的字节相同（相同的库版本、相同的算法），其后的 kernel 读取的是这些*已舍入*的字节。融合的 epilogue 看到的则是 f32 累加器。如果它使用累加器，就少了一次舍入，字节随之改变：X1 测得，加在累加器上的 bias 与加在已舍入输出上的 bias，在约 30% 的元素上不同。如果它先舍入，那么除非融合 kernel 内部的 GEMM 算出与库的 GEMM 相同的位，字节仍然会变。因此本 RFC 定义三种契约，并把它们区分开：

* **(i) 精确。** 融合后的 plan 写出未融合组合的字节，GEMM 的位也相同。有四类融合遵守这一契约：
  * 在一个本身就是参考的 kernel 的舍入之后的 hook；
  * 进入一个保持不变的 kernel 的任何 prologue；
  * 一个经测量与它所替代的库 GEMM 相等的生成 GEMM；
  * 一个在组合舍入之处舍入的库融合。

  两个实验表明后两者确实存在。X3 发现，在 sm_103 上，只要库的 pin 不切分 `k`，一个普通的 Triton `tcgen05` GEMM 写出的位与 cuBLASLt 已 pin 的 GEMM 相同：在每个形状上、在 99 种 tile 配置的每一种上、在两种姿态（posture）下都是如此，操作数中含特殊值时也是如此。X5 发现，带 16 位中间值的 cuDNN 9.13 融合图与未融合的卷积 + bias（+ ReLU）逐位相等。
* **(ii) 累加器。** 用户声明对程序的一个改写：在 epilogue 把 GEMM 或卷积的结果加宽到 f32 的每一处，去掉该结果的舍入。这样，融合后的 plan 相对于*改写后程序的*未融合组合是精确的：融合本身仍然不改变任何字节，改变字节的是声明的模式。结果更快，精度至少一样，并且计入 digest（摘要）、被 pin 住、由每个 host 复现。它没有两 kernel 栈的字节。cuBLASLt 的 bias、ReLU 和 `beta * C` epilogue 经测量在每个元素上都等于这一契约。
* **(iii) 库。** cuBLASLt epilogue 或 cuDNN 融合图在一个 pin 下写出库的字节：算法或引擎、版本、SM（流式多处理器）数和 epilogue。在 epilogue 的算术可以写下来的地方，导出时会测量它等于 (ii)。否则字节由二进制文件定义：cuBLASLt 的 GELU 在每个元素上都不等于所尝试的任何公式，最好的候选也只匹配其中 99.8%。加载时，canary（哨兵值）检查这些字节。

需要现有栈字节的用户保留契约 (i)；如果该栈本身就以库 epilogue 的方式融合，也可以借助该栈自己的库 epilogue 选择 (iii)。新模型可以选择 (ii)。没有声明，什么都不会改变字节。共有三条路线：

* **路线 (b)，库 epilogue**；
* **路线 (c)，tilefish 自己 kernel 中的 hook**：先在用户的 extern 中，然后在 tilefish 按架构生成的 GEMM 和卷积算子中（tcgen05、wgmma、mma.sync）；
* **prologue**，进入 A 操作数。

测量结果也给出了速度的界限：

* 当 pin 保留了一个快速算法时，cuBLASLt epilogue 可以省下 GEMM 后接其 epilogue kernel 所用时间的 9 到 48%。当 epilogue 的 pin 是一个较慢的算法时，它反而更费时间，GELU 在五个形状中的两个上就是如此；
* 一个直接的 Triton GEMM 用时是 cuBLASLt 的 1.22 到 1.44 倍，把一个朴素的 epilogue 融合进去比不融合还慢。生成的 GEMM 需要 kernel 方面的工作（实验 X8）才能划算。X8 的持久化 warp 特化 Gluon GEMM 在 2048 和 8192 行时，在测试集 32 个形状中的 25 个上不超过已 pin 的 GEMM 的 1.10 倍（32 和 256 行时则不然），在测量的每个形状上都写出其字节；融合进它的 bias + GELU 或门控激活，比库 GEMM 后接区域快 1.33 到 1.70 倍（见“X8：结果”）。

因此第 7 节建议先做 extern 上的 hook 和 cuBLASLt epilogue，在 X8 之后再做生成的 GEMM；评审确定了这个顺序（见“决定”）。随后 advisor 会为每个 epilogue 候选报告能够承接它的路线和契约。

## 目录

* 决定（2026-11-16）
* E1：实现说明（2026-11-17）
* E1 与 E2 结合（2026-11-17）
* X8：结果（2026-11-17）
* 0\. 背景：已有的东西
* 1\. 用例
* 2\. 数值问题与三种契约
* 3\. 路线
* 4\. 表层语法
* 5\. 与 plan、报告、advisor、SASS 断言、trace（追踪记录）、host 和 schema（模式）的交互
* 6\. 实验：阶段 0 结果和仍需要的实验
* 7\. 分阶段计划、里程碑和测试
* 8\. 问题（2026-11-16 已解决）
* 9\. 已有工作

附录 A：一个完整的示例。

## 决定（2026-11-16）

草案于 2026-11-16 评审。第 8 节的每个问题都已解决。标签沿用问题的编号（RFC 0001 的 D1 到 D7 保留其名称，本 RFC 中的“D6”仍指 RFC 0001 的导出时验证）：

* **Q1 和 Q2：采用三种契约。** 第 2.3 节定义的 (i) 精确、(ii) 累加器和 (iii) 库，是融合进调用时的契约，RFC 0001 的原则也延伸到它们：融合本身绝不改变字节；改变字节的是声明的模式（`R_acc` 定义在 IR 上，所以融合失败时仍然运行 `U(R_acc(P))`）。**默认是 `epilogues="exact"`**，无需声明即开启。`"accumulator"` 和 `"library"` 只在显式声明时才生效。
* **Q3：在哪里声明模式。** 在 graph 级别（`tilefish.fusion(epilogues=...)`）或在 `tilefish.fused` 作用域中，绝不按调用声明。
* **Q12：工作顺序。** E1 和 E2 先做，并行进行：
  * **E1**：extern 上的 epilogue hook，`value="out"` 和 `value="acc"`，以及第 5 节中 advisor 的改动；
  * **E2**：通过新的 impl `cublaslt.matmul` 实现 cuBLASLt epilogue：先做 `BIAS`、ReLU 和 `beta * C`，按 pin 验证，开启 canary；GELU 以后再做，作为由二进制文件定义的 epilogue。

  E3（生成的 GEMM）和 E6（cuDNN 融合图）以后再做；E4 和 E5 在它们所需的 RFC 0001 阶段之后做。
* **Q4 到 Q11：采用第 8 节推荐的默认值。** 简言之：
  * Q4：必须等于其所替代的库 pin 的生成 GEMM，是 E3 的显式选择加入（`equal="cublaslt"`），在每个 bucket（分桶）上验证，验证失败之处保留库 GEMM 和路线 (a)；
  * Q5：当生成的 GEMM 在移植模型所用形状的一个固定测试集上不超过已 pin 的 cuBLASLt GEMM 的 1.10 倍时，E3 按架构被接受；
  * Q6：先做 bias、ReLU 和 `beta * C`，经测量相对 (ii) 精确，并按 pin 验证；GELU 以后再做，由二进制文件定义，需通过 `epilogues="library"` 开启；epilogue 条目默认开启 canary；在 epilogue 的算法没有 f32 形式的地方，以普通 pin 的 f32 输出为参考，辅助输出必须等于 `ρ(acc + b)`；
  * Q7：新的 impl `cublaslt.matmul`（`epilogue`、`aux`、`beta`、可选的 `canary`）；`cublaslt.gemm` 和 `cublaslt.gemm_bias` 保持原样；
  * Q8：名称按提案；
  * Q9：门控激活，每个 program 两个累加器；
  * Q10：先做覆盖整行 tile 的 norm，以声明的顺序跨 tile 切分的形式放在 E4；
  * Q11：cuDNN 融合图（E6）放在 E2 和 X13 之后，sm_103 上的融合条目要求 cuDNN 9.13 或更高版本。

## E1：实现说明（2026-11-17）

在 tilefish 0.0.11 上实现，随 0.0.12 发布，包括 `tilefish.hook`、`@tilefish.extern(..., epilogue=...)`、`tilefish.fusion(epilogues=...)` 和作用域 `with tilefish.fused(epilogues=...):`，`tilefish._epilogue` 的生成模块，hook 的下沉（lowering）（`tilefish.lower.lower_hook`，note 22），`tilefish._fusion` 中的合并、改写 `R_acc` 和检查，融合报告的新字段，以及 advisor 的路线。两个 schema、crate 和 C ABI 都没有变化。

* **按设计实现的部分（3.2.1、4、5）：** 两种 hook（`value="out"` 在 kernel 的舍入之后，契约 (i)；`value="acc"` 作用于累加器，契约 (ii)）；`epilogues="exact"` 默认开启，`"accumulator"` 按 graph 或按作用域声明（Q2、Q3）；`R_acc` 在形成任何区域之前作用于 IR，所以 hook 没有接收的区域仍然运行 `U(R_acc(P))`；每个配对一个融合算子，以其内容命名；被规划器或检查拒绝的配对保持路线 (a)，原因写在报告中；导出时检查每个配对；门控激活用双累加器（Q9，`split=2`）；norm 只覆盖整行 tile（Q10，`block_n=`）；advisor 的路线及其契约和节省。
* **ABI**（`tilefish._epilogue.ABI`）。kernel 每个 hook 有两个 stub 不列出的参数：`<NAME>`（一个 `tl.constexpr`：一个 jit 函数）和 `<NAME>_ARGS`（一个元组），并在每个输出 tile 本来要存储的地方调用 `tile = NAME(tile, rows, cols, row_mask, col_mask, NAME_ARGS)`：`tile` 是已舍入的 tile（`out`）或 f32 累加器（`acc`），或者是并排的 `S` 个 tile 组成的元组（`split=S`）；`rows` 是该 tile 在视为 `[R, L]` 的输出中的行，`cols` 是它在一个块内的列，在 Gluon kernel 中采用该 tile 布局的切片布局；它返回 kernel 要存储的内容（`acc` hook 的返回值由 kernel 转换为输出的类型，`tile.to(OUT.dtype.element_ty)`；split hook 返回一个 tile 或并排的 `S` 个 tile，输出的行宽为那么多个块）。Triton 3.5.1 接受 jit 函数作为 `constexpr`，也接受由张量和标量组成的元组作为嵌套调用的参数，已实测。
* **实现过程中做出的决定：**
  * **生成模块，而不是编译规格中的 hook。** 规格、manifest 条目和两个 host 描述的是值，不是函数。因此，声明了 hook 的 extern 通过一个生成模块编译：`tf_<extern>__hooks(<the stub's parameters>)` 以关键字方式调用 kernel，传入恒等 hook；一个配对 `tf_<extern>_epi_<hash>(<the stub's parameters>, ep_*)` 则以区域的 hook 及其参数（作为元组）调用它。kernel 的主循环在两者中是同一份源码，内联进一个 IR，由这个 IR 标识二进制文件（TTIR，Gluon 则为 TTGIR）；模块的 sha256 计入声明 digest（`source_sha256`）。没有 hook 的 extern 保持原样。
  * **kernel 存储什么。** 当有其他东西读取结果时，存储结果本身（此时区域的每个输出都是旁路输出，由 hook 存储）；否则存储区域中第一个与结果形状相同的输出（对 `out` 还要求类型相同：kernel 在 hook 之前舍入到输出的类型），融合算子的输出取该类型（`acc`），对于 split hook，则取一个块的形状；否则照样存储结果，写入一个没有任何东西读取的缓冲区（仍然比路线 (a) 少一次启动和一次读取）。
  * **`U(R_acc(P))` 的累加器形式** 不是“带 f32 输出的 kernel”（一个写出其输出类型的 kernel 不必支持这种形式），而是 `<extern>__acc`：带一个 `acc` hook 的 extern，这个 hook 把 f32 tile 存入额外的一个输出，并原样返回它。对任何带 `acc` hook 的 kernel，这都是同一个 kernel、同一个累加器；它舍入后的结果就是该 extern 的结果。
  * **类型。** 区域的类型是 graph 的类型，融合算子的类型是 extern 的类型：调用把某个 graph 维度绑定到 extern 的某个维度时，该 graph 维度被替换为 extern 的维度（静态维度替换为它的值），因此融合算子拥有 extern 的规格（stem、bucket、`by_bucket` 配置、grid、整除性）；只属于区域的带 bucket 的维度会被拒绝（`epilogue-dims`）。
  * **Gluon。** hook 是一个 `@triton.jit` 函数，Gluon kernel 可以调用它（已实测）。逐元素 hook 的每个值都是完整的 tile，所以它的布局就是 tile 的布局；在那里 `-0.0` 标量写作 `tl.cast(bits)`（Gluon 的 `tl.full` 需要一个布局）。在 E1 中，Gluon kernel 的 hook 里的行区域会被拒绝（归约的结果是另一种布局）。
  * **整行。** 对一个 tile 的每一行做有序求和（`_tf_hsum_*`，`tilefish.lower` 的 note 22）：各条链通过对半拆分求得，lane（通道）之上的树通过 gather 求得，只计算有列的 lane（没有列的 lane 始终为 `+0.0`，即 revision 2 的填充给出的值，所以伙伴是这种 lane 的那一级树加的是 `+0.0`）：按行模板的顺序得到行模板的各项；`max` 与顺序无关；没有 `argmax`，没有不带 `order=` 的 sum（D2）。
  * **按行和按列的操作数**（bias、scale、norm 的权重）每个 tile 作为一个向量加载一次再广播（与逐元素模板在每个 lane 的偏移处读到的位相同），而不是逐元素 gather。
  * **融合版本。** 合并只作用于对带 hook 的 extern 的调用，0.0.11 及以前的 plan 都没有这种调用，版本 1 和 2 都一样：每个没有这种调用的 graph 都照旧规划（其 plan、manifest 和报告的字节都相同）。没有新版本。
  * **`tilefish.barrier(y)`** 停止对 `y` 的 epilogue 融合（第 4 节）；**每个调用一个区域**（第一个合适的区域；其他区域为 `epilogue-taken`），每个区域一个调用（读取两个结果的区域融合进第一个调用的 hook）。
  * **合并不能做的事。** 在单元图（包括区域和融合配对）中形成环（`epilogue-cycle`：一条路径可能从区域的一个节点离开，又从另一个节点回来）；在 hook 中读取 extern 原地更新的张量，无论哪个版本（`epilogue-inout`）。一个 split，如果其块由 hook 持有、同时又被其他东西读取，则在融合调用之后仍是存储结果的视图；偏移不是 16 字节倍数的稠密视图会为 hook 复制一份（hook 的参数是按对齐指针编译的）；hook 的行模板断言 kernel 交给它的 tile 包含一整行。
  * **融合配对的 `reference` 模式** 是 extern 的 `reference=` 后接解释执行的区域；在 `R_acc` 下，f32 结果是 hook 的 `reference=`（`tilefish.hook(..., reference=...)`），否则是加宽后的 extern 参考。
  * **`epilogues="accumulator"` 下的库调用** 在 E1 中不改写：它的 f32 形式属于 E2（把 pin 重新编码为 f32 的 `D`，3.1.1）；它保持路线 (a)，advisor 会指向 E2 和 E3。
* **检查**（配对的 D6）：每个融合配对都与 extern（或其累加器形式）后接区域各算子（每个算子一个 kernel，每个中间值都存储）的组合进行比较，在每个 bucket 上、用两个种子：extern 的操作数取正态分布，在最前面的元素放入特殊值（两个零和两个无穷、NaN、最大的有限值、一个次正规数），每个旁路输入使用 `_sample` 的特殊值和随机位模式。有差异就拒绝导出；把 `out` hook 应用到未舍入值上的 kernel 会被拒绝（已测试）。
* **示例**（`python/examples/epilogue_gemm.py`）：一个带两种 hook 的 Triton GEMM（TMA 描述符、`tl.dot`、分组顺序，epilogue 作用在列的 1、2 或 4 个子 tile 上，每个子 tile 调用一次 hook；sm_100/103 上为 `tcgen05.mma`，sm_120 上为 `mma.sync`，各自由 `tilefish.sass` 断言），它的双累加器形式，一个 Gluon `tcgen05` GEMM，以及一个作为隐式 GEMM 的一维卷积（行数为 `B * Lo`）。
* **测量结果**（B300 MIG 切片，sm_103，30 个 SM；在 CUDA graph（CUDA 图）中 capture（捕获）的每次 plan 运行的微秒数，`profile_plan` 的 `plan_us`，200 次的截尾均值；hook 与路线 (a)（先 extern，再把区域作为一个 kernel，对结果做 `tilefish.barrier` 即得到这种形式）以及未融合组合对比）：

  | 形态（pattern；bf16；`H = 1024`，`F = 4096`） | 行数 | hook（1 个步骤） | 路线 (a)（2 个步骤） | 未融合（步骤数） | hook，`"accumulator"` | (a) / hook |
  |---|---|---|---|---|---|---|
  | GEMM + bias + GELU | 256 | 25.0 | 25.1 | 37.1 (6) | 26.7 | 1.00 |
  | | 2048 | 111.6 | 135.3 | 231.1 (6) | 117.2 | 1.21 |
  | GEMM + bias + 周期激活 | 256 | 31.2 | 27.2 | 61.8 (10) | 32.6 | 0.87 |
  | | 2048 | 139.2 | 143.9 | 402.9 (10) | 143.9 | 1.03 |
  | 双 GEMM（`2F` 行）+ SwiGLU | 256 | 41.2 | 44.0 | 61.8 (6) | 41.0 | 1.07 |
  | | 2048 | 229.8 | 261.1 | 410.2 (6) | 240.4 | 1.14 |
  | GEMM（`D = 256`）+ 残差相加 + RMSNorm，返回残差 | 256 | 17.0 | 14.9 | 23.2 (11) | 16.8 | 0.88 |
  | | 2048 | 16.9 | 16.9 | 35.6 (11) | 16.9 | 1.00 |
  | conv1d（`k = 7`，512 到 512 通道，`L = 4096`）+ bias + 周期激活 | 4096 | 164.6 | 127.5 | 193.4 (10) | 167.6 | 0.77 |

  这些 GEMM 是示例中的（`gemm_sm100`：8 个 warp，epilogue 作用在两个 64 列的子 tile 上；norm 用的是一个能看到整行的 128 x 256 tile；卷积 8 个 warp），它们既不是持久化的，也不是 warp 特化的：hook 的算术在主循环之后、在 kernel 自己的 warp 上、以 kernel 的 occupancy（占用率）运行，不与任何东西重叠。所以，在 epilogue 轻而其操作数大的地方融合划算（2048 行时的 bias + GELU：1.21x；门控激活，其融合算子只存储一半的列：1.07 到 1.14x），在 GEMM 短的地方持平，在 epilogue 对 kernel 那几个 warp 来说太重的地方亏损（卷积上每个元素一个 `sin`：0.77x）。配置与融合一样重要：用 4 个 warp，或在一个 hook 中处理整个 128 列的 tile 时，hook 的寄存器发生 spill（寄存器溢出；`sin`：255 个寄存器和一个栈帧），同样的周期激活只有 0.30 到 0.45x；第一版通过对 MMA 布局的 tile 做 reshape 和拆分来实现的有序行求和，每个线程 spill 了 15 KB（0.04x），上面基于 gather 的求和消除了这个问题。拿到剩下的部分是 X8 的 kernel 工作（让 epilogue 与下一个 tile 的主循环重叠），而不是 hook 的工作。路线是声明的选择，从不根据耗时来决定（第 5 节）：advisor 报告每条路线会带来的节省。

* **测试：** `tests/test_epilogue.py`（无 GPU：声明及其拒绝、生成模块、每个形态在两种模式下的合并、hook 的代码与区域逐个算子对比、两个 graph 中的同一配对是同一个算子、`R_acc` 的 IR 输入与输出、作用域、每种拒绝及其原因、报告、边界、advisor 的路线）；`tests/test_gpu_epilogue.py`（每个形态在每个 bucket 上并通过 `verify_fusion` 与 `U(P)` 和 `U(R_acc(P))` 比较；每个有限 bf16 值作为一个 tile 经过各激活的 hook；GEMM 之后的 40 个随机区域，两种模式；Gluon GEMM；导出、replay（回放）和 capture；融合二进制文件的 MMA 序列与 extern 的相同；撒谎的 hook 被拒绝；参考）；Rust 和 C++ 门禁把这些形态作为 `fused_epi_*` plan 运行，既启动也 capture。除 Gluon 测试（sm_100、sm_103）外都与架构无关；sm_120 属于以后的门禁。
* **尚未实现：** prologue；`R_acc` 下的库调用（E2 中也没有：见“E1 与 E2 结合”）；比一个 tile 更宽的行上的 norm（E4）；Gluon kernel 的 hook 中的行区域；用 `by_arch=` 声明的 extern 接收 hook；把 MMA 序列断言作为声明的 SASS 断言（目前由一个测试检查）；在真正的 Gluon 隐式 GEMM 卷积上做 X12。

## E1 与 E2 结合（2026-11-17）

E1 和 E2 从 0.0.11 开始并行构建，各自带有一个两值的 `epilogues=`：E1 的 `("exact", "accumulator")`，E2 的 `("exact", "library")`。合并后，模式是三者之一，按 graph（`tilefish.fusion(epilogues=...)`）或按 `tilefish.fused(epilogues=...)` 作用域设定；对在作用域中 trace 的算子，作用域的模式优先于 graph 的模式（Q3）：

| 模式 | 带 hook 的 extern | 结果被 graph 加宽的 `cublaslt.gemm`（或 `gemm_bias`） |
|---|---|---|
| `"exact"`（默认） | 它的 `value="out"` hook：契约 (i) | 保持原样；报告的 `library_epilogues` 列出 `"library"` 会做什么 |
| `"accumulator"` | `R_acc` 及其 `value="acc"` hook：契约 (ii) | **保持原样**：库调用的 `R_acc` 不做规划，所以 GEMM 保留它的舍入；报告的条目会说明这一点（`"mode": "accumulator"`，一个 `note`） |
| `"library"` | 它的 `value="out"` hook：契约 (i)，与 `"exact"` 相同 | 在 7.3 的形态匹配之处改写成 `cublaslt.matmul`：契约 (iii)，经测量相对 (ii) 精确 |

* **`"accumulator"` 对库 GEMM 意味着什么。** 契约 (ii) 要求由声明的模式改变程序，而不是由融合改变（第 2.3 节）：在 `R_acc` 下，融合失败时仍运行 `U(R_acc(P))`，GEMM 写出 f32。E1 和 E2 都没有为库调用规划这一点（E1 的 `R_acc` 作用于带 `acc` hook 的 extern；E2 的改写是它自己的模式），因此在 `"accumulator"` 下，cuBLASLt GEMM 按 `"exact"` 下的方式规划：其结果被舍入，对该 GEMM 而言就是 `U(P)`；这是一个带消息的空操作，而不是拒绝（一个 graph 可能既调用一个希望作用于累加器的 extern，又调用一个 cuBLASLt GEMM）。消息是该 GEMM 在融合报告中的 `library_epilogues` 条目：`"mode": "accumulator"`，`"note"`：为它的 epilogue 声明 `"library"`（在 graph 上或在作用域上）。为库调用规划 `U(R_acc(P))`（pin 的 f32 形式，然后是区域）留给以后的里程碑；advisor 的 E2 路线会说明 `"library"` 会在哪里接收该区域。
* **`"library"` 保留 `"exact"` 的 extern hook。** 第 4 节把 `"library"` 勾勒为 (ii) 和 (iii)；E2 构建它时没有 `R_acc`，合并后仍然如此：在 `"library"` 下，extern 的区域进入它的 `value="out"` hook；一个既希望某个 extern 作用于累加器、又想要 cuBLASLt epilogue 的 graph，声明其中一个，并用作用域划定另一个，例如在 graph 上用 `epilogues="library"`，并在 extern 的算术外围用 `with tilefish.fused(epilogues="accumulator"):`。
* **作用域与库改写。** E2 只读取 graph 的模式。合并后，一个 cuBLASLt 形态只有在它的每个算子（从 `y.to(f32)` 到最后一个 `.to()`，以及 `c` 和 bias 的加宽）都处于 `"library"` 模式时才被改写（在 `"accumulator"` 作用域中对 extern 结果的加宽属于 `R_acc`，所以读取它的形态留给它）；算子处于其他模式的形态会在报告中列为已匹配但未应用，并附上这些模式。改写先在 graph 自己的算术上运行，然后是 `opmath`、`mul-add` 和 `R_acc`，与两条开发线的排序一致；`cublaslt.matmul` 调用保留 GEMM 的作用域和位置。
* **名称与 digest。** 在两条开发线中，`epilogues` 都不在任何区域的规范设置里，所以模式没有改动的区域在每种模式下都是同一个算子；当模式不是 `"exact"` 时，融合报告的 `fusion` 会说明模式。advisor 的阻碍项来自两条开发线：`epilogue-hook-consumer`（E1，阶段 4）、`epilogue-library-cublaslt`（E2，阶段 `E2`），以及 `epilogue-library`，附带 GEMM 的区域未被匹配的原因（E3 和 E6 以后再做）。
* **测试：** `tests/test_epilogue_modes.py`（无 GPU：在一个同时调用带两种 hook 的 extern 和一个 cuBLASLt GEMM 的 graph 上测试三种模式和作用域、每种模式下报告的条目、对未知模式的拒绝），以及它在 `tests/test_gpu_epilogue.py` 中的 GPU 对应测试（这个混合 graph 被导出、回放，并逐字节比较：extern 对照 `U(R_acc(P))`，GEMM 对照公式）。

## X8：结果（2026-11-17）

X8 要回答的是：tilefish 自己的 GEMM 能否快到让融合进它变得划算（Q5），以及它的字节是否就是库的字节（E3 的问题，契约 (i-d)）。该 kernel 是一个示例，`python/examples/persistent_gemm.py`，它在一个快到值得融合进去的 kernel 中展示这些 hook；它不是受支持的功能（`docs/scope.md`：tilefish 不在 kernel 速度上与库竞争）。下面是该实验的记录。

**kernel。** `c = a @ b`，bf16 或 f16 操作数，f32 累加器，`cublaslt.gemm` 的四种布局（`TN`、`NN`、`TT`、`NT`），E1 的 hook `ACC` 和 `EPILOGUE`，以及门控激活的双累加器（`split=2`）。

* **sm_100 和 sm_103**（`gemm_sm100_kernel`，Gluon）。一个持久化 grid（`tilefish.persistent()`，每个 SM 一个 program，分组的 tile 顺序），其各个分区运行在各自的 warp 上：
  * 一个单 warp 的 TMA 生产者，填充一个最多 8 个共享内存 stage 的环，每个 stage 有一个 `ready` 和一个 `empty` mbarrier；
  * 一个单 warp 的 `tcgen05_mma` 发射者，把每个 tile 累加到两个张量内存缓冲区之一（`2 * split * BN <= 512` 列）；
  * epilogue 运行在 `PARTS` 个分区上，每个分区 `NW` 个 warp，默认是两个分区、每个八个 warp，每个分区排空一个缓冲区，同时发射者填充另一个。它以 32 列的子 tile 读取累加器（`tcgen05.ld 32x32b`），每个子 tile 调用一次 hook，并经由共享内存用 TMA 存储。子 tile 循环不展开，所以 hook 的代码在二进制文件中只有一份。

  一个 tile 的整个 `k` 由一个 program 按顺序处理，每个步骤 64 列。它的各个阶段带有 `tilefish.probe`。
* **sm_120**（`gemm_sm120_kernel`，Triton）：TMA 加载、软件流水的 `tl.dot`（`mma.sync m16n8k16`）、持久化循环、epilogue 作用在子 tile 上。它为 sm_120 构建，并在 sm_103 上按其 SASS 断言检查；它的逻辑在 sm_103 上编译为 `tcgen05` 运行（`impl="triton"`，与 Gluon kernel 的字节相同）。它的速度属于以后的 sm_120 门禁。
* **按 bucket 的配置**（在 `M` 上用 `by_bucket=`）。`config_sm100` 选取模型估算时间最少的 tile：128 行时一个 `k` 步骤的代价是 `BN / 256 * 271 ns` 的 tensor core 时间（64 行时减半），或者其操作数字节以 220 GB/s 进入一个 SM 的时间，取较长者，再加上每个 tile 0.3 us，共 `ceil(tiles / SMs)` 轮。它取共享内存能容纳的最多 stage 数。在测试集的 64 个形状上，它选的 tile 平均比 16 种 tile 扫描中最好的慢 0.4%，最多慢 8%。
* **描述符 box。** 描述符的块大小由声明固定。每个块是所有 bucket 中最小的 tile，更大的 tile 被拷贝为多个 box，所以 tile 仍然可以按 bucket 变化。
* **Triton 3.5.1 不允许的东西。**
  * Gluon 没有双 CTA（线程块）MMA（`cta_group::2`），也没有 TMA 多播，所以一个 tile 只属于一个 CTA。一个 `num_ctas=2` 的 `tl.dot` 生成了 `cta_group::2`，但运行时出错。
  * Gluon 的每个形状都是 2 的幂，所以 96、160 或 192 列的 tile（在 30 个 SM 上量化得更好）会被拒绝。
  * 张量内存加载只为 4 或 8 个 warp 做了下沉。

**环境。** 一个 B300 MIG 切片：sm_103，30 个 SM，torch 2.9.1，Triton 3.5.1，ptxas 12.9，cuBLASLt 12.8.4。GEMM 的时间是一个包含 20 次背靠背调用的 CUDA graph 30 次 replay 的中位数。plan 的时间是 `profile_plan` 的 `plan_us`，即 200 次 replay 的截尾均值。

**与 cuBLASLt 比较速度（Q5）。** 下表每个单元格给出三个以微秒为单位的时间和两个比值：

* 时间分别是这个 GEMM 的、已 pin 的 cuBLASLt GEMM 的（`heuristic="torch"`，即 `torch.mm` 运行的那个），以及启发式提供的 8 个算法中最快那个的；
* 比值是这个 GEMM 除以 pin，以及这个 GEMM 除以最快算法（一个导入的算法，`algos=AlgoTable`）。

操作数为 `TN` 布局的 bf16。

| M | N \ K | 768 | 1024 | 3072 | 4096 |
|---|---|---|---|---|---|
| 32 | 768 | 4.7 / 2.9 / 2.8 (1.64, 1.65) | 4.9 / 3.1 / 3.2 (1.55, 1.55) | 8.0 / 5.5 / 5.5 (1.46, 1.46) | 9.6 / 6.6 / 6.6 (1.45, 1.45) |
| 32 | 1024 | 4.6 / 3.2 / 2.9 (1.44, 1.61) | 5.0 / 3.5 / 3.1 (1.41, 1.60) | 8.2 / 6.7 / 5.3 (1.21, 1.54) | 9.8 / 8.3 / 6.5 (1.19, 1.50) |
| 32 | 3072 | 5.3 / 3.3 / 3.3 (1.59, 1.60) | 6.2 / 3.7 / 3.7 (1.68, 1.68) | 10.0 / 6.8 / 6.8 (1.48, 1.46) | 12.1 / 8.4 / 8.4 (1.44, 1.44) |
| 32 | 4096 | 7.1 / 4.8 / 4.2 (1.49, 1.69) | 8.2 / 5.3 / 4.7 (1.54, 1.73) | 16.8 / 14.2 / 10.9 (1.18, 1.54) | 23.0 / 16.8 / 17.1 (1.37, 1.35) |
| 256 | 768 | 5.2 / 3.4 / 3.2 (1.55, 1.65) | 5.7 / 3.7 / 3.5 (1.56, 1.65) | 9.6 / 6.0 / 6.0 (1.60, 1.60) | 11.5 / 7.2 / 7.2 (1.61, 1.61) |
| 256 | 1024 | 5.8 / 4.0 / 3.6 (1.45, 1.62) | 6.3 / 4.5 / 3.9 (1.41, 1.64) | 10.9 / 8.2 / 6.5 (1.34, 1.67) | 13.3 / 10.0 / 8.2 (1.32, 1.62) |
| 256 | 3072 | 7.8 / 5.6 / 5.6 (1.38, 1.38) | 8.8 / 7.5 / 6.3 (1.18, 1.39) | 17.4 / 16.1 / 14.1 (1.08, 1.23) | 21.8 / 20.4 / 17.8 (1.07, 1.22) |
| 256 | 4096 | 10.5 / 6.6 / 6.6 (1.59, 1.59) | 12.2 / 7.5 / 7.5 (1.61, 1.61) | 26.1 / 17.2 / 17.2 (1.52, 1.52) | 42.5 / 27.9 / 28.0 (1.52, 1.52) |
| 2048 | 768 | 11.8 / 9.5 / 8.9 (1.25, 1.33) | 14.0 / 13.0 / 10.6 (1.08, 1.32) | 31.3 / 30.3 / 26.0 (1.03, 1.20) | 39.9 / 38.9 / 33.7 (1.03, 1.19) |
| 2048 | 1024 | 15.3 / 10.0 / 10.0 (1.53, 1.53) | 18.0 / 12.4 / 12.4 (1.45, 1.45) | 41.0 / 31.7 / 31.8 (1.29, 1.29) | 52.1 / 41.5 / 41.5 (1.26, 1.26) |
| 2048 | 3072 | 31.1 / 32.2 / 25.4 (0.97, 1.22) | 38.2 / 39.6 / 31.7 (0.96, 1.20) | 108.5 / 98.7 / 92.1 (1.10, 1.18) | 145.6 / 127.5 / 122.2 (1.14, 1.19) |
| 2048 | 4096 | 39.1 / 42.2 / 32.0 (0.93, 1.22) | 48.3 / 52.0 / 40.7 (0.93, 1.19) | 142.4 / 130.1 / 120.6 (1.09, 1.18) | 186.3 / 168.7 / 159.3 (1.10, 1.17) |
| 8192 | 768 | 31.2 / 32.2 / 25.3 (0.97, 1.23) | 38.3 / 39.5 / 31.9 (0.97, 1.20) | 103.5 / 98.3 / 93.2 (1.05, 1.11) | 135.1 / 127.6 / 122.3 (1.06, 1.10) |
| 8192 | 1024 | 39.4 / 42.8 / 32.2 (0.92, 1.22) | 51.6 / 53.2 / 43.9 (0.97, 1.18) | 131.4 / 130.5 / 120.6 (1.01, 1.09) | 171.7 / 170.5 / 158.4 (1.01, 1.08) |
| 8192 | 3072 | 108.7 / 121.0 / 94.3 (0.90, 1.15) | 135.9 / 148.8 / 119.9 (0.91, 1.13) | 388.4 / 371.9 / 370.6 (1.04, 1.05) | 518.3 / 483.7 / 481.5 (1.07, 1.08) |
| 8192 | 4096 | 142.2 / 159.5 / 124.0 (0.89, 1.15) | 177.7 / 196.6 / 157.5 (0.90, 1.13) | 519.3 / 493.9 / 493.8 (1.05, 1.05) | 680.8 / 645.2 / 645.4 (1.05, 1.05) |

按行数统计，有多少形状不超过 pin 的 1.10 倍（括号中是比值的几何平均）：

| 行数 | 不超过 pin 的 1.10 倍 | 相对 pin 的比值 |
|---|---|---|
| 8192 | 16 个中的 16 个（0.98） | 0.89 到 1.07 |
| 2048 | 16 个中的 9 个（1.12） | 0.93 到 1.53 |
| 256 | 16 个中的 2 个（1.41） | 1.07 到 1.61 |
| 32 | 16 个中的 0 个（1.44） | 1.18 到 1.68 |

总计是 64 个形状中的 27 个。与最快的算法相比，64 个中只有 6 个在 1.10 以内；在 `k` 为 768 和 1024 时，它比 pin 快。其他布局和 f16 在各自 12 个形状的子集上（bf16 的 `NN`、`NT`、`TT`，f16 的 `TN`）表现相同：是 pin 的 0.87 到 1.70 倍，按行数呈现同样的规律。

时间花在哪里是测量出来的，不是猜的：

* **每次启动的固定开销。** 在 `k = 64` 时，这个 GEMM 用时 2.95 us，cuBLASLt 用时 2.03 us，差距随 `k` 只缓慢增长。
  * probe（探针）显示，在生产者第一次加载之前约有 1.2 us：barrier 初始化、张量内存分配，以及各个分区的准备。
  * 在两个 GEMM 之间放一个空 kernel，两者的时间都不变，所以 cuBLASLt 并没有从重叠启动中获益。
* **在 30 个 SM 上的量化。** 例如，2048 x 1024 可以分成 128 个 128 x 128 的 tile，即 4.3 轮，或 64 个 128 x 256 的 tile，即 2.1 轮。cuBLASLt 的 pin 使用 cluster（线程块簇；`cluster_shape_id` 3 和 5）。
* **进入 SM 的字节。** 单 CTA 的 tile 每个步骤读取 `(BM + BN) * BK` 个操作数。双 CTA 的 tile，或向一个 cluster 多播，每个 CTA 读得更少，而 Triton 3.5.1 的 Gluon 两者都无法表达。
  * 在 tile 填满 SM 的地方（2048 和 8192 行，`N` 为 3072 或更大），kernel 的用时是 pin 的 0.9 到 1.1 倍。在那里发射者约 88% 的时间处于忙碌：它在三分之一的步骤中等待一个槽位。
  * 在 32 和 256 行时，启动开销和轮数占主导。

**Q5 解读。** E3 的验收线（在一个固定测试集上不超过已 pin 的 GEMM 的 1.10 倍）在 2048 和 8192 行时满足，即那 32 个形状中的 25 个。在 32 和 256 行时不满足，相对最快的导入算法也不满足。用生成的 GEMM 替代库调用，在形状大且 epilogue 被融合的地方划算（见下一段）。在此之下，它需要 Triton 3.5.1 所缺少的东西：双 CTA MMA 或多播，以及不是 2 的幂的 tile。更便宜的 prologue 也会有帮助。

**字节（契约 (i-d)）。** 在上面 64 个形状中的每一个上，该 kernel 的字节都是 cuBLASLt pin 的字节。在其他布局和 f16 的 48 个形状上、在测试的每个 bucket 上（`tests/test_gpu_persistent_gemm.py`），它的字节也是 pin 的字节，也是最快算法的字节。在这个切片上，cuBLASLt 12.8.4 的 pin 对这些形状都不切分 `k`。

在六个 `k` 较长的形状上（`32 x 768 x 4096`、`32 x 1024 x 4096`、`32 x 4096 x 4096`、`256 x 768 x 4096`、f16 的 `256 x 1024 x 3072`，以及 `2048 x 1024 x 4096`）：

* 启发式提供的每个不切分 `k` 的算法（每个形状 8 个：其他 id、tile 和 cluster）都写出了这个 kernel 的字节，没有一个元素不同；
* pin 的配置在 `splitk_num` 为 2、3 或 4 且使用 f32 部分和（归约方案 `COMPUTE_TYPE`）时，在 bf16 中有 0.18 到 0.24% 的元素不同，在 f16 中有 1.06 到 1.19%。X3b 在 32 行时对 13.1.0 的 split-K（K 维切分）pin 也发现了同样的 0.2%。

所以在 sm_103 上，`kind::f16` GEMM 的字节取决于沿 `k` 的求和顺序，不取决于测量过的任何其他因素。

**相等的代价。** 在 pin 不切分 `k` 的地方，没有代价：无论配置如何，该 kernel 的字节都是 pin 的字节（已测试：tile、stage、epilogue 分区和 warp），所以只按速度选择配置。

要等于一个 split-K pin，kernel 必须把 `k` 切成 pin 的那些块，并按 pin 的顺序加 f32 部分和（X9）。这需要一个 workspace（工作区）、第二遍计算，以及每种切分数一个 kernel。上面每个 split-K 算法都比最快的不切分算法慢，所以这里没有为了相等而放弃什么。在 32 行时，这个 kernel 的切分版本会比现在的 kernel 快（cuBLASLt 的 2 路切分在 32 x 768 x 4096 上用时 8.5 us，这个 kernel 9.6 us），但那样它的字节就与任何不切分的 pin 都不匹配了。

**与它一起做 epilogue 融合（再看 E1 的表）。** 形态和尺寸与 E1 相同（bf16，`H = 1024`，`F = 4096`），单位为每个 plan 的微秒数。各列为：

* *hook*：配对，一个步骤，精确；
* *hook，acc*：`epilogues="accumulator"` 下的配对；
* *(a)，本 GEMM*：本 GEMM，然后是区域（一个 barrier）；
* *(a)，cuBLASLt*：RFC 定义的路线 (a)，即已 pin 的 cuBLASLt GEMM（torch 姿态）后接生成的区域；
* *未融合*：cuBLASLt 后接区域的各个算子，每个算子一个 kernel，附带其步骤数。

| 形态 | 行数 | hook | hook，acc | (a)，本 GEMM | (a)，cuBLASLt | 未融合（步骤数） | (a)，cuBLASLt / hook |
|---|---|---|---|---|---|---|---|
| GEMM + bias + GELU（erf） | 256 | 21.7 | 20.8 | 22.8 | 18.7 | 29.9 (6) | 0.86 |
| | 2048 | 77.8 | 75.9 | 103.5 | 103.5 | 199.0 (6) | 1.33 |
| GEMM + bias + 周期激活 | 256 | 26.6 | 26.6 | 24.6 | 20.5 | 51.2 (10) | 0.77 |
| | 2048 | 127.9 | 124.5 | 112.5 | 112.4 | 370.5 (10) | 0.88 |
| 双 GEMM（`2F` 行）+ SwiGLU | 256 | 22.4 | 26.5 | 28.8 | 24.9 | 43.0 (6) | 1.11 |
| | 2048 | 90.1 | 124.8 | 153.5 | 153.5 | 302.5 (6) | 1.70 |
| GEMM（`D = 256`）+ 残差相加 + RMSNorm | 256 | 10.8（未融合） | 10.9（未融合） | 11.3 | 9.1 | 18.0 (11) | (0.84) |
| | 2048 | 17.0（未融合） | 17.1（未融合） | 16.9 | 14.1 | 32.8 (11) | (0.83) |

* **划算之处。** 在主循环长到足以掩盖 epilogue、且 GEMM 接近库的速度时，融合进这个 GEMM 划算：
  * 在 2048 行时，bias + GELU 比路线 (a) 快 1.33 倍，比未融合的 plan 快 2.6 倍；
  * 门控激活（其融合算子只存储一半的列）比路线 (a) 快 1.70 倍。

  与 E1 的朴素示例 GEMM 相比，融合后的时间从 111.6 降到 77.8 us（GELU），从 229.8 降到 90.1 us（SwiGLU）。
* **不划算之处。** 在 256 行时，GEMM 本身就比 cuBLASLt 的慢 1.4 到 1.6 倍（见速度表），而融合省下的时间比这还少。
* **周期激活**（每个元素一个 `sin` 和一次除法）在 epilogue 的 warp 上受限于指令发射。probe 显示了这一点：
  * 每个 128 x 32 子 tile，hook 用时 2.5 us（erf GELU 为 1.2 us，恒等为 0.04 us），所以一个 tile 的 epilogue 在两个分区上需要 11 us，而主循环只需 5 us；
  * 发射者平均每个 tile 要为一个累加器缓冲区等待 6.6 us（GELU 时为 2.6 us，没有 hook 时不用等）。

  每个分区 8 个 warp、共两个分区，已是 Triton 3.5.1 的张量内存加载对每个分区所允许的上限。这个 GEMM 之后的周期激活是路线 (a) 的 0.88 倍；E1 的朴素 GEMM 给出 1.03，那是相对于它自己更慢的路线 (a)。
* **配置与融合一样重要。** 在 2048 行的 bias + GELU 配对上：
  * 一个 4 warp 的分区比两个 8 warp 的分区慢 1.9 倍（155 us）；
  * 展开子 tile 循环使周期 hook 的代码翻倍（4000 到 7900 条指令），其他什么都没变。
* **norm 没有融合进 Gluon kernel。** E1 不把行区域融合进 Gluon hook，所以 norm 那两行的前两列是用这个 GEMM 的路线 (a)。该 kernel 的 Triton 形式（`impl="triton"`，其子 tile 容纳 256 的整行）会融合它，但用时 18.3 us，比路线 (a) 慢。它的主循环是 Triton 自己的流水线，不是这个 kernel 的。
* **在 `epilogues="accumulator"` 下，双 GEMM 的配对也存储了已舍入的结果**（已在 0.0.13 修复）。拆分结果的视图在 `R_acc` 之后留在程序中，使结果保持存活（`tilefish._fusion` 中的 `y_needed`），因此 kernel 除了 hook 的输出之外还写出了 `[M, 2F]`：124.8 对 90.1 us。现在 `R_acc` 会删除结果中只被它所替换的那些加宽读取的 split：字节相同，89.7 us，`"exact"` 下为 90.2 us（上面的 hook，acc 列是修复之前的测量）。
* **未测量。** 作为 GEMM 的卷积没有用这个 kernel 测量：它的操作数是二维张量的 TMA tile，而隐式 GEMM 的生产者（im2col box）是 X12 和 E3 的工作。
* **契约 (i) 在每种情况下都成立。** 融合配对在每个 bucket 上、在特殊值下都与未融合的 plan 逐字节一致（测试）。每个二进制文件，无论是否融合，在构建时都满足 `tilefish.sass(mma="tcgen05.mma kind::f16", no_tf32=True)`，并且融合二进制文件的 MMA 序列就是 GEMM 的序列（已测试）。

**流水线的 probe 结果**（对 2048 x 4096 x 1024 plan 的一次插桩运行；每次等待前后都有 `tilefish.probe`）。没有 hook 时：

* epilogue 分区为每个累加器等待 6.7 us（中位数），所以 GEMM 受限于其主循环；
* 发射者发射一个步骤用 159 ns，平均为生产者等待 138 ns；
* 生产者平均为一个空闲槽位等待 137 ns。

有 bias + GELU 时，epilogue 的等待降到 2.8 us，hook 每个子 tile 用 1.2 us，发射者每个 tile 为一个累加器等待 2.6 us：现在 epilogue 成了瓶颈，约高出 20%。有周期激活时，发射者的等待是 6.6 us。生产构建不含任何 probe（`docs/instrument.md`）。

## 0. 背景：已有的东西

* **library op 承诺一次舍入。** `cublaslt.gemm` 用 f32 累加器计算 `out = a @ b^T`，并一次舍入到输出类型。`cublaslt.gemm_bias` 在这次舍入之前以 f32 加上一个输出类型的 bias（`CUBLASLT_EPILOGUE_BIAS`，tilefish 目前唯一使用的 epilogue）。算法被 pin 住（它的 64 字节、cuBLASLt 版本和 SM 数）。pin 会对照 cuBLASLt 自己的编码进行审核，并在加载时由 `cublasLtMatmulAlgoCheck` 检查。它按 tilefish 或 torch 的方式选择，或者从另一个引擎导入（`AlgoTable`）。带 bias 的 GEMM 除非显式选择加入，否则不使用 split-K，因为测得带 bias epilogue 的 split-K 会舍入两次（`docs/numerics.md`，第 4 节）。`cudnn.conv_fwd` 没有 bias。它的引擎配置按 cuDNN 版本和 SM 数被 pin 住，并在加载时由 canary 检查（`docs/manifest-schema-3.md`，第 9.4 节）。
* **plan 融合止于调用。** RFC 0001 的阶段 1 把 graph 函数体的算术融合进生成的区域。每个区域都与未融合组合 `U(P)` 逐字节一致，每次导出都会验证这一点（D6）。对 library op、extern 或 kernel 的调用是区域边界。GEMM 之后的算术按 0001 第 7 节的路线 (a) 运行：一个读取 GEMM 已存储、已舍入结果的生成 kernel。
* **移植中已经包含手工融合的计算 kernel。** 对手工融合引擎的移植会逐字节复现该引擎的 kernel。其中一些 kernel 是带融合激活 epilogue 的卷积，以 Gluon extern 的形式手写（sm_100 和 sm_103 上为 `tcgen05`，sm_120 上为 `mma.sync`），遵循原 kernel 的累加顺序及其 epilogue。对 tilefish 来说，这样的 epilogue 只是 extern 的一部分。下面的路线 (c) 把同样的事情变成规划器的一个功能。
* **advisor 看到了这个缺口。** `tilefish-explain --advise` 把最后的阻碍项为 `epilogue-library` 或 `prologue-library` 的每个候选列为未来工作（`docs/advise.md`，第 4 节）。它把两个 norm 之间的每条残差流列为 `library-in-the-middle`。在三个移植模型上，它按测得的时间把这些候选排在最前。最大的单个形态是：
  * GEMM 后接 col2im、overlap-add 和周期激活：每个实例 0.1 到 1.2 ms，在短 bucket 和长 bucket 上都是如此；
  * 卷积后接 bias 和周期激活：每个 0.2 到 0.3 ms；
  * 每个 transformer 层的输出投影之后的残差相加和 norm；
  * 门控激活；
  * QKV 投影之后带 rotary embedding 的逐 head norm。

## 1. 用例

本节列出一个模型在它已 pin 的 GEMM 和卷积之间需要什么。表中有两列需要定义：

* **tile 局部** 表示 GEMM 的一个输出 tile（`BM` 行乘 `BN` 列）是否拥有其 epilogue 需要的一切。这决定了路线 (c) 的难度。
* **库** 表示该形态是否存在库 epilogue。

| 用例 | 形态（graph 函数体草图） | tile 局部？ | 库 | 说明 |
|---|---|---|---|---|
| bias + 激活 | `act(y.to(f32) + b.to(f32)).to(bf16)`，其中 `act` 是 GELU（erf 或 tanh）、SiLU 或 ReLU | 是（一个按列的向量） | cuBLASLt `BIAS`、`RELU_BIAS`、`GELU_BIAS`（仅 tanh GELU）；卷积用 cuDNN 融合图 | 最常见的 epilogue |
| 门控激活 | `g, u = gemm(x, w).split(F, -1)`；`(silu(g.to(f32)) * u.to(f32)).to(bf16)` | 否：第 `j` 列需要第 `F + j` 列 | 无 | 需要 `g` 的一个 tile 和 `u` 的对应 tile：每个 program 两个累加器，或者把权重的行交错排列 |
| 周期激活 | `s = sin(a * h)`；`h + s * s / a`，其中 `h = y + b`，`a` 按通道 | 是 | 卷积可用 cuDNN 逐点链（X5）；cuBLASLt 中没有 | 卷积模型中最大的形态 |
| 残差相加（+ norm），残差作为旁路输出 | `h = y + r`（存储：残差流）；`n = rmsnorm(h, w)` | 相加：是；norm：仅当一个 tile 容纳整行时（`BN >= N`） | 相加用 cuBLASLt `beta * C`；没有库 norm | 当隐藏维度大于一个 tile 时，norm 的行归约会跨越 tile（3.2.3） |
| 卷积 + bias + 激活 | `act(conv(x, w) + b)` | 是（隐式 GEMM） | cuDNN 融合图（X5） | 卷积本身是 extern 或 `cudnn.conv_fwd` |
| 作为 GEMM + col2im 的转置卷积，然后是 overlap-add + bias + 激活 | `cols = gemm(x, w)`；`y = col2im(cols)`；`act(y + b)` | 否：一个输出样本要对 `cols` 的 `kernel / stride` 行求和 | 无 | 每个 tile 需要 `kernel / stride - 1` 行的 halo，或者换一种算法（3.2.3） |
| 逐 head norm + rotary embedding | `q, k, v = qkv(x).split(...)`；每个 head 做 `rope(rmsnorm(q_h))`，`k` 也一样 | 是，当 `BN` 是 head 大小的倍数、且没有 tile 跨越 `q`、`k` 和 `v` 时 | 无 | 需要在 tile 内做行局部的数据移动（RFC 0001 阶段 3）和分段归约（阶段 4），以及向 `q`、`k`、`v` 的带步长存储 |
| 进入下一个 GEMM 的 norm（一个 prologue） | `a = (x * rstd[:, None] * w).to(bf16)`；`gemm(a, W)` | A 操作数的变换；`rstd` 是一个已存储的按行值 | 无 | 对 GEMM 的每个列 tile 重新计算 |
| 反量化 prologue | `w = (q.to(f32) * scale).to(bf16)`，`scale` 按通道或按组 | 是 | 只有 cuBLASLt 的 FP8 按张量 scale 和块 scale | 仅当 GEMM 乘的是反量化后的操作数时才精确 |
| 量化 epilogue | 按行或按 32 元素块求 `amax`；`q = (y / scale).to(e4m3)` | 按 32 列的块：是；按行：同 norm | 带 `D_SCALE` 和 `AMAX_D` 的 cuBLASLt FP8 输出（延迟缩放） | MX 块 scale 是 tile 局部的；它们需要 RFC 0001 阶段 4 的分段归约 |

**非目标。**

* 不把两个计算算子融合进一个 kernel（GEMM 链、attention）。
* 在精确模式下不做代数改写。两个例子是把按行 scale 从 A 操作数移到输出，以及把 norm 的权重折叠进下一个 GEMM 的权重。每一个都会改变字节，RFC 0001 反对代数化简的规则依然有效。值得拥有的此类改写，会在以后的 RFC 中成为一个独立的模式。
* 不做改变字节的自动调优。在给出不同字节的路线之间做选择，从不依据计时（第 5 节）。

## 2. 数值问题与三种契约

### 2.1 如今字节一致依赖什么

把 GEMM 这个步骤写成 `y = ρ_T(acc_π(a, b) [+ bias])`：

* `π` 是已 pin 的库配置：算法、库版本和 SM 数；
* `acc_π` 是其 kernel 在最终转换之前持有的 f32 值；
* `ρ_T` 是舍入到声明的输出类型 `T`。

epilogue 是 graph 函数体的一个区域 `R`（RFC 0001），对区域的其他输入 `s`，有 `z = R(y, s)`。移植与其参考栈逐字节一致，原因有二。参考栈运行相同的 `π`（相同的 cuBLASLt 版本和一个导入的算法），所以写出相同的 `y`。而区域在 RFC 0001 的契约下复现了参考栈的 epilogue kernel。

### 2.2 融合的 epilogue 改变了什么

有三样东西可能改变，每条路线改变的是其中不同的子集：

1. **GEMM 与 epilogue 之间的舍入。** 读取累加器的融合 epilogue 计算的是 `ρ(f(acc))`，而参考栈计算的是 `ρ(f(ρ_T(acc)))`。即使是 `f(x) = x + b`，在参考栈中也舍入两次，融合后只舍入一次。X1 测量了结果不同的频率，bf16 和 f16 输出都一样：
   * bias 相加：30% 的元素；
   * 残差相加：29%；
   * bias 相加和 ReLU：15%；
   * bias 相加和 tanh GELU：31%。

   只有不带 bias 的 ReLU 不受影响，因为它与舍入可交换。
2. **GEMM 本身。** 如果融合 kernel 不是库的 kernel，`acc` 就是另一个 kernel 的累加器，对 `k` 的求和顺序也不同。顺序可能在 MMA 指令自己的 `k` 步骤、tile 的 `k` 循环、split-K 或 stream-K 归约，或者双 CTA MMA 上不同。因此，先舍入的 epilogue `ρ(f(ρ_T(acc')))` 只有在 `acc' == acc_π` 逐位相等之处才等于参考栈的结果。在 sm_103 上，X3 发现只要库的 kernel 不切分 `k`，就有 `acc' == acc_π`（第 6 节）。这是在一个架构上的测量，不是普遍成立的性质。
3. **epilogue 自己的算术。** 库 epilogue 按自己的公式计算激活，并按自己的顺序加操作数：
   * cuBLASLt 的 GELU 是 tanh 近似，其指令序列没有文档说明（X2）；
   * cuBLASLt 在 bias 之前加 `beta * C`（X1）。

| 路线 | 去掉了舍入 (1) | 改变了 GEMM (2) | 改变了 epilogue 算术 (3) |
|---|---|---|---|
| (a) 调用之后的生成 kernel（现状） | 否 | 否 | 否（RFC 0001 的区域是精确的） |
| (b) cuBLASLt epilogue | 是：每个 cuBLASLt epilogue 都在转换之前起作用 | 在 12.8.4 下否：是另一个算法，但 X1 发现累加器相同；在 13.1.0 下有时是（X6） | 仅当 epilogue 无法精确写下时（GELU） |
| (b) 带 16 位中间值的 cuDNN 融合图 | 否（X5，cuDNN 9.13） | 否（X5） | 当其逐点算子不是区域的算子时（GELU、`sin`） |
| (c) 在作为参考的 extern 的舍入之后的 hook | 否 | 否：同一份源码，已验证 | 否：区域的代码 |
| (c) 作用于累加器的 hook | 是 | 否 | 否 |
| (c) 替代库 GEMM 的生成 GEMM | 都有可能 | 在库不切分 `k` 时否（X3，sm_103）；按 bucket 测量 | 否 |
| 进入一个保持不变的 kernel 的 prologue | 否：操作数字节就是区域的输出 | 否 | 否 |

### 2.3 契约

记号：

* `P` 是 graph 函数体的声明程序（RFC 0001，第 5.1 节）；
* `U(P)` 是它的未融合组合：每个区域单独运行，每个中间值都以其声明的类型存储；
* `F(P)` 是融合后的 plan。

“精确”指在每个输出、`InOut` 和 carried buffer（携带缓冲区）上、对每个输入位模式都逐字节一致，这就是 RFC 0001 的主张（第 5.2 节）。

**(i) 精确：`F(P) == U(P)`，使用参考 GEMM。** 参考 GEMM 就是 `U(P)` 所调用的东西：已 pin 的库配置，或者一个 extern。融合后的 plan 必须逐位算出 GEMM 舍入后的结果 `y`，再对它应用 `R`。有五种做法：

* **(i-a) 默认路线**：调用之后的一个区域。这是现状，不省时间。
* **(i-b) 在作为参考的 kernel 的舍入之后的 hook。** 适用于 GEMM 或卷积是一个 extern（用户的 Triton 或 Gluon kernel）或一个 tilefish 生成的算子、并且 `U(P)` 调用的正是同一个 kernel 的情况。hook 作用于 kernel 本来要存储的值，在其舍入之后，所以 kernel 的算术保持不变：
  * 主循环的源码不变；
  * hook 的算术是区域的上下文无关代码（RFC 0001 的义务 O2 和 O3，以及隐患 H1、H8 和 H9）。

  这在构造上就是精确的，导出时的验证（D6）会检查它。
* **(i-c) 进入一个保持不变的 kernel 的任何 prologue。** prologue 用区域的代码算出 kernel 本来要从内存读取的操作数 tile。所以 kernel 乘的是相同的字节。prologue 对每个列 tile 重复执行，这是精确的：它是对一个上下文无关区域的重新计算。
* **(i-d) 经测量相等的替代 GEMM。** 生成的 GEMM `κ` 只在 `κ` 的输出等于 `π` 的输出的那些 bucket 上替代库的 `π`。这无法证明，因为库没有 IR，它的 kernel 也没有文档说明。但它可以测量。GEMM 的运算顺序不依赖于数据，所以两个按不同顺序求和的 kernel 在随机输入上几乎必然不同。因此，在一个 bucket 上对带种子的随机输入和特殊值输入相等，就是顺序相同的有力证据。这种相等按形状、库版本、pin 和工具链成立，它是*经测量精确*的，也就是 RFC 0001 的 E4 给予 bias epilogue 的状态。

  X3 测得，它成立的频率远高于预期。在 sm_103 上，一个普通的 Triton `tcgen05` GEMM 在测量的每个形状上、在每种 tile 配置下、在操作数含特殊值时、在 cuBLASLt 12.8.4 和 13.1.0 下，都等于 cuBLASLt 的 pin。它只在库的 pin 切分 `k` 时失败：cuBLASLt 13.1.0 对 32 行的问题会这样做。sm_120 和 sm_90 还没有测量（X7）。
* **(i-e) 在组合舍入之处舍入的库融合。** cuBLASLt 没有这种融合：每个 cuBLASLt epilogue 都作用于转换之前的 f32 值（头文件这样说明，X1 也测量了这一点）。不带 bias 的 ReLU 是唯一的例外，因为它与舍入可交换。cuDNN 确实有这种融合。在卷积输出为 f16 类型虚拟张量的 cuDNN 9.13 融合图中，卷积 + bias 和卷积 + bias + ReLU 与单独的卷积后接 epilogue 逐位相等（X5）。

**(ii) 累加器：`F(P) == U(R_acc(P))`。** `R_acc` 是对程序的一个声明的改写。与 RFC 0001 第 5.4 节的改写一样，它在分组之前定义在 IR 上。它适用于其算子声明了 f32 形式的调用：库 GEMM 或卷积、带累加器 hook 的 extern，或生成的算子。

* 调用结果 `y` 的每个加宽 `y.to(f32)` 都被替换为同一调用的 f32 结果：`acc`，即舍入之前的值。
* 以声明类型读取 `y` 的消费者继续读取 `ρ_T(acc)`。
* 当 `y` 还被返回、被 tap（观测点）或被另一个调用读取时，它仍然以舍入后的形式存储。

改写后的程序在每条这样的路径上少一次舍入。于是：

* 融合后的 plan 相对于改写后程序的未融合组合是精确的。在那个组合中，调用写出它的 f32 结果（相同的 kernel 配置，带 f32 输出），区域读取它。因此融合本身仍然不改变任何字节，正如 RFC 0001 所承诺的；改变字节的是对模式的选择。导出时像 D6 一样验证 `F == U(R_acc(P))`；
* 字节是 kernel（`κ`、`π` 或 extern）和区域声明的数值语义的字节。它们是确定的，计入 digest，被 plan 的 digest pin 住，并由 `jit`、`replay`、两个 host 和 CUDA graph 复现；
* 它们不是 `U(P)` 的字节，也不是任何在 GEMM 与其 epilogue 之间舍入的栈的字节；
* 它们的精度至少一样，因为同样的算术少了一次舍入；
* 当没有 hook 或库 epilogue 能接收区域时，运行的是 `U(R_acc(P))`：GEMM 写出 f32，区域读取它，这比 `U(P)` 慢。规划器会报告这一点（第 5 节），而不是悄悄保留舍入，这样 plan 的数值语义永远不取决于融合是否发生。X1 确认，这个参考对 cuBLASLt 是可观测的：同一个算法配置以 f32 的 `D` 重新编码后，cuBLASLt 接受它，对其输出舍入就逐位得到 16 位输出，在两种姿态的每次运行中都是如此。GELU 算法是例外：它们的配置无法用 f32 的 `D` 编码（3.1.1）。

**(iii) 库：由库在 pin 下定义的字节。** cuBLASLt epilogue 或 cuDNN 融合图对 pin `π` 和带属性（bias 类型、`beta`、辅助输出）的 epilogue `e` 计算 `L_{π,e}(a, b, s)`。分两种情况：

* **相对 (ii) 经测量精确。** epilogue 的算术可以写成一个区域：bias 相加、ReLU、`beta = 1` 的 `beta * C`，或者按通道的 scale。导出时在每个 bucket 上对带种子的输入和特殊值输入运行库 epilogue 和 `U(R_acc(P))`，有差异就拒绝。`U(R_acc(P))` 使用库对同一配置的 f32 形式。X1 发现 cuBLASLt 的 `BIAS`、`RELU_BIAS` 和 `beta * C` 在各 64 次运行的每个元素上都是经测量精确的。它也发现了一个 bias + `beta * C` 不精确的算法（cuBLASLt 13.1.0，第 6.1 节）。验证必须按 pin 进行，绝不能按 epilogue 进行。
* **由二进制文件定义**（RFC 0001，第 11.1 节）。算术无法精确写下，例如 cuBLASLt 的 GELU（X2）。字节是二进制文件的字节：确定且被 pin 住。导出时记录它们与 `U(R_acc(P))` 的距离，距离超过声明的容差就拒绝。融合报告会标记该区域。

在两种情况下，cuBLASLt 版本和 SM 数都像现在一样在加载时检查。canary 能区分同一版本的两个构建：它是 epilogue 在带种子的操作数上的输出的 digest，就像 cuDNN 条目所具有的那样。一个与现有栈使用相同库 epilogue、相同 pin 的 plan，与该栈逐字节一致。契约 (iii) 是移植匹配一个本身通过 cuBLASLt 融合的引擎的方式。

### 2.4 谁采用哪种契约

| 情形 | 契约 | 路线 |
|---|---|---|
| 必须匹配一个不融合其 epilogue 的栈的移植 | (i) | (a)；(i-b) 在它自己的 extern 上用 hook；(i-c) prologue；(i-d) 经测量相等之处用生成的 GEMM；(i-e) 带 16 位中间值的 cuDNN 图 |
| 对一个通过 cuBLASLt 或 cuDNN 融合的栈的移植 | (iii)，使用该栈的 pin | (b) |
| 对一个在自己的 kernel 中融合（CUTLASS）的栈的移植 | 相对该栈的 (i) | 一个复现该栈 kernel 和 epilogue 的 extern，就像现在的移植所做的 |
| 新模型，或以容差相对原始模型验收的移植 | (ii) 或 (iii) | 作用于累加器的 (c)，(b) |

没有声明，什么都不会改变字节（第 4 节）：

* 默认保持精确；
* 模式是 plan 推导依据的一部分，所以模式改变了的 graph 会被当作过期而拒绝；
* 融合报告列出模式去掉的每个舍入点。

需要某个栈的字节、又想要速度的用户有两个选择。速度可以来自保留字节的路线，(i-d) 和 (i-e) 让这类路线比预期的多。或者用户接受这些字节成为新的参考：以容差相对原始模型验证，从此被 pin 住并精确复现，就像其他任何 plan 一样。

### 2.5 什么不变

* pin 和 digest。
* `jit == replay`，以及 Rust 和 C++ host 对照 Python。
* CUDA graph 的 capture。
* plan 的 arena（内存池）及其顺序。
* 按架构的相等仍然按架构成立：`tcgen05` 上的 epilogue 和 `mma.sync` 上的同一 epilogue 是两个二进制文件，有两个 pin。
* RFC 0001 对区域本身的契约（除非声明否则不做收缩（contraction），每个收窄转换都用内联 PTX，上下文无关的代码）原样适用于 hook 内联的代码。

## 3. 路线

### 3.1 路线 (b)：库 epilogue

#### 3.1.1 cuBLASLt

下表列出 cuBLASLt 对 f16 和 bf16 操作数、f32 累加器提供的功能。X1 列中的“精确”指在每个元素上等于契约 (ii)，特殊值也包括在内。

| epilogue | 算术 | 旁路输出 | X1（sm_103，cuBLASLt 12.8.4，各 64 次运行） |
|---|---|---|---|
| `BIAS` | `acc + bias`（输出类型的 bias），一次舍入 | 无 | 精确 |
| `RELU` | `max(acc, 0)` | `RELU_AUX`：一个位掩码 | 精确，并且也等于契约 (i)：ReLU 与舍入可交换 |
| `RELU_BIAS` | `max(acc + bias, 0)` | `RELU_AUX_BIAS`：一个位掩码 | 精确 |
| `GELU`、`GELU_BIAS` | `acc [+ bias]` 的 tanh 近似 GELU | `GELU_AUX[_BIAS]`：激活前的值，已舍入 | 由二进制文件定义：与 torch 对同一累加器的 tanh GELU 相比，在 0.1 到 0.2%（bf16）和 0.8 到 1.1%（f16）的元素上不同，最多差 16 ULP（f16）；辅助输出恰好是 `ρ(acc + b)` |
| `beta * C`（`beta = 1`） | `acc + C`，一次舍入 | 无 | 精确 |
| `BIAS` 加 `beta * C` | `(acc + C) + bias`，一次舍入 | 无 | 在 12.8.4 上按该顺序精确；13.1.0 的一个算法不精确（6.1） |
| 按通道的 `alpha` 向量 | `alpha[n] * acc`（`POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST`） | 无 | 未测量 |
| FP8 scale、`AMAX_D` | 按张量和按块的 scale，一个绝对值最大值 | `amax` | 未测量 |

cuBLASLt 做不到的：

* SiLU、erf GELU、周期激活、门控激活、norm、rotary embedding 或 overlap-add；
* 与输出类型不同的 bias，所以 bias 为 f32、输出为 bf16 的 graph 匹配不上；
* norm prologue 需要的按行（token）scale：在 tilefish 的行主序布局中，cuBLASLt 的 alpha 向量沿 `N` 而不是 `M`。

* **pin。** epilogue 是启发式查询和 `cublasLtMatmulAlgoCheck` 的一部分，所以 pin 是为 epilogue 选择的。在每个形状上，GELU epilogue 得到的算法都与普通 GEMM 不同（id 70、71 和 21，而不是 66）。其他一切保持现状：从配置重新编码的 64 字节算法、版本、SM 数、姿态和导入的表，以及对字节的审核。带 bias 的 GEMM 不做 split-K 的规则推广到除 `DEFAULT` 之外的每个 epilogue，除非声明选择加入。
* **导出时验证**，按 bucket、按 pin，在带种子的输入和特殊值输入上进行。经测量精确的 epilogue 对照 `U(R_acc(P))` 检查，后者用库对同一配置的 f32 形式计算。由二进制文件定义的 epilogue 对照同一参考检查，在其声明的容差之内。算法编码了它的输出类型，所以 f32 形式就是以 f32 的 `D` 重新编码的同一配置。对 GELU 算法无法编码这种 f32 形式。于是 X1 使用普通 GEMM 的 pin 的 f32 输出，辅助输出表明其累加器在每次运行中都相同。Q6（第 8 节）建议把这作为此类算法的规则。
* **加载时**：像现在一样检查版本和 SM 数，用 epilogue 的描述符做 `cublasLtMatmulAlgoCheck`，以及一个 canary。canary 是 `D`（以及辅助输出）的 digest，操作数由 cuDNN canary 所用的 PTX 模块（`tilefish._canary`）在设备上从一个种子生成。它对每个条目是可选的，对 epilogue 条目默认写入。
* **速度不是必然的。** 在五个形状中的三个上，`GELU_BIAS` 比普通 GEMM 后接生成的 bias + GELU kernel 节省 22 到 48%。在另外两个形状上，它的 pin 是一个较慢的算法，慢了 9 到 31%（X4）。库 epilogue 会改变字节，所以规划器不能依据计时选择它。advisor 报告测得的时间，由声明来选择。
* **声明，或改写。** 库 epilogue 有两种用法：
  * 作为声明了 epilogue 的 library op（`impl="cublaslt.matmul", epilogue="gelu_bias"`）。这是一个具有库数值语义的普通调用，就像现在的 `gemm_bias`；
  * 作为把 graph 算术改写成 epilogue 的改写。规划器只在 `epilogues="library"` 下（第 4 节）、只对表中的形态、只在验证通过时这样做。否则回退到路线 (a)。

#### 3.1.2 cuDNN 融合图

cuDNN 的后端图把卷积与逐点运算融合进一个引擎：bias 相加、GELU、SiLU（通过 `SWISH`）、`SIN`、乘法或加法。这些引擎大多是运行时融合引擎，在构建 plan 时用 NVRTC 编译一个 kernel。它们覆盖了 cuBLASLt 做不到的卷积 + bias + 周期激活。X5（sm_103，f16，7 抽头的一维卷积）测得以下结果：

* **cuDNN 9.10.2**，即 tilefish 目前 pin 的版本，搭配 NVRTC 12.8 或 12.9：只有带 f32 中间值的卷积 + bias 和卷积 + bias + ReLU 有引擎，而且这些引擎是预编译的。每个运行时融合引擎在 sm_103 上都编译失败：GELU、周期激活、任何 16 位中间值、f32 输出。
* **cuDNN 9.13.0** 搭配 NVRTC 13.0：每个图都能构建。
  * 用 f32 中间值时，bias 和 bias + ReLU 与舍入后的组合在 31% 和 15.5% 的元素上不同（类似契约 (ii)）。
  * 把卷积的虚拟输出设为 f16 类型时，融合图在那里舍入。卷积 + bias 和卷积 + bias + ReLU 在全部三个形状上都与单独的已 pin 的卷积后接 epilogue **逐位相等**：契约 (i)。
  * 周期激活与 torch 的 `sin` 组合在 0.04% 的元素上相差 1 ULP；tanh GELU 在 1.1% 的元素上不同，最多差 16 ULP。

融合图的引擎配置会像卷积的一样被 pin 住（引擎索引、knob、cuDNN 版本、SM 数），并由 canary 检查。代价比 cuBLASLt 的大：

* Rust host 的 cuDNN 绑定必须从 manifest 条目构建相同的运算图，这需要在条目中加入新的节点种类；
* NVRTC 在加载时编译，这会增加启动时间，并且必须在任何 capture 之前完成；
* 运行时编译的 kernel 的字节取决于编译它的 NVRTC，也取决于 cuDNN 版本，从 9.10.2 换到 9.13 就说明了这一点。canary 能发现这种变化；版本号不能。

### 3.2 路线 (c)：tilefish kernel 中的 epilogue 和 prologue hook

#### 3.2.1 extern 上的 hook

extern 声明它的 kernel 在哪里接受 epilogue，正如 RFC 0001 第 7 节所勾勒的：

```python
@tilefish.extern(gluon=conv_kernel, ...,
                 epilogue=tilefish.hook("EPILOGUE", value="acc", out="y"),     # or value="out": after the kernel's own rounding
                 prologue=tilefish.hook("PROLOGUE", operand="x"))
def conv(x: Tensor[...], w: Tensor[...], y: Out[Tensor[...]], ...): ...
```

kernel 在其源码中本来要存储的地方调用 hook（`tile = EPILOGUE(acc, rows, cols, row_mask, col_mask, *EPILOGUE_ARGS)`），或者在加载一个操作数 tile 之后调用。没有融合时，hook 是恒等函数，extern 照旧编译。它的 digest 也不变，因为 hook 参数默认是 tilefish 提供的恒等函数。有融合时，规划器传入一个生成的 `@triton.jit`（或 `@gluon.jit`）函数，由 `tilefish.lower` 从区域生成，区域的其他输入成为该步骤的额外参数：

* **值。** `value="acc"` 传入 f32 累加器 tile（契约 (ii)）。`value="out"` 传入 kernel 舍入到其输出类型之后的 tile（契约 (i)）。两者都能提供的 extern 就两者都声明。
* **hook 可以读取的**：tile、它的行列索引、按行、按列和按元素的旁路输入（带步长，通过现有的 `Strided[...]` 和 `StrideOf` 机制），以及设备标量。
* **hook 可以写入的**：主输出（它把主输出返回给 kernel），以及旁路输出（它自己存储），每个都按 tile 的元素写入。kernel 对主输出保留自己的存储路径（TMA 或指针）。旁路输出的一个例子是 add + norm 的残差流。
* **Gluon。** Gluon kernel 把累加器保存在张量内存中，并以它选择的布局加载到寄存器。生成的 hook 以 `constexpr` 接收该布局，并在对应的切片布局中构建它的索引向量。把 epilogue 拆成子 tile 的 Gluon kernel 对每个子 tile 调用一次 hook。
* **标识。** extern 与区域的每个配对都是一个新算子（`<extern>+<region hash>`），有自己的条目和 pin。区域的内容哈希在它的名称中，与融合区域的哈希一样（RFC 0001，第 4.2 节）。如果导出时的验证拒绝了一个配对，该配对回退到路线 (a)，原因写在融合报告中。

这是能省时间的最便宜的路线。卷积已经是 Gluon extern 的移植，无需把 bias + 激活 epilogue 手写进 kernel，就能让它们被融合。当 hook 接收已舍入的值时，融合遵守契约 (i)。

#### 3.2.2 生成的 GEMM 和卷积算子

对于现在是库调用的 GEMM，路线 (c) 需要 tilefish 自己的 GEMM。它会是一个生成的算子，就像 `tilefish.contiguous` 和 `im2col` 一样：tilefish 按架构写出它的源码，规划器填充它的 epilogue 和 prologue hook。

| 架构 | MMA | kernel 结构 |
|---|---|---|
| sm_100, sm_103 | `tcgen05.mma`（`kind::f16`），累加器在张量内存中 | Gluon（或 Triton）：TMA 加载、warp 特化的流水线、持久化的数据并行 tile 顺序（字节与 SM 数无关）、与下一个 tile 的主循环重叠的 epilogue 子 tile，以及作为以后选项的双 CTA MMA |
| sm_120 | `mma.sync.m16n8k16` | Triton `tl.dot` 或 Gluon，TMA 加载 |
| sm_90 | `wgmma.mma_async` | Triton `tl.dot` 或 Gluon |

* **数值语义。**
  * 累加用 f32。
  * 默认不做 split-K，也不做 stream-K。它们的归约顺序会取决于切分数或 SM 数，而一个持久化 extern 只有在声明了的情况下才可以依赖 SM 数（manifest schema 3，第 4.3 节）。
  * 对 f32 操作数，现有的守卫（`tilefish._tf32guard`）会拒绝 TF32。曾观察到 Triton 把 IEEE f32 的 `tl.dot` 编译成 TF32 的 `tcgen05` MMA。
  * X3 测得，在 sm_103 上，tile 配置和 kernel 都不改变字节：全部 99 种配置写出相同的字节，并且只要 cuBLASLt 不切分 `k`，这些字节就是 cuBLASLt 的字节。于是按 bucket 的配置只是速度上的选择，就像逐元素区域的配置一样（RFC 0001，O5）。这必须按架构测量，不能假定。
* **按 bucket 的配置** 通过现有的 `by_bucket` 机制实现：tile 大小、stage、warp、CTA 和 cluster。它在导出时按固定规则选择，或从记录的表中读取（类似 `AlgoTable`），绝不悄悄地自动调优。每个二进制文件都像任何 Triton 条目一样被 pin 住。
* **SASS 断言。** 生成器在它自己的 kernel 上声明断言：该架构的 MMA（`tcgen05.mma kind::f16`、`HGMMA` 或 `HMMA.16816.F32`）、`no_tf32` 和 `no_spills`。一个新的结构断言能让“hook 没有改变 GEMM”在构建时可见：带 hook 的二进制文件的 MMA 指令序列等于不带 hook 的二进制文件的序列（操作码、形状、kind 和累加标志，按顺序）。它是必要条件，不是证明，字节验证仍然是守卫。
* **实现的选择。** GEMM 的声明说明它是哪种实现（`impl="cublaslt.gemm"` 或 `impl="tilefish.gemm"`）。规划器绝不会自行把库 GEMM 换成生成的 GEMM。契约 (i-d) 是一个显式的选择加入（Q4）：一个声明为等于其所替代的库 pin 的生成 GEMM。导出时在每个 bucket 上验证这一点，在不成立的 bucket 上保留库 GEMM，走路线 (a)。
* **风险在于速度。** 一个比它所替代的已 pin 的库 GEMM 更慢的生成 GEMM，花掉的时间可能比 epilogue 省下的还多：
  * X3 测得，一个直接的 Triton `tcgen05` GEMM（99 种配置中最好的）在一个 30 SM 切片的五个形状上，用时是 cuBLASLt 的 1.22 到 1.44 倍；
  * X4 测量了该 kernel 中的一个朴素 epilogue，它不是 warp 特化的，旁路输入采用累加器的布局。在每个形状上，它都比库 GEMM 后接一个单独的 epilogue kernel 慢。

  对测得的比值 `r`，带生成 GEMM 的路线 (c) 在 `saving > (r - 1) * T_gemm` 时才值得。要做到这一点，需要上表中的 kernel 结构（X8）。

#### 3.2.3 epilogue tile 能计算什么

* **lane 局部的工作，以及按行或按列的输入**（bias、激活、残差相加、反量化 scale、类型转换）：任何 tile 形状都可以。
* **tile 内的分段**（逐 head 的 norm、跨 32 列的 MX 块 scale、一个 head 内的 rotary 对）。tile 的列必须容纳完整的分段：`BN` 是分段的倍数，并且没有 tile 跨越 `q`、`k` 和 `v` 之间的边界。归约需要一个顺序，因为无序归约要单独运行（RFC 0001，D2）。所以这要等待有序分段归约（0001 阶段 4）和行局部的数据移动（阶段 3）。
* **整行**（覆盖隐藏维度的 norm、按行的量化 scale）。只有当 `N <= BN` 时，一个 tile 才容纳整行，`tcgen05` 允许最多 256 到 512 列（它的张量内存每个 SM 容纳 128 x 512 个 f32 值）。更大的行必须拆分：
  * epilogue 存储旁路输出和每个 tile 的部分和（`[M, N / BN]`，对 tile 的列做有序求和）；
  * 下一个 GEMM 的 prologue 或一个小 kernel 按 tile 顺序对部分和求和，完成 norm。

  只有当区域声明的恰好是这个顺序（对 `BN` 大小的 tile 做 `tilefish.sum(order=...)`）时，这相对 `U(P)` 才是精确的。graph 可以声明它，因为 tilefish 的顺序是显式的。norm 对 `h` 本身的舍入遵循契约 (i) 或 (ii)，如上所述。
* **halo**（作为 GEMM + col2im 计算的转置卷积之后的 overlap-add）。一个输出样本要对 GEMM 结果的 `kernel / stride` 行求和。重新计算 `kernel / stride - 1` 个额外行的 tile 无需原子操作就能完成其输出；重新计算的量就是 GEMM 的这一比例。tile 按 col2im 的顺序加上各部分贡献，再应用 bias 和激活。在 (ii) 下，这相对于同样顺序的 `U(R_acc(P))` 是精确的。多相直接卷积则会把 overlap-add 折叠进 `k` 循环：这是另一种求和顺序，因而是另一种算法，在这些契约之外。

#### 3.2.4 prologue

prologue 计算一个操作数 tile，而不是加载它：norm 的 `(x * rstd * w).to(bf16)`，或者反量化 `(q.to(f32) * scale).to(bf16)`。使用 `tcgen05` 和 `wgmma` 时，操作数由 TMA 从全局内存送到共享内存，MMA 直接读取共享内存。因此一个变换需要以下两者之一：

* 在它自己的 warp 上经寄存器往返一次：加载 tile、变换、存回并做 fence；
* 一个从寄存器（`mma.sync`、`wgmma`）或从张量内存（`tcgen05`）读取操作数的 MMA。

这些工作对每个列 tile 都要重复，每个元素重复 `N / BN` 次。在 `BN = 256` 时，这大约是 MMA 运算量的 0.6%，但它在 CUDA core 上执行，所以必须与 MMA 重叠。

数值语义是容易的部分。操作数字节就是区域的字节，所以只要 kernel 保持不变（i-c），prologue 就保持契约 (i)；而在 sm_103 上，X3 表明生成的 kernel 可以得到库的位。库 GEMM 不接受 prologue（FP8 scale 因子除外），所以进入库 GEMM 的 prologue 意味着要用生成的 GEMM，带有路线 (c) 的速度风险。

### 3.3 每条路线的价值与代价

| 路线 | 节省 | 速度风险 | 覆盖 | 工程代价 |
|---|---|---|---|---|
| (a) 调用之后的生成 kernel | 无（现状） | 无 | 全部 | 已完成 |
| (b) cuBLASLt epilogue | `y` 的往返和一次启动，而 GEMM 仍是库的：在测量的形状上为 GEMM + epilogue 的 9 到 48% | epilogue 的 pin 可能是较慢的算法（GELU：在 5 个形状中的 2 个上比未融合慢） | bias、ReLU、tanh GELU、残差相加、按通道 scale、FP8 | 小到中：绑定、两个 host 中的新库 impl、验证、canary |
| (b) cuDNN 融合图 | 同上，用于卷积 | 引擎选择，加载时的 NVRTC；sm_103 上需要 cuDNN 9.13+ | 卷积 + bias + 逐点链，带 16 位中间值时为契约 (i) | 中到大：两个 host 中的图构建、运行时编译 |
| (c) extern 上的 hook | 往返和一次启动 | 无：kernel 是用户的 | extern 是什么就覆盖什么 | 中：hook ABI、规划器、生成 hook、验证 |
| (c) 生成的 GEMM 或卷积 | 往返和一次启动，外加没有库提供的形态，经测量相等之处为契约 (i) | 直接的 Triton GEMM 的 `r` 为 1.22 到 1.44；朴素的融合 epilogue 会亏 | 所有 tile 局部的形态 | 大：每个架构一族 kernel、按 bucket 的配置、性能追平工作 |
| prologue | 操作数的往返 | 变换必须与 MMA 重叠 | norm、反量化 | 大（生成 GEMM 的一部分） |

advisor 的估计对各形态做了排序（第 0 节）：

* 卷积模型的 GEMM 和卷积 epilogue（带周期激活和 overlap-add）有最大的绝对节省。cuBLASLt 不覆盖它们；cuDNN 融合图（用于卷积，cuDNN 9.13）和路线 (c) 覆盖它们。
* transformer 模型的 add + norm、门控和 GELU epilogue 部分由 cuBLASLt 覆盖（bias + GELU、残差相加），完全由路线 (c) 覆盖。

## 4. 表层语法

一个草图；名称曾是问题 Q8，已按提案决定。

```python
@tilefish.graph(fusion=tilefish.fusion())                              # epilogues="exact": the default, contract (i) only
@tilefish.graph(fusion=tilefish.fusion(epilogues="accumulator"))       # contract (ii): R_acc, hooks on the accumulator
@tilefish.graph(fusion=tilefish.fusion(epilogues="library"))           # (ii) and (iii): library epilogues where they exist
with tilefish.fused(epilogues="accumulator"):                          # per region, as RFC 0001's scopes
    h = (gemm_up(x, w).to(f32) + b.to(f32))

@tilefish.library("gemm_gelu", impl="cublaslt.matmul", epilogue="gelu_bias")   # a declared library epilogue: an ordinary call
@tilefish.library("gemm", impl="tilefish.gemm", acc=f32)                       # a generated GEMM, with hooks the planner may fill
@tilefish.library("gemm", impl="tilefish.gemm", acc=f32, equal="cublaslt")     # (i-d): must equal the cuBLASLt pin, bucket by bucket
```

* `epilogues="exact"` 是默认值。它只融合保持 `U(P)` 字节的东西：
  * extern 和作为参考的生成算子上 `value="out"` 的 hook；
  * 进入保持不变的 kernel 的 prologue；
  * 在 `U(P)` 舍入之处舍入的库融合。

  它默认开启，就像 RFC 0001 的阶段 1（D1）一样，因为它不可能改变任何字节。导出时验证每一个这样的融合。
* `epilogues="accumulator"` 应用 `R_acc`，并使用 `value="acc"` 的 hook。
* `epilogues="library"` 增加改写为库 epilogue 的改写。
* `tilefish.barrier(y)` 让 `y` 保持存储，并阻止对它的任何 epilogue 融合，与 RFC 0001 中一样。

## 5. 与 plan、报告、advisor、SASS 断言、trace、host 和 schema 的交互

* **plan。** 带 hook 的 extern 或生成的 GEMM 是一个生成条目的 `kernel` 步骤，区域的输入作为额外参数。库 epilogue 是一个 `library` 步骤，其参数遵循条目的 ABI（bias、辅助输出、`C`）。plan schema 1 不变。
  * 被融合掉的结果不再有缓冲区，所以 arena 会缩小。
  * 通过 `beta * C` 的残差相加可以原地更新残差流（`D == C`，manifest schema 3 第 5.4 节的 `alias`）。
  * hook 可以在 RFC 0001 的规则 4 下写入 `InOut` 或 carried buffer，该规则只允许对旧版本做 lane 局部的读取。
* **tap 和 `run_step`。** 被融合掉的 GEMM 结果没有缓冲区。`taps=` 会拒绝它，并建议加上 `tilefish.barrier`，或者 tap `U(P)`（或 `U(R_acc(P))`）；在契约下，它们的中间值就是融合 kernel 计算的值。
* **`reference` 模式** 解释执行 `R_acc(P)`。调用的 f32 结果是 torch 的 f32 GEMM 或卷积，和现在一样在库的容差之内。
* **CUDA graph。** 不变。cuDNN 运行时融合引擎在加载时、在任何 capture 之前编译。
* **融合报告**（`<graph>.fusion.json`）为每个融合进调用的区域增加以下字段：
  * 调用和路线（`library`、`hook`、`prologue`）；
  * hook 的值（`acc` 或 `out`）；
  * 契约（`exact`、`accumulator`、`measured-exact`、`binary-defined`）；
  * `R_acc` 去掉的舍入点；
  * 库 epilogue 及其 pin；
  * 验证的结果：相等，或者对由二进制文件定义的 epilogue 给出最大的 ULP 距离。

  请求了但没有发生的融合会连同原因一起列出，在 `accumulator` 下还会列出它所需的 f32 中间值。
* **advisor** 把 `epilogue-library` 和 `prologue-library` 从未来工作变成行动。对每个候选，它报告在 graph 声明的模式下能接收它的路线、每条路线给出的契约，以及节省。
  * 对生成的 GEMM，它还报告盈亏平衡的减速 `saving / T_gemm`：GEMM 最多能慢多少，融合才不亏。它不预测生成 GEMM 的速度。
  * 需要 graph 没有声明的模式的候选会说明这一点，例如 "exact: none; accumulator: hook on the extern, 150 us"。
  * 形态按路线分组，这样报告能说明哪条路线解锁得最多。
  * 给出不同字节的路线永远不会被 advisor 或规划器根据耗时选择；advisor 测得的时间为声明提供参考。
* **SASS 断言**（`tilefish.sass`）。带 hook 的二进制文件与任何构建一样被评判。生成的 GEMM 声明它们自己的断言（MMA kind、`no_tf32`、`no_spills`）。3.2.2 的 MMA 序列断言是新的。
* **trace**（`trace/`）。hook 的生成源码带有 graph 函数体的行（区域的 `loc`），所以 `tilefish-explain annotate` 会把 epilogue 的指令归到写出它们的 graph 函数体行。kernel 内插桩（issue #5，第 2 节）可以对融合 kernel 的 epilogue 阶段计时。
* **host 和 C ABI。** 路线 (c) 对两个 host 都没有要求：带指针的 kernel 步骤、TMA 描述符（`tensordesc`）和全局 scratch 都已经存在。路线 (b) 需要：
  * Python 中（`tilefish._cublaslt`、`_library`）和 crate 中（`ffi/cublaslt.rs`、`gemm.rs`、`library.rs`）的新库 impl；
  * canary；
  * 在 C ABI 中，为 GEMM 的 epilogue 及其额外张量增加一个访问器（ABI 次版本）。C++ 门禁通过 crate 覆盖它。

  cuDNN 融合图需要两个 host 中都有图构建器。
* **schema。** 路线 (c) 什么都不改变：它的条目是生成的 `triton` 条目。路线 (b) 是增量的：一个新的库 impl（`cublaslt.matmul`），带有一个取值更多的 `epilogue` 字段、`aux` 和 `beta`，以及可选的 `canary`。
  * 两个现有的 impl 逐字节保持原样，所以现存的每个 manifest 都能继续加载。
  * 早于新 impl 的读取器会拒绝含有它的条目。
  * 这一变化记录在 `docs/manifest-schema-3.md` 第 9 节，Python 写入器、两个读取器和向量同步更新。
  * 模式不需要 manifest 字段（同 RFC 0001 Q8）：它们在每个生成源码的文件头和融合报告中。

## 6. 实验

### 6.1 阶段 0 结果（2026-11-16，sm_103）

每个实验的环境：

* 一个 B300 MIG 切片：sm_103，30 个 SM；
* torch 2.9.1、Triton 3.5.1 和 ptxas 12.9；
* 除非另有说明，cuBLASLt 为 12.8.4；
* tilefish 的 cuBLASLt 绑定，在实验脚本中扩展了每个 epilogue 的描述符属性；
* 两种启发式姿态：`tilefish`（对除 `DEFAULT` 之外的任何 epilogue 都要求不做 split-K）和 `torch`；
* 时间是由 20 次背靠背调用组成的 CUDA graph 上的中位数。

脚本不在仓库中；每个实验都给出了足以重复它的信息。

**X1：哪些 cuBLASLt epilogue 等于一个组合。**

矩阵：

* 16 个形状：`M` 取 {32, 128, 640, 2048}，乘以 `N x K` 取 {1536 x 512, 3072 x 1024, 1024 x 3072, 4096 x 4096}；
* bf16 和 f16 的操作数与输出；
* 两种姿态，所以每个 epilogue 64 次运行。

输入：`A ~ N(0, 1)`，`B ~ N(0, 1/K)`，这样累加器的量级为 1，GELU 能看到它弯曲的部分。bias 为 `N(0, 0.25)`，`C` 为 `N(0, 1)`。两者都有八个特殊列：`+0`、`-0`、`+inf`、`-inf`、NaN、两种符号的最大有限值，以及一个次正规数。

每个 epilogue 都用它自己的 pin 运行，并与两个参考逐位比较：

* **(ii)**：`ρ(epi(acc))`，由 torch 以 f32 计算，其中 `acc` 是同一算法配置以 f32 的 `D` 重新编码后的 f32 输出（对于无法这样编码的 GELU 算法，用普通 GEMM 的 pin 的 f32 输出）；
* **(i)**：`ρ(epi(ρ(acc)))`，其中 `ρ(acc)` 是同一配置（或普通 pin）的 16 位普通输出。

| epilogue | 等于 (ii) | 与 (i) 不同的元素 | 说明 |
|---|---|---|---|
| `DEFAULT` | 64/64 次运行 | 0 | f32 重新编码的输出舍入后，在 64/64 次运行中都是 16 位输出：累加器是可观测的 |
| `BIAS` | 64/64，包括特殊值 | 30.3 到 31.0% | 在 16 个形状中的 14 个上，pin 就是普通 GEMM 的 pin；在另外两个上字节相同 |
| `RELU` | 64/64 | 0 | 与舍入可交换 |
| `RELU_BIAS` | 64/64 | 15.1 到 15.6% | |
| `beta * C` | 64/64 | 28.9 到 29.6% | |
| `BIAS` + `beta * C` | 按 `ρ((acc + C) + b)` 为 64/64；按 `ρ((acc + b) + C)` 只有 6/64 | 26.2 到 26.7% | 库的顺序 |
| `GELU` | 0/64：0.11 到 0.14% 的元素不同（bf16），0.81 到 0.94%（f16），最多差 16 ULP（f16） | 26.8 到 28.6% | 在 16/16 个形状上与普通 GEMM 的算法不同 |
| `GELU_BIAS` | 0/64：0.18 到 0.21%（bf16），1.02 到 1.11%（f16） | 31.1 到 31.6% | |
| `GELU_AUX_BIAS` | `D` 同 `GELU_BIAS`；辅助输出在 64/64 中等于 `ρ(acc + b)` | `D` 与 `gelu(aux)` 在 28 到 30% 上不同 | GELU 作用于未舍入的激活前值；GELU kernel 的累加器就是普通 GEMM 的累加器 |

X1 的每个 pin 的 split-K 都是 1，所以 X1 对带 epilogue 的 split-K 什么也说明不了。

**X6：cuBLASLt 13.1.0 下的 X1**（torch 2.9.1+cu130，相同的矩阵）。X1 的结论成立，有三项发现：

* `tilefish` 姿态现在在 `M = 32`、`N x K` 为 1024 x 3072 和 4096 x 4096 时，为普通 GEMM 和 `beta * C` pin 了 split-K（3 或 4，使用 f32 部分和）。这些 pin 仍然等于 (ii)（用同一配置的 f32 输出计算）。
* 算法 21 在 32 x 1024 x 3072 上的 `BIAS` + `beta * C`，在两种输出类型下都在 30% 的元素上与两种顺序都不同。其他 15 个形状是精确的。库 epilogue 的算术是 pin 的性质，所以验证必须按 pin 进行。
* GELU 的距离变大（最多 2.4% 的元素，f16 中 47 ULP）。GELU kernel 的辅助输出现在每组 16 个形状中有 1 或 2 个与 `ρ(acc + b)` 不同：它们的累加器不再总是普通 pin 的累加器。

**X2：cuBLASLt 的 GELU 计算的是什么。** 我们在普通 GEMM 的算法上用 f32 的 `D` 运行 GELU epilogue（形状 640 x 3072 x 1024 和 2048 x 4096 x 4096），使其输入就是普通 GEMM 的 f32 输出。我们把它与候选公式比较，每个候选都写成标量 PTX（`mul.rn`、`add.rn`、`fma.rn`、`tanh.approx.f32`），这样任何编译器收缩都不会把它们弄模糊：

| 候选 | 相等的元素 |
|---|---|
| torch 的 tanh GELU | 6.8% |
| torch 的 erf GELU | 2.7% |
| 64 个变体中最好的（8 种内层形式、4 种外层形式、常数的 2 种舍入）：`0.5 x (1 + tanh.approx(k x fma(c, x^2, 1)))`，其中 `k = f32(sqrt(2/pi))`，`c = f32(0.044715)` | 99.80% |

剩下 0.2% 的元素 `|x|` 在 0.07 到 4.5 之间，相对差的中位数为 1.7e-6，最大为 2.1e-2。所以 cuBLASLt 的 GELU 是由二进制文件定义的：接近一个已知公式，但不等于所尝试的任何一个。

**X3：Triton GEMM 对照已 pin 的 cuBLASLt GEMM。**

kernel：TMA 描述符、编译到 `tcgen05`（`kind::f16`）的 `tl.dot`、分组的 tile 顺序，没有 split-K。

扫描：在五个形状上测试 99 种配置（`BM` 为 64 或 128，`BN` 为 64 到 256，`BK` 为 64 或 128，2 到 5 个 stage，4 或 8 个 warp，带或不带 warp 特化的循环）。

字节：

* 每种配置写出的字节都与其他配置相同（每个形状上 99/99）。
* 在两种姿态下，这些字节在每个元素上都是 cuBLASLt pin 的字节。

X3b 把比较扩展到另外六个形状（32 x 1024 x 3072、32 x 4096 x 4096、bf16 和 f16 的 640 x 3072 x 1024、f16 的 2048 x 4096 x 4096，以及 100 x 1000 x 1000）和三种配置，操作数既有随机值，也有特殊值。特殊值包括次正规数、下溢的乘积、溢出到无穷的行、`+inf`、`-inf`、带载荷的 NaN，以及 `-0`。

* 在 cuBLASLt 12.8.4 下，输出处处相等。
* 在 13.1.0 下，只要 pin 没有 split-K，输出就相等。pin 有 split-K 时，在 0.19 到 0.20% 的元素上不同（32 768 个中的 63 个，131 072 个中的 259 个，即 32 行的形状）。

X1 的 GELU kernel（算法 70、71 和 21）也具有普通 GEMM 的累加器。综合起来：在 sm_103 上，只要没有人切分 `k`，这些 `kind::f16` GEMM 的字节就取决于沿 `k` 的求和顺序，而不取决于 tile、算法或库。这是在 11 种形状与类型组合、两个 cuBLASLt 版本上的测量；sm_120 和 sm_90 没有测量。

速度（微秒），与 X4 并列（`r` = Triton / cuBLASLt）：

| 形状 `M x N x K` | cuBLASLt | Triton（最佳配置） | `r` |
|---|---|---|---|
| 640 x 4096 x 1024, bf16 | 18.9 | 23.2 | 1.24 |
| 640 x 1024 x 4096, bf16 | 15.0 | 21.4 | 1.43 |
| 2048 x 3072 x 1024, bf16 | 39.6 | 48.3 | 1.22 |
| 128 x 3072 x 1024, bf16 | 4.4 | 6.4 | 1.44 |
| 8192 x 3072 x 768, f16 | 121.3 | 152.8 | 1.26 |

**X4：GEMM 后接 bias + tanh GELU、以及 GEMM 后接残差相加的各条路线**（微秒，X3 的形状）。

bias + GELU 的路线：

* **(a)** 是已 pin 的 GEMM 后接一个生成的逐元素 kernel；
* **(b)** 是 `GELU_BIAS`；
* **(c, acc)** 是 epilogue 作用于累加器的 Triton GEMM；
* **(c, rounded)** 是同一 kernel，epilogue 在舍入到输出类型之后应用。

| 形状 | (a) | (b) | (c, acc) | (c, rounded) | 残差 (a) | 残差：`beta * C` |
|---|---|---|---|---|---|---|
| 640 x 4096 x 1024 | 31.2 | 24.2 | 33.4 | 35.7 | 24.3 | 20.5 |
| 640 x 1024 x 4096 | 18.9 | 24.8 | 29.1 | 30.2 | 17.0 | 15.5 |
| 2048 x 3072 x 1024 | 68.4 | 40.8 | 75.6 | 82.2 | 60.2 | 44.4 |
| 128 x 3072 x 1024 | 7.1 | 7.7 | 10.3 | 10.8 | 5.9 | 4.9 |
| 8192 x 3072 x 768 | 231.8 | 120.2 | 239.8 | 264.6 | 206.8 | 134.8 |

* **库的残差相加**（`beta * C`）在每个形状上都胜过 (a)，快 9 到 35%。
* **`GELU_BIAS`** 在三个形状上胜过 (a)，快 22 到 48%。它在两个形状上落败，那里它的 pin 是算法 71，而普通 GEMM 的是 66。
* **朴素的 Triton epilogue** 处处落败：它不是 warp 特化的，它用 `exp` 和一次除法实现 `tanh`，并以累加器的寄存器布局加载旁路输入。采用该布局的 Triton 残差相加比 (a) 慢 1.3 到 2.2 倍，不在表中。
* **(c, rounded) 在全部五个形状上与 (a) 逐位相等。** 这在实践中就是带生成 GEMM 的契约 (i)，只是速度还不划算。

**X5：cuDNN 融合图。** 这些图：卷积 + bias、+ ReLU、+ tanh GELU，以及 + 周期激活（`h + sin(a h)^2 / a`），每个都分别带 f32 中间值和 f16 中间值。卷积是 7 抽头的一维卷积，以二维形式表达，NHWC，f16：`N x C x L` 为 1 x 256 x 4096、1 x 512 x 2048 和 4 x 128 x 8192，`K = C`。我们使用 cuDNN 的 Python 前端 1.32.3。结果见第 3.1.2 节。时间没有测量：测试框架没有把前端的启动 capture 进 CUDA graph（X13）。

**结果改变了什么。** 在这些测量之前，融合 epilogue 的契约 (i) 看上去只能通过 extern 上的 hook 实现。现在：

* 在 sm_103 上，只要库不切分 `k`，生成的 GEMM 就可以相对库栈精确（i-d）；
* 带 16 位中间值的 cuDNN 9.13 融合图对 bias 和 ReLU 是精确的（i-e）；
* cuBLASLt 的 bias、ReLU 和残差 epilogue 相对 (ii) 经测量精确；
* 生成路线的代价是 GEMM 的速度，而不是它的数值语义。

### 6.2 仍需要的实验

| | 实验 | 决定什么 | 需要 |
|---|---|---|---|
| X7 | 在 sm_120（`mma.sync`，cuBLASLt 的 f16 split-K pin）和 sm_90（`wgmma`）上做 X1、X3 和 X3b | (i-d) 在 sm_103 之外是否成立 | 一台 sm_120 设备和一台 sm_90 设备 |
| X8 | 一个 Gluon `tcgen05` GEMM：持久化、warp 特化、epilogue 子 tile 与下一个 tile 重叠、旁路输入用 TMA 加载、双 CTA MMA；在一个固定的形状测试集上对照 cuBLASLt，分别带和不带契约 (i) 和 (ii) 下的 bias + GELU 与残差 epilogue。**已完成**（见“X8：结果”），但没有双 CTA MMA，Triton 3.5.1 的 Gluon 无法表达它 | E3 是否值得构建（Q5） | sm_103 |
| X9 | 一个复现 cuBLASLt split-K 顺序的生成 split-K GEMM（f32 部分和放在 workspace 中，按固定顺序归约） | (i-d) 能否覆盖 split-K pin | sm_103 |
| X10 | X8 kernel 中的 norm prologue（`x * rstd * w` 进入 A 操作数），对照单独的 kernel + GEMM | E5 | sm_103 |
| X11 | 带部分和拆分和声明的 tile 顺序的 add + norm；带 halo 的 overlap-add；对照它们未融合的 plan | E4 | sm_103, sm_120 |
| X12 | 在一个真正的 Gluon 隐式 GEMM 卷积 extern 上的 `out` hook 和 `acc` hook：在每个 bucket 上，带 hook 的二进制文件对照不带 hook 的二进制文件后接区域；MMA 序列断言 | E1 | sm_103, sm_120 |
| X13 | cuDNN 融合图：capture 句柄的流时的耗时、canary 的代价、加载时 NVRTC 的编译时间，以及字节是否随 NVRTC 版本变化 | Q11 | sm_103 |
| X14 | 带 `alpha` 向量和 FP8 scale 的 cuBLASLt epilogue；`GELU_AUX`；带 epilogue 时的 split-K 规则（`torch` 姿态） | E2 的范围 | sm_103, sm_120 |

## 7. 分阶段计划、里程碑和测试

### 7.1 里程碑

| 里程碑 | 内容 | 完成标准 |
|---|---|---|
| E0：实验 | X1 到 X6（第 6.1 节，已完成）；X7 到 X14（6.2），各自在需要它的里程碑之前完成 | 第 8 节中每个依赖实验的决定都有了它的数字 |
| E1：extern 上的 hook | `tilefish.hook`（`value="out"` 和 `"acc"`）；从区域生成的 hook 函数（Triton 和 Gluon，布局作为 `constexpr`）；规划器把区域合并进带 hook 的调用（lane 局部的工作、按行和按列的输入、按元素的旁路输出、规则 4 下的 `InOut`）；`epilogues="exact"` 默认开启，`"accumulator"` 带 `R_acc`；导出时对每个配对对照 `U(P)` 或 `U(R_acc(P))` 验证，失败时回退到路线 (a)；融合报告的新字段；advisor 对 extern 候选给出的路线和契约 | 一个 Triton 和一个 Gluon 的 GEMM 或卷积 extern 能融合 bias + GELU、bias + 周期激活，以及带旁路输出的残差相加，在 (i) 和 (ii) 下，在 sm_103 和 sm_120 的每个 bucket 上都与其参考逐字节一致；Rust 和 C++ 门禁运行融合后的 plan，既启动也 capture |
| E2：cuBLASLt epilogue | `cublaslt.matmul` impl（bias、ReLU、GELU 及其辅助输出、`beta * C`）、它的 canary、两个 host 以及 C ABI 访问器；对经测量精确的形态启用 `epilogues="library"`；GELU 作为由二进制文件定义的 epilogue，带容差；按 pin 验证 | X1 的矩阵在 sm_103 和 sm_120 上、两种姿态下成为 GPU 测试；每个改写在导出时都经过验证；伪造的 canary 在加载时被拒绝 |
| E3：生成的 GEMM | `impl="tilefish.gemm"`：用于 sm_100 和 sm_103 的 X8 kernel，带 hook、按 bucket 的配置、SASS 断言和 MMA 序列断言；`equal="cublaslt"`（i-d），按 bucket 验证；然后是用于 sm_120 的 `mma.sync` 和用于 sm_90 的 `wgmma`（在 X7 之后）；在相同模板上用隐式 GEMM 实现卷积 | 字节不依赖于配置，这在每个架构上都是一个测试；在一个固定的形状测试集上，生成 GEMM 与已 pin 的库 GEMM 的比值在评审设定的范围内（Q5）；advisor 的盈亏平衡点说明在哪里值得使用 |
| E4：行、分段和 halo | 逐 head 的 norm + rotary（在 tile 内用 RFC 0001 的阶段 3 和 4）；带旁路输出的 add + norm（先是整行，然后是以声明的顺序跨 tile 拆分）；双累加器的门控激活；带 halo 的 overlap-add | 第 1 节的每个用例都被融合，在 (i) 或 (ii) 下等于其参考，并且在 sm_103 和 sm_120 上比其未融合的 plan 快 |
| E5：prologue 和量化 | 进入生成 GEMM 的 norm 和反量化 prologue；MX 块 scale 和按行的量化 epilogue | 同 E4 |
| E6：cuDNN 融合图 | 如果 X13 证实了速度：带逐点链的 `cudnn.conv_fwd`，契约 (i) 所需的 16 位中间值，sm_103 上要求 cuDNN 9.13 或更高版本 | 同 E2，用于卷积 |

E1 不需要其他任何里程碑。它服务于卷积已经是 extern 的那些移植模型。E2 独立于 E1。E3 是最大的一个，对于现在是库调用的 GEMM，它是 E4 和 E5 的前提。

### 7.2 测试

* **构造（无 GPU）。**
  * `R_acc`，IR 输入、IR 输出：调用结果的加宽被替换；16 位消费者、被返回或被 tap 的结果，以及扇出，都保留舍入后的值。
  * 规划器的合并与拒绝：barrier、扇出、跨 tile 的归约、对正在写入的 `InOut` 的广播读取、两个步骤之间的调用。
  * extern 与区域配对的名称和标识：两个 graph 中的同一区域是一个算子。
  * hook 的源码等于区域下沉后的源码。
  * 融合报告和 advisor 的标签。
  * 新库 impl 的 manifest 向量，在 Python 和 Rust 读取器中。
  * 对格式错误的 hook 声明的拒绝。
* **对照参考（GPU，核心）。** 每个融合后的 plan 都在 sm_103 和 sm_120 的每个 bucket 上，与 `U(P)`（契约 (i)）或 `U(R_acc(P))`（契约 (ii)）逐字节比较。输入是带种子的随机值，以及每个旁路输入中的特殊值：两种零、无穷、带非规范载荷的 NaN、次正规数。plan 是作为 graph 的第 1 节用例，以及融合进一个 Triton 和一个 Gluon GEMM extern 的随机区域（RFC 0001 的生成器）。
* **库 epilogue。**
  * 把 X1 作为测试：每个 epilogue 对照其经测量精确的参考或其容差，按姿态、输出类型和 pin。
  * canary 在两个 host 中的计算方式相同。
  * 一个不同的构建（用伪造的 digest 模拟）会被拒绝。
* **生成的 GEMM。**
  * 一个 bucket 的每种配置之间字节都相等（X3 的检查）。
  * `equal="cublaslt"` 在 pin 没有 split-K 时成立，在 pin 有 split-K 时被拒绝（X3b 的情况）。
  * `no_tf32` 在 f32 操作数上成立。
  * MMA 序列断言成立，带 hook 对照不带 hook。
  * TF32 守卫。
* **host。** Rust 门禁和 C++ 门禁获得每条路线的融合 plan：启动、在 CUDA graph 中、带 shared arena（共享内存池）和 carried buffer。Rust 和 C++ 与 Python 逐字节比较。
* **速度（仅供参考）。** 在 `python/tests` 中的合成模型上，融合之前用 advisor 的估计，融合之后用 profiler 的测量。移植模型的测量在私下进行。

### 7.3 E2 的实现情况（2026-11-17）

**impl `cublaslt.matmul`**（Q7；`docs/manifest-schema-3.md`，第 9.5 节）。`@tilefish.library(..., impl="cublaslt.matmul", epilogue=..., beta=...)`：`epilogue` 为 `none`、`bias`、`relu` 或 `relu_bias`（cuBLASLt 的 `DEFAULT`、`BIAS`、`RELU`、`RELU_BIAS`），`beta=1` 读取 `c [M, N]`（`beta = 1` 的 `beta * C`）；bf16 或 f16 操作数，结果为同一类型，`cublaslt.gemm` 的四种布局，两种姿态，各项选择加入，`sm_count_target` 和 `workspace_bytes`。它的算术是 `ρ(f(acc))`，`c` 在 bias 之前加上，这是库的顺序（X1）。`cublaslt.gemm` 和 `cublaslt.gemm_bias` 不变，字节和 digest 都与原来一致。带 bias 的 GEMM 不做 split-K 的规则推广到除 `none` 之外的每个 epilogue（3.1.1），用它自己的措辞表述。一个条目带有 `beta`、`contract`、`verification` 和可选的 `canary`；0.0.11 的读取器会拒绝它。Python 绑定、两个读取器、crate 的加载和启动、C ABI（一个 epilogue 和 `c` 的访问器，一个次版本）以及生成的 host 都支持它。

**按 pin 验证**（Q6，契约 (iii) 相对 (ii) 测量）。每个 pin 在生成时都会验证，在 `jit` 和 `export` 时都是如此：pin 在两组带种子的操作数上运行（X1 的操作数；bias 和 `c` 的前八列放特殊值；`a` 的若干行使每个 epilogue 得到 NaN、零和无穷的累加器，第二组中还有累加器溢出的行），并与 torch 以 f32 计算的公式逐位比较；公式作用于同一配置以 f32 的 `D` 重新编码后的累加器（`"f32-form"`；当 cuBLASLt 不接受这种编码时，用同一姿态的 f32 pin，即 `"plain-f32-pin"`，这里没有 pin 需要它），只舍入一次。CUDA 上的 torch 给出的位与 tilefish 自己对相同算子的下沉相同（一次 f32 加法；`torch.relu`，它逐位等于 `where(h <= 0, 0, h)`；以及转换到规范 NaN 的类型转换）。条目记录 `"measured-exact"`（没有元素不同），或者 `"binary-defined"` 以及不同元素的数量和它们以 ULP 计的最大距离。`contract="measured-exact"` 使构建拒绝后者。

X1 作为 GPU 测试（`python/tests/test_gpu_matmul.py`）：14 个形状（`M` 取 32、128、640、2048，乘以 `N x K` 取 1536 x 512、3072 x 1024、1024 x 3072，再加上 32 x 4096 x 4096、100 x 1000 x 1000），bf16 和 f16，两种姿态，七种情况（`bias`、`relu` 和 `relu_bias`，不带和带 `c`，以及带 `c` 的 `none`）：在一个 B300 MIG 切片（sm_103，30 个 SM）上，每个 cuBLASLt 版本 392 个 pin。

* **cuBLASLt 12.8.4：392 个中 392 个经测量精确**，包括特殊值和累加器（`a` 中使每个 epilogue 得到 NaN、零、无穷和溢出的行）。没有 pin 切分 `k`。
* **cuBLASLt 13.1.0：392 个中 372 个经测量精确。** 其余 20 个是算法 21 在 32 x 1024 x 3072 上的 pin，两种类型、两种姿态：
  * 带 `c` 的 `bias`：26% 的元素不同，最多差 15 176 ULP（bf16）和 2481（f16），与 X6 的发现一致；
  * `relu`、`relu_bias`、带 `c` 的 `relu` 和带 `c` 的 `relu_bias`：65 536 个元素中有 2048 到 2111 个不同，全部是激活前为 NaN 的元素：**这个算法的 ReLU 对 NaN 给出 `0`**，而其他每个 pin 都保留 NaN（`torch.relu`）。这是 X6 之外的发现：epilogue 的 NaN 规则也是 pin 的性质，只有特殊的累加器才能显示出来（对带有它们的情况，是 bias 或 `c` 中的 NaN 显示了它；对单独的 ReLU，是 `a` 的那些行）。
  * 带 `c` 的 `none` 在 `M = 32` 时切分 `k` 的 pin（3 和 4，f32 部分和）是经测量精确的，与 X6 的发现一致。

在另一个架构或版本上运行该测试，会在它自己的操作数组上重新检查每条记录并报告结果；在 sm_120 上，一个使用 `allow_f16_reduction=True` 的 f16 问题测试会验证 cuBLASLt 在那里给出的 split-K pin（它在等待一个 sm_120 门禁）。

**canary**（Q6：默认开启）。`canary=True`（默认）让 `export` 记录 `{"seed": 1, "out_digest": d}`：pin 在由 cuDNN canary 的 PTX 模块从该种子在设备上生成的操作数上运行（`a`、`b`、bias 和 `c` 分别作为 tag 0 到 3），`out` 和 workspace 清零，然后取 `out` 的 digest。Python replay 在进程和设备中首次使用该条目时运行它，crate 的加载在每个上下文中运行一次，两者都拒绝不同的 digest；在一个会运行它的 capture 内部的 Python replay 会被拒绝。

**plan**（`tilefish.fusion(epilogues="library")`，Q3：按 graph）。改写在形成区域之前在 IR 上进行。一个 16 位结果的 `cublaslt.gemm`，如果其 `y` 只被 `y.to(f32)` 读取，随后是 `+ c.to(f32)`（`[M, N]`）、`+ bias.to(f32)`（`[N]`）和 `tilefish.where(h <= 0.0, 0.0, h)`，这几项都可选但至少要有一项，并且每项只被下一项读取，然后 `.to()` 回 GEMM 的类型，且 `c` 和 bias 在 GEMM 之前生成，那么它就变成一次 `cublaslt.matmul` 调用（`<gemm>__<epilogue>[_c]`，GEMM 的选项，`contract="measured-exact"`，开启它的 canary）。后接 ReLU 的 `cublaslt.gemm_bias` 变成 `relu_bias`。这是相对 (ii) 测量的契约 (iii)（匹配到的值的 `R_acc`）：每个 pin 的验证就是对改写相对 `U(R_acc(P))` 的检查，而如 3.1.1 所述，这个参考用改写自己的 pin 的 f32 形式计算，因为公式就是区域在累加器上的算术，逐个算子对应；不是经测量精确的 pin 会使构建被拒绝（`tilefish.barrier(y)` 让 GEMM 保持不融合）。pin 是为 epilogue 选择的，而不是从它所替代的 GEMM 那里拿来的：在两个 pin 计算相同求和的地方，它的累加器就是 GEMM 的累加器（X1：16 个形状中的 14 个上是同一个 pin，在其他形状上字节相同；在 13.1.0 下，普通 GEMM 的 pin 在 `M = 32` 时切分 `k`，而 epilogue 的 pin 可能不切分），目前还没有构建检查这一点。在成立的地方，单独的 ReLU（作用于 `cublaslt.gemm`，或作用于带 bias 的 `cublaslt.gemm_bias`）写出 `U(P)` 的字节，因为 ReLU 与舍入可交换（X1）：这是在测试的形状上测得的，不是承诺。其他一切保持原样：`(y + bias) + c`（f32 中的另一次舍入）、f32 bias、被读取两次、被返回或被 barrier 的值、写成 `where(h > 0.0, h, 0.0)` 的 ReLU（它把 NaN 映射为 0，而 `torch.relu` 不会）。融合报告增加 `library_epilogues`：所做的改写，以及未改写内容的原因；在以 `"exact"` 规划的 graph 中，则是该模式会做什么。`mode="reference"` 解释执行改写后的程序（f32 累加器上的公式）。测量结果（上面的测试，`python/tests/test_gpu_matmul.py` 的两个 plan）：改写后的 plan 的 `jit`、`replay` 和 capture 与 pin 的累加器上的公式逐字节一致，并且除单独的 ReLU（0）之外，在 14 到 29% 的元素上与精确的 plan 不同。

**advisor。** 如果报告把某个 GEMM 步骤的消费者区域列为已匹配，该 GEMM 步骤就是阶段 `E2` 的阻碍项 `epilogue-library-cublaslt`："declare tilefish.fusion(epilogues='library')"，附带 epilogue 和契约，和每个候选一样按 advisor 估计或测量（`profile_plan`）的节省排序。未匹配的则保持为 `epilogue-library`（未来工作），并附带原因。

**速度**（sm_103，B300 MIG 切片，cuBLASLt 12.8.4；每个 plan 20 次 replay 的 CUDA graph 的中位数，微秒）：精确的 plan（已 pin 的 GEMM 和一个生成的区域）对照改写后的 plan（一个 `cublaslt.matmul`），X3 和 X4 的形状：

| 形状 `M x N x K` | bias | bias + ReLU | 残差 | 残差 + bias |
|---|---|---|---|---|
| 640 x 4096 x 1024, bf16 | 24.8 → 20.6 (-17%) | 24.8 → 22.2 (-10%) | 24.9 → 22.6 (-9%) | 26.8 → 24.7 (-8%) |
| 640 x 1024 x 4096, bf16 | 18.5 → 16.5 (-11%) | 18.5 → 16.6 (-11%) | 18.6 → 16.6 (-11%) | 18.6 → 18.5 (-0%) |
| 2048 x 3072 x 1024, bf16 | 54.5 → 41.1 (-25%) | 55.3 → 43.1 (-22%) | 61.8 → 46.5 (-25%) | 65.8 → 49.3 (-25%) |
| 128 x 3072 x 1024, bf16 | 8.3 → 6.3 (-24%) | 8.3 → 6.3 (-24%) | 8.3 → 6.3 (-24%) | 8.3 → 6.3 (-24%) |
| 8192 x 3072 x 768, f16 | 192.7 → 124.7 (-35%) | 192.5 → 135.7 (-30%) | 208.0 → 136.3 (-34%) | 215.9 → 147.0 (-32%) |

没有哪种情况更慢；640 x 1024 x 4096 上带 bias 的残差相加什么也没省下（它的 pin 是与普通 GEMM 不同的算法）。

**尚未解决。**

* GELU 和 `GELU_BIAS`（由二进制文件定义，带声明的容差），以及辅助输出（`RELU_AUX`、`GELU_AUX`）：尚未提供。
* `cublaslt.matmul` 的 `algos=`（对一个通过 cuBLASLt 融合的栈的移植会导入它的 pin）：目前在声明时被拒绝。
* 如果改写的 pin 在某个 bucket 上不是经测量精确的，构建会被拒绝：不做按 bucket 回退到路线 (a)（或回退到 `U(R_acc(P))`）。
* 改写的 pin 是否计算与 GEMM 的 pin 相同的求和，目前没有记录：构建可以在验证的操作数上比较两个累加器（或者在 cuBLASLt 接受带 epilogue 的 GEMM 配置时，用 GEMM 的配置 pin 改写），并说明 plan 何时保留 `U(P)` 的 GEMM。
* 原地写入的 `beta * C`（`D == C`，一个 `alias`）和 f32 结果；库调用的 `R_acc`（在 `"accumulator"` 下，cuBLASLt GEMM 保持原样）。自与 E1 合并以来，`tilefish.fused(epilogues=...)` 的作用域对改写生效（见“E1 与 E2 结合”）。
* sm_90（X7）。
* **sm_120，在 0.0.12 门禁中测量**（RTX 5090，170 个 SM；`tests/test_gpu_matmul.py` 和上面的矩阵）：cuBLASLt 12.8.4 和 13.1.0 都对除 `none` 之外的每个 epilogue 给出算法 21（对所问的六个问题，启发式返回的每个候选都是如此）。每个版本的 392 个 pin 中，有 140 个经测量精确（`bias`、带 `c` 的 `none`、带 `c` 的 `bias` 的一半）；每个 ReLU（224 个）都是由二进制文件定义的，其 ReLU 把 NaN 映射为 0（每个有限值都与公式一致），带 `c` 的 `bias` 中有 28 个把两者相加的结果与两种顺序都不符（最多差 15 298 ULP）。所以在 sm_120 上，`epilogues="library"` 会按其声明的要求，在导出时和首次 jit 运行时拒绝 ReLU 以及那些带 `c` 的 `bias` 问题：没有 plan 写出别的字节，门禁在那里的 plan 使用 bias。带选择加入的 f16 split-K pin 会像其他 pin 一样被采用和验证（全部是由二进制文件定义的）；没有选择加入时，没有 epilogue 切分 `k`。两条前进的路：写成 `tilefish.where(h > 0.0, h, 0.0)` 的 ReLU（NaN 变为 0）正是算法 21 的公式，一个匹配这种形式的匹配器可以把它验证为经测量精确；或者采用带声明容差的、由二进制文件定义的契约，就像 GELU 那样。

## 8. 问题（2026-11-16 已解决）

以下是评审时看到的问题，各附其推荐的默认值。所有问题都于 2026-11-16 按这些默认值决定，Q12 的顺序按评审所述（见上文“决定”）。

* **Q1. 三种契约。** 第 2.3 节定义的 (i) 精确、(ii) 累加器和 (iii) 库，是否应当是 tilefish 为融合进调用提供的契约？RFC 0001 的原则“融合本身绝不改变字节；改变字节的是声明的模式”是否延伸到它们？这意味着 `R_acc` 定义在 IR 上，所以融合失败时仍然运行 `U(R_acc(P))`。**推荐：是。**
* **Q2. 默认值。** **推荐：`epilogues="exact"` 默认开启。** 它不可能改变任何字节，并且每次导出都会验证它，与 RFC 0001 的 D1 和 D6 一样。`"accumulator"` 和 `"library"` 只在声明时生效，按 graph 或按 `tilefish.fused` 作用域。
* **Q3. 在哪里声明累加器模式。** 像 RFC 0001 的各个开关一样按 graph 和按作用域，还是也按调用（`gemm(x, w, epilogue="acc")`）？**推荐：只按 graph 和按作用域。** 调用处的开关会把数值语义放在两个地方。
* **Q4. 必须等于其所替代的库 pin 的生成 GEMM（i-d）。** X3 和 X3b 发现，在 sm_103 上只要 pin 不切分 `k`，它就相等。**推荐：是，作为 E3 中的显式选择加入**（`equal="cublaslt"`）。导出时在每个 bucket 上用带种子的随机操作数和特殊值操作数验证它。在验证失败的 bucket（split-K pin）上，保留库 GEMM，走路线 (a)。这使融合的 epilogue 相对现有栈精确。它是一个经测量的契约，不是经证明的契约，并在每次导出、每个库版本上重新验证。
* **Q5. 生成的 GEMM 何时足够好。** **推荐：当 X8 kernel 在移植模型所用形状的一个固定测试集上不超过已 pin 的 cuBLASLt GEMM 的 1.10 倍时，E3 按架构被接受。** advisor 的盈亏平衡点说明较慢的 GEMM 在哪里仍然划算。在此之前，E1 和 E2 先做，E3 仍是研究项。直接的 Triton kernel 是 1.22 到 1.44 倍。
* **Q6. 先做哪些 cuBLASLt epilogue。** **推荐：bias、ReLU 和 `beta * C`，相对 (ii) 经测量精确并按 pin 验证；然后把 GELU 作为由二进制文件定义的 epilogue**，需通过 `epilogues="library"` 开启，epilogue 条目默认开启 canary。GELU 的字节可能随 cuBLASLt 版本变化，就像任何 pin 的字节一样。在 epilogue 自己的算法无法编码 f32 形式的地方（GELU 算法），用普通 pin 的 f32 输出作为参考，并要求辅助输出等于 `ρ(acc + b)`（X1 发现总是如此）。
* **Q7. 库 epilogue 的 schema。** **推荐：一个新的 impl，`cublaslt.matmul`**，带有 `epilogue` 字段（更多取值）、`aux`、`beta` 和可选的 `canary`。`cublaslt.gemm` 和 `cublaslt.gemm_bias` 完全保持原样。
* **Q8. 名称。** **推荐：**
  * `tilefish.fusion(epilogues="exact" | "accumulator" | "library")`；
  * `tilefish.hook(name, value="acc" | "out", out=...)`，以及用于 prologue 的 `tilefish.hook(name, operand=...)`；
  * 带 `equal="cublaslt"` 的 `impl="tilefish.gemm"`；
  * 融合报告的 `contract` 取值 `exact`、`accumulator`、`measured-exact`、`binary-defined`。
* **Q9. 门控激活。** 每个 program 两个累加器（gate 的一个 tile 和 up 投影的一个 tile），还是把权重的行交错排列，让一个 tile 同时容纳两者？**推荐：两个累加器。** 不需要变换权重，每个累加器的字节都是普通 GEMM 的字节。
* **Q10. 比一个 tile 更长的行上的 norm。** **推荐：先做整行 tile**（`N <= BN`：逐 head 的 norm、较小的隐藏维度）。以声明的 tile 顺序跨 tile 拆分的形式放在 E4，在 RFC 0001 阶段 2 让有序求和变快之后。
* **Q11. cuDNN 融合图。** X5 发现，带 16 位中间值时，它们对 bias 和 ReLU 相对未融合组合是精确的，但只限于 sm_103 上的 cuDNN 9.13 和 NVRTC 13。tilefish 目前 pin 的是 9.10.2。**推荐：E6 放在 E2 和 X13 之后。** 它需要测量速度，需要对 sm_103 上的融合条目要求 cuDNN 9.13，还需要两个 host 中都有图构建器。
* **Q12. 工作顺序。** **推荐：E1（extern 上的 hook）和 E2（cuBLASLt epilogue）先做，并行进行。** E3（生成的 GEMM）在 X8 表明 Gluon GEMM 能弥合 X3 测得的差距之后再做。E4 和 E5 在它们所需的 RFC 0001 阶段之后做。advisor 的改动（第 5 节）随 E1 一起发布。

## 9. 已有工作

| 系统 | epilogue 融合 | 数值语义 |
|---|---|---|
| cuBLASLt | 按调用选择的固定 epilogue（bias、ReLU、GELU、辅助输出、`beta * C`、alpha 向量、FP8 scale 和 `amax`） | 库自己的；最后做一次转换 |
| CUTLASS | epilogue visitor tree：编译进 GEMM 的任意逐元素树以及行或列广播树 | 用户的 C++；累加器可见 |
| cuDNN 图 | 一个卷积或 matmul 加上逐点和归约节点，在运行时融合 | 图的计算类型，以及每个虚拟张量的数据类型 |
| XLA | GEMM 改写器把 bias、激活和 `beta * C` 折叠进 cuBLASLt epilogue；Triton GEMM 融合处理其余部分 | 相对未融合程序没有字节契约 |
| torch.compile / Inductor | 带融合 epilogue 和 prologue 的 Triton GEMM 模板，通过对照库的自动调优选择 | 没有字节契约；对照 eager 的容差测试 |
| tilefish（本 RFC） | 自己 kernel 中的 hook、在存在之处使用库 epilogue、prologue | 三种命名的契约：相对未融合组合精确、相对声明的累加器改写精确，或由库定义并被 pin 住 |

## 附录 A：一个完整的示例

一个 GEMM 及其 bias + GELU，为 `M` 取 `{128, 640}` 规划，`F = 4096`，`H = 1024`：

```python
@tilefish.library("up", impl="cublaslt.gemm", acc=f32)            # or impl="tilefish.gemm", or an extern with hooks
def up(a: Tensor[M, H, bf16], b: Tensor[F, H, bf16]) -> Tensor[M, F, bf16]: ...

def gelu_tanh(h):                                                   # a plain helper traces inline
    return 0.5 * h * (1.0 + tilefish.tanh(0.7978845608 * (h + 0.044715 * h * h * h)))

@tilefish.graph(fusion=tilefish.fusion(epilogues=MODE))
def mlp_up(x: Tensor[M, H, bf16], w: Tensor[F, H, bf16], bias: Tensor[F, bf16]):
    y = up(x, w)
    return gelu_tanh(y.to(f32) + bias.to(f32)).to(bf16)
```

| `MODE`，`up` 的实现 | 每个 bucket 的步骤 | 输出等于什么 |
|---|---|---|
| `"exact"`，`cublaslt.gemm` | `up`（库）以 bf16 写出 `y`；`fused_<hash>` 读取 `y` 和 `bias` 并写出输出 | `U(P)`：栈的字节 |
| `"exact"`，带 `out` hook 的 extern | 一个步骤：extern，区域在它的 hook 中，在其舍入之后 | 以 extern 作为 GEMM 的 `U(P)`，在构造上成立，并经过验证 |
| `"exact"`，带 `equal="cublaslt"` 的 `tilefish.gemm` | 在生成 GEMM 等于 pin 的每个 bucket 上为一个步骤；其他地方为路线 (a) | 以 cuBLASLt pin 作为 GEMM 的 `U(P)`：栈的字节，按 bucket 测量 |
| `"accumulator"`，带 `acc` hook 的 extern，或 `tilefish.gemm` | 一个步骤：kernel，区域在它的 hook 中，作用于累加器 | `U(R_acc(P))`：kernel 写出 f32，然后是区域；比 `U(P)` 少一次舍入 |
| `"accumulator"`，`cublaslt.gemm` | `up` 写出 f32；`fused_<hash>` 读取它（报告说明没有路线接收该区域） | `U(R_acc(P))`，比 `U(P)` 慢 |
| `"library"`，`cublaslt.gemm` | 一个步骤：带 `GELU_BIAS` 的 `cublaslt.matmul`，与其 canary 一起被 pin 住 | 由二进制文件定义：在与 `U(R_acc(P))` 的测量距离之内（X1、X2） |

库这一行需要 GELU 的 tanh 形式，因为 cuBLASLt 没有 erf GELU。改写究竟是否应当把手写的 tanh GELU 识别为 `GELU_BIAS`，属于 Q6 的一部分。该 epilogue 是由二进制文件定义的，所以这种识别是一种声明的等价，而不是精确的等价。

这里的 `R_acc(P)` 是把 `y.to(f32)` 替换为 `up` 的 f32 结果之后的程序：`gelu_tanh(up_f32(x, w) + bias.to(f32)).to(bf16)`。累加器情况下的融合报告条目会是：

```json
{"op": "up+fused_<hash>", "route": "hook", "value": "acc", "contract": "accumulator", "call": "up",
 "rounding_points_removed": [{"value": "y", "dtype": "bf16", "reason": "R_acc: widened by the epilogue"}],
 "nodes": [{"index": 1, "op": "cast"}, {"index": 2, "op": "cast"}, {"index": 3, "op": "add"}, "...", {"index": 12, "op": "cast"}],
 "verified": {"against": "U(R_acc(P))", "buckets": ["M=128", "M=640"], "equal": true}}
```
