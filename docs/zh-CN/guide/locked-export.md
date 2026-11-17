# 锁定导出：没有 GPU 的导出

[English](../../en/guide/locked-export.md) | **简体中文**

一次导出会从它运行的设备上得知一些事实，其余的一切都是源码、声明和工具链的函数。**lock（锁定表）** 在目标设备的一块 GPU 上把这些事实记录一次；之后 **锁定导出（locked export）** 在一台没有 GPU、没有 CUDA 驱动、也没有 torch 的机器上（例如 CI runner 或镜像构建）逐字节一致地写出同样的文件，某个文件不是 lock 中记录的那个时就失败。

```python
# on a GPU of the target device, into an empty directory; once per device, and again whenever an implementation changes
tilefish.runtime.export([encoder, decoder], "build/sm_103")
tilefish.runtime.write_lock("build/sm_103", "locks/sm_103.lock.json")          # commit the lock

# in CI and the image build, on a machine without a GPU
s = tilefish.runtime.export([encoder, decoder], "plans/sm_103", lock="locks/sm_103.lock.json")
s.digest   # the lock's digests.set: the pin the hosts load the plans with
```

`write_lock(directory, path)` 只读文件（manifest（清单）、各个 plan（计划）、融合报告以及每个文件的 sha256），它不做导出。`export(..., lock=path)` 接受与它所锁定的那次导出相同的算子和选项（`buckets=`、`reuse=`、`verify=`），架构取自 lock（给出 `arch=` 时，它必须是 lock 的架构）。plan set（plan 集）、单个 graph（计算图），以及单独的 kernel（核函数）和 extern（外部 kernel），都以同样的方式锁定。两者都会抛出 `tilefish.TilefishLockError`，它指出第一处差异以及该怎么做。

## 设备提供了什么

| 事实 | 在设备上由什么产生 | 导出把它写在哪里 | lock 把它保存在哪里 |
|---|---|---|---|
| 架构 | 设备 | 每个条目的 `arch`、每个 plan 的 `arch` | `device.arch` |
| SM（流式多处理器）数 | 设备 | 每个库条目的 `sm_count` 和 `declaration_digest` | `device.sm_count`（没有库条目时为 `null`） |
| 一个 cuBLASLt 算法（`cublaslt.gemm`、`gemm_bias`、`matmul`） | 启发式，或一张导入的表，由 `cublasLtMatmulAlgoCheck` 检查 | 条目的 `algo` | `library.<stem>.algo` |
| 一个 `cublaslt.matmul` pin（固定）的验证 | 在特殊操作数上运行这个 pin | 条目的 `contract` 和 `verification` | `library.<stem>.contract`、`.verification` |
| 一个 cuDNN pin（一个 engine 及其 knob，或一个 legacy 算法） | cuDNN 的启发式，或一张导入的表，并经过检查 | 条目的 `pin` | `library.<stem>.pin` |
| 一个 canary（哨兵值；`cublaslt.matmul`、`cudnn.conv_fwd`） | 在由种子生成的操作数上运行这个 pin | 条目的 `canary` | `library.<stem>.canary` |
| cuBLASLt 和 cuDNN 的版本 | 这两个库 | 每个条目的 `cublaslt_version`、`cudnn_version` | `toolchain.cublaslt`、`toolchain.cudnn` |
| 融合区域的验证 | 每个区域与其算子逐个运行的结果对比 | `<graph>.fusion.json`，每个区域的 `verified` | `fusion.<graph>` |

其余的都不需要设备。Triton 针对一个架构做提前编译（Triton 和 Gluon kernel、生成的 kernel 和融合区域），`ptxas` 汇编预构建的 PTX，预构建的 cubin 原样使用；plan（步骤、缓冲区、arena（内存池）、workspace（工作区）：声明的 `workspace_bytes` 和 Triton 报告的全局 scratch）、每个 bucket（分桶）的 warp（线程束）数（`by_bucket=`）、TF32 保护和 SASS 断言（`nvdisasm`，即 `tilefish[sass]` extra）都由声明和二进制文件得出。实测：`python/tests/locked_graphs.py` 的 plan set（一个生成的 kernel、一个 Triton extern、一个 Gluon extern、一个预构建的 PTX 和 cubin、一个已 pin 的 cuBLASLt GEMM（通用矩阵乘）、一个带 epilogue（尾部）的 `cublaslt.matmul`、一个 cuDNN 卷积、两个融合区域、一个全局 scratch 是 plan 的 workspace 的 kernel、每个 graph 两个 bucket、三个 canary），在一块 B300 MIG 切片（sm_103，30 个 SM，Python 3.11）上导出并锁定，然后在一台只有 CPU 的机器（Python 3.12，没有 torch，没有 CUDA 驱动）上用同样的 Triton 3.5.1 和 ptxas 12.9.86 按 lock 导出：14 个文件，0 处差异，构建缓存为空时用时 1.3 秒。

## lock 文件（`tilefish.lock.v1`）

规范化 JSON（键排序，缩进一个空格，末尾一个换行），所以一次什么都没改变的重新锁定（re-lock）不会改变一个字节，它的 diff 说明了改变了什么。每个字段都是必需的：

| 字段 | 内容 |
|---|---|
| `schema` | `"tilefish.lock.v1"` |
| `tilefish` | 写出这次导出的版本（manifest 和 plan 中记有它，所以锁定导出需要同一个版本） |
| `device` | `{"arch": "sm_103", "sm_count": 30}`：导出运行所在的设备（没有库条目记录 SM 数时，`sm_count` 为 `null`） |
| `toolchain` | 导出记录的版本，每一项只在有它的二进制文件或条目时出现：`triton` 和 `ptxas`（manifest 的 `toolchain` 文本），`cublaslt` 和 `cudnn`（整数） |
| `library` | 每个 stem：`impl`、`declaration_digest`（条目的：声明和 SM 数）、`workspace_bytes`，以及 pin：`algo`（cuBLASLt，128 个十六进制数字）或 `pin`（cuDNN）；`cublaslt.matmul` 还有 `contract` 和 `verification`；canary 是 `canary`（`{"seed", "out_digest"}` 或 `{"seed", "y_digest"}`） |
| `fusion` | 每个 graph 中导出验证过的区域：`[{"region": 区域在融合报告中的下标, "op": 它的 kernel, "verified": 报告记录的内容}]` |
| `kernels` | 每个 stem：`cubin`（文件）、`sha256` 和 `sha256_code` |
| `digests` | `manifest`（manifest 的 digest（摘要））、`plans`（每个 plan 的 digest，按 graph），以及 plan set 的 `set`：各 host（宿主）使用的 pin |
| `files` | 目录中每个文件的 sha256，按路径（不包括 `trace/`，因为它在所有 digest 之外） |

`write_lock` 会拒绝不是一个设备的一次导出的目录：两种架构、两套工具链或两个 SM 数的条目，导出不会写出的文件，目录中没有 plan 对应的条目或融合报告（由另一个 graph 导出到该目录时遗留下来），区域未经验证的融合报告（不在设备上做的导出会发出警告并记录 `skipped`；请在设备上导出，或者用 `verify=False`），以及 CuTe DSL kernel（见下文）。

## 锁定导出按顺序检查什么

1. lock 被严格读取（没有重复的键，每个字段都有类型）；`arch=` 是 lock 的架构；tilefish、Triton 和 ptxas 的版本是 lock 的版本（Triton 编译所用的 ptxas，即 `TRITON_PTXAS_PATH`，或者对只有预构建 PTX 的导出，`TILEFISH_PTXAS_PATH`）。
2. 每个库步骤取 lock 中它的 stem 的记录，其 `declaration_digest` 必须是该声明在一个 SM 数与 lock 相同的设备上的 digest（改变了的算子、选项或 bucket 会被拒绝）；声明了 `contract="measured-exact"` 的 `cublaslt.matmul` 会拒绝一个 `binary-defined` 的 pin，带 canary 的（默认如此）需要 lock 中的 canary，与导出一样；由它生成的条目必须通过 manifest 读取方的每一条规则（`docs/zh-CN/manifest-schema-3.md` 第 9 节）。**不询问任何启发式，不加载任何库，不在设备上运行任何东西，无论有没有设备。**
3. 导出会验证的每个融合区域（graph 的 `verify`，或 `verify=`）取 lock 中对它的记录。
4. 每个二进制文件构建出来后，在写出任何东西之前，先与 `kernels.<stem>` 对照（先比工具链，再比 `sha256`，并说明机器码是否不同）。
5. 文件按导出的方式写出，目录中的每个文件都与 `files` 对照：缺少的文件、字节不同的文件（先是 cubin，然后是 manifest，然后是其余的）以及 lock 中没有的文件（请导出到一个空目录）都会被拒绝。在这一步被拒绝时留下的文件用于与 GPU 上的导出做比较；不要发布它们。
6. 对该目录重新锁定，得到的就是这个 lock 本身：要求 host pin 的 digest、二进制文件的 `sha256_code`、各 pin 和各区域的记录都与文件所说的一致，所以手工编辑了 lock 而没有任何文件体现出来的情况也会被拒绝。

lock 中没有的 stem、区域或二进制文件会被拒绝，并提示 **在目标设备的 GPU 上重新锁定**：不会回退到启发式。

## 什么在哪里检查

| 检查 | 在 GPU 上导出 | 锁定导出（没有 GPU） | 在 GPU 上加载（Python、Rust、C++） |
|---|---|---|---|
| 每个二进制文件都是它的源码针对其架构的产物 | 构建 | 构建，并与 lock 的 `sha256` 对照 | 每个 cubin 与 manifest 的 `sha256` 和 `sha256_code` 对照；manifest 和 plan 与 pin 对照 |
| pin 就是声明所要求的 | 按声明的姿态（posture）询问启发式 | lock 中对这个声明的记录（`declaration_digest`） | Python 的 replay（回放）把条目与声明比较（导入的表的算法也比较）；Rust 和 C++ 依靠 plan 的 pin |
| 算法有效：`cublasLtMatmulAlgoCheck`、重新编码、归约方案及其 opt-in、没有 TF32 或模拟的 f32、workspace | 检查 | 不检查（没有库） | 每次加载都检查（Rust 和 C++：`Manifest::gemm`；Python：在准备 plan 时以及每次运行之前） |
| cuBLASLt 和 cuDNN 的版本、SM 数、架构 | 记录 | 取自 lock | 拒绝不同的值 |
| cuDNN pin 是确定性的、其数值 note 被允许、knob 在范围内、其 plan 能够 finalize；没有设置 `NVIDIA_TF32_OVERRIDE` | 检查 | 不检查 | 每次加载都检查 |
| canary | 计算 | 取自 lock | 首次加载时运行，digest 不同则拒绝 |
| `cublaslt.matmul` pin 与其公式对照（契约 (ii)） | 测量 | lock 中的记录 | 该记录（声明说是 `measured-exact` 而 pin 是 `binary-defined` 时 Python 拒绝）；canary 把字节与做测量的那次运行联系起来 |
| 融合区域与其算子逐个对照 | 测量 | lock 中的记录 | 不重复 |

所以导出所做的检查，在没有 GPU 时运行也没有一项被漏掉：加载所做的每项检查都和以前一样进行，而只有导出才测量的东西（启发式的选择、matmul pin 的验证和融合区域的验证）是在生成 lock 的那块 GPU 上测量的，测量的正是锁定导出被要求逐字节一致的那些二进制文件。lock 和 pin 一样是信任锚：一个手工编辑、pin 和哈希一起改过的 lock，会写出它所说的东西。请把 lock 的 diff 当作它所描述的产物的变更来审阅，并把它和代码提交在一起。

## 何时重新锁定

当实现改变时（一个 kernel、一个声明、一个选项、一个 bucket、导出的算子），当 tilefish、Triton 或 ptxas 升级时，当 cuBLASLt 或 cuDNN 升级时（一个 pin 属于一个库版本：host 会拒绝别的版本），以及每个新的目标设备。失败的锁定导出会说明哪里不同；请在那个设备的 GPU 上重新锁定，并审阅 diff。

## 限制

* **CuTe DSL kernel** 由 DSL 在 GPU 上编译，它们的启动（grid、cluster（线程块簇）、编译出的 host 函数的参数）也在 GPU 上捕获（`docs/zh-CN/cute.md`）：锁定导出拒绝它们，`write_lock` 也拒绝。含有 CuTe DSL kernel 的 plan set 请在 GPU 上导出。
* **两台机器上使用同一套工具链**：tilefish、Triton 和 ptxas 的版本会被检查；其他任何改变二进制文件的东西（一个 Triton 设置、`TILEFISH_DEBUG_PATHS`）都会改变它的 `sha256`，锁定导出会拒绝它。Python 版本不会影响二进制文件（上面的 3.11 和 3.12）。
* **每个设备一个 lock。** 架构会被检查；同一架构但 SM 数不同的两种设备（一块 MIG 切片、整块 GPU）在没有 GPU 时无法区分：用错 lock 的锁定导出会成功，而 host 在加载时会拒绝它的库条目（"pinned on a device with 30 SMs, but this device has 148"）。
* **每个目录一次导出**：lock 覆盖其目录中的每个文件，所以合并了多个架构的 manifest 的目录不能被锁定。
* lock 不是导入的表（`algos=`、`AlgoTable`、`CudnnTable`）：那些是声明的一部分，也是其 digest 的一部分，由它们生成的条目写着 `"algo_source": "imported"`，因此无法重现一次询问了启发式的导出。lock 记录导出 pin 了什么，无论它是怎么 pin 的，`cublaslt.matmul` 也包括在内（它不接受 `algos=`）。

门禁会在锁定的产物上运行各个 host：`python/tests/rust_gate.py prepare WORK --locked DIR --lock FILE` 从 `DIR`（一次锁定导出，在没有 GPU 的机器上生成，与 lock 对照，并在 lock 记录的 set digest 下回放）取 `locked_graphs.py` 的 plan set，而不是自己导出，于是 Python 的 replay、Rust host 和 C++ host 加载它、运行它的 canary 和用例，`compare` 再把它们的字节与 Python 的对照。
