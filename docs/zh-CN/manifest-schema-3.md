# Manifest schema 3

[English](../en/manifest-schema-3.md) | **简体中文**

本文是 manifest（清单）schema（模式）3 相对 schema 2 新增内容的规范。Python 包写入和读取 schema 3（`tilefish.runtime.export`、`tilefish.Manifest.load`），本仓库的 Rust crate `tilefish` 也是如此（已发布的 0.0.1 只读取 schema 2；第 12 节说明 crate 如何处理本文的每个部分）。这里的所有内容都由 `python/tests/` 和 `crates/tilefish/tests/` 中的测试检查（在最后一节列出），每个字段在描述处都写明了写入它的 Python 代码。

本文假定读者了解 schema 2：Triton kernel（核函数）条目的字段、`abi`、`dims`、`preconditions`、维度表达式语法、digest（摘要）的规范 JSON，以及 crate 已经做的启动检查。本文只规定新增或变更的内容；未提及的字段与 schema 2 相同。

目录：1 变更内容；2 顶层；3 digest（3.1 pin（固定）担保什么，可复现的导出）；4 kernel 条目（4.1 每个架构一个实现，4.2 warp（线程束）特化的 kernel，4.3 持久化 grid，4.4 每个 bucket（分桶）一个配置，4.5 cluster（线程块簇），4.6 全局 scratch，4.7 生成的 kernel 所用的 CUDA 数学函数，4.8 插桩构建的 probe（探针），4.9 CuTe DSL kernel）；5 ABI（带步长的视图、设备标量、固定参数、scratch 槽位、写在其输入之上的 `out`、TMA 描述符、空指针、64 位维度标量）；6 前置条件；7 维度表；8 预构建二进制文件；9 `library` 段（一个已 pin 的 cuBLASLt GEMM（通用矩阵乘），9.4 中为一个已 pin 的 cuDNN 卷积，9.5 中为一个带 cuBLASLt epilogue（尾部）的 GEMM）；10 严格读取，所有拒绝情形；11 迁移 schema 2 manifest；12 Rust host（宿主）做什么；13 测试位置。

## 1. 变更内容

| 方面 | Schema 2 | Schema 3 |
|---|---|---|
| 顶层 | `schema`, `tilefish`, `kernels` | 新增 `library`（一个列表，必需，可以为空） |
| digest | 标签 `tilefish.manifest-digest.v2\0`，覆盖 `{schema, kernels}` | 标签 `tilefish.manifest-digest.v3\0`，覆盖 `{schema, kernels, library}` |
| kernel 条目 | 仅 Triton kernel | 新增 `kind`（`"triton"` 或 `"binary"`）；`binary` 条目是一个预构建的 PTX 或 cubin |
| `abi` | `params`, `scratch` | 新增 `scratch_slots`（Triton 为 2，预构建二进制文件为 0） |
| ABI 指针 | 稠密、行主序、偏移 0 | 可以接受带步长的视图：新增 `strides` 和 `offset` |
| ABI 指针，形状 `[]` | 未使用 | 一个*设备标量*：设备内存中的一个元素 |
| ABI 标量 | `i32`，携带一个维度（`dim`）或一个元素个数（`count_of`） | 可以是 `i64`；可以携带一个步长（`stride_of`），属于一个视图，或由多个视图共享 |
| ABI 参数种类 | `pointer`, `scalar` | 新增 `fixed`：值写在 manifest 中的标量；以及 `given`（0.0.5，crate 0.0.3）：值由每次调用给出的固定参数 |
| 前置条件 | `pointer_alignment` 总是 16 字节 | 带步长的指针可以只承诺其元素的对齐；新增 `max_extent` |
| `dims` | 来自形状、标量和 grid 的名称 | 还包括带步长指针的步长维度和偏移维度 |
| library | 无 | 一个针对某一设备（其架构和 SM（流式多处理器）数量）已 pin 的 cuBLASLt GEMM：`library` 条目（没有文件） |
| library 条目（0.0.5） | — | 三个可选字段，仅在不是默认值时出现：`heuristic`、`algo_source`、`allow_bf16_reduction`（第 9.1 节） |
| library 条目（0.0.6，crate 0.0.4） | — | 在条目一直就有的字段（`dtypes`、`transa`、`transb`、`lda`、`ldb`、各操作数）中表示 f16 和 f32 操作数、f32 结果以及布局 `NN`、`TT` 和 `NT`；可选的 `allow_f16_reduction` 和 `allow_f32_reduction`（第 9.2 节）以及 `sm_count_target`（第 9.3 节） |
| kernel 条目（0.0.5，crate 0.0.3） | — | 一个可选的 `implementation`：对于一个按架构声明实现的算子，该二进制文件是哪一个实现（第 4.1 节） |
| ABI 参数种类（0.0.6，crate 0.0.4） | — | 新增 `tensordesc`：一个张量的 TMA 描述符（host 在每次启动时编码、按值传递的 `CUtensorMap`），它占用 `kernelParams` 的 `1 + 2 * rank` 个槽位，因此 `slot` 是一个参数的槽位开始的位置（第 5.5 节） |
| kernel 条目（0.0.7，crate 0.0.5） | — | `implementation` 可以是一个命名实现的名称，多个架构共享它，并且可以出现在 `binary` 条目上（预构建二进制文件作为一个实现）；这类算子同一个 stem 的各条目可以每个架构一个 ABI（第 4.1 节） |
| kernel 条目（0.0.6，crate 0.0.4） | — | 一个可选的 `total_num_warps`：warp 特化 kernel 一个块的 warp 数，其各分区在默认分区的 `num_warps` 之外增加 warp；它的 `block` 是 `32 * total_num_warps`（第 4.2 节） |
| library 条目（0.0.6，crate 0.0.4） | — | `impl: "cudnn.conv_fwd"`：一个已 pin 的 cuDNN 前向卷积，一个有自己字段的条目（问题、数学模式、legacy 或 backend API 的 pin 及其选取方式、cuDNN 版本、设备）（第 9.4 节） |
| kernel 条目（0.0.7，crate 0.0.5） | — | 生成的 kernel 上一个可选的 `cuda_math`：其算子的位与之一致的 CUDA 数学函数、测量这些函数所用的 CUDA 版本，以及它链接的 libdevice 的 sha256（第 4.7 节） |
| ABI 参数种类（0.0.7，crate 0.0.5） | — | 新增 `null`：每次启动都作为空指针传递的指针（Python 的 `Null[dtype]`），可用于 Triton kernel 或预构建二进制文件（第 5.6 节） |
| ABI 标量（0.0.7，crate 0.0.5） | — | 维度标量（有 `dim` 而没有 `stride_of` 的 `scalar`）可以是 `i64`（Python 的 `i64[n]`），而此前只有步长可以（第 5.7 节） |
| 前置条件（0.0.7，crate 0.0.5） | — | 不是设备标量的稠密指针可以只承诺其元素的对齐：这是下沉（lowering）生成的 kernel 针对不从 16 字节开始的视图的特化（第 6 节） |
| `grid`（0.0.7，crate 0.0.5） | 关于维度是仿射的（Python 写入的形式） | 可以将维度相乘（`(cdiv(C*T, 1024),)`）：语法一直有 `*`，两个读取器也一直接受它；host 计算 grid，从不求解 grid（第 7.1 节） |
| kernel 条目（0.0.7，crate 0.0.5） | — | 一个可选的 `persistent`：持久化 grid，在启动时把 tile 切分给设备（或其分区）的 SM 运行的 program（第 4.3 节） |
| kernel 条目（0.0.7，crate 0.0.5） | — | cluster（Triton kernel 的 `num_ctas`，预构建二进制文件声明的 cluster）的 `launch.cluster_dims` 和 `launch.num_ctas` 用 `cuLaunchKernelEx` 启动，`options.num_ctas` 表明这一点（第 4.5 节） |
| kernel 条目（0.0.7，crate 0.0.5） | — | `launch.global_scratch_size` 大于 0 时，启动时为每个 CTA（线程块）提供这么多字节的 scratch，`launch.global_scratch_align` 记录其对齐（第 4.6 节） |
| library 条目（0.0.12，crate 0.0.10） | — | `impl: "cublaslt.matmul"`：一个带 cuBLASLt epilogue 的 GEMM（`none`、`bias`、`relu`、`relu_bias`，以及用 `beta` 1 加上 `c`），其 pin 按 epilogue 的公式验证，并记录为 `measured-exact` 或 `binary-defined`（`contract`、`verification`），以及一个可选的 `canary`（第 9.5 节） |
| kernel 条目（0.0.11，crate 0.0.9） | — | 插桩构建（`export(..., instrument=True)`）条目上一个可选的 `instrument`：它的 probe，记录到 host 在 profile scratch 槽位中给出的缓冲区；含有这种条目的 manifest 是一个插桩导出，两个读取器都会拒绝它，除非被要求加载这种导出（第 4.8 节） |
| kernel 条目（0.0.13，crate 0.0.11） | — | `binary.source` 为 `"cute"` 的 `binary` 条目：一个 CuTe DSL kernel（`@tilefish.extern(cute=...)`），其工具链是该 DSL 的，其 `source_digest` 是其 PTX 的 sha256（种类 `tilefish.cute.v1`），其 grid 为整数；以及 ABI 种类 `tensormap`（DSL 编码的 TMA 描述符：一个模板，张量地址写入其前 8 个字节）和 `bytes`（常量字节），仅用于这种条目（第 4.9 节） |

没有任何内容被删除。一个由 Triton kernel 组成的 schema 2 manifest 通过三处添加成为 schema 3 manifest（第 11 节）。

## 2. 顶层

```json
{"schema": 3, "tilefish": "0.0.3", "kernels": [ ... ], "library": [ ... ]}
```

* `schema` 是 JSON 整数 `3`：不是 `3.0`，不是 `"3"`，不是 `true`。读取器拒绝任何其他值。对于 `schema` 是小于 3 的整数的文档，错误信息说明它是旧的 schema，需要重新导出；对于其他情况，错误信息说明它不是 schema 3 的 manifest。
* `kernels` 和 `library` 都是必需的 JSON 数组。`tilefish` 仅供参考，不计入 digest。
* 一个 `(stem, arch)` 对在 `kernels` **和** `library` 两者合起来的范围内唯一：重复的对被拒绝（Python：`Manifest.load`）。
* 文件按严格 JSON 读取，与 schema 2 相同（没有重复的键，没有 `NaN`/`Infinity`，没有无法放入 double 的数）。

## 3. digest

`manifest_digest(doc)`（`python/src/tilefish/_manifest.py`）是

```
sha256( b"tilefish.manifest-digest.v3\0" + canonical_json({"schema": 3, "kernels": K, "library": L}) )
```

* `K` 是按 `(stem, arch)` 排序的 `doc.kernels`，`L` 是按 `(stem, arch)` 排序的 `doc.library`；比较的是这两个字符串（按码位顺序，即 Python 对 `str` 的排序方式）。条目在文件中的顺序无关紧要。
* `canonical_json` 与 schema 2 的完全相同：对象键排序，分隔符为 `","` 和 `":"`，没有空白，非 ASCII 字符转义（`ensure_ascii`），没有 `NaN`/`Infinity`，浮点数采用 Python 最短往返的 `repr`（`2.0`、`1e-06`）。
* **每个新字段都计入 digest，没有例外，也没有特殊处理：** 条目整体计入 digest，因此 `kind`、`abi.scratch_slots`、`strides`、`offset`、`stride_of`、`alias`、`fixed` 参数及其 `value`、`binary`、`max_extent` 前置条件、`dims` 的新增内容、`library` 条目的每个字段（算法、cuBLASLt 版本、设备的 SM 数量、workspace（工作区））都在其中。顶层的 `tilefish` 是唯一在其之外的字段。
* 空的 schema 3 manifest `{"schema":3,"kernels":[],"library":[]}` 的 digest 是 `f66955caeb4e878dbe95483da2607d19ea4f734977e86282fff6f92f1a87fe6e`（一个测试向量；`test_manifest_schema3.py` 按上面的定义重新计算它）。空的 schema 2 manifest `{"schema":2,"kernels":[]}` 的 digest 是 `a3593eed6b463f7408711b92645197eda2e7bc5f2830b138b40e33ceedc1cfc8`。
* schema 2 文档（`{"schema":2,"kernels":[...]}`，标签 `v2`）的 digest 仍可由同一个函数计算：这正是已发布的 crate 0.0.1 为它读取的 manifest 计算的值。`schema` 为 3 以外任何值的文档，都以同样的 schema 2 形式计算 digest，`schema` 取其所写的值（读取器拒绝它，digest 函数则不拒绝：`crates/tilefish/tests/data/python_manifests.in` 中的跨语言测试向量包含这类文档，`export_rust_vectors.py` 逐字节一致地重现它们）。“是 3”指 Python 的 `schema == 3`：整数 `3` 和浮点数 `3.0` 是（`"schema": 3.0` 的文档以 schema 3 形式计算 digest，文本中为 `3.0`），`true` 和 `"3"` 不是。没有 `library` 列表的 schema 3 文档、不是列表的 `kernels` 或 `library`、不是对象的条目，都无法计算 digest（同一个文件 `python_manifests.in` 在 `MANIFEST_ERRORS` 中列出了它们；Rust 的 `manifest_digest` 对每一种都返回错误）。schema 3 的 digest 永远不会等于 schema 2 的 digest。

### 3.1 pin 担保什么，以及可复现的导出（tilefish 0.0.7）

一个 pin 是一个导出产物的 digest：即上面的各条目，并通过每个条目的 `sha256` 逐字节覆盖每个 cubin 文件（包括调试段）。这一点没有变，schema、digest 函数、两个读取器以及此前写入的每个 manifest 也都没有变。

变的是 cubin 里装了什么。Triton 把一个 kernel 的文件，以及它调用的每个 jit 函数的文件，记录在其 IR 的位置信息中，`ptxas` 把它们写进 cubin 的调试段（`.debug_line`、`.nv_debug_ptx_txt`、`.rela.debug_line` 及其 `.nv.merc.*` 副本）。到 0.0.6 为止，这些都是绝对路径：手写 kernel 所在的检出目录、Triton 自己的 `triton/language/standard.py` 所在的虚拟环境，以及下沉生成的 kernel 或生成的算子的源码所写入的随机目录 `/tmp/tilefish-gen-XXXXXXXX`。因此，同一组算子的两次导出有相同的 `sha256_code` 和不同的 `sha256`，digest 也不同，随目录或进程而变。现在，构建用一个不依赖 host 的路径来命名每个文件（`python/src/tilefish/_source.py`，`source_name`）：

* 文件以 `a/b/c.py`（或 `a/b/c/__init__.py`）结尾的模块 `a.b.c`：取这段结尾，即它相对于导入它的那个 `sys.path` 条目的路径；
* 生成的源码（下沉生成的 kernel、`tilefish.contiguous`、`im2col`、`im2col1d`、`col2im1d`）：`tilefish_generated/<name>.py`，即 `LoweredKernel.write(directory)` 写入并以 `tilefish_generated.<name>` 导入的路径；文件本身是 `<cache>/generated/<sha256 of the source>/tilefish_generated/<name>.py`（`$TILEFISH_CACHE_DIR`，否则为 `~/.cache/tilefish`），以原子方式写入，并且只在它保存的正是该源码时才使用；无法写入时，使用进程的一个私有临时目录；
* 任何其他文件（作为 `__main__` 运行的脚本）：它的基本文件名。

这些名称应用在一次编译的前端（`python/src/tilefish/_ir.py`，`host_independent_paths`），因此位置信息、PTX 和 cubin 都带着它们。`ptxas` 在一个空目录中运行（`ptxas_in_an_empty_directory`；预构建的 PTX 也是如此）：它会从工作目录查找每个名称，并把在那里找到的文件的修改时间和大小写进行号表；否则，从包含 `kernels/rms.py` 的目录运行的构建就会带上这些信息。行号、列号以及机器码都不变：这个版本的导出，与更早版本用相同源码和相同工具链构建的每个二进制文件有相同的 `sha256_code`，而 `sha256` 不同。同一组算子用相同工具链（Triton、`ptxas`、cuBLASLt、cuDNN）为同一设备做两次导出，得到的是相同的文件和相同的 digest，不论在哪个目录、哪个进程、哪台机器上：在一台机器上取得的 pin 可以检查另一台机器上的重新构建。考虑过但没有采用的方案：

* *让 digest 覆盖 `sha256_code` 而不是文件*：pin 将不再担保 host 加载的字节（调试段会在其外），每个读取器都需要第二种 digest 形式，旧的 pin 也会改变含义；
* *在计算哈希前规范化 cubin 内部的路径*：同上，另外两个读取器都要改写 ELF；
* *去掉行信息*：这会删除段并重新编号，因此每个 `sha256_code` 都会改变，profiler 也会丢失源码行。

**供 profiler 使用的绝对路径（可选关闭）。** 显示源码行的 profiler（Nsight Compute 的源码视图、`nvdisasm`）会在给定的源码目录中解析这些名称。在这样做不方便时，在构建的环境中设置 `TILEFISH_DEBUG_PATHS=absolute`，就会像以前一样保留 Triton 记录的绝对路径（检出目录、虚拟环境、生成源码的缓存目录）。它默认关闭，并且**破坏可复现性**：这样的导出在其他任何目录或其他任何机器上都有不同的 cubin 字节、`sha256` 和 digest（`sha256_code` 相同），因此从它取得的 pin 只能检查这一个导出本身。用它构建的二进制文件以单独的键缓存，从不提供给不使用它的构建；该变量的任何其他值都会被拒绝。

由 0.0.6 或更早版本导出的 manifest 仍然带有其构建目录的名称；它的加载、pin 和 replay（回放）都与以前完全一样。预构建二进制文件（第 8 节）按原样接受：带 `.file` 指令的 PTX 或在别处构建的 cubin 保留其原有的任何路径。

**锁定导出（locked export；tilefish 0.0.14）。** `tilefish.runtime.export(..., lock=path)` 在一台没有 GPU 的机器上逐字节一致地写出这个文档：由设备决定的字段（每个库条目的 `algo` 或 `pin`、`cublaslt_version` 或 `cudnn_version`、`sm_count`，`cublaslt.matmul` 的 `contract` 和 `verification`，每个 `canary`）取自 `tilefish.runtime.write_lock` 根据设备上的一次导出生成的 lock（锁定表）文件，并且每个文件都与 lock 记录的 sha256 对照（`docs/zh-CN/guide/locked-export.md`）。schema 中没有为它设的字段：读取方无法、也无需分辨是哪台机器写出了一个 manifest。

**manifest 旁边的 trace（追踪记录）（tilefish 0.0.9）。** 使用 `trace=True`（或 `TILEFISH_TRACE=1`）的导出还会在 manifest 所在目录中写入 `trace/`：每个二进制文件的每一层，供 `tilefish-explain` 使用（`docs/zh-CN/explain.md`）。manifest 中没有任何字段提到它，也没有 digest 覆盖它，因此不论有没有它，manifest、其 digest 和每个 pin 都相同；两个读取器都不查看它（读取器读取 `manifest.json` 及其提到的 cubin，不读该目录的其他内容）。它包含源码（kernel 的文件、生成的源码、Triton 的源码）：它不属于交付的内容，为 host 打包导出的工具会把它排除在外。

## 4. kernel 条目

`kernels` 条目有一个 `kind`，它是必需的，并且恰好是 `"triton"` 和 `"binary"` 之一。

**`kind: "triton"`** 是 schema 2 的条目加上 `kind` 和 `abi.scratch_slots`。它的字段是

`stem op kind arch ptx_target symbol cubin abi grid block shared num_warps num_stages launch constexprs buckets dims preconditions options
declaration_digest sha256 sha256_code toolchain`，以及可选的 `implementation`（第 4.1 节）、`total_num_warps`（第 4.2 节）、`persistent`（第 4.3 节）、`cuda_math`（第 4.7 节）、`instrument`（第 4.8 节）、`source_sha256`、`source_digest` + `source_digest_kind`（与 schema 2 相同：`source_digest_kind` 对 Triton IR 的 digest，即 `triton.jit` kernel 的 TTIR，为 `"tilefish.triton-ir.v1"`；对 Gluon kernel（`@tilefish.extern(gluon=...)`）为 `"tilefish.gluon-ir.v1"`：Gluon 前端构建的 TTGIR，包括布局和 `num_warps`，以它自己的标签计算哈希。Gluon kernel 由 Triton 编译，启动方式与其他 kernel 相同，因此它的条目是一个 `triton` 条目，其他方面没有任何不同；不重新计算 IR digest 的读取器，例如 Rust host，无需为它做任何事。Python 的 `replay` 用算子声明所指定的前端重新计算 digest，并拒绝种类属于另一个前端的条目）。`toolchain` 是 `{"triton": ..., "ptxas": ...}`。`options` 是传给 Triton 的编译选项：`num_warps` 和 `num_stages`（与条目的相等），以及对于声明中写明了它的 kernel，`enable_fp_fusion`（一个布尔值：tilefish 生成的 kernel 总是有它；手写 kernel 用 `@tilefish.extern(enable_fp_fusion=...)` 声明时也有它；没有这个键时适用 Triton 的默认值，即 `true`）。与每个字段一样，它计入 digest，replay 会将它与算子的编译规格（compile spec）比较。

**`kind: "binary"`**（第 8 节）没有 `num_warps`、`num_stages` 或 `options`（读取器拒绝含有其中之一的条目），取而代之的是 `binary`。它的字段是

`stem op kind arch ptx_target symbol cubin abi grid block shared launch constexprs buckets dims preconditions declaration_digest sha256 sha256_code
toolchain binary source_digest source_digest_kind`，以及可选的 `implementation`（第 4.1 节）和 `persistent`（第 4.3 节）

其中 `constexprs` 总是 `[]`，`source_digest_kind` 总是 `"tilefish.binary.v1"`，`source_digest` 总是存在，并且没有 `source_sha256`。

新增的验证规则（Python：`Manifest._read`）：

* `kind` 是这两个字符串之一；任何其他值，或者缺失，都被拒绝。
* 如果存在 `abi.scratch_slots`，它是一个整数（不是 bool），对 `triton` 等于 2，对 `binary` 等于 0。（Python 的读取器在该字段存在时检查它，而与算子编译规格的比较要求它恰好等于规格所写的值；不依赖规格读取的 host 必须要求它存在，并要求 `len(abi.scratch) == abi.scratch_slots`。）
* 对 `triton`，原有规则仍然有效：`shared` 为 0 但 cubin 中有 `.nv.shared.<symbol>` 段的条目被拒绝。**对 `binary` 不适用**：使用*静态*共享内存的手写 kernel 也有这个段，而声明（`shared`）说明的是它寻址多少*动态*共享内存；加载器的检查（静态 + 动态 ≤ 设备的 opt-in 最大值）起保护作用。
* **非零但过小的 `shared` 检测不出来，也无法从文件中检测出来。** Triton 给需要共享内存的 kernel 一个*动态*共享内存的 `extern` 数组，其大小由 Triton 决定（`metadata.shared`，来自它对 TTGIR 的分配），ptxas 从不知道这个大小：cubin 不记录它。在 sm_103 上用 Triton 3.5.1 和 ptxas 12.9 测量（`python/tests/test_gpu_shared_size.py`）：三个 `metadata.shared` 分别为 16 400、49 168 和 65 552 字节的 `tl.dot` kernel 有相同的 `.nv.shared.<symbol>` 段（1024 字节：sm_90 起每个块的系统保留），除参数记录外相同的 `.nv.info.<symbol>` 属性，以及驱动报告的相同静态共享内存（`CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`，0）。因此加载器能检查文件所写的内容（该段，以及 `shared ≠ 0`；静态 + `shared` ≤ opt-in 最大值），仅此而已；要再次得知该大小，就意味着在加载时再次编译 kernel 的 TTGIR（只有 Python 能做，并且对于大的集合需要数秒），而 `shared` 被编辑过的 manifest 也同样可能被替换了 cubin。为 `shared` 担保的，正是为 cubin 担保的东西：pin（manifest digest 覆盖 `shared`，而导出写入的是 Triton 的值）。没有 pin 时（`trust_directory=True`），二者都信任目录。当以后的 Triton 或 ptxas 记录该大小时，这个测试会失败，那时就可以在加载时检查了。
* `binary` 的 `toolchain` 是 `{}`（按原样使用的 cubin）或 ``{"ptxas": <the `--version` text>}``（经过汇编的 PTX）。

### 4.1 每个架构一个实现：`implementation`（tilefish 0.0.5，crate 0.0.3；名称、预构建二进制文件以及每个实现一个 ABI：tilefish 0.0.7，crate 0.0.5）

一个按架构声明实现的算子（`@tilefish.extern(by_arch={"sm_103": tilefish.impl(...), "sm_120": ...},
fallback=...)`：一个公开签名，每个架构一个 Triton 或 Gluon kernel，各有自己的编译选项、constexpr 值和 grid，或者一个预构建的 PTX 或 cubin），在每个架构上拥有该架构所用实现的二进制文件：为它声明的那个实现，否则是 `fallback`；两者都没有的架构会被导出拒绝（`jit` 和 `replay` 也会拒绝）。这样一个二进制文件的条目是一个 `triton` 或 `binary` 条目，多一个字段：

| 字段 | 类型与规则 |
|---|---|
| `implementation` | 可选字符串：`arch`（条目自己的架构：为它声明的实现）、`"fallback"`，或一个命名实现的**名称**：一个 ASCII 标识符（`[A-Za-z_][A-Za-z0-9_]*`），至多 64 个字符，不区分大小写时不是 `fallback`，也不区分大小写地不以 `sm_` 加一个数字开头（`tcgen05`、`ptx`、`fallback2`；不可以是 `Fallback`、`sm_120`、`SM_90a`）。只有一个实现的算子没有该字段（或为 `null`）。用于 `triton` 或 `binary` 条目。 |

* **命名实现。** `tilefish.impl(..., name="tcgen05")` 是映射发送给它的每个架构共用的一个实现（`by_arch={"sm_100":
  tc, "sm_103": tc}`，`fallback=` 可以是它，也可以另外命名）：它的键是它的名称，因此这些架构共享一个声明 digest 和一组编译规格，它们的条目写的是 `"implementation": "tcgen05"`。不是同一个声明的两个实现不得共用一个名称（以同一名称声明相同 kernel、选项、constexpr 值、grid 和 stub 的两个 `tilefish.impl` 对象是同一个实现）。没有名称时，一个实现的键是为它声明的架构，或 `"fallback"`，与以前相同：同一个实现对象挂在两个架构下就是两个实现，有两个键、两个 digest 和两组规格，这类声明的 digest 与 0.0.6 的相同。
* **预构建二进制文件。** `tilefish.impl(ptx="k.ptx", symbol=..., block=..., shared=...)` 和 `tilefish.impl(cubin={"sm_120": "k.cubin"}, symbol=...,
  block=...)` 都是实现（相对路径相对于声明该算子的文件，与 `@tilefish.extern(ptx=...)` 相同）。它们的条目是带 `implementation` 的 `binary` 条目（第 8 节）。cubin 实现对运行它的每个架构都有一个 cubin：声明会拒绝映射把它发送到一个它没有 cubin 的架构（以及其 `.target` 该架构无法构建的 PTX），构建会拒绝一个回退到它却没有 cubin 的架构。
* **各实现共享什么，不共享什么。** 调用者、plan（计划）和类型化的 host 启动器所看到的是共享的：算子的**公开签名**。每个实现接受算子的张量（它们的名称、顺序、`access`、`dtype`、声明的形状、是否接受视图（`Strided`）以及 `out` 可以写在哪个输入之上）、每次调用的固定参数（`given`，按名称和类型，按顺序）、这些形状所命名维度的约束，以及相同的 stem（二进制文件）和相同的 bucket（Python 拒绝不满足这一点的声明，`decorators._check_implementations`）。每个实现有**它自己的 ABI**，由它的条目携带：它可以带自己的 stub（`tilefish.impl(...,
  stub=f)`，kernel 的参数按顺序排列），因此一个张量在一个实现中可以是 `tensordesc`（TMA 描述符，第 5.5 节），在另一个实现中是 `pointer`，并有不同的 `scalar`、`fixed` 和 `constexprs`、不同的 scratch 槽位（`kind` 为 `triton` 或 `binary`）、`dims` 中不同的维度（视图的描述符带来它自己的步长维度，指针形式带来其步长标量所携带的维度）以及不同的 `preconditions`（描述符的地址 16 字节对齐，视图的指针可以只承诺其元素的对齐）。如何编译和启动也是它自己的：`grid`、`block`、`num_warps`、`num_stages`、`options`、`shared`、源码 digest，以及 `declaration_digest`。
* **身份。** 实现的键在它的声明中（`_canonical_declaration` 加入 `"implementation"`，对 kernel 和预构建二进制文件都是如此），因此也在它的 `declaration_digest` 和它的编译规格（`"implementation"`）中，条目也记录它。与每个字段一样，它在 manifest digest 中。一个实现的声明 digest 只覆盖该实现：manifest 是按架构的，因此另一个架构的另一个实现，或映射到某个命名实现的另一个架构，都不会改变这个条目；而这个架构的实现发生变化（它的 kernel、选项、constexpr 值、grid、stub、二进制文件，或映射把这个架构发送到另一个实现、另一个名称或 fallback）会使它过期。
* **构建。** 一个实现的编译规格知道，在生成或依据声明验证它时，声明在哪些架构上运行它；构建（`_build.check_implementation_arch`，对 kernel 和预构建二进制文件都是如此）拒绝其他任何架构：为 sm_103 声明的实现用在 sm_120 上，命名实现用在未映射到它的架构上，fallback 用在有自己实现的架构上。
* **replay。** Python 的 `replay` 像导出时一样选择设备架构的实现，并拒绝以下条目：其 `implementation` 不是该实现（`Manifest.built_kernel`："was built from the implementation 'tcgen05' ..., but the declaration now runs its implementation 'sm_103'
  on sm_103"）、其声明 digest 不是该实现的，或其 IR digest（对预构建二进制文件则是源码 digest）不是它现在的值；不论是否 pin。它用该实现的 ABI 启动条目。
* **plan** 用 stem 指定条目，stem 对每个实现都相同，并按名称、按它们共有的顺序绑定算子的张量，因此不论运行的是什么，一个 graph（计算图）的 plan 都是同一个文档：plan 的架构选择二进制文件（其 manifest 含有该架构的条目），host 遵循它所拥有条目的 ABI。Python 的规划器用每个实现绑定每个步骤，包括它们的前置条件，因此只有当每个实现都能运行时才会生成 plan（指针形式可以接受、但 TMA 描述符无法描述的视图，在每个架构上都被拒绝）。单次调用则按它所运行的实现检查：该实现的前置条件拒绝的张量，只在该架构上被拒绝。
* **读取器。** 两个读取器都拒绝以下 `implementation`：不是条目的 `arch`、不是 `"fallback"`，也不是上述形式的名称（另一个架构、另一个字符串、不是字符串的值）；哪些架构映射到某个名称由声明决定，读取器无从得知。Rust host 把它读入 `Entry::implementation`，不再需要它的其他信息：它像对任何算子一样，用条目的 ABI 启动其设备架构的条目。
* **旧的读取器。** 没有名称、且 `binary` 条目上没有 `implementation` 的 manifest 不变，包括 digest。tilefish 0.0.6 和 crate 0.0.4 拒绝是名称的 `implementation`，或出现在 `binary` 条目上的 `implementation`。

### 4.2 warp 特化的 kernel：`total_num_warps`（tilefish 0.0.6，crate 0.0.4）

当一个 Triton 或 Gluon kernel 的各分区在各自的 warp 上运行时，它就是 **warp 特化**的：Gluon 的 `gl.warp_specialize(default_args,
default_partition, worker_args, worker_partitions, worker_num_warps, worker_num_regs)`（例如在计算 warp 旁边放一个 1-warp 的 TMA 生产者和一个 1-warp 的 MMA 发射者），或 Triton 自己对 `tl.range(..., warp_specialize=True)` 循环做的 warp 特化（sm_100 及以上）。它的 TTGIR 有 `ttg.warp_specialize` 区域。

**`num_warps` 的含义。** 声明的 `num_warps`（`@tilefish.extern(..., num_warps=4)`，传给 Triton 的编译选项）是 kernel **默认分区**的 warp 数：即其 warp 特化区域之外的代码和其默认分区所运行的 warp，Gluon kernel 的布局就是为它们写的。对于不是 warp 特化的 kernel，这就是它全部的 warp。它不是一个提示。worker 分区增加它们自己的 warp，Triton 3.5.1 把 warp 特化之后的总数（`ttg.total-num-warps`，补齐到一个 warp group：一个 4-warp 的默认分区加两个 1-warp 的 worker 是 8，一个 8-warp 的默认分区加同样的 worker 是 12）报告为编译后 kernel 的 `metadata.num_warps`，即它自己的启动器用来组成一个块的 warp 数。

**条目。**

| 字段 | 类型与规则 |
|---|---|
| `num_warps` | 与任何 Triton 条目相同：编译选项，等于 `options.num_warps`，因此是默认分区的 warp 数 |
| `total_num_warps` | 可选 int，仅用于 `triton` 条目：编译后 kernel 一个块的 warp 数，即其所有分区的 warp。恰好在 kernel 是 warp 特化时存在，此时大于 `num_warps` 且至多为 32。其他任何 kernel 没有该字段（或为 `null`） |
| `block` | warp 特化 kernel 为 `[32 * total_num_warps, 1, 1]`，其他任何 Triton kernel 为 `[32 * num_warps, 1, 1]` |

**编译后的启动配置。** 一次启动除参数和 grid 之外所需的一切，都是条目记录的编译事实：
* `block`：`32 *` 编译后的 warp 数；
* `shared`：Triton 的 `metadata.shared`，即动态共享内存；
* `launch`：`cluster_dims`、`num_ctas`、scratch 大小、PDL 以及协作启动。

cluster（第 4.5 节）和全局 scratch（第 4.6 节）自 tilefish 0.0.7 和 crate 0.0.5 起可以启动；两个 host 都不设置的东西，在加载二进制文件时仍被拒绝：profile scratch、programmatic dependent launch 以及协作启动。每次启动都使用这些值，在 Python 和 Rust 中都是如此：单次启动、plan 的一个步骤，以及在 CUDA graph（CUDA 图）中 capture（捕获）的启动。

**构建**（`_build.build`）检查 Triton 做了什么：
* TTGIR 中没有 `ttg.warp_specialize` 区域的 kernel 必须恰好有声明的 `num_warps`。悄悄的改变会被拒绝："the compile
  option num_warps=4 was not applied (Triton used 8)"；
* warp 特化的 kernel 的 warp 数必须多于 `num_warps`，且至多为 32；
* 不论哪种情况，其 PTX 入口的 `.reqntid` 都必须是 `32 *` 元数据报告的 warp 数。Triton 为每个 kernel 都写入它，驱动会拒绝使用其他块大小的启动。

构建记录总数（`BuiltKernel.num_warps`、`BuiltKernel.warp_specialized`），构建缓存也记录，并在每次读取时对照选项和 PTX 检查它。

**身份。** 总数是标识二进制文件的那些东西（它的 IR、编译选项、架构和工具链）的函数，因此它不给编译规格或声明 digest 增加任何内容。编译规格的 `block` 仍为 `32 * num_warps`：规格在任何编译之前生成。条目记录总数，因此 manifest digest 像覆盖每个字段一样覆盖它：warp 数不同的二进制文件就是另一个 manifest。replay 像对任何 kernel 一样将条目与规格比较，用块大小 `32 * total_num_warps` 代替规格的块大小（`_build.expected_entry`），用该块大小启动，并拒绝 warp 数不等于规格的二进制文件，或对于 warp 特化的二进制文件，warp 数不多于规格的（`_build.check_launch_config`）。合并会拒绝替换 warp 数不同的条目（`total_num_warps` 是 `_conflicts` 比较的字段之一）。

**一个算子用于多个架构。** Triton 只在 sm_100 及以上对循环做 warp 特化，因此一个声明可能在一个架构上有 warp 特化的二进制文件，在另一个架构上有普通的二进制文件。`block` 和 `total_num_warps` 与 `shared` 一样是按条目的；Rust 代码生成器列出每个架构的块大小。

**读取器。** 两个读取器都拒绝：
* 不是整数、不大于 `num_warps`，或大于 32 的 `total_num_warps`；
* 不是 `[32 * total_num_warps, 1, 1]` 的 `block`；
* 出现在 `binary` 条目上的该字段。

JSON `null` 视同缺失。这一变更之前的读取器会拒绝带有该字段的条目：crate 的读取器拒绝未知字段，Python 0.0.5 的 replay 会发现块大小不等于规格的。因此不会有 host 用声明的块大小启动 warp 特化的 kernel。没有该字段的 manifest 不变，它们的 digest 也不变。

### 4.3 持久化 grid：`persistent`（tilefish 0.0.7，crate 0.0.5）

**持久化**（persistent）kernel 启动的 program 数大约等于设备能同时运行的数量，每个 program 以已启动 program 数为步长遍历工作项：`for t in range(tl.program_id(0), tiles, tl.num_programs(0))`。它最好的 grid 是 `min(tiles, programs per SM * SMs)`，任何维度表达式都无法表达，因为 SM 数量属于设备。用 `@tilefish.extern(..., grid=lambda M, N: (tilefish.cdiv(M, 128) *
tilefish.cdiv(N, 128),), persistent=tilefish.persistent(programs_per_sm=1))` 声明（Triton 或 Gluon kernel，或预构建二进制文件）时，它的 grid 是 **tile 的数量**，host 在加载 kernel 时切分它：

| 字段 | 类型与规则 |
|---|---|
| `persistent` | 可选对象，用于 `triton` 或 `binary` 条目：恰好包含 `programs_per_sm`，以及可选的 `sm_count`。其他所有 kernel 没有该字段（或为 `null`） |
| `persistent.programs_per_sm` | 1 到 32 的 int：一次启动在每个 SM 上至多运行的 program 数 |
| `persistent.sm_count` | 可选的 1 到 2^20 的 int（`null` 视同缺失）：一个**其字节依赖于运行多少个 program** 的 kernel 所声明的 SM 数量 |
| `grid` | 持久化条目的 grid：恰好一项，即 tile 数 |

**一次启动运行什么。** `programs = max(1, min(tiles, programs_per_sm * SMs // ctas))`，其中 `tiles` 是启动时 grid 的那一项，`ctas` 是一个 program 的 CTA 数（`launch.num_ctas`，第 4.5 节：一个 2-CTA cluster 的 program 占 2 个 SM），`SMs` 是 host **在加载二进制文件时**确定的多处理器数量：设备的（`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`；MIG 切片有自己的数量），或者当 host 运行在设备 SM 的一个分区上时，分区的（green context 对每次查询都报告设备的数量，因此由 host 来说明：Rust 中为 `Manifest::with_sm_partition(Some(n))`，Python 中为 `tilefish.runtime.sm_partition(n)`，一个包住加载和运行的上下文管理器）。一次启动沿 x 有 `programs` 个 program（乘以其 cluster，第 4.5 节）。块的 grid 像任何 grid 一样按设备的限制检查；一个二进制文件适用于每种设备形状，因此为一个 SM 数量（一个 MIG 切片）导出的 manifest 无需重新导出就能在另一个 SM 数量（整个设备）上运行。launch plan（启动计划）和 CUDA graph 保存的是它们被准备或 capture 时所在的设备和分区的 program 数。

**持久化 kernel 的字节。** SM 数量改变的是运行多少个 program，以及哪个 program 计算哪个 tile：从不改变二进制文件。因此：

* 没有 `sm_count`（默认）是 kernel 的承诺：**它的字节不依赖于 program 的数量**：每个 tile 由一个 program 从其输入计算出来，不论是哪个 program、有多少个 program（一个逐元素 tile、一个 GEMM tile、一行）。这是常见情况，也是声明所陈述的；tilefish 无法检查它（就像它无法检查 `MayAlias` 一样），gpu 测试要求它在 1、8、16 和 30 个 SM 上得到相同的字节。
* 字节**确实**依赖于它的 kernel（一个持久化归约，每个 program 先对自己的 tile 求和，再合并部分和：program 数量不同，相加的顺序就不同）要声明它适用的 SM 数量，`tilefish.persistent(sm_count=30)`。条目记录它，在数量不同的设备或分区上的 host 加载该 kernel 时会拒绝它（"declared for 30 SMs, but this device has 148"）：它在 30 个 SM 上的字节才是验证过的。在另一种设备形状上运行它是另一个声明（另一个值），因此是另一次导出，就像另一个 SM 数量的 cuBLASLt pin 一样（第 9 节）。

该字段像每个字段一样计入 digest，声明 digest 中也有它（仅在声明了它时）；早于它的读取器会拒绝带有它的条目（crate 的读取器拒绝未知字段；Python 0.0.6 的 replay 会发现条目与规格不符）。没有它的 manifest 不变。

### 4.4 每个 bucket 一个配置（tilefish 0.0.7）：`@tilefish.extern(..., by_bucket=...)`

一个带分桶维度的 extern（外部 kernel） 可以按 bucket 选择其编译期配置；以前这需要每个配置一个算子，并在 plan 时做选择：

```python
@tilefish.extern(triton=k.rows, num_warps=4, grid=lambda TB, BLOCK_T: (tilefish.cdiv(TB, BLOCK_T),),
                 by_bucket=lambda TB: tilefish.config(num_warps=2, BLOCK_T=16) if TB <= 256 else tilefish.config(num_warps=8, BLOCK_T=64))
def rows(x: Tensor[TB, 128, f32], y: Out[Tensor[TB, 128, f32]], n_rows: TB, BLOCK_T: const = 32): ...
```

`by_bucket` 是一个以签名中分桶维度（`Dim("TB", buckets=[...])`）的名称为参数的可调用对象，对它所命名维度的每种 bucket 组合调用一次，返回 `tilefish.config(num_warps=, num_stages=, num_ctas=, **consts)` 或 `None`；配置中省略的部分取声明的值。这些维度是算子的分桶维度（每个 bucket 一个二进制文件，就像依赖于 bucket 的 constexpr 会产生的那样：在 stem 中排在这类 constexpr 的 bucket 之后，按可调用对象参数的顺序）。每个二进制文件用其 bucket 的 warp、stage 和 CTA 以及其 consts 的值（tile 大小、任何 constexpr）编译，并以 grid 可调用对象在这些值下给出的 grid 启动。

**manifest 中没有任何新内容**：每个条目本来就有自己的 `num_warps`、`num_stages`、`options`、`constexprs`、`grid` 和 `block`，每个 bucket 的这些字段就是其配置的值。**身份**：声明 digest 覆盖每个 bucket 的完整配置（规范声明的 `by_bucket` 键按 bucket 组合列出 warp、stage、CTA、它设置的每个 const 的类型化值以及在这些值下的 grid；仅在给出 `by_bucket=` 时），因此它与以前一样是算子所有 bucket 共用的一个 digest，改变一个 bucket 的配置会使算子的每个二进制文件都过期；每个二进制文件的编译规格包含它的配置（编译选项、constexpr 值、grid），因此构建缓存的键和 manifest 条目都是按配置的。调用不能给出配置所设置的 const（它的值是 bucket 的）。预构建二进制文件不接受 `by_bucket=`（它的配置已经编译好了），使用 `by_arch=` 时每个实现各自说明（`tilefish.impl(...,
by_bucket=...)`）。

### 4.5 cluster：`launch.cluster_dims`、`launch.num_ctas`（tilefish 0.0.7，crate 0.0.5）

用 `num_ctas=2` 编译的 Triton 或 Gluon kernel（`@tilefish.extern(..., num_ctas=2)`，1、2、4 或 8；Triton 3.5.1 的编译选项，sm_90 及以上）把每个 program 运行在一个由 `num_ctas` 个 CTA 沿 x 组成的**cluster**上，这些 CTA 共享分布式共享内存和 TMA 多播；Triton 的元数据写有 `num_ctas` 和 `cluster_dims`（`(num_ctas, 1, 1)`）。预构建二进制文件声明它的 cluster，`@tilefish.extern(ptx=..., cluster=(2, 1,
1))`（至多 8 个块），用于为 cluster 编写的 kernel（`%cluster_ctarank`、`mapa`、`barrier.cluster`；`.explicitcluster` 使驱动拒绝没有 cluster 的启动）。

**条目。** `launch.cluster_dims` 和 `launch.num_ctas`（条目一直就有的字段）就是 cluster，`num_ctas` 是它们的乘积；每个 program 用多于一个 CTA 编译的 Triton 条目有 `options.num_ctas`（它的编译选项，恰好在它不为 1 时存在，并等于 `launch.num_ctas`）。预构建二进制文件的 `launch` 除了它的 cluster 之外都是零值。

**启动。** 条目的 grid 计的是 **program**（cluster）：一次启动是 `cuLaunchKernelEx`，块的 grid 为 `grid * cluster_dims`（逐轴；Triton 自己的启动器在 `num_ctas` 不为 1 时把 grid 乘以 `cluster_dims`），属性 `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` 为 `cluster_dims`，`CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE` = `SPREAD`（Triton 的启动器所要求的），每次启动都如此：单次启动、plan 的一个步骤，以及在 CUDA graph 中 capture 的启动（graph 的 kernel 节点带有该 cluster）。不带 cluster 启动的 `num_ctas=2` kernel 会计算出不同的字节（gpu 测试展示了这一点），这就是更早版本的读取器拒绝它的原因。两个 host 都在加载时拒绝不是 `num_ctas` 个 CTA、或大于可移植上限 8 的 cluster（没有 host 请求 CUDA 的不可移植 cluster 大小）。构建拒绝元数据中的 cluster 不等于声明的 `num_ctas` 的二进制文件。

### 4.6 全局 scratch：`launch.global_scratch_size`、`launch.global_scratch_align`（tilefish 0.0.7，crate 0.0.5）

当 kernel 需要在全局内存中有自己的内存时，Triton 3.5.1 给它一个**全局 scratch**：设备端 TMA 描述符（`tl.make_tensor_descriptor`）把它们的 `CUtensorMap` 写在那里，每个 128 字节。kernel 在其 ABI 的第一个 scratch 槽位（`global_scratch`，第 5 节）接收 scratch 指针，并从该指针起为每个 CTA 寻址 `global_scratch_size` 字节（`metadata.global_scratch_size`，按 `metadata.global_scratch_align` 对齐）：Triton 自己的启动器每次启动用 `triton.set_allocator` 安装的分配器分配 `global_scratch_size * gridX * gridY * gridZ * num_ctas` 字节，并传入地址。scratch 属于这次启动：其中的内容在启动之后都不保留，下一次启动可以覆盖它。

**声明**说明它，`@tilefish.extern(..., global_scratch=256)`（每个 CTA 的字节数，Triton 或 Gluon kernel），因为launch plan 在不构建任何东西的情况下确定其 arena（内存池）的大小（plan schema 1，第 6 节）；构建拒绝元数据给出另一个大小的 kernel（"the compiled kernel addresses
256 bytes of global scratch per CTA, but declare global_scratch=256"），以及对齐不能整除 256 字节的 kernel。它在声明 digest 中（仅在声明时）以及编译规格中（`global_scratch`）。

**条目。** `launch.global_scratch_size` 是每个 CTA 的字节数，`launch.global_scratch_align`（新增，一个 int，2 的幂）是其对齐；tilefish 为每个全局 scratch 都写入它，读取器拒绝没有它的全局 scratch：其他每个条目的 `launch` 仍是一直以来的六个字段（tilefish 0.0.6 中带全局 scratch 的条目没有对齐，会像以前一样被读取，并在加载其二进制文件时被拒绝）。预构建二进制文件没有全局 scratch（它只用它的 C 参数启动）。

**启动**给 kernel `global_scratch_size * (CTAs of its grid)` 字节，即该次启动时持久化 grid 的 program 数乘以 `num_ctas`，起始地址按 256 字节对齐（Triton 记录的每种对齐都能整除它），放在 `global_scratch` 槽位（profile scratch 槽位保持为空）。Python 的单次启动在启动的 stream 上分配它（torch 的缓存分配器，启动后释放，按 stream 排序；在 `torch.cuda.graph` capture 中从 graph 的内存池分配）；Rust host 从调用者获取它，作为**一次启动的最后一个张量**（一个可写的 `u8` 张量，至少 `Prepared::scratch_bytes` 字节，256 字节对齐，不与任何其他张量重叠：`Plan::tensor_names()` 以 `global_scratch` 结尾），这样启动中不分配任何内存，启动也可以被 capture；launch plan 把 plan 的 workspace 给每个带全局 scratch 的 kernel 步骤（plan schema 1，第 6 和第 7 节）。两个 host 都在加载时拒绝带全局 scratch 的 tilefish 0.0.6 manifest（它没有记录对齐：请重新导出）。profile scratch 仍被拒绝。

### 4.7 生成的 kernel 所用的 CUDA 数学函数：`cuda_math`（tilefish 0.0.7，crate 0.0.5）

tilefish 生成的 kernel 中，有些算子按照数值语义契约（contract），其位就是 CUDA 数学库某个函数的位：在 IEEE 数学下，`exp` 和 `silu` 建立在 `expf` 之上，`rsqrt` 是 `rsqrtf`，`erf` 是 `erff`（从 NVRTC 读取的内联 PTX）；在任一种数学模式下，`sin`、`cos` 和 `tanh` 是 libdevice 的 `sinf`、`cosf` 和 `tanhf`（`tilefish.lower` 的注释 10 以及 13 到 15）。这些函数并非在每个 CUDA 版本中都相同（CUDA 13.2 改变了 `erff`），而 libdevice 是 Triton 链接的一个文件（Triton 自带的，或 `TRITON_LIBDEVICE_PATH` 指定的）。这类二进制文件的条目会说明这一点：

| 字段 | 类型与规则 |
|---|---|
| `cuda_math` | 可选对象，仅用于有 `source_sha256` 的 `triton` 条目（一个生成的 kernel）。其他所有条目，以及不使用这些函数且不链接 libdevice 的生成的 kernel，没有该字段（或为 `null`） |
| `cuda_math.functions` | 二进制文件的算子的位与之一致的 CUDA 数学函数列表（`"erff"`、`"expf"`……：由小写字母、数字和 `_` 组成的名称），已排序，每个出现一次；对于只链接 libdevice 的 kernel 为空（`gelu` 的 `erf` 是 libdevice 自己的，不声称对应任何版本） |
| `cuda_math.cuda` | CUDA 版本列表（`"13.0"`：两个十进制数，没有前导零），按版本排序（`"12.9"` 在 `"12.10"` 之前），每个出现一次：在导出该二进制文件时，测得在这些版本上，每个列出的函数经该版本的 NVRTC 以 nvcc 的默认设置编译后，在每个 f32 上都与算子的位一致（`tilefish._cudamath.VERIFIED`）。`functions` 为空时它为空；当这些函数没有共同的版本时可以为空 |
| `cuda_math.libdevice` | 可选字符串，64 个小写十六进制数字：二进制文件链接的 libdevice 文件的 sha256，当 kernel 调用 libdevice 函数时存在 |

* **它是什么。** 它陈述二进制文件的位从何而来，而不是构建内容的一部分：构建从生成源码的代码中读取函数（`tilefish.lower` 为它们写入的辅助函数和调用；源码不为此改变，因此没有源码 digest 变化），从 tilefish 的表中读取版本，从 Triton 为编译选项链接的文件读取 libdevice。与每个字段一样，它在 manifest digest 中。它不与编译规格比较，合并也不比较它（不同的 libdevice 会产生不同的机器码，`sha256_code` 能反映出来）。
* **Python** 在每次运行时检查它：当 `functions` 不为空而 `torch.version.cuda`（torch 自身算子构建所用的版本）不在 `cuda` 中时，以及在 replay 中，当进程内的构建会链接与 `libdevice` 不同的 libdevice 时，它用 `TilefishCudaMathWarning` 发出警告（在 `warnings.simplefilter("error", TilefishCudaMathWarning)` 下是错误）。二进制文件仍然运行：警告说的是，在那里 torch 的算子以及建立在其上的参考模式可能与它的位不一致。参考模式也以同样方式根据 kernel 的 IR 发出警告。
* **Rust host** 把它读入 `Entry::cuda_math`（`CudaMath { functions, cuda, libdevice }`），自己不检查任何内容（它无法得知 host 其他 CUDA 代码的版本）；`Manifest::check_cuda_math("13.0")` 拒绝这样的 manifest：其中有条目的 `functions` 不为空而其 `cuda` 不包含该版本，并逐一列出。
* **读取器。** 两个读取器都拒绝：不是对象或含有其他字段的 `cuda_math`；缺少 `functions` 或 `cuda`，或者它们不是上述形式的列表、未排序或有重复；没有函数时 `cuda` 不为空；不是 sha256 的 `libdevice`；既没有函数也没有 libdevice；出现在 `binary` 条目或没有 `source_sha256` 的条目上的该字段。
* 测量和维护它的地方：`docs/zh-CN/numerics.md`（CUDA 或 torch 变化时要执行的步骤）、`python/tests/test_gpu_cuda_math.py`。

### 4.8 插桩构建的 probe：`instrument`（tilefish 0.0.11，crate 0.0.9）

插桩构建（`tilefish.runtime.export(..., instrument=True)` 或 `tilefish.instrument(capacity=..., ops=...)`，`docs/zh-CN/instrument.md`）把 kernel 的 `tilefish.probe` 调用，以及 `@begin`、`@calib` 和 `@end`，记录到 host 在 profile scratch 槽位（`abi.scratch` 的第二个槽位，生产启动传入空指针）中给出的 probe 缓冲区。它的条目会说明这一点：

```json
"instrument": {"format": "tilefish.probes.v1", "capacity": 256,
               "probes": [{"id": 3, "name": "load.wait", "sites": [{"loc": "kernels/ws.py:44", "role": "worker 0", "function": "kernels.ws._load"}]}]}
```

| 字段 | 类型与规则 |
|---|---|
| `instrument` | 可选对象，恰好包含这三个字段，仅用于 `triton` 条目。生产导出的所有条目都没有它 |
| `instrument.format` | `"tilefish.probes.v1"`：记录格式、缓冲区的布局以及构建添加的 probe（`docs/zh-CN/instrument.md`，第 3 节） |
| `instrument.capacity` | 1 到 65536 的整数：每个 warp 每次启动保留的记录数；一次启动的缓冲区为 `ctas * warps * (16 + 16 * capacity)` 字节，`warps` 是块的 warp 数，`ctas` 是其 CUDA grid |
| `instrument.probes` | kernel 的 probe 列表，每项为 `{"id", "name", "sites"}`：`id` 是 3 到 65535 的整数（0、1 和 2 是构建的 `@begin`、`@end`、`@calib`），`name` 是 probe 的名称（1 到 64 个字符，首字符为字母或 `_`，之后为字母、数字和 `_ . : / -`），每个 id 和每个名称各出现一次；`sites` 是 `{"loc", "role", "function"}` 的列表，均为字符串：写它的那一行（相对于模块）、它所运行的 warp 特化 kernel 的分区（`"default"`、`"worker 0"`……；kernel 没有分区时为 `""`），以及 jit 函数。没有 probe 的 kernel 为 `[]` |

* **它是什么。** 它是一个非生产二进制文件的条目：它的 `sha256`、`sha256_code`（以及对于有 probe 的 kernel，它的 `source_digest`：带 probe 的 IR）都是它自己的，该字段像每个字段一样在 manifest digest 中，因此插桩导出有不同的 digest，它的 plan 也是如此。声明 digest、ABI、grid 和启动属性都与生产条目相同：host 传入的内容中，除了 profile scratch 是缓冲区的地址（16 字节对齐）而不是空指针之外，没有任何变化。
* **插桩导出。** 含有带 `instrument` 条目的 manifest 是一个插桩导出：Python 的 `Manifest.load` 拒绝它，除非 `instrumented=True`；replay 拒绝它，除非它收集 probe（`probes=`）；Rust host 的 `Manifest::load` 和 `from_embedded` 拒绝它（`Manifest::load_instrumented` 接受它）。导出从不把两种变体写进同一个 manifest。早于该字段的读取器：crate 0.0.7 拒绝它（未知字段）；以空的 profile scratch 启动这种二进制文件时什么都不记录，写出生产的字节。
* **读取器。** 两者都拒绝：不是由这三个字段组成的对象的 `instrument`；其他 `format`；超出范围的 capacity（或布尔值）；不是由这三个字段组成的对象的 probe，超出范围或重复的 id，不合规或重复的名称；不是由三个字符串组成的对象的 site；出现在 `binary` 条目上的该字段。
* **Rust host** 把它读入 `Entry::instrument`（`Instrument { capacity, probes }`，每项为 `Probe { id, name, sites }`）；`Kernel::launch_probed` 在 profile scratch 槽位中传入一个 `ProbeBuffer`，并对生产二进制文件拒绝这样做（`docs/zh-CN/instrument.md`，第 4 节）。
* 测试位置：`python/tests/test_instrument.py`、`python/tests/test_gpu_instrument.py`、`crates/tilefish/tests/instrument.rs`、Rust 门禁 的插桩用例。

### 4.9 CuTe DSL kernel：`binary.source: "cute"`、种类 `tensormap` 和 `bytes`（tilefish 0.0.13，crate 0.0.11）

用 CUTLASS 的 CuTe DSL 编写的 kernel（`nvidia-cutlass-dsl`，`import cutlass.cute as cute`：一个启动 `@cute.kernel` 的 `@cute.jit` host 函数）用 `@tilefish.extern(cute=host, cute_args=..., cute_options=...)` 声明（`docs/zh-CN/cute.md`）。DSL 针对其参数的布局编译它，因此它的形状是静态的：每个 bucket 一个二进制文件。它的 cubin 是 DSL 自己的（NVVM 生成的 PTX 和 DSL 自带的 `ptxas` 生成的 cubin），启动则是编译后 host 函数的启动，在构建时被 capture（一个 CUDA graph：kernel 并不运行）：grid、block、cluster、动态共享内存以及每个 kernel 参数的字节。它的条目是第 8 节的 `binary` 条目，有以下不同：

| 字段 | 类型与规则 |
|---|---|
| `binary` | 恰好是 `{"source": "cute", "cutlass_dsl": <version>, "options": <string>}`：DSL 包的版本（`[0-9]+(\.[0-9]+){1,3}` 加上一个可选的 `[A-Za-z0-9.+_-]` 后缀）以及声明给出的 DSL 编译选项（`[A-Za-z0-9 =,._:/+-]` 中的字符，以单个空格分隔，可以为空） |
| `toolchain` | 恰好是 `{"cutlass_dsl", "nvvm", "ptxas"}`，均为字符串：DSL（`binary.cutlass_dsl`）、其 PTX 头部所写的 NVVM 版本、其 cubin 的 `.note.nv.tkinfo` 所写的 `ptxas` 版本 |
| `source_digest`, `source_digest_kind` | DSL 生成的 PTX 的 sha256，种类为 `"tilefish.cute.v1"`（必需） |
| `ptx_target` | PTX 的 `.target`，`sm_<arch>` 加上一个可选字母（`sm_103a`），属于条目的架构 |
| `grid` | 只有整数（没有维度名称：形状是静态的），即启动的 program 数：其块的 grid 逐轴除以其 cluster（第 4.5 节） |
| `block`, `shared` | 编译后 kernel 的块和动态共享内存 |
| `launch` | 除了编译后启动的 cluster 之外都是零值（至多 8 个 CTA，像每个 cluster 一样以 spread 调度策略启动） |
| `persistent`, `options`, `num_warps`, `num_stages`, `source_sha256` | 不存在 |
| `abi.params` | 只有种类 `pointer`（稠密：没有 `strides`、`offset` 或 `alias`）、`tensormap` 和 `bytes`，每个一个槽位；算子的每个张量恰好出现在一个 `pointer` 或 `tensormap` 中，按声明的顺序；`scratch` 为空，`scratch_slots` 为 0 |

**种类。**

* `tensormap`：`{"slot", "index", "name", "kind": "tensormap", "type": "tensormap", "ptx_type": "b8[128]", "size": 128, "dtype", "access", "shape",
  "template"}`。一个以 TMA 描述符传递的张量（它绑定维度，有前置条件，调用像传 `pointer` 一样传它），按值传递 128 字节，就像 DSL 的 host 代码为静态形状编码的那样：`template` 是 256 个小写十六进制数字，即这 128 字节，其中第 0 到 7 字节和第 64 到 127 字节为零。启动时传入模板，并在其第 0 到 7 字节写入张量地址（小端序 `u64`）（在 host 内存中 64 字节对齐，与 PTX 参数相同）；地址必须 16 字节对齐（其 `pointer_alignment`）并小于 2^57（DSL 保留 `address / 16` 的 53 位），否则启动被拒绝。它没有 `dim`、`count_of`、`stride_of`、`value`、`alias`、`strides`、`offset` 或 `tensor_map`。
* `bytes`：`{"slot", "index", "name", "kind": "bytes", "type": "bytes", "ptx_type": "b8[<size>]", "size": <1 to 32764>, "value"}`，`value` 是该参数的 `2 * size` 个小写十六进制数字。启动时传入这些字节；调用不为它提供任何东西。它没有其他字段。它的 `index` 就是它的 `slot`，它的 `name` 是 `param_<slot>`（它不是声明中的参数）。

**模板从哪里来。** DSL 在它自己的 host 代码中编码静态形状的 `CUtensorMap`，而不是用 `cuTensorMapEncodeTiled`：它写入八个 64 位字，第一个保存地址（`((address / 16) & (2^53 - 1)) << 4`），其余七个是形状、box、swizzle 和类型的常量，并让它们之后的 64 字节保持其栈上原有的内容（驱动忽略它们；host 传入零）。构建 capture 启动三次，每次每个张量都在不同的地址，并要求 kernel 读取的参数中的每个字节在三次中都相同，或者是某个张量的地址（一个对齐的 8 字节字，被整体读取，在每次 capture 中都等于该张量的地址）；kernel 读取的字节是其 PTX 在常量偏移处用 `ld.param` 加载的字节，或者是代码以其他方式取地址的参数（一个描述符）的全部字节；一个描述符是一个按 64 对齐、首字为地址的 128 字节参数，其第 64 到 127 字节除外。kernel 不读取的字节（结构体的填充，DSL 的 host 代码让它保持栈上原有的内容）在 `value` 和模板中为零。以下情况会被拒绝：在其他位置保存地址的参数（结构体内的指针）、会变化却不是地址的字节（host 函数自己的值：它分配的指针、动态形状）、出现在两个参数中或不出现在任何参数中的张量，以及顺序与声明不同的张量：否则 kernel 会接收到 host 无法从 manifest 重现的东西。

**身份。** 声明 digest 中，在预构建二进制文件那一段的位置上是 `{"kind": "cute", "host": "<module>:<qualname>", "source_sha256":
<the sha256 of the file of the host function>, "cutlass_dsl": <the DSL's version>, "options": <the options>}`（编译规格的 `binary` 段相同），因此不同的 DSL、不同的文件或不同的选项都是不同的声明。二进制文件由其 PTX 标识，就像 Triton kernel 由其 IR 标识一样：Python 的 replay 用进程中的 DSL（必须是声明的版本）编译声明，并将 PTX 的 sha256 与 `source_digest` 比较；不匹配会被拒绝，不论是否 pin；无法编译时，条目只有在 pin 的情况下才被接受。Rust host 和 C ABI 只比较 pin（manifest digest 一如既往地覆盖每个字段和每个文件）。编译规格的 `grid`（`"(1,)"`）和 `block`（`[1, 1, 1]`）是占位值：条目中的是编译后 kernel 的值，replay 将条目的张量（每个张量的 index、名称、dtype、access 和形状）与规格的比较。

**读取器。** 两个读取器都拒绝本节不允许的内容：形式不同的 `binary` 段、工具链、源码 digest、`ptx_target` 或 grid；`persistent` grid；在这种条目上，其他种类的参数、视图或 alias；在其他任何条目上的 `tensormap` 或 `bytes` 参数；长度不对、不是小写十六进制的模板或值，或者第 0 到 7 字节和第 64 到 127 字节不为零的模板；超出范围的 `bytes` 大小，或不是 `b8[<size>]` 的 `ptx_type`；属于其他种类的字段。早于本节的读取器（tilefish 0.0.12、crate 0.0.10 及更早版本）拒绝 source 为 `cute` 的条目（`binary must be {"source": "cubin"} or ...`）。没有这种条目的 manifest 及其 digest 不变。

## 5. ABI

`abi` 是 `{"params": [...], "scratch": [...], "scratch_slots": <int>}`。`params` 是 kernel 按顺序排列的启动参数，每个占一个 `kernelParams` 槽位（`slot` 是在 `params` 中的位置），但 `tensordesc` 除外，它占 `1 + 2 * rank` 个槽位（第 5.5 节）：`slot` 是一个参数的槽位开始的位置，计入它之前各参数的槽位。`scratch` 是 Triton kernel 在 `params` 之后启动时带的东西：两个 64 位指针（`global_scratch`、`profile_scratch`，`value` 为 0，位于最后一个参数的槽位之后），不变：空指针，但 `launch.global_scratch_size` 大于 0 的条目的 `global_scratch` 除外，它是启动给它的 scratch 的地址（第 4.6 节）。**预构建二进制文件有 `scratch: []` 和 `scratch_slots: 0`：它恰好用 `params` 启动，不多不少。**

每个 `params` 条目都有 `slot`、`index`（在声明中的位置）、`name`、`kind`、`type`、`ptx_type`、`size`。各种类及其各自的其他字段：

| `kind` | `type` | `ptx_type`, `size` | 其他字段 |
|---|---|---|---|
| `pointer` | `*bf16`、`*fp32`、`*i32`……（该 dtype 的 Triton 指针类型） | `u64`, 8 | `dtype`、`access`（`in`/`out`/`inout`）、`shape`；可选的 `strides` 和 `offset`（同时出现）；在 `out` 指针上，可选的 `alias`（第 5.4 节） |
| `tensordesc`（仅限 Triton 和 Gluon kernel） | `tensordesc` | `b8[128]`, 128（它的第一个槽位：`CUtensorMap`） | 指针的字段（`dtype`、`access`、`shape`，以及在 `in` 描述符上可选的 `strides` 和 `offset`），以及 `tensor_map`；没有 `alias`（第 5.5 节） |
| `scalar` | `i32` 或 `i64` | `u32`/4 或 `u64`/8 | `dim`（一个表达式）；可选地有 `count_of` 或 `stride_of` 之一 |
| `fixed`（仅限预构建二进制文件） | `i32`、`i64` 或 `f32` | `u32`/4, `u64`/8, `f32`/4 | `value` |
| `given`（仅限预构建二进制文件） | `i32` 或 `i64` | `u32`/4 或 `u64`/8 | 无：每次调用给出值（第 5.3 节） |
| `null` | `*bf16`、`*fp32`……（其 `dtype` 的指针类型） | `u64`, 8 | `dtype`：启动时传入 0（第 5.6 节） |
| `tensormap`（仅限 CuTe DSL kernel） | `tensormap` | `b8[128]`, 128 | 稠密指针的字段（`dtype`、`access`、`shape`）以及 `template`：启动时把张量地址写入其中（第 4.9 节） |
| `bytes`（仅限 CuTe DSL kernel） | `bytes` | `b8[<size>]`，1 到 32764 | `value`：启动时传入这些字节（第 4.9 节） |

一次启动这样传参：`pointer` 传其 64 位地址（见 5.1），`null` 传 64 位零，`scalar` 传其 `type` 的有符号整数，`fixed` 传其 `type` 的 `value`（`ctypes.c_int32`/`c_int64`/`c_float`：`f32` 值是该 JSON 数舍入到最近的 `f32`）。它是 `cuLaunchKernel`，grid 为条目的 `grid` 求值的结果，block 为条目的 `block`（`32 *` 编译后 kernel 的 warp 数，对 warp 特化的 kernel 是其所有分区的：第 4.2 节），动态共享内存为其 `shared` 字节。

### 5.1 带步长的视图：指针上的 `strides` 和 `offset`

有 `strides` 和 `offset` 的指针接受张量的一个视图：任意元素步长和任意起始偏移。（没有它们的指针与以前一样，是偏移为 0 的稠密行主序张量。）这两个字段要么同时出现，要么都不出现。

* `strides` 是一个列表，`shape` 的每个轴恰好对应一项（至少一项）。每一项是**一个整数**（一个*固定*步长）或**一个字符串**（一个*步长维度*的名称）。
  * 固定步长是 `1`，只会出现在**最后一个**轴上（kernel 沿该轴连续加载）。读取器拒绝任何其他整数，以及出现在任何其他轴上的固定步长。
  * 字符串项是恰好一个 `scalar` 参数的 `dim`，该标量的 `stride_of` 指向这个指针和这个轴（是对象 `{"arg": <this pointer's name>, "axis": <this axis>}`，或在其列表中包含它）；该标量携带这个步长。Python 双向检查（`_specvalid._check_strides`）：`stride_of` 的每个对象都指向一个带步长的指针及其一个轴，该轴的项就是该标量的 `dim` 字符串，每个轴一次；每个字符串项都有它的标量。
* `offset` 是一个字符串：这个指针的**偏移维度**的名称。（它是 `"<pointer name>_offset"`。）它不由标量携带：传给 kernel 的指针已经是 `base + offset * itemsize`（视图第一个元素的地址）。
* 标量上的 `stride_of` 是对象 `{"arg": <identifier>, "axis": <int ≥ 0>}`，或者是**由两个或更多这样的对象组成的列表**，其 `(arg, axis)` 对各不相同；一个标量有 `stride_of` **或** `count_of`，从不同时拥有两者。

**多个视图共享的步长**（Python 中为 `StrideOf(("q", "k", "v"), 0)`，按声明的顺序写成列表 `[{"arg": "q", "axis": 0}, {"arg": "k", "axis": 0}, {"arg": "v", "axis": 0}]`）是多个带步长指针的某个轴共用的一个标量：一个 kernel 以一对步长读取一个缓冲区的多个视图，例如一个融合的 `[B, 3 * D]` 张量在偏移 0、`D` 和 `2 * D` 处的 q、k、v 列块。这些指针中的每一个在该轴的 `strides` 项都是该标量的 `dim`（因此该维度在 `dims` 中只命名一次，带着它的约束），并各有自己的 `offset` 和自己的地址。该标量只传一次。一个视图的步长仍是对象，与以往一样，因此没有共享步长的 manifest 及其 digest 不变；早于列表形式的读取器（已发布的 crate 0.0.2、tilefish 0.0.4）拒绝含有它的 manifest。

**host 为启动做什么。** host 得到一个视图：它的指针（第一个元素的地址）、形状、元素步长（以及，如果它知道的话，从分配起点算起的元素偏移）。

1. 与 schema 2 一样，从 `shape` 绑定形状维度。
2. 对视图的每个轴 `a`，令 `stride[a]` 为其元素步长，并**规范化**：对大小为 1 的轴，从最后一个轴到第一个轴依次替换其步长，最后一个轴替换为 `1`，其他轴替换为 `size[a+1] * stride[a+1]`（kernel 从不读取大小为 1 的轴的步长，因此 `unsqueeze` 或 `expand` 不会传入一个没有 kernel 读取、又可能放不进 32 位的步长）。其他步长按原样使用。步长必须 ≥ 0（允许步长为 0，如扩展视图）；负步长被拒绝。
3. 对 `strides` 项为整数 `1` 的每个轴：视图沿该轴的规范步长必须为 1，否则被拒绝（种类 `stride`）。
4. 对项为名称 `x_s0` 的每个轴：把维度 `x_s0` 绑定到规范步长（它像形状维度一样是一个待求解的未知量：它出现在 `dims` 中，`dim: "x_s0"` 的 `scalar` 携带它的值）。如果存在约束 `dims["x_s0"].divisible_by`，则检查它。多个视图共享的名称（共享步长）是一个未知量：ABI 中这些指针的第一个绑定它，其他每一个沿其轴都必须有这个规范步长，否则在启动任何东西之前，调用就被拒绝（种类 `stride`，指出该指针、它具有的步长以及要求的步长）。
5. 有 `stride_of` 的 `scalar` 以该绑定的步长传入，作为其 `type` 的有符号整数。**它必须能放入该类型：** `i32` 要求 `0 ≤ v ≤ 2^31 − 1`，`i64` 要求 `0 ≤ v ≤ 2^63 − 1`；否则被拒绝。（当算子声明 `index_bits=64` 或其 `StrideOf` 写明 `dtype=i64` 时，步长的 `type` 为 `i64`；维度或元素个数为 `i32`，除非声明写的是 `i64[n]`，第 5.7 节。）
6. 传入的指针是视图的地址；在它上面检查 `pointer_alignment` 前置条件（第 6 节）。
7. 偏移维度不从步长标量绑定；知道元素偏移的 host *可以*用它检查 `dims[offset].divisible_by`，但要求的是第 6 步的地址检查（对于 16 字节对齐的 base，二者是同一个检查）。

`max_extent`（第 6 节）限定视图的范围。对于别名判断，带步长的张量占据 `[address, address + extent * itemsize)`，`extent` 的定义见该节；被写入的张量（`out`、`inout`）不得与任何其他参数重叠。（这个版本中只有 `in` 指针可以带步长：Python 前端拒绝 `inout` 或 `out` 带步长指针的声明，两个读取器都拒绝 `access` 不是 `in` 的指针上的 `strides`。）

### 5.2 设备标量：形状为 `[]` 的指针

`shape` 为空列表的指针是一个**设备标量**：设备内存中其 `dtype` 的一个元素（Python 类型 `DevScalar[i32]`）。它的 `access` 是 `in`。host 传入恰好一个元素的区域（rank 0 或形状 `[1]`），并检查其 dtype 和 `pointer_alignment`（16 字节）；它没有 `static_size`、`max_elements` 或 `max_extent` 前置条件，也没有 `strides`/`offset`。kernel 用 `tl.load(ptr)` 读取它。在其他方面它是普通指针；它之所以作为一个单独的情况存在，只是为了让 host 能在启动前把值写进每次调用的 arena。

### 5.3 `fixed` 参数

`fixed` 参数是值写在 manifest 中的标量：预构建二进制文件的常量参数（一个 `float alpha`、一个必须为零的 `long`、一个写成显式参数的 Triton scratch 槽位）。它有 `value`，没有 `dim`。对 `i32`/`i64`，`value` 是一个 JSON 整数（不是 bool，在该类型的范围内）；对 `f32`，是一个**带小数部分或指数**的 JSON 数（`2.0`）；非有限的 `f32` 是带标签的对象 `{"float": "inf" | "-inf" | "nan"}`（与 schema 2 中 constexpr 值的写法相同）。算子的调用者可以重述这个值，但不能改变它。

**每次调用给出的固定参数**（`"kind": "given"`）是预构建二进制文件中值不在 manifest 中的标量：每次调用给出它（Python：`h_out: const[i32]` 或 `const[i64]`，**没有默认值**，`k(x, h_out=5)`；调用的几何信息，例如输出大小或偏移，调用中没有张量能确定它）。它的 `type` 是 `i32` 或 `i64`（`u32`/4 或 `u64`/8），没有 `value`、`dim` 或任何其他字段，并且只有 `binary` 条目才有它。**一个条目适用于每个值**：启动时在该参数的槽位传入调用给出的值，作为其类型的有符号整数（任意符号；放不进的值被拒绝；没有为其中某个参数给值的启动，或为条目中没有的名称给值的启动，也被拒绝：Python `tilefish.run(op, x, h_out=5)`，Rust `Kernel::launch_with(stream, tensors, &[("h_out", 5)])`）；launch plan 的步骤记录其调用的值（`"given"`，`docs/zh-CN/plan-schema-1.md` 第 7 节）。浮点数仍是 `fixed` 参数（每个声明固定）。声明 digest 中该参数记为 given（`"given": true`，没有默认值），因此带默认值的声明是另一个算子。早于这一种类的读取器（tilefish 0.0.4、已发布的 crate 0.0.2）拒绝含有它的 manifest。

### 5.4 可以写在输入之上的 `out` 指针：`alias`

指针上的 `"alias": "<name>"` 表示，kernel 可以在这个指针和指针 `<name>` 是**同一个张量**的情况下启动：一个有两个指针的原地 kernel，例如 `k(x, r, y, r_out)`，它在写入 `r_out` 的某个元素之前先读取 `r` 的同一个元素（Python：`r_out: Out[Tensor[R, 64, f32], MayAlias("r")]`，只有 `@tilefish.extern` 能声明它：这是手写 kernel 的承诺）。两个读取器都检查的规则：

* `alias` 是一个字符串，位于 `access` 为 `out` 的 `pointer` 上（`scalar`、`fixed` 参数和任何其他指针都没有它）；
* 它指向同一个 ABI 中 `access` 为 `in` 的一个 `pointer`，该指针没有 `strides`（一个完整的张量），`shape` 不为空（不是设备标量），`dtype` 相同，声明的 `shape` 也相同（同一个列表，逐项相同）；
* 一个给定的输入至多被一个指针指向。

**host 做什么。** 启动的无别名规则（第 5.1 节：被写入的张量不得与任何其他参数重叠）有一个例外：`out` 指针和它指向的 `in` 指针可以是同一个张量，即**同一个地址、同一个形状和相同的规范步长**（Python：`_launch.same_tensor`；Rust：`plan.rs`）。二者的任何其他重叠（错位的视图、不同的形状）都以种类 `alias` 被拒绝，二者之一与第三个参数的任何重叠也是如此。没有 `alias` 的指针与以往相同，因此没有它的 manifest 及其 digest 不变；早于它的读取器（已发布的 crate 0.0.2、tilefish 0.0.4）拒绝含有它的 manifest（未知字段）。算子的声明 digest 中有它（参数上的 `"alias"`），仅在存在时。

### 5.5 TMA 描述符：`tensordesc`（tilefish 0.0.6，crate 0.0.4）

用 Triton 或 Gluon 编写的 Hopper 或 Blackwell kernel，当一个参数是 *host 张量描述符*时，用 Tensor Memory Accelerator 搬运 tile：host 用 `cuTensorMapEncodeTiled` 编码、**按值**传递的一个 `CUtensorMap`（在 PTX 中为 `.param .align 64 .b8 name[128]`，一个 `__grid_constant__`）。Python 用 `x: TensorDesc(Tensor[T, C, f16], block=(128, 64))` 声明它（`Out[TensorDesc(...)]`、`InOut[TensorDesc(...)]`；对 kernel 读取的视图用 `TensorDesc(Strided[Tensor[...]], ...)`），调用传入的是**张量**：它的描述符由 host 制作。

**槽位。** Triton 3.5.1 把一个描述符参数（其启动器 `backends/nvidia/driver.py` 中的 `make_tensordesc_arg`）按以下顺序展开为：

1. `CUtensorMap` 的 128 字节，按值；
2. 张量的全局形状，每个轴一个 `i32`，最外层在前（`u32`，每个 4 字节）；
3. 张量的元素步长，每个轴一个 `i64`，最外层在前（`u64`，每个 8 字节）；

因此 rank 为 `r` 的张量的 `tensordesc` 占 `kernelParams` 的 **`1 + 2r` 个槽位**（加载的函数报告 `128`，然后 `r` 次 `4`，然后 `r` 次 `8`，两个 host 都检查这一点），全部来自那一个参数条目，其 `slot` 是第一个槽位。形状和步长是启动时那个张量的（第 5.1 节第 2 步的规范步长）。下一个参数的 `slot` 紧跟其最后一个槽位，scratch 槽位也是如此。

**条目。** `{"slot", "index", "name", "kind": "tensordesc", "type": "tensordesc", "ptx_type": "b8[128]", "size": 128, "dtype", "access", "shape",
["strides", "offset"], "tensor_map"}`：指针的字段（第 5 节），其 dtype 为 `bf16 f16 f32 i32 i64 u8 e4m3 e5m2` 之一，形状有 1 到 5 个轴，没有 `alias`（它自己没有，也没有别的指针以 alias 指向它）；在视图的描述符上（仅限 `in`），`strides` 的最后一项是固定的 `1`，其他每一项是一个步长维度的名称（`x_s0`；从视图绑定，在描述符自己的步长槽位中传入，因此没有 `stride_of` 标量携带它，`stride_of` 也不得指向描述符），以及 `offset` 维度（`divisible_by` 16 字节中的元素数）。`tensor_map` 是 `cuTensorMapEncodeTiled` 中不属于张量的每个参数，恰好是以下字段：

| 字段 | 类型与规则 |
|---|---|
| `data_type` | dtype 对应的 `CUtensorMapDataType`，用其枚举值的名称表示：`CU_TENSOR_MAP_DATA_TYPE_UINT8`（`u8`、`e4m3`、`e5m2`）、`_INT32`、`_INT64`、`_FLOAT16`、`_FLOAT32`、`_BFLOAT16` |
| `block` | 描述符的块（kernel 一次 `load` 或 `store` 搬运的 tile，在其 Triton 类型 `tensordesc<fp16[128, 64]>` 中），最外层在前：`rank` 个 1 到 65536 的 2 的幂 |
| `box` | map 的 box，最外层在前：`rank` 个 1 到 256 的 int，每个整除块的对应项；最内层是 16 字节的倍数，在有 swizzle 时至多为 swizzle 跨度。声明把它设为块，并把其最内维截到 swizzle 跨度（一个使用 128 字节 swizzle 的 [128, 128] f16 块以两个 [128, 64] box 搬运：这正是 Triton 记录的） |
| `element_strides` | `[1, ..., 1]`（`rank` 个） |
| `interleave` | `CU_TENSOR_MAP_INTERLEAVE_NONE` |
| `swizzle` | `CU_TENSOR_MAP_SWIZZLE_NONE`、`_32B`、`_64B` 或 `_128B` |
| `l2_promotion` | `CU_TENSOR_MAP_L2_PROMOTION_NONE`、`_L2_64B`、`_L2_128B`（Triton 的）或 `_L2_256B` |
| `oob_fill` | `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`（张量外的元素读为零，写到那里的 store 被丢弃）或 `_NAN_REQUEST_ZERO_FMA`（浮点数读为 NaN；仅限浮点 dtype） |

**map 从哪里来，以及由什么检查。** 声明给出块、swizzle（默认是 128、64 和 32 字节中能整除块的行字节数的最大者；rank 为 1 或行数少于 8 的块没有 swizzle）、L2 promotion 和填充；它们在声明 digest 和编译规格中（规格的 `signature` 项是 Triton 的 `tensordesc<fp16[128, 64]>`，对于 Gluon kernel，其后跟着该 tile 的 `NVMMASharedLayout`，按 Gluon 的 `TensorDescriptor` 的写法）。编译后的 kernel 记录它对每个描述符的期望（`metadata.tensordesc_meta`：swizzle、元素类型和大小、box），构建拒绝记录与声明不符的 kernel（Triton kernel 的编译器会选择 swizzle：拒绝信息说明应该声明哪一个），以及 Triton 没有下沉到 TMA 的 kernel（sm_90 之前）。读取器按表中的规则（驱动和 Triton 拒绝的情形）检查 `tensor_map`，而不按声明推导 box 的方式检查。

**host 在启动时做什么。** 它像绑定指针一样绑定描述符的张量（从形状得到维度，从视图的规范步长得到其步长维度，dtype、rank、稠密张量的连续性、别名，16 字节的 `pointer_alignment`：map 的全局地址），然后，在第 6 节的每个前置条件之后、grid 之前，按 ABI 的顺序对每个 `tensordesc` 检查 `cuTensorMapEncodeTiled` 对张量的要求，每一项都是种类为 **`tensor_map`** 的拒绝：每个维度至多为 `2**31 - 1`（驱动接受 `2**32`；kernel 以 `i32` 接收它），最内层规范步长为 1，其他每个步长都是 16 字节的倍数且小于 `2**40` 字节（步长 0，即扩展的轴，也是倍数）。然后它编码 map：`cuTensorMapEncodeTiled(map, data_type, rank, address, globalDim, globalStrides, box, elementStrides, NONE, swizzle, l2_promotion, oob_fill)`，维度和 box 最内层在前，外层轴的步长以字节为单位、最内层在前，后面跟着整个张量的字节数（Triton 的 `fillTMADescriptor` 传入的：`rank` 项，驱动读取其中 `rank - 1` 项），元素步长为 1。Python 通过 cuda-python 调用它，crate 通过 `cudarc::driver::sys` 调用；二者计算出相同的数（`_tensormap.encode_arguments`、`TensorMapEncode`）。

**每次启动，以及在 graph 中。** map 是张量的地址、形状和步长以及声明的函数：它在**每次启动时**根据该次启动的张量编码（Python 按参数保留最近 4096 次编码，因此重复的启动不调用驱动；任何其他地址、形状或步长都是另一次编码）；对于 plan，在每次执行时为每个步骤根据该步骤绑定的缓冲区和视图（位于其偏移处的 arena 缓冲区，或调用者的张量）编码，在每个前置条件之后、第一次启动之前。在正在 capture 的 stream 上的启动会把 map **按值**捕获到 kernel 节点中：plan 的 CUDA graph 保存着它被 capture 时所用的 bucket、arena 布局和外部张量的 map，就像它保存它们的地址一样，因此只要这些地址对每次 replay 都正确，它就正确（plan schema 1，第 10 节）。绑定另一个张量就是另一次启动或另一次 capture，会编码它自己的 map；没有任何东西修补已 capture 的 map。

**设备端描述符**（`tl.make_tensor_descriptor`，把它们的 map 写进全局 scratch）自 tilefish 0.0.7 和 crate 0.0.5 起可以启动（第 4.6 节）。当张量是参数时，host 描述符仍然更便宜（每次启动在 host 上编码一次；设备端描述符则由每个 program 在设备上、在其第一次加载之前编码）。

早于这一种类的读取器（tilefish 0.0.5、crate 0.0.3）拒绝含有它的 manifest（`kind must be pointer, scalar, fixed or given`）。不含它的 manifest 及其 digest 不变。

### 5.6 空指针：`null`（tilefish 0.0.7，crate 0.0.5）

kernel 可以接受一个在所声明的用法中从不解引用的指针：一个预构建 kernel `k(const half* x, const half* bias, half* y, int n)`，只在 `bias` 不为空时读取它，或者一个 Triton kernel，只在声明设为 false 的某个 constexpr 标志下读取它。Python 把该参数声明为 `bias: Null[f16]`（仅限 extern），它的 ABI 条目是

```json
{"slot": 1, "index": 1, "name": "bias", "kind": "null", "type": "*fp16", "ptx_type": "u64", "size": 8, "dtype": "f16"}
```

* `dtype` 是 manifest 的 dtype 之一，`type` 是它的指针类型（与指针相同），以 8 字节的 `u64` 传入；它没有其他字段（没有 `access`、`shape`、`strides`、`offset`、`dim`、`count_of`、`stride_of`、`value`、`alias`、`tensor_map`）。`triton` 条目和 `binary` 条目都可以有它。
* **启动时在其槽位传入 64 位零**，就像写入 `fixed` 参数的值一样：调用不为它提供张量，它不绑定维度，也没有前置条件（指向它的 `pointer_alignment` 没有指向任何指针，会被拒绝）。plan 步骤不给它绑定缓冲区。用它编译的 Triton kernel 把该参数作为一个不带属性的指针参数（没有 `tt.divisibility`：不通过它加载任何东西）。
* 它在声明 digest 中（参数上的 `"kind": "null"` 和 dtype），因此在该位置用一个张量声明同一个 kernel 是另一个算子，有自己的条目（一个二进制文件，两个声明，两个 stem：`python/tests/types_kernels.py` 中的 `scale_null` 和 `scale_bias`）。

早于这一种类的读取器（tilefish 0.0.6、crate 0.0.4）拒绝含有它的 manifest（`kind must be pointer, tensordesc, scalar, fixed or given`）。不含它的 manifest 及其 digest 不变。

### 5.7 64 位维度标量（tilefish 0.0.7，crate 0.0.5）

携带一个维度或维度表达式（有 `dim` 而没有 `stride_of`）的 `scalar` 可以是 `i64`（`u64`，8 字节）：Python 把它声明为 `n_elem: i64[n]`（一个 extern；`n_elem: n` 仍是一直以来的 `i32`）。它与其他标量一样：它的值是在启动的张量所绑定的维度上对表达式求值的结果（它从不由调用给出，这与 `given` 参数不同），启动会拒绝放不进的值（`0 ≤ v ≤ 2^63 − 1`）。因此，接受 `long` 元素个数的预构建 kernel，或者接受一个超过 `2^31 − 1` 个元素的张量的 `i64` 个数的 Triton kernel（`index_bits=64`），在声明时其个数会按形状检查。`divisible` 前置条件和 `tt.divisibility` 对它的适用方式与对 `i32` 标量相同。声明 digest 在参数上有 `"scalar_type": "i64"`（没有它的声明的 digest 与以前相同）。两个读取器自 schema 3 开始就接受 `i64` 标量（用于步长），因此没有读取器需要改变：tilefish 0.0.6 的 manifest 从未有过只带 `dim` 的 `i64` 标量，其 digest 不变。

## 6. 前置条件

列表与 schema 2 相同，有以下不同。

* `pointer_alignment`：`bytes` 是*传入的*指针必须具有的对齐（`address % bytes == 0`）。对普通指针，它与以前一样是 16，或者（tilefish 0.0.7，crate 0.0.5）对不是设备标量的稠密指针，是其元素的字节大小：即**下沉生成的 kernel 针对不从 16 字节开始的视图的特化**（见下文）。对有 `strides` 的指针：**要么是 16，要么是其元素的字节大小**（其 dtype 的 `max(1, bits // 8)`）。16 表示算子被声明为 16 字节对齐（`Strided[..., 16]`），偏移维度 `divisible_by` 16 字节中的元素数，并且对于 Triton kernel，该指针有 `tt.divisibility=16`（见下文）；元素大小表示除元素自身的对齐外不假定任何对齐，也没有给 Triton 任何提示。设备标量的是 16。`tensordesc` 的是 16（TMA map 的全局地址），不论它描述的是稠密张量还是视图；Triton 不会为它得到任何属性。
* 新种类 **`max_extent`**：`{"kind": "max_extent", "arg": <pointer>, "index": <its position>, "value": 2147483647}`，对以 `i32` 索引的 kernel 中每个有 `strides` 的指针出现一次，在任何指针上都不超过一次（并且只出现在有 `strides` 的指针上；不允许在其上出现 `max_elements`）。它表示：对于以 `i32` 索引的 kernel（`index_bits=32`，默认），视图的 **extent** 不得超过 `value` 个元素，其中 `extent = 1 + Σ over axes of (size[a] − 1) * canonical_stride[a]`（用无界整数计算：乘积可能超过 64 位）。检验是 `extent ≤ value`。（`max_elements` 限定的是元素个数，对视图而言，它对 kernel 计算的偏移没有任何约束。）`index_bits=64` 的 kernel 在其指针上两种前置条件都没有。`value` 恰好是 `2**31 − 1`。
* 步长标量上的 `divisible`：与 schema 2 相同（`scalars[arg] % by == 0`）；当声明说明步长是 16 或更大值的倍数时出现。
* `static_size`、`bucket`、`max_elements`：不变。`max_elements`/`static_size` 不出现在设备标量上。

对于 `triton` 条目，**给 Triton 的属性**不在 manifest 中（它们在编译规格中），但可以推导出来：指针上恰好在其 `pointer_alignment` 为 16 字节时有 `tt.divisibility=16`，标量上恰好在它有 `divisible` 前置条件时有。

**下沉生成的 kernel 针对不从 16 字节开始的视图的特化**（tilefish 0.0.7，crate 0.0.5）。下沉生成的 kernel 把每个指针都按 16 字节对齐编译。`kernel.unaligned("a")`（Python）是同一个 kernel，只读参数 `a` 只按其元素对齐编译：它没有 `tt.divisibility`，`pointer_alignment` 为其元素大小。它是一个独立的算子 `<kernel>__ua<indices>`（这些参数在声明中的索引：`add_rows__ua0`），因此它的条目有自己的 stem，它的声明 digest 在这些参数上有 `"align": <bytes>`，它的 Triton IR 对它们没有可整除性提示（因此其构建缓存键不同），而它的生成源码和 `source_sha256` 与原 kernel 相同。一个 graph 如果把偏移不能证明是 16 字节倍数的稠密视图（一个 `R` 为奇数的 `[K, R]` int32 平面中的一行）传给这样的参数，就会调用该特化，因此其 plan 的步骤指向该条目；对地址不是 16 字节对齐的张量执行 `tilefish.run(kernel, ...)` 也会运行它（manifest 中缺少它时，Python 的 replay 会说明应该导出什么）。host 像启动任何条目一样启动它：它检查的对齐就是条目所承诺的。早于这一变更的读取器（tilefish 0.0.6、crate 0.0.4）拒绝稠密指针承诺少于 16 字节的条目。不含它的 manifest 及其 digest 不变。

## 7. 维度表（`dims`）

与 schema 2 相同，`dims` 把条目提到的每个命名维度映射到它的约束（`divisible_by`、`buckets`、`bucket`）。schema 3 增加了带步长指针的维度：每个步长维度（`x_s0`）和偏移维度（`x_offset`）都在其中命名，每个带 `{}` 或带它的约束（`Strided[..., 16]` 指针的偏移维度上为 `{"divisible_by": 8}`；声明说明是 `k` 的倍数的步长上为 `{"divisible_by": k}`）。在形状、`strides`/`offset` 项、标量的 `dim` 或 grid 中使用的名称必须在 `dims` 中。

**哪些张量绑定一个维度。** 启动从 kernel 读取的张量（`in` 和 `inout`：它们的形状，以及它们视图的步长）绑定维度，然后，对这些张量留下未绑定的维度，按 ABI 的顺序从给定的 `out` 张量的形状绑定：声明形状提到这种维度的 `out` 张量先单独检查（它的 dtype、rank、不为空以及连续），再用输入已绑定的值求解其形状方程。然后每个维度像其他维度一样检查：它的 `divisible_by`、它的 bucket 以及二进制文件的 `bucket`、携带它的标量。只由 `out` 张量确定的维度，例如一个只根据设备标量写出 `[B]` 掩码的 extern，`key_mask(mask: Out[Tensor[B, bf16]], n_valid: DevScalar[i32], b: B)`：它的条目中没有任何新内容，而只从 kernel 读取的张量绑定维度的 host 会因为不确定而拒绝它的启动（crate 0.0.2 和 tilefish 0.0.4 就是如此）。Python 的单次启动通过 `tilefish.run(op, ..., outputs=...)` 得到它写入的张量（没有它们时，它无法分配它们，并会说明这一点）；plan 步骤和 Rust 启动总是有它们。这种维度的声明允许用于 extern（kernel 的形状来自其输入），但从不允许用于带步长指针的偏移维度。

### 7.1 维度表达式：两个读取器都遵守的限制

形状的轴、步长维度、标量的 `dim`、`divisible` 前置条件的 `dim`、`max_elements` 前置条件的 `shape` 中的一个轴，以及每个 grid 项，都是*维度表达式*：整数、名称、`cdiv(a, b)`、`+`、`-`、`*`、`//`、一元 `-` 和括号（Python 中为 `_specvalid._ExprParser`；`crates/tilefish/src/expr.rs` 是它的移植）。**grid 项可以把维度相乘**（tilefish 0.0.7）：`@tilefish.extern(...,
grid=lambda T, C: (tilefish.cdiv(T * C, 1024),))` 写作 `(cdiv(C*T, 1024),)`，即不论 `T` 和 `C` 是多少，`[T, C]` 张量每 1024 个元素一个 program（grid 可调用对象在允许乘积的情况下求值，`tilefish.dims.products()`；因子按规范顺序写出，声明 digest 把乘积记为 `{"prod": [...]}`）。host 在每次启动时根据张量绑定的维度计算 grid，从不求解它，因此乘积在那里没有问题；形状、步长和标量仍是仿射的，因为 host 通过求解它们来绑定一次调用的维度。两个读取器一直接受这样的 grid：不是仿射的表达式按其写法求值（见下文）。一个表达式至多 512 个字符和 256 个记号，嵌套深度至多 24 层，整数字面量至多 12 位数字（`1234567890123` 是两个记号，会被拒绝）。两个读取器在读取 manifest 时都会解析每一个表达式（Python：`Manifest.load`、`_manifest._check_dimension_names`），因此二者拒绝相同的文本。

**每个 kernel 条目至多 32 个名称。** 单位是条目，而不是形状：`dims` 的名称加上上述表达式提到的每个名称，合起来至多 32 个（`expr.rs` 中的 `MAX_DIMS`，`tilefish._manifest.MAX_DIMS`）。rank 为 `r` 的带步长指针至多占用其中 `r + 1` 个（每个步长不是固定 `1` 的轴一个步长维度，加上它的偏移维度）。有第 33 个名称的条目，不论该名称在 `dims` 中还是只在表达式中，都以 "a kernel may use at most 32 dimension names" 被拒绝（Python 中为 `TilefishManifestError`，Rust 中为 `Error::Manifest`），而表达式使用了 `dims` 中没有的名称则以 "dims does not name [...]" 被拒绝。（launch plan 的 Rust 运行器也至多绑定一个 plan 的 32 个维度。）

**在 128 位中折叠。** 为了得知一个常量表达式（静态 grid 项、`//` 或 `cdiv` 的除数）的值，以及比较两个表达式，读取器把它折叠成仿射形式 `constant + Σ coefficient × atom`（atom 是一个名称或一个不能折叠的 `cdiv`）：即 Python 的 `DimExpr`。Rust 读取器在有符号 128 位整数 `[-2**127, 2**127 - 1]` 中用带检查的算术折叠，Python 的读取器在同一范围内折叠（`_specvalid.FOLD_RANGE`；它的整数是无界的，因此它做检查）。对于 12 位数字的字面量，只有四个字面量的乘积（或这种乘积的和）才会超出范围：`549755813888*549755813888*549755813888*1024` 是 `2**127`。超出范围的折叠**没有折叠值**：超出范围的常量或系数、`-2**127` 的取反，因此还有当 `b` 含有 `-2**127` 时的 `a - b`（它是 `a + (-1) × b`），不论差值会是多少。这样的表达式随后被当作非仿射表达式读取：不会因它而拒绝任何东西，需要它的值的检查也不做。因此，设 `X` 为该乘积，`Y = 549755813888*549755813888*549755813888*512`（`2**126`）：

* `(X - X,)` 对任一读取器都不是静态 grid `(0,)`，会被读取；`(Y - Y,)` 是，会被拒绝（"entry 0 is the static size 0; grid sizes must be
  at least 1"）；同样，`cdiv(n, X - X)` 会被读取，`cdiv(n, Y - Y)` 会被拒绝（"cdiv by 0 (the divisor must be positive)"）；
* 除数为符号的 `cdiv`，其商超出范围时（唯一的情况是 `-2**127` 除以 `-1`：`cdiv(-X*n, -n)`）仍是 `cdiv` atom，因此 `(cdiv(-X*n, -n)*0,)` 是静态的 `0`，会被拒绝，而 `(cdiv(-X*n, -n) - Y - Y,)` 会被读取；
* 在比较两个表达式的地方（Rust 读取器对 `count_of` 标量所携带的元素个数以及 `max_elements` 前置条件的 `shape` 的检查，以及 Python 中对编译规格的同样检查），没有折叠值的表达式不等于任何有折叠值的表达式。

`tilefish` 写出的 manifest 都远不会接近这个范围。启动时计算的是值，而不是折叠：Rust host 在 128 位中计算折叠形式的值，按写法计算非仿射表达式且每个中间结果的绝对值至多为 `2**100`，并拒绝溢出（"a dimension expression overflows"）；Python 精确计算。二者都拒绝放不进 kernel 所接收类型的值。

## 8. 预构建二进制文件（`kind: "binary"`）

预构建二进制文件是 Triton 不编译的 kernel：由 `ptxas` 汇编的 PTX，或按原样使用的 cubin。在 Python 中，它按其 C ABI 声明。

**条目。** 见第 4 节。说明它由什么组成的字段：

* `symbol`：kernel 的入口点；cubin 必须有代码段 `.text.<symbol>`（否则被拒绝）。
* `block`：`[x, y, z]`，来自声明（每个 1 到 1024，乘积 ≤ 1024）。它代替 `32 * num_warps`。
* `shared`：启动时使用的**动态**共享内存字节数，来自声明（读取器检查 `static + shared ≤` 设备的 opt-in 最大值，与 Triton 相同）。
* `grid`：关于维度名称的表达式，与 schema 2 相同。
* `launch`：零值 `{"global_scratch_size": 0, "profile_scratch_size": 0, "launch_pdl": false, "cluster_dims": [1,1,1], "num_ctas": 1,
  "cooperative": false}`，因此一次普通的 `cuLaunchKernel` 就能运行它；或者，对于声明了 cluster 的二进制文件（`cluster=(2, 1, 1)`，第 4.5 节），是该值加上它的 `cluster_dims` 和作为 `num_ctas` 的乘积，以 `cuLaunchKernelEx` 启动（其 grid 计的是 cluster）。
* `ptx_target`：对 PTX，是汇编该 PTX 时的 `--gpu-name`（PTX 有 `.target sm_103a` 时为 `sm_103a`，否则为 `sm_<arch>`）；对 cubin 为 `""`。
* `binary`：`{"source": "cubin"}` 或 `{"source": "ptx", "ptxas_flags": [<strings>]}`（恰好这些键；flags 是在文件名之前传给 `ptxas` 的参数向量：`["-lineinfo", "-v", "--gpu-name=sm_103a"]`，即 Triton 3.5 传入的 flags，拼写相同，因为 ptxas 会把它的命令行记录在 cubin 中，机器码 digest 依赖于它）。
* `source_digest`：PTX 文件字节的 sha256（64 个小写十六进制数字），或用于该 `arch` 的 cubin 文件字节的 sha256；`source_digest_kind` 是 `"tilefish.binary.v1"`。对于 cubin 来源，它等于 `sha256`。
* `toolchain`：对 PTX 为 `{"ptxas": <version text>}`，对 cubin 为 `{}`。`declaration_digest`、`sha256`、`sha256_code`：与任何条目相同。

**启动。** `cuModuleLoadData(cubin)`、`cuModuleGetFunction(symbol)`，然后 `cuLaunchKernel(grid, block, shared)`，其 `kernelParams` = 按顺序排列的 `abi.params`：`pointer` 为其地址（`u64`），`scalar` 为其值（`i32`/`i64`），`fixed` 为其 `value`（`i32`/`i64`/`f32`），`given` 为调用给出的值（`i32`/`i64`）；**不追加任何东西**（`scratch_slots` 为 0）。驱动能回答时，`cuFuncGetParamInfo` 必须恰好报告每个参数的 `size`，不多不少（Python 在加载时检查这一点）。cubin 用于其自身以外的架构时会被拒绝：声明中每个架构一个 cubin，在 `sm_90` 上的 replay 找不到条目（也不会使用 `sm_103` 的那个）。

**身份与过期。** 二进制文件的身份是 `source_digest`。replay 将它与声明所持有字节的 digest 比较（声明在算子被声明时读取一次它的文件），不匹配视为过期而拒绝，不论是否 pin；判断这一点不需要 Triton，也不需要 GPU。没有 IR digest。

**每个架构一个实现。** 预构建二进制文件可以是用 `by_arch=` 声明的算子的一个实现（`tilefish.impl(ptx=...)`、`tilefish.impl(cubin={...})`，第 4.1 节）：它的条目有 `implementation`，它在声明 digest 中；它的 ABI 是它自己的（它的 C 参数，没有 scratch 槽位），不论其他架构的实现的 ABI 是什么。cubin 实现对运行它的每个架构都有一个 cubin。

**前置条件和 dims** 与 Triton kernel 一样来自声明：指针对齐、形状、`divisible`、`max_elements`（默认的 `index_bits=32` 也适用于预构建 kernel：如果它以 64 位索引，作者要写明 `index_bits=64`）、bucket。预构建二进制文件不由 Triton 编译，因此没有 `tt.divisibility` 属性；前置条件就是对 kernel 的承诺。

## 9. `library` 段：一个已 pin 的 cuBLASLt GEMM

`library` 条目描述由厂商库运行、没有文件的操作：目前是一个通过 cuBLASLt、算法已 pin 的 bf16、f16 或 f32 操作数的 GEMM。

**操作。** 用 tilefish 的行主序术语表示，`out[M, N] = a[M, K] @ b[N, K]^T`（对 `impl: "cublaslt.gemm_bias"` 还有 `+ bias[N]`）：`a`、`b`、`bias` 和 `out` 是 bf16、连续、16 字节对齐的；累加器是 f32；结果**一次**舍入到 bf16；bias 在这次舍入之前以 f32 加到累加器上（`CUBLASLT_EPILOGUE_BIAS`）。这是 tilefish 0.0.5 的形式，也是它写出的每个条目的形式；第 9.2 节说明其他形式：f16 和 f32 操作数、f32 结果、以 `[K, N]` 存储的 `b`（`out = a @ b`，布局 `NN`）以及以 `[K, M]` 存储的 `a`（布局 `TT` 和 `NT`）。`M`、`N`、`K` 是静态的或分桶的：每种 bucket 组合一个条目（stem 命名 bucket，与 kernel 的相同）。`N` 和 `K` 可以是任意大小（tilefish 0.0.5；0.0.4 和 crate 0.0.2 要求是 8 的倍数）：当 `k` 或 `n` 个元素的一行不从 16 字节开始时，leading dimension 决定哪些算法有效，而 `cublasLtMatmulAlgoCheck`（它能看到 leading dimension）在算法被 pin 时、host 加载它时，以及在 Python 中每次运行之前，判断已 pin 的算法；指针仍是 16 字节对齐的。

**字段**（恰好这些；读取器拒绝缺失的和未知的字段；`python/src/tilefish/_library.py`，`ENTRY_KEYS`，`validate_entry`）：

| 字段 | 类型与规则 |
|---|---|
| `stem` | 字符串，`op` 加上每个分桶维度的 `.<Dim><value>`，按维度名称排序的顺序（`gemm.M64`、`gemm_bias.Kb1024.M32`）；必须等于 `op` + 各 bucket |
| `op` | 标识符 |
| `arch` | `sm_<2 or 3 digits>`，没有后缀（`sm_103`） |
| `impl` | `"cublaslt.gemm"` 或 `"cublaslt.gemm_bias"` |
| `abi` | `{"params": [...]}`，恰好这一个键；指针 `a`、`b`、[`bias`]、`out`，按此顺序（见下文） |
| `buckets` | 对象：维度名称 → 它的 bucket 值（int ≥ 1） |
| `dims` | 第 7 节的表（`divisible_by`、`buckets`、`bucket`），每个分桶维度都带着它的 `bucket` |
| `preconditions` | 列表：每个指针一个 16 字节的 `pointer_alignment`，每个静态轴一个 `static_size`，每个分桶维度一个 `bucket`；没有其他种类 |
| `dtypes` | `{"a": "bf16", "b": "bf16", "out": "bf16"}`，bias 算子再加上 `"bias": "bf16"`（其他类型：第 9.2 节） |
| `compute_type` | `"CUBLAS_COMPUTE_32F"` |
| `scale_type` | `"CUDA_R_32F"` |
| `transa`, `transb` | `"T"` 和 `"N"`：**调用 cuBLASLt 时**的转置（见下文）；对于其他布局（第 9.2 节），`b` 为 `[k, n]` 时 `transa` 是 `"N"`，`a` 为 `[k, m]` 时 `transb` 是 `"T"` |
| `m`, `n`, `k` | 1 到 2147483647 的 int：**tilefish** 的大小，`a` 是 `[m, k]`（`TT` 和 `NT` 为 `[k, m]`），`b` 是 `[n, k]`（`NN` 和 `NT` 为 `[k, n]`），`out` 是 `[m, n]` |
| `lda`, `ldb`, `ldc`, `ldd` | int：调用 cuBLASLt 时的 leading dimension；恰好 `lda = ldb = k`，`ldc = ldd = n`（`b` 为 `[k, n]` 时 `lda = n`，`a` 为 `[k, m]` 时 `ldb = m`） |
| `epilogue` | `gemm` 为 `"none"`，`gemm_bias` 为 `"bias"` |
| `workspace_bytes` | 1 到 2^40 的 int：询问启发式时所用的 workspace 上限，**也是启动提供的 workspace 的大小**（默认 `8519680`，torch 默认的 `CUBLAS_WORKSPACE_CONFIG`） |
| `algo` | 128 个小写十六进制数字：`cublasLtMatmulAlgo_t` 的 64 字节，按内存顺序（八个 `uint64_t`，与 host 存储它们的方式相同） |
| `cublaslt_version` | int ≥ 1：选择 `algo` 的那个库的 `cublasLtGetVersion()` 值（例如 12.8.4 为 `120804`） |
| `sm_count` | int ≥ 1（JSON 整数：不是 `30.0`，不是 `true`，不是 `"30"`；没有上限）：`algo` 被 pin 时所在设备的多处理器数量（`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`，torch 的 `multi_processor_count`）。MIG 切片的 SM 比整个 GPU 少（B300 的 1g 切片有 30 个，整个 GPU 有 148 个），按 SM 数量划分工作的算法（stream-K）是针对该数量选出的 |
| `declaration_digest` | 64 个小写十六进制数字：对规范声明（名称、impl、workspace 上限、带维度约束的签名，以及第 9.1 节中不是默认值的选项）**和 `sm_count`**（加上键 `"sm_count"` 的规范声明对象，格式 2）计算的 sha256：一个 pin 是一个设备上的声明 |
| `heuristic` | *可选*（第 9.1 节）：不存在，或恰好是 `"torch"` |
| `algo_source` | *可选*（第 9.1 节）：不存在，或恰好是 `"imported"` |
| `allow_bf16_reduction` | *可选*（第 9.1 节）：不存在，或恰好是 `true`；只用于 bf16 操作数的条目 |
| `allow_f16_reduction` | *可选*（第 9.2 节）：不存在，或恰好是 `true`；只用于 f16 操作数的条目 |
| `allow_f32_reduction` | *可选*（第 9.2 节）：不存在，或恰好是 `true`；只用于 f32 操作数的条目 |
| `sm_count_target` | *可选*（第 9.3 节）：不存在，或 8 到 `sm_count` 的 int：启发式据以选择的 SM 数量，每次检查和运行都会设置它 |

`abi.params[i]` 恰好有 `index`（= i）、`name`、`kind`（`"pointer"`）、`dtype`（它在 `dtypes` 中的类型：0.0.5 的形式中为 `"bf16"`）、`access`（`"in"`，最后一个为 `"out"`）、`shape`（int 和维度表达式的列表：`a` 的 rank 为 2，`b` 为 2，`bias` 为 1，`out` 为 2）。形状必须与 `m`、`n`、`k`（以及布局）一致：整数项等于该大小，表达式项在分桶维度取其 `buckets` 值时求值为该大小（提到没有 bucket 的维度的表达式会被拒绝）。

**cuBLASLt 问题。** cuBLASLt 是列主序的，因此 tilefish 的 `a`、`b`、`out` 以 TN gemm `m_cublas = n`、`n_cublas = m`、`k_cublas = k` 传入：cuBLAS 的 "A" 操作数是 tilefish 的 **`b`**（`transa = T`，一个 `lda = k` 的 `k × n` 列主序矩阵），cuBLAS 的 "B" 操作数是 tilefish 的 **`a`**（`transb = N`，`k × m`，`ldb = k`），`C` 和 `D` 是 tilefish 的 `out`（`n × m` 列主序，`ldc = ldd = n`）。`alpha = 1.0f`，`beta = 0.0f`，host 指针模式。所有矩阵布局都是 `CUDA_R_16BF`，列主序；matmul 描述符有 `compute_type`、`scale_type`、`TRANSA`、`TRANSB`，对于 bias 算子还有 `EPILOGUE = CUBLASLT_EPILOGUE_BIAS`、`BIAS_DATA_TYPE = CUDA_R_16BF` 以及 `BIAS_POINTER` = bias 指针。Python 的绑定（`python/src/tilefish/_cublaslt.py`，`describe`）是每个值的参考，`python/tests/test_cublaslt.py` 检查这张表与 `describe` 一致。

**replay。**

1. 像对 kernel 一样从形状绑定维度并检查前置条件（对齐、静态大小、精确的 bucket、连续性）。
2. 如果设备的多处理器数量不是 `sm_count`，则拒绝，错误信息提示在这个设备上重新导出 manifest（Python 在比较条目的任何其他字段之前：`TilefishManifestError`，"the manifest pinned the cuBLASLt algorithm of '<stem>' (<arch>) on a device with X SMs,
   but this device has Y: ... re-export the manifest on this device"）。条目按 `(stem, arch)` 查找，而同一架构的两个设备的 SM 数量可能不同（MIG 切片和整个 GPU），因此仅凭架构无法说明 pin 适用于这个设备。
3. 如果进程的 `cublasLtGetVersion()` 不是 `cublaslt_version`，则拒绝：一个算法 id 在另一个版本中对应其他 kernel。（Python：`TilefishManifestError`，"the manifest pinned the cuBLASLt algorithm of '<stem>' (<arch>) with cuBLASLt X, but this process has cuBLASLt Y"。）**不比较构建号，因为 cuBLASLt 不公开构建号**：`cublasLtGetVersion()` 是 `major * 10000 + minor * 100 + patch`，`cublasLtGetProperty` 回答的也是这三个值（`MAJOR_VERSION`、`MINOR_VERSION`、`PATCH_LEVEL`）；12.8.4 的库没有其他版本或构建的入口点（已检查）。因此，这个检查无法区分同一版本的两个构建。
4. 从 `algo` 重建 `cublasLtMatmulAlgo_t`；针对该问题询问 `cublasLtMatmulAlgoCheck`（算法必须有效，其 `workspaceSize` 必须 ≤ `workspace_bytes`），然后在启动 stream 上用至少这么大的 workspace 调用 `cublasLtMatmul`。调用者给出的 workspace（launch plan 的 arena 缓冲区，Rust 启动的最后一个张量）是 **256 字节对齐**的，两个 host 都要求这一点。**`cublasLtMatmulAlgoCheck` 对字节的判断很弱**：在 12.8.4 上测得，它接受被改动的版本字段、被改动的填充、保留的 CTA swizzling 值和无意义的自定义选项，并且在 tile id ≥ 2^31 时会使进程崩溃。因此 Python 绑定首先用 `cublasLtMatmulAlgoInit` + `cublasLtMatmulAlgoConfigSetAttribute`，按这些字节读回的配置重建算法，并要求字节完全相同；要求 tile 和 stages 的 id 在该算法 id 的能力列表中，swizzling 和自定义选项在范围内，f32 累加器（对于 f32 操作数还要求 f32 乘积：第 9.2 节），对齐能力 ≤ 16 字节，split-K（K 维切分）配合归约方案 `NONE` 或 `COMPUTE_TYPE`（f32 部分和；bf16 部分和会舍入不止一次），并且 `gemm_bias` 完全不能有 split-K（使用 bias epilogue 时，测得 f32 部分和的 split-K 会把 bias 加到已经舍入为 bf16 的和上：在 RTX 5090、sm_120 上的 cuBLASLt 12.8.4，算法 id 21，其中 32x1024x3072 有 32768 个元素中的 8409 个不是正确舍入的结果，超出了一次舍入加 f32 累加所允许的界限，而同一算法不带 bias 或不用 split-K 时则保持在界限内）；或者，对于 `allow_bf16_reduction: true` 的条目，还允许 `INPLACE`（1）或 `OUTPUT_TYPE`（4），以及带 bias 的 split-K。这条规则及其措辞是 `tilefish._cublaslt.reduction_problem`（以及 crate 的 `reduction_problem`，`gemm.rs`）：没有 split-K 就没有方案，有 split-K 就有一个方案；0、1、2 和 4 以外的方案被拒绝。运行 `library` 条目的 host 应当这样做。任何字节检查都看不出的是翻转成*另一个有效*算法：只有 manifest 的 digest 为文件担保。
5. pin 属于一个 `arch` 和一个 SM 数量：条目按设备架构的 `(stem, arch)` 查找，缺少该对是一个错误，错误信息列出 manifest 中有的架构；找到的条目必须有设备的 `sm_count`（第 2 步）。

**pin（条目如何生成）**，供写入方参考：在 `arch` 的设备上询问启发式，偏好设置为 `MAX_WORKSPACE_BYTES =
workspace_bytes`、`MIN_ALIGNMENT_{A,B,C,D}_BYTES = 16`，以及归约方案掩码 `NONE | COMPUTE_TYPE`（对 `gemm_bias` 只用 `NONE`，即掩码 `0`：第 4 步），请求若干个结果（另一种姿态（posture），或导入的算法：第 9.1 节）；结果 0 就是 pin，设备的多处理器数量就是 `sm_count`。pin 是结果 0 的配置（它的九个 `CUBLASLT_ALGO_CONFIG_*` 属性）用 `cublasLtMatmulAlgoInit` 和 `cublasLtMatmulAlgoConfigSetAttribute` **重新编码**的结果，而不是启发式返回的那 64 字节：那些字节可能含有不属于配置的字节（在 RTX 5090、sm_120 上用 cuBLASLt 12.8.4 测得：id 21 的 split-K 算法的第 34 和 35 字节，在不同进程中会变化），因此它返回的字节既不可复现，也不是第 4 步的检查所接受的。配置就是 host 重建和检查的全部内容，重新编码的算法运行出的字节与 torch 用返回的算法得到的字节相同（第 9.1 节，`test_the_torch_posture_pins_what_torch_runs`）。写入条目之前，其算法要通过一次运行的检查（第 4 步）。`jit` 以同样的方式在内存中 pin，针对它运行所在的设备（它在内存中的 pin 按设备保存），因此 `jit` 和 `replay` 在构造上就运行同一个算法。pin 不与其问题绑定（同样的字节对其他问题也会被接受），这就是 `m`、`n`、`k` 和 epilogue 在条目中的原因。在另一个 SM 数量的设备上重新导出，会为同一个 `(stem, arch)` 生成另一个条目，只有在 `overwrite=True` 时才替换旧条目（它的 `sm_count` 和 `declaration_digest` 不同）。

**digest。** 条目在 `library` 中整体计入 digest（第 3 节）；`algo`、`cublaslt_version` 和 `sm_count` 都在其中（`sm_count` 也在它的 `declaration_digest` 中）。

### 9.1 算法是如何选出的：三个可选字段（tilefish 0.0.5，crate 0.0.3）

声明说明它的算法如何选出（`@tilefish.library(..., heuristic=, allow_bf16_reduction=, algos=)`，`python/src/tilefish/_library.py`）。它的条目用三个字段表达这一点，这些字段**在声明使用默认值时不存在，只以它们唯一的另一个值出现**；读取器拒绝任何其他值，包括 `null` 和显式写出的默认值（Python `validate_entry`、`OPTIONAL_ENTRY_FIELDS`；Rust `LibraryEntry::from_raw`）。因此，一个不使用其中任何一个的声明，其条目与 tilefish 0.0.4 写出的完全相同（它的 digest 和 `declaration_digest` 不变），而 0.0.4 或 crate 0.0.2 的读取器会拒绝含有其中之一的条目（未知字段），否则它会违背其含义运行该条目。

| 字段 | 不存在（默认） | 存在 |
|---|---|---|
| `heuristic` | 上面的姿态：若干个结果，`MIN_ALIGNMENT_{A,B,C,D}_BYTES = 16`，启发式看到的 bias 指针是一个 16 字节对齐的占位指针 | `"torch"`：torch 2.9 为 `torch.addmm(bias, a, b.t())`（`at::cuda::blas::gemm_and_bias`）询问的方式：**一个**结果，只在 `MAX_WORKSPACE_BYTES` 下（torch 把对齐最小值设为其指针的对齐，即来自其分配器的 256 字节，这也是库的默认值，因此不设置任何值），一个 256 字节对齐的占位 bias 指针。此时声明默认的 `workspace_bytes` 是 torch 的：`cublaslt.gemm_bias` 为 `1048576`（`CUBLASLT_WORKSPACE_SIZE`），`cublaslt.gemm` 为 `8519680`（`torch.mm` 在一个带有 `CUBLAS_WORKSPACE_CONFIG` 的 workspace 的句柄上运行 `cublasGemmEx`，以这种方式在该上限下询问的启发式会回答它所运行的算法；torch 的默认值恰好在 sm_90 上是 32 MiB） |
| `allow_bf16_reduction` | 以归约方案掩码 `NONE \| COMPUTE_TYPE` 询问启发式（`gemm_bias`：`0`，没有 split-K），split-K 以 bf16 相加部分和的算法，或 `gemm_bias` 的 split-K，会被拒绝（第 4 步） | `true`：没有掩码（torch 的默认值，`allow_bf16_reduced_precision_reduction = True`；掩码 `NONE \| COMPUTE_TYPE` 是 torch 在它为 `False` 时设置的），并接受 `INPLACE`（1）、`OUTPUT_TYPE`（4）以及 `gemm_bias` 的 split-K：结果可能被舍入不止一次 |
| `algo_source` | `algo` 是启发式的结果 0，经过重新编码 | `"imported"`：由声明的 `algos=` 给出（`tilefish.AlgoTable(arch, sm_count, cublaslt_version, {(m, n, k, bias): config})`，每个设备一张表；配置是九个属性，用 `cublasLtMatmulAlgoInit` 和各 setter 编码，或者是按原样接受的 128 个十六进制数字）。不询问启发式；写入条目之前算法要通过第 4 步；声明的表都不适用的设备在构建时被拒绝（从不交给启发式）。声明使用的表的行（以及各表的设备）在它的规范声明中，因此其中一行的变化会改变 `declaration_digest`；`replay` 也会把 `algo` 与表为该设备给出的值比较 |

这些字段对于运行 pin 的 host 只是来源信息（它在任何情况下都运行 `algo`），但 `allow_bf16_reduction` 除外，第 4 步需要它。使用 `"torch"` 姿态和 `allow_bf16_reduction` 时，已 pin 的结果与 torch 的逐字节一致（`python/tests/test_gpu_cublaslt.py`，`test_the_torch_posture_pins_what_torch_runs`，在 48 种形状上，带和不带 bias，每种三组操作数；在 B300 MIG 切片 sm_103 和 RTX 5090 sm_120 上用 cuBLASLt 12.8.4 测得）。

### 9.2 f16 和 f32 操作数、f32 结果、布局 `NN`、`TT` 和 `NT`（tilefish 0.0.6，crate 0.0.4）

一个 GEMM 的形式是其操作数的类型、其结果的类型以及 `b` 的存储方式。条目在它一直就有的字段中表达这些，因此 schema 不变，tilefish 0.0.5 的条目（bf16，`TN`）与以前逐字节一致，digest 也相同。0.0.5 或 crate 0.0.3 的读取器会拒绝其他形式的条目（它的 `dtypes` 或 `transa` 不是它认识的），这是正确的：它会把它当作 bf16 `TN` 运行。

| | 0.0.5 的形式 | 其他形式 |
|---|---|---|
| 操作数 `a`、`b` | bf16 | bf16、f16 或 f32，两者相同（`dtypes.a`、`dtypes.b`，指针 `a` 和 `b` 的 `dtype`） |
| 结果 `out` | bf16 | 操作数的类型，或 f32（`dtypes.out`）；f32 操作数的结果为 f32（f32 操作数没有 16 位结果） |
| `bias` | bf16 | 结果的类型（`dtypes.bias`）：cuBLASLt 不接受其他类型。在 12.8.4（sm_103）上测得：带 16 位结果的 f32 bias，以及带 f32 结果的 16 位 bias，启发式都返回 `CUBLAS_STATUS_INVALID_VALUE`，因此声明不能有它们 |
| 布局 | `TN`：`b` 是 `[n, k]`，`out = a @ b^T`；`transa = "T"`，`lda = k` | `NN`：`b` 是 `[k, n]`，`out = a @ b`；`transa = "N"`，`transb = "N"`，`lda = n`，`ldb = k`，`ldc = ldd = n`；指针 `b` 的形状为 `[k, n]`。`TT`：`a` 是 `[k, m]`，`out = a^T @ b^T`；`transa = "T"`，`transb = "T"`，`lda = k`，`ldb = m`。`NT`：`a` 是 `[k, m]`，`b` 是 `[k, n]`，`out = a^T @ b`；`transa = "N"`，`transb = "T"`，`lda = n`，`ldb = m`。`TT` 和 `NT` 的指针 `a` 的形状为 `[k, m]` |
| 选择以输出类型累加 split-K 部分和 | `allow_bf16_reduction` | 按**操作数的**类型命名，与 torch 的标志一样：f16 操作数为 `allow_f16_reduction`（不论结果是什么类型），f32 操作数为 `allow_f32_reduction`。每个都在其他类型的条目上被拒绝 |

cuBLASLt 的调用方式：`a` 和 `b`（cuBLAS 的 B 和 A 操作数）的布局为操作数的类型，`C` 和 `D` 为结果的类型和 `D` 的布局，`BIAS_DATA_TYPE` 为结果的类型；对 `NN` 和 `NT`，cuBLAS 的 A 操作数是 `b [k, n]` 所对应的 `n x k` 列主序矩阵，配合 `transa = N` 使用；对 `TT` 和 `NT`，cuBLAS 的 B 操作数是 `a [k, m]` 所对应的 `m x k` 列主序矩阵，配合 `transb = T` 使用。`cublasLtMatmulAlgoInit` 把四种类型写进算法，因此一个 pin 的 64 字节针对的是一对操作数类型和结果类型（而不是一种布局）：第 4 步用条目的类型重建算法，其他类型的 pin 在那里被拒绝（"its bytes are not those
cuBLASLt encodes"）。第 4 步的规则相同，只是措辞换成该形式的：split-K 部分和 "in f16"，或者对于 f32 结果 "in the output
(f32)"，需要操作数类型的 opt-in。

**以 `[k, m]` 存储的 `a`：布局 `TT` 和 `NT`。** `@tilefish.library(..., layout="TT")` 接受 `a: Tensor[K, M, T]` 和 `b: Tensor[N, K, T]`，计算 `a^T @ b^T`；`layout="NT"` 接受它和 `b: Tensor[K, N, T]`，计算 `a^T @ b`。布局按调用 cuBLASLt 时的转置命名（`transa` 对应 `b`，`transb` 对应 `a`），因此条目在 `transa`、`transb`、`lda`、`ldb` 和 `a` 的形状中表达它，crate 0.0.3 的读取器会拒绝它（它的 `transb` 不是 `"N"`）。上面的每种操作数类型和结果、bias、opt-in、`algos=`（以 `(m, n, k, bias, "TT")` 为键的一行）以及 `sm_count_target` 都与其他布局相同。在 `heuristic="torch"` 下并带有操作数类型的 opt-in 时，对于 bf16、f16 和 f32（TF32 关闭）操作数，pin 与 torch 对连续的 `a [k, m]` 所运行的 `torch.mm(a.t(), b.t())`（`TT`）和 `torch.mm(a.t(), b)`（`NT`），以及带一维 bias 的 `torch.addmm`，逐字节一致：`python/tests/test_gpu_gemm_layouts.py`，6 种形状（`K` 从 8 到 3072），`M` 取 2 到 2048 中的 6 个值，每种两组操作数，在 B300 MIG 切片（sm_103）上用 cuBLASLt 12.8.4 测得；对于 f32，这就是 cuBLAS 为两个转置都开启的 `cublasSgemm` 选择的算法。在 `M = 1` 时，`a [k, 1]` 也是一个连续的 `[1, k]`，torch 把它当作 `TN`（或 `NN`）相乘：此时 torch 的字节是该布局的字节（测得：bf16 `a [1024, 1].t()` 的 `torch.addmm` 在 6 种形状中的 2 种上与 `TT` pin 不同），因此只有一行的声明使用 `TN` 或 `NN`。同一测试在 RTX 5090（sm_120）上用 cuBLASLt 13.1.0 也通过，例外是在那里、`M = 2` 时，启发式对 `TT`（12 个中的 11 个）和 `NT`（12 个中的 5 个）的某些 f32 问题回答算法 10、split-K 数为 0、归约方案为 1，第 4 步的审查在两个 host 中都拒绝它（"split-K count 0 is below 1"）：这样的 pin 不会生成，因此在那里，存储为 `[k, 2]` 的 f32 GEMM 使用另一个算法（`algos=`）或布局 `TN`/`NN`。

`workspace_bytes` 是按声明的，可以是至多 2^40 的任意大小（64 MiB 的上限是 `64 << 20`）：启发式在它之下被询问，启动提供它。

第 9.1 节的 `"torch"` 姿态对 f16 也与 torch 一致：`@tilefish.library(..., heuristic="torch", allow_f16_reduction=True)` pin 的正是 `torch.mm(a, b.t())` / `torch.addmm(bias, a, b.t())` 在 f16 下运行的（`TN`），以及 `torch.mm(a, b)` / `torch.addmm(bias, a, b)` 运行的（`NN`），逐字节一致：`python/tests/test_gpu_gemm_f16.py`，`test_the_torch_posture_pins_what_torch_runs_in_f16`，63 种形状（`M` 从 1 到 2048，`N` 和 `K` 取自投影以及作为 GEMM 的卷积），每种三组操作数，两种布局，带和不带 bias，在 B300 MIG 切片（sm_103）上用 cuBLASLt 12.8.4 测得。

布局 `NN` 的 `AlgoTable` 行在其键中表达这一点，`(m, n, k, bias, "NN")`（`(m, n, k, bias)` 是 `TN` 的行）；配置用导入它的声明的类型编码。当布局为 `NN` 时，规范声明有 `"layout": "NN"`（当 `K = N` 时，仅凭签名无法区分布局），在选择 opt-in 时有 `"allow_f16_reduction": true`（`"allow_f32_reduction": true`）；类型在它的签名中。

**f32 操作数，TF32 关闭。** f32 操作数的条目（`dtypes` 全为 `"f32"`，结果和 bias 也是）按 f32 相乘：它的 `compute_type` 是 `CUBLAS_COMPUTE_32F`，`scale_type` 是 `CUDA_R_32F`，即每个条目都有的值，从不是 `CUBLAS_COMPUTE_32F_FAST_TF32`（torch 在 `allow_tf32` 时选择的）、`_FAST_16F`、`_FAST_16BF` 或 `CUBLAS_COMPUTE_32F_EMULATED_16BFX9`（cuBLAS 12.9 及以后：用九次 bf16 乘积模拟 f32 乘积）。除了第 9 节的第 4 步之外，host 还必须要求算法满足以下各点（在 B300 MIG 切片 sm_103 上用 cuBLASLt 12.8.4 和 13.1.0 测得，在 RTX 5090 sm_120 上用 13.1.0 测得；数值取自 `nvidia-cublas` wheel 12.8.4.1 和 13.1.0.3 的头文件）：

* **compute type 在字节中。** `cublasLtMatmulAlgoInit` 把它以小端序 `int32` 写在算法的第 52 到 55 字节（对 `CUBLAS_COMPUTE_32F` 为 `0x44`，即 68），因此第 4 步的字节比较已经会拒绝为另一种 compute type 编码的 pin；而 TF32 和模拟 kernel 的 id（72 和 73，sm_10x 上的 CUTLASS 3 `tf32gemm`；21、23、43、45、58、63；77 和 78，带 `bf16x9_inf_patching` kernel 的 `9xbf16gemm`）根本无法在 `CUBLAS_COMPUTE_32F` 下编码（`CUBLAS_STATUS_NOT_SUPPORTED`），因此它们作为未知 id 被拒绝。
* **环境可以把 `CUBLAS_COMPUTE_32F` 变成另一种 compute type**，编码会显示这一点，而 matmul 描述符不会（它在每种情况下都读回 68）：设置 `NVIDIA_TF32_OVERRIDE=1` 时，cuBLASLt 把 `CUBLAS_COMPUTE_32F` 编码为 `_FAST_TF32`（77），其启发式回答 TF32 tensor-core kernel（sm_103 上为 `cutlass3x_sm100_tensorop_*tf32gemm`，sm_120 上为 `cutlass_80_tensorop_s1688gemm`）；设置 `CUBLAS_EMULATE_SINGLE_PRECISION=1` 时（cuBLAS 13.1；12.8 没有模拟功能，会忽略它），它编码为 `_EMULATED_16BFX9`（78），在 sm_103 上其启发式回答 bf16x9 模拟。单独设置 `CUBLAS_EMULATION_STRATEGY`，以及 `NVIDIA_TF32_OVERRIDE=0`，对 `CUBLAS_COMPUTE_32F` 都没有影响。因此**两个 host 都读取新编码算法的 compute type**（在 `CUBLAS_COMPUTE_32F` 下对 pin 的 id 调用 `cublasLtMatmulAlgoInit` 所得的第 52 到 55 字节），不是 68 时拒绝 f32 条目："cuBLASLt encodes CUBLAS_COMPUTE_32F as CUBLAS_COMPUTE_32F_FAST_TF32 (77) in this
  process, which does not multiply f32 operands as f32 (NVIDIA_TF32_OVERRIDE=1 turns on TF32, CUBLAS_EMULATE_SINGLE_PRECISION=1 a bf16x9 emulation):
  an f32 GEMM is not run here"。在这样的进程中，每个 f32 pin 都被拒绝，不论是在别处生成的（它的字节也不是该进程的编码）还是它自己的，因此在那里的导出不会写出任何 f32 pin（`test_gpu_gemm_f32.py`）。
* **数值实现标志**（`CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS`），在库报告它们时，必须表明 f32 累加器（`ACCUMULATOR_32F`）和 f32 输入（`INPUT_32F` 且没有其他输入位：`INPUT_TF32` 是 TF32 kernel）："its products read the f32 operands as tf32,
  not as f32 (CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS 0x40202)"。在 `CUBLAS_COMPUTE_32F` 下，f32 的 id 0、1、10 到 16、20、55 和 57 报告 `0x80201`（`FMA`、`ACCUMULATOR_32F`、`INPUT_32F`），id 80 报告 `0x201`（没有输入类型：被拒绝），而 **id 76 报告 0**：完全没有标志。id 76 是 CUTLASS 3 的 SIMT sgemm（`cutlass3x_sm100_simt_sgemm_*` kernel），是 sm_103 上大多数 f32 问题的启发式答案（两种姿态、12.8.4 和 13.1.0 下所询问的 56 个中的 44 个），也是 torch 的 f32 `mm` 在那里运行的，其误差在 f64 结果的一次 f32 累加之内。因此，不报告标志的 f32 算法只按其 compute type 检查（上面两点）；16 位的算法仍因此被拒绝（"it does not accumulate in f32"），与以前相同。TF32 和模拟的 id 也报告 0，或报告 `INPUT_TF32`（`0x40201`、`0x40202`），这就是仅凭标志无法把它们排除在外的原因。（`CUBLASLT_ALGO_CAP_FLOATING_POINT_EMULATION_SUPPORT` 是 cuBLAS 13 新增的，只有 id 77 和 78 为 1；没有 host 读取它。）

这条规则的确切措辞是 `tilefish._cublaslt.numerics_problems` 和 crate 的 `numerics_problems`（`gemm.rs`）。

**f32 操作数的 opt-in。** 对于 f32 结果，split-K 归约方案 `INPLACE`（1）和 `OUTPUT_TYPE`（4）在输出中而不是在 compute type 的 workspace 中相加部分和；没有 `allow_f32_reduction` 时它们被拒绝，与其他形式相同。torch 对 f32 没有标志：它从不在 f32 GEMM 上设置归约方案掩码，因此它的选择需要这个 opt-in（在 `heuristic="torch"` 下，cuBLASLt 13.1.0 在 sm_103 上对 `test_gpu_gemm_f32.py` 的 252 个问题中的 44 个、在 sm_120 上对其中 122 个选择 `INPLACE` split-K）。

**torch 运行什么。** TF32 关闭的 torch 2.9.1（`torch.backends.cuda.matmul.allow_tf32 = False`、`torch.set_float32_matmul_precision("highest")`，即它的默认值）把 f32 张量的 `torch.mm` 作为 `cublasSgemm`（数学模式 `CUBLAS_DEFAULT_MATH`）运行在一个 workspace 为 `CUBLAS_WORKSPACE_CONFIG` 所设的句柄上（在 sm_90 以外默认为 8 519 680 字节；`:4096:8` 为 32 MiB），把带一维 bias 的 `torch.addmm` 作为 cuBLASLt 的 `gemm_and_bias` 运行（`CUBLAS_COMPUTE_32F`，没有归约方案掩码，其 cuBLASLt workspace 为 1 MiB）。使用这些 workspace 的 `@tilefish.library(..., heuristic="torch", allow_f32_reduction=True)`（该姿态的默认值：`cublaslt.gemm` 为 8 519 680，`cublaslt.gemm_bias` 为 1 048 576；在 `:4096:8` 下为 `workspace_bytes=32 << 20`）pin 的正是 torch 运行的，逐字节一致：`test_the_torch_posture_pins_what_torch_runs_in_f32_with_tf32_off`（63 种形状，`M` 从 1 到 2048，三组操作数，`mm` 和 `addmm`，`TN` 和 `NN`：756 次运行）在 sm_103 上用 cuBLASLt 12.8.4 和 13.1.0、在 sm_120 上用 13.1.0，使用默认 workspace，以及在单独的进程中使用 `CUBLAS_WORKSPACE_CONFIG=:4096:8`（各另外 756 次运行），都没有一次不一致。`cublasSgemm` 不是 cuBLASLt 调用，但对这里的每一种形状，包括 `M = 1`（一个 GEMV kernel），cuBLAS 为它选择的算法就是 cuBLASLt 的启发式在这种姿态和 workspace 下回答的算法。在 `allow_tf32 = True` 时，torch 的 f32 `mm` 不是这些字节（TF32 tensor-core kernel；在随机操作数上断言）。

### 9.3 SM 分区：`sm_count_target`（tilefish 0.0.6，crate 0.0.4）

一个进程可以在设备 SM 的一个分区上运行它的 GEMM：一个 *green context*（基于 `cuDevSmResourceSplitByCount` 的 `cuGreenCtxCreate`）。cuBLASLt 在那里的行为是在 B300 的 30 SM MIG 切片（sm_103）上用 cuBLASLt 12.8.4 测得的，green context 为 8 和 16 个 SM，124 个问题的 f16 GEMM（31 种形状，`TN` 和 `NN`，带和不带 bias）：

* **在 green context 中，每个 SM 数量查询都报告设备的数量**（`cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)`、`cudaGetDeviceProperties`、torch 的 `multi_processor_count`：30）；只有 `cuCtxGetDevResource` 报告分区。因此条目的 `sm_count` 仍是设备的，分区上的 host 能通过第 9 节第 2 步的检查。
* **启发式的答案取决于进程第一次 `cublasLtCreate` 时的当前 context**，而不取决于 SM 数量：在第一个句柄是在 8（16）个 SM 的 green context 中创建的进程里，124 个答案列表中有 90（80）个与 primary context 的不同（大多是更小的 cluster），对之后在任何 context 中的每个句柄都是如此，而它的 wave 数仍是 30 个 SM 的。
* **`CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = T`** 使启发式为 `T` 个 SM 做选择（wave 数按 30/`T` 缩放，tile 和 split-K 数改变），它的答案在 primary context 中，以及在第一个句柄于至少 `T` 个 SM 的 green context 中创建的进程中，都是相同的（`T` = 8 和 16 时 124 个中 0 个不同）。设备数量（或更多）的目标与没有目标的回答相同（124 个中 0 个不同）。**1 到 7 的目标会杀死进程**（`cublasLtMatmulAlgoGetHeuristic` 内部的 `SIGFPE`）。
* **已 pin 的算法计算出什么，既不取决于 context，也不取决于目标**：864 次运行（10 个问题，32 个算法，9 种 context 和目标设置，3 个首个 context 不同的进程）以及 192 次手工构造的 split-K 算法运行给出了相同的字节，在每种设置中每个算法都只有一种启动配置；`cublasLtMatmulAlgoCheck` 的 workspace 不随目标改变。

因此，一个 pin 的字节就是在分区上的字节，分区改变的是启发式会选择哪个算法。想要为分区选择算法的声明要说明这一点：`@tilefish.library(..., sm_count_target=T)`，`T` 是至少为 8 的 int。于是：

| | |
|---|---|
| 条目 | 有可选字段 `sm_count_target`：声明没有目标时不存在（启发式为设备的数量做选择，与以前相同），否则是 8 到条目 `sm_count` 的 int（读取器拒绝任何其他值，包括 `null`）。它在规范声明中（`"sm_count_target": T`），因此也在 `declaration_digest` 中 |
| pin | 以 `SM_COUNT_TARGET = T` 询问启发式（设备必须至少有 `T` 个 SM）；导入的算法（`algos=`）按原样接受，其表以设备的 `sm_count` 为键 |
| 每次检查和运行 | 两个 host 都在 `cublasLtMatmulAlgoCheck` 和 `cublasLtMatmul` 的 matmul 描述符上设置 `SM_COUNT_TARGET = T`，因此一次调用就是检查 pin 时所用的那次调用 |
| replay | Python 拒绝目标与声明不同的条目（"pinned ... for 16 SMs (sm_count_target=16), but the declaration asks for the whole device"） |

**分区上的 host 如何使用它。** 用 `sm_count_target=P` 声明在 `P` 个 SM 的分区上运行的 GEMM，并在该设备上、在任意 context 中导出它们（pin 在 primary context 中和在至少 `P` 个 SM 的分区中是相同的）。Rust host 用 `Manifest::with_sm_partition(Some(P))` 说明它运行在哪个分区上（manifest 的一个克隆；plan 通过 `LaunchPlan::from_embedded` 获得它）：之后 `Manifest::gemm` 会拒绝为另一个数量选出的条目（`LibraryEntry::sm_count_chosen_for`：它的 `sm_count_target`，没有时为它的 `sm_count`），信息为 "was chosen for ..., but this host runs it on a partition of P SMs"。没有目标的 pin 在分区上运行时，字节与它在整个设备上的相同；只是选择是为整个设备做的。做 pin 的进程（`jit`、`export`）在 primary context 中（在 Python 中是 torch 的）创建它的第一个 cuBLASLt 句柄，否则它没有目标的 pin 就是创建第一个句柄时所在分区的 pin；有目标的则不是这样。`Gemm::sm_count_target` 和 `BuiltLibrary.spec.sm_count_target` 说明已加载 pin 的目标。`test_gpu_gemm_sm_target.py` 在它所运行的 GPU 上，在 green context 中检查第一个句柄的规则和目标。

### 9.4 已 pin 的 cuDNN 卷积：`impl: "cudnn.conv_fwd"`（tilefish 0.0.6，crate 0.0.4）

`impl` 为 `"cudnn.conv_fwd"` 的 `library` 条目是由 cuDNN 运行的前向卷积，cuDNN 运行的内容已被 pin。它的字段是它自己的（读取器首先按 `impl` 分派：Python 的 `tilefish._library.validate_entry` 把它交给 `tilefish._conv.validate_entry`，crate 把它读入一个 `ConvEntry`），第 2 和第 3 节的一切都适用于它：一个 `(stem, arch)` 对在 kernel 和每个 library 条目中唯一，条目整体计入 digest。

**操作。** `y = conv(x, w)`，即 torch 的 `conv1d` / `conv2d` / `conv3d` 计算的互相关，有 `G` 个分组，每个空间轴（一、二或三个）有一个步长、两侧相同的零填充和一个膨胀：

| 布局 | `x` | `w` | `y` |
|---|---|---|---|
| `NCHW` | `[N, C, *S]` | `[K, C/G, *R]` | `[N, K, *P]` |
| `NHWC` | `[N, *S, C]` | `[K, *R, C/G]` | `[N, *P, K]` |

每个轴上 `P = (S + 2 * padding - dilation * (R - 1) - 1) // stride + 1`。`x`、`w` 和 `y` 是同一种类型，`f16`、`bf16` 或 `f32`，按布局稠密，按条目的 `alignment` 对齐；累加器是 f32（`CUDNN_DATA_FLOAT`），结果舍入一次；`alpha = 1`，`beta = 0`。没有 bias：torch 的 cuDNN 卷积用单独的 `add_` 加上 bias（对已舍入结果的 f16 加法），tilefish kernel 也以同样方式实现；而融合进 backend operation graph 的 bias 会同时改变能运行它的 engine 和舍入。不提供非对称填充和转置卷积。

**如何调用 cuDNN**（`python/src/tilefish/_cudnn.py`，`describe`；crate 的 `ConvProblem::describe` 在跨语言向量的每个条目上给出相同的结果，逐值一致）。一个空间轴的问题被表述为两个空间轴的问题，其第一个空间轴大小为 1（填充 0，步长 1，膨胀 1）：这是 cuDNN 的做法，也是 torch 的（`conv1d` 把 `[N, C, L]` 看作 `[N, C, 1, L]`），并且是 legacy API 的 4-D 描述符所接受的。每个张量都按 cuDNN 的顺序给出，dims 为 `[N, C, (D,) H, W]`（`w` 为 `[K, C/G, ...]`），步长按布局（大小为 1 的轴的步长是其内侧各轴大小的乘积，与稠密张量相同，也与 torch 的 `fixSizeOneDimStride` 写入的相同）。卷积为 `CUDNN_CROSS_CORRELATION`，compute type 为 `CUDNN_DATA_FLOAT`，前后填充相同。然后，按 pin 的 API：

* **legacy**（`cudnnConvolutionForward`）：对 `x` 和 `y` 用布局的格式（`CUDNN_TENSOR_NCHW` 或 `_NHWC`）调用 `cudnnSetTensorNdDescriptorEx`，对 `w` 用该格式调用 `cudnnSetFilterNdDescriptor`，然后是 `cudnnSetConvolutionNdDescriptor`、`cudnnSetConvolutionGroupCount` 和 `cudnnSetConvolutionMathType`：数学模式 `"tensor_op"` 为 `CUDNN_TENSOR_OP_MATH`，`"fma"` 为 `CUDNN_FMA_MATH`，`"tf32"` 为 `CUDNN_DEFAULT_MATH`。每次启动都给调用一个大小为条目 `workspace_bytes` 的 workspace（算法需要的至多是这么多），这样每次启动的调用都相同：cuDNN 是否会因 workspace 大小不同而在一个算法内部选择另一个 kernel 尚不清楚，而 pin 不得依赖于此；
* **backend**（`cudnnBackendExecute`）：三个 backend 张量（`CUDNN_ATTR_TENSOR_UNIQUE_ID`：`x` 为 120，`w` 为 119，`y` 为 121，即 torch 的 `'x'`、`'w'`、`'y'`；`CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT` 为条目的 `alignment`）、一个卷积描述符、一个 `CONVOLUTION_FORWARD` 操作（`alpha` 和 `beta` 为 `CUDNN_TYPE_FLOAT`）及其 operation graph；一个具有 pin 的全局索引的 engine、由 pin 的 `(type, value)` 对构成的 knob 选择、由它们构成的 engine 配置以及一个执行计划，在加载条目时 finalize 一次；每次启动一个由三个指针和 workspace 组成的 variant pack。

**字段**（恰好这些；`sm_count_target` 和 `canary` 是可选的）：

| 字段 | 类型与规则 |
|---|---|
| `stem`, `op`, `arch`, `abi`, `buckets`, `dims`, `preconditions`, `declaration_digest` | 与 GEMM 条目相同；`abi.params` 是指针 `x`、`w`（`in`）和 `out`（`out`，每个 library 结果都用这个名称），类型为数据类型，形状按布局（表达式项在 `buckets` 下求值为问题的大小）；每个指针有一个值为条目 `alignment` 的 `pointer_alignment` |
| `impl` | `"cudnn.conv_fwd"` |
| `dtypes` | `{"x": T, "w": T, "y": T}`，`T` 为 `f16`、`bf16`、`f32` 之一 |
| `compute_type` | `"CUDNN_DATA_FLOAT"` |
| `math` | 16 位数据为 `"tensor_op"` 或 `"fma"`；f32 数据为 `"fma"`（TF32 关闭）或 `"tf32"` |
| `layout` | `"NCHW"` 或 `"NHWC"` |
| `n`, `c`, `k`, `groups` | 1 到 2^31 - 1 的 int；`groups` 整除 `c` 和 `k` |
| `input`, `filter`, `output`, `stride`, `padding`, `dilation` | 每个空间轴一个 int 的列表（1 到 3 个），`padding` 从 0 起，其他从 1 起；`output` 是卷积的输出 |
| `alignment` | 16、32、64、128 或 256：每个指针具有、backend 张量声明的字节对齐 |
| `workspace_bytes` | 1 到 2^40 的 int：选择 pin 时的上限、启动提供的大小，以及 legacy 调用得到的大小 |
| `choice` | pin 是如何选出的：`"v7"`（legacy pin）、`"instant"` 或 `"mode_b"`（backend pin），或 `"imported"`（两者皆可） |
| `pin` | `{"api": "legacy", "algo": a}`，`a` 是一个 `cudnnConvolutionFwdAlgo_t`（0 到 7），或 `{"api": "backend", "engine": e, "knobs": [[t, v], ...]}`，`e` 是一个 engine 的全局索引（从 0 起），`t` 是 knob 类型（0 到 42），`v` 是 -1 以外的 int64，各对按类型排序，每种类型一次。-1 是 cuDNN 的“未设置”（执行计划的 JSON 和启发式的 knob 选择用它表示保持 engine 默认值的 knob；所测 engine 的 knob 都不把它当作取值，每个 `CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE` 都是 0 或更大）：pin 省略这样的 knob，因此由它生成的执行计划让该 knob 保持默认值，与启发式的一样 |
| `cudnn_version` | int >= 1：选择该 pin 的库的 `cudnnGetVersion()`（9.10.2 为 `91002`）；它不能区分同一版本的两个构建（canary（哨兵值）可以） |
| `sm_count` | int >= 1：选择 pin 时所在设备的多处理器数量 |
| `sm_count_target` | *可选*：1 到 `sm_count` 的 int，即 pin 所针对的分区；每次加载时在 backend pin 的 engine 上设置（`CUDNN_ATTR_ENGINE_SM_COUNT_TARGET`）；在 legacy pin 上（legacy API 不接受目标）只有在它是导入的时才出现 |
| `canary` | *可选*（没有它的条目像以前一样加载）：`{"seed": s, "y_digest": d}`，`s` 是 0 到 2^32 - 1 的 int，`d` 是 32 个小写十六进制数字：pin 在导出时根据 `s` 的操作数计算出的 `y` 的 digest，两者都在设备上生成（见下文）。`export` 以种子 1 记录它。该字段的第一种形式 `{"seed", "y_sha256"}`（host 生成的操作数，host 的 sha256）被拒绝：请重新导出 manifest |

**pin 如何选出**（`@tilefish.library(..., impl="cudnn.conv_fwd", api=..., heuristic=...)`，`python/src/tilefish/_conv.py`）：

| `choice` | 询问什么 |
|---|---|
| `"v7"`（legacy；`api="legacy"` 的默认值） | `cudnnGetConvolutionForwardAlgorithm_v7` 的所有答案，取第一个状态为 `CUDNN_STATUS_SUCCESS`、确定性为 `CUDNN_DETERMINISTIC` 且 `cudnnGetConvolutionForwardWorkspaceSize` 至多为 `workspace_bytes` 的答案 |
| `"instant"`（backend；`api="backend"` 的默认值） | torch 2.9 在 `torch.backends.cudnn.deterministic = True` 和 `benchmark = False` 下询问的内容（`Conv_v8.cpp`，`get_configs_from_heuristics`）：先是 `CUDNN_HEUR_MODE_INSTANT` 下启发式的 engine 配置，然后是 `CUDNN_HEUR_MODE_FALLBACK` 下的，每个都像 torch 那样过滤（没有数值注记 `NONDETERMINISTIC` 或 `DOWN_CONVERT_INPUTS`；对于数学模式 `"fma"`，没有带 `TENSOR_CORE` 的，即 torch 的 `allow_tf32 = False`），取第一个执行计划能 finalize 且需要的 workspace 至多为 `workspace_bytes` 的。pin 是该配置的 engine 及其每个 knob 选择；由 pin 重新生成的执行计划必须有启发式自己的执行计划的 `CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION`（它写明 engine 和每个 knob，默认的 knob 写为 -1），否则什么都不 pin |
| `"mode_b"`（backend） | 同上，但先用 `CUDNN_HEUR_MODE_B`（torch 的 `TORCH_CUDNN_USE_HEURISTIC_MODE_B`） |
| `"imported"` | 由声明的 `algos=` 给出（`tilefish.CudnnTable(arch, sm_count, cudnn_version, {(n, c, k, input, filter, stride, padding, dilation, groups): pin})`，pin 对 legacy API 是一个 int，对 backend 是 `{"engine": e, "knobs": {t: v}}`）；不询问 cuDNN；没有表适用的设备在构建时被拒绝。表的 pin 中值为 -1 的 knob（未设置）会被丢弃，因此带它的表和不带它的表给出同一个 pin 和同一个 digest。声明使用的行在它的规范声明中 |

规范声明（格式 1）有名称、`impl`、`acc`、`workspace_bytes`、签名、`layout`、`stride`、`padding`、`dilation`、`groups`、`math`、`api`、`heuristic`、`alignment`、有 `sm_count_target` 时的 `sm_count_target`，以及各表的行；条目的 `declaration_digest` 再加上它的 `sm_count`。

**pin 的检查**，在记录它之前以及加载它时（两个 host；Python 的 `tilefish._cudnn.check_pin`，crate 的 `Manifest::conv`）：legacy 算法必须是 `cudnnGetConvolutionForwardAlgorithm_v7` 对该问题给出的、带 `CUDNN_STATUS_SUCCESS` 和 `CUDNN_DETERMINISTIC` 的答案，其 workspace 至多为 `workspace_bytes`；engine 必须是该问题的 engine 之一（它对 operation graph 能 finalize，带目标），其数值注记必须通过上面的过滤（`notes_problem`，两个 host 中措辞相同），其 knob 选择必须是该 engine 的 knob，每个在其 `CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE`..`MAXIMUM_VALUE` 范围内并符合其 `STRIDE`（`knobs_problem`），配置和执行计划必须能 finalize，执行计划的 workspace 必须至多为 `workspace_bytes`。

**replay**（以及 crate 的加载），按以下顺序：按 `(stem, arch)` 找到条目；设备的 SM 数量必须是 `sm_count`（"re-export the manifest
on this device"）；Python 将条目与声明比较（指出第一个不同之处），并将 pin 的 API 与声明的比较；进程的 `cudnnGetVersion()` 必须是 `cudnn_version`（pin 在另一个版本中对应其他 kernel；cuDNN 没有构建号，因此无法区分同一版本的两个构建）；**设置了 `NVIDIA_TF32_OVERRIDE`（任何值）的进程会被拒绝**（"unset it"）：它会在所声明的数学模式背后改变 cuDNN 选择和运行的内容（已测得），因此在这样的进程中也不会生成 pin；导入的 pin 必须是表中的；然后是 pin 的检查；然后，在条目有 canary 时，运行 canary。分区上的 Rust host（`Manifest::with_sm_partition(Some(P))`）只接受为 `P` 个 SM 选出的条目（它的 `sm_count_target`，没有时为它的 `sm_count`），与 GEMM 相同。

**canary。** 同一 cuDNN 版本的两个构建（例如 9.13.0.50 的 PyPI wheel 和系统包）回答相同的 `cudnnGetVersion()`，却映射不同的 engine 库。因此 `export` 在由种子生成的操作数上把每个 pin 再运行一次，并记录其 `y` 的 digest；replay（Python，在进程和设备中第一次 replay 该条目时：`tilefish._conv.check_canary`）和 crate 的加载（`Manifest::conv`，每个 context 一次）在使用 pin 之前再次运行它，digest 不同时拒绝该 pin（"the canary of the cuDNN convolution ... gives y with digest ... here, and the
manifest recorded ..."）。在 CUDA graph capture 内部、会运行 canary 的 Python replay 会被拒绝（"run it once before the capture"）。

**canary 的操作数和 digest 在设备上生成**，由一个 PTX 模块中的两个 kernel 完成，两个 host 中是同一段文本（Python 的 `tilefish._canary.CANARY_PTX`，crate 的 `src/canary.ptx`；有测试保证二者相同），由驱动为加载它的设备编译（面向 sm_70 及以后的 PTX：host 上不需要编译器）。`mix(v)` 是 splitmix64 对 `v + 0x9E3779B97F4A7C15`（模 2^64）的终结函数。`x` 和 `w` 按布局稠密，分别有 `n * c * prod(input)` 和 `k * (c / groups) * prod(filter)` 个数据类型的元素，小端序；操作数的第 `i` 个元素（`x` 的标签为 0，`w` 为 1）由 `h = mix(base + i)` 生成，`base = mix(2 * seed + tag)`：取其低 16 位（f16、bf16）或 32 位（f32），并把元素的最高有效字节 `b` 变换为：对 f16（`s eeeee mm`），`(b & 0x80) | ((7 + ((b >> 2) & 7)) << 2) | (b & 3)`（指数字段为 7 到 14：数值大小在 [2^-8, 1) 内），对 bf16 和 f32（`s eeeeeee`），`(b & 0x80) | (55 + (b & 7))`（指数 110 到 125：数值大小在 [2^-17,
2^-1) 内）：有限的规格化数，数量级多样，正负都有，它们的和按相加顺序舍入。`y`（`n * k * prod(output)` 个元素，用零填充到整数个 32 位字）和 workspace 开始时清零。digest 覆盖整个 `y`：它的字 `w_j`（小端序），`s1 = sum w_j * (mix(j ^ K1) | 1)` 和 `s2 = sum w_j * (mix(j ^ K2) | 1)` 模 2^64，其中 `K1 = 0x243F6A8885A308D3`，`K2 = 0x13198A2E03707344`，依次写出 `s1` 和 `s2`，各为 16 个小写十六进制数字。和不依赖于相加顺序，因此设备可以在任意 grid 中求和（先求每个 warp 的和，再做原子加）；改变一个字总会改变它（乘数是奇数），改变多个字时有大约 2^-128 的概率不被察觉：canary 仍然表明，这个 pin 在这个形状上、在这个进程中给出这些字节。只有和的 16 字节传回 host。`tilefish._canary.canary_bytes` / `digest_of` 和 crate 的 `canary_bytes` / `canary_digest` 是 host 上的相同计算（参考实现；`python_conv.rs` 保证 crate 的与 Python 的一致，`test_gpu_conv.py` 保证设备的与 host 的一致）。**canary 在它自己的 stream 上运行**（torch 的池，Rust 中为 `ctx.new_stream()`：非阻塞，不与任何其他 stream 排序）：清零的缓冲区、填充、卷积、digest 以及其和的回拷，回拷只等待那个 stream。

**canary 的代价。** 第一种形式在 host 上生成操作数（计数器模式的 SHA-256），并在 host 上对 `y` 计算哈希：对于一个有 743 个条目的 plan set（plan 集）（总共 38 GB 的 `x` 和 38 GB 的 `y`，最大的各 169 MB），Rust 加载在 B300 MIG 切片上用了 93 s，在 RTX 5090 上用了 150 s，其中 64 s / 101 s 用于生成操作数，15 s / 21 s 用于对 `y` 计算哈希，8 s / 19 s 用于把它拷回（按阶段测量，release 构建，cuDNN 9.13.0）。改为在设备上生成并计算 digest 后，同样的 743 个条目在 B300 切片上 1.4 s、在 RTX 5090 上 1.1 s 加载完成（第一次为 2.1 s，驱动在编译 PTX），而没有 canary 时为 0.08 s / 0.10 s；Python 在两者上给出同样的 743 个 canary 都用 1.1 s。它从不使用 NULL stream，CUDA green context 不会让 NULL stream 与其他 stream 排序（一个把拷贝放在 NULL stream 上的测试工具在 green context 内与卷积发生了竞争）。把一次导出合并进 manifest 时，会替换没有 canary 的条目；canary 不同的两个条目是两个二进制文件（用 `overwrite=True` 替换）。不记录所映射 engine 库的 `cudnn_engines_sha256`：每次加载都对它们计算哈希（9.10.2 约 570 MB），在 Python 中要花数秒，在 crate 的 debug 构建中还要多得多，而 canary 就是这个检查。

**stream。** 每次调用都在运行之前，在句柄的锁下，把句柄的 stream（`cudnnSetStream`）设为调用所给的 stream（一次启动的、一个 plan 的）；host 为卷积所访问的内存也在那个 stream 上：Python 的 workspace 在调用没有给出时，来自 torch 在该 stream 上的缓存分配器（不做 memset），crate 的 `Conv::alloc_workspace` 在给定的 stream 上清零，plan 的 arena 是 plan 自己的。`test_gpu_conv.py` 在一个非阻塞 stream 上以 `capture_error_mode="global"` capture 两种 API 的调用（workspace 由调用给出或由分配得到）：在 replay 之前输出不会被触碰，replay 得到的是 eager 调用的字节，因此调用的任何部分都不在 NULL stream 或其他 stream 上；并且在 NULL stream 保持忙碌时，manifest 的 canary 运行并通过，它们返回时 NULL stream 仍然忙碌。Rust 门禁在 `CU_STREAM_CAPTURE_MODE_GLOBAL` 下 capture 它的卷积，在 NULL stream 被一个 host 函数占住时再次运行每个 canary（`Conv::check_canary`，即加载所用的代码），它们返回时 NULL stream 仍被占住；并在 NULL stream 被占住时加载另一个 manifest 的卷积，它们共享 context 的句柄：它们在 NULL stream 仍被占住时返回（`the_canary_of_a_load_is_never_on_the_null_stream`）。真正会等待的是 cuDNN 的 `cudnnCreate`，而且它**等待设备上所有 stream 中排队的全部工作**，而不仅是 NULL stream 的：在 B300 MIG 切片上用 cuDNN 9.10.2 测得，不论一秒的自旋 kernel 排在 NULL stream、阻塞 stream、非阻塞 stream 还是 torch stream 上，`cudnnCreate` 都在其后花了 985 ms（在空闲设备上为 5 到 10 ms）；Rust 门禁用一个 host 函数占住一个非阻塞 stream，看到 `CudnnHandle::new` 只在它完成后才返回（`a_cudnn_handle_made_first_spares_the_loads_the_wait_of_cudnn_create`）。这种等待不会造成错误的排序，并且每个 context 只发生一次：**每个 CUDA context 一个 cuDNN 句柄**，由其中从任何 manifest 加载的每个卷积共享，不论是单独的还是在 plan 中，存活到最后一个卷积为止（Python 每个设备一个，在进程中保留）。因此，一个 context 中卷积的第一次加载会停在 host 已经排队的所有工作之后（在 crate 中，在该 context 的每个卷积都被释放之后也会再次如此）；在工作运行时加载卷积的 host 应在启动时、在排队任何工作之前创建句柄，并保留它：crate 中为 `tilefish::CudnnHandle::new(&ctx)`（它存活期间，context 中没有任何加载会创建句柄：同一个测试释放每个卷积，然后在再次占住 stream 时从一个尚未加载任何东西的 manifest 加载它们，它们在 stream 完成之前就返回了；句柄也被释放后，下一次加载会等待），Python 中为 `tilefish.runtime.warm_cudnn(device)`（或对带卷积的 plan 调用 `tilefish.runtime.prepare`）。会等待的不只是句柄：一个 context 中某个 cuDNN engine 的第一次运行（加载的 canary）也可能等待设备，因为 CUDA 惰性加载 engine 的 kernel（对门禁十个卷积中的一个测得，句柄已创建），因此这样的 host 也应在启动时加载它的卷积。在两个 host 中，对句柄的每次使用都在其锁下：启发式查询、加载或 pin 的描述符和执行计划、canary，以及调用（它在运行前设置 stream）；crate 还在锁下销毁卷积的执行计划。在 B300 MIG 切片上测得（cuDNN 9.10.2，release 构建），在一个 context 中加载 700 个小条目（`[1, T, 64]` f16 配 k7 滤波器，T 从 16 到 365，一半 legacy，一半 backend）：像以前那样每个条目一个句柄时，每次加载 7.1 到 7.6 ms（其中 6.5 到 6.8 ms 用于句柄），占用 6.0 GiB 设备内存（每个句柄约 8.6 MiB）；一个句柄时，每次加载连同其 canary 为 1.6 ms（不带 canary 为 0.8 ms），占用 12 MiB；仅 canary 本身，在已加载的卷积上再次运行，每个 0.7 到 0.8 ms。对同样 700 个条目的 Python replay 加载每个条目用 2.1 ms（准备 pin 并运行其 canary；canary 为 0.85 ms）。

**workspace 与 plan。** 启动由调用者给出 workspace，作为最后一个张量（一个 `u8` 向量，256 字节对齐，不与任何操作数重叠）：对 backend pin，至少为执行计划所需的大小；对 legacy pin，为条目的整个 `workspace_bytes`（调用得到的就是它）。launch plan 的 library 步骤像接受 GEMM 一样接受 cuDNN 卷积：arena 的那一个 workspace 缓冲区是其各 library 步骤中最大的 `workspace_bytes`，因此启动中不分配任何内存。**两种 API 都可以 capture**：`cudnnConvolutionForward` 和 `cudnnBackendExecute` 即使作为进程的第一次执行（没有预热调用），也能被 torch 的 `torch.cuda.graph` capture 到 CUDA graph 中，replay 得到的是 eager 调用的字节（已测得，B300 MIG 切片上的 cuDNN 9.10.2）；`test_gpu_conv.py` 中 plan 的 capture 和 Rust 门禁的 `launch_raw` capture 在 replay 时得到 Python 的字节。

**测量了什么**（`python/tests/test_gpu_conv.py`，在三个设备上：B300 的 30 SM MIG 切片 sm_103，配 torch 2.9.1+cu128 及其 cuDNN 9.10.2；同一切片配 torch 2.9.1+cu130 及其 cuDNN 9.13.0；RTX 5090 sm_120，配 torch 2.9.1+cu130 和 cuDNN 9.13.0；下面每一点在每个设备上都成立）：

* `"instant"` pin 与 torch 的字节一致（`torch.backends.cudnn.deterministic = True`，`benchmark = False`，TF32 关闭；NHWC 通过对 channels-last 张量调用 `conv2d`），覆盖 torch 运行 cuDNN 的 15 个问题（`torch._C._select_conv_backend` 为 `Cudnn`），每个三组操作数：1 到 1024 通道的一维 NHWC f16，步长 1、2 和 8，膨胀 1、3 和 9，带分组的一维 NCHW f16 和 bf16（包括 depthwise f16），带步长和分组的二维 NHWC 和 NCHW f16，以及 TF32 关闭的一维、二维和三维 f32。在 sm_103 上，cuDNN 9.10.2 的启发式对这些问题在 `CUDNN_HEUR_MODE_INSTANT`（以及 mode B）下什么都不回答，torch 和 tilefish 都采用 `CUDNN_HEUR_MODE_FALLBACK` 的列表：姿态仍与 torch 一致；
* 对于数学模式为 `"fma"` 的 16 位数据，没有 torch 的 cuDNN 字节可比（torch 只对 f32 过滤 tensor-core engine：在 cuDNN 9.13.0 中，f16 问题的第一个 engine 是 tensor-core engine，`"fma"` 拒绝它，字节也不同），对于 torch 不用 cuDNN 运行的卷积（bf16 depthwise 卷积作为 `CudaDepthwise2d`，f16 三维卷积作为 `SlowDilated3d`）也是如此：在这些情况下，pin 能运行、能重复其字节，并且与 f64 结果的差在一次 f32 累加和一次舍入之内；
* `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`（1）的 legacy pin 对 f16 和 f32 问题就是 torch 的 legacy 路径（`TORCH_CUDNN_V8_API_DISABLED=1`：在 `deterministic` 且不 benchmark 时，torch 以其数据的数学类型运行该算法，`onlyDefaultAlgorithm`）；`"v7"` 姿态对尝试过的每个问题都 pin 了该算法；
* pin 在同一进程中和在另一个进程中都重复其字节；`jit`、`replay`、plan、plan 的 capture 以及单独的调用给出相同的字节；
* 等于设备数量的 SM 数量目标等同于没有目标；对于低于它的目标（B300 切片上 30 个 SM 中的 15 个，RTX 5090 上 170 个中的 85 个），这些卷积在 cuDNN 9.10.2 或 9.13.0 中都没有 engine 能 finalize（启发式什么都不回答，普通 pin 的 engine 为 `CUDNN_STATUS_NOT_SUPPORTED`），因此带这种目标的声明在构建时被拒绝，从不在没有其目标的情况下运行。

### 9.5 带 cuBLASLt epilogue 的 GEMM：`impl: "cublaslt.matmul"`（tilefish 0.0.12，crate 0.0.10）

`impl` 为 `"cublaslt.matmul"` 的 `library` 条目是一个 GEMM，其 epilogue 由 cuBLASLt 在 f32 累加器上、在一次舍入到结果类型之前运行（RFC 0003，里程碑 E2）。第 9 节的两种 impl 保持逐字节不变；0.0.11 或 crate 0.0.9 的读取器拒绝这种 impl 的条目（它的 `impl` 不是它认识的）。

**操作。** 用 tilefish 的行主序术语表示，布局和问题同第 9 和第 9.2 节：

`out[M, N] = ρ(f(acc))`，`acc = a @ b^T`（`NN` 为 `a @ b`，`TT` 和 `NT` 为 `a^T`）以 f32 计算，`ρ` 是到结果类型的那一次舍入，`f` 由 `epilogue` 和 `beta` 决定，按以下顺序（库在 bias 之前加上 `c`：已测得，RFC 0003 X1）：

| `epilogue` | cuBLASLt 的 | `beta` 0 | `beta` 1 |
|---|---|---|---|
| `"none"` | `CUBLASLT_EPILOGUE_DEFAULT` (1) | `acc` | `acc + c` |
| `"bias"` | `CUBLASLT_EPILOGUE_BIAS` (4) | `acc + bias` | `(acc + c) + bias` |
| `"relu"` | `CUBLASLT_EPILOGUE_RELU` (2) | `relu(acc)` | `relu(acc + c)` |
| `"relu_bias"` | `CUBLASLT_EPILOGUE_RELU_BIAS` (6) | `relu(acc + bias)` | `relu((acc + c) + bias)` |

`a` 和 `b` 是 bf16 或 f16（两者同为一种类型 `T`）；`out`、`bias [N]` 和 `c [M, N]`（`out` 的布局）也是 `T`。这个公式是条目的 pin 据以验证的标准（见下文），而不是一个承诺：字节是库在该 pin 下给出的字节（契约 (iii)）。每个操作数都是连续且 16 字节对齐的。

**如何调用 cuBLASLt。** 与第 9 节相同，matmul 描述符的 `EPILOGUE` 为表中的值（对每个 epilogue 都设置，包括 `DEFAULT`），带 bias 的两种还有 `BIAS_DATA_TYPE`（结果的类型）和 `BIAS_POINTER`。`alpha = 1.0f`；`beta` 为 0 时 `beta = 0.0f`，`C` 为 `out` 指针（从不读取），为 1 时 `beta = 1.0f`，`C` 为 `c` 指针；`C` 的布局就是 `D` 的布局。每次检查和运行（Python 的 `tilefish._cublaslt.matmul(..., epilogue=, c_ptr=)`，crate 的带 epilogue 的 `cublaslt::Problem`）都设置同样的描述符，因此 `cublasLtMatmulAlgoCheck` 连同其 epilogue 一起判断算法。

**字段。** 第 9 节的字段（`ENTRY_KEYS`），加上 `beta`、`contract` 和 `verification`，在有不同之处时遵循以下规则：

| 字段 | 类型与规则 |
|---|---|
| `impl` | `"cublaslt.matmul"` |
| `epilogue` | `"none"`、`"bias"`、`"relu"` 或 `"relu_bias"` |
| `beta` | int `0` 或 `1`（不是 `1.0`，不是 `true`） |
| `dtypes` | `{"a": T, "b": T, ["bias": T,] ["c": T,] "out": T}`，`T` 为 `"bf16"` 或 `"f16"`：`bias` 恰好在 `"bias"` 和 `"relu_bias"` 时存在，`c` 恰好在 `beta` 1 时存在 |
| `abi.params` | 指针 `a`、`b`、[`bias` `[n]`]、[`c` `[m, n]`]、`out` `[m, n]`，按此顺序，类型为 `dtypes` 中的类型（`out` 为 `out`，其他为 `in`） |
| `contract` | `"measured-exact"` 或 `"binary-defined"`：pin 的验证发现了什么（见下文） |
| `verification` | 恰好由 `formula`、`accumulator`、`operand_sets`、`elements`、`differ`、`max_ulp` 组成的对象（见下文） |
| `canary` | *可选*：`{"seed": s, "out_digest": d}`，`s` 是 0 到 2^32 - 1 的 int，`d` 是 32 个小写十六进制数字（见下文） |
| `heuristic`, `allow_bf16_reduction`, `allow_f16_reduction`, `sm_count_target` | *可选*，同第 9.1 到 9.3 节（`allow_f32_reduction` 按操作数类型的规则被拒绝） |

`algo_source` 被拒绝：这种 impl 目前还不导入算法。`declaration_digest` 覆盖声明的 `epilogue`、`beta`、`canary`（导出是否记录 canary）和 `contract`（当它要求 `"measured-exact"` 时）。

**pin。** 它像 GEMM 的 pin 一样选出和审查（第 9 节，第 4 步），针对该 epilogue：用上面的描述符询问启发式，并且**除 `"none"` 之外的每个 epilogue 都没有 split-K**，除非声明选择了 opt-in（即带 bias 的 GEMM 的归约方案掩码 `0`；对 `"none"` 和 `beta` 1，接受 f32 部分和的 split-K，与 `cublaslt.gemm` 相同）。第 4 步的规则适用，措辞换成该 epilogue 的："split-K count 3 with the relu
epilogue may apply it to a sum already rounded to bf16 (as the bias epilogue was measured to, with f32 partial sums too), which rounds twice; a GEMM with an
epilogue has no split-K unless its declaration opts in with allow_bf16_reduction=True"（Python 的 `reduction_problem`，crate 的 `reduction_problem`；对 `"bias"` 是第 9 节的措辞）。

**验证**（`tilefish._matmul.verify`，每次 pin 时：`jit` 和 `export`）。在一个 epilogue 的算术可以写下来的地方，它的字节可以是契约 (ii)：在 pin 自己的累加器上以 f32 计算该公式，舍入一次。这是 pin 的性质，而不是 epilogue 的性质，因此每个 pin 都在两组带种子的操作数上运行，并与该参考逐位比较：

| 字段 | 记录什么 |
|---|---|
| `formula` | 表中对应条目 `epilogue` 和 `beta` 的公式（读取器检查它就是那个公式） |
| `accumulator` | `"f32-form"`：累加器是 pin 的配置以 f32 `D` 重新编码（`DEFAULT`，`beta` 0）后的 f32 输出，X1 发现以这种方式可以观测到它；`"plain-f32-pin"`：cuBLASLt 不接受以 f32 `D` 使用该配置，由同一姿态 pin 出的 f32 GEMM 提供累加器 |
| `operand_sets` | 2：`a ~ N(0, 1)` 和 `b ~ N(0, 1/k)`，按它们的布局，bias `~ N(0, 1/4)`，`c ~ N(0, 1)`，在 bias 和 `c` 的前 8 列中放入 `+0`、`-0`、`+inf`、`-inf`、NaN、正负两个符号的最大有限值以及一个次正规数（第二组中，`c` 的这些值位于它们旁边），并且 `a` 的某些行使每个 epilogue 得到特殊的累加器：每组中有一个 NaN、一个零和一个无穷大，第二组中还有两行会溢出 |
| `elements` | 比较的元素数：`operand_sets * m * n`（读取器检查它） |
| `differ` | 位与参考不同的元素数（设备上的 torch：f32 加法、`torch.relu`、它的类型转换，在 CUDA 上它们给出的位与 tilefish 自己对同样算子的下沉相同：规范 NaN） |
| `max_ulp` | 在两者都是有限值的元素上，二者之间以最后一位单位（ULP）计的最大距离（没有不同时为 0） |

`contract` 恰好在 `differ` 为 0 时为 `"measured-exact"`，读取器拒绝二者不一致的条目。声明了 `contract="measured-exact"` 的声明（以及把 plan 的算术改写成 epilogue 的每一次改写，它会这样声明）在构建时拒绝不满足的 pin，它的 replay 拒绝不满足的条目。

验证在 B300 MIG 切片（sm_103，30 个 SM）上的发现，覆盖 14 种形状（`M` 为 32、128、640、2048，乘以 `N x K` 为 1536 x 512、3072 x 1024、1024 x 3072，以及 32 x 4096 x 4096、100 x 1000 x 1000），bf16 和 f16，两种姿态以及七种情形 `none`+`c`、`bias`、`relu`、`relu_bias`、`bias`+`c`、`relu`+`c`、`relu_bias`+`c`（每个版本 392 个 pin；`python/tests/test_gpu_matmul.py`）：

* **cuBLASLt 12.8.4：每个 pin 都是 measured-exact**，包括特殊累加器和特殊值；没有 pin 切分 `k`。
* **cuBLASLt 13.1.0：392 个中 372 个是 measured-exact。** 其余 20 个是算法 21 在 32 x 1024 x 3072 上的 pin（`relu`、`relu_bias`、`bias`+`c`、`relu`+`c` 和 `relu_bias`+`c`，两种类型，两种姿态）：`bias`+`c` 在 26% 的元素上不同，最多相差 15176 ULP（bf16），与 X6 的发现一致；带 ReLU 的四种在 65 536 个元素中的 2048 到 2111 个上不同，全部是激活前为 NaN 的元素：这个算法的 ReLU 对 NaN 给出 `0`，而其他算法给出 NaN（`torch.relu`）。切分 `k` 的 `none`+`c` pin（在 `M = 32` 时用 f32 部分和做 3 和 4 的 split-K）是 measured-exact。

**canary。** cuBLASLt 不公开构建号，因此对于 `canary=True`（默认）的声明，`export` 记录 `{"seed": 1, "out_digest": d}`，加载时在使用 pin 之前运行它：Python 在进程和设备中第一次 replay 该条目时（`tilefish._matmul.check_canary`），crate 在 `Manifest::gemm` 中（每个 context 一次），二者都拒绝不同的 digest（"the canary of the cuBLASLt
matmul '<stem>' (<arch>, ...) gives out with digest ... here, and the manifest recorded ..."）。没有 canary 的条目像以前一样加载。它的操作数由第 9.4 节的 PTX 模块（`tilefish._canary`，crate 的 `src/canary.ptx`）在设备上生成，按第 9.4 节针对条目类型 `T` 的元素规则：`a`（标签 0，`m * k` 个元素）、`b`（标签 1，`n * k`）、`bias`（标签 2，`n`）和 `c`（标签 3，`m * n`），在条目有它们时生成，每个按其布局稠密；`out`（`m * n` 个元素，用零填充到整数个 32 位字）和大小为条目 `workspace_bytes` 的 workspace 开始时清零；pin 用条目的描述符在它自己的 stream 上运行（非阻塞，不与其他任何 stream 排序：与卷积的 canary 相同）；digest 是第 9.4 节的 digest，覆盖 `out` 的字。

**replay**，在第 9 节的步骤之后：条目的 `epilogue`、`beta`、`dtypes` 和操作数必须与声明的相同（Python 将条目与规格比较），带 `contract="measured-exact"` 的声明拒绝 `"binary-defined"` 条目，带 canary 的声明（默认）拒绝没有 canary 的条目，并运行 canary。一次启动接受 `a`、`b`、[`bias`]、[`c`]、`out` 和 workspace，按此顺序，像 GEMM 一样检查（crate 的 `Gemm::launch` 和 `launch_raw`，C ABI 的 `tilefish_gemm_launch`；`tilefish_gemm_epilogue` 说明 epilogue 以及它是否读取 `c`）。这种 impl 的 launch plan `library` 步骤列出相同的参数。

## 10. 严格读取：所有拒绝情形

schema 3 的读取器拒绝以下情形（Python：`TilefishManifestError`，标记为 *spec* 的除外，它们是每个二进制文件的编译规格的 `TilefishSpecError`；编译规格是 tilefish 写在构建旁边、host 不读取的文档）：

* 顶层：`schema` 的类型或值错误；`kernels` 或 `library` 缺失或不是列表；不是对象的条目；在两个列表中重复的 `(stem, arch)`。
* 每个 kernel 条目：不是 `triton`/`binary` 的 `kind`；缺少 `stem`、`arch`、`symbol`、`cubin`、`sha256`、`sha256_code`、`shared`、`launch`、`toolchain`；`triton` 上缺少 `options`，`binary` 上有 `options`；`binary` 上有 `num_warps`/`num_stages`；`binary` 上缺少 `binary`；`shared` 不是非负 int；`toolchain`、`launch`、`abi`、`options` 不是对象；`abi.scratch_slots` 存在但按种类不是 `2`/`0`；`source_sha256`/`source_digest` 不是 64 个小写十六进制数字；`source_digest` 和 `source_digest_kind` 不是同时存在或同时缺失；不是条目的 `arch`、`"fallback"` 或名称（至多 64 个字符的标识符，不区分大小写时不是 `fallback`，也不以 `sm_` 加数字开头）的 `implementation`（第 4.1 节）；格式错误或不在生成的 kernel 的 `triton` 条目上的 `cuda_math`（第 4.7 节）；格式错误或在 `binary` 条目上的 `instrument`（第 4.8 节），以及任何带有它的条目，除非读取器被要求读取插桩导出；`binary` 条目上的、不是整数的、不大于 `num_warps` 或大于 32 的、或 `block` 不是 `[32 * total_num_warps, 1, 1]` 的 `total_num_warps`（第 4.2 节）（Rust 读取器拒绝没有它而 `block` 不是 `[32 * num_warps, 1, 1]` 的 Triton 条目；Python 将该块与规格的比较）；不恰好是 `{programs_per_sm[, sm_count]}` 的 `persistent`（第 4.3 节）、不是 1 到 32 的 int 的 `programs_per_sm`、不是 1 到 2^20 的 int 的 `sm_count`，或多于一项的 grid；没有全局 scratch 或不是 2 的幂的 `launch.global_scratch_align`（第 4.6 节）；对 Triton 条目，为 1、不等于 `launch.num_ctas`，或在 `launch.num_ctas` 不为 1 时缺失的 `options.num_ctas`（第 4.5 节）；对 `binary` 条目，除了 `num_ctas` 个块（至多 8 个）的 cluster 之外不是零值的 `launch`，或带全局 scratch 的 `launch`；对 `binary`：`binary` 不恰好是 `{"source":"cubin"}` 和 `{"source":"ptx","ptxas_flags":[strings]}` 之一，`source_digest_kind` 不是 `"tilefish.binary.v1"`，没有 `source_digest`，有 `source_sha256`，按原样使用的 cubin 的 `source_digest` 不是它的 `sha256`；cubin 文件名不是一个普通文件名；cubin 的 `sha256`、`sha256_code` 或 `.text.<symbol>` 段错误；对 `triton`，共享内存段规则。
* 条目关于自身的陈述，两个读取器在读取时都会检查（Python：`Manifest._read`，与 `crates/tilefish/src/manifest.rs` 一样；调用者重新 pin 的文档的 digest 担保的是文档本身，而不是其含义）：缺少 `op`、`abi`、`grid`、`buckets`、`dims`、`preconditions` 或 `declaration_digest`；不是 `sm_` 加两位或三位数字的 `arch`，或者不是 cubin 的 ELF 头所写架构的 `arch`（`_elf.sm_of`，Rust 的 `Elf::sm`：一个被重新标为 `sm_90` 却装着 sm_103 cubin 的条目），或者对 `binary` 而言，头部没有写任何架构；不是 `op` 加上 `buckets` 中每一项的 `.<dim><value>`（按顺序）的 `stem`（`rowop.S32`）；不是 64 个小写十六进制数字的 `declaration_digest`（包括 `null`）；带有任何参数都没有的字段的参数，没有 `slot`、`index`、`name`、`kind`、`type`、`ptx_type`、`size` 的参数，或 `slot` 不在其槽位开始处（它的位置，计入它之前每个 `tensordesc` 额外的 `2 * rank` 个槽位）的参数，`slot` 不在最后一个参数的槽位之后的 scratch 槽位，以及各种类的字段（`read_params`）：`binary` 条目上的 `tensordesc`（第 5.5 节），`type`、`ptx_type` 和 `size` 不是 `tensordesc`、`b8[128]` 和 128 的，`dtype` 没有对应 map 的，带 `dim`、`count_of`、`stride_of`、`value` 或 `alias` 的，形状没有轴或多于 5 个轴的，在 `in` 以外的 access 上有 `strides` 的，或 `strides` 的最后一项不是固定的 `1`、或其他某项不是名称的，或者 `tensor_map` 不恰好是其各字段并符合第 5.5 节规则的（box 超过 256 或不能平铺块，内层 box 不是 16 字节的倍数或宽于 swizzle 跨度，块不是 2 的幂，元素步长不是 1，interleave 不是 `NONE`，整数类型用 NaN 填充，数据类型属于另一个 dtype）；任何其他种类上的 `tensor_map`；`type` 不是其 `dtype` 的指针类型的 `pointer`，`access` 不是 `in`/`inout`/`out` 的，带 `dim`、`count_of`、`stride_of` 或 `value` 的，或 `alias` 不是字符串或位于 `out` 以外的 access 上的，不是 8 字节 `u64` 的，或没有 `shape` 的（轴：≥ 1 的 int 或表达式），或者 `strides` 和 `offset` 不同时出现的、位于 `in` 以外的 access 上的、不是其形状每个轴一项的（一个名称，或只在最后一个轴上的固定 `1`），或 `offset` 不是名称的；带 `dtype`、`access`、`shape`、`strides`、`offset`、`value` 或 `alias` 的 `scalar`，没有 `dim` 的，同时有 `count_of` 和 `stride_of` 的，或宽度不是 `u32`/4 或 `u64`/8 的；Triton kernel 上的 `fixed` 参数，或带 `dim`、`count_of`、`stride_of`、`access`、`shape`、`strides`、`offset`、`dtype` 或 `alias` 的，或没有其类型的 `value` 的（在每一项中，JSON `null` 视同缺失）；Triton kernel 上的 `given` 参数，类型不是 `i32`/`i64` 或宽度不是 `u32`/4 或 `u64`/8 的，或带 `value`、`dim`、`count_of`、`stride_of`、`access`、`shape`、`strides`、`offset`、`dtype` 或 `alias` 的；`type` 不是其 `dtype` 的指针类型的 `null` 指针（第 5.6 节），不是 8 字节 `u64` 的，或带有任何其他字段的；语法之外的表达式，`dims` 中没有的名称，多于 32 个名称，小于 1 的静态 grid 项或除数（第 7.1 节）。
* ABI 整体（不依赖规格读取的 host 要自己检查这些；Python 通过将条目与编译规格比较来检查它们，`_specvalid` 在规格中检查它们）：`params` 的种类在 `pointer`/`tensordesc`/`scalar`/`fixed`/`given`/`null` 之外；指向 `tensordesc` 的 `stride_of`；`scalar` 的类型不是 `i32`/`i64`，或 `dtype` ≠ `type`；有 `strides` 而没有 `offset`，或反之；`strides` 长度错误或为空，含有 `1` 以外的整数，在非最后一个轴上有整数，有一个没有任何 `stride_of` 标量携带的字符串，或一个轴有两个标量；不是字符串的 `offset`；少于两个对象或有重复 `(arg, axis)` 对的 `stride_of` 列表；缺少或多出键、轴为负、轴超出范围、目标不是带步长的指针、`dim` 与目标的 `strides` 项不同，或与 `count_of` 同时出现的 `stride_of` 对象；类型不在 `i32`/`i64`/`f32` 中的 `fixed` 参数，不是浮点数的 `f32` 值，整数类型配非整数、bool 或超出范围的值；没有指向 `in` 指针、指向带步长指针或设备标量、指向 `dtype` 或声明的 `shape` 不同的指针，或指向另一个 `out` 指针已指向的输入的 `alias`（第 5.4 节）。
* 前置条件：在一个指针上出现多于一次的 `max_extent`（`index_bits=64` 的 kernel 没有它，因此它不是必需的），值不是 `2147483647` 的，或在没有 `strides` 的指针上的；有 `strides` 的指针上的 `max_elements`；`tensordesc` 上的 `max_elements` 或 `max_extent`；指针的 `pointer_alignment` 不是 16 或其元素大小（设备标量和每个 `tensordesc` 的，不是 16）；每个指针和每个 `tensordesc` 必须恰好有一个 `pointer_alignment`，`null` 指针则一个都没有。
* 每个 `library` 条目：第 9 节的字段列表、类型和字段间的相等关系，各按该节给出的规则；第 9.1 或 9.2 节的可选字段取了其唯一值（`"torch"`、`"imported"`、`true`）以外的任何值，包括 `null`；不是 8 到 `sm_count` 的 int 的 `sm_count_target`（第 9.3 节）；第 9.2 节没有列出的形式（两种类型的操作数，或 bf16、f16 和 f32 以外类型的操作数，类型既不是操作数的类型也不是 f32 的结果，类型与结果不同的 bias，`T` 和 `N` 以外的 `transa` 或 `transb`，不符合该形式的 `lda`、`ldb`、`a` 或 `b` 的形状或指针 `dtype`）；另一种操作数类型的 opt-in。
* 每个 `cublaslt.matmul` 条目（第 9.5 节）：GEMM 条目的规则，配上它自己的字段列表（`beta`、`contract`、`verification` 必需；`canary` 以及第 9.1 到 9.3 节中除 `algo_source` 外的选项为可选）；不属于那四种的 `epilogue`；不是 int 0 或 1 的 `beta`；不是 bf16 和 f16 中同一种类型的操作数或结果；`dtypes` 和 ABI 中的 `bias` 不恰好对应 `bias` 和 `relu_bias`，`c` 不恰好对应 `beta` 1；不属于那两种的 `contract`；不恰好是其六个字段的 `verification`，`formula` 不是该 epilogue 的，`accumulator` 不属于那两种，`operand_sets` 或 `elements` 小于 1，`differ` 或 `max_ulp` 小于 0，`differ` 大于 `elements`，`contract` 为 `measured-exact` 而 `differ` 大于 0 或反之，`max_ulp` 大于 0 而没有不同的元素；不恰好是 `{seed, out_digest}` 的 `canary`（种子超出 0..2^32 - 1，digest 不是 32 个小写十六进制数字），包括 `null`。
* 每个 `cudnn.conv_fwd` 条目（第 9.4 节）：缺失或未知的字段；不是 `f16`、`bf16`、`f32` 中同一种类型的 `dtypes`；其他 `compute_type`；不属于第 9.4 节或不符合数据类型的 `math`；`NCHW`/`NHWC` 以外的 `layout`；`n`、`c`、`k`、`groups` 不是从 1 起的 int，`groups` 不整除 `c` 和 `k`；空间列表长度不一致或不在 1 到 3 之间，值超出范围，或问题没有输出或元素过多；不是卷积输出的 `output`；不是 16 到 256 之一的 `alignment`，或值为其他对齐的 `pointer_alignment` 前置条件；超出范围的 `workspace_bytes`；不属于那四种的 `choice`，或其 API 与 pin 不符的 `choice`；不恰好是两种形式之一的 `pin`（算法大于 7，engine 为负，knob 类型大于 42，knob 未排序或重复）；不是正 int 的 `cudnn_version` 或 `sm_count`；超出 1..`sm_count` 的 `sm_count_target`，或在非导入的 legacy pin 上的 `sm_count_target`；不恰好是 `{seed, y_digest}` 的 `canary`（种子超出 0..2^32 - 1、不是 int 或为 `true`；digest 不是 32 个小写十六进制数字），包括 `null`，以及第一种形式 `{seed, y_sha256}`；值为 -1 的 knob（未设置：在 pin 中省略）；`x`、`w`、`out` 的形状不是该问题在该布局、这些 bucket 下的形状。
* CuTe DSL 条目（第 4.9 节）：不恰好是 `{source: "cute", cutlass_dsl, options}` 的 `binary`（版本和选项符合其形式），不恰好是 `{cutlass_dsl, nvvm, ptxas}` 的 `toolchain`（字符串，DSL 为二进制文件的 DSL），`"tilefish.cute.v1"` 以外的 `source_digest_kind` 或没有 `source_digest`，不属于条目架构的 `ptx_target`，命名了维度的 grid，`persistent` grid，`source_sha256`；不是 `pointer`、`tensormap` 或 `bytes` 的参数，或带 `strides`、`offset` 或 `alias` 的 `pointer`；`type`、`ptx_type`、`size` 不是 `tensormap`、`b8[128]`、128 的 `tensormap`，带其他种类字段的，没有 dtype、access 或 shape 的，或 `template` 不是第 0 到 7 字节和第 64 到 127 字节为零的 256 个小写十六进制数字的；`type` 不是 `bytes` 的 `bytes`，`size` 不在 1 到 32764 之间或 `ptx_type` 不是 `b8[<size>]` 的，带任何其他字段的，或 `value` 不是 `2 * size` 个小写十六进制数字的；任何其他条目上的 `tensormap` 或 `bytes` 参数，或 `source_digest_kind` `"tilefish.cute.v1"`。
* 仅限 *spec*，针对预构建二进制文件：不是 `{kind, symbol, shared, digests}` 的 `binary` 段（对 CuTe DSL kernel：`{kind: "cute", host,
  source_sha256, cutlass_dsl, options}`）；`attrs` 不为空；`launch_params` 不是按顺序排列的每个参数；`block` 不是乘积 ≤ 1024 的三个 1 到 1024 的 int；`constexpr` 参数。

## 11. 迁移 schema 2 manifest

一个由 Triton kernel 组成的 schema 2 manifest，恰好经过以下改动后就是 schema 3 中的同一个 manifest，其 cubin 不受影响（`python/tests/schema2_fixtures.py: migrate_doc`）：

1. `schema` 变为 `3`；
2. 每个 kernel 条目增加 `"kind": "triton"`，其 `abi` 增加 `"scratch_slots": 2`；
3. manifest 增加 `"library": []`。

它的 digest 会改变（标签 `v3`，以及 `library` 键），因此必须重新取得 pin。两个 host 都拒绝 schema 2（`Re-export it` / `Export it again`），这一检查在查看文件的任何其他内容之前进行，因此对 schema 2 的 manifest，给出的提示是重新导出，而不是缺少某个字段。Rust crate 已提交的 fixture（`crates/tilefish/tests/fixtures`，由 `python/tests/export_rust_fixtures.py` 在 B300 上构建）是 schema 3 manifest，`expected.json` 保存着 Python 为它们计算的 digest。

## 12. Rust host 做什么

本仓库的 Rust crate `tilefish` 完整读取 schema 3，并启动本文描述的一切，安全模型与其 schema 2 的启动相同（`DevicePtr<'a>` 生命周期，`from_cuda` 只对 cudarc 的 slice 和视图开放，在 `CudaStream` 上的安全 `launch` 带有所有者等待以及写者等待读者，拒绝对 slice 的 capture；`unsafe` 只出现在 `region.rs` 和 `ffi.rs` 及其 `ffi/cublaslt.rs` 中；使用 cudarc 的安全层及其 `cublaslt::result`，没有手写的驱动 FFI）。它比 Python 更严格或与之不同的地方在此列出。C++ host（`crates/tilefish-capi`：C ABI `tilefish.h` 和 `tilefish.hpp`，`docs/zh-CN/rfcs/0002-cpp-host.md`）通过这个 crate 读取 manifest，因此这里的一切照原样适用于它；它的张量是 region（`DevicePtr::from_raw`）。

**读取**（`crates/tilefish/src/manifest.rs`、`library.rs`）。

* 第 10 节的一切都被拒绝，第 3 节的 digest 被精确计算（`manifest_digest`，对任何其他 `schema` 使用 schema 2 形式）。`abi.scratch_slots` 是必需的（不依赖规格的 host 必须要求它）。`max_extent` 是可选的：*至多一次，并且只在带步长的指针上*（第 6 节说“每个带步长的指针一次”，而 `index_bits=64` 的 kernel 没有它，因此 Python 的读取器不要求它）。设备标量上的 `max_elements` 或 `max_extent` 被拒绝，Triton kernel 上的 `fixed` 被拒绝（第 5 节的表），不是 `in` 的指针上的 `strides` 被拒绝（第 5.1 节允许读取器这样做），步长项是一个维度表达式，必须恰好是携带它的标量的 `dim`。
* 两个读取器在读取 cubin 时都对照其条目的架构检查它：`Elf::sm()`（Python 的 `_elf.sm_of`：对 OS ABI `0x33`，SM 在 `e_flags` 的低字节，对 `0x41` 在第二个字节）。头部没有写可读架构或写的架构与其条目不同的预构建二进制文件被拒绝（第 8 节）；Triton cubin 在写了另一个架构时被拒绝，头部无法读取时交给驱动处理。
* 驱动加载的函数必须接受 ABI 的参数大小（`cuFuncGetParamInfo`，与 Python 在加载时的检查相同，对每个 kernel）。
* 一个条目至多可以命名 32 个维度（带步长的指针每个轴一个，偏移一个），其表达式在 128 位中折叠（第 7.1 节）；Python 的读取器两者都遵守。
* `implementation`（第 4.1 节）被读入 `Entry::implementation`（`Option<String>`），按第 10 节的规则检查，不作其他用途：crate 加载设备架构的条目，即导出为该架构选择的实现，并用该条目的 ABI 启动它（同一个 stem 在其他架构上的条目可以有其他 ABI：每个 `Plan` 是按条目的）。
* `null` 指针（第 5.6 节）被读入 `ParamKind::Null { dtype_name }`；条目的 plan 在其槽位写入 0（如同 `fixed` 参数的位），不为它接受张量（`Plan::tensor_names()` 不列出它），也不对它做检查。64 位维度标量（第 5.7 节）是 `ScalarType::I64` 的 `ParamKind::Scalar`，与 64 位步长一直以来的形式相同；稠密指针的元素对齐（第 6 节）就是条目的 plan 所检查的对齐；crate 对这两者都不需要其他东西。
* `total_num_warps`（第 4.2 节）被读入 `Entry::total_num_warps`（`Option<u32>`，与表示声明 warp 数的 `Entry::num_warps` 并列），按第 10 节的规则检查。每次启动都采用条目的 `block`（`Plan::block()`）：单次启动、launch plan 的一个步骤，以及 capture 的 graph。对于 warp 特化的 kernel，该块为 `32 * total_num_warps`，因此 crate 启动它不需要其他东西。
* `persistent`（第 4.3 节）被读入 `Entry::persistent`（`Option<Persistent>`：`programs_per_sm`、`sm_count`）。`Manifest::kernel` 在加载持久化 kernel 时确定它的 SM，即 `Manifest::with_sm_partition` 的分区或设备的多处理器数量（`Kernel::sm_count()`；kernel 也按 SM 数量缓存），拒绝为另一个数量声明的 kernel，并且每次启动都把 grid 切分为这些 SM 运行的 program（`Persistent::programs`；`Plan::prepare_for(tensors, given, device, sm_count)`，`prepare` 和 `prepare_with` 是其未切分的 grid，即 tile 数）。
* `launch.cluster_dims` 和 `launch.num_ctas`（第 4.5 节），配合 `LaunchAttributes::cluster()` 和 `Plan::cluster()`：`Prepared::grid` 是块的 grid（program 数乘以 cluster），`Prepared::cluster` 是 cluster；cluster kernel 的启动（`Kernel::launch`、`launch_raw`、plan 的一个步骤、一次 capture）是带 cluster 属性的 `cuLaunchKernelEx`（`cudarc::driver::sys::cuLaunchKernelEx`，cudarc 没有为它提供封装：安全启动像 GEMM 一样通过 cudarc 的 guard 对张量的 slice 排序）。`LaunchAttributes::global_scratch_align`（第 4.6 节）被读取（`Option<u32>`）；`blockers()` 列出 crate 不启动的东西。
* 全局 scratch（第 4.6 节）是一次启动的最后一个张量（`Plan::global_scratch()`，`GLOBAL_SCRATCH` 位于 `Plan::tensor_names()` 的最后），按启动的字节数（`Prepared::scratch_bytes`）、256 字节对齐、`u8`、kernel 所在设备上的一个被写入的稠密张量且不与任何其他张量重叠来检查；它的地址就是 `global_scratch` 槽位。代码生成器为这种 kernel 生成的结构体最后接受 `global_scratch: TensorMut<U8>`。plan 步骤给出它的 workspace。
* CuTe DSL 条目（第 4.9 节）被读入 `BinarySource::Cute { cutlass_dsl, options }` 和 `Entry::nvvm`；`tensormap` 是带 `template` 的 `ParamKind::Pointer`（`Param::template()`：它绑定、被检查并作为张量给出），`bytes` 是 `ParamKind::Bytes`。一次启动（单次启动、plan 的步骤、capture 的 graph）传入模板，并在其前 8 个字节写入张量地址（`Prepared::by_value`；地址为空、不是 16 字节对齐或不小于 2^57 时为 `Error::Launch`），并按所写的值传入字节。
* `cuda_math`（第 4.7 节）被读入 `Entry::cuda_math`（`Option<CudaMath>`），按第 10 节的规则检查。crate 无法得知 host 其他代码的 CUDA 版本，因此它在加载时不做任何检查；`Manifest::check_cuda_math(release)` 是 host 对它的声明，会拒绝其生成的 kernel 的函数没有在该版本上测量过的 manifest。

**启动 kernel**（`plan.rs`）。第 5.1 节的步骤，按 Python 的顺序：形状、步长和标量绑定维度（然后 `out` 张量绑定那些留下未绑定的维度，第 7 节）、`divisible_by`、精确的 bucket、每个标量的范围（`i32` 或 `i64`）、别名（视图按其 extent）、每个指针承诺的对齐、`max_elements`/`max_extent`、grid。拒绝的种类与 Python 的相同（`stride`、`max_extent`、`contiguous` 是新的 `PreconditionKind`）。视图是 `TensorRef::from_cuda_strided(&slice, &shape, &strides, offset)`（还有 `TensorMut`，以及基于 region 的 `new_strided`）：元素步长，在这里是无符号的，因此无法构造负步长（第 5.1 节第 2 步拒绝它）；视图从其第一个元素算起的 extent 必须放得进它的内存（在构造时拒绝）；偏移以元素为单位，可以从任意元素类型的 slice 的任意字节开始，kernel 得到的是第一个元素的地址（对于 slice，交给 cudarc 的是从那里开始的字节的视图，因此它对 slice 的跟踪不变）。不是视图的张量是带稠密步长的视图。给不带步长的参数的视图必须是连续的（torch 的规则：不看大小为 1 的轴的步长）。带步长指针的偏移维度不被绑定（第 5.1 节第 7 步）：所做的检查是地址检查。设备标量是形状为 `[]` 或 `[1]` 的张量。`fixed` 值按其类型写在其槽位（`f32` 是 JSON double 舍入到最近的 `f32`）。`given` 参数的值来自调用：`Kernel::launch_with(stream, tensors, &[(name, value)])`（`check_with`、`launch_raw_with`；`Plan::prepare_with`）：条目没有的名称、重复的名称和缺失的名称首先被拒绝（`Error::Launch`），放不进 `i32` 的值在检查各标量范围的地方被拒绝（种类 `shape`）；`launch` 不给值，因此对有这种参数的条目它会被拒绝。plan 步骤的值就是它的 `given`。

`tensordesc`（第 5.5 节）被读入带 `tensor_map: Some(TensorMap)` 的 `ParamKind::Pointer`（它的块、box、swizzle、L2 promotion、填充和数据类型；`Param::slots()` 是它的 `1 + 2 * rank` 个槽位大小，`Param::tensor_map()`），因此它像指针一样绑定和检查；一次启动接受它的张量（`TensorRef`/`TensorMut`），在前置条件之后，`Plan::prepare` 拒绝驱动会拒绝的情形（种类 `PreconditionKind::TensorMap`，即 Python 的 `tensor_map`），并计算 `cuTensorMapEncodeTiled` 的参数（`Prepared::tensor_maps()`、`TensorMapEncode`：与 Python 的 `encode_arguments` 逐数一致），形状和步长位于它们的槽位（`Prepared::params()` 在 map 自己的槽位为 0）。`Kernel::check`（因此每次启动，以及每次执行时 plan 的每个步骤）通过 `cudarc::driver::sys::cuTensorMapEncodeTiled` 编码 map；`launch` 用 cudarc 的 launch builder 按值传入这 128 字节，并在启动期间通过 cudarc 的跟踪 guard 对张量的 slice 排序，`launch_raw` 把槽位指向它们，capture 把它们保存在 kernel 节点中。

**启动 library 条目**（`gemm.rs`、`ffi/cublaslt.rs`）。`Manifest::gemm(&ctx, name, buckets, arch)` 为 `ctx` 的设备加载一次 `Gemm`，`Gemm::launch(&stream, &mut [a, b, [bias], [c], out, workspace])` 运行它：

* 加载按 Python 的顺序拒绝：无法打开的 cuBLASLt（cudarc 在第一次使用时打开它，缺失时会 *panic*，因此 `ffi/cublaslt.rs` 先询问），`cublasLtGetVersion()` 不是 `cublaslt_version` 的 cuBLASLt（错误信息写出两个版本），未通过第 9 节审查的算法（Python 的 `_validate_algo` 的检查：用 `cublasLtMatmulAlgoInit` 和配置属性重建且逐字节一致，tile 和 stages 的 id 以及 swizzling 和自定义选项在其 id 的能力范围内，f32 累加器（对 f32 操作数：`CUBLAS_COMPUTE_32F` 编码为它自身，并在标志说明输入时为 f32 输入，第 9.2 节），对齐至多 16 字节，split-K 配合 `NONE` 或 `COMPUTE_TYPE` 归约方案，对于有 `allow_bf16_reduction` 的条目还可以是 `INPLACE` 或 `OUTPUT_TYPE`），`cublasLtMatmulAlgoCheck` 不认可该问题的算法，以及需要超过 `workspace_bytes` 的算法。第 9.1 节的可选字段被读入 `LibraryEntry::heuristic`（`Heuristic`）、`algo_source`（`AlgoSource`）和 `allow_bf16_reduction`；第 9.2 节的形式被读入 `LibraryEntry::dtype`、`out_dtype`（`DType`）、`layout`（`GemmLayout::Tn`、`Nn`、`Tt` 或 `Nt`）、`allow_f16_reduction` 和 `allow_f32_reduction`，算法以这些类型和该布局审查和运行；`sm_count_target`（第 9.3 节）被读入 `LibraryEntry::sm_count_target`，设置在检查和每次启动的描述符上，并对照 host 给出的分区（`Manifest::with_sm_partition`）检查。检查在加载时运行一次，而不是每次启动时（条目的问题是固定的）；一次调用的描述符在调用中创建和销毁。
* 进程必须映射选择该 pin 的同一个 cuBLASLt 构建（cudarc 尝试 `libcublasLt.so`，然后是 `.so.<major>`、`.so.12`……；在 torch 自带 cuBLASLt 的机器上，把那个目录放进 `LD_LIBRARY_PATH`）：不同的构建会被版本检查拒绝，此外没有别的检查。
* **workspace 属于调用者**，是一次启动的最后一个张量：一个 `TensorMut<U8>`（cudarc slice 或 region），至少 `Gemm::workspace_required()` 字节（算法所需的大小，至多为 `workspace_bytes`；`workspace_bytes` 是 Python 提供的大小，也是 `Gemm::alloc_workspace` 分配的大小），256 字节对齐，不与任何操作数重叠。启动中不分配任何内存，因此启动可以被 capture。共享一个 workspace 的两次启动由 cudarc 的写跟踪和 crate 的写者等待读者规则排序（在同一个 stream 上，按 stream 顺序）；在多次 replay 之间共享 workspace 的 graph 由 host 负责排序。
* 操作数经过条目的 plan（形状绑定到条目的维度、精确的 bucket、dtype、连续性、16 字节对齐、设备、context、别名），检查 stream 和 context，并像 kernel 一样对 slice 排序（cudarc 通过 `device_ptr` 的 guard 进行等待和记录，所有者等待工作结束，stream 正在 capture 时拒绝）。带 region 的 `unsafe Gemm::launch_raw` 可以 capture 一个 GEMM。
* **带 epilogue 的 GEMM**（`impl: "cublaslt.matmul"`，第 9.5 节）被读入同一个 `LibraryEntry`（`implementation` 为 `LibraryImpl::Matmul`），带有 `epilogue`（`GemmEpilogue::None`、`Bias`、`Relu` 或 `ReluBias`；`GemmEpilogue::formula` 就是 Python 的 `formula`）、`beta`（0 或 1；`has_bias()`、`has_c()`）、`contract`（`MatmulContract`）、`verification`（`MatmulVerification`）和 `canary`（`MatmulCanary`），全部按第 10 节的规则、以 Python 的措辞检查（`fields_of`、`matmul_fields`；`gemm` 和 `gemm_bias` 上出现 `cublaslt.matmul` 的字段，以及它上面出现 `algo_source`，都按名称被拒绝）；`gemm` 的条目有 `GemmEpilogue::None`，`gemm_bias` 的条目有 `GemmEpilogue::Bias`，`beta` 为 0，没有契约。每次检查和运行都在 matmul 描述符上设置条目的 epilogue（包括 `DEFAULT`，与 Python 的 `_Problem` 相同），对带 bias 的 epilogue 设置 `BIAS_DATA_TYPE` 和 `BIAS_POINTER`，条目有 `c` 时设置 `beta = 1.0` 并以调用的 `c` 为 `C`（`D` 的布局），否则设置 `beta = 0.0` 并以 `out` 指针为 `C`；审查要求除 `none` 之外的每个 epilogue 在没有 opt-in 时都不使用 split-K，措辞与带 `epilogue=` 的 Python `reduction_problem` 相同（`reduction_problem` 接受 `GemmEpilogue`：即合二为一的 Python 的 `bias` 和 `epilogue`）。一次调用是 `a`、`b`、[`bias`]、[`c`]、`out` 和 workspace，按此顺序，像 GEMM 一样检查（`c` 是结果类型的 `[m, n]`，不与任何被写入的东西重叠）；launch plan 的 library 步骤列出相同的参数（`Call::c`）。`Gemm::epilogue`、`beta`、`has_c`、`contract`、`canary` 和 `implementation` 说明加载了什么。**canary**：对于有 canary 的条目，`Manifest::gemm` 在审查和检查之后、使用 pin 之前，每个 context 运行它一次，与 Python 的 `canary_digest` 的计算完全相同：在它自己的 stream 上（`ctx.new_stream()`，从不在 NULL stream 上），操作数由 `canary.ptx` 的填充 kernel 生成（`a` 标签 0，`b` 标签 1，`bias` 标签 2，`c` 标签 3，为条目的类型，按其布局稠密：`LibraryEntry::canary_elements`），`out` 用零填充到整数个字，大小为条目 `workspace_bytes` 的 workspace 清零，用条目的描述符运行 GEMM，对 `out` 运行 digest kernel，只把它的 16 字节拷回；不同的 digest 以 Python 的 `check_canary` 的措辞被拒绝（"the canary of the cuBLASLt matmul '<stem>' (<arch>, algorithm id ..., epilogue ...) gives out with digest ... here, and the
  manifest recorded ... (seed s): ..."）。canary 的 kernel 在每个 context 中加载一次，只要有已加载的 GEMM 或卷积持有它们（`src/canary.rs`，与卷积共享）；`Gemm::check_canary` 再次运行它。契约只被读取和报告，不被强制执行：只想要 measured-exact pin 的 host 检查 `Gemm::contract`（Python 的 replay 只对要求 `contract="measured-exact"` 的声明拒绝 binary-defined 条目，而没有该声明的 host 并不拥有它）。
* **C ABI**（ABI 1.1）：`tilefish_gemm_launch` 按同样的顺序接受张量（`c` 在 `out` 之前），`tilefish_gemm_epilogue(gemm,
  &epilogue, &has_c)` 以 `TILEFISH_EPILOGUE_NONE`、`_BIAS`、`_RELU` 或 `_RELU_BIAS`（0 到 3：tilefish 的值，不是 cuBLASLt 的 `cublasLtEpilogue_t`）报告 epilogue，并报告一次启动是否接受 `c`；`tilefish.hpp` 中为 `tilefish::Gemm::epilogue()`、`has_c()` 和 `has_bias()`。被 canary 拒绝的加载是 `TILEFISH_ERROR_MANIFEST`。

**启动卷积**（`conv.rs`、`ffi/cudnn.rs`）。`Manifest::conv(&ctx, name, buckets, arch)` 每个 context 加载一次 `Conv`，`Conv::launch(&stream, &mut [x, w, out, workspace])` 运行它（带 region 的 `launch_raw` 可以 capture 它）。加载按以下顺序拒绝：pin 不是为之选出的分区，SM 数量不同的设备，设置了 `NVIDIA_TF32_OVERRIDE`，无法打开的 libcudnn（cudarc 的 `cudnn` feature，动态加载：只在加载卷积时打开，因此没有卷积的 manifest 从不打开它），不同的 `cudnnGetVersion()`，未通过第 9.4 节检查的 pin，以及 digest 不同的 canary（在设备上生成并计算 digest，在它自己的 stream 上；`canary_bytes`、`canary_operands` 和 `canary_digest` 是 host 上的参考实现；`Conv::check_canary` 再次运行它）；它用 context 的 cuDNN 句柄创建一次 legacy 描述符或 finalize 一次执行计划，context 中加载的每个卷积共享该句柄（`Conv::shares_cudnn_handle`）。一次启动获取句柄的锁（句柄的 stream 是它的状态），设置 stream 并运行；张量像 GEMM 的一样经过条目的 plan，workspace 经过同样的检查，至少 `Conv::workspace_required()` 字节（执行计划所需的大小，或 legacy pin 的整个 `workspace_bytes`）。`ConvEntry` 保存读取的结果（`problem: ConvProblem`、`pin: ConvPin`、`choice`、`cudnn_version`、`sm_count`、`sm_count_target`、`canary: Option<ConvCanary>`）；`Manifest::library()` 是 GEMM，`Manifest::convolutions()` 是卷积。launch plan 的 library 步骤加载其条目对应的那一种。

**代码生成器**（`tilefish.hosts.rust`）。它为每个 kernel（Triton 或预构建：`launch` 以 `TensorRef` 或 `TensorMut` 接受每个指针；步长标量、`fixed` 参数和 `null` 指针不是参数）以及每个 library 条目（`launch` 接受 `a`、`b`、`bias`、`out`、`workspace`；`gemm()`、`M`、`N`、`K`、`BIAS`、`WORKSPACE_BYTES`、`PINNED_WITH`；`cublaslt.matmul` 在其 `beta` 为 1 时还在 `out` 之前接受 `c`，有 `EPILOGUE` 和 `BETA`，其文档用启动的名称写出公式、epilogue、每个架构的 pin 所验证的契约及其 canary；`gemm` 和 `gemm_bias` 的结构体保持原样）写一个结构体，并嵌入 manifest 的 digest。`tilefish.hosts.cpp` 为 C++ 写出同样的类（`EPILOGUE` 和 `BETA` 为 `static constexpr` 成员）。它拒绝读取器拒绝的内容。同一个 stem 在多个架构上的条目，除了设备之外必须在一切方面一致，但按架构声明实现的算子的条目（每个都有 `implementation`）可以在第 4.1 节允许的方面不同，包括它们的 ABI：它们必须共享张量（名称、顺序、access、dtype、形状、是否为视图）、每次调用的固定参数以及其张量和 bucket 的维度约束，这就是 `launch` 的类型来源；结构体的文档随后写出每个架构的实现、grid、block 和 constexpr，并在它们的 ABI 不同或其中之一是预构建二进制文件时，写出每一个的种类、描述符、标量、固定参数和视图；`ops=` 对照每个条目的架构所选的实现检查该条目。

**launch plan**（crate 尚未读取）。`docs/zh-CN/plan-schema-1.md`：`plan.json` pin 一个 manifest，并在一个 arena 上安排其条目的启动顺序，因此 host 可以据此运行一次完整的前向，并且可以在 CUDA graph 中 capture（`test_gpu_plan.py`；golden fixture `python/tests/golden/plans/` 带有要复现的 digest）。

## 13. 测试位置

| 内容 | 测试（`python/tests/`） |
|---|---|
| 顶层、digest、三处添加、拒绝 schema 2 | `test_manifest.py`、`test_manifest_schema3.py`、`test_rust_host.py` |
| `kind`、`abi.scratch_slots`、预构建二进制文件条目 | `test_binary_externs.py`、`test_manifest_schema3.py` |
| `cuda_math`（第 4.7 节） | `test_cudamath.py`、`crates/tilefish/tests/cuda_math.rs`；GPU 上 `test_gpu_cuda_math.py` |
| cubin 不记录 Triton kernel 的动态共享内存（过小的 `shared` 无法检测，第 4 节） | GPU 上 `test_gpu_shared_size.py` |
| 带步长的指针、`strides`/`offset`/`stride_of`、`max_extent`、对齐 | `test_strided_specs.py`、`test_views.py`；GPU 上 `test_gpu_strided.py` |
| 多个视图共享的步长（`stride_of` 为列表） | `test_shared_strides.py`、`test_rust_host_schema3.py`；GPU 上 `test_gpu_shared_strides.py` |
| 只由 `out` 张量确定的维度（第 7 节） | `test_output_dims.py`；GPU 上 `test_gpu_output_dims.py` |
| 可以写在输入之上的 `out` 指针（`alias`，第 5.4 节） | `test_alias.py`；GPU 上 `test_gpu_alias.py` |
| 设备标量 | `test_devscalar.py`；GPU 上 `test_gpu_devscalar.py` |
| 预构建 PTX 和 cubin，与 Triton 路径的逐字节一致 | `test_binary_externs.py`；GPU 上 `test_gpu_binary.py`、`ptx/*.ptx` |
| `library` 段 | `test_library_ops.py`、`test_cublaslt.py`；GPU 上 `test_gpu_gemm.py`、`test_gpu_cublaslt.py` |
| 它的可选字段（第 9.1 节）：姿态、导入的算法、bf16 split-K 部分和 | `test_library_options.py`、`test_cublaslt.py`；GPU 上 `test_gpu_cublaslt.py`（torch 的字节、opt-in）、`test_gpu_gemm.py`（jit 和 replay） |
| f16 操作数、f32 结果、布局 NN、`allow_f16_reduction`（第 9.2 节） | `test_library_f16.py`；GPU 上 `test_gpu_gemm_f16.py`（两种布局下 torch 的字节、f64 参考、jit、replay、plan、导入的表、opt-in） |
| 布局 TT 和 NT（第 9.2 节） | `test_library_layouts.py`；GPU 上 `test_gpu_gemm_layouts.py`（bf16、f16 和 f32 下 torch 的字节、f64 参考、jit、replay、一个 plan 及其 capture、一张导入的表）以及 Rust 门禁 |
| cuDNN 卷积（第 9.4 节） | `test_library_conv.py`（声明、条目、读取器、replay 的检查、plan、代码生成器、绑定的规则）；GPU 上 `test_gpu_conv.py`（torch 的字节、torch 的 legacy 路径、跨进程的可重复性、jit、replay、一个 plan 及其 capture、导入的 pin、各种拒绝、SM 数量目标、canary 及其 stream、global 模式的 capture）以及 Rust 门禁 |
| f32 操作数、TF32 关闭、`allow_f32_reduction`（第 9.2 节） | `test_library_f32.py`（声明、审查规则、旧条目的 digest）；GPU 上 `test_gpu_gemm_f32.py`（两种布局和两种 workspace 配置下 torch 的字节、TF32 和布局的陷阱、f64 参考、拒绝 TF32 和模拟、jit、replay、plan、导入的表、opt-in） |
| `sm_count_target`（第 9.3 节） | `test_library_sm_target.py`；GPU 上 `test_gpu_gemm_sm_target.py`（green context：第一个句柄的规则和目标；字节不依赖于目标） |
| 带 cuBLASLt epilogue 的 GEMM（第 9.5 节） | `test_matmul.py`；代码生成器：`test_rust_host_schema3.py`（`golden/rust_matmul.rs`，对启动器的 `cargo check`）、`test_hosts_cpp.py`；GPU 上 `test_gpu_matmul.py`，以及 Rust 和 C++ 门禁（`rust_gate.py`：`MATMUL_FORMS`，`beta` 为 0 和 1 的每个 epilogue，bf16 和 f16，四种布局，`M` 的三个 bucket，有和没有 canary；plan `epilogue_chain`） |
| 代码生成器 | `test_rust_host.py`、`test_rust_host_schema3.py`（golden 在 `python/tests/golden/` 中） |
| 手写 kernel 的 `options.enable_fp_fusion`（`@tilefish.extern(enable_fp_fusion=...)`） | `test_lower_options.py`；GPU 上 `test_gpu_extern_fusion.py` |
| `implementation`：按架构声明实现的算子（第 4.1 节） | `test_per_arch.py`；GPU 上 `test_gpu_per_arch.py`（以及 Rust 门禁） |
| 命名实现、作为实现的预构建二进制文件、每个实现一个 ABI（第 4.1 节） | `test_per_arch_impls.py`（声明、0.0.6 的 digest、规格、构建的拒绝、导出和 replay、读取器、调用、plan、代码生成器及对其输出的 `cargo check`）；GPU 上 `test_gpu_per_arch_impls.py`（同一个拷贝的 TMA 描述符形式、指针形式和 PTX 形式：jit、导出、replay、plan；sm_120 的用例在 sm_120 GPU 上运行） |
| `tensordesc`：TMA 描述符（第 5.5 节）：声明、规格、槽位、读取器、启动检查、编码、假后端上的 plan、代码生成器 | `test_tensordesc.py`；GPU 上 `test_gpu_tma.py`（Triton 和 Gluon kernel 对照 Triton 自己的启动、plan 和 capture）以及 Rust 门禁 |
| `total_num_warps`：warp 特化的 kernel（第 4.2 节）：构建的检查、缓存、读取器、启动的块大小、代码生成器 | `test_warp_specialization.py`；GPU 上 `test_gpu_warp_specialization.py`（Gluon 和 Triton warp 特化 kernel 对照 Triton 自己的启动、CUDA graph、plan 和 capture）以及 Rust 门禁 |
| 可复现的导出（第 3.1 节）：源文件的名称、按内容寻址的生成源码（原子写入、检查、回退、多个进程同时进行）、从两个检出目录的两次导出 | `test_reproducible_builds.py`；GPU 上 `test_gpu_repro.py`（一个包含下沉生成的 kernel、生成的算子、一个手写 extern 和一个 PTX 的 graph，在两个进程中从两个目录导出：相同的文件和 digest）、`test_gpu_build.py`、`test_gpu_plan.py` |
| `null` 指针（第 5.6 节）：声明、规格、读取器、启动、plan、代码生成器 | `test_null_pointers.py`；GPU 上 `test_gpu_types.py`（一个 Triton kernel 和一个手写 PTX，jit、replay、一个 plan）以及 Rust 门禁 |
| CuTe DSL kernel（第 4.9 节）：声明、规格、来自 capture 的启动的参数、PTX 的读取、条目、读取器的每种拒绝、replay 的 PTX 检查、代码生成器、启动的字节 | `test_cute_externs.py`、`crates/tilefish/tests/cute_externs.rs`；GPU 上 `test_gpu_cute.py`（CUTLASS 示例中的 kernel 对照 DSL 自己的启动、torch、cuBLASLt；在两个进程中导出；replay、plan、capture）以及 Rust 和 C++ 门禁 |
| 64 位维度标量（第 5.7 节） | `test_dim_scalars.py`；GPU 上 `test_gpu_types.py`（一个 `2^31 + 1000` 个元素的张量，jit 和 replay）以及 Rust 门禁 |
| 按其元素对齐的稠密指针，下沉生成的 kernel 的特化（第 6 节） | `test_unaligned_views.py`；GPU 上 `test_gpu_types.py`（plan 中和单独运行的平面行，jit、replay、reference）以及 Rust 门禁 |
| 启动形状（第 4.3 到 4.6 节，以及相乘的 grid，7.1）：声明、规格及其 digest、构建的检查、读取器、启动得到什么、plan 的 workspace 作为 scratch | `test_launch_shapes.py`；GPU 上 `test_gpu_launch_shapes.py`（`launch_kernels.py`：jit、replay、plan、`tilefish.capture` 和 `torch.cuda.graph` 中 torch 的字节；1 到 30 个 SM 上的持久化 kernel、一个 Triton cluster 和一个 PTX cluster、设备端描述符）以及 Rust 门禁 |

Rust crate（`crates/tilefish/tests/`）：

| 内容 | 测试 |
|---|---|
| schema 3 的严格读取、更旧的 schema | `strict_reading.rs`、`library_section.rs`（也包括第 9.1 节的可选字段和 9.2 节的形式）、`schema3.rs`（合成的 manifest，`common/synthetic.rs`；第 4.1 节的 `implementation`；`null` 指针、64 位维度标量以及按其元素对齐的稠密指针） |
| 第 9.2 节的形式：Python 对每种形式的条目以及对每个只改动一个字段的条目的判定 | `python_library.rs`（`data/python_library.json`，由 `export_rust_vectors.py` 生成） |
| 第 9.5 节带 epilogue 的 GEMM：读取和每种拒绝；Python 对每种形式的条目以及对每个只改动一个字段的条目（以及带 matmul 字段的 `gemm` 条目）的判定，epilogue、`beta`、契约、验证和 canary 的读取，canary 的操作数；带 `c` 的调用的检查，带 epilogue 时的 split-K 规则 | `library_section.rs`（`a_matmul_entry_...`）；`python_library.rs`（`export_rust_vectors.py` 的 `matmul_library_entries`、`matmul_reading`）；`gemm.rs`（`a_call_of_a_matmul_has_its_bias_and_c_before_out`、`every_epilogue_but_none_has_no_split_k_unless_the_entry_opts_in`）；C ABI 的值：`tilefish-capi/tests/abi.rs`、`tests/c/abi.c`、`tests/cpp/wrapper.cpp`；GPU 上 `gpu_gate.rs`（`the_gate_has_gemms_with_an_epilogue`：加载及其 canary、每一个的 capture、在重新计算的 pin 下被拒绝的伪造 canary；`the_canary_of_a_gemm_is_never_on_the_null_stream`）、`gpu_plan_gate.rs`（plan `epilogue_chain`）和 `cpp_gate.cpp`（`matmuls`：`tilefish_gemm_epilogue`，被拒绝的伪造 canary） |
| 第 9.4 节的卷积：Python 对每种形式的条目以及对每个只改动一个字段的条目的判定，每个条目描述的 pin、canary 和 cuDNN 问题，以及 canary 的操作数 | `python_conv.rs`（`data/python_conv.json`，由 `export_rust_vectors.py` 生成）；不需要 GPU 的调用检查：`conv.rs` |
| 视图、设备标量、固定参数、二进制文件、GEMM 的 plan、digest | `schema3.rs`、`library_section.rs`、`launch_checks.rs` |
| Python 对 992 次调用（包括视图、设备标量、二进制文件）的决定、digest 向量、Python 拒绝计算 digest 的情形 | `python_launches.rs`、`python_canonical.rs`、`cross_language.rs` |
| `tensordesc`（第 5.5 节）：读取、拒绝、槽位、启动检查以及编码的参数 | `tensor_maps.rs` |
| `total_num_warps`（第 4.2 节）：读取、拒绝、plan 的块大小、digest | `warp_specialization.rs` |
| `implementation`（第 4.1 节）：名称、预构建二进制文件、同一个 stem 每个架构一个 ABI 的条目从相同的张量启动 | `schema3.rs`、`by_arch.rs` |
| 启动形状（第 4.3 到 4.6 节，7.1）：相乘的 grid、切分到 SM 的持久化 grid、cluster 及其编译选项、作为最后一个张量的全局 scratch、各种拒绝、digest | `launch_shapes.rs`；GPU 上 `gpu_gate.rs`（`the_gate_has_launch_shapes`、`a_persistent_kernel_gives_the_same_bytes_on_every_partition`、`graph_capture_of_launch_shapes`、`a_global_scratch_that_does_not_hold_the_launch_is_refused`）和 `gpu_plan_gate.rs`（`the_plans_of_launch_shapes_have_their_scratch_and_run_on_a_partition`） |
| GPU 上，与 Python 的 `replay` 在每个特性上的逐字节一致、各种拒绝、GEMM 和带步长 kernel 的 graph capture、其他 stream 的风险 | `gpu_gate.rs` 配合 `python/tests/rust_gate.py`（TMA 描述符：`the_gate_has_tma_descriptors`、`a_descriptor_of_a_tensor_the_driver_refuses_...`、`graph_capture_of_kernels_with_tma_descriptors`） |
