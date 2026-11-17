# Plan schema 1（`plan.json`，`"tilefish.plan.v1"`）

[English](../plan-schema-1.md) | **简体中文**

*launch plan*（启动计划）是把一个 `@tilefish.graph` 编译成的结果：对其各维度的 bucket（分桶）值的每一种组合，都有一个有序的 **步骤（step）** 列表（每个步骤是对一个 manifest（清单）条目的一次启动：一个 Triton kernel（核函数）、一个预构建的 PTX 或 cubin，或一个已 pin（固定）的 cuBLASLt GEMM（通用矩阵乘）），其中每个指针参数都绑定到一个 **缓冲区** 或它的一个视图；另外还有一份 **内存计划**：每个不由调用方提供的缓冲区都位于唯一一个 **arena**（内存池）中的固定字节偏移处，在每个 bucket 中都相同。持有一份经过验证的 plan（计划）及其 manifest 的 host（宿主）在一个 stream 上按顺序启动这些步骤，期间不分配任何内存，因此整个执行可以在固定地址上、按每个 bucket 一次 capture（捕获）进一个 CUDA graph（CUDA 图）。

Python 包写入并读取本文档（`tilefish.plan`、`tilefish.runtime.export`、`tilefish.run(graph, ..., mode="replay", plan=...)`）。Rust crate 实现了本规范，C++ host（`crates/tilefish-capi`、`docs/rfcs/0002-cpp-host.md`）通过这个 Rust 实现运行 plan，所以本文档对 Rust host 所说的内容对它同样成立。所有内容都由 `python/tests/` 中的测试检查（在第 14 节列出），写入或检查每个部分的 Python 代码在描述该部分的地方给出（`python/src/tilefish/_plan.py`、`_planrun.py`）。

本文档假定读者了解 manifest schema（模式）3（`docs/manifest-schema-3.md`）：`kernels` 和 `library` 的条目、`abi`、维度表达式语法、`preconditions`、规范 JSON，以及 host 对一次启动所做的启动检查。plan 不在这些之上增加任何东西：**一个步骤就是对一个 manifest 条目的一次启动，其检查和执行方式与单次启动完全相同；plan 只说明是哪个条目、使用哪块内存。**

目录：1 plan 由什么组成；2 顶层；3 digest（摘要）与 pin；4 严格读取；5 维度、bucket 与形状表达式；6 缓冲区与 arena；7 步骤（7.5 copy 步骤，7.6 memset 步骤）；8 顺序、存活性与复用；9 host 做什么；10 CUDA graph；11 Python API；12 plan 不做什么（版本 1）；13 一个示例；14 在哪里测试；15 plan set（plan 集，一个 manifest 上的多个 plan），15.1 一个 set 的各 plan 共享的 arena，15.2 carried buffer（携带缓冲区）。

## 1. plan 由什么组成

* **`dims`**：graph（计算图）的维度（其参数形状中的自由维度），每个维度带有为它规划的尺寸列表（它的 *bucket*）。一个 bucket 组合是为每个维度各取一个尺寸。每个组合有一个步骤列表，一次调用按其张量的尺寸选择其中一个，这些尺寸必须恰好等于某个 bucket（没有填充）。
* **`buffers`**：plan 命名的每个张量，放在一张表中。缓冲区要么是 *外部的*（由调用方提供：一个参数或一个返回值），要么位于 *arena* 中（一个中间张量、一个设备标量的槽位、各 GEMM 的 workspace（工作区）），要么是 *carried* 的（一个参数，其张量由 arena 在多次执行之间保存，位于 host 放在各 plan 自身区域之后的一块区域中：第 15.2 节）。它的 dtype 和形状由 graph 固定；形状是关于 `dims` 的表达式。
* **`buckets[i].steps`**：一个 bucket 组合的各次启动，按顺序排列。一个步骤指定一个 manifest 条目（`entry`，其 stem 包含该条目自身的 bucket，例如 `gemm_up.M64`），并按该条目 ABI 的顺序把它的每个指针参数绑定到一个缓冲区，以及可选地绑定到该缓冲区的一个视图。
* **内存计划**：`arena_bytes` 以及每个 arena 缓冲区的 `offset`，单位是距 arena 起点的字节数，按 256 对齐，在每个 bucket 中都相同。

视图不复制任何东西。一个读取融合的 `[M, 3, H, D]` 张量中 `q`、`k` 和 `v` 的步骤，传入同一个缓冲区的三个视图（偏移为 0、`H*D` 和 `2*H*D` 个元素，步长为 `3*H*D`、`D`、`1`），kernel 拿到三个指针，步长以标量形式传入，与任何 `Strided[...]` 指针一样（manifest schema 3，第 5.1 节）；为这三者只接收一对步长的 kernel（多个视图共享的步长，`StrideOf(("q", "k", "v"), axis)`）只拿到一次，步骤从第一个视图绑定它，并要求其他视图与之一致，与单次启动的做法相同。

## 2. 顶层

```json
{"schema": "tilefish.plan.v1", "tilefish": "0.0.3", "graph": "block", "arch": "sm_103", "manifest_digest": "<64 hex>", "dims": {"M": [32, 64, 128]},
 "reuse": false, "arena_alignment": 256, "arena_bytes": 9634048, "buffers": [ ... ], "outputs": ["out0"], "buckets": [ ... ], "plan_digest": "<64 hex>"}
```

恰好是这些键，不多也不少（否则读取方拒绝该文件），另外可选 `roles`：

| 键 | 类型与规则 |
|---|---|
| `schema` | JSON 字符串 `"tilefish.plan.v1"`。任何其他值都被拒绝（"not a tilefish plan of schema ..."）。 |
| `tilefish` | 字符串，写入该文件的包的版本。仅供参考：**它不参与 digest 计算**。 |
| `graph` | 字符串，graph 的名字：一个 ASCII 标识符（`[A-Za-z_][A-Za-z0-9_]*`）。 |
| `arch` | 字符串 `sm_<2 or 3 digits>`，不带后缀：取用 manifest 条目时所针对的架构。host 拒绝在其他架构的设备上运行该 plan。 |
| `manifest_digest` | 64 个小写十六进制数字：同一目录下 manifest 的 digest（manifest schema 3，第 3 节）。如果 manifest 的 digest 不同，该 plan 被拒绝。 |
| `dims` | 对象：维度名（标识符）-> 由 >= 1 的整数组成的非空、严格递增列表（即各 bucket）。可以是 `{}`（没有维度的 graph 只有一个 bucket 组合，即空组合）。 |
| `reuse` | 布尔值：内存计划是否允许缓冲区共享字节（第 8 节）。 |
| `arena_alignment` | 整数 `256`。 |
| `arena_bytes` | 整数 >= 0，**等于** 最后一个 arena 区域的末尾向上取整到 256：`align_up(max over arena buffers of offset + align_up(bytes), 256)`，没有 arena 缓冲区时为 `0`。 |
| `buffers` | 缓冲区对象的非空列表（第 6 节）。 |
| `outputs` | 缓冲区名字的列表：角色为 `output` 的缓冲区，每个出现一次，**按 graph 返回它们的顺序**。 |
| `buckets` | 每个 bucket 组合对应一项的列表（第 5 节），每项为 `{"values": {...}, "steps": [...]}`（第 7 节）。 |
| `plan_digest` | 64 个小写十六进制数字（第 3 节）。 |
| `roles`（可选） | 字符串 `"declared"`，当且仅当 graph 声明了哪些参数是输入、哪些是权重时出现（`@tilefish.graph(inputs=..., weights=...)`，第 6 节）。任何其他值都被拒绝。不作任何声明的 graph 的 plan 没有 `roles`，它和它的 digest 都与以往完全相同；早于该键的读取方（已发布的 crate 0.0.2、tilefish 0.0.4）拒绝带有它的 plan。 |

**整数是 JSON 整数**：在需要整数的位置，读取方拒绝 `2.0`、`1e3` 和 `true`（Python：`type(v) is int`）。

## 3. digest 与 pin

`plan_digest(doc)`（`python/src/tilefish/_plan.py`）为

```
sha256( b"tilefish.plan-digest.v1\0" + canonical_json(body) ),   body = doc without the keys "plan_digest" and "tilefish"
```

* `canonical_json` 与 manifest 的完全相同：对象键排序（码点顺序），分隔符为 `","` 和 `":"`，没有空白，非 ASCII 字符转义（`ensure_ascii`），没有 `NaN`/`Infinity`。plan 中只有字符串、整数、布尔值和 `null`（仅在内存中生成的 plan 的 `arch` 和 `manifest_digest` 为 `null`；*文件* 从不如此）。列表保持其顺序（缓冲区、bucket、步骤和参数的顺序是 plan 的一部分）。
* digest 包含 `manifest_digest`，因此 **pin 住 plan 的 digest 就 pin 住了 manifest，并随之 pin 住 plan 启动的每个二进制文件和每个已 pin 的 GEMM 算法。**
* 测试向量（`test_plan_schema.py`）：`{}` 的 digest 是 `345611870fcf5832cd7533b55c9bcb061b78c17c4f53635c9bcf8f7cffca6893`；`{"schema": "tilefish.plan.v1", "tilefish": "x", "graph": "g", "plan_digest": "ignored"}` 的 digest 是 `f59e1096fcfb737bcb745fc0a11258a25a1e2eb2fd3039b8958c777d4dadc71e`（它是标签与 `{"graph":"g","schema":"tilefish.plan.v1"}` 的 sha256）。

**plan 和 manifest 一样需要一个信任锚。** 目录中的 digest 就放在它们所描述的文件旁边，所以能写这个目录的人可以改写其中所有内容。信任锚是 **pin**（导出方返回的 `plan_digest`，保存在某个可信的地方），或者一个明确的声明，表示目录里有什么都可以接受（Python 中的 `trust_directory=True`）；两者都没有或两者都有的 replay（回放）会被拒绝。检查 **按以下顺序** 进行（`_plan.read_plan`、`_plan.load_plan`）：

1. 文件按严格 JSON 读取（第 4 节），且 `schema` 为 `"tilefish.plan.v1"`；
2. 计算它的 digest；有 pin 时它必须等于 pin（否则拒绝："has digest X, but the pin is Y"）；并且它必须等于文件自己的 `plan_digest`（否则拒绝："was edited: its plan_digest says ..."，在只信任目录时这也能发现改动）；
3. 第 2 节以及第 5 至 8 节的每条规则（`validate_doc`）；
4. 加载旁边的 manifest 并按 manifest schema 3 的规定验证（每个 cubin 的 digest），且它的 digest 必须等于 `manifest_digest`（否则拒绝："built against the manifest with digest ..."）；
5. 每个步骤都与 manifest 条目所说的一致（第 7.3 节，`check_against_manifest`）；
6. 持有 graph 的 host（Python 就是）要求文档恰好是它针对相同的 `dims` 和 `reuse` 从 graph 推导出的 plan，忽略 `arch`、`manifest_digest`、`tilefish` 和 `plan_digest`：导出之后发生过变化的 graph 被视为 *过期（stale）* 而拒绝。（没有 graph 的 host，比如 Rust host，有第 1 至 5 步和 pin。）

## 4. 严格读取

文件与 manifest 一样按严格 JSON 读取：**对象中没有重复的键，没有 `NaN`/`Infinity`，没有超出 double 表示范围的数字**（整数的绝对值必须至多为 2**53，这样每种语言读到的都是同一个数）；大于 256 MiB、嵌套过深或不是 UTF-8 的文件被拒绝。一个对象恰好具有为它列出的字段：缺少字段或有未知字段都会导致拒绝。

**限制**，使文档无法让读取方做与其大小不成比例的工作（读取方先计数再列举）：至多 **4096** 个 bucket 组合（各 `dims` 列表长度之积）、**32768** 个缓冲区、一个 bucket 中 **1 048 576** 个步骤、一个步骤中 **256** 个参数。引用文档内容的拒绝信息至多引用它所拒绝内容的前 1000 个字符。

## 5. 维度、bucket 与形状表达式

* `dims` 指定出现在任何形状中的每个维度。**bucket 组合** 是这些列表的笛卡尔积，名字排序（码点顺序），**第一个名字变化最慢**，每个列表升序：对 `{"B": [1, 2], "M": [32, 64]}`，组合依次为 `B1 M32`、`B1 M64`、`B2 M32`、`B2 M64`。`buckets` 中每个组合恰好有一项，按此顺序排列，`buckets[i].values` 是第 i 个组合的对象 `{name: size}`。
* （缓冲区的）**形状项** 是一个 >= 1 的 JSON 整数或一个字符串：一个 *维度表达式*，采用 manifest 形状的语法：整数、名字、`+`、`-`、`*`、`//`（精确整除）、一元 `-`、括号和 `cdiv(a, b)`，至多 512 个字符。Python 只写仿射形式（`"M"`、`"3*M"`、`"2*M + 3"`）。它必须只引用 `dims` 中的名字，并在每个组合下求值为 >= 1 的整数；不是“在该组合的取值下折叠为整数的仿射形式”的表达式被拒绝。
* **调用时选择 bucket。** host 根据传给它的外部张量的尺寸绑定维度（每个外部张量的 `shape` 表达式都是方程；`3*M` 对上 192 得到 `M = 64`，不是整倍数的尺寸会被拒绝），与它绑定单次启动的维度完全相同。`dims` 中的每个名字都必须被绑定，且其值必须在它的列表中；否则调用以 kind `bucket` 被拒绝，什么都不运行。两个以不同方式绑定同一名字的外部张量被拒绝。
* **只有算子的 `Out` 参数提到的维度。** extern（外部 kernel）可以写一个张量，其某个维度在调用的任何参数中都没有（由设备标量生成的 `[B]` 掩码：manifest schema 3，第 7 节）。在 graph 中，这样的调用把它绑定到 graph 中作为同一个 `Dim` 的维度（名字、`divisible_by` 和 `buckets` 都相同）：plan 是按 graph 维度的每个 bucket 生成的，所以该步骤的 `Out` 缓冲区具有该 bucket 的尺寸，步骤从这个缓冲区绑定条目的维度，就像任何拿到其 `Out` 张量的启动一样（第 7.2 节）。没有这样维度的 graph 不能进行该调用（在调用处报 `TilefishTypeError`）。

## 6. 缓冲区与 arena

`buffers` 是对象的列表，每个对象恰好具有字段 `name role dtype shape offset bytes`：

| 字段 | 类型与规则 |
|---|---|
| `name` | 字符串，一个标识符，在 plan 中唯一。graph 的参数保留自己的名字；返回值使用 graph 给它们的名字（`@tilefish.graph(outputs=("y", "residual"))`，按返回的顺序），graph 没有命名时为 `out0`、`out1`、……；中间张量按步骤生成它们的顺序命名为 `t0`、`t1`、……，workspace 命名为 `workspace`。会与参数名或返回值名冲突的名字会加上下划线。**host 按名字取返回值，或者用 `out<i>` 取第 `i` 个返回值而不管它叫什么**（这是 graph 能给返回值命名之前每个返回值都有的名字），前提是 plan 中没有缓冲区叫 `out<i>`（Python `_plan.output_buffer`，Rust `LaunchPlan::output_buffer`）。 |
| `role` | 下面八种之一。 |
| `dtype` | `bf16 f16 f32 i32 i64 u8 e4m3 e5m2` 之一（具有存储类型的 dtype）。 |
| `shape` | 由 >= 1 的整数和维度表达式组成的列表（第 5 节）；设备标量为 `[]`。 |
| `offset` | 外部角色和 `carried` 为 `null`（由 host 放置：第 15.2 节）；arena 角色为 >= 0 且是 256 倍数的整数。 |
| `bytes` | 整数 >= 1，**恰好** 是各 bucket 组合下 `product(shape) * itemsize(dtype)` 的最大值。 |

| `role` | 是什么 | 由谁提供 | 步骤可以 |
|---|---|---|---|
| `input` | 每次调用都会给出的 graph 参数（激活值） | 调用方，每次调用时 | 读 |
| `weight` | host 只绑定一次的 graph 参数（模型权重） | 调用方，一次（或每次调用时） | 读 |
| `inout` | `InOut` 参数：原地更新 | 调用方 | 读和写 |
| `output` | 返回值（调用的结果） | 调用方（除非 `outputs=` 给出，否则由 Python 的 `run` 分配） | 先写，然后读和写 |
| `intermediate` | 不返回的结果 | arena | 先写，然后读和写 |
| `devscalar` | `DevScalar` 参数：设备内存中的一个元素（`DevScalar[i32]`：一个 token 计数） | host 在第一次启动之前把值写入它的槽位 | 读 |
| `workspace` | cuBLASLt 的 workspace，以及带有 global scratch 的 kernel 的 global scratch：dtype `u8`，形状 `[N]`，`N` 为静态值 | arena | （仅作为步骤的 `workspace`） |
| `carried` | graph 声明为 carried 的参数（`@tilefish.graph(carried=("h",))`）：arena 在多次执行之间保存的张量（第 15.2 节） | arena，在它的 carried 区域中（arena 上较早的一次执行写入了它） | 读和写 |

* 最先列出的缓冲区是 graph 的参数，按其签名的顺序；它们的角色为 `input`、`weight`、`inout`、`devscalar` 和 `carried`。其余的由步骤生成，最后是 workspace。`input` 和 `weight` 都不会被写，读取方对这两者不做其他检查。**由谁选定：** 有 `"roles": "declared"` 时由 graph 选定（`@tilefish.graph(inputs=("x",))`、`weights=(...)` 或两者：每个只读的张量参数必是其中之一，`InOut` 是 `inout`，`DevScalar` 是 `devscalar`），host 可以每次实例化只绑定一次权重、每次调用只绑定输入（第 9 节）。没有 `roles` 时，写入方按形状给参数打标签（不提及任何维度时为 `weight`），这不是契约：完全静态的 graph 的每个参数都是 `weight`，包括其每次调用的输入，所以 host 不会对这种 plan 只绑定一次权重（两个 host 都拒绝这样做）。
* `devscalar` 的形状为 `[]`（一个元素）。`workspace` 至多一个，并且当且仅当某个步骤有 `workspace` 时才存在：kind 为 `library` 的步骤，或者（tilefish 0.0.7，crate 0.0.5）kind 为 `kernel` 且其条目带有 global scratch 的步骤（manifest schema 3，第 4.6 节）。它的 `bytes` 是 plan 使用的各 library 条目的 `workspace_bytes`（读取方要求它至少为其中每一个：第 7.3 节）与每个这类 kernel 步骤在每个 bucket 下的 global scratch 中的最大值：每个 CTA（线程块）的 `global_scratch` 字节数（取声明中的值：生成 plan 时不构建任何东西）乘以该步骤的 grid 在该 bucket 下的 CTA 数（持久化 grid 取 tile 数，设备运行的 program 至多为此数，再乘以 `num_ctas`）。步骤一个接一个运行，所以一块 scratch 服务所有这些步骤，就像一块 workspace 服务所有 GEMM 一样。kernel 都没有 global scratch 的 plan 与以往完全相同。
* **arena** 是 `arena_bytes` 字节的设备内存，由 host 分配一次，**按 256 字节对齐**。arena 缓冲区的区域是 `[offset, offset + align_up(bytes, 256))`，它必须位于 arena 之内。缓冲区的区域对每个 bucket 都足够大：较小的 bucket 使用它的一个前缀，所以一个缓冲区在每个 bucket 中都位于同一地址。没有 `reuse`（第 8 节）时，任何两个区域都不重叠。`carried` 缓冲区在这个布局中没有区域：host 分配 arena 时在其各 plan 的 `arena_bytes` 之后附加一块 carried 区域，并把它放在那里（第 15.2 节），所以它不与任何 plan 的任何缓冲区重叠。
* 外部张量的 `bytes` 仅供 host 自己分配时参考（最大的 bucket）；调用时它的尺寸是该 bucket 的 `product(shape) * itemsize`，由 host 检查。

## 7. 步骤

### 7.1 字段

`buckets[i].steps` 是一个对象列表（在每个 bucket 中长度相同），对象具有字段 `entry kind args`，kind 为 `library` 时以及其条目带有全局 scratch 的 kernel 步骤还有 `workspace`，对于其条目带有按调用给出的固定参数的 kernel 步骤还有 `given`；kind 为 `copy` 的步骤恰好具有字段 `kind args`（第 7.5 节），kind 为 `memset` 的步骤也是如此（第 7.6 节）：

| 字段 | 类型与规则 |
|---|---|
| `entry` | 字符串：manifest 条目的 stem（`rmsnorm`、`attention`、`gemm_qkv.M64`：一个标识符，然后算子每个分 bucket 的维度一个 `.<Dim><value>`）。它是条目自身的 bucket，对于二进制文件按 bucket 划分的算子，它在 plan 的不同 bucket 之间不同。 |
| `kind` | `"kernel"`：manifest 的 `kernels` 中的一个条目（Triton 或预构建二进制文件）。`"library"`：它的 `library` 中的一个条目（一个已 pin 的 GEMM）。manifest 必须在该部分中有 **针对 `arch`** 的这个条目。`"copy"`：一次设备到设备的复制（第 7.5 节），不指定条目。`"memset"`：把字节置零（第 7.6 节），不指定条目。 |
| `args` | 非空列表，**条目 `abi.params` 中的每个指针参数按顺序** 对应一个对象（kernel 的标量参数和 `fixed` 参数不是 args 项：由 host 推导）。`tensordesc` 参数（张量的 TMA 描述符，manifest schema 3，第 5.5 节）在这里算作指针参数：步骤绑定它的张量。同一步骤中的名字互不相同。 |
| `workspace` | `kind` 为 `"library"` 时出现；在 kind 为 `"kernel"` 的步骤上，当且仅当其条目带有 global scratch 时出现（`launch.global_scratch_size` 大于 0，manifest schema 3，第 4.6 节；tilefish 0.0.7，crate 0.0.5）：角色为 `workspace` 的缓冲区的名字。早于 kernel 步骤 `workspace` 的读取方拒绝它。 |
| `given` | 在 kind 为 `"kernel"` 的步骤上，当且仅当条目带有按调用给出的固定参数时出现（manifest schema 3，第 5.3 节：预构建二进制文件的 `kind: "given"` 参数）：一个非空对象，把其中每一个参数的名字（标识符）映射到调用给出的 JSON 整数，在其类型范围内（`i32`、`i64`；plan 中的数绝对值至多为 `2**53`）。值可以在不同 bucket 之间不同（调用可以给出关于 graph 维度的表达式，`c=Rg`，在每个 bucket 下求值）；名字不可以。它是 digest 的一部分。 |

一个参数是 `{"name": <string>, "binding": <buffer name>}`，可选地带 `"view"`：恰好是键 `name`、`binding` 和 `view`。

* `name` 是 ABI 指针参数的名字（它必须与之相等；它的存在是为了让读取方能检查顺序）。对于 library 条目，指针是该运算的操作数，名字取自它的 `abi.params`（对带 bias 的 GEMM 为 `a`、`b`、`bias`，与声明一致；对 `beta` 为 1 的 `cublaslt.matmul` 还有 `c`，manifest schema 3 第 9.5 节），然后是 `out`；对于下沉（lowering）生成的 kernel，它们包括其结果的生成名字（`out`、`out0`、`out1`）。
* `binding` 指定 plan 的一个缓冲区。它不得是 `workspace`。
* `view` 为 `{"shape": [...], "strides": [...], "offset": n}`：`shape` 是 >= 1 的整数列表，`strides` 是同样长度的 >= 0 的整数列表，`offset` 是 >= 0 的整数，**以缓冲区 dtype 的元素为单位**。没有 `view` 时参数是 **整个缓冲区**：该 bucket 下缓冲区的形状、行主序步长、偏移 0。视图必须位于其缓冲区之内：`offset + 1 + sum((shape[a] - 1) * strides[a]) <= product(buffer shape at the bucket)`。plan 的写入方（Python）当且仅当参数是整个缓冲区时省略 `view`；读取方也接受等于整个缓冲区的视图。设备标量整体绑定，不带 `view`。这些数是该 bucket 下视图的数值，无论它们来自什么表达式：形状为 `[4, "L", "L"]` 的缓冲区（每一项都是仿射形式，第 5 节）沿第一个轴的步长为 `L*L`，这是两个尺寸之积，没有任何形状项写出它，它的转置在 `L = 24` 时为 `"strides": [576, 1, 24]`（Python 0.0.5 拒绝为这样的缓冲区生成 plan；对读取方来说没有任何变化）。

**参数的指针是 `address(buffer) + view.offset * itemsize(buffer dtype)`**，其中 `address(buffer)` 是外部张量的地址，或 `arena + buffer.offset`。

### 7.2 host 如何处理一个步骤

该步骤是用其参数的张量对 manifest 条目 `entry` 的一次启动：指针 = 如上，形状 = `view.shape`（或缓冲区的形状），元素步长 = `view.strides`（或行主序步长）。其余一切与 manifest schema 3 中的单次启动相同：host **根据这些形状和步长绑定条目的维度**（那里的第 5.1 节，对大小为 1 的轴使用其规范步长；条目读取的参数没有绑定的维度从其 `out` 参数的形状绑定，那里的第 7 节），推导运行时标量（维度、元素个数、步长、`fixed` 值）并把步骤的 `given` 值传到它们的槽位，对 grid 表达式求值，并检查条目的每个前置条件（所传指针的 `pointer_alignment`、`static_size`、`divisible`、`bucket`、`max_elements`、`max_extent`），检查 ABI 条目中没有 `strides` 的指针拿到的是 **稠密** 张量（其大小 > 1 的轴的步长为行主序步长），检查秩和 dtype 与 ABI 一致，检查被写的指针（`access` 为 `out` 或 `inout`）不与该步骤的其他参数重叠（带 `alias` 的 `out` 指针与它所指的 `in` 指针除外，二者可以恰好是同一个张量：manifest schema 3，第 5.4 节），并对照设备的限制检查 grid。kernel 用 `cuLaunchKernel` 启动（Triton kernel 带上它的两个为零的 scratch 槽位，预构建 kernel 恰好带它自己的参数），使用条目的 `block`（对 warp（线程束）特化的 kernel：其所有分区的 warp，manifest schema 3，第 4.2 节）和它的 `shared` 内存；manifest schema 3 的启动形状与单次启动相同：持久化 grid 被截到 host 为之加载它的那些 SM（流式多处理器）所运行的 program 数（第 4.3 节：设备的 SM，或分区的 SM，所以在分区上准备的 plan 运行该分区对应的 program），使用 cluster（线程块簇）的 kernel 用 `cuLaunchKernelEx` 及其 cluster 启动（第 4.5 节），带 global scratch 的 kernel（第 4.6 节）以步骤的 `workspace` 所指的缓冲区作为它的 scratch，位于该缓冲区在 arena 中的地址，host 在准备该 bucket 时检查它能容纳该 bucket 下这次启动的 scratch；library 条目用 `cublasLtMatmul` 及其已 pin 的算法启动（manifest schema 3，第 9 节），其 **workspace 是 `workspace` 所指的缓冲区**，位于它在 arena 中的地址（条目的 `workspace_bytes` 总是在那里；matmul 得到的是算法所需的大小，至多为此），所以调用不分配任何内存。

**TMA 描述符参数**（条目的 `tensordesc` 参数：manifest schema 3，第 5.5 节）按该节所说进行检查（其地址 16 字节对齐，步长为 16 字节的倍数，维度在 `i32` 范围内：kind `tensor_map`），它的 `CUtensorMap` **在每次执行时根据参数的张量编码，并带有第 9 节第 3 步的每项检查**：地址 `address(buffer) + view.offset * itemsize`（arena 缓冲区位于 `arena + offset`，外部张量位于调用方的张量），视图的形状和规范步长，以及条目的 map。因此驱动拒绝的编码不会启动任何东西，步骤以它在本次执行中所绑定内存的 map 启动；生成或加载 plan 时不计算任何 map（文档中没有地址）。Python 按参数保留最近的 4096 个编码结果，这使同一绑定的重复执行无需重新编码，而任何其他绑定都对应另一个键。

**每个架构各有一个实现的算子**（manifest schema 3，第 4.1 节）无论运行哪个实现，都是一个条目名，即它的 stem，它的步骤的 `args` 指定算子的张量，每个实现都以相同顺序接收这些张量；plan 的 `arch` 对应的 manifest 中有该架构所运行实现的条目，带有它自己的 ABI（张量以 `pointer` 或 `tensordesc` 传入，它自己的标量和 `fixed` 参数，它自己的前置条件），host 像对待任何条目一样遵循它。plan 中没有任何内容说明是哪个实现，所以一个 graph 的 plan 对所有架构是同一份文档。Python 的规划器用每个实现解析每个步骤，包括它们的前置条件（比如 TMA 描述符的 16 字节步长），所以只有当每个实现都能运行时才会生成 plan。

条目所指的那一个二进制文件服务于其未分 bucket 维度的每个尺寸：一个行 kernel 针对 `M in {32, 64, 128}` 的 plan 在全部三个 bucket 中都指定条目 `rmsnorm`，`M` 从视图的形状绑定；它也服务于其按调用给出的固定参数的每个取值：同一预构建二进制文件的两个 `given` 值不同的步骤指定同一个条目。

### 7.3 plan 对照 manifest 检查什么

`check_against_manifest`（以及 Python 的 `export`，它拒绝写出未通过该检查的 plan）。对每个 bucket 的每个步骤：

* 条目针对 `arch` 存在于 `kind` 所指的部分中；对 `library` 步骤，`workspace` 缓冲区的 `bytes` 至少为条目的 `workspace_bytes`；`kernel` 步骤当且仅当其条目带有 global scratch 时有 `workspace`（`launch.global_scratch_size` 大于 0；它的一次启动需要多少字节取决于它在该 bucket 下的 grid，由 host 在准备该 bucket 时检查）；
* 条目的指针参数（`pointer` 和 `tensordesc`）个数与步骤的参数个数相同，且参数 `i` 的名字是指针 `i` 的名字；
* 步骤的 `given` 恰好指定条目中 kind 为 `given` 的参数（没有这类参数：就没有 `given`），且每个值都在参数类型的范围内；
* 缓冲区的 **dtype** 是指针的 `dtype`；视图（或缓冲区）的 **秩** 是 ABI 指针的 `len(shape)`；
* 形状为 `[]` 的指针（设备标量）绑定到角色为 `devscalar` 的缓冲区，并且只有这样的指针才如此绑定；
* access 为 `out` 或 `inout` 的指针绑定到角色为 `intermediate`、`output`、`inout` 或 `carried` 的缓冲区（步骤从不写 `input`、`weight`、`devscalar`）；
* **缓冲区先写后读**：access 为 `in` 或 `inout` 的指针绑定到 `input`、`weight`、`inout`、`devscalar` 或 `carried` 缓冲区（arena 上较早的一次执行写入了它：host 在运行 plan 时检查这一点，第 15.2 节），或者绑定到同一 bucket 中较早步骤写过的缓冲区（access 为 `out` 的指针；指向尚未写入的缓冲区的 `inout` 指针被拒绝）；
* **每个 `output` 都被写入**：每个 bucket 中都有某个步骤写它（没有步骤写的返回值会带着其张量原有的内容返回）。

copy 步骤（第 7.5 节）没有条目：上面的规则中它适用最后两条，即它的 `src` 被读（之前已写入），它的 `dst` 被写。memset 步骤（第 7.6 节）也没有条目：它的 `dst` 被写。

### 7.4 `validate_doc` 的规则对步骤和 bucket 的要求（不需要 manifest）

* 每个 bucket 的步骤数相同，并且 **在步骤 `k` 中，`kind`、`workspace`、`given` 的名字，以及每个参数的 `name` 和 `binding`，在每个 bucket 中都相同**；只有 `entry`、视图的数值和 `given` 的值可以在 bucket 之间不同；
* `given` 是从标识符到 JSON 整数的非空对象，位于 kernel 步骤上；
* 每个 `binding` 都是一个缓冲区；`workspace` 是角色为 `workspace` 的缓冲区，kind 为 `library` 的步骤有一个；`view` 格式正确，并在该 bucket 下位于其缓冲区之内；
* 在每个 bucket 中，copy 步骤符合第 7.5 节的规定，memset 步骤符合第 7.6 节的规定。

### 7.5 copy 步骤

kind 为 `"copy"` 的步骤是 `{"kind": "copy", "args": [{"name": "src", ...}, {"name": "dst", ...}]}`：恰好这两个字段（没有 `entry`：它不启动任何二进制文件；没有 `workspace`），恰好两个参数，依次名为 `src` 和 `dst`，每个都是一个缓冲区，可选地带它的一个视图，与任何参数一样。在每个 bucket 中：

* `src` 和 `dst` 都是 **稠密** 的：整个缓冲区，或者其大小不为 1 的轴具有行主序步长的视图（从视图的 `offset` 开始的一段连续元素）；
* 两者具有 **相同的 dtype**（即它们缓冲区的 dtype）和 **相同的元素个数**（形状可以不同：复制的是字节）；
* `dst` 是步骤可以写的缓冲区（`intermediate`、`output`、`inout`、`carried`），并且在同一个缓冲区中两段元素不重叠。

**host 做什么。** 复制是在 plan 的 stream 上、在步骤序列中它所在的位置执行的 `cuMemcpyDtoDAsync(address(dst) + dst.offset * itemsize, address(src) + src.offset * itemsize, count * itemsize)`；在真实地址下两段字节不得重叠（两个相互重叠的外部张量在此之前就被拒绝，第 9 节）。正在被 capture 的 stream 把它记录为一个 memcpy 节点，所以带 copy 步骤的 plan 照常可以 capture 进 CUDA graph。

**它们从哪里来。** graph 函数体中的 `tilefish.copy(x)`（一个属于它自己的新张量：从 `x`（一个张量或它的稠密视图）到一个新中间张量的 copy 步骤，之后原地调用可以更新这个中间张量，而 `x` 作为参数属于调用方，从不被写），以及对 graph 的 `InOut` 参数的 `param[...] = value`（一个写入调用方张量的 copy 步骤；`param[...] = param` 不产生步骤）。在某个 bucket 下不稠密的视图（一个置换、一个列块）的复制不是 copy 步骤：写入方在每个 bucket 中把它变成一个 kind 为 `kernel` 的步骤，启动生成的算子 `tilefish.contiguous`（把秩为 1 到 4 的带步长的视图按其步长读出，写入稠密的目标；`python/src/tilefish/_copyops.py`），所以文档中没有新东西（秩为 5 或更高的视图被拒绝）。store 也是如此：把不稠密的视图（一个转置、一个列块）通过 `param[...] = view` 存入形状和 dtype 与之相同的 `InOut` 参数，是一次输出写入调用方张量的 `tilefish.contiguous` 启动（tilefish 0.0.6 及更早版本的 tracer 拒绝这样的 store；kernel 函数体仍然拒绝）。参数自身的视图被拒绝（源和目标重叠）。早于 copy 步骤的读取方（已发布的 crate 0.0.2、tilefish 0.0.4）拒绝带有 copy 步骤的 plan（`kind` 必须是 `kernel` 或 `library`）；没有 copy 步骤的 plan 与以往完全相同。

### 7.6 memset 步骤（tilefish 0.0.7，crate 0.0.5）

kind 为 `"memset"` 的步骤是 `{"kind": "memset", "args": [{"name": "dst", ...}]}`：恰好这两个字段（没有 `entry`：它不启动任何二进制文件；没有 `workspace`），恰好一个参数，名为 `dst`，是一个缓冲区，可选地带它的一个视图，与任何参数一样。在每个 bucket 中，`dst` 是 **稠密** 的（整个缓冲区，或者其大小不为 1 的轴具有行主序步长的视图：从视图的 `offset` 开始的一段连续元素），并且是步骤可以写的缓冲区（`intermediate`、`output`、`inout`、`carried`）。它写 `dst`（第 7.3 节），并像任何参数一样计入其缓冲区的存活区间（第 8 节）。

**host 做什么。** memset 是在 plan 的 stream 上、在步骤序列中它所在的位置执行的 `cuMemsetD8Async(address(dst) + dst.offset * itemsize, 0, count * itemsize)`：这段区域的每个字节都为零（浮点数为 `+0.0`，整数为 `0`）。正在被 capture 的 stream 把它记录为一个 memset 节点，所以带 memset 步骤的 plan 照常可以 capture 进 CUDA graph，并且每次 replay 都会重新置零。

**它们从哪里来。** graph 函数体中的 `tilefish.zeros_like(x, dtype=None)`：一个形状与 `x` 相同（不读取 `x` 的任何内容）、dtype 为 `x` 的 dtype 或 `dtype` 的新张量，全为零，之后原地调用可以更新它（一个累加器），graph 也可以返回它（此时 memset 写的是调用方的输出缓冲区）。对 `InOut`（或 `carried`）参数的 `param[...] = tilefish.zeros_like(param)` 是对该缓冲区的一次 memset：当之后没有任何东西再读取这个置零后的值时，memset 直接写它，没有中间张量也没有复制。kernel 函数体中没有 memset（在那里 `zeros_like` 被拒绝）。该步骤是增量的：没有 memset 步骤的 plan 与以往完全相同，包括 digest；早于 memset 步骤的读取方（已发布的 crate 0.0.4、tilefish 0.0.6）拒绝带有 memset 步骤的 plan。

## 8. 顺序、存活性与复用

* **顺序。** 一个 bucket 的步骤按列表顺序运行，**全部在同一个 stream 上，彼此之间没有同步，也不隐含任何同步**：写一个缓冲区的步骤排在读取该缓冲区这一版本的每个步骤之前，按 graph IR 的顺序（原地调用沿用它所消费版本的存储，所以读取它的步骤排在它之后；在其缓冲区被原地更新之后才被读取的视图会被 plan 的写入方拒绝，因为 IR 赋予视图的是它被创建时的内容，而视图共享存储）。host 不对步骤重新排序、不让它们重叠或并行，也不在它们之间插入事件（第 9 节的顺序事件位于执行之间）。
* **存活性。** 缓冲区的 *存活区间* 是 `[first, last]`：以该缓冲区为参数的第一个步骤和最后一个步骤的索引（library 步骤的 workspace 在整个 plan 中都存活；`devscalar` 槽位在步骤 0 之前写入，在整个 plan 中存活）。两端都包含在内：一个读一个缓冲区、写另一个缓冲区的步骤上两者都存活。
* **复用。** `"reuse": false` 表示任何两个 arena 区域都不重叠。`"reuse": true` 允许两个区域重叠，**仅当** 两个缓冲区都是 `intermediate` 且它们的存活区间不相交（一个的 `last` < 另一个的 `first`）。设备标量的槽位和 workspace 从不共享字节。`validate_doc` 根据第一个 bucket 的步骤检查这一点（绑定在每个 bucket 中都相同）。Python 按首个步骤以 first-fit 方式为复用的 plan 布局；布局是文档的一部分，host 原样使用它。
* 没有步骤使用的缓冲区（没有任何算子读取的参数）是允许的；它没有存活区间。`carried` 缓冲区和任何缓冲区一样有存活区间，但在 plan 的布局中没有区域：它从不参与复用。
* **写在输入之上的结果。** 对于其 `Out` 可以写在某个输入之上的算子（`Out[..., MayAlias("r")]`，manifest 中的 `alias`），当这样做不会改变 graph 计算的任何值时，调用的结果绑定到 `r` 的缓冲区、同一个视图：`r` 是处于当前版本的完整 `intermediate`（由 graph 中的调用生成：不是参数，参数属于调用方，也不是视图），graph 中之后没有节点读取它的存储（它的视图、它的某个版本或 graph 的返回），并且调用的其他参数都不在该存储中。此时步骤把 `r` 和 `r_out` 绑定到同一个缓冲区（`python/src/tilefish/_plan.py`，`_in_place`）；该缓冲区由生成 `r` 的步骤写入，又被这个步骤再次写入，它的存活区间覆盖两者。否则结果有自己的缓冲区。文档不需要任何新东西：步骤中绑定到同一缓冲区的两个参数由启动检查（第 7.2 节），与任何两个参数一样。

## 9. host 做什么

1. **加载**（第 3 节）：文件、pin 或信任声明、manifest、规则、条目。每个设备分配一次 arena：`arena_bytes` 字节，256 对齐（为 0 时不分配），若 plan 有 carried buffer，则在其后再加上这些缓冲区的区域（第 15.2 节）。只要 plan 还在使用就保留它，**并且只要还有从该 plan capture 的 CUDA graph 可能被回放，就一直保留**（graph 持有它的地址：host 拥有的 graph（第 10 节）会让它的 arena 保持存活；对于调用方用 `torch.cuda.graph` capture 的 graph，没有任何东西让 arena 保持存活：Python 为每种 plan 布局和设备保留一个 arena，直到 `tilefish.runtime.clear_caches()`，无论有多少 runner 和 plan 对象来来去去；释放 arena 的 host 必须丢弃这些 graph，并且必须先等待其上最后一次执行的工作完成，无论它在哪个 stream 上）。**使用同一个 arena 的两次执行不得在设备上重叠**，无论它们被加入哪些 stream：它们共享中间张量、设备标量和 workspace。由 host 为它们排序。Python 为每个 arena 使用一个 *顺序事件（order event）*（`_planrun._Arena`）来做到这一点：一次执行在持有 arena 的锁时，让它的 stream 在第一次写（第 4 步）之前等待该事件的最近一次记录（`cuStreamWaitEvent`），并在最后一次启动（第 5 步）之后在它的 stream 上记录该事件（`cuEventRecord`），在其他启动已入队之后某次启动失败时、以及在等待之后发生中断时也同样记录（失败的记录会重试，若再次失败，host 在抛出错误之前等待该 stream；失败的等待没有入队任何东西，之后也不做记录）。**当事件的最近一次记录是在本次执行自己的 stream 上做的时，跳过等待**（在 capture 之外）：stream 顺序已经把本次执行排在该记录所代表的工作之后，所以留在同一个 stream 上的执行每次只花一次 `cuEventRecord`，换到另一个 stream 上的执行多花一次 `cuStreamWaitEvent`。记录总是会做；由于每次执行都在等待之后（或在同一 stream 上它之前的记录之后）记录，最近一次记录代表它之前的每一次执行，无论它们在哪些 stream 上。Python 在 capture 内、在每线程默认 stream 上、或在被 `torch.cuda.graph` capture 过的 arena 上从不跳过（那个 graph 的 replay 会在 host 看不到的地方记录事件）；事件在分配 arena 时于分配它的 stream 上记录一次，所以另一个 stream 上的第一次执行会等待此前使用这块内存的工作。因此，同一个 plan 在不同 stream 上、来自一个或多个线程的执行都是正确的，并按 host 发出它们的顺序在设备上一个接一个运行（只对入队排序的锁是不够的：它留下的竞争在 5 次尝试中有 5 次破坏了两个 stream 上两次运行的输出）。同一个 stream 上的执行由该 stream 排序。`clear_caches` 不会在执行进行中释放任何东西：它等待任何线程上每个进行中的执行都已记录事件（执行、capture 和 replay 以共享方式持有一个闸门，`clear_caches` 以独占方式获取它，期间开始的执行等待它，然后重新分配和加载），获取每个 arena 的锁，在释放 arena 之前等待事件（`cuEventSynchronize`），并在卸载模块之前同步每个 context。布局不同的 plan 有不同的 arena，彼此之间不排序。**顺序属于 arena，而不属于 plan**：被多个 plan 共享的 arena（第 15.1 节）用同一把锁和同一个事件为所有这些 plan 的执行排序。**所以必须并发运行的执行各需要一个 arena。** 同一个 plan 或同一个 shared arena（共享内存池）的运行在设备上是串行的，无论在哪些 stream 上；同时在多个 stream 上运行相同 plan 的 host（比如每个请求 lane（通道）一个 stream）给每个 lane 一个自己的 arena，带有自己的锁、顺序和 carried buffer。Python：`tilefish.runtime.share_arena([g, ...], name="lane0")`、`share_arena([g, ...], name="lane1")`（相同 plan、相同名字的所有 arena 是同一个；单独的一个 plan 是只有一个 plan 的 shared arena）；Rust：每个 `LaunchPlan::instantiate` 和每个 `SharedArena::new` 都分配自己的 arena（每个 lane 一个实例化，或一个 `SharedArena`）。一个 arena 上的执行从不等待另一个 arena 上的执行（一个 stream 在 arena `lane0` 上的一次运行之前睡眠 600 ms，此时另一个 stream 上 `lane1` 的运行约一毫秒完成；在 `lane0` 上的运行则要等待这次睡眠：`test_gpu_arena_lanes.py`）。Rust host（`PlanRun`，每个实例化一个 arena）用它自己的顺序事件（`plan_run::Order`）给出相同的顺序：一次执行在持有 arena 的锁时，让它的 stream 等待该事件（当事件最近一次记录就在该 stream 上时跳过），并在最后一次启动之后记录它，所以同一个实例化在任何 stream 上、来自任何线程的执行，都按发出的顺序一个接一个运行；除了本次执行自己的 stream 和它的外部张量被释放所在的 stream，不会在任何其他 stream 上入队任何东西；丢弃一个实例化的最后一个句柄时，会在 host 上等待该事件，然后才卸载其模块、释放其 arena。（到 crate 0.0.4 为止，Rust host 通过分配 arena 的那个 stream 为 arena 排序，把它当作该 stream 拥有的一块切片。）capture（`PlanRun::capture`）在持有 arena 的锁时开始，所以该 arena 在正被 capture 的 stream 上的执行会被拒绝，而不是被 capture；此后 capture 不再持有锁，因为没有执行会触及被 capture 的 stream（当执行确实会让 arena 自己的 stream 等待时，对该 stream 的 capture 必须持锁直到结束：在另一个线程上循环启动的情况下，200 次 capture 中有 3 到 18 次以 `CUDA_ERROR_STREAM_CAPTURE_ISOLATION` 或 `..._INVALIDATED` 失败）。
2. **绑定外部张量。** 调用方为每个角色为 `input`、`weight` 和 `inout` 的缓冲区以及每个 `output`（或由 host 分配）给出一个张量（地址、形状、步长、dtype），为每个 `devscalar` 给出一个值（对 `carried` 缓冲区什么都不给：由 arena 保存）。对照缓冲区检查每个外部张量：dtype、秩、连续的行主序（参数是稠密张量），并绑定维度（第 5 节；plan 先读后写的 carried buffer 也会用它被写入时的形状绑定维度：第 15.2 节）。选择 bucket。同一次调用的外部张量在其中有被写的（`output`、`inout`）时不得相互重叠，并且都不得与 arena（包括其 carried 区域）重叠。
3. **在启动任何东西之前检查**：对所选 bucket 的每个步骤，用真实地址解析它（第 7.2 节）并运行单次启动的每项检查，并在持有 arena 的锁时检查 plan 先读后写的每个 carried buffer 都持有它要读的形状（第 15.2 节）。**只要有任何检查失败，就不启动任何东西，也不写任何东西**，连设备标量也不写：先完成全部检查，再进行写入和启动。
4. **写入设备标量**，在 stream 等待 arena 上的前一次执行（第 1 步）之后。每个 `devscalar` 槽位（位于 `arena + offset`）得到调用方给出的值，以其 dtype 的小端字节表示，在 stream 上排在第一次启动之前：一次从设备内存的复制（按 stream 排序），或一次从锁页 host 内存的异步复制。（Python：`_planrun.TorchBackend.write_devscalar`；Rust：每个实例化一个锁页 staging 缓冲区，`ffi::plan::Staging`。从可分页内存的复制可能与 stream 同步，正如 CUDA 文档对从可分页内存执行 `cuMemcpyHtoDAsync` 的说明。）
5. 在 stream 上按顺序 **启动** 各步骤（第 7.2 节），然后在其上记录 arena 的顺序事件（第 1 步），并记下 plan 写入的每个 carried buffer 现在持有什么（它在该 bucket 下的形状；某次启动失败时为未知）。除此之外什么都不做：不分配，不同步 host。

host 可以按 bucket 缓存第 1 至 3 步中不依赖地址的一切（维度绑定、标量、grid、条目已加载的函数和已 pin 的算法），Python 的 runner 就是这样做的；依赖地址的检查在每次执行时重复进行。

**只绑定一次的权重（不属于文档）。** 对于带 `"roles": "declared"` 的 plan，host 可以只接收一次角色为 `weight` 的缓冲区的张量，而在每次调用时接收其他每个外部张量：它检查和启动的内容恰好是带着所有这些张量的第 2 至 5 步。权重在绑定时按外部张量的方式检查（dtype、秩、稠密性、静态尺寸；权重只会被读），并在每次调用时与其余张量一起再次检查（权重形状中提到的维度、重叠）。在调用时再次给出的权重被拒绝。Python：`tilefish.runtime.bind_weights(graph, {"w": t, ...}, mode=, plan=, pin=)`，得到一个 `WeightedRun`，其 `run(*inputs)` 和 `capture(*inputs, devscalars=...)` 接收不是权重的参数；Rust：`PlanRun::bind_weights(vec![("w", t), ...])`，得到一个 `WeightedRun<'w>`（它借用权重），其 `launch`、`launch_tapped`、`check` 和 `capture` 接收其他外部张量。两者都拒绝没有 `roles` 的 plan。

**调试 tap（观测点，不属于文档）。** host 可以在执行期间把 plan 的某个缓冲区复制出来，以观察产品实际运行的那个 plan 的中间张量：plan 及其 digest 不变。一个 *tap* 指定一个缓冲区（除 workspace 外的任何缓冲区；返回值按名字或 `out<i>` 指定）和该 bucket 的一个步骤（默认是写该缓冲区的最后一个步骤：kernel 的 `out` 或 `inout` 指针、GEMM 的 `out`、copy 的 `dst`），并给出一个 dtype 与缓冲区相同、形状为该 bucket 下缓冲区形状的稠密张量，它不与任何外部张量、任何其他 tap 以及 arena 重叠。它在第 3 步与其他一切一起检查，在第 5 步中，紧接在它的步骤之后、在下一个步骤可能覆写该缓冲区的字节之前，把整个缓冲区复制进去（`cuMemcpyDtoDAsync`，在 stream 上）（有 `reuse` 时，中间张量的字节会被复用）。Python：`tilefish.run(graph, ..., taps={"t3": tensor, ("t5", 4): tensor})` 和 `tilefish.capture(..., taps=...)`（这些复制是 graph 的 memcpy 节点）；Rust：`PlanRun::launch_tapped(stream, externals, devscalars, &mut [Tap { buffer, after, into }])`，以及 `PlanRun::capture_tapped(stream, externals, devscalars, taps)`，其 graph 把这些复制作为 memcpy 节点持有（每次 replay 都会再次复制；`CapturedPlan` 借用 tap 张量，`CapturedPlan::tap(buffer)` 是其中一个的内存）。

**单独一个步骤（不属于文档）。** host 可以在调用方的张量上启动某个 bucket 的一个步骤：使用经过验证的 manifest（即 plan 的 pin 所 pin 住的那个）中该步骤条目的二进制文件和该步骤的绑定（每个参数是步骤指定的缓冲区和视图，由为该缓冲区给出的张量构成），并先用这些地址完成该步骤一次启动的每项检查（第 7.2 节）。调用方为步骤绑定的每个缓冲区给出一个张量，不为其他任何缓冲区给出（返回值按名字或 `out<i>`；不包括 workspace，它属于 arena），张量的 dtype 与缓冲区相同，形状为该 bucket 下缓冲区的形状（设备标量：一个元素），稠密，既不与 arena 重叠，在步骤写其中某个张量时彼此也不重叠。这次启动与该 plan 的各次执行一起排序（第 9 节第 1 步）。这就是在产品所用的 pin 下、用参考实现的输入回放已导出 plan 中某一个算子的方法。Python：`tilefish.runtime.run_step(graph, step, buffers, mode=, plan=, pin=, bucket=None)`（当某个 bucket 适合这些形状时，从形状找出该 bucket）；Rust：`PlanRun::launch_step(stream, bucket, step, buffers)`。**它不会被 capture**（两个 host 都拒绝正在 capture 的 stream）：它是在产品的 pin 下、用参考实现的输入对一个算子做的检查，只做一次，graph 在这里没有任何好处，而且步骤会在 arena 的顺序下读取 plan 的 arena（GEMM 的 workspace）；想把 plan 的某一个二进制文件放进自己的 graph 的 host，从 plan 所 pin 住的 manifest 启动它（`LaunchPlan::manifest().kernel(...)`、`Kernel::launch_raw`），或者 capture 整个 plan。

## 10. CUDA graph

plan 在执行期间不做任何 stream capture 所禁止的事：不分配（GEMM 的 workspace、中间张量和设备标量都在 arena 中），不同步 host（arena 的顺序事件在 stream 上被等待和记录，作为事件节点：第 7 项），除了设备标量的复制之外没有 host 对设备内存的写入，不加载模块，不创建句柄，每次启动都在同一个 stream 上（copy 步骤是一个 memcpy 节点，memset 步骤是一个 memset 节点）。因此 **plan 可以按每个 bucket 一次、在固定地址上被 capture，并且该 graph 的 replay 逐字节一致**（Python 门禁：`test_gpu_plan.py`，与未被 capture 的 replay 以及逐算子 eager 执行逐字节一致，capture 内 0 次分配）。

plan 被 capture 进两种 graph 之一。**host 拥有的 graph**（Python：`tilefish.capture`，返回一个 `CapturedPlan`；Rust：`PlanRun::capture`，一个 `CapturedPlan`）只持有各步骤的启动，别无其他，并且只通过 host 的 replay 来回放，host 的 replay 完全像一次执行那样为它排序（第 9 节第 1 步：在持有 arena 的锁时，stream 等待顺序，从所设置的值写入设备标量，启动 graph，之后记录顺序），所以它的 replay 与该 plan 的每次执行和每次其他 replay 一起排序，无论来自哪个线程、在哪个 stream 上。**调用方拥有的 graph**（Python：在调用方的 `torch.cuda.graph(...)` 中调用 `tilefish.run`）由持有它的人回放，不持有 host 的任何锁：第 7 项说明了它的事件节点为什么排序、不为什么排序。必须满足的条件：

1. **所有会分配或等待的操作都在 capture 之前完成**：arena、该 bucket 的每个函数的加载（`cuModuleLoadData`，以及驱动提供时的 `cuFuncLoad`，使惰性加载不会发生在 capture 内）、cuBLASLt 句柄（`cublasLtCreate`），以及只依赖 bucket 的检查（对照设备的限制检查 grid，每个已 pin 算法的 `cublasLtMatmulAlgoCheck`：Python 在准备 bucket 时完成它们，所以拒绝发生在任何运行的第一次启动之前）。Python：先运行一次 plan，或调用 `tilefish.runtime.prepare(graph, ...)`。Python 拒绝会做其中任何一件事的 capture（"has not been prepared"）。
2. **输出** 最好是调用方给出的张量（Python 中的 `outputs=`），并且与 graph 存活得一样久：在 capture 内分配的内存属于该 graph（torch 允许这样做，从 capture 的私有池中分配，Python 也不拒绝；但这终究是 capture 内的一次分配，所以证明没有分配的测试会传入 `outputs=`）。
3. **设备标量必须由一次被 capture 的、设备端的复制写入**，来源是一个单元素的 CUDA 张量：graph 在每次 replay 时重新读取该张量，所以在两次 replay 之间改变值就是写这个张量。`int` 或 CPU 张量是一次 host 写入（从锁页内存的异步复制不能作为一个会变化的值被 capture），Python 在 capture 下拒绝它。（host 拥有的 graph 把这次写入放在 graph 之外。Python 的 `CapturedPlan.replay` 在等待顺序之后、启动 graph 之前写每个槽位，方式是从锁页 host 内存异步复制 `CapturedPlan.set_devscalar` 设置的值（或者对在那里给出的 CUDA 张量做一次按 stream 排序的复制）：host 不被同步，期间插入的执行虽然也写这些槽位，也不会改变 graph 读到的内容。Rust host 的 `CapturedPlan::replay` 在启动 graph 之前，只要槽位中不是该 graph 的值，就用从锁页 host 内存的异步复制写入它们；`CapturedPlan::set_devscalar` 改变这些值。）
4. **graph 持有的是地址，而不是值**：每次 replay 的外部张量都是同样的张量（原地写入它们），并且 **被 capture 的 graph 的 arena 就是 plan 的 arena**，即未被 capture 的执行所用的那个，位于 graph 持有的地址（对于在 shared arena 上运行的 plan，就是该 shared arena：第 15.1 节）。**TMA 描述符按值持有**：带 `tensordesc` 参数（第 7.2 节）的步骤的 kernel 节点持有为该 capture 编码的 `CUtensorMap` 的 128 字节，其中写明了 capture 所绑定的地址、形状和步长：arena 缓冲区位于它在这个 arena 中的偏移，外部张量位于 capture 时给出的张量，以及该步骤的视图和 bucket。因此 graph 的描述符由它的 bucket、它的 arena（地址和 plan 的布局）以及它的外部张量固定，与它的指针完全一样，并且只要这些不变，它们对每次 replay 都是正确的（第 4 项本来就要求这些不变：不需要别的）。graph 中没有任何需要打补丁的东西：绑定另一个张量或另一个 arena 是另一次执行或另一次 capture，它会对所绑定内容的 map 进行编码（Python 的 `tilefish.capture` 和对 `tilefish.run` 的 `torch.cuda.graph`、Rust 的 `PlanRun::capture`，都在 capture 开始之前的检查中编码）。在两次 replay 之间向同样的张量写入新值（第 4 项）不会改变任何描述符。
5. **一个 stream**：capture 的是启动步骤的那个 stream；所有步骤都放在那里。同一次 capture 中一个 plan 的每次执行都在同一个 stream 上：对于同一次 capture（相同的 `cuStreamGetCaptureInfo` id）中在另一个 stream（比如从正在 capture 的 stream 分叉出来的 stream）上对某个 plan 的 arena 的第二次执行，Python 在它入队任何东西之前就拒绝（它的等待节点不会依赖 graph 内第一次执行的记录节点，所以一次 replay 可能在该 arena 上同时运行两者；CUDA 没有说明会为它们排序，尽管 B300 驱动在这种 graph 的 2 800 次 replay 中都是一个接一个运行的）。
6. **每个 graph 一个 bucket**：一个 graph 是为一个 bucket 组合的步骤生成的，从第一次启动到最后一次，并为该组合的张量而 replay。
7. **replay 的顺序。** 对 arena 顺序事件的等待及其记录（第 9 节第 1 步）被 capture 为 *外部*（external）事件节点（`CU_EVENT_WAIT_EXTERNAL`、`CU_EVENT_RECORD_EXTERNAL`；capture 中不允许对在 capture 之外记录的事件做普通等待，普通记录只会加入 capture）。因此 graph 的每次 replay 都等待它启动之前对该事件的最近一次记录，并在其工作完成时记录该事件。**这只让一次 replay 相对于它能看到其记录的那些执行排序。** 调用方拥有的 graph 的 replay 不持有 host 的任何锁：在 replay 启动时，另一个线程上一次正处于等待与记录之间的执行已经等过它之前的记录，但还没有记录，所以 replay 等待的是与该执行相同的记录，两者在 arena 上重叠，产生错误的字节而没有任何报错（Python，在 B300 上：一个线程上的 4 000 次执行中有 3 次在另一个线程循环回放一个 `torch.cuda.graph` 的情况下结果错误；同一测试中改用 `tilefish.capture` 时 8 000 次中 0 次）。所以，在另一个线程可能使用同一 plan 时（一次执行、一次 host 拥有的 graph 的 replay（它的等待和记录也是两次调用），或者另一个调用方拥有的 graph 的 replay）回放自己 graph 的调用方，必须自己为它们排序（使用一个线程，或者用它自己的一把锁包住所有这些操作），或者把 plan capture 进 host 拥有的 graph。从一个线程发出时，同一 plan 的 replay 和未被 capture 的执行，无论在哪些 stream 上，都按 host 发出的顺序相互排序，与执行完全一样。调用方只需为它自己的张量排序（在另一个 stream 上写入的输入、在某个 stream 上读取的输出），与任何启动一样。**这个隐患会被明确报出，补救只需一次调用。** 调用方拥有的 graph 的 replay 是对 torch 的一次调用，tilefish 看不到，所以无法在调用方背后让它变得安全（replay 必须获取 arena 的锁，而只有 host 的调用才能做到；等待和记录无法合并为一次驱动调用；设备上的无锁排序需要一张 graph 在 replay 时无法领取的票据）。因此 Python (a) 发出警告：被调用方拥有的 graph 持有的 arena 被第二个线程使用时（一次运行、一次 host 拥有的 graph 的 replay、单独一个步骤、另一次 capture），给出一次 `TilefishOrderWarning`（使用 `warnings.simplefilter("error", tilefish.TilefishOrderWarning)` 时，它是一个错误，在入队任何东西之前就拒绝每一次这样的使用；单个线程从不警告，并且一旦调用方通过 (b) 回放，就不再给出该警告）；(b) 提供 `tilefish.runtime.replay_graph(g, stream=None)`，它在调用方拥有的 graph 被 capture 时所在的每个 arena 的锁下回放 `g`，因此在启动 `g` 时没有其他线程的执行处于其等待与记录之间，其余的由事件节点完成：在另一个线程循环 `replay_graph` 的情况下 3 000 次运行中 0 次错误（`test_gpu_arena_lanes.py`）。警告看不到在一个从不调用 tilefish 的线程上直接用 `g.replay()` 做的 replay；它看到的是调用 tilefish 的第二个线程。Rust host 没有调用方拥有的 graph：在正在 capture 的 stream 上的启动被拒绝，`PlanRun::capture` 是 capture plan 的方式。（host 拥有的 graph 没有事件节点：Python 的 `CapturedPlan.replay` 和 Rust 的 `CapturedPlan::replay` 完全像一次执行那样为 replay 排序，即第 9 节第 1 步，所以同一 plan（或同一个 Rust 实例化）的 replay 和执行以相同方式排序，无论由哪个线程发出。）
8. 不是 plan 造成的、但会导致 capture 失败的原因：本身不能被 capture 的二进制文件（两个 host 都拒绝 cooperative launch、PDL 和 profile scratch，所以这些都不会出现在 plan 中；使用 cluster 的 kernel 连同其 cluster 一起被 capture，global scratch 是 arena 的 workspace，位于 graph 持有的地址），以及框架的 capture 模式（torch 的 `global` 模式在 capture 期间拒绝来自另一个线程的不安全调用）。

## 11. Python API

```python
plan = tilefish.plan(graph, buckets={"M": [32, 64, 128]}, reuse=False)      # a Plan: no GPU, no torch, nothing built
plan.doc, plan.digest, plan.arena_bytes, plan.arena_bytes_without_reuse, plan.arena_bytes_with_reuse, plan.steps(M=64), plan.buffers, plan.describe()
p = tilefish.runtime.export(graph, "build/", buckets=..., reuse=...)        # builds the binaries, writes manifest.json and plan.json; returns the Plan loaded back (p.digest to pin)
y = tilefish.run(graph, *inputs)                                            # jit: the plan in memory (tilefish.plan(graph), or plan=<a Plan>), binaries built in memory
y = tilefish.run(graph, *inputs, mode="replay", plan="build/", pin=p.digest)  # replay: the exported plan, verified; or trust_directory=True
y = tilefish.run(graph, *inputs, mode="reference")                          # the ops' references composed in torch (an extern's reference= is its oracle)
tilefish.run(graph, *inputs, outputs=out)                                   # write the returned values into these tensors instead of allocating
tilefish.run(graph, *inputs, outputs={"y": y, "residual": r})              # the same by name (the graph's names, or out0, out1, ...)
tilefish.run(graph, *inputs, taps={"t3": t, ("t5", 4): u})                 # copy buffers out after a step (section 9: debug taps; the plan is unchanged)
plan.shape_of("t3", M=64)                                                   # the shape of a buffer at a bucket (what a tap is given)
w = tilefish.runtime.bind_weights(graph, {"w_o": w_o, ...}, mode="replay", plan="build/", pin=p.digest)  # a plan with declared roles (section 6)
w.run(x, n); w.capture(x, devscalars={"n": 5})                              # the inputs only: the weights are bound (section 9)
tilefish.runtime.prepare(graph, mode="replay", plan="build/", pin=p.digest) # arena, binaries, handle: before a CUDA graph capture
tilefish.runtime.run_step(graph, 3, {"t2": a, "w_o": w, "t3": o}, mode="replay", plan="build/", pin=p.digest)  # one step alone (section 9)
cap = tilefish.capture(graph, *externals, devscalars={"n": 5}, outputs=out) # a CapturedPlan: the plan in a CUDA graph of its own (mode=, plan=, pin= as for run)
cap.replay(stream=None)                                                     # ordered with every run and replay of the plan, from any thread
cap.set_devscalar("n", 7)                                                   # the value the next replays write (nothing is enqueued now)
s = tilefish.runtime.share_arena([g1, g2], mode="replay", plan="build/", pin=planset.digest)  # plans of a set on one arena they share (section 15.1)
a = s.run(g1, x); y = s.run(g2, a); s.capture(g2, a)                        # one at a time: every run and replay on the arena is ordered with the others
lane1 = tilefish.runtime.share_arena([g1, g2], name="lane1", ...)          # another arena of the same plans: its executions run concurrently with s's
s.run(stage_a, x); z = s.run(stage_b)                                       # carried buffers (section 15.2): stage_a leaves h in the arena, stage_b reads it
tilefish.runtime.replay_graph(g)                                            # replay a torch.cuda.graph holding plans, ordered with every thread (section 10)
```

在 graph 函数体中，`tilefish.copy(x)` 是一个 copy 步骤（第 7.5 节），`tilefish.zeros_like(x, dtype=None)` 是一个 memset 步骤（第 7.6 节），对 `InOut` 参数的 `param[...] = x` 是一个 copy 步骤或一次 `tilefish.contiguous` 的启动，`@tilefish.graph(carried=("h",))` 声明 carried 参数（第 15.2 节）。

* `buckets` 把维度名映射到要规划的尺寸；默认取维度自己的 `buckets=`，对没有 `buckets=` 的维度则必须给出。`reuse` 默认为 `False`。
* 调用时，`DevScalar` 参数接收一个 `int`（不是其 dtype 整数的数，比如对 `i32` 给 `3.7`，在 `jit`、`replay` 和 `reference` 中同样被拒绝：它从不被截断）、一个 CPU 张量或一个单元素的 CUDA 张量。
* graph 中某个调用给出的值不同于声明默认值的编译期参数（`const`），在生成 plan 时被拒绝，与单次启动时一样（二进制文件中编译进了声明的值）；`reference` 模式下也是如此。
* `tilefish.capture(graph, *externals, mode=..., plan=..., pin=..., trust_directory=..., devscalars=..., outputs=..., stream=None)` 准备 plan，然后在 `stream`（默认：一个新 stream）上、以线程局部的 capture 模式，把外部张量所选 bucket 的各次启动 capture 进它自己的 `torch.cuda.CUDAGraph`，内部不分配任何东西。`externals` 是 graph 按签名顺序的参数，不含其 `DevScalar` 参数，后者的值由 `devscalars=` 给出（每一个都要给：一个数，或 host 或设备上的单元素张量）；`outputs=` 默认是在 capture 之前分配的张量（`cap.outputs`）。`CapturedPlan` 让外部张量和 arena 保持存活。在 capture 内以及在 `clear_caches` 之后，`replay` 被拒绝。
* `tilefish.runtime.clear_caches()` 等待每个线程上进行中的执行，重置每个 runner 并释放每个 arena（在其上最后一次执行完成之后），所以只有在不会再回放任何用 `torch.cuda.graph` capture 的 graph 时才调用它（`CapturedPlan` 让它的 arena 保持存活，但会拒绝 replay：需要重新 capture）。
* `jit` 在内存中用相同的编译规格构建 plan 的二进制文件，并在内存中 pin 相同的 GEMM 算法，运行与 `replay` *相同* 的 runner，`replay` 的不同之处在于二进制文件的来源（一个对照规格验证过的 manifest）；所以两者在构造上就是一致的（有测试要求它们逐字节一致）。`jit` 运行的 plan 要遵守读取方要求文件遵守的每一条规则（第 2 节以及第 5 至 8 节，`validate_doc(doc, made=True)`，其中包括对 arena 的重叠扫描；只有它的 `arch` 和 `manifest_digest` 为 `null`）：`tilefish.plan` 检查它生成的每个 plan。

## 12. plan 不做什么（版本 1）

plan 无法生成的 graph 函数体会被 `tilefish.plan` 拒绝，并给出相应的规则（`TilefishPlanError`）：

* graph 函数体中对 `InOut` 参数的 **scatter**，以及由 `tilefish.load` 读出、又被返回、存储或传给某个调用的值。（其他 **在 graph 函数体中追踪到的算术**，如 `x + y`、类型转换、比较、归约，都会被规划：写入方把它们分组为融合区域，每个融合区域是一个 kind 为 `kernel` 的步骤，启动一个由 tilefish 生成、按内容命名为 `fused_<hash>` 的 kernel，与每个算子一个 kernel 的组合逐字节一致；`docs/rfcs/0001-plan-fusion.md`。文档中没有新东西：读取方看到的是 manifest 条目的 kernel 步骤，与任何 `@tilefish.kernel` 一样。）
* graph 的 **`Out`** 参数（因此也包括对它的 `store`）以及运行时的 **`Dim`** 参数：graph 的结果就是它的返回值（对 `InOut` 参数的 `store` 是一个 copy 步骤，对不稠密的视图则是一次 `tilefish.contiguous` 的启动，第 7.5 节）；
* 对 **另一个 graph** 的调用（目前尚不内联）。
* 对另一个张量的 **视图做原地调用**（IR 规定更新视图不会改变其父张量；plan 没有能使之成立的复制），以及 **在其张量被原地更新之后被读取** 的视图。
* 是 **`DevScalar`** 的返回值，或者在视图创建之后其张量被原地更新的视图（**不稠密** 的视图是一次输出写入调用方输出的 `tilefish.contiguous` 启动，第 7.5 节）。（其他任何返回值都会被规划：调用生成的张量就是调用方的输出缓冲区本身；**以另一种形状表示的整个张量**（一次 reshape：偏移 0、行主序步长、元素个数相同）也是如此，此时输出缓冲区具有返回的形状，写入和读取它的步骤使用它们自己形状的视图；**任何其他稠密视图、参数以及被返回两次的值**，在最后一次调用之后由一个 copy 步骤复制进调用方的缓冲区，第 7.5 节。）
* **没有 bucket 的维度**、**没有实现的** 算子（没有 `impl` 的 `library` op（库算子））、同名的两个不同算子、block-scaled 张量。

## 13. 一个示例

一个带原地张量的小 graph 的 plan（`python/tests/plan_model.py` 中的 `accumulate`，`buckets={"Q": [16, 32]}`，`reuse=True`；`tilefish` 键和两个 digest 不是真实导出中的值，真实导出有 manifest；`test_plan_docs.py` 检查这段文本就是代码生成的内容，且 digest 正确）：

<!-- plan-example -->
```json
{"schema": "tilefish.plan.v1", "tilefish": "0.0.3", "graph": "accumulate", "arch": "sm_103", "manifest_digest": "abababababababababababababababababababababababababababababababab", "dims": {"Q": [16, 32]}, "reuse": true, "arena_alignment": 256, "arena_bytes": 16384, "buffers": [{"name": "a", "role": "input", "dtype": "f32", "shape": ["Q", 128], "offset": null, "bytes": 16384}, {"name": "state", "role": "inout", "dtype": "f32", "shape": ["Q", 128], "offset": null, "bytes": 16384}, {"name": "t0", "role": "intermediate", "dtype": "f32", "shape": ["Q", 128], "offset": 0, "bytes": 16384}, {"name": "out0", "role": "output", "dtype": "f32", "shape": ["Q", 128], "offset": null, "bytes": 16384}], "outputs": ["out0"], "buckets": [{"values": {"Q": 16}, "steps": [{"entry": "bump", "kind": "kernel", "args": [{"name": "state", "binding": "state"}, {"name": "y", "binding": "a"}]}, {"entry": "bump", "kind": "kernel", "args": [{"name": "state", "binding": "state"}, {"name": "y", "binding": "a"}]}, {"entry": "twice", "kind": "kernel", "args": [{"name": "x", "binding": "a"}, {"name": "out", "binding": "t0"}]}, {"entry": "bump", "kind": "kernel", "args": [{"name": "state", "binding": "t0"}, {"name": "y", "binding": "a"}]}, {"entry": "plus", "kind": "kernel", "args": [{"name": "x", "binding": "t0"}, {"name": "y", "binding": "state"}, {"name": "out", "binding": "out0"}]}]}, {"values": {"Q": 32}, "steps": [{"entry": "bump", "kind": "kernel", "args": [{"name": "state", "binding": "state"}, {"name": "y", "binding": "a"}]}, {"entry": "bump", "kind": "kernel", "args": [{"name": "state", "binding": "state"}, {"name": "y", "binding": "a"}]}, {"entry": "twice", "kind": "kernel", "args": [{"name": "x", "binding": "a"}, {"name": "out", "binding": "t0"}]}, {"entry": "bump", "kind": "kernel", "args": [{"name": "state", "binding": "t0"}, {"name": "y", "binding": "a"}]}, {"entry": "plus", "kind": "kernel", "args": [{"name": "x", "binding": "t0"}, {"name": "y", "binding": "state"}, {"name": "out", "binding": "out0"}]}]}], "plan_digest": "e4dc0d05d73b157aa1860171601687ce7d40aba3b8dba1f2f41c1dd2c6112bcf"}
```

解读：`state` 被原地更新两次（步骤 0 和 1），`twice` 从 `a` 生成位于 arena 偏移 0 处的 `t0`，`bump` 原地更新 `t0`（步骤 3：没有第二个缓冲区），`plus` 读取两者的最新版本并把返回值写入 `out0`。两个 bucket 之间除了缓冲区在其中的尺寸之外没有任何不同（以及每个步骤的 stem，这里它没有自己的 bucket）。

带视图的 plan 的步骤，即门禁中 transformer block（`python/tests/plan_model.py`）的 attention 在 `M = 64` 时的步骤：`q`、`k` 和 `v` 是融合输出 `t1` 的视图，结果整体绑定：

```json
{"entry": "attention", "kind": "kernel", "args": [
  {"name": "q", "binding": "t1", "view": {"shape": [64, 4, 64], "strides": [768, 64, 1], "offset": 0}},
  {"name": "k", "binding": "t1", "view": {"shape": [64, 4, 64], "strides": [768, 64, 1], "offset": 256}},
  {"name": "v", "binding": "t1", "view": {"shape": [64, 4, 64], "strides": [768, 64, 1], "offset": 512}},
  {"name": "o", "binding": "t2"},
  {"name": "n_valid", "binding": "n"}]}
```

该 block 的 plan 在 3 个 bucket 中每个都有 9 个步骤（5 个 kernel 和 4 个 GEMM），不复用时 arena 为 9 634 048 字节（其中 8 519 680 是 cuBLASLt workspace，1 114 112 是中间张量），复用时为 9 240 832 字节（中间张量占 720 896）。

## 14. 在哪里测试

| 内容 | 测试（`python/tests/`） |
|---|---|
| 遍历、步骤和视图、缓冲区、角色、bucket、第 12 节中的拒绝 | `test_plan.py` |
| 用一对步长读取三个视图的步骤；某个维度只有其 `Out` 缓冲区才有的步骤（第 5 节） | `test_shared_strides.py`、`test_output_dims.py`；GPU 上 `test_gpu_shared_strides.py`、`test_gpu_output_dims.py` |
| 写在输入之上的结果（`MayAlias`，第 8 节） | `test_alias.py`；GPU 上 `test_gpu_alias.py` |
| 步长是尺寸之积的缓冲区（`[4, Lb, Lb]`：步长和偏移为 `Lb*Lb` 的视图，第 7.1 节） | `test_product_strides.py`；GPU 上 `test_gpu_product_strides.py`；Rust 门禁的 plan `product_strides`（`gpu_plan_gate.rs`） |
| copy 步骤（`tilefish.copy`、对 `InOut` 参数的 store；第 7.5 节） | `test_plan_copy.py`；GPU 上 `test_gpu_plan_copy.py`；Rust 读取方：`python_plans.json`（`copy ...` 文档） |
| 存入 `InOut` 参数的不稠密视图：一次 `tilefish.contiguous` 的启动（第 7.5 节） | `test_plan_strided_store.py`、`test_trace.py`；GPU 上 `test_gpu_plan_strided_store.py`（jit、replay、reference、`tilefish.capture`、`torch.cuda.graph`）以及门禁 plan `strided_store` |
| memset 步骤（`tilefish.zeros_like`；第 7.6 节） | `test_plan_memset.py`；Rust 读取方：`python_plans.json`（`memset ...` 文档）；GPU 上 `test_gpu_plan_memset.py`（调用方的张量先用 NaN 填充；被 capture 为 memset 节点）以及门禁 plan `memset_steps` |
| carried buffer（第 15.2 节）：角色、由生成值的步骤写入的 store、carried 区域的布局、各自所持内容的顺序、从中绑定的维度、各种拒绝 | `test_plan_carried.py`；`test_plan_fixtures.py` 和 `launch_plans.rs`（golden set `carried`：`the_golden_set_of_carried_buffers_...`）；Rust 读取方：`python_plans.json`（`carried ...` 文档）；GPU 上 `test_gpu_plan_carried.py`（jit 和 replay 对照 torch、在其他 stream 上的运行之间的 `tilefish.capture`、两个阶段的 `torch.cuda.graph`、来自两个线程的两个 arena）以及 `gpu_plan_gate.rs`（`carried_buffers_handed_from_plan_to_plan_...`：`rust_gate.py` 的 `CARRIED_CALLS` 中的调用在同一个 `SharedArena` 上启动并 replay） |
| 每个 stream 一个 arena（`share_arena(..., name=)`，第 9 节），以及针对调用方 graph 的警告和 `replay_graph`（第 10 节第 7 项） | `test_arena_lanes.py`；GPU 上 `test_gpu_arena_lanes.py`（另一个 arena 上的运行不等待正在睡眠的 stream；两个线程；在另一个线程上 `replay_graph` 的情况下 3 000 次运行中 0 次错误） |
| 返回的视图和参数（第 12 节） | `test_plan_returns.py`；GPU 上 `test_gpu_plan_returns.py` |
| 单独一个步骤（第 9 节） | `test_plan_step.py`；GPU 上 `test_gpu_plan_step.py` 和 `gpu_plan_gate.rs`（`the_steps_launched_one_by_one_give_the_bytes_of_the_plan`） |
| 带 TMA 描述符的步骤（第 7.2 节和第 10 节）：在第一次启动之前根据所绑定的缓冲区编码，在 graph 中按值持有 | `test_tensordesc.py`（假的后端）；GPU 上 `test_gpu_tma.py`（jit、replay、`tilefish.capture`、`torch.cuda.graph`、单独一个步骤）以及门禁 plan `tma_chain`、`tma_chain-reuse`、`tma_view`（`gpu_plan_gate.rs`：启动，并从 graph 回放） |
| 调试 tap（第 9 节） | `test_plan_taps.py`；GPU 上 `test_gpu_plan_taps.py` 和 `gpu_plan_gate.rs`（`taps_are_pythons_bytes`：Rust 的 tap 对照 Python 的） |
| 参数的角色，声明的或按形状选定的（第 2、6 节），以及只绑定一次的权重（第 9 节） | `test_plan_roles.py`；golden plan `steps`（`launch_plans.rs`）；GPU 上 `test_gpu_plan_roles.py` 和 `gpu_plan_gate.rs`（`weights_bound_once_give_pythons_bytes`） |
| 预构建二进制文件按调用给出的固定参数：步骤的 `given`（第 7 节） | `test_given_args.py`；golden plan `given`（`launch_plans.rs`，以及 `python_plans.json` 中的 `given ...` 文档）；GPU 上 `test_gpu_given_args.py`、`gpu_gate.rs`（`one_binary_serves_the_values_each_call_gives`）以及门禁 plan `given_steps` |
| 被返回或被复制的不稠密视图：`tilefish.contiguous` 的启动（第 7.5 节） | `test_plan_materialise.py`；GPU 上 `test_gpu_plan_materialise.py` 以及门禁 plan `strided_returns` |
| 返回值的名字（第 6 节） | `test_plan_names.py`；golden plan `steps`（`launch_plans.rs`）；GPU 上 `gpu_plan_gate.rs`（`a_named_output_is_given_by_its_name_or_by_out_i`） |
| 内存计划：对齐、无重叠、对照模拟 arena 的按存活性复用 | `test_plan_arena.py` |
| digest、pin、格式的每条规则、manifest 检查、篡改 | `test_plan_schema.py` |
| 第 13 节的文档 | `test_plan_docs.py` |
| runner：顺序、地址、标量、第一次启动之前的前置条件、capture、arena 的顺序事件（假的后端） | `test_planrun.py` |
| GPU 上：replay == jit == 逐算子 eager、replay 对比 reference、CUDA graph capture、篡改、arena | `test_gpu_plan.py` |
| GPU 上：一个 plan 在两个 stream 上的运行（评审中发现的竞争，20 轮中 0 轮被破坏）、来自两个线程的运行，以及在其他 stream 上的运行之间对被 capture 的 graph 的 replay | `test_gpu_plan_streams.py` |
| plan set：一个 manifest、每个 graph 一个 `<graph>.plan.json`、索引 `plans.json`、两种 pin 之一（第 15 节） | `test_plan_sets.py`；golden set `golden/plans/set`（`launch_plans.rs`）；GPU 上 `test_gpu_plan_sets.py` 和 `gpu_plan_gate.rs`（`a_plan_set_is_loaded_with_one_pin_and_its_plans_give_pythons_bytes`） |
| 一个 set 的各 plan 共享的 arena：其大小和布局、唯一的顺序、各种拒绝、pin（第 15.1 节） | `test_shared_arena.py`；`test_plan_fixtures.py` 和 `launch_plans.rs`（golden set 的 `shared_arena_bytes`）；GPU 上 `test_gpu_shared_arena.py`（block 的各阶段在一个 arena 上：得到分开的 arena 和整个 block 的字节，被 capture 时以及在 torch graph 中也是如此；两个 plan 在两个 stream 上竞争，10 轮中 0 轮错误，没有顺序事件时 10 轮中 10 轮错误）以及 `gpu_plan_gate.rs`（`the_plans_of_a_set_on_one_shared_arena_...`、`launches_and_replays_of_different_plans_of_a_shared_arena_...`、`a_shared_arena_refuses_...`） |
| **供第二个实现使用的 golden 样例**：三个 plan 目录（`manifest.json`、cubin 和 `plan.json`；cubin 是假的，任何驱动都不会加载：这些样例是给读取方用的，可在任何机器上使用）：transformer block（不复用和复用两种）以及 `steps`（写在输入之上的结果、两个 copy 步骤、一个返回的 reshape、命名的输出）；`expected.json`，包含读取方必须仅凭这些文件复现的 plan digest、manifest digest、arena 大小、每个 arena 偏移以及输出 | `golden/plans/`（由 `export_plan_fixtures.py` 写出，不需要 GPU）、`test_plan_fixtures.py` |

golden 样例的 cubin 是假的（每个 kernel 条目一个很小的 ELF 文件，每个 GEMM 一个虚构的算法）：格式、digest 和 plan 是真实的，所以可以用它们检查读取方；它们不能被启动。

## 15. plan set：一个 manifest 上的多个 plan

启动次数取决于某个键的阶段（处理 `ceil(T / tile)` 个 tile 的 prologue（头部），每个 tile 是一串固定的启动）是每个键一个 plan：一个 plan 的步骤在每个 bucket 中都相同。逐个导出时，每个 plan 目录都有自己的 manifest，以及同一批二进制文件的一份自己的副本。**plan set** 把它们一起导出：`tilefish.runtime.export([g1, g2, ...], directory)`（一项可以是 `(graph, buckets)`）把所有这些的二进制文件构建一次，并写出

* 一个 `manifest.json`（schema 3，见那里的各节）：每个二进制文件一个条目，无论哪些 plan 启动它；
* 每个 graph 一个 `<graph>.plan.json`：本 schema 的一个 plan，与 `plan.json` 完全相同，其 `manifest_digest` 是那唯一一个 manifest 的 digest；
* `plans.json`，即 **索引**：`{"schema": "tilefish.plan-set.v1", "tilefish": <version>, "arch": "sm_103", "manifest_digest": <64 hex>, "plans": {"<graph>": "<plan digest>", ...}, "set_digest": <64 hex>}`，恰好这些键；`plans` 是从 graph 名（标识符）到 plan digest 的非空对象（至多 65 536 项）。`set_digest` 是 `sha256(b"tilefish.plan-set-digest.v1\0" + canonical_json(index without "set_digest" and "tilefish"))`（`_plan.set_digest`；Rust `plan_set_digest`）。

**pin。** set 的 digest 覆盖它列出的每个 plan 的 digest，而每个 plan 的 digest 覆盖 manifest 的 digest（第 3 节），所以 **一个 pin，即 set 的 pin，覆盖每个 plan、它们的 manifest 以及它们启动的每个二进制文件**。host 严格读取索引（与 plan 一样，第 4 节），对照 pin 和它自己的 `set_digest` 检查它的 digest，然后检查上面的每条规则，以索引的 `manifest_digest` 作为 pin 加载 manifest 一次，并以索引列出的 digest 作为 pin、配合该 manifest 从 `<graph>.plan.json` 加载 plan（所以不是所列 plan 的 plan 文件会被拒绝）。每个 plan 文件也是一个普通的 plan：它可以以自己的 digest 作为 pin 单独加载（使用它旁边的 manifest）。

各 host：Python `tilefish.run(graph, ..., mode="replay", plan=directory, pin=<the set's digest or the plan's>)`（目录中的 `<graph>.plan.json`；`tilefish.capture`、`prepare`、`run_step` 和 `bind_weights` 同理），`export` 返回一个 `PlanSet`（`.digest`、`.manifest_digest`、`.plans`）。Rust：`PlanSet::load(directory, Pin::Digest(set_digest))`、`set.plan("tiles2")`（一个共享 set 的 manifest 的 `LaunchPlan`，所以其二进制文件的模块对所有 plan 每个 context 只加载一次）、`set.names()`、`set.manifest()`；对于 manifest 已加载的 plan 文件，用 `LaunchPlan::load_with_manifest(file, &manifest, pin)`。一个目录持有一个 plan（`plan.json`）或一个 plan set，不能两者都有：把其中一种导出到另一种的目录中会被拒绝（没有 `overwrite=True` 时）。带着另一个 graph 导出 set 会构建另一个 manifest，所以 set 中的每个 plan 都得到新的 digest，set 也是如此。

为什么用 set，而不用步骤重复的 plan：重复的步骤范围需要两个读取方都认同的循环携带规则（存活性、读写顺序、每次迭代的视图、tap 或 `run_step` 所指的步骤），而 tile 不完全相同的阶段（最后一个不完整的 tile 有不同的形状）并不是一个范围的重复；每个键一个 plan 让每个 plan 都保持为一个直线列表，其每次启动都在第一次启动之前检查，而 set 去掉了重复的东西，即二进制文件和 pin。

### 15.1 一个 set 的各 plan 共享的 arena

**一次一个** 地运行多个 plan 的 host（一个请求的阶段 A，然后 B，然后 C；数量可变的 tile plan）不需要每个 plan 一个 arena：这些 plan 可以共享一个。**这是 host 的选择，不属于任何文档**：每个 plan、它的 digest、索引及其 digest 都保持原样。

* **布局。** shared arena 的大小是共享它的各 plan（host 指定的 plan：一个 set 的全部 plan，或其中一部分）的 `max(arena_bytes)` 字节，分配一次，与任何 arena 一样按 256 对齐。**每个 plan 在其中保持自己的内存计划，从第一个字节开始**：plan `P` 的缓冲区 `b` 位于 `arena + offset(P, b)`，与在它自己的 arena 上完全一样，所以第 6 节的每条规则对每个 plan 都成立，每个 plan 都在固定地址上被 capture 进 CUDA graph（第 10 节）。不同 plan 的区域相互重叠：它们的中间张量、设备标量槽位和 cuBLASLt workspace 共享字节。所以 arena 持有每个 plan 的 workspace（位于该 plan 的偏移处，大小至少是该 plan 每个 library 步骤所声明的，第 7.3 节）：实际上是一个 workspace，即各 plan 中最大的那个，设备标量槽位也是如此。门禁中 transformer block 的三个阶段（`plan_model.STAGES`）的 arena 分别为 8 782 080、8 585 216 和 9 109 504 字节，共享一个 9 109 504 字节的 arena（三个 arena：26 476 800）。
* **为什么这是可靠的：顺序。** 使用同一个 arena 的两次执行不得在设备上重叠（第 9 节第 1 步），这是针对 arena 说的，无论其上运行哪些 plan：host 用一把锁和一个顺序事件（两个 host 都是如此）把 **shared arena 上的每次执行，无论属于它的哪个 plan**（一次运行、一次 capture、一次 host 拥有的 graph 的 replay、单独一个步骤）与其他每次执行排序，所以该 arena 的两个 plan 的执行无论来自哪些线程、在哪些 stream 上，都不能并发运行。plan A 的 graph 的一次 replay 与 plan B 的每次运行和 replay 一起排序。于是一次执行在 arena 中看到的是它之前那次执行留下的任何内容，而它除了 carried buffer 之外不需要其中任何东西（第 15.2 节，由顺序传递下去）：它在读取自己区域中的每个 arena 缓冲区之前先写它（中间张量，第 7.3 节；设备标量的槽位，在第一次启动之前，第 9 节第 4 步；workspace 是 GEMM 的 scratch）。调用方拥有的 graph（`torch.cuda.graph`）按第 10 节第 7 项所说在 arena 的顺序中占有一席之地，带有同样的注意事项；同一次 capture 中在两个 stream 上对该 arena 的两次执行被拒绝（第 5 项）。（只要该 arena 上任何 plan 的另一次执行可能覆写了 host 拥有的 graph 的设备标量，Rust host 就会在该 graph 的 replay 时重新写入它们。）
* **执行之后还留存什么：plan 区域中什么都不留；留下的是 carried buffer。** 一个 plan 生成、后一个 plan 读取的值，要么是前者的 **返回值**，即调用方内存中的一个外部张量（角色 `output`：Python 分配它或接收 `outputs=`，Rust 接收调用方的张量），作为后者的参数给出；要么是一个 **carried buffer**（第 15.2 节），arena 把它保存在各 plan 区域之后的一块自己的区域中，所以除了绑定到它的步骤之外，没有任何 plan 的任何缓冲区会写它。**会被覆写的东西会被拒绝，而从不会被破坏**：调用方的张量只要与 shared arena 在 *任何地方* 重叠（不仅是正在运行的 plan 所用的部分：下一个 plan 会写其余部分；carried 区域也算），就在启动任何东西之前以 kind `alias` 被拒绝，无论它是外部张量、tap 的张量还是单独一个步骤的张量（第 9 节）。
* **pin。** shared arena 的布局是各 plan 文档（它们的 `arena_bytes` 和偏移）的函数，这些由它们的 digest 覆盖，而它们的 digest 又由 set 的 digest 覆盖：**set 的 pin 覆盖 shared arena**，`plans.json` 中没有新东西。被改成另一种布局的 plan 文件在 arena 按它确定大小之前就被 pin 拒绝。golden 样例的 `expected.json` 包含每个 plan 的 `arena_bytes` 和 golden set 的 `shared_arena_bytes`，读取方要根据这些文件复现它们。

各 host。Python：`s = tilefish.runtime.share_arena([g1, g2, g3], mode="replay", plan="build/", pin=planset.digest)`（在 `jit` 中，各项是 graph 或 `tilefish.plan` 生成的 `Plan`），然后是 `s.run(g1, *args, outputs=, taps=)`、`s.capture(g2, *externals, devscalars=, outputs=)`、`s.bind_weights(g3, {...})`、`s.run_step(g1, step, buffers)`、`s.prepare()`；`s.arena_bytes`、`s.sizes`（每个 plan 的 `arena_bytes`）、`s.carried`（第 15.2 节）。与单个 plan 一样，Python 按布局保留 arena：同一设备上相同 plan、相同 `name=`（或都没有 `name=`）的每个 `SharedArena` 是同一个 arena，保留到 `clear_caches`；另一个名字就是另一个 arena（`share_arena([...], name="lane1")`：每个 stream 一个）；单独运行的 plan（`tilefish.run`）有自己的 arena。Rust：`let runs = set.instantiate(&stream, &["g1", "g2", "g3"])?`（一个 `SetRun`：`runs.run("g1")?` 是 shared arena 上的一个 `PlanRun`，`runs.arena()` 是 `SharedArena`），或者 `SharedArena::new(&stream, &[&p1, &p2])` 加上 `p1.instantiate_in(&arena, values)` / `instantiate_all_in(&arena)`，它们拒绝不是为之创建该 arena 的 plan（按 digest 判断）；每个 Rust `SharedArena`（以及每个 `SetRun`）都是一个自己的 arena（有多个 lane 的 host 每个 lane 创建一个）。`PlanSet::shared_arena_bytes(names)` 和 `SharedArena::bytes_for(plans)` 无需 GPU 即可给出大小。

### 15.2 carried buffer（tilefish 0.0.7，crate 0.0.5）

把一个中间张量交给下一阶段的阶段，不必把它变成由调用方分配的返回值：各 graph 把它声明为 **carried**，由 arena 保存它。

* **声明。** `@tilefish.graph(carried=("h",))` 指定 graph 的张量参数，可以是 `In`（plan 读取 `h`）或 `InOut`（它写 `h`，或原地更新它），不能是 `DevScalar`，也不能是 `inputs=`/`weights=` 中的输入或权重。名字是该缓冲区跨 plan 的身份：一个 arena 中具有 carried 参数 `h` 的每个 plan 都使用该 arena 中唯一的张量 `h`。在文档中，这样的参数是角色为 `carried` 的缓冲区，`offset` 为 `null`（与外部缓冲区一样），`bytes` 取其最大 bucket 的值（第 6 节）。它与其他参数一起按签名顺序列出；步骤可以读它（它算作已提供，第 7.3 节），也可以写它。文档中没有其他新东西：没有 carried 参数的 plan 与以往完全相同，包括 digest；早于该角色的读取方（已发布的 crate 0.0.4、tilefish 0.0.6）拒绝带有它的 plan。
* **写入方。** 当值是紧邻的前一个步骤生成的完整中间张量、且之后没有任何东西再读取它时，对 carried 参数的 `h[...] = value` 由生成该值的步骤写入（它的结果绑定到 `h`：没有中间张量也没有复制）；否则它是一个 copy 步骤（或一次 `tilefish.contiguous` 的启动）。`h[...] = tilefish.zeros_like(h)` 是对 `h` 的一次 memset（第 7.6 节）。原地调用（`accumulate(h, y)`，一个 `InOut` 操作数）在原处更新它。
* **布局**（`_plan.carried_layout`；Rust `CarriedLayout::of`）。一组 plan 的 arena（对于自有 arena，就是一个 plan）先是它们的区域（从 arena 的第一个字节开始，各 plan 自己的 `offset`，`base = max(arena_bytes)`），然后是这些 plan 中任意一个的每个 carried 名字各一块区域，按名字的码点顺序排列，每块长 `align_up(max bytes over the plans that name it, 256)`，位于 256 的倍数处。给同一名字两种 dtype 的两个 plan 被拒绝。布局只是这些文档的函数，所以 plan 或 set 的 pin 像覆盖 arena 的其余部分一样覆盖它（15.1），`plans.json` 中没有新东西。golden set `carried`（`expected.json`：`carried_base`、`carried`、`shared_arena_bytes`）在一个有自己区域的 plan 之后有两个名字；读取方根据这些文件复现它。
* **生命周期。** carried buffer 与它的 arena 存活得一样久（Python：直到 `clear_caches`；Rust：直到 arena 的最后一个句柄），并持有 arena 上最后一次写它的执行留下的内容。没有其他东西写它的字节（它不在任何 plan 的区域中；调用方的张量不可以与 arena 重叠）。自有的 arena（一个 plan 的 `tilefish.run`、Rust 的 `instantiate`）有自己的 carried 区域，所以不在同一个 shared arena 上的两个 plan 不共享 carried buffer，并且先读 carried buffer 的 plan 单独运行时被拒绝（除了它自己，没有任何东西能写过它的 arena 中的这个缓冲区）：它要在与写该缓冲区的 plan 共享的 arena 上运行。
* **各自持有什么，以及检查。** host 在持有 arena 的锁时（因此按执行运行的顺序）记录最后一次写每个 carried buffer 的执行所写的形状（在其 bucket 下），或者什么都不记录（没有执行写过它，或者写过它的执行在入队之后失败了）。先读后写 carried buffer 的 plan（它在步骤中第一次使用该缓冲区时，是一个 `in` 或 `inout` 指针、library 步骤的一个操作数，或 copy 的源）**从该形状绑定它的维度**，就像从外部张量的形状绑定一样（所以调用 `stage_b(h)` 时什么都不用给：它的 `T` 就是 `h` 被写入时的尺寸），并且除非该缓冲区恰好持有它要读的形状，否则 **在入队任何东西之前被拒绝**（kind `shape`；"no execution on this arena has written it"，或 "left the shape ..."）。一次执行之后，它写入的每个 carried buffer 都持有它的形状。一次运行、一次 host 拥有的 graph 的 replay（`CapturedPlan.replay`，Rust `CapturedPlan::replay`）以及 Rust 的 `check` 都会这样做；capture（`tilefish.capture`、`PlanRun::capture`）不运行任何东西，也不检查它们持有什么，但像运行一样绑定维度（所以只有 carried buffer 才能绑定其维度的 plan，要在该缓冲区被写入之后才能 capture）；调用方拥有的 graph（`torch.cuda.graph`）在被 capture 时检查和记录，而不是在它的 replay 时（tilefish 看不到这些 replay）：这种 graph 的某次 replay 若以不同于 arena 最后记录的执行的形状写 `h`，就会让记录出错，之后读取 `h` 的运行会绑定记录下的形状（把传递缓冲区的各阶段按顺序 capture 进同一个 graph，或者使用 `tilefish.capture`，它的 replay 会被检查和记录）。维度从 plan 先读的 carried buffer 绑定，而不从它只写的那些绑定：只写 `h` 的阶段（用 memset 重置）接收一个带有其维度的张量。单独一个步骤（`run_step`、`launch_step`）像对待任何缓冲区一样接收调用方为 carried buffer 给出的张量，并且不触及任何 carried 区域。
* **调用。** 一次运行给出除 carried 参数之外的每个参数（Python：按不含 carried 参数的签名顺序按位置给出，或按名字给出；给出 carried 参数会被拒绝："held by the plan's arena"）；`tilefish.capture` 和 `bind_weights` 同理。reference 模式没有 arena，它像对待其他参数一样把 carried 参数作为实参接收。参数全部是 carried 的 plan 在 torch 的当前设备上运行。

各 host。Python：`@tilefish.graph(carried=...)`、`Plan.carried`、`SharedArena.carried`（`{name: (offset, bytes, dtype)}`）、如 15.1 中的 `share_arena`、`tilefish.run`、`capture`、`bind_weights`、对 carried buffer 的 tap（`taps={"h": t}`）。Rust：`BufferRole::Carried`、`LaunchPlan::carried()`、`CarriedLayout`、`SharedArena::carried()`；`PlanRun::launch` 和 `CapturedPlan::replay` 检查并记录 carried buffer 持有什么。测试：第 14 节。
