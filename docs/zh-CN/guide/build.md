# 构建如何进行

[English](../../en/guide/build.md) | **简体中文**

构建就是 Python API，没有构建用的命令行：

```python
from tilefish.hosts import cpp, rust

p = tilefish.runtime.export(block, "build/")      # binaries, manifest.json and plan.json for the device's architecture (or arch="sm_100")
rust.emit_plan(p, "runner.rs")                     # a typed Rust runner of the pinned plan (rust.emit: launchers of a manifest)
cpp.emit_plan(p, "runner.hpp")                     # the same for C++ (cpp.emit)

(spec,) = tilefish.aot.compile_specs(gelu)         # step 4 for one extern: its compile inputs, one spec per bucket
k = tilefish.aot.build(spec, arch="sm_100")        # one cubin
```

1. 把 graph（计算图）追踪成类型化的 IR 并检查：形状、dtype、整除性、别名，以及每个算子支持哪些架构。
2. 运行重写规则（[`rewrites.md`](rewrites.md)），确定最终的 kernel（核函数）列表。
3. 把每个 `@kernel` 下沉（lowering）为 Triton 源码，写到磁盘上供审阅。
4. 从类型推导出 Triton 的编译输入：指针的 dtype 和 16 字节对齐，由静态维度得到的 `constexpr` 值，由 `divisible_by` 得到的整除性提示，以及分桶维度每个 bucket（分桶）一个二进制文件。提前编译，每个架构一个 cubin。
5. 写出 manifest（清单）：符号、按顺序排列的参数 ABI、grid 规则、block 大小、共享内存、启动属性、工具链版本，以及机器码的 digest（摘要）。
6. 由 `tilefish.hosts` 从 manifest 和类型生成各个 host（宿主）：类型化的启动器和 plan（计划）运行器，C++ 的一个头文件和 Rust crate 的一个模块（Python 自己启动 manifest，即 `replay`）。每个用来选中某个二进制文件的前置条件（对齐、整除性、静态大小）都会变成生成的启动器中的一项检查，所以任何 host 都不能在一个二进制文件编译时未曾假定的条件下运行它。

library operator（库算子）在同一种 manifest 条目背后放的是另一种代码对象：一个已 pin（固定）的 cuBLASLt 算法，连同它的 workspace（工作区）大小一起记录，并原样回放；或者一个已 pin 的 cuDNN 卷积（[`operators.md`](operators.md)）。CuTe DSL 的提前编译导出是一个预构建 cubin 的 extern（外部 kernel；[`../cute.md`](../cute.md)）。

Python 这边有三种模式：

| 模式 | 运行的是什么 | 用途 |
|---|---|---|
| `jit` | 当前的源码，通过它的编译规格（即构建所走的路径）在内存中构建，并通过 CUDA driver 启动 | 迭代开发一个 kernel |
| `reference` | 同样的函数体，用 torch 解释执行 | 作为对照（oracle），在容差内比较 |
| `replay` | manifest 目录中预构建的 cubin，对照其 digest 检查，通过 CUDA driver 启动；它需要一个信任锚：manifest 已 pin 的 digest，或显式的 `trust_directory=True` | C++ 和 Rust host 运行的那份完全相同的机器码 |

```python
y = tilefish.run(residual_add_rmsnorm, x, residual, w)                        # jit
m = tilefish.runtime.export(residual_add_rmsnorm, "build/")                    # cubins and manifest.json
y = tilefish.run(residual_add_rmsnorm, x, residual, w, mode="replay", manifest="build/", pin=m.digest)           # a manifest whose digest you hold
y = tilefish.run(residual_add_rmsnorm, x, residual, w, mode="replay", manifest="build/", trust_directory=True)  # whatever the directory holds
```

没有这两者之一（`pin=` 或 `trust_directory=True`）时，`replay` 拒绝运行。这是 0.0.x 中一个有意为之的不兼容变更：直到 0.0.2，replay（回放）默认信任目录，而目录里的 digest 就放在它们所描述的文件旁边，所以谁能写这个目录，谁就能让它运行任意二进制文件。

pin 针对的是一个导出的产物，而不是一份配方：manifest 的 digest 逐字节覆盖 cubin 文件。导出是可复现的。cubin 的调试段用模块路径来指称每个源文件（`kernels/rms.py`、`tilefish_generated/softmax.py`、`triton/language/standard.py`），而不是用检出目录、虚拟环境或临时目录；生成的 kernel 的源码写入一个以其 sha256 寻址的文件。所以同一个 graph 用同样的工具链（Triton、`ptxas`、cuBLASLt、cuDNN）在同类设备上再导出一次，无论是从另一个目录、在另一个进程里还是在另一台机器上，都会得到同样的文件、同样的 manifest digest 和同样的 plan digest，记录下来的 pin 可以用来检查一次重新构建。tilefish 0.0.6 及更早版本导出的 manifest 会写明构建时所在的目录；它照旧能加载和回放，只是它的 digest 不等于同一个 graph 新导出所得到的 digest。pin 你导出的东西，并交付那个目录。如果 profiler 需要 cubin 里的绝对源码路径（而不是对照某个源码目录去解析 `kernels/rms.py`），就用 `TILEFISH_DEBUG_PATHS=absolute` 构建。它默认关闭，并且会破坏可复现性：那次导出的 cubin、`sha256` 和 digest 会写明构建地点（各个 `sha256_code` 不变），它的二进制文件也单独缓存。

参数是 torch 张量，按追踪时匹配一次调用的方式与签名匹配：输入和 `InOut` 张量要传入（`InOut` 张量会被原地更新），`Out` 张量会被分配并返回。启动之前会检查二进制文件的每个前置条件（dtype、静态大小、`divisible_by`、16 字节对齐的指针、连续性），分桶的维度必须恰好等于某个 bucket：没有填充。

`jit` 和 `replay` 的启动方式相同，所以运行的是同样的机器码：`jit` 构建出的 cubin 与 manifest 中那个 cubin 有同样的 `sha256_code`。`replay` 会拒绝 digest 与 manifest 不符的 cubin、与算子的编译规格要求不符的 manifest 条目，以及过时的二进制文件：一个以另一个函数体、但同样的名字和签名重新声明的 kernel，不会回放它旧的二进制文件。

**二进制文件的身份，就是它的 kernel 编译成的 Triton IR。** Triton 编译出哪个 kernel，取决于其前端运行时 Python 进程所持有的一切：函数的文本和它调用的 jit 函数的文本，它读取的模块常量（`cfg.SCALE`、`getattr(cfg, "SCALE")`、`getattr(cfg, "SC" + "ALE")`、类属性 `cfg.Inner.SCALE`），闭包变量，读取环境的 `triton.constexpr_function`，`repr` 隐藏了自身状态的对象所构成的 `tl.constexpr`，隔了两个模块的辅助函数，与标准库模块同名的用户模块。五轮审查中，每一种*查找*这些东西的方式（对 kernel 触及的文件做 digest、模块属性、闭包、监视导入）都被找出一条它没有跟踪到的路径，而每一条都导致构建缓存，或一次带 pin 回放的导出，提供了一个过时的二进制文件。所以现在什么都不去查找。Triton 的前端会把它们每一个都折叠进它构建的 IR，而被哈希的正是这个 IR（`tilefish._ir`）：`ASTSource.make_ir` 构建的 TTIR，用 `Module.str_nodebug()` 打印（没有位置信息，没有值的名字）。kernel 所在的位置（路径、行号）、注释、空行和局部变量名都不在其中，所以改动它们都不算修改；操作、类型、常量和被调用的函数则在其中，改动它们就是修改。进程持有什么，就哈希什么，因为被编译的就是它。每次构建的 digest 在 `BuiltKernel.ir_digest` 中，在 manifest 条目中则是 `source_digest`，其 `source_digest_kind` 为 `"tilefish.triton-ir.v1"`（这是 schema（模式）2 已有的字段，所以 manifest 的读取方照旧读取它；下沉生成的 kernel 还有它的 `source_sha256`，读它不需要 Triton）。Gluon kernel 的 IR 是 Gluon 前端（`GluonASTSource`）构建的 TTGIR，其中包含它的布局和 `num_warps`，其 kind 为 `"tilefish.gluon-ir.v1"`。当 Triton 可以导入、且正是构建该条目的那个版本时，`replay` 会用进程持有的 kernel 重新计算 digest，不一致就作为过时而拒绝，不论有没有 pin（没有 Triton，或者 Triton 是另一个版本、其 IR 可能打印得不同时，只有 pin 能为手写的 kernel 担保；版本会先比较，所以前端不同的 Triton 永远不会被要求运行它）。在 B300 上，前端每个 kernel 耗时 2 到 8 ms（不编译，不用 GPU），每次构建、每次 `jit` 调用和每次 `replay` 调用各运行一次。它与编译在同一把锁下运行，所以在任何线程中都不会有两个前端（包括编译自身的那个）同时运行。一次导出（以及 plan 的 `prepare`）对每个不同的输入运行一次前端，而不是每个 bucket 一次：同一个 `by_bucket=` 配置的各个 bucket 是同一个 IR、同一个 digest，也是构建缓存中的同一个二进制文件。一次运行保留下来的是它的 digest，而不是它的 IR，模块会被释放（否则 Triton 的绑定会在进程的整个生命周期里保留它构建的每个模块，大约是其文本的两倍，每次前端运行和每次编译都如此）。导出一个 IR 很大的 Gluon kernel 的 132 个 bucket、三种配置（`python/tests/irmem_export.py`，在 B300 上），以前耗时 77 s，常驻内存峰值 1.27 GB；现在耗时 9 s，峰值 0.40 GB（缓存已热时为 2.7 s 和 0.25 GB；追踪时，以前 2.66 GB，现在 0.43 GB），写出的字节相同。plan set（plan 集）读回时对照的是导出写出的那一个 manifest，而不是每个 plan 一份 manifest 及其二进制文件的副本：一个包含 1319 个 plan、涉及 147 个二进制文件和 1199 个 GEMM（通用矩阵乘）pin 的 plan set，以前峰值 54 GB，现在峰值 1.3 GB。

所以没有重新加载的修改并不是特例。在导入了该模块的进程里，`export` 编译并标记的是那个进程所持有的东西；一个读取了修改后文件的新进程会算出另一个 digest，并拒绝这个 manifest。不使用 Triton 自己的 `cache_key`，构建也从不读取 Triton 的磁盘缓存。被移除的有：kernel 触及的文件的 digest；导入监视（不再包装 `SourceFileLoader.exec_module`，`TILEFISH_IMPORT_WATCH` 也已去掉）；模块属性的快照，以及对被重新赋值的属性的拒绝（程序所赋的值就在 IR 里）；以及对 kernel 的值和闭包的遍历，连同对没有稳定 `repr` 的值的拒绝。保留下来的有：对*下沉生成的* kernel，检查它的模块文件具有其规格中的 `source_sha256`，并且进程持有的函数正是从那些行导入的（IR digest 说明的是进程持有什么，而不是说明它就是规格所声称的源码）；检查 `build(spec, fn=)` 编译的是规格指定的那个函数而不是别的；以及拒绝一切让 Triton 编译出 IR 之外的东西的做法：它的 kernel 覆盖（`TRITON_KERNEL_OVERRIDE`）和编译选项 `ir_override` 会用一个文件替换某个阶段的 IR（这样二进制文件就不再出自该 IR），而启用的插桩插件（`CUDABackend.instrumentation`）会修改编译 pass。

构建会从头编译（B300 上每个二进制文件 25 到 120 ms），除非 tilefish 自己的缓存里已经有：`~/.cache/tilefish`，或 `$TILEFISH_CACHE_DIR`（设为空值即关闭；`aot.build(..., cache=False)` 在单次构建中跳过它）。它的键包括：缓存配方（一个数字，每当 tilefish 改变进入二进制文件的内容时就递增）和 tilefish 版本、IR digest、编译选项、架构、Triton 和 `ptxas` 的版本，以及会改变代码生成的 Triton 设置：`TRITON_`、`MLIR_`、`LLVM_`、`NVPTX_`、`AMDGCN_`、`DISABLE_`、`ENABLE_`、`ALLOW_` 和 `USE_IR_LOC` 这些前缀族中的每个环境变量，已知不会改变代码生成的除外（缓存和 dump 目录、工具路径、调试开关），这比 Triton 列为会使缓存失效的 30 个变量要多（Triton 漏掉了 `TRITON_LIBDEVICE_PATH`、`TRITON_MOCK_PTX_VERSION` 和 `TRITON_KERNEL_OVERRIDE` 等）；`triton.knobs` 中每个代码生成旋钮的有效值；以及设置了路径时 libdevice 文件的 sha256。指定 Triton 所读文件的设置，以文件的 sha256 而不是路径作为键，所以在同名下被改写的文件就是另一个二进制文件：编译选项 `extern_libs`、`TRITON_LIBDEVICE_PATH` 和 `LLVM_PASS_PLUGIN_PATH`（文件不可读时拒绝构建）；Triton 3.5.1 中其余指定路径的设置，要么是工具（`ptxas`，其版本在键里），要么是输出，要么属于启动器，`_cache.py` 列出了它们。每次读取都会校验：重新计算 cubin 的 `sha256` 和 `sha256_code`（并检查 PTX、入口点及其 ABI、选项、IR digest 和版本），与写入时不一致的条目会产生一个警告并重新构建，绝不会被使用；缓存不可读或不可写也永远不会让构建失败，在 `python -W error` 下同样如此（此时警告会记录在 `tilefish.cache` 日志上，而不是抛出）。条目先写入临时文件再重命名，所以相互竞争的构建不会把一个条目写坏。一次查找就是运行一次前端加一次读取：3 到 6 ms。缓存不会淘汰任何条目；它不能防御能写这个目录的人（要 pin 的是 manifest）；对于保留了原版版本字符串的打过补丁的 Triton，它的分辨能力并不比版本号更强。同一个目录还在 `generated/<sha256>/tilefish_generated/` 下存放每个下沉生成的 kernel 和生成的算子的 Triton 源码（Triton 从文件读取 kernel 的源码）：原子写入，只有当它保存的恰好是那份源码时才使用；目录不可写时，改为保存在进程私有的临时目录中（并给出一次警告）。它放在哪里不会影响二进制文件。

`replay` 还会拒绝 cubin 和设备不可能具有的 `shared` 内存（driver 给出的函数静态共享内存加上 manifest 的动态大小，必须在设备的 opt-in 上限之内）、与 ABI 不符的参数列表，以及超出设备限制的 grid。manifest 按严格 JSON 读取（没有重复键，没有 `NaN`，没有超出浮点数范围的数字），文件先写入临时文件再重命名，要写文件的位置上如果是符号链接则拒绝。

**IEEE f32 点积不会被构建成 TF32 MMA。** tensor core 没有 f32 乘法：f32 MMA 会把输入舍入到 TF32，即尾数 23 位中的 10 位。然而 Triton 3.5.1 在 sm_100 和 sm_103 上会把某些 f32 操作数的 `tl.dot(a, b, input_precision="ieee")` 编译成 `tcgen05.mma ... kind::tf32`：当一个操作数来自窄于 32 位的 load（转换为 f32 的 f16 或 bf16 tile）或来自 `tl.join`，且 tile 满足 `M % 64 == 0`、`N % 16 == 0` 并有 4 或 8 个 warp（线程束）时，它的 MMAv5 路径会跳过精度检查。在 B300 上实测，`a = 1`（f16），`b = 1 + 2**-12`，16 项：每个输出都是 16 而不是 `16 + 2**-8`；同一个 kernel 在 `M = 32` 时，或在 sm_90 和 sm_120 上，是精确（exact）的。所以每次构建（`jit`、`export`、plan、缓存命中）都会拒绝这样的 kernel：它的 IR 要求 IEEE f32 点积，而它的二进制文件里有由这种点积生成的 TF32 tensor-core MMA（`tilefish._tf32guard`）。这项检查读取的是编译器生成的东西，而不是照抄 Triton 的规则：前端构建的 TTIR 中的点积及其精度，以及 PTX 中的 TF32 MMA 指令（`kind::tf32` 的 `tcgen05.mma`，`.tf32` 操作数的 `wgmma` 和 `mma.sync`）。如果一个 kernel 的 f32 点积全都要求 IEEE，那么只要它的二进制文件里有任何 TF32 MMA 就会被拒绝；如果它还有允许 TF32 的点积，就逐个点积判断：按源码位置把它 TTGIR 中的每个 `ttng.tc_gen5_mma` 追溯到生成它的 TTIR 点积，无法追溯的 MMA 也会被拒绝。写明 `input_precision="tf32"` 或 `"tf32x3"` 的点积要求的就是 TF32，永远不会被拒绝。错误信息会指出该点积和修复办法：为这个点积使用少于 64 行的 tile，把乘积写成 `tl.fma` 的循环（不用 `tl.dot`），或者在 kernel 可以用 TF32 计算的地方写 `input_precision="tf32"`。Gluon kernel 也以同样方式检查（在 Triton 3.5.1 中 Gluon 没有能编译的 IEEE f32 点积：`warpgroup_mma` 会拒绝它，而 f32 操作数的 `tcgen05_mma` 是 kernel 自己选择的 TF32）；下沉生成的 `@kernel` 没有点积。预构建的 PTX 或 cubin 没有 IR 来说明它要求了什么，因此不检查；在这项检查出现之前导出的 manifest 也不检查（请重新导出）。

**信任。** 回放需要一个信任锚，`pin=` 或 `trust_directory=True`，没有就抛出异常。manifest 目录里的 digest 就放在它们描述的文件旁边，所以谁能写这个目录，谁就能把两者都改写。`export` 返回 manifest 及其 `digest`（覆盖每个条目的 sha256），并检查它写出的目录具有这个 digest；把它保存在你信任的地方，作为 `pin=` 传入，就能拒绝被改动过的目录。`trust_directory=True` 是一个明确的声明：目录里有什么都可以（比如你刚刚写出的目录）；它不能与 `pin=` 同时使用。pin 担保的是 manifest，不担保已知不一致的 IR digest，也不担保非零但对 kernel 来说太小的 `shared`（除了 manifest，没有别的东西说明它需要多少：Triton kernel 的 cubin 不记录其动态共享内存，这在 `python/tests/test_gpu_shared_size.py` 中实测过；导出写入的是 Triton 给出的值，pin 覆盖它，而 `trust_directory=True` 连同 cubin 一起信任它）。Triton 自己那个按参数值特化的 JIT 启动器不是一种模式。`jit` 和 `replay` 都不需要 C 编译器，生成的 kernel 的 `replay` 既不需要 Triton 也不需要 `ptxas`。

CI 在 GPU 上（标记为 `gpu` 的测试）断言：`jit` 构建的 cubin 具有 manifest 中那个 cubin 的 digest，`jit` 和 `replay` 逐字节一致，`replay` 与 `reference` 的差异在容差之内。容差按输入分布和 dtype 分别测量（输入输出为 f32、bf16 和 f16；正态、均匀、重尾、稀疏离群值、各种数量级的值），并在 `python/tests/test_gpu_reference.py` 中列成表：ULP 计数只有在结果远离零时才有意义（对均值很大的数据做 layernorm，输出很小），所以 bf16 输出的界限是最大参考值的半个 bf16 步长加上允许不同的元素比例，f32 softmax 的界限是绝对误差 5e-7，gelu 是 3e-5。在 `replay` 中通过的实验，逐个 kernel 来看，就是生产环境运行的东西。
