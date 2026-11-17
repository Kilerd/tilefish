# SASS 溯源：trace、`tilefish-explain` 与 SASS 断言

[English](../en/explain.md) | **简体中文**

每个 kernel（核函数）都要经过 DSL / Triton / Gluon → TTIR → TTGIR → LLVM IR → PTX → `ptxas` → SASS。过去要把一个热点或一处比特差异沿这条链追溯回去，就得手动导出每一层再逐层 diff；而采样式 profiler 会把异步 `wgmma` / `tcgen05` 代码的时间算到等待它的位置上，而不是算到真正干活的地方。tilefish 让每个 kernel 的 SASS 都可解释、可检查：

1. **trace**（追踪记录）：一次导出的 trace 保留每个二进制文件的每一层，放在已 pin（固定）的产物之外（第 1 节）。
2. **`tilefish-explain`** 为一个 kernel、一个 plan（计划）步骤或一个融合区域的 SASS 标注产生每条指令的源码行和 graph（计算图）算子，汇总指令构成，按源码行计数，并对比两个二进制文件（第 2 节）。
3. **SASS 断言**是声明的一部分，和 TF32 防护、收缩（contraction）检查一样，在每次构建时检查（第 4 节）。

代码在 `tilefish._trace`、`tilefish.explain` 和 `tilefish._sass`。一个 plan 的哪些步骤可以合成一个 kernel、这样能省多少，由 `tilefish-explain --advise` 给出（`docs/zh-CN/advise.md`）；kernel 内部发生了什么、按它声明的阶段边界计时，由 `tilefish-explain --probes` 给出（`docs/zh-CN/instrument.md`）。

## 1. trace

```python
runtime.export(ops, "build/out", trace=True)
```

`trace=True`（或者在没有给出 `trace` 时，环境里的 `TILEFISH_TRACE=1`：用于由你不修改的脚本、CI 作业或模型自带的导出脚本运行的导出；`0` 或未设置表示不生成 trace，其他任何值都会被拒绝）会写出 `build/out/trace/`。参数才是 API；这个环境变量之所以存在，是因为导出往往不是你的代码。kernel 和 extern（外部 kernel）、graph 和 plan set（plan 集）都以同样的方式生成 trace。

```
trace/
  index.json                         tilefish.trace.v1: every file and its sha256, the manifest digest, the binaries' sha256, trace_digest
  <stem>.<arch>/
    source/<module-relative name>    the source files the line information names, as the debug information names them
    frontend.mlir                    the IR the frontend built, with its locations (a Gluon kernel: TTGIR)
    ttir.mlir  ttgir.mlir  llir.ll   Triton's stages (a Gluon kernel has no TTIR)
    ptx.ptx                          the PTX (a prebuilt PTX extern: its PTX)
    sass.txt                         nvdisasm -c -g -gi: the SASS with its line information
    lines.json                       the line table: per instruction its address, source file and line, and PTX line (nvdisasm -gp)
    nodes.json                       a lowered kernel or a fused region: what each line of its generated source was written for
  graphs/<graph>.json                a graph: the steps of each bucket of its plan, the ops of the graph body each fused region took (and its
                                     template), and why the regions are where they are ("boundaries", docs/en/advise.md section 4)
  graphs/source/<name>               the files of the graph body those ops were written on
```

* **源码**按二进制文件调试信息给出的模块相对名称存放（`kernels/mm.py`、`tilefish_generated/rmsnorm.py`、`triton/language/standard.py`：`tilefish._source.source_name`），因此 trace 可以在任何目录、任何机器上读取。只复制本进程持有的文件：生成的源码、kernel 的 jit 函数所在的文件，以及本进程已加载、模块相对名称正是该名称的模块的 `__file__`。不做任何搜索（不搜 `sys.path`，也不搜工作目录）；预构建二进制文件的 PTX 里提到的文件属于另一次构建：一个都不读。不是模块相对的名称（`TILEFISH_DEBUG_PATHS=absolute`）也不复制；索引会记下哪些没有复制。
* **trace 包含源码。** 不要把 `trace/` 随导出一起发布：它是给构建导出的人看的，不是给运行它的 host（宿主）用的。
* **`nodes.json`** 把生成源码的一行映射到 `{"node": <index in the kernel's graph>, "op": "mul", "loc": "model.py:17"}`（`loc`：写下该算子的 `@tilefish.kernel` 函数体中的行，对融合区域则是 graph 函数体中的行），把不属于任何算子的代码映射到 `{"what": "load x"}`，把辅助函数的行（`mul.rn.f32`、`cvt.rn.bf16.f32` 的内联 PTX）映射到调用它的算子（`"helper": "_tf_mul_rn_f32"`）。融合区域的条目带有 `origin`，即该算子在区域中的位置；`graphs/<graph>.json` 按区域给出被 trace 的 graph 在每个位置上的算子（它的序号，与融合报告的编号一致，它的算子和它的行），因此两个 graph 共享的区域会按各自 graph 的说法来解释。
* **SASS 从哪里来**：索引条目的 `"sass"` 说明是哪个 `nvdisasm`（及其版本）读取的，以及它是二进制文件自身的 SASS，还是其 PTX 为所属架构族的基础架构汇编出的 SASS（`"proxy": true`，以及 `"matches"`：这次重新汇编的寄存器、栈帧和代码大小是否与二进制文件一致，见第 5 节）；`"sass_checked"` 说明构建是如何判定声明中的 SASS 断言的。
* **`graphs/<graph>.json`** 记录它所描述的 plan 及其 manifest（清单）的 digest（摘要）；属于另一个 plan 的这类文件会被 `tilefish-explain` 忽略（之后不带 trace 再次导出这个 graph 时，旧文件会留在那里）。

**在已 pin 的产物之外。** 没有任何 manifest、plan 或 plan set 引用 trace，也没有任何 digest 覆盖它，所以无论带不带 trace，导出写出的 manifest 和 plan 都逐字节一致（已在整个 GPU 门禁上检查：第 6 节）。host 从不读取它。构建缓存保存的是二进制文件而不是它们的 IR，所以带 trace 的导出会编译每一个 Triton kernel（它会存入缓存条目，而条目的字节相同：构建是可复现的）。导出到一个已有 trace 的目录时会合并索引：只要 manifest 仍包含另一个二进制文件（同一个 sha256），该二进制文件的条目就保留。

索引有自己的 digest `trace_digest`（对其规范 JSON 取 sha256，标签为 `tilefish.trace-digest.v1`），并记录每个文件的 sha256。索引或文件与记录不符的 trace 会被 `tilefish-explain` 拒绝；manifest 已不再包含的二进制文件的条目会被忽略。

## 2. `tilefish-explain`

```console
tilefish-explain DIR KERNEL                     # a stem of the manifest, or an op name with one binary
tilefish-explain DIR --graph G --step N         # step N of G's plan (--bucket D=96 when the plan has several buckets)
tilefish-explain DIR --graph G --region R       # a fused region: its kernel's name (fused_...) or its index in G.fusion.json
  --arch sm_103                                 # when the binary is built for several architectures
  --view annotate | summary | lines             # (annotate by default)
  --ptx                                         # annotate: the PTX line of each instruction
  --diff OTHER_DIR | --diff-arch sm_120         # the same kernel in another export, or for another architecture
  --source-root PATH                            # where module-relative names are found after the trace's copies (repeatable; default .)
  --json                                        # summary and lines as JSON
```

`python -m tilefish.explain` 与之相同；`tilefish.explain.load(directory, kernel, graph=, step=, bucket=, region=, arch=, source_roots=)` 返回一个 `Explained`，带有 `annotate()`、`summary()`、`lines()`、`summary_text()`、`lines_text()`；另有 `tilefish.explain.diff(a, b)`。

**annotate** 打印 kernel 函数的 SASS，每一段连续的指令放在其行号信息所指的源码行之下，并附上该行的文本：

```
fused_370fc65537cf on sm_103 (fused): its SASS, read by nvdisasm 13.5.37
  step 0 of the plan of 'model' at {'R': 64}: launches 'fused_370fc65537cf'; fused region 0 of the ops #0 cast (model.py:16), #1 cast (model.py:16),
  #2 mul (model.py:16), #3 add (model.py:17), #4 cast (model.py:17)
...
tilefish_generated/fused_370fc65537cf.py:64  v2 = v0.to(tl.float32)    [node 0 cast; graph op #0 cast at model.py:16: h = x.to(f32) * w.to(f32)]
    /*0260*/ IMAD.U32 R11, R11, 0x10000, RZ ;  [integer]  ptx 84
    /*0270*/ SHF.L.U32 R12, R12, 0x10, RZ ;  [integer]  ptx 79
...
tilefish_generated/fused_370fc65537cf.py:18  [a, b],    [node 2 mul (helper _tf_mul_rn_f32); graph op #2 mul at model.py:16: h = x.to(f32) * w.to(f32)]
    /*0300*/ FMUL R12, R12, R17 ;  [float]  ptx 101
    /*0310*/ FMUL R4, R15, R4 ;  [float]  ptx 107
```

（乘法是辅助函数里的内联 PTX `mul.rn.f32`，所以它的行是辅助函数的行，trace 把这一行归到调用它的算子上。）

对 Triton 或 Gluon kernel，这一行是 kernel 自己的（`mm_kernels.py:29  acc = tl.dot(a, b, acc)` 位于它的 `UTCHMMA` 指令之上）；对下沉（lowering）生成的 kernel，是生成源码的行，然后是这段代码所对应的 kernel graph 中的算子，以及函数体中的行；对融合区域，则是 graph 函数体中的算子。plan 步骤会给出该步骤、它的 bucket（分桶）和它的条目；区域会列出它包含的算子。

**summary** 按类别（第 3 节）统计指令数，列出 SASS MMA（操作码和形状；对 `UTC*MMA`，在反汇编列表把指令描述符设为常量的地方，给出描述符所说的 kind、类型和形状）、PTX MMA（其操作码写明了 kind）、TF32 MMA、local memory 的加载和存储（以及其中有多少被 nvdisasm 13.x 标记为 spill（寄存器溢出））、TMA、barrier 和 fence，并从 ELF 文件中读出每线程寄存器数、栈帧、静态和动态共享内存、命名 barrier，再给出 occupancy（占用率）的估计（每个 SM（流式多处理器）上的 block 数和 warp（线程束）数，以及限制它们的因素：根据这些属性，按 CUDA occupancy 计算器 `cuda_occupancy.h` 的规则计算，carveout 取最大的共享内存：寄存器按每 warp 以 256 为单位在 SM 的 4 个子分区内分配，共享内存按每 block 以 128 字节为单位分配，再加上该架构的常驻 block 数和 block barrier 数上限，sm_120 为 24 个 block，sm_90 到 sm_103 为 32 个。这不是驱动给出的数字，驱动的数字可能因为属性中看不出的原因而更低：在 sm_103 上实测，无论属性允许多少，驱动都只给由 `tcgen05` MMA 组成的 kernel 每个 SM 一个 block）。

**lines** 按列统计每个源码行的指令数：`mma`、`gmem`（global 加载和存储）、`smem`、`tma`、`tmem`（tensor memory）、`local`、`atom`、`sync`（barrier 和 fence）、`alu`（其余）。

**diff** 比较同一个 kernel 的两个二进制文件：summary（指令、寄存器、类别、MMA、PTX MMA、spill、occupancy），以及指令有差异的源码行，变化最大的排在最前。比较的可以是两次导出（`--diff`：另一套工具链、另一个声明），也可以是同一次导出的两个架构（`--diff-arch`）。

没有 trace 时，`tilefish-explain` 反汇编 manifest 中的 cubin（需要一个能解码它的 `nvdisasm`：没有 PTX 可供为架构族的基础架构汇编），并说明这一点；此时生成源码的行不会映射到算子。有 trace 时，它在任何地方都能运行，不需要任何 CUDA 工具。

**行号信息能说明什么。** 一条指令的行是 `ptxas` 记录的行，而 `ptxas` 会调度指令：sm_90 上一条异步 `wgmma` 可能落在等待它的那条存储所在的行下，地址计算可能落在第一次需要它的行下。Triton 自己的函数（`tl.max`、`tl.sum`）落在它在 `triton/language/standard.py` 中的行下：Triton 3.5.1 的 PTX 没有 `inlined_at`，所以不知道调用点在哪里。下沉生成的 kernel 的辅助函数归到调用它们的算子上（见上文）。

## 3. 指令类别

每条指令按操作码归入一个类别（`tilefish._sass.CLASS_RULES`；类别名是 `tilefish.sass(forbid=...)` 接受的一个词）：

| 类别 | 操作码 |
|---|---|
| `mma` | `HMMA`、`IMMA`、`QMMA`、`OMMA`、`DMMA`（`mma.sync`），`HGMMA`、`IGMMA`、`QGMMA`（`wgmma`），`UTCHMMA`、`UTCQMMA`、`UTCIMMA`、`UTCOMMA`（`tcgen05.mma`） |
| `tma` | `UTMALDG`、`UTMASTG`、`UTMAPF`、`UTMAREDG`、`UTMACCTL`、`UBLKCP`、`UBLKRED`、`UBLKPF` |
| `tensor_memory` | `LDTM`、`STTM`、`UTCCP`、`UTCATOMSWS`、`UTCSHIFT`（`tcgen05.ld/st/cp/alloc/shift`） |
| `local` | `LDL`、`STL`：spill 和局部数组 |
| `global_load`、`global_store` | `LDG`、`LDGSTS`、`LD`；`STG`、`ST` |
| `shared_load`、`shared_store` | `LDS`、`LDSM`；`STS`、`STSM` |
| `atomic`、`shared_atomic` | `ATOM`、`ATOMG`、`RED`、`REDG`；`ATOMS`（tcgen05 kernel 分配 tensor memory 时会用到 shared atomic） |
| `barrier`、`fence` | `BAR`、`SYNCS`（mbarrier）、`WARPGROUP`、`WARPSYNC`、`UTCBAR`（`tcgen05.commit`）、`DEPBAR`、cluster（线程块簇）barrier；`MEMBAR`、`FENCE`、`ERRBAR`、`CCTL` |
| `constant`、`fp64`、`mufu`、`convert`、`shuffle`、`control`、`special`、`float`、`uniform`、`integer` | 其余 |

各架构的 MMA，以录制的 kernel 所呈现的为准（Triton 3.5.1，ptxas 12.9）：

| 架构 | PTX | SASS |
|---|---|---|
| sm_90 | `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16` | `HGMMA.64x64x16.F32.BF16`、`HGMMA.64x64x8.F32.TF32` |
| sm_100、sm_103 | `tcgen05.mma.cta_group::1.kind::f16`、`...kind::tf32` | 两者都是 `UTCHMMA`：kind 在指令描述符里（`idesc[URn]`） |
| sm_120 | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | `HMMA.16816.F32.BF16`、`HMMA.1688.F32.TF32` |

## 4. SASS 断言

```python
@tilefish.extern(triton=mm_kernel, num_warps=8, grid=..., sass=tilefish.sass(
    no_tf32=True,                                  # no TF32 tensor-core MMA
    mma=["tcgen05.mma kind::f16", "UTCHMMA"],      # each used by at least one instruction
    no_spills=True,                                # no local memory: no stack frame, no LDL/STL
    max_registers=168,                             # registers per thread, from the ELF file
    forbid=["atomic", "fp64", "ATOMG"],            # classes (section 3) or SASS opcode patterns
    by_arch={"sm_120": tilefish.sass(mma="HMMA.16816.F32", no_spills=False)},
))
def mm(...): ...
```

接受 `sass=` 的有 `@tilefish.kernel`、`@tilefish.extern`（Triton、Gluon 和预构建二进制文件；带 `by_arch=` 时对每个实现都生效）和 `tilefish.fusion(sass=...)`（graph 的每个区域）。tilefish 自己写的生成算子（`contiguous`、`im2col`、`col2im1d`）不接受：它们的代码是 tilefish 的，由 tilefish 的测试覆盖。

* **匹配模式。** 小写的是 PTX 匹配模式：它的 token（按点和空格切分）按顺序出现在一个 PTX MMA 操作码中，前两个是操作码本身的（`"tcgen05.mma kind::f16"`、`"wgmma.mma_async bf16"`、`"mma.sync m16n8k16 bf16"`）。大写的是 SASS 操作码匹配模式：操作码相同，且匹配模式中的修饰符按顺序出现在指令的修饰符中（`"HMMA.16816.F32"`、`"HGMMA.F32.BF16"`、`"UTCHMMA"`、`"LDL.64"`）。在 sm_100 及更新的架构上，用 PTX 写出 tcgen05 的 kind。
* **`no_tf32`** 读取 PTX（`tcgen05.mma`、`wgmma`、`wmma.mma` 和 `mma.sync` 的 `kind::tf32`、`.tf32`），并在 SASS 反正要读、或二进制文件没有 PTX（预构建 cubin）时读取 SASS（`.TF32`，或描述符写明 `kind::tf32` 的 `UTCHMMA`）。在没有 PTX 的 cubin 中，如果 SASS 没有把某条 `UTCHMMA` 的描述符设为常量，它可能是 `kind::f16` 也可能是 `kind::tf32`，无从分辨：这种情况会被拒绝，并给出说明这一点的消息。
* **`no_spills`** 由 ELF 文件判定：二进制文件不记录任何栈帧（`EIATTR_FRAME_SIZE` 和栈大小），这样无论由哪条指令访问，都能抓到 spill 和局部数组（通用地址上的 `LD`/`ST` 与 `LDL`/`STL` 一样），不需要反汇编。在能读取 SASS 的地方，违反时会列出 local memory 指令（nvdisasm 13.x 把 spill 标为 `SpillRefill`）；在 SASS 反正要读的地方，`LDL`/`STL` 也会导致拒绝。
* **`max_registers`** 读取二进制文件自身的 ELF 文件（`EIATTR_REGCOUNT`），从不读重新汇编的结果。
* **`by_arch`**：在有对应条目的架构上，条目中写明的每个字段都替换该算子的字段（在那里写 `no_spills=False` 就会关闭它）。
* **digest。** 只有给出断言时，它们才进入声明 digest（`sass=None` 和 `tilefish.sass()` 不改变任何 digest）；区域的名称涵盖其 fusion 的断言。它们不在编译规格的文档或 manifest 条目里（schema（模式）没有变化）：读回的规格从 `verify_against` 获得它们。
* **判定范围。** 入口函数，以及它通过 `CALL` 到达的函数（`noinline` 函数、ptxas 的子程序），其中的每一条指令：每个函数的反汇编列表中的指令数必须等于其代码的 16 字节槽数，否则构建被拒绝（另一种 nvdisasm 格式、解析器不认识的操作码或节：不会基于没有覆盖全部代码的列表做任何判定，空列表永远不会通过）。多入口预构建二进制文件中的其他 `.entry` 不随之判定，PTX 检查读取该入口及其调用的 `.func`。
* **何时判定。** 每次构建时：一次编译、一次缓存命中（缓存键不包含断言：同一份 IR 在不同断言下会重新判定）、一次 `jit` 运行中的构建、每次导出，以及预构建二进制文件的汇编。只需要 PTX 和 ELF 文件的检查（`max_registers`、`no_spills`、PTX MMA 匹配模式、有 PTX 的二进制文件的 `no_tf32`）不做任何反汇编；其余的读取 SASS（第 5 节），每个进程对每个二进制文件读一次。`BuiltKernel.sass_checked` 记录构建读了什么（`"its PTX and its ELF file"`、`"its SASS, read by nvdisasm 13.5.37"`，或代码与二进制文件一致的重新汇编）。

违反断言时构建被拒绝，并列出每条被违反的断言、相关指令及其源码行，以及 SASS 是从哪里读取的：

```
'mm_tf32_checked' (sass_kernels:mm_tf32_kernel) for sm_103 violates the SASS assertions of its declaration {"max_registers": 64, "no_tf32": true}:
  - max_registers: the binary uses 121 registers per thread, more than the 64 declared
  - no_tf32: TF32 tensor-core MMAs, which keep 10 of the 23 bits of each f32 mantissa (8 in its PTX): PTX line 446:
    tcgen05.mma.cta_group::1.kind::tf32 (sass_kernels.py:61); PTX line 457: tcgen05.mma.cta_group::1.kind::tf32 (sass_kernels.py:61); ... (2 more)
Refusing to build it: change the kernel (or its launch options) until its binary is what the declaration says, or change the declaration. ...

'spill_checked' (sass_kernels:spill_kernel) for sm_103 violates the SASS assertions of its declaration {"no_spills": true} (read in its SASS, read
by nvdisasm 13.5.37):
  - no_spills: local memory, spills or a local array: a stack frame of 4256 bytes per thread in its ELF file, 4618 local-memory instructions:
    /*0180*/ STL.64 [R1+0xa0], R4 (sass_kernels.py:79) [spill]; /*01b0*/ STL.64 [R1+0xa8], R6 (sass_kernels.py:79) [spill]; ... (4612 more)
```

第一个例子不需要 SASS（PTX 和 ELF 文件已经说明了一切），所以没有提到反汇编器。

## 5. 反汇编器

与收缩检查使用同一个 `nvdisasm`（`tilefish._fpcontract.nvdisasm_candidates`）：`$TILEFISH_NVDISASM_PATH`、`nvidia-cuda-nvdisasm` wheel 中的（`pip install 'tilefish[sass]'` 在 Linux 和 Windows 上会安装 CUDA 13 的版本，它能读取 tilefish 构建的所有架构；在 macOS 上不安装任何东西，因为那里没有这个 wheel，而读取 trace 也不需要它；通过 wheel 的已安装文件定位，不管 pip 把它装在哪里）、Triton 的（`$TRITON_NVDISASM_PATH` 或其自带的副本）、它的 `ptxas` 旁边的那个，然后是 `PATH` 和 `$CUDA_HOME/bin`；取第一个能解码该架构的。反汇编列表用 `nvdisasm -c -g -gi`（代码，加上行号信息，以及 `ptxas` 记录了的指令内联位置的行），PTX 行用 `nvdisasm -c -gp`。解析器能读取 CUDA 12.8 和 13.x 的列表：12.8 会在 sm_120 指令内部打印调度注解（`&req={0}`、`?trans3`），13.x 在指令之后加注释（`(*"SpillRefill"*)`），sm_90 的列表在其头部标志中写明目标架构。

Triton 3.5.1 自带 CUDA 12.8 的 `nvdisasm`，它无法解码 sm_103。在没有任何 `nvdisasm` 能解码该架构时，PTX 会用汇编该二进制文件时所用的 `ptxas` 和标志，为其架构族的基础架构重新汇编（`sm_103a` 用 `sm_100a`，`sm_121a` 用 `sm_120a`；指令集相同；只改写成员架构的 `.target`）：Triton 的（`TRITON_PTXAS_PATH` 或其自带的；`-lineinfo`，kernel 不做收缩时加 `--fmad=false`，`-v`，`disable_ptxas_opt` 时加 `--opt-level 0` 和 `-g`，`ptx_options`），或预构建 PTX 的（`find_ptxas()`，`-lineinfo -v`）。然后把它的 ELF 文件与二进制文件的相比较：每个函数的寄存器、栈帧和代码大小。

* **一致时**：SASS 在重新汇编的结果上判定，每条消息、每个视图、trace 条目和 `BuiltKernel.sass_checked` 都会说明这一点（"the SASS of its PTX assembled again for sm_100a, the base of its family, with the ptxas and the flags of the binary, read by nvdisasm 12.8.55, which cannot decode sm_103a: judged on a re-assembly whose registers, stack frames and code sizes are the binary's"）。
* **不一致时**（实测：`sass_kernels.py` 中单 warp 的 softmax 在 sm_100a 上有 4264 字节的栈帧和 211968 字节的代码，在 sm_103a 上是 4256 和 215936）：需要 SASS 的断言会拒绝构建，并给出修复方法：`pip install 'tilefish[sass]'`（或 `TILEFISH_NVDISASM_PATH`）。由 ELF 文件判定的违反（`no_spills`、`max_registers`）照样报告，只是不列出指令。trace 会记录这种不一致，并在解释重新汇编的结果时给出警告。

断言用到的寄存器和栈帧总是来自二进制文件本身。反汇编结果在本进程内按 cubin、PTX、编译选项、构建种类和 `nvdisasm` 候选列表保存。`-gp` 运行失败只会缺少 PTX 行，其他不受影响。

## 6. 如何测试

* `python/tests/sass_fixtures/`：四个 kernel（bf16 和 TF32 的 `tl.dot`、一个会 spill 的 kernel、一个带原子操作和 f64 运算的 kernel）在 sm_90、sm_100、sm_103 和 sm_120 上的反汇编列表，由 CUDA 13.5 的 `nvdisasm` 生成，sm_100 和 sm_120 另有 CUDA 12.8 的版本；裁剪成测试关注的指令附近的窗口（保留这些指令所在的行号信息和描述符常量）。`python/tests/record_sass_fixtures.py` 重新录制它们（用 Triton，不需要 GPU），并让其中的 kernel 保持在列表所指的行上。`test_sass.py`（解析、类别、描述符及其追踪、调用图、单个入口的 PTX、ELF 属性、occupancy、声明、每种断言、拒绝消息、未覆盖全部代码的列表、反汇编器的选择、重新汇编及其比较、缓存、缓存命中后的重新判定）和 `test_explain.py`（trace 和各个视图，使用假编译器；trace 复制的源码；属于另一个 plan 的 graph 文件）可以在任何机器上运行。
* `test_gpu_sass.py` 在设备上运行：Triton extern、Gluon extern（sm_100 和 sm_103 上的 `tcgen05_mma`）、DSL kernel 和融合区域的带 trace 导出，每个都与不带 trace 导出的 manifest 和 plan 对照；`tl.dot` 和 `tcgen05_mma` 的 MMA 落在各自的行上；每种断言满足和违反的情况；被拒绝的缓存命中；ELF 文件中的寄存器和栈帧与驱动给出的值（`cuFuncGetAttribute`）对照；sm_103 上的架构族基础架构。它已在 sm_103 上分别用 CUDA 13.5 的 `nvdisasm` 和 Triton 自带的运行过；其中 sm_90 和 sm_120 的部分（`sass_kernels.MM_CHECKS` 中按架构区分的 MMA 匹配模式、不带 tcgen05 的 Gluon kernel）还在等这些架构的设备。
* 整个 GPU 门禁在 `TILEFISH_TRACE=1` 下导出时，写出的 manifest、plan 和 plan set 与该版本不带 trace 的导出逐字节一致。
