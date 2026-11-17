# SASS：解释与断言

[English](../../en/guide/sass.md) | **简体中文**

每个 kernel（核函数）都经过 DSL / Triton / Gluon → TTIR → TTGIR → LLVM IR → PTX → ptxas → SASS。tilefish 可以按需保留一次导出的每一层，把每条 SASS 指令映射回产生它的那一行源码和那个算子，并在每次构建时检查所声明的关于 SASS 的事实（`docs/zh-CN/explain.md`）。

```python
# every level of every binary next to the manifest: sources, frontend IR, TTIR, TTGIR, LLVM IR, PTX, SASS and its line table
runtime.export(block, "build/block", trace=True)          # or TILEFISH_TRACE=1 for an export run by a script you do not edit

@tilefish.extern(triton=mm_kernel, num_warps=8, grid=..., sass=tilefish.sass(
    no_tf32=True, no_spills=True, max_registers=168,
    mma="tcgen05.mma kind::f16",                          # a PTX pattern; upper case is a SASS one ("HMMA.16816.F32")
    by_arch={"sm_120": tilefish.sass(mma="HMMA.16816.F32")},
))
def mm(...): ...
```

```console
$ tilefish-explain build/mm mm --view lines
mm on sm_103 (extern): the SASS of its PTX assembled again for sm_100a, the base of its family, read by nvdisasm 12.8.55, which cannot decode sm_103a (...)
  registers 122, frame_size 0, static_shared 1024, num_barriers 1; dynamic shared 131088

 count   mma  gmem  smem   tma  tmem local  atom  sync   alu  line
    79     .    20    15     .     .     .     .    13    31  mm_kernels.py:27  a = tl.load(a_ptr + offs_m[:, None] * k + (kk + offs_k)[None, :])
   112     .    20     .     .     .     .     .     5    87  mm_kernels.py:28  b = tl.load(b_ptr + (kk + offs_k)[:, None] * n + offs_n[None, :])
   156     8     .     .     .     2     .     .     6   140  mm_kernels.py:29  acc = tl.dot(a, b, acc)
   297     .    16    33     .     1     .     1     8   238  mm_kernels.py:30  tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], acc)
...
$ tilefish-explain build/swiglu --graph swiglu --step 0 --bucket R=8 --view lines
fused_ad9189093e91 on sm_103 (fused): the SASS of its PTX assembled again for sm_100a, the base of its family, read by nvdisasm 12.8.55, ...
  step 0 of the plan of 'swiglu' at {'R': 8}: launches 'fused_ad9189093e91'; fused region 0 of the ops #1 cast (fusion_graphs.py:51), #2 silu (...
 count   mma  gmem  smem   tma  tmem local  atom  sync   alu  line
     8     .     .     .     .     .     .     .     .     8  tilefish_generated/fused_ad9189093e91.py:57 [#1 cast @ fusion_graphs.py:51]  v2 = v0.to(tl.float32)
   121     .     .     .     .     .     .     .     .   121  triton/language/standard.py:48  return 1 / (1 + math.exp(-x))
    16     .     .     .     .     .     .     .     .    16  tilefish_generated/fused_ad9189093e91.py:20 [helper _tf_mul_rn_f32 (for nodes 1 silu, 3 mul)]  [a, b],
...
$ tilefish-explain build/mm mm --view summary          # MMA kinds and shapes, TMA, spills, barriers, registers, occupancy
$ tilefish-explain build/mm mm --diff-arch sm_120       # or --diff OTHER_EXPORT: what changed, per source line
```

* **trace**（追踪记录；即 manifest（清单）旁边的 `trace/`，有自己的索引和 digest（摘要））位于已 pin（固定）的产物之外：有没有它，manifest、各个 plan（计划）和每个 digest 都相同。它覆盖下沉（lowering）生成的 kernel、融合区域、生成的算子、Triton 和 Gluon extern（外部 kernel），以及预构建二进制文件的 PTX 和 SASS。
* **`tilefish-explain`**（`python -m tilefish.explain`、`tilefish.explain.load`）为 kernel、plan 步骤或融合区域的 SASS 标注每条指令的源码行：Triton 或 Gluon kernel 的行；或者生成源码的行、该行为之生成的 kernel graph（计算图）中的算子以及 `@tilefish.kernel` 函数体的行；对于融合区域，还有按融合报告编号的 graph 函数体中的算子。它可以在任何地方读取 trace，无需任何 CUDA 工具，依据的是相对于模块的源文件名（`--source-root`）；没有 trace 时，它会反汇编 manifest 中的 cubin。
* **SASS 断言**（`@tilefish.kernel`、`@tilefish.extern` 和 `tilefish.fusion` 上的 `tilefish.sass(...)`）一旦给出就计入声明 digest，每次构建（包括命中缓存时）都会拒绝违反断言的二进制文件，并指出相应的指令及其源码行。断言读取 PTX（MMA 种类：`tcgen05.mma` 的 SASS 不会写明其种类）、ELF 文件（寄存器、栈帧）和 SASS，SASS 通过收缩（contraction）检查所用的 `nvdisasm` 读取；在没有任何工具能解码该架构的地方（Triton 3.5.1 的 CUDA 12.8 工具与 sm_103），SASS 是用该二进制文件的 ptxas 和标志、为该架构族的基础架构（`sm_100a`）汇编 PTX 所得的结果；只有当这次重新汇编的寄存器、栈帧和代码大小与该二进制文件一致时，才据此作出判断（否则构建会被拒绝，并给出修复方法），每条消息都会说明这一点。`pip install 'tilefish[sass]'`（CUDA 13 的 `nvidia-cuda-nvdisasm`，存在时优先使用）或 `TILEFISH_NVDISASM_PATH` 可以直接读取 sm_103。`no_spills` 和 `max_registers` 取自 ELF 文件，无需反汇编。
* trace 中含有源码：不要把它随导出一起发布。

**哪些步骤可以合成一个 kernel**（`docs/zh-CN/advise.md`）：`tilefish-explain build/block --graph block --advise` 按可节省的字节数和时间，对 plan 的融合候选（由中间结果连接起来的步骤，而合成一个 kernel 后就不必存储这些中间结果）排序，说明每个候选为什么目前不是一个 kernel 以及该怎么做（把 extern 的算术写进 graph 函数体；加上 `order=`，让融合第 2 版把归约纳入行区域；去掉一个 barrier；对于 GEMM（通用矩阵乘）的 epilogue（尾部）：在 extern 上用 epilogue hook（钩子），或对 cuBLASLt GEMM 用 `tilefish.fusion(epilogues="library")`，[`fusion.md`](fusion.md)），并按类别汇总 plan 的时间。`tilefish.profile_plan(block, "build/block", bucket="M=128", ...)`（或 `--profile`）在设备上测量每个步骤，这是一条位于所有产物之外的运行时路径，`--timing` 则按其测量结果排序。

**kernel 内部发生了什么**（`docs/zh-CN/instrument.md`）：`tilefish.probe("load.wait")` 在 `@tilefish.kernel` 函数体、`@triton.jit` 或 `@gluon.jit` kernel 中标记一个阶段边界，生产构建会把它编译为空。插桩变体（`runtime.export(..., instrument=True)`，或在 jit 中用 `tilefish.run(..., probes=tilefish.Probes())`）为每个 warp（线程束）把 `%globaltimer`、`%clock`、SM（流式多处理器）和 probe（探针）记录到 host（宿主）在 profile scratch 槽位中给出的缓冲区里，它有自己的二进制文件和 digest，除非明确要求，否则每个 host 都会拒绝它们；Python 和 Rust 在一次启动、一次 plan 运行或一次被 capture（捕获）的 replay（回放）之后收集这些记录，`tilefish-explain DIR --probes FILE` 给出各阶段及其按 warp 角色（warp 特化 kernel 的各分区）的分布、按 SM 和 CTA（线程块）的时间线，以及一份 Chrome trace。
