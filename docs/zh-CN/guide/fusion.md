# 融合：graph 函数体中的算术

[English](../../en/guide/fusion.md) | **简体中文**

graph（计算图）的函数体可以做计算。两次调用之间的算术（两个 GEMM（通用矩阵乘）之间的 `silu(g.to(f32)) * u.to(f32)`、一个 bias 加一个激活、一次残差加、一个由设备标量得到的掩码、一次反量化）会像 kernel（核函数）函数体一样被追踪，并遵循同样的规则（没有隐式类型提升，标量舍入到张量的 dtype，f32 数学函数）；规划器再把它分组成 **区域**（region）：每个区域是 tilefish 生成的一个独立的 `@tilefish.kernel`（按内容命名为 `fused_<hash>`，用同样的模板下沉（lowering）），也是 plan（计划）中一个普通的 kernel 步骤。两种 schema（模式）和三个 host（宿主）都不变（`docs/zh-CN/rfcs/0001-plan-fusion.md`）。

```python
@tilefish.graph(inputs=("x",))
def mlp(x: Tensor[M, H, bf16], w_gu: Tensor[2 * F, H, bf16], w_d: Tensor[H, F, bf16]):
    g, u = gemm_gu(x, w_gu).split(F, axis=-1)
    a = (tilefish.silu(g.to(f32)) * u.to(f32)).to(bf16)      # one kernel, not five
    return gemm_d(a, w_d)
```

**融合从不改变字节。** 融合后的 plan 写出的字节与它的 *未融合组合* 相同：每个算子是一个独立的 kernel，每个中间值都以 IR 给它的 dtype 存储（`tilefish.fusion(grouping="none")` 就是这个 plan，它的 tap（观测点）能看到每个中间值）。区域 kernel 不收缩（contraction）任何运算（`fp_fusion=False`），它的代码也不依赖相邻的算子（`tilefish.lower` 的第 20 条说明；所有 `fp_fusion=False` 的 kernel 的代码，第 2 条说明）：每个浮点 `+ - *` 都是一条标量 `.rn` 内联 PTX 指令，因为 LLVM 会把两个 lane（通道）的 f32 运算打包成 `mul.rn.f32x2` / `add.rn.f32x2`，而 ptxas 12.9 即使在 `--fmad=false` 下也会收缩这一对指令（在 sm_100 和 sm_103 上收缩成 `FFMA2`，在 sm_120 上收缩成 `FFMA`）；每个收窄的浮点转换都是内联 PTX，因为 MLIR 和 LLVM 的折叠会让输入 NaN 的载荷在同一个 kernel 内穿过 `x.to(f32).to(bf16)` 而保留下来；取负则显式给出规范 NaN（所有 kernel 都如此），因为 ptxas 会把 `neg.f32` 作为符号位翻转折叠进 select。当一个归约的字节不可能依赖于区域时，它就并入区域：有序的 `sum` 或 `mean`（`order=`）、与顺序无关的 `max`、`argmax` 和 `argmin`（以及 `gather_rows`）会和周围的算子一起融合成 **行区域**（row region）；没有 `order=` 的 `sum` 或 `mean` 是一个独立的 kernel，它的操作数会被存储下来（当其他加载与 Triton 的树形归约共用一个 kernel 时，树形归约会按另一棵树做加法：实测 1024 行中有 357 到 399 行不同）。没有任何东西读取其值的算子不会被计算。要让 `a * b + c` 只舍入一次，用 `tilefish.fma`。

```python
@tilefish.graph
def add_rmsnorm(x: Tensor[R, D, bf16], res: Tensor[R, D, bf16], w: Tensor[D, bf16]):
    h = x + res                                                           # the residual stream, returned too
    h32 = h.to(f32)
    ms = tilefish.mean(h32 * h32, -1, keepdims=True, order=tilefish.tree(128, offsets=(16, 8, 4, 2, 1, 64, 32), vector=8))
    return (h32 * tilefish.rsqrt(ms + 1e-6) * w.to(f32)).to(bf16), h      # one kernel, two outputs: the unfused composition is ten
```

layer norm（两个有序均值和一个仿射收尾）、softmax（一次缩放、一个由设备标量得到的掩码、`max`、`exp`、一次有序求和、一次除法）、按行的 fp8 量化（一个绝对值最大值、一个同样被返回的每行缩放因子、一次饱和转换），以及对 GEMM 输出的两个列块做的 norm（就地读取）也都各自是一个 kernel。行区域使用第 2 版下沉（`tilefish.lower` 的第 21 条说明）：有序求和通过对半拆分得到各 lane 的链，而不是每一步做一次 gather（同样的加法、同样的顺序，已在每种长度上对照每种顺序的 CUDA kernel 检查过），一行有序求和使用适合它的 warp（线程束）数，fp8 转换是内联 PTX。在一个 B300 切片上，4096 行、每行 1024 列的融合 add + rmsnorm 耗时 16 us，使用 Triton 树形求和的三个 kernel 耗时 65 us，十个未融合 kernel 耗时 103 us（速度表见 `docs/zh-CN/rfcs/0001-plan-fusion.md` 第 2 阶段）。

**每次导出都会检查。** `export` 会把每个包含多个算子的区域与它的各个算子逐一对照运行：在设备上用 `jit`，覆盖每个 bucket（分桶），输入带种子并覆盖所有边角情况（带载荷的 NaN、无穷大、次正规数、带符号零），一旦出现差异就拒绝导出（`verify=False` 可以关闭检查；在其他架构的设备上，检查会被跳过并给出 `TilefishFusionWarning`）。它在 plan 旁边写出 `<graph>.fusion.json`，即 **融合报告**：对每个区域，列出它取走的函数体算子、它读取什么（就地读取，或读一份拷贝）、写出什么、保留的每次舍入以及该模式的改写；`tilefish.fusion_report(graph)` 和 `tilefish.verify_fusion(graph)` 可以按需给出同样的内容。非稠密的视图（上面的两半、一个 head、一次转置）按每个轴一个步长就地读取（版本 1 的行区域则先用 `tilefish.contiguous` 拷贝）。测试在每个 bf16 和 f16 位模式上、在 2**22 个位模式组成的网格上、在 120 个最多 24 个算子的随机 graph 和 80 个由行算子组成的随机 graph 上、在上述行形态（pattern）的每个 bucket 上（每个 bf16 位模式、由特殊值组成的行），保证融合与未融合的结果一致，并在每个 f32 上保证内联转换与 `.to()` 一致。

**模式** 需要声明，`@tilefish.graph(fusion=tilefish.fusion(...))`，而且每个选择都是精确（exact）的（区域的源码带有它，因此 pin（固定）也带有它）：

| 参数 | 取值（默认值在前） | 含义 |
|---|---|---|
| `version` | `2`、`1` | 分组规则、限制和改写，已冻结：`2` 把有序归约融合进行区域；`1` 让每个归约单独运行（即 tilefish 0.0.9 和 0.0.10 的 plan，保持不变）。用另一个版本生成的 plan 会被拒绝，并指出该版本 |
| `grouping` | `"auto"`、`"none"` | 区域取规则允许的最大范围（区域从不跨越一次调用，最多 64 个算子），或每个算子一个 kernel |
| `math` | `"fast"`、`"ieee"` | 函数体中 f32 算子的默认数学模式；算子自己的 `math=` 优先 |
| `intermediates` | `"declared"`、`"opmath"` | 每个算子舍入到它的 IR dtype；或者对 bf16 或 f16 与 Python 标量的 `+ - *` 采用 torch eager 的规则（在 f32 中计算，标量也取 f32，只舍入一次）：配合 `math="ieee"`，得到的位就是 torch eager 组合的位（所有非 NaN 值，已在每个位模式上检查） |
| `fp_contract` | `"none"`、`"mul-add"` | 不收缩任何运算；或者把乘积不被其他地方读取的 f32 `a * b + c`（或 `- c`）变成 `tilefish.fma`，只舍入一次 |
| `verify` | `True`、`False` | `export` 是否检查区域 |

每个改写都在区域形成之前在 IR 上完成，因此它的舍入点从不取决于区域落在哪里。`tilefish.barrier(x)` 让 `x` 成为区域的边界（它得到 plan 中的一个缓冲区，`taps=` 可以读取它）；它从不改变任何字节。尚未规划的有：graph 函数体中的 scatter、并入区域的不带 `order=` 的归约、由区域完成的对 `InOut` 张量或 carried（携带）张量的写入（它是一个拷贝步骤）、区域内的稠密 reshape、带步长的写入、`tilefish.fused(name=...)` 作用域、`intermediates="f32"`，以及非精确的模式（`docs/zh-CN/rfcs/0001-plan-fusion.md` 第 11 节）。融合进 GEMM 或卷积，即 epilogue（尾部）和 prologue（头部），在 RFC 0003 中设计（`docs/zh-CN/rfcs/0003-epilogue-fusion.md`：它区分开的三种数值语义契约、各条路线，以及背后的实验）；它最初的里程碑，extern（外部 kernel）上的 epilogue hook（钩子；E1）和 cuBLASLt epilogue（E2），见下文。没有声明就不会改变任何字节：`epilogues="exact"` 是默认值。

## Epilogue hook：extern 的 GEMM 之后的算术，放进它的 kernel

extern 声明它的 kernel 在哪里接受 epilogue，调用它并对其结果做计算的 graph 就会得到一个 kernel，而不是两个（RFC 0003，里程碑 E1）。kernel 带有一个 `tl.constexpr` hook 参数及其实参，存根不列出它们；kernel 在原本要存储每个输出 tile 的地方对该 tile 调用 hook（`tilefish._epilogue.ABI`）：

```python
@triton.jit
def gemm_kernel(a, b, c, m, n, k, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ..., ACC: tl.constexpr, ACC_ARGS, EPILOGUE: tl.constexpr, EPILOGUE_ARGS):
    ...                                                                   # the main loop: acc, an f32 [BM, BN] tile
    tile = ACC(acc, rows, cols, rows < m, cols < n, ACC_ARGS)            # value="acc": the accumulator
    tile = EPILOGUE(tile.to(c.dtype.element_ty), rows, cols, rows < m, cols < n, EPILOGUE_ARGS)   # value="out": the rounded tile
    tl.store(c + rows[:, None] * n + cols[None, :], tile, mask=...)

@tilefish.extern(triton=gemm_kernel, grid=..., epilogue=(tilefish.hook("ACC", value="acc", out="c"), tilefish.hook("EPILOGUE", value="out", out="c")))
def gemm(a: TensorDesc(Tensor[M, K, bf16], block=(128, 64)), b: TensorDesc(Tensor[N, K, bf16], block=(128, 64)), c: Out[Tensor[M, N, bf16]], m: M, n: N, k: K, ...): ...

@tilefish.graph
def up(x: Tensor[T, H, bf16], w: Tensor[F, H, bf16], bias: Tensor[F, bf16]):
    return tilefish.gelu(gemm(x, w).to(f32) + bias.to(f32)).to(bf16)     # one step: the GEMM with the bias and the GELU in its hook
```

* **默认精确**（`tilefish.fusion()`，`epilogues="exact"`）：区域放进 `value="out"` hook，它看到的是 kernel 已舍入到输出类型、正准备存储的 tile，因此融合后的算子写出的字节与先运行 extern、再运行区域的字节相同（契约 (i)），`export` 会在每个 bucket 上用带种子、含特殊值的输入检查这一点（`verify_fusion` 也会）。
* **作用于累加器**，需要声明：`tilefish.fusion(epilogues="accumulator")`，或一个作用域 `with tilefish.fused(epilogues="accumulator"):`（从不按单次调用声明）。此时 graph 就是其函数体的改写 `R_acc`：对 extern 结果的每个加宽 `y.to(f32)` 都变成 extern 的 f32 结果（由 extern 的累加器形式 `<extern>__acc` 写出），少一次舍入，区域放进 `value="acc"` hook。字节是改写后程序的未融合 plan 的字节（契约 (ii)），在导出时检查；它们不是先 extern 后区域的字节（一个 bias + 激活的输出中约有三分之一不同），而且精度至少一样高。
* **hook 接受什么**：在结果形状上的逐元素区域（bias、激活、带逐通道参数的周期激活、把残差流作为旁路输出的残差加、掩码、类型转换），门控激活的双累加器 tile（`tilefish.hook(..., split=2)`：一个 program 持有 gate 的 tile 和对应的 up projection 的 tile），以及在 extern 的一个 tile 包含整行时的行区域（rmsnorm）（`block_n=`，Triton kernel）。它不接受的部分仍走路线 (a)，即调用之后的一个区域，融合报告、`boundaries()` 和 advisor（建议器）会说明原因（一个 barrier、hook 不持有的视图、中间夹着一次调用、比 tile 更宽的行……）。
* **速度取决于 kernel**：hook 的算术在 extern 的主循环之后，在 extern 的 warp 上运行。在一个 B300 切片上，用示例 GEMM，2048 x 4096 x 1024 GEMM 之后的 bias + GELU 融合后的速度是 GEMM 后接其区域的 1.21 倍，门控激活是 1.14 倍；对 kernel 的 warp 来说过重的 epilogue 可能更慢（卷积之后逐元素做一次 `sin`：0.77 倍；表格见 RFC 0003）。`tilefish.barrier(y)` 把区域留在调用之后。`python/examples/persistent_gemm.py` 是一个示例 kernel（不是受支持的功能，见 `docs/zh-CN/scope.md`），展示了持久化、warp 特化的 GEMM 中的 hook：sm_100/103 上是 Gluon（一个 TMA 生产者、一个 `tcgen05` 发射者，它的 epilogue 在自己的分区上运行，与下一个 tile 的主循环重叠），sm_120 上是 Triton；它的测量结果就是 RFC 0003 的实验 X8。
* **对 host 和 schema 来说没有新东西**：带 hook 的 extern 通过一个生成的小模块启动，该模块用恒等 hook 调用它的 kernel（计入它的声明 digest（摘要）），融合后的一对是一个独立生成的条目 `<extern>_epi_<hash>`，其身份是它编译成的 IR。`python/examples/epilogue_gemm.py` 包含一个 Triton GEMM（sm_100/103 上是 `tcgen05.mma`，sm_120 上是 `mma.sync`，两者都在 SASS 中断言）、它的双累加器形式、一个 Gluon `tcgen05` GEMM，以及一个以隐式 GEMM 实现的 1-D 卷积，每个都带两个 hook。

## cuBLASLt epilogue：cuBLASLt GEMM 之后的算术，放进库里

`tilefish.fusion(epilogues="library")`（RFC 0003，里程碑 E2）声明：结果为 16 位的 `cublaslt.gemm`（或 `gemm_bias`）之后的算术，可以作为它的某个 epilogue 在累加器上运行：

```python
@tilefish.graph(fusion=tilefish.fusion(epilogues="library"))
def block(x: Tensor[M, H, bf16], w: Tensor[F, H, bf16], bias: Tensor[F, bf16], r: Tensor[M, F, bf16]):
    h = (proj(x, w).to(f32) + r.to(f32)) + bias.to(f32)                # c before the bias: the library's order
    return tilefish.where(h <= 0.0, 0.0, h).to(bf16)                   # ReLU with torch.relu's bits: one step, cublaslt.matmul
```

形态是 `y.to(f32)`，然后 `+ c.to(f32)`（`[M, N]`），然后 `+ bias.to(f32)`（`[N]`），然后 `tilefish.where(h <= 0.0, 0.0, h)`，然后 `.to()` 到 GEMM 的类型；除一步外每一步都可选，每个值只被下一步读取，`c` 和 bias 都在 GEMM 之前生成。它会变成一个声明为 `contract="measured-exact"` 的 `cublaslt.matmul` 步骤：`y` 的舍入被去掉（契约 (ii)），每个 pin 都要验证与区域在累加器上的算术相等，否则拒绝构建（`tilefish.barrier(y)` 让 GEMM 保持不融合）。pin 是为 epilogue 选择的，所以在两个 pin 计算同一个和的地方，它的累加器就是 GEMM 的累加器（X1 发现，除了 13.1.0 切分 `k` 的地方，两者在所有形状上都相同）；于是单独一个 ReLU 会写出未融合 plan 的字节（它与舍入可交换），测试也测到了这一点，但没有任何东西逐个 pin 检查这一点。其他顺序（`(y + bias) + c` 在 f32 中的舍入不同）、f32 bias 或在别处被读取的值保持原样，融合报告（`library_epilogues`）会说明原因；在精确的 graph 中，它列出该模式会做什么，advisor 则把这样的区域报告为一项操作（`epilogue-library-cublaslt`）。在一个 B300 MIG 切片上（sm_103，cuBLASLt 12.8.4，20 次 replay（回放）的 CUDA graph（CUDA 图））测得的精确 plan（已 pin 的 GEMM 加一个区域）与改写后 plan 的对比，单位为微秒：

| 形状 `M x N x K` | bias | bias + ReLU | 残差 | 残差 + bias |
|---|---|---|---|---|
| 640 x 4096 x 1024, bf16 | 24.8 → 20.6 (-17%) | 24.8 → 22.2 (-10%) | 24.9 → 22.6 (-9%) | 26.8 → 24.7 (-8%) |
| 640 x 1024 x 4096, bf16 | 18.5 → 16.5 (-11%) | 18.5 → 16.6 (-11%) | 18.6 → 16.6 (-11%) | 18.6 → 18.5 (-0%) |
| 2048 x 3072 x 1024, bf16 | 54.5 → 41.1 (-25%) | 55.3 → 43.1 (-22%) | 61.8 → 46.5 (-25%) | 65.8 → 49.3 (-25%) |
| 128 x 3072 x 1024, bf16 | 8.3 → 6.3 (-24%) | 8.3 → 6.3 (-24%) | 8.3 → 6.3 (-24%) | 8.3 → 6.3 (-24%) |
| 8192 x 3072 x 768, f16 | 192.7 → 124.7 (-35%) | 192.5 → 135.7 (-30%) | 208.0 → 136.3 (-34%) | 215.9 → 147.0 (-32%) |

在 14% 到 29% 的元素上，字节与精确 plan 的不同（少一次舍入）。在 sm_120（一块 RTX 5090）上，cuBLASLt 为 epilogue 提供的唯一算法会把 ReLU 的 NaN 映射为 0，因此在 `"library"` 下，ReLU（以及某些 bias + 残差的形状）在那里会被拒绝，绝不会以不同的字节运行（RFC 0003，7.3）。GELU（由二进制定义）、cuDNN 融合 graph 和生成的 GEMM 以后再做。

**一个 graph，多种模式。** `epilogues=` 取 `"exact"`（默认）、`"accumulator"` 或 `"library"`，按 graph 或按 `tilefish.fused` 作用域设定（对在其中追踪的算子，最内层的作用域优先；从不按单次调用设定）。`"library"` 保留 extern 的精确 hook，并加上 cuBLASLt 改写；`"accumulator"` 对带 `value="acc"` hook 的 extern 应用 `R_acc`，并让 cuBLASLt GEMM 保持原样（尚未规划对库调用做 `R_acc`：GEMM 保留它的舍入，融合报告的 `library_epilogues` 会说明这一点）。只有当一个 cuBLASLt 形态的所有算子都处于 `"library"` 模式时，它才会被改写。两者都想要的 graph 声明其中一种，再用作用域指定另一种：

```python
@tilefish.graph(fusion=tilefish.fusion(epilogues="library"))            # the cuBLASLt GEMM's bias: one cublaslt.matmul
def layer(x: Tensor[M, H, bf16], w: Tensor[F, H, bf16], bias: Tensor[F, bf16], w2: Tensor[N, F, bf16], b2: Tensor[N, bf16]):
    h = (proj(x, w).to(f32) + bias.to(f32)).to(bf16)
    with tilefish.fused(epilogues="accumulator"):                         # the extern's GELU on its accumulator (contract (ii))
        return tilefish.gelu(gemm(h, w2).to(f32) + b2.to(f32)).to(bf16)
```
