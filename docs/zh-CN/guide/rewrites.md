# 感知舍入的重写

[English](../../en/guide/rewrites.md) | **简体中文**

有了类型，判断融合是否合法就成了机械的事：逐元素运算链融入 GEMM（通用矩阵乘）的 epilogue（尾部），norm 融入它前面的残差加法，量化融入下一个 GEMM 的 prologue（头部），排列变换变成步长而不是复制 kernel（核函数），缓冲区原地复用。每个算子的字节数和 FLOP 数由形状和 dtype 宽度决定，所以构建可以为每个 kernel 报告 roofline，并统计每个步骤的启动次数。

融合也可能改变数值语义。每一次到更窄类型的转换都是 IR 中的一个舍入点，规则分为两类：

- **精确（exact）**：保留每一个舍入点，只移动 kernel 的边界。这类规则自动应用：graph（计算图）函数体中算术运算的融合（[`fusion.md`](fusion.md)）就是其中之一。
- **改变数值语义**：归约顺序、累加器 dtype 或舍入点发生变化。只在明确要求时应用，并以 reference 为基准设置门禁。

`num_warps` 同样是 kernel 数值身份的一部分：它决定了 `tl.sum` 相加所用的树，所以同一个 kernel 换一个 warp（线程束）数，最后几位就会不同。它记录在声明的 digest（摘要）、编译规格和 manifest（清单）中，replay（回放）会拒绝一个取值不同的条目。实测表明，依赖它的是 Triton 自己的归约；逐元素运算、`argmax` 和带 `order=` 的求和在任何 warp 数下字节都相同，所以一个每个归约都指定了顺序的 kernel 可以单纯为了速度而改变它（`docs/zh-CN/numerics.md`）。FMA（乘加融合）收缩（contraction）也是如此：生成的 kernel 总会说明 `a * b + c` 是否可以只做一次舍入（`@tilefish.kernel(fp_fusion=False)` 表示不可以），手写的 kernel 则用 `@tilefish.extern(triton=..., enable_fp_fusion=False)` 来说明（即 Triton 的同名编译选项；默认值 `None` 交给 Triton 决定）。这个设置包含在声明的 digest、每一份编译规格以及 manifest 条目的 `options` 中，所以用另一种设置构建的二进制文件会被 replay 拒绝。“不可以”是确实守住的，而不是寄希望于它：单靠这个编译选项，在 sm_100 及之后的架构上守不住，因为 LLVM 会把两个 lane（通道）的 f32 运算打包在一起，而无论 `--fmad` 怎么设置，ptxas 12.9 都会收缩打包后的这一对运算。所以，`fp_fusion=False` 的生成 kernel 会把每个浮点 `+ - *` 都写成一条标量 `.rn` 指令；而每个用 `enable_fp_fusion=False` 构建的二进制文件（包括手写的 Triton 或 Gluon kernel 的二进制文件）都会在构建时被读回检查：如果它的 SASS 中的融合乘加多于“去掉乘法后的 PTX”所得 SASS 中的融合乘加，它就会被拒绝，并给出 kernel 的名字和修复方法（对关键的舍入使用标量 `.rn` 内联 PTX，或使用 `enable_fp_fusion=True`；`docs/zh-CN/numerics.md`）。

一个合法的融合是否值得做，仍然要靠测量，而不是假设。测量结果就放在仓库中，与规则放在一起。
