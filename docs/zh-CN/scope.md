# tilefish 是什么，不是什么

[English](../scope.md) | **简体中文**

tilefish 是一门面向 GPU 推理的类型化 DSL。你用类型（形状、dtype、布局、量化格式）描述张量和算子，把一个模型一次构建成 GPU 二进制文件和一份 manifest（清单），再在 Python、Rust 或 C/C++ 中运行同一份产物。

## 核心

1. **类型。** 每个张量都带有它的形状、dtype、布局和量化格式；算子有类型化的签名。
2. **四种声明。**
   - `@tilefish.kernel`：由 tilefish 自己下沉（lowering）的小型 kernel（内核）。
   - `@tilefish.extern`：在别处编写的 kernel。Triton、Gluon、CuTe DSL 以及预构建的 PTX 或 cubin 都通过这种方式接入。
   - `@tilefish.library`：已 pin（固定）的库调用（cuBLASLt GEMM（通用矩阵乘）、cuDNN 卷积）。
   - `@tilefish.graph`：一个模型步骤，表示为基于这些算子的一份 plan（计划）。
3. **一次构建，pin 住结果。** cubin、manifest 和 plan 只导出一次，并由 digest（摘要）检查。一次导出在任意目录下都可复现，在工具链和 GPU 架构相同的另一台机器上也可复现。
4. **轻量的 host（宿主）。** Python、Rust 和 C/C++ 加载同一份产物，并以同样的方式启动它。没有哪个 host 会重新推导任何东西。
5. **写明的数值语义。** 舍入点、累加类型、归约顺序以及“精确（exact）”的含义都是契约的一部分。正是这一点让一个模型可以从现有实现逐字节一致地迁移过来。

核心之外是一些辅助工具：带 SASS 检查的 `tilefish-explain`、probe（探针）、融合 advisor（建议器），以及 graph（计算图）函数体中算术运算的精确融合。

## 非目标

- **在 kernel 速度上与 cuBLASLt、cuDNN 或 CUTLASS 竞争。** 高性能 kernel 请用 Triton、Gluon 或 CuTe DSL 编写，再作为 extern（外部 kernel）接入。tilefish 为它们赋予类型、pin 住它们并运行它们。`python/examples/` 中的示例 kernel（例如 persistent GEMM）只是示例，不是受支持的功能。
- **通用的优化编译器。** 融合仅限于能被证明是精确的情况，或 graph 显式声明的情况。
- **基准测试套件。** 每个改动都必须通过正确性门禁（字节和 digest）。当某个决定取决于速度时，速度在真实模型上端到端地测量。

## 如何添加功能

从 0.0.13 起，只有当真实的模型迁移需要新功能时才会添加它们，而不会提前添加。
