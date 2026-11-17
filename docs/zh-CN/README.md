# 文档

[English](../en/README.md) | **简体中文**

* [`scope.md`](scope.md)：tilefish 是什么、不是什么，以及如何添加功能。
* [`../../README.zh-CN.md`](../../README.zh-CN.md)：概览、安装，以及每种 host（宿主）的快速上手。
* [`../en/README.md`](../en/README.md)：本索引的英文版，列出英文页面。

## 指南

从声明到在 host 中启动，每个阶段的参考文档：

* [`guide/build.md`](guide/build.md)：构建如何工作：它的步骤，`jit`、`reference` 和 `replay` 三种模式，pin（固定）与可复现的导出，什么标识一个二进制文件（它的 Triton IR），构建缓存，`replay` 拒绝什么，TF32 防护，信任，以及 CI 在 GPU 上检查什么。
* [`guide/operators.md`](guide/operators.md)：视图、设备标量、预构建的二进制文件和已 pin 的库调用：带步长的视图与共享的步长，起始地址不在 16 字节边界上的视图，设备标量，只有 `Out` 张量才有的维度，预构建的 PTX 和 cubin，空指针与 64 位维度标量，Gluon kernel（核函数），每个架构一个实现，TMA 描述符，warp（线程束）特化的 kernel，启动形状，已 pin 的 cuBLASLt GEMM（通用矩阵乘；f16、关闭 TF32 的 f32、四种布局、SM（流式多处理器）分区、cuBLASLt epilogue（尾部）），已 pin 的 cuDNN 卷积，以及 CuTe DSL kernel。
* [`guide/lowering.md`](guide/lowering.md)：下沉（lowering）生成的 `@kernel` 中的掩码、索引、gather 和复制，以及它们的数值语义：比较与 select，`iota`，`argmax`，`gather_rows` 与 `scatter_rows`，整数类型转换，IEEE 数学函数，复现一个 CUDA kernel 所需的原语，CUDA 数学函数防护，有序求和，广播，`contiguous`，`im2col`、`im2col1d` 和 `col2im1d`。
* [`guide/plans.md`](guide/plans.md)：launch plan（启动计划）：graph（计算图）作为基于同一个 arena（内存池）的一份静态启动列表，CUDA graph（CUDA 图），写在自身输入之上的 `Out`，copy 和 memset 步骤，返回的视图，具名的输出、输入和权重，plan set（plan 集），一个 plan set 中各个 plan（计划）共享的 arena，carried buffer（携带缓冲区），调试 tap（观测点），以及单独运行一个步骤。
* [`guide/fusion.md`](guide/fusion.md)：graph 函数体中算术运算的融合、它的几种模式、extern（外部 kernel）上的 epilogue hook（钩子），以及 cuBLASLt epilogue。
* [`guide/rewrites.md`](guide/rewrites.md)：感知舍入的重写：精确（exact）规则与改变数值语义的规则，`num_warps` 与 FMA（乘加融合）收缩（contraction）。
* [`guide/sass.md`](guide/sass.md)：SASS 的解释与断言：trace（追踪记录）、`tilefish-explain`、SASS 断言，以及 advisor（建议器）和 probe（探针）的简介。
* [`guide/rust.md`](guide/rust.md)：Rust host、`tilefish.hosts.rust.emit` 和逐字节一致门禁。
* [`guide/cpp.md`](guide/cpp.md)：C++ host（C ABI 和 `tilefish.hpp`）、`tilefish.hosts.cpp`，以及门禁中 C++ 的那一半。

## 参考

* host 读取的格式：
  * [`manifest-schema-3.md`](manifest-schema-3.md)：manifest（清单；kernel 条目、ABI、预构建的二进制文件、已 pin 的库调用）；
  * [`plan-schema-1.md`](plan-schema-1.md)：launch plan 与 plan set。
* [`numerics.md`](numerics.md)：数值语义契约所依赖的东西，以及 CUDA 或 torch 变更后如何重新检查它。
* [`cute.md`](cute.md)：作为 extern 的 CuTe DSL kernel（`nvidia-cutlass-dsl` 的 pin 及其许可证）。
* 工具：
  * [`explain.md`](explain.md)：trace、`tilefish-explain` 与 SASS 断言；
  * [`instrument.md`](instrument.md)：probe 与插桩变体；
  * [`advise.md`](advise.md)：融合 advisor。
* Rust crate 自己的页面：[`../../crates/tilefish/README.zh-CN.md`](../../crates/tilefish/README.zh-CN.md)。

## 设计记录与历史

* [`rfcs/`](rfcs)：[0001，plan 融合](rfcs/0001-plan-fusion.md)；[0002，C++ host](rfcs/0002-cpp-host.md)；[0003，epilogue 与 prologue（头部）融合](rfcs/0003-epilogue-fusion.md)。
* [`../../CHANGELOG.md`](../../CHANGELOG.md)：每个版本改了什么。
