# tilefish

[English](https://github.com/Kilerd/tilefish/blob/main/python/README.md) | **简体中文**

一门面向 GPU 推理的类型化张量语言：形状、dtype、布局和量化格式都在类型里，一次构建得到 GPU 二进制文件，并生成 Python、C++ 和 Rust host（宿主）。

**状态：早期预览。** 这个版本可以做类型检查：张量类型和块缩放格式（MXFP8、MXFP4、NVFP4），`extern` / `library` / `kernel` / `graph` 声明，追踪成类型化的 IR，以及为 `extern` kernel（核函数）生成 Triton 预先编译（ahead-of-time）规格。纯 Python，无依赖。`@kernel` 还能下沉（lowering）为 Triton 源码（`tilefish.lower`）；装上 `gpu` extra（`pip install "tilefish[gpu]"`：torch、Triton、cuda-python）后，一份规格可以构建成 cubin（`tilefish.aot.build`），一个 manifest（清单）目录存放这些 cubin（`tilefish.runtime.export`），kernel 可以以 `jit`（通过它的规格在内存中构建）、`replay`（manifest 中的 cubin，按它的 digest（摘要）校验；过期的二进制文件会被拒绝，并且它需要一个信任锚：manifest 已 pin（固定）的 digest，或显式的 `trust_directory=True`）和 `reference` 模式运行（`tilefish.run`）。一个二进制文件由它所编译的 Triton IR 标识。库 GEMM（通用矩阵乘）运行一个已 pin 的 cuBLASLt 算法；`@tilefish.graph` 编译成基于同一个 arena（内存池）的 launch plan（启动计划；`tilefish.plan`，由导出写入 `plan.json`），它以同样的三种模式运行，并可以被 capture（捕获）进 CUDA graph（CUDA 图；`tilefish.capture`）。下沉生成的 kernel 还可以做比较、选择、按行 gather 和 scatter、归约到一个索引、沿任意轴广播，并可以使用 IEEE f32 数学函数（`math="ieee"`）和 `tanh`；`im2col1d` 和 `col2im1d` 是为一维卷积生成的算子；extern（外部 kernel）可以是 Gluon kernel，可以每个架构一个实现，可以接收 TMA 张量描述符（`TensorDesc`），也可以是 warp（线程束）特化的；已 pin 的 GEMM 接受 bf16、f16 或 f32 操作数、f32 结果、布局 `TN`、`NN`、`TT` 和 `NT` 以及一个 SM（流式多处理器）分区，它的算法按 torch 的方式选择，或者导入；已 pin 的 cuDNN 前向卷积（`cudnn.conv_fwd`）也是一种 library op（库算子）；任何构建都不会把 IEEE f32 的点积变成 TF32 tensor core MMA；一起导出的 plan（计划）共享同一份 manifest，并且可以共享同一个 arena。crates.io 上的 `tilefish` crate 在 Rust 中运行同样的 manifest 和 plan，`tilefish.hosts.rust.emit` 为它生成类型化的 Rust 启动器。API 还会变化。

tilefish 是什么、不是什么，以及从 0.0.13 起如何添加功能，见 [`docs/zh-CN/scope.md`](https://kilerd.github.io/tilefish/zh-CN/scope/)。

## 安装

```sh
pip install tilefish                  # types, tracing, lowering, compile specs: pure Python, no dependencies
pip install "tilefish[gpu]"           # torch, Triton and cuda-python: build, jit, replay, plans
pip install "tilefish[sass]"          # CUDA 13's nvdisasm, for tilefish-explain and SASS assertions on sm_103
```

## 文档

- [README](https://github.com/Kilerd/tilefish/blob/main/README.zh-CN.md)：一个示例、工作原理、每种 host 的快速上手和功能一览。
- [文档索引](https://kilerd.github.io/tilefish/zh-CN/)，以及其中的指南：[构建](https://kilerd.github.io/tilefish/zh-CN/guide/build/)、[算子](https://kilerd.github.io/tilefish/zh-CN/guide/operators/)、[下沉生成的 kernel](https://kilerd.github.io/tilefish/zh-CN/guide/lowering/)、[启动计划](https://kilerd.github.io/tilefish/zh-CN/guide/plans/)、[融合](https://kilerd.github.io/tilefish/zh-CN/guide/fusion/)、[重写](https://kilerd.github.io/tilefish/zh-CN/guide/rewrites/)、[SASS](https://kilerd.github.io/tilefish/zh-CN/guide/sass/)、[Rust host](https://kilerd.github.io/tilefish/zh-CN/guide/rust/) 和 [C++ host](https://kilerd.github.io/tilefish/zh-CN/guide/cpp/)。
- [更新日志](https://github.com/Kilerd/tilefish/blob/main/CHANGELOG.md)：每个版本改了什么。

可任选 Apache License, Version 2.0 或 MIT license 之一授权。
