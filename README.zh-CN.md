# tilefish

[English](README.md) | **简体中文**

一门面向 GPU 推理的类型化张量语言。每个张量的形状、dtype、布局和量化格式都写在它的类型里，算子来自一个类型化的算子库，一个模型只需构建一次，得到 GPU 二进制文件和一份 manifest（清单）。轻量的 Python、C++ 和 Rust host（宿主）由同一套签名生成。

**状态：早期预览（PyPI 上为 0.0.14，crates.io 上为 0.0.11；C++ host 从本仓库构建）。** tilefish 为这些 GPU kernel（核函数）赋予类型：用 Triton、Gluon、CuTe DSL 或 PTX 编写的 kernel，已 pin（固定）的 cuBLASLt 和 cuDNN 调用，以及它自己下沉（lowering）生成的 kernel。它把一个模型的 graph（计算图）构建成基于同一个 arena（内存池）的 launch plan（启动计划），并在 Python、Rust 和 C++ 中运行同一份已 pin 的产物。tilefish 是什么、不是什么，以及今后如何添加功能，见 [`docs/zh-CN/scope.md`](docs/zh-CN/scope.md)；其他文档列在 [`docs/zh-CN/README.md`](docs/zh-CN/README.md) 中。

## 为什么

在推理代码里，一个 kernel 的契约通常要写三遍，而其中没有一处会检查它：

- 形状、dtype、对齐和别名规则散落在一段 docstring、几个 `assert` 和 C ABI 上方的一段注释里；
- 块缩放的权重格式（MXFP8、MXFP4、NVFP4）用文字描述，连同它的缩放因子在各个架构上的字节布局；
- 同一个 kernel 被手工从 Triton 移植到 CUDA C++，再移植到 Rust，一致性测试事后去追最后一个 ULP（FMA（乘加融合）收缩（contraction）、flush-to-zero、归约顺序）；
- 在哪块 GPU 上、对哪个 `(M, N, K)` 运行哪个实现，是某人曾经测过一次的一张表。

tilefish 把这些全部移进类型里，让编译器可以检查它们、生成它们并对它们进行推理。

## 代码长什么样

```python
import tilefish
from tilefish import Dim, Tensor, InOut, Out, BlockScaled, const
from tilefish import bf16, f32, e4m3, e2m1, ue8m0

M = Dim("M")                          # runtime dimension
D = Dim("D", 2560)                    # static: baked into the binary
H = Dim("H", 1024)
F = Dim("F", 3072)
K = Dim("K", divisible_by=32)         # constraint, checked at build time
N = Dim("N", divisible_by=128)

mxfp8 = BlockScaled(elem=e4m3, scale=ue8m0, block=32, axis=-1)
mxfp4 = BlockScaled(elem=e2m1, scale=ue8m0, block=32, axis=-1)
nvfp4 = BlockScaled(elem=e2m1, scale=e4m3, block=16, global_scale=f32)

# A kernel whose body tilefish understands: it can be fused, and it is
# lowered to Triton.
@tilefish.kernel
def residual_add_rmsnorm(
    x: Tensor[M, D, bf16],
    residual: InOut[Tensor[M, D, f32]],
    w: Tensor[D, bf16],
    *, eps: const[f32] = 1e-6,
) -> Tensor[M, D, bf16]:
    r = residual + x.to(f32)
    residual[...] = r
    rstd = tilefish.rsqrt(tilefish.mean(r * r, axis=-1, keepdims=True) + eps)
    return (r * rstd * w.to(f32)).to(bf16)      # the only rounding point

# An existing hand-written kernel, unchanged, given a type.
n = Dim("n", divisible_by=16)

# (index_bits=32 is the default: a kernel that offsets with i32 gets a max_elements
# limit of 2**31 - 1 on every tensor, which both hosts check; a kernel that
# indexes in i64 says index_bits=64.)
@tilefish.extern(triton=my_kernels.gelu, num_warps=4,
                 grid=lambda n: (tilefish.cdiv(n, 1024),))
def gelu(x: Tensor[n, bf16], y: Out[Tensor[n, bf16]],
         n_elem: n, BLOCK: const = 1024): ...

# A library operator: opaque body, typed signature, pluggable epilogue.
@tilefish.library("gemm", acc=f32)
def gemm(a: Tensor[M, K, bf16], b: Tensor[N, K, mxfp8]) -> Tensor[M, N, bf16]: ...

# A model is ordinary Python over typed values.
@tilefish.graph
def mlp(x: Tensor[M, H, bf16], w_gu: Tensor[2 * F, H, mxfp8], w_d: Tensor[H, F, mxfp8]):
    g, u = gemm(x, w_gu).split(F, axis=-1)
    a = (tilefish.silu(g.to(f32)) * u.to(f32)).to(bf16)
    return gemm(a, w_d)
```

## 工作原理

1. **声明。** 类型（`Dim`、`Tensor`、`BlockScaled`、`DevScalar`、视图）和四种声明：`@tilefish.kernel`（一段由 tilefish 自己下沉到 Triton 的函数体）、`@tilefish.extern`（一个被赋予类型的 Triton、Gluon 或 CuTe DSL kernel，或预构建的 PTX 或 cubin）、`@tilefish.library`（一个已 pin 的 cuBLASLt GEMM（通用矩阵乘）或 cuDNN 卷积）和 `@tilefish.graph`（基于它们的一个模型步骤）。
2. **一次构建。** `tilefish.runtime.export` 追踪并检查 graph，预先编译每个 kernel，pin 住每个库算法，并写出 cubin、`manifest.json` 和 launch plan（`plan.json`：一个步骤在同一个 arena 上的每一次启动）。manifest 的 digest（摘要）覆盖每个二进制文件和每个 pin，plan（计划）的 digest 覆盖 plan 及其 manifest；host pin 住其中一个。一次导出在任意目录下都可复现，在工具链和 GPU 架构相同的另一台机器上也可复现。
3. **运行同一份产物。** Python（`replay`）、Rust crate 和 C/C++ 库加载这个目录，按 pin 校验它，并在启动前检查每一个前置条件。`jit` 运行在内存中构建出的同一份机器码，`reference` 则在 torch 中运行同样的函数体。
4. **看清内部。** `tilefish-explain` 把每条 SASS 指令映射到它的源码行，SASS 断言在每次构建时都会检查；probe（探针）为 kernel 内部的各个阶段计时；`tilefish-explain --advise` 为 plan 中可以合成一个 kernel 的步骤排序。

各个阶段的细节见[指南](docs/zh-CN/README.md#指南)。

## 安装

```sh
pip install tilefish                  # types, tracing, lowering, compile specs: pure Python, no dependencies
pip install "tilefish[gpu]"           # torch, Triton and cuda-python: build, jit, replay, plans
pip install "tilefish[sass]"          # CUDA 13's nvdisasm, for tilefish-explain and SASS assertions on sm_103
cargo add tilefish                    # the Rust host
cmake -S crates/tilefish-capi -B build && cmake --build build   # the C/C++ host, from this repository
```

为 sm_103 构建需要比 Triton 3.5.1 自带的版本更新的 `ptxas`（自带的是 CUDA 12.8 的 `ptxas`，无法以 `sm_103a` 为目标）：把 `TRITON_PTXAS_PATH` 设为 CUDA 12.9 或更新版本的 `ptxas`。因这个原因失败的构建会说明这一点；为该 `ptxas` 无法作为目标的架构构建的 GPU 测试会跳过，并给出该架构的名字。

## 快速上手

### Python host

```python
# x, residual, w: torch tensors on the GPU (residual, an InOut, is updated in place)
y = tilefish.run(residual_add_rmsnorm, x, residual, w)                        # jit: built in memory
m = tilefish.runtime.export(residual_add_rmsnorm, "build/")                    # cubins and manifest.json
y = tilefish.run(residual_add_rmsnorm, x, residual, w, mode="replay", manifest="build/", pin=m.digest)
y = tilefish.run(residual_add_rmsnorm, x, residual, w, mode="reference")      # the same body in torch
```

`@tilefish.graph` 会把它的 launch plan 导出到 manifest 旁边，`tilefish.run(graph, ..., mode="replay", plan="build/", pin=p.digest)` 回放它（[构建](docs/zh-CN/guide/build.md)、[启动计划](docs/zh-CN/guide/plans.md)）。

### Rust host

```rust
let manifest = Manifest::load("build/", Pin::Digest("<manifest digest>"))?;
let kernel = manifest.kernel(&ctx, "gelu_erf", &[], Arch::of_context(&ctx)?)?;
kernel.launch(&stream, &mut [x.into_arg(), y.into_arg()])?;     // every precondition is checked first

let plan = LaunchPlan::load("build/", Pin::Digest("<plan digest>"))?;
let run = plan.instantiate_all(&stream)?;
run.launch(&stream, &mut [("x", x.into_arg()), ("out0", y.into_arg())], &[("n", DevScalar::Int(64))])?;
```

`tilefish.hosts.rust.emit` 为一份已 pin 的 manifest 生成类型化的启动器（[指南](docs/zh-CN/guide/rust.md)、[`crates/tilefish/README.md`](crates/tilefish/README.zh-CN.md)）。

### C++ host

```cpp
#include "tilefish.hpp"

auto ctx = tilefish::Context::primary(0);
auto run = tilefish::LaunchPlan::load("build/", "<the plan's digest>").instantiate(ctx, stream);
run.launch(stream, {{"x", xt}, {"out0", yt}}, {{"n", 64}});
```

`cmake --install` 之后用 `find_package(tilefish)`；`tilefish.hosts.cpp.emit` 生成类型化的启动器和 plan 运行器（[指南](docs/zh-CN/guide/cpp.md)）。

## 功能一览

| 功能 | 文档 |
|---|---|
| tilefish 是什么、不是什么，以及如何添加功能 | [`docs/zh-CN/scope.md`](docs/zh-CN/scope.md) |
| 构建：各个步骤，`jit`、`reference` 和 `replay`，pin，可复现的导出，什么标识一个二进制文件，构建缓存，TF32 防护，信任 | [`docs/zh-CN/guide/build.md`](docs/zh-CN/guide/build.md) |
| 带步长的视图、设备标量、预构建的 PTX 和 cubin、Gluon kernel、每个架构一个实现、TMA 描述符、warp（线程束）特化的 kernel、启动形状 | [`docs/zh-CN/guide/operators.md`](docs/zh-CN/guide/operators.md) |
| 已 pin 的 cuBLASLt GEMM（bf16、f16，以及关闭 TF32 的 f32；四种布局；SM（流式多处理器）分区；epilogue（尾部））和已 pin 的 cuDNN 卷积 | [`docs/zh-CN/guide/operators.md`](docs/zh-CN/guide/operators.md) |
| 作为 extern（外部 kernel）的 CuTe DSL kernel | [`docs/zh-CN/cute.md`](docs/zh-CN/cute.md) |
| 下沉生成的 kernel：掩码、索引、`argmax`、gather 与 scatter、整数类型转换、IEEE 数学函数、有序求和、广播、`contiguous`、`im2col`、`col2im1d` | [`docs/zh-CN/guide/lowering.md`](docs/zh-CN/guide/lowering.md) |
| launch plan：同一个 arena、CUDA graph（CUDA 图）、copy 和 memset 步骤、只绑定一次的权重、plan set（plan 集）、shared arena（共享内存池）、carried buffer（携带缓冲区）、调试 tap（观测点） | [`docs/zh-CN/guide/plans.md`](docs/zh-CN/guide/plans.md) |
| 锁定导出（locked export）：在一块 GPU 上生成一次 lock（锁定表），然后在一台没有 GPU 的机器（CI、镜像构建）上导出同样的文件 | [`docs/zh-CN/guide/locked-export.md`](docs/zh-CN/guide/locked-export.md) |
| graph 函数体中算术运算的融合、extern 上的 epilogue hook（钩子）、cuBLASLt epilogue | [`docs/zh-CN/guide/fusion.md`](docs/zh-CN/guide/fusion.md) |
| 舍入点、精确（exact）重写与改变数值语义的重写、`num_warps` 与 FMA 收缩 | [`docs/zh-CN/guide/rewrites.md`](docs/zh-CN/guide/rewrites.md) |
| CUDA 数学函数防护，以及 CUDA 或 torch 变更后需要重新检查的内容 | [`docs/zh-CN/numerics.md`](docs/zh-CN/numerics.md) |
| trace（追踪记录）、`tilefish-explain` 与 SASS 断言 | [`docs/zh-CN/guide/sass.md`](docs/zh-CN/guide/sass.md)、[`docs/zh-CN/explain.md`](docs/zh-CN/explain.md) |
| 融合 advisor（建议器）与 plan 性能分析器 | [`docs/zh-CN/advise.md`](docs/zh-CN/advise.md) |
| probe 与插桩变体 | [`docs/zh-CN/instrument.md`](docs/zh-CN/instrument.md) |
| Rust host 与生成的 Rust 启动器 | [`docs/zh-CN/guide/rust.md`](docs/zh-CN/guide/rust.md)、[`crates/tilefish/README.md`](crates/tilefish/README.zh-CN.md) |
| C ABI、C++ 封装与生成的 C++ 启动器 | [`docs/zh-CN/guide/cpp.md`](docs/zh-CN/guide/cpp.md)、[RFC 0002](docs/zh-CN/rfcs/0002-cpp-host.md) |
| host 读取的文件 | [`docs/zh-CN/manifest-schema-3.md`](docs/zh-CN/manifest-schema-3.md)、[`docs/zh-CN/plan-schema-1.md`](docs/zh-CN/plan-schema-1.md) |

## tilefish 不是什么

- **不是 kernel 编译器。** 编写 kernel 的是 Triton、CuTe DSL、cuTile、Helion 和 TileLang。tilefish 为它们赋予类型、构建它们，并作为 host 运行它们。
- **不是推理服务运行时。** 调度、KV cache、批处理和请求处理都留在宿主程序中。tilefish 负责的是在它规划好的缓冲区上执行的一个前向步骤。
- **不承诺生成每一个 kernel。** 手工调优的 attention 和 GEMM 以 `extern` 或 library operator（库算子）的形式接入，并保留它们自己的数值语义契约。

## 状态与版本

早期预览：API 还会变化。

| 包 | 版本 | 来源 |
|---|---|---|
| `tilefish`（Python） | 0.0.14 | PyPI |
| `tilefish`（Rust host） | 0.0.11 | crates.io |
| `tilefish-capi`（C ABI 1.1，`tilefish.h` 和 `tilefish.hpp`） | 0.0.2 | 本仓库 |

manifest 采用 schema（模式）3，plan 采用 `tilefish.plan.v1`。没有一份待开发功能的清单。只有当真实的模型迁移需要某个功能时，才会添加它（`docs/zh-CN/scope.md`）；未完成的事项记录在 issue [#4](https://github.com/Kilerd/tilefish/issues/4) 中，[#5](https://github.com/Kilerd/tilefish/issues/5)（SASS 溯源、probe、CuTe DSL extern）已关闭。每个版本改了什么见 [`CHANGELOG.md`](CHANGELOG.md)，所有文档都列在 [`docs/zh-CN/README.md`](docs/zh-CN/README.md) 中。

## 许可证

可任选 [Apache License, Version 2.0](LICENSE-APACHE) 或 [MIT license](LICENSE-MIT) 之一授权。
