# C++ host

[English](../../en/guide/cpp.md) | **简体中文**

`crates/tilefish-capi` 是 Rust host（宿主）的 C ABI：`libtilefish_capi`（一个共享库和一个静态库）和 `tilefish.h`（由 cbindgen 从 crate 生成，并由一个测试保持最新），外加一个仅头文件的 C++17 封装 `tilefish.hpp`。每一次加载、验证、检查和启动都由 `tilefish` crate 完成：没有第二套实现，因此 C++ host 拒绝 Rust host 所拒绝的，并运行相同的机器码（`docs/zh-CN/rfcs/0002-cpp-host.md`）。manifest（清单）和 plan（计划）的 schema（模式）都没有变化。

```sh
cmake -S crates/tilefish-capi -B build && cmake --build build   # cargo builds the library
ctest --test-dir build                                         # C99 and C++17 tests, no GPU needed
cmake --install build --prefix /opt/tilefish                   # find_package(tilefish): tilefish::tilefish, tilefish::tilefish_static
```

```cpp
#include "tilefish.hpp"

auto ctx = tilefish::Context::primary(0);   // the device's primary context: the CUDA runtime API's
auto manifest = tilefish::Manifest::load("build/", "<the manifest's digest>");
auto kernel = manifest.kernel(ctx, "gelu_erf");
tilefish::Region x{x_ptr, x_bytes, 0}, y{y_ptr, y_bytes, 0};   // device memory the host owns
kernel.launch(stream, {tilefish::TensorRef<tilefish::Bf16>(x, {4096}), tilefish::TensorMut<tilefish::Bf16>(y, {4096})});

auto run = tilefish::LaunchPlan::load("build/", "<the plan's digest>").instantiate(ctx, stream);
run.launch(stream, {{"x", xt}, {"out0", yt}}, {{"n", 64}});
auto graph = run.capture(stream, {{"x", xt}, {"out0", yt}}, {{"n", 64}});
graph.replay(stream);
```

`tilefish.hosts.cpp` 与 `tilefish.hosts.rust` 一样写出类型化的启动器和 plan 运行器，只从已 pin（固定）的 manifest 或 plan 生成，并校验每个字段：每个二进制文件、GEMM（通用矩阵乘）和卷积各一个类，其 `load` 拒绝 digest（摘要）不同的 manifest，其 `launch` 为每个参数接受一个类型化的张量（声明为 `f32` 的地方不能传入 `bf16` 张量）；每个 graph（计算图）一个类，带有一个 `Externals` 和一个 `Scalars` struct。使用 `embed=True` 时，manifest 及其 cubin 进入一个配套的源文件（`launchers_embedded.cpp`），由 `load_embedded()` 用 pin 验证。

```python
from tilefish.hosts import cpp
cpp.emit(m, "launchers.hpp", embed=True)          # m = tilefish.runtime.export(...), or a directory with pin=
cpp.emit_plan("build/", "runner.hpp", pin=plan_digest)
```

host 拥有什么，以及边界上的行为：

- **内存** 是一块区域（`tilefish_tensor`、`tilefish::Region`）：由 host 担保的一个地址、一个长度和一个设备，与 Rust 中的 `DevicePtr::from_raw` 一样；host 要让它一直保持分配，直到使用它的启动完成（对于被 capture（捕获）的 plan，直到该 plan 被销毁），并负责它与自己其他 stream 之间的排序。其余一切都在任何工作入队之前检查，包括张量是否放得进其区域。
- **context 和 stream** 属于 host。context 句柄要么是设备的 primary context（会被 retain 和 release），要么是 host 借出的 context（`Context::borrow`，例如一个 green context），tilefish 从不销毁后者。stream 是一个 `CUstream`，与 `cudaStream_t` 是同一类型。每次调用都让线程的当前 context 保持调用前的状态。
- **CUDA graph（CUDA 图）。** 在 host 正在 capture 的 stream 上的单次启动，会像任何 kernel（核函数）节点一样被 capture。plan 用 `capture` 来 capture（在正在 capture 的 stream 上启动 plan 会被拒绝）：被 capture 的 plan 拥有自己的图，并负责它的 arena（内存池）与其他每次运行之间的排序。
- **错误** 是每次调用返回的一个状态（`TILEFISH_ERROR_*`：crate 的每种错误各一个，另有无效参数、没有驱动和捕获到的 panic），以及线程的最后一个错误（`tilefish_last_error`：消息、前置条件、`CUresult`）；`tilefish.hpp` 把它们作为 `tilefish::Error` 抛出。不启用异常构建的程序使用 `tilefish.h`。
- **线程。** 每个句柄都可以同时被多个线程使用，被 capture 的 plan 除外（同一时间只能一个线程）。同时在多个 stream 上运行需要各用一个 arena。
- **cuBLASLt 和 cuDNN** 在运行时按名字打开（`LD_LIBRARY_PATH`，或 host 先行加载的一份副本），与 Rust 中一样，用其他版本做出的 pin 在加载时会被拒绝。
- **版本。** `tilefish_abi_version()` 以及头文件中的 `TILEFISH_ABI_MAJOR` 和 `TILEFISH_ABI_MINOR`：次版本只会新增函数和值，`tilefish.hpp` 会拒绝主版本不同的库。

逐字节一致门禁的 C++ 部分在同一个目录上运行 Rust 门禁的用例：`python rust_gate.py prepare WORK` 同时写出 C++ 启动器和 plan 运行器（block 的，以及每个融合 plan 的），`cmake -S crates/tilefish-capi -B build -DTILEFISH_BUILD_GPU_GATE=ON -DTILEFISH_GATE_DIR=WORK` 构建 `tilefish_cpp_gate`（`crates/tilefish-capi/tests/gpu/cpp_gate.cpp`：kernel、GEMM、卷积、启动并 replay（回放）的 plan、融合的 plan、shared arena（共享内存池）、carried buffer（携带缓冲区）、只绑定一次的权重、拒绝、由单次启动构成的图），`python rust_gate.py compare WORK --host cpp` 要求它写出的结果与 Python 的 replay 逐字节一致。
