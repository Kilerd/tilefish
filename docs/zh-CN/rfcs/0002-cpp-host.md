# RFC 0002：C++ host

[English](../../rfcs/0002-cpp-host.md) | **简体中文**

状态：已接受并已实现，2026-11-15（第 12 节的默认选项已被接受）。本 RFC 处理开放事项“C++ host（宿主）：从同一份 manifest（清单）和 plan（计划）生成的头文件”（从 issue #3 延续到 issue #4）。下面各节描述实际构建的内容；凡是提案未定、由实现确定下来的细节，正文会说明它是什么。

## 1. 概要

C++ 程序获得与 Rust 程序相同的 host，而且不需要第二套实现：

* **`crates/tilefish-capi`**：基于 `tilefish` crate 的 C ABI（一个 `cdylib` 和一个 `staticlib`），带一个有版本号的头文件 `tilefish.h`。它涵盖：加载并校验 manifest、plan 和 plan set（plan 集），启动 kernel（核函数）、已 pin（固定）的 GEMM（通用矩阵乘）和已 pin 的卷积，在 arena（内存池）和 shared arena（共享内存池）上运行 plan，以及 capture（捕获）和 replay（回放）plan。crate 的每一项检查都原样在它背后运行。
* **`tilefish.hpp`**：基于 `tilefish.h` 的纯头文件 C++17 封装，提供 RAII 句柄、类型化的张量和异常。
* **`tilefish.hosts.cpp`**：Python 生成器根据已 pin 的 manifest 和 plan 写出类型化的 C++ 启动器（`emit`）和类型化的 plan 运行器（`emit_plan`）。它的工作方式与 `tilefish.hosts.rust` 相同。

manifest 和 plan 的 schema（模式）都没有变化：C++ host 通过 crate 读取它们。

## 2. host 必须做什么

Rust host 是参照实现。它严格读取 manifest，并对照 pin 检查它的 digest（摘要）。它检查每个 cubin 的 `sha256` 和 `sha256_code`，以及每个条目的一致性。加载二进制文件时，它检查函数的参数大小。加载 GEMM 时，它检查 cuBLASLt 版本并审查已 pin 的算法。加载卷积时，它检查 cuDNN 版本并运行 canary（哨兵值）。每次启动之前，它对照 manifest 检查这次调用：形状、dtype、bucket（分桶）、`divisible_by`、对齐、别名、视图的范围、grid 以及 TMA 描述符。它在一个 arena 上按自己的顺序运行 plan，并把 plan 捕获到 CUDA graph（CUDA 图）中。这大约是 20,000 行 Rust。一套测试向量（`crates/tilefish/tests/data`）让它与 Python 读取器保持一致，一个 GPU 门禁（`python/tests/rust_gate.py`）让它与 Python 的 `replay` 逐字节一致。每个版本都会往里添加内容。在当前周期中，这包括乘积 grid 和持久化 grid、cluster（线程块簇）、全局 scratch、空指针、i64 维度、memset 步骤、carried buffer（携带缓冲区）以及每个 context 一个 cuDNN 句柄。

## 3. 可选方案

| 方案 | 优点 | 缺点 |
|---|---|---|
| **A. 用 C++ 重新实现运行时** | C++ 用户不需要 Rust 工具链。 | 第三个严格读取器和第二个启动器必须与 Python 和 Rust 保持同步。每次 schema 变化都要落到四个地方，测试向量和 GPU 门禁会多出第三列。安全性论证（在第一次启动之前完成每一项检查）必须在一种无法表达 crate 所依赖的借用的语言中重新做一遍。 |
| **B. 基于 crate 的 C ABI、C++ 封装和生成的启动器**（采用） | 每项检查只有一份实现。schema 变化只落到 Python 和 Rust。字节来自同一份代码。C ABI 还能服务 C 以及任何有 C FFI 的语言。 | 在预构建库出现之前（第 9 节），构建 `libtilefish_capi` 需要 Rust 工具链。C 边界用写在文档中的契约取代了 crate 的编译期借用，Python 的 host 也是如此。 |
| **C. 生成自包含的 C++**（一个用固化参数调用 `cuLaunchKernel` 的头文件） | 不需要运行时库。 | 加载时不会校验 manifest（digest、pin、ELF 检查、cuBLASLt 和 cuDNN 的 pin），否则校验也得生成为 C++。GEMM、卷积和 plan 无论如何都需要运行时。 |
| **D. `cxx` 桥接** | 类型化的 C++ 和 Rust 在同一次构建中。 | 每个使用者都需要 Rust 工具链和 `cxx` 的构建集成，而且结果不是稳定的 ABI。 |

决定是 **B**。方案 C 可以作为日后针对单个 kernel 的扩展，构建在同一个生成器之上。

## 4. 分层

```text
generated.hpp, runner.hpp   (tilefish.hosts.cpp: one class per stem and per graph, typed tensors, MANIFEST_DIGEST)
tilefish.hpp                (header-only C++17: RAII, tilefish::Error, TensorRef<T>/TensorMut<T>)
tilefish.h                  (the C ABI, versioned; C99 and C++)
libtilefish_capi            (crates/tilefish-capi: extern "C" over the tilefish crate)
tilefish crate -> cudarc    (libcuda, libcublasLt, libcudnn opened at run time)
```

## 5. C ABI（`crates/tilefish-capi`、`tilefish.h`）

### 5.1 约定

* **命名。** 每个符号都以 `tilefish_` 开头，每个宏都以 `TILEFISH_` 开头。不用短前缀 `tf_`，因为 TensorFlow 的 C API 用的就是它。
* **返回值。** 每个可能失败的函数都返回 `tilefish_status`（一个 `int32_t`；`TILEFISH_OK` 为 0），并通过输出参数写出结果。失败时输出句柄被置为 NULL。不会失败的访问函数直接返回它的值。
* **句柄。** 句柄是指向堆对象的不透明指针，由 `tilefish_*_create` 或 `tilefish_*_load` 创建，由 `tilefish_*_destroy` 释放。`destroy(NULL)` 什么也不做。
* **字符串。** 输入字符串是以 NUL 结尾的 UTF-8。库返回的字符串归它来源的句柄所有，或归线程的错误状态所有（第 5.8 节），只要所有者存在就一直有效。
* **参数结构体。** 它们是按指针传递的普通 C 结构体，例如 `tilefish_tensor`、`tilefish_named_tensor`、`tilefish_devscalar`、`tilefish_bucket` 和 `tilefish_given`。数组以指针加个数的形式传递。除标量外，没有任何东西按值传递或返回。
* **Panic。** 每个入口点都在 `std::panic::catch_unwind` 内运行。panic 是 bug，它会变成带有其消息的 `TILEFISH_ERROR_INTERNAL`，永远不会展开到 C 中。库以 `panic = "unwind"` 构建。
* **没有驱动。** 在没有驱动的机器上，cudarc 第一次被调用时会 panic。在任何需要驱动的调用之前，ABI 检查 `tilefish::driver_available()` 并返回 `TILEFISH_ERROR_NO_DRIVER`。加载和校验 manifest 与 plan 不需要驱动，所以在没有 GPU 的机器上也能工作。
* **不通过宏导出。** cbindgen 按源码原样读取，不展开任何宏，因此每个导出都是普通的 `#[unsafe(no_mangle)] pub unsafe extern "C" fn`；crate 源码中出现 `macro_rules!`，或者出现头文件没有声明的导出，都会让一个测试失败。

### 5.2 句柄

| 句柄 | 封装的对象 | 创建方式 |
|---|---|---|
| `tilefish_context` | `Arc<CudaContext>` 和一条 tilefish 自己的 stream（第 5.4 节） | `tilefish_context_primary(device)`、`tilefish_context_borrow(CUcontext)` |
| `tilefish_cudnn_handle` | `CudnnHandle` | `tilefish_cudnn_handle_create(ctx)` |
| `tilefish_manifest` | `Manifest` | `tilefish_manifest_load(dir, pin)`、`..._load_trusting_directory(dir)`、`..._from_embedded(bytes, files, pin)`、`..._with_sm_partition(m, sms)` |
| `tilefish_kernel`、`tilefish_gemm`、`tilefish_conv` | `Kernel`、`Gemm`、`Conv` | `tilefish_kernel_load(manifest, ctx, stem, buckets, arch)`，GEMM 和卷积同理 |
| `tilefish_plan` | `LaunchPlan` | `tilefish_plan_load(path, pin)`、`..._load_with_manifest`、`..._from_embedded`、`tilefish_plan_set_plan(set, name)`、`tilefish_plan_set_plans(set, names, n, out)`（多个，在多个线程上加载：`PlanSet::plans`） |
| `tilefish_plan_set` | `PlanSet` | `tilefish_plan_set_load(dir, pin)` |
| `tilefish_shared_arena` | `SharedArena` | `tilefish_shared_arena_create(ctx, stream, plans, n)` |
| `tilefish_plan_run` | `PlanRun` | `tilefish_plan_instantiate(plan, ctx, stream, buckets)`、`..._instantiate_in(plan, arena, buckets)`、`tilefish_set_run_plan(set_run, name)` |
| `tilefish_set_run` | `SetRun` | `tilefish_plan_set_instantiate(set, ctx, stream, names, n)` |
| `tilefish_weighted_run` | `WeightedRun<'static>` | `tilefish_plan_run_bind_weights(run, weights, n)` |
| `tilefish_captured_plan` | `CapturedPlan<'static>` | `tilefish_plan_run_capture(run, stream, externals, devscalars)`，在带权重的运行实例上同理 |

对于 `arch`（字符串，`"sm_103"`），NULL 表示 context 的架构（`Arch::of_context`）。对于 `buckets`，NULL 加个数 0 表示没有 bucket（单个二进制文件），对 plan 而言则表示所有 bucket（`instantiate_all`）。句柄彼此独立：kernel 让它的 context 和 module 保持存活，运行实例让它的 plan 和 arena 保持存活（Rust 的 `Arc`），因此句柄可以按任意顺序销毁。通过带权重的运行实例得到的捕获后的 plan 持有那个带权重的运行实例。例外是第 5.10 节中的借用。只读访问函数涵盖通用 host 和测试所需的内容：digest、label 和版本；对 kernel，是按启动顺序排列的张量名；对 plan，是它的缓冲区（名称、角色、dtype、秩、字节数）、输出、维度及其 bucket，以及 arena 大小；对 plan set，是它的 digest 和 graph（计算图）名；对 GEMM 或卷积，是它的尺寸和 workspace（工作区）字节数；对 shared arena，是它的字节数和地址；以及进程中 cuBLASLt 和 cuDNN 的版本。

### 5.3 张量即内存区域

```c
typedef struct tilefish_tensor {
    uint64_t address;         /* a CUdeviceptr: the start of the region */
    uint64_t bytes;           /* the length of the region */
    int32_t device;           /* the device ordinal of the region */
    int32_t dtype;            /* tilefish_dtype: BF16, F16, F32, I32, I64, U8, E4M3, E5M2 */
    int32_t access;           /* tilefish_access: IN, OUT, INOUT */
    uint32_t rank;
    const uint64_t *shape;    /* rank sizes */
    const uint64_t *strides;  /* NULL: dense row-major; otherwise rank element strides (a view) */
    uint64_t offset;          /* a view: the element offset of its first element in the region */
} tilefish_tensor;
```

每个张量映射为基于 `DevicePtr::from_raw(address, bytes, device)` 的 `TensorArg::new` 或 `TensorArg::new_strided`。它的契约就是 `from_raw` 的契约，写在 `tilefish.h` 中。这块内存是该设备上已分配的设备内存（`cuMemAlloc`、`cudaMalloc`、`cudaMallocAsync`、`cuMemMap` 或某个框架的分配器）。它在所有用到它的启动完成之前一直保持分配状态；对于捕获后的 plan，则直到该 plan 被销毁。调用者负责它与自己的 stream 之间的顺序。其余一切仍然在任何东西入队之前检查，与 Rust 相同：张量放得进它的内存区域，以及第 2 节的每一项前置条件。C ABI 不接受 cudarc 切片。它的内存模型就是 Python host 基于 torch 张量的模型，各个 host 本来就共享这一模型。稠密张量（`strides` 为 NULL）的偏移为 0：它的地址就是它第一个元素的地址。启动时形状和步长就地读取；捕获后的 plan 和带权重的运行实例在存活期间保留它们的副本。

具名张量（`{const char *name; tilefish_tensor tensor;}`）是 plan 的外部张量和权重；tap（观测点）是 `{const char *buffer; int64_t after; tilefish_tensor into;}`（`after` 为负数：在最后一个写该缓冲区的步骤之后）。设备标量是 `{const char *name; int32_t kind; float f32_value; int64_t int_value; uint64_t bits;}`，映射为 `DevScalar::{Int, F32, Bits}`。给定值是 `{const char *name; int64_t value;}`，映射为 `launch_with`。每个参数结构体的大小和字段偏移都是固定的（一个 Rust 测试和一个 C 测试检查同样的数字：`tilefish_tensor` 为 56 字节）。

### 5.4 context 与 stream；与驱动 API 和运行时 API 的互操作

* **context 和 stream 归调用者所有。** `tilefish_context_primary(device)` retain 设备的 primary context（cudarc 的 `CudaContext::new`、`cuDevicePrimaryCtxRetain`），`destroy` 释放它。CUDA 运行时 API 使用 primary context，所以它就是基于 `cudaSetDevice` 和 `cudaStream_t` 构建的 host 的 context。`tilefish_context_borrow(CUcontext)` 接收调用者创建的 context（`cuCtxCreate`，或用于 SM（流式多处理器）分区的 `cuCtxFromGreenCtx`）。如果该 context 是其设备的 primary context，`borrow` 就等同于 `primary`，会被 retain 和 release。否则 tilefish 永远不会销毁它。cudarc 的 `from_raw_context` 会接管所有权并在 drop 时销毁它，所以 ABI 在进程的整个生命周期内保留对这个封装的一个引用：每次借用一个小的固定分配。调用者要让借用的 context 保持存活，直到在其中创建的每个句柄都被销毁。
* **stream 是参数。** 每个启动函数都接收一个 `CUstream`，它与 `cudaStream_t` 是同一类型。stream 在调用期间被借用，且必须是句柄所属 context 的 stream：单次启动把它传给 crate 的 `launch_raw`，plan 用 cudarc 的 `wrap_borrowed_stream` 封装它（cudarc 0.19.10 中就有，这是该 crate 支持的最旧版本）。crate 的规则原样适用。旧式的 NULL stream 是接受的。按线程区分的句柄（`CU_STREAM_PER_THREAD`、`cudaStreamPerThread`）在每个线程中指向不同的 stream，在 crate 拒绝它的地方它也被拒绝：plan 的实例化、启动、捕获和回放。
* **每个 context 句柄一条 tilefish 自己的 stream。** cudarc 只在已经创建过自己 stream 的 context 中，才在 stream 之间为它分配的内存（crate 分配的：arena、其中的 workspace、设备标量的暂存）排序，借用的 stream 不算数。所以每个 context 句柄创建一条 stream（`CudaContext::new_stream`，从不用来做实际工作），在该 context 中创建的每个句柄都持有它，于是只要其中任何一个句柄存活，context 就保持这种模式。创建它时跳过了 cudarc 一次性的 `cuCtxSynchronize`，那一步是为新的 `CudaContext` 值尚未分配的内存准备的。
* **arena 的 stream。** 运行实例（`tilefish_plan_instantiate`、`tilefish_plan_set_instantiate`、`tilefish_shared_arena_create`）在给定的 stream 上分配它的 arena，并在最后一个持有该 arena 的句柄被销毁时在同一条 stream 上释放它。在此之前该 stream 一直被借用。
* **当前 context。** 每次调用在执行期间把句柄的 context 设为当前 context，之后恢复原来的当前 context，失败时也是如此。crate 的 `launch_raw` 本来就是这样做的。线程调用 tilefish 不需要有当前 context，tilefish 也不会留下当前 context。
* **单次启动的 CUDA graph。** `tilefish_kernel_launch`、`tilefish_gemm_launch` 和 `tilefish_conv_launch` 是基于内存区域的 `launch_raw`。在调用者正在捕获的 stream 上，它们像任何 kernel 节点一样被捕获，调用者拥有该 graph 及其内存的生命周期，与 Rust 相同。
* **plan 的 CUDA graph。** plan 用 `tilefish_plan_run_capture` 捕获。得到的 `tilefish_captured_plan` 拥有该 graph，并为 arena 与该运行实例的每次启动和回放排序，与 crate 的 `CapturedPlan` 相同。在正在捕获的 stream 上启动 plan 会被拒绝，与 Rust 相同，因为 arena 的排序无法加入调用者的捕获。捕获 plan 所在的 stream 只在调用期间被借用：cudarc 的 graph 保留它的句柄，但只用它在同一个句柄上启动回放。
* **SM 分区。** `tilefish_manifest_with_sm_partition(m, sms)` 就是 `Manifest::with_sm_partition`，用于分区上的持久化 grid 和 GEMM 的 SM 目标。

### 5.5 加载与校验

`tilefish_manifest_load(dir, pin)` 要求提供 pin（64 个小写十六进制数字）。开发时使用另起名字的 `tilefish_manifest_load_trusting_directory(dir)`，这样就不会因为误传 NULL 而走到 `Pin::TrustDirectory`。plan 和 plan set 也是如此。`..._from_embedded` 接收 manifest 字节和一个 `{file name, bytes, length}` 数组，与 `Manifest::from_embedded` 相同。crate 的每个校验错误都以自己的状态码出现（第 5.8 节）。

### 5.6 启动

```c
tilefish_status tilefish_kernel_launch(const tilefish_kernel *k, CUstream stream, const tilefish_tensor *tensors, size_t n,
                                       const tilefish_given *given, size_t n_given);
tilefish_status tilefish_kernel_check(const tilefish_kernel *k, const tilefish_tensor *tensors, size_t n,
                                      const tilefish_given *given, size_t n_given);
```

GEMM 接收 `a`、`b`、`bias`（如果有）、`c`（如果有；一个 `beta` 为 1 的 `cublaslt.matmul`）、`out` 以及调用者的 workspace，后者是一个至少 `tilefish_gemm_workspace_required` 字节的 `U8` 张量（`tilefish_gemm_workspace_bytes` 是该条目被 pin 时使用的大小）；`tilefish_gemm_epilogue`（ABI 1.1）给出它的 epilogue（尾部）以及它是否接收 `c`。卷积接收 `x`、`w`、`y` 和它的 workspace。带全局 scratch 的 kernel 把它作为最后一个参数，即 crate 的 `GLOBAL_SCRATCH` 张量（`tilefish_kernel_tensor_name` 给出它的名字）。M3.5 给启动增加的内容不需要 ABI 做任何事：`null` 参数不是调用参数（crate 在它的位置传 0），`i64` 维度由形状推出，持久化 grid 的大小由设备的 SM 数决定（或由 `tilefish_manifest_with_sm_partition` 的分区决定），cluster 启动在 crate 中通过 `cuLaunchKernelEx` 进行。`tilefish_cudnn_handle_create(ctx)` 提前创建 context 的 cuDNN 句柄，并在该句柄存活期间保留它（crate 的 `CudnnHandle::new`：`cudnnCreate` 会等待整个设备）。

### 5.7 plan、arena 与捕获

`tilefish_plan_run_launch(run, stream, externals, n, devscalars, n_dev)`、`..._check` 和 `..._launch_tapped(..., taps, n_taps)` 接收具名张量。带权重的运行实例只接收输入、原地更新的张量和输出。`tilefish_plan_run_capture` 返回一个捕获后的 plan，它的函数是 `tilefish_captured_plan_replay(cp, stream)` 和 `tilefish_captured_plan_set_devscalar(cp, value)`；`tilefish_plan_run_capture_tapped` 和 `tilefish_weighted_run_capture` 分别带 tap 和通过带权重的运行实例进行捕获。两次回放之间，调用者在回放所用的 stream 上把新的输入写进自己的内存。carried buffer 和 memset 步骤不需要 ABI 做任何事，因为运行实例会处理它们：carried buffer 从不作为外部张量传入，读取它的 plan 根据 arena 记录的形状绑定维度，与 Rust 相同。crate 用 arena 自己的 event 为其上的每次执行排序（当 arena 的上一个使用者是同一条 stream 时跳过等待），所以 ABI 不另外增加排序。

### 5.8 错误

`tilefish_status` 与 crate 的 `Error` 一一对应：`IO`、`MANIFEST`、`BAD_PIN`、`PIN_MISMATCH`、`PLAN`、`PLAN_PIN_MISMATCH`、`NO_BINARY`、`PRECONDITION`、`UNSUPPORTED`、`LAUNCH`、`CUBLAS`、`CUDNN` 和 `DRIVER`，每个都带 `TILEFISH_ERROR_` 前缀。边界另加三个码：`INVALID_ARGUMENT`（NULL 指针、错误的枚举值、不是 UTF-8 的字符串、放不进 `usize` 的大小）、`NO_DRIVER` 和 `INTERNAL`（捕获到的 panic）。最近一次失败的详情按线程保存，与 `errno`、`dlerror` 和 `cudaGetLastError` 的做法相同：

```c
typedef struct tilefish_error_info {
    tilefish_status status;
    int32_t precondition;    /* tilefish_precondition (DTYPE, RANK, SHAPE, ..., TENSOR_MAP) for TILEFISH_ERROR_PRECONDITION, else -1 */
    int32_t cu_result;       /* the CUresult for TILEFISH_ERROR_DRIVER, else 0 */
    const char *message;     /* the crate's message, the same text Rust prints */
    const char *argument;    /* the argument or dimension the precondition is about, or NULL */
    const char *call;        /* the driver, cuBLASLt or cuDNN function, or NULL */
} tilefish_error_info;
void tilefish_last_error(tilefish_error_info *out);   /* valid until the next failing tilefish call on this thread */
const char *tilefish_status_name(tilefish_status status);
```

调用者把未知的状态码当作错误处理，次版本可能会增加新的码（第 5.11 节）。

### 5.9 线程安全

除 `tilefish_captured_plan` 外，每个句柄都可以同时被多个线程使用：它们背后 crate 的类型全都是 `Send + Sync`，C ABI crate 的一个编译期测试断言了这一点，并且 crate 的 arena 锁为同一个 arena 上的并发执行排序。`tilefish_captured_plan` 可以从任何线程使用，但同一时间只能由一个线程使用，因为回放和 `set_devscalar` 会修改它。crate 的 `CapturedPlan` 不是 `Send`，因为它持有原始的 graph 句柄，并且可能持有对 cudarc 切片的借用；如果外部张量全是内存区域，从另一个线程使用它是安全的（C 调用者持有的是指针，所以 Rust 的标记触及不到它）。`WeightedRun` 出于同样的原因既不是 `Send` 也不是 `Sync`（`TensorArg` 可能借用一个切片）：ABI 自己的那个只持有内存区域，并用一个写明理由的 `unsafe impl` 断言这两者。`destroy` 不得与同一句柄的其他使用发生竞争。错误详情是线程局部的。不同 stream 上的并发执行各需要一个 arena（每条 stream 一个运行实例或一个 shared arena），与 Rust 相同。

### 5.10 调用者需要保持的借用

| 对象 | 保持到 |
|---|---|
| 借用的 `CUcontext` | 在其中创建的每个句柄都被销毁 |
| 创建运行实例或 shared arena 时使用的 stream | arena 被释放（它的最后一个运行实例和捕获后的 plan 被销毁） |
| 捕获 plan 时所在的 stream | 调用返回 |
| 一次启动的张量的内存 | 该启动在其 stream 上完成 |
| 捕获后的 plan 的外部张量和 tap 的内存 | 捕获后的 plan 被销毁（回放在这些地址上启动） |
| 带权重的运行实例的权重内存 | 该带权重的运行实例（及其每个捕获）被销毁 |
| 参数中的数组和字符串 | 调用返回（不保留任何东西） |

### 5.11 ABI 版本管理

* `tilefish.h` 定义 `TILEFISH_ABI_MAJOR` 和 `TILEFISH_ABI_MINOR`。`uint32_t tilefish_abi_version(void)` 返回已加载库的 `major << 16 | minor`。头文件中的内联函数 `tilefish_abi_compatible()` 要求主版本相同，且库的次版本不低于头文件的次版本。`tilefish.hpp` 在创建第一个句柄时检查这一点。`tilefish_version()` 是 crate 的版本，`tilefish_manifest_schema()` 和 `tilefish_plan_schema()` 给出库所读取的 schema 的名称。
* **次版本只做添加。** 它可以添加函数和枚举值。它从不改变签名、结构体布局或某个值的含义。新字段意味着新的结构体和新的函数（`..._v2`）。其他任何变化都是新的主版本。
* **共享库。** 它是 `libtilefish_capi.so.1`，SONAME 也是这个（由 crate 的构建脚本设置；一个测试保证构建脚本、CMake 文件和 `TILEFISH_ABI_MAJOR` 是同一个数字），在 macOS 上它的 install name 是 `@rpath/libtilefish_capi.dylib`。它只导出 `tilefish_*` 符号（`#[unsafe(no_mangle)] extern "C"`）。第一个 ABI 是 1.0；1.1 增加了 `tilefish_gemm_epilogue` 和 `TILEFISH_EPILOGUE_*` 值（带 cuBLASLt epilogue 的 GEMM，manifest schema 3 第 9.5 节）。
* **头文件与导出保持同步。** `tilefish.h` 由 cbindgen（`=0.29.4`，一个 dev-dependency，配合 `cbindgen.toml`）从 crate 的 `extern "C"` 条目生成并提交到仓库。一个测试重新生成它，有任何差异就失败（`TILEFISH_BLESS=1` 会写入它），另一个测试在出现头文件没有声明的导出时失败。一个 C99 程序和一个 C++17 程序编译该头文件（两次：检验 include guard），其中 C 程序把参数结构体的 `sizeof` 和 `offsetof` 与一个 Rust 测试同样检查的数字进行比对。

## 6. `tilefish.hpp`

`tilefish.hpp` 是一个纯头文件的 C++17 封装，除 `tilefish.h` 外没有其他依赖。CUDA 头文件是可选的：`tilefish.h` 把 `CUcontext` 和 `CUstream` 声明为与 `cuda.h` 所声明的相同的不完整结构体指针，并加了保护，所以先包含或后包含 `cuda.h` 都没问题。

* **RAII。** 只可移动的类各拥有一个句柄：`Context`（`primary`、`borrow`）、`CudnnHandle`、`Manifest`、`Kernel`、`Gemm`、`Conv`、`LaunchPlan`、`PlanSet`、`SharedArena`（`make_shared_arena`）、`PlanRun`、`SetRun`、`WeightedRun` 和 `CapturedPlan`。如有需要，用 `std::shared_ptr` 共享句柄。
* **错误。** `tilefish::Error` 派生自 `std::runtime_error`，携带 `status()`、`status_name()`、`precondition()`、`argument()`、`call()`、`cu_result()` 和消息。每个失败的调用都抛出它。用 `-fno-exceptions` 构建的程序直接使用 `tilefish.h`。以后可以在不改变 ABI 的情况下增加返回 `expected` 的变体。
* **类型化的张量。** `TensorRef<T>`（访问方式 `IN`）和 `TensorMut<T>`（访问方式 `INOUT`），元素标签为 `Bf16`、`F16`、`F32`、`I32`、`I64`、`U8`、`E4m3` 和 `E5m2`，与 crate 的类型一一对应；两者都是 `Tensor`，即无类型的形式。它们由一个 `Region{address, bytes, device}` 加形状构建，还可以带步长和偏移。因此类型化的启动器不可能在声明为 `bf16` 的地方收到 `f32` 张量。一次调用中最多 8 个尺寸和 16 个张量以内联方式存储，所以启动时不做内存分配。
* **plan。** `run.launch(stream, {{"x", x}, {"out0", y}}, {{"n", 64}})` 接收 `Named` 张量和 `DevScalar` 值（整数、`f32` 或 `DevScalar::bits`），`CapturedPlan cp = run.capture(stream, ...); cp.replay(stream);` 进行捕获和回放；`Tap` 列表是 `launch` 和 `capture` 的最后一个参数。

## 7. 生成的 C++（`tilefish.hosts.cpp`）

`emit(manifest, "launchers.hpp", embed=False, namespace="tilefish_launchers")` 和 `emit_plan(plan, "runner.hpp", pin=..., namespace="tilefish_plans")` 与 `tilefish.hosts.rust` 相对应。它们遵循同样的规则：只接受已 pin 的 manifest 或 plan，进入输出的每个字段都经过校验。因此 manifest 无法逃出注释、标识符或字符串字面量。校验代码就是 Rust 生成器的代码，通过导入复用；是 C++ 关键字或被生成代码使用的名字会加上一个 `_`，C++ 保留的名字会被拒绝。

* **启动器。** kernel、GEMM 或卷积的每个 stem 对应一个类，其中固化了 `STEM`、`BUCKETS`（一个 `std::array<tilefish_bucket, N>`）、每个张量声明的形状和各个尺寸（对 GEMM 或卷积还有 `M`、`N`、`K`、`BIAS`、`WORKSPACE_BYTES`、`PINNED_ARCHS` 和 `PINNED_WITH`）。`static Cls load(const Manifest&, const Context&)` 拒绝 digest 不是 `MANIFEST_DIGEST` 的 manifest。`void launch(CUstream, const TensorRef<Bf16>& x, const TensorMut<Bf16>& y) const` 为每个张量接收一个类型化的参数，然后是全局 scratch 和每次调用给定的值（当二进制文件有这些时）；`check(...)` 接收同样的参数，但没有 stream。文档注释写明 manifest 所说的内容，与 Rust 相同。
* **嵌入。** `embed=True` 把 manifest 和 cubin 以字节数组的形式写进一个配套的源文件 `launchers_embedded.cpp`，并在头文件中声明 `load_embedded()`。这样大数组只编译一次，而不是在每个包含该头文件的翻译单元中都编译。以后可以用 C23 和 C++26 的 `#embed` 取代这些数组。生成器在嵌入文件之前对照 pin 检查它们。
* **plan 运行器。** 每个 graph 对应一个类，带有 `PLAN_DIGEST`、`GRAPH`、`ARCH` 和 `ARENA_BYTES`，`load(directory, ctx, stream)`（pin 到 plan 的 digest，实例化所有 bucket），以及接收一个 `Externals` 聚合体（每个输入、权重、原地更新的张量和输出各一个类型化成员，按 plan 中的顺序）和一个 `Scalars` 聚合体（每个设备标量一个成员）的 `launch` 与 `capture`。这两个结构体位于命名空间作用域（`<Graph>Externals`、`<Graph>Scalars`），类中有它们的别名。
* **输出。** 输出是确定的（它的 include guard 带有 digest 的前缀，因此同一个 graph 名的两个运行器可以被包含进不同的命名空间），并通过单元测试（已提交 fixture 和一个 golden plan 的 golden 文件）以及在 C++ 测试中编译它来检查。

## 8. cuBLASLt 和 cuDNN 如何加载

`libtilefish_capi` 在构建时不链接任何 CUDA 库（cudarc 的 `dynamic-loading`）。它在第一次调用驱动时打开 `libcuda.so.1`，在加载第一个 GEMM 时打开 cuBLASLt，在加载第一个卷积时打开 cuDNN。不含这些的 manifest 永远不会打开它们。这些库按名字打开，顺序就是 crate 的 README 所列的顺序，因此会通过 `LD_LIBRARY_PATH`、系统搜索路径，或进程已经以相同 SONAME 加载的库找到。因此，自己链接 cuBLASLt 或 cuDNN、或者先打开它的 C++ host，会让 tilefish 使用那一份。随后加载过程对照 pin 检查那一份的版本。在另一个 cuBLASLt 下 pin 的 GEMM，或在另一个 cuDNN 下 pin 的卷积，会被拒绝，返回 `TILEFISH_ERROR_MANIFEST` 以及一条写明两个版本的消息。无法打开的库给出 `TILEFISH_ERROR_UNSUPPORTED`。每个 context 一个 cuDNN 句柄，由该 context 的所有卷积共享，包括同一进程中通过 Rust 加载的卷积。`tilefish_cudnn_handle_create` 把需要等待设备的句柄创建挪到初始化阶段。如果 host 需要在多份库之间选择，以后可能会增加显式的库路径（环境变量或函数）。

## 9. 构建与分发

* **cargo 优先。** `cargo build -p tilefish-capi --release` 产出 `libtilefish_capi.{so,a}`（macOS 上为 `.dylib`）。该 crate 具有与 `tilefish` 相同的 `cuda-*` feature：绑定决定了它能运行的最旧驱动。
* **CMake 包。** `crates/tilefish-capi/CMakeLists.txt`（CMake 3.19 或更新）通过一个自定义命令用 cargo 构建该 crate（`TILEFISH_CARGO_PROFILE`；`TILEFISH_CARGO_FEATURES` 用于选择另一个 `cuda-*` 绑定），并安装头文件、两个库和 `tilefishConfig.cmake`。该配置提供导入目标 `tilefish::tilefish`（共享库）和 `tilefish::tilefish_static`（静态库，带有 Rust 标准库所需的链接接口：Linux 上为 `pthread`、`dl`、`m`、`rt`，macOS 上为 `iconv`）。使用者写 `find_package(tilefish CONFIG REQUIRED)` 和 `target_link_libraries(app PRIVATE tilefish::tilefish)`。对该 crate 目录执行 `add_subdirectory` 得到同样的目标。
* **预构建库**是本 RFC 落地之后的一个发布步骤：按平台（Linux x86_64 和 aarch64，有一个 glibc 基线）和按驱动基线（用于 12.x 驱动的 12.4 绑定，以及 13.x 绑定）分别提供，连同 sha256 附在 GitHub release 上。在此之前，C++ host 需要 Rust 工具链才能构建。
* **把多个 Rust 静态库链接进同一个二进制文件**可能导致 Rust 标准库的符号重复。这样的二进制文件使用共享库。
* **macOS。** 该 crate 和 C++ 单元测试在没有 CUDA 的情况下也能构建和运行：`tilefish_driver_available()` 为 false，加载和校验都能工作。

## 10. 测试与门禁

* **Mac，无 GPU。**
  * `cargo test -p tilefish-capi`：每个 `Error` 变体和每种前置条件对应的状态码，NULL 参数和无效参数，捕获到的 panic，`TILEFISH_ERROR_NO_DRIVER`，ABI 版本，参数结构体的布局，句柄所封装对象的 `Send + Sync`，头文件是最新的且声明了每个导出，以及加载已提交的 fixture manifest：用它的 pin、用错误的 pin、用格式不对的 pin、信任目录、从嵌入的字节、从被篡改的副本（每种拒绝的状态码和消息），一个 golden plan 及其对自身的描述，以及一个 golden plan set。
  * CMake 和 ctest：一个 C99 程序和一个 C++17 程序，只使用头文件，分别链接静态库和共享库，重复上述加载检查；C++ 程序还编译已提交的 `generated.hpp` 和 `block_runner.hpp`。第三个测试把包安装到一个临时前缀，并用 `find_package(tilefish)` 构建一个使用者。
  * `uv run pytest`：生成器的单元测试（校验，拒绝未 pin 的 manifest，确定性，golden 输出；生成的头文件用 `-Wall -Wextra -Werror` 编译；嵌入的 manifest 与库链接并通过校验）。
  * 已有的门禁不变。
* **GPU：C++ 门禁。** `rust_gate.py prepare WORK` 在 `generated.rs` 旁边写出 `generated.hpp`、`generated_embedded.cpp`、`block_runner.hpp` 和 `fused_runners.hpp`（每个融合后的 plan `fused_*` 的运行器，以及从某个用例对每个运行器的一次调用），输入相同，回放字节也相同。一个 C++ 程序（`crates/tilefish-capi/tests/gpu/cpp_gate.cpp`，由 CMake 构建）通过 C ABI 运行 `cases.json` 和 `plans.json` 中的每个用例，在有对应分支的地方使用类型化的启动器。覆盖范围包括：kernel（视图、设备标量、预构建的二进制文件、TMA 描述符、warp（线程束）特化的 kernel、cluster、持久化 grid、全局 scratch、给定值）、GEMM（以及 workspace 用例）和卷积，分别使用 manifest 目录和嵌入的 manifest；plan 的启动、捕获以及两次回放；一个 plan set；一个 set 的各个阶段共用一个 shared arena；在 plan 之间传递的 carried buffer；只绑定一次的权重；plan `block` 的生成运行器；以及 graph 函数体中有计算的 plan（`fused_*`，plan 融合），既以无类型方式运行，也通过它们的生成运行器运行。它把输出写到 `cpp/`、`cpp_embedded/`、`cpp_plans/`、`cpp_plans_graph/`、`cpp_shared/`、`cpp_shared_graph/`、`cpp_carried/`、`cpp_carried_graph/` 和 `cpp_fused_runners/`，`rust_gate.py compare WORK --host cpp` 要求它们与 Python 的回放逐字节一致，与它对 Rust 的要求相同。门禁还按状态码检查拒绝：它能构造的每种前置条件（形状、访问方式、dtype）、大于其内存区域的张量、错误的 pin、被篡改的 cubin、另一种架构、在正在捕获的 stream 上启动的 plan，以及在新的 arena 上一个没有任何东西写过的 carried buffer。它还检查在 host 自己的 CUDA graph 中捕获的 kernel 和 GEMM 回放出相同的字节，以及借用的 primary context 能够启动，并让线程的当前 context 保持原样。门禁用测试目录中的一个小型读取器读取 JSON，并为自己的内存打开 `libcuda.so.1`，所以构建它不需要 CUDA toolkit。
* **边界的开销。** `tests/gpu_overhead.rs`（`--features gpu-gate`）在同一个进程中分别通过 crate 和通过 C ABI 启动一个 kernel 和一个 plan，并报告各自的 host 端时间。
* **门禁证明了什么。** C++ host 运行的是 Rust 代码，所以从构造上讲它的 kernel 就是 Rust 的 kernel。门禁测试的是边界：张量和步长的转换、偏移、设备标量的编码、外部张量的名字、捕获的生命周期以及错误映射。

## 11. crate 需要什么

什么都不需要：`tilefish` crate 没有变化。kernel、GEMM 和卷积基于内存区域使用 `launch_raw`；plan 使用 `wrap_borrowed_stream`，它在 cudarc 0.19.10（该 crate 支持的最旧版本）中就有；每个 context 的 cuDNN 句柄是 `CudnnHandle`，并行加载 plan 是 `PlanSet::plans`。每次调用的边界开销（参数转换、`catch_unwind`、恢复当前 context，以及对 plan 而言的 stream 封装）由 `tests/gpu_overhead.rs` 测量。在一个 B300 MIG 切片上，release 构建中，一次 kernel 启动通过 crate 耗费 1.77 µs host 端时间，通过 `tilefish_kernel_launch` 耗费 1.93 µs；门禁中 plan `block`（九个步骤）的一次启动分别为 33.1 µs 和 31.8 µs（差异在噪声范围内），因此不需要缓存 stream 封装。

## 12. 风险与开放问题

* **用契约代替借用。** C ABI 无法检查生命周期。调用者释放某个外部张量之后再回放，会写进已释放的内存，任何 CUDA graph 都是如此。头文件写明了每一项契约（第 5.10 节），其他所有检查照常运行。
* 在预构建库发布之前需要**两套工具链**。
* **调用者 CUDA graph 中的 plan** 会被拒绝，与 Rust 相同。需要用一个 graph 容纳一切的 host 自己捕获单次启动。导出捕获后 plan 的 graph 句柄、以便作为子节点嵌入，是以后可能增加的功能。
* **本版本不包括**：显式的 cuBLASLt 或 cuDNN 路径、返回 `expected` 的 C++ API、自包含的生成启动器（方案 C），以及预构建库。
* **决定**（2026-11-15 接受）：
  * 前缀 `tilefish_` 和库名 `libtilefish_capi`；
  * 只提供带异常的 C++17（`-fno-exceptions` 的调用者使用返回状态码的 C API）；
  * 头文件由 cbindgen 生成并由测试保持同步，而不是手写；
  * 预构建库在本实现之后提供，不在本实现之中。
