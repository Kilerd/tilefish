# tilefish（Rust host）

[English](https://github.com/Kilerd/tilefish/blob/main/crates/tilefish/README.md) | **简体中文**

[tilefish](https://github.com/Kilerd/tilefish/blob/main/README.zh-CN.md) manifest（清单）的 Rust host（宿主）。tilefish 把 kernel（核函数）一次性构建成 CUDA 二进制文件和一个 `manifest.json`。这个 crate 加载该目录，从 Rust 启动其中的 cubin，不需要 Python，也不需要 Triton，并做 Python 的 `replay` 所做的每一项检查。机器码与 Python 的 `replay` 运行的字节完全相同，`tilefish.hosts.rust.emit` 为一个 manifest 生成类型化的启动器。

**状态：早期预览。** API 还会变化。这个 host 的指南是 [`docs/guide/rust.md`](https://github.com/Kilerd/tilefish/blob/main/docs/zh-CN/guide/rust.md)，它读取的 manifest 和 plan（计划）的规范见 [`docs/manifest-schema-3.md`](https://github.com/Kilerd/tilefish/blob/main/docs/zh-CN/manifest-schema-3.md) 和 [`docs/plan-schema-1.md`](https://github.com/Kilerd/tilefish/blob/main/docs/zh-CN/plan-schema-1.md)，所有文档都列在 [`docs/README.md`](https://github.com/Kilerd/tilefish/blob/main/docs/zh-CN/README.md) 中。

```rust
use tilefish::cudarc::driver::CudaContext;
use tilefish::{Arch, Bf16, Manifest, Pin, TensorMut, TensorRef};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !tilefish::driver_available() {
        return Ok(()); // CudaContext::new panics without an NVIDIA driver
    }
    // Production pins the digest that `tilefish.runtime.export` returned.
    let manifest = Manifest::load("build/", Pin::Digest("<manifest digest>"))?;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;
    let kernel = manifest.kernel(&ctx, "gelu_erf", &[], Arch::of_context(&ctx)?)?;

    let (x, mut y) = (stream.alloc_zeros::<u8>(4096 * 2)?, stream.alloc_zeros::<u8>(4096 * 2)?);
    let shape = [4096usize];
    let x = TensorRef::<Bf16>::from_cuda(&x, &shape)?;
    let y = TensorMut::<Bf16>::from_cuda(&mut y, &shape)?;
    kernel.launch(&stream, &mut [x.into_arg(), y.into_arg()])?; // every precondition is checked first
    Ok(())
}
```

以下检查都在任何调用到达驱动之前完成：

- **加载时：** manifest 是 schema（模式）3 的严格 JSON（更旧的 schema 会被拒绝：请重新导出），它的 digest（摘要）就是已 pin（固定）的那一个，每个 cubin 都与其 `sha256` 和机器码 digest 相符，并且针对其条目所写的架构，加载出的函数接受 ABI 规定的参数大小。
- **启动时：** 由形状绑定维度（视图的步长绑定其步长维度），先用 kernel 读取的张量的形状，再用 `Out` 张量的形状绑定只有它才有的维度（由设备标量生成的 `[B]` 掩码）；dtype、bucket（分桶）、`divisible_by`、`max_elements` 和 `max_extent`、别名、每个指针承诺的对齐、连续性、设备和 context，以及 grid 限制，全都会检查。

schema 3 描述的一切都能启动：Triton kernel 和预构建的 PTX 与 cubin 二进制文件（严格按其参数启动，`fixed` 参数写在各自的槽位上，`given` 参数的值每次调用固定，通过 `Kernel::launch_with(stream, tensors, &[("h_out", 5)])` 或 plan 步骤传入）、空指针（kind 为 `null`：槽位上为 0，没有张量）、带步长的视图（`TensorRef::from_cuda_strided`）、设备标量（只有一个元素的张量）、`i64` 步长和维度标量、只承诺其元素对齐的稠密指针（下沉（lowering）生成的 kernel 针对视图的特化版本），以及已 pin 的 cuBLASLt GEMM（通用矩阵乘；`Manifest::gemm`、`Gemm::launch`：进程中的 cuBLASLt 必须是选出该算法的那个版本，workspace（工作区）是调用方的最后一个张量，因此启动时不分配任何内存，可以被 capture（捕获）到 CUDA graph（CUDA 图）中）。条目会说明其算法是如何选出的（`LibraryEntry::heuristic`、`algo_source`：在 tilefish 或 torch 的设定下由启发式选出，或是导入的），split-K（K 维切分）以 bf16 累加部分和的算法只对明确选择启用的条目运行（`allow_bf16_reduction`）。GEMM 可以有 f16 操作数、f32 结果、以 `[K, N]` 存储的 `b` 和以 `[K, M]` 存储的 `a`（`LibraryEntry::dtype`、`out_dtype`、`layout`：`GemmLayout::Tn`、`Nn`、`Tt` 或 `Nt`）；它的张量会按这些类型和形状检查，f16 条目通过 `allow_f16_reduction` 选择启用 f16 部分和。f32 操作数（以及 f32 结果）的 GEMM 按 f32 相乘，TF32 关闭：凡是库会把它按 TF32 或 bf16x9 模拟运行的地方（设置了 `NVIDIA_TF32_OVERRIDE=1` 或 `CUBLAS_EMULATE_SINGLE_PRECISION=1` 的进程），crate 都会拒绝其算法；f32 条目通过 `allow_f32_reduction`（`LibraryEntry::allow_f32_reduction`）选择启用在输出中累加 split-K 部分和。为设备 SM（流式多处理器）的一个 SM 分区 pin 的条目（`sm_count_target`）在每次启动时都设置该目标，运行在某个分区上的 host 用 `Manifest::with_sm_partition(Some(sms))` 声明这一点：为其他数量选出的条目会被拒绝。GEMM 可以在其 f32 累加器上带一个 cuBLASLt epilogue（尾部；`impl: "cublaslt.matmul"`：`LibraryEntry::epilogue`，取值为 `none`、`bias`、`relu` 或 `relu_bias` 的 `GemmEpilogue`，`beta` 为 1 时在它之前加上一个 `c [M, N]`）：启动接受 `a`、`b`、[`bias`]、[`c`]、`out` 和 workspace，每次检查和运行都会设置该 epilogue，未选择启用时 epilogue 不会使用 split-K，条目会说明对其 pin 的验证发现了什么（`LibraryEntry::contract`、`verification`），加载时会运行它的 canary（哨兵值；`LibraryEntry::canary`：pin 在由一个种子在设备上生成的操作数上的输出，在它自己的 stream 上运行）并拒绝其他字节，因为 cuBLASLt 没有可供比较的构建号（`Gemm::epilogue`、`has_c`、`contract`、`check_canary`）。库条目也可以是已 pin 的 cuDNN 前向卷积（`impl: "cudnn.conv_fwd"`、`Manifest::conv`、`Conv`、`ConvEntry`）：pin 是一个旧版算法枚举值或一个后端引擎配置（一个引擎及其调节参数），附带选出它的 cuDNN 版本；crate 拒绝其他 cuDNN 版本、其他 SM 数量以及设置了 `NVIDIA_TF32_OVERRIDE` 的进程，像 Python 一样审查 pin，运行条目的 canary（在由一个种子生成的操作数上的输出，在它自己的 stream 上运行）并拒绝其他字节，在每个 CUDA context 中让它的所有卷积共享一个 cuDNN 句柄，并且只为含有卷积的 manifest 打开 libcudnn。创建这个句柄（`cudnnCreate`）会等待设备上所有 stream 中排队的全部工作：在有工作运行时加载卷积的 host，应在启动阶段、排入任何工作之前调用 `CudnnHandle::new(&ctx)` 并保留它，这样任何加载都不会再创建句柄（cuDNN 引擎的首次运行，即加载时的 canary，也可能在 CUDA 加载其 kernel 时等待：同样在启动阶段加载它们）。按架构声明了各自实现的算子（`@tilefish.extern(by_arch=...)`），其条目会说明自己是哪一个实现（`Entry::implementation`：它自己的架构、`"fallback"`，或几个架构共享的某个实现的名字；可以是 Triton kernel 或预构建的二进制文件）；crate 用该条目自己的 ABI 启动设备架构对应的条目（同一算子在其他架构上的条目可能把张量作为 TMA 描述符而不是指针传入，或接受不同的标量），与任何算子一样。作为张量 TMA 描述符的参数（manifest kind 为 `tensordesc`，即带 `tensor_map` 的 `ParamKind::Pointer`）像指针一样传入其张量：每次启动（以及 plan 的每个步骤的每次执行）都会检查驱动会拒绝的情况（`PreconditionKind::TensorMap`），用 `cuTensorMapEncodeTiled`（cudarc 的 `sys`）编码该张量的 `CUtensorMap`，再连同张量的形状和步长按值传入，与 Triton 启动 host 端 `TensorDescriptor` 的方式相同；被 capture 的图保存的是它 capture 时那些地址的 map。CuTe DSL kernel（source 为 `cute` 的 `binary` 条目，`BinarySource::Cute`）的启动方式与其编译出的 host 函数启动它的方式相同：条目中由整数构成的 grid、block、cluster（线程块簇）和共享内存，它的张量作为指针或 DSL 编码好的 TMA 描述符（kind 为 `tensormap`，即带 `template` 的 `ParamKind::Pointer`：像指针一样传入张量，启动时传入模板，其前八个字节为张量的地址，共 128 字节按值传递；2^57 及以上的地址会被拒绝），以及它的常量字节（kind 为 `bytes`，`ParamKind::Bytes`，按 manifest 中的内容传入），单次启动、plan 步骤和被 capture 的图都是如此。warp（线程束）特化的 kernel（Gluon 的 `gl.warp_specialize`，或 Triton 自己的 warp 特化）在 `Entry::total_num_warps` 中记录其所有分区的 warp 数（与之并列的 `Entry::num_warps` 是其默认分区声明的 warp 数），每次启动、plan 的每个步骤和每个被 capture 的图都使用它们组成的 block（`Plan::block()`：条目的 `block`，即 `32 * total_num_warps`）。cudarc 按名字打开 cuBLASLt（先是 `libcublasLt.so`，对默认的 `cuda-13020` 绑定再试 `libcublasLt.so.13`，然后是更早的主版本）：如果 cuBLASLt 来自某个 Python wheel（`nvidia-cublas`，torch 用的那个），请把该 wheel 的 `nvidia/cublas/lib` 加入 `LD_LIBRARY_PATH`。缺少该库会报错，它的其他构建会被版本检查拒绝。

完整的一次前向计算由一个 **launch plan（启动计划）** 运行（`plan.json`，`tilefish.runtime.export` 为 `@tilefish.graph` 把它写在 manifest 旁边）：`LaunchPlan::load` 对它做验证（pin、各个 digest、[`docs/plan-schema-1.md`](https://github.com/Kilerd/tilefish/blob/main/docs/zh-CN/plan-schema-1.md) 的每一条规则、构建它所依据的 manifest、每个步骤与其条目是否相符），`instantiate` 加载它的二进制文件并分配它唯一的 arena（内存池：中间结果、设备标量、cuBLASLt workspace），`PlanRun::launch` 按名字绑定调用方的张量，在第一次启动前检查每个步骤，然后按顺序启动它们，不分配任何内存。`PlanRun::capture` 把它 capture 到一个 CUDA graph 中：`CapturedPlan` 拥有该 arena，并借用它 capture 的那些张量，因此它安全的 `replay` 不可能在已释放的内存上运行；两次 replay（回放）之间通过它写入新的输入。

```rust
let plan = LaunchPlan::load("build/", Pin::Digest("<plan digest>"))?;
let run = plan.instantiate_all(&stream)?;
run.launch(&stream, &mut [("x", x.into_arg()), /* the weights, */ ("out0", y.into_arg())], &[("n", DevScalar::Int(64))])?;
```

外部张量按 plan 缓冲区的名字给出：参数用它的名字，返回值用 graph（计算图）给它起的名字（Python 中的 `@tilefish.graph(outputs=("y", ...))`，`LaunchPlan::outputs`），或按 graph 返回它们的顺序用 `out0`、`out1`、……（`LaunchPlan::output_buffer`），因此在 graph 能为输出命名之前写的 host 仍然可用。如果 plan 的 graph 声明了各参数的角色（`@tilefish.graph(inputs=..., weights=...)`，`LaunchPlan::roles_declared`），它的权重只需绑定一次：`PlanRun::bind_weights` 检查权重并返回一个 `WeightedRun`，其 `launch`、`launch_tapped`、`check` 和 `capture` 只接受输入、`inout` 张量和输出（权重在 `WeightedRun` 的生命周期内被借用）。plan set（plan 集；多个 plan 与一个 manifest 一起导出，`plans.json` 是其索引）用一个 pin 加载：`PlanSet::load(dir, Pin::Digest(set_digest))`，然后对每个 `LaunchPlan` 调用 `set.plan(name)`，它们共享该集合的 manifest（克隆 `Manifest` 的开销很小，其克隆共享它们加载的模块）；本进程已经验证过且仍持有的集合，不会被第二次 `PlanSet::load`（比如每个 worker 一次）再读一遍，`set.plans(&names)` 在多个线程上加载多个 plan。host 一次只运行其中一个的那些集合内 plan 共享一个 arena：`set.instantiate(&stream, &["a", "b", "c"])` 是一个 `SetRun`，其 `run("a")` 是运行在一个 `SharedArena` 上的 `PlanRun`，该 arena 的大小为最大 plan 的字节数（每个 plan 使用自己的偏移；对选定的一组 bucket，用 `SharedArena::new` 和 `LaunchPlan::instantiate_in`），这些 plan 在多个线程上加载和实例化。其中任何一个的每次启动和 replay 都通过该 arena 的顺序事件与其他的排序（在该事件最近一次记录所在的 stream 上的启动不会等待它：只要它们留在同一个 stream 上，每次启动只有一次 `cuEventRecord`），所以它们在设备上永远不会重叠；每个 `SetRun`、`SharedArena` 和每次实例化都是一个独立的 arena，所以并发运行的 lane（通道）各取一个。plan 自己区域中的任何内容都不会在一次启动之后继续存在：一个阶段的输出要么是下一阶段的外部张量，位于调用方的内存中（arena 中的张量会被拒绝），要么是一个 **carried buffer**（携带缓冲区；`BufferRole::Carried`，graph 声明为 `carried=("h",)` 的参数），arena 把它放在各 plan 的区域之后（`CarriedLayout`、`SharedArena::carried`）：一个阶段写入它，下一阶段无需传入就能读取它；如果启动或 replay 在该 arena 上还没有任何启动写入它之前就读取它，或以另一种形状读取它，就会被拒绝。memset 步骤（`StepKind::Memset`，graph 中的 `tilefish.zeros_like`）用 `cuMemsetD8Async` 把缓冲区清零，在图中是一个 memset 节点。`PlanRun::launch_tapped` 是带 debug tap（调试观测点）的启动：每个 `Tap` 在某个步骤之后立即把 plan 的一个缓冲区复制到一个 `TensorMut` 中，plan 及其 digest 保持不变（`PlanRun::capture_tapped` 把它们 capture 为 memcpy 节点，每次 replay 都会再复制一次）。`PlanRun::launch_step` 在调用方的张量上启动某个 bucket 的一个步骤，使用已 pin 的 plan 的二进制文件和绑定（不能在正在 capture 的 stream 上）。

**插桩导出**（`tilefish.runtime.export(..., instrument=True)`，[`docs/instrument.md`](https://github.com/Kilerd/tilefish/blob/main/docs/zh-CN/instrument.md)）是一种诊断用的变体，它的二进制文件把其 kernel 中 `tilefish.probe("name")` 标记的阶段边界记录到通过 profile scratch 参数传入的 probe（探针）缓冲区中；它有自己的 digest，`Manifest::load`、`Manifest::from_embedded`、`LaunchPlan::load` 和 `PlanSet::load` 都会拒绝它。`Manifest::load_instrumented`（以及 `LaunchPlan::load_instrumented`、`PlanSet::load_instrumented`）读取它；`Kernel::launch_probed` 把一次启动记录到一个 `ProbeBuffer` 中（`Kernel::probe_bytes` 给出所需大小；`ProbeBuffer::collect` 读回记录），`LaunchPlan::instantiate_probed` 为 plan 中每个插桩的步骤分配该实例的 probe 缓冲区中的一块区域（无论是启动还是 capture），用 `PlanRun::collect_probes` 读取，`ProbeTrace::save` 为 `tilefish-explain DIR --probes FILE` 写出 `tilefish.probe-trace.v1`。

在任何 cudarc context 中，安全代码都不可能在正在运行的 kernel 或 GEMM 之下释放 slice，在正在 capture CUDA graph 的 stream 上启动 slice 也会被拒绝。每个安全启动都拒绝 per-thread 默认 stream（`ctx.per_thread_stream()`）：它的同一个句柄在每个线程中指向不同的 stream，因此在其上的启动无法与其他线程对 slice 的使用排序；请使用 `ctx.new_stream()` 或 `ctx.default_stream()`。恰好只有两处需要 `unsafe`：为一块原始设备内存区域作担保（这也是 capture CUDA graph 的方式，配合 `launch_raw`），以及在原始 `CUstream` 上启动。确切的契约见 crate 文档。

**在自己也使用 cudarc 的 workspace 中。** 这个 crate 需要 cudarc 0.19.10 或更高的 0.19 版本：它把 `CudaFunction` 持有的函数交给驱动（`CudaFunction::cu_function`，用于检查其参数大小、立即加载它以及在原始 stream 上启动它），并以不带标志的方式实例化被 capture 的图（`CUgraphInstantiate_flags(0)`），这两者都是 cudarc 0.19.10 新增的；用 0.19.9 无法编译它。cudarc 只为恰好一个 `cuda-*` feature 构建其绑定，而这个 crate 默认启用 `cuda-13020`：自己选择 feature 的 workspace 必须以同一个 feature 依赖本 crate，否则 cudarc 会带着两个 feature 构建，无法编译：

```toml
cudarc = { version = "0.19.10", default-features = false, features = ["driver", "cuda-13000"] }
tilefish = { version = "0.0.11", default-features = false, features = ["cuda-13000"] }
```

**host 的可复现构建。** 这个 crate 不嵌入任何自身的路径：它没有 build script，只在测试中使用 `env!` 和 `file!`。曾有两件事让同一个 host 在两台机器上的构建结果不同：

- **嵌入的 manifest**（`tilefish.hosts.rust.emit(..., embed=True)`，其 `include_bytes!` 路径相对于生成的文件）。直到 tilefish 0.0.6，cubin 的调试段都会写入构建它的目录，因此每次导出的字节不同，`MANIFEST_DIGEST` 也不同。tilefish 0.0.7 按模块路径命名源文件，因此用同一工具链为同一设备导出的相同算子，在任何机器上都是相同的文件。要以可复现的方式嵌入 0.0.6 或更早版本的 manifest，请重新导出它。
- **rustc 写入二进制文件的路径**：每个 panic 位置（`std::panic::Location`）的源文件，包括本 crate、cudarc 和 serde 的。workspace 成员的路径相对于 workspace 根目录，但来自 registry 的 crate 位于 `$CARGO_HOME`（默认是 `~/.cargo`）之下，其中含有用户名。请用相同的工具链、`Cargo.lock`、target 和 profile 构建，并重映射这些前缀：

  ```sh
  RUSTFLAGS="--remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo --remap-path-prefix=$PWD=/build --remap-path-prefix=$(rustc --print sysroot)=/rustc" \
    cargo build --release --locked
  ```

  安装了 `rust-src` 组件时，sysroot 前缀很重要：此时 rustc 会写入 std 源文件的本地路径。Cargo 的 `trim-paths` profile 选项有同样的作用，但 cargo 1.100 仍需 nightly 才能使用它。位于 workspace 根目录之外的路径依赖（从另一个 workspace 写 `tilefish = { path = "../tilefish" }`）无法重映射：cargo 会把它的绝对路径哈希进该 crate 的元数据，这会改变符号哈希和代码布局。请在 workspace 内保留一份 vendored 副本，或从 registry 依赖本 crate。

已在 Linux（cargo 1.100）和 macOS（cargo 1.99）上验证：一个嵌入 manifest 的 host，在两个路径不同、`CARGO_HOME` 不同的检出目录中构建，使用这些标志时二进制文件的 sha256 相同，不使用时则不同。

这个 crate 不开启 serde_json 的任何 feature，因此不会改变 host 自己的 JSON 的键顺序（若开启，cargo 会为整个构建开启 `preserve_order`）。自己开启了它的 host 对 manifest 或 plan 的读取结果相同：凡是对象键的顺序有意义的地方（条目的各个 bucket 按其顺序组成条目的 stem），crate 都从文件中取得该顺序。

可任选 Apache License, Version 2.0 或 MIT license 之一授权。
