# kernel 内插桩：probe、插桩变体、时间线

[English](../instrument.md) | **简体中文**

采样式 profiler 会把异步 `wgmma` 或 `tcgen05` MMA 的时间算到等待它的指令上；而对于外部的计时器，一个 warp（线程束）特化或持久化的 kernel（核函数）只是一次很长的启动（`tilefish.profile_plan` 按步骤计时，见 `docs/advise.md`）。这类 kernel 内部发生了什么，最好从 kernel 自身读出来：在作者指定的阶段起止位置记录时间戳。tilefish 用以下几部分做到这一点：

1. **probe**（探针，第 1 节）：写在 `@tilefish.kernel` 函数体、`@triton.jit` 或 `@gluon.jit` kernel 中的 `tilefish.probe("name")`。生产构建会把它们编译成空；
2. **插桩变体**（第 2 节）：`export(..., instrument=True)`，或 `jit` 中的 `tilefish.run(..., probes=tilefish.Probes())`，会把每个 kernel 连同它的 probe 以及 tilefish 自己的三个 probe（`@begin`、`@calib`、`@end`）一起构建，得到各自独立的二进制文件、manifest（清单）digest（摘要）和 plan（计划）digest；除非明确要求加载，host（宿主）会拒绝它们；
3. **记录**（第 3 节）：每个到达 probe 的 warp 由其 lane 0 把 `%globaltimer`、`%clock`、probe 标识和 `%smid` 写进 probe 缓冲区中属于它的槽位，这个缓冲区由 host 通过 kernel 的 profile scratch 参数传入；
4. **host 端工具**（第 4 节）：Python 和 Rust 在一次启动、一次 plan 运行或一次被 capture（捕获）的 replay（回放）之后收集缓冲区；`tilefish-explain DIR --probes FILE` 按 warp 角色呈现各阶段及其耗时和分布，按 SM（流式多处理器）、CTA（线程块）和角色呈现时间线，并生成 Chrome trace（Perfetto），附上每个 probe 的源码行。

代码在 `tilefish._probe`（probe、PTX 改写、缓冲区）、`tilefish._probes`（`Probes`、`ProbeTrace`、各个视图），以及 crate 中的 `crates/tilefish/src/probes.rs`。第 5 节说明为什么不用 Triton 的 Proton，第 6 节说明它的开销，第 7 节说明它的限制。

## 1. probe

```python
import tilefish

@gluon.jit
def _load(x, bufs, ready, empty, n_cols, BM: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr):
    m = gl.program_id(0) * BM
    for i in range(gl.cdiv(n_cols, BN)):
        s = i % STAGES
        tilefish.probe("load.wait")                       # the producer waits for a free slot ...
        mbarrier.wait(empty.index(s), ((i // STAGES) & 1) ^ 1)
        tilefish.probe("load.issue")                      # ... and issues the copy
        mbarrier.expect(ready.index(s), x.block_type.nbytes)
        tma.async_copy_global_to_shared(x, [m, i * BN], ready.index(s), bufs.index(s))

@tilefish.kernel
def scale_add(x: Tensor[N, f32], y: Tensor[N, f32]) -> Tensor[N, f32]:
    a = x * 2.0
    tilefish.probe("scaled")                              # between the multiply and the add
    return a + y
```

* **Triton 和 Gluon kernel**（extern（外部 kernel））像调用任何 builtin 一样调用 `tilefish.probe`：在 kernel 所在的模块中 `import tilefish`，然后把它写在 kernel 里，或写在 kernel 调用的 jit 函数里（Triton 会把它内联）。它是一个 Python 函数，Triton 的代码生成器会带着自己的 builder 调用它（它是一个 builtin，和 `tl.load` 一样）；导入 tilefish 既不会导入 Triton，也不会导入 torch。
* **在 `@tilefish.kernel` 函数体中**，probe 标记的是写在它之前和之后的算子之间的那个点。下沉（lowering）会在生成源码的这个位置写入 `_tf_probe("scaled")`（函数体中的行写在注释里，也写进 trace（追踪记录）的 `nodes.json`）；下沉的模板先加载输入、最后存储结果，所以 probe 位于加载之后、存储之前。graph（计算图）函数体中没有 probe（它没有自己的 kernel；插桩 plan 的每个步骤都会记录 `@begin` 和 `@end`）。
* **名称**长 1 到 64 个字符，以字母或 `_` 开头，其后是字母、数字和 `_ . : / -`（`load.wait`、`mma/issue`）。同一个名称可以写在多个地方（两个分支、两个分区）：它们是同一个 probe。
* **阶段**是同一个 warp 的两条相邻记录之间的时间，命名为 `"a -> b"`。probe 记录的是它的 warp *发射*它的时刻：要给异步拷贝或 MMA 计时，就像上面那样把 probe 放在等待它的地方（`mbarrier.wait`）前后，而不是放在发射它的地方前后。

**在生产构建中，probe 什么都不是。** Triton 前端不会为它生成任何操作，所以 IR、IR 的 digest 和机器码都与不含这次调用的 kernel 相同（在 sm_103 上，`probe_kernels.ws_scale_p` 的生产二进制文件与 `ws_kernels.ws_scale`（同一个 kernel，只是没有写 probe）的 `sha256_code` 相同）；只有在源码行发生移动的地方，行号表才随之变动。不含 probe 的 kernel 的导出，与引入这个功能之前的文件完全相同（已在整个 Rust 门禁的导出上检查，第 8 节）。

**在你自己写的 extern 上启用它**（比如一个私有的 SIMT attention）：

1. 在你关心的边界放上 `tilefish.probe("...")`（流水线中各个等待的前后，持久化循环中一个 tile 的开始和结束）；
2. 在 `jit` 中迭代：`rec = tilefish.Probes(); tilefish.run(attn, q, k, v, probes=rec); print(rec.trace().summary_text())`；或者在生产导出旁边导出插桩变体 `tilefish.runtime.export(ops, "build/inst", instrument=True, trace=True)`，再用 `probes=rec`（Python）或 `Manifest::load_instrumented`（Rust）回放它；
3. `rec.trace().save("attn.json")`，然后 `tilefish-explain build/inst --probes attn.json`（可加上 `--timeline role`、`--chrome attn.trace.json`）。

其他都不变：声明、ABI、plan 和生产导出都和原来一样。

## 2. 插桩变体

```python
runtime.export([attn, mm], "build/inst", instrument=True)                          # every Triton, Gluon and lowered binary
runtime.export(block, "build/inst", instrument=tilefish.instrument(capacity=1024, ops=["attn"]))   # a graph; only attn's binaries
rec = tilefish.Probes(tilefish.instrument(capacity=512)); tilefish.run(op, *args, probes=rec)      # jit: built in memory
```

`instrument=True` 就是 `tilefish.instrument()`：`capacity` 是每个 warp 每次启动的记录数（默认 256），`ops` 是要插桩的算子（默认全部；预编译 PTX 和 cubin 的 extern 以及 library op（库算子）从不插桩）。每个二进制文件的构建：

* 在打开 probe 的情况下运行前端，因此 IR 中包含它们（每个都是一条内联汇编语句），其 digest 也是这份 IR 的；
* 在 Triton 的 `ptx` 和 `cubin` 阶段之间改写入口函数的 PTX（通过包装 `CUDABackend.add_stages` 实现，只在构建插桩变体的线程中生效）：先是 probe 用到的寄存器和该 warp 在缓冲区中的槽位，然后是 `@begin`；在每个 CTA 的 warp 0 中，用同样的指令记录两次 `@calib`；在每条 `ret` 之前记录 `@end`，连同该 warp 的记录数（带谓词的 `ret` 会以分支绕过它）；
* 每次都重新编译：插桩变体从不读写构建缓存，所以插桩二进制文件永远不会被交给生产构建，反之亦然。TF32 防护和收缩检查会判定它；声明中的 SASS 断言不会（它们针对的是生产二进制文件：probe 会多用几个寄存器和几条指令）。

**有自己的键和 digest。** 它的条目带有 `instrument`（manifest schema（模式）3 第 4.8 节：记录格式、容量和 kernel 的 probe 表：id、名称，以及每个位置的模块相对行号、所在分区和 jit 函数）；这些条目的二进制文件、`sha256`、`sha256_code` 和 `source_digest`（对含 probe 的 kernel）都是插桩变体自己的，所以插桩变体的 manifest digest 以及每个 plan 和 plan set（plan 集）的 digest 也都是它自己的。声明 digest 和 ABI 与生产条目相同：probe 缓冲区就是每个 Triton kernel 本来就有的 profile scratch 参数（它两个 scratch 槽中的第二个，在生产启动中为空）。

**它永远不会被当成生产产物。**

* Python：除非给出 `instrumented=True`，`Manifest.load` 拒绝带插桩条目的 manifest；除非给出 `probes=`，`tilefish.run(..., mode="replay")` 和 `tilefish.capture` 拒绝它（对生产导出则拒绝 `probes=`）；插桩导出的 plan 同样如此；`tilefish.hosts.rust.emit` 和 `tilefish.hosts.cpp.emit` 拒绝它（它们写的是生产 host）。
* Rust：`Manifest::load` 和 `Manifest::from_embedded` 拒绝它；`Manifest::load_instrumented` 接受它（`LaunchPlan` 同样如此）。
* 导出从不把两者合并：把插桩变体写进存有生产 manifest 的目录（或者反过来）会被拒绝。
* 早于它的读取方：crate 0.0.8 及更早版本拒绝这个条目字段（它们拒绝一切未知字段）；tilefish 0.0.10 及更早版本的 replay 拒绝含 probe 的 kernel（它的 IR digest 与进程算出的不同）。没有 probe 的 kernel 在那里可以加载；旧 host 启动它时传入空的 profile scratch，每个 probe 都会跳过存储（槽位指针为空），所以它照常运行，写出生产字节。

## 3. probe 缓冲区与记录

一次启动的缓冲区为每个 CTA 的每个 warp 提供一个槽位，共 `ctas * warps * (16 + 16 * capacity)` 字节，16 字节对齐；其中 `ctas` 是实际启动的 CUDA grid（cluster 中的 CTA 都计入，持久化 grid 按裁剪到设备后的大小计），`warps` 是 block 的 warp 数（warp 特化 kernel 取 `total_num_warps`）。线性索引为 `c = x + y * nx + z * nx * ny` 的 CTA 中 warp `w` 的槽位位于 `(c * warps + w) * (16 + 16 * capacity)`：

| 槽内偏移 | 类型 | 内容 |
|---|---|---|
| 0 | u32 | `0x31706674`（字节 `tfp1`），在 warp 进入时写入 |
| 4 | u32 | CTA 的线性索引 |
| 8 | u32 | warp 编号（`%tid.x / 32`） |
| 12 | u32 | 该 warp 尝试写入的记录数，在其 `@end` 时写入（超过容量：多出的被丢弃） |
| 16 + 16 k | u64 | 记录 `k`：`%globaltimer`（ns） |
| 24 + 16 k | u32 | `%clock`，SM 的周期计数器，低 32 位 |
| 28 + 16 k | u16 | probe：0 为 `@begin`，1 为 `@end`，2 为 `@calib`，3 及以上为 kernel 自己的 probe，按前端遇到它们的顺序编号 |
| 30 + 16 k | u16 | `%smid` |

由每个 warp 的 lane 0 写入；一个 warp 保留它的前 `capacity` 条记录，其余的只计数（`tilefish.instrument(capacity=...)` 可以保留更多）。缓冲区为空时，每个 probe 都跳过存储。头部不完整的槽位（没有运行的 warp、属于另一个 kernel 的缓冲区）列为缺失。

**缓冲区从哪里来。** 单次启动：`Probes` 每次启动分配一个，或者记录到调用方的缓冲区里（`tilefish.Probes(buffer=t)`，一个 CUDA `uint8` 张量）。plan：插桩 plan 的每个 bucket（分桶）都有一个 probe 区域，与 arena（内存池）一样，在任何 capture 之外随该 bucket 的二进制文件一起分配；其中每个插桩步骤都有一段 256 字节对齐的切片，大小按该步骤在该 bucket 下的 grid 确定；被 capture 的 plan 持有它的地址，所以每次 replay 都记录到其中；给出调用方缓冲区时，由调用方缓冲区取代它。这个区域属于 host，不是 `plan.json` 中的缓冲区：它的大小取决于设备（持久化 grid 每个 SM 一个 program）和插桩变体，plan schema 保持不变，而没有 `plan.json` 的 `jit` 运行也使用同样的布局。Rust host 的做法相同（第 4 节）。

## 4. 收集与解释

```python
rec = tilefish.Probes(tilefish.instrument(capacity=1024))
y = tilefish.run(op, *args, probes=rec)                                           # jit: the variant is built in memory
y = tilefish.run(op, *args, mode="replay", manifest="build/inst", pin=digest, probes=rec)
y = tilefish.run(graph, x, probes=rec)                                            # a plan: one launch per instrumented step
cap = tilefish.capture(graph, x, mode="replay", plan="build/inst", pin=digest, probes=rec)
cap.replay(); cap.collect()                                                       # the records of the last replay
trace = rec.trace()                                                               # a ProbeTrace of rec.launches
print(trace.summary_text()); print(trace.timeline_text(by="role"))
trace.save("run.json")                                                            # tilefish.probe-trace.v1
```

每次启动之后（或 `collect()` 时），缓冲区会被读回（这会等待 stream），并解码成一个 `Launch`：stem、架构、算子、grid 和 block、plan 的 graph、步骤和 bucket、manifest 的 digest、条目的 probe 表、每个 warp 的记录以及缺失的槽位。文件 `tilefish.probe-trace.v1` 保存这些启动（时间从该次启动的第一个 `%globaltimer` 起算，这个值是十进制字符串：JSON 数字放不下它）；Rust host 写出同样的文件。

**Rust**（`crates/tilefish/src/probes.rs`）：

```rust
let manifest = Manifest::load_instrumented("build/inst", Pin::Digest(digest))?;      // Manifest::load refuses it
let kernel = manifest.kernel(&ctx, "attn", &[], arch)?;                               // kernel.instrument(): its capacity and probes
let prepared = kernel.check(&tensors)?;
let mut probes = ProbeBuffer::new(&stream, kernel.probe_bytes(&prepared).unwrap() as usize)?;
kernel.launch_probed(&stream, &mut tensors, &mut probes)?;                            // the buffer in the profile scratch slot
let launch = probes.collect()?;                                                       // a ProbeLaunch, decoded after the stream

let plan = LaunchPlan::load_instrumented("build/inst", Pin::Digest(plan_digest))?;    // LaunchPlan::load refuses it
let mut run = plan.instantiate_probed(&stream, &[("M", 128)])?;                       // a probe region with the run, per bucket
run.launch(&stream, &mut externals, &devscalars)?;                                    // or capture(...) and replay(&stream)
ProbeTrace { launches: run.collect_probes()? }.save("run.json")?;                     // tilefish.probe-trace.v1
```

`Manifest::is_instrumented`、`Entry::instrument`（`Instrument { capacity, probes }`）、`Kernel::launch_probed_with`（给定参数）、`LaunchPlan::instantiate_all_probed`、`PlanSet::load_instrumented` 和 `PlanRun::probed` 补齐了其余部分；生产二进制文件、属于另一个 context 的缓冲区和太小的缓冲区都会被拒绝，`instantiate_probed` 拒绝生产导出的 plan。C ABI（`tilefish-capi`）没有 probe 相关的调用：它通过 `Manifest::load` 加载，所以会拒绝插桩导出。

```console
tilefish-explain DIR --probes FILE [KERNEL]           # the phases of each launch (of KERNEL: a stem or an op)
  --timeline sm | cta | role                          # per SM its CTAs; per CTA every warp's phases; per role one warp each
  --chrome OUT                                        # a Chrome trace: chrome://tracing, ui.perfetto.dev
  --json --limit N --source-root PATH
```

`probe_kernels.ws_scale_p` 中生产者和消费者的 summary（sm_103，4 个 CTA，每个 8 个 warp；`tilefish-explain build/inst --probes ws.json`）：

```
launch 0: ws_scale_p on sm_103, grid (4, 1, 1), block (256, 1, 1)
  span 17.25 us (first record to last), 4 CTAs on 4 SMs, 32 warps (12 ran none of the kernel's probes: not in the phases)
  records 2632 kept, 0 dropped (capacity 256 per warp); SM clock 2.023 GHz; a probe costs a warp 53 cycles (26 ns); the probes' share of a warp's
  time at most (records times that cost, median over the warps): 19.7%
  phase (role)                                   count     median       mean        p10        p90        max   share
  compute.work -> compute.wait (default)           1008     202 ns     202 ns     199 ns     205 ns     223 ns   59.3%
  compute.wait -> compute.work (default)           1024      53 ns      55 ns      48 ns      54 ns     284 ns   16.5%
  load.wait -> load.issue (worker 0)                256     197 ns     188 ns     197 ns     198 ns     236 ns   14.1%
  load.issue -> load.wait (worker 0)                252      56 ns      56 ns      56 ns      56 ns     109 ns    4.1%
  @begin -> compute.wait (default)                   16     491 ns     483 ns     443 ns     511 ns     514 ns    2.3%
  ...
  probes:
    compute.wait             probe_kernels.py:62 [default]   tilefish.probe("compute.wait")
    load.wait                probe_kernels.py:49 [worker 0]   tilefish.probe("load.wait")
    ...
```

以及 `--timeline role --limit 1`，即第一个 CTA 中每个角色各一个 warp：

```
  CTA 0 on SM 28: 0.000-17.248
    warp  0 default     0.064: compute.wait +444 ns | compute.work +280 ns | compute.wait +207 ns | compute.work +54 ns | ...
    warp  4 -           0.000: @end +17.21 us
    warp  6 worker 0    0.000: load.wait +490 ns | load.issue +47 ns | load.wait +51 ns | load.issue +42 ns | ...
```

每个 tile 中，生产者在它的 253 ns 里有 197 ns 在等空闲槽位，消费者在它的 255 ns 里等 53 ns：流水线受消费者限制（它的计算和 TMA 存储），两者以同一速率轮转，正如处于稳态的流水线那样。

* **耗时**是 `%clock` 的周期数，按该次启动测得的时钟频率换算（在 2 us 及以上的区间内，各 warp 每纳秒 `%globaltimer` 对应周期数的中位数）；**起始时刻**取 `%globaltimer`，所有 SM 共用一条时间轴。`@calib` 给出一个 probe 让一个 warp 付出的周期数（两条背靠背记录中的第二条，此时其代码已在指令缓存中）；"the probes' share of a warp's time" 是记录数乘以这个开销再除以该 warp 的时间跨度，是一个上界（在反正要等待的 warp 中，probe 的开销更小：第 6 节）。
* **角色**是 Gluon `gl.warp_specialize` 的分区，取自前端 IR：`default` 和 `worker N`（以及各自运行的 jit 函数）；分区之外的 probe 在 default warp 上运行；没有分区的 kernel 只有一个角色 `""`。一个 warp 的角色就是它记录的 probe 的角色；没有记录 kernel 任何 probe 的 warp（没有 probe 的分区，或 Triton 为把分区凑整到 warp group 而添加的 warp：4 + 1 个 warp 的 `ws_scale` 以 8 个 warp 启动，它的生产者在 warp 6 上）标为 `-`，不计入各阶段。
* **源码行**：表中的每个 probe 都有它的模块相对行号（`kernels/attn.py:88`、`tilefish_generated/scale_add.py:26`）；有导出的 trace（`trace=True`，`docs/explain.md`）或 `--source-root` 时，summary 会打印这一行，对下沉生成的 kernel 还会打印它写在函数体中的那一行。

## 5. 为什么不用 Proton

Triton 3.5.1 带有 Proton 的 kernel 内插桩（`triton.profiler.language.enter_scope` / `exit_scope`、`proton.start(backend="instrumentation")`）。在编写本功能之前，已在同样的 kernel 上评估过它：

* 它把记录保存在 **shared memory** 中，在每个 CTA 结束时再拷出：普通循环 kernel 多用了 8 KB shared memory，`ws_kernels` 中的 tcgen05 矩阵乘从 131 260 字节增加到 196 800 字节（缓冲区占满 kernel 用剩的 SM 容量），这会改变 occupancy，对接近上限的 kernel 会失败；由线程 0 完成的拷出会拉长每个 CTA；
* 它的记录**只有 SM 的时钟**（43 位），没有 `%globaltimer`：不同 SM 上的 CTA 无法放到同一条时间轴上（它的 Chrome trace 让每个 CTA 都从 0 开始，并假定 1 GHz）；
* 它是**进程级状态**（前端读取的全局标志 `CUDABackend.instrumentation`、被打补丁的 `JITFunction.run`），而 tilefish 的构建有理由拒绝这种状态（`_refuse_unidentified`：二进制文件将不是从标识它的那份 IR 编译出来的），并且 Triton kernel 默认关闭（只支持 Gluon）；
* 它的缓冲区格式是 Triton 自己的，由 `libproton` 中的 C++ 读取：Rust host 需要为每个 Triton 版本移植一份；
* 它每条记录的开销大致相同（紧凑循环中 24 ns，这里的 probe 是 21 ns）。

从 Triton 沿用的是：profile scratch 槽位（所以 ABI 和每个 host 的参数列表保持不变）、内联汇编，以及它的阶段流水线（PTX 阶段是被包装，而不是被替换）。

## 6. 开销

在一个 B300 MIG 切片上实测（sm_103，30 个 SM，Triton 3.5.1，ptxas 12.9），每个 kernel 放在只有一个步骤的 CUDA graph（CUDA 图）中，分别带 probe 和不带 probe 进行 capture，各回放 50 次（`test_gpu_instrument.py::test_what_the_probes_cost`，加 `-s` 会打印结果）：

| kernel | 生产 | 插桩 | 每 warp 记录数（最多） | 一个 warp 中一个 probe 的开销（`@calib`） |
|---|---|---|---|---|
| `ws_scale_p`（Gluon 生产者和消费者，1024 x 4096 f16） | 20.6 us | 19.3 us（-6%） | 132 | 53 个周期 |
| `persistent_scale_p`（每个 SM 一个 program，8192 个 tile） | 144.5 us | 154.8 us（+7%） | 552 | 58 个周期 |
| `scale_add_p`（`@tilefish.kernel`，4096 个短 CTA） | 29.5 us | 31.7 us（+8%） | 6 | 77 个周期 |
| `ws_matmul_p`（tcgen05，512 x 512 x 8192 f16） | 28.8 us | 30.8 us（+7%） | 259 | 58 个周期 |

* **一个 probe** 大约是十条 SASS 指令（读 `%globaltimer` 和 `%clock` 的 `CS2R`、读 `%smid` 的 `S2R`、地址计算、从 lane 0 发出的带谓词 `STG.128`、计数），让一个 warp 付出 45 到 80 个周期的延迟（它的存储要等特殊寄存器）；在 warp 反正要等待的地方（在 `mbarrier` 后面），开销更小，这就是生产者和消费者 kernel 没有变慢的原因。短 CTA 要承担固定部分的开销（槽位、`@begin`、`@end` 和计数；`@calib` 只在 warp 0 中）。
* **资源**：不用 shared memory，没有栈帧；寄存器：`scale_add_p` 从 26 个增加到 30 个，其他不变（warp 特化 kernel 已经处于其 `maxnreg`）；代码：多出约五分之一（`ws_scale_p` 的 cubin 从 28 040 字节增加到 35 688 字节）。
* **内存**：缓冲区，每个 warp `16 + 16 * capacity` 字节（默认 256 时为 4.1 KB）。

## 7. 限制

* **两个时钟。** `%clock` 计数一个 SM 的周期（保留 32 位：在 2 GHz 下每 2 秒回绕一次；在同一个 warp 内，视图会展开回绕）：粒度细，但永远不要比较两个 SM 的时钟。`%globaltimer` 是整个 GPU 共用的一个以纳秒计的时钟，分辨率由驱动决定：在 B300 上步长为 32 ns（一次启动记录到的每个步长都是 32 的倍数），所以起始时刻按它量化；耗时则来自 `%clock`。
* **probe 看到的**是它的 warp 发射它的时刻，取决于 `ptxas` 如何调度它周围的指令（`tilefish-explain` 会在 probe 的行下显示它的 SASS）。它看不到异步工作何时完成：要给等待计时。
* 由每个 warp 的 **lane 0** 记录：位于 lane 0 不走的路径上的 probe 不会被记录；没有到达 `ret` 的 warp（trap）不写计数，所以它的记录不会被读取。每条记录都带有 `%smid`（被抢占的 CTA 可能会迁移）。
* **Triton（非 Gluon）kernel**：编译器可能移动或复制 probe 周围的代码（软件流水线、它对 `tl.range(..., warp_specialize=True)` 循环自行做的 warp 特化），而且它生成的分区没有角色（前端 IR 中没有）：只能按 id 区分 warp。
* **`noinline` jit 函数中的 probe** 会被拒绝（它要记录到 kernel 入口函数的寄存器中）。
* **预编译 PTX 或 cubin 的 extern 以及 library op**（cuBLASLt、cuDNN）不插桩：这类 plan 步骤不记录任何内容。
* **容量**：一个 warp 保留它的前 `capacity` 条记录；summary 会说明丢弃了多少条。

## 8. 如何测试

* `python/tests/test_instrument.py`（任何机器）：Triton 入口函数上的 PTX 改写（槽位、`@begin`、`@calib`，以及带谓词的和普通的 `ret` 之前的 `@end`；它拒绝什么），probe 生成的 PTX，warp 特化 kernel 前端 IR 中的 probe 表（角色、函数、行号），名称和设置，缓冲区解码（缺失的槽位、丢弃的记录），阶段，耗时，probe 的开销和空闲 warp，trace 文件及其视图，`tilefish-explain --probes`，kernel 函数体中的 probe 及其下沉（不含 probe 的 kernel 下沉结果与以前相同），以及使用假编译器的插桩变体（它的条目、它的二进制文件、从不缓存、`Manifest.load`、replay 和各 emitter 的拒绝、合并的拒绝、`ops=`）。
* `python/tests/test_gpu_instrument.py`（需要设备；已在 sm_103 上运行，sm_120 的运行属于之后的门禁，那里会跳过其 tcgen05 测试）：第 6 节的测量；含 probe 的 kernel 的生产二进制文件与不含 probe 的对照（相同的 `sha256_code`）；插桩变体的输出与生产字节对照（jit 和 replay）；空缓冲区；生产者和消费者；tcgen05 流水线（它的 epilogue（尾部）等待的时间与主循环运行的时间一样长）；持久化 kernel（每个 program 在自己的 SM 上，遍历自己的 tile）；逐元素 kernel；plan（jit、插桩导出的 replay、被 capture 的 replay）；以及在真实 trace 上运行 `tilefish-explain`。
* Rust host 的测试：`crates/tilefish/tests/instrument.rs`（任何机器：该字段只由 `load_instrumented` 读取，第 4.8 节的每种格式错误都被拒绝，golden plan 和 plan set 的插桩副本被 `load` 拒绝），`probes.rs` 的单元测试（解码器、名称、trace 文件），以及 GPU 门禁的插桩用例（`python/tests/rust_gate_probed.py`、`crates/tilefish/tests/gpu_instrument_gate.rs`）：Rust host 分别在有 probe 缓冲区、没有缓冲区和缓冲区太小的情况下启动带 probe 的 kernel，并启动、capture 和回放插桩 plan；`rust_gate.py compare` 把字节与生产结果对照，把记录与 Python 的逐个 warp 对照（id、计数、角色）。
* 生产路径：本版本在 `TILEFISH_TRACE=1` 下为整个 GPU 门禁做的 Rust 门禁导出，与 0.0.10 不带 trace 的导出对照：3650 个文件、184 个 cubin、106 个 manifest、plan 和 plan set 的 digest，全部相同（门禁的插桩用例是新增的，放在单独的目录里）。
