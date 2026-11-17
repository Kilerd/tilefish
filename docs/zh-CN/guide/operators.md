# 视图、设备标量、预构建二进制文件和已 pin 的 GEMM

[English](../../guide/operators.md) | **简体中文**

一个真实的模型需要四样东西，而一个稠密的、编译期写死的、只用 Triton 的算子表达不了它们。每一样都是 manifest（清单）的一部分，manifest 是 schema（模式）3（`docs/manifest-schema-3.md` 规定了每个字段；Python 包只读写 schema 3，遇到 schema 2 会提示 "re-export"）。

**带步长的视图。** 张量类型有以元素计的步长和以元素计的偏移，用维度代数表示（`TensorType.strides`、`.offset`）；默认是稠密且 16 字节对齐（`contiguous`）。`x.reshape(M, 3, H, D)`（对稠密输入）、`x.permute(1, 0, 2)`、`x[:, 1]` / `x.select(axis, i)` 和 `x.split(...)` 都不复制任何东西：它们是视图，具有该操作对应的步长和偏移。块数取决于某个维度的 split（`x: [M, C]` 且 `C = Dim("C", divisible_by=16)` 时的 `x.split(16, axis=1)`）返回 `Chunks`，一个长度为符号的序列：`chunks[0]` 和 `chunks[-1]` 总是存在，其他块只在该维度的最小尺寸能证明其存在时才存在，`len()` 和解包会抛出异常。步长或偏移可以是符号尺寸的乘积：`[H, L, L]` 张量沿第一个轴的步长是 `L*L`，`x[h]` 的偏移是 `h*L*L`（`tilefish.dims.mul`；形状仍是仿射的，`L * L` 在形状中依然被拒绝，只有启动 grid 可以让维度相乘），`Strided[...]` 参数会绑定它，plan（计划）会在每个 bucket（分桶）上对它求值，所以注意力分数 `[H, L, L]` 是 plan 的一个缓冲区，它的转置和各个 head 都是它的视图。视图是 `View[...]`，可以进入被追踪的逐元素运算和归约运算；声明的算子只在其参数注明时才接受视图。kernel（核函数）、extern（外部 kernel）、library op（库算子）或 graph（计算图）的其他所有张量参数（以及结果）都是连续的：注解如果是带有其他步长、带偏移或没有对齐的 `TensorType`，会在声明时被拒绝；对这样的参数，启动时会拒绝不连续的张量。`extern` 参数 `x: Strided[Tensor[S, H, D, bf16]]` 接受一个视图，kernel 的步长是 stub 按 kernel 参数顺序声明的标量参数：

```python
@tilefish.extern(triton=head_copy, grid=lambda S: (S, 8))
def head_copy(x: Strided[Tensor[S, 8, 64, bf16]], y: Out[Tensor[S, 8, 64, bf16]],
              x_s0: StrideOf("x", 0), x_s1: StrideOf("x", 1), D: const = 64): ...      # the last axis has stride 1

y = tilefish.run(head_copy, qkv[:, 1])        # a view of a fused qkv tensor: its pointer and its two strides go to the kernel
```

如果一个 kernel 用**一对**步长读取同一个缓冲区的多个视图（融合的 `[B, 3 * D]` 张量中 q、k、v 三个列块），它就为它们一次性声明步长：`s_row: StrideOf(("q", "k", "v"), 0)`。这些视图共享这个步长维度，所以一次调用从 `q` 绑定它，并拒绝不具有该步长的 `k` 或 `v`（追踪中的调用、Python 和 Rust 中的启动、plan 的每个步骤都如此），kernel 得到一次步长和每个视图一个指针。

没有 `StrideOf` 的轴必须是最后一个轴（它的步长此时为 1，不满足这一点的视图会被拒绝）；步长在调用时像形状一样绑定，按 `i32`（`index_bits=64` 时为 `i64`）做范围检查，带步长的指针以其跨度（`max_extent`）而不是元素个数为界。步长标量的类型就是偏移的宽度，除非 `StrideOf` 另有说明：一个接受 `i32` 步长、并用它们计算 64 位偏移（`row.to(tl.int64) * s_row`）的 kernel，声明 `index_bits=64` 和 `s_row: StrideOf("x", 0, dtype=tilefish.i32)`，这时启动会把步长限制在 `i32` 范围内，而对视图的跨度不做限制。传入的指针是 `base + offset * itemsize`。`Strided[Tensor[...], 16]` 按 16 字节对齐的指针编译 kernel（`tt.divisibility=16`），这时偏移必须能被证明允许这一点（追踪中的调用拒绝不是 16 字节倍数的偏移，启动时检查地址）；没有 `16` 时 Triton 得不到提示，任何按元素对齐的视图都能运行。下沉（lowering）生成的 `@kernel` 仍然只接受连续张量：kernel 函数体中的视图会被拒绝，并给出说明这一点的规则。

**传给下沉生成的 kernel 的、不从 16 字节边界开始的视图。** 下沉生成的 kernel 的每个指针都按 16 字节对齐编译，所以一个不能证明从 16 字节边界开始的稠密视图，比如 `R` 为奇数的 `[K, R]` int32 平面中的一行 `plane[k]`，就不是连续张量。这样的视图传给只读参数时，运行的是 kernel 针对它的**特化版本**：`kernel.unaligned("a")`，即同一个 kernel，但 `a` 只按其元素对齐编译（它没有 `tt.divisibility`，有一个等于其元素大小的 `pointer_alignment`，每次启动都会检查）。它是一个独立的算子，在凡是涉及二进制文件的地方都与原 kernel 分开标识（它的名字 `add_rows__ua0` 取自这些参数的下标，它的文件名主干、声明 digest（摘要）、Triton IR 以及由此而来的构建缓存键、manifest 条目），它生成的源码与原 kernel 相同：

```python
@tilefish.graph
def plane_sum(plane: Tensor[3, R, i32]):
    s = add_rows(plane[1], plane[0])        # plane[1] starts 4 * R bytes in: the step launches add_rows__ua0
    return add_rows(s, plane[2])            # plane[2] starts 8 * R bytes in: add_rows__ua1

y = tilefish.run(add_rows, plane[1], plane[0])   # an address that is not 16-byte aligned runs add_rows__ua0 too
```

graph 的导出包含其 plan 的步骤所启动的特化版本；`tilefish.runtime.export(add_rows.unaligned("a"), directory)` 为单独的 kernel 写出一个特化版本，对缺少它的 manifest 做 replay（回放）时会指出这一点（`docs/manifest-schema-3.md`，第 6 节）。

**设备标量。** `n_tok: DevScalar[i32]` 是设备内存中的一个元素：在 ABI 中是一个形状为 `[]` 的指针，在 API 中是一个只有一个元素的张量。二进制文件用 `tl.load` 读取它，所以一个二进制文件适用于所有取值，改变取值靠写内存，而绝不靠构建；下沉生成的 kernel 用 `tilefish.load(s)` 读取它（一次 `tl.load(ptr)`，再广播）。

**只有 `Out` 张量才有的维度。** extern 可以写出一个尺寸不属于任何输入的张量：`key_mask(mask: Out[Tensor[B, bf16]], n_valid: DevScalar[i32], b: B)` 仅凭一个设备标量就生成一个 `[B]` 的掩码。启动时从传给它的 `Out` 张量绑定 `B`（`tilefish.run(key_mask, n, outputs=mask)`；Rust 的启动和每个 plan 步骤都有自己的 `Out` 张量），并像检查任何维度一样检查它，包括 bucket。在 graph 中，调用把 `B` 绑定到 graph 自己的 `B`（同一个 `Dim`），所以按 bucket 划分的 plan 决定掩码的大小，步骤再从那个缓冲区绑定它。

**预构建的二进制文件。** `@tilefish.extern(ptx="k.ptx", symbol="k", block=(256,), grid=...)` 或 `cubin={"sm_103": "k.cubin"}` 按 C ABI 声明一个不由 Triton 编译的 kernel：依次是指针、`Dim` 标量和类型化的 `const[i32|i64|f32]` 固定参数，并且**没有 Triton 的 scratch 槽位**。它的身份是文件字节的 sha256（kind 为 `tilefish.binary.v1` 的 `source_digest`）；PTX 由 `ptxas` 用固定且有记录的参数汇编（即 Triton 传的那些参数，所以一份 Triton 构建产出的 PTX 以这种方式声明时，得到的机器码和输出字节与走 Triton 路径时相同）；cubin 在其本身的架构上原样使用，在其他架构上被拒绝。`replay` 拒绝不是构建 manifest 时所用的那个文件。

预构建二进制文件的一个**没有默认值**的 `const[i32]` 或 `const[i64]`，是一个*每次调用时给出*的固定参数：即没有任何张量决定的调用几何信息（输出尺寸、偏移）。一个声明和一个二进制文件适用于所有取值：`k(x, h_out=5)`、`tilefish.run(k, x, h_out=5)`、Rust 的 `Kernel::launch_with(stream, tensors, &[("h_out", 5)])`；在 graph 中，取值可以是 graph 维度的表达式（`h_out=H // 2`），plan 的每个步骤都会记录它在自己 bucket 上那次调用的取值（`"given": {"h_out": 5}`），所以这些值在 plan 的 digest 中，两个 host（宿主）都会对照条目检查它们（`docs/manifest-schema-3.md`，第 5.3 节）。

**空指针和 64 位维度标量。** extern 的 kernel 在所声明的用法下不使用的指针（一个只有非空时才读取的 bias，或只在某个被声明关闭的 constexpr 开关下才读取的 bias）声明为 `Null[dtype]`：调用时不传它，两个 host 都在它的位置上传空指针（manifest kind `null`，第 5.6 节）。维度标量是 `i32`，除非声明为 `i64[n]`：同一个标量以 `i64` 传入，其值仍由形状推出并对照形状检查，用于计数宽度为 64 位的 kernel（第 5.7 节）。两者合用：

```python
@tilefish.extern(ptx="scale_bias.ptx", symbol="scale_bias_f32", block=(256,), grid=lambda n: (tilefish.cdiv(n, 256),), index_bits=64)
def scale(x: Tensor[n, f32], bias: Null[f32], out: Out[Tensor[n, f32]], n_elem: i64[n]): ...    # (const float*, const float*, float*, long long)
```

**Gluon kernel。** 用 Triton 的 Gluon 方言（`triton.experimental.gluon`：`@gluon.jit`，布局显式写出，所以 kernel 固定了哪个线程持有哪个元素，从而也固定了其归约的顺序）编写的 kernel，用 `gluon=` 代替 `triton=` 声明，其他一切都相同：带步长的视图和共享步长、`const` 参数、bucket、`index_bits`、`enable_fp_fusion`、plan、`jit`、`replay` 和 Rust host。

```python
@tilefish.extern(gluon=my_gluon_kernels.row_sum, num_warps=4, grid=lambda R: (tilefish.cdiv(R, 4),), index_bits=64)
def row_sum(x: Strided[Tensor[R, 256, f32]], y: Out[Tensor[R, f32]], s_row: StrideOf("x", 0, dtype=tilefish.i32), n_rows: R, C: const = 256): ...
```

声明要说明由哪个前端编译 kernel，因为编译规格是一个二进制文件的完整配方，它由声明生成，不需要导入 kernel：`"language": "gluon"` 在规格中，也在声明 digest 中，构建时把 Gluon 的 `GluonASTSource` 交给 `triton.compile`。它的身份是该前端构建的 TTGIR（`source_digest_kind` 为 `"tilefish.gluon-ir.v1"`，有自己的标签），所以改动布局、`num_warps`、函数体或常量就是另一个二进制文件，`replay` 会拒绝过时的那个，不论是否 pin（固定），与 Triton kernel 一样。以 `triton=` 给出的 Gluon kernel（或以 `gluon=` 给出的 `triton.jit` kernel）会被拒绝，并指出应使用的参数：对 kernel 对象是在声明时，对 `"module:function"` 字符串是在构建时。manifest 条目是一个 `triton` 条目（它由 Triton 编译，有两个 scratch 槽位），所以 host 像启动其他 kernel 一样启动它。

**每个架构一个实现。** 能给出模型所需比特的最快 kernel 可能因 GPU 而异：Triton 在不同架构上把同一个 `tl.dot` 下沉为不同的指令，这些指令可能以另一种顺序把宽点积的乘积相加。一个声明就可以表达这一点：

```python
@tilefish.extern(
    by_arch={
        "sm_103": tilefish.impl(triton=my_kernels.row_sum_tiled, num_warps=8, consts={"ROWS": 4}),
        "sm_120": tilefish.impl(triton=my_kernels.row_sum, num_warps=4),
    },
    fallback=None,                                       # or a tilefish.impl(...) for every other architecture
    grid=lambda R, ROWS: (tilefish.cdiv(R, ROWS),),      # an impl may have its own grid=
)
def row_sum(x: Tensor[R, 1024, f32], y: Out[Tensor[R, f32]], n_rows: R, ROWS: const = 1): ...
```

对 graph、plan、调用方和 `reference=` 模式而言，这个算子是同一个类型化签名。`export` 构建目标架构（`arch=`，或设备的架构）的实现，`jit` 构建设备架构的实现，二者都拒绝既没有自己的实现也没有 `fallback` 的架构。每个实现是一个 Triton 或 Gluon kernel，有自己的编译选项（`num_warps`、`num_stages`、`enable_fp_fusion`）、constexpr 值（`consts=`）和 `grid=`；每个实现都接受 stub 的参数，并具有相同的启动 ABI、bucket、维度和前置条件，这由声明检查，所以无论运行哪个实现，plan 都是同一份文档，host 也以同样方式启动每个实现。每个实现都是一个独立的算子（`row_sum.for_arch("sm_103")`、`tilefish.aot.compile_specs(row_sum, arch=103)`）：它的键在它的声明 digest 和编译规格中，它的 manifest 条目也记录了它（`"implementation": "sm_103"` 或 `"fallback"`；`docs/manifest-schema-3.md`，第 4.1 节）。`replay` 像导出时一样选择设备架构的实现，所以另一个实现的条目，或一个已改变（kernel、选项、常量或 grid 改变）的实现的条目，会作为过时而被拒绝；另一个架构的实现改变时，本架构的 manifest 仍然有效。Rust crate 读取这个字段（`Entry::implementation`），并启动其设备架构的条目。

一个实现可以被共享，可以是预构建的二进制文件，也可以有自己的 ABI：

```python
def row_sum_tma(x: TensorDesc(Tensor[R, 1024, f32], block=(64, 1024)), y: Out[Tensor[R, f32]], BLOCK_R: const = 64): ...

@tilefish.extern(
    by_arch={
        "sm_100": tilefish.impl(triton=my_kernels.row_sum_tma, stub=row_sum_tma, grid=lambda R: (tilefish.cdiv(R, 64),), name="tma"),
        "sm_103": tilefish.impl(triton=my_kernels.row_sum_tma, stub=row_sum_tma, grid=lambda R: (tilefish.cdiv(R, 64),), name="tma"),
        "sm_120": tilefish.impl(ptx="row_sum_sm120.ptx", symbol="row_sum", block=(256,)),   # the op's stub is its C ABI
    },
    grid=lambda R: (R,),
)
def row_sum(x: Tensor[R, 1024, f32], y: Out[Tensor[R, f32]], n_rows: R): ...
```

`name="tma"` 让映射到它的所有架构成为同一个实现：一个键（`"implementation": "tma"`）、一个声明 digest、一套编译规格（没有名字时，每个架构各有自己的，和以前一样）。带 `symbol=` 和 `block=` 的 `ptx=` / `cubin=` 是预构建的二进制文件；cubin 实现需要为运行它的每个架构各提供一个 cubin。`stub=` 列出 kernel 自己的参数，所以一个实现可以把某个张量作为 TMA 描述符接收，而另一个实现把它作为指针接收，并带有不同的标量、constexpr 或固定参数。所有实现必须共有的是算子的公开签名：它的张量（名字、顺序、访问方式、dtype、形状、是否 `Strided`）、它的每次调用固定参数和它的 bucket，这些正是调用方、plan 步骤和类型化的 Rust 启动器所绑定的。每个 manifest 条目带有其实现的 ABI，plan 会对照每个实现检查，所以它从不依赖某个特定架构。

**TMA 描述符。** 用 Triton 或 Gluon 写的 Hopper 或 Blackwell kernel 要靠 Tensor Memory Accelerator 达到最佳速度，它接收一个*张量描述符*：一个 128 字节的 `CUtensorMap`，由 host 编码（`cuTensorMapEncodeTiled`）并按值传递。声明为 `TensorDesc(...)` 的参数就是这样一个描述符：

```python
@tilefish.extern(gluon=my_gluon_kernels.tma_copy, num_warps=4, grid=lambda R, C: (tilefish.cdiv(R, 64), tilefish.cdiv(C, 64)))
def tma_copy(x: TensorDesc(Tensor[R, C, f16], block=(64, 64)), y: Out[TensorDesc(Tensor[R, C, f16], block=(64, 64))]): ...

y = tilefish.run(tma_copy, x)    # a call passes the tensor: the host encodes its descriptor (address, shape, strides) for this launch
```

声明给出 block（一次 `load` 或 `store` 搬运的 tile）、swizzle（默认是 block 的行所允许的最大值；Gluon kernel 的类型里带有它，Triton kernel 则由编译器选一个，构建会拒绝与编译器选择不符的声明）、L2 promotion 和越界填充（`"zero"`：张量之外的坐标，包括负的行，读出零）；map 的 box 就是 block，只是其行被截到 swizzle 跨度。kernel 启动时传入 map，然后是张量的形状（每个 `i32`）和步长（每个 `i64`），与 Triton 3.5.1 启动 host 端 `TensorDescriptor` 的方式完全相同，所以字节与 Triton 自己启动时一致（`test_gpu_tma.py` 对它们做比较）。manifest 记录每个描述符（kind `tensordesc`，它的 `tensor_map`：编码时除张量本身之外的所有参数；manifest schema 3，第 5.5 节），两个 host 在编码之前都会检查 driver 会拒绝的情况（16 字节对齐的地址：`pointer_alignment`；是 16 字节倍数的步长以及在 `i32` 范围内的维度：`tensor_map`）。map 在每次启动时根据那次启动的张量编码，对 plan 则在每次执行时根据每个步骤绑定的缓冲区编码；CUDA graph（CUDA 图）保存的是它被 capture（捕获）时那些地址的 map，就像它保存那些地址一样。`TensorDesc(Strided[Tensor[...]], ...)` 描述 kernel 读取的一个视图（它的行步长就是视图的行步长）。Triton 的*设备端*描述符（`tl.make_tensor_descriptor`）把它们的 map 写进一块全局 scratch，extern 要声明它（`global_scratch=`，见下文）。

**warp（线程束）特化的 kernel。** Blackwell kernel 的各个分区在各自的 warp 上运行时往往最快：Gluon 的 `gl.warp_specialize`（比如一个 1-warp 的 TMA 生产者和一个 1-warp 的 MMA 发射者，为 4 个计算 warp 供数），或 Triton 自己对 `tl.range(..., warp_specialize=True)` 循环做的 warp 特化。这样的 kernel 和其他 kernel 一样声明：

```python
@tilefish.extern(gluon=my_gluon_kernels.ws_matmul, num_warps=4, grid=lambda M, N: (tilefish.cdiv(M, 128), tilefish.cdiv(N, 128)))
def ws_matmul(a: TensorDesc(Tensor[M, K, f16], block=(128, 64)), b: TensorDesc(Tensor[N, K, f16], block=(128, 64)),
              c: Out[TensorDesc(Tensor[M, N, f16], block=(128, 128))], k_len: K): ...
```

`num_warps` 是 kernel **默认分区**的 warp 数（即 worker 分区之外的布局所针对的那些 warp；kernel 不做 warp 特化时就是它的全部 warp）。worker 会加上自己的 warp，编译出的 kernel 拥有的是总数：对 4 + 1 + 1，Triton 3.5.1 报告 8（对 8 + 1 + 1 报告 12，所以声明 8 并不会得到 8）。当 TTGIR 中有 warp 特化区域时，构建从编译出的 kernel 中取这个总数；对于 Triton 改变了 warp 数的其他任何 kernel，构建照旧拒绝；无论哪种情况，PTX 的 `.reqntid` 都必须等于 block。manifest 条目保留 `num_warps`（编译选项），并记录 `total_num_warps` 以及由它得出的 block（`32 * total_num_warps`；manifest schema 3，第 4.2 节），它在 manifest digest 之中，每次启动都使用这个 block：`jit`、`replay`、plan 的一个步骤、CUDA graph，以及 Rust crate（`Entry::total_num_warps`、`Plan::block()`）。

**启动形状。** extern 的启动方式可以表达比仿射 grid 更多的东西（manifest schema 3，第 4.3 到 4.6 节）：

```python
@tilefish.extern(triton=k.scale, grid=lambda T, C, BLOCK: (tilefish.cdiv(T * C, BLOCK),))       # a grid may multiply dimensions
def scale(x: Tensor[T, C, f32], y: Out[Tensor[T, C, f32]], n_t: T, n_c: C, BLOCK: const = 1024): ...

@tilefish.extern(triton=k.tiles, grid=lambda M, N: (tilefish.cdiv(M, 128) * tilefish.cdiv(N, 128),),
                 persistent=tilefish.persistent(programs_per_sm=1))                              # min(tiles, 1 * SMs) programs
def tiles(a: Tensor[M, K, f16], b: Tensor[N, K, f16], c: Out[Tensor[M, N, f16]], k_len: K): ...

@tilefish.extern(triton=k.rows, grid=lambda TB, BLOCK_T: (tilefish.cdiv(TB, BLOCK_T),),
                 by_bucket=lambda TB: tilefish.config(num_warps=2, BLOCK_T=16) if TB <= 256 else tilefish.config(num_warps=8, BLOCK_T=64))
def rows(x: Tensor[TB, 128, f32], y: Out[Tensor[TB, 128, f32]], n_rows: TB, BLOCK_T: const = 32): ...
```

grid 在每次启动时计算，从不求解，所以它可以让维度相乘（一个二进制文件适用于所有 `T` 和 `C`）。**持久化（persistent）** grid 是 tile 的个数，host 在加载该二进制文件的设备上（或在它声明自己运行的分区上：Rust 中是 `Manifest::with_sm_partition`，Python 中是 `tilefish.runtime.sm_partition(n)`）每个 SM（流式多处理器）最多启动 `programs_per_sm` 个 program，所以在 MIG 切片上导出的 manifest 也能在整块 GPU 上运行；kernel 以 `tl.num_programs(0)` 为步长遍历 tile，并保证其字节不取决于运行多少个 program，除非它声明自己针对的 SM 数（`sm_count=30`：一个持久化归约），此时 SM 数不同的 host 会拒绝它。`by_bucket=` 为分桶维度的每个 bucket 指定各自的编译期配置（warp 数、stage 数、CTA（线程块）数、const 的取值），每个二进制文件都按自己的配置构建、计算键和 digest。`num_ctas=2`（Triton 的编译选项）让一个 program 在由 2 个 CTA 组成的 cluster（线程块簇）上运行，预构建二进制文件则声明 `cluster=(2, 1, 1)`：两个 host 都用 `cuLaunchKernelEx` 加上该 cluster 启动它，grid 以 cluster 为单位计数。`global_scratch=256` 是编译出的 kernel 的每个 CTA 所寻址的全局 scratch（设备端 TMA 描述符把它们的描述符写在那里），构建会对照 Triton 的元数据检查它：Python 每次启动时分配它，Rust crate 把它作为一次启动的最后一个张量接收（`Plan::tensor_names()` 以 `global_scratch` 结尾），plan 则由其 workspace（工作区）提供。profile scratch、PDL 和 cooperative launch 仍然被拒绝。

**已 pin 的 cuBLASLt GEMM（通用矩阵乘）。** 作用于 `(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16]) -> Tensor[M, N, bf16]` 的 `@tilefish.library("gemm", acc=f32, impl="cublaslt.gemm")` 用 f32 累加器和一次舍入计算 `a @ b.T`（`impl="cublaslt.gemm_bias"` 增加 `bias: Tensor[N, bf16]`，在舍入之前以 f32 相加）。对每个 `(M, N, K, bias)`（`M` 取自其 bucket），在固定的 workspace 上限（8 519 680 字节，torch 的默认值；可用 `workspace_bytes=` 修改）下 cuBLASLt 启发式排在第一位的算法被 pin 住：它的 64 字节连同选出它的 cuBLASLt 版本、以及选出它时所在设备的 SM 数（`sm_count`：MIG 切片的 SM 比整块 GPU 少），写入 manifest 的 `library` 段；`replay` 运行这个算法，并拒绝其他 cuBLASLt 版本和 SM 数不同的设备（"re-export the manifest on this device"）。`jit` 以同样方式在内存中按设备 pin，所以 `jit` 和 `replay` 运行同一个算法。绑定是通过 ctypes 调用进程已加载的 `libcublasLt`（torch 的那个）；操作数是连续且 16 字节对齐的（尚不支持带步长的操作数）。

算法如何选择是声明（及其 digest）的一部分：

```python
# torch's choice: the result is torch.addmm(bias, a, b.t()) byte for byte (torch.mm(a, b.t()) without a bias)
@tilefish.library("proj", acc=f32, impl="cublaslt.gemm_bias", heuristic="torch", allow_bf16_reduction=True)
def proj(a: Tensor[M, 512, bf16], b: Tensor[1536, 512, bf16], bias: Tensor[1536, bf16]) -> Tensor[M, 1536, bf16]: ...

# algorithms chosen elsewhere (a tuner, another engine's pin file), one table per device: the heuristic is not asked there
config = {"id": 21, "tile_id": 11, "splitk_num": 1, "reduction_scheme": 0, "cta_swizzling": 0, "custom_option": 0,
          "stages_id": 20, "inner_shape_id": 0, "cluster_shape_id": 0}
tuned = tilefish.AlgoTable("sm_120", 84, 120804, {(32, 1536, 512, True): config})  # arch, SM count, cuBLASLt version

@tilefish.library("proj32", acc=f32, impl="cublaslt.gemm_bias", algos=[tuned])
def proj32(a: Tensor[32, 512, bf16], b: Tensor[1536, 512, bf16], bias: Tensor[1536, bf16]) -> Tensor[32, 1536, bf16]: ...
```

* `heuristic="torch"` 像 torch 2.9 的 `addmm`（`at::cuda::blas::gemm_and_bias`）那样询问启发式：只要一个结果，只受 workspace 上限约束（没有对齐下限：torch 的指针是 256 字节对齐的，即库的默认值），默认使用 torch 的上限，`gemm_bias` 为 1 048 576 字节（`CUBLASLT_WORKSPACE_SIZE`），`gemm` 为 8 519 680（`torch.mm` 用 `CUBLAS_WORKSPACE_CONFIG` 的 workspace 运行 `cublasGemmEx`，而该上限下启发式的答案正是它运行的算法；在恰为 sm_90 的设备上，torch 的默认值是 32 MiB，即 `workspace_bytes=33_554_432`）。使用 torch 的默认归约（`allow_bf16_reduction=True`）时，结果与 torch 逐字节一致：`test_gpu_cublaslt.py` 在 48 种形状上断言这一点，有 bias 和无 bias 都测，每种各用三组操作数；在 B300 MIG 切片（sm_103）和 RTX 5090（sm_120）上用 cuBLASLt 12.8.4、在 B300 切片上用 13.1.0 实测过（`test_gpu_gemm.py`，30 种形状）。默认姿态（posture）`"tilefish"` 增加 16 字节的对齐下限，请求 8 个结果，并且在没有显式开启时，不给带 bias 的 GEMM 使用 split-K（K 维切分；见下文），而 torch 不这样做。它不是 torch 的姿态，它的结果有多少与 torch 一致取决于 cuBLASLt 版本：在 `test_gpu_gemm.py` 的 15 种带 bias 的形状上，12.8.4 全部一致，13.1.0 有 13 种一致，13.1.0 的启发式在 32 和 64 x 1024 x 3072 上把使用 f32 部分和、切分数为 3 的 split-K 排在第一（在这两处，workspace 上限、对齐下限、请求的结果数都不会改变它的第一个答案；归约方案掩码会）。
* `allow_bf16_reduction=True` 允许 split-K 以 bf16 累加部分和（`INPLACE`、`OUTPUT_TYPE`：torch 的默认值，`allow_bf16_reduced_precision_reduction`），这会舍入不止一次。没有它时，只向启发式请求 f32 部分和，带 bias 的 GEMM 则完全不用 split-K（实测在 bias epilogue（尾部）下，使用 f32 部分和的 split-K 会舍入两次：RTX 5090 上的 cuBLASLt 12.8.4）；违反其中任一条的算法会被两个 host 拒绝。尽管有些 split-K 算法保持了一次舍入，这条规则仍保留在默认姿态中：启发式的答案里没有任何东西能把它们区分开（`docs/numerics.md`，第 4 节）；`heuristic="torch"` 给出与 `addmm` 相同的字节。
* `algos=` 导入算法（`tilefish.AlgoTable(arch, sm_count, cublaslt_version, {(m, n, k, bias): config})`，config 是九个 `cublasLtMatmulAlgoConfig` 属性，或一个 pin 的 128 个十六进制数字）：在表所针对的设备上，每个条目 pin 住表中的算法，按 cuBLASLt 的方式编码，并接受同样的检查（针对该问题的 `cublasLtMatmulAlgoCheck`、workspace 上限、只有显式开启时才允许 bf16 部分和）；没有任何表针对的设备会被拒绝，绝不会交给启发式。

manifest 条目把它们记录为 `heuristic`、`algo_source` 和 `allow_bf16_reduction`，只有不是默认值时才会出现（`docs/manifest-schema-3.md`，第 9 节）。pin 是启发式所选的配置，由库编码：启发式返回的 64 字节中可能带有不属于配置的字节，这些字节在不同进程间会变化。

**f16、f32 结果、`[K, N]` 形式的 `b`。** 操作数可以是 f16（两者同一类型），结果可以是 f32（bias 的类型与结果相同：实测 cuBLASLt 不接受别的类型），`layout="NN"` 接受 `b: Tensor[K, N, T]` 并计算 `a @ b`（默认的 `"TN"` 是 `b: Tensor[N, K, T]` 时的 `a @ b.T`）：

```python
@tilefish.library("up", acc=f32, impl="cublaslt.gemm_bias", layout="NN", heuristic="torch", allow_f16_reduction=True, workspace_bytes=64 << 20)
def up(a: Tensor[M, 7168, f16], b: Tensor[7168, 1536, f16], bias: Tensor[1536, f16]) -> Tensor[M, 1536, f16]: ...   # torch.addmm(bias, a, b), byte for byte

@tilefish.library("down", acc=f32, impl="cublaslt.gemm")
def down(a: Tensor[M, 3072, f16], b: Tensor[1024, 3072, f16]) -> Tensor[M, 1024, f32]: ...   # a @ b.T, rounded once to f32
```

允许 split-K 以输出类型累加部分和的开关，按操作数类型命名，与 torch 的开关一样（f16 操作数用 `allow_f16_reduction`）。在 `heuristic="torch"` 下打开它，结果在两种布局下都与 f16 的 `torch.mm` / `torch.addmm` 逐字节一致（`test_gpu_gemm_f16.py`：63 种形状，每种三组操作数，在 B300 MIG 切片上用 cuBLASLt 12.8.4 实测）。布局 `NN` 的 `AlgoTable` 行以 `(m, n, k, bias, "NN")` 为键；配置按声明的类型编码。条目在 `dtypes`、`transa` 和 `lda` 中说明这种形式（`docs/manifest-schema-3.md`，第 9.2 节）。

**以 `[K, M]` 存储的 `a`。** `layout="TT"` 接受 `a: Tensor[K, M, T]` 和 `b: Tensor[N, K, T]`，计算 `a.T @ b.T`；`layout="NT"` 接受同样的 `a` 和 `b: Tensor[K, N, T]`，计算 `a.T @ b`，上面每种操作数类型都适用：

```python
@tilefish.library("scores", acc=f32, impl="cublaslt.gemm", layout="TT", heuristic="torch", allow_f32_reduction=True)
def scores(a: Tensor[8, M, f32], b: Tensor[1024, 8, f32]) -> Tensor[M, 1024, f32]: ...   # torch.mm(a.t(), b.t()), TF32 off, byte for byte
```

在 `heuristic="torch"` 下，对连续的 `a [K, M]`，pin 就是 torch 为 `torch.mm(a.t(), b.t())` / `torch.mm(a.t(), b)` 和 `torch.addmm` 运行的算法，在 bf16、f16 和 f32 下都逐字节一致（`test_gpu_gemm_layouts.py`，`M` 从 2 到 2048；在 `M = 1` 时，torch 把 `a [K, 1]` 当作它同时也是的连续 `[1, K]` 来乘，所以单行 GEMM 要声明为 `TN` 或 `NN`）。条目在 `transb` 和 `ldb` 中说明这一点（`docs/manifest-schema-3.md`，第 9.2 节）。

**f32，关闭 TF32。** f32 操作数的结果是 f32（bias 也是 f32），并按 f32 相乘：

```python
@tilefish.library("proj32", acc=f32, impl="cublaslt.gemm_bias", heuristic="torch", allow_f32_reduction=True)
def proj32(a: Tensor[M, 1024, f32], b: Tensor[3072, 1024, f32], bias: Tensor[3072, f32]) -> Tensor[M, 3072, f32]: ...   # torch.addmm(bias, a, b.t()), TF32 off
```

计算类型是 `CUBLAS_COMPUTE_32F`，绝不是 `CUBLAS_COMPUTE_32F_FAST_TF32`（torch 的 `allow_tf32` 所选的类型），也不是 cuBLAS 13 的 bf16x9 模拟（`CUBLAS_COMPUTE_32F_EMULATED_16BFX9`）。两个 host 都拒绝不会把 f32 按 f32 相乘的 f32 算法：为另一种计算类型编码的算法（cuBLASLt 会把计算类型写进那 64 字节，而 TF32 和模拟算法的 id 根本无法在 `CUBLAS_COMPUTE_32F` 下编码），numerical-implementation 标志表明 TF32 或 16 位输入的算法，以及在 cuBLASLt 把 `CUBLAS_COMPUTE_32F` 按 TF32（`NVIDIA_TF32_OVERRIDE=1`）或按模拟（`CUBLAS_EMULATE_SINGLE_PRECISION=1`）运行的进程中的每个 f32 pin：实测，编码能体现这两者。允许 split-K 以输出类型累加部分和的开关是 `allow_f32_reduction`；torch 没有这样的开关，也从不限制 f32 GEMM 的归约，所以要得到 torch 的选择就需要打开它。在 `heuristic="torch"` 下打开它并使用 torch 的 workspace（`mm` 为 8 519 680 字节，即 `CUBLAS_WORKSPACE_CONFIG` 的值；`addmm` 为 1 MiB），结果在两种布局下都与关闭 TF32 的 f32 `torch.mm` / `torch.addmm` 逐字节一致（`test_gpu_gemm_f32.py`：756 次运行中 756 次一致，63 种形状、`M` 从 1 到 2048，三组操作数，`mm` 和 `addmm`，`TN` 和 `NN`，分别用默认 workspace 以及 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 加 32 MiB 上限；在 B300 MIG 切片上用 cuBLASLt 12.8.4 和 13.1.0、在 RTX 5090 上用 13.1.0 实测）。torch 的 `mm` 运行 `cublasSgemm`，cuBLAS 为它选择的算法就是 cuBLASLt 启发式在该姿态下给出的答案。见 `docs/manifest-schema-3.md`，第 9.2 节。

**SM 分区。** 在 green context（设备 SM 的一个分区）上，每次 SM 数查询报告的仍是整个设备的数量，已 pin 的算法无论有无分区都计算出相同的字节（实测，cuBLASLt 12.8.4）；改变的是启发式的选择，它经由进程第一个 cuBLASLt 句柄的 context 起作用。`sm_count_target=P` pin 住启发式为 `P` 个 SM 所选的算法（`CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET`，每次检查和运行也都会设置它），在任何 context 中都一样；条目记录它，运行在 `P` 个 SM 的分区上的 Rust host 通过 `Manifest::with_sm_partition(Some(P))` 只接受为该分区选出的 pin（`docs/manifest-schema-3.md`，第 9.3 节）。

**带 cuBLASLt epilogue 的 GEMM。** `impl="cublaslt.matmul"`（RFC 0003，里程碑 E2）在那一次舍入之前，对 f32 累加器执行 bias、ReLU 和残差加（`beta * C`），所用算法是为该 epilogue pin 住的：

```python
# out = relu((acc + c) + bias), rounded once to bf16: the library adds c before the bias
@tilefish.library("up", acc=f32, impl="cublaslt.matmul", epilogue="relu_bias", beta=1)
def up(a: Tensor[M, H, bf16], b: Tensor[F, H, bf16], bias: Tensor[F, bf16], c: Tensor[M, F, bf16]) -> Tensor[M, F, bf16]: ...
```

`epilogue` 取 `"none"`、`"bias"`、`"relu"` 或 `"relu_bias"`（cuBLASLt 的 `DEFAULT`、`BIAS`、`RELU`、`RELU_BIAS`），`beta=1` 加上 `c [M, N]`；操作数是 bf16 或 f16，结果与操作数同类型，四种布局都可用，`heuristic`、`workspace_bytes`、各个开关和 `sm_count_target` 与上文相同（尚不接受 `algos=`）。在 pin 之下，字节由库决定，所以**每个 pin 在生成时都会被验证**（`jit` 和 `export`）：在两组带种子的操作数上运行，bias 和 `c` 中含特殊值，`a` 中的某些行会让累加器出现 NaN、零、无穷和溢出，然后与在 pin 自己的累加器（其配置以 f32 结果重新编码）上用 f32 计算公式、再舍入一次的结果逐位比较。条目记录 `"measured-exact"` 或 `"binary-defined"`，以及不同元素的个数和它们的最大距离；`contract="measured-exact"` 会拒绝不是 measured-exact 的 pin。在 B300 MIG 切片上实测（14 种形状，bf16 和 f16，两种姿态，七种情形：每个版本 392 个 pin，`test_gpu_matmul.py`）：用 cuBLASLt 12.8.4 时每个 pin 都是 measured-exact；用 13.1.0 时有 372 个是，其余 20 个都是 32 x 1024 x 3072 上的算法 21，它把 bias 加到 `c` 上的方式不符合任何一种顺序（26% 的元素不同），而且它的 ReLU 对 NaN 给出 `0`，而其他算法会保留 NaN。与 bias 一样，没有显式开启时，epilogue 不使用 split-K。`export` 记录一个 canary（哨兵值，`canary=True`，默认）：在设备上由种子生成的操作数上运行该 pin，记录其结果的 digest；回放（Python）和加载（Rust 和 C++ host）在使用该 pin 之前都会运行它，digest 不同就拒绝：这样同一 cuBLASLt 版本的两个不同构建也能区分开。见 `docs/manifest-schema-3.md`，第 9.5 节。

**已 pin 的 cuDNN 卷积。** `impl="cudnn.conv_fwd"` 按签名声明一个前向卷积：`(x, w) -> y`，维度顺序由 `layout` 决定（默认 `"NCHW"`：`x [N, C, *S]`、`w [K, C/G, *R]`；`"NHWC"`：`x [N, *S, C]`、`w [K, *R, C/G]`），有一到三个空间轴，f16、bf16 或 f32 数据，f32 累加器和一次舍入，并为每个 bucket pin 住 cuDNN 运行的东西：

```python
T = Dim("T", buckets=[64, 646])

# the backend API (the default): an engine configuration, an engine and its knobs, chosen as torch chooses it
@tilefish.library("conv", impl="cudnn.conv_fwd", layout="NHWC", padding=3)
def conv(x: Tensor[1, T, 64, f16], w: Tensor[64, 7, 64, f16]) -> Tensor[1, T, 64, f16]: ...

# the legacy API: the algorithm enum of cudnnConvolutionForward, cudnnGetConvolutionForwardAlgorithm_v7's first deterministic answer that fits
@tilefish.library("down", impl="cudnn.conv_fwd", layout="NHWC", stride=2, padding=1, api="legacy")
def down(x: Tensor[1, T, 64, f16], w: Tensor[128, 4, 64, f16]) -> Tensor[1, tilefish.cdiv(T, 2), 128, f16]: ...
```

声明的结果在每个 bucket 上都必须是卷积的输出（`(S + 2 * padding - dilation * (R - 1) - 1) // stride + 1`；这里就是 `cdiv(T, 2)`）。math 是声明的一部分：`"tensor_op"`（16 位数据的默认值）或 `"fma"`；对 f32 数据则是 `"fma"`（关闭 TF32）或 `"tf32"`，需显式写明。`api="backend"` pin 住一个 engine 配置（engine 的全局索引和每个 knob 的选择：它指定了一个 kernel），在 `heuristic="instant"` 下选出，即 torch 2.9 在 `deterministic = True` 和 `benchmark = False` 时的姿态（`CUDNN_HEUR_MODE_INSTANT`，然后是 fallback 列表，按确定性、无降精度转换以及对 `"fma"` 不用 tensor core 过滤），或在 `"mode_b"` 下选出；`api="legacy"` pin 住算法枚举值。`algos=tilefish.CudnnTable(arch, sm_count, cudnn_version, {(n, c, k, input, filter, stride, padding, dilation, groups): pin})` 则导入 pin。条目记录问题、math、pin、它是如何选出的、cuDNN 版本（`cudnnGetVersion()`）、SM 数和一个 `sm_count_target`；`replay` 和 Rust host 拒绝其他 cuDNN 版本、其他 SM 数，以及设置了 `NVIDIA_TF32_OVERRIDE` 的进程（它会在声明的 math 背后改变 cuDNN 的选择），并审核 pin（是 `cudnnGetConvolutionForwardAlgorithm_v7` 的一个确定性答案；是 note 都被允许、knob 都在范围内且其执行计划可以 finalize 的 engine）。由于同一 cuDNN 版本的两个构建对 `cudnnGetVersion()` 给出相同的答案，`export` 还为每个条目记录一个 canary：pin 在由种子生成的操作数上的输出的 digest（在两个 host 中，两者都由同一个 PTX 模块在设备上生成：一个含 743 个条目的 plan set（plan 集）加载耗时 1 到 2 s），`replay` 和 Rust host 会再运行一次（一次，在它们自己的 stream 上，绝不是 NULL stream），digest 不同就拒绝；没有 canary 的条目照旧加载。导入的 pin 中值为 -1 的 knob 是 cuDNN 的“未设置”，会被略去。每次调用都会把 cuDNN 的 stream 设为传给它的那个 stream。没有 bias（torch 的 cuDNN 卷积用单独的 `add_` 加 bias，tilefish 的 kernel 也是这样做的）。

在 torch 运行 cuDNN 的情况下，`"instant"` pin 给出 `torch.nn.functional.conv1d` / `conv2d` / `conv3d` 的字节（`test_gpu_conv.py`：15 个问题，三组操作数，1-D、2-D 和 3-D，f16、bf16 和关闭 TF32 的 f32，各种 stride、dilation 和 group；B300 MIG 切片上用 cuDNN 9.10.2 和 9.13.0，RTX 5090 上用 9.13.0），而 `IMPLICIT_PRECOMP_GEMM` 的 legacy pin 给出 torch legacy 路径的字节。两种 API 都能在 launch plan（启动计划）中运行（workspace 来自 arena（内存池）），也都能被 capture 进 CUDA graph；绑定是通过 ctypes 调用进程已有的 `libcudnn`（torch 的那个），Rust crate 只在 manifest 含有卷积时才加载 cuDNN（`Manifest::conv`、`Conv`）。见 `docs/manifest-schema-3.md`，第 9.4 节。

**CuTe DSL kernel。** CUTLASS 的 CuTe DSL（`nvidia-cutlass-dsl`：用 Python 写 TMA、`tcgen05`、cluster、持久化 tile 调度器）的 kernel 也是 extern，由 DSL 编译，每个 host 都把它当作预构建 cubin，按编译出的 host 函数所做的那种启动方式来启动（`docs/cute.md`）：

```python
@tilefish.extern(cute=PersistentDenseGemmKernel(cutlass.Float32, True, (256, 128), (2, 1), True),   # a CUTLASS example's kernel class
                 cute_args=lambda a, b, c: (a.unsqueeze(-1), b.unsqueeze(-1), c.unsqueeze(-1), max_active_clusters))
def gemm(a: Tensor[M, 1024, f16], b: Tensor[2048, 1024, f16], c: Out[Tensor[M, 2048, f16]]): ...   # static shapes: one binary per bucket of M
```

构建为设备编译它（DSL 自己的 cubin 和 PTX），并在一个被 capture 进 CUDA graph 的 stream 上调用编译出的 host 函数，因此 kernel 并不真正运行：由此得到的是 grid、block、cluster 和动态共享内存，以及每个 kernel 参数，参数必须是张量的指针、DSL 编码的 TMA 描述符（host 把张量地址写进去的模板）或常量字节，并通过在其他地址上的三次 capture 来检查。条目是 source 为 `cute` 的 `binary` 条目（schema 第 4.9 节），像其他条目一样被 pin；它的身份是 PTX，Python 回放时会重新编译它。在一块 sm_103 GPU 的 30-SM 分区上，示例中的持久化 tcgen05 GEMM 写出的字节与已 pin 的 cuBLASLt GEMM 相同，在 `4096 x 2048 x 1024` 时速度是它的 1.17 倍（`M = 1024` 时为 0.92 倍）。请自行安装经过测试的 DSL 版本并固定它（`nvidia-cutlass-dsl==4.8.1`：它处于 beta 阶段，换一个版本就是另一个声明）。它是 NVIDIA 的软件，有自己的许可证，tilefish 既不随附它也不依赖它；导出写出的 cubin 和 PTX 是它的输出：分发它们是你的责任（`docs/cute.md`，第 6 节）。
