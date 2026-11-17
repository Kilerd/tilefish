# Views, device scalars, prebuilt binaries and a pinned GEMM

A real model needs four things that a dense, compiled-in, Triton-only operator cannot say. Each is part of the manifest, which is schema 3
(`docs/manifest-schema-3.md` specifies every field; the Python package reads and writes only schema 3, and says "re-export" for schema 2).

**Strided views.** A tensor type has element strides and an element offset, in the dimension algebra (`TensorType.strides`, `.offset`); dense
and 16-byte aligned (`contiguous`) is the default. `x.reshape(M, 3, H, D)` (a dense input), `x.permute(1, 0, 2)`, `x[:, 1]` / `x.select(axis, i)` and
`x.split(...)` copy nothing: they are views with the strides and the offset of the operation. A split whose number of chunks depends on a
dimension (`x.split(16, axis=1)` of `x: [M, C]` with `C = Dim("C", divisible_by=16)`) returns `Chunks`, a sequence of symbolic length:
`chunks[0]` and `chunks[-1]` always exist, any other chunk where the dimension's smallest size proves it, and `len()` and unpacking
raise. A stride or an offset may be a product of
symbolic sizes: an `[H, L, L]` tensor has the stride `L*L` along its first axis, and `x[h]` the offset `h*L*L` (`tilefish.dims.mul`; a shape stays
affine, `L * L` is still refused there, and only a launch grid may multiply dimensions), which a `Strided[...]` argument binds and a plan evaluates at every bucket, so attention
scores `[H, L, L]` are a buffer of a plan and their transposes and heads are views of it. A view is a `View[...]` and can go into traced
elementwise and reduction ops; a declared operator takes one only where its parameter says so. Every other tensor parameter (and result) of a kernel,
extern, library op or graph is contiguous: an annotation that is a `TensorType` with other strides, an offset or no alignment is refused at the
declaration, and a launch refuses a tensor that is not contiguous for such a parameter. An `extern` parameter `x: Strided[Tensor[S, H, D, bf16]]`
takes a view, and the kernel's strides are scalar parameters the stub declares in the kernel's argument order:

```python
@tilefish.extern(triton=head_copy, grid=lambda S: (S, 8))
def head_copy(x: Strided[Tensor[S, 8, 64, bf16]], y: Out[Tensor[S, 8, 64, bf16]],
              x_s0: StrideOf("x", 0), x_s1: StrideOf("x", 1), D: const = 64): ...      # the last axis has stride 1

y = tilefish.run(head_copy, qkv[:, 1])        # a view of a fused qkv tensor: its pointer and its two strides go to the kernel
```

A kernel that reads several views of one buffer with **one** stride pair (the q, k and v column blocks of a fused `[B, 3 * D]` tensor) declares the stride
once for all of them, `s_row: StrideOf(("q", "k", "v"), 0)`: the views share the stride dimension, so a call binds it from `q` and refuses a `k` or a `v` that
does not have it (a trace call, a launch in Python and in Rust, and every step of a plan), and the kernel is given the stride once and a pointer per view.

An axis without a `StrideOf` must be the last one (its stride is then 1, and a view that does not have it is refused); the strides are bound at a
call like shapes are, range-checked as `i32` (`i64` for `index_bits=64`), and a strided pointer is bounded by its extent (`max_extent`), not its element
count. The type of a stride scalar is the width of the offsets unless the `StrideOf` says otherwise: a kernel that takes `i32` strides and computes
64-bit offsets with them (`row.to(tl.int64) * s_row`) declares `index_bits=64` and `s_row: StrideOf("x", 0, dtype=tilefish.i32)`, and a launch then holds
the stride to an `i32` and the extent of the view to nothing. The pointer passed is `base + offset * itemsize`. `Strided[Tensor[...], 16]` compiles the kernel for a 16-byte aligned pointer
(`tt.divisibility=16`), which the offset must then provably allow (a trace call refuses an offset that is not a multiple of 16 bytes, a launch
checks the address); without the `16` Triton gets no hint and any element-aligned view runs. Lowered `@kernel`s still take contiguous tensors only: a
view in a kernel body is refused with the rule that says so.

**Views that do not start on 16 bytes, given to a lowered kernel.** Every pointer of a lowered kernel is compiled as 16-byte aligned, so a dense view that
does not provably start on 16 bytes, such as a row `plane[k]` of a `[K, R]` int32 plane with an odd `R`, is not a contiguous tensor. Such a view given to a
read-only parameter runs the kernel's **specialisation** for it: `kernel.unaligned("a")`, the same kernel with `a` compiled aligned to its element only (no
`tt.divisibility` for it, a `pointer_alignment` of its element size that every launch checks). It is an operator of its own, keyed apart from the kernel
everywhere a binary is (its name `add_rows__ua0`, after the indices of those parameters, its stems, its declaration digest, its Triton IR and so its build cache
key, its manifest entries), and its generated source is the kernel's:

```python
@tilefish.graph
def plane_sum(plane: Tensor[3, R, i32]):
    s = add_rows(plane[1], plane[0])        # plane[1] starts 4 * R bytes in: the step launches add_rows__ua0
    return add_rows(s, plane[2])            # plane[2] starts 8 * R bytes in: add_rows__ua1

y = tilefish.run(add_rows, plane[1], plane[0])   # an address that is not 16-byte aligned runs add_rows__ua0 too
```

A graph's export has the specialisations its plan's steps launch; `tilefish.runtime.export(add_rows.unaligned("a"), directory)` writes one for the kernel
alone, and a replay of a manifest that lacks it says so (`docs/manifest-schema-3.md`, section 6).

**Device scalars.** `n_tok: DevScalar[i32]` is one element in device memory: in the ABI a pointer with the shape `[]`, at the API a tensor of one
element. A binary reads it with `tl.load`, so one binary serves every value and a value changes by writing memory, never by building; a lowered kernel
reads it with `tilefish.load(s)` (one `tl.load(ptr)`, broadcast).

**A dimension only an `Out` tensor has.** An extern may write a tensor whose size no input has: `key_mask(mask: Out[Tensor[B, bf16]], n_valid: DevScalar[i32],
b: B)` makes a `[B]` mask from a device scalar alone. A launch binds `B` from the `Out` tensor it is given (`tilefish.run(key_mask, n, outputs=mask)`; a Rust
launch and every plan step have their `Out` tensors) and checks it like any dimension, buckets included. In a graph the call binds `B` to the graph's own `B`
(the same `Dim`), so the plan, which is per bucket, sizes the mask and the step binds it from that buffer.

**Prebuilt binaries.** `@tilefish.extern(ptx="k.ptx", symbol="k", block=(256,), grid=...)` or `cubin={"sm_103": "k.cubin"}` declares a kernel that
Triton does not compile, by its C ABI: pointers, `Dim` scalars and typed `const[i32|i64|f32]` fixed arguments, in order, and **no Triton scratch slots**.
Its identity is the sha256 of the bytes of the file (`source_digest` of kind `tilefish.binary.v1`); PTX is assembled by `ptxas` with fixed, recorded flags
(the ones Triton passes, so the PTX of a Triton build declared this way gives the machine code, and the output bytes, of the Triton path); a cubin is used
as it is for its own architecture and refused for another. `replay` refuses a file that is not the one the manifest was built from.

A `const[i32]` or `const[i64]` of a prebuilt binary **without a default** is a fixed argument *given per call*: the geometry of a call (an output
size, an offset) that no tensor determines. One declaration and one binary serve every value: `k(x, h_out=5)`, `tilefish.run(k, x, h_out=5)`,
Rust `Kernel::launch_with(stream, tensors, &[("h_out", 5)])`; in a graph the value may be an expression of the graph's dimensions (`h_out=H // 2`),
and each step of the plan records the values of its call at its bucket (`"given": {"h_out": 5}`), so they are in the plan's digest and both hosts
check them against the entry (`docs/manifest-schema-3.md`, section 5.3).

**Null pointers and 64-bit dimension scalars.** A pointer an extern's kernel does not use in the declared use (a bias it reads only when it is not
null, or only under a constexpr flag the declaration turns off) is declared `Null[dtype]`: a call does not pass it, and both hosts pass the null pointer
at its slot (manifest kind `null`, section 5.6). A dimension scalar is an `i32` unless it is declared `i64[n]`: the same scalar passed as an `i64`, its value
still derived from the shapes and checked against them, for a kernel whose count is 64 bits wide (section 5.7). Both together:

```python
@tilefish.extern(ptx="scale_bias.ptx", symbol="scale_bias_f32", block=(256,), grid=lambda n: (tilefish.cdiv(n, 256),), index_bits=64)
def scale(x: Tensor[n, f32], bias: Null[f32], out: Out[Tensor[n, f32]], n_elem: i64[n]): ...    # (const float*, const float*, float*, long long)
```

**Gluon kernels.** A kernel written in Triton's Gluon dialect (`triton.experimental.gluon`: `@gluon.jit`, layouts written out, so the kernel pins
which thread holds which element and so the order of its reductions) is declared with `gluon=` in place of `triton=`, and everything else is the same:
strided views and shared strides, `const` arguments, buckets, `index_bits`, `enable_fp_fusion`, plans, `jit`, `replay` and the Rust host.

```python
@tilefish.extern(gluon=my_gluon_kernels.row_sum, num_warps=4, grid=lambda R: (tilefish.cdiv(R, 4),), index_bits=64)
def row_sum(x: Strided[Tensor[R, 256, f32]], y: Out[Tensor[R, f32]], s_row: StrideOf("x", 0, dtype=tilefish.i32), n_rows: R, C: const = 256): ...
```

The declaration says which frontend compiles the kernel because a compile spec is the whole recipe of a binary, made from the declaration without
importing the kernel: `"language": "gluon"` is in the spec and in the declaration digest, and the build hands Gluon's `GluonASTSource` to
`triton.compile`. Its identity is the TTGIR that frontend builds (`source_digest_kind` `"tilefish.gluon-ir.v1"`, under its own tag), so a changed
layout, `num_warps`, body or constant is another binary, and `replay` refuses a stale one, pinned or not, as for a Triton kernel. A Gluon kernel given as
`triton=` (or a `triton.jit` one as `gluon=`) is refused with the argument to use: at the declaration for a kernel object, at the build for a
`"module:function"` string. The manifest entry is a `triton` entry (Triton compiled it; it has the two scratch slots), so a host launches it like any other.

**An implementation per architecture.** The fastest kernel that gives the bits a model needs can differ by GPU: Triton lowers the same `tl.dot` to other
instructions on other architectures, and those may add the products of a wide dot in another order. One declaration can say so:

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

The op is one typed signature for a graph, a plan, a caller and `reference=` mode. `export` builds the implementation of the target architecture (`arch=`,
or the device's), `jit` the one of the device's, and both refuse an architecture that has neither its own nor a `fallback`. Each implementation is a Triton or
Gluon kernel with its own compile options (`num_warps`, `num_stages`, `enable_fp_fusion`), constexpr values (`consts=`) and `grid=`; every one takes the stub's
parameters and has the same launch ABI, buckets, dimensions and preconditions, which the declaration checks, so a plan is the same document whichever runs
and a host launches each the same way. Each is an op of its own (`row_sum.for_arch("sm_103")`, `tilefish.aot.compile_specs(row_sum, arch=103)`): its key is in
its declaration digest and its compile specs, and its manifest entries record it (`"implementation": "sm_103"`, or `"fallback"`;
`docs/manifest-schema-3.md`, section 4.1). `replay` selects the implementation of the device's architecture as the export did, so an entry of another
implementation, or of one that changed (its kernel, options, constants or grid), is refused as stale; another architecture's implementation changing
leaves this architecture's manifest valid. The Rust crate reads the field (`Entry::implementation`) and launches the entry of its device's architecture.

An implementation can be shared, be a prebuilt binary, and have an ABI of its own:

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

`name="tma"` makes one implementation of every architecture mapped to it: one key (`"implementation": "tma"`), one declaration digest, one set of compile
specs (without a name each architecture has its own, as before). `ptx=` / `cubin=` with `symbol=` and `block=` is a prebuilt binary; a cubin implementation
needs a cubin for every architecture that runs it. `stub=` lists the kernel's own parameters, so one implementation takes a tensor as a TMA descriptor
where another takes a pointer, with other scalars, constexprs or fixed arguments. What every implementation must share is the op's public signature: its
tensors (names, order, access, dtype, shape, `Strided` or not), its fixed arguments per call and its buckets, which is what a caller, a plan step and a typed
Rust launcher bind. Each manifest entry carries its implementation's ABI, and a plan is checked against every implementation, so it never relies on an
architecture.

**TMA descriptors.** A Hopper or Blackwell kernel in Triton or Gluon gets its best speed from the Tensor Memory Accelerator, which takes a *tensor
descriptor*: a 128-byte `CUtensorMap` that the host encodes (`cuTensorMapEncodeTiled`) and passes by value. A parameter declared `TensorDesc(...)` is one:

```python
@tilefish.extern(gluon=my_gluon_kernels.tma_copy, num_warps=4, grid=lambda R, C: (tilefish.cdiv(R, 64), tilefish.cdiv(C, 64)))
def tma_copy(x: TensorDesc(Tensor[R, C, f16], block=(64, 64)), y: Out[TensorDesc(Tensor[R, C, f16], block=(64, 64))]): ...

y = tilefish.run(tma_copy, x)    # a call passes the tensor: the host encodes its descriptor (address, shape, strides) for this launch
```

The declaration says the block (the tile one `load` or `store` moves), the swizzle (by default the largest the block's rows allow; a Gluon kernel's type has it,
a Triton kernel's compiler picks one and the build refuses a declaration that is not its choice), the L2 promotion and the out-of-bounds fill (`"zero"`: a
coordinate outside the tensor, a negative row included, reads zeros); the box of the map is the block with its rows cut to the swizzle span. The kernel is
launched with the map, then the tensor's shape (`i32` each) and strides (`i64` each), exactly as Triton 3.5.1 launches a host `TensorDescriptor`, so the
bytes are those of Triton's own launch (`test_gpu_tma.py` compares them). The manifest records each descriptor (kind `tensordesc`, its `tensor_map`: every
argument of the encode but the tensor's; manifest schema 3, section 5.5), and both hosts check what the driver would refuse before they encode (a 16-byte
address: `pointer_alignment`; strides that are multiples of 16 bytes and dimensions within `i32`: `tensor_map`). A map is encoded at every launch from the
tensor of that launch, and for a plan at every execution from the buffers each step binds; a CUDA graph holds the maps of the addresses it was captured
with, as it holds the addresses. `TensorDesc(Strided[Tensor[...]], ...)` describes a view the kernel reads (its row stride is the view's). Triton's
*device-side* descriptors (`tl.make_tensor_descriptor`) write their maps into a global scratch, which an extern declares (`global_scratch=`, below).

**Warp-specialised kernels.** A Blackwell kernel is often fastest when its partitions run on warps of their own: Gluon's `gl.warp_specialize` (a
1-warp TMA producer and a 1-warp MMA issuer feeding 4 compute warps, say), or Triton's own warp specialisation of a `tl.range(..., warp_specialize=True)`
loop. Such a kernel is declared like any other:

```python
@tilefish.extern(gluon=my_gluon_kernels.ws_matmul, num_warps=4, grid=lambda M, N: (tilefish.cdiv(M, 128), tilefish.cdiv(N, 128)))
def ws_matmul(a: TensorDesc(Tensor[M, K, f16], block=(128, 64)), b: TensorDesc(Tensor[N, K, f16], block=(128, 64)),
              c: Out[TensorDesc(Tensor[M, N, f16], block=(128, 128))], k_len: K): ...
```

`num_warps` is the warps of the kernel's **default partition** (those its layouts outside the worker partitions are written for; all its warps when it is
not warp-specialised). The workers add warps of their own, and the compiled kernel has the total: Triton 3.5.1 reports 8 for 4 + 1 + 1 (and 12 for
8 + 1 + 1, so declaring 8 would not give 8). The build takes that total from the compiled kernel when its TTGIR has warp-specialised regions, and refuses
any other kernel whose warps Triton changed, as before; either way the PTX's `.reqntid` must be the block. The manifest entry keeps `num_warps` (the compile
option) and records `total_num_warps` with the block it makes (`32 * total_num_warps`; manifest schema 3, section 4.2), which is in the manifest digest, and
every launch takes that block: `jit`, `replay`, a step of a plan, a CUDA graph, and the Rust crate (`Entry::total_num_warps`, `Plan::block()`).

**Launch shapes.** How an extern is launched can say more than an affine grid (manifest schema 3, sections 4.3 to 4.6):

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

A grid is computed at each launch and never solved, so it may multiply dimensions (one binary for every `T` and `C`). A **persistent** grid is the
number of tiles, and the host launches at most `programs_per_sm` programs per SM of the device it loads the binary on (or of the partition it says it runs
on: `Manifest::with_sm_partition` in Rust, `tilefish.runtime.sm_partition(n)` in Python), so a manifest exported on a MIG slice runs on the whole GPU;
the kernel walks the tiles with a stride of `tl.num_programs(0)`, and promises that its bytes do not depend on how many programs run, unless it states the
SM count it is for (`sm_count=30`: a persistent reduction), which a host on another count refuses. `by_bucket=` gives each bucket of a bucketed
dimension its own compile-time configuration (warps, stages, CTAs, the values of consts), each binary built, keyed and digested with its own.
`num_ctas=2` (Triton's compile option) runs a program on a thread-block cluster of 2 CTAs, and a prebuilt binary declares `cluster=(2, 1, 1)`: both hosts
launch it with `cuLaunchKernelEx` and the cluster, the grid counting clusters. `global_scratch=256` is the global scratch each CTA of the compiled kernel
addresses (device-side TMA descriptors write theirs there), which the build checks against Triton's metadata: Python allocates it per launch, the Rust
crate takes it as the last tensor of a launch (`Plan::tensor_names()` ends with `global_scratch`), and a plan gives its workspace. Profile scratch, PDL
and cooperative launches stay refused.

**A pinned cuBLASLt GEMM.** `@tilefish.library("gemm", acc=f32, impl="cublaslt.gemm")` over `(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16]) ->
Tensor[M, N, bf16]` computes `a @ b.T` with an f32 accumulator and one rounding (`impl="cublaslt.gemm_bias"` adds `bias: Tensor[N, bf16]`, added in f32 before it).
For each `(M, N, K, bias)` (`M` from its buckets) the algorithm that cuBLASLt's heuristic ranks first under a fixed workspace limit (8 519 680 bytes,
torch's default; `workspace_bytes=` changes it) is pinned: its 64 bytes go to the manifest's `library` section with the cuBLASLt version that chose
it and the SM count of the device it chose it on (`sm_count`: a MIG slice has fewer SMs than the whole GPU), and `replay` runs that algorithm and refuses
another cuBLASLt version and a device of another SM count ("re-export the manifest on this device"). `jit` pins in memory the same way, per device, so
`jit` and `replay` run the same algorithm. The binding is ctypes on the `libcublasLt` the process already has loaded (torch's); operands are contiguous and 16-byte aligned (strided
operands are not supported yet).

How the algorithm is chosen is part of the declaration (and of its digest):

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

* `heuristic="torch"` asks the heuristic as torch 2.9's `addmm` does (`at::cuda::blas::gemm_and_bias`): one result, under the workspace limit alone (no
  alignment minimums: torch's pointers are 256-byte aligned, the library's default), with torch's limits by default, 1 048 576 bytes for `gemm_bias`
  (`CUBLASLT_WORKSPACE_SIZE`) and 8 519 680 for `gemm` (`torch.mm` runs `cublasGemmEx` with `CUBLAS_WORKSPACE_CONFIG`'s workspace, and the heuristic under that
  limit answers what it runs; on an sm_90 exactly torch's default is 32 MiB, `workspace_bytes=33_554_432`). With torch's default reduction
  (`allow_bf16_reduction=True`) the result is torch's byte for byte: `test_gpu_cublaslt.py` asserts it over 48 shapes, with and without a bias, on three
  operand sets each, and it was measured on a B300 MIG slice (sm_103) and an RTX 5090 (sm_120) with cuBLASLt 12.8.4, and on the B300 slice with 13.1.0
  (`test_gpu_gemm.py`, 30 shapes). The default posture, `"tilefish"`, adds 16-byte alignment minimums, asks for 8 results and, without the opt-in, gives a
  GEMM with a bias no split-K (below), which torch does not do. It is not torch's, and how often its result is torch's depends on the cuBLASLt version: on
  the 15 biased shapes of `test_gpu_gemm.py`, all of them with 12.8.4, 13 with 13.1.0, whose heuristic ranks a split-K of 3 with f32 partial sums first
  for 32 and 64 x 1024 x 3072 (neither the workspace limit, nor the alignment minimums, nor the number of results asked changes its first answer there;
  the reduction-scheme mask does).
* `allow_bf16_reduction=True` lets split-K add its partial sums in bf16 (`INPLACE`, `OUTPUT_TYPE`: torch's default,
  `allow_bf16_reduced_precision_reduction`), which rounds more than once. Without it the heuristic is asked for f32 partial sums only, and a GEMM with a
  bias for no split-K at all (with the bias epilogue, a split-K with f32 partial sums was measured to round twice: cuBLASLt 12.8.4 on an RTX 5090); an
  algorithm that breaks either is refused by both hosts. The rule stays in the default posture although some split-K algorithms keep the one
  rounding: nothing in the heuristic's answer tells them apart (`docs/numerics.md`, section 4); `heuristic="torch"` gives `addmm`'s bytes.
* `algos=` imports the algorithms (`tilefish.AlgoTable(arch, sm_count, cublaslt_version, {(m, n, k, bias): config})`, a config being the nine
  `cublasLtMatmulAlgoConfig` attributes or the 128 hex digits of a pin): on the device a table is for, each entry pins the table's algorithm, encoded as
  cuBLASLt encodes it and held to the same checks (`cublasLtMatmulAlgoCheck` for the problem, the workspace limit, bf16 partial sums only with the
  opt-in); a device no table is for is refused, never handed to the heuristic.

A manifest entry records them as `heuristic`, `algo_source` and `allow_bf16_reduction`, which are there only when they are not the default
(`docs/manifest-schema-3.md`, section 9). A pin is the configuration the heuristic chose, encoded by the library: the 64 bytes the heuristic returns can carry
bytes that are no part of it, which change from process to process.

**f16, an f32 result, `b` as `[K, N]`.** The operands may be f16 (both of one type), the result f32 (and a bias has the result's type: cuBLASLt takes no
other, measured), and `layout="NN"` takes `b: Tensor[K, N, T]` and computes `a @ b` (the default, `"TN"`, is `a @ b.T` with `b: Tensor[N, K, T]`):

```python
@tilefish.library("up", acc=f32, impl="cublaslt.gemm_bias", layout="NN", heuristic="torch", allow_f16_reduction=True, workspace_bytes=64 << 20)
def up(a: Tensor[M, 7168, f16], b: Tensor[7168, 1536, f16], bias: Tensor[1536, f16]) -> Tensor[M, 1536, f16]: ...   # torch.addmm(bias, a, b), byte for byte

@tilefish.library("down", acc=f32, impl="cublaslt.gemm")
def down(a: Tensor[M, 3072, f16], b: Tensor[1024, 3072, f16]) -> Tensor[M, 1024, f32]: ...   # a @ b.T, rounded once to f32
```

The opt-in to split-K partial sums in the output type is named by the operands' type, as torch's flags are (`allow_f16_reduction` for f16 operands). Under
`heuristic="torch"` with it, the result is `torch.mm` / `torch.addmm` in f16 byte for byte in both layouts (`test_gpu_gemm_f16.py`: 63 shapes, three
operand sets each, measured on a B300 MIG slice with cuBLASLt 12.8.4). An `AlgoTable` row of the layout `NN` is keyed `(m, n, k, bias, "NN")`; a
configuration is encoded with the declaration's types. The entry says the form in `dtypes`, `transa` and `lda` (`docs/manifest-schema-3.md`, section 9.2).

**`a` stored `[K, M]`.** `layout="TT"` takes `a: Tensor[K, M, T]` with `b: Tensor[N, K, T]` and computes `a.T @ b.T`; `layout="NT"` takes it with
`b: Tensor[K, N, T]` and computes `a.T @ b`, for every operand type above:

```python
@tilefish.library("scores", acc=f32, impl="cublaslt.gemm", layout="TT", heuristic="torch", allow_f32_reduction=True)
def scores(a: Tensor[8, M, f32], b: Tensor[1024, 8, f32]) -> Tensor[M, 1024, f32]: ...   # torch.mm(a.t(), b.t()), TF32 off, byte for byte
```

Under `heuristic="torch"` the pin is what torch runs for `torch.mm(a.t(), b.t())` / `torch.mm(a.t(), b)` and `torch.addmm` of a contiguous `a [K, M]`, byte
for byte, for bf16, f16 and f32 (`test_gpu_gemm_layouts.py`, `M` from 2 to 2048; at `M = 1` torch multiplies `a [K, 1]` as the contiguous `[1, K]` it also
is, so a one-row GEMM is declared `TN` or `NN`). The entry says it in `transb` and `ldb` (`docs/manifest-schema-3.md`, section 9.2).

**f32, TF32 off.** f32 operands have an f32 result (and an f32 bias), and are multiplied as f32:

```python
@tilefish.library("proj32", acc=f32, impl="cublaslt.gemm_bias", heuristic="torch", allow_f32_reduction=True)
def proj32(a: Tensor[M, 1024, f32], b: Tensor[3072, 1024, f32], bias: Tensor[3072, f32]) -> Tensor[M, 3072, f32]: ...   # torch.addmm(bias, a, b.t()), TF32 off
```

The compute type is `CUBLAS_COMPUTE_32F`, never `CUBLAS_COMPUTE_32F_FAST_TF32` (what torch's `allow_tf32` selects) nor the bf16x9 emulation of cuBLAS 13
(`CUBLAS_COMPUTE_32F_EMULATED_16BFX9`). Both hosts refuse an f32 algorithm that would not multiply f32 as f32: one encoded for another compute type
(cuBLASLt writes it into the 64 bytes, and the TF32 and emulated algorithm ids cannot be encoded under `CUBLAS_COMPUTE_32F` at all), one whose
numerical-implementation flags say TF32 or 16-bit inputs, and every f32 pin in a process where cuBLASLt runs `CUBLAS_COMPUTE_32F` as TF32
(`NVIDIA_TF32_OVERRIDE=1`) or as the emulation (`CUBLAS_EMULATE_SINGLE_PRECISION=1`): measured, the encoding shows both. The opt-in to split-K partial sums
in the output is `allow_f32_reduction`; torch has no such flag and never restricts the reduction of an f32 GEMM, so its choice needs it. Under
`heuristic="torch"` with it and torch's workspaces (8 519 680 bytes for `mm`, `CUBLAS_WORKSPACE_CONFIG`'s; 1 MiB for `addmm`), the result is `torch.mm` /
`torch.addmm` in f32 with TF32 off, byte for byte, in both layouts (`test_gpu_gemm_f32.py`: 756 of 756 runs, 63 shapes with `M` from 1 to 2048, three
operand sets, `mm` and `addmm`, `TN` and `NN`, with the default workspace and with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and a 32 MiB limit; measured on a B300
MIG slice with cuBLASLt 12.8.4 and 13.1.0 and on an RTX 5090 with 13.1.0). torch's `mm` runs `cublasSgemm`, and cuBLAS chooses for it the algorithm the
cuBLASLt heuristic answers under that posture. See `docs/manifest-schema-3.md`, section 9.2.

**SM partitions.** On a green context (a partition of the device's SMs) every SM-count query still reports the device's count, and a pinned
algorithm computes the same bytes with or without a partition (measured, cuBLASLt 12.8.4); what changes is the heuristic's choice, through the
context of the process's first cuBLASLt handle. `sm_count_target=P` pins the algorithm the heuristic chooses for `P` SMs
(`CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET`, which every check and run sets too), the same in any context; the entry records it, and a Rust host on a
partition of `P` SMs takes only pins chosen for it with `Manifest::with_sm_partition(Some(P))` (`docs/manifest-schema-3.md`, section 9.3).

**A GEMM with a cuBLASLt epilogue.** `impl="cublaslt.matmul"` (RFC 0003, milestone E2) runs a bias, ReLU and a residual add (`beta * C`) on the f32
accumulator, before the one rounding, with the algorithm pinned for the epilogue:

```python
# out = relu((acc + c) + bias), rounded once to bf16: the library adds c before the bias
@tilefish.library("up", acc=f32, impl="cublaslt.matmul", epilogue="relu_bias", beta=1)
def up(a: Tensor[M, H, bf16], b: Tensor[F, H, bf16], bias: Tensor[F, bf16], c: Tensor[M, F, bf16]) -> Tensor[M, F, bf16]: ...
```

`epilogue` is `"none"`, `"bias"`, `"relu"` or `"relu_bias"` (cuBLASLt's `DEFAULT`, `BIAS`, `RELU`, `RELU_BIAS`), `beta=1` adds `c [M, N]`; the operands
are bf16 or f16 and the result their type, in any of the four layouts, with `heuristic`, `workspace_bytes`, the opt-ins and `sm_count_target` as above
(`algos=` is not taken yet). The bytes are the library's under the pin, so **every pin is verified when it is made** (`jit` and `export`): run on two
seeded operand sets with special values in the bias and `c` and rows of `a` that make NaN, zero, infinite and overflowing accumulators, and compared bit for bit with the formula computed in f32 on the
pin's own accumulator (its configuration re-encoded with an f32 result) and rounded once. The entry records `"measured-exact"` or `"binary-defined"`
with the count of elements that differ and their largest distance; `contract="measured-exact"` refuses a pin that is not. Measured on a B300 MIG slice
(14 shapes, bf16 and f16, both postures, seven cases: 392 pins per version, `test_gpu_matmul.py`): every pin is measured-exact with cuBLASLt 12.8.4; with
13.1.0, 372 are, and the 20 others are algorithm 21 at 32 x 1024 x 3072, which adds a bias to `c` in neither order (26% of the elements differ) and
whose ReLU gives `0` for a NaN where the others keep it. An epilogue gets no split-K without the opt-in, as a bias does. `export` records a canary
(`canary=True`, the default): the pin run on operands made of a seed on the device and the digest of its result, which a replay (Python) and a load
(the Rust and C++ hosts) run before the pin is used, refusing another digest: two builds of one cuBLASLt version are told apart. See
`docs/manifest-schema-3.md`, section 9.5.

**A pinned cuDNN convolution.** `impl="cudnn.conv_fwd"` declares a forward convolution by its signature, `(x, w) -> y` in the order of `layout`
(`"NCHW"`, the default: `x [N, C, *S]`, `w [K, C/G, *R]`; `"NHWC"`: `x [N, *S, C]`, `w [K, *R, C/G]`) with one to three spatial axes, f16, bf16 or f32
data, an f32 accumulator and one rounding, and pins what cuDNN runs for every bucket:

```python
T = Dim("T", buckets=[64, 646])

# the backend API (the default): an engine configuration, an engine and its knobs, chosen as torch chooses it
@tilefish.library("conv", impl="cudnn.conv_fwd", layout="NHWC", padding=3)
def conv(x: Tensor[1, T, 64, f16], w: Tensor[64, 7, 64, f16]) -> Tensor[1, T, 64, f16]: ...

# the legacy API: the algorithm enum of cudnnConvolutionForward, cudnnGetConvolutionForwardAlgorithm_v7's first deterministic answer that fits
@tilefish.library("down", impl="cudnn.conv_fwd", layout="NHWC", stride=2, padding=1, api="legacy")
def down(x: Tensor[1, T, 64, f16], w: Tensor[128, 4, 64, f16]) -> Tensor[1, tilefish.cdiv(T, 2), 128, f16]: ...
```

The declared result must be the convolution's output at every bucket (`(S + 2 * padding - dilation * (R - 1) - 1) // stride + 1`; `cdiv(T, 2)` is that
here). The math is part of the declaration: `"tensor_op"` (the default of 16-bit data) or `"fma"`, and for f32 data `"fma"` (TF32 off) or `"tf32"`, said
explicitly. `api="backend"` pins an engine configuration (the engine's global index and every knob choice: it names a kernel) under `heuristic="instant"`,
torch 2.9's posture with `deterministic = True` and `benchmark = False` (`CUDNN_HEUR_MODE_INSTANT`, then the fallback list, filtered for determinism, no
down-conversion and, for `"fma"`, no tensor cores), or `"mode_b"`; `api="legacy"` pins the algorithm enum. `algos=tilefish.CudnnTable(arch, sm_count,
cudnn_version, {(n, c, k, input, filter, stride, padding, dilation, groups): pin})` imports pins instead. The entry records the problem, the math, the pin,
how it was chosen, the cuDNN version (`cudnnGetVersion()`), the SM count and an `sm_count_target`; `replay` and the Rust host refuse another cuDNN version,
another SM count and a process with `NVIDIA_TF32_OVERRIDE` set (it changes what cuDNN picks behind the declared math), and vet the pin (a deterministic answer
of `cudnnGetConvolutionForwardAlgorithm_v7`; an engine of allowed notes whose knobs are in range and whose plan finalizes). Since two builds of one cuDNN
version answer the same `cudnnGetVersion()`, `export` also records a canary per entry, the digest of the pin's output on operands made of a seed (both
made on the device by one PTX module in both hosts: a plan set of 743 entries loads in 1 to 2 s), which `replay` and the Rust host run again (once,
on a stream of their own, never the NULL stream) and refuse when the digest differs; an entry without one loads as before. A knob of -1 in an imported pin
is cuDNN's "not set" and is left out. Every call sets cuDNN's stream to the one it is given. There is no bias (torch's cuDNN convolution adds it with a separate `add_`, as a tilefish
kernel does).

The `"instant"` pin is `torch.nn.functional.conv1d` / `conv2d` / `conv3d`'s bytes where torch runs cuDNN (`test_gpu_conv.py`: 15 problems, three operand sets,
1-D, 2-D and 3-D, f16, bf16 and f32 with TF32 off, strides, dilations, groups; a B300 MIG slice with cuDNN 9.10.2 and 9.13.0, an RTX 5090 with 9.13.0),
and a legacy pin of
`IMPLICIT_PRECOMP_GEMM` is torch's legacy path's. Both APIs run in launch plans (the workspace is the arena's) and capture in CUDA graphs; the binding is
ctypes on the `libcudnn` the process has (torch's), and the Rust crate loads cuDNN only for a manifest that has a convolution (`Manifest::conv`, `Conv`). See
`docs/manifest-schema-3.md`, section 9.4.

**A CuTe DSL kernel.** A kernel of CUTLASS's CuTe DSL (`nvidia-cutlass-dsl`: TMA, `tcgen05`, clusters, persistent tile schedulers, in Python) is an
extern too, compiled by the DSL and launched by every host as a prebuilt cubin with the launch the compiled host function makes (`docs/cute.md`):

```python
@tilefish.extern(cute=PersistentDenseGemmKernel(cutlass.Float32, True, (256, 128), (2, 1), True),   # a CUTLASS example's kernel class
                 cute_args=lambda a, b, c: (a.unsqueeze(-1), b.unsqueeze(-1), c.unsqueeze(-1), max_active_clusters))
def gemm(a: Tensor[M, 1024, f16], b: Tensor[2048, 1024, f16], c: Out[Tensor[M, 2048, f16]]): ...   # static shapes: one binary per bucket of M
```

The build compiles it for the device (the DSL's own cubin and PTX) and calls the compiled host function on a stream captured into a CUDA graph, so the
kernel does not run: the grid, block, cluster and dynamic shared memory, and every kernel parameter, which must be a tensor's pointer, a TMA descriptor
the DSL encoded (a template the host writes the tensor's address into) or constant bytes, checked over three captures at other addresses. The entry is a
`binary` entry of source `cute` (section 4.9 of the schema), pinned like any other; its identity is the PTX, which a Python replay compiles again. On a
30-SM partition of an sm_103 GPU the persistent tcgen05 GEMM of the examples writes the bytes of the pinned cuBLASLt GEMM, and runs at 1.17x its speed at
`4096 x 2048 x 1024` (0.92x at `M = 1024`). Install the DSL yourself at the tested version and pin it (`nvidia-cutlass-dsl==4.8.1`: it is in beta, and
another version is another declaration). It is NVIDIA software under its own licence, which tilefish neither ships nor depends on, and the cubins
and PTX an export writes are its output: shipping them is your responsibility (`docs/cute.md`, section 6).
