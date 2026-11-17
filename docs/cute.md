# CuTe DSL kernels as externs

CUTLASS's CuTe DSL (the `nvidia-cutlass-dsl` package, `import cutlass.cute as cute`) writes peak-performance kernels with asynchronous MMA pipelines
(TMA, `tcgen05`, warp specialisation, clusters) in Python. tilefish takes such a kernel like any other extern: it compiles it to a cubin per
architecture, pins it, and launches it from Python, Rust and the C ABI, in a single launch, a launch plan and a CUDA graph, with the launch
attributes the compiled kernel has (its grid, block, cluster, dynamic shared memory and TMA descriptors). The manifest entry is specified in
`docs/manifest-schema-3.md`, section 4.9; the code is `python/src/tilefish/_cute.py`.

Contents: 1 declaring one; 2 how it is built; 3 what a host launches; 4 identity, pins and reproducibility; 5 limits; 6 the licence; 7 how it is tested.

## 1. Declaring one

A CuTe DSL kernel is a `@cute.jit` host function that launches a `@cute.kernel`. The declaration names the host function (or an object whose `__call__`
is one: an instance of a kernel class of the CUTLASS examples) and a stub of the operator's **tensors**:

```python
import cutlass
import tilefish
from tilefish import Dim, Out, Tensor, f16
from dense_gemm import DenseGemmKernel  # CUTLASS examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py

M = Dim("M", buckets=[256, 1024, 4096])

@tilefish.extern(
    cute=DenseGemmKernel(cutlass.Float32, True, (256, 128), (2, 1), True),   # f32 accumulator, 2-CTA MMA 256x128, a 2x1 cluster, a TMA store
    cute_args=lambda a, b, c: (a.unsqueeze(-1), b.unsqueeze(-1), c.unsqueeze(-1)),  # its (M, K, L) tensors, L = 1
)
def gemm(a: Tensor[M, 1024, f16], b: Tensor[2048, 1024, f16], c: Out[Tensor[M, 2048, f16]]): ...

c = tilefish.run(gemm, a, b)  # a @ b.T, compiled by the DSL and launched by tilefish
```

* **The stub** lists the operator's tensors only (`Tensor`, `Out`, `InOut`; dense), in the order the kernel takes them, **of static shapes**: every
  axis is a number or an expression of bucketed dimensions (`Dim("M", buckets=[...])`). The DSL compiles a kernel for the layouts it is given, so
  tilefish compiles one binary per bucket (bucket semantics are exact, as everywhere). A `Strided`, a `TensorDesc`, a `Null`, a scalar or a `const` is
  refused: the kernel's parameters are the compiled kernel's.
* **`cute_args=`** makes the positional arguments of the host function from the tensors: torch tensors in the order of the stub, of the bucket's shapes,
  16-byte aligned. It returns them as the host function takes them; a torch tensor anywhere in them (a list of inputs too) becomes a CuTe tensor of its
  static layout (`from_dlpack(t, assumed_align=16)`), anything else is passed as it is (a `Constexpr` such as `operator.add`, a value of the
  configuration). A view must start at its tensor's address (an added axis, a permutation). tilefish appends the stream: the host function takes it
  last. Without `cute_args` the tensors are passed as they are.
* **`cute_options=`** are compile options of the DSL (`"--opt-level 3"`). tilefish passes `--keep-cubin --keep-ptx` itself, for the architecture of
  the device, without line information; an option that changes those is refused.
* `sass=`, `index_bits=`, `reference=` and `name=` are as for any extern. A grid, a block, a cluster, shared memory and Triton's compile options are
  the compiled kernel's, so they are not arguments.
* **An implementation per architecture.** The tcgen05 GEMM of sm_100 and sm_103 does not run on sm_120: `tilefish.impl(cute=..., cute_args=...,
  cute_options=...)` is one implementation of an operator declared with `by_arch=`, beside Triton kernels and prebuilt binaries:

  ```python
  tcgen05 = tilefish.impl(cute=DenseGemmKernel(...), cute_args=batched, name="tcgen05")
  @tilefish.extern(by_arch={"sm_100": tcgen05, "sm_103": tcgen05, "sm_120": tilefish.impl(cute=Sm120GemmKernel(...), cute_args=sm120_args)})
  def gemm(a: Tensor[M, 1024, f16], b: Tensor[2048, 1024, f16], c: Out[Tensor[M, 2048, f16]]): ...
  ```

The declaration reads the DSL's version (`importlib.metadata`) and the file of the host function; it imports neither the DSL nor torch. The DSL
must be installed, at the version tilefish is tested with: `pip install nvidia-cutlass-dsl==4.8.1` (Linux or Windows, a driver of CUDA 12.9 or 13).
Pin it in the lock file of every environment that exports or replays (section 4): the DSL is in beta, and another version is another declaration.

## 2. How it is built

`export`, `jit` and a plan's export build a CuTe DSL kernel **on a device of its architecture** (its launch is captured there, so a cross-build is
refused; a cuBLASLt pin is per device too):

1. **Compile.** `cute.compile(host, *cute_args(tensors), stream, options="--keep-cubin --keep-ptx --dump-dir=<a temporary directory> <cute_options>")`
   (it reads no cache of the DSL's, and the PTX and cubin it keeps go to a private directory, not the working one): Python to MLIR, MLIR to PTX with the NVVM the DSL carries, PTX to a cubin with the `ptxas` it carries, for the
   device's architecture (`sm_103a`). The host function must compile to exactly one kernel. The cubin and the PTX are kept; the cubin's ELF header
   must be of the architecture and have the kernel's code section.
2. **Capture the launch, three times.** The compiled host function is called on a stream that is being captured into a CUDA graph, so the kernel does
   not run, each time with every tensor at another address. The graph must hold exactly one operation, a kernel node of the compiled kernel; from it
   tilefish reads the grid, the block, the dynamic shared memory, the cluster and its scheduling policy, the launch attributes, and the bytes of every
   kernel parameter (`cuFuncGetParamInfo` sizes, which must be the PTX entry's).
3. **Make the ABI** (`classify_params`). The bytes of a parameter the kernel reads are those its PTX loads (`ld.param` at a constant offset:
   `ld.param.v2.b8 {%rs2, %rs3}, [k_param_4+8]` reads the bytes 8 and 9), or all of it when the code takes its address otherwise (a TMA descriptor,
   which `cp.async.bulk.tensor` reads by its address). Every byte the kernel reads is the same in the three captures, or is the address of one of the
   operator's tensors (an aligned 8-byte word, read whole, equal to that tensor's address in each capture). A parameter is then a `pointer` (8 bytes, a
   tensor's address), a `tensormap` (128 bytes aligned to 64 whose first word is a tensor's address: a TMA descriptor as the DSL's host code encodes it
   for a static shape; its bytes 64 to 127, which that code does not write, are left out and passed as zeros) or `bytes` (constants: a static layout, an
   empty structure, a value compiled for the bucket, the fast divisors of a persistent tile scheduler). A byte the kernel does not read (the padding of a
   structure, which the host function leaves as it was on its stack) is not compared and is passed as zero, so a rebuild writes the same manifest.
   Anything else is refused, with what it is: a byte that varies and is no address (a pointer the host function allocates, a dynamic shape), an
   address inside a structure, a tensor in two parameters or in none, the tensors in another order than the stub's.
4. **Check the launch.** The block must be the PTX entry's `.reqntid` and the cluster its `.reqnctapercluster` when it has them; a cluster of at most 8
   CTAs with the spread (or default) scheduling policy, no programmatic dependent or cooperative launch, a grid of blocks that is a whole number of
   clusters (the entry's grid counts clusters, section 4.5 of the schema), and no function attribute but the dynamic shared memory size.
5. **SASS assertions** (`sass=tilefish.sass(...)`) are judged on the cubin and its PTX as for a prebuilt binary (`docs/explain.md`), and a traced export
   (`trace=True`) keeps the PTX the DSL generated and the SASS (`trace/<stem>.<arch>/ptx.ptx`, `sass.txt`). Probes are not available (the DSL compiles
   the kernel, not Triton).

Measured on sm_103 with nvidia-cutlass-dsl 4.8.1 (cu12 libraries: NVVM of CUDA 13.4, `ptxas` 12.9.83): the examples' tcgen05 GEMMs compile in about
0.25 s per bucket and the elementwise kernel in less. The non-persistent GEMM's ABI is a 64-byte constant and three TMA descriptors (A, B and the TMA
store of C), a cluster of 2 CTAs and 213 248 bytes of dynamic shared memory; the persistent one adds the four fast divisors of its tile scheduler
(12-byte structures whose last 2 bytes are padding the kernel does not read), and its grid is the clusters the device runs at once.

Speed, through `tilefish.run` replays captured in a CUDA graph (device time per call), `C = A @ B.T` in f16 with an f32 accumulator, `N = 2048`,
`K = 1024`, on a 30-SM partition of an sm_103 GPU, against the pinned cuBLASLt GEMM of the same problem (`impl="cublaslt.gemm"`):

| M | persistent tcgen05 (2-CTA 256x128, cluster 2x1) | non-persistent tcgen05 (the same tiles) | cuBLASLt |
|---|---|---|---|
| 256 | 7.1 µs (0.71x) | 10.2 µs (0.49x) | 5.0 µs |
| 1024 | 14.2 µs (0.92x) | 25.1 µs (0.52x) | 13.0 µs |
| 4096 | 44.4 µs (1.17x, 387 TFLOP/s) | 88.6 µs (0.59x) | 52.1 µs |

With 256x256 tiles the persistent GEMM is 1.25x cuBLASLt at `M = 4096` (408 TFLOP/s). Every configuration writes the bytes of the cuBLASLt pin (the same
tcgen05 accumulation order), so it can replace it without changing a bit of a model's output.

## 3. What a host launches

The entry is a `binary` entry (`binary.source` `"cute"`), so every host launches it as a prebuilt cubin, with exactly its parameters and no scratch
slot, `cuLaunchKernelEx` with its cluster when it has one, and the dynamic shared memory attribute set above 48 KiB:

* a `pointer` is the tensor's address;
* a `tensormap` is its template with the tensor's address, a little-endian `u64`, in its bytes 0 to 7, passed by value (the address must be 16-byte
  aligned, which the tensor's `pointer_alignment` precondition checks, and below 2^57);
* `bytes` are passed as the manifest has them.

A call (and a step of a plan) gives the tensors only, in the order of the stub; the dimensions bind from their shapes and the binary of the bucket is
chosen as for any extern. The grid is the entry's integers (the shapes are static). Python (`tilefish.run`, plans, `tilefish.capture`), the Rust crate
(`Kernel::launch`, `launch_raw`, `LaunchPlan`) and the C ABI do the same, from the manifest alone; a CUDA graph holds the descriptors of the addresses
it was captured with, as it holds their addresses.

## 4. Identity, pins and reproducibility

* **The declaration digest** has the host function (`module:qualname`), the sha256 of its file, the DSL's version and the options: another file,
  another DSL or other options is another declaration, whose binaries a replay refuses (`first difference at declaration_digest`).
* **The binary is identified by its PTX** (`source_digest`, kind `tilefish.cute.v1`), as a Triton kernel by its IR: Python's replay compiles the
  declaration with the DSL of the process (it must be the declaration's version; the PTX of each spec is made once per process) and refuses an entry
  whose PTX differs, pinned or not. Where it cannot compile it (no DSL, another version), it accepts the entry only with `pin=`. The Rust host and the C
  ABI need no DSL: the pin covers every field and every file, as always.
* **Pin the DSL's version** in the environment that exports and replays (`nvidia-cutlass-dsl==4.8.1` in a lock file): the binary is the DSL's output,
  and the DSL is in beta.
* **Reproducible.** Two exports of the same operators with the same DSL on the same architecture are the same files and the same manifest digest, from
  another process and another directory (measured on sm_103: the GEMM and the elementwise kernels, cubins and PTX byte for byte). The DSL writes no
  path into the binary without line information, which tilefish does not ask for.
* The manifest records the DSL's version (`binary.cutlass_dsl`) and the NVVM and `ptxas` releases it carries (`toolchain`), as read from the PTX
  header and the cubin's `.note.nv.tkinfo`.

## 5. Limits

* **Static shapes only.** A dynamic layout (`mark_layout_dynamic`, a `SymInt`, an `Int32` shape of the host function) passes its shape inside a
  structure, or computes values on the host (a fast divisor) that tilefish would have to reproduce from the dimensions: refused, as section 2 says.
  Declare the dimension bucketed (one binary per bucket).
* One kernel per host function, launched on the stream tilefish passes, and nothing else enqueued (no memset, no second kernel, no allocation the
  kernel takes the address of).
* A tensor is a parameter at its own address (offset 0): a view that `cute_args` makes starts at the tensor.
* A persistent kernel's grid is the host function's on the device it was built on (for example `max_active_clusters`): the binary is right on a
  device of another SM count, its grid may not fill it. Export on the device shape it runs on.
* The descriptors are the DSL's own encoding of a static shape (the DSL writes them in its host code, not with `cuTensorMapEncodeTiled`). tilefish
  relies only on what every build checks again: that its first word is the address and the rest constant. A DSL that encodes them otherwise is
  refused by the build, not misread.

## 6. The licence

The CuTe DSL is NVIDIA software under the NVIDIA Software License Agreement (a proprietary licence; it is in beta). tilefish does not ship it, depend
on it, or import it unless a CuTe DSL kernel is declared; install it yourself and read its terms. The cubins and PTX an export writes are the DSL's
output from your kernels, and what you do with them is your responsibility: check the agreement before you ship or redistribute them (in a
manifest directory, a package or an embedded host). The CUTLASS examples the tests use (`examples/python/CuTeDSL`, BSD-3-Clause)
are not copied into this repository: the tests load them from a checkout (`TILEFISH_CUTLASS_EXAMPLES`).

## 7. How it is tested

* `python/tests/test_cute_externs.py` (no GPU, no DSL: the DSL's version is replaced): declarations and their refusals, the compile specs and digests,
  `classify_params` and `launch_of` on synthetic captures (a TMA GEMM's, a plain kernel's, every refusal), the PTX parsers, the manifest entry, every
  refusal of the reader, a replay's checks of the PTX digest, the Rust and C++ emitters, and the bytes a launch packs (through a driver that records
  them).
* `python/tests/test_gpu_cute.py` (a GPU, the DSL and `TILEFISH_CUTLASS_EXAMPLES`): the tcgen05 GEMM (sm_100, sm_103) or the sm_120 GEMM and the
  elementwise kernels of the CUTLASS examples; tilefish's launch is byte for byte the DSL's own launch; `add` is torch's bytes and the GEMM within an
  f16 ulp of the reference; an export is the same files in another process and directory; a pinned replay is the bytes of `jit`; stale and other
  declarations are refused; a plan of a CuTe GEMM and a CuTe add replays and captures with the eager bytes; SASS assertions and traces; the GEMM's time
  against the pinned cuBLASLt GEMM (printed).
* The Rust and C++ gates (`python/tests/rust_gate.py`, the cases of `cute_kernels.py` when the DSL and the examples are there): the Rust and C++ hosts
  write the bytes of Python's replay.
* `crates/tilefish/tests/`: the crate's reader refuses what section 4.9 of the schema refuses, and packs the descriptors and bytes as Python does.
