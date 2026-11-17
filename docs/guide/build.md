# How a build works

```sh
tilefish build --arch sm_90,sm_100,sm_120 --host python,rust,cpp
```

1. Trace the graph into a typed IR and check it: shapes, dtypes, divisibility,
   aliasing, and which architectures each operator supports.
2. Run the rewrite rules ([`rewrites.md`](rewrites.md)) and settle the final list of kernels.
3. Lower each `@kernel` to Triton source, written to disk for review.
4. Derive Triton's compile inputs from the types: pointer dtypes and 16-byte
   alignment, `constexpr` values from static dimensions, divisibility hints
   from `divisible_by`, and one binary per bucket for bucketed dimensions.
   Compile ahead of time to one cubin per architecture.
5. Write a manifest: symbol, parameter ABI in order, grid rule, block size,
   shared memory, launch attributes, toolchain versions, and a digest of the
   machine code.
6. Generate the hosts from the manifest and the types: a `torch.library`
   custom op for Python, a header for C++, a `cudarc` crate for Rust. Every
   precondition that selected a binary (alignment, divisibility, static sizes)
   becomes a check in the generated launcher, so no host can run a binary
   under assumptions it was not compiled for.

Library operators put a different code object behind the same kind of
manifest entry: a CUTLASS instantiation with a C entry point, a CuTe DSL
ahead-of-time export, or a pinned cuBLASLt algorithm recorded with its
workspace size and replayed as is (the last exists: a bf16 GEMM, in [`operators.md`](operators.md)).

The Python side has three modes:

| mode | what runs | used for |
|---|---|---|
| `jit` | the current source, built through its compile spec (the path of the build) in memory and launched through the CUDA driver | iterating on a kernel |
| `reference` | the same bodies interpreted with torch | an oracle, compared within a tolerance |
| `replay` | the prebuilt cubins of a manifest directory, checked against its digests, through the CUDA driver; it needs a trust anchor, the manifest's pinned digest or an explicit `trust_directory=True` | the exact machine code the C++ and Rust hosts run |

```python
y = tilefish.run(residual_add_rmsnorm, x, residual, w)                        # jit
m = tilefish.runtime.export(residual_add_rmsnorm, "build/")                    # cubins and manifest.json
y = tilefish.run(residual_add_rmsnorm, x, residual, w, mode="replay", manifest="build/", pin=m.digest)           # a manifest whose digest you hold
y = tilefish.run(residual_add_rmsnorm, x, residual, w, mode="replay", manifest="build/", trust_directory=True)  # whatever the directory holds
```

`replay` refuses to run without one of the two (`pin=` or `trust_directory=True`). This
is a deliberate breaking change in 0.0.x: until 0.0.2 a replay trusted the directory by
default, and the digests in it sit next to the files they describe, so whoever can write
the directory could make it run any binary.

A pin is of one exported artifact, not of a recipe: the manifest digest covers the cubin
files byte for byte. Exports are reproducible. A cubin's debug sections name each source
by its module path (`kernels/rms.py`, `tilefish_generated/softmax.py`,
`triton/language/standard.py`), not by the checkout, the virtual environment or a scratch
directory, and a generated kernel's source is written to a file addressed by its sha256.
So the same graph exported again with the same toolchain (Triton, `ptxas`, cuBLASLt,
cuDNN) on the same kind of device, from another directory, in another process or on another
machine, gives the same files, the same manifest digest and the same plan digest, and a
recorded pin checks a rebuild. A manifest exported by tilefish 0.0.6 or earlier names the
directory it was built in; it loads and replays as before, and only its digest is not what
a new export of the same graph gives. Pin what you exported, and ship that directory.
For a profiler that wants the absolute source paths in the cubins (rather than resolving
`kernels/rms.py` against a source directory), build with `TILEFISH_DEBUG_PATHS=absolute`.
It is off by default and breaks reproducibility: that export's cubins, `sha256`s and digests
name where it was built (the `sha256_code`s are the same), and its binaries are cached apart.

Arguments are torch tensors, matched to the signature the way a call is
matched in a trace: inputs and `InOut` tensors are passed (an `InOut` tensor
is updated in place), `Out` tensors are allocated and returned. Before a
launch every precondition of the binary is checked (dtypes, static sizes,
`divisible_by`, 16-byte aligned pointers, contiguity), and a bucketed
dimension must equal a bucket exactly: there is no padding.

`jit` and `replay` launch the same way, so they run the same machine code: a
cubin that `jit` builds has the `sha256_code` of the one in the manifest.
`replay` refuses a cubin whose digests are not the manifest's, a manifest
entry that is not what the operator's compile spec asks for, and a stale
binary: a kernel redeclared with another body but the same name and signature
does not replay its old binary.

**What a binary is, is the Triton IR its kernel compiles to.** Which kernel
Triton compiles depends on everything the Python process holds when its
frontend runs: the text of the function and of the jit functions it calls, the
module constants it reads (`cfg.SCALE`, `getattr(cfg, "SCALE")`, `getattr(cfg,
"SC" + "ALE")`, a class attribute `cfg.Inner.SCALE`), closure variables, a
`triton.constexpr_function` that reads the environment, a `tl.constexpr` of an
object whose `repr` hides its state, a helper two modules away, a user module
named like a standard library module. Five rounds of review found a route that
each way of *looking for* these (a digest of the files a kernel reaches, the
module attributes, the closures, a watch on the imports) did not follow, and each
was a stale binary served by the build cache or by an export replayed with a
pin. So nothing is looked for. Triton's frontend folds every one of them into
the IR it builds, and that IR is hashed (`tilefish._ir`): the TTIR that
`ASTSource.make_ir` builds, printed with `Module.str_nodebug()` (no locations,
no names of values). Where the kernel is (its path, its line numbers), comments,
blank lines and the names of local variables are not in it, so none of them is an
edit; the operations, the types, the constants and the functions that are called
are. Whatever the process holds is what is hashed, because that is what is
compiled. The digest of every build is in `BuiltKernel.ir_digest`, and in the
manifest entry as `source_digest` with `source_digest_kind` `"tilefish.triton-ir.v1"`
(the fields schema 2 already had, so a reader of the manifest reads it as before;
a lowered kernel has its `source_sha256` as well, which needs no Triton). A Gluon
kernel's IR is the TTGIR that Gluon's frontend (`GluonASTSource`) builds, with its
layouts and `num_warps` in it, and its kind is `"tilefish.gluon-ir.v1"`.
`replay` recomputes the digest from the kernel the process holds, when Triton
can be imported and is the version that built the entry, and refuses a mismatch
as stale, with a pin or not (without Triton, or with another version, whose IR
may print differently, only a pin vouches for a hand-written kernel; the versions
are compared first, so a Triton whose frontend differs is never asked to run it).
The frontend costs 2 to 8 ms per kernel on a B300 (no compile, no GPU), once for
each build, `jit` call and `replay` call. It runs under the same lock as a
compile, so no two frontends, a compile's own included, run at once in any threads.
An export (and the `prepare` of a plan) runs it once per distinct input, not once per
bucket: the buckets of one `by_bucket=` configuration are one IR, one digest and one
binary of the build cache. What is kept of a run is its digest, not its IR, and the
module is freed (Triton's binding keeps every module it builds for the life of the
process otherwise, about twice its text, at every frontend and every compile). An
export of 132 buckets in three configurations of a Gluon kernel with a large IR
(`python/tests/irmem_export.py`, on a B300) took 77 s and peaked at 1.27 GB of
resident memory; it now takes 9 s and peaks at 0.40 GB (2.7 s and 0.25 GB with a warm
cache; traced, 2.66 GB before and 0.43 GB now), and writes the same bytes. A plan
set is read back against the one manifest the export wrote, not a copy of it and its
binaries per plan: a set of 1319 plans over 147 binaries and 1199 GEMM pins peaked at
54 GB and now peaks at 1.3 GB.

So an edit that was not reloaded is not a special case. `export` in the process
that imported the module compiles, and stamps, what that process holds; a fresh
process with the edited file computes another digest and refuses the manifest.
Triton's own `cache_key` is not used, and a build never reads Triton's disk
cache. What was removed: the digest of the files a kernel reaches, the import
watch (`SourceFileLoader.exec_module` is not wrapped and `TILEFISH_IMPORT_WATCH`
is gone), the snapshot of module attributes and the refusal of a reassigned one
(a value that a program assigned is in the IR), and the walk over the values
and closures of a kernel with its refusal of a value that has no stable `repr`.
What stays: the check, for a *lowered* kernel, that its module file has the
`source_sha256` of its spec and that the function the process holds was
imported from those lines (the IR digest says what the process holds, not that it
is the source the spec claims); that `build(spec, fn=)` compiles the function
the spec names and no other; and the refusal of whatever makes Triton compile
something other than the IR: its kernel override (`TRITON_KERNEL_OVERRIDE`) and
the compile option `ir_override` replace the IR of a stage with a file (so the
binary would not be of the IR), and an active instrumentation plugin
(`CUDABackend.instrumentation`) patches the passes.

A build compiles from scratch (25 to 120 ms per binary on a B300) unless
tilefish's own cache has it: `~/.cache/tilefish`, or `$TILEFISH_CACHE_DIR` (an
empty value switches it off; `aot.build(..., cache=False)` skips it for one
build). Its key is the cache recipe (a number that is bumped whenever tilefish
changes what goes into a binary) and the tilefish version, the IR digest, the
compile options, the architecture, the Triton and `ptxas` versions, and the
Triton settings that change code generation: every environment variable of the
families `TRITON_`, `MLIR_`, `LLVM_`, `NVPTX_`, `AMDGCN_`, `DISABLE_`, `ENABLE_`,
`ALLOW_` and `USE_IR_LOC` except those known not to (cache and dump directories,
tool paths, debugging switches), which is more than the 30 that Triton lists as
cache-invalidating (it leaves out `TRITON_LIBDEVICE_PATH`,
`TRITON_MOCK_PTX_VERSION` and `TRITON_KERNEL_OVERRIDE`, among others), the
effective value of each codegen knob of `triton.knobs` and the sha256 of the
libdevice file when a path is set. A setting that names a file Triton reads is
keyed by the sha256 of the file, not by its path, so a file rewritten under its name
is another binary: the compile option `extern_libs`, `TRITON_LIBDEVICE_PATH` and
`LLVM_PASS_PLUGIN_PATH` (a file that cannot be read refuses the build); the rest
of what Triton 3.5.1 has that names a path is a tool (`ptxas`, whose version is in
the key), an output, or the launcher's, and `_cache.py` lists it. Every read is verified: the `sha256` and
`sha256_code` of the cubin are recomputed (and the PTX, the entry point and its
ABI, the options, the IR digest and the versions are checked), and an entry that
is not as it was written is a warning and a rebuild, never used, and a cache
that cannot be read or written never fails a build, also under `python -W error`
(the warning is then logged on `tilefish.cache` instead of raised). Entries are
written through a temporary file and renamed, so builds that race do not tear
one. A lookup is a run of the frontend and a read: 3 to 6 ms. Nothing evicts the
cache, it is no defence against someone who can write the directory (a manifest
is what to pin), and it tells a patched Triton that keeps its version string from
the original no better than the version does. The same directory holds, under
`generated/<sha256>/tilefish_generated/`, the Triton source of each lowered kernel
and generated op (Triton reads a kernel's source from a file): written atomically,
used only when it holds exactly that source, and, when the directory cannot be
written, kept in a private scratch directory of the process instead (with one
warning). Where it is does not reach a binary.

`replay` also refuses `shared` memory the cubin and the device cannot have (the
driver's static shared memory of the function plus the manifest's dynamic size
must fit the device's opt-in maximum), an argument list that does not fit the
ABI, and a grid beyond the device's limits. The manifest is read as strict JSON
(no duplicate keys, no `NaN`, no number that does not fit a float), files are
written through a temporary file and renamed, and a symbolic link where a file
is to go is refused.

**An IEEE f32 dot is not built into TF32 MMAs.** A tensor core has no f32 multiply: an f32 MMA rounds its inputs to TF32, 10 of the 23 bits of
the mantissa. Triton 3.5.1 nevertheless compiles some `tl.dot(a, b, input_precision="ieee")` of f32 operands to `tcgen05.mma ... kind::tf32` on
sm_100 and sm_103: its MMAv5 path skips the precision check when an operand derives from a load narrower than 32 bits (an f16 or bf16 tile
converted to f32) or from `tl.join`, and the tile has `M % 64 == 0`, `N % 16 == 0` and 4 or 8 warps. Measured on a B300 with `a = 1` (f16),
`b = 1 + 2**-12`, 16 terms: every output is 16 instead of `16 + 2**-8`; the same kernel at `M = 32`, or on sm_90 and sm_120, is exact. So every
build (`jit`, `export`, a plan, a cache hit) refuses a kernel whose IR asks for an IEEE f32 dot when its binary has TF32 tensor-core MMAs made of
such a dot (`tilefish._tf32guard`). The check reads what the compiler made, not a copy of Triton's rule: the dots and their precision in the TTIR
the frontend builds, and the TF32 MMA instructions of the PTX (`tcgen05.mma` of `kind::tf32`, `wgmma` and `mma.sync` of `.tf32` operands). A kernel
whose f32 dots all ask for IEEE is refused when its binary has any TF32 MMA; one that also has dots that allow TF32 is judged dot by dot, each `ttng.tc_gen5_mma` of
its TTGIR traced to the TTIR dot it was made of by source location, and an MMA it cannot trace is refused too. A dot that says
`input_precision="tf32"` or `"tf32x3"` asks for TF32 and is never refused. The error names the dot and the fixes: a tile of fewer than 64 rows
for that dot, the product written as a loop of `tl.fma` (no `tl.dot`), or `input_precision="tf32"` where TF32 is what the kernel may compute.
Gluon kernels are checked the same way (in Triton 3.5.1 Gluon has no IEEE f32 dot that compiles: `warpgroup_mma` refuses one, and `tcgen05_mma`
of f32 operands is TF32 by the kernel's own choice); a lowered `@kernel` has no dot. A prebuilt PTX or cubin has no IR to say what it asked for and
is not checked, and neither is a manifest exported before this check (re-export it).

**Trust.** A replay needs a trust anchor, `pin=` or `trust_directory=True`, and
raises without one. The digests in a manifest directory sit next to the files
they describe, so whoever can write the directory can rewrite both.
`export` returns the manifest with its `digest`, a sha256 over every entry, and
checks that the directory it wrote has that digest; keep it somewhere you trust
and pass it as `pin=` to refuse a directory that changed.
`trust_directory=True` is the explicit statement that whatever the directory
holds is fine (a directory you wrote a moment ago); it does not combine with
`pin=`. A pin vouches for the manifest, not for an IR digest that is known to
differ, and not for a `shared` that is nonzero but too small for the kernel
(nothing but the manifest says how much it needs: the cubin of a Triton kernel
does not record its dynamic shared memory, measured in
`python/tests/test_gpu_shared_size.py`; the export writes Triton's value, which
the pin covers, and `trust_directory=True` trusts it with the cubins). Triton's own JIT launcher,
which specializes on argument values, is not a mode. Neither `jit` nor `replay`
needs a C compiler, and a `replay` of a generated kernel needs neither Triton nor
`ptxas`.

CI asserts, on a GPU (the tests marked `gpu`), that a `jit`-built cubin has the
digest of the manifest's, that `jit` and `replay` agree byte for byte, and that
`replay` stays within tolerance of `reference`. The tolerances are measured per
input distribution and dtype (f32, bf16 and f16 in and out; normal, uniform,
heavy-tailed, sparse outliers, values of every magnitude) and tabulated in
`python/tests/test_gpu_reference.py`: a ULP
count only means something where the result is far from zero (a layernorm of
data with a large mean has tiny outputs), so a bf16 output is bounded by half a
bf16 step of the largest reference value and a fraction of elements that may
differ, the f32 softmax by an absolute error of 5e-7, the gelu by 3e-5. An
experiment that passes in `replay` is, kernel for kernel, what production runs.
