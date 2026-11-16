# SASS traceability: traces, `tilefish-explain` and SASS assertions

Every kernel goes DSL / Triton / Gluon → TTIR → TTGIR → LLVM IR → PTX → `ptxas` → SASS. Tracing a hot spot or a bit difference back through that
chain used to mean dumping and diffing every level by hand, and sampling profilers attribute the time of asynchronous `wgmma` / `tcgen05` code to the
points that wait for it rather than to the work. tilefish makes the SASS of every kernel explainable and checkable:

1. **A trace** of an export keeps every level of every binary, outside the pinned artefact (section 1).
2. **`tilefish-explain`** annotates the SASS of a kernel, a plan step or a fused region with the source lines and graph ops that produced each
   instruction, summarises the instruction mix, counts per source line and diffs two binaries (section 2).
3. **SASS assertions** are part of a declaration and checked at every build like the TF32 guard and the contraction check (section 4).

The code is `tilefish._trace`, `tilefish.explain` and `tilefish._sass`.

## 1. The trace

```python
runtime.export(ops, "build/out", trace=True)
```

`trace=True` (or `TILEFISH_TRACE=1` in the environment, when `trace` is not given: for an export run by a script you do not edit, a CI job or a
model's own export script; `0` or unset is no trace, anything else is refused) writes `build/out/trace/`. The argument is the API; the variable exists
because the export is often not your code. Kernels and externs, a graph and a plan set are traced alike.

```
trace/
  index.json                         tilefish.trace.v1: every file and its sha256, the manifest digest, the binaries' sha256, trace_digest
  <stem>.<arch>/
    source/<module-relative name>    the source files the line information names, as the debug information names them
    frontend.mlir                    the IR the frontend built, with its locations (a Gluon kernel: TTGIR)
    ttir.mlir  ttgir.mlir  llir.ll   Triton's stages (a Gluon kernel has no TTIR)
    ptx.ptx                          the PTX (a prebuilt PTX extern: its PTX)
    sass.txt                         nvdisasm -c -g -gi: the SASS with its line information
    lines.json                       the line table: per instruction its address, source file and line, and PTX line (nvdisasm -gp)
    nodes.json                       a lowered kernel or a fused region: what each line of its generated source was written for
  graphs/<graph>.json                a graph: the steps of each bucket of its plan, and the ops of the graph body each fused region took
  graphs/source/<name>               the files of the graph body those ops were written on
```

* **Sources** are stored under the module-relative names the binary's debug information gives them (`kernels/mm.py`, `tilefish_generated/rmsnorm.py`,
  `triton/language/standard.py`: `tilefish._source.source_name`), so a trace is read from any directory and on any machine. A name that is not one
  (`TILEFISH_DEBUG_PATHS=absolute`) is not copied, and the index says so.
* **`nodes.json`** maps a line of a generated source to `{"node": <index in the kernel's graph>, "op": "mul", "loc": "model.py:17"}` (`loc`: the line of
  the `@tilefish.kernel` body, or of the graph body for a fused region, the op was written on), `{"what": "load x"}` for code of no op, and for the lines of a
  helper function (the inline PTX of a `mul.rn.f32`, a `cvt.rn.bf16.f32`) the op that calls it (`"helper": "_tf_mul_rn_f32"`). A fused region's entries
  carry `origin`, the op's position in the region; `graphs/<graph>.json` gives, per region, the op of the traced graph at each position (its index, as the
  fusion report numbers it, its op and its line), so a region that two graphs share is explained in each graph's terms.
* **Where the SASS comes from**: the index entry's `"sass"` says which `nvdisasm` (version) read it and whether it is the binary's own SASS or the SASS of its
  PTX assembled for the base of its family (`"proxy": true`, section 5).

**Outside the pinned artefact.** No manifest, plan or plan set names the trace and no digest covers it, so an export with or without it writes the same
manifest and plans, byte for byte (checked on the whole GPU gate: section 6). A host never reads it. The build cache keeps binaries, not their IR, so a
traced export compiles every Triton kernel (it stores the cache entry, which is the same bytes: builds are reproducible). An export into a directory that
has a trace merges the index: an entry of another binary stays while the manifest still has that binary (the same sha256).

The index has its own digest, `trace_digest` (sha256 over its canonical JSON, tag `tilefish.trace-digest.v1`), and records each file's sha256.
`tilefish-explain` refuses a trace whose index or files are not as recorded, and ignores the entry of a binary the manifest no longer has.

## 2. `tilefish-explain`

```console
tilefish-explain DIR KERNEL                     # a stem of the manifest, or an op name with one binary
tilefish-explain DIR --graph G --step N         # step N of G's plan (--bucket D=96 when the plan has several buckets)
tilefish-explain DIR --graph G --region R       # a fused region: its kernel's name (fused_...) or its index in G.fusion.json
  --arch sm_103                                 # when the binary is built for several architectures
  --view annotate | summary | lines             # (annotate by default)
  --ptx                                         # annotate: the PTX line of each instruction
  --diff OTHER_DIR | --diff-arch sm_120         # the same kernel in another export, or for another architecture
  --source-root PATH                            # where module-relative names are found after the trace's copies (repeatable; default .)
  --json                                        # summary and lines as JSON
```

`python -m tilefish.explain` is the same; `tilefish.explain.load(directory, kernel, graph=, step=, bucket=, region=, arch=, source_roots=)` returns an
`Explained` with `annotate()`, `summary()`, `lines()`, `summary_text()`, `lines_text()`, and `tilefish.explain.diff(a, b)`.

**annotate** prints the SASS of the kernel's function, each run of instructions under the source line its line information names, with that line's text:

```
fused_370fc65537cf on sm_103 (fused): its SASS, read by nvdisasm 13.5.37
  step 0 of the plan of 'model' at {'R': 64}: launches 'fused_370fc65537cf'; fused region 0 of the ops #0 cast (model.py:16), #1 cast (model.py:16),
  #2 mul (model.py:16), #3 add (model.py:17), #4 cast (model.py:17)
...
tilefish_generated/fused_370fc65537cf.py:64  v2 = v0.to(tl.float32)    [node 0 cast; graph op #0 cast at model.py:16: h = x.to(f32) * w.to(f32)]
    /*0260*/ IMAD.U32 R11, R11, 0x10000, RZ ;  [integer]  ptx 84
    /*0270*/ SHF.L.U32 R12, R12, 0x10, RZ ;  [integer]  ptx 79
...
tilefish_generated/fused_370fc65537cf.py:18  [a, b],    [node 2 mul (helper _tf_mul_rn_f32); graph op #2 mul at model.py:16: h = x.to(f32) * w.to(f32)]
    /*0300*/ FMUL R12, R12, R17 ;  [float]  ptx 101
    /*0310*/ FMUL R4, R15, R4 ;  [float]  ptx 107
```

(The multiply is inline PTX in a helper function, `mul.rn.f32`, so its line is the helper's, which the trace attributes to the op that calls it.)

For a Triton or Gluon kernel the line is the kernel's own (`mm_kernels.py:29  acc = tl.dot(a, b, acc)` over its `UTCHMMA` instructions); for a lowered
kernel it is the generated source's, then the op of the kernel's graph it was written for and the line of the body; for a fused region the op of the graph
body. A plan step names the step, its bucket and its entry; a region lists the ops it took.

**summary** counts the instructions per class (section 3), lists the SASS MMAs (opcode and shape; for a `UTC*MMA` the kind, types and shape its
instruction descriptor says where the listing sets it as a constant), the PTX MMAs (whose opcode spells the kind), the TF32 MMAs, local-memory loads and
stores (and how many nvdisasm 13.x marks as spills), TMA, barriers and fences, and from the ELF file the registers per thread, the stack frame, the static
and dynamic shared memory, the named barriers, with an estimate of the occupancy (blocks and warps per SM and what limits them: from the attributes, not the
driver's figure).

**lines** counts the instructions of each source line by column: `mma`, `gmem` (global loads and stores), `smem`, `tma`, `tmem` (tensor memory), `local`,
`atom`, `sync` (barriers and fences), `alu` (the rest).

**diff** compares two binaries of one kernel: the summary (instructions, registers, classes, MMAs, PTX MMAs, spills, occupancy) and the source lines
whose instructions differ, largest change first. Two exports (`--diff`: another toolchain, another declaration) or two architectures of one export
(`--diff-arch`).

Without a trace `tilefish-explain` disassembles the manifest's cubin (an `nvdisasm` that decodes it is needed: there is no PTX to assemble for a family
base) and says so; the lines of a generated source are not mapped to ops then. With a trace it runs anywhere and needs no CUDA tool.

**What the line information can say.** The line of an instruction is what `ptxas` records, and `ptxas` schedules: an asynchronous `wgmma` of sm_90 may sit
under the line of the store that waits for it, address arithmetic under the line that first needs it. A function of Triton's own (`tl.max`,
`tl.sum`) is under its line of `triton/language/standard.py`: Triton 3.5.1's PTX has no `inlined_at`, so the call site is not known. A lowered kernel's helper
functions are attributed to the op that calls them (above).

## 3. Instruction classes

Each instruction is in one class, by its opcode (`tilefish._sass.CLASS_RULES`; a class name is a word `tilefish.sass(forbid=...)` takes):

| class | opcodes |
|---|---|
| `mma` | `HMMA`, `IMMA`, `QMMA`, `OMMA`, `DMMA` (`mma.sync`), `HGMMA`, `IGMMA`, `QGMMA` (`wgmma`), `UTCHMMA`, `UTCQMMA`, `UTCIMMA`, `UTCOMMA` (`tcgen05.mma`) |
| `tma` | `UTMALDG`, `UTMASTG`, `UTMAPF`, `UTMAREDG`, `UTMACCTL`, `UBLKCP`, `UBLKRED`, `UBLKPF` |
| `tensor_memory` | `LDTM`, `STTM`, `UTCCP`, `UTCATOMSWS`, `UTCSHIFT` (`tcgen05.ld/st/cp/alloc/shift`) |
| `local` | `LDL`, `STL`: spills and local arrays |
| `global_load`, `global_store` | `LDG`, `LDGSTS`, `LD`; `STG`, `ST` |
| `shared_load`, `shared_store` | `LDS`, `LDSM`; `STS`, `STSM` |
| `atomic`, `shared_atomic` | `ATOM`, `ATOMG`, `RED`, `REDG`; `ATOMS` (a tcgen05 kernel's tensor-memory allocation uses shared atomics) |
| `barrier`, `fence` | `BAR`, `SYNCS` (mbarrier), `WARPGROUP`, `WARPSYNC`, `UTCBAR` (`tcgen05.commit`), `DEPBAR`, cluster barriers; `MEMBAR`, `FENCE`, `ERRBAR`, `CCTL` |
| `constant`, `fp64`, `mufu`, `convert`, `shuffle`, `control`, `special`, `float`, `uniform`, `integer` | the rest |

The MMA of each architecture, as the recorded kernels show it (Triton 3.5.1, ptxas 12.9):

| arch | PTX | SASS |
|---|---|---|
| sm_90 | `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16` | `HGMMA.64x64x16.F32.BF16`, `HGMMA.64x64x8.F32.TF32` |
| sm_100, sm_103 | `tcgen05.mma.cta_group::1.kind::f16`, `...kind::tf32` | `UTCHMMA` for both: the kind is in the instruction descriptor (`idesc[URn]`) |
| sm_120 | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | `HMMA.16816.F32.BF16`, `HMMA.1688.F32.TF32` |

## 4. SASS assertions

```python
@tilefish.extern(triton=mm_kernel, num_warps=8, grid=..., sass=tilefish.sass(
    no_tf32=True,                                  # no TF32 tensor-core MMA
    mma=["tcgen05.mma kind::f16", "UTCHMMA"],      # each used by at least one instruction
    no_spills=True,                                # no LDL/STL
    max_registers=168,                             # registers per thread, from the ELF file
    forbid=["atomic", "fp64", "ATOMG"],            # classes (section 3) or SASS opcode patterns
    by_arch={"sm_120": tilefish.sass(mma="HMMA.16816.F32", no_spills=False)},
))
def mm(...): ...
```

`sass=` is taken by `@tilefish.kernel`, `@tilefish.extern` (Triton, Gluon and prebuilt binaries; with `by_arch=` it holds for every implementation) and
`tilefish.fusion(sass=...)` (every region of the graph). The generated ops tilefish writes itself (`contiguous`, `im2col`, `col2im1d`) take none: their code
is tilefish's and is covered by its tests.

* **Patterns.** Lower case is a PTX pattern: its tokens (split on dots and spaces) appear in a PTX MMA opcode in order, the first two being the opcode's own
  (`"tcgen05.mma kind::f16"`, `"wgmma.mma_async bf16"`, `"mma.sync m16n8k16 bf16"`). Upper case is a SASS opcode pattern: the same opcode and the pattern's
  modifiers among the instruction's in order (`"HMMA.16816.F32"`, `"HGMMA.F32.BF16"`, `"UTCHMMA"`, `"LDL.64"`). On sm_100 and later say a tcgen05 kind in PTX.
* **`no_tf32`** reads the PTX (`kind::tf32`, `.tf32`) and, when the SASS is read anyway or the binary has no PTX (a prebuilt cubin), the SASS (`.TF32`, or a
  `UTCHMMA` whose descriptor says `kind::tf32`).
* **`no_spills`** refuses any local-memory instruction, a spill or a local array, and names each (nvdisasm 13.x marks spills `SpillRefill`).
* **`max_registers`** reads the ELF file of the binary itself (`EIATTR_REGCOUNT`), never a re-assembly.
* **`by_arch`**: on an architecture with an entry, every field the entry says replaces the op's (`no_spills=False` there switches it off).
* **Digest.** The assertions are in the declaration digest only when given (`sass=None` and `tilefish.sass()` leave every digest as it was); a region's name
  covers its fusion's assertions. They are not in a compile spec's document or a manifest entry (no schema changes): a spec read back gets them from
  `verify_against`.
* **When.** At every build: a compile, a cache hit (the cache key does not cover them: the same IR under other assertions is judged again), a `jit` run's
  build, every export, and a prebuilt binary's assembly. What needs only the PTX and the ELF file (`max_registers`, a PTX MMA pattern, `no_tf32` of a
  binary with PTX) disassembles nothing; the rest reads the SASS (section 5), once per binary per process.

A violation refuses the build with every violated assertion, the instructions and their source lines, and where the SASS was read:

```
'mm_tf32_checked' (sass_kernels:mm_tf32_kernel) for sm_103 violates the SASS assertions of its declaration {"max_registers": 64, "no_tf32": true}:
  - max_registers: the binary uses 121 registers per thread, more than the 64 declared
  - no_tf32: TF32 tensor-core MMAs, which keep 10 of the 23 bits of each f32 mantissa (8 in its PTX): PTX line 446:
    tcgen05.mma.cta_group::1.kind::tf32 (sass_kernels.py:61); PTX line 457: tcgen05.mma.cta_group::1.kind::tf32 (sass_kernels.py:61); ... (2 more)
Refusing to build it: change the kernel (or its launch options) until its binary is what the declaration says, or change the declaration. ...

'spill_checked' (sass_kernels:spill_kernel) for sm_103 violates the SASS assertions of its declaration {"no_spills": true} (read in its SASS, read
by nvdisasm 13.5.37):
  - no_spills: 4618 local-memory instructions (spills or a local array; a stack frame of 4256 bytes per thread): /*0180*/ STL.64 [R1+0xa0], R4
    (sass_kernels.py:79) [spill]; /*01b0*/ STL.64 [R1+0xa8], R6 (sass_kernels.py:79) [spill]; ... (4612 more)
```

The first needed no SASS (the PTX and the ELF file say it all), so it names no disassembler.

## 5. The disassembler

The `nvdisasm` of the contraction check (`tilefish._fpcontract.nvdisasm_candidates`): `$TILEFISH_NVDISASM_PATH`, Triton's (`$TRITON_NVDISASM_PATH` or its
bundled copy), the one next to its `ptxas`, an `nvidia-cuda-nvdisasm` wheel, then the `PATH` and `$CUDA_HOME/bin`; the first that decodes the architecture.
The listing is `nvdisasm -c -g -gi` (the code, with line information and the lines an instruction was inlined at where `ptxas` has them) and the PTX lines
`nvdisasm -c -gp`. The parser reads CUDA 12.8's and 13.x's listings: 12.8 prints scheduling annotations inside an sm_120 instruction (`&req={0}`,
`?trans3`), 13.x notes after one (`(*"SpillRefill"*)`), an sm_90 listing names its target in its header flags.

Triton 3.5.1 bundles CUDA 12.8's `nvdisasm`, which cannot decode sm_103. Where no `nvdisasm` decodes the architecture, the PTX is assembled again for the base
of its family (`sm_100a` for `sm_103a`, `sm_120a` for `sm_121a`; the same instruction set) with the flags it was built with (`-lineinfo`, `--fmad=false`
for `enable_fp_fusion=False`, `ptx_options`), and that SASS is read. Every message, view and trace entry says so ("the SASS of its PTX assembled again for
sm_100a, the base of its family, read by nvdisasm 12.8.55, which cannot decode sm_103a"). The registers always come from the binary itself. `pip install
nvidia-cuda-nvdisasm` (CUDA 13) or `TILEFISH_NVDISASM_PATH` reads sm_103 directly.

## 6. How it is tested

* `python/tests/sass_fixtures/`: listings of four kernels (`tl.dot` in bf16 and in TF32, a kernel that spills, one with an atomic and f64 arithmetic) for
  sm_90, sm_100, sm_103 and sm_120, by CUDA 13.5's `nvdisasm` and, for sm_100 and sm_120, CUDA 12.8's; trimmed to windows around the instructions the tests
  look at (the line information and descriptor constants they are under are kept). `python/tests/record_sass_fixtures.py` records them again (Triton, no
  GPU) and keeps its kernels on the lines the listings name. `test_sass.py` (parsing, classes, descriptors, the ELF attributes, the occupancy, the
  declaration, every assertion, the refusal message, a cache hit judged again) and `test_explain.py` (traces and the views, with a fake compiler) run on
  any machine.
* `test_gpu_sass.py` on a device: a traced export of a Triton extern, a Gluon extern (`tcgen05_mma` on sm_100 and sm_103), a DSL kernel and a fused region,
  each with the manifest and plan of an export without a trace; the MMAs of `tl.dot` and of `tcgen05_mma` on their lines; each assertion met and violated;
  a cache hit refused; the registers and stack frame of the ELF file against the driver's (`cuFuncGetAttribute`); the family base on sm_103. It ran on
  sm_103 with CUDA 13.5's `nvdisasm` and with Triton's own; its sm_90 and sm_120 parts (the per-architecture MMA patterns of `sass_kernels.MM_CHECKS`, the
  Gluon kernel without tcgen05) wait for a device of those architectures.
* The whole GPU gate exported with `TILEFISH_TRACE=1` writes the manifests, plans and plan sets of an export of the release without traces, byte for byte.
