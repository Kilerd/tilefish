# SASS: explained and asserted

Every kernel goes DSL / Triton / Gluon → TTIR → TTGIR → LLVM IR → PTX → ptxas → SASS. tilefish keeps every level of an export on request, maps each SASS
instruction back to the line and the op that produced it, and checks declared facts about the SASS at every build (`docs/explain.md`).

```python
# every level of every binary next to the manifest: sources, frontend IR, TTIR, TTGIR, LLVM IR, PTX, SASS and its line table
runtime.export(block, "build/block", trace=True)          # or TILEFISH_TRACE=1 for an export run by a script you do not edit

@tilefish.extern(triton=mm_kernel, num_warps=8, grid=..., sass=tilefish.sass(
    no_tf32=True, no_spills=True, max_registers=168,
    mma="tcgen05.mma kind::f16",                          # a PTX pattern; upper case is a SASS one ("HMMA.16816.F32")
    by_arch={"sm_120": tilefish.sass(mma="HMMA.16816.F32")},
))
def mm(...): ...
```

```console
$ tilefish-explain build/mm mm --view lines
mm on sm_103 (extern): the SASS of its PTX assembled again for sm_100a, the base of its family, read by nvdisasm 12.8.55, which cannot decode sm_103a (...)
  registers 122, frame_size 0, static_shared 1024, num_barriers 1; dynamic shared 131088

 count   mma  gmem  smem   tma  tmem local  atom  sync   alu  line
    79     .    20    15     .     .     .     .    13    31  mm_kernels.py:27  a = tl.load(a_ptr + offs_m[:, None] * k + (kk + offs_k)[None, :])
   112     .    20     .     .     .     .     .     5    87  mm_kernels.py:28  b = tl.load(b_ptr + (kk + offs_k)[:, None] * n + offs_n[None, :])
   156     8     .     .     .     2     .     .     6   140  mm_kernels.py:29  acc = tl.dot(a, b, acc)
   297     .    16    33     .     1     .     1     8   238  mm_kernels.py:30  tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], acc)
...
$ tilefish-explain build/swiglu --graph swiglu --step 0 --bucket R=8 --view lines
fused_ad9189093e91 on sm_103 (fused): the SASS of its PTX assembled again for sm_100a, the base of its family, read by nvdisasm 12.8.55, ...
  step 0 of the plan of 'swiglu' at {'R': 8}: launches 'fused_ad9189093e91'; fused region 0 of the ops #1 cast (fusion_graphs.py:51), #2 silu (...
 count   mma  gmem  smem   tma  tmem local  atom  sync   alu  line
     8     .     .     .     .     .     .     .     .     8  tilefish_generated/fused_ad9189093e91.py:57 [#1 cast @ fusion_graphs.py:51]  v2 = v0.to(tl.float32)
   121     .     .     .     .     .     .     .     .   121  triton/language/standard.py:48  return 1 / (1 + math.exp(-x))
    16     .     .     .     .     .     .     .     .    16  tilefish_generated/fused_ad9189093e91.py:20 [helper _tf_mul_rn_f32 (for nodes 1 silu, 3 mul)]  [a, b],
...
$ tilefish-explain build/mm mm --view summary          # MMA kinds and shapes, TMA, spills, barriers, registers, occupancy
$ tilefish-explain build/mm mm --diff-arch sm_120       # or --diff OTHER_EXPORT: what changed, per source line
```

* **The trace** (`trace/` next to the manifest, its own index and digest) is outside the pinned artefact: the manifest, the plans and every digest are the
  same with or without it. It covers lowered kernels, fused regions, generated ops, Triton and Gluon externs, and the PTX and SASS of prebuilt binaries.
* **`tilefish-explain`** (`python -m tilefish.explain`, `tilefish.explain.load`) annotates the SASS of a kernel, a plan step or a fused region with the
  source line of each instruction: the line of a Triton or Gluon kernel, or the line of the generated source, the op of the kernel's graph it was written
  for and the line of the `@tilefish.kernel` body, and for a fused region the op of the graph body as the fusion report numbers it. It reads the trace
  anywhere, with no CUDA tool, against the module-relative source names (`--source-root`); without a trace it disassembles the manifest's cubin.
* **SASS assertions** (`tilefish.sass(...)` on `@tilefish.kernel`, `@tilefish.extern` and `tilefish.fusion`) are in the declaration digest when given, and
  every build, a cache hit included, refuses a binary that violates one, naming the instructions and their source lines. They read the PTX (MMA kinds:
  the SASS of a `tcgen05.mma` does not spell its kind), the ELF file (registers, stack frame) and the SASS, through the `nvdisasm` of the contraction check;
  where none decodes the architecture (Triton 3.5.1's CUDA 12.8 tools and sm_103) the SASS is that of the PTX assembled for the base of the family (`sm_100a`)
  with the binary's ptxas and flags;
  that re-assembly is judged only where its registers, stack frames and code sizes are the binary's (else the build is refused with the fix), and every
  message says so. `pip install 'tilefish[sass]'` (CUDA 13's `nvidia-cuda-nvdisasm`, preferred when present) or `TILEFISH_NVDISASM_PATH` reads sm_103
  itself. `no_spills` and `max_registers` are the ELF file's, with no disassembly.
* The trace holds source code: do not ship it with an export.

**Which steps could be one kernel** (`docs/advise.md`): `tilefish-explain build/block --graph block --advise` ranks the fusion candidates of a plan
(steps joined by intermediates one kernel would not store) by the bytes and time they would save, says why each is not one kernel today and what to do
(write an extern's arithmetic in the graph body, add `order=` so that fusion version 2 takes a reduction into a row region, remove a barrier; a GEMM
epilogue is future work, RFC 0003), and sums the plan's time by class. `tilefish.profile_plan(block, "build/block", bucket="M=128", ...)` (or `--profile`) measures every step on the device, a runtime path
outside every artefact, and `--timing` ranks by what it measured.

**What goes on inside a kernel** (`docs/instrument.md`): `tilefish.probe("load.wait")` marks a phase boundary in a `@tilefish.kernel` body, a
`@triton.jit` or a `@gluon.jit` kernel, and a production build compiles it to nothing. The instrumented variant (`runtime.export(..., instrument=True)`,
or `tilefish.run(..., probes=tilefish.Probes())` in jit) records `%globaltimer`, `%clock`, the SM and the probe for every warp into a buffer the host
gives in the profile scratch slot, with binaries and digests of its own that every host refuses unless asked; Python and Rust collect the records after a
launch, a plan run or a captured replay, and `tilefish-explain DIR --probes FILE` gives the phases with their distributions per warp role (the
partitions of a warp-specialised kernel), timelines per SM and CTA, and a Chrome trace.
