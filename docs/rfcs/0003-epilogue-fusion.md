# RFC 0003: Epilogue and prologue fusion

* **Status:** reviewed; the decisions of 2026-11-16 are recorded in "Decisions" below. Design and phase 0 experiments (section 6). Milestones E1,
  epilogue hooks on externs ("E1: implementation notes" below), and E2, cuBLASLt epilogues (section 7.3: what was built, what was measured and what
  is left), are implemented, and work together in one graph ("E1 and E2 together" below); experiment X8, a persistent warp-specialised GEMM with
  the hooks, is measured ("X8: results" below; the kernel is an example under `python/examples/`, not a supported feature); the rest is design.
* **Scope:** fusion across a call. An *epilogue* is the elementwise or row work that reads a GEMM's or a convolution's result; a *prologue* makes one
  of its operands. The RFC touches the planner and the fusion pass (`tilefish._plan`, `tilefish._fusion`), the library ops (`cublaslt.*`,
  `cudnn.conv_fwd`), externs, a new family of generated GEMM and convolution ops and the advisor. Only library epilogues touch the manifest and the
  hosts.
* **Baseline:** tilefish 0.0.10 / crate 0.0.8, which include the fusion advisor (`docs/advise.md`).
* **Relation to RFC 0001.** RFC 0001 (plan fusion) fuses graph-body arithmetic between calls and promises that fusion never changes bytes. Its
  section 7 sketched three routes into a call, and its section 11.2 deferred library epilogues. This RFC replaces those sketches. It is a separate
  document, not a section of 0001, for two reasons. RFC 0001 is reviewed, and its decisions D1 to D7 rest on a contract that has exact modes only.
  This RFC has to introduce the first contracts whose bytes are *not* those of the unfused composition, and it asks for decisions of its own. Every
  decision of RFC 0001 stands. Where this RFC builds on a phase of 0001, it says which.

## Summary

The fusion advisor measured three models already ported to tilefish on a B300 MIG slice: two transformer stacks and a convolutional encoder and
decoder. Nearly every byte that could be fused sits next to a GEMM or a convolution:

* the candidates that need a GEMM or convolution epilogue add up to 3.6 to 13.3% of the measured plan time per plan, and 6 to 13% on every plan but
  one, which an attention kernel dominates;
* the prologues of the convolutional model add up to another 10.9%, overlapping the epilogues;
* the candidates that phases 1 and 2 of RFC 0001 can reach add up to less than 0.2%.

So the fusion that pays is fusion into the compute op.

That fusion collides with tilefish's promise. Today a pinned cuBLASLt or cuDNN op writes the bytes that a reference stack writes (the same library
version and the same algorithm), and the kernel after it reads those *rounded* bytes. A fused epilogue sees the f32 accumulator instead. If it uses
the accumulator, one rounding disappears and the bytes change: X1 measured that a bias added to the accumulator differs from a bias added to the
rounded output on about 30% of the elements. If it rounds first, the bytes still change unless the GEMM inside the fused kernel computes the same
bits as the library's GEMM. This RFC therefore defines three contracts and keeps them apart:

* **(i) Exact.** The fused plan writes the bytes of the unfused composition, with the same GEMM bits. Four kinds of fusion keep this contract:
  * a hook after the rounding of a kernel that is itself the reference;
  * any prologue into a kernel that stays the same;
  * a generated GEMM that is measured equal to the library GEMM it replaces;
  * a library fusion that rounds where the composition rounds.

  Two experiments show that the last two exist. X3 found that a plain Triton `tcgen05` GEMM writes the bits of cuBLASLt's pinned GEMM on sm_103: on
  every shape, every one of 99 tile configurations and both postures, with special values in the operands, whenever the library's pin does not split
  `k`. X5 found that a cuDNN 9.13 fused graph with a 16-bit intermediate equals the unfused convolution + bias (+ ReLU) bit for bit.
* **(ii) Accumulator.** The user declares a rewrite of the program that removes the rounding of a GEMM or convolution result wherever an epilogue
  widens that result to f32. The fused plan is then exact against the unfused composition *of the rewritten program*: fusion still changes no byte
  by itself, the declared mode does. The result is faster and at least as accurate, and it is digested, pinned and reproduced by every host. It does
  not have the two-kernel stack's bytes. cuBLASLt's bias, ReLU and `beta * C` epilogues were measured equal to this contract on every element.
* **(iii) Library.** A cuBLASLt epilogue or a cuDNN fused graph writes the library's bytes under a pin: the algorithm or engine, the version, the SM
  count and the epilogue. Where the epilogue's arithmetic can be written down, export measures it equal to (ii). Otherwise the bytes are defined by
  the binary: cuBLASLt's GELU equals no formula tried on every element, and the best candidate matches 99.8% of them. At load, a canary checks the
  bytes.

A user who needs the bytes of an existing stack keeps contract (i), or takes (iii) with the stack's own library epilogue if the stack fuses that way.
A new model can take (ii). Nothing changes bytes without a declaration. There are three routes:

* **route (b), library epilogues**;
* **route (c), hooks in tilefish's own kernels**: first in user externs, then in GEMM and convolution ops that tilefish generates per architecture
  (tcgen05, wgmma, mma.sync);
* **prologues** into the A operand.

The measurements also bound the speed:

* a cuBLASLt epilogue saves 9 to 48% of the time of a GEMM followed by its epilogue kernel when its pin keeps a fast algorithm. It loses time when
  the epilogue's pin is a slower algorithm, which happened for GELU on two of five shapes;
* a straightforward Triton GEMM takes 1.22 to 1.44 times cuBLASLt's time, and fusing a naive epilogue into it was slower than leaving the epilogue
  unfused. Generated GEMMs need kernel work (experiment X8) before they pay. X8's persistent warp-specialised Gluon GEMM is within 1.10 times the
  pinned GEMM on 25 of the suite's 32 shapes at 2048 and 8192 rows (not at 32 and 256 rows), writes its bytes on every shape measured, and a bias +
  GELU or a gated activation fused into it is 1.33 to 1.70 times faster than the library GEMM followed by the region ("X8: results").

Section 7 therefore proposes hooks on externs and cuBLASLt epilogues first, and generated GEMMs after X8; the review decided that order ("Decisions"). The advisor then reports each epilogue
candidate with the routes and contracts that can take it.

## Contents

Decisions (2026-11-16)
E1: implementation notes (2026-11-17)
X8: results (2026-11-17)
0. Background: what exists
1. Use cases
2. The numerics problem and the three contracts
3. Routes
4. Surface syntax
5. Interaction with plans, reports, the advisor, SASS assertions, traces, hosts and schemas
6. Experiments: phase 0 results and the experiments still needed
7. Staged plan, milestones and tests
8. Questions (resolved 2026-11-16)
9. Prior art

Appendix A: a worked example.

## Decisions (2026-11-16)

The draft was reviewed on 2026-11-16. Every question of section 8 is settled. The labels are the questions' (RFC 0001's D1 to D7 keep their
names, and "D6" in this RFC is still RFC 0001's export-time verification):

* **Q1 and Q2: the three contracts are adopted.** (i) exact, (ii) accumulator and (iii) library, as section 2.3 defines them, are the contracts
  for fusion into a call, and RFC 0001's principle extends to them: fusion never changes bytes by itself; a declared mode does (`R_acc` is defined
  on the IR, so a failed fusion still runs `U(R_acc(P))`). **The default is `epilogues="exact"`**, on without a declaration. `"accumulator"` and
  `"library"` apply only when declared explicitly.
* **Q3: where a mode is declared.** At the graph level (`tilefish.fusion(epilogues=...)`) or in a `tilefish.fused` scope, never per call.
* **Q12: the order of work.** E1 and E2 go first, in parallel:
  * **E1**: epilogue hooks on externs, `value="out"` and `value="acc"`, with the advisor's changes of section 5;
  * **E2**: cuBLASLt epilogues through the new impl `cublaslt.matmul`: `BIAS`, ReLU and `beta * C` first, verified per pin, with the canary on;
    GELU later, as binary-defined.

  E3 (generated GEMMs) and E6 (cuDNN fused graphs) come later; E4 and E5 after the RFC 0001 phases they need.
* **Q4 to Q11: the recommended defaults of section 8.** In short:
  * Q4: a generated GEMM that must equal the library pin it replaces is an explicit opt-in of E3 (`equal="cublaslt"`), verified at every bucket,
    with the library GEMM and route (a) kept where it fails;
  * Q5: E3 is accepted per architecture when the generated GEMM is within 1.10 times the pinned cuBLASLt GEMM on a fixed suite of the shapes the
    ported models use;
  * Q6: bias, ReLU and `beta * C` first, measured-exact against (ii) and verified per pin; GELU later, binary-defined, behind `epilogues="library"`;
    the canary on by default for epilogue entries; where an epilogue's algorithm has no f32 form, the plain pin's f32 output is the reference and
    the auxiliary output must equal `ρ(acc + b)`;
  * Q7: a new impl `cublaslt.matmul` (`epilogue`, `aux`, `beta`, the optional `canary`); `cublaslt.gemm` and `cublaslt.gemm_bias` stay as they are;
  * Q8: the names as proposed;
  * Q9: gated activations with two accumulators per program;
  * Q10: norms over whole-row tiles first, the split over tiles with a declared order in E4;
  * Q11: cuDNN fused graphs (E6) after E2 and X13, with cuDNN 9.13 or later required for fused entries on sm_103.

## E1: implementation notes (2026-11-17)

Implemented on tilefish 0.0.11, released in 0.0.12, as `tilefish.hook`, `@tilefish.extern(..., epilogue=...)`, `tilefish.fusion(epilogues=...)` and the scope `with
tilefish.fused(epilogues=...):`, the generated modules of `tilefish._epilogue`, the hook lowering (`tilefish.lower.lower_hook`, note 22), the merge, the
rewrite `R_acc` and the check in `tilefish._fusion`, the fusion report's new fields and the advisor's routes. Neither schema, the crate nor the C ABI
changes.

* **As designed (3.2.1, 4, 5):** the two hook flavours (`value="out"` after the kernel's rounding, contract (i); `value="acc"` on the accumulator,
  contract (ii)); `epilogues="exact"` on by default and `"accumulator"` declared per graph or per scope (Q2, Q3); `R_acc` on the IR before any region
  is formed, so a region the hook does not take still runs `U(R_acc(P))`; one fused op per pair, named by its content; a pair the planner or the
  check refuses stays route (a), with the reason in the report; the export-time check of every pair; dual accumulators for a gated activation (Q9,
  `split=2`); norms over whole-row tiles only (Q10, `block_n=`); the advisor's routes with their contracts and savings.
* **The ABI** (`tilefish._epilogue.ABI`). The kernel has two parameters per hook that the stub does not list, `<NAME>` (a `tl.constexpr`: a jit
  function) and `<NAME>_ARGS` (a tuple), and calls `tile = NAME(tile, rows, cols, row_mask, col_mask, NAME_ARGS)` on each output tile where it would
  store it: `tile` the rounded tile (`out`) or the f32 accumulator (`acc`), or a tuple of `S` tiles side by side (`split=S`); `rows` the tile's rows of
  the output seen as `[R, L]`, `cols` its columns within a chunk, in a Gluon kernel in the slice layouts of the tile's layout; it returns what the
  kernel stores (an `acc` hook's return converted to the output's type by the kernel, `tile.to(OUT.dtype.element_ty)`; a split hook's one or `S`
  tiles side by side, the output's rows that many chunks wide). Triton 3.5.1 takes a jit function as a `constexpr` and a tuple of tensors and scalars
  as a nested call's argument, measured.
* **Decisions made in implementing it:**
  * **A generated module, not a hook in the compile spec.** A spec, a manifest entry and both hosts describe values, not functions. So an extern
    that declares hooks is compiled through a generated module: `tf_<extern>__hooks(<the stub's parameters>)` calls the kernel by keyword with the
    identity hook; a pair, `tf_<extern>_epi_<hash>(<the stub's parameters>, ep_*)`, calls it with the region's hook and its arguments as a tuple. The
    kernel's main loop is the same source in both, inlined into one IR, which identifies the binary (TTIR, or TTGIR for Gluon); the module's sha256 is
    in the declaration digest (`source_sha256`). An extern without hooks is as it was.
  * **What the kernel stores.** The result itself when anything else reads it (every output of the region then a side output, which the hook stores);
    else the region's first output of the result's shape (and of its type, for `out`: the kernel rounds to the output's type before the hook), the
    fused op's output given that type (`acc`) and, for a split hook, the shape of one chunk; else the result all the same, into a buffer nothing reads
    (still one launch and one read fewer than route (a)).
  * **The accumulator form** of `U(R_acc(P))` is not "the kernel with an f32 output", which a kernel that writes its output's type need not
    support, but `<extern>__acc`: the extern with an `acc` hook that stores the f32 tile into one more output and returns it as it is. The same kernel
    and accumulator, for any kernel with an `acc` hook; its rounded result is the extern's.
  * **Types.** A region's types are the graph's, a fused op's the extern's: a graph dimension the call binds to a dimension of the extern is replaced
    by it (a static one by its value), so the fused op has the extern's specs (stems, buckets, `by_bucket` configurations, grid, divisibility); a
    bucketed dimension of the region alone is refused (`epilogue-dims`).
  * **Gluon.** The hook is a `@triton.jit` function, which a Gluon kernel calls (measured). Every value of an elementwise hook is a full tile, so its
    layout is the tile's; a `-0.0` scalar is `tl.cast(bits)` there (Gluon's `tl.full` needs a layout). A row region is refused in a Gluon kernel's hook
    in E1 (a reduction's result has another layout).
  * **Whole rows.** An ordered sum of every row of a tile (`_tf_hsum_*`, note 22 of `tilefish.lower`): the chains by halving splits, the tree over the
    lanes by gathers, only the lanes that have columns computed (a lane with none is `+0.0` all along, what revision 2's padding gives, so a level of
    the tree whose partner is such a lane adds `+0.0`): the row template's terms in its order; `max` order-free; no `argmax`, no sum without
    `order=` (D2).
  * **Per-row and per-column operands** (a bias, a scale, a norm's weight) are loaded once per tile as a vector and broadcast (the bits the
    elementwise template reads at each lane's offset), not gathered per element.
  * **Fusion versions.** The merge applies only to calls of externs with hooks, which no plan of 0.0.11 or before has, in versions 1 and 2 alike:
    every graph without such a call plans as before (its plan, manifest and report the same bytes). No new version.
  * **`tilefish.barrier(y)`** stops the epilogue fusion of `y` (section 4); **one region per call** (the first that fits; the others `epilogue-taken`),
    and one call per region (a region that reads two results fuses into the first call's hook).
  * **What a merge must not do.** Make a cycle in the graph of units, regions and fused pairs included (`epilogue-cycle`: a path may leave by one
    node of a region and come back by another); read, in the hook, a tensor the extern updates in place, either version (`epilogue-inout`). A split
    whose chunks the hook holds and something else reads too stays a view of the stored result after the fused call; a dense view at an offset that
    is no multiple of 16 bytes is copied for the hook (its parameter is compiled for an aligned pointer); the hook's row template asserts that the
    tile the kernel gives it holds a whole row.
  * **`reference` mode** of a fused pair is the extern's `reference=` followed by the region interpreted; under `R_acc` the f32 result is the hook's
    `reference=` (`tilefish.hook(..., reference=...)`), else the extern's reference widened.
  * **A library call under `epilogues="accumulator"`** is not rewritten in E1: its f32 form is E2's (the pin re-encoded with an f32 `D`, 3.1.1); it
    keeps route (a), and the advisor points to E2 and E3.
* **The check** (D6 for pairs): every fused pair against the extern (or its accumulator form) followed by the region's ops one kernel each, every
  intermediate stored, at every bucket, on two seeds: the extern's operands normal with the special values (both zeros and infinities, NaN, the largest
  finite values, a subnormal) at their first elements, every side input with the special values and random bit patterns of `_sample`. A difference
  refuses the export; a kernel that applies its `out` hook to the unrounded value is refused (tested).
* **Examples** (`python/examples/epilogue_gemm.py`): a Triton GEMM with both hooks (TMA descriptors, `tl.dot`, a grouped order, its epilogue on 1, 2
  or 4 sub-tiles of the columns, a hook call each; `tcgen05.mma` on sm_100/103 and `mma.sync` on sm_120, each asserted by `tilefish.sass`), its
  dual-accumulator form, a Gluon `tcgen05` GEMM, and a 1-D convolution as an implicit GEMM (rows `B * Lo`).
* **Measured** (B300 MIG slice, sm_103, 30 SMs; microseconds per plan run captured in a CUDA graph, `profile_plan`'s `plan_us`, trimmed mean of
  200; the hook against route (a), the extern then the region in one kernel, which `tilefish.barrier` of the result gives, and the unfused
  composition):

  | pattern (bf16; `H = 1024`, `F = 4096`) | rows | hook (1 step) | route (a) (2 steps) | unfused (steps) | hook, `"accumulator"` | (a) / hook |
  |---|---|---|---|---|---|---|
  | GEMM + bias + GELU | 256 | 25.0 | 25.1 | 37.1 (6) | 26.7 | 1.00 |
  | | 2048 | 111.6 | 135.3 | 231.1 (6) | 117.2 | 1.21 |
  | GEMM + bias + periodic activation | 256 | 31.2 | 27.2 | 61.8 (10) | 32.6 | 0.87 |
  | | 2048 | 139.2 | 143.9 | 402.9 (10) | 143.9 | 1.03 |
  | dual GEMM (`2F` rows) + SwiGLU | 256 | 41.2 | 44.0 | 61.8 (6) | 41.0 | 1.07 |
  | | 2048 | 229.8 | 261.1 | 410.2 (6) | 240.4 | 1.14 |
  | GEMM (`D = 256`) + residual add + RMSNorm, residual returned | 256 | 17.0 | 14.9 | 23.2 (11) | 16.8 | 0.88 |
  | | 2048 | 16.9 | 16.9 | 35.6 (11) | 16.9 | 1.00 |
  | conv1d (`k = 7`, 512 to 512 channels, `L = 4096`) + bias + periodic | 4096 | 164.6 | 127.5 | 193.4 (10) | 167.6 | 0.77 |

  The GEMMs are the example's (`gemm_sm100`: 8 warps, the epilogue on two sub-tiles of 64 columns; the norm's a tile of 128 x 256 that sees whole
  rows; the convolution 8 warps), which are not persistent and not warp-specialised: the hook's arithmetic runs after the main loop on the kernel's
  own warps, at its occupancy, overlapped with nothing. So the fusion pays where the epilogue is light and its operands are large (bias + GELU at
  2048 rows: 1.21x; the gated activation, whose fused op stores half the columns: 1.07 to 1.14x), is even where the GEMM is short, and loses where the
  epilogue is heavy for the kernel's few warps (a `sin` per element over a convolution: 0.77x). The configuration matters as much as the fusion: with
  4 warps, or the whole 128-column tile in one hook, the hook's registers spill (`sin`: 255 registers and a stack frame) and the same periodic
  activation was 0.30 to 0.45x; a first ordered row sum by reshapes and splits of an MMA-layout tile spilled 15 KB per thread (0.04x), which the
  gather-based sum above removed. Getting the rest is the kernel work of X8 (an epilogue that overlaps the next tile's main loop), not the hook. The
  route is a declaration's choice, never made on time (section 5): the advisor reports the saving each route would give.

* **Tests:** `tests/test_epilogue.py` (no GPU: the declaration and its refusals, the generated module, the merge of every pattern in both modes, the
  hook's code against the region's op by op, one op for one pair in two graphs, `R_acc` IR in and out, scopes, every refusal with its reason, the
  report, the boundaries, the advisor's routes); `tests/test_gpu_epilogue.py` (every pattern fused against `U(P)` and `U(R_acc(P))` at every bucket
  and through `verify_fusion`; every finite bf16 value as a tile through the activations' hooks; 40 random regions after a GEMM in both modes; the
  Gluon GEMM; export, replay and capture; the fused binary's MMA sequence equal to the extern's; a lying hook refused; the reference); the Rust and C++
  gates run the patterns as `fused_epi_*` plans, launched and captured. Arch-agnostic but the Gluon tests (sm_100, sm_103); sm_120 is a later gate's.
* **Not yet:** prologues; a library call under `R_acc` (not in E2 either: "E1 and E2 together"); norms over rows wider than a tile (E4); row regions in a Gluon kernel's hook; an extern
  declared with `by_arch=` taking hooks; the MMA-sequence assertion as a declared SASS assertion (a test checks it); X12 on a real Gluon
  implicit-GEMM convolution.

## E1 and E2 together (2026-11-17)

E1 and E2 were built in parallel from 0.0.11, each with a two-valued `epilogues=`: E1's `("exact", "accumulator")`, E2's `("exact", "library")`.
Merged, the mode is one of three, per graph (`tilefish.fusion(epilogues=...)`) or per `tilefish.fused(epilogues=...)` scope, whose mode wins over the
graph's for the ops traced in it (Q3):

| mode | an extern with hooks | a `cublaslt.gemm` (or `gemm_bias`) whose result the graph widens |
|---|---|---|
| `"exact"` (default) | its `value="out"` hook: contract (i) | left as it is; the report's `library_epilogues` lists what `"library"` would do |
| `"accumulator"` | `R_acc` and its `value="acc"` hook: contract (ii) | **left as it is**: `R_acc` of a library call is not planned, so the GEMM keeps its rounding; the report's entry says so (`"mode": "accumulator"`, a `note`) |
| `"library"` | its `value="out"` hook: contract (i), as `"exact"` | rewritten into a `cublaslt.matmul` where the pattern of 7.3 matches: contract (iii), measured-exact against (ii) |

* **What `"accumulator"` means for a library GEMM.** Contract (ii) asks that a declared mode change the program, not the fusion (section 2.3): under
  `R_acc` a failed fusion still runs `U(R_acc(P))`, the GEMM writing f32. Neither E1 nor E2 plans that for a library call (E1's `R_acc` acts on the
  externs with an `acc` hook; E2's rewrite is its own mode), so under `"accumulator"` a cuBLASLt GEMM is planned as under `"exact"`: its result rounded,
  `U(P)` for that GEMM, which is a no-op with a message rather than a refusal (a graph may call both an extern it wants on the accumulator and a
  cuBLASLt GEMM). The message is the fusion report's `library_epilogues` entry of the GEMM: `"mode": "accumulator"`, `"note"`: declare
  `"library"` (the graph's or a scope's) for its epilogue. Planning `U(R_acc(P))` for a library call (the f32 form of the pin, then the region) is
  left for a later milestone; the advisor's E2 route says where `"library"` would take the region.
* **`"library"` keeps the extern hooks of `"exact"`.** Section 4 sketched `"library"` as (ii) and (iii); E2 built it without `R_acc`, and the merge
  keeps it so: an extern's region goes into its `value="out"` hook under `"library"`, and a graph that wants an extern on its accumulator and a
  cuBLASLt epilogue declares one and scopes the other, e.g. `epilogues="library"` on the graph and `with tilefish.fused(epilogues="accumulator"):`
  around the extern's arithmetic.
* **Scopes and the library rewrite.** E2 read the graph's mode only. Merged, a cuBLASLt pattern is rewritten where every op of it (from `y.to(f32)` to
  the last `.to()`, and the widenings of `c` and the bias) is in a `"library"` mode (a widening of an extern's result in an `"accumulator"` scope
  is `R_acc`'s, so a pattern that reads it is left to it); a pattern whose ops are in other modes is listed in the report as matched and not
  applied, with the modes. The rewrite runs first, on the graph's own arithmetic, then `opmath`, `mul-add` and `R_acc`, as both lanes ordered them; the
  `cublaslt.matmul` call keeps the GEMM's scope and location.
* **Names and digests.** `epilogues` is in no region's canonical settings in either lane, so a region the mode leaves is the same operator in every mode;
  the fusion report's `fusion` says the mode when it is not `"exact"`. The advisor's blockers are both lanes': `epilogue-hook-consumer` (E1, phase
  4), `epilogue-library-cublaslt` (E2, phase `E2`), and `epilogue-library` with the reason a GEMM's region was not matched (E3 and E6 later).
* **Tests:** `tests/test_epilogue_modes.py` (no GPU: the three modes and scopes on one graph that calls an extern with both hooks and a cuBLASLt
  GEMM, the report's entries in each, the refusal of an unknown mode) and its GPU counterpart in `tests/test_gpu_epilogue.py` (the mixed graph
  exported, replayed and compared byte for byte with `U(R_acc(P))` for the extern and the formula for the GEMM).

## X8: results (2026-11-17)

X8 asked whether a GEMM of tilefish's own can be fast enough that fusing into it pays (Q5), and whether its bytes are the library's (the E3
question, contract (i-d)). The kernel is an example, `python/examples/persistent_gemm.py`, which shows the hooks in a kernel fast enough to
fuse into; it is not a supported feature (`docs/scope.md`: tilefish does not compete with the libraries on kernel speed). What follows is the
experiment's record.

**The kernel.** `c = a @ b`, bf16 or f16 operands, an f32 accumulator, the four layouts of `cublaslt.gemm` (`TN`, `NN`, `TT`, `NT`), the hooks
`ACC` and `EPILOGUE` of E1 and the dual accumulator of a gated activation (`split=2`).

* **sm_100 and sm_103** (`gemm_sm100_kernel`, Gluon). A persistent grid (`tilefish.persistent()`, one program per SM, a grouped tile order) whose
  partitions run on warps of their own:
  * a one-warp TMA producer filling a ring of up to 8 shared-memory stages, each with a `ready` and an `empty` mbarrier;
  * a one-warp `tcgen05_mma` issuer accumulating each tile into one of two tensor-memory buffers (`2 * split * BN <= 512` columns);
  * the epilogue on `PARTS` partitions of `NW` warps, by default two of eight, each draining one buffer while the issuer fills the other. It reads
    the accumulator in sub-tiles of 32 columns (`tcgen05.ld 32x32b`), calls the hooks once per sub-tile and stores through shared memory by TMA.
    The sub-tile loop is not unrolled, so a hook's code is in the binary once.

  The whole `k` of a tile is one program's, in order, 64 columns a step. Its phases carry `tilefish.probe`s.
* **sm_120** (`gemm_sm120_kernel`, Triton): TMA loads, a software-pipelined `tl.dot` (`mma.sync m16n8k16`), a persistent loop, the epilogue on
  sub-tiles. It is built for sm_120 and held to its SASS assertions on sm_103; its logic runs on sm_103 compiled to `tcgen05` (`impl="triton"`,
  the Gluon kernel's bytes). Its speed is a later sm_120 gate's.
* **Configurations per bucket** (`by_bucket=` on `M`). `config_sm100` takes the tile of least modelled time: a `k` step costs
  `BN / 256 * 271 ns` of tensor core for 128 rows (half that for 64) or its operands' bytes at 220 GB/s into an SM, whichever is longer, plus 0.3
  us a tile, over `ceil(tiles / SMs)` rounds. It takes as many stages as shared memory holds. Over the suite's 64 shapes its tile is 0.4% slower
  than the best of a sweep of 16 tiles on average, 8% at most.
* **Descriptor boxes.** The descriptors' blocks are fixed by the declaration. Each block is the smallest tile of any bucket, and a larger tile is
  copied as several boxes, so the tile still varies per bucket.
* **What Triton 3.5.1 does not allow.**
  * Gluon has no two-CTA MMA (`cta_group::2`) and no TMA multicast, so a tile is one CTA's. A `tl.dot` with `num_ctas=2` emitted `cta_group::2`
    but faulted when run.
  * Every Gluon shape is a power of two, so tiles of 96, 160 or 192 columns, which would quantise better over 30 SMs, are refused.
  * A tensor-memory load is lowered for 4 or 8 warps only.

**The setup.** A B300 MIG slice: sm_103, 30 SMs, torch 2.9.1, Triton 3.5.1, ptxas 12.9, cuBLASLt 12.8.4. A GEMM's time is the median of 30 replays
of a CUDA graph of 20 back-to-back calls. A plan's time is `profile_plan`'s `plan_us`, the trimmed mean of 200 replays.

**Speed against cuBLASLt (Q5).** Each cell of the table below gives three times in microseconds and two ratios:

* the times are this GEMM's, the pinned cuBLASLt GEMM's (`heuristic="torch"`, what `torch.mm` runs), and the fastest of the 8 algorithms the
  heuristic offers;
* the ratios are this GEMM over the pin, and this GEMM over the fastest algorithm (an imported algorithm, `algos=AlgoTable`).

The operands are bf16 in the `TN` layout.

| M | N \ K | 768 | 1024 | 3072 | 4096 |
|---|---|---|---|---|---|
| 32 | 768 | 4.7 / 2.9 / 2.8 (1.64, 1.65) | 4.9 / 3.1 / 3.2 (1.55, 1.55) | 8.0 / 5.5 / 5.5 (1.46, 1.46) | 9.6 / 6.6 / 6.6 (1.45, 1.45) |
| 32 | 1024 | 4.6 / 3.2 / 2.9 (1.44, 1.61) | 5.0 / 3.5 / 3.1 (1.41, 1.60) | 8.2 / 6.7 / 5.3 (1.21, 1.54) | 9.8 / 8.3 / 6.5 (1.19, 1.50) |
| 32 | 3072 | 5.3 / 3.3 / 3.3 (1.59, 1.60) | 6.2 / 3.7 / 3.7 (1.68, 1.68) | 10.0 / 6.8 / 6.8 (1.48, 1.46) | 12.1 / 8.4 / 8.4 (1.44, 1.44) |
| 32 | 4096 | 7.1 / 4.8 / 4.2 (1.49, 1.69) | 8.2 / 5.3 / 4.7 (1.54, 1.73) | 16.8 / 14.2 / 10.9 (1.18, 1.54) | 23.0 / 16.8 / 17.1 (1.37, 1.35) |
| 256 | 768 | 5.2 / 3.4 / 3.2 (1.55, 1.65) | 5.7 / 3.7 / 3.5 (1.56, 1.65) | 9.6 / 6.0 / 6.0 (1.60, 1.60) | 11.5 / 7.2 / 7.2 (1.61, 1.61) |
| 256 | 1024 | 5.8 / 4.0 / 3.6 (1.45, 1.62) | 6.3 / 4.5 / 3.9 (1.41, 1.64) | 10.9 / 8.2 / 6.5 (1.34, 1.67) | 13.3 / 10.0 / 8.2 (1.32, 1.62) |
| 256 | 3072 | 7.8 / 5.6 / 5.6 (1.38, 1.38) | 8.8 / 7.5 / 6.3 (1.18, 1.39) | 17.4 / 16.1 / 14.1 (1.08, 1.23) | 21.8 / 20.4 / 17.8 (1.07, 1.22) |
| 256 | 4096 | 10.5 / 6.6 / 6.6 (1.59, 1.59) | 12.2 / 7.5 / 7.5 (1.61, 1.61) | 26.1 / 17.2 / 17.2 (1.52, 1.52) | 42.5 / 27.9 / 28.0 (1.52, 1.52) |
| 2048 | 768 | 11.8 / 9.5 / 8.9 (1.25, 1.33) | 14.0 / 13.0 / 10.6 (1.08, 1.32) | 31.3 / 30.3 / 26.0 (1.03, 1.20) | 39.9 / 38.9 / 33.7 (1.03, 1.19) |
| 2048 | 1024 | 15.3 / 10.0 / 10.0 (1.53, 1.53) | 18.0 / 12.4 / 12.4 (1.45, 1.45) | 41.0 / 31.7 / 31.8 (1.29, 1.29) | 52.1 / 41.5 / 41.5 (1.26, 1.26) |
| 2048 | 3072 | 31.1 / 32.2 / 25.4 (0.97, 1.22) | 38.2 / 39.6 / 31.7 (0.96, 1.20) | 108.5 / 98.7 / 92.1 (1.10, 1.18) | 145.6 / 127.5 / 122.2 (1.14, 1.19) |
| 2048 | 4096 | 39.1 / 42.2 / 32.0 (0.93, 1.22) | 48.3 / 52.0 / 40.7 (0.93, 1.19) | 142.4 / 130.1 / 120.6 (1.09, 1.18) | 186.3 / 168.7 / 159.3 (1.10, 1.17) |
| 8192 | 768 | 31.2 / 32.2 / 25.3 (0.97, 1.23) | 38.3 / 39.5 / 31.9 (0.97, 1.20) | 103.5 / 98.3 / 93.2 (1.05, 1.11) | 135.1 / 127.6 / 122.3 (1.06, 1.10) |
| 8192 | 1024 | 39.4 / 42.8 / 32.2 (0.92, 1.22) | 51.6 / 53.2 / 43.9 (0.97, 1.18) | 131.4 / 130.5 / 120.6 (1.01, 1.09) | 171.7 / 170.5 / 158.4 (1.01, 1.08) |
| 8192 | 3072 | 108.7 / 121.0 / 94.3 (0.90, 1.15) | 135.9 / 148.8 / 119.9 (0.91, 1.13) | 388.4 / 371.9 / 370.6 (1.04, 1.05) | 518.3 / 483.7 / 481.5 (1.07, 1.08) |
| 8192 | 4096 | 142.2 / 159.5 / 124.0 (0.89, 1.15) | 177.7 / 196.6 / 157.5 (0.90, 1.13) | 519.3 / 493.9 / 493.8 (1.05, 1.05) | 680.8 / 645.2 / 645.4 (1.05, 1.05) |

How many shapes are within 1.10 of the pin, by rows (the geometric mean of the ratio in brackets):

| rows | within 1.10 of the pin | ratio to the pin |
|---|---|---|
| 8192 | 16 of 16 (0.98) | 0.89 to 1.07 |
| 2048 | 9 of 16 (1.12) | 0.93 to 1.53 |
| 256 | 2 of 16 (1.41) | 1.07 to 1.61 |
| 32 | 0 of 16 (1.44) | 1.18 to 1.68 |

That is 27 of 64 shapes in all. Against the fastest algorithm, only 6 of 64 are within 1.10; it is faster than the pin at `k` of 768 and 1024.
The other layouts and f16 behave the same on a subset of 12 shapes each (`NN`, `NT`, `TT` in bf16, `TN` in f16): from 0.87 to 1.70 times the pin,
the same pattern by rows.

What costs the time is measured, not guessed:

* **A fixed cost per launch.** At `k = 64` this GEMM takes 2.95 us and cuBLASLt 2.03 us, and the difference grows only slowly with `k`.
  * The probes put about 1.2 us before the producer's first load: barrier initialisation, the tensor-memory allocation, and the partitions'
    setup.
  * Placing an empty kernel between the GEMMs changes neither library's time, so cuBLASLt gains nothing from overlapping its launches.
* **Quantisation over 30 SMs.** For example, 2048 x 1024 makes 128 tiles of 128 x 128, which is 4.3 rounds, or 64 tiles of 128 x 256, which is
  2.1. cuBLASLt's pins use clusters (`cluster_shape_id` 3 and 5).
* **Bytes into an SM.** A one-CTA tile reads `(BM + BN) * BK` operands a step. A two-CTA tile, or a multicast to a cluster, reads fewer per CTA,
  and Triton 3.5.1's Gluon can express neither.
  * Where the tiles fill the SMs (2048 and 8192 rows, `N` of 3072 or more), the kernel runs at 0.9 to 1.1 times the pin. The issuer is busy
    about 88% of the time there: it waits for a slot in a third of the steps.
  * At 32 and 256 rows the launch cost and the rounds dominate.

**Q5 reading.** The acceptance line of E3 (within 1.10 times the pinned GEMM on a fixed suite) is met at 2048 and 8192 rows, on 25 of those 32
shapes. It is not met at 32 and 256 rows, and not against the fastest imported algorithm. A generated GEMM in place of a library call pays
where the shape is large and the epilogue is fused (next paragraph). Below that it needs what Triton 3.5.1 lacks: two-CTA MMAs or multicast, and
tiles that are not powers of two. A cheaper prologue would help too.

**Bytes (contract (i-d)).** The kernel's bytes are the cuBLASLt pin's on every one of the 64 shapes above. They are also the pin's on the 48 shapes
of the other layouts and f16, at every bucket of the tests (`tests/test_gpu_persistent_gemm.py`), and the fastest algorithm's bytes. No pin of
cuBLASLt 12.8.4 on this slice splits `k` for these shapes.

On six shapes with a long `k` (`32 x 768 x 4096`, `32 x 1024 x 4096`, `32 x 4096 x 4096`, `256 x 768 x 4096`, `256 x 1024 x 3072` in f16 and
`2048 x 1024 x 4096`):

* every algorithm the heuristic offers that does not split `k` (8 per shape: other ids, tiles and clusters) wrote this kernel's bytes, with no
  element different;
* the pin's configuration with `splitk_num` 2, 3 or 4 and f32 partial sums (reduction scheme `COMPUTE_TYPE`) differed on 0.18 to 0.24% of the
  elements in bf16 and 1.06 to 1.19% in f16. X3b found the same 0.2% for 13.1.0's split-K pins at 32 rows.

So on sm_103 the bytes of a `kind::f16` GEMM depend on the summation order over `k` and on nothing else measured.

**What equality costs.** Where the pin does not split `k`, nothing: the kernel's bytes are the pin's whatever its configuration (tested: tiles,
stages, epilogue partitions and warps), so a configuration is chosen for speed alone.

To equal a split-K pin, the kernel would have to split `k` into the pin's chunks and add the f32 partial sums in the pin's order (X9). That is a
workspace, a second pass and a kernel per split count. Every split-K algorithm above was slower than the fastest unsplit algorithm, so nothing
was given up for equality here. At 32 rows a split of this kernel would be faster than the kernel as it is (cuBLASLt's split of 2 takes 8.5 us at
32 x 768 x 4096, this kernel 9.6 us), but its bytes would then match no unsplit pin.

**Epilogue fusion with it (E1's table again).** The patterns and sizes are E1's (bf16, `H = 1024`, `F = 4096`), in microseconds per plan. The
columns are:

* *hook*: the pair, one step, exact;
* *hook, acc*: the pair under `epilogues="accumulator"`;
* *(a), this GEMM*: this GEMM, then the region (a barrier);
* *(a), cuBLASLt*: route (a) as the RFC defines it, the pinned cuBLASLt GEMM (torch posture) followed by the generated region;
* *unfused*: cuBLASLt followed by the region's ops one kernel each, with its number of steps.

| pattern | rows | hook | hook, acc | (a), this GEMM | (a), cuBLASLt | unfused (steps) | (a), cuBLASLt / hook |
|---|---|---|---|---|---|---|---|
| GEMM + bias + GELU (erf) | 256 | 21.7 | 20.8 | 22.8 | 18.7 | 29.9 (6) | 0.86 |
| | 2048 | 77.8 | 75.9 | 103.5 | 103.5 | 199.0 (6) | 1.33 |
| GEMM + bias + periodic activation | 256 | 26.6 | 26.6 | 24.6 | 20.5 | 51.2 (10) | 0.77 |
| | 2048 | 127.9 | 124.5 | 112.5 | 112.4 | 370.5 (10) | 0.88 |
| dual GEMM (`2F` rows) + SwiGLU | 256 | 22.4 | 26.5 | 28.8 | 24.9 | 43.0 (6) | 1.11 |
| | 2048 | 90.1 | 124.8 | 153.5 | 153.5 | 302.5 (6) | 1.70 |
| GEMM (`D = 256`) + residual add + RMSNorm | 256 | 10.8 (not fused) | 10.9 (not fused) | 11.3 | 9.1 | 18.0 (11) | (0.84) |
| | 2048 | 17.0 (not fused) | 17.1 (not fused) | 16.9 | 14.1 | 32.8 (11) | (0.83) |

* **Where it pays.** Fusing into this GEMM pays where its main loop is long enough to hide the epilogue and the GEMM is near the library's speed:
  * at 2048 rows bias + GELU is 1.33 times faster than route (a) and 2.6 times faster than the unfused plan;
  * the gated activation, whose fused op stores half the columns, is 1.70 times faster than route (a).

  Against E1's naive example GEMM the fused times fall from 111.6 to 77.8 us (GELU) and from 229.8 to 90.1 us (SwiGLU).
* **Where it does not.** At 256 rows the GEMM itself is 1.4 to 1.6 times slower than cuBLASLt's (the speed table), and the fusion saves less than
  that.
* **The periodic activation** (`sin` and a division per element) is issue-bound on the epilogue's warps. The probes show it:
  * the hook takes 2.5 us per 128 x 32 sub-tile (1.2 us for erf GELU, 0.04 us for the identity), so a tile's epilogue takes 11 us on two
    partitions against 5 us of main loop;
  * the issuer waits for an accumulator buffer for 6.6 us a tile on average (2.6 us with GELU, nothing without a hook).

  Two partitions of 8 warps are the most Triton 3.5.1's tensor-memory loads allow per partition. A periodic activation after this GEMM is 0.88
  times route (a); E1's naive GEMM gave 1.03, against its own slower route (a).
* **Configuration matters as much as the fusion.** On the bias + GELU pair at 2048 rows:
  * one partition of 4 warps is 1.9 times slower (155 us) than two partitions of 8;
  * an unrolled sub-tile loop doubled a periodic hook's code (4000 to 7900 instructions), with no other change.
* **The norm is not fused into the Gluon kernel.** E1 fuses no row region into a Gluon hook, so the first two columns of the norm rows are route
  (a) with this GEMM. The Triton form of the kernel (`impl="triton"`, whose sub-tile holds the whole row of 256) fuses it, but at 18.3 us, slower
  than route (a). Its main loop is Triton's own pipeline, not this kernel's.
* **Under `epilogues="accumulator"` the dual GEMM's pair stored the rounded result too** (fixed in 0.0.13). The view that splits it was left in
  the program after `R_acc` and kept it live (`y_needed` in `tilefish._fusion`), so the kernel wrote `[M, 2F]` besides the hook's output: 124.8
  against 90.1 us. `R_acc` now removes a split of the result that only the widenings it replaced read: the same bytes, 89.7 us against 90.2 us
  under `"exact"` (the hook, acc column above, is the measurement before the fix).
* **Not measured.** The convolution as a GEMM was not measured with this kernel: its operands are TMA tiles of 2-D tensors, and an implicit-GEMM
  producer (im2col boxes) is X12's and E3's work.
* **Contract (i) held in every case.** The fused pairs equal the unfused plans byte for byte at every bucket with the special values (tests). Every
  binary, fused or not, holds `tilefish.sass(mma="tcgen05.mma kind::f16", no_tf32=True)` at its build, and the fused binary's MMA sequence is the
  GEMM's (tested).

**The pipeline, probed** (an instrumented run of the 2048 x 4096 x 1024 plan; `tilefish.probe` around each wait). Without a hook:

* the epilogue partitions wait 6.7 us for each accumulator (median), so the GEMM is bound by its main loop;
* the issuer spends 159 ns issuing a step and waits 138 ns on average for the producer;
* the producer waits 137 ns on average for a free slot.

With bias + GELU the epilogue's wait falls to 2.8 us, the hook takes 1.2 us a sub-tile, and the issuer waits 2.6 us a tile for an accumulator:
the epilogue is now the bottleneck by about 20%. With the periodic activation the issuer's wait is 6.6 us. Production builds hold none of the
probes (`docs/instrument.md`).

## 0. Background: what exists

* **Library ops promise one rounding.** `cublaslt.gemm` computes `out = a @ b^T` with an f32 accumulator and rounds once to the output type.
  `cublaslt.gemm_bias` adds a bias of the output type in f32 before that rounding (`CUBLASLT_EPILOGUE_BIAS`, the only epilogue tilefish uses
  today). The algorithm is pinned (its 64 bytes, the cuBLASLt version and the SM count). The pin is vetted against what cuBLASLt itself encodes and
  checked by `cublasLtMatmulAlgoCheck` at load. It is chosen as tilefish or torch would choose it, or imported from another engine (`AlgoTable`). A
  GEMM with a bias gets no split-K unless it opts in, because a split-K with the bias epilogue was measured to round twice (`docs/numerics.md`,
  section 4). `cudnn.conv_fwd` has no bias. Its engine configuration is pinned per cuDNN version and SM count and checked at load by a canary
  (`docs/manifest-schema-3.md`, section 9.4).
* **Plan fusion stops at a call.** Phase 1 of RFC 0001 fuses the arithmetic of a graph body into generated regions. Each region is byte-identical to
  the unfused composition `U(P)`, and every export verifies it (D6). A call to a library op, an extern or a kernel is a region boundary. The
  arithmetic after a GEMM runs as route (a) of 0001 section 7: a generated kernel that reads the GEMM's stored, rounded result.
* **Ports already contain hand-fused compute kernels.** A port of a hand-fused engine reproduces the engine's kernels byte for byte. Some of those
  kernels are convolutions with a fused activation epilogue, written by hand as Gluon externs (`tcgen05` on sm_100 and sm_103, `mma.sync` on
  sm_120) that follow the original kernel's order of accumulation and its epilogue. To tilefish such an epilogue is just part of an extern. Route (c)
  below makes the same thing a feature of the planner.
* **The advisor sees the gap.** `tilefish-explain --advise` lists every candidate whose latest blocker is `epilogue-library` or `prologue-library` as
  future work (`docs/advise.md`, section 4). It lists every residual stream between two norms as `library-in-the-middle`. On the three ported models
  it ranks those candidates first by measured time. The largest single patterns are:
  * a GEMM followed by col2im, overlap-add and a periodic activation: 0.1 to 1.2 ms per instance, at short and long buckets;
  * a convolution followed by a bias and a periodic activation: 0.2 to 0.3 ms each;
  * a residual add and a norm after the output projection of every transformer layer;
  * gated activations;
  * a per-head norm with rotary embeddings after the QKV projection.

## 1. Use cases

This section lists what a model needs between the GEMMs and convolutions it pins. Two columns of the table need a definition:

* **Tile-local** says whether one output tile of the GEMM (`BM` rows by `BN` columns) has everything its epilogue needs. That decides how hard
  route (c) is.
* **Library** says whether a library epilogue exists for the pattern.

| use case | pattern (graph-body sketch) | tile-local? | library | notes |
|---|---|---|---|---|
| bias + activation | `act(y.to(f32) + b.to(f32)).to(bf16)`, where `act` is GELU (erf or tanh), SiLU or ReLU | yes (a per-column vector) | cuBLASLt `BIAS`, `RELU_BIAS`, `GELU_BIAS` (tanh GELU only); cuDNN fused graphs for convolutions | the most common epilogue |
| gated activation | `g, u = gemm(x, w).split(F, -1)`; `(silu(g.to(f32)) * u.to(f32)).to(bf16)` | no: column `j` needs column `F + j` | none | needs a tile of `g` and the matching tile of `u`: two accumulators per program, or the weight's rows interleaved |
| periodic activation | `s = sin(a * h)`; `h + s * s / a`, with `h = y + b` and a per-channel `a` | yes | a cuDNN pointwise chain for convolutions (X5); nothing in cuBLASLt | the convolutional model's largest pattern |
| residual add (+ norm), with the residual as a side output | `h = y + r` (stored: the residual stream); `n = rmsnorm(h, w)` | the add: yes; the norm: only when a tile holds whole rows (`BN >= N`) | cuBLASLt `beta * C` for the add; no library norm | the norm's row reduction crosses tiles for hidden sizes larger than a tile (3.2.3) |
| convolution + bias + activation | `act(conv(x, w) + b)` | yes (implicit GEMM) | cuDNN fused graphs (X5) | the convolution itself is an extern or `cudnn.conv_fwd` |
| transposed convolution as GEMM + col2im, then overlap-add + bias + activation | `cols = gemm(x, w)`; `y = col2im(cols)`; `act(y + b)` | no: an output sample sums `kernel / stride` rows of `cols` | none | needs a halo of `kernel / stride - 1` rows per tile, or a different algorithm (3.2.3) |
| per-head norm + rotary embedding | `q, k, v = qkv(x).split(...)`; per head `rope(rmsnorm(q_h))`, and the same for `k` | yes, when `BN` is a multiple of the head size and no tile straddles `q`, `k` and `v` | none | needs row-local movement (RFC 0001 phase 3) and segmented reductions (phase 4) inside a tile, and strided stores into `q`, `k`, `v` |
| norm into the next GEMM (a prologue) | `a = (x * rstd[:, None] * w).to(bf16)`; `gemm(a, W)` | an A-operand transform; `rstd` is a stored per-row value | none | recomputed for every column tile of the GEMM |
| dequantisation prologue | `w = (q.to(f32) * scale).to(bf16)`, with a per-channel or per-group `scale` | yes | cuBLASLt FP8 per-tensor and block scales only | exact only while the GEMM multiplies the dequantised operand |
| quantisation epilogue | `amax` per row or per 32-element block; `q = (y / scale).to(e4m3)` | per block of 32 columns: yes; per row: as the norm | cuBLASLt FP8 output with `D_SCALE` and `AMAX_D` (delayed scaling) | MX block scales are tile-local; they need RFC 0001 phase 4 segmented reductions |

**Non-goals.**

* No fusion of two compute ops into one kernel (a chain of GEMMs, attention).
* No algebraic rewrite in an exact mode. Two examples are moving a per-row scale from the A operand to the output, and folding a norm's weight into
  the next GEMM's weight. Each changes bytes, and the rule of RFC 0001 against algebraic simplification holds. A rewrite like this that is worth
  having becomes a mode of its own, in a later RFC.
* No autotuning that changes bytes. A choice between routes that give different bytes is never made by timing (section 5).

## 2. The numerics problem and the three contracts

### 2.1 What byte identity rests on today

Write the GEMM step as `y = ρ_T(acc_π(a, b) [+ bias])`:

* `π` is the pinned library configuration: the algorithm, the library version and the SM count;
* `acc_π` is the f32 value that its kernel holds before the final conversion;
* `ρ_T` is the rounding to the declared output type `T`.

The epilogue is a region `R` of the graph body (RFC 0001), and `z = R(y, s)` for the region's other inputs `s`. A port is byte-identical to its
reference stack for two reasons. The stack runs the same `π` (the same cuBLASLt version and an imported algorithm), so it writes the same `y`. And
the region reproduces the stack's epilogue kernel under RFC 0001's contract.

### 2.2 What a fused epilogue changes

Three things can change, and each route changes a different subset of them:

1. **The rounding between the GEMM and the epilogue.** A fused epilogue that reads the accumulator computes `ρ(f(acc))`, where the stack computes
   `ρ(f(ρ_T(acc)))`. Even `f(x) = x + b` rounds twice in the stack and once fused. X1 measured how often the results differ, with bf16 and f16
   outputs alike:
   * a bias add: 30% of elements;
   * a residual add: 29%;
   * a bias add and ReLU: 15%;
   * a bias add and tanh GELU: 31%.

   Only ReLU without a bias is unaffected, because it commutes with the rounding.
2. **The GEMM itself.** If the fused kernel is not the library's kernel, `acc` is another kernel's accumulator, with another order for the sum over
   `k`. The order can differ in the MMA instruction's own `k` step, the tile's `k` loop, split-K or stream-K reductions, or two-CTA MMAs. An epilogue
   that rounds first, `ρ(f(ρ_T(acc')))`, therefore equals the stack's result only where `acc' == acc_π` bit for bit. On sm_103, X3 found
   `acc' == acc_π` whenever the library's kernel does not split `k` (section 6). It is a measurement on one architecture, not a property that holds
   in general.
3. **The epilogue's own arithmetic.** A library epilogue computes its activation by its own formula, and it adds its operands in its own order:
   * cuBLASLt's GELU is a tanh approximation whose instruction sequence is not documented (X2);
   * cuBLASLt adds `beta * C` before the bias (X1).

| route | rounding removed (1) | GEMM changed (2) | epilogue arithmetic changed (3) |
|---|---|---|---|
| (a) a generated kernel after the call (today) | no | no | no (RFC 0001 regions are exact) |
| (b) a cuBLASLt epilogue | yes: every cuBLASLt epilogue acts before the conversion | no under 12.8.4: another algorithm, but X1 found the same accumulator; sometimes under 13.1.0 (X6) | only where the epilogue cannot be written down exactly (GELU) |
| (b) a cuDNN fused graph with a 16-bit intermediate | no (X5, cuDNN 9.13) | no (X5) | where its pointwise ops are not the region's (GELU, `sin`) |
| (c) a hook after the rounding of an extern that is the reference | no | no: same source, verified | no: the region's code |
| (c) a hook on the accumulator | yes | no | no |
| (c) a generated GEMM that replaces a library GEMM | either | not where the library does not split `k` (X3, sm_103); measured per bucket | no |
| a prologue into a kernel that stays the same | no: the operand bytes are the region's output | no | no |

### 2.3 The contracts

Notation:

* `P` is the declared program of the graph body (RFC 0001, section 5.1);
* `U(P)` is its unfused composition: every region alone, and every intermediate stored in its declared type;
* `F(P)` is the fused plan.

"Exact" means byte for byte on every output, `InOut` and carried buffer, for every input bit pattern, which is RFC 0001's claim (section 5.2).

**(i) Exact: `F(P) == U(P)`, with the reference GEMM.** The reference GEMM is whatever `U(P)` calls: the pinned library configuration, or an
extern. The fused plan must compute the GEMM's rounded result `y` bit for bit and then apply `R` to it. There are five ways to do that:

* **(i-a) The default route**: a region after the call. This is what happens today, and it saves no time.
* **(i-b) A hook after the rounding of a kernel that is the reference.** This applies when the GEMM or convolution is an extern (a user's Triton or
  Gluon kernel) or a tilefish-generated op, and `U(P)` calls that same kernel. The hook is applied to the value the kernel would store, after its
  rounding, so the kernel's arithmetic stays as it is:
  * the main loop's source is unchanged;
  * the hook's arithmetic is the region's context-free code (RFC 0001 obligations O2 and O3, and hazards H1, H8 and H9).

  This is exact by construction, and the export-time verification (D6) checks it.
* **(i-c) Any prologue into a kernel that stays the same.** A prologue computes, with the region's code, the operand tile that the kernel would have
  read from memory. So the kernel multiplies the same bytes. The prologue is repeated for every column tile, which is exact: it is recomputation of
  a context-free region.
* **(i-d) A replacement GEMM measured equal.** A generated GEMM `κ` replaces the library's `π` only at the buckets where `κ`'s output equals `π`'s.
  This cannot be proven, because the library has no IR and its kernels are not documented. It can be measured. A GEMM's order of operations does
  not depend on the data, so two kernels that sum in different orders differ on random inputs with near certainty. Equality on seeded random and
  special-value inputs at a bucket is therefore strong evidence that the order is the same. The equality holds per shape, library version, pin and
  toolchain, and it is *measured-exact*, the status that E4 of RFC 0001 gave the bias epilogue.

  X3 measured that it holds far more often than expected. On sm_103 a plain Triton `tcgen05` GEMM equals cuBLASLt's pin on every shape measured,
  with every tile configuration, with special values in the operands, and under cuBLASLt 12.8.4 and 13.1.0. It fails only where the library's pin
  splits `k`: cuBLASLt 13.1.0 does that for 32-row problems. sm_120 and sm_90 are not measured yet (X7).
* **(i-e) A library fusion that rounds where the composition rounds.** cuBLASLt has none: every cuBLASLt epilogue acts on the f32 value before the
  conversion (the header says so, and X1 measures it). ReLU without a bias is the one exception, because it commutes with the rounding. cuDNN does
  have such a fusion. In a cuDNN 9.13 fused graph whose convolution output is a virtual tensor typed f16, convolution + bias and convolution + bias +
  ReLU equal the standalone convolution followed by the epilogue bit for bit (X5).

**(ii) Accumulator: `F(P) == U(R_acc(P))`.** `R_acc` is a declared rewrite of the program. Like the rewrites of RFC 0001 section 5.4, it is defined
on the IR before grouping. It applies to a call whose op declares an f32 form: a library GEMM or convolution, an extern with an accumulator hook, or
a generated op.

* Every widening `y.to(f32)` of the call's result `y` is replaced by the f32 result of the same call: `acc`, before the rounding.
* A consumer that reads `y` in its declared type keeps reading `ρ_T(acc)`.
* When `y` is also returned, tapped or read by another call, it is still stored, rounded.

The rewritten program has one rounding fewer on every such path. Then:

* the fused plan is exact against the unfused composition of the rewritten program. In that composition the call writes its f32 result (the same
  kernel configuration with an f32 output) and the region reads it. Fusion therefore still changes no byte by itself, as RFC 0001 promises; choosing
  the mode does. Export verifies `F == U(R_acc(P))` as D6 does;
* the bytes are those of the kernel (`κ`, `π` or the extern) and of the region's declared numerics. They are deterministic, digested, pinned by the
  plan digest, and reproduced by `jit`, `replay`, both hosts and CUDA graphs;
* they are not those of `U(P)`, nor of any stack that rounds between the GEMM and its epilogue;
* they are at least as accurate, with one rounding fewer on the same arithmetic;
* when no hook or library epilogue can take the region, `U(R_acc(P))` is what runs: the GEMM writes f32 and the region reads it, which is slower
  than `U(P)`. The planner reports it (section 5) instead of silently keeping the rounding, so that a plan's numerics never depend on whether a
  fusion happened. X1 confirms that this reference is observable for cuBLASLt: the same algorithm configuration, re-encoded with an f32 `D`, is
  accepted by cuBLASLt, and rounding its output gives the 16-bit output bit for bit, in every run of both postures. The GELU algorithms are the
  exception: their configuration cannot be encoded with an f32 `D` (3.1.1).

**(iii) Library: bytes defined by the library under a pin.** A cuBLASLt epilogue or a cuDNN fused graph computes `L_{π,e}(a, b, s)` for the pin
`π` and the epilogue `e` with its attributes (the bias type, `beta`, the auxiliary output). There are two cases:

* **Measured-exact against (ii).** The epilogue's arithmetic can be written as a region: a bias add, ReLU, `beta * C` with `beta = 1`, or a
  per-channel scale. Export runs the library epilogue and `U(R_acc(P))` on seeded and special-value inputs at every bucket and refuses a difference.
  `U(R_acc(P))` uses the library's f32 form of the same configuration. X1 found cuBLASLt's `BIAS`, `RELU_BIAS` and `beta * C` measured-exact on every
  element of 64 runs each. It also found an algorithm whose bias + `beta * C` is not (cuBLASLt 13.1.0, section 6.1). The verification has to be per
  pin, never per epilogue.
* **Binary-defined** (RFC 0001, section 11.1). The arithmetic cannot be written down exactly, as for cuBLASLt's GELU (X2). The bytes are the
  binary's: deterministic and pinned. Export records their distance from `U(R_acc(P))` and refuses a distance above a declared tolerance. The fusion
  report marks the region.

In both cases the cuBLASLt version and the SM count are checked at load as today. A canary tells two builds of one version apart: the digest of the
epilogue's output on seeded operands, as cuDNN entries have. A plan that uses the same library epilogue with the same pin as an existing stack is
byte-identical to that stack. Contract (iii) is how a port matches an engine that itself fuses through cuBLASLt.

### 2.4 Who takes which contract

| situation | contract | routes |
|---|---|---|
| a port that must match a stack that does not fuse its epilogues | (i) | (a); (i-b) hooks on its own externs; (i-c) prologues; (i-d) a generated GEMM where measured equal; (i-e) cuDNN graphs with a 16-bit intermediate |
| a port of a stack that fuses through cuBLASLt or cuDNN | (iii), with the stack's pins | (b) |
| a port of a stack that fuses in its own kernels (CUTLASS) | (i) against that stack | an extern that reproduces the stack's kernel and epilogue, as ports do now |
| a new model, or a port accepted by tolerance against its original model | (ii), or (iii) | (c) on the accumulator, (b) |

Nothing changes bytes without a declaration (section 4):

* the default stays exact;
* a mode is part of what the plan is derived from, so a graph whose mode changed is refused as stale;
* the fusion report lists every rounding point that a mode removed.

A user who needs the bytes of a stack and wants the speed has two choices. The speed can come from routes that keep the bytes, which (i-d) and (i-e)
make larger than expected. Or the user accepts that the bytes are a new reference: validated against the original model by tolerance, and from then
on pinned and reproduced exactly, like any other plan.

### 2.5 What does not change

* Pins and digests.
* `jit == replay`, and the Rust and C++ hosts against Python.
* CUDA graph capture.
* The plan's arena and its order.
* Equality per architecture stays per architecture: an epilogue on `tcgen05` and the same epilogue on `mma.sync` are two binaries with two pins.
* RFC 0001's contract for the region itself (no contraction unless declared, every narrowing cast in inline PTX, context-free code) applies
  unchanged to the code that a hook inlines.

## 3. Routes

### 3.1 Route (b): library epilogues

#### 3.1.1 cuBLASLt

The table lists what cuBLASLt offers for f16 and bf16 operands with an f32 accumulator. "Exact" in the X1 column means equal to contract (ii) on
every element, special values included.

| epilogue | arithmetic | side output | X1 (sm_103, cuBLASLt 12.8.4, 64 runs each) |
|---|---|---|---|
| `BIAS` | `acc + bias` (a bias of the output type), one rounding | none | exact |
| `RELU` | `max(acc, 0)` | `RELU_AUX`: a bit mask | exact, and also equal to contract (i): ReLU commutes with the rounding |
| `RELU_BIAS` | `max(acc + bias, 0)` | `RELU_AUX_BIAS`: a bit mask | exact |
| `GELU`, `GELU_BIAS` | tanh-approximated GELU of `acc [+ bias]` | `GELU_AUX[_BIAS]`: the pre-activation, rounded | binary-defined: differs from torch's tanh GELU of the same accumulator on 0.1 to 0.2% (bf16) and 0.8 to 1.1% (f16) of elements, by up to 16 ULP (f16); the auxiliary output is exactly `ρ(acc + b)` |
| `beta * C` (`beta = 1`) | `acc + C`, one rounding | none | exact |
| `BIAS` with `beta * C` | `(acc + C) + bias`, one rounding | none | exact in that order on 12.8.4; one algorithm of 13.1.0 is not (6.1) |
| per-channel `alpha` vector | `alpha[n] * acc` (`POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST`) | none | not measured |
| FP8 scales, `AMAX_D` | per-tensor and block scales, an absolute maximum | `amax` | not measured |

What cuBLASLt cannot do:

* SiLU, erf GELU, periodic activations, gated activations, norms, rotary embeddings or overlap-add;
* a bias of another type than the output, so a graph whose bias is f32 for a bf16 output does not match;
* the per-row (token) scale that a norm prologue would need: in tilefish's row-major layout, cuBLASLt's alpha vector runs along `N`, not `M`.

* **Pinning.** The epilogue is part of the heuristic query and of `cublasLtMatmulAlgoCheck`, so the pin is chosen for the epilogue. The GELU
  epilogues got another algorithm than the plain GEMM on every shape (ids 70, 71 and 21 instead of 66). Everything else stays as today: the 64-byte
  algorithm re-encoded from its configuration, the version, the SM count, the postures and imported tables, and the vetting of the bytes. The
  no-split-K rule of a GEMM with a bias extends to every epilogue but `DEFAULT`, unless the declaration opts in.
* **Verification at export**, per bucket and per pin, on seeded and special-value inputs. A measured-exact epilogue is checked against
  `U(R_acc(P))`, computed with the library's f32 form of the same configuration. A binary-defined one is checked against the same reference, within
  its declared tolerance. An algorithm encodes its output type, so the f32 form is the same configuration re-encoded with an f32 `D`. That f32 form
  cannot be encoded for the GELU algorithms. X1 then used the f32 output of the plain GEMM's pin, whose accumulator was the same in every run, as the
  auxiliary outputs show. Q6 (section 8) recommends making that the rule for such algorithms.
* **At load**: the version and SM count checks as today, `cublasLtMatmulAlgoCheck` with the epilogue's descriptor, and a canary. The canary is the
  digest of `D` (and of the auxiliary output) for operands made on the device from a seed, by the PTX module that cuDNN canaries use
  (`tilefish._canary`). It is optional per entry and written by default for epilogue entries.
* **Speed is not a given.** On three of five shapes `GELU_BIAS` saved 22 to 48% over the plain GEMM followed by a generated bias + GELU kernel. On two
  shapes its pin was a slower algorithm and lost 9 to 31% (X4). The library epilogue changes the bytes, so the planner cannot choose it by timing.
  The advisor reports the measured times, and the declaration chooses.
* **Declared, or rewritten.** A library epilogue can be used in two ways:
  * as a library op declared with the epilogue (`impl="cublaslt.matmul", epilogue="gelu_bias"`). That is an ordinary call with the library's
    numerics, as `gemm_bias` is today;
  * as a rewrite of graph arithmetic into an epilogue. The planner does it only under `epilogues="library"` (section 4), only for the patterns of
    the table, and only where verification passes. Otherwise it falls back to route (a).

#### 3.1.2 cuDNN fused graphs

cuDNN's backend graphs fuse a convolution with pointwise operations into one engine: a bias add, GELU, SiLU (through `SWISH`), `SIN`, a multiply or
an add. Most of these engines are runtime-fusion engines, which compile a kernel with NVRTC when the plan is built. They cover convolution + bias +
periodic activation, which cuBLASLt cannot. X5 (sm_103, f16, 1-D convolutions with 7 taps) measured the following:

* **cuDNN 9.10.2**, the version tilefish pins today, with NVRTC 12.8 or 12.9: only convolution + bias and convolution + bias + ReLU with an f32
  intermediate have engines, and those engines are precompiled. Every runtime-fusion engine fails to compile on sm_103: GELU, the periodic
  activation, any 16-bit intermediate, an f32 output.
* **cuDNN 9.13.0** with NVRTC 13.0: every graph builds.
  * With an f32 intermediate, bias and bias + ReLU differ from the rounded composition on 31% and 15.5% of elements (contract (ii)-like).
  * With the convolution's virtual output typed f16, the fused graph rounds there. Convolution + bias and convolution + bias + ReLU equal the
    standalone pinned convolution followed by the epilogue, **bit for bit**, on all three shapes: contract (i).
  * The periodic activation differs from torch's `sin` composition by 1 ULP on 0.04% of elements; tanh GELU differs on 1.1%, by up to 16 ULP.

The engine configuration of a fused graph would be pinned as a convolution's is (engine index, knobs, cuDNN version, SM count) and checked by the
canary. The costs are larger than cuBLASLt's:

* the Rust host's cuDNN binding has to build the same operation graph from the manifest entry, which needs new node kinds in the entry;
* NVRTC compiles at load, which costs start-up time and must happen before any capture;
* a runtime-compiled kernel's bytes depend on the NVRTC that compiled it, and also on the cuDNN version, as the move from 9.10.2 to 9.13 showed. A
  canary catches that; a version number does not.

### 3.2 Route (c): epilogue and prologue hooks in tilefish's kernels

#### 3.2.1 Hooks on externs

An extern declares where its kernel would accept an epilogue, as RFC 0001 section 7 sketched:

```python
@tilefish.extern(gluon=conv_kernel, ...,
                 epilogue=tilefish.hook("EPILOGUE", value="acc", out="y"),     # or value="out": after the kernel's own rounding
                 prologue=tilefish.hook("PROLOGUE", operand="x"))
def conv(x: Tensor[...], w: Tensor[...], y: Out[Tensor[...]], ...): ...
```

The kernel calls the hook in its source where it would store (`tile = EPILOGUE(acc, rows, cols, row_mask, col_mask, *EPILOGUE_ARGS)`), or after it
loads an operand tile. Without a fusion the hook is the identity and the extern compiles as before. Its digest is unchanged too, because the hook
parameter defaults to an identity function that tilefish provides. With a fusion, the planner passes a generated `@triton.jit` (or `@gluon.jit`)
function, which `tilefish.lower` makes from the region, and the region's other inputs become extra arguments of the step:

* **The value.** `value="acc"` passes the f32 accumulator tile (contract (ii)). `value="out"` passes the tile after the kernel's rounding to its
  output type (contract (i)). An extern that can offer both declares both.
* **What the hook may read**: the tile, its row and column indices, side inputs per row, per column and per element (strided, through the existing
  `Strided[...]` and `StrideOf` machinery), and device scalars.
* **What the hook may write**: the main output, which it returns to the kernel, and side outputs, which it stores itself, each per element of the
  tile. The kernel keeps its own store path (TMA or pointers) for the main output. An example of a side output is the residual stream of an add +
  norm.
* **Gluon.** A Gluon kernel holds the accumulator in tensor memory and loads it into registers in a layout of its choice. The generated hook takes
  that layout as a `constexpr` and builds its index vectors in the matching slice layouts. A Gluon kernel that splits its epilogue into sub-tiles
  calls the hook once per sub-tile.
* **Identity.** Each pair of an extern and a region is a new op (`<extern>+<region hash>`) with its own entries and pins. The region's content hash
  is in its name, as fused regions' hashes are (RFC 0001, section 4.2). If export-time verification refuses a pair, the pair falls back to route (a),
  with the reason in the fusion report.

This is the cheapest route that saves time. Ports whose convolutions are already Gluon externs get their bias + activation epilogues fused without
writing them into the kernel by hand. When the hook takes the rounded value, the fusion is under contract (i).

#### 3.2.2 Generated GEMM and convolution ops

For GEMMs that are library calls today, route (c) needs a GEMM of tilefish's own. It would be a generated op, as `tilefish.contiguous` and `im2col`
are: tilefish writes its source per architecture, and the planner fills its epilogue and prologue hooks.

| arch | MMA | kernel structure |
|---|---|---|
| sm_100, sm_103 | `tcgen05.mma` (`kind::f16`), with the accumulator in tensor memory | Gluon (or Triton): TMA loads, a warp-specialised pipeline, a persistent data-parallel tile order (bytes independent of the SM count), epilogue sub-tiles that overlap the next tile's main loop, and two-CTA MMAs as a later option |
| sm_120 | `mma.sync.m16n8k16` | Triton `tl.dot` or Gluon, TMA loads |
| sm_90 | `wgmma.mma_async` | Triton `tl.dot` or Gluon |

* **Numerics.**
  * The accumulation is in f32.
  * There is no split-K and no stream-K by default. Their reduction order would depend on a split count or on the SM count, and a persistent extern
    may only depend on the SM count when it declares it (manifest schema 3, section 4.3).
  * TF32 is refused for f32 operands by the existing guard (`tilefish._tf32guard`). Triton has been seen to compile an IEEE f32 `tl.dot` into TF32
    `tcgen05` MMAs.
  * X3 measured that on sm_103 neither the tile configuration nor the kernel changes the bytes: all 99 configurations wrote the same bytes, and those
    were cuBLASLt's whenever cuBLASLt did not split `k`. A configuration per bucket is then a speed choice, as an elementwise region's is (RFC 0001,
    O5). This has to be measured per architecture, not assumed.
* **Configuration per bucket** goes through the existing `by_bucket` mechanism: tile sizes, stages, warps, CTAs and clusters. It is chosen at export
  by a fixed rule or read from a recorded table (like `AlgoTable`), and never autotuned silently. Every binary is pinned like any Triton entry.
* **SASS assertions.** The generator declares assertions on its own kernels: the architecture's MMA (`tcgen05.mma kind::f16`, `HGMMA` or
  `HMMA.16816.F32`), `no_tf32` and `no_spills`. A new structural assertion would make "the hook did not change the GEMM" visible at build: the MMA
  instruction sequence of a hooked binary equals the unhooked binary's (opcodes, shapes, kinds and accumulate flags, in order). It is a necessary
  condition, not a proof, and the byte verification stays the guard.
* **Choice of implementation.** The GEMM's declaration says which implementation it is (`impl="cublaslt.gemm"` or `impl="tilefish.gemm"`). The
  planner never swaps a library GEMM for a generated one on its own. Contract (i-d) is an explicit opt-in (Q4): a generated GEMM declared to equal
  the library pin it replaces. Export verifies that at every bucket and keeps the library GEMM, with route (a), at the buckets where it does not hold.
* **The risk is speed.** A generated GEMM that is slower than the pinned library GEMM it replaces can cost more than the epilogue saves:
  * X3 measured a straightforward Triton `tcgen05` GEMM, the best of 99 configurations, at 1.22 to 1.44 times cuBLASLt's time on five shapes of a
    30-SM slice;
  * X4 measured a naive epilogue in that kernel, non-warp-specialised, with its side inputs in the accumulator's layout. It was slower than the
    library GEMM followed by a separate epilogue kernel on every shape.

  Route (c) with a generated GEMM is worth it where `saving > (r - 1) * T_gemm` for the measured ratio `r`. Getting there needs the kernel structure
  of the table above (X8).

#### 3.2.3 What an epilogue tile can compute

* **Lane-local work and per-row or per-column inputs** (bias, activations, residual add, dequantisation scales, casts): any tile shape.
* **Segments inside a tile** (a per-head norm, MX block scales over 32 columns, rotary pairs within a head). The tile's columns must hold whole
  segments: `BN` is a multiple of the segment, and no tile straddles a boundary between `q`, `k` and `v`. The reduction needs an order, because an
  unordered reduction runs alone (RFC 0001, D2). So this waits for ordered segmented reductions (0001 phase 4) and row-local movement (phase 3).
* **Whole rows** (a norm over the hidden size, a per-row quantisation scale). A tile holds a whole row only when `N <= BN`, which `tcgen05` allows up
  to 256 to 512 columns (its tensor memory holds 128 x 512 f32 values per SM). Larger rows have to be split:
  * the epilogue stores the side output and a partial sum per tile (`[M, N / BN]`, an ordered sum over the tile's columns);
  * the next GEMM's prologue, or a small kernel, finishes the norm by summing the partials in tile order.

  That is exact against `U(P)` only when the region declares that very order (`tilefish.sum(order=...)` over tiles of `BN`). A graph can declare it,
  because tilefish's orders are explicit. The norm's rounding of `h` itself follows contract (i) or (ii), as above.
* **Halos** (overlap-add after a transposed convolution computed as GEMM + col2im). An output sample sums `kernel / stride` rows of the GEMM's result.
  A tile that recomputes `kernel / stride - 1` extra rows completes its outputs without atomics; the recomputation is that fraction of the GEMM. The
  tile adds the contributions in col2im's order, then applies the bias and the activation. That is exact under (ii) against `U(R_acc(P))` with the
  same order. A polyphase direct convolution would instead fold the overlap-add into the `k` loop: another summation order, so another algorithm,
  outside these contracts.

#### 3.2.4 Prologues

A prologue computes an operand tile instead of loading it: a norm's `(x * rstd * w).to(bf16)`, or a dequantisation `(q.to(f32) * scale).to(bf16)`.
With `tcgen05` and `wgmma` the operand goes from global memory to shared memory by TMA, and the MMA reads shared memory directly. A transform
therefore needs one of two things:

* a round trip through registers on warps of its own: load the tile, transform it, store it back and fence;
* an MMA that reads the operand from registers (`mma.sync`, `wgmma`) or from tensor memory (`tcgen05`).

The work is repeated for every column tile, `N / BN` times per element. At `BN = 256` that is about 0.6% of the MMA's operations, but on CUDA cores,
so it has to overlap with the MMAs.

The numerics are the easy part. The operand bytes are the region's, so a prologue keeps contract (i) whenever the kernel stays the same (i-c), and,
on sm_103, X3 suggests a generated kernel can be the library's bits. Library GEMMs take no prologue (FP8 scale factors aside), so a prologue into a
library GEMM means a generated GEMM, with route (c)'s speed risk.

### 3.3 What each route is worth and costs

| route | saves | speed risk | covers | engineering cost |
|---|---|---|---|---|
| (a) a generated kernel after the call | nothing (today) | none | everything | done |
| (b) a cuBLASLt epilogue | the round trip of `y` and one launch, while the GEMM stays the library's: 9 to 48% of GEMM + epilogue on the measured shapes | the epilogue's pin may be a slower algorithm (GELU: slower than unfused on 2 of 5 shapes) | bias, ReLU, tanh GELU, residual add, per-channel scale, FP8 | small to medium: binding, a new library impl in both hosts, verification, canary |
| (b) a cuDNN fused graph | the same, for convolutions | engine choice, NVRTC at load; needs cuDNN 9.13+ on sm_103 | conv + bias + pointwise chains, contract (i) with a 16-bit intermediate | medium to large: graph building in both hosts, runtime compilation |
| (c) a hook on an extern | the round trip and one launch | none: the kernel is the user's | whatever the extern is | medium: hook ABI, planner, generating the hook, verification |
| (c) a generated GEMM or convolution | the round trip and one launch, plus patterns no library has, with contract (i) where measured equal | `r` of 1.22 to 1.44 for a straightforward Triton GEMM; a naive fused epilogue loses | everything tile-local | large: a kernel family per architecture, configurations per bucket, performance parity work |
| prologue | the round trip of the operand | the transform must overlap the MMAs | norm, dequantisation | large (part of the generated GEMM) |

The advisor's estimates rank the patterns (section 0):

* The convolutional model's GEMM and convolution epilogues, with periodic activations and overlap-add, have the largest absolute savings. cuBLASLt
  does not cover them; cuDNN fused graphs (for convolutions, cuDNN 9.13) and route (c) do.
* The transformer models' add + norm, gated and GELU epilogues are partly covered by cuBLASLt (bias + GELU, the residual add) and fully by route (c).

## 4. Surface syntax

A sketch; the names were question Q8, decided as proposed.

```python
@tilefish.graph(fusion=tilefish.fusion())                              # epilogues="exact": the default, contract (i) only
@tilefish.graph(fusion=tilefish.fusion(epilogues="accumulator"))       # contract (ii): R_acc, hooks on the accumulator
@tilefish.graph(fusion=tilefish.fusion(epilogues="library"))           # (ii) and (iii): library epilogues where they exist
with tilefish.fused(epilogues="accumulator"):                          # per region, as RFC 0001's scopes
    h = (gemm_up(x, w).to(f32) + b.to(f32))

@tilefish.library("gemm_gelu", impl="cublaslt.matmul", epilogue="gelu_bias")   # a declared library epilogue: an ordinary call
@tilefish.library("gemm", impl="tilefish.gemm", acc=f32)                       # a generated GEMM, with hooks the planner may fill
@tilefish.library("gemm", impl="tilefish.gemm", acc=f32, equal="cublaslt")     # (i-d): must equal the cuBLASLt pin, bucket by bucket
```

* `epilogues="exact"` is the default. It fuses only what keeps `U(P)`'s bytes:
  * hooks with `value="out"` on externs and generated ops that are the reference;
  * prologues into kernels that stay the same;
  * library fusions that round where `U(P)` rounds.

  It is on by default, as phase 1 of RFC 0001 is (D1), because it cannot change a byte. Export verifies every such fusion.
* `epilogues="accumulator"` applies `R_acc` and uses hooks with `value="acc"`.
* `epilogues="library"` adds the rewrite into library epilogues.
* `tilefish.barrier(y)` keeps `y` stored and stops any epilogue fusion of it, as in RFC 0001.

## 5. Interaction with plans, reports, the advisor, SASS assertions, traces, hosts and schemas

* **Plans.** A hooked extern or a generated GEMM is a `kernel` step of a generated entry, with the region's inputs as extra arguments. A library
  epilogue is a `library` step whose arguments follow the entry's ABI (bias, auxiliary output, `C`). Plan schema 1 is unchanged.
  * Results that are fused away get no buffer, so arenas shrink.
  * A residual add through `beta * C` can update the residual stream in place (`D == C`, an `alias` of manifest schema 3 section 5.4).
  * A hook may write an `InOut` or carried buffer under RFC 0001's rule 4, which allows only lane-local reads of the old version.
* **Taps and `run_step`.** A GEMM result that is fused away has no buffer. `taps=` refuses it, with the advice to add `tilefish.barrier` or to tap
  `U(P)` (or `U(R_acc(P))`), whose intermediates are, under the contract, the values the fused kernel computes.
* **`reference` mode** interprets `R_acc(P)`. The call's f32 result is torch's f32 GEMM or convolution, within the library's tolerance as today.
* **CUDA graphs.** Unchanged. cuDNN runtime-fusion engines compile at load, before any capture.
* **The fusion report** (`<graph>.fusion.json`) gains these fields for each region fused into a call:
  * the call and the route (`library`, `hook`, `prologue`);
  * the hook's value (`acc` or `out`);
  * the contract (`exact`, `accumulator`, `measured-exact`, `binary-defined`);
  * the rounding points that `R_acc` removed;
  * the library epilogue and its pin;
  * the verification's result: equal, or the largest ULP distance for a binary-defined epilogue.

  A requested fusion that did not happen is listed with its reason, and under `accumulator` with the f32 intermediate it costs.
* **The advisor** turns `epilogue-library` and `prologue-library` from future work into actions. For each candidate it reports the routes that can
  take it under the graph's declared mode, the contract each route gives, and the saving.
  * For a generated GEMM it also reports the break-even slowdown, `saving / T_gemm`: how much slower the GEMM may be before the fusion loses. It
    does not predict the generated GEMM's speed.
  * A candidate that needs a mode the graph does not declare says so, for example "exact: none; accumulator: hook on the extern, 150 us".
  * Patterns are grouped by route, so that the report says which route would unlock the most.
  * Routes that give different bytes are never chosen by the advisor or the planner on time; the advisor's measured times inform the declaration.
* **SASS assertions** (`tilefish.sass`). A hooked binary is judged as any build is. The generated GEMMs declare their own assertions (MMA kind,
  `no_tf32`, `no_spills`). The MMA-sequence assertion of 3.2.2 is new.
* **Traces** (`trace/`). A hook's generated source carries the graph body's lines (the region's `loc`s), so `tilefish-explain annotate` attributes
  the epilogue's instructions to the graph-body lines that wrote them. In-kernel instrumentation (issue #5, section 2) can time the epilogue phase of
  a fused kernel.
* **Hosts and the C ABI.** Route (c) needs nothing from either host: kernel steps with pointers, TMA descriptors (`tensordesc`) and global scratch
  already exist. Route (b) needs:
  * the new library impl in Python (`tilefish._cublaslt`, `_library`) and in the crate (`ffi/cublaslt.rs`, `gemm.rs`, `library.rs`);
  * the canary;
  * in the C ABI, an additive accessor for a GEMM's epilogue and its extra tensors (a minor ABI version). The C++ gate covers it through the crate.

  cuDNN fused graphs need the graph builder in both hosts.
* **Schemas.** Route (c) changes nothing: its entries are generated `triton` entries. Route (b) is additive: a new library impl (`cublaslt.matmul`)
  with an `epilogue` field that has more values, `aux` and `beta`, and the optional `canary`.
  * The two existing impls stay byte for byte as they are, so every manifest that exists keeps loading.
  * A reader that predates the new impl refuses an entry that has it.
  * The change is documented in `docs/manifest-schema-3.md` section 9, with the Python writer, both readers and the vectors in step.
  * The modes need no manifest field (as RFC 0001 Q8): they are in each generated source's header and in the fusion report.

## 6. Experiments

### 6.1 Phase 0 results (2026-11-16, sm_103)

The setup for every experiment:

* a B300 MIG slice: sm_103, 30 SMs;
* torch 2.9.1, Triton 3.5.1 and ptxas 12.9;
* cuBLASLt 12.8.4 unless stated otherwise;
* the tilefish cuBLASLt binding, extended in the experiment scripts with the descriptor attributes of each epilogue;
* both heuristic postures: `tilefish`, which asks for no split-K for any epilogue but `DEFAULT`, and `torch`;
* times are medians over CUDA graphs of 20 back-to-back calls.

The scripts are not part of the repository; each experiment says enough to repeat it.

**X1: which cuBLASLt epilogues equal a composition.**

The matrix:

* 16 shapes: `M` in {32, 128, 640, 2048}, by `N x K` in {1536 x 512, 3072 x 1024, 1024 x 3072, 4096 x 4096};
* bf16 and f16 operands and outputs;
* both postures, so 64 runs per epilogue.

The inputs: `A ~ N(0, 1)` and `B ~ N(0, 1/K)`, so that the accumulators are of order 1 and GELU sees its curved part. The bias is `N(0, 0.25)` and
`C` is `N(0, 1)`. Both have eight special columns: `+0`, `-0`, `+inf`, `-inf`, NaN, the largest finite values of both signs, and a subnormal.

Each epilogue ran with its own pin and was compared bit for bit with two references:

* **(ii)**: `ρ(epi(acc))`, computed in f32 by torch, where `acc` is the f32 output of the same algorithm configuration re-encoded with an f32 `D`
  (for the GELU algorithms, which cannot be encoded that way, the f32 output of the plain GEMM's pin);
* **(i)**: `ρ(epi(ρ(acc)))`, where `ρ(acc)` is the 16-bit plain output of the same configuration (or of the plain pin).

| epilogue | equal to (ii) | elements that differ from (i) | notes |
|---|---|---|---|
| `DEFAULT` | 64/64 runs | 0 | the f32 re-encoding's output, rounded, is the 16-bit output in 64/64 runs: the accumulator is observable |
| `BIAS` | 64/64, special values included | 30.3 to 31.0% | the pin is the plain GEMM's on 14 of 16 shapes; the same bytes on the other two |
| `RELU` | 64/64 | 0 | commutes with the rounding |
| `RELU_BIAS` | 64/64 | 15.1 to 15.6% | |
| `beta * C` | 64/64 | 28.9 to 29.6% | |
| `BIAS` + `beta * C` | 64/64 as `ρ((acc + C) + b)`; as `ρ((acc + b) + C)` only 6/64 | 26.2 to 26.7% | the library's order |
| `GELU` | 0/64: 0.11 to 0.14% of elements differ (bf16), 0.81 to 0.94% (f16), by up to 16 ULP (f16) | 26.8 to 28.6% | another algorithm than the plain GEMM's on 16/16 shapes |
| `GELU_BIAS` | 0/64: 0.18 to 0.21% (bf16), 1.02 to 1.11% (f16) | 31.1 to 31.6% | |
| `GELU_AUX_BIAS` | `D` as `GELU_BIAS`; the auxiliary output equals `ρ(acc + b)` in 64/64 | `D` differs from `gelu(aux)` on 28 to 30% | GELU is applied to the unrounded pre-activation; the GELU kernels' accumulator is the plain GEMM's |

Every pin of X1 had a split-K of 1, so X1 says nothing about split-K with an epilogue.

**X6: X1 under cuBLASLt 13.1.0** (torch 2.9.1+cu130, the same matrix). The conclusions of X1 hold, with three findings:

* The `tilefish` posture now pins split-K (3 or 4, with f32 partial sums) for the plain GEMM and for `beta * C` at `M = 32` and `N x K` of
  1024 x 3072 and 4096 x 4096. These pins remain equal to (ii), computed with the same configuration's f32 output.
* `BIAS` + `beta * C` with algorithm 21 at 32 x 1024 x 3072 differs from both orders on 30% of elements, with both output types. The other 15 shapes
  are exact. A library epilogue's arithmetic is a property of the pin, so verification must be per pin.
* The GELU distances grow (up to 2.4% of elements, 47 ULP in f16). The GELU kernels' auxiliary output now differs from `ρ(acc + b)` on 1 or 2 of 16
  shapes per group: their accumulator is no longer always the plain pin's.

**X2: what cuBLASLt's GELU computes.** We ran the GELU epilogue with an f32 `D` on the plain GEMM's algorithm (shapes 640 x 3072 x 1024 and
2048 x 4096 x 4096), so that its input is the plain GEMM's f32 output. We compared it with candidate formulas, each written as scalar PTX (`mul.rn`,
`add.rn`, `fma.rn`, `tanh.approx.f32`) so that no compiler contraction blurs them:

| candidate | elements equal |
|---|---|
| torch's tanh GELU | 6.8% |
| torch's erf GELU | 2.7% |
| the best of 64 variants (8 inner forms, 4 outer forms, 2 roundings of the constants): `0.5 x (1 + tanh.approx(k x fma(c, x^2, 1)))`, with `k = f32(sqrt(2/pi))` and `c = f32(0.044715)` | 99.80% |

The remaining 0.2% of elements have `|x|` from 0.07 to 4.5 and differ by a median relative 1.7e-6, at most 2.1e-2. So cuBLASLt's GELU is
binary-defined: close to a known formula, equal to none of those tried.

**X3: a Triton GEMM against the pinned cuBLASLt GEMM.**

The kernel: TMA descriptors, `tl.dot` to `tcgen05` (`kind::f16`), a grouped tile order and no split-K.

The sweep: 99 configurations (`BM` 64 or 128, `BN` 64 to 256, `BK` 64 or 128, 2 to 5 stages, 4 or 8 warps, with and without a warp-specialised
loop) on five shapes.

The bytes:

* Every configuration wrote the same bytes as every other (99/99 on every shape).
* Those bytes are cuBLASLt's pin, under both postures, on every element.

X3b extended the comparison to six more shapes (32 x 1024 x 3072, 32 x 4096 x 4096, 640 x 3072 x 1024 in bf16 and f16, 2048 x 4096 x 4096 in f16,
and 100 x 1000 x 1000) and three configurations, with random operands and with special values in the operands. The special values were subnormals,
products that underflow, rows that overflow to infinity, `+inf`, `-inf`, a NaN with a payload, and `-0`.

* Under cuBLASLt 12.8.4 the outputs are equal everywhere.
* Under 13.1.0 they are equal wherever the pin has no split-K. They differ on 0.19 to 0.20% of elements where it has one (63 of 32 768 and 259 of
  131 072, the 32-row shapes).

The GELU kernels of X1 (algorithms 70, 71 and 21) also have the plain GEMM's accumulator. Taken together: on sm_103 the bytes of these `kind::f16`
GEMMs depend on the summation order over `k`, and not on the tile, the algorithm or the library, as long as nobody splits `k`. That is a
measurement on 11 combinations of shape and type and two cuBLASLt versions; sm_120 and sm_90 are not measured.

The speed, in microseconds, alongside X4 (`r` = Triton / cuBLASLt):

| shape `M x N x K` | cuBLASLt | Triton (best config) | `r` |
|---|---|---|---|
| 640 x 4096 x 1024, bf16 | 18.9 | 23.2 | 1.24 |
| 640 x 1024 x 4096, bf16 | 15.0 | 21.4 | 1.43 |
| 2048 x 3072 x 1024, bf16 | 39.6 | 48.3 | 1.22 |
| 128 x 3072 x 1024, bf16 | 4.4 | 6.4 | 1.44 |
| 8192 x 3072 x 768, f16 | 121.3 | 152.8 | 1.26 |

**X4: the routes for GEMM followed by bias + tanh GELU, and GEMM followed by a residual add** (microseconds, the shapes of X3).

The bias + GELU routes:

* **(a)** is the pinned GEMM followed by a generated elementwise kernel;
* **(b)** is `GELU_BIAS`;
* **(c, acc)** is the Triton GEMM with the epilogue on the accumulator;
* **(c, rounded)** is the same kernel with the epilogue applied after rounding to the output type.

| shape | (a) | (b) | (c, acc) | (c, rounded) | residual (a) | residual: `beta * C` |
|---|---|---|---|---|---|---|
| 640 x 4096 x 1024 | 31.2 | 24.2 | 33.4 | 35.7 | 24.3 | 20.5 |
| 640 x 1024 x 4096 | 18.9 | 24.8 | 29.1 | 30.2 | 17.0 | 15.5 |
| 2048 x 3072 x 1024 | 68.4 | 40.8 | 75.6 | 82.2 | 60.2 | 44.4 |
| 128 x 3072 x 1024 | 7.1 | 7.7 | 10.3 | 10.8 | 5.9 | 4.9 |
| 8192 x 3072 x 768 | 231.8 | 120.2 | 239.8 | 264.6 | 206.8 | 134.8 |

* **The library's residual add** (`beta * C`) beats (a) on every shape, by 9 to 35%.
* **`GELU_BIAS`** beats (a) on three shapes, by 22 to 48%. It loses on two, where its pin is algorithm 71 and the plain GEMM's is 66.
* **The naive Triton epilogue** loses everywhere: it is not warp-specialised, it uses `exp` and a division for `tanh`, and it loads side inputs in the
  accumulator's register layout. A Triton residual add in that layout was 1.3 to 2.2 times slower than (a), and is not in the table.
* **(c, rounded) equals (a) bit for bit** on all five shapes. That is contract (i) with a generated GEMM in practice, at a speed that does not pay
  yet.

**X5: cuDNN fused graphs.** The graphs: convolution + bias, + ReLU, + tanh GELU, and + a periodic activation (`h + sin(a h)^2 / a`), each with an
f32 and with an f16 intermediate. The convolutions are 1-D with 7 taps, posed as 2-D, NHWC, f16: `N x C x L` of 1 x 256 x 4096, 1 x 512 x 2048 and
4 x 128 x 8192, with `K = C`. We used cuDNN's Python frontend 1.32.3. Section 3.1.2 gives the results. Times were not measured: the harness did not
capture the frontend's launches in the CUDA graph (X13).

**What the results change.** Before these measurements, contract (i) for a fused epilogue looked reachable only through hooks on externs. Now:

* on sm_103, a generated GEMM can be exact against the library stack wherever the library does not split `k` (i-d);
* cuDNN 9.13 fused graphs with a 16-bit intermediate are exact for bias and ReLU (i-e);
* cuBLASLt's bias, ReLU and residual epilogues are measured-exact against (ii);
* the price of the generated route is the GEMM's speed, not its numerics.

### 6.2 Experiments still needed

| | experiment | decides | needs |
|---|---|---|---|
| X7 | X1, X3 and X3b on sm_120 (`mma.sync`, cuBLASLt's f16 split-K pins) and on sm_90 (`wgmma`) | whether (i-d) holds beyond sm_103 | an sm_120 and an sm_90 device |
| X8 | a Gluon `tcgen05` GEMM: persistent, warp-specialised, epilogue sub-tiles overlapping the next tile, TMA loads of side inputs, two-CTA MMAs; against cuBLASLt on a fixed suite of shapes, with and without a bias + GELU and a residual epilogue under contracts (i) and (ii). **Done** ("X8: results"), without two-CTA MMAs, which Triton 3.5.1's Gluon cannot express | whether E3 is worth building (Q5) | sm_103 |
| X9 | a generated split-K GEMM that reproduces cuBLASLt's split-K order (f32 partial sums in a workspace, reduced in a fixed order) | whether (i-d) can cover split-K pins | sm_103 |
| X10 | a norm prologue (`x * rstd * w` into the A operand) in the X8 kernel, against a separate kernel + GEMM | E5 | sm_103 |
| X11 | add + norm with the partial-sum split and a declared tile order; overlap-add with a halo; against their unfused plans | E4 | sm_103, sm_120 |
| X12 | an `out` hook and an `acc` hook on a real Gluon implicit-GEMM convolution extern: the hooked binary against the unhooked one followed by the region, at every bucket; the MMA-sequence assertion | E1 | sm_103, sm_120 |
| X13 | cuDNN fused graphs: times with the handle's stream captured, the canary's cost, NVRTC's compile time at load, and whether the bytes change with the NVRTC version | Q11 | sm_103 |
| X14 | cuBLASLt epilogues with `alpha` vectors and FP8 scales; `GELU_AUX`; the split-K rule with epilogues (the `torch` posture) | E2's scope | sm_103, sm_120 |

## 7. Staged plan, milestones and tests

### 7.1 Milestones

| milestone | content | done when |
|---|---|---|
| E0: experiments | X1 to X6 (section 6.1, done); X7 to X14 (6.2), each before the milestone that needs it | every decision of section 8 that depends on an experiment has its number |
| E1: hooks on externs | `tilefish.hook` (`value="out"` and `"acc"`); the hook functions generated from regions (Triton and Gluon, layouts as `constexpr`); the planner's merge of a region into a hooked call (lane-local work, per-row and per-column inputs, side outputs per element, `InOut` under rule 4); `epilogues="exact"` on by default and `"accumulator"` with `R_acc`; export verification of every pair against `U(P)` or `U(R_acc(P))`, with fallback to route (a); the fusion report's new fields; the advisor's routes and contracts for extern candidates | a Triton and a Gluon GEMM or convolution extern fuse bias + GELU, bias + periodic activation, and residual add with a side output, under (i) and under (ii), byte for byte against their references at every bucket on sm_103 and sm_120; the Rust and C++ gates run the fused plans launched and captured |
| E2: cuBLASLt epilogues | the `cublaslt.matmul` impl (bias, ReLU, GELU and their auxiliary outputs, `beta * C`), its canary, both hosts and the C ABI accessor; `epilogues="library"` for the measured-exact patterns; GELU as binary-defined with a tolerance; verification per pin | X1's matrix is a GPU test on sm_103 and sm_120 under both postures; every rewrite is verified at export; a forged canary is refused at load |
| E3: generated GEMMs | `impl="tilefish.gemm"`: the X8 kernel for sm_100 and sm_103 with hooks, configurations per bucket, SASS assertions and the MMA-sequence assertion; `equal="cublaslt"` (i-d) with verification per bucket; then `mma.sync` for sm_120 and `wgmma` for sm_90 (after X7); convolutions by implicit GEMM on the same templates | bytes that do not depend on the configuration are a test per architecture; on a fixed suite of shapes the generated GEMM is within the ratio the review sets (Q5) of the pinned library GEMM; the advisor's break-even says where it is worth using |
| E4: rows, segments and halos | per-head norm + rotary (with RFC 0001 phases 3 and 4 inside a tile); add + norm with a side output (whole rows, then the split over tiles with a declared order); gated activations with two accumulators; overlap-add with a halo | each use case of section 1 fused, equal to its reference under (i) or (ii), and faster than its unfused plan on sm_103 and sm_120 |
| E5: prologues and quantisation | norm and dequantisation prologues into generated GEMMs; MX block-scale and per-row quantisation epilogues | as E4 |
| E6: cuDNN fused graphs | if X13 confirms speed: `cudnn.conv_fwd` with a pointwise chain, a 16-bit intermediate for contract (i), cuDNN 9.13 or later on sm_103 | as E2, for convolutions |

E1 needs none of the others. It serves the ported models whose convolutions are externs already. E2 is independent of E1. E3 is the large one, and
it is the precondition of E4 and E5 for GEMMs that are library calls today.

### 7.2 Tests

* **Construction (no GPU).**
  * `R_acc`, IR in and IR out: a widening of a call's result is replaced; a 16-bit consumer, a returned or tapped result, and a fan-out keep the
    rounded value.
  * The planner's merges and refusals: a barrier, a fan-out, a reduction across tiles, a broadcast read of an `InOut` that is being written, a call
    between the two steps.
  * The names and identities of pairs of an extern and a region: the same region in two graphs is one op.
  * The hook's source equals the region's lowered source.
  * The fusion report and the advisor's labels.
  * The manifest vectors of the new library impl, in the Python and Rust readers.
  * Refusals of a malformed hook declaration.
* **Against the reference (GPU, the core).** Every fused plan is compared byte for byte with `U(P)` (contract (i)) or `U(R_acc(P))` (contract (ii)) at
  every bucket, on sm_103 and sm_120. The inputs are seeded random values, and special values in every side input: both zeros, infinities, NaNs with
  non-canonical payloads, subnormals. The plans are the use cases of section 1 as graphs, and random regions (RFC 0001's generator) fused into a
  Triton and a Gluon GEMM extern.
* **Library epilogues.**
  * X1 as a test: each epilogue against its measured-exact reference or its tolerance, per posture, output type and pin.
  * The canary is computed the same in both hosts.
  * A different build, simulated by a forged digest, is refused.
* **Generated GEMMs.**
  * The bytes are equal across every configuration of a bucket (X3's check).
  * `equal="cublaslt"` holds where the pin has no split-K and is refused where it has (X3b's cases).
  * `no_tf32` holds on f32 operands.
  * The MMA-sequence assertion holds, hooked against unhooked.
  * The TF32 guard.
* **Hosts.** The Rust gate and the C++ gate get fused plans of each route: launched, in CUDA graphs, with shared arenas and carried buffers. Rust
  and C++ are compared with Python byte for byte.
* **Speed (informative).** The advisor's estimate before fusion and the profiler's measurement after it, on synthetic models in `python/tests`. The
  ported models are measured privately.

### 7.3 E2 as built (2026-11-17)

**The impl `cublaslt.matmul`** (Q7; `docs/manifest-schema-3.md`, section 9.5). `@tilefish.library(..., impl="cublaslt.matmul", epilogue=..., beta=...)`:
`epilogue` is `none`, `bias`, `relu` or `relu_bias` (cuBLASLt's `DEFAULT`, `BIAS`, `RELU`, `RELU_BIAS`), `beta=1` reads `c [M, N]` (`beta * C` with
`beta = 1`); bf16 or f16 operands and a result of their type, the four layouts of `cublaslt.gemm`, both postures, the opt-ins, `sm_count_target` and
`workspace_bytes`. Its arithmetic is `ρ(f(acc))` with `c` added before the bias, the library's order (X1). `cublaslt.gemm` and `cublaslt.gemm_bias` are
unchanged, byte for byte and digest for digest. The no-split-K rule of a GEMM with a bias extends to every epilogue but `none` (3.1.1), in its own words.
An entry has `beta`, `contract`, `verification` and an optional `canary`; a reader of 0.0.11 refuses it. The Python binding, both readers, the crate's
load and launch, the C ABI (an accessor of the epilogue and of `c`, a minor version) and the generated hosts take it.

**Verification per pin** (Q6, contract (iii) measured against (ii)). Every pin is verified when it is made, at `jit` and at `export`: the pin runs on two
seeded operand sets (the operands of X1; special values in the first eight columns of the bias and of `c`; rows of `a` that give every epilogue a
NaN, a zero and an infinite accumulator, and in the second set rows whose accumulators overflow) and is compared bit for bit with the formula computed by torch in f32 on the accumulator of the same configuration re-encoded with
an f32 `D` (`"f32-form"`; the f32 pin of the same posture where cuBLASLt does not take that, `"plain-f32-pin"`, which no pin needed here), rounded once.
torch on CUDA gives the bits of tilefish's own lowering of the same ops (an f32 add, `torch.relu`, which is `where(h <= 0, 0, h)` bit for bit, and the cast
to the canonical NaN). The entry records `"measured-exact"` (no element differs) or `"binary-defined"` with the count of elements that differ and their
largest distance in ULP. `contract="measured-exact"` makes the build refuse the other.

X1 as a GPU test (`python/tests/test_gpu_matmul.py`): 14 shapes (`M` 32, 128, 640, 2048 by `N x K` of 1536 x 512, 3072 x 1024, 1024 x 3072, and
32 x 4096 x 4096, 100 x 1000 x 1000), bf16 and f16, both postures, seven cases (`none`, `bias`, `relu`, `relu_bias` without and the last three with `c`,
and `none` with `c`): 392 pins per cuBLASLt version on a B300 MIG slice (sm_103, 30 SMs).

* **cuBLASLt 12.8.4: 392 of 392 measured-exact**, the special values and accumulators included (rows of `a` that give a NaN, a zero, an infinity
  and an overflow to every epilogue). No pin splits `k`.
* **cuBLASLt 13.1.0: 372 of 392 measured-exact.** The 20 others are the pins of algorithm 21 at 32 x 1024 x 3072, both types, both postures:
  * `bias` with `c`: 26% of the elements differ, by up to 15 176 ULP (bf16) and 2 481 (f16), as X6 found;
  * `relu`, `relu_bias`, `relu` with `c` and `relu_bias` with `c`: 2048 to 2111 of 65 536 elements, all of them NaN pre-activations: **this algorithm's
    ReLU gives `0` for a NaN**, where every other pin keeps the NaN (`torch.relu`). A finding beyond X6: an epilogue's NaN rule is a property of the pin
    too, which only special accumulators show (a NaN in a bias or `c` showed it for the cases with them; the rows of `a` for a ReLU alone).
  * The pins of `none` with `c` that split `k` at `M = 32` (3 and 4, f32 partial sums) are measured-exact, as X6 found.

A run of the test on another architecture or version checks every record again on operand sets of its own and reports the outcome; on sm_120 a test
of f16 problems with `allow_f16_reduction=True` verifies the split-K pins cuBLASLt answers there (it waits for an sm_120 gate).

**The canary** (Q6: on by default). `canary=True` (the default) has `export` record `{"seed": 1, "out_digest": d}`: the pin run on operands made of the
seed on the device by the PTX module of the cuDNN canaries (`a`, `b`, the bias and `c` as tags 0 to 3), `out` and the workspace zeroed, and the digest of
`out`. A Python replay runs it at the first use of the entry in the process and device, the crate's load once per context, and both refuse another
digest; a Python replay inside a capture that would run it is refused.

**Plans** (`tilefish.fusion(epilogues="library")`, Q3: per graph). The rewrite is made on the IR before the regions are formed. A `cublaslt.gemm` of a
16-bit result whose `y` is read only by `y.to(f32)`, followed by `+ c.to(f32)` (`[M, N]`), `+ bias.to(f32)` (`[N]`) and `tilefish.where(h <= 0.0, 0.0,
h)`, each optional but one and each read by the next alone, then `.to()` the GEMM's type, with `c` and the bias made before the GEMM, becomes one call
of a `cublaslt.matmul` (`<gemm>__<epilogue>[_c]`, the GEMM's options, `contract="measured-exact"`, its canary on). A `cublaslt.gemm_bias` followed by the
ReLU becomes `relu_bias`. That is contract (iii) measured against (ii) (`R_acc` of the matched values): the verification of every pin is the check of
the rewrite against `U(R_acc(P))` computed, as 3.1.1 says, with the f32 form of the rewrite's own pin, because the formula is the region's arithmetic on
the accumulator, op for op; a pin that is not measured-exact refuses the build (`tilefish.barrier(y)` keeps the GEMM unfused). The pin is chosen for
the epilogue, not taken from the GEMM it replaces: its accumulator is the GEMM's where the two pins compute the same sum (X1: the same pin on 14 of 16
shapes and the same bytes on the others; under 13.1.0 the plain GEMM's pin splits `k` at `M = 32` where an epilogue's may not), which no build
checks yet. Where it is, a ReLU alone (of a `cublaslt.gemm`, or of a `cublaslt.gemm_bias` with its bias) writes the bytes of `U(P)`, because ReLU
commutes with the rounding (X1): measured on the test's shapes, not promised. Anything else is left as it is: `(y + bias) + c`
(another rounding in f32), an f32 bias, a value read twice, returned or barriered, a ReLU written as `where(h > 0.0, h, 0.0)` (it maps a NaN to 0, which
`torch.relu` does not). The fusion report gains `library_epilogues`: the rewrites and the reasons of what was not rewritten; in a graph planned
`"exact"`, what the mode would do. `mode="reference"` interprets the rewritten program (the formula on an f32 accumulator). Measured (the test above, both
plans of `python/tests/test_gpu_matmul.py`): `jit`, `replay` and a capture of a rewritten plan agree byte for byte with the formula on the pin's
accumulator, and differ from the exact plan on 14 to 29% of the elements but for ReLU alone (0).

**The advisor.** A GEMM step whose consumer region the report lists as matched is the blocker `epilogue-library-cublaslt`, of phase `E2`: "declare
tilefish.fusion(epilogues='library')", with the epilogue and the contract, ranked by the saving the advisor estimates or measures (`profile_plan`) like
every candidate. One that is not matched stays `epilogue-library` (future work), with the reason.

**Speed** (sm_103, B300 MIG slice, cuBLASLt 12.8.4; medians of CUDA graphs of 20 replays of each plan, microseconds): the exact plan (the pinned GEMM
and a generated region) against the rewritten one (one `cublaslt.matmul`), the shapes of X3 and X4:

| shape `M x N x K` | bias | bias + ReLU | residual | residual + bias |
|---|---|---|---|---|
| 640 x 4096 x 1024, bf16 | 24.8 → 20.6 (-17%) | 24.8 → 22.2 (-10%) | 24.9 → 22.6 (-9%) | 26.8 → 24.7 (-8%) |
| 640 x 1024 x 4096, bf16 | 18.5 → 16.5 (-11%) | 18.5 → 16.6 (-11%) | 18.6 → 16.6 (-11%) | 18.6 → 18.5 (-0%) |
| 2048 x 3072 x 1024, bf16 | 54.5 → 41.1 (-25%) | 55.3 → 43.1 (-22%) | 61.8 → 46.5 (-25%) | 65.8 → 49.3 (-25%) |
| 128 x 3072 x 1024, bf16 | 8.3 → 6.3 (-24%) | 8.3 → 6.3 (-24%) | 8.3 → 6.3 (-24%) | 8.3 → 6.3 (-24%) |
| 8192 x 3072 x 768, f16 | 192.7 → 124.7 (-35%) | 192.5 → 135.7 (-30%) | 208.0 → 136.3 (-34%) | 215.9 → 147.0 (-32%) |

No case is slower; the residual add with a bias at 640 x 1024 x 4096 saves nothing (its pin is another algorithm than the plain GEMM's).

**Left open.**

* GELU and `GELU_BIAS` (binary-defined, with a declared tolerance), and the auxiliary outputs (`RELU_AUX`, `GELU_AUX`): not offered yet.
* `algos=` for `cublaslt.matmul` (a port of a stack that fuses through cuBLASLt imports its pins): refused at the declaration for now.
* A rewrite whose pin is not measured-exact at one bucket refuses the build: a fallback per bucket to route (a) (or to `U(R_acc(P))`) is not made.
* Whether a rewrite's pin computes the sum of the GEMM's pin is not recorded: a build could compare the two accumulators on the verification's
  operands (or pin the rewrite with the GEMM's configuration, where cuBLASLt takes it with the epilogue), and say when the plan keeps `U(P)`'s GEMM.
* `beta * C` written in place (`D == C`, an `alias`) and f32 results; `R_acc` of a library call (under `"accumulator"` a cuBLASLt GEMM is left as
  it is). The scopes of `tilefish.fused(epilogues=...)` apply to the rewrite since the merge with E1 ("E1 and E2 together").
* sm_90 (X7).
* **sm_120, measured at the 0.0.12 gate** (RTX 5090, 170 SMs; `tests/test_gpu_matmul.py` and the matrix above): cuBLASLt 12.8.4 and 13.1.0 alike
  answer every epilogue but `none` with algorithm 21 (every candidate the heuristic returned, for the six problems asked). Of the 392 pins per
  version 140 are measured-exact (`bias`, `none` with `c`, half of `bias` with `c`); every ReLU (224) is binary-defined, its ReLU mapping a NaN to 0
  (every finite value is the formula's), and 28 of `bias` with `c` add the two in neither order (up to 15298 ULP). So on sm_120
  `epilogues="library"` refuses a ReLU, and those `bias` with `c` problems, at export and at the first jit run, as its declaration asks: no plan
  writes other bytes, and the gate's plans there take a bias. f16 split-K pins with the opt-in are taken and verified like any other (all
  binary-defined); without it no epilogue splits `k`. Two ways forward: a ReLU written `tilefish.where(h > 0.0, h, 0.0)` (a NaN to 0) is algorithm
  21's formula, which a matcher of that form could verify measured-exact; or a binary-defined contract with a declared tolerance, as for GELU.

## 8. Questions (resolved 2026-11-16)

The questions as the review saw them, each with the default it recommended. All were decided on 2026-11-16 with these defaults, and Q12's order
as the review stated it ("Decisions", above).

* **Q1. The three contracts.** Should (i) exact, (ii) accumulator and (iii) library, as defined in 2.3, be the contracts that tilefish offers for
  fusion into a call? And does RFC 0001's principle "fusion never changes bytes by itself; a declared mode does" extend to them? That means `R_acc`
  is defined on the IR, so a failed fusion still runs `U(R_acc(P))`. **Recommended: yes.**
* **Q2. Defaults.** **Recommended: `epilogues="exact"` on by default.** It cannot change a byte, and every export verifies it, as with D1 and D6 of
  RFC 0001. `"accumulator"` and `"library"` apply only when declared, per graph or per `tilefish.fused` scope.
* **Q3. Where the accumulator mode is declared.** Per graph and per scope, as RFC 0001's knobs are, or also per call
  (`gemm(x, w, epilogue="acc")`)? **Recommended: per graph and per scope only.** A knob at the call site would put numerics in two places.
* **Q4. A generated GEMM that must equal the library pin it replaces (i-d).** X3 and X3b found it equal on sm_103 wherever the pin does not split
  `k`. **Recommended: yes, as an explicit opt-in in E3** (`equal="cublaslt"`). Export verifies it at every bucket on seeded random and special-value
  operands. At a bucket where it fails (a split-K pin), the library GEMM stays, with route (a). This makes fused epilogues exact against an existing
  stack. It is a measured contract, not a proven one, and it is re-verified at every export and on every library version.
* **Q5. When a generated GEMM is good enough.** **Recommended: E3 is accepted per architecture when the X8 kernel is within 1.10 times the pinned
  cuBLASLt GEMM on a fixed suite of the shapes the ported models use.** The advisor's break-even says where a slower one still pays. Until then,
  E1 and E2 come first, and E3 stays a research item. The straightforward Triton kernel is at 1.22 to 1.44 times.
* **Q6. Which cuBLASLt epilogues first.** **Recommended: bias, ReLU and `beta * C`, measured-exact against (ii) and verified per pin; then GELU as
  binary-defined**, behind `epilogues="library"`, with the canary on by default for epilogue entries. GELU's bytes may change with a cuBLASLt
  version, like any pin's. Where the f32 form of the epilogue's own algorithm cannot be encoded (the GELU algorithms), use the plain pin's f32
  output as the reference, and require that the auxiliary output equals `ρ(acc + b)` (X1 found it always does).
* **Q7. The schema of library epilogues.** **Recommended: a new impl, `cublaslt.matmul`**, with an `epilogue` field (more values), `aux`, `beta` and
  the optional `canary`. `cublaslt.gemm` and `cublaslt.gemm_bias` stay exactly as they are.
* **Q8. Names.** **Recommended:**
  * `tilefish.fusion(epilogues="exact" | "accumulator" | "library")`;
  * `tilefish.hook(name, value="acc" | "out", out=...)`, and `tilefish.hook(name, operand=...)` for prologues;
  * `impl="tilefish.gemm"` with `equal="cublaslt"`;
  * the fusion report's `contract` values `exact`, `accumulator`, `measured-exact`, `binary-defined`.
* **Q9. Gated activations.** Two accumulators per program (one tile of the gate and one of the up projection), or the weight's rows interleaved so
  that one tile holds both? **Recommended: two accumulators.** There is no weight transform, and the bytes of each accumulator are the plain GEMM's.
* **Q10. Norms over rows longer than a tile.** **Recommended: whole-row tiles first** (`N <= BN`: per-head norms, small hidden sizes). The split over
  tiles, with a declared tile order, comes in E4, after RFC 0001 phase 2 makes ordered sums fast.
* **Q11. cuDNN fused graphs.** X5 found them exact against the unfused composition for bias and ReLU with a 16-bit intermediate, but only with
  cuDNN 9.13 and NVRTC 13 on sm_103. tilefish pins 9.10.2 today. **Recommended: E6 after E2 and X13.** It needs speed measured, a cuDNN 9.13
  requirement for fused entries on sm_103, and the graph builder in both hosts.
* **Q12. Order of work.** **Recommended: E1 (hooks on externs) and E2 (cuBLASLt epilogues) first, in parallel.** E3 (generated GEMMs) comes after X8
  shows that a Gluon GEMM closes the gap X3 measured. E4 and E5 come after the RFC 0001 phases they need. The advisor's change (section 5) ships with
  E1.

## 9. Prior art

| system | epilogue fusion | numerics |
|---|---|---|
| cuBLASLt | fixed epilogues (bias, ReLU, GELU, auxiliary outputs, `beta * C`, alpha vectors, FP8 scales and `amax`) chosen per call | the library's; one conversion at the end |
| CUTLASS | epilogue visitor trees: arbitrary elementwise and row- or column-broadcast trees compiled into the GEMM | the user's C++; the accumulator is visible |
| cuDNN graphs | a convolution or matmul plus pointwise and reduction nodes, fused at run time | the compute type of the graph, and a data type per virtual tensor |
| XLA | a GEMM rewriter folds bias, activations and `beta * C` into cuBLASLt epilogues; Triton GEMM fusions handle the rest | no byte contract against the unfused program |
| torch.compile / Inductor | Triton GEMM templates with fused epilogues and prologues, chosen by autotuning against the library | no byte contract; tolerance tests against eager |
| tilefish (this RFC) | hooks in its own kernels, library epilogues where they exist, prologues | three named contracts: exact against the unfused composition, exact against a declared accumulator rewrite, or defined by the library and pinned |

## Appendix A: a worked example

A GEMM and its bias + GELU, planned for `M` in `{128, 640}`, with `F = 4096` and `H = 1024`:

```python
@tilefish.library("up", impl="cublaslt.gemm", acc=f32)            # or impl="tilefish.gemm", or an extern with hooks
def up(a: Tensor[M, H, bf16], b: Tensor[F, H, bf16]) -> Tensor[M, F, bf16]: ...

def gelu_tanh(h):                                                   # a plain helper traces inline
    return 0.5 * h * (1.0 + tilefish.tanh(0.7978845608 * (h + 0.044715 * h * h * h)))

@tilefish.graph(fusion=tilefish.fusion(epilogues=MODE))
def mlp_up(x: Tensor[M, H, bf16], w: Tensor[F, H, bf16], bias: Tensor[F, bf16]):
    y = up(x, w)
    return gelu_tanh(y.to(f32) + bias.to(f32)).to(bf16)
```

| `MODE`, implementation of `up` | steps per bucket | what the output equals |
|---|---|---|
| `"exact"`, `cublaslt.gemm` | `up` (library) writes `y` in bf16; `fused_<hash>` reads `y` and `bias` and writes the output | `U(P)`: the stack's bytes |
| `"exact"`, an extern with an `out` hook | one step: the extern with the region in its hook, after its rounding | `U(P)` with the extern as the GEMM, by construction, verified |
| `"exact"`, `tilefish.gemm` with `equal="cublaslt"` | one step at every bucket where the generated GEMM equals the pin; route (a) elsewhere | `U(P)` with the cuBLASLt pin as the GEMM: the stack's bytes, measured per bucket |
| `"accumulator"`, an extern with an `acc` hook, or `tilefish.gemm` | one step: the kernel with the region in its hook, on the accumulator | `U(R_acc(P))`: the kernel writing f32, then the region; one rounding fewer than `U(P)` |
| `"accumulator"`, `cublaslt.gemm` | `up` writes f32; `fused_<hash>` reads it (the report says no route took the region) | `U(R_acc(P))`, slower than `U(P)` |
| `"library"`, `cublaslt.gemm` | one step: `cublaslt.matmul` with `GELU_BIAS`, pinned with its canary | binary-defined: within the measured distance of `U(R_acc(P))` (X1, X2) |

The library row needs the tanh form of GELU, because cuBLASLt has no erf GELU. Whether the rewrite should recognise a hand-written tanh GELU as
`GELU_BIAS` at all is part of Q6. The epilogue is binary-defined, so the recognition is a declared equivalence, not an exact one.

`R_acc(P)` here is the program in which `y.to(f32)` is replaced by the f32 result of `up`: `gelu_tanh(up_f32(x, w) + bias.to(f32)).to(bf16)`. The
fusion report entry of the accumulator case would read:

```json
{"op": "up+fused_<hash>", "route": "hook", "value": "acc", "contract": "accumulator", "call": "up",
 "rounding_points_removed": [{"value": "y", "dtype": "bf16", "reason": "R_acc: widened by the epilogue"}],
 "nodes": [{"index": 1, "op": "cast"}, {"index": 2, "op": "cast"}, {"index": 3, "op": "add"}, "...", {"index": 12, "op": "cast"}],
 "verified": {"against": "U(R_acc(P))", "buckets": ["M=128", "M=640"], "equal": true}}
```
