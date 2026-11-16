# RFC 0001: Plan fusion

* **Status:** reviewed; the decisions of 2026-11-15 are recorded in "Decisions" below. Phase 0 is measured, phase 1 is implemented
  ("Phase 1" below), and phase 2's row regions are implemented as `tilefish.fusion(version=2)` ("Phase 2" below); the rest is design.
* **Date:** 2026-11-15.
* **Scope:** the Python planner (`tilefish.plan`, `tilefish.runtime.export`), the lowering (`tilefish.lower`), the tracer of graph bodies and
  the `reference` mode. The hosts (the Python plan runner and the Rust crate) and both schemas are unchanged.
* **Baseline:** the surface of the next release: 0.0.6 / crate 0.0.4 plus the unreleased changes: carried buffers, memset steps, per-bucket configs
  (`by_bucket`), product and persistent grids, ordered sums (`tilefish.sum(order=...)`), the one-instruction primitives (`tilefish.fma`, `rcp`,
  `neg`, `sin_approx`), CUDA's `erff`, float-to-integer casts and static slices. Note numbers below are those of the numerics contract in
  `python/src/tilefish/lower.py` with those changes (notes 1 to 19).

## Summary

Today a `@tilefish.graph` body may only call declared operators and take views; `y * scale + bias` in a graph body is refused, so every
elementwise step of a model needs a hand-written kernel or a `@tilefish.kernel` of its own. Even the graph in the README's first example
(`silu(g) * u` between two GEMMs) does not plan.

This RFC proposes **plan fusion**: the planner groups the arithmetic traced in a graph body into regions and lowers each region, with the
templates and the numerics contract that `@tilefish.kernel` already has, into a *generated op*, as `tilefish.contiguous` and `im2col` are
generated today. A fused region is an ordinary `kernel` step of the plan: no plan or manifest schema change and no host change.

The numerics rule is the point of the design: **fusion never changes bytes by itself.** Every graph has a declared numerics mode; the default,
and every mode called *exact*, makes the fused plan byte-identical to the *unfused composition*, the plan in which every arithmetic op is a
kernel of its own and every intermediate is stored in its declared dtype, and `export` verifies it byte for byte by default. Version 1 has
exact modes only; modes that would not be exact (compiler contraction, Triton tree reductions inside a region, library epilogues) are future
work (section 11). Section 5 states the claim, the obligations it rests on, the hazards found while designing it (one of them a compiler fold
that can change NaN payloads), and the tests that prove it.

## Contents

Decisions (2026-11-15)
Phase 0 results (2026-11-15, sm_103)
Phase 1 (2026-11-15)
Phase 2 (2026-11-16): row regions, fusion version 2
0. Background: what exists
1. Use cases and non-goals
2. Surface syntax and capture
3. The fusion algorithm
4. Code generation
5. The numerics contract and its modes
6. Interaction with plans
7. Epilogue and prologue fusion into library ops and externs
8. Schema impact and migration
9. Staged implementation plan, test plan and open questions
10. Prior art
11. Future work: modes that are not exact, and other extensions

Appendix A: a worked example.

## Decisions (2026-11-15)

The draft was reviewed on 2026-11-15. The questions of section 9.4 that are settled:

* **D1 (Q1): on by default.** A graph whose body has arithmetic fuses it, in the exact default mode, without a declaration. No existing plan
  changes: until now such a graph was refused.
* **D2 (Q2): an unordered reduction runs alone.** A reduction with no `order=` is a kernel of its own, unfused, its operand stored (as in the
  unfused composition); it joins a region once the body gives it an `order=`.
* **D3 (Q3): no contraction by default.** A graph contracts nothing; one rounding of `a * b + c` is asked for with `tilefish.fma` or with the
  declared rewrite `fp_contract="mul-add"` (an exact mode).
* **D4 (Q4): scalars round to the tensor's dtype**, as in a kernel body; torch's rule is the opt-in exact mode `intermediates="opmath"`.
* **D5 (Q5): exact modes only in version 1.** Compiler contraction, Triton tree reductions inside regions and rewrites into library epilogues
  are future work (section 11).
* **D6 (Q6): export verifies.** `export` runs the fused plan and the unfused composition byte for byte on seeded and special-value inputs at
  every bucket and refuses a difference; `verify=False` opts out.
* **D7 (Q7): frozen grouping rules.** `tilefish.fusion(version=N)` fixes the grouping rules, limits and rewrites; version 1 is the first, and a
  later tilefish keeps every earlier version. A plan made with another version is refused as stale with a message that names the version.

D3 and D4 were the recommended defaults and were not objected to. Q4b and Q8 to Q11 stay open; section 9.4 gives the recommended default of
each.

## Phase 0 results (2026-11-15, sm_103)

Measured on a B300 MIG slice (sm_103) with torch 2.9.1 (CUDA 12.8), Triton 3.5.1, ptxas 12.9 and cuBLASLt 12.8.4. None of the results changes
a decision; E5 confirms hazard H1, and phase 1 closes it.

* **E1, tree reductions in context.** A `tl.sum` row reduction of `r * r` alone (the operand stored by an elementwise kernel) against the same
  reduction fused after loads of a bf16 `x` and an f32 residual, with the same warp count: at 96 columns (1 warp) no row differs; at 1000,
  1024 and 4096 columns (4, 4 and 16 warps) **357 to 399 of 1024 rows differ** (three seeds). The ordered sum (`tree(128)`) differs in no row.
  This is the premise of D2: a Triton tree reduction's bytes depend on the rest of its kernel.
* **E2, `tl.max`.** At 1, 2, 4, 8 and 16 warps, and with every row's elements permuted, `max` over rows of `+0.0` / `-0.0` mixes, zeros with
  negatives, and NaNs with random payloads among numbers (2048 rows each of 100, 1000 and 4096 columns): **no row differs**. A row with a
  `+0.0` gives `+0.0`, and NaNs are dropped. So `max` is order-free on these inputs and may join row regions (phase 2) without a new lowering;
  the phase 2 tests keep checking it.
* **E3, speed** (plans captured in CUDA graphs; microseconds per plan run; the fused kernel is what a region generates, `fp_fusion=False`):

  | case | rows | fused | unfused (one kernel per op) | notes |
  |---|---|---|---|---|
  | bias + SiLU, `[M, 3072]` bf16 | 32 / 4096 | 1.45 / 58.6 | 8.61 / 208.5 | fused and unfused byte-identical |
  | SwiGLU on contiguous halves, `[M, 3072]` | 32 / 4096 | 1.34 / 51.5 | 8.74 / 278.2 | fused and unfused byte-identical |
  | residual add + rmsnorm, 1024 columns | 32 / 4096 | ordered `tree(128)` 2.05 / 80.2; tree 1.23 / 28.6 | D2 (three kernels) 6.07 / 61.4 | |
  | softmax, 1024 columns | 32 / 4096 | ordered 2.05 / 54.5; tree 1.44 / 18.0 | D2 (four steps) 6.04 / 44.8 | |

  Elementwise fusion is 3.6 to 6.5 times faster. D2 costs about 5 times the tree form at 32 rows and 2 to 2.5 times at 4096; an ordered sum
  recovers most of it at 32 rows, but the current ordered-sum lowering is slower than D2 at 4096 rows, so its speed is phase 2 work.
* **E4, the cuBLASLt bias epilogue.** `(gemm_f32(a, b) + bias.to(f32)).to(bf16)` against `gemm_bias(a, b, bias)` with a bf16 output, both
  pinned under the torch posture and under the default posture, `M` of 32, 64, 128 and 1024 with `N x K` of 1536 x 512, 3072 x 1024 and
  1024 x 3072, special values in the bias: **no element differs** in any of the 24 comparisons. Measured, not proven: it informs section 11.2.
* **E5, H1.** Inside one kernel, `x.to(f32).to(bf16)` of every bf16 bit pattern **keeps the payload of all 254 NaNs**, where the two kernels of
  the unfused composition give `0x7FFF` for each; the same holds for f16 (2046 of 2046 NaNs). A `where` between two widened values, narrowed
  after it, differs on 506 outputs, all NaNs: the fold reaches through a select too. Every non-NaN output is the same. Phase 1 therefore emits
  **every** narrowing float cast of a region (not only one directly after a widening cast) as inline PTX, which no fold can see.

## Phase 1 (2026-11-15)

Implemented on the integration branch of the next release, as `tilefish._fusion` (the pass), note 20 of `tilefish.lower` (the code of a region),
`tilefish.fusion`, `tilefish.barrier`, `tilefish.fusion_report` and `tilefish.verify_fusion`, the export-time verification and
`<graph>.fusion.json`:

* **As designed:** D1 to D7; elementwise regions (whatever the lowering takes: a region may lower with the row template when an operand is a
  per-row or per-column value); a reduction as a region of its own; free ops (`load`, `iota`) copied into every region that reads them; masks
  across regions as `u8`; `intermediates="opmath"` and `fp_contract="mul-add"` (both brought forward from phase 3); dead ops not computed; a
  region with no input of its output's shape (`iota(x, 0) < load(n)` alone) joined to its consumer, so that every region is a kernel that can
  be launched; the fusion report; the Rust gate runs the use cases as plans (`fused_*`).
* **Found while testing, and closed:** H8 and H9 (section 5.3). With H1 they make the code of a region *context-free*: the lowering option
  `context_free=True` of note 20, which every region kernel and every kernel of the unfused composition is lowered with.
* **Strided inputs** in the elementwise template, as rule 5 of 3.5 says: a view that is not dense is a `Strided[...]` parameter of the region's
  kernel with a `StrideOf` per axis, read at `sum(coordinate_k * stride_k)`; SwiGLU's halves are one kernel step over two views of the GEMM's
  output. A region the row template lowers still copies the view with `tilefish.contiguous` first.
* **Not yet:** strided inputs in the row template; `tilefish.fused(...)` scopes; the `intermediates="f32"` mode.
* **Since then** (the decision of 2026-11-15, "fix the implementation and keep the promises"): H8 and H9 apply to every kernel, and are closed
  there too. A `@tilefish.kernel(fp_fusion=False)` has the code of a region (every float `+ - *` one scalar `.rn` instruction, every narrowing
  cast inline PTX; note 2), every kernel's negation gives the canonical NaN explicitly (note 16), and a narrowing cast of a widened value is
  inline PTX in every kernel (note 1). Every binary built with `enable_fp_fusion=False`, a region's and a hand-written extern's included, is
  read back by the contraction check (`tilefish._fpcontract`), which refuses one whose SASS fused a multiply and an add.
* **Tests:** `tests/test_fusion.py` (no GPU: grouping, legality, names, modes, report, refusals, the same plans in another process) and
  `tests/test_gpu_fusion.py` (sm_103; arch-agnostic): the inline casts are `.to()` on every f32 and every bf16 and f16 pattern; the use cases
  fused against unfused at every bucket on special values, and against the reference where it is exact; chains of ops on every bf16 and f16
  pattern; binary ops on `2**22` bit patterns; 120 random graphs of up to 24 ops, each also through `verify_fusion`; `opmath` in IEEE math
  against torch's eager ops on every bit pattern, and two-tensor 16-bit `+ - *` against torch's on every pair (`2**32` per op and dtype); the
  export check refuses a contracting region, writes the report, and the plan replays and captures.

## Phase 2 (2026-11-16): row regions, fusion version 2

Implemented as `tilefish.fusion(version=2)`, now the default (D1: fusion on by default; D7: version 1 kept). What changed is the grouping rules
(so the version) and, for the regions of version 2 only, the code of the lowering (revision 2, note 21 of `tilefish.lower`); the exact contract,
the export-time verification, the schemas and the hosts are those of phase 1.

* **Row regions.** An ordered `sum` or `mean` (`mean(order=)` is new: the ordered sum times `f32(1 / D)`, one `mul.rn`), the order-free `max`
  (E2), `argmax`, `argmin` and `gather_rows` (a copy) join the ops around them; the dry run of the lowering decides, as in phase 1, and a union
  that includes a reduction is lowered by the row template. A `sum` or `mean` without `order=` is still a region of its own, its operand stored
  (D2). A region may write several values (an rmsnorm and its residual sum; a quantised row and its scale). An ordered sum with `fma=True` and the
  multiply whose operands each of its steps takes are one unit in every grouping, the unfused composition's too (version 1 raised a `KeyError`
  on one: its sum ran alone without its product).
* **The use cases of 1.1** as graphs (`tests/fusion_graphs.py`, each with its unfused twin, run by the Rust and C++ gates as `fused_*` plans):
  a residual add and an rmsnorm with the residual stream a second output (one kernel; the unfused composition has ten), the same with an fma
  sum, a layer norm with a gelu and per-channel tail, softmax with a mask from a device scalar, a row-wise fp8 quantisation, a SwiGLU of two
  column blocks and a norm, a gather and an argmax.
* **Version 1 is frozen** with its lowering: its regions keep revision 1, and every use case declared `fusion(version=1)` plans as 0.0.9 planned
  it (plan digests checked in `tests/test_fusion.py`). A plan made with another version is refused as stale with a message that names it
  (`_fusion.other_version_of_plan`, tried only when a plan is stale).
* **Why the ordered sum was slow (E3), and revision 2.** Revision 1 takes each step of a lane's chain with `tl.gather(x, k, 0)` of the whole
  row; Triton lowers a gather whose source spans warps through shared memory, so every step stores the row, waits at a barrier and loads
  (`lanes * vector` columns a step: 8 steps for `tree(128, vector=8)` at 1024 columns, 128 for `tree(32)` at 4096, where it took 2.5 ms for
  4096 rows), and the template's warp rule (a warp per 256 columns) gave 16 warps to 128 lanes at 4096 columns. Revision 2 reshapes the row to
  `[lanes, K]` (lane `j`'s columns in its order along the last axis, the columns past `D` made `-0.0`, which `add.rn` leaves every chain value as
  it is) and takes each chain by halving splits, register moves where each thread holds its lane's columns, which the layout of the loads gives.
  A first version that also took the tree over the lanes by splits made Triton's layout propagation give every value of the row to every
  thread (926 spills, 4 ms): the tree goes through `tl.gather` of `[lanes]` instead (warp shuffles within a warp), and lane 0 comes out of a
  `tl.reduce` with a select combine. The warps: one up to `BLOCK_D = 2048`, then one per 1024 columns (per 512 for 512 lanes or more), within 2%
  of the best of 1 to 8 warps over the measured grid. Checked against the CUDA kernels of each order at every length from 1 to 4096 columns and at
  every warp count; 1.6 to 37 times faster than revision 1.
* **H10, found here: a literal `-0.0` is `+0.0` in Triton 3.5.1** (`tl.where(m, x, -0.0)`, `tl.full(..., -0.0)`, measured with and without a
  `+0.0` in the kernel). Fusion stays exact (the fused and unfused kernels write the same literal), but a body's `where(x > 0, x, -0.0)` is not
  what it says in revision 1. Revision 2 writes a `-0.0` scalar, and the padding of its ordered sums, by its bits (an `i32` bitcast).
* **Also in revision 2:** views of rank 1 and 2 read where they are by the row template (rule 5; version 1's row regions copy them first), an
  fp8 cast inline (`cvt.rn.satfinite.e4m3x2.f32`, `.to()`'s bits on every input). A hand-written kernel may ask for revision 2
  (`@tilefish.kernel(lowering=2)`); by default it keeps revision 1 and its source.
* **The advisor** (`docs/advise.md`) and `boundaries()` follow the graph's version: in version 2 an ordered reduction, `max`, `argmax` or a gather
  is not reported, and a row region neither blocks its neighbours nor copies a view.
* **Speed** (B300 MIG slice, sm_103, 30 SMs; microseconds per plan run captured in a CUDA graph; v2: the region, one kernel; tree: the same body
  without `order=`, D2's three to five kernels with Triton's tree reductions, other bytes; v1: version 1, the reductions alone in revision 1;
  unfused: one kernel per op; v1 and unfused write v2's bytes, checked in every cell). Over the whole grid, rows 32, 128, 512, 1024 and 4096 and
  512, 1024, 2048 and 4096 columns, v2 is the fastest in every cell: 2.6 to 8 times faster than the tree composition.

  | pattern | rows | D = 512: v2 / tree / v1 / unfused | D = 1024 | D = 2048 | D = 4096 |
  |---|---|---|---|---|---|
  | add + rmsnorm, residual returned (`mean`, `tree(128, vector=8)`) | 32 | 1.3 / 5.3 / 6.4 / 10.8 | 1.7 / 9.9 / 10.9 / 16.7 | 1.6 / 9.5 / 12.9 / 16.0 | 2.1 / 10.8 / 22.1 / 18.1 |
  | | 4096 | 8.1 / 37.1 / 66.3 / 62.6 | 15.6 / 64.7 / 120.9 / 102.7 | 39.8 / 125.8 / 300.9 / 228.3 | 81.4 / 246.8 / 887.8 / 462.9 |
  | layer norm (two means, `tree(256)`) | 32 | 2.7 / 11.4 / 11.2 / 18.1 | 3.0 / 12.2 / 13.6 / 18.3 | 3.5 / 12.3 / 16.1 / 18.3 | 3.0 / 13.5 / 24.3 / 20.8 |
  | | 4096 | 31.2 / 49.5 / 58.0 / 75.0 | 40.2 / 84.8 / 135.1 / 124.7 | 56.2 / 174.0 / 369.4 / 290.4 | 131.9 / 347.2 / 903.8 / 610.9 |
  | softmax (`max`, `exp`, sum `tree(32)`) | 32 | 1.2 / 8.6 / 11.1 / 11.2 | 1.4 / 9.1 / 13.5 / 11.8 | 1.8 / 9.4 / 18.1 / 11.8 | 2.4 / 10.9 / 35.6 / 14.0 |
  | | 4096 | 9.0 / 28.6 / 71.5 / 62.5 | 15.0 / 43.6 / 175.2 / 95.9 | 33.5 / 95.7 / 545.5 / 227.8 | 95.6 / 193.6 / 1769.6 / 477.4 |
  | fp8 quantisation (`max`; no sum: no tree) | 32 | 1.1 / - / 8.9 / 14.2 | 1.1 / - / 8.9 / 13.8 | 1.3 / - / 9.4 / 14.0 | 1.7 / - / 10.5 / 16.0 |
  | | 4096 | 8.5 / - / 32.1 / 58.6 | 15.9 / - / 55.6 / 97.8 | 31.6 / - / 107.0 / 236.1 | 69.4 / - / 204.9 / 483.9 |

  At 4096 rows of 4096 columns the add + rmsnorm moves 134 MB in 81 us, the bandwidth of the slice (1.6 TB/s).
* **Tests:** `tests/test_fusion.py` (version 1 frozen, row regions, the fma unit, strided row inputs, the region's source as a hand-written
  kernel's lowered by revision 2, the warp rule, the stale message); `tests/test_gpu_fusion_rows.py` (each row pattern fused against unfused at
  every bucket on seeded inputs, on rows of special values and on every bf16 pattern; every warp count; 80 random graphs of row ops, a quarter also
  through `verify_fusion`; the inline fp8 casts on every f32 and 16-bit pattern; `-0.0`; export, replay and capture); `tests/test_gpu_reduction_order.py`
  (revision 2 against the CUDA kernels of each order, at every warp count); the use cases in `tests/test_gpu_fusion.py`, the Rust and C++ gates.
  Every GPU test is arch-agnostic; on sm_120 they are a later gate's, and so are the speed rule's measurements there.
* **Not yet** (phase 2 work left open): a store into an `InOut` or carried tensor made by a region (rule 4: it is still a copy step), strided
  outputs (rule 7), dense reshapes inside a region (rule 6), configurations per bucket other than the warp count, `tilefish.fused(name=...)`
  scopes, `intermediates="f32"`, revision 2's split form for the sequential order (one lane: revision 1's helper), and the reference against
  the RFC's acceptance (Q11): the ported models' norms and activations written in the graph body.

## 0. Background: what exists

* **Graph bodies already trace arithmetic.** A graph body runs over the same `Value` proxies as a kernel body, so `x + y`, a cast or a
  reduction in a graph becomes a typed node of the graph's IR. The planner (`_plan._walk`) refuses such a node ("arithmetic traced in the graph
  body ... put it in a `@tilefish.kernel`"), and the graph `reference` mode refuses it too.
* **The lowering is a fuser already.** `tilefish.lower` turns the SSA IR of a `@tilefish.kernel` into one Triton function, by one of two
  templates: *elementwise* (one lane per element, broadcasts along any axis, note 12) and *row* (one program per row, reductions over the last
  axis, notes 6, 8, 17). Every op it emits has a numerics contract measured on the PTX Triton 3.5.1 and ptxas 12.9 produce (notes 1 to 19),
  much of it on every one of the `2**32` f32 bit patterns.
* **Generated ops exist.** `tilefish.contiguous`, `im2col`, `im2col1d` and `col2im1d` are extern ops whose Triton source tilefish writes
  (`_copyops.py`): one op per shape class, made the first time it is asked for, cached, with a torch reference. A plan launches one where the
  graph needs it (a non-dense copy is a `kernel` step of `tilefish.contiguous`), "so the document has nothing new".
* **Identity is mechanical.** A generated kernel's spec carries `source_sha256`; the TTIR the frontend builds is hashed (`source_digest`); the
  compile options (`num_warps`, `num_stages`, `enable_fp_fusion`) are in the declaration digest and the manifest entry; the plan digest covers
  the manifest digest. Exports are reproducible across machines (sources are content-addressed, debug paths host-independent).
* **What a reduction's bytes depend on is known.** Triton's `tl.sum` and `tl.max` reduce in a tree that `num_warps` shapes (669 to 878 of
  1024 rows differ between warp counts); per-element ops, casts, `argmax` / `argmin` and ordered sums do not depend on it.
* **FMA contraction is a declared option.** `enable_fp_fusion=False` makes Triton emit `mul.rn` / `add.rn` and pass `-fmad false` to
  ptxas; `tilefish.fma` is one `fma.rn.f32` whatever that option says.

## 1. Use cases and non-goals

### 1.1 What real inference graphs need

The chains below are what remains between the library calls and the hand-written attention and convolution kernels of a typical decoder,
encoder or codec, once the GEMMs are pinned. Each is memory-bound: run as one kernel per op, it reads and writes its tensors once per op.

| use case | graph-body form (sketch) | template | phase |
|---|---|---|---|
| bias + activation after a GEMM | `silu(h.to(f32) + b.to(f32)).to(bf16)`, `b: [F]` broadcast over rows | elementwise | 1 |
| gated activations (SwiGLU, GeGLU) | `g, u = gemm(x, w).split(F, -1)`; `(silu(g.to(f32)) * u.to(f32)).to(bf16)`: `g`, `u` are column blocks of one buffer | elementwise, strided inputs | 1 |
| residual add | `h = x + attn` (and return or store `h`) | elementwise | 1 |
| per-channel affine and periodic activations | `s = sin(a * x)`; `x + s * s / a` with `a: [C, 1]`, `x: [N, C, T]` (f32) | elementwise, broadcasts | 1 |
| masks from a device scalar | `where(iota(s, 1) < load(n), s, float("-inf"))` | elementwise | 1 |
| dequantisation | `w.to(f32) * scale` with a per-row or per-column `scale` | elementwise | 1 |
| norm + residual, residual stream as a side output | `h = x + r`; `var = sum(h32 * h32, -1, order=...) * (1 / D)`; `y = (h32 * rsqrt(var + eps) * w32).to(bf16)`; return `y, h` | row, two outputs | 2 |
| layer norm | two ordered sums (mean, variance) and an affine tail | row | 2 |
| softmax pieces | scale, mask, `max`, `exp(x - m)`, ordered `sum`, divide | row | 2 |
| quantisation with a per-row scale | `amax = max(where(x < 0, -x, x), -1)`; `q = (x / scale).to(e4m3)` (`satfinite`), or an integer type with `rounding=` and `saturate=`; return `q, scale` | row, two outputs | 2 |
| rotary embeddings | half split or pair interleave of the last axis, `cos` / `sin` rows gathered by position | row + data movement | 3 |
| block-scaled quantisation (MX formats) | reductions over 32-element blocks of the last axis, `ue8m0` exponents | segmented row | 4 (open) |
| GEMM epilogues | bias, activation, residual, output cast and scale folded into the GEMM | library or extern epilogue | 4 (section 7) |

### 1.2 Non-goals

* **No algebraic simplification, ever, in an exact mode.** No `x * 1 -> x`, no `x + 0 -> x`, no reassociation, no strength reduction, no
  constant propagation through a value the unfused composition would load from memory (section 5.3). Fusion moves kernel boundaries; it
  does not rewrite arithmetic unless a declared mode says how.
* **No recomputation** (no duplicating a node into two regions) in version 1. A value that two regions need is stored once.
* **No fusion across a call** in phases 1 to 3: a call to a kernel, an extern or a library op is a region boundary. Section 7 is the only
  route into a call.
* **No autotuning that can change bytes.** A configuration is a function of the shapes (and, for exact regions, is a speed choice only).
* **Not a kernel compiler.** GEMMs, attention and convolutions stay `extern` or library ops. Fusion generates memory-bound elementwise and
  row kernels only.
* **No new dynamic-shape machinery.** Buckets are strict as before; a region is specialised per bucket where its template needs it.
* **Block-scaled tensors** stay unplanned (as today); the MX quantisation chain is listed for completeness.

## 2. Surface syntax and capture

### 2.1 Graph bodies

Arithmetic in a graph body is written exactly as in a kernel body:

```python
@tilefish.graph(inputs=("x",))
def mlp(x: Tensor[M, H, bf16], w_gu: Tensor[2 * F, H, bf16], w_d: Tensor[H, F, bf16]):
    g, u = gemm_gu(x, w_gu).split(F, axis=-1)
    a = (tilefish.silu(g.to(f32)) * u.to(f32)).to(bf16)     # fused into one generated kernel
    return gemm_d(a, w_d)
```

The ops, their dtype rules and their broadcasting are the kernel body's (section 2.5), so a body can move between a `@tilefish.kernel` and a
graph unchanged. What the graph adds is a declaration of how the arithmetic is evaluated.

### 2.2 Declaring the numerics: `tilefish.fusion(...)`

```python
@tilefish.graph(fusion=tilefish.fusion())                    # the default: exact, as below
@tilefish.graph(fusion=tilefish.fusion(math="ieee"))         # the default math of the body's ops (an op's own math= wins)
@tilefish.graph(fusion=tilefish.fusion(grouping="none"))     # the unfused composition: one kernel per op (the oracle, section 5.1)
```

| parameter | values (default first) | what it says | phase |
|---|---|---|---|
| `version` | `2`, `1` | the grouping rules, limits and rewrite rules, frozen (D7): `2` makes row regions (phase 2), `1` runs every reduction alone | 1, 2 |
| `grouping` | `"auto"`, `"none"` | regions as section 3 makes them, or one kernel per op (the unfused composition) | 1 |
| `math` | `"fast"`, `"ieee"` | the default math of the body's f32 ops, as `@tilefish.kernel(math=...)` | 1 |
| `intermediates` | `"declared"`, `"opmath"`, `"f32"` | how a 16-bit intermediate is rounded: as its IR dtype says; as torch's eager ops round (D4); widened to f32 between calls | `"declared"`: 1, the others: 3 |
| `fp_contract` | `"none"`, `"mul-add"` | no contraction (D3); a declared rewrite into `tilefish.fma` | `"none"`: 1, `"mul-add"`: 3 |
| `verify` | `True`, `False` | `export` runs the fused plan and the unfused composition on seeded inputs at every bucket and refuses a difference; `False` opts out (D6, section 5.6) | 1 |

Every value is an exact mode (section 5.4). A reduction joins a region only with an `order=`; without one it is a kernel of its own (D2).
Knobs for modes that are not exact are future work (section 11).

**Defaults.** A graph that declares nothing and has arithmetic gets `tilefish.fusion()`. This changes no existing plan: until now such a graph
was refused, so every plan that exists has no arithmetic and fuses nothing (D1).

### 2.3 Controls in the body

* `tilefish.barrier(x)` returns `x` and makes it a region boundary: the value gets a buffer of the plan. A barrier never changes bytes
  (every mode is exact); it exists to tap an intermediate, to keep a buffer for a later stage, and to shape the kernels by
  hand. In a kernel body it is refused.
* `with tilefish.fused(name="norm"):` asks for the ops traced inside to be one region; the plan is refused, with the rule that failed, when
  they cannot be (a reduction without an order, two iteration spaces, a cycle through a call). `name` prefixes the
  generated op's name for profiles and reports. Regions may not nest or straddle a call.
* `tilefish.sum(..., order=...)` exists; this RFC adds `order=` to `tilefish.mean` (`sum` with that order, then `* f32(1 / D)`, as `mean` is
  today) and proposes a total-order `max` if experiment E2 (section 9.2) shows that Triton's `tl.max` is not order-free on signed zeros and
  NaNs.

### 2.4 Tracer changes

Small, and confined to graph traces, so no kernel's IR, source or digest changes:

1. `@tilefish.graph(fusion=...)` stores a `Fusion` value on the `GraphOp` (Python only; it is part of what the plan is derived from).
2. Graph-body nodes get two attributes that kernel bodies never have: `region` (the `tilefish.fused` scope, if any) and `loc` (file and line,
   named as the reproducible-build paths are, for reports and messages). The fusion pass strips both before it builds a region's kernel, so
   neither reaches a generated source.
3. New node `barrier` (graph traces only).
4. `mean(order=)` and, if needed, a total-order `max`: new attributes written only when given, so existing kernels keep their IR.
5. Masks may now cross a region boundary in `grouping="none"` and when a region must be split: such a mask is stored as `u8` (0 or 1) and read
   back as `!= 0`, which is exact (note 7).

The tracer's dtype rules do not change: a 16-bit tensor still needs `.to(f32)` before `exp`, in every mode (whether `intermediates="opmath"`
should accept 16-bit unary math, as torch does, is open question Q4b; recommended: no).

### 2.5 Dtype and broadcast semantics in a graph body

The same as a kernel body. Torch's behaviour is in the last column, because a graph is often a port of a torch model.

| construct | operand dtypes | broadcast | numerics (note) | torch eager differs where |
|---|---|---|---|---|
| `a + b`, `a - b`, `a * b` | one dtype: f32, bf16, f16, i32, i64; **no implicit promotion** (`bf16 + f32` is refused) | numpy's: aligned on the trailing axes, only a static size 1 broadcasts, a dynamic size never | one rounding in that dtype; 16-bit ops native (`add.rn.bf16x2`); contraction per `fp_contract` (2) | torch promotes mixed dtypes; for two 16-bit tensors the bytes agree (torch's f32 op then rounding gives the correctly rounded value; argued in 5.4, tested on every pair in 5.6) |
| python scalar or `const` operand | takes the tensor's dtype and is **rounded to it first** | broadcasts | (3) | torch computes a 16-bit op with a scalar in f32 (`x_bf16 * 0.1` uses `f32(0.1)`); `intermediates="opmath"` gives torch's rule |
| `a / b` | f32 only | numpy's | `div.full.f32` in fast math, a constant divisor a multiply by `f32(1 / c)` (4); `div.rn.f32` in IEEE math (13) | IEEE math is torch's division of tensors |
| `i // c`, `i % c` | integers, positive constant | numpy's | Python's floor semantics, exact (11) | none |
| `==`, `!=`, `<`, ... | one dtype; a scalar rounded to it | numpy's | IEEE comparison, gives a mask (7) | none once the scalar is rounded |
| `&`, `|`, `^`, `~` | masks | numpy's | exact (7) | none |
| `where(m, a, b)` | `a`, `b` one dtype | numpy's | selects bits, NaN payloads and signed zeros kept (7) | none |
| `iota(x, axis)` | gives i32 or i64 | (a coordinate) | exact (12) | none |
| `rsqrt exp sqrt silu gelu sin cos tanh erf` | f32 only | (unary) | per math mode (5, 10, 13, 14, 15) | IEEE math is torch's CUDA op on every f32 for `exp sqrt rsqrt sin tanh silu erf`; `gelu` keeps libdevice's `erf` |
| `fma`, `rcp`, `sin_approx`, `neg` / `-x` | f32 (`neg`: every float) | numpy's | one PTX instruction, never contracted (16) | `rcp` is `torch.reciprocal`; `neg` is CUDA's |
| `x.to(dt)` | any storage dtype; float to integer needs `rounding=` | (unary) | the only rounding points besides the ops; fp8 is `satfinite` (1, 18) | torch's fp8 overflows to NaN or inf; torch's `u8` wraps |
| `sum`, `mean`, `max` | f32 | the last axis of a row | Triton's tree (depends on `num_warps`, 6) or an explicit `order=` (17) | torch's own order |
| `argmax`, `argmin` | numbers | the last axis | exact, order-free (8) | none |
| `gather_rows`, `scatter_rows` | any | rows | raw copies, out-of-range policy declared (9) | none |
| `load(devscalar)` | any | rank 0, broadcasts | a raw read | n/a |
| views (`reshape`, `permute`, `select`, `split`, `x[i]`) | any | n/a | no data movement; a view feeds a region as a strided input (section 3.5) | n/a |

## 3. The fusion algorithm

### 3.1 Pipeline

```
graph IR G  --(numerics rewrite R_mode, section 5.4)-->  declared program P
P  --(grouping, 3.4; legality, 3.5)-->  regions
each region  --(synthetic @tilefish.kernel, 4.1; tilefish.lower)-->  a generated op
P with each region replaced by one call to its generated op  --(the existing _walk)-->  the plan
```

Everything is pure Python and deterministic: no torch, no Triton, no GPU. Python's stale check (a host that has the graph re-derives the plan
and requires the document to be that plan) therefore keeps working, with fusion inside the derivation. The existing walk is unchanged: it never
sees an arithmetic node, only calls.

### 3.2 Node classes

| class | nodes | role |
|---|---|---|
| CALL | calls to `kernel`, `extern`, `library` ops | a boundary (phases 1 to 3) |
| VIEW | `reshape`, `permute`, `select`, `split`, `getitem` | no data; feeds a region as a strided input, or ends one |
| MOVE | `copy`, `store` into an `InOut` or carried parameter, `zeros_like` (memset) | a copy or memset step; a `store` of a region's value becomes that region's write instead (3.5) |
| ARITH | every traced op of section 2.5 | fusible |
| BARRIER | `barrier` | a boundary |

### 3.3 Iteration spaces

Each ARITH node has an iteration space, which says which template can compute it:

* **E(S)**, elementwise over a shape `S`: a per-element op whose result has shape `S`, and an `iota` or cast of that shape. A device-scalar
  `load` (rank 0) joins any region.
* **R(r, D)**, rows: a reduction over the last axis of a value `[..., D]` (the leading axes flattened into `r` rows); and a per-element op on
  `[r, D]`, `[r, 1]` / `[r]` (one value per row) or on a read-only `[D]` input.

E(S) and R(r, D) are compatible when `S` is `[r, D]` (up to flattening the leading axes): an elementwise node can join a row region over the
same rows. A row region needs `D` static or bucketed (every dimension of a plan has buckets, so that holds; one binary per bucket of `D`, as the
row template does now). A broadcast operand of smaller shape is read, never computed, inside a region: a value of shape `[D]` that a region
computes cannot feed an `[r, D]` value of the same region (that would compute it once per row: recomputation); it ends its own region instead.

### 3.4 Grouping

Deterministic greedy producer-consumer merging over the IR order, the same answer on every machine and Python version:

1. Every ARITH node starts as its own region.
2. Visit the ARITH nodes in IR order. For a node `n`, visit its operands in order; for an operand made by an ARITH node `p`, merge
   `region(p)` and `region(n)` if the union is legal (3.5). With `grouping="none"` nothing is merged.
3. `tilefish.fused` scopes are merged first, and are refused (with the rule) if they cannot be one region.
4. A region's **inputs** are the values it reads that it does not make (parameters, call results, views, other regions' outputs), in the order
   of first use; its **outputs** are the values it makes that something outside reads: a call, another region, a VIEW or MOVE node, or the
   graph's return. A value only the region reads is never stored.

Version 1 does no *horizontal* fusion (two regions that share an input but neither feeds the other stay two kernels) and no recomputation.
Both are exact and could come later under a new `version`.

### 3.5 Legality

A union `A ∪ B` is legal when all of these hold; the check that decides is the last one, a dry run of the lowering.

1. **One iteration space.** Every node is E(S) for one `S`, or the union is a row region R(r, D) (3.3).
2. **Acyclic through the rest of the graph.** No path leaves `A ∪ B` and comes back (through a call, a view, a move or another region). The
   edges are the data edges and the **anti-dependencies** of in-place updates: a node that reads version `k` of a buffer comes before the node
   that makes version `k + 1`.
3. **Reductions** (D2): a reduction without an `order=` is a region of its own, and its operand is a stored value (its
   producer is in another region). Its bytes then depend on nothing but its own kernel, as in the unfused composition (section 5.3, hazard H4).
   An ordered sum, a total-order `max` and `argmax` / `argmin` join regions freely.
4. **In-place writes.** A region may write an `InOut` or carried parameter (a `store` of one of its values into it, or the update a call would
   otherwise make) only if every read of that parameter's **old** version is inside the region and **lane-local**: at the element (elementwise)
   or row (row template) that the same program writes, never through a broadcast, a gather, a static slice or a reduction over another axis. The
   templates load before they store (an `Out` or `InOut` is written once, with its final version), so such a write races with nothing. Any
   other in-place write ends the region and stays a copy step.
5. **Views as inputs.** A dense input that provably starts on 16 bytes is an ordinary contiguous parameter; a dense one that does not is
   compiled element-aligned (as `kernel.unaligned` does). A non-dense input (a transpose, a column block, a head of a fused `qkv`) is a
   `Strided[...]` parameter of the generated op with its strides as scalar arguments (the machinery every strided extern and every plan step
   already has), read at `sum(coordinate_k * stride_k)` (phase 1 for the elementwise template, phase 2 for the row template). Until a template
   takes one, the view is materialised by `tilefish.contiguous` first: correct, slower, and reported.
6. **Views inside a region.** A view between two ARITH nodes ends the region in phase 1. Phase 2 lets a dense `reshape` stay inside an
   elementwise region (the lane order is the same; broadcast operands are addressed against the shape at their point of use).
7. **Outputs.** A region writes each output into a fresh buffer, into the caller's output buffer (a returned value, or a whole reshape of
   one), or into an `InOut` / carried parameter under rule 4. An output that is a non-dense view of a larger buffer (writing `q`, `k` and `v`
   into column blocks of one tensor) needs a strided store: phase 2; until then it is a buffer and a copy.
8. **Index range and limits.** The union obeys the template's index range (offsets are `i32`; a dynamic size gets a `max_elements`
   precondition) and the limits of 3.7.
9. **It lowers.** `tilefish.lower` of the union's synthetic kernel (4.1) succeeds. The lowering is the authority on what a template can do,
   so the grouping never has to restate its rules.

### 3.6 The order of the steps

Steps are emitted in a topological order of the condensed graph (calls, moves, regions; data edges and anti-dependencies), taking among the
ready nodes the one whose first node has the smallest IR index. Without fusion this is the IR order, so a graph with no arithmetic plans
exactly as before. The live range of a buffer may grow or shrink; `reuse=True` lays the arena out from the new ranges as it does today.

### 3.7 Limits (version 1, part of what `version` fixes)

* at most 64 ARITH nodes per region;
* at most 32 tensor parameters and outputs per region (a step has at most 256 pointer arguments);
* a row region keeps at most 8 full-row f32 values live at once (an estimate of registers per lane from `BLOCK_D` and the warp count); above
  that it is split between two nodes, deterministically;
* strided inputs of rank at most 4 (as `tilefish.contiguous`);
* no region spans a call, a barrier, a copy or a memset step.

A limit only splits regions. In an exact mode a split changes no byte; it is a speed choice.

## 4. Code generation

### 4.1 A region is a synthetic `@tilefish.kernel`

The fusion pass builds, for each region, the `KernelOp` that a user would have written by hand:

* **signature**: the region's inputs in order of first use, each `Tensor[shape, dtype]` over the graph's own `Dim`s (a strided input
  `Strided[...]` with its `StrideOf` scalars; a `DevScalar` as is; an `InOut` for rule 4), then one result per output;
* **body**: the region's nodes, copied into a fresh kernel trace with `region` and `loc` stripped and the graph's default `math` applied to
  every node that has no `math=` of its own;
* **options**: `fp_fusion=False` (D3: a declared contraction is an explicit `fma` node); `math` as declared; `num_warps` by the template's
  rule (4.4).

`tilefish.lower` turns it into Triton source exactly as it lowers any kernel. This is the central choice of the design: **a region inherits
the numerics contract of the lowering, which is written down and measured, instead of a new code generator that would need all of it measured
again.** A region and a hand-written `@tilefish.kernel` of the same nodes and options generate the same source (up to the function name), so
they compile to the same instructions.

### 4.2 The generated op, its name and its source

* The op is an `ExternOp` with the generated source attached, made like `contiguous` (`_copyops._generated`): it builds, launches, exports,
  replays and plans like any generated op, and its `reference=` is the `reference` mode's interpretation of the region's nodes.
* Its **name** is `fused_<hash>` (or `<name>_<hash>` under `tilefish.fused(name=...)`), where `<hash>` is the first 12 hex digits of the sha256
  of the canonical form of the synthetic kernel: the fusion `version`, the options, the signature (dtypes, shapes as dimension expressions,
  strides) and the nodes (ops, attributes, scalars as exact hex floats). The same region in two graphs, or in two plans of a set, is one op and
  one manifest entry. Because a name is a content hash, a source never names the graph or the node indices it came from; provenance goes to the
  fusion report (5.5).
* The source is written content-addressed (`<cache>/generated/<sha256>/tilefish_generated/fused_<hash>.py`), as the other generated sources
  are, so exports stay reproducible.
* The module header states the region's numerics in a comment (template, `fp_contract`, `intermediates`, `reductions`, `math`). A comment is in
  `source_sha256` but not in the TTIR digest; it documents, the IR decides.

### 4.3 Template extensions that fusion needs

| extension | needed for | phase |
|---|---|---|
| strided (non-dense) inputs in the elementwise template: an operand read at `sum(coordinate_k * stride_k)`, the coordinates as note 12 computes them | gated activations, heads of a fused `qkv`, transposed operands | 1 |
| every narrowing float cast in a region as inline PTX (hazard H1, confirmed by E5) | NaN-payload exactness | 1 |
| strided inputs in the row template (a row stride; column stride 1 or general) | norms over views | 2 |
| strided outputs (a store into a column block) | writing `q`, `k`, `v` in place | 2 |
| dense reshapes inside an elementwise region | heads, `[B, T, D]` reshaped | 2 |
| row-local data movement: half split, pair interleave and concatenation of the last axis (`tl.reshape`, `tl.split`, `tl.join`: data movement, exact) | rotary embeddings | 3 |
| segmented row reductions (over blocks of `k` elements of the last axis) | block-scaled quantisation | 4 |

Multiple outputs, `InOut` parameters, broadcasts along any axis, device scalars, gathers and scatters, ordered sums and static slices exist
already.

### 4.4 Configurations per bucket

* **Elementwise regions** have per-element ops only, whose bytes do not depend on the block size, the vector width or `num_warps` (obligation
  O5). Their configuration is a speed choice: the template's default (`BLOCK = 1024`, 4 warps), or a per-bucket table through the
  `by_bucket` mechanism of externs (a smaller block for a small bucket). Changing it changes the binary and the pin, never the output bytes.
* **Row regions** with ordered sums, `argmax` / `argmin` and per-element ops only are layout-independent too: `num_warps` per bucket
  (`clamp(BLOCK_D // 256, 1, 16)` today) is a speed choice.
* **A reduction without an order** (D2) is a kernel of its own, the same kernel in the fused plan and in the unfused composition. Its warp
  count follows the template's rule per bucket and is part of its numerics (note 6), recorded and pinned, as for any `@tilefish.kernel`.
* Nothing is autotuned at export.

### 4.5 Grid or persistent

A region launches on the template's grid: `(cdiv(n, BLOCK),)` for elementwise, `(r,)` for rows (product grids such as `T * C` are fine now).
Every element or row is computed by one program with the same code, so a persistent launch (`tilefish.persistent`) gives the same bytes; it is a
later speed option for many short rows, and it keeps the default promise of a persistent launch that the bytes do not depend on the SM count.

## 5. The numerics contract and its modes

### 5.1 The declared program and the unfused composition

* The **declared program** `P = R_mode(G)` is the graph body after the numerics rewrite its mode names (5.4). Every rewrite is an IR-to-IR
  function that does not depend on how the regions are formed; in the default mode it is the identity. Each ARITH node of `P` has a dtype, and
  its result is a value of that dtype computed by the contract of its op: one rounding for an arithmetic operator or a cast, the fixed
  instruction sequence of its note for a math function.
* The **unfused composition** `U(P)` is the plan of `P` with `grouping="none"`: every ARITH node of `P` is a generated kernel of its own,
  lowered by the same templates with the same per-node options, and every intermediate is stored in its dtype (a mask as `u8`, a rank-0 value as
  a one-element tensor). A node that does no arithmetic on data (a device-scalar `load`, an `iota`) is recomputed in each consumer instead of
  stored, which is exact.
* `U(P)` is a real plan: it plans, exports, runs in `jit` and `replay`, captures, and its taps show every intermediate. It is the oracle.

### 5.2 The claim, and what it rests on

> **Exactness.** In an exact mode, for every bucket and every input (every bit pattern, NaN payloads, signed zeros, subnormals and infinities
> included), the fused plan `F(P)` writes the same bytes as `U(P)` to every output, every `InOut` and every carried buffer.

It holds by construction when these obligations hold. Each has a test (5.6) and, where it can, a build-time guard.

| | obligation | why it holds | check |
|---|---|---|---|
| O1 | storing a value and loading it back is the identity | loads and stores move bits (every storage dtype, NaN payloads included); a mask stored as `u8` 0/1 and read as `!= 0` is the same mask | special-value tests |
| O2 | an op's code does not depend on its neighbours | the lowering emits each node's expression from the node alone; scalars are typed per node | source equality: the region's source equals the kernel of the same nodes |
| O3 | nothing contracts across two ops | `enable_fp_fusion=False` is not enough on sm_100 and later (H9): a region writes every float `+ - *` as one scalar `.rn` instruction of inline PTX (`context_free=True`), which ptxas does not contract; a declared contraction is an explicit `tilefish.fma` (inline PTX); IEEE helpers carry `.rn` | the options in every entry; the SASS of the case H9 was found in; the random graphs |
| O4 | no fold, reassociation or fast-math rewrite across two ops changes a bit | Triton is not expected to set fast-math flags other than contraction (the guard checks it on every build); LLVM folds only what it considers exact, and one fold it considers exact is not exact on NaN payloads (H1) | a guard scanning each region's LLVM IR for `reassoc`, `nnan`, `ninf`, `nsz`, `arcp`, `afn`; NaN-payload tests |
| O5 | a per-element op's result depends only on its operands at that element | per-lane code; a packed `x2` instruction computes each half as the scalar one does; the block size, vector width and warp count only distribute lanes | the same region at several configurations gives the same bytes |
| O6 | every reduction inside a region is layout-independent | ordered sums (note 17), `argmax` / `argmin` (note 8); a Triton tree reduction is not (H4), so it runs alone (D2) | E1 and E2; ordered-sum tests at several warp counts |
| O7 | `F` and `U` are compiled by one toolchain, with one libdevice and one CUDA math library | one export | the CUDA math guard and the manifest's `toolchain` |
| O8 | the flush-to-zero behaviour of each op is its own | same compile options, same per-op code (`.ftz` comes from the op, note 5) | subnormal sweeps |

### 5.3 Hazards found while designing, and how each is closed

| | hazard | effect | resolution |
|---|---|---|---|
| H1 | MLIR's arith folders or LLVM's instcombine fold `fptrunc(fpext(x))` to `x`, which they treat as exact, and reach through a select between widened values too (confirmed by E5). In `U` the widening and the narrowing are two kernels and the narrowing is `cvt.rn.bf16.f32`, which makes every NaN canonical (`0x7FFF`, note 1); fused and folded, the input's payload survives | `x.to(f32).to(bf16)` of an input NaN with a non-canonical payload differs (only NaNs, and only NaNs that come from an input: GPU arithmetic produces canonical ones) | every narrowing float cast in a region is emitted as inline PTX (`cvt.rn.bf16.f32` / `cvt.rn.f16.f32`, the instruction the standalone cast compiles to), which no fold can see; tested on every bit pattern of each cast and on cast pairs with NaN payloads |
| H2 | a value `U` loads from memory becomes a constant in `F` (a memset folded into a region, a known zero), and MLIR or LLVM folders then apply (`x + (-0.0) -> x` keeps a NaN payload that a GPU `add` would canonicalise) | NaN payload and signed-zero differences | rule: **a value `U` loads, `F` loads or computes by the same ops; it is never replaced by a constant**. Constants are only the IR's own literals, which are per node and so folded identically in `F` and `U` |
| H3 | contraction across an op boundary (`a * b` in one op, `+ c` in the next) | 23% of elements differ (note 2) | O3: no contraction unless declared |
| H4 | a Triton tree reduction's operand layout (elements per thread, from the loads that anchor the layout) depends on the whole kernel, so the same `tl.sum` may add in another tree when other loads of other dtypes share its kernel | last-bit differences, data-dependent | D2: unordered reductions run alone, and their operand is stored; experiment E1 measures how often the tree really changes |
| H5 | `tl.max` on `+0.0` and `-0.0`, and on NaN, might depend on the order | a sign of zero | E2: order-free on every tested row at 1 to 16 warps and under permutation; `max` may join row regions, and the phase 2 tests keep checking it |
| H6 | the library behind a math op changes with the CUDA or libdevice version (CUDA 13.2 changed `erff`) | other bytes, `F` and `U` alike | not a fusion hazard (O7); the CUDA math guard warns |
| H8 | (found in phase 1) ptxas folds `neg.f32` into its consumer as a negated operand (a `where` of `-x`: `FSEL` of `-R`), which flips a NaN's sign, where `neg.f32` alone gives the canonical NaN | a NaN `0xFFFFFFFF` fused, `0x7FFFFFFF` unfused (61 of 1536 elements of a random graph) | a region's negation gives the canonical NaN explicitly (`testp.notanumber` or `setp.nan`, `selp`) |
| H9 | (found in phase 1) LLVM packs two lanes' f32 ops into `mul.rn.f32x2` / `add.rn.f32x2`, and ptxas 12.9 contracts that pair into `FFMA2` even with `--fmad=false` (read in the SASS of the same PTX assembled for sm_100a) | one ULP, 33 of 1536 elements of a random graph; and `@tilefish.kernel(fp_fusion=False)` is not two roundings where its ops are packed | a region writes every float `+ - *` (and the multiplies of `silu` and `gelu`) as scalar `.rn` inline PTX: separate `FMUL` and `FADD` in the SASS |

### 5.4 Modes

Every knob of version 1 is an IR rewrite, so every mode keeps `F(P) == U(P)` for the rewritten `P`. A rewrite is defined on the IR, before
grouping, so its rounding points never depend on where the regions fall.

| knob | value | what it is | relation to the unfused composition |
|---|---|---|---|
| `intermediates` | `"declared"` (default) | the identity: every op rounds to the dtype the IR gives it (a `bf16 * bf16` is a bf16 multiply; `x.to(f32)` keeps f32 until a cast) | exact |
| | `"opmath"` | torch's eager rule as a rewrite (D4): a 16-bit op with a scalar operand becomes `op(a.to(f32), f32(c)).to(bf16)`, with the scalar's original Python value rounded to f32 (the IR keeps it). Two-tensor 16-bit `+ - *` are unchanged: an f32 product of two 16-bit values is exact, an f32 sum is exact or so far from a 16-bit midpoint that the second rounding cannot differ, so torch's bytes are the native op's (the exhaustive pair test of 5.6 checks it) | exact for the rewritten program; with `math="ieee"` and the ops whose IEEE form is torch's (notes 13, 15, 16) it is meant to be torch's eager composition byte for byte (to be measured, 5.6) |
| | `"f32"` | Inductor-like widening as a rewrite: every 16-bit value made by an ARITH node is computed in f32 and rounded to its declared dtype only where it leaves the arithmetic: an argument of a call, a returned value, a store, a barrier. A cast written in the body stays a rounding point. Defined on the IR (calls, returns, stores, casts), not on regions | exact for the rewritten program (a region split stores the f32 value); not equal to `"declared"` |
| `fp_contract` | `"none"` (default) | no contraction (D3) | exact |
| | `"mul-add"` | a rewrite: an f32 `add(mul(a, b), c)` or `add(c, mul(a, b))` whose `mul` has no other use becomes `fma(a, b, c)`; `sub(mul(a, b), c)` becomes `fma(a, b, neg(c))`, `sub(c, mul(a, b))` becomes `fma(neg(a), b, c)`; when both operands of an `add` qualify, the left one is contracted. 16-bit contraction needs a 16-bit `fma` primitive (Q10) | exact for the rewritten program (`fma` and `neg` are exact in the reference, note 16) |
| `math` | `"fast"` / `"ieee"` | the per-op math (note 13) | exact either way (the same code in `F` and `U`) |
| `grouping` | `"auto"` / `"none"` | `"none"` is `U(P)` itself | exact |

A reduction without an order is not a mode: it is a kernel of its own in `F` and `U` alike (D2).

### 5.5 How it is digested, and what a pin means

* The mode determines `P`, so it determines each generated source (`source_sha256`) and its TTIR (`source_digest`); it determines the compile
  options (`enable_fp_fusion`, `num_warps`), which are in the entry's `options` and declaration digest. The entry is in the manifest digest,
  the manifest digest is in the plan digest. **A plan pin therefore pins the numerics of every fused kernel**, with no new field.
* The fusion settings are part of what Python derives the plan from, so a graph whose mode changed is refused as stale, like a graph whose ops
  changed.
* In an exact mode a pin also implies, by 5.2, that the bytes are `U(P)`'s; `tilefish.verify_fusion(graph, *inputs, buckets=...)` checks it on
  any inputs at any time, and `export` checks it on seeded inputs before writing anything, unless `verify=False` (D6, 5.6).
* **The fusion report.** `tilefish.fusion_report(graph, buckets=...)` and every `export` (as `<graph>.fusion.json` next to the plan) list each
  region: its generated op, template, iteration space, the graph nodes it took (index, op, `loc`), its inputs (and whether a view was read
  strided or materialised), its outputs, every rounding point it keeps, every rewrite applied (contractions made, roundings removed by
  `"f32"`), every reduction and its order, and the mode. It is informative (hosts never read it) and
  derived (Python regenerates it from the graph). It is the "documented fusion step" that a port of a hand-fused engine needs: which original
  ops were merged, and what changed numerically.

### 5.6 How tests prove it

* **Construction (no GPU).** A region's source equals the source of the hand-written `@tilefish.kernel` of the same nodes and options.
  Each rewrite is tested IR-in, IR-out. Grouping is deterministic (two processes, same names, sources and plan digests) and invariant under
  renaming locals. Every legality rule has a refusal or a split test (a broadcast read of an `InOut` being written, a cycle through a call, a view
  read after an in-place update).
* **Fused against unfused (GPU, the core).** `F(P)` and `U(P)` byte for byte, at every bucket:
  * exhaustive: chains of one and two unary ops (with the casts they need) on every bf16 and f16 bit pattern (65 536 inputs) for every op pair
    and cast pair; chosen f32 chains on all `2**32` patterns, as the IEEE tests do;
  * large grids: binary ops on `2**24` to `2**26` random bit-pattern pairs, the cross product of special values (both zeros, both infinities,
    NaNs with canonical and non-canonical payloads, the smallest and largest subnormals and normals), and cancellation cases;
  * property-based: a seeded generator of random arithmetic DAGs (2 to 30 nodes, 1 to 4 inputs, random dtypes, casts, scalars, broadcasts,
    masks, device scalars, strided views) between a call and a return: 10 000 graphs planned and lowered on any machine, 500 run on each GPU
    architecture against `U` on special-value inputs;
  * reductions: ordered sums fused against alone, rows of 1 to 16 384 columns of every magnitude, signed zeros, NaNs, infinities and
    subnormals, at 1 to 16 warps;
  * the use cases of 1.1 as graphs, against `U` and against hand-written kernels of the same declared numerics.
* **Against the reference.** `reference` mode interprets `P` (the rewritten program) node by node. Where every op's reference is exact (IEEE
  math, casts, comparisons, `fma`, `rcp`, `neg`, ordered sums) the fused binary is the reference byte for byte; elsewhere the tabulated
  tolerances apply. `intermediates="opmath"` with IEEE math is compared with torch's eager composition of the same ops, and two-tensor
  bf16 and f16 `+`, `-` and `*` are compared with torch's eager op on every pair of bit patterns (`2**32` pairs per op and dtype), which is
  what lets `opmath` leave them unchanged.
* **Hosts.** The Rust gate gets fused plans: Rust against Python replay byte for byte, launched and in CUDA graphs, with carried buffers and a
  shared arena.
* **Export-time verification** (D6). `export` runs `F` and `U` in `jit` on seeded inputs with special values at every bucket on the export
  device and refuses a difference; `verify=False` opts out. Its own tests: a deliberately broken region (a contraction slipped in) is refused.
* **Both architectures.** Everything above runs on sm_103 and sm_120. `F == U` is claimed per architecture; cross-architecture equality is
  not (it holds only where the ops' own PTX is the same, as today).

## 6. Interaction with plans

**No host changes.** A region is a `kernel` step of a generated entry, as a `tilefish.contiguous` step is. The Python runner and the Rust crate
launch it like any Triton entry; nothing in either schema is new.

* **Arena and reuse.** A fused intermediate never gets a buffer, so arenas shrink. `reuse=True` lays out the remaining buffers from the new live
  ranges.
* **Carried buffers.** A carried parameter is a region input like any parameter; a region may write it under rule 4 (3.5), which extends
  "`h[...] = op(x)` is written by op's launch directly" from calls to regions. The arena's bookkeeping of carried shapes is unchanged.
* **Memset steps.** `zeros_like` stays a memset step and is never folded into a region as a constant (H2). A region reads the zeros from memory.
* **Shared arenas and plan sets.** Unaffected. A region that occurs in several plans of a set is one manifest entry (its name is a content hash).
* **Captures.** Fused steps are kernel launches; `tilefish.capture`, `replay_graph` and capture in a caller's graph work as before.
* **Taps and `run_step`.** A value fused away has no buffer, so `taps=` refuses it with the advice to add `tilefish.barrier` or to tap the
  unfused plan (`grouping="none"`), whose intermediates are, in an exact mode, the very values the fused kernel computes. `run_step` indexes the
  fused plan's steps.
* **Weights, inputs and roles.** Unchanged: a region's parameters are buffers of the plan with their roles.
* **`reference` mode** of a graph interprets ARITH nodes (it refuses them today), through `P`, so a declared mode shows in the reference.
* **`jit`** builds the generated ops in memory; the build cache keys them like any generated kernel.
* **Rust launcher emission** (`hosts.rust.emit`) sees one more Triton entry per region; nothing changes there (whether to skip generated entries
  in the typed launchers is cosmetic).

## 7. Epilogue and prologue fusion into library ops and externs

Elementwise work after a GEMM can run as (a) a generated kernel after it, (b) a cuBLASLt or cuDNN epilogue, or (c) an epilogue inside an
extern's own kernel. Version 1 has route (a) only (D5); route (c) is phase 4; route (b) and prologues are future work (section 11.2).

| route | what is pinned | exact against `U`? | status |
|---|---|---|---|
| (a) generated kernel after the call | the region's machine code | yes, by 5.2 | the default, phases 1 to 3; one extra read and write of the call's output |
| (c) an extern's epilogue hook | the extern's binary with the hook inlined (Triton folds a `tl.constexpr` jit function into the IR, so the TTIR digest covers it) | yes if the hook is applied to the value the kernel would store, after its own rounding to the output dtype, and its arithmetic cannot contract with the kernel's (the hook's ops emitted as `.rn` inline PTX, or a cast at the boundary); that the kernel's own arithmetic is unchanged is checked by the export-time verification (D6), not proven | phase 4: `@tilefish.extern(..., epilogue=tilefish.hook("EPILOGUE", out="c"))`; the kernel calls `c_tile = EPILOGUE(c_tile, ...)` before its store, and the plan passes a generated `@triton.jit` function of the region; each extern-plus-epilogue pair is a new binary, and a pair that verification refuses falls back to route (a) |
| (b) cuBLASLt or cuDNN epilogues | the library's algorithm or engine pin, version, canary | only by measurement (section 11.2) | future work |

An epilogue that a user declares as a library op (`impl="cublaslt.gemm_bias"` today) is an ordinary call with the library's stated numerics,
not fusion, and is unaffected.

## 8. Schema impact and migration

### 8.1 Schemas

* **Plan schema 1:** no change. Regions are `kernel` steps.
* **Manifest schema 3:** no change. A region is a `triton` entry with `source_sha256`, as every generated op is.
* **Python-only additions:** `@tilefish.graph(fusion=...)`, `tilefish.fusion`, `tilefish.barrier`, `tilefish.fused`, `mean(order=)`,
  `tilefish.fusion_report`, `tilefish.verify_fusion`, `export(..., verify=...)`, and the informative `<graph>.fusion.json`.
* **Possible later, additive:** an optional entry field naming a fused entry's mode, so that an auditor without Python can read it from the
  manifest. Both strict readers would have to accept it first, and an older reader would refuse a manifest that has it; this RFC does not
  propose it for version 1 (Q8).

### 8.2 Migrating hand-written ops

* **Nothing has to migrate.** Every existing `@tilefish.kernel`, extern and generated op keeps its identity; a graph without arithmetic plans
  as before.
* **A `@tilefish.kernel` body moved into a graph** keeps its bytes when the kernel is already exact-expressible: declared
  `fp_fusion=False` and every reduction ordered (or none). Under `with tilefish.fused():` the region then generates the kernel's own source. A
  kernel that relies on compiler contraction (the kernel default) or on a Triton tree reduction stays a `@tilefish.kernel` in version 1 (the
  matching graph modes are future work, section 11.1). The better migration is to say the numerics: `tilefish.fma` where the kernel meant one
  rounding, `order=` where it meant an order.
* **Hand-written elementwise externs** (bias plus activation, residual adds, casts) can be replaced by graph arithmetic once a test shows the
  graph's declared numerics reproduce them (`verify_fusion` against the extern on special-value inputs). The one-instruction primitives and
  ordered sums exist so that a hand-fused CUDA kernel's bytes can be written in the DSL.
* **Calls to another graph** stay unsupported; a plain Python helper function traces inline and is the way to reuse arithmetic.
* **Inlining `@tilefish.kernel` calls into regions** (treating a DSL kernel call as its nodes) is exact only when its options equal the region's
  and its reductions are ordered; it would change existing plans, so it is a later opt-in (`fusion(inline_kernels=True)`), not part of
  version 1.

## 9. Staged implementation plan, test plan and open questions

### 9.1 Milestones

| phase | content | milestone (done when) |
|---|---|---|
| 0: experiments | E1 to E5 below, on sm_103 (then sm_120) | the `max` lowering is decided, H1 is confirmed or ruled out, speed expectations are set; E1 and E4 inform future work |
| 1: elementwise regions | `tilefish.fusion(version=1)` with the default mode (`intermediates="declared"`, `fp_contract="none"`, `math="fast"` / `"ieee"`), on by default (D1), `grouping="none"`, `barrier`, `fused`; the fusion pass and the synthetic kernel; strided inputs in the elementwise template; the H1 cast emission; the fast-math flag guard; graph `reference` mode for ARITH nodes; fusion report; `verify_fusion` and export-time verification on by default (D6); a reduction runs as a region of its own (the row template as it is) | the README's `mlp` graph plans and runs; bias + activation, residual add, dequantisation, periodic activations and masks fuse; `F == U` on the exhaustive, grid and property-based suites; Rust gate green; exports reproducible |
| 2: row regions | ordered reductions in regions (`sum`, `mean(order=)`, order-free `max`, `argmax`); multiple outputs and side outputs; `InOut` and carried writes (rule 4); strided inputs in the row template; strided outputs; dense reshapes inside regions; per-bucket configurations | rmsnorm + residual, layer norm, softmax pieces and per-row quantisation fuse exactly and run at least as fast as the hand-written `@tilefish.kernel` equivalents |
| 3: exact modes and movement | `intermediates="opmath"` and `"f32"`, `fp_contract="mul-add"` (with 16-bit `fma` if Q10 is accepted); row-local split / interleave / concatenate for rotary embeddings | each mode's tests: `opmath` + IEEE against torch eager; `mul-add` against explicit `fma`; `F == U` for every mode |
| 4: extern epilogues | extern epilogue hooks (route (c)); horizontal fusion and segmented reductions as separate items | an extern GEMM with a fused bias-activation epilogue equals `U` at every bucket |

Everything that is not exact is outside these phases (section 11).

### 9.2 Phase 0 experiments (each small, on a free GPU)

* **E1, tree reductions in context.** The same `tl.sum` row kernel alone, and fused after loads of other dtypes (bf16 and f32 operands, as
  rmsnorm has): count differing rows over the reduction-order suite. Confirms D2's premise and informs section 11.1 (expected: rows differ, so a tree
  reduction can never join a region exactly).
* **E2, `tl.max` order-freedom.** Rows with `+0.0` / `-0.0` mixes and NaNs at 1 to 16 warps. Decides the `max` lowering in regions.
* **E3, speed.** Fused against unfused against the hand-written `@tilefish.kernel` for bias + SiLU, SwiGLU on strided halves, rmsnorm +
  residual and softmax, at decode and prefill row counts, on a B300 slice and an RTX 5090. Measures what D2 costs (an unordered reduction alone against an ordered
  one fused).
* **E4, the cuBLASLt bias epilogue.** `(gemm_f32 + b.to(f32)).to(bf16)` against `gemm_bias` with a bf16 output, every pin of a test set,
  special-value biases. Informs section 11.2 (whether route (b) is worth building later).
* **E5, H1.** Confirm the `fptrunc(fpext(x))` fold on Triton 3.5.1 and that the inline-PTX cast closes it.

### 9.3 Test plan summary

Section 5.6, by phase: construction tests and refusals on every machine (`pytest`); `F == U` suites, reference comparisons and the Rust gate
on GPU (`pytest -m gpu`, `rust_gate.py`), on sm_103 and sm_120; the reproducible-export tests on fused exports; the use-case
graphs as goldens with their fusion reports committed.

### 9.4 Open questions

Q1 to Q7 are decided (D1 to D7 above). Open, each with the default that will be implemented unless the review says otherwise:

* **Q4b. 16-bit unary math under `opmath`.** Should `intermediates="opmath"` accept `exp(x_bf16)` and the like, as torch does? That couples
  the tracer's dtype rules to the graph's mode. **Recommended default: no**; the body writes `.to(f32)`, in every mode.
* **Q8. A manifest field for the mode.** **Recommended default: none in version 1**: the mode is in each generated source's header (in
  `source_sha256`) and in the fusion report. A field would need both strict readers updated first, and older readers would refuse manifests
  that have it.
* **Q9. Names.** **Recommended default: as proposed**: `@tilefish.graph(fusion=tilefish.fusion(...))`, `tilefish.barrier`, `tilefish.fused`,
  `tilefish.fusion_report`, `tilefish.verify_fusion`, `export(..., verify=False)` as the opt-out, generated ops named `fused_<hash>` (or
  `<name>_<hash>`), and `<graph>.fusion.json`.
* **Q10. 16-bit `fma`.** **Recommended default: yes, in phase 3** together with `fp_contract="mul-add"`: `tilefish.fma` for bf16 and f16
  (`fma.rn.bf16`, `fma.rn.f16`, inline PTX), measured as the f32 `fma` is (note 16). Until then `"mul-add"` contracts f32 only.
* **Q11. Acceptance.** **Recommended default:** phase 2 is measured on the models already ported to tilefish, each written in its original op
  structure: `F == U` at every bucket, and speed against their hand-fused kernels, on sm_103 and sm_120; the use-case graphs of 1.1 are
  committed as goldens with their fusion reports.

## 10. Prior art

| system | how it fuses | intermediates of 16-bit chains | FMA contraction | reduction order | relation to the unfused program |
|---|---|---|---|---|---|
| torch.compile / Inductor | lowers to a loop IR; its scheduler fuses pointwise and reduction nodes (vertically and horizontally) into Triton kernels; epilogue fusion into its own GEMM templates under autotuning | computed in f32 inside a kernel by default (`codegen_upcast_to_fp32`), so roundings eager would do are skipped; an option emulates eager's casts | the backend compiler's default | Triton's tree, block sizes from heuristics or autotuning | no byte contract; compared with eager within a tolerance |
| XLA | HLO passes fuse elementwise ops into loop fusions, reductions into input fusions, multi-output fusions; a GEMM rewriter folds bias and activations into cuBLASLt epilogues | may normalise low-precision ops to f32 on some targets; algebraic simplification rewrites arithmetic | the backend's | the emitter's; flags exist for run-to-run determinism | no byte contract; determinism across runs, not against the unfused program |
| TVM | fuses by operator pattern (injective, reduction, elementwise into an output-fusable op such as a dense layer); schedules come from a tuner | per the schedule | per the target | chosen by the schedule a tuner finds, so it can change with tuning | no byte contract; an optional pass swaps in fast approximations |
| Triton-based fusers and kernel libraries | compilers that emit Triton from a graph, and libraries of hand-fused Triton kernels (norms, rotary embeddings, gated activations) | whatever the kernel computes, usually f32 | Triton's default (on) | Triton's tree | parity by tolerance tests |
| tilefish (this RFC) | regions lowered by the DSL's own templates into generated ops, between pinned library calls | declared: `declared`, `opmath` or `f32`, each a rewrite defined on the IR | none unless declared (`tilefish.fma`, `mul-add`) | ordered or order-free inside a region; an unordered reduction is a kernel of its own | byte-identical to the unfused composition, by construction, tested, and verified at every export; no non-exact mode in version 1 |

The difference is not the fusing, which these systems do more aggressively; it is that each of them treats the numerics of a fused kernel as
an implementation detail checked within a tolerance, where tilefish makes the rounding points, the contraction and the reduction order part of
the declaration, so that fusion is a change of kernel boundaries only.

## 11. Future work: modes that are not exact, and other extensions

Version 1 has exact modes only (D5). What follows was designed in the draft and is kept here so that a later RFC can start from it; none of
it is planned.

### 11.1 Binary-defined modes

A *binary-defined* mode is one whose bytes are deterministic, pinned and reproduced by every replay and both hosts (the digest covers the
machine code), but are not those of any unfused composition: adding a barrier or changing the grouping may change them, the reference holds
them only within the measured tolerance, and the fusion report would mark each such region. It is the contract every `@tilefish.kernel` with
a Triton reduction or with default contraction has today.

* `fp_contract="compiler"`: `enable_fp_fusion=True`, so LLVM and ptxas contract what they see. What they see depends on the region's shape, so
  the bytes depend on the grouping. It is what a migrated `@tilefish.kernel` that relies on contraction would need.
* `reductions="tree"`: Triton tree reductions join regions, each row region's warp rule part of its numerics. E1 measures how often the tree
  changes in context (hazard H4).
* Tests such modes would need: determinism across runs, `jit` equal to `replay`, Rust equal to Python, within tolerance of the reference, and
  a report entry per region; `verify` would compare against a pinned earlier export instead of `U`.

### 11.2 Library epilogues (route (b))

* **cuBLASLt epilogues** (bias, ReLU, GELU and their auxiliary-output variants, `beta * C` as a residual, per-tensor FP8 scales and an
  absolute-maximum output) are pinned as the GEMM is: the 64-byte algorithm, the cuBLASLt version, the SM count, a canary. They are exact against
  `U` only by measurement: the library has no IR, and its epilogue's rounding is not documented to match any tilefish op. The existing pins show
  the risk: the bias epilogue with a split-K and f32 partial sums was measured to round twice on one architecture, which is why, without the
  opt-in to reduced-precision partial sums, a pinned GEMM with a bias takes no split-K.
* A rewrite of graph arithmetic into an epilogue (`epilogues="cublaslt"`) would only apply to patterns whose rounding points agree on paper
  (`(gemm_f32(x, w) + b.to(f32)).to(bf16)` is `gemm_bias` with a bf16 output: an f32 sum, one rounding), and only after `export` measured the
  rewrite equal to `U` at every bucket on seeded and special-value inputs for that pin (E4 is the first measurement). A rewrite that changes a
  rounding point (`round(acc) + b` into `round(acc + b)`) would never be automatic.
* **cuDNN runtime-fusion engines** (a convolution followed by pointwise ops) are the same class: a declared library op, no automatic rewrite.

### 11.3 Other extensions (exact, not yet planned)

* **Prologue hooks** (route (c')): dequantise or normalise a GEMM operand as an extern loads it; per element and exact under the conditions of
  route (c), with the work repeated for every tile that loads the element.
* **Horizontal fusion** and **recomputation** of cheap nodes into several regions (a new `version`).
* **Inlining `@tilefish.kernel` calls** into regions (8.2), opt-in.
* **A canonical default order** for unordered reductions, chosen from `D` (exact and fused, but a second meaning of `sum` without `order`
  next to the kernel body's), if what D2 costs (E3) turns out to matter.

## Appendix A: a worked example

The graph of 2.1, planned for `M` in `{32, 64}` with `F = 3072`, `H = 1024`:

```
IR of mlp                                   regions (version 1, default mode)
#0 t = gemm_gu(x, w_gu)        CALL         -
#1 g, u = split(t, F, -1)      VIEW         -
#2 g32 = g.to(f32)             ARITH E(M,F) ┐
#3 s = silu(g32)               ARITH E(M,F) │ region fused_<hash>: elementwise over [M, F]
#4 u32 = u.to(f32)             ARITH E(M,F) │ inputs g, u: strided views of t (strides (2F, 1), offsets 0 and F)
#5 p = s * u32                 ARITH E(M,F) │ output a: [M, F] bf16
#6 a = p.to(bf16)              ARITH E(M,F) ┘
#7 y = gemm_d(a, w_d)          CALL         -
```

The plan's steps (each bucket): `gemm_gu.M32` writes `t0`; `fused_<hash>` reads two views of `t0` and writes `t1`; `gemm_d.M32` writes `out0`.
The unfused composition has five kernels between the GEMMs and four `[M, F]` f32 intermediates; the fused plan has one kernel and none.
They are byte-identical: the widening casts are exact, `silu` (its instruction sequence of note 5) and the multiply run the same instructions
on the same f32 values in both, nothing is contracted, the narrowing cast is `cvt.rn.bf16x2.f32` in both, and the fused kernel loads `g` and `u` from `t0` instead of loading the stored casts, which
changes nothing (O1). The fusion report entry:

```json
{"op": "fused_<hash>", "template": "elementwise", "space": ["M", "F"], "exact": true,
 "nodes": [{"index": 2, "op": "cast"}, {"index": 3, "op": "silu"}, {"index": 4, "op": "cast"}, {"index": 5, "op": "mul"}, {"index": 6, "op": "cast"}],
 "inputs": [{"value": "g", "read": "strided"}, {"value": "u", "read": "strided"}],
 "outputs": [{"value": "a", "dtype": "bf16", "buffer": "t1"}],
 "rounding_points": [{"index": 3, "dtype": "f32"}, {"index": 5, "dtype": "f32"}, {"index": 6, "dtype": "bf16"}],
 "rewrites": [], "reductions": [], "contractions": 0}
```
