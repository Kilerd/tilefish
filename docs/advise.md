# The fusion advisor: which steps of a plan could be one kernel

`tilefish-explain` explains one kernel; it does not say which parts of a model would go faster as fewer kernels. The advisor does that for a whole
plan: it ranks the **fusion candidates** of a graph's plan (steps joined by intermediates that one kernel would never store) by the time they would
save, says why each is not one kernel today and what to do about it, and sums the time of the plan by class of step. A profiler measures the device
time of every step, so that the ranking can be by measured time. It also says which fusion patterns would help most, which is what phase 2 of plan
fusion (`docs/rfcs/0001-plan-fusion.md`) should do first.

The code is `tilefish._advise` (the analysis) and `tilefish._profile` (the profiler).

```console
tilefish-explain DIR --graph G --advise                     # every bucket analysed, ranked at the one that moves the most bytes
tilefish-explain DIR --graph G --advise --bucket M=128      # one bucket
tilefish-explain DIR --graph G --advise --timing t.json     # ranked by measured times (the timing file says the bucket)
  --machine sm_120 | --bandwidth 1600                       # another roofline (default: the plan's architecture), a measured bandwidth
  --keep t7                                                 # a buffer you tap: it stays stored (repeatable)
  --top 40 | --json                                         # how many candidates the text lists; the whole advice as JSON
tilefish-explain DIR --graph G --profile t.json --load models.block:graph --bucket M=128 --devscalar n=77 --pin DIGEST
  --reps 50 --warmup 5 --method graph|events                # with --advise too: profile, then advise with what was measured
```

```python
adv = tilefish.advise("build/block", "block", bucket="M=128", timing="t.json")   # an Advice: adv.text(), adv.doc (JSON), adv.candidates
t = tilefish.profile_plan(block, "build/block", pin=p.digest, bucket="M=128", devscalars={"n": 77}, reps=50, out="t.json")
```

**Why a flag of `tilefish-explain` and not a tool of its own.** The advisor reads what `tilefish-explain` reads (an export directory: the manifest,
a plan or a plan set, the fusion report, the trace and its checks), takes the same `--graph`, `--bucket` and `--json`, and answers the question the
user asked of it ("which parts can be fused to go faster"). One command for "what does this export do and what would make it faster" is easier to
find than two; the code is a module of its own.

## 1. What it reads

Everything from the export directory, with no GPU and no CUDA tool:

* **the plan** (`plan.json`, or `<graph>.plan.json` of a plan set): the steps of every bucket, the buffers and views each binds, the roles of the
  buffers (input, weight, output, intermediate, carried, inout, workspace);
* **the manifest**: what each step launches. A library entry's sizes (`m`, `n`, `k` of a GEMM; `n`, `c`, `k`, the output and filter sizes and the groups
  of a convolution) and data type; a kernel entry's ABI (the access of each pointer: `in`, `out`, `inout`), whether tilefish generated it
  (`source_sha256`) and with which template (a lowered kernel of the row template has a `BLOCK_D` constant), and its options (`enable_fp_fusion`);
* **the fusion report** (`<graph>.fusion.json`): the regions of the graph body, the ops each took, its template, how it reads its inputs (a view
  `materialised` by `tilefish.contiguous`), its reductions and their order;
* **the trace** when the export has one (`trace/graphs/<graph>.json`, `docs/explain.md`): the ops and source lines of each region and, new, the
  **boundaries** of the regions with their reasons (section 4), and each kernel's SASS, from which the advisor tells whether a hand-written kernel
  computes MMAs. A trace of another plan is ignored. Without a trace, `tilefish.advise(..., gop=graph)` computes the boundaries from the graph.

## 2. Each step

At each bucket, for each step:

* **bytes read and written**: the elements of every argument (its view's shape, or the buffer's shape at the bucket) times the size of its data type,
  read when the pointer is `in` or `inout`, written when it is `out` or `inout` (a library step reads its operands and writes `out`; a copy reads `src`
  and writes `dst`; a memset writes `dst`). Two views of one buffer count twice. This is the least a step moves: a GEMM that reads its operands
  several times moves more.
* **FLOPs** of a library step: `2 m n k` for a GEMM, `2 n k prod(output) (c / groups) prod(filter)` for a convolution. A kernel's FLOPs are not known.
* **a class**:

  | class | what |
  |---|---|
  | `library` | a pinned cuBLASLt GEMM or cuDNN convolution |
  | `fused` | a region of the graph body (`fused_<hash>`): elementwise, row, or a reduction alone |
  | `elementwise`, `row` | a `@tilefish.kernel`, lowered by that template |
  | `copy` | a copy or memset step, or a generated `tilefish.contiguous`, `im2col`, `im2col1d`, `col2im1d` |
  | `opaque` | a hand-written Triton or Gluon extern, or a prebuilt binary (with the trace: with or without MMAs) |

* **the roofline**: a library step is memory-bound when its FLOPs per byte are below the machine balance (peak over bandwidth) and compute-bound
  above; an elementwise, row, fused or copy step is memory-bound; an opaque one is unknown (a kernel without MMAs: memory-bound). The estimate of
  its time is `max(bytes / bandwidth, FLOPs / peak)`, a lower bound.

**The machine table** (`tilefish._advise.MACHINES`): the vendor's numbers of one part of each architecture, dense, rounded.

| arch | part | bandwidth (GB/s) | f16 / bf16 MMA, f32 accumulator (TFLOPS) | TF32 | f32 FMA | SMs | balance (FLOP/B) |
|---|---|---|---|---|---|---|---|
| sm_90 | H100 SXM | 3350 | 989 | 495 | 67 | 132 | 295 |
| sm_100 | B200 | 8000 | 2250 | 1100 | 80 | 148 | 281 |
| sm_103 | B300 | 8000 | 2250 | 1100 | 75 | 148 | 281 |
| sm_120 | RTX 5090 | 1792 | 209.5 | 104.8 | 104.8 | 170 | 117 |

A partition (a MIG slice, a green context) has a share of these. A timing file records the SMs of the device it ran on and the bandwidth of a
device-to-device copy it measured; the advisor then uses that bandwidth and scales the peaks by the SMs (a 30-SM slice of a B300 measured 1614 GB/s:
456 TFLOPS, balance 283). `--bandwidth` sets the bandwidth by hand. The ranking by bytes does not depend on the table; the microseconds and the
memory- or compute-bound do.

## 3. Producer and consumer

The steps are the same in every bucket (only sizes differ), so the analysis of who writes and reads what is made once, and its bytes per bucket.
Every write of a buffer starts a new version of it; a step that reads a version is its consumer. A version is a **fusible intermediate** when

* its buffer is an `intermediate` (not an input, a weight, a returned value, a carried buffer, an `InOut` parameter, a device scalar or the workspace);
* the buffer is written once in the plan (not updated in place by an `InOut` pointer, not a result written over its input, `MayAlias`);
* exactly one step other than its writer reads it;
* it is not a buffer you keep (`--keep`, `keep=`: a tap).

One kernel of the producer and the consumer would not store it: it saves the producer's write and the consumer's read of it (the bytes of each
argument bound to it). Steps joined by fusible intermediates form a **candidate**:

* **a pair** or **a chain** (`i -> j -> k`) or **a group** (a step reading two fusible intermediates, or writing two) of steps that are not library
  steps: one kernel of all of them, with one launch instead of several;
* **an epilogue** (a library step and the step that alone reads its result) or **a prologue** (a step whose result only a library step reads): a
  library op is never fused with another, and its epilogue and prologue are future work, so each is a candidate of its own, bounded by it.

**No step may sit between them** (rule 2 of RFC 0001, 3.5): one kernel of a candidate's steps must not need a step outside it, one that depends on a
member and that a member depends on. The candidates are grown from the largest intermediate down; an intermediate whose union would have such a
step in the middle waits, and is tried again once the others have joined their groups (the step may have joined one), until nothing changes. One that
still has a step in the middle is not saved: it is listed under `blocked`, with the steps in the middle, as **`library-in-the-middle`** (a GEMM or
convolution between them: the residual stream of a transformer, written by one norm and read by the next, has the layer's GEMMs between them; future
work) or **`step-in-the-middle`** (another kernel that cannot join them); an intermediate between two library ops is listed as **`library-library`**.
The dependencies are the plan's: a step that reads a version another wrote, and the order of the accesses to a buffer.

A version read by several steps is a **fan-out**, listed but not a candidate: one region of all of them, or recomputation, which version 1 does not do.
The buffers kept stored are listed with why.

## 4. Why it is not one kernel today, and what to do

Each candidate lists its **blockers**, each with the action that removes it and the phase of RFC 0001 that would; its phase is the latest. A step's
class gives one; the fusion pass gives the rest.

| blocker | what to do | phase |
|---|---|---|
| `opaque-extern` | a hand-written kernel without MMAs: write its arithmetic in the graph body so phase 1 fuses it (a row one needs phase 2) | 1 |
| `opaque-unknown` | the same, when there is no trace to tell whether it computes MMAs (and if it does: an epilogue hook where it makes the intermediate, a prologue hook where it reads it) | 1 |
| `opaque-compute-epilogue`, `-prologue` | a hand-written kernel bound by its arithmetic: MMAs in its SASS, or (with a timing file) moving its bytes at less than a quarter of the bandwidth, such as an attention of FMAs or a search; an epilogue hook of the extern (route (c)); a prologue hook | 4, future |
| `dsl-kernel` | a `@tilefish.kernel` (elementwise): write its body in the graph body (RFC 8.2); exact as it is when it declares `fp_fusion=False` | 1 |
| `dsl-kernel-contracts` | the same, for a kernel that lets the compiler contract: write `tilefish.fma` where it meant one rounding, or its bytes change | 1 |
| `row-kernel` | a row `@tilefish.kernel`: needs phase 2 row fusion (write it in the graph body, with `order=` on its sums) | 2 |
| `epilogue-library`, `prologue-library` | a GEMM or convolution's epilogue or prologue: future work (RFC 0001, section 7 route (b), sections 11.2 and 11.3) | future |
| `copy-step` | a copy step: return or store the value where it is made | 1 |
| `memset-step` | a memset step: its zeros are read from memory, never folded into a region (H2) | future |
| `strided-output` | the producer writes the intermediate into part of a buffer (a column block, a head): a strided store (rule 7) | 2 |
| `materialised` | a `tilefish.contiguous` copy of a view, or a row step reading the intermediate through a view that is not dense (rule 5) | 2 |
| `generated-movement` | `im2col` / `col2im` around a convolution: its prologue or epilogue, future work | future |
| `unknown-boundary` | two regions, without the trace or the graph to say why | 1 |

**Graph-body arithmetic that was not fused.** For a graph with arithmetic, the fusion pass says why its regions are where they are
(`tilefish._fusion.boundaries`, written into the trace as `boundaries`; it repeats the pass's own rules, `merge_problem`, and changes no region): one
entry per value that crosses a boundary, with the ops on both sides and their lines, the reason, and the fix:

| reason | why | fix | phase |
|---|---|---|---|
| `barrier` | the value passes `tilefish.barrier` | remove it: a tap of the unfused plan (`grouping="none"`) shows the same bytes | 1 |
| `reduction-unordered` | a `sum` / `mean` without `order=` runs alone (D2) | add `order=`: phase 1 still runs it alone, phase 2 fuses it | 2 |
| `reduction` | an ordered reduction, `max`, `argmax`: a region of its own in version 1 (also a cycle through one) | phase 2 row regions | 2 |
| `gather` | `gather_rows` is a region of its own | phase 2 | 2 |
| `view` | a view between two ops ends a region (rule 6) | compute on one shape, or phase 2 (dense reshapes inside a region) | 2 |
| `materialised` | a view a row region reads is copied by `tilefish.contiguous` first | phase 2 (strided inputs in the row template) | 2 |
| `template` | the templates cannot compute both in one kernel | phase 2 where it is a row region | 2 |
| `cycle` | one region of both would make a cycle through a call or a move | move the call | 1 |
| `limit` | more than 64 ops or 32 tensors in a region | a split is a speed choice only | 1 |
| `move` | a copy or a store between them | return or store the value where it is made | 1 |
| `epilogue-<kind>`, `prologue-<kind>` | a call's result read by a region, a region's value read by a call: no fusion across a call | library: future work; extern: a hook (phase 4); `@tilefish.kernel`: write its body in the graph body | future, 4, 1 |
| `grouping` | the union is legal, but the greedy merge of version 1 did not make it | a later fusion version | 1 |
| `unfused` | the graph declares `tilefish.fusion(grouping="none")`: one kernel per op | drop it (the default groups them) | 1 |

Without the trace or the graph, the fusion report says which regions are a reduction alone (and whether it has an order) and which read a view
copied first; two regions joined by an intermediate are then `unknown-boundary`.

## 5. Ranking

Each candidate's saving is computed at every bucket analysed and ranked at one: the bucket of the timing file, the one `--bucket` names, or the one
whose steps move the most bytes.

* **Without a timing file**: the bytes saved over the bandwidth, `S / BW`, counting the bytes of every step but a compute-bound library step (a GEMM
  that is compute-bound does not get faster for writing less). The launches it removes are listed, not counted.
* **With a timing file** (the steps' *net* times, section 6): the **launch floor** `t0` is the shortest kernel step, what a kernel costs whatever it
  moves. A step spends `t - t0` moving its bytes at its achieved bandwidth `min(BW, bytes / (t - t0))`. Where that can be its bytes' time (an
  elementwise, row, fused or copy step, a GEMM, a kernel without MMAs) and it achieves a quarter of the bandwidth or more (`STREAMING`), it saves
  `min(t - t0, S_step / achieved)`; any other step (an attention, a kernel whose MMAs are not known, a kernel far slower than its bytes) saves
  `S_step / BW`, and a compute-bound GEMM nothing. Each kernel the fusion removes saves `t0`: one fewer than the candidate's kernel steps, or all of
  them when a library op takes them in. The saving is at most the candidate's time less one floor. Candidates overlap (an elementwise step between
  two GEMMs is an epilogue of one and a prologue of the other), so their savings do not add up.

`patterns` sums the candidates by the blocker of the latest phase each needs removed (each candidate counted once; the others it needs are listed
as `also`): what that phase of fusion would unlock. A candidate whose latest blocker is `epilogue-library` needs GEMM epilogues whatever else is done.
`recurring` sums the candidates whose steps run the same ops in the same order (a layer of a model, repeated): the fusion patterns, in the model's
terms, that would save the most.

The **time by class** sums the steps' times (measured, and the roofline estimate) per class: `library`, `fused`, `elementwise`, `row`, `copy`,
`opaque`; **the time by op** per manifest op.

**The JSON** (`--json`, `Advice.doc`, `tilefish.advice.v1`): `graph`, `arch`, `plan_digest`, `bucket` (ranked at), `buckets` (analysed), `machine`,
`timing`, `estimate`, `steps` (per step: `class`, `label`, `bytes_read`, `bytes_written`, `flops`, `intensity`, `bound`, `est_us`, and `us` and
`achieved_GBps` when measured), `by_class`, `by_op`, `candidates` (each: `shape`, `steps`, `labels`, `classes`, `buffers`, `saved_bytes` and
`saved_us` per bucket, `launches_removed`, `blockers`, `why`, `actions`, `phase`, `gate`, `rank`), `patterns`, `recurring`, `blocked`, `fan_outs`,
`kept`, `unfused_arithmetic`. Informative, like the fusion report: no host reads it.

## 6. The profiler

`tilefish.profile_plan(graph, plan, ...)` runs the plan of a graph at one bucket as `tilefish.run` runs it (the same verified plan and binaries, the
same arena and its order) and times every step on the device. It needs the graph, as every Python host does (the plan is checked against it), so
the command line imports it (`--load MODULE:ATTR`). The inputs and weights are random unless `externals=` gives them (the times depend on values only
where a kernel branches on them); every device scalar must be given (`devscalars=`, `--devscalar`): a count changes what the steps do.

* **`method="graph"`** (the default): the steps are captured once in a CUDA graph with an event record node (`CU_EVENT_RECORD_EXTERNAL`) before the
  first step and after each, and replayed `reps` times after `warmup`: how a host that captures a plan runs it. A step's time is the time between
  the nodes around it. The plan is captured again without the events and timed as a whole.
* **`method="events"`**: the steps are launched as a run launches them, with an event recorded on the stream before the first and after each; a
  spinning kernel first keeps the device busy while the host enqueues them (about `20 n + 50` microseconds for `n` steps), so that they run back to
  back and not as fast as the host launches them. For plans of up to about a thousand steps (the driver's queue).

**The timing file** (`tilefish.step-timing.v1`): `graph`, `plan_digest`, `manifest_digest`, `arch`, `bucket`, `method`, `reps`, `warmup`,
`device` (`name`, `arch`, `sm_count`, `clock_khz`, `copy_GBps`), `steps` (per step: `index`, `kind`, `entry`, `us`: `median`, `mean`, `trimmed`
(the mean of the middle 80%), `min`, `p10`, `p90`, `net`), `sum_us` (of the trimmed means), `plan_us` (the plan without the events), `event_us`
and `overhead` (`sum_us / plan_us - 1`). The advisor refuses a file without a `plan_digest` or of another plan, graph or bucket, one that does not
time every step of the bucket exactly once, a time that is not a finite number of microseconds, and a device number that is not positive.

**What it changes: nothing that outlives it.** It is a runtime path of its own: no manifest, plan, plan set or digest names it, it writes no file of
the export, and the runs it makes are runs of the plan. What those runs change that would outlive them, the caller's `InOut` tensors and the carried
buffers the plan writes, is copied aside (in the arena's order) before the first run and back after the last, and what the arena notes of each
carried buffer is as before: a run after the profile writes the bytes a run before it wrote, and finds the state it would have found
(`test_gpu_advise.py`). A plan that reads a carried buffer before it writes it is refused, as a run of it alone is (it runs on a shared arena
after the plan that writes it, which the profiler does not set up).

**Overhead and accuracy** (measured on a B300 MIG slice, sm_103, 30 SMs, on the plans of `plan_model` in `python/tests`, with Nsight Systems
2026.5.2 on the same graph replays, `--cuda-graph-trace=node`):

* **The events cost each step about 3 us** on this device (3 to 4.5 us on larger plans): the 9 steps of `plan_model.block` at `M = 128` sum to
  94 us between events, and the plan alone takes 64 us, in either method. An event node waits for the step before it to finish, where consecutive
  kernels of a graph overlap one's tail with the next one's launch. So the file records that cost (`event_us`: the steps' sum less the plan's time,
  shared evenly) and each step's **`net`**: its trimmed mean less it. The nets add up to the plan's time, and the advisor ranks by them.
* **The timestamps tick in steps** (the medians of short steps are 4.096 and 6.144 us), which a median snaps to and a mean averages over: hence the
  trimmed mean.
* **Against nsys** (200 replays of the plan's CUDA graph, without events, under `nsys profile --trace=cuda --cuda-graph-trace=node`; the
  profiler's 200 replays with events; microseconds):

  | step of `plan_model.block`, `M = 128` | profiler median | profiler net | nsys kernel | nsys kernel + gap before it |
  |---|---|---|---|---|
  | `rmsnorm` | 4.70 | 1.72 | 1.35 | 1.35 |
  | `gemm_qkv` (cuBLASLt) | 6.18 | 2.91 | 2.50 | 2.54 |
  | `attention` | 49.12 | 46.00 | 46.52 | 46.54 |
  | `gemm_o` | 5.98 | 2.74 | 2.22 | 2.25 |
  | `add_rmsnorm` | 4.29 | 1.28 | 1.20 | 1.23 |
  | `gemm_up` | 6.14 | 2.88 | 2.52 | 2.56 |
  | `gelu` | 6.14 | 2.87 | 1.65 | 1.68 |
  | `gemm_down` | 6.14 | 2.88 | 3.04 | 3.08 |
  | `add` | 4.10 | 0.83 | 1.08 | 1.09 |
  | **sum** | 92.8 | **64.1** | 62.1 | **62.3** (first kernel to last: 62.3) |

  The plan alone, timed by the profiler without events: 64.1 us. For `plan_model.deep_block(4)` (36 steps) the nets sum to 244.2 us and nsys's
  kernels and gaps to 249.4 us. **A step's net is within 0.6 us of nsys's kernel time but for one step (1.2 us), and the nets' sum within 3% of
  nsys's**; the medians, with the events in them, are 3 to 4 us too long each. Rank short steps by `net`, and take a difference of less than a
  microsecond between two steps as noise.
* **What it cannot see**: work inside a step (a library step of several kernels is one step), and the overlap of a step with the next one, which
  `net` shares evenly instead of measuring. For a time inside a kernel, see issue #5 section 2.

## 7. How it is tested

* `python/tests/test_advise.py` (any machine): synthetic plans (bytes, FLOPs, classes and roofline; a pair, a prologue and an epilogue around a
  GEMM; chains, groups and fan-outs; returned, carried, `InOut`, tapped and twice-written buffers kept; a GEMM or another kernel in the middle;
  each bucket's entries and views; views of one buffer; the trace's SASS telling a kernel with MMAs from one without; a kernel bound by its arithmetic;
  the ranking by bytes and by measured time, the floor and the achieved bandwidth; timing files of another plan, bucket or step refused); the golden
  plan `block` on the command line (its residual, blocked by the MLP's GEMMs); the boundaries of a graph body (a barrier, unordered and ordered sums, calls, a materialised view) with their fixes, and that they change no
  region; a traced export by the fake compiler (its manifest, plan and fusion report the same bytes as without the trace; the trace's boundaries; the
  advice with the trace, without it, and with `gop=`); the profiler's runtime path on a fake backend (the events around each step, the same launches
  as a run, both methods).
* `python/tests/test_gpu_advise.py` (a device; arch-agnostic: ran on sm_103, the sm_120 and sm_90 runs are a later gate's): every step timed,
  the export's files and a run's bytes unchanged by profiling, the nets adding up to the plan's time, both methods agreeing, the command line
  (`--profile`, then `--advise --timing`).
