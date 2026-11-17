# Intra-kernel instrumentation: probes, the instrumented variant, timelines

**English** | [简体中文](../zh-CN/instrument.md)

A sampling profiler attributes the time of an asynchronous `wgmma` or `tcgen05` MMA to the instructions that wait for it, and a warp-specialised or
persistent kernel is one long launch to a timer outside it (`tilefish.profile_plan` times steps, `docs/en/advise.md`). What goes on inside such a kernel
is best read from the kernel itself: timestamps taken where its author says a phase begins and ends. tilefish does that with

1. **probes** (section 1): `tilefish.probe("name")` in a `@tilefish.kernel` body, a `@triton.jit` or a `@gluon.jit` kernel. A production build
   compiles them to nothing;
2. **an instrumented variant** (section 2): `export(..., instrument=True)`, or `tilefish.run(..., probes=tilefish.Probes())` in `jit`, builds every
   kernel with its probes, and with three of tilefish's own (`@begin`, `@calib`, `@end`), into binaries, a manifest digest and plan digests of their
   own, which a host refuses unless asked to load them;
3. **records** (section 3): lane 0 of every warp that reaches a probe writes `%globaltimer`, `%clock`, the probe and `%smid` into its slot of a probe
   buffer the host gives in the kernel's profile scratch parameter;
4. **host tooling** (section 4): Python and Rust collect the buffer after a launch, a plan run or a captured replay; `tilefish-explain DIR --probes FILE`
   renders the phases with their durations and distributions per warp role, timelines per SM, CTA and role, and a Chrome trace (Perfetto), with the
   source line of every probe.

The code is `tilefish._probe` (the probes, the PTX rewrite, the buffer), `tilefish._probes` (`Probes`, `ProbeTrace`, the views) and, in the crate,
`crates/tilefish/src/probes.rs`. Section 5 says why it is not Triton's Proton, section 6 what it costs, section 7 its limits.

## 1. Probes

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

* **Triton and Gluon kernels** (externs) call `tilefish.probe` like any builtin: `import tilefish` in the kernel's module and write it in the kernel or
  in a jit function it calls (which Triton inlines). It is a Python function that Triton's code generator calls with its builder (a builtin, as
  `tl.load` is); importing tilefish imports neither Triton nor torch.
* **A `@tilefish.kernel` body** marks the point between the ops written before it and after it. The lowering writes `_tf_probe("scaled")` there in the
  generated source (with the line of the body in a comment and in the trace's `nodes.json`); its templates load the inputs first and store the results
  last, so a probe is after the loads and before the stores. A graph body has no probes (it has no kernel of its own; every step of an instrumented
  plan records `@begin` and `@end`).
* **A name** is 1 to 64 characters, a letter or `_` first and then letters, digits and `_ . : / -` (`load.wait`, `mma/issue`). One name may be written
  at several places (two branches, two partitions): they are one probe.
* **A phase** is the time between two consecutive records of one warp, named `"a -> b"`. A probe records when its warp *issues* it: to time an
  asynchronous copy or MMA, put probes around the wait for it (`mbarrier.wait`), as above, not around its issue.

**In a production build a probe is nothing.** Triton's frontend emits no operation for it, so the IR, its digest and the machine code are those of
the kernel without the call (on sm_103 the production binary of `probe_kernels.ws_scale_p` has the `sha256_code` of `ws_kernels.ws_scale`, the same
kernel written without probes); only the line table moves where source lines moved. An export of kernels without probes is the same files as before
this feature (checked on the whole Rust gate's export, section 8).

**Enabling it on an extern you write** (a private SIMT attention, say):

1. put `tilefish.probe("...")` at the boundaries you care about (around the waits of a pipeline, at the start and end of a tile of a persistent loop);
2. iterate in `jit`: `rec = tilefish.Probes(); tilefish.run(attn, q, k, v, probes=rec); print(rec.trace().summary_text())`, or export the variant
   next to the production one, `tilefish.runtime.export(ops, "build/inst", instrument=True, trace=True)`, and replay it with `probes=rec` (Python) or
   `Manifest::load_instrumented` (Rust);
3. `rec.trace().save("attn.json")`, then `tilefish-explain build/inst --probes attn.json` (add `--timeline role`, `--chrome attn.trace.json`).

Nothing else changes: the declaration, the ABI, the plan and the production export are the same.

## 2. The instrumented variant

```python
runtime.export([attn, mm], "build/inst", instrument=True)                          # every Triton, Gluon and lowered binary
runtime.export(block, "build/inst", instrument=tilefish.instrument(capacity=1024, ops=["attn"]))   # a graph; only attn's binaries
rec = tilefish.Probes(tilefish.instrument(capacity=512)); tilefish.run(op, *args, probes=rec)      # jit: built in memory
```

`instrument=True` is `tilefish.instrument()`: `capacity` records per warp and launch (default 256), `ops` the operators to instrument (default all;
prebuilt PTX and cubin externs and library ops are never instrumented). The build of each binary:

* runs the frontend with the probes on, so the IR has them (an inline assembly statement each) and its digest is of that IR;
* rewrites the PTX of the entry between Triton's `ptx` and `cubin` stages (a wrapper of `CUDABackend.add_stages`, active only in a thread that builds the
  variant): the registers of the probes and the warp's slot of the buffer first, then `@begin`; in warp 0 of each CTA `@calib` twice, from the same
  instructions; and `@end` with the count of the warp's records before every `ret` (a predicated `ret` branches around it);
* is compiled every time: the variant never reads nor writes the build cache, so an instrumented binary is never served for a production build, nor the
  other way round. The TF32 guard and the contraction check judge it; the declaration's SASS assertions do not (they are the production binary's: the
  probes add a few registers and instructions).

**Its own key and digest.** Its entries carry `instrument` (manifest schema 3, section 4.8: the record format, the capacity and the table of the
kernel's probes: id, name, and per site the module-relative line, the partition it runs in and the jit function); their binaries, `sha256`,
`sha256_code` and `source_digest` (for a kernel with probes) are its own, so the manifest digest and the digest of every plan and plan set of the
variant are too. The declaration digest and the ABI are the production entry's: the probe buffer is the profile scratch argument every Triton kernel
already has (the second of its two scratch slots, null in a production launch).

**It is never taken for the production artefact.**

* Python: `Manifest.load` refuses a manifest with an instrumented entry unless `instrumented=True`; `tilefish.run(..., mode="replay")` and
  `tilefish.capture` refuse it unless `probes=` is given (and refuse `probes=` with a production export); the plan of an instrumented export likewise;
  `tilefish.hosts.rust.emit` and `tilefish.hosts.cpp.emit` refuse it (they write production hosts).
* Rust: `Manifest::load` and `Manifest::from_embedded` refuse it; `Manifest::load_instrumented` takes it (and `LaunchPlan` likewise).
* An export never merges the two: writing the variant into a directory that holds a production manifest (or the other way round) is refused.
* Readers that predate it: the crate 0.0.8 and earlier refuse the entry field (they refuse every unknown field); the replay of tilefish 0.0.10 and
  earlier refuses a kernel with probes (its IR digest is not the one the process computes). A kernel without probes would load there; launched with the null profile scratch an
  older host gives it, every probe skips its store (the slot pointer is null), so it runs and writes the production bytes.

## 3. The probe buffer and the records

The buffer of a launch has one slot per warp of every CTA, `ctas * warps * (16 + 16 * capacity)` bytes, `ctas` the CUDA grid launched (a cluster's CTAs
counted, a persistent grid as cut to the device), `warps` the block's (`total_num_warps` of a warp-specialised kernel), 16-byte aligned. The slot of warp
`w` of the CTA of linear index `c = x + y * nx + z * nx * ny` is at `(c * warps + w) * (16 + 16 * capacity)`:

| offset in the slot | type | what |
|---|---|---|
| 0 | u32 | `0x31706674` (the bytes `tfp1`), written at the warp's entry |
| 4 | u32 | the CTA's linear index |
| 8 | u32 | the warp (`%tid.x / 32`) |
| 12 | u32 | the records the warp attempted, written at its `@end` (more than the capacity: the rest were dropped) |
| 16 + 16 k | u64 | record `k`: `%globaltimer` (ns) |
| 24 + 16 k | u32 | `%clock`, the SM's cycle counter, low 32 bits |
| 28 + 16 k | u16 | the probe: 0 `@begin`, 1 `@end`, 2 `@calib`, 3 and up the kernel's in the order the frontend met them |
| 30 + 16 k | u16 | `%smid` |

Lane 0 of each warp writes; a warp keeps its first `capacity` records and counts the others (`tilefish.instrument(capacity=...)` keeps more). A null
buffer makes every probe skip its store. A slot whose header is not intact (a warp that did not run, a buffer of another kernel) is listed as missing.

**Where the buffer comes from.** A single launch: `Probes` allocates one per launch, or records into the caller's (`tilefish.Probes(buffer=t)`, a CUDA
`uint8` tensor). A plan: each bucket of an instrumented plan has a probe region, allocated with the bucket's binaries outside any capture as the arena
is, in which every instrumented step has a 256-byte aligned slice sized for its grid at the bucket; a captured plan holds its address, so every replay
records into it; a caller's buffer takes its place when one is given. The region is the host's, not a buffer of `plan.json`: its size depends on the
device (a persistent grid has one program per SM) and on the variant, the plan schema stays as it is, and a `jit` run, which has no `plan.json`, uses the
same layout. The Rust host does the same (section 4).

## 4. Collecting and explaining

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

After each launch (or `collect()`) the buffer is read back, which waits for the stream, and decoded into a `Launch`: the stem, architecture, op, grid
and block, the plan's graph, step and bucket, the manifest's digest, the entry's probe table, every warp's records and the missing slots. The file
`tilefish.probe-trace.v1` holds the launches (times from the launch's first `%globaltimer`, which is a decimal string: it does not fit a JSON number);
the Rust host writes the same file.

**Rust** (`crates/tilefish/src/probes.rs`):

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

`Manifest::is_instrumented`, `Entry::instrument` (`Instrument { capacity, probes }`), `Kernel::launch_probed_with` (given arguments),
`LaunchPlan::instantiate_all_probed`, `PlanSet::load_instrumented` and `PlanRun::probed` complete it; a production binary, a buffer of another context
and a buffer too small are refused, and `instantiate_probed` refuses the plan of a production export. The C ABI (`tilefish-capi`) has no probe calls:
it loads through `Manifest::load`, so it refuses an instrumented export.

```console
tilefish-explain DIR --probes FILE [KERNEL]           # the phases of each launch (of KERNEL: a stem or an op)
  --timeline sm | cta | role                          # per SM its CTAs; per CTA every warp's phases; per role one warp each
  --chrome OUT                                        # a Chrome trace: chrome://tracing, ui.perfetto.dev
  --json --limit N --source-root PATH
```

The summary of the producer and consumer of `probe_kernels.ws_scale_p` (sm_103, 4 CTAs of 8 warps; `tilefish-explain build/inst --probes ws.json`):

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

and `--timeline role --limit 1`, one warp of each role of the first CTA:

```
  CTA 0 on SM 28: 0.000-17.248
    warp  0 default     0.064: compute.wait +444 ns | compute.work +280 ns | compute.wait +207 ns | compute.work +54 ns | ...
    warp  4 -           0.000: @end +17.21 us
    warp  6 worker 0    0.000: load.wait +490 ns | load.issue +47 ns | load.wait +51 ns | load.issue +42 ns | ...
```

The producer waits 197 ns of its 253 ns per tile for a free slot, the consumer 53 ns of its 255: the pipeline is bound by the consumer (its work and its
TMA store), and the two go round at one rate, as a pipeline in its steady state does.

* **Durations** are cycles of `%clock` converted at the clock rate the launch measured (the median over its warps of cycles per nanosecond of
  `%globaltimer` over spans of 2 us or more); **starts** are `%globaltimer`, one axis for every SM. `@calib` gives what one probe costs a warp in cycles
  (the second of two back-to-back records, its code in the instruction cache); "the probes' share of a warp's time" is the records times that cost over
  the warp's span, an upper bound (a probe in a warp that waits anyway costs less: section 6).
* **Roles** are the partitions of a Gluon `gl.warp_specialize`, from the frontend's IR: `default` and `worker N` (with the jit function each runs); a
  probe outside the partitions runs on the default warps; a kernel that is not partitioned has one role, `""`. A warp's role is the role of the probes it
  recorded; a warp that recorded none of the kernel's probes (a partition without probes, or a warp Triton adds to round the partitions up to warp
  groups: `ws_scale` of 4 + 1 warps is launched with 8, its producer on warp 6) is `-` and is not in the phases.
* **Source lines**: every probe of the table has its module-relative line (`kernels/attn.py:88`, `tilefish_generated/scale_add.py:26`); with the
  export's trace (`trace=True`, `docs/en/explain.md`) or `--source-root` the summary prints the line, and for a lowered kernel the line of the body it was
  written on.

## 5. Why not Proton

Triton 3.5.1 has Proton's intra-kernel instrumentation (`triton.profiler.language.enter_scope` / `exit_scope`, `proton.start(backend="instrumentation")`).
It was evaluated on the same kernels before this was written:

* it keeps its records in **shared memory** and copies them out at the end of each CTA: the plain loop kernel gained 8 KB of shared memory, the tcgen05
  matmul of `ws_kernels` went from 131 260 to 196 800 bytes (the buffer takes what the kernel leaves of the SM), which changes occupancy and fails for a
  kernel near the limit; the copy-out by thread 0 lengthens every CTA;
* its records hold the **SM's clock only** (43 bits), no `%globaltimer`: CTAs on different SMs cannot be put on one axis (its Chrome trace starts each
  CTA at 0 and assumes 1 GHz);
* it is **process-wide state** (a global flag the frontend reads, `CUDABackend.instrumentation`, a patched `JITFunction.run`), which tilefish's build
  refuses for a reason (`_refuse_unidentified`: the binary would not be compiled from the IR that identifies it), and Triton kernels are off by default
  (Gluon only);
* its buffer format is Triton's own, read by C++ in `libproton`: a Rust host would need a port of it, for every Triton release;
* it costs about the same per record (24 ns in a tight loop, against 21 ns for a probe here).

What is taken from Triton: the profile scratch slot (so the ABI and every host's argument list stay as they are), inline assembly, and its stage
pipeline (the PTX stage is wrapped, not replaced).

## 6. What it costs

Measured on a B300 MIG slice (sm_103, 30 SMs, Triton 3.5.1, ptxas 12.9), each kernel in a CUDA graph of one step, captured with and without probes and
replayed 50 times (`test_gpu_instrument.py::test_what_the_probes_cost`, `-s` prints them):

| kernel | production | instrumented | records per warp (at most) | a probe in a warp (`@calib`) |
|---|---|---|---|---|
| `ws_scale_p` (Gluon producer and consumer, 1024 x 4096 f16) | 20.6 us | 19.3 us (-6%) | 132 | 53 cycles |
| `persistent_scale_p` (one program per SM, 8192 tiles) | 144.5 us | 154.8 us (+7%) | 552 | 58 cycles |
| `scale_add_p` (`@tilefish.kernel`, 4096 short CTAs) | 29.5 us | 31.7 us (+8%) | 6 | 77 cycles |
| `ws_matmul_p` (tcgen05, 512 x 512 x 8192 f16) | 28.8 us | 30.8 us (+7%) | 259 | 58 cycles |

* **A probe** is about ten SASS instructions (`CS2R` of `%globaltimer` and `%clock`, `S2R` of `%smid`, the address, a predicated `STG.128` from lane
  0, the count) and costs a warp 45 to 80 cycles of latency (its store waits for the special registers); where the warp waits anyway (behind an `mbarrier`) it
  costs less, which is why the producer and consumer kernel does not get slower. A short CTA pays for the fixed part (the slot, `@begin`, `@end` and
  the count; `@calib` in warp 0 only).
* **Resources**: no shared memory, no stack frame; registers: `scale_add_p` 26 to 30, the others unchanged (the warp-specialised kernels are at their
  `maxnreg`); code: a fifth more (28 040 to 35 688 bytes of cubin for `ws_scale_p`).
* **Memory**: the buffer, `16 + 16 * capacity` bytes per warp (4.1 KB at the default 256).

## 7. Limits

* **Two clocks.** `%clock` counts the cycles of one SM (32 bits kept: it wraps every 2 s at 2 GHz; within a warp the views unwrap it): fine-grained,
  but never compare two SMs' clocks. `%globaltimer` is one clock for the GPU in nanoseconds, of the driver's resolution: steps of 32 ns on a B300 (every
  step a launch recorded was a multiple of 32), so starts are quantised to it; durations come from the clock.
* **What a probe sees** is when its warp issued it, as `ptxas` scheduled the instructions around it (`tilefish-explain` shows the SASS of a probe under
  its line). It does not see when asynchronous work completes: time the waits.
* **Lane 0** of each warp records: a probe on a path lane 0 does not take is not recorded, and a warp that does not reach a `ret` (a trap) writes no
  count, so its records are not read. `%smid` is recorded with every record (a preempted CTA could move).
* **Triton (not Gluon) kernels**: the compiler may move or duplicate code around a probe (software pipelining, its own warp specialisation of a
  `tl.range(..., warp_specialize=True)` loop), and the partitions it makes have no role (the frontend's IR has none): the warps are told apart by id.
* **A probe in a `noinline` jit function** is refused (it records into registers of the kernel's entry).
* **Prebuilt PTX or cubin externs and library ops** (cuBLASLt, cuDNN) are not instrumented: a plan step of one records nothing.
* **Capacity**: a warp keeps its first `capacity` records; the summary says how many were dropped.

## 8. How it is tested

* `python/tests/test_instrument.py` (any machine): the PTX rewrite on a Triton entry (the slot, `@begin`, `@calib`, `@end` before a predicated and a
  plain `ret`; what it refuses), the probe as PTX, the probe table of a warp-specialised kernel's frontend IR (roles, functions, lines), names and
  settings, decoding a buffer (missing slots, dropped records), phases, durations, the cost of a probe and idle warps, the trace file and its views,
  `tilefish-explain --probes`, a kernel body's probes and their lowering (a kernel without probes lowered as before), and the variant with the fake
  compiler (its entry, its binary, never cached, the refusals of `Manifest.load`, replay and the emitters, the merge refusal, `ops=`).
* `python/tests/test_gpu_instrument.py` (a device; ran on sm_103, the sm_120 run is a later gate's, its tcgen05 test skips there): section 6's
  measurements, the production binary of a kernel with probes against the one without (the same `sha256_code`), the outputs of the variant against the
  production bytes (jit and replay), a null buffer, the producer and consumer, the tcgen05 pipeline (its epilogue waits as long as the main loop runs),
  the persistent kernel (each program on an SM of its own, walking its own tiles), the elementwise kernel, a plan (jit, replay of an instrumented
  export, a captured replay) and `tilefish-explain` on a real trace.
* The Rust host's tests: `crates/tilefish/tests/instrument.rs` (any machine: the field read only by `load_instrumented`, every malformed case of
  section 4.8 refused, instrumented copies of the golden plan and plan set refused by `load`), the unit tests of `probes.rs` (the decoder, names, the
  trace file), and the GPU gate's instrumented case (`python/tests/rust_gate_probed.py`, `crates/tilefish/tests/gpu_instrument_gate.rs`): the Rust host
  launches the probed kernels with a probe buffer, without one and with one too small, and the instrumented plan launched, captured and replayed;
  `rust_gate.py compare` checks the bytes against production and the records against Python's, warp by warp (ids, counts, roles).
* The production path: the Rust gate's export of the whole GPU gate by this version with `TILEFISH_TRACE=1`, against the export of 0.0.10 without a
  trace: 3650 files, 184 cubins, 106 manifest, plan and plan-set digests, all the same (the gate's instrumented case is new, in a directory of its own).
