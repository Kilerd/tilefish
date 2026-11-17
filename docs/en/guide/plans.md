# Launch plans: a graph as a static list of launches over one arena

**English** | [简体中文](../../zh-CN/guide/plans.md)

A model forward is not one kernel. `tilefish.plan(graph, buckets=...)` compiles a `@tilefish.graph` into a **launch plan**: for every combination of the bucket values of its
dimensions, the ordered launches of manifest entries (a lowered kernel, a hand-written Triton extern, a prebuilt binary or a pinned cuBLASLt GEMM), with every pointer argument bound to a
buffer or to a *view* of one (a `reshape`, `permute`, `select` or `split` copies nothing), and **one arena of device memory** for everything the caller does not provide: the intermediates, one
slot per `DevScalar` input and the cuBLASLt workspace. Every arena buffer has a fixed byte offset, a multiple of 256, **the same in every bucket** (the arena is sized for the largest), so a plan
can be captured in a CUDA graph once per bucket at fixed addresses. The steps run on one stream in the order of the IR, with no synchronization between them and none implied (an in-place call
keeps the storage of the version it consumes). `reuse=True` lets intermediates share bytes where no step uses both: the transformer block of the gate (rmsnorm, a bias GEMM whose output is split
into q, k and v views, an attention that masks the keys at a device scalar `n`, a GEMM, a residual add and rmsnorm, two GEMMs around a gelu, a residual add; rows bucketed over 32, 64 and 128)
needs 9 634 048 bytes without reuse (8 519 680 of them the workspace) and 9 240 832 with it; unrolled 24 times (216 launches per bucket, the weights of the layers views of stacked
parameters) the arena is 36 765 952 bytes without reuse and 9 306 368 with it, one layer's intermediates plus the workspace whatever the depth.

```python
plan = tilefish.plan(block, buckets={"M": [32, 64, 128]}, reuse=False)       # no GPU, no torch: steps, buffers, offsets
p = tilefish.runtime.export(block, "build/")                                  # binaries, manifest.json and plan.json (schema "tilefish.plan.v1"); p.digest is the pin
y = tilefish.run(block, x, n, *weights)                                       # jit: the same plan, binaries built in memory
y = tilefish.run(block, x, n, *weights, mode="replay", plan="build/", pin=p.digest)   # a verified plan directory, or trust_directory=True
y = tilefish.run(block, x, n, *weights, mode="reference")                     # each op's reference composed in torch
cap = tilefish.capture(block, x, *weights, devscalars={"n": n}, outputs=y)    # the plan in a CUDA graph of its own
cap.set_devscalar("n", 17); cap.replay()                                      # a replay ordered with every run of the plan, from any thread
```

`plan.json` carries the plan and the digest of the manifest it was built against, so **pinning the plan digest pins every binary and every pinned GEMM algorithm it launches**. A replay loads and
verifies the file and its manifest (a tampered plan, a wrong pin, a plan built against another manifest, and a plan of a graph that changed are all refused), allocates the arena once per device
and keeps it, and checks **every step's preconditions before the first launch**: a failure launches nothing, not even the write of a device scalar. A run allocates nothing but its outputs
(`outputs=` gives them) and waits for nothing: the GEMM workspace is in the arena, a device scalar is written by a stream-ordered copy. So **a replay under `torch.cuda.graph` captures and the
graph replays byte-identically** (`runtime.prepare` does the allocating and the loading that cannot happen inside a capture; `docs/en/plan-schema-1.md` lists every reason a plan could fail to
capture). Every run of a plan on a device uses the one arena, so **runs of one plan on several streams (or threads) are ordered on the device** by an event of the arena: a run waits on its
stream for the run before it and records the event after its last launch, so they never overlap and their results are correct, at the cost of running one after the other
(a run on the stream of the event's last record skips the wait: runs that stay on one stream cost one `cuEventRecord` each).
Runs that must run concurrently need an arena each: `lane = tilefish.runtime.share_arena([graph], name="lane1")` is another arena of the same plan
(its own memory, lock and event; `lane.run(graph, x)` on its stream does not wait for the runs on the first), one per stream (Rust: one instantiation per lane).
`tilefish.capture(graph, *inputs, devscalars=..., outputs=...)` captures the plan in a CUDA graph of its own: its `replay()` takes the arena's lock and waits for and records that event
around the graph, as a run does, so **its replays are ordered with every run and replay of the plan, from any thread and on any stream**, and `set_devscalar` changes a device scalar for
the next replays with no synchronization. A plan run inside your own `torch.cuda.graph(...)` captures too: the wait and the record are external event nodes of the graph, so its arena is
the plan's arena and its replays are ordered with the runs whose record they see. But torch's `replay()` takes no lock: **a raw graph replayed while another thread is in the middle of a run
of the same plan overlaps that run** (wrong bytes, no error), so order the two yourself, replay with `tilefish.runtime.replay_graph(g)` (which takes the arena's lock around `g.replay()`), or use
`tilefish.capture`. The second thread that uses an arena such a graph holds gets a `TilefishOrderWarning` (an error with `warnings.simplefilter("error", tilefish.TilefishOrderWarning)`).
`jit` and `replay` are one code path over binaries from two places, so they are identical by construction, and the gate checks on a GPU that plan replay, plan jit and the per-op
eager composition of the same ops give the same bytes for every bucket and two values of `n`.

**An `Out` written over its input.** A hand-written kernel `k(x, r, y, r_out)` that reads every element of `r` before it writes the same element of `r_out`
declares `r_out: Out[Tensor[M, D, f32], MayAlias("r")]`: a launch (Python or Rust) then accepts `r_out` and `r` that are exactly one tensor and refuses any
other overlap, and a plan writes `r_out` over `r` when nothing reads that `r` after the call (a whole intermediate, no later reader, not returned), with no
buffer of its own; otherwise the result gets one, so the values are the same either way (manifest schema 3, section 5.4; `docs/en/plan-schema-1.md`, section 8).

**Copy steps.** `r = tilefish.copy(x)` in a graph is a step of kind `copy`: the dense bytes of `x` (a tensor, or a dense view such as `x[1]`) copied into a
buffer of its own with `cuMemcpyDtoDAsync`, a memcpy node in a CUDA graph. The copy is an intermediate an in-place call may update, where `x`, the caller's,
is never written. `state[...] = r` into an `InOut` parameter is a copy step into the caller's tensor. A copy of a view that is not dense is a launch of
`tilefish.contiguous` instead (a strided view of rank 1 to 4 into a dense tensor), as the
plan is made, and so is a store of one: `state[...] = t.permute(1, 0)` writes the transpose into the caller's tensor (`docs/en/plan-schema-1.md`, section 7.5).

**Memset steps.** `acc = tilefish.zeros_like(x)` (or `zeros_like(x, dtype=i32)`) in a graph is a step of kind `memset`: a buffer of its own whose bytes are
set to zero with `cuMemsetD8Async` (a memset node in a CUDA graph), which an in-place call may then accumulate into, or which the graph returns (the memset
writes your output). `state[...] = tilefish.zeros_like(state)` is one memset of your `InOut` tensor. No kernel has to zero a buffer any more
(`docs/en/plan-schema-1.md`, section 7.6).

**Returned views.** A graph may return a reshape of a result at no cost (`return y.reshape(M, N)` of a flat `y`: the caller's output is `y`'s buffer, in the
returned shape, and the step that writes it binds it through a flat view), and any other dense view, a parameter or a value twice, which a copy step after
the last call puts into the caller's output. A view that is not dense (a transpose, a column block) is materialised there by a launch of
`tilefish.contiguous`.

**Named outputs.** `@tilefish.graph(outputs=("y", "residual"))` names the returned values: the plan's output buffers have those names (else `out0`,
`out1`, ...), `tilefish.run(..., outputs={"y": y, "residual": r})` takes them by name, and the Rust and C++ hosts bind them by name; all three hosts also take `out<i>` for
the `i`-th returned value, so a host written for the names of before keeps working.

**Inputs and weights.** `@tilefish.graph(inputs=("x",))` (or `weights=(...)`, or both) says which parameters a call gives every time and which a
host binds once, when it loads the model. The plan records the roles (`"roles": "declared"`), and all three hosts bind the weights once:
`w = tilefish.runtime.bind_weights(graph, {"w1": t1, ...}, mode="replay", plan="build/", pin=p.digest)`, then `w.run(x)` and `w.capture(x, ...)`
(Rust: `PlanRun::bind_weights`, a `WeightedRun` whose `launch` and `capture` take the inputs only). Without the declaration a plan labels a
parameter by its shape, as before (`weight` when it mentions no dimension: every parameter of a fully static graph), and `bind_weights` refuses it.

**Plan sets.** A stage whose launch count depends on a key is one plan per key; `tilefish.runtime.export([g1, g2, ...], "build/")` exports them as a
*plan set*: one manifest (each binary once, whichever plans launch it), `<graph>.plan.json` per graph and `plans.json`, an index whose digest pins every
plan, their manifest and every binary. A replay takes `plan="build/"` and the set's digest or the plan's; Rust `PlanSet::load(dir, pin)` and
`set.plan("g2")` (the plans share the manifest and its loaded modules). `docs/en/plan-schema-1.md`, section 15, says why a set and not a repeated step range.

**An arena shared by the plans of a set.** A host that runs plans of a set one at a time (a request's stage A, then B, then C) can give them one
arena: `s = tilefish.runtime.share_arena([a, b, c], mode="replay", plan="build/", pin=planset.digest)`, then `s.run(a, x)`, `s.capture(b, ...)`,
`s.bind_weights(c, {...})`. The arena has the bytes of the largest plan's arena (the gate's transformer block cut into three stages: 9 109 504 bytes
instead of 26 476 800), each plan's buffers at their own offsets in it, so the documents and their pins are unchanged and every plan captures at
fixed addresses. Every run, capture and replay of any of the plans is ordered with all the others (one lock, one order event), so two of them never
overlap on the device; nothing in the plans' own regions outlives a run, and a tensor of yours anywhere in the arena is refused. Rust:
`set.instantiate(&stream, &["a", "b", "c"])`, a `SetRun` of `PlanRun`s on one `SharedArena`. `share_arena(..., name="lane1")` is another arena of the
same plans, for a lane that runs concurrently. `docs/en/plan-schema-1.md`, section 15.1.

**Carried buffers.** A stage can hand an intermediate to the next with no tensor of yours: `@tilefish.graph(carried=("h",))` on both graphs makes the
parameter `h` a buffer the arena holds (role `carried`), in a region after the plans' own regions, the same for every plan of the arena that names it.
`stage_a(x, h: InOut[...])` writes it (`h[...] = encode(x)`: the launch of `encode` writes the arena's `h` directly), `stage_b(h)` reads it, and a run gives
every parameter but the carried ones: `s.run(stage_a, x)`, then `s.run(stage_b)`, whose sizes are the ones `h` was written with. A run that reads `h` before
any run on the arena wrote it, or in another shape, is refused before anything is enqueued; the captures of `tilefish.capture` check and note `h` at every
replay. All three hosts compute the same layout from the documents, which the pins cover (`docs/en/plan-schema-1.md`, section 15.2).

**Debug taps.** `tilefish.run(graph, ..., taps={"t3": t, ("t5", 4): u})` copies buffers of the plan into your tensors right after a step (by default the
last one that writes the buffer), before a later step may reuse their bytes: the intermediates of the very plan a product runs, with the plan and its digest
unchanged (`plan.shape_of("t3", M=64)` gives the shape; `tilefish.capture` takes `taps=` too; Rust: `PlanRun::launch_tapped`, and
`PlanRun::capture_tapped`, whose replays copy the taps again).

**One step alone.** `tilefish.runtime.run_step(graph, 3, {"t2": a, "w_o": w, "t3": o}, mode="replay", plan="build/", pin=p.digest)` launches one step of the
plan on your tensors, with the binary and the binding of its arguments that the pinned plan says (Rust: `PlanRun::launch_step`): one op of an exported plan
replayed on a reference's inputs, under the pin a product uses.

A plan launches declared operators, views and the arithmetic of the body as fused regions ([`fusion.md`](fusion.md)). An `Out` parameter, a call to another graph, an in-place call on a view
and a view read after its tensor was updated in place are refused with the rule that says so. `docs/en/plan-schema-1.md` specifies the file exactly (every field, the validation
rules, the digest) for the host that will run it without Python.
