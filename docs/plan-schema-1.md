# Plan schema 1 (`plan.json`, `"tilefish.plan.v1"`)

A *launch plan* is a `@tilefish.graph` compiled into, for every combination of the bucket values of its dimensions, an ordered list of **steps** (each one
launch of a manifest entry: a Triton kernel, a prebuilt PTX or cubin, or a pinned cuBLASLt GEMM) with every pointer argument bound to a **buffer** or a view of one,
and one **memory plan**: every buffer that is not provided by the caller lives in a single **arena** at a fixed byte offset, the same in every bucket. A host
that holds a verified plan and its manifest launches the steps in order on one stream and allocates nothing while it does, so the whole execution can be captured in
a CUDA graph once per bucket at fixed addresses.

The Python package writes and reads this document (`tilefish.plan`, `tilefish.runtime.export`, `tilefish.run(graph, ..., mode="replay", plan=...)`). A Rust lane
implements this specification. Everything is checked by tests in `python/tests/` (named in the last section) and the Python code that writes or checks each part is
named where the part is described (`python/src/tilefish/_plan.py`, `_planrun.py`).

A reader of this document is assumed to know manifest schema 3 (`docs/manifest-schema-3.md`): the entries of `kernels` and `library`, the `abi`, the dimension
expression grammar, the `preconditions`, canonical JSON and the launch checks a host makes for one launch. A plan adds nothing to those: **a step is a launch of
a manifest entry, checked and made exactly as a single launch is; the plan says which entry, with which memory.**

Contents: 1 what a plan is made of; 2 the top level; 3 the digest and the pin; 4 strict reading; 5 dimensions, buckets and shape expressions; 6 buffers and the
arena; 7 steps; 8 order, liveness and reuse; 9 what a host does; 10 CUDA graphs; 11 the Python API; 12 what a plan does not do (version 1); 13 an example; 14 where it
is tested.

## 1. What a plan is made of

* **`dims`**: the dimensions of the graph (the free dimensions of its parameters' shapes), each with the list of sizes planned for it (its *buckets*). A bucket
  combination is one size for every dimension. There is one list of steps per combination, and a call selects one by the sizes of its tensors, which must equal a
  bucket exactly (there is no padding).
* **`buffers`**: every tensor the plan names, in one table. A buffer is *external* (the caller provides it: a parameter or a returned value) or lives in the *arena*
  (an intermediate, the slot of a device scalar, the workspace of the GEMMs). Its dtype and shape are fixed by the graph; the shape is an expression over `dims`.
* **`buckets[i].steps`**: the launches of one bucket combination, in order. A step names a manifest entry (`entry`, whose stem includes the entry's own buckets, such as
  `gemm_up.M64`) and binds each pointer parameter of that entry, in the order of the entry's ABI, to a buffer and, optionally, to a view of it.
* **the memory plan**: `arena_bytes` and the `offset` of every arena buffer, in bytes from the start of the arena, aligned to 256, the same in every bucket.

Views copy nothing. A step that reads the `q`, `k` and `v` of a fused `[M, 3, H, D]` tensor passes three views of one buffer (offsets 0, `H*D` and `2*H*D`
elements, strides `3*H*D`, `D`, `1`), and the kernel is handed three pointers and the strides as scalars, as for any `Strided[...]` pointer (manifest schema 3, section 5.1).

## 2. The top level

```json
{"schema": "tilefish.plan.v1", "tilefish": "0.0.3", "graph": "block", "arch": "sm_103", "manifest_digest": "<64 hex>", "dims": {"M": [32, 64, 128]},
 "reuse": false, "arena_alignment": 256, "arena_bytes": 9634048, "buffers": [ ... ], "outputs": ["out0"], "buckets": [ ... ], "plan_digest": "<64 hex>"}
```

Exactly these keys, no others and none missing (a reader refuses the file otherwise):

| key | type and rule |
|---|---|
| `schema` | the JSON string `"tilefish.plan.v1"`. Any other value is refused ("not a tilefish plan of schema ..."). |
| `tilefish` | string, the version of the package that wrote the file. Informative: **it is not digested**. |
| `graph` | string, the name of the graph: an ASCII identifier (`[A-Za-z_][A-Za-z0-9_]*`). |
| `arch` | string `sm_<2 or 3 digits>`, no suffix: the architecture the entries of the manifest are taken for. A host refuses to run the plan on a device of another architecture. |
| `manifest_digest` | 64 lowercase hex digits: the digest (manifest schema 3, section 3) of the manifest in the same directory. The plan is refused if the manifest has another digest. |
| `dims` | object: dimension name (identifier) -> non-empty, strictly increasing list of integers >= 1 (the buckets). May be `{}` (a graph with no dimension has one bucket combination, the empty one). |
| `reuse` | boolean: whether the memory plan lets buffers share bytes (section 8). |
| `arena_alignment` | the integer `256`. |
| `arena_bytes` | integer >= 0, **equal to** the end of the last arena region rounded up to 256: `align_up(max over arena buffers of offset + align_up(bytes), 256)`, `0` if there is none. |
| `buffers` | non-empty list of buffer objects (section 6). |
| `outputs` | list of buffer names: the buffers of role `output`, each once, **in the order the graph returns them**. |
| `buckets` | list with one entry per bucket combination (section 5), each `{"values": {...}, "steps": [...]}` (section 7). |
| `plan_digest` | 64 lowercase hex digits (section 3). |

**Integers are JSON integers**: a reader refuses `2.0`, `1e3` and `true` where an integer is wanted (Python: `type(v) is int`).

## 3. The digest and the pin

`plan_digest(doc)` (`python/src/tilefish/_plan.py`) is

```
sha256( b"tilefish.plan-digest.v1\0" + canonical_json(body) ),   body = doc without the keys "plan_digest" and "tilefish"
```

* `canonical_json` is exactly the manifest's: object keys sorted (code point order), separators `","` and `":"`, no whitespace, non-ASCII escaped (`ensure_ascii`), no
  `NaN`/`Infinity`. A plan has only strings, integers, booleans and `null` (a plan that was only made in memory has `null` for `arch` and `manifest_digest`; a *file*
  never does). Lists keep their order (the order of buffers, buckets, steps and arguments is part of the plan).
* The digest includes `manifest_digest`, so **pinning the plan digest pins the manifest, and with it every binary and every pinned GEMM algorithm the plan launches.**
* Test vectors (`test_plan_schema.py`): the digest of `{}` is `345611870fcf5832cd7533b55c9bcb061b78c17c4f53635c9bcf8f7cffca6893`; the digest of
  `{"schema": "tilefish.plan.v1", "tilefish": "x", "graph": "g", "plan_digest": "ignored"}` is
  `f59e1096fcfb737bcb745fc0a11258a25a1e2eb2fd3039b8958c777d4dadc71e` (it is the sha256 of the tag and `{"graph":"g","schema":"tilefish.plan.v1"}`).

**A plan needs a trust anchor, as a manifest does.** The digests in a directory sit next to the files they describe, so whoever can write the directory can rewrite
all of them. The anchor is the **pin** (the `plan_digest` the exporter returned, kept somewhere trusted) or an explicit statement that whatever the directory holds is
fine (`trust_directory=True` in Python); a replay with neither, or with both, is refused. The checks, **in this order** (`_plan.read_plan`, `_plan.load_plan`):

1. the file is read as strict JSON (section 4) and has `schema` `"tilefish.plan.v1"`;
2. its digest is computed; with a pin it must equal the pin (else refused: "has digest X, but the pin is Y"); and it must equal the file's own `plan_digest` (else
   refused: "was edited: its plan_digest says ...", which also catches an edit when only the directory is trusted);
3. every rule of sections 2 and 5 to 8 (`validate_doc`);
4. the manifest next to it is loaded and verified as manifest schema 3 says (every cubin's digests), and its digest must equal `manifest_digest`
   (else refused: "built against the manifest with digest ...");
5. every step is what the manifest's entries say (section 7.3, `check_against_manifest`);
6. a host that has the graph (Python does) requires the document to be exactly the plan it derives from the graph for the same `dims` and `reuse`, ignoring `arch`,
   `manifest_digest`, `tilefish` and `plan_digest`: a graph that changed since the export is refused as *stale*. (A host that has no graph, such as the Rust one,
   has steps 1 to 5 and the pin.)

## 4. Strict reading

The file is read as strict JSON, as the manifest is: **no duplicate keys in an object, no `NaN`/`Infinity`, no number that does not fit a double**; a file larger than
256 MiB, nested too deeply, or that is not UTF-8 is refused. An object has exactly the fields listed for it: a missing or an unknown field is a refusal.

## 5. Dimensions, buckets and shape expressions

* `dims` names every dimension that appears in any shape. The **bucket combinations** are the Cartesian product of the lists, with the names sorted (code point order),
  **the first name varying slowest**, each list ascending: for `{"B": [1, 2], "M": [32, 64]}` they are `B1 M32`, `B1 M64`, `B2 M32`, `B2 M64`. `buckets` has exactly one
  entry per combination, in this order, and `buckets[i].values` is the object `{name: size}` of the i-th.
* A **shape entry** (of a buffer) is a JSON integer >= 1 or a string: a *dimension expression*, the grammar of the manifest's shapes: integers, names, `+`, `-`, `*`, `//`
  (exact), unary `-`, parentheses and `cdiv(a, b)`, at most 512 characters. Python writes only affine forms (`"M"`, `"3*M"`, `"2*M + 3"`). It must mention only names in `dims`
  and evaluate to an integer >= 1 for every combination; an expression that is not an affine form that folds to an integer under the combination's values is refused.
* **Selecting a bucket at a call.** The host binds the dimensions from the sizes of the external tensors it is given (each external's `shape` expressions are equations;
  `3*M` against 192 gives `M = 64`, and a size that is not a whole multiple is a refusal), exactly as it binds the dimensions of one launch. Every name in `dims` must be
  bound, and its value must be in its list; else the call is refused with the kind `bucket` and nothing runs. Two externals that bind a name differently are refused.

## 6. Buffers and the arena

`buffers` is a list of objects with exactly the fields `name role dtype shape offset bytes`:

| field | type and rule |
|---|---|
| `name` | string, an identifier, unique in the plan. The graph's parameters keep their own names; the returned values are `out0`, `out1`, ... (in the order the graph returns them), an intermediate `t0`, `t1`, ... in the order the steps make them, the workspace `workspace`. A name that would collide with a parameter's gets underscores. |
| `role` | one of the seven below. |
| `dtype` | one of `bf16 f16 f32 i32 i64 u8 e4m3 e5m2` (the dtypes that have a storage type). |
| `shape` | list of integers >= 1 and dimension expressions (section 5); `[]` for a device scalar. |
| `offset` | `null` for an external role; for an arena role an integer >= 0 that is a multiple of 256. |
| `bytes` | integer >= 1, **exactly** the largest of `product(shape) * itemsize(dtype)` over the bucket combinations. |

| `role` | what it is | who provides it | a step may |
|---|---|---|---|
| `input` | a graph parameter whose shape mentions a dimension (an activation) | the caller | read |
| `weight` | a graph parameter whose shape mentions no dimension | the caller | read |
| `inout` | an `InOut` parameter: updated in place | the caller | read and write |
| `output` | a returned value (a result of a call) | the caller (allocated by the Python `run` unless `outputs=` gives it) | write (first), then read and write |
| `intermediate` | a result that is not returned | the arena | write (first), then read and write |
| `devscalar` | a `DevScalar` parameter: one element in device memory (`DevScalar[i32]`: a token count) | the host writes the value into its slot before the first launch | read |
| `workspace` | the cuBLASLt workspace: dtype `u8`, shape `[N]` with a static `N` | the arena | (only as the `workspace` of a library step) |

* The buffers listed first are the graph's parameters, in the order of its signature; they have the roles `input`, `weight`, `inout` and `devscalar`. The rest are made by the
  steps, then the workspace.
* A `devscalar` has shape `[]` (one element). There is at most one `workspace`, and it exists exactly when some step has kind `library`; its `bytes` is the
  largest `workspace_bytes` of the library entries the plan uses (a reader requires it to be at least each one's: section 7.3).
* **The arena** is `arena_bytes` bytes of device memory, allocated once by the host and **aligned to 256 bytes**. The region of an arena buffer is
  `[offset, offset + align_up(bytes, 256))`, which must lie inside the arena. A buffer's region is large enough for every bucket: a smaller bucket uses a prefix of it, so a
  buffer is at the same address in every bucket. Without `reuse` (section 8) no two regions overlap.
* An external's `bytes` is informative for the host's own allocation (the largest bucket); at a call its size is the bucket's `product(shape) * itemsize`, which the host checks.

## 7. Steps

### 7.1 The fields

`buckets[i].steps` is a list (the same length in every bucket) of objects with the fields `entry kind args` and, for kind `library` only, `workspace`:

| field | type and rule |
|---|---|
| `entry` | string: the stem of a manifest entry (`rmsnorm`, `attention`, `gemm_qkv.M64`: an identifier, then `.<Dim><value>` per bucketed dimension of the operator). It is the entry's own bucket, which differs between buckets of the plan for an operator whose binaries are bucketed. |
| `kind` | `"kernel"`: an entry of the manifest's `kernels` (Triton or prebuilt binary). `"library"`: an entry of its `library` (a pinned GEMM). The manifest must have the entry **for `arch`** in that section. |
| `args` | non-empty list, one object per **pointer parameter of the entry's `abi.params`, in order** (a kernel's scalar and `fixed` parameters are not arguments: the host derives them). Names in one step are distinct. |
| `workspace` | present exactly when `kind` is `"library"`: the name of the buffer of role `workspace`. |

An argument is `{"name": <string>, "binding": <buffer name>}` and optionally `"view"`: exactly the keys `name`, `binding` and `view`.

* `name` is the name of the ABI pointer parameter (it must equal it; it is there so that a reader can check the order). For a library entry the pointers are `a`, `b`,
  [`bias`], `out`.
* `binding` names a buffer of the plan. It must not be the `workspace`.
* `view` is `{"shape": [...], "strides": [...], "offset": n}`: `shape` a list of integers >= 1, `strides` a list of integers >= 0 of the same length, `offset` an integer >= 0, **in elements
  of the buffer's dtype**. Without a `view` the argument is **the whole buffer**: the buffer's shape at the bucket, the row-major strides, offset 0. A view must lie inside its
  buffer: `offset + 1 + sum((shape[a] - 1) * strides[a]) <= product(buffer shape at the bucket)`. The writer of a plan (Python) omits `view` exactly when the argument is the whole
  buffer; a reader accepts a view that equals the whole buffer too. A device scalar is bound whole, with no `view`.

**The pointer of an argument is `address(buffer) + view.offset * itemsize(buffer dtype)`**, where `address(buffer)` is the address of the external tensor, or `arena + buffer.offset`.

### 7.2 What a host does with a step

The step is a launch of the manifest entry `entry` with the tensors of its arguments: pointer = as above, shape = `view.shape` (or the buffer's), element strides = `view.strides` (or
the row-major ones). Everything else is as for a single launch in manifest schema 3: the host **binds the entry's dimensions from those shapes and strides** (section 5.1 there, with its
canonical strides for axes of size 1), derives the runtime scalars (dimensions, element counts, strides, `fixed` values), evaluates the grid expression, and checks every precondition of the entry
(`pointer_alignment` of the passed pointer, `static_size`, `divisible`, `bucket`, `max_elements`, `max_extent`), that a pointer without `strides` in its ABI entry gets a **dense** tensor
(the strides of its axes of size > 1 are the row-major ones), that the ranks and dtypes are the ABI's, that a written pointer (`access` `out` or `inout`) overlaps no other argument of the step, and
the grid against the device's limits. A kernel is launched with `cuLaunchKernel` (a Triton kernel with its two zero scratch slots, a prebuilt one with exactly its parameters); a library entry
with `cublasLtMatmul` and its pinned algorithm (manifest schema 3, section 9), whose **workspace is the buffer named by `workspace`**, at its address in the arena (`workspace_bytes` of the entry
are always there; the matmul is given the size the algorithm needs, at most that), so the call allocates nothing.

The one binary an entry names serves every size of its unbucketed dimensions: a plan for `M in {32, 64, 128}` of a row kernel names the entry `rmsnorm` in all three buckets, with `M` bound from
the shapes of the views.

### 7.3 What a plan is checked against in the manifest

`check_against_manifest` (and Python's `export`, which refuses to write a plan that fails it). For every step of every bucket:

* the entry exists in the section `kind` says, for `arch`; for a `library` step the `workspace` buffer's `bytes` is at least the entry's `workspace_bytes`;
* the entry has as many pointer parameters as the step has arguments, and argument `i` has the name of pointer `i`;
* **the dtype** of the buffer is the pointer's `dtype`; **the rank** of the view (or buffer) is `len(shape)` of the ABI pointer;
* a pointer with shape `[]` (a device scalar) is bound to a buffer of role `devscalar`, and only such a pointer is;
* a pointer with access `out` or `inout` is bound to a buffer whose role is `intermediate`, `output` or `inout` (a step never writes an `input`, a `weight`, a `devscalar`);
* **a buffer is written before it is read**: a pointer with access `in` or `inout` is bound to a buffer that is `input`, `weight`, `inout` or `devscalar`, or that an earlier step of the same bucket wrote
  (a pointer with access `out`; an `inout` pointer on a buffer not yet written is refused).

### 7.4 What the rules of `validate_doc` say about steps and buckets (no manifest needed)

* every bucket has the same number of steps, and **in step `k` the `kind`, the `workspace` and, for every argument, the `name` and the `binding` are the same in every bucket**; only
  `entry` and the views' numbers may differ between buckets;
* every `binding` is a buffer; a `workspace` is the buffer of role `workspace`; a `view` is well formed and inside its buffer at that bucket.

## 8. Order, liveness and reuse

* **Order.** The steps of a bucket run in list order, **all on one stream, with no synchronization between them and none implied**: a step that writes a buffer comes before every step that reads
  that version of it, in the order of the graph's IR (an in-place call keeps the storage of the version it consumes, so its readers come after it, and a view that is read after its buffer
  was updated in place is refused by the writer of a plan, because the IR gives a view the contents it had when it was made and a view shares the storage). A host does not reorder, overlap or
  parallelize steps, and does not insert events.
* **Liveness.** The *live range* of a buffer is `[first, last]`: the index of the first step, and of the last, that has the buffer as an argument (a library step's workspace is live for the whole
  plan; a `devscalar` slot is written before step 0 and lives for the whole plan). Both ends are inclusive: a step that reads one buffer and writes another has both live.
* **Reuse.** `"reuse": false` means no two arena regions overlap. `"reuse": true` allows two regions to overlap **only if** both buffers are `intermediate` and their live ranges are disjoint
  (`last` of one < `first` of the other). A device scalar's slot and the workspace never share bytes. `validate_doc` checks this from the steps of the first bucket (the bindings are the same
  in every bucket). Python lays out a reusing plan first-fit by first step; the layout is part of the document, and a host uses it as it is.
* A buffer with no step (a parameter that no operator reads) is allowed; it has no live range.

## 9. What a host does

1. **Load** (section 3): the file, the pin or the trust statement, the manifest, the rules, the entries. Allocate the arena once per device: `arena_bytes` bytes, 256-aligned (nothing if it is 0).
   Keep it for as long as the plan is in use; **the plan is not reentrant**: two executions of one plan are ordered by the caller (one stream, or an event), because they share the arena.
2. **Bind the externals.** The caller gives a tensor (address, shape, strides, dtype) for every buffer of role `input`, `weight` and `inout`, and for every `output` (or the host allocates it), and a
   value for every `devscalar`. Check each external against its buffer: dtype, rank, contiguous row-major (a parameter is a dense tensor), and bind the dimensions (section 5). Select the bucket.
   The external tensors of one call must not overlap each other where one of them is written (`output`, `inout`).
3. **Check, before anything is launched**: for every step of the selected bucket, resolve it (section 7.2) with the real addresses and run every check of a single launch. **If any check fails,
   nothing is launched and nothing is written**, not even a device scalar: the checks are all done first, then the writes and the launches.
4. **Write the device scalars.** Each `devscalar` slot (at `arena + offset`) gets the value the caller gave, as the little-endian bytes of its dtype, ordered on the stream before the first
   launch: a copy from device memory (stream-ordered), or an asynchronous copy from pinned host memory. (Python: `_planrun.TorchBackend.write_devscalar`.)
5. **Launch** the steps in order on the stream (section 7.2). Nothing else: no allocation, no synchronization.

A host may cache everything of steps 1 to 3 that does not depend on an address (the dimension bindings, scalars, grids, the entries' loaded functions and pinned algorithms) per bucket, as Python's
runner does; the address-dependent checks are repeated at every execution.

## 10. CUDA graphs

A plan does nothing during execution that a stream capture forbids: no allocation (the workspace of a GEMM, the intermediates and the device scalars are in the arena), no synchronization, no host
write to device memory but the copy of a device scalar, no module loading, no handle creation, every launch on one stream. So **a plan can be captured once per bucket, at fixed addresses, and the
graph replays byte-identically** (Python gate: `test_gpu_plan.py`, byte-identical to the uncaptured replay and to per-op eager execution, with 0 allocations inside the capture). What has to be so:

1. **Everything that allocates or waits is done before the capture**: the arena, the loading of every function of the bucket (`cuModuleLoadData`, and `cuFuncLoad` where the driver has it, so
   that lazy loading does not happen inside the capture), the cuBLASLt handle (`cublasLtCreate`), the validation of the pinned algorithms. Python: run the plan once, or `tilefish.runtime.prepare(graph, ...)`.
   Python refuses a capture that would do one of these ("has not been prepared").
2. **The outputs** are tensors the caller gives (`outputs=` in Python) and that stay alive as long as the graph: memory allocated inside a capture belongs to the graph.
3. **A device scalar must be written by a captured, device-side copy** from a CUDA tensor of one element: the graph re-reads that tensor at every replay, so changing the value between replays is
   writing that tensor. An `int` or a CPU tensor is a host write (an asynchronous copy from pinned memory is not capturable as a value that changes), and Python refuses it under capture.
4. **A graph holds addresses, not values**: the externals are the same tensors at every replay (write into them in place), and the arena is the same arena.
5. **One stream**: the capture is of the stream the steps are launched on; all steps go there.
6. **One bucket per graph**: a graph is made for one bucket combination's steps, from the first launch to the last, and replayed for that combination's tensors.
7. A plan is **not reentrant** (section 9): two replays of graphs of one plan on two streams at the same time share the arena.
8. Not a plan's doing, but a reason a capture can fail: a binary that is not capturable by itself (the manifest refuses cooperative launches, PDL, clusters and scratch memory for a plain
   replay, so none of those is in a plan), and the capture mode of the framework (torch's `global` mode refuses an unsafe call from another thread while it captures).

## 11. The Python API

```python
plan = tilefish.plan(graph, buckets={"M": [32, 64, 128]}, reuse=False)      # a Plan: no GPU, no torch, nothing built
plan.doc, plan.digest, plan.arena_bytes, plan.arena_bytes_without_reuse, plan.arena_bytes_with_reuse, plan.steps(M=64), plan.buffers, plan.describe()
p = tilefish.runtime.export(graph, "build/", buckets=..., reuse=...)        # builds the binaries, writes manifest.json and plan.json; returns the Plan loaded back (p.digest to pin)
y = tilefish.run(graph, *inputs)                                            # jit: the plan in memory (tilefish.plan(graph), or plan=<a Plan>), binaries built in memory
y = tilefish.run(graph, *inputs, mode="replay", plan="build/", pin=p.digest)  # replay: the exported plan, verified; or trust_directory=True
y = tilefish.run(graph, *inputs, mode="reference")                          # the ops' references composed in torch (an extern's reference= is its oracle)
tilefish.run(graph, *inputs, outputs=out)                                   # write the returned values into these tensors instead of allocating
tilefish.runtime.prepare(graph, mode="replay", plan="build/", pin=p.digest) # arena, binaries, handle: before a CUDA graph capture
```

* `buckets` maps a dimension name to the sizes to plan; it defaults to the dimension's own `buckets=` and is required for a dimension that has none. `reuse` defaults to `False`.
* A `DevScalar` parameter takes an `int`, a CPU tensor or a CUDA tensor of one element at a call.
* `jit` builds the binaries of the plan in memory from the same compile specs and pins the same GEMM algorithms in memory, and runs the *same* runner as `replay`, which differs in where a binary
  comes from (a manifest verified against the specs); so the two are identical by construction (a test holds them to byte equality).

## 12. What a plan does not do (version 1)

A graph body that a plan cannot make is refused by `tilefish.plan` with the rule that says so (`TilefishPlanError`):

* **arithmetic traced in the graph body** (`x + y`, a cast, a reduction, a load): a plan launches declared operators; put it in a `@tilefish.kernel`. Views (`reshape`, `permute`, `select`, `split`,
  `x[i]`) are fine. There is no fusion in this version: the steps are the calls the graph makes.
* a **`store`** into a parameter (a copy), an **`Out`** parameter of the graph and a runtime **`Dim`** parameter: a graph's results are its returned values.
* a call to **another graph** (it is not inlined yet).
* an **in-place call on a view** of another tensor (the IR says updating a view does not change its parent; a plan has no copy to make that true), and a view **read after its tensor was updated
  in place**.
* a returned value that is **an input, a view of anything, or returned twice**.
* a **dimension without buckets**, an operator **without an implementation** (a `library` op with no `impl`), two different operators with one name, a block-scaled tensor.

## 13. An example

The plan of a small graph with an in-place tensor (`accumulate` of `python/tests/plan_model.py`, `buckets={"Q": [16, 32]}`, `reuse=True`; the `tilefish` key and the two digests are not the ones of a
real export, which has a manifest; `test_plan_docs.py` checks that this text is what the code makes and that the digest is right):

<!-- plan-example -->
```json
{"schema": "tilefish.plan.v1", "tilefish": "0.0.3", "graph": "accumulate", "arch": "sm_103", "manifest_digest": "abababababababababababababababababababababababababababababababab", "dims": {"Q": [16, 32]}, "reuse": true, "arena_alignment": 256, "arena_bytes": 16384, "buffers": [{"name": "a", "role": "input", "dtype": "f32", "shape": ["Q", 128], "offset": null, "bytes": 16384}, {"name": "state", "role": "inout", "dtype": "f32", "shape": ["Q", 128], "offset": null, "bytes": 16384}, {"name": "t0", "role": "intermediate", "dtype": "f32", "shape": ["Q", 128], "offset": 0, "bytes": 16384}, {"name": "out0", "role": "output", "dtype": "f32", "shape": ["Q", 128], "offset": null, "bytes": 16384}], "outputs": ["out0"], "buckets": [{"values": {"Q": 16}, "steps": [{"entry": "bump", "kind": "kernel", "args": [{"name": "state", "binding": "state"}, {"name": "y", "binding": "a"}]}, {"entry": "bump", "kind": "kernel", "args": [{"name": "state", "binding": "state"}, {"name": "y", "binding": "a"}]}, {"entry": "twice", "kind": "kernel", "args": [{"name": "x", "binding": "a"}, {"name": "out", "binding": "t0"}]}, {"entry": "bump", "kind": "kernel", "args": [{"name": "state", "binding": "t0"}, {"name": "y", "binding": "a"}]}, {"entry": "plus", "kind": "kernel", "args": [{"name": "x", "binding": "t0"}, {"name": "y", "binding": "state"}, {"name": "out", "binding": "out0"}]}]}, {"values": {"Q": 32}, "steps": [{"entry": "bump", "kind": "kernel", "args": [{"name": "state", "binding": "state"}, {"name": "y", "binding": "a"}]}, {"entry": "bump", "kind": "kernel", "args": [{"name": "state", "binding": "state"}, {"name": "y", "binding": "a"}]}, {"entry": "twice", "kind": "kernel", "args": [{"name": "x", "binding": "a"}, {"name": "out", "binding": "t0"}]}, {"entry": "bump", "kind": "kernel", "args": [{"name": "state", "binding": "t0"}, {"name": "y", "binding": "a"}]}, {"entry": "plus", "kind": "kernel", "args": [{"name": "x", "binding": "t0"}, {"name": "y", "binding": "state"}, {"name": "out", "binding": "out0"}]}]}], "plan_digest": "e4dc0d05d73b157aa1860171601687ce7d40aba3b8dba1f2f41c1dd2c6112bcf"}
```

Read it: `state` is updated in place twice (steps 0 and 1), `twice` makes `t0` in the arena at offset 0 from `a`, `bump` updates `t0` in place (step 3: no second buffer), and `plus` reads
the newest versions of both and writes the returned value into `out0`. The two buckets differ in nothing but the sizes the buffers have there (and each step's stem, which here has no bucket of
its own).

The steps of a plan with views, the attention of the transformer block of the gate (`python/tests/plan_model.py`) at `M = 64`: `q`, `k` and `v` are views of the fused output `t1`, and the result is
bound whole:

```json
{"entry": "attention", "kind": "kernel", "args": [
  {"name": "q", "binding": "t1", "view": {"shape": [64, 4, 64], "strides": [768, 64, 1], "offset": 0}},
  {"name": "k", "binding": "t1", "view": {"shape": [64, 4, 64], "strides": [768, 64, 1], "offset": 256}},
  {"name": "v", "binding": "t1", "view": {"shape": [64, 4, 64], "strides": [768, 64, 1], "offset": 512}},
  {"name": "o", "binding": "t2"},
  {"name": "n_valid", "binding": "n"}]}
```

The block's plan has 9 steps (5 kernels and 4 GEMMs) in each of 3 buckets and an arena of 9 634 048 bytes without reuse (8 519 680 of them the cuBLASLt workspace, 1 114 112 the intermediates)
and 9 240 832 with it (720 896 for the intermediates).

## 14. Where it is tested

| what | tests (`python/tests/`) |
|---|---|
| the walk, the steps and views, the buffers, roles, buckets, the refusals of section 12 | `test_plan.py` |
| the memory plan: alignment, no overlap, liveness reuse against a simulated arena | `test_plan_arena.py` |
| the digest, the pin, every rule of the format, the manifest checks, tampering | `test_plan_schema.py` |
| the document of section 13 | `test_plan_docs.py` |
| the runner: order, addresses, scalars, preconditions before the first launch, captures (fake backend) | `test_planrun.py` |
| on a GPU: replay == jit == per-op eager, replay vs reference, CUDA graph capture, tampering, the arena | `test_gpu_plan.py` |
