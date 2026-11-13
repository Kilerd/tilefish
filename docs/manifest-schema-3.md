# Manifest schema 3

This is the specification of what manifest schema 3 adds to schema 2. The Python package writes and reads schema 3 (`tilefish.runtime.export`,
`tilefish.Manifest.load`); the Rust crate `tilefish` 0.0.1 reads schema 2 only. A Rust lane implements this document. Everything here is checked
by tests in `python/tests/` (named in the last section), and the Python code that writes each field is named where the field is described.

A reader of this document is assumed to know schema 2: the entry fields of a Triton kernel, `abi`, `dims`, `preconditions`, the dimension
expression grammar, the canonical JSON of the digest, and the launch checks the crate already makes. Only what is new or changed is specified; a
field that is not mentioned is as in schema 2.

Contents: 1 what changed; 2 the top level; 3 the digest; 4 kernel entries; 5 the ABI (strided views, device scalars, fixed arguments, scratch
slots); 6 preconditions; 7 the table of dimensions; 8 prebuilt binaries; 9 the `library` section (a pinned cuBLASLt GEMM); 10 strict reading, every
refusal; 11 migrating a schema 2 manifest; 12 what the Rust host does today, and the order to implement it in; 13 where it is tested.

## 1. What changed

| Area | Schema 2 | Schema 3 |
|---|---|---|
| top level | `schema`, `tilefish`, `kernels` | adds `library` (a list, required, may be empty) |
| digest | tag `tilefish.manifest-digest.v2\0`, over `{schema, kernels}` | tag `tilefish.manifest-digest.v3\0`, over `{schema, kernels, library}` |
| kernel entry | Triton kernels only | adds `kind` (`"triton"` or `"binary"`); a `binary` entry is a prebuilt PTX or cubin |
| `abi` | `params`, `scratch` | adds `scratch_slots` (2 for Triton, 0 for a prebuilt binary) |
| ABI pointer | dense, row-major, offset 0 | may take a strided view: adds `strides` and `offset` |
| ABI pointer, shape `[]` | not used | a *device scalar*: one element in device memory |
| ABI scalar | `i32`, carries a dimension (`dim`) or an element count (`count_of`) | may be `i64`; may carry a stride (`stride_of`) |
| ABI argument kinds | `pointer`, `scalar` | adds `fixed`: a scalar whose value is in the manifest |
| preconditions | `pointer_alignment` always 16 bytes | a strided pointer may promise only its element alignment; adds `max_extent` |
| `dims` | names from shapes, scalars and the grid | also the stride dimensions and the offset dimension of a strided pointer |
| library | none | a pinned cuBLASLt GEMM: `library` entries (no file) |

Nothing is removed. A schema 2 manifest of Triton kernels becomes a schema 3 manifest by three additions (section 11).

## 2. The top level

```json
{"schema": 3, "tilefish": "0.0.3", "kernels": [ ... ], "library": [ ... ]}
```

* `schema` is the JSON integer `3`: not `3.0`, not `"3"`, not `true`. A reader refuses any other value. For a document whose `schema` is an integer
  below 3 the error says it is an older schema and to export again; for anything else it says it is not a manifest of schema 3.
* `kernels` and `library` are both required JSON arrays. `tilefish` is informative and is not digested.
* A `(stem, arch)` pair is unique across `kernels` **and** `library` together: a repeated pair is refused (Python: `Manifest.load`).
* The file is read as strict JSON, as in schema 2 (no duplicate keys, no `NaN`/`Infinity`, no number that does not fit a double).

## 3. The digest

`manifest_digest(doc)` (`python/src/tilefish/_manifest.py`) is

```
sha256( b"tilefish.manifest-digest.v3\0" + canonical_json({"schema": 3, "kernels": K, "library": L}) )
```

* `K` is `doc.kernels` sorted by `(stem, arch)` and `L` is `doc.library` sorted by `(stem, arch)`; the comparison is of the two strings (code point
  order, as Python sorts `str`). The order of the entries in the file does not matter.
* `canonical_json` is exactly schema 2's: object keys sorted, separators `","` and `":"`, no whitespace, non-ASCII escaped (`ensure_ascii`), no
  `NaN`/`Infinity`, floats in Python's shortest round-trip `repr` (`2.0`, `1e-06`).
* **Every new field is digested, with no exception and no special treatment:** an entry is digested whole, so `kind`, `abi.scratch_slots`,
  `strides`, `offset`, `stride_of`, a `fixed` argument and its `value`, `binary`, the `max_extent` precondition, the `dims` additions, every field
  of a `library` entry (the algorithm, the cuBLASLt version, the workspace) are all in it. The top-level `tilefish` is the only field outside it.
* The digest of the empty schema 3 manifest, `{"schema":3,"kernels":[],"library":[]}`, is
  `f66955caeb4e878dbe95483da2607d19ea4f734977e86282fff6f92f1a87fe6e` (a test vector; `test_manifest_schema3.py` recomputes it from the definition
  above). The empty schema 2 manifest `{"schema":2,"kernels":[]}` has `a3593eed6b463f7408711b92645197eda2e7bc5f2830b138b40e33ceedc1cfc8`.
* The digest of a schema 2 document (`{"schema":2,"kernels":[...]}`, tag `v2`) is still computable by the same function: it is what the crate 0.0.1
  computes for the manifests it reads, and `crates/tilefish/tests/fixtures/expected.json` holds. A document whose `schema` is any other value than 3 is
  digested in that same schema 2 form with whatever `schema` says (a reader refuses it, a digest function does not: the cross-language test vectors of
  `crates/tilefish/tests/data/python_manifests.in` include such documents, and `export_rust_vectors.py` still reproduces them byte for byte). A
  schema 3 digest is never equal to a schema 2 one.

## 4. Kernel entries

A `kernels` entry has a `kind`, which is required and is exactly one of `"triton"` and `"binary"`.

**`kind: "triton"`** is schema 2's entry plus `kind` and `abi.scratch_slots`. Its fields are

`stem op kind arch ptx_target symbol cubin abi grid block shared num_warps num_stages launch constexprs buckets dims preconditions options
declaration_digest sha256 sha256_code toolchain` and optionally `source_sha256`, `source_digest` + `source_digest_kind` (as in schema 2:
`source_digest_kind` is `"tilefish.triton-ir.v1"` for the digest of the Triton IR).
`toolchain` is `{"triton": ..., "ptxas": ...}`.

**`kind: "binary"`** (section 8) has no `num_warps`, `num_stages` or `options` (a reader refuses an entry that has one) and has instead `binary`.
Its fields are

`stem op kind arch ptx_target symbol cubin abi grid block shared launch constexprs buckets dims preconditions declaration_digest sha256 sha256_code
toolchain binary source_digest source_digest_kind`

with `constexprs` always `[]`, `source_digest_kind` always `"tilefish.binary.v1"`, `source_digest` always present, and no `source_sha256`.

Validation rules that are new (Python: `Manifest._read`):

* `kind` is one of the two strings; any other value, or its absence, is refused.
* If `abi.scratch_slots` is present it is an integer (not a bool) and equals 2 for `triton` and 0 for `binary`. (Python's reader checks it when the
  field is there, and the comparison with the operator's compile spec requires it to be exactly what the spec says; a host that reads without a spec
  must require it, and require `len(abi.scratch) == abi.scratch_slots`.)
* For `triton` the existing rule stands: an entry whose `shared` is 0 but whose cubin has a section `.nv.shared.<symbol>` is refused. **For `binary`
  it does not apply**: a hand-written kernel with *static* shared memory has that section too, and the declaration (`shared`) says how much
  *dynamic* shared memory it addresses; the loader's check (static + dynamic ≤ the device's opt-in maximum) is the guard.
* `toolchain` for `binary` is `{}` (a cubin used as it is) or `{"ptxas": <the `--version` text>}` (PTX that was assembled).

## 5. The ABI

`abi` is `{"params": [...], "scratch": [...], "scratch_slots": <int>}`. `params` are the kernel's launch arguments in order, one `kernelParams` slot
each (`slot` is the position in `params`). `scratch` is what a Triton kernel is launched with after `params`: two null 64-bit pointers
(`global_scratch`, `profile_scratch`, `value` 0), unchanged. **A prebuilt binary has `scratch: []` and `scratch_slots: 0`: it is launched with exactly
`params`, no more.**

Every `params` entry has `slot`, `index` (the position in the declaration), `name`, `kind`, `type`, `ptx_type`, `size`. The kinds and what else each has:

| `kind` | `type` | `ptx_type`, `size` | other fields |
|---|---|---|---|
| `pointer` | `*bf16`, `*fp32`, `*i32`, ... (Triton's pointer type of the dtype) | `u64`, 8 | `dtype`, `access` (`in`/`out`/`inout`), `shape`; optional `strides` and `offset` (together) |
| `scalar` | `i32` or `i64` | `u32`/4 or `u64`/8 | `dim` (an expression); optionally one of `count_of` or `stride_of` |
| `fixed` (prebuilt binaries only) | `i32`, `i64` or `f32` | `u32`/4, `u64`/8, `f32`/4 | `value` |

A launch passes: a `pointer` as its 64-bit address (see 5.1), a `scalar` as a signed integer of its `type`, a `fixed` as its `value` in its `type`
(`ctypes.c_int32`/`c_int64`/`c_float`: an `f32` value is the JSON number rounded to the nearest `f32`).

### 5.1 Strided views: `strides` and `offset` on a pointer

A pointer with `strides` and `offset` accepts a view of a tensor: any element strides and any starting offset. (A pointer without them is a dense,
row-major tensor at offset 0, as before.) Both fields are present together or neither is.

* `strides` is a list with exactly one entry per axis of `shape` (at least one). An entry is **an integer** (a *fixed* stride) or **a string** (the
  name of a *stride dimension*).
  * A fixed stride is `1`, only ever on the **last** axis (the kernel loads along that axis contiguously). A reader refuses any other integer and a
    fixed stride on any other axis.
  * A string entry is the `dim` of exactly one `scalar` argument whose `stride_of` is `{"arg": <this pointer's name>, "axis": <this axis>}`; that
    scalar carries the stride. Python checks both directions (`_specvalid._check_strides`): each `stride_of` names a strided pointer and an axis of
    it, whose entry is the scalar's `dim` string, once per axis; each string entry has its scalar.
* `offset` is a string: the name of the **offset dimension** of this pointer. (It is `"<pointer name>_offset"`.) It is not carried by a scalar:
  the pointer passed to the kernel is already `base + offset * itemsize` (the address of the first element of the view).
* `stride_of` on a scalar is the object `{"arg": <identifier>, "axis": <int ≥ 0>}`; a scalar has `stride_of` **or** `count_of`, never both.

**What a host does to launch.** The host is given a view: its pointer (the address of its first element), shape, element strides (and, if it
knows it, the element offset from the start of the allocation).

1. Bind the shape dimensions from `shape` as in schema 2.
2. For each axis `a` of the view, let `stride[a]` be its element stride, **made canonical**: for an axis of size 1 the stride is replaced, going from
   the last axis to the first, by `1` for the last axis and `size[a+1] * stride[a+1]` otherwise (a size-1 axis's stride is never read by a kernel, so
   an `unsqueeze` or an `expand` does not pass a stride that no kernel reads and that might not fit 32 bits). Other strides are used as they are.
   A stride must be ≥ 0 (a stride of 0, as of an expanded view, is allowed); a negative stride is refused.
3. For each axis whose `strides` entry is an integer `1`: the view's canonical stride along it must be 1, else refused (kind `stride`).
4. For each axis whose entry is a name `x_s0`: bind the dimension `x_s0` to the canonical stride (it is an unknown to solve like a shape dimension:
   it appears in `dims`, and a `scalar` with `dim: "x_s0"` carries its value). Check the constraint `dims["x_s0"].divisible_by` if present.
5. A `scalar` with `stride_of` is passed as that bound stride, as a signed integer of its `type`. **It must fit the type:** `0 ≤ v ≤ 2^31 − 1` for
   `i32`, `0 ≤ v ≤ 2^63 − 1` for `i64`; else refused. (The `type` is `i64` exactly when the op declares `index_bits=64`; every other scalar, a
   dimension or an element count, stays `i32`.)
6. The pointer passed is the view's address; the `pointer_alignment` precondition (section 6) is checked on it.
7. The offset dimension is not bound from the stride scalars; a host that knows the element offset *may* check `dims[offset].divisible_by` against
   it, but the address check of step 6 is what is required (for a base that is 16-byte aligned they are the same check).

`max_extent` (section 6) bounds the view. For aliasing, a strided tensor occupies `[address, address + extent * itemsize)`, with `extent` as there;
a tensor that is written (`out`, `inout`) must not overlap any other argument. (Only `in` pointers are strided in this version: the declaration of an
`inout` or `out` strided pointer is refused by the Python frontend, so a reader may refuse `strides` on a pointer whose `access` is not `in`.)

### 5.2 Device scalars: a pointer with the shape `[]`

A pointer whose `shape` is the empty list is a **device scalar**: one element of its `dtype` in device memory (the Python type `DevScalar[i32]`).
Its `access` is `in`. A host passes a region of exactly one element (rank 0 or shape `[1]`) and checks its dtype and `pointer_alignment` (16 bytes);
it has no `static_size`, `max_elements` or `max_extent` precondition and no `strides`/`offset`. The kernel reads it with `tl.load(ptr)`. It is an
ordinary pointer otherwise; it exists as a distinct case only so that a host can write the value into a per-call arena before a launch.

### 5.3 `fixed` arguments

A `fixed` argument is a scalar whose value is in the manifest: a prebuilt binary's constant argument (a `float alpha`, a `long` that must be zero,
a Triton scratch slot written out as an explicit parameter). It has `value` and no `dim`. `value` is a JSON integer for `i32`/`i64` (not a bool, in
range of the type) and a JSON number **with a fraction or exponent** (`2.0`) for `f32`; a non-finite `f32` is the tagged object
`{"float": "inf" | "-inf" | "nan"}` (as a constexpr value is in schema 2). A caller of the operator may restate the value but not change it.

## 6. Preconditions

The list is as in schema 2, with these differences.

* `pointer_alignment`: `bytes` is the alignment the pointer *passed* must have (`address % bytes == 0`). For an ordinary pointer it is 16, as before.
  For a pointer that has `strides`: **either 16, or the size in bytes of its element** (`max(1, bits // 8)` of its dtype). 16 means the operator was
  declared as 16-byte aligned (`Strided[..., 16]`), the offset dimension is `divisible_by` the elements in 16 bytes, and, for a Triton kernel, the
  pointer has `tt.divisibility=16` (see below); the element size means no alignment is assumed beyond the element's own, and Triton was given no hint.
  A device scalar's is 16.
* new kind **`max_extent`**: `{"kind": "max_extent", "arg": <pointer>, "index": <its position>, "value": 2147483647}`, once per pointer that has
  `strides` (and only on such a pointer; `max_elements` is not allowed on it). It says: for a kernel that indexes with `i32` (`index_bits=32`, the
  default), the view's **extent** must not exceed `value` elements, where
  `extent = 1 + Σ over axes of (size[a] − 1) * canonical_stride[a]` (computed with unbounded integers: the product can exceed 64 bits). The test is
  `extent ≤ value`. (`max_elements` bounds an element count, which for a view says nothing about the offsets the kernel computes.) A kernel with
  `index_bits=64` has neither precondition on its pointers. `value` is exactly `2**31 − 1`.
* `divisible` on a stride scalar: as in schema 2 (`scalars[arg] % by == 0`); present when the declaration says the stride is a multiple of 16 or more.
* `static_size`, `bucket`, `max_elements`: unchanged. `max_elements`/`static_size` do not appear for a device scalar.

For a `triton` entry the **attributes Triton was given** are not in the manifest (they are in the compile spec), but they follow: `tt.divisibility=16`
on a pointer exactly when its `pointer_alignment` is 16 bytes, and on a scalar exactly when it has a `divisible` precondition.

## 7. The table of dimensions (`dims`)

`dims` maps every named dimension the entry mentions to its constraints (`divisible_by`, `buckets`, `bucket`), as in schema 2. Schema 3 adds the
dimensions of a strided pointer: every stride dimension (`x_s0`) and the offset dimension (`x_offset`) are named in it, each with `{}` or with its
constraints (`{"divisible_by": 8}` on the offset dimension of a `Strided[..., 16]` pointer; `{"divisible_by": k}` on a stride that the declaration
says is a multiple of `k`). A name used in a shape, a `strides`/`offset` entry, a scalar's `dim` or the grid must be in `dims`.

## 8. Prebuilt binaries (`kind: "binary"`)

A prebuilt binary is a kernel Triton does not compile: PTX assembled by `ptxas`, or a cubin used as it is. It is declared in Python by its C ABI.

**The entry.** As section 4. The fields that say what it is made of:

* `symbol`: the kernel's entry point; the cubin must have a code section `.text.<symbol>` (refused otherwise).
* `block`: `[x, y, z]`, from the declaration (each 1 to 1024, product ≤ 1024). It replaces `32 * num_warps`.
* `shared`: the bytes of **dynamic** shared memory to launch with, from the declaration (a reader checks `static + shared ≤` the device's opt-in
  maximum, as for Triton).
* `grid`: an expression over the dimension names, as in schema 2.
* `launch`: always the zero value `{"global_scratch_size": 0, "profile_scratch_size": 0, "launch_pdl": false, "cluster_dims": [1,1,1], "num_ctas": 1,
  "cooperative": false}`, so a plain `cuLaunchKernel` runs it.
* `ptx_target`: for PTX the `--gpu-name` the PTX was assembled for (`sm_103a` when the PTX has `.target sm_103a`, else `sm_<arch>`); for a cubin `""`.
* `binary`: `{"source": "cubin"}` or `{"source": "ptx", "ptxas_flags": [<strings>]}` (exactly those keys; the flags are the argument vector given to
  `ptxas` before the file names: `["-lineinfo", "-v", "--gpu-name=sm_103a"]`, the flags Triton 3.5 passes, in the same spelling, since ptxas records
  its command line in the cubin and the machine code digest depends on it).
* `source_digest`: the sha256 (64 lowercase hex digits) of the bytes of the PTX file, or of the cubin file used for this `arch`;
  `source_digest_kind` is `"tilefish.binary.v1"`. For a cubin source it equals `sha256`.
* `toolchain`: `{"ptxas": <version text>}` for PTX, `{}` for a cubin. `declaration_digest`, `sha256`, `sha256_code`: as for any entry.

**Launch.** `cuModuleLoadData(cubin)`, `cuModuleGetFunction(symbol)`, then `cuLaunchKernel(grid, block, shared)` with `kernelParams` = the `abi.params`
in order: a `pointer` is its address (`u64`), a `scalar` its value (`i32`/`i64`), a `fixed` its `value` (`i32`/`i64`/`f32`); **nothing is appended**
(`scratch_slots` is 0). `cuFuncGetParamInfo`, when the driver can answer, must report exactly the `size` of each parameter and no more (Python
checks this at load). A cubin is refused for an architecture other than its own: the declaration has one cubin per architecture, and a replay on
`sm_90` finds no entry (and does not use the `sm_103` one).

**Identity and staleness.** A binary's identity is `source_digest`. A replay compares it with the digest of the bytes the declaration holds
(the declaration reads its file once, when the operator is declared) and refuses a mismatch as stale, pinned or not; no Triton and no GPU are needed
to tell. There is no IR digest.

**Preconditions and dims** are the declaration's, as for a Triton kernel: pointer alignment, shapes, `divisible`, `max_elements` (the default
`index_bits=32` applies to a prebuilt kernel too: its author says `index_bits=64` if it indexes in 64 bits), buckets. A prebuilt binary is not
compiled by Triton, so there are no `tt.divisibility` attributes; the preconditions are what is promised to the kernel.

## 9. The `library` section: a pinned cuBLASLt GEMM

`library` entries describe operations that a vendor library runs, with no file: today, a bf16 GEMM through cuBLASLt with the algorithm pinned.

**The operation.** In tilefish's row-major terms, `out[M, N] = a[M, K] @ b[N, K]^T` (`+ bias[N]` for `impl: "cublaslt.gemm_bias"`): `a`, `b`, `bias`
and `out` are bf16, contiguous, 16-byte aligned; the accumulator is f32; the result is rounded to bf16 **once**; the bias is added in f32 to the
accumulator before that rounding (`CUBLASLT_EPILOGUE_BIAS`). `K` and `N` are multiples of 8. `M`, `N`, `K` are static or bucketed: one entry per
bucket combination (the stem names the buckets, like a kernel's).

**The fields** (exactly these; a reader refuses missing and unknown ones; `python/src/tilefish/_library.py`, `ENTRY_KEYS`, `validate_entry`):

| field | type and rule |
|---|---|
| `stem` | string, `op` plus `.<Dim><value>` per bucketed dimension in the order the dimension names sort (`gemm.M64`, `gemm_bias.Kb1024.M32`); must equal `op` + the buckets |
| `op` | identifier |
| `arch` | `sm_<2 or 3 digits>`, no suffix (`sm_103`) |
| `impl` | `"cublaslt.gemm"` or `"cublaslt.gemm_bias"` |
| `abi` | `{"params": [...]}`, exactly that key; the pointers `a`, `b`, [`bias`], `out` in this order (see below) |
| `buckets` | object: dimension name → its bucket value (int ≥ 1) |
| `dims` | the table of section 7 (`divisible_by`, `buckets`, `bucket`), every bucketed dimension with its `bucket` |
| `preconditions` | list: one `pointer_alignment` of 16 bytes per pointer, a `static_size` per static axis, a `bucket` per bucketed dimension; no other kind |
| `dtypes` | `{"a": "bf16", "b": "bf16", "out": "bf16"}` plus `"bias": "bf16"` for the bias op |
| `compute_type` | `"CUBLAS_COMPUTE_32F"` |
| `scale_type` | `"CUDA_R_32F"` |
| `transa`, `transb` | `"T"` and `"N"`: the transposes **as cuBLASLt is called** (below) |
| `m`, `n`, `k` | ints from 1 to 2147483647: the **tilefish** sizes, `a` is `[m, k]`, `b` is `[n, k]`, `out` is `[m, n]` |
| `lda`, `ldb`, `ldc`, `ldd` | ints: the leading dimensions as cuBLASLt is called; exactly `lda = ldb = k`, `ldc = ldd = n` |
| `epilogue` | `"none"` for `gemm`, `"bias"` for `gemm_bias` |
| `workspace_bytes` | int from 1 to 2^40: the workspace limit the heuristic was asked under **and the size of the workspace a launch provides** (default `8519680`, torch's default `CUBLAS_WORKSPACE_CONFIG`) |
| `algo` | 128 lowercase hex digits: the 64 bytes of the `cublasLtMatmulAlgo_t`, in memory order (eight `uint64_t`, as the host stores them) |
| `cublaslt_version` | int ≥ 1: the value of `cublasLtGetVersion()` of the library that chose `algo` (e.g. `120804` for 12.8.4) |
| `declaration_digest` | 64 lowercase hex digits: sha256 over the canonical declaration (name, impl, workspace limit, signature with the dimensions' constraints) |

`abi.params[i]` has exactly `index` (= i), `name`, `kind` (`"pointer"`), `dtype` (`"bf16"`), `access` (`"in"`, and `"out"` for the last), `shape` (a list
of ints and dimension expressions: `a` is rank 2, `b` rank 2, `bias` rank 1, `out` rank 2). The shapes must agree with `m`, `n`, `k`: an integer entry
equals the size, an expression entry evaluates to it when the bucketed dimensions have their `buckets` values (an expression that mentions a
dimension with no bucket is refused).

**The cuBLASLt problem.** cuBLASLt is column-major, so tilefish's `a`, `b`, `out` are passed as the TN gemm `m_cublas = n`, `n_cublas = m`, `k_cublas = k`:
the cuBLAS "A" operand is tilefish's **`b`** (`transa = T`, a `k × n` column-major matrix with `lda = k`), the cuBLAS "B" operand is tilefish's **`a`**
(`transb = N`, `k × m`, `ldb = k`), and `C` and `D` are tilefish's `out` (`n × m` column-major, `ldc = ldd = n`). `alpha = 1.0f`, `beta = 0.0f`,
host pointer mode. All matrix layouts are `CUDA_R_16BF`, column-major; the matmul descriptor has `compute_type`, `scale_type`, `TRANSA`, `TRANSB`
and, for the bias op, `EPILOGUE = CUBLASLT_EPILOGUE_BIAS`, `BIAS_DATA_TYPE = CUDA_R_16BF` and `BIAS_POINTER` = the bias pointer. Python's
binding (`python/src/tilefish/_cublaslt.py`, `describe`) is the reference for every value, and `python/tests/test_cublaslt.py` checks that this
table and `describe` agree.

**Replay.**

1. Bind the dimensions from the shapes and check the preconditions as for a kernel (alignment, static sizes, the exact bucket, contiguity).
2. Refuse if the process's `cublasLtGetVersion()` is not `cublaslt_version`: an algorithm id names other kernels in another version. (Python:
   `TilefishManifestError`, "the manifest pinned the cuBLASLt algorithm of '<stem>' (<arch>) with cuBLASLt X, but this process has cuBLASLt Y".)
3. Rebuild the `cublasLtMatmulAlgo_t` from `algo`; ask `cublasLtMatmulAlgoCheck` for the problem (the algo must be valid, and its
   `workspaceSize` must be ≤ `workspace_bytes`), then `cublasLtMatmul` with a workspace of at least that size on the launch stream.
   **`cublasLtMatmulAlgoCheck` is a weak judge of the bytes**: measured on 12.8.4 it accepts a changed version field, changed padding, a reserved CTA
   swizzling and a garbage custom option, and it crashes the process on a tile id ≥ 2^31. The Python binding therefore first rebuilds the algorithm
   with `cublasLtMatmulAlgoInit` + `cublasLtMatmulAlgoConfigSetAttribute` from the configuration the bytes read back as and requires identical bytes,
   requires the tile and stages ids to be in the capability lists of the algorithm id, the swizzling and the custom option to be in range, an f32
   accumulator, alignment capabilities ≤ 16 bytes, and split-K to go with the reduction scheme `NONE` or `COMPUTE_TYPE` (f32 partial sums; a bf16
   partial sum would round more than once). A host that runs a `library` entry should do the same. What no byte check can see is a flip to *another
   valid* algorithm: only the digest of the manifest vouches for the file.
4. The pin belongs to one `arch`: an entry is found by `(stem, arch)` for the architecture of the device, and a missing pair is an error naming the
   architectures the manifest has.

**Pinning (how an entry is made)**, for the writers: the heuristic is asked on a device of `arch` with the preferences `MAX_WORKSPACE_BYTES =
workspace_bytes`, `MIN_ALIGNMENT_{A,B,C,D}_BYTES = 16` and a reduction-scheme mask of `NONE | COMPUTE_TYPE`, for a few results; result 0 is the pin.
`jit` pins in memory the same way, so `jit` and `replay` run the same algorithm by construction. A pin does not bind itself to its problem
(the same bytes are accepted for other problems), which is why `m`, `n`, `k` and the epilogue are in the entry.

**Digest.** The entry is digested whole, in `library` (section 3); `algo` and `cublaslt_version` are in it.

## 10. Strict reading: every refusal

A reader of schema 3 refuses (Python: `TilefishManifestError` unless marked *spec*, which are `TilefishSpecError` of the per-binary compile spec, a
document tilefish writes next to a build and the host does not read):

* the top level: wrong `schema` type/value; `kernels` or `library` missing or not a list; an entry that is not an object; a repeated `(stem, arch)`
  across both lists.
* every kernel entry: a `kind` that is not `triton`/`binary`; a missing `stem`, `arch`, `symbol`, `cubin`, `sha256`, `sha256_code`, `shared`, `launch`,
  `toolchain`; `options` missing on `triton` and present on `binary`; `num_warps`/`num_stages` present on `binary`; `binary` missing on `binary`;
  `shared` not a non-negative int; `toolchain`, `launch`, `abi`, `options` not objects; `abi.scratch_slots` present and not `2`/`0` by kind;
  `source_sha256`/`source_digest` not 64 lowercase hex digits; `source_digest` and `source_digest_kind` not both present or both absent;
  for `binary`: `binary` not exactly one of `{"source":"cubin"}` and `{"source":"ptx","ptxas_flags":[strings]}`, `source_digest_kind` not
  `"tilefish.binary.v1"`, no `source_digest`, a `source_sha256`; the cubin file name not a plain file name; the cubin's `sha256`, `sha256_code` or
  `.text.<symbol>` section wrong; for `triton` the shared-memory section rule.
* the ABI (a host that reads without a spec checks these itself; Python checks them by comparing the entry with the compile spec, and `_specvalid`
  checks them in the spec): `params` kinds in `pointer`/`scalar`/`fixed`; a `scalar` type other than `i32`/`i64`, or `dtype` ≠ `type`; `strides` without
  `offset` or the reverse; `strides` of the wrong length or empty, with an integer other than `1`, an integer on a non-last axis, a string that no
  `stride_of` scalar carries, or two scalars for one axis; an `offset` that is not a string; `stride_of` with a missing/extra key, a negative axis, an
  axis out of range, a target that is not a strided pointer, a `dim` different from the target's `strides` entry, or together with `count_of`; a `fixed`
  argument with a type not in `i32`/`i64`/`f32`, an `f32` value that is not a float, an integer type with a non-integer, a bool, or an out-of-range value.
* preconditions: `max_extent` not exactly once on a strided pointer, with a value other than `2147483647`, or on a pointer without `strides`;
  `max_elements` on a pointer with `strides`; `pointer_alignment` of a strided pointer other than 16 or its element size (of any other pointer other
  than 16); every pointer must have exactly one `pointer_alignment`.
* every `library` entry: the field list, the types and the cross-field equalities of section 9, each by the rule given there.
* *spec* only, for prebuilt binaries: a `binary` section that is not `{kind, symbol, shared, digests}`; `attrs` not empty; `launch_params` not every arg
  in order; a `block` that is not three ints from 1 to 1024 with product ≤ 1024; a `constexpr` argument.

## 11. Migrating a schema 2 manifest

A schema 2 manifest of Triton kernels is the same manifest in schema 3 after exactly these changes, and its cubins are untouched
(`python/tests/schema2_fixtures.py: migrate_doc`):

1. `schema` becomes `3`;
2. every kernel entry gains `"kind": "triton"` and its `abi` gains `"scratch_slots": 2`;
3. the manifest gains `"library": []`.

Its digest changes (tag `v3`, and the `library` key), so a pin must be taken again. Python refuses schema 2 (`Re-export it`), so it cannot read the
crate's committed fixtures directly; its tests migrate a copy (`test_rust_host.py: migrated_fixture`) and compute the schema 2 digest of the original
with `manifest_digest`, which equals `expected.json`. The `export_rust_fixtures.py` script now writes schema 3: the Rust lane regenerates the
fixtures (and `generated.rs`, whose `MANIFEST_DIGEST` is the digest of the manifest) when it moves the crate to schema 3.

## 12. What the Rust host does today, and an order to implement it

`tilefish.hosts.rust.emit` still writes typed launchers for a schema 3 manifest of Triton kernels that use nothing of schema 2's features beyond
what schema 2 had (every pointer dense, every scalar an `i32` dimension), and refuses, with a message that ends "not yet supported by the Rust
host", everything else: a `binary` kernel, a pointer with `strides`/`offset`, a pointer with shape `[]`, a non-`i32` scalar or one with
`stride_of`, a `fixed` argument, and a manifest that has `library` entries. The launchers it writes embed the schema 3 digest, so they load under a
crate that reads schema 3 and fail at `load` with the crate's "unsupported schema" error under 0.0.1.

A reasonable order for the Rust lane, each step testable against Python on a GPU (`python/tests/rust_gate.py` and `crates/tilefish/tests/gpu_gate.rs`):

1. read schema 3 and the new digest (tag, `library`); migrate the fixtures; keep every kernel entry working (`kind: "triton"`, `scratch_slots`).
2. `fixed` arguments and `binary` entries (launch with exactly `params`, no scratch; `block`, `shared` from the entry; the `.text.<symbol>` and
   param-size checks). Gate: the PTX of a lowered golden as a prebuilt extern is byte-identical to its Triton path
   (`test_gpu_binary.py::test_the_ptx_of_the_elementwise_gelu...`).
3. device scalars (a pointer with shape `[]`), then strided pointers (`strides`, `offset`, `stride_of`, `max_extent`, canonical strides, the
   ranges). Gates: `test_gpu_devscalar.py`, `test_gpu_strided.py` (an aligned and an unaligned offset, an expanded view, a transposed one).
4. `library` entries: version check, algorithm validation, workspace, the launch (`test_gpu_gemm.py`: byte-identical across two processes and to `jit`).

## 13. Where it is tested

| what | tests (`python/tests/`) |
|---|---|
| top level, digest, the three additions, refusal of schema 2 | `test_manifest.py`, `test_manifest_schema3.py`, `test_rust_host.py` |
| `kind`, `abi.scratch_slots`, prebuilt binary entries | `test_binary_externs.py`, `test_manifest_schema3.py` |
| strided pointers, `strides`/`offset`/`stride_of`, `max_extent`, alignment | `test_strided_specs.py`, `test_views.py`; on a GPU `test_gpu_strided.py` |
| device scalars | `test_devscalar.py`; on a GPU `test_gpu_devscalar.py` |
| prebuilt PTX and cubin, byte identity with the Triton path | `test_binary_externs.py`; on a GPU `test_gpu_binary.py`, `ptx/*.ptx` |
| the `library` section | `test_library_ops.py`, `test_cublaslt.py`; on a GPU `test_gpu_gemm.py`, `test_gpu_cublaslt.py` |
