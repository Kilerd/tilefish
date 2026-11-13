# Manifest schema 3

This is the specification of what manifest schema 3 adds to schema 2. The Python package writes and reads schema 3 (`tilefish.runtime.export`,
`tilefish.Manifest.load`), and so does the Rust crate `tilefish` of this repository (the published 0.0.1 reads schema 2 only; section 12 says what the
crate does with each part of this document). Everything here is checked by tests in `python/tests/` and in `crates/tilefish/tests/` (named in the last
section), and the Python code that writes each field is named where the field is described.

A reader of this document is assumed to know schema 2: the entry fields of a Triton kernel, `abi`, `dims`, `preconditions`, the dimension
expression grammar, the canonical JSON of the digest, and the launch checks the crate already makes. Only what is new or changed is specified; a
field that is not mentioned is as in schema 2.

Contents: 1 what changed; 2 the top level; 3 the digest; 4 kernel entries; 5 the ABI (strided views, device scalars, fixed arguments, scratch
slots); 6 preconditions; 7 the table of dimensions; 8 prebuilt binaries; 9 the `library` section (a pinned cuBLASLt GEMM); 10 strict reading, every
refusal; 11 migrating a schema 2 manifest; 12 what the Rust host does; 13 where it is tested.

## 1. What changed

| Area | Schema 2 | Schema 3 |
|---|---|---|
| top level | `schema`, `tilefish`, `kernels` | adds `library` (a list, required, may be empty) |
| digest | tag `tilefish.manifest-digest.v2\0`, over `{schema, kernels}` | tag `tilefish.manifest-digest.v3\0`, over `{schema, kernels, library}` |
| kernel entry | Triton kernels only | adds `kind` (`"triton"` or `"binary"`); a `binary` entry is a prebuilt PTX or cubin |
| `abi` | `params`, `scratch` | adds `scratch_slots` (2 for Triton, 0 for a prebuilt binary) |
| ABI pointer | dense, row-major, offset 0 | may take a strided view: adds `strides` and `offset` |
| ABI pointer, shape `[]` | not used | a *device scalar*: one element in device memory |
| ABI scalar | `i32`, carries a dimension (`dim`) or an element count (`count_of`) | may be `i64`; may carry a stride (`stride_of`), of one view or of several views that share it |
| ABI argument kinds | `pointer`, `scalar` | adds `fixed`: a scalar whose value is in the manifest |
| preconditions | `pointer_alignment` always 16 bytes | a strided pointer may promise only its element alignment; adds `max_extent` |
| `dims` | names from shapes, scalars and the grid | also the stride dimensions and the offset dimension of a strided pointer |
| library | none | a pinned cuBLASLt GEMM, for one device (its architecture and SM count): `library` entries (no file) |
| library entry (0.0.5) | — | three optional fields, present only when not the default: `heuristic`, `algo_source`, `allow_bf16_reduction` (section 9.1) |

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
  `strides`, `offset`, `stride_of`, `alias`, a `fixed` argument and its `value`, `binary`, the `max_extent` precondition, the `dims` additions, every field
  of a `library` entry (the algorithm, the cuBLASLt version, the SM count of the device, the workspace) are all in it. The top-level `tilefish` is the only field outside it.
* The digest of the empty schema 3 manifest, `{"schema":3,"kernels":[],"library":[]}`, is
  `f66955caeb4e878dbe95483da2607d19ea4f734977e86282fff6f92f1a87fe6e` (a test vector; `test_manifest_schema3.py` recomputes it from the definition
  above). The empty schema 2 manifest `{"schema":2,"kernels":[]}` has `a3593eed6b463f7408711b92645197eda2e7bc5f2830b138b40e33ceedc1cfc8`.
* The digest of a schema 2 document (`{"schema":2,"kernels":[...]}`, tag `v2`) is still computable by the same function: it is what the published
  crate 0.0.1 computed for the manifests it read. A document whose `schema` is any other value than 3 is digested in that same schema 2 form with
  whatever `schema` says (a reader refuses it, a digest function does not: the cross-language test vectors of
  `crates/tilefish/tests/data/python_manifests.in` include such documents, and `export_rust_vectors.py` reproduces them byte for byte). "Is 3" is Python's
  `schema == 3`: the integer `3` and the float `3.0` are (a document with `"schema": 3.0` is digested in the schema 3 form, with `3.0` in the text), `true`
  and `"3"` are not. A schema 3 document without a `library` list, a `kernels` or `library` that is not a list, and an entry that is not an object cannot be
  digested (the same file `python_manifests.in` lists them in `MANIFEST_ERRORS`; the Rust `manifest_digest` returns an error for each). A schema 3 digest is
  never equal to a schema 2 one.

## 4. Kernel entries

A `kernels` entry has a `kind`, which is required and is exactly one of `"triton"` and `"binary"`.

**`kind: "triton"`** is schema 2's entry plus `kind` and `abi.scratch_slots`. Its fields are

`stem op kind arch ptx_target symbol cubin abi grid block shared num_warps num_stages launch constexprs buckets dims preconditions options
declaration_digest sha256 sha256_code toolchain` and optionally `source_sha256`, `source_digest` + `source_digest_kind` (as in schema 2:
`source_digest_kind` is `"tilefish.triton-ir.v1"` for the digest of the Triton IR, the TTIR of a `triton.jit` kernel, and `"tilefish.gluon-ir.v1"` for that of
a Gluon kernel, `@tilefish.extern(gluon=...)`: the TTGIR Gluon's frontend builds, layouts and `num_warps` included, hashed under its own tag. A Gluon kernel is
compiled by Triton and launched like any other, so its entry is a `triton` entry with nothing else that differs; a reader that does not recompute IR
digests, such as the Rust host, needs nothing for it. Python's `replay` recomputes the digest with the frontend the operator's declaration names and refuses
an entry whose kind is the other frontend's).
`toolchain` is `{"triton": ..., "ptxas": ...}`. `options` are the compile options Triton was given: `num_warps` and `num_stages` (equal to the entry's) and,
for a kernel whose declaration says it, `enable_fp_fusion` (a boolean: always for a kernel tilefish generated, and for a hand-written one declared with
`@tilefish.extern(enable_fp_fusion=...)`; without the key Triton's default applies, which is `true`). Like every field it is digested, and a replay compares it
with the compile spec of the operator.

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
| `pointer` | `*bf16`, `*fp32`, `*i32`, ... (Triton's pointer type of the dtype) | `u64`, 8 | `dtype`, `access` (`in`/`out`/`inout`), `shape`; optional `strides` and `offset` (together); on an `out` pointer, optionally `alias` (section 5.4) |
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
  * A string entry is the `dim` of exactly one `scalar` argument whose `stride_of` names this pointer and this axis (is the object
    `{"arg": <this pointer's name>, "axis": <this axis>}`, or has it in its list); that scalar carries the stride. Python checks both directions
    (`_specvalid._check_strides`): each object of a `stride_of` names a strided pointer and an axis of it, whose entry is the scalar's `dim` string, once
    per axis; each string entry has its scalar.
* `offset` is a string: the name of the **offset dimension** of this pointer. (It is `"<pointer name>_offset"`.) It is not carried by a scalar:
  the pointer passed to the kernel is already `base + offset * itemsize` (the address of the first element of the view).
* `stride_of` on a scalar is the object `{"arg": <identifier>, "axis": <int ≥ 0>}`, or **a list of two or more such objects** with distinct
  `(arg, axis)` pairs; a scalar has `stride_of` **or** `count_of`, never both.

**A stride that several views share** (`StrideOf(("q", "k", "v"), 0)` in Python, written as the list
`[{"arg": "q", "axis": 0}, {"arg": "k", "axis": 0}, {"arg": "v", "axis": 0}]` in the order of the declaration) is one scalar for an axis of several strided
pointers: a kernel that reads several views of one buffer with one stride pair, such as the q, k and v column blocks of a fused `[B, 3 * D]` tensor at
offsets 0, `D` and `2 * D`. Each of those pointers has the scalar's `dim` as its `strides` entry for that axis (so the dimension is named once in `dims`,
with its constraints), and its own `offset` and its own address. The scalar is passed once. A stride of one view is the object, as it always was, so a
manifest without a shared stride, and its digest, are unchanged; a reader that predates the list (the published crate 0.0.2, tilefish 0.0.4) refuses
a manifest that has one.

**What a host does to launch.** The host is given a view: its pointer (the address of its first element), shape, element strides (and, if it
knows it, the element offset from the start of the allocation).

1. Bind the shape dimensions from `shape` as in schema 2.
2. For each axis `a` of the view, let `stride[a]` be its element stride, **made canonical**: for an axis of size 1 the stride is replaced, going from
   the last axis to the first, by `1` for the last axis and `size[a+1] * stride[a+1]` otherwise (a size-1 axis's stride is never read by a kernel, so
   an `unsqueeze` or an `expand` does not pass a stride that no kernel reads and that might not fit 32 bits). Other strides are used as they are.
   A stride must be ≥ 0 (a stride of 0, as of an expanded view, is allowed); a negative stride is refused.
3. For each axis whose `strides` entry is an integer `1`: the view's canonical stride along it must be 1, else refused (kind `stride`).
4. For each axis whose entry is a name `x_s0`: bind the dimension `x_s0` to the canonical stride (it is an unknown to solve like a shape dimension:
   it appears in `dims`, and a `scalar` with `dim: "x_s0"` carries its value). Check the constraint `dims["x_s0"].divisible_by` if present. A name that
   several views share (a shared stride) is one unknown: the first of those pointers in the ABI binds it, and each other one must have that canonical
   stride along its axis, else the call is refused (kind `stride`, naming the pointer, the stride it has and the one required), before anything is
   launched.
5. A `scalar` with `stride_of` is passed as that bound stride, as a signed integer of its `type`. **It must fit the type:** `0 ≤ v ≤ 2^31 − 1` for
   `i32`, `0 ≤ v ≤ 2^63 − 1` for `i64`; else refused. (The `type` is `i64` exactly when the op declares `index_bits=64`; every other scalar, a
   dimension or an element count, stays `i32`.)
6. The pointer passed is the view's address; the `pointer_alignment` precondition (section 6) is checked on it.
7. The offset dimension is not bound from the stride scalars; a host that knows the element offset *may* check `dims[offset].divisible_by` against
   it, but the address check of step 6 is what is required (for a base that is 16-byte aligned they are the same check).

`max_extent` (section 6) bounds the view. For aliasing, a strided tensor occupies `[address, address + extent * itemsize)`, with `extent` as there;
a tensor that is written (`out`, `inout`) must not overlap any other argument. (Only `in` pointers are strided in this version: the declaration of an
`inout` or `out` strided pointer is refused by the Python frontend, and both readers refuse `strides` on a pointer whose `access` is not `in`.)

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

### 5.4 An `out` pointer that may be written over an input: `alias`

`"alias": "<name>"` on a pointer says that the kernel may be launched with this pointer and the pointer `<name>` as **exactly one tensor**: an
in-place kernel with two pointers, such as `k(x, r, y, r_out)` that reads every element of `r` before it writes the same element of `r_out` (Python:
`r_out: Out[Tensor[R, 64, f32], MayAlias("r")]`, which only `@tilefish.extern` declares: it is the hand-written kernel's promise). The rules, which
both readers check:

* `alias` is a string, on a `pointer` whose `access` is `out` (a `scalar`, a `fixed` argument and any other pointer have none);
* it names a `pointer` of the same ABI whose `access` is `in`, that has no `strides` (a whole tensor) and a non-empty `shape` (not a device scalar),
  with the same `dtype` and the same declared `shape` (the same list, entry for entry);
* at most one pointer names a given input.

**What a host does.** The no-alias rule of a launch (section 5.1: a tensor that is written must not overlap any other argument) has one exception:
the `out` pointer and the `in` pointer it names may be the same tensor, **one address, one shape and the same canonical strides** (Python:
`_launch.same_tensor`; Rust: `plan.rs`). Any other overlap of the two (a shifted view, another shape) is refused with the kind `alias`, as is any
overlap of either with a third argument. A pointer without `alias` is what it always was, so a manifest without it, and its digest, are unchanged;
a reader that predates it (the published crate 0.0.2, tilefish 0.0.4) refuses a manifest that has one (an unknown field). The declaration digest of
the operator has it (`"alias"` on the parameter), only when present.

## 6. Preconditions

The list is as in schema 2, with these differences.

* `pointer_alignment`: `bytes` is the alignment the pointer *passed* must have (`address % bytes == 0`). For an ordinary pointer it is 16, as before.
  For a pointer that has `strides`: **either 16, or the size in bytes of its element** (`max(1, bits // 8)` of its dtype). 16 means the operator was
  declared as 16-byte aligned (`Strided[..., 16]`), the offset dimension is `divisible_by` the elements in 16 bytes, and, for a Triton kernel, the
  pointer has `tt.divisibility=16` (see below); the element size means no alignment is assumed beyond the element's own, and Triton was given no hint.
  A device scalar's is 16.
* new kind **`max_extent`**: `{"kind": "max_extent", "arg": <pointer>, "index": <its position>, "value": 2147483647}`, once per pointer that has
  `strides` of a kernel that indexes with `i32`, and no more than once on any pointer (and only on a pointer that has `strides`; `max_elements` is not allowed on it). It says: for a kernel that indexes with `i32` (`index_bits=32`, the
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

**Which tensors bind a dimension.** A launch binds the dimensions from the tensors the kernel reads (`in` and `inout`: their shapes, and the strides of
their views) and then, for the dimensions those leave unbound, from the shapes of the `out` tensors it is given, in the order of the ABI: an `out` tensor
whose declared shape mentions such a dimension is checked on its own first (its dtype, its rank, that it is not empty and that it is contiguous), and its
shape equations are solved with what the inputs bound. Every dimension is then checked as any other: its `divisible_by`, its buckets and the `bucket` of
the binary, the scalar that carries it. A dimension that only an `out` tensor determines is that of an extern that writes a `[B]` mask from a device
scalar alone, `key_mask(mask: Out[Tensor[B, bf16]], n_valid: DevScalar[i32], b: B)`: nothing in its entry is new, and a host that binds only from the
tensors the kernel reads refuses its launch as ambiguous (the crate 0.0.2 and tilefish 0.0.4 do). Python's single launch is given the tensors it writes
with `tilefish.run(op, ..., outputs=...)` (without them it cannot allocate them, and says so); a plan step and a Rust launch always have them. The
declaration of such a dimension is allowed for an extern (a kernel's shapes come from its inputs), never for the offset dimension of a strided
pointer.

### 7.1 Dimension expressions: the limits both readers hold

A shape axis, a stride dimension, a scalar's `dim`, the `dim` of a `divisible` precondition, an axis of the `shape` of a `max_elements` precondition and
every grid entry are *dimension expressions*: integers, names, `cdiv(a, b)`, `+`, `-`, `*`, `//`, unary `-` and parentheses (`_specvalid._ExprParser`
in Python; `crates/tilefish/src/expr.rs` is a port of it). An expression is at most 512 characters and 256 tokens, nested at most 24 deep, and an integer
literal has at most 12 digits (`1234567890123` is two tokens, and refused). Both readers parse every one of them when they read the manifest (Python:
`Manifest.load`, `_manifest._check_dimension_names`), so both refuse the same texts.

**At most 32 names per kernel entry.** The unit is the entry, not a shape: the names of `dims` and every name the expressions above mention, together,
are at most 32 (`MAX_DIMS` in `expr.rs`, `tilefish._manifest.MAX_DIMS`). A strided pointer of rank `r` takes up to `r + 1` of them (a stride
dimension for each axis whose stride is not the fixed `1`, and its offset dimension). An entry with a 33rd name, in `dims` or only in an expression, is
refused with "a kernel may use at most 32 dimension names" (`TilefishManifestError` in Python, `Error::Manifest` in Rust), and a name an expression uses
that `dims` does not have with "dims does not name [...]". (The Rust runner of a launch plan binds at most 32 dimensions of a plan too.)

**Folding in 128 bits.** To know the value of an expression that is a constant (a static grid entry, the divisor of a `//` or a `cdiv`) and to compare two
expressions, a reader folds it to its affine form, `constant + Σ coefficient × atom` (an atom is a name or a `cdiv` that does not fold): what Python's
`DimExpr` is. The Rust reader folds in signed 128-bit integers, `[-2**127, 2**127 - 1]`, with checked arithmetic, and Python's reader folds in the same
range (`_specvalid.FOLD_RANGE`; its integers are unbounded, so it checks). With 12-digit literals only a product of four of them (or a sum of such
products) leaves the range: `549755813888*549755813888*549755813888*1024` is `2**127`. A fold that leaves the range has **no folded value**: a
constant or a coefficient outside it, a negation of `-2**127`, and so `a - b` when `b` holds `-2**127` (it is `a + (-1) × b`), whatever the difference
would be. Such an expression is then read as one that is not affine: nothing is refused for it, and the checks that need its value are not made. So,
with `X` that product and `Y = 549755813888*549755813888*549755813888*512` (`2**126`):

* `(X - X,)` is not the static grid `(0,)` for either reader and is read; `(Y - Y,)` is, and is refused ("entry 0 is the static size 0; grid sizes must be
  at least 1"); likewise `cdiv(n, X - X)` is read and `cdiv(n, Y - Y)` refused ("cdiv by 0 (the divisor must be positive)");
* a `cdiv` by a symbolic divisor whose quotient is beyond the range (the one case is `-2**127` by `-1`: `cdiv(-X*n, -n)`) stays the `cdiv` atom, so
  `(cdiv(-X*n, -n)*0,)` is the static `0` and refused, and `(cdiv(-X*n, -n) - Y - Y,)` is read;
* where two expressions are compared (the Rust reader's checks of the element count a `count_of` scalar carries and of the `shape` of a `max_elements`
  precondition, and the same checks of a compile spec in Python), an expression without a folded value is equal to no folded one.

No manifest `tilefish` writes comes near the range. At launch the values are computed, not folded: the Rust host computes the value of a folded form in
128 bits, and an expression that is not affine as it is written with every intermediate result at most `2**100` in magnitude, and refuses an overflow
("a dimension expression overflows"); Python computes exactly. Either refuses a value that does not fit the type the kernel takes it as.

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
accumulator before that rounding (`CUBLASLT_EPILOGUE_BIAS`). `M`, `N`, `K` are static or bucketed: one entry per bucket combination (the stem names
the buckets, like a kernel's). `N` and `K` may be any size (tilefish 0.0.5; 0.0.4 and crate 0.0.2 required multiples of 8): where a row of `k` or `n`
bf16 elements does not start on 16 bytes, the leading dimensions are what decides which algorithms are valid, and `cublasLtMatmulAlgoCheck` (which
sees them) judges the pinned one when it is pinned, when a host loads it and, in Python, before every run; the pointers stay 16-byte aligned.

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
| `sm_count` | int ≥ 1 (a JSON integer: not `30.0`, not `true`, not `"30"`; no upper bound): the multiprocessor count (`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`, torch's `multi_processor_count`) of the device `algo` was pinned on. A MIG slice has fewer SMs than the whole GPU (a B300 slice of 1g has 30, the GPU 148), and an algorithm that partitions its work by the SM count (stream-K) is chosen for that count |
| `declaration_digest` | 64 lowercase hex digits: sha256 over the canonical declaration (name, impl, workspace limit, signature with the dimensions' constraints, and the options of section 9.1 that are not the default) **and the `sm_count`** (the canonical declaration object with the key `"sm_count"` added, format 2): a pin is the declaration on one device |
| `heuristic` | *optional* (section 9.1): absent, or exactly `"torch"` |
| `algo_source` | *optional* (section 9.1): absent, or exactly `"imported"` |
| `allow_bf16_reduction` | *optional* (section 9.1): absent, or exactly `true` |

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
2. Refuse if the device's multiprocessor count is not `sm_count`, with an error that says to re-export the manifest on this device (Python, before any
   other field of the entry is compared: `TilefishManifestError`, "the manifest pinned the cuBLASLt algorithm of '<stem>' (<arch>) on a device with X SMs,
   but this device has Y: ... re-export the manifest on this device"). An entry is found by `(stem, arch)`, and two devices of one architecture may
   differ in their SM count (a MIG slice and the whole GPU), so the architecture alone does not say the pin is for this device.
3. Refuse if the process's `cublasLtGetVersion()` is not `cublaslt_version`: an algorithm id names other kernels in another version. (Python:
   `TilefishManifestError`, "the manifest pinned the cuBLASLt algorithm of '<stem>' (<arch>) with cuBLASLt X, but this process has cuBLASLt Y".)
   **The build number is not compared, because cuBLASLt does not expose one**: `cublasLtGetVersion()` is `major * 10000 + minor * 100 + patch`, and
   `cublasLtGetProperty` answers the same three (`MAJOR_VERSION`, `MINOR_VERSION`, `PATCH_LEVEL`); the 12.8.4 library has no other version or build
   entry point (checked). Two builds of one version are therefore not told apart by this check.
4. Rebuild the `cublasLtMatmulAlgo_t` from `algo`; ask `cublasLtMatmulAlgoCheck` for the problem (the algo must be valid, and its
   `workspaceSize` must be ≤ `workspace_bytes`), then `cublasLtMatmul` with a workspace of at least that size on the launch stream. A workspace the
   caller gives (a launch plan's arena buffer, the last tensor of a Rust launch) is **256-byte aligned**, which both hosts require.
   **`cublasLtMatmulAlgoCheck` is a weak judge of the bytes**: measured on 12.8.4 it accepts a changed version field, changed padding, a reserved CTA
   swizzling and a garbage custom option, and it crashes the process on a tile id ≥ 2^31. The Python binding therefore first rebuilds the algorithm
   with `cublasLtMatmulAlgoInit` + `cublasLtMatmulAlgoConfigSetAttribute` from the configuration the bytes read back as and requires identical bytes,
   requires the tile and stages ids to be in the capability lists of the algorithm id, the swizzling and the custom option to be in range, an f32
   accumulator, alignment capabilities ≤ 16 bytes, and split-K to go with the reduction scheme `NONE` or `COMPUTE_TYPE` (f32 partial sums; a bf16
   partial sum would round more than once), and no split-K at all for `gemm_bias` (with the bias epilogue, a split-K with f32 partial sums was measured
   to add the bias to a sum already rounded to bf16: cuBLASLt 12.8.4 on an RTX 5090, sm_120, algorithm id 21, where 32x1024x3072 has 8409 of 32768
   elements that are not the correctly rounded result, past the bound one rounding and an f32 accumulation allow, while the same algorithm without the
   bias, or without split-K, keeps it); or, for an entry with `allow_bf16_reduction: true`, also `INPLACE` (1) or `OUTPUT_TYPE` (4), and split-K with a
   bias. The rule, with its words, is `tilefish._cublaslt.reduction_problem` (and the crate's `reduction_problem`, `gemm.rs`): no split-K has no scheme
   and split-K has one; a scheme other than 0, 1, 2 and 4 is refused. A host that runs a `library` entry should do the same. What no byte check can see is a flip to *another
   valid* algorithm: only the digest of the manifest vouches for the file.
5. The pin belongs to one `arch` and one SM count: an entry is found by `(stem, arch)` for the architecture of the device, and a missing pair is an error
   naming the architectures the manifest has; the entry found must have the device's `sm_count` (step 2).

**Pinning (how an entry is made)**, for the writers: the heuristic is asked on a device of `arch` with the preferences `MAX_WORKSPACE_BYTES =
workspace_bytes`, `MIN_ALIGNMENT_{A,B,C,D}_BYTES = 16` and a reduction-scheme mask of `NONE | COMPUTE_TYPE` (`NONE` alone, the mask `0`, for
`gemm_bias`: step 4), for a few results (another posture, or an
imported algorithm: section 9.1); result 0 is the pin, and the device's multiprocessor count is `sm_count`. The pin is the configuration of result 0 (its
nine `CUBLASLT_ALGO_CONFIG_*` attributes) **encoded again** with `cublasLtMatmulAlgoInit` and `cublasLtMatmulAlgoConfigSetAttribute`, not the 64 bytes the
heuristic returned: those can hold bytes that are no part of the configuration (measured with cuBLASLt 12.8.4 on an RTX 5090, sm_120: bytes 34 and 35 of
split-K algorithms of id 21, which change from process to process), so the bytes it returned are neither reproducible nor what the check of step 4
accepts. The configuration is all a host rebuilds and checks, and the re-encoded algorithm runs to the bytes torch gets from the returned one (section 9.1,
`test_the_torch_posture_pins_what_torch_runs`). Before an entry is written its algorithm passes the checks of a run (step 4). `jit` pins in memory the same way, for the device it runs on (its in-memory pins are kept per device), so
`jit` and `replay` run the same algorithm by construction. A pin does not bind itself to its problem (the same bytes are accepted for other problems), which
is why `m`, `n`, `k` and the epilogue are in the entry. Re-exporting on a device of another SM count makes another entry for the same `(stem, arch)`, which
replaces the old one only with `overwrite=True` (its `sm_count` and `declaration_digest` differ).

**Digest.** The entry is digested whole, in `library` (section 3); `algo`, `cublaslt_version` and `sm_count` are in it (and `sm_count` is in its
`declaration_digest` too).

### 9.1 How the algorithm was chosen: three optional fields (tilefish 0.0.5, crate 0.0.3)

A declaration says how its algorithms are chosen (`@tilefish.library(..., heuristic=, allow_bf16_reduction=, algos=)`, `python/src/tilefish/_library.py`).
Its entries say it in three fields that are **absent when the declaration has the default, and present only with their one other value**; a reader
refuses any other value, `null` and the default written out included (Python `validate_entry`, `OPTIONAL_ENTRY_FIELDS`; Rust `LibraryEntry::from_raw`).
So an entry of a declaration that uses none of them is exactly what tilefish 0.0.4 wrote (its digest and `declaration_digest` are unchanged), and a
reader of 0.0.4 or of crate 0.0.2 refuses an entry that has one (an unknown field), which it would otherwise run against its meaning.

| field | absent (the default) | present |
|---|---|---|
| `heuristic` | the posture above: a few results, `MIN_ALIGNMENT_{A,B,C,D}_BYTES = 16`, the bias pointer the heuristic sees is a 16-byte aligned dummy | `"torch"`: what torch 2.9 asks for `torch.addmm(bias, a, b.t())` (`at::cuda::blas::gemm_and_bias`): **one** result, under `MAX_WORKSPACE_BYTES` alone (torch sets the alignment minimums to the alignment of its pointers, 256 bytes from its allocator, which is the library's default, so none is set), a 256-byte aligned dummy bias pointer. The declaration's default `workspace_bytes` is then torch's: `1048576` for `cublaslt.gemm_bias` (`CUBLASLT_WORKSPACE_SIZE`), `8519680` for `cublaslt.gemm` (`torch.mm` runs `cublasGemmEx` on a handle with `CUBLAS_WORKSPACE_CONFIG`'s workspace, and the heuristic asked this way under that limit answers what it runs; torch's default is 32 MiB on sm_90 exactly) |
| `allow_bf16_reduction` | the heuristic is asked with the reduction-scheme mask `NONE \| COMPUTE_TYPE` (`gemm_bias`: `0`, no split-K), and an algorithm whose split-K adds its partial sums in bf16, or a split-K of `gemm_bias`, is refused (step 4) | `true`: no mask (torch's default, `allow_bf16_reduced_precision_reduction = True`; the mask `NONE \| COMPUTE_TYPE` is what torch sets when it is `False`), and `INPLACE` (1), `OUTPUT_TYPE` (4) and a split-K of `gemm_bias` are accepted: the result may be rounded more than once |
| `algo_source` | `algo` is the heuristic's result 0, re-encoded | `"imported"`: the declaration's `algos=` gave it (`tilefish.AlgoTable(arch, sm_count, cublaslt_version, {(m, n, k, bias): config})`, one table per device; a config is the nine attributes, encoded with `cublasLtMatmulAlgoInit` and the setters, or 128 hex digits taken as they are). The heuristic is not asked; the algorithm passes step 4 before the entry is written; a device no table of the declaration is for is refused at the build (never handed to the heuristic). The rows of the tables the declaration uses (and the tables' devices) are in its canonical declaration, so a change of one changes `declaration_digest`; `replay` also compares `algo` with what the table gives for the device |

These fields are provenance for a host that runs the pin (it runs `algo` either way), except `allow_bf16_reduction`, which step 4 needs. With the
`"torch"` posture and `allow_bf16_reduction`, the pinned result is torch's byte for byte (`python/tests/test_gpu_cublaslt.py`,
`test_the_torch_posture_pins_what_torch_runs`, on 48 shapes with and without a bias and three operand sets each; measured on a B300 MIG slice, sm_103, and
an RTX 5090, sm_120, with cuBLASLt 12.8.4).

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
  `"tilefish.binary.v1"`, no `source_digest`, a `source_sha256`, a cubin used as it is whose `source_digest` is not its `sha256`; the cubin file name not a plain file name; the cubin's `sha256`, `sha256_code` or
  `.text.<symbol>` section wrong; for `triton` the shared-memory section rule.
* what an entry says of itself, which both readers check when they read it (Python: `Manifest._read`, as `crates/tilefish/src/manifest.rs` does; the
  digests of a document a caller pinned again vouch for the document, not for its sense): a missing `op`, `abi`, `grid`, `buckets`, `dims`,
  `preconditions` or `declaration_digest`; an `arch` that is not `sm_` and two or three digits, or that is not the architecture the cubin's ELF header
  names (`_elf.sm_of`, the Rust `Elf::sm`: an entry relabelled `sm_90` that holds an sm_103 cubin), or, for `binary`, a header that names none; a `stem`
  that is not `op` and `.<dim><value>` for each of `buckets`, in order (`rowop.S32`); a `declaration_digest` that is not 64 lowercase hex digits (`null`
  included); a parameter with a field no parameter has, without `slot`, `index`, `name`, `kind`, `type`, `ptx_type`, `size`, or at another `slot` than
  its position, and the fields of each kind (`read_params`): a `pointer` whose `type` is not the pointer type of its `dtype`, whose `access` is not
  `in`/`inout`/`out`, that has `dim`, `count_of`, `stride_of` or `value`, or an `alias` that is not a string or is on an access other than `out`, is not a `u64` of 8 bytes, or has no `shape` (axes: an int ≥ 1 or an
  expression), or has `strides` and `offset` not together, on an access other than `in`, not one per axis of its shape (a name, or the fixed `1` on
  the last axis only), or an `offset` that is not a name; a `scalar` that has `dtype`, `access`, `shape`, `strides`, `offset`, `value` or `alias`, no `dim`, both
  `count_of` and `stride_of`, or another width than `u32`/4 or `u64`/8; a `fixed` argument on a Triton kernel, or with a `dim`, `count_of`,
  `stride_of`, `access`, `shape`, `strides`, `offset`, `dtype` or `alias`, or without a `value` of its type (a JSON `null` is as absent, in each); an
  expression outside the grammar, a name that `dims` does not have, more than 32 names, a static grid entry or divisor below 1 (section 7.1).
* the ABI as a whole (a host that reads without a spec checks these itself; Python checks them by comparing the entry with the compile spec, and
  `_specvalid` checks them in the spec): `params` kinds in `pointer`/`scalar`/`fixed`; a `scalar` type other than `i32`/`i64`, or `dtype` ≠ `type`; `strides` without
  `offset` or the reverse; `strides` of the wrong length or empty, with an integer other than `1`, an integer on a non-last axis, a string that no
  `stride_of` scalar carries, or two scalars for one axis; an `offset` that is not a string; a `stride_of` list of fewer than two objects or with an
  `(arg, axis)` pair twice; a `stride_of` object with a missing/extra key, a negative axis, an
  axis out of range, a target that is not a strided pointer, a `dim` different from the target's `strides` entry, or together with `count_of`; a `fixed`
  argument with a type not in `i32`/`i64`/`f32`, an `f32` value that is not a float, an integer type with a non-integer, a bool, or an out-of-range value;
  an `alias` (section 5.4) that names no `in` pointer, a strided one or a device scalar, one of another `dtype` or declared `shape`, or an input another `out` pointer names.
* preconditions: `max_extent` more than once on a pointer (a kernel with `index_bits=64` has none, so it is not required), with a value other than `2147483647`, or on a pointer without `strides`;
  `max_elements` on a pointer with `strides`; `pointer_alignment` of a strided pointer other than 16 or its element size (of any other pointer other
  than 16); every pointer must have exactly one `pointer_alignment`.
* every `library` entry: the field list, the types and the cross-field equalities of section 9, each by the rule given there; an optional field of section 9.1
  with any value but its one (`"torch"`, `"imported"`, `true`), `null` included.
* *spec* only, for prebuilt binaries: a `binary` section that is not `{kind, symbol, shared, digests}`; `attrs` not empty; `launch_params` not every arg
  in order; a `block` that is not three ints from 1 to 1024 with product ≤ 1024; a `constexpr` argument.

## 11. Migrating a schema 2 manifest

A schema 2 manifest of Triton kernels is the same manifest in schema 3 after exactly these changes, and its cubins are untouched
(`python/tests/schema2_fixtures.py: migrate_doc`):

1. `schema` becomes `3`;
2. every kernel entry gains `"kind": "triton"` and its `abi` gains `"scratch_slots": 2`;
3. the manifest gains `"library": []`.

Its digest changes (tag `v3`, and the `library` key), so a pin must be taken again. Both hosts refuse schema 2 (`Re-export it` / `Export it again`),
checked before anything else of the file is looked at, so a manifest of schema 2 is told to be exported again and not that a field is missing. The
committed fixtures of the Rust crate (`crates/tilefish/tests/fixtures`, built on a B300 by `python/tests/export_rust_fixtures.py`) are schema 3
manifests, and `expected.json` holds the digests Python computed for them.

## 12. What the Rust host does

The Rust crate `tilefish` of this repository reads schema 3 in full and launches everything this document describes, with the safety model of its
schema 2 launches (`DevicePtr<'a>` lifetimes, `from_cuda` sealed to cudarc's slices and views, the safe `launch` on a `CudaStream` with owner waits and
writer-waits-for-readers, the refusal of a capture of slices; `unsafe` only in `region.rs` and `ffi.rs` with its `ffi/cublaslt.rs`; cudarc's safe layer
and its `cublaslt::result`, no hand-written driver FFI). Where it is stricter than Python, or differs from it, it is listed here.

**Reading** (`crates/tilefish/src/manifest.rs`, `library.rs`).

* Everything of section 10 is refused, and the digest of section 3 is computed exactly (`manifest_digest`, with the schema 2 form for any other `schema`).
  `abi.scratch_slots` is required (a host without a spec must require it). `max_extent` is optional: *at most once, and only on a strided pointer* (section 6
  says "once per strided pointer", and a kernel with `index_bits=64` has none, so Python's reader does not require it). `max_elements` or `max_extent`
  on a device scalar is refused, `fixed` on a Triton kernel is refused (section 5's table), `strides` on a pointer that is not `in` is refused
  (section 5.1 allows a reader to), and a stride entry is a dimension expression that must be exactly the `dim` of the scalar that carries it.
* A cubin is checked against the architecture of its entry when it is read, by both readers: `Elf::sm()` (Python's `_elf.sm_of`: the SM in the low byte of `e_flags` for OS
  ABI `0x33`, in the second byte for `0x41`). A prebuilt binary whose header names no readable architecture, or another one than its entry, is refused
  (section 8); a Triton cubin is refused when it names another one, and left to the driver when the header cannot be read.
* The function the driver loads must take the parameter sizes of the ABI (`cuFuncGetParamInfo`, as Python checks at load, for every kernel).
* An entry may name at most 32 dimensions (a strided pointer has one per axis and one for its offset), and its expressions are folded in 128 bits
  (section 7.1); Python's reader holds both.

**Launching a kernel** (`plan.rs`). The steps of section 5.1, in Python's order: shapes, strides and scalars bind the dimensions (then the `out` tensors bind
the dimensions those leave unbound, section 7), `divisible_by`, the exact
bucket, the range of every scalar (`i32` or `i64`), aliasing (a view by its extent), the alignment each pointer promises, `max_elements`/`max_extent`,
the grid. The kinds of the refusals are Python's (`stride`, `max_extent`, `contiguous` are new `PreconditionKind`s). A view is
`TensorRef::from_cuda_strided(&slice, &shape, &strides, offset)` (also `TensorMut`, and `new_strided` over a region): element strides, which are
unsigned here, so a negative stride cannot be made (section 5.1 step 2 refuses it); the view's extent from its first element must fit its memory
(refused at construction); the offset is in elements, may start at any byte of a slice of any element type, and the kernel is given the address of the
first element (for a slice, cudarc is handed a view of the bytes from there, so its tracking of the slice is unchanged). A tensor that is not a view is a
view with the dense strides. A view given to a parameter that is not strided must be contiguous (torch's rule: the strides of size-1 axes are not looked at).
The offset dimension of a strided pointer is not bound (section 5.1 step 7): the check of the address is the one made. A device scalar is a tensor of shape
`[]` or `[1]`. `fixed` values are written at their slots as their type says (an `f32` is the JSON double rounded to the nearest `f32`).

**Launching a library entry** (`gemm.rs`, `ffi/cublaslt.rs`). `Manifest::gemm(&ctx, name, buckets, arch)` loads a `Gemm` for the device of `ctx` once and
`Gemm::launch(&stream, &mut [a, b, [bias], out, workspace])` runs it:

* Load refuses, in Python's order: a cuBLASLt that cannot be opened (cudarc opens it when first used, and *panics* if it is missing, so `ffi/cublaslt.rs` asks
  first), a cuBLASLt whose `cublasLtGetVersion()` is not `cublaslt_version` (an error naming both versions), an algorithm that fails the vetting of section 9
  (the checks of Python's `_validate_algo`: rebuilt with `cublasLtMatmulAlgoInit` and the configuration attributes and byte-for-byte equal, tile and stages ids
  and swizzling and custom option within the capabilities of its id, an f32 accumulator, alignment at most 16 bytes, split-K with a reduction scheme of `NONE` or
  `COMPUTE_TYPE`, or also `INPLACE` or `OUTPUT_TYPE` for an entry with `allow_bf16_reduction`), an algorithm `cublasLtMatmulAlgoCheck` does not approve for the
  problem, and one that needs more than `workspace_bytes`. The optional fields of section 9.1 are read into `LibraryEntry::heuristic` (`Heuristic`),
  `algo_source` (`AlgoSource`) and `allow_bf16_reduction`. The check runs once at
  load, not at every launch (the problem of an entry is fixed); the descriptors of a call are made and destroyed in the call.
* The process must map the same cuBLASLt build that chose the pin (cudarc tries `libcublasLt.so` and then `.so.<major>`, `.so.12`, ...; on a machine whose torch
  ships its own cuBLASLt, put that directory on `LD_LIBRARY_PATH`): a different build is refused by the version check, and by nothing else.
* **The workspace is the caller's**, the last tensor of a launch: a `TensorMut<U8>` (cudarc slice or region) of at least `Gemm::workspace_required()` bytes (the
  algorithm's need, at most `workspace_bytes`; `workspace_bytes` is what Python provides and what `Gemm::alloc_workspace` allocates), 256-byte aligned,
  overlapping no operand. Nothing is allocated in a launch, so a launch can be captured. Two launches that share a workspace are ordered by cudarc's write tracking
  and the crate's writer-waits-for-readers rule (on one stream, by stream order); a graph that shares one between replays is the host's to order.
* The operands go through the plan of the entry (shapes bound to the entry's dimensions, the exact bucket, dtype, contiguity, 16-byte alignment, device,
  context, aliasing), the stream and context are checked, and the slices are ordered as a kernel's (cudarc's wait and record through the guards of
  `device_ptr`, the owners' wait for the end of the work, a refusal while a stream is capturing). `unsafe Gemm::launch_raw` with regions captures a GEMM.

**The emitter** (`tilefish.hosts.rust`). It writes a struct for each kernel (Triton or prebuilt: `launch` takes every pointer as a `TensorRef` or `TensorMut`;
stride scalars and `fixed` arguments are not parameters) and for each library entry (`launch` takes `a`, `b`, `bias`, `out`, `workspace`; `gemm()`, `M`, `N`, `K`,
`BIAS`, `WORKSPACE_BYTES`, `PINNED_WITH`), with the digest of the manifest embedded. It refuses what the reader refuses.

**Launch plans** (not read by the crate yet). `docs/plan-schema-1.md`: `plan.json` pins a manifest and orders the launches of its entries over one arena, so a host runs a whole forward from it, capturable in a CUDA graph (`test_gpu_plan.py`; the golden fixtures `python/tests/golden/plans/` have the digests to reproduce).

## 13. Where it is tested

| what | tests (`python/tests/`) |
|---|---|
| top level, digest, the three additions, refusal of schema 2 | `test_manifest.py`, `test_manifest_schema3.py`, `test_rust_host.py` |
| `kind`, `abi.scratch_slots`, prebuilt binary entries | `test_binary_externs.py`, `test_manifest_schema3.py` |
| strided pointers, `strides`/`offset`/`stride_of`, `max_extent`, alignment | `test_strided_specs.py`, `test_views.py`; on a GPU `test_gpu_strided.py` |
| a stride that several views share (`stride_of` a list) | `test_shared_strides.py`, `test_rust_host_schema3.py`; on a GPU `test_gpu_shared_strides.py` |
| a dimension only an `out` tensor determines (section 7) | `test_output_dims.py`; on a GPU `test_gpu_output_dims.py` |
| an `out` pointer that may be written over an input (`alias`, section 5.4) | `test_alias.py`; on a GPU `test_gpu_alias.py` |
| device scalars | `test_devscalar.py`; on a GPU `test_gpu_devscalar.py` |
| prebuilt PTX and cubin, byte identity with the Triton path | `test_binary_externs.py`; on a GPU `test_gpu_binary.py`, `ptx/*.ptx` |
| the `library` section | `test_library_ops.py`, `test_cublaslt.py`; on a GPU `test_gpu_gemm.py`, `test_gpu_cublaslt.py` |
| its optional fields (section 9.1): the posture, imported algorithms, bf16 split-K partial sums | `test_library_options.py`, `test_cublaslt.py`; on a GPU `test_gpu_cublaslt.py` (torch's bytes, the opt-in), `test_gpu_gemm.py` (jit and replay) |
| the emitter | `test_rust_host.py`, `test_rust_host_schema3.py` (goldens in `python/tests/golden/`) |
| `options.enable_fp_fusion` of a hand-written kernel (`@tilefish.extern(enable_fp_fusion=...)`) | `test_lower_options.py`; on a GPU `test_gpu_extern_fusion.py` |

The Rust crate (`crates/tilefish/tests/`):

| what | tests |
|---|---|
| strict reading of schema 3, the older schemas | `strict_reading.rs`, `library_section.rs` (the optional fields of section 9.1 too), `schema3.rs` (synthetic manifests, `common/synthetic.rs`) |
| views, device scalars, fixed arguments, binaries, the plan of a GEMM, the digests | `schema3.rs`, `library_section.rs`, `launch_checks.rs` |
| Python's decision on 992 calls (views, device scalars, binaries included), the digest vectors, Python's refusals of digesting | `python_launches.rs`, `python_canonical.rs`, `cross_language.rs` |
| on a GPU, byte identity with Python's `replay` of every feature, the refusals, graph capture of a GEMM and of a strided kernel, the hazards of other streams | `gpu_gate.rs` with `python/tests/rust_gate.py` |
