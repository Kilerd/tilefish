# Manifest schema 3

**English** | [简体中文](../zh-CN/manifest-schema-3.md)

This is the specification of what manifest schema 3 adds to schema 2. The Python package writes and reads schema 3 (`tilefish.runtime.export`,
`tilefish.Manifest.load`), and so does the Rust crate `tilefish` of this repository (the published 0.0.1 reads schema 2 only; section 12 says what the
crate does with each part of this document). Everything here is checked by tests in `python/tests/` and in `crates/tilefish/tests/` (named in the last
section), and the Python code that writes each field is named where the field is described.

A reader of this document is assumed to know schema 2: the entry fields of a Triton kernel, `abi`, `dims`, `preconditions`, the dimension
expression grammar, the canonical JSON of the digest, and the launch checks the crate already makes. Only what is new or changed is specified; a
field that is not mentioned is as in schema 2.

Contents: 1 what changed; 2 the top level; 3 the digest (3.1 what a pin vouches for, reproducible exports); 4 kernel entries (4.1 an implementation per
architecture, 4.2 warp-specialised kernels, 4.3 persistent grids, 4.4 a configuration per bucket, 4.5 thread-block clusters, 4.6 a global scratch,
4.7 the CUDA math functions of a generated kernel, 4.8 the probes of an instrumented build, 4.9 CuTe DSL kernels); 5 the ABI (strided views, device scalars, fixed arguments, scratch slots, an `out` written over its
input, TMA descriptors, null pointers, 64-bit dimension scalars); 6 preconditions; 7 the table of dimensions; 8 prebuilt binaries; 9 the `library` section (a pinned cuBLASLt GEMM, in 9.4 a pinned cuDNN convolution, in 9.5 a GEMM with a cuBLASLt epilogue); 10 strict reading, every
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
| ABI argument kinds | `pointer`, `scalar` | adds `fixed`: a scalar whose value is in the manifest; and `given` (0.0.5, crate 0.0.3): a fixed argument whose value each call gives |
| preconditions | `pointer_alignment` always 16 bytes | a strided pointer may promise only its element alignment; adds `max_extent` |
| `dims` | names from shapes, scalars and the grid | also the stride dimensions and the offset dimension of a strided pointer |
| library | none | a pinned cuBLASLt GEMM, for one device (its architecture and SM count): `library` entries (no file) |
| library entry (0.0.5) | — | three optional fields, present only when not the default: `heuristic`, `algo_source`, `allow_bf16_reduction` (section 9.1) |
| library entry (0.0.6, crate 0.0.4) | — | f16 and f32 operands, an f32 result and the layouts `NN`, `TT` and `NT` in the fields an entry always had (`dtypes`, `transa`, `transb`, `lda`, `ldb`, the operands); the optional `allow_f16_reduction` and `allow_f32_reduction` (section 9.2) and `sm_count_target` (section 9.3) |
| kernel entry (0.0.5, crate 0.0.3) | — | an optional `implementation`: which implementation of an operator declared with an implementation per architecture the binary is (section 4.1) |
| ABI argument kinds (0.0.6, crate 0.0.4) | — | adds `tensordesc`: a TMA descriptor of a tensor (a `CUtensorMap` the host encodes at every launch, passed by value), which takes `1 + 2 * rank` slots of `kernelParams`, so `slot` is where a parameter's slots begin (section 5.5) |
| kernel entry (0.0.7, crate 0.0.5) | — | `implementation` may be the name of a named implementation, which several architectures share, and may be on a `binary` entry (a prebuilt binary as an implementation); the entries of one stem of such an operator may have an ABI per architecture (section 4.1) |
| kernel entry (0.0.6, crate 0.0.4) | — | an optional `total_num_warps`: the warps of a block of a warp-specialised kernel, whose partitions add warps to the `num_warps` of its default partition; its `block` is `32 * total_num_warps` (section 4.2) |
| library entry (0.0.6, crate 0.0.4) | — | `impl: "cudnn.conv_fwd"`: a pinned cuDNN forward convolution, an entry of fields of its own (the problem, the math, the pin of the legacy or the backend API and how it was chosen, the cuDNN version, the device) (section 9.4) |
| kernel entry (0.0.7, crate 0.0.5) | — | an optional `cuda_math` on a generated kernel: the CUDA math functions whose bits its ops are, the CUDA releases they were measured on, and the sha256 of the libdevice it links (section 4.7) |
| ABI argument kinds (0.0.7, crate 0.0.5) | — | adds `null`: a pointer every launch passes as the null pointer (Python's `Null[dtype]`), of a Triton kernel or a prebuilt binary (section 5.6) |
| ABI scalar (0.0.7, crate 0.0.5) | — | a dimension scalar (a `scalar` with a `dim` and no `stride_of`) may be `i64` (Python's `i64[n]`) where only a stride could be before (section 5.7) |
| preconditions (0.0.7, crate 0.0.5) | — | a dense pointer that is not a device scalar may promise only its element's alignment: the specialisation of a lowered kernel for views that do not start on 16 bytes (section 6) |
| `grid` (0.0.7, crate 0.0.5) | affine in the dimensions (what Python wrote) | may multiply dimensions (`(cdiv(C*T, 1024),)`): the grammar always had `*`, and both readers always took it; a host computes a grid, it never solves one (section 7.1) |
| kernel entry (0.0.7, crate 0.0.5) | — | an optional `persistent`: a persistent grid, the tiles cut at launch to the programs the SMs of the device (or of its partition) run (section 4.3) |
| kernel entry (0.0.7, crate 0.0.5) | — | `launch.cluster_dims` and `launch.num_ctas` of a thread-block cluster (a Triton kernel's `num_ctas`, a prebuilt binary's declared cluster) are launched with `cuLaunchKernelEx`, and `options.num_ctas` says it (section 4.5) |
| kernel entry (0.0.7, crate 0.0.5) | — | `launch.global_scratch_size` above 0 is launched with a scratch of that many bytes per CTA, and `launch.global_scratch_align` records its alignment (section 4.6) |
| library entry (0.0.12, crate 0.0.10) | — | `impl: "cublaslt.matmul"`: a GEMM with a cuBLASLt epilogue (`none`, `bias`, `relu`, `relu_bias`, and `beta` 1 to add `c`), its pin verified against the epilogue's formula and recorded `measured-exact` or `binary-defined` (`contract`, `verification`), and an optional `canary` (section 9.5) |
| kernel entry (0.0.11, crate 0.0.9) | — | an optional `instrument` on the entry of an instrumented build (`export(..., instrument=True)`): its probes, recorded into the buffer a host gives in the profile scratch slot; a manifest with such an entry is an instrumented export, which both readers refuse unless asked to load one (section 4.8) |
| kernel entry (0.0.13, crate 0.0.11) | — | a `binary` entry of `binary.source` `"cute"`: a CuTe DSL kernel (`@tilefish.extern(cute=...)`), its toolchain the DSL's, its `source_digest` the sha256 of its PTX (kind `tilefish.cute.v1`), its grid integers; and the ABI kinds `tensormap` (a TMA descriptor the DSL encodes: a template with the tensor's address written into its first 8 bytes) and `bytes` (constant bytes), of such an entry only (section 4.9) |

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

### 3.1 What a pin vouches for, and reproducible exports (tilefish 0.0.7)

A pin is the digest of one exported artifact: the entries above, and through each entry's `sha256` every cubin file byte for byte (debug sections
included). This is unchanged, and so are the schema, the digest function, both readers and every manifest written before.

What changed is what goes into a cubin. Triton records the file of a kernel, and of every jit function it calls, in the locations of its IR, and
`ptxas` writes them into the cubin's debug sections (`.debug_line`, `.nv_debug_ptx_txt`, `.rela.debug_line` and their `.nv.merc.*` copies). Up to
0.0.6 those were absolute paths: the checkout of a hand-written kernel, the virtual environment of Triton's own `triton/language/standard.py`, and the
random `/tmp/tilefish-gen-XXXXXXXX` directory a lowered kernel's or a generated op's source was written to. So two exports of the same operators had
the same `sha256_code` and different `sha256`s, and a different digest, from one directory or process to the next. Now a build names each file by a
path that does not depend on the host (`python/src/tilefish/_source.py`, `source_name`):

* a module `a.b.c` whose file ends in `a/b/c.py` (or `a/b/c/__init__.py`): that tail, its path relative to the `sys.path` entry it was imported from;
* a generated source (a lowered kernel, `tilefish.contiguous`, `im2col`, `im2col1d`, `col2im1d`): `tilefish_generated/<name>.py`, what
  `LoweredKernel.write(directory)` writes and imports as `tilefish_generated.<name>`; the file itself is
  `<cache>/generated/<sha256 of the source>/tilefish_generated/<name>.py` (`$TILEFISH_CACHE_DIR`, else `~/.cache/tilefish`), written atomically and used
  only when it holds the source, or a private scratch directory of the process when that cannot be written;
* any other file (a script run as `__main__`): its base name.

The names are applied to the frontend of a compile (`python/src/tilefish/_ir.py`, `host_independent_paths`), so the locations, the PTX and the cubin
all carry them. `ptxas` runs in an empty directory (`ptxas_in_an_empty_directory`; a prebuilt PTX's too): it looks each name up from its
working directory and writes the modification time and size of a file it finds there into the line table, which a build run from the
directory that holds `kernels/rms.py` would otherwise carry. Line and column numbers, and the machine code, are unchanged: an export of this version has the `sha256_code` of every binary an earlier
version built from the same sources with the same toolchain, and other `sha256`s. Two exports of the same operators with the same toolchain (Triton,
`ptxas`, cuBLASLt, cuDNN) for the same device are the same files with the same digest, from any directory, in any process, on any machine: a pin taken on
one machine checks a rebuild on another. Options that were considered and not taken:

* *let the digest cover `sha256_code` instead of the files*: a pin would no longer vouch for the bytes a host loads (the debug sections would be
  outside it), every reader would need a second digest form, and old pins would change meaning;
* *normalise the paths inside the cubins before hashing*: the same, plus an ELF rewrite in both readers;
* *drop the line information*: it removes and renumbers sections, so every `sha256_code` would change, and profilers lose the source lines.

**Absolute paths for a profiler (opt-out).** A profiler that shows source lines (Nsight Compute's source view, `nvdisasm`) resolves these names
against a source directory it is given. Where that is inconvenient, `TILEFISH_DEBUG_PATHS=absolute` in the environment of the build keeps the absolute
paths Triton records (the checkout, the virtual environment, the cache directory of the generated sources), as before. It is off by default, and it
**breaks reproducibility**: such an export has other cubin bytes, `sha256`s and digests in every other directory or on every other machine (the
`sha256_code`s are the same), so a pin taken from it checks only that very export. A binary built with it is cached under a key of its own and never
served to a build without it; any other value of the variable is refused.

A manifest exported by 0.0.6 or earlier still names the directories of its build; it loads, pins and replays exactly as before. A prebuilt binary
(section 8) is taken as it is: a PTX with `.file` directives or a cubin built elsewhere keeps whatever paths it has.

**A trace next to the manifest (tilefish 0.0.9).** An export with `trace=True` (or `TILEFISH_TRACE=1`) also writes `trace/` in the manifest's directory:
every level of every binary, for `tilefish-explain` (`docs/en/explain.md`). No field of the manifest names it and no digest covers it, so the manifest,
its digest and every pin are the same with or without it; neither reader looks at it (a reader reads `manifest.json` and the cubins it names, and nothing
else of the directory). It holds source code (the kernels' files, the generated sources, Triton's): it is not part of what is shipped, and a tool that
packages an export for a host leaves it out.

## 4. Kernel entries

A `kernels` entry has a `kind`, which is required and is exactly one of `"triton"` and `"binary"`.

**`kind: "triton"`** is schema 2's entry plus `kind` and `abi.scratch_slots`. Its fields are

`stem op kind arch ptx_target symbol cubin abi grid block shared num_warps num_stages launch constexprs buckets dims preconditions options
declaration_digest sha256 sha256_code toolchain` and optionally `implementation` (section 4.1), `total_num_warps` (section 4.2), `persistent` (section 4.3), `cuda_math` (section 4.7), `instrument` (section 4.8), `source_sha256`,
`source_digest` + `source_digest_kind` (as in schema 2:
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
toolchain binary source_digest source_digest_kind` and optionally `implementation` (section 4.1) and `persistent` (section 4.3)

with `constexprs` always `[]`, `source_digest_kind` always `"tilefish.binary.v1"`, `source_digest` always present, and no `source_sha256`.

Validation rules that are new (Python: `Manifest._read`):

* `kind` is one of the two strings; any other value, or its absence, is refused.
* If `abi.scratch_slots` is present it is an integer (not a bool) and equals 2 for `triton` and 0 for `binary`. (Python's reader checks it when the
  field is there, and the comparison with the operator's compile spec requires it to be exactly what the spec says; a host that reads without a spec
  must require it, and require `len(abi.scratch) == abi.scratch_slots`.)
* For `triton` the existing rule stands: an entry whose `shared` is 0 but whose cubin has a section `.nv.shared.<symbol>` is refused. **For `binary`
  it does not apply**: a hand-written kernel with *static* shared memory has that section too, and the declaration (`shared`) says how much
  *dynamic* shared memory it addresses; the loader's check (static + dynamic ≤ the device's opt-in maximum) is the guard.
* **A `shared` that is nonzero but too small is not detected, and cannot be from the file.** Triton gives a kernel that needs shared memory an
  `extern` array of *dynamic* shared memory, and the size is Triton's (`metadata.shared`, from its allocation of the TTGIR), which ptxas never sees:
  the cubin does not record it. Measured on sm_103 with Triton 3.5.1 and ptxas 12.9 (`python/tests/test_gpu_shared_size.py`): three `tl.dot` kernels
  whose `metadata.shared` are 16 400, 49 168 and 65 552 bytes have the same `.nv.shared.<symbol>` section (1024 bytes: the per-block system reserve
  from sm_90 on), the same `.nv.info.<symbol>` attributes but for their parameter records, and the same static shared memory as the driver reports it
  (`CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`, 0). So a loader can check what the file says (the section, and `shared ≠ 0`; static + `shared` ≤ the
  opt-in maximum) and no more; knowing the size again would mean compiling the kernel's TTGIR again at load (Python only, and seconds for a large
  set), and a manifest whose `shared` was edited may as well have had its cubin replaced. What vouches for `shared` is what vouches for the cubin:
  the pin (the manifest digest covers `shared`, and the export wrote Triton's value). Without a pin (`trust_directory=True`) the directory is
  trusted for both. The test fails when a later Triton or ptxas records the size, and a check at load becomes possible then.
* `toolchain` for `binary` is `{}` (a cubin used as it is) or ``{"ptxas": <the `--version` text>}`` (PTX that was assembled).

### 4.1 An implementation per architecture: `implementation` (tilefish 0.0.5, crate 0.0.3; names, prebuilt binaries and an ABI per implementation: tilefish 0.0.7, crate 0.0.5)

An operator declared with an implementation per architecture (`@tilefish.extern(by_arch={"sm_103": tilefish.impl(...), "sm_120": ...},
fallback=...)`: one public signature, and for each architecture a Triton or Gluon kernel with its own compile options, constexpr values and grid, or a
prebuilt PTX or cubin) has, on each architecture, the binaries of the implementation of that architecture: the one declared for it, else the
`fallback`, and an architecture with neither is refused by the export (and by `jit` and `replay`). The entry of such a binary is a `triton` or a
`binary` entry with one more field:

| field | type and rule |
|---|---|
| `implementation` | optional string: `arch` (the entry's own architecture: the implementation declared for it), `"fallback"`, or the **name** of a named implementation: an ASCII identifier (`[A-Za-z_][A-Za-z0-9_]*`) of at most 64 characters that is not `fallback` in any case and does not begin with `sm_` and a digit in any case (`tcgen05`, `ptx`, `fallback2`; not `Fallback`, `sm_120`, `SM_90a`). Absent (or `null`) for an operator with one implementation. On a `triton` or a `binary` entry. |

* **Named implementations.** `tilefish.impl(..., name="tcgen05")` is one implementation for every architecture the map sends to it (`by_arch={"sm_100":
  tc, "sm_103": tc}`, and `fallback=` may be it, or be named too): its key is its name, so those architectures share one declaration digest and one set of
  compile specs, and their entries say `"implementation": "tcgen05"`. Two implementations that are not the same declaration may not share a name (two
  `tilefish.impl` objects that declare the same kernel, options, constexpr values, grid and stub under one name are one implementation). Without a name an
  implementation's key is the architecture it is declared for, or `"fallback"`, as before: one implementation object under two architectures is two
  implementations, with two keys, two digests and two sets of specs, and the digests of such declarations are those of 0.0.6.
* **Prebuilt binaries.** `tilefish.impl(ptx="k.ptx", symbol=..., block=..., shared=...)` and `tilefish.impl(cubin={"sm_120": "k.cubin"}, symbol=...,
  block=...)` are implementations (a relative path is relative to the file that declares the operator, as for `@tilefish.extern(ptx=...)`). Their entries are
  `binary` entries (section 8) with `implementation`. A cubin implementation has a cubin for every architecture that runs it: the declaration refuses one the
  map sends to an architecture it has no cubin for (and PTX whose `.target` that architecture cannot build), and the build refuses an architecture that falls
  back to it without one.
* **What the implementations share, and what they do not.** What a caller, a plan and a typed host launcher see is shared: the operator's **public
  signature**. Every implementation takes the operator's tensors (their names, their order, `access`, `dtype`, declared shape, whether a view is accepted
  (`Strided`) and the input an `out` may be written over), its fixed arguments per call (`given`, by name and type, in order), the constraints of the
  dimensions those shapes name, and the same stems (binaries) with the same buckets (Python refuses a declaration that does not,
  `decorators._check_implementations`). Each implementation has **its own ABI**, which its entries carry: it may bring its own stub (`tilefish.impl(...,
  stub=f)`, the kernel's parameters in order), so a tensor may be a `tensordesc` (a TMA descriptor, section 5.5) in one and a `pointer` in another, with
  other `scalar`, `fixed` and `constexprs`, other scratch slots (`kind` `triton` or `binary`), other dimensions in `dims` (a descriptor of a view brings its own
  stride dimension, a pointer form the one its stride scalar carries) and other `preconditions` (a descriptor's address is 16-byte aligned, a pointer of a
  view may promise only its element's). How it is compiled and launched is its own too: `grid`, `block`, `num_warps`, `num_stages`, `options`, `shared`, the
  source digest, and the `declaration_digest`.
* **Identity.** The implementation's key is in its declaration (`_canonical_declaration` adds `"implementation"`, for a kernel and for a prebuilt binary),
  so in its `declaration_digest` and in its compile specs (`"implementation"`), and the entry records it. Like every field it is in the manifest digest. The
  declaration digest of one implementation covers that implementation only: a manifest is per architecture, so another implementation for another
  architecture, or another architecture mapped to a named one, leaves this entry what it was, and a changed implementation for this architecture (its kernel,
  its options, its constexpr values, its grid, its stub, its binary, or the map sending this architecture to another implementation, another name or the
  fallback) makes it stale.
* **Build.** A compile spec of one implementation knows which architectures the declaration runs it on when it was generated or verified against the
  declaration, and the build (`_build.check_implementation_arch`, for a kernel and for a prebuilt binary) refuses any other: the implementation declared for
  sm_103 on sm_120, a named one on an architecture that is not mapped to it, the fallback on an architecture that has an implementation of its own.
* **Replay.** Python's `replay` selects the implementation of the device's architecture as the export did, and refuses an entry whose `implementation`
  is not that one (`Manifest.built_kernel`: "was built from the implementation 'tcgen05' ..., but the declaration now runs its implementation 'sm_103'
  on sm_103"), whose declaration digest is not that implementation's, or whose IR digest (or, for a prebuilt binary, source digest) is not the one it has now;
  pinned or not. It launches the entry with that implementation's ABI.
* **A plan** names the entry by its stem, which is the same for every implementation, and binds the operator's tensors by name, in the order they share,
  so the plan of a graph is one document whatever runs: the architecture of the plan selects the binaries (its manifest has the entries of that
  architecture), and a host follows the ABI of the entry it has. Python's planner binds each step with every implementation, their preconditions included,
  so a plan is made only when every implementation can run it (a view the pointer form takes but a TMA descriptor cannot describe is refused on every
  architecture). A single call is checked against the implementation it runs: a tensor that implementation's preconditions refuse is refused on that
  architecture only.
* **Readers.** Both readers refuse an `implementation` that is not the entry's `arch`, `"fallback"` or a name of the form above (another architecture,
  another string, a value that is not a string); which architectures map to a name is the declaration's to say, and the reader cannot know it. The Rust host
  reads it into `Entry::implementation` and needs nothing else from it: it launches the entry of its device's architecture, with that entry's ABI, as for any
  operator.
* **Older readers.** A manifest without a name and without an `implementation` on a `binary` entry is unchanged, digests included. tilefish 0.0.6 and the
  crate 0.0.4 refuse an `implementation` that is a name, or on a `binary` entry.

### 4.2 Warp-specialised kernels: `total_num_warps` (tilefish 0.0.6, crate 0.0.4)

A Triton or Gluon kernel is **warp-specialised** when its partitions run on warps of their own: Gluon's `gl.warp_specialize(default_args,
default_partition, worker_args, worker_partitions, worker_num_warps, worker_num_regs)` (say a 1-warp TMA producer and a 1-warp MMA issuer beside the
compute warps), or Triton's own warp specialisation of a `tl.range(..., warp_specialize=True)` loop (sm_100 and up). Its TTGIR has
`ttg.warp_specialize` regions.

**What `num_warps` means.** The declaration's `num_warps` (`@tilefish.extern(..., num_warps=4)`, the compile option Triton is given) is the warps of
the kernel's **default partition**: those its code outside the warp-specialised regions and its default partition run on, which a Gluon kernel's
layouts are written for. For a kernel that is not warp-specialised that is all its warps. It is not a hint. The worker partitions add warps of their
own, and Triton 3.5.1 reports the total after warp specialisation (`ttg.total-num-warps`, padded to a warp group: a 4-warp default partition and two
1-warp workers are 8, an 8-warp one with the same workers 12) as the compiled kernel's `metadata.num_warps`, the warps its own launcher makes a block of.

**The entry.**

| field | type and rule |
|---|---|
| `num_warps` | as for any Triton entry: the compile option, equal to `options.num_warps`, so the warps of the default partition |
| `total_num_warps` | optional int, only on a `triton` entry: the warps of a block of the compiled kernel, all its partitions'. Present exactly when the kernel is warp-specialised, and then more than `num_warps` and at most 32. Absent (or `null`) for any other kernel |
| `block` | `[32 * total_num_warps, 1, 1]` for a warp-specialised kernel, `[32 * num_warps, 1, 1]` for any other Triton kernel |

**The compiled launch configuration.** Everything a launch takes besides its arguments and its grid is a compiled fact the entry records:
* `block`: `32 *` the compiled warps;
* `shared`: Triton's `metadata.shared`, the dynamic shared memory;
* `launch`: `cluster_dims`, `num_ctas`, the scratch sizes, PDL and a cooperative launch.

A thread-block cluster (section 4.5) and a global scratch (section 4.6) are launched since tilefish 0.0.7 and crate 0.0.5; what neither host sets up is
still refused when the binary is loaded: profile scratch, programmatic dependent launch and a cooperative launch. Every launch uses these values, in
Python and in Rust: a single launch, a step of a plan, and a launch captured in a CUDA graph.

**The build** (`_build.build`) checks what Triton did:
* a kernel whose TTGIR has no `ttg.warp_specialize` region must have exactly the declared `num_warps`. A silent change is refused: "the compile
  option num_warps=4 was not applied (Triton used 8)";
* a warp-specialised kernel must have more warps than `num_warps`, and at most 32;
* either way, the `.reqntid` of its PTX entry must be `32 *` the warps the metadata reports. Triton writes it for every kernel, and the driver refuses
  a launch with another block.

The build records the total (`BuiltKernel.num_warps`, `BuiltKernel.warp_specialized`), and so does the build cache, which checks it against the
options and the PTX on every read.

**Identity.** The total is a function of what identifies the binary (its IR, its compile options, its architecture and its toolchain), so it adds
nothing to a compile spec or a declaration digest. A compile spec's `block` stays `32 * num_warps`: a spec is made before anything compiles. The entry
records the total, so the manifest digest covers it, like every field: a binary of other warps is another manifest. A replay compares the entry with
the spec as for any kernel, with the block `32 * total_num_warps` in place of the spec's (`_build.expected_entry`), launches with that block, and
refuses a binary whose warps are not the spec's, or more than them for a warp-specialised one (`_build.check_launch_config`). A merge refuses to
replace an entry of other warps (`total_num_warps` is one of the fields `_conflicts` compares).

**One operator on several architectures.** Triton warp-specialises a loop on sm_100 and up only, so one declaration may have a warp-specialised
binary on one architecture and a plain one on another. `block` and `total_num_warps` are per entry, like `shared`; the Rust emitter lists the block of
each architecture.

**Readers.** Both readers refuse:
* a `total_num_warps` that is not an integer, that is not more than `num_warps`, or that is more than 32;
* a `block` that is not `[32 * total_num_warps, 1, 1]`;
* the field on a `binary` entry.

A JSON `null` is as absent. A reader from before this change refuses an entry with the field: the crate's reader refuses unknown fields, and the
replay of Python 0.0.5 finds a block that is not the spec's. So no host launches a warp-specialised kernel with the declared block. Manifests without
the field are unchanged, and so are their digests.

### 4.3 Persistent grids: `persistent` (tilefish 0.0.7, crate 0.0.5)

A **persistent** kernel launches about as many programs as the device runs at once, and each program walks the work items with a stride of the programs
launched: `for t in range(tl.program_id(0), tiles, tl.num_programs(0))`. Its best grid is `min(tiles, programs per SM * SMs)`, which no dimension
expression can say, since the SM count is the device's. Declared with `@tilefish.extern(..., grid=lambda M, N: (tilefish.cdiv(M, 128) *
tilefish.cdiv(N, 128),), persistent=tilefish.persistent(programs_per_sm=1))` (a Triton or Gluon kernel, or a prebuilt binary), its grid is **the number
of tiles**, and the host cuts it when the kernel is loaded:

| field | type and rule |
|---|---|
| `persistent` | optional object, on a `triton` or `binary` entry: exactly `programs_per_sm` and, optionally, `sm_count`. Absent (or `null`) for every other kernel |
| `persistent.programs_per_sm` | int from 1 to 32: the programs a launch runs per SM, at most |
| `persistent.sm_count` | optional int from 1 to 2^20 (`null` is as absent): the SM count a kernel **whose bytes depend on how many programs run** is declared for |
| `grid` | of a persistent entry: exactly one entry, the tiles |

**What a launch runs.** `programs = max(1, min(tiles, programs_per_sm * SMs // ctas))`, where `tiles` is the grid's one entry at the launch, `ctas` the
CTAs of one program (`launch.num_ctas`, section 4.5: a program of a cluster of 2 CTAs takes 2 SMs) and `SMs` the multiprocessor count the host resolves
**when it loads the binary**: the device's (`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`; a MIG slice has its own), or the partition's when the host runs
on a partition of the device's SMs (a green context reports the device's count to every query, so the host says it: `Manifest::with_sm_partition(Some(n))`
in Rust, `tilefish.runtime.sm_partition(n)` in Python, a context manager around the loads and runs). A launch has `programs` along x (times its cluster,
section 4.5). The grid of blocks is checked against the device's limits as any grid; one binary serves every device shape, so a manifest exported for
one SM count (a MIG slice) runs on another (the whole device) without being exported again. A launch plan and a CUDA graph hold the programs of the
device and partition they were prepared or captured on.

**The bytes of a persistent kernel.** The SM count changes how many programs run, and which program computes which tile: never the binary. So:

* `sm_count` absent (the default) is the kernel's promise that **its bytes do not depend on the number of programs**: each tile is computed by one
  program from its inputs, whichever program and however many there are (an elementwise tile, a GEMM tile, a row). This is the usual case, and what
  the declaration states; tilefish cannot check it (as it cannot check a `MayAlias`), and the gpu tests hold it to the same bytes on 1, 8, 16 and 30 SMs.
* A kernel whose bytes **do** depend on it (a persistent reduction whose programs each sum their own tiles and then combine the partial sums: another
  number of programs adds in another order) declares the SM count it is for, `tilefish.persistent(sm_count=30)`. The entry records it, and a host on
  a device or a partition of another count refuses the kernel when it loads it ("declared for 30 SMs, but this device has 148"): its bytes on 30 SMs
  are the ones it was validated with. Running it on another device shape is another declaration (another value), so another export, as a cuBLASLt
  pin of another SM count is (section 9).

The field is digested like every field, and the declaration digest has it (only when it is declared); a reader that predates it refuses an entry with it
(the crate's reader refuses unknown fields; Python 0.0.6's replay finds an entry that is not the spec's). Manifests without it are unchanged.

### 4.4 A configuration per bucket (tilefish 0.0.7): `@tilefish.extern(..., by_bucket=...)`

An extern of a bucketed dimension may choose its compile-time configuration per bucket, which used to take one operator per configuration and a choice
at plan time:

```python
@tilefish.extern(triton=k.rows, num_warps=4, grid=lambda TB, BLOCK_T: (tilefish.cdiv(TB, BLOCK_T),),
                 by_bucket=lambda TB: tilefish.config(num_warps=2, BLOCK_T=16) if TB <= 256 else tilefish.config(num_warps=8, BLOCK_T=64))
def rows(x: Tensor[TB, 128, f32], y: Out[Tensor[TB, 128, f32]], n_rows: TB, BLOCK_T: const = 32): ...
```

`by_bucket` is a callable over names of bucketed dimensions of the signature (`Dim("TB", buckets=[...])`), called once per bucket combination of the
dimensions it names, returning `tilefish.config(num_warps=, num_stages=, num_ctas=, **consts)` or `None`; what a configuration leaves out is the
declaration's. Those dimensions are bucketed dimensions of the operator (one binary per bucket, as a constexpr that depends on a bucket makes one: after
the ones of such constexprs in the stem, in the order of the callable's parameters). Each binary is compiled with its bucket's warps, stages and CTAs
and its consts' values (the tile sizes, any constexpr), and launched with the grid the grid callable gives under those values.

**Nothing is new in the manifest**: every entry already has its own `num_warps`, `num_stages`, `options`, `constexprs`, `grid` and `block`, and those of
each bucket are its configuration's. **The identity**: the declaration digest covers the whole configuration of every bucket (the `by_bucket` key of
the canonical declaration lists, per bucket combination, the warps, stages, CTAs, the typed value of every const it sets and the grid under them; only
when `by_bucket=` is given), so it is one digest for every bucket of the operator, as before, and changing the configuration of one bucket makes every
binary of the operator stale; each binary's compile spec has its configuration (its compile options, its constexpr values, its grid), so the build
cache keys and the manifest entries are per configuration. A call cannot give a const that a configuration sets (its value is the bucket's). A prebuilt
binary takes no `by_bucket=` (its configuration is compiled already), and with `by_arch=` each implementation says its own (`tilefish.impl(...,
by_bucket=...)`).

### 4.5 Thread-block clusters: `launch.cluster_dims`, `launch.num_ctas` (tilefish 0.0.7, crate 0.0.5)

A Triton or Gluon kernel compiled with `num_ctas=2` (`@tilefish.extern(..., num_ctas=2)`, 1, 2, 4 or 8; the compile option of Triton 3.5.1, sm_90 and
up) runs each program on a **thread-block cluster** of `num_ctas` CTAs along x, which share distributed shared memory and TMA multicast; Triton's
metadata says `num_ctas` and `cluster_dims` (`(num_ctas, 1, 1)`). A prebuilt binary declares its cluster, `@tilefish.extern(ptx=..., cluster=(2, 1,
1))` (at most 8 blocks), for a kernel written for one (`%cluster_ctarank`, `mapa`, `barrier.cluster`; `.explicitcluster` makes the driver refuse a launch
without it).

**The entry.** `launch.cluster_dims` and `launch.num_ctas` (the fields an entry always had) are the cluster, `num_ctas` their product; a Triton entry
compiled with more than one CTA per program has `options.num_ctas` (its compile option, present exactly when it is not 1, and equal to
`launch.num_ctas`). A prebuilt binary's `launch` is the zero value but for its cluster.

**The launch.** The grid of the entry counts **programs** (clusters): a launch is `cuLaunchKernelEx` with the grid of blocks `grid * cluster_dims` (each
axis; Triton's own launcher multiplies the grid by `cluster_dims` when `num_ctas` is not 1), the attribute `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` of
`cluster_dims`, and `CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE` = `SPREAD` (what Triton's launcher asks for), on every launch: a single launch,
a step of a plan, and a launch captured in a CUDA graph (the graph's kernel node has the cluster). A kernel of `num_ctas=2` launched without its cluster
computes other bytes (the gpu test shows it), which is why a reader of an earlier version refused it. Both hosts refuse at load a cluster that is not
`num_ctas` CTAs or is larger than the portable 8 (no host asks for CUDA's non-portable cluster size). The build refuses a binary whose metadata cluster
is not the declared `num_ctas`.

### 4.6 A global scratch: `launch.global_scratch_size`, `launch.global_scratch_align` (tilefish 0.0.7, crate 0.0.5)

Triton 3.5.1 gives a kernel a **global scratch** when it needs memory of its own in global memory: device-side TMA descriptors (`tl.make_tensor_descriptor`)
write their `CUtensorMap`s there, 128 bytes each. The kernel takes the scratch pointer in the first scratch slot of its ABI (`global_scratch`, section 5),
and addresses `global_scratch_size` bytes per CTA from it (`metadata.global_scratch_size`, aligned to `metadata.global_scratch_align`): Triton's own launcher
allocates `global_scratch_size * gridX * gridY * gridZ * num_ctas` bytes per launch with the allocator `triton.set_allocator` installs, and passes the address.
The scratch is the launch's: nothing in it survives the launch, and the next launch may write over it.

**The declaration** states it, `@tilefish.extern(..., global_scratch=256)` (bytes per CTA, a Triton or Gluon kernel), because a launch plan sizes its
arena without building anything (plan schema 1, section 6); the build refuses a kernel whose metadata says another size ("the compiled kernel addresses
256 bytes of global scratch per CTA, but declare global_scratch=256"), and one whose alignment does not divide 256 bytes. It is in the declaration digest
(only when declared) and in the compile spec (`global_scratch`).

**The entry.** `launch.global_scratch_size` is the bytes per CTA, and `launch.global_scratch_align` (new, an int, a power of two) its alignment, which
tilefish writes with every global scratch and a reader refuses without one: the `launch` of every other entry is the six fields it always was (an entry
of tilefish 0.0.6 with a global scratch has no alignment, and is read as before and refused when its binary is loaded). A prebuilt binary has no global
scratch (its C parameters are all it is launched with).

**The launch** gives the kernel `global_scratch_size * (CTAs of its grid)` bytes, the programs of a persistent grid at that launch times `num_ctas`, starting
on 256 bytes (every alignment Triton records divides it), at the `global_scratch` slot (the profile scratch slot stays null). Python's single launch
allocates it on the launch's stream (torch's caching allocator, freed after the launch, stream-ordered; under a `torch.cuda.graph` capture from the graph's
pool); the Rust host takes it from the caller, **the last tensor of a launch** (a writable `u8` tensor of at least `Prepared::scratch_bytes` bytes,
256-byte aligned, overlapping no other tensor: `Plan::tensor_names()` ends with `global_scratch`), so that nothing is allocated in a launch and a launch
can be captured; a launch plan gives every kernel step with a global scratch the plan's workspace (plan schema 1, sections 6 and 7). Both hosts refuse at
load a manifest of tilefish 0.0.6 that has a global scratch (it did not record the alignment: export it again). Profile scratch stays refused.

### 4.7 The CUDA math functions of a generated kernel: `cuda_math` (tilefish 0.0.7, crate 0.0.5)

Some ops of a kernel tilefish generates are, by the numerics contract, the bits of a function of CUDA's math library: in IEEE math `exp` and `silu`
are built on `expf`, `rsqrt` is `rsqrtf` and `erf` is `erff` (inline PTX read from NVRTC), and in either math `sin`, `cos` and `tanh` are libdevice's
`sinf`, `cosf` and `tanhf` (notes 10 and 13 to 15 of `tilefish.lower`). Those functions are not the same in every CUDA release (CUDA 13.2 changed
`erff`), and libdevice is a file Triton links (Triton's own, or the one `TRITON_LIBDEVICE_PATH` names). The entry of such a binary says so:

| field | type and rule |
|---|---|
| `cuda_math` | optional object, only on a `triton` entry that has a `source_sha256` (a generated kernel). Absent (or `null`) for every other entry and for a generated kernel that uses none of these functions and links no libdevice |
| `cuda_math.functions` | list of the CUDA math functions whose bits the binary's ops are (`"erff"`, `"expf"`, ...: names of lowercase letters, digits and `_`), sorted, each once; empty for a kernel that only links libdevice (`gelu`'s `erf` is libdevice's own and claims no release) |
| `cuda_math.cuda` | list of CUDA releases (`"13.0"`: two decimal numbers, no leading zero), sorted by release (`"12.9"` before `"12.10"`), each once: those on which every listed function, compiled by that release's NVRTC with nvcc's defaults, was measured to be the op's bits on every f32, when the binary was exported (`tilefish._cudamath.VERIFIED`). Empty when `functions` is; may be empty when no release is common to them |
| `cuda_math.libdevice` | optional string, 64 lowercase hex digits: the sha256 of the libdevice file the binary was linked with, present when the kernel calls a libdevice function |

* **What it is.** A statement of where the binary's bits come from, not a part of what is built: the build reads the functions from the code of the
  generated source (the helpers and calls `tilefish.lower` writes for them; the source is not changed for it, so no source digest moves), the
  releases from tilefish's table, and the libdevice from the file Triton links for the compile options. Like every field it is in the manifest
  digest. It is not compared with the compile spec, and a merge does not compare it (another libdevice gives other machine code, which
  `sha256_code` tells).
* **Python** checks it on every run: where `torch.version.cuda` (the release torch's own ops are built with) is not in `cuda` while `functions` is not
  empty, and, in a replay, where a build in the process would link another libdevice than `libdevice`, it warns with `TilefishCudaMathWarning`
  (an error under `warnings.simplefilter("error", TilefishCudaMathWarning)`). The binary still runs: what the warning says is that torch's ops,
  and the reference mode built on them, may not be its bits there. The reference mode warns the same way from the kernel's IR.
* **The Rust host** reads it into `Entry::cuda_math` (`CudaMath { functions, cuda, libdevice }`) and checks nothing on its own (it cannot tell the
  release of the host's other CUDA code); `Manifest::check_cuda_math("13.0")` refuses a manifest with an entry whose `functions` is not empty and
  whose `cuda` does not list that release, naming each.
* **Readers.** Both readers refuse: a `cuda_math` that is not an object or has another field; `functions` or `cuda` missing, not lists of the forms
  above, not sorted or with a repeat; `cuda` not empty for no function; a `libdevice` that is not a sha256; no function and no libdevice; the field on
  a `binary` entry or on an entry without `source_sha256`.
* Where it is measured and kept: `docs/en/numerics.md` (the procedure to run on a CUDA or torch change), `python/tests/test_gpu_cuda_math.py`.

### 4.8 The probes of an instrumented build: `instrument` (tilefish 0.0.11, crate 0.0.9)

An instrumented build (`tilefish.runtime.export(..., instrument=True)` or `tilefish.instrument(capacity=..., ops=...)`, `docs/en/instrument.md`) records the
kernel's `tilefish.probe` calls, and `@begin`, `@calib` and `@end`, into a probe buffer that a host gives in the profile scratch slot (the second slot of
`abi.scratch`, which a production launch passes as null). Its entry says so:

```json
"instrument": {"format": "tilefish.probes.v1", "capacity": 256,
               "probes": [{"id": 3, "name": "load.wait", "sites": [{"loc": "kernels/ws.py:44", "role": "worker 0", "function": "kernels.ws._load"}]}]}
```

| field | type and rule |
|---|---|
| `instrument` | optional object of exactly these three fields, only on a `triton` entry. Absent from every entry of a production export |
| `instrument.format` | `"tilefish.probes.v1"`: the record format, the layout of the buffer and the probes the build adds (`docs/en/instrument.md`, section 3) |
| `instrument.capacity` | integer from 1 to 65536: the records each warp keeps per launch; a launch's buffer is `ctas * warps * (16 + 16 * capacity)` bytes, `warps` the block's warps, `ctas` its CUDA grid |
| `instrument.probes` | list of the kernel's probes, each `{"id", "name", "sites"}`: `id` an integer from 3 to 65535 (0, 1 and 2 are the build's `@begin`, `@end`, `@calib`), `name` the probe's (1 to 64 characters, a letter or `_` first, then letters, digits and `_ . : / -`), each id and each name once; `sites` a list of `{"loc", "role", "function"}`, all strings: the module-relative line it is written on, the partition of a warp-specialised kernel it runs in (`"default"`, `"worker 0"`, ...; `""` when the kernel is not partitioned) and the jit function. A kernel without probes has `[]` |

* **What it is.** The entry of a binary that is not the production one: its `sha256`, `sha256_code` (and, for a kernel with probes, its
  `source_digest`: the IR with its probes) are its own, and the field is in the manifest digest like every field, so an instrumented export has another
  digest, and so have its plans. The declaration digest, the ABI, the grid and the launch attributes are the production entry's: nothing a host passes
  changes but the profile scratch, the address of the buffer (16-byte aligned) instead of null.
* **An instrumented export.** A manifest with an entry that has `instrument` is an instrumented export: Python's `Manifest.load` refuses it unless
  `instrumented=True`, a replay refuses it unless it collects probes (`probes=`), and the Rust host's `Manifest::load` and `from_embedded` refuse it
  (`Manifest::load_instrumented` takes it). An export never writes the two variants into one manifest. A reader that predates the field: the crate
  0.0.7 refuses it (an unknown field); a launch of such a binary with a null profile scratch records nothing and writes the production bytes.
* **Readers.** Both refuse: an `instrument` that is not an object of the three fields; another `format`; a capacity out of range (or a boolean); a
  probe that is not an object of the three fields, an id out of range or repeated, a name that is not one or repeated; a site that is not an object of
  three strings; the field on a `binary` entry.
* **The Rust host** reads it into `Entry::instrument` (`Instrument { capacity, probes }`, each `Probe { id, name, sites }`); `Kernel::launch_probed`
  passes a `ProbeBuffer` in the profile scratch slot and refuses it for a production binary (`docs/en/instrument.md`, section 4).
* Where it is tested: `python/tests/test_instrument.py`, `python/tests/test_gpu_instrument.py`, `crates/tilefish/tests/instrument.rs`, the Rust gate's
  instrumented case.

### 4.9 CuTe DSL kernels: `binary.source: "cute"`, the kinds `tensormap` and `bytes` (tilefish 0.0.13, crate 0.0.11)

A kernel written in CUTLASS's CuTe DSL (`nvidia-cutlass-dsl`, `import cutlass.cute as cute`: a `@cute.jit` host function that launches a `@cute.kernel`)
is declared with `@tilefish.extern(cute=host, cute_args=..., cute_options=...)` (`docs/en/cute.md`). The DSL compiles it for the layouts of its arguments,
so its shapes are static: one binary per bucket. Its cubin is the DSL's own (the PTX of the NVVM and the cubin of the `ptxas` the DSL carries), and the
launch is the compiled host function's, captured when it is built (a CUDA graph: the kernel does not run): the grid, the block, the cluster, the
dynamic shared memory and the bytes of every kernel parameter. Its entry is a `binary` entry of section 8 with these differences:

| field | type and rule |
|---|---|
| `binary` | exactly `{"source": "cute", "cutlass_dsl": <version>, "options": <string>}`: the version of the DSL package (`[0-9]+(\.[0-9]+){1,3}` and an optional suffix of `[A-Za-z0-9.+_-]`) and the DSL compile options the declaration gives (characters of `[A-Za-z0-9 =,._:/+-]`, one space apart, possibly empty) |
| `toolchain` | exactly `{"cutlass_dsl", "nvvm", "ptxas"}`, strings: the DSL (`binary.cutlass_dsl`), the NVVM release its PTX header names, the `ptxas` release its cubin's `.note.nv.tkinfo` names |
| `source_digest`, `source_digest_kind` | the sha256 of the PTX the DSL generated, of kind `"tilefish.cute.v1"` (required) |
| `ptx_target` | the `.target` of the PTX, `sm_<arch>` with an optional letter (`sm_103a`), of the entry's architecture |
| `grid` | integers only (no dimension name: the shapes are static), the programs of the launch: its grid of blocks divided by its cluster, axis by axis (section 4.5) |
| `block`, `shared` | the compiled kernel's block and dynamic shared memory |
| `launch` | the zero value but for the cluster of the compiled launch (at most 8 CTAs, launched with the spread scheduling policy as every cluster is) |
| `persistent`, `options`, `num_warps`, `num_stages`, `source_sha256` | absent |
| `abi.params` | kinds `pointer` (dense: no `strides`, `offset` or `alias`), `tensormap` and `bytes` only, one slot each; every tensor of the operator in exactly one `pointer` or `tensormap`, in the order of the declaration; `scratch` empty, `scratch_slots` 0 |

**The kinds.**

* `tensormap`: `{"slot", "index", "name", "kind": "tensormap", "type": "tensormap", "ptx_type": "b8[128]", "size": 128, "dtype", "access", "shape",
  "template"}`. A tensor (it binds dimensions, has preconditions and is passed by a call as a `pointer` is) passed as a TMA descriptor, 128 bytes by value,
  as the DSL's host code encodes one for a static shape: `template` is 256 lowercase hex digits, the 128 bytes with the bytes 0 to 7 and 64 to 127 zero. A
  launch passes the template with the tensor's address, a little-endian `u64`, in its bytes 0 to 7 (64-byte aligned in the host's memory, as the PTX
  parameter is); the address is 16-byte aligned (its `pointer_alignment`) and below 2^57 (the DSL keeps 53 bits of `address / 16`), or the launch is
  refused. It has no `dim`, `count_of`, `stride_of`, `value`, `alias`, `strides`, `offset` or `tensor_map`.
* `bytes`: `{"slot", "index", "name", "kind": "bytes", "type": "bytes", "ptx_type": "b8[<size>]", "size": <1 to 32764>, "value"}`, `value` the `2 * size`
  lowercase hex digits of the parameter. A launch passes the bytes; a call gives nothing for it. It has no other field. Its `index` is its `slot` and its
  `name` `param_<slot>` (it is no parameter of the declaration).

**Where the template comes from.** The DSL encodes the `CUtensorMap` of a static shape in its own host code, not with `cuTensorMapEncodeTiled`: it writes
eight 64-bit words, the first of which holds the address (`((address / 16) & (2^53 - 1)) << 4`) and the seven others constants of the shape, the box,
the swizzle and the type, and leaves the 64 bytes after them as they were on its stack (the driver ignores them; a host passes zeros). The build
captures the launch three times, each time with every tensor at another address, and requires every byte of a parameter that the kernel reads to be
the same in the three, or to be a tensor's address (an aligned 8-byte word, read whole, equal to that tensor's address in each capture); the bytes the
kernel reads are those its PTX loads with `ld.param` at constant offsets, or all of a parameter whose address the code takes otherwise (a descriptor),
and a descriptor is a parameter of 128 bytes aligned to 64 whose first word is an address, its bytes 64 to 127 excepted. A byte the kernel does not read
(the padding of a structure, which the DSL's host code leaves as it was on its stack) is zero in `value` and in a template. A parameter that holds an address elsewhere (a pointer
inside a structure), a byte that varies and is no address (a value of the host function's own: a pointer it allocated, a dynamic shape), a tensor in
two parameters or in none, and the tensors in another order than the declaration's are refused: the kernel would take something a host cannot
reproduce from the manifest.

**Identity.** The declaration digest has, in place of a prebuilt binary's section, `{"kind": "cute", "host": "<module>:<qualname>", "source_sha256":
<the sha256 of the file of the host function>, "cutlass_dsl": <the DSL's version>, "options": <the options>}` (the compile spec's `binary` section is the
same), so another DSL, another file or other options is another declaration. The binary is identified, as a Triton kernel by its IR, by its PTX:
Python's replay compiles the declaration with the DSL of the process (it must be the declaration's version) and compares the sha256 of the PTX with
`source_digest`; a mismatch is refused, pinned or not, and where it cannot be compiled the entry is accepted only pinned. The Rust host and the C ABI
compare nothing but the pin (the manifest digest covers every field and every file, as always). The compile spec's `grid` (`"(1,)"`) and `block`
(`[1, 1, 1]`) are placeholders: the entry's are the compiled kernel's, and a replay compares the entry's tensors (index, name, dtype, access and shape
of each) with the spec's.

**Readers.** Both readers refuse what this section does not allow: a `binary` section, a toolchain, a source digest, a `ptx_target` or a grid of another
form; a `persistent` grid; a parameter of another kind, a view or an alias, on such an entry; a `tensormap` or a `bytes` parameter on any other entry; a
template or a value of another length, not lowercase hex, or a template not zero in its bytes 0 to 7 and 64 to 127; a `bytes` size out of range or a
`ptx_type` that is not `b8[<size>]`; a field another kind has. A reader that predates the section (tilefish 0.0.12, crate 0.0.10, and older) refuses an entry of
source `cute` (`binary must be {"source": "cubin"} or ...`). A manifest without such an entry, and its digest, are unchanged.

## 5. The ABI

`abi` is `{"params": [...], "scratch": [...], "scratch_slots": <int>}`. `params` are the kernel's launch arguments in order, one `kernelParams` slot
each (`slot` is the position in `params`), except a `tensordesc`, which takes `1 + 2 * rank` slots (section 5.5): `slot` is where the slots of a
parameter begin, the slots of the parameters before it counted. `scratch` is what a Triton kernel is launched with after `params`: two 64-bit pointers
(`global_scratch`, `profile_scratch`, `value` 0, at the slots after those of the last parameter), unchanged: null, but the `global_scratch` of an entry
whose `launch.global_scratch_size` is above 0, which is the address of the scratch the launch gives it (section 4.6). **A prebuilt binary has `scratch: []` and
`scratch_slots: 0`: it is launched with exactly `params`, no more.**

Every `params` entry has `slot`, `index` (the position in the declaration), `name`, `kind`, `type`, `ptx_type`, `size`. The kinds and what else each has:

| `kind` | `type` | `ptx_type`, `size` | other fields |
|---|---|---|---|
| `pointer` | `*bf16`, `*fp32`, `*i32`, ... (Triton's pointer type of the dtype) | `u64`, 8 | `dtype`, `access` (`in`/`out`/`inout`), `shape`; optional `strides` and `offset` (together); on an `out` pointer, optionally `alias` (section 5.4) |
| `tensordesc` (Triton and Gluon kernels only) | `tensordesc` | `b8[128]`, 128 (its first slot: the `CUtensorMap`) | the fields of a pointer (`dtype`, `access`, `shape`, and on an `in` one optional `strides` and `offset`), and `tensor_map`; no `alias` (section 5.5) |
| `scalar` | `i32` or `i64` | `u32`/4 or `u64`/8 | `dim` (an expression); optionally one of `count_of` or `stride_of` |
| `fixed` (prebuilt binaries only) | `i32`, `i64` or `f32` | `u32`/4, `u64`/8, `f32`/4 | `value` |
| `given` (prebuilt binaries only) | `i32` or `i64` | `u32`/4 or `u64`/8 | none: every call gives the value (section 5.3) |
| `null` | `*bf16`, `*fp32`, ... (the pointer type of its `dtype`) | `u64`, 8 | `dtype`: the launch passes 0 (section 5.6) |
| `tensormap` (CuTe DSL kernels only) | `tensormap` | `b8[128]`, 128 | the fields of a dense pointer (`dtype`, `access`, `shape`) and `template`: the launch writes the tensor's address into it (section 4.9) |
| `bytes` (CuTe DSL kernels only) | `bytes` | `b8[<size>]`, 1 to 32764 | `value`: the launch passes these bytes (section 4.9) |

A launch passes: a `pointer` as its 64-bit address (see 5.1), a `null` as the 64-bit zero, a `scalar` as a signed integer of its `type`, a `fixed` as its `value` in its `type`
(`ctypes.c_int32`/`c_int64`/`c_float`: an `f32` value is the JSON number rounded to the nearest `f32`). It is `cuLaunchKernel` with the grid the
entry's `grid` evaluates to, the entry's `block` (`32 *` the warps of the compiled kernel, all its partitions' for a warp-specialised one: section 4.2)
and its `shared` bytes of dynamic shared memory.

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
   `i32`, `0 ≤ v ≤ 2^63 − 1` for `i64`; else refused. (A stride's `type` is `i64` when the op declares `index_bits=64` or its `StrideOf` says
   `dtype=i64`; a dimension or an element count is `i32` unless the declaration says `i64[n]`, section 5.7.)
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

**A fixed argument given per call** (`"kind": "given"`) is a scalar of a prebuilt binary whose value is not in the manifest: each call gives it
(Python: `h_out: const[i32]` or `const[i64]` **with no default**, `k(x, h_out=5)`; the geometry of a call, such as an output size or an offset, that
no tensor of the call determines). It has `type` `i32` or `i64` (`u32`/4 or `u64`/8) and no `value`, `dim` or anything else, and only a `binary`
entry has one. **One entry serves every value**: a launch passes the value the call gives at the parameter's slot, as a signed integer of its type (any
sign; a value that does not fit is refused, and so is a launch that gives no value for one of them, or one for a name the entry does not have:
Python `tilefish.run(op, x, h_out=5)`, Rust `Kernel::launch_with(stream, tensors, &[("h_out", 5)])`); a step of a launch plan records the values
of its call (`"given"`, `docs/en/plan-schema-1.md` section 7). A float stays a `fixed` argument (fixed per declaration). The declaration digest has the
parameter as given (`"given": true`, no default), so a declaration with a default is another operator. A reader that predates the kind (tilefish
0.0.4, the published crate 0.0.2) refuses a manifest that has one.

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

### 5.5 TMA descriptors: `tensordesc` (tilefish 0.0.6, crate 0.0.4)

A Hopper or Blackwell kernel written in Triton or Gluon moves tiles with the Tensor Memory Accelerator when a parameter is a *host tensor descriptor*:
a `CUtensorMap` that the host encodes with `cuTensorMapEncodeTiled` and passes **by value** (in PTX `.param .align 64 .b8 name[128]`, a
`__grid_constant__`). Python declares one with `x: TensorDesc(Tensor[T, C, f16], block=(128, 64))` (`Out[TensorDesc(...)]`, `InOut[TensorDesc(...)]`;
`TensorDesc(Strided[Tensor[...]], ...)` for a view the kernel reads), and a call passes the **tensor**: its descriptor is made by the host.

**The slots.** Triton 3.5.1 expands a descriptor parameter (its launcher, `backends/nvidia/driver.py`, `make_tensordesc_arg`) into, in this order:

1. the 128 bytes of the `CUtensorMap`, by value;
2. the global shape of the tensor, one `i32` per axis, outermost first (`u32`, 4 bytes each);
3. the element strides of the tensor, one `i64` per axis, outermost first (`u64`, 8 bytes each);

so a `tensordesc` of a tensor of rank `r` is **`1 + 2r` slots** of `kernelParams` (the loaded function reports `128`, then `4` `r` times, then `8` `r` times,
which both hosts check), all from the one parameter entry, whose `slot` is the first. The shape and strides are those of the tensor of the launch (the
canonical strides of section 5.1, step 2). The next parameter's `slot` follows the last of them, and so do the scratch slots.

**The entry.** `{"slot", "index", "name", "kind": "tensordesc", "type": "tensordesc", "ptx_type": "b8[128]", "size": 128, "dtype", "access", "shape",
["strides", "offset"], "tensor_map"}`: the fields of a pointer (section 5), its dtype one of `bf16 f16 f32 i32 i64 u8 e4m3 e5m2`, a shape of 1 to 5 axes, no
`alias` (neither on it nor naming it), and on a descriptor of a view (`in` only) `strides` whose last entry is the fixed `1` and every other one the name of
a stride dimension (`x_s0`; bound from the view, passed in the descriptor's own stride slots, so no `stride_of` scalar carries it, and a `stride_of` may
not name a descriptor) and the `offset` dimension (`divisible_by` the elements in 16 bytes). `tensor_map` is every argument of `cuTensorMapEncodeTiled`
that is not the tensor's, exactly these fields:

| field | type and rule |
|---|---|
| `data_type` | the `CUtensorMapDataType` of the dtype, by the name of its enumerator: `CU_TENSOR_MAP_DATA_TYPE_UINT8` (`u8`, `e4m3`, `e5m2`), `_INT32`, `_INT64`, `_FLOAT16`, `_FLOAT32`, `_BFLOAT16` |
| `block` | the block of the descriptor (the tile one `load` or `store` of the kernel moves, in its Triton type `tensordesc<fp16[128, 64]>`), outermost first: `rank` powers of two from 1 to 65536 |
| `box` | the box of the map, outermost first: `rank` ints from 1 to 256, each dividing the block's; the inner one is a multiple of 16 bytes and, with a swizzle, at most the swizzle span. A declaration makes it the block with its inner dimension cut to the swizzle span (a [128, 128] f16 block with a 128-byte swizzle is moved as two [128, 64] boxes: what Triton records) |
| `element_strides` | `[1, ..., 1]` (`rank` of them) |
| `interleave` | `CU_TENSOR_MAP_INTERLEAVE_NONE` |
| `swizzle` | `CU_TENSOR_MAP_SWIZZLE_NONE`, `_32B`, `_64B` or `_128B` |
| `l2_promotion` | `CU_TENSOR_MAP_L2_PROMOTION_NONE`, `_L2_64B`, `_L2_128B` (Triton's) or `_L2_256B` |
| `oob_fill` | `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` (an element outside the tensor reads as zero, a store there is dropped) or `_NAN_REQUEST_ZERO_FMA` (a float reads as NaN; a float dtype only) |

**Where the map comes from, and what checks it.** The declaration states the block, the swizzle (by default the largest of 128, 64 and 32 bytes that
divides the bytes of the block's rows; none for a block of rank 1 or of fewer than 8 rows), the L2 promotion and the fill; they are in its declaration
digest and in the compile spec (whose `signature` entry is Triton's `tensordesc<fp16[128, 64]>`, followed for a Gluon kernel by the `NVMMASharedLayout` of
the tile, as Gluon's `TensorDescriptor` spells it). The compiled kernel records what it expects of each descriptor (`metadata.tensordesc_meta`: the swizzle,
the element type and size, the box), and the build refuses a kernel whose record is not the declaration's (a Triton kernel's compiler picks the swizzle:
the refusal says which one to declare), and one Triton did not lower to TMA (before sm_90). A reader holds `tensor_map` to the rules of the table (what
the driver and Triton refuse), not to how a declaration derives the box.

**What a host does at a launch.** It binds the descriptor's tensor as a pointer's (dimensions from the shape, a view's stride dimensions from its canonical
strides, dtype, rank, contiguity of a dense one, aliasing, the `pointer_alignment` of 16 bytes: the global address of a map), then, after every
precondition of section 6 and before the grid, for each `tensordesc` in the order of the ABI, what `cuTensorMapEncodeTiled` requires of the tensor, each a
refusal of kind **`tensor_map`**: every dimension at most `2**31 - 1` (the driver takes `2**32`; the kernel gets it as `i32`), the innermost canonical stride
1, every other one a multiple of 16 bytes and below `2**40` bytes (a stride of 0, an expanded axis, is one). Then it encodes the map:
`cuTensorMapEncodeTiled(map, data_type, rank, address, globalDim, globalStrides, box, elementStrides, NONE, swizzle, l2_promotion, oob_fill)` with the
dimensions and the box innermost first, the strides of the outer axes in bytes innermost first followed by the bytes of the whole tensor (what Triton's
`fillTMADescriptor` passes: `rank` entries, of which the driver reads `rank - 1`), and element strides of 1. Python calls it through cuda-python, the crate
through `cudarc::driver::sys`; both compute the same numbers (`_tensormap.encode_arguments`, `TensorMapEncode`).

**Per launch, and in a graph.** A map is a function of the tensor's address, shape and strides and of the declaration: it is encoded **at every launch**
from the tensor of that launch (Python keeps the last 4096 encodings by their arguments, so a repeated launch does not call the driver; any other address,
shape or stride is another encoding), and for a plan at every execution, for every step, from the buffer and the view the step binds (an arena buffer at
its offset, or the caller's tensor), with every precondition and before the first launch. A launch on a capturing stream is captured with the map **by
value** in the kernel node: a CUDA graph of a plan holds the maps of the bucket and the arena layout and the externals it was captured with, as it holds
their addresses, so it is right for every replay exactly as long as those addresses are (plan schema 1, section 10). Binding another tensor is another
launch or another capture, which encodes its own maps; nothing patches a captured map.

**Device-side descriptors** (`tl.make_tensor_descriptor`, which write their maps into a global scratch) are launched since tilefish 0.0.7 and crate
0.0.5 (section 4.6). A host descriptor is still the cheaper one where the tensor is a parameter (encoded once per launch, on the host; a device-side one
by every program, on the device, before its first load).

A reader that predates the kind (tilefish 0.0.5, crate 0.0.3) refuses a manifest that has one (`kind must be pointer, scalar, fixed or given`). A manifest
without one, and its digest, are unchanged.

### 5.6 A null pointer: `null` (tilefish 0.0.7, crate 0.0.5)

A kernel may take a pointer that the use being declared never dereferences: a prebuilt kernel `k(const half* x, const half* bias, half* y, int n)` that reads
`bias` only when it is not null, or a Triton kernel that reads it only under a constexpr flag the declaration sets to false. Python declares that parameter
`bias: Null[f16]` (an extern only), and its ABI entry is

```json
{"slot": 1, "index": 1, "name": "bias", "kind": "null", "type": "*fp16", "ptx_type": "u64", "size": 8, "dtype": "f16"}
```

* `dtype` is one of the dtypes of the manifest and `type` its pointer type (as a pointer's), passed as a `u64` of 8 bytes; it has no other field (no
  `access`, `shape`, `strides`, `offset`, `dim`, `count_of`, `stride_of`, `value`, `alias`, `tensor_map`). A `triton` entry and a `binary` entry may have
  one.
* **A launch passes the 64-bit zero at its slot**, as it writes a `fixed` argument's value: a call gives no tensor for it, it binds no dimension, and it
  has no precondition (a `pointer_alignment` naming it names no pointer and is refused). A plan step binds no buffer to it. A Triton kernel compiled with
  one has the parameter as a pointer argument with no attribute (no `tt.divisibility`: nothing is loaded through it).
* It is in the declaration digest (`"kind": "null"` and the dtype, on the parameter), so a declaration of the same kernel with a tensor there is another
  operator, with entries of its own (one binary, two declarations, two stems: `scale_null` and `scale_bias` in `python/tests/types_kernels.py`).

A reader that predates the kind (tilefish 0.0.6, crate 0.0.4) refuses a manifest that has one (`kind must be pointer, tensordesc, scalar, fixed or given`).
A manifest without one, and its digest, are unchanged.

### 5.7 64-bit dimension scalars (tilefish 0.0.7, crate 0.0.5)

A `scalar` that carries a dimension or an expression of dimensions (a `dim` and no `stride_of`) may be `i64` (`u64`, 8 bytes): Python declares it
`n_elem: i64[n]` (an extern; `n_elem: n` is the `i32` it always was). It is a scalar like any other: its value is the expression evaluated at the dimensions
the tensors of the launch bind (it is never given by the call, unlike a `given` argument), and a launch refuses a value that does not fit (`0 ≤ v ≤ 2^63 − 1`).
So a prebuilt kernel that takes a `long` element count, or a Triton kernel that takes an `i64` count of a tensor of more than `2^31 − 1` elements
(`index_bits=64`), is declared with its count checked against the shapes. A `divisible` precondition and `tt.divisibility` apply to it as to an `i32` one.
The declaration digest has `"scalar_type": "i64"` on the parameter (a declaration without it digests as before). Both readers took an `i64` scalar since
schema 3 began (for a stride), so no reader changes: a manifest of tilefish 0.0.6 never had one with a `dim` only, and its digest is unchanged.

## 6. Preconditions

The list is as in schema 2, with these differences.

* `pointer_alignment`: `bytes` is the alignment the pointer *passed* must have (`address % bytes == 0`). For an ordinary pointer it is 16, as before,
  or (tilefish 0.0.7, crate 0.0.5) the size in bytes of its element, for a dense pointer that is not a device scalar: the **specialisation of a lowered
  kernel for views that do not start on 16 bytes** (below). For a pointer that has `strides`: **either 16, or the size in bytes of its element** (`max(1, bits // 8)` of its dtype). 16 means the operator was
  declared as 16-byte aligned (`Strided[..., 16]`), the offset dimension is `divisible_by` the elements in 16 bytes, and, for a Triton kernel, the
  pointer has `tt.divisibility=16` (see below); the element size means no alignment is assumed beyond the element's own, and Triton was given no hint.
  A device scalar's is 16. A `tensordesc`'s is 16 (the global address of a TMA map), whether it describes a dense tensor or a view; Triton is given no
  attribute for it.
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

**The specialisation of a lowered kernel for views that do not start on 16 bytes** (tilefish 0.0.7, crate 0.0.5). A lowered kernel compiles every pointer
as 16-byte aligned. `kernel.unaligned("a")` (Python) is the same kernel with the read-only parameter `a` compiled aligned to its element only: no
`tt.divisibility` for it, and a `pointer_alignment` of its element size. It is an operator of its own, `<kernel>__ua<indices>` (the indices of those
parameters in the declaration: `add_rows__ua0`), so its entries have their own stems, its declaration digest has `"align": <bytes>` on those parameters,
its Triton IR has no divisibility hint for them (so its build cache key differs), and its generated source and `source_sha256` are the kernel's. A graph
that passes a dense view at an offset that is not provably a multiple of 16 bytes to such a parameter (a row of a `[K, R]` int32 plane with an odd `R`)
calls the specialisation, so its plan's step names that entry; `tilefish.run(kernel, ...)` of a tensor whose address is not 16-byte aligned runs it too
(Python's replay says what to export when the manifest lacks it). A host launches it as any entry: the alignment it checks is the one the entry promises.
A reader that predates the change (tilefish 0.0.6, crate 0.0.4) refuses an entry whose dense pointer promises less than 16 bytes. Manifests without one,
and their digests, are unchanged.

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
in Python; `crates/tilefish/src/expr.rs` is a port of it). **A grid entry may multiply dimensions** (tilefish 0.0.7): `@tilefish.extern(...,
grid=lambda T, C: (tilefish.cdiv(T * C, 1024),))` is written `(cdiv(C*T, 1024),)`, one program per 1024 elements of a `[T, C]` tensor whatever `T` and `C`
are (a grid callable is evaluated with products allowed, `tilefish.dims.products()`; the factors are written in a canonical order, and the declaration
digest has the product as `{"prod": [...]}`). A host computes a grid at each launch, from dimensions the tensors bound, and never solves one, so a product
is fine there; a shape, a stride and a scalar stay affine, as a host binds the dimensions of a call by solving them. Both readers always took such a
grid: an expression that is not affine is evaluated as it is written (below). An expression is at most 512 characters and 256 tokens, nested at most 24 deep, and an integer
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
* `launch`: the zero value `{"global_scratch_size": 0, "profile_scratch_size": 0, "launch_pdl": false, "cluster_dims": [1,1,1], "num_ctas": 1,
  "cooperative": false}`, so a plain `cuLaunchKernel` runs it; or, for a binary declared with a thread-block cluster (`cluster=(2, 1, 1)`, section 4.5),
  that value with its `cluster_dims` and their product as `num_ctas`, launched with `cuLaunchKernelEx` (its grid counts clusters).
* `ptx_target`: for PTX the `--gpu-name` the PTX was assembled for (`sm_103a` when the PTX has `.target sm_103a`, else `sm_<arch>`); for a cubin `""`.
* `binary`: `{"source": "cubin"}` or `{"source": "ptx", "ptxas_flags": [<strings>]}` (exactly those keys; the flags are the argument vector given to
  `ptxas` before the file names: `["-lineinfo", "-v", "--gpu-name=sm_103a"]`, the flags Triton 3.5 passes, in the same spelling, since ptxas records
  its command line in the cubin and the machine code digest depends on it).
* `source_digest`: the sha256 (64 lowercase hex digits) of the bytes of the PTX file, or of the cubin file used for this `arch`;
  `source_digest_kind` is `"tilefish.binary.v1"`. For a cubin source it equals `sha256`.
* `toolchain`: `{"ptxas": <version text>}` for PTX, `{}` for a cubin. `declaration_digest`, `sha256`, `sha256_code`: as for any entry.

**Launch.** `cuModuleLoadData(cubin)`, `cuModuleGetFunction(symbol)`, then `cuLaunchKernel(grid, block, shared)` with `kernelParams` = the `abi.params`
in order: a `pointer` is its address (`u64`), a `scalar` its value (`i32`/`i64`), a `fixed` its `value` (`i32`/`i64`/`f32`), a `given` the value
the call gives (`i32`/`i64`); **nothing is appended**
(`scratch_slots` is 0). `cuFuncGetParamInfo`, when the driver can answer, must report exactly the `size` of each parameter and no more (Python
checks this at load). A cubin is refused for an architecture other than its own: the declaration has one cubin per architecture, and a replay on
`sm_90` finds no entry (and does not use the `sm_103` one).

**Identity and staleness.** A binary's identity is `source_digest`. A replay compares it with the digest of the bytes the declaration holds
(the declaration reads its file once, when the operator is declared) and refuses a mismatch as stale, pinned or not; no Triton and no GPU are needed
to tell. There is no IR digest.

**An implementation per architecture.** A prebuilt binary may be one implementation of an operator declared with `by_arch=`
(`tilefish.impl(ptx=...)`, `tilefish.impl(cubin={...})`, section 4.1): its entry has `implementation`, which is in its declaration digest, and its ABI is
its own (its C parameters, no scratch slots), whatever the ABI of the implementations of other architectures. A cubin implementation has a cubin for every
architecture that runs it.

**Preconditions and dims** are the declaration's, as for a Triton kernel: pointer alignment, shapes, `divisible`, `max_elements` (the default
`index_bits=32` applies to a prebuilt kernel too: its author says `index_bits=64` if it indexes in 64 bits), buckets. A prebuilt binary is not
compiled by Triton, so there are no `tt.divisibility` attributes; the preconditions are what is promised to the kernel.

## 9. The `library` section: a pinned cuBLASLt GEMM

`library` entries describe operations that a vendor library runs, with no file: today, a GEMM of bf16, f16 or f32 operands through cuBLASLt with
the algorithm pinned.

**The operation.** In tilefish's row-major terms, `out[M, N] = a[M, K] @ b[N, K]^T` (`+ bias[N]` for `impl: "cublaslt.gemm_bias"`): `a`, `b`, `bias`
and `out` are bf16, contiguous, 16-byte aligned; the accumulator is f32; the result is rounded to bf16 **once**; the bias is added in f32 to the
accumulator before that rounding (`CUBLASLT_EPILOGUE_BIAS`). That is the form of tilefish 0.0.5, and the form of every entry it wrote; section 9.2
says the others: f16 and f32 operands, an f32 result, `b` stored `[K, N]` (`out = a @ b`, the layout `NN`) and `a` stored `[K, M]` (the layouts `TT` and
`NT`). `M`, `N`, `K` are static or bucketed: one
entry per bucket combination (the stem names the buckets, like a kernel's). `N` and `K` may be any size (tilefish 0.0.5; 0.0.4 and crate 0.0.2
required multiples of 8): where a row of `k` or `n` elements does not start on 16 bytes, the leading dimensions are what decides which algorithms
are valid, and `cublasLtMatmulAlgoCheck` (which sees them) judges the pinned one when it is pinned, when a host loads it and, in Python, before every
run; the pointers stay 16-byte aligned.

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
| `dtypes` | `{"a": "bf16", "b": "bf16", "out": "bf16"}` plus `"bias": "bf16"` for the bias op (the other types: section 9.2) |
| `compute_type` | `"CUBLAS_COMPUTE_32F"` |
| `scale_type` | `"CUDA_R_32F"` |
| `transa`, `transb` | `"T"` and `"N"`: the transposes **as cuBLASLt is called** (below); for the other layouts (section 9.2) `transa` is `"N"` when `b` is `[k, n]` and `transb` is `"T"` when `a` is `[k, m]` |
| `m`, `n`, `k` | ints from 1 to 2147483647: the **tilefish** sizes, `a` is `[m, k]` (`[k, m]` for `TT` and `NT`), `b` is `[n, k]` (`[k, n]` for `NN` and `NT`), `out` is `[m, n]` |
| `lda`, `ldb`, `ldc`, `ldd` | ints: the leading dimensions as cuBLASLt is called; exactly `lda = ldb = k`, `ldc = ldd = n` (`lda = n` when `b` is `[k, n]`, `ldb = m` when `a` is `[k, m]`) |
| `epilogue` | `"none"` for `gemm`, `"bias"` for `gemm_bias` |
| `workspace_bytes` | int from 1 to 2^40: the workspace limit the heuristic was asked under **and the size of the workspace a launch provides** (default `8519680`, torch's default `CUBLAS_WORKSPACE_CONFIG`) |
| `algo` | 128 lowercase hex digits: the 64 bytes of the `cublasLtMatmulAlgo_t`, in memory order (eight `uint64_t`, as the host stores them) |
| `cublaslt_version` | int ≥ 1: the value of `cublasLtGetVersion()` of the library that chose `algo` (e.g. `120804` for 12.8.4) |
| `sm_count` | int ≥ 1 (a JSON integer: not `30.0`, not `true`, not `"30"`; no upper bound): the multiprocessor count (`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`, torch's `multi_processor_count`) of the device `algo` was pinned on. A MIG slice has fewer SMs than the whole GPU (a B300 slice of 1g has 30, the GPU 148), and an algorithm that partitions its work by the SM count (stream-K) is chosen for that count |
| `declaration_digest` | 64 lowercase hex digits: sha256 over the canonical declaration (name, impl, workspace limit, signature with the dimensions' constraints, and the options of section 9.1 that are not the default) **and the `sm_count`** (the canonical declaration object with the key `"sm_count"` added, format 2): a pin is the declaration on one device |
| `heuristic` | *optional* (section 9.1): absent, or exactly `"torch"` |
| `algo_source` | *optional* (section 9.1): absent, or exactly `"imported"` |
| `allow_bf16_reduction` | *optional* (section 9.1): absent, or exactly `true`; only on an entry of bf16 operands |
| `allow_f16_reduction` | *optional* (section 9.2): absent, or exactly `true`; only on an entry of f16 operands |
| `allow_f32_reduction` | *optional* (section 9.2): absent, or exactly `true`; only on an entry of f32 operands |
| `sm_count_target` | *optional* (section 9.3): absent, or an int from 8 to `sm_count`: the SM count the heuristic chose for, which every check and run sets |

`abi.params[i]` has exactly `index` (= i), `name`, `kind` (`"pointer"`), `dtype` (its type in `dtypes`: `"bf16"` in the form of 0.0.5), `access`
(`"in"`, and `"out"` for the last), `shape` (a list of ints and dimension expressions: `a` is rank 2, `b` rank 2, `bias` rank 1, `out` rank 2). The
shapes must agree with `m`, `n`, `k` (and the layout): an integer entry equals the size, an expression entry evaluates to it when the bucketed
dimensions have their `buckets` values (an expression that mentions a dimension with no bucket is refused).

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
   accumulator (and, for f32 operands, f32 products: section 9.2), alignment capabilities ≤ 16 bytes, and split-K to go with the reduction scheme `NONE` or `COMPUTE_TYPE` (f32 partial sums; a bf16
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

### 9.2 f16 and f32 operands, an f32 result, the layouts `NN`, `TT` and `NT` (tilefish 0.0.6, crate 0.0.4)

The form of a GEMM is the type of its operands, the type of its result and how `b` is stored. An entry says it in the fields it always had, so the
schema is unchanged and an entry of tilefish 0.0.5 (bf16, `TN`) is byte for byte what it was, with the same digests. A reader of 0.0.5 or crate
0.0.3 refuses an entry of another form (its `dtypes` or `transa` are not the ones it knows), which is right: it would run it as bf16 `TN`.

| | the form of 0.0.5 | the other forms |
|---|---|---|
| operands `a`, `b` | bf16 | bf16, f16 or f32, both the same (`dtypes.a`, `dtypes.b`, the `dtype` of the pointers `a` and `b`) |
| result `out` | bf16 | the operands' type, or f32 (`dtypes.out`); f32 for f32 operands (no 16-bit result of f32 operands) |
| `bias` | bf16 | the result's type (`dtypes.bias`): cuBLASLt takes no other. Measured on 12.8.4 (sm_103): an f32 bias with a 16-bit result, and a 16-bit bias with an f32 result, are `CUBLAS_STATUS_INVALID_VALUE` from the heuristic, so a declaration cannot have them |
| layout | `TN`: `b` is `[n, k]`, `out = a @ b^T`; `transa = "T"`, `lda = k` | `NN`: `b` is `[k, n]`, `out = a @ b`; `transa = "N"`, `transb = "N"`, `lda = n`, `ldb = k`, `ldc = ldd = n`; the pointer `b` has the shape `[k, n]`. `TT`: `a` is `[k, m]`, `out = a^T @ b^T`; `transa = "T"`, `transb = "T"`, `lda = k`, `ldb = m`. `NT`: `a` is `[k, m]` and `b` `[k, n]`, `out = a^T @ b`; `transa = "N"`, `transb = "T"`, `lda = n`, `ldb = m`. The pointer `a` of `TT` and `NT` has the shape `[k, m]` |
| opt-in to split-K partial sums in the output type | `allow_bf16_reduction` | named by the **operands'** type, as torch's flags are: `allow_f16_reduction` for f16 operands (whatever the result's type), `allow_f32_reduction` for f32 operands. Each is refused on an entry of another type |

How cuBLASLt is called: the layouts of `a` and `b` (cuBLAS's B and A operands) have the operands' type, `C` and `D` the result's type and the layout
of `D`, `BIAS_DATA_TYPE` is the result's type, for `NN` and `NT` cuBLAS's A operand is the `n x k` column-major matrix that `b [k, n]` is, used with
`transa = N`, and for `TT` and `NT` cuBLAS's B operand is the `m x k` column-major matrix that `a [k, m]` is, used with `transb = T`. `cublasLtMatmulAlgoInit` writes the four types into the algorithm, so the 64 bytes of a pin are for one pair of operand and result types
(not for one layout): step 4 rebuilds the algorithm with the entry's types, and a pin of other types is refused there ("its bytes are not those
cuBLASLt encodes"). The rule of step 4 is the same with the words of the form: split-K partial sums "in f16", or, for an f32 result, "in the output
(f32)", need the opt-in of the operands' type.

**`a` stored `[k, m]`: the layouts `TT` and `NT`.** `@tilefish.library(..., layout="TT")` takes `a: Tensor[K, M, T]` with `b: Tensor[N, K, T]` and
computes `a^T @ b^T`; `layout="NT"` takes it with `b: Tensor[K, N, T]` and computes `a^T @ b`. The layout is named by the transposes cuBLASLt is called
with (`transa` for `b`, `transb` for `a`), so the entry says it in `transa`, `transb`, `lda`, `ldb` and the shape of `a`, and a reader of crate 0.0.3
refuses it (its `transb` is not `"N"`). Every operand type and result above, the bias, the opt-ins, `algos=` (a row keyed `(m, n, k, bias, "TT")`) and
`sm_count_target` are as for the other layouts. Under `heuristic="torch"` with the opt-in of the operands' type, the pin is what torch runs for
`torch.mm(a.t(), b.t())` (`TT`) and `torch.mm(a.t(), b)` (`NT`) of a contiguous `a [k, m]`, and `torch.addmm` with a 1-D bias, byte for byte, for bf16,
f16 and f32 (TF32 off) operands: `python/tests/test_gpu_gemm_layouts.py`, 6 shapes (`K` from 8 to 3072) at 6 values of `M` from 2 to 2048, two operand sets
each, measured on a B300 MIG slice (sm_103) with cuBLASLt 12.8.4; for f32 that is the algorithm cuBLAS chooses for `cublasSgemm` with both transposes.
At `M = 1` an `a [k, 1]` is also a contiguous `[1, k]`, which torch multiplies as `TN` (or `NN`): there torch's bytes are those of that layout (measured:
`torch.addmm` of a bf16 `a [1024, 1].t()` differs from the `TT` pin at 2 of 6 shapes), so a declaration for one row takes `TN` or `NN`. The same test passes
on an RTX 5090 (sm_120) with cuBLASLt 13.1.0, except that there, at `M = 2`, the heuristic answers some f32 problems of `TT` (11 of 12) and `NT` (5 of 12)
with algorithm 10, a split-K count of 0 and the reduction scheme 1, which the vetting of step 4 refuses in both hosts ("split-K count 0 is below 1"): such a
pin is not made, so an f32 GEMM of a stored `[k, 2]` there takes another algorithm (`algos=`) or the layout `TN`/`NN`.

`workspace_bytes` is per declaration, any size up to 2^40 (a 64 MiB limit is `64 << 20`): the heuristic is asked under it and a launch provides it.

The `"torch"` posture of section 9.1 is torch's for f16 too: `@tilefish.library(..., heuristic="torch", allow_f16_reduction=True)` pins what
`torch.mm(a, b.t())` / `torch.addmm(bias, a, b.t())` run in f16 (`TN`), and what `torch.mm(a, b)` / `torch.addmm(bias, a, b)` run (`NN`), byte for
byte: `python/tests/test_gpu_gemm_f16.py`, `test_the_torch_posture_pins_what_torch_runs_in_f16`, 63 shapes (`M` from 1 to 2048, `N` and `K` of
projections and of convolutions as GEMMs) with three operand sets each, in both layouts, with and without a bias, measured on a B300 MIG slice
(sm_103) with cuBLASLt 12.8.4.

An `AlgoTable` row of the layout `NN` says it in its key, `(m, n, k, bias, "NN")` (`(m, n, k, bias)` is a row of `TN`); a configuration is encoded
with the types of the declaration that imports it. The canonical declaration has `"layout": "NN"` when the layout is `NN` (with `K = N` the signature
alone would not tell the layouts apart) and `"allow_f16_reduction": true` (`"allow_f32_reduction": true`) when it opts in; the types are in its
signature.

**f32 operands, TF32 off.** An entry of f32 operands (`dtypes` all `"f32"`, the result and a bias too) is multiplied as f32: its `compute_type` is
`CUBLAS_COMPUTE_32F` and its `scale_type` `CUDA_R_32F`, the values every entry has, and never `CUBLAS_COMPUTE_32F_FAST_TF32` (what torch selects with
`allow_tf32`), `_FAST_16F`, `_FAST_16BF` or `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` (cuBLAS 12.9 and later: f32 products emulated with nine bf16 ones).
What a host must hold the algorithm to, besides step 4 of section 9 (measured with cuBLASLt 12.8.4 and 13.1.0 on a B300 MIG slice, sm_103, and with
13.1.0 on an RTX 5090, sm_120; the values are those of the headers of the `nvidia-cublas` wheels 12.8.4.1 and 13.1.0.3):

* **The compute type is in the bytes.** `cublasLtMatmulAlgoInit` writes it as a little-endian `int32` at bytes 52 to 55 of the algorithm (`0x44`, 68,
  for `CUBLAS_COMPUTE_32F`), so the byte comparison of step 4 already refuses a pin encoded for another compute type; and the ids of the TF32 and
  emulated kernels (72 and 73, CUTLASS 3 `tf32gemm` on sm_10x; 21, 23, 43, 45, 58, 63; 77 and 78, `9xbf16gemm` with `bf16x9_inf_patching` kernels)
  cannot be encoded under `CUBLAS_COMPUTE_32F` at all (`CUBLAS_STATUS_NOT_SUPPORTED`), so they are refused as unknown ids.
* **The environment can turn `CUBLAS_COMPUTE_32F` into another compute type**, and the encoding shows it, while the matmul descriptor does not (it reads
  back 68 in every case): with `NVIDIA_TF32_OVERRIDE=1` cuBLASLt encodes `CUBLAS_COMPUTE_32F` as `_FAST_TF32` (77) and its heuristic answers TF32
  tensor-core kernels (`cutlass3x_sm100_tensorop_*tf32gemm` on sm_103, `cutlass_80_tensorop_s1688gemm` on sm_120); with
  `CUBLAS_EMULATE_SINGLE_PRECISION=1` (cuBLAS 13.1; 12.8 has no emulation and ignores it) it encodes it as `_EMULATED_16BFX9` (78), and on sm_103 its
  heuristic answers the bf16x9 emulation. `CUBLAS_EMULATION_STRATEGY` alone, and `NVIDIA_TF32_OVERRIDE=0`, change nothing for `CUBLAS_COMPUTE_32F`.
  So **both hosts read the compute type of a freshly encoded algorithm** (bytes 52 to 55 of `cublasLtMatmulAlgoInit` for the pin's id under
  `CUBLAS_COMPUTE_32F`) and refuse an f32 entry when it is not 68: "cuBLASLt encodes CUBLAS_COMPUTE_32F as CUBLAS_COMPUTE_32F_FAST_TF32 (77) in this
  process, which does not multiply f32 operands as f32 (NVIDIA_TF32_OVERRIDE=1 turns on TF32, CUBLAS_EMULATE_SINGLE_PRECISION=1 a bf16x9 emulation):
  an f32 GEMM is not run here". In such a process every f32 pin is refused, one made elsewhere (its bytes are not the process's encoding either) and
  its own, so an export there writes none (`test_gpu_gemm_f32.py`).
* **The numerical-implementation flags** (`CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS`), where the library reports them, must say an f32 accumulator
  (`ACCUMULATOR_32F`) and f32 inputs (`INPUT_32F` and no other input bit: `INPUT_TF32` is a TF32 kernel): "its products read the f32 operands as tf32,
  not as f32 (CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS 0x40202)". Under `CUBLAS_COMPUTE_32F` the f32 ids 0, 1, 10 to 16, 20, 55 and 57 report `0x80201`
  (`FMA`, `ACCUMULATOR_32F`, `INPUT_32F`), id 80 reports `0x201` (no input type: refused), and **id 76 reports 0**: no flags at all. Id 76 is a CUTLASS 3
  SIMT sgemm (`cutlass3x_sm100_simt_sgemm_*` kernels), the heuristic's answer for most f32 problems on sm_103 (44 of 56 asked with both postures,
  12.8.4 and 13.1.0) and what torch's f32 `mm` runs there, with an error within one f32 accumulation of the f64 result. An f32 algorithm that reports
  no flags is therefore held to its compute type alone (the two points above); a 16-bit one is still refused for it ("it does not accumulate in f32"),
  as before. The TF32 and emulated ids report 0 too, or `INPUT_TF32` (`0x40201`, `0x40202`), which is why the flags alone could not keep them out.
  (`CUBLASLT_ALGO_CAP_FLOATING_POINT_EMULATION_SUPPORT`, new in cuBLAS 13, is 1 for ids 77 and 78 only; no host reads it.)

The rule, word for word, is `tilefish._cublaslt.numerics_problems` and the crate's `numerics_problems` (`gemm.rs`).

**The opt-in of f32 operands.** With an f32 result, the split-K reduction schemes `INPLACE` (1) and `OUTPUT_TYPE` (4) add the partial sums in the output
rather than in a workspace of the compute type; without `allow_f32_reduction` they are refused, as for the other forms. torch has no flag for f32: it
never sets a reduction-scheme mask on an f32 GEMM, so its choice needs the opt-in (under `heuristic="torch"` cuBLASLt 13.1.0 chooses `INPLACE` split-K for
44 of the 252 problems of `test_gpu_gemm_f32.py` on sm_103 and for 122 on sm_120).

**What torch runs.** torch 2.9.1 with TF32 off (`torch.backends.cuda.matmul.allow_tf32 = False`, `torch.set_float32_matmul_precision("highest")`, its
defaults) runs `torch.mm` of f32 tensors as `cublasSgemm` (math mode `CUBLAS_DEFAULT_MATH`) on a handle whose workspace is `CUBLAS_WORKSPACE_CONFIG`'s
(8 519 680 bytes by default off sm_90; `:4096:8` is 32 MiB), and `torch.addmm` with a 1-D bias as cuBLASLt's `gemm_and_bias` (`CUBLAS_COMPUTE_32F`, no
reduction-scheme mask, its cuBLASLt workspace of 1 MiB). `@tilefish.library(..., heuristic="torch", allow_f32_reduction=True)` with those workspaces
(the defaults of the posture: 8 519 680 for `cublaslt.gemm`, 1 048 576 for `cublaslt.gemm_bias`; `workspace_bytes=32 << 20` under `:4096:8`) pins
what torch runs, byte for byte: `test_the_torch_posture_pins_what_torch_runs_in_f32_with_tf32_off` (63 shapes, `M` from 1 to 2048, three operand sets,
`mm` and `addmm`, `TN` and `NN`: 756 runs) has no miss on sm_103 with cuBLASLt 12.8.4 and 13.1.0 and on sm_120 with 13.1.0, with the default workspace
and, in a process of its own, with `CUBLAS_WORKSPACE_CONFIG=:4096:8` (756 more runs each). `cublasSgemm` is not a cuBLASLt call, but the algorithm
cuBLAS chooses for it is the one cuBLASLt's heuristic answers under this posture and workspace for every one of these shapes, `M = 1` (a GEMV kernel)
included. With `allow_tf32 = True` torch's f32 `mm` is not those bytes (TF32 tensor-core kernels; asserted on random operands).

### 9.3 SM partitions: `sm_count_target` (tilefish 0.0.6, crate 0.0.4)

A process may run its GEMMs on a partition of the device's SMs: a *green context* (`cuGreenCtxCreate` over `cuDevSmResourceSplitByCount`). What
cuBLASLt does there was measured with cuBLASLt 12.8.4 on a B300 MIG slice of 30 SMs (sm_103), green contexts of 8 and 16 SMs, f16 GEMMs of 124 problems
(31 shapes, `TN` and `NN`, with and without a bias):

* **Every SM-count query reports the device's count in a green context** (`cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)`, `cudaGetDeviceProperties`,
  torch's `multi_processor_count`: 30); only `cuCtxGetDevResource` reports the partition. So the entry's `sm_count` stays the device's, and a host
  on a partition passes the check of step 2 of section 9.
* **The heuristic's answers depend on the context that was current at the process's first `cublasLtCreate`**, not on an SM count: in a process
  whose first handle was made in a green context of 8 (16) SMs, 90 (80) of the 124 answer lists differ from the primary context's (mostly smaller
  clusters), for every later handle in every context, and its wave counts are still those of 30 SMs.
* **`CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = T`** makes the heuristic choose for `T` SMs (the wave counts scale by 30/`T`, the tiles and split-K
  counts change), and its answers are the same in the primary context and in a process whose first handle was made in a green context of at least
  `T` SMs (0 of 124 differ for `T` = 8 and 16). A target of the device's count (or more) answers as no target (0 of 124 differ). **Targets from 1 to 7
  kill the process** (`SIGFPE` inside `cublasLtMatmulAlgoGetHeuristic`).
* **What a pinned algorithm computes depends on neither the context nor the target**: 864 runs (10 problems, 32 algorithms, 9 context and target
  settings, 3 processes of different first contexts) and 192 runs of hand-made split-K algorithms gave identical bytes, with one launch
  configuration per algorithm in every setting; `cublasLtMatmulAlgoCheck`'s workspace does not change with the target.

So the bytes of a pin are the bytes on a partition, and what a partition changes is which algorithm a heuristic would choose. A declaration
that wants the algorithm for a partition says so: `@tilefish.library(..., sm_count_target=T)`, `T` an int of at least 8. Then:

| | |
|---|---|
| the entry | has the optional field `sm_count_target`: absent when the declaration has none (the heuristic chose for the device's count, as before), else an int from 8 to the entry's `sm_count` (a reader refuses any other value, `null` included). It is in the canonical declaration (`"sm_count_target": T`), so in `declaration_digest` too |
| pinning | the heuristic is asked with `SM_COUNT_TARGET = T` (and the device must have at least `T` SMs); an imported algorithm (`algos=`) is taken as it is, its table keyed by the device's `sm_count` |
| every check and run | both hosts set `SM_COUNT_TARGET = T` on the matmul descriptor of `cublasLtMatmulAlgoCheck` and of `cublasLtMatmul`, so a call is the call the pin was checked with |
| replay | Python refuses an entry whose target is not the declaration's ("pinned ... for 16 SMs (sm_count_target=16), but the declaration asks for the whole device") |

**How a host on a partition uses it.** Declare the GEMMs that run on a partition of `P` SMs with `sm_count_target=P`, and export them on the
device, in any context (the pins are the same in the primary context and in a partition of at least `P` SMs). The Rust host says which partition it
runs on with `Manifest::with_sm_partition(Some(P))` (a clone of the manifest; a plan takes it through `LaunchPlan::from_embedded`): `Manifest::gemm`
then refuses an entry chosen for another count (`LibraryEntry::sm_count_chosen_for`: its `sm_count_target`, or its `sm_count` when it has none), with
"was chosen for ..., but this host runs it on a partition of P SMs". A pin without a target runs on a partition with the bytes it has on the whole
device; only the choice was made for the whole device. A process that pins (`jit`, `export`) makes its first cuBLASLt handle in the primary
context (torch's, in Python), or else its pins without a target are those of the partition its first handle was made in; with a target they are not.
`Gemm::sm_count_target` and `BuiltLibrary.spec.sm_count_target` say the target of a loaded pin. `test_gpu_gemm_sm_target.py` checks the first-handle
rule and the target in a green context on the GPU it runs on.

### 9.4 A pinned cuDNN convolution: `impl: "cudnn.conv_fwd"` (tilefish 0.0.6, crate 0.0.4)

A `library` entry whose `impl` is `"cudnn.conv_fwd"` is a forward convolution that cuDNN runs, with what cuDNN runs pinned. Its fields are its own (a reader
dispatches on `impl` first: Python's `tilefish._library.validate_entry` hands it to `tilefish._conv.validate_entry`, the crate reads it into a `ConvEntry`),
and everything of sections 2 and 3 holds for it: a `(stem, arch)` pair is unique across kernels and every library entry, and the entry is digested whole.

**The operation.** `y = conv(x, w)`, the cross-correlation torch's `conv1d` / `conv2d` / `conv3d` compute, with `G` groups and, per spatial axis (one, two
or three), a stride, a zero padding on both sides and a dilation:

| layout | `x` | `w` | `y` |
|---|---|---|---|
| `NCHW` | `[N, C, *S]` | `[K, C/G, *R]` | `[N, K, *P]` |
| `NHWC` | `[N, *S, C]` | `[K, *R, C/G]` | `[N, *P, K]` |

`P = (S + 2 * padding - dilation * (R - 1) - 1) // stride + 1` per axis. `x`, `w` and `y` are of one type, `f16`, `bf16` or `f32`, dense in the layout,
aligned to the entry's `alignment`; the accumulator is f32 (`CUDNN_DATA_FLOAT`) and the result is rounded once; `alpha = 1`, `beta = 0`. There is no
bias: torch's cuDNN convolution adds its bias with a separate `add_` (an f16 add of the rounded result), which a tilefish kernel does the same way, and a
bias fused into a backend operation graph would change both the engines that can run it and the roundings. Asymmetric padding and transposed
convolutions are not offered.

**How cuDNN is called** (`python/src/tilefish/_cudnn.py`, `describe`; the crate's `ConvProblem::describe` gives the same, value for value, on every entry
of the cross-language vectors). A problem of one spatial axis is posed as one of two whose first spatial axis has size 1 (padding 0, stride 1, dilation 1):
cuDNN's way, torch's (`conv1d` views `[N, C, L]` as `[N, C, 1, L]`) and what the legacy API's 4-D descriptors take. Every tensor is given in cuDNN's
order, dims `[N, C, (D,) H, W]` (`[K, C/G, ...]` for `w`), with the strides of the layout (an axis of size 1 has the product of the sizes of the axes
inside it, as a dense tensor has and as torch's `fixSizeOneDimStride` writes). The convolution is `CUDNN_CROSS_CORRELATION` with the compute type
`CUDNN_DATA_FLOAT`, the paddings both pre and post. Then, by the API of the pin:

* **legacy** (`cudnnConvolutionForward`): `cudnnSetTensorNdDescriptorEx` with the layout's format (`CUDNN_TENSOR_NCHW` or `_NHWC`) for `x` and `y`,
  `cudnnSetFilterNdDescriptor` with that format for `w`, `cudnnSetConvolutionNdDescriptor`, `cudnnSetConvolutionGroupCount` and
  `cudnnSetConvolutionMathType`: `CUDNN_TENSOR_OP_MATH` for the math `"tensor_op"`, `CUDNN_FMA_MATH` for `"fma"`, `CUDNN_DEFAULT_MATH` for `"tf32"`. The
  call is given a workspace of the entry's `workspace_bytes` at every launch (what the algorithm needs is at most that), so that the call is the same at
  every launch: whether cuDNN chooses another kernel inside an algorithm for another workspace size is not known, and a pin must not depend on it;
* **backend** (`cudnnBackendExecute`): three backend tensors (`CUDNN_ATTR_TENSOR_UNIQUE_ID` 120 for `x`, 119 for `w`, 121 for `y`, torch's `'x'`,
  `'w'`, `'y'`; `CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT` the entry's `alignment`), a convolution descriptor, a `CONVOLUTION_FORWARD` operation (`alpha` and
  `beta` of `CUDNN_TYPE_FLOAT`) and an operation graph of it; an engine of the pin's global index, knob choices of the pin's `(type, value)` pairs, an
  engine configuration of them and an execution plan, finalized once when the entry is loaded; a variant pack of the three pointers and the workspace
  per launch.

**The fields** (exactly these; `sm_count_target` and `canary` are optional):

| field | type and rule |
|---|---|
| `stem`, `op`, `arch`, `abi`, `buckets`, `dims`, `preconditions`, `declaration_digest` | as in a GEMM's entry; `abi.params` are the pointers `x`, `w` (`in`) and `out` (`out`, the name every library result has), of the data type, with the shapes of the layout (an expression entry evaluates to the problem's size under `buckets`); every pointer has one `pointer_alignment` of the entry's `alignment` |
| `impl` | `"cudnn.conv_fwd"` |
| `dtypes` | `{"x": T, "w": T, "y": T}`, `T` one of `f16`, `bf16`, `f32` |
| `compute_type` | `"CUDNN_DATA_FLOAT"` |
| `math` | `"tensor_op"` or `"fma"` for 16-bit data; `"fma"` (TF32 off) or `"tf32"` for f32 data |
| `layout` | `"NCHW"` or `"NHWC"` |
| `n`, `c`, `k`, `groups` | ints from 1 to 2^31 - 1; `groups` divides `c` and `k` |
| `input`, `filter`, `output`, `stride`, `padding`, `dilation` | lists of one int per spatial axis (1 to 3 of them), `padding` from 0, the others from 1; `output` is the convolution's |
| `alignment` | 16, 32, 64, 128 or 256: the byte alignment every pointer has and the backend tensors declare |
| `workspace_bytes` | int from 1 to 2^40: the limit the pin was chosen under, the size a launch provides, and what a legacy call is given |
| `choice` | how the pin was chosen: `"v7"` (a legacy pin), `"instant"` or `"mode_b"` (a backend pin), or `"imported"` (either) |
| `pin` | `{"api": "legacy", "algo": a}` with `a` a `cudnnConvolutionFwdAlgo_t` (0 to 7), or `{"api": "backend", "engine": e, "knobs": [[t, v], ...]}` with `e` an engine's global index (from 0), `t` a knob type (0 to 42) and `v` an int64 other than -1, the pairs sorted by type, each type once. -1 is cuDNN's "not set" (how a plan's JSON and the heuristic's knob choices show a knob left at the engine's default; no knob of the engines measured takes it as a value, every `CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE` being 0 or more): a pin leaves such a knob out, so the plan made from it leaves the knob at its default, as the heuristic's did |
| `cudnn_version` | int >= 1: `cudnnGetVersion()` of the library that chose the pin (`91002` for 9.10.2); it does not tell two builds of one version apart (the canary does) |
| `sm_count` | int >= 1: the multiprocessor count of the device the pin was chosen on |
| `sm_count_target` | *optional*: an int from 1 to `sm_count`, the partition the pin was chosen for; set on the engine (`CUDNN_ATTR_ENGINE_SM_COUNT_TARGET`) of a backend pin at every load; on a legacy pin (the legacy API takes no target) only when it was imported |
| `canary` | *optional* (an entry without one loads as before): `{"seed": s, "y_digest": d}`, `s` an int from 0 to 2^32 - 1, `d` 32 lowercase hex digits: the digest of the `y` the pin computed at export from the operands of `s`, both made on the device (below). `export` records it with the seed 1. The first form of the field, `{"seed", "y_sha256"}` (host-made operands, a host sha256), is refused: export the manifest again |

**How the pin is chosen** (`@tilefish.library(..., impl="cudnn.conv_fwd", api=..., heuristic=...)`, `python/src/tilefish/_conv.py`):

| `choice` | what is asked |
|---|---|
| `"v7"` (legacy; the default of `api="legacy"`) | `cudnnGetConvolutionForwardAlgorithm_v7`, all of its answers, and the first whose status is `CUDNN_STATUS_SUCCESS`, whose determinism is `CUDNN_DETERMINISTIC` and whose `cudnnGetConvolutionForwardWorkspaceSize` is at most `workspace_bytes` |
| `"instant"` (backend; the default of `api="backend"`) | what torch 2.9 asks with `torch.backends.cudnn.deterministic = True` and `benchmark = False` (`Conv_v8.cpp`, `get_configs_from_heuristics`): the engine configurations of the heuristic in `CUDNN_HEUR_MODE_INSTANT`, then those of `CUDNN_HEUR_MODE_FALLBACK`, each filtered as torch filters them (no numerical note `NONDETERMINISTIC` or `DOWN_CONVERT_INPUTS`; for the math `"fma"` none with `TENSOR_CORE`, torch's `allow_tf32 = False`), and the first whose execution plan finalizes and needs at most `workspace_bytes`. The pin is that configuration's engine and every knob choice of it; the plan made again from the pin must have the heuristic's own plan's `CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION` (which names the engine and every knob, the default ones as -1), or nothing is pinned |
| `"mode_b"` (backend) | the same with `CUDNN_HEUR_MODE_B` first (torch's `TORCH_CUDNN_USE_HEURISTIC_MODE_B`) |
| `"imported"` | the declaration's `algos=` (`tilefish.CudnnTable(arch, sm_count, cudnn_version, {(n, c, k, input, filter, stride, padding, dilation, groups): pin})`, a pin being an int for the legacy API or `{"engine": e, "knobs": {t: v}}` for the backend) gives it; cuDNN is not asked; a device no table is for is refused at the build. A knob of the value -1 in a table's pin (not set) is dropped, so a table with it and one without give one pin and one digest. The rows the declaration uses are in its canonical declaration |

The canonical declaration (format 1) has the name, `impl`, `acc`, `workspace_bytes`, the signature, `layout`, `stride`, `padding`, `dilation`, `groups`,
`math`, `api`, `heuristic`, `alignment`, `sm_count_target` when it has one, and the tables' rows; an entry's `declaration_digest` adds its `sm_count`.

**The checks of a pin**, before it is recorded and when it is loaded (both hosts; Python's `tilefish._cudnn.check_pin`, the crate's `Manifest::conv`): a
legacy algorithm must be an answer of `cudnnGetConvolutionForwardAlgorithm_v7` for the problem with `CUDNN_STATUS_SUCCESS` and `CUDNN_DETERMINISTIC`, and its
workspace at most `workspace_bytes`; an engine must be one of the problem's (it finalizes for the operation graph, with the target), its numerical notes
must pass the filter above (`notes_problem`, the same words in both hosts), its knob choices must be knobs of the engine, each in its
`CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE`..`MAXIMUM_VALUE` and on its `STRIDE` (`knobs_problem`), the configuration and the plan must finalize, and the plan's
workspace must be at most `workspace_bytes`.

**Replay** (and the crate's load), in this order: the entry is found by `(stem, arch)`; the device's SM count must be `sm_count` ("re-export the manifest
on this device"); Python compares the entry with the declaration (the first difference is named) and the pin's API with the declaration's; the process's
`cudnnGetVersion()` must be `cudnn_version` (a pin names other kernels in another version; cuDNN has no build number, so two builds of one version are not
told apart); **a process with `NVIDIA_TF32_OVERRIDE` set, to any value, is refused** ("unset it"): it changes what cuDNN picks and runs behind the
declared math (measured), so no pin is made in such a process either; an imported pin must be the table's; then the
checks of the pin; then the canary, when the entry has one. A Rust host on a partition (`Manifest::with_sm_partition(Some(P))`) takes only an entry
chosen for `P` SMs (its `sm_count_target`, or its `sm_count` without one), as for a GEMM.

**The canary.** Two builds of one cuDNN version (the PyPI wheel and the system package of 9.13.0.50, say) answer the same `cudnnGetVersion()` and map
other engine libraries. So `export` runs each pin once more, on operands made of a seed, and records the digest of its `y`; a replay (Python, at the
first replay of the entry in the process and device: `tilefish._conv.check_canary`) and the crate's load (`Manifest::conv`, once per context) run it
again before the pin is used, and refuse the pin when the digest differs ("the canary of the cuDNN convolution ... gives y with digest ... here, and the
manifest recorded ..."). A Python replay inside a CUDA graph capture that would run a canary is refused ("run it once before the capture").

**The canary's operands and digest are made on the device**, by two kernels of one PTX module, the same text in both hosts (Python's
`tilefish._canary.CANARY_PTX`, the crate's `src/canary.ptx`; a test holds them equal), which the driver compiles for the device it is loaded on (PTX for
sm_70 and later: no compiler on the host). `mix(v)` is splitmix64's finalizer of `v + 0x9E3779B97F4A7C15` (modulo 2^64). `x` and `w` are dense in
the layout, of `n * c * prod(input)` and `k * (c / groups) * prod(filter)` elements of the data type, little-endian; element `i` of an operand (tag 0
for `x`, 1 for `w`) is made of `h = mix(base + i)`, `base = mix(2 * seed + tag)`: its low 16 bits (f16, bf16) or 32 bits (f32), with the most
significant byte `b` of the element becoming, for f16 (`s eeeee mm`), `(b & 0x80) | ((7 + ((b >> 2) & 7)) << 2) | (b & 3)` (an exponent field of 7
to 14: magnitudes in [2^-8, 1)), and for bf16 and f32 (`s eeeeeee`), `(b & 0x80) | (55 + (b & 7))` (exponents 110 to 125: magnitudes in [2^-17,
2^-1)): finite, normal numbers of many magnitudes and both signs, whose sums round by their order. `y` (`n * k * prod(output)` elements, zero-padded to
a whole 32-bit word) and the workspace start zeroed. The digest is of all of `y`: its words `w_j` (little-endian), `s1 = sum w_j * (mix(j ^ K1) | 1)`
and `s2 = sum w_j * (mix(j ^ K2) | 1)` modulo 2^64 with `K1 = 0x243F6A8885A308D3`, `K2 = 0x13198A2E03707344`, written as `s1` then `s2` in 16 lowercase
hex digits each. A sum does not depend on its order, so the device adds it in any grid (each warp's sums, then an atomic add); a change of one word
always changes it (the multipliers are odd), of several words with a chance of about 2^-128 to go unseen: the canary still says that this pin, at this
shape, gives these bytes in this process. Only the 16 bytes of the sums cross to the host. `tilefish._canary.canary_bytes` / `digest_of` and the crate's
`canary_bytes` / `canary_digest` are the same computations on the host (references; `python_conv.rs` holds the crate's to Python's, and
`test_gpu_conv.py` the device's to the host's). **The canary runs on a stream of its own** (torch's pool, `ctx.new_stream()` in Rust: non-blocking,
ordered with no other stream): the zeroed buffers, the fills, the convolution, the digest and the copy of its sums back, which waits for that stream
alone.

**What a canary costs.** The first form made the operands on the host (SHA-256 in counter mode) and hashed `y` on the host: for a plan set of 743
entries (38 GB of `x` and 38 GB of `y` in all, the largest 169 MB each) a Rust load took 93 s on a B300 MIG slice and 150 s on an
RTX 5090, of it 64 s / 101 s making the operands, 15 s / 21 s hashing `y` and 8 s / 19 s copying it back (measured per phase, release build, cuDNN
9.13.0). Made and digested on the device, the same 743 entries load in 1.4 s on the B300 slice and 1.1 s on the RTX 5090 (2.1 s the first time, the driver
compiling the PTX), against 0.08 s / 0.10 s with no canary; Python gives the same 743 canaries in 1.1 s on each. It never uses the NULL stream, which a CUDA green context does not order
against another stream (a harness whose copies were on the NULL stream raced the convolution inside a green context). Merging an export into a manifest
replaces an entry without a canary; two entries whose canaries differ are two binaries (`overwrite=True` to replace). A `cudnn_engines_sha256` of the
mapped engine libraries is not recorded: hashing them (about 570 MB for 9.10.2) at every load costs seconds in Python and much more in a debug build of
the crate, and the canary is the check.

**Streams.** Every call sets the handle's stream (`cudnnSetStream`) right before it runs, under the handle's lock, to the stream the call is given (a
launch's, a plan's); the memory the hosts touch for a convolution is on that stream too: Python's workspace, when the call is given none, comes from
torch's caching allocator on that stream (no memset), the crate's `Conv::alloc_workspace` zeroes on the stream it is given, and a plan's arena is the
plan's. `test_gpu_conv.py` captures calls of both APIs (the workspace given or allocated) with `capture_error_mode="global"` on a non-blocking stream: the
output is untouched until the replay, which is the eager bytes, so no part of a call is on the NULL stream or another; and the canaries of a manifest
run and pass with the NULL stream held busy, which is still busy when they return. The Rust gate captures its convolutions in
`CU_STREAM_CAPTURE_MODE_GLOBAL`, and runs every canary again (`Conv::check_canary`, the code of the load) with the NULL stream held by a host function,
which is still held when they return, and loads the convolutions of another manifest, which share the context's handle, with the NULL stream held:
they return while it is still held (`the_canary_of_a_load_is_never_on_the_null_stream`). What does wait is cuDNN's `cudnnCreate`, and **for all the
work queued on the device, on every stream**, not only the NULL stream's: measured with cuDNN 9.10.2 on a B300 MIG slice, a `cudnnCreate` took 985 ms
behind a one-second spin kernel queued on the NULL stream, on a blocking stream, on a non-blocking stream or on a torch stream alike (5 to 10 ms on an
idle device); the Rust gate holds a non-blocking stream with a host function and sees `CudnnHandle::new` return only when it is done
(`a_cudnn_handle_made_first_spares_the_loads_the_wait_of_cudnn_create`). That wait orders nothing wrongly, and it is once per context: **one cuDNN
handle per CUDA context**, shared by every convolution loaded in it from any manifest, alone or in a plan, and alive as long as the last of them (Python
has one per device, kept for the process). The first load of a convolution in a context therefore stalls behind whatever the host has queued (in the
crate, again after every convolution of the context was dropped); a host that loads convolutions while work runs makes the handle at start-up, before
it queues any work, and keeps it: `tilefish::CudnnHandle::new(&ctx)` in the crate (no load in the context makes a handle while it lives: the same test
drops every convolution, then loads them from a manifest that has loaded nothing with the stream held again, and they return before it is done; with
the handle dropped too, the next load waits), `tilefish.runtime.warm_cudnn(device)` in Python (or `tilefish.runtime.prepare` of a plan with
convolutions). The handle is not all that can wait: the first run of a cuDNN engine in a context (a load's canary) may wait for the device too, while
CUDA loads the engine's kernels lazily (measured for one of the gate's ten convolutions, with the handle made), so such a host loads its
convolutions at start-up as well. Every
use of a handle is under its lock in both hosts: the heuristic queries and the descriptors and plans of a load or a pin, a canary, and a call (which
sets the stream right before it runs); the crate also destroys a convolution's plan under it. Measured on a B300 MIG slice (cuDNN 9.10.2, release
build), loading 700 small entries (`[1, T, 64]` f16 by a k7 filter, T from 16 to 365, half legacy, half backend) in one context: with a handle per
entry, as before, 7.1 to 7.6 ms per load (6.5 to 6.8 of them the handle) and 6.0 GiB of device memory held (about 8.6 MiB per handle); with one handle,
1.6 ms per load with its canary (0.8 ms without), 12 MiB held; the canaries alone, run again on the loaded convolutions, 0.7 to 0.8 ms each. A Python
replay load of the same 700 takes 2.1 ms per entry (the pin prepared and its canary run; the canary 0.85 ms).

**The workspace and plans.** A launch is given a workspace by the caller, the last tensor (a `u8` vector, 256-byte aligned, overlapping no operand): at
least what the plan needs for a backend pin, and the entry's whole `workspace_bytes` for a legacy pin (which the call is given). A launch plan's library
steps take cuDNN convolutions as they take GEMMs: the arena's one workspace buffer is the largest `workspace_bytes` of its library steps, so nothing is
allocated in a launch. **Capture works with both APIs**: `cudnnConvolutionForward` and `cudnnBackendExecute` were captured in a CUDA graph by torch's
`torch.cuda.graph` even as the first execution of the process (no warm call), and the replay is the eager call's bytes (measured, cuDNN 9.10.2 on a B300
MIG slice); the plans' captures of `test_gpu_conv.py` and the Rust gate's `launch_raw` captures replay to Python's bytes.

**What was measured** (`python/tests/test_gpu_conv.py`, on three devices: a B300 MIG slice of 30 SMs, sm_103, with torch 2.9.1+cu128 and its cuDNN
9.10.2; the same slice with torch 2.9.1+cu130 and its cuDNN 9.13.0; an RTX 5090, sm_120, with torch 2.9.1+cu130 and cuDNN 9.13.0; every point below on each):

* the `"instant"` pin is torch's bytes (`torch.backends.cudnn.deterministic = True`, `benchmark = False`, TF32 off; NHWC through `conv2d` of channels-last
  tensors) for 15 problems, three operand sets each, where torch runs cuDNN (`torch._C._select_conv_backend` is `Cudnn`): 1-D NHWC f16 of 1 to 1024 channels
  with strides 1, 2 and 8, dilations 1, 3 and 9, 1-D NCHW f16 and bf16 with groups (depthwise f16 included), 2-D NHWC and NCHW f16 with a stride and groups,
  and f32 with TF32 off in 1-D, 2-D and 3-D. On sm_103 cuDNN 9.10.2's heuristic answers nothing in `CUDNN_HEUR_MODE_INSTANT` (nor in mode B) for these
  problems, and both torch and tilefish take `CUDNN_HEUR_MODE_FALLBACK`'s list: the posture is still torch's;
* there are no cuDNN bytes of torch for 16-bit data with the math `"fma"` (torch filters tensor-core engines for f32 only: with cuDNN 9.13.0 the first
  engine of an f16 problem is a tensor-core one, which `"fma"` refuses, and the bytes differ), and for the convolutions torch runs without cuDNN (a bf16
  depthwise one as `CudaDepthwise2d`, an f16 3-D one as `SlowDilated3d`): there a pin runs, repeats its bytes and is the f64 result within one f32
  accumulation and one rounding;
* a legacy pin of `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` (1) is torch's legacy path (`TORCH_CUDNN_V8_API_DISABLED=1`: with `deterministic`
  and no benchmark torch runs that algorithm with the math type of its data, `onlyDefaultAlgorithm`) for the f16 and f32 problems; the `"v7"` posture
  pinned that algorithm for every problem tried;
* a pin repeats its bytes in the process and in another process; `jit`, `replay`, a plan, its capture and the call alone give the same bytes;
* an SM count target of the device's count is as none; with targets below it (15 of 30 SMs on the B300 slice, 85 of 170 on the RTX 5090) no engine of
  these convolutions finalizes in cuDNN 9.10.2 or 9.13.0 (the heuristic answers nothing, and the engine of the plain pin is
  `CUDNN_STATUS_NOT_SUPPORTED`), so a declaration with such a target is refused at the build, never run without its target.

### 9.5 A GEMM with a cuBLASLt epilogue: `impl: "cublaslt.matmul"` (tilefish 0.0.12, crate 0.0.10)

A `library` entry whose `impl` is `"cublaslt.matmul"` is a GEMM whose epilogue cuBLASLt runs on the f32 accumulator, before the one rounding to the
result's type (RFC 0003, milestone E2). The two impls of section 9 stay byte for byte as they are; a reader of 0.0.11 or of crate 0.0.9 refuses an entry
of this impl (its `impl` is not one it knows).

**The operation.** In tilefish's row-major terms, with the layouts and the problem of sections 9 and 9.2:

`out[M, N] = ρ(f(acc))`, `acc = a @ b^T` (`a @ b` for `NN`, and `a^T` for `TT` and `NT`) in f32, `ρ` the one rounding to the result's type, and `f` by
`epilogue` and `beta`, in this order (the library adds `c` before the bias: measured, RFC 0003 X1):

| `epilogue` | cuBLASLt's | `beta` 0 | `beta` 1 |
|---|---|---|---|
| `"none"` | `CUBLASLT_EPILOGUE_DEFAULT` (1) | `acc` | `acc + c` |
| `"bias"` | `CUBLASLT_EPILOGUE_BIAS` (4) | `acc + bias` | `(acc + c) + bias` |
| `"relu"` | `CUBLASLT_EPILOGUE_RELU` (2) | `relu(acc)` | `relu(acc + c)` |
| `"relu_bias"` | `CUBLASLT_EPILOGUE_RELU_BIAS` (6) | `relu(acc + bias)` | `relu((acc + c) + bias)` |

`a` and `b` are bf16 or f16 (both one type `T`); `out`, the `bias [N]` and `c [M, N]` (the layout of `out`) are `T` too. That formula is what the entry's
pin is verified against (below), not a promise: the bytes are the library's under the pin (contract (iii)). Every operand is contiguous and 16-byte
aligned.

**How cuBLASLt is called.** As in section 9, with the matmul descriptor's `EPILOGUE` the value of the table (set for every epilogue, `DEFAULT` included),
and `BIAS_DATA_TYPE` (the result's type) and `BIAS_POINTER` for the two with a bias. `alpha = 1.0f`; `beta = 0.0f` with `C` the `out` pointer (never read)
when `beta` is 0, and `beta = 1.0f` with `C` the `c` pointer when it is 1; the layout of `C` is the layout of `D`. Every check and run (Python's
`tilefish._cublaslt.matmul(..., epilogue=, c_ptr=)`, the crate's `cublaslt::Problem` with an epilogue) sets the same descriptor, so
`cublasLtMatmulAlgoCheck` judges the algorithm with its epilogue.

**The fields.** Those of section 9 (`ENTRY_KEYS`), with `beta`, `contract` and `verification` added, and these rules where they differ:

| field | type and rule |
|---|---|
| `impl` | `"cublaslt.matmul"` |
| `epilogue` | `"none"`, `"bias"`, `"relu"` or `"relu_bias"` |
| `beta` | the int `0` or `1` (not `1.0`, not `true`) |
| `dtypes` | `{"a": T, "b": T, ["bias": T,] ["c": T,] "out": T}`, `T` `"bf16"` or `"f16"`: `bias` exactly for `"bias"` and `"relu_bias"`, `c` exactly for `beta` 1 |
| `abi.params` | the pointers `a`, `b`, [`bias` `[n]`], [`c` `[m, n]`], `out` `[m, n]`, in this order, of the types of `dtypes` (`out` is `out`, the others `in`) |
| `contract` | `"measured-exact"` or `"binary-defined"`: what the verification of the pin found (below) |
| `verification` | an object of exactly `formula`, `accumulator`, `operand_sets`, `elements`, `differ`, `max_ulp` (below) |
| `canary` | *optional*: `{"seed": s, "out_digest": d}`, `s` an int from 0 to 2^32 - 1, `d` 32 lowercase hex digits (below) |
| `heuristic`, `allow_bf16_reduction`, `allow_f16_reduction`, `sm_count_target` | *optional*, as in sections 9.1 to 9.3 (`allow_f32_reduction` is refused by the rule of the operands' type) |

`algo_source` is refused: the impl imports no algorithm yet. The `declaration_digest` covers the declaration's `epilogue`, `beta`, `canary` (whether
export records one) and `contract` (when it requires `"measured-exact"`).

**The pin.** It is chosen and vetted as a GEMM's (section 9, step 4), for the epilogue: the heuristic is asked with the descriptor above, and **every
epilogue but `"none"` gets no split-K** unless the declaration opts in (the reduction-scheme mask `0` of a GEMM with a bias; with `"none"` and `beta` 1
split-K with f32 partial sums is taken, as for `cublaslt.gemm`). The rule of step 4 holds with the words of the epilogue: "split-K count 3 with the relu
epilogue may apply it to a sum already rounded to bf16 (as the bias epilogue was measured to, with f32 partial sums too), which rounds twice; a GEMM with an
epilogue has no split-K unless its declaration opts in with allow_bf16_reduction=True" (Python's `reduction_problem`, the crate's `reduction_problem`;
for `"bias"` the words of section 9).

**The verification** (`tilefish._matmul.verify`, at every pin: `jit` and `export`). Where an epilogue's arithmetic can be written down its bytes may be
contract (ii): the formula on the pin's own accumulator, in f32, rounded once. That is a property of the pin, not of the epilogue, so each pin is run on
two seeded operand sets and compared bit for bit with that reference:

| field | what it records |
|---|---|
| `formula` | the formula of the table for the entry's `epilogue` and `beta` (a reader checks it is that one) |
| `accumulator` | `"f32-form"`: the accumulator is the f32 output of the pin's configuration re-encoded with an f32 `D` (`DEFAULT`, `beta` 0), which X1 found observable this way; `"plain-f32-pin"`: cuBLASLt does not take that configuration with an f32 `D`, and the f32 GEMM the same posture pins gave it |
| `operand_sets` | 2: `a ~ N(0, 1)` and `b ~ N(0, 1/k)` in their layouts, a bias `~ N(0, 1/4)` and `c ~ N(0, 1)`, with `+0`, `-0`, `+inf`, `-inf`, NaN, the largest finite values of both signs and a subnormal in the first 8 columns of the bias and of `c` (the second set has `c`'s next to them), and rows of `a` that give every epilogue special accumulators: a NaN, a zero and an infinity in each set, and two rows that overflow in the second |
| `elements` | the elements compared: `operand_sets * m * n` (a reader checks it) |
| `differ` | the elements whose bits differ from the reference (torch on the device: f32 adds, `torch.relu`, its cast, which on CUDA gives the bits of tilefish's own lowering of the same ops: the canonical NaN) |
| `max_ulp` | the largest distance between the two in units in the last place over the elements where both are finite (0 when none differs) |

`contract` is `"measured-exact"` exactly when `differ` is 0, and a reader refuses an entry where they disagree. A declaration with
`contract="measured-exact"` (and every rewrite of a plan's arithmetic into an epilogue, which declares it) refuses a pin that is not, at the build, and
its replay refuses an entry that is not.

What the verification found on a B300 MIG slice (sm_103, 30 SMs), for 14 shapes (`M` 32, 128, 640, 2048 by `N x K` of 1536 x 512, 3072 x 1024,
1024 x 3072, and 32 x 4096 x 4096, 100 x 1000 x 1000), bf16 and f16, both postures and the seven cases `none`+`c`, `bias`, `relu`, `relu_bias`,
`bias`+`c`, `relu`+`c`, `relu_bias`+`c` (392 pins per version; `python/tests/test_gpu_matmul.py`):

* **cuBLASLt 12.8.4: every pin is measured-exact**, the special accumulators and values included; no pin splits `k`.
* **cuBLASLt 13.1.0: 372 of 392 are measured-exact.** The 20 others are the pins of algorithm 21 at 32 x 1024 x 3072 (`relu`, `relu_bias`, `bias`+`c`,
  `relu`+`c` and `relu_bias`+`c`, both types, both postures): `bias`+`c` differs on 26% of the elements, by up to 15176 ULP (bf16), as X6 found; the
  four with a ReLU differ on 2048 to 2111 of 65 536 elements, all of them NaN pre-activations: this algorithm's ReLU gives `0` for NaN, where the others
  give the NaN (`torch.relu`). The `none`+`c` pins that split `k` (a split-K of 3 and 4 with f32 partial sums at `M = 32`) are measured-exact.

**The canary.** cuBLASLt exposes no build number, so `export` records `{"seed": 1, "out_digest": d}` for a declaration with `canary=True` (the
default), and a load runs it before the pin is used: Python at the first replay of the entry in the process and device
(`tilefish._matmul.check_canary`), the crate in `Manifest::gemm` (once per context), each refusing a different digest ("the canary of the cuBLASLt
matmul '<stem>' (<arch>, ...) gives out with digest ... here, and the manifest recorded ..."). An entry without one loads as before. Its operands are
made on the device by the PTX module of section 9.4 (`tilefish._canary`, the crate's `src/canary.ptx`), with the element rule of section 9.4 for the
entry's type `T`: `a` (tag 0, `m * k` elements), `b` (tag 1, `n * k`), `bias` (tag 2, `n`) and `c` (tag 3, `m * n`) when the entry has them, each dense
in its layout; `out` (`m * n` elements, zero-padded to a whole 32-bit word) and a workspace of the entry's `workspace_bytes` start zeroed; the pin runs
with the entry's descriptor, on a stream of its own (non-blocking, ordered with no other: as a convolution's canary); the digest is that of section 9.4
over the words of `out`.

**Replay**, after the steps of section 9: the entry's `epilogue`, `beta`, `dtypes` and operands must be the declaration's (Python compares the entry with
the spec), a declaration with `contract="measured-exact"` refuses a `"binary-defined"` entry, one with a canary (the default) refuses an entry without
one, and the canary runs. A launch takes `a`, `b`, [`bias`],
[`c`], `out` and the workspace, in this order, checked as a GEMM's (the crate's `Gemm::launch` and `launch_raw`, the C ABI's `tilefish_gemm_launch`;
`tilefish_gemm_epilogue` says the epilogue and whether it reads `c`). A launch plan's `library` step of this impl lists the same arguments.

## 10. Strict reading: every refusal

A reader of schema 3 refuses (Python: `TilefishManifestError` unless marked *spec*, which are `TilefishSpecError` of the per-binary compile spec, a
document tilefish writes next to a build and the host does not read):

* the top level: wrong `schema` type/value; `kernels` or `library` missing or not a list; an entry that is not an object; a repeated `(stem, arch)`
  across both lists.
* every kernel entry: a `kind` that is not `triton`/`binary`; a missing `stem`, `arch`, `symbol`, `cubin`, `sha256`, `sha256_code`, `shared`, `launch`,
  `toolchain`; `options` missing on `triton` and present on `binary`; `num_warps`/`num_stages` present on `binary`; `binary` missing on `binary`;
  `shared` not a non-negative int; `toolchain`, `launch`, `abi`, `options` not objects; `abi.scratch_slots` present and not `2`/`0` by kind;
  `source_sha256`/`source_digest` not 64 lowercase hex digits; `source_digest` and `source_digest_kind` not both present or both absent; an
  `implementation` (section 4.1) that is not the entry's `arch`, `"fallback"` or a name (an identifier of at most 64 characters, not `fallback` and not
  beginning with `sm_` and a digit, in any case); a `cuda_math` (section 4.7) that is malformed or not on a generated kernel's `triton` entry; an
  `instrument` (section 4.8) that is malformed or on a `binary` entry, and any entry with one unless the reader was asked for an instrumented export; a
  `total_num_warps` (section 4.2) on a
  `binary` entry, not an integer, not more than `num_warps` or more than 32, or with a `block` that is not `[32 * total_num_warps, 1, 1]` (the Rust
  reader refuses a Triton entry whose `block` is not `[32 * num_warps, 1, 1]` without it; Python compares that block with the spec's); a `persistent`
  (section 4.3) that is not exactly `{programs_per_sm[, sm_count]}`, a `programs_per_sm` not an int from 1 to 32, an `sm_count` not an int from 1 to
  2^20, or a grid of more than one entry; a `launch.global_scratch_align` (section 4.6) without a global scratch or that is not a power of two; for a
  Triton entry, an `options.num_ctas` (section 4.5) that is 1, is not `launch.num_ctas`, or is missing while `launch.num_ctas` is not 1; for a `binary`
  entry, a `launch` that is not the zero value but for a cluster of `num_ctas` blocks (at most 8), or that has a global scratch;
  for `binary`: `binary` not exactly one of `{"source":"cubin"}` and `{"source":"ptx","ptxas_flags":[strings]}`, `source_digest_kind` not
  `"tilefish.binary.v1"`, no `source_digest`, a `source_sha256`, a cubin used as it is whose `source_digest` is not its `sha256`; the cubin file name not a plain file name; the cubin's `sha256`, `sha256_code` or
  `.text.<symbol>` section wrong; for `triton` the shared-memory section rule.
* what an entry says of itself, which both readers check when they read it (Python: `Manifest._read`, as `crates/tilefish/src/manifest.rs` does; the
  digests of a document a caller pinned again vouch for the document, not for its sense): a missing `op`, `abi`, `grid`, `buckets`, `dims`,
  `preconditions` or `declaration_digest`; an `arch` that is not `sm_` and two or three digits, or that is not the architecture the cubin's ELF header
  names (`_elf.sm_of`, the Rust `Elf::sm`: an entry relabelled `sm_90` that holds an sm_103 cubin), or, for `binary`, a header that names none; a `stem`
  that is not `op` and `.<dim><value>` for each of `buckets`, in order (`rowop.S32`); a `declaration_digest` that is not 64 lowercase hex digits (`null`
  included); a parameter with a field no parameter has, without `slot`, `index`, `name`, `kind`, `type`, `ptx_type`, `size`, or at another `slot` than
  where its slots begin (its position, the `2 * rank` extra slots of every `tensordesc` before it counted), a scratch slot at another `slot` than after the
  last parameter's, and the fields of each kind (`read_params`): a `tensordesc` (section 5.5) on a `binary` entry, whose `type`, `ptx_type` and
  `size` are not `tensordesc`, `b8[128]` and 128, whose `dtype` has no map, with `dim`, `count_of`, `stride_of`, `value` or `alias`, a shape of no axis or more
  than 5, `strides` on an access other than `in` or whose last entry is not the fixed `1` or another one not a name, or a `tensor_map` that is not exactly
  its fields with the rules of section 5.5 (a box beyond 256 or that does not tile the block, an inner box that is not a multiple of 16 bytes or is
  wider than the swizzle span, a block that is not powers of two, element strides other than 1, an interleave other than `NONE`, a NaN fill of an
  integer, a data type of another dtype); a `tensor_map` on any other kind; a `pointer` whose `type` is not the pointer type of its `dtype`, whose `access` is not
  `in`/`inout`/`out`, that has `dim`, `count_of`, `stride_of` or `value`, or an `alias` that is not a string or is on an access other than `out`, is not a `u64` of 8 bytes, or has no `shape` (axes: an int ≥ 1 or an
  expression), or has `strides` and `offset` not together, on an access other than `in`, not one per axis of its shape (a name, or the fixed `1` on
  the last axis only), or an `offset` that is not a name; a `scalar` that has `dtype`, `access`, `shape`, `strides`, `offset`, `value` or `alias`, no `dim`, both
  `count_of` and `stride_of`, or another width than `u32`/4 or `u64`/8; a `fixed` argument on a Triton kernel, or with a `dim`, `count_of`,
  `stride_of`, `access`, `shape`, `strides`, `offset`, `dtype` or `alias`, or without a `value` of its type (a JSON `null` is as absent, in each); a
  `given` argument on a Triton kernel, of a type other than `i32`/`i64` or another width than `u32`/4 or `u64`/8, or with a `value`, `dim`,
  `count_of`, `stride_of`, `access`, `shape`, `strides`, `offset`, `dtype` or `alias`; a `null` pointer (section 5.6) whose `type` is not the pointer
  type of its `dtype`, that is not a `u64` of 8 bytes, or that has any other field; an
  expression outside the grammar, a name that `dims` does not have, more than 32 names, a static grid entry or divisor below 1 (section 7.1).
* the ABI as a whole (a host that reads without a spec checks these itself; Python checks them by comparing the entry with the compile spec, and
  `_specvalid` checks them in the spec): `params` kinds in `pointer`/`tensordesc`/`scalar`/`fixed`/`given`/`null`; a `stride_of` that names a `tensordesc`; a
  `scalar` type other than `i32`/`i64`, or `dtype` ≠ `type`; `strides` without
  `offset` or the reverse; `strides` of the wrong length or empty, with an integer other than `1`, an integer on a non-last axis, a string that no
  `stride_of` scalar carries, or two scalars for one axis; an `offset` that is not a string; a `stride_of` list of fewer than two objects or with an
  `(arg, axis)` pair twice; a `stride_of` object with a missing/extra key, a negative axis, an
  axis out of range, a target that is not a strided pointer, a `dim` different from the target's `strides` entry, or together with `count_of`; a `fixed`
  argument with a type not in `i32`/`i64`/`f32`, an `f32` value that is not a float, an integer type with a non-integer, a bool, or an out-of-range value;
  an `alias` (section 5.4) that names no `in` pointer, a strided one or a device scalar, one of another `dtype` or declared `shape`, or an input another `out` pointer names.
* preconditions: `max_extent` more than once on a pointer (a kernel with `index_bits=64` has none, so it is not required), with a value other than `2147483647`, or on a pointer without `strides`;
  `max_elements` on a pointer with `strides`; `max_elements` or `max_extent` on a `tensordesc`; `pointer_alignment` of a pointer other than 16
  or its element size (of a device scalar and of every `tensordesc`, other than 16); every pointer and every `tensordesc` must have exactly one
  `pointer_alignment`, and a `null` pointer none.
* every `library` entry: the field list, the types and the cross-field equalities of section 9, each by the rule given there; an optional field of section 9.1
  or 9.2 with any value but its one (`"torch"`, `"imported"`, `true`), `null` included; an `sm_count_target` (section 9.3) that is not an int from 8 to
  `sm_count`; a form section 9.2 does not list (operands of two types or of
  another type than bf16, f16 and f32, a result of another type than theirs or f32, a bias of another type than the result, a `transa` or `transb`
  other than `T` and `N`, an `lda`, an `ldb`, an `a` or `b` shape or a pointer `dtype` that is not the form's); the opt-in of the other operand type.
* every `cublaslt.matmul` entry (section 9.5): the rules of a GEMM's entry with its own field list (`beta`, `contract`, `verification` required;
  `canary` and the options of sections 9.1 to 9.3 but `algo_source` optional); an `epilogue` not of the four; a `beta` that is not the int 0 or 1; operands
  or a result that are not one type of bf16 and f16; a `bias` in `dtypes` and the ABI exactly for `bias` and `relu_bias`, a `c` exactly for `beta` 1; a
  `contract` not of the two; a `verification` that is not exactly its six fields, with a `formula` that is not the epilogue's, an `accumulator` not of the
  two, an `operand_sets` or `elements` below 1, a `differ` or `max_ulp` below 0, a `differ` above `elements`, a `contract` of `measured-exact` with a
  `differ` above 0 or the reverse, a `max_ulp` above 0 with no element that differs; a `canary` that is not exactly `{seed, out_digest}` (a seed out of
  0..2^32 - 1, a digest not 32 lowercase hex digits), `null` included.
* every `cudnn.conv_fwd` entry (section 9.4): a missing or unknown field; `dtypes` that are not one type of `f16`, `bf16`, `f32`; another `compute_type`;
  a `math` not of section 9.4 or not of the data type; a `layout` other than `NCHW`/`NHWC`; `n`, `c`, `k`, `groups` not ints from 1, `groups` not dividing
  `c` and `k`; spatial lists not of one length from 1 to 3, with values out of range, or a problem with no output or too many elements; an `output` that is
  not the convolution's; an `alignment` not one of 16 to 256, or a `pointer_alignment` precondition of another; a `workspace_bytes` out of range; a
  `choice` not of the four, or one whose API is not the pin's; a `pin` that is not exactly one of the two forms (an algorithm above 7, a negative engine, a
  knob type above 42, knobs unsorted or repeated); a `cudnn_version` or `sm_count` that is not a positive int; an `sm_count_target` out of 1..`sm_count`,
  or on a legacy pin that was not imported; a `canary` that is not exactly `{seed, y_digest}` (a seed out of 0..2^32 - 1, not an int, or `true`; a
  digest not 32 lowercase hex digits), `null` included, and the first form `{seed, y_sha256}`; a knob of the value -1 (not set: left out of a pin); the
  shapes of `x`, `w`, `out` not the problem's in the layout under the buckets.
* a CuTe DSL entry (section 4.9): a `binary` that is not exactly `{source: "cute", cutlass_dsl, options}` (a version and options of their forms), a
  `toolchain` that is not exactly `{cutlass_dsl, nvvm, ptxas}` (strings, the DSL the binary's), a `source_digest_kind` other than `"tilefish.cute.v1"` or
  no `source_digest`, a `ptx_target` that is not of the entry's architecture, a grid that names a dimension, a `persistent` grid, a `source_sha256`; a
  parameter that is not a `pointer`, a `tensormap` or a `bytes`, or a `pointer` with `strides`, `offset` or `alias`; a `tensormap` whose `type`,
  `ptx_type`, `size` are not `tensormap`, `b8[128]`, 128, with a field of another kind, without a dtype, an access or a shape, or with a `template` that is
  not 256 lowercase hex digits zero in its bytes 0 to 7 and 64 to 127; a `bytes` whose `type` is not `bytes`, whose `size` is not from 1 to 32764 or
  `ptx_type` not `b8[<size>]`, with any other field, or whose `value` is not `2 * size` lowercase hex digits; a `tensormap` or `bytes` parameter, or a
  `source_digest_kind` `"tilefish.cute.v1"`, on any other entry.
* *spec* only, for prebuilt binaries: a `binary` section that is not `{kind, symbol, shared, digests}` (for a CuTe DSL kernel: `{kind: "cute", host,
  source_sha256, cutlass_dsl, options}`); `attrs` not empty; `launch_params` not every arg
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
and its `cublaslt::result`, no hand-written driver FFI). Where it is stricter than Python, or differs from it, it is listed here. The C++ host
(`crates/tilefish-capi`: the C ABI `tilefish.h` and `tilefish.hpp`, `docs/en/rfcs/0002-cpp-host.md`) reads a manifest through this crate, so everything
here holds for it as written; its tensors are regions (`DevicePtr::from_raw`).

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
* `implementation` (section 4.1) is read into `Entry::implementation` (`Option<String>`), held to the rule of section 10, and used for nothing else: the
  crate loads the entry of the device's architecture, which is the implementation the export chose for it, and launches it with that entry's ABI (the entries
  of one stem on other architectures may have other ABIs: each `Plan` is per entry).
* A `null` pointer (section 5.6) is read into `ParamKind::Null { dtype_name }`; the plan of the entry writes 0 at its slot (as a `fixed` argument's bits),
  takes no tensor for it (`Plan::tensor_names()` does not list it) and has no check of it. A 64-bit dimension scalar (section 5.7) is a
  `ParamKind::Scalar` of `ScalarType::I64`, as a 64-bit stride always was, and a dense pointer's element alignment (section 6) is the alignment the plan of the
  entry checks; the crate needed nothing else for either.
* `total_num_warps` (section 4.2) is read into `Entry::total_num_warps` (`Option<u32>`, next to `Entry::num_warps`, the declared warps) and held to the
  rules of section 10. Every launch takes the entry's `block` (`Plan::block()`): a single launch, a step of a launch plan, and a captured graph. For a
  warp-specialised kernel that block is `32 * total_num_warps`, so the crate needs nothing else to launch one.
* `persistent` (section 4.3) is read into `Entry::persistent` (`Option<Persistent>`: `programs_per_sm`, `sm_count`). `Manifest::kernel` resolves the SMs of a
  persistent kernel when it loads it, the partition of `Manifest::with_sm_partition` or the device's multiprocessor count (`Kernel::sm_count()`; a kernel is
  cached per SM count too), refuses one declared for another count, and every launch cuts the grid to the programs they run (`Persistent::programs`;
  `Plan::prepare_for(tensors, given, device, sm_count)`, of which `prepare` and `prepare_with` are the uncut grid, the tiles).
* `launch.cluster_dims` and `launch.num_ctas` (section 4.5), with `LaunchAttributes::cluster()` and `Plan::cluster()`: `Prepared::grid` is the grid of
  blocks (the programs times the cluster) and `Prepared::cluster` the cluster, and a launch of a kernel of clusters (`Kernel::launch`, `launch_raw`, a step
  of a plan, a capture) is `cuLaunchKernelEx` with the cluster attribute (`cudarc::driver::sys::cuLaunchKernelEx`, which cudarc has no wrapper of: the
  safe launch orders the tensors' slices through cudarc's guards, as a GEMM's). `LaunchAttributes::global_scratch_align` (section 4.6) is read
  (`Option<u32>`); `blockers()` names what the crate does not launch.
* A global scratch (section 4.6) is the last tensor of a launch (`Plan::global_scratch()`, `GLOBAL_SCRATCH` last in `Plan::tensor_names()`), held to the
  bytes of the launch (`Prepared::scratch_bytes`), 256-byte alignment, `u8`, a written dense tensor of the kernel's device overlapping no other tensor;
  its address is the `global_scratch` slot. The emitter's struct of such a kernel takes `global_scratch: TensorMut<U8>` last. A plan step gives its
  workspace.
* A CuTe DSL entry (section 4.9) is read into `BinarySource::Cute { cutlass_dsl, options }` and `Entry::nvvm`; a `tensormap` is a `ParamKind::Pointer`
  with a `template` (`Param::template()`: it binds, is checked and is given as a tensor), and `bytes` a `ParamKind::Bytes`. A launch (a single one, a
  plan's step, a captured graph) passes the template with the tensor's address in its first 8 bytes (`Prepared::by_value`; an address that is null, not
  16-byte aligned or not below 2^57 is an `Error::Launch`) and the bytes as written.
* `cuda_math` (section 4.7) is read into `Entry::cuda_math` (`Option<CudaMath>`) and held to the rules of section 10. The crate cannot tell the CUDA
  release of the host's other code, so it checks nothing at load; `Manifest::check_cuda_math(release)` is the host's statement of it, and refuses a
  manifest whose generated kernels' functions were not measured on that release.

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
`[]` or `[1]`. `fixed` values are written at their slots as their type says (an `f32` is the JSON double rounded to the nearest `f32`). A `given`
argument's value is the call's: `Kernel::launch_with(stream, tensors, &[(name, value)])` (`check_with`, `launch_raw_with`; `Plan::prepare_with`): a name
the entry does not have, a name twice and a missing one are refused first (`Error::Launch`), and a value that does not fit an `i32` where the scalars'
ranges are checked (kind `shape`); `launch` gives no value, so it is refused for an entry that has one. A plan step's values are its `given`.

A `tensordesc` (section 5.5) is read into `ParamKind::Pointer` with `tensor_map: Some(TensorMap)` (its block, box, swizzle, L2 promotion, fill and data
type; `Param::slots()` is its `1 + 2 * rank` slot sizes, `Param::tensor_map()`), so it binds and is checked as a pointer is; a launch takes its tensor
(`TensorRef`/`TensorMut`), and after the preconditions `Plan::prepare` refuses what the driver would (kind `PreconditionKind::TensorMap`, Python's
`tensor_map`) and computes the arguments of `cuTensorMapEncodeTiled` (`Prepared::tensor_maps()`, `TensorMapEncode`: Python's `encode_arguments`, number
for number), with the shape and the strides at their slots (`Prepared::params()` has 0 at the map's own slot). `Kernel::check` (so every launch, and every
step of a plan at every execution) encodes the maps through `cudarc::driver::sys::cuTensorMapEncodeTiled`; `launch` passes the 128 bytes by value with
cudarc's launch builder and orders the tensor's slice through cudarc's tracking guards for the time of the launch, `launch_raw` points the slot at them, and
a capture holds them in the kernel node.

**Launching a library entry** (`gemm.rs`, `ffi/cublaslt.rs`). `Manifest::gemm(&ctx, name, buckets, arch)` loads a `Gemm` for the device of `ctx` once and
`Gemm::launch(&stream, &mut [a, b, [bias], [c], out, workspace])` runs it:

* Load refuses, in Python's order: a cuBLASLt that cannot be opened (cudarc opens it when first used, and *panics* if it is missing, so `ffi/cublaslt.rs` asks
  first), a cuBLASLt whose `cublasLtGetVersion()` is not `cublaslt_version` (an error naming both versions), an algorithm that fails the vetting of section 9
  (the checks of Python's `_validate_algo`: rebuilt with `cublasLtMatmulAlgoInit` and the configuration attributes and byte-for-byte equal, tile and stages ids
  and swizzling and custom option within the capabilities of its id, an f32 accumulator (for f32 operands: `CUBLAS_COMPUTE_32F` encoded as itself and f32
  inputs where the flags say them, section 9.2), alignment at most 16 bytes, split-K with a reduction scheme of `NONE` or
  `COMPUTE_TYPE`, or also `INPLACE` or `OUTPUT_TYPE` for an entry with `allow_bf16_reduction`), an algorithm `cublasLtMatmulAlgoCheck` does not approve for the
  problem, and one that needs more than `workspace_bytes`. The optional fields of section 9.1 are read into `LibraryEntry::heuristic` (`Heuristic`),
  `algo_source` (`AlgoSource`) and `allow_bf16_reduction`; the form of section 9.2 into `LibraryEntry::dtype`, `out_dtype` (`DType`), `layout`
  (`GemmLayout::Tn`, `Nn`, `Tt` or `Nt`), `allow_f16_reduction` and `allow_f32_reduction`, and the algorithm is vetted and run with those types and that layout; `sm_count_target` (section
  9.3) into `LibraryEntry::sm_count_target`, set on the descriptor of the check and of every launch, and checked against the partition a host gives
  (`Manifest::with_sm_partition`). The check runs once at
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
* **A GEMM with an epilogue** (`impl: "cublaslt.matmul"`, section 9.5) is read into the same `LibraryEntry` (`implementation` `LibraryImpl::Matmul`), with
  `epilogue` (`GemmEpilogue::None`, `Bias`, `Relu` or `ReluBias`; `GemmEpilogue::formula` is Python's `formula`), `beta` (0 or 1; `has_bias()`,
  `has_c()`), `contract` (`MatmulContract`), `verification` (`MatmulVerification`) and `canary` (`MatmulCanary`), all held to the rules of section 10
  with Python's words (`fields_of`, `matmul_fields`; the fields of a `cublaslt.matmul` on `gemm` and `gemm_bias`, and `algo_source` on it, are refused
  by name); an entry of `gemm` has `GemmEpilogue::None`, one of `gemm_bias` `GemmEpilogue::Bias`, `beta` 0 and no contract. Every check and run sets
  the entry's epilogue on the matmul descriptor (`DEFAULT` included, as Python's `_Problem` does), `BIAS_DATA_TYPE` and `BIAS_POINTER` for the
  epilogues with a bias, and `beta = 1.0` with the call's `c` as `C` (the layout of `D`) when the entry has `c`, else `beta = 0.0` with `C` the
  `out` pointer; the vetting holds every epilogue but `none` to no split-K without the opt-in, in the words of Python's `reduction_problem` with
  its `epilogue=` (`reduction_problem` takes the `GemmEpilogue`: Python's `bias` and `epilogue` in one). A call is `a`, `b`, [`bias`], [`c`],
  `out` and the workspace, in that order, checked as a GEMM's (`c` is `[m, n]` of the result's type and overlaps nothing written); a launch
  plan's library step lists the same arguments (`Call::c`). `Gemm::epilogue`, `beta`, `has_c`, `contract`, `canary` and `implementation` say
  what was loaded. **The canary**: `Manifest::gemm` of an entry with one runs it once per context before the pin is used, after the vetting and
  the check, exactly as Python's `canary_digest` computes it: on a stream of its own (`ctx.new_stream()`, never the NULL stream), the operands made
  by the fill kernel of `canary.ptx` (`a` tag 0, `b` tag 1, `bias` tag 2, `c` tag 3, of the entry's type, dense in their layouts:
  `LibraryEntry::canary_elements`), `out` zero-padded to a whole word and a workspace of the entry's `workspace_bytes` zeroed, the GEMM with the
  entry's descriptor, the digest kernel over `out`, and only its 16 bytes copied back; another digest is refused with the words of Python's
  `check_canary` ("the canary of the cuBLASLt matmul '<stem>' (<arch>, algorithm id ..., epilogue ...) gives out with digest ... here, and the
  manifest recorded ... (seed s): ..."). The canary's kernels are loaded once per context while a loaded GEMM or convolution holds them
  (`src/canary.rs`, shared with the convolutions); `Gemm::check_canary` runs it again. The contract is read and said, not enforced: a host that
  wants only measured-exact pins checks `Gemm::contract` (Python's replay refuses a binary-defined entry only for a declaration that asks for
  `contract="measured-exact"`, which a host without the declaration does not have).
* **The C ABI** (ABI 1.1): `tilefish_gemm_launch` takes the tensors in the same order (`c` before `out`), and `tilefish_gemm_epilogue(gemm,
  &epilogue, &has_c)` says the epilogue as `TILEFISH_EPILOGUE_NONE`, `_BIAS`, `_RELU` or `_RELU_BIAS` (0 to 3: tilefish's values, not cuBLASLt's
  `cublasLtEpilogue_t`) and whether a launch takes `c`; `tilefish::Gemm::epilogue()`, `has_c()` and `has_bias()` in `tilefish.hpp`. A load
  refused by the canary is `TILEFISH_ERROR_MANIFEST`.

**Launching a convolution** (`conv.rs`, `ffi/cudnn.rs`). `Manifest::conv(&ctx, name, buckets, arch)` loads a `Conv` once per context and
`Conv::launch(&stream, &mut [x, w, out, workspace])` runs it (`launch_raw` with regions captures it). Load refuses, in order: a partition the pin was
not chosen for, a device of another SM count, `NVIDIA_TF32_OVERRIDE` set, a libcudnn that cannot be opened (cudarc's `cudnn` feature, dynamically loaded:
opened only when a convolution is loaded, so a manifest without one never opens it), another `cudnnGetVersion()`, a pin that fails the checks of
section 9.4, and a canary of another digest (made and digested on the device, on a stream of its own; `canary_bytes`, `canary_operands` and
`canary_digest` are the host references; `Conv::check_canary` runs it again); it
makes the legacy descriptors or the finalized execution plan once, with the context's cuDNN handle, which every convolution loaded in the context
shares (`Conv::shares_cudnn_handle`). A launch takes the handle's lock (the handle's stream is its state), sets the stream and runs; the tensors go through the plan of the entry as a GEMM's do, and the workspace through the same checks, with at least
`Conv::workspace_required()` bytes (the plan's need, or the whole `workspace_bytes` of a legacy pin). `ConvEntry` holds the reading
(`problem: ConvProblem`, `pin: ConvPin`, `choice`, `cudnn_version`, `sm_count`, `sm_count_target`, `canary: Option<ConvCanary>`); `Manifest::library()` is the GEMMs and
`Manifest::convolutions()` the convolutions. A launch plan's library step loads whichever its entry is.

**The emitter** (`tilefish.hosts.rust`). It writes a struct for each kernel (Triton or prebuilt: `launch` takes every pointer as a `TensorRef` or `TensorMut`;
stride scalars, `fixed` arguments and `null` pointers are not parameters) and for each library entry (`launch` takes `a`, `b`, `bias`, `out`, `workspace`; `gemm()`, `M`, `N`, `K`,
`BIAS`, `WORKSPACE_BYTES`, `PINNED_WITH`; a `cublaslt.matmul` also takes `c` before `out` when its `beta` is 1, has `EPILOGUE` and `BETA`, and its
docs say the formula with the names of the launch, the epilogue, the contract the pin of each architecture was verified to and its canary; the
structs of `gemm` and `gemm_bias` are what they were), with the digest of the manifest embedded. `tilefish.hosts.cpp` writes the same classes for
C++ (`EPILOGUE` and `BETA` as `static constexpr` members). It refuses what the reader refuses. The entries of one stem on several
architectures must agree on everything but the device, except that the entries of an operator with an implementation per architecture (each with an
`implementation`) may differ in what section 4.1 lets differ, their ABI included: they must share the tensors (name, order, access, dtype, shape, a view or
not), the fixed arguments per call and the constraints of the dimensions of their tensors and buckets, which is what `launch` is typed from; the docs of the
struct then say each architecture's implementation, grid, block and constexprs and, when their ABIs differ or one is a prebuilt binary, each one's kind,
descriptors, scalars, fixed arguments and views; and `ops=` checks each entry against the implementation its architecture selects.

**Launch plans** (not read by the crate yet). `docs/en/plan-schema-1.md`: `plan.json` pins a manifest and orders the launches of its entries over one arena, so a host runs a whole forward from it, capturable in a CUDA graph (`test_gpu_plan.py`; the golden fixtures `python/tests/golden/plans/` have the digests to reproduce).

## 13. Where it is tested

| what | tests (`python/tests/`) |
|---|---|
| top level, digest, the three additions, refusal of schema 2 | `test_manifest.py`, `test_manifest_schema3.py`, `test_rust_host.py` |
| `kind`, `abi.scratch_slots`, prebuilt binary entries | `test_binary_externs.py`, `test_manifest_schema3.py` |
| `cuda_math` (section 4.7) | `test_cudamath.py`, `crates/tilefish/tests/cuda_math.rs`; on a GPU `test_gpu_cuda_math.py` |
| the cubin does not record the dynamic shared memory of a Triton kernel (`shared` too small is not detectable, section 4) | on a GPU `test_gpu_shared_size.py` |
| strided pointers, `strides`/`offset`/`stride_of`, `max_extent`, alignment | `test_strided_specs.py`, `test_views.py`; on a GPU `test_gpu_strided.py` |
| a stride that several views share (`stride_of` a list) | `test_shared_strides.py`, `test_rust_host_schema3.py`; on a GPU `test_gpu_shared_strides.py` |
| a dimension only an `out` tensor determines (section 7) | `test_output_dims.py`; on a GPU `test_gpu_output_dims.py` |
| an `out` pointer that may be written over an input (`alias`, section 5.4) | `test_alias.py`; on a GPU `test_gpu_alias.py` |
| device scalars | `test_devscalar.py`; on a GPU `test_gpu_devscalar.py` |
| prebuilt PTX and cubin, byte identity with the Triton path | `test_binary_externs.py`; on a GPU `test_gpu_binary.py`, `ptx/*.ptx` |
| the `library` section | `test_library_ops.py`, `test_cublaslt.py`; on a GPU `test_gpu_gemm.py`, `test_gpu_cublaslt.py` |
| its optional fields (section 9.1): the posture, imported algorithms, bf16 split-K partial sums | `test_library_options.py`, `test_cublaslt.py`; on a GPU `test_gpu_cublaslt.py` (torch's bytes, the opt-in), `test_gpu_gemm.py` (jit and replay) |
| f16 operands, an f32 result, the layout NN, `allow_f16_reduction` (section 9.2) | `test_library_f16.py`; on a GPU `test_gpu_gemm_f16.py` (torch's bytes in both layouts, the f64 reference, jit, replay, plans, imported tables, the opt-in) |
| the layouts TT and NT (section 9.2) | `test_library_layouts.py`; on a GPU `test_gpu_gemm_layouts.py` (torch's bytes for bf16, f16 and f32, the f64 reference, jit, replay, a plan and its capture, an imported table) and the Rust gate |
| a cuDNN convolution (section 9.4) | `test_library_conv.py` (the declaration, the entry, the reader, a replay's checks, a plan, the emitter, the binding's rules); on a GPU `test_gpu_conv.py` (torch's bytes, torch's legacy path, repeatability across processes, jit, replay, a plan and its capture, imported pins, the refusals, the SM count target, the canary and its stream, global-mode captures) and the Rust gate |
| f32 operands, TF32 off, `allow_f32_reduction` (section 9.2) | `test_library_f32.py` (the declaration, the vetting rule, the digests of older entries); on a GPU `test_gpu_gemm_f32.py` (torch's bytes in both layouts and two workspace configurations, the TF32 and layout traps, the f64 reference, TF32 and emulation refused, jit, replay, plans, imported tables, the opt-in) |
| `sm_count_target` (section 9.3) | `test_library_sm_target.py`; on a GPU `test_gpu_gemm_sm_target.py` (a green context: the first-handle rule and the target; the bytes do not depend on the target) |
| a GEMM with a cuBLASLt epilogue (section 9.5) | `test_matmul.py`; the emitters: `test_rust_host_schema3.py` (`golden/rust_matmul.rs`, `cargo check` of the launchers), `test_hosts_cpp.py`; on a GPU `test_gpu_matmul.py`, and the Rust and C++ gates (`rust_gate.py`: `MATMUL_FORMS`, every epilogue with `beta` 0 and 1, bf16 and f16, the four layouts, three buckets of `M`, with a canary and without; the plan `epilogue_chain`) |
| the emitter | `test_rust_host.py`, `test_rust_host_schema3.py` (goldens in `python/tests/golden/`) |
| `options.enable_fp_fusion` of a hand-written kernel (`@tilefish.extern(enable_fp_fusion=...)`) | `test_lower_options.py`; on a GPU `test_gpu_extern_fusion.py` |
| `implementation`: an operator with an implementation per architecture (section 4.1) | `test_per_arch.py`; on a GPU `test_gpu_per_arch.py` (and the Rust gate) |
| named implementations, prebuilt binaries as implementations, an ABI per implementation (section 4.1) | `test_per_arch_impls.py` (the declaration, the digests of 0.0.6, the specs, the build's refusals, export and replay, the readers, calls, plans, the emitter and `cargo check` of its output); on a GPU `test_gpu_per_arch_impls.py` (a TMA descriptor form, a pointer form and a PTX form of one copy: jit, export, replay, plans; the sm_120 cases run on an sm_120 GPU) |
| `tensordesc`: TMA descriptors (section 5.5): the declaration, the spec, the slots, the readers, the launch checks, the encode, a plan on a fake backend, the emitter | `test_tensordesc.py`; on a GPU `test_gpu_tma.py` (Triton and Gluon kernels against Triton's own launch, plans and captures) and the Rust gate |
| `total_num_warps`: warp-specialised kernels (section 4.2): the build's checks, the cache, the readers, the block of a launch, the emitter | `test_warp_specialization.py`; on a GPU `test_gpu_warp_specialization.py` (Gluon and Triton warp-specialised kernels against Triton's own launch, CUDA graphs, plans and captures) and the Rust gate |
| reproducible exports (section 3.1): the names of source files, the content-addressed generated sources (atomic, checked, the fallback, processes at once), two exports from two checkouts | `test_reproducible_builds.py`; on a GPU `test_gpu_repro.py` (a graph with a lowered kernel, generated ops, a hand-written extern and a PTX, exported in two processes from two directories: the same files and digests), `test_gpu_build.py`, `test_gpu_plan.py` |
| `null` pointers (section 5.6): the declaration, the spec, the readers, the launch, a plan, the emitter | `test_null_pointers.py`; on a GPU `test_gpu_types.py` (a Triton kernel and a hand-written PTX, jit, replay, a plan) and the Rust gate |
| CuTe DSL kernels (section 4.9): the declaration, the spec, the parameters from captured launches, the PTX's reads, the entry, every refusal of the reader, a replay's PTX check, the emitters, the launch's bytes | `test_cute_externs.py`, `crates/tilefish/tests/cute_externs.rs`; on a GPU `test_gpu_cute.py` (kernels of the CUTLASS examples against the DSL's own launch, torch, cuBLASLt; exports in two processes; replay, plans, captures) and the Rust and C++ gates |
| 64-bit dimension scalars (section 5.7) | `test_dim_scalars.py`; on a GPU `test_gpu_types.py` (a tensor of `2^31 + 1000` elements, jit and replay) and the Rust gate |
| a dense pointer aligned to its element, the specialisation of a lowered kernel (section 6) | `test_unaligned_views.py`; on a GPU `test_gpu_types.py` (rows of a plane in a plan and alone, jit, replay, reference) and the Rust gate |
| launch shapes (sections 4.3 to 4.6, and a grid that multiplies, 7.1): the declarations, the specs and their digests, the build's checks, the readers, what a launch is given, a plan's workspace as a scratch | `test_launch_shapes.py`; on a GPU `test_gpu_launch_shapes.py` (`launch_kernels.py`: torch's bytes in jit, replay, plans, `tilefish.capture` and `torch.cuda.graph`; a persistent kernel on 1 to 30 SMs, a Triton cluster and a PTX one, device-side descriptors) and the Rust gate |

The Rust crate (`crates/tilefish/tests/`):

| what | tests |
|---|---|
| strict reading of schema 3, the older schemas | `strict_reading.rs`, `library_section.rs` (the optional fields of section 9.1 and the forms of 9.2 too), `schema3.rs` (synthetic manifests, `common/synthetic.rs`; `implementation` of section 4.1; `null` pointers, 64-bit dimension scalars and dense pointers aligned to their element) |
| the forms of section 9.2: Python's verdict on entries of every form and on each changed in one field | `python_library.rs` (`data/python_library.json`, made by `export_rust_vectors.py`) |
| the GEMMs with an epilogue of section 9.5: reading and every refusal; Python's verdict on entries of every form and on each changed in one field (and on `gemm` entries with the fields of a matmul), the epilogue, `beta`, contract, verification and canary read, the canary's operands; the checks of a call with `c`, the rule of split-K with an epilogue | `library_section.rs` (`a_matmul_entry_...`); `python_library.rs` (`matmul_library_entries`, `matmul_reading` of `export_rust_vectors.py`); `gemm.rs` (`a_call_of_a_matmul_has_its_bias_and_c_before_out`, `every_epilogue_but_none_has_no_split_k_unless_the_entry_opts_in`); the C ABI's values: `tilefish-capi/tests/abi.rs`, `tests/c/abi.c`, `tests/cpp/wrapper.cpp`; on a GPU `gpu_gate.rs` (`the_gate_has_gemms_with_an_epilogue`: the loads and their canaries, a capture of each, a forged canary refused under a recomputed pin; `the_canary_of_a_gemm_is_never_on_the_null_stream`), `gpu_plan_gate.rs` (the plan `epilogue_chain`) and `cpp_gate.cpp` (`matmuls`: `tilefish_gemm_epilogue`, a forged canary refused) |
| the convolutions of section 9.4: Python's verdict on entries of every form and on each changed in one field, the pin, the canary and the cuDNN problem each describes, and the canary's operands | `python_conv.rs` (`data/python_conv.json`, made by `export_rust_vectors.py`); the checks of a call with no GPU: `conv.rs` |
| views, device scalars, fixed arguments, binaries, the plan of a GEMM, the digests | `schema3.rs`, `library_section.rs`, `launch_checks.rs` |
| Python's decision on 992 calls (views, device scalars, binaries included), the digest vectors, Python's refusals of digesting | `python_launches.rs`, `python_canonical.rs`, `cross_language.rs` |
| `tensordesc` (section 5.5): reading, refusals, the slots, the launch checks and the arguments of the encode | `tensor_maps.rs` |
| `total_num_warps` (section 4.2): reading, refusals, the block of a plan, the digest | `warp_specialization.rs` |
| `implementation` (section 4.1): names, prebuilt binaries, the entries of one stem with an ABI per architecture launched from the same tensors | `schema3.rs`, `by_arch.rs` |
| launch shapes (sections 4.3 to 4.6, 7.1): a grid that multiplies, a persistent grid cut to the SMs, clusters and their compile option, a global scratch as the last tensor, the refusals, the digests | `launch_shapes.rs`; on a GPU `gpu_gate.rs` (`the_gate_has_launch_shapes`, `a_persistent_kernel_gives_the_same_bytes_on_every_partition`, `graph_capture_of_launch_shapes`, `a_global_scratch_that_does_not_hold_the_launch_is_refused`) and `gpu_plan_gate.rs` (`the_plans_of_launch_shapes_have_their_scratch_and_run_on_a_partition`) |
| on a GPU, byte identity with Python's `replay` of every feature, the refusals, graph capture of a GEMM and of a strided kernel, the hazards of other streams | `gpu_gate.rs` with `python/tests/rust_gate.py` (TMA descriptors: `the_gate_has_tma_descriptors`, `a_descriptor_of_a_tensor_the_driver_refuses_...`, `graph_capture_of_kernels_with_tma_descriptors`) |
