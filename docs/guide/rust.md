# The Rust host

The `tilefish` crate (`crates/tilefish`, version 0.0.1) is the
host of a manifest directory, without Python or Triton: it reads the manifest as
strict JSON, recomputes the digests exactly as Python does (the `sha256` of each
cubin, the `sha256_code` of its machine code, and the digest of the whole
manifest) and refuses a directory whose digest is not the pin it is given
(`Pin::Digest`; `Pin::TrustDirectory` is explicit). It builds on cudarc's safe
layer and calls the driver directly nowhere: `Manifest::kernel(&ctx, ...)` loads a
cubin into a `CudaContext` that the kernel then keeps alive, and the kernel
launches on a cudarc `CudaStream` (`Kernel::launch`, safe: the stream is checked
to be of the kernel's context) or, with the `unsafe` `Kernel::launch_raw`, on a
raw `CUstream` (it makes the kernel's context current for the call and puts back
what was). A tensor is made of a cudarc `CudaSlice` or view (`from_cuda`: no
`unsafe`; it borrows the slice, so the slice cannot be freed under it, and making
it records nothing on the slice) or of a region the host vouches for (`unsafe
DevicePtr::from_raw`, which has a lifetime and no tracking). The launch gives a
slice to cudarc's launch builder on the stream it launches on, which orders it
after the slice's earlier writes, and makes the slice's own stream wait for the
kernel, so freeing the slice (or any later work on its stream) runs after the
kernel however many other streams read it; a kernel that writes a slice from
another stream first waits for everything enqueued on the slice's own stream, so
it runs after all earlier readers. (A reader after a writer on another stream is
cudarc's own tracking, which needs a context that has made a stream of its own;
an `Err` from a launch does not promise that nothing was launched, and the crate
orders the free behind a kernel that may have run.) A CUDA graph is captured
with `launch_raw` (or `launch`) and regions: the graph holds the addresses and
nothing that keeps the memory alive, so it must not be replayed after the memory
is gone, and a launch of slices on a capturing stream is an error, not a graph,
whatever the cudarc context. The loaded function's parameter sizes are compared
with the ABI of the manifest (`cuFuncGetParamInfo`, as Python's `replay` does), and
a cubin whose ELF header names another architecture than its entry is refused.
`cudarc` panics when there is no NVIDIA driver, so `tilefish::driver_available()`
says first.

A manifest (schema 3) carries what a host needs to check a launch on its own:
every pointer parameter records its declared shape, with static axes as sizes
(`["Rows", 1024]`), and every entry the constraints of its named dimensions
(`divisible_by`, `buckets`, and the bucket the binary is built for). A launch
takes nothing but its tensors. The crate binds the dimensions from their sizes
(and the stride dimensions of a strided view from its canonical strides; the
tensors the kernel reads first, then an `Out` tensor for a dimension only it has),
checks that they agree across arguments and with the static sizes, the exact
bucket and `divisible_by`, the dtype and rank of each tensor (a device scalar is
one element), `max_elements` and, for a view, `max_extent`, that the runtime
scalars (derived from the dimensions, from the element count of the tensor they
count, or from a stride) fit the 32 or 64 bits they are passed as, aliasing, the
alignment each pointer promises (16 bytes, or a strided pointer's element size),
contiguity of a tensor given for a parameter that is not strided, and the grid
against the device's limits, and evaluates the grid from its expression. There is
no way to tell a launch what a dimension is. A manifest of an older schema is
refused by both hosts: export it again.

**Schema 3 and the crate.** The crate reads manifest schema 3
(`docs/manifest-schema-3.md`) in full and launches everything it describes:
Triton kernels and prebuilt PTX and cubin binaries (exactly their parameters, no
scratch slots, `fixed` arguments at their slots), strided views
(`TensorRef::from_cuda_strided`, or over a region with `new_strided`; any element
strides and any offset), device scalars (a tensor of shape `[]` or `[1]`), `i64`
strides, and the pinned cuBLASLt GEMMs of the `library` section
(`Manifest::gemm`, `Gemm::launch`, and `unsafe Gemm::launch_raw` for a CUDA
graph). A GEMM runs the 64-byte algorithm the manifest pinned through cudarc's
`cublaslt::result`: it is refused when the process has another cuBLASLt than the
one that chose it, when the algorithm is not one cuBLASLt would have produced,
and when the library does not approve it for the problem. Its workspace is the
caller's last tensor (`Gemm::alloc_workspace` allocates the size the entry says),
so a launch allocates nothing and can be captured; the slices of a GEMM are
ordered like a kernel's. The published crate 0.0.1 reads schema 2 only. `emit`
writes typed launchers for all of it; the committed fixtures of the crate are
schema 3 manifests.

```python
from tilefish.hosts import rust
rust.emit(m, "launchers.rs")   # m = tilefish.runtime.export(...); ops=[...] cross-checks the declarations
rust.emit("build/", "launchers.rs", pin=m.digest)   # a directory is read only with its pin
```

`emit` takes a pinned manifest (the one `export` returns, `Manifest.load(path, pin=...)`,
or a directory with `pin=`: the file it writes is compiled and run, so a manifest
nobody vouched for does not write it, and every field that reaches it is
validated) and writes one Rust module with a struct per binary (per bucket), whose
`launch` takes typed tensors (`TensorRef<'a, Bf16>`, `TensorMut<'a, F32>`: the
borrow of a cudarc slice, or a region of device memory with its length that the
host vouches for with `unsafe DevicePtr::from_raw`; a shape that needs more bytes
than the memory has is refused, and so is a launch on a stream of another context
or of a tensor of another context) and documents the binary: its tensors, its
dimensions and their constraints, what is compiled in. It also embeds the digest
of the manifest it was generated from and refuses any other, and with
`embed=True` the manifest and the cubins as well. The output is deterministic.
`ops=` is optional: each declaration it lists must be exactly what the manifest's
entry says, and gives the docs the tilefish signature.

The byte-identity gate (`python/tests/rust_gate.py` with
`crates/tilefish/tests/gpu_gate.rs`) checks on a GPU that the Rust host and
`replay` agree byte for byte, on a fresh export and with the manifest directory
and with the embedded manifest, for the five goldens, a bucketed row kernel over
each bucket, the hand-written kernels (outputs and in-place tensors), a `tl.dot`
kernel that needs more than 48 KiB of dynamic shared memory, strided views
(a fused qkv, any element offset, aligned and not, transposed and expanded
views, 64-bit strides), device scalars, prebuilt PTX and cubin binaries (the PTX
of a Triton build and hand-written PTX with `fixed` arguments) and the pinned
cuBLASLt GEMMs with and without a bias at every bucket; it also checks the
refusals of schema 3, a strided kernel and a GEMM captured in a CUDA graph, and
the order of the frees of their slices against other streams. On any machine,
`cargo test` and `pytest` check that the digests Rust computes from a committed
manifest directory (`sm_103`, and a cross-compiled `sm_120` one) are the ones
Python computes, that Rust decides the same as Python on a thousand calls, and
that Rust refuses what Python refuses.
