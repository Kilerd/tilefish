# The C++ host

**English** | [简体中文](../zh-CN/guide/cpp.md)

`crates/tilefish-capi` is the C ABI of the Rust host: `libtilefish_capi` (a
shared and a static library) and `tilefish.h`, which cbindgen generates from the
crate and a test keeps current, with a header-only C++17 wrapper, `tilefish.hpp`.
Every load, verification, check and launch is the `tilefish` crate's: there is
no second implementation, so a C++ host refuses what the Rust host refuses and
runs the same machine code (`docs/rfcs/0002-cpp-host.md`). No manifest or plan
schema changes.

```sh
cmake -S crates/tilefish-capi -B build && cmake --build build   # cargo builds the library
ctest --test-dir build                                         # C99 and C++17 tests, no GPU needed
cmake --install build --prefix /opt/tilefish                   # find_package(tilefish): tilefish::tilefish, tilefish::tilefish_static
```

```cpp
#include "tilefish.hpp"

auto ctx = tilefish::Context::primary(0);   // the device's primary context: the CUDA runtime API's
auto manifest = tilefish::Manifest::load("build/", "<the manifest's digest>");
auto kernel = manifest.kernel(ctx, "gelu_erf");
tilefish::Region x{x_ptr, x_bytes, 0}, y{y_ptr, y_bytes, 0};   // device memory the host owns
kernel.launch(stream, {tilefish::TensorRef<tilefish::Bf16>(x, {4096}), tilefish::TensorMut<tilefish::Bf16>(y, {4096})});

auto run = tilefish::LaunchPlan::load("build/", "<the plan's digest>").instantiate(ctx, stream);
run.launch(stream, {{"x", xt}, {"out0", yt}}, {{"n", 64}});
auto graph = run.capture(stream, {{"x", xt}, {"out0", yt}}, {{"n", 64}});
graph.replay(stream);
```

`tilefish.hosts.cpp` writes typed launchers and plan runners, as
`tilefish.hosts.rust` does, from a pinned manifest or plan only, with every
field validated: one class per binary, GEMM and convolution, whose `load`
refuses a manifest of another digest and whose `launch` takes one typed tensor
per parameter (a `bf16` tensor cannot be given where an `f32` one is declared),
and one class per graph with an `Externals` and a `Scalars` struct. With
`embed=True`, the manifest and its cubins go into a companion source file
(`launchers_embedded.cpp`) and `load_embedded()` verifies them with the pin.

```python
from tilefish.hosts import cpp
cpp.emit(m, "launchers.hpp", embed=True)          # m = tilefish.runtime.export(...), or a directory with pin=
cpp.emit_plan("build/", "runner.hpp", pin=plan_digest)
```

What the host owns, and how the boundary behaves:

- **Memory** is a region (`tilefish_tensor`, `tilefish::Region`): an address,
  a length and a device that the host vouches for, as for
  `DevicePtr::from_raw` in Rust; the host keeps it allocated until the launches
  that use it are done (for a captured plan, until it is destroyed) and orders
  it against its other streams. Everything else is checked before anything is
  enqueued, the tensor fitting its region included.
- **Contexts and streams** are the host's. A context handle is a device's
  primary context, retained and released, or a context the host lends
  (`Context::borrow`, a green context's for example), which tilefish never
  destroys. A stream is a `CUstream`, the same type as `cudaStream_t`. Every call
  leaves the thread's current context as it found it.
- **CUDA graphs.** A single launch on a stream the host is capturing is captured
  like any kernel node. A plan is captured with `capture` (a plan launch on a
  capturing stream is refused): the captured plan owns its graph and orders the
  arena with every other run.
- **Errors** are a status per call (`TILEFISH_ERROR_*`: one per kind of error of
  the crate, and invalid arguments, no driver and a caught panic) with the
  thread's last error (`tilefish_last_error`: the message, the precondition, the
  `CUresult`); `tilefish.hpp` throws them as `tilefish::Error`. A program built
  without exceptions uses `tilefish.h`.
- **Threads.** Every handle may be used from several threads at once, except a
  captured plan (one thread at a time). Runs on several streams at once need an
  arena each.
- **cuBLASLt and cuDNN** are opened at run time by name (`LD_LIBRARY_PATH`, or a
  copy the host loaded first), as in Rust, and a pin made with another version is
  refused at load.
- **Versions.** `tilefish_abi_version()` and the header's `TILEFISH_ABI_MAJOR`
  and `TILEFISH_ABI_MINOR`: a minor version only adds functions and values, and
  `tilefish.hpp` refuses a library of another major version.

The C++ half of the byte-identity gate runs the cases of the Rust gate from the
same directory: `python rust_gate.py prepare WORK` also writes the C++
launchers and plan runners (the block's, and every fused plan's), `cmake -S crates/tilefish-capi -B build
-DTILEFISH_BUILD_GPU_GATE=ON -DTILEFISH_GATE_DIR=WORK` builds
`tilefish_cpp_gate` (`crates/tilefish-capi/tests/gpu/cpp_gate.cpp`: kernels,
GEMMs, convolutions, plans launched and replayed, fused plans, a shared arena,
carried buffers, weights bound once, refusals, graphs of single launches), and
`python rust_gate.py compare WORK --host cpp` holds what it wrote to Python's
replay byte for byte.
