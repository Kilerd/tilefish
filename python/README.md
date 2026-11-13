# tilefish

A typed tensor language for GPU inference: shapes, dtypes, layouts and
quantization formats in the type, one build into GPU binaries, and generated
Python, C++ and Rust hosts.

**Status: early preview.** This release type-checks: tensor types and
block-scaled formats (MXFP8, MXFP4, NVFP4), `extern` / `library` / `kernel` /
`graph` declarations, tracing into a typed IR, and Triton ahead-of-time
compile specs for `extern` kernels. Pure Python, no dependencies. A `@kernel`
also lowers to Triton source (`tilefish.lower`), and with the `gpu` extra
(`pip install "tilefish[gpu]"`: torch, Triton, cuda-python) a spec builds into
a cubin (`tilefish.aot.build`), a manifest directory holds the cubins
(`tilefish.runtime.export`), and a kernel runs in `jit` (built through its
spec, in memory), `replay` (the manifest's cubins, checked against its
digests; a stale binary is refused, and it needs a trust anchor: the manifest's
pinned digest, or an explicit `trust_directory=True`) and `reference` mode
(`tilefish.run`). A binary is identified by the Triton IR it compiles. The
`tilefish` crate on crates.io runs the same manifest from Rust, and
`tilefish.hosts.rust.emit` generates typed Rust launchers for it. The API
will change.

The design and the roadmap are at https://github.com/Kilerd/tilefish.
