# tilefish (Rust host)

The Rust host of a [tilefish](https://github.com/Kilerd/tilefish) manifest.
tilefish builds kernels once into CUDA binaries plus a `manifest.json`. This
crate loads that directory and launches its cubins from Rust, with no Python
and no Triton, making every check Python's `replay` makes. The machine code is
the exact bytes Python's `replay` runs, and `tilefish.hosts.rust.emit`
generates typed launchers for a manifest.

**Status: early preview.** The API will change.

```rust
use tilefish::cudarc::driver::CudaContext;
use tilefish::{Arch, Bf16, Manifest, Pin, TensorMut, TensorRef};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !tilefish::driver_available() {
        return Ok(()); // CudaContext::new panics without an NVIDIA driver
    }
    // Production pins the digest that `tilefish.runtime.export` returned.
    let manifest = Manifest::load("build/", Pin::Digest("<manifest digest>"))?;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;
    let kernel = manifest.kernel(&ctx, "gelu_erf", &[], Arch::of_context(&ctx)?)?;

    let (x, mut y) = (stream.alloc_zeros::<u8>(4096 * 2)?, stream.alloc_zeros::<u8>(4096 * 2)?);
    let shape = [4096usize];
    let x = TensorRef::<Bf16>::from_cuda(&x, &shape)?;
    let y = TensorMut::<Bf16>::from_cuda(&mut y, &shape)?;
    kernel.launch(&stream, &mut [x.into_arg(), y.into_arg()])?; // every precondition is checked first
    Ok(())
}
```

These checks happen before anything reaches the driver:

- **At load:** the manifest is strict JSON, its digest is the pinned one, and
  every cubin matches its `sha256` and machine-code digest.
- **At launch:** the shapes bind the dimensions; dtypes, buckets,
  `divisible_by`, `max_elements`, aliasing, alignment, device and context, and
  the grid limits are all checked.

Safe code cannot free a slice under a running kernel, and a launch of slices
on a stream that is capturing a CUDA graph is refused, in every cudarc context.
`unsafe` is needed in exactly two places: vouching for a raw device region,
which is also how CUDA graphs are captured (with `launch_raw`), and launching on
a raw `CUstream`. See the crate docs for the exact contracts.

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
