//! The Rust half of the byte-identity gate, on a real GPU: launch the cubins of a manifest through the generated launchers
//! and write what they computed, for `python/tests/rust_gate.py` to compare with what Python's `replay` computed.
//!
//! Built only with `--features gpu-gate`, and needs
//!
//! * `TILEFISH_GATE_GENERATED` (at compile time): the launchers `tilefish.hosts.rust.emit` wrote for the manifest (with `embed=True`);
//! * `TILEFISH_GATE_DIR` (at run time): the directory `rust_gate.py prepare` filled: `manifest/` (the manifest directory),
//!   `cases.json` (each case: the launcher, its stem, whether it is a kernel or a GEMM, its tensors, the files of its inputs and of the
//!   outputs Python's `replay` gave; a view has its element `strides` and `offset` and the files hold the storage it is a view of) and the
//!   raw bytes of every tensor; and `workspace_case/`, which `rust_workspace_case.py` writes (a GEMM whose algorithm needs a workspace).
//!
//! ```text
//! python rust_gate.py prepare $WORK
//! python rust_workspace_case.py $WORK
//! TILEFISH_GATE_GENERATED=$WORK/generated.rs TILEFISH_GATE_DIR=$WORK \
//!     cargo test -p tilefish --features gpu-gate --test gpu_gate -- --test-threads=1 --nocapture
//! python rust_gate.py compare $WORK
//! ```
//!
//! (`byte_identity` launches each case through its typed launcher where this file has an arm for it, and through `Manifest::kernel` or
//! `Manifest::gemm` and untyped `TensorArg`s otherwise: the strided kernels, the device scalars, the prebuilt binaries and the GEMMs are
//! generated launchers that compile with this file, and are run through the checked path of the crate.)
//!
//! The tests: `byte_identity` runs every case and writes every tensor after the launch (outputs, and the inputs, so that an `InOut`
//! tensor that was updated in place is compared too), through tensors made of cudarc slices (no `unsafe`) and the safe `launch` on a
//! cudarc stream; `refusals` shows that Rust refuses what Python refuses (a precondition, a tensor that does not fit its region, a
//! tampered cubin, a wrong pin, a manifest of another architecture) and what only Rust has (a read-only tensor for a parameter that
//! is written); `graph_capture` launches inside a CUDA graph capture, with the safe `launch` and with `unsafe launch_raw`, and runs the
//! graphs; `a_launch_on_another_stream_...` and the tests after it are the hazards of cudarc's event tracking between streams (a free
//! under a queued launch, a second reader, a writer after two readers), with the launch stream held back by a host callback; the two
//! `..._on_a_capturing_..._stream_...` tests are the refusal of a safe launch of slices while a stream captures a CUDA graph, in a
//! context that tracks events and in one that does not; `launch_raw_puts_back_the_current_context` is what it says. Schema 3 adds
//! `the_gate_has_a_case_for_every_feature_of_schema_3`, `refusals_of_schema_3`, `graph_capture_of_a_strided_kernel_and_of_a_gemm`,
//! `a_gemm_of_slices_on_a_capturing_stream_...`, the free of the slices of a GEMM and of a strided kernel on another stream,
//! `a_written_view_lands_at_its_offset_and_nowhere_else` and `a_gemm_whose_algorithm_needs_a_workspace_...`. M3.3 adds
//! `the_gate_has_a_case_for_every_feature_of_m3_3` and `views_that_share_a_stride_must_all_have_it`, and `the_gate_has_a_gluon_kernel`.

#[allow(dead_code)]
mod generated {
    include!(env!("TILEFISH_GATE_GENERATED"));
}

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr as _, result};
use serde_json::Value;
use tilefish::{
    Access, AlgoSource, Arch, Bf16, DType, DevicePtr, Element, Error, F16, F32, I32, I64, Manifest, Pin,
    PreconditionKind, TensorArg, TensorMut, TensorRef, U8, elf::sha256_hex,
};

fn gate_dir() -> PathBuf {
    PathBuf::from(std::env::var("TILEFISH_GATE_DIR").expect("TILEFISH_GATE_DIR"))
}

fn cases() -> Value {
    let text = std::fs::read_to_string(gate_dir().join("cases.json")).expect("cases.json");
    serde_json::from_str(&text).expect("cases.json is JSON")
}

fn manifest() -> Manifest {
    // what production does: pin the digest of the manifest. Python's `export` returned it; the launchers carry it.
    Manifest::load(gate_dir().join(cases()["manifest"].as_str().unwrap()), Pin::Digest(generated::MANIFEST_DIGEST))
        .expect("the manifest loads with its pin")
}

struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl Gpu {
    fn new() -> Self {
        let ctx = CudaContext::new(0).expect("a CUDA context on device 0");
        let stream = ctx.new_stream().expect("a stream");
        Self { ctx, stream }
    }

    fn upload(&self, bytes: &[u8]) -> Buf {
        Buf { slice: self.stream.clone_htod(bytes).expect("upload"), shape: Vec::new(), view: None }
    }

    fn zeros(&self, nbytes: usize) -> Buf {
        Buf { slice: self.stream.alloc_zeros::<u8>(nbytes).expect("alloc"), shape: Vec::new(), view: None }
    }

    fn download(&self, buf: &Buf) -> Vec<u8> {
        self.stream.synchronize().expect("synchronize");
        self.stream.clone_dtoh(&buf.slice).expect("download")
    }

    /// The region of a buffer, as a raw pointer bound to the borrow of the buffer: only for what a safe constructor cannot do (a
    /// misaligned address, a tensor made before a graph capture without cudarc's event tracking). Nothing orders a launch on it against
    /// other streams: the tests that use it use the one stream.
    ///
    /// # Safety
    ///
    /// The buffer must outlive every launch that uses the region (the borrow says so), and nothing else may use it meanwhile.
    unsafe fn region<'a>(&self, buf: &'a Buf, len: u64) -> DevicePtr<'a> {
        // SAFETY: the address is the buffer's, on device 0, for the length given (its own unless the test says it lies).
        unsafe { DevicePtr::from_raw(buf.slice.device_ptr(&self.stream).0, len, 0) }
    }

    /// As [`Gpu::region`], with a lifetime that is not tied to the buffer: for a test that vouches falsely on purpose, or for two
    /// tensors over one buffer.
    ///
    /// # Safety
    ///
    /// As `region`, and the caller keeps the buffer alive and unused by anything else for as long as the region is.
    unsafe fn unbound_region(&self, buf: &Buf, len: u64) -> DevicePtr<'static> {
        // SAFETY: as above, with the lifetime the caller answers for.
        unsafe { DevicePtr::from_raw(buf.slice.device_ptr(&self.stream).0, len, 0) }
    }
}

/// A device buffer of bytes and the tensor it holds: a shape, and for a view of the buffer its element strides and the element offset of
/// its first element (the buffer is then the storage of a larger tensor, and the tensor a view of it).
struct Buf {
    slice: CudaSlice<u8>,
    shape: Vec<usize>,
    view: Option<(Vec<usize>, usize)>,
}

impl Buf {
    fn shaped(mut self, shape: &[usize]) -> Self {
        self.shape = shape.to_vec();
        self
    }

    /// The tensor is the view with these element strides, starting at this element of the buffer.
    fn viewed(mut self, strides: &[usize], offset: usize) -> Self {
        self.view = Some((strides.to_vec(), offset));
        self
    }

    fn nbytes(&self) -> u64 {
        self.slice.len() as u64
    }
}

/// A read-only tensor over a buffer, as cudarc sees it: no `unsafe`, and refused when the shape does not fit. A buffer that holds a view
/// is a view of it (`TensorRef::from_cuda_strided`).
fn r<T: Element>(b: &Buf) -> Result<TensorRef<'_, T>, Error> {
    match &b.view {
        Some((strides, offset)) => TensorRef::from_cuda_strided(&b.slice, &b.shape, strides, *offset),
        None => TensorRef::from_cuda(&b.slice, &b.shape),
    }
}

/// A tensor the kernel writes, over a buffer.
fn m<T: Element>(b: &mut Buf) -> Result<TensorMut<'_, T>, Error> {
    match &b.view {
        Some((strides, offset)) => TensorMut::from_cuda_strided(&mut b.slice, &b.shape, strides, *offset),
        None => TensorMut::from_cuda(&mut b.slice, &b.shape),
    }
}

/// The tensor of a case as a launch argument without a Rust element type: `dtype` is the name the manifest uses, `role` says whether the
/// kernel writes it.
fn arg_of<'a>(buf: &'a mut Buf, dtype: &str, role: &str) -> Result<TensorArg<'a>, Error> {
    macro_rules! typed {
        ($t:ty) => {
            if role == "in" { r::<$t>(buf)?.into_arg() } else { m::<$t>(buf)?.into_arg() }
        };
    }
    Ok(match dtype {
        "bf16" => typed!(Bf16),
        "f16" => typed!(F16),
        "f32" => typed!(F32),
        "i32" => typed!(I32),
        "i64" => typed!(I64),
        "u8" => typed!(U8),
        other => panic!("the gate does not know the dtype {other}"),
    })
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Launch the launcher a case names, on typed tensors made of cudarc slices: a launcher whose signature changed does not compile.
/// `bufs` are the tensors of the case in the order of the launcher's parameters. A case that no typed arm below knows (the library
/// operations among them, which have a workspace the typed arms would also need) goes through `Manifest::kernel` or `Manifest::gemm` and
/// untyped `TensorArg`s of the same slices (`launch_generic`).
fn launch(gpu: &Gpu, manifest: &Manifest, case: &Value, bufs: &mut [Buf]) -> Result<(), Error> {
    use generated::*;
    let s: &CudaStream = &gpu.stream;
    let launcher = case["launcher"].as_str().expect("the case names its launcher");
    if case["kind"] == "gemm" {
        return launch_generic(gpu, manifest, case, bufs);
    }
    match (launcher, &mut *bufs) {
        ("Softmax", [x, o]) => Softmax::load(manifest, &gpu.ctx)?.launch(s, r::<F32>(x)?, m::<F32>(o)?),
        ("Layernorm", [x, w, b, o]) => {
            Layernorm::load(manifest, &gpu.ctx)?.launch(s, r::<Bf16>(x)?, r::<Bf16>(w)?, r::<Bf16>(b)?, m::<Bf16>(o)?)
        }
        ("ResidualAddRmsnorm", [x, res, w, o]) => ResidualAddRmsnorm::load(manifest, &gpu.ctx)?.launch(
            s,
            r::<Bf16>(x)?,
            m::<F32>(res)?,
            r::<Bf16>(w)?,
            m::<Bf16>(o)?,
        ),
        ("GeluExact", [x, o]) => GeluExact::load(manifest, &gpu.ctx)?.launch(s, r::<Bf16>(x)?, m::<Bf16>(o)?),
        ("Swiglu", [g, u, o]) => {
            Swiglu::load(manifest, &gpu.ctx)?.launch(s, r::<Bf16>(g)?, r::<Bf16>(u)?, m::<Bf16>(o)?)
        }
        ("BucketedRmsDb96", [x, w, o]) => {
            BucketedRmsDb96::load(manifest, &gpu.ctx)?.launch(s, r::<Bf16>(x)?, r::<Bf16>(w)?, m::<Bf16>(o)?)
        }
        ("BucketedRmsDb512", [x, w, o]) => {
            BucketedRmsDb512::load(manifest, &gpu.ctx)?.launch(s, r::<Bf16>(x)?, r::<Bf16>(w)?, m::<Bf16>(o)?)
        }
        ("BucketedRmsDb1000", [x, w, o]) => {
            BucketedRmsDb1000::load(manifest, &gpu.ctx)?.launch(s, r::<Bf16>(x)?, r::<Bf16>(w)?, m::<Bf16>(o)?)
        }
        ("Rmsnorm", [x, w, y]) => {
            Rmsnorm::load(manifest, &gpu.ctx)?.launch(s, r::<Bf16>(x)?, r::<Bf16>(w)?, m::<Bf16>(y)?)
        }
        ("GeluErf", [x, y]) => GeluErf::load(manifest, &gpu.ctx)?.launch(s, r::<Bf16>(x)?, m::<Bf16>(y)?),
        ("Axpy", [x, y]) => Axpy::load(manifest, &gpu.ctx)?.launch(s, r::<F32>(x)?, m::<F32>(y)?),
        ("Mm", [a, b, c]) => Mm::load(manifest, &gpu.ctx)?.launch(s, r::<F16>(a)?, r::<F16>(b)?, m::<F32>(c)?),
        _ => launch_generic(gpu, manifest, case, bufs),
    }
}

/// The same launch through `Manifest::kernel` (or `Manifest::gemm`, with a workspace of the entry's size) and `TensorArg`s.
fn launch_generic(gpu: &Gpu, manifest: &Manifest, case: &Value, bufs: &mut [Buf]) -> Result<(), Error> {
    let s: &CudaStream = &gpu.stream;
    let stem = case["stem"].as_str().expect("the case names its stem");
    let arch = Arch::of_context(&gpu.ctx)?;
    let mut args: Vec<TensorArg<'_>> = Vec::new();
    for (t, buf) in case["tensors"].as_array().unwrap().iter().zip(bufs.iter_mut()) {
        args.push(arg_of(buf, t["dtype"].as_str().unwrap(), t["role"].as_str().unwrap())?);
    }
    if case["kind"] == "gemm" {
        let gemm = manifest.gemm(&gpu.ctx, stem, &[], arch)?;
        let mut workspace = gemm.alloc_workspace(&gpu.stream)?;
        let shape = [workspace.len()];
        args.push(TensorMut::<U8>::from_cuda(&mut workspace, &shape)?.into_arg());
        gemm.launch(s, &mut args)?;
        // the workspace is freed after the launch: cudarc orders its free behind the GEMM that used it
        return Ok(());
    }
    manifest.kernel(&gpu.ctx, stem, &[], arch)?.launch(s, &mut args)
}

/// Upload the tensors of a case: the inputs from their files, the outputs as zeros. In the order of the launcher's ABI.
fn tensors_of(gpu: &Gpu, case: &Value) -> Vec<(String, Buf)> {
    case["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            let shape: Vec<usize> =
                t["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
            let nbytes = t["nbytes"].as_u64().unwrap() as usize;
            let mut buf = if t["role"] == "out" {
                gpu.zeros(nbytes)
            } else {
                gpu.upload(&read(&gate_dir().join(t["input"].as_str().unwrap())))
            };
            assert_eq!(buf.slice.len(), nbytes);
            if let Some(strides) = t.get("strides") {
                // a view: the buffer is the storage of a larger tensor, and the tensor starts `offset` elements into it
                let strides: Vec<usize> =
                    strides.as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
                buf = buf.viewed(&strides, t["offset"].as_u64().unwrap() as usize);
            }
            (t["name"].as_str().unwrap().to_owned(), buf.shaped(&shape))
        })
        .collect()
}

/// Run every case of the gate through the launchers, with the kernels of `manifest`, and write what they computed to `<gate>/<subdir>`.
fn run_cases(gpu: &Gpu, manifest: &Manifest, subdir: &str) -> usize {
    let cases = cases();
    let out_dir = gate_dir().join(subdir);
    let _ = std::fs::remove_dir_all(&out_dir);
    let mut ran = 0;
    for case in cases["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let (names, mut bufs): (Vec<String>, Vec<Buf>) = tensors_of(gpu, case).into_iter().unzip();
        launch(gpu, manifest, case, &mut bufs).unwrap_or_else(|e| panic!("{id}: {e}"));
        let case_dir = out_dir.join(id);
        std::fs::create_dir_all(&case_dir).unwrap();
        for (name, buf) in names.iter().zip(&bufs) {
            std::fs::write(case_dir.join(format!("{name}.bin")), gpu.download(buf)).unwrap();
        }
        ran += 1;
    }
    assert_eq!(ran, cases["cases"].as_array().unwrap().len());
    ran
}

#[test]
fn byte_identity() {
    let (gpu, manifest) = (Gpu::new(), manifest());
    let ran = run_cases(&gpu, &manifest, "rust");
    println!("rust gate: launched {ran} cases through the generated launchers");
    assert_no_nvrtc();
}

/// Loading a cubin goes through cudarc's `load_module`, which is behind its `nvrtc` feature: with `dynamic-loading` the NVRTC library
/// must still not be opened, since nothing here compiles anything. (On Linux, where the libraries a process has mapped are listed.)
fn assert_no_nvrtc() {
    if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
        let nvrtc: Vec<&str> = maps.lines().filter(|l| l.contains("nvrtc")).collect();
        assert!(nvrtc.is_empty(), "libnvrtc is mapped in this process: {nvrtc:?}");
        let cuda = maps.lines().filter(|l| l.contains("libcuda")).count();
        assert!(cuda > 0, "the driver (libcuda) is expected to be mapped after a launch");
        println!("rust gate: libcuda is mapped, libnvrtc is not (after the launches)");
    }
}

/// The same cases, with the manifest and the cubins that the generated module embeds with `include_bytes!`.
#[test]
fn byte_identity_of_the_embedded_manifest() {
    let gpu = Gpu::new();
    let manifest = generated::load_embedded().expect("the embedded manifest verifies and is pinned");
    assert_eq!(manifest.digest(), generated::MANIFEST_DIGEST);
    assert_eq!(manifest.label(), "<embedded>");
    let ran = run_cases(&gpu, &manifest, "rust_embedded");
    println!("rust gate: launched {ran} cases with the embedded manifest");
}

/// A kernel that needs more than the default 48 KiB of dynamic shared memory is loaded with `cuFuncSetAttribute`: the gate has one
/// (`mm`, a `tl.dot` kernel), and its cases are compared with `replay` like the others.
#[test]
fn a_kernel_with_more_than_48_kib_of_dynamic_shared_memory_is_in_the_gate() {
    let (gpu, manifest) = (Gpu::new(), manifest());
    let arch = Arch::of_context(&gpu.ctx).unwrap();
    let entry = manifest.entry("mm", &arch.to_string()).expect("the gate's manifest has mm");
    assert!(
        entry.shared > 48 * 1024,
        "mm needs more than 48 KiB of dynamic shared memory for this to test the opt-in: {}",
        entry.shared
    );
    let _ = generated::Mm::load(&manifest, &gpu.ctx).expect("it loads: the opt-in is within the device's maximum");
    assert!(cases()["cases"].as_array().unwrap().iter().any(|c| c["launcher"] == "Mm"));
    drop(gpu);
    println!("rust gate: mm needs {} bytes of dynamic shared memory", entry.shared);
}

fn kind(e: &Error) -> PreconditionKind {
    e.precondition_kind().unwrap_or_else(|| panic!("not a precondition error: {e:?}"))
}

#[test]
fn refusals() {
    let (gpu, manifest) = (Gpu::new(), manifest());
    let s: &CudaStream = &gpu.stream;
    let mut refused = 0;
    let mut check = |what: &str, ok: bool| {
        assert!(ok, "{what}");
        refused += 1;
        println!("rust gate: refused: {what}");
    };

    // a tensor whose shape needs more bytes than its region has is refused when it is made, whatever the launch would have done
    let small = gpu.zeros(64);
    // SAFETY: the region is the buffer's own, and it outlives the (refused) constructions.
    let region = unsafe { gpu.region(&small, small.nbytes()) };
    let e = TensorRef::<F32>::new(region, &[2_200_000, 1000]).unwrap_err();
    check(
        "a tensor larger than its region (raw pointer)",
        matches!(&e, Error::Launch(m) if m.contains("needs 8800000000 bytes")),
    );
    let e = TensorRef::<F32>::from_cuda(&small.slice, &[17]).unwrap_err();
    check(
        "a tensor larger than its slice (cudarc)",
        matches!(&e, Error::Launch(m) if m.contains("needs 68 bytes, but its region has 64")),
    );
    check(
        "a sub-region that does not fit",
        region.offset(65).is_err() && region.slice(32, 33).is_err() && region.slice(32, 32).is_ok(),
    );

    // a misaligned pointer (8 bytes past a 16-byte boundary): nothing is launched, the output stays untouched
    let x = gpu.upload(&vec![1u8; 2 * 4096 + 64]);
    let mut y = gpu.upload(&vec![0xAAu8; 2 * 4096]).shaped(&[4096]);
    // SAFETY: the region is the buffer's own, shifted inside it; both outlive the call.
    let shifted = unsafe { gpu.region(&x, x.nbytes()) }.offset(8).unwrap();
    let xs = [4096usize];
    let misaligned = TensorRef::<Bf16>::new(shifted, &xs).unwrap();
    let launcher = generated::GeluErf::load(&manifest, &gpu.ctx).unwrap();
    let e = launcher.launch(s, misaligned, m::<Bf16>(&mut y).unwrap()).unwrap_err();
    check(
        "a misaligned pointer",
        kind(&e) == PreconditionKind::PointerAlignment && e.to_string().contains("not aligned to 16 bytes"),
    );
    check("... and nothing was written", gpu.download(&y).iter().all(|b| *b == 0xAA));

    // a size the binary has baked in
    let mut rows = gpu.zeros(3 * 2000 * 2).shaped(&[3, 2000]);
    let w = gpu.zeros(2000 * 2).shaped(&[2000]);
    let x2 = gpu.zeros(3 * 2000 * 2).shaped(&[3, 2000]);
    let e = generated::Rmsnorm::load(&manifest, &gpu.ctx)
        .unwrap()
        .launch(s, r::<Bf16>(&x2).unwrap(), r::<Bf16>(&w).unwrap(), m::<Bf16>(&mut rows).unwrap())
        .unwrap_err();
    check("a wrong static size", kind(&e) == PreconditionKind::StaticSize && e.to_string().contains("requires 2048"));

    // a bucketed dimension that is not the bucket: there is no padding
    let (x, w, mut o) = (
        gpu.zeros(3 * 100 * 2).shaped(&[3, 100]),
        gpu.zeros(100 * 2).shaped(&[100]),
        gpu.zeros(3 * 100 * 2).shaped(&[3, 100]),
    );
    let e = generated::BucketedRmsDb96::load(&manifest, &gpu.ctx)
        .unwrap()
        .launch(s, r::<Bf16>(&x).unwrap(), r::<Bf16>(&w).unwrap(), m::<Bf16>(&mut o).unwrap())
        .unwrap_err();
    check("a wrong bucket", kind(&e) == PreconditionKind::Bucket && e.to_string().contains("no padding"));

    // more elements than a generated kernel's 32-bit offsets reach. A safe constructor cannot make this tensor over a small buffer,
    // so the test vouches falsely on purpose, with two regions far apart (they would be refused as an alias first): the call is
    // refused before the driver sees it, and not a byte past the buffers is touched.
    let big = [2_200_000usize, 1000];
    let a = gpu.zeros(64);
    // SAFETY: deliberately false lengths; the launch below is refused by `max_elements` before anything reaches the device.
    let (ra, rb) = unsafe {
        (
            DevicePtr::from_raw(a.slice.device_ptr(s).0, 8_800_000_000, 0),
            DevicePtr::from_raw((1 << 36) + a.slice.device_ptr(s).0, 8_800_000_000, 0),
        )
    };
    let e = generated::Softmax::load(&manifest, &gpu.ctx)
        .unwrap()
        .launch(s, TensorRef::<F32>::new(ra, &big).unwrap(), TensorMut::<F32>::new(rb, &big).unwrap())
        .unwrap_err();
    check("max_elements", kind(&e) == PreconditionKind::MaxElements);

    // a runtime dimension that is not a multiple of 16, and a written tensor that aliases another
    let a = gpu.zeros(64 * 2).shaped(&[60]);
    let mut b = gpu.zeros(64 * 2).shaped(&[60]);
    let e = generated::GeluErf::load(&manifest, &gpu.ctx)
        .unwrap()
        .launch(s, r::<Bf16>(&a).unwrap(), m::<Bf16>(&mut b).unwrap())
        .unwrap_err();
    check("divisible_by", kind(&e) == PreconditionKind::Divisible);
    let a = gpu.zeros(4096 * 4).shaped(&[4096]);
    // SAFETY: two tensors over one buffer is the point of the test (it is refused as an alias before any launch); the buffer outlives it.
    let (same, same_again) = unsafe { (gpu.unbound_region(&a, a.nbytes()), gpu.unbound_region(&a, a.nbytes())) };
    let sh = [4096usize];
    let e = generated::Axpy::load(&manifest, &gpu.ctx)
        .unwrap()
        .launch(s, TensorRef::<F32>::new(same, &sh).unwrap(), TensorMut::<F32>::new(same_again, &sh).unwrap())
        .unwrap_err();
    check("an in-place tensor that aliases another", kind(&e) == PreconditionKind::Alias);

    // a read-only tensor where the kernel writes: the raw `Kernel::launch` refuses it (the generated launchers cannot be given one)
    let (x, y) = (gpu.zeros(4096 * 2).shaped(&[4096]), gpu.zeros(4096 * 2).shaped(&[4096]));
    let gelu = generated::GeluErf::load(&manifest, &gpu.ctx).unwrap();
    let e = gelu
        .kernel()
        .launch(s, &mut [r::<Bf16>(&x).unwrap().into_arg(), r::<Bf16>(&y).unwrap().into_arg()])
        .unwrap_err();
    check(
        "a read-only tensor for a parameter that is written",
        kind(&e) == PreconditionKind::Access && e.to_string().contains("y_ptr"),
    );
    let x_arg = r::<Bf16>(&x).unwrap().into_arg();
    check("... and that tensor is read-only", x_arg.access() == Access::In);

    // a dimension that is not a multiple of what the binary assumes (mm: MmM is a multiple of 128, and so is the 2-D grid's x)
    let (a, b, mut c) = (
        gpu.zeros(100 * 64 * 2).shaped(&[100, 64]),
        gpu.zeros(64 * 128 * 2).shaped(&[64, 128]),
        gpu.zeros(100 * 128 * 4).shaped(&[100, 128]),
    );
    let e = generated::Mm::load(&manifest, &gpu.ctx)
        .unwrap()
        .launch(s, r::<F16>(&a).unwrap(), r::<F16>(&b).unwrap(), m::<F32>(&mut c).unwrap())
        .unwrap_err();
    check(
        "divisible_by of a dimension (MmM = 100)",
        kind(&e) == PreconditionKind::Divisible && e.to_string().contains("MmM"),
    );
    // tensors that disagree about a dimension (mm: the K of a is 64, the K of b is 128)
    let (a, b, mut c) = (
        gpu.zeros(128 * 64 * 2).shaped(&[128, 64]),
        gpu.zeros(128 * 128 * 2).shaped(&[128, 128]),
        gpu.zeros(128 * 128 * 4).shaped(&[128, 128]),
    );
    let e = generated::Mm::load(&manifest, &gpu.ctx)
        .unwrap()
        .launch(s, r::<F16>(&a).unwrap(), r::<F16>(&b).unwrap(), m::<F32>(&mut c).unwrap())
        .unwrap_err();
    check(
        "tensors that disagree about a dimension",
        kind(&e) == PreconditionKind::Shape && e.to_string().contains("MmK"),
    );

    // a tampered cubin: the manifest is intact, so a pin accepts it; the digest of the file does not
    let dir = gate_dir().join(cases()["manifest"].as_str().unwrap());
    let tmp = std::env::temp_dir().join(format!("tilefish-gate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    for f in std::fs::read_dir(&dir).unwrap() {
        let f = f.unwrap();
        std::fs::copy(f.path(), tmp.join(f.file_name())).unwrap();
    }
    let cubin = tmp.join(format!("softmax.{}.cubin", cases()["arch"].as_str().unwrap()));
    let mut bytes = std::fs::read(&cubin).unwrap();
    let before = sha256_hex(&bytes);
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&cubin, &bytes).unwrap();
    let e = Manifest::load(&tmp, Pin::Digest(generated::MANIFEST_DIGEST)).unwrap_err();
    check(
        "a tampered cubin",
        matches!(&e, Error::Manifest(m) if m.contains(&before) && m.contains("not the one that was built")),
    );
    let _ = std::fs::remove_dir_all(&tmp);

    // a wrong pin
    let e = Manifest::load(&dir, Pin::Digest(&"0".repeat(64))).unwrap_err();
    check("a wrong pin", matches!(e, Error::PinMismatch { .. }));

    // a manifest of another architecture only: sm_120 on this device
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("sm_120");
    let other = Manifest::load(&fixtures, Pin::TrustDirectory).unwrap();
    let here = Arch::of_context(&gpu.ctx).unwrap();
    assert_ne!(here, Arch::sm(120), "the refusal test needs a device that is not an sm_120");
    let e = other.kernel(&gpu.ctx, "softmax", &[], here).unwrap_err();
    check(
        "an sm_120-only manifest on this device",
        matches!(&e, Error::NoBinary(m) if m.contains(&format!("no binary for 'softmax' on {here}")) && m.contains("sm_120")),
    );
    // ... and a launcher typed from one manifest refuses another, whatever it holds
    let e = generated::Softmax::load(&other, &gpu.ctx).unwrap_err();
    check("a launcher refuses a manifest it was not generated from", matches!(e, Error::PinMismatch { .. }));
    // a manifest that has the kernel, but not for this architecture, asked for explicitly
    let e = manifest.kernel(&gpu.ctx, "softmax", &[], Arch::sm(120)).unwrap_err();
    check("an explicit architecture the manifest does not have", matches!(e, Error::NoBinary(_)));
    // an architecture that is not the device's
    let e = other.kernel(&gpu.ctx, "softmax", &[], Arch::sm(120)).unwrap_err();
    check(
        "an architecture that is not the current device's",
        matches!(&e, Error::Unsupported(m) if m.contains("of the context")),
    );
    println!("rust gate: {refused} refusals checked");
}

/// Launch inside a CUDA graph capture, and run the graph: with the safe `launch` on the cudarc stream and with `unsafe launch_raw`
/// on its raw handle. The tensors are regions made before the capture: cudarc's event tracking of its own slices is not meant for
/// capture (it records and waits on events), and a region has none.
#[test]
fn graph_capture() {
    let (gpu, manifest, cases) = (Gpu::new(), manifest(), cases());
    // the first gelu_erf case of the gate: its inputs, and the bytes Python's replay computed
    let case = cases["cases"].as_array().unwrap().iter().find(|c| c["launcher"] == "GeluErf").expect("a GeluErf case");
    let mut tensors = tensors_of(&gpu, case).into_iter();
    let ((x_name, x), (y_name, mut y)) = (tensors.next().unwrap(), tensors.next().unwrap());
    assert_eq!((x_name.as_str(), y_name.as_str()), ("x_ptr", "y_ptr"));
    let expect = read(&gate_dir().join(case["tensors"][1]["expect"].as_str().unwrap()));
    let launcher = generated::GeluErf::load(&manifest, &gpu.ctx).unwrap(); // loads the module: not allowed during capture
    gpu.stream.synchronize().unwrap();
    let stream = gpu.stream.cu_stream();

    for raw in [false, true] {
        // the tensors are made before the capture: cudarc's `device_ptr` (which gives the address of a slice) waits on the slice's
        // events, which a capturing stream may not do
        // SAFETY: the buffers outlive the graph's launches (they are used until the end of this function), and nothing else
        // uses them while the graph is captured or run.
        let (xr, yr) = unsafe { (gpu.region(&x, x.nbytes()), gpu.region(&y, y.nbytes())) };
        let (xt, yt) = (TensorRef::<Bf16>::new(xr, &x.shape).unwrap(), TensorMut::<Bf16>::new(yr, &y.shape).unwrap());
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        // SAFETY (the driver calls below): `stream` is a live stream of the current context; the graph handles come from the driver.
        unsafe {
            let begin =
                sys::cuStreamBeginCapture_v2(stream, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            assert_eq!(begin, sys::CUresult::CUDA_SUCCESS, "cuStreamBeginCapture");
        }
        let launched = if raw {
            // SAFETY: `stream` is the handle of a stream of the kernel's context; the buffers outlive the graph.
            unsafe { launcher.launch_raw(stream, xt, yt) }
        } else {
            launcher.launch(&gpu.stream, xt, yt)
        };
        unsafe {
            let end = sys::cuStreamEndCapture(stream, &mut graph);
            assert_eq!(end, sys::CUresult::CUDA_SUCCESS, "cuStreamEndCapture");
        }
        launched.expect("the launch is captured");
        assert!(gpu.download(&y).iter().all(|b| *b == 0), "a captured launch runs nothing: the output is untouched");

        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        unsafe {
            assert_eq!(
                sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0),
                sys::CUresult::CUDA_SUCCESS,
                "cuGraphInstantiate"
            );
        }
        for run in 0..3 {
            // the graph computes the output again each time: clear it first
            gpu.stream.memset_zeros(&mut y.slice).unwrap();
            unsafe {
                assert_eq!(sys::cuGraphLaunch(exec, stream), sys::CUresult::CUDA_SUCCESS, "cuGraphLaunch");
            }
            let got = gpu.download(&y);
            assert_eq!(
                got, expect,
                "launch_raw {raw}, run {run}: the graph's output is Python's replay, byte for byte"
            );
        }
        unsafe {
            let _ = sys::cuGraphExecDestroy(exec);
            let _ = sys::cuGraphDestroy(graph);
        }
        gpu.stream.memset_zeros(&mut y.slice).unwrap();
    }
    println!(
        "rust gate: a launch captured in a CUDA graph (launch, and launch_raw) ran 3 times each and matched Python's replay"
    );
}

static RELEASE: AtomicBool = AtomicBool::new(false);

/// A host function for a stream: it returns when `RELEASE` is set, so everything queued behind it on the stream waits.
unsafe extern "C" fn hold_until_released(_: *mut c_void) {
    while !RELEASE.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// What the tests of launches on other streams share. Everything is allocated on `s1` and freed there (as cudarc frees a slice on the
/// stream it was allocated on, with `cuMemFreeAsync`); the launch under test is on `s2`, which is held back by a host function as a busy
/// stream would be; `s3` and `s4` are streams with other work. The launch is `y = gelu(x)` over bf16 1.5s (`ones`), and one of the two is freed
/// while it is queued and allocated again: the new memory for `x` is filled with 3.0s (`threes`), so a kernel that reads it by mistake
/// writes `gelu(3.0)` into `y`, which is compared with what it must be; the new memory for `y` is filled with `ones`, so a kernel that
/// writes it by mistake changes it. A kernel that read zeros, or wrote zeros over zeros, would hide the hazard.
struct Race {
    gpu: Gpu,
    s1: Arc<CudaStream>,
    s2: Arc<CudaStream>,
    s3: Arc<CudaStream>,
    s4: Arc<CudaStream>,
    launcher: generated::GeluErf,
    ones: Vec<u8>,
    threes: Vec<u8>,
    /// `gelu(ones)`, computed on one stream with nothing else going on.
    expect: Vec<u8>,
    shape: [usize; 1],
}

/// Which of the two slices of a launch is freed (and allocated again) while the launch is queued.
#[derive(Clone, Copy, Debug)]
enum Free {
    /// The input: the kernel must still read 1.5s.
    Input,
    /// The output: the kernel must not write into what is allocated in its place.
    Output,
}

impl Race {
    const N: usize = 4096;

    fn new() -> Self {
        let (gpu, manifest) = (Gpu::new(), manifest());
        let launcher = generated::GeluErf::load(&manifest, &gpu.ctx).unwrap();
        let (s1, s2, s3, s4) = (
            gpu.stream.clone(),
            gpu.ctx.new_stream().unwrap(),
            gpu.ctx.new_stream().unwrap(),
            gpu.ctx.new_stream().unwrap(),
        );
        let fill = |bits: u16| (0..Self::N).flat_map(move |_| bits.to_le_bytes()).collect::<Vec<u8>>();
        let (ones, threes) = (fill(0x3FC0), fill(0x4040)); // bf16 1.5 and 3.0
        let mut race = Self { gpu, s1, s2, s3, s4, launcher, ones, threes, expect: Vec::new(), shape: [Self::N] };
        let (x, mut y) = race.pair();
        race.gelu(&race.s1, &x, &mut y).unwrap();
        race.s1.synchronize().unwrap();
        race.expect = race.s1.clone_dtoh(&y).unwrap();
        assert!(race.expect != race.ones && race.expect != race.threes);
        race
    }

    /// A slice of `ones`, and a slice of zeros, on `s1`.
    fn pair(&self) -> (CudaSlice<u8>, CudaSlice<u8>) {
        let (x, y) = (self.s1.clone_htod(&self.ones).unwrap(), self.s1.alloc_zeros::<u8>(Self::N * 2).unwrap());
        self.s1.synchronize().unwrap();
        (x, y)
    }

    /// Hold `s2` back: nothing queued on it from now runs until the returned thread has released it (after 150 ms).
    fn hold(&self) -> std::thread::JoinHandle<()> {
        RELEASE.store(false, Ordering::Release);
        // SAFETY: the function only waits for an atomic and takes no argument.
        unsafe { result::stream::launch_host_function(self.s2.cu_stream(), hold_until_released, std::ptr::null_mut()) }
            .expect("a host function on the second stream");
        std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(150));
            RELEASE.store(true, Ordering::Release);
        })
    }

    /// `y = gelu(x)` on `stream`, through the safe launcher, with tensors made of the slices.
    fn gelu(&self, stream: &CudaStream, x: &CudaSlice<u8>, y: &mut CudaSlice<u8>) -> Result<(), Error> {
        self.gelu_of(stream, x, y, &self.shape)
    }

    fn gelu_of(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<u8>,
        y: &mut CudaSlice<u8>,
        shape: &[usize],
    ) -> Result<(), Error> {
        self.launcher.launch(stream, TensorRef::<Bf16>::from_cuda(x, shape)?, TensorMut::<Bf16>::from_cuda(y, shape)?)
    }

    /// Free one of `x` and `y`, and `others`, on `s1` while the held-back launch has not run, allocate memory again in their place,
    /// release the stream, and say whether what the kernel did was not what it must have done.
    fn freed_and_reused(
        &self,
        free: Free,
        x: CudaSlice<u8>,
        y: CudaSlice<u8>,
        others: Vec<CudaSlice<u8>>,
        releaser: std::thread::JoinHandle<()>,
    ) -> bool {
        match free {
            Free::Input => {
                drop((x, others));
                let again = (0..3).map(|_| self.s1.clone_htod(&self.threes).unwrap()).collect::<Vec<_>>();
                releaser.join().unwrap();
                self.gpu.ctx.synchronize().unwrap();
                drop(again);
                self.s1.clone_dtoh(&y).unwrap() != self.expect // the kernel read what x held, not what was put there
            }
            Free::Output => {
                drop((y, others));
                let again = (0..3).map(|_| self.s1.clone_htod(&self.ones).unwrap()).collect::<Vec<_>>();
                releaser.join().unwrap();
                self.gpu.ctx.synchronize().unwrap();
                again.iter().any(|z| self.s1.clone_dtoh(z).unwrap() != self.ones) // the kernel wrote into what was put there
            }
        }
    }

    /// Run `round` 20 times with the input freed and 20 times with the output freed; none may have gone wrong.
    fn rounds(&self, what: &str, round: impl Fn(&Race, Free) -> bool) {
        let counts = [Free::Input, Free::Output].map(|free| (0..20).filter(|_| round(self, free)).count());
        println!(
            "rust gate: {what}: {} of 20 rounds wrong with the input freed, {} of 20 with the output freed",
            counts[0], counts[1]
        );
        assert_eq!(counts, [0, 0], "{what}: a kernel read or wrote memory that had been freed and allocated again");
    }
}

/// cudarc frees a `CudaSlice` on the stream it was allocated on, and orders the free after the launches that used it by the events it
/// records. A launch of slices allocated on `s1` and launched on `s2` (held back): the slices are freed on `s1` while the kernel has not
/// run, and allocated again. The launch makes `s1` wait for the kernel, so the new memory is not touched.
#[test]
fn a_launch_on_another_stream_is_ordered_against_the_free_of_its_slices() {
    let race = Race::new();
    race.rounds("a launch on a held-back second stream, its slices freed on the first", |r, free| {
        let (x, mut y) = r.pair();
        let releaser = r.hold();
        r.gelu(&r.s2, &x, &mut y).unwrap();
        r.freed_and_reused(free, x, y, vec![], releaser)
    });
}

/// (1) Making a tensor over the slice, and throwing it away, while the launch is queued. Making a tensor used to record on the slice's
/// single `read` event, replacing the record of the queued launch: the free then waited for the wrong point.
#[test]
fn a_tensor_made_and_thrown_away_while_a_launch_is_queued_changes_nothing() {
    let race = Race::new();
    race.rounds("a tensor over the slices, made and discarded while the launch is queued", |r, free| {
        let (x, mut y) = r.pair();
        let releaser = r.hold();
        r.gelu(&r.s2, &x, &mut y).unwrap();
        // (a tensor over `y`, which the queued launch writes, would make `s1` wait for that launch, as cudarc's own `device_ptr` does
        // when it gives the address of a slice that has a write pending: harmless, but it would hide what this is about)
        for _ in 0..3 {
            let _ = TensorRef::<Bf16>::from_cuda(&x, &r.shape).unwrap();
        }
        r.freed_and_reused(free, x, y, vec![], releaser)
    });
}

/// (2) A second launch on the same slices that a precondition refuses (a size that is not a multiple of 16), while the first is queued.
#[test]
fn a_second_launch_that_is_refused_while_a_launch_is_queued_changes_nothing() {
    let race = Race::new();
    race.rounds("a refused second launch while the first is queued", |r, free| {
        let (x, mut y) = r.pair();
        let mut y2 = r.s1.alloc_zeros::<u8>(Race::N * 2).unwrap();
        let releaser = r.hold();
        r.gelu(&r.s2, &x, &mut y).unwrap();
        // the refused launch reads the same x, and writes another slice (not `y`, which the first launch writes: see above)
        let e = r.gelu_of(&r.s2, &x, &mut y2, &[60]).unwrap_err();
        assert_eq!(e.precondition_kind(), Some(PreconditionKind::Divisible), "{e}");
        r.freed_and_reused(free, x, y, vec![y2], releaser)
    });
}

/// (3) A second reader of the same slice on a third stream, which runs at once: cudarc keeps one `read` event per slice, so its record
/// replaces the queued launch's, and a free that waited only for that one would not wait for the first.
#[test]
fn a_second_reader_on_a_third_stream_does_not_hide_the_first() {
    let race = Race::new();
    race.rounds("a second reader of the slice on a third stream", |r, free| {
        let (x, mut y) = r.pair();
        // the second reader's output is on its own stream: a kernel that writes a slice of another stream waits for what is on that
        // stream (see (5)), which would make this launch wait for the first
        let mut y3 = r.s3.alloc_zeros::<u8>(Race::N * 2).unwrap();
        r.s3.synchronize().unwrap();
        let releaser = r.hold();
        r.gelu(&r.s2, &x, &mut y).unwrap(); // queued behind the held-back stream
        r.gelu(&r.s3, &x, &mut y3).unwrap(); // runs at once, and is recorded on x after the first
        r.s3.synchronize().unwrap();
        r.freed_and_reused(free, x, y, vec![y3], releaser)
    });
}

/// (4) A writer of the slice on another stream while a reader is queued: building a `TensorMut` over the slice used to replace the
/// record of the queued reader, so the writer did not wait for it and the reader read what the writer wrote.
#[test]
fn a_writer_on_another_stream_waits_for_the_reader_that_is_queued() {
    let race = Race::new();
    let expect = race.expect.clone();
    let wrong = (0..20)
        .filter(|_| {
            let (mut x, mut y) = race.pair();
            let w = race.s1.clone_htod(&race.threes).unwrap();
            let releaser = race.hold();
            race.gelu(&race.s2, &x, &mut y).unwrap(); // the reader of x, queued
            race.gelu(&race.s3, &w, &mut x).unwrap(); // the writer of x, on another stream: it must wait for the reader
            releaser.join().unwrap();
            race.gpu.ctx.synchronize().unwrap();
            race.s1.clone_dtoh(&y).unwrap() != expect // the reader read what the writer wrote
        })
        .count();
    println!(
        "rust gate: a writer on a third stream while the reader on the second is queued: {wrong} of 20 rounds wrong"
    );
    assert_eq!(wrong, 0, "a writer on another stream did not wait for the reader that was queued");
}

/// (5) Two readers of the slice on two streams, then a writer on a third. cudarc's writer waits for the slice's last `read` record, which
/// is the second reader's (done by the time the writer is launched): it did not wait for the first reader, still queued on the
/// held-back stream, and the reader then read what the writer wrote. The first reader's output is on `s1` and the second's on its own
/// stream, so that the second runs at once.
#[test]
fn a_writer_on_a_third_stream_waits_for_both_readers_and_not_only_the_last() {
    let race = Race::new();
    // what the writer writes into `x`: gelu of the 3.0s, computed on one stream with nothing else going on
    let written = {
        let w_in = race.s1.clone_htod(&race.threes).unwrap();
        let mut w_out = race.s1.alloc_zeros::<u8>(Race::N * 2).unwrap();
        race.gelu(&race.s1, &w_in, &mut w_out).unwrap();
        race.s1.synchronize().unwrap();
        race.s1.clone_dtoh(&w_out).unwrap()
    };
    let wrong = (0..20)
        .filter(|_| {
            let (mut x, mut y1) = race.pair();
            let mut y2 = race.s3.alloc_zeros::<u8>(Race::N * 2).unwrap();
            let w = race.s1.clone_htod(&race.threes).unwrap();
            race.s1.synchronize().unwrap();
            race.s3.synchronize().unwrap();
            let releaser = race.hold();
            race.gelu(&race.s2, &x, &mut y1).unwrap(); // reader 1 of x, queued
            race.gelu(&race.s3, &x, &mut y2).unwrap(); // reader 2 of x, runs at once: cudarc's `read` record of x is now this one
            race.s3.synchronize().unwrap();
            race.gelu(&race.s4, &w, &mut x).unwrap(); // the writer of x: it must wait for reader 1 as well
            releaser.join().unwrap();
            race.gpu.ctx.synchronize().unwrap();
            let read_what_was_written = race.s1.clone_dtoh(&y1).unwrap() != race.expect;
            let writer_did_not_write = race.s1.clone_dtoh(&x).unwrap() != written;
            read_what_was_written || writer_did_not_write
        })
        .count();
    println!("rust gate: a writer on a third stream after two readers on two streams: {wrong} of 20 rounds wrong");
    assert_eq!(wrong, 0, "a writer did not wait for a reader that was not the last one recorded on the slice");
}

/// A safe `launch` (or `launch_raw`) of tensors made of cudarc slices on a stream that is capturing is an error, in a context that
/// tracks events (it has made a stream of its own): nothing is captured and nothing is launched, and the stream works afterwards.
/// The alternative is a graph that holds the slices' addresses and nothing that keeps them alive.
#[test]
fn a_safe_launch_of_slices_on_a_capturing_stream_is_an_error_and_not_a_graph() {
    let (gpu, manifest, cases) = (Gpu::new(), manifest(), cases());
    assert!(gpu.ctx.is_in_multi_stream_mode(), "this context has made a stream of its own: cudarc tracks events");
    let case = cases["cases"].as_array().unwrap().iter().find(|c| c["launcher"] == "GeluErf").expect("a GeluErf case");
    let expect = read(&gate_dir().join(case["tensors"][1]["expect"].as_str().unwrap()));
    let mut tensors = tensors_of(&gpu, case).into_iter();
    let ((_, x), (_, mut y)) = (tensors.next().unwrap(), tensors.next().unwrap());
    let mut again = tensors_of(&gpu, case).into_iter();
    let ((_, x2), (_, mut y2)) = (again.next().unwrap(), again.next().unwrap());
    let launcher = generated::GeluErf::load(&manifest, &gpu.ctx).unwrap();
    gpu.stream.synchronize().unwrap();

    // the tensors are made before the capture begins: making one is not what is under test
    let (xt, yt) = (r::<Bf16>(&x).unwrap(), m::<Bf16>(&mut y).unwrap());
    let (xt2, yt2) = (r::<Bf16>(&x2).unwrap(), m::<Bf16>(&mut y2).unwrap());
    gpu.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).unwrap();
    let refused = launcher.launch(&gpu.stream, xt, yt);
    // SAFETY: the stream is the kernel's, and the call is refused before it does anything (slices on a capturing stream).
    let refused_raw = unsafe { launcher.launch_raw(gpu.stream.cu_stream(), xt2, yt2) };
    let ended = gpu.stream.end_capture(sys::CUgraphInstantiate_flags(0));
    let e = refused.expect_err("the launch of slices was captured");
    assert!(matches!(&e, Error::Launch(why) if why.contains("capturing") && why.contains("launch_raw")), "{e:?}");
    let e_raw = refused_raw.expect_err("the raw launch of slices was captured");
    assert!(matches!(&e_raw, Error::Launch(why) if why.contains("capturing")), "{e_raw:?}");
    if let Ok(Some(graph)) = ended {
        graph.launch().unwrap(); // whatever was captured (nothing) runs
    }

    // nothing was captured, nothing was launched, and the stream works
    assert!(gpu.download(&y).iter().all(|b| *b == 0) && gpu.download(&y2).iter().all(|b| *b == 0));
    launcher.launch(&gpu.stream, r::<Bf16>(&x).unwrap(), m::<Bf16>(&mut y).unwrap()).unwrap();
    assert_eq!(gpu.download(&y), expect, "and the next launch on the stream is Python's replay, byte for byte");
    println!("rust gate: a safe launch of slices on a capturing stream: {e}");
}

/// The same in a context that never made a stream of its own and uses `ctx.per_thread_stream()`: cudarc does not track events there, so
/// no wait of its can make the driver refuse the launch, and the kernel used to be captured with the slices' addresses. `CudaGraph::launch`
/// is safe, so a replay after the slices are freed wrote into memory that something else had been given. (Since the review of M3.2 the
/// per-thread default stream is refused by every safe launch, captured or not: both launches below are refused, for that reason.)
#[test]
fn a_safe_launch_of_slices_on_a_capturing_per_thread_stream_is_an_error_and_not_a_graph() {
    let manifest = manifest();
    let ctx = CudaContext::new(0).expect("a CUDA context on device 0");
    let s = ctx.per_thread_stream();
    assert!(!ctx.is_in_multi_stream_mode(), "this context never made a stream of its own: cudarc tracks nothing in it");
    let launcher = generated::GeluErf::load(&manifest, &ctx).unwrap();
    let n = Race::N;
    let ones = (0..n).flat_map(|_| 0x3FC0u16.to_le_bytes()).collect::<Vec<u8>>(); // bf16 1.5
    let shape = [n];
    let x = s.clone_htod(&ones).unwrap();
    let mut y = s.alloc_zeros::<u8>(n * 2).unwrap();
    s.synchronize().unwrap();

    // outside a capture the launch is refused too: the per-thread default stream is refused by every safe launch
    let (xt, yt) =
        (TensorRef::<Bf16>::from_cuda(&x, &shape).unwrap(), TensorMut::<Bf16>::from_cuda(&mut y, &shape).unwrap());
    let e = launcher.launch(&s, xt, yt).expect_err("a launch on the per-thread default stream");
    assert!(e.to_string().contains("per-thread default stream"), "{e}");

    let (xt, yt) =
        (TensorRef::<Bf16>::from_cuda(&x, &shape).unwrap(), TensorMut::<Bf16>::from_cuda(&mut y, &shape).unwrap());
    s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).unwrap();
    let refused = launcher.launch(&s, xt, yt);
    let ended = s.end_capture(sys::CUgraphInstantiate_flags(0));

    // free the slices, allocate memory in their place (filled with values the kernel would change), and replay what was captured
    drop((x, y));
    let reused = (0..4).map(|_| s.clone_htod(&ones).unwrap()).collect::<Vec<_>>();
    s.synchronize().unwrap();
    if let Ok(Some(graph)) = ended {
        graph.launch().unwrap();
    }
    s.synchronize().unwrap();
    let changed = reused.iter().filter(|z| s.clone_dtoh(*z).unwrap() != ones).count();
    println!(
        "rust gate: a safe launch of slices on a capturing per-thread stream: {refused:?}; the replay changed {changed} of 4 reused buffers"
    );
    assert_eq!(
        changed, 0,
        "a graph captured from a launch of slices wrote into memory allocated after they were freed"
    );
    let e = refused.expect_err("the launch of slices was captured");
    assert!(matches!(&e, Error::Launch(why) if why.contains("per-thread default stream")), "{e:?}");
}

/// The per-thread default stream is refused by every safe launch (the review of M3.2): its handle, `CU_STREAM_PER_THREAD`, is 0x2 in every
/// thread and names another stream in each, so a slice allocated on thread A's per-thread stream and launched on thread B's looked like a
/// slice of the launch stream, neither the wait of A's stream for the kernel nor the wait of the writer for A's stream was made, and the slice
/// was freed (and its memory reused) under the kernel: the reviewer's repro wrote into reallocated memory in 10 of 10 rounds. Here it is
/// refused and writes nothing, and so is every other use of the handle: as the launch stream on one thread, as the owner of a slice launched
/// on a stream of `new_stream`, for a GEMM. The legacy default stream (`ctx.default_stream()`) launches.
#[test]
fn the_per_thread_default_stream_is_refused() {
    let (gpu, manifest) = (Gpu::new(), manifest());
    let launcher = generated::GeluErf::load(&manifest, &gpu.ctx).unwrap();
    let n = Race::N;
    let ones = (0..n).flat_map(|_| 0x3FC0u16.to_le_bytes()).collect::<Vec<u8>>(); // bf16 1.5
    let shape = [n];
    let per_thread = |e: Result<(), Error>, what: &str| {
        let e = e.expect_err(what);
        assert!(matches!(&e, Error::Launch(why) if why.contains("per-thread default stream")), "{what}: {e:?}");
    };
    // -- the repro: the slices are thread A's (allocated on its per-thread stream), the launch is thread B's, on its per-thread stream
    let a = gpu.ctx.per_thread_stream();
    let x = a.clone_htod(&ones).unwrap();
    let mut y = a.alloc_zeros::<u8>(n * 2).unwrap();
    a.synchronize().unwrap();
    let launched = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let b = gpu.ctx.per_thread_stream();
                launcher.launch(
                    &b,
                    TensorRef::<Bf16>::from_cuda(&x, &shape)?,
                    TensorMut::<Bf16>::from_cuda(&mut y, &shape)?,
                )
            })
            .join()
            .unwrap()
    });
    per_thread(launched, "a launch on thread B's per-thread stream of thread A's slices");
    // -- on one thread: the per-thread stream as the launch stream, and as the owner of slices launched elsewhere
    let (xs, mut ys) = (gpu.stream.clone_htod(&ones).unwrap(), gpu.stream.alloc_zeros::<u8>(n * 2).unwrap());
    let refused = launcher.launch(
        &a,
        TensorRef::<Bf16>::from_cuda(&xs, &shape).unwrap(),
        TensorMut::<Bf16>::from_cuda(&mut ys, &shape).unwrap(),
    );
    per_thread(refused, "a launch on the per-thread stream of slices of a stream of new_stream");
    let refused = launcher.launch(
        &gpu.stream,
        TensorRef::<Bf16>::from_cuda(&x, &shape).unwrap(),
        TensorMut::<Bf16>::from_cuda(&mut y, &shape).unwrap(),
    );
    per_thread(refused, "a launch on a stream of new_stream of slices of the per-thread stream");
    gpu.ctx.synchronize().unwrap();
    assert!(a.clone_dtoh(&y).unwrap().iter().all(|b| *b == 0), "a refused launch wrote its output");
    assert!(gpu.stream.clone_dtoh(&ys).unwrap().iter().all(|b| *b == 0), "a refused launch wrote its output");
    // -- a GEMM (the same ordering, `enqueue_gemm`)
    let arch = Arch::of_context(&gpu.ctx).unwrap();
    let cases = cases();
    let case =
        cases_where(&cases, |c| is_gemm(c) && tensors_json(c).len() == 3).into_iter().next().expect("a GEMM case");
    let gemm = manifest.gemm(&gpu.ctx, case["stem"].as_str().unwrap(), &[], arch).unwrap();
    let mut workspace = gemm.alloc_workspace(&gpu.stream).unwrap();
    let mut tensors = tensors_of(&gpu, &case);
    let ws_shape = [workspace.len()];
    let mut args: Vec<TensorArg<'_>> = Vec::new();
    let last = tensors.len() - 1;
    for (i, (_, buf)) in tensors.iter_mut().enumerate() {
        args.push(if i == last { m::<Bf16>(buf).unwrap().into_arg() } else { r::<Bf16>(buf).unwrap().into_arg() });
    }
    args.push(TensorMut::<U8>::from_cuda(&mut workspace, &ws_shape).unwrap().into_arg());
    per_thread(gemm.launch(&a, &mut args), "a GEMM on the per-thread stream");
    drop(args);
    // -- the legacy default stream is one stream per context: it launches
    let legacy = gpu.ctx.default_stream();
    launcher
        .launch(
            &legacy,
            TensorRef::<Bf16>::from_cuda(&xs, &shape).unwrap(),
            TensorMut::<Bf16>::from_cuda(&mut ys, &shape).unwrap(),
        )
        .expect("the legacy default stream launches");
    gpu.ctx.synchronize().unwrap();
    assert!(gpu.stream.clone_dtoh(&ys).unwrap() != vec![0u8; n * 2]);
    println!(
        "rust gate: the per-thread default stream is refused (across threads, as the launch stream, as a slice's owner, for a GEMM)"
    );
}

/// `launch_raw` makes the kernel's context current for the call, and puts back what was current: nothing, on a thread that had no
/// context; and another device's context, when there is one.
#[test]
fn launch_raw_puts_back_the_current_context() {
    let (gpu, manifest) = (Gpu::new(), manifest());
    let launcher = generated::GeluErf::load(&manifest, &gpu.ctx).unwrap();
    let case = cases()["cases"].as_array().unwrap().iter().find(|c| c["launcher"] == "GeluErf").cloned().unwrap();
    let mut tensors = tensors_of(&gpu, &case).into_iter();
    let ((_, x), (_, y)) = (tensors.next().unwrap(), tensors.next().unwrap());
    let expect = read(&gate_dir().join(case["tensors"][1]["expect"].as_str().unwrap()));
    let stream = gpu.stream.cu_stream() as usize;
    let devices = CudaContext::device_count().unwrap();

    // the regions are made here, on the thread that has the context: `device_ptr` binds it, and the thread below must start with none
    // SAFETY: the buffers outlive the launches, which are waited for below; nothing else uses them meanwhile.
    let (xr, yr) = unsafe { (gpu.region(&x, x.nbytes()), gpu.region(&y, y.nbytes())) };
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                assert!(result::ctx::get_current().unwrap().is_none(), "a new thread has no current context");
                let args = |xr, yr| {
                    [
                        TensorRef::<Bf16>::new(xr, &x.shape).unwrap().into_arg(),
                        TensorMut::<Bf16>::new(yr, &y.shape).unwrap().into_arg(),
                    ]
                };
                // SAFETY: the stream is the kernel's; the memory is what the regions say, and stays until the join.
                unsafe { launcher.kernel().launch_raw(stream as sys::CUstream, &args(xr, yr)) }.unwrap();
                assert!(
                    result::ctx::get_current().unwrap().is_none(),
                    "nothing was current before, and nothing is after"
                );

                // a call that is refused (a read-only tensor where the kernel writes) leaves the current context alone as well
                let read_only = [
                    TensorRef::<Bf16>::new(xr, &x.shape).unwrap().into_arg(),
                    TensorRef::<Bf16>::new(yr, &y.shape).unwrap().into_arg(),
                ];
                // SAFETY: as above (nothing is launched: the call is refused first).
                let e = unsafe { launcher.kernel().launch_raw(stream as sys::CUstream, &read_only) }.unwrap_err();
                assert_eq!(e.precondition_kind(), Some(PreconditionKind::Access));
                assert!(result::ctx::get_current().unwrap().is_none());

                if devices > 1 {
                    let other = CudaContext::new(1).unwrap(); // makes device 1's context current on this thread
                    let before = result::ctx::get_current().unwrap();
                    assert_eq!(before, Some(other.cu_ctx()));
                    // SAFETY: as above.
                    unsafe { launcher.kernel().launch_raw(stream as sys::CUstream, &args(xr, yr)) }.unwrap();
                    assert_eq!(
                        result::ctx::get_current().unwrap(),
                        before,
                        "the other device's context is current again"
                    );
                }
                // the kernel's own context current: it stays
                gpu.ctx.bind_to_thread().unwrap();
                let own = result::ctx::get_current().unwrap();
                // SAFETY: as above.
                unsafe { launcher.kernel().launch_raw(stream as sys::CUstream, &args(xr, yr)) }.unwrap();
                assert_eq!(result::ctx::get_current().unwrap(), own);
            })
            .join()
            .unwrap();
    });
    assert_eq!(gpu.download(&y), expect, "and the launches computed what Python's replay did");
    println!("rust gate: launch_raw restores the current context ({devices} device(s))");
}

// -- manifest schema 3: views, device scalars, prebuilt binaries and the library section ---------------------------------------------------

fn cases_where(cases: &Value, keep: impl Fn(&Value) -> bool) -> Vec<Value> {
    cases["cases"].as_array().unwrap().iter().filter(|c| keep(c)).cloned().collect()
}

fn tensors_json(case: &Value) -> &Vec<Value> {
    case["tensors"].as_array().unwrap()
}

fn has_view(case: &Value) -> bool {
    tensors_json(case).iter().any(|t| t.get("strides").is_some())
}

fn has_device_scalar(case: &Value) -> bool {
    tensors_json(case).iter().any(|t| t["shape"].as_array().is_some_and(|s| s.is_empty()))
}

fn is_gemm(case: &Value) -> bool {
    case["kind"] == "gemm"
}

fn element_size(dtype: &str) -> u64 {
    match dtype {
        "bf16" | "f16" => 2,
        "f32" | "i32" => 4,
        "i64" => 8,
        _ => 1,
    }
}

/// What the gate holds, by the feature of schema 3 each case is about: the claim is that every one of them is launched by Rust and the bytes it
/// leaves are the bytes Python's replay left (that is `byte_identity`), so a gate that does not hold them says so here.
#[test]
fn the_gate_has_a_case_for_every_feature_of_schema_3() {
    let cases = cases();
    let all = cases_where(&cases, |_| true);
    let views = cases_where(&cases, has_view);
    let device_scalars = cases_where(&cases, has_device_scalar);
    let gemms = cases_where(&cases, is_gemm);
    let biased = cases_where(&cases, |c| is_gemm(c) && tensors_json(c).len() == 4);
    // an unaligned view: one whose first element is not on a 16-byte boundary
    let unaligned = cases_where(&cases, |c| {
        tensors_json(c).iter().any(|t| {
            t.get("offset")
                .is_some_and(|o| !(o.as_u64().unwrap() * element_size(t["dtype"].as_str().unwrap())).is_multiple_of(16))
        })
    });
    let ms: std::collections::BTreeSet<u64> =
        gemms.iter().map(|c| tensors_json(c)[0]["shape"][0].as_u64().unwrap()).collect();
    println!(
        "rust gate: {} cases: {} of views ({} of them starting off a 16-byte boundary), {} of device scalars, {} of GEMMs ({} with a bias; M in {:?})",
        all.len(),
        views.len(),
        unaligned.len(),
        device_scalars.len(),
        gemms.len(),
        biased.len(),
        ms
    );
    assert!(views.len() >= 30 && !unaligned.is_empty(), "the strided cases of the gate");
    assert!(device_scalars.len() >= 7, "the device scalar cases of the gate");
    assert!(
        !gemms.is_empty() && !biased.is_empty() && [32, 64, 320].iter().all(|m| ms.contains(m)),
        "the GEMM cases of the gate"
    );
    // the prebuilt binaries: PTX with fixed arguments of each type, and a cubin
    let stems: std::collections::BTreeSet<String> =
        all.iter().map(|c| c["stem"].as_str().unwrap().to_owned()).collect();
    for stem in ["gelu_ptx", "softmax_ptx", "axpy_ptx", "strided_copy", "axpy_cubin"] {
        assert!(stems.contains(stem), "the gate has no case of the prebuilt binary {stem}: {stems:?}");
    }
}

/// What M3.3 added is in the gate: one hand-written kernel compiled without and with FMA contraction (`enable_fp_fusion` of its
/// declaration, in the options of its entry), a kernel that reads three views with one stride pair (`StrideOf(("q", "k", "v"), axis)`),
/// given the q, k and v of one fused tensor and three tensors of their own, and a `[B]` mask made from a device scalar whose `B` only the
/// `Out` tensor determines; `byte_identity` holds their bytes to Python's.
#[test]
fn the_gate_has_a_case_for_every_feature_of_m3_3() {
    let cases = cases();
    let arch = cases["arch"].as_str().unwrap().to_owned();
    let stems: std::collections::BTreeSet<String> =
        cases_where(&cases, |_| true).iter().map(|c| c["stem"].as_str().unwrap().to_owned()).collect();
    let manifest = manifest();
    for (stem, fused) in [("fma_plain", false), ("fma_fused", true)] {
        assert!(stems.contains(stem), "the gate has no case of {stem}: {stems:?}");
        let entry = manifest.entry(stem, &arch).expect("the gate's manifest has the entry");
        assert_eq!(entry.enable_fp_fusion, Some(fused), "{stem}: the contraction its binary was compiled with");
    }
    for stem in ["qkv_sum", "qkv_sum16"] {
        let shared = cases_where(&cases, |c| {
            c["stem"] == stem
                && tensors_json(c)[..3].iter().all(|t| t["strides"] == serde_json::json!([768, 64, 1]))
                && tensors_json(c)[..3].iter().map(|t| t["offset"].as_u64().unwrap()).collect::<Vec<_>>()
                    == [0, 256, 512]
        });
        assert!(shared.len() >= 3, "{stem}: the gate has no case of the q, k and v of one fused tensor");
        let entry = manifest.entry(stem, &arch).expect("the gate's manifest has the entry");
        let carried = entry
            .params
            .iter()
            .filter_map(|p| match &p.kind {
                tilefish::ParamKind::Scalar { stride_of, .. } if !stride_of.is_empty() => Some(stride_of.len()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(carried, [3, 3], "{stem}: two stride scalars, each the stride of three views");
    }
    for (stem, sizes) in [("key_mask", &[8u64, 24][..]), ("key_mask_any", &[1, 130, 1000][..])] {
        let masks: std::collections::BTreeSet<u64> = cases_where(&cases, |c| c["stem"] == stem)
            .iter()
            .map(|c| tensors_json(c)[0]["shape"][0].as_u64().unwrap())
            .collect();
        assert!(
            sizes.iter().all(|b| masks.contains(b)),
            "{stem}: the gate has masks of {masks:?} keys, not every one of {sizes:?}"
        );
        // no tensor the kernel reads has B: the mask is the first tensor, the device scalar the second
        let entry = manifest.entry(stem, &arch).expect("the gate's manifest has the entry");
        let reads_b = entry.params.iter().any(
            |p| matches!(&p.kind, tilefish::ParamKind::Pointer { access: Access::In, shape, .. } if !shape.is_empty()),
        );
        assert!(!reads_b, "{stem}: the kernel reads no tensor that has B");
    }
}

/// A Gluon kernel (`@tilefish.extern(gluon=...)`, `python/tests/gluon_kernels.py`) is a `triton` entry like any other, whose source digest is of
/// the TTGIR Gluon's frontend builds (`tilefish.gluon-ir.v1`). The gate runs `row_dot` and `row_dot_fma` (a layout-pinned row reduction, without and
/// with FMA contraction) for each bucket of their columns, over the two halves of one tensor (two views read with one `i32` row stride) and over two
/// tensors of their own; `byte_identity` holds their bytes to Python's. A committed fixture has no Gluon kernel: with one as the manifest there is
/// nothing to check here.
#[test]
fn the_gate_has_a_gluon_kernel() {
    let cases = cases();
    if cases["manifest"] != "manifest" {
        eprintln!("the gate runs a committed fixture, which has no Gluon kernel");
        return;
    }
    let arch = cases["arch"].as_str().unwrap().to_owned();
    let manifest = manifest();
    for op in ["row_dot", "row_dot_fma"] {
        for c in [128u64, 256] {
            let stem = format!("{op}.Cb{c}");
            let entry = manifest.entry(&stem, &arch).unwrap_or_else(|| panic!("the gate's manifest has no {stem}"));
            assert_eq!(entry.kind, tilefish::EntryKind::Triton, "{stem}");
            assert_eq!(entry.source_digest_kind.as_deref(), Some("tilefish.gluon-ir.v1"), "{stem}");
            assert_eq!(entry.enable_fp_fusion, Some(op == "row_dot_fma"), "{stem}");
            match &entry.param("s_row").expect("the row stride").kind {
                tilefish::ParamKind::Scalar { ty, stride_of, .. } => {
                    assert_eq!(
                        (*ty, stride_of.len()),
                        (tilefish::ScalarType::I32, 2),
                        "{stem}: one i32 stride of two views"
                    )
                }
                other => panic!("{stem}: s_row is {other:?}"),
            }
            let halves = cases_where(&cases, |cs| {
                cs["stem"] == stem.as_str()
                    && tensors_json(cs)[..2].iter().all(|t| t["strides"] == serde_json::json!([2 * c, 1]))
                    && tensors_json(cs)[..2].iter().map(|t| t["offset"].as_u64().unwrap()).collect::<Vec<_>>() == [0, c]
            });
            assert!(halves.len() >= 3, "{stem}: the gate has no case of the two halves of one tensor");
        }
    }
}

/// The bf16 nearest to an f32 (round to nearest, ties to even), as Triton's `.to(tl.bfloat16)` rounds a finite value.
fn bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

/// One stride pair for three views (`StrideOf(("q", "k", "v"), axis)`), on the GPU through the checked `Kernel::launch`: the q, k and v of one
/// fused buffer run and give `(q + k) + v`, and a `k` of other strides (a tensor of its own, whose row stride is 256 where `q` bound 768) is
/// refused before anything is written, as Python refuses it.
#[test]
fn views_that_share_a_stride_must_all_have_it() {
    let (gpu, manifest) = (Gpu::new(), manifest());
    let s: &CudaStream = &gpu.stream;
    let arch = Arch::of_context(&gpu.ctx).unwrap();
    let kernel = manifest.kernel(&gpu.ctx, "qkv_sum", &[], arch).unwrap();
    let rows = 7usize;
    // q, k and v hold 1.0, 2.0 and 4.0 (bf16 0x3f80, 0x4000, 0x4080) in each row of the fused [7, 3, 4, 64] tensor
    let mut fused = Vec::with_capacity(rows * 768 * 2);
    for _ in 0..rows {
        for bits in [0x3f80u16, 0x4000, 0x4080] {
            for _ in 0..256 {
                fused.extend_from_slice(&bits.to_le_bytes());
            }
        }
    }
    let fused = gpu.upload(&fused);
    let other = gpu.upload(&vec![0u8; rows * 256 * 2]);
    let mut o = gpu.zeros(rows * 256 * 2);
    let (shape, fused_strides, dense_strides) = ([rows, 4, 64], [768usize, 64, 1], [256usize, 64, 1]);
    // (a function, not a closure: a launch's tensors have one lifetime, and a closure would keep the first launch's output borrowed)
    fn view<'a>(fused: &'a Buf, shape: &'a [usize], strides: &'a [usize], offset: usize) -> TensorRef<'a, Bf16> {
        TensorRef::<Bf16>::from_cuda_strided(&fused.slice, shape, strides, offset).unwrap()
    }
    let k_other = TensorRef::<Bf16>::from_cuda_strided(&other.slice, &shape, &dense_strides, 0).unwrap();
    let out = TensorMut::<Bf16>::from_cuda(&mut o.slice, &shape).unwrap();
    let e = kernel
        .launch(
            s,
            &mut [
                view(&fused, &shape, &fused_strides, 0).into_arg(),
                k_other.into_arg(),
                view(&fused, &shape, &fused_strides, 512).into_arg(),
                out.into_arg(),
            ],
        )
        .unwrap_err();
    assert_eq!(e.precondition_kind(), Some(PreconditionKind::Stride), "{e}");
    assert!(
        e.to_string().contains("argument 'k'")
            && e.to_string().contains("the stride of axis 0 is 256")
            && e.to_string().contains("768"),
        "{e}"
    );
    assert!(gpu.download(&o).iter().all(|b| *b == 0), "a refused launch writes nothing");
    let out = TensorMut::<Bf16>::from_cuda(&mut o.slice, &shape).unwrap();
    let qkv = [0, 256, 512].map(|offset| view(&fused, &shape, &fused_strides, offset).into_arg());
    let [q, k, v] = qkv;
    kernel.launch(s, &mut [q, k, v, out.into_arg()]).unwrap();
    let want = bf16_bits((1.0f32 + 2.0) + 4.0).to_le_bytes();
    let got = gpu.download(&o);
    assert!(got.chunks(2).all(|b| b == want), "every element is 1 + 2 + 4 = 7 (bf16 {want:02x?})");
}

/// A copy of the gate's manifest directory in a fresh temporary directory.
fn copy_of_the_manifest(name: &str) -> PathBuf {
    let dir = gate_dir().join(cases()["manifest"].as_str().unwrap());
    let tmp = std::env::temp_dir().join(format!("tilefish-gate-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    for f in std::fs::read_dir(&dir).unwrap() {
        let f = f.unwrap();
        std::fs::copy(f.path(), tmp.join(f.file_name())).unwrap();
    }
    tmp
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn write_json(path: &Path, doc: &Value) {
    std::fs::write(path, serde_json::to_string_pretty(doc).unwrap()).unwrap();
}

/// What only the new entries can refuse: a view that reaches past its slice or starts where its binary was not compiled for, a device scalar
/// that is not one element, a workspace that is too small or missing, a bucket that is not exact, a cuBLASLt of another version, an
/// algorithm that cuBLASLt would not have written, a cubin of another architecture, an old schema.
#[test]
fn refusals_of_schema_3() {
    let (gpu, manifest, cases) = (Gpu::new(), manifest(), cases());
    let s: &CudaStream = &gpu.stream;
    let arch = Arch::of_context(&gpu.ctx).unwrap();
    let mut refused = 0;
    let mut check = |what: &str, ok: bool| {
        assert!(ok, "{what}");
        refused += 1;
        println!("rust gate: refused: {what}");
    };

    // a view that reaches further than its slice
    let dense_bytes = 40 * 8 * 64 * 2;
    let x = gpu.zeros(dense_bytes);
    let (shape, strides, dense_strides) = ([40usize, 8, 64], [1536usize, 64, 1], [512usize, 64, 1]);
    let e = TensorRef::<Bf16>::from_cuda_strided(&x.slice, &shape, &strides, 0).unwrap_err();
    check(
        "a view whose extent passes its slice (a [40, 8, 64] bf16 view of row stride 1536 of a slice for the dense tensor)",
        matches!(&e, Error::Launch(m) if m.contains("reaches") && m.contains("but the region has")),
    );
    let e = TensorRef::<Bf16>::from_cuda_strided(&x.slice, &shape, &dense_strides, 1).unwrap_err();
    check("... and a dense view that starts one element in", matches!(&e, Error::Launch(_)));
    // SAFETY: the region is the buffer's own and outlives the (refused) construction.
    let region = unsafe { gpu.region(&x, x.nbytes()) };
    check("... and the same over a region", TensorRef::<Bf16>::new_strided(region, &shape, &strides, 0).is_err());

    // Strided[..., 16]: a view that starts where 16 does not divide the address is refused, and one that does is run
    assert!(cases_where(&cases, |c| c["stem"] == "head_copy16").len() >= 3, "the gate has head_copy16");
    let kernel = manifest.kernel(&gpu.ctx, "head_copy16", &[], arch).unwrap();
    let flat = gpu.zeros((40 * 8 * 64 + 32) * 2);
    let mut y = gpu.zeros(dense_bytes);
    let y_shape = [40usize, 8, 64];
    for offset in [1usize, 2, 3, 5, 7, 9] {
        let view = TensorRef::<Bf16>::from_cuda_strided(&flat.slice, &shape, &dense_strides, offset).unwrap();
        let out = TensorMut::<Bf16>::from_cuda(&mut y.slice, &y_shape).unwrap();
        let e = kernel.launch(s, &mut [view.into_arg(), out.into_arg()]).unwrap_err();
        assert!(
            e.precondition_kind() == Some(PreconditionKind::PointerAlignment)
                && e.to_string().contains("not aligned to 16 bytes"),
            "offset {offset}: {e}"
        );
    }
    check(
        "a Strided[..., 16] view that starts off a 16-byte boundary (offsets 1, 2, 3, 5, 7, 9), and nothing was written",
        gpu.download(&y).iter().all(|b| *b == 0),
    );
    for offset in [0usize, 8, 16] {
        let view = TensorRef::<Bf16>::from_cuda_strided(&flat.slice, &shape, &dense_strides, offset).unwrap();
        let out = TensorMut::<Bf16>::from_cuda(&mut y.slice, &y_shape).unwrap();
        kernel.launch(s, &mut [view.into_arg(), out.into_arg()]).unwrap();
    }
    check("... and the views that start on 16 bytes (offsets 0, 8, 16) are launched", true);
    // a stride the binary fixes: the last axis is unit stride
    let wide = [1024usize, 128, 2];
    let flat2 = gpu.zeros(40 * 1024 * 2);
    let view = TensorRef::<Bf16>::from_cuda_strided(&flat2.slice, &shape, &wide, 0).unwrap();
    let out = TensorMut::<Bf16>::from_cuda(&mut y.slice, &y_shape).unwrap();
    let e = kernel.launch(s, &mut [view.into_arg(), out.into_arg()]).unwrap_err();
    check("a view whose last axis is not unit stride", e.precondition_kind() == Some(PreconditionKind::Stride));

    // a device scalar is one element
    assert!(!cases_where(&cases, |c| c["stem"] == "masked_rows").is_empty(), "the gate has masked_rows");
    let masked_rows = manifest.kernel(&gpu.ctx, "masked_rows", &[], arch).unwrap();
    let (xs, mut ys, two) = (gpu.zeros(64 * 128 * 4), gpu.zeros(64 * 128 * 4), gpu.zeros(8));
    let (rows, tok) = ([64usize, 128], [2usize]);
    let args = [
        TensorRef::<F32>::from_cuda(&xs.slice, &rows).unwrap().into_arg(),
        TensorMut::<F32>::from_cuda(&mut ys.slice, &rows).unwrap().into_arg(),
        TensorRef::<I32>::from_cuda(&two.slice, &tok).unwrap().into_arg(),
    ];
    let e = masked_rows.launch(s, &mut { args }).unwrap_err();
    check("a device scalar of two elements", e.precondition_kind() == Some(PreconditionKind::Shape));

    // the library: a GEMM with a bias for this architecture
    assert!(!manifest.library().is_empty(), "the gate's manifest has library entries");
    // (one whose pinned algorithm needs a workspace, so that a workspace that is too small is refused; some algorithms need none)
    let candidates: Vec<_> =
        manifest.library().iter().filter(|e| e.implementation.has_bias() && e.arch == arch.to_string()).collect();
    assert!(!candidates.is_empty(), "a gemm_bias for this architecture");
    let needs: Vec<(u64, &str)> = candidates
        .iter()
        .map(|e| (manifest.gemm(&gpu.ctx, &e.stem, &[], arch).unwrap().workspace_required(), e.stem.as_str()))
        .collect();
    println!("rust gate: the workspace each pinned gemm_bias needs: {needs:?}");
    let entry = candidates.iter().zip(&needs).find(|(_, (bytes, _))| *bytes > 16).map_or(candidates[0], |(e, _)| *e);
    let gemm = manifest.gemm(&gpu.ctx, &entry.stem, &[], arch).unwrap();
    let (m, n, k) = (entry.m as usize, entry.n as usize, entry.k as usize);
    let (a, b, bias) = (gpu.zeros(m * k * 2), gpu.zeros(n * k * 2), gpu.zeros(n * 2));
    let (mut out, mut tiny, mut workspace) =
        (gpu.zeros(m * n * 2), gpu.zeros(16), gemm.alloc_workspace(&gpu.stream).unwrap());
    let (a_shape, b_shape, bias_shape, out_shape, tiny_shape, ws_shape) =
        ([m, k], [n, k], [n], [m, n], [16usize], [workspace.len()]);
    macro_rules! operands {
        ($out:expr, $workspace:expr $(,)?) => {
            [
                TensorRef::<Bf16>::from_cuda(&a.slice, &a_shape).unwrap().into_arg(),
                TensorRef::<Bf16>::from_cuda(&b.slice, &b_shape).unwrap().into_arg(),
                TensorRef::<Bf16>::from_cuda(&bias.slice, &bias_shape).unwrap().into_arg(),
                $out,
                $workspace,
            ]
        };
    }
    if gemm.workspace_required() > 16 {
        let e = gemm
            .launch(
                s,
                &mut operands!(
                    TensorMut::<Bf16>::from_cuda(&mut out.slice, &out_shape).unwrap().into_arg(),
                    TensorMut::<U8>::from_cuda(&mut tiny.slice, &tiny_shape).unwrap().into_arg(),
                ),
            )
            .unwrap_err();
        check(
            "a GEMM workspace that is too small",
            e.precondition_kind() == Some(PreconditionKind::Shape) && e.to_string().contains("workspace"),
        );
    } else {
        // (every algorithm pinned on this device needs no workspace: any workspace is enough, the other checks of it still hold)
        println!("rust gate: the pinned algorithm of '{}' needs no workspace: no workspace is too small", entry.stem);
    }
    // the workspace is bytes
    let e = gemm
        .launch(
            s,
            &mut operands!(
                TensorMut::<Bf16>::from_cuda(&mut out.slice, &out_shape).unwrap().into_arg(),
                TensorMut::<Bf16>::from_cuda(&mut tiny.slice, &[8usize]).unwrap().into_arg(),
            ),
        )
        .unwrap_err();
    check("a GEMM workspace that is not bytes", e.precondition_kind() == Some(PreconditionKind::Dtype));
    // 256-byte aligned: a workspace that starts 8 bytes into its slice is not
    let strides = [1usize];
    let e = gemm
        .launch(
            s,
            &mut operands!(
                TensorMut::<Bf16>::from_cuda(&mut out.slice, &out_shape).unwrap().into_arg(),
                TensorMut::<U8>::from_cuda_strided(&mut workspace, &[ws_shape[0] - 8], &strides, 8).unwrap().into_arg(),
            ),
        )
        .unwrap_err();
    check(
        "a GEMM workspace that is not 256-byte aligned",
        e.precondition_kind() == Some(PreconditionKind::PointerAlignment),
    );
    // it must not overlap the result: two regions over one buffer, the result and a workspace of the bytes the algorithm needs (at least)
    let shared = gpu.zeros((m * n * 2).max(gemm.workspace_required() as usize));
    // SAFETY: both regions are the shared buffer's own memory, which outlives the call; nothing is launched (it is refused).
    let (out_region, ws_region) =
        unsafe { (gpu.unbound_region(&shared, shared.nbytes()), gpu.unbound_region(&shared, shared.nbytes())) };
    let overlap_shape = [shared.nbytes() as usize];
    let e = gemm
        .launch(
            s,
            &mut operands!(
                TensorMut::<Bf16>::new(out_region, &out_shape).unwrap().into_arg(),
                TensorMut::<U8>::new(ws_region, &overlap_shape).unwrap().into_arg(),
            ),
        )
        .unwrap_err();
    check("a GEMM workspace that overlaps the result", e.precondition_kind() == Some(PreconditionKind::Alias));
    let e = gemm
        .launch(
            s,
            &mut operands!(
                TensorRef::<Bf16>::from_cuda(&out.slice, &out_shape).unwrap().into_arg(),
                TensorMut::<U8>::from_cuda(&mut workspace, &ws_shape).unwrap().into_arg(),
            ),
        )
        .unwrap_err();
    check("a GEMM result that is read only", e.precondition_kind() == Some(PreconditionKind::Access));
    let mut without = Vec::from(operands!(
        TensorMut::<Bf16>::from_cuda(&mut out.slice, &out_shape).unwrap().into_arg(),
        TensorMut::<U8>::from_cuda(&mut workspace, &ws_shape).unwrap().into_arg(),
    ));
    without.pop();
    let e = gemm.launch(s, &mut without).unwrap_err();
    check("a GEMM with no workspace", matches!(&e, Error::Launch(m) if m.contains("workspace")));
    let a_more = gpu.zeros((m + 1) * k * 2);
    let mut out_more = gpu.zeros((m + 1) * n * 2);
    let (a_more_shape, out_more_shape) = ([m + 1, k], [m + 1, n]);
    let mut wrong = operands!(
        TensorMut::<Bf16>::from_cuda(&mut out_more.slice, &out_more_shape).unwrap().into_arg(),
        TensorMut::<U8>::from_cuda(&mut workspace, &ws_shape).unwrap().into_arg(),
    );
    wrong[0] = TensorRef::<Bf16>::from_cuda(&a_more.slice, &a_more_shape).unwrap().into_arg();
    let e = gemm.launch(s, &mut wrong).unwrap_err();
    check("a GEMM at a size that is not its bucket", e.precondition_kind() == Some(PreconditionKind::Bucket));
    let e = manifest.gemm(&gpu.ctx, &entry.stem, &[], Arch::sm(120)).unwrap_err();
    check(
        "a GEMM asked for another architecture than the device's",
        matches!(&e, Error::NoBinary(_) | Error::Unsupported(_)),
    );

    // a cuBLASLt of another version than the one that pinned the algorithm (the edited manifest is trusted, as a pin would be made again)
    let here = tilefish::cublaslt_version().expect("cuBLASLt is on this machine");
    let tmp = copy_of_the_manifest("version");
    let mut doc = read_json(&tmp.join("manifest.json"));
    for e in doc["library"].as_array_mut().unwrap() {
        e["cublaslt_version"] = (here + 100).into();
    }
    write_json(&tmp.join("manifest.json"), &doc);
    let other = Manifest::load(&tmp, Pin::TrustDirectory).unwrap();
    let e = other.gemm(&gpu.ctx, &entry.stem, &[], arch).unwrap_err();
    check(
        "a cuBLASLt of another version than the one that pinned the algorithm",
        matches!(&e, Error::Manifest(m) if m.contains(&format!("pinned the cuBLASLt algorithm of '{}'", entry.stem))
            && m.contains(&format!("with cuBLASLt {}", here + 100))
            && m.contains(&format!("this process has cuBLASLt {here}"))),
    );
    // ... and an algorithm that cuBLASLt would not have written (a tile id past what its id has) is refused before cublasLtMatmulAlgoCheck, which
    // crashes the process on some of those, can see it
    let mut doc = read_json(&tmp.join("manifest.json"));
    for e in doc["library"].as_array_mut().unwrap() {
        e["cublaslt_version"] = here.into();
        let mut algo = e["algo"].as_str().unwrap().to_owned().into_bytes();
        algo[8..16].fill(b'f'); // the second 32-bit word of the first eight bytes: the tile id
        e["algo"] = String::from_utf8(algo).unwrap().into();
    }
    write_json(&tmp.join("manifest.json"), &doc);
    let corrupt = Manifest::load(&tmp, Pin::TrustDirectory).unwrap();
    let e = corrupt.gemm(&gpu.ctx, &entry.stem, &[], arch).unwrap_err();
    check(
        "a pinned algorithm with a tile id that its algorithm id does not have",
        matches!(&e, Error::Manifest(m) if m.contains("is not one cuBLASLt produces")),
    );
    let _ = std::fs::remove_dir_all(&tmp);

    // an algorithm whose split-K adds its partial sums in bf16 runs only for an entry that opts in (`allow_bf16_reduction`): the imported entries
    // of the gate (on a fresh export) load as they are, and the same entries without the opt-in are refused
    let opted: Vec<_> = manifest
        .library()
        .iter()
        .filter(|e| e.allow_bf16_reduction && e.algo_source == AlgoSource::Imported && e.arch == arch.to_string())
        .collect();
    if opted.is_empty() {
        println!(
            "rust gate: no entry opts in to bf16 split-K partial sums (a committed fixture, or no such split-K on this device)"
        );
    } else {
        for e in &opted {
            assert!(manifest.gemm(&gpu.ctx, &e.stem, &[], arch).is_ok(), "{} loads", e.stem);
        }
        let tmp = copy_of_the_manifest("bf16-reduction");
        let mut doc = read_json(&tmp.join("manifest.json"));
        for e in doc["library"].as_array_mut().unwrap() {
            e.as_object_mut().unwrap().remove("allow_bf16_reduction");
        }
        write_json(&tmp.join("manifest.json"), &doc);
        let strict = Manifest::load(&tmp, Pin::TrustDirectory).unwrap();
        let refused: Vec<String> = opted
            .iter()
            .filter_map(|e| strict.gemm(&gpu.ctx, &e.stem, &[], arch).err().map(|err| err.to_string()))
            .collect();
        check(
            "a split-K with bf16 partial sums in an entry that does not opt in",
            refused.len() >= 2 && refused.iter().all(|m| m.contains("bf16")),
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // a cubin of another architecture, under the name of this one: its ELF header says so
    let tmp = copy_of_the_manifest("arch");
    let sm120 = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("sm_120");
    let theirs_doc = read_json(&sm120.join("manifest.json"));
    let theirs = theirs_doc["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["stem"] == "gelu_erf")
        .expect("gelu_erf in sm_120")
        .clone();
    let mut doc = read_json(&tmp.join("manifest.json"));
    let mine = doc["kernels"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|k| k["stem"] == "gelu_erf" && k["arch"] == arch.to_string())
        .unwrap();
    std::fs::copy(sm120.join(theirs["cubin"].as_str().unwrap()), tmp.join(mine["cubin"].as_str().unwrap())).unwrap();
    mine["sha256"] = theirs["sha256"].clone();
    mine["sha256_code"] = theirs["sha256_code"].clone();
    write_json(&tmp.join("manifest.json"), &doc);
    let e = Manifest::load(&tmp, Pin::TrustDirectory).unwrap_err();
    check(
        "a cubin of sm_120 under the name of this architecture",
        matches!(&e, Error::Manifest(m) if m.contains("is for sm_120") && m.contains(&arch.to_string())),
    );
    let _ = std::fs::remove_dir_all(&tmp);

    // a manifest of schema 2 is told to be exported again
    let tmp = copy_of_the_manifest("schema2");
    let mut doc = read_json(&tmp.join("manifest.json"));
    doc["schema"] = 2.into();
    doc.as_object_mut().unwrap().remove("library");
    write_json(&tmp.join("manifest.json"), &doc);
    let e = Manifest::load(&tmp, Pin::TrustDirectory).unwrap_err();
    check(
        "a manifest of schema 2",
        matches!(&e, Error::Manifest(m) if m.contains("schema 2") && m.contains("Export it again")),
    );
    let _ = std::fs::remove_dir_all(&tmp);
    println!("rust gate: {refused} refusals of schema 3 checked");
}

/// Capture `launch` (which enqueues on the stream of `gpu`) in a CUDA graph, run the graph three times (`out` cleared before each) and compare
/// `out` with what Python's replay computed.
fn capture_and_run(gpu: &Gpu, launch: &dyn Fn() -> Result<(), Error>, out: &mut CudaSlice<u8>, expect: &[u8]) {
    let stream = gpu.stream.cu_stream();
    let mut graph: sys::CUgraph = std::ptr::null_mut();
    // SAFETY (the driver calls below): `stream` is a live stream of the current context; the graph handles come from the driver.
    unsafe {
        let begin = sys::cuStreamBeginCapture_v2(stream, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
        assert_eq!(begin, sys::CUresult::CUDA_SUCCESS, "cuStreamBeginCapture");
    }
    let launched = launch();
    unsafe {
        let end = sys::cuStreamEndCapture(stream, &mut graph);
        assert_eq!(end, sys::CUresult::CUDA_SUCCESS, "cuStreamEndCapture");
    }
    launched.expect("the launch is captured");
    gpu.stream.synchronize().unwrap();
    assert!(
        gpu.stream.clone_dtoh(out).unwrap().iter().all(|b| *b == 0),
        "a captured launch runs nothing: the output is untouched"
    );
    let mut exec: sys::CUgraphExec = std::ptr::null_mut();
    unsafe {
        assert_eq!(
            sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0),
            sys::CUresult::CUDA_SUCCESS,
            "cuGraphInstantiate"
        );
    }
    for run in 0..3 {
        gpu.stream.memset_zeros(out).unwrap();
        unsafe {
            assert_eq!(sys::cuGraphLaunch(exec, stream), sys::CUresult::CUDA_SUCCESS, "cuGraphLaunch");
        }
        gpu.stream.synchronize().unwrap();
        assert_eq!(
            gpu.stream.clone_dtoh(out).unwrap(),
            expect,
            "run {run}: the graph's output is Python's replay, byte for byte"
        );
    }
    unsafe {
        let _ = sys::cuGraphExecDestroy(exec);
        let _ = sys::cuGraphDestroy(graph);
    }
    gpu.stream.memset_zeros(out).unwrap();
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect()
}

/// A strided kernel and a GEMM captured in a CUDA graph with regions and `unsafe launch_raw` (and the kernel with the safe `launch`, which
/// captures regions too): the graph computes what Python's replay did, byte for byte, every time it runs. The loads (the module, the cuBLASLt
/// handle and the check of the algorithm) and the allocation of the workspace are before the capture.
#[test]
fn graph_capture_of_a_strided_kernel_and_of_a_gemm() {
    let (gpu, manifest, cases) = (Gpu::new(), manifest(), cases());
    let arch = Arch::of_context(&gpu.ctx).unwrap();
    let stream = gpu.stream.cu_stream();

    // -- a strided kernel: a view of a fused qkv
    let case = cases_where(&cases, |c| c["stem"] == "head_copy" && has_view(c))
        .into_iter()
        .next()
        .expect("a strided head_copy case");
    let kernel = manifest.kernel(&gpu.ctx, "head_copy", &[], arch).unwrap();
    let mut tensors = tensors_of(&gpu, &case);
    let expect = read(&gate_dir().join(case["tensors"][1]["expect"].as_str().unwrap()));
    let (x_shape, x_strides, x_offset) = (
        usizes(&case["tensors"][0]["shape"]),
        usizes(&case["tensors"][0]["strides"]),
        case["tensors"][0]["offset"].as_u64().unwrap() as usize,
    );
    let y_shape = usizes(&case["tensors"][1]["shape"]);
    gpu.stream.synchronize().unwrap();
    // SAFETY: the buffers are used until the end of this function, outlive the graph, and nothing else uses them while it is captured or run.
    let (xr, yr) = unsafe {
        (
            gpu.unbound_region(&tensors[0].1, tensors[0].1.nbytes()),
            gpu.unbound_region(&tensors[1].1, tensors[1].1.nbytes()),
        )
    };
    for raw in [true, false] {
        let launch = || {
            let x = TensorRef::<Bf16>::new_strided(xr, &x_shape, &x_strides, x_offset)?;
            let y = TensorMut::<Bf16>::new(yr, &y_shape)?;
            if raw {
                // SAFETY: `stream` is the handle of a stream of the kernel's context; the buffers outlive the graph.
                unsafe { kernel.launch_raw(stream, &[x.into_arg(), y.into_arg()]) }
            } else {
                kernel.launch(&gpu.stream, &mut [x.into_arg(), y.into_arg()])
            }
        };
        capture_and_run(&gpu, &launch, &mut tensors[1].1.slice, &expect);
    }
    println!(
        "rust gate: a strided kernel captured in a CUDA graph (launch and launch_raw, with regions) ran 3 times each and matched Python's replay"
    );

    // -- a GEMM with a bias, the smallest of the gate
    let case = cases_where(&cases, |c| is_gemm(c) && tensors_json(c).len() == 4)
        .into_iter()
        .min_by_key(|c| tensors_json(c)[1]["nbytes"].as_u64().unwrap())
        .expect("a GEMM case with a bias");
    let stem = case["stem"].as_str().unwrap().to_owned();
    let gemm = manifest.gemm(&gpu.ctx, &stem, &[], arch).unwrap();
    let mut workspace = gemm.alloc_workspace(&gpu.stream).unwrap();
    let mut tensors = tensors_of(&gpu, &case);
    let last = tensors.len() - 1;
    let expect = read(&gate_dir().join(case["tensors"][last]["expect"].as_str().unwrap()));
    let shapes: Vec<Vec<usize>> = tensors_json(&case).iter().map(|t| usizes(&t["shape"])).collect();
    let ws_shape = [workspace.len()];
    gpu.stream.synchronize().unwrap();
    // SAFETY: as above.
    let regions: Vec<DevicePtr<'static>> =
        unsafe { tensors.iter().map(|(_, b)| gpu.unbound_region(b, b.nbytes())).collect() };
    let ws = unsafe { DevicePtr::from_raw(workspace.device_ptr(&gpu.stream).0, workspace.len() as u64, 0) };
    let launch = || {
        let mut args: Vec<TensorArg<'_>> = Vec::new();
        for (i, shape) in shapes.iter().enumerate() {
            let access = if i == last { Access::InOut } else { Access::In };
            args.push(TensorArg::new(regions[i], shape, DType::Bf16, access)?);
        }
        args.push(TensorArg::new(ws, &ws_shape, DType::U8, Access::InOut)?);
        // SAFETY: the stream is the GEMM's; the regions are the buffers' own, which outlive the graph.
        unsafe { gemm.launch_raw(stream, &args) }
    };
    capture_and_run(&gpu, &launch, &mut tensors[last].1.slice, &expect);
    let _ = &mut workspace;
    println!(
        "rust gate: the GEMM '{stem}' captured in a CUDA graph with launch_raw and regions ran 3 times and matched Python's replay"
    );
}

/// A GEMM launch of slices is refused while the stream captures, as a kernel's is: a graph would hold their addresses.
#[test]
fn a_gemm_of_slices_on_a_capturing_stream_is_an_error_and_not_a_graph() {
    let (gpu, manifest, cases) = (Gpu::new(), manifest(), cases());
    let arch = Arch::of_context(&gpu.ctx).unwrap();
    let case = cases_where(&cases, |c| is_gemm(c) && tensors_json(c).len() == 3)
        .into_iter()
        .min_by_key(|c| tensors_json(c)[1]["nbytes"].as_u64().unwrap())
        .expect("a GEMM case without a bias");
    let gemm = manifest.gemm(&gpu.ctx, case["stem"].as_str().unwrap(), &[], arch).unwrap();
    let mut workspace = gemm.alloc_workspace(&gpu.stream).unwrap();
    let ws_shape = [workspace.len()];
    let mut bufs: Vec<Buf> = tensors_of(&gpu, &case).into_iter().map(|(_, b)| b).collect();
    gpu.stream.synchronize().unwrap();
    assert!(gpu.ctx.is_in_multi_stream_mode());
    let (a, rest) = bufs.split_first_mut().unwrap();
    let (b, rest) = rest.split_first_mut().unwrap();
    let out = &mut rest[0];
    let mut args =
        vec![r::<Bf16>(a).unwrap().into_arg(), r::<Bf16>(b).unwrap().into_arg(), m::<Bf16>(out).unwrap().into_arg()];
    args.push(TensorMut::<U8>::from_cuda(&mut workspace, &ws_shape).unwrap().into_arg());
    gpu.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).unwrap();
    let refused = gemm.launch(&gpu.stream, &mut args);
    let ended = gpu.stream.end_capture(sys::CUgraphInstantiate_flags(0));
    let e = refused.expect_err("the launch of slices was captured");
    assert!(matches!(&e, Error::Launch(why) if why.contains("capturing") && why.contains("launch_raw")), "{e:?}");
    if let Ok(Some(graph)) = ended {
        graph.launch().unwrap(); // whatever was captured (nothing) runs
    }
    gpu.stream.synchronize().unwrap();
    assert!(gpu.download(&bufs[2]).iter().all(|b| *b == 0), "nothing was captured and nothing was launched");
    println!("rust gate: a GEMM of slices on a capturing stream: {e}");
}

/// What the tests of launches on other streams of a GEMM and of a strided kernel share, as `Race` does for the gelu: the slices are allocated and
/// freed on `s1`, the launch under test is on `s2`, which a host function holds back, so that the slices are freed and allocated again while it is
/// queued. The expected output is what the launch gave on one stream with nothing else going on.
struct Held {
    gpu: Gpu,
    s1: Arc<CudaStream>,
    s2: Arc<CudaStream>,
}

impl Held {
    fn new() -> Self {
        let gpu = Gpu::new();
        let (s1, s2) = (gpu.stream.clone(), gpu.ctx.new_stream().unwrap());
        Self { gpu, s1, s2 }
    }

    /// Hold `s2` back: nothing queued on it from now runs until the returned thread has released it (after 150 ms).
    fn hold(&self) -> std::thread::JoinHandle<()> {
        RELEASE.store(false, Ordering::Release);
        // SAFETY: the function only waits for an atomic and takes no argument.
        unsafe { result::stream::launch_host_function(self.s2.cu_stream(), hold_until_released, std::ptr::null_mut()) }
            .expect("a host function on the second stream");
        std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(150));
            RELEASE.store(true, Ordering::Release);
        })
    }

    /// Upload the inputs of a case on `s1`, and make the outputs zeros there.
    fn bufs(&self, case: &Value) -> Vec<Buf> {
        tensors_of(&self.gpu, case).into_iter().map(|(_, b)| b).collect()
    }
}

/// Run `case` of the gate on `stream` with the typed arguments of `launch_generic`, on slices that live on `s1`.
fn launch_case_on(
    held: &Held,
    manifest: &Manifest,
    case: &Value,
    stream: &CudaStream,
    bufs: &mut [Buf],
) -> Result<(), Error> {
    let arch = Arch::of_context(&held.gpu.ctx)?;
    let stem = case["stem"].as_str().unwrap();
    let mut args: Vec<TensorArg<'_>> = Vec::new();
    for (t, buf) in tensors_json(case).iter().zip(bufs.iter_mut()) {
        args.push(arg_of(buf, t["dtype"].as_str().unwrap(), t["role"].as_str().unwrap())?);
    }
    if is_gemm(case) {
        let gemm = manifest.gemm(&held.gpu.ctx, stem, &[], arch)?;
        // the workspace on `s1` as well, and kept until the end of the test through `args`' borrow
        let mut workspace = gemm.alloc_workspace(&held.s1)?;
        let shape = [workspace.len()];
        args.push(TensorMut::<U8>::from_cuda(&mut workspace, &shape)?.into_arg());
        let launched = gemm.launch(stream, &mut args);
        drop(args);
        drop(workspace);
        return launched;
    }
    manifest.kernel(&held.gpu.ctx, stem, &[], arch)?.launch(stream, &mut args)
}

/// The rounds of the hazard tests of a case: freed and allocated again, on `s1`, while the launch is queued on `s2`: its first tensor (an input) with
/// the memory of another input put in its place, and its last (the output) with a known pattern; the kernel must read what the input held
/// and must not write into what is allocated in place of the output. 20 rounds of each; none may go wrong.
fn rounds_of_a_launch_on_another_stream(what: &str, case_filter: impl Fn(&Value) -> bool) {
    let (held, manifest, cases) = (Held::new(), manifest(), cases());
    let case = cases_where(&cases, case_filter)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("{what}: no such case in the gate"));
    let expect = read(&gate_dir().join(tensors_json(&case).last().unwrap()["expect"].as_str().unwrap()));
    let last = tensors_json(&case).len() - 1;
    let mut wrong = [0usize; 2];
    for (which, free_input) in [(0usize, true), (1, false)] {
        for _ in 0..20 {
            let mut bufs = held.bufs(&case);
            held.s1.synchronize().unwrap();
            let releaser = held.hold();
            launch_case_on(&held, &manifest, &case, &held.s2, &mut bufs).unwrap();
            // the slices are freed while the launch is queued behind the held-back stream, and the memory is allocated again
            let again = if free_input {
                let victim = bufs.remove(0);
                let size = victim.slice.len();
                drop(victim);
                // the same size as the input, with bytes that are not the input's, so that a read of it is told apart
                let filler: Vec<u8> = (0..size).map(|i| (i % 251) as u8 | 0x11).collect();
                (0..3).map(|_| held.s1.clone_htod(&filler).unwrap()).collect::<Vec<_>>()
            } else {
                let victim = bufs.pop().unwrap();
                let size = victim.slice.len();
                drop(victim);
                (0..3).map(|_| held.s1.clone_htod(&vec![0xA5u8; size]).unwrap()).collect::<Vec<_>>()
            };
            releaser.join().unwrap();
            held.gpu.ctx.synchronize().unwrap();
            let bad = if free_input {
                // the output is the last buffer that is left
                held.s1.clone_dtoh(&bufs[last - 1].slice).unwrap() != expect
            } else {
                again.iter().any(|z| held.s1.clone_dtoh(z).unwrap().iter().any(|b| *b != 0xA5))
            };
            wrong[which] += usize::from(bad);
            drop(again);
        }
    }
    println!(
        "rust gate: {what}: {} of 20 rounds wrong with the input freed, {} of 20 with the output freed",
        wrong[0], wrong[1]
    );
    assert_eq!(wrong, [0, 0], "{what}: a launch read or wrote memory that had been freed and allocated again");
}

/// The free of a GEMM's slices is ordered after the GEMM, which is enqueued on a stream that is held back: it reads its inputs as they were and does
/// not write into the memory that took the place of its output.
#[test]
fn a_gemm_on_another_stream_is_ordered_against_the_free_of_its_slices() {
    rounds_of_a_launch_on_another_stream("a GEMM on a held-back second stream, its slices freed on the first", |c| {
        is_gemm(c) && tensors_json(c).len() == 3 && tensors_json(c)[0]["shape"][0] == 32
    });
}

/// The same for a strided kernel: the launch is given a view of the bytes of its slice from the first element on, and cudarc still orders and
/// records the use of the slice.
#[test]
fn a_strided_launch_on_another_stream_is_ordered_against_the_free_of_its_slice() {
    rounds_of_a_launch_on_another_stream(
        "a strided kernel on a held-back second stream, its slices freed on the first",
        |c| c["stem"] == "head_copy" && has_view(c) && tensors_json(c)[0]["offset"].as_u64().unwrap() > 0,
    );
}

/// A tensor the kernel writes may be a view of a larger slice that starts at an element of it: the kernel writes there and nowhere else (the
/// launch hands cudarc the bytes of the slice from the first element on, and the kernel is given that address).
#[test]
fn a_written_view_lands_at_its_offset_and_nowhere_else() {
    let (gpu, manifest, cases) = (Gpu::new(), manifest(), cases());
    let case = cases["cases"].as_array().unwrap().iter().find(|c| c["launcher"] == "GeluErf").expect("a GeluErf case");
    let mut tensors = tensors_of(&gpu, case).into_iter();
    let ((_, x), (_, y)) = (tensors.next().unwrap(), tensors.next().unwrap());
    let expect = read(&gate_dir().join(case["tensors"][1]["expect"].as_str().unwrap()));
    let launcher = generated::GeluErf::load(&manifest, &gpu.ctx).unwrap();
    let n = y.shape[0];
    for offset in [8usize, 16, 24] {
        // a slice for the result with `offset` elements before it and 8 after, all 0xAB
        let mut storage = gpu.upload(&vec![0xABu8; (offset + n + 8) * 2]);
        let strides = [1usize];
        {
            let out = TensorMut::<Bf16>::from_cuda_strided(&mut storage.slice, &y.shape, &strides, offset).unwrap();
            launcher.launch(&gpu.stream, r::<Bf16>(&x).unwrap(), out).unwrap();
        }
        let got = gpu.download(&storage);
        assert!(got[..offset * 2].iter().all(|b| *b == 0xAB), "offset {offset}: bytes before the view were written");
        assert_eq!(&got[offset * 2..(offset + n) * 2], &expect[..], "offset {offset}: the view holds Python's replay");
        assert!(
            got[(offset + n) * 2..].iter().all(|b| *b == 0xAB),
            "offset {offset}: bytes after the view were written"
        );
    }
    println!(
        "rust gate: a written view of a slice (offsets 8, 16 and 24 elements) holds Python's replay and nothing around it changed"
    );
}

/// The GEMMs of the gate are pinned to algorithms that need no workspace on this device, so none of them exercises the workspace of a launch.
/// `python/tests/rust_workspace_case.py` pins one whose algorithm does (split-K: `workspace_case/`): it runs with the caller's workspace, a
/// workspace of exactly the size the algorithm needs, one that is dirty (the result does not depend on its contents) and one captured in a CUDA
/// graph with regions, each the bytes Python's `replay` computed with the workspace of torch's allocator; and one byte less is refused.
#[test]
fn a_gemm_whose_algorithm_needs_a_workspace_runs_with_the_callers_and_matches_replay() {
    let dir = gate_dir().join("workspace_case");
    assert!(
        dir.exists(),
        "{} is not there: run python/tests/rust_workspace_case.py on the gate directory",
        dir.display()
    );
    let case = read_json(&dir.join("case.json"));
    let gpu = Gpu::new();
    let arch = Arch::of_context(&gpu.ctx).unwrap();
    let manifest = Manifest::load(dir.join("manifest"), Pin::TrustDirectory).unwrap();
    let stem = case["stem"].as_str().unwrap();
    let gemm = manifest.gemm(&gpu.ctx, stem, &[], arch).unwrap();
    let (m, n, k) = (
        case["m"].as_u64().unwrap() as usize,
        case["n"].as_u64().unwrap() as usize,
        case["k"].as_u64().unwrap() as usize,
    );
    assert_eq!(gemm.sizes(), (m as u64, n as u64, k as u64));
    let required = gemm.workspace_required() as usize;
    assert!(
        required > 0 && required as u64 <= gemm.workspace_bytes(),
        "the pinned algorithm needs a workspace: {required}"
    );
    assert_eq!(
        required as u64,
        case["workspace_required"].as_u64().unwrap(),
        "what cublasLtMatmulAlgoCheck says is what the heuristic said"
    );
    let (a, b, bias) = (read(&dir.join("a.bin")), read(&dir.join("b.bin")), read(&dir.join("bias.bin")));
    let expect = read(&dir.join("expect.bin"));
    let (a_shape, b_shape, bias_shape, out_shape) = ([m, k], [n, k], [n], [m, n]);
    let (a_buf, b_buf, bias_buf) = (gpu.upload(&a), gpu.upload(&b), gpu.upload(&bias));

    // a workspace of the entry's size (zeros), of exactly what the algorithm needs (dirty: 0xFF), and what a launch needs minus one
    for (what, mut workspace) in [
        ("the entry's workspace", gemm.alloc_workspace(&gpu.stream).unwrap()),
        ("exactly the workspace the algorithm needs, dirty", gpu.stream.clone_htod(&vec![0xFFu8; required]).unwrap()),
    ] {
        let mut out = gpu.zeros(m * n * 2);
        let ws_shape = [workspace.len()];
        gemm.launch(
            &gpu.stream,
            &mut [
                TensorRef::<Bf16>::from_cuda(&a_buf.slice, &a_shape).unwrap().into_arg(),
                TensorRef::<Bf16>::from_cuda(&b_buf.slice, &b_shape).unwrap().into_arg(),
                TensorRef::<Bf16>::from_cuda(&bias_buf.slice, &bias_shape).unwrap().into_arg(),
                TensorMut::<Bf16>::from_cuda(&mut out.slice, &out_shape).unwrap().into_arg(),
                TensorMut::<U8>::from_cuda(&mut workspace, &ws_shape).unwrap().into_arg(),
            ],
        )
        .unwrap();
        assert_eq!(gpu.download(&out), expect, "{what}: the result is Python's replay, byte for byte");
    }
    let mut out = gpu.zeros(m * n * 2);
    let mut short = gpu.zeros(required - 1);
    let short_shape = [required - 1];
    let e = gemm
        .launch(
            &gpu.stream,
            &mut [
                TensorRef::<Bf16>::from_cuda(&a_buf.slice, &a_shape).unwrap().into_arg(),
                TensorRef::<Bf16>::from_cuda(&b_buf.slice, &b_shape).unwrap().into_arg(),
                TensorRef::<Bf16>::from_cuda(&bias_buf.slice, &bias_shape).unwrap().into_arg(),
                TensorMut::<Bf16>::from_cuda(&mut out.slice, &out_shape).unwrap().into_arg(),
                TensorMut::<U8>::from_cuda(&mut short.slice, &short_shape).unwrap().into_arg(),
            ],
        )
        .unwrap_err();
    assert!(e.precondition_kind() == Some(PreconditionKind::Shape) && e.to_string().contains("workspace"), "{e}");
    assert!(gpu.download(&out).iter().all(|b| *b == 0), "a refused launch writes nothing");

    // captured in a CUDA graph with regions: the workspace is the caller's, so nothing is allocated inside the capture
    let workspace = gemm.alloc_workspace(&gpu.stream).unwrap();
    let ws_len = workspace.len();
    gpu.stream.synchronize().unwrap();
    let stream = gpu.stream.cu_stream();
    // SAFETY: the buffers outlive the graph's launches (until the end of this function), and nothing else uses them meanwhile.
    let (ar, br, biasr, outr) = unsafe {
        (
            gpu.unbound_region(&a_buf, a_buf.nbytes()),
            gpu.unbound_region(&b_buf, b_buf.nbytes()),
            gpu.unbound_region(&bias_buf, bias_buf.nbytes()),
            gpu.unbound_region(&out, out.nbytes()),
        )
    };
    let wsr = unsafe { DevicePtr::from_raw(workspace.device_ptr(&gpu.stream).0, ws_len as u64, 0) };
    let ws_shape = [ws_len];
    let launch = || {
        let args = [
            TensorRef::<Bf16>::new(ar, &a_shape)?.into_arg(),
            TensorRef::<Bf16>::new(br, &b_shape)?.into_arg(),
            TensorRef::<Bf16>::new(biasr, &bias_shape)?.into_arg(),
            TensorMut::<Bf16>::new(outr, &out_shape)?.into_arg(),
            TensorMut::<U8>::new(wsr, &ws_shape)?.into_arg(),
        ];
        // SAFETY: the stream is the GEMM's; the regions are the buffers' own, which outlive the graph.
        unsafe { gemm.launch_raw(stream, &args) }
    };
    capture_and_run(&gpu, &launch, &mut out.slice, &expect);
    println!(
        "rust gate: the GEMM '{stem}' (m={m} n={n} k={k}, a split-K algorithm that needs {required} bytes of workspace) matches replay with the entry's workspace, with exactly {required} bytes (dirty) and in a CUDA graph; one byte less is refused"
    );
}
