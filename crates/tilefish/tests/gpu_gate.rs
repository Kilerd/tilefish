//! The Rust half of the byte-identity gate, on a real GPU: launch the cubins of a manifest through the generated launchers
//! and write what they computed, for `python/tests/rust_gate.py` to compare with what Python's `replay` computed.
//!
//! Built only with `--features gpu-gate`, and needs
//!
//! * `TILEFISH_GATE_GENERATED` (at compile time): the launchers `tilefish.hosts.rust.emit` wrote for the manifest (with `embed=True`);
//! * `TILEFISH_GATE_DIR` (at run time): the directory `rust_gate.py prepare` filled: `manifest/` (the manifest directory),
//!   `cases.json` (each case: the launcher, its tensors, the files of its inputs and of the outputs Python's `replay`
//!   gave) and the raw bytes of every tensor.
//!
//! ```text
//! python rust_gate.py prepare $WORK
//! TILEFISH_GATE_GENERATED=$WORK/generated.rs TILEFISH_GATE_DIR=$WORK \
//!     cargo test -p tilefish --features gpu-gate --test gpu_gate -- --test-threads=1 --nocapture
//! python rust_gate.py compare $WORK
//! ```
//!
//! The tests: `byte_identity` runs every case and writes every tensor after the launch (outputs, and the inputs, so that an `InOut`
//! tensor that was updated in place is compared too), through tensors made of cudarc slices (no `unsafe`) and the safe `launch` on a
//! cudarc stream; `refusals` shows that Rust refuses what Python refuses (a precondition, a tensor that does not fit its region, a
//! tampered cubin, a wrong pin, a manifest of another architecture) and what only Rust has (a read-only tensor for a parameter that
//! is written); `graph_capture` launches inside a CUDA graph capture, with the safe `launch` and with `unsafe launch_raw`, and runs the
//! graphs; `a_launch_on_another_stream_...` is the two-stream hazard of cudarc's event tracking, with the launch stream held back
//! by a host callback; `launch_raw_puts_back_the_current_context` is what it says.

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
    Access, Arch, Bf16, DevicePtr, Element, Error, F16, F32, Manifest, Pin, PreconditionKind, TensorMut, TensorRef,
    elf::sha256_hex,
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
        Buf { slice: self.stream.clone_htod(bytes).expect("upload"), shape: Vec::new() }
    }

    fn zeros(&self, nbytes: usize) -> Buf {
        Buf { slice: self.stream.alloc_zeros::<u8>(nbytes).expect("alloc"), shape: Vec::new() }
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

/// A device buffer of bytes and the shape of the tensor it holds.
struct Buf {
    slice: CudaSlice<u8>,
    shape: Vec<usize>,
}

impl Buf {
    fn shaped(mut self, shape: &[usize]) -> Self {
        self.shape = shape.to_vec();
        self
    }

    fn nbytes(&self) -> u64 {
        self.slice.len() as u64
    }
}

/// A read-only tensor over a buffer, as cudarc sees it: no `unsafe`, and refused when the shape does not fit.
fn r<T: Element>(b: &Buf) -> Result<TensorRef<'_, T>, Error> {
    TensorRef::from_cuda(&b.slice, &b.shape)
}

/// A tensor the kernel writes, over a buffer.
fn m<T: Element>(b: &mut Buf) -> Result<TensorMut<'_, T>, Error> {
    TensorMut::from_cuda(&mut b.slice, &b.shape)
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Launch the launcher a case names, on typed tensors made of cudarc slices: a launcher whose signature changed does not compile.
/// `bufs` are the tensors of the case in the order of the launcher's parameters.
fn launch(gpu: &Gpu, manifest: &Manifest, launcher: &str, bufs: &mut [Buf]) -> Result<(), Error> {
    use generated::*;
    let s: &CudaStream = &gpu.stream;
    match (launcher, bufs) {
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
        (other, bufs) => panic!("the gate does not know the launcher {other} with {} tensors", bufs.len()),
    }
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
            let buf = if t["role"] == "out" {
                gpu.zeros(nbytes)
            } else {
                gpu.upload(&read(&gate_dir().join(t["input"].as_str().unwrap())))
            };
            assert_eq!(buf.slice.len(), nbytes);
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
        launch(gpu, manifest, case["launcher"].as_str().unwrap(), &mut bufs).unwrap_or_else(|e| panic!("{id}: {e}"));
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

/// cudarc frees a `CudaSlice` on the stream it was allocated on (`cuMemFreeAsync`), and orders the free after the launches that used
/// it only through the events it records when the slice is given to ITS launch builder on the stream that launches. A tensor over a
/// slice, launched on another stream, used to be made for one stream and launched on another, so the free was not ordered after the
/// kernel: the memory was reused by the next allocation while the kernel (held back here by a host function on its stream, as a busy
/// stream would be) had yet to write it. Now the launch gives the slices to cudarc on the launch stream: nothing is written into the
/// memory that was allocated after the free.
#[test]
fn a_launch_on_another_stream_is_ordered_against_the_free_of_its_slices() {
    let (gpu, manifest) = (Gpu::new(), manifest());
    let (s1, s2) = (gpu.stream.clone(), gpu.ctx.new_stream().expect("a second stream"));
    let launcher = generated::GeluErf::load(&manifest, &gpu.ctx).unwrap();
    let n = 4096usize;
    let ones: Vec<u8> = (0..n).flat_map(|_| 0x3FC0u16.to_le_bytes()).collect(); // bf16 1.5: gelu_erf(1.5) is not zero
    let shape = [n];

    let mut stale_writes = 0;
    for round in 0..20 {
        RELEASE.store(false, Ordering::Release);
        let x = s1.clone_htod(&ones).unwrap();
        let mut y = s1.alloc_zeros::<u8>(n * 2).unwrap();
        s1.synchronize().unwrap();
        // the launch stream is busy: nothing behind this runs until it is released
        unsafe { result::stream::launch_host_function(s2.cu_stream(), hold_until_released, std::ptr::null_mut()) }
            .expect("a host function on the second stream");
        let releaser = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(150));
            RELEASE.store(true, Ordering::Release);
        });
        // slices allocated on s1, launched on s2
        launcher
            .launch(
                &s2,
                TensorRef::<Bf16>::from_cuda(&x, &shape).unwrap(),
                TensorMut::<Bf16>::from_cuda(&mut y, &shape).unwrap(),
            )
            .unwrap();
        // freed on s1 (where they were allocated) while the kernel has not run; then the memory is allocated again, and filled with the 1.5s of `ones` (what the kernel would change)
        drop(x);
        drop(y);
        let z1 = s1.clone_htod(&ones).unwrap();
        let z2 = s1.clone_htod(&ones).unwrap();
        releaser.join().unwrap();
        gpu.ctx.synchronize().unwrap();
        let (got1, got2) = (s1.clone_dtoh(&z1).unwrap(), s1.clone_dtoh(&z2).unwrap());
        if got1 != ones || got2 != ones {
            stale_writes += 1;
            println!("round {round}: the kernel wrote into memory that was allocated after its slices were freed");
        }
    }
    assert_eq!(stale_writes, 0, "a launch on another stream was not ordered against the free of its slices");
    println!("rust gate: 20 launches on a held-back second stream, their slices freed on the first: memory intact");
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
