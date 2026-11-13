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

/// What the tests of launches on other streams share. Everything is allocated on `s1` and freed there (as cudarc frees a slice on the
/// stream it was allocated on, with `cuMemFreeAsync`); the launch under test is on `s2`, which is held back by a host function as a busy
/// stream would be; `s3` is a stream with other work. The launch is `y = gelu(x)` over bf16 1.5s (`ones`), and one of the two is freed
/// while it is queued and allocated again: the new memory for `x` is filled with 3.0s (`threes`), so a kernel that reads it by mistake
/// writes `gelu(3.0)` into `y`, which is compared with what it must be; the new memory for `y` is filled with `ones`, so a kernel that
/// writes it by mistake changes it. A kernel that read zeros, or wrote zeros over zeros, would hide the hazard.
struct Race {
    gpu: Gpu,
    s1: Arc<CudaStream>,
    s2: Arc<CudaStream>,
    s3: Arc<CudaStream>,
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
        let (s1, s2, s3) = (gpu.stream.clone(), gpu.ctx.new_stream().unwrap(), gpu.ctx.new_stream().unwrap());
        let fill = |bits: u16| (0..Self::N).flat_map(move |_| bits.to_le_bytes()).collect::<Vec<u8>>();
        let (ones, threes) = (fill(0x3FC0), fill(0x4040)); // bf16 1.5 and 3.0
        let mut race = Self { gpu, s1, s2, s3, launcher, ones, threes, expect: Vec::new(), shape: [Self::N] };
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
        let mut y3 = r.s1.alloc_zeros::<u8>(Race::N * 2).unwrap();
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

/// A safe `launch` on a stream that is capturing, of tensors made of cudarc slices, is an error: cudarc's tracking waits on events that
/// were recorded outside the capture, which the driver refuses (`CUDA_ERROR_STREAM_CAPTURE_ISOLATION`). The alternative would be a graph
/// that holds the slices' addresses and nothing that keeps them alive. The stream is usable afterwards.
#[test]
fn a_safe_launch_of_slices_on_a_capturing_stream_is_an_error_and_not_a_graph() {
    let (gpu, manifest, cases) = (Gpu::new(), manifest(), cases());
    let case = cases["cases"].as_array().unwrap().iter().find(|c| c["launcher"] == "GeluErf").expect("a GeluErf case");
    let expect = read(&gate_dir().join(case["tensors"][1]["expect"].as_str().unwrap()));
    let mut tensors = tensors_of(&gpu, case).into_iter();
    let ((_, x), (_, mut y)) = (tensors.next().unwrap(), tensors.next().unwrap());
    let launcher = generated::GeluErf::load(&manifest, &gpu.ctx).unwrap();
    gpu.stream.synchronize().unwrap();

    // the tensors are made before the capture begins: making one is not what is under test
    let (xt, yt) = (r::<Bf16>(&x).unwrap(), m::<Bf16>(&mut y).unwrap());
    gpu.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).unwrap();
    let refused = launcher.launch(&gpu.stream, xt, yt);
    let ended = gpu.stream.end_capture(sys::CUgraphInstantiate_flags(0));
    let e = refused.expect_err("the launch of slices was captured");
    assert!(
        matches!(&e, Error::Driver { source, .. } if source.0 == sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_ISOLATION),
        "{e:?}"
    );
    assert!(!matches!(ended, Ok(Some(_))), "a graph was made of a launch that was refused");

    // nothing was captured, nothing was launched, and the stream works
    assert!(gpu.download(&y).iter().all(|b| *b == 0));
    launcher.launch(&gpu.stream, r::<Bf16>(&x).unwrap(), m::<Bf16>(&mut y).unwrap()).unwrap();
    assert_eq!(gpu.download(&y), expect, "and the next launch on the stream is Python's replay, byte for byte");
    println!("rust gate: a safe launch of slices on a capturing stream: {e}");
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
