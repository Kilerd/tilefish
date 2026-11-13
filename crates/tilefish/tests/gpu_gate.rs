//! The Rust half of the byte-identity gate, on a real GPU: launch the cubins of a manifest through the generated launchers
//! and write what they computed, for `python/tests/rust_gate.py` to compare with what Python's `replay` computed.
//!
//! Built only with `--features gpu-gate`, and needs
//!
//! * `TILEFISH_GATE_GENERATED` (at compile time): the launchers `tilefish.hosts.rust.emit` wrote for the manifest;
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
//! The tests: `byte_identity` runs every case and writes every tensor after the launch (outputs, and the inputs, so
//! that an `InOut` tensor that was updated in place is compared too); `refusals` shows that Rust refuses what Python
//! refuses (a precondition, a tampered cubin, a wrong pin, a manifest of another architecture); `graph_capture` launches
//! inside a CUDA graph capture and runs the graph.

#[allow(dead_code)]
mod generated {
    include!(env!("TILEFISH_GATE_GENERATED"));
}

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr as _};
use serde_json::Value;
use tilefish::{
    Arch, Bf16, DevicePtr, Error, F32, Manifest, Pin, PreconditionKind, TensorMut, TensorRef, elf::sha256_hex,
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
    stream: Arc<CudaStream>,
}

impl Gpu {
    fn new() -> Self {
        let ctx = CudaContext::new(0).expect("a CUDA context on device 0");
        Self { stream: ctx.new_stream().expect("a stream") }
    }

    fn raw(&self) -> sys::CUstream {
        self.stream.cu_stream()
    }

    fn upload(&self, bytes: &[u8]) -> Buf {
        let slice = self.stream.clone_htod(bytes).expect("upload");
        Buf::new(self, slice)
    }

    fn zeros(&self, nbytes: usize) -> Buf {
        Buf::new(self, self.stream.alloc_zeros::<u8>(nbytes).expect("alloc"))
    }

    fn download(&self, buf: &Buf) -> Vec<u8> {
        self.stream.synchronize().expect("synchronize");
        self.stream.clone_dtoh(&buf.slice).expect("download")
    }
}

/// A device buffer and its address (taken once, outside of any launch).
struct Buf {
    slice: CudaSlice<u8>,
    addr: u64,
    shape: Vec<usize>,
}

impl Buf {
    fn new(gpu: &Gpu, slice: CudaSlice<u8>) -> Self {
        let addr = slice.device_ptr(&gpu.stream).0;
        Self { slice, addr, shape: Vec::new() }
    }

    fn shaped(mut self, shape: &[usize]) -> Self {
        self.shape = shape.to_vec();
        self
    }

    fn ptr(&self) -> DevicePtr {
        // SAFETY: the buffer is alive as long as `self`, on device 0, and the test sizes it for its shape.
        unsafe { DevicePtr::from_raw(self.addr) }
    }

    fn input<T: tilefish::Element>(&self) -> TensorRef<'_, T> {
        TensorRef::new(self.ptr(), &self.shape, 0)
    }

    fn output<T: tilefish::Element>(&self) -> TensorMut<'_, T> {
        TensorMut::new(self.ptr(), &self.shape, 0)
    }
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Launch the launcher a case names, on typed tensors: a launcher whose signature changed does not compile.
fn launch(gpu: &Gpu, manifest: &Manifest, launcher: &str, t: &HashMap<String, Buf>) -> Result<(), Error> {
    use generated::*;
    let s = gpu.raw();
    let (i, o) = (Buf::input::<F32>, Buf::output::<F32>);
    let (ib, ob) = (Buf::input::<Bf16>, Buf::output::<Bf16>);
    match launcher {
        "Softmax" => Softmax::load(manifest)?.launch(s, i(&t["x"]), o(&t["out"])),
        "Layernorm" => Layernorm::load(manifest)?.launch(s, ib(&t["x"]), ib(&t["w"]), ib(&t["b"]), ob(&t["out"])),
        "ResidualAddRmsnorm" => {
            ResidualAddRmsnorm::load(manifest)?.launch(s, ib(&t["x"]), o(&t["residual"]), ib(&t["w"]), ob(&t["out"]))
        }
        "GeluExact" => GeluExact::load(manifest)?.launch(s, ib(&t["x"]), ob(&t["out"])),
        "Swiglu" => Swiglu::load(manifest)?.launch(s, ib(&t["gate"]), ib(&t["up"]), ob(&t["out"])),
        "BucketedRmsDb96" => BucketedRmsDb96::load(manifest)?.launch(s, ib(&t["x"]), ib(&t["w"]), ob(&t["out"])),
        "BucketedRmsDb512" => BucketedRmsDb512::load(manifest)?.launch(s, ib(&t["x"]), ib(&t["w"]), ob(&t["out"])),
        "BucketedRmsDb1000" => BucketedRmsDb1000::load(manifest)?.launch(s, ib(&t["x"]), ib(&t["w"]), ob(&t["out"])),
        "Rmsnorm" => Rmsnorm::load(manifest)?.launch(s, ib(&t["x_ptr"]), ib(&t["w_ptr"]), ob(&t["y_ptr"])),
        "GeluErf" => GeluErf::load(manifest)?.launch(s, ib(&t["x_ptr"]), ob(&t["y_ptr"])),
        "Axpy" => Axpy::load(manifest)?.launch(s, i(&t["x_ptr"]), o(&t["y_ptr"])),
        other => panic!("the gate does not know the launcher {other}"),
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

#[test]
fn byte_identity() {
    let (gpu, manifest, cases) = (Gpu::new(), manifest(), cases());
    let out_dir = gate_dir().join("rust");
    let _ = std::fs::remove_dir_all(&out_dir);
    let mut ran = 0;
    for case in cases["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let names: Vec<String> =
            case["tensors"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap().to_owned()).collect();
        let map: HashMap<String, Buf> = tensors_of(&gpu, case).into_iter().collect();
        launch(&gpu, &manifest, case["launcher"].as_str().unwrap(), &map).unwrap_or_else(|e| panic!("{id}: {e}"));
        let case_dir = out_dir.join(id);
        std::fs::create_dir_all(&case_dir).unwrap();
        for name in &names {
            std::fs::write(case_dir.join(format!("{name}.bin")), gpu.download(&map[name])).unwrap();
        }
        ran += 1;
    }
    println!("rust gate: launched {ran} cases through the generated launchers");
    assert_eq!(ran, cases["cases"].as_array().unwrap().len());
}

fn kind(e: &Error) -> PreconditionKind {
    e.precondition_kind().unwrap_or_else(|| panic!("not a precondition error: {e:?}"))
}

#[test]
fn refusals() {
    let (gpu, manifest) = (Gpu::new(), manifest());
    let mut refused = 0;
    let mut check = |what: &str, ok: bool| {
        assert!(ok, "{what}");
        refused += 1;
        println!("rust gate: refused: {what}");
    };

    // a misaligned pointer (8 bytes past a 16-byte boundary): nothing is launched, the output stays untouched
    let (x, y) = (gpu.upload(&vec![1u8; 2 * 4096 + 64]), gpu.upload(&vec![0xAAu8; 2 * 4096]));
    let misaligned = Buf { slice: gpu.zeros(1).slice, addr: x.addr + 8, shape: vec![4096] };
    let y = y.shaped(&[4096]);
    let launcher = generated::GeluErf::load(&manifest).unwrap();
    let e = launcher.launch(gpu.raw(), misaligned.input::<Bf16>(), y.output::<Bf16>()).unwrap_err();
    check(
        "a misaligned pointer",
        kind(&e) == PreconditionKind::PointerAlignment && e.to_string().contains("not aligned to 16 bytes"),
    );
    check("... and nothing was written", gpu.download(&y).iter().all(|b| *b == 0xAA));

    // a size the binary has baked in
    let rows = gpu.zeros(3 * 2000 * 2).shaped(&[3, 2000]);
    let w = gpu.zeros(2000 * 2).shaped(&[2000]);
    let e = generated::Rmsnorm::load(&manifest).unwrap().launch(
        gpu.raw(),
        rows.input::<Bf16>(),
        w.input::<Bf16>(),
        rows.output::<Bf16>(),
    );
    let e = e.unwrap_err();
    check("a wrong static size", kind(&e) == PreconditionKind::StaticSize && e.to_string().contains("requires 2048"));

    // a bucketed dimension that is not the bucket: there is no padding
    let (x, w, o) = (
        gpu.zeros(3 * 100 * 2).shaped(&[3, 100]),
        gpu.zeros(100 * 2).shaped(&[100]),
        gpu.zeros(3 * 100 * 2).shaped(&[3, 100]),
    );
    let e = generated::BucketedRmsDb96::load(&manifest)
        .unwrap()
        .launch(gpu.raw(), x.input::<Bf16>(), w.input::<Bf16>(), o.output::<Bf16>())
        .unwrap_err();
    check("a wrong bucket", kind(&e) == PreconditionKind::Bucket && e.to_string().contains("no padding"));

    // more elements than a generated kernel's 32-bit offsets reach. The shape is only claimed: the buffers are small, and the
    // second one's address is put far from the first so that the claimed extents do not overlap (they would be refused as an
    // alias first). The call is refused before anything is launched, so no byte past the buffers is touched.
    let big = [2_200_000usize, 1000];
    let a = gpu.zeros(64).shaped(&big);
    let b = Buf { slice: gpu.zeros(64).slice, addr: a.addr + (1 << 36), shape: big.to_vec() };
    let e = generated::Softmax::load(&manifest)
        .unwrap()
        .launch(gpu.raw(), a.input::<F32>(), b.output::<F32>())
        .unwrap_err();
    check("max_elements", kind(&e) == PreconditionKind::MaxElements);

    // a runtime dimension that is not a multiple of 16, and a written tensor that aliases another
    let a = gpu.zeros(64 * 2).shaped(&[60]);
    let b = gpu.zeros(64 * 2).shaped(&[60]);
    let e = generated::GeluErf::load(&manifest)
        .unwrap()
        .launch(gpu.raw(), a.input::<Bf16>(), b.output::<Bf16>())
        .unwrap_err();
    check("divisible_by", kind(&e) == PreconditionKind::Divisible);
    let a = gpu.zeros(4096 * 4).shaped(&[4096]);
    let e =
        generated::Axpy::load(&manifest).unwrap().launch(gpu.raw(), a.input::<F32>(), a.output::<F32>()).unwrap_err();
    check("an in-place tensor that aliases another", kind(&e) == PreconditionKind::Alias);

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
    let here = Arch::current().unwrap();
    assert_ne!(here, Arch::sm(120), "the refusal test needs a device that is not an sm_120");
    let e = other.kernel("softmax", &[], here).unwrap_err();
    check(
        "an sm_120-only manifest on this device",
        matches!(&e, Error::NoBinary(m) if m.contains(&format!("no binary for 'softmax' on {here}")) && m.contains("sm_120")),
    );
    // ... and a launcher typed from one manifest refuses another, whatever it holds
    let e = generated::Softmax::load(&other).unwrap_err();
    check("a launcher refuses a manifest it was not generated from", matches!(e, Error::PinMismatch { .. }));
    // a manifest that has the kernel, but not for this architecture, asked for explicitly
    let e = manifest.kernel("softmax", &[], Arch::sm(120)).unwrap_err();
    check("an explicit architecture the manifest does not have", matches!(e, Error::NoBinary(_)));
    // an architecture that is not the device's
    let e = other.kernel("softmax", &[], Arch::sm(120)).unwrap_err();
    check(
        "an architecture that is not the current device's",
        matches!(&e, Error::Unsupported(m) if m.contains("current CUDA device")),
    );
    println!("rust gate: {refused} refusals checked");
}

#[test]
fn graph_capture() {
    let (gpu, manifest, cases) = (Gpu::new(), manifest(), cases());
    // the first gelu_erf case of the gate: its inputs, and the bytes Python's replay computed
    let case = cases["cases"].as_array().unwrap().iter().find(|c| c["launcher"] == "GeluErf").expect("a GeluErf case");
    let mut tensors = tensors_of(&gpu, case).into_iter();
    let (x, mut y) = (tensors.next().unwrap(), tensors.next().unwrap());
    assert_eq!((x.0.as_str(), y.0.as_str()), ("x_ptr", "y_ptr"));
    let (x, y_name) = (x.1, y.0.clone());
    let y_buf = &mut y.1;
    let expect = read(&gate_dir().join(case["tensors"][1]["expect"].as_str().unwrap()));
    assert_eq!(y_name, "y_ptr");
    let launcher = generated::GeluErf::load(&manifest).unwrap(); // loads the module: not allowed during capture
    gpu.stream.synchronize().unwrap();

    let stream = gpu.raw();
    let mut graph: sys::CUgraph = std::ptr::null_mut();
    // SAFETY (the driver calls below): `stream` is a live stream of the current context; the graph handles come from the driver.
    unsafe {
        let begin = sys::cuStreamBeginCapture_v2(stream, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
        assert_eq!(begin, sys::CUresult::CUDA_SUCCESS, "cuStreamBeginCapture");
    }
    let launched = launcher.launch(stream, x.input::<Bf16>(), y_buf.output::<Bf16>());
    unsafe {
        let end = sys::cuStreamEndCapture(stream, &mut graph);
        assert_eq!(end, sys::CUresult::CUDA_SUCCESS, "cuStreamEndCapture");
    }
    launched.expect("the launch is captured");
    assert!(gpu.download(y_buf).iter().all(|b| *b == 0), "a captured launch runs nothing: the output is untouched");

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
        gpu.stream.memset_zeros(&mut y_buf.slice).unwrap();
        unsafe {
            assert_eq!(sys::cuGraphLaunch(exec, stream), sys::CUresult::CUDA_SUCCESS, "cuGraphLaunch");
        }
        let got = gpu.download(y_buf);
        assert_eq!(got, expect, "run {run}: the graph's output is Python's replay, byte for byte");
    }
    unsafe {
        let _ = sys::cuGraphExecDestroy(exec);
        let _ = sys::cuGraphDestroy(graph);
    }
    println!("rust gate: a launch captured in a CUDA graph ran 3 times and matched Python's replay");
}
