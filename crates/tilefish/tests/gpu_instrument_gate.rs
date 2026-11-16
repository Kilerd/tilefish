//! The instrumented cases of the Rust gate, on a real GPU (`docs/instrument.md`): the instrumented export and plan that `python/tests/rust_gate.py
//! prepare` wrote (`WORK/instrumented.json`, `prepare_instrumented`), launched by this host with probes, for `rust_gate.py compare` to hold the outputs to
//! the production bytes and the records to Python's.
//!
//! Built only with `--features gpu-gate`; needs `TILEFISH_GATE_DIR` at run time (and nothing at compile time):
//!
//! ```text
//! python rust_gate.py prepare $WORK
//! TILEFISH_GATE_DIR=$WORK cargo test -p tilefish --features gpu-gate --test gpu_instrument_gate -- --test-threads=1 --nocapture
//! python rust_gate.py compare $WORK
//! ```
//!
//! What it does: `Manifest::load` refuses the instrumented export and `Manifest::load_instrumented` reads it; each case's kernel is launched with a probe
//! buffer of the bytes `Kernel::probe_bytes` says (`Kernel::launch_probed`, then `ProbeBuffer::collect`), and again with `Kernel::launch` and no buffer
//! (its profile scratch null: the same bytes, nothing recorded); a buffer one byte short is refused before anything is launched. `LaunchPlan::load`
//! refuses the instrumented plan and `LaunchPlan::load_instrumented` reads it; `instantiate_probed` runs it (launched, then captured and replayed, the
//! records collected after each with `PlanRun::collect_probes`), and `instantiate` runs it with no probe buffer; `instantiate_probed` of a production
//! plan of the gate is refused. Files: `WORK/rust_instrumented/<case>/<tensor>.bin` (and `.null.bin`), `plan/<output>.bin` (`.graph.bin`,
//! `.null.bin`), and the records as `tilefish.probe-trace.v1`: `probes.json`, `plan_probes.json`, `plan_graph_probes.json`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use serde_json::Value;
use tilefish::{
    Arch, Bf16, Element, Error, F16, F32, I32, I64, LaunchPlan, Manifest, Pin, PreconditionKind, ProbeBuffer,
    ProbeTrace, TensorArg, TensorMut, TensorRef, U8,
};

fn gate_dir() -> PathBuf {
    PathBuf::from(std::env::var("TILEFISH_GATE_DIR").expect("TILEFISH_GATE_DIR"))
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A tensor over a slice, for a launch: read-only for an `in` tensor, written otherwise; `dtype` is the name the manifest uses.
fn arg<'a>(slice: &'a mut CudaSlice<u8>, shape: &'a [usize], dtype: &str, role: &str) -> TensorArg<'a> {
    fn typed<'a, T: Element>(slice: &'a mut CudaSlice<u8>, shape: &'a [usize], role: &str) -> TensorArg<'a> {
        if role == "in" || role == "input" {
            TensorRef::<T>::from_cuda(&*slice, shape).expect("a tensor of its slice").into_arg()
        } else {
            TensorMut::<T>::from_cuda(slice, shape).expect("a tensor of its slice").into_arg()
        }
    }
    match dtype {
        "bf16" => typed::<Bf16>(slice, shape, role),
        "f16" => typed::<F16>(slice, shape, role),
        "f32" => typed::<F32>(slice, shape, role),
        "i32" => typed::<I32>(slice, shape, role),
        "i64" => typed::<I64>(slice, shape, role),
        "u8" => typed::<U8>(slice, shape, role),
        other => panic!("the gate does not know the dtype {other}"),
    }
}

/// The device memory of the tensors of a case or a plan (their inputs uploaded, the others zeroed) and their shapes.
struct Bufs {
    slices: Vec<CudaSlice<u8>>,
    shapes: Vec<Vec<usize>>,
    dtypes: Vec<String>,
    roles: Vec<String>,
    names: Vec<String>,
}

impl Bufs {
    fn of(stream: &Arc<CudaStream>, tensors: &[Value], dir: &Path) -> Self {
        let mut b =
            Bufs { slices: Vec::new(), shapes: Vec::new(), dtypes: Vec::new(), roles: Vec::new(), names: Vec::new() };
        for t in tensors {
            assert!(t.get("strides").is_none(), "the instrumented cases have no views");
            let shape: Vec<usize> =
                t["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
            let dtype = t["dtype"].as_str().unwrap().to_owned();
            let nbytes = shape.iter().product::<usize>().max(1) * dtype_size(&dtype);
            let slice = match t.get("input").and_then(Value::as_str) {
                Some(input) => stream.clone_htod(&read(&dir.join(input))).expect("upload"),
                None => stream.alloc_zeros::<u8>(nbytes).expect("alloc"),
            };
            b.slices.push(slice);
            b.shapes.push(shape);
            b.dtypes.push(dtype);
            b.roles.push(t["role"].as_str().unwrap().to_owned());
            b.names.push(t["name"].as_str().unwrap().to_owned());
        }
        b
    }

    fn args(&mut self) -> Vec<TensorArg<'_>> {
        Self::args_of(&mut self.slices, &self.shapes, &self.dtypes, &self.roles)
    }

    fn named(&mut self) -> Vec<(&str, TensorArg<'_>)> {
        let names: Vec<&str> = self.names.iter().map(String::as_str).collect();
        names.into_iter().zip(Self::args_of(&mut self.slices, &self.shapes, &self.dtypes, &self.roles)).collect()
    }

    fn args_of<'a>(
        slices: &'a mut [CudaSlice<u8>],
        shapes: &'a [Vec<usize>],
        dtypes: &'a [String],
        roles: &'a [String],
    ) -> Vec<TensorArg<'a>> {
        slices
            .iter_mut()
            .zip(shapes)
            .zip(dtypes.iter().zip(roles))
            .map(|((s, shape), (dtype, role))| arg(s, shape, dtype, role))
            .collect()
    }

    /// Write every tensor to `dir/<name><suffix>.bin` (the inputs too: a launch must leave them as they were), and zero the written ones again
    /// for the next launch.
    fn write_outputs(&mut self, stream: &Arc<CudaStream>, dir: &Path, suffix: &str) {
        std::fs::create_dir_all(dir).unwrap();
        stream.synchronize().unwrap();
        for (k, s) in self.slices.iter_mut().enumerate() {
            let bytes = stream.clone_dtoh(&*s).expect("download");
            std::fs::write(dir.join(format!("{}{suffix}.bin", self.names[k])), bytes).unwrap();
            if self.roles[k] != "in" && self.roles[k] != "input" {
                stream.memset_zeros(s).expect("zero");
            }
        }
        stream.synchronize().unwrap();
    }
}

fn dtype_size(dtype: &str) -> usize {
    match dtype {
        "u8" => 1,
        "bf16" | "f16" => 2,
        "f32" | "i32" => 4,
        "i64" => 8,
        other => panic!("{other}"),
    }
}

#[test]
fn instrumented_kernels_and_plans_record_their_probes() {
    let dir = gate_dir();
    let path = dir.join("instrumented.json");
    if !path.exists() {
        eprintln!("no {}: rust_gate.py prepared no instrumented case (a device older than sm_90)", path.display());
        return;
    }
    let doc: Value = serde_json::from_slice(&read(&path)).unwrap();
    let ctx = CudaContext::new(0).expect("a CUDA context on device 0");
    let stream = ctx.new_stream().expect("a stream");
    let arch = Arch::of_context(&ctx).unwrap();
    let out = dir.join("rust_instrumented");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();

    // -- the instrumented export: refused unless asked for
    let mdir = dir.join(doc["manifest"].as_str().unwrap());
    let digest = doc["digest"].as_str().unwrap();
    match Manifest::load(&mdir, Pin::Digest(digest)) {
        Err(Error::Manifest(m)) => assert!(m.contains("is an instrumented export"), "{m}"),
        other => panic!("Manifest::load took the instrumented export: {other:?}"),
    }
    let manifest = Manifest::load_instrumented(&mdir, Pin::Digest(digest)).expect("load_instrumented reads it");
    assert!(manifest.is_instrumented());

    let mut trace = ProbeTrace::default();
    for case in doc["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let kernel = manifest.kernel(&ctx, case["stem"].as_str().unwrap(), &[], arch).expect("the kernel loads");
        assert!(kernel.instrument().is_some(), "{id}: an instrumented binary");
        let mut bufs = Bufs::of(&stream, case["tensors"].as_array().unwrap(), &dir);
        let bytes = {
            let args = bufs.args();
            let prepared = kernel.check(&args).expect("the call checks");
            kernel.probe_bytes(&prepared).expect("an instrumented binary needs a probe buffer")
        };
        // a buffer one byte short is refused, before anything is launched
        let mut short = ProbeBuffer::new(&stream, bytes as usize - 1).unwrap();
        match kernel.launch_probed(&stream, &mut bufs.args(), &mut short) {
            Err(Error::Precondition { kind: PreconditionKind::Shape, .. }) => {}
            other => panic!("{id}: a short probe buffer was not refused: {other:?}"),
        }
        assert!(short.collect().is_err(), "{id}: nothing was launched into the short buffer");
        let mut probes = ProbeBuffer::new(&stream, bytes as usize).unwrap();
        kernel.launch_probed(&stream, &mut bufs.args(), &mut probes).expect("the probed launch");
        let mut launch = probes.collect().expect("the records");
        launch.manifest_digest = Some(digest.to_owned());
        assert!(launch.missing.is_empty(), "{id}: every warp of every CTA wrote its slot");
        assert!(
            launch
                .warps
                .iter()
                .all(|w| w.records.first().map(|r| r.id) == Some(0) && w.records.last().map(|r| r.id) == Some(1))
        );
        trace.launches.push(launch);
        bufs.write_outputs(&stream, &out.join(id), "");
        // the same binary launched with no probe buffer: its profile scratch is null, and it records nothing
        kernel.launch(&stream, &mut bufs.args()).expect("a launch without probes");
        bufs.write_outputs(&stream, &out.join(id), ".null");
    }
    trace.save(out.join("probes.json")).unwrap();

    // -- the instrumented plan
    let plan_doc = &doc["plan"];
    let pdir = dir.join(plan_doc["dir"].as_str().unwrap());
    let pin = plan_doc["digest"].as_str().unwrap();
    match LaunchPlan::load(&pdir, Pin::Digest(pin)) {
        Err(Error::Manifest(m)) => assert!(m.contains("is an instrumented export"), "{m}"),
        other => panic!("LaunchPlan::load took the instrumented plan: {other:?}"),
    }
    let plan = LaunchPlan::load_instrumented(&pdir, Pin::Digest(pin)).expect("load_instrumented reads it");
    let bucket: Vec<(String, u64)> =
        plan_doc["bucket"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap())).collect();
    let bucket: Vec<(&str, u64)> = bucket.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let run = plan.instantiate_probed(&stream, &bucket).expect("instantiate_probed");
    assert!(run.probed());
    assert!(run.collect_probes().is_err(), "nothing was run yet");
    let externals = plan_doc["externals"].as_array().unwrap();
    let mut bufs = Bufs::of(&stream, externals, &dir);
    run.launch(&stream, &mut bufs.named(), &[]).expect("the plan launches");
    let launched = run.collect_probes().expect("the records of the launch");
    assert!(launched.iter().all(|l| l.graph.as_deref() == Some(plan.graph()) && l.missing.is_empty()));
    ProbeTrace { launches: launched }.save(out.join("plan_probes.json")).unwrap();
    bufs.write_outputs(&stream, &out.join("plan"), "");
    {
        // captured and replayed: the graph holds the addresses of the steps' regions of the probe buffer
        let mut graph = run.capture(&stream, bufs.named(), &[]).expect("the plan captures");
        graph.replay(&stream).expect("the replay");
        let replayed = run.collect_probes().expect("the records of the replay");
        ProbeTrace { launches: replayed }.save(out.join("plan_graph_probes.json")).unwrap();
        stream.synchronize().unwrap();
        let name = externals.iter().find(|t| t["role"] == "output").unwrap()["name"].as_str().unwrap();
        let bytes = stream.clone_dtoh(&graph.external(name).unwrap()).unwrap();
        std::fs::write(out.join("plan").join(format!("{name}.graph.bin")), bytes).unwrap();
    }
    // the instrumented plan instantiated without probes: its steps' profile scratch is null, it records nothing
    let plain = plan.instantiate(&stream, &bucket).expect("instantiate");
    assert!(!plain.probed() && plain.collect_probes().is_err());
    let mut bufs = Bufs::of(&stream, externals, &dir);
    plain.launch(&stream, &mut bufs.named(), &[]).expect("the plan launches without probes");
    bufs.write_outputs(&stream, &out.join("plan"), ".null");

    // -- instantiate_probed of a production plan of the gate is refused
    if let Ok(text) = std::fs::read(dir.join("plans.json")) {
        let plans: Value = serde_json::from_slice(&text).unwrap();
        let first = &plans["plans"][0];
        let production =
            LaunchPlan::load(dir.join(first["dir"].as_str().unwrap()), Pin::Digest(first["digest"].as_str().unwrap()))
                .expect("the production plan loads");
        match production.instantiate_all_probed(&stream) {
            Err(Error::Launch(m)) => assert!(m.contains("production export"), "{m}"),
            other => panic!("instantiate_probed took a production plan: {other:?}"),
        }
    }
    eprintln!(
        "the instrumented cases ran: {} kernels with probes, the plan launched, replayed and without probes",
        doc["cases"].as_array().unwrap().len()
    );
}
