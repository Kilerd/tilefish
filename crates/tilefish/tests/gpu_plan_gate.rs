//! The Rust half of the gate of launch plans, on a real GPU: the plans `python/tests/rust_gate.py prepare` exported and ran through Python's
//! plan replay, loaded with their pins, instantiated once, launched and captured in CUDA graphs here, byte for byte against Python.
//!
//! Built only with `--features gpu-gate`, and needs `TILEFISH_GATE_DIR` (at run time): the directory `rust_gate.py prepare` filled, with
//! `plans.json` and `plans/` (the exported plan directories, the inputs and the bytes Python's replay left in every output and in-place tensor).
//!
//! ```text
//! python rust_gate.py prepare $WORK
//! TILEFISH_GATE_DIR=$WORK cargo test -p tilefish --features gpu-gate --test gpu_plan_gate -- --test-threads=1 --nocapture
//! python rust_gate.py compare $WORK
//! ```
//!
//! The tests: every case of every plan launched (`PlanRun::launch`) and captured and replayed (`CapturedPlan::replay`), each equal to Python's
//! bytes (and written to `rust_plans/` and `rust_plans_graph/` for `compare`); one graph per bucket replayed with the inputs and the device
//! scalars of every case of its bucket written between replays, a launch with other values in between; two launches of one run on two streams
//! back to back; the refusals (a tampered plan, a wrong pin, a plan of another manifest, a missing or extra external, a wrong shape, dtype or
//! bucket, ...), each of which launches nothing; the device memory over 100 launches and 100 replays after instantiation, which does not grow;
//! and the host time of a launch and of a replay (reported).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr as _, sys};
use serde_json::Value;
use tilefish::{
    Access, Bf16, BufferRole, DType, DevScalar, DevicePtr, Element, Error, F16, F32, I32, I64, LaunchPlan, Pin,
    PreconditionKind, Tap, TensorArg, TensorMut, TensorRef, U8, plan_digest,
};

fn gate_dir() -> PathBuf {
    PathBuf::from(std::env::var("TILEFISH_GATE_DIR").expect("TILEFISH_GATE_DIR"))
}

fn plans() -> Vec<Value> {
    let text = std::fs::read_to_string(gate_dir().join("plans.json")).expect("plans.json (run rust_gate.py prepare)");
    let doc: Value = serde_json::from_str(&text).expect("plans.json is JSON");
    doc["plans"].as_array().unwrap().clone()
}

fn plan_named(name: &str) -> Value {
    plans().into_iter().find(|p| p["name"] == name).unwrap_or_else(|| panic!("a plan '{name}'"))
}

fn read(path: &str) -> Vec<u8> {
    std::fs::read(gate_dir().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn load(plan: &Value) -> LaunchPlan {
    LaunchPlan::load(gate_dir().join(plan["dir"].as_str().unwrap()), Pin::Digest(plan["digest"].as_str().unwrap()))
        .expect("the plan loads with its pin")
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
}

/// One external of a case on the device.
struct Ext {
    name: String,
    role: String,
    dtype: DType,
    shape: Vec<usize>,
    slice: CudaSlice<u8>,
    input: Option<Vec<u8>>,
    expect: Option<Vec<u8>>,
}

fn dtype_of(name: &str) -> DType {
    DType::from_name(name).unwrap_or_else(|| panic!("a dtype {name}"))
}

fn upload(stream: &Arc<CudaStream>, case: &Value) -> Vec<Ext> {
    case["externals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            let dtype = dtype_of(t["dtype"].as_str().unwrap());
            let shape: Vec<usize> =
                t["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
            let bytes = shape.iter().product::<usize>() * dtype.size();
            let input = t.get("input").map(|p| read(p.as_str().unwrap()));
            let slice = match &input {
                Some(data) => stream.clone_htod(data).unwrap(),
                None => stream.alloc_zeros::<u8>(bytes).unwrap(),
            };
            Ext {
                name: t["name"].as_str().unwrap().to_owned(),
                role: t["role"].as_str().unwrap().to_owned(),
                dtype,
                shape,
                slice,
                input,
                expect: t.get("expect").map(|p| read(p.as_str().unwrap())),
            }
        })
        .collect()
}

fn typed<'a>(slice: &'a mut CudaSlice<u8>, shape: &'a [usize], dtype: DType, write: bool) -> TensorArg<'a> {
    fn make<'a, T: Element>(slice: &'a mut CudaSlice<u8>, shape: &'a [usize], write: bool) -> TensorArg<'a> {
        if write {
            TensorMut::<T>::from_cuda(slice, shape).unwrap().into_arg()
        } else {
            TensorRef::<T>::from_cuda(&*slice, shape).unwrap().into_arg()
        }
    }
    match dtype {
        DType::Bf16 => make::<Bf16>(slice, shape, write),
        DType::F16 => make::<F16>(slice, shape, write),
        DType::F32 => make::<F32>(slice, shape, write),
        DType::I32 => make::<I32>(slice, shape, write),
        DType::I64 => make::<I64>(slice, shape, write),
        DType::U8 => make::<U8>(slice, shape, write),
        other => panic!("the gate does not know {other:?}"),
    }
}

/// The externals of a case: a `TensorRef` for what the plan reads (a `TensorMut` for an input too when `inputs_mut`, so that a graph lets it be
/// rewritten), a `TensorMut` for what it writes.
fn args(exts: &mut [Ext], inputs_mut: bool) -> Vec<(&str, TensorArg<'_>)> {
    exts.iter_mut()
        .map(|e| {
            let write = matches!(e.role.as_str(), "output" | "inout") || (inputs_mut && e.role == "input");
            (e.name.as_str(), typed(&mut e.slice, &e.shape, e.dtype, write))
        })
        .collect()
}

fn devscalars(case: &Value) -> Vec<(String, DevScalar)> {
    case["devscalars"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(n, v)| (n.clone(), DevScalar::Int(v.as_i64().unwrap())))
        .collect()
}

fn named(values: &[(String, DevScalar)]) -> Vec<(&str, DevScalar)> {
    values.iter().map(|(n, v)| (n.as_str(), *v)).collect()
}

/// Put an in-place tensor back to its input and clear the outputs.
fn reset(stream: &Arc<CudaStream>, exts: &mut [Ext]) {
    for e in exts.iter_mut() {
        match e.role.as_str() {
            "inout" => stream.memcpy_htod(e.input.as_ref().unwrap(), &mut e.slice).unwrap(),
            "output" => stream.memset_zeros(&mut e.slice).unwrap(),
            _ => {}
        }
    }
}

fn write_out(subdir: &str, case: &str, name: &str, bytes: &[u8]) {
    let dir = gate_dir().join(subdir).join(case);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(format!("{name}.bin")), bytes).unwrap();
}

fn differing(got: &[u8], expect: &[u8]) -> usize {
    got.iter().zip(expect).filter(|(a, b)| a != b).count() + got.len().abs_diff(expect.len())
}

/// Every case of every plan: the launch, then a graph captured with the same tensors and replayed (twice), against Python's bytes.
#[test]
fn every_case_launched_and_replayed_is_pythons_bytes() {
    let gpu = Gpu::new();
    let (mut cases, mut tensors) = (0, 0);
    for plan in plans() {
        let lp = load(&plan);
        assert_eq!(lp.digest(), plan["digest"].as_str().unwrap());
        assert_eq!(lp.arena_bytes(), plan["arena_bytes"].as_u64().unwrap());
        let run = lp.instantiate_all(&gpu.stream).expect("the plan instantiates");
        for case in plan["cases"].as_array().unwrap() {
            let id = case["id"].as_str().unwrap();
            let values = devscalars(case);
            let mut exts = upload(&gpu.stream, case);
            run.launch(&gpu.stream, &mut args(&mut exts, false), &named(&values))
                .unwrap_or_else(|e| panic!("{id}: {e}"));
            gpu.stream.synchronize().unwrap();
            for e in exts.iter().filter(|e| e.expect.is_some()) {
                let got = gpu.stream.clone_dtoh(&e.slice).unwrap();
                write_out("rust_plans", id, &e.name, &got);
                let expect = e.expect.as_ref().unwrap();
                assert!(
                    got == *expect,
                    "{id}: launch: '{}' differs from Python's replay in {} bytes",
                    e.name,
                    differing(&got, expect)
                );
                tensors += 1;
            }
            // the same case in a CUDA graph: the in-place tensors back to their inputs, the outputs cleared
            reset(&gpu.stream, &mut exts);
            gpu.stream.synchronize().unwrap();
            let expects: Vec<(String, Vec<u8>, String)> =
                exts.iter().filter_map(|e| e.expect.clone().map(|x| (e.name.clone(), x, e.role.clone()))).collect();
            let inputs: Vec<(String, Vec<u8>)> =
                exts.iter().filter(|e| e.role == "inout").map(|e| (e.name.clone(), e.input.clone().unwrap())).collect();
            let mut graph = run
                .capture(&gpu.stream, args(&mut exts, false), &named(&values))
                .unwrap_or_else(|e| panic!("{id}: {e}"));
            gpu.stream.synchronize().unwrap();
            for (name, _, role) in &expects {
                if role == "output" {
                    let now = gpu.stream.clone_dtoh(&graph.external(name).unwrap()).unwrap();
                    assert!(now.iter().all(|b| *b == 0), "{id}: a capture runs nothing, but '{name}' was written");
                }
            }
            for round in 0..2 {
                if round == 1 {
                    // a graph holds addresses: the in-place tensors are written back through the graph, the outputs cleared
                    for (name, input) in &inputs {
                        gpu.stream.memcpy_htod(input, &mut graph.external_mut(name).unwrap()).unwrap();
                    }
                    for (name, _, role) in &expects {
                        if role == "output" {
                            gpu.stream.memset_zeros(&mut graph.external_mut(name).unwrap()).unwrap();
                        }
                    }
                }
                graph.replay(&gpu.stream).unwrap_or_else(|e| panic!("{id}: {e}"));
                gpu.stream.synchronize().unwrap();
                for (name, expect, _) in &expects {
                    let got = gpu.stream.clone_dtoh(&graph.external(name).unwrap()).unwrap();
                    if round == 0 {
                        write_out("rust_plans_graph", id, name, &got);
                    }
                    assert!(
                        got == *expect,
                        "{id}: replay {round}: '{name}' differs from Python's replay in {} bytes",
                        differing(&got, expect)
                    );
                }
            }
            drop(graph);
            cases += 1;
        }
    }
    println!(
        "rust plan gate: {cases} cases of plan replay: PlanRun::launch and two CapturedPlan::replay each byte-identical to Python ({tensors} tensors)"
    );
}

/// Debug taps (`PlanRun::launch_tapped`) of the block with reuse: every intermediate after the last step that writes it (later steps reuse its
/// bytes) and `t2` after step 3, against what Python's `taps=` copied out of its replay of the same case; the outputs are still Python's.
#[test]
fn taps_are_pythons_bytes() {
    let gpu = Gpu::new();
    let plan = plan_named("block-reuse");
    let run = load(&plan).instantiate_all(&gpu.stream).unwrap();
    let mut tapped = 0;
    for case in plan["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let specs = case["taps"].as_array().expect("the case records taps");
        let values = devscalars(case);
        let mut exts = upload(&gpu.stream, case);
        let mut slices: Vec<(CudaSlice<u8>, Vec<usize>, DType)> = specs
            .iter()
            .map(|t| {
                let dtype = dtype_of(t["dtype"].as_str().unwrap());
                let shape: Vec<usize> =
                    t["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
                (gpu.stream.alloc_zeros::<u8>(shape.iter().product::<usize>() * dtype.size()).unwrap(), shape, dtype)
            })
            .collect();
        {
            let mut taps: Vec<Tap<'_>> = specs
                .iter()
                .zip(slices.iter_mut())
                .map(|(t, (slice, shape, dtype))| Tap {
                    buffer: t["buffer"].as_str().unwrap(),
                    after: t["after"].as_u64().map(|a| a as usize),
                    into: typed(slice, shape, *dtype, true),
                })
                .collect();
            run.launch_tapped(&gpu.stream, &mut args(&mut exts, false), &named(&values), &mut taps)
                .unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        gpu.stream.synchronize().unwrap();
        for (t, (slice, _, _)) in specs.iter().zip(&slices) {
            let got = gpu.stream.clone_dtoh(slice).unwrap();
            let expect = read(t["expect"].as_str().unwrap());
            assert!(
                got == expect,
                "{id}: tap {} after {}: {} bytes differ",
                t["buffer"],
                t["after"],
                differing(&got, &expect)
            );
            tapped += 1;
        }
        for e in exts.iter().filter(|e| e.expect.is_some()) {
            assert!(
                gpu.stream.clone_dtoh(&e.slice).unwrap() == *e.expect.as_ref().unwrap(),
                "{id}: '{}' with taps",
                e.name
            );
        }
    }
    // a tap is checked before anything runs: a buffer that is not one, the workspace, a step past the last, a tensor of another shape
    let case = &plan["cases"].as_array().unwrap()[0];
    let values = devscalars(case);
    let mut exts = upload(&gpu.stream, case);
    let mut wrong = gpu.stream.alloc_zeros::<u8>(4).unwrap();
    let one = [1usize];
    for (buffer, after, message) in [
        ("nope", None, "not a buffer"),
        ("workspace", None, "not a buffer"),
        ("t1", Some(99), "but the bucket has"),
        ("t1", None, "shape"),
    ] {
        let mut taps = [Tap { buffer, after, into: typed(&mut wrong, &one, DType::Bf16, true) }];
        let e = run.launch_tapped(&gpu.stream, &mut args(&mut exts, false), &named(&values), &mut taps).unwrap_err();
        assert!(e.to_string().contains(message), "{buffer}: {e}");
    }
    println!("rust plan gate: {tapped} taps byte-identical to Python's");
}

/// `PlanRun::launch_step`: the steps of `residual_chain` launched one by one on the caller's tensors (the intermediate too), each with the binary
/// and the binding of the pinned plan, give Python's outputs of the whole plan byte for byte; a step given a buffer it does not bind, or not given
/// one it binds, is refused.
#[test]
fn the_steps_launched_one_by_one_give_the_bytes_of_the_plan() {
    let gpu = Gpu::new();
    let plan = plan_named("residual_chain");
    let lp = load(&plan);
    let run = lp.instantiate_all(&gpu.stream).unwrap();
    for case in plan["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let bucket: Vec<(String, u64)> =
            case["bucket"].as_object().unwrap().iter().map(|(n, v)| (n.clone(), v.as_u64().unwrap())).collect();
        let values: Vec<(&str, u64)> = bucket.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        let at = lp.bucket(&values).unwrap();
        let mut exts = upload(&gpu.stream, case);
        // the intermediates: buffers of the arena, given here as tensors of their own
        for (i, b) in lp.buffers().iter().enumerate().filter(|(_, b)| b.role == BufferRole::Intermediate) {
            let shape = at.shape_of(i).unwrap().to_vec();
            let slice = gpu.stream.alloc_zeros::<u8>(shape.iter().product::<usize>() * b.dtype.size()).unwrap();
            exts.push(Ext {
                name: b.name.clone(),
                role: "intermediate".to_owned(),
                dtype: b.dtype,
                shape,
                slice,
                input: None,
                expect: None,
            });
        }
        for (si, step) in at.steps.iter().enumerate() {
            let bound: Vec<&str> = step.args.iter().map(|a| a.binding.as_str()).collect();
            let mut given: Vec<(&str, TensorArg<'_>)> = exts
                .iter_mut()
                .filter(|e| bound.contains(&e.name.as_str()))
                .map(|e| (e.name.as_str(), typed(&mut e.slice, &e.shape, e.dtype, true)))
                .collect();
            run.launch_step(&gpu.stream, &values, si, &mut given).unwrap_or_else(|e| panic!("{id} step {si}: {e}"));
        }
        gpu.stream.synchronize().unwrap();
        for e in exts.iter().filter(|e| e.expect.is_some()) {
            let got = gpu.stream.clone_dtoh(&e.slice).unwrap();
            assert!(got == *e.expect.as_ref().unwrap(), "{id}: '{}' launched step by step is not the plan's", e.name);
        }
    }
    let case = &plan["cases"].as_array().unwrap()[0];
    let values: Vec<(String, u64)> =
        case["bucket"].as_object().unwrap().iter().map(|(n, v)| (n.clone(), v.as_u64().unwrap())).collect();
    let values: Vec<(&str, u64)> = values.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let mut exts = upload(&gpu.stream, case);
    let mut given = args(&mut exts, true); // x, y and residual: step 0 binds x and residual only
    let e = run.launch_step(&gpu.stream, &values, 0, &mut given).unwrap_err();
    assert!(e.to_string().contains("does not bind the buffer 'y'"), "{e}");
    given.retain(|(n, _)| *n == "x");
    let e = run.launch_step(&gpu.stream, &values, 0, &mut given).unwrap_err();
    assert!(e.to_string().contains("which is not given"), "{e}");
}

/// The outputs of a plan whose graph names them (`residual_chain`: `y` and `residual`) are given by those names or by `out0`, `out1` (the names
/// of before), with the same bytes; both names of one output are refused as one external given twice.
#[test]
fn a_named_output_is_given_by_its_name_or_by_out_i() {
    let gpu = Gpu::new();
    let plan = plan_named("residual_chain");
    let lp = load(&plan);
    assert_eq!(lp.outputs(), ["y", "residual"]);
    let run = lp.instantiate_all(&gpu.stream).unwrap();
    let case = &plan["cases"].as_array().unwrap()[0];
    for legacy in [false, true] {
        let mut exts = upload(&gpu.stream, case);
        let mut given = args(&mut exts, false);
        if legacy {
            for (name, _) in given.iter_mut() {
                *name = match *name {
                    "y" => "out0",
                    "residual" => "out1",
                    other => other,
                };
            }
        }
        run.launch(&gpu.stream, &mut given, &[]).unwrap_or_else(|e| panic!("legacy {legacy}: {e}"));
        drop(given);
        gpu.stream.synchronize().unwrap();
        for e in exts.iter().filter(|e| e.expect.is_some()) {
            let got = gpu.stream.clone_dtoh(&e.slice).unwrap();
            assert!(got == *e.expect.as_ref().unwrap(), "legacy {legacy}: '{}' is not Python's", e.name);
        }
    }
    let mut exts = upload(&gpu.stream, case);
    let extra = gpu.stream.alloc_zeros::<u8>(exts[1].slice.len()).unwrap();
    let mut extra = Ext {
        name: "out0".to_owned(),
        role: "output".to_owned(),
        dtype: exts[1].dtype,
        shape: exts[1].shape.clone(),
        slice: extra,
        input: None,
        expect: None,
    };
    let mut given = args(&mut exts, false);
    given.push(("out0", typed(&mut extra.slice, &extra.shape, extra.dtype, true)));
    let e = run.launch(&gpu.stream, &mut given, &[]).unwrap_err();
    assert!(e.to_string().contains("given twice"), "{e}");
}

/// One graph per bucket, replayed with the inputs and the device scalar of every case of its bucket written between replays (the graph holds
/// addresses; the contents and the slots change); a launch of the same run with other values in between changes nothing the graph reads.
#[test]
fn a_captured_plan_replays_new_inputs_and_new_device_scalars() {
    let gpu = Gpu::new();
    let mut replays = 0;
    for plan in plans() {
        let lp = load(&plan);
        let run = lp.instantiate_all(&gpu.stream).unwrap();
        let all = plan["cases"].as_array().unwrap();
        let mut buckets: Vec<&Value> = Vec::new();
        for c in all {
            if !buckets.contains(&&c["bucket"]) {
                buckets.push(&c["bucket"]);
            }
        }
        for bucket in buckets {
            let group: Vec<&Value> = all.iter().filter(|c| c["bucket"] == *bucket).collect();
            if group.len() < 2 {
                continue;
            }
            let first = group[0];
            let mut exts = upload(&gpu.stream, first);
            let mut other = upload(&gpu.stream, group[1]); // for the launch in between
            let names: Vec<(String, String)> = exts.iter().map(|e| (e.name.clone(), e.role.clone())).collect();
            let mut graph = run.capture(&gpu.stream, args(&mut exts, true), &named(&devscalars(first))).unwrap();
            // replay the cases in the order last, ..., first, and the last again after a launch of the second with its own device scalars
            let mut order: Vec<&Value> = group.iter().rev().copied().collect();
            order.push(group[group.len() - 1]);
            for (k, case) in order.iter().enumerate() {
                let id = case["id"].as_str().unwrap();
                let inputs = upload_inputs(case);
                for (name, role) in &names {
                    match role.as_str() {
                        "input" | "inout" => {
                            gpu.stream.memcpy_htod(&inputs[name], &mut graph.external_mut(name).unwrap()).unwrap()
                        }
                        "output" => gpu.stream.memset_zeros(&mut graph.external_mut(name).unwrap()).unwrap(),
                        _ => {}
                    }
                }
                for (name, value) in devscalars(case) {
                    graph.set_devscalar(&name, value).unwrap();
                }
                if k == order.len() - 1 {
                    // a launch of the run with other tensors and other device scalars: it writes the slots, and the graph writes its own back
                    run.launch(&gpu.stream, &mut args(&mut other, false), &named(&devscalars(group[1]))).unwrap();
                }
                graph.replay(&gpu.stream).unwrap_or_else(|e| panic!("{id}: {e}"));
                gpu.stream.synchronize().unwrap();
                for t in case["externals"].as_array().unwrap().iter().filter(|t| t.get("expect").is_some()) {
                    let name = t["name"].as_str().unwrap();
                    let expect = read(t["expect"].as_str().unwrap());
                    let got = gpu.stream.clone_dtoh(&graph.external(name).unwrap()).unwrap();
                    assert!(got == expect, "{id}: replay {k}: '{name}' differs in {} bytes", differing(&got, &expect));
                }
                replays += 1;
            }
            drop(graph);
            // and the launch in between computed its own case
            gpu.stream.synchronize().unwrap();
            for e in other.iter().filter(|e| e.expect.is_some()) {
                assert!(
                    gpu.stream.clone_dtoh(&e.slice).unwrap() == *e.expect.as_ref().unwrap(),
                    "the launch in between"
                );
            }
        }
    }
    assert!(replays >= 10, "{replays}");
    println!(
        "rust plan gate: {replays} replays of graphs with new inputs and device scalars written between them matched Python's replay"
    );
}

fn upload_inputs(case: &Value) -> std::collections::HashMap<String, Vec<u8>> {
    case["externals"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t.get("input").map(|p| (t["name"].as_str().unwrap().to_owned(), read(p.as_str().unwrap()))))
        .collect()
}

/// Two launches of one run on two other streams than the arena's, back to back with no synchronization: the arena orders them, and both
/// are Python's bytes.
#[test]
fn two_launches_of_one_run_on_two_streams_do_not_disturb_each_other() {
    let gpu = Gpu::new();
    let plan = plan_named("block");
    let run = load(&plan).instantiate_all(&gpu.stream).unwrap();
    let cases: Vec<&Value> = plan["cases"].as_array().unwrap().iter().filter(|c| c["bucket"]["M"] == 128).collect();
    let (s1, s2) = (gpu.ctx.new_stream().unwrap(), gpu.ctx.new_stream().unwrap());
    for round in 0..5 {
        let mut a = upload(&s1, cases[0]);
        let mut b = upload(&s2, cases[1]);
        s1.synchronize().unwrap();
        s2.synchronize().unwrap();
        run.launch(&s1, &mut args(&mut a, false), &named(&devscalars(cases[0]))).unwrap();
        run.launch(&s2, &mut args(&mut b, false), &named(&devscalars(cases[1]))).unwrap();
        for (exts, stream) in [(&a, &s1), (&b, &s2)] {
            stream.synchronize().unwrap();
            for e in exts.iter().filter(|e| e.expect.is_some()) {
                assert!(
                    stream.clone_dtoh(&e.slice).unwrap() == *e.expect.as_ref().unwrap(),
                    "round {round}: '{}'",
                    e.name
                );
            }
        }
    }
    println!("rust plan gate: two launches of one run on two streams, back to back, 5 rounds: both are Python's bytes");
}

/// The order of the runs of one instantiation (`docs/plan-schema-1.md`, section 9: two executions that use one arena must not overlap on the
/// device): two threads at once, one launching a case of one bucket 12 times on its stream (each into an output of its own), the other replaying
/// the graph of a case of another bucket 12 times on its stream (copying the output after each replay), with no synchronization between them.
/// They share the arena; every one of the 24 results is Python's.
#[test]
fn launches_and_replays_from_two_threads_on_two_streams_run_one_after_the_other() {
    fn shared<T: Send + Sync>(t: T) -> T {
        t
    }
    let gpu = Gpu::new();
    let plan = plan_named("block");
    let run = shared(load(&plan).instantiate_all(&gpu.stream).unwrap());
    let cases = plan["cases"].as_array().unwrap();
    let (a, b) = (
        cases.iter().find(|c| c["bucket"]["M"] == 128).unwrap(),
        cases.iter().find(|c| c["bucket"]["M"] == 32).unwrap(),
    );
    let rounds = 12;
    let expect = |case: &Value| {
        let t = case["externals"].as_array().unwrap().iter().find(|t| t["name"] == "out0").unwrap();
        read(t["expect"].as_str().unwrap())
    };
    for round in 0..3 {
        let (got_a, got_b) = std::thread::scope(|scope| {
            let launcher = scope.spawn(|| {
                let s = gpu.ctx.new_stream().unwrap();
                let mut exts = upload(&s, a);
                let out = exts.iter().position(|e| e.name == "out0").unwrap();
                let bytes = exts[out].slice.len();
                let mut outs = Vec::new();
                for _ in 0..rounds {
                    exts[out].slice = s.alloc_zeros::<u8>(bytes).unwrap();
                    run.launch(&s, &mut args(&mut exts, false), &named(&devscalars(a))).unwrap();
                    outs.push(std::mem::replace(&mut exts[out].slice, s.alloc_zeros::<u8>(1).unwrap()));
                }
                s.synchronize().unwrap();
                outs.iter().map(|o| s.clone_dtoh(o).unwrap()).collect::<Vec<_>>()
            });
            let replayer = scope.spawn(|| {
                let s = gpu.ctx.new_stream().unwrap();
                let mut exts = upload(&s, b);
                let mut copies: Vec<CudaSlice<u8>> = Vec::new();
                {
                    let mut graph = run.capture(&s, args(&mut exts, false), &named(&devscalars(b))).unwrap();
                    for _ in 0..rounds {
                        graph.replay(&s).unwrap();
                        let view = graph.external("out0").unwrap();
                        let mut copy = s.alloc_zeros::<u8>(view.len()).unwrap();
                        s.memcpy_dtod(&view, &mut copy).unwrap();
                        copies.push(copy);
                    }
                }
                s.synchronize().unwrap();
                copies.iter().map(|o| s.clone_dtoh(o).unwrap()).collect::<Vec<_>>()
            });
            (launcher.join().unwrap(), replayer.join().unwrap())
        });
        let (want_a, want_b) = (expect(a), expect(b));
        let wrong_a = got_a.iter().filter(|g| **g != want_a).count();
        let wrong_b = got_b.iter().filter(|g| **g != want_b).count();
        assert_eq!((wrong_a, wrong_b), (0, 0), "round {round}: results that are not Python's (launches, replays)");
    }
    println!(
        "rust plan gate: launches (M=128) and graph replays (M=32) of one run from two threads on two streams, {rounds} each, 3 rounds: every \
         result is Python's"
    );
}

/// `PlanRun::capture` holds the arena's lock from the beginning of the capture to its end: one thread captures the plan on the arena's own
/// stream (and replays the graph) in a loop while another launches it on another stream in a loop. A launch makes the arena's stream wait for
/// its end; inside a capture of that stream that wait was `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`, then the capture invalidated (and, in some
/// runs, a crash of the driver). Now the launch waits for the capture to end: no capture and no launch fails, and the results are Python's.
#[test]
fn captures_on_the_arena_stream_and_launches_from_another_thread_do_not_disturb_each_other() {
    fn shared<T: Send + Sync>(t: T) -> T {
        t
    }
    let gpu = Gpu::new();
    let plan = plan_named("block");
    let run = shared(load(&plan).instantiate_all(&gpu.stream).unwrap());
    let cases = plan["cases"].as_array().unwrap();
    let (a, b) = (
        cases.iter().find(|c| c["bucket"]["M"] == 128).unwrap(),
        cases.iter().find(|c| c["bucket"]["M"] == 32).unwrap(),
    );
    let expect = |case: &Value| {
        let t = case["externals"].as_array().unwrap().iter().find(|t| t["name"] == "out0").unwrap();
        read(t["expect"].as_str().unwrap())
    };
    let rounds = 200;
    // everything but the captures, the replays and the launches is done before the two threads start and after they end: an allocation or a
    // copy of the test's own while the arena's stream captures is not what is held here
    let s = gpu.ctx.new_stream().unwrap();
    let mut launched = upload(&s, a);
    let mut captured = upload(&gpu.stream, b);
    s.synchronize().unwrap();
    gpu.stream.synchronize().unwrap();
    let start = std::sync::Barrier::new(2);
    let (capture_errors, launch_errors) = std::thread::scope(|scope| {
        let capturer = scope.spawn(|| {
            let mut errors = Vec::new();
            start.wait();
            for _ in 0..rounds {
                let replayed = run
                    .capture(&gpu.stream, args(&mut captured, false), &named(&devscalars(b)))
                    .and_then(|mut graph| graph.replay(&gpu.stream));
                if let Err(e) = replayed {
                    errors.push(e.to_string());
                }
            }
            errors
        });
        let launcher = scope.spawn(|| {
            let mut errors = Vec::new();
            start.wait();
            for _ in 0..rounds {
                if let Err(e) = run.launch(&s, &mut args(&mut launched, false), &named(&devscalars(a))) {
                    errors.push(e.to_string());
                }
            }
            errors
        });
        (capturer.join().unwrap(), launcher.join().unwrap())
    });
    s.synchronize().unwrap();
    gpu.stream.synchronize().unwrap();
    let out = |exts: &[Ext], stream: &Arc<CudaStream>| {
        let e = exts.iter().find(|e| e.name == "out0").unwrap();
        stream.clone_dtoh(&e.slice).unwrap()
    };
    let launched_wrong = usize::from(out(&launched, &s) != expect(a));
    let captured_wrong = usize::from(out(&captured, &gpu.stream) != expect(b));
    println!(
        "rust plan gate: {rounds} captures (each replayed) on the arena's stream and {rounds} launches on another stream, from two threads: \
         {} captures failed, {} launches failed; the last replay wrong: {captured_wrong}, the last launch wrong: {launched_wrong}",
        capture_errors.len(),
        launch_errors.len()
    );
    for (what, errors) in [("capture", &capture_errors), ("launch", &launch_errors)] {
        for e in errors.iter().take(4) {
            println!("  {what} error: {e}");
        }
    }
    let first = |errors: &[String]| errors.first().cloned().unwrap_or_default();
    assert!(
        capture_errors.is_empty(),
        "{} captures failed, the first: {}",
        capture_errors.len(),
        first(&capture_errors)
    );
    assert!(launch_errors.is_empty(), "{} launches failed, the first: {}", launch_errors.len(), first(&launch_errors));
    assert_eq!((captured_wrong, launched_wrong), (0, 0));
}

/// A copy of an exported plan directory, under the gate directory.
fn copy_dir(plan: &Value, name: &str) -> PathBuf {
    let from = gate_dir().join(plan["dir"].as_str().unwrap());
    let to = gate_dir().join("tmp").join(name);
    let _ = std::fs::remove_dir_all(&to);
    std::fs::create_dir_all(&to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }
    to
}

fn refused_load(path: &Path, pin: Pin<'_>, fragment: &str) {
    match LaunchPlan::load(path, pin) {
        Ok(_) => panic!("{path:?} loads, but it must be refused ({fragment})"),
        Err(e) => assert!(e.to_string().contains(fragment), "{fragment:?} is not in: {e}"),
    }
}

/// What is refused, and that a refusal launches nothing: the output keeps the bytes it had (a sentinel), not even a device scalar is
/// written (the next launch's own scalar would hide that, so the graph of the run, which reads the slot, is replayed against its bytes).
#[test]
fn refusals_launch_nothing() {
    let gpu = Gpu::new();
    let plan = plan_named("block");
    let digest = plan["digest"].as_str().unwrap();
    let dir = gate_dir().join(plan["dir"].as_str().unwrap());
    let mut refused = 0;
    // -- loading: a wrong pin, a tampered file, a plan of another manifest, a manifest changed under the plan
    assert!(matches!(LaunchPlan::load(&dir, Pin::Digest(&"1".repeat(64))), Err(Error::PlanPinMismatch { .. })));
    assert!(matches!(LaunchPlan::load(&dir, Pin::Digest("not a digest")), Err(Error::BadPin(_))));
    let tampered = copy_dir(&plan, "tampered");
    let text = std::fs::read_to_string(tampered.join("plan.json")).unwrap();
    let edited = text.replacen("\"offset\": 256", "\"offset\": 512", 1);
    assert_ne!(text, edited);
    std::fs::write(tampered.join("plan.json"), edited).unwrap();
    refused_load(&tampered, Pin::Digest(digest), "but the pin is");
    refused_load(&tampered, Pin::TrustDirectory, "was edited");
    let other = copy_dir(&plan, "other-manifest");
    let mut doc: Value = serde_json::from_str(&text).unwrap();
    doc["manifest_digest"] = Value::from("0".repeat(64));
    doc["plan_digest"] = Value::from(plan_digest(&doc));
    std::fs::write(other.join("plan.json"), serde_json::to_string(&doc).unwrap()).unwrap();
    refused_load(&other, Pin::Digest(doc["plan_digest"].as_str().unwrap()), "built against the manifest with digest");
    refused_load(&other, Pin::TrustDirectory, "built against the manifest with digest");
    let changed = copy_dir(&plan, "manifest-changed");
    let mut manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(changed.join("manifest.json")).unwrap()).unwrap();
    manifest["kernels"][0]["shared"] = Value::from(manifest["kernels"][0]["shared"].as_u64().unwrap() + 16);
    std::fs::write(changed.join("manifest.json"), serde_json::to_string(&manifest).unwrap()).unwrap();
    refused_load(&changed, Pin::Digest(digest), "built against the manifest with digest");
    refused += 7;
    let _ = std::fs::remove_dir_all(gate_dir().join("tmp"));

    // -- calls: the run is instantiated for M = 64 only
    let lp = load(&plan);
    let run = lp.instantiate(&gpu.stream, &[("M", 64)]).unwrap();
    assert!(matches!(
        lp.instantiate(&gpu.stream, &[("M", 48)]),
        Err(Error::Precondition { kind: PreconditionKind::Bucket, .. })
    ));
    let case = plan["cases"].as_array().unwrap().iter().find(|c| c["bucket"]["M"] == 64).unwrap();
    let values = devscalars(case);
    let mut exts = upload(&gpu.stream, case);
    let sentinel = vec![0x3fu8; 64 * 256 * 2];
    let out = exts.iter().position(|e| e.name == "out0").unwrap();
    gpu.stream.memcpy_htod(&sentinel, &mut exts[out].slice).unwrap();
    // the graph of the run, captured with the case's device scalar: what it computes shows whether a slot was written
    let mut graph_exts = upload(&gpu.stream, case);
    let mut graph = run.capture(&gpu.stream, args(&mut graph_exts, false), &named(&values)).unwrap();
    graph.replay(&gpu.stream).unwrap();
    let mut other_bucket =
        upload(&gpu.stream, plan["cases"].as_array().unwrap().iter().find(|c| c["bucket"]["M"] == 32).unwrap());
    let x48 = gpu.stream.alloc_zeros::<u8>(48 * 256 * 2).unwrap();
    let x64_f16 = gpu.stream.alloc_zeros::<u8>(64 * 256 * 2).unwrap();
    let (s48, s64, s_bad) = ([48usize, 256], [64usize, 256], [64usize, 128]);
    let x_bad = gpu.stream.alloc_zeros::<u8>(64 * 256 * 2).unwrap();
    let wrong_devscalar = [("n", DevScalar::Int(1 << 40))];
    let check_untouched = |what: &str, exts: &[Ext]| {
        gpu.stream.synchronize().unwrap();
        assert_eq!(gpu.stream.clone_dtoh(&exts[out].slice).unwrap(), sentinel, "{what}: the output was written");
    };
    for (what, kind) in [
        ("a missing external", None),
        ("an extra external", None),
        ("an unknown external", None),
        ("a static size that is not the plan's", Some(PreconditionKind::StaticSize)),
        ("a dtype that is not the plan's", Some(PreconditionKind::Dtype)),
        ("a size that is no bucket", Some(PreconditionKind::Bucket)),
    ] {
        let mut a = args(&mut exts, false);
        match what {
            "a missing external" => a.retain(|(n, _)| *n != "w_up"),
            "an extra external" => a.push(("t0", TensorRef::<Bf16>::from_cuda(&x_bad, &s64).unwrap().into_arg())),
            "an unknown external" => {
                a.push(("nothing", TensorRef::<Bf16>::from_cuda(&x_bad, &s64).unwrap().into_arg()))
            }
            "a static size that is not the plan's" => {
                a[0].1 = TensorRef::<Bf16>::from_cuda(&x_bad, &s_bad).unwrap().into_arg()
            }
            "a dtype that is not the plan's" => {
                a[0].1 = TensorRef::<F16>::from_cuda(&x64_f16, &s64).unwrap().into_arg()
            }
            _ => a[0].1 = TensorRef::<Bf16>::from_cuda(&x48, &s48).unwrap().into_arg(),
        }
        let e = run.launch(&gpu.stream, &mut a, &named(&values)).expect_err(what);
        if let Some(kind) = kind {
            assert_eq!(e.precondition_kind(), Some(kind), "{what}: {e}");
        }
        drop(a);
        check_untouched(what, &exts);
        refused += 1;
    }
    // a bucket of the plan that the run was not instantiated for
    let e = run.launch(&gpu.stream, &mut args(&mut other_bucket, false), &named(&devscalars(case))).unwrap_err();
    assert_eq!(e.precondition_kind(), Some(PreconditionKind::Bucket), "{e}");
    assert!(e.to_string().contains("instantiated for"), "{e}");
    // a device scalar that does not fit its dtype, one missing, one unknown
    for (what, values) in [
        ("a device scalar that does not fit i32", wrong_devscalar.to_vec()),
        ("a missing device scalar", Vec::new()),
        ("an unknown device scalar", vec![("n", DevScalar::Int(3)), ("m", DevScalar::Int(3))]),
        ("an f32 for an i32 device scalar", vec![("n", DevScalar::F32(3.0))]),
    ] {
        run.launch(&gpu.stream, &mut args(&mut exts, false), &values).expect_err(what);
        check_untouched(what, &exts);
        refused += 1;
    }
    // an output given read-only
    {
        let (before, rest) = exts.split_at_mut(out);
        let (o, after) = rest.split_first_mut().unwrap();
        let mut a = args(before, false);
        a.extend(args(after, false));
        a.push(("out0", TensorRef::<Bf16>::from_cuda(&o.slice, &o.shape).unwrap().into_arg()));
        let e = run.launch(&gpu.stream, &mut a, &named(&values)).unwrap_err();
        assert_eq!(e.precondition_kind(), Some(PreconditionKind::Access), "{e}");
    }
    check_untouched("a read-only output", &exts);
    refused += 1;
    // an output that overlaps an input (regions of one buffer: the safe tensors of slices cannot say it)
    {
        let x = exts.iter().position(|e| e.name == "x").unwrap();
        let addr = exts[x].slice.device_ptr(&gpu.stream).0;
        let shape = exts[x].shape.clone();
        let mut a: Vec<(&str, TensorArg<'_>)> = Vec::new();
        // SAFETY: the slice is alive for the whole call and the call is refused before anything is launched
        let (rx, ro) =
            unsafe { (DevicePtr::from_raw(addr, 64 * 256 * 2, 0), DevicePtr::from_raw(addr, 64 * 256 * 2, 0)) };
        let rest: Vec<usize> = (0..exts.len()).filter(|i| *i != x && *i != out).collect();
        for e in exts.iter_mut().enumerate().filter(|(i, _)| rest.contains(i)).map(|(_, e)| e) {
            let write = e.role == "output";
            a.push((e.name.as_str(), typed(&mut e.slice, &e.shape, e.dtype, write)));
        }
        a.push(("x", TensorArg::new(rx, &shape, DType::Bf16, Access::In).unwrap()));
        a.push(("out0", TensorArg::new(ro, &shape, DType::Bf16, Access::InOut).unwrap()));
        let e = run.launch(&gpu.stream, &mut a, &named(&values)).unwrap_err();
        assert_eq!(e.precondition_kind(), Some(PreconditionKind::Alias), "{e}");
    }
    check_untouched("an output that overlaps an input", &exts);
    refused += 1;
    // the graph reads the slot the refused calls did not write: it still computes the case
    gpu.stream.memset_zeros(&mut graph.external_mut("out0").unwrap()).unwrap();
    graph.replay(&gpu.stream).unwrap();
    gpu.stream.synchronize().unwrap();
    let expect = exts[out].expect.clone().unwrap();
    assert_eq!(gpu.stream.clone_dtoh(&graph.external("out0").unwrap()).unwrap(), expect);
    // and the same call, fixed, runs
    run.launch(&gpu.stream, &mut args(&mut exts, false), &named(&values)).unwrap();
    gpu.stream.synchronize().unwrap();
    assert_eq!(gpu.stream.clone_dtoh(&exts[out].slice).unwrap(), expect);
    println!("rust plan gate: {refused} refusals (loading and calls), none of which launched anything");
}

/// The per-thread default stream is refused by a plan as by a single launch: as the stream an instantiation allocates its arena on (which
/// orders every launch of the run), as the stream of a launch or a capture, and as the owner of an external.
#[test]
fn the_per_thread_default_stream_is_refused_by_a_plan() {
    let gpu = Gpu::new();
    let plan = plan_named("block");
    let lp = load(&plan);
    let per_thread = gpu.ctx.per_thread_stream();
    let is_refused = |e: Error, what: &str| {
        assert!(matches!(&e, Error::Launch(why) if why.contains("per-thread default stream")), "{what}: {e:?}");
    };
    is_refused(lp.instantiate_all(&per_thread).unwrap_err(), "an instantiation on the per-thread stream");
    let run = lp.instantiate(&gpu.stream, &[("M", 64)]).unwrap();
    let case = plan["cases"].as_array().unwrap().iter().find(|c| c["bucket"]["M"] == 64).unwrap();
    let values = devscalars(case);
    let mut exts = upload(&gpu.stream, case);
    is_refused(run.launch(&per_thread, &mut args(&mut exts, false), &named(&values)).unwrap_err(), "a launch on it");
    is_refused(run.capture(&per_thread, args(&mut exts, false), &named(&values)).unwrap_err(), "a capture on it");
    let mut owned = upload(&per_thread, case);
    per_thread.synchronize().unwrap();
    is_refused(run.launch(&gpu.stream, &mut args(&mut owned, false), &named(&values)).unwrap_err(), "externals of it");
    gpu.ctx.synchronize().unwrap();
    for e in exts.iter().chain(&owned).filter(|e| e.role == "output") {
        assert!(
            gpu.stream.clone_dtoh(&e.slice).unwrap().iter().all(|b| *b == 0),
            "a refused launch wrote '{}'",
            e.name
        );
    }
    println!(
        "rust plan gate: the per-thread default stream is refused (instantiate, launch, capture, the owner of an external)"
    );
}

/// The used memory of the device's default memory pool (what cudarc allocates from) and the free memory of the device.
fn memory(ctx: &CudaContext) -> (u64, u64, usize) {
    ctx.bind_to_thread().unwrap();
    let (free, _) = cudarc::driver::result::mem_get_info().unwrap();
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    let (mut used, mut reserved) = (0u64, 0u64);
    // SAFETY (the driver calls below): the out-pointers are live locals; the device is the context's.
    unsafe {
        assert_eq!(sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()), sys::CUresult::CUDA_SUCCESS);
        let attr = sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT;
        assert_eq!(sys::cuMemPoolGetAttribute(pool, attr, (&raw mut used).cast()), sys::CUresult::CUDA_SUCCESS);
        let attr = sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT;
        assert_eq!(sys::cuMemPoolGetAttribute(pool, attr, (&raw mut reserved).cast()), sys::CUresult::CUDA_SUCCESS);
    }
    (used, reserved, free)
}

/// Nothing is allocated after instantiation: the memory of the device is the same after 100 launches, and after 100 replays (the
/// instantiation of the graph itself is measured on its own); and the host time of a launch and of a replay at M = 64 (reported).
#[test]
fn launches_and_replays_allocate_nothing_and_their_time_is_reported() {
    let gpu = Gpu::new();
    for name in ["block", "deep4"] {
        let plan = plan_named(name);
        let lp = load(&plan);
        let case = plan["cases"].as_array().unwrap().iter().find(|c| c["bucket"]["M"] == 64).unwrap();
        let values = devscalars(case);
        let mut exts = upload(&gpu.stream, case);
        let mut graph_exts = upload(&gpu.stream, case);
        gpu.stream.synchronize().unwrap();
        let start = memory(&gpu.ctx);
        let run = lp.instantiate(&gpu.stream, &[("M", 64)]).unwrap();
        gpu.stream.synchronize().unwrap();
        let instantiated = memory(&gpu.ctx);
        let mut launch_time = Duration::ZERO;
        let n = 100;
        for _ in 0..n {
            let mut a = args(&mut exts, false);
            let t = Instant::now();
            run.launch(&gpu.stream, &mut a, &named(&values)).unwrap();
            launch_time += t.elapsed();
        }
        gpu.stream.synchronize().unwrap();
        let launched = memory(&gpu.ctx);
        let t = Instant::now();
        let mut graph = run.capture(&gpu.stream, args(&mut graph_exts, false), &named(&values)).unwrap();
        let capture_time = t.elapsed();
        gpu.stream.synchronize().unwrap();
        let captured = memory(&gpu.ctx);
        let mut replay_time = Duration::ZERO;
        for _ in 0..n {
            let t = Instant::now();
            graph.replay(&gpu.stream).unwrap();
            replay_time += t.elapsed();
        }
        gpu.stream.synchronize().unwrap();
        let replayed = memory(&gpu.ctx);
        // device time per forward, launched and replayed
        let t = Instant::now();
        for _ in 0..n {
            run.launch(&gpu.stream, &mut args(&mut exts, false), &named(&values)).unwrap();
        }
        gpu.stream.synchronize().unwrap();
        let launch_wall = t.elapsed();
        let t = Instant::now();
        for _ in 0..n {
            graph.replay(&gpu.stream).unwrap();
        }
        gpu.stream.synchronize().unwrap();
        let replay_wall = t.elapsed();
        // a launch behind queued work: it writes its device scalar from pinned memory, so the host does not wait for the queue (reported: its
        // host time is that of a launch, not that of the queue)
        for _ in 0..n {
            graph.replay(&gpu.stream).unwrap();
        }
        let t = Instant::now();
        run.launch(&gpu.stream, &mut args(&mut exts, false), &named(&values)).unwrap();
        let behind = t.elapsed();
        let t = Instant::now();
        gpu.stream.synchronize().unwrap();
        let queued = t.elapsed();
        println!(
            "rust plan gate [{name}, M=64]: a launch behind {n} queued replays takes {:.1} us of the host; the queue then takes {:.1} us more",
            behind.as_secs_f64() * 1e6,
            queued.as_secs_f64() * 1e6
        );
        for e in exts.iter().filter(|e| e.expect.is_some()) {
            assert!(gpu.stream.clone_dtoh(&e.slice).unwrap() == *e.expect.as_ref().unwrap(), "{name}: '{}'", e.name);
        }
        let out = graph.external("out0").unwrap();
        let expect = case["externals"].as_array().unwrap().iter().find(|t| t["name"] == "out0").unwrap();
        assert_eq!(gpu.stream.clone_dtoh(&out).unwrap(), read(expect["expect"].as_str().unwrap()));
        let steps = lp.bucket(&[("M", 64)]).unwrap().steps.len();
        println!(
            "rust plan gate [{name}, M=64, {steps} steps]: memory (pool used, pool reserved, device free) at start {start:?}, instantiated \
             {instantiated:?} (arena {} bytes), after {n} launches {launched:?}, after the capture {captured:?}, after {n} replays {replayed:?}",
            lp.arena_bytes()
        );
        println!(
            "rust plan gate [{name}, M=64]: host time per launch {:.1} us, per replay {:.1} us (the capture {:.1} us); wall time per forward with \
             the device: launched {:.1} us, replayed {:.1} us",
            launch_time.as_secs_f64() * 1e6 / n as f64,
            replay_time.as_secs_f64() * 1e6 / n as f64,
            capture_time.as_secs_f64() * 1e6,
            launch_wall.as_secs_f64() * 1e6 / n as f64,
            replay_wall.as_secs_f64() * 1e6 / n as f64
        );
        assert_eq!(launched, instantiated, "{name}: {n} launches allocated device memory");
        assert_eq!(replayed, captured, "{name}: {n} replays allocated device memory");
        drop(graph);
    }
}
