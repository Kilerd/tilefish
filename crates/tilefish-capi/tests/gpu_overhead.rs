//! What the C boundary costs a launch, on a real GPU (built only with `--features gpu-gate`): the same kernel of the committed fixture
//! (`gelu_erf`, for the device's architecture) launched through the crate's `Kernel::launch_raw` and through `tilefish_kernel_launch`, and (with
//! `TILEFISH_GATE_DIR`, the directory `rust_gate.py prepare` filled) its plan `block` run through `PlanRun::launch` and `tilefish_plan_run_launch`,
//! in the same process; the host time per call of each is
//! reported, and the C ABI's must stay within a few microseconds of the crate's.
//!
//! ```text
//! TILEFISH_GATE_DIR=$WORK cargo test -p tilefish-capi --features gpu-gate --test gpu_overhead -- --nocapture
//! ```

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::time::Instant;

use tilefish::cudarc::driver::{CudaContext, DevicePtr as _};
use tilefish::{Arch, DType, DevScalar, DevicePtr, LaunchPlan, Manifest, Pin, TensorArg};
use tilefish_capi::*;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn pin_of(path: &Path, keys: &[&str]) -> String {
    let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    keys.iter().fold(&doc, |v, k| &v[*k]).as_str().unwrap().to_owned()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Host microseconds per call of `f`, the median of 7 rounds of `n` calls.
fn per_call(n: usize, mut f: impl FnMut()) -> f64 {
    let rounds: Vec<f64> = (0..7)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..n {
                f();
            }
            t.elapsed().as_secs_f64() * 1e6 / n as f64
        })
        .collect();
    median(rounds)
}

#[test]
fn the_c_boundary_costs_a_launch_little() {
    assert!(tilefish::driver_available(), "this test needs a GPU");
    let ctx = CudaContext::new(0).unwrap();
    let arch = Arch::of_context(&ctx).unwrap();
    let dir = repo().join(format!("crates/tilefish/tests/fixtures/{arch}"));
    let pin = pin_of(
        &repo().join("crates/tilefish/tests/fixtures/expected.json"),
        &["fixtures", &arch.to_string(), "manifest_digest"],
    );
    let stream = ctx.new_stream().unwrap();
    let count = 4096usize;
    let (x, y) = (stream.alloc_zeros::<u8>(count * 2).unwrap(), stream.alloc_zeros::<u8>(count * 2).unwrap());
    let (xa, ya) = (x.device_ptr(&stream).0, y.device_ptr(&stream).0);
    stream.synchronize().unwrap();

    // the crate
    let manifest = Manifest::load(&dir, Pin::Digest(&pin)).unwrap();
    let kernel = manifest.kernel(&ctx, "gelu_erf", &[], arch).unwrap();
    let shape = [count];
    // SAFETY: the slices outlive every launch (the stream is synchronized before they are dropped), and nothing else uses them.
    let (xr, yr) =
        unsafe { (DevicePtr::from_raw(xa, (count * 2) as u64, 0), DevicePtr::from_raw(ya, (count * 2) as u64, 0)) };
    let raw_stream = stream.cu_stream();
    let rust = per_call(2000, || {
        let args = [
            TensorArg::new(xr, &shape, DType::Bf16, tilefish::Access::In).unwrap(),
            TensorArg::new(yr, &shape, DType::Bf16, tilefish::Access::InOut).unwrap(),
        ];
        // SAFETY: as above; the stream is the kernel's context's.
        unsafe { kernel.launch_raw(raw_stream, &args) }.unwrap();
    });
    stream.synchronize().unwrap();

    // the C ABI, in the same process
    let (cdir, cpin, stem) =
        (CString::new(dir.to_str().unwrap()).unwrap(), CString::new(pin.clone()).unwrap(), c"gelu_erf");
    let (mut m, mut c, mut k) = (null_mut(), null_mut(), null_mut());
    unsafe {
        assert_eq!(tilefish_manifest_load(cdir.as_ptr(), cpin.as_ptr(), &mut m), TILEFISH_OK);
        assert_eq!(tilefish_context_primary(0, &mut c), TILEFISH_OK);
        assert_eq!(
            tilefish_kernel_load(m, c, stem.as_ptr(), std::ptr::null(), 0, std::ptr::null(), &mut k),
            TILEFISH_OK
        );
    }
    let dims = [count as u64];
    let tensor = |address: u64, access: i32| tilefish_tensor {
        address,
        bytes: (count * 2) as u64,
        device: 0,
        dtype: TILEFISH_DTYPE_BF16,
        access,
        rank: 1,
        shape: dims.as_ptr(),
        strides: std::ptr::null(),
        offset: 0,
    };
    let tensors = [tensor(xa, TILEFISH_ACCESS_IN), tensor(ya, TILEFISH_ACCESS_INOUT)];
    let capi = per_call(2000, || {
        let status = unsafe { tilefish_kernel_launch(k, raw_stream.cast(), tensors.as_ptr(), 2, std::ptr::null(), 0) };
        assert_eq!(status, TILEFISH_OK);
    });
    stream.synchronize().unwrap();

    // a plan: the `block` of the gate (`rust_gate.py prepare`, TILEFISH_GATE_DIR: nine steps, a GEMM among them), through the crate and through
    // the C ABI, on the tensors of its first case
    let plan_report = std::env::var_os("TILEFISH_GATE_DIR").map(PathBuf::from).map(|gate| {
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(gate.join("plans.json")).unwrap()).unwrap();
        let entry =
            doc["plans"].as_array().unwrap().iter().find(|p| p["name"] == "block").expect("the gate's plan 'block'");
        let (dir, plan_pin) = (gate.join(entry["dir"].as_str().unwrap()), entry["digest"].as_str().unwrap().to_owned());
        let case = &entry["cases"][0];
        let plan = LaunchPlan::load(&dir, Pin::Digest(&plan_pin)).unwrap();
        let run = plan.instantiate_all(&stream).unwrap();
        // (name, dtype, shape, bytes, written)
        let externals: Vec<(String, DType, Vec<usize>, u64, bool)> = case["externals"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                let dtype = DType::from_name(t["dtype"].as_str().unwrap()).unwrap();
                let shape: Vec<usize> =
                    t["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
                let bytes = (shape.iter().product::<usize>() * dtype.size()) as u64;
                let written = matches!(t["role"].as_str().unwrap(), "output" | "inout");
                (t["name"].as_str().unwrap().to_owned(), dtype, shape, bytes, written)
            })
            .collect();
        let values: Vec<(String, i64)> =
            case["devscalars"].as_object().unwrap().iter().map(|(n, v)| (n.clone(), v.as_i64().unwrap())).collect();
        let memory: Vec<_> = externals.iter().map(|e| stream.alloc_zeros::<u8>(e.3 as usize).unwrap()).collect();
        let addrs: Vec<u64> = memory.iter().map(|s| s.device_ptr(&stream).0).collect();
        stream.synchronize().unwrap();
        let rust_values: Vec<(&str, DevScalar)> =
            values.iter().map(|(n, v)| (n.as_str(), DevScalar::Int(*v))).collect();
        let rust_plan = per_call(500, || {
            let mut args: Vec<(&str, TensorArg<'_>)> = externals
                .iter()
                .zip(&addrs)
                .map(|(e, a)| {
                    // SAFETY: as above.
                    let r = unsafe { DevicePtr::from_raw(*a, e.3, 0) };
                    let access = if e.4 { tilefish::Access::InOut } else { tilefish::Access::In };
                    (e.0.as_str(), TensorArg::new(r, &e.2, e.1, access).unwrap())
                })
                .collect();
            run.launch(&stream, &mut args, &rust_values).unwrap();
        });
        stream.synchronize().unwrap();
        let cplan = CString::new(dir.to_str().unwrap()).unwrap();
        let cplan_pin = CString::new(plan_pin.clone()).unwrap();
        let (mut p, mut r) = (null_mut(), null_mut());
        unsafe {
            assert_eq!(tilefish_plan_load(cplan.as_ptr(), cplan_pin.as_ptr(), &mut p), TILEFISH_OK);
            assert_eq!(tilefish_plan_instantiate(p, c, raw_stream.cast(), std::ptr::null(), 0, &mut r), TILEFISH_OK);
        }
        let names: Vec<CString> = externals.iter().map(|e| CString::new(e.0.clone()).unwrap()).collect();
        let shapes: Vec<Vec<u64>> = externals.iter().map(|e| e.2.iter().map(|d| *d as u64).collect()).collect();
        let named: Vec<tilefish_named_tensor> = externals
            .iter()
            .enumerate()
            .map(|(i, e)| tilefish_named_tensor {
                name: names[i].as_ptr(),
                tensor: tilefish_tensor {
                    address: addrs[i],
                    bytes: e.3,
                    device: 0,
                    dtype: match e.1 {
                        DType::F32 => TILEFISH_DTYPE_F32,
                        DType::Bf16 => TILEFISH_DTYPE_BF16,
                        DType::F16 => TILEFISH_DTYPE_F16,
                        DType::I64 => TILEFISH_DTYPE_I64,
                        DType::U8 => TILEFISH_DTYPE_U8,
                        _ => TILEFISH_DTYPE_I32,
                    },
                    access: if e.4 { TILEFISH_ACCESS_INOUT } else { TILEFISH_ACCESS_IN },
                    rank: shapes[i].len() as u32,
                    shape: shapes[i].as_ptr(),
                    strides: std::ptr::null(),
                    offset: 0,
                },
            })
            .collect();
        let value_names: Vec<CString> = values.iter().map(|(n, _)| CString::new(n.clone()).unwrap()).collect();
        let c_values: Vec<tilefish_devscalar> = values
            .iter()
            .zip(&value_names)
            .map(|((_, v), n)| tilefish_devscalar {
                name: n.as_ptr(),
                kind: TILEFISH_DEVSCALAR_INT,
                f32_value: 0.0,
                int_value: *v,
                bits: 0,
            })
            .collect();
        let capi_plan = per_call(500, || {
            let status = unsafe {
                tilefish_plan_run_launch(
                    r,
                    raw_stream.cast(),
                    named.as_ptr(),
                    named.len(),
                    c_values.as_ptr(),
                    c_values.len(),
                )
            };
            assert_eq!(status, TILEFISH_OK);
        });
        stream.synchronize().unwrap();
        unsafe {
            tilefish_plan_run_destroy(r);
            tilefish_plan_destroy(p);
        }
        (rust_plan, capi_plan)
    });
    unsafe {
        tilefish_kernel_destroy(k);
        tilefish_context_destroy(c);
        tilefish_manifest_destroy(m);
    }
    println!(
        "C boundary: a kernel launch takes {rust:.2} us of host time through the crate, {capi:.2} us through tilefish_kernel_launch"
    );
    if let Some((rust_plan, capi_plan)) = plan_report {
        println!(
            "C boundary: a plan launch takes {rust_plan:.2} us through the crate, {capi_plan:.2} us through tilefish_plan_run_launch"
        );
        assert!(capi_plan < rust_plan + 5.0, "the C boundary costs a plan launch {:.2} us", capi_plan - rust_plan);
    }
    assert!(capi < rust + 3.0, "the C boundary costs a launch {:.2} us", capi - rust);
}
