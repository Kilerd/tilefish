//! The Rust host decides every call as Python's `replay` does, from the manifest alone.
//!
//! `data/python_launches.json` is made by `python/tests/export_rust_vectors.py`: for each binary of the `sm_103` fixture, a valid call
//! per size and every single change of one input (dtype, rank, an axis made larger, twice as large or empty, a misaligned address, a
//! tensor too large for 32-bit offsets, an in-place tensor aliased, a bucketed dimension moved), and what Python's `runtime._prepare`
//! decided for each on fake tensors: the grid and the `kernelParams` values of a call that runs, or the kind of the refusal. Python
//! has the declarations; Rust has only the manifest (the declared shape of every pointer and the constraints of every dimension).
//! `Plan::prepare` must give the same grid and the same parameters for every call that Python accepts, and refuse every call that Python
//! refuses, with the same kind.

mod common;

use common::fixtures;
use tilefish::{Arch, DType, DevicePtr, Manifest, Pin, TensorArg};

#[test]
fn every_launch_is_decided_as_python_decides_it() {
    let text = std::fs::read_to_string(common::data().join("python_launches.json")).unwrap();
    let records: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
    let manifest = Manifest::load(fixtures().join("sm_103"), Pin::TrustDirectory).unwrap();
    let (mut accepted, mut refused) = (0, 0);
    let mut differences = Vec::new();
    let mut stems = std::collections::BTreeSet::new();
    for (n, record) in records.iter().enumerate() {
        let stem = record["stem"].as_str().unwrap();
        stems.insert(stem.to_owned());
        let shapes: Vec<Vec<usize>> = record["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect())
            .collect();
        let tensors: Vec<TensorArg<'_>> = record["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&shapes)
            .map(|(t, shape)| {
                let dtype = DType::from_name(t["dtype"].as_str().unwrap()).unwrap();
                // SAFETY: the address is never dereferenced: `prepare` only looks at the number. The region is as long as the tensor.
                let region = unsafe {
                    DevicePtr::from_raw(
                        t["ptr"].as_u64().unwrap(),
                        shape.iter().product::<usize>() as u64 * dtype.size() as u64,
                        0,
                    )
                };
                TensorArg::new(region, shape, dtype, tilefish::Access::InOut).unwrap()
            })
            .collect();
        let plan = manifest.plan(stem, &[], Arch::sm(103)).unwrap();
        let expect = &record["expect"];
        let mismatch = match (expect["ok"].as_bool().unwrap(), plan.prepare(&tensors, None)) {
            (true, Ok(prepared)) => {
                accepted += 1;
                let grid: Vec<u64> = prepared.grid.iter().map(|g| u64::from(*g)).collect();
                let want_grid: Vec<u64> =
                    expect["grid"].as_array().unwrap().iter().map(|g| g.as_u64().unwrap()).collect();
                let want_params: Vec<u64> =
                    expect["params"].as_array().unwrap().iter().map(|g| g.as_u64().unwrap()).collect();
                (grid != want_grid || prepared.params() != want_params).then(|| {
                    format!(
                        "grid {grid:?} and params {:?}, Python has grid {want_grid:?} and params {want_params:?}",
                        prepared.params()
                    )
                })
            }
            (false, Err(e)) => {
                refused += 1;
                let kind = e.precondition_kind().map_or("other", |k| k.as_str());
                (kind != expect["kind"].as_str().unwrap())
                    .then(|| format!("refused as {kind} ({e}), Python says {}", expect["kind"]))
            }
            (true, Err(e)) => Some(format!("refused ({e}), Python accepts it")),
            (false, Ok(_)) => Some(format!("accepted, Python refuses it as {}", expect["kind"])),
        };
        if let Some(m) = mismatch {
            differences.push(format!("record {n} ({stem}, shapes {shapes:?}): {m}"));
        }
    }
    println!(
        "compared {} launches of {} binaries with Python: {accepted} accepted, {refused} refused, {} differences",
        records.len(),
        stems.len(),
        differences.len()
    );
    assert!(
        differences.is_empty(),
        "{} differences, the first ones:\n{}",
        differences.len(),
        differences[..differences.len().min(10)].join("\n")
    );
    assert!(records.len() > 400 && accepted > 80 && stems.len() == 12);
}
