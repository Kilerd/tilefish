//! The strict reader of a `library` entry decides what Python's decides, on entries of every form.
//!
//! `data/python_library.json` is made by `python/tests/export_rust_vectors.py`: the entries `tilefish._library` writes for GEMMs of bf16, f16
//! and f32 operands, results of their type and of f32, with and without a bias, in the layouts TN and NN, with and without the opt-in of their
//! operand type, and each of them changed in one field (the transposes, `lda`, a type in `dtypes`, the shape of `b`, the dtype of a pointer, an
//! opt-in of another type or written out as `false`, an SM count target in and out of its range, a 16-bit result of f32 operands), with what
//! Python's `validate_entry` decides of each: accepted, with the form it reads (operand type, result type, layout, the opt-ins of bf16 and f16,
//! the SM count target, the opt-in of f32), or refused. Rust reads each in a manifest of its own and must accept exactly those, with the same form
//! (`docs/manifest-schema-3.md`, section 9.2). No GPU.

mod common;

use common::synthetic::read;
use serde_json::Value;
use tilefish::{DType, GemmLayout};

fn vectors() -> Vec<Value> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join("python_library.json");
    serde_json::from_str::<Value>(&std::fs::read_to_string(path).unwrap()).unwrap().as_array().unwrap().clone()
}

#[test]
fn the_reader_accepts_and_refuses_the_entries_python_does_and_reads_the_same_form() {
    let cases = vectors();
    assert!(cases.len() >= 200, "{}", cases.len());
    assert!(
        cases.iter().any(|c| c["accepted"] == true && c["form"][0] == "f32" && c["form"][6] == true),
        "an f32 entry that opts in"
    );
    let (mut accepted, mut refused) = (0, 0);
    for case in &cases {
        let entry = case["entry"].clone();
        let stem = entry["stem"].as_str().unwrap().to_owned();
        match (case["accepted"].as_bool().unwrap(), read(&[], &[entry])) {
            (true, Ok(manifest)) => {
                let e = &manifest.library()[0];
                let form = &case["form"];
                let layout = match e.layout {
                    GemmLayout::Tn => "TN",
                    GemmLayout::Nn => "NN",
                };
                assert_eq!(
                    (
                        e.dtype.name(),
                        e.out_dtype.name(),
                        layout,
                        e.allow_bf16_reduction,
                        e.allow_f16_reduction,
                        e.sm_count_target,
                        e.allow_f32_reduction
                    ),
                    (
                        form[0].as_str().unwrap(),
                        form[1].as_str().unwrap(),
                        form[2].as_str().unwrap(),
                        form[3].as_bool().unwrap(),
                        form[4].as_bool().unwrap(),
                        form[5].as_u64(),
                        form[6].as_bool().unwrap()
                    ),
                    "{stem}"
                );
                assert!(
                    matches!(e.dtype, DType::Bf16 | DType::F16 | DType::F32)
                        && (e.out_dtype == e.dtype || e.out_dtype == DType::F32)
                );
                accepted += 1;
            }
            (false, Err(_)) => refused += 1,
            (true, Err(e)) => panic!("Python accepts {stem} and Rust refuses it: {e}\n{}", case["entry"]),
            (false, Ok(_)) => {
                panic!("Python refuses {stem} ({}) and Rust accepts it\n{}", case["refusal"], case["entry"])
            }
        }
    }
    assert!(accepted >= 10 && refused >= 70, "{accepted} accepted, {refused} refused");
}
