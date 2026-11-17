//! The strict reader of a `library` entry decides what Python's decides, on entries of every form.
//!
//! `data/python_library.json` is made by `python/tests/export_rust_vectors.py`: the entries `tilefish._library` writes for GEMMs of bf16, f16
//! and f32 operands, results of their type and of f32, with and without a bias, in the layouts TN, NN, TT and NT, with and without the opt-in of
//! their operand type, and each of them changed in one field (the transposes, `lda`, `ldb`, a type in `dtypes`, the shape of `a` or of `b`, the
//! dtype of a pointer, an
//! opt-in of another type or written out as `false`, an SM count target in and out of its range, a 16-bit result of f32 operands), with what
//! Python's `validate_entry` decides of each: accepted, with the form it reads (operand type, result type, layout, the opt-ins of bf16 and f16,
//! the SM count target, the opt-in of f32), or refused. Rust reads each in a manifest of its own and must accept exactly those, with the same form
//! (`docs/en/manifest-schema-3.md`, section 9.2). No GPU.
//!
//! The vectors also have `cublaslt.matmul` entries (section 9.5) of every form (each epilogue with `beta` 0 and 1, bf16 and f16, the four layouts, the
//! opt-ins, the `"torch"` posture, SM count targets, with a canary and without, both contracts), each changed in one field (every rule of the matmul
//! bullet of section 10), and the fields of a matmul added to `gemm` and `gemm_bias` entries: Rust decides each as Python does, reads the epilogue,
//! `beta`, the contract, the verification and the canary Python reads, and makes the canary's operands (`a` tag 0, `b` 1, `bias` 2, `c` 3, their first
//! elements) Python makes.

mod common;

use common::synthetic::read;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tilefish::{DType, LibraryEntry, LibraryImpl, canary_bytes};

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
    for layout in ["TN", "NN", "TT", "NT"] {
        assert!(
            cases.iter().any(|c| c["accepted"] == true && c["form"][2] == layout),
            "an entry of the layout {layout}"
        );
    }
    let (mut accepted, mut refused, mut matmuls) = (0, 0, 0);
    for case in &cases {
        let entry = case["entry"].clone();
        let stem = entry["stem"].as_str().unwrap().to_owned();
        match (case["accepted"].as_bool().unwrap(), read(&[], &[entry])) {
            (true, Ok(manifest)) => {
                let e = &manifest.library()[0];
                let form = &case["form"];
                let layout = e.layout.as_str();
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
                assert_eq!(e.epilogue.as_str(), case["entry"]["epilogue"], "{stem}");
                match case.get("matmul") {
                    Some(mm) => {
                        matmul_read_as_python_reads_it(e, mm, &stem);
                        matmuls += 1;
                    }
                    None => assert!(
                        e.implementation != LibraryImpl::Matmul
                            && e.beta == 0
                            && e.contract.is_none()
                            && e.verification.is_none()
                            && e.canary.is_none(),
                        "{stem}"
                    ),
                }
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
    assert!(matmuls >= 9, "{matmuls} accepted cublaslt.matmul entries");
    let refused_matmuls =
        cases.iter().filter(|c| c["accepted"] == false && c["entry"]["impl"] == "cublaslt.matmul").count();
    assert!(refused_matmuls >= 300, "{refused_matmuls} refused cublaslt.matmul entries");
}

/// What the reader reads of a `cublaslt.matmul` entry Python accepts is what Python reads (`export_rust_vectors.matmul_reading`): the epilogue, `beta`,
/// the contract, the verification, the canary, and the canary's operands (name, tag, element count, and the first 512 elements made on the host).
fn matmul_read_as_python_reads_it(e: &LibraryEntry, mm: &Value, stem: &str) {
    assert_eq!(e.implementation, LibraryImpl::Matmul, "{stem}");
    assert_eq!(
        (e.epilogue.as_str(), u64::from(e.beta)),
        (mm["epilogue"].as_str().unwrap(), mm["beta"].as_u64().unwrap()),
        "{stem}"
    );
    assert_eq!(e.contract.map(|c| c.as_str()), mm["contract"].as_str(), "{stem}");
    let v = e.verification.as_ref().unwrap();
    let want = &mm["verification"];
    assert_eq!(
        json!({"formula": v.formula, "accumulator": v.accumulator.as_str(), "operand_sets": v.operand_sets, "elements": v.elements,
               "differ": v.differ, "max_ulp": v.max_ulp}),
        *want,
        "{stem}"
    );
    let canary = e.canary.as_ref().map(|c| json!({"seed": c.seed, "out_digest": c.out_digest}));
    assert_eq!(canary.unwrap_or(Value::Null), mm["canary"], "{stem}");
    let seed = mm["canary_seed"].as_u64().unwrap();
    let operands = mm["canary_operands"].as_array().unwrap();
    let mine = e.canary_elements();
    assert_eq!(mine.len(), operands.len(), "{stem}");
    for ((name, tag, elements), want) in mine.into_iter().zip(operands) {
        assert_eq!(
            (name, u64::from(tag), elements),
            (want["name"].as_str().unwrap(), want["tag"].as_u64().unwrap(), want["elements"].as_u64().unwrap()),
            "{stem}"
        );
        let head = canary_bytes(e.dtype, seed, tag, elements.min(512));
        let digest: String = Sha256::digest(&head).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(digest, want["head_sha256"], "{stem}: the canary's {name}");
    }
}
