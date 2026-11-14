//! The strict reader of a `cudnn.conv_fwd` library entry decides what Python's decides, and poses the same cuDNN problem.
//!
//! `data/python_conv.json` is made by `python/tests/export_rust_vectors.py`: the entries `tilefish._conv` writes for convolutions of every form
//! (the legacy and the backend API and every choice, both layouts, one to three spatial axes, f16, bf16 and f32 with each math, groups, strides,
//! paddings and dilations, a bucketed length, the alignments, an SM count target, an imported pin), each changed in one field, with what Python's
//! `validate_entry` decides: accepted, with the pin and the cuDNN-level problem it reads (`tilefish._cudnn.describe`), or refused. Rust reads each in
//! a manifest of its own and must accept exactly those, with the same pin, the same canary and the same `describe` (`docs/manifest-schema-3.md`,
//! section 9.4), and make the canary's operands (their first elements) and digests Python makes, on the host. No GPU.

mod common;

use common::synthetic::read;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tilefish::{ConvPin, canary_bytes, canary_digest};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn vectors() -> Vec<Value> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join("python_conv.json");
    serde_json::from_str::<Value>(&std::fs::read_to_string(path).unwrap()).unwrap().as_array().unwrap().clone()
}

#[test]
fn the_reader_accepts_and_refuses_the_entries_python_does_and_describes_the_same_problem() {
    let cases = vectors();
    assert!(cases.len() >= 300, "{}", cases.len());
    let (mut accepted, mut refused, mut with_canary) = (0, 0, 0);
    for case in &cases {
        let entry = case["entry"].clone();
        let stem = entry["stem"].as_str().unwrap().to_owned();
        match (case["accepted"].as_bool().unwrap(), read(&[], &[entry])) {
            (true, Ok(manifest)) => {
                assert!(manifest.library().is_empty(), "{stem}: a convolution is not a GEMM");
                let e = &manifest.convolutions()[0];
                assert_eq!(e.problem.describe(), case["describe"], "{stem}");
                let pin = match &e.pin {
                    ConvPin::Legacy { algo } => json!({"api": "legacy", "algo": algo}),
                    ConvPin::Backend { engine, knobs } => {
                        json!({"api": "backend", "engine": engine, "knobs": knobs.iter().map(|(t, v)| json!([t, v])).collect::<Vec<_>>()})
                    }
                };
                assert_eq!(pin, case["pin"], "{stem}");
                assert_eq!(e.sm_count_target, case["entry"]["sm_count_target"].as_u64(), "{stem}");
                let canary = e.canary.as_ref().map(|c| json!({"seed": c.seed, "y_digest": c.y_digest}));
                assert_eq!(canary.as_ref(), case["entry"].get("canary"), "{stem}");
                let operands = &case["canary_operands"];
                let seed = operands["seed"].as_u64().unwrap();
                let heads = operands["head_elements"].as_array().unwrap();
                let x = canary_bytes(e.problem.dtype, seed, 0, heads[0].as_u64().unwrap());
                let w = canary_bytes(e.problem.dtype, seed, 1, heads[1].as_u64().unwrap());
                assert_eq!(hex(&x[..16.min(x.len())]), operands["x_head"], "{stem}: the first bytes of the canary's x");
                assert_eq!(hex(&Sha256::digest(&x)), operands["x_head_sha256"], "{stem}: the canary's x");
                assert_eq!(hex(&Sha256::digest(&w)), operands["w_head_sha256"], "{stem}: the canary's w");
                assert_eq!(canary_digest(&x), operands["x_head_digest"], "{stem}: the digest of bytes");
                assert_eq!(
                    canary_digest(&x[..x.len() - 2]),
                    operands["x_head_odd_digest"],
                    "{stem}: the digest of bytes, a half word last"
                );
                if canary.is_some() {
                    with_canary += 1;
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
    assert!(accepted >= 20 && refused >= 300, "{accepted} accepted, {refused} refused");
    assert!(with_canary >= 10 && with_canary < accepted, "{with_canary} of {accepted} accepted entries have a canary");
}

/// Every form is among the accepted entries: both APIs, both layouts, every spatial rank, every data type.
#[test]
fn the_vectors_cover_every_form() {
    let cases = vectors();
    let accepted: Vec<&Value> = cases.iter().filter(|c| c["accepted"] == true).collect();
    for (key, values) in [
        ("layout", vec![json!("NCHW"), json!("NHWC")]),
        ("choice", vec![json!("v7"), json!("instant"), json!("mode_b"), json!("imported")]),
        ("math", vec![json!("tensor_op"), json!("fma"), json!("tf32")]),
    ] {
        for v in values {
            assert!(accepted.iter().any(|c| c["entry"][key] == v), "{key} {v}");
        }
    }
    for rank in 1..=3 {
        assert!(accepted.iter().any(|c| c["entry"]["input"].as_array().unwrap().len() == rank), "{rank} spatial axes");
    }
    for dtype in ["f16", "bf16", "f32"] {
        assert!(accepted.iter().any(|c| c["entry"]["dtypes"]["x"] == dtype), "{dtype}");
    }
}
