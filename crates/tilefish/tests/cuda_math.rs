//! The CUDA math functions of a generated kernel (`cuda_math`, `docs/manifest-schema-3.md` section 4.7), read and checked with no GPU on
//! synthetic manifests (`common::synthetic`): the reader takes the field on a generated kernel's entry and refuses a malformed one or one on
//! another entry, it is in the manifest digest, and `Manifest::check_cuda_math` refuses a release the functions were not measured on
//! (`python/tests/test_cudamath.py` has the same refusals).

mod common;

use common::synthetic::{Kernel, alignment, json, pointer, read, refusal};
use serde_json::Value;
use tilefish::CudaMath;

const LIBDEVICE: &str = "5c2fae37c86e68c3a38605a95f512d7d12d5f3db986310be47f57304aa72a5ee";

/// `act(x: Tensor[n, f32]) -> Tensor[n, f32]`, a generated kernel, with `cuda_math` set to `math` (none: `Value::Null`).
fn act(math: Value) -> Kernel {
    let mut k = Kernel::new(
        "triton",
        "act",
        103,
        vec![pointer(0, "x", "f32", "in", json!(["n"]), None), pointer(1, "out", "f32", "out", json!(["n"]), None)],
        vec![alignment(0, "x", 16), alignment(1, "out", 16)],
        json!({"n": {}}),
        "(cdiv(n, 1024),)",
    );
    if !math.is_null() {
        k.entry["cuda_math"] = math;
    }
    k
}

fn erff() -> Value {
    json!({"functions": ["erff", "tanhf"], "cuda": ["12.8", "12.9", "13.0", "13.1"], "libdevice": LIBDEVICE})
}

#[test]
fn the_functions_of_a_generated_kernel_are_read() {
    let m = read(&[act(erff())], &[]).expect("the manifest reads");
    let math = m.entry("act", "sm_103").unwrap().cuda_math.clone().expect("cuda_math");
    assert_eq!(
        math,
        CudaMath {
            functions: vec!["erff".into(), "tanhf".into()],
            cuda: vec!["12.8".into(), "12.9".into(), "13.0".into(), "13.1".into()],
            libdevice: Some(LIBDEVICE.into()),
        }
    );
    let gelu =
        read(&[act(json!({"functions": [], "cuda": [], "libdevice": LIBDEVICE}))], &[]).expect("only libdevice reads");
    assert!(gelu.entry("act", "sm_103").unwrap().cuda_math.as_ref().unwrap().functions.is_empty());
    let none = read(&[act(Value::Null)], &[]).expect("an entry without cuda_math reads as before");
    assert!(none.entry("act", "sm_103").unwrap().cuda_math.is_none());
    assert_ne!(m.digest(), none.digest()); // the field is digested like every other
}

#[test]
fn a_release_the_functions_were_not_measured_on_is_refused_by_check_cuda_math() {
    let m = read(&[act(erff())], &[]).unwrap();
    m.check_cuda_math("13.0").expect("13.0 is a release the functions were measured on");
    m.check_cuda_math("12.8").expect("so is 12.8");
    let err = m.check_cuda_math("13.2").expect_err("13.2 changed erff").to_string();
    assert!(
        err.contains("'act' (sm_103): erff, tanhf measured on CUDA [\"12.8\", \"12.9\", \"13.0\", \"13.1\"]"),
        "{err}"
    );
    assert!(err.contains("not measured on CUDA 13.2"), "{err}");
    assert!(m.check_cuda_math("13").is_err() && m.check_cuda_math("13.02").is_err());
    // a kernel that only links libdevice, and one with no cuda_math, claim no release: they pass whatever the host's
    let gelu = read(&[act(json!({"functions": [], "cuda": [], "libdevice": LIBDEVICE}))], &[]).unwrap();
    gelu.check_cuda_math("13.2").unwrap();
    read(&[act(Value::Null)], &[]).unwrap().check_cuda_math("11.0").unwrap();
}

#[test]
fn a_malformed_cuda_math_is_refused() {
    for (math, why) in [
        (json!("13.0"), "cuda_math must be an object"),
        (json!({"functions": ["erff"], "cuda": ["13.0"], "extra": 1}), "unknown field `extra`"),
        (json!({"cuda": ["13.0"]}), "missing field `functions`"),
        (json!({"functions": ["tanhf", "erff"], "cuda": []}), "cuda_math.functions must be sorted, each once"),
        (json!({"functions": ["erff", "erff"], "cuda": []}), "cuda_math.functions must be sorted, each once"),
        (json!({"functions": ["Erff"], "cuda": []}), "cuda_math.functions must be a list of function names"),
        (json!({"functions": ["erff"], "cuda": ["13"]}), "cuda_math.cuda must be a list of CUDA releases"),
        (json!({"functions": ["erff"], "cuda": ["13.01"]}), "cuda_math.cuda must be a list of CUDA releases"),
        (
            json!({"functions": ["erff"], "cuda": ["13.1", "13.0"]}),
            "cuda_math.cuda must be sorted by release, each once",
        ),
        (
            json!({"functions": ["erff"], "cuda": ["12.10", "12.9"]}),
            "cuda_math.cuda must be sorted by release, each once",
        ),
        (json!({"functions": [], "cuda": ["13.0"], "libdevice": LIBDEVICE}), "cuda_math.cuda lists releases"),
        (json!({"functions": ["erff"], "cuda": [], "libdevice": "ABC"}), "cuda_math.libdevice must be a sha256"),
        (json!({"functions": [], "cuda": []}), "cuda_math names no function and no libdevice"),
    ] {
        let message = refusal(&[act(math.clone())], &[]);
        assert!(message.contains(why), "{math}: {why}: {message}");
    }
    // a release of two digits is ordered by its number: 12.9 before 12.10
    read(&[act(json!({"functions": ["erff"], "cuda": ["12.9", "12.10"]}))], &[]).expect("12.9 then 12.10 is in order");
}

#[test]
fn cuda_math_is_only_on_a_generated_kernel() {
    let mut extern_ = act(erff());
    extern_.entry.as_object_mut().unwrap().remove("source_sha256"); // a hand-written Triton kernel: no generated source
    assert!(refusal(&[extern_], &[]).contains("cuda_math is only on the entry of a kernel tilefish generated"));
}
