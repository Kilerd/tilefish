//! Warp-specialised kernels (`total_num_warps`, `docs/manifest-schema-3.md` section 4.2), read and checked with no GPU, on synthetic manifests
//! (`common::synthetic`): the block of such an entry is its partitions' warps in all, which is what a launch and a plan take; the reader refuses
//! a total that is not more than the warps of the default partition (`num_warps`), a block that is not the total's, and the field on a prebuilt
//! binary; and the field is in the manifest digest (`python/tests/test_warp_specialization.py` has the same refusals).

mod common;

use common::synthetic::{Kernel, alignment, json, read, refusal, scalar, tensordesc};
use serde_json::Value;
use tilefish::Arch;

/// `ws_mm(a: TensorDesc(Tensor[R, K, f16], block=(128, 64)), b: ..., c: Out[TensorDesc(Tensor[R, C, f16], block=(128, 128))], k_len: K)` declared
/// with `num_warps=4`, whose build found a warp-specialised kernel of `total` warps (a 4-warp default partition, a 1-warp TMA producer and a
/// 1-warp MMA issuer are 8 in Triton 3.5.1), or a kernel that is not warp-specialised (`None`).
fn ws_mm(total: Option<u32>) -> Kernel {
    let mut k_len = scalar(15, "k_len", "i32", "K", json!({}));
    k_len["index"] = 3.into();
    let mut k = Kernel::new(
        "triton",
        "ws_mm",
        103,
        vec![
            tensordesc(0, 0, "a", "f16", "in", json!(["R", "K"]), json!([128, 64]), json!([128, 64]), 128, None),
            tensordesc(5, 1, "b", "f16", "in", json!(["C", "K"]), json!([128, 64]), json!([128, 64]), 128, None),
            tensordesc(10, 2, "c", "f16", "out", json!(["R", "C"]), json!([128, 128]), json!([128, 64]), 128, None),
            k_len,
        ],
        vec![alignment(0, "a", 16), alignment(1, "b", 16), alignment(2, "c", 16)],
        json!({"R": {}, "C": {}, "K": {}}),
        "(cdiv(R, 128), cdiv(C, 128))",
    );
    if let Some(total) = total {
        k.entry["total_num_warps"] = total.into();
        k.entry["block"] = json!([32 * total, 1, 1]);
    }
    k
}

#[test]
fn a_warp_specialised_entry_is_launched_with_the_warps_of_all_its_partitions() {
    let m = read(&[ws_mm(Some(8))], &[]).expect("the manifest reads");
    let entry = m.entry("ws_mm", "sm_103").unwrap();
    assert_eq!((entry.num_warps, entry.total_num_warps, entry.block), (Some(4), Some(8), [256, 1, 1]));
    assert_eq!(m.plan("ws_mm", &[], Arch::sm(103)).unwrap().block(), [256, 1, 1]); // what every launch and every step of a plan takes
    let plain = read(&[ws_mm(None)], &[]).expect("the manifest reads");
    let entry = plain.entry("ws_mm", "sm_103").unwrap();
    assert_eq!((entry.num_warps, entry.total_num_warps, entry.block), (Some(4), None, [128, 1, 1]));
    assert_eq!(plain.plan("ws_mm", &[], Arch::sm(103)).unwrap().block(), [128, 1, 1]);
    // a default partition of 8 warps with the same workers: 12 in all (Triton 3.5.1 pads the workers to a warp group)
    let mut wide = ws_mm(Some(12));
    wide.entry["num_warps"] = 8.into();
    wide.entry["options"]["num_warps"] = 8.into();
    let w = read(&[wide], &[]).expect("the manifest reads");
    assert_eq!(w.plan("ws_mm", &[], Arch::sm(103)).unwrap().block(), [384, 1, 1]);
}

#[test]
fn the_compiled_warps_are_in_the_manifest_digest() {
    let eight = read(&[ws_mm(Some(8))], &[]).unwrap();
    let sixteen = read(&[ws_mm(Some(16))], &[]).unwrap();
    let plain = read(&[ws_mm(None)], &[]).unwrap();
    assert_ne!(eight.digest(), sixteen.digest());
    assert_ne!(eight.digest(), plain.digest());
    // a JSON null is as absent (as Python reads it), and the field is digested as it is written
    let mut null = ws_mm(None);
    null.entry["total_num_warps"] = Value::Null;
    let n = read(&[null], &[]).expect("a null total_num_warps is as absent");
    assert_eq!(n.entry("ws_mm", "sm_103").unwrap().total_num_warps, None);
    assert_ne!(n.digest(), plain.digest());
}

/// `edit` applied to the entry of `ws_mm(total)`, and the message the reader refuses it with must contain `why`.
fn refused(total: Option<u32>, edit: impl Fn(&mut Value), why: &str) {
    let mut k = ws_mm(total);
    edit(&mut k.entry);
    let message = refusal(&[k], &[]);
    assert!(message.contains(why), "{why}: {message}");
}

#[test]
fn the_reader_refuses_warps_that_are_not_those_of_a_warp_specialised_kernel() {
    // a silent change of the warps of a kernel that is not warp-specialised: the block is not 32 * num_warps
    refused(
        None,
        |e| e["block"] = json!([256, 1, 1]),
        "block must be [32 * num_warps, 1, 1] = [128, 1, 1], got [256, 1, 1]",
    );
    // the partitions of a warp-specialised kernel add warps to its default partition's, and a block has at most 32
    refused(Some(8), |e| e["total_num_warps"] = 4.into(), "total_num_warps must be more than num_warps (4");
    refused(Some(8), |e| e["total_num_warps"] = 2.into(), "total_num_warps must be more than num_warps (4");
    refused(Some(8), |e| e["total_num_warps"] = 33.into(), "and at most 32, got 33");
    // the block is the total's
    refused(
        Some(8),
        |e| e["block"] = json!([128, 1, 1]),
        "[32 * total_num_warps, 1, 1] = [256, 1, 1], got [128, 1, 1]",
    );
    refused(Some(8), |e| e["block"] = json!([128, 2, 1]), "[32 * total_num_warps, 1, 1] = [256, 1, 1]");
    // not an integer (serde says so, as for every integer field of an entry)
    refused(Some(8), |e| e["total_num_warps"] = json!("8"), "expected u32");
    refused(Some(8), |e| e["total_num_warps"] = json!(8.5), "expected u32");
    refused(Some(8), |e| e["total_num_warps"] = json!(true), "expected u32");
}

#[test]
fn a_prebuilt_binary_has_no_total_num_warps() {
    let mut k = Kernel::new(
        "binary",
        "scale",
        103,
        vec![
            common::synthetic::pointer(0, "x", "f32", "in", json!(["n"]), None),
            common::synthetic::pointer(1, "y", "f32", "out", json!(["n"]), None),
            scalar(2, "n_elem", "i32", "n", json!({})),
        ],
        vec![alignment(0, "x", 16), alignment(1, "y", 16)],
        json!({"n": {}}),
        "(cdiv(n, 64),)",
    );
    read(std::slice::from_ref(&k), &[]).expect("the binary reads");
    k.entry["total_num_warps"] = 4.into();
    let message = refusal(&[k], &[]);
    assert!(message.contains("a prebuilt binary has no [\"total_num_warps\"]"), "{message}");
}
