//! The `library` section of manifest schema 3 (a pinned cuBLASLt GEMM), the digest that covers it, and the checks of the kernel entries that
//! schema 3 added, on synthetic manifests (`common::synthetic`). No GPU: what needs one (the version of cuBLASLt, the vetting of the
//! algorithm, the launch) is in `tests/gpu_gate.rs`.

mod common;

use common::synthetic::{Kernel, alignment, cubin, elf, fixed, json, pointer, read, refusal, scalar};
use serde_json::Value;
use tilefish::json::{canonical, parse_strict};
use tilefish::{
    Access, Arch, DType, DevicePtr, Error, LibraryImpl, Manifest, ParamKind, Pin, PreconditionKind, TensorArg,
    manifest_digest,
};

const ALGO: &str = "0100000000000000030000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000f";

/// The entry Python writes for `out[M, N] = a[M, K] @ b[N, K]^T (+ bias[N])` at the bucket `m`.
fn gemm(bias: bool, m: u64, n: u64, k: u64) -> Value {
    let op = if bias { "gemm_bias" } else { "gemm" };
    let ptr = |i: usize, name: &str, access: &str, shape: Value| json!({"index": i, "name": name, "kind": "pointer", "dtype": "bf16", "access": access, "shape": shape});
    let mut params = vec![ptr(0, "a", "in", json!(["M", k])), ptr(1, "b", "in", json!([n, k]))];
    if bias {
        params.push(ptr(2, "bias", "in", json!([n])));
    }
    let last = params.len();
    params.push(ptr(last, "out", "out", json!(["M", n])));
    let names: Vec<String> = params.iter().map(|p| p["name"].as_str().unwrap().to_owned()).collect();
    let mut pres: Vec<Value> = names.iter().enumerate().map(|(i, n)| alignment(i, n, 16)).collect();
    let statics = |pres: &mut Vec<Value>, i: usize, name: &str, axis: usize, value: u64| {
        pres.push(json!({"kind": "static_size", "arg": name, "index": i, "axis": axis, "value": value}));
    };
    statics(&mut pres, 0, "a", 1, k);
    statics(&mut pres, 1, "b", 0, n);
    statics(&mut pres, 1, "b", 1, k);
    if bias {
        statics(&mut pres, 2, "bias", 0, n);
    }
    statics(&mut pres, last, "out", 1, n);
    pres.push(json!({"kind": "bucket", "dim": "M", "value": m, "buckets": [32, 64, 320]}));
    let mut dtypes = json!({"a": "bf16", "b": "bf16", "out": "bf16"});
    if bias {
        dtypes["bias"] = "bf16".into();
    }
    json!({
        "stem": format!("{op}.M{m}"), "op": op, "arch": "sm_103", "impl": format!("cublaslt.{op}"),
        "abi": {"params": params}, "buckets": {"M": m},
        "dims": {"M": {"buckets": [32, 64, 320], "bucket": m}},
        "preconditions": pres, "dtypes": dtypes,
        "compute_type": "CUBLAS_COMPUTE_32F", "scale_type": "CUDA_R_32F", "transa": "T", "transb": "N",
        "m": m, "n": n, "k": k, "lda": k, "ldb": k, "ldc": n, "ldd": n,
        "epilogue": if bias { "bias" } else { "none" }, "workspace_bytes": 8519680,
        "algo": ALGO, "cublaslt_version": 120804, "sm_count": 30, "declaration_digest": "a".repeat(64),
    })
}

fn dense(addr: u64, shape: &[usize], access: Access) -> TensorArg<'_> {
    let bytes = shape.iter().product::<usize>() as u64 * 2;
    // SAFETY: nothing is dereferenced: `prepare` only looks at the numbers.
    let region = unsafe { DevicePtr::from_raw(addr, bytes, 0) };
    TensorArg::new(region, shape, DType::Bf16, access).unwrap()
}

// -- reading the section -----------------------------------------------------------------------------------------------------------

#[test]
fn a_library_entry_is_read_with_its_operands_its_problem_and_its_pinned_algorithm() {
    let manifest = read(&[], &[gemm(true, 64, 1024, 1024), gemm(false, 32, 1024, 3072)]).unwrap();
    assert!(manifest.entries().is_empty() && manifest.library().len() == 2);
    let e = manifest.library_entry("gemm_bias.M64", "sm_103").unwrap();
    assert_eq!(
        (e.implementation, e.m, e.n, e.k, e.workspace_bytes, e.cublaslt_version),
        (LibraryImpl::GemmBias, 64, 1024, 1024, 8_519_680, 120_804)
    );
    assert_eq!(e.params.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), ["a", "b", "bias", "out"]);
    assert_eq!(e.buckets, [("M".to_owned(), 64)]);
    assert_eq!((e.algo[0], e.algo[8], e.algo[63]), (0x01, 0x03, 0x0f), "the 64 bytes in memory order");
    assert!(matches!(&e.params[3].kind, ParamKind::Pointer { access: Access::Out, dtype: Some(DType::Bf16), .. }));
    let plain = manifest.library_entry("gemm.M32", "sm_103").unwrap();
    assert_eq!(plain.implementation, LibraryImpl::Gemm);
    assert_eq!(plain.params.len(), 3);
    assert!(manifest.library_entry("gemm.M32", "sm_90").is_none());
}

#[test]
fn the_operands_of_a_gemm_are_checked_like_a_kernels() {
    let manifest = read(&[], &[gemm(true, 64, 1024, 512)]).unwrap();
    let plan = manifest.library_plan("gemm_bias", &[("M", 64)], Arch::sm(103)).unwrap();
    assert_eq!(plan.tensor_names().collect::<Vec<_>>(), ["a", "b", "bias", "out"]);
    let (a, b, bias, out) = ([64usize, 512], [1024usize, 512], [1024usize], [64usize, 1024]);
    let at = |x: u64| 0x7000_0000_0000 + x * 0x1000_0000;
    let call = |ptrs: [u64; 4]| {
        plan.prepare(
            &[
                dense(ptrs[0], &a, Access::In),
                dense(ptrs[1], &b, Access::In),
                dense(ptrs[2], &bias, Access::In),
                dense(ptrs[3], &out, Access::InOut),
            ],
            None,
        )
    };
    let ok = call([at(0), at(1), at(2), at(3)]).unwrap();
    assert_eq!(ok.params(), [at(0), at(1), at(2), at(3)], "a, b, bias, out: the addresses are all a GEMM is given");
    // 16-byte alignment of every pointer
    for i in 0..4 {
        let mut ptrs = [at(0), at(1), at(2), at(3)];
        ptrs[i] += 8;
        let e = call(ptrs).unwrap_err();
        assert_eq!(e.precondition_kind(), Some(PreconditionKind::PointerAlignment), "operand {i}: {e}");
    }
    // the result must not overlap an operand
    let e = call([at(0), at(1), at(2), at(0)]).unwrap_err();
    assert_eq!(e.precondition_kind(), Some(PreconditionKind::Alias), "{e}");
    // an exact bucket: 48 rows is not 64, and the other bucket is another binary
    let a48 = [48usize, 512];
    let out48 = [48usize, 1024];
    let e = plan
        .prepare(
            &[
                dense(at(0), &a48, Access::In),
                dense(at(1), &b, Access::In),
                dense(at(2), &bias, Access::In),
                dense(at(3), &out48, Access::InOut),
            ],
            None,
        )
        .unwrap_err();
    assert_eq!(e.precondition_kind(), Some(PreconditionKind::Bucket), "{e}");
    assert!(e.to_string().contains("no padding"), "{e}");
    // a K that is not the one the entry is built for
    let b_wrong = [1024usize, 1024];
    let e = plan
        .prepare(
            &[
                dense(at(0), &a, Access::In),
                dense(at(1), &b_wrong, Access::In),
                dense(at(2), &bias, Access::In),
                dense(at(3), &out, Access::InOut),
            ],
            None,
        )
        .unwrap_err();
    assert_eq!(e.precondition_kind(), Some(PreconditionKind::StaticSize), "{e}");
    // the result is written: a read-only tensor for `out` is refused
    let e = plan
        .prepare(
            &[
                dense(at(0), &a, Access::In),
                dense(at(1), &b, Access::In),
                dense(at(2), &bias, Access::In),
                dense(at(3), &out, Access::In),
            ],
            None,
        )
        .unwrap_err();
    assert_eq!(e.precondition_kind(), Some(PreconditionKind::Access), "{e}");
}

#[test]
fn a_gemm_and_a_kernel_are_found_by_their_own_names_and_say_where_the_other_is() {
    let manifest = read(&[axpy()], &[gemm(false, 32, 1024, 1024)]).unwrap();
    assert!(manifest.library_plan("gemm", &[("M", 32)], Arch::sm(103)).is_ok());
    assert!(manifest.library_plan("gemm.M32", &[], Arch::sm(103)).is_ok());
    // the wrong bucket, the wrong architecture, the wrong name
    let e = manifest.library_plan("gemm", &[("M", 48)], Arch::sm(103)).unwrap_err();
    assert_eq!(e.precondition_kind(), Some(PreconditionKind::Bucket), "{e}");
    match manifest.library_plan("gemm", &[("M", 32)], Arch::sm(90)) {
        Err(Error::NoBinary(m)) => assert!(m.contains("no binary for 'gemm' on sm_90") && m.contains("sm_103"), "{m}"),
        other => panic!("{other:?}"),
    }
    match manifest.library_plan("axpy", &[], Arch::sm(103)) {
        Err(Error::NoBinary(m)) => assert!(
            m.contains("no library operation 'axpy'") && m.contains("is a kernel") && m.contains("Manifest::kernel"),
            "{m}"
        ),
        other => panic!("{other:?}"),
    }
    match manifest.plan("gemm.M32", &[], Arch::sm(103)) {
        Err(Error::NoBinary(m)) => assert!(m.contains("no kernel 'gemm.M32'") && m.contains("Manifest::gemm"), "{m}"),
        other => panic!("{other:?}"),
    }
}

fn axpy() -> Kernel {
    Kernel::new(
        "binary",
        "axpy",
        103,
        vec![
            pointer(0, "x", "f32", "in", json!(["n"]), None),
            pointer(1, "y", "f32", "out", json!(["n"]), None),
            fixed(2, "alpha", "f32", json!(2.0)),
        ],
        vec![alignment(0, "x", 16), alignment(1, "y", 16)],
        json!({"n": {}}),
        "(cdiv(n, 128),)",
    )
}

#[test]
fn a_stem_and_an_architecture_are_unique_across_kernels_and_library_together() {
    // the kernel is called gemm.M32 as well
    let mut k = axpy();
    k.entry["stem"] = "gemm.M32".into();
    k.entry["op"] = "gemm".into();
    k.entry["buckets"] = json!({"M": 32});
    k.entry["dims"] = json!({"n": {}, "M": {"buckets": [32, 64, 320], "bucket": 32}});
    k.entry["preconditions"]
        .as_array_mut()
        .unwrap()
        .push(json!({"kind": "bucket", "dim": "M", "value": 32, "buckets": [32, 64, 320]}));
    k.entry["cubin"] = "gemm.M32.sm_103.cubin".into();
    assert!(refusal(&[k], &[gemm(false, 32, 1024, 1024)]).contains("lists 'gemm.M32' for sm_103 twice"));
    // the same library entry twice
    assert!(refusal(&[], &[gemm(false, 32, 1024, 1024), gemm(false, 32, 1024, 1024)]).contains("twice"));
    // another architecture is another entry
    let mut other = gemm(false, 32, 1024, 1024);
    other["arch"] = "sm_120".into();
    assert!(read(&[], &[gemm(false, 32, 1024, 1024), other]).is_ok());
}

#[test]
fn a_library_entry_that_is_not_what_python_writes_is_refused() {
    type Change = fn(&mut Value);
    let cases: [(&str, Change, &str); 35] = [
        ("an unknown field", |e| e["note"] = 1.into(), "unknown field"),
        (
            "a missing field",
            |e| {
                e.as_object_mut().unwrap().remove("algo");
            },
            "missing field",
        ),
        ("an impl that is not one", |e| e["impl"] = "cublaslt.conv".into(), "impl must be one of"),
        ("a stem that is not op and buckets", |e| e["stem"] = "gemm.M64".into(), "does not match op"),
        ("an arch that is not one", |e| e["arch"] = "sm_103a".into(), "arch must look like"),
        ("m = 0", |e| e["m"] = 0.into(), "m must be an int from 1"),
        ("k above int32", |e| e["k"] = 2147483648u64.into(), "k must be an int from 1"),
        ("m as a float", |e| e["m"] = 32.0.into(), "m must be an int"),
        ("lda that is not k", |e| e["lda"] = 8.into(), "leading dimensions must be"),
        ("ldd that is not n", |e| e["ldd"] = 1.into(), "leading dimensions must be"),
        ("transa N", |e| e["transa"] = "N".into(), "transa and transb must be"),
        ("transb T", |e| e["transb"] = "T".into(), "transa and transb must be"),
        ("another compute type", |e| e["compute_type"] = "CUBLAS_COMPUTE_16F".into(), "compute_type and scale_type"),
        ("another scale type", |e| e["scale_type"] = "CUDA_R_16F".into(), "compute_type and scale_type"),
        ("an epilogue that is not the impl's", |e| e["epilogue"] = "bias".into(), "the epilogue of"),
        ("dtypes with a bias for the plain op", |e| e["dtypes"]["bias"] = "bf16".into(), "dtypes must be"),
        ("dtypes that are f16", |e| e["dtypes"]["a"] = "f16".into(), "dtypes must be"),
        ("a workspace of 0", |e| e["workspace_bytes"] = 0.into(), "workspace_bytes must be"),
        ("a workspace above 2^40", |e| e["workspace_bytes"] = ((1u64 << 40) + 1).into(), "workspace_bytes must be"),
        ("an algo that is upper case", |e| e["algo"] = ALGO.to_uppercase().into(), "algo must be 128 lowercase hex"),
        ("an algo that is short", |e| e["algo"] = ALGO[..126].into(), "algo must be 128 lowercase hex"),
        ("a version of 0", |e| e["cublaslt_version"] = 0.into(), "cublaslt_version must be a positive int"),
        ("a declaration digest that is not one", |e| e["declaration_digest"] = "xyz".into(), "declaration_digest"),
        ("an operand that is not bf16", |e| e["abi"]["params"][0]["dtype"] = "f16".into(), "must be the bf16 pointer"),
        ("a result that is read only", |e| e["abi"]["params"][2]["access"] = "in".into(), "with access out"),
        ("a shape that disagrees with m", |e| e["abi"]["params"][0]["shape"][0] = 48.into(), "m, n and k say"),
        ("a shape expression with no bucket", |e| e["abi"]["params"][0]["shape"][0] = "S".into(), "no bucket"),
        (
            "a precondition that does not belong",
            |e| {
                e["preconditions"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"kind": "max_extent", "arg": "a", "index": 0, "value": 2147483647}))
            },
            "does not belong in a library entry",
        ),
        ("an alignment of 8", |e| e["preconditions"][0]["bytes"] = 8.into(), "pointer_alignment must be 16"),
        ("a bucket that is not the entry's", |e| e["dims"]["M"]["bucket"] = 64.into(), "bucket"),
        (
            "no sm_count",
            |e| {
                e.as_object_mut().unwrap().remove("sm_count");
            },
            "missing field `sm_count`",
        ),
        ("sm_count 0", |e| e["sm_count"] = 0.into(), "sm_count must be a positive int"),
        ("sm_count as a float", |e| e["sm_count"] = 30.0.into(), "sm_count must be a positive int"),
        ("sm_count as a bool", |e| e["sm_count"] = true.into(), "sm_count must be a positive int"),
        ("sm_count as a string", |e| e["sm_count"] = "30".into(), "sm_count must be a positive int"),
    ];
    for (what, change, why) in cases {
        let mut entry = gemm(false, 32, 1024, 1024);
        change(&mut entry);
        let message = refusal(&[], &[entry]);
        assert!(message.contains(why), "{what}: {message}");
        assert!(message.contains("library entry #0"), "{what}: {message}");
    }
    // the bias op needs its operand, in its place
    let mut entry = gemm(true, 32, 1024, 1024);
    entry["abi"]["params"].as_array_mut().unwrap().remove(2);
    assert!(refusal(&[], &[entry]).contains("abi.params must list 4 pointers"));
    // a section that is not a list
    let text = common::synthetic::manifest_text(&[], &[]).replace("\"library\": []", "\"library\": {}");
    match Manifest::from_embedded(text.as_bytes(), &[], Pin::TrustDirectory) {
        Err(Error::Manifest(m)) => assert!(m.contains("not a tilefish manifest of schema 3"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn n_and_k_are_multiples_of_eight_as_the_declaration_asks() {
    // Python's `check_declaration`: every row of a, b and the result starts on 16 bytes (M may be any size)
    assert!(read(&[], &[gemm(false, 32, 8, 8), gemm(true, 64, 3072, 1024), gemm(false, 320, 1024, 3072)]).is_ok());
    for (n, k, why) in
        [(1020, 1024, "n = 1020"), (1024, 1028, "k = 1028"), (4, 1024, "n = 4"), (1024, 1, "k = 1"), (12, 12, "n = 12")]
    {
        for bias in [false, true] {
            let message = refusal(&[], &[gemm(bias, 32, n, k)]);
            assert!(message.contains(why) && message.contains("must be a multiple of 8"), "{n}, {k}: {message}");
            assert!(message.contains("library entry #0"), "{message}");
        }
    }
}

// -- the digest --------------------------------------------------------------------------------------------------------------------

#[test]
fn the_digest_of_the_empty_manifests_is_what_python_computes() {
    let v3 = parse_strict(r#"{"schema": 3, "kernels": [], "library": []}"#).unwrap();
    assert_eq!(manifest_digest(&v3).unwrap(), "f66955caeb4e878dbe95483da2607d19ea4f734977e86282fff6f92f1a87fe6e");
    // the empty schema 2 manifest is still digested in its own form
    let v2 = parse_strict(r#"{"schema": 2, "kernels": []}"#).unwrap();
    assert_eq!(manifest_digest(&v2).unwrap(), "a3593eed6b463f7408711b92645197eda2e7bc5f2830b138b40e33ceedc1cfc8");
    // and the manifest read from a file says the same
    let manifest = read(&[], &[]).unwrap();
    assert_eq!(manifest.digest(), "f66955caeb4e878dbe95483da2607d19ea4f734977e86282fff6f92f1a87fe6e");
}

#[test]
fn the_digest_covers_every_field_of_every_entry_and_does_not_depend_on_their_order() {
    let kernels = [axpy()];
    let lib = [gemm(false, 32, 1024, 1024), gemm(true, 64, 1024, 1024), gemm(false, 320, 1024, 1024)];
    let doc = |kernels: &[Kernel], lib: &[Value]| -> Value {
        serde_json::from_str(&common::synthetic::manifest_text(kernels, lib)).unwrap()
    };
    let base = manifest_digest(&doc(&kernels, &lib)).unwrap();
    // the order of the entries in the file
    let reversed = [lib[2].clone(), lib[1].clone(), lib[0].clone()];
    assert_eq!(manifest_digest(&doc(&kernels, &reversed)).unwrap(), base);
    // the version of the tool that wrote it is not part of it
    let mut d = doc(&kernels, &lib);
    d["tilefish"] = "9.9.9".into();
    assert_eq!(manifest_digest(&d).unwrap(), base);
    // each library field is
    for (field, value) in [
        ("algo", json!(ALGO.replace('f', "e"))),
        ("cublaslt_version", json!(120805)),
        ("workspace_bytes", json!(1)),
        ("declaration_digest", json!("b".repeat(64))),
        ("epilogue", json!("bias")),
    ] {
        let mut d = doc(&kernels, &lib);
        d["library"][0][field] = value;
        assert_ne!(manifest_digest(&d).unwrap(), base, "{field}");
    }
    // and each of the fields schema 3 added to a kernel entry
    for (what, change) in [
        ("kind", (|e: &mut Value| e["kind"] = "triton".into()) as fn(&mut Value)),
        ("scratch_slots", |e| e["abi"]["scratch_slots"] = 2.into()),
        ("a fixed value", |e| e["abi"]["params"][2]["value"] = 3.0.into()),
        ("binary", |e| e["binary"]["source"] = "cubin".into()),
        ("max_extent", |e| {
            e["preconditions"]
                .as_array_mut()
                .unwrap()
                .push(json!({"kind": "max_extent", "arg": "x", "index": 0, "value": 2147483647}))
        }),
    ] {
        let mut d = doc(&kernels, &lib);
        change(&mut d["kernels"][0]);
        assert_ne!(manifest_digest(&d).unwrap(), base, "{what}");
    }
    // kernels and library are two lists: an entry in the other one is another manifest
    let mut d = doc(&kernels, &lib);
    let moved = d["kernels"][0].clone();
    d["kernels"] = json!([]);
    d["library"].as_array_mut().unwrap().push(moved);
    assert_ne!(manifest_digest(&d).unwrap(), base);
}

#[test]
fn a_document_of_another_schema_is_digested_in_the_form_of_schema_2_and_a_schema_3_document_needs_its_library() {
    let entry = json!({"stem": "a", "arch": "sm_90", "shared": 16});
    for (schema, library) in [
        (json!(2), false),
        (json!(1), false),
        (json!(4), false),
        (json!(true), false),
        (json!("3"), false),
        (json!(3), true),
        (json!(3.0), true),
    ] {
        let mut doc = json!({"schema": schema, "kernels": [entry]});
        if library {
            doc["library"] = json!([]);
        }
        let digest = manifest_digest(&doc).unwrap();
        // what Python does: schema == 3 (the integer 3 and the float 3.0) is the form of schema 3, anything else the legacy form
        let v3 = matches!(doc["schema"].as_f64(), Some(f) if f == 3.0) && doc["schema"].as_bool().is_none();
        let tag: &[u8] = if v3 { b"tilefish.manifest-digest.v3\0" } else { b"tilefish.manifest-digest.v2\0" };
        let body = if v3 {
            json!({"schema": doc["schema"], "kernels": [entry], "library": []})
        } else {
            json!({"schema": doc["schema"], "kernels": [entry]})
        };
        let mut bytes = tag.to_vec();
        bytes.extend_from_slice(canonical(&body).as_bytes());
        assert_eq!(digest, tilefish::elf::sha256_hex(&bytes), "{}", doc["schema"]);
    }
    // schema 3 without a library, and a list that is not one, cannot be digested
    assert!(manifest_digest(&json!({"schema": 3, "kernels": []})).is_err());
    assert!(manifest_digest(&json!({"schema": 3, "kernels": [], "library": {}})).is_err());
    assert!(manifest_digest(&json!({"schema": 3, "kernels": [5], "library": []})).is_err());
    assert!(manifest_digest(&json!({"schema": 3, "kernels": [], "library": [[]]})).is_err());
}

// -- the kernel entries of schema 3 --------------------------------------------------------------------------------------------------

#[test]
fn a_kernel_entry_has_a_kind_and_the_number_of_its_scratch_slots() {
    type KernelChange = fn(&mut Kernel);
    let cases: [(&str, KernelChange, &str); 12] = [
        (
            "no kind",
            |k| {
                k.entry.as_object_mut().unwrap().remove("kind");
            },
            "missing field `kind`",
        ),
        ("another kind", |k| k.entry["kind"] = "cubin".into(), "kind must be 'triton' or 'binary'"),
        (
            "no scratch_slots",
            |k| {
                k.entry["abi"].as_object_mut().unwrap().remove("scratch_slots");
            },
            "missing field `scratch_slots`",
        ),
        (
            "a binary kernel with 3 scratch slots",
            |k| k.entry["abi"]["scratch_slots"] = 3.into(),
            "has 0 scratch slots, but abi.scratch_slots is 3",
        ),
        ("scratch_slots as 0.0", |k| k.entry["abi"]["scratch_slots"] = 0.0.into(), "invalid type"),
        (
            "a prebuilt binary that lists scratch slots",
            |k| {
                k.entry["abi"]["scratch"] = json!([{"slot": 3, "name": "global_scratch", "type": "u64", "ptx_type": "u64", "size": 8, "value": 0}]);
            },
            "exactly its parameters",
        ),
        ("a binary with num_warps", |k| k.entry["num_warps"] = 4.into(), "has no [\"num_warps\"]"),
        (
            "a binary with options",
            |k| k.entry["options"] = json!({"num_warps": 4, "num_stages": 3}),
            "has no [\"options\"]",
        ),
        (
            "a binary with no binary section",
            |k| {
                k.entry.as_object_mut().unwrap().remove("binary");
            },
            "has a `binary` section",
        ),
        ("a binary section with another source", |k| k.entry["binary"] = json!({"source": "fatbin"}), "binary must be"),
        ("a binary section with another key", |k| k.entry["binary"]["extra"] = 1.into(), "binary must be"),
        (
            "a binary without its source digest",
            |k| {
                k.entry.as_object_mut().unwrap().remove("source_digest");
                k.entry.as_object_mut().unwrap().remove("source_digest_kind");
            },
            "records the sha256 of its PTX or cubin",
        ),
    ];
    for (what, change, why) in cases {
        let mut k = axpy();
        change(&mut k);
        let message = refusal(&[k], &[]);
        assert!(message.contains(why), "{what}: {message}");
    }
    // a Triton kernel with no options, with a binary section, with a toolchain that is not triton's
    let mut t = Kernel::new(
        "triton",
        "t",
        103,
        vec![pointer(0, "x", "f32", "in", json!(["n"]), None), alignment_only()],
        vec![alignment(0, "x", 16)],
        json!({"n": {}}),
        "(n,)",
    );
    t.entry["abi"]["params"].as_array_mut().unwrap().pop();
    t.entry["abi"]["scratch"][0]["slot"] = 1.into();
    t.entry["abi"]["scratch"][1]["slot"] = 2.into();
    assert!(read(&[t], &[]).is_ok());
    for (what, change, why) in [
        (
            "no options",
            (|k: &mut Kernel| {
                k.entry.as_object_mut().unwrap().remove("options");
            }) as fn(&mut Kernel),
            "has the compile options",
        ),
        ("a binary section", |k| k.entry["binary"] = json!({"source": "cubin"}), "no `binary` section"),
        (
            "no triton in the toolchain",
            |k| k.entry["toolchain"] = json!({"ptxas": "x"}),
            "toolchain of a triton kernel",
        ),
    ] {
        let mut t = Kernel::new(
            "triton",
            "t",
            103,
            vec![pointer(0, "x", "f32", "in", json!(["n"]), None)],
            vec![alignment(0, "x", 16)],
            json!({"n": {}}),
            "(n,)",
        );
        change(&mut t);
        let message = refusal(&[t], &[]);
        assert!(message.contains(why), "{what}: {message}");
    }
}

fn alignment_only() -> Value {
    scalar(1, "n_unused", "i32", "n", json!({}))
}

#[test]
fn the_source_and_the_toolchain_of_a_prebuilt_binary_agree_with_how_it_was_made() {
    // a cubin used as it is: its source digest is the digest of the file, and no ptxas was run
    let mut k = axpy();
    k.entry["binary"] = json!({"source": "cubin"});
    k.entry["toolchain"] = json!({});
    k.entry["ptx_target"] = "".into();
    let sha = k.entry["sha256"].clone();
    k.entry["source_digest"] = sha;
    assert!(read(&[k], &[]).is_ok());
    for (what, change, why) in [
        (
            "a source digest that is not the file's",
            (|k: &mut Kernel| k.entry["source_digest"] = "9".repeat(64).into()) as fn(&mut Kernel),
            "is the sha256 of the cubin file",
        ),
        ("a ptxas for a cubin", |k| k.entry["toolchain"] = json!({"ptxas": "x"}), "{} for a cubin"),
    ] {
        let mut k = axpy();
        k.entry["binary"] = json!({"source": "cubin"});
        k.entry["toolchain"] = json!({});
        let sha = k.entry["sha256"].clone();
        k.entry["source_digest"] = sha;
        change(&mut k);
        let message = refusal(&[k], &[]);
        assert!(message.contains(why), "{what}: {message}");
    }
    // PTX that was assembled: ptxas is recorded; a Triton in the toolchain of a binary, a source_sha256 and constexprs are not its
    for (what, change, why) in [
        (
            "no ptxas for PTX",
            (|k: &mut Kernel| k.entry["toolchain"] = json!({})) as fn(&mut Kernel),
            "{} for a cubin and",
        ),
        ("a triton", |k| k.entry["toolchain"] = json!({"triton": "3.5.1", "ptxas": "x"}), "has no triton"),
        ("a source_sha256", |k| k.entry["source_sha256"] = "5".repeat(64).into(), "no source_sha256"),
        (
            "a source digest of another kind",
            |k| k.entry["source_digest_kind"] = "tilefish.triton-ir.v1".into(),
            "tilefish.binary.v1",
        ),
        ("a constexpr", |k| k.entry["constexprs"] = json!([{"index": 3, "name": "C", "value": 1}]), "no constexprs"),
        ("a block of 2048", |k| k.entry["block"] = json!([2048, 1, 1]), "three sizes from 1 to 1024"),
        ("a block of 1024 x 2", |k| k.entry["block"] = json!([1024, 2, 1]), "whose product is at most 1024"),
        ("a launch that needs a cluster", |k| k.entry["launch"]["num_ctas"] = 2.into(), "zero ones"),
    ] {
        let mut k = axpy();
        change(&mut k);
        let message = refusal(&[k], &[]);
        assert!(message.contains(why), "{what}: {message}");
    }
}

#[test]
fn a_cubin_is_for_the_architecture_the_entry_says() {
    // a cubin of sm_120 under the name of sm_103: refused at the reading, with both architectures named
    let wrong = cubin("tf_axpy", 120);
    let k = axpy().with_cubin(wrong);
    let message = refusal(&[k], &[]);
    assert!(message.contains("is for sm_120") && message.contains("sm_103"), "{message}");
    // for a Triton kernel as well, where the header can be read
    let mut t = Kernel::new(
        "triton",
        "t",
        103,
        vec![pointer(0, "x", "f32", "in", json!(["n"]), None)],
        vec![alignment(0, "x", 16)],
        json!({"n": {}}),
        "(n,)",
    );
    t = t.with_cubin(cubin("tf_t", 120));
    assert!(refusal(&[t], &[]).contains("is for sm_120"));
    // sm_90 is an osabi 0x33 cubin with the SM in the low byte
    let sm90 = elf(&[(".text.tf_axpy", 1, 6, &[1, 2, 3, 4])], 0x33, 0x0050055A);
    let mut k = axpy().with_cubin(sm90);
    k.entry["arch"] = "sm_90".into();
    assert!(read(&[k], &[]).is_ok());
    // a prebuilt binary whose header names no architecture cannot be told, and is refused; a Triton cubin of one is left to the driver
    let odd = elf(&[(".text.tf_axpy", 1, 6, &[1, 2, 3, 4])], 0x66, 0);
    let k = axpy().with_cubin(odd.clone());
    assert!(refusal(&[k], &[]).contains("architecture cannot be read"));
}
