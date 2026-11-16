//! The `library` section of manifest schema 3 (a pinned cuBLASLt GEMM), the digest that covers it, and the checks of the kernel entries that
//! schema 3 added, on synthetic manifests (`common::synthetic`). No GPU: what needs one (the version of cuBLASLt, the vetting of the
//! algorithm, the launch) is in `tests/gpu_gate.rs`.

mod common;

use common::synthetic::{Kernel, alignment, cubin, elf, fixed, json, pointer, read, refusal, scalar};
use serde_json::Value;
use tilefish::json::{canonical, parse_strict};
use tilefish::{
    Access, AlgoSource, Arch, DType, DevicePtr, Error, GemmLayout, Heuristic, LibraryImpl, Manifest, ParamKind, Pin,
    PreconditionKind, TensorArg, manifest_digest,
};

const ALGO: &str = "0100000000000000030000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000f";

/// The entry Python writes for `out[M, N] = a[M, K] @ b[N, K]^T (+ bias[N])` of bf16 at the bucket `m`.
fn gemm(bias: bool, m: u64, n: u64, k: u64) -> Value {
    gemm_of(bias, m, n, k, "bf16", "bf16", "TN")
}

/// The entry Python writes for a GEMM of `dtype` operands, an `out` result (and bias), in `layout` (`"TN"`: `b` is `[n, k]`; `"NN"`:
/// `b` is `[k, n]`; `"TT"` and `"NT"`: those, and `a` is `[k, m]`).
fn gemm_of(bias: bool, m: u64, n: u64, k: u64, dtype: &str, out: &str, layout: &str) -> Value {
    let op = if bias { "gemm_bias" } else { "gemm" };
    let nn = layout.starts_with('N');
    let at = layout.ends_with('T');
    let ptr = |i: usize, name: &str, dtype: &str, access: &str, shape: Value| json!({"index": i, "name": name, "kind": "pointer", "dtype": dtype, "access": access, "shape": shape});
    let b_shape = if nn { [k, n] } else { [n, k] };
    let a_shape = if at { json!([k, "M"]) } else { json!(["M", k]) };
    let mut params = vec![ptr(0, "a", dtype, "in", a_shape), ptr(1, "b", dtype, "in", json!(b_shape))];
    if bias {
        params.push(ptr(2, "bias", out, "in", json!([n])));
    }
    let last = params.len();
    params.push(ptr(last, "out", out, "out", json!(["M", n])));
    let names: Vec<String> = params.iter().map(|p| p["name"].as_str().unwrap().to_owned()).collect();
    let mut pres: Vec<Value> = names.iter().enumerate().map(|(i, n)| alignment(i, n, 16)).collect();
    let statics = |pres: &mut Vec<Value>, i: usize, name: &str, axis: usize, value: u64| {
        pres.push(json!({"kind": "static_size", "arg": name, "index": i, "axis": axis, "value": value}));
    };
    statics(&mut pres, 0, "a", if at { 0 } else { 1 }, k);
    statics(&mut pres, 1, "b", 0, b_shape[0]);
    statics(&mut pres, 1, "b", 1, b_shape[1]);
    if bias {
        statics(&mut pres, 2, "bias", 0, n);
    }
    statics(&mut pres, last, "out", 1, n);
    pres.push(json!({"kind": "bucket", "dim": "M", "value": m, "buckets": [32, 64, 320]}));
    let mut dtypes = json!({"a": dtype, "b": dtype, "out": out});
    if bias {
        dtypes["bias"] = out.into();
    }
    json!({
        "stem": format!("{op}.M{m}"), "op": op, "arch": "sm_103", "impl": format!("cublaslt.{op}"),
        "abi": {"params": params}, "buckets": {"M": m},
        "dims": {"M": {"buckets": [32, 64, 320], "bucket": m}},
        "preconditions": pres, "dtypes": dtypes,
        "compute_type": "CUBLAS_COMPUTE_32F", "scale_type": "CUDA_R_32F", "transa": if nn { "N" } else { "T" }, "transb": if at { "T" } else { "N" },
        "m": m, "n": n, "k": k, "lda": if nn { n } else { k }, "ldb": if at { m } else { k }, "ldc": n, "ldd": n,
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
    let cases: [(&str, Change, &str); 36] = [
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
        ("transa X", |e| e["transa"] = "X".into(), "transa and transb must each be \"T\" or \"N\""),
        ("transb X", |e| e["transb"] = "X".into(), "transa and transb must each be \"T\" or \"N\""),
        // T, T is the layout TT, whose ldb is m
        ("transb T", |e| e["transb"] = "T".into(), "the leading dimensions of the layout TT must be lda = k, ldb = m"),
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

/// The fields an entry has only when its declaration is not the default (`docs/manifest-schema-3.md`, section 9): absent they are the
/// default (an entry tilefish 0.0.4 wrote reads as before), present they have their one value, and anything else is refused.
#[test]
fn how_an_algorithm_was_chosen_is_read_from_fields_that_are_absent_by_default() {
    let manifest = read(&[], &[gemm(true, 64, 1024, 1024)]).unwrap();
    let e = manifest.library_entry("gemm_bias.M64", "sm_103").unwrap();
    assert_eq!(
        (e.heuristic, e.algo_source, e.allow_bf16_reduction),
        (Heuristic::Tilefish, AlgoSource::Heuristic, false)
    );
    let mut entry = gemm(true, 64, 1024, 1024);
    entry["heuristic"] = "torch".into();
    entry["algo_source"] = "imported".into();
    entry["allow_bf16_reduction"] = true.into();
    let manifest = read(&[], &[entry.clone()]).unwrap();
    let e = manifest.library_entry("gemm_bias.M64", "sm_103").unwrap();
    assert_eq!((e.heuristic, e.algo_source, e.allow_bf16_reduction), (Heuristic::Torch, AlgoSource::Imported, true));
    // each of them on its own
    for (key, value) in
        [("heuristic", json!("torch")), ("algo_source", json!("imported")), ("allow_bf16_reduction", json!(true))]
    {
        let mut one = gemm(false, 32, 1024, 1024);
        one[key] = value;
        assert!(read(&[], &[one]).is_ok(), "{key}");
    }
    type Change = fn(&mut Value);
    let cases: [(&str, Change, &str); 9] = [
        (
            "the default heuristic written out",
            |e| e["heuristic"] = "tilefish".into(),
            "heuristic is absent (the default) or \"torch\", got \"tilefish\"",
        ),
        ("a heuristic that is not one", |e| e["heuristic"] = "Torch".into(), "heuristic is absent"),
        ("a heuristic that is null", |e| e["heuristic"] = Value::Null, "heuristic is absent"),
        (
            "the default source written out",
            |e| e["algo_source"] = "heuristic".into(),
            "algo_source is absent (the default) or \"imported\"",
        ),
        ("a source that is not one", |e| e["algo_source"] = 1.into(), "algo_source is absent"),
        (
            "the opt-in written as false",
            |e| e["allow_bf16_reduction"] = false.into(),
            "allow_bf16_reduction is absent (the default) or true, got false",
        ),
        ("the opt-in as 1", |e| e["allow_bf16_reduction"] = 1.into(), "allow_bf16_reduction is absent"),
        ("the opt-in as a string", |e| e["allow_bf16_reduction"] = "true".into(), "allow_bf16_reduction is absent"),
        ("a field of the same family that does not exist", |e| e["reduction"] = "bf16".into(), "unknown field"),
    ];
    for (what, change, why) in cases {
        let mut entry = gemm(false, 32, 1024, 1024);
        change(&mut entry);
        let message = refusal(&[], &[entry]);
        assert!(message.contains(why), "{what}: {message}");
        assert!(message.contains("library entry #0"), "{what}: {message}");
    }
    // the digest covers them: an entry that opts in is another document than one that does not
    let doc = |e: Value| -> Value { serde_json::from_str(&common::synthetic::manifest_text(&[], &[e])).unwrap() };
    let plain = manifest_digest(&doc(gemm(true, 64, 1024, 1024))).unwrap();
    assert_ne!(plain, manifest_digest(&doc(entry)).unwrap());
}

#[test]
fn n_and_k_may_be_any_size() {
    // Python's `check_declaration` takes any N and K: an algorithm that cannot take the leading dimensions is refused by the library when it
    // is loaded (`cublasLtMatmulAlgoCheck`)
    assert!(read(&[], &[gemm(false, 32, 8, 8), gemm(true, 64, 3072, 1024), gemm(false, 320, 1024, 3072)]).is_ok());
    for (n, k) in [(1020, 1024), (1024, 1028), (4, 1024), (1024, 1), (12, 12), (96, 9)] {
        for bias in [false, true] {
            let manifest = read(&[], &[gemm(bias, 32, n, k)]).unwrap();
            let e = &manifest.library()[0];
            assert_eq!((e.n, e.k), (n, k));
        }
    }
}

/// f16 operands, an f32 result and the layout NN (`b` is `[k, n]`) are read from the fields an entry always had: `dtypes`, `transa` and the
/// leading dimensions, the dtypes and shapes of the operands.
#[test]
fn an_entry_of_f16_operands_an_f32_result_or_the_layout_nn_is_read_with_its_types_and_layout() {
    let entries = [
        gemm_of(true, 32, 3072, 1024, "f16", "f16", "NN"),
        gemm_of(false, 64, 3072, 1024, "f16", "f32", "TN"),
        gemm_of(true, 320, 512, 1536, "bf16", "f32", "NN"),
        gemm_of(false, 32, 96, 9, "f16", "f16", "NN"),
    ];
    let manifest = read(&[], &entries).unwrap();
    let e = manifest.library_entry("gemm_bias.M32", "sm_103").unwrap();
    assert_eq!((e.dtype, e.out_dtype, e.layout), (DType::F16, DType::F16, GemmLayout::Nn));
    let dtypes: Vec<_> = e
        .params
        .iter()
        .map(|p| match &p.kind {
            ParamKind::Pointer { dtype, .. } => dtype.unwrap(),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(dtypes, [DType::F16, DType::F16, DType::F16, DType::F16]);
    let f32_out = manifest.library_entry("gemm.M64", "sm_103").unwrap();
    assert_eq!((f32_out.dtype, f32_out.out_dtype, f32_out.layout), (DType::F16, DType::F32, GemmLayout::Tn));
    let bf16_nn = manifest.library_entry("gemm_bias.M320", "sm_103").unwrap();
    assert_eq!((bf16_nn.dtype, bf16_nn.out_dtype, bf16_nn.layout), (DType::Bf16, DType::F32, GemmLayout::Nn));
    assert_eq!(
        (GemmLayout::Nn.as_str(), GemmLayout::Nn.lda(3072, 1024), GemmLayout::Nn.b_shape(3072, 1024)),
        ("NN", 3072, [1024, 3072])
    );
    // a bf16 entry of 0.0.5 reads as before: TN, bf16 in and out
    let old = read(&[], &[gemm(true, 64, 1024, 1024)]).unwrap();
    let e = &old.library()[0];
    assert_eq!(
        (e.dtype, e.out_dtype, e.layout, e.allow_f16_reduction),
        (DType::Bf16, DType::Bf16, GemmLayout::Tn, false)
    );
}

/// The layouts TT and NT (`a` stored `[k, m]`) are read from `transa`, `transb` and the leading dimensions, for every operand type, and their
/// plan binds `a` as `[K, M]`.
#[test]
fn an_entry_of_the_layout_tt_or_nt_is_read_and_binds_a_as_k_by_m() {
    let entries = [
        gemm_of(true, 32, 3072, 1024, "f16", "f16", "TT"),
        gemm_of(false, 64, 3072, 1024, "bf16", "f32", "NT"),
        gemm_of(true, 320, 96, 9, "f32", "f32", "NT"),
    ];
    let manifest = read(&[], &entries).unwrap();
    let tt = manifest.library_entry("gemm_bias.M32", "sm_103").unwrap();
    assert_eq!((tt.dtype, tt.out_dtype, tt.layout), (DType::F16, DType::F16, GemmLayout::Tt));
    let nt = manifest.library_entry("gemm.M64", "sm_103").unwrap();
    assert_eq!((nt.dtype, nt.out_dtype, nt.layout), (DType::Bf16, DType::F32, GemmLayout::Nt));
    let f32_nt = manifest.library_entry("gemm_bias.M320", "sm_103").unwrap();
    assert_eq!((f32_nt.dtype, f32_nt.layout, f32_nt.m, f32_nt.n, f32_nt.k), (DType::F32, GemmLayout::Nt, 320, 96, 9));
    for (layout, word, lda, ldb, a, b) in [
        (GemmLayout::Tn, "TN", 1024, 1024, [64, 1024], [3072, 1024]),
        (GemmLayout::Nn, "NN", 3072, 1024, [64, 1024], [1024, 3072]),
        (GemmLayout::Tt, "TT", 1024, 64, [1024, 64], [3072, 1024]),
        (GemmLayout::Nt, "NT", 3072, 64, [1024, 64], [1024, 3072]),
    ] {
        assert_eq!(
            (
                layout.as_str(),
                layout.lda(3072, 1024),
                layout.ldb(64, 1024),
                layout.a_shape(64, 1024),
                layout.b_shape(3072, 1024)
            ),
            (word, lda, ldb, a, b)
        );
        assert_eq!((layout.b_transposed(), layout.a_transposed()), (word.starts_with('T'), word.ends_with('T')));
    }
    let plan = manifest.library_plan("gemm", &[("M", 64)], Arch::sm(103)).unwrap();
    let at = |x: u64| 0x7000_0000_0000 + x * 0x1000_0000;
    let typed = |addr: u64, shape: &'static [usize], dtype: DType, access: Access| {
        let bytes = shape.iter().product::<usize>() as u64 * dtype.size() as u64;
        // SAFETY: nothing is dereferenced: `prepare` only looks at the numbers.
        let region = unsafe { DevicePtr::from_raw(addr, bytes, 0) };
        TensorArg::new(region, shape, dtype, access).unwrap()
    };
    let call = |a: &'static [usize]| {
        plan.prepare(
            &[
                typed(at(0), a, DType::Bf16, Access::In),
                typed(at(1), &[1024, 3072], DType::Bf16, Access::In),
                typed(at(2), &[64, 3072], DType::F32, Access::Out),
            ],
            None,
        )
    };
    assert!(call(&[1024, 64]).is_ok());
    let e = call(&[64, 1024]).unwrap_err();
    assert_eq!(e.precondition_kind(), Some(PreconditionKind::StaticSize), "a given as [M, K] to an NT GEMM: {e}");
    // the leading dimensions of a stored [k, m]: ldb = k is refused, with the rule in Python's words
    let mut wrong = gemm_of(false, 64, 3072, 1024, "bf16", "f32", "NT");
    wrong["ldb"] = 1024.into();
    assert!(refusal(&[], &[wrong]).contains(
        "the leading dimensions of the layout NT must be lda = n, ldb = m and ldc = ldd = n (cuBLAS sees b [k, n] as its A, a [k, m] as its B \
         and the result as C and D)"
    ));
    let mut wrong = gemm_of(true, 32, 3072, 1024, "f16", "f16", "TT");
    wrong["abi"]["params"][0]["shape"] = json!(["M", 1024]);
    assert!(refusal(&[], &[wrong]).contains("m, n and k say"));
}

/// The plan of a GEMM of the layout NN binds `b` as `[K, N]` and takes tensors of the entry's dtypes.
#[test]
fn the_operands_of_an_nn_gemm_are_bound_as_k_by_n_and_of_the_entrys_dtypes() {
    let manifest = read(&[], &[gemm_of(true, 64, 3072, 1024, "f16", "f32", "NN")]).unwrap();
    let plan = manifest.library_plan("gemm_bias", &[("M", 64)], Arch::sm(103)).unwrap();
    let at = |x: u64| 0x7000_0000_0000 + x * 0x1000_0000;
    let typed = |addr: u64, shape: &'static [usize], dtype: DType, access: Access| {
        let bytes = shape.iter().product::<usize>() as u64 * dtype.size() as u64;
        // SAFETY: nothing is dereferenced: `prepare` only looks at the numbers.
        let region = unsafe { DevicePtr::from_raw(addr, bytes, 0) };
        TensorArg::new(region, shape, dtype, access).unwrap()
    };
    let call = |b: &'static [usize], bias: DType, out: DType| {
        plan.prepare(
            &[
                typed(at(0), &[64, 1024], DType::F16, Access::In),
                typed(at(1), b, DType::F16, Access::In),
                typed(at(2), &[3072], bias, Access::In),
                typed(at(3), &[64, 3072], out, Access::Out),
            ],
            None,
        )
    };
    assert!(call(&[1024, 3072], DType::F32, DType::F32).is_ok());
    let e = call(&[3072, 1024], DType::F32, DType::F32).unwrap_err();
    assert_eq!(e.precondition_kind(), Some(PreconditionKind::StaticSize), "b given as [N, K] to an NN GEMM: {e}");
    let e = call(&[1024, 3072], DType::F16, DType::F32).unwrap_err();
    assert_eq!(
        e.precondition_kind(),
        Some(PreconditionKind::Dtype),
        "a bias of the operands' type where the result is f32: {e}"
    );
    let e = call(&[1024, 3072], DType::F32, DType::F16).unwrap_err();
    assert_eq!(e.precondition_kind(), Some(PreconditionKind::Dtype), "{e}");
}

#[test]
fn an_entry_whose_types_or_layout_disagree_is_refused() {
    type Change = fn(&mut Value);
    let cases: [(&str, Change, &str); 12] = [
        (
            "NN with lda = k",
            |e| e["lda"] = 1024.into(),
            "the leading dimensions of the layout NN must be lda = n, ldb = k",
        ),
        ("TN with the lda of NN", |e| e["transa"] = "T".into(), "the leading dimensions must be lda = ldb = k"),
        (
            "transb T: the layout NT, whose ldb is m",
            |e| e["transb"] = "T".into(),
            "the leading dimensions of the layout NT must be lda = n, ldb = m and ldc = ldd = n (cuBLAS sees b [k, n] as its A, a [k, m] as its B",
        ),
        ("b of another type", |e| e["dtypes"]["b"] = "bf16".into(), "dtypes must be"),
        ("an f32 bias with an f16 result", |e| e["dtypes"]["bias"] = "f32".into(), "dtypes must be"),
        ("a bf16 result of f16 operands", |e| e["dtypes"]["out"] = "bf16".into(), "dtypes must be"),
        (
            "f32 dtypes over f16 pointers",
            |e| e["dtypes"] = json!({"a": "f32", "b": "f32", "bias": "f32", "out": "f32"}),
            "must be the f32 pointer #0",
        ),
        (
            "a pointer of another type than dtypes",
            |e| e["abi"]["params"][2]["dtype"] = "f32".into(),
            "must be the f16 pointer #2",
        ),
        ("b given as [n, k]", |e| e["abi"]["params"][1]["shape"] = json!([3072, 1024]), "m, n and k say"),
        (
            "the opt-in of bf16 on f16 operands",
            |e| e["allow_bf16_reduction"] = true.into(),
            "allow_bf16_reduction is the opt-in of bf16 operands, and the operands of this entry are f16",
        ),
        (
            "the f16 opt-in written as false",
            |e| e["allow_f16_reduction"] = false.into(),
            "allow_f16_reduction is absent (the default) or true, got false",
        ),
        ("the f16 opt-in as 1", |e| e["allow_f16_reduction"] = 1.into(), "allow_f16_reduction is absent"),
    ];
    for (what, change, why) in cases {
        let mut entry = gemm_of(true, 32, 3072, 1024, "f16", "f16", "NN");
        change(&mut entry);
        let message = refusal(&[], &[entry]);
        assert!(message.contains(why), "{what}: {message}");
    }
    // the f16 opt-in on f16 operands is read; on bf16 operands it is refused
    let mut opted = gemm_of(false, 32, 3072, 1024, "f16", "f32", "TN");
    opted["allow_f16_reduction"] = true.into();
    let manifest = read(&[], &[opted]).unwrap();
    let e = &manifest.library()[0];
    assert!(e.allow_f16_reduction && !e.allow_bf16_reduction && e.allows_output_type_reduction());
    let mut wrong = gemm(false, 32, 1024, 1024);
    wrong["allow_f16_reduction"] = true.into();
    assert!(
        refusal(&[], &[wrong])
            .contains("allow_f16_reduction is the opt-in of f16 operands, and the operands of this entry are bf16")
    );
}

/// f32 operands (section 9.2) have an f32 result and bias, in both layouts, and opt in to split-K partial sums in the output with
/// `allow_f32_reduction`; a 16-bit result of f32 operands, or an opt-in of another type, is refused.
#[test]
fn an_entry_of_f32_operands_is_read_with_an_f32_result_and_its_own_opt_in() {
    let manifest =
        read(&[], &[gemm_of(true, 32, 1024, 3072, "f32", "f32", "TN"), gemm_of(false, 64, 96, 9, "f32", "f32", "NN")])
            .unwrap();
    let tn = manifest.library_entry("gemm_bias.M32", "sm_103").unwrap();
    assert_eq!((tn.dtype, tn.out_dtype, tn.layout), (DType::F32, DType::F32, GemmLayout::Tn));
    let dtypes: Vec<_> = tn
        .params
        .iter()
        .map(|p| match &p.kind {
            ParamKind::Pointer { dtype, .. } => dtype.unwrap(),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(dtypes, [DType::F32; 4]);
    assert!(!tn.allows_output_type_reduction());
    let nn = manifest.library_entry("gemm.M64", "sm_103").unwrap();
    assert_eq!((nn.dtype, nn.out_dtype, nn.layout, nn.k), (DType::F32, DType::F32, GemmLayout::Nn, 9));
    let mut opted = gemm_of(false, 32, 1024, 1024, "f32", "f32", "NN");
    opted["allow_f32_reduction"] = true.into();
    let manifest = read(&[], &[opted]).unwrap();
    let e = &manifest.library()[0];
    assert!(
        e.allow_f32_reduction && !e.allow_f16_reduction && !e.allow_bf16_reduction && e.allows_output_type_reduction()
    );
    type Change = fn(&mut Value);
    let cases: [(&str, Change, &str); 7] = [
        ("an f16 result of f32 operands", |e| e["dtypes"]["out"] = "f16".into(), "dtypes must be"),
        (
            "a bf16 result and bias of f32 operands",
            |e| {
                e["dtypes"]["out"] = "bf16".into();
                e["dtypes"]["bias"] = "bf16".into();
            },
            "dtypes must be",
        ),
        ("f32 a with an f16 b", |e| e["dtypes"]["b"] = "f16".into(), "dtypes must be"),
        ("a pointer of another type", |e| e["abi"]["params"][1]["dtype"] = "f16".into(), "must be the f32 pointer #1"),
        (
            "the f16 opt-in on f32 operands",
            |e| e["allow_f16_reduction"] = true.into(),
            "allow_f16_reduction is the opt-in of f16 operands, and the operands of this entry are f32",
        ),
        (
            "the f32 opt-in written as false",
            |e| e["allow_f32_reduction"] = false.into(),
            "allow_f32_reduction is absent (the default) or true, got false",
        ),
        ("the f32 opt-in as a string", |e| e["allow_f32_reduction"] = "true".into(), "allow_f32_reduction is absent"),
    ];
    for (what, change, why) in cases {
        let mut entry = gemm_of(true, 32, 1024, 3072, "f32", "f32", "TN");
        change(&mut entry);
        let message = refusal(&[], &[entry]);
        assert!(message.contains(why), "{what}: {message}");
    }
    let mut wrong = gemm(false, 32, 1024, 1024);
    wrong["allow_f32_reduction"] = true.into();
    assert!(
        refusal(&[], &[wrong])
            .contains("allow_f32_reduction is the opt-in of f32 operands, and the operands of this entry are bf16")
    );
}

/// `sm_count_target` (section 9.3): absent, the heuristic chose for the device's count; present, an integer from 8 to `sm_count`.
#[test]
fn an_sm_count_target_is_read_from_8_to_the_sm_count() {
    let mut entry = gemm_of(true, 32, 3072, 1024, "f16", "f16", "NN");
    entry["sm_count_target"] = 16.into();
    let manifest = read(&[], &[entry.clone()]).unwrap();
    let e = &manifest.library()[0];
    assert_eq!((e.sm_count, e.sm_count_target, e.sm_count_chosen_for()), (30, Some(16), 16));
    let plain = read(&[], &[gemm(true, 32, 1024, 1024)]).unwrap();
    assert_eq!((plain.library()[0].sm_count_target, plain.library()[0].sm_count_chosen_for()), (None, 30));
    for bad in [json!(7), json!(31), json!(0), json!(16.0), json!("16"), json!(true), Value::Null] {
        let mut wrong = entry.clone();
        wrong["sm_count_target"] = bad.clone();
        let message = refusal(&[], &[wrong]);
        assert!(
            message.contains("sm_count_target is absent or an int from 8 to sm_count (30), got"),
            "{bad}: {message}"
        );
    }
    // the digest covers it
    let doc = |e: Value| -> Value { serde_json::from_str(&common::synthetic::manifest_text(&[], &[e])).unwrap() };
    let mut other = entry.clone();
    other["sm_count_target"] = 24.into();
    assert_ne!(manifest_digest(&doc(entry)).unwrap(), manifest_digest(&doc(other)).unwrap());
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

// -- a GEMM with a cuBLASLt epilogue: impl cublaslt.matmul (section 9.5) ----------------------------------------------------------------------

/// The formula of each epilogue and `beta`, as the table of section 9.5 has it (Python's `tilefish._library.formula`).
const FORMULAS: [(&str, u64, &str); 8] = [
    ("none", 0, "acc"),
    ("none", 1, "acc + c"),
    ("bias", 0, "acc + bias"),
    ("bias", 1, "(acc + c) + bias"),
    ("relu", 0, "relu(acc)"),
    ("relu", 1, "relu(acc + c)"),
    ("relu_bias", 0, "relu(acc + bias)"),
    ("relu_bias", 1, "relu((acc + c) + bias)"),
];

/// The entry Python writes for a `cublaslt.matmul` `mm` of `dtype` operands in TN at the bucket `m`, with `epilogue` and `beta` (`c [M, n]` before
/// `out` when it is 1), measured-exact, with a canary.
fn matmul(epilogue: &str, beta: u64, m: u64, n: u64, k: u64, dtype: &str) -> Value {
    let bias = epilogue == "bias" || epilogue == "relu_bias";
    let mut e = gemm_of(bias, m, n, k, dtype, dtype, "TN");
    e["impl"] = "cublaslt.matmul".into();
    e["op"] = "mm".into();
    e["stem"] = format!("mm.M{m}").into();
    e["epilogue"] = epilogue.into();
    e["beta"] = beta.into();
    if beta == 1 {
        let params = e["abi"]["params"].as_array_mut().unwrap();
        let mut out = params.pop().unwrap();
        let i = params.len();
        params.push(
            json!({"index": i, "name": "c", "kind": "pointer", "dtype": dtype, "access": "in", "shape": ["M", n]}),
        );
        out["index"] = (i + 1).into();
        params.push(out);
        let pres = e["preconditions"].as_array_mut().unwrap();
        for p in pres.iter_mut() {
            if p["arg"] == "out" {
                p["index"] = (i + 1).into();
            }
        }
        pres.push(alignment(i, "c", 16));
        pres.push(json!({"kind": "static_size", "arg": "c", "index": i, "axis": 1, "value": n}));
        e["dtypes"]["c"] = dtype.into();
    }
    let formula = FORMULAS.iter().find(|(ep, b, _)| *ep == epilogue && *b == beta).map_or("acc", |(.., f)| *f);
    e["contract"] = "measured-exact".into();
    e["verification"] = json!({"formula": formula, "accumulator": "f32-form", "operand_sets": 2, "elements": 2 * m * n, "differ": 0, "max_ulp": 0});
    e["canary"] = json!({"seed": 1, "out_digest": "0123456789abcdef0123456789abcdef"});
    e
}

#[test]
fn a_matmul_entry_is_read_with_its_epilogue_its_beta_its_verification_and_its_canary() {
    use tilefish::{GemmEpilogue, MatmulAccumulator, MatmulContract};
    for (epilogue, beta, formula) in FORMULAS {
        let manifest = read(&[], &[matmul(epilogue, beta, 64, 1024, 512, "f16")]).unwrap();
        let e = manifest.library_entry("mm.M64", "sm_103").unwrap();
        let ep = GemmEpilogue::from_name(epilogue).unwrap();
        assert_eq!((e.implementation, e.epilogue, u64::from(e.beta)), (LibraryImpl::Matmul, ep, beta));
        assert_eq!((e.dtype, e.out_dtype, e.layout), (DType::F16, DType::F16, GemmLayout::Tn));
        assert_eq!(ep.formula(beta as u8), formula, "the formula of {epilogue} with beta {beta}");
        let mut names = vec!["a", "b"];
        if ep.has_bias() {
            names.push("bias");
        }
        if beta == 1 {
            names.push("c");
        }
        names.push("out");
        assert_eq!(e.params.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), names);
        assert_eq!((e.has_bias(), e.has_c()), (ep.has_bias(), beta == 1));
        assert_eq!(e.contract, Some(MatmulContract::MeasuredExact));
        let v = e.verification.as_ref().unwrap();
        assert_eq!(
            (v.formula.as_str(), v.accumulator, v.operand_sets, v.elements, v.differ, v.max_ulp),
            (formula, MatmulAccumulator::F32Form, 2, 2 * 64 * 1024, 0, 0)
        );
        let canary = e.canary.as_ref().unwrap();
        assert_eq!((canary.seed, canary.out_digest.as_str()), (1, "0123456789abcdef0123456789abcdef"));
        // the canary's operands: a, b, the bias and c when the entry has them, by their tags
        let mut want = vec![("a", 0, 64 * 512), ("b", 1, 1024 * 512)];
        if ep.has_bias() {
            want.push(("bias", 2, 1024));
        }
        if beta == 1 {
            want.push(("c", 3, 64 * 1024));
        }
        assert_eq!(e.canary_elements(), want);
        // the plan of a call: the operands in that order, c of the shape of out
        let plan = manifest.library_plan("mm", &[("M", 64)], Arch::sm(103)).unwrap();
        assert_eq!(plan.tensor_names().collect::<Vec<_>>(), names);
    }
    // the cuBLASLt values of the epilogues (CUBLASLT_EPILOGUE_DEFAULT, BIAS, RELU, RELU_BIAS)
    assert_eq!(GemmEpilogue::ALL.map(GemmEpilogue::cublaslt_value), [1, 4, 2, 6]);
    // a binary-defined pin and no canary: read as they are
    let mut entry = matmul("relu_bias", 1, 32, 1024, 3072, "bf16");
    entry["contract"] = "binary-defined".into();
    entry["verification"]["differ"] = 64.into();
    entry["verification"]["max_ulp"] = 15176.into();
    entry["verification"]["accumulator"] = "plain-f32-pin".into();
    entry.as_object_mut().unwrap().remove("canary");
    let manifest = read(&[], &[entry]).unwrap();
    let e = &manifest.library()[0];
    assert_eq!(e.contract, Some(MatmulContract::BinaryDefined));
    assert_eq!(
        e.verification.as_ref().map(|v| (v.differ, v.max_ulp, v.accumulator)),
        Some((64, 15176, MatmulAccumulator::PlainF32Pin))
    );
    assert!(e.canary.is_none());
    // a GEMM of the impls of section 9 has the epilogue of its impl, beta 0 and none of the rest
    let manifest = read(&[], &[gemm(true, 64, 1024, 1024), gemm(false, 32, 1024, 1024)]).unwrap();
    for (stem, epilogue) in [("gemm_bias.M64", GemmEpilogue::Bias), ("gemm.M32", GemmEpilogue::None)] {
        let e = manifest.library_entry(stem, "sm_103").unwrap();
        assert_eq!(
            (e.epilogue, e.beta, e.contract, e.verification.is_none(), e.canary.is_none()),
            (epilogue, 0, None, true, true)
        );
    }
}

/// Every operand of a matmul entry of `dtype`.
fn set_dtype(e: &mut Value, dtype: &str) {
    for key in ["a", "b", "bias", "c", "out"] {
        if e["dtypes"].get(key).is_some() {
            e["dtypes"][key] = dtype.into();
        }
    }
    for p in e["abi"]["params"].as_array_mut().unwrap() {
        p["dtype"] = dtype.into();
    }
}

fn remove(e: &mut Value, key: &str) {
    e.as_object_mut().unwrap().remove(key);
}

#[test]
fn a_matmul_entry_that_is_not_what_python_writes_is_refused() {
    type Change = fn(&mut Value);
    let cases: [(&str, Change, &str); 35] = [
        ("no beta", |e| remove(e, "beta"), "missing [\"beta\"]"),
        ("no contract", |e| remove(e, "contract"), "missing [\"contract\"]"),
        ("no verification", |e| remove(e, "verification"), "missing [\"verification\"]"),
        ("an imported algorithm", |e| e["algo_source"] = "imported".into(), "unknown [\"algo_source\"]"),
        ("an unknown field", |e| e["note"] = 1.into(), "unknown field"),
        (
            "an epilogue that is not one",
            |e| e["epilogue"] = "gelu".into(),
            "the epilogue of cublaslt.matmul must be one of",
        ),
        ("beta 2", |e| e["beta"] = 2.into(), "beta must be the int 0 or 1"),
        ("beta 1.0", |e| e["beta"] = 1.0.into(), "beta must be the int 0 or 1"),
        ("beta true", |e| e["beta"] = true.into(), "beta must be the int 0 or 1"),
        ("beta null", |e| e["beta"] = Value::Null, "beta must be the int 0 or 1"),
        ("a contract that is not one", |e| e["contract"] = "exact".into(), "contract must be one of"),
        (
            "a verification of another field",
            |e| e["verification"]["note"] = 0.into(),
            "verification must be an object of exactly",
        ),
        (
            "a verification without one",
            |e| remove(&mut e["verification"], "max_ulp"),
            "verification must be an object of exactly",
        ),
        (
            "a verification that is a list",
            |e| e["verification"] = json!([]),
            "verification must be an object of exactly",
        ),
        (
            "the formula of another epilogue",
            |e| e["verification"]["formula"] = "acc + bias".into(),
            "verification.formula must be",
        ),
        (
            "an accumulator that is not one",
            |e| e["verification"]["accumulator"] = "f64".into(),
            "verification.accumulator must be one of",
        ),
        (
            "no operand set",
            |e| e["verification"]["operand_sets"] = 0.into(),
            "verification.operand_sets must be a positive int",
        ),
        ("no element", |e| e["verification"]["elements"] = 0.into(), "verification.elements must be a positive int"),
        (
            "elements of another count",
            |e| e["verification"]["elements"] = 4096.into(),
            "verification.elements must be operand_sets * m * n = 131072, got 4096",
        ),
        (
            "three operand sets",
            |e| e["verification"]["operand_sets"] = 3.into(),
            "verification.elements must be operand_sets * m * n = 196608, got 131072",
        ),
        (
            "a negative differ",
            |e| e["verification"]["differ"] = (-1).into(),
            "verification.differ must be an int of at least 0",
        ),
        (
            "a max_ulp that is a float",
            |e| e["verification"]["max_ulp"] = 1.5.into(),
            "verification.max_ulp must be an int of at least 0",
        ),
        (
            "more elements that differ than elements",
            |e| {
                e["contract"] = "binary-defined".into();
                e["verification"]["differ"] = (2 * 64 * 1024 + 1).into();
            },
            "is more than verification.elements",
        ),
        (
            "measured-exact with elements that differ",
            |e| e["verification"]["differ"] = 3.into(),
            "but the verification found 3 elements",
        ),
        (
            "binary-defined with none that differs",
            |e| e["contract"] = "binary-defined".into(),
            "but the verification found 0 elements",
        ),
        ("a max_ulp with none that differs", |e| e["verification"]["max_ulp"] = 5.into(), "but no element differs"),
        ("a canary of null", |e| e["canary"] = Value::Null, "canary must be"),
        ("a canary without its digest", |e| remove(&mut e["canary"], "out_digest"), "canary must be"),
        (
            "a canary of a convolution",
            |e| e["canary"] = json!({"seed": 1, "y_digest": "ab".repeat(16)}),
            "canary must be",
        ),
        ("a canary seed of 2^32", |e| e["canary"]["seed"] = (1u64 << 32).into(), "canary must be"),
        ("a canary digest in upper case", |e| e["canary"]["out_digest"] = "AB".repeat(16).into(), "canary must be"),
        ("f32 operands", |e| set_dtype(e, "f32"), "dtypes must be"),
        ("an f32 result", |e| e["dtypes"]["out"] = "f32".into(), "dtypes must be"),
        (
            "the opt-in of f32 operands",
            |e| e["allow_f32_reduction"] = true.into(),
            "allow_f32_reduction is the opt-in of f32 operands",
        ),
        (
            "the opt-in of f16 operands",
            |e| e["allow_f16_reduction"] = true.into(),
            "allow_f16_reduction is the opt-in of f16 operands",
        ),
    ];
    for (what, change, why) in cases {
        let mut entry = matmul("relu_bias", 1, 64, 1024, 512, "bf16");
        change(&mut entry);
        let message = refusal(&[], &[entry]);
        assert!(message.contains(why), "{what}: {message}");
        assert!(message.contains("library entry #0"), "{what}: {message}");
    }
    // a bias exactly for bias and relu_bias, c exactly for beta 1: in dtypes and in the ABI
    let mut entry = matmul("relu", 0, 64, 1024, 512, "bf16");
    entry["dtypes"]["bias"] = "bf16".into();
    assert!(refusal(&[], &[entry]).contains("dtypes must be"));
    let mut entry = matmul("none", 0, 64, 1024, 512, "bf16");
    entry["dtypes"]["c"] = "bf16".into();
    assert!(refusal(&[], &[entry]).contains("dtypes must be"));
    let mut entry = matmul("bias", 1, 64, 1024, 512, "bf16");
    entry["beta"] = 0.into();
    entry["verification"]["formula"] = "acc + bias".into();
    let message = refusal(&[], &[entry]);
    assert!(message.contains("dtypes must be"), "{message}");
    let mut entry = matmul("bias", 1, 64, 1024, 512, "bf16");
    entry["abi"]["params"].as_array_mut().unwrap().remove(3);
    let message = refusal(&[], &[entry]);
    assert!(message.contains("abi.params must list 5 pointers (a, b, bias, c, out in this order)"), "{message}");
    // the fields of a cublaslt.matmul are refused on the impls of section 9, each by its name
    for key in ["beta", "contract", "verification", "canary"] {
        let mut entry = gemm(true, 32, 1024, 1024);
        entry[key] = matmul("bias", 0, 32, 1024, 1024, "bf16")[key].clone();
        let message = refusal(&[], &[entry]);
        assert!(message.contains(&format!("unknown [\"{key}\"]")), "{key}: {message}");
    }
    // and a gemm is not a cublaslt.matmul of another epilogue: its impl's holds
    let mut entry = gemm(false, 32, 1024, 1024);
    entry["epilogue"] = "relu".into();
    assert!(refusal(&[], &[entry]).contains("the epilogue of cublaslt.gemm must be \"none\""));
}
