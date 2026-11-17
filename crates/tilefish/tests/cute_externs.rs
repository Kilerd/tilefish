//! CuTe DSL kernels as externs (`docs/manifest-schema-3.md` section 4.9), read and checked with no GPU: a `binary` entry of source `cute`, its
//! toolchain, the digest of its PTX, its grid of integers, and the parameter kinds `tensormap` (a TMA descriptor the DSL encoded: a tensor, whose
//! launch passes the template with the tensor's address in its first eight bytes) and `bytes` (constant bytes), on synthetic manifests
//! (`common::synthetic`) and on the manifest Python's fake CuTe build exports (`tests/fixtures/cute_sm_103`, written by
//! `python/tests/export_rust_fixtures.py`, its digest and the `sha256` of its cubins the ones Python computed: `expected.json`, `"cute"`). The refusals are those of Python's reader (`_manifest._check_tensormap`, `_check_bytes`, `_check_cute_entry`).

mod common;

use common::synthetic::{
    Kernel, alignment, bytes_param, cute_template, fixed, hex, json, pointer, read, refusal, scalar, tensormap,
};
use serde_json::Value;
use tilefish::{
    Access, Arch, BinarySource, CUTE_ADDRESS_LIMIT, DType, DevicePtr, Error, MAX_PARAM_BYTES, Manifest, ParamKind, Pin,
    PreconditionKind, TensorArg, manifest_digest,
};

const A: u64 = 0x7F00_0010_0000; // 16-byte aligned
const B: u64 = 0x7F00_0020_0000;
const C: u64 = 0x7F00_0030_0000;

/// `gemm(a: Tensor[M, 64, f16], b: Tensor[64, 64, f16], c: Out[Tensor[M, 64, f16]])` built by the CuTe DSL for the bucket `M = 128`, as
/// `fake_cute_build` of `python/tests/test_cute_externs.py` makes it: `a` and `b` as TMA descriptors, four constant bytes, then `c` as a pointer
/// (its index 2, the one of the declaration; the bytes' index is their slot, 2 too), a cluster of 2 CTAs and the grid `(1, 2, 1)`.
fn gemm() -> Kernel {
    let mut c = pointer(3, "c", "f16", "out", json!(["M", 64]), None);
    c["index"] = 2.into();
    let mut k = Kernel::cute(
        "gemm.M128",
        103,
        vec![
            tensormap(0, 0, "a", "f16", "in", json!(["M", 64])),
            tensormap(1, 1, "b", "f16", "in", json!([64, 64])),
            bytes_param(2, &[0, 4, 0, 0]),
            c,
        ],
        vec![
            alignment(0, "a", 16),
            json!({"kind": "static_size", "arg": "a", "index": 0, "axis": 1, "value": 64}),
            json!({"kind": "max_elements", "arg": "a", "index": 0, "value": 2147483647, "shape": ["128", "64"]}),
            alignment(1, "b", 16),
            json!({"kind": "static_size", "arg": "b", "index": 1, "axis": 0, "value": 64}),
            json!({"kind": "static_size", "arg": "b", "index": 1, "axis": 1, "value": 64}),
            alignment(2, "c", 16),
            json!({"kind": "static_size", "arg": "c", "index": 2, "axis": 1, "value": 64}),
            json!({"kind": "bucket", "dim": "M", "value": 128, "buckets": [128, 256]}),
        ],
        json!({"M": {"buckets": [128, 256], "bucket": 128}}),
        "(1, 2, 1)",
        2,
    );
    k.entry["op"] = "gemm".into();
    k.entry["buckets"] = json!({"M": 128});
    k
}

fn manifest(k: Kernel) -> Manifest {
    read(&[k], &[]).expect("the manifest reads")
}

fn region(addr: u64, bytes: u64) -> DevicePtr<'static> {
    // SAFETY: the address is never dereferenced: `prepare` only looks at the number.
    unsafe { DevicePtr::from_raw(addr, bytes.max(1), 0) }
}

fn dense(addr: u64, shape: &[usize], access: Access) -> TensorArg<'_> {
    let bytes = shape.iter().product::<usize>() as u64 * 2;
    TensorArg::new(region(addr, bytes), shape, DType::F16, access).unwrap()
}

/// The tensors of a call of `gemm()`: `a` at `a`, `b` and `c` at their addresses.
fn call(a: u64, a_shape: &[usize]) -> Vec<TensorArg<'_>> {
    vec![dense(a, a_shape, Access::In), dense(B, &[64, 64], Access::In), dense(C, &[128, 64], Access::Out)]
}

#[test]
fn a_cute_entry_reads_its_binary_its_toolchain_and_its_parameters() {
    let m = manifest(gemm());
    let e = m.entry("gemm.M128", "sm_103").unwrap();
    assert_eq!(e.binary, Some(BinarySource::Cute { cutlass_dsl: "4.8.1".into(), options: "--opt-level 3".into() }));
    assert_eq!(e.nvvm.as_deref(), Some("Cuda compilation tools, release 13.4, V13.4.59"));
    assert_eq!((e.ptxas.as_deref(), e.triton.as_deref()), (Some("ptxas 12.9"), None));
    assert_eq!(e.source_digest_kind.as_deref(), Some("tilefish.cute.v1"));
    assert_eq!((e.ptx_target.as_str(), e.launch.cluster()), ("sm_103a", Some([2, 1, 1])));
    // one slot each: the descriptors 128 bytes, the constant bytes their size, the pointer 8 (what `cuFuncGetParamInfo` must report)
    let layout: Vec<(&str, usize, usize, Vec<usize>)> =
        e.params.iter().map(|p| (p.name.as_str(), p.slot, p.index, p.slots())).collect();
    assert_eq!(
        layout,
        [("a", 0, 0, vec![128]), ("b", 1, 1, vec![128]), ("param_2", 2, 2, vec![4]), ("c", 3, 2, vec![8])]
    );
    // a descriptor is a dense pointer with its template; the constant bytes are their value
    let a = e.param("a").unwrap();
    assert_eq!(a.template(), Some(&cute_template()));
    assert_eq!(a.size(), 128);
    assert!(a.tensor_map().is_none() && !a.is_device_scalar());
    assert!(matches!(
        &a.kind,
        ParamKind::Pointer {
            access: Access::In,
            dtype: Some(DType::F16),
            view: None,
            alias: None,
            template: Some(_),
            ..
        }
    ));
    assert!(matches!(&e.param("param_2").unwrap().kind, ParamKind::Bytes { value } if value == &[0, 4, 0, 0]));
    assert!(e.param("c").unwrap().template().is_none());
}

#[test]
fn a_descriptor_takes_a_tensor_and_the_constant_bytes_take_nothing() {
    let plan = manifest(gemm()).plan("gemm.M128", &[], Arch::sm(103)).unwrap();
    assert_eq!(plan.tensor_names().collect::<Vec<_>>(), ["a", "b", "c"]);
    assert_eq!(plan.given_names().count(), 0);
    assert_eq!((plan.scratch_slots(), plan.cluster(), plan.block()), (0, Some([2, 1, 1]), [128, 1, 1]));
    let e = plan.prepare(&call(A, &[128, 64])[..2], None).unwrap_err();
    assert!(e.to_string().contains("takes 3 tensors (a, b, c), got 2"), "{e}");
}

#[test]
fn a_launch_writes_each_address_into_its_template_and_passes_the_bytes() {
    let plan = manifest(gemm()).plan("gemm.M128", &[], Arch::sm(103)).unwrap();
    let prepared = plan.prepare(&call(A, &[128, 64]), None).unwrap();
    // the grid of blocks: the programs (1, 2, 1) times the cluster (2, 1, 1)
    assert_eq!((prepared.grid, prepared.cluster), ([2, 2, 1], Some([2, 1, 1])));
    // the 64-bit words: the pointer's address; the slots passed by value hold 0 there
    assert_eq!(prepared.params(), &[0, 0, 0, C]);
    // a descriptor: its template with the tensor's address, little-endian, in bytes 0 to 7, and nothing else changed
    let mut want = cute_template();
    want[..8].copy_from_slice(&A.to_le_bytes());
    assert_eq!(prepared.by_value(0), Some(&want[..]));
    let b = prepared.by_value(1).unwrap();
    assert_eq!((&b[..8], &b[8..]), (&B.to_le_bytes()[..], &cute_template()[8..]));
    // the constant bytes, verbatim
    assert_eq!(prepared.by_value(2), Some(&[0u8, 4, 0, 0][..]));
    assert_eq!(prepared.by_value(3), None);
    // another call writes its own addresses
    let other = plan.prepare(&call(C + 0x10_0000, &[128, 64]), None).unwrap();
    assert_eq!(&other.by_value(0).unwrap()[..8], &(C + 0x10_0000).to_le_bytes());
}

#[test]
fn a_descriptor_is_a_tensor_with_the_preconditions_of_one() {
    let plan = manifest(gemm()).plan("gemm.M128", &[], Arch::sm(103)).unwrap();
    let kind = |e: Error| e.precondition_kind().unwrap_or_else(|| panic!("not a precondition: {e}"));
    // it binds M: another bucket is refused, and so is a static axis of another size
    assert_eq!(kind(plan.prepare(&call(A, &[256, 64]), None).unwrap_err()), PreconditionKind::Bucket);
    let e = plan
        .prepare(
            &[dense(A, &[128, 64], Access::In), dense(B, &[64, 32], Access::In), dense(C, &[128, 64], Access::Out)],
            None,
        )
        .unwrap_err();
    assert_eq!(kind(e), PreconditionKind::StaticSize);
    // its pointer_alignment, 16 bytes, its dtype and its contiguity
    assert_eq!(kind(plan.prepare(&call(A + 8, &[128, 64]), None).unwrap_err()), PreconditionKind::PointerAlignment);
    let bf16 = TensorArg::new(region(A, 128 * 64 * 2), &[128, 64], DType::Bf16, Access::In).unwrap();
    let e =
        plan.prepare(&[bf16, dense(B, &[64, 64], Access::In), dense(C, &[128, 64], Access::Out)], None).unwrap_err();
    assert_eq!(kind(e), PreconditionKind::Dtype);
    let gaps =
        TensorArg::new_strided(region(A, 128 * 80 * 2), &[128, 64], &[80, 1], 0, DType::F16, Access::In).unwrap();
    let e =
        plan.prepare(&[gaps, dense(B, &[64, 64], Access::In), dense(C, &[128, 64], Access::Out)], None).unwrap_err();
    assert_eq!(kind(e), PreconditionKind::Contiguous);
    // a written tensor that overlaps a descriptor's
    let e = plan
        .prepare(
            &[dense(A, &[128, 64], Access::In), dense(B, &[64, 64], Access::In), dense(A, &[128, 64], Access::Out)],
            None,
        )
        .unwrap_err();
    assert_eq!(kind(e), PreconditionKind::Alias);
}

#[test]
fn an_address_the_descriptor_cannot_hold_is_refused_naming_the_parameter() {
    let plan = manifest(gemm()).plan("gemm.M128", &[], Arch::sm(103)).unwrap();
    for address in [CUTE_ADDRESS_LIMIT, CUTE_ADDRESS_LIMIT + 0x1000, u64::MAX - 0xFFFF] {
        match plan.prepare(&call(address, &[128, 64]), None) {
            Err(Error::Launch(message)) => {
                assert!(message.contains("argument 'a'") && message.contains("below 2**57"), "{message}");
            }
            other => panic!("{address:#x}: {other:?}"),
        }
    }
    // the last address it holds
    let last = CUTE_ADDRESS_LIMIT - 16;
    let prepared = plan.prepare(&call(last, &[128, 64]), None).unwrap();
    assert_eq!(&prepared.by_value(0).unwrap()[..8], &last.to_le_bytes());
}

#[test]
fn the_cute_export_python_wrote_is_read_with_pythons_digest() {
    // `python/tests/cute_fixture.py` (the fake CuTe build of `test_cute_externs.py`), as `export_rust_fixtures.py` wrote it, and what Python
    // computed of it: the digest `Manifest.load(...).digest` gave and the `sha256` of each cubin
    let want = &common::expected()["cute"]["cute_sm_103"];
    let python_digest = want["manifest_digest"].as_str().expect("the digest of the CuTe fixture in expected.json");
    let dir = common::fixtures().join("cute_sm_103");
    let text = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    assert_eq!(manifest_digest(&tilefish::json::parse_strict(&text).unwrap()).unwrap(), python_digest);
    let m = Manifest::load(&dir, Pin::Digest(python_digest)).unwrap();
    assert_eq!(m.entries().len(), 2);
    let kernels = want["kernels"].as_array().unwrap();
    assert_eq!(kernels.len(), 2);
    for k in kernels {
        let entry = m.entry(k["stem"].as_str().unwrap(), k["arch"].as_str().unwrap()).unwrap();
        assert_eq!(entry.cubin_name, k["cubin"].as_str().unwrap());
        let bytes = std::fs::read(dir.join(&entry.cubin_name)).unwrap();
        assert_eq!(tilefish::elf::sha256_hex(&bytes), k["sha256"].as_str().unwrap(), "{}", entry.cubin_name);
    }
    let e = m.entry("gemm.M256", "sm_103").unwrap();
    assert_eq!(e.binary, Some(BinarySource::Cute { cutlass_dsl: "4.8.1".into(), options: String::new() }));
    let kinds: Vec<(&str, Vec<usize>, bool)> =
        e.params.iter().map(|p| (p.name.as_str(), p.slots(), p.template().is_some())).collect();
    assert_eq!(
        kinds,
        [("a", vec![128], true), ("b", vec![128], true), ("param_2", vec![4], false), ("c", vec![8], false)]
    );
    assert_eq!(e.param("a").unwrap().template(), Some(&cute_template()));
    let plan = m.plan("gemm", &[("M", 256)], Arch::sm(103)).unwrap();
    let tensors =
        [dense(A, &[256, 64], Access::In), dense(B, &[64, 64], Access::In), dense(C, &[256, 64], Access::Out)];
    let prepared = plan.prepare(&tensors, None).unwrap();
    assert_eq!((prepared.grid, prepared.params()), ([2, 2, 1], &[0, 0, 0, C][..]));
    assert_eq!(prepared.by_value(2), Some(&[0u8, 4, 0, 0][..]));
    assert_eq!(&prepared.by_value(1).unwrap()[..8], &B.to_le_bytes());
}

#[test]
fn well_formed_versions_options_and_targets_are_read() {
    for version in ["4.8.1", "13.0", "1.2.3.4", "4.8.1.dev0+cu13", "4.8.1rc1", "4.8_post-1"] {
        let mut k = gemm();
        k.entry["binary"]["cutlass_dsl"] = version.into();
        k.entry["toolchain"]["cutlass_dsl"] = version.into();
        assert!(read(&[k], &[]).is_ok(), "{version}");
    }
    for options in ["", "--opt-level 3", "--opt-level=3 --x a,b:c/d+e_f.g"] {
        let mut k = gemm();
        k.entry["binary"]["options"] = options.into();
        assert!(read(&[k], &[]).is_ok(), "{options:?}");
    }
    for target in ["sm_103", "sm_103a", "sm_103f"] {
        let mut k = gemm();
        k.entry["ptx_target"] = target.into();
        assert!(read(&[k], &[]).is_ok(), "{target}");
    }
    // a grid of integers in any form, a cluster of 1 (launched with cuLaunchKernel), the largest constant bytes
    let mut k = gemm();
    k.entry["grid"] = "(2*3,)".into();
    k.entry["launch"]["cluster_dims"] = json!([1, 1, 1]);
    k.entry["launch"]["num_ctas"] = 1.into();
    k.entry["abi"]["params"][2] = bytes_param(2, &vec![7u8; MAX_PARAM_BYTES]);
    let m = manifest(k);
    let plan = m.plan("gemm.M128", &[], Arch::sm(103)).unwrap();
    let prepared = plan.prepare(&call(A, &[128, 64]), None).unwrap();
    assert_eq!((prepared.grid, prepared.cluster), ([6, 1, 1], None));
    assert_eq!(prepared.by_value(2).map(<[u8]>::len), Some(MAX_PARAM_BYTES));
    assert_eq!(m.entries()[0].params[2].slots(), [MAX_PARAM_BYTES]);
}

/// `edit` applied to the entry of `gemm()`, and the message the reader refuses it with must contain `why`.
fn refused(what: &str, edit: impl FnOnce(&mut Value), why: &str) {
    let mut k = gemm();
    edit(&mut k.entry);
    let message = refusal(&[k], &[]);
    assert!(message.contains(why), "{what}: expected {why:?} in {message:?}");
}

#[test]
fn the_reader_refuses_a_descriptor_that_is_not_what_it_says() {
    let p = |e: &mut Value, i: usize, key: &str, v: Value| e["abi"]["params"][i][key] = v;
    let zero_tail = |e: &mut Value| p(e, 0, "template", ("00".repeat(64) + &"ff".repeat(64)).into());
    refused(
        "an address in the template",
        |e| p(e, 0, "template", ("11".to_owned() + &"00".repeat(127)).into()),
        "bytes 0 to 7",
    );
    refused("bytes after 64 in the template", zero_tail, "and 64 to 127 of the template must be zero");
    refused(
        "a short template",
        |e| p(e, 0, "template", "00".repeat(127).into()),
        "template must be 256 lowercase hex digits",
    );
    refused(
        "an uppercase template",
        |e| p(e, 0, "template", hex(&cute_template()).to_uppercase().into()),
        "lowercase hex",
    );
    refused("a template of another type", |e| p(e, 0, "template", json!(0)), "template must be 256");
    refused(
        "no template",
        |e| drop(e["abi"]["params"][0].as_object_mut().unwrap().remove("template")),
        "template must be 256",
    );
    refused(
        "a view",
        |e| {
            p(e, 0, "strides", json!(["s", 1]));
            p(e, 0, "offset", json!("o"));
        },
        "has a dtype, an access, a shape and a template, and no [\"strides\", \"offset\"]",
    );
    refused("an alias", |e| p(e, 0, "alias", json!("b")), "and no [\"alias\"]");
    refused("a dim", |e| p(e, 0, "dim", json!("M")), "and no [\"dim\"]");
    refused("a tensor_map", |e| p(e, 0, "tensor_map", json!({})), "only a tensor descriptor (kind tensordesc) has one");
    refused("passed in 64 bytes", |e| p(e, 0, "size", json!(64)), "passed as 128 bytes");
    refused("passed as a u64", |e| p(e, 0, "ptx_type", json!("u64")), "passed as 128 bytes");
    refused("of a pointer type", |e| p(e, 0, "type", json!("*fp16")), "has type 'tensormap'");
    refused("of an unknown dtype", |e| p(e, 0, "dtype", json!("f8")), "dtype must be a dtype of the manifest");
    refused(
        "without a dtype",
        |e| drop(e["abi"]["params"][0].as_object_mut().unwrap().remove("dtype")),
        "dtype must be a dtype of the manifest, got None",
    );
    refused("of another access", |e| p(e, 0, "access", json!("rw")), "access must be in, inout or out");
    refused(
        "without a shape",
        |e| drop(e["abi"]["params"][0].as_object_mut().unwrap().remove("shape")),
        "shape must be a list",
    );
    refused("of an empty axis", |e| p(e, 1, "shape", json!([0, 64])), "shape[0] must be an int size of at least 1");
    // its alignment is the 16 bytes of the address the DSL keeps
    refused("aligned to 2 bytes", |e| e["preconditions"][0]["bytes"] = json!(2), "pointer_alignment must be 16 bytes");
    refused(
        "without an alignment",
        |e| drop(e["preconditions"].as_array_mut().unwrap().remove(0)),
        "pointer argument a has no pointer_alignment precondition",
    );
    // a template is a descriptor's only
    refused(
        "a template on a pointer",
        |e| p(e, 3, "template", hex(&cute_template()).into()),
        "has a template: only a TMA descriptor",
    );
}

#[test]
fn the_reader_refuses_constant_bytes_that_are_not_what_they_say() {
    let p = |e: &mut Value, key: &str, v: Value| e["abi"]["params"][2][key] = v;
    refused(
        "a value of 2 bytes",
        |e| p(e, "value", json!("0004")),
        "value must be 8 lowercase hex digits (its 4 bytes)",
    );
    refused("an uppercase value", |e| p(e, "value", json!("000A0000")), "value must be 8 lowercase hex digits");
    refused("a number", |e| p(e, "value", json!(1024)), "value must be 8 lowercase hex digits");
    refused(
        "no value",
        |e| drop(e["abi"]["params"][2].as_object_mut().unwrap().remove("value")),
        "value must be 8 lowercase hex digits",
    );
    refused("a dtype", |e| p(e, "dtype", json!("f16")), "is constant bytes: it has a value and no [\"dtype\"]");
    refused("a shape", |e| p(e, "shape", json!([4])), "and no [\"shape\"]");
    refused("another ptx_type", |e| p(e, "ptx_type", json!("b8[8]")), "ptx_type 'b8[<size>]'");
    refused("another type", |e| p(e, "type", json!("u32")), "constant bytes have type 'bytes'");
    refused(
        "no bytes",
        |e| {
            p(e, "size", json!(0));
            p(e, "ptx_type", json!("b8[0]"));
            p(e, "value", json!(""));
        },
        "a size from 1 to 32764",
    );
    refused(
        "more than the parameter space",
        |e| {
            let n = MAX_PARAM_BYTES + 1;
            p(e, "size", json!(n));
            p(e, "ptx_type", json!(format!("b8[{n}]")));
            p(e, "value", json!("00".repeat(n)));
        },
        "a size from 1 to 32764",
    );
    refused("a template", |e| p(e, "template", hex(&cute_template()).into()), "has a template: only a TMA descriptor");
}

#[test]
fn a_cute_kernel_takes_dense_pointers_descriptors_and_bytes_only() {
    let add = |e: &mut Value, param: Value| e["abi"]["params"].as_array_mut().unwrap().push(param);
    let mut n = scalar(4, "n", "i32", "M", json!({}));
    n["index"] = 3.into();
    refused("a scalar", |e| add(e, n), "parameter n is a scalar: a CuTe DSL kernel takes pointers, TMA descriptors");
    let mut k = fixed(4, "k", "i32", json!(1));
    k["index"] = 3.into();
    refused("a fixed argument", |e| add(e, k), "parameter k is a fixed: a CuTe DSL kernel takes pointers");
    refused(
        "a tensor descriptor",
        |e| e["abi"]["params"][3]["kind"] = "tensordesc".into(),
        "parameter c is a tensordesc: a CuTe DSL kernel takes",
    );
    refused(
        "a strided pointer",
        |e| {
            e["abi"]["params"][3]["strides"] = json!(["c_s0", 1]);
            e["abi"]["params"][3]["offset"] = json!("c_offset");
        },
        "parameter c is a view or an alias: a CuTe DSL kernel takes dense tensors",
    );
    refused("an alias", |e| e["abi"]["params"][3]["alias"] = json!("a"), "is a view or an alias");
    refused(
        "of an unknown kind",
        |e| e["abi"]["params"][2]["kind"] = "blob".into(),
        "parameter param_2 is a blob: a CuTe DSL kernel takes",
    );
    // the tensors keep the indices of the declaration, in its order (the constant bytes' index is their slot)
    refused(
        "two tensors of one index",
        |e| e["abi"]["params"][1]["index"] = 0.into(),
        "the indices 0, 1, 2, ... once each",
    );
    refused(
        "the tensors out of order",
        |e| {
            e["abi"]["params"][0]["index"] = 1.into();
            e["abi"]["params"][1]["index"] = 0.into();
        },
        "declaration order",
    );
    refused("a name twice", |e| e["abi"]["params"][2]["name"] = "a".into(), "argument names must be unique");
}

#[test]
fn the_reader_refuses_a_cute_entry_of_another_form() {
    refused(
        "no options",
        |e| e["binary"] = json!({"source": "cute", "cutlass_dsl": "4.8.1"}),
        "binary of a CuTe DSL kernel must be",
    );
    refused(
        "another key",
        |e| e["binary"]["ptxas_flags"] = json!([]),
        "binary of a CuTe DSL kernel must be {\"source\": \"cute\", \"cutlass_dsl\": <version>, \"options\": <options>}",
    );
    for version in ["four", "4", "4.", ".4.8", "v4.8.1", "4.8 1", "4.8.1;", ""] {
        refused(version, |e| e["binary"]["cutlass_dsl"] = version.into(), "binary of a CuTe DSL kernel must be");
    }
    refused(
        "a version of another type",
        |e| e["binary"]["cutlass_dsl"] = json!(4.8),
        "binary of a CuTe DSL kernel must be",
    );
    for options in [" --opt-level 3", "--opt-level 3 ", "--opt-level  3", "a;b", "--x=\"y\"", "--a\t3"] {
        refused(options, |e| e["binary"]["options"] = options.into(), "binary of a CuTe DSL kernel must be");
    }
    refused(
        "a toolchain without nvvm",
        |e| e["toolchain"] = json!({"cutlass_dsl": "4.8.1", "ptxas": "x"}),
        "the toolchain of a CuTe DSL kernel is {\"cutlass_dsl\", \"nvvm\", \"ptxas\"} (strings), got {\"cutlass_dsl\":\"4.8.1\",\"ptxas\":\"x\"}",
    );
    refused(
        "a toolchain with triton",
        |e| e["toolchain"]["triton"] = "3.5.1".into(),
        "the toolchain of a CuTe DSL kernel is",
    );
    refused("a toolchain with another key", |e| e["toolchain"]["nvcc"] = "13.0".into(), "unknown field");
    refused(
        "another DSL in the toolchain",
        |e| e["toolchain"]["cutlass_dsl"] = "4.8.0".into(),
        "toolchain.cutlass_dsl \"4.8.0\" is not binary.cutlass_dsl \"4.8.1\"",
    );
    refused(
        "a binary's digest kind",
        |e| e["source_digest_kind"] = "tilefish.binary.v1".into(),
        "records the sha256 of its PTX as source_digest, with source_digest_kind 'tilefish.cute.v1'",
    );
    refused(
        "no source digest",
        |e| {
            let o = e.as_object_mut().unwrap();
            o.remove("source_digest");
            o.remove("source_digest_kind");
        },
        "with source_digest_kind 'tilefish.cute.v1'",
    );
    refused(
        "a grid that names a dimension",
        |e| e["grid"] = "(M, 2, 1)".into(),
        "the grid of a CuTe DSL kernel is integers",
    );
    refused("a grid that is no grid", |e| e["grid"] = "(1, 2".into(), "grid");
    for target in ["sm_90a", "sm_1030", "sm_103ab", "sm_103A", "compute_103a", "", "sm_"] {
        refused(
            target,
            |e| e["ptx_target"] = target.into(),
            "ptx_target of a CuTe DSL kernel is the .target of its PTX for sm_103",
        );
    }
    refused(
        "a persistent grid",
        |e| {
            e["persistent"] = json!({"programs_per_sm": 1});
            e["grid"] = "(2,)".into();
        },
        "a CuTe DSL kernel has no persistent grid",
    );
    refused(
        "a source_sha256",
        |e| e["source_sha256"] = "5".repeat(64).into(),
        "a CuTe DSL kernel has no source_sha256",
    );
    refused(
        "a cluster of 16",
        |e| {
            e["launch"]["cluster_dims"] = json!([4, 4, 1]);
            e["launch"]["num_ctas"] = 16.into();
        },
        "its launch attributes are the zero ones but cluster_dims",
    );
    refused("Triton's warps", |e| e["num_warps"] = 4.into(), "a prebuilt binary has no [\"num_warps\"]");
    refused(
        "constexprs",
        |e| e["constexprs"] = json!([{"index": 4, "name": "BLOCK", "value": 64}]),
        "a prebuilt binary has no constexprs",
    );
    refused("scratch slots", |e| e["abi"]["scratch_slots"] = 2.into(), "a binary kernel has 0 scratch slots");
}

#[test]
fn the_kinds_and_the_digest_of_a_cute_kernel_are_refused_on_any_other_entry() {
    // the entry made a cubin used as it is
    refused(
        "a cubin",
        |e| {
            e["binary"] = json!({"source": "cubin"});
            e["toolchain"] = json!({});
            e["source_digest_kind"] = "tilefish.binary.v1".into();
            e["source_digest"] = e["sha256"].clone();
        },
        "parameter a is a tensormap, which is a parameter of a CuTe DSL kernel (a binary entry of source 'cute'), and this is not one",
    );
    // a PTX binary whose parameter is constant bytes
    let mut k = Kernel::new(
        "binary",
        "blob",
        103,
        vec![pointer(0, "x", "f32", "inout", json!([64]), None), bytes_param(1, &[1, 2])],
        vec![alignment(0, "x", 16)],
        json!({}),
        "(1,)",
    );
    let message = refusal(&[k], &[]);
    assert!(message.contains("parameter param_1 is a bytes, which is a parameter of a CuTe DSL kernel"), "{message}");
    // a Triton kernel with a descriptor of a CuTe DSL kernel
    k = Kernel::new(
        "triton",
        "copy",
        103,
        vec![tensormap(0, 0, "x", "f32", "in", json!([64])), pointer(1, "y", "f32", "out", json!([64]), None)],
        vec![alignment(0, "x", 16), alignment(1, "y", 16)],
        json!({}),
        "(1,)",
    );
    assert!(refusal(&[k], &[]).contains("parameter x is a tensormap, which is a parameter of a CuTe DSL kernel"));
    // the digest kind of a CuTe DSL kernel, and the fields of its toolchain, on a PTX binary and a Triton kernel
    let ptx = || {
        Kernel::new(
            "binary",
            "axpy",
            103,
            vec![pointer(0, "x", "f32", "inout", json!([64]), None)],
            vec![alignment(0, "x", 16)],
            json!({}),
            "(1,)",
        )
    };
    let triton = || {
        Kernel::new(
            "triton",
            "axpy",
            103,
            vec![pointer(0, "x", "f32", "inout", json!([64]), None)],
            vec![alignment(0, "x", 16)],
            json!({}),
            "(1,)",
        )
    };
    for mut k in [ptx(), triton()] {
        k.entry["source_digest_kind"] = "tilefish.cute.v1".into();
        let message = refusal(&[k], &[]);
        assert!(message.contains("source_digest_kind 'tilefish.cute.v1' is a CuTe DSL kernel's"), "{message}");
    }
    let mut k = ptx();
    k.entry["toolchain"]["nvvm"] = "13.4".into();
    assert!(refusal(&[k], &[]).contains("cutlass_dsl and nvvm are a CuTe DSL kernel's"));
    let mut k = triton();
    k.entry["toolchain"]["cutlass_dsl"] = "4.8.1".into();
    assert!(refusal(&[k], &[]).contains("the toolchain of a triton kernel is"));
    // the two kinds, and the cute source, are what an older reader refuses: a binary section of another form names every form
    let mut k = ptx();
    k.entry["binary"] = json!({"source": "fatbin"});
    let message = refusal(&[k], &[]);
    assert!(
        message.contains(
            "binary must be {\"source\": \"cubin\"}, {\"source\": \"ptx\", \"ptxas_flags\": [...]} or {\"source\": \"cute\", ...}"
        ),
        "{message}"
    );
}
