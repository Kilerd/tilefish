//! TMA descriptors (manifest kind `tensordesc`, `docs/en/manifest-schema-3.md` section 5.5), read and checked with no GPU, on synthetic manifests
//! (`common::synthetic`): the reader takes a descriptor as a pointer with its `tensor_map` and `1 + 2 * rank` slots of `kernelParams`, refuses one
//! that is not what it says, and a launch binds the descriptor's tensor as a pointer's, refuses with a kind what the driver could not encode, and
//! computes the arguments of `cuTensorMapEncodeTiled` as Python does (`python/tests/test_tensordesc.py` has the same numbers).

mod common;

use common::synthetic::{Kernel, alignment, json, pointer, read, refusal, scalar, tensordesc};
use serde_json::Value;
use tilefish::{
    Access, Arch, DType, DevicePtr, Error, Manifest, ParamKind, PreconditionKind, TensorArg, TensorMapEncode,
};

const A: u64 = 0x7000_0000_0000; // 16-byte aligned
const B: u64 = 0x7100_0000_0000;
const C: u64 = 0x7200_0000_0000;
const X: u64 = 0x7300_0000_0000;

fn kind(e: &Error) -> PreconditionKind {
    e.precondition_kind().unwrap_or_else(|| panic!("not a precondition error: {e:?}"))
}

/// `mm(a: TensorDesc(Tensor[R, K, f16], block=(128, 64)), b: TensorDesc(Tensor[C, K, f16], block=(128, 64)),
/// c: Out[TensorDesc(Tensor[R, C, f16], block=(128, 128))], k_len: K)`: three descriptors (the last one's box is half its block), then a scalar.
fn mm() -> Kernel {
    let mut k_len = scalar(15, "k_len", "i32", "K", json!({}));
    k_len["index"] = 3.into();
    Kernel::new(
        "triton",
        "mm",
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
    )
}

/// `view_copy(x: TensorDesc(Strided[Tensor[R, C, bf16]], block=(64, 32)), y: Out[Tensor[R, C, bf16]])`: a descriptor of a view (its own stride
/// dimension, carried by no scalar), then a pointer.
fn view_copy() -> Kernel {
    let mut y = pointer(5, "y", "bf16", "out", json!(["R", "C"]), None);
    y["index"] = 1.into();
    Kernel::new(
        "triton",
        "view_copy",
        103,
        vec![
            tensordesc(
                0,
                0,
                "x",
                "bf16",
                "in",
                json!(["R", "C"]),
                json!([64, 32]),
                json!([64, 32]),
                64,
                Some((json!(["x_s0", 1]), "x_offset")),
            ),
            y,
        ],
        vec![alignment(0, "x", 16), alignment(1, "y", 16)],
        json!({"R": {}, "C": {}, "x_s0": {}, "x_offset": {"divisible_by": 8}}),
        "(cdiv(R, 64), cdiv(C, 32))",
    )
}

fn manifest(k: Kernel) -> Manifest {
    read(&[k], &[]).expect("the manifest reads")
}

fn region(addr: u64, bytes: u64) -> DevicePtr<'static> {
    // SAFETY: the address is never dereferenced: `prepare` only looks at the number.
    unsafe { DevicePtr::from_raw(addr, bytes.max(1), 0) }
}

fn dense(addr: u64, shape: &[usize], dtype: DType, access: Access) -> TensorArg<'_> {
    let bytes = shape.iter().product::<usize>() as u64 * dtype.size() as u64;
    TensorArg::new(region(addr, bytes), shape, dtype, access).unwrap()
}

fn strided<'a>(addr: u64, shape: &'a [usize], strides: &'a [usize], dtype: DType) -> TensorArg<'a> {
    let reach = 1 + shape.iter().zip(strides).map(|(n, s)| (n - 1) * s).sum::<usize>();
    TensorArg::new_strided(region(addr, (reach * dtype.size()) as u64), shape, strides, 0, dtype, Access::In).unwrap()
}

#[test]
fn a_descriptor_is_a_pointer_with_its_map_and_its_slots() {
    let m = manifest(mm());
    let entry = m.entry("mm", "sm_103").unwrap();
    let layout: Vec<(&str, usize, Vec<usize>)> =
        entry.params.iter().map(|p| (p.name.as_str(), p.slot, p.slots())).collect();
    assert_eq!(
        layout,
        [
            ("a", 0, vec![128, 4, 4, 8, 8]),
            ("b", 5, vec![128, 4, 4, 8, 8]),
            ("c", 10, vec![128, 4, 4, 8, 8]),
            ("k_len", 15, vec![4])
        ]
    );
    let c = entry.param("c").unwrap();
    assert_eq!(c.size(), 128);
    let map = c.tensor_map().expect("a map");
    assert_eq!(
        (map.block.as_slice(), map.box_shape.as_slice(), map.swizzle, map.l2_promotion, map.oob_nan),
        (&[128, 128][..], &[128, 64][..], 128, 128, false)
    );
    assert_eq!((map.data_type_value(), map.swizzle_value(), map.l2_promotion_value()), (6, 3, 2));
    assert!(matches!(&c.kind, ParamKind::Pointer { access: Access::Out, tensor_map: Some(_), .. }));
    assert!(entry.param("k_len").unwrap().tensor_map().is_none());
    let v = manifest(view_copy());
    assert!(matches!(
        &v.entry("view_copy", "sm_103").unwrap().params[0].kind,
        ParamKind::Pointer { view: Some(_), tensor_map: Some(_), .. }
    ));
}

#[test]
fn a_launch_passes_the_map_then_the_shape_and_the_strides_of_the_tensor() {
    let plan = manifest(mm()).plan("mm", &[], Arch::sm(103)).unwrap();
    let prepared = plan
        .prepare(
            &[
                dense(A, &[200, 72], DType::F16, Access::In),
                dense(B, &[136, 72], DType::F16, Access::In),
                dense(C, &[200, 136], DType::F16, Access::Out),
            ],
            None,
        )
        .unwrap();
    assert_eq!(prepared.grid, [2, 2, 1]);
    // the map's slot holds 0 (its bytes are passed from the encoded map), then the shape (i32) and the element strides (i64) of each tensor
    let params = prepared.params();
    assert_eq!(params.len(), 18);
    assert_eq!(&params[..5], &[0, 200, 72, 72, 1]);
    assert_eq!(&params[5..10], &[0, 136, 72, 72, 1]);
    assert_eq!(&params[10..15], &[0, 200, 136, 136, 1]);
    assert_eq!(&params[15..], &[72, 0, 0]);
    let maps: Vec<&TensorMapEncode> = prepared.tensor_maps().collect();
    assert_eq!(maps.len(), 3);
    assert_eq!(
        *maps[0],
        TensorMapEncode {
            data_type: 6,
            rank: 2,
            address: A,
            global_dim: vec![72, 200],
            global_strides: vec![144, 200 * 144],
            box_dim: vec![64, 128],
            element_strides: vec![1, 1],
            interleave: 0,
            swizzle: 3,
            l2_promotion: 2,
            oob_fill: 0,
        }
    );
    assert_eq!(
        (maps[2].address, maps[2].global_dim.as_slice(), maps[2].box_dim.as_slice()),
        (C, &[136, 200][..], &[64, 128][..])
    );
}

#[test]
fn a_descriptor_of_a_view_passes_the_views_strides() {
    let plan = manifest(view_copy()).plan("view_copy", &[], Arch::sm(103)).unwrap();
    let shape = [96usize, 192];
    let prepared = plan
        .prepare(&[strided(X, &shape, &[256, 1], DType::Bf16), dense(B, &shape, DType::Bf16, Access::Out)], None)
        .unwrap();
    assert_eq!(&prepared.params()[..5], &[0, 96, 192, 256, 1]);
    let map = prepared.tensor_maps().next().unwrap();
    assert_eq!(
        (map.global_strides.as_slice(), map.box_dim.as_slice(), map.swizzle),
        (&[512, 96 * 512][..], &[32, 64][..], 2)
    );
}

#[test]
fn a_tensor_the_driver_could_not_describe_is_refused_with_a_kind() {
    let plan = manifest(mm()).plan("mm", &[], Arch::sm(103)).unwrap();
    let (rows72, rows36, b72, b36, c_shape) =
        ([200usize, 72], [200usize, 36], [136usize, 72], [136usize, 36], [200usize, 136]);
    let call = |a: &[usize], a_addr: u64, b: &[usize], gaps: bool| {
        let a = if gaps { strided(a_addr, a, &[80, 1], DType::F16) } else { dense(a_addr, a, DType::F16, Access::In) };
        let b = dense(B, b, DType::F16, Access::In);
        let c = dense(C, &c_shape, DType::F16, Access::Out);
        plan.prepare(&[a, b, c], None)
    };
    // 2 bytes past a 16-byte boundary
    assert_eq!(kind(&call(&rows72, A + 2, &b72, false).unwrap_err()), PreconditionKind::PointerAlignment);
    // rows of 72 bytes
    let e = call(&rows36, A, &b36, false).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::TensorMap, "{e}");
    assert!(e.to_string().contains("multiples of 16 bytes"), "{e}");
    // a tensor with gaps given to a dense descriptor
    assert_eq!(kind(&call(&rows72, A, &b72, true).unwrap_err()), PreconditionKind::Contiguous);
    // a dimension beyond i32
    let huge = TensorArg::new(region(A, (1u64 << 31) * 64 * 2), &[1usize << 31, 64], DType::F16, Access::In).unwrap();
    let e = plan
        .prepare(
            &[
                huge,
                dense(B, &[128, 64], DType::F16, Access::In),
                TensorArg::new(region(C, (1u64 << 31) * 128 * 2), &[1usize << 31, 128], DType::F16, Access::Out)
                    .unwrap(),
            ],
            None,
        )
        .unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::TensorMap, "{e}");
    assert!(e.to_string().contains("i32"), "{e}");

    let plan = manifest(view_copy()).plan("view_copy", &[], Arch::sm(103)).unwrap();
    let shape = [96usize, 192];
    // a row stride of 392 bytes, a last stride that is not 1, an address 8 bytes off, an expanded view (stride 0: the driver takes it)
    for (strides, addr, want) in [
        ([196usize, 1], X, Some(PreconditionKind::TensorMap)),
        ([256, 2], X, Some(PreconditionKind::Stride)),
        ([256, 1], X + 8, Some(PreconditionKind::PointerAlignment)),
        ([0, 1], X, None),
    ] {
        let out = dense(B, &shape, DType::Bf16, Access::Out);
        let got = plan.prepare(&[strided(addr, &shape, &strides, DType::Bf16), out], None);
        match want {
            Some(k) => assert_eq!(kind(&got.unwrap_err()), k, "{strides:?} at {addr:#x}"),
            None => assert!(got.is_ok(), "{strides:?}: {got:?}"),
        }
    }
}

/// `edit` applied to the params of `mm()`, and the message the reader refuses it with must contain `why`.
fn refused(edit: impl Fn(&mut Vec<Value>), why: &str) {
    let mut k = mm();
    let mut params = k.entry["abi"]["params"].as_array().unwrap().clone();
    edit(&mut params);
    k.entry["abi"]["params"] = Value::Array(params);
    let message = refusal(&[k], &[]);
    assert!(message.contains(why), "{why}: {message}");
}

#[test]
fn the_reader_refuses_a_descriptor_that_is_not_what_it_says() {
    refused(|p| drop(p[0].as_object_mut().unwrap().remove("tensor_map")), "tensor_map must be an object");
    refused(|p| p[0]["tensor_map"]["box"] = json!([512, 64]), "1 to 256");
    refused(|p| p[0]["tensor_map"]["box"] = json!([48, 64]), "does not tile");
    refused(|p| p[0]["tensor_map"]["block"] = json!([128, 48]), "power of two");
    refused(|p| p[0]["tensor_map"]["box"] = json!([128, 4]), "not a multiple of 16");
    refused(|p| p[0]["tensor_map"]["swizzle"] = json!("CU_TENSOR_MAP_SWIZZLE_32B"), "swizzle span");
    refused(|p| p[0]["tensor_map"]["swizzle"] = json!("CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B"), "swizzle must be one of");
    refused(|p| p[0]["tensor_map"]["data_type"] = json!("CU_TENSOR_MAP_DATA_TYPE_FLOAT32"), "data_type");
    refused(|p| p[0]["tensor_map"]["element_strides"] = json!([1, 2]), "element_strides");
    refused(|p| p[0]["tensor_map"]["interleave"] = json!("CU_TENSOR_MAP_INTERLEAVE_32B"), "interleave");
    refused(|p| p[0]["tensor_map"]["l2_promotion"] = json!("L2_128B"), "l2_promotion");
    refused(|p| p[0]["tensor_map"]["oob_fill"] = json!(1), "oob_fill");
    refused(|p| p[0]["tensor_map"]["extra"] = json!(1), "exactly the fields");
    refused(|p| p[0]["ptx_type"] = json!("u64"), "128 bytes of a CUtensorMap");
    refused(|p| p[0]["size"] = json!(8), "128 bytes of a CUtensorMap");
    refused(|p| p[0]["type"] = json!("*fp16"), "128 bytes of a CUtensorMap");
    refused(|p| p[0]["dtype"] = json!("e2m1"), "describes one of");
    refused(|p| p[2]["alias"] = json!("a"), "no dim, count_of, stride_of, value or alias");
    refused(|p| p[0]["shape"] = json!(["R", "K", 1, 1, 1, 1]), "1 to 5 axes");
    refused(
        |p| {
            p[2]["strides"] = json!(["c_s0", 1]);
            p[2]["offset"] = json!("c_offset");
        },
        "only an `in` pointer may be strided",
    );
    refused(|p| p[1]["slot"] = json!(1), "its slots begin at 5");
    refused(|p| p[3]["tensor_map"] = p[0]["tensor_map"].clone(), "only a tensor descriptor");
    // a pointer whose dtype has no map, and a descriptor on a prebuilt binary
    let mut k = mm();
    k.entry["kind"] = json!("binary");
    let message = refusal(&[k], &[]);
    assert!(!message.is_empty());
    // the last stride of a descriptor of a view is the fixed 1, the others names
    let mut k = view_copy();
    k.entry["abi"]["params"][0]["strides"] = json!(["x_s0", "x_s1"]);
    k.entry["dims"]["x_s1"] = json!({});
    assert!(refusal(&[k], &[]).contains("the fixed 1"));
    // a max_elements precondition, and an alignment other than 16 bytes, are not those of a descriptor
    let mut k = mm();
    k.entry["preconditions"]
        .as_array_mut()
        .unwrap()
        .push(json!({"kind": "max_elements", "arg": "a", "index": 0, "value": 2147483647, "shape": ["R", "K"]}));
    assert!(refusal(&[k], &[]).contains("max_elements must be once, on a dense pointer"));
    let mut k = mm();
    k.entry["preconditions"][0]["bytes"] = json!(2);
    assert!(refusal(&[k], &[]).contains("pointer_alignment"));
    // a scalar that names a descriptor's axis as the stride it carries
    let mut k = view_copy();
    let mut s = scalar(6, "x_s0", "i32", "x_s0", json!({"stride_of": {"arg": "x", "axis": 0}}));
    s["index"] = 2.into();
    k.entry["abi"]["params"].as_array_mut().unwrap().push(s);
    k.entry["abi"]["scratch"][0]["slot"] = 7.into();
    k.entry["abi"]["scratch"][1]["slot"] = 8.into();
    assert!(refusal(&[k], &[]).contains("passes its strides in its own slots"));
    // the scratch slots follow the slots of the descriptors
    let mut k = mm();
    k.entry["abi"]["scratch"][0]["slot"] = 4.into();
    assert!(refusal(&[k], &[]).contains("scratch slots"));
}
