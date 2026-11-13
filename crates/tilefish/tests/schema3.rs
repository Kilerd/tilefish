//! What manifest schema 3 adds, read and checked with no GPU: strided views, device scalars, 64-bit scalars, fixed arguments, prebuilt
//! binaries (no scratch slots) and the `library` section, on synthetic manifests around a small ELF file (`common::synthetic`).
//!
//! Each test builds one entry the way Python writes it (`docs/manifest-schema-3.md`), loads it through `Manifest::from_embedded`, and
//! either asks the plan of the entry what it decides for a call made of regions (nothing is dereferenced: `Plan::prepare` looks at the
//! numbers) or breaks the entry in one way and shows it refused.

mod common;

use common::synthetic::{Kernel, alignment, fixed, json, max_extent, pointer, read, refusal, scalar, stride_scalar};
use serde_json::Value;
use tilefish::{
    Access, Arch, DType, DevicePtr, Error, Manifest, ParamKind, Plan, PreconditionKind, StrideEntry, TensorArg,
};

// -- calls -----------------------------------------------------------------------------------------------------------------------

/// A dense tensor over a made-up region at `addr` (never dereferenced).
fn dense(addr: u64, shape: &[usize], dtype: DType, access: Access) -> TensorArg<'_> {
    let bytes = shape.iter().product::<usize>() as u64 * dtype.size() as u64;
    // SAFETY: the address is never dereferenced: `prepare` only looks at the number.
    let region = unsafe { DevicePtr::from_raw(addr, bytes.max(1), 0) };
    TensorArg::new(region, shape, dtype, access).unwrap()
}

/// A view whose first element is at `addr`, over a made-up region that holds exactly what it reaches.
fn view<'a>(addr: u64, shape: &'a [usize], strides: &'a [usize], dtype: DType) -> TensorArg<'a> {
    view_as(addr, shape, strides, dtype, Access::In)
}

fn view_as<'a>(addr: u64, shape: &'a [usize], strides: &'a [usize], dtype: DType, access: Access) -> TensorArg<'a> {
    let reach = 1 + shape.iter().zip(strides).map(|(n, s)| (n - 1) * s).sum::<usize>();
    // SAFETY: as above.
    let region = unsafe { DevicePtr::from_raw(addr, (reach * dtype.size()) as u64, 0) };
    TensorArg::new_strided(region, shape, strides, 0, dtype, access).unwrap()
}

fn kind(e: &Error) -> PreconditionKind {
    e.precondition_kind().unwrap_or_else(|| panic!("not a precondition error: {e:?}"))
}

const X: u64 = 0x7000_0000_0400; // 16-byte aligned
const Y: u64 = 0x7100_0000_0000;

// -- a strided pointer ---------------------------------------------------------------------------------------------------------

/// `head_copy(x: Strided[Tensor[S, 8, 64, bf16]], y: Out[Tensor[S, 8, 64, bf16]], x_s0: StrideOf("x", 0), x_s1: StrideOf("x", 1))`, with the
/// alignment `x_alignment` promised for `x` (2: the element's; 16: `Strided[..., 16]`) and, when `wide`, 64-bit strides and no `max_extent`.
fn head_copy(x_alignment: u64, wide: bool) -> Kernel {
    let ty = if wide { "i64" } else { "i32" };
    let mut pres = vec![alignment(0, "x", x_alignment), alignment(1, "y", 16)];
    if !wide {
        pres.push(max_extent(0, "x"));
    }
    let mut dims = json!({"S": {}, "x_s0": {}, "x_s1": {}, "x_offset": {}});
    if x_alignment == 16 {
        dims["x_offset"] = json!({"divisible_by": 8});
        dims["x_s0"] = json!({"divisible_by": 8});
        dims["x_s1"] = json!({"divisible_by": 8});
    }
    Kernel::new(
        "triton",
        if wide { "head_copy64" } else { "head_copy" },
        103,
        vec![
            pointer(0, "x", "bf16", "in", json!(["S", 8, 64]), Some((json!(["x_s0", "x_s1", 1]), "x_offset"))),
            pointer(1, "y", "bf16", "out", json!(["S", 8, 64]), None),
            stride_scalar(2, "x_s0", ty, "x", 0),
            stride_scalar(3, "x_s1", ty, "x", 1),
        ],
        pres,
        dims,
        "(S, 8)",
    )
}

fn plan_of(manifest: &Manifest, stem: &str) -> Plan {
    manifest.plan(stem, &[], Arch::sm(103)).unwrap()
}

#[test]
fn a_strided_pointer_is_read_with_its_strides_its_offset_and_the_scalars_that_carry_the_strides() {
    let manifest = read(&[head_copy(2, false)], &[]).unwrap();
    let entry = manifest.entry("head_copy", "sm_103").unwrap();
    let ParamKind::Pointer { view: Some(view), shape, access, .. } = &entry.params[0].kind else {
        panic!("x is a strided pointer")
    };
    assert_eq!(view.strides, [StrideEntry::Dim("x_s0".into()), StrideEntry::Dim("x_s1".into()), StrideEntry::Unit]);
    assert_eq!(view.offset, "x_offset");
    assert_eq!((shape.len(), *access), (3, Access::In));
    let ParamKind::Scalar { stride_of, dim, .. } = &entry.params[2].kind else { panic!("x_s0 carries a stride") };
    let [so] = stride_of.as_slice() else { panic!("x_s0 carries the stride of one view") };
    assert_eq!((so.arg.as_str(), so.axis, dim.as_str()), ("x", 0, "x_s0"));
    assert_eq!(entry.scratch_slots, 2);
}

#[test]
fn a_view_of_a_fused_qkv_binds_its_strides_and_the_kernel_is_given_the_address_of_its_first_element() {
    let manifest = read(&[head_copy(2, false)], &[]).unwrap();
    let plan = plan_of(&manifest, "head_copy");
    // qkv[:, which] of a [7, 3, 8, 64] bf16 tensor: [7, 8, 64] with strides (1536, 64, 1)
    let (shape, strides) = ([7usize, 8, 64], [1536usize, 64, 1]);
    let out = [7usize, 8, 64];
    let prepared = plan
        .prepare(&[view(X, &shape, &strides, DType::Bf16), dense(Y, &out, DType::Bf16, Access::InOut)], None)
        .unwrap();
    assert_eq!(prepared.grid, [7, 8, 1]);
    // the pointer, the pointer, the two strides, and the two null scratch slots of a Triton kernel
    assert_eq!(prepared.params(), [X, Y, 1536, 64, 0, 0]);
}

#[test]
fn a_dense_tensor_is_a_view_with_the_dense_strides_and_the_strides_of_size_one_axes_are_canonical() {
    let manifest = read(&[head_copy(2, false)], &[]).unwrap();
    let plan = plan_of(&manifest, "head_copy");
    let (shape, out) = ([4usize, 8, 64], [4usize, 8, 64]);
    let p = plan
        .prepare(&[dense(X, &shape, DType::Bf16, Access::In), dense(Y, &out, DType::Bf16, Access::InOut)], None)
        .unwrap();
    assert_eq!(p.params(), [X, Y, 512, 64, 0, 0]);
    // an unsqueeze: whatever stride the view has on a size-1 axis, the kernel is given the one a dense tensor would have there
    let (shape, strides, out) = ([1usize, 8, 64], [999_999_999_999usize, 64, 1], [1usize, 8, 64]);
    let p = plan
        .prepare(&[view(X, &shape, &strides, DType::Bf16), dense(Y, &out, DType::Bf16, Access::InOut)], None)
        .unwrap();
    assert_eq!(
        p.params(),
        [X, Y, 512, 64, 0, 0],
        "a stride that does not fit 32 bits, on an axis the kernel never steps along"
    );
}

#[test]
fn an_expanded_view_has_stride_zero_and_runs() {
    let manifest = read(&[head_copy(2, false)], &[]).unwrap();
    let plan = plan_of(&manifest, "head_copy");
    let (shape, strides, out) = ([40usize, 8, 64], [0usize, 64, 1], [40usize, 8, 64]);
    let p = plan
        .prepare(&[view(X, &shape, &strides, DType::Bf16), dense(Y, &out, DType::Bf16, Access::InOut)], None)
        .unwrap();
    assert_eq!(p.params(), [X, Y, 0, 64, 0, 0]);
    assert_eq!(p.grid, [40, 8, 1]);
}

#[test]
fn the_axis_the_kernel_loads_along_must_have_unit_stride() {
    let manifest = read(&[head_copy(2, false)], &[]).unwrap();
    let plan = plan_of(&manifest, "head_copy");
    // x[..., ::2]: a [40, 8, 64] view with strides (1024, 128, 2)
    let (shape, strides, out) = ([40usize, 8, 64], [1024usize, 128, 2], [40usize, 8, 64]);
    let e = plan
        .prepare(&[view(X, &shape, &strides, DType::Bf16), dense(Y, &out, DType::Bf16, Access::InOut)], None)
        .unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Stride, "{e}");
    assert!(e.to_string().contains("the stride of axis 2 is 2, but the signature requires 1"), "{e}");
}

#[test]
fn a_stride_that_does_not_fit_the_integer_the_binary_takes_is_refused_and_a_wide_one_is_not() {
    let narrow = read(&[head_copy(2, false)], &[]).unwrap();
    let wide = read(&[head_copy(2, true)], &[]).unwrap();
    // 2**31 along the first axis of a 2-row view
    let (shape, strides, out) = ([2usize, 8, 64], [1usize << 31, 64, 1], [2usize, 8, 64]);
    let call = || vec![view(X, &shape, &strides, DType::Bf16), dense(Y, &out, DType::Bf16, Access::InOut)];
    let e = plan_of(&narrow, "head_copy").prepare(&call(), None).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Shape, "{e}");
    assert!(e.to_string().contains("x_s0") && e.to_string().contains("32-bit"), "{e}");
    // the same view of a kernel indexing with 64 bits: i64 strides, no max_extent
    let p = plan_of(&wide, "head_copy64").prepare(&call(), None).unwrap();
    assert_eq!(p.params(), [X, Y, 1 << 31, 64, 0, 0]);
}

/// A Gluon kernel that takes `i32` strides and computes 64-bit offsets with them (`@tilefish.extern(gluon=..., index_bits=64)` with
/// `StrideOf("x", 0, dtype=i32)`): its entry is a `triton` one whose `source_digest_kind` is `tilefish.gluon-ir.v1`, its strides are `i32`
/// and there is no `max_extent`. The stride is held to the `i32` the binary takes, and the extent of the view to nothing.
fn head_copy_gluon() -> Kernel {
    let mut k = Kernel::new(
        "triton",
        "head_copy_gluon",
        103,
        vec![
            pointer(0, "x", "bf16", "in", json!(["S", 8, 64]), Some((json!(["x_s0", "x_s1", 1]), "x_offset"))),
            pointer(1, "y", "bf16", "out", json!(["S", 8, 64]), None),
            stride_scalar(2, "x_s0", "i32", "x", 0),
            stride_scalar(3, "x_s1", "i32", "x", 1),
        ],
        vec![alignment(0, "x", 2), alignment(1, "y", 16)],
        json!({"S": {}, "x_s0": {}, "x_s1": {}, "x_offset": {}}),
        "(S, 8)",
    );
    k.entry["source_digest_kind"] = "tilefish.gluon-ir.v1".into();
    k
}

#[test]
fn a_gluon_entry_with_i32_strides_and_no_offset_limit_is_read_and_its_strides_are_held_to_i32() {
    let manifest = read(&[head_copy_gluon()], &[]).unwrap();
    let entry = manifest.entry("head_copy_gluon", "sm_103").unwrap();
    assert_eq!(entry.source_digest_kind.as_deref(), Some("tilefish.gluon-ir.v1"));
    let plan = plan_of(&manifest, "head_copy_gluon");
    let out = [2usize, 8, 64];
    // the widest stride an i32 carries: a view that reaches past 2**31 elements, which no max_extent refuses
    let (shape, strides) = ([2usize, 8, 64], [(1usize << 31) - 1, 64, 1]);
    let p = plan
        .prepare(&[view(X, &shape, &strides, DType::Bf16), dense(Y, &out, DType::Bf16, Access::InOut)], None)
        .unwrap();
    assert_eq!(p.params(), [X, Y, (1 << 31) - 1, 64, 0, 0]);
    // one more does not fit the i32
    let strides = [1usize << 31, 64, 1];
    let e = plan
        .prepare(&[view(X, &shape, &strides, DType::Bf16), dense(Y, &out, DType::Bf16, Access::InOut)], None)
        .unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Shape, "{e}");
    assert!(e.to_string().contains("x_s0") && e.to_string().contains("32-bit"), "{e}");
}

#[test]
fn a_view_that_reaches_further_than_its_memory_cannot_be_made() {
    // SAFETY: nothing is dereferenced.
    let region = unsafe { DevicePtr::from_raw(X, 1024, 0) };
    // 40 rows of 64 bf16 values with a row stride of 128 elements need 39 * 128 + 64 elements = 10112 bytes
    let (shape, strides) = ([40usize, 64], [128usize, 1]);
    let e = TensorArg::new_strided(region, &shape, &strides, 0, DType::Bf16, Access::In).unwrap_err();
    assert!(e.to_string().contains("reaches 10112 bytes") && e.to_string().contains("has 1024 bytes"), "{e}");
    // the offset counts: 4 rows of 64 elements are 512 bytes, and they fit 1024 bytes from offset 0 to 256 elements, not from 257
    let (shape, strides) = ([4usize, 64], [64usize, 1]);
    assert!(TensorArg::new_strided(region, &shape, &strides, 256, DType::Bf16, Access::In).is_ok());
    assert!(TensorArg::new_strided(region, &shape, &strides, 257, DType::Bf16, Access::In).is_err());
    // the pointer of the view is the address of its first element
    let view = TensorArg::new_strided(region, &shape, &strides, 9, DType::Bf16, Access::In).unwrap();
    assert_eq!(view.addr(), X + 18);
    assert_eq!(view.strides(), Some(&strides[..]));
    // a stride of 0 adds nothing to what the view reaches, and a stride that is too large does
    let (shape, strides) = ([1000usize, 8], [0usize, 1]);
    assert!(TensorArg::new_strided(region, &shape, &strides, 0, DType::Bf16, Access::In).is_ok());
    let strides = [1usize << 62, 1];
    assert!(TensorArg::new_strided(region, &shape, &strides, 0, DType::Bf16, Access::In).is_err());
    // one stride per axis
    let strides = [1usize];
    let e = TensorArg::new_strided(region, &shape, &strides, 0, DType::Bf16, Access::In).unwrap_err();
    assert!(e.to_string().contains("one stride per axis"), "{e}");
    // a view whose first element is past the end of its memory
    let (shape, strides) = ([1usize], [1usize]);
    assert!(TensorArg::new_strided(region, &shape, &strides, 513, DType::Bf16, Access::In).is_err());
}

#[test]
fn the_extent_of_a_view_is_bounded_by_what_i32_offsets_reach() {
    let manifest = read(&[head_copy(2, false)], &[]).unwrap();
    let plan = plan_of(&manifest, "head_copy");
    // each stride fits an i32, and the view reaches 1999 * 2_000_000 + ... elements
    let (shape, strides, out) = ([2000usize, 8, 64], [2_000_000usize, 64, 1], [2000usize, 8, 64]);
    let e = plan
        .prepare(&[view(X, &shape, &strides, DType::Bf16), dense(Y, &out, DType::Bf16, Access::InOut)], None)
        .unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::MaxExtent, "{e}");
    assert!(e.to_string().contains("spans") && e.to_string().contains("2147483647"), "{e}");
    // the limit itself: a 2-row view whose second row starts 2147483135 elements in reaches 2147483135 + 7 * 64 + 63 + 1 = 2**31 - 1 elements
    let out = [2usize, 8, 64];
    let (shape, strides) = ([2usize, 8, 64], [2_147_483_135usize, 64, 1]);
    let ok = plan.prepare(&[view(X, &shape, &strides, DType::Bf16), dense(Y, &out, DType::Bf16, Access::InOut)], None);
    assert!(ok.is_ok(), "{ok:?}");
    // and one element more is not
    let strides = [2_147_483_136usize, 64, 1];
    let e = plan
        .prepare(&[view(X, &shape, &strides, DType::Bf16), dense(Y, &out, DType::Bf16, Access::InOut)], None)
        .unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::MaxExtent, "{e}");
    // a kernel that indexes with 64 bits has no such limit (its strides are i64)
    let wide = read(&[head_copy(2, true)], &[]).unwrap();
    let (shape, strides) = ([2000usize, 8, 64], [2_000_000usize, 64, 1]);
    let out = [2000usize, 8, 64];
    let ok = plan_of(&wide, "head_copy64")
        .prepare(&[view(X, &shape, &strides, DType::Bf16), dense(Y, &out, DType::Bf16, Access::InOut)], None);
    assert!(ok.is_ok(), "{ok:?}");
}

#[test]
fn a_strided_pointer_promises_its_elements_alignment_or_sixteen_bytes() {
    let (shape, strides, out) = ([7usize, 8, 64], [1536usize, 64, 1], [7usize, 8, 64]);
    let call = |x: u64| vec![view(x, &shape, &strides, DType::Bf16), dense(Y, &out, DType::Bf16, Access::InOut)];
    // `Strided[...]`: aligned to a bf16 (2 bytes), whatever the offset; an odd address is not
    let loose = read(&[head_copy(2, false)], &[]).unwrap();
    for offset in 0..9u64 {
        assert!(plan_of(&loose, "head_copy").prepare(&call(X + 2 * offset), None).is_ok(), "{offset}");
    }
    let e = plan_of(&loose, "head_copy").prepare(&call(X + 1), None).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::PointerAlignment, "{e}");
    assert!(e.to_string().contains("not aligned to 2 bytes"), "{e}");
    // `Strided[..., 16]`: the pointer passed is the address of the first element, and must be 16-byte aligned
    let tight = read(&[head_copy(16, false)], &[]).unwrap();
    for offset in [0u64, 8, 16] {
        assert!(plan_of(&tight, "head_copy").prepare(&call(X + 2 * offset), None).is_ok(), "{offset}");
    }
    for offset in [1u64, 2, 3, 4, 5, 6, 7, 9] {
        let e = plan_of(&tight, "head_copy").prepare(&call(X + 2 * offset), None).unwrap_err();
        assert_eq!(kind(&e), PreconditionKind::PointerAlignment, "{offset}: {e}");
        assert!(e.to_string().contains("not aligned to 16 bytes"), "{e}");
    }
    // the stride scalars of that variant are multiples of 8 (their dimensions are `divisible_by`): a stride of 12 is refused
    let (shape, strides) = ([7usize, 8, 64], [1548usize, 64, 1]);
    let e = plan_of(&tight, "head_copy")
        .prepare(&[view(X, &shape, &strides, DType::Bf16), dense(Y, &out, DType::Bf16, Access::InOut)], None)
        .unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Divisible, "{e}");
}

#[test]
fn a_tensor_for_a_parameter_that_is_not_strided_must_be_contiguous() {
    let manifest = read(&[head_copy(2, false)], &[]).unwrap();
    let plan = plan_of(&manifest, "head_copy");
    let (shape, strides) = ([7usize, 8, 64], [1536usize, 64, 1]);
    // `y` is a dense parameter: a view with gaps is refused, whatever it is a view of
    let x = [7usize, 8, 64];
    let y_view = view_as(Y, &shape, &strides, DType::Bf16, Access::InOut);
    let e = plan.prepare(&[dense(X, &x, DType::Bf16, Access::In), y_view], None).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Contiguous, "{e}");
    // a view with the dense strides is contiguous (and a size-1 axis's stride is not looked at)
    let ok_strides = [512usize, 64, 1];
    let y_dense = view_as(Y, &shape, &ok_strides, DType::Bf16, Access::InOut);
    assert!(plan.prepare(&[dense(X, &x, DType::Bf16, Access::In), y_dense], None).is_ok());
}

#[test]
fn a_view_made_for_a_parameter_that_is_written_is_refused_at_the_reading_of_the_manifest() {
    // only an `in` pointer takes strides
    let mut k = head_copy(2, false);
    k.entry["abi"]["params"][0]["access"] = "inout".into();
    assert!(refusal(&[k], &[]).contains("only an `in` pointer may be strided"));
}

// -- views that share a stride ----------------------------------------------------------------------------------------------------

/// `qkv_sum(q, k, v: Strided[Tensor[R, 4, 64, bf16]], o: Out[Tensor[R, 4, 64, bf16]], s_row: StrideOf(("q", "k", "v"), 0),
/// s_head: StrideOf(("q", "k", "v"), 1))`: one stride pair for the three views, each with its own offset.
fn qkv_sum() -> Kernel {
    let strided = |slot: usize, name: &str| {
        pointer(
            slot,
            name,
            "bf16",
            "in",
            json!(["R", 4, 64]),
            Some((json!(["s_row", "s_head", 1]), &*format!("{name}_offset"))),
        )
    };
    let shared =
        |axis: usize| json!([{"arg": "q", "axis": axis}, {"arg": "k", "axis": axis}, {"arg": "v", "axis": axis}]);
    Kernel::new(
        "triton",
        "qkv_sum",
        103,
        vec![
            strided(0, "q"),
            strided(1, "k"),
            strided(2, "v"),
            pointer(3, "o", "bf16", "out", json!(["R", 4, 64]), None),
            scalar(4, "s_row", "i32", "s_row", json!({"stride_of": shared(0)})),
            scalar(5, "s_head", "i32", "s_head", json!({"stride_of": shared(1)})),
        ],
        vec![
            alignment(0, "q", 2),
            max_extent(0, "q"),
            alignment(1, "k", 2),
            max_extent(1, "k"),
            alignment(2, "v", 2),
            max_extent(2, "v"),
            alignment(3, "o", 16),
        ],
        json!({"R": {}, "k_offset": {}, "q_offset": {}, "s_head": {}, "s_row": {}, "v_offset": {}}),
        "(R, 4)",
    )
}

#[test]
fn a_scalar_carries_the_one_stride_that_several_views_share() {
    let manifest = read(&[qkv_sum()], &[]).unwrap();
    let entry = manifest.entry("qkv_sum", "sm_103").unwrap();
    let ParamKind::Scalar { stride_of, .. } = &entry.params[4].kind else { panic!("s_row is a stride") };
    let carried: Vec<(&str, usize)> = stride_of.iter().map(|s| (s.arg.as_str(), s.axis)).collect();
    assert_eq!(carried, [("q", 0), ("k", 0), ("v", 0)]);
    // the q, k and v of a fused [7, 3, 4, 64] tensor: offsets 0, 256 and 512 elements, strides (768, 64, 1) each
    let (shape, strides, out) = ([7usize, 4, 64], [768usize, 64, 1], [7usize, 4, 64]);
    let (q, k, v) = (X, X + 512, X + 1024);
    let plan = plan_of(&manifest, "qkv_sum");
    let call = |k_strides: &'static [usize]| {
        vec![
            view(q, &shape, &strides, DType::Bf16),
            view(k, &shape, k_strides, DType::Bf16),
            view(v, &shape, &strides, DType::Bf16),
            dense(Y, &out, DType::Bf16, Access::InOut),
        ]
    };
    let p = plan.prepare(&call(&[768, 64, 1]), None).unwrap();
    assert_eq!(p.params(), [q, k, v, Y, 768, 64, 0, 0], "three pointers, and the one stride pair once");
    assert_eq!(p.grid, [7, 4, 1]);
    // a view that does not have the stride the first one bound is refused, as Python refuses it
    for (k_strides, axis, got) in
        [(&[769usize, 64, 1][..], 0, 769), (&[768, 65, 1][..], 1, 65), (&[256, 64, 1][..], 0, 256)]
    {
        let e = plan.prepare(&call(k_strides), None).unwrap_err();
        assert_eq!(kind(&e), PreconditionKind::Stride, "{e}");
        let text = e.to_string();
        assert!(text.contains("argument 'k'") && text.contains(&format!("the stride of axis {axis} is {got}")), "{e}");
        assert!(text.contains("each must have it"), "{e}");
    }
}

#[test]
fn a_shared_stride_that_is_not_what_it_says_is_refused() {
    let set = |s_row: Value| {
        let mut k = qkv_sum();
        k.entry["abi"]["params"][4]["stride_of"] = s_row;
        k
    };
    let cases: [(&str, Value, &str); 7] = [
        ("a list of one", json!([{"arg": "q", "axis": 0}]), "two or more"),
        (
            "a view twice",
            json!([{"arg": "q", "axis": 0}, {"arg": "q", "axis": 0}, {"arg": "k", "axis": 0}]),
            "names axis 0 of 'q' twice",
        ),
        ("an item that is not an object", json!([{"arg": "q", "axis": 0}, "k"]), "stride_of must be an object"),
        ("an item without its axis", json!([{"arg": "q", "axis": 0}, {"arg": "k"}]), "axis"),
        (
            "a pointer that is not strided",
            json!([{"arg": "q", "axis": 0}, {"arg": "o", "axis": 0}]),
            "not a strided pointer",
        ),
        ("another axis", json!([{"arg": "q", "axis": 0}, {"arg": "k", "axis": 1}]), "the strides of that pointer say"),
        ("a view left out", json!([{"arg": "q", "axis": 0}, {"arg": "k", "axis": 0}]), "carried by no scalar"),
    ];
    for (what, value, why) in cases {
        let message = refusal(&[set(value)], &[]);
        assert!(message.contains(why), "{what}: {message}");
    }
    // the axis of a view carried by two scalars: s_head made a second carrier of s_row
    let mut k = qkv_sum();
    k.entry["abi"]["params"][5] =
        scalar(5, "s_head", "i32", "s_row", json!({"stride_of": [{"arg": "q", "axis": 0}, {"arg": "k", "axis": 0}]}));
    let message = refusal(&[k], &[]);
    assert!(message.contains("axis 0 of 'q' has two stride scalars (s_row and s_head)"), "{message}");
}

// -- a dimension only an Out tensor determines ----------------------------------------------------------------------------------------

/// `key_mask(mask: Out[Tensor[shape, bf16]], n_valid: DevScalar[i32], b: B)` with `B` in `dims` as `b_dim` says: no tensor the kernel reads has
/// `B`, which the `Out` tensor given for `mask` determines.
fn key_mask(shape: Value, b_dim: Value) -> Kernel {
    Kernel::new(
        "triton",
        "key_mask",
        103,
        vec![
            pointer(0, "mask", "bf16", "out", shape, None),
            pointer(1, "n_valid", "i32", "in", json!([]), None),
            scalar(2, "b", "i32", "B", json!({})),
        ],
        vec![alignment(0, "mask", 16), alignment(1, "n_valid", 16)],
        json!({"B": b_dim}),
        "(cdiv(B, 128),)",
    )
}

#[test]
fn a_dimension_only_an_out_tensor_determines_is_bound_from_it() {
    let manifest = read(&[key_mask(json!(["B"]), json!({"buckets": [8, 24]}))], &[]).unwrap();
    let plan = plan_of(&manifest, "key_mask");
    let n = [1usize];
    for b in [8usize, 24] {
        let shape = [b];
        let p = plan
            .prepare(&[dense(Y, &shape, DType::Bf16, Access::InOut), dense(X, &n, DType::I32, Access::In)], None)
            .unwrap();
        assert_eq!(p.params(), [Y, X, b as u64, 0, 0], "the pointers, B from the Out tensor, the scratch slots");
        assert_eq!(p.grid, [1, 1, 1]);
    }
    // bound from the Out tensor, B is held to its buckets as any dimension is
    let shape = [16usize];
    let e = plan
        .prepare(&[dense(Y, &shape, DType::Bf16, Access::InOut), dense(X, &n, DType::I32, Access::In)], None)
        .unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Bucket, "{e}");
    assert!(e.to_string().contains("dimension B is 16"), "{e}");
    // the Out tensor is checked on its own before it binds anything: its dtype, its rank, that it is not empty
    let (shape, two) = ([24usize], [24usize, 1]);
    let e = plan
        .prepare(&[dense(Y, &shape, DType::F32, Access::InOut), dense(X, &n, DType::I32, Access::In)], None)
        .unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Dtype, "{e}");
    let e = plan
        .prepare(&[dense(Y, &two, DType::Bf16, Access::InOut), dense(X, &n, DType::I32, Access::In)], None)
        .unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Rank, "{e}");
    let empty = [0usize];
    let e = plan
        .prepare(&[dense(Y, &empty, DType::Bf16, Access::InOut), dense(X, &n, DType::I32, Access::In)], None)
        .unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Shape, "{e}");
    // and the device scalar is still one element: an input is checked before any Out tensor
    let (shape, two_n) = ([24usize], [2usize]);
    let e = plan
        .prepare(&[dense(Y, &shape, DType::F32, Access::InOut), dense(X, &two_n, DType::I32, Access::In)], None)
        .unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Shape, "{e}");
}

#[test]
fn an_out_dimension_that_is_an_expression_is_solved_from_the_out_tensor() {
    let manifest = read(&[key_mask(json!(["2*B"]), json!({}))], &[]).unwrap();
    let plan = plan_of(&manifest, "key_mask");
    let (n, six, seven) = ([1usize], [6usize], [7usize]);
    let p = plan
        .prepare(&[dense(Y, &six, DType::Bf16, Access::InOut), dense(X, &n, DType::I32, Access::In)], None)
        .unwrap();
    assert_eq!(p.params(), [Y, X, 3, 0, 0]);
    let e = plan
        .prepare(&[dense(Y, &seven, DType::Bf16, Access::InOut), dense(X, &n, DType::I32, Access::In)], None)
        .unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Shape, "{e}");
    assert!(e.to_string().contains("would need B = 7/2"), "{e}");
}

// -- a device scalar -------------------------------------------------------------------------------------------------------------

/// `masked_rows(x: Tensor[R, 128, f32], y: Out[Tensor[R, 128, f32]], n_tok: DevScalar[i32])`: the last pointer has the shape `[]`.
fn masked_rows() -> Kernel {
    Kernel::new(
        "triton",
        "masked_rows",
        103,
        vec![
            pointer(0, "x", "f32", "in", json!(["R", 128]), None),
            pointer(1, "y", "f32", "out", json!(["R", 128]), None),
            pointer(2, "n_tok", "i32", "in", json!([]), None),
        ],
        vec![
            alignment(0, "x", 16),
            alignment(1, "y", 16),
            alignment(2, "n_tok", 16),
            json!({"kind": "max_elements", "arg": "x", "index": 0, "shape": ["R", "128"], "value": 2147483647}),
            json!({"kind": "max_elements", "arg": "y", "index": 1, "shape": ["R", "128"], "value": 2147483647}),
            json!({"kind": "static_size", "arg": "x", "index": 0, "axis": 1, "value": 128}),
            json!({"kind": "static_size", "arg": "y", "index": 1, "axis": 1, "value": 128}),
        ],
        json!({"R": {}}),
        "(R,)",
    )
}

#[test]
fn a_device_scalar_is_one_element_of_device_memory_and_its_address_is_passed() {
    let manifest = read(&[masked_rows()], &[]).unwrap();
    let entry = manifest.entry("masked_rows", "sm_103").unwrap();
    assert!(entry.params[2].is_device_scalar() && !entry.params[0].is_device_scalar());
    let plan = plan_of(&manifest, "masked_rows");
    let (x_shape, n_shape) = ([64usize, 128], []);
    fn run<'a>(plan: &Plan, x_shape: &'a [usize], n_tok: TensorArg<'a>) -> tilefish::Result<tilefish::Prepared> {
        plan.prepare(
            &[dense(X, x_shape, DType::F32, Access::In), dense(Y, x_shape, DType::F32, Access::InOut), n_tok],
            None,
        )
    }
    macro_rules! call {
        ($t:expr) => {
            run(&plan, &x_shape, $t)
        };
    }
    let n = 0x7200_0000_0000;
    // a tensor of shape [] and one of shape [1] are the same scalar
    let p = call!(dense(n, &n_shape, DType::I32, Access::In)).unwrap();
    assert_eq!((p.grid, p.params()), ([64, 1, 1], &[X, Y, n, 0, 0][..]));
    assert!(call!(dense(n, &[1], DType::I32, Access::In)).is_ok());
    // not two elements, not a matrix of one, not another dtype, not a misaligned address
    for (what, shape, dtype, addr, want) in [
        ("two elements", &[2usize][..], DType::I32, n, PreconditionKind::Shape),
        ("a 1 x 1 matrix", &[1usize, 1][..], DType::I32, n, PreconditionKind::Shape),
        ("an i64", &[][..], DType::I64, n, PreconditionKind::Dtype),
        ("a misaligned address", &[][..], DType::I32, n + 4, PreconditionKind::PointerAlignment),
    ] {
        let e = call!(dense(addr, shape, dtype, Access::In)).unwrap_err();
        assert_eq!(kind(&e), want, "{what}: {e}");
    }
    // the device scalar of a kernel that writes it would be a kernel that changes what the host put there
    let mut k = masked_rows();
    k.entry["abi"]["params"][2]["access"] = "out".into();
    assert!(refusal(&[k], &[]).contains("device scalar n_tok must be read only"));
    // a device scalar has neither strides nor a size limit
    let mut k = masked_rows();
    k.entry["abi"]["params"][2]["strides"] = json!([]);
    k.entry["abi"]["params"][2]["offset"] = "n_tok_offset".into();
    assert!(refusal(&[k], &[]).contains("strides"));
    let mut k = masked_rows();
    k.entry["preconditions"]
        .as_array_mut()
        .unwrap()
        .push(json!({"kind": "max_elements", "arg": "n_tok", "index": 2, "shape": [], "value": 2147483647}));
    assert!(refusal(&[k], &[]).contains("max_elements"));
}

// -- a prebuilt binary -----------------------------------------------------------------------------------------------------------

/// `axpy(x: Tensor[n, f32], y: Tensor[n, f32], out: Out[Tensor[n, f32]], n_elem: n, alpha: const[f32] = 2.0, zero: const[i64] = 0)`, a PTX
/// extern: five parameters and a fixed `f32`, and no scratch slots after them.
fn axpy_ptx() -> Kernel {
    Kernel::new(
        "binary",
        "axpy_ptx",
        103,
        vec![
            pointer(0, "x", "f32", "in", json!(["n"]), None),
            pointer(1, "y", "f32", "in", json!(["n"]), None),
            pointer(2, "out", "f32", "out", json!(["n"]), None),
            scalar(3, "n_elem", "i32", "n", json!({})),
            fixed(4, "alpha", "f32", json!(2.0)),
            fixed(5, "zero", "i64", json!(0)),
        ],
        vec![
            alignment(0, "x", 16),
            alignment(1, "y", 16),
            alignment(2, "out", 16),
            json!({"kind": "divisible", "arg": "n_elem", "index": 3, "dim": "n", "by": 16}),
            json!({"kind": "max_elements", "arg": "x", "index": 0, "shape": ["n"], "value": 2147483647}),
            json!({"kind": "max_elements", "arg": "y", "index": 1, "shape": ["n"], "value": 2147483647}),
            json!({"kind": "max_elements", "arg": "out", "index": 2, "shape": ["n"], "value": 2147483647}),
        ],
        json!({"n": {"divisible_by": 16}}),
        "(cdiv(n, 128),)",
    )
}

#[test]
fn a_prebuilt_binary_is_launched_with_its_parameters_and_nothing_after_them() {
    let manifest = read(&[axpy_ptx()], &[]).unwrap();
    let entry = manifest.entry("axpy_ptx", "sm_103").unwrap();
    assert_eq!((entry.kind, entry.scratch_slots, entry.block), (tilefish::EntryKind::Binary, 0, [64, 1, 1]));
    assert_eq!((entry.num_warps, entry.num_stages, entry.triton.as_deref()), (None, None, None));
    assert_eq!(entry.ptxas.as_deref(), Some("ptxas 12.9"));
    assert!(
        matches!(&entry.binary, Some(tilefish::BinarySource::Ptx { ptxas_flags }) if ptxas_flags[0] == "-lineinfo")
    );
    assert!(matches!(&entry.params[4].kind, ParamKind::Fixed { value: tilefish::FixedValue::F32(v) } if *v == 2.0));
    let plan = plan_of(&manifest, "axpy_ptx");
    let shape = [1040usize];
    let tensors = |a, b, c| {
        vec![
            dense(a, &shape, DType::F32, Access::In),
            dense(b, &shape, DType::F32, Access::In),
            dense(c, &shape, DType::F32, Access::InOut),
        ]
    };
    let p = plan.prepare(&tensors(X, X + 0x10_0000, Y), None).unwrap();
    // the three pointers, `n_elem`, the bits of 2.0f32 in the low four bytes and the i64 zero: six slots and no scratch
    assert_eq!(p.params(), [X, X + 0x10_0000, Y, 1040, u64::from(2.0f32.to_bits()), 0]);
    assert_eq!(p.grid, [9, 1, 1]);
    assert_eq!(p.params().len(), 6);
    // the same checks as for any binary
    let e = plan.prepare(&tensors(X, X + 0x10_0000, Y + 8), None).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::PointerAlignment, "{e}");
}

#[test]
fn the_values_of_fixed_arguments_are_passed_as_their_type_says() {
    let cases: [(&str, Value, u64); 8] = [
        ("f32", json!(2.0), u64::from(2.0f32.to_bits())),
        ("f32", json!(1e-6), u64::from((1e-6f64 as f32).to_bits())),
        ("f32", json!(-0.0), u64::from((-0.0f32).to_bits())),
        ("f32", json!({"float": "inf"}), u64::from(f32::INFINITY.to_bits())),
        ("i32", json!(-1), 0xFFFF_FFFF),
        ("i32", json!(2147483647), 0x7FFF_FFFF),
        ("i64", json!(-1), u64::MAX),
        ("i64", json!(1u64 << 40), 1 << 40),
    ];
    for (ty, value, bits) in cases {
        let mut k = axpy_ptx();
        k.entry["abi"]["params"][4] = fixed(4, "alpha", ty, value.clone());
        let manifest = read(&[k], &[]).unwrap();
        let shape = [16usize];
        let tensors = [
            dense(X, &shape, DType::F32, Access::In),
            dense(X + 0x1000, &shape, DType::F32, Access::In),
            dense(Y, &shape, DType::F32, Access::InOut),
        ];
        let p = plan_of(&manifest, "axpy_ptx").prepare(&tensors, None).unwrap();
        assert_eq!(p.params()[4], bits, "{ty} {value}");
    }
}

#[test]
fn a_fixed_argument_that_is_not_what_it_says_is_refused() {
    let cases: [(&str, &str, Value, &str); 9] = [
        ("an f32 that is an integer", "f32", json!(2), "must be a float"),
        ("an i32 that is a float", "i32", json!(2.0), "must be an int"),
        ("an i32 that is a bool", "i32", json!(true), "must be an int"),
        ("an i32 that does not fit", "i32", json!(2147483648u64), "does not fit i32"),
        ("an i64 that does not fit", "i64", json!(9223372036854775808u64), "does not fit"),
        ("an f32 that is a string", "f32", json!("2.0"), "must be a float"),
        ("a tagged float that is not one", "f32", json!({"float": "big"}), "not a valid tagged float"),
        ("a type that is not a scalar type", "u8", json!(1), "one of i32, i64, f32"),
        ("an i64 that is null", "i64", Value::Null, "has no value"),
    ];
    for (what, ty, value, why) in cases {
        let mut k = axpy_ptx();
        k.entry["abi"]["params"][4] = fixed(4, "alpha", ty, value);
        let message = refusal(&[k], &[]);
        assert!(message.contains(why), "{what}: {message}");
    }
    // a fixed argument has no dim and no shape
    let mut k = axpy_ptx();
    k.entry["abi"]["params"][4]["dim"] = "n".into();
    assert!(refusal(&[k], &[]).contains("fixed argument"));
    // and belongs to a prebuilt binary, not to a Triton kernel
    let mut k = masked_rows();
    k.entry["abi"]["params"][2] = fixed(2, "alpha", "f32", json!(1.0));
    assert!(refusal(&[k], &[]).contains("belongs to a prebuilt binary"));
}
