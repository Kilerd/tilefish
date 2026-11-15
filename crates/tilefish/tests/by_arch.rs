//! An operator with an implementation per architecture whose implementations have ABIs of their own (`docs/manifest-schema-3.md`, section 4.1),
//! read and checked with no GPU on synthetic manifests (`common::synthetic`).
//!
//! `copy2d(x: Strided[Tensor[R, C, f16]], y: Out[Tensor[R, C, f16]])` is one signature with three implementations, as Python writes their entries
//! (`python/tests/test_per_arch_impls.py` declares the same): on sm_103 a Triton kernel named `tma` that takes both tensors as TMA descriptors, on
//! sm_100 a prebuilt binary named `ptx` that takes pointers and a 64-bit stride, and as the fallback (sm_120 here) a Triton kernel that takes
//! pointers, the column count and an `i32` stride. The crate reads each entry with its own ABI and launches each from the same tensors: a host binds
//! the tensors by name, in the order every implementation shares, and the entry of its device's architecture says the rest.

mod common;

use common::synthetic::{
    Kernel, alignment, json, max_extent, pointer, read, refusal, scalar, stride_scalar, tensordesc,
};
use serde_json::Value;
use tilefish::{Access, Arch, DType, DevicePtr, Error, Manifest, PreconditionKind, TensorArg};

const X: u64 = 0x7000_0000_0400; // 16-byte aligned
const Y: u64 = 0x7100_0000_0000;

fn max_elements(index: usize, arg: &str) -> Value {
    json!({"kind": "max_elements", "arg": arg, "index": index, "shape": ["R", "C"], "value": 2147483647})
}

/// The TMA form (Triton, sm_103, `"implementation": "tma"`): `x` a descriptor of a view (its stride dimension `x_s0` is its own, passed in its
/// slots), `y` a descriptor of a dense tensor, a 64x64 tile per program.
fn copy_tma(sm: u32, implementation: &str) -> Kernel {
    let mut k = Kernel::new(
        "triton",
        "copy2d",
        sm,
        vec![
            tensordesc(
                0,
                0,
                "x",
                "f16",
                "in",
                json!(["R", "C"]),
                json!([64, 64]),
                json!([64, 64]),
                128,
                Some((json!(["x_s0", 1]), "x_offset")),
            ),
            tensordesc(5, 1, "y", "f16", "out", json!(["R", "C"]), json!([64, 64]), json!([64, 64]), 128, None),
        ],
        vec![alignment(0, "x", 16), alignment(1, "y", 16)],
        json!({"R": {}, "C": {}, "x_s0": {}, "x_offset": {"divisible_by": 8}}),
        "(cdiv(R, 64), cdiv(C, 64))",
    );
    k.entry["implementation"] = implementation.into();
    k
}

/// The pointer form (Triton, the fallback): `x` a strided pointer whose row stride is the scalar `s_row`, the column count `n_cols`, a row per program.
fn copy_rows(sm: u32, implementation: &str) -> Kernel {
    let mut k = Kernel::new(
        "triton",
        "copy2d",
        sm,
        vec![
            pointer(0, "x", "f16", "in", json!(["R", "C"]), Some((json!(["s_row", 1]), "x_offset"))),
            pointer(1, "y", "f16", "out", json!(["R", "C"]), None),
            scalar(2, "n_cols", "i32", "C", json!({})),
            stride_scalar(3, "s_row", "i32", "x", 0),
        ],
        vec![alignment(0, "x", 2), alignment(1, "y", 16), max_extent(0, "x"), max_elements(1, "y")],
        json!({"R": {}, "C": {}, "s_row": {}, "x_offset": {}}),
        "(R,)",
    );
    k.entry["implementation"] = implementation.into();
    k
}

/// The prebuilt binary (PTX, sm_100, `"implementation": "ptx"`): pointers, the column count, the row stride as an `i64`; no scratch slots.
fn copy_ptx(sm: u32, implementation: &str) -> Kernel {
    let mut k = Kernel::new(
        "binary",
        "copy2d",
        sm,
        vec![
            pointer(0, "x", "f16", "in", json!(["R", "C"]), Some((json!(["x_s0", 1]), "x_offset"))),
            pointer(1, "y", "f16", "out", json!(["R", "C"]), None),
            scalar(2, "n_cols", "i32", "C", json!({})),
            stride_scalar(3, "x_s0", "i64", "x", 0),
        ],
        vec![alignment(0, "x", 2), alignment(1, "y", 16), max_extent(0, "x"), max_elements(1, "y")],
        json!({"R": {}, "C": {}, "x_s0": {}, "x_offset": {}}),
        "(R,)",
    );
    k.entry["implementation"] = implementation.into();
    k
}

fn manifest() -> Manifest {
    read(&[copy_tma(103, "tma"), copy_ptx(100, "ptx"), copy_rows(120, "fallback")], &[]).expect("the manifest reads")
}

fn region(addr: u64, bytes: u64) -> DevicePtr<'static> {
    // SAFETY: the address is never dereferenced: `prepare` only looks at the number.
    unsafe { DevicePtr::from_raw(addr, bytes.max(1), 0) }
}

fn view<'a>(addr: u64, shape: &'a [usize], strides: &'a [usize]) -> TensorArg<'a> {
    let reach = 1 + shape.iter().zip(strides).map(|(n, s)| (n - 1) * s).sum::<usize>();
    TensorArg::new_strided(region(addr, (reach * 2) as u64), shape, strides, 0, DType::F16, Access::In).unwrap()
}

fn dense(addr: u64, shape: &[usize]) -> TensorArg<'_> {
    TensorArg::new(region(addr, (shape.iter().product::<usize>() * 2) as u64), shape, DType::F16, Access::Out).unwrap()
}

fn kind(e: &Error) -> PreconditionKind {
    e.precondition_kind().unwrap_or_else(|| panic!("not a precondition error: {e:?}"))
}

#[test]
fn the_entries_of_one_stem_may_have_an_abi_per_architecture() {
    let m = manifest();
    let names = |arch: &str| -> Vec<String> {
        m.entry("copy2d", arch).unwrap().params.iter().map(|p| p.name.clone()).collect()
    };
    assert_eq!(names("sm_103"), ["x", "y"]);
    assert_eq!(names("sm_100"), ["x", "y", "n_cols", "x_s0"]);
    assert_eq!(names("sm_120"), ["x", "y", "n_cols", "s_row"]);
    let implementations: Vec<Option<&str>> = ["sm_103", "sm_100", "sm_120"]
        .iter()
        .map(|a| m.entry("copy2d", a).unwrap().implementation.as_deref())
        .collect();
    assert_eq!(implementations, [Some("tma"), Some("ptx"), Some("fallback")]);
}

#[test]
fn each_architecture_launches_its_implementation_with_its_own_abi_from_the_same_tensors() {
    let m = manifest();
    let (shape, strides) = ([96usize, 192], [256usize, 1]);
    let launch = |arch: u32| {
        let plan = m.plan("copy2d", &[], Arch::sm(arch)).unwrap();
        let p = plan.prepare(&[view(X, &shape, &strides), dense(Y, &shape)], None).unwrap();
        (p.grid, p.params().to_vec())
    };
    // the descriptors: the map's slot (its bytes are encoded at the launch), then the shape and the strides of each tensor; the scratch slots
    assert_eq!(launch(103), ([2, 3, 1], vec![0, 96, 192, 256, 1, 0, 96, 192, 192, 1, 0, 0]));
    // the prebuilt binary: its parameters and nothing more
    assert_eq!(launch(100), ([96, 1, 1], vec![X, Y, 192, 256]));
    // the pointer form: the addresses, the scalars, the scratch slots
    assert_eq!(launch(120), ([96, 1, 1], vec![X, Y, 192, 256, 0, 0]));
}

#[test]
fn a_precondition_of_one_implementation_is_checked_on_its_architecture() {
    let m = manifest();
    // rows of 193 elements (386 bytes): a pointer takes the view, a TMA descriptor cannot be made of it
    let (shape, strides) = ([96usize, 192], [193usize, 1]);
    let args = [view(X, &shape, &strides), dense(Y, &shape)];
    let e = m.plan("copy2d", &[], Arch::sm(103)).unwrap().prepare(&args, None).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::TensorMap, "{e}");
    let p = m.plan("copy2d", &[], Arch::sm(120)).unwrap().prepare(&args, None).unwrap();
    assert_eq!(p.params(), [X, Y, 192, 193, 0, 0]);
    // a view that starts 2 bytes into its memory: a descriptor's address is 16-byte aligned, the pointer form's view the element's
    let shifted = [view(X + 2, &shape, &[256, 1]), dense(Y, &shape)];
    let e = m.plan("copy2d", &[], Arch::sm(103)).unwrap().prepare(&shifted, None).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::PointerAlignment, "{e}");
    assert!(m.plan("copy2d", &[], Arch::sm(100)).unwrap().prepare(&shifted, None).is_ok());
}

#[test]
fn a_named_implementation_may_be_the_entry_of_several_architectures() {
    let m = read(&[copy_tma(100, "tma"), copy_tma(103, "tma"), copy_rows(120, "fallback")], &[]).unwrap();
    for arch in ["sm_100", "sm_103"] {
        assert_eq!(m.entry("copy2d", arch).unwrap().implementation.as_deref(), Some("tma"));
    }
    // the reader holds a name to its form only: which architectures map to it is the declaration's (Python's replay compares the two)
    let e = refusal(&[copy_tma(103, "sm_100")], &[]);
    assert!(
        e.contains(
            "implementation must be the entry's architecture \"sm_103\", \"fallback\" or the name of an implementation"
        ),
        "{e}"
    );
}
