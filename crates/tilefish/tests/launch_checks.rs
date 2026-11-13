//! Every precondition of a binary, checked before a launch from the manifest alone, with no GPU.
//!
//! `Plan::prepare` is what `Kernel::launch` runs before it touches the driver: it binds the named dimensions from the sizes of the
//! tensors (each pointer's declared shape is in the manifest, `["Rows", 1024]`), checks that they agree across arguments and with
//! the static sizes, the exact bucket and `divisible_by` of every dimension, the dtype and rank of every tensor, `max_elements`,
//! aliasing, 16-byte alignment and the 32-bit range of the runtime scalars (derived from the dimensions, or from the element count
//! of the tensor they count), and evaluates the grid and checks it against the device's limits. These tests run it on the plans of the
//! `sm_103` fixture, with a device that they make up (`DeviceInfo`) and made-up addresses: nothing is launched. A launch has nothing
//! but its tensors.

mod common;

use common::{Copy, fixtures};
use tilefish::{Access, Arch, DType, DeviceInfo, DevicePtr, Error, Manifest, Pin, Plan, PreconditionKind, TensorArg};

#[allow(dead_code)]
#[rustfmt::skip] // generated: regenerate it, do not format it
#[path = "fixtures/generated.rs"]
mod generated;

fn manifest() -> Manifest {
    Manifest::load(fixtures().join("sm_103"), Pin::Digest(&common::expected_digest("sm_103"))).unwrap()
}

fn plan(stem: &str, buckets: &[(&str, u64)]) -> Plan {
    manifest().plan(stem, buckets, Arch::sm(103)).unwrap()
}

/// The plan of a generated launcher: its stem and its buckets.
macro_rules! typed {
    ($ty:ident) => {
        plan(generated::$ty::STEM, generated::$ty::BUCKETS)
    };
}

fn t<'a>(addr: u64, shape: &'a [usize], dtype: DType) -> TensorArg<'a> {
    // SAFETY: these addresses are never dereferenced: `prepare` only looks at the number. The region is as long as the tensor.
    let ptr = unsafe { DevicePtr::from_raw(addr, shape.iter().product::<usize>() as u64 * dtype.size() as u64, 0) };
    TensorArg::new(ptr, shape, dtype, Access::InOut).unwrap()
}

const A: u64 = 0x7000_0000_0000;
const B: u64 = 0x7000_1000_0000;
const C: u64 = 0x7000_2000_0000;
const D: u64 = 0x7000_3000_0000;
// two addresses that are far enough apart for tensors of gigabytes
const BIG_A: u64 = 0x1000_0000_0000;
const BIG_B: u64 = 0x2000_0000_0000;
// two addresses for tensors of up to 2^63 bytes: a tensor at LOW ends before HIGH
const LOW: u64 = 1 << 20;
const HIGH: u64 = 1 << 63;

/// A device like the B300's: 2^31 - 1 blocks along x, 65535 along y and z.
const DEVICE: DeviceInfo = DeviceInfo { ordinal: 0, context: None, max_grid: [(1 << 31) - 1, 65535, 65535] };

fn kind(e: &Error) -> PreconditionKind {
    e.precondition_kind().unwrap_or_else(|| panic!("not a precondition error: {e:?}"))
}

fn refused(plan: &Plan, tensors: &[TensorArg<'_>]) -> Error {
    plan.prepare(tensors, Some(&DEVICE)).expect_err("the call was accepted")
}

#[test]
fn every_launcher_is_typed_with_shapes_that_agree_with_its_manifest() {
    macro_rules! all {
        ($($ty:ident),*) => {$(
            // what the launcher says of its tensors is what the manifest says: the declared shape of each pointer
            let p = typed!($ty);
            let loaded = manifest();
            let entry = loaded.entry(generated::$ty::STEM, "sm_103").unwrap();
            let shapes: Vec<Vec<String>> = entry
                .params
                .iter()
                .filter_map(|p| match &p.kind {
                    tilefish::ParamKind::Pointer { shape, .. } => Some(shape.iter().map(|d| d.to_string()).collect()),
                    _ => None,
                })
                .collect();
            let declared: Vec<Vec<String>> = generated::$ty::SHAPES.iter().map(|s| s.iter().map(|d| d.to_string()).collect()).collect();
            assert_eq!(shapes, declared, "{}", generated::$ty::STEM);
            assert_eq!(p.tensor_names().count(), declared.len());
        )*};
    }
    all!(
        Axpy,
        BucketedRmsDb1000,
        BucketedRmsDb512,
        BucketedRmsDb96,
        GeluErf,
        GeluExact,
        Layernorm,
        Mm,
        ResidualAddRmsnorm,
        Rmsnorm,
        Softmax,
        Swiglu
    );
    assert_eq!(generated::MANIFEST_DIGEST, common::expected_digest("sm_103"));
}

#[test]
fn a_gelu_is_checked_and_its_grid_and_scalar_come_from_the_shapes() {
    let plan = typed!(GeluErf);
    for (n, blocks) in [(16usize, 1u32), (1024, 1), (1040, 2), (4096, 4), (1 << 20, 1024)] {
        let shape = [n];
        let prepared = plan.prepare(&[t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16)], Some(&DEVICE)).unwrap();
        assert_eq!(prepared.grid, [blocks, 1, 1], "n = {n}");
        // kernelParams: the two pointers, the i32 in the low bytes of its slot, then the two null scratch pointers
        assert_eq!(prepared.params(), [A, B, n as u64, 0, 0]);
    }
}

#[test]
fn a_misaligned_pointer_is_refused() {
    let plan = typed!(GeluErf);
    let shape = [4096usize];
    let e = refused(&plan, &[t(A + 8, &shape, DType::Bf16), t(B, &shape, DType::Bf16)]);
    assert_eq!(kind(&e), PreconditionKind::PointerAlignment);
    let e = refused(&plan, &[t(A, &shape, DType::Bf16), t(B + 4, &shape, DType::Bf16)]);
    assert!(e.to_string().contains("not aligned to 16 bytes"), "{e}");
}

#[test]
fn the_dtype_the_rank_and_an_empty_tensor_are_checked() {
    let plan = typed!(GeluErf);
    let shape = [4096usize];
    let e = refused(&plan, &[t(A, &shape, DType::F32), t(B, &shape, DType::Bf16)]);
    assert_eq!(kind(&e), PreconditionKind::Dtype);
    let e = refused(&plan, &[t(A, &[64, 64], DType::Bf16), t(B, &shape, DType::Bf16)]);
    assert_eq!(kind(&e), PreconditionKind::Rank);
    let e = refused(&plan, &[t(A, &[0], DType::Bf16), t(B, &[0], DType::Bf16)]);
    assert_eq!(kind(&e), PreconditionKind::Shape);
    let e = plan.prepare(&[t(A, &shape, DType::Bf16)], Some(&DEVICE)).unwrap_err();
    assert!(matches!(e, Error::Launch(m) if m.contains("takes 2 tensors")));
}

#[test]
fn a_dimension_that_is_not_a_multiple_of_what_the_manifest_says_is_refused() {
    // gelu_erf: `Count` is declared divisible_by=16, and the runtime scalar that carries it has the precondition too
    let plan = typed!(GeluErf);
    let shape = [60usize];
    let e = refused(&plan, &[t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16)]);
    assert_eq!(kind(&e), PreconditionKind::Divisible);
    assert!(e.to_string().contains("Count") && e.to_string().contains("divisible_by=16"), "{e}");
    // mm: MmM, MmN are multiples of 128 and MmK of 64, and nothing but the manifest says so (they are named by `dims`)
    let plan = typed!(Mm);
    let ok = plan
        .prepare(
            &[t(A, &[128, 64], DType::F16), t(B, &[64, 256], DType::F16), t(C, &[128, 256], DType::F32)],
            Some(&DEVICE),
        )
        .unwrap();
    assert_eq!(ok.grid, [1, 2, 1]); // a 2-D grid: (cdiv(MmM, 128), cdiv(MmN, 128))
    assert_eq!(ok.params(), [A, B, C, 128, 256, 64, 0, 0]);
    for (m, n, k, dim) in [(100usize, 128usize, 64usize, "MmM"), (128, 130, 64, "MmN"), (128, 128, 60, "MmK")] {
        let e = refused(&plan, &[t(A, &[m, k], DType::F16), t(B, &[k, n], DType::F16), t(C, &[m, n], DType::F32)]);
        assert_eq!(kind(&e), PreconditionKind::Divisible, "{dim}");
        assert!(e.to_string().contains(dim), "{e}");
    }
}

#[test]
fn tensors_that_disagree_about_a_dimension_are_refused() {
    let plan = typed!(GeluErf);
    let e = refused(&plan, &[t(A, &[4096], DType::Bf16), t(B, &[2048], DType::Bf16)]);
    assert_eq!(kind(&e), PreconditionKind::Shape);
    let plan = typed!(Rmsnorm);
    let e = refused(&plan, &[t(A, &[3, 2048], DType::Bf16), t(B, &[2048], DType::Bf16), t(C, &[4, 2048], DType::Bf16)]);
    assert_eq!(kind(&e), PreconditionKind::Shape);
    assert!(e.to_string().contains("Rows = 3"), "{e}");
    // mm: the K of `a` is 64, the K of `b` is 128
    let e = refused(
        &typed!(Mm),
        &[t(A, &[128, 64], DType::F16), t(B, &[128, 128], DType::F16), t(C, &[128, 128], DType::F32)],
    );
    assert_eq!(kind(&e), PreconditionKind::Shape);
    assert!(e.to_string().contains("MmK"), "{e}");
}

#[test]
fn a_size_baked_into_the_binary_is_checked() {
    let plan = typed!(Rmsnorm);
    let ok = plan
        .prepare(
            &[t(A, &[3, 2048], DType::Bf16), t(B, &[2048], DType::Bf16), t(C, &[3, 2048], DType::Bf16)],
            Some(&DEVICE),
        )
        .unwrap();
    assert_eq!(ok.grid, [3, 1, 1]); // one program per row: the grid is (Rows,), and Rows is bound by the shapes
    let bad = [t(A, &[3, 2000], DType::Bf16), t(B, &[2000], DType::Bf16), t(C, &[3, 2000], DType::Bf16)];
    let e = refused(&plan, &bad);
    assert_eq!(kind(&e), PreconditionKind::StaticSize);
    assert!(e.to_string().contains("requires 2048"), "{e}");
}

#[test]
fn a_bucket_is_exact_there_is_no_padding() {
    let plan = typed!(BucketedRmsDb96);
    let ok = plan
        .prepare(&[t(A, &[3, 96], DType::Bf16), t(B, &[96], DType::Bf16), t(C, &[3, 96], DType::Bf16)], Some(&DEVICE))
        .unwrap();
    assert_eq!(ok.grid, [3, 1, 1]);
    for db in [95usize, 97, 100, 512] {
        let e = refused(&plan, &[t(A, &[3, db], DType::Bf16), t(B, &[db], DType::Bf16), t(C, &[3, db], DType::Bf16)]);
        assert_eq!(kind(&e), PreconditionKind::Bucket, "Db = {db}");
        let text = e.to_string();
        assert!(
            text.contains(&format!("dimension Db is {db}"))
                && text.contains("exactly 96")
                && text.contains("no padding"),
            "{e}"
        );
    }
    // the 512 binary takes 512
    let plan = typed!(BucketedRmsDb512);
    let ok = [t(A, &[2, 512], DType::Bf16), t(B, &[512], DType::Bf16), t(C, &[2, 512], DType::Bf16)];
    assert!(plan.prepare(&ok, Some(&DEVICE)).is_ok());
}

#[test]
fn a_tensor_with_more_elements_than_the_i32_offsets_of_a_generated_kernel_reach_is_refused() {
    let plan = typed!(Softmax);
    let ok = [t(BIG_A, &[2_000_000, 1000], DType::F32), t(BIG_B, &[2_000_000, 1000], DType::F32)];
    assert_eq!(plan.prepare(&ok, Some(&DEVICE)).unwrap().grid, [2_000_000, 1, 1]);
    // 2_200_000 * 1000 > 2^31 - 1
    let big = [t(BIG_A, &[2_200_000, 1000], DType::F32), t(BIG_B, &[2_200_000, 1000], DType::F32)];
    let e = refused(&plan, &big);
    assert_eq!(kind(&e), PreconditionKind::MaxElements);
    assert!(e.to_string().contains("2200000000 elements") && e.to_string().contains("2147483647"), "{e}");
}

#[test]
fn a_scalar_that_does_not_fit_32_bits_is_refused() {
    let plan = typed!(GeluErf);
    let shape = [1usize << 31];
    let e = refused(&plan, &[t(BIG_A, &shape, DType::Bf16), t(BIG_B, &shape, DType::Bf16)]);
    assert_eq!(kind(&e), PreconditionKind::Shape);
    assert!(e.to_string().contains("32-bit signed integer"), "{e}");
    let shape = [(1usize << 31) - 16];
    assert!(plan.prepare(&[t(BIG_A, &shape, DType::Bf16), t(BIG_B, &shape, DType::Bf16)], Some(&DEVICE)).is_ok());
}

#[test]
fn what_is_written_must_not_overlap_anything() {
    let plan = typed!(Axpy);
    let shape = [4096usize];
    // y += alpha * x, in place: y is written
    assert!(plan.prepare(&[t(A, &shape, DType::F32), t(B, &shape, DType::F32)], Some(&DEVICE)).is_ok());
    let e = refused(&plan, &[t(A, &shape, DType::F32), t(A, &shape, DType::F32)]);
    assert_eq!(kind(&e), PreconditionKind::Alias);
    let e = refused(&plan, &[t(A, &shape, DType::F32), t(A + 4096 * 4 - 16, &shape, DType::F32)]);
    assert_eq!(kind(&e), PreconditionKind::Alias, "partial overlap");
    let adjacent = [t(A, &shape, DType::F32), t(A + 4096 * 4, &shape, DType::F32)];
    assert!(plan.prepare(&adjacent, Some(&DEVICE)).is_ok(), "adjacent is not overlapping");
    // two read-only arguments may alias
    let plan = typed!(Swiglu);
    let shape = [2usize, 3072];
    assert!(
        plan.prepare(&[t(A, &shape, DType::Bf16), t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16)], Some(&DEVICE))
            .is_ok()
    );
}

#[test]
fn a_swiglus_scalar_is_the_element_count_and_the_grid_follows_from_it() {
    for m in [1usize, 2, 33, 130] {
        let shape = [m, 3072];
        let args = [t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16), t(C, &shape, DType::Bf16)];
        // `n` carries 3072*M and counts the elements of `gate`: M is bound by the shapes, and the scalar is checked against the count
        let prepared = typed!(Swiglu).prepare(&args, Some(&DEVICE)).unwrap();
        assert_eq!(prepared.grid, [3 * m as u32, 1, 1]);
        assert_eq!(prepared.params(), [A, B, C, (3072 * m) as u64, 0, 0]);
    }
    // an element count that is not a multiple of the static axis cannot be a shape of this binary: the 3000 is refused as a size
    let shape = [5usize, 3000];
    let args = [t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16), t(C, &shape, DType::Bf16)];
    let e = refused(&typed!(Swiglu), &args);
    assert_eq!(kind(&e), PreconditionKind::StaticSize);
}

#[test]
fn a_kernel_is_launched_on_its_own_device() {
    let plan = typed!(GeluErf);
    let shape = [4096usize];
    let args = [t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16)];
    assert!(plan.prepare(&args, Some(&DEVICE)).is_ok());
    let e = plan.prepare(&args, Some(&DeviceInfo { ordinal: 1, ..DEVICE })).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Device);
    // with no device to check against, neither the device nor the grid is
    assert!(plan.prepare(&args, None).is_ok());
}

#[test]
fn a_residual_add_rmsnorm_updates_its_inout_tensor_in_place_and_nothing_else_may_alias_it() {
    let plan = typed!(ResidualAddRmsnorm);
    let (shape, w) = ([4usize, 2560], [2560usize]);
    let ok = |a: u64, b: u64, c: u64, d: u64| {
        plan.prepare(
            &[t(a, &shape, DType::Bf16), t(b, &shape, DType::F32), t(c, &w, DType::Bf16), t(d, &shape, DType::Bf16)],
            Some(&DEVICE),
        )
    };
    assert!(ok(A, B, C, D).is_ok());
    assert_eq!(kind(&ok(A, A, C, D).unwrap_err()), PreconditionKind::Alias); // the residual (written) over x
    assert_eq!(kind(&ok(A, B, C, B + 16).unwrap_err()), PreconditionKind::Alias); // the output over the residual
    // the dtype of the residual is f32: a bf16 residual is another binary
    let bad = [t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16), t(C, &w, DType::Bf16), t(D, &shape, DType::Bf16)];
    assert_eq!(kind(&plan.prepare(&bad, Some(&DEVICE)).unwrap_err()), PreconditionKind::Dtype);
}

// -- the grid against the device's limits ------------------------------------------------------------------------------------------------

/// A copy of the fixture whose `softmax` has the grid `grid`, read without a pin (the edit changes the digest).
fn with_grid(grid: &str) -> (Copy, Manifest) {
    let copy = Copy::of("sm_103");
    copy.edit_json(|doc| Copy::entry(doc, "softmax")["grid"] = grid.into());
    let manifest = Manifest::load(&copy.dir, Pin::TrustDirectory).expect("the grid is a valid one");
    (copy, manifest)
}

#[test]
fn a_grid_beyond_what_the_device_takes_is_refused_along_every_axis() {
    // x is `M` blocks, and y and z are static: softmax's tensors have M = 1 row here
    let shape = [1usize, 1000];
    let args = [t(A, &shape, DType::F32), t(B, &shape, DType::F32)];
    for (grid, ok) in [
        ("(M, 65535, 65535)", true),
        ("(M, 65536, 1)", false),
        ("(M, 1, 65536)", false),
        ("(M, 70000, 2)", false),
        ("(M, 2, 70000)", false),
        ("(M, 65535)", true),
        ("(M, 100000)", false),
    ] {
        let (_copy, manifest) = with_grid(grid);
        let plan = manifest.plan("softmax", &[], Arch::sm(103)).unwrap();
        let result = plan.prepare(&args, Some(&DEVICE));
        match (ok, result) {
            (true, Ok(prepared)) => assert!(prepared.grid.iter().all(|g| *g >= 1)),
            (false, Err(Error::Unsupported(message))) => {
                assert!(
                    message.contains(grid) && message.contains("this device takes 1 to 65535"),
                    "{grid}: {message}"
                );
            }
            (_, other) => panic!("{grid}: {other:?}"),
        }
        // the device's limits are what is checked: the same grid is fine for a device that takes it
        let roomy = DeviceInfo { ordinal: 0, context: None, max_grid: [i64::from(u32::MAX), 1 << 20, 1 << 20] };
        assert!(plan.prepare(&args, Some(&roomy)).is_ok(), "{grid}");
    }
    // x, from the shapes: M is the number of rows
    let (_copy, manifest) = with_grid("(M, 1, 1)");
    let plan = manifest.plan("softmax", &[], Arch::sm(103)).unwrap();
    let rows = [3_000_000_000usize / 1000, 1000];
    let args = [t(BIG_A, &rows, DType::F32), t(BIG_B, &rows, DType::F32)];
    // 3_000_000 rows: fine along x, but not for a device that takes 1_000_000 there
    assert!(plan.prepare(&args, Some(&DEVICE)).is_err()); // (it is also more than the i32 offsets reach: max_elements)
    let small = DeviceInfo { ordinal: 0, context: None, max_grid: [1_000_000, 65535, 65535] };
    let rows = [1_500_000usize, 1000];
    let args = [t(BIG_A, &rows, DType::F32), t(BIG_B, &rows, DType::F32)];
    match plan.prepare(&args, Some(&small)) {
        Err(Error::Unsupported(message)) => assert!(message.contains("1500000 blocks along x"), "{message}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_grid_that_is_not_a_positive_size_is_refused() {
    // a grid entry that is 0 or below once the dimensions are bound
    let (_copy, manifest) = with_grid("(M - 1,)");
    let plan = manifest.plan("softmax", &[], Arch::sm(103)).unwrap();
    let shape = [1usize, 1000];
    let e = plan.prepare(&[t(A, &shape, DType::F32), t(B, &shape, DType::F32)], Some(&DEVICE)).unwrap_err();
    assert!(matches!(&e, Error::Launch(m) if m.contains("must be at least 1")), "{e}");
}

// -- the runtime scalars: what they count, their divisor, their 32-bit range ----------------------------------------------------------------

/// The plan of `stem` in a copy of the fixture whose entry `change` edited (read without a pin: the edit changes the digest).
fn edited(stem: &str, change: impl FnOnce(&mut serde_json::Value)) -> (Copy, Plan) {
    let copy = Copy::of("sm_103");
    copy.edit_json(|doc| change(Copy::entry(doc, stem)));
    let manifest = Manifest::load(&copy.dir, Pin::TrustDirectory).expect("the edit is a manifest that reads");
    let plan = manifest.plan(stem, &[], Arch::sm(103)).unwrap();
    (copy, plan)
}

fn param<'a>(entry: &'a mut serde_json::Value, name: &str) -> &'a mut serde_json::Value {
    entry["abi"]["params"].as_array_mut().unwrap().iter_mut().find(|p| p["name"] == name).expect("a parameter")
}

fn precondition<'a>(entry: &'a mut serde_json::Value, kind: &str, arg: &str) -> &'a mut serde_json::Value {
    entry["preconditions"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|p| p["kind"] == kind && p["arg"] == arg)
        .expect("a precondition")
}

#[test]
fn a_runtime_scalar_that_is_not_a_multiple_of_its_divisor_is_refused_whatever_its_dimensions_say() {
    // swiglu's `n` is 3072*M and M is not declared divisible by anything: only the scalar's own precondition can refuse. With the
    // divisor 2048 (3072 is not a multiple of it), M = 1 is refused and M = 2 is not.
    let (_copy, plan) = edited("swiglu", |e| precondition(e, "divisible", "n")["by"] = 2048.into());
    let call = |m: usize| {
        let shape = [m, 3072];
        let args = [t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16), t(C, &shape, DType::Bf16)];
        plan.prepare(&args, Some(&DEVICE)).map(|p| p.params().to_vec())
    };
    let e = call(1).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Divisible);
    let text = e.to_string();
    assert!(
        text.contains("runtime argument 'n'") && text.contains("is 3072") && text.contains("not a multiple of 2048"),
        "{text}"
    );
    assert_eq!(call(2).unwrap(), [A, B, C, 6144, 0, 0]);
    assert_eq!(kind(&call(3).unwrap_err()), PreconditionKind::Divisible);
    assert!(call(4).is_ok());
    // the unmodified entry: a multiple of 16 for every M
    let shape = [1usize, 3072];
    let args = [t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16), t(C, &shape, DType::Bf16)];
    assert!(typed!(Swiglu).prepare(&args, Some(&DEVICE)).is_ok());
}

#[test]
fn a_runtime_scalar_is_in_the_range_of_a_signed_32_bit_integer_edges_included() {
    // gelu_erf with `n_elements = Count - 4096`, then `Count + 15`: the value is 0 (accepted), -16, 2^31 - 1 (accepted) and 2^31 + 15
    // (2^31 - 1 is odd: the scalar's divisor goes, so that the range is all that is left to refuse it)
    let shaped = |dim: &str| {
        edited("gelu_erf", |e| {
            param(e, "n_elements")["dim"] = dim.into();
            e["preconditions"].as_array_mut().unwrap().retain(|p| p["kind"] != "divisible");
        })
    };
    let call = |plan: &Plan, count: usize| {
        let shape = [count];
        plan.prepare(&[t(BIG_A, &shape, DType::Bf16), t(BIG_B, &shape, DType::Bf16)], Some(&DEVICE))
    };
    let (_copy, low) = shaped("Count - 4096");
    assert_eq!(call(&low, 4096).unwrap().params(), [BIG_A, BIG_B, 0, 0, 0], "zero is in the range");
    let e = call(&low, 4080).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Shape);
    assert!(e.to_string().contains("is -16") && e.to_string().contains("32-bit signed"), "{e}");
    assert_eq!(call(&low, 4112).unwrap().params()[2], 16);
    let (_copy, high) = shaped("Count + 15");
    let top = (1usize << 31) - 16;
    assert_eq!(call(&high, top).unwrap().params()[2], i32::MAX as u64, "2^31 - 1 is in the range");
    let e = call(&high, top + 16).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Shape);
    assert!(e.to_string().contains(&format!("is {}", (1u64 << 31) + 15)), "{e}");
    let (_copy, exact) = shaped("Count");
    assert!(call(&exact, top).is_ok() && call(&exact, top + 16).is_err());
}

#[test]
fn a_scalar_counts_the_elements_of_the_tensor_it_names_and_not_of_another() {
    // mm, with `m` the element count of `c_ptr` (M*N) and `k` the element count of `b_ptr` (K*N): three tensors of three sizes. A product of
    // two symbolic sizes is not affine, so each carries a fresh dimension name (Python's `_check_count_of`), which only its count binds
    let (_copy, plan) = edited("mm", |e| {
        for (name, dim, target) in [("m", "Mc", "c_ptr"), ("k", "Kb", "b_ptr")] {
            param(e, name)["dim"] = dim.into();
            param(e, name)["count_of"] = target.into();
            precondition(e, "divisible", name)["dim"] = dim.into();
            e["dims"][dim] = serde_json::json!({});
        }
    });
    let args = [t(A, &[128, 64], DType::F16), t(B, &[64, 256], DType::F16), t(C, &[128, 256], DType::F32)];
    let prepared = plan.prepare(&args, Some(&DEVICE)).unwrap();
    // a: 128*64 = 8192, b: 64*256 = 16384, c: 128*256 = 32768
    assert_eq!(prepared.params(), [A, B, C, 32768, 256, 16384, 0, 0]);
    // a scalar whose dimension is not what its tensor holds says so, and names the tensor and the count
    let (_copy, plan) = edited("mm", |e| {
        param(e, "n")["dim"] = "MmN".into();
        param(e, "n")["count_of"] = "a_ptr".into();
    });
    let e = plan.prepare(&args, Some(&DEVICE)).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Shape);
    let text = e.to_string();
    assert!(
        text.contains("runtime argument 'n' is MmN = 256") && text.contains("the tensor 'a_ptr' has 8192 elements"),
        "{text}"
    );
}

// -- sizes at the edges of 64 bits -----------------------------------------------------------------------------------------------------

#[test]
fn shapes_of_two_to_the_sixtieth_and_element_counts_of_two_to_the_sixty_third_are_refused_and_never_wrap() {
    // gelu_erf over 2^60 bf16 elements: the count and the bytes fit, the scalar does not fit 32 bits
    let plan = typed!(GeluErf);
    let n = 1usize << 60;
    let e = refused(&plan, &[t(BIG_A, &[n], DType::Bf16), t(BIG_B, &[n], DType::Bf16)]);
    assert_eq!(kind(&e), PreconditionKind::Shape);
    assert!(e.to_string().contains("32-bit signed"), "{e}");
    // softmax over (2^59, 4): its second axis is a static 1000 in the manifest
    let plan = typed!(Softmax);
    let big = n / 2;
    let e = refused(&plan, &[t(BIG_A, &[big, 4], DType::F32), t(BIG_B, &[big, 4], DType::F32)]);
    assert_eq!(kind(&e), PreconditionKind::StaticSize);
    // (2^50, 1000) elements: more than the i32 offsets of a generated kernel reach
    let rows = 1usize << 50;
    let e = refused(&plan, &[t(LOW, &[rows, 1000], DType::F32), t(HIGH, &[rows, 1000], DType::F32)]);
    assert_eq!(kind(&e), PreconditionKind::MaxElements);

    // one-byte pointers are the only ones for which 2^63 elements fit a region: gelu_erf and softmax with u8 tensors
    let (_copy, bytes) = edited("gelu_erf", |e| {
        for name in ["x_ptr", "y_ptr"] {
            param(e, name)["dtype"] = "u8".into();
            param(e, name)["type"] = "*u8".into();
        }
    });
    let count = |count: usize| {
        bytes.prepare(&[t(BIG_A, &[count], DType::U8), t(BIG_B, &[count], DType::U8)], Some(&DEVICE)).unwrap_err()
    };
    // 2^63 - 1 elements (the largest count that fits an i64) get past the count: it is not a multiple of 16
    let e = count(i64::MAX as usize);
    assert_eq!(kind(&e), PreconditionKind::Divisible, "{e}");
    assert!(e.to_string().contains("dimension Count is 9223372036854775807"), "{e}");
    // 2^63 elements are more than 64 bits can count
    let e = count(1usize << 63);
    assert_eq!(kind(&e), PreconditionKind::Shape);
    assert!(e.to_string().contains("more elements than fit 64 bits"), "{e}");
    let e = count(usize::MAX);
    assert!(e.to_string().contains("more elements than fit 64 bits"), "{e}");
    // 2^62 elements are a count, and then too many for a 32-bit scalar
    let e = count(1usize << 62);
    assert!(e.to_string().contains("32-bit signed"), "{e}");
    // the same for two axes: 2^54 rows of 1000 bytes is 1.8e19 elements, 2^53 rows is 9.0e18
    let (_copy, bytes) = edited("softmax", |e| {
        for name in ["x", "out"] {
            param(e, name)["dtype"] = "u8".into();
            param(e, name)["type"] = "*u8".into();
        }
    });
    let rows = |rows: usize| {
        bytes
            .prepare(&[t(LOW, &[rows, 1000], DType::U8), t(HIGH, &[rows, 1000], DType::U8)], Some(&DEVICE))
            .unwrap_err()
    };
    assert!(rows(1 << 54).to_string().contains("more elements than fit 64 bits"));
    assert_eq!(kind(&rows(1 << 53)), PreconditionKind::MaxElements);
}

// -- externs index with i32 unless they declare index_bits=64 -----------------------------------------------------------------------------

#[test]
fn every_pointer_of_a_hand_written_extern_has_max_elements_and_the_host_checks_it() {
    // rmsnorm (`row * N_COLS`) and mm used to pass every check at sizes where their i32 offsets overflow
    let manifest = manifest();
    for stem in ["rmsnorm", "mm", "gelu_erf", "axpy"] {
        let entry = manifest.entry(stem, "sm_103").unwrap();
        for p in entry.params.iter().filter(|p| matches!(p.kind, tilefish::ParamKind::Pointer { .. })) {
            let limited = entry.preconditions.iter().any(
                |c| matches!(c, tilefish::Precondition::MaxElements { arg, value, .. } if arg == &p.name && *value == 2147483647),
            );
            assert!(limited, "{stem}: {} has no max_elements", p.name);
        }
    }
    // rmsnorm: x is (Rows, 2048), so (2^20 - 1) rows are 2147481600 elements, and 2^20 rows are 2^31
    let plan = typed!(Rmsnorm);
    let call = |rows: usize| {
        let (x, w) = ([rows, 2048], [2048usize]);
        plan.prepare(&[t(BIG_A, &x, DType::Bf16), t(A, &w, DType::Bf16), t(BIG_B, &x, DType::Bf16)], Some(&DEVICE))
    };
    assert_eq!(call((1 << 20) - 1).unwrap().grid, [(1 << 20) - 1, 1, 1]);
    let e = call(1 << 20).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::MaxElements);
    assert!(e.to_string().contains("'x_ptr'") || e.to_string().contains("x_ptr"), "{e}");
    assert!(e.to_string().contains("2147483648 elements") && e.to_string().contains("2147483647"), "{e}");

    // mm: c is (MmM, MmN) f32, and MmN is at least 128: 2^24 - 128 rows of 128 columns are 2147467264 elements, 2^24 rows are 2^31
    let plan = typed!(Mm);
    let call = |m: usize| {
        let (a, b, c) = ([m, 64usize], [64usize, 128], [m, 128usize]);
        plan.prepare(&[t(BIG_A, &a, DType::F16), t(A, &b, DType::F16), t(BIG_B, &c, DType::F32)], Some(&DEVICE))
    };
    let ok = call((1 << 24) - 128).unwrap();
    assert_eq!(ok.grid, [(1 << 17) - 1, 1, 1]);
    let e = call(1 << 24).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::MaxElements);
    assert!(e.to_string().contains("c_ptr") && e.to_string().contains("2147483648 elements"), "{e}");
}

#[test]
fn an_extern_that_says_it_indexes_with_i64_has_no_limit_in_its_manifest_and_the_host_launches_it() {
    // a manifest whose rmsnorm lost its max_elements (what `index_bits=64` writes): the same call is not refused for its size
    let (_copy, plan) = edited("rmsnorm", |e| {
        e["preconditions"].as_array_mut().unwrap().retain(|p| p["kind"] != "max_elements");
    });
    let rows = 1usize << 20;
    let (x, w) = ([rows, 2048], [2048usize]);
    let prepared = plan
        .prepare(&[t(BIG_A, &x, DType::Bf16), t(A, &w, DType::Bf16), t(BIG_B, &x, DType::Bf16)], Some(&DEVICE))
        .unwrap();
    assert_eq!(prepared.grid, [1 << 20, 1, 1]);
}

// -- what a tensor may be used for -----------------------------------------------------------------------------------------------------------

fn ta<'a>(addr: u64, shape: &'a [usize], dtype: DType, access: Access) -> TensorArg<'a> {
    // SAFETY: these addresses are never dereferenced: `prepare` only looks at the number. The region is as long as the tensor.
    let ptr = unsafe { DevicePtr::from_raw(addr, shape.iter().product::<usize>() as u64 * dtype.size() as u64, 0) };
    TensorArg::new(ptr, shape, dtype, access).unwrap()
}

#[test]
fn a_tensor_that_is_only_read_is_refused_for_a_parameter_the_kernel_writes() {
    let shape = [4096usize];
    let gelu = typed!(GeluErf); // x_ptr is In, y_ptr is Out
    let call = |x, y| gelu.prepare(&[ta(A, &shape, DType::Bf16, x), ta(B, &shape, DType::Bf16, y)], Some(&DEVICE));
    for (x, y) in [
        (Access::In, Access::Out),
        (Access::In, Access::InOut),
        (Access::Out, Access::Out),
        (Access::InOut, Access::InOut),
    ] {
        assert!(call(x, y).is_ok(), "{x:?}, {y:?}");
    }
    let e = call(Access::In, Access::In).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Access);
    assert!(
        e.to_string().contains("y_ptr") && e.to_string().contains("read-only") && e.to_string().contains("out"),
        "{e}"
    );
    assert_eq!(PreconditionKind::Access.as_str(), "access");
    // an InOut parameter (axpy's y, residual_add_rmsnorm's residual) needs a tensor that may be written too, and takes any that may
    let axpy = typed!(Axpy);
    let shape = [4096usize];
    let call = |y| axpy.prepare(&[ta(A, &shape, DType::F32, Access::In), ta(B, &shape, DType::F32, y)], Some(&DEVICE));
    assert!(call(Access::InOut).is_ok() && call(Access::Out).is_ok());
    let e = call(Access::In).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Access);
    assert!(e.to_string().contains("inout"), "{e}");
    // an In parameter takes whatever it is given: a tensor that may be written is read
    assert!(
        axpy.prepare(
            &[ta(A, &shape, DType::F32, Access::InOut), ta(B, &shape, DType::F32, Access::InOut)],
            Some(&DEVICE)
        )
        .is_ok()
    );
    // the check is made without a device as well
    assert_eq!(
        kind(
            &gelu
                .prepare(&[ta(A, &[4096], DType::Bf16, Access::In), ta(B, &[4096], DType::Bf16, Access::In)], None)
                .unwrap_err()
        ),
        PreconditionKind::Access
    );
}
