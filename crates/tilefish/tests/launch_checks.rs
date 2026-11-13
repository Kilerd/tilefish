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
use tilefish::{Arch, DType, DeviceInfo, DevicePtr, Error, Manifest, Pin, Plan, PreconditionKind, TensorArg};

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
    // SAFETY: these addresses are never dereferenced: `prepare` only looks at the number.
    TensorArg::new(unsafe { DevicePtr::from_raw(addr) }, shape, dtype, 0)
}

const A: u64 = 0x7000_0000_0000;
const B: u64 = 0x7000_1000_0000;
const C: u64 = 0x7000_2000_0000;
const D: u64 = 0x7000_3000_0000;
// two addresses that are far enough apart for tensors of gigabytes
const BIG_A: u64 = 0x1000_0000_0000;
const BIG_B: u64 = 0x2000_0000_0000;

/// A device like the B300's: 2^31 - 1 blocks along x, 65535 along y and z.
const DEVICE: DeviceInfo = DeviceInfo { ordinal: 0, max_grid: [(1 << 31) - 1, 65535, 65535] };

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
        let roomy = DeviceInfo { ordinal: 0, max_grid: [i64::from(u32::MAX), 1 << 20, 1 << 20] };
        assert!(plan.prepare(&args, Some(&roomy)).is_ok(), "{grid}");
    }
    // x, from the shapes: M is the number of rows
    let (_copy, manifest) = with_grid("(M, 1, 1)");
    let plan = manifest.plan("softmax", &[], Arch::sm(103)).unwrap();
    let rows = [3_000_000_000usize / 1000, 1000];
    let args = [t(BIG_A, &rows, DType::F32), t(BIG_B, &rows, DType::F32)];
    // 3_000_000 rows: fine along x, but not for a device that takes 1_000_000 there
    assert!(plan.prepare(&args, Some(&DEVICE)).is_err()); // (it is also more than the i32 offsets reach: max_elements)
    let small = DeviceInfo { ordinal: 0, max_grid: [1_000_000, 65535, 65535] };
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
