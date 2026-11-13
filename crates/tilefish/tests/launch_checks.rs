//! Every precondition of a binary, checked before a launch, with no GPU.
//!
//! `Plan::prepare` is what `Kernel::launch` runs before it touches the driver: the dtype of each tensor, its rank,
//! the static sizes, 16-byte alignment, the exact bucket, `divisible_by`, `max_elements`, aliasing, the 32-bit range of
//! the runtime scalars, and the grid, which is evaluated from its expression. These tests run it on the plans of the `sm_103`
//! fixture, with shapes from the *generated launchers* (`tests/fixtures/generated.rs`, compiled here), and with made-up
//! addresses: nothing is launched.

mod common;

use common::fixtures;
use tilefish::{Arch, Args, DType, DevicePtr, Error, Manifest, Pin, Plan, PreconditionKind, TensorArg};

#[allow(dead_code)]
#[rustfmt::skip] // generated: regenerate it, do not format it
#[path = "fixtures/generated.rs"]
mod generated;

fn manifest() -> Manifest {
    Manifest::load(fixtures().join("sm_103"), Pin::Digest(&common::expected_digest("sm_103"))).unwrap()
}

fn raw_plan(stem: &str, buckets: &[(&str, u64)]) -> Plan {
    manifest().plan(stem, buckets, Arch::sm(103)).unwrap()
}

/// The plan of a generated launcher: its stem and buckets, and the shapes it was typed with.
macro_rules! typed {
    ($ty:ident) => {
        raw_plan(generated::$ty::STEM, generated::$ty::BUCKETS).with_shapes(generated::$ty::SHAPES).unwrap()
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

fn kind(e: &Error) -> PreconditionKind {
    e.precondition_kind().unwrap_or_else(|| panic!("not a precondition error: {e:?}"))
}

fn refused(plan: &Plan, tensors: &[TensorArg<'_>]) -> Error {
    plan.prepare(&Args::new(tensors), Some(0)).expect_err("the call was accepted")
}

#[test]
fn every_launcher_is_typed_with_shapes_that_agree_with_its_manifest() {
    macro_rules! all {
        ($($ty:ident),*) => {$( let _ = typed!($ty); )*};
    }
    all!(
        Axpy,
        BucketedRmsDb1000,
        BucketedRmsDb512,
        BucketedRmsDb96,
        GeluErf,
        GeluExact,
        Layernorm,
        ResidualAddRmsnorm,
        Rmsnorm,
        Softmax,
        Swiglu
    );
    assert_eq!(generated::MANIFEST_DIGEST, common::expected_digest("sm_103"));
}

#[test]
fn shapes_that_contradict_the_manifest_are_refused() {
    let plan = raw_plan("softmax", &[]);
    // the binary is built for 1000 columns
    let wrong: &[&[&str]] = &[&["M", "999"], &["M", "1000"]];
    assert!(matches!(plan.clone().with_shapes(wrong), Err(Error::Launch(m)) if m.contains("1000")));
    let wrong: &[&[&str]] = &[&["M"], &["M", "1000"]];
    assert!(plan.clone().with_shapes(wrong).is_err());
    let wrong: &[&[&str]] = &[&["M", "1000"]];
    assert!(plan.clone().with_shapes(wrong).is_err());
    let wrong: &[&[&str]] = &[&["M", "1000 +"], &["M", "1000"]];
    assert!(plan.with_shapes(wrong).is_err());
}

#[test]
fn a_gelu_is_checked_and_its_grid_and_scalar_come_from_the_shapes() {
    let plan = typed!(GeluErf);
    for (n, blocks) in [(16usize, 1u32), (1024, 1), (1040, 2), (4096, 4), (1 << 20, 1024)] {
        let shape = [n];
        let prepared =
            plan.prepare(&Args::new(&[t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16)]), Some(0)).unwrap();
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
    let e = plan.prepare(&Args::new(&[t(A, &shape, DType::Bf16)]), Some(0)).unwrap_err();
    assert!(matches!(e, Error::Launch(m) if m.contains("takes 2 tensors")));
}

#[test]
fn a_dimension_that_is_not_a_multiple_of_what_the_binary_assumes_is_refused() {
    let plan = typed!(GeluErf);
    let shape = [60usize];
    let e = refused(&plan, &[t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16)]);
    assert_eq!(kind(&e), PreconditionKind::Divisible);
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
}

#[test]
fn a_size_baked_into_the_binary_is_checked() {
    let plan = typed!(Rmsnorm);
    let ok = plan
        .prepare(
            &Args::new(&[t(A, &[3, 2048], DType::Bf16), t(B, &[2048], DType::Bf16), t(C, &[3, 2048], DType::Bf16)]),
            Some(0),
        )
        .unwrap();
    assert_eq!(ok.grid, [3, 1, 1]); // one program per row: the grid is (Rows,)
    let e = refused(&plan, &[t(A, &[3, 2000], DType::Bf16), t(B, &[2000], DType::Bf16), t(C, &[3, 2000], DType::Bf16)]);
    assert_eq!(kind(&e), PreconditionKind::StaticSize);
    assert!(
        refused(&plan, &[t(A, &[3, 2000], DType::Bf16), t(B, &[2000], DType::Bf16), t(C, &[3, 2000], DType::Bf16)])
            .to_string()
            .contains("requires 2048")
    );
    // without shape patterns the manifest's own static_size says the same
    let raw = raw_plan("rmsnorm", &[]);
    let args = [t(A, &[3, 2000], DType::Bf16), t(B, &[2000], DType::Bf16), t(C, &[3, 2000], DType::Bf16)];
    let e = raw.prepare(&Args::new(&args).dims(&[("Rows", 3)]), Some(0)).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::StaticSize);
}

#[test]
fn a_bucket_is_exact_there_is_no_padding() {
    let plan = typed!(BucketedRmsDb96);
    let ok = plan
        .prepare(
            &Args::new(&[t(A, &[3, 96], DType::Bf16), t(B, &[96], DType::Bf16), t(C, &[3, 96], DType::Bf16)]),
            Some(0),
        )
        .unwrap();
    assert_eq!(ok.grid, [3, 1, 1]);
    for db in [95usize, 97, 100, 512] {
        let e = refused(&plan, &[t(A, &[3, db], DType::Bf16), t(B, &[db], DType::Bf16), t(C, &[3, db], DType::Bf16)]);
        assert_eq!(kind(&e), PreconditionKind::Bucket, "Db = {db}");
        let e = refused(&plan, &[t(A, &[3, db], DType::Bf16), t(B, &[db], DType::Bf16), t(C, &[3, db], DType::Bf16)]);
        assert!(
            e.to_string().contains(&format!("dimension Db is {db}"))
                && e.to_string().contains("exactly 96")
                && e.to_string().contains("no padding"),
            "{e}"
        );
    }
    // the 512 binary takes 512
    let plan = typed!(BucketedRmsDb512);
    assert!(
        plan.prepare(
            &Args::new(&[t(A, &[2, 512], DType::Bf16), t(B, &[512], DType::Bf16), t(C, &[2, 512], DType::Bf16)]),
            Some(0)
        )
        .is_ok()
    );
}

#[test]
fn a_tensor_with_more_elements_than_the_i32_offsets_of_a_generated_kernel_reach_is_refused() {
    let plan = typed!(Softmax);
    let ok = plan
        .prepare(
            &Args::new(&[t(BIG_A, &[2_000_000, 1000], DType::F32), t(BIG_B, &[2_000_000, 1000], DType::F32)]),
            Some(0),
        )
        .unwrap();
    assert_eq!(ok.grid, [2_000_000, 1, 1]);
    // 2_200_000 * 1000 > 2^31 - 1
    let e = refused(&plan, &[t(BIG_A, &[2_200_000, 1000], DType::F32), t(BIG_B, &[2_200_000, 1000], DType::F32)]);
    assert_eq!(kind(&e), PreconditionKind::MaxElements);
    let e = refused(&plan, &[t(BIG_A, &[2_200_000, 1000], DType::F32), t(BIG_B, &[2_200_000, 1000], DType::F32)]);
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
    assert!(plan.prepare(&Args::new(&[t(BIG_A, &shape, DType::Bf16), t(BIG_B, &shape, DType::Bf16)]), Some(0)).is_ok());
}

#[test]
fn what_is_written_must_not_overlap_anything() {
    let plan = typed!(Axpy);
    let shape = [4096usize];
    // y += alpha * x, in place: y is written
    assert!(plan.prepare(&Args::new(&[t(A, &shape, DType::F32), t(B, &shape, DType::F32)]), Some(0)).is_ok());
    let e = refused(&plan, &[t(A, &shape, DType::F32), t(A, &shape, DType::F32)]);
    assert_eq!(kind(&e), PreconditionKind::Alias);
    let e = refused(&plan, &[t(A, &shape, DType::F32), t(A + 4096 * 4 - 16, &shape, DType::F32)]);
    assert_eq!(kind(&e), PreconditionKind::Alias, "partial overlap");
    assert!(
        plan.prepare(&Args::new(&[t(A, &shape, DType::F32), t(A + 4096 * 4, &shape, DType::F32)]), Some(0)).is_ok(),
        "adjacent is not overlapping"
    );
    // two read-only arguments may alias
    let plan = typed!(Swiglu);
    let shape = [2usize, 3072];
    assert!(
        plan.prepare(
            &Args::new(&[t(A, &shape, DType::Bf16), t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16)]),
            Some(0)
        )
        .is_ok()
    );
}

#[test]
fn a_swiglus_scalar_is_the_element_count_and_the_grid_follows_from_it() {
    for m in [1usize, 2, 33, 130] {
        let shape = [m, 3072];
        let args = [t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16), t(C, &shape, DType::Bf16)];
        // typed: M is bound by the shapes
        let prepared = typed!(Swiglu).prepare(&Args::new(&args), Some(0)).unwrap();
        assert_eq!(prepared.grid, [3 * m as u32, 1, 1]);
        assert_eq!(prepared.params(), [A, B, C, (3072 * m) as u64, 0, 0]);
        // with no shape patterns at all: the element count of `gate` is 3072*M, which gives M for the grid
        let raw = raw_plan("swiglu", &[]).prepare(&Args::new(&args), Some(0)).unwrap();
        assert_eq!(raw.grid, prepared.grid);
        assert_eq!(raw.params(), prepared.params());
    }
    // the count is derived: it is not an argument
    let shape = [2usize, 3072];
    let args = [t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16), t(C, &shape, DType::Bf16)];
    let e = typed!(Swiglu).prepare(&Args::new(&args).scalars(&[("n", 6144)]), Some(0)).unwrap_err();
    assert!(matches!(e, Error::Launch(m) if m.contains("derived")));
    // 3072 does not divide an element count that is not a multiple of it
    let shape = [5usize, 3000];
    let args = [t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16), t(C, &shape, DType::Bf16)];
    assert!(raw_plan("swiglu", &[]).prepare(&Args::new(&args), Some(0)).is_err());
}

#[test]
fn without_shapes_the_caller_gives_what_the_tensors_cannot() {
    let shape = [3usize, 2048];
    let w = [2048usize];
    let args = [t(A, &shape, DType::Bf16), t(B, &w, DType::Bf16), t(C, &shape, DType::Bf16)];
    let raw = raw_plan("rmsnorm", &[]);
    // the grid is (Rows,) and nothing in the manifest says what Rows is
    let e = raw.prepare(&Args::new(&args), Some(0)).unwrap_err();
    assert!(matches!(&e, Error::Launch(m) if m.contains("Rows") && m.contains("no argument determines")), "{e}");
    assert_eq!(raw.prepare(&Args::new(&args).dims(&[("Rows", 3)]), Some(0)).unwrap().grid, [3, 1, 1]);
    assert!(matches!(raw.prepare(&Args::new(&args).dims(&[("Nope", 3)]), Some(0)), Err(Error::Launch(_))));
    assert!(raw.prepare(&Args::new(&args).dims(&[("Rows", 0)]), Some(0)).is_err());
    assert!(raw.prepare(&Args::new(&args).dims(&[("Rows", 3), ("Rows", 4)]), Some(0)).is_err());

    // a scalar the manifest does not derive: `n_elements` carries `Count`
    let shape = [4096usize];
    let args = [t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16)];
    let raw = raw_plan("gelu_erf", &[]);
    let e = raw.prepare(&Args::new(&args), Some(0)).unwrap_err();
    assert!(matches!(&e, Error::Launch(m) if m.contains("n_elements")), "{e}");
    let prepared = raw.prepare(&Args::new(&args).scalars(&[("n_elements", 4096)]), Some(0)).unwrap();
    assert_eq!((prepared.grid, prepared.params()), ([4, 1, 1], &[A, B, 4096, 0, 0][..]));
    // a value that is given is checked against the rest: 4096 is not a multiple of 16 less one
    assert_eq!(
        kind(&raw.prepare(&Args::new(&args).scalars(&[("n_elements", 4095)]), Some(0)).unwrap_err()),
        PreconditionKind::Divisible
    );
    assert!(matches!(raw.prepare(&Args::new(&args).scalars(&[("nope", 1)]), Some(0)), Err(Error::Launch(_))));
    // and with the dimension instead of the scalar
    let prepared = raw.prepare(&Args::new(&args).dims(&[("Count", 2048)]), Some(0)).unwrap();
    assert_eq!(prepared.grid, [2, 1, 1], "the scalar is what the dimension says");
}

#[test]
fn a_kernel_is_launched_on_its_own_device() {
    let plan = typed!(GeluErf);
    let shape = [4096usize];
    let args = [t(A, &shape, DType::Bf16), t(B, &shape, DType::Bf16)];
    assert!(plan.prepare(&Args::new(&args), Some(0)).is_ok());
    let e = plan.prepare(&Args::new(&args), Some(1)).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Device);
}

#[test]
fn a_residual_add_rmsnorm_updates_its_inout_tensor_in_place_and_nothing_else_may_alias_it() {
    let plan = typed!(ResidualAddRmsnorm);
    let (shape, w) = ([4usize, 2560], [2560usize]);
    let ok = |a: u64, b: u64, c: u64, d: u64| {
        plan.prepare(
            &Args::new(&[
                t(a, &shape, DType::Bf16),
                t(b, &shape, DType::F32),
                t(c, &w, DType::Bf16),
                t(d, &shape, DType::Bf16),
            ]),
            Some(0),
        )
    };
    assert!(ok(A, B, C, D).is_ok());
    assert_eq!(kind(&ok(A, A, C, D).unwrap_err()), PreconditionKind::Alias); // the residual (written) over x
    assert_eq!(kind(&ok(A, B, C, B + 16).unwrap_err()), PreconditionKind::Alias); // the output over the residual
    // the dtype of the residual is f32: a bf16 residual is another binary
    let e = plan
        .prepare(
            &Args::new(&[
                t(A, &shape, DType::Bf16),
                t(B, &shape, DType::Bf16),
                t(C, &w, DType::Bf16),
                t(D, &shape, DType::Bf16),
            ]),
            Some(0),
        )
        .unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Dtype);
}
