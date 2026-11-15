//! Launch shapes (`docs/manifest-schema-3.md`, sections 4.3 to 4.6), read and checked with no GPU, on synthetic manifests (`common::synthetic`): a
//! grid that multiplies dimensions, a persistent grid (cut to the programs the SMs run), thread-block clusters (the grid counts clusters, a launch
//! has the grid times the cluster in blocks, and the compile option says so) and a global scratch (the last tensor of a launch, held to the bytes
//! the launch needs and the alignment every scratch is given at). The refusals are Python's (`python/tests/test_launch_shapes.py`), and every new
//! field is in the manifest digest.

mod common;

use common::synthetic::{Kernel, alignment, json, read, refusal, scalar};
use tilefish::{Access, Arch, DType, DeviceInfo, DevicePtr, Error, PreconditionKind, TensorArg};

const DEVICE: DeviceInfo = DeviceInfo { ordinal: 0, context: None, max_grid: [(1 << 31) - 1, 65535, 65535] };

/// `scale(x: Tensor[T, C, f32], y: Out[Tensor[T, C, f32]], n_t: T, n_c: C)` with the grid `grid`.
fn scale(grid: &str) -> Kernel {
    Kernel::new(
        "triton",
        "scale",
        103,
        vec![
            common::synthetic::pointer(0, "x", "f32", "in", json!(["T", "C"]), None),
            common::synthetic::pointer(1, "y", "f32", "out", json!(["T", "C"]), None),
            scalar(2, "n_t", "i32", "T", json!({})),
            scalar(3, "n_c", "i32", "C", json!({})),
        ],
        vec![alignment(0, "x", 16), alignment(1, "y", 16)],
        json!({"T": {}, "C": {}}),
        grid,
    )
}

fn tensor(addr: u64, shape: &[usize], dtype: DType, access: Access) -> TensorArg<'_> {
    let bytes = shape.iter().product::<usize>() as u64 * dtype.size() as u64;
    // SAFETY: the address is never dereferenced: the checks only look at the number.
    #[allow(unsafe_code)]
    let region = unsafe { DevicePtr::from_raw(addr, bytes, 0) };
    TensorArg::new(region, shape, dtype, access).unwrap()
}

fn args(shape: &[usize]) -> Vec<TensorArg<'_>> {
    vec![tensor(1 << 40, shape, DType::F32, Access::In), tensor(1 << 41, shape, DType::F32, Access::Out)]
}

// -- a grid that multiplies -------------------------------------------------------------------------------------------------------------

#[test]
fn a_grid_that_multiplies_two_dimensions_is_computed_at_each_launch() {
    let m = read(&[scale("(cdiv(C*T, 1024),)")], &[]).expect("the manifest reads");
    let plan = m.plan("scale", &[], Arch::sm(103)).unwrap();
    for (shape, programs) in [([7, 300], 3), ([1, 1], 1), ([1024, 1024], 1024), ([5, 2048], 10)] {
        let p = plan.prepare(&args(&shape), Some(&DEVICE)).unwrap();
        assert_eq!(p.grid, [programs, 1, 1], "{shape:?}");
    }
    let m = read(&[scale("(2*C*T + 1, T, C*C)")], &[]).unwrap();
    let p = m.plan("scale", &[], Arch::sm(103)).unwrap().prepare(&args(&[3, 5]), Some(&DEVICE)).unwrap();
    assert_eq!(p.grid, [31, 3, 25]);
    // against the device's limits, like any grid
    let small = DeviceInfo { max_grid: [100, 65535, 65535], ..DEVICE };
    let e = m.plan("scale", &[], Arch::sm(103)).unwrap().prepare(&args(&[10, 10]), Some(&small)).unwrap_err();
    assert!(matches!(&e, Error::Unsupported(msg) if msg.contains("201 blocks along x")), "{e}");
}

// -- persistent grids -------------------------------------------------------------------------------------------------------------------

fn tiles(persistent: serde_json::Value, grid: &str) -> Kernel {
    let mut k = scale(grid);
    k.entry["persistent"] = persistent;
    k
}

#[test]
fn a_persistent_grid_is_cut_to_the_programs_the_sms_run() {
    let m =
        read(&[tiles(json!({"programs_per_sm": 1}), "(cdiv(C, 128)*cdiv(T, 128),)")], &[]).expect("the manifest reads");
    let entry = m.entry("scale", "sm_103").unwrap();
    assert_eq!(entry.persistent, Some(tilefish::Persistent { programs_per_sm: 1, sm_count: None }));
    let plan = m.plan("scale", &[], Arch::sm(103)).unwrap();
    let shape = [4096, 4096]; // 32 * 32 = 1024 tiles
    assert_eq!(plan.prepare(&args(&shape), Some(&DEVICE)).unwrap().grid, [1024, 1, 1]); // no SM count: the tiles
    for (sms, programs) in [(30, 30), (148, 148), (2000, 1024), (1, 1)] {
        let p = plan.prepare_for(&args(&shape), &[], Some(&DEVICE), Some(sms)).unwrap();
        assert_eq!(p.grid, [programs, 1, 1], "{sms}");
    }
    let two = read(&[tiles(json!({"programs_per_sm": 2}), "(cdiv(C, 128)*cdiv(T, 128),)")], &[]).unwrap();
    let p = two
        .plan("scale", &[], Arch::sm(103))
        .unwrap()
        .prepare_for(&args(&shape), &[], Some(&DEVICE), Some(30))
        .unwrap();
    assert_eq!(p.grid, [60, 1, 1]);
    let p = tilefish::Persistent { programs_per_sm: 1, sm_count: None };
    assert_eq!((p.programs(100, 30, 2), p.programs(100, 1, 2), p.programs(5, 30, 1)), (15, 1, 5));
    // in the digest
    let plain = read(&[scale("(cdiv(C, 128)*cdiv(T, 128),)")], &[]).unwrap();
    assert_ne!(m.digest(), plain.digest());
    assert_ne!(m.digest(), two.digest());
}

#[test]
fn a_persistent_entry_is_read_strictly() {
    let grid = "(cdiv(C, 128)*cdiv(T, 128),)";
    for (persistent, want) in [
        (json!({"programs_per_sm": 0}), "programs_per_sm must be an int from 1 to 32"),
        (json!({"programs_per_sm": 33}), "programs_per_sm must be an int from 1 to 32"),
        (json!({"programs_per_sm": 1, "sm_count": 0}), "sm_count must be an int from 1 to"),
        (json!({"programs_per_sm": 1, "extra": 1}), "unknown field"),
        (json!({"sm_count": 30}), "missing field"),
        (json!({"programs_per_sm": 1.5}), "invalid type"),
        (json!([1]), "persistent must be an object"),
    ] {
        let e = refusal(&[tiles(persistent.clone(), grid)], &[]);
        assert!(e.contains(want), "{persistent}: {e}");
    }
    let e = refusal(&[tiles(json!({"programs_per_sm": 1}), "(T, C)")], &[]);
    assert!(e.contains("a persistent grid has one entry"), "{e}");
    // a JSON null is as absent
    let m = read(&[tiles(json!(null), grid)], &[]).unwrap();
    assert_eq!(m.entry("scale", "sm_103").unwrap().persistent, None);
    let m = read(&[tiles(json!({"programs_per_sm": 1, "sm_count": null}), grid)], &[]).unwrap();
    assert_eq!(m.entry("scale", "sm_103").unwrap().persistent.unwrap().sm_count, None);
}

// -- thread-block clusters -----------------------------------------------------------------------------------------------------------------

fn clustered(num_ctas: u32, option: Option<u32>) -> Kernel {
    let mut k = scale("(cdiv(C*T, 1024),)");
    k.entry["launch"]["cluster_dims"] = json!([num_ctas, 1, 1]);
    k.entry["launch"]["num_ctas"] = num_ctas.into();
    if let Some(n) = option {
        k.entry["options"]["num_ctas"] = n.into();
    }
    k
}

#[test]
fn a_kernel_of_clusters_is_launched_with_the_grid_times_its_cluster() {
    let m = read(&[clustered(2, Some(2))], &[]).expect("the manifest reads");
    let entry = m.entry("scale", "sm_103").unwrap();
    assert_eq!(
        (entry.launch.cluster_dims, entry.launch.num_ctas, entry.launch.cluster()),
        ([2, 1, 1], 2, Some([2, 1, 1]))
    );
    let plan = m.plan("scale", &[], Arch::sm(103)).unwrap();
    assert_eq!(plan.cluster(), Some([2, 1, 1]));
    let p = plan.prepare(&args(&[7, 300]), Some(&DEVICE)).unwrap();
    assert_eq!((p.grid, p.cluster), ([6, 1, 1], Some([2, 1, 1]))); // 3 programs of 2 CTAs along x
    // a persistent grid of clusters: a program is 2 CTAs, so 30 SMs run 15 of them
    let mut k = clustered(2, Some(2));
    k.entry["persistent"] = json!({"programs_per_sm": 1});
    let m = read(&[k], &[]).unwrap();
    let p = m
        .plan("scale", &[], Arch::sm(103))
        .unwrap()
        .prepare_for(&args(&[1024, 1024]), &[], Some(&DEVICE), Some(30))
        .unwrap();
    assert_eq!(p.grid, [30, 1, 1]);
    // the limits of the device are of the blocks
    let small = DeviceInfo { max_grid: [5, 65535, 65535], ..DEVICE };
    let e = plan.prepare(&args(&[7, 300]), Some(&small)).unwrap_err();
    assert!(matches!(&e, Error::Unsupported(msg) if msg.contains("6 blocks along x")), "{e}");
}

#[test]
fn the_compile_option_of_a_cluster_is_the_launchs() {
    for (k, want) in [
        (clustered(2, None), "options.num_ctas"),
        (clustered(2, Some(4)), "options.num_ctas"),
        (clustered(1, Some(1)), "options.num_ctas"),
        (clustered(1, Some(2)), "options.num_ctas"),
    ] {
        let e = refusal(&[k], &[]);
        assert!(e.contains(want), "{e}");
    }
    // a cluster this host does not launch (more than the portable 8, another product than num_ctas) is read, and refused by the plan
    let mut k = clustered(16, Some(16));
    k.entry["launch"]["cluster_dims"] = json!([16, 1, 1]);
    let m = read(&[k], &[]).unwrap();
    let e = m.plan("scale", &[], Arch::sm(103)).unwrap_err();
    assert!(matches!(&e, Error::Unsupported(msg) if msg.contains("cluster_dims=(16, 1, 1), num_ctas=16")), "{e}");
}

#[test]
fn a_prebuilt_binary_launches_its_declared_cluster_and_nothing_else() {
    let binary = |launch: serde_json::Value| {
        let mut k = Kernel::new(
            "binary",
            "swap",
            103,
            vec![
                common::synthetic::pointer(0, "x", "f32", "in", json!(["N"]), None),
                common::synthetic::pointer(1, "y", "f32", "out", json!(["N"]), None),
            ],
            vec![alignment(0, "x", 16), alignment(1, "y", 16)],
            json!({"N": {}}),
            "(cdiv(N, 256),)",
        );
        k.entry["launch"] = launch;
        k
    };
    let launch = |cluster: [u32; 3], ctas: u32, scratch: u64| json!({"global_scratch_size": scratch, "profile_scratch_size": 0, "launch_pdl": false, "cluster_dims": cluster, "num_ctas": ctas, "cooperative": false});
    let m = read(&[binary(launch([2, 1, 1], 2, 0))], &[]).expect("a binary of clusters reads");
    let n = [1000usize];
    let p = m.plan("swap", &[], Arch::sm(103)).unwrap().prepare(&args(&n), Some(&DEVICE)).unwrap();
    assert_eq!((p.grid, p.cluster), ([8, 1, 1], Some([2, 1, 1]))); // 4 clusters of 2 blocks
    for bad in [launch([2, 1, 1], 1, 0), launch([16, 1, 1], 16, 0), launch([1, 1, 1], 1, 256)] {
        let e = refusal(&[binary(bad.clone())], &[]);
        assert!(e.contains("a prebuilt binary is launched with its parameters"), "{bad}: {e}");
    }
}

/// `plan` prepared for `x` and `y` of `shape`, and the global scratch `scratch`.
fn with_scratch<'a>(
    plan: &tilefish::Plan,
    shape: &'a [usize],
    scratch: TensorArg<'a>,
) -> tilefish::Result<tilefish::Prepared> {
    let mut a = args(shape);
    a.push(scratch);
    plan.prepare(&a, Some(&DEVICE))
}

// -- a global scratch -----------------------------------------------------------------------------------------------------------------------

fn scratchy(size: u64, align: Option<u32>) -> Kernel {
    let mut k = scale("(cdiv(C*T, 1024),)");
    k.entry["launch"]["global_scratch_size"] = size.into();
    if let Some(a) = align {
        k.entry["launch"]["global_scratch_align"] = a.into();
    }
    k
}

#[test]
fn a_global_scratch_is_the_last_tensor_of_a_launch_and_holds_what_the_launch_needs() {
    let m = read(&[scratchy(256, Some(128))], &[]).expect("the manifest reads");
    let plan = m.plan("scale", &[], Arch::sm(103)).unwrap();
    assert_eq!(plan.global_scratch(), Some(256));
    assert_eq!(plan.tensor_names().collect::<Vec<_>>(), ["x", "y", tilefish::GLOBAL_SCRATCH]);
    let shape = [7usize, 300]; // 3 programs: 768 bytes
    let base = 1u64 << 42;
    let p = with_scratch(&plan, &shape, tensor(base, &[768], DType::U8, Access::Out)).unwrap();
    assert_eq!(p.scratch_bytes, 768);
    let params = p.params();
    assert_eq!(params[4], base, "the global scratch slot is the scratch's address");
    assert_eq!(params[5], 0, "the profile scratch stays null");
    // what it is held to: its size, its alignment, its dtype, that it may be written, that it overlaps no tensor
    for (addr, bytes, dtype, access, kind) in [
        (base, 767, DType::U8, Access::Out, PreconditionKind::Shape),
        (base + 128, 768, DType::U8, Access::Out, PreconditionKind::PointerAlignment),
        (base, 768, DType::F32, Access::Out, PreconditionKind::Dtype),
        (base, 768, DType::U8, Access::In, PreconditionKind::Access),
        (1 << 41, 768, DType::U8, Access::Out, PreconditionKind::Alias),
    ] {
        let size = [bytes];
        let e = with_scratch(&plan, &shape, tensor(addr, &size, dtype, access)).unwrap_err();
        assert!(matches!(&e, Error::Precondition { kind: k, .. } if *k == kind), "{kind:?}: {e}");
    }
    // a launch without it gives one tensor fewer than the kernel takes
    let e = plan.prepare(&args(&shape), Some(&DEVICE)).unwrap_err();
    assert!(matches!(&e, Error::Launch(msg) if msg.contains("takes 3 tensors (x, y, global_scratch)")), "{e}");
}

#[test]
fn a_global_scratch_is_read_strictly() {
    for (k, want) in
        [(scratchy(0, Some(128)), "global_scratch_align"), (scratchy(256, Some(96)), "global_scratch_align")]
    {
        let e = refusal(&[k], &[]);
        assert!(e.contains(want), "{e}");
    }
    // a manifest of tilefish 0.0.6 records no alignment: it reads, and its plan is refused (export it again)
    let m = read(&[scratchy(256, None)], &[]).unwrap();
    let e = m.plan("scale", &[], Arch::sm(103)).unwrap_err();
    assert!(
        matches!(&e, Error::Unsupported(msg) if msg.contains("of an alignment the manifest does not record")),
        "{e}"
    );
    // in the digest
    assert_ne!(
        read(&[scratchy(256, Some(128))], &[]).unwrap().digest(),
        read(&[scratchy(256, Some(64))], &[]).unwrap().digest()
    );
}
