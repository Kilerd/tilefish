//! Checks that a mutation run left weakly tested, each pinned by a test that fails when the check is weakened: a comparison made loose
//! (`<` for `<=`, one side of an overlap dropped), an extent computed from the element count or with wrapping arithmetic, an offset that
//! does not reach the address of a view, a written view that is let past its memory.
//!
//! They run on the plans of the `sm_103` fixture (read with `Pin::TrustDirectory`; an edited copy where a case needs an entry the fixture
//! does not have), with a made-up device and made-up addresses: nothing is launched and nothing is dereferenced (`Plan::prepare` and the
//! constructors of a tensor only look at the numbers).

mod common;

use common::{Copy, fixtures};
use tilefish::{
    Access, Arch, DType, DeviceInfo, DevicePtr, Error, F32, Manifest, Pin, Plan, PreconditionKind, TensorArg,
    TensorMut, TensorRef, U8,
};

fn manifest() -> Manifest {
    Manifest::load(fixtures().join("sm_103"), Pin::TrustDirectory).unwrap()
}

fn plan(stem: &str) -> Plan {
    manifest().plan(stem, &[], Arch::sm(103)).unwrap()
}

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

/// A device like the B300's: 2^31 - 1 blocks along x, 65535 along y and z.
const DEVICE: DeviceInfo = DeviceInfo { ordinal: 0, context: None, max_grid: [(1 << 31) - 1, 65535, 65535] };

const A: u64 = 0x7000_0000_0000;
const B: u64 = 0x7000_1000_0000;
const C: u64 = 0x7000_2000_0000;
// addresses far enough apart for tensors of gigabytes
const BIG_A: u64 = 0x1000_0000_0000;
const BIG_B: u64 = 0x2000_0000_0000;
const BIG_C: u64 = 0x3000_0000_0000;

fn region(addr: u64, bytes: u64) -> DevicePtr<'static> {
    // SAFETY: the address is never dereferenced: `prepare` and the constructors of a tensor only look at the number.
    unsafe { DevicePtr::from_raw(addr, bytes, 0) }
}

/// A dense tensor at `addr`, over a region that holds exactly it.
fn dense(addr: u64, shape: &[usize], dtype: DType, access: Access) -> TensorArg<'_> {
    let bytes = shape.iter().product::<usize>() as u64 * dtype.size() as u64;
    TensorArg::new(region(addr, bytes), shape, dtype, access).unwrap()
}

/// A view whose first element is at `addr`, over a region that holds exactly what it reaches.
fn view<'a>(addr: u64, shape: &'a [usize], strides: &'a [usize], dtype: DType, access: Access) -> TensorArg<'a> {
    let reach = 1 + shape.iter().zip(strides).map(|(n, s)| (n - 1) * s).sum::<usize>();
    TensorArg::new_strided(region(addr, (reach * dtype.size()) as u64), shape, strides, 0, dtype, access).unwrap()
}

fn kind(e: &Error) -> PreconditionKind {
    e.precondition_kind().unwrap_or_else(|| panic!("not a precondition error: {e:?}"))
}

// -- aliasing ------------------------------------------------------------------------------------------------------------------------

/// A written tensor overlaps another whichever of the two starts first, and one that only touches the other (it ends where the other
/// starts, or starts where it ends) does not.
#[test]
fn a_written_tensor_that_starts_before_another_and_runs_into_it_is_refused_and_one_that_touches_it_is_not() {
    // axpy: y (InOut) += alpha * x (In), 4096 f32 each: 16384 bytes
    let axpy = plan("axpy");
    let shape = [4096usize];
    let bytes = 4096 * 4;
    let call = |x: u64, y: u64| {
        axpy.prepare(
            &[dense(x, &shape, DType::F32, Access::In), dense(y, &shape, DType::F32, Access::InOut)],
            Some(&DEVICE),
        )
    };
    for (x, y, what) in [
        (A + bytes - 16, A, "y starts before x and runs into its first 16 bytes"),
        (A + 16, A, "y starts 16 bytes before x"),
        (A, A + bytes - 16, "y starts in the last 16 bytes of x"),
        (A, A, "y is x"),
    ] {
        let e = call(x, y).unwrap_err();
        assert_eq!(kind(&e), PreconditionKind::Alias, "{what}");
        assert!(
            e.to_string().contains("argument 'y_ptr' is written in place but overlaps argument 'x_ptr'"),
            "{what}: {e}"
        );
    }
    assert!(call(A + bytes, A).is_ok(), "y ends where x starts");
    assert!(call(A, A + bytes).is_ok(), "y starts where x ends");

    // two written tensors: residual_add_rmsnorm's residual (InOut, f32) and out (Out, bf16), [4, 2560] each
    let fused = plan("residual_add_rmsnorm");
    let (shape, w) = ([4usize, 2560], [2560usize]);
    let (residual_bytes, out_bytes) = (4 * 2560 * 4, 4 * 2560 * 2);
    let call = |residual: u64, out: u64| {
        fused.prepare(
            &[
                dense(A, &shape, DType::Bf16, Access::In),
                dense(residual, &shape, DType::F32, Access::InOut),
                dense(C, &w, DType::Bf16, Access::In),
                dense(out, &shape, DType::Bf16, Access::Out),
            ],
            Some(&DEVICE),
        )
    };
    assert_eq!(kind(&call(B + out_bytes - 16, B).unwrap_err()), PreconditionKind::Alias, "out runs into the residual");
    assert_eq!(kind(&call(B, B + residual_bytes - 16).unwrap_err()), PreconditionKind::Alias, "out starts in it");
    assert!(call(B + out_bytes, B).is_ok(), "out ends where the residual starts");
    assert!(call(B, B + residual_bytes).is_ok(), "out starts where the residual ends");
}

/// What a strided view covers is its extent, not its element count: a written tensor placed after the elements a view has, but within
/// what it reaches, overlaps it.
#[test]
fn aliasing_is_decided_by_the_extent_of_a_view_and_not_by_its_element_count() {
    // head_copy: x is a strided view (In), y a dense tensor it writes. x is a third of a fused QKV, [4, 8, 64] bf16 with row stride
    // 1536: 2048 elements (4096 bytes), and an extent of 3 * 1536 + 7 * 64 + 63 + 1 = 5120 elements (10240 bytes)
    let head_copy = plan("head_copy");
    let (shape, strides) = ([4usize, 8, 64], [1536usize, 64, 1]);
    let (count_bytes, extent_bytes) = (2048 * 2, 5120 * 2);
    let out = [4usize, 8, 64];
    let out_strides = [512usize, 64, 1];
    let call = |y: u64| {
        // y is a written view with the dense strides: what a launch is given for a parameter that takes a contiguous tensor
        let y = view(y, &out, &out_strides, DType::Bf16, Access::Out);
        head_copy.prepare(&[view(A, &shape, &strides, DType::Bf16, Access::In), y], Some(&DEVICE))
    };
    for (y, what) in [
        (A + count_bytes, "y starts right after as many bytes as x has elements, inside its extent"),
        (A + extent_bytes - 16, "y starts in the last 16 bytes of x's extent"),
        (A - count_bytes + 16, "y starts before x and runs 16 bytes into it"),
    ] {
        let e = call(y).unwrap_err();
        assert_eq!(kind(&e), PreconditionKind::Alias, "{what}");
        assert!(e.to_string().contains("argument 'y'") && e.to_string().contains("argument 'x'"), "{what}: {e}");
    }
    assert!(call(A + extent_bytes).is_ok(), "y starts where x's extent ends");
    assert!(call(A - count_bytes).is_ok(), "y ends where x starts");
}

// -- max_elements ----------------------------------------------------------------------------------------------------------------------

/// `max_elements` is 2^31 - 1 (the largest `i32`): a tensor of exactly that many elements is accepted, and one of one more is refused.
#[test]
fn max_elements_takes_two_to_the_thirty_one_minus_one_elements_and_refuses_one_more() {
    // axpy_ptx: x, y, out are [n] f32 with max_elements, and n has no divisor (2^31 - 1 is odd, and prime)
    let axpy_ptx = plan("axpy_ptx");
    let call = |plan: &Plan, n: usize| {
        let shape = [n];
        plan.prepare(
            &[
                dense(BIG_A, &shape, DType::F32, Access::In),
                dense(BIG_B, &shape, DType::F32, Access::In),
                dense(BIG_C, &shape, DType::F32, Access::Out),
            ],
            Some(&DEVICE),
        )
        .map(|p| (p.grid, p.params().to_vec()))
    };
    let top = (1usize << 31) - 1;
    let (grid, params) = call(&axpy_ptx, top).unwrap();
    assert_eq!(grid, [1 << 24, 1, 1]); // cdiv(2^31 - 1, 128)
    assert_eq!(params[..4], [BIG_A, BIG_B, BIG_C, top as u64]);
    // one more element: `n_elem`, which carries n, no longer fits an i32 and refuses it first. With `n_elem = n - 1` the scalar fits,
    // and max_elements is what refuses the 2^31 elements
    let (_copy, shifted) = edited("axpy_ptx", |e| param(e, "n_elem")["dim"] = "n - 1".into());
    assert_eq!(call(&shifted, top).unwrap().1[3], top as u64 - 1);
    let e = call(&shifted, top + 1).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::MaxElements, "{e}");
    assert!(e.to_string().contains("has 2147483648 elements") && e.to_string().contains("the 2147483647 that"), "{e}");
}

// -- contiguity -------------------------------------------------------------------------------------------------------------------------

/// An expanded view (a stride of 0 along an axis of more than one element) is not contiguous: a parameter that takes a dense tensor
/// refuses it, whether the kernel reads it or writes it (a written one would have every row written to the same memory). A stride of 0 on
/// an axis of one element is not looked at, and a strided parameter takes a stride of 0.
#[test]
fn an_expanded_view_is_refused_for_a_dense_parameter_read_or_written() {
    // softmax: x (In) and out (Out) are dense [M, 1000] f32
    let softmax = plan("softmax");
    let shape = [4usize, 1000];
    let call = |args: [TensorArg<'_>; 2]| softmax.prepare(&args, Some(&DEVICE));
    for strides in [[0usize, 1], [1000, 0], [0, 0]] {
        let e = call([view(A, &shape, &strides, DType::F32, Access::In), dense(B, &shape, DType::F32, Access::Out)])
            .unwrap_err();
        assert_eq!(kind(&e), PreconditionKind::Contiguous, "x with strides {strides:?}");
        assert!(e.to_string().contains("argument 'x' is not contiguous"), "{e}");
        let e = call([dense(A, &shape, DType::F32, Access::In), view(B, &shape, &strides, DType::F32, Access::Out)])
            .unwrap_err();
        assert_eq!(kind(&e), PreconditionKind::Contiguous, "out with strides {strides:?}");
        assert!(e.to_string().contains("argument 'out' is not contiguous"), "{e}");
    }
    // one row: the stride of the size-1 axis is not looked at
    let row = [1usize, 1000];
    let expanded = [0usize, 1];
    assert!(
        call([view(A, &row, &expanded, DType::F32, Access::In), view(B, &row, &expanded, DType::F32, Access::Out)])
            .is_ok()
    );
    // head_copy's x is strided: an expanded view is what it takes, and its stride 0 is passed
    let head_copy = plan("head_copy");
    let (shape, strides) = ([4usize, 8, 64], [0usize, 64, 1]);
    let out = [4usize, 8, 64];
    let prepared = head_copy
        .prepare(
            &[view(A, &shape, &strides, DType::Bf16, Access::In), dense(B, &out, DType::Bf16, Access::Out)],
            Some(&DEVICE),
        )
        .unwrap();
    assert_eq!(prepared.params()[..4], [A, B, 0, 64]);
}

// -- solving the dimensions -------------------------------------------------------------------------------------------------------------

/// A dimension is at least 1: a size that would bind one to 0 is refused (`NotPositive`), not bound.
#[test]
fn a_dimension_that_solves_to_zero_is_refused() {
    // softmax with x declared [M + 1, 1000]: one row of x would be M = 0
    let (_copy, plan) = edited("softmax", |e| param(e, "x")["shape"] = serde_json::json!(["M + 1", 1000]));
    let call = |x_rows: usize, out_rows: usize| {
        let (x, out) = ([x_rows, 1000], [out_rows, 1000]);
        plan.prepare(&[dense(A, &x, DType::F32, Access::In), dense(B, &out, DType::F32, Access::Out)], Some(&DEVICE))
    };
    let e = call(1, 1).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Shape, "{e}");
    assert!(e.to_string().contains("argument 'x'") && e.to_string().contains("would need M = 0"), "{e}");
    // two rows of x are M = 1, which out agrees with
    assert_eq!(call(2, 1).unwrap().grid, [1, 1, 1]);
}

// -- the extent of a view, with no wrapping -----------------------------------------------------------------------------------------

/// The extent of a view (`1 + sum((n - 1) * s)`) and its bytes are computed without wrapping: a view whose extent passes 2^64 is refused
/// when it is made, however small the number it would wrap to; one that reaches exactly the last byte of its region is not.
#[test]
fn a_view_whose_extent_overflows_64_bits_is_refused_and_does_not_wrap_to_a_size_that_fits() {
    let tiny = region(A, 16);
    for (shape, strides, offset, dtype, what) in [
        (&[3usize][..], &[1usize << 63][..], 0usize, DType::U8, "1 + 2 * 2^63 wraps to 1"),
        (&[2, 2][..], &[1 << 63, 1 << 63][..], 0, DType::U8, "1 + 2^63 + 2^63 wraps to 1"),
        (&[(1 << 32) + 1][..], &[1 << 32][..], 0, DType::U8, "(2^32) * 2^32 wraps to 0"),
        (&[2][..], &[(1 << 62) + 1][..], 0, DType::F32, "2^62 + 2 elements of 4 bytes wrap to 8 bytes"),
        (&[1][..], &[1][..], usize::MAX, DType::U8, "an offset of 2^64 - 1 and one element wrap to 0"),
    ] {
        let e = TensorArg::new_strided(tiny, shape, strides, offset, dtype, Access::In).unwrap_err();
        assert!(e.to_string().contains("more than 2^64"), "{what}: {e}");
        // the typed tensors, read or written, are refused alike
        assert!(TensorRef::<U8>::new_strided(tiny, shape, strides, offset).is_err(), "{what}");
        assert!(TensorMut::<F32>::new_strided(tiny, shape, strides, offset).is_err(), "{what}");
    }
    // a dense tensor of 2^62 f32 elements is 2^64 bytes, which wraps to 0
    let e = TensorArg::new(region(A, 0), &[1 << 62], DType::F32, Access::In).unwrap_err();
    assert!(e.to_string().contains("needs more than 2^64 bytes"), "{e}");
    assert!(TensorMut::<F32>::new(region(A, 0), &[1 << 62]).is_err());
    // the largest extent there is: 2^64 - 1 bytes, in a region of as many
    let all = region(0, u64::MAX);
    assert!(TensorArg::new_strided(all, &[2], &[usize::MAX - 1], 0, DType::U8, Access::In).is_ok());
    assert!(TensorArg::new_strided(all, &[2], &[usize::MAX], 0, DType::U8, Access::In).is_err());
    assert!(TensorArg::new_strided(all, &[2], &[usize::MAX - 1], 1, DType::U8, Access::In).is_err());
}

// -- the offset of a view -------------------------------------------------------------------------------------------------------------

/// A view whose first element is element `offset` of its memory is at `base + offset * itemsize`, for every element size, and that is
/// the address a launch passes; an offset past the end of the memory is refused.
#[test]
fn a_view_starts_offset_elements_into_its_memory_and_not_past_its_end() {
    // (`from_cuda_strided` makes the same address from a cudarc slice; that needs a GPU: here, the region a host vouched for)
    let memory = region(A, 4096);
    let (shape, strides) = ([4usize], [1usize]);
    for dtype in [DType::U8, DType::Bf16, DType::F32, DType::I64] {
        for offset in [0usize, 1, 7, 100] {
            let view = TensorArg::new_strided(memory, &shape, &strides, offset, dtype, Access::In).unwrap();
            assert_eq!(view.addr(), A + (offset * dtype.size()) as u64, "{dtype:?} at element {offset}");
        }
    }
    assert_eq!(TensorRef::<F32>::new_strided(memory, &shape, &strides, 9).unwrap().addr(), A + 36);
    assert_eq!(TensorMut::<U8>::new_strided(memory, &shape, &strides, 9).unwrap().addr(), A + 9);

    // the kernel is given that address: strided_copy's x is a [R, 64] bf16 view (2-byte aligned) with a row stride to bind
    let strided_copy = plan("strided_copy");
    let (x, x_strides, y) = ([3usize, 64], [128usize, 1], [3usize, 64]);
    let source = region(A, 4096);
    let call = |offset: usize| {
        let x = TensorArg::new_strided(source, &x, &x_strides, offset, DType::Bf16, Access::In).unwrap();
        strided_copy.prepare(&[x, dense(B, &y, DType::Bf16, Access::Out)], Some(&DEVICE)).map(|p| p.params().to_vec())
    };
    assert_eq!(call(0).unwrap(), [A, B, 64, 128]);
    assert_eq!(call(3).unwrap(), [A + 6, B, 64, 128]);
    assert_eq!(call(1728).unwrap(), [A + 3456, B, 64, 128]); // 2 * 128 + 64 elements from element 1728: the last of the 2048
    assert!(TensorArg::new_strided(source, &x, &x_strides, 1729, DType::Bf16, Access::In).is_err());

    // past the end: 1024 f32 elements; element 1023 is the last one a view of one element may start at
    let (one, empty) = ([1usize], [0usize]);
    assert_eq!(TensorArg::new_strided(memory, &one, &[1], 1023, DType::F32, Access::In).unwrap().addr(), A + 4092);
    assert!(TensorArg::new_strided(memory, &one, &[1], 1024, DType::F32, Access::In).is_err());
    // a view of no elements may start at the end, and not after it
    assert_eq!(TensorArg::new_strided(memory, &empty, &[1], 1024, DType::F32, Access::In).unwrap().addr(), A + 4096);
    assert!(TensorArg::new_strided(memory, &empty, &[1], 1025, DType::F32, Access::In).is_err());
    assert!(TensorArg::new_strided(memory, &empty, &[1], usize::MAX, DType::F32, Access::In).is_err());
}

// -- a written view that reaches out of its memory ---------------------------------------------------------------------------------------

/// A written view that reaches past the end of its region cannot be made; and one that fits its region is still refused by a launch when
/// it is not what the parameter writes (a view with gaps for a dense parameter) or when it runs into another argument.
#[test]
fn a_written_view_that_reaches_out_of_its_region_is_refused_when_made_and_by_a_launch() {
    // 4 rows of 1000 f32 values are 16000 bytes: from element 1 on they reach 16004
    let memory = region(A, 16000);
    let (shape, strides) = ([4usize, 1000], [1000usize, 1]);
    let e = TensorMut::<F32>::new_strided(memory, &shape, &strides, 1).unwrap_err();
    assert!(
        e.to_string().contains("reaches 16004 bytes from the start of its region, but the region has 16000 bytes"),
        "{e}"
    );
    for access in [Access::Out, Access::InOut] {
        assert!(TensorArg::new_strided(memory, &shape, &strides, 1, DType::F32, access).is_err(), "{access:?}");
    }
    assert!(TensorMut::<F32>::new_strided(memory, &shape, &strides, 0).is_ok());
    // a row stride of 1001 reaches 3 * 1001 + 1000 elements: 16012 bytes
    assert!(TensorMut::<F32>::new_strided(memory, &shape, &[1001, 1], 0).is_err());
    // and a dense one: 4 rows do not fit 15996 bytes
    let e = TensorMut::<F32>::new(region(A, 15996), &shape).unwrap_err();
    assert!(e.to_string().contains("needs 16000 bytes, but its region has 15996 bytes"), "{e}");

    // by a launch: softmax writes out [M, 1000] contiguously
    let softmax = plan("softmax");
    let x = dense(A, &shape, DType::F32, Access::In);
    // a written view with gaps between its rows fits its region, and is refused for the dense parameter
    let gaps = [2000usize, 1];
    let out = TensorMut::<F32>::new_strided(region(B, 8 * 2000 * 4), &shape, &gaps, 0).unwrap().into_arg();
    assert_eq!(kind(&softmax.prepare(&[x, out], Some(&DEVICE)).unwrap_err()), PreconditionKind::Contiguous);
    // a written view that fits its region (a region that is not x's) and still reaches into x: the address space is all a launch sees
    let x = dense(A, &shape, DType::F32, Access::In);
    let out = TensorMut::<F32>::new_strided(region(A - 16000, 32000), &shape, &strides, 8).unwrap().into_arg();
    let e = softmax.prepare(&[x, out], Some(&DEVICE)).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Alias, "{e}");
}

/// The workspace of a GEMM is written: a `TensorRef` cannot be one. (`Gemm::check_call` refuses it; the unit tests of `gemm` run that check
/// without a GPU. Here: what a `TensorRef<U8>` of a region is, for a launch.)
#[test]
fn a_tensor_ref_is_read_only_and_a_launch_refuses_it_where_a_kernel_writes() {
    let workspace = TensorRef::<U8>::new(region(A, 4096), &[4096]).unwrap().into_arg();
    assert_eq!(workspace.access(), Access::In);
    // a kernel's written parameter refuses it as the GEMM's workspace is refused: softmax's out, given a TensorRef
    let shape = [4usize, 1000];
    let out = TensorRef::<F32>::new(region(B, 16000), &shape).unwrap().into_arg();
    let e = plan("softmax").prepare(&[dense(A, &shape, DType::F32, Access::In), out], Some(&DEVICE)).unwrap_err();
    assert_eq!(kind(&e), PreconditionKind::Access);
    assert!(e.to_string().contains("argument 'out' is a read-only tensor"), "{e}");
}
