//! The regions a host vouches for, and the tensors that must fit them: no GPU.
//!
//! A tensor is made of a [`DevicePtr`], which has an address, a length in bytes and a device. A launch can only reach what a region
//! covers, because a tensor whose shape needs more bytes than its region has is refused when it is made, and a sub-region is checked
//! against the region it is cut from. The addresses here are made up and never dereferenced.

use tilefish::{Access, Bf16, DType, DevicePtr, Error, F16, F32, TensorArg, TensorMut, TensorRef, U8};

const BASE: u64 = 0x7000_0000_0000;

fn region<'a>(addr: u64, len: u64, device: i32) -> DevicePtr<'a> {
    // SAFETY: nothing here is ever dereferenced: the tests look at the numbers a region and a tensor carry.
    unsafe { DevicePtr::from_raw(addr, len, device) }
}

fn launch_error(e: Error) -> String {
    match e {
        Error::Launch(message) => message,
        other => panic!("expected a launch error, got {other:?}"),
    }
}

#[test]
fn a_region_reports_what_it_was_made_from() {
    let r = region(BASE, 4096, 3);
    assert_eq!((r.addr(), r.len_bytes(), r.device()), (BASE, 4096, 3));
}

#[test]
fn an_offset_moves_the_start_and_keeps_the_end_and_the_device() {
    let r = region(BASE, 100, 2);
    let o = r.offset(36).unwrap();
    assert_eq!((o.addr(), o.len_bytes(), o.device()), (BASE + 36, 64, 2));
    // an offset of 0 is the region, one of the whole length is an empty region at its end, one byte more is past the end
    assert_eq!((r.offset(0).unwrap().addr(), r.offset(0).unwrap().len_bytes()), (BASE, 100));
    let end = r.offset(100).unwrap();
    assert_eq!((end.addr(), end.len_bytes()), (BASE + 100, 0));
    let message = launch_error(r.offset(101).unwrap_err());
    assert!(message.contains("101 bytes") && message.contains("100 bytes"), "{message}");
    // an offset is checked against the length of the region, not against the address space
    assert!(region(BASE, 8, 0).offset(9).is_err());
    // an address that would wrap is refused too (a region that lies about its length cannot make one)
    let high = region(u64::MAX - 8, 64, 0);
    assert!(high.offset(8).is_ok());
    let message = launch_error(high.offset(16).unwrap_err());
    assert!(message.contains("address space"), "{message}");
}

#[test]
fn a_slice_is_cut_from_inside_its_region() {
    let r = region(BASE, 100, 1);
    let s = r.slice(10, 20).unwrap();
    assert_eq!((s.addr(), s.len_bytes(), s.device()), (BASE + 10, 20, 1));
    // the whole of it, and the empty one at the end, fit
    assert_eq!(r.slice(0, 100).unwrap().len_bytes(), 100);
    assert_eq!(r.slice(100, 0).unwrap().len_bytes(), 0);
    assert_eq!(r.slice(99, 1).unwrap().addr(), BASE + 99);
    // one byte past the end, a start past the end, a length that wraps around: refused
    for (start, len) in [(0, 101), (1, 100), (101, 0), (100, 1), (50, u64::MAX), (u64::MAX, 2), (u64::MAX, u64::MAX)] {
        let message = launch_error(r.slice(start, len).unwrap_err());
        assert!(message.contains("not within a region of 100 bytes"), "{start}+{len}: {message}");
    }
    let high = region(u64::MAX - 8, 64, 0);
    assert!(high.slice(8, 8).is_ok());
    assert!(launch_error(high.slice(16, 8).unwrap_err()).contains("address space"));
}

#[test]
fn a_tensor_that_needs_more_bytes_than_its_region_has_is_refused() {
    // exactly the bytes, and one byte less than they need
    let shape = [4usize, 8];
    let bytes = 4 * 8 * 4;
    let ok = TensorArg::new(region(BASE, bytes, 0), &shape, DType::F32, Access::InOut).unwrap();
    assert_eq!((ok.addr(), ok.shape(), ok.dtype(), ok.device()), (BASE, &shape[..], DType::F32, 0));
    let message =
        launch_error(TensorArg::new(region(BASE, bytes - 1, 0), &shape, DType::F32, Access::InOut).unwrap_err());
    assert!(message.contains("needs 128 bytes, but its region has 127 bytes"), "{message}");
    // a bigger region than the tensor needs is fine
    assert!(TensorArg::new(region(BASE, bytes + 1, 0), &shape, DType::F32, Access::InOut).is_ok());
    // the element size counts
    assert!(TensorArg::new(region(BASE, 64, 0), &[32], DType::Bf16, Access::InOut).is_ok());
    assert!(TensorArg::new(region(BASE, 63, 0), &[32], DType::Bf16, Access::InOut).is_err());
    assert!(TensorArg::new(region(BASE, 32, 0), &[32], DType::U8, Access::InOut).is_ok());
    assert!(TensorArg::new(region(BASE, 255, 0), &[32], DType::I64, Access::InOut).is_err());
    assert!(TensorArg::new(region(BASE, 256, 0), &[32], DType::I64, Access::InOut).is_ok());
}

#[test]
fn a_tensor_whose_size_overflows_is_refused_not_wrapped() {
    // 2^32 * 2^32 elements of 4 bytes is 2^66 bytes, and a rank-3 product overflows even 128 bits
    let whole = region(BASE, u64::MAX, 0);
    for shape in [vec![1usize << 32, 1 << 32], vec![1 << 63, 2], vec![1 << 62, 1 << 62, 1 << 62], vec![usize::MAX; 3]] {
        let message = launch_error(TensorArg::new(whole, &shape, DType::F32, Access::InOut).unwrap_err());
        assert!(message.contains("more than 2^64"), "{shape:?}: {message}");
    }
    // 2^64 bytes is one more than a region can have
    let message = launch_error(TensorArg::new(whole, &[1usize << 62], DType::F32, Access::InOut).unwrap_err());
    assert!(message.contains("more than 2^64"), "{message}");
    // what fits 64 bits as a number of bytes, but not the region: the message says how many it needs
    let message = launch_error(
        TensorArg::new(region(BASE, (1 << 40) - 1, 0), &[1usize << 38], DType::F32, Access::InOut).unwrap_err(),
    );
    assert!(message.contains("needs 1099511627776 bytes") && message.contains("has 1099511627775 bytes"), "{message}");
    // the largest tensors a region of u64::MAX bytes takes
    assert!(TensorArg::new(whole, &[usize::MAX], DType::U8, Access::InOut).is_ok());
    assert!(TensorArg::new(whole, &[usize::MAX / 4], DType::F32, Access::InOut).is_ok());
    assert!(TensorArg::new(whole, &[usize::MAX / 2, 2], DType::U8, Access::InOut).is_ok());
}

#[test]
fn the_typed_tensors_are_refused_like_the_untyped_one() {
    let shape = [16usize, 16];
    let r = region(BASE, 16 * 16 * 2, 4);
    let x = TensorRef::<Bf16>::new(r, &shape).unwrap();
    let y = TensorMut::<F16>::new(r, &shape).unwrap();
    assert_eq!((x.addr(), x.shape(), x.device(), x.elements()), (BASE, &shape[..], 4, Some(256)));
    assert_eq!((y.addr(), y.device(), y.elements()), (BASE, 4, Some(256)));
    // the untyped form it turns into is what a launch checks, with what the kernel may do to it
    let arg = x.into_arg();
    assert_eq!(
        (arg.addr(), arg.shape(), arg.dtype(), arg.device(), arg.access()),
        (BASE, &shape[..], DType::Bf16, 4, Access::In)
    );
    let arg = y.into_arg();
    assert_eq!((arg.dtype(), arg.access()), (DType::F16, Access::InOut));
    // one element too many for the region, in each of the four ways to make one
    let message = launch_error(TensorRef::<F32>::new(r, &shape).unwrap_err());
    assert!(message.contains("needs 1024 bytes, but its region has 512 bytes"), "{message}");
    assert!(TensorMut::<F32>::new(r, &shape).is_err());
    assert!(TensorRef::<Bf16>::new(r, &[16, 17]).is_err());
    assert!(TensorMut::<U8>::new(region(BASE, 255, 0), &[16, 16]).is_err());
    assert!(TensorMut::<U8>::new(region(BASE, 256, 0), &[16, 16]).is_ok());
    // an offset region is what a sub-tensor of a bigger buffer is
    let rows = region(BASE, 8 * 1024, 0);
    let second = TensorRef::<F32>::new(rows.slice(1024, 1024).unwrap(), &[16, 16]).unwrap();
    assert_eq!(second.addr(), BASE + 1024);
    assert!(TensorRef::<F32>::new(rows.slice(1024, 1023).unwrap(), &[16, 16]).is_err());
}

#[test]
fn the_element_count_of_a_broadcast_view_never_wraps() {
    // a view broadcast along both axes fits one element of memory, and has more elements than a usize counts: no count, rather than a
    // panic (a debug build) or a wrapped number (a release build)
    let r = region(BASE, 4, 0);
    let (huge, zero) = ([usize::MAX, 2], [0usize, 0]);
    let x = TensorRef::<F32>::new_strided(r, &huge, &zero, 0).unwrap();
    assert_eq!(x.elements(), None);
    let y = TensorMut::<F32>::new_strided(r, &huge, &zero, 0).unwrap();
    assert_eq!(y.elements(), None);
    // the largest count that fits is exact, and a size of 0 makes 0 whatever the others are (Python's `math.prod`)
    let (edge, ones) = ([usize::MAX, 1], [0usize, 0]);
    assert_eq!(TensorRef::<F32>::new_strided(r, &edge, &ones, 0).unwrap().elements(), Some(usize::MAX));
    let (empty, strides) = ([usize::MAX, 2, 0], [0usize, 0, 1]);
    assert_eq!(TensorRef::<F32>::new_strided(r, &empty, &strides, 0).unwrap().elements(), Some(0));
    // such a view is a tensor like any other for a launch: the checks count its elements exactly, as Python does (`launch_checks.rs`)
    assert_eq!(x.into_arg().shape(), huge);
}
