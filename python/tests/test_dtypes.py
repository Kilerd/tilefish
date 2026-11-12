import pytest

from tilefish import DType, bf16, e2m1, e4m3, e5m2, f16, f32, i32, i64, u8, ue8m0

ALL = [bf16, f16, f32, i32, i64, u8, e4m3, e5m2, e2m1, ue8m0]


def test_triton_pointer_names():
    assert {d.name: d.triton_ptr for d in ALL} == {
        "bf16": "*bf16",
        "f16": "*fp16",
        "f32": "*fp32",
        "i32": "*i32",
        "i64": "*i64",
        "u8": "*u8",
        "e4m3": "*fp8e4nv",
        "e5m2": "*fp8e5",
        "e2m1": "*u8",
        "ue8m0": "*u8",
    }


def test_triton_scalar_names():
    assert f32.triton_scalar == "fp32"
    assert i32.triton_scalar == "i32"
    assert e2m1.triton_scalar is None


def test_bit_widths():
    assert [d.bits for d in ALL] == [16, 16, 32, 32, 64, 8, 8, 8, 4, 8]


def test_e2m1_is_stored_packed():
    assert e2m1.bits == 4
    assert e2m1.elems_per_byte == 2
    assert e2m1.storage_bits == 8
    assert bf16.elems_per_byte == 1


def test_identity_and_repr():
    assert repr(bf16) == "bf16"
    assert str(e4m3) == "e4m3"
    assert bf16 == bf16 and bf16 != f16
    assert len({bf16, bf16, f16}) == 2
    assert isinstance(bf16, DType)


@pytest.mark.parametrize(
    "src,dst,narrowing",
    [
        (f32, bf16, True),
        (f32, f16, True),
        (bf16, f32, False),
        (f16, f32, False),
        (bf16, f16, True),  # same width, different range
        (f16, bf16, True),
        (e4m3, bf16, False),
        (e4m3, f16, False),
        (e5m2, f16, False),
        (e5m2, e4m3, True),
        (e4m3, e5m2, True),
        (f32, e4m3, True),
        (e2m1, bf16, False),
        (ue8m0, f32, False),
        (f32, ue8m0, True),
        (i32, i64, False),
        (i64, i32, True),
        (u8, i32, False),
        (i32, u8, True),
        (u8, bf16, False),
        (i32, f32, True),  # 32-bit ints do not fit a 24-bit significand
        (f32, i32, True),
        (bf16, bf16, False),
    ],
)
def test_narrowing_casts(src, dst, narrowing):
    assert src.is_narrowing_to(dst) is narrowing
