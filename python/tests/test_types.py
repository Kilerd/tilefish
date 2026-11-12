import pytest

import tilefish
from tilefish import (
    BlockScaled,
    Dim,
    InOut,
    Out,
    Tensor,
    TensorType,
    TilefishDeclarationError,
    TilefishTypeError,
    bf16,
    cdiv,
    const,
    e2m1,
    e4m3,
    f32,
    ue8m0,
)

M = Dim("M")
D = Dim("D", 2560)
K = Dim("K", divisible_by=32)
N = Dim("N", divisible_by=128)

mxfp8 = BlockScaled(elem=e4m3, scale=ue8m0, block=32, axis=-1)
mxfp4 = BlockScaled(elem=e2m1, scale=ue8m0, block=32, axis=-1)
nvfp4 = BlockScaled(elem=e2m1, scale=e4m3, block=16, global_scale=f32)


def test_tensor_type_basics():
    t = Tensor[M, D, bf16]
    assert isinstance(t, TensorType)
    assert t.shape == (M, 2560)
    assert t.dtype is bf16
    assert t.rank == 2
    assert repr(t) == "Tensor[M, D, bf16]"
    assert Tensor[M, 2560, bf16] == t  # a static Dim is its value
    assert Tensor[M, D, f32] != t
    assert Tensor[D, M, bf16] != t
    assert hash(Tensor[M, 2560, bf16]) == hash(t)
    assert Tensor[bf16].shape == ()


def test_tensor_type_dim_expressions():
    F = Dim("F", 3072)
    t = Tensor[2 * F, D, bf16]
    assert t.shape == (6144, 2560)
    assert repr(Tensor[M + M, bf16]) == "Tensor[2*M, bf16]"


@pytest.mark.parametrize("build", [lambda: Tensor[M, D], lambda: Tensor[M, "x", bf16], lambda: Tensor[0, bf16]])
def test_invalid_tensor_types(build):
    with pytest.raises(TilefishDeclarationError):
        build()


def test_tensor_type_is_immutable():
    with pytest.raises(AttributeError):
        Tensor[M, bf16].dtype = f32


def test_access_markers_and_const():
    t = Tensor[M, D, f32]
    assert InOut[t].mode == "inout" and InOut[t].type == t
    assert Out[t].mode == "out"
    assert repr(Out[t]) == "Out[Tensor[M, D, f32]]"
    assert const[f32].dtype is f32
    assert const[f32] == const[f32]
    assert repr(const[f32]) == "const[f32]"
    with pytest.raises(TilefishDeclarationError):
        InOut[bf16]
    with pytest.raises(TilefishDeclarationError):
        const[3]


# -- BlockScaled -----------------------------------------------------------------------


def test_blockscaled_equality_and_repr():
    assert mxfp8 == BlockScaled(elem=e4m3, scale=ue8m0, block=32, axis=-1)
    assert mxfp8 != mxfp4
    assert mxfp8 != BlockScaled(elem=e4m3, scale=ue8m0, block=16)
    assert hash(mxfp8) == hash(BlockScaled(e4m3, ue8m0, 32))
    assert repr(mxfp8) == "BlockScaled(elem=e4m3, scale=ue8m0, block=32, axis=-1)"
    assert repr(nvfp4) == "BlockScaled(elem=e2m1, scale=e4m3, block=16, axis=-1, global_scale=f32)"
    assert nvfp4.global_scale is f32 and mxfp8.global_scale is None


def test_blockscaled_scale_shape_static():
    t = Tensor[128, D, mxfp8]
    assert t.scale_shape == (128, 80)  # 2560 / 32
    assert t.scale_type == Tensor[128, 80, ue8m0]


def test_blockscaled_scale_shape_symbolic():
    t = Tensor[N, K, mxfp8]
    assert t.scale_shape == (N, cdiv(K, 32))
    assert str(t.scale_shape[1]) == "cdiv(K, 32)"
    # a stricter divisibility implies the weaker one, and the scale shape is exact
    K64 = Dim("K64", divisible_by=64)
    assert Tensor[N, K64, mxfp8].scale_shape[1] == cdiv(K64, 32)


def test_blockscaled_scale_shape_other_blocks_and_axes():
    assert Tensor[128, 256, nvfp4].scale_shape == (128, 16)
    axis0 = BlockScaled(e4m3, ue8m0, 32, axis=0)
    t = Tensor[256, 64, axis0]
    assert t.scale_shape == (8, 64)
    # axis=-1 and the explicit last axis are the same format on a rank-2 tensor
    assert Tensor[8, 64, BlockScaled(e4m3, ue8m0, 32, axis=1)] == Tensor[8, 64, mxfp8]


def test_blockscaled_static_divisibility_violation():
    with pytest.raises(TilefishTypeError, match=r"axis 1 has size 100.*not divisible by block=32"):
        Tensor[M, 100, mxfp8]


def test_blockscaled_symbolic_divisibility_violation():
    with pytest.raises(TilefishTypeError, match=r"size M.*divisible_by=32"):
        Tensor[K, M, mxfp8]
    weak = Dim("W", divisible_by=16)
    with pytest.raises(TilefishTypeError, match="not divisible by block=32"):
        Tensor[N, weak, mxfp8]


def test_blockscaled_axis_out_of_range_and_bad_args():
    with pytest.raises(TilefishTypeError, match="out of range"):
        Tensor[K, BlockScaled(e4m3, ue8m0, 32, axis=2)]
    with pytest.raises(TilefishDeclarationError):
        BlockScaled(elem=e4m3, scale=ue8m0, block=0)
    with pytest.raises(TilefishDeclarationError):
        BlockScaled(elem="e4m3", scale=ue8m0, block=32)
    with pytest.raises(TilefishDeclarationError, match="multiple of 2"):
        BlockScaled(elem=e2m1, scale=ue8m0, block=3)


def test_blockscaled_scale_tensor_of_plain_dtype_is_none():
    assert Tensor[M, D, bf16].scale_shape is None
    assert tilefish.Tensor[K, mxfp4].dtype == mxfp4
