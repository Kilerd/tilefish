"""Strided views, on a real GPU: a public Triton extern that takes a view runs it as it would run the same elements made contiguous.

The kernels are ``gpu_kernels_p1.py``: a head split that copies ``y[s, h, :] = x[s, h, :]`` for a view ``x`` of any strides (the q, k or v
slice of a fused qkv tensor, a transposed tensor, an expanded one, a slice that starts at any element), and a row softmax over a view whose
first two axes are permuted. A view's pointer is ``base + offset * itemsize`` and its strides reach the kernel as scalar parameters
(``StrideOf``); the binary is compiled without a divisibility hint for the pointer (``Strided[...]``) or with one (``Strided[..., 16]``,
which then refuses a view that does not start on 16 bytes). For each: ``jit`` and ``replay`` agree byte for byte, and both equal a torch
reference of the same view, with offsets that are 16-byte aligned and offsets that are not.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels_p1 as k  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishPreconditionError, runtime  # noqa: E402

BF16 = torch.bfloat16


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Every op of this file, exported once to a manifest directory."""
    directory = tmp_path_factory.mktemp("strided-manifest")
    manifest = runtime.export([k.head_copy, k.head_copy16, k.head_copy64, k.softmax3d], directory)
    return directory, manifest.digest


def run_both(op, exported, *args):
    """The result of ``jit`` and of ``replay`` of the exported manifest, which must be the same bytes."""
    directory, digest = exported
    jit = tilefish.run(op, *args)
    replay = tilefish.run(op, *args, mode="replay", manifest=directory, pin=digest)
    assert same_bytes(jit, replay), f"{op.name}: jit and replay differ"
    return jit


@pytest.mark.parametrize("rows", [1, 7, 40])
@pytest.mark.parametrize("which", [0, 1, 2])
def test_the_q_k_and_v_of_a_fused_qkv_are_copied_as_the_same_elements_made_contiguous(exported, rows, which):
    qkv = randn((rows, 3, 8, 64), BF16, seed=rows * 10 + which)
    view = qkv[:, which]  # [rows, 8, 64], strides (1536, 64, 1), starting which * 512 elements (1024 bytes) in
    assert view.stride() == (1536, 64, 1) and view.data_ptr() % 16 == 0
    for op in (k.head_copy, k.head_copy16, k.head_copy64):
        assert same_bytes(run_both(op, exported, view), view.contiguous()), op.name


@pytest.mark.parametrize("offset", [0, 1, 2, 3, 5, 8, 9])
def test_a_view_that_starts_at_any_element_runs_without_a_divisibility_hint(exported, offset):
    """``Strided[...]`` assumes the element's alignment only: a pointer 2, 4, 6 ... bytes past a 16-byte boundary is fine."""
    flat = randn((40 * 8 * 64 + 16,), BF16, seed=offset)
    view = flat[offset : offset + 40 * 8 * 64].view(40, 8, 64)
    assert view.data_ptr() % 16 == (2 * offset) % 16
    assert same_bytes(run_both(k.head_copy, exported, view), view.contiguous())
    assert same_bytes(run_both(k.head_copy64, exported, view), view.contiguous())


@pytest.mark.parametrize("offset", [0, 8, 16])
def test_the_aligned_variant_takes_a_view_that_starts_on_16_bytes(exported, offset):
    flat = randn((40 * 8 * 64 + 32,), BF16, seed=100 + offset)
    view = flat[offset : offset + 40 * 8 * 64].view(40, 8, 64)
    assert view.data_ptr() % 16 == 0
    assert same_bytes(run_both(k.head_copy16, exported, view), view.contiguous())


@pytest.mark.parametrize("offset", [1, 2, 3, 4, 5, 6, 7, 9])
def test_the_aligned_variant_refuses_a_view_that_does_not(exported, offset):
    """``Strided[..., 16]`` is compiled with tt.divisibility=16 on the pointer: the launch checks the address, in jit and in replay."""
    flat = randn((40 * 8 * 64 + 16,), BF16, seed=200 + offset)
    view = flat[offset : offset + 40 * 8 * 64].view(40, 8, 64)
    directory, digest = exported
    with pytest.raises(TilefishPreconditionError, match="not aligned to 16 bytes") as e:
        tilefish.run(k.head_copy16, view)
    assert e.value.kind == "pointer_alignment"
    with pytest.raises(TilefishPreconditionError, match="not aligned to 16 bytes"):
        tilefish.run(k.head_copy16, view, mode="replay", manifest=directory, pin=digest)


def test_a_transposed_tensor_is_a_view_with_a_huge_stride(exported):
    base = randn((8, 40, 64), BF16, seed=3)
    view = base.permute(1, 0, 2)  # [40, 8, 64], strides (64, 2560, 1)
    assert view.stride() == (64, 2560, 1)
    assert same_bytes(run_both(k.head_copy, exported, view), view.contiguous())


def test_an_expanded_tensor_has_stride_zero_and_runs(exported):
    base = randn((1, 8, 64), BF16, seed=4)
    view = base.expand(40, 8, 64)
    assert view.stride(0) == 0
    assert same_bytes(run_both(k.head_copy, exported, view), view.contiguous())


def test_a_contiguous_tensor_is_a_view_too(exported):
    x = randn((40, 8, 64), BF16, seed=5)
    assert same_bytes(run_both(k.head_copy, exported, x), x)


def test_the_inner_axis_must_have_stride_one(exported):
    """The kernel loads along the last axis contiguously: that stride is fixed to 1 by the declaration."""
    wide = randn((40, 8, 128), BF16, seed=6)
    view = wide[..., ::2]  # [40, 8, 64] with strides (1024, 128, 2)
    with pytest.raises(TilefishPreconditionError, match=r"the stride of axis 2 is 2, but the signature requires 1") as e:
        tilefish.run(k.head_copy, view)
    assert e.value.kind == "stride"


def test_a_view_of_another_dtype_is_refused(exported):
    with pytest.raises(TilefishPreconditionError, match="dtype"):
        tilefish.run(k.head_copy, randn((4, 8, 64), torch.float32, seed=7))


@pytest.mark.parametrize("a, b", [(5, 3), (16, 9), (2, 33)])
def test_a_row_softmax_over_a_permuted_view(exported, a, b):
    base = randn((b, a, 128), torch.float32, seed=a * 100 + b, scale=3.0)
    view = base.permute(1, 0, 2)  # [a, b, 128], strides (128, a * 128, 1)
    got = run_both(k.softmax3d, exported, view)
    ref = torch.softmax(view, dim=-1)
    assert (got - ref).abs().max() < 1e-6
    assert torch.allclose(got.sum(-1), torch.ones(a, b, device=got.device), atol=1e-5)


def test_the_permuted_view_and_its_contiguous_copy_give_the_same_bytes(exported):
    """A view and the same elements made contiguous are the same input: the kernel reads through the strides, the result is the same."""
    base = randn((6, 9, 128), torch.float32, seed=11)
    view = base.permute(1, 0, 2)
    assert same_bytes(tilefish.run(k.softmax3d, view), tilefish.run(k.softmax3d, view.contiguous()))


def test_the_stride_scalars_and_the_pointer_of_the_launch_are_the_views(exported):
    """What a launch passes: the pointer of the view (base + offset * itemsize) and its strides, canonical for the size-1 axes."""
    from tilefish._launch import canonical_strides

    flat = randn((1000,), BF16, seed=8)
    view = flat[3:3 + 4 * 8 * 8].view(4, 8, 8)
    assert view.data_ptr() == flat.data_ptr() + 3 * 2
    one = flat[:64].view(1, 8, 8)  # a size-1 axis: whatever torch says, the kernel is given the dense stride
    assert canonical_strides(one.shape, one.stride()) == (64, 8, 1)
