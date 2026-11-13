"""``tilefish.contiguous`` and ``tilefish.im2col`` on a real GPU: raw copies of strided views, byte for byte against torch.

The sources hold every bf16 bit pattern (NaN payloads, signed zeros, subnormals), so a copy that interprets a value would show. Views are
permutations (NCHW to NHWC, heads to tokens, a transpose), expanded axes (stride 0) and offsets. ``im2col`` reads a one-channel image
through a transposing view and an NHWC activation through a permutation to NCHW, and agrees with
``torch.nn.functional.unfold``. ``jit``, ``replay`` and ``reference`` agree byte for byte, alone and in a plan.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import torch.nn.functional as F  # noqa: E402
from gpu_util import same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, bf16, f32, runtime  # noqa: E402
from tilefish._copyops import contiguous_op, im2col_op  # noqa: E402

BF16 = torch.bfloat16
CUDA = "cuda"


def words(*shape, seed=0):
    """A bf16 tensor of ``shape`` holding the 65536 bit patterns in a shuffled cycle."""
    gen = torch.Generator().manual_seed(seed)
    n = 1
    for s in shape:
        n *= s
    w = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)[torch.randperm(65536, generator=gen)]
    return w.repeat(-(-n // 65536))[:n].view(BF16).reshape(shape).to(CUDA)


def three(op, *args, directory, digest):
    jit = tilefish.run(op, *args)
    replay = tilefish.run(op, *args, mode="replay", manifest=directory, pin=digest)
    ref = tilefish.run(op, *args, mode="reference")
    assert same_bytes(jit, replay) and same_bytes(jit, ref)
    return jit


def views():
    nhwc = words(5, 16, 35, 200, seed=1).permute(0, 2, 3, 1)  # [N, C, H, W] read as [N, H, W, C]
    heads = words(16, 77, 128, seed=2).permute(1, 0, 2)  # [H, S, D] read as [S, H, D]
    transposed = words(300, 70, seed=3).t()
    expanded = words(1, 50, 64, seed=4).expand(3, 50, 64)  # stride 0 on the first axis
    offset = words(40, 130, seed=5)[3:, 1:129].t()  # an offset, both strides odd
    flat = words(9000, seed=6)[1::3]
    return [nhwc, heads, transposed, expanded, offset, flat]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("copyops")
    ops = [contiguous_op(v.dim(), bf16, last) for v in views() for last in (None, v.shape[-1])]
    ops += [im2col_op(bf16, 3, 2, 1, channels=1), im2col_op(bf16, 3, 2, 1, channels=192), im2col_op(bf16, (2, 3), 1, 0)]
    unique = list({id(o): o for o in ops}.values())
    return directory, runtime.export(unique, directory)


def test_contiguous_copies_every_view_as_torch_does(exported):
    directory, m = exported
    for v in views():
        assert not v.is_contiguous()
        for last in (None, v.shape[-1]):  # the last axis a runtime size, or compiled in with a tile fitted to it
            got = three(contiguous_op(v.dim(), bf16, last), v, directory=directory, digest=m.digest)
            assert got.is_contiguous() and got.shape == v.shape
            assert torch.equal(got.view(torch.int16), v.contiguous().view(torch.int16))


def test_contiguous_of_f32_and_a_contiguous_input():
    x = torch.randn(7, 33, 65, device=CUDA)
    x.view(torch.int32)[0, 0, :5] = torch.tensor([0x7FC00001, -4194303, -(2**31), 1, 0x7F800000], dtype=torch.int32)
    for v in (x, x.permute(2, 0, 1)):
        assert same_bytes(tilefish.run(contiguous_op(3, f32), v), v.contiguous())


def unfold(x, k, s, p):
    """``torch.nn.functional.unfold`` as ``[N, Ho, Wo, C * KH * KW]``."""
    n, c, h, w = x.shape
    cols = F.unfold(x, k, padding=p, stride=s)  # [N, C*KH*KW, L], channel-major
    ho = (h + 2 * p[0] - k[0]) // s[0] + 1
    wo = (w + 2 * p[1] - k[1]) // s[1] + 1
    return cols.transpose(1, 2).reshape(n, ho, wo, -1)


def test_im2col_of_one_channel_through_a_transposing_view(exported):
    directory, m = exported
    img = words(2, 101, 40, seed=7)  # [N, W, 40]: read as [N, 1, 40, W]
    x = img.reshape(2, 1, 101, 40).permute(0, 1, 3, 2)
    got = three(im2col_op(bf16, 3, 2, 1, channels=1), x, directory=directory, digest=m.digest)
    assert got.shape == (2, 20, 51, 9)
    assert torch.equal(got.view(torch.int16), unfold(x, (3, 3), (2, 2), (1, 1)).view(torch.int16))
    border = got[:, 0, :, :3]  # the first output row reads the row above the input: padding
    assert (border.view(torch.int16) == 0).all()  # +0.0, all bits clear


def test_im2col_of_an_nhwc_activation(exported):
    directory, m = exported
    act = words(2, 30, 27, 192, seed=8)  # NHWC
    x = act.permute(0, 3, 1, 2)
    got = three(im2col_op(bf16, 3, 2, 1, channels=192), x, directory=directory, digest=m.digest)
    assert got.shape == (2, 15, 14, 1728)  # 1728 columns: two chunks of 1024, the last one masked
    assert torch.equal(got.view(torch.int16), unfold(x, (3, 3), (2, 2), (1, 1)).view(torch.int16))


def test_im2col_with_a_runtime_channel_count_and_another_window(exported):
    directory, m = exported
    x = words(3, 5, 9, 7, seed=9)
    got = three(im2col_op(bf16, (2, 3), 1, 0), x, directory=directory, digest=m.digest)
    assert torch.equal(got.view(torch.int16), unfold(x, (2, 3), (1, 1), (0, 0)).view(torch.int16))


# -- in a launch plan ----------------------------------------------------------------------------------------------------------

N = Dim("N", buckets=[3, 5])
H = Dim("H", buckets=[35])
B = Dim("B", buckets=[2])
L = Dim("L", buckets=[101])


@tilefish.graph
def permute_then_im2col(x: Tensor[N, 16, H, 200, bf16], img: Tensor[B, L, 40, bf16]):
    nhwc = tilefish.contiguous(x.permute(0, 2, 3, 1))
    patches = tilefish.im2col(img.reshape(B, 1, L, 40).permute(0, 1, 3, 2), kernel=3, stride=2, padding=1)
    return nhwc, patches


def test_a_plan_of_generated_ops(tmp_path):
    p = runtime.export(permute_then_im2col, tmp_path)
    img = words(2, 101, 40, seed=10)
    for n in (3, 5):
        x = words(n, 16, 35, 200, seed=n)
        jit = tilefish.run(permute_then_im2col, x, img)
        replay = tilefish.run(permute_then_im2col, x, img, mode="replay", plan=tmp_path, pin=p.digest)
        ref = tilefish.run(permute_then_im2col, x, img, mode="reference")
        for a, b, c in zip(jit, replay, ref):
            assert same_bytes(a, b) and same_bytes(a, c)
        assert same_bytes(jit[0], x.permute(0, 2, 3, 1).contiguous())
