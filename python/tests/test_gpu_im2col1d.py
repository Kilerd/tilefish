"""``tilefish.im2col1d`` on a real GPU: the patches of a 1-D convolution, byte for byte against torch.

The sources hold every 16-bit pattern (NaN payloads, signed zeros, subnormals), so a copy that interprets a value would show. Each case is
checked against an independent computation on the integer bits (a gather at the window's positions, zero where a position is outside the
input), and the channel order with a symmetric padding against ``torch.nn.functional.unfold`` too. The inputs are channel-last activations
read through ``.permute(0, 2, 1)``, contiguous tensors and a view with an offset and odd strides; the channel counts are static (several
column blocks, with and without a masked tail) and runtime. ``jit``, ``replay`` and ``reference`` agree byte for byte, alone and in a plan.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import torch.nn.functional as F  # noqa: E402
from gpu_util import same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, bf16, f16, runtime  # noqa: E402
from tilefish._copyops import im2col1d_op  # noqa: E402

CUDA = "cuda"
TORCH = {bf16: torch.bfloat16, f16: torch.float16}


def words(*shape, seed=0, dtype=bf16):
    """A 16-bit tensor of ``shape`` holding the 65536 bit patterns in a shuffled cycle."""
    gen = torch.Generator().manual_seed(seed)
    n = 1
    for s in shape:
        n *= s
    w = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)[torch.randperm(65536, generator=gen)]
    return w.repeat(-(-n // 65536))[:n].view(TORCH[dtype]).reshape(shape).to(CUDA)


def patches(x, k, s, d, pad, order):
    """The patches computed on the bits: ``[N, Lo, C * K]``, column ``c * K + j`` (channel order) or ``j * C + c`` (tap order)."""
    n, c, length = x.shape
    left, right = pad
    lo = (length + left + right - d * (k - 1) - 1) // s + 1
    pos = torch.arange(lo, device=x.device)[:, None] * s - left + torch.arange(k, device=x.device)[None, :] * d  # [Lo, K]
    inside = (pos >= 0) & (pos < length)
    taken = x.view(torch.int16)[:, :, pos.clamp(0, length - 1)]  # [N, C, Lo, K]
    bits = torch.where(inside, taken, torch.zeros((), dtype=torch.int16, device=x.device))
    cols = bits.permute(0, 2, 1, 3) if order == "channel" else bits.permute(0, 2, 3, 1)
    return cols.reshape(n, lo, c * k)


def unfold(x, k, s, d, p):
    """``torch.nn.functional.unfold`` of ``x: [N, C, L]`` as ``[1, L]`` images: the channel order, a symmetric padding."""
    cols = F.unfold(x.unsqueeze(2), (1, k), dilation=(1, d), padding=(0, p), stride=(1, s))  # [N, C * K, Lo]
    return cols.transpose(1, 2)


def channel_last(n, length, c, seed, dtype=bf16):
    return words(n, length, c, seed=seed, dtype=dtype).permute(0, 2, 1)  # [N, L, C] read as [N, C, L]


def odd_view(seed):
    return words(3, 50, 401, seed=seed)[:, 3:40, 5::3]  # [3, 37, 132]: an offset, strides (20050, 401, 3)


# (input, dtype, K, stride, dilation, padding, order, static C)
CASES = {
    "c1024_k7_d3_causal_tap": (lambda: channel_last(2, 300, 1024, 1), bf16, 7, 1, 3, (18, 0), "tap", True),  # 7 full blocks
    "c1024_k7_d3_p9_channel": (lambda: channel_last(2, 300, 1024, 1), bf16, 7, 1, 3, (9, 9), "channel", True),
    "c300_k7_s2_d9_p2x5_tap": (lambda: channel_last(3, 129, 300, 2), bf16, 7, 2, 9, (2, 5), "tap", True),  # 2100 columns: a masked tail
    "c300_k7_s2_d9_p2x5_channel": (lambda: channel_last(3, 129, 300, 2), bf16, 7, 2, 9, (2, 5), "channel", True),
    "odd_k3_s2_d1_p1_channel": (lambda: odd_view(3), bf16, 3, 2, 1, (1, 1), "channel", False),
    "odd_k3_s1_d3_p2x5_tap": (lambda: odd_view(3), bf16, 3, 1, 3, (2, 5), "tap", False),
    "k1_s1_channel": (lambda: words(2, 40, 77, seed=4), bf16, 1, 1, 1, (0, 0), "channel", False),
    "k1_s2_tap_channel_last": (lambda: channel_last(2, 77, 40, 5), bf16, 1, 2, 1, (0, 0), "tap", False),
    "k3_d9_causal_channel": (lambda: words(2, 24, 100, seed=6), bf16, 3, 1, 9, (18, 0), "channel", False),
    "k7_d1_p2x5_tap": (lambda: words(2, 24, 100, seed=7), bf16, 7, 1, 1, (2, 5), "tap", False),
    "k7_d9_p27_channel_runtime": (lambda: channel_last(2, 90, 33, 8), bf16, 7, 1, 9, (27, 27), "channel", False),
    "k3_s2_d3_causal_tap_runtime": (lambda: channel_last(2, 90, 33, 9), bf16, 3, 2, 3, (6, 0), "tap", False),
    "f16_c64_k7_d3_p9_tap": (lambda: channel_last(2, 150, 64, 10, dtype=f16), f16, 7, 1, 3, (9, 9), "tap", True),
}


def op_of(case, x):
    _, dtype, k, s, d, pad, order, static = CASES[case]
    return im2col1d_op(dtype, k, stride=s, dilation=d, padding=pad, order=order, channels=x.shape[1] if static else None)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("im2col1d")
    ops = {id(o): o for o in (op_of(case, CASES[case][0]()) for case in CASES)}
    return directory, runtime.export(list(ops.values()), directory)


@pytest.mark.parametrize("case", list(CASES))
def test_im2col1d_jit_replay_and_reference_agree_with_torch(exported, case):
    directory, m = exported
    make, dtype, k, s, d, pad, order, _ = CASES[case]
    x = make()
    op = op_of(case, x)
    jit = tilefish.run(op, x)
    replay = tilefish.run(op, x, mode="replay", manifest=directory, pin=m.digest)
    ref = tilefish.run(op, x, mode="reference")
    assert same_bytes(jit, replay) and same_bytes(jit, ref)
    expected = patches(x, k, s, d, pad, order)
    assert jit.is_contiguous() and jit.dtype == TORCH[dtype] and jit.shape == expected.shape
    assert torch.equal(jit.view(torch.int16), expected)
    if order == "channel" and pad[0] == pad[1]:
        assert torch.equal(jit.view(torch.int16), unfold(x, k, s, d, pad[0]).view(torch.int16))
    if pad[0]:  # the first output position reads the left padding at its first tap: +0.0, all bits clear
        first_tap = jit[:, 0, :: k] if order == "channel" else jit[:, 0, : x.shape[1]]
        assert (first_tap.view(torch.int16) == 0).all()


def test_the_two_orders_are_one_permutation():
    x = channel_last(2, 300, 1024, 1)
    tap = tilefish.run(op_of("c1024_k7_d3_causal_tap", x), x)
    channel = tilefish.run(im2col1d_op(bf16, 7, dilation=3, padding=(18, 0), channels=1024), x)
    n, lo, _ = tap.shape
    assert torch.equal(tap.view(n, lo, 7, 1024).transpose(2, 3).reshape(n, lo, -1).view(torch.int16), channel.view(torch.int16))


# -- in a launch plan ----------------------------------------------------------------------------------------------------------

B = Dim("B", buckets=[2])
T = Dim("T", buckets=[64, 200])


@tilefish.graph
def conv_patches(x: Tensor[B, T, 256, bf16], y: Tensor[B, 48, T, bf16]):
    taps = tilefish.im2col1d(x.permute(0, 2, 1), 7, dilation=3, padding=(18, 0), order="tap")  # a causal convolution of a channel-last input
    channels = tilefish.im2col1d(y, 3, stride=2, padding=(2, 5))
    return taps, channels


def test_a_plan_of_im2col1d(tmp_path):
    p = runtime.export(conv_patches, tmp_path)
    for t in (64, 200):
        x, y = words(2, t, 256, seed=t), words(2, 48, t, seed=t + 1)
        jit = tilefish.run(conv_patches, x, y)
        replay = tilefish.run(conv_patches, x, y, mode="replay", plan=tmp_path, pin=p.digest)
        ref = tilefish.run(conv_patches, x, y, mode="reference")
        for a, b, c in zip(jit, replay, ref):
            assert same_bytes(a, b) and same_bytes(a, c)
        assert torch.equal(jit[0].view(torch.int16), patches(x.permute(0, 2, 1), 7, 1, 3, (18, 0), "tap"))
        assert torch.equal(jit[1].view(torch.int16), patches(y, 3, 2, 1, (2, 5), "channel"))
