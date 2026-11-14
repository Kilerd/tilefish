"""``tilefish.col2im1d`` on a real GPU: the overlap-add of a 1-D transposed convolution, byte for byte against torch.

The contract (:mod:`tilefish._copyops`) is ATen's CUDA ``col2im`` order: an f32 sum from ``+0.0`` over the input positions in ascending order,
one rounding per add, then the bias, then one rounding to the dtype of ``cols``. So:

* ``jit``, ``replay`` and ``reference`` agree byte for byte, over both column orders, both layouts, kernels 1, 2, 3, 7 and 16, strides 1, 2, 3 and 8
  (a stride larger than the kernel too), dilations 1, 2 and 3, symmetric and asymmetric paddings, output paddings, with and without a bias,
  static and runtime channel counts (one block of channels, several, a masked tail), contiguous ``cols``, views with an offset and transposed
  views, f32 (most), bf16 and f16, and f32 values that are all subnormal (a sum flushed to zero would show);
* each is ``torch.nn.functional.fold`` on CUDA (ATen's ``col2im``) byte for byte: in f32 with the bias added after, and in the dtype of ``cols``
  as it is when there is no bias;
* a transposed convolution, ``cols = x^T @ W.reshape(C_in, C * K)`` then ``col2im1d``, is ``torch.nn.functional.conv_transpose1d`` (cuDNN on)
  byte for byte on small-integer data, where every product and partial sum is exact in f32 and any order gives the same bits, over a grid of
  strides, paddings, dilations, output paddings and biases;
* on random data it is ``conv_transpose1d`` within a tolerance; the fraction of byte-identical elements is measured and printed (cuDNN on, and
  off: ATen's GEMM + ``col2im`` + bias), not asserted;
* the traps: a reference that adds in ascending tap order (the opposite order) differs from the binary on random data, and a wrong padding or
  stride differs from torch, so these tests can tell them apart;
* a launch plan of a GEMM (the pinned cuBLASLt library op, bf16 operands, an f32 result) into ``col2im1d`` runs the same in every mode and is
  ``conv_transpose1d``'s bytes on small integers.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import torch.nn.functional as F  # noqa: E402
from gpu_util import differing, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, TilefishRuntimeError, bf16, f16, f32, runtime  # noqa: E402
from tilefish._copyops import col2im1d_op  # noqa: E402

CUDA = "cuda"
TORCH = {f32: torch.float32, bf16: torch.bfloat16, f16: torch.float16}


@pytest.fixture(autouse=True)
def _no_tf32():
    """Full-precision f32 matmuls and convolutions for the torch side of every comparison."""
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.enabled
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.enabled = saved


def rand(shape, dtype, seed, scale=1.0):
    gen = torch.Generator(device=CUDA).manual_seed(seed)
    return (torch.randn(shape, generator=gen, device=CUDA) * scale).to(dtype)


def cols_of(n, li, c, k, dtype, seed, view="contiguous", scale=1.0):
    """``[n, li, c * k]`` random values: a contiguous tensor, a view with an offset into a larger one, or a transposed view (a ``[n, c * k, li]``
    product read as ``[n, li, c * k]``)."""
    if view == "offset":
        return rand((n, li + 5, c * k + 7), dtype, seed, scale)[:, 3 : li + 3, 5 : c * k + 5]
    if view == "transposed":
        return rand((n, c * k, li), dtype, seed, scale).transpose(1, 2)
    return rand((n, li, c * k), dtype, seed, scale)


def length(li, k, s, d, pad, extra):
    return (li - 1) * s - pad[0] - pad[1] + d * (k - 1) + extra + 1


def channel_major(cols, k, order):
    """``cols`` with column ``c * K + j`` (a pure permutation of the tap order's ``j * C + c``)."""
    if order == "channel":
        return cols
    n, li, ck = cols.shape
    return cols.reshape(n, li, k, ck // k).transpose(2, 3).reshape(n, li, ck)


def folded(cols, k, s, d, pad, extra, order):
    """``F.fold`` (ATen's ``col2im``, on the device of ``cols``, in its dtype) of the untrimmed output, trimmed by ``(left, right)`` and extended
    by ``+0.0`` where ``output_padding`` reaches past it: ``[N, C, Lo]``. Trimming and extending move values, they add nothing."""
    n, li, ck = cols.shape
    c = ck // k
    full = (li - 1) * s + d * (k - 1) + 1
    lo = length(li, k, s, d, pad, extra)
    y = F.fold(channel_major(cols, k, order).transpose(1, 2), (1, full), (1, k), dilation=(1, d), stride=(1, s))[:, :, 0, :]
    if pad[0] + lo > full:
        y = torch.cat([y, y.new_zeros(n, c, pad[0] + lo - full)], dim=2)
    return y[:, :, pad[0] : pad[0] + lo]


def expected(cols, bias, k, s, d, pad, extra, order, layout):
    """The contract through ``F.fold``: the f32 sum of the widened ``cols`` (ATen's order), ``+ bias`` in f32, one rounding, in the layout."""
    y = folded(cols.float(), k, s, d, pad, extra, order)
    if bias is not None:
        y = y + bias.float()[None, :, None]
    y = y.to(cols.dtype)
    return y if layout == "ncl" else y.transpose(1, 2).contiguous()


def ascending_taps(cols, bias, k, s, d, pad, extra, order, layout):
    """The trap: the reference's sum with the taps in ascending order (descending input positions), which is not the contract."""
    n, li, ck = cols.shape
    c = ck // k
    lo = length(li, k, s, d, pad, extra)
    taps = channel_major(cols, k, order).reshape(n, li, c, k)
    span = (li - 1) * s + 1
    acc = torch.zeros((n, c, lo), dtype=torch.float32, device=cols.device)
    for j in range(k):
        part = torch.zeros((n, c, max(pad[0] + lo, j * d + span)), dtype=torch.float32, device=cols.device)
        part[:, :, j * d : j * d + span : s] = taps[:, :, :, j].transpose(1, 2).float()
        acc = acc + part[:, :, pad[0] : pad[0] + lo]
    if bias is not None:
        acc = acc + bias.float()[None, :, None]
    y = acc.to(cols.dtype)
    return y if layout == "ncl" else y.transpose(1, 2).contiguous()


# ((N, Li, C, the view of cols), dtype, K, stride, dilation, padding, output_padding, order, layout, bias, static C)
CASES = {
    "f32_k7_s1_p3_channel_ncl_bias_c64": ((2, 40, 64, "contiguous"), f32, 7, 1, 1, (3, 3), 0, "channel", "ncl", True, True),
    "f32_k7_s1_p3_tap_nlc_c64": ((2, 40, 64, "contiguous"), f32, 7, 1, 1, (3, 3), 0, "tap", "nlc", False, True),
    "f32_k4_s2_p1_channel_ncl_bias_runtime": ((3, 33, 48, "contiguous"), f32, 4, 2, 1, (1, 1), 0, "channel", "ncl", True, False),
    "f32_k4_s2_p1_op1_tap_ncl_c40": ((2, 29, 40, "contiguous"), f32, 4, 2, 1, (1, 1), 1, "tap", "ncl", False, True),
    "f32_k16_s8_p4_channel_ncl_bias_c33": ((2, 21, 33, "contiguous"), f32, 16, 8, 1, (4, 4), 0, "channel", "ncl", True, True),
    "f32_k16_s8_p4_tap_nlc_bias_c33_transposed": ((2, 21, 33, "transposed"), f32, 16, 8, 1, (4, 4), 0, "tap", "nlc", True, True),
    "f32_k3_s8_op2_channel_ncl_c20": ((2, 15, 20, "contiguous"), f32, 3, 8, 1, (0, 0), 2, "channel", "ncl", False, True),  # stride > kernel
    "f32_k2_s3_p1x0_tap_nlc_runtime_offset": ((2, 26, 30, "offset"), f32, 2, 3, 1, (1, 0), 0, "tap", "nlc", False, False),
    "f32_k1_s1_channel_ncl_bias_c100": ((2, 37, 100, "contiguous"), f32, 1, 1, 1, (0, 0), 0, "channel", "ncl", True, True),
    "f32_k1_s2_op1_tap_nlc_runtime": ((2, 37, 50, "contiguous"), f32, 1, 2, 1, (0, 0), 1, "tap", "nlc", False, False),
    "f32_k3_d2_p4x0_channel_ncl_bias_c24": ((2, 31, 24, "contiguous"), f32, 3, 1, 2, (4, 0), 0, "channel", "ncl", True, True),
    "f32_k7_d3_p0x9_tap_nlc_runtime_transposed": ((2, 31, 24, "transposed"), f32, 7, 1, 3, (0, 9), 0, "tap", "nlc", False, False),
    "f32_k7_s3_d2_p2x5_op1_channel_nlc_bias_c36_offset": ((3, 19, 36, "offset"), f32, 7, 3, 2, (2, 5), 1, "channel", "nlc", True, True),
    "f32_k16_s2_d3_p7_tap_ncl_c1100": ((1, 12, 1100, "contiguous"), f32, 16, 2, 3, (7, 7), 0, "tap", "ncl", False, True),  # two blocks
    "f32_k3_s2_op1_channel_nlc_bias_runtime_c1500": ((2, 9, 1500, "contiguous"), f32, 3, 2, 1, (1, 1), 1, "channel", "nlc", True, False),
    "f32_k7_s2_op1_channel_ncl_bias_c1": ((2, 50, 1, "contiguous"), f32, 7, 2, 1, (3, 3), 1, "channel", "ncl", True, True),
    "f32_subnormal_k7_d2_tap_ncl_c16": ((2, 30, 16, "subnormal"), f32, 7, 1, 2, (6, 6), 0, "tap", "ncl", False, True),
    "bf16_k7_s2_p3_op1_channel_ncl_bias_c64": ((2, 40, 64, "contiguous"), bf16, 7, 2, 1, (3, 3), 1, "channel", "ncl", True, True),
    "bf16_k16_s8_p4_tap_nlc_runtime_transposed": ((2, 21, 40, "transposed"), bf16, 16, 8, 1, (4, 4), 0, "tap", "nlc", False, False),
    "bf16_k3_d3_p6x0_channel_nlc_c48_offset": ((2, 30, 48, "offset"), bf16, 3, 1, 3, (6, 0), 0, "channel", "nlc", False, True),
    "f16_k4_s2_p1_channel_ncl_bias_c32": ((2, 33, 32, "contiguous"), f16, 4, 2, 1, (1, 1), 0, "channel", "ncl", True, True),
    "f16_k7_s3_d2_p2x3_op2_tap_nlc_runtime": ((2, 25, 24, "contiguous"), f16, 7, 3, 2, (2, 3), 2, "tap", "nlc", False, False),
}


def op_of(case):
    (_, _, c, _), dtype, k, s, d, pad, extra, order, layout, bias, static = CASES[case]
    return col2im1d_op(
        dtype, k, stride=s, dilation=d, padding=pad, output_padding=extra, order=order, layout=layout, bias=bias, channels=c if static else None
    )


def args_of(case):
    """``(cols, bias)`` of a case (``bias`` is ``None`` when the op takes none), from seeds of the case's own."""
    (n, li, c, view), dtype, k, *_, bias, _ = CASES[case]
    seed = list(CASES).index(case)
    if view == "subnormal":  # f32 values of magnitude about 2**-140: every sum stays below 2**-126
        cols = cols_of(n, li, c, k, TORCH[dtype], seed, scale=2.0**-140)
    else:
        cols = cols_of(n, li, c, k, TORCH[dtype], seed, view)
    return cols, (rand((c,), TORCH[dtype], seed=1000 + seed) if bias else None)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("col2im1d")
    ops = {id(o): o for o in (op_of(case) for case in CASES)}
    return directory, runtime.export(list(ops.values()), directory)


@pytest.mark.parametrize("case", list(CASES))
def test_col2im1d_jit_replay_and_reference_are_f_fold(exported, case):
    directory, m = exported
    _, dtype, k, s, d, pad, extra, order, layout, _, _ = CASES[case]
    cols, bias = args_of(case)
    op = op_of(case)
    args = (cols,) if bias is None else (cols, bias)
    jit = tilefish.run(op, *args)
    replay = tilefish.run(op, *args, mode="replay", manifest=directory, pin=m.digest)
    ref = tilefish.run(op, *args, mode="reference")
    assert same_bytes(jit, replay) and same_bytes(jit, ref)
    n, li, ck = cols.shape
    lo = length(li, k, s, d, pad, extra)
    assert jit.is_contiguous() and jit.dtype == TORCH[dtype]
    assert tuple(jit.shape) == ((n, ck // k, lo) if layout == "ncl" else (n, lo, ck // k))
    assert same_bytes(jit, expected(cols, bias, k, s, d, pad, extra, order, layout))
    if bias is None:  # F.fold in the dtype of cols as it is: ATen accumulates in f32 and rounds once, at the store
        native = folded(cols, k, s, d, pad, extra, order)
        assert same_bytes(jit, native if layout == "ncl" else native.transpose(1, 2).contiguous())
    if pad[0] == pad[1] and extra < s and order == "channel" and layout == "ncl" and bias is None:  # F.fold with its own padding
        direct = F.fold(cols.transpose(1, 2), (1, lo), (1, k), dilation=(1, d), padding=(0, pad[0]), stride=(1, s))[:, :, 0, :]
        assert same_bytes(jit, direct)
    if "subnormal" in case:  # every sum is subnormal: a flush to zero would show
        assert (jit != 0).float().mean() > 0.9 and jit.abs().max() < 2.0**-126


def test_f32_channel_ncl_is_f_fold_on_cuda_byte_for_byte():
    """The main case of the contract, bare: f32 ``cols`` in the channel order, ``ncl``, no bias, against ``F.fold`` with torch's padding."""
    total = 0
    for seed, (k, s, d, p, extra) in enumerate([(7, 1, 1, 3, 0), (4, 2, 1, 1, 1), (16, 8, 1, 4, 0), (3, 8, 3, 0, 2), (5, 3, 2, 4, 1)]):
        cols = cols_of(2, 64, 96, k, torch.float32, 100 + seed)
        y = tilefish.run(col2im1d_op(f32, k, stride=s, dilation=d, padding=p, output_padding=extra), cols)
        lo = length(64, k, s, d, (p, p), extra)
        want = F.fold(cols.transpose(1, 2), (1, lo), (1, k), dilation=(1, d), padding=(0, p), stride=(1, s))[:, :, 0, :]
        assert same_bytes(y, want), (k, s, d, p, extra, differing(y, want))
        total += y.numel()
    print(f"\ncol2im1d measure: F.fold f32 channel ncl, {total} elements, 0 differing")


def test_the_two_orders_and_layouts_are_one_permutation():
    cols = cols_of(2, 30, 64, 7, torch.float32, 40)
    tap_cols = cols.reshape(2, 30, 64, 7).transpose(2, 3).reshape(2, 30, 448)  # the same products, column j * C + c
    channel = tilefish.run(col2im1d_op(f32, 7, stride=2, padding=3, channels=64), cols)
    tap = tilefish.run(col2im1d_op(f32, 7, stride=2, padding=3, order="tap", layout="nlc", channels=64), tap_cols)
    assert same_bytes(tap, channel.transpose(1, 2).contiguous())


def test_a_launch_whose_output_would_be_empty_is_refused():
    op = col2im1d_op(f32, 4, stride=4, padding=2)  # Lo = 4 Li - 4
    assert tilefish.run(op, cols_of(1, 2, 3, 4, torch.float32, 0)).shape == (1, 3, 4)
    with pytest.raises(TilefishRuntimeError, match="every grid size must be at least 1"):
        tilefish.run(op, cols_of(1, 1, 3, 4, torch.float32, 0))


# -- a transposed convolution end to end -----------------------------------------------------------------------------------------


def small_ints(shape, seed, low=1, high=3):
    """Integers of magnitude ``low..high`` with random signs, never zero (so no product is ``-0.0``), as f32."""
    gen = torch.Generator(device=CUDA).manual_seed(seed)
    magnitude = torch.randint(low, high + 1, shape, generator=gen, device=CUDA)
    sign = torch.randint(0, 2, shape, generator=gen, device=CUDA) * 2 - 1
    return (magnitude * sign).float()


def products(x, w, order):
    """``cols = x^T @ W.reshape(C_in, C * K)`` (``order="tap"``: ``W.permute(0, 2, 1)`` first), an f32 GEMM: ``[N, Li, C * K]``."""
    c_in, c, k = w.shape
    w2 = w.reshape(c_in, c * k) if order == "channel" else w.permute(0, 2, 1).reshape(c_in, k * c)
    return x.transpose(1, 2) @ w2


def transposed(x, w, bias, s, d, pad, extra):
    """``conv_transpose1d`` with ``padding=(left, right)``: torch's own padding when it is symmetric and ``output_padding`` is one torch takes,
    else the untrimmed result trimmed (and extended by the bias where ``output_padding`` reaches past it)."""
    left, right = pad
    if left == right and extra < max(s, d):
        return F.conv_transpose1d(x, w, bias, stride=s, padding=left, output_padding=extra, dilation=d)
    full = F.conv_transpose1d(x, w, bias, stride=s, dilation=d)
    n, c, lf = full.shape
    lo = lf - left - right + extra
    if left + lo > lf:
        tail = torch.zeros((n, c, left + lo - lf), device=full.device) + (0 if bias is None else bias[None, :, None])
        full = torch.cat([full, tail], dim=2)
    return full[:, :, left : left + lo]


E2E = [  # (K, stride, dilation, padding, output_padding)
    (7, 1, 1, (3, 3), 0),
    (4, 2, 1, (1, 1), 0),
    (4, 2, 1, (1, 1), 1),
    (16, 8, 1, (4, 4), 0),
    (3, 8, 1, (0, 0), 2),
    (2, 3, 1, (0, 0), 0),
    (3, 1, 2, (2, 2), 0),
    (3, 1, 3, (6, 0), 0),
    (7, 3, 2, (2, 5), 1),
    (5, 2, 3, (0, 0), 1),
    (1, 2, 1, (0, 0), 1),
    (16, 2, 3, (7, 7), 1),
]


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("order", ["channel", "tap"])
@pytest.mark.parametrize("window", E2E, ids=lambda w: "k{}_s{}_d{}_p{}x{}_op{}".format(w[0], w[1], w[2], *w[3], w[4]))
def test_a_transposed_convolution_of_small_integers_is_conv_transpose1d(window, order, bias):
    k, s, d, pad, extra = window
    c_in, c, n, li = 24, 20, 2, 33
    seed = 10 * E2E.index(window) + 2 * (order == "tap") + int(bias)
    x, w = small_ints((n, c_in, li), seed), small_ints((c_in, c, k), seed + 1)
    b = small_ints((c,), seed + 2, high=8) if bias else None
    torch.backends.cudnn.enabled = True
    want = transposed(x, w, b, s, d, pad, extra)
    for layout in ("ncl", "nlc"):
        op = col2im1d_op(f32, k, stride=s, dilation=d, padding=pad, output_padding=extra, order=order, layout=layout, bias=bias, channels=c)
        cols = products(x, w, order)
        y = tilefish.run(op, cols, b) if bias else tilefish.run(op, cols)
        assert same_bytes(y, want if layout == "ncl" else want.transpose(1, 2).contiguous()), (layout, differing(y, want))


def test_random_data_against_conv_transpose1d_is_close_and_the_byte_identity_is_measured():
    """cuDNN, and ATen's slow path with cuDNN off (a GEMM per batch element, ``col2im``, the bias): within a tolerance (asserted); the fraction of
    byte-identical elements is printed. With ``cols`` made by ``W2^T @ x[n]`` per batch element, the GEMM of ATen's slow path, cuDNN off is
    measured too."""
    lines = []
    for seed, (k, s, d, p, extra) in enumerate([(16, 8, 1, 4, 0), (7, 1, 1, 3, 0), (4, 2, 1, 1, 0), (3, 1, 2, 2, 0), (8, 4, 1, 2, 0)]):
        c_in, c, n, li = 64, 48, 2, 50
        x = rand((n, c_in, li), torch.float32, 200 + seed)
        w = rand((c_in, c, k), torch.float32, 300 + seed, scale=(c_in * k / s) ** -0.5)
        b = rand((c,), torch.float32, 400 + seed)
        op = col2im1d_op(f32, k, stride=s, dilation=d, padding=p, output_padding=extra, bias=True, channels=c)
        y = tilefish.run(op, products(x, w, "channel"), b)
        w2 = w.reshape(c_in, c * k)
        y_aten_gemm = tilefish.run(op, torch.stack([(w2.t() @ x[i]).t() for i in range(n)]), b)
        for cudnn in (True, False):
            torch.backends.cudnn.enabled = cudnn
            want = F.conv_transpose1d(x, w, b, stride=s, padding=p, output_padding=extra, dilation=d)
            torch.testing.assert_close(y, want, rtol=1e-5, atol=1e-5)
            same = 1 - differing(y, want) / y.numel()
            same_aten = 1 - differing(y_aten_gemm, want) / y.numel()
            lines.append(f"k{k}_s{s}_d{d}_p{p} cudnn={'on' if cudnn else 'off'}: {same:.4f} byte-identical (cols by x^T @ W2), "
                         f"{same_aten:.4f} (cols by W2^T @ x[n]), max |diff| {float((y - want).abs().max()):.3g}")
        torch.backends.cudnn.enabled = True
    print("\ncol2im1d measure: random f32 vs conv_transpose1d on " + torch.cuda.get_device_name() + "\n  " + "\n  ".join(lines))


# -- the traps -------------------------------------------------------------------------------------------------------------------


def test_the_order_of_the_adds_is_visible():
    """A reference that adds in ascending tap order gives other bits on random data (counted): the byte comparisons tell the orders apart."""
    k, s, d, pad = 16, 2, 1, (7, 7)  # 8 terms at every output position
    cols = cols_of(2, 64, 96, k, torch.float32, 50)
    op = col2im1d_op(f32, k, stride=s, dilation=d, padding=pad, channels=96)
    y = tilefish.run(op, cols)
    wrong = ascending_taps(cols, None, k, s, d, pad, 0, "channel", "ncl")
    fold = folded(cols, k, s, d, pad, 0, "channel")
    assert same_bytes(y, fold)
    against_binary, against_fold = differing(y, wrong), differing(fold, wrong)
    assert against_binary > 0 and against_fold > 0
    torch.testing.assert_close(y, wrong, rtol=1e-5, atol=1e-5)  # the same sum, another order: close, not equal
    print(f"\ncol2im1d measure: ascending-tap reference differs in {against_binary} of {y.numel()} elements (and from F.fold in {against_fold})")


def test_a_wrong_padding_or_stride_is_caught():
    k, s, d, pad = 4, 2, 1, (1, 1)
    x, w = small_ints((2, 16, 30), 60), small_ints((16, 12, k), 61)
    want = transposed(x, w, None, s, d, pad, 0)
    cols = products(x, w, "channel")
    right = tilefish.run(col2im1d_op(f32, k, stride=s, dilation=d, padding=pad), cols)
    assert same_bytes(right, want)
    shifted = tilefish.run(col2im1d_op(f32, k, stride=s, dilation=d, padding=(2, 0)), cols)  # the same length, shifted by one
    assert shifted.shape == want.shape and differing(shifted, want) > want.numel() // 2
    strided = tilefish.run(col2im1d_op(f32, k, stride=s + 1, dilation=d, padding=pad), cols)
    assert strided.shape != want.shape
    lo = min(strided.shape[2], want.shape[2])
    assert differing(strided[:, :, :lo].contiguous(), want[:, :, :lo].contiguous()) > 0
    print(f"\ncol2im1d measure: padding (2, 0) for (1, 1) differs in {differing(shifted, want)} of {want.numel()} elements")


# -- in a launch plan --------------------------------------------------------------------------------------------------------------

T = Dim("T", buckets=[16, 50])
M = Dim("M", buckets=[32, 100])


@tilefish.library("col2im1d_products", acc=f32, impl="cublaslt.gemm", layout="NN")
def products_gemm(a: Tensor[M, 32, bf16], b: Tensor[32, 96, bf16]) -> Tensor[M, 96, f32]: ...


@tilefish.graph(inputs=("x",), weights=("w", "bias"))
def upsample(x: Tensor[2, T, 32, bf16], w: Tensor[32, 96, bf16], bias: Tensor[24, f32]):
    cols = products_gemm(x.reshape(2 * T, 32), w).reshape(2, T, 96)  # a channel-last input: x^T is its rows
    return tilefish.col2im1d(cols, 4, stride=2, padding=1, bias=bias)


def test_a_plan_of_a_gemm_and_col2im1d_is_conv_transpose1d(tmp_path):
    p = runtime.export(upsample, tmp_path)
    for t in (16, 50):
        x = small_ints((2, t, 32), t).to(torch.bfloat16)
        w = small_ints((32, 24, 4), t + 1)
        b = small_ints((24,), t + 2, high=8)
        args = (x, w.reshape(32, 96).to(torch.bfloat16), b)
        jit = tilefish.run(upsample, *args)
        replay = tilefish.run(upsample, *args, mode="replay", plan=tmp_path, pin=p.digest)
        ref = tilefish.run(upsample, *args, mode="reference")
        assert same_bytes(jit, replay) and same_bytes(jit, ref)
        cols = tilefish.run(products_gemm, x.reshape(2 * t, 32), args[1]).reshape(2, t, 96)
        alone = tilefish.run(col2im1d_op(f32, 4, stride=2, padding=1, bias=True, channels=24), cols, b)  # the two ops, one after the other
        assert same_bytes(jit, alone)
        torch.backends.cudnn.enabled = True
        want = F.conv_transpose1d(x.permute(0, 2, 1).float(), w, b, stride=2, padding=1)
        assert jit.shape == (2, 24, 2 * t) and same_bytes(jit, want)
