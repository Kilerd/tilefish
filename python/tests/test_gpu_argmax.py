"""``argmax`` and ``argmin`` on a real GPU: the binaries against torch's own ``argmax`` / ``argmin`` and against the reference, on adversarial rows.

The rows hold NaNs of both signs and several payloads (alone, first, last, several), signed zeros in both orders, infinities, ties, subnormals,
rows of nothing but ``-inf``, ``+inf`` or NaN, and random data; the row lengths are not powers of two (masked lanes) and are (none). ``jit``,
``replay`` and ``reference`` agree byte for byte, and they agree with ``torch.argmax`` / ``torch.argmin`` on CUDA (the contract is torch's rule).
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import argmax_cases as ac  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, bf16, runtime  # noqa: E402

BF16, F16, F32 = torch.bfloat16, torch.float16, torch.float32
CUDA = "cuda"
INT = {BF16: torch.int16, F16: torch.int16, F32: torch.int32}
NAN_WORDS = {
    BF16: [0x7FC0, 0xFFC0, 0x7F81, 0xFFFF, 0x7FFF],
    F16: [0x7E00, 0xFE00, 0x7C01, 0xFFFF, 0x7FFF],
    F32: [0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFFFFFFF, 0x7FFFFFFF],
}


def nan_of(dtype, w):
    """The NaN with the bit pattern ``w``, as a 0-d tensor of ``dtype``."""
    bitsize = 16 if dtype in (BF16, F16) else 32
    signed = w - (1 << bitsize) if w >= 1 << (bitsize - 1) else w
    return torch.tensor([signed], dtype=INT[dtype]).view(dtype)[0]


def adversarial(n, dtype, seed=0):
    """Rows of length ``n`` that probe every clause of the rule, then random ones: a ``[R, n]`` tensor on the GPU.

    The first rows, by index: 0 all ``-inf``, 1 all ``+inf``; for each NaN pattern ``k`` the rows ``2 + 4k`` .. ``5 + 4k``: all NaN, a NaN
    last (zeros before it), a NaN first (``+inf`` after it), two NaNs at ``n // 2`` and ``n // 3`` (``-inf`` elsewhere). Then, from ``SPECIAL``:
    ``-0.0`` at 1 and ``+0.0`` at 2 (``-1`` elsewhere), ``+0.0`` at 1 and ``-0.0`` at 2 (``+1`` elsewhere), ``+inf`` last (``-inf`` elsewhere),
    ``+inf`` at 3 and ``n - 2`` (zeros), ``-inf`` at 0 and ``n - 1`` (zeros), ``1.0`` last (``-inf`` elsewhere), subnormals.
    """
    gen = torch.Generator().manual_seed(seed)
    inf, ninf = float("inf"), float("-inf")
    nans = [nan_of(dtype, w) for w in NAN_WORDS[dtype]]

    def row(fill, at=()):
        r = torch.full((n,), fill, dtype=F32).to(dtype)
        for i, v in at:
            r[i] = v
        return r

    out = [row(ninf), row(inf)]
    for k, nan in enumerate(nans):
        other = nans[(k + 1) % len(nans)]
        out += [nan.repeat(n), row(0.0, [(n - 1, nan)]), row(inf, [(0, nan)]), row(ninf, [(n // 2, nan), (n // 3, other)])]
    out += [
        row(-1.0, [(1, -0.0), (2, 0.0)]),
        row(1.0, [(1, 0.0), (2, -0.0)]),
        row(ninf, [(n - 1, inf)]),
        row(0.0, [(3, inf), (n - 2, inf)]),
        row(0.0, [(0, ninf), (n - 1, ninf)]),
        row(ninf, [(n - 1, 1.0)]),
        row(0.0, [(5 % n, 1e-40), (6 % n, -1e-40), (4 % n, 1e-39)]),
    ]
    ties = torch.randint(0, 3, (16, n), generator=gen).to(F32).to(dtype)  # many ties
    rand = torch.randn((16, n), generator=gen).to(dtype)
    sprinkled = torch.randn((8, n), generator=gen).to(dtype)
    for k in range(8):
        sprinkled[k, torch.randint(0, n, (3,), generator=gen)] = nans[k % len(nans)]
    return torch.cat([torch.stack(out), ties, rand, sprinkled]).to(CUDA)


SPECIAL = 2 + 4 * len(NAN_WORDS[BF16])  # the first row after the NaN rows


def torch_arg(x, largest, keepdim=False):
    return (x.argmax if largest else x.argmin)(dim=-1, keepdim=keepdim)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("argmax")
    return directory, runtime.export([ac.argmax_bf16, ac.argmin_bf16, ac.arg_f32, ac.arg_f16, ac.arg_ints, ac.greedy_token], directory)


def check(k, args, expected, exported):
    directory, m = exported
    jit = tilefish.run(k, *args)
    replay = tilefish.run(k, *args, mode="replay", manifest=directory, pin=m.digest)
    ref = tilefish.run(k, *args, mode="reference")
    jit, replay, ref = (r if isinstance(r, tuple) else (r,) for r in (jit, replay, ref))
    for j, p, r, e in zip(jit, replay, ref, expected):
        assert same_bytes(j, p) and same_bytes(j, r)
        assert torch.equal(j, e.to(j.dtype)), (j.flatten().tolist()[:40], e.flatten().tolist()[:40])
    return jit


def test_argmax_and_argmin_of_bf16_rows_are_torchs(exported):
    x = adversarial(5000, BF16)
    check(ac.argmax_bf16, (x,), [torch_arg(x, True)], exported)
    check(ac.argmin_bf16, (x,), [torch_arg(x, False, keepdim=True)], exported)


def test_the_rule_on_the_trap_rows():
    n = 5000
    x = adversarial(n, BF16)
    high = tilefish.run(ac.argmax_bf16, x).tolist()
    low = tilefish.run(ac.argmin_bf16, x).flatten().tolist()
    assert high[:2] == [0, 0] and low[:2] == [0, 0]  # a row of -inf, a row of +inf
    for k in range(len(NAN_WORDS[BF16])):  # a NaN wins, for argmax and argmin alike; the first one
        rows = slice(2 + 4 * k, 6 + 4 * k)
        assert high[rows] == [0, n - 1, 0, n // 3] and low[rows] == [0, n - 1, 0, n // 3], k
    s = SPECIAL
    assert high[s] == 1 and low[s + 1] == 1  # -0.0 == +0.0: the first of the two zeros, in either order
    assert high[s + 2] == n - 1 and high[s + 3] == 3 and low[s + 4] == 0 and high[s + 5] == n - 1
    assert high[s + 6] == 4  # 1e-39 is the largest of the subnormals


def test_argmax_of_f32_rows_without_masked_lanes(exported):
    x = adversarial(256, F32, seed=1)
    check(ac.arg_f32, (x,), [torch_arg(x, True), torch_arg(x, False)], exported)


@pytest.mark.parametrize("width", [7, 96, 1000])
def test_argmax_of_f16_rows_at_every_bucket(exported, width):
    x = adversarial(width, F16, seed=width)
    check(ac.arg_f16, (x,), [torch_arg(x, True), torch_arg(x, False)], exported)


@pytest.mark.parametrize("width", [7, 96, 1000])
def test_argmax_of_integer_rows(exported, width):
    gen = torch.Generator().manual_seed(width)
    a = torch.randint(-3, 3, (40, width), generator=gen, dtype=torch.int32)
    a[0] = -(2**31)
    a[1, -1] = 2**31 - 1
    b = torch.randint(-3, 3, (40, width), generator=gen, dtype=torch.int64)
    b[0] = -(2**63)
    b[1] = 2**63 - 1
    b[2, -1] = -(2**63)
    c = torch.randint(0, 256, (40, width), generator=gen, dtype=torch.uint8)
    c[0] = 0
    c[1] = 255
    c[2, -1] = 255
    a, b, c = a.to(CUDA), b.to(CUDA), c.to(CUDA)
    want = [torch_arg(t, largest) for t in (a, b, c) for largest in (True, False)]
    check(ac.arg_ints, (a, b, c), want, exported)


def test_argmax_does_not_depend_on_the_warp_count():
    x = adversarial(5000, BF16, seed=3)
    want = tilefish.run(ac.argmax_bf16, x)
    for warps in (1, 4, 8, 32):
        low = tilefish.lower(ac.argmax_bf16, num_warps=warps)
        assert torch.equal(tilefish.run(low, x), want), warps


def test_the_chunked_walk_and_one_block_agree():
    assert "static_range" in ac.argmax_bf16.lower().source and "static_range" not in ac.argmax_and_sum.lower().source
    for seed in range(3):
        x = adversarial(5000, BF16, seed=seed)
        assert torch.equal(tilefish.run(ac.argmax_and_sum, x)[0], tilefish.run(ac.argmax_bf16, x))


def test_argmax_of_a_computed_row(exported):
    logits = randn((33, 5000), BF16, seed=4)
    bias = randn((5000,), BF16, seed=5)
    want = (logits.float() + bias.float()).to(BF16).argmax(-1)
    check(ac.greedy_token, (logits, bias), [want], exported)


# -- in a launch plan ----------------------------------------------------------------------------------------------------------

Sb = Dim("Sb", buckets=[16, 48])


@tilefish.graph
def pick(logits: Tensor[Sb, ac.C, bf16], bias: Tensor[ac.C, bf16]):
    return ac.greedy_token(logits, bias), ac.argmax_bf16(logits)


def test_a_graph_of_argmax_kernels_plans_exports_and_replays(tmp_path):
    p = runtime.export(pick, tmp_path)
    for rows in (16, 48):
        logits = adversarial(5000, BF16, seed=rows)[:rows].contiguous()
        bias = randn((5000,), BF16, seed=6)
        jit = tilefish.run(pick, logits, bias)
        replay = tilefish.run(pick, logits, bias, mode="replay", plan=tmp_path, pin=p.digest)
        ref = tilefish.run(pick, logits, bias, mode="reference")
        for a, b, c in zip(jit, replay, ref):
            assert same_bytes(a, b) and same_bytes(a, c)
        assert torch.equal(jit[1], torch_arg(logits, True)) and jit[0].dtype == torch.int64
