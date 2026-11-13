"""Comparisons, masks, ``where`` and ``iota`` on a real GPU: the binaries against torch, byte for byte, on random data and on traps.

Every kernel of ``mask_cases`` runs in ``jit``, ``replay`` (an exported manifest) and ``reference`` (torch), and the three agree byte for byte on
every bf16 bit pattern, on f32 traps (signed zeros, infinities, NaNs of both signs and several payloads, subnormals) and on random data. The
claims of note 7 of the numerics contract are checked against torch directly: IEEE comparisons, a scalar rounded to the tensor's dtype before
it is compared, ``where`` passing the chosen bits through, a mask cast to exactly 0 or 1. A graph of mask kernels plans, exports, replays.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import mask_cases as mc  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import DevScalar, Dim, Tensor, bf16, i32, runtime  # noqa: E402

BF16, F32, F16 = torch.bfloat16, torch.float32, torch.float16
CUDA = "cuda"

F32_TRAPS = [
    0x00000000, 0x80000000,  # +0, -0
    0x3F800000, 0xBF800000, 0x3DCCCCCD, 0x3DCD0000,  # 1, -1, f32(0.1), bf16(0.1)
    0x7F800000, 0xFF800000,  # +inf, -inf
    0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FFFFFFF, 0xFFC00001, 0x7FA00000,  # NaNs: quiet, negative, signalling payloads
    0x00000001, 0x80000001, 0x007FFFFF, 0x00800000,  # subnormals, the smallest normal
    0x7F7FFFFF, 0xFF7FFFFF,  # the largest finite values
]  # fmt: skip


def f32_bits(words):
    signed = [w - (1 << 32) if w >= 1 << 31 else w for w in words]
    return torch.tensor(signed, dtype=torch.int32).view(F32).to(CUDA)


def every_bf16(rows_of=None):
    """All 65536 bf16 bit patterns, flat or as rows of ``rows_of`` (padded with zeros)."""
    words = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(BF16)
    if rows_of is None:
        return words.to(CUDA)
    pad = (-words.numel()) % rows_of
    return torch.cat([words, torch.zeros(pad, dtype=BF16)]).reshape(-1, rows_of).to(CUDA)


def every_f16():
    return torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(F16).to(CUDA)


def trap_pairs():
    t = f32_bits(F32_TRAPS)
    a = t.repeat_interleave(t.numel())
    b = t.repeat(t.numel())
    r1, r2 = randn((4096,), F32, seed=11), randn((4096,), F32, seed=12)
    r2[::3] = r1[::3]  # some equal pairs
    return torch.cat([a, r1]), torch.cat([b, r2])


def n_tok(v):
    return torch.tensor([v], dtype=torch.int32, device=CUDA)


def as_tuple(r):
    return r if isinstance(r, tuple) else (r,)


def all_same(a, b):
    a, b = as_tuple(a), as_tuple(b)
    return len(a) == len(b) and all(same_bytes(x, y) for x, y in zip(a, b))


KERNELS = [mc.mask_rows, mc.zero_rows, mc.key_mask, mc.compare_f32, mc.compare_scalar_bf16, mc.compare_scalar_f16, mc.select, mc.logic, mc.int_compare, mc.positions, mc.flat_positions]


def inputs_of(k):
    """Trap and random inputs of each kernel: a list of argument tuples."""
    if k in (mc.mask_rows, mc.zero_rows):
        x = every_bf16(96)  # 683 rows: every bf16 pattern goes through both arms for some n
        return [(x, n_tok(n)) for n in (0, 1, 300, x.shape[0] - 1, x.shape[0], x.shape[0] + 5, -3)]
    if k is mc.key_mask:
        keys = torch.zeros(1280, dtype=BF16, device=CUDA)
        return [(keys, n_tok(n)) for n in (0, 1, 17, 1279, 1280)]
    if k in (mc.compare_f32, mc.logic):
        return [trap_pairs()]
    if k is mc.compare_scalar_bf16:
        return [(every_bf16(),)]
    if k is mc.compare_scalar_f16:
        return [(every_f16(),)]
    if k is mc.select:
        a, b = every_bf16(), every_bf16().flip(0)
        m = (torch.arange(a.numel(), device=CUDA) % 3 == 0).to(torch.uint8) * torch.randint(1, 256, (a.numel(),), device=CUDA, dtype=torch.uint8)
        return [(m, a, b)]
    if k is mc.int_compare:
        x = torch.tensor([0, -1, 7, 2999999999, 3000000000, 3000000001, -(2**63), 2**63 - 1], dtype=torch.int64, device=CUDA)
        y = torch.tensor([0, 199, 200, 201, 255, 128, 1, 254], dtype=torch.uint8, device=CUDA)
        return [(x, y)]
    if k is mc.positions:
        return [(randn((37, 96), F32, seed=3),)]
    if k is mc.flat_positions:
        return [(randn((5000,), F32, seed=4),)]
    raise AssertionError(k)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("masks")
    return directory, runtime.export(KERNELS, directory)


@pytest.mark.parametrize("k", KERNELS, ids=lambda k: k.name)
def test_jit_replay_and_reference_are_byte_identical(exported, k):
    directory, m = exported
    for args in inputs_of(k):
        jit = tilefish.run(k, *args)
        replay = tilefish.run(k, *args, mode="replay", manifest=directory, pin=m.digest)
        ref = tilefish.run(k, *args, mode="reference")
        assert all_same(jit, replay), k.name
        assert all_same(jit, ref), k.name


def test_comparisons_are_ieee_and_agree_with_torch():
    a, b = trap_pairs()
    got = tilefish.run(mc.compare_f32, a, b)
    for out, fn in zip(got, (torch.eq, torch.ne, torch.lt, torch.le, torch.gt, torch.ge)):
        assert torch.equal(out, fn(a, b).to(torch.uint8)), fn.__name__
    nan = torch.isnan(a) | torch.isnan(b)
    assert (got[1][nan] == 1).all() and all((o[nan] == 0).all() for i, o in enumerate(got) if i != 1)  # a NaN is unordered: only != holds
    zeros = f32_bits([0x00000000, 0x80000000])
    eq, ne, lt, *_ = tilefish.run(mc.compare_f32, zeros, zeros.flip(0))
    assert eq.tolist() == [1, 1] and ne.tolist() == [0, 0] and lt.tolist() == [0, 0]  # -0.0 == +0.0


def test_a_scalar_is_rounded_to_the_tensor_dtype_before_it_is_compared():
    x = every_bf16()
    gt, le, eq, sel = tilefish.run(mc.compare_scalar_bf16, x)
    c = torch.tensor(0.1, dtype=BF16, device=CUDA)  # 0.10009765625
    assert torch.equal(gt, (x > c).to(torch.uint8)) and torch.equal(le, (x <= c).to(torch.uint8)) and torch.equal(eq, (x == c).to(torch.uint8))
    at = (x.view(torch.int16) == 0x3DCD).nonzero().item()
    assert gt[at] == 0 and eq[at] == 1  # bf16(0.1) is not > 0.1 once 0.1 is rounded to bf16 ...
    assert bool(x[at].float() > 0.1)  # ... though it is > the f32 0.1 that a hand-written Triton comparison would use
    keep = x >= -2.5
    assert same_bytes(sel, torch.where(keep, x, c))
    lt16, sel16 = tilefish.run(mc.compare_scalar_f16, every_f16())
    y = every_f16()
    assert torch.equal(lt16, (y < torch.tensor(0.1, dtype=F16, device=CUDA)).to(torch.uint8))
    assert (sel16[torch.isnan(y)].view(torch.int16) == 0x7BFF).all()  # 65519 rounded to f16 is 65504


def test_where_passes_the_chosen_bits_through():
    (m, a, b), = inputs_of(mc.select)
    got = tilefish.run(mc.select, m, a, b)
    want = torch.where(m != 0, a.view(torch.int16), b.view(torch.int16))
    assert torch.equal(got.view(torch.int16), want)  # every NaN payload and both zeros, as they were


def test_masks_cast_to_exactly_zero_or_one_and_combine_logically():
    a, b = trap_pairs()
    land, lor, lxor, lnot = tilefish.run(mc.logic, a, b)
    p, q = a > 0, b > 0
    assert torch.equal(land, (p & q).to(torch.uint8)) and torch.equal(lor, (p | q).to(torch.uint8)) and torch.equal(lxor, (p ^ q).to(torch.uint8))
    assert torch.equal(lnot.view(torch.int32), (~p).to(F32).view(torch.int32))
    assert set(lnot.unique().tolist()) <= {0.0, 1.0}


def test_mask_rows_multiplies_by_zero_and_zero_rows_writes_positive_zero():
    x = every_bf16(96)
    for n in (0, 200, x.shape[0]):
        got = tilefish.run(mc.mask_rows, x, n_tok(n))
        assert torch.equal(got[:n].view(torch.int16), x[:n].view(torch.int16))  # a raw copy below n
        tail = (x[n:].float() * 0.0).to(BF16)
        assert torch.equal(got[n:].view(torch.int16), tail.view(torch.int16))
        zero = tilefish.run(mc.zero_rows, x, n_tok(n))
        assert (zero[n:].view(torch.int16) == 0).all()
    got = tilefish.run(mc.mask_rows, x, n_tok(0)).view(torch.int16).flatten()[:65536]
    words = x.view(torch.int16).flatten()[:65536]
    assert (got[words == -16448] == -32768).all()  # -1.5 (0xBFC0) times 0 is -0.0 (0x8000): a multiply, not a memset
    inf_or_nan = ((words & 0x7F80) == 0x7F80)
    assert (got[inf_or_nan] == 0x7FFF).all()  # inf * 0 and NaN * 0 are the canonical NaN


def test_key_mask_is_zero_then_the_most_negative_finite_bf16():
    keys = torch.zeros(1280, dtype=BF16, device=CUDA)
    for n in (0, 1, 640, 1280):
        got = tilefish.run(mc.key_mask, keys, n_tok(n)).view(torch.int16)
        assert (got[:n] == 0).all() and (got[n:] == -129).all()  # 0xFF7F


def test_iota_is_the_row_and_the_column():
    x = randn((37, 96), F32, seed=3)
    rc, r = tilefish.run(mc.positions, x)
    rows = torch.arange(37, device=CUDA).reshape(37, 1)
    assert torch.equal(rc, (rows * 1000 + torch.arange(96, device=CUDA)).to(torch.int32))
    assert torch.equal(r, rows.to(torch.int64))
    assert torch.equal(tilefish.run(mc.flat_positions, randn((5000,), F32)), torch.arange(5000, device=CUDA))


# -- in a launch plan ----------------------------------------------------------------------------------------------------------

Rb = Dim("Rb", buckets=[64, 128])


@tilefish.graph
def masked_block(x: Tensor[Rb, mc.D, bf16], n: DevScalar[i32]):
    y = mc.mask_rows(x, n)
    return mc.zero_rows(y, n), mc.mask_rows(x, n)


def test_a_graph_of_mask_kernels_plans_exports_and_replays_byte_identically(tmp_path):
    p = runtime.export(masked_block, tmp_path)
    for rows, n in ((64, 0), (64, 31), (128, 128), (128, 77)):
        x = every_bf16(96)[:rows].contiguous()
        jit = tilefish.run(masked_block, x, n_tok(n))
        replay = tilefish.run(masked_block, x, n_tok(n), mode="replay", plan=tmp_path, pin=p.digest)
        ref = tilefish.run(masked_block, x, n, mode="reference")
        eager = tilefish.run(mc.zero_rows, tilefish.run(mc.mask_rows, x, n_tok(n)), n_tok(n))
        assert all_same(jit, replay) and all_same(jit, ref) and same_bytes(jit[0], eager)
