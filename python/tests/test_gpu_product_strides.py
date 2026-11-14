"""Strides that are products of dimensions, and broadcasts along any axis, on a real GPU, byte for byte.

The kernels of ``broadcast_cases.py`` read an operand broadcast along an axis that is not a leading one (``[H, 1, L]`` over ``[H, L, L]``, ``[K, T, 1]``,
``[K, 1, D]``, ``[T, 1]``, both operands of an outer product) and count coordinates with ``iota`` along any axis (a causal mask): each is one select,
multiply or add per element, so ``jit``, ``replay``, ``reference`` and torch's own broadcasting give the same bytes, NaN, infinities and signed zeros
included. The plan ``gpu_kernels_p7.attention_mask`` has ``[H, Lb, Lb]`` buffers (the stride ``Lb*Lb``), a transpose of one handed to the generated op
``tilefish.contiguous`` with the strides ``(Lb*Lb, 1, Lb)``, a head at the offset ``Lb*Lb`` copied out and a row kernel over the reshape ``[H * Lb, Lb]``:
plan ``jit``, plan ``replay`` and the composition of the references are torch's bytes at every bucket.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import broadcast_cases as bc  # noqa: E402
import gpu_kernels_p7 as p7  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import runtime  # noqa: E402

F32, BF16, U8 = torch.float32, torch.bfloat16, torch.uint8
CUDA = "cuda"
SPECIALS = (float("nan"), float("inf"), float("-inf"), -0.0, 0.0, 1e-40, -1e-40)
PLAIN = ()  # the plan's row maximum: tilefish.max skips a NaN where torch's amax returns it, and the sign of a maximal zero is the tree's


def values(shape, dtype=F32, seed=0, specials=SPECIALS):
    """Normal values with the special ones sprinkled in (NaN, infinities, signed zeros, subnormals)."""
    t = randn(shape, F32, seed=seed)
    if not specials:
        return t.to(dtype)
    flat = t.view(-1)
    gen = torch.Generator(device=CUDA)
    gen.manual_seed(seed + 1000)
    where = torch.randint(0, flat.numel(), (max(1, flat.numel() // 16),), generator=gen, device=CUDA)
    flat[where] = torch.tensor(specials, device=CUDA)[torch.arange(where.numel(), device=CUDA) % len(specials)]
    return t.to(dtype)


def mask(shape, seed=0):
    gen = torch.Generator(device=CUDA)
    gen.manual_seed(seed)
    return torch.randint(0, 3, shape, generator=gen, device=CUDA, dtype=U8)  # 0, 1 and 2: 2 is a kept key too (!= 0)


def causal_torch(s):
    n = s.shape[-1]
    i = torch.arange(n, device=CUDA)
    return torch.where(i[None, :] <= i[:, None], s, float("-inf"))


CASES = {
    "key_mask_heads": (bc.key_mask_heads, lambda L, seed: (values((4, L, L), seed=seed), mask((4, 1, L), seed)), lambda s, k: torch.where(k != 0, s, float("-inf"))),
    "causal_heads": (bc.causal_heads, lambda L, seed: (values((4, L, L), seed=seed),), causal_torch),
    "scale_each_row": (bc.scale_each_row, lambda L, seed: (values((3, L, 5), seed=seed), values((3, L, 1), seed=seed + 1)), lambda x, r: x * r),
    "middle_rows": (
        bc.middle_rows,
        lambda L, seed: (values((2, L, 7), BF16, seed), values((2, 1, 7), BF16, seed + 1)),
        lambda x, z: (x.float() * z.float()).to(BF16),
    ),
    "column_add": (bc.column_add, lambda L, seed: (values((3, L, 6), seed=seed), values((L, 1), seed=seed + 1)), lambda x, c: x + c),
    "outer_heads": (bc.outer_heads, lambda L, seed: (values((4, L, 1), seed=seed), values((4, 1, L), seed=seed + 1)), lambda q, k: q * k),
}
LENGTHS = (1, 2, 7, 33, 128, 515)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("broadcast_any_axis")
    return directory, runtime.export([op for op, _, _ in CASES.values()], directory)


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_broadcast_along_any_axis_is_torchs_bytes_in_every_mode(exported, name):
    directory, manifest = exported
    op, make, want = CASES[name]
    for length in LENGTHS:
        for seed in (0, 1):
            args = make(length, seed)
            jit = tilefish.run(op, *args)
            assert same_bytes(jit, want(*args)), (name, length, seed)
            assert same_bytes(jit, tilefish.run(op, *args, mode="replay", manifest=directory, pin=manifest.digest)), (name, length, seed)
            assert same_bytes(jit, tilefish.run(op, *args, mode="reference")), (name, length, seed)


def test_one_binary_serves_every_length():
    """The sizes the coordinates need are runtime scalars of the dimensions: one build, any length, the bytes of each."""
    for length in (3, 64, 1000):
        s, keep = values((4, length, length), seed=length), mask((4, 1, length), length)
        assert same_bytes(tilefish.run(bc.key_mask_heads, s, keep), torch.where(keep != 0, s, float("-inf")))


def test_a_wrong_offset_would_be_seen():
    """The cases can tell a broadcast along the middle axis from one along the leading axes: reading the mask as if it were [L] (lane o at o % L,
    every head the mask of head 0) gives other bytes."""
    s, keep = values((4, 33, 33), seed=5), mask((4, 1, 33), 5)
    wrong = torch.where(keep[0:1].expand(4, 1, 33) != 0, s, float("-inf"))
    assert not same_bytes(tilefish.run(bc.key_mask_heads, s, keep), wrong)


@pytest.fixture(scope="module")
def plan_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("attention_mask_plan")
    return directory, runtime.export(p7.attention_mask, directory)


@pytest.mark.parametrize("lb", [8, 24])
def test_a_plan_over_buffers_whose_strides_are_products(plan_dir, lb):
    directory, exported = plan_dir
    for seed in (0, 1, 2):
        q, k = values((4, lb, 1), seed=seed, specials=PLAIN), values((4, 1, lb), seed=seed + 10, specials=PLAIN)
        keep = mask((4, 1, lb), seed)
        want = p7.attention_mask_torch(q, k, keep)
        outs = []
        for kw in ({}, {"mode": "replay", "plan": directory, "pin": exported.digest}, {"mode": "reference"}):
            got = tilefish.run(p7.attention_mask, q, k, keep, **kw)
            assert len(got) == 4
            outs.append(got)
        for i, w in enumerate(want):
            for mode, got in zip(("jit", "replay", "reference"), outs):
                assert same_bytes(got[i], w), (lb, seed, mode, i)
