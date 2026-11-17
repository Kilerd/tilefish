"""IEEE math (``math="ieee"``, note 13 of the numerics contract) on a real GPU, against torch's CUDA kernels, byte for byte.

* Over **every one of the 2**32 f32 bit patterns**: ``exp`` is torch's ``exp`` (CUDA's ``expf``), ``sqrt`` torch's ``sqrt``, ``rsqrt`` torch's ``rsqrt``,
  ``sin`` torch's ``sin`` and ``silu`` ``torch.nn.functional.silu``, NaNs, infinities, zeros and subnormals included; the default math is not
  (``exp`` differs on most inputs, ``sqrt`` and ``sin`` on the subnormals: the traps that show the comparison can see a difference).
* ``/`` is torch's division of tensors on random bit-pattern pairs, every subnormal numerator, every subnormal divisor and the special values; a
  constant divisor is divided by too (``x / 3.0`` is ``torch.div(x, tensor(3.0))``, where torch's ``x / 3.0`` multiplies by ``1 / 3``).
* An op may say its own math in a kernel of the default math; a periodic activation in IEEE math without contraction is torch's eager
  composition of the same ops; ``jit``, ``replay`` and ``reference`` agree, alone and in a plan.
"""

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.cuda_math]  # the exhaustive comparisons with CUDA's math library: docs/en/numerics.md

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import ieee_cases as ic  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, f32, runtime  # noqa: E402

F32, I32 = torch.float32, torch.int32
CUDA = "cuda"
CHUNK = 1 << 26  # f32 bit patterns per launch: 2**32 in 64 chunks
SUBNORMAL = 2.0**-126


def chunks():
    """Every f32 bit pattern, as 64 tensors of 2**26 (ascending as unsigned words)."""
    for start in range(0, 1 << 32, CHUNK):
        words = torch.arange(start, start + CHUNK, dtype=torch.int64, device=CUDA)
        yield words.to(I32).view(F32)  # the low 32 bits: the pattern


def mismatches(a, b):
    return int((a.view(I32) != b.view(I32)).sum())


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ieee")
    ops = [ic.div_ieee, ic.div_by_three_ieee, ic.math_ieee, ic.per_op, ic.periodic, ic.softmax_ieee]
    return directory, runtime.export(ops, directory)


def test_exp_sqrt_rsqrt_sin_and_silu_are_torchs_on_every_f32():
    """Every f32 bit pattern: IEEE math is torch's bits; the default math's exp, sqrt and sin are not (the traps)."""
    fast_differs = {name: 0 for name in ("exp", "sqrt", "rsqrt", "sin", "silu")}
    fast_off_subnormal = {name: 0 for name in fast_differs}
    for x in chunks():
        want = (torch.exp(x), torch.sqrt(x), torch.rsqrt(x), torch.sin(x), torch.nn.functional.silu(x))
        got = tilefish.run(ic.math_ieee, x)
        for name, g, w in zip(fast_differs, got, want):
            assert mismatches(g, w) == 0, (name, float(x[0]))
        subnormal = (x != 0) & (x.abs() < SUBNORMAL)
        for name, g, w in zip(fast_differs, tilefish.run(ic.math_fast, x), want):
            differs = g.view(I32) != w.view(I32)
            fast_differs[name] += int(differs.sum())
            fast_off_subnormal[name] += int((differs & ~subnormal).sum())
        del want, got
    print(f"[ieee] default math vs torch on 2**32 f32: differing {fast_differs}, of them not on a subnormal argument {fast_off_subnormal}")
    # measured (B300, torch 2.9.1 with CUDA 12.8 and 13.0): exp 129 346 491 inputs, silu 146 046 470, the others every subnormal and only those
    assert fast_differs["exp"] > 10**8 and fast_differs["silu"] > 10**8  # ex2.approx of x * log2(e); x * sigmoid(x)
    for name in ("sqrt", "rsqrt", "sin"):  # flush to zero: the 2 * (2**23 - 1) subnormal arguments, and nothing else
        assert fast_differs[name] == 2 * ((1 << 23) - 1) and fast_off_subnormal[name] == 0, name


def f32_pairs(count, seed):
    gen = torch.Generator(device=CUDA)
    gen.manual_seed(seed)
    words = torch.randint(-(2**31), 2**31, (2, count), generator=gen, device=CUDA, dtype=torch.int64).to(I32)
    return words[0].view(F32), words[1].view(F32)


SPECIALS = [0.0, -0.0, 1e-45, -1e-45, 1e-40, SUBNORMAL, 1.0, -1.0, 3.0, 2.0**127, 3.4028234663852886e38, float("inf"), float("-inf"), float("nan")]


def test_division_is_torchs_division_of_tensors(exported):
    directory, m = exported
    cases = [f32_pairs(1 << 26, seed) for seed in (0, 1, 2)]
    sub = torch.arange(1, 1 << 23, dtype=I32, device=CUDA).view(F32)  # every positive subnormal
    gen = torch.Generator(device=CUDA)
    gen.manual_seed(6)
    scales = torch.tensor(2.0, device=CUDA) ** torch.randint(-60, 60, (sub.numel(),), generator=gen, device=CUDA).float()
    other = randn((sub.numel(),), F32, seed=5) * scales  # normal partners over 120 binades
    cases += [(sub, other), (other, sub), (-sub, other * 1e-30), (other * 1e30, sub)]
    s = torch.tensor(SPECIALS, device=CUDA)
    cases.append((s.repeat_interleave(len(SPECIALS)), s.repeat(len(SPECIALS))))
    fast = 0
    for x, y in cases:
        want = x / y
        got = tilefish.run(ic.div_ieee, x, y)
        assert mismatches(got, want) == 0
        assert same_bytes(got, tilefish.run(ic.div_ieee, x, y, mode="replay", manifest=directory, pin=m.digest))
        assert same_bytes(got, tilefish.run(ic.div_ieee, x, y, mode="reference"))
        fast += mismatches(tilefish.run(ic.per_op, x, y)[2], want)  # the default math's x / y: div.full
    print(f"[ieee] the default math's division differs from torch's in {fast} of {sum(x.numel() for x, _ in cases)} pairs")
    assert fast > 0


def test_a_constant_divisor_is_divided_by():
    """``x / 3.0`` in IEEE math is the division; torch's ``x / 3.0`` (a python scalar) multiplies by ``1 / 3``, its ``x / tensor(3.0)`` divides."""
    x = f32_pairs(1 << 24, 7)[0]
    got = tilefish.run(ic.div_by_three_ieee, x)
    assert mismatches(got, torch.div(x, torch.tensor(3.0, device=CUDA))) == 0
    assert mismatches(got, x / 3.0) > 0  # torch's scalar path rounds twice
    assert same_bytes(got, tilefish.run(ic.div_by_three_ieee, x, mode="reference"))


def test_an_op_says_its_own_math(exported):
    directory, m = exported
    x, y = randn((1 << 20,), F32, seed=1, scale=30.0), randn((1 << 20,), F32, seed=2)
    d, e, fd, fe = tilefish.run(ic.per_op, x, y)
    assert same_bytes(d, x / y) and same_bytes(e, torch.exp(x))
    assert mismatches(fd, x / y) > 0 and mismatches(fe, torch.exp(x)) > 0  # the default math, in the same kernel
    for got, want in zip(tilefish.run(ic.per_op, x, y, mode="replay", manifest=directory, pin=m.digest), (d, e, fd, fe)):
        assert same_bytes(got, want)
    rd, re, _, _ = tilefish.run(ic.per_op, x, y, mode="reference")
    assert same_bytes(rd, d) and same_bytes(re, e)


@pytest.mark.parametrize("cols", [96, 1000])
def test_a_periodic_activation_is_torchs_eager_composition(exported, cols):
    directory, m = exported
    x = randn((300, cols), F32, seed=cols, scale=4.0)
    a = randn((cols,), F32, seed=cols + 1).abs() + 0.05
    a[:3] = torch.tensor([0.0, 1e-40, -1e-9], device=CUDA)  # a + 1e-9 of 0, a subnormal, and 0 exactly: 1 / 1e-9 and 1 / 0
    want = x + (1.0 / (a + 1e-9)) * torch.sin(a * x) ** 2
    got = tilefish.run(ic.periodic, x, a)
    assert same_bytes(got, want)
    assert same_bytes(got, tilefish.run(ic.periodic, x, a, mode="replay", manifest=directory, pin=m.digest))
    assert same_bytes(got, tilefish.run(ic.periodic, x, a, mode="reference"))


@pytest.mark.parametrize("cols", [96, 1000])
def test_a_softmax_in_ieee_math(exported, cols):
    directory, m = exported
    x = randn((257, cols), F32, seed=cols, scale=8.0)
    got = tilefish.run(ic.softmax_ieee, x)
    assert same_bytes(got, tilefish.run(ic.softmax_ieee, x, mode="replay", manifest=directory, pin=m.digest))
    # the exponentials and the division are torch's operations; the sum's tree is Triton's (a few ULP of values up to 1, as the f32 softmax of
    # test_gpu_reference.py)
    e = torch.exp(x - x.amax(dim=-1, keepdim=True))
    assert (got - torch.softmax(x, dim=-1)).abs().max() <= 5e-7 and (got - e / e.sum(dim=-1, keepdim=True)).abs().max() <= 5e-7
    assert (got - tilefish.run(ic.softmax_ieee, x, mode="reference")).abs().max() <= 5e-7


Rb = Dim("Rb", buckets=[8, 24])


@tilefish.graph
def activate(x: Tensor[Rb, 1000, f32], a: Tensor[1000, f32]):
    return ic.periodic(ic.softmax_ieee(x), a)


def test_in_a_plan(tmp_path):
    p = runtime.export(activate, tmp_path)
    a = randn((1000,), F32, seed=3).abs() + 0.1
    for rows in (8, 24):
        x = randn((rows, 1000), F32, seed=rows)
        outs = [tilefish.run(activate, x, a, **kw) for kw in ({}, {"mode": "replay", "plan": tmp_path, "pin": p.digest})]
        assert same_bytes(outs[0], outs[1])
        s = tilefish.run(ic.softmax_ieee, x)
        assert same_bytes(outs[0], s + (1.0 / (a + 1e-9)) * torch.sin(a * s) ** 2)
