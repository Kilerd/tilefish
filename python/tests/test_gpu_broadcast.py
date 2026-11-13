"""An operand broadcast over the leading axes in the elementwise template, on a real GPU, byte for byte.

``pos_add`` adds a ``[T, D]`` table to every ``[T, D]`` slice of a ``[K, T, D]`` batch in place (one f32 add, one rounding): over every one of the
2**32 pairs of bf16 bit patterns it matches torch and the reference. ``scale_cols`` (a row length the row template cannot take) matches the
reference; ``jit``, ``replay`` and ``reference`` agree, alone and in a plan.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import broadcast_cases as bc  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, InOut, Tensor, bf16, runtime  # noqa: E402

BF16 = torch.bfloat16
CUDA = "cuda"


def patterns():
    return torch.arange(-32768, 32768, dtype=torch.int32, device=CUDA).to(torch.int16).view(BF16)


def test_pos_add_over_every_pair_of_bf16_patterns():
    words = patterns()
    pe = words.reshape(1, 65536)  # T = 1, D = 65536: every pattern as the table
    for k0 in range(0, 65536, 4096):
        y = words[k0 : k0 + 4096].reshape(4096, 1, 1).expand(4096, 1, 65536).contiguous()  # row k: pattern k0 + k everywhere
        want = (y.float() + pe.float()).to(BF16)
        ref = y.clone()
        tilefish.run(bc.pos_add, ref, pe, mode="reference")
        tilefish.run(bc.pos_add, y, pe)
        assert same_bytes(y, want) and same_bytes(y, ref), k0


def test_jit_replay_and_reference_agree(tmp_path):
    m = runtime.export([bc.pos_add, bc.scale_cols], tmp_path)
    y = patterns()[:7 * 33 * 160].reshape(7, 33, 160).flip(0).contiguous()
    pe = randn((33, 160), BF16, seed=1)
    outs = []
    for kw in ({}, {"mode": "replay", "manifest": tmp_path, "pin": m.digest}, {"mode": "reference"}):
        t = y.clone()
        tilefish.run(bc.pos_add, t, pe, **kw)
        outs.append(t)
    assert same_bytes(outs[0], outs[1]) and same_bytes(outs[0], outs[2])
    x, w, b = randn((300, 77), torch.float32, seed=2), randn((77,), torch.float32, seed=3), randn((77,), torch.float32, seed=4)
    jit = tilefish.run(bc.scale_cols, x, w, b)
    assert same_bytes(jit, tilefish.run(bc.scale_cols, x, w, b, mode="replay", manifest=tmp_path, pin=m.digest))
    assert (jit - (x * w + b)).abs().max() <= 2e-7 * (x * w).abs().max() + 1e-6  # a * b + c may round once (an FMA)


Kb = Dim("Kb", buckets=[3, 9])
Tb = Dim("Tb", buckets=[50])


@tilefish.graph
def add_positions(y: InOut[Tensor[Kb, Tb, 1024, bf16]], pe: Tensor[Tb, 1024, bf16]):
    bc.pos_add(y, pe)


def test_in_a_plan(tmp_path):
    p = runtime.export(add_positions, tmp_path)
    pe = randn((50, 1024), BF16, seed=5)
    for k in (3, 9):
        y = randn((k, 50, 1024), BF16, seed=k)
        results = []
        for kw in ({}, {"mode": "replay", "plan": tmp_path, "pin": p.digest}, {"mode": "reference"}):
            t = y.clone()
            assert tilefish.run(add_positions, t, pe, **kw) is None
            results.append(t)
        assert same_bytes(results[0], results[1]) and same_bytes(results[0], results[2])
        assert same_bytes(results[0], (y.float() + pe.float()).to(BF16))
