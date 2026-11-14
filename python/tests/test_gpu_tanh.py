"""``tanh`` on a real GPU: libdevice's ``tanhf`` built with flush to zero, against torch and against the reference (note 14 of the contract).

Over every one of the 2**32 f32 bit patterns the binary is torch's CUDA ``tanh`` except on the 16 777 214 subnormals, which it gives as a zero of
their sign; in IEEE math (``math="ieee"``) the subnormal is passed through and the binary is torch's on every input. The reference models both,
byte for byte. A bias and a tanh of a bf16 activation (the row template) match torch, alone and in a plan.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import tanh_cases as tc  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, bf16, runtime  # noqa: E402

F32, I32, BF16 = torch.float32, torch.int32, torch.bfloat16
CUDA = "cuda"
CHUNK = 1 << 26


def chunks():
    for start in range(0, 1 << 32, CHUNK):
        yield torch.arange(start, start + CHUNK, dtype=torch.int64, device=CUDA).to(I32).view(F32)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tanh")
    return directory, runtime.export([tc.tanh_fast, tc.tanh_ieee, tc.tanh_rows], directory)


def test_tanh_is_torchs_but_on_a_subnormal_and_in_ieee_math_on_every_f32():
    fast_differs = ieee_differs = 0
    for x in chunks():
        want = torch.tanh(x).view(I32)
        fast = tilefish.run(tc.tanh_fast, x)
        subnormal = (x != 0) & (x.abs() < 2.0**-126)
        differs = fast.view(I32) != want
        assert torch.equal(differs, subnormal)  # torch's bits on every other input (NaN payloads included) ...
        assert torch.equal(fast[subnormal].view(I32), (x[subnormal] * 0.0).view(I32))  # ... and a zero of the argument's sign on a subnormal
        fast_differs += int(differs.sum())
        ieee_differs += int((tilefish.run(tc.tanh_ieee, x).view(I32) != want).sum())
    print(f"[tanh] vs torch over 2**32 f32: libdevice (FTZ) differs on {fast_differs}, IEEE math on {ieee_differs}")
    assert fast_differs == 2 * ((1 << 23) - 1) and ieee_differs == 0


def f32_patterns(count=1 << 22, seed=0):
    gen = torch.Generator(device=CUDA)
    gen.manual_seed(seed)
    words = torch.randint(-(2**31), 2**31, (count,), generator=gen, device=CUDA, dtype=torch.int64).to(I32)
    words[:6] = torch.tensor([0, -(2**31), 1, -(2**31) + 1, 0x007FFFFF, 0x7FC00000], device=CUDA, dtype=torch.int64).to(I32)
    return words.view(F32)


def test_jit_replay_and_reference_agree(exported):
    directory, m = exported
    x = f32_patterns()
    for op in (tc.tanh_fast, tc.tanh_ieee):
        jit = tilefish.run(op, x)
        assert same_bytes(jit, tilefish.run(op, x, mode="replay", manifest=directory, pin=m.digest)), op.name
        assert same_bytes(jit, tilefish.run(op, x, mode="reference")), op.name


def test_a_bias_and_tanh_of_bf16_rows(exported):
    directory, m = exported
    for rows in (1, 7, 300):
        x, b = randn((rows, 64), BF16, seed=rows, scale=3.0), randn((64,), BF16, seed=rows + 1)
        got = tilefish.run(tc.tanh_rows, x, b)
        assert same_bytes(got, torch.tanh(x.float() + b.float()).to(BF16))
        assert same_bytes(got, tilefish.run(tc.tanh_rows, x, b, mode="replay", manifest=directory, pin=m.digest))
        assert same_bytes(got, tilefish.run(tc.tanh_rows, x, b, mode="reference"))


Rb = Dim("Rb", buckets=[16, 48])


@tilefish.graph
def head(x: Tensor[Rb, 64, bf16], b: Tensor[64, bf16]):
    return tc.tanh_rows(x, b)


def test_in_a_plan(tmp_path):
    p = runtime.export(head, tmp_path)
    b = randn((64,), BF16, seed=9)
    for rows in (16, 48):
        x = randn((rows, 64), BF16, seed=rows, scale=3.0)
        jit = tilefish.run(head, x, b)
        assert same_bytes(jit, tilefish.run(head, x, b, mode="replay", plan=tmp_path, pin=p.digest))
        assert same_bytes(jit, torch.tanh(x.float() + b.float()).to(BF16))
