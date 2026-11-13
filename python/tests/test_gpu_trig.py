"""``sin`` and ``cos`` on a real GPU: libdevice's ``sinf`` / ``cosf`` with Triton's flush to zero, against torch and against the reference.

On random f32 bit patterns (every exponent, NaNs, infinities, subnormals) the binary is byte-identical to the reference, and to torch's CUDA
``sin`` / ``cos`` except ``sin`` of a subnormal, which the binary flushes to a zero of the same sign (note 10 of the numerics contract).
A rotary table (an angle per position and frequency, its cosine and sine in bf16) matches torch byte for byte, and plans.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import trig_cases as tc  # noqa: E402
from gpu_util import same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, f32, i32, runtime  # noqa: E402

F32, BF16 = torch.float32, torch.bfloat16
CUDA = "cuda"


def f32_patterns(count=1 << 22, seed=0):
    gen = torch.Generator().manual_seed(seed)
    words = torch.randint(-(2**31), 2**31, (count,), generator=gen, dtype=torch.int64).to(torch.int32)
    specials = torch.tensor(
        [0, -(2**31), 1, -(2**31) + 1, 0x007FFFFF, 0x00800000, 0x7F800000, -8388608, 0x7FC00000, -4194304, 0x7F800001, 0x4B000000, 0x7F7FFFFF],
        dtype=torch.int64,
    ).to(torch.int32)  # ±0, ±smallest subnormal, largest subnormal, smallest normal, ±inf, NaNs, 2**23, the largest finite
    return torch.cat([specials, words]).view(F32).to(CUDA)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("trig")
    return directory, runtime.export([tc.sin_cos, tc.rope_tables], directory)


def test_sin_and_cos_are_libdevice_with_ftz_byte_for_byte(exported):
    directory, m = exported
    x = f32_patterns()
    s, c = tilefish.run(tc.sin_cos, x)
    rs, rc = tilefish.run(tc.sin_cos, x, mode="reference")
    ps, pc = tilefish.run(tc.sin_cos, x, mode="replay", manifest=directory, pin=m.digest)
    assert same_bytes(s, rs) and same_bytes(c, rc) and same_bytes(s, ps) and same_bytes(c, pc)
    assert same_bytes(c, torch.cos(x))  # cos: torch's bits on every input
    subnormal = (x != 0) & (x.abs() < 2.0**-126)
    differs = s.view(torch.int32) != torch.sin(x).view(torch.int32)
    assert torch.equal(differs, subnormal)  # sin: torch's bits except on a subnormal ...
    assert torch.equal(s[subnormal].view(torch.int32), (x[subnormal] * 0.0).view(torch.int32))  # ... which is a zero of its sign


def test_rotary_tables_match_torch(exported):
    directory, m = exported
    half = torch.arange(0, 128, 2, dtype=F32) / 128
    inv = 1.0 / (10000.0**half)  # computed once on the CPU, as a host would
    inv_freq = torch.cat([inv, inv]).to(CUDA)
    positions = torch.arange(4096, dtype=torch.int32, device=CUDA).reshape(-1, 1)
    cos, sin = tilefish.run(tc.rope_tables, inv_freq, positions)
    angle = inv_freq * positions.float()
    assert same_bytes(cos, angle.cos().to(BF16)) and same_bytes(sin, angle.sin().to(BF16))
    for got, want in zip(tilefish.run(tc.rope_tables, inv_freq, positions, mode="replay", manifest=directory, pin=m.digest), (cos, sin)):
        assert same_bytes(got, want)


Sb = Dim("Sb", buckets=[256, 1024])


@tilefish.graph
def tables(inv_freq: Tensor[tc.H2, f32], positions: Tensor[Sb, 1, i32]):
    return tc.rope_tables(inv_freq, positions)


def test_rotary_tables_in_a_plan(tmp_path):
    p = runtime.export(tables, tmp_path)
    inv_freq = torch.rand(128, device=CUDA)
    for rows in (256, 1024):
        positions = torch.randint(0, 1 << 20, (rows, 1), dtype=torch.int32, device=CUDA)
        jit = tilefish.run(tables, inv_freq, positions)
        replay = tilefish.run(tables, inv_freq, positions, mode="replay", plan=tmp_path, pin=p.digest)
        ref = tilefish.run(tables, inv_freq, positions, mode="reference")
        for a, b, c in zip(jit, replay, ref):
            assert same_bytes(a, b) and same_bytes(a, c)
