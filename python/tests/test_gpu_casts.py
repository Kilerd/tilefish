"""The two NaN modes of the f32 to bf16 cast, on a real GPU, over every one of the 2**32 f32 bit patterns.

``x.to(bf16)`` (``nan="canonical"``) is ``cvt.rn.bf16.f32``: torch's CUDA cast, and the canonical NaN ``0x7FFF``. ``x.to(bf16, nan="preserve")`` is
the integer round to nearest even on the bits, which keeps a NaN's sign and top payload bits. Both binaries are byte-identical to the
reference everywhere, and they differ from each other on exactly the 16 646 142 NaNs whose upper half is neither ``0x7FFF`` nor ``0x7FBF``.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import nan_cast_cases as nc  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import runtime  # noqa: E402

CUDA = "cuda"
CHUNK = 1 << 28


def words(start, count):
    return torch.arange(start, start + count, dtype=torch.int64, device=CUDA).to(torch.int32).view(torch.float32)


def test_every_f32_in_both_modes(tmp_path):
    m = runtime.export([nc.to_bf16], tmp_path)
    differ = 0
    for start in range(-(2**31), 2**31, CHUNK):
        x = words(start, CHUNK)
        canonical, preserve = tilefish.run(nc.to_bf16, x)
        ref_c, ref_p = tilefish.run(nc.to_bf16, x, mode="reference")
        assert same_bytes(canonical, ref_c) and same_bytes(preserve, ref_p), start
        assert same_bytes(canonical, x.to(torch.bfloat16)), start  # torch's CUDA cast is cvt.rn.bf16.f32
        d = canonical.view(torch.int16) != preserve.view(torch.int16)
        assert not d[~torch.isnan(x)].any(), start  # the modes differ on NaNs only
        differ += int(d.sum())
        if start == -(2**31):
            rep_c, rep_p = tilefish.run(nc.to_bf16, x, mode="replay", manifest=tmp_path, pin=m.digest)
            assert same_bytes(rep_c, canonical) and same_bytes(rep_p, preserve)
        del x, canonical, preserve, ref_c, ref_p, d
    assert differ == 16_646_142


def test_what_each_mode_makes_of_a_nan():
    x = torch.tensor([0x7FC00000, 0xFFC00000 - 2**32, 0x7F800001, 0xFFFFFFFF - 2**32, 0x7FFFFFFF, 0x7FBF0000], dtype=torch.int64)
    x = x.to(torch.int32).view(torch.float32).to(CUDA)
    canonical, preserve = tilefish.run(nc.to_bf16, x)
    hexes = lambda t: [w & 0xFFFF for w in t.view(torch.int16).tolist()]  # noqa: E731
    assert hexes(canonical) == [0x7FFF] * 6
    assert hexes(preserve) == [0x7FC0, 0xFFC0, 0x7FC0, 0xFFFF, 0x7FFF, 0x7FFF]


def test_a_preserving_cast_in_the_row_template():
    x = randn((40, 1024), torch.float32, seed=1)
    x.view(torch.int32)[::3, ::5] = -4194303  # 0xFFC00001: a negative NaN with a payload
    scale = randn((1024,), torch.float32, seed=2)
    scale[::2] = 0.0
    got = tilefish.run(nc.to_bf16_rows, x, scale)
    assert same_bytes(got, tilefish.run(nc.to_bf16_rows, x, scale, mode="reference"))
    kept = got[:, ::2].view(torch.int16)
    assert (kept[::3, ::5] == -64).all()  # 0xFFC0: the sign and the payload's top bits, through the preserving cast
