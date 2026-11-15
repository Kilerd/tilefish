"""Float to integer casts with an explicit rounding and saturation (note 15 of the numerics contract) on a real GPU.

* For every integer type, saturation mode and rounding, the binary is byte-identical to the ``reference`` mode on **every one of the 2**32 f32
  bit patterns**, and both are the exact model of ``f2i_cases.expected`` on the adversarial inputs (ties, the bounds of every type, subnormals,
  infinities, NaNs with payloads); ``jit``, ``replay`` and ``reference`` agree.
* From bf16 and f16 (every bit pattern) the source is widened exactly, and the row template casts like the elementwise one.
* What torch's own casts are, in these terms, measured over every f32 (the statements of note 15).
"""

import collections

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import f2i_cases as fc  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import runtime  # noqa: E402

CUDA = "cuda"
I32, F32 = torch.int32, torch.float32
CHUNK = 1 << 26  # f32 bit patterns per launch: 2**32 in 64 chunks
NAN_PATTERNS = 2 * ((1 << 23) - 1)  # the f32 NaNs: every exponent bit set and a payload, either sign


def chunks():
    """Every f32 bit pattern, as 64 tensors of 2**26 (ascending as unsigned words)."""
    for start in range(0, 1 << 32, CHUNK):
        words = torch.arange(start, start + CHUNK, dtype=torch.int64, device=CUDA)
        yield words.to(I32).view(F32)


def f32s(patterns):
    return torch.tensor([b - (1 << 32) if b >= 1 << 31 else b for b in patterns], dtype=I32).view(F32).to(CUDA)


def mismatches(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    return int((a != b).sum())


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("f2i")
    ops = [*fc.BY_MODE.values(), fc.from_bf16, fc.quantize_rows]
    return directory, runtime.export(ops, directory)


def test_the_adversarial_values_are_the_model(exported):
    directory, m = exported
    x = f32s(fc.ADVERSARIAL)
    for (dtype, saturate), kernel in sorted(fc.BY_MODE.items()):
        got = tilefish.run(kernel, x)
        ref = tilefish.run(kernel, x, mode="reference")
        rep = tilefish.run(kernel, x, mode="replay", manifest=directory, pin=m.digest)
        for rounding, g, r, p in zip(fc.ROUNDINGS, got, ref, rep):
            assert same_bytes(g, r) and same_bytes(g, p), (dtype, saturate, rounding)
            want = [fc.expected(fc.f32_of(b), dtype, rounding, saturate) for b in fc.ADVERSARIAL]
            assert g.tolist() == want, (dtype, saturate, rounding)


@pytest.mark.parametrize("dtype, saturate", sorted(fc.BY_MODE))
def test_every_f32_is_the_reference(dtype, saturate):
    kernel = fc.BY_MODE[(dtype, saturate)]
    for x in chunks():
        got = tilefish.run(kernel, x)
        ref = tilefish.run(kernel, x, mode="reference")
        for rounding, g, r in zip(fc.ROUNDINGS, got, ref):
            assert same_bytes(g, r), (dtype, saturate, rounding, x.view(I32)[0].item())
        del got, ref


# (an e4m3 or e5m2 parameter is not loaded by a lowered kernel on Triton 3.5.1: ``tl.load(..., other=0)`` cannot make an fp8 zero, which is not
# this cast's; ``test_lower_f2i.py`` checks that the cast of one widens it first)
@pytest.mark.parametrize("kernel, dtype", [(fc.from_bf16, torch.bfloat16), (fc.from_f16, torch.float16)])
def test_every_narrower_float_is_widened_exactly(exported, kernel, dtype):
    directory, m = exported
    x = torch.arange(1 << 16, dtype=torch.int64, device=CUDA).to(torch.int16).view(dtype)
    got = tilefish.run(kernel, x)
    ref = tilefish.run(kernel, x, mode="reference")
    values = x.float().tolist()
    for (dt, rounding, saturate), g, r in zip(fc.MODES, got, ref):
        assert same_bytes(g, r), (kernel.name, dt, rounding, saturate)
        assert g.tolist() == [fc.expected(v, dt, rounding, saturate) for v in values], (kernel.name, dt, rounding, saturate)
    if kernel is fc.from_bf16:
        for g, p in zip(got, tilefish.run(kernel, x, mode="replay", manifest=directory, pin=m.digest)):
            assert same_bytes(g, p)


def test_the_row_template(exported):
    directory, m = exported
    x = randn((300, 256), F32, seed=4, scale=100.0)
    x.view(-1)[:7] = torch.tensor([float("inf"), float("-inf"), 0.5, 1.5, -0.5, 2.5, 1e-45], device=CUDA)  # no NaN: max() is not the subject
    scale = randn((300, 1), F32, seed=5)
    outs = [tilefish.run(fc.quantize_rows, x, scale, **kw) for kw in ({}, {"mode": "replay", "manifest": directory, "pin": m.digest}, {"mode": "reference"})]
    for o in outs[1:]:
        assert all(same_bytes(a, b) for a, b in zip(outs[0], o))
    q, d = outs[0]
    assert int(q[0, 0]) in (0, 255) and int(q[0, 1]) in (0, 255)  # +-inf scaled: a bound
    assert int(d[0, 0]) == 0  # inf - inf is a NaN, which is 0


def test_what_torchs_casts_are_in_these_terms():
    """torch's casts, measured over every f32 on this GPU, against the modes of note 15 (counts of differing outputs)."""
    count = collections.Counter()
    for x in chunks():
        i32_sat, i32_wrap = tilefish.run(fc.i32_sat, x), tilefish.run(fc.i32_wrap, x)
        i64 = tilefish.run(fc.i64_sat, x)
        u8_sat, u8_wrap = tilefish.run(fc.u8_sat, x), tilefish.run(fc.u8_wrap, x)
        count["int32 / rz saturating"] += mismatches(x.to(torch.int32), i32_sat[1])
        count["int64 / rz"] += mismatches(x.to(torch.int64), i64[1])
        count["int64 on a non-NaN / rz"] += mismatches(x.to(torch.int64)[~torch.isnan(x)], i64[1][~torch.isnan(x)])
        count["uint8 / rz wrapping"] += mismatches(x.to(torch.uint8), u8_wrap[1])
        count["uint8 / rz saturating"] += mismatches(x.to(torch.uint8), u8_sat[1])
        count["int64 then int32 / rz wrapping"] += mismatches(x.to(torch.int64).to(torch.int32), i32_wrap[1])
        count["round then int32 / rn saturating"] += mismatches(torch.round(x).to(torch.int32), i32_sat[0])
        count["floor then int32 / rd saturating"] += mismatches(torch.floor(x).to(torch.int32), i32_sat[2])
        count["ceil then int32 / ru saturating"] += mismatches(torch.ceil(x).to(torch.int32), i32_sat[3])
        del i32_sat, i32_wrap, i64, u8_sat, u8_wrap
    print(f"[f2i] torch's casts against the modes of note 15, differing f32 inputs of 2**32: {dict(count)}")
    assert count["int32 / rz saturating"] == 0  # torch's float to int32 on CUDA is cvt.rzi.s32.f32: rz, saturating, NaN 0
    assert count["int64 / rz"] == NAN_PATTERNS and count["int64 on a non-NaN / rz"] == 0  # cvt.rzi.s64.f32: a NaN is -2**63, not 0
    assert count["uint8 / rz wrapping"] == 0  # c10 casts a float to uint8 through int64: rz, then the low byte (-1.5 is 255)
    assert count["uint8 / rz saturating"] == 712_048_640  # every f32 whose truncation is outside [0, 255] but for those with the bound as low byte
    assert count["int64 then int32 / rz wrapping"] == 0  # C's cast of a narrowing: the low 32 bits of the i64
    for name in ("round then int32 / rn saturating", "floor then int32 / rd saturating", "ceil then int32 / ru saturating"):
        assert count[name] == 0, name
