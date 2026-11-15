"""Float to integer casts: ``x.to(i32, rounding=..., saturate=...)`` (note 15 of the numerics contract). What they trace to, what they refuse,
the inline PTX they lower to, and (with torch) what the ``reference`` mode computes; the bytes are checked on a GPU by ``test_gpu_f2i.py``.
"""

import f2i_cases as fc
import pytest
from test_lower import check_golden

import tilefish
from tilefish import Dim, Tensor, TilefishTypeError, bf16, e4m3, f16, f32, i32, i64, u8

N = Dim("N")


# -- tracing ------------------------------------------------------------------------------------------------------------------------


def test_the_mode_is_recorded_on_the_cast(tr):
    x = tr.val(Tensor[N, f32])
    a, b = x.to(i32, rounding="rn"), x.to(u8, rounding="rd", saturate=False)
    first, second = tr.nodes
    assert first.attrs == {"to": i32, "rounding": True, "round": "rn", "saturate": True} and first.is_rounding_point
    assert second.attrs == {"to": u8, "rounding": True, "round": "rd", "saturate": False}
    assert a.type == Tensor[N, i32] and b.type == Tensor[N, u8]
    assert str(first).endswith("cast(%p0) {to=i32, rounding=true, round=rn, saturate=true}  # rounding point")


@pytest.mark.parametrize("src", [f32, bf16, f16, e4m3], ids=str)
def test_a_float_to_integer_cast_has_no_default_rounding(tr, src):
    x = tr.val(Tensor[N, src])
    for dst in (i32, i64, u8):
        with pytest.raises(TilefishTypeError, match=rf"to: casting Tensor\[N, {src}\] to {dst} needs rounding='rn' \| 'rz' \| 'rd' \| 'ru'"):
            x.to(dst)
        with pytest.raises(TilefishTypeError, match="needs rounding="):
            x.to(dst, saturate=False)
    assert not tr.nodes


def test_the_modes_are_checked(tr):
    x = tr.val(Tensor[N, f32])
    for bad, match in (
        (lambda: x.to(i32, rounding="rne"), r"rounding must be one of \['rn', 'rz', 'rd', 'ru'\], got 'rne'"),
        (lambda: x.to(i32, rounding="RN"), "rounding must be one of"),
        (lambda: x.to(i32, rounding="rn", saturate=1), "saturate must be True or False, got 1"),
        (lambda: x.to(i32, rounding="rn", saturate="wrap"), "saturate must be True or False"),
    ):
        with pytest.raises(TilefishTypeError, match=match):
            bad()


def test_no_other_cast_takes_a_rounding_or_a_saturation(tr):
    x, n, m = tr.val(Tensor[N, f32]), tr.val(Tensor[N, i32]), tr.val(Tensor[N, u8])
    for bad, what in (
        (lambda: x.to(bf16, rounding="rn"), "rounding="),
        (lambda: x.to(f32, rounding="rz"), "rounding="),
        (lambda: x.to(e4m3, saturate=True), "saturate="),
        (lambda: n.to(f32, rounding="rn"), "rounding="),
        (lambda: n.to(i64, rounding="rz"), "rounding="),
        (lambda: n.to(u8, saturate=False), "saturate="),
        (lambda: m.to(i32, rounding="rn"), "rounding="),
        (lambda: (x > 0).to(i32, rounding="rn"), "rounding="),
    ):
        with pytest.raises(TilefishTypeError, match=f"to: {what} is a mode of a float to integer cast; casting .* has none"):
            bad()


# -- lowering -----------------------------------------------------------------------------------------------------------------------


def test_golden_source():
    check_golden("f2i_u8_wrap", fc.u8_wrap.lower().source)


@pytest.mark.parametrize("rounding, ptx", [("rn", "rni"), ("rz", "rzi"), ("rd", "rmi"), ("ru", "rpi")])
def test_each_mode_is_one_helper_of_inline_ptx(rounding, ptx):
    i32_sat = fc.i32_sat.lower().source
    assert f'"cvt.{ptx}.s32.f32 $0, $1;",\n        "=r,f",\n        [x],\n        dtype=tl.int32,' in i32_sat
    u8_sat = fc.u8_sat.lower().source
    assert f'"cvt.{ptx}.s32.f32 t, $1; "\n        "max.s32 t, t, 0; "\n        "min.s32 $0, t, 255; "' in u8_sat
    for k in (fc.i64_sat, fc.i64_wrap, fc.i32_wrap, fc.u8_wrap):  # through i64: a NaN, which cvt.s64 makes -2**63, is 0
        src = k.lower().source
        assert f'"cvt.{ptx}.s64.f32 t, $1; "\n        "testp.notanumber.f32 p, $1; "\n        "selp.s64 $0, 0, t, p; "' in src
        assert '"=l,f",' in src and "dtype=tl.int64," in src
    assert ".ftz" not in i32_sat + u8_sat  # a subnormal is rounded as it is, not flushed to zero first


def test_the_wrapping_casts_take_the_low_bits_of_the_i64():
    body = fc.i32_wrap.lower().source.split("def tf_i32_wrap(")[1]
    assert "v1 = _tf_f2i_s64_rn(v0).to(tl.int32)  # rounding point" in body
    assert "v1 = _tf_f2i_u8_rn(v0).to(tl.uint8)  # rounding point" in fc.u8_sat.lower().source  # in [0, 255]: the low byte is the value
    assert "v1 = _tf_f2i_s64_rn(v0).to(tl.uint8)  # rounding point" in fc.u8_wrap.lower().source
    # for i64 the two modes are one: the same source but for the kernel's name
    assert fc.i64_sat.lower().source.replace("i64_sat", "K") == fc.i64_wrap.lower().source.replace("i64_wrap", "K")


def test_a_narrower_float_is_widened_to_f32_first():
    src = fc.from_bf16.lower().source
    assert "v1_x = v0.to(tl.float32)  # exact" in src and "v1 = _tf_f2i_s32_rn(v1_x)  # rounding point" in src
    assert src.count("@triton.jit") == 1 + 4 * 3  # the kernel, and a helper per (s32, u8, s64) and rounding, each written once
    for k in (fc.from_f16, fc.from_e4m3, fc.from_e5m2):
        assert "v1_x = v0.to(tl.float32)  # exact" in k.lower().source


def test_the_row_template():
    low = fc.quantize_rows.lower()
    assert low.template == "row"
    assert "v4 = _tf_f2i_u8_rn(v3).to(tl.uint8)  # rounding point" in low.source
    assert "v7 = _tf_f2i_s64_rd(v6).to(tl.int32)  # rounding point" in low.source


def test_the_math_mode_does_not_change_a_cast():
    @tilefish.kernel(math="ieee")
    def ieee(x: Tensor[N, f32]) -> Tensor[N, i32]:
        return x.to(i32, rounding="rn")

    @tilefish.kernel
    def fast(x: Tensor[N, f32]) -> Tensor[N, i32]:
        return x.to(i32, rounding="rn")

    def body(k):
        return k.lower().source.split("@triton.jit")[1:]

    assert [b.replace("ieee", "K") for b in body(ieee)] == [b.replace("fast", "K") for b in body(fast)]


# -- the model, and the reference mode ----------------------------------------------------------------------------------------------


def test_the_model_of_each_mode():
    """Spot values of :func:`f2i_cases.expected`, written out: the reference and the binaries are held to it on every adversarial input."""
    e = fc.expected
    assert [e(v, "i32", "rn", True) for v in (0.5, 1.5, 2.5, -0.5, -1.5, -2.5)] == [0, 2, 2, 0, -2, -2]  # ties to even
    assert [e(v, "i32", r, True) for r in fc.ROUNDINGS for v in (1.7, -1.7)] == [2, -2, 1, -1, 1, -2, 2, -1]
    assert [e(1e-45, "i32", r, True) for r in fc.ROUNDINGS] == [0, 0, 0, 1] and [e(-1e-45, "i32", r, True) for r in fc.ROUNDINGS] == [0, 0, -1, 0]
    assert e(2147483648.0, "i32", "rn", True) == 2**31 - 1 and e(2147483648.0, "i32", "rn", False) == -(2**31)
    assert e(-2147483904.0, "i32", "rz", True) == -(2**31) and e(-2147483904.0, "i32", "rz", False) == 2**31 - 256
    assert e(4294967296.0, "i32", "rz", False) == 0 and e(1e30, "i32", "rz", False) == -1  # 2**63 - 1, then its low 32 bits
    assert [e(v, "u8", "rz", True) for v in (-1.5, 255.5, 256.0, 300.0, 1e30)] == [0, 255, 255, 255, 255]
    assert [e(v, "u8", "rz", False) for v in (-1.5, 255.5, 256.0, 300.0, 1e30, -1e30)] == [255, 255, 0, 44, 255, 0]
    assert [e(v, "u8", "rn", False) for v in (-0.5, -1.5, 254.5, 255.5)] == [0, 254, 254, 0]
    assert e(float("nan"), "i64", "ru", True) == 0 and e(float("-inf"), "i64", "rn", True) == -(2**63) and e(1e19, "i64", "rd", False) == 2**63 - 1
    for dt in ("i32", "i64", "u8"):
        for s in (True, False):
            assert e(float("nan"), dt, "rn", s) == 0


def _reference_case(torch, kernel, dtype, saturate):
    x = torch.tensor([b - (1 << 32) if b >= 1 << 31 else b for b in fc.ADVERSARIAL], dtype=torch.int32).view(torch.float32)
    got = tilefish.run(kernel, x, mode="reference")
    for rounding, out in zip(fc.ROUNDINGS, got):
        want = [fc.expected(fc.f32_of(b), dtype, rounding, saturate) for b in fc.ADVERSARIAL]
        assert out.tolist() == want, (dtype, rounding, saturate)


@pytest.mark.parametrize("dtype, saturate", sorted(fc.BY_MODE))
def test_the_reference_is_the_model_on_adversarial_inputs(dtype, saturate):
    torch = pytest.importorskip("torch")
    _reference_case(torch, fc.BY_MODE[(dtype, saturate)], dtype, saturate)


def test_the_reference_widens_a_narrower_float_exactly():
    torch = pytest.importorskip("torch")
    x = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)  # every bf16 bit pattern
    got = tilefish.run(fc.from_bf16, x, mode="reference")
    values = x.float().tolist()
    for (dt, r, s), out in zip(fc.MODES, got):
        assert out.tolist() == [fc.expected(v, dt, r, s) for v in values], (dt, r, s)
