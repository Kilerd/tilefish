"""The NaN modes of the f32 to bf16 cast: ``nan="canonical"`` (the default) and ``nan="preserve"`` (bytes: ``test_gpu_casts.py``)."""

import nan_cast_cases as nc
import pytest
from test_lower import check_golden

from tilefish import Dim, Tensor, TilefishTypeError, bf16, f16, f32

N = Dim("N")


def test_preserve_is_recorded_and_canonical_is_the_plain_cast(tr):
    x = tr.val(Tensor[N, f32])
    plain, canonical, preserve = x.to(bf16), x.to(bf16, nan="canonical"), x.to(bf16, nan="preserve")
    nodes = tr.ctx.nodes
    assert nodes[0].attrs == nodes[1].attrs == {"to": bf16, "rounding": True}
    assert nodes[2].attrs == {"to": bf16, "rounding": True, "nan": "preserve"} and nodes[2].is_rounding_point
    assert plain.type == canonical.type == preserve.type == Tensor[N, bf16]


def test_only_the_f32_to_bf16_cast_has_a_nan_mode(tr):
    x = tr.val(Tensor[N, f32])
    for bad, match in (
        (lambda: x.to(bf16, nan="payload"), "nan must be 'canonical' or 'preserve'"),
        (lambda: x.to(f16, nan="preserve"), "nan= is a mode of the f32 to bf16 cast"),
        (lambda: tr.val(Tensor[N, f16]).to(bf16, nan="preserve"), "nan= is a mode of the f32 to bf16 cast"),
        (lambda: x.to(f32, nan="canonical"), "nan= is a mode of the f32 to bf16 cast"),
    ):
        with pytest.raises(TilefishTypeError, match=match):
            bad()


def test_golden_source():
    check_golden("to_bf16", nc.to_bf16.lower().source)


def test_preserve_lowers_to_the_integer_rounding_on_the_bits():
    src = nc.to_bf16.lower().source
    assert "v1 = v0.to(tl.bfloat16)  # rounding point" in src
    assert "v2_b = v0.to(tl.uint32, bitcast=True)" in src
    assert "v2_r = (v2_b + 0x7FFF + ((v2_b >> 16) & 1)) >> 16" in src
    assert "v2_q = (v2_b >> 16) | 0x40" in src
    assert "v2 = tl.where(v0 != v0, v2_q, v2_r).to(tl.uint16).to(tl.bfloat16, bitcast=True)  # rounding point" in src
    assert nc.to_bf16_rows.lower().template == "row"
