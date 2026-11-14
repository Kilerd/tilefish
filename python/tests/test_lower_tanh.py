"""``tanh``: traced like the other math functions, f32 only, lowered to libdevice's ``tanhf`` (bytes: ``test_gpu_tanh.py``; note 14)."""

import pytest
import tanh_cases as tc
from test_lower import check_golden

import tilefish
from tilefish import Dim, Tensor, TilefishLoweringError, bf16, f32
from tilefish.aot import compile_specs

N = Dim("N")


def test_tanh_traces_like_the_other_math_functions(tr):
    x = tr.val(Tensor[N, f32])
    assert tilefish.tanh(x).type == Tensor[N, f32]
    tilefish.tanh(x, math="ieee")
    assert [(n.op, n.attrs.get("math")) for n in tr.ctx.nodes] == [("tanh", None), ("tanh", "ieee")]


def test_golden_sources():
    check_golden("tanh_fast", tc.tanh_fast.lower().source)
    check_golden("tanh_ieee", tc.tanh_ieee.lower().source)


def test_it_is_libdevice_tanh_imported_by_the_module():
    src = tc.tanh_fast.lower().source
    assert "from triton.language.extra import libdevice" in src and "v1 = libdevice.tanh(v0)" in src
    ieee = tc.tanh_ieee.lower().source
    assert "return tl.where(tl.abs(x) < 1.1754943508222875e-38, x, libdevice.tanh(x))" in ieee and "v1 = _tf_tanh_ieee(v0)" in ieee
    rows = tc.tanh_rows.lower()
    assert rows.template == "row" and "_tf_tanh_ieee(" in rows.source and "# math: ieee" in rows.source
    assert "libdevice" not in tilefish.lower(tc.tanh_fast).source.split("def tf_tanh_fast")[0].replace(
        "from triton.language.extra import libdevice", ""
    )


def test_a_kernel_without_tanh_imports_no_libdevice():
    @tilefish.kernel
    def double(x: Tensor[N, f32]) -> Tensor[N, f32]:
        return x * 2.0

    assert "libdevice" not in double.lower().source


def test_a_parameter_named_libdevice_is_renamed():
    @tilefish.kernel
    def shadow(libdevice: Tensor[N, f32]) -> Tensor[N, f32]:
        return tilefish.tanh(libdevice)

    low = shadow.lower()
    assert "libdevice_1," in low.source and "libdevice.tanh(v0)" in low.source
    (spec,) = compile_specs(low.extern)
    assert "libdevice_1" in spec.signature


def test_tanh_is_f32_only():
    @tilefish.kernel
    def half(x: Tensor[N, bf16]) -> Tensor[N, bf16]:
        return tilefish.tanh(x)

    with pytest.raises(TilefishLoweringError) as e:
        half.lower()
    assert e.value.rule == "math-dtype"


def test_the_math_changes_the_binary():
    fast, ieee = compile_specs(tc.tanh_fast.lower().extern)[0], compile_specs(tc.tanh_ieee.lower().extern)[0]
    assert fast.source_sha256 != ieee.source_sha256
