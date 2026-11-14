"""``tilefish.contiguous``, ``tilefish.im2col`` and ``tilefish.im2col1d``: generated ops over a strided view, their declarations, sources and
plans (bytes: ``test_gpu_copyops.py``)."""

import pytest
from test_lower import check_golden

import tilefish
from tilefish import Dim, Tensor, TilefishTypeError, bf16, f16, f32
from tilefish._copyops import contiguous_op, generated_source, im2col1d_op, im2col_op
from tilefish.aot import compile_specs

N = Dim("N", buckets=[4, 8])
C = Dim("C", 16)
H = Dim("H", buckets=[40])
W = Dim("W", 200)
B = Dim("B", buckets=[2])
L = Dim("L", buckets=[100])


@tilefish.graph
def to_nhwc(x: Tensor[N, C, H, W, bf16]):
    return tilefish.contiguous(x.permute(0, 2, 3, 1))


@tilefish.graph
def transposed_patches(x: Tensor[B, L, 40, bf16]):
    return tilefish.im2col(x.reshape(B, 1, L, 40).permute(0, 1, 3, 2), kernel=3, stride=2, padding=1)


def test_contiguous_is_a_call_of_a_generated_op_on_the_view():
    view, out = to_nhwc.graph.nodes
    assert view.op == "permute" and not view.outputs[0].type.contiguous
    assert out.attrs["kind"] == "extern" and out.callee is contiguous_op(4, bf16, 16)  # the last axis (C = 16) is static
    assert out.outputs[0].type == Tensor[N, H, W, C, bf16]
    op = out.callee
    params = [(p.name, p.kind, p.strided, p.stride_of) for p in op.signature.params]
    assert params[0] == ("x", "tensor", True, None) and params[1][:2] == ("y", "tensor")
    assert [p[3] for p in params[6:10]] == [(("x", 0),), (("x", 1),), (("x", 2),), (("x", 3),)]  # one view per stride
    assert [(p.name, p.kind, p.default) for p in op.signature.params if p.kind == "const"] == [("a3", "const", 16), ("TR", "const", 128), ("TC", "const", 16)]
    assert op.index_bits == 64 and op.grid_string == "(cdiv(A2, 128), A1, A0)"  # tiles of 128 x 16


def test_one_op_per_rank_dtype_and_static_last_axis():
    assert contiguous_op(4, bf16) is contiguous_op(4, bf16)
    assert contiguous_op(3, bf16) is not contiguous_op(3, f32) and contiguous_op(3, bf16) is not contiguous_op(3, bf16, 128)
    (spec,) = compile_specs(contiguous_op(2, f32))
    assert spec.signature == {
        "x": "*fp32", "y": "*fp32", "a0": "i32", "a1": "i32", "x_s0": "i64", "x_s1": "i64", "TR": "constexpr", "TC": "constexpr",
    }  # fmt: skip
    assert spec.source_sha256 is not None and spec.grid == "(cdiv(A0, 32),)"


@pytest.mark.parametrize("rank,last", [(1, None), (2, None), (3, 128), (4, 16)])
def test_generated_sources(rank, last):
    check_golden(f"contiguous_r{rank}", generated_source(contiguous_op(rank, bf16, last)))


def test_im2col_shapes_and_sources():
    (_, _, call) = transposed_patches.graph.nodes
    assert call.outputs[0].type == Tensor[B, 20, tilefish.cdiv(L, 2), 9, bf16]
    assert call.callee is im2col_op(bf16, 3, 2, 1, channels=1)
    check_golden("im2col_c1", generated_source(call.callee))
    check_golden("im2col_c192", generated_source(im2col_op(bf16, 3, 2, 1, channels=192)))  # 1728 columns: two chunks, the last one masked
    dynamic = im2col_op(bf16, (3, 3), (2, 2), (1, 1))
    assert "    C,\n" in generated_source(dynamic) and "ncol = C * 9" in generated_source(dynamic)
    (spec,) = compile_specs(im2col_op(bf16, 3, 2, 1, channels=192))
    assert spec.grid == "(cdiv(W, 2), cdiv(H, 2), N)" and next(a.value for a in spec.args if a.name == "BLOCK") == 1024


def test_refusals(tr):
    x5 = tr.val(Tensor[2, 2, 2, 2, 2, bf16])
    with pytest.raises(TilefishTypeError, match="rank 1 to 4"):
        tilefish.contiguous(x5)
    with pytest.raises(TilefishTypeError, match="takes a traced tensor"):
        tilefish.contiguous(3)
    x = tr.val(Tensor[2, 3, 2, 2, bf16])
    with pytest.raises(TilefishTypeError, match="too small for a 3-wide window"):
        tilefish.im2col(x, kernel=3)
    with pytest.raises(TilefishTypeError, match="kernel must be"):
        tilefish.im2col(x, kernel=(0, 3))
    with pytest.raises(TilefishTypeError, match="it takes \\[N, C, H, W\\]"):
        tilefish.im2col(tr.val(Tensor[2, 2, bf16]))
    with pytest.raises(TilefishTypeError, match="rank must be 1 to 4"):
        contiguous_op(5, bf16)


def test_graphs_of_generated_ops_plan():
    p = tilefish.plan(to_nhwc)
    assert p.buckets() == [{"N": 4, "H": 40}, {"N": 8, "H": 40}]
    (step,) = p.steps(N=8, H=40)
    assert "contiguous_r4_bf16" in str(step)
    (step,) = tilefish.plan(transposed_patches).steps(B=2, L=100)
    assert "im2col_bf16_k3x3_s2x2_p1x1_c1" in str(step)


# -- im2col1d -----------------------------------------------------------------------------------------------------------------------

T = Dim("T", buckets=[64, 200])


@tilefish.graph
def causal_taps(x: Tensor[B, T, 1024, bf16]):  # a channel-last activation, read as [N, C, L]
    return tilefish.im2col1d(x.permute(0, 2, 1), 7, dilation=3, padding=(18, 0), order="tap")


@tilefish.graph
def strided_channels(x: Tensor[B, 80, T, f16]):
    return tilefish.im2col1d(x, 3, stride=2, padding=(2, 5))


def test_im2col1d_is_a_call_of_a_generated_op_on_the_view():
    view, call = causal_taps.graph.nodes
    assert view.op == "permute" and not view.outputs[0].type.contiguous
    op = im2col1d_op(bf16, 7, dilation=3, padding=(18, 0), order="tap", channels=1024)
    assert call.attrs["kind"] == "extern" and call.callee is op and op.name == "im2col1d_bf16_k7_s1_d3_p18x0_tap_c1024"
    assert call.outputs[0].type == Tensor[B, T, 7168, bf16]  # a causal padding keeps the length
    params = [(p.name, p.kind, p.strided, p.stride_of) for p in op.signature.params]
    assert params[0] == ("x", "tensor", True, None) and params[1][:2] == ("out", "tensor")
    assert [p[0] for p in params] == ["x", "out", "L", "Lo", "x_s0", "x_s1", "x_s2", "BLOCK"]
    assert [p[3] for p in params[4:7]] == [(("x", 0),), (("x", 1),), (("x", 2),)]
    assert op.index_bits == 64 and op.grid_string == "(L, N)"
    (call,) = strided_channels.graph.nodes  # a contiguous input: no view
    assert call.outputs[0].type == Tensor[B, tilefish.cdiv(T + 5, 2), 240, f16]  # (T + 2 + 5 - 2 - 1) // 2 + 1
    assert call.callee.name == "im2col1d_f16_k3_s2_d1_p2x5_channel_c80"


def test_im2col1d_compile_spec_and_sources():
    (spec,) = compile_specs(im2col1d_op(f16, 7, dilation=9, padding=27, order="tap"))
    assert spec.signature == {
        "x": "*fp16", "out": "*fp16", "C": "i32", "L": "i32", "Lo": "i32", "x_s0": "i64", "x_s1": "i64", "x_s2": "i64", "BLOCK": "constexpr",
    }  # fmt: skip
    assert spec.source_sha256 is not None and spec.grid == "(L, N)" and next(a.value for a in spec.args if a.name == "BLOCK") == 1024
    check_golden("im2col1d_tap_c1024", generated_source(im2col1d_op(bf16, 7, dilation=3, padding=(18, 0), order="tap", channels=1024)))
    check_golden("im2col1d_channel_c80", generated_source(im2col1d_op(f16, 3, stride=2, padding=(2, 5), channels=80)))  # 240 columns, one block
    check_golden("im2col1d_tap", generated_source(im2col1d_op(f16, 7, dilation=9, padding=27, order="tap")))  # a runtime channel count
    assert compile_specs(im2col1d_op(bf16, 7, dilation=3))[0].grid == "(L - 18, N)"
    (spec,) = compile_specs(im2col1d_op(f16, 3, stride=2, padding=(2, 5), channels=80))
    assert spec.grid == "(cdiv(L + 5, 2), N)" and next(a.value for a in spec.args if a.name == "BLOCK") == 256 and spec.num_warps == 1


def test_one_im2col1d_op_per_configuration():
    op = im2col1d_op(bf16, 7, dilation=3, padding=9, order="tap")
    assert op is im2col1d_op(bf16, 7, dilation=3, padding=(9, 9), order="tap") is im2col1d_op(bf16, 7, dilation=3, padding=[9, 9], order="tap")
    assert op.name == "im2col1d_bf16_k7_s1_d3_p9x9_tap"
    others = [
        im2col1d_op(bf16, 7, dilation=3, padding=9),  # the channel order
        im2col1d_op(f16, 7, dilation=3, padding=9, order="tap"),
        im2col1d_op(bf16, 7, stride=2, dilation=3, padding=9, order="tap"),
        im2col1d_op(bf16, 7, dilation=1, padding=9, order="tap"),
        im2col1d_op(bf16, 7, dilation=3, padding=(18, 0), order="tap"),
        im2col1d_op(bf16, 7, dilation=3, padding=(0, 18), order="tap"),
        im2col1d_op(bf16, 7, dilation=3, padding=9, order="tap", channels=64),
        im2col1d_op(bf16, 5, dilation=3, padding=9, order="tap"),
    ]
    assert len({op.name, *(o.name for o in others)}) == len(others) + 1 and all(o is not op for o in others)


def test_im2col1d_refusals(tr):
    x = tr.val(Tensor[2, 8, 20, bf16])
    with pytest.raises(TilefishTypeError, match="it takes \\[N, C, L\\]"):
        tilefish.im2col1d(tr.val(Tensor[2, 8, 4, 5, bf16]), 3)
    with pytest.raises(TilefishTypeError, match="takes a traced tensor"):
        tilefish.im2col1d(3, 3)
    with pytest.raises(TilefishTypeError, match="not a plain storage type"):
        tilefish.im2col1d(tr.val(Tensor[2, 8, 20, tilefish.bool_]), 3)
    for kwargs, match in [
        ({"kernel": True}, "kernel must be an int of at least 1, got True"),
        ({"kernel": 0}, "kernel must be an int of at least 1"),
        ({"kernel": 3, "stride": 0}, "stride must be an int of at least 1"),
        ({"kernel": 3, "dilation": True}, "dilation must be an int of at least 1"),
        ({"kernel": 3, "padding": -1}, "padding must be an int or a pair \\(left, right\\) of ints of at least 0, got -1"),
        ({"kernel": 3, "padding": (1, -2)}, "padding must be"),
        ({"kernel": 3, "padding": (1, 2, 3)}, "padding must be"),
        ({"kernel": 3, "padding": (True, 0)}, "padding must be"),
        ({"kernel": 3, "padding": 1.0}, "padding must be"),
        ({"kernel": 3, "order": "row"}, 'order must be "channel" or "tap", got \'row\''),
        ({"kernel": 3, "order": None}, "order must be"),
    ]:
        with pytest.raises(TilefishTypeError, match=match):
            tilefish.im2col1d(x, **kwargs)
    assert tilefish.im2col1d(x, 7, dilation=3).type == Tensor[2, 2, 56, bf16]  # a window of 19 in 20: two positions
    assert tilefish.im2col1d(x, 7, dilation=3, padding=(0, 1)).type == Tensor[2, 3, 56, bf16]
    assert tilefish.im2col1d(x, 11, dilation=2, padding=(0, 1)).type == Tensor[2, 1, 88, bf16]  # a window of 21: exactly one position
    with pytest.raises(TilefishTypeError, match="is 20, too short for 7 taps at dilation 4 \\(a window of 25\\) with padding \\(2, 2\\)"):
        tilefish.im2col1d(x, 7, dilation=4, padding=2)
    with pytest.raises(TilefishTypeError, match="channels must be a positive int"):
        im2col1d_op(bf16, 3, channels=0)
    with pytest.raises(TilefishTypeError, match="im2col1d_op: .* is not a plain storage type"):
        im2col1d_op(tilefish.bool_, 3)


def test_graphs_of_im2col1d_plan():
    p = tilefish.plan(causal_taps)
    assert p.buckets() == [{"B": 2, "T": 64}, {"B": 2, "T": 200}]
    (step,) = p.steps(B=2, T=200)
    assert "im2col1d_bf16_k7_s1_d3_p18x0_tap_c1024" in str(step)
    (step,) = tilefish.plan(strided_channels).steps(B=2, T=64)
    assert "im2col1d_f16_k3_s2_d1_p2x5_channel_c80" in str(step)
