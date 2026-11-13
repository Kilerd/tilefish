"""``tilefish.contiguous`` and ``tilefish.im2col``: generated ops over a strided view, their declarations, sources and plans (bytes:
``test_gpu_copyops.py``)."""

import pytest
from test_lower import check_golden

import tilefish
from tilefish import Dim, Tensor, TilefishTypeError, bf16, f32
from tilefish._copyops import contiguous_op, generated_source, im2col_op
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
