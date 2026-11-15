"""Plan fusion (``docs/rfcs/0001-plan-fusion.md``, :mod:`tilefish._fusion`) without a GPU: how the arithmetic of a graph body becomes regions,
each a kernel step of a generated kernel; what the regions read and write; the numerics rewrites; the fusion report; and what is refused.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import json
import os
import re
import pathlib
import subprocess
import sys

import fusion_graphs as fg
import pytest

import tilefish
from tilefish import DevScalar, Dim, InOut, Tensor, TilefishPlanError, TilefishTypeError, bf16, f16, f32, i32, u8
from tilefish import _fusion
from tilefish.lower import lower

HERE = pathlib.Path(__file__).resolve().parent
R, D, N = fg.R, fg.D, fg.N


def steps(graph, **bucket):
    p = tilefish.plan(graph)
    return p.steps(**bucket) if bucket else p._doc["buckets"][0]["steps"]


def ops_of(region, f):
    return [f.program[k].op for k in region.nodes]


# -- the declaration ----------------------------------------------------------------------------------------------------------------------


def test_fusion_is_on_by_default_and_its_settings_are_checked():
    assert _fusion.settings_of(fg.bias_act) == tilefish.fusion() == tilefish.Fusion()
    assert tilefish.fusion().canonical() == {"version": 1, "grouping": "auto", "math": "fast", "intermediates": "declared", "fp_contract": "none"}
    for bad in (dict(version=2), dict(grouping="some"), dict(math="exact"), dict(intermediates="f32"), dict(fp_contract="compiler"), dict(verify=1)):
        with pytest.raises(TilefishPlanError, match="fusion:"):
            tilefish.fusion(**bad)
    with pytest.raises(tilefish.TilefishDeclarationError, match="fusion= takes tilefish.fusion"):

        @tilefish.graph(fusion={"grouping": "none"})
        def g(x: Tensor[R, D, f32]):
            return x + x


def test_a_graph_without_arithmetic_plans_exactly_as_before():
    import plan_model as pm

    assert _fusion.planned_graph(pm.block) is pm.block.graph
    assert not _fusion.has_arithmetic(pm.block)


# -- regions -------------------------------------------------------------------------------------------------------------------------------


def test_an_elementwise_chain_is_one_kernel_step_of_a_generated_kernel():
    s = steps(fg.bias_act, R=8)
    assert [x["kind"] for x in s] == ["kernel"] and s[0]["entry"].startswith("fused_")
    assert [a["binding"] for a in s[0]["args"]] == ["x", "b", "out0"]
    f = _fusion.fused(fg.bias_act)
    (r,) = f.regions
    assert ops_of(r, f) == ["cast", "cast", "add", "silu", "cast"]
    assert r.kernel.fp_fusion is False and r.kernel.context_free is True and r.kernel.math == "fast"
    assert _fusion.fused(fg.bias_act_ieee).regions[0].kernel.math == "ieee"


def test_the_unfused_composition_is_one_kernel_per_op():
    f = _fusion.fused(fg.bias_act_unfused)
    assert [ops_of(r, f) for r in f.regions] == [["cast"], ["cast"], ["add"], ["silu"], ["cast"]]
    assert len(steps(fg.bias_act_unfused, R=8)) == 5
    # the same graph with grouping="none" given at the call: the same regions
    g = _fusion.fused(fg.bias_act, "none")
    assert [ops_of(r, g) for r in g.regions] == [["cast"], ["cast"], ["add"], ["silu"], ["cast"]]


def test_two_values_that_leave_a_region_are_two_outputs_and_independent_chains_are_two_regions():
    f = _fusion.fused(fg.masked)
    (r,) = f.regions
    assert ops_of(r, f) == ["lt", "mul", "add", "where", "mul"]
    assert len(r.outputs) == 2  # y, and y * y that the barrier keeps
    m = _fusion.fused(fg.modes)
    assert len(m.regions) == 2  # a * a + 1 and the other output share no op: no horizontal fusion in version 1


def test_a_region_never_spans_a_call_and_the_steps_keep_the_order_of_the_values():
    s = steps(fg.around_call, R=8)
    assert [x["entry"].split(".")[0] for x in s][1] == "fg_scale_rows"
    assert s[0]["entry"].startswith("fused_") and s[2]["entry"].startswith("fused_")
    f = _fusion.fused(fg.around_call)
    assert [ops_of(r, f) for r in f.regions] == [["mul"], ["add", "mul"]]


def test_a_reduction_without_an_order_is_a_kernel_of_its_own():
    f = _fusion.fused(fg.rms)
    assert [ops_of(r, f) for r in f.regions] == [["mul"], ["sum"], ["mul", "add", "rsqrt", "mul", "mul", "cast"]]

    @tilefish.graph
    def ordered(x: Tensor[R, D, f32]):
        return tilefish.sum(x * x, axis=-1, keepdims=True, order=tilefish.tree(32)) * 2.0

    g = _fusion.fused(ordered)
    assert [ops_of(r, g) for r in g.regions] == [["mul"], ["sum"], ["mul"]]  # version 1: every reduction alone, ordered or not


def test_free_ops_are_copied_into_each_region_and_a_lone_chain_with_no_shape_joins_its_consumer():
    f = _fusion.fused(fg.masked_unfused)
    regions = [ops_of(r, f) for r in f.regions]
    assert ["lt", "where"] in regions  # iota(s, 0) < load(n) has no input of shape [N]: it joins the where that reads it
    lt = next(r for r in f.regions if ops_of(r, f) == ["lt", "where"])
    assert sorted(f.program[k].op for k in lt.free) == ["iota", "load"]


def test_a_mask_leaves_its_region_as_u8_and_is_read_back_as_a_comparison():
    @tilefish.graph(fusion=fg.UNFUSED)
    def sel(x: Tensor[R, D, f32], y: Tensor[R, D, f32]):
        return tilefish.where(x > y, x, y)

    f = _fusion.fused(sel)
    gt, where = f.regions
    assert gt.kernel.signature.returns[0].dtype == u8
    assert [p.type.dtype for p in where.kernel.signature.tensor_inputs][0] == u8
    assert where.kernel.graph.nodes[0].op == "ne"
    assert "ne" in [n.op for n in where.kernel.graph.nodes] and steps(sel, R=8)[0]["entry"].startswith("fused_")


def test_a_barrier_makes_a_value_a_buffer_of_the_plan_and_is_refused_where_it_means_nothing():
    @tilefish.graph
    def two(x: Tensor[R, D, f32]):
        return tilefish.barrier(x * 2.0) + 1.0

    f = _fusion.fused(two)
    assert [ops_of(r, f) for r in f.regions] == [["mul"], ["add"]]
    assert "barrier" not in [n.op for n in f.graph.nodes]

    with pytest.raises(TilefishTypeError, match="barrier: .* is in a kernel body"):

        @tilefish.kernel
        def k(x: Tensor[R, D, f32]) -> Tensor[R, D, f32]:
            return tilefish.barrier(x) * 2.0

    with pytest.raises(TilefishTypeError, match="a mask, which has no storage"):

        @tilefish.graph
        def m(x: Tensor[R, D, f32]):
            return tilefish.barrier(x > 0)


def test_views_are_read_where_they_are_or_materialised_where_the_template_cannot():
    # SwiGLU's halves, two column blocks of one tensor: read where they are (Strided parameters, a stride per axis), no copy
    s = steps(fg.swiglu, R=8)
    assert [x["entry"].split("_")[0] for x in s] == ["fused"]
    assert [a.get("view", {}).get("strides") for a in s[0]["args"]] == [[128, 1], [128, 1], None]
    rep = tilefish.fusion_report(fg.swiglu)
    assert [i["read"] for i in rep["regions"][0]["inputs"]] == ["strided", "strided"]
    k = _fusion.fused(fg.swiglu).regions[0].kernel
    assert [p.name for p in k.signature.params if p.stride_of] == ["a0_s0", "a0_s1", "a1_s0", "a1_s1"]
    src = k.lower().source
    assert "tl.load(a0 + ((offs // 64) * a0_s0 + (offs % 64) * a0_s1)" in src

    @tilefish.graph
    def transposed(x: Tensor[D, R, f32], y: Tensor[R, D, f32]):
        return x.permute(1, 0) * 2.0 + y  # a transpose: no axis of stride 1

    assert [i["read"] for i in tilefish.fusion_report(transposed)["regions"][0]["inputs"]] == ["strided", "contiguous"]

    # a row region (a per-row scale) cannot read a view where it is: the view is copied by tilefish.contiguous first
    @tilefish.graph
    def scaled_half(x: Tensor[R, 2 * D, f32], w: Tensor[D, f32]):
        g, _ = x.split(D, axis=-1)
        return g * w + 1.0

    rep = tilefish.fusion_report(scaled_half)
    assert rep["regions"][0]["template"] in ("row", "elementwise")
    if rep["regions"][0]["template"] == "row":
        assert rep["regions"][0]["inputs"][0]["read"] == "materialised"
        assert [x["entry"].split("_")[0] for x in steps(scaled_half, R=8)] == ["contiguous", "fused"]

    P = Dim("P", 6)

    @tilefish.graph
    def rows(x: Tensor[R, P, D, f32]):
        return x[1] * 2.0 + x[3]  # dense rows of a [R, P, D] tensor

    f = _fusion.fused(rows)
    (r,) = f.regions
    assert set(r.reads.values()) <= {"contiguous", "unaligned", "strided"}


def test_a_region_is_named_by_its_content():
    a, b = _fusion.fused(fg.bias_act).regions[0].kernel, _fusion.fused(fg.bias_act_unfused).regions[0].kernel
    assert a.name.startswith("fused_") and len(a.name) == len("fused_") + 12

    @tilefish.graph
    def again(x: Tensor[R, D, bf16], b: Tensor[D, bf16]):
        return tilefish.silu(x.to(f32) + b.to(f32)).to(bf16)

    assert _fusion.fused(again).regions[0].kernel is a  # the same region anywhere is one operator
    R2 = Dim("R", buckets=[8, 24], divisible_by=8)

    @tilefish.graph
    def other_dim(x: Tensor[R2, D, bf16], b: Tensor[D, bf16]):
        return tilefish.silu(x.to(f32) + b.to(f32)).to(bf16)

    assert _fusion.fused(other_dim).regions[0].kernel.name != a.name  # another dimension constraint, another compile spec
    assert b is not a


def test_names_and_plans_are_the_same_in_another_process():
    script = (
        "import sys, json; sys.path.insert(0, %r); import tilefish, fusion_graphs as fg\n"
        "print(json.dumps({n: tilefish.plan(g).digest for n, (g, _) in sorted(fg.CASES.items())}))" % str(HERE)
    )
    env = {**os.environ, "PYTHONHASHSEED": "123"}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True).stdout
    theirs = json.loads(out)
    ours = {n: tilefish.plan(g).digest for n, (g, _) in sorted(fg.CASES.items())}
    assert theirs == ours


def test_long_chains_are_split_by_the_limit_of_a_region():
    @tilefish.graph
    def long(x: Tensor[R, D, f32]):
        y = x
        for _ in range(100):
            y = y * 1.5
        return y

    f = _fusion.fused(long)
    assert [len(r.nodes) for r in f.regions] == [_fusion.MAX_REGION_NODES, 100 - _fusion.MAX_REGION_NODES]


# -- the numerics ------------------------------------------------------------------------------------------------------------------------


def test_every_narrowing_float_cast_of_a_region_is_inline_ptx_and_a_default_kernel_keeps_to():
    f = _fusion.fused(fg.bias_act)
    source = f.regions[0].kernel.lower().source
    assert "cvt.rn.bf16.f32 $0, $1;" in source and ".to(tl.bfloat16)" not in source
    assert "context-free: every narrowing float cast is inline PTX" in source

    @tilefish.kernel
    def default(x: Tensor[R, D, bf16], b: Tensor[D, bf16]) -> Tensor[R, D, bf16]:
        return tilefish.silu(x.to(f32) + b.to(f32)).to(bf16)

    plain = lower(default).source  # fp_fusion=True: the compiler's choice, and no fold can reach this cast (its operand is not widened)
    assert "cvt.rn" not in plain and ".to(tl.bfloat16)" in plain

    @tilefish.kernel(fp_fusion=False)
    def k(x: Tensor[R, D, bf16], b: Tensor[D, bf16]) -> Tensor[R, D, bf16]:
        return tilefish.silu(x.to(f32) + b.to(f32)).to(bf16)

    assert "cvt.rn.bf16.f32" in lower(k).source and ".to(tl.bfloat16)" not in lower(k).source  # fp_fusion=False: every rounding written out
    opaque = lower(k, context_free=True).source
    # the region's kernel is the hand-written kernel of the same ops, lowered with opaque casts: the same source but for the names
    def norm(s, name):
        s = re.sub(rf"\b(tf_)?{name}\b", lambda m: (m.group(1) or "") + "K", s)
        return re.sub(r"\ba0\b", "x", re.sub(r"\ba1\b", "b", s))

    assert norm(source, f.regions[0].kernel.name) == norm(opaque, "k")

    @tilefish.kernel(fp_fusion=False)
    def h(x: Tensor[R, D, f16]) -> Tensor[R, D, bf16]:
        return x.to(bf16)

    assert "cvt.rn.bf16.f32" in lower(h, context_free=True).source and ".to(tl.float32)" in lower(h, context_free=True).source


def test_mul_add_contracts_a_product_nothing_else_reads_and_nothing_else():
    f = _fusion.fused(fg.modes)
    first, second = f.regions
    assert ops_of(first, f) == ["fma"]
    assert [w["kind"] for w in f.rewrites] == ["opmath", "mul-add", "mul-add"]
    assert ops_of(second, f).count("fma") == 1 and "neg" in ops_of(second, f)

    @tilefish.graph(fusion=tilefish.fusion(fp_contract="mul-add"))
    def shared(a: Tensor[R, D, f32]):
        p = a * a
        return p + 1.0, p  # the product is returned too: not contracted

    g = _fusion.fused(shared)
    assert [ops_of(r, g) for r in g.regions] == [["mul", "add"]] and not g.rewrites

    @tilefish.graph(fusion=tilefish.fusion(fp_contract="mul-add"))
    def halves(a: Tensor[R, D, bf16]):
        return a * a + a  # bf16: no 16-bit fma, nothing contracted

    assert not _fusion.fused(halves).rewrites


def test_opmath_computes_a_half_op_with_a_scalar_in_f32_with_the_scalar_in_f32():
    f = _fusion.fused(fg.modes)
    second = f.regions[1]
    nodes = [f.program[k] for k in second.nodes]
    mul = next(n for n in nodes if n.op == "mul")
    assert mul.outputs[0].type.dtype == f32
    scalar = next(i for i in mul.inputs if isinstance(i, tilefish.Scalar))
    assert scalar.dtype == f32 and scalar.value == 0.1
    src = second.kernel.lower().source
    # the multiply by 0.1 has an f32 operand (the widened bf16), so the literal is f32(0.1), not bf16(0.1) = 0.10009765625
    line = next(ln for ln in src.splitlines() if "_tf_mul_rn_f32(" in ln and ", 0.1)" in ln)  # one f32 multiply (note 20)
    widened = line.split("_tf_mul_rn_f32(")[1].split(",")[0].strip()
    assert next(ln for ln in src.splitlines() if ln.strip().startswith(f"{widened} = ")).endswith(".to(tl.float32)")


def test_a_graph_const_is_its_value_in_a_region():
    @tilefish.graph
    def scaled(x: Tensor[R, D, f32], *, k: tilefish.const = 3.0):
        return x * k

    src = _fusion.fused(scaled).regions[0].kernel.lower().source
    assert "3.0" in src


# -- the report ------------------------------------------------------------------------------------------------------------------------


def test_the_report_names_the_ops_of_each_region_and_what_it_keeps():
    rep = tilefish.fusion_report(fg.bias_act)
    assert rep["schema"] == "tilefish.fusion-report.v1" and rep["graph"] == "bias_act"
    (r,) = rep["regions"]
    assert [n["op"] for n in r["nodes"]] == ["cast", "cast", "add", "silu", "cast"]
    assert [x["op"] for x in r["roundings"]] == ["add", "silu", "cast"]
    assert r["template"] in ("elementwise", "row") and r["exact"] is True and r["rewrites"] == []
    json.dumps(rep)
    m = tilefish.fusion_report(fg.modes)
    assert any(w["kind"] == "mul-add" for reg in m["regions"] for w in reg["rewrites"])


# -- refusals ------------------------------------------------------------------------------------------------------------------------------


def test_what_fusion_does_not_plan_yet_is_refused_with_the_rule():
    @tilefish.graph
    def scatter(dst: InOut[Tensor[N, f32]], idx: Tensor[R, i32], src: Tensor[R, f32]):
        tilefish.scatter_rows(dst, idx, src, oob="skip")

    with pytest.raises(TilefishPlanError, match="scatter into an InOut parameter in a graph body is not planned yet"):
        tilefish.plan(scatter)

    @tilefish.graph
    def scalar(x: Tensor[R, D, f32], n: DevScalar[i32]):
        return fg.fg_scale_rows(x), tilefish.load(n)

    with pytest.raises(TilefishPlanError, match="used outside the arithmetic"):
        tilefish.plan(scalar)
