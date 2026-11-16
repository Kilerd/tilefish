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
    assert tilefish.fusion().canonical() == {"version": 2, "grouping": "auto", "math": "fast", "intermediates": "declared", "fp_contract": "none"}
    assert tilefish.fusion(version=1).canonical()["version"] == 1
    for bad in (dict(version=3), dict(version=True), dict(grouping="some"), dict(math="exact"), dict(intermediates="f32"), dict(fp_contract="compiler"), dict(verify=1)):
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
    f = _fusion.fused(fg.rms)  # version 2 (the default) too: D2
    assert f.settings.version == 2
    assert [ops_of(r, f) for r in f.regions] == [["mul"], ["sum"], ["mul", "add", "rsqrt", "mul", "mul", "cast"]]

    def body(x):
        return tilefish.sum(x * x, axis=-1, keepdims=True, order=tilefish.tree(32)) * 2.0

    @tilefish.graph
    def ordered(x: Tensor[R, D, f32]):
        return body(x)

    @tilefish.graph(fusion=fg.V1)
    def ordered_v1(x: Tensor[R, D, f32]):
        return body(x)

    assert [ops_of(r, _fusion.fused(ordered)) for r in _fusion.fused(ordered).regions] == [["mul", "sum", "mul"]]  # version 2: one row region
    g = _fusion.fused(ordered_v1)
    assert [ops_of(r, g) for r in g.regions] == [["mul"], ["sum"], ["mul"]]  # version 1: every reduction alone, ordered or not


# -- version 1 is frozen; the row regions of version 2 ----------------------------------------------------------------------------------------

# the plan digests of these graphs as tilefish 0.0.9 and 0.0.10 planned them (fusion version 1, their default): version 1 plans them the same, byte for byte
V1_DIGESTS = {
    "around_call": "968e4f545d2fdab1a7b53e2a67f944799c86902b56e09ff3a8465a6544917a97",
    "bias_act": "209d35ba2b31f3a9dcc24ee8e0377118af57441b2a938ab8850f521fb46befe8",
    "bias_act_ieee": "39c88afe2b81beeca53e8bfe9a0b17551c3285f731639cfc3138237f222204a2",
    "dequant": "3036481bb0a773cd7b6b75ae6de8edf6139fcc0a47f01a1aacd43a6d46165969",
    "masked": "d666b269599e0b4456d96aafaaebc01c4fee557fe13b29aab87e8cafc1d9d6a0",
    "modes": "16287d0047efcb06e7a3d8f67544339f527ea8671c814989bb1e84eca28e52d3",
    "periodic": "8ee6af059e2572861f595b7ee45c71df6d0fd1e7365ee3a926d6ade2ca3c9d0e",
    "residual": "94c94bcaa359e8d6d151d7cee9efa2de47420d773f7127e86573827a8dcb3ea6",
    "rms": "ba6df555c739da36354cf2f45392318a3d5fa6f0191ddfb096c2da0969d65b80",
    "swiglu": "224019ec9fc75e0d7ebf7d32ad5993d6d6e79a14d9e69eaa111724ea51a2d2f6",
    "views": "a15de9b99e0ffa6b44430aef66628e3e4c816d4c817de877fef40172018c0630",
}


def as_version_1(g):
    s = _fusion.settings_of(g)
    return tilefish.graph(g.fn, name=g.name, fusion=tilefish.fusion(version=1, grouping=s.grouping, math=s.math, intermediates=s.intermediates,
                                                                    fp_contract=s.fp_contract))


@pytest.mark.parametrize("name", sorted(V1_DIGESTS))
def test_version_1_plans_what_it_planned(name):
    """D7: a later tilefish keeps every earlier version. The use cases declared with fusion(version=1) plan as tilefish 0.0.9 and 0.0.10 planned them (their
    default was version 1): the same regions, sources, names and plan digests."""
    assert tilefish.plan(as_version_1(fg.CASES[name][0])).digest == V1_DIGESTS[name]


def test_a_plan_of_another_version_is_refused_naming_it(tmp_path):
    """D7: a plan made with another fusion version is refused as stale, with a message that names the version (a plan exported when version 1
    was the default, loaded by a graph that declares nothing)."""
    import plan_fakes

    from tilefish import _plan
    from tilefish.errors import TilefishManifestError

    old = as_version_1(fg.bias_act)
    exported = plan_fakes.export(tilefish.plan(old), tmp_path / "p")
    with pytest.raises(TilefishManifestError, match=r"made with tilefish.fusion\(version=1\) and the graph is planned with version 2 \(the default\)"):
        _plan.load_plan(tmp_path / "p", fg.bias_act, pin=exported.digest)
    assert _fusion.settings_of(fg.bias_act).version == 2 and _fusion.fused(fg.bias_act).settings.version == 2  # the check changed nothing of the graph
    assert _plan.load_plan(tmp_path / "p", old, pin=exported.digest).digest == exported.digest  # declared: it is that graph's plan

    @tilefish.graph(name="bias_act")
    def changed(x: Tensor[R, D, bf16], b: Tensor[D, bf16]):
        return tilefish.silu(x.to(f32) - b.to(f32)).to(bf16)

    with pytest.raises(TilefishManifestError, match="changed since the plan was exported"):  # not a matter of versions: no version named
        _plan.load_plan(tmp_path / "p", changed, pin=exported.digest)


def test_hand_written_kernels_keep_their_sources():
    """Revision 2 of the lowering is the regions' of version 2 only: a kernel's own source (revision 1) is what it was (0.0.10)."""
    import hashlib

    import primitive_cases as pc

    want = {
        "sum_sequential": "07f04cce15ce149fe69fd9789bd9faf072de80a5a3aaba2b15d4c7d5c6881373",
        "sumsq_shfl128v8": "bb6a5fc8ade0428ddc58c55d30b2d17707df6450f9c65641833e2fee33608412",
        "dot_halving64": "356954f61252e522b1f44f11414c1185519587a430a3c0e20357dd248bb36188",
        "rmsnorm_ordered": "7a24ae9addc61a9c84befaaa849a4ee4232bfb298d9af58abf83bb73fb18f805",
    }
    for name, sha in want.items():
        k = getattr(pc, name)
        assert k.lowering == 1 and hashlib.sha256(lower(k).source.encode()).hexdigest() == sha, name
        assert "revision 2" not in lower(k).source and "revision 2" in lower(k, lowering=2).source
    with pytest.raises(tilefish.TilefishLoweringError, match="lowering must be one of"):
        lower(pc.sum_sequential, lowering=3)

    @tilefish.kernel(lowering=2)  # a hand-written kernel asks for revision 2: the same bytes, other code
    def rev2_rows(x: Tensor[R, D, f32]) -> Tensor[R, f32]:
        return tilefish.sum(x * x, -1, order=fg.T128, fma=True)

    assert rev2_rows.lowering == 2 and "_split(" in rev2_rows.lower().source and rev2_rows.unaligned("x").lowering == 2
    with pytest.raises(tilefish.TilefishDeclarationError, match="lowering must be 1 or 2"):
        tilefish.kernel(lowering=3)


@pytest.mark.parametrize("name", fg.ROW_CASES)
def test_a_row_pattern_is_one_kernel_of_its_ops(name):
    fused, unfused = fg.CASES[name]
    f = _fusion.fused(fused)
    (r,) = f.regions
    assert r.kernel.lowering == 2 and r.kernel.context_free and r.kernel.lower().template == "row"
    assert len(r.outputs) == len(fused.signature.returns) if fused.signature.returns else True
    assert [x["kind"] for x in steps(fused)] == ["kernel"]
    u = _fusion.fused(unfused)  # one kernel per op (an fma sum with its product; a comparison with no tensor of its shape with its consumer)
    assert len(u.regions) >= len(r.nodes) - 2 and all(len(x.nodes) <= 2 for x in u.regions)
    assert not _fusion.boundaries(fused)  # nothing left that a later phase would fuse


def test_the_residual_sum_is_a_second_output_of_the_norm():
    f = _fusion.fused(fg.add_rmsnorm)
    (r,) = f.regions
    assert [o.type.dtype for o in r.outputs] == [bf16, bf16] and len(steps(fg.add_rmsnorm, R=8, DR=96)) == 1
    v1 = _fusion.fused(fg.add_rmsnorm_v1)
    assert [ops_of(x, v1) for x in v1.regions] == [["add", "cast", "mul"], ["mean"], ["add", "rsqrt", "mul", "cast", "mul", "cast"]]
    assert {b["reason"] for b in _fusion.boundaries(fg.add_rmsnorm_v1)} == {"reduction"}
    fix = next(b["fix"] for b in _fusion.boundaries(fg.add_rmsnorm_v1))
    assert "version 2 (the default) fuses ordered sums" in fix


def test_an_fma_sum_and_its_product_are_one_unit_in_every_grouping():
    f = _fusion.fused(fg.rmsnorm_fma_unfused)
    assert ["mul", "sum"] in [ops_of(r, f) for r in f.regions]  # the multiply is the sum's (each step fma(v, v, acc)): never rounded on its own

    @tilefish.graph(fusion=fg.V1)
    def fma_v1(x: Tensor[R, fg.DS, bf16], w: Tensor[fg.DS, bf16]):
        return fg._rmsnorm_fma(x, w)

    g = _fusion.fused(fma_v1)  # version 1 runs the sum alone: with its product
    assert ["mul", "sum"] in [ops_of(r, g) for r in g.regions]
    assert len(steps(fma_v1, R=8)) == 3
    reasons = {b["reason"] for b in _fusion.boundaries(fma_v1)}
    assert "reduction" in reasons and "grouping" not in reasons  # an edge into the product is an edge into the sum's region


def test_a_row_region_reads_views_where_they_are():
    s = steps(fg.split_norm, R=8, DR=96)
    assert [x["entry"].split("_")[0] for x in s] == ["fused"]  # no tilefish.contiguous copy
    assert [a.get("view", {}).get("strides") for a in s[0]["args"]][:2] == [[192, 1], [192, 1]]
    k = _fusion.fused(fg.split_norm).regions[0].kernel
    src = k.lower().source
    assert "tl.load(a0 + (row * a0_s0 + cols * a0_s1), mask=mask, other=0)" in src
    assert "'a0' is a view read where it is: row r, column c at r * a0_s0 + c * a0_s1 (note 21)" in src

    def half_mean(x):
        g, _ = x.split(fg.DR, axis=-1)
        return tilefish.mean(g, -1, keepdims=True, order=fg.T128) * 2.0

    @tilefish.graph(fusion=fg.V1)
    def half_mean_v1(x: Tensor[R, 2 * fg.DR, f32]):
        return half_mean(x)

    @tilefish.graph
    def half_mean_v2(x: Tensor[R, 2 * fg.DR, f32]):
        return half_mean(x)

    # version 1: the mean is a region of its own, of the row template, which reads no view where it is: a copy first
    assert [x["entry"].split("_")[0] for x in steps(half_mean_v1, R=8, DR=96)] == ["contiguous", "fused", "fused"]
    assert {b["reason"] for b in _fusion.boundaries(half_mean_v1)} >= {"materialised", "reduction"}
    assert [x["entry"].split("_")[0] for x in steps(half_mean_v2, R=8, DR=96)] == ["fused"]
    assert not _fusion.boundaries(half_mean_v2)


def test_a_negative_zero_scalar_is_written_by_its_bits_in_every_revision():
    """Triton 3.5.1 writes a literal -0.0 as +0.0 (note 3): every kernel writes a -0.0 scalar by its bits; a +0.0 one stays a literal."""

    @tilefish.kernel
    def nz(x: Tensor[R, D, f32], h: Tensor[R, D, bf16]) -> tuple[Tensor[R, D, f32], Tensor[R, D, bf16]]:
        return tilefish.where(x > 0.0, x, -0.0) * 1.0, h * -0.0 + 0.0

    for lowering in (1, 2):
        src = lower(nz, lowering=lowering).source
        assert "tl.full((), -2147483648, tl.int32).to(tl.float32, bitcast=True)" in src
        assert "tl.full((), -32768, tl.int16).to(tl.bfloat16, bitcast=True)" in src
        assert "-0.0" not in src.split("def tf_")[1]


def test_a_view_gathered_from_is_copied_first():
    """A gather addresses its table's rows by an index, contiguously: a view it reads is copied by tilefish.contiguous first, in the row template
    (version 2) and in the elementwise one (both versions), even when the region also reads it as an operand."""

    @tilefish.graph
    def gather_half(x: Tensor[R, 2 * D, f32], idx: Tensor[R, i32]):
        h, _ = x.split(D, axis=-1)
        return tilefish.gather_rows(h, idx, oob="clamp") + h

    s = steps(gather_half, R=8)
    assert [x["entry"].split("_")[0] for x in s] == ["contiguous", "fused"]
    src = _fusion.fused(gather_half).regions[0].kernel.lower().source
    assert "_s0" not in src.split("def tf_")[1]

    @tilefish.graph
    def gather_half_buckets(x: Tensor[R, 2 * fg.DR, f32], idx: Tensor[R, i32]):
        h, _ = x.split(fg.DR, axis=-1)
        return tilefish.gather_rows(h, idx, oob="clamp") + h

    # a table of bucketed rows next to a bucketed row length: a binary per combination, each with the warps of its own row length
    low = _fusion.fused(gather_half_buckets).regions[0].kernel.lower()
    assert low.extern.bucket_num_warps[".DR2304.R24"] == low.extern.bucket_num_warps[".DR2304.R8"] and len(low.extern.bucket_num_warps) == 6
    assert len(steps(gather_half_buckets, R=24, DR=2304)) == 2

    for version in (1, 2):
        @tilefish.graph(fusion=tilefish.fusion(version=version))
        def gather_column(x: Tensor[N, 2, f32], idx: Tensor[N, i32]):
            return tilefish.gather_rows(x.select(1, 0), idx, oob="zero") * 2.0

        want = ["contiguous", "fused", "fused"] if version == 1 else ["contiguous", "fused"]  # (version 1: the gather alone)
        assert [x["entry"].split("_")[0] for x in steps(gather_column, N=1000)] == want, version


def test_a_region_is_the_hand_written_kernel_of_its_ops_lowered_by_revision_2():
    """O2 for row regions: the region's source is the source of a @tilefish.kernel of the same ops, lowered context-free by revision 2."""
    f = _fusion.fused(fg.add_rmsnorm)
    source = f.regions[0].kernel.lower().source

    @tilefish.kernel(fp_fusion=False)
    def hand_norm(x: Tensor[R, fg.DR, bf16], res: Tensor[R, fg.DR, bf16], w: Tensor[fg.DR, bf16]) -> tuple[Tensor[R, fg.DR, bf16], Tensor[R, fg.DR, bf16]]:
        y, h = fg._add_rmsnorm(x, res, w)
        return h, y  # (the region writes its outputs in the order of the ops that make them)

    def norm(s, name):
        s = re.sub(rf"\b(tf_)?{name}\b", lambda m: (m.group(1) or "") + "K", s)
        return re.sub(r"\ba2\b", "w", re.sub(r"\ba1\b", "res", re.sub(r"\ba0\b", "x", s)))

    assert norm(source, f.regions[0].kernel.name) == norm(lower(hand_norm, context_free=True, lowering=2).source, "hand_norm")


def test_revision_2_lowers_ordered_sums_by_splits_and_gives_a_row_of_them_its_own_warps():
    r = _fusion.fused(fg.add_rmsnorm).regions[0]
    low = r.kernel.lower()
    assert "_split(" in low.source and "_tf_chain(" in low.source and "tl.gather(p, tl.minimum(lane + 16, 127), 0)" in low.source
    assert low.extern.bucket_num_warps == {".DR96": 1, ".DR1000": 1, ".DR2304": 4}  # 1 warp up to BLOCK_D 2048, then one per 1024 columns
    assert "1.0 / D)" in low.source and "_tf_rcp" not in low.source  # mean(order=): the ordered sum times f32(1 / D), one mul.rn
    # a Triton tree reduction keeps the template's rule (its numerics): the region of the unordered sum
    tree = next(x for x in _fusion.fused(fg.rms).regions if [y for y in x.nodes if _fusion.fused(fg.rms).program[y].op == "sum"])
    assert tree.kernel.lowering == 2 and tree.kernel.lower().num_warps == 1  # BLOCK_D 64: the template's 256 lanes per warp
    q = _fusion.fused(fg.quantise).regions[0].kernel.lower().source  # an fp8 cast of a region of version 2 is inline cvt
    assert "cvt.rn.satfinite.e4m3x2.f32 $0, 0f00000000, $1;" in q and ".to(tl.float8e4nv)" not in q


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

    # a per-column scale of a view: read where it is by both versions (a row region of version 1 would copy it: see the row regions below)
    def half(x, w):
        g, _ = x.split(D, axis=-1)
        return g * w + 1.0

    @tilefish.graph(fusion=fg.V1)
    def scaled_half(x: Tensor[R, 2 * D, f32], w: Tensor[D, f32]):
        return half(x, w)

    @tilefish.graph
    def scaled_half_v2(x: Tensor[R, 2 * D, f32], w: Tensor[D, f32]):
        return half(x, w)

    rep = tilefish.fusion_report(scaled_half)  # version 1: the elementwise template, broadcasting w, reads the view
    assert rep["regions"][0]["template"] == "elementwise" and rep["regions"][0]["inputs"][0]["read"] == "strided"
    rep2 = tilefish.fusion_report(scaled_half_v2)  # version 2: the row template reads it where it is
    assert rep2["regions"][0]["template"] == "row" and rep2["regions"][0]["inputs"][0]["read"] == "strided"
    assert [x["entry"].split("_")[0] for x in steps(scaled_half_v2, R=8)] == ["fused"]

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
    opaque = lower(k, context_free=True, lowering=2).source  # (a region of version 2, the default, is lowered by revision 2)
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
