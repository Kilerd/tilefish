"""Epilogue hooks on externs (``docs/rfcs/0003-epilogue-fusion.md``, milestone E1, :mod:`tilefish._epilogue`) without a GPU: the declaration and
its refusals; the generated module an extern with hooks is launched through; the planner's merge of a region into a hooked call (one step) and its
refusals, each with its reason; ``R_acc`` on the IR; ``tilefish.fused`` scopes; names and identity; the hook's code against the region's; the fusion
report, the boundaries and the advisor's routes.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import hashlib
import json
import re

import epilogue_graphs as eg
import pytest

import tilefish
from tilefish import Dim, Out, Tensor, TilefishDeclarationError, TilefishPlanError, bf16, const, f32, i32
from tilefish import _advise, _epilogue, _fusion
from tilefish.aot import compile_specs, declaration_digest

GEMM, DUAL, CONV = eg.standins()
G = eg.build((GEMM, DUAL, CONV), prefix="t")
T, H, F, D = eg.T, eg.H, eg.F, eg.D
M, N, K = eg.M, eg.N, eg.K


def steps(graph, **bucket):
    p = tilefish.plan(graph)
    return [s.get("entry") for s in (p.steps(**bucket) if bucket else p._doc["buckets"][0]["steps"])]


# -- the declaration --------------------------------------------------------------------------------------------------------------------


def test_a_hook_is_checked_where_it_is_declared():
    h = tilefish.hook("EPILOGUE", value="out", out="c")
    assert (h.name, h.args_name, h.value, h.out, h.split, h.block_n) == ("EPILOGUE", "EPILOGUE_ARGS", "out", "c", 1, None)
    assert h.canonical() == {"name": "EPILOGUE", "value": "out", "out": "c", "split": 1}
    for bad, match in (
        (dict(name="1x", out="c"), "identifier"),
        (dict(name="E", value="rounded", out="c"), "value must be"),
        (dict(name="E", out=3), "out= names"),
        (dict(name="E", out="c", split=0), "split must be"),
        (dict(name="E", out="c", split=True), "split must be"),
        (dict(name="E", out="c", block_n=4), "block_n="),
        (dict(name="E", out="c", reference=lambda a: a), "value='acc'"),
    ):
        with pytest.raises(TilefishDeclarationError, match=match):
            tilefish.hook(**bad)


def _declare(epilogue, stub=None, **kw):
    def gemm(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16], c: Out[Tensor[M, N, bf16]], m: M, BM: const = 128): ...

    return tilefish.extern(triton="mod:kernel", grid=lambda M: (M,), epilogue=epilogue, **kw)(stub or gemm)


class Kernel:
    """A kernel object that says its parameters (as a triton.JITFunction does), for the checks against the stub."""

    def __init__(self, names, constexpr):
        self.__module__, self.__name__ = "mod", "kernel"
        self.arg_names = list(names)
        self.params = [type("P", (), {"is_constexpr": c, "name": n})() for n, c in zip(names, constexpr)]

    def is_gluon(self):
        return False


def test_an_extern_with_hooks_is_refused_where_its_hooks_cannot_be():
    h = tilefish.hook("EPILOGUE", out="c")
    with pytest.raises(TilefishDeclarationError, match="epilogue= takes a tilefish.hook"):
        _declare("EPILOGUE")
    with pytest.raises(TilefishDeclarationError, match="at most one hook of each value"):
        _declare((h, tilefish.hook("OTHER", out="c")))
    with pytest.raises(TilefishDeclarationError, match="the same out= and split="):
        _declare((h, tilefish.hook("ACC", value="acc", out="c", split=2)))
    with pytest.raises(TilefishDeclarationError, match="not an Out parameter"):
        _declare(tilefish.hook("EPILOGUE", out="a"))

    def ints(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16], c: Out[Tensor[M, N, i32]], m: M): ...

    with pytest.raises(TilefishDeclarationError, match="a float tensor"):
        _declare(h, ints)

    def listed(a: Tensor[M, K, bf16], c: Out[Tensor[M, N, bf16]], m: M, n: N, EPILOGUE: const = 0): ...

    with pytest.raises(TilefishDeclarationError, match="the stub leaves the hook's two parameters out"):
        _declare(h, listed)

    def reserved(tl: Tensor[M, K, bf16], c: Out[Tensor[M, N, bf16]], m: M, n: N): ...

    with pytest.raises(TilefishDeclarationError, match="has a name the generated module"):
        _declare(h, reserved)
    with pytest.raises(TilefishDeclarationError, match="block_n="):
        _declare(tilefish.hook("EPILOGUE", out="c", block_n="BN"))
    Q = Dim("Q")

    def odd(a: Tensor[M, K, bf16], b: Tensor[Q, K, bf16], c: Out[Tensor[M, Q, bf16]], m: M): ...

    with pytest.raises(TilefishDeclarationError, match="not provably divisible by 2"):
        _declare(tilefish.hook("EPILOGUE", out="c", split=2), odd)
    with pytest.raises(TilefishDeclarationError, match="a prebuilt binary has none"):
        tilefish.extern(ptx="x.ptx", symbol="k", block=(128,), grid=(1,), epilogue=h)
    with pytest.raises(TilefishDeclarationError, match="by_arch= does not take one"):
        tilefish.extern(by_arch={"sm_103": tilefish.impl(triton="mod:kernel")}, grid=(1,), epilogue=h)


def test_the_stub_lists_the_kernels_parameters_but_the_hooks_two():
    names = ["a", "b", "c", "m", "BM", "EPILOGUE", "EPILOGUE_ARGS"]
    flags = [False, False, False, False, True, True, False]
    h = tilefish.hook("EPILOGUE", out="c")

    def gemm(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16], c: Out[Tensor[M, N, bf16]], m: M, BM: const = 128): ...

    ok = tilefish.extern(triton=Kernel(names, flags), grid=lambda M: (M,), epilogue=h)(gemm)
    assert ok.hooks == (h,) and ok.kernel_ref == "mod:kernel"
    with pytest.raises(TilefishDeclarationError, match="no parameter 'EPILOGUE_ARGS'"):
        tilefish.extern(triton=Kernel(names[:-1], flags[:-1]), grid=lambda M: (M,), epilogue=h)(gemm)
    with pytest.raises(TilefishDeclarationError, match="are not the kernel's"):
        tilefish.extern(triton=Kernel(["b", "a", *names[2:]], flags), grid=lambda M: (M,), epilogue=h)(gemm)
    with pytest.raises(TilefishDeclarationError, match="not a constexpr"):
        tilefish.extern(triton=Kernel(names, [False] * 7), grid=lambda M: (M,), epilogue=h)(gemm)


def test_an_extern_with_hooks_is_launched_through_a_generated_module_that_calls_its_kernel_with_the_identity_hook():
    source = GEMM.__dict__["_tf_generated"].source
    assert GEMM.triton == "tilefish_generated.epi_gemm__hooks:tf_epi_gemm__hooks"
    assert GEMM.source_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert "from epilogue_gemm import gemm_kernel as _tf_kernel" in source
    assert "ACC=_tf_identity," in source and "EPILOGUE=_tf_identity," in source and "EPILOGUE_ARGS=()," in source
    assert re.search(r"def tf_epi_gemm__hooks\(\n    a,\n    b,\n    c,\n    m,\n    n,\n    k,\n    BM: tl.constexpr,", source)
    specs = compile_specs(GEMM)
    assert len(specs) == 1 and specs[0].triton == GEMM.triton and specs[0].source_sha256 == GEMM.source_sha256
    # the declaration digest has the module (so the kernel's reference and the hooks); the same stub without hooks is another declaration
    plain = tilefish.extern(triton="epilogue_gemm:gemm_kernel", name="epi_gemm", grid=GEMM._grid_source)(GEMM.fn)
    assert declaration_digest(plain) != declaration_digest(GEMM)
    assert not plain.hooks and plain.triton == "epilogue_gemm:gemm_kernel"  # (an extern without a hook is as it was)
    assert "@gluon.jit" not in source and "import gluon" not in source


# -- the merge ------------------------------------------------------------------------------------------------------------------------


def test_a_region_after_a_hooked_extern_is_one_step_with_it():
    f = _fusion.fused(G["bias_gelu"]["fused"])
    (h,) = f.hooked
    assert h.contract == "exact" and h.hook.value == "out" and h.op.name.startswith("epi_gemm_epi_") and re.fullmatch(r"epi_gemm_epi_[0-9a-f]{12}", h.op.name)
    assert steps(G["bias_gelu"]["fused"]) == [h.op.name]
    assert steps(G["bias_gelu"]["unfused"])[0] == "epi_gemm" and len(steps(G["bias_gelu"]["unfused"])) == 6  # the extern, then each of 5 ops alone
    node = f.graph.nodes[0]
    assert node.callee is h.op and node.attrs["kind"] == "extern" and len(f.graph.nodes) == 1
    # the fused op: the extern's parameters, the hook's arguments after them, the result written into the extern's output
    sig = h.op.signature
    assert [p.name for p in sig.params] == ["a", "b", "c", "m", "n", "k", "BM", "BN", "BK", "SPLIT", "GROUP_M", "EPI_SPLIT", "ep_a1"]
    assert sig.params[-1].keyword_only and str(sig.params[-1].type) == "Tensor[256, bf16]"
    source = h.op.__dict__["_tf_generated"].source
    assert "EPILOGUE=_tf_epilogue," in source and "EPILOGUE_ARGS=(ep_a1,)," in source and "ACC=_tf_identity," in source
    assert "def _tf_epilogue(tile, rows, cols, row_mask, col_mask, args):" in source
    assert "offs = rows[:, None] * 256 + cols[None, :]" in source and "return result" in source
    p = tilefish.plan(G["bias_gelu"]["fused"])
    (st,) = p.steps(T=64)
    assert [a["name"] for a in st["args"]] == ["a", "b", "c", "ep_a1"]


def test_the_hooks_code_is_the_regions_op_by_op():
    """Obligation O2 of RFC 0001: the hook writes each op as the region's kernel does (context-free, no contraction): the same expression of every op."""
    f = _fusion.fused(G["periodic"]["fused"])
    (h,) = f.hooked
    hook = h.op.__dict__["_tf_generated"].source
    region = h.region.kernel.lower().source

    def exprs(src):
        return sorted(re.sub(r"v\d+", "v", line.split("=", 1)[1].split("#")[0].strip()) for line in src.splitlines() if re.match(r"\s+v\d+ = ", line))

    ops = [e for e in exprs(region) if not e.startswith("tl.load(")]
    got = [e for e in exprs(hook) if not e.startswith("tl.load(") and e != "tile"]
    assert ops and sorted(ops) == sorted(got)


def test_the_same_region_after_the_same_extern_in_two_graphs_is_one_op():
    again = eg.build((GEMM, DUAL, CONV), prefix="t2")
    a = _fusion.fused(G["bias_silu"]["fused"]).hooked[0].op
    b = _fusion.fused(again["bias_silu"]["fused"]).hooked[0].op
    assert a is b
    other = _fusion.fused(G["bias_gelu"]["fused"]).hooked[0].op
    assert other.name != a.name


def test_every_pattern_of_e1_fuses_into_one_step_in_both_modes():
    for name, graphs in G.items():
        for mode in ("fused", "acc"):
            f = _fusion.fused(graphs[mode])
            assert len(f.hooked) == 1 and not f.refused, (name, mode, f.refused)
            assert len(steps(graphs[mode])) == 1, (name, mode)
            contract = f.hooked[0].contract
            widened = name not in ("add_norm",)  # add_norm adds the result in bf16: nothing to rewrite, its hook is the out one in both modes
            assert contract == ("accumulator" if mode == "acc" and widened else "exact"), (name, mode)
        assert len(steps(graphs["unfused"])) > 2 and len(steps(graphs["acc_unfused"])) > 2


def test_a_dual_accumulator_hook_holds_both_chunks_and_stores_one():
    f = _fusion.fused(G["swiglu"]["fused"])
    (h,) = f.hooked
    assert sorted(h.sources.values(), key=str) == [("tile", 0), ("tile", 1)]
    c = next(p for p in h.op.signature.params if p.name == "c")
    assert [str(d) for d in c.type.shape] == ["M", "256"]  # the output is one chunk: the kernel stores the one tile the hook returns
    assert "return (result,)" in h.op.__dict__["_tf_generated"].source
    assert len(h.views) == 1  # the split of the result: in the unit, no node of the plan's graph


def test_a_norm_fuses_where_a_tile_holds_whole_rows_and_its_residual_is_a_side_output():
    f = _fusion.fused(G["add_norm"]["fused"])
    (h,) = f.hooked
    assert h.op.__dict__["_tf_generated"].template == "hook-row"
    src = h.op.__dict__["_tf_generated"].source
    assert "_tf_hsum_l128_v8_o64_32_16_8_4_2_1(" in src and "tl.store(out1 + offs" in src  # the ordered mean, the residual stream stored
    assert len(h.outputs) == 2 and h.main is not None
    narrow = eg.standin_gemm("epi_gemm_narrow", BN=64)
    body, params = eg.bodies(narrow, DUAL, CONV)["add_norm"]
    g = eg.graph_of(body, params, tilefish.fusion(), "t_norm_narrow")
    f = _fusion.fused(g)
    assert not f.hooked and f.refused[0]["reason"] == "epilogue-row" and "64 columns wide" in f.refused[0]["detail"]


# -- R_acc and the modes ----------------------------------------------------------------------------------------------------------------


def test_r_acc_replaces_every_widening_of_the_result_by_the_f32_result():
    f = _fusion.fused(G["periodic"]["acc_unfused"])  # (the unfused composition of R_acc(P): the accumulator form, then each op)
    call = next(n for n in f.program if n.attrs.get("kind") == "extern")
    assert call.op == "epi_gemm__acc" and call.callee.acc_of == (GEMM, GEMM.hooks[0])
    y, acc = call.outputs
    assert str(acc.type) == "Tensor[T, F, f32]"
    assert not any(n.op == "cast" and n.inputs[0] is y for n in f.program if n.attrs.get("kind") is None and n.outputs and any(c for c in f.program if any(i is n.outputs[0] for i in c.inputs)))
    assert f.rewrites == [{"kind": "accumulator", "node": 0, "op": "epi_gemm", "value": y.ref, "dtype": "bf16", "widenings": [1]}]
    assert steps(G["periodic"]["acc_unfused"])[0] == "epi_gemm__acc"
    form = call.callee
    assert [p.name for p in form.signature.tensor_outputs] == ["c", "ep_acc"]
    assert "tl.store(acc + offs, tile, mask=mask)" in form.__dict__["_tf_generated"].source


def test_r_acc_keeps_what_reads_the_result_in_its_own_type():
    @tilefish.graph(fusion=tilefish.fusion(epilogues="accumulator"))
    def mixed(x: Tensor[T, H, bf16], w: Tensor[F, H, bf16], b: Tensor[F, bf16]):
        y = GEMM(x, w)
        return ((y.to(f32) + b.to(f32)) * (y * b).to(f32)).to(bf16)  # y * b reads y rounded, as the kernel rounds it

    f = _fusion.fused(mixed)
    (h,) = f.hooked
    assert h.contract == "accumulator" and h.main is not None
    assert sorted(h.sources.values(), key=str) == [("rounded", None), ("tile", None)]
    src = h.op.__dict__["_tf_generated"].source
    assert "= tile.to(tl.bfloat16)" in src and "= tile  #" in src  # the accumulator, and the result rounded as the kernel rounds it


def test_a_scope_declares_the_mode_of_the_ops_traced_in_it():
    @tilefish.graph
    def scoped(x: Tensor[T, H, bf16], w: Tensor[F, H, bf16], b: Tensor[F, bf16]):
        y = GEMM(x, w)
        with tilefish.fused(epilogues="accumulator"):
            h = y.to(f32) + b.to(f32)
        return tilefish.silu(h).to(bf16)

    @tilefish.graph(fusion=tilefish.fusion(epilogues="accumulator"))
    def exact_scope(x: Tensor[T, H, bf16], w: Tensor[F, H, bf16], b: Tensor[F, bf16]):
        y = GEMM(x, w)
        with tilefish.fused(epilogues="exact"):
            h = y.to(f32) + b.to(f32)
        return tilefish.silu(h).to(bf16)

    assert _fusion.fused(scoped).hooked[0].contract == "accumulator"
    assert _fusion.fused(exact_scope).hooked[0].contract == "exact" and not _fusion.fused(exact_scope).rewrites
    with pytest.raises(TilefishPlanError, match="a scope of a @tilefish.graph body"):
        with tilefish.fused(epilogues="accumulator"):
            pass
    with pytest.raises(TilefishPlanError, match="epilogues must be one of"):
        tilefish.fused(epilogues="library")
    with pytest.raises(TilefishPlanError, match="name= ..* not planned yet|say what the scope sets"):
        tilefish.fused()
    with pytest.raises(TilefishPlanError, match="epilogues must be one of"):
        tilefish.fusion(epilogues="library")
    assert tilefish.fusion(epilogues="accumulator").canonical() == tilefish.fusion().canonical()  # (a mode changes nodes, not a region's name)


def test_an_extern_that_hooks_its_accumulator_only_is_not_fused_in_the_exact_mode():
    def gemm(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16], c: Out[Tensor[M, N, bf16]], m: M): ...

    acc_only = tilefish.extern(triton="mod:kernel", name="acc_only", grid=lambda M: (M,), epilogue=tilefish.hook("ACC", value="acc", out="c"))(gemm)

    def body(x, w, b):
        return tilefish.silu(acc_only(x, w).to(f32) + b.to(f32)).to(bf16)

    params = [("x", Tensor[T, H, bf16]), ("w", Tensor[F, H, bf16]), ("b", Tensor[F, bf16])]
    exact = eg.graph_of(body, params, tilefish.fusion(), "t_acc_only_exact")
    acc = eg.graph_of(body, params, tilefish.fusion(epilogues="accumulator"), "t_acc_only_acc")
    f = _fusion.fused(exact)
    assert not f.hooked and f.refused[0]["reason"] == "epilogue-acc-only"
    assert _fusion.fused(acc).hooked[0].contract == "accumulator"
    (b,) = [b for b in _fusion.boundaries(exact) if b["reason"].startswith("epilogue")]
    assert b["reason"] == "epilogue-acc-only" and "epilogues='accumulator'" in b["fix"]


# -- refusals ---------------------------------------------------------------------------------------------------------------------------


def test_what_a_hook_does_not_take_is_refused_with_its_reason():
    def plan_of(body):
        g = eg.graph_of(body, [("x", Tensor[T, H, bf16]), ("w", Tensor[F, H, bf16]), ("b", Tensor[F, bf16])], tilefish.fusion(), f"t_ref_{len(seen)}")
        seen.append(g)
        return _fusion.fused(g)

    seen = []
    # a barrier keeps the result stored and stops its epilogue fusion
    f = plan_of(lambda x, w, b: (tilefish.barrier(GEMM(x, w)).to(f32) * 2.0).to(bf16))
    assert not f.hooked and f.refused[0]["reason"] == "barrier"
    # a view of the result the hook does not hold
    f = plan_of(lambda x, w, b: (GEMM(x, w).reshape(T, 2, 128).to(f32) * 2.0).to(bf16))
    assert not f.hooked and f.refused[0]["reason"] == "epilogue-view"
    # a row region over rows wider than a tile of the extern
    f = plan_of(lambda x, w, b: tilefish.max(GEMM(x, w).to(f32), -1, keepdims=True))
    assert not f.hooked and f.refused[0]["reason"] == "epilogue-row"
    # a region that writes nothing of the result's shape and type: the kernel stores the result all the same, into a buffer nothing reads
    f = plan_of(lambda x, w, b: GEMM(x, w).to(f32) * 2.0)
    (h,) = f.hooked
    assert h.main is None and not h.result_read and len(h.outputs) == 2
    assert "nothing reads" in tilefish.fusion_report(seen[-1])["regions"][0]["stored"]
    # two regions read the result: the first is fused, the other told why not
    f = plan_of(_two)
    assert len(f.hooked) == 1 and [r["reason"] for r in f.refused] == ["epilogue-taken"]
    # a call between the extern and the region (it reads the result and feeds the region): one kernel would wait for itself; the region fuses into
    # the second call's hook instead, and reads the first's result from memory
    f = plan_of(lambda x, w, b: _cycle(x, w, b))
    assert [r["reason"] for r in f.refused] == ["epilogue-cycle"] and f.refused[0]["call"] == 0
    assert len(f.hooked) == 1 and f.hooked[0].call == 1


def _two(x, w, b):
    y = GEMM(x, w)
    return (y.to(f32) + b.to(f32)).to(bf16), tilefish.barrier(y.to(f32) * 3.0).to(bf16)


def _cycle(x, w, b):
    y = GEMM(x, w)
    z = GEMM(y, w)  # another call reads y
    return (y.to(f32) + z.to(f32)).to(bf16)


# -- the report, the boundaries, the advisor --------------------------------------------------------------------------------------------


def test_the_report_says_the_route_the_contract_and_the_roundings_removed():
    rep = tilefish.fusion_report(G["periodic"]["acc"])
    (r,) = rep["regions"]
    assert r["route"] == "hook" and r["value"] == "acc" and r["contract"] == "accumulator" and r["call"] == {"op": "epi_gemm", "node": 0}
    assert r["op"].startswith("epi_gemm_epi_") and r["region"].startswith("fused_") and r["template"] == "hook" and r["exact"] is False
    assert r["rounding_points_removed"] == [{"value": "%0", "dtype": "bf16", "widenings": [1], "reason": "R_acc: widened by the epilogue"}]
    assert rep["fusion"]["epilogues"] == "accumulator" and set(rep["hooks"]) == {"epi_gemm"}
    exact = tilefish.fusion_report(G["periodic"]["fused"])
    assert exact["regions"][0]["contract"] == "exact" and exact["regions"][0]["exact"] is True and "epilogues" not in exact["fusion"]
    # a graph that calls no extern with a hook has the report it always had
    import fusion_graphs as fg

    assert set(tilefish.fusion_report(fg.bias_act)) == {"schema", "graph", "fusion", "regions"}
    assert all("route" not in r for r in tilefish.fusion_report(fg.bias_act)["regions"])


def test_the_boundaries_have_no_entry_for_a_fused_pair_and_say_why_another_is_not():
    assert not [b for b in _fusion.boundaries(G["bias_gelu"]["fused"]) if b["reason"].startswith("epilogue")]
    unfused = _fusion.boundaries(G["bias_gelu"]["unfused"])
    assert unfused and {b["reason"] for b in unfused} == {"unfused"}  # (grouping='none' says so: the extern as it is, every op alone)

    def plain(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16], c: Out[Tensor[M, N, bf16]], m: M): ...

    nohook = tilefish.extern(triton="mod:kernel", name="nohook", grid=lambda M: (M,))(plain)
    g = eg.graph_of(lambda x, w, b: tilefish.silu(nohook(x, w).to(f32) + b.to(f32)).to(bf16), [("x", Tensor[T, H, bf16]), ("w", Tensor[F, H, bf16]), ("b", Tensor[F, bf16])],
                    tilefish.fusion(), "t_nohook")  # fmt: skip
    (b,) = _fusion.boundaries(g)
    assert b["reason"] == "epilogue-extern" and "declares no epilogue hook" in b["detail"] and "tilefish.hook" in b["fix"] and b["phase"] == "4"


def _doc_of(steps_, buffers):
    return {"graph": "g", "arch": "sm_103", "buckets": [{"values": {"T": 64}, "steps": steps_}], "buffers": buffers, "dims": {"T": [64]}}


def test_the_advisor_reports_an_extern_with_hooks_as_actionable_with_the_contract_of_each_route():
    rows = 64 * 256 * 2
    buf = lambda n, role, shape: {"name": n, "role": role, "dtype": "bf16", "shape": shape}  # noqa: E731
    buffers = [buf("x", "input", [64, 256]), buf("w", "weight", [256, 256]), buf("t0", "intermediate", [64, 256]), buf("out0", "output", [64, 256])]
    gemm_step = {"kind": "kernel", "entry": "epi_gemm", "args": [{"name": "a", "binding": "x"}, {"name": "b", "binding": "w"}, {"name": "c", "binding": "t0"}]}
    act_step = {"kind": "kernel", "entry": "act", "args": [{"name": "x", "binding": "t0"}, {"name": "out", "binding": "out0"}]}
    abi = lambda *ps: {"params": [{"name": n, "kind": "pointer", "access": a} for n, a in ps]}  # noqa: E731
    manifest = {"kernels": [
        {"stem": "epi_gemm", "op": "epi_gemm", "arch": "sm_103", "source_sha256": "0" * 64, "constexprs": [{"name": "BM"}], "abi": abi(("a", "in"), ("b", "in"), ("c", "out"))},
        {"stem": "act", "op": "act", "arch": "sm_103", "source_sha256": "1" * 64, "constexprs": [{"name": "BLOCK"}], "abi": abi(("x", "in"), ("out", "out")),
         "options": {"enable_fp_fusion": False}},
    ], "library": []}  # fmt: skip
    report = {"fusion": {"version": 2}, "regions": [], "hooks": {"epi_gemm": [{"value": "acc"}, {"value": "out"}]}}
    adv = _advise.analyse(_doc_of([gemm_step, act_step], buffers), manifest, report=report)
    s0 = adv.doc["steps"][0]
    assert s0["class"] == "opaque" and "epilogue hooks: acc, out" in s0["label"]
    (c,) = adv.candidates
    assert c["blockers"] == ["epilogue-hook-consumer", "dsl-kernel"] and c["phase"] == "4" and c["saved_bytes"]["T=64"] == 2 * rows
    routes = {r["value"]: r for r in c["routes"]}
    assert routes["out"]["contract"].startswith("(i) exact") and routes["out"]["status"] == "available"
    assert routes["acc"]["contract"].startswith("(ii)") and "epilogues='accumulator'" in routes["acc"]["status"]
    text = adv.text()
    assert "route E1 hook (out): contract (i) exact" in text
    # no report: a generated module with no template's constexpr is an extern with hooks, of hooks unknown
    adv = _advise.analyse(_doc_of([gemm_step, act_step], buffers), manifest)
    assert adv.doc["steps"][0]["class"] == "opaque" and {r["value"] for r in adv.candidates[0]["routes"]} == {"out", "acc"}
    # a step that is an extern with a region in its hook
    report = {"fusion": {"version": 2}, "regions": [{"op": "epi_gemm_epi_0123456789ab", "route": "hook", "value": "out", "contract": "exact", "call": {"op": "epi_gemm"},
                                                     "nodes": [{"index": 1, "op": "cast"}]}]}  # fmt: skip
    manifest["kernels"][0]["op"] = manifest["kernels"][0]["stem"] = gemm_step["entry"] = "epi_gemm_epi_0123456789ab"
    adv = _advise.analyse(_doc_of([gemm_step, act_step], buffers), manifest, report=report)
    assert adv.doc["steps"][0]["class"] == "opaque" and "contract (i), exact" in adv.doc["steps"][0]["label"]
    assert "routes" not in adv.candidates[0]


def test_every_epilogue_reason_has_a_fix():
    for reason in ("epilogue-extern", "epilogue-acc-only", "epilogue-taken", "epilogue-view", "epilogue-acc-shared", "epilogue-cycle", "epilogue-row",
                   "epilogue-template", "epilogue-hook", "epilogue-dims", "epilogue-library"):
        fix, phase = _fusion.BOUNDARY_FIXES[reason]
        assert fix and phase in ("4", "future") and _advise.BLOCKERS[reason] == (fix, phase)
    assert "E2" in _fusion.BOUNDARY_FIXES["epilogue-library"][0] and "E3" in _fusion.BOUNDARY_FIXES["epilogue-library"][0]
    assert json.dumps(_epilogue.ABI)
