"""The three modes of ``epilogues=`` in one graph (``docs/en/rfcs/0003-epilogue-fusion.md``, "E1 and E2 together"), without a GPU: ``"exact"`` (the
default: extern hooks through ``value="out"``), ``"accumulator"`` (E1: ``R_acc`` and ``value="acc"`` hooks; a cuBLASLt GEMM left as it is, with a
note in the report) and ``"library"`` (E2: the cuBLASLt rewrite, and the extern hooks of ``"exact"``), per graph and per ``tilefish.fused`` scope, on
a graph that calls both a cuBLASLt GEMM and an extern with both hooks. ``test_gpu_epilogue.py`` runs the mixed graph.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import epilogue_graphs as eg
import pytest

import tilefish
from tilefish import Tensor, TilefishPlanError, bf16, f32
from tilefish import _fusion

T, H, F = eg.T, eg.H, eg.F
GEMM = eg.standin_gemm(name="modes_gemm")  # an extern with an ACC (value="acc") and an EPILOGUE (value="out") hook, never compiled here


@tilefish.library("modes_up", acc=f32, impl="cublaslt.gemm")
def up(a: Tensor[T, H, bf16], b: Tensor[F, H, bf16]) -> Tensor[T, F, bf16]: ...


PARAMS = [("x", Tensor[T, H, bf16]), ("w1", Tensor[F, H, bf16]), ("b1", Tensor[F, bf16]), ("w2", Tensor[F, F, bf16]), ("b2", Tensor[F, bf16])]


def plain(x, w1, b1, w2, b2):
    """A cuBLASLt GEMM and its bias (a cuBLASLt epilogue: BIAS), then the extern and a bias + SiLU (a region its hooks take)."""
    h = (up(x, w1).to(f32) + b1.to(f32)).to(bf16)
    return tilefish.silu(GEMM(h, w2).to(f32) + b2.to(f32)).to(bf16)


def extern_on_acc(x, w1, b1, w2, b2):
    """``plain`` with the extern's arithmetic in an ``"accumulator"`` scope."""
    h = (up(x, w1).to(f32) + b1.to(f32)).to(bf16)
    with tilefish.fused(epilogues="accumulator"):
        return tilefish.silu(GEMM(h, w2).to(f32) + b2.to(f32)).to(bf16)


def library_scope(x, w1, b1, w2, b2):
    """``plain`` with the cuBLASLt pattern in a ``"library"`` scope."""
    with tilefish.fused(epilogues="library"):
        h = (up(x, w1).to(f32) + b1.to(f32)).to(bf16)
    return tilefish.silu(GEMM(h, w2).to(f32) + b2.to(f32)).to(bf16)


def half_in_scope(x, w1, b1, w2, b2):
    """Only the widening of the GEMM's result in a ``"library"`` scope: the rest of the pattern is not."""
    with tilefish.fused(epilogues="library"):
        y = up(x, w1).to(f32)
    h = (y + b1.to(f32)).to(bf16)
    return tilefish.silu(GEMM(h, w2).to(f32) + b2.to(f32)).to(bf16)


def graph(body, mode, name, **kw):
    return eg.graph_of(body, PARAMS, tilefish.fusion(epilogues=mode, **kw), name)


def entries(g):
    return [s["entry"].split(".")[0] for s in tilefish.plan(g).steps(T=64)]


def library_epilogues(g):
    return _fusion.report(g).get("library_epilogues") or []


def test_each_mode_on_a_graph_of_a_cublaslt_gemm_and_a_hooked_extern():
    # exact: the GEMM and its region, the extern with the region in its value="out" hook; the report says what "library" would do
    g = graph(plain, "exact", "modes_exact")
    f = _fusion.fused(g)
    assert entries(g)[0] == "modes_up" and len(entries(g)) == 3 and entries(g)[2].startswith("modes_gemm_epi_")
    assert [h.contract for h in f.hooked] == ["exact"] and not f.rewrites
    (w,) = library_epilogues(g)
    assert (w["matched"], w["applied"], w["epilogue"]) == (True, False, "bias") and "mode" not in w

    # accumulator: R_acc and the value="acc" hook for the extern; the cuBLASLt GEMM left as it is, and the report says why
    g = graph(plain, "accumulator", "modes_acc")
    f = _fusion.fused(g)
    assert entries(g)[0] == "modes_up" and len(entries(g)) == 3
    assert [h.contract for h in f.hooked] == ["accumulator"] and [r["kind"] for r in f.rewrites] == ["accumulator"]
    (w,) = library_epilogues(g)
    assert (w["matched"], w["applied"], w["mode"]) == (True, False, "accumulator")
    assert "R_acc of a cuBLASLt GEMM is not planned" in w["note"] and "epilogues='library'" in w["note"]

    # library: the cuBLASLt rewrite, and the extern's value="out" hook as in "exact" (no R_acc)
    g = graph(plain, "library", "modes_library")
    f = _fusion.fused(g)
    assert entries(g)[0] == "modes_up__bias" and len(entries(g)) == 2 and entries(g)[1].startswith("modes_gemm_epi_")
    assert [h.contract for h in f.hooked] == ["exact"] and [r["kind"] for r in f.rewrites] == ["library-epilogue"]
    (w,) = library_epilogues(g)
    assert (w["matched"], w["applied"], w["into"]) == (True, True, "modes_up__bias")
    assert _fusion.report(g)["fusion"]["epilogues"] == "library" and "epilogues" not in _fusion.report(graph(plain, "exact", "modes_exact2"))["fusion"]


def test_a_graph_mixes_the_library_rewrite_and_an_extern_on_its_accumulator_through_a_scope():
    """Both ways round: the graph "library" with an "accumulator" scope around the extern's arithmetic, or the graph "accumulator" with a "library"
    scope around the cuBLASLt pattern: one cublaslt.matmul and the extern with the region in its value="acc" hook, the same plan."""
    plans = []
    for body, mode, name in ((extern_on_acc, "library", "modes_mix_a"), (library_scope, "accumulator", "modes_mix_b")):
        g = graph(body, mode, name)
        f = _fusion.fused(g)
        assert entries(g)[0] == "modes_up__bias" and len(entries(g)) == 2, (name, entries(g))
        assert [h.contract for h in f.hooked] == ["accumulator"], name
        assert sorted(r["kind"] for r in f.rewrites) == ["accumulator", "library-epilogue"], name
        (w,) = library_epilogues(g)
        assert w["applied"] and w["into"] == "modes_up__bias"
        plans.append([(s["entry"].split(".")[0].split("_epi_")[0], s["kind"]) for s in tilefish.plan(g).steps(T=200)])
    assert plans[0] == plans[1] == [("modes_up__bias", "library"), ("modes_gemm", "kernel")]
    # the unfused composition of the mixed program (what export checks the pair against) keeps both rewrites: they are numerics, not fusion
    u = _fusion.fused(graph(extern_on_acc, "library", "modes_mix_unfused", grouping="none"))
    assert sorted(r["kind"] for r in u.rewrites) == ["accumulator", "library-epilogue"] and not u.hooked


def test_a_cublaslt_pattern_is_rewritten_only_where_all_its_ops_are_in_a_library_mode():
    g = graph(half_in_scope, "exact", "modes_half")
    assert entries(g)[0] == "modes_up" and not any("__" in e for e in entries(g))
    (w,) = library_epilogues(g)
    assert (w["matched"], w["applied"], w["mode"]) == (True, False, "exact/library") and "only where all are in 'library'" in w["note"]
    # the region the cuBLASLt pattern left still goes into the extern's hook (exact)
    assert [h.contract for h in _fusion.fused(g).hooked] == ["exact"]


def test_an_acc_only_hook_names_the_mode_of_the_region_it_does_not_take():
    from tilefish import Out

    def gemm(a: Tensor[eg.M, eg.K, bf16], b: Tensor[eg.N, eg.K, bf16], c: Out[Tensor[eg.M, eg.N, bf16]], m: eg.M): ...

    acc_only = tilefish.extern(triton="mod:kernel", name="modes_acc_only", grid=lambda M: (M,), epilogue=tilefish.hook("ACC", value="acc", out="c"))(gemm)

    def body(x, w1, b1, w2, b2):
        return tilefish.silu(acc_only(x, w1).to(f32) + b1.to(f32)).to(bf16)

    f = _fusion.fused(graph(body, "library", "modes_acc_only_library"))
    (r,) = f.refused
    assert r["reason"] == "epilogue-acc-only" and "epilogues='library' here" in r["detail"]


def test_the_modes_are_three_in_the_graph_and_in_a_scope():
    assert _fusion.EPILOGUES == ("exact", "accumulator", "library")
    for mode in _fusion.EPILOGUES:
        assert tilefish.fusion(epilogues=mode).epilogues == mode
        assert tilefish.fusion(epilogues=mode).canonical() == tilefish.fusion().canonical()  # no region is named by the mode
    with pytest.raises(TilefishPlanError, match=r"epilogues must be one of \['exact', 'accumulator', 'library'\]"):
        tilefish.fusion(epilogues="acc")

    with pytest.raises(TilefishPlanError, match=r"tilefish.fused: epilogues must be one of \['exact', 'accumulator', 'library'\]"):

        @tilefish.graph
        def bad_scope(x: Tensor[T, H, bf16]):
            with tilefish.fused(epilogues="libraries"):
                return x

        tilefish.plan(bad_scope)


def test_a_pattern_that_reads_a_widening_of_another_mode_is_left_to_that_mode():
    """The widenings of ``c`` and the bias are ops of the pattern: a residual widened in an ``"accumulator"`` scope (the f32 result of an extern, by
    ``R_acc``) is not read as a rounded ``c`` by a rewrite of the graph's ``"library"`` mode; the GEMM stays, and ``R_acc`` is the widening's."""

    def body(x, w1, b1, w2, b2):
        r = GEMM(x, w1)
        with tilefish.fused(epilogues="accumulator"):
            r32 = r.to(f32)
        return (up(x, w1).to(f32) + r32).to(bf16)

    g = graph(body, "library", "modes_residual_acc")
    f = _fusion.fused(g)
    assert not any("__" in e for e in entries(g)), entries(g)
    (w,) = library_epilogues(g)
    assert (w["matched"], w["applied"], w["mode"]) == (True, False, "accumulator/library")
    assert [r["kind"] for r in f.rewrites if r.get("applied", True)] == ["accumulator"]


def test_an_unmatched_gemm_widened_in_an_accumulator_scope_has_the_note_on_either_path_of_the_report():
    def body(x, w1, b1, w2, b2):
        with tilefish.fused(epilogues="accumulator"):
            a = (up(x, w1).to(f32) * 2.0).to(bf16)  # no epilogue of the library, widened in an "accumulator" scope
        return (a.to(f32) + b1.to(f32)).to(bf16)  # the graph's mode (in a "library" graph, the rewrite's pass runs)

    for mode in ("exact", "library"):
        (w,) = library_epilogues(graph(body, mode, f"modes_note_{mode}"))
        assert (w["matched"], w["mode"]) == (False, "accumulator") and "R_acc of a cuBLASLt GEMM is not planned" in w["note"], (mode, w)
