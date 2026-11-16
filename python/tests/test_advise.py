"""The plan-level fusion advisor (:mod:`tilefish._advise`, ``tilefish-explain DIR --graph G --advise``) and the per-step profiler's runtime path
(``PlanRunner.profile``), with no GPU: synthetic plan documents, the golden plan fixtures, exports by the fake compiler, and a fake backend. The profiler on
a device: ``test_gpu_advise.py``.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import json
import pathlib
import sys

import fake_compile
import pytest
from test_planrun import FakeBackend, Weights
from test_planrun import runner as fake_runner

import tilefish
from tilefish import Dim, Tensor, _advise, _fusion, _trace, bf16, explain, f32, runtime

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE / "golden" / "plans"
M = 64
ROW = M * 1024 * 2  # bytes of one [M, 1024] bf16 tensor


# -- synthetic documents ------------------------------------------------------------------------------------------------------------------------


def buf(name, role, shape=("M", 1024), dtype="bf16"):
    return {"name": name, "role": role, "dtype": dtype, "shape": list(shape), "offset": None, "bytes": 1}


def kstep(entry, *args):
    """A kernel step; ``args`` are ``(parameter, buffer)`` or ``(parameter, buffer, view)``."""
    return {"entry": entry, "kind": "kernel", "args": [{"name": a[0], "binding": a[1], **({"view": a[2]} if len(a) > 2 else {})} for a in args]}


def lstep(entry, a, b, out):
    return {"entry": entry, "kind": "library", "args": [{"name": "a", "binding": a}, {"name": "b", "binding": b}, {"name": "out", "binding": out}],
            "workspace": "workspace"}


def kernel(stem, params, generated=False, row=False, contracts=False, kind="triton"):
    e = {"stem": stem, "op": stem.split(".", 1)[0], "arch": "sm_103", "kind": kind,
         "abi": {"params": [{"name": n, "kind": "pointer", "access": acc} for n, acc in params]},
         "constexprs": [{"name": "D"}, {"name": "BLOCK_D"}] if row else [{"name": "BLOCK"}]}
    if kind == "triton":
        e["options"] = {"num_warps": 4, "num_stages": 3, **({} if contracts else {"enable_fp_fusion": False})}
    if generated:
        e["source_sha256"] = "ab" * 32
    return e


def gemm(stem, m, n, k):
    return {"stem": stem, "op": stem.split(".", 1)[0], "arch": "sm_103", "impl": "cublaslt.gemm", "m": m, "n": n, "k": k,
            "dtypes": {"a": "bf16", "b": "bf16", "out": "bf16"}, "compute_type": "CUBLAS_COMPUTE_32F",
            "abi": {"params": [{"name": "a"}, {"name": "b"}, {"name": "out"}]}}


def plan(buffers, steps, values=None):
    values = values or {"M": M}
    return {"schema": "tilefish.plan.v1", "graph": "g", "arch": "sm_103", "dims": {k: [v] for k, v in values.items()}, "buffers": buffers,
            "buckets": [{"values": values, "steps": steps}], "plan_digest": "cd" * 32, "manifest_digest": "ef" * 32}


IO = [("x", "in"), ("y", "out")]


def chain_plan():
    """ext (a hand-written kernel) -> act (a @tilefish.kernel) -> gemm -> act -> out0."""
    buffers = [buf("x", "input"), buf("w", "weight", (1024, 1024)), buf("t0", "intermediate"), buf("t1", "intermediate"), buf("t2", "intermediate"),
               buf("out0", "output"), buf("workspace", "workspace", (1024,), "u8")]
    steps = [kstep("ext", ("x", "x"), ("y", "t0")), kstep("act", ("x", "t0"), ("y", "t1")), lstep("gemm.M64", "t1", "w", "t2"),
             kstep("act", ("x", "t2"), ("y", "out0"))]
    manifest = {"kernels": [kernel("ext", IO), kernel("act", IO, generated=True)], "library": [gemm("gemm.M64", M, 1024, 1024)]}
    return plan(buffers, steps), manifest


# -- the steps, producer and consumer -----------------------------------------------------------------------------------------------------------


def test_each_step_has_its_bytes_flops_class_and_roofline():
    doc, manifest = chain_plan()
    adv = _advise.analyse(doc, manifest)
    s = adv.doc["steps"]
    assert [x["class"] for x in s] == ["opaque", "elementwise", "library", "elementwise"]
    assert s[0]["bytes_read"] == ROW and s[0]["bytes_written"] == ROW
    assert s[2]["flops"] == 2 * M * 1024 * 1024 and s[2]["bytes_read"] == ROW + 1024 * 1024 * 2
    # 64 rows of a 1024 x 1024 GEMM: 56 FLOPs per byte, far below the balance of sm_103 (2250 TFLOPS over 8000 GB/s): memory-bound
    assert s[2]["bound"] == "memory" and 50 < s[2]["intensity"] < 60 and adv.doc["machine"]["balance"] == pytest.approx(281.2, abs=0.1)
    assert s[0]["bound"] == "unknown" and s[1]["bound"] == "memory"  # a hand-written kernel without a trace: its MMAs are not known
    assert s[1]["est_us"] == pytest.approx(2 * ROW / 8000e9 * 1e6, abs=1e-4)
    big = _advise.analyse(plan(doc["buffers"], doc["buckets"][0]["steps"], {"M": 4096}), {**manifest, "library": [gemm("gemm.M64", 4096, 1024, 1024)]})
    assert big.doc["steps"][2]["bound"] == "compute"  # 4096 rows: 683 FLOPs per byte


def test_an_intermediate_with_one_producer_and_one_consumer_is_a_candidate_bounded_by_the_library_step():
    doc, manifest = chain_plan()
    adv = _advise.analyse(doc, manifest)
    by_steps = {tuple(c["steps"]): c for c in adv.candidates}
    assert set(by_steps) == {(0, 1), (1, 2), (2, 3)}
    pair, pro, epi = by_steps[(0, 1)], by_steps[(1, 2)], by_steps[(2, 3)]
    assert (pair["shape"], pro["shape"], epi["shape"]) == ("pair", "prologue", "epilogue")
    assert pair["saved_bytes"]["M=64"] == 2 * ROW and pair["launches_removed"] == 1
    assert pair["saved_us"]["M=64"] == pytest.approx(2 * ROW / 8000e9 * 1e6, rel=1e-6)  # bytes saved over the bandwidth
    assert pair["blockers"] == ["opaque-unknown", "dsl-kernel"] and pair["phase"] == "1"
    assert epi["blockers"][0] == "epilogue-library" and epi["phase"] == "future" and epi["gate"] == "epilogue-library"
    assert pro["blockers"][0] == "prologue-library"
    assert any("GEMM / convolution epilogue: future work" in a for a in epi["actions"])
    # the GEMM is memory-bound at 64 rows: its read of t1 counts in what the prologue saves
    assert pro["saved_bytes"]["M=64"] == 2 * ROW


def test_chains_merge_and_fan_outs_are_not_candidates():
    buffers = [buf("x", "input"), *[buf(f"t{i}", "intermediate") for i in range(4)], buf("out0", "output"), buf("out1", "output")]
    steps = [kstep("a", ("x", "x"), ("y", "t0")), kstep("a", ("x", "t0"), ("y", "t1")), kstep("a", ("x", "t1"), ("y", "t2")),
             kstep("a", ("x", "t2"), ("y", "out0")), kstep("a", ("x", "t2"), ("y", "out1"))]
    adv = _advise.analyse(plan(buffers, steps), {"kernels": [kernel("a", IO, generated=True)], "library": []})
    (c,) = adv.candidates
    assert c["shape"] == "chain" and c["steps"] == [0, 1, 2] and c["buffers"] == ["t0", "t1"]
    assert c["saved_bytes"]["M=64"] == 4 * ROW and c["launches_removed"] == 2
    assert adv.fanouts == [{"buffer": "t2", "producer": 2, "consumers": [3, 4]}]
    assert "read by several steps" in adv.text()


def test_outputs_carried_inout_taps_and_buffers_written_twice_stay_stored():
    buffers = [buf("x", "input"), buf("h", "carried"), buf("s", "inout"), buf("t0", "intermediate"), buf("t1", "intermediate"), buf("t2", "intermediate"),
               buf("out0", "output")]
    steps = [
        kstep("a", ("x", "x"), ("y", "h")),  # a carried buffer, read next
        kstep("a", ("x", "h"), ("y", "t0")),
        kstep("u", ("x", "t0"), ("y", "s")),  # an InOut parameter, read next
        kstep("a", ("x", "s"), ("y", "t1")),
        kstep("i", ("x", "t1"), ("y", "t1")),  # a result written over its input (MayAlias): t1 is written twice
        kstep("a", ("x", "t1"), ("y", "t2")),  # t2: the caller taps it
        kstep("a", ("x", "t2"), ("y", "out0")),
    ]
    manifest = {"kernels": [kernel("a", IO, generated=True), kernel("u", IO, generated=True), kernel("i", IO, generated=True)], "library": []}
    adv = _advise.analyse(plan(buffers, steps), manifest, keep=["t2"])
    assert [tuple(c["steps"]) for c in adv.candidates] == [(1, 2)]  # t0 alone: written by step 1, read by step 2 only
    why = {k["buffer"]: k["why"] for k in adv.kept}
    assert why == {"h": "carried to a later plan", "s": "updated in place (an InOut parameter)",
                   "t1": "written more than once (an in-place update, or a result written over its input)", "t2": "kept (a tap)"}
    assert [c["buffers"] for c in _advise.analyse(plan(buffers, steps), manifest).candidates] == [["t0"], ["t2"]]  # untapped, t2 is one
    # an InOut parameter of a kernel bound to an intermediate (an accumulator updated in place): written twice, kept
    acc = [buf("x", "input"), buf("t0", "intermediate"), buf("out0", "output")]
    steps2 = [kstep("a", ("x", "x"), ("y", "t0")), kstep("acc", ("x", "x"), ("y", "t0")), kstep("a", ("x", "t0"), ("y", "out0"))]
    kernels = [kernel("a", IO, generated=True), kernel("acc", [("x", "in"), ("y", "inout")], generated=True)]
    adv2 = _advise.analyse(plan(acc, steps2), {"kernels": kernels, "library": []})
    assert adv2.candidates == [] and {k["buffer"]: k["why"] for k in adv2.kept}["t0"].startswith("written more than once")


def test_a_pair_with_a_step_between_them_is_no_candidate():
    # a writes t0 and t1; the GEMM reads t1; c reads t0 and the GEMM's t2: c depends on a through the GEMM, so a and c alone are not one kernel
    two = [("x", "in"), ("y0", "out"), ("y1", "out")]
    buffers = [buf("x", "input"), buf("w", "weight", (1024, 1024)), *[buf(f"t{i}", "intermediate") for i in range(3)], buf("out0", "output"),
               buf("out1", "output"), buf("workspace", "workspace", (1024,), "u8")]
    steps = [kstep("a", ("x", "x"), ("y0", "t0"), ("y1", "t1")), lstep("gemm.M64", "t1", "w", "t2"), kstep("c", ("x", "t0"), ("y0", "t2"), ("y1", "out0"))]
    manifest = {"kernels": [kernel("a", two, generated=True), kernel("c", [("x", "in"), ("y0", "in"), ("y1", "out")], generated=True)],
                "library": [gemm("gemm.M64", M, 1024, 1024)]}
    adv = _advise.analyse(plan(buffers, steps), manifest)
    assert sorted(tuple(c["steps"]) for c in adv.candidates) == [(0, 1), (1, 2)]  # the GEMM's prologue and epilogue
    (b,) = adv.blocked
    assert (b["buffer"], b["reason"], b["middle"]) == ("t0", "library-in-the-middle", [1]) and b["phase"] == "future"
    # a kernel in the middle whose input another step reads too (stored either way): a and c alone would have it between them, but once it joined c
    # (t2) the union of all three has no step in the middle, so a joins them too (t0, tried again): one group saving t0 and t2
    steps2 = [kstep("a", ("x", "x"), ("y0", "t0"), ("y1", "t1")), kstep("m", ("x", "t1"), ("y", "t2")),
              kstep("c", ("x", "t0"), ("y0", "t2"), ("y1", "out0")), kstep("m", ("x", "t1"), ("y", "out1"))]
    manifest2 = {"kernels": [*manifest["kernels"], kernel("m", IO, generated=True)], "library": []}
    adv2 = _advise.analyse(plan(buffers, steps2), manifest2)
    (g,) = adv2.candidates
    assert g["steps"] == [0, 1, 2] and g["shape"] == "group" and sorted(g["buffers"]) == ["t0", "t2"] and adv2.blocked == []
    assert adv2.fanouts[0]["buffer"] == "t1"
    # with a step that cannot join (a GEMM) in the middle, a and c stay apart however long it is tried
    steps3 = [kstep("a", ("x", "x"), ("y0", "t0"), ("y1", "t1")), lstep("gemm.M64", "t1", "w", "t2"), kstep("m", ("x", "t2"), ("y", "out1")),
              kstep("c", ("x", "t0"), ("y0", "out1"), ("y1", "out0"))]
    adv3 = _advise.analyse(plan([*buffers], steps3), {**manifest2, "library": manifest["library"]})
    assert [(b["buffer"], b["reason"]) for b in adv3.blocked] == [("t0", "library-in-the-middle")]


class FakeTrace:
    """A trace whose SASS of each kernel is given: what ``tilefish.explain.Trace.kernel`` returns once it has checked the files."""

    def __init__(self, root, listings):
        self.root = root
        for stem, text in listings.items():
            (root / f"{stem}.sm_103").mkdir(parents=True)
            (root / f"{stem}.sm_103" / "sass.txt").write_text(text)
        self.listings = listings

    def kernel(self, stem, arch, sha256):
        return {"sass": {"label": "x"}} if stem in self.listings and sha256 == "ab" * 32 else None


def test_the_trace_tells_a_hand_written_kernel_with_mmas_from_one_without(tmp_path):
    doc, manifest = chain_plan()
    manifest["kernels"][0]["sha256"] = "ab" * 32
    with_mma = FakeTrace(tmp_path / "a", {"ext": "        /*0100*/                   UTCHMMA gdesc[UR4], gdesc[UR6], tmem[UR8], idesc[UR9], UPT ;\n"})
    a = _advise.analyse(doc, manifest, trace=with_mma)
    assert a.infos[0].mma is True and a.infos[0].label == "ext [Triton extern, MMA]"
    assert a.doc["steps"][0]["bound"] == "unknown"
    pair = next(c for c in a.candidates if c["steps"] == [0, 1])
    assert pair["blockers"][0] == "opaque-compute-epilogue" and pair["phase"] == "4"
    without = FakeTrace(tmp_path / "b", {"ext": "        /*0100*/                   FMUL R5, R2, 2 ;\n"})
    b = _advise.analyse(doc, manifest, trace=without)
    assert b.infos[0].mma is False and b.doc["steps"][0]["bound"] == "memory"
    assert next(c for c in b.candidates if c["steps"] == [0, 1])["blockers"][0] == "opaque-extern"
    stale = FakeTrace(tmp_path / "c", {"other": "HMMA"})  # no SASS of this binary in the trace: unknown
    assert _advise.analyse(doc, manifest, trace=stale).infos[0].mma is None


def test_a_hand_written_kernel_far_slower_than_its_bytes_is_bound_by_its_arithmetic():
    doc, manifest = chain_plan()
    adv = _advise.analyse(doc, manifest, timing=timing_of(doc, [400.0, 3.0, 10.0, 3.0], copy=1000.0))
    assert adv.infos[0].arith_bound and adv.doc["steps"][0]["bound"] == "compute (measured)" and adv.doc["steps"][0]["achieved_GBps"] < 1
    pair = next(c for c in adv.candidates if c["steps"] == [0, 1])
    assert pair["blockers"][0] == "opaque-compute-epilogue" and "bound by its arithmetic" in pair["actions"][0]
    fast = _advise.analyse(doc, manifest, timing=timing_of(doc, [3.3, 3.0, 10.0, 3.0], copy=1000.0))
    assert not fast.infos[0].arith_bound and fast.doc["steps"][0]["bound"] == "memory (measured)"
    assert fast.doc["steps"][1]["achieved_GBps"] is None  # a step at the launch floor: its bandwidth is not measured


def test_each_bucket_has_its_own_entries_and_views():
    # one plan for M in {32, 64}: the GEMM's entry and the view of its output are the bucket's (only the bindings are every bucket's)
    buffers = [buf("x", "input"), buf("w", "weight", (1024, 1024)), buf("t0", "intermediate"), buf("t1", "intermediate"), buf("out0", "output"),
               buf("workspace", "workspace", (1024,), "u8")]

    def steps(m):
        return [lstep(f"gemm.M{m}", "x", "w", "t0"), kstep("act", ("x", "t0", {"shape": [m * 1024], "strides": [1], "offset": 0}), ("y", "t1")),
                kstep("act", ("x", "t1"), ("y", "out0"))]

    doc = {**plan(buffers, steps(32), {"M": 32}), "dims": {"M": [32, 64]},
           "buckets": [{"values": {"M": 32}, "steps": steps(32)}, {"values": {"M": 64}, "steps": steps(64)}]}
    manifest = {"kernels": [kernel("act", IO, generated=True)], "library": [gemm("gemm.M32", 32, 1024, 1024), gemm("gemm.M64", 64, 1024, 1024)]}
    adv = _advise.analyse(doc, manifest)
    assert adv.doc["bucket"] == {"M": 64}
    assert adv.doc["steps"][0]["entry"] == "gemm.M64" and "GEMM 64x1024x1024" in adv.doc["steps"][0]["label"]
    assert adv.doc["steps"][0]["flops"] == 2 * 64 * 1024 * 1024
    assert adv.doc["steps"][1]["bytes_read"] == 64 * 1024 * 2  # the view at M = 64, not M = 32's
    epi = next(c for c in adv.candidates if c["steps"] == [0, 1])
    assert epi["saved_bytes"] == {"M=32": 2 * 32 * 2048, "M=64": 2 * 64 * 2048}


def test_a_strided_store_a_memset_and_two_library_ops_are_said_so():
    buffers = [buf("x", "input"), buf("w", "weight", (1024, 1024)), buf("t0", "intermediate", ("M", 2048)), buf("z", "intermediate"),
               buf("g0", "intermediate"), buf("g1", "intermediate"), buf("out0", "output"), buf("out1", "output"), buf("out2", "output"),
               buf("workspace", "workspace", (1024,), "u8")]
    block = {"shape": [M, 1024], "strides": [2048, 1], "offset": 1024}  # the right half of t0's columns
    steps = [
        kstep("p", ("x", "x"), ("y", "t0", block)),  # writes a column block of t0
        kstep("q", ("x", "t0"), ("y", "out0")),
        {"kind": "memset", "args": [{"name": "dst", "binding": "z"}]},
        kstep("q", ("x", "z"), ("y", "out1")),
        lstep("gemm.M64", "x", "w", "g0"),
        lstep("gemm.M64", "g0", "w", "g1"),  # a GEMM of a GEMM's result
        kstep("q", ("x", "g1"), ("y", "out2")),
    ]
    manifest = {"kernels": [kernel("p", IO, generated=True), kernel("q", IO, generated=True)], "library": [gemm("gemm.M64", M, 1024, 1024)]}
    adv = _advise.analyse(plan(buffers, steps), manifest)
    by = {tuple(c["buffers"]): c for c in adv.candidates}
    assert "strided-output" in by[("t0",)]["blockers"] and by[("t0",)]["phase"] == "2"
    assert by[("z",)]["blockers"][0] == "memset-step" and by[("z",)]["phase"] == "future"
    assert [(b["buffer"], b["reason"]) for b in adv.blocked] == [("g0", "library-library")]


def test_views_of_one_buffer_count_once_each():
    buffers = [buf("x", "input"), buf("t0", "intermediate", ("M", 3072)), buf("out0", "output")]
    view = {"shape": [M, 1024], "strides": [3072, 1], "offset": 1024}
    steps = [kstep("p", ("x", "x"), ("y", "t0")), kstep("q", ("a", "t0", {**view, "offset": 0}), ("b", "t0", view), ("y", "out0"))]
    manifest = {"kernels": [kernel("p", IO, generated=True), kernel("q", [("a", "in"), ("b", "in"), ("y", "out")], generated=True)], "library": []}
    (c,) = _advise.analyse(plan(buffers, steps), manifest).candidates
    assert c["saved_bytes"]["M=64"] == 3 * ROW + 2 * ROW  # the whole [M, 3072] written, two [M, 1024] views read


# -- ranking --------------------------------------------------------------------------------------------------------------------------------


def timing_of(doc, us, copy=None):
    st = doc["buckets"][0]["steps"]
    return {"schema": _advise.TIMING_SCHEMA, "graph": doc["graph"], "plan_digest": doc["plan_digest"], "bucket": doc["buckets"][0]["values"],
            "method": "graph", "reps": 3, "device": {"copy_GBps": copy} if copy else {},
            "steps": [{"index": i, "entry": s.get("entry"), "us": {"median": t}} for i, (s, t) in enumerate(zip(st, us))]}


def test_ranking_by_bytes_without_timing_and_by_measured_time_with_it():
    buffers = [buf("x", "input"), buf("big0", "intermediate", ("M", 8192)), buf("mid", "intermediate", ("M", 4096)),
               buf("out0", "output", ("M", 8192)), buf("out1", "output", ("M", 4096))]
    steps = [kstep("p", ("x", "x"), ("y", "big0")), kstep("q", ("x", "big0"), ("y", "out0")),
             kstep("r", ("x", "x"), ("y", "mid")), kstep("s", ("x", "mid"), ("y", "out1"))]
    manifest = {"kernels": [kernel(n, IO, generated=True) for n in "pqrs"], "library": []}
    doc = plan(buffers, steps)
    static = _advise.analyse(doc, manifest)
    assert [c["steps"] for c in static.candidates] == [[0, 1], [2, 3]]  # 2 MiB saved before 1 MiB
    big, mid, x = 8192 * M * 2, 4096 * M * 2, ROW
    assert static.candidates[0]["saved_us"]["M=64"] == pytest.approx(2 * big / 8000e9 * 1e6)
    # measured, at a copy bandwidth of 1000 GB/s: p takes the launch floor (3 us, the fastest step) and q moves its 2 MiB at the bandwidth; r and s move
    # theirs at 260 GB/s, slower but still bound by their bytes (a quarter of the bandwidth or more)
    bw, slow = 1000e9, 260e9
    t = [3.0, 3.0 + 2 * big / bw * 1e6, 3.0 + (x + mid) / slow * 1e6, 3.0 + 2 * mid / slow * 1e6]
    timed = _advise.analyse(doc, manifest, timing=timing_of(doc, t, copy=1000.0))
    assert timed.machine.bandwidth == 1000.0 and timed.floor == 3.0
    assert [c["steps"] for c in timed.candidates] == [[2, 3], [0, 1]]
    # p -> q: p saves what its time above the floor allows (0.001 us at least), q its read of big0 at the bandwidth, and one kernel fewer the floor
    assert timed.candidates[1]["saved_us"]["M=64"] == pytest.approx(0.001 + big / bw * 1e6 + 3.0, abs=1e-3)
    # r -> s: each side's bytes of mid at the 260 GB/s it achieves, and the floor
    assert timed.candidates[0]["saved_us"]["M=64"] == pytest.approx(2 * mid / slow * 1e6 + 3.0, abs=1e-3)
    assert timed.doc["estimate"] == "measured" and timed.doc["total_us"] == pytest.approx(sum(t), abs=1e-3)
    assert timed.doc["by_class"]["elementwise"]["us"] == pytest.approx(sum(t), abs=1e-3)
    # a step far slower than its bytes (an attention of FMAs, a slow kernel) is not bound by them: what it no longer moves saves the bytes' time only
    slowest = _advise.analyse(doc, manifest, timing=timing_of(doc, [3.0, t[1], 400.0, 400.0], copy=1000.0))
    assert next(c for c in slowest.candidates if c["steps"] == [2, 3])["saved_us"]["M=64"] == pytest.approx(2 * mid / bw * 1e6 + 3.0, abs=1e-3)


def test_the_machine_is_the_plans_architecture_and_a_partition_scales_it():
    doc, manifest = chain_plan()
    for arch, example in (("sm_90", "H100 SXM"), ("sm_100", "B200"), ("sm_103", "B300"), ("sm_120", "RTX 5090")):
        adv = _advise.analyse({**doc, "arch": arch}, {**manifest, "kernels": [{**k, "arch": arch} for k in manifest["kernels"]],
                                                         "library": [{**g, "arch": arch} for g in manifest["library"]]})
        assert adv.machine.example == example and adv.doc["machine"]["balance"] == pytest.approx(_advise.MACHINES[arch].balance, abs=0.1)
        assert adv.doc["steps"][2]["flops"] == 2 * M * 1024 * 1024  # the GEMM found in the manifest of that architecture
    # on an RTX 5090 (balance 117) the 64-row GEMM (56 FLOPs per byte) is still memory-bound; a 4096-row one compute-bound
    assert _advise.analyse(doc, manifest, machine="sm_120").doc["steps"][2]["bound"] == "memory"
    with pytest.raises(_advise.TilefishAdviseError, match="no machine balance for sm_75"):
        _advise.analyse(doc, manifest, machine="sm_75")
    slice_ = _advise.MACHINES["sm_103"].scaled(bandwidth=1614.0, sms=30)
    assert slice_.bandwidth == 1614.0 and slice_.tensor16 == pytest.approx(2250.0 * 30 / 148) and 280 < slice_.balance < 286
    assert _advise.analyse(doc, manifest, bandwidth=1000.0).machine.bandwidth == 1000.0


def test_a_timing_file_of_another_plan_or_bucket_is_refused():
    doc, manifest = chain_plan()
    t = timing_of(doc, [1.0, 1.0, 1.0, 1.0])
    with pytest.raises(_advise.TilefishAdviseError, match="of the plan"):
        _advise.analyse(doc, manifest, timing={**t, "plan_digest": "00" * 32})
    with pytest.raises(_advise.TilefishAdviseError, match="is not step 1"):
        _advise.analyse(doc, manifest, timing={**t, "steps": [{**s, "entry": "other"} if s["index"] == 1 else s for s in t["steps"]]})
    with pytest.raises(_advise.TilefishAdviseError, match="not M=1|not \\{'M': 1\\}"):
        _advise.analyse(doc, manifest, timing=t, bucket="M=1")
    with pytest.raises(_advise.TilefishAdviseError, match="step-timing"):
        _advise.analyse(doc, manifest, timing={**t, "schema": "x"})
    bad = [
        ({"plan_digest": None}, "of the plan"),
        ({"steps": t["steps"][:3]}, "times 3 steps, but the plan has 4"),
        ({"steps": []}, "times 0 steps"),
        ({"steps": [t["steps"][0], t["steps"][0], *t["steps"][2:]]}, "listed twice"),
        ({"steps": [{**s, "us": "fast"} for s in t["steps"]]}, "no time in microseconds"),
        ({"steps": [{**s, "us": {"median": float("nan")}} for s in t["steps"]]}, "no time in microseconds"),
        ({"device": {"copy_GBps": -1}}, "copy_GBps is not a positive number"),
        ({"bucket": {"M": "64"}}, "bucket is not an object of dimension sizes"),
    ]
    for change, message in bad:
        with pytest.raises(_advise.TilefishAdviseError, match=message):
            _advise.analyse(doc, manifest, timing={**t, **change})
    # a step with only its net time (no median) is read
    nets = _advise.analyse(doc, manifest, timing={**t, "steps": [{**s, "us": {"net": 2.0}} for s in t["steps"]]})
    assert nets.doc["total_us"] == 8.0


def test_the_golden_block_on_the_command_line(capsys):
    assert explain.main([str(GOLDEN / "block"), "--graph", "block", "--advise", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == _advise.ADVICE_SCHEMA and doc["bucket"] == {"M": 128} and len(doc["buckets"]) == 3
    assert [s["class"] for s in doc["steps"]] == ["row", "library", "opaque", "library", "row", "library", "elementwise", "library", "elementwise"]
    first = doc["candidates"][0]
    assert first["steps"] == [5, 6] and first["shape"] == "epilogue"  # gemm_up's [M, 1024] output read by gelu
    assert {p["blocker"] for p in doc["patterns"]} == {"epilogue-library", "prologue-library"}
    # the residual r of add_rmsnorm, read by the last add alone: the GEMMs of the MLP sit between them, so it is no candidate
    (r,) = doc["blocked"]
    assert (r["producer"], r["consumer"], r["reason"]) == (4, 8, "library-in-the-middle") and r["middle"] == [5, 6, 7]
    assert "step 5 (gemm_up)" in r["why"]
    assert explain.main([str(GOLDEN / "block"), "--graph", "block", "--advise", "--bucket", "M=32", "--top", "2"]) == 0
    text = capsys.readouterr().out
    assert "ranked at bucket {'M': 32}" in text and "#1 " in text and "... 6 more" in text and "time by class" in text and "library-in-the-middle" in text
    assert explain.main([str(GOLDEN / "block"), "--graph", "nope", "--advise"]) == 2
    assert "no plan of the graph 'nope'" in capsys.readouterr().err
    assert explain.main([str(GOLDEN / "set"), "--advise"]) == 2  # a plan set: say which graph
    assert "say which with --graph" in capsys.readouterr().err


# -- why the arithmetic of a graph body was not fused ----------------------------------------------------------------------------------------------

R = Dim("R", buckets=[64])


@tilefish.kernel
def adv_scale(x: Tensor[R, 256, f32]) -> Tensor[R, 256, f32]:
    return x * 2.0


@tilefish.graph
def adv_body(x: Tensor[R, 256, bf16], w: Tensor[R, 256, bf16]) -> Tensor[R, 256, bf16]:
    h = tilefish.barrier(x.to(f32) * w.to(f32))
    h2 = h + 1.0  # across the barrier
    s = adv_scale(h2)  # a call between regions
    n = tilefish.sum(s * s, axis=-1, keepdims=True)  # no order=: a region of its own, its operand stored (D2)
    o = tilefish.sum(s, axis=-1, keepdims=True, order=tilefish.tree(128))  # ordered: still a region of its own in version 1
    return (s * tilefish.rsqrt(n + o)).to(bf16)


@tilefish.graph
def adv_view(x: Tensor[R, 512, f32]) -> Tensor[R, 256, f32]:
    g, _ = x.split(256, axis=-1)
    n = tilefish.sum(g * g, axis=-1, keepdims=True, order=tilefish.tree(128))
    return g * tilefish.rsqrt(n)  # a per-row operand: the row template, which reads no strided input


def test_every_boundary_of_the_regions_has_a_reason_and_a_fix():
    got = _fusion.boundaries(adv_body)
    reasons = [(b["reason"], b["from"]["op"], b["to"]["op"]) for b in got]
    assert ("barrier", "mul", "add") in reasons
    assert ("prologue-kernel", "add", "adv_scale") in reasons and ("epilogue-kernel", "adv_scale", "mul") in reasons
    assert ("reduction-unordered", "mul", "sum") in reasons and ("reduction-unordered", "sum", "add") in reasons
    assert any(r == "reduction" and t == "add" for r, _, t in reasons)  # the ordered sum feeds the tail: phase 2
    for b in got:
        fix, phase = _fusion.BOUNDARY_FIXES[b["reason"]]
        assert b["fix"] == fix and b["phase"] == phase and b["detail"]
        assert _advise.BLOCKERS[b["reason"]] == (fix, phase)
    assert [b for b in got if b["reason"] == "barrier"][0]["fix"].startswith("remove tilefish.barrier")
    assert "add order=" in [b for b in got if b["reason"] == "reduction-unordered"][0]["fix"]
    view = _fusion.boundaries(adv_view)
    if _fusion.report(adv_view)["regions"][-1]["template"] == "row":
        assert any(b["reason"] == "materialised" and "row template" in b["detail"] for b in view)


def test_a_graph_that_declares_no_grouping_is_told_so():
    import fusion_graphs as fg

    got = _fusion.boundaries(fg.bias_act_unfused)
    assert got and {b["reason"] for b in got} == {"unfused"} and "grouping='none'" in got[0]["fix"]
    assert all(b["reason"] != "unfused" for b in _fusion.boundaries(fg.bias_act))


def test_the_reasons_do_not_change_the_fusion():
    # the diagnosis repeats the pass's rules; the regions, their kernels and the plan are what they were
    before = [r["op"] for r in _fusion.report(adv_body)["regions"]]
    _fusion.boundaries(adv_body)
    assert [r["op"] for r in _fusion.report(adv_body)["regions"]] == before
    assert tilefish.plan(adv_body).digest == tilefish.plan(adv_body).digest


@pytest.fixture
def fake(tmp_path, monkeypatch):
    compiler = fake_compile.install(monkeypatch)
    monkeypatch.setattr(runtime, "_modules", {})
    monkeypatch.setattr(runtime, "_gen_root", [])
    monkeypatch.setattr(runtime, "_warned_gen_roots", set())
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.delenv(_trace.TRACE_ENV, raising=False)
    monkeypatch.setattr(tilefish._sass, "disassemble", lambda *a, **k: (_ for _ in ()).throw(tilefish.TilefishBuildError("no nvdisasm here")))
    return compiler


def test_a_traced_export_says_why_and_an_untraced_one_says_what_the_report_can(fake, tmp_path):
    with pytest.warns(Warning):
        traced = runtime.export(adv_body, tmp_path / "t", arch=103, trace=True, verify=False)
    plain = runtime.export(adv_body, tmp_path / "p", arch=103, verify=False)
    assert traced.digest == plain.digest
    assert (tmp_path / "t" / "adv_body.fusion.json").read_bytes() == (tmp_path / "p" / "adv_body.fusion.json").read_bytes()
    gdoc = json.loads((tmp_path / "t" / "trace" / "graphs" / "adv_body.json").read_text())
    assert {b["reason"] for b in gdoc["boundaries"]} >= {"barrier", "reduction-unordered", "prologue-kernel", "epilogue-kernel"}
    assert all("template" in r for r in gdoc["regions"].values())

    adv = tilefish.advise(tmp_path / "t", "adv_body")
    classes = [s["class"] for s in adv.doc["steps"]]
    assert "fused" in classes and "elementwise" in classes  # the regions, and adv_scale (a @tilefish.kernel)
    reasons = {u["reason"] for u in adv.doc["unfused_arithmetic"]}
    assert {"barrier", "reduction-unordered", "reduction"} <= reasons
    assert any("reduction-unordered" in c["blockers"] for c in adv.candidates)
    barrier = next(c for c in adv.candidates if "barrier" in c["blockers"])
    assert barrier["phase"] == "1" and any(a.startswith("remove tilefish.barrier") for a in barrier["actions"])
    text = adv.text()
    assert "graph-body arithmetic that was not fused:" in text and "fix: remove tilefish.barrier" in text

    # no trace: the fusion report says which regions are reductions, and whether they have an order; the rest needs the trace or the graph
    untraced = tilefish.advise(tmp_path / "p", "adv_body")
    assert {u["reason"] for u in untraced.doc["unfused_arithmetic"]} >= {"reduction-unordered"}
    assert any("unknown-boundary" in c["blockers"] for c in untraced.candidates)
    with_graph = tilefish.advise(tmp_path / "p", "adv_body", gop=adv_body)
    assert any("barrier" in c["blockers"] for c in with_graph.candidates)
    with pytest.raises(_advise.TilefishAdviseError, match="gop= is the graph"):
        tilefish.advise(tmp_path / "p", "adv_body", gop=adv_view)


# -- the profiler's runtime path, on a fake backend ---------------------------------------------------------------------------------------------


class TimedBackend(FakeBackend):
    """A fake backend whose timing events read a clock that every launch advances by a step's own duration (step k: k + 1 microseconds), and the
    spin by its microseconds times ``spin``; each launch takes the host ``host_step`` seconds (``host``: the host's clock, for ``_host_clock``)."""

    def __init__(self, spin=1.0, host_step=0.0):
        super().__init__()
        self.clock = 0.0
        self.k = 0
        self.timed = 0
        self.spin = spin
        self.host_step = host_step
        self.host = 0.0

    def new_timing_event(self, index):
        self.timed += 1
        from types import SimpleNamespace

        return SimpleNamespace(name=f"timing{self.timed}", fresh=False, at=None)

    def record_event(self, event, stream, capturing):
        if getattr(event, "name", "").startswith("timing"):
            event.at = self.clock
            self.log.append(("tick", event.name, capturing))
            return
        super().record_event(event, stream, capturing)

    def elapsed_ms(self, start, end, index):
        return (end.at - start.at) / 1000.0

    def hold(self, stream, microseconds, index):
        self.log.append(("hold", microseconds))
        self.clock += microseconds * self.spin

    def _advance(self):
        self.host += self.host_step
        self.k += 1
        self.clock += float(self.k % 9 or 9)  # the steps of one run take 1 to 9 us, in order

    def launch(self, loaded, grid, args, stream):
        super().launch(loaded, grid, args, stream)
        self._advance()

    def gemm(self, spec, algo, a, b, bias, out, workspace, stream, index):
        super().gemm(spec, algo, a, b, bias, out, workspace, stream, index)
        self._advance()


def test_the_profile_times_every_step_between_events_and_launches_what_a_run_launches():
    be = TimedBackend()
    r, _ = fake_runner(backend=be)
    w = Weights(64)
    plain = FakeBackend()
    r2, _ = fake_runner(backend=plain)
    r2.run(w.args(), {})
    out = r.profile(w.args(), {}, reps=2, warmup=0, method="events")
    assert out["bucket"] == {"M": 64}
    assert [round(t[0], 6) for t in out["steps"]] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert len(out["plan"]) == 2 and out["plan"][0] == pytest.approx(45.0)
    launched = [e[1] for e in be.launches()]
    assert launched[:9] == [e[1] for e in plain.launches()]  # the same steps, in the same order
    assert ("hold", 20.0 * 9 + 50.0) in be.log and not any(e[0] == "begin" for e in be.log)
    ticks = [i for i, e in enumerate(be.log) if e[0] == "tick"]
    assert len(ticks) >= 10
    # the graph method captures the steps with an event node before the first and after each (external records), and once without
    be2 = TimedBackend()
    r3, _ = fake_runner(backend=be2)
    r3.profile(w.args(), {}, reps=1, warmup=1, method="graph")
    begins = [e for e in be2.log if e[0] == "begin"]
    assert len(begins) == 2 and ("tick", "timing1", True) in be2.log
    assert sum(1 for e in be2.log if e[0] == "graph") == 4  # warmup and timed replays of each graph
    with pytest.raises(tilefish.TilefishRuntimeError, match="method is 'graph' or 'events'"):
        r3.profile(w.args(), {}, reps=1, warmup=0, method="nvtx")


def test_the_events_method_spins_longer_for_a_slow_host(monkeypatch):
    # a host that takes 30 us to enqueue a step (270 us for the 9 steps) and a spin that runs faster than it is sized (an RTX 5090 boosts past the
    # clock rate the spin is sized by): the first spin (230 us, 184 us on the device) ends before the host is done, so the steps would run as fast as
    # the host launches them. That run is made again with a spin long enough for what it measured; only runs whose steps were all enqueued before
    # the device left the spin are measured
    be = TimedBackend(spin=0.8, host_step=30e-6)
    monkeypatch.setattr(tilefish._planrun, "_host_clock", lambda: be.host)
    r, _ = fake_runner(backend=be)
    out = r.profile(Weights(64).args(), {}, reps=2, warmup=0, method="events")
    holds = [e[1] for e in be.log if e[0] == "hold"]
    assert holds[0] == 20.0 * 9 + 50.0 and holds[1:] == [pytest.approx(1.5 * 270 * 230 / 184)] * 4  # one run again, then 2 reps of 2 runs
    assert [round(t[0], 6) for t in out["steps"]] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] and len(out["plan"]) == 2
    # a host that never keeps up: refused, with the method that does not depend on it
    be2 = TimedBackend(host_step=1.0)
    monkeypatch.setattr(tilefish._planrun, "_host_clock", lambda: be2.host)
    r2, _ = fake_runner(backend=be2)
    with pytest.raises(tilefish.TilefishRuntimeError, match="host took .* to enqueue the 9 steps.*Use method='graph'"):
        r2.profile(Weights(64).args(), {}, reps=1, warmup=0, method="events")
    # the graph method has no spin
    be3 = TimedBackend(host_step=1.0)
    monkeypatch.setattr(tilefish._planrun, "_host_clock", lambda: be3.host)
    r3, _ = fake_runner(backend=be3)
    r3.profile(Weights(64).args(), {}, reps=1, warmup=1, method="graph")
    assert not any(e[0] == "hold" for e in be3.log)


def test_the_profile_puts_back_what_the_runs_change_that_outlives_them():
    # an InOut parameter (and a carried buffer alike) is updated by every run: its bytes are copied aside before the first run and back after the last
    import plan_model as pm
    from fake_torch import FakeTensor

    be = TimedBackend()
    r, _ = fake_runner(tilefish.plan(pm.accumulate), backend=be)
    a = FakeTensor((16, 128), "f32", 1 << 36)
    state = FakeTensor((16, 128), "f32", 1 << 37)
    r.profile((a, state), {}, reps=2, warmup=1, method="events")
    copies = [e for e in be.log if e[0] == "copy"]
    assert len(copies) == 2
    (save, restore) = copies
    assert save[2] == state._ptr and save[3] == 16 * 128 * 4  # before the first launch: the caller's tensor into scratch
    assert restore[1] == state._ptr and restore[2] == save[1]  # after the last: back from that scratch
    first, last = be.log.index(save), be.log.index(restore)
    assert all(be.log.index(e) > first for e in be.launches() if e[0] == "launch") and last > max(i for i, e in enumerate(be.log) if e[0] == "launch")
    assert len([e for e in be.log if e[0] == "allocated" or e[0] == "record"]) >= 2
    # a carried buffer the plan writes: its region of the arena, aside and back; what the arena notes of it is as before
    import gpu_kernels_plans as kp

    be2 = TimedBackend()
    r2, _ = fake_runner(tilefish.plan(kp.stage_a), backend=be2)
    region = r2._carried.regions["h"]
    r2.profile((FakeTensor((8, 64), "f32", 1 << 36),), {}, reps=1, warmup=0, method="events")
    c2 = [e for e in be2.log if e[0] == "copy"]
    assert [c2[0][2], c2[1][1]] == [ARENA + region[0]] * 2 and c2[0][3] == region[1]
    assert r2.arena_of(0).shapes.get("h") is None  # no run wrote it before the profile, and none has after it


ARENA = 1 << 46  # the fake arena's address (test_planrun)
