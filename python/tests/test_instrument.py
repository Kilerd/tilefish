"""Intra-kernel instrumentation (``tilefish._probe``, ``tilefish._probes``, ``docs/en/instrument.md``) on any machine: the PTX rewrite, the probe
table of the frontend's IR, the records and what they say, the trace file and ``tilefish-explain --probes``, a kernel body's probes and their
lowering, and the instrumented variant (its manifest entry, its digest, the refusals of the hosts) with the fake compiler. The real thing, on a
device: ``test_gpu_instrument.py``.
"""

import json
import pathlib
import struct
import sys

import fake_compile
import pytest

import tilefish
from tilefish import Dim, Tensor, TilefishBuildError, TilefishManifestError, TilefishTypeError, _probe, bf16, explain, f32, runtime
from tilefish._probes import Launch, ProbeTrace, TilefishProbeError
from tilefish._probe import BEGIN, CALIB, END, MAGIC, WarpRecords
from tilefish.aot import build, compile_specs, write_manifest

N = Dim("N", divisible_by=16)
HERE = pathlib.Path(__file__).resolve().parent

# The PTX of a Triton kernel, trimmed: its parameters (the profile scratch last), a probe in a loop, a predicated early return and a ret.
PTX = """//
.version 8.8
.target sm_103a
.address_size 64

.visible .entry k_kernel(
\t.param .u64 .ptr .global .align 1 k_kernel_param_0,
\t.param .u32 k_kernel_param_1,
\t.param .u64 .ptr .global .align 1 k_kernel_param_2,
\t.param .u64 .ptr .global .align 1 k_kernel_param_3
)
.reqntid 128
{
\t.reg .pred \t%p<3>;
\t.reg .b32 \t%r<9>;
\t.reg .b64 \t%rd<5>;
$L__func_begin0:
\tld.param.b32 \t%r1, [k_kernel_param_1];
\tsetp.eq.b32 \t%p1, %r1, 0;
\t@%p1 ret;
$L__BB0_1:
\t// begin inline asm
\t// tilefish.probe 3 iter
{
mov.u32 %r2, 0;
}
\t// end inline asm
\tadd.s32 \t%r1, %r1, -1;
\tsetp.ne.b32 \t%p2, %r1, 0;
\t@%p2 bra \t$L__BB0_1;
\tret;
$L__func_end0:
}
\t.file\t1 "kernels/k.py"
"""

# The frontend's IR of a warp-specialised Gluon kernel with probes (Triton 3.5.1's asm["source"], trimmed): a default partition calling the
# compute function, a worker partition calling the load function, and a probe before the warp_specialize.
SOURCE_IR = r'''#loc = loc("kernels/ws.py":20:0)
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @ws_kernel(%x: !tt.tensordesc<tensor<64x64xf16, #shared>> loc("x"(#loc1)), %n: i32 loc("n"(#loc1))) attributes {noinline = false} {
    %0 = tt.elementwise_inline_asm "// tilefish.probe 3 setup\0A{\0Amov.u32 $0, 0;\0A}\0A" {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32 loc(#loc2)
    ttg.warp_specialize(%x, %n) attributes {requestedRegisters = array<i32: 24>}
    default {
      tt.call @"kernels.ws._compute__TDfp16S64_64S_i32"(%x, %n) : (!tt.tensordesc<tensor<64x64xf16, #shared>>, i32) -> () loc(#loc3)
      ttg.warp_yield loc(#loc3)
    }
    partition0(%arg2: !tt.tensordesc<tensor<64x64xf16, #shared>> loc(unknown), %arg3: i32 loc(unknown)) num_warps(1) {
      tt.call @"kernels.ws._load__TDfp16S64_64S_i32__NW1"(%arg2, %arg3) : (!tt.tensordesc<tensor<64x64xf16, #shared>>, i32) -> () loc(#loc4)
      ttg.warp_return loc(#loc4)
    } : (!tt.tensordesc<tensor<64x64xf16, #shared>>, i32) -> () loc(#loc)
    tt.return loc(#loc5)
  } loc(#loc)
  tt.func private @"kernels.ws._compute__TDfp16S64_64S_i32"(%x: !tt.tensordesc<tensor<64x64xf16, #shared>> loc("x"(#loc6)), %n: i32 loc("n"(#loc6))) attributes {noinline = false} {
    scf.for %i = %c0 to %n step %c1 : i32 {
      %6 = tt.elementwise_inline_asm "// tilefish.probe 4 compute.wait\0A{\0A.reg .pred %tfq;\0Amov.b64 {%tfl, %tfh}, %tft;\0A}\0Amov.u32 $0, 0;\0A" {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32 loc(#loc7)
    } loc(#loc8)
    tt.return loc(#loc8)
  } loc(#loc6)
  tt.func private @"kernels.ws._load__TDfp16S64_64S_i32__NW1"(%x: !tt.tensordesc<tensor<64x64xf16, #shared>> loc("x"(#loc9)), %n: i32 loc("n"(#loc9))) attributes {noinline = false, "ttg.num-warps" = 1 : i32} {
    %6 = tt.elementwise_inline_asm "// tilefish.probe 5 load.wait\0A{\0A}\0Amov.u32 $0, 0;\0A" {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32 loc(#loc10)
    %7 = tt.elementwise_inline_asm "// tilefish.probe 4 compute.wait\0A{\0A}\0Amov.u32 $0, 0;\0A" {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32 loc(callsite(#loc11 at #loc12))
    tt.return loc(#loc10)
  } loc(#loc9)
} loc(#loc)
#loc1 = loc("kernels/ws.py":20:0)
#loc2 = loc("kernels/ws.py":22:4)
#loc3 = loc("kernels/ws.py":24:4)
#loc4 = loc("kernels/ws.py":25:4)
#loc5 = loc("kernels/ws.py":26:4)
#loc6 = loc("kernels/ws.py":10:0)
#loc7 = loc("kernels/ws.py":12:8)
#loc8 = loc("kernels/ws.py":13:4)
#loc9 = loc("kernels/ws.py":4:0)
#loc10 = loc("kernels/ws.py":6:8)
#loc11 = loc("kernels/helpers.py":3:4)
#loc12 = loc("kernels/ws.py":7:8)
'''


# -- the PTX rewrite --------------------------------------------------------------------------------------------------------------------------


def test_the_rewrite_puts_the_slot_begin_calib_and_end_in_the_entry():
    out = _probe.instrument_ptx(PTX, 64)
    head, _, body = out.partition("{\n")
    assert head == PTX.partition("{\n")[0]  # the parameters and directives are as they were: the ABI does not change
    assert "ld.param.u64 %tfa, [k_kernel_param_3];" in body  # the probe buffer is the profile scratch, the last parameter
    assert f"mul.wide.u32 %tfo, %tfslot, {16 + 16 * 64};" in body and "selp.u32 %tfp_cap, 64, 0, %tfon;" in body
    assert f"{{{MAGIC}, %tfcta, %tfw, 0}}" in body
    # @begin, then @calib twice from one block of instructions, all before Triton's own code
    begin, calib, triton = body.index("tilefish.probe 0 @begin"), body.index("tilefish.probe 2 @calib"), body.index("$L__func_begin0:")
    assert begin < calib < triton and body.count("tilefish.probe 2 @calib") == 1 and "@%tfp_again bra $L__tfp_calib;" in body
    # @end and the count before each ret: the predicated one branches around it
    assert body.count("tilefish.probe 1 @end") == 2 and body.count("@%tfe st.global.b32 [%tfp_hdr+12], %tfp_n;") == 2
    assert "@!%p1 bra $L__tfp_skip_0;" in body and "$L__tfp_skip_0:" in body
    k = body.index("@!%p1 bra $L__tfp_skip_0;")
    assert body.index("tilefish.probe 1 @end", k) < body.index("ret;", k) < body.index("$L__tfp_skip_0:", k)
    assert body.count("ret;") == 2 and out.endswith('\t.file\t1 "kernels/k.py"\n')
    assert "tilefish.probe 3 iter" in body  # the kernel's own probe stays where the frontend put it


def test_the_probe_writes_lane_0_of_the_warp_under_the_capacity():
    asm = _probe.probe_asm(7, "mma.issue")
    assert asm.startswith("// tilefish.probe 7 mma.issue\n{\n") and asm.endswith("}\n")
    for part in ("mov.u64 %tft, %globaltimer;", "mov.u32 %tfc, %clock;", "mov.u32 %tfs, %smid;", "setp.lt.u32 %tfq, %tfp_n, %tfp_cap;",
                 "or.b32 %tfm, %tfm, 7;", "@%tfq st.global.v4.b32 [%tfa], {%tfl, %tfh, %tfc, %tfm};", "add.u32 %tfp_n, %tfp_n, 1;"):
        assert part in asm
    assert "$" not in asm  # (an operand of LLVM's inline assembly: the frontend adds the one dummy output)


def test_the_rewrite_refuses_what_is_not_one_triton_entry():
    with pytest.raises(TilefishBuildError, match="64-bit profile scratch parameter"):
        _probe.instrument_ptx(PTX.replace(".param .u64 .ptr .global .align 1 k_kernel_param_3", ".param .u32 k_kernel_param_3"), 4)
    with pytest.raises(TilefishBuildError, match="not one .entry"):
        _probe.instrument_ptx(PTX + PTX[PTX.index(".visible") :], 4)
    with pytest.raises(TilefishBuildError, match="no ret or exit"):
        _probe.instrument_ptx(PTX.replace("\t@%p1 ret;\n", "").replace("\tret;\n", ""), 4)
    noinline = PTX.replace(".visible .entry", ".func helper(\n\t.param .b32 helper_param_0\n)\n{\n\t// tilefish.probe 3 iter\n\tret;\n}\n\n.visible .entry", 1)
    with pytest.raises(TilefishBuildError, match="in the function 'helper'"):
        _probe.instrument_ptx(noinline, 4)


# -- the probe table --------------------------------------------------------------------------------------------------------------------------


def test_the_probe_table_says_where_each_probe_is_and_which_partition_runs_it():
    table = _probe.probe_sites(SOURCE_IR)
    assert [(p["id"], p["name"]) for p in table] == [(3, "setup"), (4, "compute.wait"), (5, "load.wait")]
    by = {p["name"]: p["sites"] for p in table}
    assert by["setup"] == [{"loc": "kernels/ws.py:22", "role": "default", "function": "ws_kernel"}]  # outside the partitions: the default warps
    assert by["load.wait"] == [{"loc": "kernels/ws.py:6", "role": "worker 0", "function": "kernels.ws._load"}]
    # one name written in two functions: both sites (the callee of an inlined helper at its own line)
    assert by["compute.wait"] == [
        {"loc": "kernels/ws.py:12", "role": "default", "function": "kernels.ws._compute"},
        {"loc": "kernels/helpers.py:3", "role": "worker 0", "function": "kernels.ws._load"},
    ]
    plain = SOURCE_IR.replace("ttg.warp_specialize", "ttg.not_specialized").replace("default {", "region {").replace("partition0(", "other(")
    roles = {s["role"] for p in _probe.probe_sites(plain) for s in p["sites"]}
    assert roles == {""}  # a kernel Gluon does not partition: one role
    assert _probe.probe_sites(None) == [] and _probe.probe_sites("module {}") == []


def test_the_name_of_a_probe():
    for good in ("load", "load.wait", "mma/issue", "_x", "tile:3", "a-b"):
        assert _probe.check_name(good) == good
    for bad in ("", "3load", "has space", "x" * 65, 3, None, "@begin", 'quote"'):
        with pytest.raises(TilefishTypeError, match="the name of a probe"):
            _probe.check_name(bad)


def test_the_settings_of_an_instrumented_build():
    assert _probe.config(True) == _probe.Instrument() and _probe.config(False) is None and _probe.config(None) is None
    cfg = tilefish.instrument(capacity=8, ops=["mm"])
    assert cfg.covers("mm") and not cfg.covers("other") and cfg.block_bytes == 16 + 8 * 16
    assert cfg.key() == {"format": "tilefish.probes.v1", "capacity": 8, "rewrite": _probe.REWRITE}
    for bad in (0, -1, 1 << 17, True, 2.0):
        with pytest.raises(TilefishTypeError, match="capacity"):
            tilefish.instrument(capacity=bad)
    with pytest.raises(TilefishTypeError, match="instrument="):
        _probe.config("yes")


# -- the records --------------------------------------------------------------------------------------------------------------------------------


def slot(cta, warp, count, records, capacity):
    """A warp's slot of a probe buffer: its header and ``records`` (``(id, sm, globaltimer, clock)``), padded to the capacity."""
    data = struct.pack("<IIII", MAGIC, cta, warp, count)
    for pid, sm, t, clk in records:
        data += struct.pack("<QII", t, clk, (sm << 16) | pid)
    return data + b"\xab" * (16 * (capacity - len(records)))


def test_decode_reads_every_warp_and_counts_what_was_dropped():
    cap = 4
    recs = [(BEGIN, 7, 1000, 10), (CALIB, 7, 1010, 30), (CALIB, 7, 1020, 70), (3, 7, 1500, 1000)]
    data = slot(0, 0, 9, recs, cap) + b"\0" * (16 + 16 * cap) + slot(1, 0, 2, recs[:2], cap) + slot(1, 1, 0, [], cap)
    warps, missing = _probe.decode(data, cap, 2, 2)
    assert missing == [(0, 1)]  # a slot without its header (no warp wrote it)
    assert [(w.cta, w.warp, w.count, len(w.records), w.dropped) for w in warps] == [(0, 0, 9, 4, 5), (1, 0, 2, 2, 0), (1, 1, 0, 0, 0)]
    assert warps[0].records[3] == (3, 7, 1500, 1000)
    wrong = slot(1, 0, 1, recs[:1], cap) + slot(0, 0, 1, recs[:1], cap)  # a header of another slot's CTA or warp is not this slot's
    assert _probe.decode(wrong, cap, 2, 1)[1] == [(0, 0), (0, 1)]
    with pytest.raises(TilefishBuildError, match="need"):
        _probe.decode(data[:-1], cap, 2, 2)
    assert _probe.buffer_bytes(cap, 2, 2) == len(data)


PROBES = [
    {"id": 3, "name": "mma.wait", "sites": [{"loc": "kernels/mm.py:40", "role": "worker 1", "function": "kernels.mm._issue"}]},
    {"id": 4, "name": "load.wait", "sites": [{"loc": "kernels/mm.py:30", "role": "worker 0", "function": "kernels.mm._load"}]},
]


def warp(cta, w, sm, t0, steps, cost=40):
    """The records of a warp that starts at ``t0`` (ns) on SM ``sm`` and then makes ``steps``: ``(probe id, ns after the previous)``, with a clock of
    2 GHz and ``@calib`` twice ``cost`` cycles apart."""
    recs = [(BEGIN, sm, t0, 2 * t0), (CALIB, sm, t0 + 10, 2 * t0 + 20), (CALIB, sm, t0 + 10 + cost // 2, 2 * t0 + 20 + cost)]
    t, clk = recs[-1][2], recs[-1][3]
    for pid, dt in steps:
        t, clk = t + dt, clk + 2 * dt
        recs.append((pid, sm, t, clk & 0xFFFFFFFF))
    return WarpRecords(cta, w, len(recs), tuple(recs))


def launch_of(warps, **kw):
    return Launch(stem="mm.M256", arch="sm_103", grid=(2, 1, 1), block=(96, 1, 1), capacity=64, probes=PROBES, warps=warps, **kw)


def test_the_phases_their_durations_the_cost_of_a_probe_and_the_idle_warps():
    warps = [
        warp(0, 0, 5, 1000, [(4, 100), (4, 300), (4, 300), (END, 2000)]),  # the loader
        warp(0, 1, 5, 1000, [(3, 500), (3, 200), (3, 200), (END, 2000)]),  # the issuer
        warp(0, 2, 5, 1000, [(END, 3000)]),  # a warp Triton added: no probe of the kernel
        warp(1, 0, 9, 1200, [(4, 100), (4, 300), (END, 2000)]),
    ]
    launch = launch_of(warps)
    assert launch.clock_ghz() == pytest.approx(2.0) and launch.probe_cycles() == [40, 40, 40, 40]
    phases = launch.phases()
    first = [p for p in phases if p.cta == 0 and p.warp == 0]
    assert [p.name for p in first] == ["@begin -> load.wait", "load.wait -> load.wait", "load.wait -> load.wait", "load.wait -> @end"]
    assert [p.ns for p in first] == pytest.approx([100, 300, 300, 2000]) and first[0].role == "worker 0" and first[0].sm == 5
    assert first[0].start_ns == pytest.approx(30)  # from the launch's first record, at the second @calib (30 ns after @begin)
    assert launch.role_of(warps[1]) == "worker 1" and launch.role_of(warps[2]) == "-"
    s = launch.summary()
    assert (s["ctas"], s["sms"], s["warps"], s["idle_warps"], s["records"], s["dropped"]) == (2, 2, 4, 1, sum(len(w.records) for w in warps), 0)
    assert s["probe_cycles"] == 40 and s["probe_ns"] == pytest.approx(20)
    rows = {(r["phase"], r["role"]): r for r in s["phases"]}
    assert ("@begin -> @end", "-") not in rows  # the idle warp is counted, not a phase
    row = rows[("load.wait -> load.wait", "worker 0")]
    assert row["count"] == 3 and row["ns"]["median"] == pytest.approx(300) and row["cycles_median"] == 600
    assert 0 < s["overhead"] < 0.2
    assert [c["cta"] for c in launch.ctas()] == [0, 1] and launch.ctas()[1]["sm"] == 9


def test_the_trace_file_and_its_views(tmp_path):
    warps = [warp(0, 0, 5, 1000, [(4, 100), (4, 300), (END, 2000)]), warp(0, 1, 5, 1000, [(3, 500), (END, 2000)])]
    trace = ProbeTrace([launch_of(warps, graph="model", step=2, bucket={"M": 256}, manifest_digest="ab" * 32)])
    path = trace.save(tmp_path / "p.json")
    doc = json.loads(path.read_text())
    assert doc["schema"] == "tilefish.probe-trace.v1" and isinstance(doc["launches"][0]["t0"], str)
    back = ProbeTrace.load(path)
    assert back.launches[0].warps == trace.launches[0].warps and back.launches[0].step == 2 and back.launches[0].probes == PROBES
    for bad in ({"schema": "other", "launches": []}, {"schema": "tilefish.probe-trace.v1"}, []):
        with pytest.raises(TilefishProbeError):
            ProbeTrace.from_json(bad)
    broken = json.loads(path.read_text())
    broken["launches"][0]["probes"][0]["id"] = 1  # an id of the build's own probes
    with pytest.raises(TilefishProbeError, match="launch 0"):
        ProbeTrace.from_json(broken)
    text = trace.summary_text(lambda loc: "tilefish.probe(...)" if loc == "kernels/mm.py:30" else None)
    assert "launch 0: mm.M256 on sm_103 (step 2 of the plan of 'model')" in text and "SM clock 2.000 GHz" in text and "40 cycles (20 ns)" in text
    assert "load.wait -> load.wait (worker 0)" in text and "kernels/mm.py:30 [worker 0]   tilefish.probe(...)" in text
    timeline = trace.timeline_text(by="sm")
    assert "SM   5: CTA 0 0.000-" in timeline
    roles = trace.timeline_text(by="role")
    assert "warp  0 worker 0" in roles and "load.wait +100 ns" in roles
    chrome = trace.chrome()
    xs = [e for e in chrome["traceEvents"] if e["ph"] == "X"]
    assert {e["name"] for e in xs} == {"@begin -> load.wait", "load.wait -> load.wait", "load.wait -> @end", "@begin -> mma.wait", "mma.wait -> @end"}
    assert all(e["dur"] > 0 for e in xs) and any(e["ph"] == "M" and e["args"]["name"].endswith("SM 5") for e in chrome["traceEvents"])


def test_the_command_line(tmp_path, capsys):
    warps = [warp(0, 0, 5, 1000, [(4, 100), (END, 2000)])]
    path = ProbeTrace([launch_of(warps)]).save(tmp_path / "p.json")
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "mm.py").write_text("\n" * 29 + "        tilefish.probe('load.wait')\n")
    assert explain.main([str(tmp_path), "--probes", str(path), "--source-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "@begin -> load.wait (worker 0)" in out and "kernels/mm.py:30 [worker 0]   tilefish.probe('load.wait')" in out
    assert explain.main([str(tmp_path), "mm", "--probes", str(path), "--timeline", "cta", "--chrome", str(tmp_path / "c.json")]) == 0
    assert "CTA 0 on SM 5" in capsys.readouterr().out and json.loads((tmp_path / "c.json").read_text())["traceEvents"]
    assert explain.main([str(tmp_path), "--probes", str(path), "--json"]) == 0
    assert {r["phase"] for r in json.loads(capsys.readouterr().out)[0]["phases"]} == {"@begin -> load.wait", "load.wait -> @end"}
    assert explain.main([str(tmp_path), "other", "--probes", str(path)]) == 2
    assert "has no launch of 'other'" in capsys.readouterr().err


# -- a kernel body's probes ---------------------------------------------------------------------------------------------------------------------


@tilefish.kernel
def probed(x: Tensor[N, f32], y: Tensor[N, f32]) -> Tensor[N, f32]:
    a = x * 2.0
    tilefish.probe("scaled")
    return a + y


@tilefish.kernel
def unprobed(x: Tensor[N, f32], y: Tensor[N, f32]) -> Tensor[N, f32]:
    a = x * 2.0
    return a + y


PROBE_LINE = 1 + next(i for i, line in enumerate((HERE / "test_instrument.py").read_text().splitlines()) if 'tilefish.probe("scaled")' in line)


def test_a_kernel_body_marks_a_point_between_its_ops():
    assert probed.graph.probes == (("scaled", 1, ("test_instrument.py", PROBE_LINE)),)
    assert unprobed.graph.probes == () and [n.op for n in probed.graph.nodes] == [n.op for n in unprobed.graph.nodes]
    low = probed.lower()
    lines = low.source.splitlines()
    assert "from tilefish._probe import generated_probe as _tf_probe" in lines
    k = next(i for i, line in enumerate(lines) if line.strip().startswith('_tf_probe("scaled")'))
    assert "* 2.0" in lines[k - 1] or "2.0" in lines[k - 1]  # after the multiply, before the add
    assert lines[k].endswith(f"# probe (test_instrument.py:{PROBE_LINE})")
    assert low.line_map[k] == {"what": "probe scaled", "loc": f"test_instrument.py:{PROBE_LINE}"}
    assert any(line.startswith("# probes: scaled (tilefish.probe") for line in lines)
    # a kernel without probes is lowered exactly as before: no import, no header line
    plain = unprobed.lower().source
    assert "_tf_probe" not in plain and "probes:" not in plain
    kept = [line for i, line in enumerate(lines) if i != k and "_tf_probe" not in line and not line.startswith("# probes:")]
    assert plain.replace("unprobed", "probed") == "\n".join(kept) + "\n"


def test_a_probe_is_in_a_kernel():
    with pytest.raises(TilefishTypeError, match="called outside one"):
        tilefish.probe("x")
    with pytest.raises(TilefishTypeError, match="graph body"):

        @tilefish.graph
        def g(x: Tensor[N, f32]) -> Tensor[N, f32]:
            tilefish.probe("x")
            return probed(x, x)

    with pytest.raises(TilefishTypeError, match="the name of a probe"):

        @tilefish.kernel
        def k(x: Tensor[N, f32]) -> Tensor[N, f32]:
            tilefish.probe("not a name")
            return x * 2.0

    from tilefish._probe import generated_probe

    assert generated_probe("anything") is None  # a generated source run as Python (an interpreter): nothing


def test_no_fused_region_has_a_probe_and_a_probed_kernel_is_a_step_of_its_own():
    """Fusion version 2 (the default, its row regions lowered by revision 2) and probes. A probe is in a kernel body only (a graph body refuses it),
    so no region of either version has one; a ``@tilefish.kernel`` with probes that a graph calls is a step of its own (a call is never fused), with
    its probes; and revision 2 of the lowering puts a kernel's probes where revision 1 puts them."""
    import fusion_graphs as fg
    from tilefish import _fusion
    from tilefish.lower import lower

    R, DR = fg.R, fg.DR

    @tilefish.kernel
    def rows_p(x: Tensor[R, DR, f32]) -> Tensor[R, DR, f32]:
        a = x * 2.0
        tilefish.probe("scaled")
        return a + 1.0

    def body(x, res, w):
        out, h = fg._add_rmsnorm(x, res, w)
        return out, rows_p(h.to(f32) * 0.5)

    @tilefish.graph
    def g2(x: Tensor[R, DR, bf16], res: Tensor[R, DR, bf16], w: Tensor[DR, bf16]):
        return body(x, res, w)

    @tilefish.graph(fusion=tilefish.fusion(version=1))
    def g1(x: Tensor[R, DR, bf16], res: Tensor[R, DR, bf16], w: Tensor[DR, bf16]):
        return body(x, res, w)

    for g, version in ((g2, 2), (g1, 1)):
        f = _fusion.fused(g)
        assert f.settings.version == version
        assert f.regions and all(r.kernel.graph.probes == () and "_tf_probe" not in r.kernel.lower().source for r in f.regions), version
        assert {r.kernel.lowering for r in f.regions} == {version}
        calls = [n.op for n in f.graph.nodes if n.attrs.get("kind") == "kernel" and n.op == "rows_p"]
        assert calls == ["rows_p"], version  # the probed kernel: one step of the plan, not a region's op
    assert any(n.op in ("sum", "mean") for r in _fusion.fused(g2).regions for n in (_fusion.fused(g2).program[k] for k in r.nodes) if len(r.nodes) > 1)

    with pytest.raises(TilefishTypeError, match="graph body"):

        @tilefish.graph(fusion=tilefish.fusion(version=2))
        def refused(x: Tensor[R, DR, f32]) -> Tensor[R, DR, f32]:
            y = x * 2.0
            tilefish.probe("in-a-graph")
            return y + 1.0

    @tilefish.kernel
    def rows_sum_p(x: Tensor[R, DR, f32]) -> Tensor[R, f32]:
        sq = x * x
        tilefish.probe("squared")
        s = tilefish.sum(sq, -1, order=fg.T128)
        tilefish.probe("summed")
        return s * 0.5

    def probes_in(low):
        """The probe calls of a lowered kernel's source, in order, with what its line map says of each."""
        lines = low.source.splitlines()
        at = [i for i, line in enumerate(lines) if line.strip().startswith("_tf_probe(")]
        return [(lines[i].split("#")[0].strip(), low.line_map[i]["what"]) for i in at]

    one, two = lower(rows_sum_p), lower(rows_sum_p, lowering=2)
    assert "revision 2" in two.source and two.source != one.source
    want = [('_tf_probe("squared")', "probe squared"), ('_tf_probe("summed")', "probe summed")]
    assert probes_in(one) == probes_in(two) == want


# -- the instrumented variant ---------------------------------------------------------------------------------------------------------------


@pytest.fixture
def fake(monkeypatch):
    compiler = fake_compile.install(monkeypatch)
    monkeypatch.setattr(runtime, "_modules", {})
    monkeypatch.setattr(runtime, "_gen_root", [])
    monkeypatch.setattr(runtime, "_warned_gen_roots", set())
    monkeypatch.setattr(runtime, "_manifests", {})
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    return compiler


def test_an_instrumented_build_is_a_variant_with_its_own_binary(fake, tmp_path):
    (spec,) = compile_specs(unprobed.lower().extern)
    fn = runtime.load_lowered(unprobed.lower())
    plain = build(spec, arch=103, fn=fn)
    calls = fake.count
    inst = build(spec, arch=103, fn=fn, instrument=tilefish.instrument(capacity=16))
    inst_again = build(spec, arch=103, fn=fn, instrument=tilefish.instrument(capacity=16))
    assert fake.count == calls + 2  # compiled every time: never from the build cache, never stored in it
    assert plain.instrument is None and inst.instrument == {"format": "tilefish.probes.v1", "capacity": 16, "probes": []}
    assert inst.sha256 != plain.sha256 and inst.sha256_code != plain.sha256_code and inst_again.sha256 == inst.sha256
    assert inst.ir_digest == plain.ir_digest  # a kernel without probes: the IR is the production one (only the PTX stage adds @begin, @calib, @end)
    assert "tilefish.probe 0 @begin" in inst.ptx and "%tfp_hdr" not in plain.ptx
    assert build(spec, arch=103, fn=fn).sha256 == plain.sha256  # a production build after it is the production binary
    write_manifest([inst], tmp_path / "inst" / "manifest.json", cubins=True)
    entry = json.loads((tmp_path / "inst" / "manifest.json").read_text())["kernels"][0]
    assert entry["instrument"] == inst.instrument
    write_manifest([plain], tmp_path / "prod" / "manifest.json", cubins=True)
    assert "instrument" not in json.loads((tmp_path / "prod" / "manifest.json").read_text())["kernels"][0]


def test_the_hosts_refuse_an_instrumented_manifest_unless_asked(fake, tmp_path):
    from tilefish import Manifest

    (spec,) = compile_specs(unprobed.lower().extern)
    fn = runtime.load_lowered(unprobed.lower())
    inst = build(spec, arch=103, fn=fn, instrument=tilefish.instrument(capacity=8))
    write_manifest([inst], tmp_path / "manifest.json", cubins=True)
    with pytest.raises(TilefishManifestError, match="is an instrumented export"):
        Manifest.load(tmp_path)
    loaded = Manifest.load(tmp_path, instrumented=True)
    assert loaded.instrumented and loaded.get(spec.stem, "sm_103").entry["instrument"]["capacity"] == 8
    with pytest.raises(TilefishManifestError, match="is an instrumented export"):
        runtime._load_manifest(tmp_path)
    with pytest.raises(TilefishManifestError, match="instrumented export"):
        runtime._load_manifest(loaded)
    assert runtime._load_manifest(tmp_path, instrumented=True).instrumented
    from tilefish.hosts import rust

    with pytest.raises(TilefishManifestError, match="instrumented export"):
        rust.emit(tmp_path, tmp_path / "out.rs", pin=loaded.digest)


def test_an_entry_says_its_probes_as_both_readers_read_them(fake, tmp_path):
    from tilefish import Manifest

    (spec,) = compile_specs(unprobed.lower().extern)
    fn = runtime.load_lowered(unprobed.lower())
    inst = build(spec, arch=103, fn=fn, instrument=tilefish.instrument(capacity=8))
    write_manifest([inst], tmp_path / "manifest.json", cubins=True)
    good = json.loads((tmp_path / "manifest.json").read_text())
    probe = {"id": 3, "name": "a", "sites": [{"loc": "k.py:1", "role": "", "function": "k"}]}
    cases = [
        ("format", {"format": "tilefish.probes.v0", "capacity": 8, "probes": []}),
        ("capacity", {"format": "tilefish.probes.v1", "capacity": 0, "probes": []}),
        ("capacity", {"format": "tilefish.probes.v1", "capacity": True, "probes": []}),
        ("object of format", {"format": "tilefish.probes.v1", "capacity": 8, "probes": [], "extra": 1}),
        ("id", {"format": "tilefish.probes.v1", "capacity": 8, "probes": [{**probe, "id": 2}]}),
        ("id", {"format": "tilefish.probes.v1", "capacity": 8, "probes": [probe, probe]}),
        ("name", {"format": "tilefish.probes.v1", "capacity": 8, "probes": [probe, {**probe, "id": 4}]}),
        ("sites", {"format": "tilefish.probes.v1", "capacity": 8, "probes": [{**probe, "sites": [{"loc": 1, "role": "", "function": ""}]}]}),
        ("object of id", {"format": "tilefish.probes.v1", "capacity": 8, "probes": [{**probe, "more": 1}]}),
    ]
    for match, field in cases:
        doc = json.loads(json.dumps(good))
        doc["kernels"][0]["instrument"] = field
        (tmp_path / "manifest.json").write_text(json.dumps(doc))
        with pytest.raises(TilefishManifestError, match=match):
            Manifest.load(tmp_path, instrumented=True)
    doc = json.loads(json.dumps(good))
    doc["kernels"][0]["instrument"]["probes"] = [probe]
    (tmp_path / "manifest.json").write_text(json.dumps(doc))
    assert Manifest.load(tmp_path, instrumented=True).doc["kernels"][0]["instrument"]["probes"] == [probe]
    assert _probe.check_field("e", {"format": "tilefish.probes.v1", "capacity": 8, "probes": [probe]}) is None


def test_an_instrumented_export_is_never_merged_with_a_production_one(fake, tmp_path):
    prod = runtime.export(unprobed, tmp_path / "out", arch=103)
    with pytest.raises(TilefishManifestError, match="is a production export, and this export is an instrumented one"):
        runtime.export(unprobed, tmp_path / "out", arch=103, instrument=True)
    inst = runtime.export(unprobed, tmp_path / "inst", arch=103, instrument=True)
    assert inst.instrumented and not prod.instrumented and inst.digest != prod.digest
    with pytest.raises(TilefishManifestError, match="is an instrumented export, and this export is a production one"):
        runtime.export(unprobed, tmp_path / "inst", arch=103)
    only = runtime.export([unprobed, probed], tmp_path / "only", arch=103, instrument=tilefish.instrument(ops=["probed"]))
    kinds = {e["op"]: "instrument" in e for e in only.doc["kernels"]}
    assert kinds == {"unprobed": False, "probed": True}  # ops= instruments those operators only (the manifest is an instrumented export)


def test_a_run_with_probes_needs_a_kernel_and_an_instrumented_export(fake, tmp_path):
    prod = runtime.export(unprobed, tmp_path / "out", arch=103)
    with pytest.raises(tilefish.TilefishRuntimeError, match="probes= takes a tilefish.Probes"):
        tilefish.run(unprobed, 1, 2, probes=object())
    with pytest.raises(tilefish.TilefishRuntimeError, match="reference mode runs no binary"):
        tilefish.run(unprobed, 1, 2, mode="reference", probes=tilefish.Probes())
    with pytest.raises(TilefishManifestError, match="is a production one"):  # (refused before the arguments are looked at)
        tilefish.run(unprobed, None, None, mode="replay", manifest=tmp_path / "out", pin=prod.digest, probes=tilefish.Probes())
