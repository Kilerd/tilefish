"""The persistent, warp-specialised GEMM of ``python/examples/persistent_gemm.py`` on a GPU (``docs/en/rfcs/0003-epilogue-fusion.md``, experiment X8).

* its bytes are cuBLASLt's pinned GEMM's (``heuristic="torch"``) wherever the pin does not split ``k``, in every layout and both 16-bit types,
  at every bucket (sm_100 and sm_103: X3 and X8 measured it there; X7, the same question on sm_120, is a later gate's);
* every configuration writes the same bytes (the tile, the stages, the epilogue's partitions and warps), and its Triton form
  (``gemm_sm120_kernel``, the sm_120 kernel, here compiled to ``tcgen05``) writes them too;
* every elementwise pattern of E1 fused into its hooks writes the bytes of the unfused plan (contract (i)) and, under
  ``epilogues="accumulator"``, of the unfused plan of ``R_acc(P)`` (contract (ii)), at every bucket, with the special values of every dtype in the
  side inputs, at sizes where each program walks one tile and several, and through ``tilefish.verify_fusion``; a norm is refused by the Gluon
  kernel (E1: no row regions in a Gluon hook) and fused into the Triton one where a sub-tile holds whole rows;
* the fused binary has the GEMM's MMAs; the sm_120 builds hold their SASS assertions (``mma.sync m16n8k16``, no TF32); the probes time the
  pipeline's partitions;
* a speed smoke test against cuBLASLt with wide margins (it guards against a regression by a factor, not a percent).
"""

import inspect
import pathlib
import sys

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import epilogue_graphs as eg  # noqa: E402
from gpu_util import differing, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, bf16, f16, f32, runtime  # noqa: E402
from tilefish import _cublaslt as lt  # noqa: E402
from tilefish import _fusion  # noqa: E402
from tilefish._launch import eval_dim  # noqa: E402

CUDA = "cuda"
CAP = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1] if torch.cuda.is_available() else 0
SMS = torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 1
TCGEN05 = CAP in (100, 103)
TORCH = {"bf16": torch.bfloat16, "f16": torch.float16}
needs_tcgen05 = pytest.mark.skipif(not TCGEN05, reason="the Gluon kernel is a tcgen05 one (sm_100, sm_103)")
needs_arch = pytest.mark.skipif(CAP not in (100, 103, 120), reason="the example is declared for sm_100, sm_103 and sm_120")
EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def pg():
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    import persistent_gemm

    return persistent_gemm


M = Dim("M", buckets=[32, 200, 1000])  # tiles of 64 rows (M = 32) and of 128 (a tail at 200 and 1000; several tiles a program at 1000)


def operands(m, n, k, dt, layout, seed=0):
    """``a`` and ``b`` as stored in ``layout``, and the f32 product (for a tolerance where no pin is the reference)."""
    g = torch.Generator(device=CUDA).manual_seed(seed * 7919 + m + n + k)
    a = torch.randn(m, k, device=CUDA, generator=g).to(TORCH[dt])
    b = (torch.randn(n, k, device=CUDA, generator=g) / k**0.5).to(TORCH[dt])
    stored_a = a if layout[1] == "N" else a.t().contiguous()
    stored_b = b if layout[0] == "T" else b.t().contiguous()
    return stored_a, stored_b, a.float() @ b.float().T


def pinned(stored_a, stored_b, m, n, k, dt, layout):
    """cuBLASLt's pinned GEMM (torch posture) on the same operands, and its configuration."""
    form = lt.Form(dt, dt, layout)
    p = lt.pin_algo(m, n, k, False, heuristic="torch", form=form, workspace_bytes=lt.DEFAULT_WORKSPACE_BYTES)
    d = torch.empty(m, n, device=CUDA, dtype=TORCH[dt])
    lt.matmul(stored_a.data_ptr(), stored_b.data_ptr(), d.data_ptr(), None, m, n, k, p.algo_hex, workspace_bytes=p.workspace_limit,
              stream=torch.cuda.current_stream().cuda_stream, form=form)  # fmt: skip
    torch.cuda.synchronize()
    return d, lt.algo_info(p.algo_hex)


def assert_same(x, y, what):
    assert same_bytes(x, y), f"{what}: {differing(x, y)} of {x.numel()} elements differ"


# -- the GEMM ------------------------------------------------------------------------------------------------------------------------------------


@needs_tcgen05
@pytest.mark.parametrize("layout,dt", [("TN", "bf16"), ("NN", "bf16"), ("NT", "f16"), ("TT", "f16")])
def test_the_gemm_writes_the_bytes_of_the_cublaslt_pin_at_every_bucket(layout, dt):
    """X3 and X8 on sm_103: no split of ``k``, so the bytes are cuBLASLt's whenever its pin does not split it (the pin's ``splitk_num`` 1)."""
    n, k = 1024, 768
    op = pg().gemm(CAP, M=M, n=n, k=k, layout=layout, dtype=bf16 if dt == "bf16" else f16, sm_count=SMS, name=f"pgt_{layout}_{dt}")
    for m in M.buckets:
        sa, sb, _ = operands(m, n, k, dt, layout)
        got = tilefish.run(op, sa, sb)
        want, cfg = pinned(sa, sb, m, n, k, dt, layout)
        if cfg["splitk_num"] != 1:  # (cuBLASLt 12.8.4 pins none here; another version may)
            continue
        assert_same(got, want, f"{layout} {dt} M={m}")


@needs_tcgen05
def test_every_configuration_writes_the_same_bytes():
    """The tile, the stages, the epilogue's sub-tile, warps and partitions change the speed, never the bytes (X3's check, on this kernel)."""
    n, k = 1024, 768
    base = {"BK": 64, "GROUP_M": 8}
    configs = [
        {**base, "BM": 64, "BN": 32, "STAGES": 8, "EPI_N": 32, "NW": 4, "PARTS": 1},
        {**base, "BM": 64, "BN": 128, "STAGES": 4, "EPI_N": 64, "NW": 8, "PARTS": 2},
        {**base, "BM": 128, "BN": 64, "STAGES": 3, "EPI_N": 64, "NW": 4, "PARTS": 1},
        {**base, "BM": 128, "BN": 256, "STAGES": 4, "EPI_N": 32, "NW": 8, "PARTS": 2, "GROUP_M": 1},
    ]
    seen = None
    for i, cfg in enumerate(configs):
        op = pg().gemm(CAP, M=Dim("M", buckets=[1000]), n=n, k=k, config=cfg, name=f"pgc{i}")
        sa, sb, _ = operands(1000, n, k, "bf16", "TN")
        got = tilefish.run(op, sa, sb)
        if seen is None:
            seen = got
            want, info = pinned(sa, sb, 1000, n, k, "bf16", "TN")
            if info["splitk_num"] == 1:
                assert_same(got, want, "against the pin")
        else:
            assert_same(got, seen, f"configuration {cfg}")


@needs_arch
def test_the_triton_kernel_writes_the_bytes_of_the_gluon_one():
    """``gemm_sm120_kernel`` (sm_120's) runs here too, its ``tl.dot`` the device's MMA: on sm_100 and sm_103 the bytes of the Gluon kernel (and of the
    pin), on sm_120 the f32 product within a bf16 rounding; the dual GEMM is the plain one over both halves."""
    for layout in ("TN", "NT"):
        tri = pg().gemm(CAP, M=M, n=1024, k=768, layout=layout, impl="triton", name=f"pgtri_{layout}")
        glu = pg().gemm(CAP, M=M, n=1024, k=768, layout=layout, sm_count=SMS, name=f"pgglu_{layout}") if TCGEN05 else None
        for m in M.buckets:
            sa, sb, ref = operands(m, 1024, 768, "bf16", layout)
            got = tilefish.run(tri, sa, sb)
            if glu is not None:
                assert_same(got, tilefish.run(glu, sa, sb), f"triton against gluon, {layout} M={m}")
            else:
                assert torch.allclose(got.float(), ref, rtol=1e-2, atol=1e-2)
    dual = pg().gemm(CAP, M=M, n=2048, k=768, split=2, impl="triton", name="pgtri_dual")
    plain = pg().gemm(CAP, M=M, n=2048, k=768, impl="triton", name="pgtri_plain")
    for m in M.buckets:
        sa, sb, _ = operands(m, 2048, 768, "bf16", "TN")
        assert_same(tilefish.run(dual, sa, sb), tilefish.run(plain, sa, sb), f"dual M={m}")


# -- epilogue hooks ------------------------------------------------------------------------------------------------------------------------------


def args_of(graph, combo, seed):
    """The extern's operands normal; every other tensor with the corners of its dtype (NaNs with payloads, infinities, subnormals, zeros) first."""
    out = []
    dims = {d: combo[d.name] for p in graph.signature.tensor_inputs for s in p.type.shape for d in s.free_dims() if d.name in combo}
    for k, p in enumerate(graph.signature.tensor_inputs):
        sizes = tuple(int(eval_dim(s, dims)) for s in p.type.shape)
        if p.name in ("x", "w"):
            gen = torch.Generator(device="cpu").manual_seed(seed * 100 + k)
            out.append((torch.randn(sizes, generator=gen) * (1.0 if p.name == "x" else 1.0 / 16)).to(TORCH[p.type.dtype.name]).to(CUDA))
        else:
            out.append(_fusion._sample(torch, p.type, sizes, seed * 1000 + k, CUDA))
    return out


def run(graph, args):
    got = tilefish.run(graph, *[a.clone() for a in args])
    return list(got) if isinstance(got, tuple) else [got]


def check_modes(modes, what, seeds=(0, 1)):
    for mode, ref in (("fused", "unfused"), ("acc", "acc_unfused")):
        f = _fusion.fused(modes[mode])
        assert len(f.hooked) == 1, (what, mode, f.refused)
        for combo in tilefish._plan.bucket_combinations(tilefish.plan(modes[mode]).dims):
            for seed in seeds:
                args = args_of(modes[mode], combo, seed)
                got, want = run(modes[mode], args), run(modes[ref], args)
                for j, (x, y) in enumerate(zip(got, want)):
                    assert_same(x, y, f"{what} {mode} at {combo}, seed {seed}, output {j}")


ELEMENTWISE = ("bias_gelu", "bias_silu", "bias_gelu_tanh", "periodic", "swiglu")


@pytest.fixture(scope="module")
def small():
    """The example's GEMM and dual GEMM over the dims of ``epilogue_graphs`` (rows ``T`` in buckets of 64 and 200)."""
    if CAP not in (100, 103, 120):
        pytest.skip("the example is declared for sm_100, sm_103 and sm_120")
    g = pg().gemm(CAP, M=eg.T, n=256, k=256, sm_count=SMS, name="pg_epi")
    d = pg().gemm(CAP, M=eg.T, n=512, k=256, split=2, sm_count=SMS, name="pg_epi_dual")
    return eg.build((g, d, eg.standin_conv("pg_conv_unused")), prefix="pg")


@pytest.mark.parametrize("pattern", ELEMENTWISE)
def test_every_elementwise_pattern_fused_writes_the_bytes_of_its_reference(small, pattern):
    check_modes(small[pattern], pattern)
    checked = tilefish.verify_fusion(small[pattern]["acc"])  # the pair against the extern and the region's ops, special values in the operands too
    (r,) = [r for r in checked["regions"] if r.get("route") == "hook"]
    assert r["verified"]["runs"] >= 2 and r["verified"]["against"] == "U(R_acc(P))"


@needs_arch
def test_patterns_where_each_program_walks_several_tiles():
    """Rows and columns enough that the persistent programs take several tiles each (both accumulator buffers, both epilogue partitions)."""
    R = Dim("R", buckets=[1000, 3000])
    g = pg().gemm(CAP, M=R, n=1024, k=256, sm_count=SMS, name="pg_big")
    d = pg().gemm(CAP, M=R, n=2048, k=256, split=2, sm_count=SMS, name="pg_big_dual")
    H, F = Dim("Hb", 256), Dim("Fb", 1024)

    def bias_gelu(x, w, b):
        return tilefish.gelu(g(x, w).to(f32) + b.to(f32)).to(bf16)

    def swiglu(x, w):
        gg, u = d(x, w).split(F, axis=-1)
        return (tilefish.silu(gg.to(f32)) * u.to(f32)).to(bf16)

    for name, body, params in (("bias_gelu", bias_gelu, [("x", Tensor[R, H, bf16]), ("w", Tensor[F, H, bf16]), ("b", Tensor[F, bf16])]),
                               ("swiglu", swiglu, [("x", Tensor[R, H, bf16]), ("w", Tensor[2 * F, H, bf16])])):  # fmt: skip
        modes = {mode: eg.graph_of(body, params, fusion, f"pgbig_{name}_{mode}") for mode, fusion in eg.MODES.items()}
        check_modes(modes, name, seeds=(3,))


@needs_arch
def test_a_norm_is_refused_by_the_gluon_kernel_and_fused_into_the_triton_one():
    """E1 fuses no row region into a Gluon kernel's hook (its tile has the kernel's layout); the Triton kernel's hook sees whole rows where its
    epilogue's sub-tile is as wide as the row (``block_n="EPI_N"``)."""
    if TCGEN05:
        glu = pg().gemm(CAP, M=eg.T, n=128, k=256, sm_count=SMS, name="pg_norm_glu")
        f = _fusion.fused(eg.graph_of(*eg.bodies(glu, None, None)["add_norm"], tilefish.fusion(), "pg_norm_glu_g"))
        assert not f.hooked and f.refused[0]["reason"] == "epilogue-row"
    cfg = {"BM": 128, "BN": 128, "BK": 64, "STAGES": 2, "EPI_N": 128, "GROUP_M": 8, "NW": 8}
    tri = pg().gemm(CAP, M=eg.T, n=128, k=256, impl="triton", config=cfg, name="pg_norm_tri")
    body, params = eg.bodies(tri, None, None)["add_norm"]
    modes = {mode: eg.graph_of(body, params, fusion, f"pg_norm_{mode}") for mode, fusion in eg.MODES.items()}
    check_modes(modes, "add_norm (triton)")


@needs_tcgen05
def test_the_fused_op_has_the_gemms_mmas(small):
    """The hook leaves the main loop alone: the fused binary's MMA instructions are the GEMM's, in order."""
    from tilefish import _sass
    from tilefish._build import build

    (h,) = _fusion.fused(small["periodic"]["fused"]).hooked

    def mmas(op):
        t = runtime._target(op)
        spec = t.compile_specs()[0]
        b = build(spec, arch=CAP, fn=t.jit_function(spec))
        try:
            d = _sass.disassemble(b.cubin, ptx=b.ptx, options=b.options)
        except tilefish.TilefishBuildError:
            pytest.skip("no nvdisasm that reads the binary")
        return [".".join((i.opcode, *i.modifiers)) for i in d.listing.instructions() if i.opcode in ("UTCHMMA", "UTCQMMA", "HMMA")]

    a, b = mmas(h.extern), mmas(h.op)
    assert a and a == b


def test_the_sm120_builds_hold_their_sass_assertions(tmp_path):
    """Built for sm_120 on any machine (ptxas 12.8 or later): the GEMM, its dual form and a fused bias + GELU, in every layout; each binary is held to
    the declaration's ``mma.sync m16n8k16`` and no TF32 at the build."""
    T = Dim("T120", buckets=[64, 2048])
    ops = [pg().gemm(120, M=T, n=1024 * s, k=768, layout=layout, split=s, name=f"pg120_{layout}_{s}") for layout in ("TN", "NN", "NT", "TT") for s in (1, 2)]
    g = ops[0]

    @tilefish.graph
    def fused(x: Tensor[T, 768, bf16], w: Tensor[1024, 768, bf16], b: Tensor[1024, bf16]) -> Tensor[T, 1024, bf16]:
        return tilefish.gelu(g(x, w).to(f32) + b.to(f32)).to(bf16)

    try:
        runtime.export(ops, tmp_path / "ops", arch=120)
        runtime.export(fused, tmp_path / "fused", arch=120, verify=False)
    except tilefish.TilefishBuildError as e:
        if "nvdisasm" in str(e) or "ptxas" in str(e) and "sm_120" in str(e):
            pytest.skip(f"no toolchain for sm_120 here: {str(e).splitlines()[0]}")
        raise


@needs_tcgen05
def test_the_probes_time_the_pipeline():
    """The instrumented build records the producer's waits for a free slot, the issuer's for a full one and the epilogue's for an accumulator: one
    ``mma.issue`` per step of ``k`` per tile, one ``epi.tile`` per tile and warp of the epilogue."""
    m, n, k = 1000, 1024, 768
    op = pg().gemm(CAP, M=Dim("M", buckets=[m]), n=n, k=k, sm_count=SMS, name="pg_probes")
    sa, sb, _ = operands(m, n, k, "bf16", "TN")
    rec = tilefish.Probes(tilefish.instrument(capacity=4096))
    got = tilefish.run(op, sa, sb, probes=rec)
    assert_same(got, tilefish.run(op, sa, sb), "instrumented")
    (launch,) = rec.trace().launches
    phases = {(p.start, p.end) for p in launch.phases()}
    for want in (("load.wait", "load.issue"), ("mma.wait", "mma.issue"), ("epi.wait", "epi.tile")):
        assert want in phases, (want, sorted(phases))
    assert not any(w.dropped for w in launch.warps)
    ids = {name: i for i, name in launch.names().items()}
    cfg = op.bucket_configs[(("M", m),)]["consts"]
    tiles = -(-m // cfg["BM"]) * -(-n // cfg["BN"])
    issues = sum(1 for w in launch.warps for r in w.records if r[0] == ids["mma.issue"])
    tiles_seen = sum(1 for w in launch.warps for r in w.records if r[0] == ids["epi.tile"])
    assert issues == tiles * (k // 64), (issues, tiles)
    assert tiles_seen == tiles * cfg["NW"], (tiles_seen, tiles)  # (every warp of the epilogue partition that takes the tile)


@needs_tcgen05
def test_speed_smoke_against_cublaslt():
    """Margins of a factor (the measured ratios are in RFC 0003, X8): the GEMM alone within 1.5 times cuBLASLt's pin, and bias + GELU fused into it
    faster than cuBLASLt followed by the region's kernel (route (a))."""
    R, H, F = Dim("Rs", buckets=[2048]), 1024, 4096
    g = pg().gemm(CAP, M=R, n=F, k=H, sm_count=SMS, name="pg_speed")

    def lib(n, k):
        def stub(a: Tensor[R, k, bf16], b: Tensor[n, k, bf16]) -> Tensor[R, n, bf16]: ...

        return tilefish.library("pg_speed_lt", acc=f32, impl="cublaslt.gemm", heuristic="torch")(stub)

    lgemm = lib(F, H)

    def graph(name, body, params):
        def fn(**kw):
            return body(*[kw[p] for p, _ in params])

        fn.__signature__ = inspect.Signature([inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=t) for p, t in params])
        fn.__name__ = name
        return tilefish.graph(fn, name=name)

    P = [("x", Tensor[R, H, bf16]), ("w", Tensor[F, H, bf16])]
    plain = graph("pgs_plain", lambda x, w: g(x, w), P)
    plain_lt = graph("pgs_plain_lt", lambda x, w: lgemm(x, w), P)
    hook = graph("pgs_hook", lambda x, w, b: tilefish.gelu(g(x, w).to(f32) + b.to(f32)).to(bf16), P + [("b", Tensor[F, bf16])])
    route_a = graph("pgs_a", lambda x, w, b: tilefish.gelu(lgemm(x, w).to(f32) + b.to(f32)).to(bf16), P + [("b", Tensor[F, bf16])])

    def us(gr):
        return tilefish.profile_plan(gr, reps=50, warmup=10)["plan_us"]["trimmed"]

    assert us(plain) < 1.5 * us(plain_lt)
    assert us(hook) < us(route_a)
