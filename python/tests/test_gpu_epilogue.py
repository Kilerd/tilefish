"""Epilogue hooks on a GPU (``docs/rfcs/0003-epilogue-fusion.md``, milestone E1): an extern with a region fused into its hook writes the bytes of
its reference, for every input.

* every pattern of E1 (``epilogue_graphs.py``: a GEMM followed by a bias and GELU, SiLU, tanh GELU or a periodic activation; the dual-accumulator
  GEMM and a gated activation; a GEMM, a residual add and an rmsnorm with the residual stream a side output; a convolution as an implicit GEMM, a
  bias and an activation) over the example externs (``python/examples/epilogue_gemm.py``) of the device's architecture: fused against unfused
  (contract (i), ``F == U(P)``) and, under ``epilogues="accumulator"``, against the unfused composition of ``R_acc(P)`` (contract (ii)), at every
  bucket, on seeded inputs with special values in every side input, and through ``tilefish.verify_fusion`` (special values in the operands too);
* every finite bf16 value as a tile (a GEMM by an identity matrix holds each, exactly) through the activations' hooks;
* random regions after a GEMM (a property test), in both modes;
* the Gluon GEMM's hooks;
* ``export`` verifies every pair and refuses one whose hook is not applied to the tile the kernel stores; the plan replays and captures as it runs;
  the fused op's binary has the extern's MMAs; ``reference`` mode is close to the fused plan.

Nothing here is specific to sm_103: the tests take the example's configuration of the device's architecture (``mma.sync`` on sm_120), and the Gluon
GEMM's tests need sm_100 or sm_103.
"""

import json
import random

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import epilogue_graphs as eg  # noqa: E402
from gpu_util import differing, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import DevScalar, Dim, Tensor, bf16, f16, f32, i32, runtime  # noqa: E402
from tilefish import _fusion  # noqa: E402
from tilefish._launch import eval_dim  # noqa: E402

CUDA = "cuda"
CAP = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1] if torch.cuda.is_available() else 0
TORCH = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16, "i32": torch.int32}


@pytest.fixture(scope="module")
def externs():
    return eg.examples(CAP)


@pytest.fixture(scope="module")
def graphs(externs):
    return eg.build(externs, prefix="g")


def args_of(graph, combo, seed):
    """Arguments of a graph at a bucket combination: the extern's operands normal (their results of order 1), every other tensor seeded with
    every corner of its dtype (NaNs with payloads, infinities, subnormals, zeros) at its first elements."""
    out = []
    dims = {d: combo[d.name] for p in graph.signature.tensor_inputs for s in p.type.shape for d in s.free_dims() if d.name in combo}
    for k, p in enumerate(graph.signature.tensor_inputs):
        sizes = tuple(int(eval_dim(s, dims)) for s in p.type.shape)
        if p.is_devscalar:
            out.append(torch.tensor([5 + seed], dtype=TORCH[p.type.dtype.name], device=CUDA))
        elif p.name in ("x", "w"):  # the extern's operands
            gen = torch.Generator(device="cpu").manual_seed(seed * 100 + k)
            scale = 1.0 if p.name == "x" else 1.0 / 16
            out.append((torch.randn(sizes, generator=gen) * scale).to(TORCH[p.type.dtype.name]).to(CUDA))
        else:
            out.append(_fusion._sample(torch, p.type, sizes, seed * 1000 + k, CUDA))
    return out


def run(graph, args):
    got = tilefish.run(graph, *[a.clone() for a in args])
    return list(got) if isinstance(got, tuple) else [got]


def assert_same(got, want, what):
    assert len(got) == len(want), what
    for k, (x, y) in enumerate(zip(got, want)):
        assert same_bytes(x, y), f"{what}: output {k}: {differing(x, y)} of {x.numel()} elements differ"


@pytest.mark.parametrize("pattern", sorted(eg.bodies(*eg.standins())))
def test_every_pattern_fused_writes_the_bytes_of_its_reference_at_every_bucket(graphs, pattern):
    modes = graphs[pattern]
    for mode, ref in (("fused", "unfused"), ("acc", "acc_unfused")):
        f = _fusion.fused(modes[mode])
        assert len(f.hooked) == 1 and len(tilefish.plan(modes[mode]).steps(**{d: v[0] for d, v in tilefish.plan(modes[mode]).dims.items()})) == 1, (pattern, mode)
        for combo in tilefish._plan.bucket_combinations(tilefish.plan(modes[mode]).dims):
            for seed in (0, 1):
                args = args_of(modes[mode], combo, seed)
                assert_same(run(modes[mode], args), run(modes[ref], args), f"{pattern} {mode} at {combo}, seed {seed}")
        checked = tilefish.verify_fusion(modes[mode])  # the pair against the extern and the region's ops, special values in the operands too
        (r,) = [r for r in checked["regions"] if r.get("route") == "hook"]
        assert r["verified"]["runs"] >= 2 and r["verified"]["against"] == ("U(R_acc(P))" if r["contract"] == "accumulator" else "U(P)")


def test_a_gemm_whose_epilogue_runs_on_sub_tiles_fuses_the_elementwise_patterns(externs):
    """Each hook called once per sub-tile of the columns (fewer registers live): the same bytes; a norm is refused (no hook sees a whole row)."""
    sub = eg.examples_subtiled(CAP)
    built = eg.build((sub, externs[1], externs[2]), prefix="sub")
    for pattern in ("bias_gelu", "periodic", "add_norm_f32"):
        modes = built[pattern]
        f = _fusion.fused(modes["fused"])
        if pattern == "add_norm_f32":
            assert not f.hooked and f.refused[0]["reason"] == "epilogue-row"
            continue
        for mode, ref in (("fused", "unfused"), ("acc", "acc_unfused")):
            for combo in tilefish._plan.bucket_combinations(tilefish.plan(modes[mode]).dims):
                args = args_of(modes[mode], combo, 6)
                assert_same(run(modes[mode], args), run(modes[ref], args), f"sub-tiles {pattern} {mode} at {combo}")


def test_a_chunk_kept_an_unaligned_bias_and_two_hooked_calls_write_the_bytes_of_their_reference(externs):
    """A split hook's chunk the graph returns too (the kernel stores the result, the chunk a view of it after the call), a bias at an element offset
    that is no multiple of 16 bytes (copied for the hook), and two hooked calls whose regions each read both results (the first takes one, the
    other stays after both): each plan the bytes of the unfused one."""
    gemm, dual, _ = externs
    X = [("x", Tensor[eg.T, eg.H, bf16]), ("w", Tensor[eg.F, eg.H, bf16])]

    def chunk(x, w):
        g, u = dual(x, w).split(eg.F, axis=-1)
        return (tilefish.silu(g.to(f32)) * u.to(f32)).to(bf16), g

    def unaligned(x, w, b2):
        _, bb = b2.split([1, 256], axis=-1)
        return (gemm(x, w).to(f32) + bb.to(f32)).to(bf16)

    def two_calls(x, w):
        y1, y2 = gemm(x, w), gemm(x, w)
        return y1 * y2, y2 - y1

    cases = {"chunk": (chunk, [X[0], ("w", Tensor[2 * eg.F, eg.H, bf16])]), "unaligned": (unaligned, [*X, ("b2", Tensor[257, bf16])]), "two_calls": (two_calls, X)}
    for name, (body, params) in cases.items():
        fused = eg.graph_of(body, params, tilefish.fusion(), f"rv_{name}")
        unfused = eg.graph_of(body, params, tilefish.fusion(grouping="none"), f"rv_{name}_unfused")
        assert len(_fusion.fused(fused).hooked) == 1, name
        for combo in tilefish._plan.bucket_combinations(tilefish.plan(fused).dims):
            for seed in (0, 1):
                args = args_of(fused, combo, seed)
                assert_same(run(fused, args), run(unfused, args), f"{name} at {combo}, seed {seed}")
        assert any(r.get("route") == "hook" and r["verified"]["runs"] >= 2 for r in tilefish.verify_fusion(fused)["regions"]), name


def test_the_accumulator_mode_has_one_rounding_fewer_and_other_bytes(graphs):
    """Contract (ii) is not contract (i): R_acc removes the rounding of the GEMM's result, which changes about a third of a bias + activation's
    outputs (RFC 0003, X1); and it is at least as close to the f32 composition."""
    g = graphs["bias_gelu"]
    args = args_of(g["fused"], {"T": 200}, 3)
    args[2] = torch.randn(args[2].shape, generator=torch.Generator(device="cpu").manual_seed(9)).mul(0.5).to(torch.bfloat16).to(CUDA)  # a finite bias
    exact, acc = run(g["fused"], args)[0], run(g["acc"], args)[0]
    x, w, b = (a.float() for a in args)
    y32 = x @ w.T
    f64 = torch.nn.functional.gelu(y32.double() + b.double())
    changed = differing(exact, acc) / exact.numel()
    assert 0.05 < changed < 0.6, changed
    err_exact = (exact.double() - f64).abs().mean().item()
    err_acc = (acc.double() - f64).abs().mean().item()
    assert err_acc <= err_exact * 1.01, (err_acc, err_exact)


# -- every finite bf16 value as a tile --------------------------------------------------------------------------------------------------

TA = Dim("TA", buckets=[256])
HA = Dim("HA", 256)


def test_every_finite_bf16_tile_value_through_the_activations(externs):
    """A GEMM of the bf16 bit patterns by an identity matrix holds each finite one exactly as a tile (x * 1 plus zeros): the bias + activation hooks
    on every one of them (the rows of NaN and infinity patterns are NaN, through the identity's zeros), fused against unfused, both modes."""
    gemm = externs[0]
    patterns = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(256, 256).to(CUDA)
    eye = torch.eye(256, dtype=torch.bfloat16, device=CUDA)
    bias = _fusion._sample(torch, tilefish.TensorType((256,), bf16), (256,), 7, CUDA)
    acts = {
        "gelu": lambda h: tilefish.gelu(h),
        "silu": lambda h: tilefish.silu(h),
        "tanh_gelu": eg._gelu_tanh,
        "periodic": lambda h: h + tilefish.sin(0.75 * h) * tilefish.sin(0.75 * h) / 0.75,
        "ieee_exp": lambda h: tilefish.exp(h, math="ieee") / (h + 3.0),
    }
    for name, act in acts.items():
        body = lambda x, w, b, act=act: act(gemm(x, w).to(f32) + b.to(f32)).to(bf16)  # noqa: E731
        params = [("x", Tensor[TA, HA, bf16]), ("w", Tensor[HA, HA, bf16]), ("b", Tensor[HA, bf16])]
        out = {mode: eg.graph_of(body, params, fusion, f"every_{name}_{mode}") for mode, fusion in eg.MODES.items()}
        assert len(_fusion.fused(out["fused"]).hooked) == 1
        assert_same(run(out["fused"], [patterns, eye, bias]), run(out["unfused"], [patterns, eye, bias]), f"{name} exact")
        assert_same(run(out["acc"], [patterns, eye, bias]), run(out["acc_unfused"], [patterns, eye, bias]), f"{name} accumulator")
    # the tiles held every normal pattern of the finite rows: the GEMM by the identity is exact there (a zero's sign, a subnormal: the MMA's)
    y = gemm.run(patterns, eye)
    rows = torch.isfinite(patterns.float()).all(dim=1)
    normal = patterns.float().abs() >= 2.0**-126
    keep = normal & rows[:, None]
    assert int(keep.sum()) > 60000 and torch.equal(y[keep].view(torch.int16), patterns[keep].view(torch.int16))


# -- random regions after a GEMM ----------------------------------------------------------------------------------------------------------

TR = Dim("TR", buckets=[64, 200])
FR = Dim("FR", 128)


def random_body(seed, gemm):
    """A random arithmetic DAG of 4 to 20 ops on the GEMM's result (bf16), another [TR, FR] input of a random float type, a [FR] f32 input that
    broadcasts and a device scalar: arithmetic, casts, comparisons and selects, math functions, fma, negations, an iota mask; one to three outputs."""
    rng = random.Random(seed)
    kind = rng.choice(["f32", "bf16", "f16"])

    def body(x, w, x1, bias, n):
        y = gemm(x, w)
        vals = [y, y.to(f32), x1, bias]
        for _ in range(rng.randint(4, 20)):
            a = rng.choice(vals)
            b = rng.choice(vals)
            if a.dtype != b.dtype:
                b = b.to(a.dtype)
            dt = a.dtype
            op = rng.choice(["add", "sub", "mul", "scal", "neg", "cast", "where", "math", "fma", "mask"])
            if op == "add":
                v = a + b
            elif op == "sub":
                v = a - b
            elif op == "mul":
                v = a * b
            elif op == "scal":
                v = a * rng.choice([0.1, 3.0, -0.5]) + rng.choice([1.0, -2.0, 0.3])
            elif op == "neg":
                v = -a
            elif op == "cast":
                v = a.to(rng.choice([f32, bf16, f16]))
            elif op == "where":
                v = tilefish.where(a > b, a, b * 2.0)
            elif op == "math":
                v = rng.choice([tilefish.exp, tilefish.silu, tilefish.tanh, tilefish.sin, tilefish.rsqrt])(a.to(f32)).to(dt)
            elif op == "fma":
                v = tilefish.fma(a.to(f32), b.to(f32), rng.choice(vals).to(f32))
            else:
                v = tilefish.where(tilefish.iota(a, -1) < tilefish.load(n), a, b)  # (the columns below a device-scalar count)
            vals.append(v)
        made = vals[4:]
        outs = [made[-1]] + rng.sample(made[:-1], min(len(made) - 1, rng.randint(0, 2)))
        return tuple(o if o.rank == 2 else o + y.to(o.dtype) for o in outs)

    params = [("x", Tensor[TR, eg.H, bf16]), ("w", Tensor[FR, eg.H, bf16]), ("x1", Tensor[TR, FR, {"f32": f32, "bf16": bf16, "f16": f16}[kind]]),
              ("bias", Tensor[FR, f32]), ("n", DevScalar[i32])]  # fmt: skip
    return body, params


def test_random_regions_after_a_gemm_fused_write_the_bytes_of_their_reference(externs):
    """Half of them in the GEMM whose epilogue runs on two sub-tiles of the columns (each hook called twice per tile)."""
    sub = eg.examples_subtiled(CAP)
    fused_any = 0
    for seed in range(40):
        gemm = externs[0] if seed % 2 else sub
        out = {}
        for mode, fusion in eg.MODES.items():
            body, params = random_body(seed, gemm)
            out[mode] = eg.graph_of(lambda *a, b=body: b(*a), params, fusion, f"rnd_epi{seed}_{mode}")
        fused_any += len(_fusion.fused(out["fused"]).hooked)
        for combo in tilefish._plan.bucket_combinations(tilefish.plan(out["fused"]).dims):
            args = args_of(out["fused"], combo, seed)
            assert_same(run(out["fused"], args), run(out["unfused"], args), f"random graph {seed} at {combo}")
            assert_same(run(out["acc"], args), run(out["acc_unfused"], args), f"random graph {seed} at {combo}, accumulator")
        if seed % 8 == 0:
            tilefish.verify_fusion(out["fused"])
            tilefish.verify_fusion(out["acc"])
    assert fused_any >= 30  # (most random regions fit a hook)


# -- the Gluon GEMM -----------------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(CAP not in (100, 103), reason="the example Gluon GEMM is a tcgen05 kernel (sm_100, sm_103)")
def test_a_gluon_externs_hooks_take_elementwise_regions_in_both_modes():
    import epilogue_gemm as ex

    gg = ex.gluon_gemm(name="epi_gluon_gemm")
    body, params = eg.bodies(gg, None, None)["periodic"]
    out = {mode: eg.graph_of(body, params, fusion, f"gl_periodic_{mode}") for mode, fusion in eg.MODES.items()}
    assert _fusion.fused(out["fused"]).hooked[0].op.language == "gluon"
    for combo in tilefish._plan.bucket_combinations(tilefish.plan(out["fused"]).dims):
        args = args_of(out["fused"], combo, 4)
        assert_same(run(out["fused"], args), run(out["unfused"], args), f"gluon exact at {combo}")
        assert_same(run(out["acc"], args), run(out["acc_unfused"], args), f"gluon accumulator at {combo}")
    tilefish.verify_fusion(out["acc"])
    body, params = eg.bodies(gg, None, None)["add_norm"]
    f = _fusion.fused(eg.graph_of(body, params, tilefish.fusion(), "gl_add_norm"))
    assert not f.hooked and f.refused[0]["reason"] == "epilogue-row"  # (a Gluon tile has its kernel's layout: no row regions in E1)


# -- E1 and E2 in one graph ---------------------------------------------------------------------------------------------------------------


def test_a_cublaslt_epilogue_and_a_hooked_extern_on_its_accumulator_in_one_graph(externs, tmp_path):
    """docs/rfcs/0003-epilogue-fusion.md, "E1 and E2 together": a graph declared epilogues="library" with an "accumulator" scope around an extern's
    arithmetic is one cublaslt.matmul (the GEMM and its bias) and one fused pair on the extern's accumulator, and writes the bytes of its unfused
    composition (both rewrites, no fusion) at every bucket, through jit, export and replay."""
    up = eg.cublaslt_up("gmix_up")
    fused = eg.mixed(externs[0], up, "gmix")
    unfused = eg.mixed(externs[0], up, "gmix_unfused", grouping="none")
    plan = tilefish.plan(fused)
    kinds = [(s["entry"].split(".")[0].split("_epi_")[0], s["kind"]) for s in plan.steps(T=64)]
    assert kinds == [("gmix_up__bias", "library"), (externs[0].name, "kernel")]
    assert [h.contract for h in _fusion.fused(fused).hooked] == ["accumulator"]

    types = {p.name: p.type for p in fused.signature.tensor_inputs}

    def args_at(t, seed):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(t, eg.H.value, generator=gen).to(torch.bfloat16).to(CUDA)
        w1 = (torch.randn(eg.F.value, eg.H.value, generator=gen) / 16).to(torch.bfloat16).to(CUDA)
        w2 = (torch.randn(eg.F.value, eg.F.value, generator=gen) / 16).to(torch.bfloat16).to(CUDA)
        b1, b2 = (_fusion._sample(torch, types[name], (eg.F.value,), seed * 10 + k, CUDA) for k, name in ((1, "b1"), (2, "b2")))
        return [x, w1, b1, w2, b2]

    for t in eg.T.buckets:
        for seed in (0, 1):
            args = args_at(t, seed)
            assert_same(run(fused, args), run(unfused, args), f"mixed at T={t}, seed {seed}")
    p = runtime.export(fused, tmp_path)
    rep = json.loads((tmp_path / f"{fused.name}.fusion.json").read_text())
    assert [w["into"] for w in rep["library_epilogues"] if w["applied"]] == ["gmix_up__bias"]
    assert [r["contract"] for r in rep["regions"] if r.get("route") == "hook"] == ["accumulator"]
    args = args_at(200, 7)
    replayed = tilefish.run(fused, *[a.clone() for a in args], mode="replay", plan=tmp_path, pin=p.digest)
    assert_same([replayed], run(fused, args), "replay")


# -- export, replay, capture, the binary, the reference ------------------------------------------------------------------------------------


def test_export_verifies_every_pair_and_the_plan_replays_and_captures(graphs, tmp_path):
    g = graphs["add_norm"]["fused"]
    p = runtime.export(g, tmp_path, trace=True)
    rep = json.loads((tmp_path / f"{g.name}.fusion.json").read_text())
    (r,) = rep["regions"]
    assert r["route"] == "hook" and r["verified"]["runs"] >= 2 and r["contract"] == "exact"
    man = json.loads((tmp_path / "manifest.json").read_text())
    (k,) = [e for e in man["kernels"] if e["op"] == r["op"]]  # a generated triton entry: nothing new for a host
    assert k["source_sha256"] and k["kind"] == "triton" and not any(e["op"] == r["region"] for e in man["kernels"])
    args = args_of(g, {"T": 200}, 5)
    jit = run(g, args)
    rep_out = tilefish.run(g, *[a.clone() for a in args], mode="replay", plan=tmp_path, pin=p.digest)
    assert_same(list(rep_out), jit, "replay")
    outs = [torch.empty_like(t) for t in jit]
    cap = tilefish.capture(g, *[a.clone() for a in args], outputs=outs, mode="replay", plan=tmp_path, pin=p.digest)
    cap.replay()
    torch.cuda.synchronize()
    assert_same(outs, jit, "capture")
    # the trace attributes the hook's lines to the graph body's ops
    gdoc = json.loads((tmp_path / "trace" / "graphs" / f"{g.name}.json").read_text())
    assert gdoc["regions"][r["op"]]["hook"]["contract"] == "exact"


def test_the_fused_op_has_the_externs_mmas(externs):
    """The hook does not change the GEMM's main loop: the fused binary's MMA instructions are the extern's, in order (a necessary condition of
    contract (i), checked here; the byte comparison is the guard)."""
    from tilefish import _sass
    from tilefish._build import build

    g = eg.build(externs, prefix="mma")["bias_gelu"]["fused"]
    (h,) = _fusion.fused(g).hooked

    def mmas(op):
        t = runtime._target(op)
        spec = t.compile_specs()[0]
        b = build(spec, arch=CAP, fn=t.jit_function(spec))
        try:
            d = _sass.disassemble(b.cubin, ptx=b.ptx, options=b.options)
        except tilefish.TilefishBuildError:
            pytest.skip("no nvdisasm that reads the binary")
        return [".".join((i.opcode, *i.modifiers)) for i in d.listing.instructions() if i.opcode in ("HMMA", "UTCHMMA", "HGMMA", "UTCQMMA", "UTCIMMA")]

    a, b = mmas(h.extern), mmas(h.op)
    assert a and a == b


def test_a_hook_applied_to_another_value_than_the_stored_tile_is_refused_at_export(tmp_path):
    import epilogue_kernels as ek

    R = Dim("RE", buckets=[96])
    C = Dim("CE", 192)

    def body(op):
        return lambda x, b: (op(x).to(f32) + b.to(f32)).to(bf16)

    params = [("x", Tensor[R, C, bf16]), ("b", Tensor[C, bf16])]
    good = eg.graph_of(body(ek.scale3), params, tilefish.fusion(), "honest")
    bad = eg.graph_of(body(ek.lying), params, tilefish.fusion(), "lying")
    runtime.export(good, tmp_path / "good")
    with pytest.raises(_fusion.TilefishFusionError, match="not the bytes of U\\(P\\), the extern"):
        runtime.export(bad, tmp_path / "bad")
    assert not (tmp_path / "bad" / "plan.json").exists()
    runtime.export(bad, tmp_path / "anyway", verify=False)  # the opt-out writes it


def test_reference_mode_is_close_to_the_fused_plan(graphs):
    for pattern in ("bias_gelu", "swiglu", "add_norm_f32", "conv_bias_silu"):
        for mode in ("fused", "acc"):
            g = graphs[pattern][mode]
            combo = {d: v[-1] for d, v in tilefish.plan(g).dims.items()}
            args = args_of(g, combo, 2)
            args = [a.nan_to_num(0.0, 1.0, -1.0).clamp(-4, 4) if a.is_floating_point() else a for a in args]
            got = run(g, args)
            ref = tilefish.run(g, *[a.clone() for a in args], mode="reference")
            ref = list(ref) if isinstance(ref, tuple) else [ref]
            for a, b in zip(got, ref):
                assert a.shape == b.shape and a.dtype == b.dtype
                assert torch.allclose(a.float(), b.float(), atol=0.08, rtol=0.05), (pattern, mode, (a.float() - b.float()).abs().max().item())
