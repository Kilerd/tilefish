"""Plan fusion on a GPU (``docs/rfcs/0001-plan-fusion.md``): a fused plan writes the bytes of its unfused composition (every op a kernel of its
own, every intermediate stored in its dtype), for every input.

* the inline casts of a region (note 20 of :mod:`tilefish.lower`) are ``.to()``'s bits on every f32, and keep no input NaN's payload through
  ``x.to(f32).to(bf16)``, where ``.to()`` in one kernel does (the fold the RFC's experiment E5 measured);
* the use cases of fusion's first version, fused and unfused, at every bucket, on inputs with special values; and the reference where it is exact;
* chains of ops on every bf16 and f16 bit pattern, binary ops on a large grid of bit patterns, and random graphs (a property test);
* ``intermediates="opmath"`` in IEEE math is torch's eager composition (every non-NaN value; a NaN is the canonical one of each);
* ``export`` verifies the regions (and refuses a region that is not its ops one by one), writes the report, and the exported plan replays and
  captures as it runs.

Nothing here is specific to sm_103: the same tests run on an sm_120 GPU.
"""

import inspect
import json
import random

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import fusion_graphs as fg  # noqa: E402
from gpu_util import differing, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import DevScalar, Dim, Tensor, bf16, f16, f32, i32, runtime  # noqa: E402
from tilefish import _fusion  # noqa: E402
from tilefish._launch import eval_dim  # noqa: E402

CUDA = "cuda"
TYPES = {"f32": f32, "bf16": bf16, "f16": f16, "i32": i32, "u8": tilefish.u8}
TORCH = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16, "i32": torch.int32, "u8": torch.uint8}


def sample(t, sizes, seed):
    if isinstance(t, tilefish.DType):
        t = tilefish.TensorType((), t)
    return _fusion._sample(torch, t, sizes, seed, CUDA)


def args_of(graph, combo, seed):
    """Seeded arguments of a graph at a bucket combination: every corner of each dtype (NaNs with payloads, infinities, subnormals, zeros)."""
    out = []
    dims = {d: combo[d.name] for p in graph.signature.tensor_inputs for s in p.type.shape for d in s.free_dims() if d.name in combo}
    for k, p in enumerate(graph.signature.tensor_inputs):
        if p.is_devscalar:
            out.append(torch.tensor([7 + seed * 101 + k], dtype=TORCH[p.type.dtype.name], device=CUDA))
            continue
        sizes = tuple(int(eval_dim(s, dims)) for s in p.type.shape)
        out.append(sample(p.type, sizes, seed * 1000 + k))
    return out


def combos(graph):
    return tilefish._plan.bucket_combinations(tilefish.plan(graph).dims)


def run_both(fused, unfused, args):
    """The fused plan and the unfused one on the same arguments (an InOut argument cloned for each), and the arguments each left."""
    a = [x.clone() for x in args]
    b = [x.clone() for x in args]
    got = tilefish.run(fused, *a)
    want = tilefish.run(unfused, *b)
    got = list(got) if isinstance(got, tuple) else [got]
    want = list(want) if isinstance(want, tuple) else [want]
    return got, want, a, b


def assert_same(got, want, what):
    for k, (x, y) in enumerate(zip(got, want)):
        assert same_bytes(x, y), f"{what}: output {k}: {differing(x, y)} of {x.numel()} elements differ"


# -- the inline casts (note 20) --------------------------------------------------------------------------------------------------------

NC = Dim("NC", buckets=[1 << 28])
H16 = Dim("H16", buckets=[1 << 16])


@tilefish.graph
def g_casts(x: Tensor[NC, f32]):
    return x.to(bf16), x.to(f16)


@tilefish.kernel
def k_casts(x: Tensor[NC, f32]) -> tuple[Tensor[NC, bf16], Tensor[NC, f16]]:
    return x.to(bf16), x.to(f16)


def test_the_inline_casts_of_a_region_are_to_on_every_f32():
    for start in range(-(2**31), 2**31, 1 << 28):
        x = torch.arange(start, start + (1 << 28), dtype=torch.int64, device=CUDA).to(torch.int32).view(torch.float32)
        rb, rh = tilefish.run(g_casts, x)
        kb, kh = tilefish.run(k_casts, x)
        assert same_bytes(rb, kb) and same_bytes(rh, kh), start
        assert same_bytes(rb, x.to(torch.bfloat16)), start  # torch's CUDA cast too
        del x, rb, rh, kb, kh


@tilefish.graph
def g_half_casts(x: Tensor[H16, bf16], y: Tensor[H16, f16]):
    return x.to(f16), y.to(bf16)


@tilefish.kernel
def k_half_casts(x: Tensor[H16, bf16], y: Tensor[H16, f16]) -> tuple[Tensor[H16, f16], Tensor[H16, bf16]]:
    return x.to(f16), y.to(bf16)


def every16(dtype):
    return torch.arange(-(2**15), 2**15, dtype=torch.int32, device=CUDA).to(torch.int16).view(dtype)


def test_the_inline_casts_between_half_types_are_to_on_every_bit_pattern():
    x, y = every16(torch.bfloat16), every16(torch.float16)
    got = tilefish.run(g_half_casts, x, y)
    want = tilefish.run(k_half_casts, x, y)
    assert_same(got, want, "bf16 <-> f16")


@tilefish.graph
def g_roundtrip(x: Tensor[H16, bf16], y: Tensor[H16, f16]):
    a, b = x.to(f32), x.to(f32)
    sel = tilefish.where(a == a, b, a)
    return x.to(f32).to(bf16), y.to(f32).to(f16), sel.to(bf16)


@tilefish.graph(fusion=fg.UNFUSED)
def g_roundtrip_unfused(x: Tensor[H16, bf16], y: Tensor[H16, f16]):
    a, b = x.to(f32), x.to(f32)
    sel = tilefish.where(a == a, b, a)
    return x.to(f32).to(bf16), y.to(f32).to(f16), sel.to(bf16)


def test_a_round_trip_in_a_region_keeps_no_input_nan_payload():
    x, y = every16(torch.bfloat16), every16(torch.float16)
    got, want, _, _ = run_both(g_roundtrip, g_roundtrip_unfused, [x, y])
    assert_same(got, want, "round trips")
    nb, nh = torch.isnan(x.float()), torch.isnan(y.float())
    assert int(nb.sum()) == 254 and int(nh.sum()) == 2046
    assert bool((got[0].view(torch.int16)[nb] == 0x7FFF).all()) and bool((got[1].view(torch.int16)[nh] == 0x7FFF).all())
    assert bool((got[0].view(torch.int16)[~nb] == x.view(torch.int16)[~nb]).all())  # every other value comes back as it was


# -- the use cases ----------------------------------------------------------------------------------------------------------------------

EXACT_REFERENCE = {"masked", "dequant", "around_call", "bias_act_ieee", "residual", "modes"}


@pytest.mark.parametrize("name", sorted(fg.CASES))
def test_a_use_case_fused_is_its_unfused_composition_at_every_bucket(name):
    fused, unfused = fg.CASES[name]
    for combo in combos(fused):
        for seed in (0, 1, 2):
            args = args_of(fused, combo, seed)
            got, want, left_f, left_u = run_both(fused, unfused, args)
            assert_same(got, want, f"{name} at {combo}, seed {seed}")
            assert_same(left_f, left_u, f"{name} at {combo}, seed {seed}: the arguments")  # an InOut tensor gets the same bytes
            if name in EXACT_REFERENCE:  # every op of these has an exact reference: the binary is its bytes
                ref = tilefish.run(fused, *[x.clone() for x in args], mode="reference")
                ref = list(ref) if isinstance(ref, tuple) else [ref]
                assert_same(got, ref, f"{name} at {combo}, seed {seed}: against the reference")


def test_a_region_is_one_launch_where_the_unfused_composition_is_many():
    assert len(tilefish.plan(fg.bias_act).steps(R=8)) == 1 and len(tilefish.plan(fg.bias_act_unfused).steps(R=8)) == 5


# -- every bit pattern, a large grid ------------------------------------------------------------------------------------------------------


def _chains(dt):
    T = TYPES[dt]
    return {
        "exp": lambda x: tilefish.exp(x.to(f32)).to(T),
        "affine": lambda x: (x.to(f32) * 0.5 + 1.0).to(T),
        "half_affine": lambda x: x * 0.5 + x,
        "tanh_silu": lambda x: tilefish.silu(tilefish.tanh(x.to(f32) * 2.0)).to(T),
        "sqrt_rsqrt": lambda x: (tilefish.sqrt(x.to(f32)) * tilefish.rsqrt(x.to(f32))).to(T),
        "neg_where": lambda x: tilefish.where(x > 0.0, -x, x * 2.0),
        "sin_cos": lambda x: (tilefish.sin(x.to(f32)) + tilefish.cos(x.to(f32))).to(T),
        "ieee": lambda x: (tilefish.exp(x.to(f32), math="ieee") / (x.to(f32) + 3.0)).to(T),
    }


def _graph_of(body, params, fusion=None, name="g"):
    def fn(**kw):
        return body(*[kw[p] for p, _ in params])

    fn.__signature__ = inspect.Signature([inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=t) for p, t in params])
    fn.__name__ = name
    return tilefish.graph(fn, name=name, fusion=fusion)


@pytest.mark.parametrize("dt", ["bf16", "f16"])
def test_chains_of_ops_on_every_half_bit_pattern(dt):
    x = every16(TORCH[dt])
    for cname, body in _chains(dt).items():
        params = [("x", Tensor[H16, TYPES[dt]])]
        fused = _graph_of(body, params, name=f"chain_{dt}_{cname}")
        unfused = _graph_of(body, params, fg.UNFUSED, name=f"chain_{dt}_{cname}_u")
        got, want, _, _ = run_both(fused, unfused, [x])
        assert len(_fusion.fused(fused).regions) == 1, cname
        assert_same(got, want, f"{dt} {cname}")


G = Dim("G", buckets=[1 << 22])


@pytest.mark.parametrize("dt", ["f32", "bf16"])
def test_binary_ops_on_a_large_grid_of_bit_patterns(dt):
    T = TYPES[dt]
    if dt == "f32":
        body = lambda a, b, c: ((a * b + c) / (a - b), tilefish.fma(a, b, c) - a * c)  # noqa: E731
    else:
        body = lambda a, b, c: ((a * b + c) * 0.25 - a, tilefish.where(a < b, a + c, b * c).to(f32))  # noqa: E731
    params = [("a", Tensor[G, T]), ("b", Tensor[G, T]), ("c", Tensor[G, T])]
    fused = _graph_of(body, params, name=f"grid_{dt}")
    unfused = _graph_of(body, params, fg.UNFUSED, name=f"grid_{dt}_u")
    for seed in (0, 1, 2):
        args = [sample(T, (1 << 22,), seed * 10 + k) for k in range(3)]
        got, want, _, _ = run_both(fused, unfused, args)
        assert_same(got, want, f"{dt} grid, seed {seed}")


# -- random graphs ----------------------------------------------------------------------------------------------------------------------

RB = Dim("RB", buckets=[16, 40])
C = Dim("C", 96)


def random_body(seed):
    """A random arithmetic DAG of 4 to 24 ops over three [RB, C] inputs of random float types, a [C] f32 input that broadcasts and a device
    scalar: arithmetic in each dtype, casts, comparisons and selects, math functions, fma, negations, an iota mask; one to three outputs."""
    rng = random.Random(seed)
    kinds = [rng.choice(["f32", "bf16", "f16"]) for _ in range(3)]

    def body(x0, x1, x2, bias, n):
        vals = [x0, x1, x2, bias]
        for _ in range(rng.randint(4, 24)):
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
                fn = rng.choice([tilefish.exp, tilefish.silu, tilefish.tanh, tilefish.sin, tilefish.rsqrt])
                v = fn(a.to(f32)).to(dt)
            elif op == "fma":
                c = rng.choice(vals)
                v = tilefish.fma(a.to(f32), b.to(f32), c.to(f32))
            else:
                v = tilefish.where(tilefish.iota(a, 0) < tilefish.load(n), a, b)
            vals.append(v)
        made = vals[4:]
        outs = [made[-1]] + rng.sample(made[:-1], min(len(made) - 1, rng.randint(0, 2)))
        return tuple(o if o.rank == 2 else o + x0.to(o.dtype) for o in outs)

    params = [("x0", Tensor[RB, C, TYPES[kinds[0]]]), ("x1", Tensor[RB, C, TYPES[kinds[1]]]), ("x2", Tensor[RB, C, TYPES[kinds[2]]]),
              ("bias", Tensor[C, f32]), ("n", DevScalar[i32])]
    return body, params, random.Random(seed)


def test_random_graphs_fused_are_their_unfused_composition():
    seeds = list(range(120))
    for seed in seeds:
        body, params, _ = random_body(seed)
        # the body draws its ops from a generator: each graph is traced from a fresh one with the same seed
        fused = _graph_of(lambda *a, s=seed: random_body(s)[0](*a), params, name=f"rnd{seed}")
        unfused = _graph_of(lambda *a, s=seed: random_body(s)[0](*a), params, fg.UNFUSED, name=f"rnd{seed}_u")
        for combo in combos(fused):
            args = args_of(fused, combo, seed)
            got, want, _, _ = run_both(fused, unfused, args)
            assert_same(got, want, f"random graph {seed} at {combo}")
        tilefish.verify_fusion(fused)


@pytest.mark.parametrize("dt", ["f32", "bf16", "f16"])
def test_a_negation_that_ptxas_could_fold_into_a_select_keeps_the_canonical_nan(dt):
    """What the random graphs found (note 20): ``neg.f32`` alone gives the canonical NaN, but folded into a select as a negated operand it flips
    a NaN's sign; a region's negation gives the canonical NaN explicitly, as its unfused composition does."""
    T = TYPES[dt]
    body = lambda a, b: (tilefish.where(a > b, a, -(b * 2.0)), tilefish.where(a < b, -a, b))  # noqa: E731
    params = [("a", Tensor[G, T]), ("b", Tensor[G, T])]
    fused = _graph_of(body, params, name=f"negsel_{dt}")
    unfused = _graph_of(body, params, fg.UNFUSED, name=f"negsel_{dt}_u")
    args = [sample(T, (1 << 22,), 50 + k) for k in range(2)]
    got, want, _, _ = run_both(fused, unfused, args)
    assert_same(got, want, f"{dt} negation in a select")


# -- opmath in IEEE math is torch's eager composition ----------------------------------------------------------------------------------------


@tilefish.graph(fusion=tilefish.fusion(intermediates="opmath", math="ieee"))
def eager_like(x: Tensor[H16, bf16], y: Tensor[H16, f16]):
    return x * 0.1 + 2.0, (y * 3.0 - 1.5) * y, tilefish.silu(x.to(f32)).to(bf16)


def test_opmath_in_ieee_math_is_torchs_eager_composition():
    x, y = every16(torch.bfloat16), every16(torch.float16)
    got = tilefish.run(eager_like, x, y)
    want = [x * 0.1 + 2.0, (y * 3.0 - 1.5) * y, torch.nn.functional.silu(x.float()).to(torch.bfloat16)]
    for k, (a, b) in enumerate(zip(got, want)):
        nan = torch.isnan(a.float())
        assert bool((nan == torch.isnan(b.float())).all()), k
        assert same_bytes(a[~nan], b[~nan]), f"output {k}: {differing(a[~nan], b[~nan])} differ"


def test_two_half_tensors_add_sub_and_multiply_as_torchs_eager_ops_on_every_pair():
    """What lets ``opmath`` leave a two-tensor 16-bit op as it is: the native op's bytes are torch's (which computes in f32 and rounds once) on
    every pair of bit patterns, but for a NaN's payload (each its canonical one)."""
    for dt in ("bf16", "f16"):
        T = TYPES[dt]
        GP = Dim("GP", buckets=[1 << 28])
        body = lambda a, b: (a + b, a - b, a * b)  # noqa: E731
        g = _graph_of(body, [("a", Tensor[GP, T]), ("b", Tensor[GP, T])], name=f"pairs_{dt}")
        low = every16(TORCH[dt])
        for hi in range(-(2**15), 2**15, 1 << 12):
            a = torch.arange(hi, hi + (1 << 12), dtype=torch.int32, device=CUDA).to(torch.int16).view(TORCH[dt]).repeat_interleave(1 << 16)
            b = low.repeat(1 << 12)
            got = tilefish.run(g, a, b)
            for res, want in zip(got, (a + b, a - b, a * b)):
                nan = torch.isnan(res.float())
                assert bool((nan == torch.isnan(want.float())).all()), (dt, hi)
                assert same_bytes(res[~nan], want[~nan]), (dt, hi)
            del a, b, got


# -- export: verification, the report, replay and capture ------------------------------------------------------------------------------------


def test_export_verifies_the_regions_writes_the_report_and_the_plan_replays_and_captures(tmp_path):
    p = runtime.export(fg.residual, tmp_path)
    rep = json.loads((tmp_path / "residual.fusion.json").read_text())
    assert rep["schema"] == "tilefish.fusion-report.v1"
    assert all(r.get("verified", {}).get("runs", 0) >= 2 for r in rep["regions"] if len(r["nodes"]) > 1)
    args = args_of(fg.residual, {"R": 24}, 5)
    a, b = [x.clone() for x in args], [x.clone() for x in args]
    jit = tilefish.run(fg.residual, *a)
    rep_out = tilefish.run(fg.residual, *b, mode="replay", plan=tmp_path, pin=p.digest)
    assert same_bytes(jit, rep_out) and same_bytes(a[1], b[1])
    c = [x.clone() for x in args]
    out = torch.empty_like(jit)
    cap = tilefish.capture(fg.residual, *c, outputs=out, mode="replay", plan=tmp_path, pin=p.digest)
    cap.replay()
    torch.cuda.synchronize()
    assert same_bytes(out, jit) and same_bytes(c[1], a[1])


def test_a_region_that_is_not_its_ops_one_by_one_is_refused_at_export(tmp_path, monkeypatch):
    real = _fusion._declare

    def contracting(name, *a, **kw):  # a broken toolchain: a region whose kernel contracts a * b + c
        op = real(name, *a, **kw)
        op.fp_fusion, op.context_free = True, False  # contraction allowed, the arithmetic left to Triton
        return op

    monkeypatch.setattr(_fusion, "_declare", contracting)

    @tilefish.graph
    def contracted(a: Tensor[fg.N, f32], b: Tensor[fg.N, f32], c: Tensor[fg.N, f32]):
        return a * b + c

    with pytest.raises(_fusion.TilefishFusionError, match="are not those of its ops one by one"):
        runtime.export(contracted, tmp_path)
    assert not (tmp_path / "plan.json").exists()  # nothing is written
    runtime.export(contracted, tmp_path / "anyway", verify=False)  # the opt-out writes it
    assert (tmp_path / "anyway" / "plan.json").exists()


def test_a_barrier_keeps_a_value_a_tap_reads():
    @tilefish.graph
    def tapped(x: Tensor[fg.N, f32]):
        return tilefish.barrier(x * 3.0) + 1.0

    x = sample(f32, (4096,), 3)
    p = tilefish.plan(tapped)
    name = next(b["name"] for b in p.buffers if b["role"] == "intermediate")
    t = torch.empty(4096, device=CUDA)
    y = tilefish.run(tapped, x, taps={name: t})
    assert same_bytes(t, x * 3.0) and same_bytes(y, t + 1.0)
