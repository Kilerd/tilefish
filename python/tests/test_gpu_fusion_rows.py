"""Row regions on a GPU (fusion version 2, ``docs/rfcs/0001-plan-fusion.md`` phase 2): an ordered sum or mean, ``max``, ``argmax`` and a gather
fused with the ops around them write the bytes of their unfused composition (every op a kernel of its own, every intermediate stored), for
every input.

* the row patterns of ``fusion_graphs.ROW_CASES`` (a residual add and an rmsnorm with the residual sum a second output, a layer norm with an
  elementwise tail, softmax, row-wise fp8 quantisation, a norm over two strided views, a gather and an argmax) fused against unfused at every
  bucket: on every bf16 bit pattern of their bf16 inputs, on seeded inputs with special values, and on rows of special values only;
* a region gives the same bytes at every warp count (its reductions are ordered or order-free, its other ops per element);
* random graphs of row ops (ordered sums and means of several orders, fma sums, ``max``, ``argmax``, broadcasts of a row's value, masks from a
  device scalar) fused against unfused, and through ``verify_fusion``;
* the inline fp8 casts of revision 2 are ``.to()``'s bits on every f32 and every bf16 and f16 pattern;
* ``export`` verifies a row region, writes its report, and the plan replays and captures as it runs.

Nothing here is specific to sm_103: the same tests run on an sm_120 GPU.
"""

import random

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import fusion_graphs as fg  # noqa: E402
from gpu_util import same_bytes  # noqa: E402
from test_gpu_fusion import CUDA, _graph_of, args_of, assert_same, combos, every16, run_both  # noqa: E402

import tilefish  # noqa: E402
from tilefish import DevScalar, Dim, Tensor, bf16, e4m3, e5m2, f16, f32, i32, runtime  # noqa: E402
from tilefish import _fusion  # noqa: E402
from tilefish._launch import eval_dim  # noqa: E402


def sizes_of(graph, combo):
    dims = {d: combo[d.name] for p in graph.signature.tensor_inputs for s in p.type.shape for d in s.free_dims() if d.name in combo}
    return [None if p.is_devscalar else tuple(int(eval_dim(s, dims)) for s in p.type.shape) for p in graph.signature.tensor_inputs]


def special_rows(shape, dtype, kind):
    """A tensor of ``shape`` whose rows are all special values of one kind: signed zeros mixed, NaNs (with payloads), infinities with numbers,
    subnormals, or a row of one NaN among numbers."""
    n = shape[-1]
    t = torch.randn(shape, device=CUDA).to(dtype)
    width = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}[dtype]
    if kind == "zeros":
        t = torch.where(torch.rand(shape, device=CUDA) < 0.5, torch.zeros_like(t), -torch.zeros_like(t))
    elif kind == "nan":
        bits = {torch.float32: [0x7FC00000, 0x7F800123, -0x003EDCBA], torch.bfloat16: [0x7FC0, 0x7F81, -0x003F], torch.float16: [0x7E00, 0x7C01, -0x01FF]}[dtype]
        pick = torch.randint(0, 3, shape, device=CUDA)
        t = torch.tensor(bits, dtype=torch.int64, device=CUDA)[pick].to(width).view(dtype)
    elif kind == "inf":
        t[..., ::3] = float("inf")
        t[..., 1::7] = float("-inf")
    elif kind == "subnormal":
        tiny = {torch.float32: 2.0**-140, torch.bfloat16: 2.0**-130, torch.float16: 2.0**-20}[dtype]
        t = (torch.randn(shape, device=CUDA) * tiny).to(dtype)
    elif kind == "one_nan":
        t[..., n // 3] = float("nan")
    return t


def with_special(args, graph, kind):
    out = []
    for a, p in zip(args, graph.signature.tensor_inputs):
        if p.is_devscalar or a.dtype not in (torch.float32, torch.bfloat16, torch.float16) or p.type.rank < 2:
            out.append(a)
        else:
            out.append(special_rows(tuple(a.shape), a.dtype, kind))
    return out


@pytest.mark.parametrize("name", fg.ROW_CASES)
def test_a_row_pattern_fused_is_its_unfused_composition(name):
    fused, unfused = fg.CASES[name]
    assert len(_fusion.fused(fused).regions) == 1
    checked = 0
    for combo in combos(fused):
        for seed in (0, 1, 2, 3):
            args = args_of(fused, combo, seed)
            got, want, _, _ = run_both(fused, unfused, args)
            assert_same(got, want, f"{name} at {combo}, seed {seed}")
            checked += 1
        for kind in ("zeros", "nan", "inf", "subnormal", "one_nan"):
            args = with_special(args_of(fused, combo, 7), fused, kind)
            got, want, _, _ = run_both(fused, unfused, args)
            assert_same(got, want, f"{name} at {combo}, rows of {kind}")
            checked += 1
        # every bf16 bit pattern of each bf16 tensor of the rows, in as many runs as it takes
        shapes = sizes_of(fused, combo)
        bf = [k for k, p in enumerate(fused.signature.tensor_inputs) if not p.is_devscalar and p.type.dtype == bf16 and p.type.rank == 2]
        if bf:
            every = every16(torch.bfloat16)
            for start in range(0, 1 << 16, min(shapes[bf[0]][0] * shapes[bf[0]][1], 1 << 16)):
                args = args_of(fused, combo, start)
                for k in bf:
                    n = shapes[k][0] * shapes[k][1]
                    idx = (torch.arange(n, device=CUDA) * (1 + 2 * k) + start) % (1 << 16)
                    args[k] = every[idx].reshape(shapes[k])
                got, want, _, _ = run_both(fused, unfused, args)
                assert_same(got, want, f"{name} at {combo}, every bf16 pattern from {start}")
                checked += 1
    assert checked > 10


@pytest.mark.parametrize("name", fg.ROW_CASES)
def test_a_row_region_gives_the_same_bytes_at_every_warp_count(name):
    fused, _ = fg.CASES[name]
    (r,) = _fusion.fused(fused).regions
    combo = combos(fused)[-1]
    args = args_of(fused, combo, 11)
    region_args = []
    by_name = {p.name: a for p, a in zip(fused.signature.tensor_inputs, args)}
    for v in r.inputs:
        if v.is_param:
            region_args.append(by_name[v.name])
        else:  # a view of a parameter (split_norm's halves): the same view of the same tensor
            src = next(p for p in fused.signature.tensor_inputs if p.type.rank == 2)
            base = by_name[src.name]
            half = base.shape[1] // 2
            region_args.append(base[:, :half] if len(region_args) == 0 else base[:, half:])
    want = tilefish.run(r.call, *region_args)
    want = list(want) if isinstance(want, tuple) else [want]
    for w in (1, 2, 4, 8, 16):
        low = tilefish.lower(r.kernel, num_warps=w, module=f"tf_rowwarps_{name}_{w}")
        got = low.run(*region_args)
        got = list(got) if isinstance(got, tuple) else [got]
        assert_same(got, want, f"{name} at num_warps={w}")


# -- random graphs of row ops -----------------------------------------------------------------------------------------------------------------

RR = Dim("RR", buckets=[16, 40])
CR = Dim("CR", buckets=[96, 1000, 2304])
ORDERS = [tilefish.tree(32), tilefish.tree(128, offsets=(16, 8, 4, 2, 1, 64, 32), vector=8), tilefish.tree(256), tilefish.tree(64, vector=4),
          tilefish.tree(512, vector=2)]


def random_row_body(seed):
    """A random DAG of row ops over two [RR, CR] inputs (f32 and bf16), a [CR] f32 input and a device scalar: elementwise arithmetic, ordered
    sums and means of random orders, fma sums, max, argmax, a row's value broadcast back over it, a mask of the columns below the scalar; one to
    three outputs, of a row or one value per row."""
    rng = random.Random(seed)

    def body(x0, x1, w, n):
        full = [x0, x1.to(f32)]
        rows = []
        for _ in range(rng.randint(3, 14)):
            a = rng.choice(full)
            kind = rng.choice(["ew", "ew", "ew", "col", "sum", "fma", "mean", "max", "argmax", "bcast", "bcast", "mask", "math"])
            if kind == "ew":
                b = rng.choice(full)
                full.append(rng.choice([lambda: a + b, lambda: a - b, lambda: a * b, lambda: a * 0.5 + 1.0])())
            elif kind == "col":
                full.append(a * w + rng.choice([0.0, 1.0, -3.0]))
            elif kind == "math":
                full.append(rng.choice([tilefish.exp, tilefish.silu, tilefish.tanh, tilefish.sin])(a * 0.25))
            elif kind == "sum":
                rows.append(tilefish.sum(a, -1, keepdims=True, order=rng.choice(ORDERS)))
            elif kind == "fma":
                b = rng.choice(full)
                rows.append(tilefish.sum(a * b, -1, keepdims=True, order=rng.choice(ORDERS), fma=True))
            elif kind == "mean":
                rows.append(tilefish.mean(a, -1, keepdims=True, order=rng.choice(ORDERS)))
            elif kind == "max":
                rows.append(tilefish.max(a, -1, keepdims=True))
            elif kind == "argmax":
                rows.append(tilefish.argmax(a, -1, dtype=i32, keepdims=True).to(f32))
            elif kind == "bcast" and rows:
                r = rng.choice(rows)
                full.append(rng.choice([lambda: a - r, lambda: a * r, lambda: tilefish.where(a > r, a, r)])())
            elif kind == "mask":
                full.append(tilefish.where(tilefish.iota(a, -1) < tilefish.load(n), a, rng.choice([0.0, float("-inf")])))
        outs = [full[-1].to(bf16)]
        if rows:
            outs.append(rows[-1])
        if len(full) > 3 and rng.random() < 0.5:
            outs.append(rng.choice(full[2:-1]))
        return tuple(outs)

    params = [("x0", Tensor[RR, CR, f32]), ("x1", Tensor[RR, CR, bf16]), ("w", Tensor[CR, f32]), ("n", DevScalar[i32])]
    return body, params


def test_random_row_graphs_fused_are_their_unfused_composition():
    regions = 0
    for seed in range(80):
        _, params = random_row_body(seed)
        fused = _graph_of(lambda *a, s=seed: random_row_body(s)[0](*a), params, name=f"rows{seed}")
        unfused = _graph_of(lambda *a, s=seed: random_row_body(s)[0](*a), params, fg.UNFUSED, name=f"rows{seed}_u")
        regions += len(_fusion.fused(fused).fused_regions)
        for combo in combos(fused):
            args = args_of(fused, combo, seed)
            args[3] = torch.tensor([[17, 950, 3000][seed % 3]], dtype=torch.int32, device=CUDA)
            got, want, _, _ = run_both(fused, unfused, args)
            assert_same(got, want, f"random row graph {seed} at {combo}")
        if seed % 4 == 0:
            tilefish.verify_fusion(fused)
    assert regions >= 60  # most graphs are one region (a reduction without an order is not drawn: D2)


# -- the inline fp8 casts of revision 2 --------------------------------------------------------------------------------------------------------

NC = Dim("NC8", buckets=[1 << 28])
H16 = Dim("H16f8", buckets=[1 << 16])


@tilefish.graph
def g_fp8(x: Tensor[NC, f32]):
    return x.to(e4m3), x.to(e5m2)


@tilefish.kernel
def k_fp8(x: Tensor[NC, f32]) -> tuple[Tensor[NC, e4m3], Tensor[NC, e5m2]]:
    return x.to(e4m3), x.to(e5m2)


@tilefish.graph
def g_half_fp8(x: Tensor[H16, bf16], y: Tensor[H16, f16]):
    return x.to(e4m3), y.to(e5m2), y.to(e4m3)


@tilefish.kernel
def k_half_fp8(x: Tensor[H16, bf16], y: Tensor[H16, f16]) -> tuple[Tensor[H16, e4m3], Tensor[H16, e5m2], Tensor[H16, e4m3]]:
    return x.to(e4m3), y.to(e5m2), y.to(e4m3)


def test_the_inline_fp8_casts_are_to_on_every_pattern():
    assert "cvt.rn.satfinite.e4m3x2.f32 $0, 0f00000000, $1;" in _fusion.fused(g_fp8).regions[0].kernel.lower().source
    for start in range(-(2**31), 2**31, 1 << 28):
        x = torch.arange(start, start + (1 << 28), dtype=torch.int64, device=CUDA).to(torch.int32).view(torch.float32)
        got = tilefish.run(g_fp8, x)
        want = tilefish.run(k_fp8, x)
        assert all(same_bytes(a, b) for a, b in zip(got, want)), start
        del x, got, want
    x, y = every16(torch.bfloat16), every16(torch.float16)
    assert_same(tilefish.run(g_half_fp8, x, y), tilefish.run(k_half_fp8, x, y), "16-bit to fp8")


D1K = Dim("D1K", 1024)


def _two_means(x, w, b):
    v = x.to(f32)
    mu = tilefish.mean(v, -1, keepdims=True, order=tilefish.tree(256))
    d = v - mu
    var = tilefish.mean(d * d, -1, keepdims=True, order=tilefish.tree(256))
    return (d * tilefish.rsqrt(var + 1e-5) * w + b).to(bf16)


@tilefish.graph
def two_means(x: Tensor[fg.R, D1K, bf16], w: Tensor[D1K, f32], b: Tensor[D1K, f32]):
    return _two_means(x, w, b)


@tilefish.graph(fusion=fg.UNFUSED)
def two_means_unfused(x: Tensor[fg.R, D1K, bf16], w: Tensor[D1K, f32], b: Tensor[D1K, f32]):
    return _two_means(x, w, b)


def test_an_ordered_mean_of_a_static_power_of_two_row_passes_the_contraction_check():
    """A mean of 1024 columns multiplies by the constant f32(1 / 1024); an rcp.rn.f32 of the constant 1024 next to that multiply was expanded by
    ptxas into fused multiply-adds, which the contraction check refused (revision 2 multiplies by the constant instead)."""
    for combo in combos(two_means):
        for seed in (0, 1):
            got, want, _, _ = run_both(two_means, two_means_unfused, args_of(two_means, combo, seed))
            assert_same(got, want, f"two means at {combo}, seed {seed}")


NZ = Dim("NZ", buckets=[4096])


def _neg_zero(x, h):
    a = tilefish.where(x > 0.0, x, -0.0)
    b = tilefish.where(h > 0.0, h, -0.0)
    return a * 1.0, a + 0.0, b * 1.0


@tilefish.graph
def neg_zero(x: Tensor[NZ, f32], h: Tensor[NZ, bf16]):
    return _neg_zero(x, h)


@tilefish.graph(fusion=fg.UNFUSED)
def neg_zero_unfused(x: Tensor[NZ, f32], h: Tensor[NZ, bf16]):
    return _neg_zero(x, h)


def test_a_scalar_negative_zero_keeps_its_sign_in_revision_2():
    """Triton 3.5.1 writes a literal -0.0 as +0.0; revision 2 writes it by its bits: where(x > 0, x, -0.0) * 1.0 is -0.0 where x <= 0, the bytes
    of the reference, fused and unfused alike."""
    x = -torch.rand(4096, device=CUDA)
    h = -torch.rand(4096, device=CUDA).to(torch.bfloat16)
    got, want, _, _ = run_both(neg_zero, neg_zero_unfused, [x, h])
    assert_same(got, want, "-0.0")
    assert bool((got[0].view(torch.int32) == -(2**31)).all()) and bool((got[2].view(torch.int16) == -(2**15)).all())
    assert bool((got[1].view(torch.int32) == 0).all())  # -0.0 + 0.0 is +0.0
    ref = tilefish.run(neg_zero, x, h, mode="reference")
    assert_same(got, list(ref), "-0.0 against the reference")


# -- export -----------------------------------------------------------------------------------------------------------------------------------


def test_export_verifies_a_row_region_and_the_plan_replays_and_captures(tmp_path):
    import json

    p = runtime.export(fg.add_rmsnorm, tmp_path)
    rep = json.loads((tmp_path / "add_rmsnorm.fusion.json").read_text())
    assert rep["fusion"]["version"] == 2 and len(rep["regions"]) == 1
    (r,) = rep["regions"]
    assert r["template"] == "row" and r["verified"]["runs"] >= 2 * len(combos(fg.add_rmsnorm))
    assert any(x.get("order") for x in r["roundings"])
    args = args_of(fg.add_rmsnorm, {"R": 24, "DR": 2304}, 5)
    jit = tilefish.run(fg.add_rmsnorm, *args)
    rep_out = tilefish.run(fg.add_rmsnorm, *args, mode="replay", plan=tmp_path, pin=p.digest)
    assert_same(list(rep_out), list(jit), "replay")
    outs = [torch.empty_like(t) for t in jit]
    cap = tilefish.capture(fg.add_rmsnorm, *args, outputs=outs, mode="replay", plan=tmp_path, pin=p.digest)
    cap.replay()
    torch.cuda.synchronize()
    assert_same(outs, list(jit), "capture")
