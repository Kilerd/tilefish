"""``gather_rows`` and ``scatter_rows`` on a real GPU: raw copies of rows and elements, and the out-of-range policies, byte for byte.

Tables hold every bf16 bit pattern (NaN payloads, signed zeros, subnormals, infinities), so a copy that interprets a value would show. The
indices mix rows in range with negative ones, ``N``, and the extremes of ``i32`` and ``i64``. ``jit``, ``replay`` and ``reference`` agree byte for
byte, and they agree with torch indexing: ``zero`` gives rows of ``+0.0``, ``clamp`` the nearest row, a scatter's ``skip`` leaves the target
alone; the rows of a scatter target that no index names keep their bytes. A graph of a masked lookup and an index copy plans and replays.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gather_cases as gc  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import DevScalar, Dim, InOut, Tensor, bf16, i32, i64, runtime  # noqa: E402

BF16, F32 = torch.bfloat16, torch.float32
CUDA = "cuda"
D = 128


def bf16_table(rows, seed=0):
    """``[rows, 128]`` bf16, the 65536 bit patterns in a shuffled cycle: every pattern appears once per 512 rows."""
    gen = torch.Generator().manual_seed(seed)
    words = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)[torch.randperm(65536, generator=gen)]
    flat = words.repeat(-(-rows * D // 65536))[: rows * D]
    return flat.view(BF16).reshape(rows, D).to(CUDA)


def f32_table(rows, cols=None, seed=0):
    t = randn((rows,) if cols is None else (rows, cols), F32, seed=seed)
    flat = t.view(-1)
    specials = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1e-45, -1e-40], device=CUDA)
    flat[: specials.numel()] = specials
    flat.view(torch.int32)[7] = -4194303  # a negative NaN with a payload (0xFFC00001)
    return t


def ids_of(n_rows, count, dtype, seed=1, extremes=True):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, n_rows, (count,), generator=gen, dtype=torch.int64)
    if extremes:
        bad = [-1, n_rows, n_rows + 1, -n_rows]
        if dtype == torch.int64:
            bad += [2**40, -(2**40), 2**63 - 1, -(2**63), 2**32 + 3]  # 2**32 + 3 is 3 in i32: masked, not wrapped
        elif dtype == torch.int32:
            bad += [2**31 - 1, -(2**31)]
        ids[: len(bad)] = torch.tensor(bad, dtype=torch.int64)
    if dtype == torch.uint8:
        ids = ids.clamp(0, 255)
    return ids.to(dtype).to(CUDA)


def in_range(ids, n_rows):
    i = ids.long()
    return (i >= 0) & (i < n_rows)


def gather_zero(table, ids):
    ok = in_range(ids, table.shape[0])
    rows = table[torch.where(ok, ids.long(), 0)]
    zero = torch.zeros((), dtype=table.dtype, device=CUDA)
    return torch.where(ok.reshape(*ok.shape, *[1] * (table.dim() - 1)), rows, zero)


def as_tuple(r):
    return r if isinstance(r, tuple) else (r,)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gather")
    ops = [gc.embed_lookup, gc.embed_masked, gc.take_clamp, gc.take_static, gc.take_elems, gc.take_elems_i64, gc.scale_rows, gc.index_copy, gc.scatter_elems, gc.permute_rows]
    return directory, runtime.export(ops, directory)


def three_modes(exported, k, args, inout=()):
    """Run ``k`` in jit, replay and reference (each on its own copy of the in-place arguments); return the results and the in-place tensors."""
    directory, m = exported
    out = []
    for kw in ({}, {"mode": "replay", "manifest": directory, "pin": m.digest}, {"mode": "reference"}):
        a = [x.clone() if i in inout else x for i, x in enumerate(args)]
        r = tilefish.run(k, *a, **kw)
        out.append((as_tuple(r) if r is not None else (), [a[i] for i in inout]))
    (jr, ji), (pr, pi), (rr, ri) = out
    for a, b, c in zip(jr, pr, rr):
        assert same_bytes(a, b) and same_bytes(a, c), k.name
    for a, b, c in zip(ji, pi, ri):
        assert same_bytes(a, b) and same_bytes(a, c), k.name
    return jr, ji


def test_an_embedding_lookup_copies_rows_and_zeroes_an_index_out_of_range(exported):
    table = bf16_table(1000)
    ids = ids_of(1000, 777, torch.int64)
    (got,), _ = three_modes(exported, gc.embed_lookup, (table, ids))
    assert torch.equal(got.view(torch.int16), gather_zero(table, ids).view(torch.int16))
    bad = ~in_range(ids, 1000)
    assert bad.sum() == 9 and (got[bad].view(torch.int16) == 0).all()  # +0.0, all bits clear
    assert torch.equal(got[~bad].view(torch.int16), table[ids[~bad]].view(torch.int16))  # every payload as it was


def test_a_masked_lookup_is_the_lookup_then_the_row_mask(exported):
    table = bf16_table(600, seed=2)
    ids = ids_of(600, 300, torch.int64, seed=3)
    for n in (0, 1, 150, 300):
        (got,), _ = three_modes(exported, gc.embed_masked, (table, ids, torch.tensor([n], dtype=torch.int32, device=CUDA)))
        rows = gather_zero(table, ids)
        want = torch.cat([rows[:n], (rows[n:].float() * 0.0).to(BF16)])
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), n


@pytest.mark.parametrize("rows", [64, 200])
def test_clamp_reads_the_nearest_row_at_every_bucket(exported, rows):
    table = f32_table(rows, D, seed=rows)
    idx = ids_of(rows, 500, torch.int32, seed=4)
    (got,), _ = three_modes(exported, gc.take_clamp, (table, idx))
    assert same_bytes(got, table[idx.long().clamp(0, rows - 1)])


def test_a_static_table_and_a_u8_index(exported):
    table = bf16_table(1000, seed=5)
    idx = torch.arange(256, dtype=torch.uint8, device=CUDA)
    (got,), _ = three_modes(exported, gc.take_static, (table, idx))
    assert same_bytes(got, table[:256])


def test_element_gathers(exported):
    x = f32_table(100, seed=6)
    idx = ids_of(256, 4099, torch.uint8, seed=7)  # many beyond the 100 rows
    (clamped, zeroed), _ = three_modes(exported, gc.take_elems, (x, idx))
    assert same_bytes(clamped, x[idx.long().clamp(0, 99)]) and same_bytes(zeroed, gather_zero(x, idx))
    xb = bf16_table(64, seed=8).flatten()
    idx64 = ids_of(xb.numel(), 6000, torch.int64, seed=9)
    (got,), _ = three_modes(exported, gc.take_elems_i64, (xb, idx64))
    assert same_bytes(got, gather_zero(xb, idx64))


def test_a_per_row_gather_from_a_flat_table(exported):
    x = randn((300, D), F32, seed=10)
    scales = f32_table(50, seed=11)
    ids = ids_of(50, 300, torch.int32, seed=12).reshape(300, 1)
    (got,), _ = three_modes(exported, gc.scale_rows, (x, scales, ids))
    assert same_bytes(got, x * gather_zero(scales, ids))


def test_an_index_copy_writes_the_named_rows_and_keeps_the_others(exported):
    dst = bf16_table(900, seed=13)
    src = bf16_table(400, seed=14)
    gen = torch.Generator().manual_seed(15)
    idx = torch.randperm(900, generator=gen)[:400].to(torch.int64)
    idx[:6] = torch.tensor([-1, 900, 2**40, -(2**63), 2**63 - 1, 2**32 + 7])  # skipped: written nowhere (2**32 + 7 does not wrap to 7)
    idx = idx.to(CUDA)
    before = dst.clone()
    _, (after,) = three_modes(exported, gc.index_copy, (dst, idx, src), inout=(0,))
    want = before.clone()
    ok = in_range(idx, 900)
    want[idx[ok]] = src[ok]
    assert torch.equal(after.view(torch.int16), want.view(torch.int16))
    named = torch.zeros(900, dtype=torch.bool, device=CUDA)
    named[idx[ok]] = True
    assert torch.equal(after[~named].view(torch.int16), before[~named].view(torch.int16))


def test_an_element_scatter(exported):
    dst = f32_table(6000, seed=16)
    gen = torch.Generator().manual_seed(17)
    idx = torch.randperm(6000, generator=gen)[:3000].to(torch.int32)
    idx[:3] = torch.tensor([-5, 6000, 2**31 - 1], dtype=torch.int32)
    idx = idx.to(CUDA)
    src = f32_table(3000, seed=18)
    _, (after,) = three_modes(exported, gc.scatter_elems, (dst, idx, src), inout=(0,))
    want = dst.clone()
    ok = in_range(idx, 6000)
    want[idx[ok].long()] = src[ok]
    assert same_bytes(after, want)


def test_a_table_that_is_also_an_input(exported):
    x = bf16_table(333, seed=19)
    perm = ids_of(333, 333, torch.int32, seed=20)
    (got,), _ = three_modes(exported, gc.permute_rows, (x, perm))
    y = x[perm.long().clamp(0, 332)]
    assert same_bytes(got, torch.where(torch.isnan(y), x, y))


# -- in a launch plan ----------------------------------------------------------------------------------------------------------

Rb = Dim("Rb", buckets=[32, 64])
Vt = Dim("Vt", buckets=[500])  # a plan has one list of steps per bucket of every dimension
Vbuf = Dim("Vbuf", buckets=[200])


@tilefish.graph
def embed_and_place(buf: InOut[Tensor[Vbuf, gc.D, bf16]], table: Tensor[Vt, gc.D, bf16], ids: Tensor[Rb, i64], pos: Tensor[Rb, i64], n: DevScalar[i32]):
    rows = gc.embed_masked(table, ids, n)
    gc.index_copy(buf, pos, rows)
    return rows


def test_a_graph_of_a_masked_lookup_and_an_index_copy_plans_and_replays(tmp_path):
    p = runtime.export(embed_and_place, tmp_path)
    table = bf16_table(500, seed=21)
    for rows, n in ((32, 32), (64, 17)):
        ids = ids_of(500, rows, torch.int64, seed=rows)
        gen = torch.Generator().manual_seed(rows)
        pos = torch.randperm(200, generator=gen)[:rows].to(torch.int64).to(CUDA)
        n_t = torch.tensor([n], dtype=torch.int32, device=CUDA)
        buf = bf16_table(200, seed=22)
        results = []
        for kw in ({}, {"mode": "replay", "plan": tmp_path, "pin": p.digest}, {"mode": "reference"}):
            b = buf.clone()
            r = tilefish.run(embed_and_place, b, table, ids, pos, n if kw.get("mode") == "reference" else n_t, **kw)
            results.append((r, b))
        (jr, jb), (pr, pb), (rr, rb) = results
        assert same_bytes(jr, pr) and same_bytes(jr, rr) and same_bytes(jb, pb) and same_bytes(jb, rb)
        eager = tilefish.run(gc.embed_masked, table, ids, n_t)
        assert same_bytes(jr, eager)
        want = buf.clone()
        want[pos] = eager
        assert same_bytes(jb, want)
