"""Static slices ``x[k]`` of a parameter in the row template (note 16) on a real GPU: each slice is read at its offset, ``k * R`` with ``R`` a runtime
dimension scalar, so one binary serves every number of rows. ``jit``, ``replay`` and ``reference`` agree byte for byte, and agree with torch's
indexing: a vector-quantizer lookup of a ``[K, R]`` codes plane with a table per book or one ``[K, N, D]`` table parameter, per-row and per-column
slices, ``[R, D]`` slabs, a chunked argmax of a slab, and a graph that plans and replays one.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import slice_cases as sc  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, f32, i64, runtime  # noqa: E402

CUDA = "cuda"
ROWS = (1, 5, 777)  # odd row counts: a slice k * R elements in is not 16-byte aligned


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("slices")
    return directory, runtime.export([sc.vq_lookup, sc.vq_tables, sc.plane_rows, sc.slab_max, sc.pick_argmax], directory)


def three_modes(exported, k, *args):
    directory, m = exported
    outs = [tilefish.run(k, *args, **kw) for kw in ({}, {"mode": "replay", "manifest": directory, "pin": m.digest}, {"mode": "reference"})]
    outs = [o if isinstance(o, tuple) else (o,) for o in outs]
    for a, b, c in zip(*outs):
        assert same_bytes(a, b) and same_bytes(a, c), k.name
    return outs[0]


def codes_of(rows, n, dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, n, (sc.BOOKS, rows), generator=gen, dtype=torch.int64)
    flat = codes.view(-1)
    bad = [-1, n, n + 7, -(2**31), 2**31 - 1] + ([2**40, -(2**63)] if dtype == torch.int64 else [])
    flat[: min(len(bad), flat.numel())] = torch.tensor(bad, dtype=torch.int64)[: flat.numel()]
    return codes.to(dtype).to(CUDA)


def gather(table, idx, oob):
    n = table.shape[0]
    i = idx.long()
    if oob == "clamp":
        return table[i.clamp(0, n - 1)]
    ok = (i >= 0) & (i < n)
    return torch.where(ok[:, None], table[torch.where(ok, i, 0)], torch.zeros((), dtype=table.dtype, device=CUDA))


@pytest.mark.parametrize("rows", ROWS)
def test_a_lookup_with_a_table_per_book(exported, rows):
    codes = codes_of(rows, 1000, torch.int32, seed=rows)
    tables = [randn((1000, 128), torch.float32, seed=10 + k) for k in range(sc.BOOKS)]
    (got,) = three_modes(exported, sc.vq_lookup, codes, *tables)
    want = gather(tables[0], codes[0], "zero")
    for k in range(1, sc.BOOKS):
        want = want + gather(tables[k], codes[k], "zero")
    assert same_bytes(got, want)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("n", [64, 200])
def test_a_lookup_in_one_table_parameter(exported, rows, n):
    codes = codes_of(rows, n, torch.int64, seed=rows + n)
    tables = randn((sc.BOOKS, n, 128), torch.float32, seed=n)
    (got,) = three_modes(exported, sc.vq_tables, codes, tables)
    want = gather(tables[0], codes[0], "clamp")
    for k in range(1, sc.BOOKS):
        want = want + gather(tables[k], codes[k], "clamp")
    assert same_bytes(got, want)


@pytest.mark.parametrize("rows", ROWS)
def test_per_row_and_per_column_slices(exported, rows):
    x = randn((rows, 128), torch.float32, seed=1)
    w = randn((3, rows, 1), torch.float32, seed=2)
    bias = randn((2, 128), torch.float32, seed=3)
    codes = codes_of(rows, 50, torch.int32, seed=4)
    y, d = three_modes(exported, sc.plane_rows, x, w, bias, codes)
    assert same_bytes(y, x * w[1] + bias[0] - w[2] * bias[1])  # no contraction: torch's eager ops, op by op
    assert same_bytes(d, codes[3] - codes[0])


@pytest.mark.parametrize("rows", ROWS)
def test_slabs_and_a_chunked_argmax_of_one(exported, rows):
    xs = randn((2, rows, 128), torch.bfloat16, seed=5)
    xs.view(torch.int16).view(-1)[:4] = torch.tensor([0x7FC1, -32768, 0x0001, -1], dtype=torch.int16)  # NaN payloads, -0.0, a subnormal
    (got,) = three_modes(exported, sc.slab_max, xs)
    assert same_bytes(got, torch.where(xs[0] > xs[1], xs[0], xs[1]))
    logits = randn((2, rows, 3000), torch.bfloat16, seed=6)
    (top,) = three_modes(exported, sc.pick_argmax, logits)
    assert torch.equal(top, torch.argmax(logits[1], dim=-1))


Rb = Dim("Rb", buckets=[8, 24])


@tilefish.graph
def lookup(codes: Tensor[sc.BOOKS, Rb, i64], tables: Tensor[sc.BOOKS, 64, 128, f32]):
    return sc.vq_tables(codes, tables)


def test_in_a_plan(tmp_path):
    p = runtime.export(lookup, tmp_path)
    tables = randn((sc.BOOKS, 64, 128), torch.float32, seed=7)
    for rows in (8, 24):
        codes = codes_of(rows, 64, torch.int64, seed=rows)
        outs = [tilefish.run(lookup, codes, tables, **kw) for kw in ({}, {"mode": "replay", "plan": tmp_path, "pin": p.digest})]
        assert same_bytes(outs[0], outs[1]) and same_bytes(outs[0], tilefish.run(sc.vq_tables, codes, tables))
