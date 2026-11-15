"""Chunks of a ``split`` whose count depends on a dimension (:class:`tilefish.ops.Chunks`) on a real GPU: a plan reads ``chunks[0]``,
``chunks[1]`` and ``chunks[-1]`` as views at their offsets, at every bucket, and ``jit``, ``replay`` and ``reference`` agree with torch's
slices byte for byte; so does a graph whose dimension is only ``divisible_by``.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, bf16, f32, runtime  # noqa: E402

Mb = Dim("Mb", buckets=[8, 24])
Cb = Dim("Cb", buckets=[64, 128])
M = Dim("M")
C = Dim("C", divisible_by=16)


@tilefish.graph
def ends(x: Tensor[Mb, Cb, bf16]):
    chunks = x.split(32, axis=1)  # 2 or 4 chunks of columns: strided views
    return tilefish.contiguous(chunks[0]), tilefish.contiguous(chunks[1]), tilefish.contiguous(chunks[-1])


@tilefish.graph
def last_rows(x: Tensor[C, M, f32]):
    return tilefish.contiguous(x.split(16, axis=0)[-1])  # a dense slab at a symbolic offset


def test_a_plan_of_chunks_at_every_bucket(tmp_path):
    p = runtime.export(ends, tmp_path)
    for rows in (8, 24):
        for cols in (64, 128):
            x = randn((rows, cols), torch.bfloat16, seed=rows + cols)
            outs = [tilefish.run(ends, x, **kw) for kw in ({}, {"mode": "replay", "plan": tmp_path, "pin": p.digest}, {"mode": "reference"})]
            for a, b, c in zip(*outs):
                assert same_bytes(a, b) and same_bytes(a, c)
            for got, want in zip(outs[0], (x[:, :32], x[:, 32:64], x[:, -32:])):
                assert same_bytes(got, want.contiguous())


def test_a_dimension_that_is_only_divisible(tmp_path):
    p = runtime.export(last_rows, tmp_path, buckets={"C": [16, 48, 160], "M": [7]})  # a plan has steps per bucket: chosen at the export
    for c in (16, 48, 160):
        x = randn((c, 7), torch.float32, seed=c)
        for kw in ({"mode": "replay", "plan": tmp_path, "pin": p.digest}, {"mode": "reference"}):
            assert same_bytes(tilefish.run(last_rows, x, **kw), x[-16:].contiguous())
