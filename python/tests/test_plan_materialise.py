"""Views that are not dense, returned or copied, in a plan: a launch of ``tilefish.contiguous``.

A copy step copies one run of bytes, so a returned value or a ``tilefish.copy`` of a view that is not dense at some bucket (a transpose, a column block)
was refused. It is now a step that launches the generated op ``tilefish.contiguous`` (``_copyops``: a raw copy of a strided view of rank 1 to 4 into a
contiguous tensor of its shape) from the view into the destination buffer. A dense view stays a copy step, so the plan of every graph that could be
planned before (and its digest) is what it was. Nothing new in the document: it is a kernel step.
"""

import gpu_kernels_p6 as k
import plan_fakes
import pytest
from fake_torch import FakeTensor
from test_planrun import FakeBackend, FakeProvider

import tilefish
from tilefish import Dim, Tensor, TilefishPlanError, f32, _plan
from tilefish._planrun import PlanRunner

Rb = Dim("Rb", buckets=[8, 24])


def test_a_transpose_and_a_column_block_are_launches_and_a_dense_row_is_a_copy():
    p = tilefish.plan(k.strided_returns)
    for rg in (8, 24):
        steps = p.steps(Rg=rg)
        assert [(s["kind"], s.get("entry", "")) for s in steps] == [
            ("kernel", "triple"),
            ("kernel", "contiguous_r2_f32_last32_t64x32"),
            ("kernel", "contiguous_r2_f32_t32x64"),
            ("copy", ""),
        ]
        block, transpose = steps[1]["args"], steps[2]["args"]
        assert block[0] == {"name": "x", "binding": "x", "view": {"shape": [rg, 32], "strides": [64, 1], "offset": 32}}
        assert block[1] == {"name": "y", "binding": "out1"}  # the copy made by tilefish.copy is the returned value itself
        assert transpose[0] == {"name": "x", "binding": "t0", "view": {"shape": [64, rg], "strides": [1, 64], "offset": 0}}
        assert transpose[1] == {"name": "y", "binding": "out0"}
    assert {b["name"]: b["shape"] for b in p.buffers if b["role"] == "output"} == {"out0": [64, "Rg"], "out1": ["Rg", 32], "out2": [64]}


def test_the_plan_exports_reads_back_and_the_runner_launches_the_views(tmp_path):
    p = tilefish.plan(k.strided_returns)
    exported = plan_fakes.export(p, tmp_path / "p")
    assert sorted(exported.callees) == ["contiguous_r2_f32_last32_t64x32", "contiguous_r2_f32_t32x64", "triple"]
    assert _plan.load_plan(tmp_path / "p", k.strided_returns, pin=exported.digest).digest == exported.digest
    be = FakeBackend()
    x = FakeTensor((8, 64), "f32", 1 << 36)
    transpose, block, row = PlanRunner(p, FakeProvider(), be).run((x,), {})
    assert transpose.shape == (64, 8) and block.shape == (8, 32) and row.shape == (64,)
    launches = be.launches()
    assert [e[0] for e in launches] == ["launch", "launch", "launch", "copy"]
    pointers = [v for kind, v in launches[1][3] if kind == "pointer"]
    assert pointers == [x.data_ptr() + 32 * 4, block.data_ptr()]  # the view's first element, into the caller's output
    strides = [v for kind, v in launches[1][3] if kind == "scalar"][-2:]
    assert strides == [64, 1]  # the strides of the view, as the generated op takes them


def test_a_dense_view_is_still_a_copy_step_and_a_rank_5_view_is_refused():
    @tilefish.graph
    def dense(x: Tensor[2, Rb, 64, f32]):
        return x[1]

    assert [s["kind"] for s in tilefish.plan(dense).steps(Rb=8)] == ["copy"]

    @tilefish.graph
    def rank5(x: Tensor[2, 2, 2, Rb, 64, f32]):
        return x.permute(4, 3, 2, 1, 0)

    with pytest.raises(TilefishPlanError, match="has rank 5: tilefish.contiguous materialises views of rank 1 to 4"):
        tilefish.plan(rank5)
