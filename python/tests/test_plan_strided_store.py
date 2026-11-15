"""A view that is not dense stored into an ``InOut`` parameter of a graph (``param[...] = x.permute(1, 0)``): a launch of ``tilefish.contiguous``.

The tracer refused such a store (a store copied a contiguous tensor). In a graph body it is now accepted when the view has the parameter's shape and
dtype, and a plan makes it what a returned view that is not dense already is: a step that launches the generated op ``tilefish.contiguous`` from the
view into the caller's tensor. A dense view stays a copy step, so the plan of every graph that could be planned before is what it was; nothing is new in
the document. A kernel body still refuses it. The real launches are ``test_gpu_plan_strided_store.py``.
"""

import gpu_kernels_plans as k
import plan_fakes
import pytest
from fake_torch import FakeTensor
from test_planrun import FakeBackend, FakeProvider

import tilefish
from tilefish import Dim, InOut, Tensor, TilefishPlanError, TilefishTypeError, _plan, f32
from tilefish._planrun import PlanRunner

Rb = Dim("Rb", buckets=[8, 24])


def test_a_transpose_and_a_column_block_stored_into_inout_parameters_are_launches_of_contiguous():
    p = tilefish.plan(k.strided_store)
    for rs in (8, 24):
        steps = p.steps(Rs=rs)
        assert [(s["kind"], s.get("entry")) for s in steps] == [
            ("kernel", "tripled"),
            ("kernel", "contiguous_r2_f32_t32x64"),
            ("kernel", "contiguous_r2_f32_last32_t64x32"),
        ]
        transpose, block = steps[1]["args"], steps[2]["args"]
        assert transpose == [
            {"name": "x", "binding": "out0", "view": {"shape": [64, rs], "strides": [1, 64], "offset": 0}},
            {"name": "y", "binding": "tr"},  # the caller's InOut tensor, whole
        ]
        assert block == [{"name": "x", "binding": "x", "view": {"shape": [rs, 32], "strides": [64, 1], "offset": 32}}, {"name": "y", "binding": "cols"}]
    roles = {b["name"]: b["role"] for b in p.buffers}
    assert roles == {"x": "input", "tr": "inout", "cols": "inout", "out0": "output"}


def test_the_plan_exports_loads_and_the_runner_writes_the_callers_tensors(tmp_path):
    p = tilefish.plan(k.strided_store)
    exported = plan_fakes.export(p, tmp_path / "p")
    assert _plan.load_plan(tmp_path / "p", k.strided_store, pin=exported.digest).digest == exported.digest
    be = FakeBackend()
    x, tr, cols = FakeTensor((8, 64), "f32", 1 << 36), FakeTensor((64, 8), "f32", 2 << 36), FakeTensor((8, 32), "f32", 3 << 36)
    t = PlanRunner(p, FakeProvider(), be).run((x, tr, cols), {})
    launches = be.launches()
    pointers = [[v for kind, v in e[3] if kind == "pointer"] for e in launches]
    assert pointers == [[x._ptr, t._ptr], [t._ptr, tr._ptr], [x._ptr + 32 * 4, cols._ptr]]


def test_a_dense_view_stored_into_an_inout_parameter_is_still_a_copy_step():
    @tilefish.graph
    def row(x: Tensor[2, Rb, 64, f32], state: InOut[Tensor[Rb, 64, f32]]):
        state[...] = x[1]
        return x[0]

    assert [s["kind"] for s in tilefish.plan(row).steps(Rb=8)] == ["copy", "copy"]


def test_a_view_of_the_parameter_itself_and_a_rank_5_view_are_refused():
    @tilefish.graph
    def onto_itself(state: InOut[Tensor[64, 64, f32]]):
        state[...] = state.permute(1, 0)  # source and destination are one tensor
        return state

    with pytest.raises(TilefishPlanError, match="stores a view of 'state' into it: the source and the destination of a copy overlap"):
        tilefish.plan(onto_itself)

    @tilefish.graph
    def rank5(x: Tensor[2, 2, 2, Rb, 64, f32], state: InOut[Tensor[64, Rb, 2, 2, 2, f32]]):
        state[...] = x.permute(4, 3, 2, 1, 0)
        return x

    with pytest.raises(TilefishPlanError, match="has rank 5: tilefish.contiguous materialises views of rank 1 to 4"):
        tilefish.plan(rank5)


def test_a_kernel_body_still_refuses_to_store_a_view():
    with pytest.raises(TilefishTypeError, match="strided view cannot be stored"):

        @tilefish.kernel
        def kb(x: Tensor[Rb, 64, f32], y: InOut[Tensor[64, Rb, f32]]):
            y[...] = x.permute(1, 0)


def test_the_reference_mode_stores_the_view():
    torch = pytest.importorskip("torch")
    x = torch.randn(8, 64)
    tr, cols = torch.zeros(64, 8), torch.zeros(8, 32)
    t = tilefish.run(k.strided_store, x, tr, cols, mode="reference")
    assert torch.equal(tr, t.t()) and torch.equal(cols, x[:, 32:])
