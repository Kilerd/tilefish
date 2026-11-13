"""What a graph may return in a launch plan: a tensor a call made, a reshape of one, any dense view, a parameter, a value twice.

* a tensor a call made is the caller's output buffer itself (the call writes there), as before;
* **a reshape of all of it** (offset 0, the row-major strides, as many elements) costs nothing either: the buffer is the caller's output in the returned
  shape, and the steps that write and read it bind it through a view of their own shape;
* **any other dense view** (a ``select`` of the first axis, a chunk of a ``split`` along it), **a parameter** and **a value returned twice** are a copy
  step (``docs/plan-schema-1.md``, section 7.5) at the end of the steps, into a buffer of the caller's;
* a view that is not dense is refused (a copy step copies one run of bytes), and so are a device scalar and a view of a tensor updated in place after it.

Nothing here needs a GPU (``test_gpu_plan_returns.py``).
"""

import plan_fakes
import pytest
from fake_torch import FakeTensor
from test_planrun import ARENA, FakeBackend, FakeProvider

import tilefish
from tilefish import DevScalar, Dim, InOut, Tensor, TilefishPlanError, f32, i32
from tilefish._planrun import PlanRunner

Rb = Dim("Rb", buckets=[8, 24])


@tilefish.kernel
def flat_twice(x: Tensor[Rb * 64, f32]) -> Tensor[Rb * 64, f32]:
    return x * 2.0


@tilefish.kernel
def twice(x: Tensor[Rb, 64, f32]) -> Tensor[Rb, 64, f32]:
    return x * 2.0


@tilefish.kernel
def bump(state: InOut[Tensor[Rb, 64, f32]], y: Tensor[Rb, 64, f32]):
    state[...] = state + y


@tilefish.graph
def reshaped(x: Tensor[Rb, 64, f32]):
    y = flat_twice(x.reshape(Rb * 64))
    return y.reshape(Rb, 64)  # all of y in another shape: y's buffer is the output


def kinds(p, rb=8):
    return [s["kind"] for s in p.steps(Rb=rb)]


def buffer(p, name):
    return next(b for b in p.buffers if b["name"] == name)


def test_a_reshape_of_a_result_is_its_buffer_in_the_returned_shape():
    p = tilefish.plan(reshaped)
    assert kinds(p) == ["kernel"] and p.outputs == ["out0"]
    out = buffer(p, "out0")
    assert out["role"] == "output" and out["shape"] == ["Rb", 64] and out["offset"] is None
    (step,) = p.steps(Rb=24)
    args = {a["name"]: a for a in step["args"]}
    assert args["x"]["view"] == {"shape": [24 * 64], "strides": [1], "offset": 0}  # x read flat
    assert args["out"] == {"name": "out", "binding": "out0", "view": {"shape": [24 * 64], "strides": [1], "offset": 0}}  # written flat
    assert p.arena_bytes == 0  # nothing in the arena


def test_a_dense_view_a_parameter_and_a_value_returned_twice_are_copied_at_the_end():
    @tilefish.graph
    def parts(x: Tensor[2, Rb, 64, f32]):
        y = twice(x[1])
        return y, x[0], y, x  # a result; a dense view of a parameter; the result again; the parameter

    p = tilefish.plan(parts)
    assert kinds(p) == ["kernel", "copy", "copy", "copy"]
    assert p.outputs == ["out0", "out1", "out2", "out3"]
    steps = p.steps(Rb=8)
    assert steps[1]["args"] == [
        {"name": "src", "binding": "x", "view": {"shape": [8, 64], "strides": [64, 1], "offset": 0}},
        {"name": "dst", "binding": "out1"},
    ]
    assert steps[2]["args"] == [{"name": "src", "binding": "out0"}, {"name": "dst", "binding": "out2"}]  # the second return of y copies the first
    assert steps[3]["args"] == [{"name": "src", "binding": "x"}, {"name": "dst", "binding": "out3"}]
    assert [buffer(p, n)["shape"] for n in p.outputs] == [["Rb", 64], ["Rb", 64], ["Rb", 64], [2, "Rb", 64]]


def test_an_updated_inout_parameter_returned_is_its_final_version_copied():
    @tilefish.graph
    def returned_state(x: Tensor[Rb, 64, f32], state: InOut[Tensor[Rb, 64, f32]]):
        bump(state, x)
        return state

    p = tilefish.plan(returned_state)
    assert kinds(p) == ["kernel", "copy"] and p.steps(Rb=8)[1]["args"][0] == {"name": "src", "binding": "state"}


def test_a_device_scalar_and_a_stale_view_are_refused():
    # (a view that is not dense is materialised by tilefish.contiguous: test_plan_materialise.py)

    @tilefish.graph
    def scalar(x: Tensor[Rb, 64, f32], n: DevScalar[i32]):
        return twice(x), n

    with pytest.raises(TilefishPlanError, match="is the DevScalar 'n'"):
        tilefish.plan(scalar)

    @tilefish.graph
    def stale(x: Tensor[Rb, 64, f32]):
        r = tilefish.copy(x)
        v = r.reshape(Rb * 64)
        bump(r, x)
        return v

    with pytest.raises(TilefishPlanError, match="updated in place after the view was made"):
        tilefish.plan(stale)


def test_the_runner_writes_a_reshaped_result_and_copies_the_others(tmp_path):
    @tilefish.graph
    def mixed(x: Tensor[2, Rb, 64, f32]):
        y = twice(x[1])
        return y.reshape(Rb * 64), x[0]

    p = tilefish.plan(mixed)
    plan_fakes.export(p, tmp_path / "mixed")
    be = FakeBackend()
    x = FakeTensor((2, 8, 64), "f32", 1 << 36)
    flat, first = PlanRunner(p, FakeProvider(), be).run((x,), {})
    assert flat.shape == (8 * 64,) and first.shape == (8, 64)
    launch, copy = be.launches()
    assert ("pointer", flat._ptr) in launch[3]  # twice writes the caller's flat output, through a view of its own shape
    assert copy[1:4] == (first._ptr, x._ptr, 8 * 64 * 4)
    assert ARENA  # (no arena buffer is used: the result is the caller's)
