"""The names of a graph's returned values in its plan: ``@tilefish.graph(outputs=("y", "residual"))``.

A plan names the buffers of the returned values by the graph's names (``out0``, ``out1``, ... when it gives none, as before: a plan of such a graph,
and its digest, are what they were). A host binds the outputs by name: Python's ``outputs=`` takes a mapping, the Rust ``PlanRun::launch`` takes the
externals by name; both accept ``out<i>`` for the ``i``-th returned value whatever its name (``_plan.output_buffer``, ``LaunchPlan::output_buffer``)
when no buffer of the plan has that name, so a host written for the names of before keeps working. No schema change.
"""

import pytest
from fake_torch import FakeTensor
from test_planrun import FakeBackend, FakeProvider

import tilefish
from tilefish import Dim, Tensor, TilefishDeclarationError, TilefishPreconditionError, f32
from tilefish._planrun import PlanRunner

Rb = Dim("Rb", buckets=[8, 24])


@tilefish.kernel
def twice(x: Tensor[Rb, 64, f32]) -> Tensor[Rb, 64, f32]:
    return x * 2.0


@tilefish.kernel
def plus(x: Tensor[Rb, 64, f32], y: Tensor[Rb, 64, f32]) -> Tensor[Rb, 64, f32]:
    return x + y


@tilefish.graph(outputs=("hidden", "residual"))
def named(x: Tensor[Rb, 64, f32]):
    r = twice(x)
    return plus(r, x), r


@tilefish.graph
def unnamed(x: Tensor[Rb, 64, f32]):
    r = twice(x)
    return plus(r, x), r


def test_the_outputs_are_the_graphs_names_and_an_unnamed_graph_keeps_out0():
    p = tilefish.plan(named)
    assert named.output_names == ("hidden", "residual") and unnamed.output_names is None
    assert p.outputs == ["hidden", "residual"]
    assert {b["name"]: b["role"] for b in p.buffers} == {"x": "input", "hidden": "output", "residual": "output"}
    q = tilefish.plan(unnamed)
    assert q.outputs == ["out0", "out1"]
    # the names are the only difference: the same steps, the buffers renamed
    assert p.doc["buckets"][0]["steps"][0]["args"][1]["binding"] == "residual" and q.doc["buckets"][0]["steps"][0]["args"][1]["binding"] == "out1"


def test_out_i_finds_the_i_th_output_unless_a_buffer_has_that_name():
    p = tilefish.plan(named)
    assert [p.output_buffer(n) for n in ("hidden", "residual", "out0", "out1", "out2", "out01", "x", "t0")] == [
        "hidden", "residual", "hidden", "residual", None, None, None, None,
    ]

    @tilefish.graph(outputs=("out1", "out0"))
    def crossed(x: Tensor[Rb, 64, f32]):
        r = twice(x)
        return plus(r, x), r

    c = tilefish.plan(crossed)
    assert c.output_buffer("out0") == "out0" and c.output_buffer("out1") == "out1"  # a name a buffer has is that buffer


@pytest.mark.parametrize(
    "outputs, message",
    [
        ("y", "is a list of the names"),
        (("y",), r"names 1 value\(s\) \(y\), but the body returns 2"),
        (("y", "y"), "given twice"),
        (("y", "x"), r"are names of parameters"),
        (("y", "1z"), "ASCII identifiers"),
        (("y", "é"), "ASCII identifiers"),
    ],
)
def test_malformed_names_are_refused_at_the_declaration(outputs, message):
    with pytest.raises(TilefishDeclarationError, match=message):

        @tilefish.graph(outputs=outputs)
        def g(x: Tensor[Rb, 64, f32]):
            r = twice(x)
            return plus(r, x), r


def test_a_name_an_intermediate_would_have_takes_precedence():
    @tilefish.graph(outputs=("t0",))
    def g(x: Tensor[Rb, 64, f32]):
        r = twice(x)
        return plus(r, x)

    p = tilefish.plan(g)
    assert p.outputs == ["t0"] and sorted(b["name"] for b in p.buffers) == ["t0", "t0_", "x"]


def test_a_run_takes_the_outputs_by_name_or_by_out_i():
    p = tilefish.plan(named)
    be = FakeBackend()
    r = PlanRunner(p, FakeProvider(), be)
    x = FakeTensor((8, 64), "f32", 1 << 36)
    hidden, residual = FakeTensor((8, 64), "f32", 2 << 36), FakeTensor((8, 64), "f32", 3 << 36)
    for given in ({"hidden": hidden, "residual": residual}, {"out0": hidden, "out1": residual}, {"residual": residual, "out0": hidden}, (hidden, residual)):
        out = r.run((x,), {}, outputs=given)
        assert out == (hidden, residual)
    for given, message in (
        ({"hidden": hidden}, "residual is missing"),
        ({"hidden": hidden, "out0": hidden, "residual": residual}, "gives the returned value 'hidden' twice"),
        ({"hidden": hidden, "nope": residual}, "names 'nope', which is not a returned value"),
    ):
        with pytest.raises(TilefishPreconditionError, match=message):
            r.run((x,), {}, outputs=given)
