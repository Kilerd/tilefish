"""The roles of a graph's parameters: ``@tilefish.graph(inputs=..., weights=...)``.

A plan labels a read-only tensor parameter ``input`` or ``weight``. Without a declaration the label comes from the shape (``weight`` when it mentions no
dimension), so a fully static graph labels its per-call inputs weights and a host cannot tell them apart. A graph that declares its roles gets them
in its plan, which says so (``"roles": "declared"``; a plan of a graph that declares nothing, and its digest, are what they were), and a host binds
the weights once (``tilefish.runtime.bind_weights``; Rust ``PlanRun::bind_weights``) and only the inputs at every run or capture.
"""

import json

import plan_fakes
import pytest
from fake_torch import FakeTensor
from test_planrun import FakeBackend, FakeProvider

import tilefish
from tilefish import DevScalar, Dim, InOut, Tensor, TilefishDeclarationError, TilefishManifestError, TilefishPreconditionError, TilefishRuntimeError, f32, i32
from tilefish import _plan
from tilefish._planrun import PlanRunner, weighted

Rb = Dim("Rb", buckets=[8, 24])


@tilefish.kernel
def affine(x: Tensor[16, 64, f32], w: Tensor[16, 64, f32], b: Tensor[16, 64, f32]) -> Tensor[16, 64, f32]:
    return x * w + b


@tilefish.kernel
def scale_rows(x: Tensor[Rb, 64, f32], w: Tensor[Rb, 64, f32]) -> Tensor[Rb, 64, f32]:
    return x * w


@tilefish.kernel
def bump(state: InOut[Tensor[16, 64, f32]], y: Tensor[16, 64, f32]):
    state[...] = state + y


def static_body(x: Tensor[16, 64, f32], w: Tensor[16, 64, f32], b: Tensor[16, 64, f32]):
    return affine(x, w, b)


static_guessed = tilefish.graph(static_body, name="static")
static_declared = tilefish.graph(name="static", inputs=("x",))(static_body)


@tilefish.graph(weights=("w", "b"))
def static_by_weights(x: Tensor[16, 64, f32], w: Tensor[16, 64, f32], b: Tensor[16, 64, f32]):
    return affine(x, w, b)


@tilefish.graph(inputs=("x",), weights=("w",))
def bucketed(x: Tensor[Rb, 64, f32], w: Tensor[Rb, 64, f32]):
    return scale_rows(x, w)


@tilefish.graph(inputs=("y",))
def with_state(state: InOut[Tensor[16, 64, f32]], y: Tensor[16, 64, f32], w: Tensor[16, 64, f32], b: Tensor[16, 64, f32], n: DevScalar[i32]):
    bump(state, y)
    return affine(state, w, b)


def roles(p):
    return {b["name"]: b["role"] for b in p.buffers if b["role"] not in ("intermediate", "output", "workspace")}


def test_a_static_graph_without_a_declaration_labels_every_parameter_a_weight_and_its_plan_is_unchanged():
    p = tilefish.plan(static_guessed)
    assert roles(p) == {"x": "weight", "w": "weight", "b": "weight"}
    assert "roles" not in p.doc and not p.roles_declared and static_guessed.param_roles is None


def test_a_declared_graph_has_its_roles_in_the_plan_and_says_so():
    p = tilefish.plan(static_declared)
    assert roles(p) == {"x": "input", "w": "weight", "b": "weight"}
    assert p.doc["roles"] == "declared" and p.roles_declared
    assert p.inputs == ["x"] and p.weights == ["w", "b"]
    # the same graph undeclared: the same steps and buffers but the labels and the key, and so another digest
    q = tilefish.plan(static_guessed)
    a, b = p.doc, q.doc
    assert a["buckets"] == b["buckets"] and [x["name"] for x in a["buffers"]] == [x["name"] for x in b["buffers"]]
    assert p.digest != q.digest
    assert roles(tilefish.plan(static_by_weights)) == {"x": "input", "w": "weight", "b": "weight"}


def test_a_declared_weight_may_mention_a_dimension_and_inout_and_devscalar_keep_their_roles():
    assert roles(tilefish.plan(bucketed)) == {"x": "input", "w": "weight"}
    assert with_state.param_roles == {"y": "input", "w": "weight", "b": "weight"}
    assert roles(tilefish.plan(with_state)) == {"state": "inout", "y": "input", "w": "weight", "b": "weight", "n": "devscalar"}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"inputs": "x"}, "inputs= is a list of parameter names"),
        ({"weights": ("w", "w")}, "a name is given twice"),
        ({"inputs": ("z",)}, "names 'z', which is not a parameter"),
        ({"inputs": ("state",)}, "names 'state', which is an Inout parameter"),
        ({"weights": ("n",)}, "names 'n', which is a DevScalar"),
        ({"inputs": ("y", "w"), "weights": ("w", "b")}, r"\['w'\] are named both"),
        ({"inputs": ("y",), "weights": ("w",)}, "b is in neither"),
        ({"inputs": (1,)}, "is a list of parameter names"),
    ],
)
def test_malformed_roles_are_refused_at_the_declaration(kwargs, message):
    with pytest.raises(TilefishDeclarationError, match=message):

        @tilefish.graph(**kwargs)
        def g(state: InOut[Tensor[16, 64, f32]], y: Tensor[16, 64, f32], w: Tensor[16, 64, f32], b: Tensor[16, 64, f32], n: DevScalar[i32]):
            bump(state, y)
            return affine(state, w, b)


def test_a_reader_takes_roles_declared_or_nothing():
    doc = tilefish.plan(static_declared).doc
    _plan.validate_doc(doc, made=True)
    for bad in ("guessed", True, None, 1):
        changed = dict(doc, roles=bad)
        with pytest.raises(TilefishManifestError, match="roles is the string 'declared' when present"):
            _plan.validate_doc(changed, made=True)
    with pytest.raises(TilefishManifestError, match=r"unknown \['role'\]"):
        _plan.validate_doc(dict(doc, role="declared"), made=True)


def test_a_plan_exported_without_roles_is_stale_for_the_graph_that_declares_them(tmp_path):
    exported = plan_fakes.export(tilefish.plan(static_guessed), tmp_path / "p")
    assert exported.doc.get("roles") is None
    with pytest.raises(TilefishManifestError, match="stale|first difference"):
        _plan.load_plan(tmp_path / "p", static_declared, pin=exported.digest)
    declared = plan_fakes.export(tilefish.plan(static_declared), tmp_path / "q")
    loaded = _plan.load_plan(tmp_path / "q", static_declared, pin=declared.digest)
    assert loaded.roles_declared and json.loads((tmp_path / "q" / "plan.json").read_text())["roles"] == "declared"


def runner(graph):
    return PlanRunner(tilefish.plan(graph), FakeProvider(), FakeBackend())


def tensors(**shapes):
    out, base = {}, 1 << 36
    for name, shape in shapes.items():
        out[name] = FakeTensor(shape, "f32", base)
        base += 1 << 32
    return out


def test_the_weights_are_bound_once_and_a_run_takes_the_inputs_only():
    t = tensors(x=(16, 64), w=(16, 64), b=(16, 64), x2=(16, 64))
    r = runner(static_declared)
    w = weighted(r, {"w": t["w"], "b": t["b"]})
    assert list(w.weights) == ["w", "b"]
    out = FakeTensor((16, 64), "f32", 9 << 36)
    w.run(t["x"], outputs=out)
    once = r.backend.launches()
    r.backend.log.clear()
    r.run((t["x"], t["w"], t["b"]), {}, outputs=out)  # the same launch as a run given everything
    assert r.backend.launches() == once
    r.backend.log.clear()
    w.run(x=t["x2"], outputs=out)  # by name, another input: the weights are where they were bound
    (launch,) = r.backend.launches()
    pointers = [v for kind, v in launch[3] if kind == "pointer"]
    assert pointers[:3] == [t["x2"].data_ptr(), t["w"].data_ptr(), t["b"].data_ptr()]


def test_a_capture_with_the_weights_bound_takes_the_inputs_and_the_device_scalars():
    t = tensors(state=(16, 64), y=(16, 64), w=(16, 64), b=(16, 64))
    r = runner(with_state)
    w = weighted(r, {"w": t["w"], "b": t["b"]})
    n = FakeTensor((1,), "i32", 11 << 36)
    cap = w.capture(t["state"], t["y"], devscalars={"n": n}, outputs=FakeTensor((16, 64), "f32", 12 << 36))
    assert cap.bucket == {}
    pointers = [v for e in r.backend.launches() for kind, v in e[3] if kind == "pointer"]
    assert t["w"].data_ptr() in pointers and t["y"].data_ptr() in pointers


def test_the_refusals_of_bind_weights_and_of_a_weighted_run():
    t = tensors(x=(16, 64), w=(16, 64), b=(16, 64))
    with pytest.raises(TilefishRuntimeError, match="does not declare which parameters are weights"):
        weighted(runner(static_guessed), {"x": t["x"], "w": t["w"], "b": t["b"]})
    r = runner(static_declared)
    with pytest.raises(TilefishPreconditionError, match="b is missing"):
        weighted(r, {"w": t["w"]})
    with pytest.raises(TilefishPreconditionError, match="x is not one"):
        weighted(r, {"x": t["x"], "w": t["w"], "b": t["b"]})
    with pytest.raises(TilefishPreconditionError, match="dtype"):
        weighted(r, {"w": FakeTensor((16, 64), "bf16", 1 << 40), "b": t["b"]})
    with pytest.raises(TilefishPreconditionError, match="rank"):
        weighted(r, {"w": FakeTensor((16, 64, 1), "f32", 1 << 40), "b": t["b"]})
    with pytest.raises(TilefishRuntimeError, match="maps the name of every weight"):
        weighted(r, [t["w"], t["b"]])
    w = weighted(r, {"w": t["w"], "b": t["b"]})
    with pytest.raises(TilefishPreconditionError, match="w is a weight, bound once"):
        w.run(t["x"], w=t["w"])
    with pytest.raises(TilefishPreconditionError, match="at most 1 argument"):
        w.run(t["x"], t["w"])
    assert r.backend.launches() == []
