"""Fixed arguments given per call: a prebuilt binary's ``const[i32]`` / ``const[i64]`` with no default.

A fixed argument with a default is fixed per declaration (its value is in the manifest), so a kernel called at several geometries needed one
declaration, and one binary, per geometry. Without a default the call gives the value: one declaration and one manifest entry serve every value;
the ABI has the parameter as ``kind: "given"`` (no value), a plan records the values of each step (``"given": {"k": 3}``, per bucket: a value may be
an expression of the graph's dimensions) and so its digest has them, and both readers hold a step to the entry (every given argument once, of its
type). The launches carry the values at their slots.
"""

import json

import gpu_kernels_p6 as k
import plan_fakes
import pytest
from fake_torch import FakeTensor
from test_planrun import FakeBackend, FakeProvider

import tilefish
from tilefish import Dim, Out, Tensor, TilefishDeclarationError, TilefishManifestError, TilefishTypeError, const, f32, i32, _plan
from tilefish._build import abi_of
from tilefish._planrun import PlanRunner
from tilefish.aot import compile_specs, declaration_digest

Rg, Nn = k.Rg, k.Nn
PTX = k.PTX / "scale_add_f32.ptx"


def test_a_const_without_a_default_is_given_per_call_in_the_spec_and_the_abi():
    (spec,) = compile_specs(k.scale_add)
    kinds = {a.name: (a.kind, a.type, a.value) for a in spec.args}
    assert kinds["k"] == ("given", "i32", None) and kinds["c"] == ("given", "i64", None)
    params = {p["name"]: p for p in abi_of(spec)["params"]}
    assert params["k"] == {"slot": 3, "index": 3, "name": "k", "kind": "given", "type": "i32", "ptx_type": "u32", "size": 4}
    assert params["c"]["kind"] == "given" and params["c"]["ptx_type"] == "u64" and "value" not in params["c"]

    @tilefish.extern(ptx=PTX, symbol="scale_add_f32", block=(128,), grid=lambda Nn: (tilefish.cdiv(Nn, 128),), name="scale_add")
    def fixed(x: Tensor[Nn, f32], y: Out[Tensor[Nn, f32]], n_elem: Nn, k: const[i32] = 3, c: const[tilefish.i64] = 0): ...

    assert declaration_digest(fixed) != declaration_digest(k.scale_add)  # a value per declaration is another declaration


@pytest.mark.parametrize(
    "annotation, message",
    [
        ("const[f32]", "a per-call value is an integer"),
        ("const", "declare it `k: const\\[i32\\]`"),
    ],
)
def test_a_per_call_argument_is_an_integer_of_a_prebuilt_binary(annotation, message):
    namespace = {"tilefish": tilefish, "Tensor": Tensor, "Out": Out, "Nn": Nn, "f32": f32, "i32": i32, "const": const, "PTX": PTX}
    source = (
        "@tilefish.extern(ptx=PTX, symbol='scale_add_f32', block=(128,), grid=lambda Nn: (tilefish.cdiv(Nn, 128),))\n"
        f"def op(x: Tensor[Nn, f32], y: Out[Tensor[Nn, f32]], n_elem: Nn, k: {annotation}): ...\n"
    )
    with pytest.raises(TilefishDeclarationError, match=message):
        exec(source, namespace)  # noqa: S102


def test_a_triton_extern_still_needs_a_default():
    with pytest.raises(TilefishDeclarationError, match="needs a default, the value compiled in as constexpr"):

        @tilefish.extern(triton="m:f", grid=(1,))
        def op(x: Tensor[Nn, f32], y: Out[Tensor[Nn, f32]], k: const[i32]): ...


@pytest.mark.parametrize(
    "call, error, message",
    [
        (lambda x: k.scale_add(x, k=3), TypeError, "missing a required argument: 'c'"),
        (lambda x: k.scale_add(x, k=True, c=0), TilefishTypeError, "it takes an int, got True"),
        (lambda x: k.scale_add(x, k=1.5, c=0), TilefishTypeError, "it takes an int, got 1.5"),
        (lambda x: k.scale_add(x, k=2**31, c=0), TilefishTypeError, "does not fit i32"),
        (lambda x: k.scale_add(x, k=1, c=2**63), TilefishTypeError, "does not fit i64"),
        (lambda x: k.scale_add(x, k=Dim("Z"), c=0), TilefishTypeError, "not an int nor an expression of the dimensions of the graph"),
    ],
)
def test_a_call_gives_an_int_of_the_type_or_an_expression_of_the_graphs_dimensions(call, error, message):
    with pytest.raises((error, TilefishTypeError), match=message):

        @tilefish.graph
        def g(x: Tensor[Rg, 64, f32]):
            return call(x.reshape(Rg * 64))


def test_a_plan_records_the_values_of_each_step_and_one_entry_serves_them_all():
    p = tilefish.plan(k.two_scales)
    assert list(p.callees) == ["scale_add"]  # one binary
    for rg in (8, 24):
        first, second = p.steps(Rg=rg)
        assert first["entry"] == second["entry"] == "scale_add"
        assert first["given"] == {"k": 3, "c": -2} and second["given"] == {"k": -1, "c": rg}  # c = Rg: a value per bucket
    # the values are part of the digest: another value is another plan

    @tilefish.graph(name="two_scales")
    def other(x: Tensor[Rg, 64, f32]):
        a = k.scale_add(x.reshape(Rg * 64), k=4, c=-2)
        return k.scale_add(a, k=-1, c=Rg).reshape(Rg, 64)

    assert tilefish.plan(other).digest != p.digest


def test_the_exported_plan_reads_back_and_both_rules_hold_a_step_to_the_entry(tmp_path):
    exported = plan_fakes.export(tilefish.plan(k.two_scales), tmp_path / "p")
    manifest = exported.manifest
    entry = manifest.get("scale_add", "sm_103").entry
    assert [p["kind"] for p in entry["abi"]["params"]] == ["pointer", "pointer", "scalar", "given", "given"]
    loaded = _plan.load_plan(tmp_path / "p", k.two_scales, pin=exported.digest)
    assert loaded.steps(Rg=24)[1]["given"] == {"k": -1, "c": 24}
    doc = exported.doc

    def refused(change, message, where=_plan.validate_doc):
        d = json.loads(json.dumps(doc))
        change(d)
        with pytest.raises(TilefishManifestError, match=message):
            if where is _plan.validate_doc:
                _plan.validate_doc(d)
            _plan.check_against_manifest(d, manifest)

    def each(change):  # the same change of step 0 in every bucket (the names of the given values are the same in every bucket)
        return lambda d: [change(b["steps"][0]) for b in d["buckets"]]

    refused(each(lambda st: st.pop("given")), r"takes the fixed arguments per call \['c', 'k'\], the step gives \[\]")
    refused(each(lambda st: st["given"].pop("c")), "the step gives \\['k'\\]")
    refused(each(lambda st: st["given"].update(z=1)), "the step gives \\['c', 'k', 'z'\\]")
    refused(each(lambda st: st["given"].update(k=2**31)), "does not fit the i32")
    refused(each(lambda st: st["given"].update(k=1.0)), "given maps argument names to ints")
    refused(each(lambda st: st["given"].update(k=True)), "given maps argument names to ints")
    refused(each(lambda st: st.update(given={})), "given is a non-empty object")
    refused(each(lambda st: st.update(given=[3, -2])), "given is a non-empty object")
    refused(lambda d: d["buckets"][0]["steps"][0]["given"].update(z=1), "other kinds, workspaces, argument names or bindings \\(or names of given")
    # the values may differ between buckets (an expression of the dimensions): a document with other values is read
    d = json.loads(json.dumps(doc))
    d["buckets"][0]["steps"][0]["given"]["k"] = 5
    _plan.validate_doc(d)
    _plan.check_against_manifest(d, manifest)


def test_a_run_launches_the_values_at_their_slots():
    p = tilefish.plan(k.two_scales)
    be = FakeBackend()
    r = PlanRunner(p, FakeProvider(), be)
    x = FakeTensor((8, 64), "f32", 1 << 36)
    r.run((x,), {}, outputs=FakeTensor((8, 64), "f32", 2 << 36))
    first, second = be.launches()
    assert [kind for kind, _ in first[3]] == ["pointer", "pointer", "scalar", "given", "given"]
    assert [v for kind, v in first[3] if kind == "given"] == [3, -2]
    assert [v for kind, v in second[3] if kind == "given"] == [-1, 8]


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda p: p.update(value=3), "it has no value, dim or shape"),
        (lambda p: p.update(dim="Nn"), "it has no value, dim or shape"),
        (lambda p: p.update(type="f32", ptx_type="f32"), "is i32 or i64"),
        (lambda p: p.update(ptx_type="u64", size=8), "is passed as a u32 of 4 bytes"),
        (lambda p: p.update(kind="giving"), "kind must be pointer, scalar, fixed or given"),
    ],
)
def test_the_manifest_reader_holds_a_given_argument_to_its_rules(tmp_path, change, message):
    from tilefish import Manifest

    plan_fakes.export(tilefish.plan(k.two_scales), tmp_path / "p")
    path = tmp_path / "p" / "manifest.json"
    doc = json.loads(path.read_text())
    (entry,) = doc["kernels"]
    change(next(p for p in entry["abi"]["params"] if p["name"] == "k"))
    path.write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match=message):
        Manifest.load(tmp_path / "p")
