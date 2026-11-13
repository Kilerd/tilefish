"""Manifest schema 2: what a host binds dimensions by, and the constraints of the dimensions.

Every pointer parameter of ``abi.params`` records its *declared* shape (a dimension expression stays a name, even in a binary built
for one bucket of it, and a static axis is its size as an int: ``["Rows", 1024]``), and every entry has ``dims``: for each named
dimension its ``divisible_by`` and ``buckets`` when it has them and, for a bucketed one, the ``bucket`` this binary is built for.
With those a host binds the dimensions from the shapes of its tensors, checks that they agree across arguments, and checks buckets,
``divisible_by``, ``max_elements`` and ``count_of`` from the manifest alone.
"""

import copy

import pytest
from built_fakes import rowop, scale
from lower_cases import gelu_exact, layernorm, softmax, swiglu

import tilefish
from tilefish import Dim, Out, Tensor, TilefishBuildError, TilefishSpecError, f32
from tilefish._build import abi_of, spec_entry
from tilefish.aot import CompileSpec, compile_specs
from tilefish.runtime import _target

from test_spec_validation import GOOD, INDEX, rejects


def entry_of(op, stem=None):
    specs = _target(op).compile_specs() if not isinstance(op, tilefish.ExternOp) else compile_specs(op)
    spec = next(s for s in specs if stem in (None, s.stem))
    return spec_entry(spec)


def shapes(entry):
    return {p["name"]: p["shape"] for p in entry["abi"]["params"] if p["kind"] == "pointer"}


def test_a_static_axis_is_an_int_and_a_dynamic_one_is_its_name():
    e = entry_of(softmax)
    assert shapes(e) == {"x": ["M", 1000], "out": ["M", 1000]}
    assert e["dims"] == {"M": {}}
    e = entry_of(layernorm)
    assert shapes(e) == {"x": ["M", 768], "w": [768], "b": [768], "out": ["M", 768]}


def test_an_expression_stays_an_expression_and_the_scalar_that_carries_it_is_folded():
    e = entry_of(swiglu)
    assert shapes(e)["gate"] == ["M", 3072] and e["grid"] == "(3*M,)"
    (n,) = [p for p in e["abi"]["params"] if p["kind"] == "scalar"]
    assert n["dim"] == "3072*M" and n["count_of"] == "gate"
    assert e["dims"] == {"M": {}}


def test_divisible_by_is_recorded_for_the_dimension_and_not_only_for_the_scalar_that_carries_it():
    assert entry_of(gelu_exact)["dims"] == {"T": {"divisible_by": 16}}
    e = entry_of(scale)
    assert shapes(e) == {"x": ["n"], "y": ["n"]} and e["dims"] == {"n": {"divisible_by": 16}}


def test_a_bucketed_binary_keeps_the_name_and_records_the_bucket_it_is_built_for():
    (a, b) = [spec_entry(s) for s in compile_specs(rowop)]
    assert shapes(a) == shapes(b) == {"x": ["M", "S"], "y": ["M", "S"]}
    assert a["dims"] == {"M": {}, "S": {"buckets": [32, 64], "bucket": 32}}
    assert b["dims"] == {"M": {}, "S": {"buckets": [32, 64], "bucket": 64}}
    assert a["grid"] == b["grid"] == "(M,)"
    # the spec itself still has the bucket value folded into its shape, which is what max_elements and static_size are written over
    assert [s.args[0].shape for s in compile_specs(rowop)] == [("M", "32"), ("M", "64")]


def test_a_dimension_only_a_shape_and_the_grid_mention_is_named():
    R = Dim("Rows")

    @tilefish.extern(triton="m:g", grid=lambda Rows: (Rows,))
    def with_dim(x: Tensor[R, 4, f32], y: Out[Tensor[R, 4, f32]]): ...

    e = entry_of(with_dim)
    assert e["dims"] == {"Rows": {}} and e["grid"] == "(Rows,)"
    assert shapes(e) == {"x": ["Rows", 4], "y": ["Rows", 4]}


def test_one_name_with_two_sets_of_constraints_is_refused():
    A, B = Dim("K"), Dim("K", divisible_by=16)

    @tilefish.extern(triton="m:f", grid=lambda K: (1,))
    def clash(x: Tensor[A, f32], y: Tensor[B, f32]): ...

    with pytest.raises(TilefishSpecError, match="declared twice with different constraints"):
        compile_specs(clash)


def test_every_spec_of_every_operator_has_shapes_that_say_what_its_spec_shape_says():
    """The declared shape with the bucket values put in is the shape the spec (and so max_elements and static_size) has."""
    from host_ops import ops

    compared = differences = 0
    for op in ops().values():
        for spec in _target(op).compile_specs():
            for a in spec.args:
                if a.kind != "pointer":
                    continue
                folded = tuple(str(spec.buckets.get(e, e)) if isinstance(e, str) else str(e) for e in a.shape_dims)
                compared += 1
                differences += folded != a.shape
    assert compared > 20 and differences == 0


# -- validation of what from_dict reads -----------------------------------------------------------------------------------------------------


def test_the_new_fields_are_checked_when_a_spec_is_read():
    x = INDEX["x"]
    rejects(lambda d: d["args"][x].update(shape_dims=["S"]), "one entry per axis")
    rejects(lambda d: d["args"][x].update(shape_dims="S"), "one entry per axis")
    rejects(lambda d: d["args"][x].update(shape_dims=["S", 3.5]), "int size or a dimension expression")
    rejects(lambda d: d["args"][x].update(shape_dims=["S", "n n"]), "shape_dims")
    rejects(lambda d: d["args"][INDEX["w"]].update(shape_dims=[4]), "is the size 4, but shape")
    rejects(lambda d: d["args"][INDEX["w"]].update(shape_dims=[0]), "is the size 0")
    rejects(lambda d: d["dims"].pop("S"), "dims does not name|every bucketed dimension")
    rejects(lambda d: d["dims"].pop("n"), "dims does not name")
    rejects(lambda d: d["dims"]["S"].update(bucket=32), "bucket of this spec")
    rejects(lambda d: d["dims"]["S"].pop("bucket"), "needs its bucket")
    rejects(lambda d: d["dims"]["S"].update(buckets=[32, 16]), "sorted list")
    rejects(lambda d: d["dims"]["n"].update(divisible_by=0), "divisible_by")
    rejects(lambda d: d["dims"]["n"].update(extra=1), "unknown fields")
    rejects(lambda d: d["dims"].update({"not a name": {}}), "identifier")
    rejects(lambda d: d.update(dims=[]), "dims must be an object")


def test_a_spec_without_the_new_fields_still_reads_and_does_not_verify():
    doc = copy.deepcopy(GOOD)
    del doc["dims"]
    for a in doc["args"]:
        a.pop("shape_dims", None)
    spec = CompileSpec.from_dict(doc)
    assert spec.dims == {} and all(a.shape_dims is None for a in spec.args)
    with pytest.raises(TilefishBuildError, match="declared shape"):
        abi_of(spec)  # a manifest cannot be written from it: regenerate it


def test_the_fields_survive_a_json_round_trip():
    for spec in compile_specs(rowop):
        assert CompileSpec.from_json(spec.to_json()) == spec
