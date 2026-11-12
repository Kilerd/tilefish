"""ast_source_inputs() only runs on a verified spec.

Specs from compile_specs are verified; specs from from_dict / from_json start unverified and become verified
only after a successful verify_against(op). The flag is not part of the serialized spec.
"""

import copy
import dataclasses
import json
import pickle

import pytest

import tilefish
from tilefish import Dim, Out, Tensor, TilefishSpecError, bf16, const, f32
from tilefish.aot import CompileSpec, compile_specs

S = Dim("S", buckets=[16, 32])
n = Dim("n", divisible_by=16)


@tilefish.extern(triton="my_kernels:fn", grid=lambda S, n: (S, n))
def kernel(x: Tensor[S, n, bf16], y: Out[Tensor[S, n, bf16]], cnt: n, *, B: const = S, E: const[f32] = 1.0): ...


@tilefish.extern(triton="my_kernels:other", grid=lambda n: (n,))
def other(x: Tensor[n, bf16], y: Out[Tensor[n, bf16]]): ...


SPECS = compile_specs(kernel)
SPEC = SPECS[0]
NOT_VERIFIED = "has not been verified against its declaration.*call spec.verify_against\\(op\\) first"


def test_specs_from_compile_specs_are_verified_and_usable():
    for spec in SPECS:
        assert spec.verified
        inp = spec.ast_source_inputs()
        assert inp.signature == spec.signature and inp.constants == spec.constants


@pytest.mark.parametrize(
    "read",
    [
        lambda spec: CompileSpec.from_dict(spec.to_dict()),
        lambda spec: CompileSpec.from_json(spec.to_json()),
        lambda spec: CompileSpec.from_json(spec.to_json(indent=None)),
    ],
    ids=["from_dict", "from_json", "from_json-compact"],
)
def test_specs_read_back_start_unverified_and_refuse_to_produce_compile_inputs(read):
    spec = read(SPEC)
    assert spec == SPEC  # same content ...
    assert not spec.verified  # ... but nobody checked it against a declaration
    with pytest.raises(TilefishSpecError, match=NOT_VERIFIED):
        spec.ast_source_inputs()


def test_a_successful_verify_against_makes_it_verified():
    spec = CompileSpec.from_json(SPEC.to_json())
    spec.verify_against(kernel)
    assert spec.verified
    assert spec.ast_source_inputs().constants == SPEC.ast_source_inputs().constants


def test_a_failed_verify_against_leaves_it_unverified():
    spec = CompileSpec.from_json(SPEC.to_json())
    with pytest.raises(TilefishSpecError):
        spec.verify_against(other)  # the wrong declaration
    assert not spec.verified
    with pytest.raises(TilefishSpecError, match=NOT_VERIFIED):
        spec.ast_source_inputs()

    # a forged spec stays unverified and cannot compile
    doc = copy.deepcopy(SPEC.to_dict())
    index = {a["name"]: a["index"] for a in doc["args"]}
    doc["constants"][str(index["B"])] = 17
    doc["args"][index["B"]]["value"] = 17
    forged = CompileSpec.from_dict(doc)
    with pytest.raises(TilefishSpecError, match=r"first difference at args\[3\]\.value"):
        forged.verify_against(kernel)
    assert not forged.verified
    with pytest.raises(TilefishSpecError, match=NOT_VERIFIED):
        forged.ast_source_inputs()


def test_a_failed_verification_withdraws_an_earlier_one():
    spec = CompileSpec.from_json(SPEC.to_json())
    spec.verify_against(kernel)
    assert spec.verified
    with pytest.raises(TilefishSpecError):
        spec.verify_against(other)
    assert not spec.verified


def test_the_flag_is_not_serialized_so_it_never_round_trips_as_verified():
    assert SPEC.verified
    doc = SPEC.to_dict()
    assert not any("verif" in key for key in doc) and "_verified" not in doc
    assert "verified" not in SPEC.to_json() and "_verified" not in json.dumps(doc)
    assert not CompileSpec.from_dict(doc).verified
    assert not CompileSpec.from_json(SPEC.to_json()).verified
    # a document that claims to be verified is not accepted at all, let alone trusted
    for claim in ({"verified": True}, {"_verified": "x"}):
        with pytest.raises(TilefishSpecError, match="unknown fields"):
            CompileSpec.from_dict({**doc, **claim})


def test_the_flag_is_not_part_of_equality_or_repr():
    unverified = CompileSpec.from_dict(SPEC.to_dict())
    assert unverified == SPEC and SPEC.verified and not unverified.verified
    assert "verified" not in repr(SPEC)


def test_an_edited_copy_of_a_verified_spec_is_not_verified():
    edited = dataclasses.replace(SPEC, num_stages=5)  # replace() builds a new spec: no flag carried over
    assert not edited.verified
    with pytest.raises(TilefishSpecError, match=NOT_VERIFIED):
        edited.ast_source_inputs()
    same = dataclasses.replace(SPEC)
    assert same == SPEC and not same.verified  # even an identical copy has to be verified on its own
    same.verify_against(kernel)
    assert same.verified


def test_editing_a_verified_spec_in_place_makes_it_unverified_again():
    spec = CompileSpec.from_json(SPEC.to_json())
    spec.verify_against(kernel)
    assert spec.verified
    spec.constants[(3,)] = 17  # the dict inside a frozen dataclass is still mutable
    assert not spec.verified
    with pytest.raises(TilefishSpecError, match=r"was modified after it was verified; call spec.verify_against\(op\) again"):
        spec.ast_source_inputs()
    spec.constants[(3,)] = SPEC.constants[(3,)]  # put it back: it is the verified content again
    assert spec.verified
    spec.ast_source_inputs()

    # the same holds for a spec straight from compile_specs
    fresh = compile_specs(kernel)[1]
    fresh.attrs[(2,)][0][1] = 8
    assert not fresh.verified
    with pytest.raises(TilefishSpecError, match="was modified after it was verified"):
        fresh.ast_source_inputs()
    with pytest.raises(TilefishSpecError):
        fresh.verify_against(kernel)  # and it really is wrong now
    assert not fresh.verified


def test_a_type_only_edit_is_a_modification_too():
    spec = compile_specs(kernel)[0]
    index = {a.name: a.index for a in spec.args}
    spec.constants[(index["B"],)] = 16.0  # equal to 16 in python, a different constant to triton
    assert not spec.verified


def test_the_unverified_message_says_how_to_fix_it():
    spec = CompileSpec.from_json(SPEC.to_json())
    with pytest.raises(TilefishSpecError) as info:
        spec.ast_source_inputs()
    text = str(info.value)
    assert "kernel.S16" in text and "verify_against" in text and "compile_specs(op)" in text


def test_copies_of_a_verified_spec_keep_the_content_they_were_verified_with():
    for other_spec in (copy.copy(SPEC), copy.deepcopy(SPEC), pickle.loads(pickle.dumps(SPEC))):
        assert other_spec == SPEC and other_spec.verified  # same content, same verification
        other_spec.ast_source_inputs()
    unverified = CompileSpec.from_dict(SPEC.to_dict())
    assert not copy.deepcopy(unverified).verified


def test_every_binary_of_a_bucketed_op_is_verified_individually():
    specs = [CompileSpec.from_json(s.to_json()) for s in SPECS]
    assert [s.verified for s in specs] == [False, False]
    specs[1].verify_against(kernel)
    assert [s.verified for s in specs] == [False, True]
    with pytest.raises(TilefishSpecError, match=NOT_VERIFIED):
        specs[0].ast_source_inputs()
    specs[1].ast_source_inputs()
