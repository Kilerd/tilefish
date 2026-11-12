"""Reordering what a compile spec carries in order must unverify it.

``ast_source_inputs()`` hands ``signature`` to Triton in its order while ``constants`` and ``attrs`` are
keyed by argument *index*, so a reordered signature would silently mis-specialize. The fingerprint that
backs ``verified`` therefore keeps every order-carrying field as an ordered list.
"""

import pytest

import tilefish
from tilefish import Dim, Out, Tensor, TilefishSpecError, bf16, const
from tilefish.aot import CompileSpec, compile_specs

S = Dim("S", buckets=[32, 64])
B = Dim("B", buckets=[1, 2])


@tilefish.extern(triton="my_kernels:k", grid=lambda S, B: (S, B), num_warps=4)
def k(x: Tensor[S, B, bf16], y: Out[Tensor[S, B, bf16]], NS: const = S, NB: const = B): ...


@pytest.fixture
def spec():
    s = compile_specs(k)[0]
    assert s.verified and s.grid == "(32, 1)" and s.buckets == {"S": 32, "B": 1}
    return s


def reorder_dict_in_place(d):
    items = list(d.items())
    d.clear()
    d.update(reversed(items))


def assert_unverified(spec):
    assert not spec.verified
    with pytest.raises(TilefishSpecError, match="modified after it was verified"):
        spec.ast_source_inputs()


def test_a_reversed_signature_is_no_longer_verified(spec):
    before = list(spec.signature)
    reorder_dict_in_place(spec.signature)
    assert list(spec.signature) == before[::-1]
    assert_unverified(spec)


def test_a_swapped_pair_in_the_signature_is_no_longer_verified(spec):
    items = list(spec.signature.items())
    items[0], items[1] = items[1], items[0]
    spec.signature.clear()
    spec.signature.update(items)
    assert_unverified(spec)


def test_reordered_args_are_no_longer_verified(spec):
    object.__setattr__(spec, "args", spec.args[::-1])
    assert_unverified(spec)


def test_reordered_launch_params_are_no_longer_verified(spec):
    object.__setattr__(spec, "launch_params", spec.launch_params[::-1])
    assert_unverified(spec)


def test_reordered_grid_entries_are_no_longer_verified(spec):
    object.__setattr__(spec, "grid", "(1, 32)")
    assert_unverified(spec)


def test_reordered_preconditions_are_no_longer_verified(spec):
    object.__setattr__(spec, "preconditions", spec.preconditions[::-1])
    assert_unverified(spec)


def test_reordered_buckets_are_no_longer_verified(spec):
    reorder_dict_in_place(spec.buckets)
    assert list(spec.buckets) == ["B", "S"]
    assert_unverified(spec)


def test_an_untouched_spec_stays_verified_and_a_regenerated_one_is_verified_again(spec):
    assert spec.verified and spec.ast_source_inputs().signature == dict(spec.signature)
    reorder_dict_in_place(spec.signature)
    assert not spec.verified
    again = CompileSpec.from_json(compile_specs(k)[0].to_json())
    again.verify_against(k)
    assert again.verified
    reorder_dict_in_place(again.signature)
    assert_unverified(again)
