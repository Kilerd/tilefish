"""``next_pow2``: the dimension a lowered kernel's bucketed ``BLOCK_D`` constexpr is derived with."""

import pickle

import pytest

from tilefish import Dim, TilefishTypeError
from tilefish.dims import next_pow2


def test_next_pow2_of_ints_and_static_dims():
    assert [next_pow2(i) for i in (1, 2, 3, 4, 5, 100, 1000, 1024, 1025, 2560)] == [1, 2, 4, 4, 8, 128, 1024, 1024, 2048, 4096]
    assert next_pow2(Dim("S", 100)) == 128
    assert next_pow2(2 * Dim("M")).is_static is False
    for bad in (0, -4):
        with pytest.raises(TilefishTypeError, match="positive int"):
            next_pow2(bad)


def test_next_pow2_of_a_bucketed_dim_is_an_ordinary_dimension_expression():
    H = Dim("H", buckets=[100, 300])
    e = next_pow2(H)
    assert str(e) == "next_pow2(H)" and not e.is_static
    assert e.free_dims() == frozenset({H}) and e.named_dims() == {"H": H}
    assert e.subst({H: 100}) == 128 and e.subst({H: 300}) == 512
    assert (2 * e + 1).subst({H: 100}) == 257  # it combines with the affine language like any atom
    assert e == next_pow2(H) and hash(e) == hash(next_pow2(H)) and e != next_pow2(2 * H) and e != H
    assert pickle.loads(pickle.dumps(e)) == e
    assert next_pow2(2 * H).subst({H: 96}) == 256
