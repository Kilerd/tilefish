import pytest

from tilefish import Dim, DimExpr, TilefishDeclarationError, TilefishTypeError, cdiv


def test_dynamic_static_and_equality():
    M = Dim("M")
    D = Dim("D", 2560)
    assert M.is_dynamic and not M.is_static
    assert D.is_static and D.value == 2560
    assert D == 2560 and hash(D) == hash(2560)
    assert M == Dim("M")
    assert M != Dim("N")
    assert M != Dim("M", divisible_by=16)  # a different dimension: different constraint
    assert Dim("D", 2560) == Dim("E", 2560)  # static dims are values
    assert repr(M) == "Dim('M')"
    assert repr(Dim("K", divisible_by=32)) == "Dim('K', divisible_by=32)"
    assert repr(Dim("S", buckets=[64, 32])) == "Dim('S', buckets=[32, 64])"
    assert str(D) == "D"


def test_symbolic_dim_has_no_value():
    with pytest.raises(TilefishTypeError, match="symbolic"):
        Dim("M").value


def test_affine_normalization():
    F = Dim("F")
    D = Dim("D")
    assert 2 * F == F + F
    assert F * 2 == F + F
    assert (F + F) - F == F
    assert F - F == 0
    assert (D + F) - F == D
    assert 2 * F + D == D + F + F
    assert hash(2 * F) == hash(F + F)
    assert (3 * F + 6).terms == ((F, 3),)
    assert (3 * F + 6).offset == 6


def test_static_dims_fold_into_arithmetic():
    F = Dim("F", 3072)
    assert 2 * F == 6144
    assert (2 * F).is_static and (2 * F).value == 6144
    assert F + Dim("G", 8) == 3080
    n = Dim("n")
    assert (n + F).offset == 3072


def test_rendering():
    F, D = Dim("F"), Dim("D")
    assert str(2 * F) == "2*F"
    assert str(D + 2 * F) == "D + 2*F"
    assert str(D - F) == "D - F"
    assert str(-F) == "-F"
    assert str(F + 3) == "F + 3"
    assert str(F - 3) == "F - 3"
    assert str(DimExpr.const(7)) == "7"


def test_arithmetic_is_affine():
    M, N = Dim("M"), Dim("N")
    with pytest.raises(TilefishTypeError, match="affine"):
        M * N
    with pytest.raises(TilefishTypeError, match="not provably exact"):
        M // 2


def test_floordiv_exact():
    F = Dim("F")
    assert (4 * F) // 2 == 2 * F
    assert Dim("D", 100) // 8 == 12


def test_cdiv_on_ints_and_static_dims():
    assert cdiv(10, 3) == 4
    assert cdiv(9, 3) == 3
    assert isinstance(cdiv(10, 3), int)
    assert cdiv(Dim("D", 2560), 1024) == 3
    assert cdiv(Dim("D", 2560), Dim("B", 512)) == 5


def test_cdiv_symbolic_node():
    n = Dim("n")
    e = cdiv(n, 1024)
    assert str(e) == "cdiv(n, 1024)"
    assert not e.is_static
    assert e == cdiv(n, 1024)
    assert e != cdiv(n, 512)
    assert e.free_dims() == {n}
    assert str(2 * e + 1) == "2*cdiv(n, 1024) + 1"
    assert cdiv(n, 1) == n


def test_cdiv_folds_exact_quotients():
    n = Dim("n")
    assert cdiv(2 * n, 2) == n
    assert cdiv(4 * n + 8, 4) == n + 2
    assert cdiv(2 * n, n) == 2


def test_cdiv_rejects_non_positive_divisor():
    with pytest.raises(TilefishTypeError):
        cdiv(10, 0)
    with pytest.raises(TilefishTypeError):
        cdiv(Dim("n"), 0)


def test_subst_is_simultaneous():
    M, N = Dim("M"), Dim("N")
    e = M + 2 * N
    assert e.subst({M: N, N: M}) == N + 2 * M
    assert e.subst({M: 4}) == 4 + 2 * N
    assert cdiv(M, 8).subst({M: 64}) == 8
    assert cdiv(M, 8).subst({M: N}) == cdiv(N, 8)


def test_divisibility_reasoning():
    K = Dim("K", divisible_by=32)
    n = Dim("n")
    assert K.is_divisible_by(32) and K.is_divisible_by(16) and not K.is_divisible_by(64)
    assert (2 * K).is_divisible_by(64)
    assert (K + 64).is_divisible_by(32)
    assert not (K + 1).is_divisible_by(32)
    assert (4 * n).is_divisible_by(4)
    assert not n.is_divisible_by(2)
    assert Dim("D", 2560).is_divisible_by(128)
    assert n.is_divisible_by(8, known={n: 16})


def test_exact_div():
    F = Dim("F")
    assert (2 * F).exact_div(F) == 2
    assert DimExpr.const(6144).exact_div(Dim("G", 3072)) == 2
    assert (2 * F + 1).exact_div(F) is None
    assert F.exact_div(2 * F) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "1bad"},
        {"name": "M", "value": 0},
        {"name": "M", "value": -4},
        {"name": "M", "divisible_by": 0},
        {"name": "M", "value": 10, "divisible_by": 4},
        {"name": "M", "value": 8, "buckets": [8]},
        {"name": "M", "buckets": []},
        {"name": "M", "buckets": [8, 8]},
        {"name": "M", "buckets": [8, 12], "divisible_by": 8},
    ],
)
def test_invalid_dims(kwargs):
    with pytest.raises(TilefishDeclarationError):
        Dim(**kwargs)


def test_dims_are_immutable():
    with pytest.raises(AttributeError):
        Dim("M").name = "N"
