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


# -- divisibility reasoning, in detail ----------------------------------------------------------------


def test_divisibility_needs_every_term_not_just_one():
    K = Dim("K", divisible_by=32)
    n = Dim("n")
    L16 = Dim("L16", divisible_by=16)
    L32 = Dim("L32", divisible_by=32)
    assert not (K + n).is_divisible_by(32)  # one divisible term is not enough
    assert not (n + K).is_divisible_by(32)
    assert not (K + L16).is_divisible_by(32)
    assert (K + L32).is_divisible_by(32)
    assert (K + 2 * L16).is_divisible_by(32)
    assert (K - L32).is_divisible_by(32)
    assert not (K + L32 + n).is_divisible_by(32)


def test_known_divisors_extend_what_is_provable():
    n = Dim("n")
    m = Dim("m", divisible_by=4)
    assert n.is_divisible_by(16, known={n: 16})
    assert n.is_divisible_by(8, known={n: 16})
    assert not n.is_divisible_by(32, known={n: 16})
    assert n.is_divisible_by(16, known={n: 64})  # 64 is a multiple of 16
    assert not n.is_divisible_by(64, known={n: 16})
    # inside a sum the known divisor of an atom counts, scaled by its coefficient
    assert (2 * n).is_divisible_by(32, known={n: 16})
    assert not (2 * n).is_divisible_by(64, known={n: 16})
    assert (n + 32).is_divisible_by(16, known={n: 16})
    # a declared divisor and a known one combine (lcm)
    assert m.is_divisible_by(12, known={m: 6})
    assert not m.is_divisible_by(24, known={m: 6})
    c = cdiv(n, 4)
    assert (3 * c).is_divisible_by(24, known={c: 8})
    assert not (3 * c).is_divisible_by(24)
    # facts about a whole expression apply to that expression
    assert (n + 1).is_divisible_by(8, known={n + 1: 16})


def test_exact_div_requires_one_consistent_ratio():
    F, G = Dim("F"), Dim("G")
    assert (2 * F + 3 * G).exact_div(F + G) is None  # 2 and 3 disagree
    assert (6 * F + 6 * G).exact_div(2 * F + 3 * G) is None  # 3 and 2 disagree
    assert (2 * F + 5).exact_div(F + 1) is None  # 2 and 5 disagree
    assert (2 * F + 2).exact_div(F + 1) == 2
    assert (F + 1).exact_div(F + 1) == 1
    assert (2 * F).exact_div(3 * F) is None
    assert (6 * F).exact_div(3 * F) == 2
    assert DimExpr.const(7).exact_div(0) is None  # no zero divisor
    assert DimExpr.const(0).exact_div(5) == 0
    assert (2 * F + 3).exact_div(F) is None


# -- static dims fold to values for hosts -----------------------------------------------------------------


def test_folded_replaces_a_static_dim_by_its_value():
    NH = Dim("NH", 32)
    folded = NH.folded()
    assert type(folded) is DimExpr and str(folded) == "32" and folded == NH
    S = Dim("S")
    assert S.folded() is S
    assert (2 * S + 1).folded() == 2 * S + 1


def test_cdiv_treats_static_dims_as_their_values():
    S, NH = Dim("S"), Dim("NH", 32)
    assert str(cdiv(S, NH)) == "cdiv(S, 32)"
    assert str(cdiv(NH, S)) == "cdiv(32, S)"
    assert cdiv(Dim("D", 2560), Dim("B", 128)) == 20


# -- operators outside the affine language ----------------------------------------------------------------


@pytest.mark.parametrize(
    "use",
    [
        lambda M: M / 2,
        lambda M: 2 / M,
        lambda M: M % 2,
        lambda M: 2 % M,
        lambda M: M**2,
        lambda M: 2**M,
        lambda M: abs(M),
        lambda M: M < 4,
        lambda M: M >= 4,
        lambda M: float(M),
        lambda M: range(M),
        lambda M: [0, 1][M],
        lambda M: 3 // M,
        lambda M: M + 0.5,
        lambda M: M - "a",
        lambda M: M * None,
        lambda M: 0.5 * M,
    ],
)
def test_unsupported_dimension_operators_raise_tilefish_errors(use):
    with pytest.raises(TilefishTypeError):
        use(Dim("M"))


def test_static_dimensions_can_be_used_as_numbers():
    D = Dim("D", 3)
    assert list(range(D)) == [0, 1, 2]
    assert D % 2 == 1 and D < 4 and D >= 3 and not D > 3
    with pytest.raises(TilefishTypeError, match="true division"):
        D / 2


def test_dimension_error_messages_say_what_to_do():
    M = Dim("M")
    with pytest.raises(TilefishTypeError, match=r"use // for an exact division or tilefish\.cdiv"):
        M / 2
    with pytest.raises(TilefishTypeError, match="symbolic and cannot be used as an integer"):
        range(M)


def test_bucketed_dims_need_a_list_of_ints():
    for bad in (3, "32", None.__class__):
        with pytest.raises(TilefishDeclarationError, match="buckets must be a list of ints"):
            Dim("a", buckets=bad)
    assert Dim("a", buckets=(32, 64)).buckets == (32, 64)
    assert Dim("a", buckets=iter([64, 32])).buckets == (32, 64)


def test_constraints_are_spelled_out_when_names_collide():
    from tilefish.dims import describe_pair

    plain, constrained = Dim("K"), Dim("K", divisible_by=32)
    assert describe_pair(plain, constrained) == ("Dim('K')", "Dim('K', divisible_by=32)")
    assert describe_pair(Dim("M"), Dim("N")) == ("M", "N")
    assert describe_pair(2 * plain, 2 * constrained) == ("2*K with Dim('K')", "2*K with Dim('K', divisible_by=32)")
