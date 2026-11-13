"""Dimensions: named symbols, static sizes and affine expressions over them.

A dimension expression is ``const + sum(coef_i * atom_i)`` with integer
coefficients, normalized so that structurally equal expressions compare equal
(``2 * F == F + F``). The atoms are

* a symbolic :class:`Dim` (dynamic, divisible-by, or bucketed), and
* a ceiling division ``cdiv(a, b)`` that could not be folded.

A static :class:`Dim` (``Dim("D", 2560)``) is just a named constant: it
participates in arithmetic as its value and compares equal to that value.
Anything that is rendered for a host (grid expressions, shapes in a manifest)
uses :meth:`DimExpr.folded`, which replaces a static ``Dim`` by its value.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from .errors import TilefishDeclarationError, TilefishTypeError


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def _is_tf_operand(x: Any) -> bool:
    """Traced tensors and consts mark themselves so a dimension defers to their reflected operators."""
    return bool(getattr(type(x), "_tf_operand", False))


class DimExpr:
    """An affine dimension expression (immutable)."""

    __slots__ = ("_terms", "_const")

    _terms: "tuple[tuple[DimExpr, int], ...]"
    _const: int

    # -- construction -------------------------------------------------------

    @staticmethod
    def _make(terms: "Mapping[DimExpr, int]", const: int) -> "DimExpr":
        items = sorted(((a, c) for a, c in terms.items() if c != 0), key=lambda ac: ac[0]._atom_key())
        if const == 0 and len(items) == 1 and items[0][1] == 1:
            return items[0][0]  # a bare atom stays the atom (keeps its name and metadata)
        self = object.__new__(DimExpr)
        object.__setattr__(self, "_terms", tuple(items))
        object.__setattr__(self, "_const", const)
        return self

    @staticmethod
    def const(value: int) -> "DimExpr":
        """A static dimension with the given value."""
        if not _is_int(value):
            raise TilefishTypeError(f"a static dimension must be an int, got {value!r}")
        return DimExpr._make({}, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __reduce__(self) -> Any:
        return (_rebuild_expr, (self._terms, self._const))

    # -- inspection ---------------------------------------------------------

    @property
    def is_static(self) -> bool:
        return not self._terms

    @property
    def value(self) -> int:
        """The integer value of a static dimension."""
        if self._terms:
            raise TilefishTypeError(f"dimension {self} is symbolic and has no static value")
        return self._const

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        if self._terms:
            raise TilefishTypeError(
                f"dimension {self} is symbolic and cannot be used as an integer (range(), indexing, ...)"
            )
        return self._const

    def folded(self) -> "DimExpr":
        """The same dimension with a static ``Dim`` replaced by its plain value (names do not leak to hosts)."""
        if not self._terms and type(self) is not DimExpr:
            return DimExpr.const(self._const)
        return self

    @property
    def terms(self) -> "tuple[tuple[DimExpr, int], ...]":
        """``(atom, coefficient)`` pairs in a canonical order."""
        return self._terms

    @property
    def offset(self) -> int:
        """The constant part."""
        return self._const

    @property
    def _key(self) -> tuple:
        return (tuple((a._atom_key(), c) for a, c in self._terms), self._const)

    def _atom_key(self) -> tuple:  # only meaningful on atoms
        raise TilefishTypeError(f"{self} is not an atom")

    def _atom_divisor(self, known: "Mapping[DimExpr, int] | None") -> int:  # only on atoms
        raise TilefishTypeError(f"{self} is not an atom")

    def _subst_atom(self, mapping: "Mapping[Dim, Any]") -> "DimExpr":  # only on non-Dim atoms
        raise TilefishTypeError(f"{self} is not an atom")

    @property
    def is_atom(self) -> bool:
        return self._const == 0 and len(self._terms) == 1 and self._terms[0][1] == 1 and self._terms[0][0] is self

    def coeff(self, atom: "DimExpr") -> int:
        """Coefficient of ``atom`` (0 when it does not occur)."""
        for a, c in self._terms:
            if a == atom:
                return c
        return 0

    def free_dims(self) -> "frozenset[Dim]":
        """Every symbolic :class:`Dim` this expression mentions."""
        out: set[Dim] = set()
        for atom, _ in self._terms:
            if isinstance(atom, Dim):
                out.add(atom)
            elif isinstance(atom, _CDiv):
                out |= atom.num.free_dims() | atom.den.free_dims()
            elif isinstance(atom, _NextPow2):
                out |= atom.arg.free_dims()
        return frozenset(out)

    def named_dims(self) -> "dict[str, Dim]":
        """Named dimensions mentioned, including this expression itself when it is a static :class:`Dim`."""
        out: dict[str, Dim] = {}
        if isinstance(self, Dim):
            out[self.name] = self
        for d in self.free_dims():
            out[d.name] = d
        return out

    def subst(self, mapping: "Mapping[Dim, Any]") -> "DimExpr":
        """Replace symbolic dims by dims/ints, all at once (the result is never re-substituted)."""
        if not self._terms:
            return self  # static: nothing to replace (keeps the name of a static Dim)
        acc = DimExpr.const(self._const)
        for atom, coef in self._terms:
            if isinstance(atom, Dim):
                repl = mapping.get(atom, atom)
            else:
                repl = atom._subst_atom(mapping)
            if coef == 1 and not self._const and len(self._terms) == 1:
                return as_dim(repl)  # a lone atom becomes its replacement as is (a static Dim keeps its name)
            acc = acc + as_dim(repl) * coef
        return acc

    def exact_div(self, other: "DimExpr | int") -> "int | None":
        """The integer ``k`` with ``self == k * other``, or ``None``."""
        other = as_dim(other)
        if not other._terms and other._const == 0:
            return None
        ratio: int | None = None
        pairs: list[tuple[int, int]] = []
        if other._const or self._const:
            pairs.append((self._const, other._const))
        atoms = {a for a, _ in self._terms} | {a for a, _ in other._terms}
        for a in atoms:
            pairs.append((self.coeff(a), other.coeff(a)))
        for mine, theirs in pairs:
            if theirs == 0:
                if mine != 0:
                    return None
                continue
            if mine % theirs:
                return None
            k = mine // theirs
            if ratio is None:
                ratio = k
            elif ratio != k:
                return None
        return ratio

    def _div_exact_int(self, k: int) -> "DimExpr | None":
        if self._const % k or any(c % k for _, c in self._terms):
            return None
        return DimExpr._make({a: c // k for a, c in self._terms}, self._const // k)

    def is_divisible_by(self, k: int, known: "Mapping[DimExpr, int] | None" = None) -> bool:
        """True when the expression is provably a multiple of ``k``.

        ``known`` maps expressions to extra divisors established elsewhere
        (for example by a requirement recorded while tracing).
        """
        if k == 1:
            return True
        if self.is_static:
            return self._const % k == 0
        if known:
            extra = known.get(self)
            if extra and extra % k == 0:
                return True
        if self._const % k:
            return False
        return all((coef * atom._atom_divisor(known)) % k == 0 for atom, coef in self._terms)

    # -- arithmetic ---------------------------------------------------------

    def _combine(self, other: "DimExpr", sign: int) -> "DimExpr":
        terms = dict(self._terms)
        for atom, coef in other._terms:
            terms[atom] = terms.get(atom, 0) + sign * coef
        return DimExpr._make(terms, self._const + sign * other._const)

    def _bad_operand(self, symbol: str, other: Any) -> Any:
        if _is_tf_operand(other):
            return NotImplemented  # let the tensor's reflected operator handle it
        raise TilefishTypeError(
            f"cannot apply '{symbol}' to dimension {self} and {type(other).__name__} {other!r}: "
            "dimensions combine with ints and other dimensions"
        )

    def __add__(self, other: Any) -> "DimExpr":
        o = _coerce(other)
        return self._bad_operand("+", other) if o is None else self._combine(o, 1)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "DimExpr":
        o = _coerce(other)
        return self._bad_operand("-", other) if o is None else self._combine(o, -1)

    def __rsub__(self, other: Any) -> "DimExpr":
        o = _coerce(other)
        return self._bad_operand("-", other) if o is None else o._combine(self, -1)

    def __neg__(self) -> "DimExpr":
        return DimExpr._make({a: -c for a, c in self._terms}, -self._const)

    def __pos__(self) -> "DimExpr":
        return self

    def __mul__(self, other: Any) -> "DimExpr":
        o = _coerce(other)
        if o is None:
            return self._bad_operand("*", other)
        if o.is_static:
            k = o._const
            return DimExpr._make({a: c * k for a, c in self._terms}, self._const * k)
        if self.is_static:
            return o * self._const
        raise TilefishTypeError(
            f"cannot multiply dimensions {self} and {o}: dimension arithmetic is affine "
            "(a symbolic dimension can only be scaled by an integer)"
        )

    __rmul__ = __mul__

    def __floordiv__(self, other: Any) -> "DimExpr":
        o = _coerce(other)
        if o is None:
            return self._bad_operand("//", other)
        if not o.is_static or o._const <= 0:
            raise TilefishTypeError(f"cannot divide {self} by {o}: the divisor must be a positive static int")
        if self.is_static:
            return DimExpr.const(self._const // o._const)
        q = self._div_exact_int(o._const)
        if q is None:
            raise TilefishTypeError(
                f"{self} // {o} is not an affine expression (not provably exact); use cdiv() instead"
            )
        return q

    # Operators that are not part of the affine language: a clear error instead of a bare TypeError.

    def _unsupported(self, what: str, hint: str) -> Any:
        raise TilefishTypeError(f"{what} on dimension {self} is not supported in M1; {hint}")

    def __truediv__(self, other: Any) -> Any:
        return self._unsupported("true division (/)", "use // for an exact division or tilefish.cdiv(a, b)")

    def __rtruediv__(self, other: Any) -> Any:
        return self._unsupported("true division (/)", "use // for an exact division or tilefish.cdiv(a, b)")

    def __rfloordiv__(self, other: Any) -> Any:
        return self._unsupported("floor division by a dimension (//)", "dimension arithmetic is affine")

    def __mod__(self, other: Any) -> Any:
        if self.is_static and _coerce(other) is not None and _coerce(other).is_static:  # type: ignore[union-attr]
            return DimExpr.const(self._const % _coerce(other)._const)  # type: ignore[union-attr]
        return self._unsupported("modulo (%)", "ask is_divisible_by(k) instead")

    def __rmod__(self, other: Any) -> Any:
        return self._unsupported("modulo (%)", "dimension arithmetic is affine")

    def __pow__(self, other: Any) -> Any:
        return self._unsupported("exponentiation (**)", "dimension arithmetic is affine")

    def __rpow__(self, other: Any) -> Any:
        return self._unsupported("exponentiation (**)", "dimension arithmetic is affine")

    def __abs__(self) -> Any:
        return self._unsupported("abs()", "dimension arithmetic is affine")

    def __float__(self) -> Any:
        return self._unsupported("float()", "use .value on a static dimension")

    def _compare(self, other: Any, symbol: str, test: Any) -> Any:
        o = _coerce(other)
        if o is not None and self.is_static and o.is_static:
            return test(self._const, o._const)
        return self._unsupported(f"comparison ({symbol})", "only static dimensions can be ordered; use == for equality")

    def __lt__(self, other: Any) -> Any:
        return self._compare(other, "<", lambda a, b: a < b)

    def __le__(self, other: Any) -> Any:
        return self._compare(other, "<=", lambda a, b: a <= b)

    def __gt__(self, other: Any) -> Any:
        return self._compare(other, ">", lambda a, b: a > b)

    def __ge__(self, other: Any) -> Any:
        return self._compare(other, ">=", lambda a, b: a >= b)

    # -- comparison / hashing ----------------------------------------------

    def __eq__(self, other: Any) -> bool:
        if _is_int(other):
            return self.is_static and self._const == other
        if isinstance(other, DimExpr):
            return self._key == other._key
        return NotImplemented

    def __hash__(self) -> int:
        if self.is_static:
            return hash(self._const)
        return hash(self._key)

    # -- rendering ----------------------------------------------------------

    def __str__(self) -> str:
        if self.is_static:
            return str(self._const)
        out = ""
        for i, (atom, coef) in enumerate(self._terms):
            body = str(atom)
            mag = abs(coef)
            if mag != 1:
                body = f"{mag}*{body}"
            if i == 0:
                out = ("-" if coef < 0 else "") + body
            else:
                out += (" - " if coef < 0 else " + ") + body
        if self._const:
            out += (" - " if self._const < 0 else " + ") + str(abs(self._const))
        return out

    def __repr__(self) -> str:
        return f"DimExpr({self})"


def _rebuild_expr(terms: "tuple[tuple[DimExpr, int], ...]", const: int) -> DimExpr:
    return DimExpr._make(dict(terms), const)


class _CDiv(DimExpr):
    """Ceiling division that could not be folded to an affine expression."""

    __slots__ = ("num", "den")

    def __init__(self, num: DimExpr, den: DimExpr) -> None:
        object.__setattr__(self, "num", num)
        object.__setattr__(self, "den", den)
        object.__setattr__(self, "_terms", ((self, 1),))
        object.__setattr__(self, "_const", 0)

    def __reduce__(self) -> Any:
        return (_CDiv, (self.num, self.den))

    def _atom_key(self) -> tuple:
        return ("cdiv", self.num._key, self.den._key)

    def _atom_divisor(self, known: "Mapping[DimExpr, int] | None") -> int:
        return (known or {}).get(self, 1)

    def _subst_atom(self, mapping: "Mapping[Dim, Any]") -> DimExpr:
        return as_dim(cdiv(self.num.subst(mapping), self.den.subst(mapping)))

    def __str__(self) -> str:
        return f"cdiv({self.num}, {self.den})"


class _NextPow2(DimExpr):
    """The smallest power of two that is at least ``arg``, when ``arg`` is not static.

    It exists so a lowered kernel's ``BLOCK`` constexpr can be derived from a bucketed
    dimension: a bucket spec substitutes the bucket and gets a static value (see ``next_pow2``).
    """

    __slots__ = ("arg",)

    def __init__(self, arg: DimExpr) -> None:
        object.__setattr__(self, "arg", arg)
        object.__setattr__(self, "_terms", ((self, 1),))
        object.__setattr__(self, "_const", 0)

    def __reduce__(self) -> Any:
        return (_NextPow2, (self.arg,))

    def _atom_key(self) -> tuple:
        return ("next_pow2", self.arg._key)

    def _atom_divisor(self, known: "Mapping[DimExpr, int] | None") -> int:
        return (known or {}).get(self, 1)

    def _subst_atom(self, mapping: "Mapping[Dim, Any]") -> DimExpr:
        return as_dim(next_pow2(self.arg.subst(mapping)))

    def __str__(self) -> str:
        return f"next_pow2({self.arg})"


class Dim(DimExpr):
    """A named dimension.

    * ``Dim("M")``: dynamic (a runtime value).
    * ``Dim("D", 2560)``: static, baked into the binary.
    * ``Dim("K", divisible_by=32)``: dynamic with a divisibility constraint.
    * ``Dim("S", buckets=[32, 64, 128])``: dynamic, one binary per bucket.

    In an operator signature a dynamic ``Dim`` is a type variable bound afresh
    at every call; a static ``Dim`` is a literal. In a ``graph`` signature the
    dynamic dims are the graph's symbolic inputs. Two dynamic dims are the
    same dimension when name, ``divisible_by`` and ``buckets`` all agree.

    **Bucket semantics are exact.** A binary built for bucket ``b`` is
    compiled for, and must be launched with, a dimension equal to exactly
    ``b``: the host picks the smallest bucket that is at least the real size
    and pads its buffers up to it (padding is the host's job, not tilefish's).
    In an ahead-of-time spec the bucketed dim is replaced by ``b`` in the grid,
    in the shapes and in every constant derived from it, and the spec carries a
    ``bucket`` precondition ``dim == b``.
    """

    __slots__ = ("name", "divisible_by", "buckets")

    def __init__(
        self,
        name: str,
        value: int | None = None,
        *,
        divisible_by: int | None = None,
        buckets: Iterable[int] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.isidentifier():
            raise TilefishDeclarationError(f"Dim name must be a valid identifier, got {name!r}")
        if value is not None and (not _is_int(value) or value < 1):
            raise TilefishDeclarationError(f"Dim {name!r}: value must be a positive int, got {value!r}")
        if divisible_by is not None and (not _is_int(divisible_by) or divisible_by < 1):
            raise TilefishDeclarationError(
                f"Dim {name!r}: divisible_by must be a positive int, got {divisible_by!r}"
            )
        bucket_tuple: tuple[int, ...] | None = None
        if buckets is not None:
            if value is not None:
                raise TilefishDeclarationError(f"Dim {name!r}: a static dim cannot also have buckets")
            if isinstance(buckets, (str, bytes)):
                raise TilefishDeclarationError(f"Dim {name!r}: buckets must be a list of ints, got {buckets!r}")
            try:
                bucket_tuple = tuple(buckets)
            except TypeError:
                raise TilefishDeclarationError(
                    f"Dim {name!r}: buckets must be a list of ints such as [32, 64, 128], got {buckets!r}"
                ) from None
            if not bucket_tuple:
                raise TilefishDeclarationError(f"Dim {name!r}: buckets must not be empty")
            if any(not _is_int(b) or b < 1 for b in bucket_tuple):
                raise TilefishDeclarationError(
                    f"Dim {name!r}: bucket values must be positive ints, got {list(bucket_tuple)}"
                )
            if len(set(bucket_tuple)) != len(bucket_tuple):
                raise TilefishDeclarationError(f"Dim {name!r}: duplicate bucket values in {list(bucket_tuple)}")
            bucket_tuple = tuple(sorted(bucket_tuple))
        if divisible_by is not None:
            if value is not None and value % divisible_by:
                raise TilefishDeclarationError(
                    f"Dim {name!r}: static value {value} is not divisible by {divisible_by}"
                )
            if bucket_tuple is not None:
                bad = [b for b in bucket_tuple if b % divisible_by]
                if bad:
                    raise TilefishDeclarationError(
                        f"Dim {name!r}: bucket values {bad} are not divisible by {divisible_by}"
                    )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "divisible_by", divisible_by)
        object.__setattr__(self, "buckets", bucket_tuple)
        if value is not None:
            object.__setattr__(self, "_terms", ())
            object.__setattr__(self, "_const", value)
        else:
            object.__setattr__(self, "_terms", ((self, 1),))
            object.__setattr__(self, "_const", 0)

    def __reduce__(self) -> Any:
        return (_rebuild_dim, (self.name, self._const if self.is_static else None, self.divisible_by, self.buckets))

    def _atom_key(self) -> tuple:
        return ("dim", self.name, self.divisible_by or 0, self.buckets or ())

    def _atom_divisor(self, known: "Mapping[DimExpr, int] | None") -> int:
        return _lcm(self.divisible_by or 1, (known or {}).get(self, 1))

    @property
    def is_dynamic(self) -> bool:
        return not self.is_static

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        parts = [repr(self.name)]
        if self.is_static:
            parts.append(str(self._const))
        if self.divisible_by is not None:
            parts.append(f"divisible_by={self.divisible_by}")
        if self.buckets is not None:
            parts.append(f"buckets={list(self.buckets)}")
        return f"Dim({', '.join(parts)})"


def _rebuild_dim(name: str, value: "int | None", divisible_by: "int | None", buckets: "tuple[int, ...] | None") -> Dim:
    return Dim(name, value, divisible_by=divisible_by, buckets=buckets)


def _coerce(x: Any) -> DimExpr | None:
    if isinstance(x, DimExpr):
        return x
    if _is_int(x):
        return DimExpr.const(x)
    return None


def as_dim(x: Any) -> DimExpr:
    """Coerce an int or dimension expression to a :class:`DimExpr`."""
    d = _coerce(x)
    if d is None:
        raise TilefishTypeError(f"expected a dimension (int, Dim or dimension expression), got {x!r}")
    return d


def cdiv(a: Any, b: Any) -> Any:
    """Ceiling division of ints or dimension expressions.

    Ints give an int. With dimension expressions the result folds to a static
    value or an exact affine quotient when it can, and is a symbolic
    ``cdiv(a, b)`` node otherwise. A static ``Dim`` operand counts as its value.
    """
    if _is_int(a) and _is_int(b):
        if b <= 0:
            raise TilefishTypeError(f"cdiv: divisor must be positive, got {b}")
        return -(-a // b)
    da, db = as_dim(a).folded(), as_dim(b).folded()
    if db.is_static:
        if db._const <= 0:
            raise TilefishTypeError(f"cdiv: divisor must be positive, got {db}")
        if db._const == 1:
            return da
        if da.is_static:
            return DimExpr.const(-(-da._const // db._const))
        exact = da._div_exact_int(db._const)
        if exact is not None:
            return exact
    else:
        k = da.exact_div(db)
        if k is not None:
            return DimExpr.const(k)
    return _CDiv(da, db)


def next_pow2(x: Any) -> Any:
    """The smallest power of two that is at least ``x`` (an int gives an int; a symbolic dimension stays symbolic)."""
    if _is_int(x):
        if x < 1:
            raise TilefishTypeError(f"next_pow2: expected a positive int, got {x}")
        return 1 << (x - 1).bit_length()
    d = as_dim(x).folded()
    if d.is_static:
        return DimExpr.const(next_pow2(d.value))
    return _NextPow2(d)


def _atom_bounds(atom: DimExpr) -> "tuple[int, int | None]":
    """Smallest and largest value of a symbolic dimension (``None``: no upper bound). Dimensions are positive."""
    if isinstance(atom, Dim):
        if atom.buckets:
            return min(atom.buckets), max(atom.buckets)  # a binary exists for these sizes only
        return (atom.divisible_by or 1), None
    num, den = getattr(atom, "num", None), getattr(atom, "den", None)
    if num is None or den is None:  # pragma: no cover - the other atoms never appear in a shape
        return 1, None
    n_lo, n_hi = dim_bounds(num)
    d_lo, d_hi = dim_bounds(den)
    lo = max(1, -(-n_lo // d_hi)) if d_hi else 1
    return lo, (None if n_hi is None else -(-n_hi // max(d_lo, 1)))  # cdiv


def dim_bounds(d: DimExpr) -> "tuple[int, int | None]":
    """Interval of an affine dimension expression: ``(lowest, highest or None)``. Every dimension is at least 1."""
    d = d.folded()
    lo: "int | None" = d.offset
    hi: "int | None" = d.offset
    for atom, coef in d.terms:
        a_lo, a_hi = _atom_bounds(atom)
        if coef > 0:
            lo_t, hi_t = coef * a_lo, None if a_hi is None else coef * a_hi
        else:
            lo_t, hi_t = (None if a_hi is None else coef * a_hi), coef * a_lo
        lo = None if lo is None or lo_t is None else lo + lo_t
        hi = None if hi is None or hi_t is None else hi + hi_t
    return (1 if lo is None else max(1, lo)), hi


def dim_text(d: DimExpr) -> str:
    """Readable text for a dimension; a constrained symbolic ``Dim`` shows its constraints."""
    if isinstance(d, Dim) and d.is_dynamic and (d.divisible_by is not None or d.buckets is not None):
        return repr(d)
    return str(d)


def describe_pair(a: DimExpr, b: DimExpr) -> tuple[str, str]:
    """Texts for two dimensions that differ, spelling out constraints when the names alone would not tell them apart."""
    ta, tb = str(a), str(b)
    if ta != tb:
        return dim_text(a), dim_text(b)

    def detailed(d: DimExpr) -> str:
        free = sorted(d.free_dims(), key=lambda x: x.name)
        return repr(d) if isinstance(d, Dim) else f"{d} with " + ", ".join(repr(f) for f in free)

    return detailed(a), detailed(b)
