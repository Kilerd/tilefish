"""Validation of serialized compile specs.

``CompileSpec.from_dict`` feeds ``triton.compile`` and, later, generated host
code, so a serialized spec is checked as untrusted input: strict keys and
types, then cross-field consistency (the signature agrees with the args, the
constants sit at the constexpr indices, the attributes are backed by
preconditions, the launch parameters follow the args, ...). Every violation is
a :class:`~tilefish.errors.TilefishSpecError`. Grid and dimension strings are
parsed with a small restricted grammar and never evaluated as code (the grid is
folded with the dimension algebra, so a static entry below 1 is rejected, as
``compile_specs`` does).

This checks that a spec is well formed and consistent with itself, not that it
is the spec of any declaration: for that, ``CompileSpec.verify_against(op)``
regenerates it. Schema 1 documents are still read; schema 2 adds the
``declaration_digest`` and the constexpr ``dtype``; schema 3 adds what a strided view, a device scalar and a prebuilt binary
need (``strides``, ``offset`` and ``stride_of`` on arguments, 64-bit scalars, the ``max_extent`` precondition, a pointer
alignment below 16 bytes for a pointer that is only element aligned, and the ``binary`` section of a spec that is not
compiled by Triton). Every new field is accepted from schema 3 only.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .dims import Dim, DimExpr, _CDiv, cdiv
from .dtypes import ALL_DTYPES
from .errors import TilefishError, TilefishSpecError

SCHEMA_VERSION = 3  # what to_dict writes: schema 2 plus strides, offsets, 64-bit scalars and prebuilt binaries
LEGACY_SCHEMA = 1  # still read, but such a spec has no declaration digest and cannot be verified
DIGEST_SCHEMA = 2  # the first schema with the declaration digest, constexpr dtypes, ``max_elements`` and ``dims``
STRIDED_SCHEMA = 3  # the first schema with strided views, device scalars and prebuilt binaries
SCHEMAS = (LEGACY_SCHEMA, DIGEST_SCHEMA, SCHEMA_VERSION)
ALIGNMENT = 16
NUM_WARPS = (1, 2, 4, 8, 16, 32)
MAX_STAGES = 16

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\Z")
_STEM = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_REF = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*):([A-Za-z_][A-Za-z0-9_]*)\Z")
_INDEX = re.compile(r"^[0-9]{1,6}\Z")
_DIGEST = re.compile(r"^[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[ ]*([0-9]{1,12}|[A-Za-z_][A-Za-z0-9_]*|//|[-+*(),])")

POINTER_TYPES = {d.triton_ptr for d in ALL_DTYPES.values()}
SCRATCH = (("global_scratch", "u64", 0), ("profile_scratch", "u64", 0))

TOP_KEYS = (
    "schema stem op triton buckets args signature constants attrs num_warps num_stages grid block "
    "launch_params preconditions"
).split()
TOP_KEYS_V2 = [*TOP_KEYS, "declaration_digest"]
PRE_FIELDS = {
    "pointer_alignment": {"arg", "index", "bytes"},
    "divisible": {"arg", "index", "dim", "by"},
    "static_size": {"arg", "index", "axis", "value"},
    "bucket": {"dim", "value", "buckets"},
    "max_elements": {"arg", "index", "shape", "value"},  # schema 2; a kernel that indexes with i32: every generated one, and every extern that does not say index_bits=64
    "max_extent": {"arg", "index", "value"},  # schema 3; the same for a strided pointer: the largest element offset it reaches
}
MAX_ELEMENTS = 2**31 - 1  # the limit of i32 offsets, the only bound a max_elements precondition may carry
FOLD_RANGE = range(-(2**127), 2**127)  # the integers a folded expression may hold: the i128 of the Rust reader (crates/tilefish/src/expr.rs)


def bad(msg: str) -> Any:
    raise TilefishSpecError(f"malformed compile spec: {msg}")


def is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _keys(d: Any, where: str, required: "set[str] | list[str]", optional: "set[str] | list[str]" = ()) -> None:
    if not isinstance(d, dict):
        bad(f"{where} must be an object, got {type(d).__name__}")
    required, optional = set(required), set(optional)
    missing = sorted(required - set(d))
    unknown = sorted(str(k) for k in set(d) - required - optional)
    if missing:
        bad(f"{where} is missing {missing}")
    if unknown:
        bad(f"{where} has unknown fields {unknown}")


def _ident(x: Any, where: str) -> str:
    if not isinstance(x, str) or not _IDENT.match(x):
        bad(f"{where} must be an identifier, got {x!r}")
    return x


def _int(x: Any, where: str, minimum: int = 0) -> int:
    if not is_int(x) or x < minimum:
        bad(f"{where} must be an int >= {minimum}, got {x!r}")
    return x


def stride_of_json(targets: "Sequence[tuple[str, int]]") -> Any:
    """``stride_of`` as a compile spec, a manifest entry and a declaration digest write it: the object ``{"arg", "axis"}`` of the one view whose
    stride a scalar carries, or a list of them, in the order of the declaration, for the views that share it (``StrideOf(("q", "k", "v"), 0)``).
    A stride of one view is written as it always was, so the digests of the declarations and manifests that have one do not change."""
    items = [{"arg": arg, "axis": axis} for arg, axis in targets]
    return items[0] if len(items) == 1 else items


def stride_of_targets(value: Any) -> "tuple[tuple[str, int], ...]":
    """The ``(argument, axis)`` pairs of a ``stride_of`` that :func:`validate_stride_of` accepted (the inverse of :func:`stride_of_json`)."""
    items = [value] if isinstance(value, dict) else value
    return tuple((item["arg"], item["axis"]) for item in items)


def validate_stride_of(so: Any, where: str) -> None:
    """A ``stride_of``: ``{"arg": <identifier>, "axis": <int >= 0>}``, or a list of two or more of them with distinct ``(arg, axis)`` pairs."""
    if isinstance(so, list):
        if len(so) < 2:
            bad(f"{where} stride_of is a list of the views that share a stride, so it has two or more of them, got {so!r} (one view is an object)")
        items = so
    elif isinstance(so, dict):
        items = [so]
    else:
        bad(f"{where} stride_of must be an object {{arg, axis}} or a list of two or more of them, got {type(so).__name__}")
    pairs = set()
    for item in items:
        _keys(item, f"{where} stride_of", ["arg", "axis"])
        _ident(item["arg"], f"{where} stride_of arg")
        _int(item["axis"], f"{where} stride_of axis")
        if (item["arg"], item["axis"]) in pairs:
            bad(f"{where} stride_of names axis {item['axis']} of '{item['arg']}' twice")
        pairs.add((item["arg"], item["axis"]))


def decode_scalar(v: Any, where: str) -> Any:
    """A constexpr value: a bool, an int, a finite float, or ``{"float": "inf" | "-inf" | "nan"}``."""
    if isinstance(v, dict):
        if set(v) != {"float"} or v["float"] not in ("inf", "-inf", "nan"):
            bad(f"{where} is not a valid tagged float: {v!r}")
        return float(v["float"])
    if isinstance(v, float):
        if not math.isfinite(v):
            bad(f"{where} is a non-finite float outside a tag: {v!r}")
        return v
    if is_int(v) or isinstance(v, bool):
        return v
    return bad(f"{where} must be a number or a bool, got {type(v).__name__}")


def same_scalar(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    return bool(a == b)


# -- the restricted expression grammar ------------------------------------------------------------------------------


def in_fold_range(e: "DimExpr | None") -> "DimExpr | None":
    """``e`` when its constant and every coefficient are in :data:`FOLD_RANGE` (128 bits, as the Rust reader folds), else ``None``.

    The atom of a ``cdiv`` holds forms that were folded (and checked) before it was made, so the terms of ``e`` are all there is to look at.
    """
    if e is None or e.offset not in FOLD_RANGE or any(c not in FOLD_RANGE for _, c in e.terms):
        return None
    return e


def _cdiv_in_range(a: DimExpr, b: DimExpr) -> DimExpr:
    """``cdiv(a, b)`` folded as the Rust reader folds it: a symbolic divisor whose ratio to the dividend is beyond 128 bits leaves the atom.

    (Rust's ``Affine::ratio`` divides coefficient by coefficient with checked arithmetic, and the one quotient that overflows, ``-2**127``
    by ``-1``, makes it find no ratio; the ``cdiv`` is then an atom of its folded parts, where Python's unbounded ``exact_div`` finds
    ``2**127``.)
    """
    out = cdiv(a, b)
    if isinstance(out, DimExpr) and out.is_static and not b.folded().is_static and in_fold_range(out) is None:
        return _CDiv(a.folded(), b.folded())
    return out


class _ExprParser:
    """ints, identifiers, ``cdiv(a, b)``, ``+ - * //``, unary minus, parentheses. Never evaluated as code.

    Alongside the check, every sub-expression is folded with the same dimension algebra that
    ``compile_specs`` uses (:class:`~tilefish.dims.DimExpr`), so a grid entry that is a constant,
    including ``1 - 1`` or ``n - n``, is known. Its methods return the folded value, or ``None`` when
    the expression is outside the affine language (two symbolic factors, an inexact ``//``).

    The fold is the one the Rust reader makes (``Expr::affine`` in ``crates/tilefish/src/expr.rs``), which is exact in 128-bit
    integers: a fold whose constant or coefficient leaves ``[-2**127, 2**127 - 1]`` (:data:`FOLD_RANGE`), or that negates the least
    of them (``a - b`` with ``-2**127`` in ``b``), has no value here either, where Python's unbounded integers would have one. So
    the two readers know the same constants: ``(X - X,)`` with a literal product ``X`` of ``2**127`` or more (four 12-digit
    factors) is not the static grid ``(0,)`` for either, and is not refused for it (its value overflows when a launch computes it);
    a divisor that folds to a constant below 1 only in unbounded arithmetic is not refused either. A ``cdiv`` by a symbolic divisor
    whose ratio leaves the range is the ``cdiv`` atom, as the Rust ``Affine::ratio`` keeps it. No expression ``tilefish`` writes comes
    near the range.
    """

    MAX_LEN, MAX_TOKENS, MAX_DEPTH = 512, 256, 24

    def __init__(self, text: Any, where: str) -> None:
        if not isinstance(text, str):
            bad(f"{where} must be a string, got {type(text).__name__}")
        if len(text) > self.MAX_LEN:
            bad(f"{where} is too long")
        self.where, self.text = where, text
        self.toks: list[str] = []
        pos, stripped = 0, text.rstrip(" ")
        while pos < len(stripped):
            m = _TOKEN.match(stripped, pos)
            if not m:
                bad(f"{where} {text!r} has a character the dimension grammar does not allow (at {pos})")
            self.toks.append(m.group(1))
            pos = m.end()
        if len(self.toks) > self.MAX_TOKENS:
            bad(f"{where} is too long")
        self.i = 0
        self.ids: set[str] = set()

    def peek(self) -> "str | None":
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, expected: "str | None" = None) -> str:
        tok = self.peek()
        if tok is None or (expected is not None and tok != expected):
            bad(f"{self.where} {self.text!r}: expected {expected or 'more input'}, got {tok!r}")
        self.i += 1
        return tok  # type: ignore[return-value]

    def done(self) -> None:
        if self.peek() is not None:
            bad(f"{self.where} {self.text!r}: unexpected {self.peek()!r}")

    def _fold(self, fn: Any, *operands: "DimExpr | None") -> "DimExpr | None":
        """Apply a dimension operation; ``None`` when an operand is opaque, the result is not affine or it leaves 128 bits."""
        if any(o is None for o in operands):
            return None
        try:
            return in_fold_range(fn(*operands))
        except TilefishError:
            return None

    def _positive_divisor(self, divisor: "DimExpr | None", what: str) -> None:
        if divisor is not None and divisor.is_static and divisor.value < 1:
            bad(f"{self.where} {self.text!r}: {what} by {divisor.value} (the divisor must be positive)")

    def expr(self, depth: int = 0) -> "DimExpr | None":
        acc = self.term(depth)
        while self.peek() in ("+", "-"):
            op = self.take()
            rhs = self.term(depth)
            # (Rust subtracts by adding the negated operand: a -2**127 in it overflows, whatever the difference)
            acc = self._fold((lambda a, b: a + b) if op == "+" else (lambda a, b: None if in_fold_range(-b) is None else a - b), acc, rhs)
        return acc

    def term(self, depth: int) -> "DimExpr | None":
        acc = self.factor(depth)
        while self.peek() in ("*", "//"):
            op = self.take()
            rhs = self.factor(depth)
            if op == "*":
                acc = self._fold(lambda a, b: a * b, acc, rhs)
            else:
                self._positive_divisor(rhs, "division")
                acc = self._fold(lambda a, b: a // b, acc, rhs)
        return acc

    def factor(self, depth: int) -> "DimExpr | None":
        if depth > self.MAX_DEPTH:
            bad(f"{self.where} is nested too deeply")
        tok = self.take()
        if tok == "-":
            return self._fold(lambda a: -a, self.factor(depth + 1))
        if tok.isascii() and tok.isdigit():
            return DimExpr.const(int(tok))
        if tok == "(":
            inner = self.expr(depth + 1)
            self.take(")")
            return inner
        if _IDENT.match(tok):
            if tok == "cdiv":
                self.take("(")
                num = self.expr(depth + 1)
                self.take(",")
                den = self.expr(depth + 1)
                self.take(")")
                self._positive_divisor(den, "cdiv")
                return self._fold(_cdiv_in_range, num, den)
            if self.peek() == "(":
                bad(f"{self.where} {self.text!r}: only cdiv(a, b) may be called, not {tok}(...)")
            self.ids.add(tok)
            return Dim(tok)
        return bad(f"{self.where} {self.text!r}: unexpected {tok!r}")


def parse_expression(text: Any, where: str) -> "set[str]":
    """Check a dimension expression such as ``2*S + 1`` or ``cdiv(n, 128)``; return its identifiers."""
    p = _ExprParser(text, where)
    p.expr()
    p.done()
    return p.ids


def parse_grid(text: Any) -> "set[str]":
    """Check a grid such as ``(M,)`` or ``(cdiv(S, 16), 32)``; return its identifiers.

    Like ``compile_specs``, it rejects a static entry below 1 (``(0,)``, ``(-1,)``, ``(n - n, 4)``).
    """
    p = _ExprParser(text, "grid")
    p.take("(")
    entries, trailing = [], False
    while True:
        entries.append(p.expr())
        if p.peek() == ",":
            p.take()
            if p.peek() == ")":
                trailing = True
                break
        else:
            break
    p.take(")")
    p.done()
    if not 1 <= len(entries) <= 3:
        bad(f"grid {text!r} must have 1 to 3 entries")
    if len(entries) == 1 and not trailing:
        bad(f"grid {text!r}: a single entry needs a trailing comma, like (n,)")
    for position, entry in enumerate(entries):
        if entry is not None and entry.is_static and entry.value < 1:
            bad(f"grid {text!r}: entry {position} is the static size {entry.value}; grid sizes must be at least 1")
    return p.ids


# -- the pieces ----------------------------------------------------------------------------------------------------------


def _value_of(text: str, where: str) -> "DimExpr | None":
    """The affine dimension expression ``text`` denotes (``None`` when it is not affine)."""
    p = _ExprParser(text, where)
    value = p.expr()
    p.done()
    return value


def _check_count_of(scalar: dict, by_name: dict) -> None:
    """``count_of`` names a pointer argument; the scalar is the product of that pointer's shape.

    When the product is affine the scalar's ``dim`` must be exactly that expression. When it is not
    (two symbolic sizes), ``dim`` is a fresh dimension name, and ``count_of`` is how a host gets the value.
    """
    where = f"arg {scalar['name']} count_of"
    target = by_name.get(scalar["count_of"])
    if target is None or target["kind"] != "pointer":
        bad(f"{where} {scalar['count_of']!r} must name a pointer argument")
    product: "DimExpr | None" = DimExpr.const(1)
    for i, entry in enumerate(target["shape"]):
        factor = _value_of(entry, f"{where} shape[{i}]")
        try:  # (in 128 bits, as the Rust reader's `check_count_of` folds the product: one beyond them is not affine)
            product = None if factor is None or product is None else in_fold_range(product * factor)
        except TilefishError:
            product = None
    if product is not None:
        if _value_of(scalar["dim"], f"{where} dim") != product:
            bad(f"{where}: arg {scalar['name']} carries {scalar['dim']!r} but the shape of {scalar['count_of']} has {product} elements")
    elif not _IDENT.match(scalar["dim"]):
        bad(f"{where}: the element count of {scalar['count_of']} is not an affine expression, so arg {scalar['name']} must carry a fresh dimension name, got {scalar['dim']!r}")


def validate_precondition(d: Any) -> None:
    if not isinstance(d, dict) or d.get("kind") not in PRE_FIELDS:
        bad(f"precondition kind must be one of {sorted(PRE_FIELDS)}, got {d.get('kind') if isinstance(d, dict) else d!r}")
    kind = d["kind"]
    _keys(d, f"{kind} precondition", {"kind"} | PRE_FIELDS[kind])
    if "arg" in d:
        _ident(d["arg"], f"{kind} precondition arg")
    if "dim" in d:
        if kind == "divisible":  # the dimension expression whose value the runtime argument carries
            parse_expression(d["dim"], "divisible precondition dim")
        else:
            _ident(d["dim"], f"{kind} precondition dim")
    for k in ("index", "axis"):
        if k in d:
            _int(d[k], f"{kind} precondition {k}")
    for k in ("value", "by", "bytes"):
        if k in d:
            _int(d[k], f"{kind} precondition {k}", 1)
    if kind == "max_elements":
        if d["value"] != MAX_ELEMENTS:
            bad(f"max_elements precondition value must be {MAX_ELEMENTS} (the limit of i32 offsets), got {d['value']!r}")
        shape = d["shape"]
        if not isinstance(shape, list):
            bad("max_elements precondition shape must be a list of dimension strings")
        for i, dim in enumerate(shape):
            parse_expression(dim, f"max_elements precondition shape[{i}]")
    if "buckets" in d:
        b = d["buckets"]
        if not isinstance(b, list) or not b or any(not is_int(x) or x < 1 for x in b) or b != sorted(set(b)):
            bad(f"bucket precondition buckets must be a sorted list of distinct positive ints, got {b!r}")


SCALAR_TYPES = ("i32", "i64", "f32")  # the C types of a scalar argument of a prebuilt binary (a runtime scalar is i32 or i64)


def _layout_entry(entry: Any, where: str) -> None:
    """One stride or the offset of a strided pointer: a size (an int from 0) or a dimension expression."""
    if is_int(entry):
        if entry < 0:
            bad(f"{where} must not be negative, got {entry}")
    elif isinstance(entry, str):
        parse_expression(entry, where)
    else:
        bad(f"{where} must be an int or a dimension expression, got {entry!r}")


def validate_arg(d: Any, schema: int = SCHEMA_VERSION) -> None:
    """Check one serialized argument. From schema 2 on a constexpr may carry its declared ``dtype``; from schema 3 a pointer
    may be strided (``strides`` and ``offset``), a scalar may carry a stride (``stride_of``) and be 64-bit, and a prebuilt
    binary has ``fixed`` arguments (a value written in the manifest)."""
    _keys(d, "arg", ["index", "name", "kind", "type"], ["access", "dtype", "shape", "shape_dims", "dim", "value", "count_of", "strides", "offset", "stride_of", "alias"])
    _int(d["index"], "arg index")
    _ident(d["name"], "arg name")
    kind, ty = d["kind"], d["type"]
    kinds = ("pointer", "scalar", "constexpr") + (("fixed",) if schema >= STRIDED_SCHEMA else ())
    if kind not in kinds or not isinstance(ty, str):
        bad(f"arg {d['name']}: kind must be {', '.join(kinds)} and type a string, got {kind!r}, {ty!r}")
    present = set(d) - {"index", "name", "kind", "type"}
    expected = {
        "pointer": {"access", "dtype", "shape"},
        "scalar": {"dtype", "dim"},
        "constexpr": {"value"},
        "fixed": {"dtype", "value"},
    }[kind]
    optional: "set[str]" = set()
    if kind == "constexpr" and schema >= DIGEST_SCHEMA:
        optional = {"dtype"}
    if kind == "scalar" and schema >= DIGEST_SCHEMA:
        optional = {"count_of"}  # the pointer argument whose element count this scalar carries
        if schema >= STRIDED_SCHEMA:
            optional = {"count_of", "stride_of"}  # or the pointer argument whose stride along an axis it carries
    if kind == "pointer" and schema >= DIGEST_SCHEMA:
        optional = {"shape_dims"}  # the declared shape: what a host binds the dimensions from
        if schema >= STRIDED_SCHEMA:
            optional = {"shape_dims", "strides", "offset", "alias"}  # a pointer that accepts a strided view; an Out that may be written over an input
    if not expected <= present <= expected | optional:
        bad(f"arg {d['name']} ({kind}) must have exactly the fields {sorted(expected)}, has {sorted(present)}")
    if ("strides" in d) != ("offset" in d):
        bad(f"arg {d['name']}: strides and offset come together")
    if kind == "fixed":
        if ty not in SCALAR_TYPES or d["dtype"] != ty:
            bad(f"arg {d['name']}: a fixed argument is one of {SCALAR_TYPES}, got {ty!r}/{d['dtype']!r}")
        value = decode_scalar(d["value"], f"arg {d['name']} value")
        if isinstance(value, bool) or (ty == "f32") != isinstance(value, float):
            bad(f"arg {d['name']}: the value of a fixed {ty} argument must be {'a float' if ty == 'f32' else 'an int'}, got {d['value']!r}")
        if ty != "f32":
            lo, hi = (-(1 << 31), (1 << 31) - 1) if ty == "i32" else (-(1 << 63), (1 << 63) - 1)
            if not lo <= value <= hi:
                bad(f"arg {d['name']}: the value {value} does not fit {ty}")
    elif kind == "pointer":
        dtype = ALL_DTYPES.get(d["dtype"]) if isinstance(d["dtype"], str) else None
        if ty not in POINTER_TYPES or dtype is None or dtype.triton_ptr != ty:
            bad(f"arg {d['name']}: pointer type {ty!r} does not match dtype {d['dtype']!r}")
        if d["access"] not in ("in", "inout", "out"):
            bad(f"arg {d['name']}: access must be in, inout or out, got {d['access']!r}")
        if not isinstance(d["shape"], list):
            bad(f"arg {d['name']}: shape must be a list of dimension strings")
        for i, dim in enumerate(d["shape"]):
            parse_expression(dim, f"arg {d['name']} shape[{i}]")
        if "shape_dims" in d:
            declared = d["shape_dims"]
            if not isinstance(declared, list) or len(declared) != len(d["shape"]):
                bad(f"arg {d['name']}: shape_dims must be a list with one entry per axis of shape")
            for i, entry in enumerate(declared):
                if is_int(entry):
                    if entry < 1 or str(entry) != d["shape"][i]:
                        bad(f"arg {d['name']}: shape_dims[{i}] is the size {entry}, but shape[{i}] is {d['shape'][i]!r}")
                elif isinstance(entry, str):
                    parse_expression(entry, f"arg {d['name']} shape_dims[{i}]")
                else:
                    bad(f"arg {d['name']}: shape_dims[{i}] must be an int size or a dimension expression, got {entry!r}")
        if "strides" in d:
            strides = d["strides"]
            if not isinstance(strides, list) or len(strides) != len(d["shape"]) or not strides:
                bad(f"arg {d['name']}: strides must be a list with one entry per axis of shape (at least one axis)")
            for i, entry in enumerate(strides):
                _layout_entry(entry, f"arg {d['name']} strides[{i}]")
            _layout_entry(d["offset"], f"arg {d['name']} offset")
    elif kind == "scalar":
        allowed = ("i32", "i64") if schema >= STRIDED_SCHEMA else ("i32",)
        if ty not in allowed or d["dtype"] != ty:
            bad(f"arg {d['name']}: a runtime scalar must be {' or '.join(allowed)}, got {ty!r}/{d['dtype']!r}")
        parse_expression(d["dim"], f"arg {d['name']} dim")
        if "count_of" in d:
            _ident(d["count_of"], f"arg {d['name']} count_of")
        if "stride_of" in d:
            validate_stride_of(d["stride_of"], f"arg {d['name']}")
            if "count_of" in d:
                bad(f"arg {d['name']}: a scalar is the element count of a pointer or the stride of one, not both")
    else:
        if ty != "constexpr":
            bad(f"arg {d['name']}: a constexpr's type must be 'constexpr', got {ty!r}")
        decode_scalar(d["value"], f"arg {d['name']} value")
        if "dtype" in d and (not isinstance(d["dtype"], str) or d["dtype"] not in ALL_DTYPES):
            bad(f"arg {d['name']}: a constexpr's dtype must be one of {sorted(ALL_DTYPES)}, got {d['dtype']!r}")


_SYMBOL = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*\Z")  # a PTX / ELF symbol of a kernel
_ARCH_KEY = re.compile(r"^sm_[0-9]{2,3}\Z")


def validate_binary(b: Any) -> None:
    """The ``binary`` section of a spec of a prebuilt (non-Triton) kernel: ``kind``, ``symbol``, dynamic ``shared`` memory and the sha256 of the source."""
    _keys(b, "binary", ["kind", "symbol", "shared", "digests"])
    if b["kind"] not in ("ptx", "cubin"):
        bad(f"binary kind must be 'ptx' or 'cubin', got {b['kind']!r}")
    if not isinstance(b["symbol"], str) or not _SYMBOL.match(b["symbol"]):
        bad(f"binary symbol must be a kernel symbol, got {b['symbol']!r}")
    _int(b["shared"], "binary shared")
    digests = b["digests"]
    if not isinstance(digests, dict) or not digests:
        bad("binary digests must be a non-empty object")
    for k, v in digests.items():
        if not (isinstance(v, str) and _DIGEST.match(v)):
            bad(f"binary digest of {k!r} must be 64 lowercase hex digits (a sha256), got {v!r}")
        if b["kind"] == "cubin" and not _ARCH_KEY.match(str(k)):
            bad(f"a cubin's digests are keyed by architecture ('sm_103'), got {k!r}")
    if b["kind"] == "ptx" and set(digests) != {"ptx"}:
        bad(f"a ptx binary has one digest, keyed 'ptx', got {sorted(digests)}")


def _check_strides(args: "list[dict]", by_name: "dict[str, dict]") -> None:
    """A scalar's ``stride_of`` and the ``strides`` of the pointers it names must say the same thing, axis by axis (a scalar that several views
    share is the stride entry of each of them)."""
    carried: "dict[tuple[str, int], str]" = {}
    for a in args:
        if a["kind"] != "scalar" or "stride_of" not in a:
            continue
        for arg, axis in stride_of_targets(a["stride_of"]):
            target = by_name.get(arg)
            if target is None or target["kind"] != "pointer" or "strides" not in target:
                bad(f"arg {a['name']} is the stride of '{arg}', which is not a strided pointer argument")
            if axis >= len(target["strides"]):
                bad(f"arg {a['name']} is the stride of axis {axis} of '{arg}', which has {len(target['strides'])} axes")
            if target["strides"][axis] != a["dim"]:
                bad(
                    f"arg {a['name']} carries {a['dim']!r} as the stride of axis {axis} of '{arg}', "
                    f"but the strides of that pointer say {target['strides'][axis]!r}"
                )
            if (arg, axis) in carried:
                bad(f"axis {axis} of '{arg}' has two stride scalars ({carried[(arg, axis)]} and {a['name']})")
            carried[(arg, axis)] = a["name"]
    for a in args:
        if a["kind"] == "pointer" and "strides" in a:
            for axis, entry in enumerate(a["strides"]):
                if isinstance(entry, str) and (a["name"], axis) not in carried:
                    bad(f"the stride {entry!r} of axis {axis} of '{a['name']}' is carried by no scalar argument (stride_of)")
                if is_int(entry) and entry != 1:  # the only fixed stride is the unit stride of the last axis, which a kernel loads along
                    bad(f"the stride of axis {axis} of '{a['name']}' is the fixed size {entry}: a fixed stride is 1 (and only on the last axis)")
                if is_int(entry) and axis != len(a["strides"]) - 1:
                    bad(f"the stride of axis {axis} of '{a['name']}' is fixed, but only the last axis may be")
            if not isinstance(a["offset"], str):
                bad(f"the offset of '{a['name']}' must be the name of the offset dimension, got {a['offset']!r}")


def check_alias_target(name: str, alias: Any, access: Any, by_name: "Mapping[str, Mapping[str, Any]]", shape_key: str = "shape") -> None:
    """``alias`` on the pointer ``name``: it is on an ``out`` pointer, and names an ``in`` pointer that is not strided and not a device scalar, with the
    pointer's dtype and declared shape (``shape_key``: ``shape_dims`` in a spec, ``shape`` in a manifest's ABI). Shared by the spec and the
    manifest reader."""
    if not isinstance(alias, str) or not _IDENT.match(alias):
        bad(f"arg {name}: alias must be the name of a pointer argument, got {alias!r}")
    if access != "out":
        bad(f"arg {name}: only an out pointer may alias an input (it is {access!r})")
    target = by_name.get(alias)
    me = by_name[name]
    if target is None or target.get("kind") != "pointer" or target.get("access") != "in":
        bad(f"arg {name} may alias '{alias}', which is not an `in` pointer argument")
    if "strides" in target or target.get(shape_key) == []:
        bad(f"arg {name} may alias '{alias}', which is a strided view or a device scalar: an Out written over an input is that whole tensor")
    if target.get("dtype") != me.get("dtype") or target.get(shape_key) != me.get(shape_key):
        bad(
            f"arg {name} may alias '{alias}', which has another dtype or declared shape ({target.get('dtype')!r} {target.get(shape_key)!r}, "
            f"not {me.get('dtype')!r} {me.get(shape_key)!r})"
        )


def _check_aliases(args: "list[dict]", by_name: "dict[str, dict]") -> None:
    """An ``out`` pointer that may be written over an ``in`` pointer (``alias``): see :func:`check_alias_target`; at most one names an input."""
    taken: "dict[str, str]" = {}
    for a in args:
        if a["kind"] != "pointer" or "alias" not in a:
            continue
        check_alias_target(a["name"], a["alias"], a["access"], by_name, "shape_dims" if "shape_dims" in a else "shape")
        if a["alias"] in taken:
            bad(f"args {taken[a['alias']]} and {a['name']} both may alias '{a['alias']}': at most one out pointer is written over an input")
        taken[a["alias"]] = a["name"]


def validate_launch_param(d: Any) -> None:
    _keys(d, "launch parameter", ["name", "type"], ["value"])
    _ident(d["name"], "launch parameter name")
    if not isinstance(d["type"], str):
        bad(f"launch parameter type must be a string, got {d['type']!r}")
    if "value" in d:
        _int(d["value"], "launch parameter value")


def validate_spec(d: Any) -> None:
    """Strict structure, then cross-field consistency of a serialized compile spec."""
    if not isinstance(d, dict):
        bad(f"a spec must be an object, got {type(d).__name__}")
    schema = d.get("schema")
    if not is_int(schema) or schema not in SCHEMAS:
        raise TilefishSpecError(
            f"unsupported compile spec schema {schema!r} (this version reads schemas {', '.join(map(str, SCHEMAS))})"
        )
    # a prebuilt binary (schema 3) has a ``binary`` section where a Triton kernel has the reference of its function, its warps and stages
    binary = d.get("binary") if schema >= STRIDED_SCHEMA else None
    is_binary = binary is not None
    # enable_fp_fusion and source_sha256 exist from schema 2 on: both in the specs of kernels that tilefish generated, enable_fp_fusion in
    # those of an extern declared with enable_fp_fusion= too
    optional = ["enable_fp_fusion", "source_sha256", "dims"] if schema >= DIGEST_SCHEMA else []
    required = TOP_KEYS_V2 if schema >= DIGEST_SCHEMA else TOP_KEYS
    if is_binary:
        required = [k for k in required if k not in ("triton", "num_warps", "num_stages")] + ["binary"]
        optional = ["dims"]
    _keys(d, "spec", required, optional)
    if "source_sha256" in d and not (isinstance(d["source_sha256"], str) and _DIGEST.match(d["source_sha256"])):
        bad(f"source_sha256 must be 64 lowercase hex digits (a sha256), got {d['source_sha256']!r}")
    if schema >= DIGEST_SCHEMA and not (isinstance(d["declaration_digest"], str) and _DIGEST.match(d["declaration_digest"])):
        bad(f"declaration_digest must be 64 lowercase hex digits (a sha256), got {d['declaration_digest']!r}")
    if "enable_fp_fusion" in d and not isinstance(d["enable_fp_fusion"], bool):
        bad(f"enable_fp_fusion must be true or false, got {d['enable_fp_fusion']!r}")

    # names
    op = _ident(d["op"], "op")
    stem = d["stem"]
    if not isinstance(stem, str) or not _STEM.match(stem):
        bad(f"stem must look like 'op' or 'op.S32', got {stem!r}")
    if is_binary:
        validate_binary(binary)
    else:
        t = d["triton"]
        _keys(t, "triton", ["module", "function", "ref"])
        m = _REF.match(t["ref"]) if isinstance(t["ref"], str) else None
        if not m:
            bad(f"triton ref must look like 'module.path:function', got {t['ref']!r}")
        if (t["module"], t["function"]) != (m.group(1), m.group(2)):
            bad("triton module and function do not match the ref")

    # buckets and stem
    buckets = d["buckets"]
    if not isinstance(buckets, dict):
        bad("buckets must be an object")
    for name, value in buckets.items():
        _ident(name, "bucket dimension name")
        _int(value, f"bucket value of {name}", 1)
    if stem != op + "".join(f".{k}{v}" for k, v in buckets.items()):
        bad(f"stem {stem!r} does not match op {op!r} and buckets {buckets}")

    # args
    args = d["args"]
    if not isinstance(args, list):
        bad("args must be a list")
    names: list[str] = []
    dims: set[str] = set(buckets)
    for position, a in enumerate(args):
        validate_arg(a, schema)
        if a["index"] != position:
            bad(f"arg {a['name']} has index {a['index']} but is at position {position}")
        names.append(a["name"])
        for expr in (a["shape"] if a["kind"] == "pointer" else [a["dim"]] if a["kind"] == "scalar" else []):
            dims |= parse_expression(expr, f"arg {a['name']} dimension")
        if a["kind"] == "pointer" and "strides" in a:
            for entry in (*a["strides"], a["offset"]):
                if isinstance(entry, str):
                    dims |= parse_expression(entry, f"arg {a['name']} layout")
        if is_binary and a["kind"] == "constexpr":
            bad(f"arg {a['name']}: a prebuilt binary has no constexprs")
    if len(set(names)) != len(names):
        bad("argument names must be unique")
    by_name = {a["name"]: a for a in args}
    pointers = [a for a in args if a["kind"] == "pointer"]
    scalars = [a for a in args if a["kind"] == "scalar"]
    constexprs = [a for a in args if a["kind"] == "constexpr"]

    # a scalar that carries an element count: the pointer's shape says how to compute it
    for a in scalars:
        if "count_of" in a:
            _check_count_of(a, by_name)
    # a scalar that carries a stride, and the strided pointer that says so: the two must agree on every axis
    _check_strides(args, by_name)
    # an Out that may be written over an input, and that input
    _check_aliases(args, by_name)

    # signature <=> args
    sig = d["signature"]
    if not isinstance(sig, dict) or list(sig.items()) != [(a["name"], a["type"]) for a in args]:
        bad("signature must list exactly the args' names and types, in order")

    # constants only at constexpr indices
    consts = d["constants"]
    if not isinstance(consts, dict):
        bad("constants must be an object")
    const_index: dict[int, Any] = {}
    for k, v in consts.items():
        if not isinstance(k, str) or not _INDEX.match(k):
            bad(f"constants key {k!r} must be an argument index")
        const_index[int(k)] = decode_scalar(v, f"constant {k}")
    if set(const_index) != {a["index"] for a in constexprs}:
        bad("constants must have exactly one entry per constexpr argument")
    for a in constexprs:
        if not same_scalar(const_index[a["index"]], decode_scalar(a["value"], "arg value")):
            bad(f"constant at index {a['index']} differs from the value of arg {a['name']}")

    # preconditions
    pres = d["preconditions"]
    if not isinstance(pres, list):
        bad("preconditions must be a list")
    aligned: set[int] = set()
    attr_pointers: set[int] = set()  # the pointers promised 16-byte alignment: a Triton kernel gets tt.divisibility on exactly these
    divisible: set[int] = set()
    limited: set[int] = set()
    bucket_pres: dict[str, int] = {}
    for p in pres:
        validate_precondition(p)
        kind = p["kind"]
        if kind == "max_elements" and schema < DIGEST_SCHEMA:
            bad(f"a max_elements precondition needs schema {DIGEST_SCHEMA}, this spec is schema {schema}")
        if kind == "max_extent" and schema < STRIDED_SCHEMA:
            bad(f"a max_extent precondition needs schema {STRIDED_SCHEMA}, this spec is schema {schema}")
        if kind == "bucket":
            if p["dim"] in bucket_pres or buckets.get(p["dim"]) != p["value"] or p["value"] not in p["buckets"]:
                bad(f"bucket precondition for {p['dim']} disagrees with the spec's buckets")
            bucket_pres[p["dim"]] = p["value"]
            continue
        a = by_name.get(p["arg"])
        if a is None or a["index"] != p["index"]:
            bad(f"{kind} precondition refers to {p['arg']}@{p['index']}, which is not an argument")
        if kind == "pointer_alignment":
            if a["kind"] != "pointer" or p["index"] in aligned:
                bad(f"pointer_alignment must be {ALIGNMENT} bytes, once, on a pointer argument ({p['arg']})")
            if "strides" in a:  # a strided pointer is 16-byte aligned only when its declaration says so, else aligned to its element
                itemsize = max(1, ALL_DTYPES[a["dtype"]].bits // 8)
                if p["bytes"] not in (ALIGNMENT, itemsize):
                    bad(f"pointer_alignment of the strided pointer {p['arg']} must be {ALIGNMENT} bytes or its element size {itemsize}, got {p['bytes']}")
            elif p["bytes"] != ALIGNMENT:
                bad(f"pointer_alignment must be {ALIGNMENT} bytes, once, on a pointer argument ({p['arg']})")
            aligned.add(p["index"])
            if p["bytes"] == ALIGNMENT:
                attr_pointers.add(p["index"])
        elif kind == "divisible":
            if a["kind"] != "scalar" or p["by"] % ALIGNMENT or p["index"] in divisible or p["dim"] != a["dim"]:
                bad(f"divisible must be by a multiple of {ALIGNMENT}, once, on a runtime scalar with its own dim ({p['arg']})")
            divisible.add(p["index"])
            dims |= parse_expression(p["dim"], "divisible dim")
        elif kind == "max_elements":
            if a["kind"] != "pointer" or p["index"] in limited or p["shape"] != a["shape"] or "strides" in a:
                bad(f"max_elements must be once, on a dense pointer argument, with that argument's own shape ({p['arg']})")
            limited.add(p["index"])
        elif kind == "max_extent":
            if a["kind"] != "pointer" or p["index"] in limited or "strides" not in a or p["value"] != MAX_ELEMENTS:
                bad(f"max_extent must be once, on a strided pointer argument, with the value {MAX_ELEMENTS} ({p['arg']})")
            limited.add(p["index"])
        else:  # static_size
            if a["kind"] != "pointer" or p["axis"] >= len(a["shape"]) or a["shape"][p["axis"]] != str(p["value"]):
                bad(f"static_size of {p['arg']} axis {p['axis']} does not match its shape")
    if bucket_pres.keys() != buckets.keys():
        bad("every bucket needs exactly one bucket precondition")
    if aligned != {a["index"] for a in pointers}:
        bad("every pointer argument needs exactly one pointer_alignment precondition")
    if divisible - {a["index"] for a in scalars}:  # pragma: no cover - rejected above
        bad("divisible preconditions must be on runtime scalars")

    # attrs: pointers and the scalars backed by a divisibility precondition, exactly [["tt.divisibility", 16]]
    attrs = d["attrs"]
    if not isinstance(attrs, dict):
        bad("attrs must be an object")
    attr_index: set[int] = set()
    for k, v in attrs.items():
        if not isinstance(k, str) or not _INDEX.match(k) or int(k) >= len(args):
            bad(f"attrs key {k!r} must be an argument index in range")
        if not (
            isinstance(v, list)
            and len(v) == 1
            and isinstance(v[0], list)
            and len(v[0]) == 2
            and v[0][0] == "tt.divisibility"
            and is_int(v[0][1])
            and v[0][1] == ALIGNMENT
        ):
            bad(f"attrs at {k} must be exactly [['tt.divisibility', {ALIGNMENT}]]")
        attr_index.add(int(k))
    if is_binary:
        if attrs:
            bad("a prebuilt binary has no Triton attributes: attrs must be empty")
    elif attr_index != attr_pointers | divisible:
        bad("attrs must be at exactly the pointer arguments promised 16-byte alignment and the runtime scalars with a divisible precondition")

    # launch settings
    block = d["block"]
    if is_binary:
        if not isinstance(block, list) or len(block) != 3 or not all(is_int(x) and 1 <= x <= 1024 for x in block) or math.prod(block) > 1024:
            bad(f"block must be three sizes from 1 to 1024 whose product is at most 1024, got {block!r}")
        expected_lp = [{"name": a["name"], "type": a["type"]} for a in args]  # every argument, and no Triton scratch slots
        why = "launch_params must be every arg in order (a prebuilt binary has no scratch slots)"
    else:
        nw, ns = d["num_warps"], d["num_stages"]
        if not is_int(nw) or nw not in NUM_WARPS:
            bad(f"num_warps must be one of {NUM_WARPS}, got {nw!r}")
        if not is_int(ns) or not 1 <= ns <= MAX_STAGES:
            bad(f"num_stages must be an int from 1 to {MAX_STAGES}, got {ns!r}")
        if not isinstance(block, list) or block != [32 * nw, 1, 1] or not all(is_int(x) for x in block):
            bad(f"block must be [32 * num_warps, 1, 1] = {[32 * nw, 1, 1]}, got {block!r}")
        expected_lp = [{"name": a["name"], "type": a["type"]} for a in args if a["kind"] != "constexpr"]
        expected_lp += [{"name": n, "type": t, "value": v} for n, t, v in SCRATCH]
        why = "launch_params must be the non-constexpr args in order followed by the two zero scratch slots"
    lp = d["launch_params"]
    if not isinstance(lp, list):
        bad("launch_params must be a list")
    for p in lp:
        validate_launch_param(p)
    if lp != expected_lp:
        bad(why)

    # the grid: restricted grammar, only this spec's dimensions
    grid_ids = parse_grid(d["grid"])
    unknown = grid_ids - dims
    if unknown:
        bad(f"grid {d['grid']!r} mentions {sorted(unknown)}, which are not dimensions of this spec")

    # the constraints of the named dimensions: the names are the ones the declared shapes, the scalars and the grid use
    if "dims" in d:
        table = d["dims"]
        if not isinstance(table, dict):
            bad("dims must be an object")
        for name, c in table.items():
            _ident(name, "dims name")
            _keys(c, f"dims.{name}", [], ["divisible_by", "buckets", "bucket"])
            if "divisible_by" in c:
                _int(c["divisible_by"], f"dims.{name}.divisible_by", 1)
            if "buckets" in c:
                b = c["buckets"]
                if not isinstance(b, list) or not b or any(not is_int(x) or x < 1 for x in b) or b != sorted(set(b)):
                    bad(f"dims.{name}.buckets must be a sorted list of distinct positive ints, got {b!r}")
            if "bucket" in c and ("buckets" not in c or c["bucket"] not in c["buckets"] or buckets.get(name) != c["bucket"]):
                bad(f"dims.{name}.bucket must be one of its buckets and the bucket of this spec")
        for name in buckets:
            if table.get(name, {}).get("bucket") != buckets[name]:
                bad(f"every bucketed dimension needs its bucket in dims: {name} has none")
        used: set[str] = set(grid_ids)
        for a in args:
            for entry in a.get("shape_dims", ()):
                if isinstance(entry, str):
                    used |= parse_expression(entry, "shape_dims")
            if a["kind"] == "pointer" and "strides" in a:
                for entry in (*a["strides"], a["offset"]):
                    if isinstance(entry, str):
                        used |= parse_expression(entry, "layout")
            if a["kind"] == "scalar":
                used |= parse_expression(a["dim"], "scalar dim")
        missing = sorted(used - set(table))
        if missing:
            bad(f"dims does not name {missing}, which the declared shapes, the scalars or the grid mention")
