"""Validation of serialized compile specs.

``CompileSpec.from_dict`` feeds ``triton.compile`` and, later, generated host
code, so a serialized spec is checked as untrusted input: strict keys and
types, then cross-field consistency (the signature agrees with the args, the
constants sit at the constexpr indices, the attributes are backed by
preconditions, the launch parameters follow the args, ...). Every violation is
a :class:`~tilefish.errors.TilefishSpecError`. Grid and dimension strings are
parsed with a small restricted grammar and never evaluated.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .dtypes import ALL_DTYPES
from .errors import TilefishSpecError

SCHEMA_VERSION = 1
ALIGNMENT = 16
NUM_WARPS = (1, 2, 4, 8, 16, 32)
MAX_STAGES = 16

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\Z")
_STEM = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_REF = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*):([A-Za-z_][A-Za-z0-9_]*)\Z")
_INDEX = re.compile(r"^[0-9]{1,6}\Z")
_TOKEN = re.compile(r"[ ]*([0-9]{1,12}|[A-Za-z_][A-Za-z0-9_]*|//|[-+*(),])")

POINTER_TYPES = {d.triton_ptr for d in ALL_DTYPES.values()}
SCRATCH = (("global_scratch", "u64", 0), ("profile_scratch", "u64", 0))

TOP_KEYS = (
    "schema stem op triton buckets args signature constants attrs num_warps num_stages grid block "
    "launch_params preconditions"
).split()
PRE_FIELDS = {
    "pointer_alignment": {"arg", "index", "bytes"},
    "divisible": {"arg", "index", "dim", "by"},
    "static_size": {"arg", "index", "axis", "value"},
    "bucket": {"dim", "value", "buckets"},
}


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


class _ExprParser:
    """ints, identifiers, ``cdiv(a, b)``, ``+ - * //``, unary minus, parentheses. Never evaluated."""

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

    def expr(self, depth: int = 0) -> None:
        self.term(depth)
        while self.peek() in ("+", "-"):
            self.take()
            self.term(depth)

    def term(self, depth: int) -> None:
        self.factor(depth)
        while self.peek() in ("*", "//"):
            self.take()
            self.factor(depth)

    def factor(self, depth: int) -> None:
        if depth > self.MAX_DEPTH:
            bad(f"{self.where} is nested too deeply")
        tok = self.take()
        if tok == "-":
            return self.factor(depth + 1)
        if tok.isascii() and tok.isdigit():
            return None
        if tok == "(":
            self.expr(depth + 1)
            self.take(")")
            return None
        if _IDENT.match(tok):
            if tok == "cdiv":
                self.take("(")
                self.expr(depth + 1)
                self.take(",")
                self.expr(depth + 1)
                self.take(")")
            elif self.peek() == "(":
                bad(f"{self.where} {self.text!r}: only cdiv(a, b) may be called, not {tok}(...)")
            else:
                self.ids.add(tok)
            return None
        return bad(f"{self.where} {self.text!r}: unexpected {tok!r}")


def parse_expression(text: Any, where: str) -> "set[str]":
    """Check a dimension expression such as ``2*S + 1`` or ``cdiv(n, 128)``; return its identifiers."""
    p = _ExprParser(text, where)
    p.expr()
    p.done()
    return p.ids


def parse_grid(text: Any) -> "set[str]":
    """Check a grid such as ``(M,)`` or ``(cdiv(S, 16), 32)``; return its identifiers."""
    p = _ExprParser(text, "grid")
    p.take("(")
    entries, trailing = 0, False
    while True:
        p.expr()
        entries += 1
        if p.peek() == ",":
            p.take()
            if p.peek() == ")":
                trailing = True
                break
        else:
            break
    p.take(")")
    p.done()
    if not 1 <= entries <= 3:
        bad(f"grid {text!r} must have 1 to 3 entries")
    if entries == 1 and not trailing:
        bad(f"grid {text!r}: a single entry needs a trailing comma, like (n,)")
    return p.ids


# -- the pieces ----------------------------------------------------------------------------------------------------------


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
    if "buckets" in d:
        b = d["buckets"]
        if not isinstance(b, list) or not b or any(not is_int(x) or x < 1 for x in b) or b != sorted(set(b)):
            bad(f"bucket precondition buckets must be a sorted list of distinct positive ints, got {b!r}")


def validate_arg(d: Any) -> None:
    _keys(d, "arg", ["index", "name", "kind", "type"], ["access", "dtype", "shape", "dim", "value"])
    _int(d["index"], "arg index")
    _ident(d["name"], "arg name")
    kind, ty = d["kind"], d["type"]
    if kind not in ("pointer", "scalar", "constexpr") or not isinstance(ty, str):
        bad(f"arg {d['name']}: kind must be pointer, scalar or constexpr and type a string, got {kind!r}, {ty!r}")
    present = set(d) - {"index", "name", "kind", "type"}
    expected = {
        "pointer": {"access", "dtype", "shape"},
        "scalar": {"dtype", "dim"},
        "constexpr": {"value"},
    }[kind]
    if present != expected:
        bad(f"arg {d['name']} ({kind}) must have exactly the fields {sorted(expected)}, has {sorted(present)}")
    if kind == "pointer":
        dtype = ALL_DTYPES.get(d["dtype"]) if isinstance(d["dtype"], str) else None
        if ty not in POINTER_TYPES or dtype is None or dtype.triton_ptr != ty:
            bad(f"arg {d['name']}: pointer type {ty!r} does not match dtype {d['dtype']!r}")
        if d["access"] not in ("in", "inout", "out"):
            bad(f"arg {d['name']}: access must be in, inout or out, got {d['access']!r}")
        if not isinstance(d["shape"], list):
            bad(f"arg {d['name']}: shape must be a list of dimension strings")
        for i, dim in enumerate(d["shape"]):
            parse_expression(dim, f"arg {d['name']} shape[{i}]")
    elif kind == "scalar":
        if ty != "i32" or d["dtype"] != "i32":
            bad(f"arg {d['name']}: a runtime scalar must be i32, got {ty!r}/{d['dtype']!r}")
        parse_expression(d["dim"], f"arg {d['name']} dim")
    else:
        if ty != "constexpr":
            bad(f"arg {d['name']}: a constexpr's type must be 'constexpr', got {ty!r}")
        decode_scalar(d["value"], f"arg {d['name']} value")


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
    if d.get("schema") != SCHEMA_VERSION:
        raise TilefishSpecError(f"unsupported compile spec schema {d.get('schema')!r} (expected {SCHEMA_VERSION})")
    _keys(d, "spec", TOP_KEYS)

    # names
    op = _ident(d["op"], "op")
    stem = d["stem"]
    if not isinstance(stem, str) or not _STEM.match(stem):
        bad(f"stem must look like 'op' or 'op.S32', got {stem!r}")
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
        validate_arg(a)
        if a["index"] != position:
            bad(f"arg {a['name']} has index {a['index']} but is at position {position}")
        names.append(a["name"])
        for expr in (a["shape"] if a["kind"] == "pointer" else [a["dim"]] if a["kind"] == "scalar" else []):
            dims |= parse_expression(expr, f"arg {a['name']} dimension")
    if len(set(names)) != len(names):
        bad("argument names must be unique")
    by_name = {a["name"]: a for a in args}
    pointers = [a for a in args if a["kind"] == "pointer"]
    scalars = [a for a in args if a["kind"] == "scalar"]
    constexprs = [a for a in args if a["kind"] == "constexpr"]

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
    divisible: set[int] = set()
    bucket_pres: dict[str, int] = {}
    for p in pres:
        validate_precondition(p)
        kind = p["kind"]
        if kind == "bucket":
            if p["dim"] in bucket_pres or buckets.get(p["dim"]) != p["value"] or p["value"] not in p["buckets"]:
                bad(f"bucket precondition for {p['dim']} disagrees with the spec's buckets")
            bucket_pres[p["dim"]] = p["value"]
            continue
        a = by_name.get(p["arg"])
        if a is None or a["index"] != p["index"]:
            bad(f"{kind} precondition refers to {p['arg']}@{p['index']}, which is not an argument")
        if kind == "pointer_alignment":
            if a["kind"] != "pointer" or p["bytes"] != ALIGNMENT or p["index"] in aligned:
                bad(f"pointer_alignment must be {ALIGNMENT} bytes, once, on a pointer argument ({p['arg']})")
            aligned.add(p["index"])
        elif kind == "divisible":
            if a["kind"] != "scalar" or p["by"] % ALIGNMENT or p["index"] in divisible or p["dim"] != a["dim"]:
                bad(f"divisible must be by a multiple of {ALIGNMENT}, once, on a runtime scalar with its own dim ({p['arg']})")
            divisible.add(p["index"])
            dims |= parse_expression(p["dim"], "divisible dim")
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
    if attr_index != aligned | divisible:
        bad("attrs must be at exactly the pointer arguments and the runtime scalars with a divisible precondition")

    # launch settings
    nw, ns = d["num_warps"], d["num_stages"]
    if not is_int(nw) or nw not in NUM_WARPS:
        bad(f"num_warps must be one of {NUM_WARPS}, got {nw!r}")
    if not is_int(ns) or not 1 <= ns <= MAX_STAGES:
        bad(f"num_stages must be an int from 1 to {MAX_STAGES}, got {ns!r}")
    block = d["block"]
    if not isinstance(block, list) or block != [32 * nw, 1, 1] or not all(is_int(x) for x in block):
        bad(f"block must be [32 * num_warps, 1, 1] = {[32 * nw, 1, 1]}, got {block!r}")
    lp = d["launch_params"]
    if not isinstance(lp, list):
        bad("launch_params must be a list")
    for p in lp:
        validate_launch_param(p)
    expected_lp = [{"name": a["name"], "type": a["type"]} for a in args if a["kind"] != "constexpr"]
    expected_lp += [{"name": n, "type": t, "value": v} for n, t, v in SCRATCH]
    if lp != expected_lp:
        bad("launch_params must be the non-constexpr args in order followed by the two zero scratch slots")

    # the grid: restricted grammar, only this spec's dimensions
    unknown = parse_grid(d["grid"]) - dims
    if unknown:
        bad(f"grid {d['grid']!r} mentions {sorted(unknown)}, which are not dimensions of this spec")
