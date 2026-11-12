"""Ahead-of-time compile specs for ``extern`` Triton kernels.

``compile_specs(op)`` derives, from the types of an ``extern`` declaration,
everything Triton's ahead-of-time compiler needs, one :class:`CompileSpec`
per bucket combination. Nothing here imports triton; a later step feeds a
spec to it:

.. code-block:: python

    inp = spec.ast_source_inputs()
    src = ASTSource(fn, signature=inp.signature, constexprs=None, attrs=inp.attrs)
    src.constants = inp.constants
    triton.compile(src, target=GPUTarget("cuda", arch, 32), options=inp.options)

Conventions (all from the declaration, nothing guessed at compile time):

* ``signature`` maps each Triton parameter name, in declaration order, to
  ``"*bf16"``-style pointer types, ``"i32"`` runtime scalars or ``"constexpr"``.
* ``constants`` and ``attrs`` are keyed by one-element tuples of argument
  indices: ``{(3,): 1024}`` and ``{(0,): [["tt.divisibility", 16]]}``.
* Every pointer is assumed 16-byte aligned, and so is every ``i32`` argument
  whose dimension is provably a multiple of 16.
* Bucket semantics are exact: in a spec for bucket ``b`` the bucketed dimension
  is replaced by ``b`` in the grid, in the shapes and in every constant derived
  from it, and the spec requires the dimension to equal ``b`` at launch (the
  host picks the smallest bucket that fits and pads).
* Static dimensions are rendered as their values (``(S, 32)``, never ``(S, NH)``).
* In JSON, a non-finite float is the tagged value ``{"float": "inf"}`` /
  ``"-inf"`` / ``"nan"``, so the output is always strict JSON.
* A constexpr declared ``const[f32]`` records its dtype (``"dtype": "f32"``), so it differs from
  an untyped ``const`` with the same value after a round trip. Its compiled-in value follows the
  dtype: a float dtype is always a float (``const[f32] = S`` in bucket 16 is ``16.0``), an integer
  dtype always an int (a non-integral value is an error). An untyped ``const`` keeps its value as is.

A spec that was read back from JSON is not trusted: ``from_dict`` checks that it is well formed
and self-consistent, which does not mean it is the spec of any particular declaration (a
self-consistent edit, such as a constexpr value changed in ``constants`` and in ``args`` alike,
still validates and would select a different binary). Before compiling from or launching with a
stored spec, call ``spec.verify_against(op)``: it regenerates the spec from the declaration and
requires an exact match, naming the first field that differs. ``ast_source_inputs()`` enforces
this: it raises on a spec that is not :attr:`~CompileSpec.verified`. Specs from ``compile_specs`` are
verified; specs from ``from_dict`` / ``from_json`` start unverified and become verified after a
successful ``verify_against`` (editing one afterwards, even inside its dicts, withdraws it). The flag
is never serialized. Every spec also carries
``declaration_digest``, a sha256 over the canonical declaration (see :func:`declaration_digest`),
which a tool without the declaration at hand can compare.

Schema 2 (this version) adds the digest and the constexpr dtypes to schema 1. Schema 1 specs are
still read, but they have neither, so they are *unverifiable*: ``verify_against`` refuses them
and the spec has to be regenerated with :func:`compile_specs`.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterator

from ._specvalid import (
    ALIGNMENT,
    LEGACY_SCHEMA,
    SCHEMA_VERSION,
    decode_scalar,
    validate_arg,
    validate_launch_param,
    validate_precondition,
    validate_spec,
)
from .decorators import ExternOp, grid_to_string
from .dims import Dim, DimExpr, _NextPow2
from .dtypes import BlockScaled
from .errors import TilefishError, TilefishSpecError
from .signature import MISSING


@contextlib.contextmanager
def _malformed() -> Iterator[None]:
    """Serialized specs are untrusted: anything that is not a tilefish error becomes a TilefishSpecError."""
    try:
        yield
    except TilefishError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError, RecursionError) as e:
        raise TilefishSpecError(f"malformed compile spec: {type(e).__name__}: {e}") from None


def _encode(v: Any) -> Any:
    """A scalar for JSON: non-finite floats become ``{"float": "inf" | "-inf" | "nan"}``."""
    if isinstance(v, float) and not math.isfinite(v):
        return {"float": "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")}
    return v


@dataclass(frozen=True)
class Precondition:
    """One condition a launcher must check before it may run this binary.

    ``kind`` is one of

    * ``"pointer_alignment"``: argument ``arg`` (index ``index``) is aligned to ``bytes``;
    * ``"divisible"``: runtime argument ``arg`` (carrying ``dim``) is a multiple of ``by``;
    * ``"static_size"``: axis ``axis`` of tensor argument ``arg`` equals ``value``;
    * ``"bucket"``: dimension ``dim`` equals ``value`` exactly. ``value`` is the bucket this
      binary was compiled for and ``buckets`` lists every bucket, so a host can pick the
      smallest one that is at least the real size and pad up to it.
    """

    kind: str
    arg: str | None = None
    index: int | None = None
    axis: int | None = None
    value: int | None = None
    dim: str | None = None
    by: int | None = None
    bytes: int | None = None
    buckets: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in ("kind", "arg", "index", "axis", "value", "dim", "by", "bytes", "buckets"):
            v = getattr(self, k)
            if v is not None:
                out[k] = list(v) if isinstance(v, tuple) else v
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Precondition":
        """The inverse of :meth:`to_dict`; strict about fields and types."""
        with _malformed():
            validate_precondition(d)
            kw = dict(d)
            if kw.get("buckets") is not None:
                kw["buckets"] = tuple(kw["buckets"])
            return cls(**kw)


@dataclass(frozen=True)
class ArgSpec:
    """One Triton kernel parameter, with the information a host needs to bind it."""

    index: int
    name: str
    kind: str  # "pointer" | "scalar" | "constexpr"
    type: str  # the Triton signature entry
    access: str | None = None  # pointers: "in" | "inout" | "out"
    dtype: str | None = None  # pointers: element dtype; constexprs: the declared const[dtype], if any
    shape: tuple[str, ...] | None = None  # pointers: dimension expressions, as strings (static dims as values)
    dim: str | None = None  # scalars: the dimension expression whose value is passed
    value: Any = None  # constexprs: the compiled-in value

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"index": self.index, "name": self.name, "kind": self.kind, "type": self.type}
        for k in ("access", "dtype", "dim"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        if self.value is not None:
            out["value"] = _encode(self.value)
        if self.shape is not None:
            out["shape"] = list(self.shape)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArgSpec":
        """The inverse of :meth:`to_dict`; strict about fields and types."""
        with _malformed():
            validate_arg(d)
            kw = dict(d)
            if kw.get("shape") is not None:
                kw["shape"] = tuple(kw["shape"])
            if "value" in kw:
                kw["value"] = decode_scalar(kw["value"], "arg value")
            return cls(**kw)


@dataclass(frozen=True)
class LaunchParam:
    """One slot of the launch parameter list (``value`` is set for fixed slots)."""

    name: str
    type: str
    value: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.value is not None:
            out["value"] = self.value
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LaunchParam":
        with _malformed():
            validate_launch_param(d)
            return cls(**d)


@dataclass(frozen=True)
class AstSourceInputs:
    """Exactly what to hand to ``triton.compile``, with no reshaping.

    ``signature`` is name-keyed; ``attrs`` and ``constants`` are keyed by
    ``(arg_index,)`` tuples. ``signature_by_index`` is the same signature keyed
    by argument position, for Triton versions that expect it.
    """

    signature: dict[str, str]
    attrs: dict[tuple[int, ...], list[list[Any]]]
    constants: dict[tuple[int, ...], Any]
    options: dict[str, int]

    @property
    def signature_by_index(self) -> dict[int, str]:
        return dict(enumerate(self.signature.values()))


def _key_to_str(k: tuple[int, ...]) -> str:
    return ",".join(str(i) for i in k)


def _key_from_str(k: str) -> tuple[int, ...]:
    return tuple(int(i) for i in k.split(","))


@dataclass(frozen=True)
class CompileSpec:
    """Everything needed to compile, launch and bind one binary of an ``extern`` Triton kernel."""

    stem: str  # op name plus ``.<Dim><value>`` per bucketed dim, e.g. "qkv_rope.S32"
    op: str
    triton: str  # "module:function"
    signature: dict[str, str]
    constants: dict[tuple[int, ...], Any]
    attrs: dict[tuple[int, ...], list[list[Any]]]
    num_warps: int
    num_stages: int
    grid: str  # string expression over dimension names, e.g. "(cdiv(n, 1024),)"
    block: tuple[int, int, int]
    launch_params: tuple[LaunchParam, ...]
    preconditions: tuple[Precondition, ...]
    args: tuple[ArgSpec, ...]
    buckets: dict[str, int] = field(default_factory=dict)  # bucketed dim name -> bucket value of this binary
    # sha256 of the canonical declaration this spec was generated from (see declaration_digest). ``None`` for
    # a spec read from schema 1, which has none: such a spec cannot be verified, only regenerated.
    declaration_digest: str | None = None
    # A fingerprint of the content this spec had when it was verified (None: not verified). Not an init
    # argument, so ``dataclasses.replace`` never carries it over to an edited copy; not compared, not
    # shown, and not in to_dict, so it never round-trips as "verified".
    _verified: str | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def triton_module(self) -> str:
        return self.triton.split(":", 1)[0]

    @property
    def triton_function(self) -> str:
        return self.triton.split(":", 1)[1]

    def _fingerprint(self) -> str:
        """A hash of everything that is serialized, types included (``1``, ``1.0`` and ``True`` differ).

        Object keys are sorted, except where their order means something: ``signature`` is the argument
        order Triton sees (``constants`` and ``attrs`` are keyed by index into it) and ``buckets`` is the
        order of the bucketed dimensions. Those are hashed as lists of items, so reordering them in place
        changes the fingerprint. Every other order-carrying field (``args``, ``launch_params``,
        ``preconditions``, the entries of ``grid``) is already a list or a string in ``to_dict``.
        """
        doc = self.to_dict()
        doc["signature"] = [[k, v] for k, v in self.signature.items()]
        doc["buckets"] = [[k, v] for k, v in self.buckets.items()]
        return hashlib.sha256(json.dumps(doc, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()

    @property
    def verified(self) -> bool:
        """Whether this spec is known to be what its declaration generates, and has not changed since.

        A spec from :func:`compile_specs` is verified. A spec from :meth:`from_dict` or :meth:`from_json`
        starts unverified and becomes verified after a successful :meth:`verify_against`. Editing it
        (including the dicts inside it) afterwards makes it unverified again.
        """
        if self._verified is None:
            return False
        try:
            return self._fingerprint() == self._verified
        except (TypeError, ValueError):
            return False

    def _mark_verified(self) -> None:
        object.__setattr__(self, "_verified", self._fingerprint())

    def ast_source_inputs(self) -> AstSourceInputs:
        """The ``signature``/``attrs``/``constants`` and compile options, as fresh copies.

        Only a :attr:`verified` spec may be used to compile: a spec that came from ``from_dict`` or
        ``from_json`` must pass :meth:`verify_against` first, otherwise this raises
        :class:`TilefishSpecError`. (Specs from :func:`compile_specs` are verified already.)
        """
        if not self.verified:
            if self._verified is not None:
                raise TilefishSpecError(
                    f"compile spec '{self.stem}' was modified after it was verified; call spec.verify_against(op) "
                    "again (or regenerate it with compile_specs(op))"
                )
            if not self.verifiable:
                raise TilefishSpecError(
                    f"compile spec '{self.stem}' is schema {LEGACY_SCHEMA} and has not been verified, and it cannot "
                    "be: regenerate it with compile_specs(op)"
                )
            raise TilefishSpecError(
                f"compile spec '{self.stem}' has not been verified against its declaration, so it must not be "
                "compiled: call spec.verify_against(op) first (a spec from compile_specs(op) is already verified)"
            )
        return AstSourceInputs(
            signature=dict(self.signature),
            attrs=copy.deepcopy(self.attrs),
            constants=dict(self.constants),
            options={"num_warps": self.num_warps, "num_stages": self.num_stages},
        )

    @property
    def schema(self) -> int:
        """The serialization schema of this spec: 2, or 1 for a spec read back from schema 1."""
        return LEGACY_SCHEMA if self.declaration_digest is None else SCHEMA_VERSION

    @property
    def verifiable(self) -> bool:
        """Whether :meth:`verify_against` can check this spec (it needs a declaration digest, so not schema 1)."""
        return self.declaration_digest is not None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-compatible manifest skeleton (no machine-code hashes yet).

        Tuple keys become strings: ``{(3,): 1024}`` is ``{"3": 1024}``. A
        non-finite float is encoded as ``{"float": "inf" | "-inf" | "nan"}``.
        ``declaration_digest`` hashes the *declaration*, not a binary. A spec read from schema 1
        is written back as schema 1 (it has no digest).
        """
        doc: dict[str, Any] = {
            "schema": self.schema,
            "stem": self.stem,
            "op": self.op,
            "triton": {"module": self.triton_module, "function": self.triton_function, "ref": self.triton},
            "buckets": dict(self.buckets),
            "args": [a.to_dict() for a in self.args],
            "signature": dict(self.signature),
            "constants": {_key_to_str(k): _encode(v) for k, v in self.constants.items()},
            "attrs": {_key_to_str(k): copy.deepcopy(v) for k, v in self.attrs.items()},
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "grid": self.grid,
            "block": list(self.block),
            "launch_params": [p.to_dict() for p in self.launch_params],
            "preconditions": [p.to_dict() for p in self.preconditions],
        }
        if self.declaration_digest is not None:
            doc["declaration_digest"] = self.declaration_digest
        return doc

    def to_json(self, indent: int | None = 2) -> str:
        """Strict JSON (never ``NaN``/``Infinity``)."""
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompileSpec":
        """The inverse of :meth:`to_dict`, for untrusted input.

        Besides strict fields and types, the whole spec must be consistent: the signature agrees
        with the args, constants sit at the constexpr indices, attrs are exactly the pointers and
        the runtime scalars with a divisibility precondition (value ``[["tt.divisibility", 16]]``),
        ``num_warps`` is a power of two up to 32 and ``block`` follows it, the launch parameters
        are the non-constexpr args plus the two zero scratch slots, the Triton reference is
        ``module.path:identifier``, and the grid parses under a restricted grammar (ints, this
        spec's dimension names, ``cdiv(a, b)``, ``+ - * //``, parentheses; it is never evaluated, and a
        static entry below 1 is rejected like ``compile_specs`` does). Any violation is a
        :class:`TilefishSpecError`.

        Schemas 1 and 2 are read. A schema 1 spec has no declaration digest and no constexpr dtypes,
        and is marked unverifiable (:attr:`verifiable`); schema 2 keeps its digest. Being valid does not
        make a spec the one a declaration generates: use :meth:`verify_against` for that.
        """
        with _malformed():
            validate_spec(d)
            return cls(
                stem=d["stem"],
                op=d["op"],
                triton=d["triton"]["ref"],
                signature=dict(d["signature"]),
                constants={_key_from_str(k): decode_scalar(v, "constant") for k, v in d["constants"].items()},
                attrs={_key_from_str(k): copy.deepcopy(v) for k, v in d["attrs"].items()},
                num_warps=d["num_warps"],
                num_stages=d["num_stages"],
                grid=d["grid"],
                block=tuple(d["block"]),  # type: ignore[arg-type]
                launch_params=tuple(LaunchParam.from_dict(p) for p in d["launch_params"]),
                preconditions=tuple(Precondition.from_dict(p) for p in d["preconditions"]),
                args=tuple(ArgSpec.from_dict(a) for a in d["args"]),
                buckets=dict(d["buckets"]),
                declaration_digest=d.get("declaration_digest"),
            )

    def verify_against(self, op: ExternOp) -> None:
        """Require this spec to be exactly what ``compile_specs(op)`` generates for its binary.

        :meth:`from_dict` only checks that a spec is well formed and consistent with itself. A
        self-consistent edit (a constexpr value changed in ``constants`` and ``args``, a pointer dtype
        changed in every field that mentions it) still validates, and would select a different cubin.
        This regenerates the spec from the declaration and compares every field, values and their types
        included (``1``, ``1.0`` and ``True`` differ; so do ``0.0`` and ``-0.0``), then the declaration
        digest. A mismatch is a :class:`TilefishSpecError` naming the first field that differs, such as
        ``args[3].value``.

        A spec read from schema 1 has no digest and no constexpr dtypes, so it cannot be verified:
        this raises, and the spec has to be regenerated with :func:`compile_specs`.

        On success the spec becomes :attr:`verified`, which :meth:`ast_source_inputs` requires; a failed
        call leaves it unverified.
        """
        object.__setattr__(self, "_verified", None)
        if self.declaration_digest is None:
            raise TilefishSpecError(
                f"compile spec '{self.stem}' is schema {LEGACY_SCHEMA}: it has no declaration digest and no "
                f"constexpr dtypes, so it cannot be verified against a declaration; regenerate it with "
                "compile_specs(op)"
            )
        generated = {s.stem: s for s in compile_specs(op)}
        expected = generated.get(self.stem)
        if expected is None:
            # name the field: `op` when this is another operator's spec, else `stem`
            field_name = "op" if self.op != op.name else "stem"
            raise TilefishSpecError(
                f"compile spec '{self.stem}' does not match the declaration of extern op '{op.name}': "
                f"first difference at {field_name}: the spec has {getattr(self, field_name)!r}, the declaration "
                f"gives binaries {sorted(generated)}"
            )
        found = _first_difference(expected.to_dict(), self.to_dict(), "")
        if found is not None:
            path, want, got = found
            raise TilefishSpecError(
                f"compile spec '{self.stem}' does not match the declaration of extern op '{op.name}': "
                f"first difference at {path}: the spec has {_show(got)}, the declaration gives {_show(want)}"
            )
        self._mark_verified()

    @classmethod
    def from_json(cls, text: str) -> "CompileSpec":
        """Parse strict JSON (``NaN`` and ``Infinity`` literals are rejected) and validate it like :meth:`from_dict`."""

        def reject(constant: str) -> Any:
            raise TilefishSpecError(f"malformed compile spec: the JSON constant {constant} is not allowed")

        with _malformed():
            doc = json.loads(text, parse_constant=reject)
        return cls.from_dict(doc)


_ABSENT = object()  # a key one side does not have


def _same_scalar(a: Any, b: Any) -> bool:
    """Equal and of the same type: ``1``, ``1.0`` and ``True`` differ, ``0.0`` and ``-0.0`` differ, ``nan`` equals itself."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return repr(a) == repr(b)
    return bool(a == b)


def _first_difference(want: Any, got: Any, path: str) -> "tuple[str, Any, Any] | None":
    """Where two JSON-like documents first differ, as ``(path, wanted, found)``; ``None`` when identical.

    Dicts are walked in the order of ``want``; lists element by element, then by length. The ``signature``
    is also compared in order: its order is the argument order Triton sees.
    """
    if isinstance(want, dict) and isinstance(got, dict):
        for k, v in want.items():
            sub = f"{path}.{k}" if path else str(k)
            if k not in got:
                return sub, v, _ABSENT
            found = _first_difference(v, got[k], sub)
            if found is not None:
                return found
        for k, v in got.items():
            if k not in want:
                return (f"{path}.{k}" if path else str(k)), _ABSENT, v
        if path == "signature" and list(want) != list(got):
            return f"{path} (order)", list(want), list(got)
        return None
    if isinstance(want, list) and isinstance(got, list):
        for i, (a, b) in enumerate(zip(want, got)):
            found = _first_difference(a, b, f"{path}[{i}]")
            if found is not None:
                return found
        if len(want) != len(got):
            return f"{path} (length)", len(want), len(got)
        return None
    if type(want) is not type(got) or not _same_scalar(want, got):
        return path, want, got
    return None


def _show(v: Any) -> str:
    if v is _ABSENT:
        return "nothing"
    text = repr(v)
    return text if len(text) <= 100 else text[:97] + "..."


# -- the declaration digest ----------------------------------------------------------------------------------------------

DIGEST_FORMAT = 1  # bump when the canonical form below changes: every digest changes with it


def _canonical_dim(d: DimExpr) -> Any:
    """A dimension expression as plain data: its constant, then its ``(atom, coefficient)`` terms in canonical order.

    A static ``Dim`` is its value (its name never reaches a host). A symbolic ``Dim`` keeps everything that
    changes a spec: its name, ``divisible_by`` and its buckets.
    """
    d = d.folded()
    return {"const": d.offset, "terms": [[_canonical_atom(a), c] for a, c in d.terms]}


def _canonical_atom(a: DimExpr) -> Any:
    if isinstance(a, Dim):
        return {"dim": a.name, "divisible_by": a.divisible_by, "buckets": None if a.buckets is None else list(a.buckets)}
    if isinstance(a, _NextPow2):  # the BLOCK constexpr a lowered kernel derives from a bucketed dimension
        return {"next_pow2": _canonical_dim(a.arg)}
    return {"cdiv": [_canonical_dim(a.num), _canonical_dim(a.den)]}  # type: ignore[attr-defined]


def _canonical_dtype(t: Any) -> Any:
    if isinstance(t, BlockScaled):
        return {
            "blockscaled": {
                "elem": _canonical_dtype(t.elem),
                "scale": _canonical_dtype(t.scale),
                "block": t.block,
                "axis": t.axis,
                "global_scale": None if t.global_scale is None else _canonical_dtype(t.global_scale),
            }
        }
    return {"name": t.name, "kind": t.kind, "bits": t.bits, "pointer": t.triton_ptr, "scalar": t.triton_scalar}


def _typed_const(op_name: str, p: Any, value: Any) -> Any:
    """The compiled-in value of a constexpr, following its declared dtype (an untyped const is unchanged).

    ``const[f32]`` is always a float (``16`` from a bucket becomes ``16.0``); an integer dtype is always an
    int, and a value with a fractional part or no finite value is rejected.
    """
    dtype = p.const_dtype
    if dtype is None or isinstance(value, bool):
        return value
    try:
        if dtype.is_float:
            return float(value)
        if dtype.is_int:
            if isinstance(value, float) and not value.is_integer():
                raise TilefishSpecError(
                    f"extern op '{op_name}': constexpr '{p.name}' is const[{dtype}] but its value {value!r} is not an integer"
                )
            return int(value)
    except OverflowError:
        raise TilefishSpecError(
            f"extern op '{op_name}': constexpr '{p.name}': the value {value!r} does not fit const[{dtype}]"
        ) from None
    return value


def _canonical_value(v: Any) -> Any:
    """A constexpr default, tagged with its python type: ``True``, ``1`` and ``1.0`` are three different constants."""
    if v is MISSING:
        return {"missing": True}
    if isinstance(v, bool):
        return {"bool": v}
    if isinstance(v, int):
        return {"int": v}
    if isinstance(v, float):
        return {"float": repr(v)}  # exact, and finite or not: 'inf', 'nan', '-0.0', '1e-06'
    if isinstance(v, DimExpr):
        return {"dim": _canonical_dim(v)}
    raise TilefishSpecError(f"cannot digest the constexpr value {v!r} ({type(v).__name__})")  # pragma: no cover


def _canonical_declaration(op: ExternOp) -> dict[str, Any]:
    """Every part of an ``extern`` declaration that reaches a compile spec, as plain data.

    That is what feeds ``ASTSource`` (the parameter names, kinds and pointer types, the constexpr values,
    the divisibility of the runtime dimensions), the launch (grid, ``num_warps``, ``num_stages``) and the
    preconditions (shapes, ``divisible_by``, buckets), plus the operator and Triton kernel names. It leaves
    out what no spec can show: call-site conventions such as keyword-only parameters, docstrings, the stub body.
    """
    params: list[dict[str, Any]] = []
    for p in op.signature.params:
        entry: dict[str, Any] = {"index": p.index, "name": p.name, "kind": p.kind}
        if p.kind == "tensor":
            assert p.type is not None
            entry["access"] = p.access
            entry["dtype"] = _canonical_dtype(p.type.dtype)
            entry["shape"] = [_canonical_dim(d) for d in p.type.shape]
        elif p.kind == "dim":
            assert p.dim is not None
            entry["dim"] = _canonical_dim(p.dim)
        else:
            entry["dtype"] = None if p.const_dtype is None else _canonical_dtype(p.const_dtype)
            # a scalar default is hashed as the typed value a spec compiles in; a dimension expression is
            # resolved per bucket (and typed then), so it is hashed with its dtype above
            typed = p.default if isinstance(p.default, DimExpr) or p.default is MISSING else _typed_const(op.name, p, p.default)
            entry["default"] = _canonical_value(typed)
        params.append(entry)
    return {
        "format": DIGEST_FORMAT,
        "op": op.name,
        "triton": op.triton,
        "num_warps": op.num_warps,
        "num_stages": op.num_stages,
        "grid": None if op.grid is None else [_canonical_dim(g) for g in op.grid],
        "params": params,
    }


def declaration_digest(op: ExternOp) -> str:
    """A stable sha256 (64 lowercase hex digits) over the canonical declaration of an ``extern`` op.

    The digest covers every field that affects the ``ASTSource`` inputs or the launch: the Triton kernel
    reference, ``num_warps``, ``num_stages``, the grid, and for each parameter, in order, its name and kind
    and either the tensor's access, dtype and shape, the runtime dimension, or the constexpr's declared
    dtype and default value *with its type* (``const[f32] = 1.0``, ``const = 1.0`` and ``const = 1`` all
    differ). Dimensions are hashed with their ``divisible_by`` and ``buckets``. It is computed from the
    declaration alone, so it is the same for all bucket variants of one op, and it is independent of
    dict ordering, of ``hash()`` and of the Python version.
    """
    if not isinstance(op, ExternOp):
        raise TilefishSpecError(f"declaration_digest takes an extern op, got {op!r}")
    text = json.dumps(_canonical_declaration(op), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bucket_dims(op: ExternOp) -> list[Dim]:
    """Bucketed dims used by const defaults, in order of first appearance."""
    dims: list[Dim] = []
    for p in op.signature.consts:
        if isinstance(p.default, DimExpr):
            for d in sorted(p.default.free_dims(), key=lambda x: x.name):
                if not d.buckets:
                    raise TilefishSpecError(
                        f"extern op '{op.name}': constexpr '{p.name}' defaults to {p.default}, which depends on the "
                        f"dynamic dimension {d} that has no buckets; give it Dim({d.name!r}, buckets=[...]) or a static value"
                    )
                if d not in dims:
                    dims.append(d)
    return dims


def _build(op: ExternOp, mapping: dict[Dim, int], digest: str) -> CompileSpec:
    signature: dict[str, str] = {}
    constants: dict[tuple[int, ...], Any] = {}
    attrs: dict[tuple[int, ...], list[list[Any]]] = {}
    preconditions: list[Precondition] = []
    args: list[ArgSpec] = []

    for p in op.signature.params:
        i = p.index
        if p.kind == "tensor":
            t = p.type
            assert t is not None
            if isinstance(t.dtype, BlockScaled):
                raise TilefishSpecError(
                    f"extern op '{op.name}': parameter '{p.name}' is {t}; block-scaled tensors (data plus scale "
                    "pointers) are not supported in AOT specs yet"
                )
            ptr = t.dtype.triton_ptr
            signature[p.name] = ptr
            attrs[(i,)] = [["tt.divisibility", ALIGNMENT]]
            preconditions.append(Precondition("pointer_alignment", arg=p.name, index=i, bytes=ALIGNMENT))
            for axis, d in enumerate(t.shape):
                if d.is_static:
                    preconditions.append(Precondition("static_size", arg=p.name, index=i, axis=axis, value=d.value))
            shape = tuple(str(d.subst(mapping).folded()) for d in t.shape)  # bucket values and static dims, folded
            args.append(ArgSpec(i, p.name, "pointer", ptr, access=p.access, dtype=str(t.dtype), shape=shape))
        elif p.kind == "dim":
            assert p.dim is not None
            signature[p.name] = "i32"
            if p.dim.is_divisible_by(ALIGNMENT):
                attrs[(i,)] = [["tt.divisibility", ALIGNMENT]]
                preconditions.append(Precondition("divisible", arg=p.name, index=i, dim=str(p.dim), by=ALIGNMENT))
            args.append(ArgSpec(i, p.name, "scalar", "i32", dtype="i32", dim=str(p.dim)))
        else:
            assert p.default is not MISSING
            value = p.default
            if isinstance(value, DimExpr):
                resolved = value.subst(mapping)
                if not resolved.is_static:  # pragma: no cover - _bucket_dims rejects this earlier
                    raise TilefishSpecError(f"extern op '{op.name}': constexpr '{p.name}' is not static: {resolved}")
                value = resolved.value
            value = _typed_const(op.name, p, value)  # const[f32] = S compiles in 16.0, not 16
            signature[p.name] = "constexpr"
            constants[(i,)] = value
            dtype = None if p.const_dtype is None else str(p.const_dtype)  # the declared const[dtype], if any
            args.append(ArgSpec(i, p.name, "constexpr", "constexpr", dtype=dtype, value=value))

    for d, v in mapping.items():
        preconditions.append(Precondition("bucket", dim=d.name, value=v, buckets=d.buckets))

    assert op.grid is not None
    grid = tuple(g.subst(mapping).folded() for g in op.grid)  # exact buckets: the bucket value, folded
    for g in grid:
        if g.is_static and g.value < 1:
            raise TilefishSpecError(f"extern op '{op.name}': the grid has a non-positive size {g} for {mapping}")

    launch = [LaunchParam(a.name, a.type) for a in args if a.kind != "constexpr"]
    launch += [LaunchParam("global_scratch", "u64", 0), LaunchParam("profile_scratch", "u64", 0)]
    stem = op.name + "".join(f".{d.name}{v}" for d, v in mapping.items())
    spec = CompileSpec(
        stem=stem,
        op=op.name,
        triton=op.triton,
        signature=signature,
        constants=constants,
        attrs=attrs,
        num_warps=op.num_warps,
        num_stages=op.num_stages,
        grid=grid_to_string(grid),
        block=(32 * op.num_warps, 1, 1),
        launch_params=tuple(launch),
        preconditions=tuple(preconditions),
        args=tuple(args),
        buckets={d.name: v for d, v in mapping.items()},
        declaration_digest=digest,
    )
    spec._mark_verified()  # generated from the declaration just now
    return spec


def compile_specs(op: ExternOp) -> list[CompileSpec]:
    """One :class:`CompileSpec` per bucket combination of an ``extern`` op (one for an unbucketed op)."""
    if not isinstance(op, ExternOp):
        raise TilefishSpecError(f"compile_specs takes an extern op, got {op!r}")
    if op.grid is None:
        raise TilefishSpecError(f"extern op '{op.name}' has no grid; pass grid=... to @tilefish.extern")
    dims = _bucket_dims(op)
    combos = itertools.product(*(d.buckets for d in dims)) if dims else [()]  # type: ignore[arg-type]
    digest = declaration_digest(op)
    return [_build(op, dict(zip(dims, combo)), digest) for combo in combos]
