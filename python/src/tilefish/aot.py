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
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterator

from ._specvalid import (
    ALIGNMENT,
    SCHEMA_VERSION,
    decode_scalar,
    validate_arg,
    validate_launch_param,
    validate_precondition,
    validate_spec,
)
from .decorators import ExternOp, grid_to_string
from .dims import Dim, DimExpr
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
    dtype: str | None = None  # pointers: element dtype
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

    @property
    def triton_module(self) -> str:
        return self.triton.split(":", 1)[0]

    @property
    def triton_function(self) -> str:
        return self.triton.split(":", 1)[1]

    def ast_source_inputs(self) -> AstSourceInputs:
        """The ``signature``/``attrs``/``constants`` and compile options, as fresh copies."""
        return AstSourceInputs(
            signature=dict(self.signature),
            attrs=copy.deepcopy(self.attrs),
            constants=dict(self.constants),
            options={"num_warps": self.num_warps, "num_stages": self.num_stages},
        )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-compatible manifest skeleton (no machine-code hashes yet).

        Tuple keys become strings: ``{(3,): 1024}`` is ``{"3": 1024}``. A
        non-finite float is encoded as ``{"float": "inf" | "-inf" | "nan"}``.
        """
        return {
            "schema": SCHEMA_VERSION,
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
        spec's dimension names, ``cdiv(a, b)``, ``+ - * //``, parentheses; it is never evaluated).
        Any violation is a :class:`TilefishSpecError`.
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
            )

    @classmethod
    def from_json(cls, text: str) -> "CompileSpec":
        """Parse strict JSON (``NaN`` and ``Infinity`` literals are rejected) and validate it like :meth:`from_dict`."""

        def reject(constant: str) -> Any:
            raise TilefishSpecError(f"malformed compile spec: the JSON constant {constant} is not allowed")

        with _malformed():
            doc = json.loads(text, parse_constant=reject)
        return cls.from_dict(doc)


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


def _build(op: ExternOp, mapping: dict[Dim, int]) -> CompileSpec:
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
            signature[p.name] = "constexpr"
            constants[(i,)] = value
            args.append(ArgSpec(i, p.name, "constexpr", "constexpr", value=value))

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
    return CompileSpec(
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
    )


def compile_specs(op: ExternOp) -> list[CompileSpec]:
    """One :class:`CompileSpec` per bucket combination of an ``extern`` op (one for an unbucketed op)."""
    if not isinstance(op, ExternOp):
        raise TilefishSpecError(f"compile_specs takes an extern op, got {op!r}")
    if op.grid is None:
        raise TilefishSpecError(f"extern op '{op.name}' has no grid; pass grid=... to @tilefish.extern")
    dims = _bucket_dims(op)
    combos = itertools.product(*(d.buckets for d in dims)) if dims else [()]  # type: ignore[arg-type]
    return [_build(op, dict(zip(dims, combo))) for combo in combos]
