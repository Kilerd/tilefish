"""Static launch plans: a ``@tilefish.graph`` compiled into an ordered list of launches over one memory arena, per shape bucket.

``tilefish.plan(graph, buckets=...)`` walks the traced graph once and makes, for every combination of the bucket values of the graph's
dimensions,

* **steps**: one launch of a manifest entry each (a Triton kernel, a prebuilt PTX or cubin, or a pinned cuBLASLt GEMM), in the order of the IR.
  Every pointer argument of a step is bound to a *buffer*, possibly through a *view* (shape, element strides and an element offset, from the view
  operations ``reshape``, ``permute``, ``select`` and ``split``, which copy nothing);
* **buffers**: the graph's inputs, weights, in-place tensors and returned values (the *externals*: the caller provides them), the
  intermediates, one slot per ``DevScalar`` input and one workspace for the cuBLASLt GEMMs. Everything that is not external lives in **one
  contiguous arena**, at a byte offset that is a multiple of 256 and that is **the same in every bucket**, so a plan can be captured in a CUDA
  graph once per bucket at fixed addresses. The arena is sized for the largest bucket (each buffer for the largest it ever is);
* **no implicit synchronization**: every step goes on one stream, in order. A buffer that a step writes is written before every later reader, in
  the order of the IR (an in-place call keeps the storage of the version it consumes), and with ``reuse=True`` two intermediates share bytes only
  where no step uses both: their live ranges, from the first step that touches a buffer to the last, do not intersect.

The plan is a document (``plan.json``, schema ``tilefish.plan.v1``, specified in ``docs/plan-schema-1.md``) with a digest that includes the digest
of the manifest it was built against, so a pin of the plan digest pins every binary it launches. This module is pure Python: it needs no torch, no
Triton and no GPU. :mod:`tilefish._planrun` runs a plan.

What a plan accepts. A graph body is calls to declared operators (``kernel``, ``extern``, ``library`` with an ``impl``) and views. A traced
arithmetic op in the graph body (``x + y``, a cast, a reduction), a ``store``, a call to another ``graph``, a ``Dim`` parameter and an ``Out``
parameter are refused with the rule that says so (wrap arithmetic in a ``@tilefish.kernel``; a graph's results are its returned values). An in-place
call (``InOut``) must take a whole buffer, not a view of another tensor (the IR says a chunk that is updated does not change its parent; a plan
has no copy to make that true). A returned value must be a fresh buffer made by a call: not an input, and not a view.
"""

from __future__ import annotations

import bisect
import hashlib
import heapq
import itertools
import json
import math
import pathlib
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from . import _specvalid as sv
from ._launch import (
    TORCH_DTYPE_NAMES,
    TensorInfo,
    bind_infos,
    check_consts,
    check_dim_constraints,
    check_no_alias,
    check_spec_preconditions,
    eval_dim,
    extent_elements,
    select_spec,
    tf_dtype_name,
)
from .decorators import ExternOp, GraphOp, KernelOp, LibraryOp
from .dims import Dim, DimExpr
from .dtypes import BlockScaled
from .errors import TilefishError, TilefishManifestError, TilefishPlanError, TilefishPreconditionError
from .ir import BucketRequirement, Requirement
from .trace import Value
from .types import row_major_strides

PLAN_SCHEMA = "tilefish.plan.v1"
PLAN_NAME = "plan.json"
ARENA_ALIGNMENT = 256
DIGEST_TAG = b"tilefish.plan-digest.v1\0"  # part of what is hashed: another canonical form must change it
MAX_PLAN_BYTES = 1 << 28
MAX_BUCKETS = 1 << 12  # bucket combinations of one plan
MAX_BUFFERS = 1 << 15
MAX_STEPS = 1 << 20  # per bucket
MAX_ARGS = 256  # pointer arguments of one step

ROLES = ("input", "output", "inout", "weight", "intermediate", "devscalar", "workspace")
EXTERNAL_ROLES = ("input", "output", "inout", "weight")
ARENA_ROLES = ("intermediate", "devscalar", "workspace")
WRITABLE_ROLES = ("intermediate", "output", "inout")  # what a step may write
TOP_KEYS = "schema tilefish graph arch manifest_digest dims reuse arena_alignment arena_bytes buffers outputs buckets plan_digest".split()
BUFFER_KEYS = "name role dtype shape offset bytes".split()
STEP_KEYS = "entry kind args".split()
KINDS = ("kernel", "library")
_VIEW_OPS = ("reshape", "permute", "select", "split")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ARCH = re.compile(r"sm_[0-9]{2,3}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _is_int(v: Any) -> bool:
    return type(v) is int


def align_up(n: int, to: int = ARENA_ALIGNMENT) -> int:
    return -(-n // to) * to


# -- the digest ---------------------------------------------------------------------------------------------------------------------


def plan_digest(doc: "Mapping[str, Any]") -> str:
    """The sha256 (lowercase hex) of a plan document: ``sha256(b"tilefish.plan-digest.v1\\0" + canonical_json(doc without "plan_digest" and "tilefish"))``.

    ``canonical_json`` is the one of the manifest digest: object keys sorted, separators ``","`` and ``":"``, no whitespace, non-ASCII escaped,
    no ``NaN`` (a plan has only strings, integers, booleans and ``null``). The top-level ``tilefish`` (the version that wrote the file) is not
    digested, as it is not in the manifest's. The plan includes ``manifest_digest``, so the digest pins every binary the plan launches.
    """
    try:
        body = {k: v for k, v in doc.items() if k not in ("plan_digest", "tilefish")}
        text = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, AttributeError, RecursionError) as e:  # RecursionError: a document nested deeper than the encoder can walk
        raise TilefishManifestError(f"the plan cannot be digested, it is not plain JSON data: {type(e).__name__}: {str(e)[:200]}") from e
    return hashlib.sha256(DIGEST_TAG + text.encode("utf-8")).hexdigest()


# -- what a graph calls -------------------------------------------------------------------------------------------------------------


@dataclass(eq=False)
class Callee:
    """A declared operator a graph calls, with what a step needs to bind and check a launch of it.

    ``kind`` is ``"kernel"`` (a Triton kernel or a prebuilt binary: a kernel, a lowered ``@tilefish.kernel`` or an extern) or ``"library"`` (a pinned
    cuBLASLt GEMM). ``specs`` are its compile specs (``CompileSpec``) or library specs (``LibrarySpec``), one per bucket.
    """

    kind: str
    op: Any
    name: str
    public: Any  # the Signature a caller sees
    target: Any  # runtime._Target for a kernel / extern, else None
    extern: "ExternOp | None"  # what is launched for a kernel / extern
    names: "dict[str, str]"  # public parameter name -> the launched kernel's parameter name
    specs: "list[Any]"

    @property
    def by_stem(self) -> "dict[str, Any]":
        return {s.stem: s for s in self.specs}


def callee_of(op: Any) -> Callee:
    """The :class:`Callee` of a declared operator (cached on the operator: an op is immutable, and no global table keeps it alive through its callee): raises
    :class:`~tilefish.errors.TilefishPlanError` for one a plan cannot launch."""
    found = getattr(op, "__dict__", {}).get("_tf_callee")
    if found is not None:
        return found
    from . import _library, runtime

    try:
        if isinstance(op, LibraryOp):
            if op.impl is None:
                raise TilefishPlanError(
                    f"the library op '{op.name}' has no implementation (impl=...), so there is nothing to launch: declare it with "
                    "impl='cublaslt.gemm' or 'cublaslt.gemm_bias'"
                )
            with runtime._lock:
                specs = runtime._library_specs.get(op)
            if specs is None:
                specs = _library.library_specs(op)
                with runtime._lock:
                    runtime._library_specs[op] = specs
            callee = Callee("library", op, op.name, op.signature, None, None, {}, list(specs))
        elif isinstance(op, (KernelOp, ExternOp)):
            target = runtime._target(op)
            extern = target.launch_extern()
            callee = Callee("kernel", op, op.name, target.public, target, extern, runtime._extern_names(target, extern), list(target.compile_specs()))
        else:
            raise TilefishPlanError(f"the {getattr(op, 'kind', type(op).__name__)} op '{getattr(op, 'name', op)}' cannot be launched by a plan")
    except TilefishPlanError:
        raise
    except TilefishError as e:
        raise TilefishPlanError(f"cannot plan the call to '{op.name}': {e}") from e
    op.__dict__["_tf_callee"] = callee
    return callee


# -- the walk: where every value lives ----------------------------------------------------------------------------------------------


@dataclass(eq=False)
class _Buffer:
    name: str
    role: str
    dtype: str  # a tilefish dtype name
    shape: "tuple[DimExpr, ...]"
    creation: int  # position in the order buffers appear, for stable names and layout
    first: int = -1  # the first and the last step that has it as an argument
    last: int = -1
    offset: "int | None" = None
    bytes: int = 0
    version: int = 0  # how many steps have written it so far, while the graph is walked (a parameter starts at 0: the caller wrote it)


@dataclass(frozen=True)
class _Loc:
    """Where a traced value lives: a buffer, and the shape, element strides and element offset of the view of it (all dimension expressions).

    ``version`` is the version of the buffer's contents the value is: the IR gives a value the contents it had when it was made (an in-place call makes a
    new version and consumes the old), but a view shares its buffer's storage, so a view that is read after its buffer was updated in place would read
    the new bytes. A step that reads a value whose ``version`` is not the buffer's current one is refused (:func:`_walk`).
    """

    buffer: _Buffer
    shape: "tuple[DimExpr, ...]"
    strides: "tuple[DimExpr, ...]"
    offset: DimExpr
    whole: bool  # the value is the buffer itself (not a view of it)
    version: int = 0


@dataclass
class _Arg:
    name: str  # the launched kernel's parameter name (a library GEMM: a, b, bias, out)
    loc: _Loc
    access: str  # "in" | "out" | "inout"


@dataclass
class _SymStep:
    callee: Callee
    args: "list[_Arg]"
    what: str  # for messages: "the call to 'rmsnorm' (node #3)"
    library: bool = False
    consts: "dict[str, Any] | None" = None  # the compile-time arguments the call gave a value other than the declared default: a binary cannot take them


@dataclass
class _Walk:
    buffers: "list[_Buffer]"
    steps: "list[_SymStep]"
    outputs: "list[_Buffer]"
    dims: "tuple[Dim, ...]"


def _brief(value: Any, limit: int = 12) -> str:
    """``repr`` of a list that is cut after ``limit`` entries (a message must not echo a list of thousands)."""
    items = list(value) if isinstance(value, (list, tuple, set, frozenset)) else None
    if items is None or len(items) <= limit:
        return repr(value if items is None else items)
    return repr(items[:limit])[:-1] + f", ... ({len(items)} in all)]"


def _fail(graph: str, message: str) -> Any:
    raise TilefishPlanError(f"graph '{graph}': {message}")


def _layout(graph: str, what: str, t: Any) -> "tuple[tuple[DimExpr, ...], DimExpr]":
    strides = t.layout_strides()
    if any(s is None for s in strides) or t.offset is None:
        _fail(graph, f"{what} has the type {t}, whose strides or offset are not affine expressions (a product of two symbolic dimensions), so it cannot be planned")
    return tuple(strides), t.offset  # type: ignore[arg-type]


def _walk(gop: GraphOp) -> _Walk:
    g = gop.graph
    name = g.name
    sig = gop.signature
    buffers: "list[_Buffer]" = []
    loc: "dict[Value, _Loc]" = {}
    steps: "list[_SymStep]" = []

    def new_buffer(role: str, dtype: Any, shape: "tuple[DimExpr, ...]", name_: str) -> _Buffer:
        try:
            tf = tf_dtype_name(dtype, f"a tensor of graph '{name}'")
        except TilefishError as e:
            raise TilefishPlanError(f"graph '{name}': {e}") from None
        b = _Buffer(name_, role, tf, tuple(shape), len(buffers))
        buffers.append(b)
        return b

    for p in sig.params:
        if p.kind == "dim":
            _fail(name, f"the runtime scalar parameter '{p.name}' is not planned (pass sizes as tensor shapes; a value the host writes is a DevScalar)")
        if p.kind == "tensor" and p.access == "out":
            _fail(name, f"the Out parameter '{p.name}' is not planned: a graph's results are its returned values, and a store into a parameter has no launch to make it")
    for v in g.inputs:
        if not _NAME.match(v.name):
            _fail(name, f"the parameter name {v.name!r} is not an ASCII identifier, which a plan needs for the name of its buffer")
        if isinstance(v.type.dtype, BlockScaled):
            _fail(name, f"the parameter '{v.name}' is block-scaled ({v.type}); block-scaled tensors are not planned yet")
        if v.type.devscalar:
            role = "devscalar"
        elif v.access == "inout":
            role = "inout"
        else:
            role = "weight" if not v.type.free_dims() else "input"
        offset = v.type.offset
        if not (v.type.contiguous and v.type.strides is None and offset is not None and offset.is_static and offset.value == 0):
            _fail(
                name,
                f"the parameter '{v.name}' has the type {v.type} ({v.type.layout_text()}), which is not a contiguous tensor: a plan binds a parameter as a "
                "contiguous row-major tensor, and a run checks that it is one (a parameter that is a view is declared Strided[...], which a graph does not take)",
            )
        b = new_buffer(role, v.type.dtype, v.type.shape, v.name)
        strides = row_major_strides(v.type.shape)
        if any(s is None for s in strides):
            _fail(name, f"the parameter '{v.name}' has the type {v.type}, whose strides are not affine expressions, so it cannot be planned")
        loc[v] = _Loc(b, tuple(v.type.shape), tuple(strides), DimExpr.const(0), True)  # type: ignore[arg-type]

    def fresh(value: Value, role: str = "intermediate") -> _Loc:
        t = value.type
        if isinstance(t.dtype, BlockScaled):
            _fail(name, f"{value.ref} is block-scaled ({t}); block-scaled tensors are not planned yet")
        b = new_buffer(role, t.dtype, t.shape, "")
        strides = row_major_strides(t.shape)
        if any(s is None for s in strides):
            _fail(name, f"{value.ref} has the type {t}, whose strides are not affine expressions, so it cannot be planned")
        b.version = 1  # the step that makes it writes it once
        return _Loc(b, tuple(t.shape), tuple(strides), DimExpr.const(0), True, 1)  # type: ignore[arg-type]

    for index, n in enumerate(g.nodes):
        kind = n.attrs.get("kind")
        where = f"the call to '{n.op}' (node #{index})" if kind else f"the op '{n.op}' (node #{index})"
        if kind is None:
            if n.op in _VIEW_OPS:
                parent = loc[n.inputs[0]]  # type: ignore[index]
                for out in n.outputs:
                    strides, offset = _layout(name, f"the view {out.ref} made by {n.op} (node #{index})", out.type)
                    loc[out] = _Loc(parent.buffer, tuple(out.type.shape), strides, offset, False, parent.version)
                continue
            if n.op == "store":
                _fail(name, f"{where} stores into the parameter '{n.attrs.get('target')}': a store is a copy, which a plan has no launch for; the last op should produce the value (return it)")
            _fail(
                name,
                f"{where} is arithmetic traced in the graph body, which a plan cannot launch (there is no kernel for it): put it in a "
                "@tilefish.kernel (a plan launches declared operators; views and the other ops are fine)",
            )
        if kind == "graph":
            _fail(name, f"{where} calls another graph, which a plan does not inline yet: write its body in this graph, or make it a kernel")
        if kind not in ("kernel", "extern", "library") or n.callee is None:
            _fail(name, f"{where} is a {kind!r} call that a plan cannot launch")
        callee = callee_of(n.callee)
        sig_op = callee.public
        ins = {p.name: v for p, v in zip(sig_op.tensor_inputs, n.inputs)}
        for pname, v in ins.items():
            l = loc[v]
            if l.version != l.buffer.version:
                _fail(
                    name,
                    f"{where} reads '{pname}' = {v.ref} ({v.type}), a view of a tensor that was updated in place after the view was made: the IR gives a view the "
                    "contents its tensor had then, but a view shares the storage, so a plan would read the updated bytes. Take the view from the updated "
                    "version (its .live), or read the view before the update",
                )
        n_ret, n_out = len(sig_op.returns), len(sig_op.tensor_outputs)
        returned, outs, inouts = n.outputs[:n_ret], n.outputs[n_ret : n_ret + n_out], n.outputs[n_ret + n_out :]
        out_locs: "dict[Value, _Loc]" = {}
        for o in (*returned, *outs):
            out_locs[o] = loc[o] = fresh(o)
        for p, version in zip((p for p in sig_op.tensor_inputs if p.access == "inout"), inouts):
            src = loc[ins[p.name]]
            if not src.whole:
                _fail(
                    name,
                    f"{where} updates '{p.name}' in place, but the argument is a view of another tensor ({ins[p.name].ref}: {ins[p.name].type}); the IR says "
                    "updating a view does not change its parent, and a plan has no copy to make that true: pass a whole tensor",
                )
            src.buffer.version += 1
            loc[version] = replace(src, version=src.buffer.version)
        args: "list[_Arg]" = []
        if callee.kind == "library":
            for p in sig_op.params:
                args.append(_Arg(p.name, loc[ins[p.name]], "in"))
            args.append(_Arg("out", loc[returned[0]], "out"))
        else:
            assert callee.extern is not None
            user = len(sig_op.params)
            appended = iter(returned)  # the stub of a lowered kernel lists its own parameters, then one Out per returned value
            outs_iter = iter(outs)
            for i, q in enumerate(callee.extern.signature.params):
                if q.kind != "tensor":
                    continue
                if i < user:
                    pp = sig_op.params[i]
                    if pp.access == "out":
                        args.append(_Arg(q.name, loc[next(outs_iter)], "out"))
                    else:
                        args.append(_Arg(q.name, loc[ins[pp.name]], pp.access))
                else:
                    args.append(_Arg(q.name, loc[next(appended)], "out"))
        given = n.attrs.get("consts") or {}
        changed = {p.name: given[p.name] for p in sig_op.consts if p.name in given and given[p.name] is not p.default}
        steps.append(_SymStep(callee, args, where, library=callee.kind == "library", consts=changed))

    outputs: "list[_Buffer]" = []
    for i, v in enumerate(g.outputs):
        l = loc[v]
        if any(l.buffer is o for o in outputs):
            _fail(name, f"the returned value {v.ref} is returned twice (a plan writes each result into one buffer of the caller's)")
        if not l.whole or l.buffer.role != "intermediate":
            why = "a view" if not l.whole else f"the {l.buffer.role} '{l.buffer.name}'"
            _fail(
                name,
                f"the returned value {v.ref} ({v.type}) is {why}, not a tensor that a call made: a plan writes a result into a buffer the caller provides, "
                "and has no copy to put an input or a view there; have an op produce it",
            )
        l.buffer.role = "output"
        outputs.append(l.buffer)
    if any(s.library for s in steps):
        words = max(s.callee.op.workspace_bytes for s in steps if s.library)
        ws = _Buffer("", "workspace", "u8", (DimExpr.const(words),), len(buffers))
        buffers.append(ws)
    dims = g.symbols
    return _Walk(buffers, steps, outputs, dims)


def _name_buffers(buffers: "list[_Buffer]", outputs: "Sequence[_Buffer]") -> None:
    """Names: a parameter keeps its own, the returned values are ``out0``, ``out1``, ... in the order the graph returns them, the rest ``t0``,
    ``t1``, ... in the order they are made, and the workspace ``workspace`` (a name that a parameter already has gets underscores)."""
    used = {b.name for b in buffers if b.name}

    def unique(base: str) -> str:
        while base in used:
            base += "_"
        used.add(base)
        return base

    for i, b in enumerate(outputs):
        b.name = unique(f"out{i}")
    count = 0
    for b in buffers:
        if b.name:
            continue
        if b.role == "workspace":
            b.name = unique("workspace")
        else:
            b.name = unique(f"t{count}")
            count += 1


# -- the dimensions and their buckets -------------------------------------------------------------------------------------------------


def _bucket_values(graph: str, dims: "Sequence[Dim]", buckets: Any) -> "dict[str, tuple[int, ...]]":
    given: "dict[str, Any]" = {}
    if buckets is not None:
        if not isinstance(buckets, Mapping):
            raise TilefishPlanError(f"graph '{graph}': buckets= maps dimension names to lists of values, such as {{'M': [32, 64]}}, got {buckets!r}")
        for k, v in buckets.items():
            key = k.name if isinstance(k, Dim) else k
            if not isinstance(key, str):
                raise TilefishPlanError(f"graph '{graph}': buckets= keys are dimension names or dimensions, got {k!r}")
            given[key] = v
        unknown = sorted(set(given) - {d.name for d in dims})
        if unknown:
            raise TilefishPlanError(
                f"graph '{graph}': buckets= names {unknown}, which are not dimensions of the graph (its dimensions: {sorted(d.name for d in dims)})"
            )
    out: "dict[str, tuple[int, ...]]" = {}
    for d in sorted(dims, key=lambda d: d.name):
        if d.name in given:
            values = given[d.name]
            try:
                if isinstance(values, (str, bytes)):
                    raise TypeError
                values = list(values)
            except TypeError:
                raise TilefishPlanError(f"graph '{graph}': buckets[{d.name!r}] is a list of sizes, got {_brief(values)}") from None
            if not values or any(not _is_int(v) or v < 1 for v in values):
                raise TilefishPlanError(f"graph '{graph}': buckets[{d.name!r}] must be a non-empty list of positive ints, got {_brief(values)}")
            chosen = tuple(sorted(set(values)))
            if d.buckets is not None and not set(chosen) <= set(d.buckets):
                raise TilefishPlanError(
                    f"graph '{graph}': buckets[{d.name!r}] = {_brief(chosen)} has values that {d.name} is not declared with (its buckets are {_brief(d.buckets)}): "
                    "an operator the graph calls may have binaries for those sizes only"
                )
        elif d.buckets is not None:
            chosen = tuple(d.buckets)
        else:
            raise TilefishPlanError(
                f"graph '{graph}': the dimension {d.name} has no buckets, and a plan has one list of steps per bucket: declare Dim({d.name!r}, buckets=[...]) "
                f"or pass buckets={{{d.name!r}: [...]}}"
            )
        if d.divisible_by:
            bad = [v for v in chosen if v % d.divisible_by]
            if bad:
                raise TilefishPlanError(f"graph '{graph}': {d.name} is declared divisible_by={d.divisible_by}, but buckets[{d.name!r}] has {_brief(bad)}")
        out[d.name] = chosen
    return out


def bucket_combinations(dims: "Mapping[str, Sequence[int]]") -> "list[dict[str, int]]":
    """Every combination of bucket values, in order: the dimensions sorted by name, the first one varying slowest, each in ascending order."""
    names = sorted(dims)
    return [dict(zip(names, combo)) for combo in itertools.product(*(sorted(dims[n]) for n in names))]


def _eval(expr: DimExpr, mapping: "Mapping[Any, int]", graph: str, what: str) -> int:
    v = eval_dim(expr, mapping)
    if v is None:
        _fail(graph, f"{what} = {expr} mentions a dimension that has no value")
    return v  # type: ignore[return-value]


def _row_major(shape: "Sequence[int]") -> "tuple[int, ...]":
    out, acc = [], 1
    for n in reversed(tuple(shape)):
        out.append(acc)
        acc *= n
    return tuple(reversed(out))


def _contiguous(shape: "Sequence[int]", strides: "Sequence[int]") -> bool:
    """Torch's ``is_contiguous``: the strides of the axes whose size is not 1 are the row-major ones."""
    expected = 1
    for n, s in reversed(list(zip(shape, strides))):
        if n != 1:
            if s != expected:
                return False
            expected *= n
    return True


# -- binding one step at one bucket ----------------------------------------------------------------------------------------------------


@dataclass(eq=False)
class Bound:
    """One step at one bucket, resolved: the binary (``spec``), the bound dimensions, the runtime scalars and the grid, and the facts about every
    pointer argument. ``infos`` have *virtual* addresses (distinct per buffer, the offset of the view inside it); a launch replaces them by the real
    ones and checks the pointer-dependent preconditions again (:meth:`check`)."""

    callee: Callee
    spec: Any
    dims: "dict[Dim, int]"
    scalars: "dict[str, int]"
    grid: "tuple[int, ...]"
    names: "tuple[str, ...]"  # the pointer arguments, in the ABI's order
    bindings: "tuple[str, ...]"  # the buffer of each
    offsets: "tuple[int, ...]"  # the byte offset of each in its buffer
    infos: "dict[str, TensorInfo]"
    written: "frozenset[str]"

    @property
    def stem(self) -> str:
        return self.spec.stem

    def check(self, bases: "Mapping[str, int]") -> "dict[str, TensorInfo]":
        """The facts with the real addresses (``bases``: buffer name -> address of its first byte), after the checks that depend on them: every
        pointer's alignment (and what else the spec's preconditions say) and that a tensor that is written overlaps no other argument."""
        real = {n: replace(self.infos[n], ptr=bases[b] + off) for n, b, off in zip(self.names, self.bindings, self.offsets)}
        if self.callee.kind == "library":
            check_no_alias(real, {"out"})
            check_spec_preconditions(self.spec, real, {}, self.dims)
        else:
            check_no_alias(real, self.written)
            check_spec_preconditions(self.spec, real, self.scalars, self.dims)
        return real


def _info(shape: "Sequence[int]", strides: "Sequence[int]", offset: int, dtype: str, base: int) -> TensorInfo:
    itemsize = _itemsize(dtype)
    return TensorInfo(
        shape=tuple(shape),
        dtype=TORCH_DTYPE_NAMES[dtype],
        ptr=base + offset * itemsize,
        contiguous=_contiguous(shape, strides),
        device="cuda:0",
        nbytes=extent_elements(shape, strides) * itemsize,
        strides=tuple(strides),
        itemsize=itemsize,
    )


ITEMSIZE = {"bf16": 2, "f16": 2, "f32": 4, "i32": 4, "i64": 8, "u8": 1, "e4m3": 1, "e5m2": 1}  # the dtypes that have a torch storage type (_launch.TORCH_DTYPE_NAMES)


def _itemsize(dtype: str) -> int:
    return ITEMSIZE[dtype]


def resolve_step(
    callee: Callee, args: "Sequence[Mapping[str, Any]]", buffers: "Mapping[str, tuple[str, tuple[int, ...]]]", where: str
) -> Bound:
    """Bind one step at one bucket: the facts about its arguments (from the document: each ``{name, binding, view?}`` against the buffers' dtype
    and shape at this bucket), the dimensions, the binary, the scalars and the grid. This is the one place that decides all of them: a plan is
    built with it, and a plan is run with it, so what was planned is what runs. Raises the errors a launch of the binary would (a precondition
    that no address can change is checked here; the address-dependent ones in :meth:`Bound.check`)."""
    infos: "dict[str, TensorInfo]" = {}
    names, bindings, offsets = [], [], []
    order = {n: i for i, n in enumerate(buffers)}
    for a in args:
        dtype, bshape = buffers[a["binding"]]
        view = a.get("view")
        if view is None:
            shape, strides, offset = tuple(bshape), _row_major(bshape), 0
        else:
            shape, strides, offset = tuple(view["shape"]), tuple(view["strides"]), view["offset"]
            if len(shape) != len(strides) or any(n < 1 for n in shape) or any(s < 0 for s in strides) or offset < 0:
                raise TilefishPlanError(f"{where}: the view of '{a['binding']}' for argument '{a['name']}' is malformed: {view}")
            if offset + extent_elements(shape, strides) > math.prod(bshape):
                raise TilefishPlanError(
                    f"{where}: the view of '{a['binding']}' for argument '{a['name']}' (shape {list(shape)}, strides {list(strides)}, offset {offset}) "
                    f"reaches past the {math.prod(bshape)} elements of the buffer"
                )
        vbase = (order[a["binding"]] + 1) << 44  # a virtual address per buffer: two arguments overlap only where views of one buffer do
        infos[a["name"]] = _info(shape, strides, offset, dtype, vbase)
        names.append(a["name"])
        bindings.append(a["binding"])
        offsets.append(offset * _itemsize(dtype))
    try:
        if callee.kind == "library":
            op = callee.op
            public = {p.name: infos[p.name] for p in op.signature.params}
            dims = bind_infos(op.name, op.signature, public)
            check_dim_constraints(dims)
            spec = select_spec(op.name, callee.specs, dims)
            want = tuple(eval_dim(d, dims) for d in op.signature.returns[0].shape)
            if infos["out"].shape != want:
                raise TilefishPlanError(f"{where}: the result of '{op.name}' has shape {list(want)}, but the buffer it is bound to has {list(infos['out'].shape)}")
            check_no_alias(infos, {"out"})
            check_spec_preconditions(spec, infos, {}, dims)
            return Bound(callee, spec, dims, {}, (), tuple(names), tuple(bindings), tuple(offsets), infos, frozenset({"out"}))
        assert callee.extern is not None
        extern = callee.extern
        public = {pp.name: infos[callee.names[pp.name]] for pp in callee.public.params if pp.kind == "tensor" and pp.access != "out"}
        dims = bind_infos(callee.name, callee.public, public)
        check_dim_constraints(dims)
        spec = select_spec(callee.name, callee.specs, dims)
        for q in extern.signature.params:  # a result is allocated by the callee's signature: the plan's buffer must be that
            if q.kind == "tensor" and q.access == "out":
                want = tuple(eval_dim(d, dims) for d in q.type.shape)  # type: ignore[union-attr]
                if infos[q.name].shape != want:
                    raise TilefishPlanError(f"{where}: the result '{q.name}' of '{callee.name}' has shape {list(want)}, but the buffer it is bound to has {list(infos[q.name].shape)}")
        from .runtime import _finish_launch

        scalars, grid = _finish_launch(extern, spec, dims, infos)
        written = frozenset(p.name for p in extern.signature.params if p.kind == "tensor" and p.access != "in")
        return Bound(callee, spec, dims, scalars, grid, tuple(names), tuple(bindings), tuple(offsets), infos, written)
    except TilefishPreconditionError as e:
        raise TilefishPreconditionError(f"{where}: {e}", kind=e.kind, arg=e.arg) from None


# -- the memory plan ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArenaItem:
    name: str
    bytes: int
    first: int  # the first and the last step that uses it
    last: int
    reusable: bool  # may share bytes with a buffer whose steps are disjoint from its own (an intermediate); a slot and the workspace live for the whole plan


def plan_arena(items: "Sequence[ArenaItem]", reuse: bool) -> "tuple[dict[str, int], int]":
    """Byte offsets (multiples of 256) of the arena buffers and the arena's size.

    Without ``reuse`` every buffer has its own region, one after the other in the order given. With it, the buffers that live for the whole plan (the
    slots and the workspace) come first, one region each, and the others are placed first-fit by the order of their first step: a buffer gets the lowest
    offset where its region meets no region of a buffer whose live range ``[first, last]`` (inclusive: a step that reads one buffer and writes
    another has both live) intersects its own. Two regions that overlap therefore belong to buffers that no step uses together, and every write of
    one comes after every use of the other in the order of the steps, which run on one stream.
    """
    offsets: "dict[str, int]" = {}
    cursor = 0
    fixed = [i for i in items if not (reuse and i.reusable)]
    for it in fixed:
        offsets[it.name] = cursor
        cursor += align_up(it.bytes)
    base = cursor
    placed: "list[tuple[int, int, ArenaItem]]" = []
    order = {i.name: n for n, i in enumerate(items)}
    for it in sorted((i for i in items if reuse and i.reusable), key=lambda i: (i.first, order[i.name])):
        size = align_up(it.bytes)
        conflicts = sorted((o, o + s) for o, s, other in placed if other.first <= it.last and it.first <= other.last)
        off = base
        for a, b in conflicts:
            if off + size <= a:
                break
            off = max(off, b)
        offsets[it.name] = off
        placed.append((off, size, it))
    end = max((offsets[i.name] + align_up(i.bytes) for i in items), default=0)
    return offsets, align_up(end)


# -- the plan --------------------------------------------------------------------------------------------------------------------------


class Plan:
    """A launch plan of a graph: the steps of every bucket and the memory plan, as the document ``plan.json`` holds them.

    ``doc`` is the document (a ``dict``; ``to_json()`` is its text), ``digest`` its digest. ``arch`` and ``manifest_digest`` are ``None`` for a plan that
    was only made (:func:`plan`, what ``jit`` runs) and the architecture and the digest of the manifest for one that was exported or loaded.
    """

    def __init__(self, graph: GraphOp, doc: "dict[str, Any]", callees: "dict[str, Callee]", arena_variants: "dict[bool, int]") -> None:
        self.graph = graph
        self._doc = doc
        self._callees = callees  # entry stem -> the operator whose binary it is
        self._arena_variants = arena_variants
        self._jit_runner: Any = None  # the runner that ``jit`` made for this plan (kept here, so that a plan and its runner are freed together)
        self._layout: "str | None" = None
        self.directory: "pathlib.Path | None" = None  # where it was exported to or loaded from (a plan that was only made has none)
        self.manifest: Any = None  # the verified manifest of that directory (:class:`~tilefish.Manifest`)

    # -- what the document says -------------------------------------------------------------------------------------------------

    @property
    def doc(self) -> "dict[str, Any]":
        """The plan document, with its ``plan_digest`` (a fresh copy: change it and the digest no longer fits)."""
        return json.loads(json.dumps(self._doc))

    @property
    def name(self) -> str:
        return self._doc["graph"]

    @property
    def arch(self) -> "str | None":
        return self._doc["arch"]

    @property
    def manifest_digest(self) -> "str | None":
        return self._doc["manifest_digest"]

    @property
    def digest(self) -> str:
        """The digest of the plan (``plan_digest``): what to pin."""
        return self._doc["plan_digest"]

    @property
    def dims(self) -> "dict[str, tuple[int, ...]]":
        return {k: tuple(v) for k, v in self._doc["dims"].items()}

    @property
    def reuse(self) -> bool:
        return self._doc["reuse"]

    @property
    def arena_bytes(self) -> int:
        """The size of the arena in bytes (with the liveness reuse this plan was made with)."""
        return self._doc["arena_bytes"]

    @property
    def arena_bytes_without_reuse(self) -> int:
        return self._arena_variants[False]

    @property
    def arena_bytes_with_reuse(self) -> int:
        return self._arena_variants[True]

    @property
    def buffers(self) -> "list[dict[str, Any]]":
        return json.loads(json.dumps(self._doc["buffers"]))

    @property
    def outputs(self) -> "list[str]":
        """The names of the buffers the graph returns, in the order it returns them."""
        return list(self._doc["outputs"])

    def buckets(self) -> "list[dict[str, int]]":
        """The bucket combinations that have steps, in the order of the document."""
        return [dict(b["values"]) for b in self._doc["buckets"]]

    def steps(self, **values: int) -> "list[dict[str, Any]]":
        """The steps of one bucket combination, as in the document (``plan.steps(M=64)``)."""
        for b in self._doc["buckets"]:
            if b["values"] == values:
                return json.loads(json.dumps(b["steps"]))
        raise TilefishPlanError(f"the plan of graph '{self.name}' has no bucket {values} (it has {self.buckets()})")

    def to_json(self) -> str:
        return json.dumps(self._doc, indent=2, allow_nan=False) + "\n"

    def describe(self) -> str:
        """A readable report: the dimensions and buckets, the arena with and without reuse, every buffer (role, shape, offset, bytes) and the steps of the largest bucket."""
        d = self._doc
        lines = [
            f"plan of graph '{self.name}' ({self.digest[:12]}): dims {self.dims}, {len(d['buckets'])} bucket(s), {len(d['buckets'][0]['steps'])} step(s)",
            f"arena: {self.arena_bytes} bytes ({'with' if self.reuse else 'without'} reuse); without reuse {self.arena_bytes_without_reuse}, with reuse {self.arena_bytes_with_reuse}",
            "buffers:",
        ]
        for b in d["buffers"]:
            place = "external" if b["offset"] is None else f"arena +{b['offset']}"
            lines.append(f"  {b['name']:<12} {b['role']:<12} {b['dtype']:<5} {b['shape']} {b['bytes']} bytes, {place}")
        last = d["buckets"][-1]
        lines.append(f"steps at {last['values'] or 'the only size'}:")
        for i, st in enumerate(last["steps"]):
            args = ", ".join(a["binding"] + ("[view]" if "view" in a else "") for a in st["args"])
            lines.append(f"  #{i} {st['entry']} ({st['kind']}): {args}")
        return "\n".join(lines)

    # -- what a runner needs ----------------------------------------------------------------------------------------------------

    @property
    def callees(self) -> "dict[str, Callee]":
        return dict(self._callees)

    @property
    def layout_key(self) -> str:
        """A digest of everything about the memory layout and the steps but the architecture and the manifest: plans that have it are one arena (a runner of any of them
        on one device uses the same memory, so an arena outlives the runner and the plan object that made it)."""
        if self._layout is None:
            body = {k: v for k, v in self._doc.items() if k not in ("plan_digest", "tilefish", "arch", "manifest_digest")}
            self._layout = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return self._layout

    def with_manifest(self, arch: str, manifest_digest: str) -> "Plan":
        """The same plan built against the manifest ``manifest_digest`` for ``arch`` (its digest changes with them)."""
        doc = {k: v for k, v in self._doc.items() if k != "plan_digest"}
        doc["arch"] = arch
        doc["manifest_digest"] = manifest_digest
        doc["plan_digest"] = plan_digest(doc)
        return Plan(self.graph, doc, self._callees, self._arena_variants)

    def with_document(self, doc: "dict[str, Any]") -> "Plan":
        """This plan's operators with another (verified) document: what a replay runs, after the document was shown to be this graph's plan."""
        return Plan(self.graph, doc, self._callees, self._arena_variants)

    def stale_reason(self, doc: "Mapping[str, Any]") -> "str | None":
        """Where ``doc`` first differs from this graph's plan (ignoring the architecture, the manifest digest, the version and the digest), or ``None``."""
        from .aot import _first_difference, _show

        skip = ("plan_digest", "tilefish", "arch", "manifest_digest")
        want = {k: v for k, v in self._doc.items() if k not in skip}
        got = {k: v for k, v in doc.items() if k not in skip}
        found = _first_difference(want, got, "")
        if found is None:
            return None
        path, w, h = found
        return f"first difference at {path}: plan.json has {_show(h)}, the graph gives {_show(w)}"

    def __repr__(self) -> str:
        return (
            f"<Plan {self.name!r}: {len(self._doc['buckets'])} buckets, {len(self._doc['buckets'][0]['steps'])} steps, "
            f"arena {self.arena_bytes} bytes, digest {self.digest[:12]}>"
        )


def plan(graph: Any, buckets: Any = None, *, reuse: bool = False) -> Plan:
    """Compile a ``@tilefish.graph`` into a :class:`Plan`: for every combination of bucket values of its dimensions, the launches in order and the
    arena offsets of every buffer.

    ``buckets`` maps a dimension name to the sizes to plan (``{"M": [32, 64, 128]}``): for a dimension that has ``buckets=`` it defaults to those
    (and may name some of them); for one that has none it must be given. ``reuse=True`` lets intermediates whose live ranges do not intersect share
    bytes (the default gives every buffer its own region). Needs no GPU: nothing is built, no binary is chosen but by name (the stem of its manifest
    entry), no torch is imported.
    """
    if not isinstance(graph, GraphOp) or isinstance(graph, KernelOp):
        raise TilefishPlanError(
            f"tilefish.plan takes a @tilefish.graph, got {graph!r}" + (" (a kernel is launched by tilefish.run directly)" if isinstance(graph, KernelOp) else "")
        )
    if not isinstance(reuse, bool):
        raise TilefishPlanError(f"reuse= is True or False, got {reuse!r}")
    name = graph.graph.name
    w = _walk(graph)
    _name_buffers(w.buffers, w.outputs)
    dims = _bucket_values(name, w.dims, buckets)
    count = math.prod(len(v) for v in dims.values())
    if count > MAX_BUCKETS:
        raise TilefishPlanError(f"graph '{name}': {count} bucket combinations are planned, more than the {MAX_BUCKETS} a plan may have (plan fewer sizes)")
    combos = bucket_combinations(dims)
    by_name = {d.name: d for d in w.dims}

    for index, s in enumerate(w.steps):  # the live range of every buffer: from the first step that has it as an argument to the last
        for a in s.args:
            b = a.loc.buffer
            if b.first < 0:
                b.first = index
            b.last = index
    for b in w.buffers:
        if b.role == "workspace":
            b.first, b.last = 0, max(len(w.steps) - 1, 0)

    sizes: "dict[str, int]" = {b.name: 0 for b in w.buffers}
    buckets_doc: "list[dict[str, Any]]" = []
    callees: "dict[str, Callee]" = {}
    first_kinds: "list[tuple[str, str]] | None" = None
    for combo in combos:
        mapping = {by_name[n]: v for n, v in combo.items()}
        for req in _requirements(graph):
            _check_requirement(name, req, mapping, combo)
        shapes: "dict[str, tuple[str, tuple[int, ...]]]" = {}
        for b in w.buffers:
            shape = tuple(_eval(d, mapping, name, f"the shape of '{b.name}'") for d in b.shape)
            shapes[b.name] = (b.dtype, shape)
            sizes[b.name] = max(sizes[b.name], math.prod(shape) * _itemsize(b.dtype))
        steps_doc: "list[dict[str, Any]]" = []
        for s in w.steps:
            args_doc: "list[dict[str, Any]]" = []
            for a in s.args:
                l = a.loc
                b = l.buffer
                shape = tuple(_eval(d, mapping, name, f"the shape of a view of '{b.name}'") for d in l.shape)
                strides = tuple(_eval(d, mapping, name, f"a stride of a view of '{b.name}'") for d in l.strides)
                offset = _eval(l.offset, mapping, name, f"the offset of a view of '{b.name}'")
                entry: "dict[str, Any]" = {"name": a.name, "binding": b.name}
                bshape = shapes[b.name][1]
                if not (offset == 0 and shape == bshape and strides == _row_major(bshape)):
                    entry["view"] = {"shape": list(shape), "strides": list(strides), "offset": offset}
                args_doc.append(entry)
            bound = resolve_step(s.callee, args_doc, shapes, f"graph '{name}' at {combo or 'its only size'}, {s.what}")
            if s.consts and s.callee.kind == "kernel":  # a constexpr is compiled into the binary: a call may restate its value, not change it (as for a single launch)
                try:
                    check_consts(bound.spec, {s.callee.names[k]: v for k, v in s.consts.items()})
                except TilefishError as e:
                    raise TilefishPlanError(f"graph '{name}': {s.what}: {e}") from None
            callees.setdefault(bound.stem, s.callee)
            if callees[bound.stem] is not s.callee:
                raise TilefishPlanError(
                    f"graph '{name}': two different operators are both named '{s.callee.name}' (the binary '{bound.stem}' would be the entry of both): rename one"
                )
            step: "dict[str, Any]" = {"entry": bound.stem, "kind": s.callee.kind, "args": args_doc}
            if s.library:
                step["workspace"] = next(b.name for b in w.buffers if b.role == "workspace")
            steps_doc.append(step)
        kinds = [(st["kind"], st.get("workspace", "")) for st in steps_doc]
        first_kinds = first_kinds or kinds
        buckets_doc.append({"values": dict(combo), "steps": steps_doc})
    if not combos:  # pragma: no cover - itertools.product of nothing is one empty combination
        raise TilefishPlanError(f"graph '{name}' has no bucket")

    items = [
        ArenaItem(b.name, sizes[b.name], b.first, b.last, b.role == "intermediate")
        for b in w.buffers
        if b.role in ARENA_ROLES
    ]
    variants = {flag: plan_arena(items, flag)[1] for flag in (False, True)}
    offsets, arena = plan_arena(items, reuse)
    buffers_doc = []
    for b in w.buffers:
        buffers_doc.append(
            {
                "name": b.name,
                "role": b.role,
                "dtype": b.dtype,
                "shape": [_shape_entry(d) for d in b.shape],
                "offset": offsets.get(b.name) if b.role in ARENA_ROLES else None,
                "bytes": sizes[b.name],
            }
        )
    from . import __version__

    doc: "dict[str, Any]" = {
        "schema": PLAN_SCHEMA,
        "tilefish": __version__,
        "graph": name,
        "arch": None,
        "manifest_digest": None,
        "dims": {n: list(v) for n, v in sorted(dims.items())},
        "reuse": reuse,
        "arena_alignment": ARENA_ALIGNMENT,
        "arena_bytes": arena,
        "buffers": buffers_doc,
        "outputs": [b.name for b in w.outputs],
        "buckets": buckets_doc,
    }
    doc["plan_digest"] = plan_digest(doc)
    made = Plan(graph, doc, callees, variants)
    _check_resolves(made)
    return made


def _shape_entry(d: DimExpr) -> "int | str":
    folded = d.folded()
    return folded.value if folded.is_static else str(folded)


def _requirements(gop: GraphOp) -> "tuple[Any, ...]":
    return tuple(gop.graph.requirements)


def _check_requirement(graph: str, req: "Requirement | BucketRequirement", mapping: "Mapping[Any, int]", combo: "Mapping[str, int]") -> None:
    value = eval_dim(req.dim, mapping)
    if value is None:
        return
    if isinstance(req, Requirement):
        if value % req.divisor:
            raise TilefishPlanError(f"graph '{graph}': at {dict(combo)} {req}: it is {value}")
    elif value not in req.buckets:
        raise TilefishPlanError(f"graph '{graph}': at {dict(combo)} {req}: it is {value}")


def _check_resolves(p: Plan) -> None:
    """Resolve every step of the finished document again, the way a runner does (from the document alone): what a runner will bind is what the plan found."""
    doc = p._doc
    for b in doc["buckets"]:
        shapes = buffer_shapes(doc, b["values"])
        for i, st in enumerate(b["steps"]):
            callee = p._callees[st["entry"]]
            bound = resolve_step(callee, st["args"], shapes, f"graph '{doc['graph']}' at {b['values'] or 'its only size'}, step #{i}")
            if bound.stem != st["entry"]:  # pragma: no cover - the entry was taken from this very resolution
                raise TilefishPlanError(f"step #{i} of graph '{doc['graph']}' resolves to '{bound.stem}', not '{st['entry']}'")


def buffer_shapes(doc: "Mapping[str, Any]", values: "Mapping[str, int]") -> "dict[str, tuple[str, tuple[int, ...]]]":
    """Every buffer of a plan document as ``(dtype, shape)`` at one bucket combination (the shape expressions evaluated)."""
    out: "dict[str, tuple[str, tuple[int, ...]]]" = {}
    table = {Dim(n): v for n, v in values.items()}
    for b in doc["buffers"]:
        shape = []
        for e in b["shape"]:
            if _is_int(e):
                shape.append(e)
            else:
                p = sv._ExprParser(e, f"the shape of buffer '{b['name']}'")
                expr = p.expr()
                p.done()
                v = None if expr is None else eval_dim(expr, table)
                if v is None:
                    raise TilefishPlanError(f"the shape {e!r} of buffer '{b['name']}' has no value at {dict(values)}")
                shape.append(v)
        out[b["name"]] = (b["dtype"], tuple(shape))
    return out


# -- reading and validating a plan file -------------------------------------------------------------------------------------------------


def _bad(message: str) -> Any:
    if len(message) > 1000:  # a refusal quotes what it refuses, and what it refuses may be a list of a million entries
        message = message[:1000] + f"... ({len(message)} characters in all)"
    raise TilefishManifestError(f"plan.json: {message}")


def _keys(obj: Any, where: str, keys: "Sequence[str]", optional: "Sequence[str]" = ()) -> None:
    if not isinstance(obj, dict):
        _bad(f"{where} must be an object, got {type(obj).__name__}")
    missing = [k for k in keys if k not in obj]
    unknown = sorted(str(k) for k in set(obj) - set(keys) - set(optional))
    if missing or unknown:
        _bad(f"{where} must have exactly the fields {list(keys)}" + (f" (and optionally {list(optional)})" if optional else "") + (f"; missing {missing}" if missing else "") + (f"; unknown {unknown}" if unknown else ""))


def validate_doc(doc: Any) -> None:
    """Strict reading of a plan document, with the rules of ``docs/plan-schema-1.md`` (raises :class:`~tilefish.errors.TilefishManifestError`).

    Structure and types of every field; the buffers (names, roles, dtypes, shapes over the dimensions, the arena offsets: aligned, inside the arena,
    and overlapping only where ``reuse`` allows it and the buffers' live ranges are disjoint); the buckets (exactly the combinations of ``dims``, in
    order) and the steps (the same kinds and bindings in every bucket, every view inside its buffer, a result bound to a buffer a step may write, a
    buffer written before it is read). What needs the manifest (that an entry exists and its pointers are what the arguments say) is
    :func:`check_against_manifest`.
    """
    try:
        _validate(doc)
    except TilefishManifestError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError, RecursionError, TilefishError) as e:
        raise TilefishManifestError(f"plan.json is malformed: {type(e).__name__}: {e}") from None


def _validate(doc: Any) -> None:
    _keys(doc, "the document", TOP_KEYS)
    if doc["schema"] != PLAN_SCHEMA or type(doc["schema"]) is not str:
        _bad(f"schema must be the string {PLAN_SCHEMA!r}, got {doc['schema']!r}")
    if not isinstance(doc["tilefish"], str):
        _bad("tilefish must be a string")
    if not isinstance(doc["graph"], str) or not _NAME.match(doc["graph"]):
        _bad(f"graph must be an identifier, got {doc['graph']!r}")
    if not isinstance(doc["arch"], str) or not _ARCH.match(doc["arch"]):
        _bad(f"arch must look like 'sm_103', got {doc['arch']!r}")
    for key in ("manifest_digest", "plan_digest"):
        if not (isinstance(doc[key], str) and _HEX64.match(doc[key])):
            _bad(f"{key} must be 64 lowercase hex digits (a sha256), got {doc[key]!r}")
    if type(doc["reuse"]) is not bool:
        _bad(f"reuse must be true or false, got {doc['reuse']!r}")
    if not _is_int(doc["arena_alignment"]) or doc["arena_alignment"] != ARENA_ALIGNMENT:
        _bad(f"arena_alignment must be {ARENA_ALIGNMENT}, got {doc['arena_alignment']!r}")
    dims = doc["dims"]
    if not isinstance(dims, dict):
        _bad("dims must be an object")
    for n, vs in dims.items():
        if not _NAME.match(n):
            _bad(f"the dimension name {n!r} is not an identifier")
        if not isinstance(vs, list) or not vs or any(not _is_int(v) or v < 1 for v in vs) or vs != sorted(set(vs)):
            _bad(f"dims.{n} must be a non-empty, strictly increasing list of positive ints, got {vs!r}")
    # buffers
    buffers = doc["buffers"]
    if not isinstance(buffers, list) or not buffers:
        _bad("buffers must be a non-empty list")
    if len(buffers) > MAX_BUFFERS:
        _bad(f"a plan has at most {MAX_BUFFERS} buffers, this one has {len(buffers)}")
    by_name: "dict[str, dict[str, Any]]" = {}
    for i, b in enumerate(buffers):
        _keys(b, f"buffers[{i}]", BUFFER_KEYS)
        if not isinstance(b["name"], str) or not _NAME.match(b["name"]) or b["name"] in by_name:
            _bad(f"buffers[{i}].name must be a unique identifier, got {b['name']!r}")
        if b["role"] not in ROLES or type(b["role"]) is not str:
            _bad(f"buffers[{i}].role must be one of {list(ROLES)}, got {b['role']!r}")
        if b["dtype"] not in TORCH_DTYPE_NAMES or type(b["dtype"]) is not str:
            _bad(f"buffers[{i}].dtype must be one of {sorted(TORCH_DTYPE_NAMES)}, got {b['dtype']!r}")
        if not isinstance(b["shape"], list) or any(not (_is_int(e) and e >= 1) and not isinstance(e, str) for e in b["shape"]):
            _bad(f"buffers[{i}].shape must be a list of positive ints and dimension expressions")
        if not _is_int(b["bytes"]) or b["bytes"] < 1:
            _bad(f"buffers[{i}].bytes must be a positive int, got {b['bytes']!r}")
        if b["role"] in EXTERNAL_ROLES:
            if b["offset"] is not None:
                _bad(f"buffers[{i}] ('{b['name']}') is external (role {b['role']}): its offset must be null, not {b['offset']!r}")
        elif not _is_int(b["offset"]) or b["offset"] < 0 or b["offset"] % ARENA_ALIGNMENT:
            _bad(f"buffers[{i}] ('{b['name']}') lives in the arena: its offset must be a non-negative multiple of {ARENA_ALIGNMENT}, got {b['offset']!r}")
        by_name[b["name"]] = b
    if sum(1 for b in buffers if b["role"] == "workspace") > 1:
        _bad("at most one buffer has the role workspace")
    outputs = doc["outputs"]
    if (
        not isinstance(outputs, list)
        or any(not isinstance(o, str) or o not in by_name or by_name[o]["role"] != "output" for o in outputs)
        or len(set(outputs)) != len(outputs)
        or set(outputs) != {b["name"] for b in buffers if b["role"] == "output"}
    ):
        _bad("outputs must list the buffers of role output, each once, in the order the graph returns them")
    for b in buffers:
        if b["role"] == "devscalar" and b["shape"] != []:
            _bad(f"the devscalar '{b['name']}' is one element in device memory: its shape must be []")
        if b["role"] == "workspace" and (b["dtype"] != "u8" or len(b["shape"]) != 1 or not _is_int(b["shape"][0])):
            _bad(f"the workspace '{b['name']}' must be a u8 buffer with one static size")
    # buckets (counted before they are listed: a document must not make a reader build a list of 10**30 combinations)
    n_combos = math.prod(len(v) for v in dims.values())
    if n_combos > MAX_BUCKETS:
        _bad(f"dims have {n_combos} bucket combinations, more than the {MAX_BUCKETS} a plan may have")
    buckets = doc["buckets"]
    if not isinstance(buckets, list) or len(buckets) != n_combos:
        _bad(f"buckets must have one entry per combination of the dims ({n_combos}), in order")
    combos = bucket_combinations({n: v for n, v in dims.items()})
    sizes = {n: 0 for n in by_name}
    shapes_by_bucket: "list[dict[str, tuple[str, tuple[int, ...]]]]" = []
    for combo, bucket in zip(combos, buckets):
        _keys(bucket, "a bucket", ["values", "steps"])
        if bucket["values"] != combo or any(not _is_int(v) for v in bucket["values"].values()):
            _bad(f"the buckets must follow the combinations of the dims in order: expected {combo}, got {bucket['values']!r}")
        try:
            shapes = buffer_shapes(doc, combo)
        except TilefishPlanError as e:
            _bad(str(e))
        for n, (dtype, shape) in shapes.items():
            if any(d < 1 for d in shape):
                _bad(f"the shape of '{n}' at {combo} has a non-positive size {list(shape)}")
            sizes[n] = max(sizes[n], math.prod(shape) * _itemsize(dtype))
        shapes_by_bucket.append(shapes)
    for n, b in by_name.items():
        if b["bytes"] != sizes[n]:
            _bad(f"buffers '{n}': bytes must be the largest size of its shape over the buckets ({sizes[n]}), got {b['bytes']}")
    # steps
    n_steps = len(buckets[0]["steps"]) if isinstance(buckets[0].get("steps"), list) else 0
    if n_steps > MAX_STEPS:
        _bad(f"a bucket has at most {MAX_STEPS} steps, this one has {n_steps}")
    skeleton: "list[Any] | None" = None
    first: "dict[str, int]" = {}
    last: "dict[str, int]" = {}
    for bi, (bucket, shapes) in enumerate(zip(buckets, shapes_by_bucket)):
        steps = bucket["steps"]
        if not isinstance(steps, list) or len(steps) != n_steps:
            _bad(f"every bucket must have the same number of steps ({n_steps})")
        shape_of = {n: s for n, (_, s) in shapes.items()}
        this: "list[Any]" = []
        for si, st in enumerate(steps):
            _keys(st, f"step #{si}", STEP_KEYS, ["workspace"])
            if not isinstance(st["entry"], str) or not sv._STEM.match(st["entry"]):
                _bad(f"step #{si}: entry must be a manifest stem such as 'gemm.M64', got {st['entry']!r}")
            if st["kind"] not in KINDS or type(st["kind"]) is not str:
                _bad(f"step #{si}: kind must be one of {list(KINDS)}, got {st['kind']!r}")
            if (st["kind"] == "library") != ("workspace" in st):
                _bad(f"step #{si}: exactly the steps of kind 'library' have a workspace")
            if "workspace" in st:
                ws = by_name.get(st["workspace"])
                if ws is None or ws["role"] != "workspace":
                    _bad(f"step #{si}: workspace must name the buffer of role workspace, got {st['workspace']!r}")
            if not isinstance(st["args"], list) or not st["args"] or len(st["args"]) > MAX_ARGS:
                _bad(f"step #{si}: args must be a non-empty list of at most {MAX_ARGS} arguments")
            names_here = []
            for ai, a in enumerate(st["args"]):
                _keys(a, f"step #{si} args[{ai}]", ["name", "binding"], ["view"])
                if not isinstance(a["name"], str) or not _NAME.match(a["name"]):
                    _bad(f"step #{si} args[{ai}]: name must be an identifier")
                b = by_name.get(a["binding"]) if isinstance(a["binding"], str) else None
                if b is None:
                    _bad(f"step #{si} args[{ai}]: binding {a['binding']!r} is not a buffer")
                if b["role"] == "workspace":
                    _bad(f"step #{si} args[{ai}]: the workspace is not an argument, it is the workspace field of a library step")
                names_here.append((a["name"], a["binding"]))
                if "view" in a:
                    v = a["view"]
                    _keys(v, f"step #{si} args[{ai}].view", ["shape", "strides", "offset"])
                    if (
                        not isinstance(v["shape"], list)
                        or not isinstance(v["strides"], list)
                        or len(v["shape"]) != len(v["strides"])
                        or any(not _is_int(x) or x < 1 for x in v["shape"])
                        or any(not _is_int(x) or x < 0 for x in v["strides"])
                        or not _is_int(v["offset"])
                        or v["offset"] < 0
                    ):
                        _bad(f"step #{si} args[{ai}].view must be {{shape: positive ints, strides: non-negative ints (same length), offset: non-negative int}}")
                    if v["offset"] + extent_elements(v["shape"], v["strides"]) > math.prod(shape_of[a["binding"]]):
                        _bad(f"step #{si} args[{ai}]: the view reaches past the end of buffer '{a['binding']}' at {bucket['values']}")
                    if b["role"] == "devscalar":
                        _bad(f"step #{si} args[{ai}]: a devscalar is bound whole, without a view")
            if len({a for a, _ in names_here}) != len(names_here):
                _bad(f"step #{si}: the argument names must be distinct, got {[a for a, _ in names_here]}")
            this.append((st["kind"], st.get("workspace"), tuple(names_here)))
            if bi == 0:
                for a in st["args"]:
                    n = a["binding"]
                    first.setdefault(n, si)
                    last[n] = si
        if skeleton is None:
            skeleton = this
        elif this != skeleton:
            _bad(f"bucket {bucket['values']} has other kinds, workspaces, argument names or bindings than the first bucket: only the entries and the views may differ")
    uses_library = any(st["kind"] == "library" for st in buckets[0]["steps"])
    if uses_library != any(b["role"] == "workspace" for b in buffers):
        _bad("there is a buffer of role workspace exactly when some step has kind library")
    # the arena: every region inside, overlaps only where the live ranges are disjoint
    arena = [b for b in buffers if b["role"] in ARENA_ROLES]
    end = max((b["offset"] + align_up(b["bytes"]) for b in arena), default=0)
    if not _is_int(doc["arena_bytes"]) or doc["arena_bytes"] != align_up(end):
        _bad(f"arena_bytes must be the end of the last region rounded up to {ARENA_ALIGNMENT} ({align_up(end)}), got {doc['arena_bytes']!r}")
    # A sweep over the regions by offset, in O(n log n): the regions that cover the current offset are the *active* ones. A region that may not share (reuse is false, or it is not an
    # intermediate that some step uses) must start with no active region and have none start under it; the ones that may share must have pairwise disjoint live ranges, which
    # is kept as a sorted list of ranges (disjoint, so a new range only has to be compared with its two neighbours).
    heap: "list[tuple[int, int, str, tuple[int, int, str] | None]]" = []  # (end of region, sequence, name, its live range if it shares)
    lives: "list[tuple[int, int, str]]" = []
    exclusive = ""  # the name of an active region that may not share, if there is one
    for seq, b in enumerate(sorted(arena, key=lambda r: (r["offset"], r["bytes"]))):
        while heap and heap[0][0] <= b["offset"]:
            _, _, gone, span = heapq.heappop(heap)
            if span is not None:
                del lives[bisect.bisect_left(lives, span)]
            elif gone == exclusive:
                exclusive = ""
        name = b["name"]
        shares = bool(doc["reuse"]) and b["role"] == "intermediate" and name in first
        overlap = "overlap, which only reuse=true allows, and only for two intermediates that no step uses together"
        if not shares:
            if heap:
                _bad(f"the regions of '{name}' and '{heap[0][2]}' {overlap}")
            span = None
            exclusive = name
        else:
            if exclusive:
                _bad(f"the regions of '{name}' and '{exclusive}' {overlap}")
            span = (first[name], last[name], name)
            at = bisect.bisect_left(lives, span)
            for neighbour in (lives[at - 1] if at else None, lives[at] if at < len(lives) else None):
                if neighbour is not None and neighbour[0] <= span[1] and span[0] <= neighbour[1]:
                    _bad(f"the regions of '{name}' and '{neighbour[2]}' {overlap}")
            lives.insert(at, span)
        heapq.heappush(heap, (b["offset"] + align_up(b["bytes"]), seq, name, span))
    # (that an intermediate or an output is written before it is read needs the access of each pointer, which the manifest has: check_against_manifest)


def check_against_manifest(doc: "Mapping[str, Any]", manifest: Any) -> None:
    """Check a plan document against the manifest it names: the digest, and for every step that the entry exists on the plan's architecture with the
    kind the step says, and that the pointers of its ABI are the arguments the step binds (count, names, dtype, rank, access, and which buffers
    may be written); a buffer is written before it is read. Raises :class:`~tilefish.errors.TilefishManifestError`."""
    if manifest.digest != doc["manifest_digest"]:
        _bad(
            f"the plan was built against the manifest with digest {doc['manifest_digest']}, but the manifest in {str(manifest.directory)!r} has digest "
            f"{manifest.digest}: the plan is refused (export it again, or restore the manifest it was built with)"
        )
    arch = doc["arch"]
    by_name = {b["name"]: b for b in doc["buffers"]}
    for bucket in doc["buckets"]:
        written: "set[str]" = {b["name"] for b in doc["buffers"] if b["role"] in ("input", "weight", "inout", "devscalar")}  # provided by the caller / the host
        for si, st in enumerate(bucket["steps"]):
            stem = st["entry"]
            where = f"step #{si} ({stem}) of bucket {bucket['values']}"
            if st["kind"] == "kernel":
                artifact = manifest.get(stem, arch)
                if artifact is None:
                    _bad(f"{where}: the manifest has no kernel '{stem}' for {arch}")
                pointers = [p for p in artifact.entry["abi"]["params"] if p["kind"] == "pointer"]
            else:
                entry = manifest.library_entry(stem, arch)
                if entry is None:
                    _bad(f"{where}: the manifest has no library entry '{stem}' for {arch}")
                pointers = entry["abi"]["params"]
                ws = by_name[st["workspace"]]
                if ws["bytes"] < entry["workspace_bytes"]:
                    _bad(f"{where}: the workspace '{ws['name']}' has {ws['bytes']} bytes, but the entry needs {entry['workspace_bytes']}")
            if len(pointers) != len(st["args"]):
                _bad(f"{where}: the binary takes {len(pointers)} pointers ({', '.join(p['name'] for p in pointers)}), the step binds {len(st['args'])}")
            for p, a in zip(pointers, st["args"]):
                buf = by_name[a["binding"]]
                if p["name"] != a["name"]:
                    _bad(f"{where}: argument '{a['name']}' is bound where the binary has '{p['name']}'")
                if p["dtype"] != buf["dtype"]:
                    _bad(f"{where}: argument '{a['name']}' is bound to '{buf['name']}' ({buf['dtype']}), but the binary takes {p['dtype']}")
                rank = len(a["view"]["shape"]) if "view" in a else len(buf["shape"])
                if rank != len(p["shape"]):
                    _bad(f"{where}: argument '{a['name']}' has rank {rank}, but the binary takes rank {len(p['shape'])}")
                if (p["shape"] == []) != (buf["role"] == "devscalar"):
                    _bad(f"{where}: argument '{a['name']}': a device scalar (a pointer of shape []) is bound to the buffer of role devscalar, and only it")
                writes = p["access"] in ("out", "inout")
                if writes and buf["role"] not in WRITABLE_ROLES:
                    _bad(f"{where}: argument '{a['name']}' is written by the binary but bound to '{buf['name']}', whose role is {buf['role']}")
                if p["access"] != "out" and buf["name"] not in written:
                    _bad(f"{where}: argument '{a['name']}' reads '{buf['name']}' before any step has written it")
                if writes:
                    written.add(buf["name"])


def check_anchor(pin: "str | None", trust_directory: bool) -> None:
    """A replay of a plan needs exactly one trust anchor: ``pin`` or ``trust_directory=True``."""
    from .errors import TilefishRuntimeError

    if pin is not None and trust_directory:
        raise TilefishRuntimeError("pass either pin= (a plan with another digest is refused) or trust_directory=True (whatever the directory holds is accepted), not both")
    if pin is None and not trust_directory:
        raise TilefishRuntimeError(
            "replaying a plan needs a trust anchor: the digests in a plan directory sit next to the files they describe, so whoever can write the "
            "directory can make a replay run any binary. Pass pin=<the plan's digest> (Plan.digest, which export returns; keep it somewhere you trust), "
            "or trust_directory=True to run whatever the directory holds"
        )


def load_plan(path: "str | pathlib.Path", graph: GraphOp, *, pin: "str | None" = None, trust_directory: bool = False) -> Plan:
    """Load the plan of ``graph`` from an exported directory (or its ``plan.json``) for a replay, and refuse everything that is not it.

    The plan needs a trust anchor like a manifest does: ``pin`` (the digest of the plan, :attr:`Plan.digest`, which pins the manifest it was built
    against and so every binary) or ``trust_directory=True``. Then, in this order: ``plan.json`` is read strictly and its digest is checked against
    the pin and against the ``plan_digest`` it carries (an edited plan is refused here); the manifest next to it is loaded (every cubin checked) and
    its digest must be the ``manifest_digest`` the plan names; every step must be what the manifest's entries say (:func:`check_against_manifest`);
    and the document must be exactly the plan this process derives from ``graph`` for the same buckets (:meth:`Plan.stale_reason`): a graph that
    changed since the export is refused, as a kernel that changed is. Raises :class:`~tilefish.errors.TilefishManifestError` (a
    :class:`~tilefish.errors.TilefishRuntimeError`).
    """
    from ._manifest import Manifest

    check_anchor(pin, trust_directory)
    if not isinstance(graph, GraphOp) or isinstance(graph, KernelOp):
        raise TilefishPlanError(f"a plan is loaded for a @tilefish.graph, got {graph!r}")
    path = pathlib.Path(path)
    directory = path if path.is_dir() else path.parent
    doc = read_plan(path, pin=pin)
    if doc["graph"] != graph.graph.name:
        raise TilefishManifestError(f"the plan in {str(directory)!r} is the plan of graph '{doc['graph']}', not of '{graph.graph.name}'")
    manifest = Manifest.load(directory)
    check_against_manifest(doc, manifest)
    try:
        derived = plan(graph, {n: v for n, v in doc["dims"].items()}, reuse=doc["reuse"])
    except (TilefishPlanError, TilefishPreconditionError) as e:  # the graph (or an operator it calls) is not what it was: the plan on disk is stale
        raise TilefishManifestError(
            f"the plan in {str(directory)!r} is not the plan of graph '{graph.graph.name}' as this process holds it: the graph can no longer be planned for the "
            f"same buckets ({str(e)[:400]}); export it again (and pin the new plan)"
        ) from e
    why = derived.stale_reason(doc)
    if why is not None:
        raise TilefishManifestError(
            f"the plan in {str(directory)!r} is not the plan of graph '{graph.graph.name}' as this process holds it: {why}; the graph (or an operator it calls) "
            "changed since the plan was exported, so its steps are stale: export it again (and pin the new plan)"
        )
    loaded = derived.with_document(doc)
    loaded.directory = directory
    loaded.manifest = manifest
    manifest.pinned = pin is not None
    return loaded


MAX_INT = 1 << 53  # the integers a double holds exactly: a plan number is read as one in every language


def _parse_int(text: str) -> int:
    value = int(text)
    if abs(value) > MAX_INT:
        raise TilefishManifestError(f"the plan is not strict JSON: the integer {text[:40]}{'...' if len(text) > 40 else ''} does not fit a double (at most 2**53 in magnitude)")
    return value


def read_plan(path: "str | pathlib.Path", *, pin: "str | None" = None) -> "dict[str, Any]":
    """Read ``plan.json`` (a directory holding it, or the file): strict JSON, the digest checked against the ``plan_digest`` it carries and against
    ``pin`` when given, then every rule of :func:`validate_doc`. Returns the document. Raises :class:`~tilefish.errors.TilefishManifestError`."""
    from ._manifest import _no_duplicate_keys, _parse_float, _reject_constant

    path = pathlib.Path(path)
    file = path / PLAN_NAME if path.is_dir() else path
    try:
        size = file.stat().st_size
        if size > MAX_PLAN_BYTES:
            raise TilefishManifestError(f"the plan {str(file)!r} is {size} bytes, more than the {MAX_PLAN_BYTES} a plan may have")
        text = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise TilefishManifestError(f"cannot read the plan {str(file)!r}: {e}") from e
    try:
        doc = json.loads(text, parse_constant=_reject_constant, parse_float=_parse_float, parse_int=_parse_int, object_pairs_hook=_no_duplicate_keys)
    except RecursionError as e:
        raise TilefishManifestError(f"the plan {str(file)!r} is nested too deeply to be read") from e
    except TilefishManifestError:
        raise
    except ValueError as e:
        raise TilefishManifestError(f"the plan {str(file)!r} is not valid JSON: {e}") from e
    if not isinstance(doc, dict) or type(doc.get("schema")) is not str or doc["schema"] != PLAN_SCHEMA:
        raise TilefishManifestError(f"{str(file)!r} is not a tilefish plan of schema {PLAN_SCHEMA!r} (schema {doc.get('schema') if isinstance(doc, dict) else None!r})")
    if pin is not None and not (isinstance(pin, str) and _HEX64.match(pin)):
        raise TilefishManifestError(f"pin= is the 64 lowercase hex digits of a plan digest, got {pin!r}")
    found = plan_digest(doc)
    if pin is not None and found != pin:
        raise TilefishManifestError(
            f"the plan in {str(file)!r} has digest {found}, but the pin is {pin}: it is not the plan that was pinned (it was changed, or it is another export)"
        )
    if doc.get("plan_digest") != found:
        raise TilefishManifestError(
            f"the plan in {str(file)!r} was edited: its plan_digest says {doc.get('plan_digest')!r}, but its content has digest {found}"
        )
    validate_doc(doc)
    return doc
