"""Epilogue hooks on externs (``docs/en/rfcs/0003-epilogue-fusion.md``, milestone E1): the arithmetic a graph body does to a GEMM's or a convolution's
result, fused into the extern's own kernel where it would store that result.

**The declaration.** An extern says where its kernel takes an epilogue::

    @tilefish.extern(triton=gemm_kernel, grid=..., epilogue=tilefish.hook("EPILOGUE", value="out", out="c"))
    def gemm(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16], c: Out[Tensor[M, N, bf16]], ...): ...

The kernel has two parameters the stub does not list, ``EPILOGUE`` (a ``tl.constexpr``: a jit function) and ``EPILOGUE_ARGS`` (its arguments, a
tuple), and calls the hook once per output tile, where it would store it (:data:`ABI`)::

    tile = EPILOGUE(tile, rows, cols, row_mask, col_mask, EPILOGUE_ARGS)

Without a fusion the hook is the identity, and the extern is the kernel it always was. tilefish compiles every extern that declares a hook through a
small generated module (:func:`identity_source`): a ``@triton.jit`` (or ``@gluon.jit``) function with the stub's parameters that calls the kernel
with the identity hook. Its source is in the declaration digest (``source_sha256``), and the binary is identified by the IR it compiles to, which has
the kernel in it (:mod:`tilefish._ir`), so nothing in a compile spec, a manifest or a host is new.

**A fusion.** The fusion pass (:mod:`tilefish._fusion`) takes a region of the graph body that reads the extern's result and fits its tile (an
elementwise region over the result's shape, or a row region whose rows a tile holds whole) and makes one op of the pair (:func:`hooked`): the same
generated module, with the region's code as the hook (lowered by :func:`tilefish.lower.lower_hook` from the region's own nodes, with the code every
region has: no contraction, every narrowing cast inline PTX) and the region's other inputs and its side outputs as extra parameters. Its name is
``<extern>_epi_<hash>`` (the content). Two contracts:

* **(i) exact**, ``value="out"``: the hook sees the tile the kernel rounded to its output type, as the kernel would store it, so the fused op writes
  the bytes of the extern followed by the region's kernel (``U(P)``) by construction; ``export`` checks it.
* **(ii) accumulator**, ``value="acc"`` under ``tilefish.fusion(epilogues="accumulator")``: the hook sees the f32 accumulator. The graph is the
  declared rewrite ``R_acc`` of its body (every widening ``y.to(f32)`` of the result is the call's f32 result, which the *accumulator form* of the
  extern writes: :func:`acc_form`), and the fused op writes the bytes of the unfused plan of that program, ``U(R_acc(P))``; ``export`` checks it.
"""

from __future__ import annotations

import hashlib
import keyword
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .errors import TilefishDeclarationError

HOOK_VALUES = ("out", "acc")
MAX_SPLIT = 4  # a hook sees at most this many tiles at once (a dual accumulator: 2)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
# names the generated module defines or imports: a parameter of an extern with a hook may not have one
RESERVED = frozenset({"tl", "triton", "gluon", "gl", "libdevice", "tile", "rows", "cols", "row_mask", "col_mask", "args"})
MODULE = "tilefish_generated"

ABI = """\
The hook ABI (docs/en/rfcs/0003-epilogue-fusion.md, section 3.2.1; E1). A kernel that declares tilefish.hook(NAME, value=..., out=OUT, split=S) takes the
parameters NAME (a tl.constexpr: a jit function) and NAME_ARGS (its arguments, a tuple), which its stub does not list, and calls

    tile = NAME(tile, rows, cols, row_mask, col_mask, NAME_ARGS)

once per output tile of OUT, where it would store it:

* tile: value="out": the tile rounded to OUT's element type (tile.to(OUT.dtype.element_ty)), as the kernel would store it; value="acc": the f32
  accumulator before that rounding. With S > 1 (a dual accumulator), a tuple of S tiles: tile k holds the columns k * C + cols of OUT, C = its last
  axis / S;
* rows: the tile's rows (i32), OUT seen as [R, L] (every axis but the last flattened, row-major); cols: its columns (i32) within a chunk of C;
  row_mask = rows < R, col_mask = cols < C; in a Gluon kernel rows and cols are in the slice layouts (1 and 0) of the tile's layout;
* it returns what the kernel stores: value="out": the tile (or S-tuple) in OUT's element type; value="acc": the tile (or tuple) the kernel then
  converts to OUT's element type (tile.to(OUT.dtype.element_ty)) and stores, in place of its own rounding. With S > 1 it returns a tuple of S or
  of 1 tiles, and the kernel stores them side by side: tile k into the columns k * C + cols of OUT, whose rows hold len(tiles) * C columns (for k
  in tl.static_range(len(tiles))): a fused op that returns one tile gives OUT the shape of one chunk. The type of OUT is the kernel's to read
  (OUT.dtype.element_ty, a descriptor's shape): a fused op may give OUT another element type (value="acc") or the shape of one chunk.

Without a fusion the hook is the identity. A kernel that offers both declares both: the "acc" hook before the rounding, the "out" hook after it.
"""


@dataclass(frozen=True, eq=False)
class Hook:
    """An epilogue hook point of an extern (:func:`hook`)."""

    name: str
    value: str
    out: str
    split: int = 1
    block_n: "str | None" = None
    reference: "Callable[..., Any] | None" = field(default=None, compare=False)

    @property
    def args_name(self) -> str:
        """The kernel's parameter that takes the hook's arguments."""
        return f"{self.name}_ARGS"

    def canonical(self) -> "dict[str, Any]":
        """What the binary depends on (in the declaration digest, through the generated source; ``reference`` is not)."""
        out: "dict[str, Any]" = {"name": self.name, "value": self.value, "out": self.out, "split": self.split}
        if self.block_n is not None:
            out["block_n"] = self.block_n
        return out

    def __repr__(self) -> str:
        extra = (f", split={self.split}" if self.split != 1 else "") + (f", block_n={self.block_n!r}" if self.block_n else "")
        return f"tilefish.hook({self.name!r}, value={self.value!r}, out={self.out!r}{extra})"


def hook(
    name: str,
    *,
    value: str = "out",
    out: str,
    split: int = 1,
    block_n: "str | None" = None,
    reference: "Callable[..., Any] | None" = None,
) -> Hook:
    """``@tilefish.extern(..., epilogue=tilefish.hook("EPILOGUE", value="out", out="c"))``: where the extern's kernel takes an epilogue.

    * ``name``: the kernel's ``tl.constexpr`` parameter that takes the hook (a jit function); ``<name>_ARGS`` takes its arguments (a tuple). The stub
      lists neither.
    * ``value``: ``"out"``, the hook sees the tile after the kernel's rounding to the output's type (a fusion keeps contract (i): the bytes of the
      extern followed by the region); ``"acc"``, the f32 accumulator before it (contract (ii), under ``tilefish.fusion(epilogues="accumulator")``).
      A kernel that offers both declares two hooks (``epilogue=(tilefish.hook(..., value="acc", ...), tilefish.hook(..., value="out", ...))``).
    * ``out``: the ``Out`` parameter whose tiles the hook sees.
    * ``split``: how many tiles of ``out`` a program holds at once, side by side along its last axis (``2``: a dual accumulator, the tile of a gate
      projection and the matching tile of the up projection, which a gated activation needs).
    * ``block_n``: the ``const`` parameter that is the width of a tile (a row region, such as a norm, is fused only where a tile holds whole rows).
    * ``reference``: for ``value="acc"``, a torch function of the extern's inputs (as ``reference=``) that returns the f32 accumulator of ``out``,
      for ``mode="reference"`` of a graph declared ``epilogues="accumulator"`` (by default the extern's reference, widened); in no digest.

    The calling convention is :data:`ABI`."""
    if not isinstance(name, str) or not _IDENT.match(name) or keyword.iskeyword(name):
        raise TilefishDeclarationError(f"tilefish.hook: name must be the identifier of the kernel's hook parameter, got {name!r}")
    if value not in HOOK_VALUES:
        raise TilefishDeclarationError(f"tilefish.hook: value must be one of {list(HOOK_VALUES)}, got {value!r}")
    if not isinstance(out, str) or not _IDENT.match(out):
        raise TilefishDeclarationError(f"tilefish.hook: out= names the Out parameter whose tiles the hook sees, got {out!r}")
    if isinstance(split, bool) or not isinstance(split, int) or not 1 <= split <= MAX_SPLIT:
        raise TilefishDeclarationError(f"tilefish.hook: split must be an int from 1 to {MAX_SPLIT}, got {split!r}")
    if block_n is not None and (not isinstance(block_n, str) or not _IDENT.match(block_n)):
        raise TilefishDeclarationError(f"tilefish.hook: block_n= names the const parameter that is a tile's width, got {block_n!r}")
    if reference is not None and not callable(reference):
        raise TilefishDeclarationError(f"tilefish.hook: reference must be a callable, got {reference!r}")
    if reference is not None and value != "acc":
        raise TilefishDeclarationError("tilefish.hook: reference= gives the f32 accumulator, for a hook with value='acc'")
    return Hook(name, value, out, split, block_n, reference)


def hooks_of(epilogue: Any) -> "tuple[Hook, ...]":
    """``extern(epilogue=...)`` as a tuple of hooks: one :func:`hook`, or a tuple of two of different ``value``s."""
    if epilogue is None:
        return ()
    hooks = (epilogue,) if isinstance(epilogue, Hook) else tuple(epilogue) if isinstance(epilogue, (tuple, list)) else None
    if hooks is None or not hooks or not all(isinstance(h, Hook) for h in hooks):
        raise TilefishDeclarationError(f"extern: epilogue= takes a tilefish.hook(...) or a tuple of them, got {epilogue!r}")
    values = [h.value for h in hooks]
    if len(set(values)) != len(values):
        raise TilefishDeclarationError(f"extern: epilogue= declares at most one hook of each value ('out', 'acc'), got {list(hooks)}")
    names = [n for h in hooks for n in (h.name, h.args_name)]
    if len(set(names)) != len(names):
        raise TilefishDeclarationError(f"extern: the hooks of epilogue= have parameters of the same name: {names}")
    if len({(h.out, h.split) for h in hooks}) != 1:
        raise TilefishDeclarationError("extern: the hooks of epilogue= see the same tiles: the same out= and split=")
    return hooks


def check_hooks(nm: str, sig: Any, hooks: "Sequence[Hook]", arg_names: "Sequence[str] | None") -> None:
    """The hooks of extern ``nm`` against its stub (``sig``) and, when the kernel object says, its parameters (``arg_names``)."""
    from .dtypes import DType

    by_name = {p.name: p for p in sig.params}
    for p in sig.params:
        if p.name in RESERVED or p.name.startswith("_tf_") or p.name.startswith("ep_"):
            raise TilefishDeclarationError(
                f"extern '{nm}': the parameter '{p.name}' has a name the generated module of an extern with an epilogue hook uses ({sorted(RESERVED)}, "
                "_tf_* and ep_*): rename it in the kernel and the stub"
            )
    for h in hooks:
        for own in (h.name, h.args_name):
            if own in by_name:
                raise TilefishDeclarationError(
                    f"extern '{nm}': the stub lists '{own}', a parameter of the hook {h!r}: the stub leaves the hook's two parameters out (tilefish passes them)"
                )
            if arg_names is not None and own not in arg_names:
                raise TilefishDeclarationError(f"extern '{nm}': the kernel has no parameter '{own}', which the hook {h!r} needs ({ABI.splitlines()[1].strip()})")
        p = by_name.get(h.out)
        if p is None or p.kind != "tensor" or p.access != "out":
            raise TilefishDeclarationError(f"extern '{nm}': the hook {h!r} sees the tiles of '{h.out}', which is not an Out parameter of the stub")
        t = p.type
        dt = t.dtype
        if not isinstance(dt, DType) or dt.name not in ("f32", "bf16", "f16") or t.devscalar or not t.shape:
            raise TilefishDeclarationError(f"extern '{nm}': the hook {h!r} sees '{h.out}' ({t}): an epilogue hook's output is a float tensor (f32, bf16, f16) of rank 1 or more")
        if p.strided:
            raise TilefishDeclarationError(f"extern '{nm}': the hook {h!r} sees '{h.out}', a Strided[...] parameter: a hook's output is a dense tensor")
        if p.alias is not None:
            raise TilefishDeclarationError(f"extern '{nm}': the hook {h!r} sees '{h.out}', which may be written over '{p.alias}' (MayAlias): not with a hook")
        if h.split > 1:
            last = t.shape[-1]
            if not last.is_divisible_by(h.split):
                raise TilefishDeclarationError(
                    f"extern '{nm}': the hook {h!r} splits the last axis of '{h.out}' ({last}) into {h.split} chunks, but it is not provably divisible by {h.split}"
                )
        if h.block_n is not None:
            q = by_name.get(h.block_n)
            if q is None or q.kind != "const":
                raise TilefishDeclarationError(f"extern '{nm}': block_n={h.block_n!r} of the hook {h!r} is not a const parameter of the stub")
    if arg_names is not None:
        mine = {n for h in hooks for n in (h.name, h.args_name)}
        rest = [n for n in arg_names if n not in mine]
        stub = [p.name for p in sig.params]
        if rest != stub:
            raise TilefishDeclarationError(
                f"extern '{nm}': the stub's parameters {stub} are not the kernel's {list(arg_names)} without the hook parameters {sorted(mine)}: the stub lists the "
                "kernel's other parameters, in order"
            )


# -- the generated module -----------------------------------------------------------------------------------------------------------------


def _decl(p: Any) -> str:
    return f"{p.name}: tl.constexpr" if p.kind == "const" else p.name


def _header_lines(lines: "Sequence[str]") -> "list[str]":
    return [f"# {line}" if line else "#" for line in lines]


def _module_head(language: str, kernel_ref: str, extra_imports: "Sequence[str]" = ()) -> "list[str]":
    module, _, function = kernel_ref.partition(":")
    head = ["import triton", "import triton.language as tl"]
    if language == "gluon":
        head.append("from triton.experimental import gluon")
    head += list(extra_imports)
    head.append(f"from {module} import {function} as _tf_kernel")
    return head


IDENTITY = [
    "@triton.jit",
    "def _tf_identity(tile, rows, cols, row_mask, col_mask, args):",
    "    # the identity hook: the tile (or the tuple of tiles) as the kernel would store it",
    "    return tile",
]


def wrapper_lines(fn_name: str, language: str, params: "Sequence[Any]", call: "Sequence[str]", hooks: "Mapping[str, tuple[str, str]]") -> "list[str]":
    """The generated kernel: ``fn_name`` with ``params`` (each a ``Param``; consts are constexprs) calling the extern's kernel with the stub's parameters
    ``call`` by keyword, and each hook's function and argument tuple (``hooks``: hook name -> (function, tuple text))."""
    deco = "@gluon.jit" if language == "gluon" else "@triton.jit"
    lines = [deco, f"def {fn_name}("]
    lines += [f"    {_decl(p)}," for p in params]
    lines += ["):", "    _tf_kernel("]
    lines += [f"        {name}={name}," for name in call]
    for hname, (fn, args) in hooks.items():
        lines.append(f"        {hname}={fn},")
        lines.append(f"        {hname}_ARGS={args},")
    lines.append("    )")
    return lines


def identity_source(nm: str, sig: Any, hooks: "Sequence[Hook]", kernel_ref: str, language: str) -> "tuple[str, str, str]":
    """The generated module of an extern with hooks, unfused: ``(module, function, source)``. Its function has the stub's parameters and calls the
    kernel with the identity hook, so the binary is the kernel's own."""
    module, fn_name = f"{MODULE}.{nm}__hooks", f"tf_{nm}__hooks"
    # The headers of the generated modules name the RFC by its path of 0.0.13 (now docs/en/rfcs/...): a header is part of the generated
    # source, and so of its digest, which a documentation move must not change.
    header = [
        f"Generated by tilefish for the extern '{nm}' ({kernel_ref}), which declares the epilogue hook"
        + ("s " if len(hooks) > 1 else " ")
        + ", ".join(repr(h) for h in hooks)
        + ". Do not edit.",
        "The kernel with the identity hook: the extern as it is (docs/rfcs/0003-epilogue-fusion.md, section 3.2.1).",
    ]
    lines = _header_lines(header) + _module_head(language, kernel_ref) + ["", ""] + IDENTITY + ["", ""]
    lines += wrapper_lines(fn_name, language, sig.params, [p.name for p in sig.params], {h.name: ("_tf_identity", "()") for h in hooks})
    return module, fn_name, "\n".join(lines) + "\n"


def make_extern_with_hooks(stub: Any, nm: str, sig: Any, hooks: "tuple[Hook, ...]", kernel_ref: str, language: str, **kwargs: Any) -> Any:
    """The ``ExternOp`` of a declaration with hooks: launched through its generated module (:func:`identity_source`)."""
    from .decorators import ExternOp
    from .lower import LoweredKernel

    module, fn_name, source = identity_source(nm, sig, hooks, kernel_ref, language)
    op = ExternOp(stub, nm, sig, f"{module}:{fn_name}", source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(), language=language, **kwargs)
    op.hooks = hooks
    op.kernel_ref = kernel_ref
    op.__dict__["_tf_generated"] = LoweredKernel(nm, fn_name, module, "hooks", source, op, op.num_warps, None)
    return op


def hooks_of_op(op: Any) -> "tuple[Hook, ...]":
    """The hooks an op declares (none for anything but an extern with ``epilogue=``)."""
    return tuple(getattr(op, "hooks", ()) or ())


def hook_of(op: Any, value: str) -> "Hook | None":
    return next((h for h in hooks_of_op(op) if h.value == value), None)


_lock = threading.Lock()
_ops: "dict[str, Any]" = {}  # name (a content hash) -> the op: the same pair in two graphs is one operator


def _ann_of(p: Any, ren: "Mapping[str, str]", translate: Callable[[Any], Any]) -> Any:
    """The annotation of a hook argument ``p`` (a parameter of a region's kernel) as a parameter of the generated kernel (names in ``ren``)."""
    from .types import Out, StrideOf, Strided, TensorType

    if p.kind == "tensor":
        t = p.type
        base = TensorType(tuple(translate(d) for d in t.shape), t.dtype) if not t.devscalar else t
        if p.access == "out":
            return Out[base]
        return Strided[base] if p.strided else base
    if p.kind == "dim" and p.stride_of is not None:
        if len(p.stride_of) == 1:
            arg, axis = p.stride_of[0]
            return StrideOf(ren[arg], axis)
        axis = p.stride_of[0][1]
        return StrideOf(tuple(ren[a] for a, _ in p.stride_of), axis)
    if p.kind == "dim":
        return translate(p.dim)
    raise TilefishDeclarationError(f"an epilogue hook's argument '{p.name}' is a {p.kind}")  # pragma: no cover - lower_hook makes none


def _signature_of(extern: Any, out: str, out_type: Any, extra: "Sequence[tuple[str, Any]]") -> "tuple[Any, Any, dict[str, int]]":
    """The stub of a generated kernel: the extern's parameters (``out`` retyped to ``out_type``, a ``TensorType``, when given), then ``extra`` (name,
    annotation), keyword-only; and the ``max_elements`` its ``index_bits`` gives its tensors."""
    import inspect

    from .decorators import _parse_extern_stub
    from .signature import _resolve
    from .types import Marked, Out, TensorType

    if out_type is not None:
        out_type = TensorType(out_type.shape, out_type.dtype)
    base = inspect.signature(extern.fn)
    params = []
    for ip in base.parameters.values():
        ip = ip.replace(annotation=_resolve(ip.annotation, extern.fn, f"extern op '{extern.name}'"))  # (a string annotation, in the stub's module)
        if ip.name == out and out_type is not None:
            p = next(q for q in extern.signature.params if q.name == out)
            if p.type.dtype != out_type.dtype or tuple(p.type.shape) != tuple(out_type.shape):
                if p.tensor_map is not None and p.type.dtype != out_type.dtype:
                    raise TilefishDeclarationError(f"extern '{extern.name}': its output '{out}' is a TMA descriptor, whose element type a fused op does not change")
                ip = ip.replace(annotation=Marked("out", out_type, None, p.tensor_map) if p.tensor_map is not None else Out[out_type])
        params.append(ip)
    params += [inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, annotation=a) for n, a in extra]

    def stub() -> None: ...

    stub.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    stub.__name__ = stub.__qualname__ = extern.name
    stub.__module__ = MODULE
    _, sig, limits = _parse_extern_stub(stub, extern.name, extern.index_bits, False)
    return stub, sig, limits


def _generated_op(extern: Any, name: str, stub: Any, sig: Any, source: str, template: str, max_elements: "Mapping[str, int]", element_counts: "Mapping[str, str]",
                  reference: Any, line_map: "Sequence[Any]" = ()) -> Any:
    from .decorators import ExternOp
    from .lower import LoweredKernel

    module, fn_name = f"{MODULE}.{name}", f"tf_{name}"
    limits = dict(max_elements)
    op = ExternOp(
        stub, name, sig, f"{module}:{fn_name}", extern.num_warps, extern.num_stages, extern._grid_source,
        max_elements=limits, element_counts={**extern.element_counts, **element_counts}, fp_fusion=extern.fp_fusion,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(), index_bits=extern.index_bits, reference=reference, language=extern.language,
        persistent=extern.persistent, num_ctas=extern.num_ctas, global_scratch=extern.global_scratch, by_bucket=extern._by_bucket, sass=extern.sass,
    )  # fmt: skip
    op.kernel_ref = extern.kernel_ref
    op.hooked_of = extern  # the extern whose kernel this is, with another hook (tilefish._fusion, the advisor)
    op.__dict__["_tf_generated"] = LoweredKernel(name, fn_name, module, template, source, op, extern.num_warps, None, tuple(line_map))
    return op


def _source(header: "Sequence[str]", extern: Any, imports: "Sequence[str]", code: "Sequence[str]", fn_name: str, params: "Sequence[Any]",
            hooks: "Mapping[str, tuple[str, str]]") -> "tuple[str, int]":
    """The text of a generated module, and the line (0-based) its ``code`` starts at."""
    lines = _header_lines(header) + _module_head(extern.language, extern.kernel_ref, imports) + ["", ""] + IDENTITY + ["", ""]
    start = len(lines)
    lines += list(code) + ["", ""]
    stub_names = [p.name for p in extern.signature.params]
    lines += wrapper_lines(fn_name, extern.language, params, stub_names, hooks)
    return "\n".join(lines) + "\n", start


def _tuple_text(names: "Sequence[str]") -> str:
    if not names:
        return "()"
    return "(" + ", ".join(names) + ("," if len(names) == 1 else "") + ")"


def _identity_hooks(extern: Any, but: str) -> "dict[str, tuple[str, str]]":
    return {h.name: ("_tf_identity", "()") for h in hooks_of_op(extern) if h.name != but}


def acc_form(extern: Any, hook: Hook) -> Any:
    """The accumulator form of an extern with an ``acc`` hook (contract (ii), ``U(R_acc(P))``): the extern, and its f32 result as one more output,
    ``ep_acc`` (the shape of the hook's output, f32): its hook stores the accumulator tile and returns it as it is, so the kernel stores what it
    always does. ``<extern>__acc``."""
    from .types import Out, TensorType

    name = f"{extern.name}__acc"
    found = extern.__dict__.get("_tf_acc_form")
    if found is not None:
        return found
    out = next(p for p in extern.signature.params if p.name == hook.out)
    shape = out.type.shape
    last = shape[-1].folded()
    extra: "list[tuple[str, Any]]" = [("ep_acc", Out[TensorType(shape, _f32())])]
    if last.is_static:
        width = str(last.value)
    else:
        extra.append(("ep_size", last))
        width = "size"
    stub, sig, limits = _signature_of(extern, hook.out, None, extra)
    store = (
        ["    tl.store(acc + offs, tile, mask=mask)"]
        if hook.split == 1
        else [f"    tl.store(acc + offs + {k} * ({width} // {hook.split}), tile[{k}], mask=mask)" for k in range(hook.split)]
    )
    code = [
        "@triton.jit",
        "def _tf_epilogue(tile, rows, cols, row_mask, col_mask, args):",
        "    # the accumulator form: the f32 tile stored into ep_acc as it is, and returned as it is (the kernel then stores what it always does)",
        "    acc = args[0]",
        *(["    size = args[1]"] if not last.is_static else []),
        f"    offs = rows[:, None] * {width} + cols[None, :]",
        "    mask = row_mask[:, None] & col_mask[None, :]",
        *store,
        "    return tile",
    ]
    args = _tuple_text(["ep_acc"] + (["ep_size"] if not last.is_static else []))
    header = [
        f"Generated by tilefish: the accumulator form of the extern '{extern.name}' ({extern.kernel_ref}) through its hook {hook!r}. Do not edit.",
        "Its f32 result, the accumulator before the kernel's rounding, is the output ep_acc (docs/rfcs/0003-epilogue-fusion.md, contract (ii): U(R_acc(P))).",
    ]
    fn_name = f"tf_{name}"
    source, _ = _source(header, extern, (), code, fn_name, sig.params, {hook.name: ("_tf_epilogue", args), **_identity_hooks(extern, hook.name)})

    def reference(**kw: Any) -> Any:
        outs = _extern_reference(extern, kw)
        k = [p.name for p in extern.signature.tensor_outputs].index(hook.out)
        acc = _acc_reference(extern, hook, kw, outs[k])
        return (*outs, acc)

    op = _generated_op(extern, name, stub, sig, source, "acc", limits, {}, reference)
    op.acc_of = (extern, hook)
    with _lock:
        return extern.__dict__.setdefault("_tf_acc_form", op)


def _f32() -> Any:
    from .dtypes import f32

    return f32


def _extern_reference(extern: Any, kw: "Mapping[str, Any]") -> "list[Any]":
    from .errors import TilefishRuntimeError

    if extern.reference is None:
        raise TilefishRuntimeError(
            f"reference mode needs a reference for the extern op '{extern.name}' (its epilogue is fused): give @tilefish.extern a reference=<a torch "
            "function of the op's tensor inputs> that returns its Out tensors"
        )
    ins = {p.name: kw[p.name] for p in extern.signature.tensor_inputs if p.name in kw}
    res = extern.reference(**ins)
    return list(res) if isinstance(res, (tuple, list)) else [res]


def _acc_reference(extern: Any, hook: Hook, kw: "Mapping[str, Any]", rounded: Any) -> Any:
    """The f32 result of the extern's output in reference mode: the hook's ``reference=``, else the extern's reference widened."""
    if hook.reference is not None:
        ins = {p.name: kw[p.name] for p in extern.signature.tensor_inputs if p.name in kw}
        return hook.reference(**ins)
    return rounded.float()


def hooked(
    extern: Any,
    hook: Hook,
    kernel: Any,
    *,
    sources: "Mapping[str, tuple[str, int | None]]",
    main: "int | None",
    out_type: Any,
    translate: Callable[[Any], Any],
) -> Any:
    """The op of ``extern`` with the region ``kernel`` (a region's synthetic ``@tilefish.kernel``, :mod:`tilefish._fusion`) in its hook ``hook``.

    ``sources`` says, for each parameter of ``kernel`` the hook holds, what it is: ``("tile", k)`` (the tile, or chunk ``k`` of a split hook: ``k``
    ``None`` for the only one) or ``("rounded", k)`` (an ``acc`` hook's tile rounded to the output's type, as the kernel rounds it). ``main`` is the
    output of ``kernel`` the kernel stores into the hook's output (``None``: the tile, as the extern stores it; every output of the region is then a
    side output); ``out_type`` is the type of the hook's output then (its dtype, and the shape of one chunk for a split hook; ``None``: the
    extern's). ``translate`` maps a dimension of the region's types (the graph's) to the extern's. The parameters
    of the op are the extern's, then the hook's arguments (``ep_*``), keyword-only. Its name is ``<extern>_epi_<hash>`` of its content."""
    from .aot import declaration_digest
    from .lower import lower_hook

    split = hook.split
    out = next(p for p in extern.signature.params if p.name == hook.out)
    tl_out = _tl_name(out.type.dtype)

    def expr(src: "tuple[str, int | None]") -> str:
        what, k = src
        t = "tile" if k is None else f"tile[{k}]"
        return f"{t}.to(tl.{tl_out})" if what == "rounded" else t

    tiles = {name: expr(src) for name, src in sources.items()}
    returns = "tile" if main is None else ("main-tuple" if split > 1 else "main")
    stub_names = {p.name for p in extern.signature.params}
    low = lower_hook(kernel, tiles=tiles, main=main, returns=returns, gluon=extern.language == "gluon")
    ren: "dict[str, str]" = {}
    for p in low.params:
        nm, i = f"ep_{p.name}", 0
        while nm in stub_names or nm in ren.values():
            i += 1
            nm = f"ep_{p.name}_{i}"
        ren[p.name] = nm
    extra = [(ren[p.name], _ann_of(p, ren, translate)) for p in low.params]
    stub, sig, limits = _signature_of(extern, hook.out, out_type, extra)
    limits.update({ren[k]: v for k, v in low.max_elements.items()})  # (the hook's offsets are i32, whatever the extern's index_bits)
    args = _tuple_text([ren[p.name] for p in low.params])
    header = [
        f"Generated by tilefish: the extern '{extern.name}' ({extern.kernel_ref}) with the region '{kernel.name}' in its hook {hook!r}. Do not edit.",
        "docs/rfcs/0003-epilogue-fusion.md: "
        + (
            "contract (i): the hook sees the tile the kernel rounded, so this writes the bytes of the extern followed by the region"
            if hook.value == "out"
            else "contract (ii): the hook sees the f32 accumulator, so this writes the bytes of the unfused composition of R_acc(P)"
        ),
        "The hook:",
        *[f"  {h}" for h in low.header],
        "Numerics contract: see the docstring of tilefish.lower (notes 20 to 22).",
    ]
    placeholder = "tf_EPILOGUE_OP"
    hooks = {hook.name: ("_tf_epilogue", args), **_identity_hooks(extern, hook.name)}
    draft, _ = _source(header, extern, low.imports, low.lines, placeholder, sig.params, hooks)
    content = {
        "extern": declaration_digest(extern),
        "hook": hook.canonical(),
        "source": draft,
        "params": [[p.name, p.kind, str(p.type) if p.type is not None else str(p.dim), p.access] for p in sig.params],
        "max_elements": sorted(low.max_elements.items()),
    }
    import json

    name = f"{extern.name}_epi_" + hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    with _lock:
        found = _ops.get(name)
    if found is not None:
        return found
    fn_name = f"tf_{name}"
    source, start = _source(header, extern, low.imports, low.lines, fn_name, sig.params, hooks)
    line_map = [None] * start + list(low.line_map)

    def reference(**kw: Any) -> Any:
        from . import runtime

        outs = _extern_reference(extern, kw)
        k_out = [p.name for p in extern.signature.tensor_outputs].index(hook.out)
        y = outs[k_out]
        tile = _acc_reference(extern, hook, kw, y) if hook.value == "acc" else y
        chunks = list(tile.chunk(split, dim=-1)) if split > 1 else [tile]
        rounded = [c.to(y.dtype) for c in chunks]
        args_in = []
        for p in kernel.signature.tensor_inputs:
            src = sources.get(p.name)
            if src is None:
                args_in.append(kw[ren[p.name]])
            else:
                what, k = src
                args_in.append((rounded if what == "rounded" else chunks)[k or 0])
        res = runtime.run(kernel, *args_in, mode="reference")
        res = list(res) if isinstance(res, tuple) else [res]
        side = [r for k, r in enumerate(res) if k != main]
        if main is not None:
            outs[k_out] = res[main]
        return (*outs, *side)

    op = _generated_op(extern, name, stub, sig, source, low.template, limits, {ren[k]: ren[v] for k, v in low.element_counts.items()}, reference, line_map)
    op.hook = hook
    op.region_kernel = kernel
    with _lock:
        return _ops.setdefault(name, op)


def _tl_name(dt: Any) -> str:
    from .lower import _TL

    return _TL[dt.name]


__all__ = ["ABI", "Hook", "hook"]
