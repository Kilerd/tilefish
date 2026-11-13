"""Typed Rust launchers for a manifest: ``emit(manifest, "launchers.rs")``; and a typed runner of a launch plan: ``emit_plan(directory, "runner.rs", pin=...)``.

The ``tilefish`` crate (``crates/tilefish``) loads a manifest directory and launches what it describes from Rust: it verifies
the digests, checks every precondition of the binary and launches through the CUDA driver. :func:`emit` writes the
part that is specific to one manifest: one module with a struct per binary (per stem, so per bucket) whose ``launch``
takes the tensors by name, with the element type in the Rust type, so that a ``bf16`` tensor cannot be passed where an
``f32`` one is wanted, and whose doc comments carry what the manifest says of the binary.

**Manifest schema 3.** Python writes manifest schema 3 (``docs/manifest-schema-3.md``) and the crate reads it, so the module launches
everything the manifest can describe:

* a Triton kernel (``kind: "triton"``) and a prebuilt PTX or cubin (``kind: "binary"``): the same struct, wrapping ``Kernel``. A prebuilt
  binary is launched with exactly its parameters (no Triton scratch slots) and its ``fixed`` arguments (a constant of the C ABI, written in
  the manifest) are passed by the launch: they are not parameters of ``launch``;
* a strided pointer (``strides`` and ``offset``, with the stride scalars ``stride_of``, possibly 64-bit): a parameter like any other, a
  ``TensorRef`` that is a dense tensor or a view made with ``TensorRef::from_cuda_strided`` or ``TensorRef::new_strided``. The crate
  derives the stride scalars from the view and checks ``max_extent`` and the alignment of its address;
* a device scalar (a pointer with the shape ``[]``): a ``TensorRef`` over one element, with the shape ``&[]`` or ``&[1]``;
* a ``library`` entry (a pinned cuBLASLt GEMM): a struct that wraps ``Gemm`` and whose ``launch`` takes ``a``, ``b``, ``bias`` when it has
  one, ``out`` and a ``workspace`` (a ``u8`` tensor of at least ``workspace_required()`` bytes).

It still **refuses** what is invalid, with the strictness of the crate's reader (``docs/manifest-schema-3.md``, section 10): ``fixed`` on a
Triton kernel, ``strides`` without ``offset`` or on a pointer that is written, a ``stride_of`` that no strided pointer owns, a fixed stride
other than a ``1`` on the last axis, ``max_extent`` where it does not belong, and so on.

It works from the manifest alone. A manifest records, for every pointer parameter, its declared shape
(``["Rows", 1024]``: a dimension expression, or the size of a static axis) and, for every entry, the constraints of its
named dimensions (``divisible_by``, ``buckets`` and the bucket this binary is built for), so ``launch`` takes nothing but
the tensors: the crate binds the dimensions from their shapes and derives every runtime scalar and the grid, like
``tilefish.run`` does, and refuses a call that does not fit. ``ops=`` (the operators the manifest was built from) is
optional: each declaration that has a binary or a library entry in the manifest is checked against it (its compile spec must be exactly
what the manifest's entry says, the declaration digest included), and gives the docs the tilefish signature.

What the module holds, in order:

* ``MANIFEST_DIGEST``: the digest of the manifest it was generated from. ``load`` of every struct refuses a manifest with
  another digest, so a launcher can only run the binaries it was typed from.
* with ``embed=True``, the manifest file and its cubins as ``include_bytes!`` (paths relative to the generated file) and
  ``load_embedded()``, which verifies them like a directory and pins them to ``MANIFEST_DIGEST``. (A library entry has no file.)
* one struct per stem of a kernel, then one per stem of a library entry (kernels and library entries share the namespace of
  ``(stem, arch)``). ``STEM`` and ``BUCKETS`` name the binary (bucket semantics are exact: ``BucketedRms96`` and
  ``BucketedRms512`` are different structs and there is no padding), ``SHAPES`` the declared shape of every tensor,
  ``load(&Manifest, &Arc<CudaContext>)`` loads the binary for the device of a context and ``launch(&CudaStream, tensors...)`` checks the call and
  launches, and ``unsafe launch_raw(CUstream, tensors...)`` does the same on a raw handle. A CUDA graph is captured with regions
  and ``launch_raw`` (a launch of cudarc slices on a capturing stream is an error). A library struct adds ``M``, ``N``, ``K``, ``BIAS``,
  ``WORKSPACE_BYTES`` and ``PINNED_WITH`` (the cuBLASLt version of each architecture's pin; its doc names the SM count each was pinned on) and ``gemm()``.

The output is deterministic: the same manifest gives the same bytes. Nothing in it depends on where the manifest was
written, except the ``include_bytes!`` paths with ``embed=True``.

What it trusts. The output is Rust source that a developer compiles and runs, and a manifest directory vouches for nothing by
itself: whoever can write it can put any text in any field. So :func:`emit` (and :func:`render`) take a *pinned* manifest only: a
directory with ``pin=<the manifest's digest>``, or a :class:`~tilefish.Manifest` that was loaded with a pin (or that
``tilefish.runtime.export`` returned, which has just been written from this process's builds). Then every field that reaches the
output is validated (``stem``, ``op``, architecture, file names, tensor and dimension names, dtypes, shapes, strides, constexprs, fixed
values, the grid, sizes, the fields of a library entry: each must be what it is for), so a manifest cannot write a doc line that breaks out
of its comment, an item, or a string literal into the generated file; and the doc comments are written a line at a time, with line breaks
that a field might carry split into lines that are all comments.

**Launch plans.** :func:`emit_plan` writes a typed runner of a ``plan.json`` (``docs/plan-schema-1.md``) for the crate's ``LaunchPlan`` and
``PlanRun``: a struct named after the graph whose ``load`` pins the plan (and so its manifest), and whose ``launch`` and ``capture`` take one field
per external tensor, with its element type, and one per device scalar. It takes a pinned plan only, and validates every name that reaches the
output, as :func:`emit` does.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Iterable

from .._build import spec_entry, write_file
from .._library import IMPLS as LIBRARY_IMPLS
from .._library import library_specs, validate_entry as validate_library_entry
from .._manifest import Manifest
from .._specvalid import parse_expression, parse_grid
from ..decorators import LibraryOp
from ..errors import TilefishManifestError, TilefishSpecError

RUST_ELEMENTS = {"bf16": "Bf16", "f16": "F16", "f32": "F32", "i32": "I32", "i64": "I64", "u8": "U8", "e4m3": "E4m3", "e5m2": "E5m2"}

# the size in bytes of an element of each dtype that has a Rust element type: the alignment a strided pointer may promise is 16 or this
_ITEMSIZE = {"bf16": 2, "f16": 2, "f32": 4, "i32": 4, "i64": 8, "u8": 1, "e4m3": 1, "e5m2": 1}

# what the binaries of one stem must agree on, whatever their architecture
_SHARED_FIELDS = (
    "op",
    "kind",
    "abi",
    "grid",
    "block",
    "num_warps",
    "num_stages",
    "constexprs",
    "buckets",
    "dims",
    "preconditions",
    "options",
    "declaration_digest",
    "source_sha256",
)

# what the library entries of one stem must agree on: everything but the device (its architecture and SM count, and the declaration digest, which
# covers the SM count) and what the pin chose on it
_LIBRARY_SHARED_FIELDS = (
    "op impl abi buckets dims preconditions dtypes compute_type scale_type transa transb m n k lda ldb ldc ldd epilogue workspace_bytes"
).split()

_RUST_KEYWORDS = frozenset(
    "as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub ref "
    "return self Self static struct super trait true type unsafe use where while abstract become box do final macro override priv "
    "typeof unsized virtual yield try gen stream _".split()
)  # `stream` is the name of the stream parameter of every launcher, `_` is not a name that can be used

# what a name in a manifest may be, to reach the generated source: plain ASCII, never what a lexer could read as something else
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*", re.ASCII)
_STEM = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", re.ASCII)
_ARCH = re.compile(r"sm_[0-9]{2,3}[a-z]?", re.ASCII)
_FILE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*", re.ASCII)
_DTYPE = re.compile(r"[a-z][a-z0-9_]*", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}", re.ASCII)
_U64_MAX = 2**64 - 1
_ACCESS = ("in", "out", "inout")
_KINDS = ("triton", "binary")
_FIXED_TYPES = ("i32", "i64", "f32")
_INT_RANGES = {"i32": (-(2**31), 2**31 - 1), "i64": (-(2**63), 2**63 - 1)}
_MAX_EXTENT = 2**31 - 1  # the `value` of every `max_extent` precondition
_PRECONDITIONS = ("pointer_alignment", "divisible", "static_size", "max_elements", "max_extent", "bucket")
_DEFAULT_ALIGNMENT = 16


@dataclass
class _Binary:
    """One stem of the manifest's kernels, as the generated code needs it."""

    stem: str
    entry: "dict[str, Any]"  # the entry of the first architecture: what the architectures agree on
    archs: "list[str]"
    struct: str = ""
    params: "list[dict[str, Any]]" = field(default_factory=list)  # the pointers of the ABI, in order
    scalars: "list[dict[str, Any]]" = field(default_factory=list)  # the runtime scalars (the ones `launch` derives)
    fixed: "list[dict[str, Any]]" = field(default_factory=list)  # the fixed arguments of a prebuilt binary
    arg_names: "dict[str, str]" = field(default_factory=dict)  # ABI name -> Rust name
    signature: "str | None" = None  # the tilefish signature of the declaration, when it was given

    @property
    def prebuilt(self) -> bool:
        return self.entry["kind"] == "binary"


@dataclass
class _Gemm:
    """One stem of the manifest's ``library`` section (a pinned cuBLASLt GEMM), as the generated code needs it."""

    stem: str
    entry: "dict[str, Any]"  # the entry of the first architecture
    entries: "list[dict[str, Any]]"  # one per architecture, sorted by it
    struct: str = ""
    params: "list[dict[str, Any]]" = field(default_factory=list)  # a, b, [bias], out
    arg_names: "dict[str, str]" = field(default_factory=dict)  # ABI name -> Rust name
    workspace: str = "workspace"  # the name of the workspace parameter
    signature: "str | None" = None

    @property
    def archs(self) -> "list[str]":
        return [e["arch"] for e in self.entries]


def _rs_str(text: str) -> str:
    """A Rust string literal for plain ASCII text (identifiers, digits and the characters of dimension expressions).

    Anything else is refused, and ``\\`` and ``"`` are escaped: no text can end the literal.
    """
    if not isinstance(text, str) or not text.isascii() or not text.isprintable():
        raise TilefishManifestError(f"cannot write {text!r} in generated Rust source: it is not printable ASCII")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# what the generated module imports from the crate, and `Self`: a struct of these names would clash with them
_RESERVED_STRUCTS = frozenset(
    {
        "Self",
        "Arc",
        "Arch",
        "CUstream",
        "CudaContext",
        "CudaStream",
        "Error",
        "Gemm",
        "Kernel",
        "Manifest",
        "Pin",
        "TensorRef",
        "TensorMut",
        "Result",
        "Option",
        "Some",
        "None",
        "Ok",
        "Err",
    }
    | set(RUST_ELEMENTS.values())
)


def _camel(stem: str) -> str:
    """The name of the struct of a stem: ``rowop.S32`` is ``RowopS32``. One that would clash with an import gets ``Op`` after it."""
    pieces = [p for p in re.split(r"[._]+", stem) if p]
    name = "".join(p[0].upper() + p[1:] for p in pieces)
    name = name if name and not name[0].isdigit() else "K" + name
    return name + "Op" if name in _RESERVED_STRUCTS else name


def _snake_ident(name: str) -> str:
    return name + "_" if name in _RUST_KEYWORDS else name


def _scalar_text(value: Any) -> str:
    if isinstance(value, dict):  # a tagged non-finite float
        return str(value["float"])
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value)


def _bad(who: str, what: str) -> Any:
    raise TilefishManifestError(f"cannot write Rust launchers for {who}: {what}")


def _name(who: str, where: str, value: Any, pattern: "re.Pattern[str]" = _IDENT) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        _bad(who, f"{where} must be a plain ASCII name, got {value!r}")
    return value


def _count(who: str, where: str, value: Any, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= _U64_MAX:
        _bad(who, f"{where} must be an integer from {minimum} to 2**64 - 1, got {value!r}")
    return value


def _expression(who: str, where: str, value: Any) -> str:
    try:
        parse_expression(value, where)
    except TilefishSpecError as e:
        _bad(who, f"{where} is not a dimension expression ({e})")
    return value


def _field(who: str, where: str, owner: Any, key: str, kind: type) -> Any:
    value = owner.get(key) if isinstance(owner, dict) else None
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        _bad(who, f"{where} is missing or is not a {kind.__name__}")
    return value


def _shape(who: str, where: str, shape: "list[Any]") -> None:
    for axis, d in enumerate(shape):
        if isinstance(d, str):
            _expression(who, f"{where}[{axis}]", d)
        else:
            _count(who, f"{where}[{axis}]", d)


def _validate_pointer(who: str, where: str, p: "dict[str, Any]") -> None:
    """A pointer of the ABI: a dense tensor, a strided view (``strides`` and ``offset``, on an ``in`` pointer) or a device scalar (shape ``[]``)."""
    _name(who, f"{where}.dtype", p.get("dtype"), _DTYPE)
    access = p.get("access")
    if access not in _ACCESS:
        _bad(who, f"{where}.access must be one of {list(_ACCESS)}, got {access!r}")
    shape = _field(who, f"{where}.shape", p, "shape", list)
    _shape(who, f"{where}.shape", shape)
    if "strides" in p or "offset" in p:
        if "strides" not in p or "offset" not in p:
            _bad(who, f"{where} has {'strides' if 'strides' in p else 'offset'} without {'offset' if 'strides' in p else 'strides'}: a strided pointer has both")
        if access != "in":
            _bad(who, f"{where} is a strided pointer with access {access!r}: only an 'in' pointer may be strided")
        strides = _field(who, f"{where}.strides", p, "strides", list)
        if not shape or len(strides) != len(shape):
            _bad(who, f"{where}.strides must have one entry for each of the {len(shape)} axes of its shape (and at least one), got {len(strides)}")
        for axis, s in enumerate(strides):
            if isinstance(s, str):
                _name(who, f"{where}.strides[{axis}]", s)
            elif isinstance(s, bool) or not isinstance(s, int) or s != 1:
                _bad(who, f"{where}.strides[{axis}] must be the name of a stride dimension or the fixed stride 1, got {s!r}")
            elif axis != len(strides) - 1:
                _bad(who, f"{where}.strides[{axis}] is the fixed stride 1, which only the last axis may have")
        _name(who, f"{where}.offset", p["offset"])
    elif not shape and access != "in":
        _bad(who, f"{where} is a device scalar (the shape []) with access {access!r}: a device scalar is an 'in' pointer")
    if "alias" in p:  # an out pointer that may be written over an in pointer: the launch takes the two as exactly one tensor (a region, DevicePtr)
        _name(who, f"{where}.alias", p["alias"])
        if access != "out":
            _bad(who, f"{where} may alias {p['alias']!r} with access {access!r}: only an 'out' pointer may be written over an input")


def _validate_scalars(who: str, kind: str, params: "list[dict[str, Any]]") -> None:
    """The scalars and fixed arguments of the ABI, once every pointer is known (a stride scalar names the pointer it carries a stride of)."""
    by_name = {p["name"]: p for p in params}
    carried: "dict[tuple[str, int], str]" = {}  # (strided pointer, axis) -> the scalar that carries its stride
    for i, p in enumerate(params):
        where = f"abi.params[{i}] ('{p['name']}')"
        pkind = p.get("kind")
        if pkind == "pointer":
            continue
        if pkind == "scalar":
            if p.get("type") not in ("i32", "i64"):
                _bad(who, f"{where}.type must be 'i32' or 'i64', got {p.get('type')!r}")
            _expression(who, f"{where}.dim", p.get("dim"))
            if "count_of" in p and "stride_of" in p:
                _bad(who, f"{where} carries both an element count (count_of) and a stride (stride_of)")
            if "count_of" in p:
                target = _name(who, f"{where}.count_of", p["count_of"])
                if target not in by_name:
                    _bad(who, f"{where}.count_of must name a parameter, got {target!r}")
            if "stride_of" in p:
                so = p["stride_of"]
                if isinstance(so, list) and len(so) < 2:
                    _bad(who, f"{where}.stride_of is a list of the views that share the stride: it has two or more, got {so!r}")
                for item in so if isinstance(so, list) else [so]:  # one view, or the views that share the stride
                    if not isinstance(item, dict) or set(item) != {"arg", "axis"}:
                        _bad(who, f"{where}.stride_of must be an object with exactly 'arg' and 'axis', or a list of them, got {so!r}")
                    owner = by_name.get(_name(who, f"{where}.stride_of.arg", item["arg"]))
                    axis = item["axis"]
                    if isinstance(axis, bool) or not isinstance(axis, int) or axis < 0:
                        _bad(who, f"{where}.stride_of.axis must be an integer from 0, got {axis!r}")
                    if owner is None or owner.get("kind") != "pointer" or "strides" not in owner:
                        _bad(who, f"{where}.stride_of names '{item['arg']}', which is not a strided pointer")
                    if axis >= len(owner["strides"]):
                        _bad(who, f"{where}.stride_of.axis is {axis}, but '{item['arg']}' has {len(owner['strides'])} axes")
                    if owner["strides"][axis] != p["dim"]:
                        _bad(who, f"{where}.dim is {p['dim']!r}, but the stride of axis {axis} of '{item['arg']}' is {owner['strides'][axis]!r}")
                    if (item["arg"], axis) in carried:
                        _bad(who, f"axis {axis} of '{item['arg']}' has two stride scalars, '{carried[(item['arg'], axis)]}' and '{p['name']}'")
                    carried[(item["arg"], axis)] = p["name"]
        elif pkind == "fixed":
            if kind != "binary":
                _bad(who, f"{where} is a fixed argument, which only a prebuilt binary has (this is a {kind} kernel)")
            _validate_fixed(who, where, p)
        else:
            _bad(who, f"{where}.kind must be 'pointer', 'scalar' or 'fixed', got {pkind!r}")
    for p in params:
        for axis, s in enumerate(p.get("strides", ()) if p.get("kind") == "pointer" else ()):
            if isinstance(s, str) and (p["name"], axis) not in carried:
                _bad(who, f"the stride '{s}' of axis {axis} of '{p['name']}' is carried by no scalar (stride_of)")


def _validate_fixed(who: str, where: str, p: "dict[str, Any]") -> None:
    ty = p.get("type")
    if ty not in _FIXED_TYPES:
        _bad(who, f"{where}.type must be one of {list(_FIXED_TYPES)}, got {ty!r}")
    if "value" not in p:
        _bad(who, f"{where} has no value")
    if "dim" in p or "count_of" in p or "stride_of" in p:
        _bad(who, f"{where} is a fixed argument: it has a value and nothing that a launch derives (dim, count_of, stride_of)")
    value = p["value"]
    if ty == "f32":
        if isinstance(value, dict):
            if set(value) != {"float"} or value["float"] not in ("inf", "-inf", "nan"):
                _bad(who, f"{where}.value is not a number or a tagged float: {value!r}")
        elif not isinstance(value, float) or not math.isfinite(value):
            _bad(who, f"{where}.value of an f32 must be a finite JSON number with a fraction or an exponent, or a tagged float, got {value!r}")
    else:
        lo, hi = _INT_RANGES[ty]
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            _bad(who, f"{where}.value of an {ty} must be an integer from {lo} to {hi}, got {value!r}")


def _validate_preconditions(who: str, entry: "dict[str, Any]", pointers: "dict[str, dict[str, Any]]") -> None:
    """What the reader requires of the preconditions of an ABI with strided pointers and device scalars (section 6 and 10 of the schema)."""
    aligned: "set[str]" = set()
    extents: "set[str]" = set()
    for i, pre in enumerate(_field(who, "preconditions", entry, "preconditions", list)):
        where = f"preconditions[{i}]"
        kind = pre.get("kind") if isinstance(pre, dict) else None
        if kind not in _PRECONDITIONS:
            _bad(who, f"{where}.kind must be one of {list(_PRECONDITIONS)}, got {kind!r}")
        if kind in ("bucket", "divisible"):
            continue
        arg = pre.get("arg")
        p = pointers.get(arg) if isinstance(arg, str) else None
        if p is None:
            _bad(who, f"{where} ({kind}) names '{arg}', which is not a pointer of the ABI")
        strided, scalar = "strides" in p, not p["shape"]
        if pre.get("index") != p.get("index") or isinstance(pre.get("index"), bool):
            _bad(who, f"{where}.index is {pre.get('index')!r}, but '{arg}' is parameter {p.get('index')!r}")
        if kind == "pointer_alignment":
            if arg in aligned:
                _bad(who, f"'{arg}' has two pointer_alignment preconditions")
            aligned.add(arg)
            allowed = {_DEFAULT_ALIGNMENT} | ({_ITEMSIZE[p["dtype"]]} if strided and p["dtype"] in _ITEMSIZE else set())
            if pre.get("bytes") not in allowed or isinstance(pre.get("bytes"), bool):
                _bad(who, f"{where}.bytes must be {' or '.join(map(str, sorted(allowed)))} for '{arg}', got {pre.get('bytes')!r}")
        elif kind == "max_extent":  # (a kernel with index_bits=64 has none: at most once)
            if not strided or arg in extents:
                _bad(who, f"{where}: max_extent belongs once on a strided pointer, and '{arg}' is {'not strided' if not strided else 'given it twice'}")
            extents.add(arg)
            if isinstance(pre.get("value"), bool) or pre.get("value") != _MAX_EXTENT:
                _bad(who, f"{where}.value must be {_MAX_EXTENT}, got {pre.get('value')!r}")
        elif kind == "max_elements" and (strided or scalar):
            _bad(who, f"{where}: max_elements does not belong on '{arg}', which is {'strided (it has max_extent)' if strided else 'a device scalar'}")
        elif kind == "static_size" and scalar:
            _bad(who, f"{where}: a device scalar ('{arg}') has no static_size")
    if aligned != set(pointers):
        _bad(who, f"every pointer needs exactly one pointer_alignment precondition; missing for {sorted(set(pointers) - aligned)}")


def _validate_dims(who: str, entry: "dict[str, Any]") -> None:
    dims = _field(who, "dims", entry, "dims", dict)
    for name, c in dims.items():
        _name(who, "a dimension name", name)
        if not isinstance(c, dict):
            _bad(who, f"the constraints of dimension '{name}' are not an object")
        for key in ("divisible_by", "bucket"):
            if key in c:
                _count(who, f"dims['{name}'].{key}", c[key], 1)
        if "buckets" in c:
            listed = c["buckets"]
            if not isinstance(listed, list):
                _bad(who, f"dims['{name}'].buckets is not a list")
            for v in listed:
                _count(who, f"dims['{name}'].buckets", v, 1)
    buckets = _field(who, "buckets", entry, "buckets", dict)
    for name, v in buckets.items():
        _name(who, "a bucket's dimension name", name)
        _count(who, f"buckets['{name}']", v, 1)


def _validate_binary_section(who: str, entry: "dict[str, Any]") -> None:
    """The ``binary`` section of a prebuilt entry: ``{"source": "cubin"}`` or ``{"source": "ptx", "ptxas_flags": [...]}``, and its digest."""
    b = entry.get("binary")
    ok = isinstance(b, dict) and b.get("source") in ("ptx", "cubin")
    if ok and b["source"] == "ptx":
        flags = b.get("ptxas_flags")
        ok = set(b) == {"source", "ptxas_flags"} and isinstance(flags, list) and all(isinstance(f, str) for f in flags)
    elif ok:
        ok = set(b) == {"source"}
    if not ok:
        _bad(who, f"binary must be {{'source': 'cubin'}} or {{'source': 'ptx', 'ptxas_flags': [...]}}, got {b!r}")
    digest = entry.get("source_digest")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest) or entry.get("source_digest_kind") != "tilefish.binary.v1":
        _bad(who, "a prebuilt binary has the sha256 of its PTX or cubin as source_digest, with source_digest_kind 'tilefish.binary.v1'")
    for key in ("options", "num_warps", "num_stages"):
        if key in entry:
            _bad(who, f"a prebuilt binary has no '{key}' (that is a Triton compile setting)")


def _validate_entry(entry: Any) -> None:
    """Refuse an entry with any field that must not reach generated Rust source as it is (see the module docstring)."""
    if not isinstance(entry, dict):
        _bad("an entry", f"it is not an object: {entry!r}")
    who = f"the entry {entry.get('stem')!r}"
    stem = _name(who, "the stem", entry.get("stem"), _STEM)
    arch = _name(who, "the architecture", entry.get("arch"), _ARCH)
    who = f"'{stem}' ({arch})"
    _name(who, "the operator", entry.get("op"))
    _name(who, "the cubin file name", entry.get("cubin"), _FILE)
    kind = entry.get("kind")
    if kind not in _KINDS:
        _bad(who, f"kind must be one of {list(_KINDS)}, got {kind!r}")
    abi = _field(who, "abi", entry, "abi", dict)
    slots = 2 if kind == "triton" else 0
    scratch = abi.get("scratch")
    if type(abi.get("scratch_slots")) is not int or abi["scratch_slots"] != slots or not isinstance(scratch, list) or len(scratch) != slots:
        _bad(who, f"a {kind} kernel has {slots} scratch slots, but abi.scratch_slots is {abi.get('scratch_slots')!r} and abi.scratch has {scratch!r}")
    if kind == "binary":
        _validate_binary_section(who, entry)
    params = _field(who, "abi.params", abi, "params", list)
    names: "set[str]" = set()
    for i, p in enumerate(params):
        where = f"abi.params[{i}]"
        name = _name(who, f"{where}.name", p.get("name") if isinstance(p, dict) else None)
        if name in names:
            _bad(who, f"{where}.name '{name}' is used twice")
        names.add(name)
    pointers = {}
    for i, p in enumerate(params):
        if p.get("kind") == "pointer":
            _validate_pointer(who, f"abi.params[{i}] ('{p['name']}')", p)
            pointers[p["name"]] = p
    _validate_scalars(who, kind, params)
    _validate_preconditions(who, entry, pointers)
    _validate_dims(who, entry)
    for i, c in enumerate(_field(who, "constexprs", entry, "constexprs", list)):
        where = f"constexprs[{i}]"
        _name(who, f"{where}.name", c.get("name") if isinstance(c, dict) else None)
        value = c.get("value")
        if isinstance(value, dict):
            if set(value) != {"float"} or value["float"] not in ("inf", "-inf", "nan"):
                _bad(who, f"{where}.value is not a number, a bool or a tagged float: {value!r}")
        elif not isinstance(value, (bool, int, float)):
            _bad(who, f"{where}.value must be a number or a bool, got {value!r}")
        elif isinstance(value, float) and value != value:
            _bad(who, f"{where}.value is not a finite number")
        if "dtype" in c:
            _name(who, f"{where}.dtype", c["dtype"], _DTYPE)
    try:
        parse_grid(entry.get("grid"))
    except TilefishSpecError as e:
        _bad(who, f"the grid is not a grid expression ({e})")
    block = _field(who, "block", entry, "block", list)
    if len(block) != 3:
        _bad(who, f"block must have three sizes, got {block!r}")
    for k, b in enumerate(block):
        _count(who, f"block[{k}]", b, 1)
    _count(who, "shared", entry.get("shared"))


def _validate_library_entry(entry: Any) -> None:
    """Refuse a ``library`` entry that the strict reader of the section refuses, or with a field that must not reach generated source as it is."""
    if not isinstance(entry, dict):
        _bad("a library entry", f"it is not an object: {entry!r}")
    who = f"the library entry {entry.get('stem')!r}"
    stem = _name(who, "the stem", entry.get("stem"), _STEM)
    arch = _name(who, "the architecture", entry.get("arch"), _ARCH)
    who = f"'{stem}' ({arch})"
    try:
        validate_library_entry(entry)  # the fields, their types and the cross-field equalities of docs/manifest-schema-3.md, section 9
    except TilefishManifestError as e:
        _bad(who, str(e))
    _name(who, "the operator", entry.get("op"))
    impl = entry.get("impl")
    if impl not in LIBRARY_IMPLS:
        _bad(who, f"impl must be one of {sorted(LIBRARY_IMPLS)}, got {impl!r}")
    for key in ("m", "n", "k", "workspace_bytes", "cublaslt_version", "sm_count"):
        _count(who, key, entry.get(key), 1)
    params = _field(who, "abi.params", _field(who, "abi", entry, "abi", dict), "params", list)
    if len(params) != (4 if LIBRARY_IMPLS[impl] else 3):
        _bad(who, f"{impl} has {4 if LIBRARY_IMPLS[impl] else 3} pointers, the ABI lists {len(params)}")
    for i, p in enumerate(params):
        where = f"abi.params[{i}]"
        _name(who, f"{where}.name", p.get("name") if isinstance(p, dict) else None)
        _validate_pointer(who, f"{where} ('{p['name']}')", p)
        if p["dtype"] != "bf16" or p["access"] != ("out" if i == len(params) - 1 else "in") or "strides" in p:
            _bad(who, f"{where} must be a dense bf16 pointer with access {'out' if i == len(params) - 1 else 'in'}")
    _validate_dims(who, entry)


def _group(manifest: Manifest) -> "tuple[list[_Binary], list[_Gemm]]":
    for artifact in manifest.artifacts:
        _validate_entry(artifact.entry)
    for lib in manifest.library_entries:
        _validate_library_entry(lib)
    by_stem: "dict[str, list[dict[str, Any]]]" = {}
    for artifact in manifest.artifacts:
        by_stem.setdefault(artifact.stem, []).append(artifact.entry)
    library_by_stem: "dict[str, list[dict[str, Any]]]" = {}
    for lib in manifest.library_entries:
        library_by_stem.setdefault(lib["stem"], []).append(lib)
    both = sorted(set(by_stem) & set(library_by_stem))
    if both:
        raise TilefishManifestError(f"{both} are both kernels and library entries of the manifest: a stem names one or the other")
    out = []
    for stem in sorted(by_stem):
        entries = sorted(by_stem[stem], key=lambda e: e["arch"])
        first = entries[0]
        for other in entries[1:]:
            for key in _SHARED_FIELDS:
                if other.get(key) != first.get(key):
                    raise TilefishManifestError(
                        f"the binaries of '{stem}' for {first['arch']} and {other['arch']} disagree about '{key}': a launcher is "
                        "typed from one signature, and the binaries of a stem must share it"
                    )
        binary = _Binary(stem, first, [e["arch"] for e in entries], struct=_camel(stem))
        binary.params = [p for p in first["abi"]["params"] if p["kind"] == "pointer"]
        binary.scalars = [p for p in first["abi"]["params"] if p["kind"] == "scalar"]
        binary.fixed = [p for p in first["abi"]["params"] if p["kind"] == "fixed"]
        for p in binary.params:
            if p["dtype"] not in RUST_ELEMENTS:
                raise TilefishManifestError(
                    f"'{stem}': the tensor '{p['name']}' has dtype {p['dtype']}, which has no element type in Rust "
                    "(packed formats cannot be run yet)"
                )
        names = [_snake_ident(p["name"]) for p in binary.params]
        if len(set(names)) != len(names):
            raise TilefishManifestError(f"'{stem}': two tensor names collide as Rust identifiers: {names}")
        binary.arg_names = {p["name"]: n for p, n in zip(binary.params, names)}
        out.append(binary)
    gemms = []
    for stem in sorted(library_by_stem):
        entries = sorted(library_by_stem[stem], key=lambda e: e["arch"])
        first = entries[0]
        for other in entries[1:]:
            for key in _LIBRARY_SHARED_FIELDS:
                if other.get(key) != first.get(key):
                    raise TilefishManifestError(
                        f"the library entries of '{stem}' for {first['arch']} and {other['arch']} disagree about '{key}': a launcher is "
                        "typed from one signature, and the entries of a stem must share it"
                    )
        gemm = _Gemm(stem, first, entries, struct=_camel(stem))
        gemm.params = list(first["abi"]["params"])
        # the names of the tensors, and `workspace`, which no tensor of the signature may take
        names = [_snake_ident(p["name"]) for p in gemm.params]
        if len(set(names)) != len(names) or gemm.workspace in names:
            raise TilefishManifestError(f"'{stem}': two tensor names collide as Rust identifiers, or one of them is 'workspace': {names}")
        gemm.arg_names = {p["name"]: n for p, n in zip(gemm.params, names)}
        gemms.append(gemm)
    structs = [b.struct for b in out] + [g.struct for g in gemms]
    if len(set(structs)) != len(structs):
        dup = sorted({s for s in structs if structs.count(s) > 1})
        raise TilefishManifestError(f"two stems of the manifest give the same Rust name {dup}")
    return out, gemms


def _check_declarations(binaries: "list[_Binary]", gemms: "list[_Gemm]", ops: "Iterable[Any]") -> None:
    """Check each declaration against the manifest entries of its operator, and take its signature for the docs."""
    from ..aot import _first_difference, _show
    from ..runtime import _target

    declared: "dict[str, tuple[Any, Any]]" = {}  # op name -> (target, launched extern)
    library: "dict[str, LibraryOp]" = {}  # op name -> a library op that has an implementation
    for op in ops:
        if isinstance(op, LibraryOp):
            if op.impl is not None:
                library[op.name] = op
            continue
        target = _target(op)
        extern = target.launch_extern()
        declared[extern.name] = (target, extern)
    for binary in binaries:
        found = declared.get(binary.entry["op"])
        if found is None:
            continue
        target, extern = found
        specs = {s.stem: s for s in target.compile_specs()}
        spec = specs.get(binary.stem)
        if spec is None:
            raise TilefishManifestError(
                f"the declaration of '{extern.name}' has no binary '{binary.stem}' (it has {sorted(specs)}): it is not what the "
                "manifest was built from"
            )
        expected = json.loads(json.dumps(spec_entry(spec)))
        got = {k: binary.entry.get(k) for k in expected}
        difference = _first_difference(expected, got, "")
        if difference is not None:
            path, want, have = difference
            raise TilefishManifestError(
                f"the declaration of '{extern.name}' is not what the manifest's binary '{binary.stem}' was built from: first "
                f"difference at {path}: the manifest has {_show(have)}, the declaration gives {_show(want)}"
            )
        public = target.public
        binary.signature = f"{target.name}{public}" if target.kernel is not None else f"{extern.name}{extern.signature}"
    for gemm in gemms:
        op = library.get(gemm.entry["op"])
        if op is None:
            continue
        specs = {s.stem: s for s in library_specs(op)}
        spec = specs.get(gemm.stem)
        if spec is None:
            raise TilefishManifestError(
                f"the declaration of '{op.name}' has no library entry '{gemm.stem}' (it has {sorted(specs)}): it is not what the "
                "manifest was built from"
            )
        for entry in gemm.entries:
            expected = json.loads(json.dumps(spec.entry(entry["sm_count"])))  # the declaration on the device the entry was pinned on
            got = {k: entry.get(k) for k in expected}
            difference = _first_difference(expected, got, "")
            if difference is not None:
                path, want, have = difference
                raise TilefishManifestError(
                    f"the declaration of '{op.name}' is not what the manifest's library entry '{gemm.stem}' ({entry['arch']}) was built from: "
                    f"first difference at {path}: the manifest has {_show(have)}, the declaration gives {_show(want)}"
                )
        gemm.signature = f"{op.name}{op.signature}"


def _wrap(text: str) -> "list[str]":
    """A paragraph of a doc comment as lines of at most 128 characters (deterministic; a word that is longer stays whole)."""
    return textwrap.wrap(text, width=128, break_long_words=False, break_on_hyphens=False)


def _shape_text(shape: "list[Any]") -> str:
    return ", ".join(str(d) for d in shape)


def _block_text(block: "list[int]") -> str:
    return f"{block[0]} threads" if block[1] == block[2] == 1 else f"{block[0]}x{block[1]}x{block[2]} threads"


def _pointer_line(binary: _Binary, p: "dict[str, Any]") -> str:
    name = binary.arg_names[p["name"]]
    if not p["shape"]:
        return f"- `{name}`: in, {p['dtype']}, a device scalar (one element: the shape `[]` or `[1]`)"
    line = f"- `{name}`: {p['access']}, {p['dtype']}, `[{_shape_text(p['shape'])}]`"
    if "strides" in p:
        line += f", a strided view (strides `[{_shape_text(p['strides'])}]`)"
    return line


def _doc(binary: _Binary) -> "list[str]":
    e = binary.entry
    lines: "list[str]" = []
    if binary.signature:
        lines += [f"`{binary.signature}`", ""]
    if binary.prebuilt:
        source = "PTX that ptxas assembled" if e["binary"]["source"] == "ptx" else "a cubin used as it is"
        lines += _wrap(f"The prebuilt binary `{binary.stem}` of operator `{e['op']}`, built for {', '.join(binary.archs)}: {source}, not compiled by Triton.")
        lines += _wrap(f"Its source file has sha256 {e['source_digest']}. It is launched with exactly its parameters: there are no Triton scratch slots.")
    else:
        lines.append(f"The binary `{binary.stem}` of operator `{e['op']}`, built for {', '.join(binary.archs)}.")
    lines.append("")
    lines.append("Tensors, in the order of `launch`:")
    for p in binary.params:
        lines.append(_pointer_line(binary, p))
    strided = [p for p in binary.params if "strides" in p]
    if strided:
        limited = any(pre["kind"] == "max_extent" for pre in e["preconditions"])
        lines.append("")
        lines += _wrap(
            "A strided tensor is a view: make it with `TensorRef::from_cuda_strided` or `TensorRef::new_strided` (its element strides, and the "
            "element offset of its first element); a dense tensor is accepted too. The strides are not arguments of `launch`: it derives "
            "the stride scalars from the view (the stride of a size-1 axis made dense) and refuses a negative stride, a last axis whose stride "
            "is not 1, a stride that does not fit the integer the binary takes it as"
            + (", a view that spans more than 2147483647 elements (`max_extent`: the binary indexes with 32 bits)" if limited else " (the binary indexes with 64 bits: its views have no `max_extent`)")
            + ", and an address that is not aligned as the binary requires:"
        )
        for p in strided:
            align = next(pre["bytes"] for pre in e["preconditions"] if pre["kind"] == "pointer_alignment" and pre["arg"] == p["name"])
            lines.append(f"- `{binary.arg_names[p['name']]}`: {align} bytes")
    if any(not p["shape"] for p in binary.params):
        lines.append("")
        lines += _wrap(
            "A device scalar is a `TensorRef` over one element (made with the shape `&[]` or `&[1]`), 16-byte aligned: the kernel reads its value "
            "when it runs, so the value may be written after the launch is enqueued."
        )
    constraints = []
    for name, c in e["dims"].items():
        if "bucket" in c:
            constraints.append(f"`{name}` is exactly {c['bucket']} (a bucket of {c['buckets']}; there is no padding, and the other buckets are other structs)")
        elif "divisible_by" in c:
            constraints.append(f"`{name}` is a multiple of {c['divisible_by']}")
    sizes = [
        f"axis {i} of `{binary.arg_names[p['name']]}` is {d}" for p in binary.params for i, d in enumerate(p["shape"]) if isinstance(d, int)
    ]
    if sizes:
        lines += ["", "Baked into the binary: " + "; ".join(sizes) + "."]
    if e["dims"]:
        lines.append("")
        lines.append("`launch` binds the dimensions from the sizes of the tensors and refuses a call where they disagree.")
        if constraints:
            lines.append("Constraints: " + "; ".join(constraints) + ".")
    if e["constexprs"]:
        lines.append("")
        lines.append(
            "Compiled in: "
            + ", ".join(
                f"`{c['name']}` = {_scalar_text(c['value'])}" + (f" ({c['dtype']})" if "dtype" in c else "") for c in e["constexprs"]
            )
            + "."
        )
    if binary.fixed:
        lines.append("")
        lines += _wrap(
            "Fixed arguments (their values are in the manifest, and every launch passes them: they are not parameters of `launch`): "
            + ", ".join(f"`{f['name']}` = {_scalar_text(f['value'])} ({f['type']})" for f in binary.fixed)
            + "."
        )
    lines.append("")
    lines.append(f"Grid `{e['grid']}`, block {_block_text(e['block'])}, {e['shared']} bytes of dynamic shared memory.")
    if binary.scalars:
        lines.append("")
        plain = all(s["type"] == "i32" and "stride_of" not in s for s in binary.scalars)
        lines.append(
            "Runtime scalars (`i32`) of the binary, derived by `launch` from the shapes:"
            if plain
            else "Runtime scalars of the binary, derived by `launch` from the tensors (they are not parameters of `launch`):"
        )
        for s in binary.scalars:
            if "count_of" in s:
                how = f"the element count of `{binary.arg_names[s['count_of']]}`"
            elif "stride_of" in s:
                how = _stride_text(s["stride_of"], binary.arg_names)
            else:
                how = f"`{s['dim']}`"
            lines.append(f"- `{s['name']}` is {how}" + ("" if s["type"] == "i32" else f" ({s['type']})"))
    return lines


def _stride_text(so: Any, arg_names: "dict[str, str]") -> str:
    """What a stride scalar carries, for the docs: the stride of one view, or the one stride the views that share it must all have."""
    items = so if isinstance(so, list) else [so]
    if len(items) == 1:
        return f"the stride of axis {items[0]['axis']} of `{arg_names[items[0]['arg']]}`, in elements"
    names = [f"`{arg_names[i['arg']]}`" for i in items]
    listed = ", ".join(names[:-1]) + f" and {names[-1]}"
    if len({i["axis"] for i in items}) == 1:
        return f"the stride of axis {items[0]['axis']} of {listed}, in elements (one stride, which each of them must have)"
    each = ", ".join(f"axis {i['axis']} of {n}" for i, n in zip(items, names))
    return f"the stride of {each}, in elements (one stride, which each of them must have)"


def _doc_comment(lines: "list[str]", indent: str = "") -> str:
    """Doc comment lines. A line break inside a line (which no validated field has) starts another comment line, never code."""
    out = []
    for line in lines:
        for ln in "".join(c if c.isprintable() or c in "\n\r" else f"\\x{ord(c):02x}" for c in line).splitlines() or [""]:
            out.append(f"{indent}///{' ' + ln if ln else ''}\n")
    return "".join(out)


def _render_binary(b: _Binary) -> str:
    e = b.entry
    elements = {p["name"]: RUST_ELEMENTS[p["dtype"]] for p in b.params}
    tensor_params = []
    for p in b.params:
        wrapper = "TensorRef" if p["access"] == "in" else "TensorMut"
        tensor_params.append(f"{b.arg_names[p['name']]}: {wrapper}<'a, {elements[p['name']]}>")
    args = ", ".join(f"{b.arg_names[p['name']]}.into_arg()" for p in b.params)
    out = [_doc_comment(_doc(b)), "#[derive(Debug, Clone)]\n", f"pub struct {b.struct} {{\n    kernel: Kernel,\n}}\n\n", f"impl {b.struct} {{\n"]
    out.append("    /// The stem of the binary in the manifest.\n")
    out.append(f"    pub const STEM: &'static str = {_rs_str(b.stem)};\n")
    buckets = ", ".join(f"({_rs_str(d)}, {v})" for d, v in e["buckets"].items())
    out.append("    /// The bucket values the binary is built for (none when it is not bucketed).\n")
    out.append(f"    pub const BUCKETS: &'static [(&'static str, u64)] = &[{buckets}];\n")
    rows = ", ".join("&[" + ", ".join(_rs_str(str(d)) for d in p["shape"]) + "]" for p in b.params)
    out.append("    /// The declared shape of each tensor, in the order of `launch`, as dimension expressions (a number is a size baked in).\n")
    out.append(f"    pub const SHAPES: &'static [&'static [&'static str]] = &[{rows}];\n")
    seen = set()
    for p in b.params:
        for axis, d in enumerate(p["shape"]):
            if not isinstance(d, int):
                continue
            name = f"{b.arg_names[p['name']].upper().rstrip('_')}_AXIS{axis}"
            if name in seen:
                raise TilefishManifestError(f"'{b.stem}': two baked-in sizes would both be called {name}")
            seen.add(name)
            out.append(f"    /// Axis {axis} of `{b.arg_names[p['name']]}`, baked into the binary.\n")
            out.append(f"    pub const {name}: usize = {d};\n")
    out.append("\n")
    out.append("    /// Load the binary for the device of `ctx` from `manifest`, in that context (which the launcher then keeps alive).\n")
    out.append("    ///\n")
    out.append("    /// Refuses a manifest whose digest is not [`MANIFEST_DIGEST`], and one without a binary for the device.\n")
    out.append("    pub fn load(manifest: &Manifest, ctx: &Arc<CudaContext>) -> Result<Self, Error> {\n")
    out.append("        manifest.require_digest(MANIFEST_DIGEST)?;\n")
    out.append("        let kernel = manifest.kernel(ctx, Self::STEM, Self::BUCKETS, Arch::of_context(ctx)?)?;\n")
    out.append("        Ok(Self { kernel })\n")
    out.append("    }\n\n")
    out.append("    /// The loaded kernel.\n")
    out.append("    pub fn kernel(&self) -> &Kernel {\n        &self.kernel\n    }\n\n")
    doc = [
        "Check the call and launch the kernel on `stream`, a stream of the context the kernel was loaded in (checked). A tensor made of",
        "a cudarc slice is given to cudarc on `stream`; the slice's own stream waits for the kernel, so freeing the slice runs after it,",
        "and a kernel that writes it first waits for everything enqueued on the slice's own stream. A tensor made of a region the host",
        "vouched for is tracked by nothing. A launch of slices on a stream that is capturing a CUDA graph is an error: capture with",
        "`launch_raw` and regions. An `Err` from the driver does not promise that nothing was launched (see `tilefish::Kernel::launch`).",
        "",
        "Every precondition of the binary is checked first (dtype, rank, static sizes, the dimensions against each other, the",
        "bucket, `divisible_by`, 16-byte alignment, `max_elements`, aliasing, the grid against the device); nothing is launched",
        "when one fails. The dimensions, the runtime scalars and the grid come from the shapes of the tensors.",
    ]
    out.append(_doc_comment(doc, "    "))
    out.append("    #[allow(clippy::too_many_arguments, non_snake_case)]\n")
    out.append("    pub fn launch<'a>(\n        &self,\n        stream: &CudaStream,\n")
    for p in tensor_params:
        out.append(f"        {p},\n")
    out.append("    ) -> Result<(), Error> {\n")
    out.append(f"        self.kernel.launch(stream, &mut [{args}])\n")
    out.append("    }\n\n")
    raw_doc = [
        "Check the call and launch the kernel on a raw `CUstream`: another library's stream, or one a CUDA graph is being captured on.",
        "",
        "# Safety",
        "",
        "The kernel's context is made current for the call and what was current is put back. `stream` must be a valid `CUstream` of",
        "the CUDA context the kernel was loaded in, and the memory of the tensors must stay allocated, and be ordered against the",
        "other streams that use it, until the launch has completed on `stream` (see `tilefish::Kernel::launch_raw`).",
    ]
    out.append(_doc_comment(raw_doc, "    "))
    out.append("    #[allow(clippy::too_many_arguments, non_snake_case)]\n")
    out.append("    pub unsafe fn launch_raw<'a>(\n        &self,\n        stream: CUstream,\n")
    for p in tensor_params:
        out.append(f"        {p},\n")
    out.append("    ) -> Result<(), Error> {\n")
    out.append("        // SAFETY: the caller vouches for the stream, as this function's contract says.\n")
    out.append(f"        unsafe {{ self.kernel.launch_raw(stream, &[{args}]) }}\n")
    out.append("    }\n}\n")
    return "".join(out)


def _cublaslt_text(version: int) -> str:
    """``120804`` as ``12.8.4 (120804)``: cuBLASLt's ``major * 10000 + minor * 100 + patch``."""
    return f"{version // 10000}.{version // 100 % 100}.{version % 100} ({version})"


def _doc_gemm(g: _Gemm) -> "list[str]":
    e = g.entry
    m, n, k, bias = e["m"], e["n"], e["k"], LIBRARY_IMPLS[e["impl"]]
    lines: "list[str]" = []
    if g.signature:
        lines += [f"`{g.signature}`", ""]
    lines += _wrap(f"The pinned cuBLASLt GEMM `{g.stem}` of operator `{e['op']}` (`{e['impl']}`), for {', '.join(g.archs)}.")
    a, b = (g.arg_names[p["name"]] for p in g.params[:2])
    out = g.arg_names[g.params[-1]["name"]]
    lines.append("")
    lines += _wrap(
        f"It computes `{out}[{m}, {n}] = {a}[{m}, {k}] @ {b}[{n}, {k}]^T"
        + (f" + {g.arg_names[g.params[2]['name']]}[{n}]" if bias else "")
        + "` on bf16 operands with an f32 accumulator, and rounds to bf16 once"
        + (" (the bias is added in f32 before the rounding)" if bias else "")
        + ". Every operand is contiguous and 16-byte aligned."
    )
    lines.append("")
    lines.append("Tensors, in the order of `launch`:")
    for p in g.params:
        lines.append(f"- `{g.arg_names[p['name']]}`: {p['access']}, {p['dtype']}, `[{_shape_text(p['shape'])}]`")
    lines.append(f"- `{g.workspace}`: u8, rank 1, at least `workspace_required()` bytes (`WORKSPACE_BYTES` is the size the algorithm was pinned under;")
    lines.append("  `Gemm::alloc_workspace` allocates one)")
    constraints = [
        f"`{name}` is exactly {c['bucket']} (a bucket of {c['buckets']}; there is no padding, and the other buckets are other structs)"
        for name, c in e["dims"].items()
        if "bucket" in c
    ]
    if constraints:
        lines.append("")
        lines.append("`launch` binds the dimensions from the sizes of the tensors and refuses a call where they disagree.")
        lines += _wrap("Constraints: " + "; ".join(constraints) + ".")
    lines.append("")
    lines += _wrap(
        "The algorithm is the one cuBLASLt's heuristic ranked first when the manifest was made, pinned in the manifest and run as it is; "
        "a cuBLASLt of another version than the one that chose it, and a device of another number of SMs, are refused. Pinned with: "
        + ", ".join(f"{x['arch']}: cuBLASLt {_cublaslt_text(x['cublaslt_version'])} on {x['sm_count']} SMs" for x in g.entries)
        + "."
    )
    return lines


def _render_gemm(g: _Gemm) -> str:
    e = g.entry
    bias = LIBRARY_IMPLS[e["impl"]]
    tensor_params = [f"{g.arg_names[p['name']]}: {'TensorMut' if p['access'] == 'out' else 'TensorRef'}<'a, Bf16>" for p in g.params]
    tensor_params.append(f"{g.workspace}: TensorMut<'a, U8>")
    names = [g.arg_names[p["name"]] for p in g.params] + [g.workspace]
    args = ", ".join(f"{n}.into_arg()" for n in names)
    out = [_doc_comment(_doc_gemm(g)), "#[derive(Debug, Clone)]\n", f"pub struct {g.struct} {{\n    gemm: Gemm,\n}}\n\n", f"impl {g.struct} {{\n"]
    out.append("    /// The stem of the library entry in the manifest.\n")
    out.append(f"    pub const STEM: &'static str = {_rs_str(g.stem)};\n")
    buckets = ", ".join(f"({_rs_str(d)}, {v})" for d, v in e["buckets"].items())
    out.append("    /// The bucket values the entry is built for (none when it is not bucketed).\n")
    out.append(f"    pub const BUCKETS: &'static [(&'static str, u64)] = &[{buckets}];\n")
    rows = ", ".join("&[" + ", ".join(_rs_str(str(d)) for d in p["shape"]) + "]" for p in g.params)
    out.append("    /// The declared shape of each tensor (not the workspace), in the order of `launch`, as dimension expressions (a number is a size baked in).\n")
    out.append(f"    pub const SHAPES: &'static [&'static [&'static str]] = &[{rows}];\n")
    out.append("    /// Rows of `a` and of the result.\n")
    out.append(f"    pub const M: usize = {e['m']};\n")
    out.append("    /// Rows of `b`, and columns of the result.\n")
    out.append(f"    pub const N: usize = {e['n']};\n")
    out.append("    /// The columns of `a` and of `b`.\n")
    out.append(f"    pub const K: usize = {e['k']};\n")
    out.append("    /// Whether the GEMM adds a bias.\n")
    out.append(f"    pub const BIAS: bool = {'true' if bias else 'false'};\n")
    out.append("    /// The workspace the algorithm was pinned under: the size of the workspace to give `launch`, in bytes.\n")
    out.append(f"    pub const WORKSPACE_BYTES: usize = {e['workspace_bytes']};\n")
    pins = ", ".join(f"({_rs_str(x['arch'])}, {x['cublaslt_version']})" for x in g.entries)
    out.append("    /// The version of cuBLASLt (`cublasLtGetVersion`) that chose the algorithm, for each architecture the manifest has it for.\n")
    out.append(f"    pub const PINNED_WITH: &'static [(&'static str, u64)] = &[{pins}];\n")
    out.append("\n")
    out.append("    /// Load the GEMM for the device of `ctx` from `manifest`, in that context (which the launcher then keeps alive).\n")
    out.append("    ///\n")
    out.append("    /// Refuses a manifest whose digest is not [`MANIFEST_DIGEST`], one without an entry for the device, a cuBLASLt of another\n")
    out.append("    /// version than the one that chose the algorithm (see `PINNED_WITH`), and a device of another number of SMs than it was pinned on.\n")
    out.append("    pub fn load(manifest: &Manifest, ctx: &Arc<CudaContext>) -> Result<Self, Error> {\n")
    out.append("        manifest.require_digest(MANIFEST_DIGEST)?;\n")
    out.append("        let gemm = manifest.gemm(ctx, Self::STEM, Self::BUCKETS, Arch::of_context(ctx)?)?;\n")
    out.append("        Ok(Self { gemm })\n")
    out.append("    }\n\n")
    out.append("    /// The loaded GEMM.\n")
    out.append("    pub fn gemm(&self) -> &Gemm {\n        &self.gemm\n    }\n\n")
    doc = [
        "Check the call and run the GEMM on `stream`, a stream of the context it was loaded in (checked). The tensors are the operands,",
        "the result and the workspace, which must hold at least `workspace_required()` bytes (see `tilefish::Gemm::launch`). Every",
        "precondition is checked first (dtype, rank, the sizes, 16-byte alignment, contiguity, aliasing); nothing is launched when one fails.",
    ]
    out.append(_doc_comment(doc, "    "))
    out.append("    #[allow(clippy::too_many_arguments, non_snake_case)]\n")
    out.append("    pub fn launch<'a>(\n        &self,\n        stream: &CudaStream,\n")
    for p in tensor_params:
        out.append(f"        {p},\n")
    out.append("    ) -> Result<(), Error> {\n")
    out.append(f"        self.gemm.launch(stream, &mut [{args}])\n")
    out.append("    }\n\n")
    raw_doc = [
        "Check the call and run the GEMM on a raw `CUstream`: another library's stream, or one a CUDA graph is being captured on.",
        "",
        "# Safety",
        "",
        "`stream` must be a valid `CUstream` of the CUDA context the GEMM was loaded in, and the memory of the tensors must stay allocated, and",
        "be ordered against the other streams that use it, until the launch has completed on `stream` (see `tilefish::Gemm::launch_raw`).",
    ]
    out.append(_doc_comment(raw_doc, "    "))
    out.append("    #[allow(clippy::too_many_arguments, non_snake_case)]\n")
    out.append("    pub unsafe fn launch_raw<'a>(\n        &self,\n        stream: CUstream,\n")
    for p in tensor_params:
        out.append(f"        {p},\n")
    out.append("    ) -> Result<(), Error> {\n")
    out.append("        // SAFETY: the caller vouches for the stream, as this function's contract says.\n")
    out.append(f"        unsafe {{ self.gemm.launch_raw(stream, &[{args}]) }}\n")
    out.append("    }\n}\n")
    return "".join(out)


def _trusted(manifest: Any, pin: "str | None") -> Manifest:
    """The manifest to generate from: a directory with ``pin``, or a :class:`Manifest` that is pinned (see the module docstring)."""
    if isinstance(manifest, (str, os.PathLike)):
        if pin is None:
            raise TilefishManifestError(
                f"the manifest directory {str(manifest)!r} would be read without a trust anchor, and the Rust source generated from it is "
                "compiled and run: pass pin=<the manifest's digest> (Manifest.digest, which runtime.export returns), or a Manifest "
                "that is pinned"
            )
        return Manifest.load(manifest, pin=pin)
    if not isinstance(manifest, Manifest):
        raise TilefishManifestError(f"emit takes a manifest directory (with pin=) or a pinned Manifest, got {type(manifest).__name__}")
    if pin is not None:
        return manifest.check_pin(pin)
    if not manifest.pinned:
        raise TilefishManifestError(
            f"the manifest in {str(manifest.directory)!r} is not pinned, and the Rust source generated from it is compiled and run: "
            "load it with Manifest.load(path, pin=<its digest>), pass pin= here, or use the Manifest that runtime.export returned"
        )
    return manifest


def render(
    manifest: Any,
    *,
    ops: "Iterable[Any]" = (),
    embed: bool = False,
    relative_to: "str | os.PathLike[str] | None" = None,
    pin: "str | None" = None,
) -> str:
    """The Rust source :func:`emit` writes, as a string. ``relative_to`` is the directory of the generated file (for ``embed``).

    ``manifest`` is a directory with ``pin``, or a pinned :class:`~tilefish.Manifest` (see the module docstring); every field of it
    that reaches the output is validated first.
    """
    manifest = _trusted(manifest, pin)
    binaries, gemms = _group(manifest)
    _check_declarations(binaries, gemms, ops)
    elements = {RUST_ELEMENTS[p["dtype"]] for b in binaries for p in b.params}
    uses = {"Arch", "CUstream", "CudaContext", "CudaStream", "Error", "Manifest"}
    if binaries or not gemms:
        uses.add("Kernel")
    if gemms:
        uses |= {"Gemm", "Bf16", "U8", "TensorRef", "TensorMut"}
    if embed:
        uses.add("Pin")
    if any(p["access"] == "in" for b in binaries for p in b.params):
        uses.add("TensorRef")
    if any(p["access"] != "in" for b in binaries for p in b.params):
        uses.add("TensorMut")
    uses = sorted(uses | elements)
    out = [
        "// @generated by tilefish.hosts.rust. Do not edit: generate it again from the manifest.\n",
        f"// The manifest it was generated from has digest {manifest.digest}.\n",
        "\n",
        "use std::sync::Arc;\n",
        "\n",
        f"use tilefish::{{{', '.join(uses)}}};\n",
        "\n",
        "/// The digest of the manifest this module was generated from: `load` refuses any other manifest.\n",
        f"pub const MANIFEST_DIGEST: &str = {_rs_str(manifest.digest)};\n",
    ]
    if embed:
        base = pathlib.Path(relative_to if relative_to is not None else ".")
        rel = pathlib.PurePosixPath(os.path.relpath(manifest.directory, base).replace(os.sep, "/"))
        files = sorted({a.entry["cubin"] for a in manifest.artifacts})
        out.append("\n/// `manifest.json`, embedded.\n")
        out.append(f"pub static MANIFEST_JSON: &[u8] = include_bytes!({_rs_str(str(rel / 'manifest.json'))});\n")
        out.append("\n/// The cubins of the manifest, embedded, by file name.\n")
        out.append("pub static CUBINS: &[(&str, &[u8])] = &[\n")
        for name in files:
            out.append(f"    ({_rs_str(name)}, include_bytes!({_rs_str(str(rel / name))})),\n")
        out.append("];\n")
        out.append("\n/// Read the embedded manifest: verified like a directory, and pinned to [`MANIFEST_DIGEST`].\n")
        out.append("pub fn load_embedded() -> Result<Manifest, Error> {\n")
        out.append("    Manifest::from_embedded(MANIFEST_JSON, CUBINS, Pin::Digest(MANIFEST_DIGEST))\n}\n")
    for binary in binaries:
        out.append("\n")
        out.append(_render_binary(binary))
    for gemm in gemms:
        out.append("\n")
        out.append(_render_gemm(gemm))
    return "".join(out)


def emit(
    manifest: Any,
    path: "str | os.PathLike[str]",
    *,
    ops: "Iterable[Any]" = (),
    embed: bool = False,
    pin: "str | None" = None,
) -> pathlib.Path:
    """Write the typed Rust launchers of ``manifest`` to ``path`` (one ``.rs`` module) and return the path.

    ``manifest`` is a manifest directory, which is loaded only with ``pin=`` (its digest: a directory with any other digest is
    refused), or a :class:`~tilefish.Manifest` that is pinned (loaded with a pin, or returned by ``tilefish.runtime.export``). The
    source is compiled and run by whoever uses it, so a manifest nobody vouched for does not get to write it, and every field that
    reaches it is validated: a name, a dtype, a shape, the grid, a file name that is not what it should be is a
    :class:`~tilefish.errors.TilefishManifestError`, never text in the generated file.

    The launchers are typed from the manifest alone (it has the declared shape of every pointer and the constraints of every
    dimension): Triton kernels, prebuilt PTX and cubins (with strided views, device scalars, 64-bit scalars and fixed arguments)
    and pinned cuBLASLt GEMMs. ``ops`` are optional: the declarations the manifest was built from (kernels, externs, lowered
    kernels or library ops). Each one that has a binary or a library entry in the manifest must be exactly what the manifest says
    (:class:`~tilefish.errors.TilefishManifestError` otherwise), and gives the docs the tilefish signature. With
    ``embed=True`` the manifest file and the cubins are embedded with ``include_bytes!``, by paths relative to ``path``. The
    file is written through a temporary file and renamed, and a symbolic link at ``path`` is refused.
    """
    path = pathlib.Path(path)
    source = render(manifest, ops=list(ops), embed=embed, relative_to=path.resolve().parent, pin=pin)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_file(path, source.encode("utf-8"))
    return path


# -- launch plans ----------------------------------------------------------------------------------------------------------------------

# what a generated plan runner imports from the crate and std, and `Self`: a struct of these names would clash with them
_RESERVED_PLAN_STRUCTS = frozenset(
    {"Self", "Arc", "Path", "CapturedPlan", "CudaStream", "DevScalar", "Error", "LaunchPlan", "Pin", "PlanRun", "Readable", "TensorArg", "TensorMut"}
    | {"Result", "Option", "Some", "None", "Ok", "Err", "Vec"}
    | set(RUST_ELEMENTS.values())
)

# the host type a device scalar's value is given as, by dtype (a dtype that is none of these is given as `DevScalar`, its bits)
_DEVSCALAR_TYPES = {"i32": "i32", "i64": "i64", "u8": "u8", "f32": "f32"}


def _trusted_plan(plan: Any, pin: "str | None") -> "dict[str, Any]":
    """The document of the plan to generate from: a plan directory (or its ``plan.json``) with ``pin``, or a :class:`~tilefish._plan.Plan` that is
    pinned (loaded with a pin, or returned by ``tilefish.runtime.export``). Verified as a replay verifies it: strict JSON, the digest against the pin
    and against the one it carries, every rule of the format, and the manifest next to it (pinned to the plan's ``manifest_digest``) against every
    step."""
    from .. import _plan

    if isinstance(plan, (str, os.PathLike)):
        if pin is None:
            raise TilefishManifestError(
                f"the plan {str(plan)!r} would be read without a trust anchor, and the Rust source generated from it is compiled and run: pass "
                "pin=<the plan's digest> (Plan.digest, which runtime.export returns), or a Plan that is pinned"
            )
        doc = _plan.read_plan(plan, pin=pin)
        path = pathlib.Path(plan)
        directory = path if path.is_dir() else path.parent
        _plan.check_against_manifest(doc, Manifest.load(directory, pin=doc["manifest_digest"]))
        return doc
    if not isinstance(plan, _plan.Plan):
        raise TilefishManifestError(f"emit_plan takes a plan directory (with pin=) or a pinned Plan, got {type(plan).__name__}")
    if pin is not None and plan.digest != pin:
        raise TilefishManifestError(f"the plan of graph '{plan.name}' has digest {plan.digest}, but the pin is {pin}")
    if pin is None and not getattr(getattr(plan, "manifest", None), "pinned", False):
        raise TilefishManifestError(
            f"the plan of graph '{plan.name}' is not pinned, and the Rust source generated from it is compiled and run: load it with a pin, pass "
            "pin= here, or use the Plan that runtime.export returned"
        )
    _plan.validate_doc(plan.doc)  # what reaches the output: validated again (a plan made only in memory has no architecture and is refused here)
    return plan.doc


def render_plan(plan: Any, *, pin: "str | None" = None) -> str:
    """The Rust source :func:`emit_plan` writes, as a string. ``plan`` is a plan directory with ``pin``, or a pinned
    :class:`~tilefish._plan.Plan`; every field of it that reaches the output is validated first."""
    doc = _trusted_plan(plan, pin)
    who = f"the plan of graph {doc.get('graph')!r}"
    graph = _name(who, "graph", doc["graph"])
    digest = _name(who, "plan_digest", doc["plan_digest"], _DIGEST)
    manifest_digest = _name(who, "manifest_digest", doc["manifest_digest"], _DIGEST)
    arch = _name(who, "arch", doc["arch"], _ARCH)
    struct = _camel(graph)
    struct = struct + "Plan" if struct in _RESERVED_PLAN_STRUCTS or struct + "Externals" in _RESERVED_PLAN_STRUCTS else struct
    externals = [b for b in doc["buffers"] if b["role"] in ("input", "weight", "inout", "output")]
    scalars = [b for b in doc["buffers"] if b["role"] == "devscalar"]
    fields: "dict[str, str]" = {}
    for b in externals + scalars:
        name = _snake_ident(_name(who, "a buffer name", b["name"]))
        if name in fields.values():
            _bad(who, f"two buffers have the field name {name!r} in Rust")
        if b["dtype"] not in RUST_ELEMENTS:
            _bad(who, f"buffer {b['name']!r} has dtype {b['dtype']!r}, which has no Rust element type")
        _shape(who, f"the shape of buffer {b['name']!r}", b["shape"])
        fields[b["name"]] = name
    dims = []
    for d, values in doc["dims"].items():
        _name(who, "a dimension", d)
        dims.append(f"({_rs_str(d)}, &[{', '.join(str(_count(who, f'a bucket of {d}', v, 1)) for v in values)}])")
    roles = {"input": "an input (read)", "weight": "a weight (read)", "inout": "updated in place", "output": "a returned value (written)"}
    outputs = list(doc["outputs"])
    uses = {"CapturedPlan", "CudaStream", "DevScalar", "Error", "LaunchPlan", "Pin", "PlanRun", "TensorArg"}
    uses |= {RUST_ELEMENTS[b["dtype"]] for b in externals}
    if any(b["role"] in ("input", "weight") for b in externals):
        uses.add("Readable")
    if any(b["role"] in ("inout", "output") for b in externals):
        uses.add("TensorMut")
    out = [
        "// @generated by tilefish.hosts.rust.emit_plan. Do not edit: generate it again from the plan.\n",
        f"// The plan of graph `{graph}` it was generated from has digest {digest}; its manifest has digest {manifest_digest}.\n",
        "\n",
        "use std::path::Path;\n",
        "use std::sync::Arc;\n",
        "\n",
        f"use tilefish::{{{', '.join(sorted(uses))}}};\n",
        "\n",
        "/// The digest of the plan this module was generated from: `load` refuses any other plan (and so any other manifest).\n",
        f"pub const PLAN_DIGEST: &str = {_rs_str(digest)};\n",
        "\n",
    ]
    lines = [f"The tensors of a call of the plan of graph `{graph}`, by name, with their element types (the caller's: the plan allocates none)."]
    lines.append("An input or a weight is a `Readable` (`x.into()` from a `TensorRef`, or from a `TensorMut` to write new contents through a")
    lines.append("`CapturedPlan` between replays); what the plan writes is a `TensorMut`. Every one is checked against the plan at a call.")
    out.append(_doc_comment(lines))
    out.append("#[allow(non_snake_case)]\n")
    out.append(f"pub struct {struct}Externals<'a> {{\n")
    for b in externals:
        what = roles[b["role"]] + (f", returned value {outputs.index(b['name'])}" if b["role"] == "output" else "")
        out.append(_doc_comment([f"`{b['name']}`: {what}, {b['dtype']} `[{_shape_text(b['shape'])}]`."], "    "))
        element = RUST_ELEMENTS[b["dtype"]]
        kind = "TensorMut" if b["role"] in ("inout", "output") else "Readable"
        out.append(f"    pub {fields[b['name']]}: {kind}<'a, {element}>,\n")
    out.append("}\n\n")
    out.append(_doc_comment([f"The device scalars of a call of the plan of graph `{graph}`: written into their slots of the arena before the first launch."]))
    out.append("#[derive(Debug, Clone, Copy)]\n")
    if scalars:
        out.append("#[allow(non_snake_case)]\n")
        out.append(f"pub struct {struct}Scalars {{\n")
        for b in scalars:
            ty = _DEVSCALAR_TYPES.get(b["dtype"], "DevScalar")
            note = "" if ty != "DevScalar" else " (its bits: `DevScalar::Bits`)"
            out.append(_doc_comment([f"`{b['name']}`: DevScalar[{b['dtype']}]{note}."], "    "))
            out.append(f"    pub {fields[b['name']]}: {ty},\n")
        out.append("}\n\n")
    else:
        out.append(f"pub struct {struct}Scalars;\n\n")
    doc_lines = [
        f"The launch plan of graph `{graph}` for {arch}, instantiated: every bucket's binaries loaded and its arena allocated, once.",
        "",
        f"The dimensions and their buckets: {', '.join(f'{d} {list(v)}' for d, v in doc['dims'].items()) or 'none (one bucket)'}. A call selects the bucket",
        "from the shapes of its tensors, which must be one exactly (there is no padding).",
    ]
    out.append(_doc_comment(doc_lines))
    out.append("#[derive(Debug, Clone)]\n")
    out.append(f"pub struct {struct} {{\n    run: PlanRun,\n}}\n\n")
    out.append(f"impl {struct} {{\n")
    out.append("    /// The name of the graph.\n")
    out.append(f"    pub const GRAPH: &'static str = {_rs_str(graph)};\n")
    out.append("    /// The architecture the plan is for.\n")
    out.append(f"    pub const ARCH: &'static str = {_rs_str(arch)};\n")
    out.append("    /// The dimensions of the plan and the sizes planned for each.\n")
    out.append(f"    pub const DIMS: &'static [(&'static str, &'static [u64])] = &[{', '.join(dims)}];\n")
    out.append("    /// The bytes of the arena.\n")
    out.append(f"    pub const ARENA_BYTES: u64 = {_count(who, 'arena_bytes', doc['arena_bytes'])};\n\n")
    out.append("    /// Load the plan directory (its `plan.json`, pinned to [`PLAN_DIGEST`], and the manifest next to it) and instantiate it for every\n")
    out.append("    /// bucket on `stream`, which the arena is allocated on.\n")
    out.append("    pub fn load(directory: impl AsRef<Path>, stream: &Arc<CudaStream>) -> Result<Self, Error> {\n")
    out.append("        let plan = LaunchPlan::load(directory, Pin::Digest(PLAN_DIGEST))?;\n")
    out.append("        Ok(Self { run: plan.instantiate_all(stream)? })\n")
    out.append("    }\n\n")
    out.append("    /// The instantiated plan (untyped).\n")
    out.append("    pub fn run(&self) -> &PlanRun {\n        &self.run\n    }\n\n")
    out.append(f"    fn tensors(externals: {struct}Externals<'_>) -> Vec<(&'static str, TensorArg<'_>)> {{\n")
    out.append("        vec![\n")
    for b in externals:
        out.append(f"            ({_rs_str(b['name'])}, externals.{fields[b['name']]}.into_arg()),\n")
    out.append("        ]\n    }\n\n")
    out.append(f"    fn values(scalars: &{struct}Scalars) -> Vec<(&'static str, DevScalar)> {{\n")
    if scalars:
        out.append("        vec![\n")
        for b in scalars:
            value = f"scalars.{fields[b['name']]}"
            out.append(f"            ({_rs_str(b['name'])}, {value if _DEVSCALAR_TYPES.get(b['dtype']) is None else value + '.into()'}),\n")
        out.append("        ]\n    }\n\n")
    else:
        out.append("        let _ = scalars;\n        Vec::new()\n    }\n\n")
    out.append("    /// Run the plan on `stream` (see `tilefish::PlanRun::launch`): every step is checked before the first launch, and nothing is\n")
    out.append("    /// allocated.\n")
    out.append(f"    pub fn launch(&self, stream: &Arc<CudaStream>, externals: {struct}Externals<'_>, scalars: &{struct}Scalars) -> Result<(), Error> {{\n")
    out.append("        self.run.launch(stream, &mut Self::tensors(externals), &Self::values(scalars))\n")
    out.append("    }\n\n")
    out.append("    /// Capture the plan in a CUDA graph on `stream` for the bucket the tensors select (see `tilefish::PlanRun::capture`): the graph\n")
    out.append("    /// borrows the tensors for as long as it lives, and `scalars` are what its first replay reads.\n")
    out.append(
        f"    pub fn capture<'a>(&self, stream: &Arc<CudaStream>, externals: {struct}Externals<'a>, scalars: &{struct}Scalars) -> Result<CapturedPlan<'a>, Error> {{\n"
    )
    out.append("        self.run.capture(stream, Self::tensors(externals), &Self::values(scalars))\n")
    out.append("    }\n}\n")
    return "".join(out)


def emit_plan(plan: Any, path: "str | os.PathLike[str]", *, pin: "str | None" = None) -> pathlib.Path:
    """Write a typed Rust runner of a launch plan (``plan.json``) to ``path`` (one ``.rs`` module) and return the path.

    The module has ``PLAN_DIGEST``; a struct named after the graph (``block`` gives ``Block``) whose ``load(directory, &stream)`` loads the plan
    pinned to that digest (and with it the manifest) and instantiates every bucket, and whose ``launch`` and ``capture`` take a
    ``<Graph>Externals`` (one field per external, named after it, with its element type: a ``Readable`` for an input or a weight, a ``TensorMut``
    for what the plan writes) and a ``<Graph>Scalars`` (one field per device scalar, in its host type). Everything else is the crate's
    ``PlanRun``. ``plan`` is a plan directory with ``pin=`` (the plan's digest), or a pinned :class:`~tilefish._plan.Plan` (loaded with a pin, or
    returned by ``tilefish.runtime.export``): the source is compiled and run by whoever uses it, so a plan nobody vouched for does not get to write
    it, and every field that reaches it is validated. The file is written through a temporary file and renamed.
    """
    path = pathlib.Path(path)
    source = render_plan(plan, pin=pin)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_file(path, source.encode("utf-8"))
    return path
