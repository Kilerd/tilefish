"""Reading a manifest directory back: the cubins that ``replay`` runs, checked against what the manifest says.

``write_manifest(built, path, cubins=True)`` writes ``manifest.json`` and one ``<stem>.<arch>.cubin`` per
kernel. :meth:`Manifest.load` reads them back and refuses anything that is not what was written: a missing or
malformed file, a cubin whose ``sha256`` (the file) or ``sha256_code`` (the machine code) differs from the
manifest's, a file name that points out of the directory, or an entry that two kernels share.
:meth:`Manifest.built_kernel` then hands a replay the binary for one compile spec, after checking that the
entry is what that spec asks for (its ABI, grid, block, constants, preconditions, options and declaration
digest), so a replay can never run a binary under assumptions it was not compiled for.

Trust. A manifest directory vouches for nothing by itself: its digests only say that the files are the ones the
manifest describes, and an attacker who can write the directory can rewrite both. :attr:`Manifest.digest` is a sha256
over the whole manifest (every entry: spec digests, source digests, file and code digests, symbol, shared memory,
block, launch fields, toolchain); a caller that holds it from a trusted place (a build log, a lock file) passes it as
``pin=`` and a manifest with any other content is refused. ``tilefish.run(..., mode="replay")`` does not run a directory
without that anchor: it takes a pin, or ``trust_directory=True``, the explicit statement that whatever the directory holds is
trusted.

Source. The identity of a binary is the Triton IR its kernel compiles to (:mod:`tilefish._ir`). Every entry records the digest
of that IR in ``source_digest``, with ``source_digest_kind`` ``"tilefish.triton-ir.v1"`` (the same fields a schema 2 reader
already reads: the field has the name of the earlier file digest and holds an IR digest). The compile spec of a generated kernel
also carries ``source_sha256``, the sha256 of the Triton source it was built from, and a replay refuses an entry built from
another source (a kernel redeclared with another body but the same name and signature) without needing Triton. A hand-written
``extern`` has no source in its declaration, so its IR digest is all there is: a replay recomputes it from the importable kernel as
the process holds it, with the same Triton version that built the entry, and refuses a mismatch; where it cannot (Triton or the module
is not importable, or the Triton is another version, whose IR may print differently), it refuses unless the caller pinned the
manifest, which says that the caller vouches for it.

The file is read as strict JSON: no ``NaN`` or ``Infinity``, no number that does not fit a float, no object with the same
key twice (which parsers disagree about), and every field a replay reads has the type it needs, so a malformed
manifest is a :class:`~tilefish.errors.TilefishManifestError` and never a bare ``KeyError`` or ``AttributeError``.

What an entry says of itself is read as the Rust reader reads it (``crates/tilefish/src/manifest.rs``), so that the two readers refuse
the same documents, pinned or not (a re-pinned digest vouches for a document, not for its sense): the architecture is ``sm_`` and two or
three digits and is the one the cubin's ELF header names (:func:`tilefish._elf.sm_of`; a prebuilt binary's header must name one); the stem
is the operator and the value of each of its buckets (``rowop.S32``); the ``declaration_digest`` is a sha256; every parameter of the ABI has
the fields of its kind and no other (a ``scalar`` has no ``dtype``, ``access``, ``shape``, ``strides``, ``offset`` or ``value``; a ``pointer``
has no ``dim``, ``count_of``, ``stride_of`` or ``value``, takes ``strides`` and ``offset`` together and only when it is ``in``; a ``fixed``
argument belongs to a prebuilt binary); and the dimension expressions (shapes, strides, offsets, scalars, preconditions, grid) are in the
grammar, mention only names of ``dims``, and at most :data:`MAX_DIMS` names in all. The rest of what an entry promises (its ABI as a whole,
its grid, its preconditions) is compared with the operator's compile spec by :meth:`Manifest.built_kernel`.

Nothing here needs Triton, torch or a GPU: it is the part of a host that is just files and digests.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from ._build import (
    BINARY_DIGEST_KIND,
    ENTRY_KIND_BINARY,
    ENTRY_KIND_TRITON,
    MANIFEST_SCHEMA,
    BuiltKernel,
    LaunchAttributes,
    spec_entry,
)
from ._ir import IR_DIGEST_KIND
from ._elf import code_digest, parse_elf, sm_of, uses_shared
from ._specvalid import parse_expression, parse_grid
from .dtypes import ALL_DTYPES
from .errors import TilefishBuildError, TilefishManifestError, TilefishSpecError

if TYPE_CHECKING:
    from .aot import CompileSpec

MANIFEST_NAME = "manifest.json"
DIGEST_TAG = b"tilefish.manifest-digest.v3\0"  # part of what is hashed: another canonical form must change it
LEGACY_DIGEST_TAG = b"tilefish.manifest-digest.v2\0"  # schema 2, the one the Rust host (crate 0.0.1) computes
MAX_DIMS = 32  # the most dimension names one kernel entry may use: the Rust reader's `expr::MAX_DIMS` (crates/tilefish/src/expr.rs)


def manifest_digest(doc: "dict[str, Any]") -> str:
    """The sha256 (lowercase hex) of a manifest document: its schema, every kernel entry and every library entry in full.

    Entries are sorted by ``(stem, arch)`` (kernels and library entries each) and serialized as canonical JSON (sorted keys, no
    whitespace), so the digest does not depend on the order of the entries or on how the file was formatted, and changes with any field
    of any entry: the spec-derived fields (ABI, grid, block, constants, preconditions, options, declaration digest,
    source digests), the build facts (symbol, shared memory, launch attributes, toolchain, architecture), the
    digests of the cubin file and of its machine code, and of a library entry the pinned algorithm and the cuBLASLt version.
    The top-level ``tilefish`` version is not part of it.

    Schema 3 is ``sha256(DIGEST_TAG + json({"schema": 3, "kernels": [...], "library": [...]}))`` with the tag ``v3``. The function also
    digests any other document in the form of schema 2 (``{"schema": <whatever it says>, "kernels": [...]}``, the tag ``v2``), which is
    what the Rust host 0.0.1 computes for the manifests it reads (and for the edge cases of its test vectors); this Python reads and
    writes schema 3 only.
    """
    try:
        key = lambda e: (str(e.get("stem")), str(e.get("arch")))  # noqa: E731
        kernels = sorted(doc["kernels"], key=key)
        schema = doc["schema"]
        if schema == MANIFEST_SCHEMA:
            body = {"schema": schema, "kernels": kernels, "library": sorted(doc["library"], key=key)}
            tag = DIGEST_TAG
        else:  # the function of schema 2 and before, which digests whatever ``schema`` says (a reader refuses a schema it does not read; the digest of a document does not)
            body = {"schema": schema, "kernels": kernels}
            tag = LEGACY_DIGEST_TAG
        text = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, AttributeError, KeyError, RecursionError) as e:  # a number that is not finite, a value JSON cannot hold, a malformed entry, a document nested too deeply
        raise TilefishManifestError(f"the manifest cannot be digested, it is not plain JSON data: {type(e).__name__}: {e}") from e
    return hashlib.sha256(tag + text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, eq=False)
class Artifact:
    """One binary of a manifest: its entry and its cubin, whose digests have been checked."""

    entry: "dict[str, Any]"
    cubin: bytes

    @property
    def stem(self) -> str:
        return self.entry["stem"]

    @property
    def arch(self) -> str:
        return self.entry["arch"]

    @property
    def sha256_code(self) -> str:
        return self.entry["sha256_code"]


def _reject_constant(constant: str) -> Any:
    raise TilefishManifestError(f"the manifest is not strict JSON: it contains the constant {constant}")


def _parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise TilefishManifestError(f"the manifest is not strict JSON: the number {text} does not fit a float")
    return value


def _no_duplicate_keys(pairs: "list[tuple[str, Any]]") -> "dict[str, Any]":
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise TilefishManifestError(f"the manifest is not strict JSON: an object has the key {key!r} twice")
        out[key] = value
    return out


def _check_pin(directory: pathlib.Path, doc: "dict[str, Any]", pin: Any) -> None:
    if not (isinstance(pin, str) and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin)):
        raise TilefishManifestError(f"pin= is the 64 lowercase hex digits of a manifest digest, got {pin!r}")
    found = manifest_digest(doc)
    if found != pin:
        raise TilefishManifestError(
            f"the manifest in {str(directory)!r} has digest {found}, but the pin is {pin}: it is not the manifest that was "
            "pinned (it was changed, or it is another export)"
        )


# -- what an entry says of itself, read as the Rust reader reads it (crates/tilefish/src/manifest.rs) ------------------------------

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*", re.ASCII)
_ARCH = re.compile(r"sm_[0-9]{2,3}", re.ASCII)  # `is_arch`
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)  # `is_digest`
_U64_MAX = 2**64 - 1
# every field a parameter of the ABI may have (`RawParam`), the ones it must have, and the width each scalar type is passed with
_PARAM_FIELDS = frozenset("slot index name kind type ptx_type size dtype access shape dim count_of strides offset stride_of value alias".split())
_PARAM_REQUIRED = ("slot", "index", "name", "kind", "type", "ptx_type", "size")
_PASSED_AS = {"scalar": {"i32": ("u32", 4), "i64": ("u64", 8)}, "fixed": {"i32": ("u32", 4), "i64": ("u64", 8), "f32": ("f32", 4)}}


def _is_ident(x: Any) -> bool:
    return isinstance(x, str) and _IDENT.fullmatch(x) is not None


def _is_count(x: Any, minimum: int) -> bool:
    """An int (not a bool) from ``minimum`` to ``2**64 - 1``: what the Rust reader reads as a ``u64`` (or a ``usize``)."""
    return type(x) is int and minimum <= x <= _U64_MAX


def _has(p: "dict[str, Any]", key: str) -> bool:
    """Whether a parameter has the field ``key``: a JSON ``null`` is as absent, as the Rust reader reads an optional field."""
    return p.get(key) is not None


def _check_identity(who: str, entry: "dict[str, Any]") -> None:
    """The architecture, the stem and the declaration digest of an entry (the Rust reader's ``Entry::check_raw``).

    The stem of a binary is its operator and the value of each of its buckets, in the order of ``buckets`` (``rowop.S32``): a stem that
    names another operator or other buckets is refused, whatever the entry's digest. The ``declaration_digest`` is a sha256 (a JSON
    ``null`` is not one).
    """

    def bad(message: str) -> NoReturn:
        raise TilefishManifestError(f"{who}: {message}")

    arch = entry["arch"]
    if not (isinstance(arch, str) and _ARCH.fullmatch(arch)):
        bad(f"arch must look like 'sm_103', got {arch!r}")
    op, buckets = entry["op"], entry["buckets"]
    if not _is_ident(op):
        bad(f"op must be an identifier, got {op!r}")
    if not isinstance(buckets, dict):
        bad(f"buckets must be an object (the value of each bucketed dimension), got {buckets!r}")
    stem = op
    for dim, value in buckets.items():
        if not _is_ident(dim):
            bad(f"bucket dimension name must be an identifier, got {dim!r}")
        if not _is_count(value, 1):
            bad(f"bucket value of {dim} must be an int >= 1, got {value!r}")
        stem += f".{dim}{value}"
    if stem != entry["stem"]:
        bad(
            f"stem {entry['stem']!r} does not match op {op!r} and buckets {buckets!r}: the stem of a binary is its operator and the value of "
            f"each of its buckets, {stem!r}"
        )
    digest = entry["declaration_digest"]
    if not (isinstance(digest, str) and _SHA256.fullmatch(digest)):
        bad(f"declaration_digest must be 64 lowercase hex digits (a sha256), got {digest!r}")


def _check_params(who: str, entry: "dict[str, Any]", kind: str) -> None:
    """Every parameter of the ABI on its own: its kind, its type and width, and which fields it has (the Rust reader's ``read_params``).

    A ``pointer`` has a ``dtype`` whose Triton pointer type is its ``type``, an ``access`` (``in``, ``inout``, ``out``) and a ``shape`` (sizes
    of at least 1 and dimension expressions), is passed as a ``u64`` of 8 bytes, and has no ``dim``, ``count_of``, ``stride_of`` or ``value``;
    it takes ``strides`` (one per axis: a stride dimension, or the fixed stride 1 on the last axis) and ``offset`` (the name of its offset
    dimension) together, and only when it is ``in``. A ``scalar`` is an ``i32`` or an ``i64`` with its ``dim``, a ``count_of`` or a
    ``stride_of`` (an object ``{arg, axis}``, or a list of two or more of them: the views that share the stride), and nothing of a pointer. A ``fixed`` argument belongs to a prebuilt binary and has a ``value`` of its type, and nothing
    else. How the parameters fit together (strides and the scalars that carry them, preconditions) is compared with the compile spec.
    """

    def bad(message: str) -> NoReturn:
        raise TilefishManifestError(f"{who}: {message}")

    abi = entry["abi"]
    params = abi.get("params")
    if not isinstance(params, list):
        bad(f"abi.params must be a list, got {params!r}")
    for position, p in enumerate(params):
        if not isinstance(p, dict):
            bad(f"abi.params[{position}] must be an object (its fields by name), got {p!r}")
        unknown = sorted(str(k) for k in set(p) - _PARAM_FIELDS)
        if unknown:
            bad(f"abi.params[{position}] ({p.get('name')!r}) has unknown fields {unknown}")
        missing = [k for k in _PARAM_REQUIRED if p.get(k) is None]
        if missing:
            bad(f"abi.params[{position}] ({p.get('name')!r}) is missing {missing}")
        name = p["name"]
        if not _is_ident(name):
            bad(f"parameter name must be an identifier, got {name!r}")
        for key in ("slot", "index", "size"):
            if not _is_count(p[key], 0):
                bad(f"parameter {name}: {key} must be an int >= 0, got {p[key]!r}")
        typed = (("kind", str), ("type", str), ("ptx_type", str), ("dtype", str), ("access", str), ("shape", list), ("dim", str))
        for key, kind_of_value in (*typed, ("count_of", str), ("strides", list), ("alias", str)):
            if _has(p, key) and not isinstance(p[key], kind_of_value):
                bad(f"parameter {name}: {key} must be a {kind_of_value.__name__}, got {p[key]!r}")
        if _has(p, "stride_of") and not isinstance(p["stride_of"], (dict, list)):
            bad(f"parameter {name}: stride_of must be an object or a list of them, got {p['stride_of']!r}")
        if p["slot"] != position:
            bad(f"parameter {name} has slot {p['slot']} but is at position {position}")
        pkind, ty = p["kind"], p["type"]
        if pkind == "pointer":
            dtype = p.get("dtype") or ""
            if ALL_DTYPES.get(dtype) is None or ALL_DTYPES[dtype].triton_ptr != ty:
                bad(f"parameter {name}: pointer type {ty!r} does not match dtype {dtype!r}")
            access = p.get("access")
            if access not in ("in", "inout", "out"):
                bad(f"parameter {name}: access must be in, inout or out, got {access!r}")
            if any(_has(p, k) for k in ("dim", "count_of", "stride_of", "value")) or p["ptx_type"] != "u64" or p["size"] != 8:
                bad(
                    f"parameter {name} is not a well-formed 64-bit pointer: a pointer is passed as a u64 of 8 bytes and has a dtype, an access "
                    "and a shape, and no dim, count_of, stride_of or value"
                )
            shape = p.get("shape")
            if shape is None:
                bad(f"parameter {name} has no shape: a pointer records its declared shape")
            for axis, d in enumerate(shape):
                if not (_is_count(d, 1) or isinstance(d, str)):
                    bad(f"parameter {name}: shape[{axis}] must be an int size of at least 1 or a dimension expression, got {d!r}")
            strides, offset = p.get("strides"), p.get("offset")
            if (strides is None) != (offset is None):
                bad(f"parameter {name}: strides and offset come together")
            if strides is not None:
                if not shape or len(strides) != len(shape):
                    bad(
                        f"parameter {name}: strides must be a list with one entry per axis of shape (at least one axis), got {len(strides)} "
                        f"for {len(shape)} axes"
                    )
                if access != "in":
                    bad(f"parameter {name} is {access} and takes strides, but only an `in` pointer may be strided")
                for axis, s in enumerate(strides):
                    if type(s) is int:
                        if s != 1:
                            bad(f"the stride of axis {axis} of '{name}' is the fixed size {s}: a fixed stride is 1 (and only on the last axis)")
                        if axis + 1 != len(strides):
                            bad(f"the stride of axis {axis} of '{name}' is fixed, but only the last axis may be")
                    elif not isinstance(s, str):
                        bad(f"parameter {name}: strides[{axis}] must be an int or a dimension expression, got {s!r}")
                if not _is_ident(offset):
                    bad(f"the offset of '{name}' (the name of the offset dimension) must be an identifier, got {offset!r}")
        elif pkind == "scalar":
            if ty not in _PASSED_AS["scalar"]:
                bad(f"parameter {name}: a runtime scalar must be i32 or i64, got {ty!r}")
            ptx, size = _PASSED_AS["scalar"][ty]
            if p["ptx_type"] != ptx or p["size"] != size:
                bad(f"parameter {name}: an {ty} scalar is passed as a {ptx} of {size} bytes, got {p['ptx_type']} of {p['size']}")
            extra = [k for k in ("dtype", "access", "shape", "strides", "offset", "value", "alias") if _has(p, k)]
            if extra:
                bad(f"parameter {name} is a scalar: it has a dim and no dtype, access, shape, strides or value (it has {extra})")
            if _has(p, "count_of") and _has(p, "stride_of"):
                bad(f"parameter {name}: a scalar is the element count of a pointer or the stride of one, not both")
            if not _has(p, "dim"):
                bad(f"parameter {name}: dim is missing")
            if _has(p, "stride_of"):
                so = p["stride_of"]
                items = so if isinstance(so, list) else [so]
                ok = (not isinstance(so, list) or len(so) >= 2) and all(
                    isinstance(i, dict) and set(i) == {"arg", "axis"} and _is_ident(i["arg"]) and _is_count(i["axis"], 0) for i in items
                )
                if not ok or len({(i["arg"], i["axis"]) for i in items}) != len(items):
                    bad(
                        f"parameter {name}: stride_of must be {{'arg': <an identifier>, 'axis': <an int >= 0>}}, or a list of two or more of them "
                        f"(the views that share the stride, each once), got {so!r}"
                    )
        elif pkind == "fixed":
            if kind != ENTRY_KIND_BINARY:
                bad(f"parameter {name}: a fixed argument belongs to a prebuilt binary, and this is a {kind} kernel")
            if ty not in _PASSED_AS["fixed"]:
                bad(f"parameter {name}: a fixed argument is one of i32, i64, f32, got {ty!r}")
            ptx, size = _PASSED_AS["fixed"][ty]
            if p["ptx_type"] != ptx or p["size"] != size:
                bad(f"parameter {name}: a fixed {ty} is passed as a {ptx} of {size} bytes, got {p['ptx_type']} of {p['size']}")
            if any(_has(p, k) for k in ("dim", "count_of", "stride_of", "access", "shape", "strides", "offset", "dtype", "alias")):
                bad(f"parameter {name} is a fixed argument: it has a value and no dim or shape")
            value = p.get("value")
            if value is None:
                bad(f"parameter {name} is a fixed argument and has no value")
            if ty == "f32":
                if isinstance(value, dict) and len(value) == 1:
                    if value.get("float") not in ("inf", "-inf", "nan"):
                        bad(f"parameter {name}: {value!r} is not a valid tagged float")
                elif not isinstance(value, float):
                    bad(f"parameter {name}: the value of a fixed f32 argument must be a float, got {value!r}")
            elif type(value) is not int:
                bad(f"parameter {name}: the value of a fixed {ty} argument must be an int, got {value!r}")
            elif not -(2**63) <= value < 2**63 or (ty == "i32" and not -(2**31) <= value < 2**31):
                bad(f"parameter {name}: the value {value} does not fit {ty}")
        else:
            bad(f"parameter {name}: kind must be pointer, scalar or fixed, got {pkind!r}")
    # an out pointer that may be written over an in pointer (`alias`, the Rust reader's `check_consistent`): that pointer is an `in` one, not strided and
    # not a device scalar, of the same dtype and declared shape, and no other out pointer names it
    from ._specvalid import check_alias_target

    by_name = {p["name"]: {k: v for k, v in p.items() if v is not None} for p in params}
    taken: "dict[str, str]" = {}
    for p in params:
        if not _has(p, "alias"):
            continue
        try:
            check_alias_target(p["name"], p["alias"], p.get("access"), by_name)
        except TilefishSpecError as e:
            bad(str(e).removeprefix("malformed compile spec: "))
        if p["alias"] in taken:
            bad(f"the out pointers {taken[p['alias']]} and {p['name']} both may alias '{p['alias']}': at most one is written over an input")
        taken[p["alias"]] = p["name"]


def _check_dimension_names(who: str, entry: "dict[str, Any]") -> None:
    """The dimension expressions of an entry are in the grammar, and name at most :data:`MAX_DIMS` dimensions, all of them in ``dims``.

    The expressions are the ones the Rust reader parses into the one table of names of the entry (``Symbols``, after the names of
    ``dims``): the ``dim`` of a ``divisible`` precondition and the ``shape`` of a ``max_elements`` one, every axis of a declared shape that is
    an expression, every stride dimension and offset dimension of a strided pointer, the ``dim`` of every scalar, and the grid. A table that
    would hold more than 32 names is refused ("a kernel may use at most 32 dimension names": a strided pointer of rank ``r`` alone takes up
    to ``r + 1``), and so is a name that ``dims`` does not have. Each expression is parsed by :mod:`tilefish._specvalid` (the Rust parser is a port
    of it, with its limits: 512 characters, 256 tokens, a depth of 24, 12 digits a number), which refuses what the grammar does not have and a
    static grid entry or divisor below 1.
    """

    def bad(message: str) -> NoReturn:
        raise TilefishManifestError(f"{who}: {message}")

    dims, preconditions = entry["dims"], entry["preconditions"]
    if not isinstance(dims, dict):
        bad(f"dims must be an object (the constraints of each named dimension), got {dims!r}")
    if not isinstance(preconditions, list):
        bad(f"preconditions must be a list, got {preconditions!r}")
    for name in dims:
        if not _is_ident(name):
            bad(f"dims name must be an identifier, got {name!r}")
    used: "set[str]" = set()

    def expression(text: Any, where: str, grid: bool = False) -> None:
        if not isinstance(text, str):
            bad(f"{where} must be a string (a dimension expression), got {text!r}")
        try:
            used.update(parse_grid(text) if grid else parse_expression(text, where))
        except TilefishSpecError as e:
            bad(str(e).removeprefix("malformed compile spec: "))

    for i, pre in enumerate(preconditions):
        if not isinstance(pre, dict):
            bad(f"a precondition must be an object, got {pre!r}")
        if pre.get("kind") == "divisible":
            expression(pre.get("dim"), f"preconditions[{i}] (divisible) dim")
        elif pre.get("kind") == "max_elements":
            shape = pre.get("shape")
            if not isinstance(shape, list):
                bad(f"preconditions[{i}] (max_elements) shape must be a list of dimension strings, got {shape!r}")
            for axis, d in enumerate(shape):
                expression(d, f"preconditions[{i}] (max_elements) shape[{axis}]")
    for p in entry["abi"]["params"]:  # (each read by `_check_params`)
        if p["kind"] == "pointer":
            for axis, d in enumerate(p["shape"]):
                if isinstance(d, str):
                    expression(d, f"parameter {p['name']} shape[{axis}]")
            if _has(p, "strides"):
                for axis, s in enumerate(p["strides"]):
                    if isinstance(s, str):
                        expression(s, f"parameter {p['name']} strides[{axis}]")
                used.add(p["offset"])
        elif p["kind"] == "scalar":
            expression(p["dim"], f"parameter {p['name']} dim")
    expression(entry["grid"], "grid", grid=True)
    names = set(dims) | used
    if len(names) > MAX_DIMS:
        bad(
            f"a kernel may use at most {MAX_DIMS} dimension names, and this one has {len(names)} (in dims and in its shapes, strides, "
            "offsets, scalars, preconditions and grid): the Rust host binds at most that many"
        )
    unnamed = sorted(used - set(dims))
    if unnamed:
        bad(f"dims does not name {unnamed}, which the declared shapes, the strides, the scalars, the preconditions or the grid mention")


class Manifest:
    """The binaries of a manifest directory, verified when it is loaded.

    ``pinned`` says whether its digest was compared with one from a trusted place: ``Manifest.load(path, pin=...)`` and
    :meth:`check_pin` do, and so does :func:`tilefish.runtime.export` for the manifest it has just written from its own builds.
    What generates source code from a manifest (:mod:`tilefish.hosts.rust`) takes only a pinned one.
    """

    def __init__(
        self,
        directory: pathlib.Path,
        doc: "dict[str, Any]",
        artifacts: "dict[tuple[str, str], Artifact]",
        path: "pathlib.Path | None" = None,
        pinned: bool = False,
        library: "dict[tuple[str, str], dict[str, Any]] | None" = None,
    ) -> None:
        self.directory = directory
        self.path = path if path is not None else directory / MANIFEST_NAME
        self.doc = doc
        self._artifacts = artifacts
        self._library = library if library is not None else {}
        self.pinned = pinned  # whether its digest was compared with one from a trusted place (see :meth:`check_pin`)

    @property
    def digest(self) -> str:
        """The digest of the whole manifest (see :func:`manifest_digest`): what ``run(..., pin=...)`` compares."""
        return manifest_digest(self.doc)

    @classmethod
    def load(cls, path: "str | pathlib.Path", *, pin: "str | None" = None) -> "Manifest":
        """Load ``<directory>/manifest.json`` (or the manifest file itself) and every cubin it names.

        Raises :class:`~tilefish.errors.TilefishManifestError` when a file is missing or malformed, a cubin is not
        an ELF file with the code section the manifest's symbol names, or its ``sha256`` or ``sha256_code`` is not
        the one the manifest records. With ``pin=`` (the digest of the manifest, from a trusted place) a manifest with any
        other digest is refused, before a cubin is read, and the result is :attr:`pinned`.
        """
        path = pathlib.Path(path)
        manifest_path = path / MANIFEST_NAME if path.is_dir() else path
        try:
            text = manifest_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise TilefishManifestError(f"cannot read the manifest {str(manifest_path)!r}: {e}") from e
        try:
            doc = json.loads(text, parse_constant=_reject_constant, parse_float=_parse_float, object_pairs_hook=_no_duplicate_keys)
        except RecursionError as e:
            raise TilefishManifestError(f"the manifest {str(manifest_path)!r} is nested too deeply to be read") from e
        except TilefishManifestError:
            raise
        except ValueError as e:
            raise TilefishManifestError(f"the manifest {str(manifest_path)!r} is not valid JSON: {e}") from e
        older = doc.get("schema") if isinstance(doc, dict) else None
        if isinstance(older, int) and not isinstance(older, bool) and 0 < older < MANIFEST_SCHEMA:
            raise TilefishManifestError(
                f"the manifest {str(manifest_path)!r} is schema {older}, which an older tilefish wrote; this tilefish reads schema "
                f"{MANIFEST_SCHEMA}, which adds strided views, device scalars, prebuilt binaries and library operations (and records the "
                "number of scratch slots of every ABI) and has another digest. Re-export it (tilefish.runtime.export) and pin the new digest"
            )
        # the schema is the integer 3: not 3.0 (which == 3 in Python, and which a strict host refuses), not "3", not True
        if (
            not isinstance(doc, dict)
            or type(doc.get("schema")) is not int
            or doc["schema"] != MANIFEST_SCHEMA
            or not isinstance(doc.get("kernels"), list)
            or not isinstance(doc.get("library"), list)
        ):
            raise TilefishManifestError(
                f"the manifest {str(manifest_path)!r} is not a tilefish manifest of schema {MANIFEST_SCHEMA} with a 'kernels' list and a "
                f"'library' list (schema {doc.get('schema') if isinstance(doc, dict) else None!r})"
            )
        if pin is not None:
            _check_pin(manifest_path.parent, doc, pin)
        artifacts: dict[tuple[str, str], Artifact] = {}
        for i, entry in enumerate(doc["kernels"]):
            artifact = cls._read(manifest_path.parent, i, entry)
            key = (artifact.stem, artifact.arch)
            if key in artifacts:
                raise TilefishManifestError(f"the manifest lists '{key[0]}' for {key[1]} twice")
            artifacts[key] = artifact
        from ._library import validate_entry

        library: dict[tuple[str, str], dict[str, Any]] = {}
        for i, lib in enumerate(doc["library"]):
            validate_entry(lib, i)
            key = (lib["stem"], lib["arch"])
            if key in library or key in artifacts:
                raise TilefishManifestError(f"the manifest lists '{key[0]}' for {key[1]} twice")
            library[key] = lib
        return cls(manifest_path.parent, doc, artifacts, manifest_path, pinned=pin is not None, library=library)

    def library_entry(self, stem: str, arch: str) -> "dict[str, Any] | None":
        """The library entry (a pinned library operation, such as a cuBLASLt GEMM) of ``stem`` for ``arch``, or ``None``."""
        return self._library.get((stem, arch))

    @property
    def library_entries(self) -> "list[dict[str, Any]]":
        return list(self._library.values())

    def check_pin(self, pin: str) -> "Manifest":
        """Refuse unless the digest of this manifest is ``pin``; then it is :attr:`pinned`. Returns the manifest."""
        _check_pin(self.directory, self.doc, pin)
        self.pinned = True
        return self

    @staticmethod
    def _read(directory: pathlib.Path, i: int, entry: Any) -> Artifact:
        if not isinstance(entry, dict):
            raise TilefishManifestError(f"manifest kernel {i} is not an object")
        entry_kind = entry.get("kind")
        if entry_kind not in (ENTRY_KIND_TRITON, ENTRY_KIND_BINARY):
            raise TilefishManifestError(
                f"manifest kernel {i} ({entry.get('stem')!r}) has kind {entry_kind!r}, which is not '{ENTRY_KIND_TRITON}' or '{ENTRY_KIND_BINARY}'"
            )
        needed = ("stem", "arch", "symbol", "cubin", "sha256", "sha256_code", "shared", "launch", "toolchain")
        needed += ("options",) if entry_kind == ENTRY_KIND_TRITON else ("binary",)
        # (what the entry says of itself, as the Rust reader reads it; the rest the spec determines, block, warps, ..., is checked against the spec)
        needed += ("op", "abi", "grid", "buckets", "dims", "preconditions", "declaration_digest")
        missing = [k for k in needed if k not in entry]
        if missing:
            raise TilefishManifestError(f"manifest kernel {i} ({entry.get('stem')!r}) is missing {missing}")
        for field in ("options", "toolchain", "launch", "abi"):
            if field in entry and not isinstance(entry[field], dict):
                raise TilefishManifestError(f"'{entry['stem']}' ({entry['arch']}): '{field}' must be an object, got {entry[field]!r}")
        who = f"'{entry['stem']}' ({entry['arch']})"
        _check_identity(who, entry)
        _check_params(who, entry, entry_kind)
        if entry_kind == ENTRY_KIND_BINARY:
            extra = [k for k in ("options", "num_warps", "num_stages") if k in entry]
            if extra:
                raise TilefishManifestError(
                    f"manifest kernel {i} ({entry.get('stem')!r}) is a prebuilt binary, which has no {extra} (Triton compile settings)"
                )
        stem, arch, name = entry["stem"], entry["arch"], entry["cubin"]
        if not isinstance(name, str) or not name or pathlib.PurePath(name).name != name or name in (".", ".."):
            raise TilefishManifestError(f"'{stem}': the cubin file name {name!r} must be a plain file name in the manifest directory")
        try:
            cubin = (directory / name).read_bytes()
        except OSError as e:
            raise TilefishManifestError(f"'{stem}' ({arch}): cannot read the cubin {name!r}: {e}") from e
        found = hashlib.sha256(cubin).hexdigest()
        if found != entry["sha256"]:
            raise TilefishManifestError(
                f"'{stem}' ({arch}): the file {name!r} has sha256 {found}, but the manifest says {entry['sha256']}; "
                "the cubin is not the one that was built"
            )
        try:
            elf = parse_elf(cubin)
            code = code_digest(cubin)
        except TilefishBuildError as e:
            raise TilefishManifestError(f"'{stem}' ({arch}): {name!r} is not a valid cubin: {e}") from e
        if code != entry["sha256_code"]:
            raise TilefishManifestError(
                f"'{stem}' ({arch}): the machine code of {name!r} has sha256_code {code}, but the manifest says "
                f"{entry['sha256_code']}; the binary was changed after it was built"
            )
        if elf.section(f".text.{entry['symbol']}") is None:
            raise TilefishManifestError(f"'{stem}' ({arch}): {name!r} has no code section '.text.{entry['symbol']}'")
        # the architecture the cubin is for is in its ELF header (as the Rust reader's `verify_cubin` reads it): a cubin is used as it is, for
        # its own architecture, so one whose header names another is refused; a prebuilt binary's header must name one (a Triton cubin whose
        # header names none is left to the driver)
        sm = sm_of(elf)
        if sm is not None and sm != int(arch[3:]):  # (the number: "sm_090" is sm_90, as the Rust reader parses it)
            raise TilefishManifestError(
                f"'{stem}' ({arch}): {name!r} is for sm_{sm} (its ELF header says so), but the manifest says {arch}: a cubin is used as it "
                "is, for the architecture it was built for"
            )
        if sm is None and entry_kind == ENTRY_KIND_BINARY:
            raise TilefishManifestError(
                f"'{stem}' ({arch}): the ELF header of {name!r} is not that of a CUDA cubin of a known architecture: its architecture cannot be read"
            )
        _check_dimension_names(f"'{stem}' ({arch})", entry)
        if "scratch_slots" in entry["abi"]:
            slots = entry["abi"]["scratch_slots"]
            if type(slots) is not int or slots != (2 if entry_kind == ENTRY_KIND_TRITON else 0):
                raise TilefishManifestError(
                    f"'{stem}' ({arch}): a {entry_kind} kernel has {2 if entry_kind == ENTRY_KIND_TRITON else 0} scratch slots, "
                    f"but abi.scratch_slots is {slots!r}"
                )
        if entry_kind == ENTRY_KIND_BINARY:
            b = entry["binary"]
            ok = isinstance(b, dict) and b.get("source") in ("ptx", "cubin")
            if ok and b["source"] == "ptx":
                flags = b.get("ptxas_flags")
                ok = set(b) == {"source", "ptxas_flags"} and isinstance(flags, list) and all(isinstance(f, str) for f in flags)
            elif ok:
                ok = set(b) == {"source"}
            if not ok:
                raise TilefishManifestError(
                    f"'{stem}' ({arch}): binary must be {{'source': 'cubin'}} or {{'source': 'ptx', 'ptxas_flags': [...]}}, got {b!r}"
                )
            if entry.get("source_digest_kind") != BINARY_DIGEST_KIND or entry.get("source_digest") is None:
                raise TilefishManifestError(
                    f"'{stem}' ({arch}): a prebuilt binary records the sha256 of its PTX or cubin as source_digest, "
                    f"with source_digest_kind '{BINARY_DIGEST_KIND}'"
                )
            if entry.get("source_sha256") is not None:
                raise TilefishManifestError(f"'{stem}' ({arch}): a prebuilt binary has no source_sha256 (that is the source of a generated Triton kernel)")
            if b["source"] == "cubin" and entry["source_digest"] != entry["sha256"]:  # as the Rust reader (manifest.rs): the file is the source
                raise TilefishManifestError(
                    f"'{stem}' ({arch}): a cubin used as it is is its own source, so its source_digest is the sha256 of the cubin file "
                    f"({entry['sha256']}), not {entry['source_digest']!r}"
                )
        # the digests of the source, as the Rust reader reads them (crates/tilefish/src/manifest.rs): a field that is there is a string
        # of 64 lowercase hex digits (a JSON null is as absent), and a source digest and its kind come together
        for field in ("source_sha256", "source_digest"):
            value = entry.get(field)
            if value is not None and not (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)):
                raise TilefishManifestError(f"'{stem}' ({arch}): {field} must be 64 lowercase hex digits (a sha256), got {value!r}")
        kind = entry.get("source_digest_kind")
        if kind is not None and not isinstance(kind, str):
            raise TilefishManifestError(f"'{stem}' ({arch}): source_digest_kind must be a string, got {kind!r}")
        if (entry.get("source_digest") is None) != (kind is None):
            raise TilefishManifestError(f"'{stem}' ({arch}): source_digest and source_digest_kind come together")
        shared = entry["shared"]
        if isinstance(shared, bool) or not isinstance(shared, int) or shared < 0:
            raise TilefishManifestError(f"'{stem}' ({arch}): shared must be a non-negative integer, got {shared!r}")
        # (Triton gives a kernel that needs shared memory dynamic shared memory, so the section means the manifest must say how much. A
        # prebuilt binary may have static shared memory, which has the same section, so its declaration is what says how much it addresses.)
        if entry_kind == ENTRY_KIND_TRITON and shared == 0 and uses_shared(elf, entry["symbol"]):
            raise TilefishManifestError(
                f"'{stem}' ({arch}): the kernel has a shared memory section ('.nv.shared.{entry['symbol']}'), so it uses "
                "shared memory, but the manifest says shared = 0; a launch without the dynamic shared memory it addresses "
                "would fault and poison the CUDA context"
            )
        return Artifact(entry, cubin)

    @property
    def artifacts(self) -> "list[Artifact]":
        return list(self._artifacts.values())

    def __contains__(self, key: "tuple[str, str]") -> bool:
        return key in self._artifacts

    def get(self, stem: str, arch: str) -> "Artifact | None":
        """The artifact of ``stem`` for ``arch`` (``"sm_103"``), or ``None``."""
        return self._artifacts.get((stem, arch))

    def built_kernel(
        self,
        spec: "CompileSpec",
        arch: str,
        *,
        ir_digest: "str | None" = None,
        ir_triton: "str | None" = None,
        pinned: bool = False,
    ) -> BuiltKernel:
        """The binary for ``spec`` on ``arch``, as a :class:`~tilefish.aot.BuiltKernel` a replay can launch.

        The manifest entry must be what the spec asks for: every field the spec determines (ABI, grid, block,
        ``num_warps``, constants, buckets, preconditions, compile options, declaration digest and, for a generated
        kernel, the ``source_sha256``) must equal the spec's. The entry's ``source_digest`` (an IR digest, see
        :mod:`tilefish._ir`) must equal ``ir_digest``, the digest the caller recomputed from the kernel in its process, made with
        Triton version ``ir_triton`` (which must be the version the entry was built with, or the two are not comparable). A
        recomputed digest that differs from the entry's is a refusal, pinned or not. For a hand-written kernel (a spec without a
        ``source_sha256``), when the digests cannot be compared (the caller could not recompute one, the entry has none, or another
        Triton made it) the binary is only accepted with ``pinned=True``: the caller vouches for the manifest. For a generated
        kernel the ``source_sha256`` is what is checked then. The result has no PTX (a manifest does not carry it).
        """
        artifact = self._artifacts.get((spec.stem, arch))
        if artifact is None:
            have = sorted(a for (s, a) in self._artifacts if s == spec.stem)
            raise TilefishManifestError(
                f"the manifest in {str(self.directory)!r} has no binary for '{spec.stem}' on {arch}"
                + (f" (it has {', '.join(have)})" if have else f" (it has {sorted({s for s, _ in self._artifacts})})")
            )
        entry = artifact.entry
        from .aot import _first_difference, _show

        if (spec.binary is None) != (entry.get("kind") == ENTRY_KIND_TRITON):
            raise TilefishManifestError(
                f"the manifest entry for '{spec.stem}' ({arch}) is a {entry.get('kind')} kernel, but the compile spec is for a "
                f"{'prebuilt binary' if spec.binary is not None else 'Triton kernel'}"
            )
        if spec.binary is not None:
            self._check_binary_source(spec, arch, entry)
        else:
            self._check_source(spec, arch, entry, ir_digest, ir_triton, pinned)

        expected = json.loads(json.dumps(spec_entry(spec)))  # as JSON read it back: lists, strings, plain numbers
        got = {k: entry.get(k) for k in expected}
        found = _first_difference(expected, got, "")
        if found is not None:
            path, want, have = found
            raise TilefishManifestError(
                f"the manifest entry for '{spec.stem}' ({arch}) is not what the compile spec asks for: first difference "
                f"at {path}: the manifest has {_show(have)}, the spec gives {_show(want)}"
            )
        toolchain = entry["toolchain"]
        try:
            launch = LaunchAttributes.from_dict(entry["launch"])
        except (KeyError, TypeError, ValueError) as e:
            raise TilefishManifestError(f"'{spec.stem}' ({arch}): the launch attributes are malformed: {e}") from e
        try:
            return BuiltKernel(
                spec=spec,
                arch=arch,
                cubin=artifact.cubin,
                ptx="",
                symbol=entry["symbol"],
                shared=int(entry["shared"]),
                num_warps=int(entry.get("num_warps", spec.num_warps)),
                sha256=entry["sha256"],
                sha256_code=entry["sha256_code"],
                triton_version=str(toolchain.get("triton", "")),
                ptxas_version=str(toolchain.get("ptxas", "")),
                ptx_target=str(entry.get("ptx_target", "")),
                options=dict(entry.get("options", {})),
                launch=launch,
                ir_digest=entry.get("source_digest") if entry.get("source_digest_kind") == IR_DIGEST_KIND else None,
                binary_digest=entry.get("source_digest") if entry.get("source_digest_kind") == BINARY_DIGEST_KIND else None,
                binary_build=dict(entry["binary"]) if spec.binary is not None else None,
            )
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise TilefishManifestError(f"'{spec.stem}' ({arch}): the manifest entry is malformed: {type(e).__name__}: {e}") from e

    @staticmethod
    def _check_binary_source(spec: "CompileSpec", arch: str, entry: "dict[str, Any]") -> None:
        """A prebuilt binary is identified by the sha256 of its PTX or cubin (kind ``tilefish.binary.v1``): the one the declaration holds.

        The declaration reads the file once, when the operator is declared, so what the process holds is what is compared: an entry
        built from other bytes is stale, pinned or not (nothing needs Triton or a GPU to tell).
        """
        assert spec.binary is not None
        where = f"'{spec.stem}' ({arch})"
        key = "ptx" if spec.binary["kind"] == "ptx" else arch
        have = spec.binary["digests"].get(key)
        if have is None:
            raise TilefishManifestError(
                f"the declaration of {where} has no {spec.binary['kind']} for {arch} (it has {sorted(spec.binary['digests'])}), so "
                "the manifest entry cannot be checked against it"
            )
        if entry.get("source_digest") != have or entry.get("source_digest_kind") != BINARY_DIGEST_KIND:
            raise TilefishManifestError(
                f"the manifest entry for {where} was built from a {spec.binary['kind']} with sha256 {entry.get('source_digest')}, but the "
                f"declaration now holds one with sha256 {have}: the file changed since the manifest was written, so its binary is stale; "
                "export it again"
            )
        if entry["binary"].get("source") != spec.binary["kind"]:
            raise TilefishManifestError(
                f"the manifest entry for {where} was made from a {entry['binary'].get('source')}, but the declaration says {spec.binary['kind']}"
            )

    @staticmethod
    def _check_source(
        spec: "CompileSpec", arch: str, entry: "dict[str, Any]", ir_digest: "str | None", ir_triton: "str | None", pinned: bool
    ) -> None:
        where = f"'{spec.stem}' ({arch})"
        if spec.source_sha256 is not None:  # a generated kernel: its source is part of its spec
            have = entry.get("source_sha256")
            if have != spec.source_sha256:
                raise TilefishManifestError(
                    f"the manifest entry for {where} was built from kernel source with sha256 {have}, but the kernel now "
                    f"generates source with sha256 {spec.source_sha256}: the kernel's body (or an option of its lowering) changed "
                    "since the manifest was written, so its binary is stale; export it again"
                )
        recorded = entry.get("source_digest")
        kind = entry.get("source_digest_kind")
        toolchain = entry.get("toolchain")
        built_with = toolchain.get("triton") if isinstance(toolchain, dict) else None
        if recorded is not None and kind != IR_DIGEST_KIND:
            recorded = None  # a digest of another definition (an older tilefish) says nothing about the IR now
            unusable = f"the manifest entry's source digest is of kind {kind!r}, not {IR_DIGEST_KIND!r}"
        else:
            unusable = "the manifest entry records no IR digest"
        comparable = ir_digest is not None and recorded is not None and ir_triton == built_with
        if comparable and ir_digest != recorded:
            raise TilefishManifestError(
                f"the manifest entry for {where} was built from Triton IR with digest {recorded}, but the kernel '{spec.triton}' in this "
                f"process now compiles to IR with digest {ir_digest}: the kernel, or something it reads (a module constant by any "
                "route, a closure variable, a class attribute, an environment variable that a constexpr_function reads) is not what "
                "it was when the manifest was written, so this binary is stale (a build here would give another one); export it "
                "again (and pin the new manifest)"
            )
        if spec.source_sha256 is not None or comparable or pinned:
            return  # a generated kernel is told by its source, a compared IR digest matched, or the caller vouches for the manifest
        if recorded is None:
            why = unusable
        elif ir_triton is not None and built_with is None:
            why = (
                f"the entry does not record which Triton built it, and this process has Triton {ir_triton}, whose IR for the same "
                "kernel may print differently from that of the Triton that did, so the two digests cannot be compared"
            )
        elif ir_triton is not None and ir_triton != built_with:
            why = (
                f"the entry was built with Triton {built_with}, this process has Triton {ir_triton}, whose IR for the same kernel "
                "may print differently, so the two digests cannot be compared"
            )
        else:
            why = f"the IR of '{spec.triton}' cannot be recomputed here (Triton or the module is not importable)"
        raise TilefishManifestError(
            f"cannot tell whether the binary for {where} was built from the kernel '{spec.triton}' as the process holds it: "
            f"{why}. Pass pin=<the manifest's digest> to vouch for the manifest, or rebuild it"
        )

    def __repr__(self) -> str:
        library = f", {len(self._library)} library entries" if self._library else ""
        return f"<Manifest {str(self.directory)!r}: {len(self._artifacts)} binaries{library}>"
