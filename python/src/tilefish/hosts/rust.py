"""Typed Rust launchers for a manifest: ``emit(manifest, "launchers.rs")``.

The ``tilefish`` crate (``crates/tilefish``) loads a manifest directory and launches its cubins from Rust: it verifies
the digests, checks every precondition of the binary and launches through the CUDA driver. :func:`emit` writes the
part that is specific to one manifest: one module with a struct per binary (per stem, so per bucket) whose ``launch``
takes the tensors by name, with the element type in the Rust type, so that a ``bf16`` tensor cannot be passed where an
``f32`` one is wanted, and whose doc comments carry what the manifest says of the binary.

**Manifest schema 3 and the crate.** Python writes manifest schema 3 (``docs/manifest-schema-3.md``); the ``tilefish`` crate 0.0.1 reads
schema 2 only, so a module generated here loads under a crate that reads schema 3, and fails at ``load`` with the crate's own error
under 0.0.1. The emitter keeps working for what schema 2 could express (every Triton kernel whose parameters are dense tensors and ``i32``
dimension scalars: nothing else of the manifest reaches the output, and ``MANIFEST_DIGEST`` is the schema 3 digest) and **refuses**
everything the Rust host cannot launch yet with an error that says "not yet supported by the Rust host": a prebuilt PTX or cubin
(``kind: "binary"``), a strided view (``strides``, ``offset``, a ``stride_of`` scalar), a device scalar (a pointer with the shape ``[]``), a
64-bit scalar and a ``fixed`` argument, and a manifest with ``library`` entries (a pinned cuBLASLt GEMM). A Rust lane lifts these.

It works from the manifest alone. A schema 2 manifest records, for every pointer parameter, its declared shape
(``["Rows", 1024]``: a dimension expression, or the size of a static axis) and, for every entry, the constraints of its
named dimensions (``divisible_by``, ``buckets`` and the bucket this binary is built for), so ``launch`` takes nothing but
the tensors: the crate binds the dimensions from their shapes and derives every runtime scalar and the grid, like
``tilefish.run`` does, and refuses a call that does not fit. ``ops=`` (the operators the manifest was built from) is
optional: each declaration that has a binary in the manifest is checked against it (its compile spec must be exactly
what the manifest's entry says, the declaration digest included), and gives the docs the tilefish signature.

What the module holds, in order:

* ``MANIFEST_DIGEST``: the digest of the manifest it was generated from. ``load`` of every struct refuses a manifest with
  another digest, so a launcher can only run the binaries it was typed from.
* with ``embed=True``, the manifest file and its cubins as ``include_bytes!`` (paths relative to the generated file) and
  ``load_embedded()``, which verifies them like a directory and pins them to ``MANIFEST_DIGEST``.
* one struct per stem. ``STEM`` and ``BUCKETS`` name the binary (bucket semantics are exact: ``BucketedRms96`` and
  ``BucketedRms512`` are different structs and there is no padding), ``SHAPES`` the declared shape of every tensor,
  ``load(&Manifest, &Arc<CudaContext>)`` loads the binary for the device of a context and ``launch(&CudaStream, tensors...)`` checks the call and
  launches, and ``unsafe launch_raw(CUstream, tensors...)`` does the same on a raw handle. A CUDA graph is captured with regions
  and ``launch_raw`` (a launch of cudarc slices on a capturing stream is an error).

The output is deterministic: the same manifest gives the same bytes. Nothing in it depends on where the manifest was
written, except the ``include_bytes!`` paths with ``embed=True``.

What it trusts. The output is Rust source that a developer compiles and runs, and a manifest directory vouches for nothing by
itself: whoever can write it can put any text in any field. So :func:`emit` (and :func:`render`) take a *pinned* manifest only: a
directory with ``pin=<the manifest's digest>``, or a :class:`~tilefish.Manifest` that was loaded with a pin (or that
``tilefish.runtime.export`` returned, which has just been written from this process's builds). Then every field that reaches the
output is validated (``stem``, ``op``, architecture, file names, tensor and dimension names, dtypes, shapes, constexprs, the
grid, sizes: each must be what it is for), so a manifest cannot write a doc line that breaks out of its comment, an item, or a
string literal into the generated file; and the doc comments are written a line at a time, with line breaks that a field might carry
split into lines that are all comments.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .._build import spec_entry, write_file
from .._manifest import Manifest
from .._specvalid import parse_expression, parse_grid
from ..errors import TilefishManifestError, TilefishSpecError

RUST_ELEMENTS = {"bf16": "Bf16", "f16": "F16", "f32": "F32", "i32": "I32", "i64": "I64", "u8": "U8", "e4m3": "E4m3", "e5m2": "E5m2"}

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
_U64_MAX = 2**64 - 1
_ACCESS = ("in", "out", "inout")


@dataclass
class _Binary:
    """One stem of the manifest, as the generated code needs it."""

    stem: str
    entry: "dict[str, Any]"  # the entry of the first architecture: what the architectures agree on
    archs: "list[str]"
    struct: str = ""
    params: "list[dict[str, Any]]" = field(default_factory=list)  # the pointers of the ABI, in order
    scalars: "list[dict[str, Any]]" = field(default_factory=list)
    arg_names: "dict[str, str]" = field(default_factory=dict)  # ABI name -> Rust name
    signature: "str | None" = None  # the tilefish signature of the declaration, when it was given


def _rs_str(text: str) -> str:
    """A Rust string literal for plain ASCII text (identifiers, digits and the characters of dimension expressions).

    Anything else is refused, and ``\\`` and ``"`` are escaped: no text can end the literal.
    """
    if not isinstance(text, str) or not text.isascii() or not text.isprintable():
        raise TilefishManifestError(f"cannot write {text!r} in generated Rust source: it is not printable ASCII")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# what the generated module imports from the crate, and `Self`: a struct of these names would clash with them
_RESERVED_STRUCTS = frozenset(
    {"Self", "Arc", "Arch", "CUstream", "CudaContext", "CudaStream", "Error", "Kernel", "Manifest", "Pin", "TensorRef", "TensorMut", "Result", "Option", "Some", "None", "Ok", "Err"}
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
    if entry.get("kind") != "triton":
        _bad(who, f"it is a {entry.get('kind')!r} kernel (a prebuilt PTX or cubin), which is not yet supported by the Rust host")
    abi = _field(who, "abi", entry, "abi", dict)
    if abi.get("scratch_slots") != 2:
        _bad(who, f"abi.scratch_slots is {abi.get('scratch_slots')!r}: the launchers pass the two scratch slots of a Triton kernel")
    params = _field(who, "abi.params", abi, "params", list)
    names = set()
    for i, p in enumerate(params):
        where = f"abi.params[{i}]"
        name = _name(who, f"{where}.name", p.get("name") if isinstance(p, dict) else None)
        names.add(name)
    for i, p in enumerate(params):
        where = f"abi.params[{i}] ('{p['name']}')"
        kind = p.get("kind")
        if kind == "pointer":
            _name(who, f"{where}.dtype", p.get("dtype"), _DTYPE)
            if p.get("access") not in _ACCESS:
                _bad(who, f"{where}.access must be one of {list(_ACCESS)}, got {p.get('access')!r}")
            shape = _field(who, f"{where}.shape", p, "shape", list)
            if "strides" in p or "offset" in p:
                _bad(who, f"{where} takes a strided view (strides, offset), which is not yet supported by the Rust host")
            if not shape:
                _bad(who, f"{where} is a DevScalar (a pointer with the shape []), which is not yet supported by the Rust host")
            for axis, d in enumerate(shape):
                if isinstance(d, str):
                    _expression(who, f"{where}.shape[{axis}]", d)
                else:
                    _count(who, f"{where}.shape[{axis}]", d)
        elif kind == "scalar":
            if p.get("type") != "i32" or "stride_of" in p:
                _bad(who, f"{where} is a {p.get('type')!r} scalar" + (" that carries a stride" if "stride_of" in p else "") + ", which is not yet supported by the Rust host (it takes i32 dimension scalars)")
            _expression(who, f"{where}.dim", p.get("dim"))
            target = p.get("count_of")
            if target is not None and target not in names:
                _bad(who, f"{where}.count_of must name a parameter, got {target!r}")
            if target is not None:
                _name(who, f"{where}.count_of", target)
        else:
            _bad(who, f"{where}.kind must be 'pointer' or 'scalar', got {kind!r}" + (" (a fixed argument belongs to a prebuilt binary, which is not yet supported by the Rust host)" if kind == "fixed" else ""))
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


def _group(manifest: Manifest) -> "list[_Binary]":
    for artifact in manifest.artifacts:
        _validate_entry(artifact.entry)
    by_stem: "dict[str, list[dict[str, Any]]]" = {}
    for artifact in manifest.artifacts:
        by_stem.setdefault(artifact.stem, []).append(artifact.entry)
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
    structs = [b.struct for b in out]
    if len(set(structs)) != len(structs):
        dup = sorted({s for s in structs if structs.count(s) > 1})
        raise TilefishManifestError(f"two stems of the manifest give the same Rust name {dup}")
    return out


def _check_declarations(binaries: "list[_Binary]", ops: "Iterable[Any]") -> None:
    """Check each declaration against the manifest entries of its operator, and take its signature for the docs."""
    from ..aot import _first_difference, _show
    from ..runtime import _target

    declared: "dict[str, tuple[Any, Any]]" = {}  # op name -> (target, launched extern)
    for op in ops:
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


def _shape_text(shape: "list[Any]") -> str:
    return ", ".join(str(d) for d in shape)


def _doc(binary: _Binary) -> "list[str]":
    e = binary.entry
    lines: "list[str]" = []
    if binary.signature:
        lines += [f"`{binary.signature}`", ""]
    lines.append(f"The binary `{binary.stem}` of operator `{e['op']}`, built for {', '.join(binary.archs)}.")
    lines.append("")
    lines.append("Tensors, in the order of `launch`:")
    for p in binary.params:
        lines.append(f"- `{binary.arg_names[p['name']]}`: {p['access']}, {p['dtype']}, `[{_shape_text(p['shape'])}]`")
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
    lines.append("")
    lines.append(f"Grid `{e['grid']}`, block {e['block'][0]} threads, {e['shared']} bytes of dynamic shared memory.")
    if binary.scalars:
        lines.append("")
        lines.append("Runtime scalars (`i32`) of the binary, derived by `launch` from the shapes:")
        for s in binary.scalars:
            how = f"the element count of `{binary.arg_names[s['count_of']]}`" if "count_of" in s else f"`{s['dim']}`"
            lines.append(f"- `{s['name']}` is {how}")
    return lines


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
    if manifest.library_entries:
        raise TilefishManifestError(
            f"the manifest in {str(manifest.directory)!r} has library entries ({', '.join(sorted(e['stem'] for e in manifest.library_entries))}: "
            "pinned cuBLASLt GEMMs), which are not yet supported by the Rust host (the tilefish crate reads manifest schema 2; the library "
            "section needs the crate that reads schema 3)"
        )
    binaries = _group(manifest)
    _check_declarations(binaries, ops)
    elements = sorted({RUST_ELEMENTS[p["dtype"]] for b in binaries for p in b.params})
    uses = ["Arch", "CUstream", "CudaContext", "CudaStream", "Error", "Kernel", "Manifest"]
    if embed:
        uses.append("Pin")
    if any(p["access"] == "in" for b in binaries for p in b.params):
        uses.append("TensorRef")
    if any(p["access"] != "in" for b in binaries for p in b.params):
        uses.append("TensorMut")
    uses = sorted(set(uses) | set(elements))
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

    The launchers are typed from the manifest alone (a schema 2 manifest has the declared shape of every pointer and the
    constraints of every dimension). ``ops`` are optional: the declarations the manifest was built from (kernels, externs or
    lowered kernels). Each one that has a binary in the manifest must be exactly what the manifest says
    (:class:`~tilefish.errors.TilefishManifestError` otherwise), and gives the docs the tilefish signature. With
    ``embed=True`` the manifest file and the cubins are embedded with ``include_bytes!``, by paths relative to ``path``. The
    file is written through a temporary file and renamed, and a symbolic link at ``path`` is refused.
    """
    path = pathlib.Path(path)
    source = render(manifest, ops=list(ops), embed=embed, relative_to=path.resolve().parent, pin=pin)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_file(path, source.encode("utf-8"))
    return path
