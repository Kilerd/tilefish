"""Typed Rust launchers for a manifest: ``emit(manifest, "launchers.rs")``.

The ``tilefish`` crate (``crates/tilefish``) loads a manifest directory and launches its cubins from Rust: it verifies
the digests, checks every precondition of the binary and launches through the CUDA driver. :func:`emit` writes the
part that is specific to one manifest: one module with a struct per binary (per stem, so per bucket) whose ``launch``
takes the tensors by name, with the element type in the Rust type, so that a ``bf16`` tensor cannot be passed where an
``f32`` one is wanted, and whose doc comments carry what the manifest says of the binary.

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
  ``load(&Manifest)`` loads the binary for the current device and ``launch(stream, tensors...)`` checks the call and
  launches on a raw ``CUstream`` (also inside a CUDA graph capture).

The output is deterministic: the same manifest gives the same bytes. Nothing in it depends on where the manifest was
written, except the ``include_bytes!`` paths with ``embed=True``.
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
from ..errors import TilefishManifestError

RUST_ELEMENTS = {"bf16": "Bf16", "f16": "F16", "f32": "F32", "i32": "I32", "i64": "I64", "u8": "U8", "e4m3": "E4m3", "e5m2": "E5m2"}

# what the binaries of one stem must agree on, whatever their architecture
_SHARED_FIELDS = (
    "op",
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
    "typeof unsized virtual yield try gen stream".split()
)


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
    """A Rust string literal for plain ASCII text (identifiers, digits and the characters of dimension expressions)."""
    if not text.isascii() or not text.isprintable():
        raise TilefishManifestError(f"cannot write {text!r} in generated Rust source: it is not printable ASCII")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _camel(stem: str) -> str:
    pieces = [p for p in re.split(r"[._]+", stem) if p]
    name = "".join(p[0].upper() + p[1:] for p in pieces)
    return name if name and not name[0].isdigit() else "K" + name


def _snake_ident(name: str) -> str:
    return name + "_" if name in _RUST_KEYWORDS else name


def _scalar_text(value: Any) -> str:
    if isinstance(value, dict):  # a tagged non-finite float
        return str(value["float"])
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value)


def _group(manifest: Manifest) -> "list[_Binary]":
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
    return "".join(f"{indent}///{' ' + ln if ln else ''}\n" for ln in lines)


def _render_binary(b: _Binary) -> str:
    e = b.entry
    elements = {p["name"]: RUST_ELEMENTS[p["dtype"]] for p in b.params}
    tensor_params = []
    for p in b.params:
        wrapper = "TensorRef" if p["access"] == "in" else "TensorMut"
        tensor_params.append(f"{b.arg_names[p['name']]}: {wrapper}<'_, {elements[p['name']]}>")
    args = ", ".join(f"{b.arg_names[p['name']]}.arg()" for p in b.params)
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
    out.append("    /// Load the binary for the current device from `manifest`, in the CUDA context that is current.\n")
    out.append("    ///\n")
    out.append("    /// Refuses a manifest whose digest is not [`MANIFEST_DIGEST`], and one without a binary for the device.\n")
    out.append("    pub fn load(manifest: &Manifest) -> Result<Self, Error> {\n")
    out.append("        manifest.require_digest(MANIFEST_DIGEST)?;\n")
    out.append("        let kernel = manifest.kernel(Self::STEM, Self::BUCKETS, Arch::current()?)?;\n")
    out.append("        Ok(Self { kernel })\n")
    out.append("    }\n\n")
    out.append("    /// The loaded kernel.\n")
    out.append("    pub fn kernel(&self) -> &Kernel {\n        &self.kernel\n    }\n\n")
    doc = [
        "Check the call and launch the kernel on `stream` (a raw `CUstream`: it may be capturing a CUDA graph).",
        "",
        "Every precondition of the binary is checked first (dtype, rank, static sizes, the dimensions against each other, the",
        "bucket, `divisible_by`, 16-byte alignment, `max_elements`, aliasing, the grid against the device); nothing is launched",
        "when one fails. The dimensions, the runtime scalars and the grid come from the shapes of the tensors.",
    ]
    out.append(_doc_comment(doc, "    "))
    out.append("    #[allow(clippy::too_many_arguments, non_snake_case)]\n")
    out.append("    pub fn launch(\n        &self,\n        stream: CUstream,\n")
    for p in tensor_params:
        out.append(f"        {p},\n")
    out.append("    ) -> Result<(), Error> {\n")
    out.append(f"        self.kernel.launch(stream, &[{args}])\n")
    out.append("    }\n}\n")
    return "".join(out)


def render(manifest: Manifest, *, ops: "Iterable[Any]" = (), embed: bool = False, relative_to: "str | os.PathLike[str] | None" = None) -> str:
    """The Rust source :func:`emit` writes, as a string. ``relative_to`` is the directory of the generated file (for ``embed``)."""
    binaries = _group(manifest)
    _check_declarations(binaries, ops)
    elements = sorted({RUST_ELEMENTS[p["dtype"]] for b in binaries for p in b.params})
    uses = ["Arch", "CUstream", "Error", "Kernel", "Manifest"]
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
    manifest: Manifest,
    path: "str | os.PathLike[str]",
    *,
    ops: "Iterable[Any]" = (),
    embed: bool = False,
) -> pathlib.Path:
    """Write the typed Rust launchers of ``manifest`` to ``path`` (one ``.rs`` module) and return the path.

    The launchers are typed from the manifest alone (a schema 2 manifest has the declared shape of every pointer and the
    constraints of every dimension). ``ops`` are optional: the declarations the manifest was built from (kernels, externs or
    lowered kernels). Each one that has a binary in the manifest must be exactly what the manifest says
    (:class:`~tilefish.errors.TilefishManifestError` otherwise), and gives the docs the tilefish signature. With
    ``embed=True`` the manifest file and the cubins are embedded with ``include_bytes!``, by paths relative to ``path``. The
    file is written through a temporary file and renamed, and a symbolic link at ``path`` is refused.
    """
    path = pathlib.Path(path)
    source = render(manifest, ops=list(ops), embed=embed, relative_to=path.resolve().parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_file(path, source.encode("utf-8"))
    return path
