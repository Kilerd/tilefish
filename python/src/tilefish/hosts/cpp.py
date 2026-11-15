"""Typed C++ launchers for a manifest: ``emit(manifest, "launchers.hpp")``; and a typed runner of a launch plan: ``emit_plan(directory, "runner.hpp", pin=...)``.

The C++ host is the C ABI of the Rust host (``libtilefish_capi``, ``crates/tilefish-capi``) with its header-only C++17 wrapper
``tilefish.hpp`` (``docs/rfcs/0002-cpp-host.md``): it loads a manifest directory, verifies its digests and launches what it describes with
every check of the Rust crate. :func:`emit` writes the part that is specific to one manifest, as :mod:`tilefish.hosts.rust` does for Rust:
one header with a class per binary (per stem, so per bucket) whose ``launch`` takes the tensors by name, with the element type in the C++
type (``tilefish::TensorRef<tilefish::Bf16>``), so that an ``f32`` tensor cannot be passed where a ``bf16`` one is wanted, and whose doc
comments carry what the manifest says of the binary.

What the header holds, in its namespace (``tilefish_launchers`` by default):

* ``MANIFEST_DIGEST``: the digest of the manifest it was generated from. ``load`` of every class refuses a manifest with another digest.
* with ``embed=True``, the declaration of ``load_embedded()``, which a companion source next to the header (``<header>_embedded.cpp``)
  defines: the manifest file and its cubins as byte arrays, verified like a directory and pinned to ``MANIFEST_DIGEST``. Compile that file
  once (large arrays do not belong in every translation unit that includes the header).
* one class per stem of a kernel, then one per stem of a library entry (a pinned GEMM or convolution). ``STEM`` and ``BUCKETS`` name the binary,
  ``<TENSOR>_SHAPE`` the declared shape of each tensor (dimension expressions; a number is a size baked in), ``<TENSOR>_AXIS<i>`` each size baked in;
  ``load(manifest, ctx)`` loads the binary for the device of a context, ``launch(stream, tensors...)`` checks the call and launches, and
  ``check(tensors...)`` checks it and launches nothing. A GEMM class adds ``M``, ``N``, ``K``, ``BIAS``, ``WORKSPACE_BYTES`` and the cuBLASLt
  version of each architecture's pin; a convolution class ``WORKSPACE_BYTES`` and the cuDNN version of each pin.

It works from the manifest alone, and takes a *pinned* manifest only (a directory with ``pin=``, or a :class:`~tilefish.Manifest` loaded with a
pin), exactly as :mod:`tilefish.hosts.rust`: it reuses that module's validation, so every field that reaches the output is what it must be,
and a manifest cannot write a doc line that breaks out of its comment, an identifier or a string literal into the generated file. The output is
deterministic.

**Launch plans.** :func:`emit_plan` writes a typed runner of a ``plan.json``: a class named after the graph whose ``load`` pins the plan (and so its
manifest) and instantiates it, and whose ``launch`` and ``capture`` take a ``<Graph>Externals`` aggregate (one member per external tensor, with its
element type) and a ``<Graph>Scalars`` (one member per device scalar).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
from typing import Any, Callable, Iterable, Iterator

from .._build import write_file
from .._library import IMPLS as LIBRARY_IMPLS
from .._manifest import Manifest, manifest_digest
from ..errors import TilefishManifestError
from . import rust
from .rust import RUST_ELEMENTS, _ARCH, _DIGEST, _FILE, _IDENT, _count, _doc_comment, _shape, _shape_text

# the element tags of tilefish.hpp have the names of the Rust crate's element types
CPP_ELEMENTS = {dtype: f"tilefish::{name}" for dtype, name in RUST_ELEMENTS.items()}

_CPP_KEYWORDS = frozenset(
    "alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t char32_t class compl concept const consteval "
    "constexpr constinit const_cast continue co_await co_return co_yield decltype default delete do double dynamic_cast else enum explicit "
    "export extern false float for friend goto if inline int long mutable namespace new noexcept not not_eq nullptr operator or or_eq private "
    "protected public register reinterpret_cast requires return short signed sizeof static static_assert static_cast struct switch template "
    "this thread_local throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t while xor xor_eq final "
    "override import module".split()
)
# the names the generated code uses inside a launch (its stream, the namespaces it names, its members), and the names the C library headers or
# the compiler may define as macros: a parameter of one of these names is renamed
_CPP_RESERVED = frozenset(
    "stream tilefish std kernel_ gemm_ conv_ run_ NULL errno EOF linux unix i386 assert offsetof int8_t int16_t int32_t int64_t uint8_t "
    "uint16_t uint32_t uint64_t size_t CUstream CUcontext".split()
)
# the members of a generated class: a class of one of these names (a stem of capitals) gets `Op` after it, as one named like an import of the
# Rust module does
_MEMBER_NAMES = frozenset({"STEM", "BUCKETS", "M", "N", "K", "BIAS"})
_SCRATCH = "global_scratch"

# the host type of a device scalar's value in a plan runner, by dtype (a dtype that is none of these is given as its bits)
_DEVSCALAR_TYPES = {"i32": "int32_t", "i64": "int64_t", "u8": "uint8_t", "f32": "float"}


def _cpp_str(text: Any) -> str:
    """A C++ string literal for plain ASCII text (identifiers, digits and the characters of dimension expressions): anything else, a ``"`` and a
    ``\\`` included, is refused, so no text can end the literal or start an escape."""
    if not isinstance(text, str) or not text.isascii() or not text.isprintable() or '"' in text or "\\" in text or "??" in text:
        raise TilefishManifestError(f"cannot write {text!r} in generated C++ source: it is not printable ASCII without quotes and backslashes")
    return '"' + text + '"'


def _cpp_ident(name: str, who: str) -> str:
    """The C++ name of a parameter or member: a keyword or a reserved name gets ``_`` after it (``arg`` after one that ends with ``_``); a name that
    C++ reserves for the implementation (``_X``, ``__``) is refused."""
    if name.startswith("_") and (len(name) == 1 or name[1].isupper() or name[1] == "_") or "__" in name:
        raise TilefishManifestError(f"cannot write C++ launchers for {who}: '{name}' is an identifier C++ reserves for the implementation")
    if name in _CPP_KEYWORDS or name in _CPP_RESERVED:
        return name + ("arg" if name.endswith("_") else "_")
    return name


def _class_name(struct: str) -> str:
    return struct + "Op" if struct in _MEMBER_NAMES else struct


def _as_cpp(message: str) -> str:
    """A message of :mod:`tilefish.hosts.rust`'s validation, about the C++ source."""
    return message.replace("Rust launchers", "C++ launchers").replace("Rust source", "C++ source").replace("in Rust", "in C++").replace(
        "Rust identifiers", "C++ identifiers"
    ).replace("Rust name", "C++ name")


@contextlib.contextmanager
def _cpp_errors() -> Iterator[None]:
    try:
        yield
    except TilefishManifestError as e:
        raise TilefishManifestError(_as_cpp(str(e))) from None


def _namespace(namespace: Any) -> str:
    pieces = namespace.split("::") if isinstance(namespace, str) else [None]
    for piece in pieces:
        if not isinstance(piece, str) or not _IDENT.fullmatch(piece) or piece in _CPP_KEYWORDS or piece.startswith("_") or "__" in piece:
            raise TilefishManifestError(f"the namespace of generated C++ must be identifiers joined by '::', got {namespace!r}")
    return namespace


def _guard(kind: str, namespace: str, *parts: str) -> str:
    return "_".join(["TILEFISH", kind, namespace.replace("::", "_"), *parts]).upper() + "_HPP"


# -- the docs -----------------------------------------------------------------------------------------------------------------------------------

# what the doc text of the Rust emitter says of the Rust crate, said of the C++ wrapper
_DOC_REPLACEMENTS = (
    (
        "make it with `TensorRef::from_cuda_strided` or `TensorRef::new_strided` (its element strides, and the element offset of its first element)",
        "make it with the `TensorRef` constructor that takes element strides and the element offset of its first element",
    ),
    ("(made with the shape `&[]` or `&[1]`)", "(made with the shape `{}` or `{1}`)"),
    ("and the other buckets are other structs", "and the other buckets are other classes"),
    ("`Gemm::alloc_workspace` allocates one)", "the host allocates it)"),
    ("`Conv::alloc_workspace` allocates one)", "the host allocates it)"),
    ("(the crate checks it; a plan gives its workspace)", "(tilefish checks it; a plan gives its workspace)"),
)
_WRAPPED = "\x00wrap:"


def _docs(make: Callable[[Any], "list[str]"], obj: Any) -> "list[str]":
    """The doc lines the Rust emitter writes for ``obj``, with what they say of the Rust crate said of the C++ wrapper: the paragraphs it wraps are
    adapted whole and wrapped again (``rust._WRAP_HOOK``, a context variable: another thread's Rust emitter wraps as usual)."""
    token = rust._WRAP_HOOK.set(lambda text: [_WRAPPED + text])
    try:
        lines = make(obj)
    finally:
        rust._WRAP_HOOK.reset(token)
    wrap = rust._wrap
    out: "list[str]" = []
    for line in lines:
        wrapped = line.startswith(_WRAPPED)
        text = line[len(_WRAPPED) :] if wrapped else line
        for old, new in _DOC_REPLACEMENTS:
            text = text.replace(old, new)
        out += wrap(text) if wrapped else [text]
    return out


# -- launchers ----------------------------------------------------------------------------------------------------------------------------------


def _array(element: str, items: str, n: int) -> str:
    return f"std::array<{element}, {n}>"


def _init(items: str) -> str:
    """The initializer of a ``std::array`` constant (an empty one has ``{}``)."""
    return f" = {{{{{items}}}}}" if items else " = {}"


def _shape_constants(names: "dict[str, str]", params: "list[dict[str, Any]]", baked: bool, who: str) -> "list[str]":
    out = []
    seen: "set[str]" = set(_MEMBER_NAMES) | {"WORKSPACE_BYTES", "PINNED_ARCHS", "PINNED_WITH"}

    def constant(name: str) -> str:
        if name in seen:
            raise TilefishManifestError(f"cannot write C++ launchers for {who}: two constants of its class would both be called {name}")
        seen.add(name)
        return name

    for p in params:
        arg = names[p["name"]]
        upper = arg.upper().rstrip("_")
        items = ", ".join(_cpp_str(str(d)) for d in p["shape"])
        out.append(f"    /// The declared shape of `{arg}`, as dimension expressions (a number is a size baked in).\n")
        out.append(f"    static constexpr {_array('const char*', items, len(p['shape']))} {constant(upper + '_SHAPE')}{_init(items)};\n")
    if baked:
        for p in params:
            arg = names[p["name"]]
            for axis, d in enumerate(p["shape"]):
                if isinstance(d, int):
                    out.append(f"    /// Axis {axis} of `{arg}`, baked into the binary.\n")
                    out.append(f"    static constexpr uint64_t {constant(f'{arg.upper().rstrip(chr(95))}_AXIS{axis}')} = {d};\n")
    return out


def _buckets(entry: "dict[str, Any]") -> str:
    items = ", ".join(f"{{{_cpp_str(d)}, {v}}}" for d, v in entry["buckets"].items())
    return f"static constexpr {_array('tilefish_bucket', items, len(entry['buckets']))} BUCKETS{_init(items)};\n"


_SPAN = "tilefish::detail::Span<tilefish_bucket>(BUCKETS.data(), BUCKETS.size())"


def _signature(name: str, params: "list[str]") -> str:
    """A member function's name and parameters, one parameter a line."""
    if not params:
        return f"{name}()"
    return f"{name}(\n" + ",\n".join(f"        {p}" for p in params) + ")"


def _braced(items: "list[str]", indent: str) -> str:
    """A braced list of items, one a line."""
    return indent + "{\n" + "".join(f"{indent}    {item},\n" for item in items) + indent + "}"


def _render_binary(b: Any) -> str:
    e = b.entry
    cls = b.struct
    tensor_params, args = [], []
    for p in b.params:
        wrapper = "TensorRef" if p["access"] == "in" else "TensorMut"
        tensor_params.append(f"const tilefish::{wrapper}<{CPP_ELEMENTS[p['dtype']]}>& {b.arg_names[p['name']]}")
        args.append(b.arg_names[p["name"]])
    if e["launch"]["global_scratch_size"]:  # a kernel with a global scratch: the last tensor (manifest schema 3, section 4.6)
        scratch = _SCRATCH
        while scratch in b.arg_names.values():
            scratch += "_"
        tensor_params.append(f"const tilefish::TensorMut<tilefish::U8>& {scratch}")
        args.append(scratch)
    given_params = [f"{'int32_t' if g['type'] == 'i32' else 'int64_t'} {b.arg_names[g['name']]}" for g in b.given]
    given = ", ".join(f"{{{_cpp_str(g['name'])}, {b.arg_names[g['name']]}}}" for g in b.given)
    call_args = "{" + ", ".join(args) + "}" + (f", {{{given}}}" if given else "")
    out = [_doc_comment(_docs(rust._doc, b)), f"class {cls} {{\npublic:\n"]
    out.append("    /// The stem of the binary in the manifest.\n")
    out.append(f"    static constexpr const char* STEM = {_cpp_str(b.stem)};\n")
    out.append("    /// The bucket values the binary is built for (none when it is not bucketed).\n")
    out.append("    " + _buckets(e))
    out += _shape_constants(b.arg_names, b.params, True, f"'{b.stem}'")
    out.append("\n")
    out.append("    /// Load the binary for the device of `ctx` from `manifest`, in that context (which the launcher keeps alive).\n")
    out.append("    ///\n")
    out.append("    /// Refuses a manifest whose digest is not `MANIFEST_DIGEST`, and one without a binary for the device.\n")
    out.append(f"    static {cls} load(const tilefish::Manifest& manifest, const tilefish::Context& ctx) {{\n")
    out.append("        manifest.require_digest(MANIFEST_DIGEST);\n")
    out.append(f"        return {cls}(manifest.kernel(ctx, STEM, {_SPAN}));\n")
    out.append("    }\n\n")
    out.append("    /// The loaded kernel.\n")
    out.append("    const tilefish::Kernel& kernel() const { return kernel_; }\n\n")
    doc = [
        "Check the call and launch the kernel on `stream`, a stream of the context the kernel was loaded in (or nullptr, the legacy default",
        "stream). Every precondition of the binary is checked first (dtype, rank, static sizes, the dimensions against each other, the",
        "bucket, `divisible_by`, alignment, `max_elements`, aliasing, the region of each tensor, the grid against the device); nothing is",
        "launched when one fails, and `tilefish::Error` is thrown. The dimensions, the runtime scalars and the grid come from the shapes of",
        "the tensors. The memory of the tensors is the host's: it stays allocated, and is ordered against the host's other streams, until",
        "the launch has completed on `stream`. On a stream the host is capturing, the launch is captured like any kernel node.",
    ]
    out.append(_doc_comment(doc, "    "))
    out.append(f"    void {_signature('launch', ['CUstream stream', *tensor_params, *given_params])} const {{\n")
    out.append(f"        kernel_.launch(stream, {call_args});\n")
    out.append("    }\n\n")
    out.append("    /// Check a call as `launch` does, and launch nothing.\n")
    out.append(f"    void {_signature('check', [*tensor_params, *given_params])} const {{\n")
    out.append(f"        kernel_.check({call_args});\n")
    out.append("    }\n\n")
    out.append("private:\n")
    out.append(f"    explicit {cls}(tilefish::Kernel loaded) : kernel_(std::move(loaded)) {{}}\n")
    out.append("    tilefish::Kernel kernel_;\n};\n")
    return "".join(out)


def _render_library(g: Any) -> str:
    e = g.entry
    cls = g.struct
    member = "conv" if g.conv else "gemm"
    kind = "Conv" if g.conv else "Gemm"
    tensor_params = [
        f"const tilefish::{'TensorMut' if p['access'] == 'out' else 'TensorRef'}<{CPP_ELEMENTS[p['dtype']]}>& {g.arg_names[p['name']]}" for p in g.params
    ]
    tensor_params.append(f"const tilefish::TensorMut<tilefish::U8>& {g.workspace}")
    call_args = "{" + ", ".join([g.arg_names[p["name"]] for p in g.params] + [g.workspace]) + "}"
    out = [_doc_comment(_docs(rust._doc_conv if g.conv else rust._doc_gemm, g)), f"class {cls} {{\npublic:\n"]
    out.append("    /// The stem of the library entry in the manifest.\n")
    out.append(f"    static constexpr const char* STEM = {_cpp_str(g.stem)};\n")
    out.append("    /// The bucket values the entry is built for (none when it is not bucketed).\n")
    out.append("    " + _buckets(e))
    out += _shape_constants(g.arg_names, g.params, False, f"'{g.stem}'")
    if not g.conv:
        bias = LIBRARY_IMPLS[e["impl"]]
        layout = e["transa"] + e["transb"]
        if e["transb"] == "N":
            out.append("    /// Rows of `a` and of the result.\n")
        else:
            out.append(f"    /// Columns of `a` (`[K, M]`: the layout {layout}), and rows of the result.\n")
        out.append(f"    static constexpr uint64_t M = {e['m']};\n")
        if e["transa"] == "N":
            out.append(f"    /// Columns of `b` (`[K, N]`: the layout {layout}), and of the result.\n")
            out.append(f"    static constexpr uint64_t N = {e['n']};\n")
            out.append("    /// The columns of `a` and the rows of `b`.\n" if e["transb"] == "N" else "    /// The rows of `a` and of `b`.\n")
        else:
            out.append("    /// Rows of `b`, and columns of the result.\n")
            out.append(f"    static constexpr uint64_t N = {e['n']};\n")
            out.append("    /// The columns of `a` and of `b`.\n" if e["transb"] == "N" else "    /// The rows of `a` and the columns of `b`.\n")
        out.append(f"    static constexpr uint64_t K = {e['k']};\n")
        out.append("    /// Whether the GEMM adds a bias.\n")
        out.append(f"    static constexpr bool BIAS = {'true' if bias else 'false'};\n")
        out.append("    /// The workspace the algorithm was pinned under: the size of the workspace to give `launch`, in bytes.\n")
    else:
        out.append("    /// The workspace the pin was chosen under: the size of the workspace to give `launch`, in bytes.\n")
    out.append(f"    static constexpr uint64_t WORKSPACE_BYTES = {e['workspace_bytes']};\n")
    library = "cuDNN (`cudnnGetVersion`)" if g.conv else "cuBLASLt (`cublasLtGetVersion`)"
    version = "cudnn_version" if g.conv else "cublaslt_version"
    archs = ", ".join(_cpp_str(x["arch"]) for x in g.entries)
    versions = ", ".join(str(_count(f"'{g.stem}'", version, x[version], 1)) for x in g.entries)
    out.append(f"    /// The architectures the manifest pinned the entry for; `PINNED_WITH[i]` is the version of {library} that chose it on `PINNED_ARCHS[i]`.\n")
    out.append(f"    static constexpr {_array('const char*', archs, len(g.entries))} PINNED_ARCHS{_init(archs)};\n")
    out.append(f"    static constexpr {_array('uint64_t', versions, len(g.entries))} PINNED_WITH{_init(versions)};\n")
    out.append("\n")
    what = "convolution" if g.conv else "GEMM"
    out.append(f"    /// Load the {what} for the device of `ctx` from `manifest`, in that context (which the launcher keeps alive).\n")
    out.append("    ///\n")
    if g.conv:
        out.append("    /// Refuses a manifest whose digest is not `MANIFEST_DIGEST`, one without an entry for the device, a cuDNN of another version than the\n")
        out.append("    /// one that chose the pin (see `PINNED_WITH`), a device of another number of SMs and a process with `NVIDIA_TF32_OVERRIDE` set.\n")
    else:
        out.append("    /// Refuses a manifest whose digest is not `MANIFEST_DIGEST`, one without an entry for the device, a cuBLASLt of another\n")
        out.append("    /// version than the one that chose the algorithm (see `PINNED_WITH`), and a device of another number of SMs than it was pinned on.\n")
    out.append(f"    static {cls} load(const tilefish::Manifest& manifest, const tilefish::Context& ctx) {{\n")
    out.append("        manifest.require_digest(MANIFEST_DIGEST);\n")
    out.append(f"        return {cls}(manifest.{member}(ctx, STEM, {_SPAN}));\n")
    out.append("    }\n\n")
    out.append(f"    /// The loaded {what}.\n")
    out.append(f"    const tilefish::{kind}& {member}() const {{ return {member}_; }}\n\n")
    if g.conv:
        doc = [
            "Check the call and run the convolution on `stream`, a stream of the context it was loaded in. The tensors are the input, the filter,",
            "the result and the workspace, which holds at least `workspace_required()` bytes. Every precondition is checked first (dtype, rank,",
            "the sizes, alignment, contiguity, aliasing); nothing is launched when one fails, and `tilefish::Error` is thrown.",
        ]
    else:
        doc = [
            "Check the call and run the GEMM on `stream`, a stream of the context it was loaded in. The tensors are the operands, the result and",
            "the workspace, which holds at least `workspace_required()` bytes. Every precondition is checked first (dtype, rank, the sizes,",
            "16-byte alignment, contiguity, aliasing); nothing is launched when one fails, and `tilefish::Error` is thrown.",
        ]
    out.append(_doc_comment(doc, "    "))
    out.append(f"    void {_signature('launch', ['CUstream stream', *tensor_params])} const {{\n")
    out.append(f"        {member}_.launch(stream, {call_args});\n")
    out.append("    }\n\n")
    out.append("    /// Check a call as `launch` does, and launch nothing.\n")
    out.append(f"    void {_signature('check', tensor_params)} const {{\n")
    out.append(f"        {member}_.check({call_args});\n")
    out.append("    }\n\n")
    out.append("private:\n")
    out.append(f"    explicit {cls}(tilefish::{kind} loaded) : {member}_(std::move(loaded)) {{}}\n")
    out.append(f"    tilefish::{kind} {member}_;\n}};\n")
    return "".join(out)


def _prepare(manifest: Any, ops: "Iterable[Any]", pin: "str | None") -> "tuple[Manifest, list[Any], list[Any]]":
    """The manifest (pinned), and its stems as :mod:`tilefish.hosts.rust` groups and validates them, with the C++ names of their classes and
    parameters."""
    with _cpp_errors():
        manifest = rust._trusted(manifest, pin)
        binaries, gemms = rust._group(manifest)
        rust._check_declarations(binaries, gemms, ops)
    for b in binaries:
        who = f"'{b.stem}'"
        names = [_cpp_ident(p["name"], who) for p in b.params + b.given]
        if len(set(names)) != len(names):
            raise TilefishManifestError(f"cannot write C++ launchers for {who}: two parameter names collide as C++ identifiers: {names}")
        b.arg_names = {p["name"]: n for p, n in zip(b.params + b.given, names)}
        b.struct = _class_name(b.struct)
    for g in gemms:
        who = f"'{g.stem}'"
        names = [_cpp_ident(p["name"], who) for p in g.params]
        if len(set(names)) != len(names) or g.workspace in names:
            raise TilefishManifestError(f"cannot write C++ launchers for {who}: two tensor names collide as C++ identifiers, or one is 'workspace': {names}")
        g.arg_names = {p["name"]: n for p, n in zip(g.params, names)}
        g.struct = _class_name(g.struct)
    classes = [x.struct for x in [*binaries, *gemms]]
    if len(set(classes)) != len(classes):
        dup = sorted({s for s in classes if classes.count(s) > 1})
        raise TilefishManifestError(f"two stems of the manifest give the same C++ class name {dup}")
    return manifest, binaries, gemms


def render(
    manifest: Any,
    *,
    ops: "Iterable[Any]" = (),
    embed: bool = False,
    pin: "str | None" = None,
    namespace: str = "tilefish_launchers",
) -> str:
    """The C++ header :func:`emit` writes, as a string. ``manifest`` is a directory with ``pin``, or a pinned :class:`~tilefish.Manifest`; every
    field of it that reaches the output is validated first. With ``embed=True`` the header declares ``load_embedded()``, which the companion
    source (:func:`render_embedded`) defines."""
    namespace = _namespace(namespace)
    manifest, binaries, gemms = _prepare(manifest, list(ops), pin)
    digest = _cpp_str(manifest.digest)
    guard = _guard("LAUNCHERS", namespace, manifest.digest[:16])
    out = [
        "// @generated by tilefish.hosts.cpp. Do not edit: generate it again from the manifest.\n",
        f"// The manifest it was generated from has digest {manifest.digest}.\n",
        f"#ifndef {guard}\n",
        f"#define {guard}\n",
        "\n",
        "#include <array>\n",
        "#include <cstdint>\n",
        "#include <utility>\n",
        "\n",
        '#include "tilefish.hpp"\n',
        "\n",
        f"namespace {namespace} {{\n",
        "\n",
        "/// The digest of the manifest these launchers were generated from: `load` refuses any other manifest.\n",
        f"inline constexpr const char* MANIFEST_DIGEST = {digest};\n",
    ]
    if embed:
        out.append("\n")
        out.append("/// The embedded manifest and its cubins (defined by the companion source of this header, `<header>_embedded.cpp`): verified like a\n")
        out.append("/// directory, and pinned to `MANIFEST_DIGEST`.\n")
        out.append("tilefish::Manifest load_embedded();\n")
    for binary in binaries:
        out.append("\n")
        out.append(_render_binary(binary))
    for gemm in gemms:
        out.append("\n")
        out.append(_render_library(gemm))
    out.append("\n")
    out.append(f"}}  // namespace {namespace}\n")
    out.append("\n")
    out.append(f"#endif  // {guard}\n")
    return "".join(out)


def _hex_lines(data: bytes) -> "list[str]":
    return ["    " + ", ".join(f"0x{b:02x}" for b in data[i : i + 16]) + ",\n" for i in range(0, len(data), 16)]


def render_embedded(
    manifest: Any,
    *,
    pin: "str | None" = None,
    namespace: str = "tilefish_launchers",
    header_name: str = "generated.hpp",
) -> str:
    """The companion source of a header written with ``embed=True``: the manifest file and every cubin of the manifest as byte arrays, and
    ``load_embedded()``, which verifies them like a directory and pins them to ``MANIFEST_DIGEST``. ``header_name`` is the file name of the
    header it includes (next to it)."""
    namespace = _namespace(namespace)
    if not isinstance(header_name, str) or not _FILE.fullmatch(header_name):
        raise TilefishManifestError(f"the header of the embedded manifest must be a plain file name, got {header_name!r}")
    manifest, _, _ = _prepare(manifest, (), pin)
    data = manifest.path.read_bytes()
    try:
        found = manifest_digest(json.loads(data))
    except (ValueError, TypeError, KeyError) as e:
        raise TilefishManifestError(f"the manifest file {str(manifest.path)!r} cannot be read again to embed it: {e}") from None
    if found != manifest.digest:
        raise TilefishManifestError(
            f"the manifest file {str(manifest.path)!r} has digest {found} now, but the manifest that was pinned has {manifest.digest}: it changed since"
        )
    files = sorted({a.entry["cubin"]: a.entry["sha256"] for a in manifest.artifacts}.items())
    out = [
        "// @generated by tilefish.hosts.cpp. Do not edit: generate it again from the manifest.\n",
        f"// The manifest of digest {manifest.digest} and its cubins, embedded: compile this file once, with the header it includes.\n",
        "\n",
        f'#include "{header_name}"\n',
        "\n",
        f"namespace {namespace} {{\n",
        "namespace {\n",
        "\n",
        "// manifest.json\n",
        "const unsigned char MANIFEST_JSON[] = {\n",
        *_hex_lines(data),
        "};\n",
    ]
    for i, (name, sha256) in enumerate(files):
        _cpp_str(name)
        blob = (manifest.directory / name).read_bytes()
        if hashlib.sha256(blob).hexdigest() != sha256:
            raise TilefishManifestError(f"the cubin {name!r} is not the file the manifest was verified with (its sha256 changed): it is not embedded")
        out += ["\n", f"// {name}\n", f"const unsigned char FILE_{i}[] = {{\n", *_hex_lines(blob), "};\n"]
    out.append("\n")
    out.append("}  // namespace\n")
    out.append("\n")
    out.append("tilefish::Manifest load_embedded() {\n")
    if files:
        out.append("    static const tilefish_file files[] = {\n")
        for i, (name, _) in enumerate(files):
            out.append(f"        {{{_cpp_str(name)}, FILE_{i}, sizeof(FILE_{i})}},\n")
        out.append("    };\n")
        span = f"tilefish::detail::Span<tilefish_file>(files, {len(files)})"
    else:
        span = "tilefish::detail::Span<tilefish_file>()"
    out.append(f"    return tilefish::Manifest::from_embedded(MANIFEST_JSON, sizeof(MANIFEST_JSON), {span}, MANIFEST_DIGEST);\n")
    out.append("}\n")
    out.append("\n")
    out.append(f"}}  // namespace {namespace}\n")
    return "".join(out)


def emit(
    manifest: Any,
    path: "str | os.PathLike[str]",
    *,
    ops: "Iterable[Any]" = (),
    embed: bool = False,
    pin: "str | None" = None,
    namespace: str = "tilefish_launchers",
) -> pathlib.Path:
    """Write the typed C++ launchers of ``manifest`` to ``path`` (one header) and return the path.

    ``manifest`` is a manifest directory, which is loaded only with ``pin=`` (its digest), or a pinned :class:`~tilefish.Manifest`, as for
    :func:`tilefish.hosts.rust.emit`. With ``embed=True`` the companion source ``<path stem>_embedded.cpp`` is written next to the header
    (:func:`render_embedded`). Both files are rendered before either is written, through a temporary file and a rename; a symbolic link at
    either path is refused.
    """
    path = pathlib.Path(path)
    ops = list(ops)
    source = render(manifest, ops=ops, embed=embed, pin=pin, namespace=namespace)
    companion = render_embedded(manifest, pin=pin, namespace=namespace, header_name=path.name) if embed else None
    path.parent.mkdir(parents=True, exist_ok=True)
    write_file(path, source.encode("utf-8"))
    if companion is not None:
        write_file(path.with_name(f"{path.stem}_embedded.cpp"), companion.encode("utf-8"))
    return path


# -- launch plans ---------------------------------------------------------------------------------------------------------------------------------

# the names a plan runner's class must not take: its nested names and its members
_RESERVED_PLAN_CLASSES = frozenset({"Externals", "Scalars", "GRAPH", "ARCH", "PLAN_DIGEST", "ARENA_BYTES"})


def render_plan(plan: Any, *, pin: "str | None" = None, namespace: str = "tilefish_plans") -> str:
    """The C++ header :func:`emit_plan` writes, as a string. ``plan`` is a plan directory with ``pin``, or a pinned :class:`~tilefish._plan.Plan`;
    every field of it that reaches the output is validated first."""
    namespace = _namespace(namespace)
    with _cpp_errors():
        doc = rust._trusted_plan(plan, pin)
        who = f"the plan of graph {doc.get('graph')!r}"
        graph = rust._name(who, "graph", doc["graph"])
        digest = rust._name(who, "plan_digest", doc["plan_digest"], _DIGEST)
        manifest_digest_ = rust._name(who, "manifest_digest", doc["manifest_digest"], _DIGEST)
        arch = rust._name(who, "arch", doc["arch"], _ARCH)
        arena = _count(who, "arena_bytes", doc["arena_bytes"])
        externals = [b for b in doc["buffers"] if b["role"] in ("input", "weight", "inout", "output")]
        scalars = [b for b in doc["buffers"] if b["role"] == "devscalar"]
        members: "dict[str, str]" = {}
        for b in externals + scalars:
            name = _cpp_ident(rust._name(who, "a buffer name", b["name"]), who)
            if name in members.values():
                rust._bad(who, f"two buffers have the member name {name!r} in C++")
            if b["dtype"] not in CPP_ELEMENTS:
                rust._bad(who, f"buffer {b['name']!r} has dtype {b['dtype']!r}, which has no C++ element type")
            _shape(who, f"the shape of buffer {b['name']!r}", b["shape"])
            members[b["name"]] = name
        dims = []
        for d, values in doc["dims"].items():
            rust._name(who, "a dimension", d)
            dims.append(f"{d} {[_count(who, f'a bucket of {d}', v, 1) for v in values]}")
    cls = rust._camel(graph)
    cls = cls + "Plan" if cls in _RESERVED_PLAN_CLASSES or cls in rust._RESERVED_PLAN_STRUCTS else cls
    ext, sca = f"{cls}Externals", f"{cls}Scalars"
    roles = {"input": "an input (read)", "weight": "a weight (read)", "inout": "updated in place", "output": "a returned value (written)"}
    outputs = list(doc["outputs"])
    guard = _guard("PLAN", namespace, graph, digest[:16])
    out = [
        "// @generated by tilefish.hosts.cpp.emit_plan. Do not edit: generate it again from the plan.\n",
        f"// The plan of graph `{graph}` it was generated from has digest {digest}; its manifest has digest {manifest_digest_}.\n",
        f"#ifndef {guard}\n",
        f"#define {guard}\n",
        "\n",
        "#include <cstdint>\n",
        "#include <string>\n",
        "#include <utility>\n",
        "\n",
        '#include "tilefish.hpp"\n',
        "\n",
        f"namespace {namespace} {{\n",
        "\n",
    ]
    lines = [f"The tensors of a call of the plan of graph `{graph}`, by name, with their element types (the host's: the plan allocates none), in this"]
    lines.append("order (an aggregate). An input or a weight is a `TensorRef`, what the plan writes a `TensorMut`; every one is checked against the plan")
    lines.append("at a call. A captured plan replays on the addresses it captured: the host writes new contents into its memory between replays.")
    out.append(_doc_comment(lines))
    out.append(f"struct {ext} {{\n")
    for b in externals:
        what = roles[b["role"]] + (f", returned value {outputs.index(b['name'])}" if b["role"] == "output" else "")
        out.append(_doc_comment([f"`{b['name']}`: {what}, {b['dtype']} `[{_shape_text(b['shape'])}]`."], "    "))
        kind = "TensorMut" if b["role"] in ("inout", "output") else "TensorRef"
        out.append(f"    tilefish::{kind}<{CPP_ELEMENTS[b['dtype']]}> {members[b['name']]};\n")
    out.append("};\n\n")
    out.append(_doc_comment([f"The device scalars of a call of the plan of graph `{graph}`: written into their slots of the arena before the first launch."]))
    out.append(f"struct {sca} {{\n")
    for b in scalars:
        ty = _DEVSCALAR_TYPES.get(b["dtype"], "uint64_t")
        note = "" if b["dtype"] in _DEVSCALAR_TYPES else " (its bits, in the low bytes)"
        out.append(_doc_comment([f"`{b['name']}`: DevScalar[{b['dtype']}]{note}."], "    "))
        out.append(f"    {ty} {members[b['name']]} = 0;\n")
    out.append("};\n\n")
    doc_lines = [
        f"The launch plan of graph `{graph}` for {arch}, instantiated: every bucket's binaries loaded and its arena allocated, once.",
        "",
        f"The dimensions and their buckets: {', '.join(dims) or 'none (one bucket)'}. A call selects the bucket from the shapes of its tensors,",
        "which must be one exactly (there is no padding).",
    ]
    out.append(_doc_comment(doc_lines))
    out.append(f"class {cls} {{\npublic:\n")
    out.append(f"    using Externals = {ext};\n")
    out.append(f"    using Scalars = {sca};\n\n")
    out.append("    /// The name of the graph.\n")
    out.append(f"    static constexpr const char* GRAPH = {_cpp_str(graph)};\n")
    out.append("    /// The architecture the plan is for.\n")
    out.append(f"    static constexpr const char* ARCH = {_cpp_str(arch)};\n")
    out.append("    /// The digest of the plan this class was generated from: `load` refuses any other plan (and so any other manifest).\n")
    out.append(f"    static constexpr const char* PLAN_DIGEST = {_cpp_str(digest)};\n")
    out.append("    /// The bytes of the arena.\n")
    out.append(f"    static constexpr uint64_t ARENA_BYTES = {arena};\n\n")
    out.append("    /// Load the plan directory (its `plan.json`, pinned to `PLAN_DIGEST`, and the manifest next to it) and instantiate it for every\n")
    out.append("    /// bucket in `ctx`, the arena allocated on `stream` (which stays valid until the arena is freed with the last handle that holds it).\n")
    out.append(f"    static {cls} load(const std::string& directory, const tilefish::Context& ctx, CUstream stream) {{\n")
    out.append(f"        return {cls}(tilefish::LaunchPlan::load(directory, PLAN_DIGEST).instantiate(ctx, stream));\n")
    out.append("    }\n\n")
    out.append("    /// The instantiated plan (untyped).\n")
    out.append("    const tilefish::PlanRun& run() const { return run_; }\n\n")
    params = [
        "CUstream stream",
        f"const {ext}& e" if externals else f"const {ext}&",
        f"const {sca}& s = {{}}" if scalars else f"const {sca}& = {{}}",
    ]
    named = [f"tilefish::Named{{{_cpp_str(b['name'])}, e.{members[b['name']]}}}" for b in externals]
    values = []
    for b in scalars:
        value = f"s.{members[b['name']]}"
        if b["dtype"] == "f32":
            values.append(f"tilefish::DevScalar({_cpp_str(b['name'])}, {value})")
        elif b["dtype"] in _DEVSCALAR_TYPES:
            values.append(f"tilefish::DevScalar({_cpp_str(b['name'])}, static_cast<int64_t>({value}))")
        else:
            values.append(f"tilefish::DevScalar::bits({_cpp_str(b['name'])}, {value})")
    call = (
        "(\n            stream,\n"
        + (_braced(named, "            ") if named else "            tilefish::detail::Span<tilefish::Named>()")
        + ",\n"
        + (_braced(values, "            ") if values else "            tilefish::detail::Span<tilefish::DevScalar>()")
        + ");\n"
    )
    out.append("    /// Run the plan on `stream`: every step is checked before the first launch, and nothing is allocated.\n")
    out.append(f"    void {_signature('launch', params)} const {{\n")
    out.append("        run_.launch" + call)
    out.append("    }\n\n")
    out.append("    /// Capture the plan in a CUDA graph on `stream` for the bucket the tensors select: the externals' memory stays allocated while the\n")
    out.append("    /// captured plan lives, and `s` holds the values its first replay reads.\n")
    out.append(f"    tilefish::CapturedPlan {_signature('capture', params)} const {{\n")
    out.append("        return run_.capture" + call)
    out.append("    }\n\n")
    out.append("private:\n")
    out.append(f"    explicit {cls}(tilefish::PlanRun loaded) : run_(std::move(loaded)) {{}}\n")
    out.append("    tilefish::PlanRun run_;\n};\n\n")
    out.append(f"}}  // namespace {namespace}\n")
    out.append("\n")
    out.append(f"#endif  // {guard}\n")
    return "".join(out)


def emit_plan(plan: Any, path: "str | os.PathLike[str]", *, pin: "str | None" = None, namespace: str = "tilefish_plans") -> pathlib.Path:
    """Write a typed C++ runner of a launch plan (``plan.json``) to ``path`` (one header) and return the path.

    The header has a class named after the graph (``block`` gives ``Block``) with ``PLAN_DIGEST``; its ``load(directory, ctx, stream)`` loads the
    plan pinned to that digest (and with it the manifest) and instantiates every bucket, and its ``launch`` and ``capture`` take a
    ``<Graph>Externals`` (one member per external, named after it, with its element type) and a ``<Graph>Scalars`` (one member per device
    scalar, in its host type). ``plan`` is a plan directory with ``pin=``, or a pinned :class:`~tilefish._plan.Plan`; every field that reaches
    the output is validated. The file is written through a temporary file and renamed.
    """
    path = pathlib.Path(path)
    source = render_plan(plan, pin=pin, namespace=namespace)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_file(path, source.encode("utf-8"))
    return path


__all__ = ["CPP_ELEMENTS", "emit", "emit_plan", "render", "render_embedded", "render_plan"]
