"""What a hand-written Triton kernel is built from, as files: the source digest of an ``extern``.

A generated kernel carries its source in its compile spec (``source_sha256``). A hand-written ``@tilefish.extern``
names a Triton function by ``module:function`` and its declaration says nothing about the body, so the manifest
entry of its binary records a *source digest* and a replay recomputes it. Triton's own ``JITFunction.cache_key`` is
not that digest: it covers the text of the function and of the jit functions it calls and the ``tl.constexpr``
globals it reads *directly*, so a constexpr read as a module attribute (``cfg.SCALE``), a dtype alias
(``DT = tl.float32``) or a constant in a helper two modules away changes the binary and not the key.

:func:`source_digest` is a sha256 over

* the full text of the file of the kernel function's module;
* recursively, the full text of the file of every *user module* reachable from the namespace of that module: the
  modules it holds (``import cfg``), the modules of the functions, classes and objects it holds (``from helpers
  import apply``, a constant that is an instance of a class from ``cfg``), then the same for each of those modules.
  Only the top level of a namespace is followed: a name that is a container of functions is not opened, and an import
  inside a function body is not seen;
* the Triton version.

Each file counts by its module name and the sha256 of its bytes, so the digest does not depend on where the files are.

A module is a *user module* unless its top-level package is ``triton``, ``torch`` or ``tilefish``, or it is part of the
standard library (``sys.stdlib_module_names``); a module that is not a Python source file (a built-in or extension
module, a namespace package, a bytecode-only file) has no text and contributes nothing. Packages installed from
elsewhere (numpy, say) are user modules like any other and are read like any other, so a kernel module that imports
one makes the digest depend on all that it reaches.

What follows from this: any edit to a reachable file, including one that cannot change the binary (a comment), makes
the digest differ, and the binary has to be exported again and its pin (``Manifest.digest``) replaced. The digest is of the files as they
are on disk when it is computed, so a process that edited a file after importing it must reload before it builds (a build
checks the kernel's own functions and refuses when the lines the process imported are not the ones in the file); it does
not see a value a program assigned to a module attribute at run time, nor a file that is not read from disk.

Nothing in this module imports Triton except to read its version.
"""

from __future__ import annotations

import hashlib
import sys
import tokenize
import types
from typing import Any, Iterator

from .errors import TilefishBuildError

SOURCE_DIGEST_KIND = "tilefish.module-sources.v1"
_TAG = b"tilefish.source-digest.v1\0"
_EXCLUDED = frozenset({"triton", "torch", "tilefish", "builtins", "__future__"}) | frozenset(getattr(sys, "stdlib_module_names", ()))
_MAX_UNWRAP = 8


def _attr(obj: Any, name: str) -> Any:
    """``getattr(obj, name, None)`` that also survives an attribute that raises something other than ``AttributeError``."""
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001 - an arbitrary object's attribute may raise anything
        return None


def _is_jit(value: Any) -> bool:
    return _attr(value, "raw_src") is not None and isinstance(_attr(value, "__globals__"), dict)


def is_user_module(name: str) -> bool:
    """Whether the files of module ``name`` count towards a source digest (see the module docstring for the rule)."""
    return name.partition(".")[0] not in _EXCLUDED


def jit_function(fn: Any) -> Any:
    """The ``triton.jit`` function behind ``fn`` (an autotuner or a heuristics wrapper holds it as ``.fn``), or ``None``."""
    for _ in range(_MAX_UNWRAP):
        if _is_jit(fn):
            return fn
        fn = _attr(fn, "fn")
        if fn is None:
            return None
    return None


def _namespace_of(value: Any) -> "tuple[str, dict[str, Any]] | None":
    """The module (name and namespace) a value belongs to: itself for a module, where it was defined for the rest."""
    if issubclass(type(value), types.ModuleType):  # not isinstance: that asks the object for its __class__
        return value.__name__, value.__dict__
    namespace = _attr(value, "__globals__")  # a function, a jit function
    if isinstance(namespace, dict) and isinstance(namespace.get("__name__"), str):
        return namespace["__name__"], namespace
    name = _attr(value, "__module__")  # a class, or an instance of one
    if not isinstance(name, str):
        name = _attr(type(value), "__module__")
    module = sys.modules.get(name) if isinstance(name, str) else None
    if module is None:
        return None
    return module.__name__, module.__dict__


def _held(value: Any) -> "Iterator[Any]":
    """``value`` and the function it wraps, if it wraps one (the ``.fn`` of a jit function or an autotuner)."""
    yield value
    if issubclass(type(value), types.ModuleType):  # not isinstance: that asks the object for its __class__
        return
    inner = _attr(value, "fn")
    if inner is not None and inner is not value:
        yield inner


def _source_file(namespace: "dict[str, Any]") -> "str | None":
    path = namespace.get("__file__")
    return path if isinstance(path, str) and path.endswith(".py") else None


class Sources:
    """The user modules a kernel reaches: ``files`` (module name to path) and ``functions`` (the jit functions among them)."""

    def __init__(self, files: "dict[str, str]", functions: "list[Any]") -> None:
        self.files = files
        self.functions = functions


def reachable(fn: Any) -> "Sources | None":
    """The files that make up the kernel ``fn`` (a ``JITFunction``), or ``None`` when its own module is not a Python file."""
    kernel = jit_function(fn)
    if kernel is None:
        return None
    start = _namespace_of(kernel)
    if start is None or _source_file(start[1]) is None:
        return None
    files: dict[str, str] = {}
    functions: list[Any] = [kernel]
    pending = [start]
    while pending:
        name, namespace = pending.pop()
        if name in files or not is_user_module(name):
            continue
        path = _source_file(namespace)
        if path is None:
            continue
        files[name] = path
        for value in list(namespace.values()):
            for held in _held(value):
                target = _namespace_of(held)
                if target is None or not is_user_module(target[0]):
                    continue
                if _is_jit(held) and not any(held is known for known in functions):
                    functions.append(held)
                if target[0] not in files:
                    pending.append(target)
    return Sources(files, functions)


def file_sha256(path: str) -> str:
    """The sha256 of the bytes of a file (``OSError`` when it cannot be read)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def triton_version() -> "str | None":
    try:
        import triton
    except ImportError:
        return None
    return str(getattr(triton, "__version__", ""))


def source_digest(fn: Any) -> "str | None":
    """The source digest of the Triton function ``fn`` (see the module docstring), or ``None`` when it cannot be computed.

    That is when Triton cannot be imported, when the function's module is not a Python source file, or when a file it
    reaches cannot be read.
    """
    return digest_of(reachable(fn))


def digest_of(found: "Sources | None") -> "str | None":
    """The source digest of what :func:`reachable` found (``None`` for ``None``, or when Triton or a file is not available)."""
    version = triton_version()
    if version is None or found is None:
        return None
    h = hashlib.sha256(_TAG)
    h.update(version.encode("utf-8") + b"\0")
    try:
        for name in sorted(found.files):
            h.update(name.encode("utf-8") + b"\0" + file_sha256(found.files[name]).encode("ascii") + b"\n")
    except OSError:
        return None
    return h.hexdigest()


def function_file(fn: Any) -> "str | None":
    """The path of the Python file the function ``fn`` is defined in, or ``None``."""
    kernel = jit_function(fn)
    if kernel is None:
        return None
    namespace = _namespace_of(kernel)
    return None if namespace is None else _source_file(namespace[1])


def check_imported_lines(functions: "list[Any]") -> None:
    """Raise :class:`~tilefish.errors.TilefishBuildError` for a jit function whose file no longer has the lines it was imported from.

    A jit function keeps the lines of its source as they were when its module was imported (``raw_src``, from line
    ``starting_line_number``). If the file has other lines there now, the process holds an older version than the one
    on disk, and a digest of the files would describe something other than what is compiled.
    """
    for fn in functions:
        path = _source_file(fn.__globals__)
        if path is None:
            continue
        try:
            with tokenize.open(path) as f:
                lines = f.readlines()
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            raise TilefishBuildError(f"cannot read {path!r}, the file of the Triton function '{fn.__name__}': {e}") from e
        first = int(fn.starting_line_number) - 1
        if lines[first : first + len(fn.raw_src)] != list(fn.raw_src):
            raise TilefishBuildError(
                f"the Triton function '{fn.__name__}' was imported from an older version of {path!r}: its lines "
                f"{first + 1}-{first + len(fn.raw_src)} are not what the file has now. Restart the process, or reload the "
                "module (and every module it imports that changed), before building"
            )
