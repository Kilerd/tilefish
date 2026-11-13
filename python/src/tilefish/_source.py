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
  Only the top level of a namespace is followed (and the closure variables of the jit functions found): a name that is a
  container of functions is not opened, and an import inside a function body is not seen;
* the Triton version;
* the *values* the kernel's jit functions read that are in no file (:func:`values_read`): the scalars and Triton constants of
  the module attributes they read, as the process holds them (a module may compute one when it is imported, from the
  environment or a device query), and the **closure variables** of every jit function reached (a kernel made by a factory
  function, ``def make(): S = tl.constexpr(...); @triton.jit def k(...): ... S``, reads ``S`` from a cell, which no file has).
  Each is written as its ``repr``, so the digest means *what this process would compile*: a replay in a process whose cell
  or constant has another value is refused, and a cache of builds keyed by the digest cannot serve one value's binary for
  another's. A closure variable that holds a module, a function or a class stands for which one it is; one that holds anything
  else (a list, an instance, an object whose ``repr`` is its address), and a module attribute that a kernel reads and that
  is such an object, has no text that is the same in every process: **it is refused** (:class:`UnstableValueError`: make it a
  module-level constant or a const parameter of the extern).

Each file counts by its module name and the sha256 of its bytes, so the digest does not depend on where the files are.

A module is a *user module* unless its top-level package is ``triton``, ``torch`` or ``tilefish``, or it is part of the
standard library (``sys.stdlib_module_names``); a module that is not a Python source file (a built-in or extension
module, a namespace package, a bytecode-only file) has no text and contributes nothing. Packages installed from
elsewhere (numpy, say) are user modules like any other and are read like any other, so a kernel module that imports
one makes the digest depend on all that it reaches.

What follows from this: any edit to a reachable file, including one that cannot change the binary (a comment), makes
the digest differ, and the binary has to be exported again and its pin (``Manifest.digest``) replaced.

Memory against disk. The digest is of the files as they are on disk, and Triton compiles what the process holds in memory.
They are the same only if no reachable file was edited after the process imported it, and nothing assigned to a module
attribute that a kernel reads. So no digest is computed from files the process may hold in another version: *every*
:func:`digest_of` (and with it every build, ``jit``, ``export`` and ``replay`` of a hand-written kernel) first checks

* **the files.** Every user module the kernel reaches must be the file the process imported. ``import tilefish`` installs a
  small import watch (:func:`install_import_watch`): for each user module that a ``SourceFileLoader`` then executes (an
  ``import``, an ``importlib.reload``, ``spec_from_file_location``) it records the sha256 of the file just before it runs. The
  check hashes the file again and refuses when it differs: "module X changed on disk since it was imported; reload it or
  restart". This is exact for any edit, also one that keeps the size, the modification time, or lands in the same second as the
  import, and a file that is edited and then restored passes. A module that was already imported when tilefish was imported has no
  record; for it the size and modification time that the file had at that moment are compared (a ``touch`` counts as a change
  there). The ``.pyc`` header is not used: it holds whole seconds and a size, it is rewritten by any process that imports the
  module, and a stale one is simply ignored when bytecode is not written, so it would give false alarms and false assurance;
* **the values.** A scalar (``int``, ``float``, ``bool``, ``str``, ``bytes``, ``None``, a tuple of them) or a Triton object
  (``tl.constexpr``, a dtype) that a module held when it finished importing, and that a jit function of the kernel reads
  by name or as ``module.name``, must still be the same: a kernel's inputs come from source files, not from a module attribute
  that a program assigned later. Only a module recorded by the watch has a snapshot; a container mutated in place, an attribute of an
  instance, and a value copied out of its module before it was changed (``from cfg import SCALE``) are not seen.

What it does not see, besides the above: a module that was already imported, and edited, before tilefish was imported; a module that no
``SourceFileLoader`` executed (a zip import, a custom loader, pytest's assertion rewriting, a notebook cell, ``exec``), which
has no file or no record; a source that Python took from an ``unchecked-hash`` or stale same-second ``.pyc`` rather than from the
file (the record is of the file); and a file that is not read from disk.

Nothing in this module imports Triton except to read its version.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib.machinery
import json
import os
import sys
import textwrap
import tokenize
import types
from dataclasses import dataclass, field
from typing import Any, Iterator

from .errors import TilefishBuildError

SOURCE_DIGEST_KIND = "tilefish.module-sources.v2"
_TAG = b"tilefish.source-digest.v2\0"
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


def _closure_of(jit: Any) -> "dict[str, Any]":
    """The closure variables of the function behind a jit function, by name (``_EMPTY`` for a cell that has no value yet)."""
    inner = _attr(jit, "fn")
    names = _attr(_attr(inner, "__code__"), "co_freevars")
    cells = _attr(inner, "__closure__")
    if not names or not cells:
        return {}
    out: dict[str, Any] = {}
    for name, cell in zip(names, cells):
        try:
            out[name] = cell.cell_contents
        except ValueError:
            out[name] = _EMPTY
    return out


def reachable(fn: Any) -> "Sources | None":
    """The files that make up the kernel ``fn`` (a ``JITFunction``), or ``None`` when its own module is not a Python file.

    The walk goes through the top level of the module namespaces and through the closure variables of every jit function it
    finds (a kernel made by a factory function holds what it reads in cells, not in its module).
    """
    kernel = jit_function(fn)
    if kernel is None:
        return None
    start = _namespace_of(kernel)
    if start is None or _source_file(start[1]) is None:
        return None
    files: dict[str, str] = {}
    functions: list[Any] = [kernel]
    pending = [start]

    def consider(value: Any) -> None:
        for held in _held(value):
            target = _namespace_of(held)
            if target is None or not is_user_module(target[0]):
                continue
            if _is_jit(held) and not any(held is known for known in functions):
                functions.append(held)
            if target[0] not in files:
                pending.append(target)

    closed = 0  # the jit functions whose closure variables were looked at
    while True:
        while pending:
            name, namespace = pending.pop()
            if name in files or not is_user_module(name):
                continue
            path = _source_file(namespace)
            if path is None:
                continue
            files[name] = path
            for value in list(namespace.values()):
                consider(value)
        if closed == len(functions):
            break
        for value in _closure_of(functions[closed]).values():
            consider(value)
        closed += 1
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
    """The source digest of what :func:`reachable` found (``None`` for ``None``, or when Triton or a file is not available).

    The files are checked against the process first (:func:`check_current`), and the digest is made of the very bytes that
    were checked: no digest exists of a file the process holds in another version. Raises
    :class:`StaleSourceError` (a :class:`~tilefish.errors.TilefishBuildError`) when one is.
    """
    version = triton_version()
    if version is None or found is None:
        return None
    try:
        hashes = _check_files(found)
    except OSError:
        return None
    check_values(found)
    values = values_read(found, strict=True)  # a value with no stable repr is refused: no digest could tell two of them apart
    h = hashlib.sha256(_TAG)
    h.update(version.encode("utf-8") + b"\0")
    for name in sorted(found.files):
        h.update(name.encode("utf-8") + b"\0" + hashes[name].encode("ascii") + b"\n")
    h.update(b"values\0" + json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n")
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


# -- memory against disk: what the process imported, and what a kernel reads from it -----------------------------------------------


class StaleSourceError(TilefishBuildError):
    """What the process holds of a module a kernel reaches is not what is on disk, or not what the module's source says."""


class UnstableValueError(TilefishBuildError):
    """A value a kernel reads (a closure variable, a module constant) that has no stable ``repr``, so no digest can say which value it is."""


_NoneType = type(None)
_SCALARS = (int, float, bool, str, bytes, complex, _NoneType)
_MISSING = object()
_EMPTY = object()  # a closure cell that has no value (yet)
_STRUCTURAL = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType, type)


def _is_scalar(value: Any) -> bool:
    if type(value) in _SCALARS:
        return True
    if type(value) in (tuple, frozenset):
        return all(_is_scalar(v) for v in value)
    return False


def _is_triton_object(value: Any) -> bool:
    """``tl.constexpr``, a ``tl.dtype``: what a kernel's module holds as its compile-time constants and dtype aliases."""
    return str(getattr(type(value), "__module__", "")).partition(".")[0] == "triton"


def _same(old: Any, new: Any) -> bool:
    if old is new:
        return True
    if type(old) is not type(new) or not _is_scalar(old):
        return False
    try:
        return bool(old == new) or bool(old != old and new != new)  # nan is the same nan
    except Exception:  # noqa: BLE001 - a scalar whose comparison fails
        return False


def _snapshot(namespace: "dict[str, Any]") -> "dict[str, Any]":
    """The names of a module namespace that hold a scalar or a Triton object, as they are (dunder names are skipped)."""
    out: dict[str, Any] = {}
    for name, value in list(namespace.items()):
        if name.startswith("__") and name.endswith("__"):
            continue
        try:
            if _is_scalar(value) or _is_triton_object(value):
                out[name] = value
        except Exception:  # noqa: BLE001 - an object whose type cannot be looked at
            continue
    return out


def _key(path: str) -> str:
    return os.path.abspath(path)


@dataclass
class _Imported:
    sha256: str  # of the file, just before a loader executed it
    values: "dict[str, Any]"  # _snapshot of the namespace when the module had finished importing


@dataclass
class _Watch:
    """What the process imported: the files, by path, and the modules that were there before the watch."""

    imported: "dict[str, _Imported]" = field(default_factory=dict)
    before: "dict[str, tuple[int, int]]" = field(default_factory=dict)  # path -> (mtime_ns, size) when the watch was installed
    original: Any = None
    loader: Any = None
    lazy: bool = False  # TILEFISH_IMPORT_WATCH=0: no loader is wrapped; a file is recorded when a build first reads it

    @property
    def installed(self) -> bool:
        return self.original is not None

    def _before(self, loader: Any, module: Any) -> "tuple[str, str] | None":
        try:
            path = loader.path
            if not isinstance(path, str) or not path.endswith(".py") or not is_user_module(str(module.__name__)):
                return None
            with open(path, "rb") as f:
                return _key(path), hashlib.sha256(f.read()).hexdigest()
        except Exception:  # noqa: BLE001 - the watch must never make an import fail
            return None

    def _after(self, token: "tuple[str, str] | None", module: Any) -> None:
        if token is None:
            return
        try:
            self.imported[token[0]] = _Imported(token[1], _snapshot(module.__dict__))
        except Exception:  # noqa: BLE001
            pass

    def install(self) -> None:
        """Wrap ``SourceFileLoader.exec_module`` (idempotent); modules already imported get their size and modification time recorded."""
        if self.installed:
            return
        loader = importlib.machinery.SourceFileLoader
        original = loader.exec_module
        watch = self

        @functools.wraps(original)
        def exec_module(self_: Any, module: Any) -> None:
            token = watch._before(self_, module)
            original(self_, module)
            watch._after(token, module)

        exec_module._tilefish_watch = watch  # type: ignore[attr-defined]
        self.before = {}
        self.record_baseline()
        self.original, self.loader = original, loader
        loader.exec_module = exec_module  # type: ignore[method-assign]

    def record_baseline(self) -> None:
        """Record the size and modification time of the file of every user module imported so far."""
        for name, module in list(sys.modules.items()):
            try:
                path = getattr(module, "__file__", None)
                if isinstance(path, str) and path.endswith(".py") and is_user_module(name):
                    st = os.stat(path)
                    self.before.setdefault(_key(path), (st.st_mtime_ns, st.st_size))
            except Exception:  # noqa: BLE001 - a module whose attributes or file cannot be looked at
                continue

    def uninstall(self) -> None:
        if not self.installed:
            return
        if self.loader.exec_module is not self.original:
            self.loader.exec_module = self.original  # type: ignore[method-assign]
        self.original = self.loader = None

    def check_file(self, name: str, path: str, sha256: str, st: "os.stat_result") -> None:
        """Raise :class:`StaleSourceError` when the file ``path`` of module ``name``, which has ``sha256`` and ``st`` now, is not what was imported."""
        key = _key(path)
        record = self.imported.get(key)
        if record is not None:
            if record.sha256 != sha256:
                raise StaleSourceError(
                    f"module '{name}' changed on disk since it was imported ({path!r}); reload it or restart the process, then "
                    "build again. The process holds the version it imported, so a digest of the file would describe a binary "
                    "that was not compiled from it (reload with importlib.reload, the modules that import names from it too)"
                )
            return
        before = self.before.get(key)
        if before is None:
            if self.lazy:
                self.before[key] = (st.st_mtime_ns, st.st_size)
            return
        if before != (st.st_mtime_ns, st.st_size):
            raise StaleSourceError(
                f"module '{name}' changed on disk since it was imported ({path!r}); reload it or restart the process, then build "
                "again. It was imported before tilefish, and its size or modification time is not what it was when tilefish was "
                "imported, so the process may hold an older version than the file (reload with importlib.reload, the modules "
                "that import names from it too)"
            )


def _adopt() -> _Watch:
    """The watch that is installed already (``tilefish._source`` was reloaded), or a new one."""
    existing = getattr(importlib.machinery.SourceFileLoader.exec_module, "_tilefish_watch", None)
    return existing if existing is not None else _Watch()


_watch = _adopt()


def install_import_watch() -> None:
    """Start recording the modules that are imported from now on (called when ``tilefish`` is imported; harmless to call again).

    ``TILEFISH_IMPORT_WATCH=0`` leaves the import system alone: no loader is wrapped. A module is then checked only by the size
    and modification time its file had when tilefish was imported or, for a module imported later, when a build first read it.
    That misses an edit made between such a module's import and the first build that reads it, and an edit that keeps both.
    """
    if os.environ.get("TILEFISH_IMPORT_WATCH", "").strip() == "0":
        _watch.lazy = True
        _watch.record_baseline()
        return
    _watch.install()


def _check_files(found: Sources) -> "dict[str, str]":
    """Check every file of ``found`` against what the process imported, and return the sha256 of each (``OSError``: a file cannot be read)."""
    hashes: dict[str, str] = {}
    for name in sorted(found.files):
        path = found.files[name]
        with open(path, "rb") as f:
            data = f.read()
            st = os.fstat(f.fileno())
        hashes[name] = hashlib.sha256(data).hexdigest()
        _watch.check_file(name, path, hashes[name], st)
    return hashes


def _names_read(fn: Any) -> "tuple[set[str], set[tuple[str, ...]]]":
    """The names a jit function reads that it does not bind itself, and the dotted chains (``cfg.SCALE``) it reads."""
    try:
        tree = ast.parse(textwrap.dedent("".join(fn.raw_src)))
    except (SyntaxError, ValueError, TypeError, AttributeError):
        return set(), set()
    bound: set[str] = set()
    loaded: set[str] = set()
    chains: set[tuple[str, ...]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name):
            (loaded if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, ast.Attribute):
            parts: list[str] = []
            inner: Any = node
            while isinstance(inner, ast.Attribute):
                parts.append(inner.attr)
                inner = inner.value
            if isinstance(inner, ast.Name):
                chains.add((inner.id, *reversed(parts)))
    return loaded - bound, {c for c in chains if c[0] not in bound}


def _check_value(owner: str, namespace: "dict[str, Any]", name: str, reader: Any) -> None:
    path = namespace.get("__file__")
    record = _watch.imported.get(_key(path)) if isinstance(path, str) else None
    if record is None or name not in record.values:
        return
    old, new = record.values[name], namespace.get(name, _MISSING)
    if not _same(old, new):
        now = "deleted" if new is _MISSING else repr(new)
        raise StaleSourceError(
            f"the module attribute '{owner}.{name}', which the Triton function '{reader.__name__}' reads, was assigned after "
            f"'{owner}' was imported (it was {old!r}, it is {now}). A kernel's inputs must come from its source files, not "
            "from module state changed at run time: the binary would be compiled from the new value while the source digest, "
            "and the cache and manifest entries made from it, describe the file. Put the value in the file (and reload the "
            "module), or make it a constexpr argument of the extern"
        )


def _reads(found: Sources) -> "Iterator[tuple[str, dict[str, Any], str, Any]]":
    """``(module name, its namespace, name, jit function)`` for every module-level name a jit function of ``found`` reads, by name or as ``module.name``.

    A name that is a closure variable of the function is not a module attribute: see :func:`values_read`.
    """
    for fn in found.functions:
        free, chains = _names_read(fn)
        namespace = _attr(fn, "__globals__")
        if not isinstance(namespace, dict):
            continue
        cells = _closure_of(fn)
        owner = str(namespace.get("__name__", "?"))
        for name in sorted(free):
            if name not in cells:
                yield owner, namespace, name, fn
        for chain in sorted(chains):
            module: Any = cells[chain[0]] if chain[0] in cells else namespace.get(chain[0])
            for attr in chain[1:]:
                if not issubclass(type(module), types.ModuleType):
                    break
                if is_user_module(module.__name__):
                    yield module.__name__, module.__dict__, attr, fn
                module = module.__dict__.get(attr)


def check_values(found: Sources) -> None:
    """Raise :class:`StaleSourceError` when a scalar or Triton constant a jit function of ``found`` reads has been reassigned since its module was imported."""
    for owner, namespace, name, fn in _reads(found):
        _check_value(owner, namespace, name, fn)


def _canonical(value: Any) -> str:
    """The ``repr`` of a scalar, the same in every process (a ``frozenset`` is written in sorted order)."""
    if type(value) is frozenset:
        return "frozenset({" + ", ".join(sorted(_canonical(v) for v in value)) + "})"
    if type(value) is tuple:
        return "(" + ", ".join(_canonical(v) for v in value) + ("," if len(value) == 1 else "") + ")"
    return repr(value)


def _function_id(fn: Any) -> str:
    """Which jit function: its module, its qualified name and its line, the same in every process that has the same files."""
    inner = _attr(fn, "fn")
    namespace = _attr(fn, "__globals__")
    module = str(namespace.get("__name__", "?")) if isinstance(namespace, dict) else "?"
    return f"{module}:{_attr(inner, '__qualname__') or _attr(fn, '__name__')}@{_attr(fn, 'starting_line_number')}"


def _identity(value: Any) -> str:
    """Which module, function or class a closure variable holds: its ``module:qualname`` (what the files say it is, not an address)."""
    if issubclass(type(value), types.ModuleType):
        return f"module:{value.__name__}"
    kernel = jit_function(value)
    if kernel is not None:
        return f"jit:{_function_id(kernel)}"
    return f"{_attr(value, '__module__')}:{_attr(value, '__qualname__') or _attr(value, '__name__')}"


def _render(value: Any, *, closure: bool, strict: bool, what: "Any") -> "str | None":
    """The text that stands for a value a kernel reads in a digest or a cache key, or ``None`` for one that is covered by the files.

    A scalar, and a Triton object with a stable ``repr`` (``tl.constexpr``, a dtype), stand for themselves; what a closure
    variable holds that is a module, a function or a class stands for which one it is. Any other value (a container, an
    instance, an object whose ``repr`` is its address) has no text that is the same in every process: with ``strict`` that is
    an :class:`UnstableValueError`, otherwise it is left out.
    """
    if value is _EMPTY:
        return "<empty cell>"
    try:
        if _is_scalar(value):
            return _canonical(value)
        if issubclass(type(value), _STRUCTURAL) or jit_function(value) is not None:
            return _identity(value) if closure else None
        if _is_triton_object(value):
            text = repr(value)
            if " at 0x" not in text:
                return text
            if not strict:
                return type(value).__qualname__
        elif not strict:
            return None
    except Exception:  # noqa: BLE001 - an object that cannot be looked at
        if not strict:
            return None
    kind = "closure variable" if closure else "module attribute"
    raise UnstableValueError(
        f"{what()} holds a {type(value).__qualname__} that has no stable repr, so its value cannot be part of the source digest "
        f"(a build, a cache entry and a manifest could not tell one value of it from another): make it a module-level constant "
        f"(a scalar or a tl.constexpr, in a file) or a const parameter of the extern, not a {kind} that holds an object"
    )


def values_read(found: "Sources | None", *, strict: bool = False) -> "dict[str, str]":
    """What the jit functions of ``found`` read that is not in a file, as the process holds it now: ``name`` to text.

    * the scalars and Triton constants of module attributes they read, by name or as ``module.name`` (``"cfg.SCALE"``);
    * the closure variables of every jit function (``"module:make.<locals>.kernel@12/S"``): a kernel made by a factory function
      reads its constants from cells, which are in no file.

    A source digest is of files, and what a module computes when it is imported (from the environment, from a device query), or a
    factory closes over, is not in them; these are the values a compile takes from the process, so the source digest (and a cache of builds) is of
    them too. With ``strict`` a value that has no stable ``repr`` raises :class:`UnstableValueError`; without it such a value is left out.
    """
    if found is None:
        return {}
    out: dict[str, str] = {}
    for owner, namespace, name, fn in _reads(found):
        value = namespace.get(name, _MISSING)
        if value is _MISSING:
            continue
        text = _render(
            value,
            closure=False,
            strict=strict,
            what=lambda: f"the module attribute '{owner}.{name}', which the Triton function '{_attr(fn, '__name__')}' reads,",
        )
        if text is not None:
            out[f"{owner}.{name}"] = text
    for fn in found.functions:
        for name, value in _closure_of(fn).items():
            text = _render(
                value,
                closure=True,
                strict=strict,
                what=lambda: f"the closure variable '{name}' of the Triton function '{_attr(fn, '__name__')}'",
            )
            if text is not None:
                out[f"{_function_id(fn)}/{name}"] = text
    return dict(sorted(out.items()))


def check_current(found: "Sources | None") -> None:
    """Raise :class:`StaleSourceError` when what the process holds of the modules of ``found`` is not what is on disk (see the module docstring)."""
    if found is None:
        return
    try:
        _check_files(found)
    except OSError as e:
        raise StaleSourceError(f"cannot read a file of the kernel's modules to compare it with what the process imported: {e}") from e
    check_values(found)
