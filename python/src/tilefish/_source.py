"""What a Triton function is, and the one file check that is left: a lowered kernel's generated source.

The identity of a binary is the IR Triton compiles (:mod:`tilefish._ir`), for a hand-written ``extern`` and a lowered kernel alike.
This module used to compute that identity *statically*, by digesting the source files a kernel reaches, and by watching the
imports, snapshotting module attributes and walking closures to find the Python state Triton would read at compile time. Every
round of review found a route that walk did not follow (``getattr(cfg, "SC" + "ALE")``, a class attribute, a
``triton.constexpr_function`` that reads the environment, a ``tl.constexpr`` of an object whose ``repr`` hides its state, a user
module named like a standard library module), and each was shown to hand out a stale binary through the build cache and through
export and replay. All of it is gone: the file digest, the import watch (the ``SourceFileLoader.exec_module`` wrapper and
``TILEFISH_IMPORT_WATCH``), the snapshot of module attributes and its check, and the walk over values and closures.

What is left is not a freshness check of an extern, which the IR digest makes. It is what a *lowered* kernel's spec says about
itself: ``source_sha256``, the sha256 of the Triton source ``lower()`` generated, which a manifest entry records and a replay compares
without Triton. ``build`` refuses a function whose module file is not exactly that source (:func:`file_sha256`), and one whose lines
the process imported are not the ones in that file now (:func:`check_imported_lines`): the IR digest is of what the process holds,
and says nothing about whether that is the source the spec claims, so without this check a process that holds an older version of
the generated file would stamp the new source's sha256 on a binary of the old one.

It also says how a binary's debug information names a source file (:func:`source_name`): by a path that does not depend on the
host, so that two builds of one kernel on two machines, from two checkouts or two working directories, are the same bytes.

Nothing in this module imports Triton.
"""

from __future__ import annotations

import hashlib
import os
import tokenize
from typing import Any

from .errors import TilefishBuildError

_MAX_UNWRAP = 8
_SOURCE_NAMES: "dict[str, str]" = {}  # the path of a file tilefish wrote a generated source to -> the name its debug information gives it
DEBUG_PATHS_ENV = "TILEFISH_DEBUG_PATHS"  # "absolute": a build keeps the absolute source paths in a binary's debug information (debug_paths)


def _attr(obj: Any, name: str) -> Any:
    """``getattr(obj, name, None)`` that also survives an attribute that raises something other than ``AttributeError``."""
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001 - an arbitrary object's attribute may raise anything
        return None


def _is_jit(value: Any) -> bool:
    return _attr(value, "raw_src") is not None and isinstance(_attr(value, "__globals__"), dict)


def jit_function(fn: Any) -> Any:
    """The ``triton.jit`` function behind ``fn`` (an autotuner or a heuristics wrapper holds it as ``.fn``), or ``None``."""
    for _ in range(_MAX_UNWRAP):
        if _is_jit(fn):
            return fn
        fn = _attr(fn, "fn")
        if fn is None:
            return None
    return None


def defined_in(fn: Any) -> "tuple[str, str]":
    """``(module name, function name)`` of the jit function ``fn`` as it was defined (``"?"`` where it cannot say)."""
    namespace = _attr(fn, "__globals__")
    module = namespace.get("__name__") if isinstance(namespace, dict) else None
    return (module if isinstance(module, str) else str(_attr(fn, "__module__") or "?"), str(_attr(fn, "__name__") or "?"))


def function_file(fn: Any) -> "str | None":
    """The path of the Python file the function ``fn`` is defined in, or ``None``."""
    kernel = jit_function(fn)
    if kernel is None:
        return None
    path = kernel.__globals__.get("__file__")
    return path if isinstance(path, str) and path.endswith(".py") else None


def file_sha256(path: str) -> str:
    """The sha256 of the bytes of a file (``OSError`` when it cannot be read)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_imported_lines(fn: Any) -> None:
    """Raise :class:`~tilefish.errors.TilefishBuildError` for a jit function whose file no longer has the lines it was imported from.

    A jit function keeps the lines of its source as they were when its module was imported (``raw_src``, from line
    ``starting_line_number``). If the file has other lines there now, the process holds an older version than the one on disk.
    """
    path = function_file(fn)
    if path is None:
        return
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
            "module, before building"
        )


def register_source_name(path: str, name: str) -> None:
    """Have the debug information of a binary name the file at ``path`` (a generated source tilefish wrote) ``name`` (its path relative to a source
    root, ``tilefish_generated/rmsnorm.py``: :attr:`tilefish.lower.LoweredKernel.filename`)."""
    _SOURCE_NAMES[path] = name


def source_name(path: str, module: "str | None") -> str:
    """The name a binary's debug information gives the source file ``path`` of a function defined in the module ``module``: a path that does not
    depend on the host.

    Triton records the path of a kernel's file (and of the file of every jit function it calls, Triton's own ``triton/language/standard.py``
    among them) in the locations of its IR, and ``ptxas`` writes it into the cubin's ``.debug_line`` and ``.nv_debug_ptx_txt``. A path that
    names the checkout, the virtual environment, the home directory or a temporary directory makes two builds of the same kernel differ byte
    for byte. The name used instead is

    * for a generated source tilefish wrote (:func:`register_source_name`), its path relative to its source root, ``tilefish_generated/<name>.py``,
      what :meth:`tilefish.lower.LoweredKernel.write` writes and imports as ``tilefish_generated.<name>``;
    * for a module ``a.b.c`` whose file is ``.../a/b/c.py`` (or ``.../a/b/c/__init__.py``), that tail, ``a/b/c.py``: the path relative to the
      directory on ``sys.path`` it was imported from;
    * for any other file (a script run as ``__main__``, a module imported under another name), its base name.

    A name that is not a path (``<string>``) is kept. The name is a label for tools (a profiler resolves it against a source directory you give
    it); it is not in the machine code, so it changes ``sha256`` and not ``sha256_code``.
    """
    if not path or (path.startswith("<") and path.endswith(">")):
        return path
    registered = _SOURCE_NAMES.get(path)
    if registered is not None:
        return registered
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    if not parts:
        return path
    if module and module != "__main__":
        names = module.split(".")
        for tail in (names[:-1] + [names[-1] + ".py"], names + ["__init__.py"]):
            if parts[-len(tail) :] == tail:
                return "/".join(tail)
    return parts[-1]


def module_of(fn: Any) -> "str | None":
    """The name of the module the Python function behind ``fn`` (a jit function, or what wraps one as ``.fn``) is defined in, or ``None``."""
    for _ in range(_MAX_UNWRAP):
        if _attr(fn, "__code__") is not None:
            namespace = _attr(fn, "__globals__")
            name = namespace.get("__name__") if isinstance(namespace, dict) else None
            if not isinstance(name, str):
                name = _attr(fn, "__module__")
            return name if isinstance(name, str) else None
        fn = _attr(fn, "fn")
        if fn is None:
            return None
    return None


def debug_paths() -> str:
    """How a build names the source files in a binary's debug information: ``"module"`` (the default: :func:`source_name`, so that builds are the
    same bytes on every host) or ``"absolute"``, when the environment variable ``TILEFISH_DEBUG_PATHS`` is ``absolute``: the paths Triton records,
    as they are on this host, which a profiler (Nsight Compute's source view, ``cuobjdump``, ``nvdisasm``) opens without being given a source
    directory.

    **That opt-out breaks reproducibility**: the cubin then names the checkout, the virtual environment and the cache directory, so an export
    from another directory or machine has other bytes, another ``sha256`` and another manifest and plan digest (the machine code,
    ``sha256_code``, is the same). A binary built with it is cached apart (the build cache key has the setting), and is never served to a build
    without it. Any other value is refused."""
    value = os.environ.get(DEBUG_PATHS_ENV, "")
    if value == "":
        return "module"
    if value == "absolute":
        return "absolute"
    raise TilefishBuildError(
        f"{DEBUG_PATHS_ENV}={value!r}: set it to 'absolute' to keep the absolute source paths in a binary's debug information (for a profiler; "
        "exports are then not reproducible across directories and machines), or leave it unset"
    )
