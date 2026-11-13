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

Nothing in this module imports Triton.
"""

from __future__ import annotations

import hashlib
import tokenize
from typing import Any

from .errors import TilefishBuildError

_MAX_UNWRAP = 8


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
