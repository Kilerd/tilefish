"""The identity of a binary: the Triton IR its kernel compiles to.

A binary is the machine code Triton makes of a kernel. Which kernel that is depends on everything the Python process holds
when Triton's frontend runs: the text of the function and of the jit functions it calls, the module constants it reads
(``cfg.SCALE``, ``getattr(cfg, "SC" + "ALE")``, a class attribute ``cfg.Inner.SCALE``), closure variables, a
``triton.constexpr_function`` that looks at the environment, a ``tl.constexpr`` of an object whose ``repr`` hides its state,
a user module that is named like a standard library module. Each of these was once treated by looking for it (in the files, in
the namespaces, in the closures) and each of them had a case that the looking missed. No look is needed: Triton's frontend
folds every one of them into the IR it builds, so

    **the identity of a binary is the IR that Triton compiles** (:func:`ir_digest`).

For a build, an export, a ``jit`` run and a replay alike, the frontend is run on the spec's ``ASTSource`` (exactly the call
``triton.compile`` makes first: ``ASTSource.make_ir(target, options, codegen_fns, module_map, context)``, the TTIR module) and the
text of that module is hashed. Whatever the process holds in memory is what is hashed, because that is what is compiled.

The text is ``module.str_nodebug()``: the module printed without locations and without the names of the values. Where the kernel is
(the path of its file, its line numbers), comments and blank lines, and the names of local variables are not in it, so none of them
changes the digest; the operations, the types, the constants, the attributes of the arguments, the names of the functions, and the
bodies of the jit functions that are called, are. (Renaming a *function* changes the symbol of the binary and so the digest.)

What the digest does not cover is what the later stages (``ttir`` to ``ttgir``, ``llir``, ``ptx``, ``cubin``) take besides the IR:
the compile options, the target architecture, the toolchain (Triton and ``ptxas``) and the Triton knobs that change code
generation. Those are part of the key of the build cache (:mod:`tilefish._cache`), and the manifest records them next to the digest.

The digest is of the IR of one compile: the options of the spec (and any a caller added) and the target architecture are what the
frontend is given, as ``triton.compile`` is given them. Two Triton versions may print the same kernel differently, so a digest is
only compared with one made by the same Triton version (the manifest records which).

**A Gluon kernel** (``@tilefish.extern(gluon=...)``: a ``gluon.jit`` function of ``triton.experimental.gluon``) is compiled by Gluon's
frontend, ``GluonASTSource``, the call ``triton.compile`` makes for it: it builds TTGIR, the IR with the layouts written out, and sets the
module's ``ttg.num-warps``, ``ttg.target`` and ``ttg.threads-per-warp`` (Triton's frontend cannot build such a kernel: its layouts need
``ttg.num-warps``, which TTIR does not have yet). The identity of a Gluon binary is that TTGIR, printed the same way (``str_nodebug()``),
with its own tag and its own kind, ``"tilefish.gluon-ir.v1"`` (:data:`GLUON_IR_DIGEST_KIND`): it is the IR Gluon compiles, so a layout,
``num_warps`` and everything the TTIR of a Triton kernel has are in it, and a digest of one frontend's IR is never compared with, or
equal to, one of the other's. Which frontend is the declaration's (``CompileSpec.language``), never guessed from the function: the
function must then be a kernel of that frontend (``tilefish._build`` refuses another).

What this costs: the frontend, about 2 to 4 ms per kernel on a B300 (no compile, no GPU), once for each build, ``jit`` call or
replay call (a large kernel takes longer: an unrolled loop of 4096 steps, half a second). Nothing here compiles anything.

**Memory.** What is kept of a frontend run is its digest and the dots of its IR (:class:`Frontend`: what the TF32 guard reads), not the IR.
The text is hashed in pieces (:func:`ir_digest`) and dropped, and the module is freed (:func:`release_module`): Triton's Python binding of a
module does not own its operations, so a module that is only dropped stays allocated for the life of the process, about twice the size of
its text, at every run of the frontend and at every compile (measured with Triton 3.5.1). Within an export, or the ``prepare`` of a plan,
the frontend runs once per distinct input (:func:`frontend_scope`): the buckets of one ``by_bucket=`` configuration are one kernel with one
IR, and the scope is one point in time of the process, so its digest is computed once for all of them. Outside a scope (a ``jit`` call, a
replay call) every call runs the frontend, as the process holds the kernel then.

**Where the source is, in the binary.** The locations of the IR a compile builds name the file of the kernel and of every jit function it
calls, and ``ptxas`` writes those names into the cubin's debug sections. So the frontend of a compile (:func:`recording_source`) names each
file by a path that does not depend on the host (:func:`tilefish._source.source_name`: ``kernels/rms.py`` for the module ``kernels.rms``,
``tilefish_generated/rmsnorm.py`` for a generated kernel, ``triton/language/standard.py`` for Triton's own), not by where the checkout, the
virtual environment or a scratch directory happens to be (:func:`host_independent_paths`). Two builds of one kernel with one toolchain are
then the same bytes on any machine, from any directory, in any process. The digest above never had a location in it.
``ptxas`` runs in an empty directory (:func:`ptxas_in_an_empty_directory`): it would otherwise record the modification time and size of a
source file it finds under such a relative name from its working directory.
``TILEFISH_DEBUG_PATHS=absolute`` keeps the absolute paths instead, for a profiler that opens the source files where they are; the
builds are then not reproducible (:func:`tilefish._source.debug_paths`).
"""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import tempfile
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping

from . import _probe, _source
from .errors import TilefishBuildError

if TYPE_CHECKING:
    from ._tf32guard import Op
    from .aot import AstSourceInputs

IR_DIGEST_KIND = "tilefish.triton-ir.v1"  # the ``source_digest_kind`` of a manifest entry whose ``source_digest`` is an IR digest
GLUON_IR_DIGEST_KIND = "tilefish.gluon-ir.v1"  # the same, of a Gluon kernel: the digest of the TTGIR Gluon's frontend builds
DIGEST_KINDS = {"triton": IR_DIGEST_KIND, "gluon": GLUON_IR_DIGEST_KIND}  # the language of a compile spec -> the kind of its IR digest
LANGUAGE_OF_KIND = {kind: language for language, kind in DIGEST_KINDS.items()}
_TAG = b"tilefish.triton-ir-digest.v1\0"
_TAGS = {"triton": _TAG, "gluon": b"tilefish.gluon-ir-digest.v1\0"}
# Triton's frontend (the code generator) and its compile are not known to be thread-safe, and the compile runs the frontend again inside
# ``triton.compile`` (``ASTSource.make_ir``) with a knob set that is process-wide. So there is one lock, held by every run of the frontend
# here and across the whole of every compile in :func:`tilefish._build.build`: no two frontends ever run at once. (Reentrant, so a
# frontend inside a compile, which nothing here does, would not deadlock.) This is a latency trade-off, taken on purpose: the lock is
# held for the *whole* compile, which takes seconds for a large kernel, so a build, a ``jit`` call or a replay in another thread that
# needs the frontend waits until it is done (a cache hit runs only the frontend, but waits for a compile in progress too).
LOCK = threading.RLock()
_naming = threading.local()  # .depth > 0 in a thread whose frontend names source files by host-independent paths (host_independent_paths)
_ptxas_cwd = threading.local()  # .depth > 0 (and .directory) in a thread whose compile runs Triton's ptxas in an empty directory
_scope = threading.local()  # .table: the frontends a frontend_scope of this thread has run, by their input (None outside one)
_CHUNK = 1 << 20  # the characters of IR text encoded and hashed at a time (ir_digest)


def triton_importable() -> bool:
    """Whether Triton can be imported here (a replay without it has only a pin to go by for a hand-written kernel)."""
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def triton_version() -> "str | None":
    try:
        import triton
    except ImportError:
        return None
    return str(getattr(triton, "__version__", ""))


def ir_digest(text: str, language: str = "triton") -> str:
    """The digest (lowercase hex sha256) of the IR text :func:`ttir_text` made for a kernel of ``language`` (each has its own tag).

    ``sha256(tag + text.encode("utf-8"))``, hashed a megabyte of text at a time, so that no copy of a large IR is made to hash it."""
    h = hashlib.sha256(_TAGS[language])
    for start in range(0, len(text), _CHUNK):
        h.update(text[start : start + _CHUNK].encode("utf-8"))
    return h.hexdigest()


def digest_kind(language: str) -> str:
    """The ``source_digest_kind`` of the IR digest of a kernel of ``language``: :data:`IR_DIGEST_KIND` or :data:`GLUON_IR_DIGEST_KIND`."""
    return DIGEST_KINDS[language]


def is_gluon(jit: Any) -> bool:
    """Whether the jit function ``jit`` is a Gluon kernel, as Triton says (``JITFunction.is_gluon()``; an object without it is not one)."""
    method = getattr(jit, "is_gluon", None)
    if not callable(method):
        return False
    try:
        return method() is True
    except Exception:  # noqa: BLE001 - an object that only looks like a kernel
        return False


def check_language(jit: Any, language: str, what: str) -> None:
    """Raise :class:`~tilefish.errors.TilefishBuildError` when ``jit`` is not a kernel of the frontend ``language`` names (``what`` names it).

    The declaration says which frontend compiles a kernel, and the digest is of that frontend's IR, so a function of the other frontend is
    refused before either runs: Triton's frontend cannot build a Gluon kernel, and Gluon's would build a ``triton.jit`` kernel into an IR
    with another meaning. Every run of a frontend here (a build, ``jit``, the digest a replay recomputes) checks it.
    """
    gluon = is_gluon(jit)
    if language == "gluon" and not gluon:
        raise TilefishBuildError(
            f"{what} is declared a Gluon kernel (@tilefish.extern(gluon=...)), but it is a triton.jit function: declare it with triton=..., "
            "or give the Gluon kernel"
        )
    if language != "gluon" and gluon:
        raise TilefishBuildError(
            f"{what} is a Gluon kernel (@gluon.jit), but it is declared a triton.jit kernel: declare it with @tilefish.extern(gluon=...), "
            "which compiles it with Gluon's frontend"
        )


def source_class(language: str) -> Any:
    """The frontend class ``triton.compile`` is given for a kernel of ``language``: ``ASTSource``, or Gluon's ``GluonASTSource``.

    Raises :class:`~tilefish.errors.TilefishBuildError` when this Triton has no Gluon (``triton.experimental.gluon``, from 3.5 on).
    """
    if language == "gluon":
        try:
            from triton.experimental.gluon._runtime import GluonASTSource
        except ImportError as e:
            raise TilefishBuildError(
                f"a Gluon kernel is compiled by Gluon's frontend (triton.experimental.gluon._runtime.GluonASTSource), which this Triton does "
                f"not have ({e}); tilefish builds Gluon kernels with Triton 3.5"
            ) from e
        return GluonASTSource
    from triton.compiler import ASTSource

    return ASTSource


def ttir_text(jit: Any, inputs: "AstSourceInputs", *, arch: int, options: "Mapping[str, Any]", what: str = "the kernel") -> str:
    """The IR of the jit function ``jit`` as its frontend builds it for ``inputs`` on ``arch``, without locations or names: the TTIR of a
    ``triton.jit`` function (``ASTSource``), the TTGIR of a Gluon one (``GluonASTSource``, when ``inputs.language`` is ``"gluon"``).

    ``inputs`` is what ``spec.ast_source_inputs()`` gives (the signature, the attributes, the constants and the language) and ``options``
    the compile options (the spec's, and any a caller added), exactly as :func:`tilefish._build.build` hands them to ``triton.compile``.
    This is the first thing ``triton.compile`` does, and none of the later stages is run. Raises
    :class:`~tilefish.errors.TilefishBuildError` when Triton is missing or too old to have the entry points, or when the
    frontend cannot build the kernel (``what`` names it in the message). The module is freed before this returns (:func:`release_module`).
    A build, a ``jit`` call and a replay take :func:`frontend` instead, which keeps the digest and not the text.
    """
    return _run_frontend(jit, inputs, arch=arch, options=options, what=what, read=lambda module: str(module.str_nodebug()))


@dataclass(frozen=True)
class Frontend:
    """What is kept of a run of the frontend: the digest of its IR (:func:`ir_digest`), and its dots and MMAs (:func:`tilefish._tf32guard.frontend_ops`:
    what the TF32 guard reads of the IR). Not the IR itself, which is hashed and dropped."""

    digest: str
    ops: "tuple[Op, ...]"


def frontend(jit: Any, inputs: "AstSourceInputs", *, arch: int, options: "Mapping[str, Any]", what: str = "the kernel") -> Frontend:
    """:func:`ttir_text`, digested: the :class:`Frontend` of the IR the frontend builds for ``inputs`` on ``arch`` with ``options``.

    Inside a :func:`frontend_scope` of this thread the frontend runs once per distinct input, the function ``jit`` and the signature, attributes,
    constants, language, options and architecture (each a value of a plain type: ``bool``, ``int``, ``float``, ``str``, ``None``, and lists, tuples and
    dicts of them; an input with any other value is not shared, and runs its own frontend), and the variant being built (production, or the
    instrumented build :func:`tilefish._probe.active` names, whose IR has its probes), and a later call with the same input is given what the first one
    got. Outside a scope every call runs the frontend. A run that raises is not kept."""
    key = _input_key(jit, inputs, arch, options)
    table = getattr(_scope, "table", None)
    if table is not None and key is not None:
        hit = table.get(key)
        if hit is not None:
            return hit[1]
    language = inputs.language

    def read(module: Any) -> Frontend:
        from ._tf32guard import frontend_ops

        text = str(module.str_nodebug())
        return Frontend(digest=ir_digest(text, language), ops=tuple(frontend_ops(text)))

    got = _run_frontend(jit, inputs, arch=arch, options=options, what=what, read=read)
    if table is not None and key is not None:
        table[key] = (jit, got)  # (the function is kept with it, so that its id is not another function's while the scope lasts)
    return got


@contextlib.contextmanager
def frontend_scope() -> "Iterator[None]":
    """Within the block, :func:`frontend` runs the frontend of this thread once per distinct input: what an export (``tilefish.runtime.export``) and
    the ``prepare`` of a plan run in. A scope inside another is the outer one. The block is one point in time of the process: a kernel whose IR would
    change between two of its builds in it (a module constant rewritten while the export runs) is given the IR of its first."""
    if getattr(_scope, "table", None) is not None:
        yield
        return
    _scope.table = {}
    try:
        yield
    finally:
        _scope.table = None


_PLAIN = (bool, int, str, type(None))


def _plain(value: Any) -> Any:
    """``value`` as a hashable key that tells apart every two values Triton's frontend might build differently (``1``, ``1.0``, ``True``; ``0.0`` and
    ``-0.0``; a list and a tuple); raises ``TypeError`` for a value of any other type."""
    kind = type(value)
    if kind in _PLAIN:
        return (kind.__name__, value)
    if kind is float:
        return ("float", repr(value))  # (repr: -0.0 is not 0.0, nan is nan)
    if kind in (list, tuple):
        return (kind.__name__, tuple(_plain(v) for v in value))
    if kind is dict:
        return ("dict", tuple((_plain(k), _plain(v)) for k, v in value.items()))  # in order: a signature's order is the arguments'
    raise TypeError(kind.__name__)


def _input_key(jit: Any, inputs: "AstSourceInputs", arch: int, options: "Mapping[str, Any]") -> "tuple[Any, ...] | None":
    """The key of a frontend run in a :func:`frontend_scope`: everything ``triton.compile`` gives the frontend, and the variant this thread builds
    (production, or the instrumented build :func:`tilefish._probe.instrumenting` names, whose IR has its probes), or ``None`` when a value is not plain."""
    variant = _probe.active()
    try:
        return (
            id(jit),
            _plain(inputs.language),
            _plain(arch),
            _plain(dict(inputs.signature)),
            _plain(dict(inputs.attrs)),
            _plain(dict(inputs.constants)),
            _plain(dict(options)),
            _plain(None if variant is None else variant.key()),
        )
    except TypeError:
        return None


def release_module(module: Any) -> bool:
    """Free the operations of a module Triton built (its frontend's, or what a compile left of it); ``True`` when they were freed.

    Triton's Python binding of a module does not own the operation: when the object goes, the module and everything in it stay allocated for the
    life of the process, and so does every module the frontend builds (a build, a ``jit`` call, a replay call) and every module a compile lowers
    (measured with Triton 3.5.1: about twice the size of the IR text, at every run). Erasing the module's body (its one block) frees every operation
    in it; what is left is the empty module, a few hundred bytes. The module's object holds its MLIR context (``module.context``), so the context is
    alive while the body is erased. Call it only when nothing will read the module again. Nothing is freed (``False``) for a module whose binding
    has no ``walk`` or whose body cannot be found (another Triton, or no Triton): that only leaves it as Triton leaves it.

    The body is the block of the module's last top-level operation, which a post-order walk visits just before the module itself."""
    walk = getattr(module, "walk", None)
    if walk is None:
        return False
    last: "list[Any]" = []

    def visit(op: Any) -> None:
        last.append(op)
        if len(last) > 2:
            del last[0]

    try:
        with LOCK:
            walk(visit)
            if len(last) != 2 or last[1].get_name() != "builtin.module":
                return False
            block = last[0].get_block()
            if block is None:
                return False
            block.erase()
            return True
    except Exception:  # noqa: BLE001 - a binding that does not have what this needs: the module stays as Triton leaves it
        return False
    finally:
        last.clear()


def _run_frontend(jit: Any, inputs: "AstSourceInputs", *, arch: int, options: "Mapping[str, Any]", what: str, read: "Callable[[Any], Any]") -> Any:
    """Run the frontend, give ``read`` the module (under :data:`LOCK`, its context alive), free the module, and return what ``read`` returned."""
    try:
        from triton._C.libtriton import ir
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import make_backend
    except ImportError as e:
        raise TilefishBuildError(
            f"the identity of a binary is the IR Triton's frontend builds, which needs Triton (>= 3.5): pip install 'tilefish[gpu]' ({e})"
        ) from e
    check_language(jit, inputs.language, what)
    try:
        source = source_class(inputs.language)
    except ImportError as e:  # the plain ASTSource of a Triton without its compiler package
        raise TilefishBuildError(f"the identity of a binary is the IR Triton's frontend builds, which needs Triton (>= 3.5) ({e})") from e
    target = GPUTarget("cuda", arch, 32)
    try:
        with LOCK:
            backend = make_backend(target)
            src = source(jit, signature=inputs.signature, constexprs=None, attrs=inputs.attrs)
            src.constants = inputs.constants
            opts = backend.parse_options(dict(options, **src.parse_options()))
            context = ir.context()
            try:
                ir.load_dialects(context)
                backend.load_dialects(context)
                with _probe.frontend():  # (an instrumented build's probes, with the ids every run of the frontend gives them)
                    module = src.make_ir(target, opts, backend.get_codegen_implementation(opts), backend.get_module_map(), context)
                try:
                    return read(module)
                finally:
                    release_module(module)
                    del module
            finally:
                context.disable_multithreading()
    except TilefishBuildError:
        raise
    except Exception as e:  # noqa: BLE001 - whatever the frontend raises for a kernel it cannot build: a build error, with its message
        raise TilefishBuildError(f"Triton could not compile {what} for sm_{arch}: {type(e).__name__}: {e}") from e


def _naming_hook(original: Any) -> Any:
    """``get_jit_fn_file_line`` of Triton's code generator, wrapped: in a thread inside :func:`host_independent_paths` the file it returns is
    named by :func:`tilefish._source.source_name`; anywhere else it is exactly what ``original`` returns."""

    def get_jit_fn_file_line(fn: Any) -> Any:
        file_name, begin_line = original(fn)
        if getattr(_naming, "depth", 0) > 0 and isinstance(file_name, str):
            file_name = _source.source_name(file_name, _source.module_of(fn))
        return file_name, begin_line

    get_jit_fn_file_line.__wrapped__ = original  # type: ignore[attr-defined]
    get_jit_fn_file_line._tilefish_source_names = True  # type: ignore[attr-defined]
    return get_jit_fn_file_line


@contextlib.contextmanager
def host_independent_paths() -> "Iterator[None]":
    """Within the block, the frontend run by this thread names every source file by a path that does not depend on the host.

    Triton's code generator takes the file of a jit function (the kernel, and each jit function it calls) from
    ``triton.compiler.code_generator.get_jit_fn_file_line``, which returns ``fn.__code__.co_filename``, and builds every location of the IR from
    it; nothing else of the IR comes from the path. That function is wrapped once per process, under :data:`LOCK`, and the wrapper changes the
    name only in a thread inside this block (another thread's frontend, a Triton JIT launch of the program's own, gets Triton's name). Line and
    column numbers are kept. Raises :class:`~tilefish.errors.TilefishBuildError` when this Triton has no such function (it is in 3.5.1).
    """
    try:
        from triton.compiler import code_generator
    except ImportError as e:
        raise TilefishBuildError(f"the frontend of a compile is Triton's code generator, which this Triton does not have ({e})") from e
    with LOCK:
        current = getattr(code_generator, "get_jit_fn_file_line", None)
        if current is None:
            raise TilefishBuildError(
                "this Triton's code generator has no get_jit_fn_file_line, through which a build names the source files in a binary's debug "
                "information by paths that do not depend on the host (Triton 3.5.1 has it)"
            )
        if not getattr(current, "_tilefish_source_names", False):
            code_generator.get_jit_fn_file_line = _naming_hook(current)
        _naming.depth = getattr(_naming, "depth", 0) + 1
        try:
            yield
        finally:
            _naming.depth -= 1


class _PtxasCwd:
    """The ``subprocess`` module as Triton's NVIDIA backend sees it, but for ``run``: in a thread inside :func:`ptxas_in_an_empty_directory`
    a command that names no working directory runs in that block's empty directory. Everything else is the module's."""

    _tilefish_ptxas_cwd = True

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if getattr(_ptxas_cwd, "depth", 0) > 0 and kwargs.get("cwd") is None:
            kwargs["cwd"] = _ptxas_cwd.directory
        return self._module.run(*args, **kwargs)


@contextlib.contextmanager
def ptxas_in_an_empty_directory() -> "Iterator[None]":
    """Within the block, the ``ptxas`` that Triton runs in this thread (``make_cubin``) runs in an empty private directory.

    ``ptxas`` looks up each file a PTX ``.file`` directive names, from its own working directory, and where it finds one it writes the file's
    modification time and size into the cubin's line table. The names are the host-independent ones (:func:`host_independent_paths`), and
    relative: so a compile run from the directory that holds ``kernels/rms.py`` (the root of a checkout, or a directory of top-level modules)
    would record when that file was last written, and two checkouts of the same sources, or one touched file, would give other bytes. In an empty
    directory nothing is found, wherever the build runs. ``subprocess`` of Triton's NVIDIA backend is wrapped once per process, under
    :data:`LOCK`, and the wrapper changes nothing in any other thread or outside the block. Nothing is done when the backend is not importable."""
    try:
        from triton.backends.nvidia import compiler
    except ImportError:  # no NVIDIA backend (or no Triton): no ptxas is run by it
        yield
        return
    with LOCK:
        current = getattr(compiler, "subprocess", None)
        if current is None:
            raise TilefishBuildError(
                "this Triton's NVIDIA backend has no subprocess module through which a build runs its ptxas in an empty directory (so that "
                "the cubin does not record the modification time of a source file; Triton 3.5.1 has it)"
            )
        if not getattr(current, "_tilefish_ptxas_cwd", False):
            compiler.subprocess = _PtxasCwd(current)
        depth = getattr(_ptxas_cwd, "depth", 0)
        if depth == 0:
            _ptxas_cwd.directory = tempfile.mkdtemp(prefix="tilefish-ptxas-")
        _ptxas_cwd.depth = depth + 1
        try:
            yield
        finally:
            _ptxas_cwd.depth -= 1
            if _ptxas_cwd.depth == 0:
                shutil.rmtree(_ptxas_cwd.directory, ignore_errors=True)


def recording_source(ASTSource: Any, language: str = "triton") -> Any:
    """A subclass of ``ASTSource`` (or of Gluon's ``GluonASTSource``, :func:`source_class`, with ``language`` ``"gluon"``) that records the digest
    of the IR its ``make_ir`` builds in ``.recorded_digest`` (of what a compile then takes) and keeps the module, and whose frontend names the source files
    by host-independent paths (:func:`host_independent_paths`), so that the binary's debug information does not depend on where the files are.

    ``build`` compiles from this and checks that the IR it recorded has the digest it looked the cache up with, so a binary is
    only ever stored under the digest of the IR it was compiled from; then it calls ``release()``, which frees what the compile left of the module
    (:func:`release_module`: Triton keeps every module it lowers otherwise).
    """

    class RecordingSource(ASTSource):  # type: ignore[misc, valid-type]
        recorded_digest: "str | None" = None
        module: Any = None

        def make_ir(self, target: Any, options: Any, codegen_fns: Any, module_map: Any, context: Any) -> Any:
            # (TILEFISH_DEBUG_PATHS=absolute keeps Triton's absolute paths, for a profiler: not reproducible, _source.debug_paths)
            naming = contextlib.nullcontext() if _source.debug_paths() == "absolute" else host_independent_paths()
            with naming, _probe.frontend():
                module = super().make_ir(target, options, codegen_fns, module_map, context)
            self.recorded_digest = ir_digest(str(module.str_nodebug()), language)  # before the passes of the first stage change the module in place
            self.module = module
            return module

        def release(self) -> None:
            """Free the module the compile lowered (once the compile is done: nothing reads it again)."""
            module, self.module = self.module, None
            if module is not None:
                release_module(module)

    return RecordingSource
