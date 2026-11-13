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
replay call. Nothing here compiles anything.
"""

from __future__ import annotations

import hashlib
import threading
from typing import TYPE_CHECKING, Any, Mapping

from .errors import TilefishBuildError

if TYPE_CHECKING:
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
    """The digest (lowercase hex sha256) of the IR text :func:`ttir_text` made for a kernel of ``language`` (each has its own tag)."""
    return hashlib.sha256(_TAGS[language] + text.encode("utf-8")).hexdigest()


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
    frontend cannot build the kernel (``what`` names it in the message).
    """
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
                module = src.make_ir(target, opts, backend.get_codegen_implementation(opts), backend.get_module_map(), context)
                return str(module.str_nodebug())
            finally:
                context.disable_multithreading()
    except TilefishBuildError:
        raise
    except Exception as e:  # noqa: BLE001 - whatever the frontend raises for a kernel it cannot build: a build error, with its message
        raise TilefishBuildError(f"Triton could not compile {what} for sm_{arch}: {type(e).__name__}: {e}") from e


def recording_source(ASTSource: Any) -> Any:
    """A subclass of ``ASTSource`` (or of Gluon's ``GluonASTSource``, :func:`source_class`) that keeps the IR text its ``make_ir`` builds in
    ``.ir_text`` (what a compile then takes).

    ``build`` compiles from this and checks that the text it recorded has the digest it looked the cache up with, so a binary is
    only ever stored under the digest of the IR it was compiled from.
    """

    class RecordingSource(ASTSource):  # type: ignore[misc, valid-type]
        ir_text: "str | None" = None

        def make_ir(self, target: Any, options: Any, codegen_fns: Any, module_map: Any, context: Any) -> Any:
            module = super().make_ir(target, options, codegen_fns, module_map, context)
            self.ir_text = str(module.str_nodebug())  # before the passes of the first stage change the module in place
            return module

    return RecordingSource
