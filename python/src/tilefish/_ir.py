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
_TAG = b"tilefish.triton-ir-digest.v1\0"
# Triton's frontend (the code generator) and its compile are not known to be thread-safe, and the compile runs the frontend again inside
# ``triton.compile`` (``ASTSource.make_ir``) with a knob set that is process-wide. So there is one lock, held by every run of the frontend
# here and across the whole of every compile in :func:`tilefish._build.build`: no two frontends ever run at once. (Reentrant, so a
# frontend inside a compile, which nothing here does, would not deadlock.)
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


def ir_digest(text: str) -> str:
    """The digest (lowercase hex sha256) of the IR text :func:`ttir_text` made."""
    return hashlib.sha256(_TAG + text.encode("utf-8")).hexdigest()


def ttir_text(jit: Any, inputs: "AstSourceInputs", *, arch: int, options: "Mapping[str, Any]", what: str = "the kernel") -> str:
    """The TTIR of the ``triton.jit`` function ``jit`` as Triton's frontend builds it for ``inputs`` on ``arch``, without locations or names.

    ``inputs`` is what ``spec.ast_source_inputs()`` gives (the signature, the attributes and the constants) and ``options`` the
    compile options (the spec's, and any a caller added), exactly as :func:`tilefish._build.build` hands them to ``triton.compile``.
    This is the first thing ``triton.compile`` does, and none of the later stages is run. Raises
    :class:`~tilefish.errors.TilefishBuildError` when Triton is missing or too old to have the entry points, or when the
    frontend cannot build the kernel (``what`` names it in the message).
    """
    try:
        from triton._C.libtriton import ir
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        from triton.compiler.compiler import make_backend
    except ImportError as e:
        raise TilefishBuildError(
            f"the identity of a binary is the IR Triton's frontend builds, which needs Triton (>= 3.5): pip install 'tilefish[gpu]' ({e})"
        ) from e
    target = GPUTarget("cuda", arch, 32)
    try:
        with LOCK:
            backend = make_backend(target)
            src = ASTSource(jit, signature=inputs.signature, constexprs=None, attrs=inputs.attrs)
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
    """A subclass of ``ASTSource`` that keeps the IR text its ``make_ir`` builds in ``.ir_text`` (what a compile then takes).

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
