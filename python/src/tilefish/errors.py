"""Exception types raised by tilefish."""

from __future__ import annotations

import pickle


class TilefishError(Exception):
    """Base class of every error raised by tilefish."""


class TilefishTypeError(TilefishError, TypeError):
    """A type, shape, dtype or divisibility rule was violated.

    The message names the operation, the argument and the mismatch.
    """


class TilefishDeclarationError(TilefishError, ValueError):
    """A dimension, dtype, signature or decorator was declared incorrectly."""


class TilefishSpecError(TilefishError, ValueError):
    """A declaration cannot be turned into an ahead-of-time compile spec."""


class TilefishLoweringError(TilefishError, ValueError):
    """A kernel body does not fit a lowering template, so no Triton source can be generated for it.

    The message says which rule failed. ``rule`` is that rule's short name (for example
    ``"row.last-axis-reduction"``); when several templates were tried and none fit, ``rule`` is
    ``"no-template"`` and ``causes`` holds the error of each template.
    """

    def __init__(self, message: str, *, rule: "str | None" = None, causes: "tuple[TilefishLoweringError, ...]" = ()) -> None:
        super().__init__(message)
        self.rule = rule
        self.causes = tuple(causes)


class TilefishAttributeError(TilefishError, AttributeError):
    """An attribute or method that traced values do not have (in M1)."""


class TilefishPicklingError(TilefishError, pickle.PicklingError):
    """A declared operator cannot be pickled by reference (it is not a module-level object)."""


class TilefishBuildError(TilefishError, RuntimeError):
    """A compile spec could not be built into a binary.

    Triton is missing or too old, the compiler failed, the binary does not match the ABI its spec promises,
    or the ELF file is malformed.
    """


class TilefishRuntimeError(TilefishError, RuntimeError):
    """A kernel cannot be run.

    There is no CUDA, the mode is unknown, a binary needs launch attributes the replay does not set, a driver
    call failed, or the mode has nothing to run for this operator (``reference`` of an ``extern``).
    """


class TilefishManifestError(TilefishRuntimeError):
    """A manifest directory cannot be replayed: a file is missing or malformed, a cubin does not match the digests
    the manifest records, or the binary is not the one the compile spec of the operator asks for."""


class TilefishPreconditionError(TilefishRuntimeError):
    """An argument violates a precondition of the binary it would run on (the types and the compile spec).

    ``kind`` names the rule: ``"dtype"``, ``"rank"``, ``"shape"``, ``"static_size"``, ``"divisible"``,
    ``"bucket"``, ``"pointer_alignment"``, ``"max_elements"``, ``"contiguous"``, ``"device"`` or ``"alias"``. ``arg`` is the
    argument it is about, when there is one.
    """

    def __init__(self, message: str, *, kind: str, arg: "str | None" = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.arg = arg


class TilefishCacheWarning(UserWarning):
    """tilefish's build cache could not be used as it is: an entry is corrupt or stale (it is ignored and the kernel is built
    again), or the cache directory cannot be written (builds are not cached there). It is a warning, never an error: a build
    does not depend on the cache."""
