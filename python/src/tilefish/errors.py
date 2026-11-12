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
