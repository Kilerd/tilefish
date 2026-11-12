"""Exception types raised by tilefish."""

from __future__ import annotations


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
