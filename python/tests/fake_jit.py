"""A stand-in for ``triton.JITFunction`` for the tests that need no Triton: the parts the source digest reads.

A real ``JITFunction`` keeps the function (``fn``), its globals, and the lines of its source as they were when the module
was imported (``raw_src`` from line ``starting_line_number``, decorators included). This one takes them the same way.
"""

import inspect


class FakeJit:
    def __init__(self, fn):
        self.fn = fn
        self.raw_src, self.starting_line_number = inspect.getsourcelines(fn)
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__
        self.__name__ = fn.__name__
        self.arg_names = list(inspect.signature(fn).parameters)

    def __call__(self, *args):
        """Running the function is what a fake "compile" does to see the values the process holds (a real one is not called)."""
        return self.fn(*args)


def jit(fn):
    return FakeJit(fn)


class Wrapper:
    """What ``triton.autotune`` and ``triton.heuristics`` return: an object that holds the jit function as ``.fn``."""

    def __init__(self, fn):
        self.fn = fn


def wrap(fn):
    return Wrapper(fn)
