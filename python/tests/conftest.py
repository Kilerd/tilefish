import itertools
import pathlib
import re
import types

import pytest

from tilefish.trace import TraceContext

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


class Tracer:
    """Makes parameter-like values in a fresh trace, for testing built-in ops directly."""

    def __init__(self):
        self.ctx = TraceContext("test", "graph")
        self._names = itertools.count()

    def val(self, type, name=None, access="in"):
        return self.ctx.new_value(type, name=name or f"p{next(self._names)}", access=access, is_param=True)

    @property
    def nodes(self):
        return self.ctx.nodes


@pytest.fixture
def tr():
    return Tracer()


def readme_snippet() -> str:
    text = (REPO_ROOT / "README.md").read_text()
    m = re.search(r"## What it looks like\n\n```python\n(.*?)```", text, re.S)
    assert m, "README.md has no python block under '## What it looks like'"
    return m.group(1)


def make_my_kernels():
    """A stand-in for the hypothetical ``my_kernels`` module the README refers to."""

    def gelu():  # never called: tilefish only records the reference
        raise NotImplementedError

    gelu.__module__ = "my_kernels"
    return types.SimpleNamespace(gelu=gelu)
