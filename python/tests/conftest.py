import functools
import itertools
import pathlib
import re
import types

import pytest

from tilefish.trace import TraceContext

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=None)
def gpu_problem():
    """Why the ``gpu`` tests cannot run here (``None`` when they can): they need torch with CUDA, triton and cuda-python."""
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    if not torch.cuda.is_available():
        return "no CUDA device is available to torch"
    try:
        import triton  # noqa: F401
    except ImportError:
        return "triton is not installed"
    try:
        from cuda.bindings import driver  # noqa: F401
    except ImportError:
        return "cuda-python is not installed"
    return None


def pytest_configure(config):
    # also registered in pyproject.toml; repeated here for a run from the repository root, which has no ini file
    config.addinivalue_line("markers", "gpu: needs a CUDA device with torch, triton and cuda-python; skipped automatically when any is missing")


def pytest_collection_modifyitems(config, items):
    """Tests marked ``gpu`` are skipped, not failed, where CUDA, torch, triton or cuda-python is missing."""
    gpu_items = [item for item in items if item.get_closest_marker("gpu")]
    if not gpu_items:
        return
    problem = gpu_problem()
    if problem is not None:
        skip = pytest.mark.skip(reason=f"gpu test: {problem}")
        for item in gpu_items:
            item.add_marker(skip)


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
