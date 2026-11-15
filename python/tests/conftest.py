import functools
import itertools
import os
import pathlib
import re
import shutil
import subprocess
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


# -- the ptxas Triton compiles with, and the architectures it can target ---------------------------------------------------------------------
# Triton 3.5.1 bundles the ptxas of CUDA 12.8, which has no sm_103a: on a machine without a newer one, a test that builds for sm_103 (or for
# every architecture) skips with what to do instead of failing in ptxas.


def triton_gpu_name(cap: int) -> str:
    """The ``--gpu-name`` Triton 3.5.1 hands ptxas for compute capability ``cap``: ``sm_<cap>a`` from sm_90 on (the architecture-specific target),
    ``sm_<cap>`` before (``sm_arch_from_capability`` of ``triton/backends/nvidia/compiler.py``)."""
    return f"sm_{cap}{'a' if cap >= 90 else ''}"


def ptxas_gpu_names(help_text: str) -> "frozenset[str]":
    """The values ``--gpu-name`` takes, from what ``ptxas --help`` prints (``Allowed values for this option:  'compute_100', ..., 'sm_90a'.``);
    empty when the text has no such list."""
    m = re.search(r"^--gpu-name\b.*?Allowed values for this option:(.*?)(?:Default value|^--)", help_text, re.S | re.M)
    return frozenset(re.findall(r"'([A-Za-z0-9_]+)'", m.group(1))) if m else frozenset()


def ptxas_release(version_text: str) -> str:
    """The version of a ptxas from what ``ptxas --version`` prints: ``12.9.86`` of ``Cuda compilation tools, release 12.9, V12.9.86``, or its last
    line when it has no such line."""
    m = re.search(r"release [0-9.]+, V([0-9.]+)", version_text)
    lines = [ln.strip() for ln in version_text.splitlines() if ln.strip()]
    return m.group(1) if m else (lines[-1] if lines else "of unknown version")


def triton_ptxas() -> "str | None":
    """The ptxas Triton compiles with: Triton's ``knobs.nvidia.ptxas`` (``TRITON_PTXAS_PATH`` when it is set, the ptxas bundled with Triton
    otherwise); without Triton, ``TRITON_PTXAS_PATH`` or the ``ptxas`` on the PATH."""
    try:
        import triton

        return str(triton.knobs.nvidia.ptxas.path)
    except Exception:  # noqa: BLE001 - no Triton, or one without the knob
        return os.environ.get("TRITON_PTXAS_PATH") or shutil.which("ptxas")


@functools.lru_cache(maxsize=None)
def _ptxas_facts(path: str) -> "tuple[str, frozenset[str]]":
    """The version of the ptxas at ``path`` and the values its ``--gpu-name`` takes: run once per path and process."""
    version = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60).stdout
    names = ptxas_gpu_names(subprocess.run([path, "--help"], capture_output=True, text=True, timeout=60).stdout)
    return ptxas_release(version), names


def ptxas_problem(*archs: "int | str") -> "str | None":
    """Why Triton cannot build for every one of ``archs`` (``103`` or ``"sm_103"``) here, or ``None`` when it can (or when its ptxas does not say
    which targets it takes: the build then shows what is wrong)."""
    caps = [int(str(a).removeprefix("sm_")) for a in archs]
    path = triton_ptxas()
    if path is None:
        return "there is no ptxas to build with: install Triton, or set TRITON_PTXAS_PATH to a ptxas of CUDA 12.9 or newer"
    try:
        version, names = _ptxas_facts(path)
    except (OSError, subprocess.SubprocessError) as e:
        return f"the ptxas Triton compiles with ({path}) cannot be run: {e}; set TRITON_PTXAS_PATH to a ptxas of CUDA 12.9 or newer"
    missing = [name for name in dict.fromkeys(map(triton_gpu_name, caps)) if name not in names]
    if not names or not missing:
        return None
    return (
        f"the ptxas Triton compiles with ({path}, version {version}) cannot target {', '.join(missing)}: set TRITON_PTXAS_PATH to a ptxas of "
        "CUDA 12.9 or newer"
    )


def require_ptxas_for(*archs: "int | str") -> None:
    """Skip the calling test when Triton's ptxas cannot target one of ``archs`` (see :func:`ptxas_problem`)."""
    problem = ptxas_problem(*archs)
    if problem is not None:
        pytest.skip(problem)


@pytest.fixture(autouse=True)
def _isolated_build_cache(tmp_path_factory, monkeypatch):
    """Every test has its own, empty tilefish build cache: none reads or writes ``~/.cache/tilefish``, and none sees another's entries."""
    monkeypatch.setenv("TILEFISH_CACHE_DIR", str(tmp_path_factory.mktemp("tilefish-cache")))


def pytest_configure(config):
    # also registered in pyproject.toml; repeated here for a run from the repository root, which has no ini file
    config.addinivalue_line("markers", "gpu: needs a CUDA device with torch, triton and cuda-python; skipped automatically when any is missing")
    config.addinivalue_line(
        "markers", "cuda_math: the exhaustive comparisons with CUDA's math library, to run again on a CUDA or torch change (docs/numerics.md)"
    )


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
    """The code block of the README that declares the design's kernels; the test that needs it is skipped without a README."""
    if not (REPO_ROOT / "README.md").exists():
        pytest.skip("README.md is not available (not running from a source checkout)")
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
