"""The CuTe DSL fixture of the Rust crate (``crates/tilefish/tests/fixtures/cute_sm_103``), which ``export_rust_fixtures.py`` writes and
``tests/cute_externs.rs`` reads with the digest Python computed (``expected.json``, ``"cute"``). Not collected.

It is the export of ``test_cute_externs.py``'s fake CuTe build (no GPU, no DSL): ``gemm(a: Tensor[M, 64, f16], b: Tensor[64, 64, f16], c:
Out[Tensor[M, 64, f16]])`` for the buckets 128 and 256 of ``M``, ``a`` and ``b`` TMA descriptors, four constant bytes and ``c`` a pointer, a cluster
of 2 CTAs. Its host function is :func:`host` of this file: a declaration digest has the sha256 of the host function's file, so the fixture changes
when this file or the fake build does, not with every test of ``test_cute_externs.py``.
"""

import pathlib
import shutil

NAME = "cute_sm_103"


def host(a, b, c, stream):
    """Stands in for a ``@cute.jit`` host function."""


def write(directory: pathlib.Path):
    """Write the fixture into ``directory`` (made afresh: ``manifest.json`` and the cubins) and return the manifest, loaded."""
    import test_cute_externs as tc

    from tilefish import _cute
    from tilefish._manifest import Manifest

    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    installed = _cute.dsl_version
    _cute.dsl_version = lambda: tc.DSL  # the version the fake build records, with or without the DSL installed
    try:
        tc.exported(directory, op=tc.declare(cute=host))
    finally:
        _cute.dsl_version = installed
    return Manifest.load(directory)
