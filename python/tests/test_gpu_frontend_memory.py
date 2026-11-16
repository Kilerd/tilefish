"""The memory of Triton's frontend and of an export of many buckets, with Triton on a GPU (``irmem_kernels.py``: a Gluon extern of 132 bucket entries
and three configurations, whose IR grows with an unrolled chain).

Triton's Python binding of a module does not own it: a module that is only dropped stays allocated for the life of the process, about twice the size
of its IR text, at every run of the frontend and at every compile (Triton 3.5.1). tilefish frees what it built (:func:`tilefish._ir.release_module`),
and an export runs the frontend once per distinct input, not once per bucket (:func:`tilefish._build.export_session`), and writes the same bytes as
an export that runs every bucket's own frontend. ``test_frontend_once.py`` checks the same without Triton; ``irmem_export.py`` measures the export of
a large kernel.
"""

import contextlib
import gc
import json
import os
import pathlib

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("triton.experimental.gluon")
pytest.importorskip("cuda.bindings.driver")

import irmem_kernels as ik  # noqa: E402

from tilefish import _ir, runtime  # noqa: E402
from tilefish.aot import build, compile_specs  # noqa: E402

CAP = torch.cuda.get_device_capability()
ARCH = CAP[0] * 10 + CAP[1]


def rss_mib():
    """The resident memory of this process now (Linux: ``/proc/self/statm``)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        pytest.skip("no /proc/self/statm: the resident memory of the process is not known here")


def test_the_frontend_frees_the_module_it_builds():
    op = ik.chain_op(4096, name="chain_mem")
    spec = compile_specs(op)[0]
    jit = runtime.resolve_triton(spec.triton)
    inputs = spec.ast_source_inputs()
    text = _ir.ttir_text(jit, inputs, arch=ARCH, options=inputs.options)  # (the first run imports and loads what every later one uses)
    size = len(text) / 2**20
    del text
    runs = 10
    gc.collect()
    before = rss_mib()
    for _ in range(runs):  # outside a scope: every call runs the frontend
        _ir.frontend(jit, inputs, arch=ARCH, options=inputs.options)
        gc.collect()  # (the code generator's Python objects are cycles: collected at each run, so that their pages are reused, not grown)
    grown = rss_mib() - before
    # a module kept per run would be about twice its text (2.2 MiB of text here): 40 MiB over the runs; freed, the allocator's noise
    assert size > 1.5 and grown < runs * size / 2, f"{grown:.1f} MiB more after {runs} runs of a frontend whose IR is {size:.1f} MiB of text"


def test_the_module_of_the_frontend_and_of_a_compile_is_found_and_freed(monkeypatch):
    """With this Triton, :func:`tilefish._ir.release_module` finds the body of the frontend's module (TTIR or TTGIR) and of what a compile lowered (the
    LLVM dialect) and erases it: a Triton whose binding changed would leave both allocated, and this says so."""
    freed = []
    release = _ir.release_module

    def spy(module):
        freed.append(release(module))
        return freed[-1]

    monkeypatch.setattr(_ir, "release_module", spy)
    spec = compile_specs(ik.chain_op(8, name="chain_free"))[0]
    built = build(spec, arch=ARCH, cache=False)
    assert freed == [True, True]  # the digest's frontend, then the compile's module
    inputs = spec.ast_source_inputs()
    text = _ir.ttir_text(runtime.resolve_triton(spec.triton), inputs, arch=ARCH, options=inputs.options)
    assert built.ir_digest == _ir.ir_digest(text, "gluon")  # (a frontend after a freed one builds the same IR)


def test_an_export_of_many_buckets_runs_the_frontend_once_per_configuration_and_writes_the_same_bytes(tmp_path, monkeypatch):
    op = ik.chain_op(64, name="chain_once")
    runs = []
    run = _ir._run_frontend

    def counting(*a, **k):
        runs.append(1)
        return run(*a, **k)

    monkeypatch.setattr(_ir, "_run_frontend", counting)
    once = runtime.export(op, tmp_path / "once", arch=f"sm_{ARCH}")
    doc = json.loads((tmp_path / "once" / "manifest.json").read_text())
    assert len(doc["kernels"]) == 132 and len({e["source_digest"] for e in doc["kernels"]}) == 3
    assert len(runs) == 3  # (a compile runs Triton's own frontend, which this does not count)
    runs.clear()
    with monkeypatch.context() as m:  # every bucket's own frontend, as an export did before its session
        m.setattr(runtime, "export_session", contextlib.nullcontext)
        each = runtime.export(op, tmp_path / "each", arch=f"sm_{ARCH}")
    assert len(runs) == 132 and each.digest == once.digest

    def files(root):
        return {str(p.relative_to(root)): p.read_bytes() for p in sorted(pathlib.Path(root).rglob("*")) if p.is_file()}

    assert files(tmp_path / "once") == files(tmp_path / "each")
