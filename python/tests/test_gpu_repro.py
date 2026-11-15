"""Reproducible exports on a GPU: the same graph exported in two processes, from two copies of its sources in two directories, is the same files.

Two exports used to differ byte for byte (issue: lowered kernels were not reproducible across exports; the manifest digest depended on the build
directory): Triton records the path of each source file in the locations of the IR, and ``ptxas`` writes it into the cubin's ``.debug_line`` and
``.nv_debug_ptx_txt``. That path named the checkout and the virtual environment (a hand-written kernel, Triton's ``triton/language/standard.py``) or
a random ``/tmp/tilefish-gen-XXXXXXXX`` directory (a lowered kernel, a generated op). So the ``sha256`` of a cubin, and the manifest and plan digests,
changed with the directory and the process; ``sha256_code`` (the machine code) did not. Now a build names each file by its module path, and a
generated source is written to a content-addressed file.

Each child process has its own working directory, its own copy of ``repro_graph.py`` (in directories of different depths), its own tilefish build
cache and Triton cache, its own ``TMPDIR`` and its own hash seed. The test checks that the two exports, and a third one made by this process from
the tests directory, are the same files (``manifest.json``, ``plan.json``, every cubin), so every ``sha256``, ``sha256_code``, the manifest digest and
the plan digest are equal, for a lowered kernel, generated ops, a hand-written extern and a prebuilt PTX; that no file names a directory of the
build; and that a replay of one export runs with the pin of another.
"""

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import repro_graph  # noqa: E402
from gpu_util import same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import runtime  # noqa: E402

TESTS = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path(tilefish.__file__).resolve().parent.parent

CHILD = r"""
import hashlib, json, os, pathlib, sys
sources, out = sys.argv[1], pathlib.Path(sys.argv[2])
sys.path.insert(0, sources)
import repro_graph
from tilefish import runtime

plan = runtime.export(repro_graph.repro, out)
doc = json.loads((out / "manifest.json").read_text())
print(json.dumps({
    "manifest_digest": plan.manifest_digest,
    "plan_digest": plan.digest,
    "kernels": {e["stem"]: {"kind": e["kind"], "sha256": e["sha256"], "sha256_code": e["sha256_code"], "generated": "source_sha256" in e} for e in doc["kernels"]},
    "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())},
    "module": repro_graph.__file__,
    "generated_files": sorted(fn.__globals__["__file__"] for fn in runtime._modules.values()),
}))
"""


def export_elsewhere(root: pathlib.Path, seed: str, **extra_env: str):
    """Copy the graph's sources under ``root`` and export it from a child process with a working directory, caches and a TMPDIR of its own."""
    sources = root / "sources"
    (sources / "ptx").mkdir(parents=True)
    shutil.copy(TESTS / "repro_graph.py", sources / "repro_graph.py")
    shutil.copy(TESTS / "ptx" / "axpy_f32.ptx", sources / "ptx" / "axpy_f32.ptx")
    for name in ("cwd", "tmp"):
        (root / name).mkdir()
    env = dict(os.environ)
    env.update(
        TILEFISH_CACHE_DIR=str(root / "tilefish-cache"),
        TRITON_CACHE_DIR=str(root / "triton-cache"),
        TMPDIR=str(root / "tmp"),
        PYTHONHASHSEED=seed,
        PYTHONPATH=os.pathsep.join([str(SRC), str(TESTS), env.get("PYTHONPATH", "")]),
    )
    env.update(extra_env)
    r = subprocess.run([sys.executable, "-c", CHILD, str(sources), str(root / "out")], cwd=root / "cwd", env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-4000:]
    result = json.loads(r.stdout.strip().splitlines()[-1])
    result["root"] = root
    return result


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    base = tmp_path_factory.mktemp("repro")
    first = export_elsewhere(base / "first", "1")
    second = export_elsewhere(base / "second" / "a" / "much" / "deeper" / "place", "2")
    return first, second


def test_two_exports_from_two_directories_in_two_processes_are_the_same_files(exports):
    first, second = exports
    assert first["module"] != second["module"]  # two copies of the sources ...
    assert all(f.startswith(str(first["root"])) for f in first["generated_files"]) and first["generated_files"]  # ... two places for generated sources
    assert all(f.startswith(str(second["root"])) for f in second["generated_files"])
    assert first["files"] == second["files"]  # manifest.json, plan.json and every cubin, byte for byte
    assert first["manifest_digest"] == second["manifest_digest"] and first["plan_digest"] == second["plan_digest"]
    assert first["kernels"] == second["kernels"]
    kinds = {stem: (k["kind"], k["generated"]) for stem, k in first["kernels"].items()}
    lowered = [s for s in kinds if s.startswith("softmax_rows")]
    extern = [s for s in kinds if s.startswith("scale_rows")]
    generated = [s for s, (kind, gen) in kinds.items() if gen and s not in lowered]
    assert lowered and extern and all(kinds[s] == ("triton", True) for s in lowered) and all(kinds[s] == ("triton", False) for s in extern)
    assert {"im2col1d", "col2im1d", "contiguous"} <= {g.split("_")[0] for g in generated}, generated  # the three generated ops
    assert kinds["repro_axpy"] == ("binary", False)  # the prebuilt PTX


def test_no_file_of_an_export_names_a_directory_of_its_build(exports):
    for export in exports:
        root = str(export["root"]).encode()
        for path in (export["root"] / "out").iterdir():
            data = path.read_bytes()
            assert root not in data and b"tilefish-gen-" not in data and b"site-packages" not in data, path.name
            if path.suffix == ".cubin" and not path.name.startswith("repro_axpy"):  # a Triton binary names its sources by module path
                assert b"triton/language" in data or b"tilefish_generated" in data or b"repro_graph.py" in data, path.name


def test_an_export_of_this_process_is_the_same_and_replays_with_the_pin_of_another(exports, tmp_path):
    first, second = exports
    plan = runtime.export(repro_graph.repro, tmp_path)  # a third place: the tests directory, this process's caches
    assert plan.digest == first["plan_digest"] and plan.manifest_digest == first["manifest_digest"]
    x = torch.randn(32, 64, device="cuda")
    c = torch.randn(2, 16, 32, device="cuda").to(torch.bfloat16)
    v = torch.randn(1024, device="cuda")
    jit = tilefish.run(repro_graph.repro, x, c, v)
    replayed = tilefish.run(repro_graph.repro, x, c, v, mode="replay", plan=second["root"] / "out", pin=first["plan_digest"])
    for a, b in zip(jit, replayed):
        assert same_bytes(a, b)


def test_a_jit_build_has_the_bytes_of_the_exported_binary(exports):
    first, _ = exports
    for op in (repro_graph.softmax_rows, repro_graph.scale_rows):
        target = runtime._target(op)
        for spec in target.compile_specs():
            built = runtime.build_for_device(spec, target.jit_function(spec), 0)
            assert hashlib.sha256(built.cubin).hexdigest() == first["kernels"][spec.stem]["sha256"], spec.stem


def test_absolute_debug_paths_name_the_build_directory_with_the_same_machine_code(exports, tmp_path):
    """``TILEFISH_DEBUG_PATHS=absolute`` (for a profiler) keeps the absolute source paths in the cubins: the files name where the sources and the
    generated sources were, so the export is not the reproducible one, but every ``sha256_code`` is."""
    first, _ = exports
    absolute = export_elsewhere(tmp_path / "absolute", "3", TILEFISH_DEBUG_PATHS="absolute")
    assert absolute["kernels"].keys() == first["kernels"].keys()
    assert all(absolute["kernels"][s]["sha256_code"] == k["sha256_code"] for s, k in first["kernels"].items())
    assert absolute["manifest_digest"] != first["manifest_digest"]
    root = str(absolute["root"]).encode()
    named = [p.name for p in (absolute["root"] / "out").iterdir() if p.suffix == ".cubin" and root in p.read_bytes()]
    assert any(n.startswith("softmax_rows") for n in named) and any(n.startswith("scale_rows") for n in named), named
