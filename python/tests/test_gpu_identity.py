"""The identity of a binary is the Triton IR its kernel compiles to: every route a kernel reads a constant by, on a real GPU.

Each earlier way of *finding* what Triton reads at compile time (a digest of the files a kernel reaches, a walk over the module
namespaces, the closure cells, a watch on the imports) was shown, in turn, to hand a stale binary through the build cache and through
export and replay: ``getattr(cfg, "SCALE")`` and ``getattr(cfg, "SC" + "ALE")``, a class attribute ``cfg.Inner.SCALE``, a
``triton.constexpr_function`` that reads the environment, a ``tl.constexpr`` of an object whose ``repr`` hides its state, a user
module with the name of a standard library module. Nothing is found any more: the frontend is run, and the IR it builds is hashed
(:mod:`tilefish._ir`). These tests run each route in real processes.

For each route, three processes share one tilefish cache directory and the same files (``identity_routes.py`` writes them and is the child):

1. ``R_VALUE=2.0``: ``jit`` (a compile) and ``export``;
2. ``R_VALUE=3.0``: ``jit`` must run the 3.0 kernel (a compile again, not the first one's binary), and a replay of the export with its pin must be
   refused as stale;
3. ``R_VALUE=2.0``: ``jit`` must be a hit (no compile), and the pinned replay is accepted.

And the digest is the same from another directory, with other line numbers and another hash seed, and is not changed by a comment, blank lines or
the name of a local variable.
"""

import concurrent.futures
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import identity_routes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import runtime  # noqa: E402
from tilefish._build import resolve_triton  # noqa: E402
from tilefish.aot import compile_specs  # noqa: E402

CAP = torch.cuda.get_device_capability()
ARCH = CAP[0] * 10 + CAP[1]
TESTS = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path(tilefish.__file__).resolve().parent.parent
NEEDS_CONSTEXPR_FUNCTION = ("constexpr_function", "hides its state")
ROUTES = [r for r in identity_routes.ROUTES if hasattr(triton, "constexpr_function") or not any(n in r for n in NEEDS_CONSTEXPR_FUNCTION)]


def child(directory, route, role, value, *, cache, export_dir, pin="", seed="0", cwd=None):
    env = dict(os.environ)
    env.update(
        R_VALUE=value,
        TILEFISH_CACHE_DIR=str(cache),
        PYTHONHASHSEED=seed,
        PYTHONPATH=os.pathsep.join([str(SRC), str(TESTS), env.get("PYTHONPATH", "")]),
    )
    r = subprocess.run(
        [sys.executable, str(TESTS / "identity_routes.py"), str(directory), route, role, str(export_dir), pin],
        env=env,
        cwd=cwd or directory,
        capture_output=True,
        text=True,
        timeout=900,
    )
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert r.returncode == 0 and lines, f"{route} ({role}, R_VALUE={value}) failed:\n{r.stderr[-3000:]}"
    return json.loads(lines[-1][len("RESULT ") :])


def run_route(root, route):
    directory, cache, export_dir = root / "project", root / "cache", root / "export"
    identity_routes.write(directory, route)
    first = child(directory, route, "first", "2.0", cache=cache, export_dir=export_dir)
    other = child(directory, route, "other", "3.0", cache=cache, export_dir=export_dir, pin=first["pin"])
    again = child(directory, route, "again", "2.0", cache=cache, export_dir=export_dir, pin=first["pin"])
    return {"first": first, "other": other, "again": again}


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory):
    """Every route's three processes, run once, a few routes at a time (each process holds a CUDA context)."""
    roots = {route: tmp_path_factory.mktemp("route") for route in ROUTES}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {route: pool.submit(run_route, roots[route], route) for route in ROUTES}
        return {route: future for route, future in futures.items()}


@pytest.mark.parametrize("route", ROUTES)
def test_whichever_route_the_cache_serves_each_value_its_own_binary(outcomes, route):
    got = outcomes[route].result()
    assert got["first"]["jit"] == 2.0 and got["first"]["compiles"] == 1  # compiled
    assert got["other"]["jit"] == 3.0 and got["other"]["compiles"] == 1  # another process, the same files, another value: a miss, not 2.0 from the cache
    assert got["again"]["jit"] == 2.0 and got["again"]["compiles"] == 0  # the first value again: a hit
    assert got["first"]["digest"] == got["again"]["digest"] != got["other"]["digest"]


@pytest.mark.parametrize("route", ROUTES)
def test_whichever_route_a_pinned_replay_in_a_fresh_process_refuses_another_value_and_accepts_its_own(outcomes, route):
    got = outcomes[route].result()
    assert got["first"]["pin"] and len(got["first"]["pin"]) == 64
    assert got["other"]["replay"] is None and "built from Triton IR with digest" in got["other"]["refused"]  # the pin is no excuse
    assert "now compiles to IR with digest" in got["other"]["refused"] and "stale" in got["other"]["refused"]
    assert got["again"]["refused"] is None and got["again"]["replay"] == 2.0  # a process with the first value replays the first value's binary


def test_a_module_edited_on_disk_and_not_reloaded_is_exported_as_held_and_a_fresh_process_with_the_edited_file_refuses_it(tmp_path):
    """The case of the issue, in real processes with a real edit: A imports, the file is edited, A exports (it compiles and stamps
    what it holds, 2.0); B imports the edited file (3.0) and refuses A's manifest, pinned."""
    route = "an attribute of a module"
    directory, cache, export_dir = tmp_path / "project", tmp_path / "cache", tmp_path / "export"
    identity_routes.write(directory, route)
    a = child(directory, route, "edit_then_export", "2.0", cache=cache, export_dir=export_dir)
    assert a["jit"] == 2.0 and a["pin"]
    assert "+ 1.0" in (directory / "r_cfg.py").read_text()  # the file was edited
    b = child(directory, route, "other", "2.0", cache=cache, export_dir=export_dir, pin=a["pin"])
    assert b["jit"] == 3.0  # B holds the edited file
    assert b["replay"] is None and "stale" in b["refused"] and a["digest"] != b["digest"]
    c = recorded_ir_digest(export_dir)
    assert c == a["digest"]  # what A stamped is what A held, not what the file said


def recorded_ir_digest(export_dir):
    entry = json.loads((export_dir / "manifest.json").read_text())["kernels"][0]
    assert entry["source_digest_kind"] == "tilefish.triton-ir.v1"
    return entry["source_digest"]


# -- determinism: the same kernel gives the same digest wherever it is, and only what the frontend reads changes it ------------------------------


def test_the_digest_is_the_same_in_two_processes_from_two_directories_with_other_line_offsets_and_hash_seeds(tmp_path):
    route = "a helper two levels away"
    cache = tmp_path / "cache"
    first_dir, second_dir = tmp_path / "a", tmp_path / "a" / "much" / "deeper" / "directory"
    identity_routes.write(first_dir, route)
    identity_routes.write(second_dir, route)
    kernel = second_dir / "r_kernel.py"
    kernel.write_text("# padding\n" * 11 + kernel.read_text().replace("tl.store", "# a comment\n\n\n    tl.store"))  # other line numbers, a comment, blank lines
    one = child(first_dir, route, "first", "2.0", cache=cache, export_dir=tmp_path / "e1", seed="1")
    two = child(second_dir, route, "first", "2.0", cache=cache, export_dir=tmp_path / "e2", seed="12345")
    assert one["digest"] == two["digest"] and len(one["digest"]) == 64
    # and so they are one entry of the cache and one binary
    assert one["compiles"] == 1 and two["compiles"] == 0
    other = child(first_dir, route, "first", "3.0", cache=cache, export_dir=tmp_path / "e3", seed="7")
    assert other["digest"] != one["digest"]  # a constant, by any of the routes above, does


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest_of(kernel, spec):
    return runtime._ir_digest_of(spec, kernel, ARCH, {})


BASE = """import triton
import triton.language as tl


@triton.jit
def kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elements
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(y_ptr + offs, x * 2.0, mask=m)
"""
VARIANTS = {
    "a comment": (BASE.replace("    m = offs", "    # a comment\n    m = offs"), True),
    "blank lines": (BASE.replace("    m = offs", "\n\n\n    m = offs"), True),
    "the lines moved down": ("# one\n# two\n# three\n" + BASE, True),
    "a local variable renamed": (BASE.replace("offs", "idx").replace("m = ", "msk = ").replace("mask=m", "mask=msk"), True),
    "a constant changed": (BASE.replace("2.0", "3.0"), False),
    "an operation changed": (BASE.replace("x * 2.0", "x + 2.0"), False),
    "the mask dropped": (BASE.replace(", mask=m)", ")").replace("x_ptr + offs, mask=m", "x_ptr + offs"), False),
    "a function renamed": (BASE.replace("def kernel", "def kernel2"), False),
}


@pytest.mark.parametrize("label", list(VARIANTS))
def test_only_what_the_frontend_reads_changes_the_digest(tmp_path, label):
    from tilefish import Dim, Out, Tensor, f32

    text, same = VARIANTS[label]
    base_dir, other_dir = tmp_path / "base", tmp_path / "x" / "y" / "z"
    base_dir.mkdir()
    other_dir.mkdir(parents=True)
    (base_dir / "kk.py").write_text(BASE)
    (other_dir / "kk.py").write_text(text)
    names = []
    try:
        N = Dim("N", divisible_by=16)

        def spec_of(function_name):
            @tilefish.extern(triton=f"kk:{function_name}", num_warps=4, grid=lambda N: (tilefish.cdiv(N, 256),))
            def op(x_ptr: Tensor[N, f32], y_ptr: Out[Tensor[N, f32]], n_elements: N, *, BLOCK: tilefish.const = 256): ...

            return compile_specs(op)[0]

        names = [f"identity_base_{hashlib.sha1(str(tmp_path).encode()).hexdigest()[:8]}", f"identity_other_{hashlib.sha1(str(tmp_path).encode()).hexdigest()[:8]}"]
        base = load(base_dir / "kk.py", names[0]).kernel
        function = "kernel2" if label == "a function renamed" else "kernel"
        other = getattr(load(other_dir / "kk.py", names[1]), function)
        assert (digest_of(base, spec_of("kernel")) == digest_of(other, spec_of(function))) is same, label
    finally:
        for name in names:
            sys.modules.pop(name, None)
