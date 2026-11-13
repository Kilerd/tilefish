"""tilefish's build cache on a real GPU, through Triton: what a hit is, when it is a miss, and that a damaged entry is never used.

* A rebuild of an unchanged kernel (generated or hand-written, ``build``, ``jit`` or ``export``) is read from the cache and calls
  ``triton.compile`` not at all; the binary is the one that was built, with the ``sha256_code`` of a fresh compile.
* A change to anything the kernel reads, by any route (an edit to a file it reaches, a helper two levels away included, a value the
  process computed), is another IR, so a miss: the key is of the IR the process holds, so an old binary is never served.
* A corrupt or tampered entry is a warning and a rebuild; builds that race write whole entries.
"""

import base64
import hashlib
import importlib
import json
import linecache
import os
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from gpu_util import randn, same_bytes  # noqa: E402
from lower_cases import softmax  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Out, Tensor, TilefishCacheWarning, f32, runtime  # noqa: E402
from tilefish.aot import build  # noqa: E402
from tilefish.dims import cdiv  # noqa: E402

CAP = torch.cuda.get_device_capability()
ARCH = f"sm_{CAP[0] * 10 + CAP[1]}"
REAL_COMPILE = triton.compile


@pytest.fixture(autouse=True)
def no_builds_in_memory():
    """Each test starts as a new process does: no build in memory, so the first ``jit`` of a kernel is the disk cache's to answer."""
    runtime.clear_caches()
    yield
    runtime.clear_caches()


@pytest.fixture
def compiles(monkeypatch):
    """The number of times ``triton.compile`` was called: ``len(compiles)``."""
    calls = []

    def spy(*a, **k):
        calls.append(1)
        return REAL_COMPILE(*a, **k)

    monkeypatch.setattr(triton, "compile", spy)
    return calls


def entries():
    return sorted(pathlib.Path(os.environ["TILEFISH_CACHE_DIR"]).rglob("*.json"))


def softmax_spec():
    low = softmax.lower()
    (spec,) = low.specs()
    return spec, runtime.load_lowered(low)


# -- a generated kernel ----------------------------------------------------------------------------------------------------------


def test_a_rebuild_of_a_generated_kernel_is_a_hit_with_the_binary_of_a_fresh_compile(compiles):
    spec, fn = softmax_spec()
    first = build(spec, arch=ARCH, fn=fn)
    assert len(compiles) == 1 and len(entries()) == 1
    second = build(spec, arch=ARCH, fn=fn)
    assert len(compiles) == 1  # not compiled
    assert second.cubin == first.cubin and second.ptx == first.ptx and second.sha256 == first.sha256
    assert (second.symbol, second.shared, second.num_warps, second.launch, second.options) == (first.symbol, first.shared, first.num_warps, first.launch, first.options)
    fresh = build(spec, arch=ARCH, fn=fn, cache=False)
    assert len(compiles) == 2 and fresh.sha256_code == second.sha256_code == first.sha256_code  # what a compile makes
    assert len(entries()) == 1


def test_jit_after_clearing_the_in_memory_builds_runs_the_cached_binary_with_the_same_bytes(compiles):
    x = randn((7, 1000), torch.float32, 0, scale=3.0)
    first = tilefish.run(softmax, x, mode="jit")
    assert len(compiles) == 1
    runtime.clear_caches()  # a new process would have no builds in memory either
    again = tilefish.run(softmax, x, mode="jit")
    assert len(compiles) == 1 and same_bytes(first, again)


def test_the_cache_serves_an_export_that_replays_identically(compiles, tmp_path):
    x = randn((7, 1000), torch.float32, 0, scale=3.0)
    first = runtime.export(softmax, tmp_path / "a")
    assert len(compiles) == 1
    second = runtime.export(softmax, tmp_path / "b")
    assert len(compiles) == 1  # the second export compiled nothing ...
    assert second.digest == first.digest  # ... and wrote the same manifest
    expected = tilefish.run(softmax, x, mode="jit")
    assert same_bytes(tilefish.run(softmax, x, mode="replay", manifest=second.directory, pin=second.digest), expected)
    assert same_bytes(tilefish.run(softmax, x, mode="replay", manifest=second.directory, trust_directory=True), expected)


def test_another_option_or_architecture_is_another_entry(compiles):
    spec, fn = softmax_spec()
    build(spec, arch=ARCH, fn=fn)
    build(spec, arch="sm_90", fn=fn)
    build(spec, arch=ARCH, fn=fn, options={"enable_fp_fusion": False})
    assert len(compiles) == 3 and len(entries()) == 3
    build(spec, arch=ARCH, fn=fn)
    build(spec, arch="sm_90", fn=fn)
    build(spec, arch=ARCH, fn=fn, options={"enable_fp_fusion": False})
    assert len(compiles) == 3


def scale_kernel(factor):
    """The same kernel (name and signature) with another body."""
    M, D = Dim("M"), Dim("D", 64)

    @tilefish.kernel(name="scale_k")
    def scale_k(x: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return x * factor

    return scale_k


def test_a_generated_kernel_with_another_body_is_a_miss_and_runs_its_own_body(compiles):
    old, new = scale_kernel(2.0), scale_kernel(3.0)
    x = torch.arange(128, dtype=torch.float32, device="cuda").reshape(2, 64)
    assert torch.equal(tilefish.run(old, x, mode="jit"), x * 2)
    assert torch.equal(tilefish.run(new, x, mode="jit"), x * 3)  # the same name and signature: not the old binary
    assert len(compiles) == 2 and len(entries()) == 2
    runtime.clear_caches()
    assert torch.equal(tilefish.run(old, x, mode="jit"), x * 2) and torch.equal(tilefish.run(new, x, mode="jit"), x * 3)
    assert len(compiles) == 2


# -- a hand-written kernel: edits in any file it reaches -------------------------------------------------------------------------


MODULES = ("gc_kernel", "gc_cfg", "gc_helper_a", "gc_helper_b")
FILES = {
    "gc_kernel": (
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "import gc_cfg\n"
        "import gc_helper_a\n"
        "\n"
        "\n"
        "@triton.jit\n"
        "def gc_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):\n"
        "    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
        "    mask = offsets < n_elements\n"
        "    x = tl.load(x_ptr + offsets, mask=mask)\n"
        "    y = gc_helper_a.apply(x) * gc_cfg.SCALE\n"
        "    tl.store(y_ptr + offsets, y, mask=mask)\n"
    ),
    "gc_cfg": "import triton.language as tl\n\nSCALE = tl.constexpr(2.0)\n",
    "gc_helper_a": "import triton\nimport triton.language as tl\n\nimport gc_helper_b\n\n\n@triton.jit\ndef apply(x):\n    return x + gc_helper_b.OFFSET\n",
    "gc_helper_b": "import triton.language as tl\n\nOFFSET = tl.constexpr(0.0)\n",
}
GcN = Dim("GcN", divisible_by=16)


@tilefish.extern(triton="gc_kernel:gc_kernel", num_warps=4, grid=lambda GcN: (cdiv(GcN, 256),))
def gc(x_ptr: Tensor[GcN, f32], y_ptr: Out[Tensor[GcN, f32]], n_elements: GcN, *, BLOCK: tilefish.const = 256): ...


class Project:
    def __init__(self, directory):
        self.directory = directory

    def write(self, **edits):
        for name, text in FILES.items():
            for old, new in edits.get(name, ()):
                text = text.replace(old, new)
            (self.directory / f"{name}.py").write_text(text)

    def load(self):
        for name in MODULES:
            sys.modules.pop(name, None)
        importlib.invalidate_caches()
        linecache.clearcache()
        return importlib.import_module("gc_kernel")


@pytest.fixture
def project(tmp_path, monkeypatch):
    directory = tmp_path / "gc"
    directory.mkdir()
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    yield Project(directory)
    for name in MODULES:
        sys.modules.pop(name, None)


def expected(x, scale="2.0", offset="0.0"):
    return (x + float(offset)) * float(scale)


EDITS = [
    ("a constexpr read as a module attribute", {"gc_cfg": [("tl.constexpr(2.0)", "tl.constexpr(3.0)")]}, {"scale": "3.0"}),
    ("a constant in a helper module two levels away", {"gc_helper_b": [("tl.constexpr(0.0)", "tl.constexpr(1.0)")]}, {"offset": "1.0"}),
    ("the body of the helper", {"gc_helper_a": [("x + gc_helper_b.OFFSET", "x + gc_helper_b.OFFSET + 1.0")]}, {"offset": "1.0"}),
]


@pytest.mark.parametrize("label, edit, change", EDITS, ids=[e[0] for e in EDITS])
def test_an_edit_in_any_file_the_kernel_reaches_is_a_miss_and_runs_the_new_code(compiles, project, label, edit, change):
    x = 1 + torch.arange(64, dtype=torch.float32, device="cuda") / 1024
    project.write()
    project.load()
    assert torch.equal(tilefish.run(gc, x, mode="jit"), expected(x))
    runtime.clear_caches()
    assert torch.equal(tilefish.run(gc, x, mode="jit"), expected(x)) and len(compiles) == 1  # unchanged: the cache
    project.write(**edit)
    project.load()
    runtime.clear_caches()
    assert torch.equal(tilefish.run(gc, x, mode="jit"), expected(x, **change))  # the edit is visible: not the cached old binary
    assert len(compiles) == 2 and len(entries()) == 2
    project.write()
    project.load()
    runtime.clear_caches()
    assert torch.equal(tilefish.run(gc, x, mode="jit"), expected(x)) and len(compiles) == 2  # undone: the first entry again


def test_a_constant_a_module_computes_when_it_is_imported_is_part_of_the_key(compiles, project, monkeypatch):
    """The files do not change; the process holds another constant, so the IR differs, and so does the binary: never the cached one."""
    x = 1 + torch.arange(64, dtype=torch.float32, device="cuda") / 1024
    project.write(gc_cfg=[("tl.constexpr(2.0)", 'tl.constexpr(float(__import__("os").environ["GC_SCALE"]))')])
    for scale, count in (("2.0", 1), ("3.0", 2), ("2.0", 2), ("3.0", 2)):
        monkeypatch.setenv("GC_SCALE", scale)
        project.load()
        runtime.clear_caches()
        assert torch.equal(tilefish.run(gc, x, mode="jit"), expected(x, scale=scale))  # never the binary of the other value
        assert len(compiles) == count
    assert len(entries()) == 2


def test_an_edit_that_was_not_reloaded_is_cached_under_the_ir_the_process_holds(compiles, project):
    """The process holds the old values and the disk has the new: what is compiled, cached and run is what the process holds, under
    its own IR digest, and a process that loads the file gets its own entry."""
    x = 1 + torch.arange(64, dtype=torch.float32, device="cuda") / 1024
    project.write()
    project.load()
    assert torch.equal(tilefish.run(gc, x, mode="jit"), expected(x))
    project.write(gc_helper_b=[("tl.constexpr(0.0)", "tl.constexpr(1.0)")])  # edited, not reloaded
    runtime.clear_caches()
    assert torch.equal(tilefish.run(gc, x, mode="jit"), expected(x)) and len(compiles) == 1  # the entry of what the process holds
    project.load()  # a fresh process: the edited file
    assert torch.equal(tilefish.run(gc, x, mode="jit"), expected(x, offset="1.0")) and len(compiles) == 2
    assert len(entries()) == 2


# -- a damaged entry -------------------------------------------------------------------------------------------------------------------


def damage_cubin(doc):
    raw = bytearray(base64.b64decode(doc["cubin"]))
    raw[len(raw) // 2] ^= 0xFF
    doc["cubin"] = base64.b64encode(bytes(raw)).decode()


def damage_cubin_and_file_digest(doc):
    raw = bytearray(base64.b64decode(doc["cubin"]))
    text = parse_text_section(bytes(raw))
    raw[text] ^= 0xFF  # a byte of the machine code itself
    doc["cubin"] = base64.b64encode(bytes(raw)).decode()
    doc["sha256"] = hashlib.sha256(bytes(raw)).hexdigest()


def parse_text_section(cubin):
    from tilefish._elf import parse_elf

    elf = parse_elf(cubin)
    section = next(s for s in elf.sections if s.name.startswith(".text."))
    return section.offset + section.size // 2


@pytest.mark.parametrize(
    "label, damage, why",
    [
        ("a changed byte", damage_cubin, "sha256 of its cubin"),
        ("a changed byte of the machine code with the file digest made to fit", damage_cubin_and_file_digest, "sha256_code"),
    ],
    ids=["a changed byte", "machine code, file digest fixed up"],
)
def test_a_damaged_real_entry_is_not_used_and_is_rebuilt(compiles, label, damage, why):
    spec, fn = softmax_spec()
    good = build(spec, arch=ARCH, fn=fn)
    (path,) = entries()
    doc = json.loads(path.read_text())
    damage(doc)
    path.write_text(json.dumps(doc))
    with pytest.warns(TilefishCacheWarning, match=why):
        rebuilt = build(spec, arch=ARCH, fn=fn)
    assert len(compiles) == 2 and rebuilt.sha256_code == good.sha256_code
    assert rebuilt.cubin[:4] == b"\x7fELF" and json.loads(path.read_text())["sha256"] == rebuilt.sha256  # rewritten from the fresh build
    build(spec, arch=ARCH, fn=fn)
    assert len(compiles) == 2


def test_a_truncated_entry_is_a_warning_and_a_rebuild(compiles):
    spec, fn = softmax_spec()
    good = build(spec, arch=ARCH, fn=fn)
    (path,) = entries()
    path.write_bytes(path.read_bytes()[:1000])
    with pytest.warns(TilefishCacheWarning, match="not JSON"):
        rebuilt = build(spec, arch=ARCH, fn=fn)
    assert len(compiles) == 2 and rebuilt.sha256_code == good.sha256_code


# -- builds that race ------------------------------------------------------------------------------------------------------------------


CHILD = r"""
import json, sys, warnings
warnings.simplefilter("error")  # a corrupt entry (or an entry that cannot be written) is a TilefishCacheWarning: any is a failure
import triton
import lower_cases
from tilefish import runtime
from tilefish.aot import build

calls = []
real = triton.compile
triton.compile = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
low = lower_cases.softmax.lower()
(spec,) = low.specs()
b = build(spec, arch=sys.argv[1], fn=runtime.load_lowered(low))
print(json.dumps({"code": b.sha256_code, "sha": b.sha256, "compiled": len(calls)}))
"""


def child_env():
    env = dict(os.environ)
    tests = pathlib.Path(__file__).resolve().parent
    src = pathlib.Path(tilefish.__file__).resolve().parent.parent
    env["PYTHONPATH"] = os.pathsep.join([str(src), str(tests), env.get("PYTHONPATH", "")])
    return env


def test_processes_that_build_the_same_kernel_at_once_leave_one_good_entry(tmp_path):
    env = child_env()
    env["TRITON_CACHE_DIR"] = str(tmp_path / "triton")
    procs = [subprocess.Popen([sys.executable, "-c", CHILD, ARCH], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
    done = [(p.communicate(timeout=600), p.returncode) for p in procs]
    results = []
    for (out, err), code in done:
        assert code == 0, err[-3000:]
        results.append(json.loads(out.strip().splitlines()[-1]))
    assert len({r["code"] for r in results}) == 1 and all(r["compiled"] <= 1 for r in results) and sum(r["compiled"] for r in results) >= 1  # they raced: each got a binary
    assert len(entries()) == 1
    last = subprocess.run([sys.executable, "-c", CHILD, ARCH], env=env, capture_output=True, text=True, timeout=600)
    assert last.returncode == 0, last.stderr[-3000:]
    after = json.loads(last.stdout.strip().splitlines()[-1])
    assert after["compiled"] == 0 and after["code"] == results[0]["code"] and after["sha"] in {r["sha"] for r in results}  # a new process: a hit, on a whole entry


# -- a kernel made by a factory function: its constant is in a closure cell ------------------------------------------------------------------

CLO_FILE = (
    "import os\n"
    "\n"
    "import triton\n"
    "import triton.language as tl\n"
    "\n"
    "\n"
    "def make():\n"
    '    S = tl.constexpr(float(os.environ["CLO_S"]))\n'
    "\n"
    "    @triton.jit\n"
    "    def clo_k(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):\n"
    "        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
    "        mask = offsets < n_elements\n"
    "        x = tl.load(x_ptr + offsets, mask=mask)\n"
    "        tl.store(y_ptr + offsets, x * S, mask=mask)\n"
    "\n"
    "    return clo_k\n"
    "\n"
    "\n"
    "clo_k = make()\n"
)
CloN = Dim("CloN", divisible_by=16)


@tilefish.extern(triton="clo_kernel:clo_k", num_warps=4, grid=lambda CloN: (cdiv(CloN, 256),))
def clo(x_ptr: Tensor[CloN, f32], y_ptr: Out[Tensor[CloN, f32]], n_elements: CloN, *, BLOCK: tilefish.const = 256): ...


@pytest.fixture
def closure(tmp_path, monkeypatch):
    directory = tmp_path / "clo"
    directory.mkdir()
    (directory / "clo_kernel.py").write_text(CLO_FILE)
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setenv("CLO_S", "2.0")

    def load():
        sys.modules.pop("clo_kernel", None)
        importlib.invalidate_caches()
        linecache.clearcache()
        runtime.clear_caches()
        return importlib.import_module("clo_kernel")

    yield load, directory
    sys.modules.pop("clo_kernel", None)


def test_a_closure_variable_is_part_of_what_the_cache_and_the_export_say_a_kernel_is(compiles, closure, monkeypatch, tmp_path):
    """The reviewer's repro, on a real GPU: ``CLO_S=2.0`` and then ``CLO_S=3.0`` shared one cache directory and gave y = 2.0 twice."""
    load, _ = closure
    x = 1 + torch.arange(64, dtype=torch.float32, device="cuda") / 1024
    load()
    assert torch.equal(tilefish.run(clo, x, mode="jit"), x * 2.0) and len(compiles) == 1
    first = runtime.export(clo, tmp_path / "two")
    assert len(compiles) == 1  # the cache: the same IR
    monkeypatch.setenv("CLO_S", "3.0")
    load()  # a new process: the same files, another value in the cell
    assert torch.equal(tilefish.run(clo, x, mode="jit"), x * 3.0)  # not the cached binary of 2.0
    assert len(compiles) == 2 and len(entries()) == 2
    # a replay of the manifest made under 2.0, here, would compile another binary than the one in it: refused, with the pin or not
    for kwargs in ({"pin": first.digest}, {"trust_directory": True}):
        with pytest.raises(tilefish.TilefishManifestError, match=r"built from Triton IR with digest .* now compiles to IR with digest .*stale"):
            tilefish.run(clo, x, mode="replay", manifest=first.directory, **kwargs)
    # and the export here stamps the binary of 3.0, which replays here
    second = runtime.export(clo, tmp_path / "three")
    assert second.doc["kernels"][0]["source_digest"] != first.doc["kernels"][0]["source_digest"]
    assert torch.equal(tilefish.run(clo, x, mode="replay", manifest=second.directory, pin=second.digest), x * 3.0)
    monkeypatch.setenv("CLO_S", "2.0")
    load()
    assert torch.equal(tilefish.run(clo, x, mode="replay", manifest=first.directory, pin=first.digest), x * 2.0)  # the same value: accepted
