"""tilefish's own build cache (``tilefish._cache``) and the identity of a binary (``tilefish._ir``): the IR Triton compiles.

No Triton and no GPU: ``fake_compile`` stands in for the frontend and the compiler (its IR is what the kernel function, run on
symbolic arguments, builds, so everything the process holds that the function reads is in it, and comments, blank lines, file paths
and local names are not), and the sources are real files in a temporary package, edited and reloaded for real. ``test_gpu_identity.py``
runs the same routes through Triton on a GPU.
"""

import base64
import hashlib
import importlib
import json
import linecache
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import types
import warnings

import fake_compile
import pytest
from built_fakes import fake_build, scale
from elf_builder import cubin_like

import tilefish
from tilefish import TilefishBuildError, TilefishCacheWarning, TilefishManifestError, _cache, _ir, _source, runtime
from tilefish.aot import build, compile_specs, write_manifest
from tilefish.dims import cdiv

# -- a hand-written kernel in a package of four modules (the helper reads a constant of the helper below it) ----------------------

MODULES = ("ic_kernel", "ic_cfg", "ic_helper_a", "ic_helper_b")
FILES = {
    "ic_kernel": (
        "import ic_cfg\n"
        "import ic_helper_a\n"
        "from fake_jit import jit\n"
        "\n"
        "\n"
        "@jit\n"
        "def kernel(x_ptr, y_ptr, n_elem, BLOCK):\n"
        "    return (ic_helper_a.apply(n_elem) * ic_cfg.SCALE, BLOCK)\n"
    ),
    "ic_cfg": "SCALE = 2.0\n",
    "ic_helper_a": "import ic_helper_b\nfrom fake_jit import jit\n\n\n@jit\ndef apply(x):\n    return x + ic_helper_b.OFFSET\n",
    "ic_helper_b": "OFFSET = 0.0\n",
}

n = tilefish.Dim("n", divisible_by=16)


def extern_of(triton):
    @tilefish.extern(triton=triton, num_warps=4, grid=lambda n: (cdiv(n, 256),))
    def op(x_ptr: tilefish.Tensor[n, tilefish.bf16], y_ptr: tilefish.Out[tilefish.Tensor[n, tilefish.bf16]], n_elem: n, *, BLOCK: tilefish.const = 256): ...

    return op, compile_specs(op)[0]


op, SPEC = extern_of("ic_kernel:kernel")
SM90 = 90


class Project:
    def __init__(self, directory, files=None, modules=None):
        self.directory = directory
        self.modules = MODULES if modules is None else modules
        directory.mkdir(parents=True, exist_ok=True)
        for name, text in (FILES if files is None else files).items():
            self.write(name, text)

    def path(self, name):
        return self.directory / f"{name}.py"

    def write(self, name, text):
        self.path(name).write_text(text)

    def edit(self, name, old, new):
        text = self.path(name).read_text()
        assert old in text, (name, old)
        self.write(name, text.replace(old, new, 1))

    def load(self, entry="ic_kernel", attr="kernel"):
        """Import the modules from their files as they are now (what a new process does), and return the kernel."""
        for name in (*self.modules, entry):
            sys.modules.pop(name, None)
        importlib.invalidate_caches()
        linecache.clearcache()
        return getattr(importlib.import_module(entry), attr)


@pytest.fixture
def world(tmp_path, monkeypatch):
    compiler = fake_compile.install(monkeypatch)
    monkeypatch.setattr(runtime, "_modules", {})  # the generated modules imported under the fake Triton are not for a test that has the real one
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    project = Project(tmp_path / "project")
    monkeypatch.syspath_prepend(str(project.directory))
    yield project, compiler
    for name in MODULES:
        sys.modules.pop(name, None)


def entries():
    return sorted(pathlib.Path(os.environ["TILEFISH_CACHE_DIR"]).rglob("*.json"))


def quiet():
    """A context in which any warning is an error: a build that should be silent."""
    context = warnings.catch_warnings()
    context.__enter__()
    warnings.simplefilter("error")
    return context


def ir_of(kernel, spec=None, arch=SM90, options=None):
    """The IR digest of ``kernel`` as the process holds it (what ``build`` identifies the binary by)."""
    return runtime._ir_digest_of(spec or SPEC, kernel, arch, dict(options or {}))


# -- a hit after a rebuild -------------------------------------------------------------------------------------------------


def test_the_second_build_of_the_same_kernel_is_read_from_the_cache_and_compiles_nothing(world):
    project, compiler = world
    kernel = project.load()
    first = build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 1 and len(entries()) == 1
    assert compiler.frontends == 2  # one for the digest, one inside the compile, which checks that it is the same IR
    second = build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 1 and compiler.frontends == 3  # nothing was compiled: the binary is the cached one; the lookup is one frontend run
    for field in ("cubin", "ptx", "symbol", "shared", "num_warps", "sha256", "sha256_code", "triton_version", "ptxas_version", "ptx_target", "options", "launch", "ir_digest", "arch"):
        assert getattr(second, field) == getattr(first, field), field
    assert second.spec is SPEC and second.ir_digest == ir_of(kernel) and len(second.ir_digest) == 64


def test_a_new_process_hits_too_and_so_does_the_same_project_somewhere_else(world, tmp_path, monkeypatch):
    project, compiler = world
    first = build(SPEC, arch="sm_90", fn=project.load())
    assert compiler.count == 1
    runtime.clear_caches()  # nothing in memory: a fresh import of the files, as a new process has
    again = build(SPEC, arch="sm_90", fn=project.load())
    assert compiler.count == 1 and again.sha256 == first.sha256
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(project.directory, elsewhere)
    monkeypatch.syspath_prepend(str(elsewhere))  # the identity is of the IR, not of where the files are
    moved = build(SPEC, arch="sm_90", fn=project.load())
    assert sys.modules["ic_kernel"].__file__ == str(elsewhere / "ic_kernel.py")
    assert compiler.count == 1 and moved.sha256 == first.sha256


def test_a_hit_returns_what_the_build_that_made_it_returned_and_launches_the_same(world):
    project, compiler = world
    kernel = project.load()
    built = build(SPEC, arch=90, fn=kernel)
    hit = build(SPEC, arch="sm_90", fn=kernel)
    assert hit.sha256_code == built.sha256_code and hit.cubin == built.cubin and hit.abi() == built.abi()
    assert hit.launch.replay_blockers() == [] and hit.shared == compiler.shared


# -- what changes the identity: the IR, and nothing else --------------------------------------------------------------------------------


EDITS = [
    ("ic_cfg", "SCALE = 2.0", "SCALE = 3.0"),  # read as ic_cfg.SCALE
    ("ic_helper_b", "OFFSET = 0.0", "OFFSET = 1.0"),  # two levels away from the kernel's module
    ("ic_helper_a", "x + ic_helper_b.OFFSET", "x - ic_helper_b.OFFSET"),
    ("ic_kernel", "return (ic_helper_a", "return (3 * ic_helper_a"),
]


@pytest.mark.parametrize("name, old, new", EDITS, ids=[e[0] for e in EDITS])
def test_an_edit_that_changes_the_ir_is_a_miss_and_the_edit_undone_is_a_hit_again(world, name, old, new):
    project, compiler = world
    before = build(SPEC, arch="sm_90", fn=project.load())
    project.edit(name, old, new)
    after = build(SPEC, arch="sm_90", fn=project.load())  # reloaded: the process holds the new file
    assert compiler.count == 2 and len(entries()) == 2
    assert after.sha256_code != before.sha256_code and after.ir_digest != before.ir_digest
    project.edit(name, new, old)
    undone = build(SPEC, arch="sm_90", fn=project.load())
    assert compiler.count == 2 and undone.sha256 == before.sha256  # served from the first entry
    assert len(entries()) == 2


FORMATTING = [
    ("a comment", "ic_helper_b", "OFFSET = 0.0", "OFFSET = 0.0  # nothing changes"),
    ("blank lines and a comment above the kernel", "ic_kernel", "@jit\ndef kernel", "# a comment\n\n\n\n@jit\ndef kernel"),
    ("the lines of the kernel moved down", "ic_kernel", "import ic_cfg\n", "# one\n# two\n# three\nimport ic_cfg\n"),
    ("a local variable renamed", "ic_helper_a", "def apply(x):\n    return x + ic_helper_b.OFFSET", "def apply(x):\n    y = x\n    return y + ic_helper_b.OFFSET"),
]


@pytest.mark.parametrize("label, name, old, new", FORMATTING, ids=[f[0] for f in FORMATTING])
def test_an_edit_that_does_not_change_the_ir_is_the_same_binary_and_a_hit(world, label, name, old, new):
    """A comment, blank lines, line offsets and the names of locals are not in the IR, so none of them is an edit of the kernel."""
    project, compiler = world
    before = build(SPEC, arch="sm_90", fn=project.load())
    project.edit(name, old, new)
    after = build(SPEC, arch="sm_90", fn=project.load())
    assert after.ir_digest == before.ir_digest and compiler.count == 1 and after.sha256 == before.sha256


def test_the_ir_digest_is_the_same_from_another_directory_with_other_line_numbers(world, tmp_path, monkeypatch):
    project, _ = world
    digest = ir_of(project.load())
    elsewhere = Project(tmp_path / "a" / "much" / "deeper" / "directory")
    elsewhere.edit("ic_kernel", "import ic_cfg\n", "# padding\n" * 7 + "import ic_cfg\n")
    monkeypatch.syspath_prepend(str(elsewhere.directory))
    kernel = elsewhere.load()
    assert sys.modules["ic_kernel"].__file__.startswith(str(elsewhere.directory)) and kernel.starting_line_number != 6
    assert ir_of(kernel) == digest


# -- an edit that was not reloaded: the process holds what it holds, and what it holds is what is compiled and stamped -------------------


def test_a_module_edited_and_not_reloaded_is_built_as_the_process_holds_it_and_a_fresh_process_refuses_that_manifest(world, tmp_path):
    """Process A imports, the file is edited, A exports: it compiles (and stamps) the old values it holds. A fresh process B, which
    imports the edited file, computes another IR digest and refuses A's manifest."""
    project, compiler = world
    a = project.load()
    held = ir_of(a)
    project.edit("ic_helper_b", "OFFSET = 0.0", "OFFSET = 1.0")  # two levels away, edited, not reloaded
    exported = runtime.export(op, tmp_path / "m", arch=90)  # process A
    (entry,) = exported.doc["kernels"]
    assert entry["source_digest"] == held and entry["source_digest_kind"] == _ir.IR_DIGEST_KIND  # what A held, not what the file says
    a_again = ir_of(a)
    assert a_again == held  # A still holds it
    b = project.load()  # process B: a fresh import of the edited file
    in_b = ir_of(b)
    assert in_b != held
    with pytest.raises(TilefishManifestError, match=rf"built from Triton IR with digest {held}.*now compiles to IR with digest {in_b}.*stale"):
        exported.built_kernel(SPEC, "sm_90", ir_digest=in_b, ir_triton="3.5.1")
    with pytest.raises(TilefishManifestError, match="stale"):  # a pin vouches for the manifest, not for an IR that is known to differ
        exported.built_kernel(SPEC, "sm_90", ir_digest=in_b, ir_triton="3.5.1", pinned=True)
    assert exported.built_kernel(SPEC, "sm_90", ir_digest=held, ir_triton="3.5.1").sha256_code == entry["sha256_code"]  # A itself: fine
    count = compiler.count
    again = runtime.export(op, tmp_path / "m2", arch=90)  # exporting from B stamps B's binary
    assert compiler.count == count + 1 and again.doc["kernels"][0]["source_digest"] == in_b
    assert again.built_kernel(SPEC, "sm_90", ir_digest=in_b, ir_triton="3.5.1").sha256_code != entry["sha256_code"]


def test_a_constant_assigned_to_a_module_at_run_time_is_in_the_ir_and_so_in_the_identity(world):
    """No rule is needed for module state a program assigned: the frontend reads it, so it is part of the digest and of the key."""
    project, compiler = world
    kernel = project.load()
    first = build(SPEC, arch="sm_90", fn=kernel)
    sys.modules["ic_cfg"].SCALE = 5.0
    second = build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 2 and second.ir_digest != first.ir_digest and second.sha256_code != first.sha256_code
    sys.modules["ic_cfg"].SCALE = 2.0
    third = build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 2 and third.sha256 == first.sha256


# -- every route a kernel can read a constant by ---------------------------------------------------------------------------------------


def route_files(route):
    """A kernel package whose constant comes by ``route`` and takes the value of ``R_VALUE`` from the environment where it can."""
    kernel_head = "from fake_jit import jit\n\n\n"
    cfg = "import os\n\nSCALE = float(os.environ['R_VALUE'])\n"
    body = {
        "an attribute": ("import r_cfg\n", "r_cfg.SCALE", cfg),
        "getattr with a literal name": ("import r_cfg\n", 'getattr(r_cfg, "SCALE")', cfg),
        "getattr with a name made at run time": ("import r_cfg\n", 'getattr(r_cfg, "SC" + "ALE")', cfg),
        "a class attribute": ("import r_cfg\n", "r_cfg.Inner.SCALE", "import os\n\n\nclass Inner:\n    SCALE = float(os.environ['R_VALUE'])\n"),
        "an object whose repr hides its state": (
            "import r_cfg\n",
            "r_cfg.HIDDEN.value",
            "import os\n\n\nclass Hidden:\n    def __init__(self, value):\n        self.value = value\n\n    def __repr__(self):\n        return 'Hidden()'\n\n\nHIDDEN = Hidden(float(os.environ['R_VALUE']))\n",
        ),
        "a function that reads the environment when it is compiled": (
            "import r_cfg\n",
            "r_cfg.scale()",
            "import os\n\n\ndef scale():\n    return float(os.environ['R_VALUE'])\n",
        ),
        "a user module named like a standard library module": ("import profile\n", "profile.SCALE", "import os\n\nSCALE = float(os.environ['R_VALUE'])\n"),
        "a helper two levels away": ("import r_cfg\n", "r_cfg.helper(BLOCK)", "import os\n\n\ndef helper(x):\n    return x * float(os.environ['R_VALUE'])\n"),
    }
    imports, expr, cfg_text = body[route]
    files = {"r_kernel": f"{imports}{kernel_head}@jit\ndef kernel(x_ptr, y_ptr, n_elem, BLOCK):\n    return (BLOCK * {expr}, n_elem)\n"}
    if route == "a user module named like a standard library module":
        files["profile"] = cfg_text
        return files, ("r_kernel", "profile")
    files["r_cfg"] = cfg_text
    return files, ("r_kernel", "r_cfg")


ROUTES = [
    "an attribute",
    "getattr with a literal name",
    "getattr with a name made at run time",
    "a class attribute",
    "an object whose repr hides its state",
    "a function that reads the environment when it is compiled",
    "a user module named like a standard library module",
    "a helper two levels away",
]


@pytest.fixture
def routed(world, monkeypatch, tmp_path):
    project, compiler = world
    other = Project(tmp_path / "routes", files={}, modules=())
    monkeypatch.syspath_prepend(str(other.directory))
    created = []

    def prepare(route):
        files, modules = route_files(route)
        for name, text in files.items():
            other.write(name, text)
        other.modules = modules
        created.extend(modules)
        stdlib = {name: sys.modules.pop(name, None) for name in modules}  # a user module that is named like a standard library one
        created.append(("restore", stdlib))
        return other

    yield prepare, compiler
    for item in created:
        if isinstance(item, tuple):
            for name, module in item[1].items():
                sys.modules.pop(name, None)
                if module is not None:
                    sys.modules[name] = module
        else:
            sys.modules.pop(item, None)


@pytest.mark.parametrize("route", ROUTES)
def test_whichever_route_a_kernel_reads_a_constant_by_another_value_is_another_binary_and_never_a_stale_hit(routed, monkeypatch, route):
    """The reviewer's routes, through the cache with two 'processes' (fresh imports) that share one directory and hold other values."""
    prepare, compiler = routed
    project = prepare(route)
    _, spec = extern_of("r_kernel:kernel")
    monkeypatch.setenv("R_VALUE", "2.0")
    first = build(spec, arch="sm_90", fn=project.load("r_kernel"))
    assert compiler.count == 1
    monkeypatch.setenv("R_VALUE", "3.0")
    runtime.clear_caches()
    other = build(spec, arch="sm_90", fn=project.load("r_kernel"))  # the same files, another value in the process
    assert compiler.count == 2 and len(entries()) == 2, route  # a miss, not the first one's binary
    assert other.ir_digest != first.ir_digest and other.sha256_code != first.sha256_code
    monkeypatch.setenv("R_VALUE", "2.0")
    runtime.clear_caches()
    again = build(spec, arch="sm_90", fn=project.load("r_kernel"))
    assert compiler.count == 2 and again.sha256 == first.sha256  # each value has its own entry


@pytest.mark.parametrize("route", ROUTES)
def test_whichever_route_export_stamps_the_value_the_process_holds_and_a_process_with_another_refuses_it(routed, monkeypatch, tmp_path, route):
    """Export in one process, replay-check in another with a pin: the pin is no excuse for another IR."""
    prepare, compiler = routed
    project = prepare(route)
    op_, spec = extern_of("r_kernel:kernel")
    monkeypatch.setenv("R_VALUE", "2.0")
    held = project.load("r_kernel")
    manifest = runtime.export(op_, tmp_path / "m", arch=90)
    digest_two = ir_of(held, spec)
    (entry,) = manifest.doc["kernels"]
    assert entry["source_digest"] == digest_two
    assert manifest.built_kernel(spec, "sm_90", ir_digest=digest_two, ir_triton="3.5.1").sha256_code == entry["sha256_code"]
    monkeypatch.setenv("R_VALUE", "3.0")
    digest_three = ir_of(project.load("r_kernel"), spec)
    assert digest_three != digest_two
    for pinned in (False, True):
        with pytest.raises(TilefishManifestError, match="stale"):
            manifest.built_kernel(spec, "sm_90", ir_digest=digest_three, ir_triton="3.5.1", pinned=pinned)


def test_a_name_that_is_a_user_module_of_the_standard_library_s_name_is_not_left_out(routed, monkeypatch):
    """The old digest left out any module whose top-level package had a standard library name: a user ``profile.py`` was not read."""
    prepare, compiler = routed
    project = prepare("a user module named like a standard library module")
    monkeypatch.setenv("R_VALUE", "2.0")
    _, spec = extern_of("r_kernel:kernel")
    kernel = project.load("r_kernel")
    assert sys.modules["profile"].__file__ == str(project.path("profile"))  # the user's file, not the standard library's
    before = ir_of(kernel, spec)
    project.edit("profile", "SCALE = float(os.environ['R_VALUE'])", "SCALE = float(os.environ['R_VALUE']) + 1")
    assert ir_of(project.load("r_kernel"), spec) != before


def test_a_kernel_made_by_a_factory_reads_its_constants_from_closure_cells_and_they_are_in_the_ir(world, monkeypatch):
    project, compiler = world
    project.write(
        "clo_kernel",
        "import os\n\nfrom fake_jit import jit\n\n\ndef make():\n    S = float(os.environ['CLO_S'])\n\n    @jit\n"
        "    def k(x_ptr, y_ptr, n_elem, BLOCK):\n        return (BLOCK * S, n_elem)\n\n    return k\n\n\nk = make()\n",
    )
    _, spec = extern_of("clo_kernel:k")
    try:
        monkeypatch.setenv("CLO_S", "2.0")
        first = build(spec, arch="sm_90", fn=project.load("clo_kernel", "k"))
        monkeypatch.setenv("CLO_S", "3.0")
        other = build(spec, arch="sm_90", fn=project.load("clo_kernel", "k"))
        assert compiler.count == 2 and other.ir_digest != first.ir_digest
    finally:
        sys.modules.pop("clo_kernel", None)


# -- the key ---------------------------------------------------------------------------------------------------------------------------


def test_the_arch_the_options_and_the_toolchain_are_part_of_the_key(world, monkeypatch):
    project, compiler = world
    kernel = project.load()
    build(SPEC, arch="sm_90", fn=kernel)
    steps = [
        ("another arch", lambda: build(SPEC, arch="sm_100", fn=kernel)),
        ("an option", lambda: build(SPEC, arch="sm_90", fn=kernel, options={"enable_fp_fusion": False})),
        ("the ptxas version", lambda: (monkeypatch.setattr(sys.modules["tilefish._build"], "_ptxas_version", lambda p: "ptxas 12.10"), build(SPEC, arch="sm_90", fn=kernel))),
        ("the triton version", lambda: (monkeypatch.setattr(sys.modules["triton"], "__version__", "3.6.0"), build(SPEC, arch="sm_90", fn=kernel))),
    ]
    for count, (label, step) in enumerate(steps, start=2):
        step()
        assert compiler.count == count, label  # each is a miss
        assert len(entries()) == count, label
    monkeypatch.setattr(sys.modules["tilefish._build"], "_ptxas_version", lambda p: "Cuda compilation tools, release 12.9, V12.9.86")
    monkeypatch.setattr(sys.modules["triton"], "__version__", "3.5.1")
    build(SPEC, arch="sm_90", fn=kernel)
    build(SPEC, arch="sm_100", fn=kernel)
    build(SPEC, arch="sm_90", fn=kernel, options={"enable_fp_fusion": False})
    assert compiler.count == len(steps) + 1  # the first three are hits; the fourth entry was made for 3.6.0 and the 3.5.1 ones remain


def test_an_environment_variable_of_the_codegen_families_is_part_of_the_key_and_one_that_is_not_codegen_is_not(world, monkeypatch):
    project, compiler = world
    kernel = project.load()
    build(SPEC, arch="sm_90", fn=kernel)
    monkeypatch.setenv("TRITON_CACHE_DIR", "/somewhere/else")  # where Triton keeps its cache: not code
    monkeypatch.setenv("TRITON_PTXAS_PATH", "/other/ptxas")  # a tool whose version is in the key
    build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 1
    for name in ("TRITON_DEFAULT_FP_FUSION", "TRITON_LIBDEVICE_PATH", "TRITON_MOCK_PTX_VERSION", "TRITON_KERNEL_OVERRIDE", "DISABLE_PTXAS_OPT", "MLIR_ENABLE_DUMP", "USE_IR_LOC", "TRITON_A_KNOB_OF_A_LATER_TRITON"):
        before = compiler.count
        monkeypatch.setenv(name, "1")
        build(SPEC, arch="sm_90", fn=kernel)
        assert compiler.count == before + 1, name  # another binary may come of it: another key
        monkeypatch.delenv(name)
    build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 1 + 8  # and with none of them set it is the first entry again


def test_a_knob_that_a_program_set_without_the_environment_is_part_of_the_key(world, monkeypatch):
    project, compiler = world
    kernel = project.load()
    build(SPEC, arch="sm_90", fn=kernel)
    knobs = sys.modules["triton"].knobs
    monkeypatch.setattr(knobs.language, "default_fp_fusion", False)
    build(SPEC, arch="sm_90", fn=kernel)
    monkeypatch.setattr(knobs.nvidia, "mock_ptx_version", "99")
    build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 3


def test_the_libdevice_file_is_part_of_the_key_by_its_content(world, monkeypatch, tmp_path):
    project, compiler = world
    kernel = project.load()
    libdevice = tmp_path / "libdevice.10.bc"
    libdevice.write_bytes(b"one")
    monkeypatch.setattr(sys.modules["triton"].knobs.nvidia, "libdevice_path", str(libdevice))
    build(SPEC, arch="sm_90", fn=kernel)
    build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 1
    libdevice.write_bytes(b"two")  # the same path, another file
    build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 2
    libdevice.unlink()
    build(SPEC, arch="sm_90", fn=kernel)  # an unreadable one is a key of its own, not a crash
    assert compiler.count == 3


def test_triton_s_kernel_override_is_refused_because_the_binary_would_not_be_of_the_ir(world, monkeypatch):
    project, compiler = world
    kernel = project.load()
    monkeypatch.setattr(sys.modules["triton"].knobs.compilation, "override", True)
    with pytest.raises(TilefishBuildError, match=r"kernel override is on \(TRITON_KERNEL_OVERRIDE\).*not be compiled from the IR"):
        build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 0 and entries() == []


def test_a_binary_is_only_recorded_under_the_digest_of_the_ir_it_was_compiled_from(world):
    """If the frontend gave another module the second time (it is not deterministic), the compile is not of the IR the build is identified by."""
    project, compiler = world
    kernel = project.load()
    compiler.second_ir = "module { another }"
    with pytest.raises(TilefishBuildError, match=r"the IR Triton compiled is not the IR whose digest [0-9a-f]{64} the build was identified by"):
        build(SPEC, arch="sm_90", fn=kernel)
    assert entries() == []  # nothing was cached


def test_the_ir_digest_a_caller_expects_is_checked(world):
    project, compiler = world
    kernel = project.load()
    digest = ir_of(kernel)
    assert build(SPEC, arch="sm_90", fn=kernel, ir_digest=digest).ir_digest == digest
    with pytest.raises(TilefishBuildError, match=rf"has digest {digest}, but {'0' * 64} was expected"):
        build(SPEC, arch="sm_90", fn=kernel, ir_digest="0" * 64)


def test_a_frontend_that_cannot_build_the_kernel_is_a_build_error_with_nothing_cached(world):
    project, compiler = world
    kernel = project.load()
    compiler.fail_frontend = ValueError("an unsupported construct")
    with pytest.raises(TilefishBuildError, match=r"Triton could not compile 'op' \(ic_kernel:kernel\) for sm_90: ValueError: an unsupported construct"):
        build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 0 and entries() == []


def test_another_declaration_of_the_same_kernel_with_the_same_ir_shares_the_binary(world):
    """The spec is not in the key: the IR, the options and the toolchain are what make a binary. (The entry is checked against the spec on read.)"""
    project, compiler = world
    kernel = project.load()
    build(SPEC, arch="sm_90", fn=kernel)
    other_op, other = extern_of("ic_kernel:kernel")
    assert other.stem == SPEC.stem  # the same declaration shape: the same binary
    build(other, arch="sm_90", fn=kernel)
    assert compiler.count == 1


def test_the_cache_can_be_switched_off_per_build_and_by_the_environment(world, monkeypatch):
    project, compiler = world
    kernel = project.load()
    build(SPEC, arch="sm_90", fn=kernel)
    build(SPEC, arch="sm_90", fn=kernel, cache=False)  # does not read the entry that is there ...
    build(SPEC, arch="sm_90", fn=kernel, cache=False)
    assert compiler.count == 3 and len(entries()) == 1  # ... and does not write
    directory = os.environ["TILEFISH_CACHE_DIR"]
    monkeypatch.setenv("TILEFISH_CACHE_DIR", "")
    assert _cache.cache_directory() is None
    build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 4 and len(list(pathlib.Path(directory).rglob("*.json"))) == 1


def test_where_the_cache_is(monkeypatch, tmp_path):
    monkeypatch.setenv("TILEFISH_CACHE_DIR", str(tmp_path / "x"))
    assert _cache.cache_directory() == tmp_path / "x"
    monkeypatch.delenv("TILEFISH_CACHE_DIR")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert _cache.cache_directory() == tmp_path / "home" / ".cache" / "tilefish"
    monkeypatch.setenv("TILEFISH_CACHE_DIR", "~/elsewhere")
    assert _cache.cache_directory() == tmp_path / "home" / "elsewhere"


# -- a generated kernel: its spec carries the sha256 of its source, and its IR is recorded as well ----------------------------------------


def scale_kernel(factor):
    """The same kernel (name and signature) with another body."""
    M, D = tilefish.Dim("M"), tilefish.Dim("D", 64)

    @tilefish.kernel(name="scale_k")
    def scale_k(x: tilefish.Tensor[M, D, tilefish.f32]) -> tilefish.Tensor[M, D, tilefish.f32]:
        return x * factor

    return scale_k


def test_a_generated_kernel_hits_and_another_body_is_a_miss_and_its_ir_is_recorded_too(world):
    _, compiler = world
    old, new = scale_kernel(2.0).lower(), scale_kernel(3.0).lower()
    (spec_old,), (spec_new,) = old.specs(), new.specs()
    assert spec_old.source_sha256 != spec_new.source_sha256
    a = build(spec_old, arch="sm_90", fn=runtime.load_lowered(old))
    assert len(a.ir_digest) == 64 and compiler.count == 1  # (a lowered kernel has its source_sha256 in the spec, and its IR digest as well)
    again = build(spec_old, arch="sm_90", fn=runtime.load_lowered(old))
    assert compiler.count == 1 and again.sha256 == a.sha256 and again.ir_digest == a.ir_digest
    b = build(spec_new, arch="sm_90", fn=runtime.load_lowered(new))
    assert compiler.count == 2 and b.sha256_code != a.sha256_code and b.ir_digest != a.ir_digest
    entry = tilefish._build._kernel_entry(a)
    assert entry["source_sha256"] == spec_old.source_sha256 and entry["source_digest"] == a.ir_digest and entry["source_digest_kind"] == _ir.IR_DIGEST_KIND


# -- a corrupt entry is detected, not used, and rebuilt ----------------------------------------------------------------------------------


def rewrite(path, change):
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc, sort_keys=True) + "\n")


def flip_cubin(doc):  # a bit of the cubin changes, the digests are as they were
    raw = bytearray(base64.b64decode(doc["cubin"]))
    raw[-1] ^= 0xFF
    doc["cubin"] = base64.b64encode(bytes(raw)).decode()


def other_code(doc):  # another machine code with its own file digest, but the recorded machine-code digest
    other = cubin_like(code=b"\x07" * 64)
    doc["cubin"] = base64.b64encode(other).decode()
    doc["sha256"] = hashlib.sha256(other).hexdigest()


def other_abi(doc):  # a PTX that takes other parameters, with its own digest
    doc["ptx"] = doc["ptx"].replace(".param .u32", ".param .u64", 1)
    doc["ptx_sha256"] = hashlib.sha256(doc["ptx"].encode()).hexdigest()


CORRUPTIONS = [
    ("a bit of the cubin", lambda p: rewrite(p, flip_cubin), "sha256 of its cubin"),
    ("the machine code with a fixed-up file digest", lambda p: rewrite(p, other_code), "sha256_code"),
    ("the PTX text", lambda p: rewrite(p, lambda d: d.update(ptx=d["ptx"] + "// edited\n")), "sha256 of its PTX"),
    ("the ABI in the PTX", lambda p: rewrite(p, other_abi), "does not have the ABI"),
    ("a truncated file", lambda p: p.write_bytes(p.read_bytes()[: len(p.read_bytes()) // 2]), "not JSON"),
    ("an empty file", lambda p: p.write_bytes(b""), "not JSON"),
    ("something that is not text", lambda p: p.write_bytes(b"\xff\xfe\x00 not an entry"), "not JSON"),
    ("a JSON list", lambda p: p.write_text("[1, 2]"), "not a JSON object"),
    ("another format", lambda p: rewrite(p, lambda d: d.update(format="tilefish.build-cache.v0")), "format"),
    ("the entry of another key", lambda p: rewrite(p, lambda d: d.update(key="0" * 64)), "another key"),
    ("a cubin that is not base64", lambda p: rewrite(p, lambda d: d.update(cubin="not base64!!")), "not base64"),
    ("a cubin that is not an ELF file", lambda p: rewrite(p, lambda d: d.update(cubin=base64.b64encode(b"hello").decode(), sha256=hashlib.sha256(b"hello").hexdigest())), "not a valid ELF"),
    ("a missing field", lambda p: rewrite(p, lambda d: d.pop("shared")), "'shared'"),
    ("other options", lambda p: rewrite(p, lambda d: d["options"].update(num_warps=8)), "compile options"),
    ("the toolchain it was built with", lambda p: rewrite(p, lambda d: d.update(ptxas_version="ptxas 1.0")), "ptxas_version"),
    ("the IR digest it was made for", lambda p: rewrite(p, lambda d: d.update(ir_digest="0" * 64)), "ir_digest"),
    ("malformed launch attributes", lambda p: rewrite(p, lambda d: d.update(launch={"num_ctas": 1})), "launch attributes"),
]


@pytest.mark.parametrize("label, corrupt, why", CORRUPTIONS, ids=[c[0] for c in CORRUPTIONS])
def test_a_corrupt_entry_is_a_miss_with_a_warning_and_is_rebuilt_and_rewritten(world, label, corrupt, why):
    project, compiler = world
    kernel = project.load()
    good = build(SPEC, arch="sm_90", fn=kernel)
    (path,) = entries()
    corrupt(path)
    damaged = path.read_bytes()
    with pytest.warns(TilefishCacheWarning, match=r"build cache entry .* is corrupt or stale .*" + why) as seen:
        rebuilt = build(SPEC, arch="sm_90", fn=kernel)
    assert len(seen) == 1 and str(path) in str(seen[0].message)
    assert compiler.count == 2 and rebuilt.sha256_code == good.sha256_code and rebuilt.cubin == good.cubin  # compiled again, not the damaged bytes
    assert path.read_bytes() != damaged  # the entry was replaced
    context = quiet()
    try:
        hit = build(SPEC, arch="sm_90", fn=kernel)  # and is a hit now, without a warning
    finally:
        context.__exit__(None, None, None)
    assert compiler.count == 2 and hit.sha256 == good.sha256


def test_an_entry_that_cannot_be_read_is_a_miss_with_a_warning(world):
    project, compiler = world
    kernel = project.load()
    build(SPEC, arch="sm_90", fn=kernel)
    (path,) = entries()
    path.unlink()
    path.mkdir()  # a directory where the entry should be: reading it fails with something other than "not found"
    with pytest.warns(TilefishCacheWarning) as seen:
        build(SPEC, arch="sm_90", fn=kernel)  # (and writing the entry fails too, with a warning of its own)
    assert any("cannot read the tilefish build cache entry" in str(w.message) for w in seen)
    assert compiler.count == 2


def test_a_cache_directory_that_cannot_be_written_never_fails_a_build_and_warns_once(world, tmp_path, monkeypatch):
    project, compiler = world
    kernel = project.load()
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the cache directory should be")
    monkeypatch.setenv("TILEFISH_CACHE_DIR", str(blocked))
    with pytest.warns(TilefishCacheWarning, match="cannot write the tilefish build cache"):
        first = build(SPEC, arch="sm_90", fn=kernel)
    context = quiet()
    try:
        second = build(SPEC, arch="sm_90", fn=kernel)  # one warning per directory, not one per build
    finally:
        context.__exit__(None, None, None)
    assert compiler.count == 2 and first.sha256_code == second.sha256_code


def test_a_symbolic_link_where_an_entry_goes_is_not_written_through(world, tmp_path):
    project, compiler = world
    kernel = project.load()
    build(SPEC, arch="sm_90", fn=kernel)
    (path,) = entries()
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    path.unlink()
    path.symlink_to(victim)
    with pytest.warns(TilefishCacheWarning):  # the entry that is read through the link is not an entry
        build(SPEC, arch="sm_90", fn=kernel)
    assert victim.read_text() == "precious"


# -- the entry and the checks that read it, without a build ---------------------------------------------------------------------------------


def valid_build(spec, **kw):
    """A ``fake_build`` whose PTX has the entry point the spec's ABI asks for (what a read checks)."""
    return fake_build(spec, ptx=fake_compile.ptx_of(spec.signature, 90), **kw)


def key_args(built):
    return dict(
        arch=built.arch,
        options=built.options,
        triton_version=built.triton_version,
        ptxas_version=built.ptxas_version,
        ir_digest=built.ir_digest,
    )


def test_an_entry_reads_back_as_the_build_it_was_made_from(tmp_path):
    (spec,) = compile_specs(scale)
    built = valid_build(spec)
    cache = _cache.BuildCache(tmp_path)
    key = "ab" * 32
    assert cache.load(key, spec, **key_args(built)) is None  # a miss is silent
    assert cache.store(key, built) and cache.path_of(key).parent.name == "ab"
    got = cache.load(key, spec, **key_args(built))
    for field in ("cubin", "ptx", "symbol", "shared", "num_warps", "sha256", "sha256_code", "ptx_target", "options", "launch", "arch"):
        assert getattr(got, field) == getattr(built, field), field


def test_the_key_depends_on_every_part_of_it(tmp_path):
    base = dict(arch=90, ir_digest="a" * 64, triton_version="3.5.1", ptxas_version="12.9", options={"num_warps": 4}, environment={}, files={})
    key = _cache.cache_key(**base)
    assert key == _cache.cache_key(**base) and len(key) == 64
    for change in (
        {"arch": 100},
        {"ir_digest": "b" * 64},
        {"triton_version": "3.5.2"},
        {"ptxas_version": "12.10"},
        {"options": {"num_warps": 8}},
        {"options": {"num_warps": 4, "enable_fp_fusion": False}},
        {"environment": {"env": {"TRITON_X": "1"}}},
        {"files": {"extern_libs.libdevice": "c" * 64}},
    ):
        assert _cache.cache_key(**{**base, **change}) != key, change


# -- atomic writes: several writers and readers of one entry ------------------------------------------------------------------------------


def test_threads_that_write_and_read_one_entry_never_see_a_torn_one(tmp_path):
    (spec,) = compile_specs(scale)
    variants = [valid_build(spec, cubin=cubin_like(code=bytes([i]) * 4096)) for i in range(3)]
    valid = {v.sha256 for v in variants}
    cache = _cache.BuildCache(tmp_path)
    key = "cd" * 32
    errors = []

    def work(i):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # a corrupt entry is a warning: any is a failure
                for _ in range(150):
                    assert cache.store(key, variants[i % 3])
                    got = cache.load(key, spec, **key_args(variants[0]))
                    assert got is not None and got.sha256 in valid
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:1]
    assert [p.name for p in sorted(cache.path_of(key).parent.iterdir())] == [f"{key}.json"]  # no temporary file is left behind


CHILD = """
import sys, warnings
warnings.simplefilter("error")
import fake_compile
from built_fakes import fake_build, scale
from elf_builder import cubin_like
from tilefish import _cache
from tilefish.aot import compile_specs

(spec,) = compile_specs(scale)
variants = [fake_build(spec, ptx=fake_compile.ptx_of(spec.signature, 90), cubin=cubin_like(code=bytes([i]) * 4096)) for i in range(3)]
valid = {v.sha256 for v in variants}
cache = _cache.BuildCache(sys.argv[1])
key = "ef" * 32
me = variants[int(sys.argv[2]) % 3]
args = dict(arch=me.arch, options=me.options, triton_version=me.triton_version, ptxas_version=me.ptxas_version, ir_digest=me.ir_digest)
for _ in range(100):
    assert cache.store(key, me)
    got = cache.load(key, spec, **args)
    assert got is not None and got.sha256 in valid
print("ok")
"""


def test_processes_that_write_and_read_one_entry_never_see_a_torn_one(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(pathlib.Path(__file__).parent), str(pathlib.Path(tilefish.__file__).resolve().parent.parent)]))
    procs = [subprocess.Popen([sys.executable, "-c", CHILD, str(tmp_path), str(i)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(6)]
    results = [p.communicate(timeout=120) + (p.returncode,) for p in procs]
    for out, err, code in results:
        assert code == 0 and out.strip() == "ok", err[-2000:]
    assert [p.name for p in sorted(next(tmp_path.joinpath("builds").iterdir()).iterdir())] == ["ef" * 32 + ".json"]


def test_concurrent_builds_of_one_kernel_all_succeed_with_one_valid_entry(world):
    project, compiler = world
    kernel = project.load()
    results, errors = [], []

    def work():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                results.append(build(SPEC, arch="sm_90", fn=kernel))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:1]
    assert len(results) == 8 and len({r.sha256 for r in results}) == 1 and 1 <= compiler.count <= 8
    assert len(entries()) == 1
    count = compiler.count
    build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == count  # and the entry is good


def test_export_builds_through_the_cache_and_writes_the_same_manifest_entry(world, tmp_path):
    """``export`` uses ``build``: its second export of the same kernel compiles nothing and the binary is the first one's."""
    project, compiler = world
    kernel = project.load()
    first = build(SPEC, arch="sm_90", fn=kernel)

    write_manifest([first], tmp_path / "a" / "manifest.json", cubins=True)
    second = build(SPEC, arch="sm_90", fn=kernel)
    write_manifest([second], tmp_path / "b" / "manifest.json", cubins=True)
    assert compiler.count == 1
    assert tilefish.Manifest.load(tmp_path / "a").digest == tilefish.Manifest.load(tmp_path / "b").digest


# -- build(spec, fn=...) compiles the function the spec names, and no other ----------------------------------------------------------------

TWO = (
    "from fake_jit import jit\n"
    "\n"
    "\n"
    "@jit\n"
    "def ka(x_ptr, y_ptr, n_elem, BLOCK):\n"
    "    return (BLOCK * 2.0, n_elem)\n"
    "\n"
    "\n"
    "@jit\n"
    "def kb(x_ptr, y_ptr, n_elem, BLOCK):\n"
    "    return (BLOCK * 3.0, n_elem)\n"
    "\n"
    "\n"
    "alias = ka\n"
)


@pytest.fixture
def two(world):
    project, compiler = world
    project.write("m5_kernels", TWO)

    def load():
        sys.modules.pop("m5_kernels", None)
        importlib.invalidate_caches()
        linecache.clearcache()
        return importlib.import_module("m5_kernels")

    yield load, compiler
    sys.modules.pop("m5_kernels", None)


def test_another_kernel_with_the_same_parameters_is_refused_and_does_not_poison_the_cache(two):
    """The reviewer's repro: ``build(spec_ka, fn=kb)`` was accepted, and the cache served kb's cubin to a later ``build(spec_ka, fn=ka)``."""
    load, compiler = two
    module = load()
    spec_ka = extern_of("m5_kernels:ka")[1]
    with pytest.raises(TilefishBuildError, match=r"the function passed as fn is 'm5_kernels:kb', but the compile spec '.*' is for 'm5_kernels:ka'.*nothing is compiled or cached"):
        build(spec_ka, arch="sm_90", fn=module.kb)
    assert compiler.count == 0 and entries() == []  # nothing was compiled, and nothing was cached under the key of ka
    right = build(spec_ka, arch="sm_90", fn=module.ka)
    assert compiler.count == 1  # a miss: there was nothing of kb's to serve
    # kb's own spec builds kb, and the two are different binaries with different digests
    spec_kb = extern_of("m5_kernels:kb")[1]
    other = build(spec_kb, arch="sm_90", fn=module.kb)
    assert other.sha256_code != right.sha256_code and compiler.count == 2  # (one file; two kernels: two IRs)


def test_the_function_the_spec_names_is_accepted_however_it_was_reached(two):
    load, compiler = two
    module = load()
    spec = extern_of("m5_kernels:ka")[1]
    build(spec, arch="sm_90", fn=module.ka)
    build(spec, arch="sm_90", fn=module.alias)  # the same object under another name
    build(spec, arch="sm_90")  # imported from spec.triton
    assert compiler.count == 1
    reloaded = load().ka  # a new import of the same file: another object, defined under the same module and name
    assert reloaded is not module.ka
    build(spec, arch="sm_90", fn=reloaded)
    assert compiler.count == 1  # and a hit: the same source


def test_a_function_of_the_same_name_in_another_module_is_refused(two, world):
    load, compiler = two
    project, _ = world
    project.write("m5_other", TWO)
    try:
        other = importlib.import_module("m5_other")
        with pytest.raises(TilefishBuildError, match="'m5_other:ka', but the compile spec '.*' is for 'm5_kernels:ka'"):
            build(extern_of("m5_kernels:ka")[1], arch="sm_90", fn=other.ka)
        assert compiler.count == 0 and entries() == []
    finally:
        sys.modules.pop("m5_other", None)


def test_a_spec_that_names_a_module_that_cannot_be_imported_is_built_from_the_function_defined_there(two):
    load, compiler = two
    module = load()
    spec = extern_of("not_installed_anywhere:ka")[1]
    with pytest.raises(TilefishBuildError, match="'m5_kernels:ka', but the compile spec '.*' is for 'not_installed_anywhere:ka'"):
        build(spec, arch="sm_90", fn=module.ka)
    assert compiler.count == 0


def named_kernel(name, factor):
    M, D = tilefish.Dim("M"), tilefish.Dim("D", 64)

    @tilefish.kernel(name=name)
    def k(x: tilefish.Tensor[M, D, tilefish.f32]) -> tilefish.Tensor[M, D, tilefish.f32]:
        return x * factor

    return k


def test_a_generated_kernel_is_told_by_its_name_and_the_sha256_of_its_source(world):
    _, compiler = world
    low1, low2, other_body = named_kernel("one_k", 2.0).lower(), named_kernel("two_k", 2.0).lower(), named_kernel("one_k", 3.0).lower()
    (spec1,) = low1.specs()
    with pytest.raises(TilefishBuildError, match=r"the function passed as fn is '.*:tf_two_k', but the compile spec 'one_k' is for '.*:tf_one_k'"):
        build(spec1, arch="sm_90", fn=runtime.load_lowered(low2))
    assert compiler.count == 0 and entries() == []
    # the same name with another body is told by the sha256 of the file it was imported from
    with pytest.raises(TilefishBuildError, match=r"whose sha256 is .*, but the compile spec '.*' is for Triton source with sha256"):
        build(spec1, arch="sm_90", fn=runtime.load_lowered(other_body))
    assert compiler.count == 0 and entries() == []
    assert build(spec1, arch="sm_90", fn=runtime.load_lowered(low1)).spec is spec1 and compiler.count == 1


# -- warnings that are errors must not fail a build that compiled fine ------------------------------------------------------------------------


@pytest.fixture
def warnings_are_errors():
    """``python -W error``: every warning raised from where it is issued (a TilefishCacheWarning included)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def cache_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "tilefish.cache" and r.levelno == logging.WARNING]


@pytest.mark.parametrize("label, corrupt, why", CORRUPTIONS[:6], ids=[c[0] for c in CORRUPTIONS[:6]])
def test_a_corrupt_entry_under_w_error_is_a_miss_and_the_build_succeeds_and_the_message_is_logged(world, label, corrupt, why, caplog):
    project, compiler = world
    kernel = project.load()
    good = build(SPEC, arch="sm_90", fn=kernel)
    (path,) = entries()
    corrupt(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with caplog.at_level(logging.WARNING, logger="tilefish.cache"):
            rebuilt = build(SPEC, arch="sm_90", fn=kernel)  # raised TilefishCacheWarning here, after the compile
    assert compiler.count == 2 and rebuilt.sha256_code == good.sha256_code
    messages = cache_messages(caplog)
    assert len(messages) == 1 and str(path) in messages[0] and "corrupt or stale" in messages[0] and why in messages[0]


def test_an_unreadable_entry_and_an_unwritable_directory_under_w_error_never_fail_a_build(world, tmp_path, monkeypatch, caplog, warnings_are_errors):
    project, compiler = world
    kernel = project.load()
    with caplog.at_level(logging.WARNING, logger="tilefish.cache"):
        build(SPEC, arch="sm_90", fn=kernel)
        (path,) = entries()
        path.unlink()
        path.mkdir()  # a directory where the entry should be
        build(SPEC, arch="sm_90", fn=kernel)  # cannot read it, cannot write it
        blocked = tmp_path / "blocked"
        blocked.write_text("a file where the cache directory should be")
        monkeypatch.setenv("TILEFISH_CACHE_DIR", str(blocked))
        build(SPEC, arch="sm_90", fn=kernel)  # cannot write the directory
    assert compiler.count == 3
    text = "\n".join(cache_messages(caplog))
    assert "cannot read the tilefish build cache entry" in text and "cannot write the tilefish build cache" in text


def test_something_unexpected_that_checking_an_entry_raises_is_a_miss_not_a_failed_build(world, monkeypatch, caplog):
    project, compiler = world
    kernel = project.load()
    build(SPEC, arch="sm_90", fn=kernel)

    def boom(*args, **kwargs):
        raise ValueError("a bug in how an entry is read")

    monkeypatch.setattr(_cache, "decode", boom)
    with pytest.warns(TilefishCacheWarning, match=r"could not be checked \(ValueError: a bug in how an entry is read\)"):
        build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with caplog.at_level(logging.WARNING, logger="tilefish.cache"):
            build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 3 and "could not be checked" in "".join(cache_messages(caplog))


def test_something_unexpected_that_writing_an_entry_raises_is_a_warning_not_a_failed_build(world, monkeypatch, tmp_path):
    project, compiler = world
    kernel = project.load()

    def boom(key, built):
        raise TypeError("a bug in how an entry is made")

    monkeypatch.setattr(_cache, "encode", boom)
    monkeypatch.setenv("TILEFISH_CACHE_DIR", str(tmp_path / "other-cache"))
    with pytest.warns(TilefishCacheWarning, match=r"cannot write the tilefish build cache .*TypeError: a bug in how an entry is made"):
        built = build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 1 and built.cubin


def test_a_warnings_hook_that_fails_does_not_fail_a_build_either(world, monkeypatch, caplog):
    project, compiler = world
    kernel = project.load()
    build(SPEC, arch="sm_90", fn=kernel)
    (path,) = entries()
    path.write_bytes(b"")

    def broken(*args, **kwargs):
        raise RuntimeError("the hook is broken")

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        monkeypatch.setattr(warnings, "showwarning", broken)
        with caplog.at_level(logging.WARNING, logger="tilefish.cache"):
            build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 2 and "corrupt or stale" in "".join(cache_messages(caplog))


def test_the_warning_is_still_a_warning_with_the_default_filters_and_points_at_the_caller_of_build(world):
    project, compiler = world
    kernel = project.load()
    build(SPEC, arch="sm_90", fn=kernel)
    (path,) = entries()
    path.write_bytes(b"")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        build(SPEC, arch="sm_90", fn=kernel)
    cache = [w for w in seen if issubclass(w.category, TilefishCacheWarning)]
    assert len(cache) == 1 and pathlib.Path(cache[0].filename).name == "test_build_cache.py"  # where build was called from, not the cache


def test_a_logger_that_fails_is_not_a_failed_build_either(world, monkeypatch):
    project, compiler = world
    kernel = project.load()
    build(SPEC, arch="sm_90", fn=kernel)
    (path,) = entries()
    path.write_bytes(b"")

    def broken(*args, **kwargs):
        raise RuntimeError("the log is broken")

    monkeypatch.setattr(_cache._log, "warning", broken)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 2


# -- the key: every input of it is one, the recipe and the version too ---------------------------------------------------------------------------


def key_of(**changes):
    """``cache_key`` with every input given, and the ones in ``changes`` different."""
    inputs = dict(
        arch=90,
        ir_digest="s" * 64,
        triton_version="3.5.1",
        ptxas_version="12.9",
        options={"num_warps": 4},
        environment={"env": {"TRITON_X": "1"}, "knobs": {"language.default_fp_fusion": "True"}},
        files={"extern_libs.libdevice": "d" * 64},
    )
    inputs.update(changes)
    return _cache.cache_key(**inputs)


KEY_INPUTS = [
    ("the IR digest", dict(ir_digest="t" * 64)),
    ("the architecture", dict(arch=100)),
    ("the Triton version", dict(triton_version="3.6.0")),
    ("the ptxas version", dict(ptxas_version="13.0")),
    ("the compile options", dict(options={"num_warps": 8})),
    ("an option that is not set and one that is", dict(options={"num_warps": 4, "enable_fp_fusion": False})),
    ("an environment variable Triton reads", dict(environment={"env": {"TRITON_X": "2"}, "knobs": {"language.default_fp_fusion": "True"}})),
    ("an environment variable that is not there", dict(environment={"env": {}, "knobs": {"language.default_fp_fusion": "True"}})),
    ("a knob Triton reads", dict(environment={"env": {"TRITON_X": "1"}, "knobs": {"language.default_fp_fusion": "False"}})),
    ("the libdevice file", dict(environment={"env": {"TRITON_X": "1"}, "knobs": {"language.default_fp_fusion": "True"}, "libdevice": "f" * 64})),
    ("the content of a file an option names", dict(files={"extern_libs.libdevice": "e" * 64})),
    ("a file an option names that is not there", dict(files={})),
]


@pytest.mark.parametrize("label, changed", KEY_INPUTS, ids=[k[0] for k in KEY_INPUTS])
def test_every_input_of_the_key_is_part_of_it(label, changed):
    assert key_of() == key_of()  # and the same inputs are the same key
    assert key_of(**changed) != key_of()
    assert re.fullmatch(r"[0-9a-f]{64}", key_of(**changed))


def test_the_key_has_no_spec_in_it():
    """The IR digest stands for the kernel; what the spec adds (grid, preconditions, the stem) does not make another binary."""
    import inspect

    assert "spec" not in inspect.signature(_cache.cache_key).parameters


def test_the_key_names_its_inputs_so_that_two_of_them_cannot_be_swapped():
    assert key_of(arch=90, triton_version="3.5.1", ptxas_version="12.9") != key_of(arch=90, triton_version="12.9", ptxas_version="3.5.1")
    assert key_of(options={"a": 1}, environment={"b": "2"}) != key_of(options={"b": "2"}, environment={"a": 1})


def test_the_cache_recipe_and_the_version_of_tilefish_are_part_of_the_key(monkeypatch):
    base = key_of()
    monkeypatch.setattr(_cache, "CACHE_RECIPE", _cache.CACHE_RECIPE + 1)
    bumped = key_of()
    assert bumped != base  # a change of what goes into a binary: nothing made by the old recipe is served
    monkeypatch.setattr(tilefish, "__version__", "99.0.0")
    released = key_of()
    assert released != bumped and released != base  # a release, whether or not anyone remembered to bump the recipe
    monkeypatch.setattr(_cache, "CACHE_RECIPE", _cache.CACHE_RECIPE - 1)
    assert key_of() != base and key_of() != released  # each alone makes another key


def test_an_entry_of_another_recipe_or_version_is_not_found_and_is_not_a_warning(world, monkeypatch):
    project, compiler = world
    kernel = project.load()
    first = build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 1 and len(entries()) == 1
    context = quiet()  # not a corrupt entry: nobody looks for it under the new key
    try:
        monkeypatch.setattr(_cache, "CACHE_RECIPE", _cache.CACHE_RECIPE + 1)
        second = build(SPEC, arch="sm_90", fn=kernel)
        monkeypatch.setattr(tilefish, "__version__", "99.0.0")
        third = build(SPEC, arch="sm_90", fn=kernel)
    finally:
        context.__exit__(None, None, None)
    assert compiler.count == 3 and len(entries()) == 3  # each recipe and each version has its own entry
    assert first.sha256_code == second.sha256_code == third.sha256_code  # (the same IR: the same code)




# -- a lowered kernel: the lines a function was imported from are checked by build ------------------------------------------------------------


def exec_module(path, name):
    """Import ``path`` the way no loader does (``exec`` of its text)."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    linecache.clearcache()
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    sys.modules[name] = module
    return module


def test_build_refuses_a_generated_kernel_whose_file_changed_under_the_function_it_holds(world, tmp_path):
    """The IR digest is of what the process holds and says nothing about whether that is the source the spec claims: the spec's
    ``source_sha256`` is the new source's, the file has it, and the function the process holds is the old one."""
    _, compiler = world
    old, new = scale_kernel(2.0).lower(), scale_kernel(3.0).lower()
    (spec_new,) = new.specs()
    src = tmp_path / "gen"
    path = old.write(src)
    try:
        held = getattr(exec_module(path, "gen_held"), new.fn_name)  # the function of the old source
        new.write(src)  # the file is the source the spec is for now: its sha256 is the spec's, so only the lines tell
        linecache.clearcache()
        assert _source.file_sha256(str(path)) == spec_new.source_sha256
        with pytest.raises(TilefishBuildError, match=r"was imported from an older version of"):
            build(spec_new, arch="sm_90", fn=held)
        assert compiler.count == 0 and entries() == []
    finally:
        sys.modules.pop("gen_held", None)


def test_a_hand_written_kernel_whose_file_changed_under_the_function_it_holds_is_built_as_held(world):
    """No file is looked at for an extern: the function the process holds is what is compiled, and its IR is what is recorded."""
    project, compiler = world
    project.write("cil_kernel", "from fake_jit import jit\n\n\n@jit\ndef kernel(x_ptr, y_ptr, n_elem, BLOCK):\n    return (BLOCK * 2.0, n_elem)\n")
    try:
        kernel = exec_module(project.path("cil_kernel"), "cil_kernel").kernel
        _, spec = extern_of("cil_kernel:kernel")
        held = ir_of(kernel, spec)
        project.edit("cil_kernel", "BLOCK * 2.0", "BLOCK * 3.0")  # the process holds the 2.0, the file says 3.0
        linecache.clearcache()
        built = build(spec, arch="sm_90", fn=kernel)
        assert built.ir_digest == held and compiler.count == 1
        assert ir_of(exec_module(project.path("cil_kernel"), "cil_kernel").kernel, spec) != held  # a new process holds 3.0
    finally:
        sys.modules.pop("cil_kernel", None)


# -- files that Triton reads while it compiles: ir_override is refused, the others are keyed by their content ----------------------------------


def test_the_compile_option_ir_override_is_refused_like_the_kernel_override(world, tmp_path):
    """``ir_override`` is a file that Triton substitutes for the IR of a stage: the binary would not be of the IR that identifies it."""
    project, compiler = world
    kernel = project.load()
    replacement = tmp_path / "replacement.ptx"
    replacement.write_text(".version 8.8\n")
    for cache in (True, False):
        with pytest.raises(TilefishBuildError, match=r"the compile option ir_override .*names a file that Triton substitutes for the IR of a stage.*TRITON_KERNEL_OVERRIDE"):
            build(SPEC, arch="sm_90", fn=kernel, options={"ir_override": str(replacement)}, cache=cache)
    assert compiler.count == 0 and compiler.frontends == 0 and entries() == []  # refused before anything was built or cached
    build(SPEC, arch="sm_90", fn=kernel, options={"ir_override": None})  # an unset option is none


def test_compile_files_is_what_the_in_memory_key_asks_too(world, tmp_path):
    replacement = tmp_path / "replacement.ptx"
    replacement.write_text("one")
    with pytest.raises(TilefishBuildError, match="ir_override"):
        _cache.compile_files({"ir_override": str(replacement)})
    assert _cache.compile_files({"num_warps": 4}) == {}


def test_extern_libs_are_keyed_by_the_content_of_the_files_not_by_their_paths(world, tmp_path):
    """A library rewritten under the same name is another binary: its sha256 is in the key, and a miss recompiles."""
    project, compiler = world
    kernel = project.load()
    library = tmp_path / "libdevice.10.bc"
    library.write_bytes(b"one")
    options = {"extern_libs": {"libdevice": str(library)}}
    first = build(SPEC, arch="sm_90", fn=kernel, options=options)
    build(SPEC, arch="sm_90", fn=kernel, options=options)
    assert compiler.count == 1 and len(entries()) == 1  # a hit
    library.write_bytes(b"two")  # the same path, another file
    second = build(SPEC, arch="sm_90", fn=kernel, options=options)
    assert compiler.count == 2 and len(entries()) == 2 and second.cubin is not first.cubin
    library.write_bytes(b"one")  # put back: the first entry again
    build(SPEC, arch="sm_90", fn=kernel, options=options)
    assert compiler.count == 2
    assert _cache.compile_files(options) == {"extern_libs.libdevice": hashlib.sha256(b"one").hexdigest()}


def test_extern_libs_that_cannot_be_hashed_refuse_the_build(world, tmp_path):
    project, compiler = world
    kernel = project.load()
    with pytest.raises(TilefishBuildError, match=r"the extern library 'libdevice' .*cannot be read"):
        build(SPEC, arch="sm_90", fn=kernel, options={"extern_libs": {"libdevice": str(tmp_path / "missing.bc")}})
    with pytest.raises(TilefishBuildError, match="extern_libs is a mapping of library names to file paths"):
        build(SPEC, arch="sm_90", fn=kernel, options={"extern_libs": 42})
    assert compiler.count == 0 and entries() == []


def test_llvm_pass_plugin_path_is_keyed_by_the_content_of_the_plugin(world, tmp_path, monkeypatch):
    project, compiler = world
    kernel = project.load()
    plugin = tmp_path / "plugin.so"
    plugin.write_bytes(b"plugin one")
    monkeypatch.setenv("LLVM_PASS_PLUGIN_PATH", str(plugin))
    build(SPEC, arch="sm_90", fn=kernel)
    build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 1
    plugin.write_bytes(b"plugin two")  # rewritten under the same path
    build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 2 and len(entries()) == 2
    assert _cache.codegen_environment()["env_files"] == {"LLVM_PASS_PLUGIN_PATH": hashlib.sha256(b"plugin two").hexdigest()}
    plugin.unlink()
    with pytest.raises(TilefishBuildError, match=r"LLVM_PASS_PLUGIN_PATH .*cannot be read"):
        build(SPEC, arch="sm_90", fn=kernel)
    monkeypatch.delenv("LLVM_PASS_PLUGIN_PATH")
    assert "env_files" not in _cache.codegen_environment()
    build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 3  # without the plug-in: its own entry


def test_an_active_instrumentation_plugin_refuses_the_build(world, monkeypatch):
    """Proton's instrumentation backend patches the passes that lower the IR (``CUDABackend.instrumentation``)."""
    project, compiler = world
    kernel = project.load()
    backend = type("CUDABackend", (), {"instrumentation": None})
    monkeypatch.setitem(sys.modules, "triton.backends.nvidia", types.ModuleType("triton.backends.nvidia"))
    monkeypatch.setitem(sys.modules, "triton.backends.nvidia.compiler", types.SimpleNamespace(CUDABackend=backend))
    build(SPEC, arch="sm_90", fn=kernel)  # the backend is there and no instrumentation is on
    backend.instrumentation = object()
    with pytest.raises(TilefishBuildError, match=r"instrumentation plugin of Triton is active \(CUDABackend.instrumentation"):
        build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 1


def test_the_arch_override_knob_is_part_of_the_key(world, monkeypatch):
    project, compiler = world
    kernel = project.load()
    build(SPEC, arch="sm_90", fn=kernel)
    monkeypatch.setattr(sys.modules["triton"].knobs.runtime, "override_arch", "sm100")
    build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 2


# -- the frontend and the compile share one lock --------------------------------------------------------------------------------------------


def test_no_two_frontends_run_at_once_in_any_threads_not_even_one_inside_a_compile(world):
    """``triton.compile`` runs the frontend again, so the lock that serialises a compile is the lock of the frontend of a digest: with
    two locks a compile's own ``make_ir`` and another thread's digest ran together (and failed closed, by the digest check)."""
    project, compiler = world
    kernel = project.load()
    compiler.delay = 0.004
    errors = []

    def digest():
        try:
            for _ in range(6):
                ir_of(kernel)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def compile_():
        try:
            for i in range(4):
                build(SPEC, arch="sm_90", fn=kernel, options={"num_stages": 2 + i}, cache=False)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=digest) for _ in range(4)] + [threading.Thread(target=compile_) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:1]
    assert compiler.max_active == 1 and compiler.frontends > 20  # one at a time, however many threads asked
    assert tilefish._build._compile_lock is _ir.LOCK


# -- a Triton that is not the version that built the entry is not asked to run its frontend -------------------------------------------------------


def test_replay_compares_the_triton_versions_before_it_runs_the_frontend(world, monkeypatch):
    """A Triton whose frontend differs would raise from the frontend; the version says first that the digests cannot be compared."""
    project, compiler = world
    kernel = project.load()
    prep = types.SimpleNamespace(spec=SPEC, target=types.SimpleNamespace(jit_function=lambda spec: kernel))
    digest, version = runtime._current_ir(prep, 90, "3.5.1")
    assert version == "3.5.1" and digest == ir_of(kernel)
    monkeypatch.setattr(sys.modules["triton"], "__version__", "9.9.9")  # another Triton: its frontend is not run
    ran = []

    def frontend(*a, **k):
        ran.append(1)
        raise TilefishBuildError("the entry points of another Triton")

    monkeypatch.setattr(_ir, "ttir_text", frontend)
    assert runtime._current_ir(prep, 90, "3.5.1") == (None, "9.9.9") and ran == []
    assert runtime._current_ir(prep, 90, None) == (None, "9.9.9") and ran == []  # an entry that does not say which Triton made it
    monkeypatch.setitem(sys.modules, "triton", None)  # no Triton: nothing to recompute, nothing said of a version
    assert runtime._current_ir(prep, 90, "3.5.1") == (None, None)
