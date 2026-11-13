"""tilefish's own build cache (``tilefish._cache``): a hit after a rebuild, a miss after any edit, a corrupt entry never used.

No Triton and no GPU: ``fake_compile`` stands in for the compiler (it "compiles" what the process holds and counts the
calls), and the sources are real files in a temporary package, edited and reloaded for real. ``test_gpu_cache.py`` runs the
same through Triton on a GPU.
"""

import base64
import hashlib
import importlib
import json
import linecache
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import warnings

import fake_compile
import pytest
from built_fakes import fake_build, scale
from elf_builder import cubin_like

import tilefish
from tilefish import TilefishCacheWarning, TilefishManifestError, _cache, _source, runtime
from tilefish.aot import build, compile_specs
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
        "    return (ic_helper_a.apply(BLOCK) * ic_cfg.SCALE, n_elem)\n"
    ),
    "ic_cfg": "SCALE = 2.0\n",
    "ic_helper_a": "import ic_helper_b\nfrom fake_jit import jit\n\n\n@jit\ndef apply(x):\n    return x + ic_helper_b.OFFSET\n",
    "ic_helper_b": "OFFSET = 0.0\n",
}

n = tilefish.Dim("n", divisible_by=16)


@tilefish.extern(triton="ic_kernel:kernel", num_warps=4, grid=lambda n: (cdiv(n, 256),))
def op(x_ptr: tilefish.Tensor[n, tilefish.bf16], y_ptr: tilefish.Out[tilefish.Tensor[n, tilefish.bf16]], n_elem: n, *, BLOCK: tilefish.const = 256): ...


(SPEC,) = compile_specs(op)


class Project:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        for name, text in FILES.items():
            self.write(name, text)

    def path(self, name):
        return self.directory / f"{name}.py"

    def write(self, name, text):
        self.path(name).write_text(text)

    def edit(self, name, old, new):
        text = self.path(name).read_text()
        assert old in text, (name, old)
        self.write(name, text.replace(old, new, 1))

    def load(self):
        """Import the modules from their files as they are now (what a new process does), and return the kernel."""
        for name in MODULES:
            sys.modules.pop(name, None)
        importlib.invalidate_caches()
        linecache.clearcache()
        return importlib.import_module("ic_kernel").kernel


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


# -- a hit after a rebuild -------------------------------------------------------------------------------------------------


def test_the_second_build_of_unchanged_sources_is_read_from_the_cache_and_compiles_nothing(world):
    project, compiler = world
    kernel = project.load()
    first = build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 1 and len(entries()) == 1
    second = build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 1  # nothing was compiled: the binary is the cached one
    for field in ("cubin", "ptx", "symbol", "shared", "num_warps", "sha256", "sha256_code", "triton_version", "ptxas_version", "ptx_target", "options", "launch", "source_digest", "arch"):
        assert getattr(second, field) == getattr(first, field), field
    assert second.spec is SPEC and second.source_digest == _source.source_digest(kernel) and len(second.source_digest) == 64


def test_a_new_process_hits_too_and_so_does_the_same_project_somewhere_else(world, tmp_path, monkeypatch):
    project, compiler = world
    first = build(SPEC, arch="sm_90", fn=project.load())
    assert compiler.count == 1
    runtime.clear_caches()  # nothing in memory: a fresh import of the files, as a new process has
    again = build(SPEC, arch="sm_90", fn=project.load())
    assert compiler.count == 1 and again.sha256 == first.sha256
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(project.directory, elsewhere)
    monkeypatch.syspath_prepend(str(elsewhere))  # the digest is of the files, not of where they are
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


# -- a miss after any edit, including one two levels away ------------------------------------------------------------------------


EDITS = [
    ("ic_cfg", "SCALE = 2.0", "SCALE = 3.0"),  # read as ic_cfg.SCALE
    ("ic_helper_b", "OFFSET = 0.0", "OFFSET = 1.0"),  # two levels away from the kernel's module
    ("ic_helper_a", "x + ic_helper_b.OFFSET", "x - ic_helper_b.OFFSET"),
    ("ic_kernel", "return (ic_helper_a", "return (3 * ic_helper_a"),
]


@pytest.mark.parametrize("name, old, new", EDITS, ids=[e[0] for e in EDITS])
def test_a_source_edit_is_a_miss_and_the_edit_undone_is_a_hit_again(world, name, old, new):
    project, compiler = world
    before = build(SPEC, arch="sm_90", fn=project.load())
    project.edit(name, old, new)
    after = build(SPEC, arch="sm_90", fn=project.load())  # reloaded: the process holds the new file
    assert compiler.count == 2 and len(entries()) == 2
    assert after.sha256_code != before.sha256_code and after.source_digest != before.source_digest
    project.edit(name, new, old)
    undone = build(SPEC, arch="sm_90", fn=project.load())
    assert compiler.count == 2 and undone.sha256 == before.sha256  # served from the first entry
    assert len(entries()) == 2


def test_a_comment_is_an_edit_too_it_changes_the_source_digest_and_so_the_key(world):
    project, compiler = world
    build(SPEC, arch="sm_90", fn=project.load())
    project.edit("ic_helper_b", "OFFSET = 0.0", "OFFSET = 0.0  # nothing changes")
    build(SPEC, arch="sm_90", fn=project.load())
    assert compiler.count == 2


# -- an edit that was not reloaded never meets the cache ---------------------------------------------------------------------------


def test_a_module_edited_and_not_reloaded_is_refused_before_the_cache_is_read_or_written(world, monkeypatch):
    project, compiler = world
    kernel = project.load()
    stale = build(SPEC, arch="sm_90", fn=kernel)
    listed = [p.read_bytes() for p in entries()]
    touched = []
    with monkeypatch.context() as spy:
        spy.setattr(_cache.BuildCache, "load", lambda self, *a, **k: touched.append("load"))
        spy.setattr(_cache.BuildCache, "store", lambda self, *a, **k: touched.append("store"))
        spy.setattr(_cache, "cache_key", lambda *a, **k: touched.append("key"))
        project.edit("ic_helper_b", "OFFSET = 0.0", "OFFSET = 1.0")  # two levels away, edited, not reloaded
        with pytest.raises(tilefish.TilefishBuildError, match=r"module 'ic_helper_b' changed on disk since it was imported .*reload it or restart"):
            build(SPEC, arch="sm_90", fn=kernel)
    assert touched == [] and compiler.count == 1  # no key was computed, nothing was read, written or compiled
    assert [p.read_bytes() for p in entries()] == listed
    fresh = build(SPEC, arch="sm_90", fn=project.load())  # reloaded: the key is of the file the process holds, a miss
    assert compiler.count == 2 and fresh.sha256_code != stale.sha256_code and len(entries()) == 2


def test_a_constant_assigned_at_run_time_is_refused_not_cached(world):
    project, compiler = world
    kernel = project.load()
    sys.modules["ic_cfg"].SCALE = 5.0
    with pytest.raises(_source.StaleSourceError, match=r"'ic_cfg.SCALE'.*was assigned after 'ic_cfg' was imported"):
        build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 0 and entries() == []  # an entry made now would be served to a clean process under the digest of the file


def test_a_replay_recomputes_a_source_digest_only_for_a_process_that_holds_the_files(world):
    """``replay`` compares the manifest's digest with the files, so it asks for the same check as a build."""
    project, _ = world
    project.load()
    digest = runtime._current_source_digest(SPEC)
    assert digest is not None and len(digest) == 64
    project.edit("ic_helper_b", "OFFSET = 0.0", "OFFSET = 1.0")  # edited, not reloaded
    with pytest.raises(tilefish.TilefishRuntimeError, match=r"module 'ic_helper_b' changed on disk since it was imported .*reload it or restart"):
        runtime._current_source_digest(SPEC)
    project.load()
    assert runtime._current_source_digest(SPEC) not in (None, digest)


# -- the key --------------------------------------------------------------------------------------------------------------------------


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


def test_the_environment_variables_triton_invalidates_its_cache_with_are_part_of_the_key(world, monkeypatch):
    project, compiler = world
    kernel = project.load()
    build(SPEC, arch="sm_90", fn=kernel)
    monkeypatch.setattr(_cache, "triton_environment", lambda: {"TRITON_DEFAULT_FP_FUSION": "0"})
    build(SPEC, arch="sm_90", fn=kernel)
    assert compiler.count == 2


def test_a_value_a_module_computes_when_it_is_imported_is_part_of_the_key(world, monkeypatch):
    """The files are the same, the process holds another constant: the binary differs, so the digest, and with it the key, must."""
    project, compiler = world
    project.write("ic_cfg", 'import os\n\nSCALE = float(os.environ.get("IC_SCALE", "2.0"))\n')
    monkeypatch.setenv("IC_SCALE", "2.0")
    first = build(SPEC, arch="sm_90", fn=project.load())
    monkeypatch.setenv("IC_SCALE", "3.0")
    other = build(SPEC, arch="sm_90", fn=project.load())  # the same files, another value in the process
    assert compiler.count == 2 and other.source_digest != first.source_digest and other.sha256_code != first.sha256_code
    monkeypatch.setenv("IC_SCALE", "2.0")
    again = build(SPEC, arch="sm_90", fn=project.load())
    assert compiler.count == 2 and again.sha256 == first.sha256  # and each is a hit for its own value


def test_the_values_a_kernel_reads_are_the_scalars_and_constants_of_its_modules(world):
    project, _ = world
    project.write("ic_cfg", "SCALE = 2.0\nOTHER = 5\nTABLE = [1]\n")
    found = _source.reachable(project.load())
    assert _source.values_read(found) == {"ic_cfg.SCALE": "2.0", "ic_helper_b.OFFSET": "0.0"}  # read by a jit function, scalar; not OTHER, not a list
    assert _source.values_read(None) == {}


def test_another_declaration_or_spec_is_another_key(world):
    project, compiler = world
    kernel = project.load()
    build(SPEC, arch="sm_90", fn=kernel)

    @tilefish.extern(triton="ic_kernel:kernel", num_warps=2, grid=lambda n: (cdiv(n, 256),))
    def op(x_ptr: tilefish.Tensor[n, tilefish.bf16], y_ptr: tilefish.Out[tilefish.Tensor[n, tilefish.bf16]], n_elem: n, *, BLOCK: tilefish.const = 256): ...

    (other,) = compile_specs(op)
    assert other.declaration_digest != SPEC.declaration_digest
    build(other, arch="sm_90", fn=kernel)
    assert compiler.count == 2


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


# -- a generated kernel's key is its spec's source --------------------------------------------------------------------------------------


def scale_kernel(factor):
    """The same kernel (name and signature) with another body."""
    M, D = tilefish.Dim("M"), tilefish.Dim("D", 64)

    @tilefish.kernel(name="scale_k")
    def scale_k(x: tilefish.Tensor[M, D, tilefish.f32]) -> tilefish.Tensor[M, D, tilefish.f32]:
        return x * factor

    return scale_k


def test_a_generated_kernel_hits_and_another_body_is_a_miss(world):
    _, compiler = world
    old, new = scale_kernel(2.0).lower(), scale_kernel(3.0).lower()
    (spec_old,), (spec_new,) = old.specs(), new.specs()
    assert spec_old.source_sha256 != spec_new.source_sha256
    compiler.evaluate = False  # the source text is what is compiled
    a = build(spec_old, arch="sm_90", fn=runtime.load_lowered(old))
    assert a.source_digest is None and compiler.count == 1
    again = build(spec_old, arch="sm_90", fn=runtime.load_lowered(old))
    assert compiler.count == 1 and again.sha256 == a.sha256 and again.source_digest is None
    b = build(spec_new, arch="sm_90", fn=runtime.load_lowered(new))
    assert compiler.count == 2 and b.sha256_code != a.sha256_code


def test_a_stale_generated_file_is_refused_before_the_cache(world, tmp_path, monkeypatch):
    _, compiler = world
    old, new = scale_kernel(2.0).lower(), scale_kernel(3.0).lower()
    (spec_new,) = new.specs()
    src = tmp_path / "src"
    old.write(src)
    monkeypatch.syspath_prepend(str(src))
    sys.modules.pop(new.module, None)
    importlib.invalidate_caches()
    imported = importlib.import_module(new.module)  # the old source
    new.write(src)  # the file now has the spec's source; the process holds the old function
    linecache.clearcache()
    with pytest.raises(tilefish.TilefishBuildError, match=r"module '.*' changed on disk since it was imported"):
        build(spec_new, arch="sm_90", fn=getattr(imported, new.fn_name))
    assert compiler.count == 0 and entries() == []
    sys.modules.pop(new.module, None)


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
        source_digest=built.source_digest,
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
    (spec,) = compile_specs(scale)
    base = dict(arch=90, source=("source_digest", "a" * 64), triton_version="3.5.1", ptxas_version="12.9", options={"num_warps": 4}, environment={}, values={})
    key = _cache.cache_key(spec, **base)
    assert key == _cache.cache_key(spec, **base) and len(key) == 64
    for change in (
        {"arch": 100},
        {"source": ("source_digest", "b" * 64)},
        {"source": ("source_sha256", "a" * 64)},
        {"triton_version": "3.5.2"},
        {"ptxas_version": "12.10"},
        {"options": {"num_warps": 8}},
        {"options": {"num_warps": 4, "enable_fp_fusion": False}},
        {"environment": {"TRITON_X": "1"}},
        {"values": {"ic_cfg.SCALE": "2.0"}},
    ):
        assert _cache.cache_key(spec, **{**base, **change}) != key, change


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
args = dict(arch=me.arch, options=me.options, triton_version=me.triton_version, ptxas_version=me.ptxas_version, source_digest=me.source_digest)
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
    from tilefish.aot import write_manifest

    write_manifest([first], tmp_path / "a" / "manifest.json", cubins=True)
    second = build(SPEC, arch="sm_90", fn=kernel)
    write_manifest([second], tmp_path / "b" / "manifest.json", cubins=True)
    assert compiler.count == 1
    assert tilefish.Manifest.load(tmp_path / "a").digest == tilefish.Manifest.load(tmp_path / "b").digest


# -- a kernel made by a factory function reads its constants from closure cells ------------------------------------------------------


CLO_FILE = (
    "import os\n"
    "\n"
    "from fake_jit import jit\n"
    "\n"
    "\n"
    "def make():\n"
    '    S = float(os.environ["CLO_S"])\n'
    "\n"
    "    @jit\n"
    "    def k(x_ptr, y_ptr, n_elem, BLOCK):\n"
    "        return (BLOCK * S, n_elem)\n"
    "\n"
    "    return k\n"
    "\n"
    "\n"
    "k = make()\n"
)


@tilefish.extern(triton="clo_kernel:k", num_warps=4, grid=lambda n: (cdiv(n, 256),))
def clo_op(x_ptr: tilefish.Tensor[n, tilefish.bf16], y_ptr: tilefish.Out[tilefish.Tensor[n, tilefish.bf16]], n_elem: n, *, BLOCK: tilefish.const = 256): ...


(CLO_SPEC,) = compile_specs(clo_op)


@pytest.fixture
def closure(world, monkeypatch):
    """``load()`` imports ``clo_kernel`` from its file as a new process does (``CLO_S`` is what it reads), and returns its kernel."""
    project, compiler = world
    project.write("clo_kernel", CLO_FILE)
    monkeypatch.setenv("CLO_S", "2.0")

    def load():
        sys.modules.pop("clo_kernel", None)
        importlib.invalidate_caches()
        linecache.clearcache()
        return importlib.import_module("clo_kernel").k

    yield load, compiler, project
    sys.modules.pop("clo_kernel", None)


def test_a_closure_variable_is_part_of_the_key_so_another_value_is_another_binary_and_not_a_stale_hit(closure, monkeypatch):
    """The reviewer's repro: two processes share one cache directory, ``CLO_S=2.0`` and then ``CLO_S=3.0``. The files are the same."""
    load, compiler, project = closure
    first = build(CLO_SPEC, arch="sm_90", fn=load())
    assert compiler.count == 1 and len(entries()) == 1
    monkeypatch.setenv("CLO_S", "3.0")
    runtime.clear_caches()
    other = build(CLO_SPEC, arch="sm_90", fn=load())  # a new process: the same files, another value in the cell
    assert compiler.count == 2 and len(entries()) == 2  # a miss; it used to be a hit on the first one's binary
    assert other.sha256_code != first.sha256_code and other.source_digest != first.source_digest
    assert compiler.values_held(fake_compile.ASTSource(load(), {}))[1] == (3.0, 1)  # and the binary is what the process holds
    monkeypatch.setenv("CLO_S", "2.0")
    runtime.clear_caches()
    again = build(CLO_SPEC, arch="sm_90", fn=load())
    assert compiler.count == 2 and again.sha256 == first.sha256 and again.source_digest == first.source_digest  # each value has its own entry


def test_export_stamps_the_binary_of_the_value_the_process_holds_and_replay_refuses_another(closure, monkeypatch, tmp_path):
    load, compiler, project = closure
    load()
    manifest = runtime.export(clo_op, tmp_path / "two", arch=90)
    (entry,) = manifest.doc["kernels"]
    assert manifest.pinned  # export has written it from its own builds: what generates source from a manifest takes it
    assert entry["source_digest_kind"] == _source.SOURCE_DIGEST_KIND and len(entry["source_digest"]) == 64
    # a replay in a process that holds the same value: the digest is the entry's, and the binary is accepted
    digest_two = runtime._current_source_digest(CLO_SPEC)
    assert digest_two == entry["source_digest"]
    assert manifest.built_kernel(CLO_SPEC, "sm_90", source_digest=digest_two).sha256_code == entry["sha256_code"]
    # a new process whose cell has another value would compile another binary: replay refuses, pinned or not
    monkeypatch.setenv("CLO_S", "3.0")
    load()
    digest_three = runtime._current_source_digest(CLO_SPEC)
    assert digest_three is not None and digest_three != digest_two
    for pinned in (False, True):
        with pytest.raises(TilefishManifestError, match=rf"now have source digest {digest_three}.*a closure variable"):
            manifest.built_kernel(CLO_SPEC, "sm_90", source_digest=digest_three, pinned=pinned)
    # ... and exporting there stamps the binary of 3.0, not the stale one of 2.0 that the cache holds
    count = compiler.count
    again = runtime.export(clo_op, tmp_path / "three", arch=90)
    (entry_three,) = again.doc["kernels"]
    assert compiler.count == count + 1
    assert entry_three["sha256_code"] != entry["sha256_code"] and entry_three["source_digest"] == digest_three
    assert again.built_kernel(CLO_SPEC, "sm_90", source_digest=digest_three).sha256_code == entry_three["sha256_code"]


def test_a_closure_variable_that_has_no_stable_repr_refuses_the_build_and_the_export_and_the_replay(closure, tmp_path):
    load, compiler, project = closure
    project.write("clo_kernel", CLO_FILE.replace('S = float(os.environ["CLO_S"])', 'S = [float(os.environ["CLO_S"])]').replace("BLOCK * S", "BLOCK * S[0]"))
    kernel = load()
    for cache in (True, False):
        with pytest.raises(_source.UnstableValueError, match=r"closure variable 'S' of the Triton function 'k' holds a list.*module-level constant.*const parameter"):
            build(CLO_SPEC, arch="sm_90", fn=kernel, cache=cache)
    with pytest.raises(_source.UnstableValueError, match="closure variable 'S'"):
        runtime.export(clo_op, tmp_path / "m", arch=90)
    assert compiler.count == 0 and entries() == [] and not (tmp_path / "m" / "manifest.json").exists()  # nothing was compiled, cached or written
    with pytest.raises(tilefish.TilefishRuntimeError, match="closure variable 'S'"):
        runtime._current_source_digest(CLO_SPEC)
